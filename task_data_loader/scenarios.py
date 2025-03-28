import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, List, Dict, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets import ImageNet
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from task_data_loader.imagenet import train_transform, valid_transform
from task_data_loader.split_cifar10 import TaskSpecificSplitCIFAR10, cifar10_basic_transform
from task_data_loader.split_cifar100 import TaskSpecificSplitCIFAR100, cifar100_basic_transform
from task_data_loader.split_imagenet32 import TaskSpecificSplitImageNet32
from task_data_loader.split_imagenet32_100 import TaskSpecificSplitImageNet32_100
# from task_data_loader.product import ProductDataset
from task_data_loader.symbol import SymbolDataset
from task_data_loader.symbol_complex import ComplexSymbolDataset
from .cub import Cub2011
from .flowers import Flowers102
from .scenes import Scenes

from glob import glob
from collections import defaultdict
import logging
from tqdm import tqdm
from itertools import product

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler(f'./logging/{os.getenv("LOG_RUN_NAME")}_{__name__}.log', 'w'))


@dataclass
class TaskConfig:
    train: Dataset
    test: Dataset
    id: str
    nb_classes: int


class Scenario(ABC):
    def __init__(self, root: str, transforms: Optional[Union[List[Callable], Callable]]):
        self.root = root
        self.transforms = transforms

    @property
    @abstractmethod
    def tasks(self) -> List[TaskConfig]:
        pass


class DuplicatedHalfCIFAR10(Scenario):
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str = "data",
        task1_classes: Tuple[int, ...] = (0, 1, 2, 3, 4),
        transforms: Optional[Callable] = cifar10_basic_transform,
    ):
        super().__init__(root, transforms)
        self.task1_classes = task1_classes
        self.download()

        train_dataset = self._get_data(downloaded_list=self.train_list)
        test_dataset = self._get_data(downloaded_list=self.test_list)

        split_train_tasks = self._split_dataset_into_two_tasks(
            features=train_dataset["features"], targets=train_dataset["targets"], task1_classes=self.task1_classes
        )
        split_test_tasks = self._split_dataset_into_two_tasks(
            features=test_dataset["features"], targets=test_dataset["targets"], task1_classes=self.task1_classes
        )

        self.task1 = TaskConfig(
            train=TaskSpecificSplitCIFAR10(
                features=split_train_tasks["task1"]["features"],
                targets=split_train_tasks["task1"]["targets"],
                task_id=1,
                transform=self.transforms,
            ),
            test=TaskSpecificSplitCIFAR10(
                features=split_test_tasks["task1"]["features"],
                targets=split_test_tasks["task1"]["targets"],
                task_id=1,
                transform=self.transforms,
            ),
            id="1",
            nb_classes=5,
        )

        self.task2 = TaskConfig(
            train=TaskSpecificSplitCIFAR10(
                features=split_train_tasks["task1"]["features"],
                targets=split_train_tasks["task1"]["targets"],
                task_id=1,
                transform=self.transforms,
            ),
            test=TaskSpecificSplitCIFAR10(
                features=split_test_tasks["task1"]["features"],
                targets=split_test_tasks["task1"]["targets"],
                task_id=1,
                transform=self.transforms,
            ),
            id="2",
            nb_classes=5,
        )

    @property
    def tasks(self) -> List[TaskConfig]:
        return [self.task1, self.task2]

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def _get_data(self, downloaded_list: List[List[str]]) -> Dict[str, Union[List, np.ndarray]]:
        features = []
        targets = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                features.append(entry["data"])
                if "labels" in entry:
                    targets.extend(entry["labels"])
                else:
                    targets.extend(entry["fine_labels"])

        features = np.vstack(features).reshape(-1, 3, 32, 32)
        features = features.transpose((0, 2, 3, 1))  # convert to HWC
        return {"features": features, "targets": targets}

    def _split_dataset_into_two_tasks(self, features: np.ndarray, targets: List[int], task1_classes: Tuple[int, ...]):
        task1_features = []
        task1_targets = []
        task2_features = []
        task2_targets = []

        for feature, target in zip(features, targets):
            if target in task1_classes:
                task1_features.append(feature)
                task1_targets.append(target)
            else:
                task2_features.append(feature)
                task2_targets.append(target)

        task1_features = np.asarray(task1_features)
        task2_features = np.asarray(task2_features)

        return {
            "task1": {"features": task1_features, "targets": task1_targets},
            "task2": {"features": task2_features, "targets": task2_targets},
        }


class SplitCIFAR10(Scenario):
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str = "data",
        task1_classes: Tuple[int, ...] = (0, 1, 2, 3, 4),
        transforms: Optional[Callable] = cifar10_basic_transform,
    ):
        super().__init__(root, transforms)
        self.task1_classes = task1_classes
        self.download()

        train_dataset = self._get_data(downloaded_list=self.train_list)
        test_dataset = self._get_data(downloaded_list=self.test_list)

        split_train_tasks = self._split_dataset_into_two_tasks(
            features=train_dataset["features"], targets=train_dataset["targets"], task1_classes=self.task1_classes
        )
        split_test_tasks = self._split_dataset_into_two_tasks(
            features=test_dataset["features"], targets=test_dataset["targets"], task1_classes=self.task1_classes
        )

        self.task1 = TaskConfig(
            train=TaskSpecificSplitCIFAR10(
                features=split_train_tasks["task1"]["features"],
                targets=split_train_tasks["task1"]["targets"],
                task_id=1,
                transform=self.transforms,
            ),
            test=TaskSpecificSplitCIFAR10(
                features=split_test_tasks["task1"]["features"],
                targets=split_test_tasks["task1"]["targets"],
                task_id=1,
                transform=self.transforms,
            ),
            id="1",
            nb_classes=5,
        )

        self.task2 = TaskConfig(
            train=TaskSpecificSplitCIFAR10(
                features=split_train_tasks["task2"]["features"],
                targets=split_train_tasks["task2"]["targets"],
                task_id=2,
                transform=self.transforms,
            ),
            test=TaskSpecificSplitCIFAR10(
                features=split_test_tasks["task2"]["features"],
                targets=split_test_tasks["task2"]["targets"],
                task_id=2,
                transform=self.transforms,
            ),
            id="2",
            nb_classes=5,
        )

    @property
    def tasks(self) -> List[TaskConfig]:
        return [self.task1, self.task2]

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def _get_data(self, downloaded_list: List[List[str]]) -> Dict[str, Union[List, np.ndarray]]:
        features = []
        targets = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                features.append(entry["data"])
                if "labels" in entry:
                    targets.extend(entry["labels"])
                else:
                    targets.extend(entry["fine_labels"])

        features = np.vstack(features).reshape(-1, 3, 32, 32)
        features = features.transpose((0, 2, 3, 1))  # convert to HWC
        return {"features": features, "targets": targets}

    def _split_dataset_into_two_tasks(self, features: np.ndarray, targets: List[int], task1_classes: Tuple[int, ...]):
        task1_features = []
        task1_targets = []
        task2_features = []
        task2_targets = []

        for feature, target in zip(features, targets):
            if target in task1_classes:
                task1_features.append(feature)
                task1_targets.append(target)
            else:
                task2_features.append(feature)
                task2_targets.append(target)

        task1_features = np.asarray(task1_features)
        task2_features = np.asarray(task2_features)

        return {
            "task1": {"features": task1_features, "targets": task1_targets},
            "task2": {"features": task2_features, "targets": task2_targets},
        }

class SplitCIFAR100(Scenario):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }
    random_target_ids =list(range(100))
    np.random.shuffle(random_target_ids)
    shuffled_label_map = {
        idx : random_idx
        for idx, random_idx in enumerate(random_target_ids)
    }
    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable] = cifar100_basic_transform,
    ):
        super().__init__(root, transforms)
        self.download()

        train_dataset = self._get_data(downloaded_list=self.train_list)
        test_dataset = self._get_data(downloaded_list=self.test_list)

        split_train_tasks = self._split_dataset(features=train_dataset["features"], targets=train_dataset["targets"])
        split_test_tasks = self._split_dataset(features=test_dataset["features"], targets=test_dataset["targets"])
        logger.debug(f'split_train_tasks : {len(split_train_tasks.keys())}')
        self._tasks = []
        for task_id in range(20):
            task = TaskConfig(
                train=TaskSpecificSplitCIFAR100(
                    features=split_train_tasks[task_id]["features"],
                    targets=split_train_tasks[task_id]["targets"],
                    task_id=task_id,
                    transform=self.transforms,
                ),
                test=TaskSpecificSplitCIFAR100(
                    features=split_test_tasks[task_id]["features"],
                    targets=split_test_tasks[task_id]["targets"],
                    task_id=task_id,
                    transform=self.transforms,
                ),
                id=str(task_id),
                nb_classes=5,
            )
            self._tasks.append(task)
            logger.debug(f'task_id : {task_id}')
            logger.debug(f'train features : {len(task.train.features)}')
            logger.debug(f'train targets : {len(task.train.targets)}')
        
            logger.debug(f'test features : {len(task.test.features)}')
            logger.debug(f'test targets : {len(task.test.targets)}')
    
    def choose_task(self, target_task: int):
        self._tasks = [self._tasks[target_task]]

    @property
    def tasks(self) -> List[TaskConfig]:
        return self._tasks

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def _get_data(self, downloaded_list: List[List[str]]) -> Dict[str, Union[List, np.ndarray]]:
        features = []
        targets = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                features.append(entry["data"])
                if "labels" in entry:
                    raise ValueError()
                    targets.extend(entry["labels"])
                else:
                    targets.extend(list(map(lambda x: self.shuffled_label_map[x], entry["fine_labels"])))

        features = np.vstack(features).reshape(-1, 3, 32, 32)
        features = features.transpose((0, 2, 3, 1))  # convert to HWC
        return {"features": features, "targets": targets}

    def _split_dataset(self, features: np.ndarray, targets: List[int]): 
        logger.debug(f'_split_dataset')   
        logger.debug(f'features : {features.shape}')   
        logger.debug(f'targets : {len(targets), targets[:30]}')
        task_splits = defaultdict(lambda: defaultdict(list))

        task_id = 0
        task_container = []
        for feature, target in zip(features, targets):
            # target_idx = targets.index(target)
            task_id = target // 5
            # if len(task_container) == 5:
            #     task_container = []
            #     task_id += 1

            task_splits[task_id]['features'].append(feature)
            task_splits[task_id]['targets'].append(target)
            task_container.append(target)            

        final_task_splits = dict()
        for task_id, _data in task_splits.items():
            final_task_splits[task_id]= {
                'features' : np.asarray(_data['features']),
                'targets' : _data['targets'],
            }
        for task_id, _data in final_task_splits.items():
            logger.debug(f'Task {task_id}\n\t features : {_data["features"].shape}\n\t targets : {len(_data["targets"])}')
        return final_task_splits


class SplitCIFAR100FixedTask50(Scenario):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable] = cifar100_basic_transform,
    ):
        super().__init__(root, transforms)
        self.download()

        train_dataset = self._get_data(downloaded_list=self.train_list)
        test_dataset = self._get_data(downloaded_list=self.test_list)

        split_train_tasks = self._split_dataset(features=train_dataset["features"], targets=train_dataset["targets"])
        split_test_tasks = self._split_dataset(features=test_dataset["features"], targets=test_dataset["targets"])
        logger.debug(f'split_train_tasks : {len(split_train_tasks.keys())}')
        self._tasks = []
        for task_id in range(50):
            task = TaskConfig(
                train=TaskSpecificSplitCIFAR100(
                    features=split_train_tasks[task_id]["features"],
                    targets=split_train_tasks[task_id]["targets"],
                    task_id=f'Task_{task_id}',
                    transform=self.transforms,
                ),
                test=TaskSpecificSplitCIFAR100(
                    features=split_test_tasks[task_id]["features"],
                    targets=split_test_tasks[task_id]["targets"],
                    task_id=f'Task_{task_id}',
                    transform=self.transforms,
                ),
                id=f'Task_{task_id}',
                nb_classes=2,
            )
            self._tasks.append(task)
            logger.debug(f'task_id : {task_id}')
            logger.debug(f'train features : {len(task.train.features)}')
            logger.debug(f'train targets : {len(task.train.targets)}')
        
            logger.debug(f'test features : {len(task.test.features)}')
            logger.debug(f'test targets : {len(task.test.targets)}')
    
    def choose_task(self, target_task: int):
        self._tasks = [self._tasks[target_task]]

    @property
    def tasks(self) -> List[TaskConfig]:
        return self._tasks

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def _get_data(self, downloaded_list: List[List[str]]) -> Dict[str, Union[List, np.ndarray]]:
        features = []
        targets = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                features.append(entry["data"])
                if "labels" in entry:
                    raise ValueError()
                    targets.extend(entry["labels"])
                else:
                    targets.extend(entry["fine_labels"])

        features = np.vstack(features).reshape(-1, 3, 32, 32)
        features = features.transpose((0, 2, 3, 1))  # convert to HWC
        return {"features": features, "targets": targets}

    def _split_dataset(self, features: np.ndarray, targets: List[int]): 
        logger.debug(f'_split_dataset')   
        logger.debug(f'features : {features.shape}')   
        logger.debug(f'targets : {len(targets), targets[:30]}')
        task_splits = defaultdict(lambda: defaultdict(list))

        task_id = 0
        task_container = []
        for feature, target in zip(features, targets):
            task_id = target // 2

            task_splits[task_id]['features'].append(feature)
            task_splits[task_id]['targets'].append(target)
            task_container.append(target)            

        final_task_splits = dict()
        for task_id, _data in task_splits.items():
            final_task_splits[task_id]= {
                'features' : np.asarray(_data['features']),
                'targets' : _data['targets'],
            }
        for task_id, _data in final_task_splits.items():
            logger.debug(f'Task {task_id}\n\t features : {_data["features"].shape}\n\t targets : {len(_data["targets"])}')
        return final_task_splits


class SplitCIFAR100Task50(Scenario):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }
    random_target_ids =list(range(100))
    np.random.shuffle(random_target_ids)
    shuffled_label_map = {
        idx : random_idx
        for idx, random_idx in enumerate(random_target_ids)
    }
    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable] = cifar100_basic_transform,
    ):
        super().__init__(root, transforms)
        self.download()

        train_dataset = self._get_data(downloaded_list=self.train_list)
        test_dataset = self._get_data(downloaded_list=self.test_list)

        split_train_tasks = self._split_dataset(features=train_dataset["features"], targets=train_dataset["targets"])
        split_test_tasks = self._split_dataset(features=test_dataset["features"], targets=test_dataset["targets"])
        logger.debug(f'split_train_tasks : {len(split_train_tasks.keys())}')
        self._tasks = []
        for task_id in range(50):
            task = TaskConfig(
                train=TaskSpecificSplitCIFAR100(
                    features=split_train_tasks[task_id]["features"],
                    targets=split_train_tasks[task_id]["targets"],
                    task_id=task_id,
                    transform=self.transforms,
                ),
                test=TaskSpecificSplitCIFAR100(
                    features=split_test_tasks[task_id]["features"],
                    targets=split_test_tasks[task_id]["targets"],
                    task_id=task_id,
                    transform=self.transforms,
                ),
                id=str(task_id),
                nb_classes=2,
            )
            self._tasks.append(task)
            logger.debug(f'task_id : {task_id}')
            logger.debug(f'train features : {len(task.train.features)}')
            logger.debug(f'train targets : {len(task.train.targets)}')
        
            logger.debug(f'test features : {len(task.test.features)}')
            logger.debug(f'test targets : {len(task.test.targets)}')
    
    def choose_task(self, target_task: int):
        self._tasks = [self._tasks[target_task]]

    @property
    def tasks(self) -> List[TaskConfig]:
        return self._tasks

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def _get_data(self, downloaded_list: List[List[str]]) -> Dict[str, Union[List, np.ndarray]]:
        features = []
        targets = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                features.append(entry["data"])
                if "labels" in entry:
                    raise ValueError()
                    targets.extend(entry["labels"])
                else:
                    targets.extend(list(map(lambda x: self.shuffled_label_map[x], entry["fine_labels"])))

        features = np.vstack(features).reshape(-1, 3, 32, 32)
        features = features.transpose((0, 2, 3, 1))  # convert to HWC
        return {"features": features, "targets": targets}

    def _split_dataset(self, features: np.ndarray, targets: List[int]): 
        logger.debug(f'_split_dataset')   
        logger.debug(f'features : {features.shape}')   
        logger.debug(f'targets : {len(targets), targets[:30]}')
        task_splits = defaultdict(lambda: defaultdict(list))

        task_id = 0
        task_container = []
        for feature, target in zip(features, targets):
            # target_idx = targets.index(target)
            task_id = target // 2
            # if len(task_container) == 5:
            #     task_container = []
            #     task_id += 1

            task_splits[task_id]['features'].append(feature)
            task_splits[task_id]['targets'].append(target)
            task_container.append(target)            

        final_task_splits = dict()
        for task_id, _data in task_splits.items():
            final_task_splits[task_id]= {
                'features' : np.asarray(_data['features']),
                'targets' : _data['targets'],
            }
        for task_id, _data in final_task_splits.items():
            logger.debug(f'Task {task_id}\n\t features : {_data["features"].shape}\n\t targets : {len(_data["targets"])}')
        return final_task_splits


class SplitCIFAR100Task200(Scenario):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }
    random_target_ids =list(range(100))
    np.random.shuffle(random_target_ids)
    shuffled_label_map = {
        idx : random_idx
        for idx, random_idx in enumerate(random_target_ids)
    }
    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable] = cifar100_basic_transform,
    ):
        super().__init__(root, transforms)
        self.download()

        train_dataset = self._get_data(downloaded_list=self.train_list)
        test_dataset = self._get_data(downloaded_list=self.test_list)

        split_train_tasks = self._split_dataset(features=train_dataset["features"], targets=train_dataset["targets"])
        split_test_tasks = self._split_dataset(features=test_dataset["features"], targets=test_dataset["targets"])
        logger.debug(f'split_train_tasks : {len(split_train_tasks.keys())}')

        self._tasks = []
        for task_id in range(50):
            task = TaskConfig(
                train=TaskSpecificSplitCIFAR100(
                    features=split_train_tasks[task_id]["features"],
                    targets=split_train_tasks[task_id]["targets"],
                    task_id=task_id,
                    transform=self.transforms,
                ),
                test=TaskSpecificSplitCIFAR100(
                    features=split_test_tasks[task_id]["features"],
                    targets=split_test_tasks[task_id]["targets"],
                    task_id=task_id,
                    transform=self.transforms,
                ),
                id=str(task_id),
                nb_classes=2,
            )
            self._tasks.append(task)
            logger.debug(f'task_id : {task_id}')
            logger.debug(f'train features : {len(task.train.features)}')
            logger.debug(f'train targets : {len(task.train.targets)}')
        
            logger.debug(f'test features : {len(task.test.features)}')
            logger.debug(f'test targets : {len(task.test.targets)}')
        
        first_task_id = 0
        del split_train_tasks[first_task_id]
        del split_test_tasks[first_task_id]
        
        for continued_idx, selected_task_id in enumerate(np.random.choice(list(split_train_tasks.keys()), size=150, replace=True)):
            task = TaskConfig(
                train=TaskSpecificSplitCIFAR100(
                    features=split_train_tasks[selected_task_id]["features"],
                    targets=split_train_tasks[selected_task_id]["targets"],
                    task_id=continued_idx+50,
                    transform=self.transforms,
                ),
                test=TaskSpecificSplitCIFAR100(
                    features=split_test_tasks[selected_task_id]["features"],
                    targets=split_test_tasks[selected_task_id]["targets"],
                    task_id=continued_idx+50,
                    transform=self.transforms,
                ),
                id=str(continued_idx+50),
                nb_classes=2,
            )
            self._tasks.append(task)
            logger.debug(f'task_id : {selected_task_id}')
            logger.debug(f'train features : {len(task.train.features)}')
            logger.debug(f'train targets : {len(task.train.targets)}')
        
            logger.debug(f'test features : {len(task.test.features)}')
            logger.debug(f'test targets : {len(task.test.targets)}')
                
    def choose_task(self, target_task: int):
        self._tasks = [self._tasks[target_task]]

    @property
    def tasks(self) -> List[TaskConfig]:
        return self._tasks

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def _get_data(self, downloaded_list: List[List[str]]) -> Dict[str, Union[List, np.ndarray]]:
        features = []
        targets = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                features.append(entry["data"])
                if "labels" in entry:
                    raise ValueError()
                    targets.extend(entry["labels"])
                else:
                    targets.extend(list(map(lambda x: self.shuffled_label_map[x], entry["fine_labels"])))

        features = np.vstack(features).reshape(-1, 3, 32, 32)
        features = features.transpose((0, 2, 3, 1))  # convert to HWC
        return {"features": features, "targets": targets}

    def _split_dataset(self, features: np.ndarray, targets: List[int]): 
        logger.debug(f'_split_dataset')   
        logger.debug(f'features : {features.shape}')   
        logger.debug(f'targets : {len(targets), targets[:30]}')
        task_splits = defaultdict(lambda: defaultdict(list))

        task_id = 0
        task_container = []
        for feature, target in zip(features, targets):
            # target_idx = targets.index(target)
            task_id = target // 2
            # if len(task_container) == 5:
            #     task_container = []
            #     task_id += 1

            task_splits[task_id]['features'].append(feature)
            task_splits[task_id]['targets'].append(target)
            task_container.append(target)            

        final_task_splits = dict()
        for task_id, _data in task_splits.items():
            final_task_splits[task_id]= {
                'features' : np.asarray(_data['features']),
                'targets' : _data['targets'],
            }
        for task_id, _data in final_task_splits.items():
            logger.debug(f'Task {task_id}\n\t features : {_data["features"].shape}\n\t targets : {len(_data["targets"])}')
        return final_task_splits


class SymbolCount(Scenario):
    def __init__(
        self,
        root: str = "data",
        train_dataset_size=2000,
        test_dataset_size=100,
        seq_len:int = 5,
        transforms: Optional[Callable] = cifar100_basic_transform,
    ):
        super().__init__(root, transforms)
        self.symbol_set = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')']
        self.symbol_list = list(product(self.symbol_set, self.symbol_set, self.symbol_set))
        self.symbol_list = list(map(lambda x: ''.join(x), self.symbol_list))
        logger.info(f'Symbol list : {self.symbol_list}')
        np.random.shuffle(self.symbol_list)
        
        self._tasks = []
        for task_id in range(200):
            task = TaskConfig(
                train=SymbolDataset(
                    target_symbol=self.symbol_list[task_id],
                    dataset_size=train_dataset_size,
                    seq_len=seq_len,#10,
                    task_id=task_id,                    
                ),
                test=SymbolDataset(
                    target_symbol=self.symbol_list[task_id],
                    dataset_size=test_dataset_size,
                    seq_len=seq_len,#10,
                    task_id=task_id,                    
                ),
                id=str(task_id),
                nb_classes=seq_len,#10,
            )
            self._tasks.append(task)
            logger.debug(f'task_id : {task_id}')
            logger.debug(f'task input size : {task.train.inputs.size()}')
            logger.debug(f'task target size : {len(task.train.targets)}')
        
    
    def choose_task(self, target_task: int):
        self._tasks = [self._tasks[target_task]]

    @property
    def tasks(self) -> List[TaskConfig]:
        return self._tasks


class ComplexSymbolCount(Scenario):
    def __init__(
        self,
        root: str = "data",
        train_dataset_size=2000,
        test_dataset_size=100,
        seq_len:int = 1000,
        transforms: Optional[Callable] = cifar100_basic_transform,
    ):
        super().__init__(root, transforms)
        self.symbol_set = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')'] # in total : 10 basic symbols
        self.symbol_list = list(product(self.symbol_set, self.symbol_set, self.symbol_set)) # compose into symbol seq with length 3 => 1000 different symbols
        self.symbol_list = list(map(lambda x: ''.join(x), self.symbol_list))
        logger.info(f'Symbol list : {self.symbol_list}')
        np.random.shuffle(self.symbol_list)
        
        assert len(self.symbol_list) == 1000
        self._tasks = []
        for task_id in tqdm(range(200), desc='Generating tasks'):
            input_integers = list(range(task_id*5, (task_id+1)*5))
            target_symbols = self.symbol_list[task_id*5 :(task_id+1)*5]
            target_classes = list(range(task_id*5, (task_id+1)*5))
            
            task = TaskConfig(
                train=ComplexSymbolDataset(
                    input_integers=input_integers,
                    target_symbols=target_symbols,
                    target_classes=target_classes,
                    dataset_size=train_dataset_size,
                    seq_len=seq_len,
                    task_id=task_id,                    
                ),
                test=ComplexSymbolDataset(
                    input_integers=input_integers,
                    target_symbols=target_symbols,
                    target_classes=target_classes,
                    dataset_size=test_dataset_size,
                    seq_len=seq_len,
                    task_id=task_id,                    
                ),
                id=str(task_id),
                nb_classes=5,
            )
            self._tasks.append(task)
            logger.debug(f'task_id : {task_id}')
            logger.debug(f'task input size : {task.train.inputs.size()}')
            logger.debug(f'task target size : {len(task.train.targets)}')
        
    
    def choose_task(self, target_task: int):
        self._tasks = [self._tasks[target_task]]

    @property
    def tasks(self) -> List[TaskConfig]:
        return self._tasks


class CIFAR100(Scenario):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }
    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable] = cifar100_basic_transform,
    ):
        super().__init__(root, transforms)
        self.download()

        train_dataset = self._get_data(downloaded_list=self.train_list)
        test_dataset = self._get_data(downloaded_list=self.test_list)

        split_train_tasks = self._split_dataset(features=train_dataset["features"], targets=train_dataset["targets"])
        split_test_tasks = self._split_dataset(features=test_dataset["features"], targets=test_dataset["targets"])
        logger.debug(f'split_train_tasks : {len(split_train_tasks.keys())}')
        self._tasks = []
        task = TaskConfig(
            train=TaskSpecificSplitCIFAR100(
                features=train_dataset["features"],
                targets=train_dataset["targets"],
                task_id=0,
                transform=self.transforms,
            ),
            test=TaskSpecificSplitCIFAR100(
                features=test_dataset["features"],
                targets=test_dataset["targets"],
                task_id=0,
                transform=self.transforms,
            ),
            id=str(0),
            nb_classes=100,
        )
        self._tasks.append(task)
        logger.debug(f'train features : {len(task.train.features)}')
        logger.debug(f'train targets : {len(task.train.targets)}')
    
        logger.debug(f'test features : {len(task.test.features)}')
        logger.debug(f'test targets : {len(task.test.targets)}')
    
    def choose_task(self, target_task: int):
        self._tasks = [self._tasks[target_task]]

    @property
    def tasks(self) -> List[TaskConfig]:
        return self._tasks

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def _get_data(self, downloaded_list: List[List[str]]) -> Dict[str, Union[List, np.ndarray]]:
        features = []
        targets = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                features.append(entry["data"])
                if "labels" in entry:
                    targets.extend(entry["labels"])
                else:
                    targets.extend(entry["fine_labels"])

        features = np.vstack(features).reshape(-1, 3, 32, 32)
        features = features.transpose((0, 2, 3, 1))  # convert to HWC
        return {"features": features, "targets": targets}

    def _split_dataset(self, features: np.ndarray, targets: List[int]): 
        logger.debug(f'_split_dataset')   
        logger.debug(f'features : {features.shape}')   
        logger.debug(f'targets : {len(targets), targets[:30]}')
        task_splits = defaultdict(lambda: defaultdict(list))

        task_id = 0
        task_container = []
        for feature, target in zip(features, targets):
            # target_idx = targets.index(target)
            task_id = target // 5
            # if len(task_container) == 5:
            #     task_container = []
            #     task_id += 1

            task_splits[task_id]['features'].append(feature)
            task_splits[task_id]['targets'].append(target)
            task_container.append(target)            

        final_task_splits = dict()
        for task_id, _data in task_splits.items():
            final_task_splits[task_id]= {
                'features' : np.asarray(_data['features']),
                'targets' : _data['targets'],
            }
        for task_id, _data in final_task_splits.items():
            logger.debug(f'Task {task_id}\n\t features : {_data["features"].shape}\n\t targets : {len(_data["targets"])}')
        return final_task_splits


class ImageNet100Resized32Task20(Scenario):
    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable] = [train_transform, valid_transform],
    ):
        super().__init__(root, transforms)

        # NOTE : fix bugs for the wrong label index of train/val dataset
        self.label_list = [label.split('/')[-1] for label in glob(f'{self.root}/train/*')]
        self.label_list = np.random.choice(self.label_list, size=100, replace=False).tolist()

        split_train_tasks = self.get_data('train')
        split_test_tasks = self.get_data('val')

        logger.debug(f'split_train_tasks task num : {len(split_train_tasks.keys())}')
        logger.debug(f'split_test_tasks task num : {len(split_test_tasks.keys())}')

        self._tasks = []
        for task_id in range(20):
            task = TaskConfig(
                train=TaskSpecificSplitImageNet32(
                    features=split_train_tasks[task_id]["features"],
                    targets=split_train_tasks[task_id]["targets"],
                    task_id=f'Task_{task_id}',
                    transform=self.transforms[0],
                ),
                test=TaskSpecificSplitImageNet32(
                    features=split_test_tasks[task_id]["features"],
                    targets=split_test_tasks[task_id]["targets"],
                    task_id=f'Task_{task_id}',
                    transform=self.transforms[1],
                ),
                id=f'Task_{task_id}',
                nb_classes=5,
            )
            self._tasks.append(task)
            print(f'task_id : {task_id}')
            print(f'train features : {len(task.train.features)}')
            print(f'train targets : {len(task.train.targets)}')
        
            print(f'test features : {len(task.test.features)}')
            print(f'test targets : {len(task.test.targets)}')
        
        
    def get_data(self, split: str) -> Dict[str, Union[List, np.ndarray]]:
        features = []
        targets = []
        
        task_splits = defaultdict(lambda: defaultdict(list))
        for iteration, label in tqdm(enumerate(self.label_list), desc=f'Loading {split}'):
            label_idx = int(label.split('_')[-1])
            task_id = iteration // 5
            label_instances = glob(f'{self.root}/{split}/{label}/*')
            task_splits[task_id]['features'].extend(label_instances)
            task_splits[task_id]['targets'].extend([label_idx]*len(label_instances))
        return task_splits

    def choose_task(self, target_task: int):
        self._tasks = [self._tasks[target_task]]
        
    @property
    def tasks(self) -> List[TaskConfig]:
        return self._tasks


class ImageNet100Resized32Task50(Scenario):
    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable] = [train_transform, valid_transform],
    ):
        super().__init__(root, transforms)

        # NOTE : fix bugs for the wrong label index of train/val dataset
        self.label_list = [label.split('/')[-1] for label in glob(f'{self.root}/train/*')]
        self.label_list = np.random.choice(self.label_list, size=250, replace=False).tolist()

        split_train_tasks = self.get_data('train')
        split_test_tasks = self.get_data('val')

        logger.debug(f'split_train_tasks task num : {len(split_train_tasks.keys())}')
        logger.debug(f'split_test_tasks task num : {len(split_test_tasks.keys())}')

        self._tasks = []
        for task_id in range(50):
            task = TaskConfig(
                train=TaskSpecificSplitImageNet32(
                    features=split_train_tasks[task_id]["features"],
                    targets=split_train_tasks[task_id]["targets"],
                    task_id=f'Task_{task_id}',
                    transform=self.transforms[0],
                ),
                test=TaskSpecificSplitImageNet32(
                    features=split_test_tasks[task_id]["features"],
                    targets=split_test_tasks[task_id]["targets"],
                    task_id=f'Task_{task_id}',
                    transform=self.transforms[1],
                ),
                id=f'Task_{task_id}',
                nb_classes=5,
            )
            self._tasks.append(task)
            # print(f'task_id : {task_id}')
            # print(f'train features : {len(task.train.features)}')
            # print(f'train targets : ({set(task.train.targets)}) {len(task.train.targets)}')
        
            # print(f'test features : {len(task.test.features)}')
            # print(f'test targets : ({set(task.test.targets)}) {len(task.test.targets)}')
        # print(f'Tasks : {self.tasks}')
        
    def get_data(self, split: str) -> Dict[str, Union[List, np.ndarray]]:
        features = []
        targets = []
        
        task_splits = defaultdict(lambda: defaultdict(list))
        for iteration, label in tqdm(enumerate(self.label_list), desc=f'Loading {split}'):
            label_idx = int(label.split('_')[-1])
            task_id = iteration // 5
            label_instances = glob(f'{self.root}/{split}/{label}/*')
            task_splits[task_id]['features'].extend(label_instances)
            task_splits[task_id]['targets'].extend([label_idx]*len(label_instances))
        task_splits = dict(task_splits)
        return task_splits

    def choose_task(self, target_task: int):
        self._tasks = [self._tasks[target_task]]
        
    @property
    def tasks(self) -> List[TaskConfig]:
        return self._tasks


class ImageNet100Resized32FixedTask50(Scenario):
    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable] = [train_transform, valid_transform],
    ):
        super().__init__(root, transforms)
        class_info = pd.read_csv('/home/yjkim/cl-git/task_data_loader/imagenet_class.txt', sep=' ', header=None)
        class_info.columns = ['name_n', 'index', 'name_str']
        class_info = class_info.set_index('index').sort_index()

        self.label_list = class_info.iloc[:250]['name_n']

        split_train_tasks = self.get_data('train')
        split_test_tasks = self.get_data('val')

        logger.debug(f'split_train_tasks task num : {len(split_train_tasks.keys())}')
        logger.debug(f'split_test_tasks task num : {len(split_test_tasks.keys())}')

        self._tasks = []
        for task_id in range(50):
            task = TaskConfig(
                train=TaskSpecificSplitImageNet32(
                    features=split_train_tasks[task_id]["features"],
                    targets=split_train_tasks[task_id]["targets"],
                    task_id=f'Task_{task_id}',
                    transform=self.transforms[0],
                ),
                test=TaskSpecificSplitImageNet32(
                    features=split_test_tasks[task_id]["features"],
                    targets=split_test_tasks[task_id]["targets"],
                    task_id=f'Task_{task_id}',
                    transform=self.transforms[1],
                ),
                id=f'Task_{task_id}',
                nb_classes=5,
            )
            self._tasks.append(task)
            # print(f'task_id : {task_id}')
            # print(f'train features : {len(task.train.features)}')
            # print(f'train targets : ({set(task.train.targets)}) {len(task.train.targets)}')
        
            # print(f'test features : {len(task.test.features)}')
            # print(f'test targets : ({set(task.test.targets)}) {len(task.test.targets)}')
        # print(f'{self.root}/train')
        # print(f'Tasks : {self.tasks}')

        # raise KeyError()

    def get_data(self, split: str) -> Dict[str, Union[List, np.ndarray]]:
        features = []
        targets = []
        
        task_splits = defaultdict(lambda: defaultdict(list))
        for label_idx, label in tqdm(enumerate(self.label_list), desc=f'Loading {split}'):
            task_id = label_idx // 5
            label_instances = glob(f'{self.root}/{split}/{label}/*')
            task_splits[task_id]['features'].extend(label_instances)
            task_splits[task_id]['targets'].extend([label]*len(label_instances))
            # print(label_instances)
        task_splits = dict(task_splits)
        return task_splits

    def choose_task(self, target_task: int):
        self._tasks = [self._tasks[target_task]]
        
    @property
    def tasks(self) -> List[TaskConfig]:
        return self._tasks


class ImageNet100Resized32FixedTask200(Scenario):
    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable] = [train_transform, valid_transform],
    ):
        super().__init__(root, transforms)
        class_info = pd.read_csv('/home/yjkim/cl-git/task_data_loader/imagenet_class.txt', sep=' ', header=None)
        class_info.columns = ['name_n', 'index', 'name_str']
        class_info = class_info.set_index('index').sort_index()

        self.label_list = class_info['name_n']

        split_train_tasks = self.get_data('train')
        split_test_tasks = self.get_data('val')

        logger.debug(f'split_train_tasks task num : {len(split_train_tasks.keys())}')
        logger.debug(f'split_test_tasks task num : {len(split_test_tasks.keys())}')

        self._tasks = []
        for task_id in range(200):
            task = TaskConfig(
                train=TaskSpecificSplitImageNet32(
                    features=split_train_tasks[task_id]["features"],
                    targets=split_train_tasks[task_id]["targets"],
                    task_id=f'Task_{task_id}',
                    transform=self.transforms[0],
                ),
                test=TaskSpecificSplitImageNet32(
                    features=split_test_tasks[task_id]["features"],
                    targets=split_test_tasks[task_id]["targets"],
                    task_id=f'Task_{task_id}',
                    transform=self.transforms[1],
                ),
                id=f'Task_{task_id}',
                nb_classes=5,
            )
            self._tasks.append(task)
            # print(f'task_id : {task_id}')
            # print(f'train features : {len(task.train.features)}')
        #     print(f'train targets : ({set(task.train.targets)}) {len(task.train.targets)}')
        
        #     print(f'test features : {len(task.test.features)}')
        #     print(f'test targets : ({set(task.test.targets)}) {len(task.test.targets)}')
        # print(f'{self.root}/train')
        # print(self.label_list)
        # print(f'Tasks : {self.tasks}')

        # raise KeyError()

    def get_data(self, split: str) -> Dict[str, Union[List, np.ndarray]]:
        features = []
        targets = []
        
        task_splits = defaultdict(lambda: defaultdict(list))
        for label_idx, label in tqdm(enumerate(self.label_list), desc=f'Loading {split}'):
            task_id = label_idx // 5
            label_instances = glob(f'{self.root}/{split}/{label}/*')
            task_splits[task_id]['features'].extend(label_instances)
            task_splits[task_id]['targets'].extend([label]*len(label_instances))
            # print(task_splits[task_id])
            # print(label_instances)
        task_splits = dict(task_splits)
        return task_splits

    def choose_task(self, target_task: int):
        self._tasks = [self._tasks[target_task]]
        
    @property
    def tasks(self) -> List[TaskConfig]:
        return self._tasks


class ImageNet32Task200Scenario(Scenario):
    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable] = [train_transform, valid_transform],
    ):
        super().__init__(root, transforms)

        split_train_tasks = self.get_data('train')
        split_test_tasks = self.get_data('val')

        logger.debug(f'split_train_tasks task num : {len(split_train_tasks.keys())}')
        logger.debug(f'split_test_tasks task num : {len(split_test_tasks.keys())}')

        self._tasks = []
        for task_id in range(200):
            task = TaskConfig(
                train=TaskSpecificSplitImageNet32(
                    features=split_train_tasks[task_id]["features"],
                    targets=split_train_tasks[task_id]["targets"],
                    task_id=f'Task_{task_id}',
                    transform=self.transforms[0],
                ),
                test=TaskSpecificSplitImageNet32(
                    features=split_test_tasks[task_id]["features"],
                    targets=split_test_tasks[task_id]["targets"],
                    task_id=f'Task_{task_id}',
                    transform=self.transforms[1],
                ),
                id=f'Task_{task_id}',
                nb_classes=5,
            )
            self._tasks.append(task)
            logger.debug(f'task_id : {task_id}')
            logger.debug(f'train features : {len(task.train.features)}')
            logger.debug(f'train targets : {len(task.train.targets)}')
        
            logger.debug(f'test features : {len(task.test.features)}')
            logger.debug(f'test targets : {len(task.test.targets)}')
        
        
    def get_data(self, split: str) -> Dict[str, Union[List, np.ndarray]]:
        features = []
        targets = []
        
        label_list = [label.split('/')[-1] for label in glob(f'{self.root}/{split}/*')]
        
        task_splits = defaultdict(lambda: defaultdict(list))
        for label in tqdm(label_list, desc=f'Loading {split}'):
            label_idx = int(label.split('_')[-1])
            task_id = label_idx // 5
            label_instances = glob(f'{self.root}/{split}/{label}/*')
            task_splits[task_id]['features'].extend(label_instances)
            task_splits[task_id]['targets'].extend([label_idx]*len(label_instances))
        return task_splits

    def choose_task(self, target_task: int):
        self._tasks = [self._tasks[target_task]]
        
    @property
    def tasks(self) -> List[TaskConfig]:
        return self._tasks


class ImageNet32Task500Scenario(Scenario):
    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable] = [train_transform, valid_transform],
    ):
        super().__init__(root, transforms)

        split_train_tasks = self.get_data('train')
        split_test_tasks = self.get_data('val')

        logger.debug(f'split_train_tasks task num : {len(split_train_tasks.keys())}')
        logger.debug(f'split_test_tasks task num : {len(split_test_tasks.keys())}')

        self._tasks = []
        for task_id in range(500):
            task = TaskConfig(
                train=TaskSpecificSplitImageNet32(
                    features=split_train_tasks[task_id]["features"],
                    targets=split_train_tasks[task_id]["targets"],
                    task_id=f'Task_{task_id}',
                    transform=self.transforms[0],
                ),
                test=TaskSpecificSplitImageNet32(
                    features=split_test_tasks[task_id]["features"],
                    targets=split_test_tasks[task_id]["targets"],
                    task_id=f'Task_{task_id}',
                    transform=self.transforms[1],
                ),
                id=f'Task_{task_id}',
                nb_classes=2,
            )
            self._tasks.append(task)
            logger.debug(f'task_id : {task_id}')
            logger.debug(f'train features : {len(task.train.features)}')
            logger.debug(f'train targets : {len(task.train.targets)}')
        
            logger.debug(f'test features : {len(task.test.features)}')
            logger.debug(f'test targets : {len(task.test.targets)}')
        
        
    def get_data(self, split: str) -> Dict[str, Union[List, np.ndarray]]:
        features = []
        targets = []
        
        label_list = [label.split('/')[-1] for label in glob(f'{self.root}/{split}/*')]
        
        task_splits = defaultdict(lambda: defaultdict(list))
        for label in tqdm(label_list, desc=f'Loading {split}'):
            label_idx = int(label.split('_')[-1])
            task_id = label_idx // 2
            label_instances = glob(f'{self.root}/{split}/{label}/*')
            task_splits[task_id]['features'].extend(label_instances)
            task_splits[task_id]['targets'].extend([label_idx]*len(label_instances))
        return task_splits

    def choose_task(self, target_task: int):
        self._tasks = [self._tasks[target_task]]
        
    @property
    def tasks(self) -> List[TaskConfig]:
        return self._tasks


class ImageNet32Task250Scenario(Scenario):
    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable] = [train_transform, valid_transform],
    ):
        super().__init__(root, transforms)

        split_train_tasks = self.get_data('train')
        split_test_tasks = self.get_data('val')

        logger.debug(f'split_train_tasks task num : {len(split_train_tasks.keys())}')
        logger.debug(f'split_test_tasks task num : {len(split_test_tasks.keys())}')

        self._tasks = []
        for task_id in range(250):
            task = TaskConfig(
                train=TaskSpecificSplitImageNet32(
                    features=split_train_tasks[task_id]["features"],
                    targets=split_train_tasks[task_id]["targets"],
                    task_id=f'Task_{task_id}',
                    transform=self.transforms[0],
                ),
                test=TaskSpecificSplitImageNet32(
                    features=split_test_tasks[task_id]["features"],
                    targets=split_test_tasks[task_id]["targets"],
                    task_id=f'Task_{task_id}',
                    transform=self.transforms[1],
                ),
                id=f'Task_{task_id}',
                nb_classes=4
            )
            self._tasks.append(task)
            logger.debug(f'task_id : {task_id}')
            logger.debug(f'train features : {len(task.train.features)}')
            logger.debug(f'train targets : {len(task.train.targets)}')
        
            logger.debug(f'test features : {len(task.test.features)}')
            logger.debug(f'test targets : {len(task.test.targets)}')
        
        
    def get_data(self, split: str) -> Dict[str, Union[List, np.ndarray]]:
        features = []
        targets = []
        
        label_list = [label.split('/')[-1] for label in glob(f'{self.root}/{split}/*')]
        
        task_splits = defaultdict(lambda: defaultdict(list))
        for label in tqdm(label_list, desc=f'Loading {split}'):
            label_idx = int(label.split('_')[-1])
            task_id = label_idx // 4
            label_instances = glob(f'{self.root}/{split}/{label}/*')
            task_splits[task_id]['features'].extend(label_instances)
            task_splits[task_id]['targets'].extend([label_idx]*len(label_instances))
        return task_splits

    def choose_task(self, target_task: int):
        self._tasks = [self._tasks[target_task]]
        
    @property
    def tasks(self) -> List[TaskConfig]:
        return self._tasks


class ImageNetScenario(Scenario):
    def __init__(self, root: str = "data", transforms: Optional[List[Callable]] = None):
        super().__init__(root, transforms)
        if self.transforms is None:
            self.transforms = [train_transform, valid_transform]

        self.imagenet = TaskConfig(
            train=ImageNet(root='/data/files/imagenet-1k', split="train", download=None, transform=self.transforms[0]),
            test=ImageNet(root='/data/files/imagenet-1k', split="val", download=None, transform=self.transforms[1]),
            id="imagenet",
            nb_classes=1000,
        )

    @property
    def tasks(self) -> List[TaskConfig]:
        return [self.imagenet]


class ScenesScenario(Scenario):
    def __init__(self, root: str = "data", transforms: Optional[List[Callable]] = None):
        super().__init__(root, transforms)
        if self.transforms is None:
            self.transforms = [train_transform, valid_transform]

        self.scenes = TaskConfig(
            train=Scenes(root=self.root, train=True, transform=self.transforms[0]),
            test=Scenes(root=self.root, train=False, transform=self.transforms[1]),
            id="scenes",
            nb_classes=67,
        )

    @property
    def tasks(self) -> List[TaskConfig]:
        return [self.scenes]


class ImageNet2CUB(Scenario):
    def __init__(self, root: str = "data", transforms: Optional[List[Callable]] = None):
        super().__init__(root, transforms)
        if self.transforms is None:
            self.transforms = [train_transform, valid_transform]

        self.imagenet = TaskConfig(
            train=ImageNet(root='/data/files/imagenet-1k', split="train", download=None, transform=self.transforms[0]),
            test=ImageNet(root='/data/files/imagenet-1k', split="val", download=None, transform=self.transforms[1]),
            id="imagenet",
            nb_classes=1000,
        )
        self.cub = TaskConfig(
            train=Cub2011(root=self.root, train=True, transform=self.transforms[0]),
            test=Cub2011(root=self.root, train=False, transform=self.transforms[1]),
            id="cub",
            nb_classes=200,
        )

    @property
    def tasks(self) -> List[TaskConfig]:
        return [self.imagenet, self.cub]


class ImageNet2Scenes(Scenario):
    def __init__(self, root: str = "data", transforms: Optional[List[Callable]] = None):
        super().__init__(root, transforms)
        if self.transforms is None:
            self.transforms = [train_transform, valid_transform]

        self.imagenet = TaskConfig(
            train=ImageNet(root='/data/files/imagenet-1k', split="train", download=None, transform=self.transforms[0]),
            test=ImageNet(root='/data/files/imagenet-1k', split="val", download=None, transform=self.transforms[1]),
            id="imagenet",
            nb_classes=1000,
        )

        self.scenes = TaskConfig(
            train=Scenes(root=self.root, train=True, transform=self.transforms[0]),
            test=Scenes(root=self.root, train=False, transform=self.transforms[1]),
            id="scenes",
            nb_classes=67,
        )

    @property
    def tasks(self) -> List[TaskConfig]:
        return [self.imagenet, self.scenes]


class ImageNet2Scenes2CUB(Scenario):
    def __init__(self, root: str = "data", transforms: Optional[List[Callable]] = None):
        super().__init__(root, transforms)
        if self.transforms is None:
            self.transforms = [train_transform, valid_transform]

        self.imagenet = TaskConfig(
            train=ImageNet(root='/data/files/imagenet-1k', split="train", download=None, transform=self.transforms[0]),
            test=ImageNet(root='/data/files/imagenet-1k', split="val", download=None, transform=self.transforms[1]),
            id="imagenet",
            nb_classes=1000,
        )

        self.scenes = TaskConfig(
            train=Scenes(root=self.root, train=True, transform=self.transforms[0]),
            test=Scenes(root=self.root, train=False, transform=self.transforms[1]),
            id="scenes",
            nb_classes=67,
        )

        self.cub = TaskConfig(
            train=Cub2011(root=self.root, train=True, transform=self.transforms[0]),
            test=Cub2011(root=self.root, train=False, transform=self.transforms[1]),
            id="cub",
            nb_classes=200,
        )

    @property
    def tasks(self) -> List[TaskConfig]:
        return [self.imagenet, self.scenes, self.cub]


class ImageNet2Flowers(Scenario):
    def __init__(self, root: str = "data", transforms: Optional[List[Callable]] = None):
        super().__init__(root, transforms)
        if self.transforms is None:
            self.transforms = [train_transform, valid_transform]

        self.imagenet = TaskConfig(
            train=ImageNet(root='/data/files/imagenet-1k', split="train", download=None, transform=self.transforms[0]),
            test=ImageNet(root='/data/files/imagenet-1k', split="val", download=None, transform=self.transforms[1]),
            id="imagenet",
            nb_classes=1000,
        )
        self.flowers = TaskConfig(
            train=Flowers102(root=self.root, train=True, transform=self.transforms[0]),
            test=Flowers102(root=self.root, train=False, transform=self.transforms[1]),
            id="flowers",
            nb_classes=102,
        )

    @property
    def tasks(self) -> List[TaskConfig]:
        return [self.imagenet, self.flowers]


class ImageNet2Scenes2CUB2Flowers(Scenario):
    def __init__(self, root: str = "data", transforms: Optional[List[Callable]] = None):
        super().__init__(root, transforms)
        if self.transforms is None:
            self.transforms = [train_transform, valid_transform]

        self.imagenet = TaskConfig(
            train=ImageNet(root='/data/files/imagenet-1k', split="train", download=None, transform=self.transforms[0]),
            test=ImageNet(root='/data/files/imagenet-1k', split="val", download=None, transform=self.transforms[1]),
            id="imagenet",
            nb_classes=1000,
        )

        self.scenes = TaskConfig(
            train=Scenes(root=self.root, train=True, transform=self.transforms[0]),
            test=Scenes(root=self.root, train=False, transform=self.transforms[1]),
            id="scenes",
            nb_classes=67,
        )

        self.cub = TaskConfig(
            train=Cub2011(root=self.root, train=True, transform=self.transforms[0]),
            test=Cub2011(root=self.root, train=False, transform=self.transforms[1]),
            id="cub",
            nb_classes=200,
        )
        self.flowers = TaskConfig(
            train=Flowers102(root=self.root, train=True, transform=self.transforms[0]),
            test=Flowers102(root=self.root, train=False, transform=self.transforms[1]),
            id="flowers",
            nb_classes=102,
        )

    @property
    def tasks(self) -> List[TaskConfig]:
        return [self.imagenet, self.scenes, self.cub, self.flowers]
