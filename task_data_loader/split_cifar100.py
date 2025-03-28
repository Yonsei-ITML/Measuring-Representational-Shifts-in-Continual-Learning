from typing import List, Optional, Callable
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import os
import logging
from tqdm import tqdm


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler(f'./logging/{os.getenv("LOG_RUN_NAME")}_{__name__}.log', 'w'))


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


class TaskSpecificSplitCIFAR100(Dataset):
    def __init__(self, features: np.ndarray, targets: List[int], task_id: int, transform: Optional[Callable] = None):
        CIFAR100_CLASSES = unpickle(f'{os.getenv("DATA_ROOT")}/cifar-100-python/meta')['fine_label_names']
        CIFAR100_CLASSES = {idx: label for idx, label in enumerate(CIFAR100_CLASSES)}

        self.classes = CIFAR100_CLASSES        
        self.features = features
        self.targets = targets
        self.task_id = task_id
        self.transform = transform
        self.target_classes_ID = set(targets)
        logger.debug(f'self.target_classes_ID : {self.target_classes_ID}')
        # logger.debug(f'self.classes : {self.classes}')
        self.target_classes_dict = {
            idx: class_str for idx, class_str in self.classes.items() if idx in self.target_classes_ID
        }
        logger.debug(f'self.target_classes_dict : {self.target_classes_dict}')
        self.internal_target_def = {
            external_id: internal_id for internal_id, external_id in enumerate(self.target_classes_ID)
        }
        logger.debug(f'self.internal_target_def : {self.internal_target_def}')
        self.reverse_internal_target = {
            internal_id: self.target_classes_dict[external_id]
            for external_id, internal_id in self.internal_target_def.items()
        }

    def __getitem__(self, item: int):
        img, target = self.features[item], self.targets[item]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.internal_target_def[target]

    def __len__(self) -> int:
        return len(self.features)

    def __str__(self) -> str:
        return f"Task ID: {self.task_id}\nTarget Classes: {self.target_classes_dict}"


cifar100_basic_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)
