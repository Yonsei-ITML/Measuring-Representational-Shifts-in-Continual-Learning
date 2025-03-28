from typing import List, Optional, Callable
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import logging
from tqdm import tqdm
import os


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler(f'./logging/{__name__}.log', 'w'))

with open(f'{os.getenv("SRC_ROOT")}/task_data_loader/imagenet_class.txt', 'r') as f:
    lines = f.readlines()
lines = list(map(lambda x: x.strip('\n').split(' '), lines))
imagenet_classes = {int(line[1]): line[0] for line in lines}
# i.e. {nXXXXXXXX : 10}

class TaskSpecificSplitImageNet32(Dataset):
    classes = imagenet_classes

    def __init__(self, features: np.ndarray, targets: List[int], task_id: int, transform: Optional[Callable] = None):
        self.features = features
        self.targets = targets
        self.task_id = task_id
        self.transform = transform
        self.target_classes_ID = set(targets)
        self.target_classes_dict = {
            class_str:idx  for idx, class_str in self.classes.items() if idx in self.target_classes_ID
        }
        self.internal_target_def = {
            external_id: internal_id for internal_id, external_id in enumerate(self.target_classes_ID)
        }
        # self.reverse_internal_target = {
        #     internal_id: self.target_classes_dict[external_id]
        #     for external_id, internal_id in self.internal_target_def.items()
        # }

    def __getitem__(self, item: int):
        img_path, target = self.features[item], self.targets[item]
        # logger.debug(f'img_path:{img_path}')        
        img = pil_loader(img_path)
        # logger.debug(f'img:{type(img)}{img}')        
        if self.transform is not None:
            img = self.transform(img)
        return img, self.internal_target_def[target]

    def __len__(self) -> int:
        return len(self.features)

    def __str__(self) -> str:
        return f"Task ID: {self.task_id}\nTarget Classes: {self.target_classes_dict}"


def pil_loader(path: str) -> Image.Image:
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # https://github.com/pytorch/vision/blob/main/torchvision/datasets/folder.py#L260
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")