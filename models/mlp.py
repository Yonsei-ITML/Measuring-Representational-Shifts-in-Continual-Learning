from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch
from torch import nn as nn, Tensor
from torch.nn import functional as F

from utilities.utils import to_numpy

import os
import logging


class TaskBasedNets(ABC, nn.Module):
    @abstractmethod
    def forward(self, features: Tensor, task_id: str):
        pass

    def block_forward(self, features: Tensor, task_id: str) -> Dict[str, torch.Tensor]:
        blocks_recorded = dict()
        for key, operations in self.blocks.items():
            features = operations(features)
            blocks_recorded[key] = torch.flatten(features, 1).detach()
        return blocks_recorded

    def predict(self, features: Tensor, task_id: str) -> np.ndarray:
        outputs = self.forward(features=features, task_id=task_id)
        _, predictions = torch.max(outputs, 1)
        return to_numpy(predictions)


class ReLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        super().__init__()

        self.layer = nn.Linear(in_channels, out_channels, bias=bias)
        self.activation = nn.ReLU()

    def forward(self, x):
        transformed = self.layer(x)
        return self.activation(transformed)


class ReLUNetCIFAR100(TaskBasedNets):
    def __init__(self, width, depth):
        super().__init__()

        self.widths = [3072] + [width] * depth

        self.blocks = nn.ModuleDict()
        self.block_output_size = {}

        for idx, (in_ch, out_ch) in enumerate(zip(self.widths[:-1], self.widths[1:])):
            self.blocks[f'block{idx}'] = ReLUBlock(in_ch, out_ch, bias=True)
            self.block_output_size[f'block{idx}'] = out_ch

        # compute conv feature size
        with torch.no_grad():
            dummy_data = torch.zeros(
                (1, 3, 32, 32),
                dtype=torch.float32,
            )
            dummy_data = torch.flatten(dummy_data, 1)
            self.feature_size = self._forward_backbone(dummy_data).view(-1).shape[0]
        self.projectors = nn.ModuleDict()
        for i in range(20):
            task_projector = nn.Linear(self.feature_size, 100)
            self.projectors[f'{i}'] = task_projector
        

    def _forward_backbone(self, features: Tensor):
        for operations in self.blocks.values():
            features = operations(features)
        return features

    def forward(self, features: Tensor, task_id: str):
        features = torch.flatten(features, 1)   
        features = self._forward_backbone(features)
        features = self.projectors[task_id](features)
        return features

