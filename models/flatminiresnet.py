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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size=3, padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )  # downsample with first conv

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                "conv",
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),  # downsample
            )
            self.shortcut.add_module("bn", nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y


class FlatMiniResNet(TaskBasedNets):
    def __init__(self, task_num, internal_channel):
        super().__init__()
        self.task_num = task_num
        self.internal_channel = internal_channel
        n_stages = 4
        n_blocks_per_stage = 2
        n_channels = [self.internal_channel for i in range(n_stages)]

        self.stage0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.internal_channel, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(num_features=self.internal_channel),
            nn.ReLU(inplace=True),
        )  # out: 32 * 224 * 224
        self.stage1 = self._make_stage(n_channels[0], n_channels[0], n_blocks_per_stage, ResidualBlock, stride=1, kernel_size=7, padding=3)
        self.stage2 = self._make_stage(n_channels[0], n_channels[1], n_blocks_per_stage, ResidualBlock, stride=1, kernel_size=7, padding=3)
        self.stage3 = self._make_stage(n_channels[1], n_channels[2], n_blocks_per_stage, ResidualBlock, stride=1, kernel_size=3, padding=1)
        self.stage4 = self._make_stage(n_channels[2], n_channels[3], n_blocks_per_stage, ResidualBlock, stride=1, kernel_size=3, padding=1)

        self.blocks = nn.ModuleDict({"block0": self.stage0})

        self.blocks.update(
            nn.ModuleDict(
                {
                    f"block{stage_id * 2 + residual_block_id + 1}": residual_block
                    for stage_id, stage in enumerate([self.stage1, self.stage2, self.stage3, self.stage4])
                    # for stage_id, stage in enumerate(self.stages)
                    for residual_block_id, residual_block in enumerate([stage.block1, stage.block2])
                }
            )
        )     

        sizes = self.internal_channel * 32 * 32
        self.block_output_size = {f'block{idx}': sizes for idx in range(2*n_stages+1)}

        # compute conv feature size
        with torch.no_grad():
            dummy_data = torch.zeros(
                (1, 3, 32, 32),
                dtype=torch.float32,
            )
            self.feature_size = self._forward_conv(dummy_data).view(-1).shape[0]
        
        self.projectors = nn.ModuleDict()
        for i in range(task_num):
            task_projector = nn.Linear(self.feature_size, 5)
            self.projectors[f'Task_{i}'] = task_projector
        
    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride, kernel_size, padding):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f"block{index + 1}"
            if index == 0:
                stage.add_module(block_name, block(in_channels, out_channels, stride=stride, kernel_size=kernel_size, padding=padding))
            else:
                stage.add_module(block_name, block(out_channels, out_channels, stride=1, kernel_size=kernel_size, padding=padding))
        return stage

    def _forward_conv(self, features: Tensor):
        for operations in self.blocks.values():
            features = operations(features)
        features = F.adaptive_avg_pool2d(features, output_size=1)
        return features

    def forward(self, features: Tensor, task_id: str):
        features = self._forward_conv(features)
        features = torch.flatten(features, 1)
        features = self.projectors[task_id](features)
        return features
