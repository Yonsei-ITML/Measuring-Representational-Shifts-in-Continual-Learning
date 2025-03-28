from typing import List

import torch
from torch import nn as nn, Tensor
from torch.nn import ModuleDict, Linear

from models.imagenet_based_models import PredictionLayerConfig
from utilities.utils import xavier_uniform_initialize

import os
import logging
from tqdm import tqdm




class LinearProbeCIFAR10(nn.Module):
    def __init__(
        self,
        under_investigation_model: nn.Module,
        intended_block: str,
    ):
        super(LinearProbeCIFAR10, self).__init__()
        self.intended_block = intended_block
        self.in_channel = under_investigation_model.block_output_size[self.intended_block]
        self.under_investigation_blocks = under_investigation_model.blocks
        # Freezing the blocks
        for block_values in self.under_investigation_blocks.values():
            for parameters in block_values.parameters():
                parameters.requires_grad = False

        self.fc_task1 = nn.Linear(self.in_channel, 5)
        self.fc_task2 = nn.Linear(self.in_channel, 5)
        xavier_uniform_initialize(self.fc_task1)
        xavier_uniform_initialize(self.fc_task2)

    def forward(self, features: Tensor, task_id: int):
        for block_name, operations in self.under_investigation_blocks.items():
            features = operations(features)
            if block_name == self.intended_block:
                break
        features = torch.flatten(features, 1)
        if task_id == 1:
            features = self.fc_task1(features)
        else:
            features = self.fc_task2(features)
        return features


class LinearProbeCIFAR100(nn.Module):
    def __init__(
        self,
        under_investigation_model: nn.Module,
        intended_block: str,
    ):
        super(LinearProbeCIFAR100, self).__init__()
        self.intended_block = intended_block
        self.in_channel = under_investigation_model.block_output_size[self.intended_block]
        self.under_investigation_blocks = under_investigation_model.blocks
        # Freezing the blocks
        for block_values in self.under_investigation_blocks.values():
            for parameters in block_values.parameters():
                parameters.requires_grad = False

        self.projectors = nn.ModuleDict()
        for i in range(20):
            task_projector = nn.Linear(self.in_channel, 5).to('cuda')
            xavier_uniform_initialize(task_projector)
            self.projectors[f'Task_{i}'] = task_projector

    def forward(self, features: Tensor, task_id):
        for block_name, operations in self.under_investigation_blocks.items():
            features = operations(features)
            
            if block_name == self.intended_block:
                break
        features = torch.flatten(features, 1).detach()
        features = self.projectors[f'Task_{task_id}'](features)
        return features


class LinearProbeCIFAR100Task50(nn.Module):
    def __init__(
        self,
        under_investigation_model: nn.Module,
        intended_block: str,
    ):
        super(LinearProbeCIFAR100Task50, self).__init__()
        self.intended_block = intended_block
        self.in_channel = under_investigation_model.block_output_size[self.intended_block]
        self.under_investigation_blocks = under_investigation_model.blocks
        # Freezing the blocks
        for block_values in self.under_investigation_blocks.values():
            for parameters in block_values.parameters():
                parameters.requires_grad = False

        self.projectors = nn.ModuleDict()
        for i in range(50):
            task_projector = nn.Linear(self.in_channel, 5).to('cuda')
            xavier_uniform_initialize(task_projector)
            self.projectors[f'Task_{i}'] = task_projector

    def forward(self, features: Tensor, task_id):
        for block_name, operations in self.under_investigation_blocks.items():
            features = operations(features)
            
            if block_name == self.intended_block:
                break
        features = torch.flatten(features, 1).detach()
        features = self.projectors[f'Task_{task_id}'](features)
        return features


class LinearProbeCIFAR100Task200(nn.Module):
    def __init__(
        self,
        under_investigation_model: nn.Module,
        intended_block: str,
    ):
        super(LinearProbeCIFAR100Task200, self).__init__()
        self.intended_block = intended_block
        self.in_channel = under_investigation_model.block_output_size[self.intended_block]
        self.under_investigation_blocks = under_investigation_model.blocks
        # Freezing the blocks
        for block_values in self.under_investigation_blocks.values():
            for parameters in block_values.parameters():
                parameters.requires_grad = False

        self.projectors = nn.ModuleDict()
        for i in range(200):
            task_projector = nn.Linear(self.in_channel, 5).to('cuda')
            xavier_uniform_initialize(task_projector)
            self.projectors[f'Task_{i}'] = task_projector

    def forward(self, features: Tensor, task_id):
        for block_name, operations in self.under_investigation_blocks.items():
            features = operations(features)
            
            if block_name == self.intended_block:
                break
        features = torch.flatten(features, 1).detach()
        features = self.projectors[f'Task_{task_id}'](features)
        return features


class LinearProbeSymbolCount(nn.Module):
    def __init__(
        self,
        under_investigation_model: nn.Module,
        intended_block: str,
    ):
        super(LinearProbeSymbolCount, self).__init__()
        self.intended_block = intended_block
        self.embedding = under_investigation_model.embedding
        self.in_channel = under_investigation_model.block_output_size[self.intended_block] * under_investigation_model.seq_len
        self.under_investigation_blocks = under_investigation_model.blocks
        # Freezing the blocks
        for block_values in self.under_investigation_blocks.values():
            for parameters in block_values.parameters():
                parameters.requires_grad = False

        self.projectors = nn.ModuleDict()
        for i in range(200):
            task_projector = nn.Linear(self.in_channel, under_investigation_model.seq_len).to('cuda')
            xavier_uniform_initialize(task_projector)
            self.projectors[f'Task_{i}'] = task_projector

    def forward(self, features: Tensor, task_id):
        features = self.embedding(features)
        for block_name, operations in self.under_investigation_blocks.items():
            features = operations(features)
            
            if block_name == self.intended_block:
                break
        features = torch.flatten(features, 1).detach()
        features = self.projectors[f'Task_{task_id}'](features)
        # raise KeyError()
        return features


class LinearProbeComplexSymbolCount(nn.Module):
    def __init__(
        self,
        under_investigation_model: nn.Module,
        intended_block: str,
    ):
        super(LinearProbeComplexSymbolCount, self).__init__()
        self.intended_block = intended_block
        self.embedding = under_investigation_model.embedding
        self.in_channel = under_investigation_model.block_output_size[self.intended_block] * under_investigation_model.seq_len
        self.under_investigation_blocks = under_investigation_model.blocks
        # Freezing the blocks
        for block_values in self.under_investigation_blocks.values():
            for parameters in block_values.parameters():
                parameters.requires_grad = False

        self.projectors = nn.ModuleDict()
        for i in range(200):
            task_projector = nn.Linear(self.in_channel, 5).to('cuda')
            xavier_uniform_initialize(task_projector)
            self.projectors[f'Task_{i}'] = task_projector

    def forward(self, features: Tensor, task_id):
        features = self.embedding(features)
        for block_name, operations in self.under_investigation_blocks.items():
            features = operations(features)
            
            if block_name == self.intended_block:
                break
        features = torch.flatten(features, 1).detach()
        features = self.projectors[f'Task_{task_id}'](features)
        return features


class LinearProbeImageNet(nn.Module):
    def __init__(
        self,
        under_investigation_model: nn.Module,
        intended_block: str,
        prediction_layers: List[PredictionLayerConfig],
        half_precision: bool = False,
    ):
        super(LinearProbeImageNet, self).__init__()
        self.half_precision = half_precision
        self.prediction_layers = prediction_layers
        self.intended_block = intended_block
        if self.half_precision:
            self.under_investigation_model = under_investigation_model.half()
        else:
            self.under_investigation_model = under_investigation_model
        self.in_channel = under_investigation_model.block_output_size[self.intended_block]
        self.task_prediction = ModuleDict(
            {
                conf.task_id: Linear(in_features=self.in_channel, out_features=conf.nb_classes)
                for conf in self.prediction_layers
            }
        )
        for layer_w in self.task_prediction.values():
            xavier_uniform_initialize(layer_w)
        # Freezing the blocks
        for parameters in self.under_investigation_model.parameters():
            parameters.requires_grad = False

    def forward(self, features: Tensor, task_id: str):
        if self.half_precision:
            features = features.half()
        with torch.inference_mode():
            features = self.under_investigation_model.block_forward(
                features=features, task_id=task_id, numpy_return=False
            )[self.intended_block].detach()
            features = torch.flatten(features, 1)
        features = features.clone()
        if self.half_precision:
            return self.task_prediction[task_id](features.float())
        return self.task_prediction[task_id](features)


class LinearProbeImageNetTask500(nn.Module):
    def __init__(
        self,
        under_investigation_model: nn.Module,
        intended_block: str,
    ):
        super(LinearProbeImageNetTask500, self).__init__()
        self.intended_block = intended_block
        self.in_channel = under_investigation_model.block_output_size[self.intended_block]
        self.under_investigation_blocks = under_investigation_model.blocks
        for block_values in self.under_investigation_blocks.values():
            for parameters in block_values.parameters():
                parameters.requires_grad = False

        self.projectors = nn.ModuleDict()
        for i in range(500):
            task_projector = nn.Linear(self.in_channel, 2).to('cuda')
            xavier_uniform_initialize(task_projector)
            self.projectors[f'Task_{i}'] = task_projector

    def forward(self, features: Tensor, task_id):
        for block_name, operations in self.under_investigation_blocks.items():
            features = operations(features)
            
            if block_name == self.intended_block:
                break

        features = torch.flatten(features, 1).detach()

        features = self.projectors[f'{task_id}'](features)
        
        return features


class LinearProbeImageNetTask20(nn.Module):
    def __init__(
        self,
        under_investigation_model: nn.Module,
        intended_block: str,
    ):
        super(LinearProbeImageNetTask20, self).__init__()
        self.intended_block = intended_block
        self.in_channel = under_investigation_model.block_output_size[self.intended_block]
        self.under_investigation_blocks = under_investigation_model.blocks
        # Freezing the blocks
        for block_values in self.under_investigation_blocks.values():
            for parameters in block_values.parameters():
                parameters.requires_grad = False

        self.projectors = nn.ModuleDict()
        for i in range(20):
            task_projector = nn.Linear(self.in_channel, 5).to('cuda')
            xavier_uniform_initialize(task_projector)
            self.projectors[f'Task_{i}'] = task_projector

    def forward(self, features: Tensor, task_id):
        for block_name, operations in self.under_investigation_blocks.items():
            features = operations(features)
            
            if block_name == self.intended_block:
                break

        features = torch.flatten(features, 1).detach()

        features = self.projectors[f'{task_id}'](features)
        return features

class LinearProbeImageNetTask50(nn.Module):
    def __init__(
        self,
        under_investigation_model: nn.Module,
        intended_block: str,
    ):
        super(LinearProbeImageNetTask50, self).__init__()
        self.intended_block = intended_block
        self.in_channel = under_investigation_model.block_output_size[self.intended_block]
        self.under_investigation_blocks = under_investigation_model.blocks
        for block_values in self.under_investigation_blocks.values():
            for parameters in block_values.parameters():
                parameters.requires_grad = False

        self.projectors = nn.ModuleDict()
        for i in range(50):
            task_projector = nn.Linear(self.in_channel, 5).to('cuda')
            xavier_uniform_initialize(task_projector)
            self.projectors[f'Task_{i}'] = task_projector

    def forward(self, features: Tensor, task_id):
        for block_name, operations in self.under_investigation_blocks.items():
            features = operations(features)
            
            if block_name == self.intended_block:
                break

        features = torch.flatten(features, 1).detach()

        features = self.projectors[f'{task_id}'](features)
        return features


class LinearProbeImageNetTask200(nn.Module):
    def __init__(
        self,
        under_investigation_model: nn.Module,
        intended_block: str,
    ):
        super(LinearProbeImageNetTask200, self).__init__()
        self.intended_block = intended_block
        self.in_channel = under_investigation_model.block_output_size[self.intended_block]
        self.under_investigation_blocks = under_investigation_model.blocks
        for block_values in self.under_investigation_blocks.values():
            for parameters in block_values.parameters():
                parameters.requires_grad = False

        self.projectors = nn.ModuleDict()
        for i in range(200):
            task_projector = nn.Linear(self.in_channel, 5).to('cuda')
            xavier_uniform_initialize(task_projector)
            self.projectors[f'Task_{i}'] = task_projector

    def forward(self, features: Tensor, task_id):
        for block_name, operations in self.under_investigation_blocks.items():
            features = operations(features)
            
            if block_name == self.intended_block:
                break

        features = torch.flatten(features, 1).detach()

        features = self.projectors[f'{task_id}'](features)
        return features

class LinearProbeFlatMiniResNet(nn.Module):
    def __init__(
        self,
        under_investigation_model: nn.Module,
        intended_block: str,
    ):
        super(LinearProbeFlatMiniResNet, self).__init__()
        self.intended_block = intended_block
        self.in_channel = under_investigation_model.block_output_size[self.intended_block]
        self.under_investigation_blocks = under_investigation_model.blocks
        self.task_num = under_investigation_model.task_num

        # Freezing the blocks
        for block_values in self.under_investigation_blocks.values():
            for parameters in block_values.parameters():
                parameters.requires_grad = False

        self.projectors = nn.ModuleDict()
        for i in range(self.task_num):
            task_projector = nn.Linear(self.in_channel, 5).to('cuda')
            xavier_uniform_initialize(task_projector)
            self.projectors[f'Task_{i}'] = task_projector

    def forward(self, features: Tensor, task_id):
        for block_name, operations in self.under_investigation_blocks.items():
            features = operations(features)
            
            if block_name == self.intended_block:
                break

        features = torch.flatten(features, 1).detach()

        features = self.projectors[f'{task_id}'](features)
        return features