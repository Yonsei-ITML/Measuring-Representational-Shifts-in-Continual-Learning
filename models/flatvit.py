import os
import logging
import numpy as np
from typing import Dict
from abc import ABC, abstractmethod

from utilities.utils import to_numpy

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


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

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
 
    def forward(self, x):
        x = self.proj(x)  # Shape: (B, embed_dim, num_patches**0.5, num_patches**0.5)
        x = x.flatten(2)  # Shape: (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # Shape: (B, num_patches, embed_dim)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # Shape: (3, B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class FlatVisionTransformer(nn.Module):
    def __init__(self, task_num, internal_channel):
        super(FlatVisionTransformer, self).__init__()

        img_size=32 # 224
        patch_size=16 # 16
        in_channels=3
        embed_dim=16 # 768
        depth=9 # 12
        num_heads=1 # 12
        mlp_ratio=4.0
        dropout=0.1
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleDict({
            f'block{depth_idx}' : TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for depth_idx in range(depth)
        })
  
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.projectors = nn.ModuleDict()
        for i in range(task_num):
            task_projector = nn.Linear(embed_dim, 5)
            self.projectors[f'Task_{i}'] = task_projector

    def forward(self, features: Tensor, task_id: str):
        B = features.shape[0]
        features = self.patch_embed(features)  # Shape: (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # Shape: (B, 1, embed_dim)
        features = torch.cat((cls_tokens, features), dim=1)  # Shape: (B, 1 + num_patches, embed_dim)
        features = features + self.pos_embed
        features = self.pos_dropout(features)

        for block in self.blocks:
            features = block(features)

        features = self.norm(features)
        cls_token_final = features[:, 0]  # Shape: (B, embed_dim)
        features = self.projectors[task_id](cls_token_final) # Shape: (B, num_classes=5)
        return features
