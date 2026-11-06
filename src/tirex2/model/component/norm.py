# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x.to(in_dtype)


class MultiHeadLayerNorm(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads * head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, NH, DH = x.shape
        assert NH == self.num_heads, f"Expected {self.num_heads} heads, got {NH}, input shape: {x.shape}"
        assert DH == self.head_dim, f"Expected {self.head_dim} head dimension, got {DH}, input shape: {x.shape}"

        in_dtype = x.dtype
        x = x.float()
        x_centered = x - x.mean(dim=-1, keepdim=True)
        x = x_centered * torch.rsqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        x = x.to(in_dtype).reshape(B, S, -1)
        return self.weight * x
