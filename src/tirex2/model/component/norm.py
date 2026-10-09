# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

"""Normalization layers; RMSNorm and MultiHeadLayerNorm are copied from xlstm.xlstm_large.components."""

import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class _NormLayer(nn.Module):
    """Base class with optional learnable weight and bias, applied after normalization."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-6,
        use_weight: bool = True,
        use_bias: bool = False,
        force_float32_reductions: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.force_float32_reductions = force_float32_reductions
        self.weight = nn.Parameter(torch.ones(num_features)) if use_weight else None
        self.bias = nn.Parameter(torch.zeros(num_features)) if use_bias else None

    def _apply_weight_bias(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class RMSNorm(_NormLayer):
    """Root mean square normalization over the last dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        if self.force_float32_reductions:
            x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self._apply_weight_bias(x.to(in_dtype))


class MultiHeadLayerNorm(_NormLayer):
    """Layer norm over the head dimension of a (B, S, NH, DH) input; returns (B, S, NH * DH)."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        use_weight: bool = True,
        use_bias: bool = False,
        force_float32_reductions: bool = True,
    ):
        super().__init__(
            num_features=num_heads * head_dim,
            eps=eps,
            use_weight=use_weight,
            use_bias=use_bias,
            force_float32_reductions=force_float32_reductions,
        )
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, NH, DH = x.shape
        assert NH == self.num_heads, f"Expected {self.num_heads} heads, got {NH}, input shape: {x.shape}"
        assert DH == self.head_dim, f"Expected {self.head_dim} head dimension, got {DH}, input shape: {x.shape}"

        in_dtype = x.dtype
        if self.force_float32_reductions:
            x = x.float()
        x_centered = x - x.mean(dim=-1, keepdim=True)
        x = x_centered * torch.rsqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        x = x.to(in_dtype).reshape(B, S, -1)
        return self._apply_weight_bias(x)
