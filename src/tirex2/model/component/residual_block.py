"""Residual MLP building blocks used across TiRex modules."""

import torch
from torch import nn

from .layernorm import LayerNorm


@torch.compile
class ResidualBlock(nn.Module):
    """Two-layer MLP with residual projection and optional layer norm."""

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn: nn.Module,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = act_fn
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = LayerNorm(out_dim)

    def forward(self, x: torch.Tensor):
        """Apply hidden MLP, residual projection, and optional layer norm."""
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)

        out = out + res

        if self.use_layer_norm:
            return self.layer_norm(out)
        return out
