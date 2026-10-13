# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

"""Feed-forward networks: a plain MLP and the gated FeedForward copied from xlstm.xlstm_large."""

import torch
from torch import nn

from .xlstm_mixed_config import xLSTMLargeConfig


class MLP(nn.Module):
    """Multi-layer perceptron with single hidden layer.

    Parameters
    ----------
    d_model : int
        Input and output dimension
    d_ff : int
        Hidden layer dimension
    dropout : float
        Dropout probability applied after activation
    act_fn : nn.Module, optional
        Activation function (default: nn.ReLU())
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float, act_fn: nn.Module = nn.ReLU()):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout

        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = act_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply MLP transformation.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor of shape [..., d_model]

        Returns
        -------
        torch.Tensor
            Output tensor of shape [..., d_model]
        """
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


def round_up_to_next_multiple_of(x: int, multiple_of: int) -> int:
    """Rounds up x to the next multiple of multiple_of."""
    return int(((x + multiple_of - 1) // multiple_of) * multiple_of)


class FeedForward(nn.Module):
    """SwiGLU feed-forward block of xLSTM Large."""

    def __init__(self, config: xLSTMLargeConfig):
        super().__init__()
        self.config = config

        self.up_proj_dim = round_up_to_next_multiple_of(
            config.embedding_dim * config.ffn_proj_factor,
            config.ffn_round_up_to_multiple_of,
        )

        if self.config.weight_mode == "single":
            self.proj_up_gate = nn.Linear(config.embedding_dim, self.up_proj_dim, bias=config.use_bias)
            self.proj_up = nn.Linear(config.embedding_dim, self.up_proj_dim, bias=config.use_bias)
        elif self.config.weight_mode == "fused":
            self.proj_up_gate_z = nn.Linear(config.embedding_dim, 2 * self.up_proj_dim, bias=config.use_bias)
        else:
            raise ValueError(f"weight_mode must be 'single' or 'fused', got {config.weight_mode!r}.")

        self.proj_down = nn.Linear(self.up_proj_dim, config.embedding_dim, bias=config.use_bias)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.weight_mode == "single":
            x = self.act_fn(self.proj_up_gate(x)) * self.proj_up(x)
        else:
            gate, z = torch.tensor_split(self.proj_up_gate_z(x), (self.up_proj_dim,), dim=-1)
            x = self.act_fn(gate) * z
        return self.proj_down(x)
