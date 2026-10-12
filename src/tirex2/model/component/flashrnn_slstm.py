# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

"""FlashRNN-backed sLSTM layers and configuration helpers."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import sqrt
from typing import Literal

import torch
from torch import nn

from .norm import MultiHeadLayerNorm
from .xlstm_mixed_config import xLSTMMixedConfig


class HeadwiseLinear(nn.Module):
    """Per-head linear projections for all gates at once: [..., H, E] -> [..., G, H, O]."""

    def __init__(self, num_heads: int, head_dim: int, num_gates=4):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_gates, num_heads, head_dim, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...HE,GHEO->...GHO", x, self.weight.transpose(-1, -2))


# Copied from xlstm.components.init.
def small_init_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    """Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution.
    Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
    """
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


@dataclass
class FlashRNNLayerConfig:
    """Configuration for FlashRNN-based sLSTM layers used inside TiRex."""

    embedding_dim: int = -1
    num_heads: int = 4  # this must divide the embedding_dim
    conv1d_kernel_size: int = 0  # 0 means no convolution included
    group_norm_weight: bool = True
    dropout: float = 0.0

    # Cell specific inits
    recurrent_weight_init: str = "standard"
    bias_init: str = "powerlaw_blockdependent"

    # Forwarded to FlashRNNConfig, with FlashRNN's defaults
    backend: str = "cuda_fused"
    function: str = "slstm"
    recurrent_shape: str = "GHDP"
    bias_shape: str = "GHD"
    dtype: str = "bfloat16"
    enable_automatic_mixed_precision: bool = True

    # The sLSTM has 4 gates and 4 states (y, c, n, m)
    num_gates_i: int = field(default=4, init=False)
    num_states: int = field(default=4, init=False)
    _flashrnn_config: object = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        """Validate dimensions and derive head information."""
        assert self.function == "slstm", f"FlashRNNLayerConfig only supports the slstm function, got {self.function!r}"
        assert self.embedding_dim % self.num_heads == 0
        self.hidden_dim = self.embedding_dim
        self.head_dim = self.embedding_dim // self.num_heads

    @property
    def torch_dtype_r(self) -> torch.dtype:
        """Recurrent kernel dtype, as FlashRNN derives it from ``dtype``."""
        return getattr(torch, self.dtype)

    @property
    def torch_dtype_b(self) -> torch.dtype:
        """Bias dtype, as FlashRNN derives it from ``dtype``."""
        return getattr(torch, self.dtype)

    def flashrnn_config(self):
        """Return the equivalent ``FlashRNNConfig``, importing FlashRNN on first use."""
        if self._flashrnn_config is None:
            from flashrnn import FlashRNNConfig

            self._flashrnn_config = FlashRNNConfig(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                backend=self.backend,
                function=self.function,
                recurrent_shape=self.recurrent_shape,
                bias_shape=self.bias_shape,
                dtype=self.dtype,
                enable_automatic_mixed_precision=self.enable_automatic_mixed_precision,
            )
        return self._flashrnn_config


class _FlashRNNLayer(nn.Module, ABC):
    """Abstract base class bridging FlashRNN kernels with TiRex expectations."""

    config_class = FlashRNNLayerConfig

    def __init__(self, config: FlashRNNLayerConfig):
        super().__init__()
        self.config = config

        if self.config.conv1d_kernel_size > 0:
            raise NotImplementedError(
                "The sLSTM layer no longer supports a causal convolution (conv1d_kernel_size > 0)."
            )

        # Input projections of the f, i, z and o gates, in that order
        self.gate_proj = HeadwiseLinear(self.config.num_heads, self.config.head_dim, self.config.num_gates_i)

        self.group_norm = MultiHeadLayerNorm(
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            eps=1e-6,
            use_weight=self.config.group_norm_weight,
            use_bias=False,
            force_float32_reductions=True,
        )
        self.dropout = nn.Dropout(self.config.dropout)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Older checkpoints store one [H, D_out, D_in] weight per gate
        legacy_keys = [f"{prefix}{gate}.weight" for gate in ("fgate", "igate", "zgate", "ogate")]
        if all(key in state_dict for key in legacy_keys):
            state_dict[f"{prefix}gate_proj.weight"] = torch.stack([state_dict.pop(key) for key in legacy_keys])
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @abstractmethod
    def get_R(self):
        """Return the recurrent weight tensor for the FlashRNN kernel."""

    @abstractmethod
    def get_bias(self):
        """Return the gate bias tensor for the FlashRNN kernel."""

    @abstractmethod
    def zero_state(self, batch_dim, input_):
        """Allocate an initial state matching the backend expectations."""

    def recurrence(self, Wx: torch.Tensor, states: torch.Tensor | None = None):
        return self.slstm_flashrnn(Wx, self.get_R(), self.get_bias(), self.config, states)

    def slstm_flashrnn(
        self,
        Wx: torch.Tensor,  # [B, T, G, H, D]
        R: torch.Tensor,  # [H, D_in, G, D_out] ("HPGD")
        b: torch.Tensor,  # [H, G, D]
        config: FlashRNNLayerConfig,
        states: torch.Tensor | None = None,  # [4, B, 1, H, D]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from flashrnn import flashrnn

        return flashrnn(Wx=Wx, R=R, b=b, states=states, config=config.flashrnn_config())

    def get_state(self, init_state, batch_dim, input_):
        """Return ``init_state`` when provided, else allocate zeros."""
        if init_state is not None:
            return init_state
        return self.zero_state(batch_dim, input_)

    def reset_parameters(self):
        """Reset parameters."""
        small_init_init_(self.gate_proj.weight, dim=self.config.embedding_dim)

    def gate_preacts(self, x: torch.Tensor) -> torch.Tensor:
        """Project [B, T, H * D] inputs to the gate pre-activations [B, T, G, H, D]."""
        return self.gate_proj(x.unflatten(-1, (self.config.num_heads, self.config.head_dim)))

    def step(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        slstm_state: torch.Tensor | None = None,
    ):
        """Perform a single recurrent update."""
        batch_size, _, _ = x.shape

        Wx = self.gate_preacts(x)
        y, slstm_state = self.recurrence(Wx, states=self.get_state(slstm_state, batch_size, x))
        y = y[0]

        y = self.dropout(y)

        out = self.group_norm(y)

        return out, {"conv_state": conv_state, "slstm_state": slstm_state}

    def forward(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Process a full sequence through the FlashRNN backend."""
        Wx = self.gate_preacts(x)
        y, _ = self.recurrence(Wx)
        y = y[0]

        y = self.dropout(y)

        out = self.group_norm(y)
        return out


class sLSTMFlashRNNLayer(_FlashRNNLayer):
    """Concrete FlashRNN-driven sLSTM layer with custom parameter initializers."""

    def __init__(self, config: FlashRNNLayerConfig, block_idx: int, num_blocks: int):
        super().__init__(config)
        assert self.config.recurrent_weight_init in ["zeros", "standard"]
        assert self.config.bias_init in ["powerlaw_blockdependent", "zeros", "standard", "small_init"]
        self._block_idx = block_idx
        self._num_blocks = num_blocks

        dtype_r = self.config.torch_dtype_r if not self.config.enable_automatic_mixed_precision else None
        dtype_b = self.config.torch_dtype_b if not self.config.enable_automatic_mixed_precision else None
        self._recurrent_kernel_ = nn.Parameter(
            torch.empty(
                self.config.num_heads,
                self.config.head_dim,
                self.config.num_gates_i,
                self.config.head_dim,
                dtype=dtype_r,
            )
        )
        # Checkpoints also store the recurrent kernel under "recurrent_kernel", so register the same tensor
        # under that name too to keep strict loading working.
        self.recurrent_kernel = self._recurrent_kernel_
        self._bias_ = nn.Parameter(
            torch.empty(self.config.num_heads, self.config.num_gates_i, self.config.head_dim, dtype=dtype_b)
        )

        self.reset_parameters()

    @torch.no_grad()
    def reset_weights(self):
        """Reset recurrent kernels according to the chosen scheme."""
        if self.config.recurrent_weight_init == "zeros":
            nn.init.zeros_(self._recurrent_kernel_)
        elif self.config.recurrent_weight_init == "standard":
            bound = 1.0 / sqrt(self.config.hidden_dim)
            nn.init.uniform_(self._recurrent_kernel_, -bound, bound)

    @torch.no_grad()
    def reset_bias(self):
        """Reset gate biases with power-law schedule for the forget gate."""
        if self.config.bias_init == "zeros":
            nn.init.zeros_(self._bias_)
        elif self.config.bias_init == "powerlaw_blockdependent":
            ratio_0_to_1 = self._block_idx / (self._num_blocks - 1) if self._num_blocks > 1 else 0.0
            positions = torch.arange(self.config.head_dim) / (self.config.head_dim - 1)
            nn.init.zeros_(self._bias_)
            # gate order is i, f, z, o
            self._bias_[:, 1, :] = -(-5.0 + 12.0 * positions ** (0.3 + 1.3 * ratio_0_to_1))

    def reset_parameters(self):
        """Reset projections, recurrent weights, and biases."""
        super().reset_parameters()
        self.reset_weights()
        self.reset_bias()

    def get_R(self):
        """Return the recurrent kernel [H, D_in, G, D_out]."""
        return self._recurrent_kernel_

    def get_bias(self):
        """Return the gate bias [H, G, D]."""
        return self._bias_

    def zero_state(self, batch_dim, input_):
        """Allocate zero states for convolutional and recurrent parts."""
        return torch.zeros(
            (self.config.num_states, batch_dim, 1, self.config.num_heads, self.config.head_dim),
            dtype=input_.dtype,
            device=input_.device,
        )


def _flashrnn_backend(device: Literal["cpu", "cuda", "mps"]) -> str:
    match device:
        case "cpu" | "mps":
            # The "vanilla" backend is pure PyTorch and device-agnostic, so it
            # runs on Apple Metal; FlashRNN's fused "cuda" backend does not.
            return "vanilla"
        case "cuda":
            return "cuda"
        case _:
            raise ValueError(f"device must be 'cpu', 'cuda', or 'mps', got {device!r}.")


def init_cell(config: xLSTMMixedConfig, block_idx: int, num_blocks: int, device: Literal["cpu", "cuda", "mps"]):
    """Instantiate an sLSTM cell for the requested runtime device."""
    return sLSTMFlashRNNLayer(
        FlashRNNLayerConfig(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_slstm_heads,
            conv1d_kernel_size=config.conv1d_kernel_size,  # 0 means no convolution included
            group_norm_weight=True,
            dropout=0,
            recurrent_weight_init="zeros",
            bias_init="powerlaw_blockdependent",
            backend=_flashrnn_backend(device),
            function="slstm",
            recurrent_shape="HPGD",
            bias_shape="HGD",
        ),
        block_idx=block_idx,
        num_blocks=num_blocks,
    )
