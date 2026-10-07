# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

"""FlashRNN-backed sLSTM layers and configuration helpers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import sqrt
from typing import Literal

import torch
from torch import nn
from xlstm.components.conv import CausalConv1d, CausalConv1dConfig
from xlstm.components.init import small_init_init_

# From original xLSTM
from xlstm.components.linear_headwise import (
    LinearHeadwiseExpand,
    LinearHeadwiseExpandConfig,
)
from xlstm.components.util import ParameterProxy

# From xLSTM Large
from xlstm.xlstm_large.components import MultiHeadLayerNorm

from .xlstm_mixed_config import xLSTMMixedConfig


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
            self.conv1d = CausalConv1d(
                config=CausalConv1dConfig(
                    feature_dim=self.config.embedding_dim,
                    kernel_size=self.config.conv1d_kernel_size,
                )
            )
            self.conv_act_fn = nn.SiLU()

        self.fgate = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config.embedding_dim,
                num_heads=self.config.num_heads,
                bias=False,
            )
        )
        self.igate = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config.embedding_dim,
                num_heads=self.config.num_heads,
                bias=False,
            )
        )
        self.zgate = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config.embedding_dim,
                num_heads=self.config.num_heads,
                bias=False,
            )
        )
        self.ogate = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config.embedding_dim,
                num_heads=self.config.num_heads,
                bias=False,
            )
        )

        self.group_norm = MultiHeadLayerNorm(
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            eps=1e-6,
            use_weight=self.config.group_norm_weight,
            use_bias=False,
            force_float32_reductions=True,
        )
        self.dropout = nn.Dropout(self.config.dropout)

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
        small_init_init_(self.igate.weight, dim=self.config.embedding_dim)
        small_init_init_(self.fgate.weight, dim=self.config.embedding_dim)
        small_init_init_(self.zgate.weight, dim=self.config.embedding_dim)
        small_init_init_(self.ogate.weight, dim=self.config.embedding_dim)

    def step(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        slstm_state: torch.Tensor | None = None,
    ):
        """Perform a single recurrent update."""
        batch_size, _, _ = x.shape

        if self.config.conv1d_kernel_size > 0:
            x_conv, conv_state = self.conv1d.step(x, conv_state=conv_state)
            x_conv = self.conv_act_fn(x_conv)
        else:
            x_conv = x

        f_gate = self.fgate(x_conv)
        i_gate = self.igate(x_conv)
        zgate = self.zgate(x)
        ogate = self.ogate(x)
        gates = (
            f_gate,
            i_gate,
            zgate,
            ogate,
        )
        Wx = torch.stack(gates, dim=2)
        Wx = Wx.reshape(*Wx.shape[:-1], self.config.num_heads, -1)

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
        batch_size, _, _ = x.shape
        if self.config.conv1d_kernel_size > 0:
            x_conv = self.conv1d(x)
            x_conv = self.conv_act_fn(x_conv)
        else:
            x_conv = x

        f_gate = self.fgate(x_conv)
        i_gate = self.igate(x_conv)
        zgate = self.zgate(x)
        ogate = self.ogate(x)
        gates = (
            f_gate,
            i_gate,
            zgate,
            ogate,
        )

        Wx = torch.stack(gates, dim=2)
        Wx = Wx.reshape(*Wx.shape[:-1], self.config.num_heads, -1)
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
        self.recurrent_kernel = ParameterProxy(
            self,
            "_recurrent_kernel",
            self._recurrent_kernel_int2ext,
            self._recurrent_kernel_ext2int,
        )
        self._recurrent_kernel_ = nn.Parameter(self._recurrent_kernel_ext2int(self._recurrent_kernel_.data))

        self._bias_ = nn.Parameter(
            torch.empty(self.config.num_heads, self.config.num_gates_i, self.config.head_dim, dtype=dtype_b)
        )
        self.bias = ParameterProxy(self, "_bias", self._bias_int2ext, self._bias_ext2int)
        self._bias_ = nn.Parameter(self._bias_ext2int(self._bias_.data))

        self.reset_parameters()

    def reset_weights(self):
        """Reset recurrent kernels according to the chosen scheme."""
        if self.config.recurrent_weight_init == "zeros":
            self.recurrent_kernel = nn.init.zeros_(self.recurrent_kernel)
        elif self.config.recurrent_weight_init == "standard":
            for h in range(self.config.num_heads):
                for i, _ in enumerate(["i", "f", "z", "o"]):
                    self.recurrent_kernel[h, :, i, :] = nn.init.uniform_(
                        self.recurrent_kernel[h, :, i, :],
                        -1.0 / sqrt(self.config.hidden_size),
                        1.0 / sqrt(self.config.hidden_size),
                    )

    def reset_bias(self):
        """Reset gate biases with power-law schedule for the forget gate."""
        if self.config.bias_init == "zeros":
            self.bias = nn.init.zeros_(self.bias)
        elif self.config.bias_init == "powerlaw_blockdependent":
            for h in range(self.config.num_heads):
                for i, gate in enumerate(["i", "f", "z", "o"]):
                    if gate == "f":
                        ratio_0_to_1 = self._block_idx / (self._num_blocks - 1) if self._num_blocks > 1 else 0.0
                        init_values = -(
                            -5.0
                            + 12.0
                            * (torch.arange(self.config.head_dim) / (self.config.head_dim - 1))
                            ** (0.3 + 1.3 * ratio_0_to_1)
                        )
                        with torch.no_grad():
                            self.bias[h, i, :] = init_values
                    else:
                        self.bias[h, i] = nn.init.zeros_(self.bias[h, i])

    def reset_parameters(self):
        """Reset projections, recurrent weights, and biases."""
        super().reset_parameters()
        self.reset_weights()
        self.reset_bias()

    def get_R(self):
        """Return the recurrent kernel in external format."""
        return self.recurrent_kernel

    def get_bias(self):
        """Return the bias tensor in external format."""
        return self.bias

    def zero_state(self, batch_dim, input_):
        """Allocate zero states for convolutional and recurrent parts."""
        return torch.zeros(
            (self.config.num_states, batch_dim, 1, self.config.num_heads, self.config.head_dim),
            dtype=input_.dtype,
            device=input_.device,
        )

    @property
    def _recurrent_kernel(self):
        """Internal parameter accessor required by ``ParameterProxy``."""
        return self._recurrent_kernel_

    @property
    def _bias(self):
        """Internal bias accessor required by ``ParameterProxy``."""
        return self._bias_

    @staticmethod
    def _recurrent_kernel_ext2int(recurrent_kernel_ext: torch.Tensor):
        """Convert external recurrent kernel representation for storage."""
        return recurrent_kernel_ext

    @staticmethod
    def _bias_ext2int(bias_ext: torch.Tensor):
        """Convert external bias representation for storage."""
        return bias_ext

    @staticmethod
    def _recurrent_kernel_int2ext(recurrent_kernel_int: torch.Tensor):
        """Convert stored recurrent kernel to the external view."""
        return recurrent_kernel_int

    @staticmethod
    def _bias_int2ext(bias_int: torch.Tensor):
        """Convert stored bias tensor to the external view."""
        return bias_int


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
