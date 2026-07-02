"""Wrapper around xlstm mLSTM blocks with TiRex-specific tweaks."""

# From xlstm large
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from xlstm.components.conv import CausalConv1d, CausalConv1dConfig
from xlstm.components.init import bias_linspace_init_
from xlstm.xlstm_large.model import (
    MultiHeadLayerNorm,
    mLSTMBackend,
    mLSTMBackendConfig,
    mLSTMLayerConfig,
    soft_cap,
)

from .xlstm_mixed_config import xLSTMMixedConfig


@dataclass
class conv_mLSTMLayerConfig(mLSTMLayerConfig):
    """Extends mlstm layer config with controls for causal convolution."""

    conv1d_kernel_size: int = 0
    conv1d_channel_mixing: bool = False
    use_rope: bool = True


class mLSTMLayer(nn.Module):
    """mLSTM implementation copied from xLSTM 7B, adapted to use convolution."""

    def __init__(self, config: conv_mLSTMLayerConfig):
        super().__init__()
        self.config = config

        self.v_dim = int(config.embedding_dim * config.v_dim_factor)
        self.qk_dim = int(config.embedding_dim * config.qk_dim_factor)

        self.q = nn.Linear(
            in_features=self.config.embedding_dim,
            out_features=self.qk_dim,
            bias=self.config.use_bias,
        )
        self.k = nn.Linear(
            in_features=self.config.embedding_dim,
            out_features=self.qk_dim,
            bias=self.config.use_bias,
        )
        self.v = nn.Linear(
            in_features=self.config.embedding_dim,
            out_features=self.v_dim,
            bias=self.config.use_bias,
        )

        self.ogate_preact = nn.Linear(
            in_features=self.config.embedding_dim,
            out_features=self.v_dim,
            bias=self.config.use_bias,
        )
        self.igate_preact = nn.Linear(
            in_features=self.config.embedding_dim,
            out_features=self.config.num_heads,
            bias=True,
        )
        self.fgate_preact = nn.Linear(
            in_features=self.config.embedding_dim,
            out_features=self.config.num_heads,
            bias=True,
        )

        if self.config.conv1d_kernel_size > 0:
            self.conv1d = CausalConv1d(
                config=CausalConv1dConfig(
                    feature_dim=self.config.embedding_dim,
                    kernel_size=self.config.conv1d_kernel_size,
                    channel_mixing=self.config.conv1d_channel_mixing,
                )
            )
            self.conv_act_fn = nn.SiLU()

        self.ogate_act_fn = nn.Sigmoid()
        self.mlstm_backend = mLSTMBackend(config=self.config.mlstm_backend)

        self.multihead_norm = MultiHeadLayerNorm(
            num_heads=self.config.num_heads,
            head_dim=self.v_dim // self.config.num_heads,
            eps=self.config.norm_eps,
            use_weight=True,
            use_bias=self.config.use_bias,
            force_float32_reductions=self.config.norm_reduction_force_float32,
        )
        self.out_proj = nn.Linear(
            in_features=self.v_dim,
            out_features=self.config.embedding_dim,
            bias=self.config.use_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a full sequence through the mlstm."""
        assert x.ndim == 3, f"Input must have shape [B, S, D], got {x.shape}"
        B, S, _ = x.shape

        if self.config.conv1d_kernel_size > 0:
            x_conv = self.conv1d(x)
            x_conv = self.conv_act_fn(x_conv)
        else:
            x_conv = x

        q = self.q(x_conv)
        k = self.k(x_conv)
        v = self.v(x)
        o_preact = self.ogate_preact(x)
        i_preact = soft_cap(self.igate_preact(x), cap_value=self.config.gate_soft_cap)
        f_preact = soft_cap(self.fgate_preact(x), cap_value=self.config.gate_soft_cap)

        q = q.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)
        k = k.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)
        v = v.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)
        i_preact = i_preact.transpose(1, 2)
        f_preact = f_preact.transpose(1, 2)
        h = self.mlstm_backend(
            q=q,
            k=k,
            v=v,
            i=i_preact,
            f=f_preact,
        )
        expected_h_shape = (
            B,
            self.config.num_heads,
            S,
            self.v_dim // self.config.num_heads,
        )
        assert h.shape == expected_h_shape, f"Got {h.shape}, expected {expected_h_shape}"

        h = h.transpose(1, 2)
        h_norm = self.multihead_norm(h)
        h_norm = h_norm.reshape(B, S, -1)

        h_out = self.ogate_act_fn(o_preact) * h_norm

        y = self.out_proj(h_out)
        return y


def _mlstm_backend_config(config: xLSTMMixedConfig, device: Literal["cpu", "cuda"]) -> mLSTMBackendConfig:
    """Return the mLSTM kernel backend matching the requested runtime device."""
    if device == "cpu":
        return mLSTMBackendConfig(
            chunkwise_kernel="chunkwise--native_autograd",
            sequence_kernel="native_sequence__native",
            step_kernel="native",
            mode=config.mode,
            chunk_size=config.chunk_size,
            return_last_states=config.return_last_states,
            autocast_kernel_dtype="float32",
            eps=config.eps,
            inference_state_dtype=config.inference_state_dtype,
        )

    if device == "cuda":
        return mLSTMBackendConfig(
            chunkwise_kernel="chunkwise--triton_limit_chunk",
            sequence_kernel="native_sequence__triton",
            step_kernel="triton",
            mode=config.mode,
            chunk_size=config.chunk_size,
            return_last_states=config.return_last_states,
            autocast_kernel_dtype="bfloat16",
            eps=config.eps,
            inference_state_dtype="float32",
        )

    raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}.")


def init_cell(config: xLSTMMixedConfig, device: Literal["cpu", "cuda"]) -> mLSTMLayer:
    """Instantiate an mLSTM cell for the requested runtime device."""
    layer = mLSTMLayer(
        conv_mLSTMLayerConfig(
            conv1d_kernel_size=config.conv1d_kernel_size,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            use_bias=config.use_bias,
            norm_eps=config.norm_eps,
            norm_reduction_force_float32=config.norm_reduction_force_float32,
            qk_dim_factor=config.qk_dim_factor,
            v_dim_factor=config.v_dim_factor,
            gate_soft_cap=config.gate_soft_cap,
            weight_mode=config.weight_mode,
            use_rope=config.use_rope,
            mlstm_backend=_mlstm_backend_config(config, device),
        )
    )
    # Match mLSTMBlock.reset_parameters gate initialisation.
    torch.nn.init.zeros_(layer.fgate_preact.weight)
    bias_linspace_init_(layer.fgate_preact.bias, start=3.0, end=6.0)
    torch.nn.init.zeros_(layer.igate_preact.weight)
    torch.nn.init.normal_(layer.igate_preact.bias, mean=0.0, std=0.1)
    return layer
