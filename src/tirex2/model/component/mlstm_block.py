# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

"""mLSTM layer copied from xLSTM Large, with TiRex-specific tweaks."""

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
from mlstm_kernels.torch.backend_module import mLSTMBackend, mLSTMBackendConfig

from .norm import MultiHeadLayerNorm
from .xlstm_mixed_config import xLSTMMixedConfig

# soft_cap, bias_linspace_init_ and the mLSTMLayerConfig fields are copied from xlstm.


def soft_cap(values: torch.Tensor, cap_value: float | torch.Tensor | None) -> torch.Tensor:
    """Soft caps a tensor to cap_value with a scaled tanh; no cap when cap_value is None."""
    if cap_value is None:
        return values
    return cap_value * torch.tanh(values / cap_value)


def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0) -> torch.Tensor:
    """Linearly spaced bias init across dimensions."""
    assert param.dim() == 1, f"param must be 1-dimensional (typically a bias), got {param.dim()}"
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)
    with torch.no_grad():
        param.copy_(init_vals)
    return param


@dataclass
class conv_mLSTMLayerConfig:
    """Configuration of the xLSTM Large mLSTM layer, plus TiRex's convolution and RoPE controls."""

    embedding_dim: int
    num_heads: int
    use_bias: bool = False
    norm_eps: float = 1e-6
    norm_reduction_force_float32: bool = True
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    gate_soft_cap: float = 15.0
    mlstm_backend: mLSTMBackendConfig = field(default_factory=mLSTMBackendConfig)
    weight_mode: str = "single"

    conv1d_kernel_size: int = 0
    conv1d_channel_mixing: bool = False
    use_rope: bool = True


class mLSTMLayer(nn.Module):
    """mLSTM implementation copied from xLSTM 7B."""

    def __init__(self, config: conv_mLSTMLayerConfig):
        super().__init__()
        self.config = config

        self.v_dim = int(config.embedding_dim * config.v_dim_factor)
        self.qk_dim = int(config.embedding_dim * config.qk_dim_factor)

        # Fused input projection of q, k, v and the o, i, f gate pre-activations, in that order.
        # The i and f gates always have a bias; without use_bias the rest of the bias starts at zero.
        self.in_proj_split = (
            self.qk_dim,
            self.qk_dim,
            self.v_dim,
            self.v_dim,
            self.config.num_heads,
            self.config.num_heads,
        )
        self.in_proj = nn.Linear(self.config.embedding_dim, sum(self.in_proj_split), bias=True)
        if not self.config.use_bias:
            nn.init.zeros_(self.in_proj.bias)

        if self.config.conv1d_kernel_size > 0:
            raise NotImplementedError(
                "The mLSTM layer no longer supports a causal convolution (conv1d_kernel_size > 0)."
            )

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

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Older checkpoints store a separate linear layer per projection
        legacy = ("q", "k", "v", "ogate_preact", "igate_preact", "fgate_preact")
        if all(f"{prefix}{name}.weight" in state_dict for name in legacy):
            weights = [state_dict.pop(f"{prefix}{name}.weight") for name in legacy]
            biases = [state_dict.pop(f"{prefix}{name}.bias", None) for name in legacy]
            biases = [w.new_zeros(w.shape[0]) if b is None else b for w, b in zip(weights, biases)]
            state_dict[f"{prefix}in_proj.weight"] = torch.cat(weights)
            state_dict[f"{prefix}in_proj.bias"] = torch.cat(biases)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a full sequence through the mlstm."""
        assert x.ndim == 3, f"Input must have shape [B, S, D], got {x.shape}"
        B, S, _ = x.shape

        q, k, v, o_preact, i_preact, f_preact = self.in_proj(x).split(self.in_proj_split, dim=-1)
        i_preact = soft_cap(i_preact, cap_value=self.config.gate_soft_cap)
        f_preact = soft_cap(f_preact, cap_value=self.config.gate_soft_cap)

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


def _mlstm_backend_config(config: xLSTMMixedConfig, device: Literal["cpu", "cuda", "mps"]) -> mLSTMBackendConfig:
    """Return the mLSTM kernel backend matching the requested runtime device.

    ``"mps"`` shares the ``"cpu"`` configuration: the pure-PyTorch native kernels
    are device-agnostic and run on Apple Metal, whereas the Triton kernels used by
    ``"cuda"`` are unavailable there.
    """
    if device in ("cpu", "mps"):
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

    raise ValueError(f"device must be 'cpu', 'cuda', or 'mps', got {device!r}.")


def init_cell(config: xLSTMMixedConfig, device: Literal["cpu", "cuda", "mps"]) -> mLSTMLayer:
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
    # Match mLSTMBlock.reset_parameters gate initialisation; i and f are the last rows of in_proj.
    num_heads = layer.config.num_heads
    i_rows = slice(-2 * num_heads, -num_heads)
    f_rows = slice(-num_heads, None)
    with torch.no_grad():
        torch.nn.init.zeros_(layer.in_proj.weight[f_rows])
        bias_linspace_init_(layer.in_proj.bias[f_rows], start=3.0, end=6.0)
        torch.nn.init.zeros_(layer.in_proj.weight[i_rows])
        torch.nn.init.normal_(layer.in_proj.bias[i_rows], mean=0.0, std=0.1)
    return layer
