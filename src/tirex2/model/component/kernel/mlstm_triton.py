# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

"""mLSTM chunkwise forward pass with Triton kernels, used on CUDA.

Vendored from https://github.com/NX-AI/mlstm_kernels (``mlstm_kernels==2.0.6``), reduced to the forward
pass of the ``chunkwise--triton_limit_chunk`` kernel used by ``mLSTMBackend`` in ``train_with_padding``
mode without initial or last states. The computations and their order are kept as upstream, so the
results are bitwise identical. See ``mlstm_native.py`` for the notation.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .mlstm_native import mlstm_chunkwise_native, pad_to_chunks


@triton.jit
def _recurrent_fw_C_kernel(
    matK,  # (B, NH, S, DHQK)
    matV,  # (B, NH, S, DHHV)
    vecB,  # (B, NH, NC, L)
    vecI,  # (B, NH, NC, L)
    matC_states,  # (B, NH, NC * DHQK, DHHV)
    vecN_states,  # (B, NH, NC * DHQK)
    scaM_states,  # (B, NH, NC)
    S: tl.constexpr,
    DHQK: tl.constexpr,
    DHHV: tl.constexpr,
    NC: tl.constexpr,
    L: tl.constexpr,
    siz_b_DHQK: tl.constexpr,
    siz_b_DHHV: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Computes the states (C, n, m) entering each chunk."""
    idx_b_DHQK, idx_b_DHHV, idx_b_BNH = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    matC_k_val = tl.zeros((siz_b_DHQK, siz_b_DHHV), dtype=tl.float32)
    vecN_k_val = tl.zeros((siz_b_DHQK,), dtype=tl.float32)
    scaM_k_val = 0.0

    for k in range(NC):
        matC_k_ptr = tl.make_block_ptr(
            base=matC_states + (idx_b_BNH * NC + k) * DHQK * DHHV,
            shape=(DHQK, DHHV),
            strides=(DHHV, 1),
            offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
            block_shape=(siz_b_DHQK, siz_b_DHHV),
            order=(1, 0),
        )
        tl.store(matC_k_ptr, matC_k_val, boundary_check=(0, 1))
        if idx_b_DHHV == 0:
            vecN_k_ptr = vecN_states + (idx_b_BNH * NC + k) * DHQK + idx_b_DHQK * siz_b_DHQK + tl.arange(0, siz_b_DHQK)
            tl.store(vecN_k_ptr, vecN_k_val)
        if (idx_b_DHQK == 0) and (idx_b_DHHV == 0):
            tl.store(scaM_states + idx_b_BNH * NC + k, scaM_k_val.to(tl.float32))

        vecBI_offset = idx_b_BNH * S + k * L
        vecB_last_k_val = tl.load(vecB + vecBI_offset + (L - 1)).to(tl.float32)
        vecB_k_val = tl.load(vecB + vecBI_offset + tl.arange(0, L)).to(tl.float32)
        vecI_k_val = tl.load(vecI + vecBI_offset + tl.arange(0, L)).to(tl.float32)

        vecA_k_val = (vecB_last_k_val - vecB_k_val) + vecI_k_val
        scaG_k_val = vecB_last_k_val
        scaM_next_val = tl.maximum(scaG_k_val + scaM_k_val, tl.max(vecA_k_val))

        # load matK transposed (siz_b_DHQK, L) and matV (L, siz_b_DHHV)
        matK_k_ptr = tl.make_block_ptr(
            base=matK + idx_b_BNH * S * DHQK,
            shape=(DHQK, S),
            strides=(1, DHQK),
            offsets=(idx_b_DHQK * siz_b_DHQK, k * L),
            block_shape=(siz_b_DHQK, L),
            order=(0, 1),
        )
        matV_k_ptr = tl.make_block_ptr(
            base=matV + idx_b_BNH * S * DHHV,
            shape=(S, DHHV),
            strides=(DHHV, 1),
            offsets=(k * L, idx_b_DHHV * siz_b_DHHV),
            block_shape=(L, siz_b_DHHV),
            order=(1, 0),
        )
        matK_k_val = tl.load(matK_k_ptr, boundary_check=(0, 1)).to(tl.float32)
        matV_k_val = tl.load(matV_k_ptr, boundary_check=(0, 1)).to(DTYPE)

        vecAbar_k_val = tl.exp(vecA_k_val - scaM_next_val)
        scaGbar_k_val = tl.exp(scaG_k_val + scaM_k_val - scaM_next_val)
        matKbar_k_val = (matK_k_val * vecAbar_k_val[None, :]).to(DTYPE)

        matC_k_val = scaGbar_k_val * matC_k_val + tl.dot(matKbar_k_val, matV_k_val)
        vecN_k_val = scaGbar_k_val * vecN_k_val + tl.sum(matKbar_k_val.to(tl.float32), axis=1)
        scaM_k_val = scaM_next_val


@triton.jit
def _parallel_fw_H_kernel(
    matQ,  # (B, NH, S, DHQK)
    matK,  # (B, NH, S, DHQK)
    matV,  # (B, NH, S, DHHV)
    matC_states,  # (B, NH, NC * DHQK, DHHV)
    vecN_states,  # (B, NH, NC * DHQK)
    scaM_states,  # (B, NH, NC)
    vecI,  # (B, NH, NC, L)
    vecB,  # (B, NH, NC, L)
    matH,  # (B, NH, S, DHHV)
    qk_scale,
    S: tl.constexpr,
    DHQK: tl.constexpr,
    DHHV: tl.constexpr,
    NC: tl.constexpr,
    L: tl.constexpr,
    siz_b_DHQK: tl.constexpr,
    siz_b_DHHV: tl.constexpr,
    DTYPE: tl.constexpr,
    EPS: tl.constexpr,
):
    """Computes the outputs H of one chunk from the chunk's inputs and the state entering it."""
    idx_b_DHHV, idx_b_NC, idx_b_BNH = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    vecBI_offset = idx_b_BNH * S + idx_b_NC * L
    vecB_val = tl.load(vecB + vecBI_offset + tl.arange(0, L)).to(tl.float32)
    vecI_val = tl.load(vecI + vecBI_offset + tl.arange(0, L)).to(tl.float32)
    scaM_km1_val = tl.load(scaM_states + idx_b_BNH * NC + idx_b_NC).to(tl.float32)

    # gate matrix matDbar (L, L) and stabilizer vecM_combine (L,)
    idx_mask = tl.arange(0, L)
    mask = idx_mask[:, None] >= idx_mask[None, :]
    matD_val = tl.where(mask, vecB_val[:, None] - vecB_val[None, :] + vecI_val[None, :], -float("inf"))
    vecM_combine_val = tl.maximum(vecB_val + scaM_km1_val, tl.max(matD_val, axis=1))
    matDbar_val = tl.exp(matD_val - vecM_combine_val[:, None])
    vecBbar_val = tl.exp(vecB_val + scaM_km1_val - vecM_combine_val)

    matS_val = tl.zeros((L, L), dtype=tl.float32)
    matH_inter_val = tl.zeros((L, siz_b_DHHV), dtype=tl.float32)
    vecH_inter_denom_val = tl.zeros((L,), dtype=tl.float32)
    for idx_b_DHQK in range(tl.cdiv(DHQK, siz_b_DHQK)):
        matQ_ptr = tl.make_block_ptr(
            base=matQ + idx_b_BNH * S * DHQK,
            shape=(S, DHQK),
            strides=(DHQK, 1),
            offsets=(idx_b_NC * L, idx_b_DHQK * siz_b_DHQK),
            block_shape=(L, siz_b_DHQK),
            order=(1, 0),
        )
        matK_ptr = tl.make_block_ptr(
            base=matK + idx_b_BNH * S * DHQK,
            shape=(DHQK, S),
            strides=(1, DHQK),
            offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_NC * L),
            block_shape=(siz_b_DHQK, L),
            order=(0, 1),
        )
        matC_km1_ptr = tl.make_block_ptr(
            base=matC_states + (idx_b_BNH * NC + idx_b_NC) * DHQK * DHHV,
            shape=(DHQK, DHHV),
            strides=(DHHV, 1),
            offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
            block_shape=(siz_b_DHQK, siz_b_DHHV),
            order=(1, 0),
        )
        vecN_km1_ptr = (
            vecN_states + (idx_b_BNH * NC + idx_b_NC) * DHQK + idx_b_DHQK * siz_b_DHQK + tl.arange(0, siz_b_DHQK)
        )

        matQ_val = tl.load(matQ_ptr, boundary_check=(0, 1)).to(DTYPE)
        matK_val = tl.load(matK_ptr, boundary_check=(0, 1)).to(DTYPE)
        matS_val += (tl.dot(matQ_val, matK_val) * qk_scale).to(DTYPE)

        matQbar_val = (matQ_val * vecBbar_val[:, None] * qk_scale).to(DTYPE)
        matC_km1_val = tl.load(matC_km1_ptr, boundary_check=(0, 1)).to(DTYPE)
        matH_inter_val += tl.dot(matQbar_val, matC_km1_val)
        vecN_km1_val = tl.load(vecN_km1_ptr).to(DTYPE)
        vecH_inter_denom_val += tl.sum((matQbar_val * vecN_km1_val[None, :]).to(tl.float32), axis=1)

    matSbar_val = (matS_val * matDbar_val).to(DTYPE)
    matV_ptr = tl.make_block_ptr(
        base=matV + idx_b_BNH * S * DHHV,
        shape=(S, DHHV),
        strides=(DHHV, 1),
        offsets=(idx_b_NC * L, idx_b_DHHV * siz_b_DHHV),
        block_shape=(L, siz_b_DHHV),
        order=(1, 0),
    )
    matV_val = tl.load(matV_ptr, boundary_check=(0, 1)).to(DTYPE)
    matH_intra_val = tl.dot(matSbar_val, matV_val)
    vecH_intra_denom_val = tl.sum(matSbar_val.to(tl.float32), axis=1)

    matH_num_val = matH_inter_val + matH_intra_val
    vecH_denom_val = tl.maximum(tl.abs(vecH_inter_denom_val + vecH_intra_denom_val), tl.exp(-vecM_combine_val))
    matH_val = matH_num_val / (vecH_denom_val[:, None] + EPS)

    matH_ptr = tl.make_block_ptr(
        base=matH + idx_b_BNH * S * DHHV,
        shape=(S, DHHV),
        strides=(DHHV, 1),
        offsets=(idx_b_NC * L, idx_b_DHHV * siz_b_DHHV),
        block_shape=(L, siz_b_DHHV),
        order=(1, 0),
    )
    tl.store(matH_ptr, matH_val.to(DTYPE), boundary_check=(0, 1))


_TRITON_DTYPES = {torch.float32: tl.float32, torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}


def mlstm_chunkwise_triton(
    q: torch.Tensor,  # (B, NH, S, DHQK)
    k: torch.Tensor,  # (B, NH, S, DHQK)
    v: torch.Tensor,  # (B, NH, S, DHHV)
    i: torch.Tensor,  # (B, NH, S) input gate preactivation
    f: torch.Tensor,  # (B, NH, S) forget gate preactivation
    chunk_size: int = 64,
    eps: float = 1e-6,
    autocast_kernel_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:  # (B, NH, S, DHHV)
    """mLSTM over a full sequence from zero states, with forward-only Triton kernels (CUDA)."""
    # The triton kernels have no backward, so fall back to the native implementation when gradients are needed.
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v, i, f)):
        return mlstm_chunkwise_native(q, k, v, i, f, chunk_size=chunk_size, eps=eps)

    S = q.shape[2]
    q, k, v, i, f = pad_to_chunks(q, k, v, i, f, chunk_size)
    # Under autocast, the inputs are cast and the kernel runs without autocast (upstream's custom_fwd).
    if torch.is_autocast_enabled("cuda"):
        q, k, v, i, f = (x.to(autocast_kernel_dtype) for x in (q, k, v, i, f))
    with torch.autocast("cuda", enabled=False):
        q, k, v, i, f = (x.contiguous() for x in (q, k, v, i, f))
        B, NH, S_pad, DHQK = q.shape
        DHHV = v.shape[-1]
        NC, L = S_pad // chunk_size, chunk_size
        assert L & (L - 1) == 0, "Chunk size must be a power of 2."

        vecI = i.reshape(B, NH, NC, L)
        vecB = F.logsigmoid(f.reshape(B, NH, NC, L).to(torch.float32)).cumsum(-1)

        siz_b_DHQK = min(64, triton.next_power_of_2(DHQK))
        siz_b_DHHV = min(64, triton.next_power_of_2(DHHV))
        num_warps = 4 if siz_b_DHQK == 64 else 2
        dims = dict(S=S_pad, DHQK=DHQK, DHHV=DHHV, NC=NC, L=L, siz_b_DHQK=siz_b_DHQK, siz_b_DHHV=siz_b_DHHV)
        DTYPE = _TRITON_DTYPES[q.dtype]

        matC_states = torch.empty(B, NH, NC * DHQK, DHHV, device=q.device, dtype=torch.float32)
        vecN_states = torch.empty(B, NH, NC * DHQK, device=q.device, dtype=torch.float32)
        scaM_states = torch.empty(B, NH, NC, device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(DHQK, siz_b_DHQK), triton.cdiv(DHHV, siz_b_DHHV), B * NH)
        _recurrent_fw_C_kernel[grid](
            k, v, vecB, vecI, matC_states, vecN_states, scaM_states,
            **dims, DTYPE=DTYPE, num_stages=1, num_warps=num_warps,
        )  # fmt: skip

        matH = torch.empty(B, NH, S_pad, DHHV, device=q.device, dtype=q.dtype)
        grid = (triton.cdiv(DHHV, siz_b_DHHV), NC, B * NH)
        _parallel_fw_H_kernel[grid](
            q, k, v, matC_states, vecN_states, scaM_states, vecI, vecB, matH, DHQK**-0.5,
            **dims, DTYPE=DTYPE, EPS=eps, num_stages=1, num_warps=num_warps,
        )  # fmt: skip
    return matH[:, :, :S, :]
