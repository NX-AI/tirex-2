# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

"""
mLSTM parallel forward pass with a Triton kernel, used on CUDA. Computes ``mlstm_parallel_native``.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .mlstm_native import mlstm_parallel_native

LOG2E: tl.constexpr = tl.constexpr(1.4426950408889634)


@triton.jit
def _max_fn(a, b):
    return tl.maximum(a, b)


@triton.jit
def _load_gates(vecI, vecF, offs, str_i_s, str_f_s, S):
    """i and log sigmoid(f) at offs; past the sequence end i = -inf (no contribution) and log f = 0 (keep b)."""
    mask = offs < S
    vecI_val = tl.load(vecI + offs * str_i_s, mask=mask, other=-float("inf")).to(tl.float32)
    vecF_val = tl.load(vecF + offs * str_f_s, mask=mask, other=float("inf")).to(tl.float32)
    vecLogF_val = tl.minimum(vecF_val, 0.0) - libdevice.log1p(tl.exp(-tl.abs(vecF_val)))
    return vecI_val, vecLogF_val


@triton.jit
def _parallel_fw_kernel(
    matQ,  # (B, NH, S, DHQK)
    matK,  # (B, NH, S, DHQK)
    matV,  # (B, NH, S, DHHV)
    vecI,  # (B, NH, S)
    vecF,  # (B, NH, S)
    matH,  # (B, S, NH, DHHV) contiguous
    matC0,  # (B, NH, DHQK, DHHV) contiguous float32, read only with HAS_INITIAL_STATE
    vecN0,  # (B, NH, DHQK) contiguous float32
    scaM0,  # (B, NH) contiguous float32
    matC_last,  # (B, NH, DHQK, DHHV) contiguous float32, written only with HAS_INITIAL_STATE
    vecN_last,  # (B, NH, DHQK) contiguous float32
    scaM_last,  # (B, NH) contiguous float32
    str_q_b, str_q_h, str_q_s,
    str_k_b, str_k_h, str_k_s,
    str_v_b, str_v_h, str_v_s,
    str_i_b, str_i_h, str_i_s,
    str_f_b, str_f_h, str_f_s,
    qk_scale,
    S,
    NH: tl.constexpr,
    DHQK: tl.constexpr,
    DHHV: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_DHQK: tl.constexpr,
    BLOCK_DHHV: tl.constexpr,
    DTYPE: tl.constexpr,
    EPS: tl.constexpr,
    HAS_INITIAL_STATE: tl.constexpr,
):  # fmt: skip
    """Computes H for one block of queries and of DHHV; the programs of one sequence and head are adjacent."""
    NB_DHHV: tl.constexpr = (DHHV + BLOCK_DHHV - 1) // BLOCK_DHHV
    num_q_blocks = tl.cdiv(S, BLOCK_Q)
    pid = tl.program_id(0).to(tl.int64)
    idx_b_DHHV = pid % NB_DHHV
    idx_b_Q = (pid // NB_DHHV) % num_q_blocks
    idx_BNH = pid // (NB_DHHV * num_q_blocks)
    idx_b, idx_h = idx_BNH // NH, idx_BNH % NH
    matQ += idx_b * str_q_b + idx_h * str_q_h
    matK += idx_b * str_k_b + idx_h * str_k_h
    matV += idx_b * str_v_b + idx_h * str_v_h
    vecI += idx_b * str_i_b + idx_h * str_i_h
    vecF += idx_b * str_f_b + idx_h * str_f_h

    q_start = idx_b_Q * BLOCK_Q
    offs_q = q_start + tl.arange(0, BLOCK_Q)
    offs_qk = tl.arange(0, BLOCK_DHQK)
    offs_hv = idx_b_DHHV * BLOCK_DHHV + tl.arange(0, BLOCK_DHHV)
    mask_q = offs_q < S
    mask_C = (offs_qk[:, None] < DHQK) & (offs_hv[None, :] < DHHV)
    offs_C = idx_BNH * DHQK * DHHV + offs_qk[:, None] * DHHV + offs_hv[None, :]

    # Stabilizer: b and M = cummax(a) with a = i - b, from the prefix before this query block. O(S) work.
    scaB_prefix = 0.0
    scaA_max = -float("inf")
    if HAS_INITIAL_STATE:
        scaA_max = tl.load(scaM0 + idx_BNH)
    for t_start in range(0, q_start, BLOCK_Q):
        vecI_val, vecLogF_val = _load_gates(vecI, vecF, t_start + tl.arange(0, BLOCK_Q), str_i_s, str_f_s, S)
        scaA_max = tl.maximum(scaA_max, tl.max(vecI_val - (scaB_prefix + tl.cumsum(vecLogF_val, axis=0)), axis=0))
        scaB_prefix += tl.sum(vecLogF_val, axis=0)
    vecI_q, vecLogF_q = _load_gates(vecI, vecF, offs_q, str_i_s, str_f_s, S)
    vecB_q = scaB_prefix + tl.cumsum(vecLogF_q, axis=0)
    vecM_cum_q = tl.maximum(tl.associative_scan(vecI_q - vecB_q, 0, _max_fn), scaA_max)

    matQ_val = tl.load(
        matQ + offs_q[:, None] * str_q_s + offs_qk[None, :], mask=mask_q[:, None] & (offs_qk[None, :] < DHQK), other=0
    ).to(DTYPE)

    matH_num = tl.zeros((BLOCK_Q, BLOCK_DHHV), dtype=tl.float32)
    vecH_den = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    if HAS_INITIAL_STATE:
        # q_t C0 with log weight b_t + m0 - m_t = m0 - M_t
        scaM0_val = tl.load(scaM0 + idx_BNH)
        matC0_val = tl.load(matC0 + offs_C, mask=mask_C, other=0)
        vecN0_val = tl.load(vecN0 + idx_BNH * DHQK + offs_qk, mask=offs_qk < DHQK, other=0)
        matQ_inter = (matQ_val * (qk_scale * tl.exp(scaM0_val - vecM_cum_q))[:, None]).to(DTYPE)
        matH_num = tl.dot(matQ_inter, matC0_val.to(DTYPE))
        vecH_den = tl.sum(matQ_inter.to(tl.float32) * vecN0_val[None, :], axis=1)
    scaB_kv = 0.0
    for kv_start in range(0, tl.minimum(q_start + BLOCK_Q, S), BLOCK_KV):
        offs_kv = kv_start + tl.arange(0, BLOCK_KV)
        vecI_kv, vecLogF_kv = _load_gates(vecI, vecF, offs_kv, str_i_s, str_f_s, S)
        vecA_kv = vecI_kv - (scaB_kv + tl.cumsum(vecLogF_kv, axis=0))
        scaB_kv += tl.sum(vecLogF_kv, axis=0)
        mask_kv = (offs_kv < S)[:, None]
        matK_val = tl.load(
            matK + offs_kv[:, None] * str_k_s + offs_qk[None, :], mask=mask_kv & (offs_qk[None, :] < DHQK), other=0
        )
        matV_val = tl.load(
            matV + offs_kv[:, None] * str_v_s + offs_hv[None, :], mask=mask_kv & (offs_hv[None, :] < DHHV), other=0
        )

        # D = exp(logD - m) = exp(a_j - M_t) <= 1 on and below the diagonal; above it (exponent may be > 0) 0.
        matD_val = tl.where(
            offs_q[:, None] >= offs_kv[None, :], tl.exp2((vecA_kv[None, :] - vecM_cum_q[:, None]) * LOG2E), 0.0
        )
        matC_val = tl.dot(matQ_val, tl.trans(matK_val.to(DTYPE))) * qk_scale * matD_val
        vecH_den += tl.sum(matC_val, axis=1)
        matH_num = tl.dot(matC_val.to(DTYPE), matV_val.to(DTYPE), matH_num)

    vecM_val = vecB_q + vecM_cum_q
    vecN_val = tl.maximum(tl.abs(vecH_den), tl.exp(-vecM_val))
    matH_val = matH_num / (vecN_val[:, None] + EPS)
    matH_ptr = matH + ((idx_b * S + offs_q[:, None]) * NH + idx_h) * DHHV + offs_hv[None, :]
    tl.store(matH_ptr, matH_val.to(matH.dtype.element_ty), mask=mask_q[:, None] & (offs_hv[None, :] < DHHV))

    if HAS_INITIAL_STATE:
        if idx_b_Q == num_q_blocks - 1:
            # The last state weighs step j with exp(a_j - M_S) <= 1, the initial state with exp(m0 - M_S).
            scaM_cum_last = tl.max(tl.where(mask_q, vecM_cum_q, -float("inf")), axis=0)
            scaDecay = tl.exp(scaM0_val - scaM_cum_last)
            matC_state = scaDecay * matC0_val
            vecN_state = scaDecay * vecN0_val
            scaB_t = 0.0
            for state_start in range(0, S, BLOCK_KV):
                offs_t = state_start + tl.arange(0, BLOCK_KV)
                vecI_t, vecLogF_t = _load_gates(vecI, vecF, offs_t, str_i_s, str_f_s, S)
                vecA_t = vecI_t - (scaB_t + tl.cumsum(vecLogF_t, axis=0))
                scaB_t += tl.sum(vecLogF_t, axis=0)
                mask_t = (offs_t < S)[:, None]
                matK_t = tl.load(
                    matK + offs_t[:, None] * str_k_s + offs_qk[None, :],
                    mask=mask_t & (offs_qk[None, :] < DHQK),
                    other=0,
                )
                matV_t = tl.load(
                    matV + offs_t[:, None] * str_v_s + offs_hv[None, :],
                    mask=mask_t & (offs_hv[None, :] < DHHV),
                    other=0,
                )
                matK_gated = (matK_t.to(tl.float32) * tl.exp2((vecA_t - scaM_cum_last) * LOG2E)[:, None]).to(DTYPE)
                matC_state = tl.dot(tl.trans(matK_gated), matV_t.to(DTYPE), matC_state)
                vecN_state += tl.sum(matK_gated.to(tl.float32), axis=0)
            tl.store(matC_last + offs_C, matC_state, mask=mask_C)
            if idx_b_DHHV == 0:
                tl.store(vecN_last + idx_BNH * DHQK + offs_qk, vecN_state, mask=offs_qk < DHQK)
                tl.store(scaM_last + idx_BNH, scaB_t + scaM_cum_last)


_TRITON_DTYPES = {torch.float32: tl.float32, torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}


def _config(B: int, NH: int, S: int, DHHV: int) -> tuple[int, int, int, int]:
    """BLOCK_Q, BLOCK_KV, BLOCK_DHHV, num_warps; tuned on an RTX PRO 6000 Blackwell for DHQK=64, DHHV=128."""
    block_dhhv = min(max(16, triton.next_power_of_2(DHHV)), 128)
    if S <= 16:
        return 16, 16, block_dhhv, 4
    if B * NH * triton.cdiv(S, 32) < torch.cuda.get_device_properties().multi_processor_count:
        # Too few programs to fill the GPU: split DHHV for more of them, and take long key blocks.
        return 32, 64, min(block_dhhv, 64), 4
    if S <= 96:
        return 32, 16, block_dhhv, 2
    return 64, 32, block_dhhv, 4


def mlstm_parallel_triton(
    q: torch.Tensor,  # (B, NH, S, DHQK)
    k: torch.Tensor,  # (B, NH, S, DHQK)
    v: torch.Tensor,  # (B, NH, S, DHHV)
    i: torch.Tensor,  # (B, NH, S) input gate preactivation
    f: torch.Tensor,  # (B, NH, S) forget gate preactivation
    eps: float = 1e-6,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    config: tuple[int, int, int, int] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    """``mlstm_parallel_native`` with a forward-only Triton kernel (CUDA): H, and the last state or None.

    H (B, NH, S, DHHV) is a transposed view of a contiguous (B, S, NH, DHHV). Given an ``initial_state``, the last
    state (C, n, m) is returned as in :func:`mlstm_parallel_native`, in float32. ``config`` overrides the
    (BLOCK_Q, BLOCK_KV, BLOCK_DHHV, num_warps) heuristic.
    """
    # The triton kernel has no backward, so fall back to the native implementation when gradients are needed.
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v, i, f)):
        return mlstm_parallel_native(q, k, v, i, f, eps=eps, initial_state=initial_state)

    # The kernel takes any strides except along the head dimensions.
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    B, NH, S, DHQK = q.shape
    DHHV = v.shape[-1]
    block_q, block_kv, block_dhhv, num_warps = config or _config(B, NH, S, DHHV)

    matH = torch.empty(B, S, NH, DHHV, device=q.device, dtype=q.dtype)
    if initial_state is None:
        last_state = None
        matC0 = vecN0 = scaM0 = matC_last = vecN_last = scaM_last = matH  # never read or written
    else:
        matC0, vecN0, scaM0 = (x.to(torch.float32).contiguous() for x in initial_state)
        matC_last, vecN_last, scaM_last = (torch.empty_like(x) for x in (matC0, vecN0, scaM0))
        last_state = matC_last, vecN_last, scaM_last
    grid = (triton.cdiv(DHHV, block_dhhv) * triton.cdiv(S, block_q) * B * NH,)
    _parallel_fw_kernel[grid](
        q, k, v, i, f, matH, matC0, vecN0, scaM0, matC_last, vecN_last, scaM_last,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *i.stride(), *f.stride(),
        DHQK**-0.5, S, NH=NH, DHQK=DHQK, DHHV=DHHV, BLOCK_Q=block_q, BLOCK_KV=block_kv,
        BLOCK_DHQK=max(16, triton.next_power_of_2(DHQK)), BLOCK_DHHV=block_dhhv,
        DTYPE=_TRITON_DTYPES[q.dtype], EPS=eps, HAS_INITIAL_STATE=initial_state is not None,
        num_warps=num_warps, num_stages=1,
    )  # fmt: skip
    return matH.transpose(1, 2), last_state
