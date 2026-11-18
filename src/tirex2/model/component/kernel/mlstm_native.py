# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

"""
mLSTM forward pass in plain PyTorch: the parallel (quadratic) form, used on CPU/MPS and for gradients, and the
step-by-step recurrent form.
"""

import math

import torch
import torch.nn.functional as F


def mlstm_parallel_native(
    q: torch.Tensor,  # (B, NH, S, DHQK)
    k: torch.Tensor,  # (B, NH, S, DHQK)
    v: torch.Tensor,  # (B, NH, S, DHHV)
    i: torch.Tensor,  # (B, NH, S) input gate preactivation
    f: torch.Tensor,  # (B, NH, S) forget gate preactivation
    eps: float = 1e-6,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:  # H (B, NH, S, DHHV)
    if initial_state is not None:
        return mlstm_parallel_native_state(q, k, v, i, f, eps, initial_state)

    S, DHQK = q.shape[-2:]
    vecB = F.logsigmoid(f).cumsum(-1)
    vecA = i - vecB
    vecM_cum = vecA.cummax(-1).values
    # The qk scale is folded into the exponent; masking before exp keeps the backward free of inf * 0.
    ltr = torch.tril(torch.ones((S, S), dtype=torch.bool, device=q.device))
    matLogD = torch.where(ltr, vecA[..., None, :] - (vecM_cum - math.log(DHQK**-0.5))[..., :, None], -float("inf"))
    matC = (q @ k.transpose(-2, -1)) * torch.exp(matLogD)  # (B, NH, S, S)
    vecN = torch.maximum(matC.sum(-1).abs(), torch.exp(-(vecB + vecM_cum)))
    return (matC / (vecN[..., None] + eps)) @ v, None


def mlstm_parallel_native_state(
    q: torch.Tensor,  # (B, NH, S, DHQK)
    k: torch.Tensor,  # (B, NH, S, DHQK)
    v: torch.Tensor,  # (B, NH, S, DHHV)
    i: torch.Tensor,  # (B, NH, S) input gate preactivation
    f: torch.Tensor,  # (B, NH, S) forget gate preactivation
    eps: float = 1e-6,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:  # H (B, NH, S, DHHV), last (C, n, m)
    S, DHQK = q.shape[-2:]
    qk_scale = DHQK**-0.5
    vecB = F.logsigmoid(f).cumsum(-1)
    vecA = i - vecB
    vecM_cum = vecA.cummax(-1).values
    if initial_state is not None:
        matC0, vecN0, scaM0 = (x.to(q.dtype) for x in initial_state)
        vecM_cum = torch.maximum(vecM_cum, scaM0[..., None])
    ltr = torch.tril(torch.ones((S, S), dtype=torch.bool, device=q.device))
    matLogD = torch.where(ltr, vecA[..., None, :] - (vecM_cum - math.log(qk_scale))[..., :, None], -float("inf"))
    matC = (q @ k.transpose(-2, -1)) * torch.exp(matLogD)  # (B, NH, S, S)
    matNum = matC @ v
    vecDen = matC.sum(-1)
    if initial_state is not None:
        matQ_inter = q * (qk_scale * torch.exp(scaM0[..., None] - vecM_cum))[..., None]
        matNum = matNum + matQ_inter @ matC0
        vecDen = vecDen + (matQ_inter @ vecN0[..., None]).squeeze(-1)
    vecM = vecB + vecM_cum
    matH = matNum / (torch.maximum(vecDen.abs(), torch.exp(-vecM))[..., None] + eps)

    scaM_last = vecM_cum[..., -1]
    matK_gated = k * torch.exp(vecA - scaM_last[..., None])[..., None]
    matC_last = matK_gated.transpose(-2, -1) @ v
    vecN_last = matK_gated.sum(-2)
    if initial_state is not None:
        scaDecay = torch.exp(scaM0 - scaM_last)
        matC_last = matC_last + scaDecay[..., None, None] * matC0
        vecN_last = vecN_last + scaDecay[..., None] * vecN0
    return matH, (matC_last, vecN_last, vecB[..., -1] + scaM_last)


def mlstm_recurrent_native(
    q: torch.Tensor,  # (B, NH, S, DHQK)
    k: torch.Tensor,  # (B, NH, S, DHQK)
    v: torch.Tensor,  # (B, NH, S, DHHV)
    i: torch.Tensor,  # (B, NH, S) input gate preactivation
    f: torch.Tensor,  # (B, NH, S) forget gate preactivation
    eps: float = 1e-6,
    initial_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:  # H (B, NH, S, DHHV), last state
    """Step-by-step stabilized mLSTM recurrence: H, and the last state (C, n, m) if an initial state is given."""
    B, NH, S, DHQK = q.shape
    if initial_state is None:
        C, n, m = q.new_zeros(B, NH, DHQK, v.shape[-1]), q.new_zeros(B, NH, DHQK), q.new_zeros(B, NH)
    else:
        C, n, m = (x.to(q.dtype) for x in initial_state)
    log_f = F.logsigmoid(f)
    hs = []
    for t in range(S):
        m_next = torch.maximum(log_f[..., t] + m, i[..., t])
        decay, gain = torch.exp(log_f[..., t] + m - m_next), torch.exp(i[..., t] - m_next)
        C = decay[..., None, None] * C + gain[..., None, None] * k[..., t, :, None] * v[..., t, None, :]
        n = decay[..., None] * n + gain[..., None] * k[..., t, :]
        m = m_next
        q_t = q[..., t, :] * DHQK**-0.5
        denom = torch.maximum((q_t * n).sum(-1).abs(), torch.exp(-m))
        hs.append((q_t[..., None, :] @ C).squeeze(-2) / (denom[..., None] + eps))
    return torch.stack(hs, dim=-2), None if initial_state is None else (C, n, m)
