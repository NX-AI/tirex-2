# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

"""
mLSTM forward pass in the parallel (quadratic) form in plain PyTorch, used on CPU/MPS and for gradients.
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
