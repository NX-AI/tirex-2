"""Assemble batches of :class:`TimeseriesType` from plain tensor/array/list inputs."""

import itertools
from collections.abc import Iterable, Iterator
from typing import Union

import numpy as np
import torch

from ..model.types import TimeseriesType

ContextType = Union[
    torch.Tensor,
    np.ndarray,
    list[torch.Tensor],
    list[np.ndarray],
]
CovariateType = Union[ContextType, list[None], None]


def _ensure_2d_tensor(sample) -> torch.Tensor:
    """Coerce a sample into a floating ``[num_variates, T]`` tensor (1D is treated as univariate)."""
    tensor = sample if isinstance(sample, torch.Tensor) else torch.as_tensor(sample)
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    assert tensor.ndim == 2, "Each sample must be 1D [T] or 2D [num_variates, T]"
    return tensor


def _normalize_target(target: ContextType) -> list:
    """Split the target into an ordered list of per-sample series (2D arrays become per-row samples)."""
    if isinstance(target, torch.Tensor):
        if target.ndim == 1:
            return [target]
        assert target.ndim == 2, "A tensor target must be 1D [T] or 2D [batch, T]"
        return list(target)
    if isinstance(target, np.ndarray):
        if target.ndim == 1:
            return [target]
        assert target.ndim == 2, "An array target must be 1D [T] or 2D [batch, T]"
        return list(target)
    if isinstance(target, (list, tuple, Iterable)):
        return list(target)
    raise ValueError(f"Target type {type(target)} not supported! Supported types: {ContextType}")


def _normalize_covariates(covariates: CovariateType, num_samples: int) -> list:
    """Align covariates to one (optionally ``None``) ``[V, T]`` tensor per target sample."""
    if covariates is None:
        return [None] * num_samples
    if isinstance(covariates, (list, tuple)):
        assert len(covariates) == num_samples, "Covariates must provide one entry per target sample"
        return [None if c is None else _ensure_2d_tensor(c) for c in covariates]
    # A bare array/tensor is only unambiguous for a single target sample.
    assert num_samples == 1, "Pass covariates as a list parallel to the target samples"
    return [_ensure_2d_tensor(covariates)]


def build_timeseries(
    target: ContextType,
    past_covariates: CovariateType = None,
    future_covariates: CovariateType = None,
) -> list[TimeseriesType]:
    """Zip a target with its optional covariates into a list of :class:`TimeseriesType`."""
    samples = _normalize_target(target)
    num_samples = len(samples)
    past = _normalize_covariates(past_covariates, num_samples)
    future = _normalize_covariates(future_covariates, num_samples)

    return [
        TimeseriesType(target=_ensure_2d_tensor(t), past_covariates=pc, future_covariates=fc)
        for t, pc, fc in zip(samples, past, future)
    ]


def _batched(iterable: Iterable, n: int) -> Iterator[list]:
    """Yield successive lists of at most ``n`` items from ``iterable``."""
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch
