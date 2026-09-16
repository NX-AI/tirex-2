# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

"""Loading utilities for inference-ready :class:`TiRex2` checkpoints."""

import warnings
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors

from ._state_dict import expand_shared_duplicates
from .api_adapter import ForecastModel
from .model import TiRex2

CONFIG_FILENAME = "model-config.yaml"
WEIGHTS_FILENAME = "model.safetensors"
#: Legacy torch-pickle weights, superseded by :data:`WEIGHTS_FILENAME`.
CKPT_FILENAME = "model.ckpt"
#: Files fetched from a Hugging Face repo, see :func:`_resolve_ckpt_dir`.
SAFETENSORS_PATTERNS = [CONFIG_FILENAME, WEIGHTS_FILENAME]
LEGACY_PATTERNS = [CONFIG_FILENAME, CKPT_FILENAME]


def _resolve_ckpt_dir(
    ckpt_path: str | Path,
    hf_kwargs: dict[str, Any] | None = None,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Resolve a local checkpoint directory or download one from Hugging Face.

    A repo keeps the legacy ``model.ckpt`` around for older clients long after it
    has gained a ``model.safetensors``, and ``allow_patterns`` is a whitelist
    rather than a preference: asking for both filenames downloads both, so every
    user would transfer the weights twice and read one copy. The safetensors file
    is therefore requested on its own, and the legacy pickle is only fetched when
    the repo turns out not to carry one. The second request costs a metadata
    lookup against an already-populated cache.

    Parameters
    ----------
    ckpt_path : str or pathlib.Path
        Local directory, ``hf://org/repo``, or ``org/repo``.
    hf_kwargs : dict, optional
        Extra keyword arguments for :func:`huggingface_hub.snapshot_download`.
    allow_patterns : list of str, optional
        Fetch exactly these files instead of preferring safetensors over the
        legacy pickle. Used by tooling that specifically wants one format.
    """
    raw_path = str(ckpt_path)
    local_path = Path(raw_path).expanduser()
    if local_path.is_dir():
        return local_path

    if raw_path.startswith("hf://"):
        repo_id = raw_path.removeprefix("hf://")
    elif not local_path.exists() and _looks_like_hf_repo_id(raw_path):
        repo_id = raw_path
    else:
        return local_path

    hf_kwargs = hf_kwargs or {}
    if allow_patterns is not None:
        return Path(snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns, **hf_kwargs))

    ckpt_dir = Path(snapshot_download(repo_id=repo_id, allow_patterns=SAFETENSORS_PATTERNS, **hf_kwargs))
    if (ckpt_dir / WEIGHTS_FILENAME).is_file():
        return ckpt_dir

    return Path(snapshot_download(repo_id=repo_id, allow_patterns=LEGACY_PATTERNS, **hf_kwargs))


def _looks_like_hf_repo_id(path: str) -> bool:
    """Heuristic for Hugging Face repo ids like ``org/model-name``."""
    return not path.startswith((".", "/", "~")) and path.count("/") == 1


def _resolve_weights_file(ckpt_dir: Path) -> Path:
    """Pick the weights file in ``ckpt_dir``, preferring safetensors over the torch pickle."""
    weights_file = ckpt_dir / WEIGHTS_FILENAME
    if weights_file.is_file():
        return weights_file

    legacy_file = ckpt_dir / CKPT_FILENAME
    if legacy_file.is_file():
        return legacy_file

    raise FileNotFoundError(f"Expected model weights at {weights_file} or {legacy_file}")


def _load_state_dict(model: TiRex2, weights_file: Path) -> dict[str, torch.Tensor]:
    """Read a state dict from a safetensors or legacy torch-pickle weights file.

    Tensors that ``model`` shares under several names are stored once (see
    :mod:`tirex2._state_dict`), so the aliases are restored here and the caller
    can keep loading strictly.
    """
    if weights_file.suffix == ".safetensors":
        state_dict = load_safetensors(weights_file, device="cpu")
    else:
        warnings.warn(
            f"Loading weights from the torch-pickle checkpoint {weights_file.name!r}. This format "
            f"relies on Python's pickle and is deprecated; convert the checkpoint directory with "
            f"scripts/convert_checkpoint.py to get a {WEIGHTS_FILENAME!r}, which is loaded in "
            f"preference whenever it is present.",
            FutureWarning,
            stacklevel=3,
        )
        checkpoint = torch.load(weights_file, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    return expand_shared_duplicates(model, state_dict)


def load_model(
    ckpt_path: str | Path = "NX-AI/TiRex-2",
    device: str = "cuda",
    *,
    hf_kwargs: dict[str, Any] | None = None,
    use_flex_attention: bool | None = None,
) -> ForecastModel:
    """Load an inference-ready :class:`TiRex2` from a checkpoint directory or HF repo.

    Parameters
    ----------
    ckpt_path : str or pathlib.Path
        Local directory holding ``model-config.yaml`` plus the weights, either as
        ``model.safetensors`` (preferred) or as the deprecated torch-pickle
        ``model.ckpt``. When both are present the safetensors file wins; loading
        the pickle emits a :class:`FutureWarning`. Values of the form
        ``hf://org/repo`` or ``org/repo`` are treated as Hugging Face model repo
        ids and downloaded with :func:`huggingface_hub.snapshot_download`.
    device : {"cpu", "cuda", "mps"}
        Runtime device and recurrent-kernel family to use. This overrides any
        device/backend stored in the checkpoint config. ``"mps"`` runs on Apple
        Metal using the same pure-PyTorch (native) kernels as ``"cpu"``.
    hf_kwargs : dict, optional
        Extra keyword arguments forwarded to ``snapshot_download`` for Hugging
        Face paths, e.g. ``{"revision": "main", "local_files_only": True}``.
    use_flex_attention : bool, optional
        Override every variate mixer's checkpoint setting. ``True`` enables
        block-sparse FlexAttention, which can reduce the cost of large grouped
        multivariate batches on CUDA but adds first-call compilation overhead.
        ``False`` forces dense attention. Leave as ``None`` to preserve the
        checkpoint configuration and package defaults.

    Returns
    -------
    ForecastModel
        The instantiated backbone (with the checkpoint weights loaded, set to
        evaluation mode) wrapped in a :class:`ForecastModel` that exposes the
        high-level ``forecast`` / ``forecast_gluon`` API.

    Examples
    --------
    >>> import torch
    >>> from tirex2 import TimeseriesType, load_model
    >>> model = load_model("NX-AI/TiRex-2", device="cpu")
    >>> ts = TimeseriesType(target=torch.randn(1, 128), past_covariates=None, future_covariates=None)
    >>> forecast = model.forecast([ts], prediction_length=32, output_type="numpy")[0]
    >>> forecast.shape
    (1, 9, 32)
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Execution on CUDA was requested but is not available.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Execution on MPS was requested but is not available.")

    ckpt_dir = _resolve_ckpt_dir(ckpt_path, hf_kwargs=hf_kwargs)
    config_file = ckpt_dir / CONFIG_FILENAME
    if not config_file.is_file():
        raise FileNotFoundError(f"Expected model config at {config_file}")
    weights_file = _resolve_weights_file(ckpt_dir)

    with config_file.open() as f:
        config: dict[str, Any] = yaml.safe_load(f)

    config["device"] = device
    if use_flex_attention is not None:
        for template in config["stack_config"]["templates"].values():
            template["variate_mixer"]["use_flex_attention"] = use_flex_attention
    model = TiRex2(**config)

    model.load_state_dict(_load_state_dict(model, weights_file), strict=True)

    return ForecastModel(model.eval())
