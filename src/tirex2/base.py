# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

"""Loading utilities for inference-ready :class:`TiRex2` checkpoints."""

import json
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
from .model.component.flashrnn_slstm import _FlashRNNLayer
from .model.component.mlstm_block import mLSTMLayer

CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "model.safetensors"
#: Legacy YAML config, superseded by :data:`CONFIG_FILENAME`.
LEGACY_CONFIG_FILENAME = "model-config.yaml"
#: Legacy torch-pickle weights, superseded by :data:`WEIGHTS_FILENAME`.
CKPT_FILENAME = "model.ckpt"
#: Files fetched from a Hugging Face repo, see :func:`_resolve_ckpt_dir`.
SAFETENSORS_PATTERNS = [CONFIG_FILENAME, WEIGHTS_FILENAME]
LEGACY_PATTERNS = [LEGACY_CONFIG_FILENAME, CKPT_FILENAME]
#: Each current file and the legacy file it supersedes.
_LEGACY_FALLBACKS = {CONFIG_FILENAME: LEGACY_CONFIG_FILENAME, WEIGHTS_FILENAME: CKPT_FILENAME}
#: Where users get the current checkpoint format, named in deprecation warnings.
_DEFAULT_REPO_ID = "NX-AI/TiRex-2"


def _resolve_ckpt_dir(
    ckpt_path: str | Path,
    hf_kwargs: dict[str, Any] | None = None,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Resolve a local checkpoint directory or download one from Hugging Face.

    A repo keeps the legacy ``model-config.yaml`` and ``model.ckpt`` around for
    older clients long after it has gained ``config.json`` and
    ``model.safetensors``, and ``allow_patterns`` is a whitelist rather than a
    preference: asking for both formats downloads both, so every user would
    transfer the weights twice and read one copy. The current files are
    therefore requested on their own, and a legacy file is only fetched when the
    repo turns out not to carry its replacement. The second request costs a
    metadata lookup against an already-populated cache.

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
    missing = [legacy for current, legacy in _LEGACY_FALLBACKS.items() if not (ckpt_dir / current).is_file()]
    if not missing:
        return ckpt_dir

    return Path(snapshot_download(repo_id=repo_id, allow_patterns=missing, **hf_kwargs))


def _looks_like_hf_repo_id(path: str) -> bool:
    """Heuristic for Hugging Face repo ids like ``org/model-name``."""
    return not path.startswith((".", "/", "~")) and path.count("/") == 1


def _resolve_file(ckpt_dir: Path, filename: str, kind: str) -> Path:
    """Pick ``filename`` in ``ckpt_dir``, falling back to the legacy file it supersedes."""
    current_file = ckpt_dir / filename
    if current_file.is_file():
        return current_file

    legacy_file = ckpt_dir / _LEGACY_FALLBACKS[filename]
    if legacy_file.is_file():
        return legacy_file

    raise FileNotFoundError(f"Expected model {kind} at {current_file} or {legacy_file}")


def _warn_legacy_file(legacy_file: Path, replacement: str, reason: str) -> None:
    """Point users at the current file on the Hub rather than at repo-only tooling.

    ``stacklevel`` attributes the warning to the caller of :func:`load_model`.
    """
    warnings.warn(
        f"Loading the legacy {legacy_file.name!r} from {legacy_file.parent}. {reason} This format is "
        f"deprecated and support will be removed in a future release. Download {replacement!r} from "
        f"the Hugging Face model repo instead, e.g. load_model({_DEFAULT_REPO_ID!r}) fetches it "
        f"automatically; it is loaded in preference whenever it is present.",
        FutureWarning,
        stacklevel=4,
    )


def _load_config(config_file: Path) -> dict[str, Any]:
    """Read the model config from ``config.json`` or the legacy ``model-config.yaml``."""
    with config_file.open() as f:
        if config_file.name == CONFIG_FILENAME:
            return json.load(f)
        _warn_legacy_file(config_file, CONFIG_FILENAME, "Hugging Face model repos store their config as JSON.")
        return yaml.safe_load(f)


def _load_state_dict(model: TiRex2, weights_file: Path) -> dict[str, torch.Tensor]:
    """Read a state dict from a safetensors or legacy torch-pickle weights file.

    Tensors that ``model`` shares under several names are stored once (see
    :mod:`tirex2._state_dict`), so the aliases are restored here and the caller
    can keep loading strictly.
    """
    if weights_file.name == WEIGHTS_FILENAME:
        state_dict = load_safetensors(weights_file, device="cpu")
    else:
        _warn_legacy_file(weights_file, WEIGHTS_FILENAME, "It is a torch pickle, which can run arbitrary code.")
        checkpoint = torch.load(weights_file, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    return expand_shared_duplicates(model, state_dict)


def load_model(
    ckpt_path: str | Path = "NX-AI/TiRex-2",
    device: str = "cuda",
    *,
    hf_kwargs: dict[str, Any] | None = None,
    use_flex_attention: bool | None = None,
    compile: bool = False,
) -> ForecastModel:
    """Load an inference-ready :class:`TiRex2` from a checkpoint directory or HF repo.

    Parameters
    ----------
    ckpt_path : str or pathlib.Path
        Hugging Face repo id (``org/repo`` or ``hf://org/repo``), downloaded with
        ``huggingface_hub``, or a local directory containing ``config.json`` and
        ``model.safetensors``. The legacy ``model-config.yaml`` and
        ``model.ckpt`` are still read when their replacement is missing, but are
        deprecated and emit a :class:`FutureWarning`.
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
    compile : bool
        If True, ``torch.compile`` the recurrent layers: mLSTM always, and sLSTM
        only on CPU/MPS.

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
    config_file = _resolve_file(ckpt_dir, CONFIG_FILENAME, "config")
    weights_file = _resolve_file(ckpt_dir, WEIGHTS_FILENAME, "weights")

    config = _load_config(config_file)
    config["device"] = device
    if use_flex_attention is not None:
        for template in config["stack_config"]["templates"].values():
            template["variate_mixer"]["use_flex_attention"] = use_flex_attention

    model = TiRex2(**config)

    model.load_state_dict(_load_state_dict(model, weights_file), strict=True)
    model.eval()
    if compile:
        # slSTM compiled on cpu/mps only.
        targets = (mLSTMLayer,) if device == "cuda" else (mLSTMLayer, _FlashRNNLayer)
        for module in model.modules():
            if isinstance(module, targets):
                module.compile()

    return ForecastModel(model)

