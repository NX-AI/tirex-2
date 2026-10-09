# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

"""Convert a legacy TiRex-2 checkpoint into the Hugging Face file layout.

The legacy layout is a ``model-config.yaml`` plus a ``model.ckpt`` torch
pickle; the current one is a ``config.json`` plus a ``model.safetensors`` file,
which is what :func:`tirex2.base.load_model` reads first. This script is the
one place the current files are produced from the legacy ones.

Tensors that the model shares under several ``state_dict`` names are written
once, because ``safetensors`` refuses to serialize shared storage. Which names
are redundant is derived from the live module graph (see
:mod:`tirex2._state_dict`), not from name patterns, and ``load_model`` restores
them on load.

Examples
--------
Convert a local checkpoint directory in place::

    python scripts/convert_checkpoint.py ./model

Convert a Hugging Face repo into a local directory::

    python scripts/convert_checkpoint.py NX-AI/TiRex-2 --out ./model

Only convert the config, for a directory that already has its safetensors::

    python scripts/convert_checkpoint.py ./model --config-only
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
from torch import nn

from tirex2._state_dict import expand_shared_duplicates, shared_tensor_groups
from tirex2.base import (
    CKPT_FILENAME,
    CONFIG_FILENAME,
    LEGACY_CONFIG_FILENAME,
    LEGACY_PATTERNS,
    WEIGHTS_FILENAME,
    _looks_like_hf_repo_id,
    _resolve_ckpt_dir,
)
from tirex2.model import TiRex2

MANIFEST_FILENAME = f"{WEIGHTS_FILENAME}.manifest.json"


def drop_shared_duplicates(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Reduce every alias group in ``state_dict`` to its single canonical name.

    The surviving name is the alphabetically first one of its group. That
    choice is a convention of the writer only:
    :func:`tirex2._state_dict.expand_shared_duplicates` restores from whichever
    group member a checkpoint happens to carry.

    Parameters
    ----------
    model : torch.nn.Module
        Model defining the alias groups, see
        :func:`tirex2._state_dict.shared_tensor_groups`.
    state_dict : dict[str, torch.Tensor]
        State dict to reduce. It is not modified.

    Returns
    -------
    kept : dict[str, torch.Tensor]
        ``state_dict`` without the redundant names, insertion order preserved.
    dropped : list[str]
        The removed names, sorted.

    Raises
    ------
    ValueError
        If two names of one alias group hold different values. That means the
        state dict was not produced by this model wiring, and dropping either
        name would silently lose information.
    """
    redundant: set[str] = set()
    for group in shared_tensor_groups(model):
        present = [name for name in group if name in state_dict]
        if not present:
            continue
        canonical, *duplicates = present
        for name in duplicates:
            if not torch.equal(state_dict[canonical], state_dict[name]):
                raise ValueError(
                    f"{name!r} and {canonical!r} alias the same tensor in the model but hold "
                    f"different values in the state dict; deduplication would lose information."
                )
        redundant.update(duplicates)

    kept = {name: tensor for name, tensor in state_dict.items() if name not in redundant}
    return kept, sorted(redundant)


def sha256(path: Path) -> str:
    """Hex digest of a file, read in chunks so large checkpoints stay out of memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(tensor: torch.Tensor) -> dict[str, float | int | None]:
    """Finite-only summary statistics, so a tensor holding NaN still reports usable numbers."""
    flat = tensor.detach().cpu().reshape(-1).to(torch.float64)
    finite = flat[torch.isfinite(flat)]
    num_nan = int(flat.isnan().sum())
    if finite.numel() == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "num_nan": num_nan}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(unbiased=True)) if finite.numel() > 1 else 0.0,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "num_nan": num_nan,
    }


def load_legacy_config(config_file: Path) -> dict:
    """Read the legacy YAML config."""
    with config_file.open() as f:
        return yaml.safe_load(f)


def write_config(config: dict, config_file: Path) -> None:
    """Write ``config`` as JSON, refusing anything JSON would not store faithfully.

    YAML is a superset of JSON: it has dates, non-string keys and non-finite
    floats, which ``json`` either rejects or silently rewrites (an integer key
    becomes a string). The written file is therefore read back and compared, so
    ``load_model`` is guaranteed to see exactly the config the YAML described.
    """
    try:
        text = json.dumps(config, indent=2, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"model config cannot be represented as JSON: {exc}") from exc
    if json.loads(text) != config:
        raise ValueError("model config changes when written as JSON, e.g. because of non-string keys")
    config_file.write_text(text)


def load_legacy_state_dict(ckpt_file: Path) -> dict[str, torch.Tensor]:
    """Read the torch-pickle checkpoint, unwrapping a Lightning-style ``state_dict`` entry."""
    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    return {name: tensor.detach().cpu().contiguous() for name, tensor in state_dict.items()}


def verify(model: TiRex2, weights_file: Path, expected: dict[str, torch.Tensor]) -> None:
    """Re-read the written file the way ``load_model`` does and require an exact match."""
    restored = expand_shared_duplicates(model, load_safetensors(weights_file, device="cpu"))
    if restored.keys() != expected.keys():
        only_written = sorted(restored.keys() - expected.keys())
        only_source = sorted(expected.keys() - restored.keys())
        raise ValueError(f"round-trip changed the key set; extra={only_written}, missing={only_source}")
    mismatched = [name for name, tensor in expected.items() if not torch.equal(tensor, restored[name])]
    if mismatched:
        raise ValueError(f"round-trip changed {len(mismatched)} tensor(s), e.g. {mismatched[:5]}")
    model.load_state_dict(restored, strict=True)


def convert(ckpt_path: str, out_dir: Path | None, *, should_verify: bool, config_only: bool = False) -> Path:
    """Write ``config.json``, and unless ``config_only`` also ``model.safetensors`` and its manifest.

    Returns the path of the last file written: the weights, or with
    ``config_only`` the config.
    """
    # The converter reads the legacy files specifically, so ask for them by name
    # rather than taking load_model's preference for the current format.
    legacy_patterns = [LEGACY_CONFIG_FILENAME] if config_only else LEGACY_PATTERNS
    ckpt_dir = _resolve_ckpt_dir(ckpt_path, allow_patterns=legacy_patterns)
    legacy_config_file = ckpt_dir / LEGACY_CONFIG_FILENAME
    ckpt_file = ckpt_dir / CKPT_FILENAME
    for required in [ckpt_dir / name for name in legacy_patterns]:
        if not required.is_file():
            raise FileNotFoundError(f"Expected {required}")

    out_dir = out_dir or ckpt_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    config_file = out_dir / CONFIG_FILENAME
    weights_file = out_dir / WEIGHTS_FILENAME

    config = load_legacy_config(legacy_config_file)
    # The model graph identifies shared tensors and, by constructing at all,
    # validates the config; neither needs the configured device, so stay on CPU.
    model = TiRex2(**{**config, "device": "cpu"})
    write_config(config, config_file)
    print(f"wrote {config_file}")
    if config_only:
        return config_file

    state_dict = load_legacy_state_dict(ckpt_file)
    kept, dropped = drop_shared_duplicates(model, state_dict)

    # Identify the source by content rather than by the local cache path, so the file
    # does not depend on where the checkpoint happened to be downloaded. The tensor
    # payload is then reproducible, but the file as a whole is not byte-stable:
    # safetensors serializes the metadata map in arbitrary order. Compare
    # ``output_sha256`` from the manifest of a given run, not across runs.
    source_sha256 = sha256(ckpt_file)
    metadata = {"format": "pt", "source_sha256": source_sha256}
    if _looks_like_hf_repo_id(ckpt_path):
        metadata["source"] = ckpt_path
    save_safetensors(kept, str(weights_file), metadata=metadata)

    if should_verify:
        verify(model, weights_file, state_dict)

    manifest = {
        "source_checkpoint": str(ckpt_file),
        "source_sha256": source_sha256,
        "output_sha256": sha256(weights_file),
        "num_tensors_in_checkpoint": len(state_dict),
        "num_tensors_written": len(kept),
        "num_parameters_written": sum(tensor.numel() for tensor in kept.values()),
        # Restored on load from the tensor they alias; see tirex2._state_dict.
        "dropped_keys": dropped,
        "tensors": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch."), **stats(tensor)}
            for name, tensor in kept.items()
        },
    }
    (out_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {weights_file}")
    print(
        f"  {len(kept)} tensors, {manifest['num_parameters_written']:,} parameters "
        f"({len(dropped)} shared-tensor aliases written once)"
    )
    print(f"wrote {out_dir / MANIFEST_FILENAME}")
    return weights_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "ckpt_path",
        nargs="?",
        default="NX-AI/TiRex-2",
        help="Checkpoint directory, or a Hugging Face repo id such as NX-AI/TiRex-2 (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory to write into (default: the checkpoint directory).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip re-reading the written file and comparing it against the source checkpoint.",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help=f"Only convert {LEGACY_CONFIG_FILENAME} to {CONFIG_FILENAME}; no {CKPT_FILENAME} is needed.",
    )
    args = parser.parse_args(argv)

    convert(args.ckpt_path, args.out, should_verify=not args.no_verify, config_only=args.config_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
