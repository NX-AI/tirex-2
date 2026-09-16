# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

"""Convert a TiRex-2 ``model.ckpt`` torch pickle into a ``model.safetensors`` file.

This is the one place the legacy pickle format is produced from; everything
downstream reads safetensors. Tensors that the model shares under several
``state_dict`` names are written once, because ``safetensors`` refuses to
serialize shared storage. Which names are redundant is derived from the live
module graph (see :mod:`tirex2._state_dict`), not from name patterns, and
:func:`tirex2.base.load_model` restores them on load.

Examples
--------
Convert a local checkpoint directory in place::

    python scripts/convert_checkpoint.py ./model

Convert a Hugging Face repo into a local directory::

    python scripts/convert_checkpoint.py NX-AI/TiRex-2 --out ./model
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors

from tirex2._state_dict import drop_shared_duplicates, expand_shared_duplicates
from tirex2.base import (
    CKPT_FILENAME,
    CONFIG_FILENAME,
    LEGACY_PATTERNS,
    WEIGHTS_FILENAME,
    _looks_like_hf_repo_id,
    _resolve_ckpt_dir,
)
from tirex2.model import TiRex2

MANIFEST_FILENAME = f"{WEIGHTS_FILENAME}.manifest.json"


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


def convert(ckpt_path: str, out_dir: Path | None, *, should_verify: bool) -> Path:
    """Write ``model.safetensors`` and its manifest, returning the weights path."""
    # The converter reads the legacy pickle specifically, so ask for it by name
    # rather than taking load_model's safetensors-first preference.
    ckpt_dir = _resolve_ckpt_dir(ckpt_path, allow_patterns=LEGACY_PATTERNS)
    config_file = ckpt_dir / CONFIG_FILENAME
    ckpt_file = ckpt_dir / CKPT_FILENAME
    for required in (config_file, ckpt_file):
        if not required.is_file():
            raise FileNotFoundError(f"Expected {required}")

    out_dir = out_dir or ckpt_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_file = out_dir / WEIGHTS_FILENAME

    with config_file.open() as f:
        config = yaml.safe_load(f)
    # The model graph only serves to identify shared tensors, so keep it on CPU
    # regardless of the device the checkpoint was configured for.
    model = TiRex2(**{**config, "device": "cpu"})

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
    args = parser.parse_args(argv)

    convert(args.ckpt_path, args.out, should_verify=not args.no_verify)
    return 0


if __name__ == "__main__":
    sys.exit(main())
