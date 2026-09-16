"""Tiny child pytest module used by the CUDA requirement contract tests."""

import os
from pathlib import Path


def _touch_from_env(name: str) -> None:
    if path := os.environ.get(name):
        Path(path).write_text("1", encoding="utf-8")


_touch_from_env("TIREX_CUDA_PROBE_COLLECTED")


def test_cuda_requirement_probe_runs():
    _touch_from_env("TIREX_CUDA_PROBE_RAN")
