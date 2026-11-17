import os
import sys
from pathlib import Path

# src layout: make ``import tirex2`` work under a bare ``pytest test/`` run.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pytest
import torch
from torch import nn

# Without a GPU, Triton kernels run in Triton's CPU interpreter so CPU-only CI still tests them.
# This must be set before triton is first imported (importing tirex2 below does).
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

from tirex2.model.component.variate_mixing_block import (
    MultivariateBlockConfig,
    TimeMixerConfig,
    VariateMixerConfig,
)
from tirex2.model.tirex2 import TiRex2

EMBEDDING_DIM = 512
PAST_LEN = 2048
FUTURE_LEN = 320
PATCH_SIZE = 32
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NUM_BLOCKS = 2
NUM_HEADS = 4
INPUT_FF_DIM = 2048
H_EXPAND = 4
DROPOUT = 0.1
EPS = 1e-7
RECIPE = ["mlstm_attn", "slstm_attn"] * (NUM_BLOCKS // 2)


def pytest_addoption(parser):
    parser.addoption(
        "--require-cuda",
        action="store_true",
        help="fail at pytest startup when CUDA is unavailable",
    )


def pytest_sessionstart(session):
    if session.config.getoption("--require-cuda") and not torch.cuda.is_available():
        raise pytest.UsageError("CUDA is required for this test run, but it is unavailable.")


@pytest.fixture
def interpreter_bfloat16(monkeypatch):
    """Make bfloat16 in Triton's CPU interpreter compute as on a GPU. Use only under the interpreter.

    The interpreter stores bfloat16 as uint16 bit patterns, and ``tl.dot`` multiplies those raw bits; the operands
    are widened to the float32 values they encode, as a GPU's bfloat16 dot with float32 accumulation computes.
    It also truncates float32 to bfloat16, where a GPU rounds to nearest even.
    """
    import numpy as np
    import triton.language as tl
    from triton.runtime.interpreter import InterpreterBuilder, TensorHandle

    create_dot, create_fp_trunc = InterpreterBuilder.create_dot, InterpreterBuilder.create_fp_trunc

    def to_float32(x):
        if x.dtype != tl.bfloat16:
            return x
        return TensorHandle((x.data.astype(np.uint32) << 16).view(np.float32), tl.float32)

    def bfloat16_dot(self, a, b, *args, **kwargs):
        return create_dot(self, to_float32(a), to_float32(b), *args, **kwargs)

    def rounding_fp_trunc(self, src, dst_type):
        if src.dtype.scalar != tl.float32 or dst_type.scalar != tl.bfloat16:
            return create_fp_trunc(self, src, dst_type)
        bits = src.data.astype(np.float32).view(np.uint32)
        return TensorHandle(((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16), tl.bfloat16)

    monkeypatch.setattr(InterpreterBuilder, "create_dot", bfloat16_dot)
    monkeypatch.setattr(InterpreterBuilder, "create_fp_trunc", rounding_fp_trunc)


def _build_block(model_type: str, act_fn: nn.Module, device: str) -> MultivariateBlockConfig:
    return MultivariateBlockConfig(
        time_mixer=TimeMixerConfig(
            model_type=model_type,
            act_fn=act_fn,
            embedding_dim=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            num_slstm_heads=NUM_HEADS,
            device=device,
        ),
        variate_mixer=VariateMixerConfig(
            act_fn=act_fn,
            embedding_dim=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
        ),
        dropout=DROPOUT,
        eps=EPS,
    )


def _build_model(device: str) -> TiRex2:
    act_fn = nn.SiLU()
    stack_config = {
        "templates": {
            "mlstm_attn": _build_block("bi-mlstm", act_fn, device),
            "slstm_attn": _build_block("bi-slstm", act_fn, device),
        },
        "recipe": RECIPE,
    }
    return TiRex2(
        stack_config=stack_config,
        num_blocks=NUM_BLOCKS,
        embedding_dim=EMBEDDING_DIM,
        input_patch_size=PATCH_SIZE,
        output_patch_size=PATCH_SIZE,
        quantiles=QUANTILES,
        tokenizer_cfg={
            "output_patch_size": PATCH_SIZE,
            "input_patch_size": PATCH_SIZE,
            "input_patch_stride": PATCH_SIZE,
        },
        scaler_cfg={"use_arcsinh": True, "binaryaware": True},
        h_expand=H_EXPAND,
        context_len=PAST_LEN,
        future_len=FUTURE_LEN,
        input_ff_dim=INPUT_FF_DIM,
        act_func="SiLU",
        dropout=DROPOUT,
        tta_diff=True,
        stack_out_norm_config={"eps": EPS},
        device=device,
    )


def _small_model_kwargs(device: str, recipe: list[str] | None = None) -> dict:
    # The mLSTM Triton chunkwise kernel requires q/k head dimension >= 16.
    # With qk_dim_factor=0.5 and four heads, this means embedding_dim >= 128.
    embedding_dim = 128
    patch_size = 4
    future_len = 8
    recipe = recipe or ["small_mlstm"]
    templates = {}
    for name, model_type in {"small_mlstm": "bi-mlstm", "small_slstm": "bi-slstm"}.items():
        templates[name] = {
            "time_mixer": {
                "model_type": model_type,
                "embedding_dim": embedding_dim,
                "num_heads": 4,
                "num_slstm_heads": 4,
                "device": device,
            },
            "variate_mixer": {
                "embedding_dim": embedding_dim,
                "num_heads": 4,
            },
            "dropout": 0.0,
            "eps": EPS,
        }
    stack_config = {"templates": templates, "recipe": recipe}

    return dict(
        stack_config=stack_config,
        num_blocks=len(recipe),
        embedding_dim=embedding_dim,
        input_patch_size=patch_size,
        output_patch_size=patch_size,
        quantiles=[0.1, 0.5, 0.9],
        tokenizer_cfg={
            "output_patch_size": patch_size,
            "input_patch_size": patch_size,
            "input_patch_stride": patch_size,
        },
        scaler_cfg={"use_arcsinh": True, "binaryaware": True},
        h_expand=2,
        context_len=16,
        future_len=future_len,
        input_ff_dim=64,
        act_func="SiLU",
        dropout=0.0,
        tta_diff=True,
        stack_out_norm_config={"eps": EPS},
        device=device,
    )


def _build_small_model(device: str, recipe: list[str] | None = None) -> TiRex2:
    return TiRex2(**_small_model_kwargs(device, recipe))


@pytest.fixture
def build_model():
    return _build_model


@pytest.fixture
def small_model_kwargs():
    return _small_model_kwargs


@pytest.fixture
def build_small_model():
    return _build_small_model
