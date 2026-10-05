import sys
from pathlib import Path

# src layout: make ``import tirex2`` work under a bare ``pytest test/`` run.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pytest
from torch import nn

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
