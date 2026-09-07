"""Instantiation test for :class:`TiRex2`."""

from statistics import median

import pytest
import torch
import yaml
from torch import nn

from tirex2.base import CKPT_FILENAME, CONFIG_FILENAME, load_model
from tirex2.model.component.attention_block import (
    AttentionBlock,
    AttentionLayer,
    is_flex_attention_available,
)
from tirex2.model.component.postprocessor import PostProcessor
from tirex2.model.tirex2 import TiRex2
from tirex2.model.types import TimeseriesType

PAST_LEN = 2048
FUTURE_LEN = 320
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NUM_BLOCKS = 2
# for tests of the predict function
CTX_LEN = 512
PRED_LEN = 64


def _assert_module_on_device(model: nn.Module, device: str):
    assert {parameter.device.type for parameter in model.parameters()} == {device}
    assert {buffer.device.type for buffer in model.buffers()} == {device}


_needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="device requires a GPU")


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=_needs_gpu),
    ],
)
def test_tirex2_instantiates_from_config(device, build_model):
    model = build_model(device)

    assert isinstance(model, nn.Module)
    assert len(model.stack) == NUM_BLOCKS
    assert model.num_quantiles == len(QUANTILES)
    assert model.future_len == FUTURE_LEN
    assert model.context_len == PAST_LEN
    assert model.tta_diff is True
    assert isinstance(model.postprocessor, PostProcessor)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=_needs_gpu),
    ],
)
def test_tirex2_parameters_are_on_requested_device(device, build_small_model):
    model = build_small_model(device, recipe=["small_mlstm", "small_slstm"])

    _assert_module_on_device(model, device)
    assert {block.config.time_mixer.device for block in model.stack} == {device}


def test_load_model_device_overrides_checkpoint_config(tmp_path, small_model_kwargs):
    config = small_model_kwargs("cpu")
    model = TiRex2(**config)
    checkpoint_config = small_model_kwargs("cuda")

    with (tmp_path / CONFIG_FILENAME).open("w") as f:
        yaml.safe_dump(checkpoint_config, f)
    torch.save(model.state_dict(), tmp_path / CKPT_FILENAME)

    loaded = load_model(str(tmp_path), device="cpu")

    assert loaded.model.device == "cpu"
    assert loaded.model.tta_diff is True
    _assert_module_on_device(loaded.model, "cpu")
    assert {block.config.time_mixer.device for block in loaded.model.stack} == {"cpu"}


@pytest.mark.parametrize(
    "checkpoint_value, override",
    [(False, True), (True, False)],
    ids=["enable", "disable"],
)
def test_load_model_can_override_flex_attention(
    tmp_path,
    small_model_kwargs,
    require_flex_attention,
    checkpoint_value,
    override,
):
    config = small_model_kwargs("cpu")
    model = TiRex2(**config)
    for template in config["stack_config"]["templates"].values():
        template["variate_mixer"]["use_flex_attention"] = checkpoint_value

    with (tmp_path / CONFIG_FILENAME).open("w") as f:
        yaml.safe_dump(config, f)
    torch.save(model.state_dict(), tmp_path / CKPT_FILENAME)

    loaded = load_model(str(tmp_path), device="cpu", use_flex_attention=override)

    assert all(block.config.variate_mixer.use_flex_attention is override for block in loaded.model.stack)
    assert all(block.variate_mixer.use_flex_attention is override for block in loaded.model.stack)
    assert all(block.variate_mixer.attn.use_flex_attention is override for block in loaded.model.stack)


def test_tirex2_init_can_opt_into_matmul_precision(small_model_kwargs, monkeypatch):
    config = small_model_kwargs("cpu")
    precision_calls = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", precision_calls.append)

    model = TiRex2(**config, matmul_precision="high")

    assert precision_calls == ["high"]
    assert model.device == "cpu"


def test_load_model_passes_matmul_precision_from_checkpoint_config(tmp_path, small_model_kwargs, monkeypatch):
    config = small_model_kwargs("cpu")
    model = TiRex2(**config)
    checkpoint_config = {**config, "matmul_precision": "high"}

    with (tmp_path / CONFIG_FILENAME).open("w") as f:
        yaml.safe_dump(checkpoint_config, f)
    torch.save(model.state_dict(), tmp_path / CKPT_FILENAME)

    precision_calls = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", precision_calls.append)

    loaded = load_model(str(tmp_path), device="cpu")

    assert precision_calls == ["high"]
    assert loaded.model.device == "cpu"


def test_predict_uses_tta_diff_checkpoint_default_and_override(build_small_model):
    model = build_small_model("cpu").eval()
    model.tta_diff = False
    calls = []

    def fake_predict_once(timeseries, prediction_length, *args, tta_diff=True, **kwargs):
        calls.append(tta_diff)
        return []

    model._predict_once = fake_predict_once

    assert model.predict([], prediction_length=1) == []
    assert model.predict([], prediction_length=1, tta_diff=True) == []
    assert calls == [False, True]


# The compute paths below exercise the CUDA device, so they still need a GPU.
_gpu_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="forward requires a GPU")


def test_tirex2_bi_mlstm_forward_runs_on_cpu(build_small_model):
    model = build_small_model("cpu", recipe=["small_mlstm"]).eval()

    num_variates = 2
    seq_len = model.context_len + model.future_len
    batch = {
        "x": torch.randn(num_variates, seq_len),
        "target_mask": torch.tensor([True, False]),
        "group_vector": torch.ones(num_variates),
    }

    with torch.no_grad():
        out = model(batch)

    assert out.device.type == "cpu"
    assert out.shape == (num_variates, model.num_quantiles, seq_len)
    assert not torch.isnan(out).all()


@_gpu_only
def test_tirex2_cpu_and_cuda_models_loaded_from_same_parameters_match_on_random_timeseries(build_small_model):
    torch.manual_seed(0)
    recipe = ["small_mlstm", "small_slstm"]
    source_model = build_small_model("cpu", recipe=recipe).eval()
    state_dict = {key: value.detach().cpu().clone() for key, value in source_model.state_dict().items()}

    cpu_model = build_small_model("cpu", recipe=recipe).eval()
    cuda_model = build_small_model("cuda", recipe=recipe).eval()
    cpu_model.load_state_dict(state_dict, strict=True)
    cuda_model.load_state_dict(state_dict, strict=True)

    timeseries = [
        TimeseriesType(
            target=torch.randn(1, cpu_model.context_len),
            past_covariates=None,
            future_covariates=None,
        )
    ]

    cpu_forecast = cpu_model.predict(timeseries, prediction_length=4)[0]
    cuda_forecast = cuda_model.predict(timeseries, prediction_length=4)[0].cpu()

    torch.testing.assert_close(cuda_forecast, cpu_forecast, rtol=3e-2, atol=3e-2)


@_gpu_only
@pytest.mark.parametrize("with_group_vector", [False, True])
def test_tirex2_forward_produces_quantiles(with_group_vector, build_model):
    model = build_model("cuda").eval()

    num_variates = 3
    seq_len = PAST_LEN + FUTURE_LEN
    batch = {
        "x": torch.randn(num_variates, seq_len, device="cuda"),
        # Every row is a target; the bi-xLSTM split needs the mask to tell
        # targets (forward-only) from covariates (bidirectional).
        "target_mask": torch.ones(num_variates, dtype=torch.bool, device="cuda"),
    }
    if with_group_vector:
        # One group_vector entry per variate; here all variates share a group so
        # the variate mixer attends across the whole batch.
        batch["group_vector"] = torch.ones(num_variates, device="cuda")

    out = model(batch)

    # forward yields per-variate quantile bands aligned with the input length.
    assert out.shape[0] == num_variates
    assert out.shape[1] == model.num_quantiles
    assert not torch.isnan(out).all()


@_gpu_only
@pytest.mark.parametrize("prediction_length", [64, 50, 3])
def test_tirex2_predict_returns_forecast_per_series(prediction_length, build_model):
    model = build_model("cuda").eval()

    # ``predict`` moves the CPU input tensors onto the model's device itself.
    series = [
        TimeseriesType(target=torch.randn(1, 512), past_covariates=None, future_covariates=None),
        TimeseriesType(target=torch.randn(2, 700), past_covariates=None, future_covariates=None),
    ]

    forecasts = model.predict(series, prediction_length)

    assert len(forecasts) == len(series)
    for ts, forecast in zip(series, forecasts):
        num_targets = ts.target.shape[0]
        assert forecast.shape == (num_targets, model.num_quantiles, prediction_length)
        assert not torch.isnan(forecast).all()


@_gpu_only
@pytest.mark.parametrize(
    "past_covariates, future_covariates",
    [
        (None, None),
        (torch.randn(1, CTX_LEN), None),
        (None, torch.randn(1, CTX_LEN + PRED_LEN)),
    ],
    ids=["target-only", "with-past-covariate", "with-future-covariate"],
)
def test_tirex2_predict_with_covariate_combinations(past_covariates, future_covariates, build_model):
    model = build_model("cuda").eval()

    series = [
        TimeseriesType(
            target=torch.randn(1, CTX_LEN),
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
    ]

    forecasts = model.predict(series, PRED_LEN)

    # Covariates are inputs only: a single target always yields a single forecast variate.
    assert len(forecasts) == 1
    assert forecasts[0].shape == (1, model.num_quantiles, PRED_LEN)
    assert not torch.isnan(forecasts[0]).all()


def test_tirex2_predict_rejects_mismatched_covariate_lengths(build_model):
    # The length checks live in the postprocessor's input transform, which runs
    # on CPU before any GPU kernel, so this needs no GPU.
    model = build_model("cpu").eval()

    # A future covariate must span the target plus the full prediction horizon;
    # here it is only ``T + 5`` long while ``prediction_length`` is 10.
    too_short_future = [
        TimeseriesType(
            target=torch.randn(1, CTX_LEN),
            past_covariates=None,
            future_covariates=torch.randn(1, CTX_LEN + 5),
        )
    ]
    with pytest.raises(ValueError, match="Future known covariates"):
        model.predict(too_short_future, prediction_length=10)

    # A past covariate must match the target length exactly.
    mismatched_past = [
        TimeseriesType(
            target=torch.randn(1, CTX_LEN),
            past_covariates=torch.randn(1, CTX_LEN + 1),
            future_covariates=None,
        )
    ]
    with pytest.raises(ValueError, match="Past covariates and targets"):
        model.predict(mismatched_past, prediction_length=10)


_attention_devices = ["cpu", pytest.param("cuda", marks=_needs_gpu)]


@pytest.fixture
def require_flex_attention():
    if not is_flex_attention_available():
        pytest.skip("FlexAttention is not available in this PyTorch installation")


def _copy_weights(source: nn.Module, target: nn.Module) -> None:
    target.load_state_dict(source.state_dict(), strict=True)


@torch.no_grad()
def _run_attention_samples(
    layer: AttentionLayer,
    samples: list[torch.Tensor],
    group_vector: torch.Tensor,
    target_mask: torch.Tensor,
) -> None:
    for x in samples:
        layer(x, group_vector=group_vector, target_mask=target_mask)


def _time_attention_samples_ms(
    layer: AttentionLayer,
    samples: list[torch.Tensor],
    group_vector: torch.Tensor,
    target_mask: torch.Tensor,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _run_attention_samples(layer, samples, group_vector, target_mask)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def test_flex_attention_is_disabled_by_default(require_flex_attention):
    layer = AttentionLayer(
        input_dim=64,
        kv_proj_dim=16,
        n_heads=4,
        dropout=0.0,
    )
    block = AttentionBlock(
        input_dim=64,
        n_heads=4,
        dropout=0.0,
        act_fn=nn.SiLU(),
    )

    assert layer.use_flex_attention is False
    assert block.use_flex_attention is False
    assert block.attn.use_flex_attention is False


@pytest.mark.parametrize("device", _attention_devices)
def test_attention_layer_flex_matches_dense_attention_without_group_mask(device, require_flex_attention):
    torch.manual_seed(0)
    flex_layer = (
        AttentionLayer(
            input_dim=64,
            kv_proj_dim=16,
            n_heads=4,
            dropout=0.0,
            use_qk_norm=True,
            use_flex_attention=True,
        )
        .to(device)
        .eval()
    )
    dense_layer = (
        AttentionLayer(
            input_dim=64,
            kv_proj_dim=16,
            n_heads=4,
            dropout=0.0,
            use_qk_norm=True,
            use_flex_attention=False,
        )
        .to(device)
        .eval()
    )
    _copy_weights(flex_layer, dense_layer)

    x = torch.randn(2, 9, 64, device=device)

    with torch.no_grad():
        flex_out = flex_layer(x)
        dense_out = dense_layer(x)

    torch.testing.assert_close(flex_out, dense_out, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", _attention_devices)
def test_attention_layer_flex_matches_dense_attention_with_group_mask(device, require_flex_attention):
    torch.manual_seed(1)
    flex_layer = (
        AttentionLayer(
            input_dim=64,
            kv_proj_dim=16,
            n_heads=4,
            dropout=0.0,
            use_qk_norm=True,
            use_flex_attention=True,
        )
        .to(device)
        .eval()
    )
    dense_layer = (
        AttentionLayer(
            input_dim=64,
            kv_proj_dim=16,
            n_heads=4,
            dropout=0.0,
            use_qk_norm=True,
            use_flex_attention=False,
        )
        .to(device)
        .eval()
    )
    _copy_weights(flex_layer, dense_layer)

    x = torch.randn(2, 8, 64, device=device)
    group_vector = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], device=device)
    target_mask = torch.tensor([True, False, False, True, False, True, False, False], device=device)

    with torch.no_grad():
        flex_out = flex_layer(x, group_vector=group_vector, target_mask=target_mask)
        dense_out = dense_layer(x, group_vector=group_vector, target_mask=target_mask)

    torch.testing.assert_close(flex_out, dense_out, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", _attention_devices)
def test_attention_block_passes_flex_flag_and_matches_dense_attention_block(device, require_flex_attention):
    torch.manual_seed(2)
    flex_block = (
        AttentionBlock(
            input_dim=64,
            n_heads=4,
            dropout=0.0,
            act_fn=nn.SiLU(),
            use_flex_attention=True,
        )
        .to(device)
        .eval()
    )
    dense_block = (
        AttentionBlock(
            input_dim=64,
            n_heads=4,
            dropout=0.0,
            act_fn=nn.SiLU(),
            use_flex_attention=False,
        )
        .to(device)
        .eval()
    )
    _copy_weights(flex_block, dense_block)

    assert flex_block.attn.use_flex_attention is True
    assert dense_block.attn.use_flex_attention is False

    x = torch.randn(3, 6, 64, device=device)
    group_vector = torch.tensor([0, 0, 1, 1, 2, 2], device=device)
    target_mask = torch.tensor([True, False, True, False, True, False], device=device)

    with torch.no_grad():
        flex_out = flex_block(x, group_vector=group_vector, target_mask=target_mask)
        dense_out = dense_block(x, group_vector=group_vector, target_mask=target_mask)

    torch.testing.assert_close(flex_out, dense_out, rtol=1e-4, atol=1e-5)


@_gpu_only
def test_flex_attention_warm_execution_is_faster_than_dense_attention_on_sparse_groups(
    require_flex_attention,
):
    torch.manual_seed(3)
    device = "cuda"
    batch_size = 1
    seq_len = 8192
    input_dim = 256
    n_heads = 4
    group_size = 16
    num_samples = 2

    flex_layer = (
        AttentionLayer(
            input_dim=input_dim,
            kv_proj_dim=input_dim // n_heads,
            n_heads=n_heads,
            dropout=0.0,
            use_qk_norm=True,
            use_flex_attention=True,
        )
        .to(device)
        .eval()
    )
    dense_layer = (
        AttentionLayer(
            input_dim=input_dim,
            kv_proj_dim=input_dim // n_heads,
            n_heads=n_heads,
            dropout=0.0,
            use_qk_norm=True,
            use_flex_attention=False,
        )
        .to(device)
        .eval()
    )
    _copy_weights(flex_layer, dense_layer)

    samples = [torch.randn(batch_size, seq_len, input_dim, device=device) for _ in range(num_samples)]
    group_vector = torch.arange(seq_len, device=device) // group_size
    target_mask = (torch.arange(seq_len, device=device) % group_size) == 0

    for _ in range(2):
        _run_attention_samples(flex_layer, samples, group_vector, target_mask)
        _run_attention_samples(dense_layer, samples, group_vector, target_mask)
    torch.cuda.synchronize()

    flex_times = [_time_attention_samples_ms(flex_layer, samples, group_vector, target_mask) for _ in range(5)]
    dense_times = [_time_attention_samples_ms(dense_layer, samples, group_vector, target_mask) for _ in range(5)]

    assert median(flex_times) < median(dense_times), (
        f"Expected warmed FlexAttention to be faster than dense attention; "
        f"flex_times_ms={flex_times}, dense_times_ms={dense_times}"
    )
