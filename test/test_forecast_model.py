"""Quick stress tests for the public ForecastModel.forecast wrapper."""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

try:
    from gluonts.dataset.field_names import FieldName
    from gluonts.model.forecast import QuantileForecast

    gluonts_available = True
except:
    gluonts_available = False

try:
    import fev

    fev_available = True
except:
    fev_available = False

from tirex2.api_adapter.forecast import ForecastModel
from tirex2.model.types import TimeseriesType

_needs_gluonts = pytest.mark.skipif(not gluonts_available, reason="gluonts is needed to run this test")
_needs_fev = pytest.mark.skipif(not fev_available, reason="fev is needed to run this test")


class RecordingForecastBackbone:
    """Small fake backbone that records forecast calls and returns traceable tensors."""

    def __init__(self, future_len: int = 32):
        self.future_len = future_len
        self.quantiles = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
        self.calls = []

    def predict(self, timeseries, prediction_length, **kwargs):
        requested_prediction_length = prediction_length
        prediction_length = min(prediction_length, self.future_len)
        self.calls.append(
            {
                "batch_size": len(timeseries),
                "requested_prediction_length": requested_prediction_length,
                "prediction_length": prediction_length,
                "target_shapes": [tuple(ts.target.shape) for ts in timeseries],
                "past_covariate_shapes": [
                    None if ts.past_covariates is None else tuple(ts.past_covariates.shape) for ts in timeseries
                ],
                "future_covariate_shapes": [
                    None if ts.future_covariates is None else tuple(ts.future_covariates.shape) for ts in timeseries
                ],
                "kwargs": dict(kwargs),
            }
        )

        forecasts = []
        horizon = torch.arange(prediction_length, dtype=torch.float32).reshape(1, 1, -1) / 1000.0
        for ts in timeseries:
            sample_id = float(ts.target[0, 0])
            num_targets = ts.target.shape[0]
            variates = torch.arange(num_targets, dtype=torch.float32).reshape(-1, 1, 1) / 10.0
            forecasts.append(
                torch.full(
                    (num_targets, len(self.quantiles), prediction_length),
                    sample_id,
                    dtype=torch.float32,
                )
                + variates
                + horizon
            )
        return forecasts


def _mixed_series(num_items: int, prediction_length: int = 8) -> list[TimeseriesType]:
    series = []
    for idx in range(num_items):
        num_targets = 1 + idx % 3
        context_length = 5 + idx % 7
        target = torch.full((num_targets, context_length), float(idx), dtype=torch.float32)
        past_covariates = (
            torch.full((1, context_length), float(idx) / 10.0, dtype=torch.float32) if idx % 2 == 0 else None
        )
        future_covariates = (
            torch.full((2, context_length + prediction_length), float(idx) / 20.0, dtype=torch.float32)
            if idx % 3 == 0
            else None
        )
        series.append(
            TimeseriesType(
                target=target,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
            )
        )
    return series


def _real_model_series(prediction_length: int) -> list[TimeseriesType]:
    first_target = torch.linspace(-1.0, 1.0, 12).reshape(1, -1)
    second_base = torch.linspace(0.2, 2.0, 16)
    second_target = torch.stack((second_base, torch.sin(second_base)))
    second_past_covariates = torch.stack((torch.cos(second_base),))
    second_future_covariates = torch.randn(2, second_target.shape[-1] + prediction_length)
    return [
        TimeseriesType(
            target=first_target,
            past_covariates=None,
            future_covariates=None,
        ),
        TimeseriesType(
            target=second_target,
            past_covariates=second_past_covariates,
            future_covariates=second_future_covariates,
        ),
    ]


@_needs_gluonts
def _gluon_entries(prediction_length: int):
    return [
        {
            FieldName.TARGET: np.stack(
                (
                    np.linspace(10.0, 16.0, 7, dtype=np.float32),
                    np.linspace(20.0, 26.0, 7, dtype=np.float32),
                )
            ),
            FieldName.START: pd.Period("2020-01-01", freq="D"),
            FieldName.ITEM_ID: "multi",
            FieldName.PAST_FEAT_DYNAMIC_REAL: np.full((1, 7), 0.1, dtype=np.float32),
            FieldName.FEAT_DYNAMIC_REAL: np.full((2, 7 + prediction_length), 0.2, dtype=np.float32),
        },
        {
            FieldName.TARGET: np.linspace(30.0, 34.0, 5, dtype=np.float32),
            FieldName.START: pd.Period("2021-03-01", freq="D"),
            FieldName.ITEM_ID: "single",
        },
        {
            FieldName.TARGET: np.linspace(40.0, 45.0, 6, dtype=np.float32),
            FieldName.START: pd.Period("2022-06-01", freq="D"),
            FieldName.ITEM_ID: "past-only",
            FieldName.PAST_FEAT_DYNAMIC_REAL: np.full((2, 6), 0.3, dtype=np.float32),
        },
    ]


@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_forecast_batches_mixed_series_preserving_order(batch_size):
    model = RecordingForecastBackbone(future_len=16)
    adapter = ForecastModel(model)
    series = _mixed_series(17, prediction_length=9)

    forecasts = adapter.forecast(
        series,
        prediction_length=9,
        batch_size=batch_size,
        output_type="torch",
        stress_flag=True,
    )

    expected_batch_sizes = [min(batch_size, len(series) - i) for i in range(0, len(series), batch_size)]
    assert [call["batch_size"] for call in model.calls] == expected_batch_sizes
    assert all(call["prediction_length"] == 9 for call in model.calls)
    assert all(call["kwargs"] == {"stress_flag": True} for call in model.calls)

    assert len(forecasts) == len(series)
    for idx, (ts, forecast) in enumerate(zip(series, forecasts)):
        assert forecast.shape == (ts.target.shape[0], len(model.quantiles), 9)
        assert forecast[0, 0, 0].item() == pytest.approx(float(idx))
        assert forecast.device.type == "cpu"


def test_forecast_yield_per_batch_is_lazy_and_formats_numpy():
    model = RecordingForecastBackbone(future_len=8)
    adapter = ForecastModel(model)
    stream = adapter.forecast(
        _mixed_series(5, prediction_length=4),
        prediction_length=4,
        batch_size=2,
        output_type="numpy",
        yield_per_batch=True,
    )

    assert model.calls == []

    first_batch = next(stream)
    assert [call["batch_size"] for call in model.calls] == [2]
    assert len(first_batch) == 2
    assert isinstance(first_batch[0], np.ndarray)
    assert first_batch[0].shape == (1, len(model.quantiles), 4)

    remaining_batches = list(stream)
    assert [call["batch_size"] for call in model.calls] == [2, 2, 1]
    assert [len(batch) for batch in remaining_batches] == [2, 1]


def test_forecast_returns_backbone_truncated_horizon():
    model = RecordingForecastBackbone(future_len=6)
    adapter = ForecastModel(model)
    series = _mixed_series(3, prediction_length=10)

    forecasts = adapter.forecast(
        series,
        prediction_length=10,
        batch_size=2,
        output_type="torch",
    )

    assert [call["requested_prediction_length"] for call in model.calls] == [10, 10]
    assert [call["prediction_length"] for call in model.calls] == [6, 6]
    assert all(forecast.shape[-1] == 6 for forecast in forecasts)


def test_forecast_real_small_model_returns_expected_shapes(build_small_model):
    torch.manual_seed(0)
    backbone = build_small_model("cpu").eval()
    adapter = ForecastModel(backbone)
    prediction_length = 4
    series = _real_model_series(prediction_length)

    forecasts = adapter.forecast(
        series,
        prediction_length=prediction_length,
        batch_size=1,
        output_type="torch",
    )

    assert len(forecasts) == len(series)
    for ts, forecast in zip(series, forecasts):
        assert forecast.shape == (ts.target.shape[0], backbone.num_quantiles, prediction_length)
        assert forecast.device.type == "cpu"
        assert not torch.isnan(forecast).all()


def test_forecast_real_small_model_logs_and_caps_overlong_prediction_length(caplog, build_small_model):
    torch.manual_seed(1)
    backbone = build_small_model("cpu").eval()
    adapter = ForecastModel(backbone)
    requested_prediction_length = backbone.future_len + 3
    series = [
        TimeseriesType(
            target=torch.randn(1, backbone.context_len),
            past_covariates=None,
            future_covariates=None,
        )
    ]

    with caplog.at_level(logging.WARNING):
        forecasts = adapter.forecast(
            series,
            prediction_length=requested_prediction_length,
            output_type="torch",
        )

    assert any(
        f"prediction_length={requested_prediction_length} exceeds the supported maximum of "
        f"{backbone.future_len}" in record.message
        for record in caplog.records
    )
    assert forecasts[0].shape == (1, backbone.num_quantiles, backbone.future_len)


def test_forecast_real_small_model_truncates_overlong_future_covariates_for_capped_horizon(build_small_model):
    torch.manual_seed(3)
    backbone = build_small_model("cpu").eval()
    adapter = ForecastModel(backbone)
    requested_prediction_length = backbone.future_len + 2
    target = torch.randn(1, backbone.context_len)
    covariate_prefix = torch.randn(2, target.shape[-1] + backbone.future_len)
    extra_covariates = torch.full(
        (2, requested_prediction_length - backbone.future_len + 3),
        1_000.0,
    )
    overlong_future_covariates = torch.cat((covariate_prefix, extra_covariates), dim=-1)

    exact_forecasts = adapter.forecast(
        [
            TimeseriesType(
                target=target,
                past_covariates=None,
                future_covariates=covariate_prefix,
            )
        ],
        prediction_length=requested_prediction_length,
        output_type="torch",
    )
    overlong_forecasts = adapter.forecast(
        [
            TimeseriesType(
                target=target,
                past_covariates=None,
                future_covariates=overlong_future_covariates,
            )
        ],
        prediction_length=requested_prediction_length,
        output_type="torch",
    )

    assert overlong_forecasts[0].shape == (1, backbone.num_quantiles, backbone.future_len)
    torch.testing.assert_close(overlong_forecasts[0], exact_forecasts[0])


def test_forecast_delegates_short_future_covariate_length_to_backbone(build_small_model, monkeypatch):
    torch.manual_seed(5)
    backbone = build_small_model("cpu").eval()
    adapter = ForecastModel(backbone)
    prediction_length = 4
    target = torch.randn(1, backbone.context_len)
    future_covariates = torch.randn(2, target.shape[-1] + prediction_length - 1)
    predict_was_called = False
    original_predict = backbone.predict

    def spy_predict(*args, **kwargs):
        nonlocal predict_was_called
        predict_was_called = True
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(backbone, "predict", spy_predict)

    with pytest.raises(ValueError, match="Future known covariates"):
        adapter.forecast(
            [
                TimeseriesType(
                    target=target,
                    past_covariates=None,
                    future_covariates=future_covariates,
                )
            ],
            prediction_length=prediction_length,
            output_type="torch",
        )

    assert predict_was_called


def test_forecast_delegates_mismatched_past_covariate_length_to_backbone(build_small_model, monkeypatch):
    torch.manual_seed(6)
    backbone = build_small_model("cpu").eval()
    adapter = ForecastModel(backbone)
    prediction_length = 4
    target = torch.randn(1, backbone.context_len)
    past_covariates = torch.randn(2, target.shape[-1] + 1)
    predict_was_called = False
    original_predict = backbone.predict

    def spy_predict(*args, **kwargs):
        nonlocal predict_was_called
        predict_was_called = True
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(backbone, "predict", spy_predict)

    with pytest.raises(ValueError, match="Past covariates and targets"):
        adapter.forecast(
            [
                TimeseriesType(
                    target=target,
                    past_covariates=past_covariates,
                    future_covariates=None,
                )
            ],
            prediction_length=prediction_length,
            output_type="torch",
        )

    assert predict_was_called


def test_forecast_real_small_model_rejects_nonpositive_prediction_length(build_small_model):
    backbone = build_small_model("cpu").eval()
    adapter = ForecastModel(backbone)

    with pytest.raises(ValueError, match="prediction_length must be >= 1"):
        adapter.forecast(
            [
                TimeseriesType(
                    target=torch.randn(1, backbone.context_len),
                    past_covariates=None,
                    future_covariates=None,
                )
            ],
            prediction_length=0,
            output_type="torch",
        )


def test_forecast_rejects_invalid_output_type():
    model = RecordingForecastBackbone()
    adapter = ForecastModel(model)

    with pytest.raises(ValueError, match="Invalid output type"):
        adapter.forecast([], prediction_length=4, output_type="invalid")


@pytest.mark.parametrize("output_type", ["torch", "numpy", pytest.param("gluonts", marks=_needs_gluonts)])
def test_forecast_empty_input_returns_empty_list(output_type):
    model = RecordingForecastBackbone()
    adapter = ForecastModel(model)

    assert adapter.forecast([], prediction_length=4, output_type=output_type) == []
    assert model.calls == []


@pytest.mark.parametrize("output_type", ["torch", "numpy", pytest.param("gluonts", marks=_needs_gluonts)])
def test_forecast_real_small_model_allowed_output_types(output_type, build_small_model):
    torch.manual_seed(4)
    backbone = build_small_model("cpu").eval()
    adapter = ForecastModel(backbone)
    prediction_length = 4
    series = _real_model_series(prediction_length)

    forecasts = adapter.forecast(
        series,
        prediction_length=prediction_length,
        batch_size=2,
        output_type=output_type,
    )

    if output_type == "torch":
        assert len(forecasts) == len(series)
        for ts, forecast in zip(series, forecasts):
            assert isinstance(forecast, torch.Tensor)
            assert forecast.shape == (ts.target.shape[0], backbone.num_quantiles, prediction_length)
            assert not torch.isnan(forecast).all()
    elif output_type == "numpy":
        assert len(forecasts) == len(series)
        for ts, forecast in zip(series, forecasts):
            assert isinstance(forecast, np.ndarray)
            assert forecast.shape == (ts.target.shape[0], backbone.num_quantiles, prediction_length)
            assert not np.isnan(forecast).all()
    else:
        expected_num_forecasts = sum(ts.target.shape[0] for ts in series)
        assert len(forecasts) == expected_num_forecasts
        assert all(isinstance(forecast, QuantileForecast) for forecast in forecasts)
        assert all(
            forecast.forecast_array.shape == (backbone.num_quantiles + 1, prediction_length) for forecast in forecasts
        )


@_needs_gluonts
def test_forecast_gluon_batches_entries_and_forwards_predict_kwargs():
    model = RecordingForecastBackbone(future_len=10)
    adapter = ForecastModel(model)
    prediction_length = 5
    entries = _gluon_entries(prediction_length)

    forecasts = adapter.forecast_gluon(
        entries,
        prediction_length=prediction_length,
        batch_size=2,
        output_type="torch",
        stress_flag=True,
    )

    assert [call["batch_size"] for call in model.calls] == [2, 1]
    assert all(call["prediction_length"] == prediction_length for call in model.calls)
    assert all(call["kwargs"] == {"stress_flag": True} for call in model.calls)

    assert [shape for call in model.calls for shape in call["target_shapes"]] == [(2, 7), (1, 5), (1, 6)]
    assert [shape for call in model.calls for shape in call["past_covariate_shapes"]] == [
        (1, 7),
        None,
        (2, 6),
    ]
    assert [shape for call in model.calls for shape in call["future_covariate_shapes"]] == [
        (2, 12),
        None,
        None,
    ]
    assert [forecast.shape for forecast in forecasts] == [
        (2, len(model.quantiles), prediction_length),
        (1, len(model.quantiles), prediction_length),
        (1, len(model.quantiles), prediction_length),
    ]


@_needs_fev
def test_forecast_fev_batches_window_and_forwards_predict_kwargs(monkeypatch):
    model = RecordingForecastBackbone(future_len=10)
    adapter = ForecastModel(model)
    prediction_length = 5
    window = SimpleNamespace(horizon=prediction_length, target_columns=["target"])
    series = _mixed_series(3, prediction_length=prediction_length)

    def fake_build_fev_timeseries(received_window, **kwargs):
        assert received_window is window
        assert kwargs == {"as_univariate": True}
        return series, [
            {
                "target_columns": ["target"],
                "window_target_columns": list(window.target_columns),
                "as_univariate": True,
            }
            for _ in series
        ]

    monkeypatch.setattr("tirex2.api_adapter.forecast.build_fev_timeseries", fake_build_fev_timeseries)

    forecasts = adapter.forecast_fev(
        window,
        prediction_length=prediction_length,
        batch_size=2,
        output_type="torch",
        data_kwargs={"as_univariate": True},
        stress_flag=True,
    )

    assert [call["batch_size"] for call in model.calls] == [2, 1]
    assert all(call["prediction_length"] == prediction_length for call in model.calls)
    assert all(call["kwargs"] == {"stress_flag": True} for call in model.calls)
    assert len(forecasts) == len(series)
    assert [forecast.shape for forecast in forecasts] == [
        (ts.target.shape[0], len(model.quantiles), prediction_length) for ts in series
    ]


@_needs_fev
def test_forecast_fev_output_type_returns_fev_datasetdict(monkeypatch):
    import datasets

    model = RecordingForecastBackbone(future_len=8)
    adapter = ForecastModel(model)
    prediction_length = 4
    window = SimpleNamespace(horizon=prediction_length, target_columns=["load", "temperature"])
    series = [
        TimeseriesType(
            target=torch.tensor([[10.0, 11.0], [20.0, 21.0]], dtype=torch.float32),
            past_covariates=None,
            future_covariates=None,
        ),
        TimeseriesType(
            target=torch.tensor([[30.0, 31.0], [40.0, 41.0]], dtype=torch.float32),
            past_covariates=None,
            future_covariates=None,
        ),
    ]

    def fake_build_fev_timeseries(received_window, **kwargs):
        assert received_window is window
        assert kwargs == {}
        return series, [
            {
                "target_columns": list(window.target_columns),
                "window_target_columns": list(window.target_columns),
                "as_univariate": False,
            }
            for _ in series
        ]

    monkeypatch.setattr("tirex2.api_adapter.forecast.build_fev_timeseries", fake_build_fev_timeseries)

    predictions = adapter.forecast_fev(
        window,
        prediction_length=prediction_length,
        output_type="fev",
        batch_size=1,
        quantile_levels=[0.2, 0.5, 0.8],
    )

    assert isinstance(predictions, datasets.DatasetDict)
    assert set(predictions.keys()) == {"load", "temperature"}
    assert predictions["load"].column_names == ["predictions", "0.2", "0.5", "0.8"]
    np.testing.assert_allclose(predictions["load"][0]["predictions"], [10.0, 10.001, 10.002, 10.003])
    np.testing.assert_allclose(
        predictions["temperature"][1]["predictions"],
        [30.1, 30.101, 30.102, 30.103],
        rtol=1e-6,
    )


@_needs_fev
def test_forecast_fev_can_return_model_only_inference_time(monkeypatch):
    import datasets

    model = RecordingForecastBackbone(future_len=8)
    adapter = ForecastModel(model)
    prediction_length = 4
    window = SimpleNamespace(horizon=prediction_length, target_columns=["target"])
    series = [
        TimeseriesType(
            target=torch.tensor([[10.0, 11.0]], dtype=torch.float32),
            past_covariates=None,
            future_covariates=None,
        ),
        TimeseriesType(
            target=torch.tensor([[30.0, 31.0]], dtype=torch.float32),
            past_covariates=None,
            future_covariates=None,
        ),
    ]

    def fake_build_fev_timeseries(received_window, **kwargs):
        assert received_window is window
        assert kwargs == {}
        return series, [
            {
                "target_columns": list(window.target_columns),
                "window_target_columns": list(window.target_columns),
                "as_univariate": False,
            }
            for _ in series
        ]

    ticks = iter([100.0, 104.25])
    monkeypatch.setattr("tirex2.api_adapter.forecast.build_fev_timeseries", fake_build_fev_timeseries)
    monkeypatch.setattr("tirex2.api_adapter.forecast.time.monotonic", lambda: next(ticks))

    predictions, inference_time_s = adapter.forecast_fev(
        window,
        prediction_length=prediction_length,
        output_type="fev",
        batch_size=1,
        return_inference_time=True,
    )

    assert inference_time_s == pytest.approx(4.25)
    assert isinstance(predictions, datasets.DatasetDict)
    assert set(predictions.keys()) == {"target"}
    np.testing.assert_allclose(predictions["target"][1]["predictions"], [30.0, 30.001, 30.002, 30.003])


@_needs_gluonts
def test_forecast_gluon_yield_per_batch_is_lazy_and_formats_numpy():
    model = RecordingForecastBackbone(future_len=10)
    adapter = ForecastModel(model)
    stream = adapter.forecast_gluon(
        _gluon_entries(prediction_length=4),
        prediction_length=4,
        batch_size=1,
        output_type="numpy",
        yield_per_batch=True,
    )

    assert model.calls == []

    first_batch = next(stream)
    assert [call["batch_size"] for call in model.calls] == [1]
    assert len(first_batch) == 1
    assert isinstance(first_batch[0], np.ndarray)
    assert first_batch[0].shape == (2, len(model.quantiles), 4)

    remaining_batches = list(stream)
    assert [call["batch_size"] for call in model.calls] == [1, 1, 1]
    assert [len(batch) for batch in remaining_batches] == [1, 1]


@_needs_gluonts
def test_forecast_gluon_honors_custom_data_columns():
    model = RecordingForecastBackbone(future_len=8)
    adapter = ForecastModel(model)
    prediction_length = 3
    entries = [
        {
            "observed": np.stack(
                (
                    np.linspace(1.0, 4.0, 4, dtype=np.float32),
                    np.linspace(5.0, 8.0, 4, dtype=np.float32),
                )
            ),
            "history_covariates": np.full((1, 4), 0.4, dtype=np.float32),
            "known_covariates": np.full((2, 4 + prediction_length), 0.5, dtype=np.float32),
            FieldName.START: pd.Period("2020-02-01", freq="D"),
            FieldName.ITEM_ID: "custom",
        }
    ]

    forecasts = adapter.forecast_gluon(
        entries,
        prediction_length=prediction_length,
        output_type="torch",
        data_kwargs={
            "target_column": "observed",
            "past_covariates_column": "history_covariates",
            "future_covariates_column": "known_covariates",
        },
    )

    assert model.calls[0]["target_shapes"] == [(2, 4)]
    assert model.calls[0]["past_covariate_shapes"] == [(1, 4)]
    assert model.calls[0]["future_covariate_shapes"] == [(2, 7)]
    assert forecasts[0].shape == (2, len(model.quantiles), prediction_length)


@_needs_gluonts
@pytest.mark.parametrize("output_type", ["torch", "numpy", "gluonts"])
def test_forecast_gluon_allowed_output_types(output_type):
    model = RecordingForecastBackbone(future_len=10)
    adapter = ForecastModel(model)
    prediction_length = 4
    entries = _gluon_entries(prediction_length)[:2]

    forecasts = adapter.forecast_gluon(
        entries,
        prediction_length=prediction_length,
        batch_size=2,
        output_type=output_type,
    )

    if output_type == "torch":
        assert len(forecasts) == len(entries)
        assert all(isinstance(forecast, torch.Tensor) for forecast in forecasts)
        assert [forecast.shape for forecast in forecasts] == [
            (2, len(model.quantiles), prediction_length),
            (1, len(model.quantiles), prediction_length),
        ]
    elif output_type == "numpy":
        assert len(forecasts) == len(entries)
        assert all(isinstance(forecast, np.ndarray) for forecast in forecasts)
        assert [forecast.shape for forecast in forecasts] == [
            (2, len(model.quantiles), prediction_length),
            (1, len(model.quantiles), prediction_length),
        ]
    else:
        assert len(forecasts) == 3
        assert all(isinstance(forecast, QuantileForecast) for forecast in forecasts)
        assert [forecast.item_id for forecast in forecasts] == ["multi_0", "multi_1", "single"]
        assert [forecast.start_date for forecast in forecasts] == [
            pd.Period("2020-01-01", freq="D") + 7,
            pd.Period("2020-01-01", freq="D") + 7,
            pd.Period("2021-03-01", freq="D") + 5,
        ]
        assert all(
            forecast.forecast_array.shape == (len(model.quantiles) + 1, prediction_length) for forecast in forecasts
        )
        assert all(forecast.forecast_keys == ["0.1", "0.5", "0.9", "mean"] for forecast in forecasts)


@_needs_gluonts
def test_forecast_gluon_empty_input_returns_empty_list():
    model = RecordingForecastBackbone()
    adapter = ForecastModel(model)

    assert adapter.forecast_gluon([], prediction_length=4, output_type="gluonts") == []
    assert model.calls == []


@_needs_gluonts
def test_forecast_gluon_delegates_short_future_covariate_length_to_backbone(build_small_model, monkeypatch):
    torch.manual_seed(7)
    backbone = build_small_model("cpu").eval()
    adapter = ForecastModel(backbone)
    prediction_length = 4
    predict_was_called = False
    original_predict = backbone.predict

    def spy_predict(*args, **kwargs):
        nonlocal predict_was_called
        predict_was_called = True
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(backbone, "predict", spy_predict)

    with pytest.raises(ValueError, match="Future known covariates"):
        adapter.forecast_gluon(
            [
                {
                    FieldName.TARGET: np.random.randn(backbone.context_len).astype(np.float32),
                    FieldName.START: pd.Period("2020-01-01", freq="D"),
                    FieldName.ITEM_ID: "bad-future",
                    FieldName.FEAT_DYNAMIC_REAL: np.random.randn(
                        2, backbone.context_len + prediction_length - 1
                    ).astype(np.float32),
                }
            ],
            prediction_length=prediction_length,
            output_type="torch",
        )

    assert predict_was_called


@_needs_gluonts
def test_forecast_gluon_delegates_mismatched_past_covariate_length_to_backbone(build_small_model, monkeypatch):
    torch.manual_seed(8)
    backbone = build_small_model("cpu").eval()
    adapter = ForecastModel(backbone)
    prediction_length = 4
    predict_was_called = False
    original_predict = backbone.predict

    def spy_predict(*args, **kwargs):
        nonlocal predict_was_called
        predict_was_called = True
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(backbone, "predict", spy_predict)

    with pytest.raises(ValueError, match="Past covariates and targets"):
        adapter.forecast_gluon(
            [
                {
                    FieldName.TARGET: np.random.randn(backbone.context_len).astype(np.float32),
                    FieldName.START: pd.Period("2020-01-01", freq="D"),
                    FieldName.ITEM_ID: "bad-past",
                    FieldName.PAST_FEAT_DYNAMIC_REAL: np.random.randn(2, backbone.context_len + 1).astype(np.float32),
                }
            ],
            prediction_length=prediction_length,
            output_type="torch",
        )

    assert predict_was_called
