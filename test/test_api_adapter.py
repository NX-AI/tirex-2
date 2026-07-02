"""CPU-only tests for the forecasting adapter's data assembly and output formatting.

These exercise the adapter logic (input normalization, GluonTS extraction, output
formatting) without running the GPU-only :class:`TiRex2` forward pass.
"""

import numpy as np
import pandas as pd
import pytest
import torch

try:
    from gluonts.dataset.field_names import FieldName
    from gluonts.model.forecast import QuantileForecast

    from tirex2.api_adapter.gluon import build_gluon_timeseries, format_gluonts_output

    gluonts_available = True
except:
    gluonts_available = False

from tirex2.api_adapter.standard_adapter import build_timeseries

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
H = 10

_needs_gluonts = pytest.mark.skipif(not gluonts_available, reason="gluonts is needed to run this test")


def test_build_timeseries_2d_tensor_is_batch_of_univariate():
    series = build_timeseries(torch.randn(4, 50))
    assert len(series) == 4
    assert all(ts.target.shape == (1, 50) for ts in series)
    assert all(ts.past_covariates is None and ts.future_covariates is None for ts in series)


def test_build_timeseries_1d_input_is_single_univariate():
    series = build_timeseries(np.arange(20.0))
    assert len(series) == 1
    assert series[0].target.shape == (1, 20)


def test_build_timeseries_aligns_multivariate_targets_and_covariates():
    target = [torch.randn(2, 30), torch.randn(1, 40)]
    past = [torch.randn(3, 30), None]
    future = [torch.randn(1, 30 + H), torch.randn(2, 40 + H)]

    series = build_timeseries(target, past, future)

    assert len(series) == 2
    assert series[0].target.shape == (2, 30)
    assert series[0].past_covariates.shape == (3, 30)
    assert series[0].future_covariates.shape == (1, 40)
    assert series[1].past_covariates is None
    assert series[1].future_covariates.shape == (2, 50)


def test_build_timeseries_rejects_mismatched_covariate_count():
    with pytest.raises(AssertionError, match="one entry per target sample"):
        build_timeseries([torch.randn(1, 10), torch.randn(1, 10)], past_covariates=[torch.randn(1, 10)])


@_needs_gluonts
def test_build_gluon_timeseries_extracts_targets_covariates_and_meta():
    dataset = [
        {
            FieldName.TARGET: np.random.randn(2, 25),
            FieldName.START: pd.Period("2020-01-01", freq="D"),
            FieldName.ITEM_ID: "a",
            FieldName.PAST_FEAT_DYNAMIC_REAL: np.random.randn(1, 25),
            FieldName.FEAT_DYNAMIC_REAL: np.random.randn(1, 25 + H),
        },
        {
            FieldName.TARGET: np.random.randn(15),
            FieldName.START: pd.Period("2020-01-01", freq="D"),
            FieldName.ITEM_ID: "b",
        },
    ]

    series, meta = build_gluon_timeseries(dataset)

    assert series[0].target.shape == (2, 25)
    assert series[0].past_covariates.shape == (1, 25)
    assert series[0].future_covariates.shape == (1, 25 + H)
    assert series[1].target.shape == (1, 15)
    assert series[1].past_covariates is None and series[1].future_covariates is None
    assert meta[0]["length"] == 25 and meta[0]["num_targets"] == 2
    assert meta[1]["length"] == 15 and meta[1]["num_targets"] == 1


@_needs_gluonts
def test_format_gluonts_output_one_forecast_per_variate_with_mean():
    meta = [
        {FieldName.START: pd.Period("2020-01-01", freq="D"), FieldName.ITEM_ID: "a", "length": 25, "num_targets": 2},
        {FieldName.START: pd.Period("2020-01-01", freq="D"), FieldName.ITEM_ID: "b", "length": 15, "num_targets": 1},
    ]
    forecasts = [torch.randn(2, len(QUANTILES), H), torch.randn(1, len(QUANTILES), H)]

    result = format_gluonts_output(forecasts, meta, QUANTILES)

    assert len(result) == 3  # two variates from "a", one from "b"
    assert all(isinstance(f, QuantileForecast) for f in result)
    assert [f.item_id for f in result] == ["a_0", "a_1", "b"]
    # Quantile levels plus the median-proxy mean row.
    assert result[0].forecast_keys == [str(q) for q in QUANTILES] + ["mean"]
    assert result[0].forecast_array.shape == (len(QUANTILES) + 1, H)
    # Forecast starts right after the observed context.
    assert result[0].start_date == pd.Period("2020-01-01", freq="D") + 25
