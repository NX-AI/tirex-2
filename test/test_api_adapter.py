"""CPU-only tests for the forecasting adapter's data assembly and output formatting.

These exercise the adapter logic (input normalization, GluonTS extraction, output
formatting) without running the GPU-only :class:`TiRex2` forward pass.
"""

import importlib.util

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


try:
    import narwhals as nw

    from tirex2.api_adapter.dataframe_adapter import build_df_timeseries, format_df_output

    df_adapter_available = True
except ImportError:
    df_adapter_available = False

_needs_df_adapter = pytest.mark.skipif(not df_adapter_available, reason="narwhals is needed to run this test")


def _backend_params():
    """Every dataframe backend to exercise, skipping the ones whose library is not installed."""
    params = [pytest.param("pandas", id="pandas")]
    for backend in ("polars", "pyarrow"):
        missing = importlib.util.find_spec(backend) is None
        marks = pytest.mark.skipif(missing, reason=f"{backend} is needed to run this test")
        params.append(pytest.param(backend, id=backend, marks=marks))
    return params


BACKENDS = _backend_params()


def _as_backend(df: pd.DataFrame, backend: str):
    """Convert a pandas frame built by the helpers below into ``backend``'s native frame."""
    if backend == "pandas":
        return df
    if backend == "polars":
        import polars as pl

        return pl.from_pandas(df)
    if backend == "pyarrow":
        import pyarrow as pa

        return pa.Table.from_pandas(df, preserve_index=False)
    raise AssertionError(f"unknown backend {backend}")


def _column(result, name) -> list:
    """Read a column of a forecast frame in whatever backend it was produced in."""
    return nw.from_native(result, eager_only=True)[name].to_list()


def _long_df(num_items=2, length=20):
    return pd.DataFrame(
        {
            "item_id": np.repeat([f"item_{i}" for i in range(num_items)], length),
            "timestamp": np.tile(pd.date_range("2020-01-01", periods=length, freq="D"), num_items),
            "sales": np.random.randn(num_items * length),
            "price": np.random.randn(num_items * length),
        }
    )


@_needs_df_adapter
@pytest.mark.parametrize("backend", BACKENDS)
def test_build_df_timeseries_long_format_one_series_per_id(backend):
    df = _as_backend(_long_df(), backend)
    series, meta = build_df_timeseries(df, target="sales", id_column="item_id", timestamp_column="timestamp")

    assert len(series) == 2
    assert all(ts.target.shape == (1, 20) for ts in series)
    assert [m["item_id"] for m in meta] == ["item_0", "item_1"]
    assert meta[0]["target_names"] == ["sales"]
    assert meta[0]["last_timestamp"] == np.datetime64("2020-01-20")
    assert meta[0]["time_step"] == pd.tseries.frequencies.to_offset("D")
    assert str(meta[0]["backend"]) == backend


@_needs_df_adapter
@pytest.mark.parametrize("backend", BACKENDS)
def test_build_df_timeseries_multiple_targets_univariate_vs_multivariate(backend):
    df = _as_backend(_long_df(), backend)

    univariate, uni_meta = build_df_timeseries(df, id_column="item_id", timestamp_column="timestamp")
    assert len(univariate) == 4  # two ids x two numeric target columns
    assert all(ts.target.shape == (1, 20) for ts in univariate)
    assert [m["target_names"] for m in uni_meta] == [["sales"], ["price"]] * 2

    joint, joint_meta = build_df_timeseries(df, id_column="item_id", timestamp_column="timestamp", multivariate=True)
    assert len(joint) == 2
    assert all(ts.target.shape == (2, 20) for ts in joint)
    assert joint_meta[0]["target_names"] == ["sales", "price"]


@_needs_df_adapter
@pytest.mark.parametrize("backend", BACKENDS)
def test_build_df_timeseries_sorts_rows_by_timestamp(backend):
    df = _long_df(num_items=1, length=5).drop(columns=["item_id", "price"])
    shuffled = _as_backend(df.iloc[[3, 0, 4, 1, 2]].reset_index(drop=True), backend)

    series, meta = build_df_timeseries(shuffled, target="sales", timestamp_column="timestamp")

    assert len(series) == 1
    np.testing.assert_allclose(series[0].target[0].numpy(), df["sales"].to_numpy(dtype=np.float32))
    assert meta[0]["id_column"] is None


@_needs_df_adapter
def test_build_df_timeseries_uses_a_pandas_datetime_index_as_the_time_axis():
    df = _long_df(num_items=1, length=5).drop(columns=["item_id", "price"])
    shuffled = df.iloc[[3, 0, 4, 1, 2]].set_index("timestamp")

    series, meta = build_df_timeseries(shuffled, target="sales")

    assert len(series) == 1
    np.testing.assert_allclose(series[0].target[0].numpy(), df["sales"].to_numpy(dtype=np.float32))
    assert meta[0]["timestamp_column"] == "timestamp"
    assert meta[0]["id_column"] is None


@_needs_df_adapter
@pytest.mark.parametrize("backend", BACKENDS)
def test_build_df_timeseries_covariates_extend_over_horizon(backend):
    df = _as_backend(_long_df(), backend)
    future_df = _as_backend(
        pd.DataFrame(
            {
                "item_id": np.repeat(["item_0", "item_1"], H),
                "timestamp": np.tile(pd.date_range("2020-01-21", periods=H, freq="D"), 2),
                "price": np.random.randn(2 * H),
            }
        ),
        backend,
    )

    series, _ = build_df_timeseries(
        df, target="sales", id_column="item_id", timestamp_column="timestamp", past_covariates="price"
    )
    assert series[0].past_covariates.shape == (1, 20)

    with_future, _ = build_df_timeseries(
        df,
        target="sales",
        id_column="item_id",
        timestamp_column="timestamp",
        future_covariates="price",
        future_df=future_df,
    )
    assert with_future[0].future_covariates.shape == (1, 20 + H)
    assert with_future[0].future_length == H


@_needs_df_adapter
def test_build_df_timeseries_reports_bad_columns():
    df = _long_df()
    with pytest.raises(ValueError, match="Target column"):
        build_df_timeseries(df, target="missing", id_column="item_id")
    with pytest.raises(ValueError, match="pass future_df"):
        build_df_timeseries(df, target="sales", id_column="item_id", future_covariates="price")
    with pytest.raises(ValueError, match="no rows for series"):
        build_df_timeseries(
            df,
            target="sales",
            id_column="item_id",
            future_covariates="price",
            future_df=df[df["item_id"] == "item_0"],
        )


@_needs_df_adapter
@pytest.mark.parametrize("backend", BACKENDS)
def test_format_df_output_long_frame_with_quantiles_and_timestamps(backend):
    df = _as_backend(_long_df(), backend)
    series, meta = build_df_timeseries(df, target="sales", id_column="item_id", timestamp_column="timestamp")
    forecasts = [torch.randn(1, len(QUANTILES), H) for _ in series]

    result = format_df_output(forecasts, meta, QUANTILES)

    # The forecast comes back in the same dataframe library the input came from.
    assert str(nw.from_native(result, eager_only=True).implementation) == backend
    assert list(nw.from_native(result, eager_only=True).columns) == [
        "item_id",
        "timestamp",
        "target",
        "prediction",
    ] + [str(q) for q in QUANTILES]
    assert len(_column(result, "item_id")) == 2 * H
    assert _column(result, "item_id") == ["item_0"] * H + ["item_1"] * H
    assert set(_column(result, "target")) == {"sales"}
    # Timestamps continue the daily context, which ends on 2020-01-20.
    timestamps = _column(result, "timestamp")
    assert pd.Timestamp(timestamps[0]) == pd.Timestamp("2020-01-21")
    assert pd.Timestamp(timestamps[H - 1]) == pd.Timestamp("2020-01-30")
    # The median column mirrors the 0.5 quantile.
    np.testing.assert_allclose(_column(result, "prediction"), _column(result, "0.5"))


@_needs_df_adapter
@pytest.mark.parametrize("backend", BACKENDS)
def test_format_df_output_multivariate_rows_per_target_column(backend):
    df = _as_backend(_long_df(), backend)
    series, meta = build_df_timeseries(df, id_column="item_id", timestamp_column="timestamp", multivariate=True)
    forecasts = [torch.randn(2, len(QUANTILES), H) for _ in series]

    result = format_df_output(forecasts, meta, QUANTILES)
    frame = nw.from_native(result, eager_only=True)

    assert len(frame) == 2 * 2 * H  # two ids x two target columns x horizon
    assert dict.fromkeys(_column(result, "target")) == dict.fromkeys(["sales", "price"])
    selected = frame.filter((nw.col("item_id") == "item_0") & (nw.col("target") == "price"))
    np.testing.assert_allclose(selected["0.9"].to_numpy(), forecasts[0][1, -1].numpy())


@_needs_df_adapter
@pytest.mark.parametrize("backend", BACKENDS)
def test_format_df_output_can_be_forced_to_pandas(backend):
    df = _as_backend(_long_df(num_items=1), backend)
    series, meta = build_df_timeseries(df, target="sales", id_column="item_id", timestamp_column="timestamp")

    result = format_df_output([torch.randn(1, len(QUANTILES), H) for _ in series], meta, QUANTILES, backend="pandas")

    assert isinstance(result, pd.DataFrame)
    assert result["timestamp"].iloc[0] == pd.Timestamp("2020-01-21")


@_needs_df_adapter
@pytest.mark.parametrize("backend", BACKENDS)
def test_format_df_output_falls_back_to_integer_positions(backend):
    df = _as_backend(pd.DataFrame({"sales": np.random.randn(12)}), backend)
    series, meta = build_df_timeseries(df, target="sales")
    result = format_df_output([torch.randn(1, len(QUANTILES), H)], meta, QUANTILES)

    assert "item_id" not in nw.from_native(result, eager_only=True).columns
    assert _column(result, "timestamp") == list(range(12, 12 + H))


@_needs_df_adapter
@pytest.mark.parametrize("backend", BACKENDS)
def test_build_df_timeseries_parses_string_timestamps_from_csv_like_frames(backend):
    df = _long_df(num_items=1, length=6)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d")

    _, meta = build_df_timeseries(
        _as_backend(df, backend), target="sales", id_column="item_id", timestamp_column="timestamp"
    )

    assert meta[0]["last_timestamp"] == np.datetime64("2020-01-06")
    assert meta[0]["time_step"] == pd.tseries.frequencies.to_offset("D")


@_needs_df_adapter
@pytest.mark.parametrize("backend", BACKENDS)
def test_build_df_timeseries_keeps_unparsable_timestamps_for_ordering_only(backend):
    df = _long_df(num_items=1, length=4)
    df["timestamp"] = ["step-b", "step-d", "step-a", "step-c"]
    values = df["sales"].to_numpy(dtype=np.float32)

    series, meta = build_df_timeseries(
        _as_backend(df, backend), target="sales", id_column="item_id", timestamp_column="timestamp"
    )

    np.testing.assert_allclose(series[0].target[0].numpy(), values[[2, 0, 3, 1]])
    assert meta[0]["time_step"] is None
    # Without a usable time axis the forecast falls back to integer positions.
    result = format_df_output([torch.randn(1, len(QUANTILES), H)], meta, QUANTILES)
    assert _column(result, "timestamp") == list(range(4, 4 + H))


@_needs_df_adapter
@pytest.mark.parametrize("backend", BACKENDS)
def test_build_df_timeseries_keeps_first_appearance_order_of_ids(backend):
    df = _long_df(num_items=3, length=5)
    # Reverse the id blocks so first-appearance order and sorted order disagree.
    reversed_ids = pd.concat([df[df["item_id"] == f"item_{i}"] for i in (2, 0, 1)], ignore_index=True)

    _, meta = build_df_timeseries(
        _as_backend(reversed_ids, backend), target="sales", id_column="item_id", timestamp_column="timestamp"
    )

    assert [m["item_id"] for m in meta] == ["item_2", "item_0", "item_1"]
