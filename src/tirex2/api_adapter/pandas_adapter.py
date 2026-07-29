"""pandas ``DataFrame`` data extraction and forecast formatting."""

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from ..model.types import TimeseriesType

DEF_ID_COLUMN = "item_id"
DEF_TIMESTAMP_COLUMN = "timestamp"
DEF_TARGET_NAME_COLUMN = "target"
DEF_PREDICTION_COLUMN = "prediction"


def _as_column_list(columns: str | Sequence[str] | None) -> list[str]:
    """Normalize a single column name / sequence of names / ``None`` into a list."""
    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    return list(columns)


def _require_columns(df: pd.DataFrame, columns: Sequence[str], role: str) -> None:
    """Raise when any of ``columns`` is missing from ``df``."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{role} column(s) {missing} not found in the DataFrame (columns: {list(df.columns)})")


def _infer_time_step(timestamps: pd.Series):
    """Return the sampling interval of a sorted timestamp series, or ``None`` if undeterminable.

    Datetime timestamps yield a pandas offset (inferred by ``pd.infer_freq``, falling back to the
    most common consecutive difference); numeric timestamps yield the most common numeric
    difference. Series that are neither datetime nor numeric - or that hold a single point - yield
    ``None``, in which case forecasts are stamped with integer positions instead of timestamps.
    """
    if timestamps is None or len(timestamps) < 2:
        return None
    if pd.api.types.is_datetime64_any_dtype(timestamps):
        index = pd.DatetimeIndex(timestamps)
        try:
            inferred = pd.infer_freq(index)
        except ValueError:  # infer_freq needs at least 3 timestamps
            inferred = None
        if inferred is not None:
            return pd.tseries.frequencies.to_offset(inferred)
        diffs = index.to_series().diff().dropna()
        return pd.tseries.frequencies.to_offset(diffs.mode().iloc[0]) if not diffs.empty else None
    if pd.api.types.is_numeric_dtype(timestamps):
        diffs = timestamps.diff().dropna()
        return diffs.mode().iloc[0] if not diffs.empty else None
    return None


def _period_start(timestamps: pd.Series | None):
    """Return the first timestamp as a ``pd.Period``, or ``None`` when it has no usable frequency."""
    if timestamps is None or not len(timestamps) or not pd.api.types.is_datetime64_any_dtype(timestamps):
        return None
    step = _infer_time_step(timestamps)
    try:
        return pd.Period(timestamps.iloc[0], freq=step)
    except (ValueError, TypeError):
        return None


def _future_timestamps(meta: dict, horizon: int):
    """Continue a series' time axis for ``horizon`` steps beyond its last observed timestamp."""
    last = meta.get("last_timestamp")
    step = meta.get("time_step")
    if last is None or step is None:
        start = meta.get("length", 0)
        return np.arange(start, start + horizon)
    if isinstance(last, pd.Timestamp):
        return pd.date_range(start=last + step, periods=horizon, freq=step)
    return last + step * np.arange(1, horizon + 1)


def _select_target_columns(
    df: pd.DataFrame,
    target: str | Sequence[str] | None,
    reserved: Sequence[str],
) -> list[str]:
    """Resolve the target columns, defaulting to every numeric column that is not reserved."""
    if target is not None:
        targets = _as_column_list(target)
        _require_columns(df, targets, "Target")
        return targets
    reserved_set = set(reserved)
    targets = [c for c in df.columns if c not in reserved_set and pd.api.types.is_numeric_dtype(df[c])]
    if not targets:
        raise ValueError("Could not infer any numeric target column; pass target=... explicitly.")
    return targets


def _values_2d(frame: pd.DataFrame, columns: Sequence[str]) -> torch.Tensor:
    """Extract ``columns`` of ``frame`` as a float32 ``[num_variates, T]`` tensor."""
    return torch.as_tensor(frame[list(columns)].to_numpy(dtype=np.float32).T)


def _with_timestamp_column(df: pd.DataFrame, timestamp_column: str | None) -> tuple[pd.DataFrame, str | None]:
    """Promote a ``DatetimeIndex`` to a real column, and parse string timestamps into datetimes.

    Parsing lets a frame read straight from a CSV keep a real time axis; a column that does not
    parse (e.g. free-form labels) is left untouched and is then only used for ordering.
    """
    if timestamp_column is None:
        if not isinstance(df.index, pd.DatetimeIndex):
            return df, None
        timestamp_column = df.index.name or DEF_TIMESTAMP_COLUMN
        df = df.reset_index(names=timestamp_column)

    _require_columns(df, [timestamp_column], "Timestamp")
    column = df[timestamp_column]
    if not pd.api.types.is_datetime64_any_dtype(column) and not pd.api.types.is_numeric_dtype(column):
        try:
            with warnings.catch_warnings():  # the format-inference warning is noise for a best-effort parse
                warnings.simplefilter("ignore", UserWarning)
                df = df.assign(**{timestamp_column: pd.to_datetime(column)})
        except (ValueError, TypeError):
            pass
    return df, timestamp_column


def build_df_timeseries(
    df: pd.DataFrame,
    target: str | Sequence[str] | None = None,
    id_column: str | None = None,
    timestamp_column: str | None = None,
    past_covariates: str | Sequence[str] | None = None,
    future_covariates: str | Sequence[str] | None = None,
    future_df: pd.DataFrame | None = None,
    multivariate: bool = False,
) -> tuple[list[TimeseriesType], list[dict]]:
    """Extract the series of a pandas ``DataFrame`` into timeseries plus formatting metadata.

    Parameters
    ----------
    df
        Observed history. Rows of one series must share the same ``id_column`` value; within a
        series, rows are ordered by ``timestamp_column`` (a ``DatetimeIndex`` is used
        automatically when no timestamp column is given).
    target
        Target column(s). Defaults to every numeric column that is not the id column, the
        timestamp column or a covariate column.
    id_column
        Column identifying the series. When omitted the whole frame is a single series.
    past_covariates, future_covariates
        Covariate columns. ``future_covariates`` must also be present in ``future_df``, which
        supplies their values over the forecast horizon.
    future_df
        Known-future covariate values, in the same layout as ``df`` (same id and timestamp
        columns). Required when ``future_covariates`` is given.
    multivariate
        ``False`` (default) treats every target column as an independent univariate series;
        ``True`` forecasts the target columns of a series jointly as one multivariate series.
    """
    past_cov_cols = _as_column_list(past_covariates)
    future_cov_cols = _as_column_list(future_covariates)

    df, timestamp_column = _with_timestamp_column(df, timestamp_column)
    if id_column is not None:
        _require_columns(df, [id_column], "Id")
    _require_columns(df, past_cov_cols, "Past covariate")
    _require_columns(df, future_cov_cols, "Future covariate")

    reserved = [c for c in (id_column, timestamp_column) if c is not None] + past_cov_cols + future_cov_cols
    target_cols = _select_target_columns(df, target, reserved)

    if future_cov_cols and future_df is None:
        raise ValueError("future_covariates need their future values; pass future_df=...")
    future_groups: dict = {}
    if future_cov_cols:
        future_df, future_ts_column = _with_timestamp_column(future_df, timestamp_column)
        _require_columns(future_df, future_cov_cols, "Future covariate")
        if id_column is not None:
            _require_columns(future_df, [id_column], "Id")
            future_groups = {key: frame for key, frame in future_df.groupby(id_column, sort=False)}
        else:
            future_groups = {None: future_df}
        if future_ts_column is not None:
            future_groups = {
                key: frame.sort_values(future_ts_column, kind="stable") for key, frame in future_groups.items()
            }

    groups = df.groupby(id_column, sort=False) if id_column is not None else [(None, df)]

    series: list[TimeseriesType] = []
    meta: list[dict] = []
    for key, frame in groups:
        if timestamp_column is not None:
            frame = frame.sort_values(timestamp_column, kind="stable")
            timestamps = frame[timestamp_column]
        else:
            timestamps = None

        past_cov = _values_2d(frame, past_cov_cols) if past_cov_cols else None
        future_cov = None
        if future_cov_cols:
            if key not in future_groups:
                raise ValueError(f"future_df has no rows for series {key!r}")
            future_cov = torch.cat(
                (_values_2d(frame, future_cov_cols), _values_2d(future_groups[key], future_cov_cols)), dim=-1
            )

        base_meta = {
            "start": _period_start(timestamps),  # lets output_type="gluonts" keep the real time axis
            "item_id": key,
            "id_column": id_column,
            "timestamp_column": timestamp_column or DEF_TIMESTAMP_COLUMN,
            "length": len(frame),
            "last_timestamp": timestamps.iloc[-1] if timestamps is not None and len(timestamps) else None,
            "time_step": _infer_time_step(timestamps),
            "multivariate": multivariate,
        }

        variate_groups = [target_cols] if multivariate else [[col] for col in target_cols]
        for columns in variate_groups:
            series.append(
                TimeseriesType(
                    target=_values_2d(frame, columns),
                    past_covariates=past_cov,
                    future_covariates=future_cov,
                )
            )
            meta.append({**base_meta, "target_names": list(columns), "num_targets": len(columns)})

    return series, meta


def format_pandas_output(
    forecasts: list[torch.Tensor],
    meta: list[dict],
    quantile_levels: list[float],
) -> pd.DataFrame:
    """Convert per-series ``[V_t, Q, H]`` quantile tensors into one long-format ``DataFrame``.

    Each row is a single forecast step of a single target variate and carries the series id (when
    the input had an id column), the forecast timestamp, the target name, the median prediction
    and one column per quantile level (named by its level, e.g. ``"0.1"``).
    """
    median_idx = min(range(len(quantile_levels)), key=lambda i: abs(quantile_levels[i] - 0.5))

    frames = []
    for series_forecast, m in zip(forecasts, meta):
        values = series_forecast.cpu().numpy()  # [V, Q, H]
        horizon = values.shape[-1]
        timestamps = _future_timestamps(m, horizon)
        names = m.get("target_names") or [f"{DEF_TARGET_NAME_COLUMN}_{v}" for v in range(values.shape[0])]

        for v, name in enumerate(names):
            columns: dict = {}
            if m.get("id_column") is not None:
                columns[m["id_column"]] = m.get("item_id")
            columns[m.get("timestamp_column", DEF_TIMESTAMP_COLUMN)] = timestamps
            columns[DEF_TARGET_NAME_COLUMN] = name
            columns[DEF_PREDICTION_COLUMN] = values[v, median_idx]
            for q_idx, level in enumerate(quantile_levels):
                columns[str(level)] = values[v, q_idx]
            frames.append(pd.DataFrame(columns))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
