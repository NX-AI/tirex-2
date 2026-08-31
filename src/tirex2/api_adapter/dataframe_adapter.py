"""Backend-agnostic ``DataFrame`` data extraction and forecast formatting.

Frames are handled through `narwhals <https://narwhals-dev.github.io/narwhals/>`_, so any eager
dataframe it supports - pandas, polars, PyArrow, Modin, cuDF, ... - can be forecast, and the
resulting forecast frame is built with the same library the input came from.

pandas is used opportunistically: when it is installed, the time axis of a datetime column is
inferred with ``pandas.infer_freq``, which understands calendar frequencies (month end, quarter,
year) that a plain timedelta cannot express. Without pandas the step falls back to the most common
consecutive difference.
"""

import warnings
from collections.abc import Sequence
from functools import lru_cache
from types import ModuleType

import narwhals as nw
import numpy as np
import torch

from ..model.types import TimeseriesType

DEF_ID_COLUMN = "item_id"
DEF_TIMESTAMP_COLUMN = "timestamp"
DEF_TARGET_NAME_COLUMN = "target"
DEF_PREDICTION_COLUMN = "prediction"

IntoDataFrame = object  # any eager dataframe narwhals understands


@lru_cache(maxsize=1)
def _pandas() -> ModuleType | None:
    """Return the pandas module, or ``None`` when pandas is not installed."""
    try:
        import pandas as pd
    except ImportError:
        return None
    return pd


def _as_column_list(columns: str | Sequence[str] | None) -> list[str]:
    """Normalize a single column name / sequence of names / ``None`` into a list."""
    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    return list(columns)


def _require_columns(df: nw.DataFrame, columns: Sequence[str], role: str) -> None:
    """Raise when any of ``columns`` is missing from ``df``."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{role} column(s) {missing} not found in the DataFrame (columns: {list(df.columns)})")


def _modal_diff(values: np.ndarray):
    """Return the most common difference between consecutive ``values`` (smallest on a tie)."""
    diffs = np.diff(values)
    if not diffs.size:
        return None
    unique, counts = np.unique(diffs, return_counts=True)
    return unique[int(counts.argmax())]


def _infer_time_step(timestamps: nw.Series | None):
    """Return the sampling interval of a sorted timestamp series, or ``None`` if undeterminable.

    Datetime timestamps yield a pandas offset when pandas is installed (inferred by
    ``pd.infer_freq``, falling back to the most common consecutive difference) and a
    ``numpy.timedelta64`` otherwise; numeric timestamps yield the most common numeric difference.
    Series that are neither datetime nor numeric - or that hold a single point - yield ``None``, in
    which case forecasts are stamped with integer positions instead of timestamps.
    """
    if timestamps is None or len(timestamps) < 2:
        return None
    dtype = timestamps.dtype
    if not dtype.is_temporal() and not dtype.is_numeric():
        return None
    values = timestamps.to_numpy()
    pd = _pandas()
    if dtype.is_temporal() and pd is not None:
        index = pd.DatetimeIndex(values)
        try:
            inferred = pd.infer_freq(index)
        except ValueError:  # infer_freq needs at least 3 timestamps
            inferred = None
        if inferred is None:
            step = _modal_diff(values)
            return pd.tseries.frequencies.to_offset(step) if step is not None else None
        return pd.tseries.frequencies.to_offset(inferred)
    return _modal_diff(values)


def _period_start(timestamps: nw.Series | None, step):
    """Return the first timestamp as a ``pandas.Period``, or ``None`` without a usable frequency."""
    pd = _pandas()
    if pd is None or timestamps is None or not len(timestamps) or not timestamps.dtype.is_temporal():
        return None
    try:
        return pd.Period(pd.Timestamp(timestamps.to_numpy()[0]), freq=step)
    except (ValueError, TypeError):
        return None


def _future_timestamps(meta: dict, horizon: int) -> np.ndarray:
    """Continue a series' time axis for ``horizon`` steps beyond its last observed timestamp."""
    last = meta.get("last_timestamp")
    step = meta.get("time_step")
    if last is None or step is None:
        start = meta.get("length", 0)
        return np.arange(start, start + horizon)
    pd = _pandas()
    if pd is not None and isinstance(last, np.datetime64):
        # a pandas offset knows calendar arithmetic (month ends, DST); plain multiplication does not
        return pd.date_range(start=pd.Timestamp(last) + step, periods=horizon, freq=step).to_numpy()
    return last + step * np.arange(1, horizon + 1)


def _select_target_columns(
    df: nw.DataFrame,
    target: str | Sequence[str] | None,
    reserved: Sequence[str],
) -> list[str]:
    """Resolve the target columns, defaulting to every numeric column that is not reserved."""
    if target is not None:
        targets = _as_column_list(target)
        _require_columns(df, targets, "Target")
        return targets
    reserved_set = set(reserved)
    schema = df.schema
    targets = [c for c in df.columns if c not in reserved_set and schema[c].is_numeric()]
    if not targets:
        raise ValueError("Could not infer any numeric target column; pass target=... explicitly.")
    return targets


def _values_2d(frame: nw.DataFrame, columns: Sequence[str]) -> torch.Tensor:
    """Extract ``columns`` of ``frame`` as a float32 ``[num_variates, T]`` tensor."""
    # astype always copies, which also lifts the read-only flag some backends put on their arrays
    return torch.as_tensor(frame[list(columns)].to_numpy().astype(np.float32).T)


def _to_narwhals(df, promote_index: bool) -> tuple[nw.DataFrame, str | None]:
    """Wrap any supported eager dataframe, promoting a pandas ``DatetimeIndex`` to a real column.

    Returns the narwhals frame plus the name of the promoted index column (``None`` when there was
    nothing to promote); only pandas-like backends carry an index at all.
    """
    pd = _pandas()
    index_column = None
    if promote_index and pd is not None and isinstance(df, pd.DataFrame) and isinstance(df.index, pd.DatetimeIndex):
        index_column = df.index.name or DEF_TIMESTAMP_COLUMN
        df = df.reset_index(names=index_column)
    return nw.from_native(df, eager_only=True), index_column


def _with_timestamp_column(df: nw.DataFrame, timestamp_column: str | None) -> tuple[nw.DataFrame, str | None]:
    """Validate the timestamp column and parse string timestamps into datetimes.

    Parsing lets a frame read straight from a CSV keep a real time axis; a column that does not
    parse (e.g. free-form labels) is left untouched and is then only used for ordering.
    """
    if timestamp_column is None:
        return df, None

    _require_columns(df, [timestamp_column], "Timestamp")
    dtype = df.schema[timestamp_column]
    if not dtype.is_temporal() and not dtype.is_numeric():
        try:
            with warnings.catch_warnings():  # the format-inference warning is noise for a best-effort parse
                warnings.simplefilter("ignore", UserWarning)
                df = df.with_columns(nw.col(timestamp_column).str.to_datetime())
        except Exception:  # noqa: BLE001 - each backend raises its own parse error; a label column is fine as-is
            pass
    return df, timestamp_column


def _sort_by_time(frame: nw.DataFrame, timestamp_column: str) -> nw.DataFrame:
    """Sort by timestamp, keeping the input order of ties (not every backend sorts stably)."""
    order = nw.generate_temporary_column_name(8, frame.columns)
    return frame.with_row_index(order).sort(timestamp_column, order).drop(order)


def _group_by_id(df: nw.DataFrame, id_column: str | None) -> list[tuple]:
    """Split ``df`` into ``(id, frame)`` pairs, ordered by where each id first appears."""
    if id_column is None:
        return [(None, df)]
    groups = [(key[0] if isinstance(key, tuple) else key, frame) for key, frame in df.group_by(id_column)]
    # group_by makes no ordering promise (polars in particular does not), so restore the order the
    # ids first appear in - the same order pandas' groupby(sort=False) would have produced.
    ids = df[id_column].to_list()
    first_position: dict = {}
    for position, value in enumerate(ids):
        first_position.setdefault(value, position)
    return sorted(groups, key=lambda item: first_position.get(item[0], len(ids)))


def build_df_timeseries(
    df: IntoDataFrame,
    target: str | Sequence[str] | None = None,
    id_column: str | None = None,
    timestamp_column: str | None = None,
    past_covariates: str | Sequence[str] | None = None,
    future_covariates: str | Sequence[str] | None = None,
    future_df: IntoDataFrame | None = None,
    multivariate: bool = False,
) -> tuple[list[TimeseriesType], list[dict]]:
    """Extract the series of a ``DataFrame`` into timeseries plus formatting metadata.

    Parameters
    ----------
    df
        Observed history, as any eager dataframe narwhals supports (pandas, polars, PyArrow, ...).
        Rows of one series must share the same ``id_column`` value; within a series, rows are
        ordered by ``timestamp_column`` (a pandas ``DatetimeIndex`` is used automatically when no
        timestamp column is given).
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

    df, index_column = _to_narwhals(df, promote_index=timestamp_column is None)
    backend = df.implementation
    df, timestamp_column = _with_timestamp_column(df, timestamp_column or index_column)
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
        future_df, future_index_column = _to_narwhals(future_df, promote_index=timestamp_column is None)
        future_df, future_ts_column = _with_timestamp_column(future_df, timestamp_column or future_index_column)
        _require_columns(future_df, future_cov_cols, "Future covariate")
        if id_column is not None:
            _require_columns(future_df, [id_column], "Id")
        future_groups = dict(_group_by_id(future_df, id_column))
        if future_ts_column is not None:
            future_groups = {key: _sort_by_time(frame, future_ts_column) for key, frame in future_groups.items()}

    series: list[TimeseriesType] = []
    meta: list[dict] = []
    for key, frame in _group_by_id(df, id_column):
        if timestamp_column is not None:
            frame = _sort_by_time(frame, timestamp_column)
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

        time_step = _infer_time_step(timestamps)
        base_meta = {
            "start": _period_start(timestamps, time_step),  # lets output_type="gluonts" keep the real time axis
            "item_id": key,
            "id_column": id_column,
            "timestamp_column": timestamp_column or DEF_TIMESTAMP_COLUMN,
            "length": len(frame),
            "last_timestamp": timestamps.to_numpy()[-1] if timestamps is not None and len(timestamps) else None,
            "time_step": time_step,
            "multivariate": multivariate,
            "backend": backend,
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


def format_df_output(
    forecasts: list[torch.Tensor],
    meta: list[dict],
    quantile_levels: list[float],
    backend=None,
) -> IntoDataFrame:
    """Convert per-series ``[V_t, Q, H]`` quantile tensors into one long-format dataframe.

    Each row is a single forecast step of a single target variate and carries the series id (when
    the input had an id column), the forecast timestamp, the target name, the median prediction
    and one column per quantile level (named by its level, e.g. ``"0.1"``).

    The frame is built with ``backend`` when given (e.g. ``"pandas"``), otherwise with the
    dataframe library the forecast input came from, defaulting to pandas.
    """
    if backend is None:
        backend = meta[0].get("backend", "pandas") if meta else "pandas"
    if not forecasts:
        return nw.from_dict({}, backend=backend).to_native()

    median_idx = min(range(len(quantile_levels)), key=lambda i: abs(quantile_levels[i] - 0.5))
    id_column = meta[0].get("id_column")
    timestamp_column = meta[0].get("timestamp_column", DEF_TIMESTAMP_COLUMN)

    ids: list = []
    timestamps: list[np.ndarray] = []
    names: list[str] = []
    predictions: list[np.ndarray] = []
    quantiles: list[list[np.ndarray]] = [[] for _ in quantile_levels]

    for series_forecast, m in zip(forecasts, meta):
        values = series_forecast.cpu().numpy()  # [V, Q, H]
        horizon = values.shape[-1]
        series_timestamps = _future_timestamps(m, horizon)
        target_names = m.get("target_names") or [f"{DEF_TARGET_NAME_COLUMN}_{v}" for v in range(values.shape[0])]

        for v, name in enumerate(target_names):
            if id_column is not None:
                ids.extend([m.get("item_id")] * horizon)
            timestamps.append(series_timestamps)
            names.extend([name] * horizon)
            predictions.append(values[v, median_idx])
            for q_idx in range(len(quantile_levels)):
                quantiles[q_idx].append(values[v, q_idx])

    columns: dict = {}
    if id_column is not None:
        columns[id_column] = ids
    columns[timestamp_column] = np.concatenate(timestamps)
    columns[DEF_TARGET_NAME_COLUMN] = names
    columns[DEF_PREDICTION_COLUMN] = np.concatenate(predictions)
    for level, parts in zip(quantile_levels, quantiles):
        columns[str(level)] = np.concatenate(parts)

    return nw.from_dict(columns, backend=backend).to_native()
