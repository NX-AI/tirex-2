"""GluonTS data extraction and forecast formatting for the multivariate adapter."""

import pandas as pd
import torch
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.forecast import QuantileForecast

from ..model.types import TimeseriesType
from .standard_adapter import _ensure_2d_tensor

DEF_TARGET_COLUMN = FieldName.TARGET  # "target"
DEF_META_COLUMNS = (FieldName.START, FieldName.ITEM_ID)
DEF_PAST_COV_COLUMN = FieldName.PAST_FEAT_DYNAMIC_REAL  # "past_feat_dynamic_real" -> [V_p, T]
DEF_FUTURE_COV_COLUMN = FieldName.FEAT_DYNAMIC_REAL  # "feat_dynamic_real" -> [V_f, T + H]


def _extract_gluon_entry(entry: dict, **gluon_kwargs) -> tuple[TimeseriesType, dict]:
    """Turn a GluonTS dataset entry into a :class:`TimeseriesType` plus its forecast metadata."""
    target_col = gluon_kwargs.get("target_column", DEF_TARGET_COLUMN)
    meta_columns = gluon_kwargs.get("meta_columns", DEF_META_COLUMNS)
    past_cov_col = gluon_kwargs.get("past_covariates_column", DEF_PAST_COV_COLUMN)
    future_cov_col = gluon_kwargs.get("future_covariates_column", DEF_FUTURE_COV_COLUMN)

    target = _ensure_2d_tensor(entry[target_col])
    past_covariates = _ensure_2d_tensor(entry[past_cov_col]) if past_cov_col in entry else None
    future_covariates = _ensure_2d_tensor(entry[future_cov_col]) if future_cov_col in entry else None

    meta = {k: entry[k] for k in meta_columns if k in entry}
    meta["length"] = target.shape[-1]
    meta["num_targets"] = target.shape[0]

    return (
        TimeseriesType(target=target, past_covariates=past_covariates, future_covariates=future_covariates),
        meta,
    )


def build_gluon_timeseries(
    dataset: Dataset, multivariate: bool = False, **gluon_kwargs
) -> tuple[list[TimeseriesType], list[dict]]:
    """Extract every entry of a GluonTS dataset into parallel lists of timeseries and metadata.

    ``multivariate`` is stamped onto every metadata entry; it selects the output layout in
    :func:`format_gluonts_output` (one joint forecast per series vs. one forecast per variate).
    """
    series: list[TimeseriesType] = []
    meta: list[dict] = []
    for entry in dataset:
        ts, m = _extract_gluon_entry(entry, **gluon_kwargs)
        m["multivariate"] = multivariate
        series.append(ts)
        meta.append(m)
    return series, meta


def format_gluonts_output(
    forecasts: list[torch.Tensor],
    meta: list[dict],
    quantile_levels: list[float],
) -> list[QuantileForecast]:
    """Convert per-series ``[V_t, Q, H]`` quantile tensors into GluonTS ``QuantileForecast`` objects.

    The layout depends on the per-series ``multivariate`` metadata flag (set by
    :func:`build_gluon_timeseries`):

    - ``multivariate=False`` (default): emit one univariate ``QuantileForecast`` per target
      variate (``[Q + 1, H]``, including a median-proxy ``"mean"`` row). Multivariate series are
      thus scored channel-by-channel -- the standard GIFT-Eval leaderboard protocol.
    - ``multivariate=True``: emit one ``QuantileForecast`` per series, retaining the variate axis
      so GluonTS can score all variates jointly against the ``[V, H]`` label. Univariate series
      collapse to ``[Q, H]``; multivariate series expose ``[Q, H, V]``.
    """
    forecast_keys = [str(q) for q in quantile_levels] + ["mean"]
    multivariate_keys = [str(q) for q in quantile_levels]
    median_idx = min(range(len(quantile_levels)), key=lambda i: abs(quantile_levels[i] - 0.5))

    results: list[QuantileForecast] = []
    for series_forecast, m in zip(forecasts, meta):
        start_date = m.get(FieldName.START, pd.Period("2000-01-01", freq=m.get("freq", "h")))
        start_date = start_date + m.get("length", 0)
        item_id = m.get(FieldName.ITEM_ID, None)
        num_targets = series_forecast.shape[0]

        if m.get("multivariate", False):
            if num_targets == 1:
                arrays = series_forecast[0].cpu().numpy()  # [Q, H]
            else:
                arrays = series_forecast.permute(1, 2, 0).cpu().numpy()  # [V, Q, H] -> [Q, H, V]
            results.append(
                QuantileForecast(
                    forecast_arrays=arrays,
                    start_date=start_date,
                    item_id=item_id,
                    forecast_keys=multivariate_keys,
                )
            )
            continue

        for v in range(num_targets):
            quantiles = series_forecast[v]  # [Q, H]
            mean = quantiles[median_idx : median_idx + 1]  # [1, H], median proxy
            arrays = torch.cat((quantiles, mean), dim=0).cpu().numpy()  # [Q + 1, H]
            variate_id = item_id if num_targets == 1 else f"{item_id}_{v}"
            results.append(
                QuantileForecast(
                    forecast_arrays=arrays,
                    start_date=start_date,
                    item_id=variate_id,
                    forecast_keys=forecast_keys,
                )
            )
    return results
