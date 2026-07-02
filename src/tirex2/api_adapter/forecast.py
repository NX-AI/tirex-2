"""High-level forecasting API wrapping a :class:`TiRex2` backbone."""

import logging
import time
from typing import Literal

import numpy as np
import torch

from ..model.types import TimeseriesType

logger = logging.getLogger(__file__)

ForecastOutputType = Literal["torch", "numpy", "gluonts", "fev"]


def _format_output(forecasts, meta, output_type, quantile_levels):
    """Render a batch of per-series ``[V_t, Q, H]`` forecasts in the requested output format."""
    if output_type == "torch":
        return [f.cpu() for f in forecasts]
    elif output_type == "numpy":
        return [f.cpu().numpy() for f in forecasts]
    elif output_type == "gluonts":
        try:
            from .gluon import format_gluonts_output
        except ImportError:
            raise ValueError("output_type gluonts needs GluonTS but GluonTS is not available (not installed)!")
        return format_gluonts_output(forecasts, meta, quantile_levels)
    elif output_type == "fev":
        return format_fev_output(forecasts, meta, quantile_levels)
    else:
        raise ValueError(f"Invalid output type: {output_type}")


def _predict_adaptive(
    model,
    timeseries,
    meta,
    prediction_length,
    output_type,
    batch_size,
    quantile_levels,
    **predict_kwargs,
):
    """Yield formatted forecasts batch by batch, halving the batch size on CUDA OOM.

    Walks contiguous ``[start, end)`` windows of at most ``batch_size`` series,
    forecasting and formatting each (slicing ``meta`` alongside ``timeseries``).
    When a window raises :class:`torch.cuda.OutOfMemoryError`, the CUDA cache is
    cleared, the batch size is halved (floor of 1), and the *same* window is retried
    at the smaller size. The reduced size persists for the rest of this call, so a
    single oversized window pins it down only here - a fresh call starts again from
    ``batch_size``. An OOM at size 1 is re-raised: a lone series that does not fit
    cannot be split further.

    Formatting - and the ``.cpu()`` move it performs - happens per window, so GPU
    memory backing completed batches is released as we go rather than accumulating
    across the whole dataset.
    """
    assert batch_size >= 1, "Batch size must be >= 1"
    num_items = len(timeseries)
    start = 0
    current = batch_size
    while start < num_items:
        end = min(start + current, num_items)
        try:
            forecasts = model.predict(timeseries[start:end], prediction_length, **predict_kwargs)
            formatted = _format_output(forecasts, meta[start:end], output_type, quantile_levels)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current == 1:
                logger.error("CUDA OOM at batch size 1 (series index %d); cannot shrink further.", start)
                raise
            current = max(1, current // 2)
            logger.warning("CUDA OOM at series index %d; halving batch size to %d and retrying.", start, current)
            continue
        yield formatted
        start = end


def _interpolate_quantile_levels(
    values: np.ndarray,
    native_levels: list[float],
    query_levels: list[float],
) -> dict[float, np.ndarray]:
    """Linearly interpolate a ``[..., Q, H]`` quantile array (axis ``-2``) onto ``query_levels``."""
    native = np.asarray(native_levels, dtype=float)
    interpolated: dict[float, np.ndarray] = {}
    for level in query_levels:
        if level <= native[0]:
            interpolated[level] = values[..., 0, :]
        elif level >= native[-1]:
            interpolated[level] = values[..., -1, :]
        else:
            hi = int(np.searchsorted(native, level))
            lo = hi - 1
            weight = (level - native[lo]) / (native[hi] - native[lo])
            interpolated[level] = (1.0 - weight) * values[..., lo, :] + weight * values[..., hi, :]
    return interpolated


def build_fev_timeseries(
    window: "fev.EvaluationWindow",
    as_univariate: bool = False,
) -> tuple[list[TimeseriesType], list[dict]]:
    """Convert a fev evaluation window into timeseries plus metadata for FEV formatting."""
    import datasets
    import fev
    import torch

    if as_univariate:
        past_data, future_data = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
        target_columns = ["target"]
        past_dynamic_columns: list[str] = []
        known_dynamic_columns: list[str] = []
    else:
        past_data, future_data = fev.convert_input_data(window, adapter="datasets", as_univariate=False)
        target_columns = list(window.target_columns)
        past_dynamic_columns = list(window.past_dynamic_columns)
        known_dynamic_columns = list(window.known_dynamic_columns)

    past_data = past_data.select_columns(target_columns + past_dynamic_columns + known_dynamic_columns)
    future_data = future_data.select_columns(known_dynamic_columns)
    future_data_renamed = future_data.rename_columns({col: f"{col}_future" for col in future_data.column_names})
    merged_ds = datasets.concatenate_datasets([past_data, future_data_renamed], axis=1).with_format("torch")

    def map_sample(sample):
        # ``.map`` does not apply the dataset's ``with_format("torch")`` transform, so the
        # callback receives plain Python lists; convert them to tensors explicitly.
        covariates = []
        for col in known_dynamic_columns:
            try:
                past_dyn = torch.as_tensor(sample[col])
                future_dyn = torch.as_tensor(sample[f"{col}_future"])
                combined = torch.concatenate((past_dyn, future_dyn))
                covariates.append(combined)
            except Exception:
                logger.warning(f"could not convert column {col}, skipping it.")

        future_covariates = torch.stack(covariates) if covariates else None
        past_covariates = (
            torch.stack([torch.as_tensor(sample[col]) for col in past_dynamic_columns])
            if past_dynamic_columns
            else None
        )
        targets = torch.stack([torch.as_tensor(sample[col]) for col in target_columns])
        return {"targets": targets, "past_covariates": past_covariates, "future_covariates": future_covariates}

    loaded = merged_ds.map(map_sample, remove_columns=merged_ds.column_names)
    timeseries = [
        TimeseriesType(
            target=item["targets"],
            past_covariates=item["past_covariates"],
            future_covariates=item["future_covariates"],
        )
        for item in loaded
    ]
    meta = [
        {
            "target_columns": target_columns,
            "window_target_columns": list(window.target_columns),
            "as_univariate": as_univariate,
        }
        for _ in timeseries
    ]
    return timeseries, meta


def format_fev_output(
    forecasts: list[torch.Tensor],
    meta: list[dict],
    quantile_levels: list[float],
) -> "datasets.DatasetDict":
    """Convert per-series ``[V_t, Q, H]`` forecasts into FEV-compatible predictions."""
    try:
        import datasets
    except ImportError:
        raise ValueError("output_type fev needs datasets but datasets is not available (not installed)!")

    if not forecasts:
        raise ValueError("output_type fev needs at least one forecast to infer the FEV schema.")

    format_meta = meta[0] if meta else {}
    target_columns = format_meta.get("target_columns")
    if target_columns is None:
        raise ValueError("output_type fev needs FEV metadata; use forecast_fev instead.")

    requested_quantiles = format_meta.get("quantile_levels", quantile_levels)
    forecasts_np = np.stack([f.cpu().numpy() for f in forecasts])
    quantile_arrays = _interpolate_quantile_levels(forecasts_np, quantile_levels, requested_quantiles)
    median = _interpolate_quantile_levels(forecasts_np, quantile_levels, [0.5])[0.5]

    predictions_dict = {}
    for v_idx, name in enumerate(target_columns):
        variate_forecast = {"predictions": median[:, v_idx]}
        for level in requested_quantiles:
            variate_forecast[str(level)] = quantile_arrays[level][:, v_idx]
        predictions_dict[name] = datasets.Dataset.from_dict(variate_forecast)
    predictions = datasets.DatasetDict(predictions_dict)
    predictions.set_format("numpy")

    if format_meta.get("as_univariate", False):
        try:
            import fev
        except ImportError:
            raise ValueError("output_type fev with as_univariate=True needs fev but fev is not available!")
        predictions = fev.utils.combine_univariate_predictions_to_multivariate(
            predictions, format_meta["window_target_columns"]
        )

    return predictions


def _gen_forecast(
    model,
    timeseries,
    meta,
    prediction_length,
    output_type,
    batch_size,
    yield_per_batch,
    quantile_levels,
    return_inference_time=False,
    **predict_kwargs,
):
    """Batch the timeseries, run :meth:`TiRex2.predict`, and accumulate or stream the formatted output.

    The batch size is reduced automatically on CUDA out-of-memory errors and
    reset for each call; see :func:`_predict_adaptive`.
    """
    if meta is None:
        meta = [{} for _ in timeseries]

    if output_type not in ["numpy", "torch", "gluonts", "fev"]:
        raise ValueError("Invalid output type")

    if output_type == "fev" and yield_per_batch:
        raise ValueError("yield_per_batch=True is not supported with output_type='fev'.")
    if return_inference_time and yield_per_batch:
        raise ValueError("return_inference_time=True is not supported with yield_per_batch=True.")

    adaptive_output_type = "torch" if output_type == "fev" else output_type
    batch_outputs = _predict_adaptive(
        model,
        timeseries,
        meta,
        prediction_length,
        adaptive_output_type,
        batch_size,
        quantile_levels,
        **predict_kwargs,
    )

    if yield_per_batch:
        return batch_outputs

    inference_start = time.monotonic() if return_inference_time else None
    all_forecasts = []
    for formatted in batch_outputs:
        all_forecasts.extend(formatted)
    inference_time_s = time.monotonic() - inference_start if inference_start is not None else None
    if output_type == "fev":
        result = _format_output(all_forecasts, meta, output_type, quantile_levels)
    else:
        result = all_forecasts
    if return_inference_time:
        return result, inference_time_s
    return result


class ForecastModel:
    """High-level, batched forecasting interface around a :class:`TiRex2` backbone.

    The wrapper takes ownership of the model only as a delegate: it batches the
    :class:`~tirex.model.types.TimeseriesType` it is given (building them from a GluonTS
    dataset in :meth:`forecast_gluon`), feeds them to :meth:`TiRex2.predict`, and formats the
    per-series quantile forecasts into the requested output type. Attribute access falls
    through to the wrapped model, so the backbone's own methods (e.g. ``predict``) remain
    reachable on the wrapper.

    Parameters
    ----------
    model : TiRex2
        An instantiated, ready-for-inference backbone exposing
        ``predict(timeseries: list[TimeseriesType], prediction_length: int) -> list[Tensor]``
        and a ``quantiles`` buffer holding the quantile levels it forecasts.
    """

    def __init__(self, model):
        self.model = model

    def _quantile_levels(self) -> list[float]:
        """Return the model's forecast quantile levels as clean Python floats (float32 noise rounded off)."""
        return [round(float(q), 6) for q in self.model.quantiles]

    def __getattr__(self, name):
        """Delegate unknown attribute lookups to the wrapped model."""
        try:
            model = object.__getattribute__(self, "model")
        except AttributeError:
            raise AttributeError(name)
        return getattr(model, name)

    def forecast(
        self,
        timeseries: list[TimeseriesType],
        prediction_length: int,
        output_type: ForecastOutputType = "torch",
        batch_size: int = 512,
        yield_per_batch: bool = False,
        **predict_kwargs,
    ):
        """Forecast a list of :class:`TimeseriesType`, each carrying a target and optional covariates.

        Extra ``predict_kwargs`` are forwarded verbatim to :meth:`TiRex2.predict`.
        In particular ``tta_sign_flip`` controls sign-flip test-time augmentation
        (roughly doubles inference cost), and ``tta_diff`` controls postprocessor
        differencing; when omitted, the checkpoint's configured defaults
        (``model-config.yaml``) are used. Pass ``True``/``False`` to override.
        """
        assert batch_size >= 1, "Batch size must be >= 1"
        return _gen_forecast(
            self.model,
            list(timeseries),
            None,
            prediction_length,
            output_type,
            batch_size,
            yield_per_batch,
            self._quantile_levels(),
            **predict_kwargs,
        )

    def forecast_gluon(
        self,
        gluonDataset,
        prediction_length: int,
        output_type: ForecastOutputType = "torch",
        batch_size: int = 512,
        yield_per_batch: bool = False,
        multivariate: bool = False,
        data_kwargs: dict = {},
        **predict_kwargs,
    ):
        """Forecast every entry of a GluonTS dataset, carrying its covariates and metadata through.

        With ``multivariate=False`` (default) each target variate is rendered as its own
        univariate ``QuantileForecast``; with ``multivariate=True`` each series yields a single
        forecast retaining the variate axis, so a multivariate dataset is scored jointly rather
        than channel-by-channel. The flag only affects ``output_type="gluonts"`` formatting.

        Extra ``predict_kwargs`` are forwarded verbatim to :meth:`TiRex2.predict`.
        In particular ``tta_sign_flip`` controls sign-flip test-time augmentation
        (roughly doubles inference cost), and ``tta_diff`` controls postprocessor
        differencing; when omitted, the checkpoint's configured defaults
        (``model-config.yaml``) are used. Pass ``True``/``False`` to override.
        """
        assert batch_size >= 1, "Batch size must be >= 1"
        try:
            from .gluon import build_gluon_timeseries
        except ImportError:
            raise ValueError("forecast_gluon needs GluonTS but GluonTS is not available (not installed)!")

        timeseries, meta = build_gluon_timeseries(gluonDataset, multivariate=multivariate, **data_kwargs)
        return _gen_forecast(
            self.model,
            timeseries,
            meta,
            prediction_length,
            output_type,
            batch_size,
            yield_per_batch,
            self._quantile_levels(),
            **predict_kwargs,
        )

    def forecast_fev(
        self,
        window: "fev.EvaluationWindow",
        prediction_length: int,
        output_type: ForecastOutputType = "torch",
        batch_size: int = 512,
        yield_per_batch: bool = False,
        data_kwargs: dict | None = None,
        quantile_levels: list[float] | None = None,
        return_inference_time: bool = False,
        **predict_kwargs,
    ):
        """Forecast a single FEV evaluation window.

        The call mirrors :meth:`forecast_gluon`: convert the external dataset
        representation into :class:`TimeseriesType`, then delegate batching,
        prediction and output rendering to the common forecast path. Use
        ``output_type="fev"`` to return predictions in the format accepted by
        ``fev.Task.evaluation_summary``. Pass ``return_inference_time=True`` to
        also return the model-only prediction time, excluding FEV input
        conversion and final ``DatasetDict`` construction.
        """
        assert batch_size >= 1, "Batch size must be >= 1"
        try:
            import fev  # noqa: F401
        except ImportError:
            raise ValueError("forecast_fev needs fev but fev is not available (not installed)!")

        data_kwargs = data_kwargs or {}
        timeseries, meta = build_fev_timeseries(window, **data_kwargs)
        if output_type == "fev":
            requested_quantiles = quantile_levels if quantile_levels is not None else self._quantile_levels()
            for item in meta:
                item["quantile_levels"] = requested_quantiles

        return _gen_forecast(
            self.model,
            timeseries,
            meta,
            prediction_length,
            output_type,
            batch_size,
            yield_per_batch,
            self._quantile_levels(),
            return_inference_time=return_inference_time,
            **predict_kwargs,
        )
