# Forecasting

## The `TimeseriesType` input

Every forecast call takes a list of [`TimeseriesType`][tirex2.model.types.TimeseriesType]
objects — one per series in the batch. Each holds:

- `target`: tensor of shape `(num_target_variates, context_length)`.
- `past_covariates`: `None`, or a tensor of shape `(num_past_covariates, context_length)`.
- `future_covariates`: `None`, or a tensor of shape
  `(num_future_covariates, >= context_length + prediction_length)` (extra trailing steps
  beyond what's needed are ignored). See [Covariates](covariates.md) for a full walkthrough.

A single-variate `target` (a plain 1D series) is still passed as a 2D tensor with
`num_target_variates == 1` — the same model and API path serve both univariate and
multivariate forecasting.

## Loading the model

```python
from tirex2 import load_model

model = load_model("NX-AI/TiRex-2", device="cpu")  # or device="cuda"
```

`load_model` returns a [`ForecastModel`][tirex2.api_adapter.forecast.ForecastModel] wrapping
the backbone; unknown attributes fall through to the underlying model, so
`model.quantiles` and similar backbone attributes remain reachable directly.

## Univariate forecasting

```python
import torch
from tirex2 import TimeseriesType

context = torch.randn(1, 512)  # (num_target_variates=1, context_length)
ts = TimeseriesType(target=context, past_covariates=None, future_covariates=None)

forecast = model.forecast([ts], prediction_length=64, output_type="numpy")[0]
# forecast.shape == (1, 9, 64)  -> (num_target_variates, num_quantiles, prediction_length)
```

## Multivariate forecasting

Pass a target with more than one row to forecast several variates jointly from a single
checkpoint — no separate model or per-variate training is needed:

```python
context = torch.randn(3, 512)  # 3 target variates sharing one context window
ts = TimeseriesType(target=context, past_covariates=None, future_covariates=None)

forecast = model.forecast([ts], prediction_length=64, output_type="numpy")[0]
# forecast.shape == (3, 9, 64)
```

## Batching multiple series

`forecast` accepts a list of `TimeseriesType` — each entry can have a different number of
variates and a different context length, and covariates are optional per series:

```python
forecasts = model.forecast([ts_a, ts_b, ts_c], prediction_length=64, output_type="numpy")
# forecasts is a list, one entry per input series
```

Internally, series are grouped into batches of at most `batch_size` (default `512`); on a
CUDA out-of-memory error the batch size is automatically halved and the failing batch
retried, without affecting the rest of the call.

## Output types

`output_type` controls the returned format:

| `output_type` | Returns | Requires |
| :------------ | :------ | :------- |
| `"torch"` (default) | list of `torch.Tensor`, shape `(V, 9, H)` | — |
| `"numpy"` | list of `numpy.ndarray`, shape `(V, 9, H)` | — |
| `"gluonts"` | list of GluonTS `QuantileForecast` | `pip install "tirex-2[gluonts]"` |
| `"fev"` | a `datasets.DatasetDict` for `fev.Task.evaluation_summary` | `pip install "tirex-2[fev]"` |

The 9 quantiles are the levels `0.1, 0.2, ..., 0.9`, with index `4` being the median.

## Test-time augmentation options

Extra keyword arguments passed to `forecast(...)` are forwarded to the backbone's
`predict`:

- `tta_sign_flip: bool` — opt-in sign-flip test-time augmentation: the model is also run on
  the sign-flipped input and the two passes are averaged in level space. Roughly doubles
  inference cost. Defaults to the checkpoint's configured setting when omitted.
- `tta_diff: bool` — opt-in differencing inside the postprocessor. Defaults to the
  checkpoint's configured setting when omitted.

```python
forecast = model.forecast([ts], prediction_length=64, tta_sign_flip=True)
```

## GluonTS and FEV integration

For GluonTS datasets, use
[`forecast_gluon`][tirex2.api_adapter.forecast.ForecastModel.forecast_gluon] instead of
building `TimeseriesType` objects by hand; it extracts targets, covariates, and item metadata
from each dataset entry directly. For FEV evaluation windows, use
[`forecast_fev`][tirex2.api_adapter.forecast.ForecastModel.forecast_fev]. See
[Benchmarks](../benchmarks.md) for the full GIFT-Eval and fev-bench reproduction paths that
build on these.
