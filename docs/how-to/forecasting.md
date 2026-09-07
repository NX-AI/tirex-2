# Forecasting

## The `TimeseriesType` input

Every forecast call takes a list of [`TimeseriesType`][tirex2.model.types.TimeseriesType]
objects — one per series in the batch. Each holds:

- `target`: tensor of shape `(num_target_variates, context_length)`.
- `past_covariates`: `None`, or a tensor of shape `(num_past_covariates, context_length)`.
- `future_covariates`: `None`, or a tensor of shape
  `(n_future_covariates, context_length + prediction_length)` (extra trailing steps
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

`use_flex_attention` overrides every variate mixer's checkpoint setting. `True` enables block-sparse FlexAttention, which can reduce the cost of large grouped multivariate batches on CUDA but adds first-call compilation overhead. `False` forces dense attention. Leave it unset to keep the checkpoint configuration.

```python
model = load_model("NX-AI/TiRex-2", device="cuda", use_flex_attention=True)
```

## Univariate forecasting

```python
import torch
from tirex2 import TimeseriesType, load_model

# (num_target_variates=1, context_length)
context = torch.sin(torch.arange(128).float() / 8)
ts_univariate = TimeseriesType(target=context, past_covariates=None, future_covariates=None)

model = load_model("NX-AI/TiRex-2", device="cpu")

forecast = model.forecast([ts_univariate], prediction_length=64, output_type="numpy")[0]
# forecast.shape == (1, 9, 64)  -> (num_target_variates, num_quantiles, prediction_length)
```

![Sine-wave context and forecast produced by plot_multivariate](../images/sine-wave-prediction.png)

## Multivariate forecasting

The primary goal of multitarget forecasting is to model complex systems where multiple signals interact jointly, allowing the model to capture both the temporal structure within each individual time series and the cross-variate dependencies among them.

Pass a target with more than one row to forecast several variates jointly from a single
checkpoint — no separate model or per-variate training is needed:


```python
import torch
from tirex2 import TimeseriesType, load_model
from tirex2.demo import Demo

demo_nonstationary = Demo.create_nonstationary_demo()
demo_holidays = Demo.create_holidays_demo()

# Stack variates together
multi_target = torch.stack(
    [
        torch.from_numpy(demo_holidays.target_context),
        torch.from_numpy(demo_nonstationary.target_context),
    ]
)

multi_target_ts = TimeseriesType(
    target=multi_target,
    past_covariates=None,
    future_covariates=None,
)

model = load_model("NX-AI/TiRex-2", device="cpu")

multi_target_forecast = model.forecast(
    [multi_target_ts],
    prediction_length=42,
    output_type="numpy",
    batch_size=1,
)[0]

# forecast.shape == (2, 9, 42)  -> (num_target_variates, num_quantiles, prediction_length)
```

See [Covariates](covariates.md) for past vs. future-known covariates.


## Batching multiple series

`forecast` accepts a list of `TimeseriesType` — each entry can have a different number of
variates and a different context length, and covariates are optional per series:

```python
import torch
from tirex2 import TimeseriesType, load_model

context_length_a = 128
ts_a = torch.randn(1, context_length_a)
ts_a = TimeseriesType(
    target=ts_a,
    past_covariates=None,
    future_covariates=None,
)

context_length_b = 128
ts_b = torch.randn(1, context_length_b)
ts_b = TimeseriesType(
    target=ts_b,
    past_covariates=None,
    future_covariates=None,
)

context_length_c = 128
ts_c = torch.randn(1, context_length_c)
ts_c = TimeseriesType(
    target=ts_c,
    past_covariates=None,
    future_covariates=None,
)

model = load_model("NX-AI/TiRex-2", device="cpu")
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
