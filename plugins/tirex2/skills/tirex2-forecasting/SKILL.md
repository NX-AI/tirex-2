---
name: tirex2-forecasting
description: Forecast time series with TiRex-2 using Python dataframes or tensors, including multiple targets, past and future-known covariates, quantile forecasts, and chronological evaluation. Use when the user requests TiRex-2 forecasting, integration, or inference troubleshooting; not for the separate original TiRex package.
---

# Time series forecasting with TiRex-2

Use `tirex-2` (Python: `tirex2`) for zero-shot forecasting with `NX-AI/TiRex-2`; no training required.

## Establish the forecast

Inspect the data to identify timestamps, frequency, series ids, targets, cutoff, and horizon in **time steps**. Separate joint targets from independent series.

Check duplicates, gaps, numeric types, and time zones. Preserve the time grid and document resampling or imputation. Use only information available at the cutoff, including for preprocessing; future covariates must be known then, such as calendar features or committed schedules.

## Environment and checkpoint

Use Python 3.11+. Always install `tirex-2` and the required dataframe library in the project's environment.

Load the model once and reuse it:

```python
from tirex2 import load_model

model = load_model("NX-AI/TiRex-2", device="cpu")
```

Prefer a supplied local checkpoint when available. Start small on CPU; use `cuda` or `mps` when supported. Report checkpoint access errors. For CUDA setup, see the [FAQ](https://nx-ai.github.io/tirex-2/faq/).

## Choose the API

- **Tabular input:** Prefer `model.forecast_df`. Read [dataframe examples](references/dataframes.md) for ids, multiple targets, future values, and output alignment.
- **Arrays/tensors:** Use `TimeseriesType` and `model.forecast`. Read [tensor examples](references/tensors.md) for shapes and covariates.

Resolve API differences using the installed version, [documentation](https://nx-ai.github.io/tirex-2/), and [API reference](https://nx-ai.github.io/tirex-2/api/forecast/).

## Validate and deliver

Check output ids, targets, horizon, and timestamps. Confirm quantile order with `model.quantiles`: standard levels are 0.1–0.9, with 0.5 as the median and 0.1–0.9 forming a nominal 80% prediction interval.

For evaluation, use a chronological holdout or rolling origins without future leakage. Align actuals by id, target, and timestamp; compare against naive or seasonal baselines using MAE/RMSE. Assess uncertainty with quantile loss and interval coverage; avoid MAPE with zeros.

Deliver labeled forecasts and record checkpoint, version, device, cutoff, horizon, and preprocessing or inference overrides. State what actually ran and any execution blockers.

## Performance and repeated updates

Start with defaults and a small batch. `batch_size` counts series; automatic reduction on device OOM cannot fix an oversized single series. Use `yield_per_batch=True` to limit output memory.

Consider `load_model(..., compile=True)` for repeated shapes, allowing for compilation overhead. Validate the speed/accuracy tradeoff of `pad_context=False`; `tta_sign_flip=True` roughly doubles inference work.

Each open-source call recomputes over its context. For updates, append observations and realign covariates. Incremental streaming without recomputation is [PRO-only](https://nx-ai.github.io/tirex-2/how-to/streaming/).
