# Tensor forecasting

References: [forecasting](https://nx-ai.github.io/tirex-2/how-to/forecasting/) and [covariate](https://nx-ai.github.io/tirex-2/how-to/covariates/) guides.

## Input contract

For context length `T` and horizon `H`, use floating-point tensors with time on the last axis:

| Field | Shape | Meaning |
| --- | --- | --- |
| `target` | `(V, T)` | Observed target history; use `(1, T)` for one target |
| `past_covariates` | `(P, T)` or `None` | Features available only through the cutoff |
| `future_covariates` | `(F, T + H)` or `None` | Features known across history and forecast horizon |

For NumPy data stored as `(time, variables)`, transpose before conversion. Supply all three dataclass fields, using `None` for absent covariates. Longer future-covariate tensors are accepted but extra trailing steps are ignored. Align all columns to the same time grid before constructing tensors.

Use `1 <= H <= model.future_len`. Larger horizons are warned about and truncated to the checkpoint's limit; check the returned length.

## Minimal forecast

```python
import numpy as np
import torch
from tirex2 import TimeseriesType, load_model

context_length, horizon = 128, 16
target = torch.sin(torch.arange(context_length, dtype=torch.float32) / 8).unsqueeze(0)
series = TimeseriesType(
    target=target,
    past_covariates=None,
    future_covariates=None,
)
model = load_model("NX-AI/TiRex-2", device="cpu")
forecast = model.forecast(
    [series], prediction_length=horizon, output_type="numpy",
)[0]
levels = np.asarray([float(q) for q in model.quantiles])
median_index = np.flatnonzero(np.isclose(levels, 0.5)).item()
median = forecast[:, median_index, :]
assert forecast.shape == (1, len(levels), horizon)
assert np.isfinite(forecast).all()
```

The result is a list in input order; each entry has shape `(target_variates, quantiles, horizon)`. It is not a `(quantiles, mean)` tuple. `output_type="torch"` is the default. Keep target names and timestamps separately when using tensors so outputs can be labeled accurately.

## Multiple targets and covariates

Extend the example with two related signals and a future-known periodic feature:

```python
time = torch.arange(context_length + horizon, dtype=torch.float32)
joint_targets = torch.stack([
    torch.sin(time[:context_length] / 8),
    torch.cos(time[:context_length] / 8),
])
known_feature = (time.remainder(7) == 0).float().unsqueeze(0)
joint_series = TimeseriesType(
    target=joint_targets,
    past_covariates=None,
    future_covariates=known_feature,
)
joint_forecast = model.forecast(
    [joint_series], prediction_length=horizon, output_type="numpy",
)[0]
assert joint_forecast.shape == (2, len(levels), horizon)
```

For past-only covariates, replace `None` with a `(P, T)` tensor. For independent series, pass separate `TimeseriesType` objects in the list; they may have different history lengths and target counts. Never concatenate independent entities along the time axis.

The standard context length is 2,048. Default inference pads short contexts with NaN and handles its context window internally; inspect checkpoint settings before assuming all supplied history is used. Missing observations should follow a documented data policy, not be silently replaced with zeros. Model padding does not resolve duplicate timestamps, irregular sampling, or absent future feature values.
