# Covariates

TiRex-2 natively conditions its forecast on two kinds of covariates, in addition to the
target history. Both are optional and independent of each other and of whether the target
itself is univariate or multivariate.

## Past covariates

`past_covariates` are known only up to the current time — like the target itself, they stop
at the end of the context window. Shape: `(num_past_covariates, context_length)`, matching
the target's `context_length` exactly.

```python
import torch
from tirex2 import TimeseriesType

context_length = 512
target = torch.randn(1, context_length)
past_covariates = torch.randn(2, context_length)  # 2 past-only covariates

ts = TimeseriesType(target=target, past_covariates=past_covariates, future_covariates=None)
forecast = model.forecast([ts], prediction_length=prediction_length, output_type="numpy")[0]
```

## Future-known covariates

`future_covariates` are known ahead of time for the whole forecast horizon — calendar
features, holidays, promotions, or scheduled interventions are typical examples. Shape:
`(num_future_covariates, context_length + prediction_length)`; if you pass more steps than
`context_length + prediction_length`, the extra trailing steps are ignored.

```python
prediction_length = 64
future_covariates = torch.zeros(1, context_length + prediction_length)
future_covariates[0, 100::7] = 1.0  # e.g. a weekly recurring event flag

ts = TimeseriesType(target=target, past_covariates=None, future_covariates=future_covariates)
forecast = model.forecast([ts], prediction_length=prediction_length, output_type="numpy")[0]
```

## Combining both

Past and future covariates can be combined freely on the same series:

```python
ts = TimeseriesType(
    target=target,
    past_covariates=past_covariates,
    future_covariates=future_covariates,
)
```

## Worked example: a non-stationary series with two covariates

The [`Demo`][tirex2.demo.Demo] class used in the [Quickstart](../getting-started/quickstart.md)
builds exactly this kind of input — a continuous future-known driver that sets a wandering
baseline level, plus a binary future-known promotion flag that adds spikes:

```python
import torch
from tirex2 import TimeseriesType
from tirex2.demo import Demo

demo_nonstationary = Demo.create_nonstationary_demo()
# univariate target shape: (1, context_length)
target = torch.from_numpy(demo_nonstationary.target_context).unsqueeze(0)

# future-known covariates shape: (n_covariates, context_length + horizon)
future_covariates = torch.from_numpy(
    np.stack([np.concatenate([c.context, c.future]) for c in demo_nonstationary.covariates]).astype(np.float32)
)

# multivariate forecast conditioning: the target plus future-known covariates.
multivariate_nonstationary = TimeseriesType(
    target=target,
    past_covariates=None,
    future_covariates=future_covariates,
)

forecast = model.forecast(
    timeseries=[multivariate_nonstationary],
    prediction_length=42,
    output_type="numpy",
)[0]

# forecast.shape == (1, 9, 42)  -> (num_target_variates, num_quantiles, prediction_length)
```
![Multivariate context and forecast, with future-known covariates plotted below](../images/multivariate-prediction.png)
