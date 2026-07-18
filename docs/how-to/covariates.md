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
```

## Future-known covariates

`future_covariates` are known ahead of time for the whole forecast horizon — calendar
features, holidays, promotions, or scheduled interventions are typical examples. Shape:
`(num_future_covariates, >= context_length + prediction_length)`; if you pass more steps than
`context_length + prediction_length`, the extra trailing steps are ignored.

```python
prediction_length = 64
future_covariates = torch.zeros(1, context_length + prediction_length)
future_covariates[0, 100::7] = 1.0  # e.g. a weekly recurring event flag

ts = TimeseriesType(target=target, past_covariates=None, future_covariates=future_covariates)
forecast = model.forecast([ts], prediction_length=prediction_length, output_type="numpy")[0]
```

## Combining both

Past and future covariates can be combined freely on the same series, and are independent of
the number of target variates:

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
from tirex2.demo import Demo

demo = Demo.create_nonstationary_demo()
ts = demo.to_timeseries_type(include_covariates=True)

print(ts.n_past_covariates, ts.n_future_covariates)  # 0, 2 — both covariates here are future-known
```

`Demo.to_timeseries_type(include_covariates=False)` builds the same series without any
covariates, which is useful for comparing forecasts with and without covariate information
side by side (as done in the [Quickstart](../getting-started/quickstart.md) covariate
example).

## HTTP API equivalent

The [deployment](../deployment.md) HTTP API exposes the same future-covariate conditioning
for the multivariate endpoints, e.g.:

```bash
curl -s -X POST "http://localhost:8000/multivariate/forecast/mean" \
  -H 'Content-Type: application/json' \
  -d '{
        "context": [{
          "target": [[1, 2, 3, 4, 5, 6, 7, 8]],
          "future_covariates": [[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]]
        }],
        "prediction_length": 5
      }'
```
