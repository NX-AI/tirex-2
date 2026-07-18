# Streaming

TiRex-2's README describes it as "built on a recurrent architecture designed for efficient
streaming settings." That describes the *architecture choice*, not a capability exposed by
this open-source release — read carefully, since it's easy to conflate the two.

## What this open-source release actually does

Every call to
[`ForecastModel.forecast`][tirex2.api_adapter.forecast.ForecastModel.forecast] (and the
underlying `TiRex2.predict`) recomputes the forecast **from scratch over the full context
array you pass in**. There is no stateful, incremental call that lets you feed only the new
observations since your last call:

- `TiRex2.forward` initializes fresh block state on every call
  (`state = {i: None for i in range(len(self.stack))}`) — nothing carries over between calls.
- `TiRex2._predict_once` pads/truncates its input to the model's fixed
  `context_len + future_len` window and runs the full stack over it every time.

So to get an updated forecast as new data points arrive, you re-call `forecast` with the
target tensor extended by the new observations (and re-run over the whole, now-longer or
truncated, context window) — there is no way to avoid recomputing over the full history in
this release.

```python
# Every call below recomputes over its full context; nothing is cached between calls.
forecast_t1 = model.forecast([ts_up_to_t1], prediction_length=32, output_type="numpy")[0]
# ... new observations arrive ...
forecast_t2 = model.forecast([ts_up_to_t2], prediction_length=32, output_type="numpy")[0]
```

## What's Pro-only

**Streaming** — incremental forecast updates as new observations arrive, without recomputing
over the full history — is listed explicitly as a [TiRex-2 Pro](../pro.md) capability. The
recurrent architecture used in this open-source release is what makes that incremental mode
possible in principle, but the incremental, no-recompute code path itself is not part of this
release. If you need it, see [TiRex-2 Pro](../pro.md).
