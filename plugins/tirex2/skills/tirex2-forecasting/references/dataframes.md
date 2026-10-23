# Dataframe forecasting

Reference: [dataframe guide](https://nx-ai.github.io/tirex-2/how-to/dataframes/). These examples use pandas; the API also accepts supported eager dataframe libraries and normally preserves the input library.

## One series and a known calendar covariate

This runnable example uses synthetic daily observations. Replace the history and horizon with the user's data; retain explicit target and time column names.

```python
import numpy as np
import pandas as pd
from tirex2 import load_model

horizon = 14
history = pd.DataFrame({
    "timestamp": pd.date_range("2025-01-01", periods=128, freq="D"),
    "sales": 20 + 3 * np.sin(np.arange(128) * 2 * np.pi / 7),
})
history["weekday"] = history["timestamp"].dt.dayofweek.astype("float32")
future = pd.DataFrame({
    "timestamp": pd.date_range(
        history["timestamp"].iloc[-1] + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    ),
})
future["weekday"] = future["timestamp"].dt.dayofweek.astype("float32")

model = load_model("NX-AI/TiRex-2", device="cpu")
result = model.forecast_df(
    history,
    prediction_length=horizon,
    target="sales",
    timestamp_column="timestamp",
    future_covariates=["weekday"],
    future_df=future,
)
assert len(result) == horizon
assert result["timestamp"].tolist() == future["timestamp"].tolist()
assert {"target", "prediction", "0.1", "0.5", "0.9"} <= set(result.columns)
```

All arguments after `prediction_length` are keyword-only. Without covariates, omit both `future_covariates` and `future_df`. For past-only features, pass their history column names as `past_covariates`.

Future-known columns must appear in both history and future data. For each id, supply the full forecast horizon with aligned timestamps and covariate values. Future target observations do not belong in the context. A feature available only in future needs historical values too; do not fabricate those values silently.

## Independent series versus joint targets

For independent series, add an id column. Continuing the example above:

```python
long_history = pd.concat([
    history.assign(item_id="A"),
    history.assign(item_id="B", sales=history["sales"] * 2),
], ignore_index=True)
result = model.forecast_df(
    long_history,
    prediction_length=horizon,
    target="sales",
    id_column="item_id",
    timestamp_column="timestamp",
)
assert result.groupby("item_id").size().to_dict() == {"A": horizon, "B": horizon}
```

Each id is forecast independently. To forecast related signals jointly within each id, set `target=["sales", "returns"]` when those columns exist. If target columns should be independent, reshape to separate ids instead. Explicitly select targets so numeric metadata is not accidentally forecast: omitting `target` selects all eligible numeric columns.

The adapter sorts timestamps within ids. A pandas `DatetimeIndex` can supply the time axis when `timestamp_column` is omitted. Inspect frequency inference, especially for gaps and calendar schedules; without a usable time axis, output may use integer positions.

## Output and large batches

Output is long format: one row per id, target, and forecast step, with `prediction` (median) and quantile columns named `"0.1"` through `"0.9"`. Check uniqueness and completeness of these keys before joining to held-out actuals.

Keep `prediction_length` within `model.future_len`. Larger requests are warned about and truncated, so always verify the returned horizon.

`forecast_df` accepts `output_type="dataframe"` (default) or `"pandas"`; use the tensor API for NumPy or Torch outputs. With `yield_per_batch=True`, iterate over returned frames instead of treating the result as one dataframe. `batch_size` is the number of series in a batch.
