# Forecasting from a DataFrame

tirex2.api_adapter.forecast.ForecastModel.forecast_df forecasts a dataframe
directly, with no need to build [`TimeseriesType`][tirex2.model.types.TimeseriesType] tensors by
hand. Frames are handled through [narwhals](https://narwhals-dev.github.io/narwhals/), so any
eager dataframe it supports  pandas, Polars, PyArrow, Modin, cuDF  works, and the forecast comes
back **in the same library the input came from**.

narwhals is a core dependency, so `forecast_df` is always available; bring your own dataframe
library with `pip install "tirex-2[pandas]"` (or `[polars]`, `[pyarrow]`).

## The example data

`aus_production` from
[Forecasting: Principles and Practice, the Pythonic Way](https://otexts.com/fpppy/data/) — 218
quarterly observations (1956 Q1 – 2010 Q2) of Australian production: beer, cement, electricity and
gas.

```bash
curl -O https://otexts.com/fpppy/data/aus_production.csv
```

```python
import pandas as pd

df = (
    pd.read_csv("aus_production.csv", parse_dates=["ds"])
    .rename(columns={"ds": "timestamp"})
    .drop(columns=["Tobacco", "Bricks"])   # these two series stop early
)
# columns: timestamp, Beer, Cement, Electricity, Gas
```

## Univariate

Point `target` at one column and `timestamp_column` at the time axis:

```python
from tirex2 import load_model

model = load_model("NX-AI/TiRex-2", device="cpu")  # or device="cuda"

forecast = model.forecast_df(
    df,
    prediction_length=8,        # eight quarters
    target="Beer",
    timestamp_column="timestamp",
)
```

The result is a  frame  one row per target column and forecast step  with a
`prediction` column (the median) and one column per quantile level:

|   | timestamp  | target | prediction | 0.1     | … | 0.9     |
|---|------------|--------|------------|---------|---|---------|
| 0 | 2010-07-01 | Beer   | 408.48     | 391.33  | … | 425.11  |
| 1 | 2010-10-01 | Beer   | 479.12     | 458.42  | … | 500.15  |
| 2 | 2011-01-01 | Beer   | 405.48     | 384.93  | … | 424.99  |

The context ends in 2010 Q2, and forecast timestamps continue the quarterly frequency inferred
from the input — inference understands calendar offsets like quarters and month ends, not just
fixed timedeltas. Without a usable time axis they fall back to integer positions.

## Multivariate

Leave `target` unset and every numeric column that is not the id, timestamp or a covariate becomes
a target. `multivariate=True` forecasts them **jointly**, so the model can use the cross-variate
structure:

```python
forecast = model.forecast_df(
    df,
    prediction_length=8,
    timestamp_column="timestamp",
    multivariate=True,
)
forecast["target"].unique().tolist()
# ['Beer', 'Cement', 'Electricity', 'Gas']
```

Without the flag (the default) each column is forecast as an independent univariate series.

## Many series in one frame

Long format panels  many series stacked, identified by an id column  are the common shape:

```python
long_df = df.melt(id_vars="timestamp", var_name="product", value_name="production")

forecast = model.forecast_df(
    long_df,
    prediction_length=8,
    target="production",
    id_column="product",
    timestamp_column="timestamp",
)
```

Each `product` becomes its own series and the id column is carried into the output. Series keep
their first appearance order and rows are sorted by timestamp, so the input need not be pre-sorted.

## Covariates

`past_covariates` names columns observed only over the context:

```python
forecast = model.forecast_df(
    df,
    prediction_length=8,
    target="Beer",
    timestamp_column="timestamp",
    past_covariates=["Cement", "Electricity", "Gas"],
)
```

`future_covariates` names columns known ahead  calendar features, holidays, promotions. They need
their horizon values in a second frame of the same layout, passed as `future_df`

## Batching

Every series in the frame is grouped into batches of at most `batch_size` (default `512`), so a
frame with thousands of series is one call rather than a Python loop. On a CUDA or MPS
out-of-memory error the batch size is halved and the failing batch retried.

For datasets too large to hold at once, `yield_per_batch=True` yields one frame per batch instead
of concatenating them:

```python
for batch in model.forecast_df(
    long_df,
    prediction_length=8,
    target="production",
    id_column="product",
    timestamp_column="timestamp",
    batch_size=2,
    yield_per_batch=True,
):
    write_somewhere(batch)
```

## Other dataframe libraries

The input library decides the output library — nothing else changes:

```python
import polars as pl

pl_df = pl.read_csv(
    "aus_production.csv", try_parse_dates=True, null_values="NA"
).drop("Tobacco", "Bricks")

forecast = model.forecast_df(
    pl_df, prediction_length=8, target="Beer", timestamp_column="ds"
)
type(forecast)   # polars.DataFrame
```

`pyarrow.csv.read_csv` works the same way and returns a `pyarrow.Table`. Pass
`output_type="pandas"` to always get pandas back. A pandas frame with a `DatetimeIndex` needs no
`timestamp_column` — the index is used automatically.

## Output types

`output_type` defaults to `"dataframe"`. The other formats from
[Forecasting](forecasting.md#output-types) work too:

| `output_type` | Returns |
| :------------ | :------ |
| `"dataframe"` (default) | one long-format frame, in the input's dataframe library |
| `"pandas"` | the same frame, always as pandas |
| `"torch"` / `"numpy"` | a list of `(V, 9, H)` arrays, one per series |
| `"gluonts"` | a list of `QuantileForecast`, starting at the first forecast timestamp |

## API reference

- [`ForecastModel.forecast_df`][tirex2.api_adapter.forecast.ForecastModel.forecast_df]
