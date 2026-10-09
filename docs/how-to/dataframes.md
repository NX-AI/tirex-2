# Forecasting from a DataFrame

Use [`ForecastModel.forecast_df`][tirex2.api_adapter.forecast.ForecastModel.forecast_df] to forecast
directly from a dataframe. You don't need to create
[`TimeseriesType`][tirex2.model.types.TimeseriesType] tensors yourself. The method uses
[narwhals](https://narwhals-dev.github.io/narwhals/) to work with eager dataframes from pandas,
Polars, PyArrow, Modin, and cuDF. It returns the forecast in the same dataframe library you used
for the input.

`narwhals` is installed with TiRex-2, so `forecast_df` is always available. Install the dataframe
library you want to use separately, for example with `pip install pandas`, `pip install polars`, or
`pip install pyarrow`.


## Downloading data

We'll use the `aus_production` dataset from
[Forecasting: Principles and Practice, the Pythonic Way](https://otexts.com/fpppy/data/). It has 218
quarterly observations of Australian beer, cement, electricity, and gas production, from 1956 Q1 to
2010 Q2.

Run the following command to download the CSV file:

```bash
curl -O https://otexts.com/fpppy/data/aus_production.csv
```

Load it into a pandas frame and drop the two series that stop early:

```python
import pandas as pd

df = (
    pd.read_csv("aus_production.csv", parse_dates=["ds"])
    .rename(columns={"ds": "timestamp"})
    .drop(columns=["Tobacco", "Bricks"])  # these two series stop early
)
# columns: timestamp, Beer, Cement, Electricity, Gas
```

## Loading the model

```python
from tirex2 import load_model

model = load_model("NX-AI/TiRex-2", device="cpu")  # or device="cuda"/"mps"
```

## Forecasting from a DataFrame

### Forecasting a single column

Simply point `target` at the column to forecast and `timestamp_column` at the time axis:

```python
forecast = model.forecast_df(
    df,
    prediction_length=8,  # eight quarters
    target="Beer",
    timestamp_column="timestamp",
)
```

!!! note
    When calling `forecast_df`, pass every argument after `prediction_length` by name.

You get a long-format frame — one row per target column and forecast step — with a `prediction`
column (the median) and one column per quantile level:

|   | timestamp  | target | prediction | 0.1     | … | 0.9     |
|---|------------|--------|------------|---------|---|---------|
| 0 | 2010-07-01 | Beer   | 408.48     | 391.33  | … | 425.11  |
| 1 | 2010-10-01 | Beer   | 479.12     | 458.42  | … | 500.15  |
| 2 | 2011-01-01 | Beer   | 405.48     | 384.93  | … | 424.99  |

The input ends in 2010 Q2, so the forecast starts in 2010 Q3. `forecast_df` picks up the quarterly
schedule from the input timestamps and continues it. It also understands calendar schedules such as
month ends. If there is no usable time axis, the forecast uses integer positions instead.

If your pandas dataframe has a `DatetimeIndex`, you can leave out `timestamp_column`. The index is
used automatically, including for `future_df`.

### Forecasting multiple columns

Leave `target` unset and every numeric column that is not the id, timestamp or a covariate becomes
a target. They are forecast **jointly**, so the model can use their cross-variate structure:

```python
forecast = model.forecast_df(
    df,
    prediction_length=8,
    timestamp_column="timestamp",
)
forecast["target"].unique().tolist()
# ['Beer', 'Cement', 'Electricity', 'Gas']
```

Use separate columns in a wide dataframe when the series are related. If each column represents a
different SKU, sensor, or customer, reshape the data into long format and give each series its own
id, as shown next.

### Forecasting many series in one frame

To forecast several series at once, stack them in one dataframe and give each series an id. Pass
the name of that column as `id_column`:

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

Each `product` is forecast as its own series, and the output includes the `product` column. Products
stay in the order they first appear in the input, while rows within each product are sorted by
timestamp. You don't need to sort the input first.

If an id has multiple target columns, the model forecasts those columns together. It never combines
data from different ids. Here, each id has just one target column, so the products are forecast
independently.

### Forecasting with covariates

`past_covariates` names columns observed only over the context — see
[Covariates](covariates.md) for what the model does with them:

```python
forecast = model.forecast_df(
    df,
    prediction_length=8,
    target="Beer",
    timestamp_column="timestamp",
    past_covariates=["Cement", "Electricity", "Gas"],
)
```

Use `future_covariates` for columns whose values you already know during the forecast period, such
as calendar features, holidays, or promotions. Put those future values in `future_df`, using the
same layout as the input dataframe, and past values in the input dataframe:

```python
df["quarter"] = df["timestamp"].dt.quarter.astype("float32")

future_df = pd.DataFrame({"timestamp": pd.date_range("2010-07-01", periods=8, freq="QS")})
future_df["quarter"] = future_df["timestamp"].dt.quarter.astype("float32")

forecast = model.forecast_df(
    df,
    prediction_length=8,
    target="Beer",
    timestamp_column="timestamp",
    future_covariates=["quarter"],
    future_df=future_df,
)
```

`future_df` needs a row for every series you are forecasting. Series it contains that `df` never
mentions are ignored, with a warning.

## Scaling up

`forecast_df` processes up to `batch_size` series at a time (512 by default). You can pass thousands
of series in one call; it handles the batches for you.

!!! note
    For each batch, TiRex-2 packs the series into one tensor and pads shorter histories on the left
    to match the longest one. This means a larger
    `batch_size` can use more GPU memory. If CUDA or MPS runs out of memory, `forecast_df` halves the
    batch size and retries, so the value you set is a starting point.

`batch_size` counts **series, not rows**. For example, 100 ids count as 100 series whether each id
has one target column or four.

If the full forecast would be too large to hold in memory, set `yield_per_batch=True`. This yields
one dataframe per batch instead of combining them all into one:

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

Each yielded frame holds the ids of one batch, in input order, so you can write it straight out
without waiting for the rest.

## Switching dataframe libraries

For example, pass in a Polars dataframe and you get a Polars dataframe back:

```python
import polars as pl

pl_df = pl.read_csv(
    "aus_production.csv", try_parse_dates=True, null_values="NA"
).drop("Tobacco", "Bricks")

forecast = model.forecast_df(
    pl_df, prediction_length=8, target="Beer", timestamp_column="ds"
)
type(forecast)  # polars.DataFrame
```

`pyarrow.csv.read_csv` works the same way and returns a `pyarrow.Table`.

## Supported output types

A dataframe call returns a dataframe. `output_type` accepts exactly two values:

| `output_type` | Returns |
| :------------ | :------ |
| `"dataframe"` (default) | one long-format frame, in the input's dataframe library |
| `"pandas"` | the same frame, always as pandas |

Anything else raises a `ValueError`.
