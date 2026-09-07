# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastmcp import FastMCP
from pydantic import BaseModel

from app.config import Settings
from app.model import Tirex2Model
from tirex2 import TimeseriesType

settings = Settings()
model = Tirex2Model(settings)
model.warmup()


app = FastAPI(title="TiRex V2 API")

MEDIAN_QUANTILE_INDEX = 4


@app.get("/health")
def health():
    return {"message": "OK"}


@app.exception_handler(Exception)
def api_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error_code": 500, "error_message": exc.__str__()})


def _to_tensor(x):
    return torch.tensor(x, dtype=torch.float32) if x is not None else None


class Series(BaseModel):
    target: list[list[float]] = [[0, 1, 2, 3]]
    past_covariates: list[list[float]] | None = None
    future_covariates: list[list[float]] | None = None

    def to_timeseries(self) -> TimeseriesType:
        return TimeseriesType(
            target=_to_tensor(self.target),
            past_covariates=_to_tensor(self.past_covariates),
            future_covariates=_to_tensor(self.future_covariates),
        )


# --------------------------------------------------------------------------- #
# Univariate API
#
# Each series is a single 1D target sequence with no covariates. The forecast
# output drops the variate dimension so callers get plain per-series results.
# return shape mean:     [series][timestep]
# return shape quantile: [series][quantile][timestep]
# --------------------------------------------------------------------------- #


class UnivariateForecastRequest(BaseModel):
    context: list[list[float]] = [[0, 1, 2, 3]]  # batch of univariate target series
    prediction_length: int = 32

    def to_timeseries(self) -> list[TimeseriesType]:
        # Wrap each 1D series as a single-variate target: [1, T].
        return [
            TimeseriesType(target=_to_tensor([s]), past_covariates=None, future_covariates=None) for s in self.context
        ]


@app.post("/univariate/forecast/mean")
def forecast_mean(req: UnivariateForecastRequest) -> list[list[float]]:
    forecasts = model.predict(req.to_timeseries(), req.prediction_length)
    # forecasts: [variate, quantile, timestep]; single variate -> f[0].
    return [f[0, MEDIAN_QUANTILE_INDEX, :].tolist() for f in forecasts]


@app.post("/univariate/forecast/quantiles")
def forecast_quantiles(req: UnivariateForecastRequest) -> list[list[list[float]]]:
    forecasts = model.predict(req.to_timeseries(), req.prediction_length)
    # forecasts: [variate, quantile, timestep]; single variate -> f[0].
    return [f[0].tolist() for f in forecasts]


# --------------------------------------------------------------------------- #
# Multivariate API
#
# Each series carries a multivariate target [V, T] plus optional past/future
# covariates. The variate dimension is preserved in the output.
# return shape mean:     [series][variate][timestep]
# return shape quantile: [series][variate][quantile][timestep]
# --------------------------------------------------------------------------- #


class MultivariateForecastRequest(BaseModel):
    context: list[Series] = [Series()]
    prediction_length: int = 32


@app.post("/multivariate/forecast/mean")
def multivariate_forecast_mean(req: MultivariateForecastRequest) -> list[list[list[float]]]:
    contexts = [s.to_timeseries() for s in req.context]
    forecasts = model.predict(contexts, req.prediction_length)
    # forecasts: [variate, quantile, timestep]; keep all variates -> f[:, ...].
    return [f[:, MEDIAN_QUANTILE_INDEX, :].tolist() for f in forecasts]


@app.post("/multivariate/forecast/quantiles")
def multivariate_forecast_quantiles(req: MultivariateForecastRequest) -> list[list[list[list[float]]]]:
    contexts = [s.to_timeseries() for s in req.context]
    forecasts = model.predict(contexts, req.prediction_length)
    # forecasts: [variate, quantile, timestep]; keep all variates -> f.
    return [f.tolist() for f in forecasts]


# --------------------------------------------------------------------------- #
# MCP API
# --------------------------------------------------------------------------- #


mcp = FastMCP("TiRex MCP")

disclaimer = (
    "Disclaimer: NXAI is not responsible for any incorrect interpretations of the "
    "forecasted values by LLMs. Check the TiRex-2 license for more details: "
    "https://github.com/NX-AI/tirex-2\n\n"
)


@mcp.tool()
async def tirex_model(context: list[float], prediction_length: int) -> str:
    """Use the TiRex model to forecast time series data."""
    # MCP is the only API that isn't batched: a single univariate series.

    input_length = len(context)
    ts = TimeseriesType(target=_to_tensor([context]), past_covariates=None, future_covariates=None)
    forecasts = model.predict([ts], prediction_length)
    # forecast: [variate, quantile, timestep]; single variate -> [0], median quantile.
    mean = forecasts[0][0, MEDIAN_QUANTILE_INDEX, :].tolist()

    return (
        "TiRex Forecast Results:\n"
        f"Input data length: {input_length}\n"
        f"Prediction length: {prediction_length}\n\n"
        f"Forecasted values: {mean}\n\n"
        f"{disclaimer}"
    )


@mcp.tool()
async def tirex_model_multivariate(
    target: list[list[float]],
    prediction_length: int,
    past_covariates: list[list[float]] | None = None,
    future_covariates: list[list[float]] | None = None,
) -> str:
    """Use the TiRex model to forecast multivariate time series data.

    `target` is a list of variates, each a list of values over time ([V, T]).
    Optional `past_covariates` / `future_covariates` follow the same [V, T] layout.
    """
    # MCP is the only API that isn't batched: a single multivariate series.

    input_length = len(target[0]) if target else 0
    ts = TimeseriesType(
        target=_to_tensor(target),
        past_covariates=_to_tensor(past_covariates),
        future_covariates=_to_tensor(future_covariates),
    )
    forecasts = model.predict([ts], prediction_length)
    # forecast: [variate, quantile, timestep]; take the median quantile per variate.
    mean = forecasts[0][:, MEDIAN_QUANTILE_INDEX, :].tolist()

    return (
        "TiRex Multivariate Forecast Results:\n"
        f"Number of variates: {len(mean)}\n"
        f"Input data length: {input_length}\n"
        f"Prediction length: {prediction_length}\n\n"
        f"Forecasted values (per variate): {mean}\n\n"
        f"{disclaimer}"
    )


mcp_app = mcp.http_app(path="/")
app.router.lifespan_context = mcp_app.router.lifespan_context
app.mount("/mcp", mcp_app)
