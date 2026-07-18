# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

import requests

from conftest import (
    MEDIAN_QUANTILE_INDEX,
    MULTITARGET_SERIES,
    MULTIVARIATE_SERIES,
    PREDICTION_LENGTH,
    REFERENCE,
    TARGET,
    assert_forecast_close,
)


def _post(api_server, path, context):
    response = requests.post(
        f"{api_server}{path}",
        json={"context": context, "prediction_length": PREDICTION_LENGTH},
    )
    assert response.status_code == 200, response.text
    return response.json()


# --- Univariate --------------------------------------- #


def test_univariate_mean(api_server):
    # return shape: [series][timestep]. For a single series the reference's
    data = _post(api_server, "/univariate/forecast/mean", [TARGET])
    assert_forecast_close(data, REFERENCE.univariate[:, MEDIAN_QUANTILE_INDEX])


def test_univariate_quantiles(api_server):
    # return shape: [series][quantile][timestep]
    data = _post(api_server, "/univariate/forecast/quantiles", [TARGET])
    assert_forecast_close(data, REFERENCE.univariate)


# --- Multivariate: target + future-known covariates ------------------------- #


def test_multivariate_mean(api_server):
    # return shape: [series][variate][timestep]
    data = _post(api_server, "/multivariate/forecast/mean", [MULTIVARIATE_SERIES])
    assert_forecast_close(data, [REFERENCE.multivariate[:, MEDIAN_QUANTILE_INDEX, :]])


def test_multivariate_quantiles(api_server):
    # return shape: [series][variate][quantile][timestep]
    data = _post(api_server, "/multivariate/forecast/quantiles", [MULTIVARIATE_SERIES])
    assert_forecast_close(data, [REFERENCE.multivariate])


# --- Multi-target -------------------------- #


def test_multitarget_mean(api_server):
    data = _post(api_server, "/multivariate/forecast/mean", [MULTITARGET_SERIES])
    assert_forecast_close(data, [REFERENCE.multitarget[:, MEDIAN_QUANTILE_INDEX, :]])


def test_multitarget_quantiles(api_server):
    data = _post(api_server, "/multivariate/forecast/quantiles", [MULTITARGET_SERIES])
    assert_forecast_close(data, [REFERENCE.multitarget])


# --- Batch --- #


def test_univariate_batch(api_server):
    data = _post(api_server, "/univariate/forecast/mean", [TARGET, TARGET])
    assert_forecast_close(
        data, [REFERENCE.univariate[0, MEDIAN_QUANTILE_INDEX], REFERENCE.univariate[0, MEDIAN_QUANTILE_INDEX]]
    )


def test_multivariate_batch(api_server):
    context = [MULTIVARIATE_SERIES, MULTITARGET_SERIES]
    data = _post(api_server, "/multivariate/forecast/mean", context)
    assert_forecast_close(data[0], REFERENCE.multivariate[:, MEDIAN_QUANTILE_INDEX, :])
    assert_forecast_close(data[1], REFERENCE.multitarget[:, MEDIAN_QUANTILE_INDEX, :])
