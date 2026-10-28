# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

from concurrent.futures import ThreadPoolExecutor

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


# --- Requests batched across clients --- #


def _post_raw(api_server, path, payload):
    return requests.post(f"{api_server}{path}", json=payload, timeout=120)


def _mixed_requests():
    covariates = [float(i % 3 == 0) for i in range(len(TARGET) + 64)]
    return (
        [("/univariate/forecast/quantiles", {"context": [TARGET], "prediction_length": h}) for h in (1, 7, 32)]
        + [
            (
                "/multivariate/forecast/quantiles",
                {
                    "context": [{"target": [TARGET], "future_covariates": [covariates[: len(TARGET) + h]]}],
                    "prediction_length": h,
                },
            )
            for h in (5, 64)
        ]
        + [("/multivariate/forecast/mean", {"context": [MULTITARGET_SERIES], "prediction_length": 16})]
    )


def test_concurrent_requests_match_single_requests(api_server):
    cases = _mixed_requests()
    single = [_post_raw(api_server, path, payload).json() for path, payload in cases]
    with ThreadPoolExecutor(len(cases) * 3) as pool:
        concurrent = list(pool.map(lambda case: _post_raw(api_server, *case), cases * 3))

    for i, response in enumerate(concurrent):
        assert response.status_code == 200, response.text
        assert_forecast_close(response.json(), single[i % len(cases)])


def test_invalid_request_does_not_fail_concurrent_requests(api_server):
    valid = ("/univariate/forecast/mean", {"context": [TARGET], "prediction_length": PREDICTION_LENGTH})
    invalid = (
        "/multivariate/forecast/mean",
        {"context": [{"target": [TARGET], "future_covariates": [[1.0, 2.0]]}], "prediction_length": PREDICTION_LENGTH},
    )
    with ThreadPoolExecutor(20) as pool:
        responses = list(pool.map(lambda case: _post_raw(api_server, *case), [valid, invalid] * 10))

    for response in responses[0::2]:
        assert response.status_code == 200, response.text
        assert_forecast_close(response.json(), [REFERENCE.univariate[0, MEDIAN_QUANTILE_INDEX]])
    for response in responses[1::2]:
        assert response.status_code == 422
        assert "Future known covariates" in response.json()["error_message"]


def test_too_many_series_are_rejected(api_server):
    response = _post_raw(
        api_server, "/univariate/forecast/mean", {"context": [[1.0, 2.0, 3.0, 4.0]] * 2049, "prediction_length": 4}
    )
    assert response.status_code == 422
    assert "at most 2048 series" in response.json()["error_message"]


def test_invalid_prediction_length_is_rejected(api_server):
    response = _post_raw(api_server, "/univariate/forecast/mean", {"context": [TARGET], "prediction_length": 0})
    assert response.status_code == 422
    assert "prediction_length must be >= 1" in response.json()["error_message"]
