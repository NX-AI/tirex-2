# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests
import torch

start_server = int(os.getenv("TEST_START_SERVER", "1")) == 1
base_host = os.getenv("TEST_HOST", "127.0.0.1")
base_port = int(os.getenv("TEST_PORT", "8002"))
mqtt_host = os.getenv("TEST_MQTT_BROKER_HOST", "broker.emqx.io")
mqtt_port = int(os.getenv("TEST_MQTT_BROKER_PORT", "1883"))


def wait_for_api(healthcheck_url, timeout=30, process=None):
    print("Start for model load")
    for i in range(timeout):
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Server exited with code {process.returncode}. See test-server.log for details.")
        try:
            response = requests.get(healthcheck_url)
            if response.status_code == 200:
                print(f"Connected in {i} seconds!")
                return
        except:
            pass
        time.sleep(1)

    raise TimeoutError(f"Can't connect to {healthcheck_url} in {timeout} seconds!")


@pytest.fixture(scope="session")
def api_server():
    base_url = f"http://{base_host}:{base_port}"
    process = None
    log_file = None

    try:
        if start_server:
            env = os.environ.copy()
            env.update(
                {
                    "HTTP_HOST": base_host,
                    "HTTP_PORT": str(base_port),
                    "MQTT_ENABLED": "1",
                    "MQTT_BROKER_HOST": mqtt_host,
                    "MQTT_BROKER_PORT": str(mqtt_port),
                }
            )
            log_file = open("test-server.log", "w", encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, "-m", "app.main"],
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        wait_for_api(f"{base_url}/health", timeout=30, process=process)

        yield base_url
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if log_file is not None:
            log_file.close()


def get_default_context():
    prediction_length = 2
    context = [[0.0, 1.0, 2.0, 3.0]]
    return context, prediction_length


def assert_default_prediction_correct(forecast: list[float]):
    data = torch.tensor(forecast, dtype=torch.float32)
    data_ref = torch.tensor([[3.751096248, 4.562105178]], dtype=torch.float32)
    # bfloat16 tolerances to allow for small differences between CPU and CUDA
    torch.testing.assert_close(data, data_ref, rtol=1.6e-2, atol=1e-5)


# --------------------------------------------------------------------------- #
# References
# --------------------------------------------------------------------------- #

TESTS_DIR = Path(__file__).parent
REFERENCES_DIR = TESTS_DIR / "references"
INPUT = json.loads((REFERENCES_DIR / "reference_input.json").read_text())
OUTPUT = json.loads((REFERENCES_DIR / "reference_output_cpu.json").read_text())

PREDICTION_LENGTH = INPUT["H"]
TARGET = INPUT["target"]
OTHER_TARGET = INPUT["other_target"]
FUTURE_COVARIATES = INPUT["future_covariates"]

UNIVARIATE_SERIES = {"target": [TARGET], "past_covariates": None, "future_covariates": None}
MULTIVARIATE_SERIES = {"target": [TARGET], "past_covariates": None, "future_covariates": FUTURE_COVARIATES}
MULTITARGET_SERIES = {"target": [TARGET, OTHER_TARGET], "past_covariates": None, "future_covariates": None}

# [num_target_variates, num_quantiles, prediction_length].
REFERENCE = SimpleNamespace(
    univariate=np.array(OUTPUT["univariate"], dtype=np.float32),
    multivariate=np.array(OUTPUT["multivariate"], dtype=np.float32),
    multitarget=np.array(OUTPUT["multitarget"], dtype=np.float32),
)
MEDIAN_QUANTILE_INDEX = 4


def assert_forecast_close(actual, expected):
    a = torch.as_tensor(np.asarray(actual, dtype=np.float32))
    e = torch.as_tensor(np.asarray(expected, dtype=np.float32))
    assert a.shape == e.shape, f"shape {tuple(a.shape)} != {tuple(e.shape)}"
    torch.testing.assert_close(a, e, rtol=1.6e-2, atol=1e-2)
