# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

# Needs the server's dependencies, which are not installed when testing against the docker image.
model_pool = pytest.importorskip("app.model_pool")
torch = pytest.importorskip("torch")

from app.config import Settings  # noqa: E402

from tirex2 import TimeseriesType  # noqa: E402

NUM_QUANTILES = 9


class FakeModel:
    """Forecasts the last target value plus the step index; fails for series that start with NaN."""

    def __init__(self):
        self.calls: list[tuple[list, int]] = []
        self.running = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def __call__(self, series, prediction_length):
        self.calls.append((series, prediction_length))
        self.running.set()
        self.release.wait()
        if any(np.isnan(target[0, 0]) for target, _, _ in series):
            raise ValueError("invalid series")
        steps = np.arange(prediction_length, dtype=np.float32)
        return [
            np.broadcast_to(target[:, -1, None, None] + steps, (target.shape[0], NUM_QUANTILES, prediction_length))
            for target, _, _ in series
        ]


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(model_pool, "_forecast_in_worker", model)
    return model


def make_pool(**settings):
    return model_pool.ModelPool(Settings(**settings), executor=ThreadPoolExecutor(settings.get("model_workers", 1)))


def series(last_value: float, length: int = 8, future_covariates=None):
    target = torch.arange(length, dtype=torch.float32).reshape(1, -1) + last_value - (length - 1)
    return TimeseriesType(target=target, past_covariates=None, future_covariates=future_covariates)


def expected(last_value: float, prediction_length: int):
    return np.broadcast_to(last_value + np.arange(prediction_length), (1, NUM_QUANTILES, prediction_length))


async def wait_until_running(model: FakeModel):
    model.running.clear()
    await asyncio.to_thread(model.running.wait, 5)


async def test_queued_requests_are_batched(fake_model):
    pool = make_pool()
    fake_model.release.clear()
    first = asyncio.create_task(pool.predict([series(0)], 4))
    await wait_until_running(fake_model)

    queued = [asyncio.create_task(pool.predict([series(i), series(i + 100)], 4)) for i in range(1, 6)]
    await asyncio.sleep(0.1)
    fake_model.release.set()
    results = await asyncio.gather(first, *queued)

    assert [len(s) for s, _ in fake_model.calls] == [1, 10]
    np.testing.assert_array_equal(results[0][0], expected(0, 4))
    for i, result in enumerate(results[1:], start=1):
        np.testing.assert_array_equal(result[0], expected(i, 4))
        np.testing.assert_array_equal(result[1], expected(i + 100, 4))


async def test_batches_are_limited_to_max_worker_batch_size(fake_model):
    pool = make_pool(max_worker_batch_size=4)
    fake_model.release.clear()
    first = asyncio.create_task(pool.predict([series(0)], 4))
    await wait_until_running(fake_model)

    sizes = [3, 3, 1, 6, 1]
    queued = [asyncio.create_task(pool.predict([series(i)] * n, 4)) for i, n in enumerate(sizes)]
    await asyncio.sleep(0.1)
    fake_model.release.set()
    results = await asyncio.gather(first, *queued)

    # A request larger than the limit runs alone; others are never combined beyond the limit.
    assert [len(s) for s, _ in fake_model.calls] == [1, 3, 4, 6, 1]
    assert [len(r) for r in results[1:]] == sizes


async def test_mixed_prediction_lengths_run_in_one_batch(fake_model):
    pool = make_pool()
    fake_model.release.clear()
    first = asyncio.create_task(pool.predict([series(0)], 4))
    await wait_until_running(fake_model)

    covariates = torch.arange(8 + 10, dtype=torch.float32).reshape(1, -1)
    short = asyncio.create_task(pool.predict([series(1, future_covariates=covariates)], 2))
    long = asyncio.create_task(pool.predict([series(2)], 6))
    await asyncio.sleep(0.1)
    fake_model.release.set()
    _, short_result, long_result = await asyncio.gather(first, short, long)

    batch, prediction_length = fake_model.calls[1]
    assert prediction_length == 6
    np.testing.assert_array_equal(short_result[0], expected(1, 2))
    np.testing.assert_array_equal(long_result[0], expected(2, 6))
    # Future covariates are cut to the request's own horizon and NaN-padded to the batch horizon.
    padded = batch[0][2][0]
    assert padded.shape == (8 + 6,)
    np.testing.assert_array_equal(padded[: 8 + 2], np.arange(8 + 2))
    assert np.isnan(padded[8 + 2 :]).all()


async def test_invalid_request_does_not_fail_others(fake_model):
    pool = make_pool()
    fake_model.release.clear()
    first = asyncio.create_task(pool.predict([series(0)], 4))
    await wait_until_running(fake_model)

    invalid = TimeseriesType(target=torch.full((1, 8), torch.nan), past_covariates=None, future_covariates=None)
    valid = [asyncio.create_task(pool.predict([series(i)], 4)) for i in range(1, 4)]
    failing = asyncio.create_task(pool.predict([invalid], 4))
    await asyncio.sleep(0.1)
    fake_model.release.set()
    await first

    for i, task in enumerate(valid, start=1):
        np.testing.assert_array_equal((await task)[0], expected(i, 4))
    with pytest.raises(ValueError, match="invalid series"):
        await failing


async def test_rejects_requests_when_queue_is_full(fake_model):
    pool = make_pool(max_queue_wait_s=0.2)
    fake_model.release.clear()
    first = asyncio.create_task(pool.predict([series(0)], 4))
    await wait_until_running(fake_model)

    waiting = asyncio.create_task(pool.predict([series(1)], 4))
    await asyncio.sleep(0.3)
    with pytest.raises(model_pool.QueueFullError):
        await pool.predict([series(2)], 4)

    fake_model.release.set()
    await first
    # The request that waited longer than the limit is rejected instead of forecast.
    with pytest.raises(model_pool.QueueFullError):
        await waiting
    assert len(fake_model.calls) == 1

    np.testing.assert_array_equal((await pool.predict([series(3)], 4))[0], expected(3, 4))


async def test_rejects_invalid_requests(fake_model):
    pool = make_pool(max_request_batch_size=3)
    with pytest.raises(ValueError, match="at most 3 series"):
        await pool.predict([series(0)] * 4, 4)
    with pytest.raises(ValueError, match="prediction_length must be >= 1"):
        await pool.predict([series(0)], 0)
    assert fake_model.calls == []


async def test_multiple_workers_run_in_parallel(fake_model):
    pool = make_pool(model_workers=2)
    fake_model.release.clear()
    first = asyncio.create_task(pool.predict([series(0)], 4))
    await wait_until_running(fake_model)
    second = asyncio.create_task(pool.predict([series(1)], 4))
    await wait_until_running(fake_model)

    assert len(fake_model.calls) == 2
    fake_model.release.set()
    start = time.monotonic()
    await asyncio.gather(first, second)
    assert time.monotonic() - start < 1
