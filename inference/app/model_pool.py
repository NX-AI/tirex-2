# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

import asyncio
import multiprocessing as mp
import os
import threading
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field

import numpy as np
import torch

from app.config import Settings
from tirex2 import ForecastModel, TimeseriesType, load_model

Series = tuple[np.ndarray, np.ndarray | None, np.ndarray | None]


class QueueFullError(Exception):
    pass


@dataclass
class Request:
    series: list[Series]
    prediction_length: int
    result: Future = field(default_factory=Future)
    queued_at: float = field(default_factory=time.monotonic)


class ModelPool:
    def __init__(self, settings: Settings):
        self.max_queue_wait = settings.max_queue_wait_s
        self.max_worker_batch_size = settings.max_worker_batch_size
        self.max_request_batch_size = settings.max_request_batch_size
        num_workers = settings.model_workers
        ctx = mp.get_context("spawn")
        worker_indices = ctx.Queue()
        for index in range(num_workers):
            worker_indices.put(index)
        self.executor = ProcessPoolExecutor(
            num_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(settings, worker_indices, ctx.Barrier(num_workers)),
        )
        for started in [self.executor.submit(_noop) for _ in range(num_workers)]:
            started.result()

        self.requests: deque[Request] = deque()
        self.requests_changed = threading.Condition()
        for _ in range(num_workers):
            threading.Thread(target=self._batch_loop, daemon=True).start()

    async def predict(self, contexts: list[TimeseriesType], prediction_length: int) -> list[np.ndarray]:
        if prediction_length < 1:
            raise ValueError(f"prediction_length must be >= 1, got {prediction_length}.")
        if len(contexts) > self.max_request_batch_size:
            raise ValueError(
                f"A request can contain at most {self.max_request_batch_size} series, got {len(contexts)}."
            )
        request = Request([_to_series(c) for c in contexts], prediction_length)
        with self.requests_changed:
            if self.requests and time.monotonic() - self.requests[0].queued_at > self.max_queue_wait:
                raise QueueFullError("The server is overloaded, try again later.")
            self.requests.append(request)
            self.requests_changed.notify()
        return await asyncio.wrap_future(request.result)

    def _batch_loop(self):
        while True:
            self._forecast(self._take_batch())

    def _take_batch(self) -> list[Request]:
        batch, size = [], 0
        with self.requests_changed:
            while not batch:
                while not self.requests:
                    self.requests_changed.wait()
                while self.requests and (
                    not batch or size + len(self.requests[0].series) <= self.max_worker_batch_size
                ):
                    request = self.requests.popleft()
                    if not request.result.set_running_or_notify_cancel():
                        continue
                    if time.monotonic() - request.queued_at > self.max_queue_wait:
                        request.result.set_exception(QueueFullError("The server is overloaded, try again later."))
                        continue
                    batch.append(request)
                    size += len(request.series)
        return batch

    def _forecast(self, requests: list[Request]):
        prediction_length = max(request.prediction_length for request in requests)
        series = [
            _pad_future_covariates(s, request.prediction_length, prediction_length)
            for request in requests
            for s in request.series
        ]
        try:
            forecasts = self.executor.submit(_forecast_in_worker, series, prediction_length).result()
        except BrokenProcessPool:
            print("A model worker died, shutting down the server.", flush=True)
            os._exit(1)
        except Exception as e:
            if len(requests) == 1:
                requests[0].result.set_exception(e)
            else:
                for request in requests:
                    self._forecast([request])
            return

        start = 0
        for request in requests:
            end = start + len(request.series)
            request.result.set_result([f[..., : request.prediction_length] for f in forecasts[start:end]])
            start = end


def _pad_future_covariates(series: Series, prediction_length: int, padded_length: int) -> Series:
    target, past_covariates, future_covariates = series
    length = target.shape[-1] + prediction_length
    if future_covariates is None or future_covariates.shape[-1] < length:
        return series
    padding = np.full((future_covariates.shape[0], padded_length - prediction_length), np.nan, np.float32)
    return target, past_covariates, np.concatenate([future_covariates[..., :length], padding], axis=-1)


def _to_series(ts: TimeseriesType) -> Series:
    def to_numpy(x):
        return x.numpy() if x is not None else None

    return to_numpy(ts.target), to_numpy(ts.past_covariates), to_numpy(ts.future_covariates)


def _to_timeseries(series: Series) -> TimeseriesType:
    def to_tensor(x):
        return torch.from_numpy(x) if x is not None else None

    return TimeseriesType(*(to_tensor(x) for x in series))


_model: ForecastModel | None = None
_max_worker_batch_size: int | None = None


def _init_worker(settings: Settings, worker_indices, all_ready):
    global _model, _max_worker_batch_size
    _max_worker_batch_size = settings.max_worker_batch_size
    index = worker_indices.get()
    if settings.model_device == "cpu":
        torch.set_num_threads(max(1, torch.get_num_threads() // settings.model_workers))
    elif settings.model_device == "cuda":
        # Not "cuda:<index>": tirex2 only uses its Triton kernels for the device "cuda".
        torch.cuda.set_device(index % torch.cuda.device_count())

    _model = load_model(settings.model_path, device=settings.model_device)
    _forecast_in_worker([(np.arange(2048, dtype=np.float32).reshape(1, -1), None, None)], prediction_length=32)
    print(f"Worker {index} ready on {next(_model.parameters()).device}.", flush=True)
    all_ready.wait()


def _forecast_in_worker(series: list[Series], prediction_length: int) -> list[np.ndarray]:
    contexts = [_to_timeseries(s) for s in series]
    return _model.forecast(
        contexts,
        prediction_length=prediction_length,
        output_type="numpy",
        batch_size=min(len(contexts), _max_worker_batch_size),
    )


def _noop():
    pass
