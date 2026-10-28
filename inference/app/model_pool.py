# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

import asyncio
import multiprocessing as mp
import os
import queue
import threading
import time
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

        self.requests: queue.Queue[Request] = queue.Queue()
        for _ in range(num_workers):
            threading.Thread(target=self._batch_loop, daemon=True).start()

    async def predict(self, contexts: list[TimeseriesType], prediction_length: int) -> list[np.ndarray]:
        if prediction_length < 1:
            raise ValueError(f"prediction_length must be >= 1, got {prediction_length}.")
        if self._oldest_wait() > self.max_queue_wait:
            raise QueueFullError("The server is overloaded, try again later.")
        request = Request([_to_series(c) for c in contexts], prediction_length)
        self.requests.put(request)
        return await asyncio.wrap_future(request.result)

    def _oldest_wait(self) -> float:
        try:
            return time.monotonic() - self.requests.queue[0].queued_at
        except IndexError:
            return 0.0

    def _batch_loop(self):
        while True:
            requests = []
            for request in self._take_batch():
                if not request.result.set_running_or_notify_cancel():
                    continue
                if time.monotonic() - request.queued_at > self.max_queue_wait:
                    request.result.set_exception(QueueFullError("The server is overloaded, try again later."))
                else:
                    requests.append(request)
            if requests:
                self._forecast(requests)

    def _take_batch(self) -> list[Request]:
        batch = [self.requests.get()]
        while True:
            try:
                batch.append(self.requests.get_nowait())
            except queue.Empty:
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


def _init_worker(settings: Settings, worker_indices, all_ready):
    global _model
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
        contexts, prediction_length=prediction_length, output_type="numpy", batch_size=min(len(contexts), 512)
    )


def _noop():
    pass
