# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

import multiprocessing as mp
import os
import queue
import threading
from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field

import numpy as np
import torch

from app.config import Settings
from tirex2 import ForecastModel, TimeseriesType, load_model

Series = tuple[np.ndarray, np.ndarray | None, np.ndarray | None]


@dataclass
class Request:
    series: list[Series]
    prediction_length: int
    result: Future = field(default_factory=Future)


class ModelPool:
    def __init__(self, settings: Settings):
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

    def predict(self, contexts: list[TimeseriesType], prediction_length: int) -> list[np.ndarray]:
        request = Request([_to_series(c) for c in contexts], prediction_length)
        self.requests.put(request)
        return request.result.result()

    def _batch_loop(self):
        while True:
            by_length = defaultdict(list)
            for request in self._take_batch():
                by_length[request.prediction_length].append(request)
            for prediction_length, requests in by_length.items():
                self._forecast(requests, prediction_length)

    def _take_batch(self) -> list[Request]:
        batch = [self.requests.get()]
        while True:
            try:
                batch.append(self.requests.get_nowait())
            except queue.Empty:
                return batch

    def _forecast(self, requests: list[Request], prediction_length: int):
        series = [s for request in requests for s in request.series]
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
                    self._forecast([request], prediction_length)
            return

        start = 0
        for request in requests:
            end = start + len(request.series)
            request.result.set_result(forecasts[start:end])
            start = end


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
