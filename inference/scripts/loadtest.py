# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

"""Load test for the TiRex inference container's HTTP API.

Starts the docker container (or targets an already running server), then sweeps
over concurrency levels and reports throughput, latency percentiles and errors.

Examples:
    # Start the CPU container and sweep concurrency 1, 4, 16
    python scripts/loadtest.py --image ghcr.io/nx-ai/tirex2-cpu --concurrency 1 4 16

    # Against an already running server
    python scripts/loadtest.py --url http://localhost:8000 --duration 60 --batch-size 32
"""

import argparse
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

ENDPOINTS = [
    "/univariate/forecast/mean",
    "/univariate/forecast/quantiles",
    "/multivariate/forecast/mean",
    "/multivariate/forecast/quantiles",
]


def wait_for_health(url: str, timeout: float, container: str | None = None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if container is not None:
            state = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container], capture_output=True, text=True
            )
            if state.stdout.strip() != "true":
                raise RuntimeError(f"Container {container} exited during startup, see `docker logs {container}`.")
        try:
            if requests.get(f"{url}/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"{url}/health not reachable within {timeout}s")


def start_container(image: str, port: int, cpus: str | None, gpus: str | None) -> str:
    cmd = ["docker", "run", "-d", "--rm", "-p", f"{port}:8000"]
    if cpus:
        cmd += ["--cpus", cpus]
    if gpus:
        cmd += ["--gpus", gpus]
    cmd.append(image)
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def make_payload(endpoint: str, batch_size: int, context_length: int, prediction_length: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(context_length)
    series = [
        (np.sin(2 * np.pi * t / rng.uniform(8, 64)) * rng.uniform(1, 10) + rng.normal(0, 0.3, context_length)).tolist()
        for _ in range(batch_size)
    ]
    if endpoint.startswith("/univariate"):
        context = series
    else:
        context = [{"target": [s]} for s in series]
    return {"context": context, "prediction_length": prediction_length}


def expected_shape(endpoint: str, batch_size: int, prediction_length: int) -> tuple[int, ...]:
    shape = (batch_size,)
    if endpoint.startswith("/multivariate"):
        shape += (1,)
    if endpoint.endswith("quantiles"):
        shape += (9,)
    return shape + (prediction_length,)


def run_level(url, endpoint, payload, shape, concurrency, duration, timeout):
    """Run `concurrency` closed-loop workers for `duration` seconds."""
    latencies: list[float] = []
    errors: Counter = Counter()
    lock = threading.Lock()
    stop_at = time.monotonic() + duration

    def worker():
        session = requests.Session()
        while time.monotonic() < stop_at:
            start = time.perf_counter()
            try:
                resp = session.post(f"{url}{endpoint}", json=payload, timeout=timeout)
                elapsed = time.perf_counter() - start
                if resp.status_code != 200:
                    error = f"HTTP {resp.status_code}"
                elif np.asarray(resp.json()).shape != shape:
                    error = "bad shape"
                else:
                    error = None
            except requests.RequestException as e:
                elapsed, error = time.perf_counter() - start, type(e).__name__
            with lock:
                if error is None:
                    latencies.append(elapsed)
                else:
                    errors[error] += 1

    wall_start = time.monotonic()
    with ThreadPoolExecutor(concurrency) as pool:
        for f in [pool.submit(worker) for _ in range(concurrency)]:
            f.result()
    wall = time.monotonic() - wall_start

    lat = np.asarray(latencies) * 1000
    pct = np.percentile(lat, [50, 90, 99]) if len(lat) else [float("nan")] * 3
    return {
        "concurrency": concurrency,
        "ok": len(lat),
        "errors": sum(errors.values()),
        "error_kinds": dict(errors),
        "req_s": len(lat) / wall,
        "series_s": len(lat) * shape[0] / wall,
        "p50": pct[0],
        "p90": pct[1],
        "p99": pct[2],
        "max": lat.max() if len(lat) else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--url", help="Base URL of an already running server")
    target.add_argument("--image", help="Docker image to start, e.g. ghcr.io/nx-ai/tirex2-cpu")
    parser.add_argument("--port", type=int, default=8010, help="Host port when starting a container")
    parser.add_argument("--cpus", help="docker --cpus limit for the container")
    parser.add_argument("--gpus", help="docker --gpus value for the container (GPU image only)")
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--endpoint", choices=ENDPOINTS, default=ENDPOINTS[0])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--duration", type=float, default=30, help="Seconds per concurrency level")
    parser.add_argument("--warmup", type=float, default=5, help="Warmup seconds before the sweep")
    parser.add_argument("--batch-size", type=int, default=1, help="Series per request")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120, help="Per-request timeout in seconds")
    parser.add_argument("--max-error-rate", type=float, default=0.0, help="Exit non-zero above this error rate")
    args = parser.parse_args()

    container = None
    url = args.url
    if args.image:
        container = start_container(args.image, args.port, args.cpus, args.gpus)
        url = f"http://127.0.0.1:{args.port}"
        print(f"Started container {container[:12]} from {args.image}, waiting for {url}/health ...")

    try:
        wait_for_health(url, args.startup_timeout, container)
        payload = make_payload(args.endpoint, args.batch_size, args.context_length, args.prediction_length)
        shape = expected_shape(args.endpoint, args.batch_size, args.prediction_length)
        print(
            f"Endpoint {args.endpoint}: batch_size={args.batch_size} context_length={args.context_length} "
            f"prediction_length={args.prediction_length}, {args.duration:g}s per level"
        )

        if args.warmup > 0:
            run_level(url, args.endpoint, payload, shape, 1, args.warmup, args.timeout)

        header = f"{'conc':>5} {'ok':>7} {'err':>5} {'req/s':>9} {'series/s':>10} {'p50 ms':>9} {'p90 ms':>9} {'p99 ms':>9} {'max ms':>9}"
        print(header)
        print("-" * len(header))
        failed = False
        for concurrency in args.concurrency:
            r = run_level(url, args.endpoint, payload, shape, concurrency, args.duration, args.timeout)
            print(
                f"{r['concurrency']:>5} {r['ok']:>7} {r['errors']:>5} {r['req_s']:>9.2f} {r['series_s']:>10.1f} "
                f"{r['p50']:>9.1f} {r['p90']:>9.1f} {r['p99']:>9.1f} {r['max']:>9.1f}"
            )
            if r["error_kinds"]:
                print(f"      errors: {r['error_kinds']}")
            total = r["ok"] + r["errors"]
            if total == 0 or r["errors"] / total > args.max_error_rate:
                failed = True
        return 1 if failed else 0
    finally:
        if container is not None:
            subprocess.run(["docker", "stop", container], capture_output=True)


if __name__ == "__main__":
    sys.exit(main())
