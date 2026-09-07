import datetime

import datasets
import fev
import pytest
import torch

from tirex2.api_adapter.forecast import build_fev_timeseries

CONTEXT, HORIZON, NUM_SERIES = 12, 4, 3
LENGTH = CONTEXT + HORIZON
TARGETS = ["load", "temp"]
BASES = {"load": 0, "temp": 1_000, "fut": 10_000, "pst": 20_000}


def _column(name: str, series: int) -> list[float]:
    return [float(BASES[name] + 100 * series + step) for step in range(LENGTH)]


def _window(**overrides) -> "fev.EvaluationWindow":
    start = datetime.datetime(2020, 1, 1)
    full_dataset = datasets.Dataset.from_list(
        [
            {
                "id": f"s{series}",
                "timestamp": [start + datetime.timedelta(hours=step) for step in range(LENGTH)],
                **{name: _column(name, series) for name in BASES},
            }
            for series in range(NUM_SERIES)
        ]
    )
    return fev.EvaluationWindow(
        **{
            "full_dataset": full_dataset,
            "cutoff": -HORIZON,
            "horizon": HORIZON,
            "min_context_length": 1,
            "max_context_length": None,
            "id_column": "id",
            "timestamp_column": "timestamp",
            "target_columns": TARGETS,
            "known_dynamic_columns": ["fut"],
            "past_dynamic_columns": ["pst"],
            "static_columns": [],
            **overrides,
        }
    )


def _assert_row(row: torch.Tensor, name: str, series: int) -> None:
    torch.testing.assert_close(row, torch.tensor(_column(name, series)[: len(row)]))


def test_build_fev_timeseries_extracts_targets_and_covariates():
    series, meta = build_fev_timeseries(_window())

    assert len(series) == NUM_SERIES
    for index, ts in enumerate(series):
        assert ts.target.shape == (len(TARGETS), CONTEXT)
        for row, name in zip(ts.target, TARGETS):
            _assert_row(row, name, index)
        assert ts.past_covariates.shape == (1, CONTEXT)
        _assert_row(ts.past_covariates[0], "pst", index)
        assert ts.future_covariates.shape == (1, LENGTH)
        _assert_row(ts.future_covariates[0], "fut", index)

    assert meta == NUM_SERIES * [{"target_columns": TARGETS, "window_target_columns": TARGETS, "as_univariate": False}]


def test_build_fev_timeseries_as_univariate_splits_targets_and_drops_covariates():
    series, meta = build_fev_timeseries(_window(), as_univariate=True)

    assert len(series) == NUM_SERIES * len(TARGETS)
    for index, ts in enumerate(series):
        assert ts.target.shape == (1, CONTEXT)
        _assert_row(ts.target[0], TARGETS[index % len(TARGETS)], index // len(TARGETS))
        assert ts.past_covariates is None and ts.future_covariates is None

    assert meta == NUM_SERIES * len(TARGETS) * [
        {"target_columns": ["target"], "window_target_columns": TARGETS, "as_univariate": True}
    ]
