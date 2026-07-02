# to prevent warnings that 'Demo' is not found in class-function annotations
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import chain

import numpy as np
import torch

from .model.types import TimeseriesType
from .plotting import (
    COVARIATE_COLORS,
    _raise_matplotlib_installation_error,
    _raise_plotly_installation_error,
    plot_covariate,
    plot_forecast,
)


@dataclass
class Covariate:
    label: str
    context: np.ndarray  # (context_length,)
    future: np.ndarray  # (horizon,)
    kind: str = "flag"  # 'flag' (binary -> event bands) | 'cont' (continuous -> line)


@dataclass
class Demo:
    title: str
    description: str
    target_context: np.ndarray  # (context_length,)
    target_future: np.ndarray  # (horizon,)
    covariates: list[Covariate]

    @property
    def horizon(self) -> int:
        return len(self.target_future)

    def to_timeseries_type(self, include_covariates=True) -> TimeseriesType:
        # build target data
        target = torch.from_numpy(self.target_context).unsqueeze(0)

        past_covariates = None
        future_covariates = None
        if include_covariates:
            # build past covariates
            past_covariates_list = [c.context for c in self.covariates if c.future is None]
            if len(past_covariates_list) > 0:
                past_covariates = torch.from_numpy(np.stack(past_covariates_list).astype(np.float32))

            # build future covariates
            future_covariates_list = [
                np.concatenate([c.context, c.future]) for c in self.covariates if c.future is not None
            ]
            if len(future_covariates_list) > 0:
                future_covariates = torch.from_numpy(np.stack(future_covariates_list).astype(np.float32))

        return TimeseriesType(
            target=target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

    def describe(self) -> str:
        print(f"{'-' * 70}")
        print("Demo Dataset")
        print(f"{'-' * 70}")
        print(f"Title:       {self.title}")
        print(f"Description: {self.description}")

    @classmethod
    def create_nonstationary_demo(cls, context_length: int = 540, horizon: int = 42, seed: int = 7) -> Demo:
        """Non-stationary demand: a CONTINUOUS future-known driver sets the wandering
        baseline level, plus a BINARY promotion flag that adds spikes.

        The baseline has no fixed mean (it follows a smoothed random walk + a slow
        swing that turns over inside the horizon), so the target's own history cannot
        say where the level is heading - only the continuous covariate can. The
        binary promotions add sharp spikes at irregular times. The model must use
        BOTH covariates: the continuous one to track the level, the flag for spikes.
        """
        rng = np.random.default_rng(seed)
        n_steps = context_length + horizon
        time_steps = np.arange(n_steps)

        # Continuous non-stationary driver (future-known): smoothed random walk plus a
        # slow swing phased to crest near t=0 and decline through the horizon.
        rw = _smooth(np.cumsum(rng.normal(0, 1, n_steps)), 31)
        rw = (rw - rw.mean()) / (rw.std() + 1e-8)
        swing = np.sin(2 * np.pi * (time_steps - (context_length - 70)) / 250.0)
        level = 120.0 + 40.0 * rw + 22.0 * swing  # wanders, no fixed mean

        weekly_profile = np.array([0.95, 0.92, 0.94, 0.98, 1.08, 1.32, 1.18])
        weekly = 34.0 * (weekly_profile[time_steps % 7] - 1.0)  # within-week deviation

        promo = _events(rng, n_steps, gap_lo=30, gap_hi=52, start=20, width_hi=3)
        promo = _force_events(promo, context_length, horizon, n_min=2, rng=rng)
        spike_abs = 55.0

        series = level + weekly + spike_abs * promo + rng.normal(0, 2.5, n_steps)

        return cls(
            title="TiRex-2 on a non-stationary series (continuous driver + promotion flag)",
            description="A continuous covariate sets the wandering level, while a binary flag adds spikes. The model model needs both covariates for solid forecasts.",
            target_context=series[:context_length].astype(np.float32),
            target_future=series[context_length:].astype(np.float32),
            covariates=[
                Covariate(
                    "demand driver (continuous, known ahead)",
                    level[:context_length],
                    level[context_length:],
                    kind="cont",
                ),
                Covariate(
                    "promotion (0/1, known ahead)",
                    promo[:context_length],
                    promo[context_length:],
                    kind="flag",
                ),
            ],
        )

    @classmethod
    def create_holidays_demo(cls, context_length: int = 540, horizon: int = 42, seed: int = 20) -> Demo:
        """ONE future-known holiday flag -> consistent multiplicative spike."""
        rng = np.random.default_rng(seed)
        base = _weekly_base(context_length, horizon, seed=seed)
        flag = _events(rng, context_length + horizon, gap_lo=26, gap_hi=46, start=8, width_hi=3)
        flag = _force_events(flag, context_length, horizon, n_min=3, rng=rng)
        spike = 0.80  # +80% demand on a holiday
        series = base * (1.0 + spike * flag)
        series = series * (1.0 + 0.012 * rng.normal(size=context_length + horizon))  # low noise

        return cls(
            title="TiRex-2 with a future-known covariate (holiday calendar)",
            description="Daily demand with known holidays. As they are irregular, they are unpredictable from history alone.",
            target_context=series[:context_length].astype(np.float32),
            target_future=series[context_length:].astype(np.float32),
            covariates=[
                Covariate(
                    "holiday flag (0/1, known ahead)",
                    flag[:context_length],
                    flag[context_length:],
                )
            ],
        )


def _events(rng, n, gap_lo, gap_hi, start, width_hi=3):
    """Irregular binary flag of length n with 1..width_hi-wide events."""
    flag = np.zeros(n)
    cur = start + int(rng.integers(0, gap_hi))
    while cur < n:
        w = int(rng.integers(1, width_hi))
        flag[cur : cur + w] = 1.0
        cur += int(rng.integers(gap_lo, gap_hi))
    return flag


def _force_events(flag, T, H, n_min, rng, width_hi=3, margin=3):
    """Guarantee at least ``n_min`` events land inside the horizon [T, T+H)."""
    have = int((np.diff((flag[T:] > 0).astype(int)) == 1).sum() + (flag[T] > 0))
    tries = 0
    while have < n_min and tries < 200:
        c = T + int(rng.integers(margin, H - margin))
        w = int(rng.integers(1, width_hi))
        if flag[max(0, c - 4) : c + w + 4].sum() == 0:  # keep events separated
            flag[c : c + w] = 1.0
            have += 1
        tries += 1
    return flag


def _weekly_base(T, H, trend_per_step=0.0004, seed=0):
    """Daily retail-style base: weekly profile x mild growth trend."""
    n = T + H
    t = np.arange(n)
    weekly_profile = np.array([0.95, 0.92, 0.94, 0.98, 1.08, 1.32, 1.18])  # Mon..Sun
    weekly = weekly_profile[t % 7]
    trend = 1.0 + trend_per_step * t
    return 100.0 * weekly * trend


def _smooth(x, k):
    return np.convolve(x, np.ones(k) / k, mode="same")


def _plot_demo_forecast_matplotlib(
    demo: Demo,
    univariate_forecast: torch.Tensor,
    multivariate_forecast: torch.Tensor,
    max_context_to_show: int = 64,
):

    n_covariates = len(demo.covariates)
    covariate_labels = [f"Cov. {i + 1}" for i in range(n_covariates)]

    x = np.arange(len(demo.target_context) + demo.horizon)

    try:
        from matplotlib import pyplot as plt
    except ImportError as e:
        _raise_matplotlib_installation_error(e)

    if n_covariates > 0:
        height_ratios = [n_covariates] * 2 + [1] * n_covariates
    else:
        height_ratios = [1, 1]

    # create figure holding the target forecast and covariates
    fig_width = 10
    fig_height = 4 + 2 * n_covariates
    fig, axes = plt.subplots(
        nrows=2 + n_covariates,
        ncols=1,
        sharex=True,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": height_ratios},
        layout="constrained",
        squeeze=False,
    )
    fig.suptitle("\n".join([demo.title, demo.description]))

    axes = axes.flatten()
    ax_uv, ax_mv, *ax_covs = axes
    ax_uv.set_ylabel("TiRex-2 Univariate")
    ax_mv.set_ylabel("TiRex-2 Multivariate")

    # univariate forecast
    plot_forecast(
        context=demo.target_context,
        forecasts=univariate_forecast[0],
        ground_truth=demo.target_future,
        x=x,
        max_context_to_show=max_context_to_show,
        engine="matplotlib",
        ax=ax_uv,
    )

    # multivariate forecast
    plot_forecast(
        context=demo.target_context,
        forecasts=multivariate_forecast[0],
        ground_truth=demo.target_future,
        x=x,
        max_context_to_show=max_context_to_show,
        engine="matplotlib",
        ax=ax_mv,
    )

    # plot covariates
    for i, (ax, ax_lbl, cov) in enumerate(zip(ax_covs, covariate_labels, demo.covariates)):
        ax.set_ylabel(ax_lbl)
        plot_covariate(
            np.concat([cov.context[-max_context_to_show:], cov.future]),
            x=x[-(max_context_to_show + demo.horizon) :],
            label=cov.label,
            engine="matplotlib",
            ax=ax,
            color=COVARIATE_COLORS[i],
        )

    return fig


def _plot_demo_forecast_plotly(
    demo: Demo,
    univariate_forecast: torch.Tensor,
    multivariate_forecast: torch.Tensor,
    max_context_to_show: int = 64,
):
    n_covariates = len(demo.covariates)
    covariate_heights = [0.3 / n_covariates] * n_covariates
    covariate_labels = [f"Cov. {i + 1}" for i in range(n_covariates)]

    x = np.arange(len(demo.target_context) + demo.horizon)

    try:
        from plotly.subplots import make_subplots
    except Exception as e:
        _raise_plotly_installation_error(e)

    fig = make_subplots(
        rows=2 + n_covariates,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.35, 0.35] + covariate_heights,
        row_titles=["TiRex-2 Univariate", "TiRex-2 Multivariate"] + covariate_labels,
    )

    plot_forecast(
        context=demo.target_context,
        forecasts=univariate_forecast[0],
        ground_truth=demo.target_future,
        x=x,
        max_context_to_show=max_context_to_show,
        engine="plotly",
        fig=fig,
        col=1,
        row=1,
    )

    plot_forecast(
        context=demo.target_context,
        forecasts=multivariate_forecast[0],
        ground_truth=demo.target_future,
        x=x,
        max_context_to_show=max_context_to_show,
        engine="plotly",
        fig=fig,
        col=1,
        row=2,
    )

    for i, cov in enumerate(demo.covariates):
        plot_covariate(
            np.concat([cov.context[-max_context_to_show:], cov.future]),
            x=x[-(max_context_to_show + demo.horizon) :],
            label=cov.label,
            engine="plotly",
            fig=fig,
            col=1,
            row=i + 3,
            color=COVARIATE_COLORS[i],
        )

    fig.update_layout(
        height=600 + 200 * n_covariates,
        title=dict(text=demo.title, subtitle_text=demo.description),
    )
    return fig


def plot_demo_forecast(
    demo: Demo,
    univariate_forecast: torch.Tensor,
    multivariate_forecast: torch.Tensor,
    max_context_to_show: int = 64,
    engine: str = "plotly",
):
    plot_params = dict(
        demo=demo,
        univariate_forecast=univariate_forecast,
        multivariate_forecast=multivariate_forecast,
        max_context_to_show=max_context_to_show,
    )

    # call appropriate plotting function based on the specified engine
    match engine:
        case "matplotlib":
            return _plot_demo_forecast_matplotlib(**plot_params)
        case "plotly":
            return _plot_demo_forecast_plotly(**plot_params)
        case _:
            raise ValueError(f"Drawing {engine=} not supported.")
