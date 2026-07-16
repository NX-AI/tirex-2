from collections.abc import Sequence
from itertools import chain

import numpy as np
import torch

from tirex2.model.types import TimeseriesType

CONNECT_FORECAST_TO_CONTEXT = True

# color from Plotly G10 color map
COLOR_CONTEXT = "#3366CC"
COLOR_FORECAST = "#DC3912"
COLOR_GROUND_TRUTH = "#3366CC"
COLOR_CUTOFF_LINE = "#000000"
COLOR_QUANTILES = "#DC3912"
ALPHA_QUANTILES = 0.1

COVARIATE_COLORS = [
    "#FF9900",
    "#109618",
    "#990099",
    "#0099C6",
    "#DD4477",
    "#66AA00",
    "#B82E2E",
    "#316395",
]


def _raise_matplotlib_installation_error(e):
    raise ImportError(
        "'_plot_forecast_matplotlib' requires matplotlib to be installed. "
        "Please install TiRex package with plotting support via "
        "\"pip install 'tirex-2[plotting]'\"."
    ) from e


def _raise_plotly_installation_error(e):
    raise ImportError(
        "'_plot_forecast_plotly' requires plotly to be installed. "
        "Please install TiRex package with plotting support via "
        "\"pip install 'tirex-2[plotting]'\"."
    ) from e


def _check_trace_name_visible(fig, trace_name):
    return any(trace_name == trace.name for trace in fig.data if trace.showlegend)


def _plot_forecast_matplotlib(
    context,
    x_context,
    x_ground_truth=None,
    x_forecast=None,
    point_forecast=None,
    ground_truth=None,
    lower_quantile=None,
    upper_quantile=None,
    label_context=None,
    label_forecast=None,
    label_ground_truth=None,
    label_quantile=None,
    view_min_x=None,
    ax=None,
):
    try:
        from matplotlib import pyplot as plt
    except ImportError as e:
        _raise_matplotlib_installation_error(e)

    if ax is None:
        # default to current axis
        ax = plt.gca()

    # plot context
    if context is not None:
        ax.plot(x_context, context, label=label_context, color=COLOR_CONTEXT)

    # plot ground truth if supplied
    if ground_truth is not None:
        ax.plot(
            x_ground_truth,
            ground_truth,
            label=label_ground_truth,
            color=COLOR_GROUND_TRUTH,
            linestyle="--",
        )

    # plot forecasts if supplied
    # forecasts are a 2D array with quantiles as rows, and data for each timestep as columns
    if point_forecast is not None:
        ax.plot(x_forecast, point_forecast, label=label_forecast, color=COLOR_FORECAST)

        if lower_quantile is not None and upper_quantile is not None:
            ax.fill_between(
                x_forecast,
                lower_quantile,
                upper_quantile,
                color=COLOR_QUANTILES,
                alpha=ALPHA_QUANTILES,
                label=label_quantile,
            )

    if context is not None and (ground_truth is not None or point_forecast is not None):
        ax.axvline(x_context[-1], color=COLOR_CUTOFF_LINE, linestyle=":")

    # Only min/max over the endpoints that actually exist to avoid comparing
    # foreign-typed sentinels (e.g. pandas Timestamp vs. np.inf) which crashes.
    x_series = [xs for xs in (x_context, x_forecast, x_ground_truth) if xs is not None]
    if x_series:
        left = view_min_x if view_min_x is not None else min(xs[0] for xs in x_series)
        ax.set_xlim(left=left)
    ax.legend()
    ax.grid()

    return ax


def _plot_forecast_plotly(
    context=None,
    x_context=None,
    x_ground_truth=None,
    x_forecast=None,
    point_forecast=None,
    ground_truth=None,
    lower_quantile=None,
    upper_quantile=None,
    label_context=None,
    label_forecast=None,
    label_ground_truth=None,
    label_quantile=None,
    view_min_x=None,
    fig=None,
    row=None,
    col=None,
):

    def hex_to_rgba(hex_color, alpha):
        r, g, b = hex_to_rgb(hex_color)
        return f"rgba({r}, {g}, {b}, {alpha})"

    try:
        import plotly.graph_objects as go
        from plotly.colors import hex_to_rgb
    except ImportError as e:
        _raise_plotly_installation_error(e)

    if fig is None:
        fig = go.Figure()

    if context is not None:
        fig.add_trace(
            go.Scatter(
                x=x_context,
                y=context,
                name=label_context,
                line={
                    "color": COLOR_CONTEXT,
                    "width": 2,
                    "dash": "solid",
                },
                marker=None,
                legendgroup=label_context,
                showlegend=not _check_trace_name_visible(fig, label_context),
            ),
            row=row,
            col=col,
        )

    # plot ground truth if supplied
    if ground_truth is not None:
        fig.add_trace(
            go.Scatter(
                x=x_ground_truth,
                y=ground_truth,
                name=label_ground_truth,
                line={
                    "color": COLOR_GROUND_TRUTH,
                    "width": 2,
                    "dash": "dash",
                },
                marker=None,
                legendgroup=label_ground_truth,
                showlegend=not _check_trace_name_visible(fig, label_ground_truth),
            ),
            row=row,
            col=col,
        )

    # plot forecasts if supplied
    # forecasts are a 2D array with quantiles as rows, and data for each timestep as columns
    if point_forecast is not None:
        fig.add_trace(
            go.Scatter(
                x=x_forecast,
                y=point_forecast,
                name=label_forecast,
                line={
                    "color": COLOR_FORECAST,
                    "width": 2,
                    "dash": "solid",
                },
                marker=None,
                legendgroup=label_forecast,
                showlegend=not _check_trace_name_visible(fig, label_forecast),
            ),
            row=row,
            col=col,
        )

        if lower_quantile is not None and upper_quantile is not None:
            quantile_color = hex_to_rgba(COLOR_QUANTILES, ALPHA_QUANTILES)

            fig.add_traces(
                [
                    go.Scatter(
                        x=x_forecast,
                        y=lower_quantile,
                        name=label_quantile,
                        line={"width": 0},
                        mode="lines",
                        showlegend=False,
                    ),
                    go.Scatter(
                        x=x_forecast,
                        y=upper_quantile,
                        name=label_quantile,
                        line={"width": 0},
                        mode="lines",
                        fill="tonexty",
                        fillcolor=quantile_color,
                        legendgroup=label_quantile,
                        showlegend=not _check_trace_name_visible(fig, label_quantile),
                    ),
                ],
                rows=row,
                cols=col,
            )

    if context is not None and (ground_truth is not None or point_forecast is not None):
        # workaround for cutoff due to known bug in plotly: https://github.com/plotly/plotly.py/issues/3065
        fig.add_vrect(
            x0=x_context[-1],
            x1=x_context[-1],
            line_width=1.5,
            line_dash="0px,7px,4px,4px",
            line_color=COLOR_CUTOFF_LINE,
            annotation_text="forecast start",
            row=row,
            col=col,
        )

    # Only min/max over the endpoints that actually exist to avoid comparing
    # foreign-typed sentinels (e.g. pandas Timestamp vs. np.inf) which crashes.
    x_series = [xs for xs in (x_context, x_forecast, x_ground_truth) if xs is not None]
    if x_series:
        min_x = view_min_x if view_min_x is not None else min(xs[0] for xs in x_series)
        max_x = max(xs[-1] for xs in x_series)
        fig.update_xaxes(range=[min_x, max_x], autorange=False, row=row, col=col)
    return fig


def plot_forecast(
    context: torch.Tensor | np.ndarray | None = None,
    forecasts: torch.Tensor | np.ndarray | None = None,
    ground_truth: torch.Tensor | np.ndarray | None = None,
    x: Sequence | None = None,
    quantiles: tuple[float, float] = (0.1, 0.9),
    quantile_levels: tuple[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    engine="plotly",
    max_context_to_show: int | None = None,
    ax=None,
    fig=None,
    **kwargs,
):
    """
    Plots the historical context, optional ground-truth future, and forecast.

    Parameters
    ----------
    context : torch.Tensor or np.ndarray
        The historical time series data to be plotted.
    forecasts : torch.Tensor or np.ndarray, optional
        The forecasts data including quantiles, of shape [Q, N],
        where Q=9 quantiles are required, and N is the number of forecast timesteps.
    ground_truth : torch.Tensor or np.ndarray, optional
        The actual future data to compare the forecast against.
    x : Sequence, optional
        X-axis values (e.g., timestamps or indices) for the data. The sequence must be slicable.
    quantiles : tuple[float], optional
        A tuple indicating the quantile levels to use to plot as shaded areas
        around the median forecast. Set to None to deactivate. Default is (0.1, 0.9).
    quantile_levels : tuple[float], optional
        A tuple indicating the quantile levels.
    engine : str, optional
        What framework to use for rendering the plots.
    max_context_to_show : int, optional
        Caps how many context points are visible by zooming the x-axis to where the
        forecast begins; no context data is cut away. Defaults to 3x the forecast horizon.
    ax : matplotlib.Axes or plotly.go.Figure, optional
        The matplotlib axes / plotly figure object to plot on.
    **kwargs
        Additional keyword arguments to pass to the plotting functions.
    Returns
    -------
    matplotlib.Axes
        The Axes object with the plotted forecast, if engine="matplotlib"
    plotly.go.Figure
        The Figure object with the plotted forecast, if engine="plotly"
    """

    if context is None and forecasts is None and ground_truth is None:
        raise ValueError("At least one of context, forecasts, or ground_truth must be provided for plotting.")

    if quantiles is not None and len(quantiles) != 2:
        raise ValueError(
            "quantiles must either be a collection of two values for min- and max quantile, respectively, or None."
        )

    # determine all lenghts for clarity
    context_size = len(context) if context is not None else 0
    forecast_size = forecasts.shape[-1] if forecasts is not None else 0
    ground_truth_size = len(ground_truth) if ground_truth is not None else 0
    full_size = context_size + max(forecast_size, ground_truth_size)

    if x is None:
        x = np.arange(full_size)
    elif len(x) < full_size:
        raise ValueError(
            "Not enough 'x' values provided to have one for every timestep in context, forecast, and ground truth window."
        )

    x_context = x[:context_size] if context is not None else None

    # Zoom the view to where the forecast begins without cutting any context data
    # away. `max_context_to_show` caps how many context points are visible; the
    # full context is still plotted, only the x-axis range is narrowed. Default to
    # showing at most 3x the forecast horizon of context.
    if max_context_to_show is None:
        max_context_to_show = 3 * max(forecast_size, ground_truth_size)

    view_min_x = None
    if context is not None and 0 < max_context_to_show < context_size:
        view_min_x = x_context[-max_context_to_show]

    def connect_to_context(v, is_x_axis=False):
        if CONNECT_FORECAST_TO_CONTEXT and context is not None and len(context) > 0:
            context_data = x_context if is_x_axis else context
            return np.hstack([np.array(context_data)[-1:], v])
        return v

    plot_params = dict(
        context=context,
        x_context=x_context,
        label_context="Context",
        label_ground_truth="Ground Truth Future",
        label_forecast="Forecast (Median)",
        view_min_x=view_min_x,
    )

    if ground_truth is not None:
        plot_params.update(
            dict(
                ground_truth=connect_to_context(ground_truth, is_x_axis=False),
                x_ground_truth=connect_to_context(x[context_size : context_size + ground_truth_size], is_x_axis=True),
            )
        )

    # plot forecasts if supplied
    # forecasts are a 2D array with quantiles as rows, and data for each timestep as columns
    if forecasts is not None:
        median_index = quantile_levels.index(0.5)
        plot_params["point_forecast"] = connect_to_context(forecasts[median_index, :], is_x_axis=False)
        if quantiles is not None:
            min_quantile, max_quantile = quantiles
            min_quantile_index, max_quantile_index = (quantile_levels.index(q) for q in quantiles)
            plot_params.update(
                dict(
                    x_forecast=connect_to_context(x[context_size : context_size + forecast_size], is_x_axis=True),
                    lower_quantile=connect_to_context(forecasts[min_quantile_index, :], is_x_axis=False),
                    upper_quantile=connect_to_context(forecasts[max_quantile_index, :], is_x_axis=False),
                    label_quantile=f"Forecast {min_quantile * 100:.0f}% - {max_quantile * 100:.0f}% Quantiles",
                )
            )

    match engine:
        case "matplotlib":
            return _plot_forecast_matplotlib(**plot_params, ax=ax, **kwargs)
        case "plotly":
            return _plot_forecast_plotly(**plot_params, fig=fig, **kwargs)
        case _:
            raise ValueError(f"Drawing {engine=} not supported.")


def _plot_covariates_plotly(
    covariates,
    fig=None,
    row=None,
    col=None,
    cutoff_x=None,
    color_cutoff_line="black",
):
    """
    Plots multivariate covariates into a specified Plotly figure/subplot.

    Args:
        covariates (dict): Format -> {'covariate_name': {'x': x_array, 'y': y_array}}
        fig (go.Figure): Plotly figure object.
        row (int): Row index for the subplot.
        col (int): Column index for the subplot.
        cutoff_x (float/datetime): The x-axis value where the forecast starts.
        color_cutoff_line (str): Color of the vertical cutoff line.
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError("Plotly is required. Please install it.") from e

    if fig is None:
        fig = go.Figure()

    if not covariates:
        return fig

    # Default qualitative palette for multiple covariates
    colors = px.colors.qualitative.Plotly

    for idx, (name, data) in enumerate(covariates.items()):
        x_cov = data.get("x", [])
        y_cov = data.get("y", [])

        fig.add_trace(
            go.Scatter(
                x=x_cov,
                y=y_cov,
                name=name,
                mode="lines",
                line={"width": 1.5, "color": colors[idx % len(colors)]},
            ),
            row=row,
            col=col,
        )

    # Mirror the vertical cutoff line if a cutoff point is provided
    if cutoff_x is not None:
        fig.add_vrect(
            x0=cutoff_x,
            x1=cutoff_x,
            line_width=1.5,
            line_dash="0px,7px,4px,4px",
            line_color=color_cutoff_line,
            annotation_text="forecast start",
            row=row,
            col=col,
        )

    return fig


def _plot_covariate_matplotlib(
    covariate: torch.Tensor | np.ndarray | None,
    label: str | None,
    color: str | None,
    x: Sequence | None = None,
    ax=None,
):
    try:
        from matplotlib import pyplot as plt
    except ImportError as e:
        _raise_matplotlib_installation_error(e)

    # Create a new figure and axis if one isn't provided
    if ax is None:
        _, ax = plt.get_gca()

    ax.plot(
        x,
        covariate,
        label=label,
        color=color,
        linewidth=2,
        linestyle="-",
    )

    if label is not None:
        ax.legend()

    return ax


def _plot_covariate_plotly(
    covariate: torch.Tensor | np.ndarray | None,
    label: str | None,
    color: str,
    x: Sequence | None = None,
    fig=None,
    row=None,
    col=None,
):
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        _raise_plotly_installation_error(e)

    if fig is None:
        fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=covariate,
            name=label,
            line={
                "color": color,
                "width": 2,
                "dash": "solid",
            },
            marker=None,
            showlegend=label is not None,
        ),
        row=row,
        col=col,
    )
    return fig


def plot_covariate(
    covariate: torch.Tensor | np.ndarray | None,
    label: str | None = None,
    color: str = COVARIATE_COLORS[0],
    x: Sequence | None = None,
    engine="plotly",
    ax=None,
    fig=None,
    **kwargs,
):
    covariate_size = len(covariate)
    if x is None:
        x = np.arange(covariate_size)
    elif len(x) < covariate_size:
        raise ValueError("Not enough 'x' values provided to have one for every timestep of the covariate.")
    x = x[:covariate_size]

    plot_params = dict(
        x=x,
        covariate=covariate,
        label=label,
        color=color,
    )

    match engine:
        case "matplotlib":
            return _plot_covariate_matplotlib(**plot_params, ax=ax, **kwargs)
        case "plotly":
            return _plot_covariate_plotly(**plot_params, fig=fig, **kwargs)
        case _:
            raise ValueError(f"Drawing {engine=} not supported.")


def _plot_multivariate_matplotlib(
    context=None,
    forecast=None,
    ground_truth=None,
    quantiles=None,
    quantile_levels=None,
    x=None,
    cov_lookup: dict[str, tuple[torch.Tensor | np.ndarray]] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
):
    try:
        from matplotlib import pyplot as plt
    except ImportError as e:
        _raise_matplotlib_installation_error(e)

    if cov_lookup is None:
        cov_lookup = {}

    n_covariates = len(cov_lookup)
    if n_covariates > 0:
        height_ratios = [n_covariates] + [1] * n_covariates
    else:
        height_ratios = [1]

    # create figure holding the target forecast and covariates
    fig_width = 10
    fig_height = 3 + 2 * n_covariates
    fig, axes = plt.subplots(
        nrows=1 + n_covariates,
        ncols=1,
        sharex=True,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": height_ratios},
        layout="constrained",
        squeeze=False,
    )
    axes = axes.flatten()

    # set figure title and subtitle if provided
    if title or subtitle:
        title_elements = [t for t in [title, subtitle] if t is not None]
        fig.suptitle("\n".join(title_elements))

    # plot target forecast
    target_ax = axes[0]
    target_ax.set_ylabel("Target")

    plot_forecast(
        context=context,
        forecasts=forecast,
        ground_truth=ground_truth,
        quantiles=quantiles,
        quantile_levels=quantile_levels,
        x=x,
        engine="matplotlib",
        ax=target_ax,
    )

    # plot covariates
    for i, (lbl, (xc, cov)) in enumerate(cov_lookup.items()):
        cov_ax = axes[i + 1]
        cov_ax.set_ylabel(f"Cov. {i + 1}")

        plot_covariate(
            cov,
            x=xc,
            label=lbl,
            engine="matplotlib",
            ax=cov_ax,
            color=COVARIATE_COLORS[i],
        )

    return fig


def _plot_multivariate_plotly(
    context=None,
    forecast=None,
    ground_truth=None,
    quantiles=None,
    quantile_levels=None,
    x=None,
    cov_lookup: dict[str, tuple[torch.Tensor | np.ndarray]] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
):
    try:
        from plotly.subplots import make_subplots
    except Exception as e:
        _raise_plotly_installation_error(e)

    if cov_lookup is None:
        cov_lookup = {}

    n_covariates = len(cov_lookup)
    if n_covariates > 0:
        target_height = 0.5
        covariate_heights = [0.5 / n_covariates] * n_covariates
        covariate_labels = [f"Cov. {i + 1}" for i in range(n_covariates)]
    else:
        target_height = 1.0
        covariate_heights = []
        covariate_labels = []

    fig = make_subplots(
        rows=1 + n_covariates,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[target_height] + covariate_heights,
        row_titles=["Target"] + covariate_labels,
    )

    plot_forecast(
        context=context,
        forecasts=forecast,
        ground_truth=ground_truth,
        quantiles=quantiles,
        quantile_levels=quantile_levels,
        x=x,
        engine="plotly",
        fig=fig,
        col=1,
        row=1,
    )

    for i, (lbl, (xc, cov)) in enumerate(cov_lookup.items()):
        plot_covariate(
            cov,
            x=xc,
            label=lbl,
            fig=fig,
            engine="plotly",
            col=1,
            row=i + 2,
            color=COVARIATE_COLORS[i],
        )

    fig.update_layout(height=300 + 200 * n_covariates, title=dict(text=title, subtitle_text=subtitle))
    return fig


def plot_multivariate(
    input: TimeseriesType,
    forecast: torch.Tensor | np.ndarray | None = None ,
    ground_truth: torch.Tensor | np.ndarray | None = None,
    x: Sequence | None = None,
    quantiles: tuple[float, float] = (0.1, 0.9),
    quantile_levels: tuple[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    max_context_to_show: int | None = None,
    target_index: int = 0,
    past_cov_labels: list[str] | None = None,
    future_cov_labels: list[str] | None = None,
    engine="plotly",
    title: str | None = None,
    subtitle: str | None = None,
):
    # determine sizes of displayed time series
    forecast_length = forecast.shape[-1] if forecast is not None else 0
    ground_truth_length = len(ground_truth) if ground_truth is not None else 0
    max_future_length = max(forecast_length, ground_truth_length, input.future_length)
    full_size = input.past_length + max_future_length

    if x is None:
        x = np.arange(full_size)
        # x = np.arange(-input.past_length, max_future_length) + 1
    elif len(x) < full_size:
        raise ValueError(
            "Not enough 'x' values provided to have one for every timestep in context, forecast, and ground truth window."
        )

    # prepare covariate slices and labels
    cov_slice_start = 0 if max_context_to_show is None else input.past_length - max_context_to_show
    past_covariates = input.past_covariates if input.past_covariates is not None else []
    future_covariates = input.future_covariates if input.future_covariates is not None else []
    if len(past_covariates) > 0:
        if past_cov_labels is not None and len(past_cov_labels) != len(past_covariates):
            raise ValueError("Length of 'past_cov_labels' must match the number of past covariates.")
        elif past_cov_labels is None:
            past_cov_labels = []
            for i, cov in enumerate(past_covariates):
                past_cov_labels.append(f"Past Covariate {i + 1}")
    else:
        past_cov_labels = []

    if len(future_covariates) > 0:
        if future_cov_labels is not None and len(future_cov_labels) != len(future_covariates):
            raise ValueError("Length of 'future_cov_labels' must match the number of future covariates.")
        elif future_cov_labels is None:
            future_cov_labels = []
            for i, cov in enumerate(future_covariates):
                future_cov_labels.append(f"Future Covariate {i + 1}")
    else:
        future_cov_labels = []

    cov_lookup = {}
    for i, (lbl, cov) in enumerate(
        chain(
            zip(past_cov_labels, past_covariates),
            zip(future_cov_labels, future_covariates),
        )
    ):
        cov_lookup[lbl] = (
            x[cov_slice_start : cov_slice_start + len(cov)],
            cov[cov_slice_start : cov_slice_start + len(cov)],
        )

    # prepare plot parameters for the plotting functions
    plot_params = dict(
        context=input.target[target_index, cov_slice_start:],
        x=x[cov_slice_start:],
        quantiles=quantiles,
        quantile_levels=quantile_levels,
        forecast=forecast[target_index] if forecast is not None else None,
        ground_truth=ground_truth,
        cov_lookup=cov_lookup,
        title=title,
        subtitle=subtitle,
    )

    # call appropriate plotting function based on the specified engine
    match engine:
        case "matplotlib":
            return _plot_multivariate_matplotlib(**plot_params)
        case "plotly":
            return _plot_multivariate_plotly(**plot_params)
        case _:
            raise ValueError(f"Drawing {engine=} not supported.")
