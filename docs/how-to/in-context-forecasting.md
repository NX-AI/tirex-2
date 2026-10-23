# In-context learning

**In-context learning (ICL)** allows TiRex-2 to adapt to your data through the
context you provide, without additional training or fine-tuning. A richer context
can improve forecast accuracy in two complementary ways:

- extending the **historical timeline** to capture recurring patterns;
- incorporating **related signals** that provide additional information about the target.

This guide demonstrates both approaches using a real industrial signal. First, we explore how a longer history helps TiRex-2 recognize recurring cycles and improve its forecast. Then, we introduce a strongly correlated signal as a covariate to show how additional information can further improve predictions.

!!! tip "The plots are interactive"
    Hover for exact values, click legend entries to show or hide traces, and use the sliders
    to change the context length.

## The data

The data comes from the UCI *Condition monitoring of hydraulic systems* dataset[^uci], from which we select two sensors: hydraulic pressure and motor power. We consider two consecutive, stable operating cycles, using the first as input context and the second as ground truth for the forecast. Each cycle captures a 60s run that has been averaged into 200 data points at a 0.3s resolution.

<iframe src="../../images/in-context/context-vs-target.html" title="Context cycle 1787 vs. target cycle 1788"
        loading="lazy" style="width:100%; height:660px; border:0;"></iframe>

Each cycle follows the same fixed load profile, so the pattern repeats across both cycles. Although hydraulic pressure and motor power operate on very different scales, they vary together.

## Forecasting the next cycle

We give TiRex-2 the first cycle as context repeated *n* times and ask for predicting the next 200
points:

```python
import torch
from tirex2 import TimeseriesType, load_model

model = load_model("NX-AI/TiRex-2", device="cpu")  # or device="cuda"

context_cycle = torch.as_tensor(pressure[1787], dtype=torch.float32)  # hydraulic pressure, cycle 1787: (200,)
target_cycle = torch.as_tensor(pressure[1788], dtype=torch.float32)   # ground truth, cycle 1788: (200,)

n = 1                                       # number of repeated cycles in the context
context = context_cycle.repeat(n)           # (n * 200,)

ts = TimeseriesType(
    target=context.unsqueeze(0),            # (1, context_length)
    past_covariates=None,
    future_covariates=None,
)

forecast = model.forecast([ts], prediction_length=200, output_type="torch")[0]

median, p10, p90 = forecast[0, 4], forecast[0, 0], forecast[0, 8]
mae = (median - target_cycle).abs().mean().item()  # mean absolute error against the real next cycle, in bar
```

The 9 quantiles correspond to levels `0.1, 0.2, ..., 0.9`. We plot the median (index `4`) as the point forecast and use the `0.1` and `0.9` quantiles (indices `0` and `8`) to define the prediction band. Forecast accuracy is evaluated by comparing the median with the actual values of cycle 1788 using the **mean absolute error (MAE)**, which measures the average absolute difference between predicted and observed values.

## One cycle of context

With a single cycle of context, TiRex-2 has seen the shape only once:

<iframe src="../../images/in-context/forecast-one-cycle.html" title="Hydraulic pressure forecast from one cycle of context"
        loading="lazy" style="width:100%; height:635px; border:0;"></iframe>

The forecast is flat and cautious, with a wide band (MAE = 11.02 bar). From one cycle alone,
the model cannot tell that the pattern will repeat.

## ICL: extending the context

The more cycles we include in the context, the more evidence the model has of how the signal evolves over time. By expanding the context from  *n* = 1 to 5 load cycles (`context_cycle.repeat(n)`), we can explore how additional history affects the model's ability to capture recurring patterns. 

!!! tip "Try it"
    Move the slider, or press play, to see how the forecast changes with *n*.

<iframe src="../../images/in-context/forecast-context-slider.html" title="Hydraulic pressure forecast as the context grows"
        loading="lazy" style="width:100%; height:610px; border:0;"></iframe>

Extending the context from one to five cycles reduces the MAE by up to 88%, with most of the improvement occurring within the first three to four cycles.

## ICL: expand the variates

ICL can also benefit from information across variates, not just across time. TiRex-2 can use related time series to help forecast the target, so here we provide motor power (W) as a `past_covariates` for hydraulic pressure. Both signals cover the same historical window, allowing the model to use their relationship when making its forecast (see [Covariates](covariates.md)):

```python
ts = TimeseriesType(
    target=context_cycle.repeat(n).unsqueeze(0),                                         # (1, context_length)
    past_covariates=torch.as_tensor(motor[1787], dtype=torch.float32).repeat(n).unsqueeze(0),  # (1, context_length)
    future_covariates=None,
)

forecast = model.forecast([ts], prediction_length=200, output_type="torch")[0]
# forecast.shape == (1, 9, 200)  -> only the target is forecast, not the covariate
```

The green forecast uses hydraulic pressure **and** motor power, while the red one uses hydraulic
pressure only. The lower panel shows the
extra signal the model sees:

<iframe src="../../images/in-context/forecast-covariate-slider.html" title="Univariate vs. covariate-informed hydraulic pressure forecast"
        loading="lazy" style="width:100%; height:810px; border:0;"></iframe>

Adding motor power as a covariate reduces the MAE by up to 42% compared with the univariate forecast, with the largest improvement at 2 cycles of context. The benefit gradually decreases to 19%, 13% and 5% as the hydraulic pressure history grows longer.

## Takeaways

- **Periodic signals benefit from context spanning multiple cycles.**
- **Longer context improves forecasts at little additional cost.** Extending the context from 1 to 5 cycles reduces the MAE by 88%.
- **Covariates can further improve forecasts, especially when historical context is limited.** Adding motor power reduces the MAE by 42% with 2 cycles of context.

[^uci]: N. Helwig, E. Pignanelli, A. Schütze. *Condition monitoring of hydraulic systems*.
    UCI Machine Learning Repository, 2015.
    <https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems>
