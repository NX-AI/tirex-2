# In-context forecasting

TiRex-2 forecasts zero-shot, so on some series it simply does not recognise the pattern, and the forecast is poor. This guide shows a simple way to fix that without any fine-tuning: take the series, repeat it several times back to back, and give the repeated sequence to the model as context. Each repetition shows the model the same pattern again, so it can learn from it in-context and forecast the next cycle correctly. We demonstrate this on a real hydraulic dataset: cycle 1787 is the context, repeated n times, and cycle 1788 is the ground truth we try to forecast. By varying n, you can see how the amount of context changes the forecast precision.

???+ info "Key terms"
    | Term | Meaning |
    | :--- | :------ |
    | **Cycle** | One 60s run of the rig's load profile, stored as 200 points (0.3s resolution). |
    | **PS1 / EPS1** | PS1 = Hydraulic pressure sensor (bar). EPS1 = motor power usage (W). |
    | **Context** | The history given to the model before making the prediction. Here: cycle 1787, repeated *n* times back to back. |
    | **Forecast horizon** | What the model predicts: the next full cycle (200 points = 60s), compared with the real cycle 1788. |
    | **10–90% band** | The range between the 0.1 and 0.9 quantile forecasts. The model expects the truth to fall inside it 80% of the time. |
    | **Covariate** | An extra series (here EPS1) given alongside the target (PS1) as additional context. |
    | **MAE** | Mean absolute error, `mean(|forecast - truth|)`, in bar. Lower is better. |

!!! tip "The plots are interactive"
    Hover for exact values, click legend entries to show or hide traces, and use the sliders
    to change the context length.

## The data

The data comes from the UCI
[Condition monitoring of hydraulic systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)
dataset. We use two of its sensors, PS1 (pressure) and EPS1 (motor power), and two consecutive
cycles in which the rig is healthy and stable: **1787** and **1788**. Each cycle is
block-averaged to 200 points. In the code below, `ps1[1787]` is the PS1 signal of cycle 1787,
an array of shape `(200,)`, and the same holds for `eps1`.

<iframe src="../../images/in-context/context-vs-target.html" title="Context cycle 1787 vs. target cycle 1788"
        loading="lazy" style="width:100%; height:660px; border:0;"></iframe>

Every cycle runs the same fixed load profile, so the two cycles are almost identical. PS1 and
EPS1 live on very different scales, but they move together.

## Forecasting the next cycle

We give TiRex-2 cycle 1787 as context repeated *n* times, and ask for predicting the next 200
points:

```python
import numpy as np
import torch
from tirex2 import TimeseriesType, load_model

model = load_model("NX-AI/TiRex-2", device="cpu")  # or device="cuda"

n = 1                                       # number of repeated cycles in the context
context = np.tile(ps1[1787], n)             # (n * 200,)

ts = TimeseriesType(
    target=torch.tensor(context, dtype=torch.float32).unsqueeze(0),  # (1, context_length)
    past_covariates=None,
    future_covariates=None,
)

forecast = model.forecast([ts], prediction_length=200, output_type="numpy")[0]

median, p10, p90 = forecast[0, 4], forecast[0, 0], forecast[0, 8]
mae = np.mean(np.abs(median - ps1[1788]))   # error against the real next cycle, in bar
```

The 9 quantiles are the levels `0.1, 0.2, ..., 0.9`. We plot the median (index `4`) as the
point forecast and use the `0.1` and `0.9` quantiles (indices `0` and `8`) as the band. We
score the median against the real cycle 1788 with the MAE.

## One cycle of context

With a single cycle of context, TiRex-2 has seen the shape only once:

<iframe src="../../images/in-context/forecast-one-cycle.html" title="PS1 forecast from one cycle of context"
        loading="lazy" style="width:100%; height:635px; border:0;"></iframe>

The forecast is flat and cautious, with a wide band (MAE = 11.02 bar). From one cycle alone,
the model cannot tell that the pattern will repeat.

## More context: repeating the cycle

Now we keep the same target, but repeat cycle 1787 *n* = 1 … 5 times
(`np.tile(ps1[1787], n)`). This adds **no new information** about the process. It only shows
the model that the signal is periodic, and how stable it is. Move the slider, or press play,
to see the forecast change with *n*:

<iframe src="../../images/in-context/forecast-context-slider.html" title="PS1 forecast as the context grows"
        loading="lazy" style="width:100%; height:610px; border:0;"></iframe>

### Forecast error for each context length

The chart shows the **MAE of the forecast of cycle 1788** for each context length, from 1 to
5 repeated cycles. The label on the right is the change from 1 to 5 cycles.

<iframe src="../../images/in-context/error-vs-context.html" title="Forecast error vs. context length"
        loading="lazy" style="width:100%; height:520px; border:0;"></iframe>

The MAE drops from 11.02 bar (1 cycle) to 5.58, 2.45, 1.64 and 1.29 bar (5 cycles). Most of
the gain comes in the first 3–4 cycles.

## Adding a covariate

TiRex-2 can also use other series while forecasting the target. Here we pass EPS1 of cycle 1787
as a past covariate. It is repeated like PS1, because `past_covariates` must have the same
length as the target context (see [Covariates](covariates.md)):

```python
ts = TimeseriesType(
    target=torch.tensor(np.tile(ps1[1787], n), dtype=torch.float32).unsqueeze(0),           # (1, context_length)
    past_covariates=torch.tensor(np.tile(eps1[1787], n), dtype=torch.float32).unsqueeze(0),  # (1, context_length)
    future_covariates=None,
)

forecast = model.forecast([ts], prediction_length=200, output_type="numpy")[0]
# forecast.shape == (1, 9, 200)  -> only the target is forecast, not the covariate
```

The green forecast uses PS1 **and** EPS1, the red one uses PS1 only. The lower panel shows the
extra signal the model sees:

<iframe src="../../images/in-context/forecast-covariate-slider.html" title="Univariate vs. covariate-informed PS1 forecast"
        loading="lazy" style="width:100%; height:810px; border:0;"></iframe>

### Univariate vs. covariate error

The **MAE of both forecasts** for each context length, side by side:

<iframe src="../../images/in-context/error-univariate-vs-covariate.html" title="Forecast error: univariate vs. covariate"
        loading="lazy" style="width:100%; height:520px; border:0;"></iframe>

MAE in bar of the forecast of cycle 1788 (lower is better):

| MAE (bar) | 1 cycle | 2 cycles | 3 cycles | 4 cycles | 5 cycles |
| :--- | ---: | ---: | ---: | ---: | ---: |
| univariate | 11.02 | 5.58 | 2.45 | 1.64 | 1.29 |
| +EPS1 covariate | 11.01 | 3.26 | 1.99 | 1.43 | 1.23 |
| covariate gain | 0% | **42%** | 19% | 13% | 5% |

## Takeaways

- **One cycle is not enough to recognise the pattern.** The forecast stays flat and cautious
  (MAE ≈ 11.0 bar).
- **Repeating the context is the cheapest improvement.** Going from 1 to 5 cycles lowers the
  error by about 88%, without adding any new information.
- **A covariate helps most when the context is short.** With 2 cycles, EPS1 cuts the error by
  42%. The gain shrinks to 5% at 5 cycles, once the target's own history is enough.
- **In practice:** if you only have 2–3 cycles of history, add related sensors as covariates.

!!! note "Scope"
    These results come from a single context/target pair (1787 → 1788) of one healthy
    operating condition. Read them as one example, not as a statistical result.

## Next steps

- [Forecasting](forecasting.md): the `TimeseriesType` input and forecast options.
- [Covariates](covariates.md): past and future-known covariates.
- [API reference](../api/index.md): full signatures and parameters.
