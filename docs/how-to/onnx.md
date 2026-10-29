# ONNX

TiRex-2 is also available as an ONNX model, so you can run it with only `numpy` and
`onnxruntime`. You don't need PyTorch or the `tirex2` package. The file is `model.onnx` in the
[Hugging Face repository](https://huggingface.co/NX-AI/TiRex-2).

## Installation

```bash
pip install numpy onnxruntime huggingface_hub
```

## Forecasting

The model always reads a fixed window of 2368 steps: 2048 steps of context followed by 320
steps of horizon. The `forecast` function below fills your series into that window (padding
with NaN), runs the model, and returns the forecast horizon.

Both univariate and multivariate series are supported. For a multivariate series, pass one row
per variate; all variates are forecast jointly.

```python
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

CONTEXT_LEN, FUTURE_LEN, PATCH_SIZE = 2048, 320, 32
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

onnx_path = hf_hub_download("NX-AI/TiRex-2", "model.onnx")
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def forecast(session, series, prediction_length):
    """Return [9, prediction_length] quantiles for a 1D series, or
    [n_variates, 9, prediction_length] for a 2D (n_variates, length) series."""
    window = CONTEXT_LEN + FUTURE_LEN
    x = np.atleast_2d(np.asarray(series, dtype=np.float32))
    n_variates = x.shape[0]
    x = np.pad(x, ((0, 0), (0, FUTURE_LEN)), constant_values=np.nan)  # empty horizon
    x = np.pad(x, ((0, 0), (max(0, window - x.shape[1]), 0)), constant_values=np.nan)
    x = x[:, -window:]  # keep only the last 2048 context steps

    out = session.run(None, {
        "context": x,
        "group_vector": np.ones(n_variates, dtype=np.float32),  # same group = forecast jointly
        "target_mask": np.ones(n_variates, dtype=bool),
    })[0]
    out = out[:, :, -FUTURE_LEN - PATCH_SIZE:-PATCH_SIZE][..., :prediction_length]
    return out[0] if np.ndim(series) == 1 else out


t = np.arange(400)
series = np.sin(2 * np.pi * t / 48)

prediction = forecast(session, series, prediction_length=32)  # (9, 32)
median = prediction[QUANTILES.index(0.5)]
low, high = prediction[0], prediction[-1]  # 10% and 90% quantiles

# Multivariate: all variates are forecast jointly
multivariate = np.stack([series, np.cos(2 * np.pi * t / 48)])  # (2, 400)
prediction = forecast(session, multivariate, prediction_length=32)  # (2, 9, 32)
```

- `series` can be a 1D array, or a 2D array of shape `(n_variates, length)` for a
  multivariate series. Use NaN for missing values.
- `prediction_length` can be at most 320.

!!! note "Differences to the `tirex2` package"

    The ONNX file contains the model only. The `tirex2` package additionally removes linear
    trends before forecasting and recalibrates the quantile band afterwards. For series without
    a strong trend, the median is the same and the quantiles are very close. For strongly
    trending series, use the [`tirex2` package](forecasting.md) if you need identical results.

## Running on a GPU

Install `onnxruntime-gpu[cuda,cudnn]` instead of `onnxruntime`, and turn TF32 off. ONNX Runtime
enables TF32 by default, which reduces accuracy for this model:

```python
session = ort.InferenceSession(
    onnx_path,
    providers=[("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"],
)
```
