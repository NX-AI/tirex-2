<h1 align="center">
  <img src="https://raw.githubusercontent.com/NX-AI/tirex-2/refs/heads/main/docs/images/tirex.svg" alt="TiRex emoji" height="48" /> TiRex-2: Generalizing TiRex to Multivariate Data and Streaming
</h1>

<div align="center">

[![Paper](https://img.shields.io/static/v1?label=Paper&message=2607.01204&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2607.01204)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-TiRex--2-yellow?logo=huggingface)](https://huggingface.co/NX-AI/TiRex-2)
[![PyPI](https://img.shields.io/pypi/v/tirex-2?color=blue)](https://pypi.org/project/tirex-2/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/tirex-2?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/tirex-2)
[![Docker](https://img.shields.io/badge/GHCR-tirex2--cpu%20%2F%20tirex2--gpu-2496ED?logo=docker&logoColor=white)](https://github.com/NX-AI/tirex-2/pkgs/container/tirex2-cpu)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![docs](https://img.shields.io/website?url=https%3A%2F%2Fnx-ai.github.io%2Ftirex-2%2F&label=docs&up_message=online&up_color=green&down_message=offline&down_color=red)](https://nx-ai.github.io/tirex-2/)
[![Tests](https://github.com/NX-AI/tirex-2/actions/workflows/test.yaml/badge.svg)](https://github.com/NX-AI/tirex-2/actions/workflows/test.yaml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NX-AI/tirex-2/blob/main/examples/getting_started.ipynb)
[![TiRex-2 Demo](https://img.shields.io/badge/HuggingFace-TiRex--2%20Demo-yellow?logo=huggingface)](https://huggingface.co/spaces/NX-AI/TiRex-2-demo)

</div>

This repository provides the pre-trained multivariate forecasting model TiRex-2 introduced in the paper [TiRex-2: Generalizing TiRex to Multivariate Data and Streaming](https://arxiv.org/pdf/2607.01204).

> **TiRex-2 Pro:** This repository is our open-source release. Our pro version extends TiRex-2 with streaming, hardware-optimized inference (edge, embedded, and industrial PCs, among others), finetuning, and classification & regression support — see [TiRex-2 Pro](#tirex-2-pro) below or contact us at [contact@nx-ai.com](mailto:contact@nx-ai.com).

## TiRex-2

TiRex-2 is a **pretrained time series foundation model** that forecasts one or many target
variates directly from their history, optionally conditioned on past and future-known
covariates. A single checkpoint serves both univariate and multivariate forecasting, built
on a recurrent architecture designed for efficient streaming settings — all zero-shot, with
no task-specific training or fine-tuning.

TiRex-2 generalizes our original univariate model, [TiRex](https://github.com/NX-AI/tirex), to
multivariate forecasting with past and future covariates.

## Key facts

- **Zero-shot multivariate forecasting**:
  TiRex-2 forecasts multiple target variates out of the box, without training or fine-tuning on your data.

- **Past and future-known covariates**:
  TiRex-2 natively conditions on past covariates and future-known covariates, such as
  calendar features, holidays, promotions, or scheduled interventions.

- **Small active footprint**:
  TiRex-2 activates 38.4M parameters in univariate mode and an additional 44.1M parameters
  for multivariate forecasting.

## Installation
### Via Pip
```bash
pip install tirex-2
```

Install with additional dependencies:

```bash
pip install "tirex-2[examples,fev,gluonts]"
```

The Python package installation is currently only tested on Linux and macOS. Docker usage is documented separately and includes Linux, macOS, and Windows Docker Desktop instructions.

### Via Pixi
We use [Pixi](https://pixi.prefix.dev/latest/) for our development and benchmarking environment to ensure that it is set up correctly. Run the following command to install it on your machine:
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

## FAQ

<details>
<summary><strong>Why does TiRex-2 fail with <code>where cl</code> on Windows?</strong></summary>

PyTorch may compile model components at runtime, so the Python process needs access to the MSVC C++ compiler, even when using `device="cpu"`.

Install Visual Studio Build Tools with **Desktop development with C++**, then run TiRex-2 from an **x64 Native Tools Command Prompt for Visual Studio**. Confirm the compiler is available before starting your script:

```bat
where cl
python your_script.py
```

Launch VS Code or Jupyter from the same prompt so it inherits the compiler environment. See [#15](https://github.com/NX-AI/tirex-2/issues/15) and [#17](https://github.com/NX-AI/tirex-2/issues/17) for related reports.

</details>

## Getting started

The most easy way for you to get started is by checking out our ["Getting Started" notebook](examples/getting_started.ipynb). Moreover, you can jump straight into testing out TiRex using [Google Colab](https://colab.research.google.com/github/NX-AI/tirex-2/blob/main/examples/getting_started.ipynb). If you have cloned this repository, you can also easily start the notebook via Pixi by running:
```bash
pixi run notebook
```
Note that for `pixi`, depending on your CUDA version and use-case, you may need to use another environment, e.g., `-e example-cu128`, that are defined in [pyproject.toml](pyproject.toml) under section `tool.pixi.environments`.

### Minimal usage predicting a simple sine wave
```python
import torch
from tirex2 import TimeseriesType, load_model
from tirex2.plotting import plot_multivariate  # requires matplotlib to be installed

# load model
model = load_model("NX-AI/TiRex-2", device="cpu")  # use `device="cuda"` if cuda is available

# generate data - target expects time series of shape (n_targets, context_length)
context = torch.sin(torch.arange(128).float() / 8)
ts = TimeseriesType(target=context.unsqueeze(0), past_covariates=None, future_covariates=None)

# perform forecast - each forecast is of shape (n_targets, 9 quantiles, prediction_length)
forecast = model.forecast([ts], prediction_length=32, output_type="numpy")[0]

# visualize result
fig = plot_multivariate(ts, forecast, engine="matplotlib")
fig.show()
```
![output of plot_multivariate function visualizing context and forecast](/resources/sine-wave-prediction.png)

### Covariate example
This example originates from the "Getting Started" notebook, showing the value of additional covariates.
```python
from tirex2 import load_model
from tirex2.demo import Demo, plot_demo_forecast

# load model
model = load_model("NX-AI/TiRex-2", device="cpu")  # use `device="cuda"` if cuda is available

# load data
demo = Demo.create_nonstationary_demo()
ts_univariate = demo.to_timeseries_type(include_covariates=False)
ts_multivariate = demo.to_timeseries_type(include_covariates=True)

# perform forecast - each forecast is of shape (n_targets, 9 quantiles, prediction_length)
forecasts = model.forecast(
    timeseries=[ts_univariate, ts_multivariate],
    prediction_length=demo.horizon,
    output_type="numpy",
)

# visualize result
fig = plot_demo_forecast(demo, *forecasts, engine="matplotlib")
fig.show()
```
![output of plot_multivariate function visualizing context and forecast of multivariate input](/resources/multivariate-prediction.png)



### Benchmarking
To reproduce our results for the [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) and [fev-bench](https://huggingface.co/spaces/autogluon/fev-bench) leaderboards, follow the instructions in
[/examples/gifteval/](./examples/gifteval/README.md) and [/examples/fevbench/](./examples/fevbench/README.md), respectively.

## TiRex Docker image

For detailed instructions on building and running TiRex-2 in a Docker container, see the [Docker README](./inference/README.md).

## TiRex-2 Pro
TiRex-2 already provides state-of-the-art performance for zero-shot prediction, so you can use this open-source release without training on your own data.

Our pro version extends TiRex-2 with additional capabilities, including:

- **Streaming**: incremental forecast updates as new observations arrive, without recomputing over the full history.
- **Speed**: performance-optimized inference, including optimization for dedicated hardware such as edge, embedded, and industrial PC deployments.
- **Finetuning**: models fine-tuned on your data or with different pretraining.
- **Classification & Regression**: TiRex-2 adapted for classification and regression tasks.

If you are interested in any of these, please contact us at [contact@nx-ai.com](mailto:contact@nx-ai.com).

## Cite

If you use TiRex in your research, please cite our work:

```bibtex
@misc{podest2026tirex2generalizingtirexmultivariate,
      title={TiRex-2: Generalizing TiRex to Multivariate Data and Streaming},
      author={Patrick Podest and Marco Pichler and Elias Bürger and Levente Zólyomi and Bernhard Voggenberger and Wilhelm Berghammer and Daniel Klotz and Sebastian Böck and Günter Klambauer and Sepp Hochreiter},
      year={2026},
      eprint={2607.01204},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2607.01204},
}
```

## License

TiRex-2 is licensed under the [Apache License 2.0](./LICENSE).
