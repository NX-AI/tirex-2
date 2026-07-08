<h1 align="center">
  <img src="https://raw.githubusercontent.com/NX-AI/tirex-2/refs/heads/main/docs/images/tirex.svg" alt="TiRex emoji" height="48" /> TiRex-2: Generalizing TiRex to Multivariate Data and Streaming
</h1>

This repository provides the pre-trained multivariate forecasting model TiRex-2 introduced in the paper [TiRex-2: Generalizing TiRex to Multivariate Data and Streaming](https://arxiv.org/pdf/2607.01204).

## TiRex-2

TiRex-2 is a **pretrained time series foundation model** that forecasts one or many target
variates directly from their history, optionally conditioned on past and future-known
covariates. A single checkpoint serves both univariate and multivariate forecasting and
operates in a streaming fashion as new observations arrive — all zero-shot, with no
task-specific training or fine-tuning.

## Key facts

- **Zero-shot multivariate forecasting**:
  TiRex-2 forecasts multiple target variates out of the box, without training or fine-tuning on you data.

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

TiRex is currently only tested on Linux and MacOS.

### Via Pixi
We use [Pixi](https://pixi.prefix.dev/latest/) for our development and benchmarking environment to ensure that it is set up correctly. Run the following command to install it on your machine:
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### Installation on MacOS/Windows

For macOS and Windows platform please use [uv](https://docs.astral.sh/uv/) package

```bash
uv pip install tirex-2
```

## Getting started

The most easy way for you to get started is by checking out our ["Getting Started" notebook](examples/getting_started.ipynb). Moreover, you can jump straight into testing out TiRex using [Google Colab](https://colab.research.google.com/github/NX-AI/tirex-2/blob/main/examples/getting_started.ipynb). If you have cloned this repository, you can also easily start the notebook via Pixi by running:
```bash
pixi run notebook
```
Note that for `pixi`, depending on your CUDA version and use-case, you may need to use another environment, e.g., `-e example-cu128`, that are defined in [pyproject.toml](pyproject.toml) under section `tool.pixi.environments`.


### Minimal usage predicting a simple sine wave
```python
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
_ = plot_multivariate(ts, forecast, engine="matplotlib")
```
![output of plot_multivariate function visualizing context and forecast](/resources/sine-wave-prediction.png)

### Covariate example
This example originates from the "Getting Started" notebook, showing the value of additional covariates.
```python
from tirex2 import load_model
from tirex2.demo import Demo
from tirex2.plotting import plot_multivariate

# load model
model = load_model("NX-AI/TiRex-2", device="cpu")  # use `device="cuda"` if cuda is available

# load data
demo_nonstationary = Demo.create_nonstationary_demo()
ts = demo_nonstationary.to_timeseries_type()

# perform forecast - each forecast is of shape (n_targets, 9 quantiles, prediction_length)
forecast = model.forecast([ts], prediction_length=demo_nonstationary.horizon, output_type="numpy")[0]

# visualize result
_ = plot_multivariate(ts, forecast, engine="matplotlib")
```
![output of plot_multivariate function visualizing context and forecast of multivariate input](/resources/multivariate-prediction.png)



### Benchmarking
To reproduce our results for the [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) and [fev-bench](https://huggingface.co/spaces/autogluon/fev-bench) leaderboards, follow the instructions in
[/examples/gifteval/](./examples/gifteval/README.md) and [/examples/fevbench/](./examples/fevbench/README.md), respectively.

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
