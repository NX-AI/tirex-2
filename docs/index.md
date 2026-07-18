# TiRex-2

**Generalizing TiRex to Multivariate Data and Streaming**

[![Paper](https://img.shields.io/static/v1?label=Paper&message=2607.01204&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2607.01204)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-TiRex--2-yellow?logo=huggingface)](https://huggingface.co/NX-AI/TiRex-2)
[![GitHub](https://img.shields.io/badge/GitHub-NX--AI%2Ftirex--2-181717?logo=github)](https://github.com/NX-AI/tirex-2)
[![PyPI](https://img.shields.io/pypi/v/tirex-2?color=blue)](https://pypi.org/project/tirex-2/)
[![Docker](https://img.shields.io/badge/GHCR-tirex2--cpu%20%2F%20tirex2--gpu-2496ED?logo=docker&logoColor=white)](https://github.com/NX-AI/tirex-2/pkgs/container/tirex2-cpu)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NX-AI/tirex-2/blob/main/examples/getting_started.ipynb)

TiRex-2 is a **pretrained time series foundation model** that forecasts one or many target
variates directly from their history, optionally conditioned on past and future-known
covariates. A single checkpoint serves both univariate and multivariate forecasting, built
on a recurrent architecture designed for efficient streaming settings — all zero-shot, with
no task-specific training or fine-tuning.

TiRex-2 generalizes the original univariate model, [TiRex](https://github.com/NX-AI/tirex), to
multivariate forecasting with past and future covariates. See the [Introduction](introduction.md)
for background and the [paper](https://arxiv.org/pdf/2607.01204) for details.

!!! info "Looking for TiRex-1?"
    This site documents **TiRex-2**. Documentation for the original univariate TiRex model
    lives at [nx-ai.github.io/tirex](https://nx-ai.github.io/tirex/) — the two are separate
    projects and this site does not attempt to unify them.

## Key facts

- **Zero-shot multivariate forecasting** — TiRex-2 forecasts multiple target variates out of
  the box, without training or fine-tuning on your data.
- **Past and future-known covariates** — TiRex-2 natively conditions on past covariates and
  future-known covariates, such as calendar features, holidays, promotions, or scheduled
  interventions.
- **Small active footprint** — TiRex-2 activates 38.4M parameters in univariate mode and an
  additional 44.1M parameters for multivariate forecasting.

## Where to go next

- [Installation](getting-started/install.md) — `pip install tirex-2`, Pixi setup, and gated
  Hugging Face weight access.
- [Quickstart](getting-started/quickstart.md) — minimal sine-wave and covariate examples.
- [How-to guides](how-to/forecasting.md) — univariate/multivariate forecasting, covariates in
  depth, and what streaming does (and doesn't) mean in this open-source release.
- [Deployment](deployment.md) — the Docker-based HTTP/MQTT/MCP inference server.
- [API reference](api/index.md) — generated reference for the public `tirex2` API.

## TiRex-2 Pro

This repository is NX-AI's open-source release. A Pro version extends TiRex-2 with:

- **Streaming**: incremental forecast updates as new observations arrive, without recomputing
  over the full history.
- **Speed**: performance-optimized inference, including optimization for dedicated hardware
  such as edge, embedded, and industrial PC deployments.
- **Finetuning**: models fine-tuned on your data or with different pretraining.
- **Classification & Regression**: TiRex-2 adapted for classification and regression tasks.

See [TiRex-2 Pro](pro.md) for details, or contact [contact@nx-ai.com](mailto:contact@nx-ai.com).
