# Introduction

TiRex-2 is introduced in the paper
[TiRex-2: Generalizing TiRex to Multivariate Data and Streaming](https://arxiv.org/pdf/2607.01204)
(arXiv:2607.01204).

## From TiRex to TiRex-2

The original [TiRex](https://github.com/NX-AI/tirex) model is a univariate, zero-shot time
series forecasting model built on the [xLSTM](https://arxiv.org/abs/2405.04517) architecture.
TiRex-2 generalizes it along two axes:

- **Multivariate forecasting**: a single checkpoint forecasts one or many target variates
  jointly, and can condition on past covariates and future-known covariates (e.g. calendar
  features, holidays, promotions, or scheduled interventions) alongside the target history.
- **Streaming-oriented architecture**: TiRex-2 is built on a recurrent architecture (extending
  the xLSTM-based design) chosen for efficient streaming settings.

Both univariate and multivariate forecasting are served zero-shot, without any task-specific
training or fine-tuning, from the same pretrained checkpoint published on
[Hugging Face](https://huggingface.co/NX-AI/TiRex-2).

## What "streaming-oriented" means in this release

The recurrent architecture is what makes efficient incremental inference possible in
principle, but this open-source release does not itself expose an incremental,
state-carrying forecast API — every call to
[`forecast`][tirex2.api_adapter.forecast.ForecastModel.forecast] recomputes over the full
context array you pass in. Incremental (no-recompute) streaming updates are part of
[TiRex-2 Pro](pro.md). See [How-to: Streaming](how-to/streaming.md) for the full explanation.

## Citation

If you use TiRex-2 in your research, please cite:

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
