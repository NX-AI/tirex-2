# GiftEval benchmark

Evaluate a TiRex2 model on the [GiftEval](https://github.com/SalesforceAIResearch/gift-eval)
benchmark.

## Data

Download the GiftEval datasets once:

```bash

pixi run -e examples --platform linux-64-cuda huggingface-cli download Salesforce/GiftEval --repo-type=dataset --local-dir PATH_TO_SAVE
```

## Run

Script (required positional arguments: the GiftEval storage directory and model type):

```bash
pixi run -e examples --platform linux-64-cuda python examples/gifteval/run_gifteval.py </path/to/gifteval_storage> pretrained
```

Model type options:

- `pretrained` loads `NX-AI/TiRex-2-gifteval-pretrain`.
- `zero-shot` loads `NX-AI/TiRex-2-gifteval-zs`.

Optional arguments:

- `--out RESULTS.csv` — where to write the results (default: `./gifteval_results.csv`).
- `--device` — device to run on (default: `cuda`).
- `--eval-mode {univariate,multivariate}` — how multivariate datasets are scored
  (default: `multivariate`). See below.

## Univariate vs. multivariate evaluation

By default (`--eval-mode multivariate`) the native `[V, T]` target is kept intact and
all variates of a series are fed to the model jointly, so the variate-mixing blocks are
used. One forecast is produced per series with a trailing variate axis; metrics are
computed against the multivariate label. Univariate datasets (`num_variates == 1`) are
unaffected and score identically in either mode.

With `--eval-mode univariate` the benchmark instead follows the standard GIFT-Eval
protocol: every multivariate dataset is split into independent univariate channels
(`MultivariateToUnivariate`) and each channel is forecast on its own. This matches the
official leaderboard numbers but never exercises the model's cross-variate path.

```bash
# univariate scoring (matches the public GIFT-Eval leaderboard protocol)
pixi run -e examples --platform linux-64-cuda-126 python examples/gifteval/run_gifteval.py \
    </path/to/gifteval_storage> <ckpt_dir> --eval-mode univariate
```

Note: the default multivariate numbers are **not** comparable to the public GIFT-Eval
leaderboard, since the leaderboard baselines are computed in univariate mode — use
`--eval-mode univariate` for a leaderboard-comparable run.

To run GIFT-Eval in an interactive manner, start jupyter lab
```bash
pixi run --platform linux-64-cuda notebook
```
and then open `./examples/gifteval/gifteval.ipynb`.
