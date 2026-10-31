# Benchmarks

TiRex-2's reported results are reproducible against two public leaderboards. Full
instructions live in the repository next to the benchmark code:

- [GIFT-Eval](https://github.com/NX-AI/tirex-2/blob/main/examples/gifteval/README.md) —
  reproduce results on the
  [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) leaderboard.
- [fev-bench](https://github.com/NX-AI/tirex-2/blob/main/examples/fevbench/README.md) —
  reproduce results on the [fev-bench](https://huggingface.co/spaces/autogluon/fev-bench)
  leaderboard.

Run commands from the repository root. Benchmark tasks use the `benchmark` environment,
which includes the shared `fev` and `gluonts` features.

## GIFT-Eval

Download the dataset once:

```bash
pixi run --platform linux-64-cuda gifteval-download PATH_TO_SAVE
```

Run the benchmark, choosing a model type:

```bash
pixi run --platform linux-64-cuda gifteval /path/to/gifteval_storage pretrained
```

- `pretrained` loads `NX-AI/TiRex-2-gifteval-pretrain`.
- `zero-shot` loads `NX-AI/TiRex-2-gifteval-zs`.

By default (`--eval-mode multivariate`) the native multivariate target is kept intact and
scored jointly, which exercises TiRex-2's cross-variate path but is **not** directly
comparable to the public GIFT-Eval leaderboard. Pass `--eval-mode univariate` to split every
multivariate dataset into independent univariate channels, matching the official leaderboard
protocol:

```bash
pixi run --platform linux-64-cuda-126 gifteval \
    /path/to/gifteval_storage pretrained --eval-mode univariate
```

For the interactive notebook, run `pixi run -e benchmark jupyter lab examples/gifteval/gifteval.ipynb`
The `benchmark` environment includes Jupyter and all benchmark dependencies.

## fev-bench

Optionally pre-download the data:

```bash
pixi run --platform linux-64-cuda fevbench-download /path/to/fevbench/store
```

Run the benchmark — this always loads `NX-AI/TiRex-2-fevbench` from Hugging Face:

```bash
pixi run --platform linux-64-cuda fevbench [/path/to/fevbench_storage] [--tasks examples/fevbench/tasks.yaml]
```

If the storage path is omitted, the dataset is downloaded at runtime and cached under
`$HOME/.cache`.
