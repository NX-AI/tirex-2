# Agent instructions for this repo

This repo contains TiRex-2 inference code. Packaging, dependencies, tasks, and Pixi environments are configured in `pyproject.toml` (not a separate `pixi.toml`). Source code uses a `src/` layout under `src/tirex2`.

## Environments

Use Pixi unless the user explicitly asks for conda/mamba.

- Install/update environments: `pixi install`
- Environments select use cases: `default`, `test`, `examples`, and `pypi-build`.
- The default Linux and Windows CUDA 13.0 platforms are `linux-64-cuda` and `win-64-cuda`.
- CUDA 13.2 platforms end in `-cuda-132`; CUDA 12.6 platforms end in `-cuda-126`.
- CPU-only platforms are `linux-64-cpu` and `win-64-cpu`.
- The macOS platform is `osx-arm64`.
- Select a named platform with `--platform` (or `-p`) and an environment with `--environment` (or `-e`).

Common commands:

```bash
pixi run --platform linux-64-cuda test      # test environment on CUDA 13.0
pixi run --platform linux-64-cuda-126 test  # test environment on CUDA 12.6
pixi run -e examples -p linux-64-cpu python examples/sine_wave.py
pixi run -e examples -p linux-64-cpu python examples/covariate_forecasts.py
```

`model` is expected to be a local checkpoint directory or symlink containing `model-config.yaml` and `model.ckpt`. It is gitignored, as are `output/` and `*.csv` benchmark outputs.

## Forecasting from Python

Minimal direct API usage:

```python
import torch
from tirex2 import TimeseriesType, load_model

model = load_model("./model", device="cpu")  # use device="cuda" in a CUDA env if needed

target = torch.randn(1, 512)  # shape: [num_target_variates, context_length]
ts = TimeseriesType(target=target, past_covariates=None, future_covariates=None)

# returns list of forecasts; each forecast has shape [num_target_variates, num_quantiles, prediction_length]
forecast = model.forecast([ts], prediction_length=64, output_type="torch", batch_size=512)[0]
```

For future-known covariates, pass `future_covariates` with shape `[num_covariates, context_length + prediction_length]`. Past-only covariates use `past_covariates` with shape `[num_covariates, context_length]`.

Supported output types: `"torch"`, `"numpy"`, `"gluonts"`, and `"fev"` where the latter two require the optional dependencies/envs. Extra kwargs passed to `forecast(...)` are forwarded to `TiRex2.predict`, e.g. `tta_diff=False` or `tta_sign_flip=True`.

## Local examples

Sine-wave smoke test:

```bash
pixi run -e examples --platform linux-64-cpu python examples/sine_wave.py
# or explicitly:
pixi run -e examples --platform linux-64-cuda python examples/sine_wave.py
```

Future-known covariate demo:

```bash
pixi run -e examples --platform linux-64-cpu python examples/covariate_forecasts.py
# custom checkpoint/output/scenarios:
pixi run -e examples --platform linux-64-cuda python examples/covariate_forecasts.py \
  --ckpt ./model \
  --device cpu \
  --scenarios holidays nonstationary \
  --out output
```

The demo writes PNGs under `output/` by default.

## FEV-bench

Script: `examples/fevbench/run_fevbench.py`

Tasks are loaded from the paths configured in the YAML, typically the HuggingFace Hub (`autogluon/fev_datasets`). The model is always loaded from `NX-AI/TiRex-2-fevbench`.

To run offline against a local dataset snapshot, point HuggingFace's datasets cache at it instead of passing a path. The snapshot must use the standard cache layout (`<cache>/autogluon___fev_datasets/<config>/...`):

```bash
export HF_DATASETS_CACHE=/path/to/fev_store
export HF_HUB_OFFLINE=1
```

Run a quick/small benchmark first:

```bash
pixi run --platform linux-64-cuda fevbench \
  --tasks examples/fevbench/tasks-mini.yaml \
  --out output/fevbench-mini.csv \
  --device cuda:0 \
  --batch_size 128
```

Full configured task list:

```bash
pixi run --platform linux-64-cuda fevbench \
  --tasks examples/fevbench/tasks.yaml \
  --out output/fevbench.csv \
  --device cuda:0 \
  --batch_size 512
```

Useful options:

- `--max_tasks N` to run only the first N tasks from the YAML.
- `--as_univariate` to ignore covariates and forecast each target independently.
- `--model_name NAME` to set the model name in the output CSV.
- The script retries CUDA OOM by halving batch size; reduce `--batch_size` if needed.

Use CUDA 12.6 if required by the machine/cluster:

```bash
pixi run --platform linux-64-cuda-126 fevbench --tasks examples/fevbench/tasks-mini.yaml
```

## GiftEval

Script: `examples/gifteval/run_gifteval.py`

Download the GiftEval data once:

```bash
pixi run -e examples --platform linux-64-cuda huggingface-cli download Salesforce/GiftEval \
  --repo-type=dataset \
  --local-dir /path/to/gifteval_storage
```

Run the benchmark:

```bash
pixi run --platform linux-64-cuda gifteval /path/to/gifteval_storage pretrained \
  --out output/gifteval.csv \
  --device cuda
```

Use `zero-shot` instead of `pretrained` to load `NX-AI/TiRex-2-gifteval-zs`.

The script sets `GIFT_EVAL=/path/to/gifteval_storage` before importing the local GiftEval helpers. `examples/gifteval` is added to `PYTHONPATH` by the `example` Pixi feature so `gift_eval_utils` imports correctly.

Interactive notebook:

```bash
pixi run --platform linux-64-cuda notebook
# open examples/gifteval/gifteval.ipynb
```

## Tests and validation

Before handing off changes, run at least:

```bash
pixi run --platform linux-64-cuda test
```

For packaging sanity:

```bash
pixi run -e test --platform linux-64-cuda python -m pip wheel . --no-deps -w /tmp/tirex2-wheel-test
```

Do not commit generated files/directories such as `.pixi/`, `__pycache__/`, `output/`, `model`, `*.csv`, or `*.egg-info`.
