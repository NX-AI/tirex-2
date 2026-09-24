# FEV-benchmark

Evaluate TiRex2 on the [FEV-Benchmark](https://huggingface.co/spaces/autogluon/fev-bench)
benchmark.

## Data
Data can be downloaded beforehand using:
```bash
pixi run fevbench-download PATH_TO_STORAGE
```

#### Tips: Platform Selection

Pixi selects a compatible platform automatically (PyTorch w/ CUDA 13.0 for Linux/Windows, PyTorch w/ default for macOS), or you can choose one explicitly, for example `pixi run -p linux-64-cuda-126 ...`. Platforms are defined in
[`pyproject.toml`](pyproject.toml).

## Run
The script always loads `NX-AI/TiRex-2-fevbench` from Hugging Face:

```bash
pixi run fevbench [PATH_TO_STORAGE] [--tasks examples/fevbench/tasks.yaml]
```

Note: if `PATH_TO_STORAGE` is not given, then the dataset is downloaded at runtime and stored in `$HOME/.cache`.
