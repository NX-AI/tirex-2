# Installation

## Via pip

```bash
pip install tirex-2
```

Install with additional dependencies:

```bash
pip install "tirex-2[examples,fev,gluonts]"
```

The Python package installation is currently only tested on Linux and macOS. Docker usage
(covered in [Deployment](../deployment.md)) additionally supports Windows via Docker Desktop.

## Via Pixi

[Pixi](https://pixi.prefix.dev/latest/) is used for the development and benchmarking
environment, to ensure it is set up correctly. Install it with:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Environments (e.g. `example-cu128`) are defined in
[`pyproject.toml`](https://github.com/NX-AI/tirex-2/blob/main/pyproject.toml) under
`tool.pixi.environments`; pick the one matching your CUDA version and use case.

## Next steps

Continue with the [Quickstart](quickstart.md) for a first forecast.
