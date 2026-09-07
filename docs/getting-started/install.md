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

Environments select a use case (`default`, `test`, `examples`, or `pypi-build`), while named
platforms select the operating system and accelerator. Both are defined in
[`pyproject.toml`](https://github.com/NX-AI/tirex-2/blob/main/pyproject.toml). Pixi selects a
compatible platform automatically, or you can choose one explicitly, for example
`--platform linux-64-cuda`, `--platform linux-64-cuda-126`, or
`--platform linux-64-cpu`.

## Next steps

Continue with the [Quickstart](quickstart.md) for a first forecast. If you plan to run on a
GPU, see the [FAQ](../faq.md) for the CUDA Toolkit and GPU architecture requirements.
