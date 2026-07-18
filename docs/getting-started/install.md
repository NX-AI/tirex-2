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

## Access to model weights

TiRex-2's model weights are **gated on Hugging Face**. To access them, either log in via the
Hugging Face CLI, or generate an access token and set it before loading the model.

### Option 1: Hugging Face CLI

```bash
# log in to Hugging Face and follow the prompts
huggingface-cli login
```

See the [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli) for
details.

### Option 2: Access token

[Generate a fine-grained access token](https://huggingface.co/settings/tokens/new?canReadGatedRepos=true&tokenType=fineGrained)
with **Read access to contents of all public gated repos you can access** enabled, then set
it before loading the model:

```python
import os
os.environ["HF_TOKEN"] = "<insert-hf-token>"
```

### Option 3: Google Colab secret

On Google Colab, store the token as a Colab secret (via the key icon in the sidebar) named
`HF_TOKEN`. It is picked up automatically without pasting the token into a cell.

## Next steps

Continue with the [Quickstart](quickstart.md) for a first forecast.
