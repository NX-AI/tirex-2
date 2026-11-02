# Contributing

Development and documentation contribution guidelines.

## Environment Setup

- **Pixi** (recommended): install [Pixi](https://pixi.prefix.dev/latest/), then run `pixi install`.
  Configuration lives in [`pyproject.toml`](pyproject.toml).

  - **Environments** select a use case with `-e`: `default`, `test`, `examples`, `benchmark`,
    `docs`, or `pypi-build`. Both `test` and `benchmark` include FEV and GluonTS;
    `examples` provides plotting and Jupyter, also included in `benchmark`.
  - **Platforms** select the operating system and accelerator with `-p`, such as
    `linux-64-cuda`, `linux-64-cuda-126`, `linux-64-cpu`, or `osx-arm64-macos14`.
    For example: `pixi run -p linux-64-cuda test`.
  - **Tasks** run common workflows with `pixi run <task>`, such as `test`, `notebook`,
    `gifteval`, `fevbench`, or `docs-build`. Each task selects its default environment.

- **pip**: install dependencies manually; package extras are no longer provided.

  - **Environment**: create and activate a virtual environment with
    `python -m venv .venv && source .venv/bin/activate` (Linux/macOS).
  - **PyTorch**: install the appropriate [CPU or GPU build](https://pytorch.org/get-started/locally/).
  - **Package**: run `pip install -e .` from the repository root for editable development.
  - **Tests**: run `pip install pytest pandas polars pyarrow fev datasets gluonts`, then `pytest test/`.

- **Tooling**:

  - **Setup**: run `pre-commit install` once to enable Git hooks.
  - **Checks**: before opening a PR, run `pre-commit run --all-files` and
    `pixi run test` (or `pytest test/` in a pip environment).
  - **CUDA**: CUDA test platforms require an available CUDA runtime. Use
    `pixi run -p linux-64-cpu test` for CPU-only checks.

## Workflow Overview

1. Branch from `main` and keep changes focused (docs versus code versus tooling).
2. Run pre-commit and tests locally before pushing.
3. Open a PR with a clear summary, test notes, and any follow-up TODOs.
4. Address CI feedback — red checks block review.

Commit messages are linted by [`conventional-pre-commit`](.pre-commit-config.yaml) and must
follow `type(scope): summary` with one of `chore`, `ci`, `docs`, `feat`, `fix`, `test`
(a scope is required).

## Documentation Specifics

The MkDocs site lives in [`docs/`](docs/) and publishes to [GitHub Pages](https://nx-ai.github.io/tirex-2/) via [`.github/workflows/docs.yml`](.github/workflows/docs.yml). To contribute:

- Add guides under `docs/` and register them in [`mkdocs.yml`](mkdocs.yml) under `nav`.
- Update [API reference](docs/api/) content through docstrings in `src/tirex2/`, including runnable examples for public functions and classes.
- Preview locally: `pixi run docs`.
- Build with warnings treated as errors (as CI does): `pixi run --frozen docs-build`.

## Agent plugin checks

Bump both manifest versions when changing `plugins/tirex2/`. CI checks Claude validation
and Codex installation without model credentials or weights.

To check locally:

```bash
pixi run python scripts/check_plugin.py --base main
pixi run python -m unittest discover -s scripts/tests
```

## Commit & Review Etiquette

- Avoid committing generated artifacts (e.g. `.pixi/`, `output/`, `model`, `*.csv`,
  `*.egg-info`, `site/` — see [`.gitignore`](.gitignore)) unless they are intended changes.
- Rebase (don't merge) when syncing from `main`.
- Respond to every review comment; clarify disagreements rather than ignoring them.

## Getting Help

- Open a draft PR early for directional feedback.
- Use GitHub Issues/Discussions for larger proposals.

## NXAI Contributor License Agreement

Read the full CLA for Individual Contributors here: [CLA](https://github.com/NX-AI/CLA/blob/main/CLA.md)

### Contact

If you have any question about the CLA, feel free to reach out to [contact@nx-ai.com](mailto:contact@nx-ai.com)
