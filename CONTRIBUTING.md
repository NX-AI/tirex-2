# Contributing

Development and documentation contribution guidelines.

## Environment Setup

- **Pixi** (recommended): install [Pixi](https://pixi.prefix.dev/latest/), then run `pixi install`.
  Environments select a use case (`default`, `test`, `examples`, `docs`, or `pypi-build`), while named
  platforms select the operating system and accelerator (e.g. `linux-64-cuda`,
  `linux-64-cuda-126`, or `linux-64-cpu`). Both are defined in
  [`pyproject.toml`](pyproject.toml). Select a platform with `--platform`, for example
  `pixi run --platform linux-64-cuda test`.
- **pip**: create a virtual environment and install the package in editable mode:
  `python -m venv .venv && source .venv/bin/activate && pip install -e ".[examples,fev,gluonts]"`
- **Tooling**: run `pre-commit install` once, then `pre-commit run --all-files` and `pixi run --platform linux-64-cuda test`
  (or `pytest test/` in a pip environment) before opening a PR.

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
