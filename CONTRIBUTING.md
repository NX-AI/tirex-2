# Contributing

Development and documentation contribution guidelines.

## Environment Setup

- **Pixi** (recommended): install [Pixi](https://pixi.prefix.dev/latest/), then run `pixi install`.
  Environments are defined in [`pyproject.toml`](pyproject.toml) under `tool.pixi.environments`
  (e.g. `cuda128`, `cuda126`, `test-cu128`, `test-cu126`, `example`, `example-cu128`, `example-cu126`).
- **pip**: create a virtual environment and install the package in editable mode:
  `python -m venv .venv && source .venv/bin/activate && pip install -e ".[examples,fev,gluonts]"`
- **Tooling**: run `pre-commit install` once, then `pre-commit run --all-files` and `pixi run test`
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

The documentation site lives under [`docs/`](docs/) and is built with
[MkDocs](https://www.mkdocs.org/) + [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
+ [mkdocstrings](https://mkdocstrings.github.io/).

- Install docs dependencies: `pip install -r docs/requirements.txt`
- Preview locally: `mkdocs serve`
- Build (as CI does): `mkdocs build --strict`
- The [API reference](docs/api/) is generated automatically from docstrings in
  `src/tirex2/` via mkdocstrings — update the docstring, not the generated page, and add a
  runnable usage example to any public function or class that doesn't already have one.
- Add new guides under `docs/` and register them in the `nav` section of [`mkdocs.yml`](mkdocs.yml).

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
