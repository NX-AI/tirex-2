# Changelog

All notable changes to TiRex-2 are documented here.

## [Unreleased]

### Improvements

### Documentation and development

### Security and CI

- Expanded checkpoint-loader tests for local and Hugging Face paths, download options, and unavailable devices, contributed by [Zhihao Dai](https://github.com/daidahao).
- Added test coverage collection with `pytest-cov` and Codecov reports for core and inference tests, with PR comments and no Codecov status checks, in [#129](https://github.com/NX-AI/tirex-2/pull/129), contributed by [Maria Zuliani](https://github.com/uruthia) and [Zhihao Dai](https://github.com/daidahao).

## [0.5.0] - 2026-11-03

### Forecasting and performance

- ⭐️🚀 **Faster GPU inference for small batches** with CUDA Graphs and improved Triton kernels in [#80](https://github.com/NX-AI/tirex-2/pull/80), contributed by [Martin Loretz](https://github.com/martinloretzzz) and [Zhihao Dai](https://github.com/daidahao):
    - Graph replays accelerate small batches by up to 5×. Enabled by default for up to 32 variate rows on CUDA; disable with `load_model(..., cuda_graphs=False)`.
    - Triton sLSTM kernels replace FlashRNN, with FP32 states and PyTorch fallbacks. Removed the `flashrnn` dependency.
    - Triton mLSTM kernels replace `mlstm_kernels`. Removed the `mlstm_kernels` dependency.
    - CUDA Toolkit is no longer required. On Linux, a C compiler is needed for kernel compilation; on Windows, PyTorch fallbacks are used. See [FAQ](https://nx-ai.github.io/tirex-2/faq/#how-do-i-run-tirex-2-on-cuda) for details.
    - Fused sLSTM gate and mLSTM input projections; existing checkpoints adapt on load.
    - `pad_context=False` now pads to a multiple of 4 patches to reduce compilation and CUDA Graph capture overhead.
    - Added kernel and CUDA Graph tests. Benchmark scores remain unchanged.
- ⭐️🚀 **Faster GPU inference for medium batches**: Preprocessing now runs on the CPU and handles all target rows at once, with trend detection via a closed-form least-squares fit. GPU forecasts of 32–512 series are 1.5–1.9× faster with identical outputs, in [#111](https://github.com/NX-AI/tirex-2/pull/111), contributed by [Daniil Yefimov](https://github.com/DaniilYefimov).
- ⭐️🤖 **Forecasting skill**: Added a forecasting skill with Claude Code and Codex plugins, including dataframe and tensor examples in [#84](https://github.com/NX-AI/tirex-2/pull/84), contributed by [Zhihao Dai](https://github.com/daidahao).
- **Monotonic quantile forecasts**: Sort predictions across quantile levels to prevent quantile crossings in [#101](https://github.com/NX-AI/tirex-2/pull/101), contributed by [Martin Loretz](https://github.com/martinloretzzz).

### Deployment

- ⭐️🚀 **Inference server request queuing**: Concurrent requests are queued and batched across configurable model workers (`MODEL_WORKERS`), increasing throughput by up to 33× on CPU and 22× on GPU. Requests that exceed the queue wait limit (`MAX_QUEUE_WAIT_S`) receive HTTP 503, in [#114](https://github.com/NX-AI/tirex-2/pull/114), contributed by [Martin Loretz](https://github.com/martinloretzzz).
- ⭐️📦 **Smaller GPU Docker image**: Switching to `python:3.12-slim` without the CUDA Toolkit reduces image size by about 60% (26.3 GB → 10.3 GB). CPU/GPU images now share the same virtualenv setup and healthcheck, in [#119](https://github.com/NX-AI/tirex-2/pull/119), contributed by [Daniil Yefimov](https://github.com/DaniilYefimov).
- ⭐️🔄 **ONNX inference**: Added an [📙 ONNX inference guide](https://nx-ai.github.io/tirex-2/how-to/onnx/) for univariate and multivariate forecasts on CPU and GPU in [#75](https://github.com/NX-AI/tirex-2/pull/75), contributed by [Suad0](https://github.com/Suad0).

### Breaking changes

- ‼️‼️ **Legacy checkpoint support removed**: Checkpoints must be in `json` and `safetensors` formats, in [#112](https://github.com/NX-AI/tirex-2/pull/112), contributed by [Martin Loretz](https://github.com/martinloretzzz) and [Zhihao Dai](https://github.com/daidahao).
    - Legacy `yaml` and `ckpt` formats are no longer supported. Use `tirex-2<0.5.0` for legacy checkpoints, or convert them locally with [this script](https://github.com/NX-AI/tirex-2/blob/b16f761e160dab75bb8ea5b74ace008419c2d121/scripts/convert_checkpoint.py).
- ‼️ **GPU Docker user changed**: Both images run as `appuser` (UID 1000). Update GPU volume mounts from `/home/ubuntu/...` to `/home/appuser/...`, in [#119](https://github.com/NX-AI/tirex-2/pull/119), contributed by [Daniil Yefimov](https://github.com/DaniilYefimov).
- Removed the legacy conversion script in [#113](https://github.com/NX-AI/tirex-2/pull/113). Use the [archived version](https://github.com/NX-AI/tirex-2/blob/b16f761e160dab75bb8ea5b74ace008419c2d121/scripts/convert_checkpoint.py).
- Moved example and benchmark dependencies from pip extras to Pixi environments and updated setup docs and notebooks in [#115](https://github.com/NX-AI/tirex-2/pull/115), contributed by [Zhihao Dai](https://github.com/daidahao).

### Dependencies and development

- ⭐️📖 Added an In-Context Learning guide: how extending the context and adding covariates improves forecasting in [#69](https://github.com/NX-AI/tirex-2/pull/69), contributed by [Maria Zuliani](https://github.com/uruthia) and [Zhihao Dai](https://github.com/daidahao).
- Migrated the documentation site to Zensical, preserving the classic theme, API reference, and existing preview/build commands, and added `docs-build-clean` Pixi task in [#123](https://github.com/NX-AI/tirex-2/pull/123), contributed by [Zhihao Dai](https://github.com/daidahao).
- PyYAML is no longer a direct runtime dependency; it remains explicitly required in the Pixi test and example environments, in [#112](https://github.com/NX-AI/tirex-2/pull/112), contributed by [Martin Loretz](https://github.com/martinloretzzz) and [Zhihao Dai](https://github.com/daidahao).
- Simplified Pixi CUDA setup, removed CUDA Toolkit dependencies, added `pixi run build`, and validated minimum dependency versions in [#117](https://github.com/NX-AI/tirex-2/pull/117), contributed by [Zhihao Dai](https://github.com/daidahao).
- Enabled Triton floating-point fusion in Pixi and modernized build license metadata in [#117](https://github.com/NX-AI/tirex-2/pull/117), contributed by [Zhihao Dai](https://github.com/daidahao).
- Simplified model code and tests in [#80](https://github.com/NX-AI/tirex-2/pull/80), contributed by [Martin Loretz](https://github.com/martinloretzzz):
    - Replaced the FlashRNN layer with `sLSTMLayer` in `slstm_block.py`.
    - Removed `xLSTMMixedConfig` and `xLSTMLargeConfig`; `BiXLSTM` now takes plain arguments.
    - Renamed `conv_mLSTMLayerConfig` to `mLSTMLayerConfig`.
    - Removed unused feed-forward and norm options.
    - Switched CPU reference tests to relative norms to tolerate BF16 rounding differences across machines.

### Security and CI

- Added Python 3.15 preview testing and refreshed Pixi dependencies in [#116](https://github.com/NX-AI/tirex-2/pull/116), contributed by [Zhihao Dai](https://github.com/daidahao).
- Added plugin CI checks for manifest validation, Codex installation, package contents, and version bumps in [#100](https://github.com/NX-AI/tirex-2/pull/100), contributed by [Zhihao Dai](https://github.com/daidahao).
- Simplified Docker image builds and consolidated CI to test CPU/GPU images before publishing in [#82](https://github.com/NX-AI/tirex-2/pull/82), contributed by [Zhihao Dai](https://github.com/daidahao).
- Updated CA certificates and CI tooling for the eMudhra/emSign security incident in [#83](https://github.com/NX-AI/tirex-2/pull/83), contributed by [Zhihao Dai](https://github.com/daidahao).
- Changed Dependabot updates to monthly and disabled separate inference dependency updates in [#96](https://github.com/NX-AI/tirex-2/pull/96) and [#108](https://github.com/NX-AI/tirex-2/pull/108), contributed by [Zhihao Dai](https://github.com/daidahao) and [Daniil Yefimov](https://github.com/DaniilYefimov).

## [0.4.0] - 2026-10-20

### Improvements

- ⭐️🏋 **Default `safetensors` weights**: All checkpoints ship model configuration and weights in `json` and `safetensors` formats. `load_model` now loads from `config.json` and `model.safetensors`, which avoids unpickling and is faster. The legacy `yaml` and `ckpt` files are deprecated with a `FutureWarning` in [#55](https://github.com/NX-AI/tirex-2/pull/55), contributed by [Suad0](https://github.com/Suad0).
    - ‼️ Legacy yaml and ckpt formats will no longer be supported from v0.5.0 onwards.
- ⭐️📦 **Offline Docker images**: Docker images for CPU/GPU inference servers now include the default model weights for offline deployment in [#82](https://github.com/NX-AI/tirex-2/pull/82), contributed by [Zhihao Dai](https://github.com/daidahao).
- Removed the `xlstm` dependency to reduce the package footprint, in [#71](https://github.com/NX-AI/tirex-2/pull/71), contributed by [Martin Loretz](https://github.com/martinloretzzz).

### Documentation and development

- Refactored the sLSTM recurrence interface to support future kernel implementations and decoupled its configuration from FlashRNN in [#70](https://github.com/NX-AI/tirex-2/pull/70), contributed by [Martin Loretz](https://github.com/martinloretzzz).
- Added the Intended Use, Limitations and EU AI Act Notice (`INTENDED_USE.md`), linked from the README and the documentation, in [#67](https://github.com/NX-AI/tirex-2/pull/67), contributed by [Lukas Fischer](https://github.com/lukfischer) and [Yipeng Sun](https://github.com/sypsyp97).
- Added [a script](https://github.com/NX-AI/tirex-2/blob/b16f761e160dab75bb8ea5b74ace008419c2d121/scripts/convert_checkpoint.py) to convert legacy checkpoints to new formats in [#55](https://github.com/NX-AI/tirex-2/pull/55), contributed by [Suad0](https://github.com/Suad0).

## [0.3.1] - 2026-10-05

### Documentation and development

- Refactored internal prediction pipeline, and renamed `_predict_once(...)` to `_predict(...)` for clarity in [#62](https://github.com/NX-AI/tirex-2/pull/62), contributed by [Christian Ganhör](https://github.com/Tigxy).

## [0.3.0] - 2026-09-28

### Improvements

- ⭐️📊 Use [`forecast_df()`][tirex2.api_adapter.forecast.ForecastModel.forecast_df] to forecast directly from pandas, Polars, PyArrow, and other supported eager dataframes. It handles multiple series, covariates, and batching, and returns results in the input dataframe library by default, in [#18](https://github.com/NX-AI/tirex-2/pull/18), contributed by [Suad0](https://github.com/Suad0) and [Zhihao Dai](https://github.com/daidahao).
- ⭐️🚀 Added a `pad_context=False` option to `forecast(...)` to skip padding to the default context length, enabling much faster inference for short contexts at a small accuracy cost, in [#57](https://github.com/NX-AI/tirex-2/pull/57), contributed by [Daniil Yefimov](https://github.com/DaniilYefimov).
- ⭐️🚀 Added opt-in recurrent-layer compilation with `load_model(..., compile=True)` for faster repeated forecasts (up to 1.82x on CPU), in [#47](https://github.com/NX-AI/tirex-2/pull/47), contributed by [Daniil Yefimov](https://github.com/DaniilYefimov).
- Added a `preserve_grad=True` option to `_predict_once(...)` and postprocessing for fine-tuning, preserving finite gradients through inverse differencing, in [#58](https://github.com/NX-AI/tirex-2/pull/58), contributed by [Zhihao Dai](https://github.com/daidahao).
- Replaced `einops` rearrangements with native PyTorch operations and removed the direct `einops` dependency in [#41](https://github.com/NX-AI/tirex-2/pull/41), contributed by [Copilot](https://github.com/apps/copilot-swe-agent) and [Zhihao Dai](https://github.com/daidahao).
- Removed a redundant `torch.no_grad()` context from prediction code in [#42](https://github.com/NX-AI/tirex-2/pull/42), contributed by [Zhihao Dai](https://github.com/daidahao).

### Documentation and development

- Added a [dataframe forecasting guide](./how-to/dataframes.md) for native dataframe forecasting via `forecast_df`, contributed by [Suad0](https://github.com/Suad0) and [Zhihao Dai](https://github.com/daidahao).
- Corrected deployment examples and guidance for covariates, model caching, compilation, and the HTTP API, and added an Open Source vs PRO feature comparison table with consistent PRO naming in [#21](https://github.com/NX-AI/tirex-2/pull/21), contributed by [Lukas Fischer](https://github.com/lukfischer), Claude Sonnet 5, and [Zhihao Dai](https://github.com/daidahao).
- Made the default Windows CUDA Pixi platform use the cu130 PyTorch index and fail CUDA tests early when CUDA is unavailable, in [#51](https://github.com/NX-AI/tirex-2/pull/51), contributed by [Yipeng Sun](https://github.com/sypsyp97).
- Added FAQs covering CUDA setup, supported NVIDIA GPU architectures, and Windows MSVC compiler setup in [#19](https://github.com/NX-AI/tirex-2/pull/19), contributed by [Yipeng Sun](https://github.com/sypsyp97) and [Daniil Yefimov](https://github.com/DaniilYefimov).
- Added a Pixi documentation environment and preview/build commands, migrated documentation deployment to GitHub Pages artifact actions, and updated contributor instructions in [#37](https://github.com/NX-AI/tirex-2/pull/37), contributed by [Zhihao Dai](https://github.com/daidahao).
- Removed the unused root `requirements.txt` in [#40](https://github.com/NX-AI/tirex-2/pull/40), contributed by [Copilot](https://github.com/apps/copilot-swe-agent) and [Zhihao Dai](https://github.com/daidahao).
- Added named Pixi platforms for Linux and Windows CPU builds and CUDA 12.6, 13.0, and 13.2 builds, with matching CUDA toolchains and updated development and benchmark commands in [#20](https://github.com/NX-AI/tirex-2/pull/20), contributed by [Zhihao Dai](https://github.com/daidahao).
- Added GitHub issue and pull request templates in [#25](https://github.com/NX-AI/tirex-2/pull/25), contributed by [Zhihao Dai](https://github.com/daidahao).

### Security and CI

- Migrated the cross-platform CPU test matrix to Pixi for Python 3.11 and 3.14, added lower-bound tests with Python 3.11 and PyTorch 2.8.0 and Python 3.15 preview coverage, and restricted test workflow permissions in [#43](https://github.com/NX-AI/tirex-2/pull/43), contributed by [Zhihao Dai](https://github.com/daidahao).
- Added a CPU test job using the frozen Pixi lockfile in [#44](https://github.com/NX-AI/tirex-2/pull/44), contributed by [Zhihao Dai](https://github.com/daidahao).
- Updated `requests` from 2.34.0 to 2.34.2 in the inference dependencies in [#35](https://github.com/NX-AI/tirex-2/pull/35), contributed by [Dependabot](https://github.com/apps/dependabot) and [Zhihao Dai](https://github.com/daidahao).
- Hardened Pixi dependency resolution with a seven-day package release cooldown (except for `xlstm`, `mlstm_kernels`, and `flashrnn`), disabled dependency wheel builds, and required Pixi 0.67.0 or newer in [#24](https://github.com/NX-AI/tirex-2/pull/24), contributed by [Zhihao Dai](https://github.com/daidahao) and [Copilot Autofix](https://github.com/apps/copilot-pull-request-reviewer).
- Enabled weekly Dependabot version updates for the root Python dependencies in [#28](https://github.com/NX-AI/tirex-2/pull/28), contributed by [Zhihao Dai](https://github.com/daidahao).
- Added a security policy documenting supported versions and private vulnerability reporting in [#23](https://github.com/NX-AI/tirex-2/pull/23), contributed by [Copilot](https://github.com/apps/copilot-swe-agent) and [Zhihao Dai](https://github.com/daidahao).
- Pinned GitHub Actions to commit SHAs and updated action versions in [#27](https://github.com/NX-AI/tirex-2/pull/27), contributed by [Copilot](https://github.com/apps/copilot-swe-agent) and [Zhihao Dai](https://github.com/daidahao).
- Added Visual Studio developer-shell initialization for Windows builds in [#27](https://github.com/NX-AI/tirex-2/pull/27), contributed by [Copilot](https://github.com/apps/copilot-swe-agent) and [Zhihao Dai](https://github.com/daidahao).

## [0.2.1] - 2026-08-05

### Improvements

- Fixed TiRex-2 functionality on macOS for CPU and MPS backends in [#10](https://github.com/NX-AI/tirex-2/pull/10), contributed by [Suad0](https://github.com/Suad0).
- Fixed TiRex-2 functionality on Windows for CPU and CUDA backends in [#5](https://github.com/NX-AI/tirex-2/pull/5), contributed by [Daniil Yefimov](https://github.com/DaniilYefimov).
- Relaxed package requirements to make installation via pip more flexible in [#12](https://github.com/NX-AI/tirex-2/pull/12) and [#13](https://github.com/NX-AI/tirex-2/pull/13), contributed by [Daniil Yefimov](https://github.com/DaniilYefimov).
- Added an option to use Flex Attention in [#6](https://github.com/NX-AI/tirex-2/pull/6), contributed by [danieleb1861](https://github.com/danieleb1861).

## [0.1.1] - 2026-07-02

Initial TiRex-2 release, contributed by [martinloretzzz](https://github.com/martinloretzzz) in the [initial commit](https://github.com/NX-AI/tirex-2/commit/abdf2898162482cc5c862905a406fc1134fbae67).

[Unreleased]: https://github.com/NX-AI/tirex-2/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/NX-AI/tirex-2/releases/tag/v0.5.0
[0.4.0]: https://github.com/NX-AI/tirex-2/releases/tag/v0.4.0
[0.3.1]: https://github.com/NX-AI/tirex-2/releases/tag/v0.3.1
[0.3.0]: https://github.com/NX-AI/tirex-2/releases/tag/v0.3.0
[0.2.1]: https://github.com/NX-AI/tirex-2/releases/tag/v0.2.1
[0.1.1]: https://github.com/NX-AI/tirex-2/releases/tag/v0.1.1
