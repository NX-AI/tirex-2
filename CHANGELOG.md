# Changelog

All notable changes to TiRex-2 are documented here.

## [Unreleased]

### Improvements

- Replaced `einops` rearrangements with native PyTorch operations and removed the direct `einops` dependency in [#41](https://github.com/NX-AI/tirex-2/pull/41), contributed by [Copilot](https://github.com/apps/copilot-swe-agent) and [Zhihao Dai](https://github.com/daidahao).
- Removed a redundant `torch.no_grad()` context from prediction code in [#42](https://github.com/NX-AI/tirex-2/pull/42), contributed by [Zhihao Dai](https://github.com/daidahao).

### Documentation and development

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

[Unreleased]: https://github.com/NX-AI/tirex-2/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/NX-AI/tirex-2/releases/tag/v0.2.1
[0.1.1]: https://github.com/NX-AI/tirex-2/releases/tag/v0.1.1
