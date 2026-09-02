# FAQ

??? question "How do I run TiRex-2 on CUDA?"

    With `device="cuda"`, TiRex-2 builds its fused sLSTM kernel (FlashRNN) with `nvcc` on the
    first forecast. In order to run TiRex-2 on CUDA you need:

    1. **A CUDA Toolkit installed and discoverable** — `nvcc` must be on `PATH` or reachable
       via `CUDA_HOME`.
    2. **A CUDA Toolkit whose major version matches your PyTorch build** — any 12.x toolkit
       for a `cu12x` torch wheel, any 13.x toolkit for a `cu13x` one. Check with
       `python -c "import torch; print(torch.version.cuda)"`.
    3. **A CUDA Toolkit no newer than your driver supports.**

??? question "Which NVIDIA GPU architectures does TiRex-2 support?"

    - TiRex-2 runs on NVIDIA GPUs with compute capability 8.0 (Ampere) or newer.
    - Older cards — Turing (7.5), Volta (7.0) and earlier — cannot run `device="cuda"`; use
      `device="cpu"` instead.

??? question "Why does TiRex-2 fail with `where cl` on Windows?"

    PyTorch may compile model components at runtime, so the Python process needs access to
    the MSVC C++ compiler, even when using `device="cpu"`.

    Install Visual Studio Build Tools with **Desktop development with C++**, then run
    TiRex-2 from an **x64 Native Tools Command Prompt for Visual Studio**. Confirm the
    compiler is available before starting your script:

    ```bat
    where cl
    python your_script.py
    ```

    Launch VS Code or Jupyter from the same prompt so it inherits the compiler environment.
    See [#15](https://github.com/NX-AI/tirex-2/issues/15) and
    [#17](https://github.com/NX-AI/tirex-2/issues/17) for related reports.
