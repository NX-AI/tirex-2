# FAQ

??? question "How do I run TiRex-2 on CUDA?"

    Install a CUDA build of PyTorch and pass `device="cuda"`; no CUDA Toolkit is needed. The
    sLSTM and mLSTM recurrences run on Triton kernels, which PyTorch's Linux CUDA wheels ship
    and which are compiled on first use (this needs a C compiler such as `gcc`). Without
    Triton, e.g. with PyTorch's Windows wheels, TiRex-2 falls back to its PyTorch
    implementations of both recurrences, which are slower but give the same forecasts up to
    rounding.

??? question "Which NVIDIA GPU architectures does TiRex-2 support?"

    - TiRex-2 runs on NVIDIA GPUs with compute capability 8.0 (Ampere) or newer.
    - Older cards — Turing (7.5), Volta (7.0) and earlier — cannot run `device="cuda"`; use
      `device="cpu"` instead.

??? question "When should I use `compile=True`?"

    Use it for long-running processes that forecast the same input shapes repeatedly: each
    new shape is compiled on its first forecast, which takes seconds on CUDA and minutes on
    CPU, after which forecasts are up to 1.9x faster on CPU. On CUDA, CUDA graphs already
    remove most of the overhead it would save. See [Forecasting](how-to/forecasting.md#loading-the-model).

??? question "How can I speed up forecasts on short contexts?"

    Pass `pad_context=False` to `forecast(...)` to skip padding to the model's full context
    length (2,048), reducing inference time for short contexts at the cost of a small loss
    in forecast accuracy. Padding to the longest context in each batch and to a multiple of
    4 patches (128 time steps) still applies. The default is `pad_context=True`.

    See [Forecasting](how-to/forecasting.md#context-padding) for details and
    [#57](https://github.com/NX-AI/tirex-2/pull/57) for the benchmarks.

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
