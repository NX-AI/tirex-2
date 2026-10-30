# FAQ

??? question "How do I run TiRex-2 on CUDA?"

    On both platforms, you need a supported NVIDIA GPU, a compatible NVIDIA driver,
    and a [CUDA build of PyTorch](https://pytorch.org/get-started/locally/).
    Then load the model with `device="cuda"`. No separate CUDA Toolkit is needed.

    - **Linux:** Install a C compiler such as `gcc`. PyTorch's CUDA wheels include
      Triton, which compiles the fast recurrence kernels on first use.
    - **Windows:** Standard PyTorch wheels do not include Triton, so TiRex-2 uses
      slower PyTorch recurrence kernels. No C compiler is needed for this fallback.
      For faster inference, try [triton-windows](https://github.com/triton-lang/triton-windows#installation),
      following its installation guide to match your PyTorch version.

??? question "Which NVIDIA GPU architectures does TiRex-2 support?"

    - TiRex-2 runs on NVIDIA GPUs with compute capability 8.0 (Ampere) or newer.
    - Older cards — Turing (7.5), Volta (7.0) and earlier — cannot run `device="cuda"`; use
      `device="cpu"` instead.

??? question "When should I use `compile=True`?"

    We recommend to use `compile=True` on CPU, when the context length and prediction length stay the same across forecasts —
    `torch.compile` re-traces on every new input shape, so the speed-up only pays off when one
    shape is reused.

    See [Forecasting](how-to/forecasting.md#loading-the-model) and
    [#47](https://github.com/NX-AI/tirex-2/pull/47) for the CPU benchmarks.

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
