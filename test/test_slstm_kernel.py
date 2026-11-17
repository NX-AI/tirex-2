"""Compare the Triton sLSTM kernel with its PyTorch reference."""

import importlib.util
import os
from types import SimpleNamespace

import pytest
import torch

# The Triton kernel runs on a GPU, or on the CPU in Triton's interpreter (TRITON_INTERPRET=1, set in conftest.py).
if torch.cuda.is_available():
    TRITON_DEVICE = "cuda"
elif importlib.util.find_spec("triton") is not None and os.environ.get("TRITON_INTERPRET") == "1":
    TRITON_DEVICE = "cpu"
else:
    TRITON_DEVICE = None
pytestmark = pytest.mark.skipif(TRITON_DEVICE is None, reason="needs a GPU or Triton's CPU interpreter")


@pytest.fixture(params=["resident", "streamed"] if TRITON_DEVICE == "cpu" else [None])
def triton_interpreter(request, monkeypatch, interpreter_bfloat16):
    """Work around what the CPU interpreter lacks: libdevice, the GPU's shared memory size, autotuning timings and
    bfloat16 that computes as on a GPU.

    On the CPU, both shared memory sizes run: weights resident in shared memory, and re-read every step.
    One autotune config (two batch rows per program, so partial blocks are masked) skips the GPU benchmark.
    """
    if TRITON_DEVICE != "cpu":
        return
    import triton
    import triton.language as tl

    from tirex2.model.component.kernel import slstm_triton

    tanh = lambda x: 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0  # noqa: E731
    monkeypatch.setattr(slstm_triton, "libdevice", SimpleNamespace(tanh=tanh))
    # The interpreter cannot evaluate ``constexpr * tensor``; a plain float is the same constant.
    monkeypatch.setattr(slstm_triton, "LOG2E", slstm_triton.LOG2E.value)
    shared_mem = 1 << 30 if request.param == "resident" else 0
    monkeypatch.setattr(slstm_triton, "_max_shared_mem", lambda *_: shared_mem)
    config = triton.Config({"B_BLOCK_SIZE": 2}, num_warps=4, num_stages=1)
    monkeypatch.setattr(slstm_triton.slstm_kernel, "configs", [config])


@pytest.mark.usefixtures("triton_interpreter")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch,length,heads,head_dim", [(1, 1, 1, 16), (3, 17, 2, 32), (5, 65, 2, 64)])
def test_slstm_triton_matches_torch(dtype, batch, length, heads, head_dim):
    from tirex2.model.component.kernel.slstm_torch import slstm_torch
    from tirex2.model.component.kernel.slstm_triton import slstm_triton_fw

    device = TRITON_DEVICE
    torch.manual_seed(42)
    x = torch.randn(batch, length, heads, 4, head_dim, device=device, dtype=dtype)
    weight = (0.05 * torch.randn(heads, 4, head_dim, head_dim, device=device)).to(dtype)
    bias = 0.1 * torch.randn(heads, 4, head_dim, device=device)

    for nonzero_state in (False, True):
        states = torch.zeros(batch, heads, 4, head_dim, device=device)
        if nonzero_state:
            states.normal_(std=0.2)
            states[:, :, 2].abs_().add_(1)
        with torch.no_grad():
            # Triton rounds recurrent weights to BF16 internally.
            expected, expected_states = slstm_torch(x, states, weight.bfloat16(), bias)
            actual, actual_states = slstm_triton_fw(x, states, weight, bias)

        assert actual.dtype == dtype
        assert actual_states.dtype == torch.float32
        atol = 4e-3 if dtype == torch.bfloat16 else 1e-3
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=atol)
        torch.testing.assert_close(actual_states, expected_states, rtol=1e-3, atol=1e-3)


@pytest.mark.usefixtures("triton_interpreter")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_slstm_triton_varlen_matches_torch_per_sequence(dtype):
    """Packed sequences of different lengths (one empty) give the outputs and states of each sequence on its own."""
    from tirex2.model.component.kernel.slstm_torch import slstm_torch
    from tirex2.model.component.kernel.slstm_triton import slstm_triton_fw

    device, heads, head_dim, lengths = TRITON_DEVICE, 2, 32, [5, 0, 17, 9, 1]
    torch.manual_seed(0)
    x = torch.randn(1, sum(lengths), heads, 4, head_dim, device=device, dtype=dtype)
    weight = (0.05 * torch.randn(heads, 4, head_dim, head_dim, device=device)).to(dtype)
    bias = 0.1 * torch.randn(heads, 4, head_dim, device=device)
    states = torch.zeros(len(lengths), heads, 4, head_dim, device=device)
    states.normal_(std=0.2)
    states[:, :, 2].abs_().add_(1)
    cu_seqlens = torch.tensor([0, *lengths], device=device).cumsum(0)

    with torch.no_grad():
        actual, actual_states = slstm_triton_fw(x, states, weight, bias, cu_seqlens)
        for b, (start, end) in enumerate(zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())):
            if start == end:  # an empty sequence keeps its state
                torch.testing.assert_close(actual_states[b], states[b])
                continue
            expected, expected_states = slstm_torch(x[:, start:end], states[b : b + 1], weight.bfloat16(), bias)
            atol = 4e-3 if dtype == torch.bfloat16 else 1e-3
            torch.testing.assert_close(actual[:, start:end], expected, rtol=1e-3, atol=atol)
            torch.testing.assert_close(actual_states[b : b + 1], expected_states, rtol=1e-3, atol=1e-3)


def test_slstm_triton_falls_back_to_torch_for_gradients():
    """The Triton kernel has no backward: with inputs that require grad it returns the differentiable torch result."""
    from tirex2.model.component.kernel.slstm_torch import slstm_torch
    from tirex2.model.component.kernel.slstm_triton import slstm_triton_fw

    torch.manual_seed(0)
    x = torch.randn(2, 5, 1, 4, 16, requires_grad=True)
    states = torch.zeros(2, 1, 4, 16)
    weight, bias = 0.05 * torch.randn(1, 4, 16, 16), 0.1 * torch.randn(1, 4, 16)
    actual, actual_states = slstm_triton_fw(x, states, weight, bias)
    expected, expected_states = slstm_torch(x, states, weight, bias)
    assert actual.requires_grad
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_states, expected_states)
