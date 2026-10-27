"""CUDA graph replay and the Triton sLSTM recurrence it relies on."""

import pytest
import torch

from tirex2.model.component.slstm_block import sLSTMLayer
from tirex2.model.types import TimeseriesType

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs require a GPU")


@pytest.fixture
def model(build_small_model):
    torch.manual_seed(0)
    model = build_small_model("cuda", recipe=["small_mlstm", "small_slstm"]).eval()
    with torch.no_grad():  # small-init weights barely move the forecast; scale them so a mismatch shows
        for param in model.parameters():
            param.mul_(4)
    return model


def _univariate(n: int, length: int = 64):
    torch.manual_seed(1)
    return [TimeseriesType(torch.randn(1, length).cumsum(-1), None, None) for _ in range(n)]


def _covariates():
    torch.manual_seed(2)
    return [TimeseriesType(torch.randn(2, 64).cumsum(-1), torch.randn(1, 64), torch.randn(2, 72))]


@pytest.mark.parametrize("make_timeseries", [lambda: _univariate(3), _covariates])
def test_cuda_graph_replay_matches_eager(model, make_timeseries):
    ts = make_timeseries()
    eager = model.predict(ts, prediction_length=8)
    model.use_cuda_graphs = True
    captured, replayed = model.predict(ts, prediction_length=8), model.predict(ts, prediction_length=8)

    assert len(model._cuda_graphs) == 1
    for expected, first, second in zip(eager, captured, replayed):
        # Padding the variates to a multiple of 4 can change the GEMM algorithm, so not bitwise;
        # the scaled weights amplify that rounding element-wise, so compare relative norms.
        assert (first - expected).norm() / expected.norm() < 1e-3
        torch.testing.assert_close(second, first, rtol=0, atol=0)


def test_cuda_graphs_are_shared_across_padded_shapes(model):
    model.use_cuda_graphs = True
    for ts in (_univariate(1), _univariate(2), _univariate(3), _univariate(4)):
        model.predict(ts, prediction_length=8)
    assert len(model._cuda_graphs) == 2  # 1 row, and 2-4 rows padded to 4

    model.predict(_univariate(33), prediction_length=8)  # more than 32 rows run eagerly
    assert len(model._cuda_graphs) == 2

    # Patch size 4 with 2 future patches: contexts of 1 and 2 patches pad to 4 patches.
    for length in (4, 8):
        model.predict(_univariate(1, length), prediction_length=8, pad_context=False)
    assert len(model._cuda_graphs) == 3


def test_cuda_graphs_are_not_used_with_gradients_training_or_after_moving(model):
    model.use_cuda_graphs = True
    _, batch, _, _ = model._prepare_input(_univariate(1), 8, tta_diff=True)
    model(batch).sum().backward()
    model.train()
    with torch.no_grad():
        model(batch)
    assert model._cuda_graphs is None

    model.eval()
    model.predict(_univariate(1), prediction_length=8)
    model.to("cuda")
    assert model._cuda_graphs is None


@pytest.mark.parametrize("D", [32, 128])
@pytest.mark.parametrize("resident_weights", [True, False])
def test_triton_slstm_matches_the_pytorch_implementation(D, resident_weights, monkeypatch):
    from tirex2.model.component.kernel import slstm_triton

    if resident_weights and 4 * D * D * 2 + slstm_triton.SMEM_HEADROOM > slstm_triton._max_shared_mem(0):
        pytest.skip("The resident weights don't fit this GPU's shared memory")
    # Force either path: resident weights, or re-reading them every step as on small shared memory GPUs
    monkeypatch.setattr(slstm_triton, "_max_shared_mem", lambda _: 2**30 if resident_weights else 0)

    torch.manual_seed(0)
    B, T, G, H = 3, 40, 4, 4
    Wx, R, b = torch.randn(B, T, G, H, D), (0.1 * 32 / D) * torch.randn(H, D, G, D), torch.randn(H, G, D)
    states = torch.randn(4, B, 1, H, D)
    states[2] = states[2].abs() + 1  # normalizer n >= 1
    Wx, R, b, states = (t.cuda() for t in (Wx, R, b, states))

    for s in (None, states):
        expected_y, expected_last = sLSTMLayer.slstm_kernels(Wx, R, b, s, backend="torch")
        y, last = sLSTMLayer.slstm_kernels(Wx, R, b, s, backend="triton")
        torch.testing.assert_close(y, expected_y, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(last, expected_last, rtol=1e-3, atol=1e-3)


def test_cuda_falls_back_to_the_pytorch_kernels_without_triton(build_small_model, monkeypatch):
    from tirex2.model.component import kernel
    from tirex2.model.component.mlstm_block import mLSTMLayer

    timeseries = _covariates()
    torch.manual_seed(0)
    expected = build_small_model("cuda", recipe=["small_mlstm", "small_slstm"]).eval().predict(timeseries, 8)

    monkeypatch.setattr(kernel, "triton_available", lambda: False)  # e.g. PyTorch's Windows wheels
    torch.manual_seed(0)
    model = build_small_model("cuda", recipe=["small_mlstm", "small_slstm"]).eval()
    assert {m.config.backend for m in model.modules() if isinstance(m, sLSTMLayer)} == {"torch"}
    assert {m.config.mlstm_kernel for m in model.modules() if isinstance(m, mLSTMLayer)} == {"native"}

    model.use_cuda_graphs = True
    for actual, want in zip(model.predict(timeseries, 8), expected):
        assert (actual - want).norm() / want.norm() < 1e-2


def test_gradients_reach_every_parameter_on_cuda(model):
    _, batch, _, _ = model._prepare_input(_covariates(), 8, tta_diff=True)
    model(batch).sum().backward()
    missing = [name for name, p in model.named_parameters() if p.requires_grad and (p.grad is None or not p.grad.any())]
    assert not missing
