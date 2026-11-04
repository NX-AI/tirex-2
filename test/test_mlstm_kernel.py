"""mLSTM kernels: pointer masks across block and head-dimension boundaries, initial and last states."""

import pytest
import torch
import torch.nn.functional as F

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels require a GPU")


def _inputs(length, qk_dim, value_dim, dtype, device, seed=0):
    torch.manual_seed(seed)
    shape = (2, 2, length)
    q, k = (0.1 * torch.randn(*shape, qk_dim, device=device, dtype=dtype) for _ in range(2))
    v = torch.randn(*shape, value_dim, device=device, dtype=dtype)
    i, f = (torch.randn(*shape, device=device, dtype=dtype) for _ in range(2))
    return q, k, v, i, f


def _recurrent(q, k, v, i, f, initial_state=None, eps=1e-6):
    """Step-by-step stabilized mLSTM in float64, the reference for H and the last state (C, n, m)."""
    q, k, v, i, f = (x.double() for x in (q, k, v, i, f))
    B, NH, S, DHQK = q.shape
    if initial_state is None:
        C, n, m = q.new_zeros(B, NH, DHQK, v.shape[-1]), q.new_zeros(B, NH, DHQK), q.new_zeros(B, NH)
    else:
        C, n, m = (x.double() for x in initial_state)
    log_f = F.logsigmoid(f)
    hs = []
    for t in range(S):
        m_next = torch.maximum(log_f[..., t] + m, i[..., t])
        decay, gain = torch.exp(log_f[..., t] + m - m_next), torch.exp(i[..., t] - m_next)
        C = decay[..., None, None] * C + gain[..., None, None] * k[..., t, :, None] * v[..., t, None, :]
        n = decay[..., None] * n + gain[..., None] * k[..., t, :]
        m = m_next
        q_t = q[..., t, :] * DHQK**-0.5
        denom = torch.maximum((q_t * n).sum(-1).abs(), torch.exp(-m))
        hs.append((q_t[..., None, :] @ C).squeeze(-2) / (denom[..., None] + eps))
    return torch.stack(hs, dim=-2), (C, n, m)


def _unstabilized(state):
    """C and n without the stabilizer; states with different m that describe the same sequence agree."""
    C, n, m = (x.double() for x in state)
    return C * torch.exp(m)[..., None, None], n * torch.exp(m)[..., None]


def _rel(actual, expected):
    return ((actual.double() - expected).norm() / expected.norm()).item()


def _kernel(name):
    if name == "native":
        from tirex2.model.component.kernel.mlstm_native import mlstm_parallel_native as kernel
    else:
        from tirex2.model.component.kernel.mlstm_triton import mlstm_parallel_triton as kernel
    return kernel


def _initial_state(kind, qk_dim, value_dim, device):
    """None, the zero state that starts a sequence, or the state after a random prefix."""
    if kind == "none":
        return None
    if kind == "zero":
        return (torch.zeros(2, 2, qk_dim, value_dim, device=device), torch.zeros(2, 2, qk_dim, device=device),
                torch.zeros(2, 2, device=device))  # fmt: skip
    return _recurrent(*_inputs(50, qk_dim, value_dim, torch.float64, device, seed=1))[1]


KERNELS = ["native", pytest.param("triton", marks=requires_cuda)]


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("initial_state", ["none", "zero", "given"])
@pytest.mark.parametrize("length,qk_dim,value_dim", [(1, 16, 16), (7, 17, 33), (74, 64, 128), (129, 80, 96)])
def test_mlstm_kernel_matches_recurrent(kernel, initial_state, length, qk_dim, value_dim):
    """H matches the step-by-step recurrence; given an initial state, so does the last state, else it is None."""
    device, dtype = ("cuda", torch.float32) if kernel == "triton" else ("cpu", torch.float64)
    inputs = _inputs(length, qk_dim, value_dim, dtype, device)
    initial_state = _initial_state(initial_state, qk_dim, value_dim, device)
    with torch.no_grad():
        expected_h, expected_state = _recurrent(*inputs, initial_state=initial_state)
        h, state = _kernel(kernel)(*inputs, initial_state=initial_state)

    assert h.shape == (2, 2, length, value_dim) and h.dtype == dtype
    # The Triton kernel computes its fp32 dots in TF32.
    tolerance = 3e-3 if kernel == "triton" else 1e-6
    assert _rel(h, expected_h) < tolerance
    if initial_state is None:
        assert state is None
    else:
        assert [x.shape for x in state] == [x.shape for x in expected_state]
        for actual, expected in zip(_unstabilized(state), _unstabilized(expected_state)):
            assert _rel(actual, expected) < tolerance


@pytest.mark.parametrize("kernel", KERNELS)
def test_mlstm_kernel_state_continues_sequence(kernel):
    """Running two halves with the state in between gives the outputs and last state of the whole sequence."""
    device, dtype = ("cuda", torch.float32) if kernel == "triton" else ("cpu", torch.float64)
    q, k, v, i, f = _inputs(74, 64, 128, dtype, device)
    zero_state = _initial_state("zero", 64, 128, device)
    run = _kernel(kernel)
    with torch.no_grad():
        h, state = run(q, k, v, i, f, initial_state=zero_state)
        h_first, state_first = run(
            q[:, :, :30], k[:, :, :30], v[:, :, :30], i[..., :30], f[..., :30], initial_state=zero_state
        )
        h_second, state_second = run(
            q[:, :, 30:], k[:, :, 30:], v[:, :, 30:], i[..., 30:], f[..., 30:], initial_state=state_first
        )

    tolerance = 3e-3 if kernel == "triton" else 1e-6
    assert _rel(torch.cat([h_first, h_second], dim=2), h.double()) < tolerance
    for actual, expected in zip(_unstabilized(state_second), _unstabilized(state)):
        assert _rel(actual, expected) < tolerance


@pytest.mark.parametrize("with_state", [False, True])
def test_mlstm_native_gradients_match_recurrent(with_state):
    """The native kernel trains the model: its gradients match those of the recurrence."""
    from tirex2.model.component.kernel.mlstm_native import mlstm_parallel_native

    inputs = [x.requires_grad_() for x in _inputs(37, 16, 24, torch.float64, "cpu")]
    if with_state:
        initial_state = [x.requires_grad_() for x in _initial_state("given", 16, 24, "cpu")]
        h, state = mlstm_parallel_native(*inputs, initial_state=initial_state)
        expected_h, expected_state = _recurrent(*inputs, initial_state=initial_state)
        # A loss on H and on the unstabilized last state, which does not depend on the stabilizer.
        loss = h.pow(2).sum() + sum(x.sum() for x in _unstabilized(state))
        expected_loss = expected_h.pow(2).sum() + sum(x.sum() for x in _unstabilized(expected_state))
        leaves = inputs + initial_state
    else:
        loss = mlstm_parallel_native(*inputs)[0].pow(2).sum()
        expected_loss = _recurrent(*inputs)[0].pow(2).sum()
        leaves = inputs
    for actual, expected in zip(torch.autograd.grad(loss, leaves), torch.autograd.grad(expected_loss, leaves)):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length,qk_dim,value_dim", [(7, 17, 33), (64, 32, 64), (65, 80, 96), (129, 128, 128)])
def test_mlstm_triton_matches_native(dtype, length, qk_dim, value_dim):
    from tirex2.model.component.kernel.mlstm_native import mlstm_parallel_native
    from tirex2.model.component.kernel.mlstm_triton import mlstm_parallel_triton

    inputs = _inputs(length, qk_dim, value_dim, dtype, "cuda")
    initial_state = _initial_state("given", qk_dim, value_dim, "cuda")
    tolerance = 2e-3 if dtype == torch.float32 else 2e-2
    for state_in in (None, initial_state):
        with torch.no_grad():
            expected, expected_state = mlstm_parallel_native(*(x.double() for x in inputs), initial_state=state_in)
            actual, state = mlstm_parallel_triton(*inputs, initial_state=state_in)

        assert actual.shape == (2, 2, length, value_dim)
        assert actual.dtype == dtype
        assert torch.isfinite(actual).all()
        assert (actual.double() - expected).norm() / expected.norm() < tolerance
        torch.testing.assert_close(
            actual.double(), expected, rtol=tolerance, atol=tolerance * expected.abs().max().item()
        )
        if state_in is None:
            assert state is None and expected_state is None
        else:
            for actual_state, expected_state in zip(_unstabilized(state), _unstabilized(expected_state)):
                assert _rel(actual_state, expected_state) < tolerance
