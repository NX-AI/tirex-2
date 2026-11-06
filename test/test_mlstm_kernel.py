"""mLSTM pointer masks across chunk and head-dimension boundaries."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels require a GPU")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length,qk_dim,value_dim", [(7, 17, 33), (64, 32, 64), (65, 80, 96), (129, 128, 128)])
def test_mlstm_triton_matches_native(dtype, length, qk_dim, value_dim):
    from tirex2.model.component.kernel.mlstm_native import mlstm_chunkwise_native
    from tirex2.model.component.kernel.mlstm_triton import mlstm_chunkwise_triton

    torch.manual_seed(0)
    shape = (2, 2, length)
    q, k = (0.1 * torch.randn(*shape, qk_dim, device="cuda", dtype=dtype) for _ in range(2))
    v = torch.randn(*shape, value_dim, device="cuda", dtype=dtype)
    i, f = (torch.randn(*shape, device="cuda", dtype=dtype) for _ in range(2))
    inputs = (q, k, v, i, f)
    with torch.no_grad():
        expected = mlstm_chunkwise_native(*(x.double() for x in inputs))
        actual = mlstm_chunkwise_triton(*inputs)

    assert actual.shape == (2, 2, length, value_dim)
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    tolerance = 2e-3 if dtype == torch.float32 else 2e-2
    assert (actual.double() - expected).norm() / expected.norm() < tolerance
    torch.testing.assert_close(actual.double(), expected, rtol=tolerance, atol=tolerance * expected.abs().max().item())
