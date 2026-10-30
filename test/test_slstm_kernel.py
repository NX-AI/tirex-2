"""Compare the Triton sLSTM kernel with its PyTorch reference."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels require a GPU")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch,length,heads,head_dim", [(1, 1, 1, 16), (3, 17, 2, 32), (5, 65, 2, 64)])
def test_slstm_triton_matches_torch(dtype, batch, length, heads, head_dim):
    from tirex2.model.component.kernel.slstm_torch import slstm_torch
    from tirex2.model.component.kernel.slstm_triton import slstm_triton_fw

    torch.manual_seed(42)
    x = torch.randn(batch, length, heads, 4, head_dim, device="cuda", dtype=dtype)
    weight = (0.05 * torch.randn(heads, 4, head_dim, head_dim, device="cuda")).to(dtype)
    bias = 0.1 * torch.randn(heads, 4, head_dim, device="cuda")

    for nonzero_state in (False, True):
        states = torch.zeros(batch, heads, 4, head_dim, device="cuda")
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
