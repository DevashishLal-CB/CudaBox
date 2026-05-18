import cudabox
import pytest
import torch


@pytest.mark.parametrize("N", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
def test_softmax(N):
    A = torch.rand((N), device="cuda")

    C_ref = torch.nn.Softmax(dim=0)(A)
    C = cudabox.elementwise.softmax(A)

    torch.testing.assert_close(C, C_ref, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
