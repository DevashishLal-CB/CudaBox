import math

import cudabox
import pytest
import torch
import torch.nn.functional as F
from utils import dtype_str_id


def flash_attention_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool,
    scale: float | None,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query, key, value, is_causal=is_causal, scale=scale
    )


@pytest.mark.parametrize(
    ("batch", "heads", "sequence", "head_dim"),
    [(1, 1, 1, 16), (1, 2, 31, 32), (2, 4, 128, 64)],
)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("scale", [None, 0.125])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32], ids=dtype_str_id
)
def test_flash_attention_matches_torch(
    batch, heads, sequence, head_dim, is_causal, scale, dtype
):
    query = torch.randn((batch, heads, sequence, head_dim), device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    expected = flash_attention_ref(query, key, value, is_causal, scale)
    actual = cudabox.algorithms.flash_attention(
        query, key, value, is_causal=is_causal, scale=scale
    )

    tolerance = {
        torch.float16: 3e-3,
        torch.bfloat16: 2e-2,
        torch.float32: 1e-5,
    }[dtype]
    assert actual.shape == query.shape
    assert actual.dtype == query.dtype
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def test_flash_attention_rejects_mismatched_shapes():
    query = torch.randn((1, 2, 16, 32), device="cuda")
    key = torch.randn((1, 2, 15, 32), device="cuda")
    value = torch.randn_like(query)

    with pytest.raises(RuntimeError, match="same shape"):
        cudabox.algorithms.flash_attention(query, key, value)


def test_flash_attention_rejects_unsupported_head_dimension():
    tensor = torch.randn((1, 2, 16, 48), device="cuda")

    with pytest.raises(RuntimeError, match="supports head dimensions"):
        cudabox.algorithms.flash_attention(tensor, tensor, tensor)


@pytest.mark.parametrize("scale", [0.0, -1.0, math.inf, math.nan])
def test_flash_attention_rejects_invalid_scale(scale):
    tensor = torch.randn((1, 1, 16, 32), device="cuda")

    with pytest.raises(RuntimeError, match="scale must be finite"):
        cudabox.algorithms.flash_attention(tensor, tensor, tensor, scale=scale)


if __name__ == "__main__":
    pytest.main([__file__])
