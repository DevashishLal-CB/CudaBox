import torch


def histogram(tensor: torch.tensor, num_bins: int) -> torch.tensor:
    return torch.ops.cudabox.histogram(tensor, num_bins)


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product self-attention on ``(B, H, S, D)`` tensors.

    The native forward kernel tiles Q, K, and V through shared memory and uses
    online softmax without materializing the full attention matrix. Supported
    head dimensions are 16, 32, 64, and 128.
    """
    return torch.ops.cudabox.flash_attention(query, key, value, is_causal, scale)
