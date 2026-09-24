import os

import torch
import torch.nn.functional as F
import triton
import triton.testing
from cudabox.algorithms import flash_attention as cudabox_flash_attention
from utils import DEFAULT_DEVICE, run_benchmark

BATCH_SIZE = 1
NUM_HEADS = 8
SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048]
HEAD_DIMS = [64, 128]
DTYPES = [torch.float16, torch.bfloat16]
CAUSAL_VALUES = [False, True]

ALL_PROVIDERS = {
    "torch_sdpa": ("PyTorch SDPA", ("black", "-")),
    "cudabox_flash_attention": (
        "Cudabox Flash Attention",
        ("blue", "--"),
    ),
}


def _run(
    sequence: int,
    head_dim: int,
    dtype: torch.dtype,
    is_causal: bool,
    provider: str,
):
    shape = (BATCH_SIZE, NUM_HEADS, sequence, head_dim)
    query = torch.randn(shape, device=DEFAULT_DEVICE, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    if provider == "torch_sdpa":
        fn = lambda: F.scaled_dot_product_attention(
            query, key, value, is_causal=is_causal
        )
    elif provider == "cudabox_flash_attention":
        fn = lambda: cudabox_flash_attention(query, key, value, is_causal=is_causal)
    else:
        raise ValueError(f"unknown provider: {provider}")

    return run_benchmark(fn)


def _make_benchmark(
    head_dim: int, dtype: torch.dtype, is_causal: bool
) -> triton.testing.Benchmark:
    providers = list(ALL_PROVIDERS)
    causal_suffix = "causal" if is_causal else "noncausal"
    dtype_name = str(dtype).removeprefix("torch.")
    return triton.testing.Benchmark(
        x_names=["sequence"],
        x_vals=SEQUENCE_LENGTHS,
        line_arg="provider",
        line_vals=providers,
        line_names=[ALL_PROVIDERS[p][0] for p in providers],
        styles=[ALL_PROVIDERS[p][1] for p in providers],
        ylabel="us",
        plot_name=f"flash-attention-{dtype_name}-d{head_dim}-{causal_suffix}",
        args={
            "head_dim": head_dim,
            "dtype": dtype,
            "is_causal": is_causal,
        },
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    out_dir = os.path.join(os.path.dirname(__file__), "results", "flash_attention")
    os.makedirs(out_dir, exist_ok=True)

    for dtype in DTYPES:
        for head_dim in HEAD_DIMS:
            for is_causal in CAUSAL_VALUES:
                benchmark = triton.testing.perf_report(
                    [_make_benchmark(head_dim, dtype, is_causal)]
                )(_run)
                benchmark.run(print_data=True, save_path=out_dir)
                plt.close("all")
