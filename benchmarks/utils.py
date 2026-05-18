from typing import Callable, List, Sequence, Tuple

import torch
import triton.testing

DEFAULT_DTYPE = torch.float32
DEFAULT_DEVICE = "cuda"
DEFAULT_QUANTILES = [0.5, 0.2, 0.8]


def run_benchmark(
    fn: Callable,
    quantiles: Sequence[float] = (),
    scale: float = 1.0,
) -> Tuple[float, float, float]:
    """Execute benchmark using CUDA graph and return times in microseconds.

    Args:
        fn: Function to benchmark
        quantiles: Quantiles for timing measurements [median, min, max]
        scale: Scale the result down (usually num_layers).

    Returns:
        Tuple of (median_us, max_us, min_us)
    """
    quantiles = list(quantiles or DEFAULT_QUANTILES)
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)
    return 1000 * ms / scale, 1000 * max_ms / scale, 1000 * min_ms / scale
