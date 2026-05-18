import itertools
import os

import torch
import triton
import triton.testing
from cudabox.elementwise import softmax as cudabox_softmax

from .utils import DEFAULT_DEVICE, DEFAULT_DTYPE, run_benchmark


def torch_softmax(x):
    return torch.softmax(x, dim=0)


SIZE = sorted(set([1536, 3072, 4096, 5120] + [1 << i for i in range(13, 21)]))

LINE_VALS = [
    "cudabox_softmax",
    "torch_softmax",
]
LINE_NAMES = [
    "Cudabox Softmax",
    "Torch Softmax",
]
STYLES = [
    ("blue", "--"),
    ("purple", "-."),
]

configs = list(itertools.product(SIZE))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="softmax-performance",
        args={},
    )
)
def benchmark(size: int, provider: str):
    input = torch.randn(size, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    FN_MAP = {
        "cudabox_softmax": cudabox_softmax,
        "torch_softmax": torch_softmax,
    }
    fn = lambda: FN_MAP[provider](input)
    return run_benchmark(fn)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    benchmark.run(print_data=True, save_path=out_dir)
