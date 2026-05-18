import torch
from cudabox.elementwise import softmax

"""
-- set full/basic/roofline

# Full profile
ncu --set full \
    --nvtx --nvtx-include "softmax_profile/" \
    --import-source yes \
    --target-processes all \
    -o profiler/softmax_profile -f \
    python -m profiler.profile_softmax

# Quick summary terminal only
ncu --set full --nvtx --nvtx-include "softmax_profile/" \
    --target-processes all \
    python -m profiler.profile_softmax

# Without nvtx
ncu --set full \
    --kernel-name softmax_kernel \
    --launch-skip 3 --launch-count 1 \
    --target-processes all \
    --import-source yes \
    -o profiler/softmax_profile -f \
    python -m benchmarks.profile_softmax
"""


def main():
    N = 1 << 20  # pick a representative size; tweak per workload
    torch.manual_seed(0)
    x = torch.randn(N, dtype=torch.float32, device="cuda")

    # Warm up: JIT/cuBLAS/loaders, autotune, etc. NCU profiles the
    # kernels AFTER this point (we'll use --target-processes + skip).
    for _ in range(3):
        _ = softmax(x)
    torch.cuda.synchronize()

    # The launch we actually care about
    torch.cuda.nvtx.range_push("softmax_profile")
    y = softmax(x)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    print(y.sum().item())  # ensure result isn't DCE'd


if __name__ == "__main__":
    main()
