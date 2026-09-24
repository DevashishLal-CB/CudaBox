# CudaBox

A personal sandbox for learning to write **high-performance CUDA kernels**.

Each kernel is implemented in CUDA C++, exposed to Python as a [PyTorch custom
operator](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html)
(`torch.ops.cudabox.*`), and then **tested for correctness** against a PyTorch
reference, **benchmarked** against PyTorch/Triton, and **profiled** with Nsight
Compute. The goal is the full optimization loop — write a kernel, prove it
correct, measure it, and iterate from a naive version toward an
architecture-aware one (e.g. Hopper / SM90 TMA + warp-specialized pipelines).

## Kernels

Kernels are grouped into three families. Most have a "naive → optimized"
progression so you can compare implementations.

| Family | Op | Python API | Notes |
| --- | --- | --- | --- |
| **elementwise** | softmax | `cudabox.elementwise.softmax` | Baseline reduction-based softmax |
| | online softmax | `cudabox.elementwise.online_softmax` | Single-pass (running max/sum) softmax |
| | online softmax (SM90) | `cudabox.elementwise.sm90_online_softmax` | Hopper-specific online softmax |
| | rmsnorm | `cudabox.elementwise.rmsnorm` | RMS normalization with `gamma`, `eps` |
| **gemm** | simple gemm | `cudabox.gemm.simple_gemm` | Naive matmul |
| | tiled gemm | `cudabox.gemm.tiled_gemm` | Shared-memory tiled matmul |
| | pipelined TMA/MMA gemm (SM90) | `cudabox.gemm.sm90_pipelined_tma_mma_gemm` | CUTLASS-based pipelined TMA + WGMMA |
| **algorithms** | histogram | `cudabox.algorithms.histogram` | Binned histogram |
| | flash attention | `cudabox.algorithms.flash_attention` | Tiled CUDA forward with online softmax |

## Project Structure

```
CudaBox/
├── CMakeLists.txt           # Top-level build: detects CUDA arch, finds Torch/CUDA, adds subdirs
├── CMakePresets.json        # linux/windows configure + build presets (Ninja)
├── pyproject.toml           # scikit-build-core packaging (builds the cudabox wheel)
│
├── include/                 # Public headers shared across kernels
│   ├── cudabox_ops.hpp      # Declarations of every kernel entry point (the C++ API surface)
│   ├── cuda_utils.cuh       # CUDA helper utilities
│   ├── torch_utils.hpp      # Torch tensor helpers
│   ├── python_utils.hpp     # REGISTER_EXTENSION macro / Python module glue
│   ├── logger.hpp           # spdlog-based logging
│   └── gemm/                # Headers for the SM90 pipelined GEMM (kernel.cuh, traits.cuh)
│
├── csrc/                    # CUDA/C++ kernel sources, one subdir per family
│   ├── algorithms/
│   │   ├── CMakeLists.txt
│   │   └── src/
│   │       ├── algorithms_ops.cpp   # TORCH_LIBRARY op registration for this family
│   │       ├── flash_attention/flash_attention.cu
│   │       └── histogram/histogram.cu
│   ├── elementwise/
│   │   ├── CMakeLists.txt
│   │   └── src/
│   │       ├── elementwise_ops.cpp  # Registers softmax / online_softmax / rmsnorm ops
│   │       ├── softmax/
│   │       ├── online_softmax/      # online_softmax.cu + sm90_online_softmax.cu
│   │       └── rmsnorm/
│   └── gemm/
│       ├── CMakeLists.txt
│       ├── src/                     # simple, tiled, and sm90_pipelined_tma_mma kernels
│       └── benchmarks/              # nvbench C++ microbenchmarks (opt-in)
│
├── python/cudabox/          # Python package: thin wrappers over torch.ops.cudabox
│   ├── __init__.py          # Imports compiled *_ops extensions + py wrappers
│   ├── algorithms.py
│   ├── elementwise.py
│   └── gemm.py
│
├── tests/                   # pytest correctness tests vs. torch references
│   ├── conftest.py
│   └── test_*.py            # test_softmax / test_online_softmax / test_rmsnorm / test_gemm / test_histogram
│
├── benchmarks/              # Python perf benchmarks (triton.testing harness)
│   ├── utils.py             # Shared benchmark runner / defaults
│   └── bench_*.py           # Per-kernel benchmarks; results written to results/
│
├── profiler/                # Nsight Compute (ncu) profiling drivers
│   └── profile_*.py         # NVTX-annotated single-launch scripts for ncu
│
├── cmake/                   # Build helpers
│   ├── CPM.cmake            # CPM dependency manager
│   ├── dependencies.cmake   # Fetches spdlog, googletest, CUTLASS, nvbench
│   └── utils.cmake          # CUDA arch detection, benchmark helpers
│
└── scripts/                 # Dev environment + build scripts
    ├── env.sh               # Shell helpers (status logging)
    └── local/build.sh       # One-shot: venv + deps + build wheel + install + verify ops
```

### How a kernel is wired together

Each kernel flows through the same layers:

1. **CUDA source** in `csrc/<family>/src/.../*.cu` implements the kernel and a
   `torch::Tensor` entry function.
2. **Header** `include/cudabox_ops.hpp` declares that entry function under the
   `cudabox::<family>` namespace.
3. **Op registration** in `csrc/<family>/src/<family>_ops.cpp` binds the
   function into the `cudabox` Torch library via `TORCH_LIBRARY_FRAGMENT` +
   `REGISTER_EXTENSION`, producing a compiled `<family>_ops` Python extension.
4. **Python wrapper** in `python/cudabox/<family>.py` calls
   `torch.ops.cudabox.<op>` so it's usable as a normal Python function.

## Build & Install

The kernels are packaged as a Python wheel via `scikit-build-core`.

The simplest path is the all-in-one script (creates a `uv` venv, installs build
deps, builds, installs, and verifies the registered ops):

```bash
./scripts/local/build.sh            # RelWithDebInfo by default
./scripts/local/build.sh --clean    # wipe build/ and dist/ first
./scripts/local/build.sh --build-type Release
```

Or build manually:

```bash
uv pip install "scikit-build-core>=0.11" wheel "torch>=2.7.0" triton numpy pytest
uv build --wheel -Cbuild-dir=build . --no-build-isolation
uv pip install ./dist/cudabox*.whl --force-reinstall
```

The CUDA architecture is auto-detected at configure time (falling back to `90a`
for Hopper). Third-party dependencies (CUTLASS, spdlog, googletest, and
optionally nvbench) are fetched automatically via CPM.

**Requirements:** an NVIDIA GPU + CUDA toolkit, Python ≥ 3.12, PyTorch ≥ 2.7,
CMake ≥ 3.30, and Ninja.

## Usage

```python
import torch
import cudabox

x = torch.randn(4096, device="cuda")

y = cudabox.elementwise.softmax(x)
y = cudabox.elementwise.online_softmax(x)

gamma = torch.ones(4096, device="cuda")
y = cudabox.elementwise.rmsnorm(x, gamma, eps=1e-5)

a = torch.randn(512, 512, device="cuda")
b = torch.randn(512, 512, device="cuda")
c = cudabox.gemm.tiled_gemm(a, b)
```

## Testing

Correctness is checked against the equivalent PyTorch op with `torch.testing`:

```bash
pytest tests/
pytest tests/test_softmax.py
```

## Benchmarking

Python benchmarks use the `triton.testing` harness and write results (and plots)
to `benchmarks/results/`:

```bash
# Need to be executed from the repo root
python ./benchmarks/bench_softmax.py
python ./benchmarks/bench_gemm.py
python ./benchmarks/bench_flash_attention.py
```

Optional C++ microbenchmarks (nvbench) for GEMM can be enabled at configure time
with `-DCUDABOX_ENABLE_BENCHMARKS=ON`.

## Profiling

Each kernel has an NVTX-annotated driver under `profiler/` for use with Nsight
Compute. See the docstring in each script for ready-to-run `ncu` invocations,
e.g.:

```bash
ncu --set full --nvtx --nvtx-include "softmax_profile/" \
    --target-processes all \
    python -m profiler.profile_softmax
```
