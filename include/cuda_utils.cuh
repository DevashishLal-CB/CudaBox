#pragma once

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#ifndef NDEBUG
#define CUDABOX_CUDA_CALL(func, ...)                                           \
  {                                                                            \
    cudaError_t e = (func);                                                    \
    if (e != cudaSuccess) {                                                    \
      std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " (" << e        \
                << ") " << __FILE__ << ": line " << __LINE__                   \
                << " at function " << STR(func) << std::endl;                  \
      return e;                                                                \
    }                                                                          \
  }
#else
#define CUDABOX_CUDA_CALL(func, ...)                                           \
  {                                                                            \
    cudaError_t e = (func);                                                    \
    if (e != cudaSuccess) {                                                    \
      return e;                                                                \
    }                                                                          \
  }
#endif

namespace cudabox::utils {
constexpr int THREADS_PER_BLOCK = 256;
constexpr int THREADS_PER_WARP = 32;
constexpr int FULL_MASK = 0xffffffff;
constexpr int NUM_WARPS = THREADS_PER_BLOCK / THREADS_PER_WARP;
constexpr int BLOCKS_PER_CLUSTER = 8;

__host__ __device__ constexpr int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

// CUDA has no atomicMax for float; emulate via CAS on the bit-pattern.
__device__ inline float atomic_max_float(float *addr, float value) {
  int *addr_as_int = reinterpret_cast<int *>(addr);
  int old_int = *addr_as_int;
  int assumed;
  do {
    assumed = old_int;
    float assumed_f = __int_as_float(assumed);
    if (value <= assumed_f) {
      break;
    }
    old_int = atomicCAS(addr_as_int, assumed, __float_as_int(value));
  } while (assumed != old_int);
  return __int_as_float(old_int);
}

} // namespace cudabox::utils
