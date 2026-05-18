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
__host__ __device__ inline int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

} // namespace cudabox::utils
