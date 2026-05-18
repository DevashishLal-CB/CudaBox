#include "cuda_utils.cuh"
#include "cudabox_ops.hpp"
#include "logger.hpp"
#include "torch_utils.hpp"

namespace cudabox::gemm {

template <typename T>
__global__ void simple_gemm_kernel(T *__restrict__ A, T *__restrict__ B,
                                   T *__restrict__ C, unsigned int M,
                                   unsigned int N, unsigned int K) {
  // Index calculation for the grid
  unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;

  // Boundary check
  if ((row >= M) || (col >= N))
    return;

  T accumulator = {0};

  // All linear indexes follow the following pattern
  // linear_index = row_no * no_of_columns [i.e row width] + col_no
  for (int i = 0; i < K; i++) {
    accumulator += A[row * K + i] * B[i * N + col];
  }

  C[row * N + col] = accumulator;
}

template <typename T>
cudaError_t simple_gemm_launch(T *A, T *B, T *C, unsigned int M, unsigned int N,
                               unsigned int K, cudaStream_t stream = 0) {
  constexpr unsigned int tile_size = 32;

  dim3 nblks(cudabox::utils::ceil_div(N, tile_size),
             cudabox::utils::ceil_div(M, tile_size), 1);
  dim3 nthrs(tile_size, tile_size, 1);

  cudaLaunchConfig_t config{};
  config.gridDim = nblks;
  config.blockDim = nthrs;
  config.stream = stream;

  auto kernel = simple_gemm_kernel<T>;

  CUDABOX_LOG_DEBUG("Dispatching simple gemm M={}, N={}, K={}", M, N, K);
  CUDABOX_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, A, B, C, M, N, K));

  return cudaSuccess;
}

torch::Tensor simple_gemm(const torch::Tensor &mat_a,
                          const torch::Tensor &mat_b) {
  TORCH_TENSOR_CHECK(mat_a);
  TORCH_TENSOR_CHECK(mat_b);

  TORCH_CHECK(mat_a.size(1) == mat_b.size(0),
              "Tensors dimensions are not compatible for matmul");

  unsigned int M = mat_a.size(0);
  unsigned int N = mat_b.size(1);
  unsigned int K = mat_a.size(1);

  torch::Tensor mat_c =
      torch::empty({M, N}, torch::dtype(mat_a.dtype()).device(torch::kCUDA));

  auto device = mat_a.device();

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  cudaError_t status =
      simple_gemm_launch(mat_a.data_ptr<float>(), mat_b.data_ptr<float>(),
                         mat_c.data_ptr<float>(), M, N, K, stream);

  TORCH_CHECK(status == cudaSuccess, "sgemm failed");

  return mat_c;
}

} // namespace cudabox::gemm
