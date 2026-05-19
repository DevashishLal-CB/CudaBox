#include <cooperative_groups.h>

#include "cuda_utils.cuh"
#include "cudabox_ops.hpp"
#include "logger.hpp"
#include "torch_utils.hpp"

namespace cg = cooperative_groups;

namespace cudabox::elementwise {

constexpr int THREADS_PER_BLOCK = 256;
constexpr int THREADS_PER_WARP = 32;
constexpr int FULL_MASK = 0xffffffff;
constexpr int NUM_WARPS = THREADS_PER_BLOCK / THREADS_PER_WARP;

// Global-memory scratch layout: [global_max, global_sum]
constexpr int GMEM_MAX_IDX = 0;
constexpr int GMEM_SUM_IDX = 1;
constexpr int GMEM_SIZE = 2;

struct SumReducer {
  __device__ inline static float invoke(float value, int mask = FULL_MASK,
                                        int thread_count = THREADS_PER_WARP) {
    for (int offset = thread_count / 2; offset > 0; offset /= 2) {
      value += __shfl_down_sync(mask, value, offset);
    }
    return value;
  }
};

struct MaxReducer {
  __device__ inline static float invoke(float value, int mask = FULL_MASK,
                                        int thread_count = THREADS_PER_WARP) {
    for (int offset = thread_count / 2; offset > 0; offset /= 2) {
      value = fmaxf(value, __shfl_down_sync(mask, value, offset));
    }
    return value;
  }
};

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

template <typename Reducer>
__device__ inline float block_reducer(float thread_val, float *smem,
                                      const int64_t tid) {
  // warp-level reduction
  float warp_val = Reducer::invoke(thread_val);

  // thread 0 of each warp writes its partial to shared memory
  if (tid % THREADS_PER_WARP == 0) {
    smem[tid / THREADS_PER_WARP] = warp_val;
  }
  __syncthreads();

  // first warp reduces the per-warp partials into the block result
  if (tid < NUM_WARPS) {
    // mask of the first NUM_WARPS lanes in the warp (e.g. 8 warps -> 0xff)
    constexpr unsigned int mask = (1u << NUM_WARPS) - 1u;
    float block_val = smem[tid];
    block_val = Reducer::invoke(block_val, mask, NUM_WARPS);

    if (tid == 0) {
      smem[0] = block_val;
    }
  }
  __syncthreads();

  return smem[0];
}

__global__ __launch_bounds__(THREADS_PER_BLOCK) void softmax_kernel(
    const float *__restrict__ in, float *__restrict__ out,
    float *__restrict__ gmem, const int64_t size) {
  cg::grid_group grid = cg::this_grid();

  const int64_t tid = threadIdx.x;
  const int64_t bid = blockIdx.x;
  const int64_t gtid = bid * blockDim.x + tid;
  const int64_t gstride = gridDim.x * blockDim.x;

  __shared__ float smem[NUM_WARPS];

  // Vectorized loads: process 4 floats at a time via float4 (128-bit) reads.
  const float4 *in4 = reinterpret_cast<const float4 *>(in);
  float4 *out4 = reinterpret_cast<float4 *>(out);
  const int64_t n4 = size >> 2;       // number of full float4s
  const int64_t tail_start = n4 << 2; // index of first scalar tail element

  // pass 1: local max -> block_max -> atomic-max into global memory
  // pass 1: thread reduction
  float thread_max = -INFINITY;
  for (int64_t i = gtid; i < n4; i += gstride) {
    float4 v = in4[i];
    thread_max = fmaxf(thread_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
  }
  // scalar tail (at most 3 elements)
  for (int64_t i = tail_start + gtid; i < size; i += gstride) {
    thread_max = fmaxf(thread_max, in[i]);
  }

  // pass 1: block reduction
  float block_max = block_reducer<MaxReducer>(thread_max, smem, tid);

  // pass 1: grid reduction
  if (tid == 0) {
    atomic_max_float(&gmem[GMEM_MAX_IDX], block_max);
  }
  grid.sync();
  float global_max = gmem[GMEM_MAX_IDX];

  // pass 2: write exp(x - global_max) to out, accumulate block_sum,
  // pass 2: thread reduction
  float thread_sum = 0.0f;
  for (int64_t i = gtid; i < n4; i += gstride) {
    float4 v = in4[i];
    float4 e;
    e.x = __expf(v.x - global_max);
    e.y = __expf(v.y - global_max);
    e.z = __expf(v.z - global_max);
    e.w = __expf(v.w - global_max);
    thread_sum += e.x + e.y + e.z + e.w;
    out4[i] = e;
  }
  for (int64_t i = tail_start + gtid; i < size; i += gstride) {
    float x = __expf(in[i] - global_max);
    thread_sum += x;
    out[i] = x;
  }

  // pass 2: block reduction
  float block_sum = block_reducer<SumReducer>(thread_sum, smem, tid);

  // pass 2: grid reduction
  if (tid == 0) {
    atomicAdd(&gmem[GMEM_SUM_IDX], block_sum);
  }
  grid.sync();
  float global_sum = gmem[GMEM_SUM_IDX];

  // pass 3: normalize. Multiply by reciprocal — cheaper than divide.
  const float inv_sum = 1.0f / global_sum;
  for (int64_t i = gtid; i < n4; i += gstride) {
    float4 v = out4[i];
    v.x *= inv_sum;
    v.y *= inv_sum;
    v.z *= inv_sum;
    v.w *= inv_sum;
    out4[i] = v;
  }
  for (int64_t i = tail_start + gtid; i < size; i += gstride) {
    out[i] = out[i] * inv_sum;
  }
}

cudaError_t softmax_launch(const float *in, float *out, float *gmem,
                           const int64_t size, cudaStream_t stream = 0) {
  // Cooperative launch requires the grid to be co-resident on the GPU.
  // Cap blocks to the device's max active cooperative blocks; the grid-stride
  // loop in the kernel handles any remaining work.
  int device = 0;
  CUDABOX_CUDA_CALL(cudaGetDevice(&device));
  int sm_count = 0;
  CUDABOX_CUDA_CALL(cudaDeviceGetAttribute(
      &sm_count, cudaDevAttrMultiProcessorCount, device));
  int blocks_per_sm = 0;
  CUDABOX_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, softmax_kernel, THREADS_PER_BLOCK, 0));
  const int max_cooperative_blocks = sm_count * 1;

  const int64_t requested_blocks =
      (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  const int blocks = static_cast<int>(
      std::min<int64_t>(requested_blocks, max_cooperative_blocks));

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(blocks);
  config.blockDim = dim3(THREADS_PER_BLOCK);
  config.stream = stream;

  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeCooperative;
  attrs[0].val.cooperative = 1;
  config.attrs = attrs;
  config.numAttrs = 1;

  CUDABOX_LOG_DEBUG(
      "Dispatching softmax, size={}, blocks={}, sm_count={}, blocks_per_sm={}",
      size, blocks, sm_count, blocks_per_sm);
  CUDABOX_CUDA_CALL(
      cudaLaunchKernelEx(&config, softmax_kernel, in, out, gmem, size));

  return cudaSuccess;
}

torch::Tensor softmax(const torch::Tensor &tensor) {
  TORCH_TENSOR_CHECK(tensor);

  TORCH_CHECK(tensor.dim() == 1, "softmax only supports 1D tensors");
  TORCH_CHECK(tensor.is_contiguous(),
              "softmax only supports contiguous tensors");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
              "softmax only supports float32 tensors");

  auto device = tensor.device();

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  const int64_t size = tensor.size(0);

  auto out = torch::empty_like(tensor);

  // gmem[0] = -inf (max identity), gmem[1] = 0 (sum identity)
  auto gmem = torch::empty({GMEM_SIZE}, tensor.options());
  gmem[GMEM_MAX_IDX] = -std::numeric_limits<float>::infinity();
  gmem[GMEM_SUM_IDX] = 0.0f;

  cudaError_t status =
      softmax_launch(tensor.data_ptr<float>(), out.data_ptr<float>(),
                     gmem.data_ptr<float>(), size, stream);

  TORCH_CHECK(status == cudaSuccess, "softmax failed");
  return out;
}
} // namespace cudabox::elementwise
