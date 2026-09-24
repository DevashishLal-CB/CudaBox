#include "cuda_utils.cuh"
#include "cudabox_ops.hpp"
#include "logger.hpp"
#include "torch_utils.hpp"

#include <ATen/Dispatch.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cmath>
#include <optional>

namespace cg = cooperative_groups;

namespace cudabox::algorithms {

namespace {

constexpr int WARP_SIZE = 32;
constexpr int QUERY_ROWS_PER_BLOCK = 8;
constexpr int KEYS_PER_TILE = 32;
constexpr int FLASH_ATTENTION_THREADS = QUERY_ROWS_PER_BLOCK * WARP_SIZE;

template <typename T, int HEAD_DIM> struct FlashAttentionSharedStorage {
  alignas(16) T query[QUERY_ROWS_PER_BLOCK][HEAD_DIM];
  alignas(16) T key[KEYS_PER_TILE][HEAD_DIM];
  alignas(16) T value[KEYS_PER_TILE][HEAD_DIM];
  float probabilities[QUERY_ROWS_PER_BLOCK][KEYS_PER_TILE];
};

// Educational Flash Attention forward kernel. A block owns a tile of query
// rows and each warp owns one row. Q stays in shared memory for the lifetime of
// the block; K and V stream through shared memory in KEYS_PER_TILE chunks. The
// online-softmax recurrence keeps only one score tile and an fp32 output
// accumulator, so no O(sequence_length^2) tensor is written to global memory.
//
// This is deliberately optimized for readability. Natural next steps are
// vectorized global-to-shared copies, tensor-core QK/PV matmuls,
// double-buffered K/V tiles, and warp-specialized TMA pipelines.
template <typename T, int HEAD_DIM>
__global__
__launch_bounds__(FLASH_ATTENTION_THREADS) void flash_attention_kernel(
    const T *__restrict__ query, const T *__restrict__ key,
    const T *__restrict__ value, T *__restrict__ output, int64_t num_heads,
    int64_t sequence_length, float scale, bool is_causal) {
  const auto block = cg::this_thread_block();
  const auto warp = cg::tiled_partition<WARP_SIZE>(block);

  __shared__ FlashAttentionSharedStorage<T, HEAD_DIM> shared;

  constexpr int OUTPUTS_PER_LANE = utils::ceil_div(HEAD_DIM, WARP_SIZE);
  const int thread_id = block.thread_rank();
  const int warp_id = warp.meta_group_rank();
  const int lane_id = warp.thread_rank();
  const int64_t query_tile_start = blockIdx.x * QUERY_ROWS_PER_BLOCK;
  const int64_t query_index = query_tile_start + warp_id;
  const int64_t head_offset =
      (static_cast<int64_t>(blockIdx.z) * num_heads + blockIdx.y) *
      sequence_length * HEAD_DIM;
  const bool query_is_valid = query_index < sequence_length;

  // Load the block's Q tile once. It is reused for every K/V tile.
  constexpr int QUERY_TILE_ELEMENTS = QUERY_ROWS_PER_BLOCK * HEAD_DIM;
  for (int index = thread_id; index < QUERY_TILE_ELEMENTS;
       index += FLASH_ATTENTION_THREADS) {
    const int row = index / HEAD_DIM;
    const int dim = index % HEAD_DIM;
    const int64_t global_row = query_tile_start + row;
    shared.query[row][dim] =
        global_row < sequence_length
            ? query[head_offset + global_row * HEAD_DIM + dim]
            : T{0};
  }

  float row_max = -INFINITY;
  float row_sum = 0.0f;
  float output_accumulator[OUTPUTS_PER_LANE];
#pragma unroll
  for (int output_index = 0; output_index < OUTPUTS_PER_LANE; ++output_index) {
    output_accumulator[output_index] = 0.0f;
  }
  block.sync();

  for (int64_t key_tile_start = 0; key_tile_start < sequence_length;
       key_tile_start += KEYS_PER_TILE) {
    // Cooperatively stage one K and V tile in shared memory. Invalid elements
    // in the final partial tile are zero-filled.
    constexpr int KEY_VALUE_TILE_ELEMENTS = KEYS_PER_TILE * HEAD_DIM;
    for (int index = thread_id; index < KEY_VALUE_TILE_ELEMENTS;
         index += FLASH_ATTENTION_THREADS) {
      const int row = index / HEAD_DIM;
      const int dim = index % HEAD_DIM;
      const int64_t key_index = key_tile_start + row;
      if (key_index < sequence_length) {
        const int64_t offset = head_offset + key_index * HEAD_DIM + dim;
        shared.key[row][dim] = key[offset];
        shared.value[row][dim] = value[offset];
      } else {
        shared.key[row][dim] = T{0};
        shared.value[row][dim] = T{0};
      }
    }
    block.sync();

    // A warp computes one row of the QK^T tile: each lane owns one key.
    const int64_t key_index = key_tile_start + lane_id;
    const bool score_is_valid = query_is_valid && key_index < sequence_length &&
                                (!is_causal || key_index <= query_index);
    float score = -INFINITY;
    if (score_is_valid) {
      score = 0.0f;
      for (int dim = 0; dim < HEAD_DIM; ++dim) {
        score += static_cast<float>(shared.query[warp_id][dim]) *
                 static_cast<float>(shared.key[lane_id][dim]);
      }
      score *= scale;
    }

    // Online softmax update for this K/V tile:
    //   m_new = max(m_old, tile_max)
    //   l_new = exp(m_old-m_new) * l_old + sum(exp(score-m_new))
    const float tile_max = cg::reduce(warp, score, cg::greater<float>());
    const float new_row_max = fmaxf(row_max, tile_max);
    const float old_row_scale =
        row_sum == 0.0f ? 0.0f : expf(row_max - new_row_max);
    const float probability = score_is_valid ? expf(score - new_row_max) : 0.0f;
    const float tile_sum = cg::reduce(warp, probability, cg::plus<float>());

    if (query_is_valid) {
      row_max = new_row_max;
      row_sum = row_sum * old_row_scale + tile_sum;
    }

    shared.probabilities[warp_id][lane_id] = probability;
    warp.sync();

    // Multiply the unnormalized probability tile by V. Each lane owns a
    // strided slice of the output head dimension and keeps it in registers.
#pragma unroll
    for (int output_index = 0; output_index < OUTPUTS_PER_LANE;
         ++output_index) {
      const int dim = lane_id + output_index * WARP_SIZE;
      if (query_is_valid && dim < HEAD_DIM) {
        float tile_output = 0.0f;
#pragma unroll
        for (int key_in_tile = 0; key_in_tile < KEYS_PER_TILE; ++key_in_tile) {
          tile_output += shared.probabilities[warp_id][key_in_tile] *
                         static_cast<float>(shared.value[key_in_tile][dim]);
        }
        output_accumulator[output_index] =
            output_accumulator[output_index] * old_row_scale + tile_output;
      }
    }

    block.sync();
  }

  // Normalize O only once after every K/V tile has participated.
#pragma unroll
  for (int output_index = 0; output_index < OUTPUTS_PER_LANE; ++output_index) {
    const int dim = lane_id + output_index * WARP_SIZE;
    if (query_is_valid && dim < HEAD_DIM) {
      output[head_offset + query_index * HEAD_DIM + dim] =
          static_cast<T>(output_accumulator[output_index] / row_sum);
    }
  }
}

template <typename T, int HEAD_DIM>
cudaError_t flash_attention_launch(const T *query, const T *key, const T *value,
                                   T *output, int64_t batch_size,
                                   int64_t num_heads, int64_t sequence_length,
                                   float scale, bool is_causal,
                                   cudaStream_t stream = nullptr) {
  cudaLaunchConfig_t config{};
  config.gridDim =
      dim3(static_cast<unsigned int>(utils::ceil_div(
               static_cast<int>(sequence_length), QUERY_ROWS_PER_BLOCK)),
           static_cast<unsigned int>(num_heads),
           static_cast<unsigned int>(batch_size));
  config.blockDim = dim3(FLASH_ATTENTION_THREADS);
  config.stream = stream;

  auto kernel = flash_attention_kernel<T, HEAD_DIM>;
  CUDABOX_LOG_DEBUG(
      "Dispatching flash_attention, batch={}, heads={}, sequence={}, "
      "head_dim={}, causal={}, q_tile={}, kv_tile={}",
      batch_size, num_heads, sequence_length, HEAD_DIM, is_causal,
      QUERY_ROWS_PER_BLOCK, KEYS_PER_TILE);
  CUDABOX_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, query, key, value,
                                       output, num_heads, sequence_length,
                                       scale, is_causal));

  return cudaSuccess;
}

template <typename T>
cudaError_t dispatch_flash_attention_head_dim(
    const T *query, const T *key, const T *value, T *output, int64_t batch_size,
    int64_t num_heads, int64_t sequence_length, int64_t head_dim, float scale,
    bool is_causal, cudaStream_t stream) {
  switch (head_dim) {
  case 16:
    return flash_attention_launch<T, 16>(query, key, value, output, batch_size,
                                         num_heads, sequence_length, scale,
                                         is_causal, stream);
  case 32:
    return flash_attention_launch<T, 32>(query, key, value, output, batch_size,
                                         num_heads, sequence_length, scale,
                                         is_causal, stream);
  case 64:
    return flash_attention_launch<T, 64>(query, key, value, output, batch_size,
                                         num_heads, sequence_length, scale,
                                         is_causal, stream);
  case 128:
    return flash_attention_launch<T, 128>(query, key, value, output, batch_size,
                                          num_heads, sequence_length, scale,
                                          is_causal, stream);
  default:
    return cudaErrorInvalidValue;
  }
}

void check_flash_attention_inputs(const torch::Tensor &query,
                                  const torch::Tensor &key,
                                  const torch::Tensor &value,
                                  const std::optional<double> scale) {
  TORCH_TENSOR_CHECK(query);
  TORCH_TENSOR_CHECK(key);
  TORCH_TENSOR_CHECK(value);

  TORCH_CHECK(query.dim() == 4,
              "flash_attention expects query, key, and value to have shape "
              "(batch, heads, sequence, head_dim)");
  TORCH_CHECK(query.sizes() == key.sizes() && query.sizes() == value.sizes(),
              "flash_attention currently requires query, key, and value to "
              "have the same shape");
  TORCH_CHECK(query.device() == key.device() &&
                  query.device() == value.device(),
              "flash_attention requires query, key, and value on the same "
              "CUDA device");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                  query.scalar_type() == value.scalar_type(),
              "flash_attention requires query, key, and value to have the "
              "same dtype");
  TORCH_CHECK(query.scalar_type() == torch::kFloat16 ||
                  query.scalar_type() == torch::kBFloat16 ||
                  query.scalar_type() == torch::kFloat32,
              "flash_attention supports float16, bfloat16, and float32");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() &&
                  value.is_contiguous(),
              "flash_attention requires contiguous query, key, and value");
  TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0 && query.size(2) > 0 &&
                  query.size(3) > 0,
              "flash_attention requires non-empty dimensions");
  TORCH_CHECK(query.size(3) == 16 || query.size(3) == 32 ||
                  query.size(3) == 64 || query.size(3) == 128,
              "flash_attention supports head dimensions 16, 32, 64, and 128; "
              "got ",
              query.size(3));
  TORCH_CHECK(!scale.has_value() || (std::isfinite(*scale) && *scale > 0.0),
              "flash_attention scale must be finite and greater than zero");
}

} // namespace

torch::Tensor flash_attention(const torch::Tensor &query,
                              const torch::Tensor &key,
                              const torch::Tensor &value, bool is_causal,
                              std::optional<double> scale) {
  check_flash_attention_inputs(query, key, value, scale);

  const c10::cuda::OptionalCUDAGuard device_guard(query.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t sequence_length = query.size(-2);
  const int64_t head_dim = query.size(-1);
  const float attention_scale = static_cast<float>(
      scale.value_or(1.0 / std::sqrt(static_cast<double>(head_dim))));
  auto output = torch::empty_like(query);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(),
      "flash_attention", [&] {
        const cudaError_t status = dispatch_flash_attention_head_dim<scalar_t>(
            query.data_ptr<scalar_t>(), key.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), batch_size,
            num_heads, sequence_length, head_dim, attention_scale, is_causal,
            stream);
        TORCH_CHECK(status == cudaSuccess, "flash_attention failed");
      });

  return output;
}

} // namespace cudabox::algorithms
