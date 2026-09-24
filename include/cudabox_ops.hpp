#pragma once

#include <optional>

#include <torch/torch.h>

namespace cudabox {

namespace algorithms {

torch::Tensor histogram(const torch::Tensor &tensor, int64_t num_bins);

torch::Tensor flash_attention(const torch::Tensor &query,
                              const torch::Tensor &key,
                              const torch::Tensor &value, bool is_causal,
                              std::optional<double> scale);

} // namespace algorithms

namespace elementwise {

torch::Tensor softmax(const torch::Tensor &tensor);

torch::Tensor online_softmax(const torch::Tensor &tensor);

torch::Tensor rmsnorm(const torch::Tensor &tensor, const torch::Tensor &gamma,
                      const double eps);

namespace sm90 {

torch::Tensor online_softmax(const torch::Tensor &tensor);

} // namespace sm90

} // namespace elementwise

namespace gemm {

torch::Tensor simple_gemm(const torch::Tensor &mat_a,
                          const torch::Tensor &mat_b);

torch::Tensor tiled_gemm(const torch::Tensor &mat_a,
                         const torch::Tensor &mat_b);

namespace sm90_pipelined_tma_mma {

torch::Tensor gemm(const torch::Tensor &mat_a, const torch::Tensor &mat_b);

} // namespace sm90_pipelined_tma_mma

} // namespace gemm

} // namespace cudabox
