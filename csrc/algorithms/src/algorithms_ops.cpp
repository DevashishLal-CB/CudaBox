#if defined(Py_LIMITED_API)
#include "cudabox_ops.hpp"
#include "python_utils.hpp"

TORCH_LIBRARY_FRAGMENT(cudabox, m) {
  m.def("histogram(Tensor tensor, int num_bins) -> Tensor");
  m.impl("histogram", torch::kCUDA, &cudabox::algorithms::histogram);

  m.def("flash_attention(Tensor query, Tensor key, Tensor value, "
        "bool is_causal=False, float? scale=None) -> Tensor");
  m.impl("flash_attention", torch::kCUDA,
         &cudabox::algorithms::flash_attention);
}

REGISTER_EXTENSION(algorithms_ops)
#endif
