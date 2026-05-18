#if defined(Py_LIMITED_API)
#include "cudabox_ops.hpp"
#include "python_utils.hpp"

TORCH_LIBRARY_FRAGMENT(cudabox, m) {
  m.def("softmax(Tensor tensor) -> Tensor");
  m.impl("softmax", torch::kCUDA, &cudabox::elementwise::softmax);
}

REGISTER_EXTENSION(elementwise_ops)
#endif
