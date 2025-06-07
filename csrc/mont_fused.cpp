#include <torch/library.h>
#include "cuda/mont_fused_cuda.h"
#include "extensions.h"

std::vector<torch::Tensor> mont_add_reduce_2q(
    const std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> b,
    const std::vector<torch::Tensor> _2q) {
  std::vector<torch::Tensor> outputs;

  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto c = mont_add_reduce_2q_cuda(a[i], b[i], _2q[i]);
    outputs.push_back(c);
  }
  return outputs;
}

TORCH_LIBRARY_FRAGMENT(tiberate_fused_ops, m) {
  m.def("mont_add_reduce_2q(Tensor[] a, Tensor[] b, Tensor[] _2q) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(tiberate_fused_ops, CUDA, m) {
  m.impl("mont_add_reduce_2q", &mont_add_reduce_2q);
}
