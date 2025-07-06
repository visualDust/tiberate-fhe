#include "cuda/cascade_mac_cuda.h"

std::vector<torch::Tensor> mont_mult_sum_many_3d(
    const std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> b,
    const std::vector<torch::Tensor> _2q,
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh) {
  std::vector<torch::Tensor> outputs;

  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto out = mont_mult_sum_many_3d_cuda(
        a[i], b[i], _2q[i], ql[i], qh[i], kl[i], kh[i]);
    outputs.push_back(out);
  }
  return outputs;
}

TORCH_LIBRARY_FRAGMENT(tiberate_he_ops, m) {
  m.def(
      "mont_mult_sum_many_3d(Tensor[] a, Tensor[] b, Tensor[] _2q, "
      "Tensor[] ql, Tensor[] qh, Tensor[] kl, Tensor[] kh) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(tiberate_he_ops, CUDA, m) {
  m.impl("mont_mult_sum_many_3d", mont_mult_sum_many_3d);
}
