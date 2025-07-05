#include "cuda/mont_fused_cuda.h"

std::vector<torch::Tensor> mont_add_many_3d(
    const std::vector<torch::Tensor> inputs,
    const std::vector<torch::Tensor> _2q) {
  std::vector<torch::Tensor> outputs;
  const auto num_devices = inputs.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto out = mont_add_many_3d_cuda(inputs[i], _2q[i]);
    outputs.push_back(out);
  }
  return outputs;
}

std::vector<torch::Tensor> mont_reduce_add_many_3d(
    const std::vector<torch::Tensor> inputs,
    const std::vector<torch::Tensor> _2q) {
  std::vector<torch::Tensor> outputs;
  const auto num_devices = inputs.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto out = mont_reduce_add_many_3d_cuda(inputs[i], _2q[i]);
    outputs.push_back(out);
  }
  return outputs;
}

std::vector<torch::Tensor> mont_add_reduce_2q(
    const std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> b,
    const std::vector<torch::Tensor> _2q) {
  std::vector<torch::Tensor> outputs;

  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto out = mont_add_reduce_2q_cuda(a[i], b[i], _2q[i]);
    outputs.push_back(out);
  }
  return outputs;
}

std::vector<torch::Tensor> mont_sub_reduce_2q(
    const std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> b,
    const std::vector<torch::Tensor> _2q) {
  std::vector<torch::Tensor> outputs;

  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto out = mont_sub_reduce_2q_cuda(a[i], b[i], _2q[i]);
    outputs.push_back(out);
  }
  return outputs;
}

std::vector<torch::Tensor> mont_enter_reduce_2q(
    const std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> Rs,
    const std::vector<torch::Tensor> _2q,
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh) {
  std::vector<torch::Tensor> outputs;

  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto out = mont_enter_reduce_2q_cuda(
        a[i], Rs[i], _2q[i], ql[i], qh[i], kl[i], kh[i]);
    outputs.push_back(out);
  }
  return outputs;
}

TORCH_LIBRARY_FRAGMENT(tiberate_fused_ops, m) {
  m.def("mont_add_many_3d(Tensor[] input, Tensor[] _2q) -> Tensor[]",
        &mont_add_many_3d);
  m.def("mont_reduce_add_many_3d(Tensor[] input, Tensor[] _2q) -> Tensor[]",
        &mont_reduce_add_many_3d);
  m.def("mont_add_reduce_2q(Tensor[] a, Tensor[] b, Tensor[] _2q) -> Tensor[]");
  m.def("mont_sub_reduce_2q(Tensor[] a, Tensor[] b, Tensor[] _2q) -> Tensor[]");
  m.def(
      "mont_enter_reduce_2q(Tensor[] a, Tensor[] Rs, "
      "Tensor[] _2q, Tensor[] ql, Tensor[] qh, "
      "Tensor[] kl, Tensor[] kh) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(tiberate_fused_ops, CUDA, m) {
  m.impl("mont_add_many_3d", &mont_add_many_3d);
  m.impl("mont_reduce_add_many_3d", &mont_reduce_add_many_3d);
  m.impl("mont_add_reduce_2q", &mont_add_reduce_2q);
  m.impl("mont_sub_reduce_2q", &mont_sub_reduce_2q);
  m.impl("mont_enter_reduce_2q", &mont_enter_reduce_2q);
}
