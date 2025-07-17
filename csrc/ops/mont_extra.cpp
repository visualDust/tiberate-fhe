#include "cuda/mont_extra_cuda.h"

std::vector<torch::Tensor> mont_add_many_3d(
    const std::vector<torch::Tensor> inputs, const int64_t sp_prime_len) {
  std::vector<torch::Tensor> outputs;
  const auto num_devices = inputs.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto out = mont_add_many_3d_cuda(inputs[i], sp_prime_len);
    outputs.push_back(out);
  }
  return outputs;
}

std::vector<torch::Tensor> mont_reduce_add_many_3d(
    const std::vector<torch::Tensor> inputs, const int64_t sp_prime_len) {
  std::vector<torch::Tensor> outputs;
  const auto num_devices = inputs.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto out = mont_reduce_add_many_3d_cuda(inputs[i], sp_prime_len);
    outputs.push_back(out);
  }
  return outputs;
}

std::vector<torch::Tensor> mont_add_reduce_2q(
    const std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> b,
    const int64_t sp_prime_len) {
  std::vector<torch::Tensor> outputs;

  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto out = mont_add_reduce_2q_cuda(a[i], b[i], sp_prime_len);
    outputs.push_back(out);
  }
  return outputs;
}

std::vector<torch::Tensor> mont_sub_reduce_2q(
    const std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> b,
    const int64_t sp_prime_len) {
  std::vector<torch::Tensor> outputs;

  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto out = mont_sub_reduce_2q_cuda(a[i], b[i], sp_prime_len);
    outputs.push_back(out);
  }
  return outputs;
}

std::vector<torch::Tensor> mont_enter_scalar_reduce_2q(
    const std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> b,
    const int64_t sp_prime_len) {
  std::vector<torch::Tensor> outputs;

  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto out = mont_enter_scalar_reduce_2q_cuda(a[i], b[i], sp_prime_len);
    outputs.push_back(out);
  }
  return outputs;
}

TORCH_LIBRARY_FRAGMENT(tiberate_mont_ops, m) {
  m.def("mont_add_many_3d(Tensor[] input, int sp_prime_len) -> Tensor[]");
  m.def(
      "mont_reduce_add_many_3d(Tensor[] input, int sp_prime_len) -> Tensor[]");
  m.def(
      "mont_add_reduce_2q(Tensor[] a, Tensor[] b, int sp_prime_len) -> "
      "Tensor[]");
  m.def(
      "mont_sub_reduce_2q(Tensor[] a, Tensor[] b, int sp_prime_len) -> "
      "Tensor[]");
  m.def(
      "mont_enter_scalar_reduce_2q(Tensor[] a, Tensor[] b, int "
      "sp_prime_len) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(tiberate_mont_ops, CUDA, m) {
  m.impl("mont_add_many_3d", &mont_add_many_3d);
  m.impl("mont_reduce_add_many_3d", &mont_reduce_add_many_3d);
  m.impl("mont_add_reduce_2q", &mont_add_reduce_2q);
  m.impl("mont_sub_reduce_2q", &mont_sub_reduce_2q);
  m.impl("mont_enter_scalar_reduce_2q", &mont_enter_scalar_reduce_2q);
}
