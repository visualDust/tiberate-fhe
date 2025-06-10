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

std::vector<torch::Tensor> pc_add_fused(
    const std::vector<torch::Tensor> ct_data,
    const std::vector<torch::Tensor> pt_data,
    const std::vector<torch::Tensor> _2q,
    const std::vector<torch::Tensor> Rs,
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh) {
  std::vector<torch::Tensor> outputs;

  const auto num_devices = ct_data.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto out = pc_add_fused_cuda(
        ct_data[i], pt_data[i], _2q[i], Rs[i], ql[i], qh[i], kl[i], kh[i]);
    outputs.push_back(out);
  }
  return outputs;
}

void rescale_exact_rounding_fused(std::vector<torch::Tensor> a,
                                  const std::vector<torch::Tensor> Rs,
                                  const std::vector<torch::Tensor> rescaler,
                                  const int64_t round_at,
                                  const std::vector<torch::Tensor> _2q,
                                  const std::vector<torch::Tensor> ql,
                                  const std::vector<torch::Tensor> qh,
                                  const std::vector<torch::Tensor> kl,
                                  const std::vector<torch::Tensor> kh) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    rescale_exact_rounding_fused_cuda(
        a[i], Rs[i], rescaler[i], round_at, _2q[i], ql[i], qh[i], kl[i], kh[i]);
  }
}

void rescale_non_exact_rounding_fused(std::vector<torch::Tensor> a,
                                      const std::vector<torch::Tensor> Rs,
                                      const std::vector<torch::Tensor> rescaler,
                                      const std::vector<torch::Tensor> _2q,
                                      const std::vector<torch::Tensor> ql,
                                      const std::vector<torch::Tensor> qh,
                                      const std::vector<torch::Tensor> kl,
                                      const std::vector<torch::Tensor> kh) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    rescale_non_exact_rounding_fused_cuda(
        a[i], Rs[i], rescaler[i], _2q[i], ql[i], qh[i], kl[i], kh[i]);
  }
}

TORCH_LIBRARY_FRAGMENT(tiberate_fused_ops, m) {
  m.def("mont_add_reduce_2q(Tensor[] a, Tensor[] b, Tensor[] _2q) -> Tensor[]");
  m.def(
      "pc_add_fused(Tensor[] ct_data, Tensor[] pt_data, "
      "Tensor[] Rs, Tensor[] ql, Tensor[] qh, "
      "Tensor[] kl, Tensor[] kh, Tensor[] _2q) -> Tensor[]");
  m.def(
      "rescale_exact_rounding_fused(Tensor[] a, Tensor[] Rs, "
      "Tensor[] rescaler, int round_at, "
      "Tensor[] _2q, Tensor[] ql, Tensor[] qh, "
      "Tensor[] kl, Tensor[] kh) -> ()");
  m.def(
      "rescale_non_exact_rounding_fused(Tensor[] a, Tensor[] Rs, "
      "Tensor[] rescaler, Tensor[] _2q, "
      "Tensor[] ql, Tensor[] qh, "
      "Tensor[] kl, Tensor[] kh) -> ()");
}

TORCH_LIBRARY_IMPL(tiberate_fused_ops, CUDA, m) {
  m.impl("mont_add_reduce_2q", &mont_add_reduce_2q);
  m.impl("pc_add_fused", &pc_add_fused);
  m.impl("rescale_exact_rounding_fused", &rescale_exact_rounding_fused);
  m.impl("rescale_non_exact_rounding_fused", &rescale_non_exact_rounding_fused);
}
