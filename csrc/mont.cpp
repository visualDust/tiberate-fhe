#include <torch/library.h>
#include <vector>
#include "cuda/mont_cuda.h"
#include "extensions.h"
//------------------------------------------------------------------
// Wrap functions for Montgomery space operations
//------------------------------------------------------------------

std::vector<torch::Tensor> mont_mult(const std::vector<torch::Tensor> a,
                                     const std::vector<torch::Tensor> b,
                                     const std::vector<torch::Tensor> ql,
                                     const std::vector<torch::Tensor> qh,
                                     const std::vector<torch::Tensor> kl,
                                     const std::vector<torch::Tensor> kh) {
  std::vector<torch::Tensor> outputs;

  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto c = mont_mult_cuda(a[i], b[i], ql[i], qh[i], kl[i], kh[i]);

    outputs.push_back(c);
  }

  return outputs;
}

void mont_enter(std::vector<torch::Tensor> a,
                const std::vector<torch::Tensor> Rs,
                const std::vector<torch::Tensor> ql,
                const std::vector<torch::Tensor> qh,
                const std::vector<torch::Tensor> kl,
                const std::vector<torch::Tensor> kh) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    mont_enter_cuda(a[i], Rs[i], ql[i], qh[i], kl[i], kh[i]);
  }
}

void mont_reduce(std::vector<torch::Tensor> a,
                 const std::vector<torch::Tensor> ql,
                 const std::vector<torch::Tensor> qh,
                 const std::vector<torch::Tensor> kl,
                 const std::vector<torch::Tensor> kh) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    mont_reduce_cuda(a[i], ql[i], qh[i], kl[i], kh[i]);
  }
}

std::vector<torch::Tensor> mont_add(const std::vector<torch::Tensor> a,
                                    const std::vector<torch::Tensor> b,
                                    const std::vector<torch::Tensor> _2q) {
  std::vector<torch::Tensor> outputs;

  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto c = mont_add_cuda(a[i], b[i], _2q[i]);
    outputs.push_back(c);
  }
  return outputs;
}

std::vector<torch::Tensor> mont_sub(const std::vector<torch::Tensor> a,
                                    const std::vector<torch::Tensor> b,
                                    const std::vector<torch::Tensor> _2q) {
  std::vector<torch::Tensor> outputs;

  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto c = mont_sub_cuda(a[i], b[i], _2q[i]);
    outputs.push_back(c);
  }
  return outputs;
}

void reduce_2q(std::vector<torch::Tensor> a,
               const std::vector<torch::Tensor> _2q) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    reduce_2q_cuda(a[i], _2q[i]);
  }
}

void make_signed(std::vector<torch::Tensor> a,
                 const std::vector<torch::Tensor> _2q) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    make_signed_cuda(a[i], _2q[i]);
  }
}

void make_unsigned(std::vector<torch::Tensor> a,
                   const std::vector<torch::Tensor> _2q) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    make_unsigned_cuda(a[i], _2q[i]);
  }
}

std::vector<torch::Tensor> tile_unsigned(std::vector<torch::Tensor> a,
                                         const std::vector<torch::Tensor> _2q) {
  std::vector<torch::Tensor> outputs;

  const auto num_devices = _2q.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto result = tile_unsigned_cuda(a[i], _2q[i]);
    outputs.push_back(result);
  }
  return outputs;
}

TORCH_LIBRARY_FRAGMENT(tiberate_ntt_ops, m) {
  m.def(
      "mont_mult(Tensor[] a, Tensor[] b, Tensor[] ql, Tensor[] qh, "
      "Tensor[] kl, Tensor[] kh) -> Tensor[]");
  m.def(
      "mont_enter(Tensor[] a, Tensor[] Rs, Tensor[] ql, Tensor[] qh, "
      "Tensor[] kl, Tensor[] kh) -> ()");
  m.def(
      "mont_reduce(Tensor[] a, Tensor[] ql, Tensor[] qh, "
      "Tensor[] kl, Tensor[] kh) -> ()");
  m.def("mont_add(Tensor[] a, Tensor[] b, Tensor[] _2q) -> Tensor[]");
  m.def("mont_sub(Tensor[] a, Tensor[] b, Tensor[] _2q) -> Tensor[]");
  m.def("reduce_2q(Tensor[] a, Tensor[] _2q) -> ()");
  m.def("make_signed(Tensor[] a, Tensor[] _2q) -> ()");
  m.def("make_unsigned(Tensor[] a, Tensor[] _2q) -> ()");
  m.def("tile_unsigned(Tensor[] a, Tensor[] _2q) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(tiberate_ntt_ops, CUDA, m) {
  m.impl("mont_mult", &mont_mult);
  m.impl("mont_enter", &mont_enter);
  m.impl("mont_reduce", &mont_reduce);
  m.impl("mont_add", &mont_add);
  m.impl("mont_sub", &mont_sub);
  m.impl("reduce_2q", &reduce_2q);
  m.impl("make_signed", &make_signed);
  m.impl("make_unsigned", &make_unsigned);
  m.impl("tile_unsigned", &tile_unsigned);
}
