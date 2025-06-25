#include <torch/library.h>
#include <torch/torch.h>
#include "cuda/he_fused_cuda.h"

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

torch::Tensor switch_key_switch_later_part_extend(
    const int64_t rns_len,
    const torch::Tensor state,
    const torch::Tensor l_enter,
    const int64_t l_enter_start_offset,
    const torch::Tensor _2q,
    const torch::Tensor Rs,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh) {
  auto out = switch_key_switch_later_part_extend_cuda(
      rns_len, state, l_enter, l_enter_start_offset, _2q, Rs, ql, qh, kl, kh);
  return out;
}

std::vector<torch::Tensor> codec_rotate_make_unsigned_reduce_2q(
    const std::vector<torch::Tensor> a,
    const std::vector<torch::Tensor> perm,
    const std::vector<torch::Tensor> _2q) {
  const auto num_devices = a.size();
  std::vector<torch::Tensor> outputs;
  for (size_t i = 0; i < num_devices; ++i) {
    auto out = codec_rotate_make_unsigned_reduce_2q_cuda(a[i], perm[i], _2q[i]);
    outputs.push_back(out);
  }
  return outputs;
}

TORCH_LIBRARY_FRAGMENT(tiberate_fused_ops, m) {
  m.def(
      "pc_add_fused(Tensor[] ct_data, Tensor[] pt_data, "
      "Tensor[] Rs, Tensor[] ql, Tensor[] qh, "
      "Tensor[] kl, Tensor[] kh, Tensor[] _2q) -> Tensor[]");
  m.def(
      "switch_key_switch_later_part_extend(int rns_len, Tensor state, Tensor "
      "l_enter, int "
      "l_enter_start_offset, Tensor _2q, Tensor Rs, Tensor ql, Tensor "
      "qh, Tensor kl, Tensor kh) -> Tensor");
  m.def(
      "codec_rotate_make_unsigned_reduce_2q(Tensor[] a, Tensor[] perm, "
      "Tensor[] _2q) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(tiberate_fused_ops, CUDA, m) {
  m.impl("pc_add_fused", &pc_add_fused);
  m.impl("switch_key_switch_later_part_extend",
         &switch_key_switch_later_part_extend);
  m.impl("codec_rotate_make_unsigned_reduce_2q",
         &codec_rotate_make_unsigned_reduce_2q);
}
