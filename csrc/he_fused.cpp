#include <torch/library.h>
#include "cuda/he_fused_cuda.h"
#include "extensions.h"

torch::Tensor switch_key_switch_later_part_extend(
    const int64_t rns_len,
    const torch::Tensor state,
    const torch::Tensor l_enter,
    const int64_t l_enter_start_offset,
    const std::vector<torch::Tensor> _2q,
    const std::vector<torch::Tensor> Rs,
    const std::vector<torch::Tensor> ql,
    const std::vector<torch::Tensor> qh,
    const std::vector<torch::Tensor> kl,
    const std::vector<torch::Tensor> kh) {
  auto out = switch_key_switch_later_part_extend_cuda(rns_len,
                                                      state,
                                                      l_enter,
                                                      l_enter_start_offset,
                                                      _2q[0],
                                                      Rs[0],
                                                      ql[0],
                                                      qh[0],
                                                      kl[0],
                                                      kh[0]);
  return out;
}

TORCH_LIBRARY_FRAGMENT(tiberate_fused_ops, m) {
  m.def(
      "switch_key_switch_later_part_extend(int rns_len, Tensor state, Tensor "
      "l_enter, int "
      "l_enter_start_offset, Tensor[] _2q, Tensor[] Rs, Tensor[] ql, Tensor[] "
      "qh, Tensor[] kl, Tensor[] kh) -> Tensor");
}

TORCH_LIBRARY_IMPL(tiberate_fused_ops, CUDA, m) {
  m.impl("switch_key_switch_later_part_extend",
         &switch_key_switch_later_part_extend);
}
