#include "cuda/ntt_fused_cuda.h"  // inside is forward declarations for cuda kernels
#include "extensions.h"

void intt_exit(std::vector<torch::Tensor> a,
               const std::vector<torch::Tensor> even,
               const std::vector<torch::Tensor> odd,
               const std::vector<torch::Tensor> psi,
               const std::vector<torch::Tensor> Ninv,
               const std::vector<torch::Tensor> _2q,
               const std::vector<torch::Tensor> ql,
               const std::vector<torch::Tensor> qh,
               const std::vector<torch::Tensor> kl,
               const std::vector<torch::Tensor> kh) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    intt_exit_cuda(a[i],
                   even[i],
                   odd[i],
                   psi[i],
                   Ninv[i],
                   _2q[i],
                   ql[i],
                   qh[i],
                   kl[i],
                   kh[i]);
  }
}

void intt_exit_reduce(std::vector<torch::Tensor> a,
                      const std::vector<torch::Tensor> even,
                      const std::vector<torch::Tensor> odd,
                      const std::vector<torch::Tensor> psi,
                      const std::vector<torch::Tensor> Ninv,
                      const std::vector<torch::Tensor> _2q,
                      const std::vector<torch::Tensor> ql,
                      const std::vector<torch::Tensor> qh,
                      const std::vector<torch::Tensor> kl,
                      const std::vector<torch::Tensor> kh) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    intt_exit_reduce_cuda(a[i],
                          even[i],
                          odd[i],
                          psi[i],
                          Ninv[i],
                          _2q[i],
                          ql[i],
                          qh[i],
                          kl[i],
                          kh[i]);
  }
}

void intt_exit_reduce_signed(std::vector<torch::Tensor> a,
                             const std::vector<torch::Tensor> even,
                             const std::vector<torch::Tensor> odd,
                             const std::vector<torch::Tensor> psi,
                             const std::vector<torch::Tensor> Ninv,
                             const std::vector<torch::Tensor> _2q,
                             const std::vector<torch::Tensor> ql,
                             const std::vector<torch::Tensor> qh,
                             const std::vector<torch::Tensor> kl,
                             const std::vector<torch::Tensor> kh) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    intt_exit_reduce_signed_cuda(a[i],
                                 even[i],
                                 odd[i],
                                 psi[i],
                                 Ninv[i],
                                 _2q[i],
                                 ql[i],
                                 qh[i],
                                 kl[i],
                                 kh[i]);
  }
}

TORCH_LIBRARY_FRAGMENT(tiberate_fused_ops, m) {
  m.def(
      "intt_exit(Tensor[](a!) a, Tensor[] even, Tensor[] odd, Tensor[] psi, "
      "Tensor[] Ninv, Tensor[] _2q, Tensor[] ql, Tensor[] qh, "
      "Tensor[] kl, Tensor[] kh) -> ()");
  m.def(
      "intt_exit_reduce(Tensor[](a!) a, Tensor[] even, Tensor[] odd, "
      "Tensor[] psi, Tensor[] Ninv, Tensor[] _2q, Tensor[] ql, "
      "Tensor[] qh, Tensor[] kl, Tensor[] kh) -> ()");
  m.def(
      "intt_exit_reduce_signed(Tensor[](a!) a, Tensor[] even, Tensor[] odd, "
      "Tensor[] psi, Tensor[] Ninv, Tensor[] _2q, Tensor[] ql, "
      "Tensor[] qh, Tensor[] kl, Tensor[] kh) -> ()");
}

TORCH_LIBRARY_IMPL(tiberate_fused_ops, CUDA, m) {
  m.impl("intt_exit", &intt_exit);
  m.impl("intt_exit_reduce", &intt_exit_reduce);
  m.impl("intt_exit_reduce_signed", &intt_exit_reduce_signed);
}
