#include <torch/script.h>
#include "cuda/ntt_cuda.h"  // inside is forward declarations for cuda kernels
#include "extensions.h"

//------------------------------------------------------------------
// Wrap functions for ntt transformation
//------------------------------------------------------------------

void ntt(std::vector<torch::Tensor> a,
         const std::vector<torch::Tensor> even,
         const std::vector<torch::Tensor> odd,
         const std::vector<torch::Tensor> psi,
         const std::vector<torch::Tensor> _2q,
         const std::vector<torch::Tensor> ql,
         const std::vector<torch::Tensor> qh,
         const std::vector<torch::Tensor> kl,
         const std::vector<torch::Tensor> kh) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    ntt_cuda(a[i], even[i], odd[i], psi[i], _2q[i], ql[i], qh[i], kl[i], kh[i]);
  }
}

void enter_ntt(std::vector<torch::Tensor> a,
               const std::vector<torch::Tensor> Rs,
               const std::vector<torch::Tensor> even,
               const std::vector<torch::Tensor> odd,
               const std::vector<torch::Tensor> psi,
               const std::vector<torch::Tensor> _2q,
               const std::vector<torch::Tensor> ql,
               const std::vector<torch::Tensor> qh,
               const std::vector<torch::Tensor> kl,
               const std::vector<torch::Tensor> kh) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    enter_ntt_cuda(a[i],
                   Rs[i],
                   even[i],
                   odd[i],
                   psi[i],
                   _2q[i],
                   ql[i],
                   qh[i],
                   kl[i],
                   kh[i]);
  }
}

void intt(std::vector<torch::Tensor> a,
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
    intt_cuda(a[i],
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

TORCH_LIBRARY_FRAGMENT(tiberate_ntt_ops, m) {
  m.def(
      "ntt(Tensor[] a, Tensor[] even, Tensor[] odd, Tensor[] psi, "
      "Tensor[] _2q, Tensor[] ql, Tensor[] qh, Tensor[] kl, "
      "Tensor[] kh) -> ()");
  m.def(
      "enter_ntt(Tensor[] a, Tensor[] Rs, Tensor[] even, Tensor[] odd, "
      "Tensor[] psi, Tensor[] _2q, Tensor[] ql, Tensor[] qh, "
      "Tensor[] kl, Tensor[] kh) -> ()");
  m.def(
      "intt(Tensor[] a, Tensor[] even, Tensor[] odd, Tensor[] psi, "
      "Tensor[] Ninv, Tensor[] _2q, Tensor[] ql, Tensor[] qh, "
      "Tensor[] kl, Tensor[] kh) -> ()");
  m.def(
      "intt_exit(Tensor[] a, Tensor[] even, Tensor[] odd, Tensor[] psi, "
      "Tensor[] Ninv, Tensor[] _2q, Tensor[] ql, Tensor[] qh, "
      "Tensor[] kl, Tensor[] kh) -> ()");
  m.def(
      "intt_exit_reduce(Tensor[] a, Tensor[] even, Tensor[] odd, "
      "Tensor[] psi, Tensor[] Ninv, Tensor[] _2q, Tensor[] ql, "
      "Tensor[] qh, Tensor[] kl, Tensor[] kh) -> ()");
  m.def(
      "intt_exit_reduce_signed(Tensor[] a, Tensor[] even, Tensor[] odd, "
      "Tensor[] psi, Tensor[] Ninv, Tensor[] _2q, Tensor[] ql, "
      "Tensor[] qh, Tensor[] kl, Tensor[] kh) -> ()");
}

TORCH_LIBRARY_IMPL(tiberate_ntt_ops, CUDA, m) {
  m.impl("ntt", &ntt);
  m.impl("enter_ntt", &enter_ntt);
  m.impl("intt", &intt);
  m.impl("intt_exit", &intt_exit);
  m.impl("intt_exit_reduce", &intt_exit_reduce);
  m.impl("intt_exit_reduce_signed", &intt_exit_reduce_signed);
}
