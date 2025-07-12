#include "cuda/ntt_radix2_cuda.h"

//------------------------------------------------------------------
// Wrap functions for ntt transformation
//------------------------------------------------------------------

void ntt_radix2(std::vector<torch::Tensor> a,
                const std::vector<torch::Tensor> even,
                const std::vector<torch::Tensor> odd,
                const std::vector<torch::Tensor> psi,
                const int64_t sp_prime_len) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    ntt_radix2_cuda(a[i], even[i], odd[i], psi[i], sp_prime_len);
  }
}

void enter_ntt_radix2(std::vector<torch::Tensor> a,
                      const std::vector<torch::Tensor> even,
                      const std::vector<torch::Tensor> odd,
                      const std::vector<torch::Tensor> psi,
                      const int64_t sp_prime_len) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    enter_ntt_radix2_cuda(a[i], even[i], odd[i], psi[i], sp_prime_len);
  }
}

TORCH_LIBRARY_FRAGMENT(tiberate_ntt2_ops, m) {
  m.def(
      "ntt_radix2(Tensor[](a!) a, Tensor[] even, Tensor[] odd, Tensor[] "
      "psi, int sp_prime_len) -> ()");
  m.def(
      "enter_ntt_radix2(Tensor[](a!) a, Tensor[] even, Tensor[] "
      "odd, Tensor[] psi, int sp_prime_len) -> ()");
}

TORCH_LIBRARY_IMPL(tiberate_ntt2_ops, CUDA, m) {
  m.impl("ntt_radix2", &ntt_radix2);
  m.impl("enter_ntt_radix2", &enter_ntt_radix2);
}
