#include "cuda/intt_radix2_cuda.h"  // inside is forward declarations for cuda kernels

void intt_radix2(std::vector<torch::Tensor> a,
                 const std::vector<torch::Tensor> even,
                 const std::vector<torch::Tensor> odd,
                 const std::vector<torch::Tensor> psi,
                 const int64_t sp_prime_len) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    intt_radix2_cuda(a[i], even[i], odd[i], psi[i], sp_prime_len);
  }
}

void intt_radix2_exit(std::vector<torch::Tensor> a,
                      const std::vector<torch::Tensor> even,
                      const std::vector<torch::Tensor> odd,
                      const std::vector<torch::Tensor> psi,
                      const int64_t sp_prime_len) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    intt_radix2_exit_cuda(a[i], even[i], odd[i], psi[i], sp_prime_len);
  }
}

void intt_radix2_exit_reduce(std::vector<torch::Tensor> a,
                             const std::vector<torch::Tensor> even,
                             const std::vector<torch::Tensor> odd,
                             const std::vector<torch::Tensor> psi,
                             const int64_t sp_prime_len) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    intt_radix2_exit_reduce_cuda(a[i], even[i], odd[i], psi[i], sp_prime_len);
  }
}

void intt_radix2_exit_reduce_signed(std::vector<torch::Tensor> a,
                                    const std::vector<torch::Tensor> even,
                                    const std::vector<torch::Tensor> odd,
                                    const std::vector<torch::Tensor> psi,
                                    const int64_t sp_prime_len) {
  const auto num_devices = a.size();
  for (size_t i = 0; i < num_devices; ++i) {
    intt_radix2_exit_reduce_signed_cuda(
        a[i], even[i], odd[i], psi[i], sp_prime_len);
  }
}

TORCH_LIBRARY_FRAGMENT(tiberate_ntt2_ops, m) {
  m.def(
      "intt_radix2(Tensor[](a!) a, Tensor[] even, Tensor[] odd, Tensor[] psi, "
      "int sp_prime_len) -> ()");
  m.def(
      "intt_radix2_exit(Tensor[](a!) a, Tensor[] even, Tensor[] odd, Tensor[] "
      "psi, int sp_prime_len) -> ()");
  m.def(
      "intt_radix2_exit_reduce(Tensor[](a!) a, Tensor[] even, Tensor[] odd, "
      "Tensor[] psi, int sp_prime_len) -> ()");
  m.def(
      "intt_radix2_exit_reduce_signed(Tensor[](a!) a, Tensor[] even, Tensor[] "
      "odd, Tensor[] psi, int sp_prime_len) -> ()");
}

TORCH_LIBRARY_IMPL(tiberate_ntt2_ops, CUDA, m) {
  m.impl("intt_radix2", &intt_radix2);
  m.impl("intt_radix2_exit", &intt_radix2_exit);
  m.impl("intt_radix2_exit_reduce", &intt_radix2_exit_reduce);
  m.impl("intt_radix2_exit_reduce_signed", &intt_radix2_exit_reduce_signed);
}
