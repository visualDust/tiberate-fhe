#pragma once

#include <torch/torch.h>

// -------------------------------------------------------------------
// forward definitions
// -------------------------------------------------------------------

void intt_radix2_cuda(torch::Tensor a,
                      const torch::Tensor even,
                      const torch::Tensor odd,
                      const torch::Tensor psi,
                      const int64_t sp_prime_len);

void intt_radix2_exit_cuda(torch::Tensor a,
                           const torch::Tensor even,
                           const torch::Tensor odd,
                           const torch::Tensor psi,
                           const int64_t sp_prime_len);

void intt_radix2_exit_reduce_cuda(torch::Tensor a,
                                  const torch::Tensor even,
                                  const torch::Tensor odd,
                                  const torch::Tensor psi,
                                  const int64_t sp_prime_len);

void intt_radix2_exit_reduce_signed_cuda(torch::Tensor a,
                                         const torch::Tensor even,
                                         const torch::Tensor odd,
                                         const torch::Tensor psi,
                                         const int64_t sp_prime_len);
