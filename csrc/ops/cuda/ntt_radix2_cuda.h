#pragma once

#include <torch/torch.h>

// ------------------------------------------------------------------
// CUDA forward declarations
// ------------------------------------------------------------------

void ntt_radix2_cuda(torch::Tensor a,
                     const torch::Tensor even,
                     const torch::Tensor odd,
                     const torch::Tensor psi,
                     const int64_t sp_prime_len);

void enter_ntt_radix2_cuda(torch::Tensor a,
                           //    const torch::Tensor Rs,
                           const torch::Tensor even,
                           const torch::Tensor odd,
                           const torch::Tensor psi,
                           const int64_t sp_prime_len);
