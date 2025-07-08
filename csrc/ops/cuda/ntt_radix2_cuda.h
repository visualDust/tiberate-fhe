#pragma once

#include <torch/torch.h>

// ------------------------------------------------------------------
// CUDA forward declarations
// ------------------------------------------------------------------

void ntt_radix2_cuda(torch::Tensor a,
                     const torch::Tensor even,
                     const torch::Tensor odd,
                     const torch::Tensor psi,
                     const torch::Tensor _2q,
                     const torch::Tensor ql,
                     const torch::Tensor qh,
                     const torch::Tensor kl,
                     const torch::Tensor kh,
                     const int64_t prime_len);

void enter_ntt_radix2_cuda(torch::Tensor a,
                           const torch::Tensor Rs,
                           const torch::Tensor even,
                           const torch::Tensor odd,
                           const torch::Tensor psi,
                           const torch::Tensor _2q,
                           const torch::Tensor ql,
                           const torch::Tensor qh,
                           const torch::Tensor kl,
                           const torch::Tensor kh,
                           const int64_t prime_len);
