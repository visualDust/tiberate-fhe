#pragma once

#include "../extensions.h"

// ------------------------------------------------------------------
// CUDA forward declarations
// ------------------------------------------------------------------

void ntt_cuda(torch::Tensor a,
              const torch::Tensor even,
              const torch::Tensor odd,
              const torch::Tensor psi,
              const torch::Tensor _2q,
              const torch::Tensor ql,
              const torch::Tensor qh,
              const torch::Tensor kl,
              const torch::Tensor kh);

void enter_ntt_cuda(torch::Tensor a,
                    const torch::Tensor Rs,
                    const torch::Tensor even,
                    const torch::Tensor odd,
                    const torch::Tensor psi,
                    const torch::Tensor _2q,
                    const torch::Tensor ql,
                    const torch::Tensor qh,
                    const torch::Tensor kl,
                    const torch::Tensor kh);

void intt_cuda(torch::Tensor a,
               const torch::Tensor even,
               const torch::Tensor odd,
               const torch::Tensor psi,
               const torch::Tensor Ninv,
               const torch::Tensor _2q,
               const torch::Tensor ql,
               const torch::Tensor qh,
               const torch::Tensor kl,
               const torch::Tensor kh);

void intt_exit_cuda(torch::Tensor a,
                    const torch::Tensor even,
                    const torch::Tensor odd,
                    const torch::Tensor psi,
                    const torch::Tensor Ninv,
                    const torch::Tensor _2q,
                    const torch::Tensor ql,
                    const torch::Tensor qh,
                    const torch::Tensor kl,
                    const torch::Tensor kh);

void intt_exit_reduce_cuda(torch::Tensor a,
                           const torch::Tensor even,
                           const torch::Tensor odd,
                           const torch::Tensor psi,
                           const torch::Tensor Ninv,
                           const torch::Tensor _2q,
                           const torch::Tensor ql,
                           const torch::Tensor qh,
                           const torch::Tensor kl,
                           const torch::Tensor kh);

void intt_exit_reduce_signed_cuda(torch::Tensor a,
                                  const torch::Tensor even,
                                  const torch::Tensor odd,
                                  const torch::Tensor psi,
                                  const torch::Tensor Ninv,
                                  const torch::Tensor _2q,
                                  const torch::Tensor ql,
                                  const torch::Tensor qh,
                                  const torch::Tensor kl,
                                  const torch::Tensor kh);
