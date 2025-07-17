#pragma once

#include <torch/torch.h>

// -------------------------------------------------------------------
// forward definitions
// -------------------------------------------------------------------

torch::Tensor mont_add_many_3d_cuda(const torch::Tensor input,
                                    const int64_t sp_prime_len);

torch::Tensor mont_reduce_add_many_3d_cuda(const torch::Tensor input,
                                           const int64_t sp_prime_len);

torch::Tensor mont_add_reduce_2q_cuda(const torch::Tensor a,
                                      const torch::Tensor b,
                                      const int64_t sp_prime_len);

torch::Tensor mont_sub_reduce_2q_cuda(const torch::Tensor a,
                                      const torch::Tensor b,
                                      const int64_t sp_prime_len);

torch::Tensor mont_enter_scalar_reduce_2q_cuda(torch::Tensor a,
                                               const torch::Tensor Rs,
                                               const int64_t sp_prime_len);
