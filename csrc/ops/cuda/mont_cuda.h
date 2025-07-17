#pragma once

#include <torch/torch.h>

torch::Tensor mont_mult_cuda(const torch::Tensor a,
                             const torch::Tensor b,
                             const int64_t sp_prime_len);

void mont_enter_scalar_cuda(torch::Tensor a,
                            const torch::Tensor b,
                            const int64_t sp_prime_len);

void mont_enter_Rs_cuda(torch::Tensor a, const int64_t sp_prime_len);

void mont_enter_Rs_scale_cuda(torch::Tensor a, const int64_t sp_prime_len);

void mont_enter_cuda(torch::Tensor a,
                     const torch::Tensor Rs,
                     const torch::Tensor ql,
                     const torch::Tensor qh,
                     const torch::Tensor kl,
                     const torch::Tensor kh);

void mont_reduce_cuda(torch::Tensor a, const int64_t sp_prime_len);

void reduce_2q_cuda(torch::Tensor a, const int64_t sp_prime_len);

torch::Tensor mont_add_cuda(const torch::Tensor a,
                            const torch::Tensor b,
                            const int64_t sp_prime_len);

// mont add legacy function
torch::Tensor mont_add_cuda(const torch::Tensor a,
                            const torch::Tensor b,
                            const torch::Tensor _2q);

torch::Tensor mont_sub_cuda(const torch::Tensor a,
                            const torch::Tensor b,
                            const int64_t sp_prime_len);

void make_signed_cuda(torch::Tensor a, const int64_t sp_prime_len);

void make_unsigned_cuda(torch::Tensor a, const int64_t sp_prime_len);

torch::Tensor tile_unsigned_cuda(torch::Tensor a, const torch::Tensor _2q);
