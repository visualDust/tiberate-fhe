#pragma once

#include "../extensions.h"

torch::Tensor mont_mult_cuda(const torch::Tensor a,
                             const torch::Tensor b,
                             const torch::Tensor ql,
                             const torch::Tensor qh,
                             const torch::Tensor kl,
                             const torch::Tensor kh);

void mont_enter_cuda(torch::Tensor a,
                     const torch::Tensor Rs,
                     const torch::Tensor ql,
                     const torch::Tensor qh,
                     const torch::Tensor kl,
                     const torch::Tensor kh);

void reduce_2q_cuda(torch::Tensor a, const torch::Tensor _2q);

void mont_reduce_cuda(torch::Tensor a,
                      const torch::Tensor ql,
                      const torch::Tensor qh,
                      const torch::Tensor kl,
                      const torch::Tensor kh);

torch::Tensor mont_add_cuda(const torch::Tensor a,
                            const torch::Tensor b,
                            const torch::Tensor _2q);

torch::Tensor mont_sub_cuda(const torch::Tensor a,
                            const torch::Tensor b,
                            const torch::Tensor _2q);

void make_signed_cuda(torch::Tensor a, const torch::Tensor _2q);

void make_unsigned_cuda(torch::Tensor a, const torch::Tensor _2q);

torch::Tensor tile_unsigned_cuda(torch::Tensor a, const torch::Tensor _2q);
