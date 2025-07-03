#pragma once

#include <torch/torch.h>

torch::Tensor mont_mult_sum_many_3d_cuda(const torch::Tensor a,
                                         const torch::Tensor b,
                                         const torch::Tensor _2q,
                                         const torch::Tensor ql,
                                         const torch::Tensor qh,
                                         const torch::Tensor kl,
                                         const torch::Tensor kh);
