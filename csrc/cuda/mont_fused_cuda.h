#pragma once

#include "../extensions.h"

torch::Tensor mont_add_reduce_2q_cuda(const torch::Tensor a,
                                      const torch::Tensor b,
                                      const torch::Tensor _2q);

torch::Tensor pc_add_fused_cuda(const torch::Tensor a,  // a should be ct_data
                                const torch::Tensor b,  // b should be pt_data
                                const torch::Tensor _2q,
                                const torch::Tensor Rs,
                                const torch::Tensor ql,
                                const torch::Tensor qh,
                                const torch::Tensor kl,
                                const torch::Tensor kh);
