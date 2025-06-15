#pragma once

#include "../extensions.h"

#define BLOCK_SIZE 256

// -------------------------------------------------------------------
// forward definitions
// -------------------------------------------------------------------

torch::Tensor mont_add_reduce_2q_cuda(const torch::Tensor a,
                                      const torch::Tensor b,
                                      const torch::Tensor _2q);

torch::Tensor mont_sub_reduce_2q_cuda(const torch::Tensor a,
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

void rescale_exact_rounding_fused_cuda(
    torch::Tensor a,  // inplace of a
    const torch::Tensor Rs,
    const torch::Tensor rescaler,  // rescaler0
    const int64_t round_at,        // round_at
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh);

void rescale_non_exact_rounding_fused_cuda(
    torch::Tensor a,  // inplace of a
    const torch::Tensor Rs,
    const torch::Tensor rescaler,  // rescaler0
    const torch::Tensor _2q,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh);

// torch::Tensor codec_rotate_make_unsigned_reduce_2q_cuda(
//     const torch::Tensor in, const torch::Tensor perm, const torch::Tensor
//     _2q);
