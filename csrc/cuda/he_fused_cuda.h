#pragma once

#include "../extensions.h"

#define BLOCK_SIZE 256

torch::Tensor pc_add_fused_cuda(
    const torch::Tensor a,  // a is typically ct_data
    const torch::Tensor b,  // b is typically pt_data
    const torch::Tensor _2q,
    const torch::Tensor Rs,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh);

torch::Tensor switch_key_switch_later_part_extend_cuda(
    const int64_t rns_len,
    const torch::Tensor state,
    const torch::Tensor l_enter,
    const int64_t l_enter_start_offset,
    const torch::Tensor _2q,
    const torch::Tensor Rs,
    const torch::Tensor ql,
    const torch::Tensor qh,
    const torch::Tensor kl,
    const torch::Tensor kh);

torch::Tensor codec_rotate_make_unsigned_reduce_2q_cuda(
    const torch::Tensor a, const torch::Tensor perm, const torch::Tensor _2q);
