#pragma once

#include <torch/torch.h>

// ------------------------------------------------------------------
// CUDA forward declarations
// ------------------------------------------------------------------

int upload_tensor_list_cuda(const std::vector<torch::Tensor> tensors,
                            const std::vector<int64_t> offsets,
                            const int64_t layout_int64_t,
                            const int64_t device_id_int64_t);

torch::Tensor read_constant_chunk_cuda(const int64_t device_id_int64_t,
                                       const int64_t offset_bytes_int64_t,
                                       const int64_t count_int64_t,
                                       const torch::Dtype dtype,
                                       const int64_t layout_int64_t);

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
