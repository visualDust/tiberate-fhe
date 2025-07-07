#pragma once

#include <torch/torch.h>

enum ConstantMemoryLayout { LeftToRight = 0, RightToLeft = 1 };

int upload_tensor_list_cuda(const std::vector<torch::Tensor>& tensors,
                            const std::vector<int64_t>& offsets,
                            ConstantMemoryLayout layout,
                            int device_id);

torch::Tensor read_constant_chunk_cuda(int device_id,
                                       size_t offset_bytes,
                                       size_t count,
                                       torch::Dtype dtype,
                                       ConstantMemoryLayout layout);
