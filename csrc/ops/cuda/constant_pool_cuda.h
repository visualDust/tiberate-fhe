#pragma once
#include <torch/torch.h>

void upload_tensor_list(const std::vector<torch::Tensor> tensor_list,
                        const std::vector<int64_t> offset_list,
                        const int64_t layout,
                        const int64_t device_id);

torch::Tensor read_constant_chunk(const torch::Tensor& dummy,
                                  const int64_t offset_bytes,
                                  const int64_t count,
                                  const torch::Dtype dtype,
                                  const int64_t layout);
