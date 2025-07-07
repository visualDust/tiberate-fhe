#pragma once

#include <torch/torch.h>

enum ConstantMemoryLayout { LeftToRight = 0, RightToLeft = 1 };

void upload_constants_cuda(torch::Tensor _2q,
                           torch::Tensor Rs,
                           torch::Tensor ql,
                           torch::Tensor qh,
                           torch::Tensor kl,
                           torch::Tensor kh,
                           torch::Tensor Ninv,
                           ConstantMemoryLayout layout);

std::vector<torch::Tensor> test_read_constants_2qRsQlQhKlKh(int device_id,
                                                            int count,
                                                            int layout);
