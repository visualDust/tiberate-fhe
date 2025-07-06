#pragma once

#include <torch/torch.h>

void upload_constants_cuda(torch::Tensor _2q,
                           torch::Tensor Rs,
                           torch::Tensor ql,
                           torch::Tensor qh,
                           torch::Tensor kl,
                           torch::Tensor kh);
std::vector<torch::Tensor> test_read_constants_2qRsQlQhKlKh(int device_id,
                                                            int count);
