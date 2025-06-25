#pragma once
#include <torch/torch.h>

// Forward declaration.
void randround_cuda(const torch::Tensor input, torch::Tensor rand_bytes);
