#pragma once

#include "../extensions.h"

void randint_cuda(torch::Tensor rand_bytes, uint64_t *q);
torch::Tensor randint_fast_cuda(torch::Tensor states,
                                uint64_t *q,
                                int64_t shift,
                                size_t step);
