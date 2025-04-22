#pragma once

#include "../extensions.h"

#define GE(x_high, x_low, y_high, y_low) \
  (((x_high) > (y_high)) | (((x_high) == (y_high)) & ((x_low) >= (y_low))))

#define COMBINE_TWO(high, low) \
  ((static_cast<uint64_t>(high) << 32) | static_cast<uint64_t>(low))

// Forward declaration.
void discrete_gaussian_cuda(torch::Tensor rand_bytes,
                            uint64_t *btree,
                            int btree_size,
                            int depth);

torch::Tensor discrete_gaussian_fast_cuda(torch::Tensor states,
                                          uint64_t *btree,
                                          int btree_size,
                                          int depth,
                                          size_t step);
