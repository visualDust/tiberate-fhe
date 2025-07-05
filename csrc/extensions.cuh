// ------------------------------------------------------------------
// macros and type definitions
// ------------------------------------------------------------------

#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#define BLOCK_SIZE 256
#define CHUNK_SIZE \
  4  // keep it safe for shared memory, each thread will use 16 *
     // sizeof(scalar_t) bytes

#define makeAcc32Restrict(tensor, scalar_t, dim) \
  tensor.packed_accessor32<scalar_t, dim, torch::RestrictPtrTraits>()

#define makeAcc32(tensor, scalar_t, dim) \
  tensor.packed_accessor32<scalar_t, dim>()

template <typename scalar_t, int dim>
using TensorAcc32Restrict =
    torch::PackedTensorAccessor32<scalar_t, dim, torch::RestrictPtrTraits>;

template <typename scalar_t, int dim>
using TensorAcc32 = torch::PackedTensorAccessor32<scalar_t, dim>;
