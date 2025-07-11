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

// ------------------------------------------------------------------
// typings
// ------------------------------------------------------------------

#define makeAcc32Restrict(tensor, scalar_t, dim) \
  tensor.packed_accessor32<scalar_t, dim, torch::RestrictPtrTraits>()

#define makeAcc32(tensor, scalar_t, dim) \
  tensor.packed_accessor32<scalar_t, dim>()

template <typename scalar_t, int dim>
using TensorAcc32Restrict =
    torch::PackedTensorAccessor32<scalar_t, dim, torch::RestrictPtrTraits>;

template <typename scalar_t, int dim>
using TensorAcc32 = torch::PackedTensorAccessor32<scalar_t, dim>;

// ------------------------------------------------------------------
// constant memory layout
// ------------------------------------------------------------------

enum ConstantMemoryGravity { Left = 0, Right = 1 };

// ===================================================================
// _2q, Rs, ql, qh, kl, kh, Ninv : static context layout
// see tiberate/context/constant_mem_context.py
// ===================================================================

#define _2Q_CONST_IDX 0
#define RS_CONST_IDX 1
#define QL_CONST_IDX 2
#define QH_CONST_IDX 3
#define KL_CONST_IDX 4
#define KH_CONST_IDX 5
#define NINV_CONST_IDX 6
// 128 elements per region with dtype int64_t/int32_t
#define CONST_MEM_REGION_LEN 128
