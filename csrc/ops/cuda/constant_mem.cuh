#pragma once
#ifndef SHARED_CONSTANTS_H
#define SHARED_CONSTANTS_H
#include <stdint.h>

#define MAX_CONST_BYTES (4 * 1024)  // at most 64* 1024, now  only need 4K
extern __device__ __constant__ uint8_t constant_mem_pool[MAX_CONST_BYTES];
#endif  // SHARED_CONSTANTS_H

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
#define CONST_MEM_REGION_LEN 64

// regions cannot exceed 64K bytes, assume they are at most int64_t
static_assert(CONST_MEM_REGION_LEN * (NINV_CONST_IDX + 1) * sizeof(int64_t) <=
                  MAX_CONST_BYTES,
              "Constant memory regions exceed maximum allowed size.");

template <typename scalar_t>
__device__ __forceinline__ const scalar_t* get_const_ptr_gright(
    const int region_idx, const int idx) {
  auto region = reinterpret_cast<const scalar_t*>(
      &constant_mem_pool[MAX_CONST_BYTES - (region_idx * CONST_MEM_REGION_LEN) *
                                               sizeof(scalar_t)]);
  return &region[idx];
}
