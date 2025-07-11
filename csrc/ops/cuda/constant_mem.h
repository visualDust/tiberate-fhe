#ifndef SHARED_CONSTANTS_H
#define SHARED_CONSTANTS_H
#include <stdint.h>

#define MAX_CONST_BYTES (64 * 1024)
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
#define CONST_MEM_REGION_LEN 128
