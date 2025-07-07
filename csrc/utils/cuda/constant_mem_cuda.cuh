#pragma once
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>
#include "constant_mem_cuda.h"

constexpr size_t MAX_CONST_BYTES = 64 * 1024;  // 64KB
constexpr size_t MAX_RNS_COUNT =
    128;  // typical value: 19 for logN15, 35 for logN16
constexpr size_t _2Q_BYTE_OFFSET = 0;
constexpr size_t RS_BYTE_OFFSET =
    _2Q_BYTE_OFFSET + MAX_RNS_COUNT * sizeof(uint64_t);
constexpr size_t QL_BYTE_OFFSET =
    RS_BYTE_OFFSET + MAX_RNS_COUNT * sizeof(uint64_t);
constexpr size_t QH_BYTE_OFFSET =
    QL_BYTE_OFFSET + MAX_RNS_COUNT * sizeof(uint64_t);
constexpr size_t KL_BYTE_OFFSET =
    QH_BYTE_OFFSET + MAX_RNS_COUNT * sizeof(uint64_t);
constexpr size_t KH_BYTE_OFFSET =
    KL_BYTE_OFFSET + MAX_RNS_COUNT * sizeof(uint64_t);
constexpr size_t NINV_BYTE_OFFSET =
    KH_BYTE_OFFSET + MAX_RNS_COUNT * sizeof(uint64_t);

static_assert(NINV_BYTE_OFFSET + MAX_RNS_COUNT * sizeof(uint64_t) <=
                  MAX_CONST_BYTES,
              "DEBUG: Constant memory offsets exceed maximum size");

// Global constant memory buffer (raw byte array)
__device__ __constant__ uint8_t constant_mem_pool[MAX_CONST_BYTES];

// Copy from host to constant memory (device-aware)
int copy_to_constant_memory(const void* src,
                            size_t size_bytes,
                            size_t offset_bytes,
                            int device_id,
                            cudaStream_t stream,
                            ConstantMemoryLayout layout);
