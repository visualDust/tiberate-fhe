#pragma once
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdint>

constexpr size_t MAX_CONST_BYTES = 64 * 1024;  // 64KB

// Declaration only — no definition
extern __device__ __constant__ uint8_t constant_mem_pool[MAX_CONST_BYTES];
