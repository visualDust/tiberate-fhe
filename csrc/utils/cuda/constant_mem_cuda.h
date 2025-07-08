#pragma once

#include <torch/torch.h>

enum ConstantMemoryGravity { Left = 0, Right = 1 };

// ===================================================================
// _2q, Rs, ql, qh, kl, kh, Ninv : static context layout
// see tiberate/context/constant_mem_context.py
// ===================================================================

constexpr int _2Q_CONST_IDX = 0;
constexpr int RS_CONST_IDX = 1;
constexpr int QL_CONST_IDX = 2;
constexpr int QH_CONST_IDX = 3;
constexpr int KL_CONST_IDX = 4;
constexpr int KH_CONST_IDX = 5;
constexpr int NINV_CONST_IDX = 6;
constexpr int CONST_MEM_REGION_LEN =
    128;  // 128 elements per region with dtype int64_t/int32_t
/**
if gravity is left:
- _2q is at constant_mem_pool[0:128]*sizeof(scalar_t)
- Rs is at constant_mem_pool[128:256]*sizeof(scalar_t)
- ql is at constant_mem_pool[256:384]*sizeof(scalar_t)
- qh is at constant_mem_pool[384:512]*sizeof(scalar_t)
- kl is at constant_mem_pool[512:640]*sizeof(scalar_t)
- kh is at constant_mem_pool[640:768]*sizeof(scalar_t)
- Ninv is at constant_mem_pool[768:896]*sizeof(scalar_t)
*/

int upload_tensor_list_cuda(const std::vector<torch::Tensor>& tensors,
                            const std::vector<int64_t>& offsets,
                            ConstantMemoryGravity layout,
                            int device_id);

torch::Tensor read_constant_chunk_cuda(int device_id,
                                       size_t offset_bytes,
                                       size_t count,
                                       torch::Dtype dtype,
                                       ConstantMemoryGravity layout);
