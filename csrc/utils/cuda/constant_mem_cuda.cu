#include "constant_mem_cuda.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

constexpr size_t MAX_CONST_BYTES = 64 * 1024;  // 64KB

__device__ __constant__ uint8_t constant_mem_pool[MAX_CONST_BYTES];

int upload_tensor_list_cuda(const std::vector<torch::Tensor>& tensors,
                            const std::vector<int64_t>& offsets,
                            ConstantMemoryGravity layout,
                            int device_id) {
  cudaSetDevice(device_id);
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto& t = tensors[i];
    TORCH_CHECK(t.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(tensors.size() == offsets.size(),
                "Mismatch: tensor list and offset list must have same length");
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    const size_t size_bytes = t.nbytes();
    size_t effective_offset = offsets[i];

    if (layout == Right) {
      effective_offset = MAX_CONST_BYTES - offsets[i] - size_bytes;
    }

    TORCH_CHECK(effective_offset + size_bytes <= MAX_CONST_BYTES,
                "Upload exceeds constant memory");
    printf("upload: constant_mem_pool address: %p\n", constant_mem_pool);

    cudaMemcpyToSymbolAsync(constant_mem_pool,
                            t.data_ptr(),
                            size_bytes,
                            effective_offset,
                            cudaMemcpyHostToDevice,
                            stream);
  }
  return 0;
}

template <typename scalar_t>
__global__ void read_chunk_kernel(scalar_t* out_ptr,
                                  size_t offset_bytes,
                                  size_t count,
                                  ConstantMemoryGravity layout) {
  int idx = threadIdx.x;
  if (idx >= count) return;

  size_t effective_offset = offset_bytes;
  if (layout == Right) {
    effective_offset =
        MAX_CONST_BYTES - offset_bytes - count * sizeof(scalar_t);
  }

  const scalar_t* const_mem =
      reinterpret_cast<const scalar_t*>(&constant_mem_pool[effective_offset]);
  out_ptr[idx] = const_mem[idx];
}

torch::Tensor read_constant_chunk_cuda(int device_id,
                                       size_t offset_bytes,
                                       size_t count,
                                       torch::Dtype dtype,
                                       ConstantMemoryGravity layout) {
  at::cuda::set_device(device_id);

  auto options =
      torch::TensorOptions().device(torch::kCUDA, device_id).dtype(dtype);
  torch::Tensor out = torch::empty({static_cast<int64_t>(count)}, options);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);
  printf("read: constant_mem_pool address: %p\n", constant_mem_pool);

  AT_DISPATCH_INTEGRAL_TYPES(
      dtype, "read_constant_chunk_cuda", ([&] {
        read_chunk_kernel<scalar_t><<<1, count, 0, stream.stream()>>>(
            out.data_ptr<scalar_t>(), offset_bytes, count, layout);
      }));

  return out;
}
