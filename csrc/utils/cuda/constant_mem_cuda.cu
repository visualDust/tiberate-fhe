#include "constant_mem_cuda.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "constant_mem_cuda.h"

int copy_to_constant_memory(const void* src,
                            size_t size_bytes,
                            size_t offset_bytes,
                            int device_id,
                            cudaStream_t stream) {
  if (offset_bytes + size_bytes > MAX_CONST_BYTES) return -1;

  cudaSetDevice(device_id);
  cudaMemcpyToSymbolAsync(constant_mem_pool,
                          src,
                          size_bytes,
                          offset_bytes,
                          cudaMemcpyHostToDevice,
                          stream);
  return 0;
}

template <typename T>
void upload_constants_cuda_typed(torch::Tensor _2q,
                                 torch::Tensor Rs,
                                 torch::Tensor ql,
                                 torch::Tensor qh,
                                 torch::Tensor kl,
                                 torch::Tensor kh) {
  TORCH_CHECK(_2q.is_contiguous() && Rs.is_contiguous());
  TORCH_CHECK(ql.is_contiguous() && qh.is_contiguous());
  TORCH_CHECK(kl.is_contiguous() && kh.is_contiguous());

  int device_id =
      ql.device().index();  // Assuming all tensors are on the same device
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  const size_t size_bytes = ql.numel() * sizeof(T);
  TORCH_CHECK(ql.numel() == qh.numel() && ql.numel() == kl.numel() &&
                  ql.numel() == kh.numel() && _2q.numel() == Rs.numel(),
              "All tensors must have the same number of elements");

  copy_to_constant_memory(_2q.data_ptr(),
                          size_bytes,
                          _2Q_OFFSET * sizeof(T),
                          device_id,
                          stream.stream());
  copy_to_constant_memory(Rs.data_ptr(),
                          size_bytes,
                          RS_OFFSET * sizeof(T),
                          device_id,
                          stream.stream());
  copy_to_constant_memory(ql.data_ptr(),
                          size_bytes,
                          QL_OFFSET * sizeof(T),
                          device_id,
                          stream.stream());
  copy_to_constant_memory(qh.data_ptr(),
                          size_bytes,
                          QH_OFFSET * sizeof(T),
                          device_id,
                          stream.stream());
  copy_to_constant_memory(kl.data_ptr(),
                          size_bytes,
                          KL_OFFSET * sizeof(T),
                          device_id,
                          stream.stream());
  copy_to_constant_memory(kh.data_ptr(),
                          size_bytes,
                          KH_OFFSET * sizeof(T),
                          device_id,
                          stream.stream());
}

void upload_constants_cuda(torch::Tensor _2q,
                           torch::Tensor Rs,
                           torch::Tensor ql,
                           torch::Tensor qh,
                           torch::Tensor kl,
                           torch::Tensor kh) {
  AT_DISPATCH_INTEGRAL_TYPES(ql.scalar_type(), "upload_constants", ([&] {
                               upload_constants_cuda_typed<scalar_t>(
                                   _2q, Rs, ql, qh, kl, kh);
                             }));
}
