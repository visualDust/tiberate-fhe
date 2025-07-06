#include "constant_mem_cuda.cuh"
#include <ATen/cuda/CUDAContext.h>
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

  copy_to_constant_memory(
      _2q.data_ptr(), size_bytes, _2Q_BYTE_OFFSET, device_id, stream.stream());
  copy_to_constant_memory(
      Rs.data_ptr(), size_bytes, RS_BYTE_OFFSET, device_id, stream.stream());
  copy_to_constant_memory(
      ql.data_ptr(), size_bytes, QL_BYTE_OFFSET, device_id, stream.stream());
  copy_to_constant_memory(
      qh.data_ptr(), size_bytes, QH_BYTE_OFFSET, device_id, stream.stream());
  copy_to_constant_memory(
      kl.data_ptr(), size_bytes, KL_BYTE_OFFSET, device_id, stream.stream());
  copy_to_constant_memory(
      kh.data_ptr(), size_bytes, KH_BYTE_OFFSET, device_id, stream.stream());
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

// Kernel: read a fixed-length int64 array from constant memory
__global__ void test_read_constant_kernel(int64_t* out_ptr,
                                          size_t offset_bytes,
                                          size_t count) {
  int idx = threadIdx.x;
  if (idx < count) {
    const int64_t* const_mem =
        reinterpret_cast<const int64_t*>(&constant_mem_pool[offset_bytes]);
    out_ptr[idx] = const_mem[idx];
  }
}

// Read one chunk
torch::Tensor test_read_constant_chunk(int device_id,
                                       size_t offset_bytes,
                                       size_t count) {
  at::cuda::set_device(device_id);
  auto options = torch::TensorOptions()
                     .dtype(torch::kInt64)
                     .device(torch::kCUDA, device_id);
  torch::Tensor out = torch::empty({static_cast<int64_t>(count)}, options);

  const int threads = static_cast<int>(count);
  const int blocks = 1;
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  test_read_constant_kernel<<<blocks, threads, 0, stream.stream()>>>(
      out.data_ptr<int64_t>(), offset_bytes, count);

  return out;
}

// Read all 6 constants: _2q, Rs, ql, qh, kl, kh
std::vector<torch::Tensor> test_read_constants_2qRsQlQhKlKh(int device_id,
                                                            int count) {
  std::vector<torch::Tensor> result;
  result.reserve(6);
  result.push_back(test_read_constant_chunk(device_id, _2Q_BYTE_OFFSET, count));
  result.push_back(test_read_constant_chunk(device_id, RS_BYTE_OFFSET, count));
  result.push_back(test_read_constant_chunk(device_id, QL_BYTE_OFFSET, count));
  result.push_back(test_read_constant_chunk(device_id, QH_BYTE_OFFSET, count));
  result.push_back(test_read_constant_chunk(device_id, KL_BYTE_OFFSET, count));
  result.push_back(test_read_constant_chunk(device_id, KH_BYTE_OFFSET, count));
  return result;
}
