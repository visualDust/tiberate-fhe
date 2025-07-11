#include "../../extensions.cuh"
#include "constant_mem.h"

// ------------------------------------------------------------------
// constant memory pool
// ------------------------------------------------------------------

__device__ __constant__ uint8_t constant_mem_pool[MAX_CONST_BYTES];

int upload_tensor_list_cuda(const std::vector<torch::Tensor> tensors,
                            const std::vector<int64_t> offsets,
                            const int64_t layout_int64_t,
                            const int64_t device_id_int64_t) {
  auto layout = static_cast<ConstantMemoryGravity>(layout_int64_t);
  auto device_id = static_cast<int>(device_id_int64_t);
  cudaSetDevice(device_id);
  TORCH_CHECK(tensors.size() == offsets.size(),
              "Mismatch: tensor list and offset list must have same length");
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto& t = tensors[i];
    TORCH_CHECK(t.is_contiguous(), "Tensors must be contiguous");
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    const size_t size_bytes = t.nbytes();
    size_t effective_offset = offsets[i];

    if (layout == Right) {
      effective_offset = MAX_CONST_BYTES - offsets[i] - size_bytes;
    }

    TORCH_CHECK(effective_offset + size_bytes <= MAX_CONST_BYTES,
                "Upload exceeds constant memory");

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
                                  const size_t offset_bytes,
                                  const size_t count,
                                  const ConstantMemoryGravity layout) {
  int idx = threadIdx.x;
  if (idx >= count) return;

  size_t effective_offset = offset_bytes;
  if (layout == Right) {
    effective_offset =
        MAX_CONST_BYTES - offset_bytes - count * sizeof(scalar_t);
  }

  const scalar_t* const_mem =
      reinterpret_cast<const scalar_t*>(&constant_mem_pool[effective_offset]);

#ifdef DEBUG_KERNEL_OUTPUT
  if (idx == 0 || idx == CONST_MEM_REGION_LEN) {
    // Debugging output.
    printf("read constant[%d]: %p : %ld\n", idx, const_mem, const_mem[idx]);
  }
#endif
  out_ptr[idx] = const_mem[idx];
}

torch::Tensor read_constant_chunk_cuda(const int64_t device_id_int64_t,
                                       const int64_t offset_bytes_int64_t,
                                       const int64_t count_int64_t,
                                       const torch::Dtype dtype,
                                       const int64_t layout_int64_t) {
  auto device_id = static_cast<int>(device_id_int64_t);
  auto offset_bytes = static_cast<size_t>(offset_bytes_int64_t);
  auto count = static_cast<size_t>(count_int64_t);
  auto layout = static_cast<ConstantMemoryGravity>(layout_int64_t);
  at::cuda::set_device(device_id);

  auto options =
      torch::TensorOptions().device(torch::kCUDA, device_id).dtype(dtype);
  torch::Tensor out = torch::empty({static_cast<int64_t>(count)}, options);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  AT_DISPATCH_INTEGRAL_TYPES(
      dtype, "read_constant_chunk_cuda", ([&] {
        read_chunk_kernel<scalar_t><<<1, count, 0, stream.stream()>>>(
            out.data_ptr<scalar_t>(), offset_bytes, count, layout);
      }));

  return out;
}

void upload_tensor_list(const std::vector<torch::Tensor> tensor_list,
                        const std::vector<int64_t> offset_list,
                        const int64_t layout,
                        const int64_t device_id) {
  upload_tensor_list_cuda(tensor_list, offset_list, layout, device_id);
}

torch::Tensor read_constant_chunk(const torch::Tensor& dummy,
                                  const int64_t offset_bytes,
                                  const int64_t count,
                                  const torch::Dtype dtype,
                                  const int64_t layout) {
  const int device_id =
      dummy.device().index();  // Extract from dummy tensor, since pytorch needs
                               // a tensor to dispatch op
  return read_constant_chunk_cuda(
      device_id, offset_bytes, count, dtype, layout);
}
