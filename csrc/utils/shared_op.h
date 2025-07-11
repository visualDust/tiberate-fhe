#pragma once
#include <torch/torch.h>

void upload_tensor_list(const std::vector<torch::Tensor> tensor_list,
                        const std::vector<int64_t> offset_list,
                        const int64_t layout,
                        const int64_t device_id);

torch::Tensor read_constant_chunk(const torch::Tensor& dummy,
                                  const int64_t offset_bytes,
                                  const int64_t count,
                                  const torch::Dtype dtype,
                                  const int64_t layout);

TORCH_LIBRARY_FRAGMENT(tiberate_const_pool, m) {
  m.def(
      "upload_tensor_list(Tensor[] tensor_list, int[] offset_list, "
      "int layout, int device_id) -> ()");
  m.def(
      "read_constant_chunk(Tensor dummy, int offset_bytes, "
      "int count, ScalarType dtype, int layout) -> Tensor");
}
TORCH_LIBRARY_IMPL(tiberate_const_pool, CUDA, m) {
  m.impl("upload_tensor_list", upload_tensor_list);
  m.impl("read_constant_chunk", read_constant_chunk);
}
