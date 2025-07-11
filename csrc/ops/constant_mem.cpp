#include "cuda/constant_pool_cuda.h"

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
