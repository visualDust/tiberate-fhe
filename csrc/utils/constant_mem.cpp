#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "cuda/constant_mem_cuda.h"

namespace py = pybind11;

void upload_tensor_list(const std::vector<torch::Tensor>& tensor_list,
                        const std::vector<int64_t>& offset_list,
                        int layout,
                        int device_id) {
  upload_tensor_list_cuda(tensor_list,
                          offset_list,
                          static_cast<ConstantMemoryGravity>(layout),
                          device_id);
}

torch::Tensor read_constant_chunk(int device_id,
                                  size_t offset_bytes,
                                  size_t count,
                                  torch::Dtype dtype,
                                  int layout) {
  return read_constant_chunk_cuda(device_id,
                                  offset_bytes,
                                  count,
                                  dtype,
                                  static_cast<ConstantMemoryGravity>(layout));
}

PYBIND11_MODULE(constant_mem, m) {
  m.def("upload_tensor_list",
        &upload_tensor_list,
        "Upload a list of tensors to constant memory",
        py::arg("tensor_list"),
        py::arg("offset_list"),
        py::arg("layout") = 0,
        py::arg("device_id") = 0);
  m.def("read_constant_chunk",
        &read_constant_chunk,
        "Read a chunk of constant memory",
        py::arg("device_id"),
        py::arg("offset_bytes"),
        py::arg("count"),
        py::arg("dtype"),
        py::arg("layout") = 0);
}
