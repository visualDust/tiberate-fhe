#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "cuda/constant_mem_cuda.h"

void upload_constants_2qRsQlQhKlKh(const std::vector<torch::Tensor> _2q,
                                   const std::vector<torch::Tensor> Rs,
                                   const std::vector<torch::Tensor> ql,
                                   const std::vector<torch::Tensor> qh,
                                   const std::vector<torch::Tensor> kl,
                                   const std::vector<torch::Tensor> kh) {
  const auto num_devices = _2q.size();
  for (size_t i = 0; i < num_devices; ++i) {
    upload_constants_cuda(_2q[i], Rs[i], ql[i], qh[i], kl[i], kh[i]);
  }
}

PYBIND11_MODULE(constant_mem, m) {
  m.def("upload_constants_2qRsQlQhKlKh",
        &upload_constants_2qRsQlQhKlKh,
        "Upload constants to CUDA constant memory");
}
