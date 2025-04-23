#include <vector>
#include "cuda/randround_cuda.h"
#include "extensions.h"

// The main function.
// rand_bytes are N 1D uint64_t tensors.
// Inputs are N 1D double tensors.
// The output will be returned in rand_bytes.
void randround(std::vector<torch::Tensor> inputs,
               std::vector<torch::Tensor> rand_bytes) {
  for (size_t i = 0; i < inputs.size(); i++) {
    CHECK_INPUT(rand_bytes[i]);

    // Run in cuda.
    randround_cuda(inputs[i], rand_bytes[i]);
  }
}

TORCH_LIBRARY_FRAGMENT(tiberate_csprng_ops, m) {
  m.def("randround(Tensor[] input, Tensor[] rand_bytes) -> ()");
}

TORCH_LIBRARY_IMPL(tiberate_csprng_ops, CUDA, m) {
  m.impl("randround", &randround);
}
