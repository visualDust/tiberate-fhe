#include <vector>
#include "cuda/chacha20_cuda.h"
#include "extensions.h"

// chacha20 is a mutating function.
// That means the input is mutated and there's no need to return a value.

// Forward declaration.
std::vector<torch::Tensor> chacha20(std::vector<torch::Tensor> inputs,
                                    int64_t step) {
  // The input must be a contiguous long tensor of size 16 x N.
  // Also, the tensor must be contiguous to enable pointer arithmetic,
  // and must be stored in a cuda device.
  // Note that the input is a vector of inputs in different devices.

  std::vector<torch::Tensor> outputs;

  for (auto& input : inputs) {
    CHECK_INPUT(input);

    // Prepare an output.
    auto dest = input.clone();

    // Run in cuda.
    chacha20_cuda(input, dest, step);

    // Store to the dest.
    outputs.push_back(dest);
  }

  return outputs;
}

TORCH_LIBRARY_FRAGMENT(tiberate_csprng_ops, m) {
  m.def("chacha20(Tensor[] input, int step) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(tiberate_csprng_ops, CUDA, m) {
  m.impl("chacha20", &chacha20);
}
