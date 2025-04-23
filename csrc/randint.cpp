#include <vector>
#include "cuda/randint_cuda.h"
#include "extensions.h"

// The main function.
//----------------
// Normal version.
void randint(std::vector<torch::Tensor> inputs, std::vector<int64_t> q_ptrs) {
  for (size_t i = 0; i < inputs.size(); i++) {
    CHECK_INPUT(inputs[i]);

    // reinterpret pointers from numpy.
    uint64_t *q = reinterpret_cast<uint64_t *>(q_ptrs[i]);

    // Run in cuda.
    randint_cuda(inputs[i], q);
  }
}

//--------------
// Fast version.
std::vector<torch::Tensor> randint_fast(std::vector<torch::Tensor> states,
                                        std::vector<int64_t> q_ptrs,
                                        int64_t shift,
                                        int64_t step) {
  std::vector<torch::Tensor> outputs;

  for (size_t i = 0; i < states.size(); i++) {
    uint64_t *q = reinterpret_cast<uint64_t *>(q_ptrs[i]);
    auto result = randint_fast_cuda(states[i], q, shift, step);
    outputs.push_back(result);
  }
  return outputs;
}

TORCH_LIBRARY_FRAGMENT(tiberate_csprng_ops, m) {
  m.def("randint(Tensor[] input, int[] q_ptrs) -> ()");
  m.def(
      "randint_fast(Tensor[] input, int[] q_ptrs, int shift, int step) -> "
      "Tensor[]");
}

TORCH_LIBRARY_IMPL(tiberate_csprng_ops, CUDA, m) {
  m.impl("randint", &randint);
  m.impl("randint_fast", &randint_fast);
}
