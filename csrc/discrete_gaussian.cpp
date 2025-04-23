#include <cstdint>
#include <vector>
#include "cuda/discrete_gaussian_cuda.h"
#include "extensions.h"

// The main function.
//----------------
// Normal version.
void discrete_gaussian(std::vector<torch::Tensor> inputs,
                       int64_t btree_ptr,
                       int64_t btree_size,
                       int64_t depth) {
  // reinterpret pointers from numpy.
  uint64_t *btree = reinterpret_cast<uint64_t *>(btree_ptr);

  for (auto &rand_bytes : inputs) {
    CHECK_INPUT(rand_bytes);

    // Run in cuda.
    discrete_gaussian_cuda(rand_bytes, btree, btree_size, depth);
  }
}

//--------------
// Fast version.

std::vector<torch::Tensor> discrete_gaussian_fast(
    std::vector<torch::Tensor> states,
    int64_t btree_ptr,
    int64_t btree_size,
    int64_t depth,
    int64_t step) {
  // reinterpret pointers from numpy.
  uint64_t *btree = reinterpret_cast<uint64_t *>(btree_ptr);

  std::vector<torch::Tensor> outputs;

  for (auto &my_states : states) {
    auto result =
        discrete_gaussian_fast_cuda(my_states, btree, btree_size, depth, step);
    outputs.push_back(result);
  }
  return outputs;
}

TORCH_LIBRARY_FRAGMENT(tiberate_csprng_ops, m) {
  m.def(
      "discrete_gaussian(Tensor[] input, int btree_ptr, int btree_size, "
      "int depth) -> ()");
  m.def(
      "discrete_gaussian_fast(Tensor[] input, int btree_ptr, int btree_size, "
      "int depth, int step) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(tiberate_csprng_ops, CUDA, m) {
  m.impl("discrete_gaussian", &discrete_gaussian);
  m.impl("discrete_gaussian_fast", &discrete_gaussian_fast);
}
