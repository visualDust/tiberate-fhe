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

static auto registry =
    torch::RegisterOperators()
        .op("torch_tiberate::discrete_gaussian",
            &discrete_gaussian,
            torch::RegisterOperators::options().schema(
                "torch_tiberate::discrete_gaussian(Tensor[] rand_bytes, "
                "int64_t btree_ptr, int64_t btree_size, int64_t depth) -> ()"))
        .op("torch_tiberate::discrete_gaussian_fast",
            &discrete_gaussian_fast,
            torch::RegisterOperators::options().schema(
                "torch_tiberate::discrete_gaussian_fast(Tensor[] states, "
                "int64_t "
                "btree_ptr, int64_t btree_size, int64_t depth, int64_t step) "
                "-> Tensor[]"));
