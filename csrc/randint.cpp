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

static auto registry =
    torch::RegisterOperators()
        .op("torch_tiberate::randint",
            &randint,
            torch::RegisterOperators::options().schema(
                "torch_tiberate::randint(Tensor[] rand_bytes, "
                "int64_t[] q_ptrs) -> ()"))
        .op("torch_tiberate::randint_fast",
            &randint_fast,
            torch::RegisterOperators::options().schema(
                "torch_tiberate::randint_fast(Tensor[] states, "
                "int64_t[] q_ptrs, int64_t shift, int64_t step) -> Tensor[]"));
