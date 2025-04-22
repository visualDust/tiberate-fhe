#include <Python.h>
#include <vector>
#include "cuda/chacha20_cuda.h"
#include "extensions.h"

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so consisting of this file
   in this extension, so that the TORCH_LIBRARY static initializers
   below are run. */
PyObject* PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_C", /* name of module */
      NULL, /* module documentation, may be NULL */
      -1,   /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
      NULL, /* methods */
  };
  return PyModule_Create(&module_def);
}
}

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

TORCH_LIBRARY(tiberate_csprng_ops, m) {
  m.def("chacha20(Tensor[] input, int step) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(tiberate_csprng_ops, CUDA, m) {
  m.impl("chacha20", &chacha20);
}
