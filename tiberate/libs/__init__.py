import os

import torch

torch_op_dirs = ['torchops']


def load_pytorch_ops(path):
    # the built libraries are in the libs directory of the package
    lib_dir = os.path.join(os.path.dirname(__file__), path)
    ext = ".so"
    # search every file in the lib directory that ends with .so and load it
    for lib_file in os.listdir(lib_dir):
        if lib_file.endswith(ext):
            lib_path = os.path.join(lib_dir, lib_file)
            if os.path.isfile(lib_path):
                # logger.debug(f"Loading library {lib_path}")
                torch.ops.load_library(lib_path)
            else:
                raise RuntimeError(
                    f"Failed to load library from {lib_path}. "
                    "Please make sure you have built the library correctly."
                )


# Load the PyTorch ops
for lib_dir in torch_op_dirs:
    load_pytorch_ops(lib_dir)

from tiberate.libs.wrapper import (
    fake_op,  # Register the FakeTensor kernel see https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0  # noqa: F401
)
