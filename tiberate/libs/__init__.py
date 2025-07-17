import os
import re
import sys

import torch

torch_op_dirs = ['torchops']


def get_current_python_tag():
    """Return the Python version tag like 'cpython-312' for the current interpreter."""
    major = sys.version_info.major
    minor = sys.version_info.minor
    return f"cpython-{major}{minor}"


def load_pytorch_ops(path):
    lib_dir = os.path.join(os.path.dirname(__file__), path)
    current_tag = get_current_python_tag()
    version_pattern = re.compile(rf".*\.{current_tag}.*\.so$")

    for lib_file in os.listdir(lib_dir):
        lib_path = os.path.join(lib_dir, lib_file)
        if os.path.isfile(lib_path) and version_pattern.match(lib_file):
            torch.ops.load_library(lib_path)
        # optional warning or debug logging if needed
        # else:
        #     print(f"Skipped non-matching file: {lib_file}")


# Load the PyTorch ops
for lib_dir in torch_op_dirs:
    load_pytorch_ops(lib_dir)

from tiberate.libs.wrapper import (
    fake_op,  # supports coexistence of multiple py versions build # noqa: F401
)
