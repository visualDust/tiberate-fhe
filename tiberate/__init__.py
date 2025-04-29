import os
from importlib.metadata import version

# from loguru import logger
import torch


def load_pytorch_ops():
    # the built libraries are in the libs directory of the package
    lib_dir = os.path.join(os.path.dirname(__file__), "libs")
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
load_pytorch_ops()

from tiberate.config import CkksConfig, Preset
from tiberate.engine import CkksEngine

__version__ = version("tiberate")

__all__ = ["CkksEngine", "Preset", "CkksConfig"]
