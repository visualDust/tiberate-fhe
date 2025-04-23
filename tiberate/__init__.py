import importlib
import os.path as osp

import torch

for library in ["chacha20", "discrete_gaussian", "mont", "ntt", "randint", "randround"]:
    spec = importlib.machinery.PathFinder().find_spec(f"lib{library}", [osp.dirname(__file__)])
    if spec is not None:
        torch.ops.load_library(spec.origin)
    else:  # pragma: no cover
        raise ImportError(
            f"Could not find the library {library}. Please ensure it is installed correctly."
        )


from tiberate.fhe.context import presets  # noqa
from tiberate.fhe.engine import CkksEngine  # noqa

__all__ = ["CkksEngine", "presets"]
