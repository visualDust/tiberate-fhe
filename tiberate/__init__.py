from importlib.metadata import version

from .fhe.context import presets
from .fhe.engine import CkksEngine

__version__ = version("tiberate")

__all__ = ["CkksEngine", "presets"]
