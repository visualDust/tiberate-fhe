import importlib
import pkgutil

from vdtoys.framing import get_frame_module_traceback
from vdtoys.registry import Registry


def _scan_sub_modules():
    __THIS_MODULE = get_frame_module_traceback(1)
    for sub_module_info in pkgutil.iter_modules(__THIS_MODULE.__path__):
        importlib.import_module(
            f"{__THIS_MODULE.__name__}.{sub_module_info.name}"
        )


_scan_sub_modules()  # Import all submodules so that they can register themselves

benchreg = Registry("benchmarks")

__all__ = [
    "benchreg",
]
