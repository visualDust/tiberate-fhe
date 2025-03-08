import logging
import pathlib
import shutil

from setuptools import setup
from setuptools.command.install import install
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

logger = logging.getLogger(__name__)


def clean_built():
    """
    Remove common build directories and *.so files in a Python project.
    """
    # Directories to remove
    dirs_to_remove = [
        "__pycache__",
        ".pytest_cache",
        "build",
        "*.egg-info",
    ]
    # File patterns to remove
    files_to_remove = ["*.so"]

    for path in pathlib.Path(".").rglob("*"):
        # Remove specified directories
        if path.is_dir() and any(path.match(d) for d in dirs_to_remove):
            shutil.rmtree(path, ignore_errors=True)
            logger.info(f"Removed directory: {path}")
        # Remove specified files
        elif path.is_file() and any(path.match(f) for f in files_to_remove):
            path.unlink()
            logger.info(f"Removed file: {path}")


ext_modules_csprng = [
    CUDAExtension(
        name="randint_cuda",
        sources=[
            "tiberate/csprng/randint.cpp",
            "tiberate/csprng/randint_cuda_kernel.cu",
        ],
    ),
    CUDAExtension(
        name="randround_cuda",
        sources=[
            "tiberate/csprng/randround.cpp",
            "tiberate/csprng/randround_cuda_kernel.cu",
        ],
    ),
    CUDAExtension(
        name="discrete_gaussian_cuda",
        sources=[
            "tiberate/csprng/discrete_gaussian.cpp",
            "tiberate/csprng/discrete_gaussian_cuda_kernel.cu",
        ],
    ),
    CUDAExtension(
        name="chacha20_cuda",
        sources=[
            "tiberate/csprng/chacha20.cpp",
            "tiberate/csprng/chacha20_cuda_kernel.cu",
        ],
    ),
]

ext_modules_ntt = [
    CUDAExtension(
        name="ntt_cuda",
        sources=[
            "tiberate/ntt/ntt.cpp",
            "tiberate/ntt/ntt_cuda_kernel.cu",
        ],
    )
]

setup(
    name="csprng",
    ext_modules=ext_modules_csprng,
    cmdclass={"build_ext": BuildExtension},
    script_args=["build_ext"],
    options={
        "build": {
            "build_lib": "tiberate/csprng",
        }
    },
)

setup(
    name="ntt",
    ext_modules=ext_modules_ntt,
    cmdclass={"build_ext": BuildExtension},
    script_args=["build_ext"],
    options={
        "build": {
            "build_lib": "tiberate/ntt",
        }
    },
)
