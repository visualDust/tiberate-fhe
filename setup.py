import logging
import pathlib
import shutil

from setuptools import setup
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
        "*.egg-info",  # "dist",
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


ext_modules = [
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

logger.info("Cleaning up built directories and files...")
clean_built()  # Clean up the build directories and *.so files

logger.info("Building csprng...")
setup(
    name="csprng",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    script_args=["build_ext"],
    options={
        "build": {
            "build_lib": "tiberate/csprng",
        }
    },
)

logger.info("Building ntt...")
setup(
    name="ntt",
    ext_modules=ext_modules_ntt,
    script_args=["build_ext"],
    cmdclass={"build_ext": BuildExtension},
    options={
        "build": {
            "build_lib": "tiberate/ntt",
        }
    },
)
