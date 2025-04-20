import logging
import os
import pathlib
import shutil
import subprocess

from setuptools import setup
from setuptools.command.build_ext import build_ext as build_ext_orig
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def clean_built():
    """
    Remove common build directories and *.so files in a Python project.
    """
    dirs_to_remove = [
        "__pycache__",
        ".pytest_cache",
        "*.egg-info",
    ]
    files_to_remove = ["*.so, *.pyi"]

    for path in pathlib.Path(".").rglob("*"):
        if path.is_dir() and any(path.match(d) for d in dirs_to_remove):
            shutil.rmtree(path, ignore_errors=True)
            logger.info(f"Removed directory: {path}")
        elif path.is_file() and any(path.match(f) for f in files_to_remove):
            path.unlink()
            logger.info(f"Removed file: {path}")


class BuildExtensionWithStub(BuildExtension):
    def run(self):
        super().run()

        # ========================================================
        # The only useful part of this function is super.run()
        # All things below are for generating stubs
        # Its okay if they fail, the library is still built
        # ========================================================

        import torch  # Needed to resolve linked library locations

        build_path = pathlib.Path(self.build_lib)

        for ext in self.extensions:
            logger.info(f"Searching for .so file for extension: {ext.name}")
            for so_path in build_path.rglob(ext.name + "*.so"):
                logger.info(f"Found .so: {so_path}")

                # Infer module name from path
                rel_so = so_path.relative_to(build_path)
                if rel_so.suffix != ".so":
                    continue

                # Clean up ABI tags, just use base name
                stem = rel_so.stem.split(".")[0]  # strip .cpython-313-*.so
                parts = list(rel_so.parent.parts) + [stem]
                module_name = ".".join(parts)  # tiberate.ntt.ntt_cuda
                logger.info(f"Inferred module name: {module_name}")

                # Prepare environment to ensure libc10 and friends are found
                torch_lib = pathlib.Path(torch.__file__).parent / "lib"
                env = os.environ.copy()
                env["PYTHONPATH"] = str(build_path) + ":" + env.get("PYTHONPATH", "")
                env["LD_LIBRARY_PATH"] = str(torch_lib) + ":" + env.get("LD_LIBRARY_PATH", "")

                try:
                    subprocess.run(
                        [
                            "pybind11-stubgen",
                            module_name,
                            "-o",
                            str(build_path),
                        ],
                        check=True,
                        env=env,
                    )
                    logger.info(f"Generated stub for {module_name}")
                except subprocess.CalledProcessError as e:
                    logger.warning(
                        f"Failed to generate stub for {module_name}: {e}. It's okay, the library is still built."
                    )


ext_modules_csprng = [
    CUDAExtension(
        name="randint_cuda",
        sources=[
            "csrc/csprng/randint.cpp",
            "csrc/csprng/randint_cuda_kernel.cu",
        ],
    ),
    CUDAExtension(
        name="randround_cuda",
        sources=[
            "csrc/csprng/randround.cpp",
            "csrc/csprng/randround_cuda_kernel.cu",
        ],
    ),
    CUDAExtension(
        name="discrete_gaussian_cuda",
        sources=[
            "csrc/csprng/discrete_gaussian.cpp",
            "csrc/csprng/discrete_gaussian_cuda_kernel.cu",
        ],
    ),
    CUDAExtension(
        name="chacha20_cuda",
        sources=[
            "csrc/csprng/chacha20.cpp",
            "csrc/csprng/chacha20_cuda_kernel.cu",
        ],
    ),
]

ext_modules_ntt = [
    CUDAExtension(
        name="ntt_cuda",
        sources=["csrc/ntt/ntt.cpp", "csrc/ntt/ntt_cuda_kernel.cu"],
    ),
]

if __name__ == "__main__":
    clean_built()

    num_cores_os = os.cpu_count() // 2
    os.environ["MAX_JOBS"] = str(num_cores_os)

    setup(
        name="ntt",
        ext_modules=ext_modules_ntt,
        cmdclass={"build_ext": BuildExtensionWithStub},
        script_args=["build_ext"],
        options={"build": {"build_lib": "tiberate/ntt"}},
    )

    setup(
        name="csprng",
        ext_modules=ext_modules_csprng,
        cmdclass={"build_ext": BuildExtensionWithStub},
        script_args=["build_ext"],
        options={"build": {"build_lib": "tiberate/rng/csprng"}},
    )
