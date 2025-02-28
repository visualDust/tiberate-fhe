from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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

if __name__ == "__main__":
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

