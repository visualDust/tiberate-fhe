cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(tiberate-csrc)

# ```bash
# ln -s $(python -c "import torch; import os; print(os.path.abspath(os.path.join(torch.utils.cmake_prefix_path, '../../')))") ./libtorch
# ```

list(APPEND CMAKE_PREFIX_PATH "libtorch")

find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(randint_cuda SHARED
    csprng/randint.cpp
    csprng/randint_cuda_kernel.cu
)

add_library(randround_cuda SHARED
    csprng/randround.cpp
    csprng/randround_cuda_kernel.cu
)

add_library(discrete_gaussian_cuda SHARED
    csprng/discrete_gaussian.cpp
    csprng/discrete_gaussian_cuda_kernel.cu
)

add_library(chacha20_cuda  SHARED
    csprng/chacha20.cpp
    csprng/chacha20_cuda_kernel.cu
)

add_library(ntt_cuda SHARED
    ntt/ntt.cpp
    ntt/ntt_cuda_kernel.cu
)

target_link_libraries(randint_cuda "${TORCH_LIBRARIES}")
target_link_libraries(randround_cuda "${TORCH_LIBRARIES}")
target_link_libraries(discrete_gaussian_cuda "${TORCH_LIBRARIES}")
target_link_libraries(chacha20_cuda "${TORCH_LIBRARIES}")
target_link_libraries(ntt_cuda "${TORCH_LIBRARIES}")

set_property(TARGET randint_cuda PROPERTY CXX_STANDARD 17)
set_property(TARGET randround_cuda PROPERTY CXX_STANDARD 17)
set_property(TARGET discrete_gaussian_cuda PROPERTY CXX_STANDARD 17)
set_property(TARGET chacha20_cuda PROPERTY CXX_STANDARD 17)
set_property(TARGET ntt_cuda PROPERTY CXX_STANDARD 17)
