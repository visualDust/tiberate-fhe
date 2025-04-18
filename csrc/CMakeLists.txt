cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(tiberate-csrc)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Python Executable: ${Python3_EXECUTABLE}")
message(STATUS "Python Include Dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Torch cmake path: ${TORCH_CMAKE_PATH}")

list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PATH})

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(
    ${Python_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
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
