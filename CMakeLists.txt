cmake_minimum_required(VERSION 3.15...3.26)
project(tiberate-csrc LANGUAGES CXX CUDA)

option(WITH_CUDA "Enable CUDA support" OFF)
option(WITH_DEVELOP "Enable development mode" OFF)

if(WITH_CUDA)
  enable_language(CUDA)
  add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
  add_definitions(-DWITH_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

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
message(STATUS "Detected Python SOABI: ${Python3_SOABI}")

set(PYTHON_MODULE_SUFFIX ".${Python3_SOABI}")
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

set(MODULE_NAMES
    chacha20
    discrete_gaussian
    mont
    ntt
    randint
    randround
)

foreach(name IN LISTS MODULE_NAMES)
    add_library(${name} MODULE
        csrc/${name}.cpp
        csrc/cuda/${name}_cuda.cu
    )

    target_link_libraries(${name} ${TORCH_LIBRARIES})
    set_target_properties(${name} PROPERTIES
        PREFIX "lib"
        CXX_STANDARD 17
        OUTPUT_NAME "${name}${PYTHON_MODULE_SUFFIX}"
    )

    install(TARGETS ${name}
        LIBRARY DESTINATION .
    )
endforeach()
