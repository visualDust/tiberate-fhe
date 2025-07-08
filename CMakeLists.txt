cmake_minimum_required(VERSION 3.15...3.26)
project(tiberate-csrc LANGUAGES CXX CUDA)

# === Optional: Use ccache ===
find_program(CCACHE_FOUND ccache)

if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif()

# === Python & Torch setup ===
find_package(
  Python3
  COMPONENTS Interpreter Development
  REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
          "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
          "import torch; print(torch.__path__[0] + '/lib')"
  OUTPUT_VARIABLE TORCH_LIB_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
          "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
  OUTPUT_VARIABLE TORCH_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE)

set(PYTHON_MODULE_SUFFIX ".${Python3_SOABI}")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PATH})
link_directories(${TORCH_LIB_PATH})

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# === Gather all .cu sources into one unified core ===
set(CKKS_CUDA_CORE_SRC
    csrc/utils/cuda/constant_mem_cuda.cu
    csrc/ops/cuda/chacha20_cuda.cu
    csrc/ops/cuda/discrete_gaussian_cuda.cu
    csrc/ops/cuda/mont_cuda.cu
    csrc/ops/cuda/mont_extra_cuda.cu
    csrc/ops/cuda/ntt_radix2_cuda.cu
    csrc/ops/cuda/intt_radix2_cuda.cu
    csrc/ops/cuda/he_fused_cuda.cu
    csrc/ops/cuda/cascade_mac_cuda.cu
    csrc/ops/cuda/randint_cuda.cu
    csrc/ops/cuda/randround_cuda.cu)

add_library(ckks_cuda_core STATIC ${CKKS_CUDA_CORE_SRC})
set_target_properties(ckks_cuda_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(ckks_cuda_core PRIVATE csrc ${TORCH_INCLUDE_DIRS}
                                                  ${Python3_INCLUDE_DIRS})

target_link_libraries(ckks_cuda_core PRIVATE ${TORCH_LIBRARIES})

# === Wrapper function to create MODULE libraries ===
function(add_pybind_module name cpp_path install_subdir)
  add_library(${name} MODULE ${cpp_path})

  target_link_libraries(${name} PRIVATE ${TORCH_LIBRARIES} torch_python
                                        ckks_cuda_core)
  target_include_directories(${name} PRIVATE ${TORCH_INCLUDE_DIRS}
                                             ${Python3_INCLUDE_DIRS})

  set_target_properties(
    ${name}
    PROPERTIES PREFIX ""
               CXX_STANDARD 17
               OUTPUT_NAME "${name}${PYTHON_MODULE_SUFFIX}")

  if(SKBUILD_STATE STREQUAL "editable")
    install(
      TARGETS ${name}
      LIBRARY
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/tiberate/libs/${install_subdir})
  else()
    install(TARGETS ${name}
            LIBRARY DESTINATION ./tiberate/libs/${install_subdir})
  endif()
endfunction()

# === Build all Python modules ===
# Compute modules (only .cpp needed now — .cu are in core)
set(OP_MODULE_NAMES
    chacha20
    discrete_gaussian
    mont
    mont_extra
    ntt_radix2
    intt_radix2
    he_fused
    cascade_mac
    randint
    randround)

foreach(name IN LISTS OP_MODULE_NAMES)
  add_pybind_module(${name} csrc/ops/${name}.cpp torchops)
endforeach()

# Non-compute CUDA module (e.g. constant_mem)
add_pybind_module(constant_mem csrc/utils/constant_mem.cpp utils)

# Pure C++ modules
add_pybind_module(cuda_info csrc/utils/cuda_info.cpp utils)

# === Optional wrapper script ===
set(WrapperGenScript ${CMAKE_SOURCE_DIR}/tiberate/libs/wrapper/genwrapper.py)
execute_process(
  COMMAND ${Python3_EXECUTABLE} ${WrapperGenScript} --verbose
  RESULT_VARIABLE WRAPPER_RESULT
  OUTPUT_VARIABLE WRAPPER_OUTPUT
  ERROR_VARIABLE WRAPPER_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)

message(STATUS "Wrapper script output: ${WRAPPER_OUTPUT}")

if(NOT "${WRAPPER_ERROR}" STREQUAL "")
  message(WARNING "Wrapper script stderr: ${WRAPPER_ERROR}")
endif()

if(NOT WRAPPER_RESULT EQUAL 0)
  message(FATAL_ERROR "Wrapper script failed with code ${WRAPPER_RESULT}")
endif()
