cmake_minimum_required(VERSION 3.15...3.26)
project(tiberate-csrc LANGUAGES CXX CUDA)

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

message(STATUS "Python Executable: ${Python3_EXECUTABLE}")
message(STATUS "Python Include Dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Torch cmake path: ${TORCH_CMAKE_PATH}")
message(STATUS "Torch lib path: ${TORCH_LIB_PATH}")
message(STATUS "Torch ABI: ${TORCH_ABI}")
message(STATUS "Detected Python SOABI: ${Python3_SOABI}")

set(PYTHON_MODULE_SUFFIX ".${Python3_SOABI}")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PATH})
link_directories(${TORCH_LIB_PATH})

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${Python_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# === Helper function to define modules ===
function(add_cuda_module name cpp_path cu_path out_subdir)
  if(cu_path)
    add_library(${name} MODULE ${cpp_path} ${cu_path})
  else()
    add_library(${name} MODULE ${cpp_path})
  endif()

  target_link_libraries(${name} ${TORCH_LIBRARIES} torch_python)
  target_compile_definitions(${name} PRIVATE TORCH_EXTENSION_NAME=${name})

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
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/tiberate/libs/${out_subdir})
  else()
    install(TARGETS ${name} LIBRARY DESTINATION ./tiberate/libs/${out_subdir})
  endif()
endfunction()

# === Define modules ===

# Compute modules
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
  add_cuda_module(
    ${name} ${CMAKE_CURRENT_SOURCE_DIR}/csrc/ops/${name}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/ops/cuda/${name}_cuda.cu torchops)
endforeach()

# Non-compute CUDA modules
set(NON_COMPUTE_CUDA_MODULE_NAMES constant_mem)

foreach(name IN LISTS NON_COMPUTE_CUDA_MODULE_NAMES)
  add_cuda_module(
    ${name} ${CMAKE_CURRENT_SOURCE_DIR}/csrc/utils/${name}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/utils/cuda/${name}_cuda.cu utils)
endforeach()

# Non-compute pure C++ modules
set(NON_COMPUTE_MODULE_NAMES cuda_info)

foreach(name IN LISTS NON_COMPUTE_MODULE_NAMES)
  add_cuda_module(${name} ${CMAKE_CURRENT_SOURCE_DIR}/csrc/utils/${name}.cpp ""
                  utils)
endforeach()

set(Python_ARGS "Tools/genwrapper.py")
execute_process(COMMAND ${Python3_EXECUTABLE} ${Python_ARGS})
