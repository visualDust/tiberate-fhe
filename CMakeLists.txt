cmake_minimum_required(VERSION 3.15...3.26)
project(tiberate-csrc LANGUAGES CXX CUDA)

find_program(CCACHE_FOUND ccache) # it's apt install ccache on Ubuntu

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

set(UNIFIED_MODULE_NAME _tiberate_ops)
set(WrapperGenScript ${CMAKE_SOURCE_DIR}/tiberate/libs/wrapper/genwrapper.py)

file(GLOB_RECURSE TORCH_OPS_CPP_SOURCES "csrc/ops/*.cpp" "csrc/utils/*.cpp")

file(GLOB_RECURSE TORCH_OPS_CU_SOURCES "csrc/ops/cuda/*.cu" "csrc/utils/*.cu")

message(STATUS "Found C++ sources: ${TORCH_OPS_CPP_SOURCES}")
message(STATUS "Found CUDA sources: ${TORCH_OPS_CU_SOURCES}")

add_library(${UNIFIED_MODULE_NAME} MODULE ${TORCH_OPS_CPP_SOURCES}
                                          ${TORCH_OPS_CU_SOURCES})

target_link_libraries(${UNIFIED_MODULE_NAME} PRIVATE ${TORCH_LIBRARIES}
                                                     torch_python)
target_include_directories(
  ${UNIFIED_MODULE_NAME} PRIVATE ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_compile_definitions(${UNIFIED_MODULE_NAME}
                           PRIVATE TORCH_EXTENSION_NAME=${UNIFIED_MODULE_NAME})

set_target_properties(
  ${UNIFIED_MODULE_NAME}
  PROPERTIES PREFIX ""
             CXX_STANDARD 17
             OUTPUT_NAME "${UNIFIED_MODULE_NAME}${PYTHON_MODULE_SUFFIX}"
             CUDA_SEPARABLE_COMPILATION ON)

if(SKBUILD_STATE STREQUAL "editable")
  install(
    TARGETS ${UNIFIED_MODULE_NAME}
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/tiberate/libs/torchops)
  install(
    CODE "
      execute_process(
          COMMAND ${Python3_EXECUTABLE} ${WrapperGenScript} --verbose
          WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
          RESULT_VARIABLE result
      )
  ")
else()
  install(TARGETS ${UNIFIED_MODULE_NAME}
          LIBRARY DESTINATION ./tiberate/libs/torchops)
endif()

# === Helper function to define modules ===
function(add_cpp_module name cpp_path out_subdir)
  add_library(${name} MODULE ${cpp_path})
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

# Non-compute pure C++ modules
set(NON_COMPUTE_MODULE_NAMES cuda_info)

foreach(name IN LISTS NON_COMPUTE_MODULE_NAMES)
  add_cpp_module(${name} ${CMAKE_CURRENT_SOURCE_DIR}/csrc/utils/${name}.cpp
                 utils)
endforeach()
