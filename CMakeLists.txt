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

# === Helper to add a CUDA+C++ Torch extension ===
function(add_torch_module name src_dir out_subdir)
  file(GLOB_RECURSE CPP_SOURCES "${src_dir}/*.cpp")
  file(GLOB_RECURSE CU_SOURCES "${src_dir}/*.cu")

  message(STATUS "Torch module ${name}:")
  message(STATUS "  C++ sources: ${CPP_SOURCES}")
  message(STATUS "  CUDA sources: ${CU_SOURCES}")

  add_library(${name} MODULE ${CPP_SOURCES} ${CU_SOURCES})
  target_link_libraries(${name} PRIVATE ${TORCH_LIBRARIES} torch_python)
  target_include_directories(${name} PRIVATE ${TORCH_INCLUDE_DIRS}
                                             ${Python3_INCLUDE_DIRS})
  target_compile_definitions(${name} PRIVATE TORCH_EXTENSION_NAME=${name})

  set_target_properties(
    ${name}
    PROPERTIES PREFIX ""
               CXX_STANDARD 17
               CUDA_SEPARABLE_COMPILATION ON
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

# === Helper to add pure C++ modules ===
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

# === Torch modules ===
add_torch_module(_ops "${CMAKE_CURRENT_SOURCE_DIR}/csrc/ops" torchops)
add_torch_module(_csprng "${CMAKE_CURRENT_SOURCE_DIR}/csrc/csprng" torchops)

# === Only run wrapper generator once ===
set(WrapperGenScript ${CMAKE_SOURCE_DIR}/tiberate/libs/wrapper/genwrapper.py)

if(SKBUILD_STATE STREQUAL "editable")
  install(
    CODE "
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${WrapperGenScript} --verbose
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE result
    )
  ")
endif()

# === Utility modules ===
set(NON_COMPUTE_MODULE_NAMES cuda_info)

foreach(name IN LISTS NON_COMPUTE_MODULE_NAMES)
  add_cpp_module(${name} ${CMAKE_CURRENT_SOURCE_DIR}/csrc/utils/${name}.cpp
                 utils)
endforeach()
