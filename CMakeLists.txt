cmake_minimum_required(VERSION 3.30 FATAL_ERROR)

set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SHARED_LIBRARY_PREFIX "")

option(CUDABOX_ENABLE_BENCHMARKS "Build nvbench microbenchmarks" OFF)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)
if(NOT DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES})
  detect_cuda_arch(_DETECTED_ARCH FALLBACK 90a)
  set(CMAKE_CUDA_ARCHITECTURES ${_DETECTED_ARCH}
      CACHE STRING "CUDA architectures (auto-detected; opts into `a` variant on Hopper+)")
endif()

set(CUDABOX_CUDA_ARCH "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING
    "Per-target CUDA arch for CudaBox kernels (snapshot before Torch clobbers global)")

project(cudabox LANGUAGES CXX CUDA)

# Dependencies
include(${CMAKE_CURRENT_LIST_DIR}/cmake/dependencies.cmake)

# Python
# scikit-build-core doesn't support cmake presets
# https://github.com/scikit-build/scikit-build-core/pull/994
find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

# CXX/CUDA
# PyTorch 2.7+ requires C++20 for both its C++ and CUDA-facing headers.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CXX Flags
set(CUDABOX_CXX_FLAGS

)

# CUDA
find_package(CUDAToolkit REQUIRED)
set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# CUDA Flags
set(CUDABOX_CUDA_FLAGS
  "--use_fast_math"
  "-lineinfo"
)

# Torch
# Also feed Torch the same arch so its own bits (if any get compiled here)
# match. Format expected by torch: dot-separated, e.g. "9.0a" not "90a".
string(REGEX REPLACE "^([0-9])([0-9])(a?)$" "\\1.\\2\\3" _TORCH_ARCH
                     "${CUDABOX_CUDA_ARCH}")
set(ENV{TORCH_CUDA_ARCH_LIST} "${_TORCH_ARCH}")
set(USE_SYSTEM_NVTX ON) # Why is it not the default in torch
find_package(Torch REQUIRED)
clear_cuda_arches(_TORCH_GENCODE_DISCARDED)

# ccache option
option(ENABLE_CCACHE "Whether to use ccache" ON)
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND AND ENABLE_CCACHE AND DEFINED ENV{CCACHE_DIR})
  message(STATUS "Building with CCACHE enabled")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "ccache")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "ccache")
endif()

# Enable ctest
enable_testing()

# Public include
include_directories(
  "${CMAKE_CURRENT_LIST_DIR}/include"
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build arch: ${CUDABOX_CUDA_ARCH}")

add_subdirectory("csrc/algorithms")
add_subdirectory("csrc/elementwise")
add_subdirectory("csrc/gemm")
