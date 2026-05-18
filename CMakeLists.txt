cmake_minimum_required(VERSION 3.30 FATAL_ERROR)
project(cudabox LANGUAGES CXX CUDA)

set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SHARED_LIBRARY_PREFIX "")

# CMake
include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/dependencies.cmake)

# Python
# scikit-build-core doesn't support cmake presets
# https://github.com/scikit-build/scikit-build-core/pull/994
find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

# CXX
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CXX Flags
set(CUDABOX_CXX_FLAGS

)

# CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# CUDA Flags
set(CUDABOX_CUDA_FLAGS
  "--use_fast_math"
  "-lineinfo"
)

# Torch
set(USE_SYSTEM_NVTX ON) # Why is it not the default in torch
find_package(Torch REQUIRED)

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

add_subdirectory("csrc/elementwise")
add_subdirectory("csrc/gemm")
