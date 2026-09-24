set(SOURCES
  "src/flash_attention/flash_attention.cu"
  "src/histogram/histogram.cu"
  "src/algorithms_ops.cpp"
)

python_add_library(algorithms_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})
add_library(algorithms_ops_static STATIC ${SOURCES})

foreach(target algorithms_ops algorithms_ops_static)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDABOX_CUDA_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${CUDABOX_CXX_FLAGS}>
  )

  target_include_directories(${target} PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/include"
  )

  # need to add the following cuz of the following compiler error
  # error: identifier "_BitInt" is undefined in fmt/base.h
  target_compile_definitions(${target} PRIVATE FMT_USE_BITINT=0)

  target_link_libraries(
    ${target} PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cuda_driver CUDA::cudart
    spdlog::spdlog_header_only
  )

  set_target_properties(${target} PROPERTIES
    CUDA_ARCHITECTURES "${CUDABOX_CUDA_ARCH}"
  )
endforeach()

install(TARGETS algorithms_ops LIBRARY DESTINATION ${PROJECT_NAME})
