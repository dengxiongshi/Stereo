cmake_minimum_required(VERSION 3.8)

set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures" FORCE)

project(image_processing_utils)

add_compile_options(-std=c++17)
add_compile_options(-O3)
set(CMAKE_CXX_STANDARD 17)

set(link_libs
    ${OpenCV_LIBS}
    deploy_core
)

if(ENABLE_TENSORRT)
  find_package(CUDA REQUIRED)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
  enable_language(CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})

  list(APPEND link_libs
          ${CUDA_CUDART_LIBRARY}  # 推荐用CUDA自带变量（比直接写cudart更鲁棒）
          # cudart  # 若CUDA_CUDART_LIBRARY找不到，可直接写cudart（兜底）
  )
endif()


set(source_file
    src/image_processing_cpu.cpp
)

if(ENABLE_TENSORRT)
  list(APPEND source_file
          src/image_processing_cuda_kernel.cu
          src/image_processing_cuda.cpp
  )
endif()

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED ${source_file})

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${link_libs}
)

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
