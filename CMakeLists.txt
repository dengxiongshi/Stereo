cmake_minimum_required(VERSION 3.8)
project(lightstereo_onnx)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (BUILD_TESTING)
    enable_testing()
endif()

add_subdirectory(deploy)
add_subdirectory(easy_deploy_tool)