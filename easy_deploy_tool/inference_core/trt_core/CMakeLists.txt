cmake_minimum_required(VERSION 3.8)
project(trt_core)


add_compile_options(-std=c++17)
add_compile_options(-O3 -Wextra -Wdeprecated -fPIC)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)


find_package(CUDA REQUIRED)


# TensorRT配置：从环境变量获取根目录并验证
if(DEFINED ENV{TENSORRT_ROOT})
    set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
    message(STATUS "从环境变量获取 TENSORRT_ROOT: ${TENSORRT_ROOT}")
else()
    message(FATAL_ERROR "未设置环境变量 TENSORRT_ROOT，请先配置 TensorRT 路径")
endif()

# 配置并打印include目录
set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
message(STATUS "===== TensorRT 路径信息 =====")
message(STATUS "TENSORRT_INCLUDE_DIR: ${TENSORRT_INCLUDE_DIR}")
if(NOT EXISTS ${TENSORRT_INCLUDE_DIR})
    message(FATAL_ERROR "TensorRT include目录不存在，请检查路径是否正确")
endif()

# 解析lib目录（处理符号链接）并打印
execute_process(
        COMMAND readlink -f "${TENSORRT_ROOT}/lib"
        OUTPUT_VARIABLE TENSORRT_LIB_DIR_ABSOLUTE
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(TENSORRT_LIB_DIR ${TENSORRT_LIB_DIR_ABSOLUTE})
message(STATUS "TENSORRT_LIB_DIR（实际路径）: ${TENSORRT_LIB_DIR}")
message(STATUS "==============================")

# 验证lib目录有效性
if(NOT EXISTS ${TENSORRT_LIB_DIR})
    message(FATAL_ERROR "TensorRT lib实际目录不存在：${TENSORRT_LIB_DIR}\n符号链接指向无效路径，请检查安装包完整性")
endif()

# 关键：显式查找每个TensorRT库，确保找到实际文件
find_library(TENSORRT_NVINFER
        NAMES nvinfer
        PATHS ${TENSORRT_LIB_DIR}
        NO_DEFAULT_PATH
        )
find_library(TENSORRT_NVONNXPARSER
        NAMES nvonnxparser
        PATHS ${TENSORRT_LIB_DIR}
        NO_DEFAULT_PATH
        )
find_library(TENSORRT_NVINFER_PLUGIN
        NAMES nvinfer_plugin
        PATHS ${TENSORRT_LIB_DIR}
        NO_DEFAULT_PATH
        )

# 验证库是否找到
if(NOT TENSORRT_NVINFER)
    message(FATAL_ERROR "未找到 nvinfer 库，路径：${TENSORRT_LIB_DIR}")
endif()
if(NOT TENSORRT_NVONNXPARSER)
    message(FATAL_ERROR "未找到 nvonnxparser 库，路径：${TENSORRT_LIB_DIR}")
endif()
if(NOT TENSORRT_NVINFER_PLUGIN)
    message(FATAL_ERROR "未找到 nvinfer_plugin 库，路径：${TENSORRT_LIB_DIR}")
endif()

# 打印找到的库路径（确认是否正确）
message(STATUS "找到 TensorRT 库：")
message(STATUS "  nvinfer: ${TENSORRT_NVINFER}")
message(STATUS "  nvonnxparser: ${TENSORRT_NVONNXPARSER}")
message(STATUS "  nvinfer_plugin: ${TENSORRT_NVINFER_PLUGIN}")

set(source_file src/trt_core.cpp
                src/trt_core_factory.cpp)


include_directories(
    include
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

link_directories(
        ${CUDA_LIBRARY_DIRS}
        ${TENSORRT_LIB_DIR}
)

add_library(${PROJECT_NAME} SHARED ${source_file})


target_link_libraries(${PROJECT_NAME} PUBLIC
    ${CUDA_LIBRARIES}
    ${TENSORRT_NVINFER}           # 显式链接nvinfer
    ${TENSORRT_NVONNXPARSER}      # 显式链接nvonnxparser
    ${TENSORRT_NVINFER_PLUGIN}    # 显式链接nvinfer_plugin
    deploy_core
)

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
