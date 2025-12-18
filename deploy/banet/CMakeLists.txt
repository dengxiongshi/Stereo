
set(CMAKE_BUILD_TYPE Debug)  # 放在 project 命令前
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")  # 禁用优化，保留调试信息
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_COMMON} -O3 -s")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_COMMON "-fPIC -Wall -lm -ldl -lpthread -std=c++17")
set(CMAKE_C_FLAGS_COMMON "-std=c99")


if(ENABLE_TENSORRT)
    list(APPEND platform_core_packages trt_core)
endif()

if(ENABLE_RKNN)
    list(APPEND platform_core_packages rknn_core)
endif()

if(ENABLE_ORT)
    list(APPEND platform_core_packages ort_core)
endif()

if(ENABLE_OM)
    list(APPEND platform_core_packages om_core)
endif()

#if (ENABLE_OM)
#    set(OpenCV_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include/opencv)
#    file(GLOB OpenCV_LIBS ${PROJECT_SOURCE_DIR}/lib/opencv/*.so)
#    # 可选：验证库文件是否存在
#    if (NOT OpenCV_LIBS)
#        message(FATAL_ERROR "ENABLE_OM=ON时，未找到OpenCV库：${PROJECT_SOURCE_DIR}/lib/opencv/*.so")
#    endif()
#else()
#    find_package(OpenCV REQUIRED)
#endif()

# 复用deploy的OpenCV缓存变量（不受project影响）
if(NOT DEFINED OpenCV_INCLUDE_DIRS OR NOT DEFINED OpenCV_LIBS)
    message(FATAL_ERROR "未找到OpenCV配置！请先编译deploy目录")
endif()


set(source_file test_stereo_banet2d_tt.cpp)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(test_stereo_banet ${source_file})

target_link_libraries(test_stereo_banet PUBLIC
        ${OpenCV_LIBS}
        deploy_core
        image_processing_utils
        deploy
        ${platform_core_packages}
)

if(ENABLE_TENSORRT)
    target_compile_definitions(test_stereo_banet PRIVATE ENABLE_TENSORRT)
endif()

if(ENABLE_RKNN)
    target_compile_definitions(test_stereo_banet PRIVATE ENABLE_RKNN)
endif()

if(ENABLE_ORT)
    target_compile_definitions(test_stereo_banet PRIVATE ENABLE_ORT)
endif()

if(ENABLE_OM)
    target_compile_definitions(test_stereo_banet PRIVATE ENABLE_OM)
endif()

