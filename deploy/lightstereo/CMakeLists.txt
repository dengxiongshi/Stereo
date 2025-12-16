
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

#set(GTEST_ROOT "/usr/local/lib" CACHE PATH "GTest installation root")
#find_package(GTest REQUIRED)
#if (NOT TARGET GTest::gtest_main)
#    message(FATAL_ERROR "GTest::gtest_main 目标不存在，请检查 GTest 安装")
#endif()
#find_package(glog REQUIRED)
#find_package(OpenCV REQUIRED)

# 复用deploy的OpenCV缓存变量（不受project影响）
if(NOT DEFINED OpenCV_INCLUDE_DIRS OR NOT DEFINED OpenCV_LIBS)
    message(FATAL_ERROR "未找到OpenCV配置！请先编译deploy目录")
endif()


set(source_file test_stereo_lightstereo.cpp)

include_directories(
        include
        ${OpenCV_INCLUDE_DIRS}
)

add_executable(test_stereo_lightstereo ${source_file})

target_link_libraries(test_stereo_lightstereo PUBLIC
        ${OpenCV_LIBS}
        deploy_core
        image_processing_utils
        deploy
        ${platform_core_packages}
)

if(ENABLE_TENSORRT)
    target_compile_definitions(test_stereo_lightstereo PRIVATE ENABLE_TENSORRT)
endif()

if(ENABLE_RKNN)
    target_compile_definitions(test_stereo_lightstereo PRIVATE ENABLE_RKNN)
endif()

if(ENABLE_ORT)
    target_compile_definitions(test_stereo_lightstereo PRIVATE ENABLE_ORT)
endif()

if(ENABLE_OM)
    target_compile_definitions(test_stereo_lightstereo PRIVATE ENABLE_OM)
endif()







