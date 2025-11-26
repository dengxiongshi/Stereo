
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

#set(GTEST_ROOT "/usr/local/lib" CACHE PATH "GTest installation root")
#find_package(GTest REQUIRED)
#if (NOT TARGET GTest::gtest_main)
#    message(FATAL_ERROR "GTest::gtest_main 目标不存在，请检查 GTest 安装")
#endif()
#find_package(glog REQUIRED)
find_package(OpenCV REQUIRED)


set(source_file test_stereo_banet2d.cpp)

include_directories(
    include
#    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
#    ${CMAKE_SOURCE_DIR}/easy_deploy_tool/deploy_core/include
#    ${CMAKE_SOURCE_DIR}/easy_deploy_tool/deploy_utils
#    ${CMAKE_SOURCE_DIR}/easy_deploy_tool/deploy_utils/common_utils/include
#    ${CMAKE_SOURCE_DIR}/easy_deploy_tool/deploy_utils/image_processing_utils/include
#    ${CMAKE_SOURCE_DIR}/easy_deploy_tool/deploy_utils/test_utils/include
#    ${CMAKE_SOURCE_DIR}/easy_deploy_tool/inference_core/ort_core/include
)

#add_library(${PROJECT_NAME} SHARED ${source_file})


add_executable(test_stereo_banet ${source_file})

#target_include_directories(test_stereo_lightstereo PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(test_stereo_banet PUBLIC

#        GTest::gtest_main
#        glog::glog
        ${OpenCV_LIBS}
        deploy_core
        image_processing_utils
#        common_utils
        deploy
#        test_utils
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

