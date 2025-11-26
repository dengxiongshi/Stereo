cmake_minimum_required(VERSION 3.24)
project(deploy)

add_compile_options(-std=c++17)
add_compile_options(-O3 -Wextra -Wdeprecated -fPIC)
set(CMAKE_CXX_STANDARD 17)

if (ENABLE_OM)
    set(OpenCV_INCLUDE_DIRS include/opencv)
    set(OpenCV_LIBS lib/opencv/*.so)
else()
    find_package(OpenCV REQUIRED)
#    find_package(glog REQUIRED)
endif()


include_directories(
        include
        ${OpenCV_INCLUDE_DIRS}
)

set(source_file
    src/lightstereo.cpp
    src/banet.cpp
)

add_library(${PROJECT_NAME} SHARED ${source_file})

target_link_libraries(${PROJECT_NAME} PUBLIC
#        glog::glog
        ${OpenCV_LIBS}
        deploy_core
        image_processing_utils
        )

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(banet)
add_subdirectory(lightstereo)