cmake_minimum_required(VERSION 3.8)
project(deploy)

add_compile_options(-std=c++17)
add_compile_options(-O3 -Wextra -Wdeprecated -fPIC)
set(CMAKE_CXX_STANDARD 17)

if (ENABLE_OM)
    set(OpenCV_INCLUDE_DIRS
        ${PROJECT_SOURCE_DIR}/include/opencv
        CACHE PATH "OpenCV include dir (local in deploy)" FORCE
    )
    file(GLOB OpenCV_LIBS ${PROJECT_SOURCE_DIR}/lib/opencv/*.so)
    if (NOT OpenCV_LIBS)
        message(FATAL_ERROR "ENABLE_OM=ON时，deploy/lib/opencv/下未找到.so库！")
    endif()
    # 设为缓存变量，供其他子目录复用
    set(OpenCV_LIBS ${OpenCV_LIBS} CACHE FILEPATH "OpenCV libs (local in deploy)" FORCE)
else()
    find_package(OpenCV REQUIRED)
    set(OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS} CACHE PATH "OpenCV include dir (system)" FORCE)
    set(OpenCV_LIBS ${OpenCV_LIBS} CACHE FILEPATH "OpenCV libs (system)" FORCE)
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
        ${OpenCV_LIBS}
        deploy_core
        image_processing_utils
        )

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(banet)
add_subdirectory(lightstereo)