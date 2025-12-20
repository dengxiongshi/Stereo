cmake_minimum_required(VERSION 3.8)
project(om_core)

add_compile_options(-std=c++17)
add_compile_options(-O3 -Wextra -Wdeprecated -fPIC)
set(CMAKE_CXX_STANDARD 17)

message(STATUS "Building Om-Core on aarch64 platform ...")
set(ASCEND_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/ascend-aarch64/include)
 FILE(GLOB ASCEND_LIBS ${PROJECT_SOURCE_DIR}/ascend-aarch64/lib/*.so)

#set(LIB_PATH
#/usr/local/Ascend/ascend-toolkit/7.0.0/aarch64-linux/lib64
## /usr/local/Ascend/ascend-toolkit/latest/runtime/lib64
## /usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub
## /usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub/aarch64
#)
#FILE(GLOB ASCEND_LIBS ${LIB_PATH}/*.so)


set(source_file
        src/om_core.cpp
        src/om_core_factory.cpp
        )

include_directories(
        include
        ${ASCEND_INCLUDE_DIRS}
)

link_directories(${LIB_PATH})

add_library(${PROJECT_NAME} SHARED ${source_file})

target_link_libraries(${PROJECT_NAME} PUBLIC
        ${ASCEND_LIBS}
        deploy_core
)

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
