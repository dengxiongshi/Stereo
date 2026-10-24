cmake_minimum_required(VERSION 3.10)
project(om_core)

add_compile_options(-std=c++17)
add_compile_options(-O3 -Wextra -Wdeprecated -fPIC)
set(CMAKE_CXX_STANDARD 17)

message(STATUS "Building Om-Core on aarch64 platform ...")
set(ASCEND_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/ascend-aarch64/include)

if(DEFINED ENV{ASCEND_HOME})
    set(ASCEND_LIB_DIR $ENV{ASCEND_HOME}/lib64)
elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_LIB_DIR $ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/lib64)
else()
#     set(ASCEND_LIB_DIR /usr/local/Ascend/ascend-toolkit/7.0.0/aarch64-linux/lib64)
    set(ASCEND_LIB_DIR ${PROJECT_SOURCE_DIR}/ascend-aarch64/lib)
endif()

message(STATUS "Ascend lib dir: ${ASCEND_LIB_DIR}")

set(source_file
        src/om_core.cpp
        src/om_core_factory.cpp
        )

include_directories(
        include
        ${ASCEND_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED ${source_file})

target_link_directories(${PROJECT_NAME} PUBLIC ${ASCEND_LIB_DIR})

target_link_libraries(${PROJECT_NAME} PUBLIC
        ascendcl
        deploy_core
)

install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include
        PRIVATE ${ASCEND_INCLUDE_DIRS})
