SET ( CMAKE_SYSTEM_NAME Linux )
SET ( CMAKE_SYSTEM_PROCESSOR aarch64 )
# when hislicon SDK was installed, toolchain was installed in the path as below:
SET ( CMAKE_C_COMPILER "/usr/bin/aarch64-linux-gnu-gcc" )
SET ( CMAKE_CXX_COMPILER "/usr/bin/aarch64-linux-gnu-g++" )
SET ( CMAKE_FIND_ROOT_PATH "/usr/bin") 

# set searching rules for cross-compiler
SET ( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER )
SET ( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
SET ( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )