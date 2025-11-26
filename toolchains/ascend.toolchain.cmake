SET ( CMAKE_SYSTEM_NAME Linux )
SET ( CMAKE_SYSTEM_PROCESSOR aarch64 )
# when hislicon SDK was installed, toolchain was installed in the path as below:
SET ( CMAKE_C_COMPILER "/opt/toolchain/7.3.0/toolchain/bin/aarch64-target-linux-gnu-gcc" )
SET ( CMAKE_CXX_COMPILER "/opt/toolchain/7.3.0/toolchain/bin/aarch64-target-linux-gnu-g++" )
SET ( CMAKE_FIND_ROOT_PATH "/opt/toolchain/7.3.0/toolchain" )

# set searching rules for cross-compiler
SET ( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER )
SET ( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
SET ( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )