set -e
export LANG=C
BUILD_DIR=build/
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}

if [ "$1" == "om" ];then
    echo "build ascend platform"
    cmake -DCMAKE_TOOLCHAIN_FILE=./toolchains/ascend.toolchain.cmake -DENABLE_OM=ON ..
elif [ "$1" == "ort" ];then
    echo "build ort platform"
    cmake -DENABLE_ORT=ON ..
elif [ "$1" == "trt" ]; then
    echo "build tensorrt platform"
    cmake -DENABLE_TENSORRT=ON ..
elif [ "$1" == "rknn" ]; then
    echo "build rknn platform"
    cmake -DENABLE_RKNN=ON ..
fi

# shellcheck disable=SC2046
make -j$(nproc)