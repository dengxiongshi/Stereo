set -e
export LANG=C

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SRC_DIR}/build"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

if [ "$1" == "om" ]; then
    echo "build ascend platform"
    cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
        -DCMAKE_TOOLCHAIN_FILE="${SRC_DIR}/toolchains/ascend.toolchain.cmake" \
        -DENABLE_OM=ON
elif [ "$1" == "ort" ]; then
    echo "build ort platform"
    cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" -DENABLE_ORT=ON
elif [ "$1" == "trt" ]; then
    echo "build tensorrt platform"
    cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" -DENABLE_TENSORRT=ON
elif [ "$1" == "rknn" ]; then
    echo "build rknn platform"
    cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" -DENABLE_RKNN=ON
else
    echo "用法: $0 om|ort|trt|rknn"
    exit 1
fi

cmake --build "${BUILD_DIR}" -j"$(nproc)"