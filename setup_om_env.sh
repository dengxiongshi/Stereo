#!/usr/bin/env bash
# 用法（必须 source，否则 export 不会留在当前终端）:
#   source ./scripts/setup_om_env.sh
# 或:
#   . ./scripts/setup_om_env.sh

set -e

# 定位仓库根目录（支持从任意 cwd source）
if [[ -n "${BASH_SOURCE[0]}" ]]; then
  _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  _SCRIPT_DIR="$(pwd)"
fi

# 若脚本在 scripts/ 下，根目录是上一级；若在仓库根，则是当前
if [[ -d "${_SCRIPT_DIR}/build" ]] || [[ -d "${_SCRIPT_DIR}/deploy" ]]; then
  REPO_ROOT="${_SCRIPT_DIR}"
elif [[ -d "${_SCRIPT_DIR}/../deploy" ]]; then
  REPO_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
else
  echo "[ERROR] 找不到仓库根目录，请在 lightstereo_onnx 仓库内执行"
  return 1 2>/dev/null || exit 1
fi

echo "[INFO] REPO_ROOT=${REPO_ROOT}"

# ---------- Ascend / CANN ----------
# _ASCEND_ENV_CANDIDATES=(
#   "/usr/local/Ascend/ascend-toolkit/set_env.sh"
#   "/usr/local/Ascend/nnrt/set_env.sh"
#   "/usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash"
#   "${HOME}/Ascend/ascend-toolkit/set_env.sh"
# )

# _ASCEND_SOURCED=0
# for _f in "${_ASCEND_ENV_CANDIDATES[@]}"; do
#   if [[ -f "${_f}" ]]; then
#     # shellcheck disable=SC1090
#     source "${_f}"
#     echo "[INFO] sourced Ascend env: ${_f}"
#     _ASCEND_SOURCED=1
#     break
#   fi
# done
# if [[ "${_ASCEND_SOURCED}" -eq 0 ]]; then
#   echo "[WARN] 未找到 Ascend set_env.sh，将仅使用工程内 ascend-aarch64/lib"
# fi

# ---------- 工程库路径 ----------
_LIB_DIRS=(
  "${REPO_ROOT}/build/deploy"
  "${REPO_ROOT}/build/easy_deploy_tool/inference_core/om_core"
  "${REPO_ROOT}/build/easy_deploy_tool/deploy_core"
  "${REPO_ROOT}/build/easy_deploy_tool/deploy_utils/common_utils"
  "${REPO_ROOT}/build/easy_deploy_tool/deploy_utils/image_processing_utils"
  "${REPO_ROOT}/deploy/lib/opencv"
  "${REPO_ROOT}/easy_deploy_tool/inference_core/om_core/ascend-aarch64/lib"
)

_NEW_PATH=""
for _d in "${_LIB_DIRS[@]}"; do
  if [[ -d "${_d}" ]]; then
    _NEW_PATH="${_NEW_PATH}:${_d}"
  else
    echo "[WARN] 目录不存在: ${_d}"
  fi
done

# 去掉开头多余的 :
_NEW_PATH="${_NEW_PATH#:}"
export LD_LIBRARY_PATH="${_NEW_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "[INFO] LD_LIBRARY_PATH 已更新"

# ---------- 可选：在 build/bin 下链模型和图片 ----------
# _BIN_DIR="${REPO_ROOT}/build/bin"
# if [[ -d "${_BIN_DIR}" ]]; then
#   mkdir -p "${_BIN_DIR}"
#   # 模型：代码里是 lightstereo-kittl.om，仓库文件可能是 lightstereo-kitti.om
#   if [[ -f "${REPO_ROOT}/weights/BANet-kittl.om" ]]; then
#     ln -sfn "${REPO_ROOT}/weights/BANet-kittl.om" "${_BIN_DIR}/BANet-kittl.om"
#   fi
#   if [[ -f "${REPO_ROOT}/weights/lightstereo-kittl.om" ]]; then
#     ln -sfn "${REPO_ROOT}/weights/lightstereo-kittl.om" "${_BIN_DIR}/lightstereo-kittl.om"
#   elif [[ -f "${REPO_ROOT}/weights/lightstereo-kitti.om" ]]; then
#     ln -sfn "${REPO_ROOT}/weights/lightstereo-kitti.om" "${_BIN_DIR}/lightstereo-kittl.om"
#     echo "[INFO] 已将 lightstereo-kitti.om 链接为 lightstereo-kittl.om"
#   fi
#   for _img in left.png right.png; do
#     if [[ -f "${REPO_ROOT}/deploy/data/${_img}" ]]; then
#       ln -sfn "${REPO_ROOT}/deploy/data/${_img}" "${_BIN_DIR}/${_img}"
#     elif [[ -f "${REPO_ROOT}/data/${_img}" ]]; then
#       ln -sfn "${REPO_ROOT}/data/${_img}" "${_BIN_DIR}/${_img}"
#     fi
#   done
# fi

# ---------- 依赖检查 ----------
_check_bin() {
  local bin="$1"
  if [[ ! -x "${bin}" ]]; then
    echo "[WARN] 可执行文件不存在: ${bin}（请先拷贝交叉编译产物到 build/）"
    return
  fi
  echo "[INFO] 检查: ${bin}"
  if command -v ldd >/dev/null 2>&1; then
    local missing
    missing="$(ldd "${bin}" 2>/dev/null | grep 'not found' || true)"
    if [[ -n "${missing}" ]]; then
      echo "[ERROR] 仍有缺失库:"
      echo "${missing}"
    else
      echo "[INFO] ${bin##*/} 依赖检查通过"
    fi
  fi
}

_check_bin "${REPO_ROOT}/build/bin/test_stereo_banet"
_check_bin "${REPO_ROOT}/build/bin/test_stereo_lightstereo"

echo "[INFO] 环境配置完成。运行示例:"
echo "  cd ${REPO_ROOT}/build/bin"
echo "  ./test_stereo_banet"
echo "  ./test_stereo_lightstereo"

unset _SCRIPT_DIR _f _ASCEND_SOURCED _ASCEND_ENV_CANDIDATES
unset _LIB_DIRS _d _NEW_PATH _BIN_DIR _img