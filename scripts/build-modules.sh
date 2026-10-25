#!/usr/bin/env bash
#
# 按模块构建媒体引擎：只编应用真正用到的部分。
#
# 模块（逗号分隔，可任意组合）：
#   audio  音频解析/解码（AAC / MP3 / FLAC / Opus）
#   video  视频解析/解码（AV1 自研软解 + H.264 三平台硬解）
#   mux    音视频合并 / 转封装（probe、mux、remux 走这条）
#
# 默认三个全开（等价于 `cargo build`）。桌面端当前只用合并，CI 的产物命令是：
#   ./scripts/build-modules.sh --modules=mux --release
#
# 用法：
#   ./scripts/build-modules.sh [--modules=audio,video,mux|all] [--release]
#                              [--target=<triple>] [--zigbuild]
#
# `--zigbuild` 用 cargo-zigbuild 交叉到 aarch64-linux（zig 兼任 C 编译器与链接器）。
set -euo pipefail

cd "$(dirname "$0")/.."

MODULES="audio,video,mux"
PROFILE="debug"
TARGET=""
ZIGBUILD=0
EXTRA=()

for arg in "$@"; do
  case "$arg" in
    --modules=*) MODULES="${arg#--modules=}" ;;
    --release) PROFILE="release" ;;
    --target=*) TARGET="${arg#--target=}" ;;
    --zigbuild) ZIGBUILD=1 ;;
    -h|--help)
      # 打印本文件开头那段用法（到 `set -euo pipefail` 为止）。
      awk 'NR > 1 && /^set -euo pipefail/ { exit } NR > 1 { print }' "$0"
      exit 0
      ;;
    *)
      echo "未知参数：$arg" >&2
      exit 2
      ;;
  esac
done

if [ "${MODULES}" = "all" ]; then
  MODULES="audio,video,mux"
fi

IFS=',' read -r -a RAW <<< "${MODULES}"
SELECTED=()
if [ "${#RAW[@]}" -gt 0 ]; then
  for m in "${RAW[@]}"; do
    m="${m//[[:space:]]/}"
    [ -n "${m}" ] || continue
    case "${m}" in
      audio|video|mux) ;;
      *)
        echo "未知模块：${m}（只认 audio / video / mux）" >&2
        exit 2
        ;;
    esac
    if [ "${#SELECTED[@]}" -gt 0 ]; then
      for s in "${SELECTED[@]}"; do
        [ "${s}" = "${m}" ] && { echo "模块重复：${m}" >&2; exit 2; }
      done
    fi
    SELECTED+=("${m}")
  done
fi
if [ "${#SELECTED[@]}" -eq 0 ]; then
  echo "至少要选一个模块：audio / video / mux" >&2
  exit 2
fi
FEATURES="$(IFS=','; echo "${SELECTED[*]}")"

if [ -n "${TARGET}" ]; then
  EXTRA+=(--target "${TARGET}")
fi
if [ "${PROFILE}" = "release" ]; then
  EXTRA+=(--release)
fi

if [ "${ZIGBUILD}" = "1" ]; then
  CARGO=(cargo zigbuild)
else
  CARGO=(cargo build)
fi

echo "模块: ${FEATURES}"
echo "构建: ${CARGO[*]} -p me-cli --bin zuvrust --no-default-features --features ${FEATURES} ${EXTRA[*]:-}"
if [ "${#EXTRA[@]}" -gt 0 ]; then
  "${CARGO[@]}" -p me-cli --bin zuvrust --no-default-features --features "${FEATURES}" "${EXTRA[@]}"
else
  "${CARGO[@]}" -p me-cli --bin zuvrust --no-default-features --features "${FEATURES}"
fi

TARGET_DIR="${CARGO_TARGET_DIR:-target}"
if [ -n "${TARGET}" ]; then
  TARGET_DIR="${TARGET_DIR}/${TARGET}"
fi
BIN_EXT=""
case "${TARGET:-${RUNNER_OS:-}}" in
  *windows*|Windows*) BIN_EXT=".exe" ;;
esac
echo "产物: ${TARGET_DIR}/${PROFILE}/zuvrust${BIN_EXT}"
