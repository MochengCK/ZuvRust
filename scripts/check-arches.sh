#!/usr/bin/env bash
#
# 用「**非本机架构** + `--all-targets`」把核心层编译一遍。
#
# ## 为什么需要它
#
# 这个引擎有大量 `#[cfg(target_arch = "x86_64")]` 的 SIMD 实现，以及与之配对的
# `#[cfg(not(target_arch = "x86_64"))]` **桩**。桩不参与 x86 构建 —— 也就是说，
# **在 Intel 机器上，桩写错了你永远看不见**：
#
#   · 少写一个桩       → arm64 报 "cannot find function"（2026-10-04 一次漏了 7 个）
#   · 桩的签名抄错了   → arm64 编译**测试目标**时才知道（2026-10-05 一次：桩写成
#                        `&mut [i32]`，真实内核要 `&mut [i16]`，Intel 上一切正常）
#
# 两类都是"只有别的架构才编译到"的错误，而它们**都是编译错误** —— 所以不需要
# 写任何解析器：让编译器在另一个架构上跑一遍就够，而且比任何自检脚本都准。
#
# `--all-targets` 不能省：漏掉它就编译不到 `#[cfg(test)]` 里的调用点，
# 而那恰恰是上面第二类错误的藏身处。
#
# ## 用法
#
#   ./scripts/check-arches.sh              # 检查下面 TARGETS 里所有已安装 std 的目标
#   ./scripts/check-arches.sh <target>...  # 只检查指定的目标
#
# ## 为什么只查核心层
#
# `me-net` 的 TLS 用 ring，ring 的构建脚本需要**目标平台的 C 工具链**，交叉时
# 拿不到 ⇒ 只要依赖它（me-net / me-stream / me-cli）就无法在这里交叉检查。
# 那三个 crate 由 CI 的**原生运行器**覆盖（每个平台自己跑一遍全量测试），
# 反而比交叉编译更可信。这里覆盖的是"没被原生覆盖的那部分 + 最容易漂移的 SIMD 层"。
set -euo pipefail

# ⚠️ 变量后面紧跟中文标点时必须写 `${t}`：UTF-8 环境下 bash 会把全角标点的
#    字节也算进变量名，`"$t：..."` 会被解析成引用变量 `t：` ⇒ unbound variable。

cd "$(dirname "$0")/.."

# 不依赖 ring 的 5 个 crate（覆盖全部 SIMD / 解码 / 容器 / BT 逻辑）。
CRATES=(-p me-core -p me-container -p me-mux -p me-decode -p me-bt)

# 宿主真正要发布的架构。装了 std 才会检查，没装就跳过并提示怎么装。
DEFAULT_TARGETS=(
  aarch64-apple-darwin        # Apple Silicon
  x86_64-apple-darwin         # Intel Mac
  x86_64-pc-windows-msvc      # Windows
  x86_64-unknown-linux-gnu    # Linux x64
  aarch64-unknown-linux-musl  # Linux arm64（静态）
)

if [ "$#" -gt 0 ]; then
  TARGETS=("$@")
else
  TARGETS=("${DEFAULT_TARGETS[@]}")
fi

INSTALLED="$(rustup target list --installed)"
HOST="$(rustc -vV | sed -n 's/^host: //p')"
echo "本机: $HOST"

checked=0
skipped=0
for t in "${TARGETS[@]}"; do
  # 本机架构由日常的 `cargo test` 覆盖，这里不必重复（也不必交叉到自己）。
  if [ "${t}" = "${HOST}" ]; then
    echo "── ${t}：跳过（就是本机）"
    continue
  fi
  if ! printf '%s\n' "$INSTALLED" | grep -qx "${t}"; then
    echo "── ${t}：跳过（未安装 std；装它：rustup target add ${t}）"
    skipped=$((skipped + 1))
    continue
  fi
  echo "── ${t}"
  cargo check --all-targets "${CRATES[@]}" --target "${t}"
  checked=$((checked + 1))
done

echo
if [ "$checked" -eq 0 ]; then
  echo '⚠️  一个目标都没检查（全被跳过了）。'
  echo '    这类检查专门用来拦住「只有别的架构才编译到」的错误，一个都没跑等于没查。'
  echo '    至少装一个非本机目标：rustup target add aarch64-apple-darwin'
  exit 1
fi
echo "✅ $checked 个非本机架构通过（跳过 $skipped 个）"
