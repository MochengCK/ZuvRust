#!/usr/bin/env bash
#
# 验收**打包好的产物**：「这个要发出去的二进制，本身能不能用」。
#
# ## 为什么要有这一步
#
# `cargo test` 验的是**源码树**，产物验收验的是**二进制**。两者之间夹着打包：
# 打错文件、漏拷 LICENSE/README、tar 损坏、架构与标注不符、下载后被加上
# quarantine 标记 —— 这些都不会被任何单元测试发现，但都会让用户拿到一个坏包。
#
# ## 用法
#
#   ./scripts/verify-artifact.sh <包.tar.gz> <包内二进制名> [期望架构关键字]
#
# 例：
#   ./scripts/verify-artifact.sh zuvrust-darwin-x86_64.tar.gz zuvrust "Mach-O 64-bit executable x86_64"
#
# 不传架构关键字时跳过架构断言（仍然会尝试运行它 —— 架构不对的话运行会直接失败，
# 只是错误信息不如 `file` 明确）。
#
# ## 注意
#
# 第 ⑤ 步会调用 `cargo test`，所以要在**仓库根**跑（它会去读 crates/）。
# 这需要 Rust 工具链；CI 里那一步本来就有。
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

TARBALL="${1:?用法: verify-artifact.sh <包.tar.gz> <包内二进制名> [期望架构关键字]}"
BINNAME="${2:?缺少包内二进制名（如 zuvrust / zuvrust.exe）}"
EXPECT_ARCH="${3:-}"

[ -f "$TARBALL" ] || { echo "!! 找不到产物包：$TARBALL"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/zuvrust-verify.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

echo "── ① 解包"
tar -xzf "$TARBALL" -C "$WORK"
ls -la "$WORK"

BIN="$WORK/$BINNAME"
[ -f "$BIN" ] || { echo "!! 包里没有 ${BINNAME}（打错文件了？）"; exit 1; }
chmod +x "$BIN" 2>/dev/null || true

echo "── ② 架构"
file "$BIN"
if [ -n "$EXPECT_ARCH" ]; then
  if ! file "$BIN" | grep -q "$EXPECT_ARCH"; then
    echo "!! 架构与标注不符：期望包含 '$EXPECT_ARCH'"
    echo "   打错架构的产物在别的机器上只会给出 \"Exec format error\"，没有信息量 —— 这里提前拦住。"
    exit 1
  fi
fi

echo "── ③ 包内容"
for f in LICENSE README.md; do
  [ -f "$WORK/$f" ] || { echo "!! 包里缺少 $f"; exit 1; }
done

echo "── ④ 能起来吗（顺带确认自报身份）"
"$BIN" version --json

echo "── ⑤ 用产物本身跑契约测试（ZUVRUST_BIN 指向它）"
# 这一步把「被测对象」从 cargo 构建的那个换成**解包出来的这个**。
# 契约测试真的 spawn 进程，所以它验的就是手上这个二进制。
ZUVRUST_BIN="$BIN" cargo test -p me-cli --test cli_contract

echo
echo "✅ 产物验收通过：$TARBALL"
