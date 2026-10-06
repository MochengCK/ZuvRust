#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
# PGO（profile-guided optimization）三趟构建 —— 2026-10-06 第四十二刀
#
# ## 为什么值得
#
# 实测（CPU 时间 / 100 帧 / 交替 A-B / 配对中位）：**−3.9%，24 轮里 21 轮更快**，
# 且输出与不开 PGO 的构建**逐字节相同**（PGO 只改分支布局与内联决策，不改语义）。
#
# 在这个仓库里 PGO 特别划算，因为热路径的形状是**高度倾斜的**：
#   * `read_symbol_inner` / `decode_partition` / `decode_block` 这类
#     "读一个符号 → 一大串位域判断" 的代码，冷分支远比热分支多；
#   * 环路滤波与 MC 的驱动层有大量 `filter_size == 4 / >= 8 / >= 16`、
#     `plane == 0` 之类的**几乎恒定**的判断 —— PGO 会把它们折成直线码。
#
# `lto = "thin"` + `codegen-units = 1` 已经在 `Cargo.toml` 里了（那些是"不跑训练
# 也能拿"的收益）；PGO 是它们之上再叠一层。
#
# ## 怎么用
#
#   tools/pgo.sh <训练素材>              # 例如 tools/pgo.sh /tmp/a.mp4
#   PGO_FRAMES=3000 tools/pgo.sh foo.mp4 # 加大训练帧数（默认 1500）
#
# 结果会被**复制到 `target/release/zuvrust`**（覆盖普通构建的产物），
# 所以 `cargo build --release` 之后想恢复 PGO 版，重跑本脚本即可。
#
# ⚠️ **改了热路径代码之后要重跑**：profile 记的是"上一版代码的执行频次"，
# 代码结构变化大了以后旧 profile 的价值会衰减（不会算错，只会少赚）。
#
# ⚠️ 训练素材要**有代表性**：用一段真实 1080p AV1（最好含 B 帧 / 复合预测 /
# 各种块尺寸）。只用短素材训练会把 profile 压偏到那一种码流上。
# ─────────────────────────────────────────────────────────────────────────────
set -eu

INPUT=${1:-}
if [ -z "$INPUT" ]; then
    echo "用法: tools/pgo.sh <训练素材(.mp4/.ivf/.obu)…>" >&2
    echo "  可用环境变量：PGO_FRAMES（默认 1500）、PGO_DIR（默认 target/pgo-data）" >&2
    exit 2
fi
if [ ! -f "$INPUT" ]; then
    echo "训练素材不存在: $INPUT" >&2
    exit 2
fi

FRAMES=${PGO_FRAMES:-1500}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

DATA=${PGO_DIR:-$ROOT/target/pgo-data}
INSTR=$ROOT/target/pgo-instr
FINAL=$ROOT/target/pgo

PROFDATA=$(xcrun -f llvm-profdata 2>/dev/null || command -v llvm-profdata || true)
if [ -z "$PROFDATA" ]; then
    echo "找不到 llvm-profdata（装 Xcode 命令行工具：xcode-select --install）" >&2
    exit 2
fi

echo "== 1/4 插桩构建（-$ROOT/target/pgo-instr）=="
rm -rf "$DATA"
mkdir -p "$DATA"
RUSTFLAGS="-Cprofile-generate=$DATA" \
    CARGO_TARGET_DIR="$INSTR" \
    cargo build --release -p me-cli

# ⚠️ `$VAR` 后面紧跟**多字节字符**时一定要写成 `${VAR}`：本机 zsh 会把全角字符
# 当成变量名的一部分，`set -u` 下直接报 "unbound variable"（第一次跑就踩了）。
echo "== 2/4 跑训练负载（${FRAMES} 帧 × ${INPUT}）=="
ME_THREADS=1 "$INSTR/release/zuvrust" decode "$INPUT" --count="$FRAMES" --json >/dev/null

echo "== 3/4 合并 profile =="
# 插桩运行会散落一堆 `.profraw`；`merge` 接受**目录**，会把它们全并进去。
"$PROFDATA" merge -o "$DATA/merged.profdata" "$DATA"

echo "== 4/4 用 profile 重建 =="
RUSTFLAGS="-Cprofile-use=$DATA/merged.profdata -Cllvm-args=-pgo-warn-missing-function" \
    CARGO_TARGET_DIR="$FINAL" \
    cargo build --release -p me-cli

cp "$FINAL/release/zuvrust" "$ROOT/target/release/zuvrust"
echo
echo "完成：$ROOT/target/release/zuvrust 现在是 PGO 版"
echo "  profile : $DATA/merged.profdata"
echo "  校验建议: ME_THREADS=1 zuvrust decode <素材> --count=24 --out=A；"
echo "            与普通构建的同一命令输出做 diff -r（应当逐字节相同）"
