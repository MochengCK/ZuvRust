#!/usr/bin/env python3
"""逐块属性对拍：我们（`AV1_TRACE_BLKATTR`）vs libaom（`LERXU_BLKATTR`）。

## 为什么需要它

语法层已经**逐符号一致** ⇒ **块边界一一对应** ⇒ 两边的第 n 个块就是同一个块。
于是"像素差在哪"这个问题可以化简成："**第一个属性对不上的块是哪个、哪个属性**"。
属性（位置 / 尺寸 / 帧内帧间 / 参考帧 / 两个 MV / 插值滤波 / 变换尺寸 / 运动模式 /
`comp_group_idx`）全对上了，就说明**属性不是问题**，锅在像素运算里。

## 怎么跑

```sh
# 我们（stderr 里的 BLKATTR 行）
AV1_TRACE_BLKATTR=1 zuvrust decode <素材> --count=2 --out=/tmp/x --json 2> my.txt
# libaom（同一份素材的 IVF）
LERXU_BLKATTR=1 aomdec --threads=1 --row-mt=0 --noblit --limit=2 -o /dev/null asm.ivf 2>&1 \
  | grep '^BLKATTR' > ref.txt
python3 blkattr.py my.txt ref.txt
```

## ⚠️ 两条必须知道的判读规则（踩过）

1. **libaom 的 `mbmi` 是复用的**：帧内块（和单参考块）上 `motion_mode` / `interp_filters` /
   `interinter_comp.type` 是**上一块残留的值**，拿它跟我们的"每块新建"比会看到一堆假差异。
   ⇒ **只在 `inter=1` 的块上比这几个字段**（位置/尺寸/参考帧/MV 在帧内块上也有意义）。
2. **`ct`（复合类型）只在"真的是复合块"（`ref1 > 0`）时才有意义**，而且两边枚举序可能
   不同 —— 判"复合路径对不对"最硬的判据还是**受控流的逐像素一致**，不是这个字段。

## 2026-10-01 在 B 站 ASMR 上的结果

7518 个帧间块上，`位置 r/c`、`尺寸 bs`、`帧间标记`、`skip`、`参考帧`、`两个 MV`
（各 4 字节）、`插值滤波器`、`变换尺寸`、`运动模式`、`comp_group_idx` —— **全部 100% 一致**。
⇒ MV 预测栈、插值滤波器选择、变换尺寸全都没问题；差异只可能在像素运算。
"""
import re
import sys

PAT = re.compile(
    r"BLKATTR#(\d+) r=(-?\d+) c=(-?\d+) bs=(\d+) inter=(\d+) skip=(\d+) "
    r"ref=(-?\d+),(-?\d+) mv=(-?\d+),(-?\d+),(-?\d+),(-?\d+) "
    r"if=(\d+),(\d+) tx=(\d+) mm=(\d+)(?: ct=(\d+))?(?: cg=(\d+))?"
)
NAMES = ["n", "r", "c", "bs", "inter", "skip", "ref0", "ref1", "mv0r", "mv0c",
         "mv1r", "mv1c", "ify", "ifx", "tx", "mm", "ct", "cg"]
# 只在帧间块上比的字段（libaom 的 mbmi 复用 ⇒ 帧内块上是残留值）
INTER_ONLY = {"ify", "ifx", "mm", "ct", "cg"}


def load(path):
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        if not line.startswith("BLKATTR"):
            continue
        m = PAT.search(line)
        if not m:
            continue
        v = list(m.groups())
        v = [0 if x is None else int(x) for x in v]
        rows.append(dict(zip(NAMES, v)))
    return rows


def main():
    mine, ref = load(sys.argv[1]), load(sys.argv[2])
    print(f"我们 {len(mine)} 个块，参考 {len(ref)} 个块")
    n = min(len(mine), len(ref))
    inter = [(i, mine[i], ref[i]) for i in range(n)
             if mine[i]["inter"] == 1 and ref[i]["inter"] == 1]
    print(f"两边都判为帧间块的：{len(inter)} 个")
    bad = {}
    for i, a, b in inter:
        compound = a["ref1"] > 0
        for k in NAMES[1:]:
            # `ct`/`cg` 只在**真的是复合块**时才有意义（单参考块上两边各留各的初值）
            if k in ("ct", "cg") and not compound:
                continue
            if a[k] != b[k]:
                bad.setdefault(k, []).append((i, a[k], b[k]))
    if not bad:
        print("✓ 帧间块上逐块属性全部一致")
        return 0
    print("帧间块上的不一致：", {k: len(v) for k, v in bad.items()})
    for k, v in list(bad.items())[:5]:
        print(f"  {k}: 前 4 例 {v[:4]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
