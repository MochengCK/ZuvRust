#!/usr/bin/env python3
"""审 `above_need` 够不够 —— row-mt 跨行依赖的"预计算上界 vs 实际读到的列"。

## 为什么需要它

row-mt 的回放阶段要把"某个块要读上一行的哪些列"**在回放之前**算出来
（`block::RowMtJobs::precompute_dec_flags` → `BlockJob::above_need`），因为回放
（尤其并行回放）里不能依赖"谁先标记"。这个上界算小了就会读到还没重建的像素 ——
症状是并行版与顺序版不逐字节一致，而且**很难从像素反推**（错误会沿预测链扩散）。

这个脚本把两侧的轨迹对齐：

* `ROWB` 行：**预计算**的每块上界（`above_need`，单位 = 亮度 mi 列，不含）
* `ROWD2` 行：帧内 `derive_edges` **实际**读到上一行的最远列（`hi`，单位 = 本平面 4×4）

`ROWD2` 行带块序号 `i=`（由 `replay_one`/`replay_blocks_with` 写 `CUR_BLOCK_IDX`），
所以对齐是**精确**的 —— 前提是按顺序回放采集（`ME_ROW_MT=1`，单线程）。

## 用法

```sh
ME=/path/to/zuvrust
ME_ROW_MT=1 AV1_TRACE_ROWNEED=1 $ME decode in.mp4 --count=4 --out=/tmp/rn --json >/dev/null 2>/tmp/rn.log
python rowneed.py /tmp/rn.log
```

**判据**：`上界不足的块 = 0` 才合格。非零就说明 `above_need_mi`（或它吃的
`DecFlags::above_right`）漏了东西，并行回放必然读到未重建像素。

⚠️ 这只是"上界够不够"的审计，**不能**替代 `ME_TRACE_RACE=1` 探针：
探针查的是"读的那一刻那格有没有写完"（能抓调度漏依赖），本脚本查的是
"算出来的上界够不够"（能抓公式错）。两者都为零才算干净。
"""

import argparse
import collections
import re
import sys

ROWB = re.compile(r"^ROWB i=(\d+) r=(-?\d+) c=(-?\d+) bs=(\d+) inter=(\d+) ii=(\d+) need=(\d+)")
ROWD2 = re.compile(r"^ROWD2 i=(\d+) p=(\d+) r4=(-?\d+) c4=(-?\d+) w4=(-?\d+) ar=(\d+) hi=(-?\d+)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="带 AV1_TRACE_ROWNEED=1 的 stderr 日志")
    ap.add_argument("--top", type=int, default=15, help="打印多少条不足样例（默认 15）")
    args = ap.parse_args()

    # ⚠️ 逐**帧**分批：日志里会解出多帧（含隐藏的 alt-ref），每帧的块序号都从 0 开始，
    # 混在一起比会跨帧撞车（踩过：把 2358 个"上界不足"全算成了帧间块，纯属错配）。
    # 分批标志 = 又一次遇到 `ROWB i=0`。
    batches: list[tuple[dict[int, dict], dict[int, int]]] = []
    blocks: dict[int, dict] = {}
    required: dict[int, int] = {}
    reads = 0
    misattributed = 0

    def flush() -> None:
        if blocks:
            batches.append((dict(blocks), dict(required)))

    with open(args.log, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = ROWB.match(line)
            if m:
                i = int(m.group(1))
                if i == 0:
                    flush()
                    blocks.clear()
                    required.clear()
                blocks[i] = dict(r=int(m.group(2)), c=int(m.group(3)), bs=int(m.group(4)),
                                 inter=int(m.group(5)), ii=int(m.group(6)), need=int(m.group(7)))
                continue
            m = ROWD2.match(line)
            if m:
                reads += 1
                i = int(m.group(1))
                plane = int(m.group(2))
                hi = int(m.group(6))
                if i == -1:
                    continue
                # 平面 4×4 → 亮度 mi 列：4:2:0 的色度一个 4×4 格 = 2 个亮度 mi 列
                luma_hi = hi if plane == 0 else 2 * hi
                req = luma_hi + 1  # 不含上界
                # 位置一致性：ROWD2 的 (r4,c4) 必须落在它声称的那个块的范围内。
                # 落不进去 = 归因错了（典型来源：**语法阶段的调色板 flush** 会调
                # `predict_intra`，那时 `CUR_BLOCK_IDX` 还是上一帧回放留下的值）。
                b = blocks.get(i)
                if b is not None:
                    W4 = [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16]
                    H4 = [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4]
                    sub = 0 if plane == 0 else 1
                    r0, c0 = b["r"] >> sub, b["c"] >> sub
                    ok = (r0 <= int(m.group(3)) < r0 + max(1, H4[b["bs"]] >> sub)
                          and c0 <= int(m.group(4)) < c0 + max(1, W4[b["bs"]] >> sub))
                    if not ok:
                        misattributed += 1
                        continue
                    if req > required.get(i, 0):
                        required[i] = req
    flush()

    if not batches:
        print("日志里没有 ROWB 行 —— 确认设了 AV1_TRACE_ROWNEED=1 且 stdout/stderr 重定向正确", file=sys.stderr)
        return 2

    bad: list[tuple[int, dict, int]] = []
    nblocks = 0
    for bl, rq in batches:
        nblocks += len(bl)
        bad.extend((i, b, rq[i]) for i, b in bl.items() if rq.get(i, 0) > b["need"])
    print(f"解码帧 {len(batches)} 个、块 {nblocks} 个，跨行读记录 {reads} 条"
          + (f"（其中 {misattributed} 条位置对不上、已剔除 —— 多半是语法阶段调色板 flush 的残留归属）"
             if misattributed else ""))
    print(f"**上界不足的块 {len(bad)} 个**" + ("（合格 ✓）" if not bad else "（不合格 ✗）"))
    if bad:
        print("  差额分布 (required - need):",
              collections.Counter(req - b["need"] for _, b, req in bad).most_common(10))
        print("  类型分布 (bs, inter, ii):",
              collections.Counter((b["bs"], b["inter"], b["ii"]) for _, b, _ in bad).most_common(10))
        print(f"  样例（i, mi=(r,c), bs, inter, ii, need, required）前 {args.top} 条：")
        for i, b, req in bad[: args.top]:
            print(f"    i={i} mi=({b['r']},{b['c']}) bs={b['bs']} inter={b['inter']} ii={b['ii']} "
                  f"need={b['need']} required={req}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
