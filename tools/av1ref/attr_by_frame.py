#!/usr/bin/env python3
"""把像素差异**归因到具体的块**（按显示序对齐）。

## 为什么需要它
`AV1_TRACE_BLKATTR` 的块轨迹是**解码序**，而落盘的帧是**显示序** —— B 金字塔下
两者下标不相等（`--count=3` 会产生 8 段块轨迹）。直接按下标对，归因结果是错的
（2026-10-01 踩过：据此得出的"差异全在 skip 块"是假的）。

正确做法：块轨迹每行带 `oh=<order_hint>` 与 `sf=<show_frame>`，
**显示帧 j ↔ order_hint == j 的那一段**。另外 `show_existing_frame` 的显示帧
**没有瓦片数据**（不在轨迹里），它的内容就是更早解码的那个隐藏帧，
而那个隐藏帧的 order_hint 正好等于显示序号 —— 所以按 order_hint 找段是对的。

用法：

    attr_by_frame.py [--frames N] [--dump DIR]

输出每个显示帧的差异像素数、所属块的属性组合（`inter/skip/ref0/ref1/滤波/运动模式/
复合类型/comp_group_idx`）、块内相对位置，以及首个差异像素的两边取值。
"""
import argparse
import os
import re
import subprocess
from collections import Counter

import numpy as np

W, H = 1920, 1080
CW, CH = W // 2, H // 2
PER = W * H * 3 // 2

W4 = [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16]
H4 = [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4]

PAT = re.compile(
    r"BLKATTR#(\d+) r=(-?\d+) c=(-?\d+) bs=(\d+) inter=(\d+) skip=(\d+) "
    r"ref=(-?\d+),(-?\d+) mv=(-?\d+),(-?\d+),(-?\d+),(-?\d+) if=(\d+),(\d+) "
    r"tx=(\d+) mm=(\d+) ct=(\d+) cg=(\d+) ii=(\d+) oh=(\d+) sf=(\d+)")


def planes_i420(buf):
    c = W * H // 4
    return (buf[:W * H].reshape(H, W),
            buf[W * H:W * H + c].reshape(CH, CW),
            buf[W * H + c:W * H + 2 * c].reshape(CH, CW))


def planes_nv12(buf):
    uv = buf[W * H:W * H + W * H // 2].reshape(CH, W)
    return buf[:W * H].reshape(H, W), uv[:, 0::2], uv[:, 1::2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--dump", default="/tmp/lerxu-attr")
    ap.add_argument("--mp4", default="/tmp/a.mp4")
    ap.add_argument("--ivf", default="/tmp/asmr.ivf")
    ap.add_argument("--aomdec", default="/tmp/libaom-3.9.0/build_g/aomdec")
    ap.add_argument("--engine", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../target/release/zuvrust"))
    a = ap.parse_args()
    os.makedirs(a.dump, exist_ok=True)

    env = dict(os.environ)
    env["AV1_TRACE_BLKATTR"] = "1"
    with open(f"{a.dump}/blk.txt", "w") as err:
        subprocess.run([a.engine, "decode", a.mp4, f"--count={a.frames}",
                        f"--out={a.dump}/my", "--json"],
                       env=env, check=True, stdout=subprocess.DEVNULL, stderr=err)
    subprocess.run([a.aomdec, "--threads=1", "--row-mt=0", "--i420", "--rawvideo",
                    f"--limit={a.frames}", "-o", f"{a.dump}/ref.yuv", a.ivf],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 按 (order_hint, show_frame) 的连续段切分
    segs = []
    for line in open(f"{a.dump}/blk.txt", encoding="utf-8", errors="replace"):
        m = PAT.search(line)
        if not m:
            continue
        row = [int(g) for g in m.groups()]
        key = (row[19], row[20])
        if not segs or segs[-1][0:2] != list(key):
            segs.append([key[0], key[1], []])
        segs[-1][2].append(row)
    by_oh = {}
    for i, s in enumerate(segs):
        by_oh.setdefault(s[0], i)
    print(f"块轨迹 {len(segs)} 段，覆盖 order_hint: {sorted(by_oh)[:20]}")

    ref = np.fromfile(f"{a.dump}/ref.yuv", dtype=np.uint8)
    for j in range(a.frames):
        if j not in by_oh:
            print(f"  显示帧{j}: 轨迹里没有对应段")
            continue
        blocks = segs[by_oh[j]][2]
        cover = np.full((H // 4, W // 4), -1, dtype=np.int32)
        for bi, b in enumerate(blocks):
            r, c, bs = b[1], b[2], b[3]
            cover[r:r + H4[bs], c:c + W4[bs]] = bi
        rp = planes_i420(ref[j * PER:(j + 1) * PER])
        mp = planes_nv12(np.fromfile(f"{a.dump}/my/frame-{j + 1:06}.nv12",
                                     dtype=np.uint8))
        for pi, name in ((0, "Y"), (1, "U"), (2, "V")):
            sub = 0 if pi == 0 else 1
            d = mp[pi].astype(int) - rp[pi].astype(int)
            ys, xs = np.nonzero(d)
            if len(ys) == 0:
                continue
            cfg = Counter()
            for cy, cx in zip(ys, xs):
                bi = cover[cy >> sub, cx >> sub]
                if bi < 0:
                    cfg["<无块>"] += 1
                    continue
                b = blocks[bi]
                cfg[(b[4], b[5], b[6], b[7], b[12], b[13], b[15], b[16], b[17])] += 1
            print(f"  显示帧{j}（段#{by_oh[j]}，oh={segs[by_oh[j]][0]}）{name}: "
                  f"{len(ys)} 个差异  首个 (x={int(xs[0])}, y={int(ys[0])}) "
                  f"参考={int(rp[pi][ys[0], xs[0]])} 我们={int(mp[pi][ys[0], xs[0]])}")
            for k, v in cfg.most_common(5):
                print(f"      (inter,skip,ref0,ref1,ifY,ifX,mm,ct,cg)={k} → {v}")


if __name__ == "__main__":
    main()
