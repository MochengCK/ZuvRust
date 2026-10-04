#!/usr/bin/env python3
"""把真素材上残留的 ±1 差异**逐点摊开**：精确坐标、聚类、以及落在哪个语法块里。

与 `attr_by_frame.py` 的区别：那个脚本依赖 libaom 的 `aomdec` 与手工 ivf 流程；
这个只依赖 `zuvrust`（NV12 落盘 + `AV1_TRACE_BLKATTR` 轨迹）与 dav1d，
自动化程度更高，且**顺带把"块归属"直接算出来**。

```sh
python3 chroma_probe.py <素材> --bin=<engine> --frames=20
```
"""
import argparse
import os
import re
import subprocess
import sys
from collections import Counter

import numpy as np
import av

# 4x4 块的宽高（block_size 编号 → 4x4 个数），与 me-decode 的 NUM_4X4_BLOCKS_* 同源
W4 = [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16]
H4 = [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4]

PAT = re.compile(
    r"BLKATTR#(\d+) r=(-?\d+) c=(-?\d+) bs=(\d+) inter=(\d+) skip=(\d+) "
    r"ref=(-?\d+),(-?\d+) mv=(-?\d+),(-?\d+),(-?\d+),(-?\d+) if=(\d+),(\d+) "
    r"tx=(\d+) mm=(\d+) ct=(\d+) cg=(\d+) ii=(\d+) oh=(\d+) sf=(\d+)")


def load_ref(path, n):
    c = av.open(path)
    vs = c.streams.video[0]
    w, h = vs.width, vs.height
    out = []
    for fr in c.decode(vs):
        pls = []
        for p in range(3):
            pl = fr.planes[p]
            pls.append(np.frombuffer(pl, dtype=np.uint8)
                       .reshape(pl.height, pl.line_size)[:, :pl.width].copy())
        out.append(pls)
        if len(out) >= n:
            break
    c.close()
    return w, h, out


def load_mine(d, w, h, n):
    cw, ch = (w + 1) // 2, (h + 1) // 2
    ys = w * h
    cs = cw * ch * 2
    out = []
    for i in range(n):
        p = os.path.join(d, f"frame-{i+1:06d}.nv12")
        if not os.path.exists(p):
            break
        b = np.fromfile(p, dtype=np.uint8)
        y = b[:ys].reshape(h, w)
        uv = b[ys:ys + cs].reshape(ch, cw * 2)
        out.append([y, uv[:, 0::2].copy(), uv[:, 1::2].copy()])
    return out


def split_by_oh(log):
    """按 `oh=` 把块轨迹切成"解码单元"，返回 {order_hint: [从该行起连续同 oh 的所有行]}。

    注意 `sf=0` 的隐藏帧也在轨迹里；显示帧 j 取 oh==j 的那一段。
    """
    segs = {}
    cur = None
    for ln in log.splitlines():
        m = PAT.search(ln)
        if not m:
            continue
        oh = int(m.group(20))
        if cur is None or oh != cur:
            cur = oh
            segs.setdefault(cur, [])
        segs[cur].append(m)
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--bin", default="target/release/zuvrust")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--out", default="/tmp/_chroma_probe")
    ap.add_argument("--top", type=int, default=40, help="最多打印多少个差异点")
    a = ap.parse_args()

    w, h, refs = load_ref(a.src, a.frames)
    d = a.out
    subprocess.run(["rm", "-rf", d], check=False)
    env = dict(os.environ, AV1_TRACE_BLKATTR="1")
    p = subprocess.run([a.bin, "decode", a.src, f"--count={a.frames}", f"--out={d}"],
                       env=env, capture_output=True, text=True)
    log = p.stderr
    mine = load_mine(d, w, h, a.frames)

    cw, ch = w // 2, h // 2
    segs = split_by_oh(log)
    # 全部轨迹段摊平：显示帧 j 与 order_hint 不是一回事（B 金字塔下更不是），
    # 所以**不按帧号选段**，而是在所有段里按"包含该亮度坐标"来找块 —— 唯一可靠的做法。
    all_blocks = []
    for oh, seg in sorted(segs.items()):
        for m in seg:
            all_blocks.append((oh, m))

    tot = Counter()
    per_block = Counter()
    printed = 0
    print(f"尺寸 {w}x{h}  帧数 {len(mine)}  轨迹段(oh) {sorted(segs)[:24]}...")
    for fi in range(min(len(mine), len(refs))):
        for pl, name in ((0, "Y"), (1, "U"), (2, "V")):
            # ⚠️ 必须先转成有符号再相减：uint8 直接减会回绕（+1 会打成 +255）
            dm = mine[fi][pl].astype(np.int16) - refs[fi][pl].astype(np.int16)
            idx = np.argwhere(dm != 0)
            if len(idx) == 0:
                continue
            tot[name] += len(idx)
            tot[name + "_maxabs"] = max(tot.get(name + "_maxabs", 0),
                                        int(np.abs(dm).max()))
            for (py, px) in idx:
                delta = int(dm[py, px])
                # 色度坐标 → 亮度坐标（用于找块）
                ly, lx = (py * 2, px * 2) if pl else (py, px)
                oh, blk = find_block(all_blocks, lx, ly)
                if printed < a.top:
                    print(f"  帧{fi:>2} {name} ({px:>4},{py:>4}) 我={mine[fi][pl][py,px]:>3} "
                          f"参={refs[fi][pl][py,px]:>3} Δ={delta:+d}  "
                          f"亮度({lx},{ly}) oh={oh} 块={blk}")
                    printed += 1
                per_block[(pl, blk)] += 1
    print("\n合计：", {k: v for k, v in tot.items() if not k.endswith("_maxabs")})
    print("max|Δ|：", {k[:-7]: v for k, v in tot.items() if k.endswith("_maxabs")})
    print("\n差异像素落在哪些块（平面, 块属性）：")
    for (pl, blk), c in per_block.most_common(25):
        print(f"  {['Y','U','V'][pl]}  {c:>4}  {blk}")


def find_block(all_blocks, lx, ly):
    """在全部轨迹段里找包含亮度坐标 (lx,ly) 的块，取**面积最小**的那个（最内层）。

    同一坐标会落在若干层级的块里（父块 + 实际重建它的子块），像素级的差异归因
    要落到**真正算这个像素的那个块**上，也就是尺寸最小的那个。
    """
    best = None
    for oh, m in all_blocks:
        r, c, bs = int(m.group(2)), int(m.group(3)), int(m.group(4))
        if bs >= len(W4):
            continue
        x0, y0 = c * 4, r * 4
        x1, y1 = x0 + W4[bs] * 4, y0 + H4[bs] * 4
        if x0 <= lx < x1 and y0 <= ly < y1:
            area = (W4[bs] * 4) * (H4[bs] * 4)
            if best is None or area < best[0]:
                tag = (f"#{m.group(1)} mi=({r},{c}) {W4[bs]*4}x{H4[bs]*4} "
                       f"inter={m.group(5)} skip={m.group(6)} ref={m.group(7)},{m.group(8)} "
                       f"mv={m.group(9)},{m.group(10)} if={m.group(13)},{m.group(14)} "
                       f"tx={m.group(15)} mm={m.group(16)} ct={m.group(17)} "
                       f"cg={m.group(18)} ii={m.group(19)}")
                best = (area, oh, tag)
    return (best[1], best[2]) if best else (None, "（无 —— 轨迹里没有块覆盖该亮度坐标）")


if __name__ == "__main__":
    sys.exit(main())
