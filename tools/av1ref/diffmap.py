#!/usr/bin/env python3
"""像素差异**定性**：把"我们 vs dav1d 差多少"变成"差在哪儿、像哪一层"。

`refcmp.py` 只给百分比与 max，回答不了"下一步该查谁"。这个脚本一次给全：

  ① 差值直方图（±1 为主还是长尾）；
  ② 差异点落在 4/8/16/32/64 网格线上的比例（**与整图基线比**，贴边 ⇒ 去块/CDEF；
     只有整图基线那一档 ⇒ 与块边界无关）;
  ③ **孤立点比例**（4 邻域内无其它差异点）—— 孤立单点 ⇒ 舍入/平局类问题；
     成片/成条 ⇒ 预测（MV）或掩码类问题；
  ④ 空间分布（按 128×128 超块计数）—— 集中在少数超块 ⇒ 与那些块用的**工具**有关；
  ⑤ `--probe`：把头几个差异点的 8×8 邻域原图/参考/差值并排打出来，肉眼看看。

2026-10-01 用它把"真素材还剩 0.2% 像素差"定性清楚：**100% 孤立单点、96% 只差 ±1、
集中在内容边缘（局部梯度 6–60 的点占 59%，整图只占 20%）、只落在帧间块里** ——
据此排除了"MV 差一点"（那会沿边缘成条），指向像素运算里的舍入平局。

用法：
    diffmap.py <源视频> --dir=<zuvrust decode --out 目录> --frames=1,10,42 [--probe]
"""
import argparse
import glob
import os

import av
import numpy as np


def plane_views(frame, bits=8):
    name = getattr(frame.format, "name", "") or ""
    if "10le" in name or "10be" in name:
        bits = 10
    out = []
    for p in range(3):
        pl = frame.planes[p]
        a = np.frombuffer(pl, dtype=np.uint8).reshape(pl.height, pl.line_size)
        if bits > 8:
            a = a.view(np.uint16)[:, : pl.width]
        else:
            a = a[:, : pl.width]
        out.append(a.copy())
    return out


def load_mine(path, w, h):
    raw = np.fromfile(path, dtype=np.uint8)
    cw, ch = (w + 1) // 2, (h + 1) // 2
    y = raw[: w * h].reshape(h, w)
    uv = raw[w * h:].reshape(ch, 2 * cw)
    return [y, uv[:, 0::2][:, :cw], uv[:, 1::2][:, :cw]]


def grid_mask(shape, step, tol=1):
    h, w = shape
    rows = np.zeros(h, dtype=bool)
    cols = np.zeros(w, dtype=bool)
    for t in range(-tol, tol + 1):
        rows |= ((np.arange(h) + t) % step) == 0
        cols |= ((np.arange(w) + t) % step) == 0
    m = np.zeros((h, w), dtype=bool)
    m |= rows[:, None]
    m |= cols[None, :]
    return m


def probe_print(mine, ref, y0, x0, k=6):
    h, w = mine.shape
    y0, x0 = max(0, y0 - k), max(0, x0 - k)
    ys, xs = slice(y0, min(h, y0 + 2 * k + 1)), slice(x0, min(w, x0 + 2 * k + 1))
    a, r = mine[ys, xs], ref[ys, xs]
    d = a.astype(int) - r.astype(int)
    print(f"    邻域 ({y0},{x0})  左=我们 右=参考 下=差")
    for i in range(a.shape[0]):
        print("      " + " ".join(f"{v:>3}" for v in a[i]) + "    " +
              " ".join(f"{v:>3}" for v in r[i]) + "    " +
              " ".join(f"{v:>+3}" for v in d[i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--dir", required=True, help="zuvrust decode --out 的目录")
    ap.add_argument("--frames", default="1,10,42")
    ap.add_argument("--probe", action="store_true", help="把头几个差异点的邻域摊开打出来")
    ap.add_argument("--probe-n", type=int, default=3)
    args = ap.parse_args()
    want = [int(x) for x in args.frames.split(",")]

    files = sorted(glob.glob(os.path.join(args.dir, "frame-*.nv12")))
    if not files:
        raise SystemExit(f"{args.dir} 里没有 frame-*.nv12")
    c = av.open(args.input)
    vs = c.streams.video[0]
    w, h = vs.width, vs.height
    refs = {}
    for i, fr in enumerate(c.decode(vs)):
        if i in want:
            refs[i] = plane_views(fr)
        if i > max(want):
            break
    c.close()

    for i in want:
        if i not in refs:
            print(f"帧 {i}: 源里没有")
            continue
        print(f"=== 帧 {i} ===")
        mine = load_mine(os.path.join(args.dir, f"frame-{i + 1:06}.nv12"), w, h)
        for pi, pname in enumerate("YUV"):
            a, r = mine[pi], refs[i][pi][: mine[pi].shape[0], : mine[pi].shape[1]]
            d = a.astype(np.int32) - r.astype(np.int32)
            nz = d != 0
            n = int(nz.sum())
            print(f"  {pname}: 不同 {n}/{d.size} = {100.0 * n / d.size:.3f}%  max {int(np.abs(d).max())}")
            if n == 0:
                continue
            v, cnt = np.unique(np.abs(d[nz]), return_counts=True)
            print(f"      差值分布: {dict(zip(v.tolist(), cnt.tolist()))}")
            grown = nz.copy()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                grown |= np.roll(np.roll(nz, dy, 0), dx, 1)
            # grown 里凡是"被邻居点亮"的格子，去掉自己后仍亮 ⇒ 有邻居
            has_neighbor = grown & ~nz
            isolated = int((nz & ~has_neighbor).sum())
            print(f"      孤立点 {isolated}/{n} = {100 * isolated / n:.0f}%")
            for step in (8, 16, 32, 64):
                m = grid_mask(d.shape, step, 1)
                print(f"      贴 {step:>2} 网格(±1)：差异 {100 * nz[m].sum() / n:5.1f}%  "
                      f"整图基线 {100 * m.mean():5.1f}%")
            ys, xs = np.nonzero(nz)
            sb = np.bincount((ys // 128) * ((d.shape[1] + 127) // 128) + xs // 128)
            top = np.argsort(sb)[::-1][:3]
            width_sb = (d.shape[1] + 127) // 128
            print(f"      集中在超块: {[(int(t // width_sb), int(t % width_sb), int(sb[t])) for t in top if sb[t] > 0]}"
                  f"（共涉及 {int((sb > 0).sum())} 个）")
            if args.probe:
                for k in range(min(args.probe_n, len(ys))):
                    print(f"    --- 第 {k} 个差异点 ({ys[k]},{xs[k]}) 我们 {a[ys[k], xs[k]]} 参考 {r[ys[k], xs[k]]}")
                    probe_print(a, r, ys[k], xs[k])


if __name__ == "__main__":
    main()
