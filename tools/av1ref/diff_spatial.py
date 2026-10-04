#!/usr/bin/env python3
"""把真素材的残留差异**按空间分布**摊开：逐帧列出差异的坐标范围与形状。

`chroma_probe.py` 回答"落在哪个块"，这个回答"在画面的哪儿"：两个色度簇各自
的包围盒、随帧的漂移、亮度/色度各自的行分布。用来判断是"帧边界类"问题
（分布贴着帧边）还是"内容类"问题（跟着物体走）。

```sh
python3 diff_spatial.py <素材> --dir=<decode --out> --n=20
```
"""
import argparse
import os
from collections import Counter

import numpy as np
import av


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
    cw, ch = w // 2, h // 2
    ys, cs = w * h, cw * ch * 2
    out = []
    for i in range(n):
        p = os.path.join(d, f"frame-{i+1:06d}.nv12")
        if not os.path.exists(p):
            break
        b = np.fromfile(p, dtype=np.uint8)
        uv = b[ys:ys + cs].reshape(ch, cw * 2)
        out.append([b[:ys].reshape(h, w), uv[:, 0::2].copy(), uv[:, 1::2].copy()])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()

    w, h, refs = load_ref(a.src, a.n)
    mine = load_mine(a.dir, w, h, a.n)
    cw, ch = w // 2, h // 2
    print(f"{w}x{h}  色度 {cw}x{ch}  帧数 {len(mine)}\n")

    for fi in range(min(len(mine), len(refs))):
        for pl, name in ((0, "Y"), (1, "U"), (2, "V")):
            dm = mine[fi][pl].astype(np.int16) - refs[fi][pl].astype(np.int16)
            idx = np.argwhere(dm != 0)
            if len(idx) == 0:
                continue
            xs, ys_ = idx[:, 1], idx[:, 0]
            # 按 x 分簇（间隔 >16 视为换簇）
            order = np.argsort(xs)
            xs_s, ys_s = xs[order], ys_[order]
            clusters, cur = [], [0]
            for i in range(1, len(xs_s)):
                if xs_s[i] - xs_s[i - 1] > 16:
                    clusters.append(cur)
                    cur = []
                cur.append(i)
            clusters.append(cur)
            desc = []
            for cc in clusters:
                cx, cy = xs_s[cc], ys_s[cc]
                desc.append(f"[x {cx.min()}~{cx.max()}({len(cc)}) y {cy.min()}~{cy.max()}]")
            hist = Counter(ys_s.tolist())
            top_rows = " ".join(f"{r}:{n}" for r, n in sorted(hist.items())[:6])
            print(f"帧{fi:>2} {name} n={len(idx):>3} Δ∈[{dm[idx[:,0],idx[:,1]].min()},"
                  f"{dm[idx[:,0],idx[:,1]].max()}] 簇 {len(clusters)}: {' '.join(desc[:6])}")
            print(f"        行分布 {top_rows}")

    # 汇总：所有帧所有差异的坐标直方图
    print("\n=== 全局坐标分布（所有帧合并）===")
    for pl, name in ((0, "Y"), (1, "U"), (2, "V")):
        pts = []
        for fi in range(min(len(mine), len(refs))):
            dm = mine[fi][pl].astype(np.int16) - refs[fi][pl].astype(np.int16)
            idx = np.argwhere(dm != 0)
            for y, x in idx:
                pts.append((x, y, int(dm[y, x])))
        if not pts:
            continue
        xs = Counter(p[0] for p in pts)
        ys = Counter(p[1] for p in pts)
        ds = Counter(p[2] for p in pts)
        print(f"\n{name} 共 {len(pts)} 个：Δ 分布 {dict(sorted(ds.items()))}")
        print(f"  x 范围 {min(xs)}~{max(xs)}  y 范围 {min(ys)}~{max(ys)}")
        print(f"  x 众数 {xs.most_common(10)}")
        print(f"  y 众数 {ys.most_common(10)}")


if __name__ == "__main__":
    main()
