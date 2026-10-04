#!/usr/bin/env python3
"""对拍结果汇总：给一个（或多个）`zuvrust decode --out` 目录，逐帧与 dav1d 比，
报 Y/U/V 的**差异像素合计、max|Δ|、有差异的帧数**，以及逐帧明细。

比 `refcmp.py` 省事的地方：`refcmp.py` 把 12 帧 × 3 平面打 36 行；这里先给一行
合计，用来**快速比较几套开关**（例如 `AV1_WARP_REFFMT=1` 前后）。

```sh
python3 totals.py <素材> /tmp/out_a /tmp/out_b --n=20
```
"""
import argparse
import glob
import os
import sys

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
            pls.append(
                np.frombuffer(pl, dtype=np.uint8)
                .reshape(pl.height, pl.line_size)[:, : pl.width]
                .copy()
            )
        out.append(pls)
        if len(out) >= n:
            break
    c.close()
    return w, h, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--detail", action="store_true", help="逐帧打明细")
    a = ap.parse_args()

    w, h, refs = load_ref(a.input, a.n)
    cw, ch = (w + 1) // 2, (h + 1) // 2
    for d in a.dirs:
        files = sorted(glob.glob(os.path.join(d, "frame-*.nv12")))[: a.n]
        tot = [0, 0, 0]
        mx = [0, 0, 0]
        nz = [0, 0, 0]
        for i, f in enumerate(files):
            raw = np.fromfile(f, dtype=np.uint8)
            y = raw[: w * h].reshape(h, w)
            uv = raw[w * h:].reshape(ch, 2 * cw)
            mine = [y, uv[:, 0::2][:, :cw], uv[:, 1::2][:, :cw]]
            for pi in range(3):
                r = refs[i][pi][: mine[pi].shape[0], : mine[pi].shape[1]]
                dd = np.abs(mine[pi].astype(np.int32) - r.astype(np.int32))
                n = int((dd > 0).sum())
                tot[pi] += n
                mx[pi] = max(mx[pi], int(dd.max()))
                if n:
                    nz[pi] += 1
        print(f"{os.path.basename(d.rstrip('/')):28s} "
              f"Y {tot[0]:7d}  U {tot[1]:6d}  V {tot[2]:6d}   "
              f"合计 {sum(tot):7d}   max {mx[0]}/{mx[1]}/{mx[2]}   "
              f"帧 {nz[0]}/{nz[1]}/{nz[2]} of {len(files)}")
        if a.detail:
            for i, f in enumerate(files):
                raw = np.fromfile(f, dtype=np.uint8)
                y = raw[: w * h].reshape(h, w)
                uv = raw[w * h:].reshape(ch, 2 * cw)
                mine = [y, uv[:, 0::2][:, :cw], uv[:, 1::2][:, :cw]]
                row = []
                for pi in range(3):
                    r = refs[i][pi][: mine[pi].shape[0], : mine[pi].shape[1]]
                    dd = np.abs(mine[pi].astype(np.int32) - r.astype(np.int32))
                    row.append(int((dd > 0).sum()))
                print(f"    帧{i:2d}  Y {row[0]:5d}  U {row[1]:4d}  V {row[2]:4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
