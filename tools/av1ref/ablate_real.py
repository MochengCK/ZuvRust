#!/usr/bin/env python3
"""真素材消融：在"开/关若干 `AV1_NO_*` 开关"下解同一条流，分别与 dav1d 比，
报 Y/U/V 三个平面的差异像素数与 `max|Δ|`。

用途（2026-10-03）：i16 溢出修完之后，真素材上还剩一层 **色度偏置的 ±1**
（Y 39 / U 260 / V 379，共 62M 样本里 678 个 = 0.0011%），而 13 条受控流全是 0。
本脚本就是用来判它落在哪一级的：关掉滤波还差 ⇒ 预测/残差；关掉就没了 ⇒ 滤波。

```sh
python3 ablate_real.py <素材> --bin=target/release/zuvrust
```
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import av


def load_ref(path, n):
    c = av.open(path)
    vs = c.streams.video[0]
    w, h = vs.width, vs.height
    refs = []
    for fr in c.decode(vs):
        pls = []
        for p in range(3):
            pl = fr.planes[p]
            pls.append(
                np.frombuffer(pl, dtype=np.uint8)
                .reshape(pl.height, pl.line_size)[:, : pl.width]
                .copy()
            )
        refs.append(pls)
        if len(refs) >= n:
            break
    c.close()
    return w, h, refs


def load_mine(d, w, h, n):
    files = sorted(f for f in os.listdir(d) if f.endswith(".nv12"))[:n]
    cw, ch = (w + 1) // 2, (h + 1) // 2
    out = []
    for f in files:
        raw = np.fromfile(os.path.join(d, f), dtype=np.uint8)
        y = raw[: w * h].reshape(h, w)
        uv = raw[w * h:].reshape(ch, 2 * cw)
        out.append([y, uv[:, 0::2][:, :cw], uv[:, 1::2][:, :cw]])
    return out


def score(binary, src, w, h, refs, env_extra, n):
    d = tempfile.mkdtemp(prefix="ablate_real_")
    try:
        env = dict(os.environ)
        env["ME_THREADS"] = "1"
        env.update(env_extra)
        r = subprocess.run(
            [binary, "decode", src, f"--count={n}", f"--out={d}", "--json"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0:
            return None
        mine = load_mine(d, w, h, n)
        tot = [0, 0, 0]
        mx = [0, 0, 0]
        for i, mp in enumerate(mine):
            for pi in range(3):
                a = mp[pi].astype(np.int32)
                b = refs[i][pi][: a.shape[0], : a.shape[1]].astype(np.int32)
                dd = np.abs(a - b)
                tot[pi] += int((dd > 0).sum())
                mx[pi] = max(mx[pi], int(dd.max()))
        return tot, mx
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--bin", default="target/release/zuvrust")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()

    w, h, refs = load_ref(a.input, a.n)
    cases = [
        ("全开（基准）", {}),
        ("关去块", {"AV1_NO_DEBLOCK": "1"}),
        ("关 CDEF", {"AV1_NO_CDEF": "1"}),
        ("关 LR", {"AV1_NO_LR": "1"}),
        ("三个滤波全关", {"AV1_NO_DEBLOCK": "1", "AV1_NO_CDEF": "1", "AV1_NO_LR": "1"}),
        ("关 OBMC", {"AV1_NO_OBMC": "1"}),
        ("关 warp", {"AV1_NO_WARP": "1"}),
        ("关 interintra", {"AV1_NO_INTERINTRA": "1"}),
        ("关 CFL", {"AV1_NO_CFL": "1"}),
        ("关残差", {"AV1_NO_RESIDUAL": "1"}),
    ]
    print(f"=== {os.path.basename(a.input)}  {w}x{h}  比 {a.n} 帧 ===")
    print(f"{'配置':16s} {'Y':>8s} {'U':>8s} {'V':>8s}   {'maxY':>4s} {'maxU':>4s} {'maxV':>4s}")
    for name, env in cases:
        s = score(a.bin, a.input, w, h, refs, env, a.n)
        if s is None:
            print(f"{name:16s} 解码失败")
            continue
        tot, mx = s
        print(f"{name:16s} {tot[0]:8d} {tot[1]:8d} {tot[2]:8d}   {mx[0]:4d} {mx[1]:4d} {mx[2]:4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
