#!/usr/bin/env python3
"""**补上"运动"这一档用例**：亚像素平移 + 每块运动各异的马赛克运动。

## 为什么非要有它

2026-10-01 盘了一遍 `make_cases.py` 的用例表才发现一个**覆盖盲区**：

- libaom 那批用例全是 `--kf-max-dist=1`（**全关键帧**，压根没有帧间帧）；
- 走 PyAV/SVT 那条用的是 `content(yy, xx)` —— **不带帧号 ⇒ 内容静止**，
  帧间帧的 MV 全是 0；
- 唯一会动的 `moving_object()` 是 **24px/8px 的整数位移**（还都是 4 的倍数）。

也就是说：**分数精度 MV 的子像素插值从来没被验过**。而真素材（B 站 ASMR 1080p）
剩下的那 0.2% 像素差恰好只落在**帧间块**、**内容边缘**上（±1、孤立单点）——
正是子像素路径的指纹，所以必须先把这一档补齐。

## 内容怎么造

| 用例 | 内容 | 会逼出什么 |
|---|---|---|
| `--kind=shift` | 一张多尺度噪声底图，每帧按 **(3.5, 1.25) px** 双线性重采样 | 分数 MV、8 抽头插值的两条相位 |
| `--kind=mosaic` | 每 32×32 单元各走各的方向/步长（含 0.5 px），内容 = 3px 棋盘 + 精细噪声 | 邻居 MV 不一致（OBMC 的温床）、复合、遮挡 |

两条都刻意**避开**纯色块与屏幕内容（否则编码器会走 palette/intrabc）。

## 实测结论（2026-10-01）

`--kind=shift` 与 `--kind=mosaic` 在当前解码器上**逐像素一致**
（filters on / off 都是），⇒ **分数 MV 的运动补偿、复合、去块/CDEF/LR
在合成内容上都精确**。真素材那 0.2% 的锅不在这条路上 —— 详见
`diffmap.py` 的抬头与 `video/mod.rs` 里"像素层"那一节。

## 坑

`--altref`（`--lag-in-frames=19 --auto-alt-ref=1`）在 8 帧的短片里**没有生效**：
产出的码流与不开时**字节完全相同**。**判"参数有没有生效"要像技能里说的那样比 md5**，
别只看命令行里有开关。

用法：
    motion_cases.py --out=/tmp/motion [--kind=shift|mosaic] [--nofilters] [--altref]
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_cases import content, ivf_to_mp4, write_y4m  # noqa: E402

AOMENC = "/tmp/libaom-3.9.0/build_g/aomenc"


def _bilinear(base, oy, ox, h, w):
    y0, x0 = int(np.floor(oy)), int(np.floor(ox))
    fy, fx = oy - y0, ox - x0
    a = base[y0 : y0 + h, x0 : x0 + w]
    b = base[y0 : y0 + h, x0 + 1 : x0 + w + 1]
    c = base[y0 + 1 : y0 + h + 1, x0 : x0 + w]
    d = base[y0 + 1 : y0 + h + 1, x0 + 1 : x0 + w + 1]
    return a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx + c * fy * (1 - fx) + d * fy * fx


def shift_frames(w, h, n, dx=3.5, dy=1.25, margin=64):
    """整幅按 (i*dx, i*dy) 亚像素平移 —— 全局同向、每帧都在半像素上。"""
    yy, xx = np.mgrid[0 : h + 2 * margin, 0 : w + 2 * margin].astype(np.float64)
    base = content(yy, xx, 0).astype(np.float64)
    return [
        np.clip(_bilinear(base, margin + i * dy, margin + i * dx, h, w), 0, 255).astype(np.uint8)
        for i in range(n)
    ]


def mosaic_frames(w, h, n, margin=96):
    """每 32×32 单元各走各的（方向/步长含 0.5px）—— 邻居 MV 不一致 ⇒ OBMC/复合/遮挡。"""
    yy, xx = np.mgrid[0 : h + 2 * margin, 0 : w + 2 * margin].astype(np.float64)
    checker = np.where(((yy // 3) + (xx // 3)) % 2 == 0, 30, 220).astype(np.float64)
    base = np.clip(0.55 * checker + 0.45 * content(yy, xx, 3).astype(np.float64), 0, 255)
    out = []
    for i in range(n):
        img = np.zeros((h, w), np.float64)
        for cy in range(0, h, 32):
            for cx in range(0, w, 32):
                k = ((cy // 32) * 7 + (cx // 32) * 3 + i) % 5
                dx, dy = (0.0, 1.5, -2.0, 3.5, -1.25)[k], (1.5, -1.0, 2.5, 0.0, -3.5)[k]
                blk = _bilinear(base, margin + cy + i * dy, margin + cx + i * dx, 32, 32)
                img[cy : cy + 32, cx : cx + 32] = blk[: min(32, h - cy), : min(32, w - cx)]
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/motion")
    ap.add_argument("--kind", choices=["shift", "mosaic"], default="shift")
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=360)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--nofilters", action="store_true", help="关掉 CDEF/环路恢复/去块")
    ap.add_argument("--altref", action="store_true", help="排 B 金字塔（短片里常常不生效，要验 md5）")
    ap.add_argument("--aomenc", default=AOMENC)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    name = f"aom_{args.kind}" + ("_nofilt" if args.nofilters else "") + ("_arf" if args.altref else "")
    y4m, ivf, mp4 = (os.path.join(args.out, name + e) for e in (".y4m", ".ivf", ".mp4"))

    frames = shift_frames(args.w, args.h, args.n) if args.kind == "shift" else mosaic_frames(args.w, args.h, args.n)
    u = np.full((args.h // 2, args.w // 2), 128, np.uint8)
    write_y4m(y4m, args.w, args.h, frames, [(u, u)] * args.n)

    cmd = [
        args.aomenc, "--codec=av1", "--passes=1", f"--limit={args.n}", "--good",
        "--cpu-used=4", "--end-usage=q", "--cq-level=40", "--sb-size=128",
        "--enable-palette=0", "--enable-intrabc=0", "--tune-content=film",
        # 只第一帧关键帧 ⇒ 后面全是 inter，必须走运动补偿
        "--kf-max-dist=64", "--kf-min-dist=64", "--threads=4",
    ]
    if args.nofilters:
        cmd += ["--enable-cdef=0", "--enable-restoration=0", "--loopfilter-control=0"]
    if args.altref:
        cmd += ["--lag-in-frames=19", "--auto-alt-ref=1", "--enable-fwd-kf=1"]
    cmd += ["-o", ivf, y4m]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("aomenc 失败:", (r.stderr or r.stdout)[-600:])
        return 1
    ivf_to_mp4(ivf, mp4)
    for f in (y4m, ivf):
        if os.path.exists(f):
            os.remove(f)
    print(f"{mp4}  {args.w}x{args.h} {args.n} 帧  {os.path.getsize(mp4)}B"
          f"{'  [滤波器全关]' if args.nofilters else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
