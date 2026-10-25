#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把引擎落盘的 NV12 与参考解码器的输出**逐样本对拍**，并给出定位用的差异图。

## 两边各自长什么样

| 侧 | 产物 | 怎么来的 |
|---|---|---|
| 被测 | `<dir>/frame-000001.nv12`（紧凑 NV12，`me-cli decode --out=`） | `zuvrust decode <片> --count=N --out=/tmp/out --json` |
| 参考 | `<ref>/ref_000.nv12`（紧凑 NV12） | `make_cases.py` 里的 `decode_reference()` |

`--noflt` 会改读 `<mp4>.refnoflt/` —— 那是**跳过去块滤波**的参考帧
（ffmpeg `-skip_loop_filter all`）。

> **为什么 `--noflt` 是合法口径**：规范 §8.7 明说去块"在整张图的重建完成之后、
> 按宏块序进行"，所以**帧内预测读的是去块前的样本**。也就是说去块不参与预测，
> 可以单独抽出这一层来验证"预测 + 变换 + 色度"。
> ⚠️ 反过来：`x264-params deblock=0:0` **不能**关掉编码器侧的去块（它只是
> "用 QP 推导的默认强度"），想拿"没去块"的参照只能在解码侧跳过循环滤波。

## 用法

```sh
P=python3
$P tools/h264ref/make_cases.py --out=/tmp/h264cases
./target/debug/zuvrust decode /tmp/h264cases/nodeblock_detail.mp4 --count=4 --out=/tmp/out --json
$P tools/h264ref/refcmp.py /tmp/h264cases/nodeblock_detail.mp4 --dir=/tmp/out --noflt --map
```

## 读数怎么用

先看**是 Y 错还是色度错**，再看差异图的形状：

- **色度错、亮度对**：色度 DC 的 2×2 Hadamard、`LevelScale4x4(qP%6,0,0)`、
  表 8-15 的 `qP_C` 映射（含式 8-311 的 `Clip3`）。
- **整块偏移一个宏块**：帧内预测的**邻居可用性**推导错了。
- **每帧固定位置的方块不同**：`I_PCM` 或 `transform_size_8x8` 的扫描表。
- **大面积糊**：反量化（`normAdjust` 抄错最典型）或 CAVLC 表选错列。
- **只有块边界两三个样本差**：去块没做（用 `--noflt` 排除/确认）。

⚠️ **别拿"平均值"当验收**：解码错一点就会整块花，平均值会被大片正确的区域摊平。
判据是**逐帧的差异样本数与最大差值**，两者都要看。
"""

import argparse
import os
import sys

import numpy as np


def load(path):
    return np.fromfile(path, dtype=np.uint8)


def report(tag, a, b, w, h):
    """比一张 NV12；返回差异数组。`a` = 我们、`b` = 参考。"""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    ysz = w * h
    dy = d[:ysz]
    dc = d[ysz:]
    print(
        f"帧 {tag}: 样本 {d.shape[0]:7d}  差异 {int((d != 0).sum()):7d} "
        f"({(d != 0).sum() / d.shape[0]:7.3%})  max {int(d.max())}"
        f"   | Y {int((dy != 0).sum()):6d}/{ysz} max {int(dy.max())}"
        f"   | C {int((dc != 0).sum()):6d}/{dc.shape[0]} max {int(dc.max())}"
    )
    return d


def print_map(d, w, h, mb=16):
    """宏块级差异计数图 —— 一眼看出"从第几个宏块开始错"，
    那通常就是"错误第一次被引入"的位置。"""
    ny = d[: w * h].reshape(h, w)
    rows = []
    for my in range(0, h, mb):
        cells = []
        for mx in range(0, w, mb):
            n = int((ny[my : my + mb, mx : mx + mb] != 0).sum())
            cells.append(" .." if n == 0 else f"{n:3d}")
        rows.append("  " + " ".join(cells))
    print("  Y 宏块差异图（每格 = 16×16 内的差异样本数，`..` = 全对）:")
    print("\n".join(rows))

    cw, ch = w // 2, h // 2
    ncv = d[w * h :].reshape(ch, cw * 2)
    rows = []
    for my in range(0, ch, 8):
        cells = []
        for mx in range(0, cw * 2, 16):
            n = int((ncv[my : my + 8, mx : mx + 16] != 0).sum())
            cells.append(" .." if n == 0 else f"{n:3d}")
        rows.append("  " + " ".join(cells))
    print("  色度宏块差异图（每格 = 16×8 内的差异样本数）:")
    print("\n".join(rows))

    flat = np.nonzero(d.reshape(-1))[0]
    if flat.size:
        i = int(flat[0])
        if i < w * h:
            print(f"  第一处差异：Y 平面 第 {i // w} 行 第 {i % w} 列")
        else:
            j = i - w * h
            print(f"  第一处差异：色度平面 第 {j // w} 行 第 {j % w} 列（偶=Cb 奇=Cr）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mp4", nargs="?", help="（只用来取默认的 ref 目录名）")
    ap.add_argument("--dir", required=True, help="引擎 `--out=` 落盘的目录")
    ap.add_argument("--ref", help="参考帧目录（默认 <mp4 去掉后缀>.ref）")
    ap.add_argument(
        "--noflt",
        action="store_true",
        help="改用**跳过去块**的参考帧（<mp4>.refnoflt/）—— 引擎 §8.7 还没做时用这个",
    )
    ap.add_argument("--map", action="store_true", help="打宏块级差异图（定位用）")
    ap.add_argument("--w", type=int, default=96)
    ap.add_argument("--h", type=int, default=64)
    args = ap.parse_args()

    ref_dir = args.ref
    if ref_dir is None:
        if not args.mp4:
            raise SystemExit("要么给 mp4，要么给 --ref")
        ref_dir = os.path.splitext(args.mp4)[0] + ".ref"
    if args.noflt:
        ref_dir = ref_dir.rstrip("/") + "noflt"

    n = 0
    bad = 0
    while True:
        ours_p = os.path.join(args.dir, f"frame-{n + 1:06d}.nv12")
        ref_p = os.path.join(ref_dir, f"ref_{n:03d}.nv12")
        if not (os.path.exists(ours_p) and os.path.exists(ref_p)):
            break
        a = load(ours_p)
        b = load(ref_p)
        if a.shape != b.shape:
            print(f"帧 {n}: 长度不同 —— 我们 {a.shape[0]} 字节、参考 {b.shape[0]} 字节")
            bad += 1
            n += 1
            continue
        d = report(f"{n}", a, b, args.w, args.h)
        if args.map and (d != 0).any():
            print_map(d, args.w, args.h, mb=16)
        if (d != 0).any():
            bad += 1
        n += 1

    if n == 0:
        raise SystemExit(f"没找到可比的帧：{args.dir}/frame-*.nv12 与 {ref_dir}/ref_*.nv12")
    print(f"\n共 {n} 帧，{bad} 帧有差异")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
