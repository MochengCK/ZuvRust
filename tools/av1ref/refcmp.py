#!/usr/bin/env python3
"""AV1 参考解码器对拍：把自研软解的输出与 dav1d 逐像素比。

## 为什么需要它

自研 AV1 软解只有两类验证：① 自己造的参照流（`tools/av1probe/`，全是小尺寸单帧）
② 真实素材。**这两类都不能回答"我的重建与参考实现是否逐像素一致"** —— 直到
有了这个脚本。它把"看起来对"变成"逐像素相同"，是本项目最有力的判据。

## 依赖

    python3 -m venv <venv>
    <venv>/bin/pip install av numpy

`av`（PyAV）的 wheel 内置换 libdav1d 的 FFmpeg，所以**不需要 brew/ffmpeg**。

## 用法

    # 1) 我们的引擎解码落盘（NV12）
    zuvrust decode <输入> --count 4 --out=/tmp/out_x

    # 2) 与 dav1d 逐像素比（Y / U / V 三个平面各报一行）
    refcmp.py <输入> --dir=/tmp/out_x --n=4

## 关键开关：--grain

`film grain synthesis`（规范 §7.18）**我们已经实现**，所以对拍时默认就该开
（`--grain=1`），比的是"含颗粒的最终像素"。

`--grain=0` 给 dav1d 加 `filmgrain=0`，比的是"加颗粒之前的重建像素"。它仍有用，
但用法反过来了：**它是用来反证颗粒真的生效的** —— 关掉颗粒时如果两边差异很大，
说明这条流里确实有颗粒，而 `--grain=1` 下逐像素一致就证明我们合成得对。

    refcmp.py aom.mp4 --dir=/tmp/out --grain=1   # 默认；应逐像素一致
    refcmp.py aom.mp4 --dir=/tmp/out --grain=0   # 应出现明显差异（= 颗粒被摘掉）

## 位深

引擎对外只出 8 位 NV12；10/12 位的参考帧在这里右移 `(bitdepth-8)` 位后再比，
与 `video/av1.rs::nv12_frame` 的归一化一致。
"""
import argparse
import glob
import os
import sys
import numpy as np
import av


def _bits(fmt, default=8):
    """从 PyAV 的格式名（`yuv420p` / `yuv420p10le` / `yuv420p12le` …）取位深。"""
    name = getattr(fmt, "name", None) or ""
    for b in (12, 10, 9):
        if f"{b}le" in name or f"{b}be" in name:
            return b
    return default


def _planes(frame):
    """PyAV 的一帧 → [Y, U, V]（按 `line_size` 去掉行填充）。"""
    bits = _bits(frame.format)
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


def load_my(out_dir, w, h):
    files = sorted(glob.glob(os.path.join(out_dir, "frame-*.nv12")))
    if not files:
        return None, f"{out_dir} 里没有 frame-*.nv12"
    cw, ch = (w + 1) // 2, (h + 1) // 2
    want = w * h + 2 * cw * ch
    my = []
    for f in files:
        raw = np.fromfile(f, dtype=np.uint8)
        if raw.size != want:
            return None, f"NV12 长度 {raw.size} != 期望 {want}"
        y = raw[: w * h].reshape(h, w)
        uv = raw[w * h:].reshape(ch, 2 * cw)
        my.append([y, uv[:, 0::2][:, :cw], uv[:, 1::2][:, :cw]])
    return my, None


def load_ref(path, grain, want_frames):
    c = av.open(path)
    vs = c.streams.video[0]
    w, h = vs.width, vs.height
    refs = []
    bits = 8
    if grain == 0:
        # 必须显式建上下文才能设解码器选项（容器的默认上下文不让我们设）
        ctx = av.CodecContext.create("libdav1d", "r")
        ctx.options = {"filmgrain": "0"}
        ctx.extradata = vs.codec_context.extradata
        ctx.open()
        for pkt in c.demux(vs):
            for fr in ctx.decode(pkt):
                bits = _bits(fr.format)
                refs.append(_planes(fr))
            if len(refs) >= want_frames:
                break
    else:
        for fr in c.decode(vs):
            bits = _bits(fr.format)
            refs.append(_planes(fr))
            if len(refs) >= want_frames:
                break
    c.close()
    return w, h, refs, bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--dir", required=True, help="zuvrust decode --out 的目录")
    ap.add_argument("--grain", type=int, default=1, help="1 = 应用胶片颗粒（默认），0 = 不应用")
    ap.add_argument("--n", type=int, default=1, help="比多少帧")
    args = ap.parse_args()

    w, h, refs, bits = load_ref(args.input, args.grain, args.n)
    my, err = load_my(args.dir, w, h)
    if err:
        print(f"✗ {err}")
        return 1
    shift = max(0, bits - 8)
    n = min(len(my), len(refs))
    print(
        f"=== {os.path.basename(args.input)}  {w}x{h} {bits}bit  "
        f"比较 {n} 帧（grain={'on' if args.grain else 'off'}）==="
    )
    total = 0
    for i in range(n):
        for pi, pname in enumerate("YUV"):
            mine = my[i][pi].astype(np.int32)
            ref = refs[i][pi][: mine.shape[0], : mine.shape[1]]
            ref = (ref >> shift) if shift else ref
            d = np.abs(mine - ref.astype(np.int32))
            nb = int((d != 0).sum())
            total += nb
            if nb == 0:
                print(f"  帧 {i} {pname}: ✓ 逐像素一致")
            else:
                print(
                    f"  帧 {i} {pname}: ✗ 不同 {nb}/{d.size} = {100.0*nb/d.size:.2f}%  "
                    f"max {d.max()} 均 {d.mean():.2f}"
                )
    print(f"结论: {'✓ 一致' if total == 0 else '✗ 有偏差'}")
    if total and args.grain == 1:
        print("提示：grain=1 有偏差，先用 --grain=0 看看是不是颗粒这一层；"
              "两边都差才是解码器本体的问题。")
    return 0 if total == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
