#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""造受控 H.264 用例 + 出参考帧（PyAV 的 FFmpeg 原生 h264 解码器当 oracle）。

自研 H.264 软解（`crates/me-decode/src/video/h264/`）的**唯一硬判据**。
合成位流能钉语法，但回答不了"重建是否正确"；真实素材又太复杂。这里补中间那层。

## 两条腿

| 角色 | 怎么来 | 说明 |
|---|---|---|
| **造流** libx264 | `pip install av`（PyAV wheel 里就有 libx264） | Baseline 的 CAVLC、Main/High 的 CABAC 都能点名造 |
| **oracle** FFmpeg 原生 h264 | 同上（`av.CodecContext.create("h264", "r")`） | **不是** dav1d 那种第三方解码器，是 FFmpeg 自带的参考实现；只在工具链里用，引擎不依赖 |

## 用例怎么分

每个用例都是 `内容 × 编码配置`，命名里点明两者：

- **内容**决定"能摸到哪些代码路径"：`flat_*` 只走 DC；`detail` 逼出 AC 系数；
  `color` 让色度残差非零；`diag` 摸盘面（Plane）预测与斜向模式；
  `quadrants` 是四个各自平坦的象限（分区边界）；`noise` 是最难的一种。
- **`nodeblock_*`** = 关掉去块滤波（`x264-params deblock=0:0`）+ 关掉 8×8 变换。
  引擎 §8.7 还没做时，只有对这些用例对拍才有意义。
- **`deblock_*`** = 开去块（x264 默认强度）。**这是 §8.7 的 oracle**：
  引擎实现去块之前这些用例必然对不上，对上了才说明去块对了。

## 用法

```sh
P=python3
$P tools/h264ref/make_cases.py --out=/tmp/h264cases
$P tools/h264ref/refcmp.py /tmp/h264cases/nodeblock_detail.mp4 --dir=/tmp/out
```

## 坑

- **Baseline 才有 CAVLC**：`profile:v baseline` + `coder 0`；Main/High 默认 CABAC。
- **选项必须写在 `codec_context.options` 上**：`st.options` 在 `add_stream` 之后
  设了不生效（实测 `8x8dct=0` 被静默忽略，排查时会以为"改过了"）。
- **要逐帧精确对齐帧数**：`stream.encode(None)` 才是 flush，少调一次会少一帧，
  后面对拍会整体错位。
- **`yuv420p` 必须显式设**：libx264 默认可能给 `yuv444p`，那是引擎明确不支持的形态。
- **`bf=0`**：B 切片是另一层的事，先不让编码器插 B 帧，帧型序列才可控。
"""

import argparse
import os
import subprocess

import av
import imageio_ffmpeg
import numpy as np

# `skip_loop_filter` 是**解码器**选项，PyAV 的 `CodecContext.create` 不接受
# options，所以这条只能用 ffmpeg 命令行。`imageio-ffmpeg` 带一个静态二进制，
# 不用装 homebrew。
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

W, H = 96, 64  # 96×64：宏块对齐（6×4 个宏块），省得裁剪，排查时少一个变量

# 对拍用的 x264 参数：帧型与场景切换都钉死，去块/8×8 变换由用例决定
_BASE_X264 = "scenecut=0:bframes=0"


def x264_params(*, deblock: bool, dct8x8: bool = False, extra: str = "") -> str:
    """拼 `x264-params`。`deblock=False` ⇒ `deblock=0:0`（真的关掉，见模块抬头的坑）。"""
    parts = [
        "deblock=1:1" if deblock else "deblock=0:0",
        "8x8dct=1" if dct8x8 else "8x8dct=0",
        _BASE_X264,
    ]
    if extra:
        parts.append(extra)
    return ":".join(parts)


def frame_rgb(kind, i):
    """按 `kind` 造第 i 帧的像素（R=G=B 时方便与 Y 平面直接对照）。

    全部**确定性**（不用随机数）—— 排查时要能重放同一帧。
    """
    yy, xx = np.mgrid[0:H, 0:W]

    if kind == "flat_128":
        return np.full((H, W, 3), 128, dtype=np.uint8)

    if kind == "flat_gradient":
        # 水平渐变：帧内预测的一点方向性能显出来（DC vs 方向模式结果不同）
        row = np.linspace(16, 235, W).astype(np.uint8)
        return np.tile(row[None, :, None], (H, 1, 3))

    if kind == "quadrants":
        a = np.zeros((H, W, 3), dtype=np.uint8)
        a[: H // 2, : W // 2] = 32
        a[: H // 2, W // 2:] = 160
        a[H // 2 :, : W // 2] = 208
        a[H // 2 :, W // 2:] = 96
        # 逐帧整体抬一点，避免所有帧完全一样看不出"帧序错了"
        return np.clip(a.astype(np.int16) + i * 3, 0, 255).astype(np.uint8)

    if kind == "moving_block":
        a = np.full((H, W, 3), 40, dtype=np.uint8)
        x = (i * 7) % (W - 24)
        y = (i * 5) % (H - 16)
        a[y : y + 16, x : x + 24] = 200
        return a

    if kind == "detail":
        # 高频细纹 + 每 8×8 一块的台阶：**逼出 AC 系数**（`coded_block_pattern` 非零），
        # 也让 I_4x4 / I_NxN 有机会被选中。周期与 4 互质，免得整块被 DC 吃掉。
        v = (xx % 7) * 9 + (yy % 5) * 13 + ((xx // 8) * 3 + (yy // 8) * 7) % 17 * 5
        g = (v // 2 + 30).clip(0, 255).astype(np.uint8)
        return np.repeat(g[:, :, None], 3, axis=2)

    if kind == "color":
        # 三个通道各走一个方向 + 棋盘：**色度残差非零**（`cbp_chroma != 0`），
        # 这样色度 DC / AC 那条链才会真的被走到。
        r = (xx * 2 + i * 5) % 256
        g = (yy * 3 + 20) % 256
        b = ((xx // 8 + yy // 8) % 2) * 200 + 27
        return np.stack([r, g, b], axis=2).astype(np.uint8)

    if kind == "diag":
        # 斜向渐变 + 折线：摸盘面（Plane）预测，以及 4×4 的斜向模式。
        v = (xx * 2 + yy * 3) % 256
        v = np.where((xx + yy) % 32 < 16, v, (v + 96) % 256)
        g = v.astype(np.uint8)
        return np.repeat(g[:, :, None], 3, axis=2)

    if kind == "noise":
        # 伪随机但确定性（用整数哈希当噪声源）：最"难"的内容，残差几乎处处非零。
        h = (xx * 2654435761 + yy * 40503 + i * 7919) & 0xFFFFFFFF
        g = (40 + (h >> 13) % 180).astype(np.uint8)
        return np.repeat(g[:, :, None], 3, axis=2)

    raise SystemExit(f"不认识的内容类型：{kind}")


def encode(path, kind, frames, *, profile="baseline", coder="0", gop=1, crf=18, x264=None):
    """用 libx264 编一段；`coder` 是 `-coder`：0 = CAVLC、1 = CABAC。"""
    out = av.open(path, mode="w")
    st = out.add_stream("libx264")
    # ⚠️ 选项必须写在 `codec_context.options` 上：`st.options` 在 add_stream 之后
    # 设了不生效（实测 `8x8dct=0` 被静默忽略），排查时会以为"改过了"。
    st.codec_context.options = {
        "profile": profile,
        "coder": coder,
        "g": str(gop),
        "bf": "0",
        "crf": str(crf),
        "x264-params": x264 or x264_params(deblock=False),
    }
    st.width, st.height = W, H
    st.pix_fmt = "yuv420p"
    for i in range(frames):
        img = frame_rgb(kind, i)
        f = av.VideoFrame.from_ndarray(img, format="rgb24")
        for pkt in st.encode(f):
            out.mux(pkt)
    for pkt in st.encode(None):  # flush：少这一句会少一帧
        out.mux(pkt)
    out.close()


def decode_reference(path, out_dir):
    """用 FFmpeg 原生 h264 解码器解出**紧凑 NV12**，一帧一个文件。

    同时写两份：

    | 目录 | 内容 |
    |---|---|
    | `<name>.ref/` | 正常解码（**含去块滤波**）—— §8.7 落地之后的验收口径 |
    | `<name>.refnoflt/` | `-skip_loop_filter all`（**跳过去块**）—— 去块还没做时的验收口径 |

    为什么要两份：`x264-params deblock=0:0` **不是**关去块，它只是"用 QP 推导的
    默认强度"（实测把去块开关打进去，同一段流仍会改 2000+ 个样本）。想拿到
    "没去块"的参照，只能在**解码侧**跳过循环滤波 —— 这就是 `refnoflt` 的用处：
    它让"预测 + 变换 + 色度"这三层可以**先于 §8.7** 被独立验证。
    """
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    with av.open(path) as c:
        st = c.streams.video[0]
        for frame in c.decode(st):
            arr = frame.to_ndarray(format="nv12")
            arr.tofile(os.path.join(out_dir, f"ref_{n:03d}.nv12"))
            n += 1

    noflt = out_dir + "noflt"
    os.makedirs(noflt, exist_ok=True)
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-skip_loop_filter", "all", "-i", path,
         "-f", "rawvideo", "-pix_fmt", "nv12", "-"],
        capture_output=True,
        check=True,
    ).stdout
    stride = W * H * 3 // 2
    for i in range(len(raw) // stride):
        with open(os.path.join(noflt, f"ref_{i:03d}.nv12"), "wb") as f:
            f.write(raw[i * stride : (i + 1) * stride])
    return n


def cases():
    """用例表：`内容 × 编码配置`。命名前缀点明能不能拿去跟"没做去块"的引擎对拍。"""
    nb = dict(deblock=False)  # nodeblock：关去块（引擎 §8.7 之前的对拍口径）
    db = dict(deblock=True)  # deblock：开去块（§8.7 的 oracle）
    out = [
        # ── 关去块：验语法层 + 重建（预测 / 变换 / 色度） ──────────────────
        ("nodeblock_flat", "flat_128", dict(profile="baseline", coder="0"), nb),
        ("nodeblock_quad", "quadrants", dict(profile="baseline", coder="0"), nb),
        ("nodeblock_grad", "flat_gradient", dict(profile="baseline", coder="0"), nb),
        ("nodeblock_moving", "moving_block", dict(profile="baseline", coder="0"), nb),
        ("nodeblock_detail", "detail", dict(profile="baseline", coder="0"), nb),
        ("nodeblock_color", "color", dict(profile="baseline", coder="0"), nb),
        ("nodeblock_diag", "diag", dict(profile="baseline", coder="0"), nb),
        ("nodeblock_noise", "noise", dict(profile="baseline", coder="0"), nb),
        # 换个 QP：`qP % 6` 会变，能摸到 `normAdjust` 的不同行
        ("nodeblock_detail_qp12", "detail", dict(profile="baseline", coder="0", crf=12), nb),
        ("nodeblock_detail_qp30", "detail", dict(profile="baseline", coder="0", crf=30), nb),
        # ── 开去块：§8.7 的 oracle ─────────────────────────────────────────
        ("deblock_grad", "flat_gradient", dict(profile="baseline", coder="0"), db),
        ("deblock_quad", "quadrants", dict(profile="baseline", coder="0"), db),
        ("deblock_detail", "detail", dict(profile="baseline", coder="0"), db),
        ("deblock_color", "color", dict(profile="baseline", coder="0"), db),
        ("deblock_noise", "noise", dict(profile="baseline", coder="0"), db),
        # ── 8×8 变换（High profile + CAVLC 也能开） ─────────────────────────
        (
            "nodeblock_detail_8x8",
            "detail",
            dict(profile="high", coder="0"),
            dict(deblock=False, dct8x8=True),
        ),
        # ── CABAC（§9.3 的 oracle） ────────────────────────────────────────
        ("cabac_detail", "detail", dict(profile="main", coder="1"), nb),
        ("cabac_color", "color", dict(profile="main", coder="1"), nb),
        ("cabac_noise", "noise", dict(profile="main", coder="1"), nb),
        ("cabac_grad", "flat_gradient", dict(profile="main", coder="1"), nb),
        # ── 帧间（§8.4 的 oracle）：I + P 交替 ─────────────────────────────
        ("inter_grad", "flat_gradient", dict(profile="baseline", coder="0", gop=4), nb),
        ("inter_moving", "moving_block", dict(profile="baseline", coder="0", gop=4), nb),
        ("inter_detail", "detail", dict(profile="baseline", coder="0", gop=4), nb),
        ("inter_color", "color", dict(profile="baseline", coder="0", gop=4), nb),
    ]
    for name, kind, enc, fl in out:
        yield name, kind, dict(enc, x264=x264_params(**fl))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/h264cases")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--only", help="只造名字里含这个子串的用例")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name, kind, opts in cases():
        if args.only and args.only not in name:
            continue
        mp4 = os.path.join(args.out, name + ".mp4")
        encode(mp4, kind, args.frames, **opts)
        ref = os.path.join(args.out, name + ".ref")
        n = decode_reference(mp4, ref)
        size = os.path.getsize(mp4)
        print(f"{name:26s} {size:8d} B  参考帧 {n} 帧 → {ref}")


if __name__ == "__main__":
    main()
