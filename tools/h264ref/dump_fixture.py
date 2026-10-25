#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从受控流里抠出**一个访问单元**，存成引擎回归测试用的夹具。

## 为什么要夹具

`cargo test` 要能在任何机器上跑（不能依赖 PyAV / 网络 / 编码器），而
"CAVLC + 宏块层是不是一位不差地读完了一条切片"这条判据必须**每次改动都跑**。
所以把一条**真实编码器写出来的** IDR 切片固化进仓库
（`crates/me-decode/src/video/h264/testdata/`）—— 3 KB 不到，
比手造位流可信得多（手造的和解析器出自同一份理解，会一起错）。

## 为什么还要连参考 NV12 一起存

"语法读完了"只证明**位没读错**，证明不了**像素对**。所以用 FFmpeg 自带的
h264 解码器把同一个访问单元解成 NV12 一并固化成夹具（见 `--ref-out`），
测试里逐样本比对 —— 这样"重建全链路"就有了一个**外部**判据，
而不是拿自己的实现跟自己的理解对。

> 参考帧取的是**正常解码**（含 §8.7 去块滤波）的结果：引擎现在去块也做了，
> 逐样本比对就该覆盖到这一层。
> ⚠️ 注意 `x264-params deblock=0:0` **不是**"关掉去块"，它只是"用 QP 推导的默认
> 强度"（去块仍然开着）—— 想拿"没去块"的参照只能在解码侧 `-skip_loop_filter all`，
> 那份由 `make_cases.py` 落到 `<name>.refnoflt/`，供开发 §8.7 之前/期间分层验证。

## 文件格式（**两段大端长度前缀**，够简单、够稳）

```
u32 be  avcC 长度 | avcC 字节 | u32 be 采样长度 | 采样字节（AVCC：4 字节长度前缀的 NAL 序列）
```

参考帧则是**裸 NV12**（`width * height * 3 / 2` 字节），另有 `--ref-out` 落盘。

## 用法

```sh
P=python3
$P tools/h264ref/make_cases.py --out=/tmp/h264cases
$P tools/h264ref/dump_fixture.py /tmp/h264cases/baseline_cavlc.mp4 \
    --out crates/me-decode/src/video/h264/testdata/i_slice_cavlc.bin \
    --ref-out crates/me-decode/src/video/h264/testdata/i_slice_cavlc.nv12
```
"""

import argparse
import struct

import av


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mp4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-out", help="把参考解码器解出的裸 NV12 也写到这里")
    ap.add_argument("--index", type=int, default=0, help="取第几个访问单元（默认第一个）")
    args = ap.parse_args()

    sample = None
    ref = None
    with av.open(args.mp4) as c:
        st = c.streams.video[0]
        avcc = bytes(st.codec_context.extradata)
        if avcc[:1] != b"\x01":
            # MP4 里的 extradata 就是 avcC 载荷；带 0x01 版本号才说明是它本体
            raise SystemExit(f"extradata 看起来不是 avcC（首字节 {avcc[:1]!r}）")
        for i, pkt in enumerate(c.demux(st)):
            if i < args.index:
                continue
            sample = bytes(pkt)
            break
        if sample is None:
            raise SystemExit("没取到采样")

        if args.ref_out:
            # ⚠️ 用 PyAV 的 `h264` 解码器（FFmpeg 原生那个）当 oracle；
            # 引擎自己**不**依赖它，只有工具链用。
            dec = av.CodecContext.create("h264", "r")
            dec.extradata = avcc
            for frame in dec.decode(av.Packet(sample)):
                ref = frame.reformat(format="nv12")
                break
            if ref is None:
                raise SystemExit("参考解码器没吐出帧")

    with open(args.out, "wb") as f:
        f.write(struct.pack(">I", len(avcc)))
        f.write(avcc)
        f.write(struct.pack(">I", len(sample)))
        f.write(sample)
    print(f"avcC {len(avcc)} B + 采样 {len(sample)} B → {args.out}")

    if args.ref_out and ref is not None:
        plane = bytes(ref.planes[0]) + bytes(ref.planes[1])
        with open(args.ref_out, "wb") as f:
            f.write(plane)
        print(f"参考 NV12 {len(plane)} B ({ref.width}×{ref.height}) → {args.ref_out}")


if __name__ == "__main__":
    main()
