#!/usr/bin/env python3
"""外科式中和实验（A/B）：**两边同时关掉某个预测工具**，看像素差是否归零。

这是把"某一层实现有细微偏差"钉到**具体某个工具**上的最快办法，而且**不依赖
"输出帧号 == 解码序帧号"**这个在 B 金字塔下不成立的映射 —— 它只做整帧逐样本对比。

用法：

    probe_neutralize.py <帧数> [用例名 ...]

不带用例名就跑全部；用例名见下面的 `CASES`。工具会把结果打成

    用例名 帧i: 合计 <差异样本数> max <最大差>   [Y:..(max..) U:..(max..) V:..(max..)]

**判读**：某一档 `合计` 归零（或骤降）⇒ 差异就出在那个工具上。
注意两边必须是**同一个语义**的开关：libaom 的 `LERXU_NO_*` 都打在"预测阶段"，
不动熵解码，所以不会改变语法解析。

⚠️ 格式陷阱：`aomdec --i420 --rawvideo` 输出的是**平面式 I420**，而我们的引擎写的是
**NV12（交织 UV）**。整块缓冲直接比对没有意义，必须按平面拆开（本脚本已处理）。
"""
import os
import subprocess
import sys

import numpy as np

W, H = 1920, 1080
CW, CH = W // 2, H // 2
PER = W * H * 3 // 2

AOM = os.environ.get("AOMDEC", "/tmp/libaom-3.9.0/build_g/aomdec")
IVF = os.environ.get("IVF", "/tmp/asmr.ivf")
ME = os.environ.get(
    "ZUVRUST",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "../../../target/release/zuvrust"))
MP4 = os.environ.get("MP4", "/tmp/a.mp4")
OUT = os.environ.get("PROBE_OUT", "/tmp/lerxu-probe")

# 两边都开着的滤波器旋钮（中和实验一律全关，免得环路滤波把差异抹平/扩散）
FILT_LIB = ["LERXU_NO_DEBLOCK=1", "LERXU_NO_CDEF=1", "LERXU_NO_LR=1"]
FILT_ME = ["AV1_NO_DEBLOCK=1", "AV1_NO_CDEF=1", "AV1_NO_LR=1"]

CASES = {
    "base": ([], []),
    "noobmc": (["LERXU_NO_OBMC=1"], ["AV1_NO_OBMC=1"]),
    "nowarp": (["LERXU_NO_WARP=1"], ["AV1_NO_WARP=1"]),
    "nolmwarp": (["LERXU_NO_LOCALWARP=1"], ["AV1_NO_LOCALWARP=1"]),
    "nogmwarp": (["LERXU_NO_GMWARP=1"], ["AV1_NO_GMWARP=1"]),
    "nointerintra": (["LERXU_NO_INTERINTRA=1"], ["AV1_NO_INTERINTRA=1"]),
}


def run_lib(tag, extra, n):
    path = f"{OUT}/ref_{tag}.yuv"
    env = dict(os.environ)
    for k in FILT_LIB + extra:
        env[k.split("=")[0]] = "1"
    subprocess.run([AOM, "--threads=1", "--row-mt=0", "--i420", "--rawvideo",
                    f"--limit={n}", "-o", path, IVF],
                   env=env, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    return path


def run_me(tag, extra, n):
    d = f"{OUT}/my_{tag}"
    env = dict(os.environ)
    for k in FILT_ME + extra:
        env[k.split("=")[0]] = "1"
    subprocess.run([ME, "decode", MP4, f"--count={n}", f"--out={d}", "--json"],
                   env=env, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    return d


def planes_i420(buf):
    """aomdec --i420 是平面式。"""
    c = W * H // 4
    return (buf[:W * H].reshape(H, W),
            buf[W * H:W * H + c].reshape(CH, CW),
            buf[W * H + c:W * H + 2 * c].reshape(CH, CW))


def planes_nv12(buf):
    """我们的引擎写 NV12（Y 平面 + 交织 UV）。"""
    uv = buf[W * H:W * H + W * H // 2].reshape(CH, W)
    return buf[:W * H].reshape(H, W), uv[:, 0::2], uv[:, 1::2]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    tags = sys.argv[2:] or list(CASES)
    os.makedirs(OUT, exist_ok=True)
    print(f"帧数 {n}；aomdec=I420 参考 / 我们=NV12，逐平面比")
    for tag in tags:
        lex, mex = CASES[tag]
        ref = np.fromfile(run_lib(tag, lex, n), dtype=np.uint8)
        run_me(tag, mex, n)          # ← 缺了这一行：我们那一侧从来没跑过
        for i in range(n):
            rp = planes_i420(ref[i * PER:(i + 1) * PER])
            mp = planes_nv12(np.fromfile(f"{OUT}/my_{tag}/frame-{i + 1:06d}.nv12",
                                         dtype=np.uint8))
            tot = mx = 0
            parts = []
            for k, name in enumerate("YUV"):
                d = np.abs(rp[k].astype(int) - mp[k].astype(int))
                c = int((d != 0).sum())
                parts.append(f"{name}:{c}(max{int(d.max())})")
                tot += c
                mx = max(mx, int(d.max()))
            print(f"  {tag:>12} 帧{i}: 合计 {tot} max {mx}   [{' '.join(parts)}]")


if __name__ == "__main__":
    main()
