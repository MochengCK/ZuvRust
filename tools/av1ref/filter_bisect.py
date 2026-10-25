#!/usr/bin/env python3
"""滤波器分级隔离：把**去块 / CDEF / 环路恢复**逐个关掉，与 libaom 的**同配置**输出比。

## 为什么需要它

`refcmp.py` 拿 dav1d 当参考，而 **dav1d 关不掉环路滤波器** —— 一旦怀疑"差异是某一级
滤波器引入的"，就没办法把两边都设成同一种配置来隔离。**libaom 的 aomdec 能关**：

| 开关 | 位置 | 作用 |
|---|---|---|
| `LERXU_NO_DEBLOCK=1` | libaom `av1/decoder/decodeframe.c` | 不调 `av1_loop_filter_frame_mt` |
| `LERXU_NO_CDEF=1` | 同上 | 强制 `do_cdef = 0` |
| `LERXU_NO_LR=1` | 同上 | 强制 `do_loop_restoration = 0` |
| `AV1_NO_DEBLOCK/NO_CDEF/NO_LR=1` | 我们 `filters.rs` / `cdef.rs` / `lr.rs` | 对应地跳过 |

（libaom 自带的 `pbi->skip_loop_filter` **只**盖 CDEF，不够用，所以自己加了三个。）

## 判读

- "全开"就一致 ⇒ 后面不用跑了；
- 关掉某一级之后**变得一致** ⇒ 差异就是那一级引入的；
- **全关之后仍然不一致** ⇒ 差异在**预测/残差**里（这是我们 2026-10-01 在 B 站 ASMR 上
  得到的结论：全关后帧 1 仍有 183 个像素差、max 4）。

## 用法

    filter_bisect.py --frames=1,2 [--src=/tmp/a.mp4] [--ivf=/tmp/asmr.ivf]
"""
import argparse
import os
import subprocess
import sys

import numpy as np

# 引擎与参照解码器：默认取仓库内 release 产物与 /tmp 下的 libaom 构建，
# 换机器时用环境变量覆盖即可（别把本机绝对路径写进仓库）。
ENGINE = os.environ.get("ZUVRUST_BIN", "target/release/zuvrust")
AOMDEC = os.environ.get("AOMDEC", "/tmp/libaom-3.9.0/build_g/aomdec")

CASES = [
    ("all_on", [], []),
    ("no_deblock", ["LERXU_NO_DEBLOCK=1"], ["AV1_NO_DEBLOCK=1"]),
    ("no_cdef", ["LERXU_NO_CDEF=1"], ["AV1_NO_CDEF=1"]),
    ("no_lr", ["LERXU_NO_LR=1"], ["AV1_NO_LR=1"]),
    ("all_off", ["LERXU_NO_DEBLOCK=1", "LERXU_NO_CDEF=1", "LERXU_NO_LR=1"],
     ["AV1_NO_DEBLOCK=1", "AV1_NO_CDEF=1", "AV1_NO_LR=1"]),
]


def run_libaom(tag, envs, count, ivf, w, h):
    os.makedirs("/tmp/fb", exist_ok=True)
    out = f"/tmp/fb/{tag}.yuv"
    env = dict(os.environ)
    for e in envs:
        k, v = e.split("=")
        env[k] = v
    # 注意：**不能带 `--noblit`** —— 那个是"不要处理解码帧"，加了就没有输出帧了。
    cmd = [AOMDEC, "--threads=1", "--row-mt=0", "--i420", "--rawvideo",
           f"--limit={count}", "-o", out, ivf]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print("  aomdec 失败:", (r.stderr or r.stdout)[-300:])
        return None
    raw = np.fromfile(out, dtype=np.uint8)
    per = w * h * 3 // 2
    frames = []
    for i in range(raw.size // per):
        f = raw[i * per : (i + 1) * per]
        frames.append([f[: w * h].reshape(h, w),
                       f[w * h : w * h + w * h // 4].reshape(h // 2, w // 2),
                       f[w * h + w * h // 4 :].reshape(h // 2, w // 2)])
    return frames


def run_mine(tag, envs, count, src, w, h):
    out = f"/tmp/fb/my_{tag}"
    env = dict(os.environ)
    for e in envs:
        k, v = e.split("=")
        env[k] = v
    subprocess.run([ENGINE, "decode", src, f"--count={count}", f"--out={out}", "--json"],
                   env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    frames = []
    for i in range(count):
        p = os.path.join(out, f"frame-{i + 1:06}.nv12")
        if not os.path.exists(p):
            break
        raw = np.fromfile(p, dtype=np.uint8)
        uv = raw[w * h :].reshape(h // 2, w)
        frames.append([raw[: w * h].reshape(h, w), uv[:, 0::2].copy(), uv[:, 1::2].copy()])
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="1,2")
    ap.add_argument("--src", default="/tmp/a.mp4")
    ap.add_argument("--ivf", default="/tmp/asmr.ivf")
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    args = ap.parse_args()
    want = [int(x) for x in args.frames.split(",")]
    count = max(want) + 1
    print(f"比较帧 {want}")
    for tag, le, me in CASES:
        a = run_libaom(tag, le, count, args.ivf, args.w, args.h)
        b = run_mine(tag, me, count, args.src, args.w, args.h)
        if a is None or b is None:
            continue
        line = f"{tag:<12}"
        for i in want:
            if i >= len(a) or i >= len(b):
                line += "  (帧数不足)"
                continue
            tot = mx = 0
            for p in range(3):
                d = np.abs(b[i][p].astype(int) - a[i][p].astype(int))
                tot += int((d != 0).sum())
                mx = max(mx, int(d.max()))
            nn = sum(b[i][p].size for p in range(3))
            line += f"  帧{i}: 不同 {tot}/{nn} = {100 * tot / nn:.4f}%  max {mx}"
        print(line)


if __name__ == "__main__":
    sys.exit(main())
