#!/usr/bin/env python3
"""**参考解码器单线程基准**：dav1d（PyAV）与本机 AV1 软解的同口径 ms/帧。

## 为什么要有它

PERF-HANDOFF 的目标是"**单线程** 33 ms/帧"，而对照物 dav1d 的数字必须**同一口径**：
PyAV 的 `codec_context.threads` **不会**作用到解码器 —— 要显式设
`thread_count = 1` + `thread_type = "NONE"` 才是单线程（否则测到的是多线程的数，
2026-10-04 就因为这条把 8.6 ms 当成单线程数）。这个脚本把这两条钉死。

同时它给出**本机的负载系数**：同一时间窗里"我们的 ms/帧 ÷ dav1d 的 ms/帧"才是有意义
的相对值 —— 绝对毫秒在本机会飘到 2 倍。

## 用法

```sh
/tmp/avvenv/bin/python3 tools/av1ref/dav1d_bench.py /tmp/a.mp4 --n=60            # 只量 dav1d
/tmp/avvenv/bin/python3 tools/av1ref/dav1d_bench.py /tmp/a.mp4 --n=60 --me=./target/release/zuvrust
```

`--me` 会给自研软解加 `ME_THREADS=1` 并跑 `decode --count=N`（不落盘，不含 IO），
用 `elapsedMs / 帧数` 作口径 —— 与 `PHASE` 求和不同，这一项里含有解封装与预热，
两个数都给出来（`PHASE` 那个更纯）。
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time


def dav1d_single_thread(path, n, rounds):
    """真正的单线程 dav1d（`thread_count=1` + `thread_type=NONE`）。"""
    import av

    out = []
    for _ in range(rounds):
        c = av.open(path)
        c.streams.video[0].thread_type = "NONE"
        cc = c.streams.video[0].codec_context
        cc.thread_count = 1
        cc.thread_type = "NONE"
        cnt = 0
        t0 = None
        for _fr in c.decode(video=0):
            if cnt == 0:
                t0 = time.perf_counter()
            cnt += 1
            if cnt >= n:
                break
        dt = time.perf_counter() - t0
        out.append(dt * 1e3 / max(cnt - 1, 1))
    return out


def bench_me(binary, path, n):
    env = dict(os.environ)
    env["ME_THREADS"] = "1"
    env["AV1_PHASE_TIMING"] = "1"
    p = subprocess.run(
        [binary, "decode", path, f"--count={n}", "--json"],
        env=env, capture_output=True, text=True,
    )
    if p.returncode != 0:
        sys.exit(f"{binary} 退出码 {p.returncode}：\n{p.stderr[-1500:]}")
    rows = re.findall(r"PHASE frame=(\d+) shown=(\d+) tiles=([\d.]+)ms filters=([\d.]+)ms", p.stderr)
    # ⚠️ 输出里**不止一条 JSON**（还有 `decode-progress` 行，它的 `frames` 是中途值）——
    # 必须只认最后那条带 `"result"` 的，否则算出 2x 的假 ms/帧（踩过）。
    result = None
    for line in p.stdout.splitlines():
        if '"result"' in line:
            result = json.loads(line)["result"]
    if result is None:
        sys.exit(f"没找到 decode 结果行：\n{p.stdout[-1000:]}")
    frames = int(result["frames"])
    wall = float(result["elapsedMs"]) / frames
    phase = statistics.mean(float(r[2]) + float(r[3]) for r in rows)
    return wall, phase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default="/tmp/a.mp4")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--me", default="", help="自研软解二进制（可选）")
    args = ap.parse_args()

    d = dav1d_single_thread(args.source, args.n, args.rounds)
    print(f"dav1d 单线程        ms/帧: 最优 {min(d):6.2f}  中位 {statistics.median(d):6.2f}")
    if args.me:
        wall, phase = bench_me(args.me, args.source, args.n)
        print(f"我们（ME_THREADS=1） ms/帧: 墙钟 {wall:6.2f}  PHASE(tiles+filters) {phase:6.2f}")
        print(f"相对 dav1d：墙钟 {wall / min(d):.2f}x，PHASE {phase / min(d):.2f}x"
              f"（目标 2.5x ⇒ {min(d) * 2.5:.1f} ms/帧）")


if __name__ == "__main__":
    main()
