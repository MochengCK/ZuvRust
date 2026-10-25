#!/usr/bin/env python3
"""同窗口交替 A/B：拿**进程内**指标（`AV1_PHASE_TIMING` 的 `PHASE` 行）比较两个二进制。

## 为什么必须这么量

本机负载会飘（8 线程机上见过 load ≈ 10），同一个二进制的 `elapsedMs` 在不同时间窗
能差 1.5~2x（内部优化记录 第七节记过 71/96/113/197 ms 四个数）。所以：

* **绝对值一律不可信**，只有"同一时间窗内交替跑 A/B、多轮取中位"能比较；
* <5% 的差异**别信 `elapsedMs`** —— 用进程内的 `PHASE` 求和（它只包解码本身，
  不含落盘 IO 与预热），同一轮里 A/B 相邻跑，机器漂移两边同吃。

## 用法

```sh
cd ZuvRust
export PATH="$HOME/.cargo/bin:$PATH" && cargo build --release

python3 tools/av1ref/perfab.py base=/tmp/me_base new=./target/release/zuvrust \\
    --count=100 --rounds=3 --env ME_THREADS=1
```

输出每个配置的 `tiles / filters / 其它 / 合计`（ms/帧，取各轮中位）与相对基线的
**加速比**；`--count` 是解码帧数，`--env` 可重复（`KEY=VAL`）。

⚠️ 跑之前先 `pgrep -fl zuvrust` 清残留进程：漏杀一个就整轮偏高。
"""

import argparse
import os
import re
import statistics
import subprocess
import sys

PHASE = re.compile(
    r"PHASE frame=(\d+) shown=(\d+) tiles=([\d.]+)ms filters=([\d.]+)ms"
    r"(?: ref=([\d.]+)ms)?(?: out=([\d.]+)ms)? setup=([\d.]+)ms"
    r"(?: mc=([\d.]+)ms txb=([\d.]+)ms mark=([\d.]+)ms)?"
)


def run_once(binary, source, count, env_extra, out_dir):
    env = dict(os.environ)
    env.update(env_extra)
    env["AV1_PHASE_TIMING"] = "1"
    cmd = [binary, "decode", source, f"--count={count}", "--json"]
    if out_dir:
        cmd.append(f"--out={out_dir}")
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"{binary} 退出码 {p.returncode}：\n{p.stderr[-2000:]}")
    rows = [m.groups() for m in PHASE.finditer(p.stderr)]
    if not rows:
        sys.exit(f"{binary} 没有 PHASE 输出（是不是没开 AV1_PHASE_TIMING？）")
    n = len(rows)
    f = lambda i: [float(r[i] or 0.0) for r in rows]
    elapsed = 0.0
    m = re.search(r'"elapsedMs":(\d+)', p.stdout)
    if m:
        elapsed = float(m.group(1))
    return {
        "frames": n,
        "tiles": sum(f(2)) / n,
        "filters": sum(f(3)) / n,
        "ref": sum(f(4)) / n,
        "out": sum(f(5)) / n,
        "setup": sum(f(6)) / n,
        "mc": sum(f(7)) / n,
        "txb": sum(f(8)) / n,
        "mark": sum(f(9)) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bins", nargs="+", help="名字=路径（第一个是基线）")
    ap.add_argument("--source", default="/tmp/a.mp4")
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--env", action="append", default=[], help="KEY=VAL，可重复")
    ap.add_argument("--out", default="", help="落盘目录（会掺进 IO，默认不落盘）")
    args = ap.parse_args()

    env_extra = {}
    for kv in args.env:
        k, _, v = kv.partition("=")
        env_extra[k] = v

    configs = []
    for spec in args.bins:
        name, _, path = spec.partition("=")
        if not path:
            sys.exit(f"参数要写成 名字=路径：{spec}")
        configs.append((name, path))

    left = subprocess.run(["pgrep", "-fl", "zuvrust"], capture_output=True, text=True)
    if left.stdout.strip():
        print("⚠️ 还有 zuvrust 进程在跑，先清掉再量：\n" + left.stdout)

    acc = {name: [] for name, _ in configs}
    for r in range(args.rounds):
        for name, path in configs:
            acc[name].append(run_once(path, args.source, args.count, env_extra, args.out))
        print(f"  round {r + 1}/{args.rounds} 完成", file=sys.stderr)

    med = {name: {k: statistics.median([a[k] for a in rows]) for k in rows[0]}
           for name, rows in acc.items()}
    base_name = configs[0][0]
    base_total = sum(med[base_name][k] for k in ("tiles", "filters", "ref", "out", "setup"))

    print(f"\n同窗口交替 {args.rounds} 轮取中位，{args.count} 帧"
          f"{'，env=' + str(env_extra) if env_extra else ''}")
    print(f"{'配置':<10} {'tiles':>8} {'filters':>8} {'mc':>7} {'txb':>7} {'其它':>6} "
          f"{'合计':>8} {'相对':>7}")
    for name, _ in configs:
        m = med[name]
        other = m["ref"] + m["out"] + m["setup"]
        total = m["tiles"] + m["filters"] + other
        rel = base_total / total if total else 0.0
        print(f"{name:<10} {m['tiles']:8.2f} {m['filters']:8.2f} {m['mc']:7.2f} "
              f"{m['txb']:7.2f} {other:6.2f} {total:8.2f} {rel:6.3f}x")


if __name__ == "__main__":
    main()
