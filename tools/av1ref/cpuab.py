#!/usr/bin/env python3
"""**子进程 CPU 时间**的交替 A/B 基准：拿两个 bin 的 `decode` 各跑一轮，比 CPU 时间。

## 为什么不用墙钟

这台机器（MacBookAir9,1 / i5-1030NG7）负载会飘：同一对二进制在同一窗口里，
`PHASE` 总和的 min 与 median 差 **20%**（内部优化记录 第九节记过 50.6 vs 61.6 ms）。
墙钟 `elapsedMs` / `PHASE` **只在两版都稳定的窗口里**才有意义。

CPU 时间（`getrusage` 的 `ru_utime + ru_stime`）量的是"这条进程真正花了多少 CPU"，
与同时有多少别的进程抢核**无关** —— 代价是它把"多线程并行"折成"总工作量"，
所以**只在同一线程配置下比较**（本脚本默认 `ME_THREADS=1`）。

## 口径

* 每轮 A、B 相邻跑（同一时间窗），交替顺序（`--swap` 打开）；
* 取 **配对比值 `B_i / A_i` 的中位数** —— 比"中位之比"稳，因为它抵消了窗口漂移；
* 同时报"B 更快的轮数 / 总轮数"。

## 用法

```sh
cd ZuvRust
export PATH="$HOME/.cargo/bin:$PATH" && cargo build --release

python3 tools/av1ref/cpuab.py base=/tmp/me_base new=./target/release/zuvrust \\
    --count=120 --rounds=12

# 多组环境变量（KEY=VAL 可重复）
python3 tools/av1ref/cpuab.py a=/tmp/x b=/tmp/y --env ME_THREADS=1 --env AV1_NO_FILTERS=1
```

⚠️ 跑之前先把残留进程清掉：`pgrep -fl zuvrust`。
"""

import argparse
import os
import resource
import statistics
import subprocess
import sys
import time


def cpu_time_of(path, source, count, env_extra):
    """跑一次 `decode --count=N`（不落盘），返回 `(cpu_seconds, wall_seconds)`。"""
    env = dict(os.environ)
    env.update(env_extra)
    cmd = [path, "decode", source, f"--count={count}", "--json"]
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall = time.time() - t0
    if p.returncode != 0:
        sys.exit(f"{path} 退出码 {p.returncode}：\n{p.stderr[-2000:]}")
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    return cpu, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bins", nargs="+", help="名字=路径（第一个是基线 A，其余是 B）")
    ap.add_argument("--source", default="/tmp/a.mp4")
    ap.add_argument("--count", type=int, default=120)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--env", action="append", default=[], help="KEY=VAL，可重复")
    ap.add_argument("--swap", action="store_true", default=True,
                    help="偶数轮把顺序反过来（默认开）")
    args = ap.parse_args()

    env_extra = {}
    for kv in args.env:
        k, _, v = kv.partition("=")
        env_extra[k] = v
    env_extra.setdefault("ME_THREADS", "1")

    configs = []
    for spec in args.bins:
        name, _, path = spec.partition("=")
        if not path:
            sys.exit(f"参数要写成 名字=路径：{spec}")
        configs.append((name, path))

    left = subprocess.run(["pgrep", "-x", "zuvrust"], capture_output=True, text=True)
    if left.stdout.strip():
        print("⚠️ 还有 zuvrust 进程在跑，先清掉再量：\n" + left.stdout)

    runs = {name: [] for name, _ in configs}
    # 预热一次，把首次 mmap / 页缓存差异摊掉
    for _, path in configs:
        cpu_time_of(path, args.source, 4, env_extra)

    for r in range(args.rounds):
        order = list(configs)
        if args.swap and r % 2 == 1:
            order = order[::-1]
        for name, path in order:
            cpu, _ = cpu_time_of(path, args.source, args.count, env_extra)
            runs[name].append(cpu)
        print(f"  round {r + 1}/{args.rounds} 完成", file=sys.stderr)

    base = configs[0][0]
    print(f"\n交替 {args.rounds} 轮，{args.count} 帧，env={env_extra}")
    print(f"{'配置':<10} {'CPU 中位(s)':>12} {'min':>8} {'max':>8} {'ms/帧':>8}")
    for name, _ in configs:
        xs = runs[name]
        med = statistics.median(xs)
        print(f"{name:<10} {med:12.3f} {min(xs):8.3f} {max(xs):8.3f} {med * 1000 / args.count:8.2f}")

    if len(configs) >= 2:
        xs = runs[base]
        for name, _ in configs[1:]:
            ys = runs[name]
            pa = [y / x for x, y in zip(xs, ys)]
            wins = sum(1 for p in pa if p < 1.0)
            print(
                f"\n{name} / {base}：配对中位 {statistics.median(pa):.4f}"
                f"（{(statistics.median(pa) - 1) * 100:+.1f}%），"
                f"{name} 更快 {wins}/{len(pa)} 轮"
            )


if __name__ == "__main__":
    main()
