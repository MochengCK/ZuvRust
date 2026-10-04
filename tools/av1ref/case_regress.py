#!/usr/bin/env python3
"""**受控流回归扫描**：把两个二进制丢到 `make_cases.py` 那批用例上跑，逐条与 dav1d 参考比。

## 为什么需要它（2026-10-03 的教训）

`refcmp.py` 一次只比一条流、还要手工编解码；而 row-mt 那类**重构**的自然验收口径是
"新路径与旧路径逐字节一致" —— **这个口径抓不到"两条路径一起改错"**。
本轮 row-mt 重构就是这样：`ME_ROW_MT` / `ME_ROW_MT_PAR` / `ME_ROW_MT_FULLWAIT` /
`ME_ROW_MT_LIVE_FLAGS` 四种组合输出完全相同，但**全都偏离 dav1d**。

所以要有一条**不依赖自己**的判据：拿 13 条受控流（单帧帧内 / LR / 超块尺寸 /
多帧帧间）当一个小回归集，**一条都不许退化**。它比真素材快得多（512×512 那条
6 帧、秒级），而且能直接指出"是哪一类内容挂的"。

## 用法

```sh
# 参考由 make_cases.py 产出：<out>/<名字>.mp4 + <out>/<名字>_ref.npy
python3 make_cases.py --out=/tmp/refs

# 拿两个二进制对比（旧 vs 新；或 当前 vs 上一版）
python3 case_regress.py --old=./me_v8 --new=./target/release/zuvrust

# 只想看新二进制有没有退化
python3 case_regress.py --new=./target/release/zuvrust
```

## 判读

| 列 | 含义 |
|---|---|
| 旧 / 新 | 该二进制相对 **dav1d 参考**不同的像素数（只比 Y 平面） |
| `← 回归` | 新比旧差 —— **这就是要修的东西** |
| `← 变好` | 新比旧好 |
| `两者同错` | 都不对，但没退化（可能是这版参考本身就没覆盖到） |
| `两者皆对` | 0 差异 |

⚠️ 参考 `.npy` 的形状有两种：多帧是 `(帧, h*3/2, w)`，单帧是 `(h*3/2, w)` ——
单帧那批要自己补一维（本脚本已处理）。
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np


def run_one(binary, mp4, n, out):
    env = dict(os.environ)
    env["ME_THREADS"] = "1"
    r = subprocess.run(
        [binary, "decode", mp4, f"--count={n}", f"--out={out}", "--json"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0:
        return None, f"退出码 {r.returncode}"
    return sorted(f for f in os.listdir(out) if f.endswith(".nv12")), None


def score(binary, mp4, ref):
    """返回 (不同像素数, max|Δ|, 错误说明)。只比 Y 平面。"""
    if ref.ndim == 2:
        ref = ref[None]
    n, H, W = ref.shape[0], ref.shape[1] * 2 // 3, ref.shape[2]
    d = tempfile.mkdtemp(prefix="case_regress_")
    try:
        fs, err = run_one(binary, mp4, n, d)
        if err:
            return None, None, err
        if not fs:
            return None, None, "没有产出帧"
        bad, mx = 0, 0
        for j, f in enumerate(fs[:n]):
            buf = np.fromfile(os.path.join(d, f), dtype=np.uint8)
            if buf.size < W * H:
                return None, None, f"{f} 尺寸不足"
            y = buf[: W * H].reshape(H, W).astype(int)
            dd = np.abs(y - ref[j][:H].astype(int))
            bad += int((dd > 0).sum())
            mx = max(mx, int(dd.max()))
        return bad, mx, None
    finally:
        subprocess.run(["rm", "-rf", d])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", default="/tmp/refs", help="make_cases.py 的产出目录")
    ap.add_argument("--old", default=None, help="基线二进制（可省）")
    ap.add_argument("--new", required=True, help="待测二进制")
    a = ap.parse_args()

    cases = sorted(f for f in os.listdir(a.refs) if f.endswith(".mp4"))
    if not cases:
        print(f"{a.refs} 里没有 .mp4 —— 先跑 make_cases.py --out={a.refs}", file=sys.stderr)
        return 2

    print(f"{'用例':26s} {'帧':>3s} {'旧':>8s} {'新':>8s}  结论")
    regressions = []
    for c in cases:
        mp4 = os.path.join(a.refs, c)
        ref = np.load(os.path.join(a.refs, c[:-4] + "_ref.npy"))
        n = ref.shape[0] if ref.ndim == 3 else 1
        o = score(a.old, mp4, ref) if a.old else (None, None, None)
        nw = score(a.new, mp4, ref)
        if o[0] is None and a.old:
            print(f"{c[:-4]:26s} {n:3d} {'—':>8s} {'—':>8s}  旧二进制跑不动：{o[2]}")
            continue
        otxt = "—" if o[0] is None else str(o[0])
        ntxt = "—" if nw[0] is None else str(nw[0])
        if nw[0] is None:
            verdict = f"✗ 新二进制跑不动：{nw[2]}"
        elif o[0] is None:
            verdict = "✅ 无差异" if nw[0] == 0 else f"✗ 与参考差 {nw[0]} 点（max {nw[1]}）"
        elif nw[0] > o[0]:
            verdict = f"← 回归（max {nw[1]}）"
        elif nw[0] < o[0]:
            verdict = "← 变好"
        elif nw[0] == 0:
            verdict = "两者皆对"
        else:
            verdict = "两者同错"
        if nw[0] is not None and o[0] is not None and nw[0] > o[0]:
            regressions.append(c[:-4])
        print(f"{c[:-4]:26s} {n:3d} {otxt:>8s} {ntxt:>8s}  {verdict}")

    if regressions:
        print(f"\n退化 {len(regressions)} 条：{', '.join(regressions)}")
        return 1
    print("\n没有退化。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
