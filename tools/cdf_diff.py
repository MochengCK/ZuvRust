#!/usr/bin/env python3
"""把规范文本里的默认 CDF 表与 `crates/me-decode/src/video/cdf_tables.rs` 逐值对拍。

用法：
    python3 tools/cdf_diff.py /tmp/av1-clean.txt

为什么需要它：`cdf_tables.rs` 是**机器抽取**的，但抽取器的维度解析、名字改写
（`Default_X_Cdf` → `DEFAULT_X_CDF`）都可能悄悄出错，而"表错了"只会表现为
`exit_symbol()` 失败 —— 极难定位。几百到上万个数字，靠眼睛看不出来。

它同时打印每张表的**维度推断**（数字总数 → 各维乘积），维度对不上会直接报出来。
"""

import re
import sys
from pathlib import Path

SPEC = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/av1-clean.txt")
RUST = Path(__file__).resolve().parent.parent / "crates/me-decode/src/video/cdf_tables.rs"

# 规范里的常量值 → 维度解析用（只列 CDF 声明里出现过的）
CONSTS = {
    "COEFF_CDF_Q_CTXS": 4,
    "TX_SIZES": 5,
    "PLANE_TYPES": 2,
    "LEVEL_CONTEXTS": 21,
    "SIG_COEF_CONTEXTS": 42,
    "SIG_COEF_CONTEXTS_EOB": 4,
    "SIG_COEF_CONTEXTS_2D": 26,
    "BR_CDF_SIZE": 4,
    "TXB_SKIP_CONTEXTS": 13,
    "DC_SIGN_CONTEXTS": 3,
    "EOB_COEF_CONTEXTS": 10,
    "INTRA_MODES": 13,
    "INTERINTRA_MODES": 4,
    "BLOCK_SIZES": 22,
    "MAX_SEGMENTS": 8,
    "SEGMENT_ID_CONTEXTS": 3,
    "SKIP_CONTEXTS": 3,
    "FILTER_INTRA_MODES": 5,
    "DELTA_Q_PROBS": 3,
    "DELTA_LF_PROBS": 3,
    "CFL_ALPHA_CONTEXTS": 6,
    "PARTITION_CONTEXTS": 4,
    "TX_SIZES_ALL": 19,
    "MAX_TX_SIZES": 5,
    "INTRA_MODE_CONTEXTS": 5,
    "UV_MODE_CONTEXTS": 14,
    "ANGLE_DELTA_PROBS": 3,
    "MAX_REF_MV_STACK_SIZE": 8,
    "MV_CONTEXTS": 2,
    "INTERP_FILTER_CONTEXTS": 16,
    "COMP_REF_TYPE_CONTEXTS": 5,
    "COMP_MODE_CONTEXTS": 5,
    "MOTION_MODES": 3,
    "MAX_SEGMENTS_": 8,
    "FWD_REFS": 4,
    "BWD_REFS": 3,
    "UNIDIR_COMP_REFS": 6,
    "TOTAL_REFS_PER_FRAME": 8,
    "REF_CONTEXTS": 5,
    "COMPOUND_TYPES": 3,
    "INTERINTRA_CONTEXTS": 3,
}


def spec_tables():
    text = SPEC.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    out = {}
    for i, line in enumerate(lines):
        m = re.match(
            r"^\s*(Default_[A-Za-z0-9_]+)\s*((?:\[[^\]]+\])+)\s*=\s*\{", line
        )
        if not m:
            continue
        name, dims_s = m.group(1), m.group(2)
        dims = []
        for d in re.findall(r"\[([^\]]+)\]", dims_s):
            d = d.strip()
            if d.isdigit():
                dims.append(int(d))
            else:
                expr = d.replace(" ", "")
                try:
                    dims.append(eval(expr, {}, CONSTS))  # noqa: S307 —— 只喂常量
                except Exception:
                    dims.append(None)
        # 从 `= {` 起按括号配平，取到真正配对的那个 `}`
        joined = "\n".join(lines[i:])
        eq = joined.index("=")
        depth = 0
        k = eq
        while k < len(joined):
            c = joined[k]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        body = joined[eq + 1 : k + 1]
        flat = [int(v) for v in re.findall(r"-?[0-9]+", body)]
        out[name] = (dims, flat)
    return out


def rust_tables():
    text = RUST.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"pub const (DEFAULT_[A-Z0-9_]+):\s*([^=]+)=\s*\[", text):
        name, ty = m.group(1), m.group(2)
        # 从 `[` 开始按括号配平找到结束
        start = m.end() - 1
        depth = 0
        k = start
        while k < len(text):
            if text[k] == "[":
                depth += 1
            elif text[k] == "]":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        body = text[start : k + 1]
        nums = [int(x) for x in re.findall(r"-?[0-9]+", re.sub(r"[\[\]]", " ", body))]
        dims = [int(x) for x in re.findall(r";\s*(\d+)\]", ty)]
        # `[[[u16; 5]; 21]; 2]` 读出来是 [5, 21, 2]（内→外），翻回外→内
        out[name] = (list(reversed(dims)), nums)
    return out


def main():
    spec = spec_tables()
    rust = rust_tables()
    names = sorted(spec)
    bad = 0
    missing = 0
    for name in names:
        rust_name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).upper()
        if rust_name not in rust:
            print(f"缺表：{name} → {rust_name}")
            missing += 1
            continue
        sdims, svals = spec[name]
        rdims, rvals = rust[rust_name]
        prod_s = 1
        for d in sdims:
            prod_s *= d or 0
        status = []
        if len(svals) != len(rvals):
            status.append(f"个数 {len(svals)} vs {len(rvals)}")
        elif svals != rvals:
            first = next(i for i in range(len(svals)) if svals[i] != rvals[i])
            status.append(
                f"首个不同 @ {first}：规范 {svals[first]} vs 本仓库 {rvals[first]}"
            )
        if status:
            bad += 1
            print(f"✗ {name}")
            print(f"    规范维度 {sdims}（乘积 {prod_s}）、实际数字 {len(svals)}")
            print(f"    本仓库维度 {rdims}、实际数字 {len(rvals)}")
            for s in status:
                print(f"    {s}")
            # 给出定位：把扁平序号翻译成多维坐标
            if len(svals) == len(rvals):
                idx = next(i for i in range(len(svals)) if svals[i] != rvals[i])
                coord = []
                rem = idx
                for d in reversed(sdims):
                    coord.append(rem % d)
                    rem //= d
                print(f"    坐标（外→内）{list(reversed(coord))}")
        else:
            print(f"✓ {name}（{len(svals)} 个数，维度 {sdims}）")
    print(f"\n共 {len(names)} 张：不一致 {bad}、缺失 {missing}")
    return 1 if (bad or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
