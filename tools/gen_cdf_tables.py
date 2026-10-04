#!/usr/bin/env python3
"""从 AV1 规范原文里抽出全部 `Default_*` 表，生成 Rust 源码。

## 为什么要有这个脚本

AV1 的默认 CDF 表是**上千个魔数**（只 `Default_Coeff_Base_Cdf` 一张就有
4×5×2×42×5 = 8400 个数）。手抄必然出错，而且抄错一个数只会让算术解码在
某处悄悄偏一位 —— 这种 bug 极难定位。所以：**表一律机器抽取，不手写**。

## 用法

```text
curl -sL -o /tmp/av1-spec.html https://aomediacodec.github.io/av1-spec/
python3 tools/gen_cdf_tables.py /tmp/av1-spec.html \
    > crates/me-decode/src/video/cdf_tables.rs
```

输入可以是规范的 HTML（会先去掉标签）或已经转好的纯文本。

## 输出

- 每张表一个 `pub const DEFAULT_XXX: <嵌套数组>`；
- 从**解析出来的结构**推数组维度，并跟规范里写的维度（数字形式）对照，
  对不上就报错 —— 那是"表没抽全"的信号，必须在生成期拦住。

脚本对每个值只接受数字与 `+ * ( )`，并且会拒绝任何残余的非数字 token，
保证不会把规范里的普通文字误当数据。
"""

from __future__ import annotations

import html
import re
import sys

# 生成的文件抬头（手写，提醒后来人别直接改这个文件）
HEADER = '''//! AV1 规范 §9 的**默认表**（CDF 表与扫描顺序表）—— **自动生成，不要手改**。
//!
//! 生成方式：
//!
//! ```text
//! curl -sL -o /tmp/av1-spec.html https://aomediacodec.github.io/av1-spec/
//! python3 tools/gen_cdf_tables.py /tmp/av1-spec.html \\
//!     > crates/me-decode/src/video/cdf_tables.rs
//! ```
//!
//! 为什么用生成的：这些表加起来上万个数（光 `DEFAULT_COEFF_BASE_CDF` 就有
//! 8400 个），手抄错一个数只会让算术解码在某个符号上悄悄偏一位 —— 那种 bug
//! 几乎不可能靠读代码发现。表必须从规范原文抽取。
//!
//! 数组维度是从原文结构推出来的，并与规范里标注的维度对照过（见
//! `tools/gen_cdf_tables.py`）。

#![allow(clippy::all)]
#![allow(dead_code)] // 表是全套抽出来的，代码按层逐步用到，没用到的不该报警

'''


def html_to_text(src: str) -> str:
    src = re.sub(r"<script.*?</script>", "", src, flags=re.S | re.I)
    src = re.sub(r"<style.*?</style>", "", src, flags=re.S | re.I)
    src = re.sub(r"<[^>]+>", "", src)
    return html.unescape(src)


def eval_value(expr: str) -> int:
    """求值表里的一个数字项（只允许数字与 + * ( )）。"""
    cleaned = expr.strip()
    if not re.fullmatch(r"[0-9+*()\s]+", cleaned):
        raise ValueError(f"不认识的数值写法：{expr!r}")
    return int(eval(cleaned, {"__builtins__": {}}, {}))  # noqa: S307（字符集已限死）


def parse_body(body: str) -> list:
    """把 `{ a, b }, { c, d }` 这种表体解析成嵌套 list。"""
    # 先把 `128 * 125` 这类算式折叠成一个数，剩下的就只剩数字与括号
    def fold(text: str) -> str:
        out = text
        while True:
            new = re.sub(r"(\d+)\s*\*\s*(\d+)", lambda m: str(int(m.group(1)) * int(m.group(2))), out)
            new = re.sub(r"(\d+)\s*\+\s*(\d+)", lambda m: str(int(m.group(1)) + int(m.group(2))), new)
            if new == out:
                return out
            out = new

    text = fold(body)
    # 括号内不做折叠的残留检查：只允许数字、逗号、空白、花括号
    leftovers = re.sub(r"[0-9,\s{}]", "", text)
    if leftovers:
        raise ValueError(f"表体里有残余 token：{leftovers[:40]!r}")

    # 递归下降：`stack[-1]` 是当前正在填的数组。**扁平表**（扫描顺序表就是
    # `= { 0, 1, 4, 8, ... }`）与**嵌套表**（CDF）走同一条路径。
    stack: list[list] = [[]]
    token = ""

    def flush() -> None:
        nonlocal token
        if token:
            stack[-1].append(int(token))
            token = ""

    for ch in text:
        if ch.isdigit():
            token += ch
        elif ch == "{":
            flush()
            stack.append([])
        elif ch == "}":
            flush()
            done = stack.pop()
            if len(stack) == 0:
                raise ValueError("花括号多了一个 `}`")
            stack[-1].append(done)
        elif ch == ",":
            flush()
    flush()
    if len(stack) != 1:
        raise ValueError("花括号不配对")
    root = stack[0]
    # `= { { ... }, { ... } }` 这种写法里，body 是紧跟 `=` 的那个 `{` 之内，
    # 但规范有时会把整张表写成单个 `{ ... }`，此时 root 只有一项。
    if len(root) == 1 and isinstance(root[0], list):
        root = root[0]
    if not root:
        raise ValueError("空表")
    return root


def shape_of(value) -> tuple:
    if isinstance(value, list):
        shapes = {shape_of(v) for v in value}
        if len(shapes) != 1:
            raise ValueError(f"数组不规则：{shapes}")
        return (len(value),) + shapes.pop()
    return ()


def dim_numbers(dim_text: str) -> list:
    """把 `[ A ][ 3 ]` 里的维度解析成数字列表（符号维度返回 -1）。"""
    out = []
    for part in re.findall(r"\[([^\]]+)\]", dim_text):
        p = part.strip()
        if re.fullmatch(r"\d+", p):
            out.append(int(p))
        elif re.fullmatch(r"[\d+*\s()]+", p):
            out.append(eval_value(p))
        else:
            out.append(-1)
    return out


def rust_type(shape: tuple) -> str:
    """shape = (M, N, ...) → `[[u16; N]; M]` 这样的 Rust 类型。"""
    t = "u16"
    for n in reversed(shape[1:]):
        t = f"[{t}; {n}]"
    return f"[{t}; {shape[0]}]"


def rust_literal(value, indent: int) -> str:
    pad = "    " * indent
    if isinstance(value, list):
        if not value:
            return "[]"
        inner = ",\n".join(rust_literal(v, indent + 1) for v in value)
        return "[\n" + inner + ",\n" + pad + "]"
    return str(value)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/av1-spec.html"
    raw = open(path, encoding="utf-8", errors="replace").read()
    text = html_to_text(raw) if "<" in raw[:2000] else raw

    # 三类表：`Default_*`（CDF 与默认扫描）、`Mrow_Scan_*` / `Mcol_Scan_*`
    # （行优先/列优先扫描 —— 非 2D 变换类型要用，见 §5.11.39 的 `get_scan`）。
    pattern = re.compile(
        r"((?:Default|Mrow_Scan|Mcol_Scan)_[A-Za-z0-9_]+)((?:\s*\[[^\]]+\])+)\s*=\s*\{"
    )
    out: list[str] = [HEADER]
    count = 0

    for m in pattern.finditer(text):
        name, dim_text = m.group(1), m.group(2)
        start = m.end()  # 紧跟在第一个 `{` 之后
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth != 0:
            raise SystemExit(f"{name}: 花括号不闭合")

        value = parse_body(text[start : i - 1])
        shape = shape_of(value)
        declared = dim_numbers(dim_text)
        for got, want in zip(shape, declared):
            if want >= 0 and got != want:
                raise SystemExit(f"{name}: 维度对不上，解析出 {shape}，规范写 {declared}")

        # 规范里的表名本来就是 `Default_Xxx_Yyy_Cdf` 这种下划线分段，直接大写即可
        # （不要按大写字母再插下划线 —— 那会造出 `DEFAULT__PARTITION__W8__CDF`）
        const = name.upper()
        out.append(f"/// `{name}{dim_text.strip()}`（规范 §9）\n")
        out.append(f"pub const {const}: {rust_type(shape)} = ")
        out.append(rust_literal(value, 0))
        out.append(";\n\n")
        count += 1

    if count == 0:
        raise SystemExit("一张表都没抽到 —— 输入文件不对？")

    sys.stdout.write("".join(out))
    print(f"// 共 {count} 张表", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
