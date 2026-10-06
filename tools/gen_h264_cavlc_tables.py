#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 ITU-T H.264 规范（Rec. ITU-T H.264 (V15) (08/2024)）里**抄**出 CAVLC 的 VLC 表。

## 为什么要脚本抄，而不是手打

`coeff_token` 那张表有 6 个 VLC、上百个码字，`total_zeros` 三张、`run_before`
一张 —— 手打一遍必然出错，而且错一位是**静默**的（解出来的系数级不对，但语法
照样走得下去，最后表现为"画面糊了一层"这种最难查的症状）。脚本抄 +
**前缀码自检（同一列里码字不能互为前缀，且 Kraft 和必须正好是 1）**
能把"抄漏/抄重/抄错位"当场抓出来。

## 用法

```sh
# 1) 下载规范（ITU 站点有 WAF，要带 cookie + 浏览器 UA + Referer）
curl -sSL -c /tmp/jar -A "Mozilla/5.0" \
  https://www.itu.int/rec/T-REC-H.264-202408-S/en -o /dev/null
curl -sSL -b /tmp/jar -c /tmp/jar -A "Mozilla/5.0" \
  -e "https://www.itu.int/rec/T-REC-H.264-202408-S/en" \
  "https://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-H.264-202408-S!!PDF-E&type=items" \
  -o /tmp/h264.pdf

# 2) 提成**带版面**的文本（layout 模式才保得住表的列）
python3 - <<'PY'
from pypdf import PdfReader
r = PdfReader("/tmp/h264.pdf", strict=False)
with open("/tmp/h264-layout.txt", "w", encoding="utf-8") as f:
    for i, p in enumerate(r.pages):
        f.write(f"\n===== PAGE {i+1} =====\n" + p.extract_text(extraction_mode="layout"))
PY

# 3) 生成
python3 tools/gen_h264_cavlc_tables.py /tmp/h264-layout.txt \
  > crates/me-decode/src/video/h264/cavlc_tables.rs
```

## 版面解析的两个坑

1. `extract_text()` 默认模式会把整行拍平，列就丢了；`layout` 模式保住列，但
   空格宽度会抖，所以按"2 个以上空格"切字段（码字内部的 4 位分组是**单个**空格，
   不会误切）。
2. 规范排版偶尔漏印 `-`（比如表 9-10 的 `run_before = 8` 那行少两个），所以下面
   按**已知的合法单元**去对齐，而不是硬数字段个数 —— 合法与否由表自己的规则给出
   （`total_zeros ≤ maxNumCoeff - T`、`run_before ≤ zerosLeft`），
   对不上的 `-` 无论印没印都跳过。
"""

import re
import sys

PAGE_RE = re.compile(r"===== PAGE (\d+) =====")


def pages(text):
    parts = PAGE_RE.split(text)
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}


def rows(page_text):
    """把"像表格行"的行拆成字段列表（按 2+ 空格切）。"""
    out = []
    for line in page_text.splitlines():
        if not line.strip():
            continue
        fields = [f for f in re.split(r"\s{2,}", line.strip()) if f]
        if len(fields) >= 3:
            out.append(fields)
    return out


def bits(tok):
    """`0000 11` → `(长度, 值)`；`-` → None。"""
    if tok == "-":
        return None
    s = tok.replace(" ", "")
    if not re.fullmatch(r"[01]+", s):
        return None
    return (len(s), int(s, 2))


def check_prefix_free(name, codes, must_be_complete=False):
    """前缀码自检：同列里不能互为前缀；Kraft 和**不能大于 1**。

    ## 为什么这里的判据是"≤ 1"而不是"= 1"

    本以为是完整前缀码，于是拿"Kraft 和 = 1"当抄写校验 —— **结果规范自己的表
    就不满足**：`0 <= nC < 2` 只用掉 1 − 2⁻¹⁵，`2 <= nC < 4` 差 2⁻¹³，
    `4 <= nC < 8` 差 2⁻¹⁰，`8 <= nC` 那一列**全是 6 位码**、64 个槽只用了 62 个。
    逐个枚举码树，空洞都是"全零那一支"（`lt2` 是 `0…0`（15 位）、`ge8` 是
    `000010` / `000111`）—— 也就是说规范的表**有意留了不用的槽**。

    这不是抄错：ITU-T H.264 (2019) 与 (2024) 两版**逐字符一致**（都验过）。
    所以判据只能是"前缀无关 + Kraft ≤ 1"（> 1 一定是抄重了），
    再叠一个**符号个数**断言。真正的"有没有抄错一位"由
    `crates/me-decode` 那条"解完一个切片恰好停在 rbsp 末尾"的绝对判据兜底。
    """
    ordered = sorted(codes)
    for i, (la, ca) in enumerate(ordered):
        for lb, cb in ordered[i + 1:]:
            if lb >= la and (cb >> (lb - la)) == ca:
                raise SystemExit(f"{name}: 码字互为前缀 {ca:0{la}b} / {cb:0{lb}b}")
    kraft = sum(2.0 ** (-l) for l, _ in ordered)
    if kraft > 1.0 + 1e-12:
        raise SystemExit(f"{name}: Kraft 和 = {kraft} > 1（一定抄重了）")
    if must_be_complete and abs(kraft - 1.0) > 1e-12:
        raise SystemExit(f"{name}: Kraft 和 = {kraft} ≠ 1")
    if len(ordered) != len(set(ordered)):
        raise SystemExit(f"{name}: 有重复码字")
    return kraft


def segment(page_text, start, end=None):
    """截出 `start` 之后、`end` 之前的那一段（表与表之间夹着正文，必须切开）。"""
    body = page_text.split(start, 1)[1]
    if end is not None and end in body:
        body = body.split(end, 1)[0]
    return body


def data_rows(text, skip_header=False):
    """只留数据行。

    `skip_header=True` 时会**跳掉第一条"整行都是数字"的行** —— 那是 `total_zeros` /
    `run_before` 那几张表的列标（`1 2 3 …`、`1 2 3 4 5 6 >6`），它排在所有数据行前面。

    ⚠️ 这个开关**不能默认开**：表 9-4 的第一条数据行恰好也是"整行都是数字"
    （`0 47 0`），默认开就会把它当列标吃掉，于是 48 行只剩 47 行 —— 而
    "codeNum 应当是 0..47" 的条数断言正好能抓出来（这就是它的用处）。
    """
    out = []
    skipped_header = False
    for f in rows(text):
        if not f[0].isdigit():
            continue
        if skip_header and not skipped_header:
            digitish = all(re.fullmatch(r"\d+|>\d+", x) for x in f)
            if digitish:
                skipped_header = True
                continue
        out.append(f)
    return out


def take(toks, ti, name):
    if ti >= len(toks):
        raise SystemExit(f"{name}: 字段不够了")
    b = bits(toks[ti])
    if b is None:
        raise SystemExit(f"{name}: 这一格印的是 `-`，但按表规则它必须有码字")
    return b, ti + 1


# ── 表 9-5：coeff_token ──────────────────────────────────────────────────────

NC_COLUMNS = ["lt2", "2to4", "4to8", "ge8", "cm1", "cm2"]


def parse_coeff_token(pg):
    """跨 PDF 243~245 页：`TrailingOnes TotalCoeff 码字×6`。"""
    tables = {c: [] for c in NC_COLUMNS}
    for page_no in (243, 244, 245):
        for f in rows(pg[page_no]):
            if len(f) < 8 or not (f[0].isdigit() and f[1].isdigit()):
                continue
            t1, total = int(f[0]), int(f[1])
            for col, tok in zip(NC_COLUMNS, f[2:8]):
                # 各列自己的上界：nC == −1 是 4:2:0 的色度 DC（2x2 ⇒ 4 个系数），
                # nC == −2 是 4:2:2 的色度 DC（2x4 ⇒ 8 个系数）。
                # ⚠️ 这两个**别按"−1 小 −2 大"想当然**：规范表里印得很清楚 ——
                # `nC == −1` 那一列到 TotalCoeff = 4 就没了，`nC == −2` 那一列到 8。
                if col == "cm1" and total > 4:
                    continue
                if col == "cm2" and total > 8:
                    continue
                # 合法组合：TotalCoeff = 0 时只有 TrailingOnes = 0；
                # 否则 TrailingOnes ≤ min(3, TotalCoeff)
                legal = (total == 0 and t1 == 0) or (total >= 1 and t1 <= min(3, total))
                if not legal:
                    continue
                b = bits(tok)
                if b is None:
                    raise SystemExit(f"表 9-5 ({t1},{total}) 的 {col} 列没有码字")
                tables[col].append((b[0], b[1], t1, total))
    expect = {"lt2": 62, "2to4": 62, "4to8": 62, "ge8": 62, "cm1": 14, "cm2": 30}
    for col in NC_COLUMNS:
        if len(tables[col]) != expect[col]:
            raise SystemExit(
                f"coeff_token[{col}]: 抄到 {len(tables[col])} 条，应当是 {expect[col]} 条"
            )
        check_prefix_free(f"coeff_token[{col}]", [(l, c) for l, c, _, _ in tables[col]])
    return tables


# ── 表 9-7 / 9-8：total_zeros（4x4 / 8x8） ───────────────────────────────────

def parse_total_zeros(page_text, tz_ids, max_coeff, header, end_marker=None):
    """按"已知合法单元"对齐：`tzVlcIndex = T` 时 `total_zeros ≤ maxNumCoeff - T`。"""
    body = segment(page_text, header, end_marker)
    cols = {T: [] for T in tz_ids}
    for f in data_rows(body, skip_header=True):
        if not f[0].isdigit():
            continue
        tz = int(f[0])
        ti = 1
        for T in tz_ids:
            # `tzVlcIndex = T` 时 `total_zeros` 只能到 `maxNumCoeff - T`（含），
            # 也就是 `maxNumCoeff - T + 1` 个取值 —— 差一位就会把整张表读串行。
            if tz > max_coeff - T:
                if ti < len(f) and f[ti] == "-":
                    ti += 1
                continue
            b, ti = take(f, ti, f"total_zeros 行 {tz} / tzVlcIndex {T}")
            cols[T].append((b[0], b[1], tz))
    for T in tz_ids:
        if len(cols[T]) != max_coeff - T + 1:
            raise SystemExit(
                f"total_zeros[T={T}]: 抄到 {len(cols[T])} 条，应当是 {max_coeff - T + 1} 条"
            )
        check_prefix_free(f"total_zeros[T={T}]", [(l, c) for l, c, _ in cols[T]])
    return cols


# ── 表 9-9(a)：色度 DC 2x2 的 total_zeros ────────────────────────────────────

def parse_tz_chroma_dc(page_text, tz_ids, max_coeff):
    cols = {T: [] for T in tz_ids}
    body = segment(page_text, "Table 9-9", "(b) Chroma DC 2x4")
    for f in data_rows(body, skip_header=True):
        if not f[0].isdigit():
            continue
        tz = int(f[0])
        ti = 1
        for T in tz_ids:
            if tz > max_coeff - T:
                if ti < len(f) and f[ti] == "-":
                    ti += 1
                continue
            b, ti = take(f, ti, f"色度 DC total_zeros 行 {tz} / tzVlcIndex {T}")
            cols[T].append((b[0], b[1], tz))
    for T in tz_ids:
        if len(cols[T]) != max_coeff - T + 1:
            raise SystemExit(f"色度 DC total_zeros[T={T}]: 条数不对")
        check_prefix_free(f"chroma_dc_total_zeros[T={T}]", [(l, c) for l, c, _ in cols[T]])
    return cols


# ── 表 9-10：run_before ─────────────────────────────────────────────────────

def parse_run_before(page_text):
    """列顺序：`zerosLeft = 1..6` 与最后一列 `>6`（合法到 `run_before = 14`）。"""
    body = segment(page_text, "Table 9-10")
    zl = [1, 2, 3, 4, 5, 6, 7]  # 7 代表 ">6"
    # `zerosLeft = z` 时 `run_before` 能到 `z`（含）；最后一列（>6）到 14。
    # ⚠️ 是 `≤ z` 不是 `< z` —— 数错一位整列就串行了，"抄到几条"的断言会当场炸。
    max_run = {z: (z if z <= 6 else 14) for z in zl}
    cols = {z: [] for z in zl}
    for f in data_rows(body, skip_header=True):
        if not f[0].isdigit():
            continue
        r = int(f[0])
        ti = 1
        for z in zl:
            if r > max_run[z]:
                if ti < len(f) and f[ti] == "-":
                    ti += 1
                continue
            b, ti = take(f, ti, f"run_before 行 {r} / zerosLeft {z}")
            cols[z].append((b[0], b[1], r))
    # 注意：这些列**各自不是一个完整前缀码**（它们共享同一棵更大的树），
    # 所以这里只查"同列内前缀无关"，不查 Kraft 和。
    for z in zl:
        if len(cols[z]) != max_run[z] + 1:
            raise SystemExit(
                f"run_before[zerosLeft={z}]: 抄到 {len(cols[z])} 条，应当是 {max_run[z] + 1} 条"
            )
        ordered = sorted((l, c) for l, c, _ in cols[z])
        for i, (la, ca) in enumerate(ordered):
            for lb, cb in ordered[i + 1:]:
                if lb >= la and (cb >> (lb - la)) == ca:
                    raise SystemExit(f"run_before[zerosLeft={z}]: 前缀冲突")
    return cols


# ── 表 9-4：coded_block_pattern（`me(v)` 的 codeNum → 图案） ──────────────────

def parse_cbp(pg):
    """表 9-4(a)：`ChromaArrayType == 1`（4:2:0）。48 个 codeNum，两列（帧内/帧间）。

    ⚠️ 这张表**跨两页**，而标题只在第一页 —— 第二页直接是 `(a) …` 的续表，
    所以第二页只能按 `(b) ChromaArrayType` 截断，不能再找标题。
    """
    body = pg[239].split("Table 9-4", 1)[1].split("(b) ChromaArrayType", 1)[0]
    body += pg[240].split("(b) ChromaArrayType", 1)[0]
    intra, inter = {}, {}
    for f in data_rows(body):
        if not f[0].isdigit() or len(f) < 3:
            continue
        code = int(f[0])
        if code > 47:
            continue
        intra[code] = int(f[1])
        inter[code] = int(f[2])
    for name, tab in (("intra", intra), ("inter", inter)):
        if sorted(tab) != list(range(48)):
            raise SystemExit(f"表 9-4 {name}：codeNum 应当是 0..47，抄到 {len(tab)} 个")
        if len(set(tab.values())) != 48:
            raise SystemExit(f"表 9-4 {name}：图案有重复 ⇒ 抄错")
    return intra, inter


# ── 生成 Rust ───────────────────────────────────────────────────────────────

def emit_lut(name, entries, extra, doc):
    es = sorted(entries, key=lambda e: (e[0], e[1]))
    types = ", ".join("u8" for _ in extra)
    lines = [
        f"/// {doc}",
        "///",
        f"/// 每项 `(位长, 码字, {', '.join(extra)})`，按 `(位长, 码字)` 升序 ——",
        "/// 解码端按位长逐级二分查找（见 [`super::cavlc`]）。",
        f"pub static {name}: &[(u8, u16, {types})] = &[",
    ]
    for e in es:
        rest = ", ".join(str(v) for v in e[2:])
        lines.append(f"    ({e[0]}, 0b{e[1]:0{e[0]}b}, {rest}),")
    lines.append("];")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
    pg = pages(text)

    ct = parse_coeff_token(pg)
    tz7 = parse_total_zeros(pg[248], range(1, 8), 16, "Table 9-7", "Table 9-8")
    tz8 = parse_total_zeros(pg[248], range(8, 16), 16, "Table 9-8")
    tz_dc = parse_tz_chroma_dc(pg[249], range(1, 4), 4)
    rb = parse_run_before(pg[250])
    cbp_intra, cbp_inter = parse_cbp(pg)

    out = [
        "//! **从规范抄出来的 CAVLC 变长码表**（自动生成，别手改）。",
        "//!",
        "//! 生成方式与自检见 `tools/gen_h264_cavlc_tables.py` 抬头。自检是三条：",
        "//! **同列前缀无关**、**Kraft 和 ≤ 1**、**每张表的符号个数**（62/62/62/62/14/30、",
        "//! `16 - T + 1`、`z + 1` 这些）。VLC 表**没法**靠单测钉（抄错一位是静默的），",
        "//! 所以这几条是 VLC 抄写这一层的机器判据。",
        "//!",
        "//! ⚠️ 规范自己的表**不是完全码**（`8 <= nC` 那一列全是 6 位码、64 个槽只用 62 个；",
        "//! `0 <= nC < 2` 差 2⁻¹⁵ …），所以 Kraft 判据取上界 1 而不是等号 —— 这一点",
        "//! ITU-T H.264 (2019) 与 (2024) 两版逐字符一致，不是抄错。",
        "//!",
        "//! 规范出处：ITU-T H.264 (V15) (08/2024) 表 9-5 / 9-7 / 9-8 / 9-9(a) / 9-10。",
        "",
        "",
    ]

    docs = {
        "lt2": "表 9-5 第 1 列（`0 <= nC < 2`）",
        "2to4": "表 9-5 第 2 列（`2 <= nC < 4`）",
        "4to8": "表 9-5 第 3 列（`4 <= nC < 8`）",
        "ge8": "表 9-5 第 4 列（`8 <= nC`）",
        "cm1": "表 9-5 第 5 列（`nC == -1`，4:2:0 的色度 DC，只到 TotalCoeff = 4）",
        "cm2": "表 9-5 第 6 列（`nC == -2`，4:2:2 的色度 DC，只到 TotalCoeff = 8）",
    }
    names = {
        "lt2": "COEFF_TOKEN_NC_LT2",
        "2to4": "COEFF_TOKEN_NC_2_4",
        "4to8": "COEFF_TOKEN_NC_4_8",
        "ge8": "COEFF_TOKEN_NC_GE8",
        "cm1": "COEFF_TOKEN_NC_M1",
        "cm2": "COEFF_TOKEN_NC_M2",
    }
    for col in NC_COLUMNS:
        out.append(emit_lut(names[col], ct[col], ["TrailingOnes", "TotalCoeff"], docs[col]))
        out.append("")

    for T in range(1, 16):
        src = "表 9-7" if T <= 7 else "表 9-8"
        col = tz7[T] if T <= 7 else tz8[T]
        out.append(emit_lut(f"TOTAL_ZEROS_TZ{T}", col, ["total_zeros"], f"{src}：`tzVlcIndex = {T}`"))
        out.append("")

    for T in range(1, 4):
        out.append(
            emit_lut(
                f"CHROMA_DC_TOTAL_ZEROS_TZ{T}",
                tz_dc[T],
                ["total_zeros"],
                f"表 9-9(a)：色度 DC 2x2，`tzVlcIndex = {T}`",
            )
        )
        out.append("")

    for z in (1, 2, 3, 4, 5, 6, 7):
        label = ">6" if z == 7 else str(z)
        out.append(
            emit_lut(
                f"RUN_BEFORE_ZL{z}", rb[z], ["run_before"], f"表 9-10：`zerosLeft = {label}`"
            )
        )
        out.append("")

    for name, tab in (
        ("CODED_BLOCK_PATTERN_INTRA", cbp_intra),
        ("CODED_BLOCK_PATTERN_INTER", cbp_inter),
    ):
        out.append(
            "/// 表 9-4(a)：`coded_block_pattern`（`ChromaArrayType == 1`，4:2:0）——\n"
            "/// 下标就是 `me(v)` 解出来的 codeNum。"
        )
        out.append(f"pub static {name}: [u8; 48] = [")
        for i in range(48):
            out.append(f"    {tab[i]}, // codeNum = {i}")
        out.append("];")
        out.append("")

    print("\n".join(out))


if __name__ == "__main__":
    main()
