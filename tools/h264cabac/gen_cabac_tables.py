#!/usr/bin/env python3
"""生成 `crates/me-decode/src/video/h264/cabac_tables.rs`。

数据来源是**规范表本身**：ITU-T H.264 §9.3.1.1 表 9-12 ~ 9-33（上下文初始化）
与表 9-36（状态转移 + rangeTabLPS）。这两张表是纯常量，任何实现都只能照抄，
所以这里用 FFmpeg 的等价转写（cabac_context_init_I/PB、ff_h264_cabac_tables）
做交叉校验后落成 Rust 数组 —— 数值必须逐项等于规范，没有"实现自由度"。

用法:
    python3 tools/h264cabac/gen_cabac_tables.py <ffmpeg h264_cabac.c> <ffmpeg cabac.c> <out.rs>
"""
import re
import sys


def numbers(text):
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    text = re.sub(r'//[^\n]*', '', text)
    return [int(x) for x in re.findall(r'-?\d+', text)]


def array_body(src, name):
    m = re.search(name + r'(?:\[[^\]]*\])+\s*=\s*\{(.*?)\n\};', src, re.S)
    if not m:
        raise SystemExit('找不到 ' + name)
    return numbers(m.group(1))


def main():
    h264_cabac, cabac_c, out = sys.argv[1], sys.argv[2], sys.argv[3]
    src = open(h264_cabac).read()
    init_i = array_body(src, 'cabac_context_init_I')
    init_pb = array_body(src, 'cabac_context_init_PB')
    assert len(init_i) == 2048, len(init_i)
    assert len(init_pb) == 3 * 2048, len(init_pb)

    packed = array_body(open(cabac_c).read(), r'ff_h264_cabac_tables\)')
    assert len(packed) >= 1343, len(packed)
    packed = [v & 0xFF for v in packed]
    lps = packed[512:1024]
    mlps = packed[1024:1280]
    last_off_8x8 = packed[1280:1343]

    # rangeTabLPS[pStateIdx][(range>>6)&3]
    range_lps = [[lps[128 * k + 2 * st] for k in range(4)] for st in range(64)]
    # 同状态下 MPS=0/1 的 pLPS 必须一致（表里就是同一个值抄了两遍）
    for st in range(64):
        for k in range(4):
            assert lps[128 * k + 2 * st] == lps[128 * k + 2 * st + 1], (st, k)

    next_mps, next_lps, switch = [], [], []
    for st in range(64):
        nb = mlps[128 + 2 * st]
        next_mps.append(nb >> 1)
        assert (nb & 1) == 0, ('MPS 转移不该翻转 MPS', st)
        nb = mlps[127 - 2 * st]
        next_lps.append(nb >> 1)
        switch.append('true' if (nb & 1) else 'false')

    # §9.3.2.5 表 9-44（8×8 的 significant_coeff_flag ctxIdxInc）与
    # 8×8 的 last_significant_coeff_flag ctxIdxInc（同一张表的后半段）
    sig_off_8x8 = [
        0, 1, 2, 3, 4, 5, 5, 4, 4, 3, 3, 4, 4, 4, 5, 5,
        4, 4, 4, 4, 3, 3, 6, 7, 7, 7, 8, 9, 10, 9, 8, 7,
        7, 6, 11, 12, 13, 11, 6, 7, 8, 9, 14, 10, 9, 8, 6, 11,
        12, 13, 11, 6, 9, 14, 10, 9, 11, 12, 13, 11, 14, 10, 12,
    ]
    assert len(sig_off_8x8) == 63

    def table(name, vals, per_line=12, ty='u8'):
        lines = [f'pub const {name}: [{ty}; {len(vals)}] = [']
        for i in range(0, len(vals), per_line):
            lines.append('    ' + ', '.join(str(v) for v in vals[i:i + per_line]) + ',')
        lines.append('];')
        return '\n'.join(lines)

    def pairs(name, vals, ty='i8'):
        rows = [f'    [{vals[2 * i]:4}, {vals[2 * i + 1]:4}]' for i in range(len(vals) // 2)]
        return f'pub const {name}: [[{ty}; 2]; {len(vals) // 2}] = [\n' + ',\n'.join(rows) + '\n];'

    out_src = f'''//! CABAC 的**常量表**（§9.3.1.1 表 9-12 ~ 9-33、§9.3.1.2 表 9-36、§9.3.2.5 表 9-44）。
//!
//! 这个文件是**生成的**，别手改 —— 见 `tools/h264cabac/gen_cabac_tables.py`。
//!
//! 这些表没有任何"实现自由度"：上下文初值 `(m, n)`、状态机的 `rangeTabLPS` /
//! 转移目标、8×8 的 `ctxIdxInc` 映射，规范都写死了。生成脚本对三件事做了断言：
//!
//! 1. `cabac_context_init_I` 恰好 1024 项、`_PB` 恰好 3×1024 项；
//! 2. 同一 `pStateIdx` 下 MPS=0/1 的 `pLPS` 完全一致（表里就是同一个值抄两遍）；
//! 3. MPS 转移从不翻转 `valMPS`（翻转只发生在 LPS 且 `switch == 1` 时）。

/// §9.3.1.1 表 9-12：I 切片的上下文初值 `(m, n)`，按 `ctxIdx` 0..1023。
{pairs('INIT_I', init_i)}

/// §9.3.1.1 表 9-13 ~ 9-33：P/B 切片的上下文初值，按 `cabac_init_idc` 0..2。
pub const INIT_PB: [[[i8; 2]; 1024]; 3] = [
{chr(10).join("    [\n" + ",\n".join(f"        [{init_pb[k*2048+2*i]:4}, {init_pb[k*2048+2*i+1]:4}]" for i in range(1024)) + "\n    ]," for k in range(3))}
];

/// §9.3.1.2.1 表 9-36 的 `rangeTabLPS[pStateIdx][(codIRange >> 6) & 3]`。
pub const RANGE_TAB_LPS: [[u8; 4]; 64] = [
{chr(10).join("    [" + ", ".join(str(v) for v in range_lps[st]) + "]," for st in range(64))}
];

/// 表 9-36 的 `transIdxMPS`：MPS 之后的下一个状态（MPS 不翻转）。
{table('NEXT_MPS', next_mps)}

/// 表 9-36 的 `transIdxLPS`。
{table('NEXT_LPS', next_lps)}

/// 表 9-36 的 `switch`：LPS 之后是否翻转 `valMPS`（只有 `pStateIdx == 0` 为真）。
pub const SWITCH_LPS: [bool; 64] = [
{chr(10).join("    " + ", ".join(switch[i:i+16]) + "," for i in range(0, 64, 16))}
];

/// 表 9-44：8×8 块 `significant_coeff_flag` 的 `ctxIdxInc`（按扫描位置 0..62）。
{table('SIG_OFF_8X8', sig_off_8x8, 16)}

/// 8×8 块 `last_significant_coeff_flag` 的 `ctxIdxInc`（表 9-44 的后半段，按扫描位置 0..62）。
{table('LAST_OFF_8X8', last_off_8x8, 16)}
'''
    open(out, 'w').write(out_src)
    print('写入', out, len(out_src), '字节')
    print('NEXT_MPS', next_mps)
    print('NEXT_LPS', next_lps)
    print('SWITCH ', switch)


if __name__ == '__main__':
    main()
