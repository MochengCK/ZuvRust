#!/usr/bin/env python3
"""算出每个参照流「瓦片应该被消耗多少算术位」—— 这是 `exit_symbol()` 给出的**绝对判据**。

原理（规范 §8.2.6 `exit_symbol`）。设瓦片 `sz` 字节（`total = 8*sz` 位）：

    N0  = Min(total, 15)            init_symbol 先读进缓冲的位数
    M0  = total - 15                初始 SymbolMaxBits
    smb = M0 - consumed
    padding_end = Min(total,15) + R + Max(0, smb)      （R = 真正从瓦片读出的位数）
    trailing    = (Min(total,15) + R) - Min(15, smb + 15)

规范要求 `trailing` 位是 1、它到 `padding_end` 全是 0，且 `padding_end` 是 8 的整数倍。
对合法码流，`trailing` 必然落在瓦片里**最后一个 1** 上。把两种 smb 情形代进去化简，
两种情形都得到同一个结论：

    消耗位 = (瓦片里最后一个 1 的位置)

这就是"正确答案的位消耗"。拿它与自研解码器的 `consumed_bits()` 对比，
差额就说明差在哪一级；哪个流是 0 差、哪个流开始不为 0，也能直接二分。

**Rust 侧只有一份等价实现**：`me_decode::video::entropy::required_consumed_bits()`
（带单测；`me-cli` 的 `bundled_probe_streams_pass_exit_symbol_exactly` 就是拿它当判据）。
本脚本是它的**独立 Python 复算**，用来交叉验证，别在这里另立一套口径。

注意本脚本只适用于**单瓦片**的参照流：它按固定的 `header_bytes`（默认 9）从
`OBU_FRAME` 载荷上切一刀当瓦片。真实多瓦片素材请用 Rust 那个函数（传 `t.data`）。

用法：
    python3 tools/av1probe/required_bits.py <样本目录 | 文件…> [header_bytes=9]
"""

import sys
from pathlib import Path

OBU_TILE_GROUP = 4
OBU_FRAME = 6


def obus(data):
    out, i = [], 0
    while i < len(data):
        hdr = data[i]
        obu_type = (hdr >> 3) & 0xF
        has_size = (hdr >> 1) & 1
        i += 1
        if has_size:
            size, shift = 0, 0
            while True:
                b = data[i]
                i += 1
                size |= (b & 0x7F) << shift
                shift += 7
                if not (b & 0x80):
                    break
        else:
            size = len(data) - i
        out.append((obu_type, data[i : i + size]))
        i += size
    return out


def tile_of(data, header_bytes):
    frame = None
    for t, p in obus(data):
        if t in (OBU_TILE_GROUP, OBU_FRAME):
            frame = p
    if frame is None or len(frame) <= header_bytes:
        return None
    return frame[header_bytes:]


def main():
    args = [a for a in sys.argv[1:]]
    header_bytes = 9
    if args and args[-1].isdigit():
        header_bytes = int(args.pop())
    if not args:
        print(__doc__)
        return 1
    paths = []
    for a in args:
        p = Path(a)
        paths.extend(sorted(p.glob("put_L*.bin")) if p.is_dir() else [p])

    print(f"{'流':<34}{'瓦片B':>6}{'末位1':>7}{'应消耗':>8}{'R':>5}{'smb':>6}   备注")
    for path in paths:
        tile = tile_of(path.read_bytes(), header_bytes)
        if tile is None:
            print(f"{path.name:<34}     解析不出")
            continue
        sz = len(tile)
        total = 8 * sz
        bits = "".join(f"{b:08b}" for b in tile)
        p = bits.rfind("1")
        if p < 0:
            print(f"{path.name:<34}{sz:>6}   ---   瓦片全是 0，不合法")
            continue
        consumed = p
        smb = total - 15 - consumed
        r_req = min(consumed, max(0, total - 15))
        note = ""
        if smb < 0:
            note = f"位已读尽，借虚拟 0 共 {-smb} 位"
        print(f"{path.name:<34}{sz:>6}{p:>7}{consumed:>8}{r_req:>5}{smb:>6}   {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
