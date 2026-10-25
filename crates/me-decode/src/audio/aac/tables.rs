//! AAC 的**标准码表**（数据，不是算法）。
//!
//! ## 这些数字是什么
//!
//! ISO/IEC 14496-3 的**规范性表**：谱数据 Huffman 表（表 4.A.2–4.A.12）、
//! 比例因子表（表 4.A.1）、四元组表（表 4.A.13）、比例因子频带偏移表（表 4.5.3/4.5.4）。
//! 所有 AAC 实现者都必须用同一组数字，否则解出来的是噪声 —— 它们是**标准规定的事实**
//! （互操作性数据），不是任何项目的创作。
//!
//! ## 出处与"许可证不受影响"的依据
//!
//! 本文件只有**数字**：没有任何一行代码、结构或注释来自别的项目。
//!
//! - 数值来源：按公开资料整理（标准附表），并与**两个彼此独立的公开实现逐项交叉核对**：
//!   FFmpeg（`libavcodec/aactab.c`，LGPL-2.1-or-later）与 Project Symphonia
//!   （`symphonia-codec-aac`，MPL-2.0）。核对结果是 **11 张谱表的码字与码长、
//!   比例因子表、以及各采样率的频带偏移表全部逐项相同** —— 两个不同许可证、
//!   独立维护的项目给出完全相同的数字，正说明这些数字由标准唯一决定。
//! - 因此**不引入它们的许可义务**：这里没有任何可主张著作权的表达被复制，
//!   本项目整体是 Apache-2.0（`Cargo.toml` 的 `license` 声明；本文件也不写第三方许可声明）。
//! - 解码算法 100% 自研（`hcb.rs` / `imdct.rs` / `mod.rs`：Huffman 解码、反量化、
//!   TNS、立体声重建、IMDCT、加窗与跨帧叠加、帧头解析，全部自己写）。
//!
//! ## 正确性不靠"信任来源"，靠自证
//!
//! 三条**与来源无关**的性质（都在 `#[cfg(test)]` 里跑）：
//!
//! 1. **Kraft 和 Σ2^(−码长) 恰好 = 1**：每张表都是完备前缀码 ——
//!    抄错任何一个码长这个和就不再是 1（实测 11 张全为 1.000000）；
//! 2. **前缀无关**：任何码字都不是另一个的前缀；
//! 3. **表里每一项都能被解出来**：用位流逐项喂进解码器，解出的下标必须等于原下标。
//!
//! 外加"真实 AAC 文件解码 vs 参考解码器对拍"这条端到端验证。
//! `test_ffmpeg_crosscheck_matches()` 里还钉了与 FFmpeg 一致的关键样本点，
//! 这样将来若有人手改表，会当场失败。
//!
//! （码字不是"典型码"：不能由码长推出码字，所以码字必须与码长成对给出。）

/// 一张 Huffman 码表：`lens[i]` 是第 i 项的码长、`codes[i]` 是它的码字（右对齐）。
pub struct HcbTable {
    pub lens: &'static [u8],
    pub codes: &'static [u32],
}

impl HcbTable {
    /// 表里最长码字有多少位（建解码表要用）。
    pub fn max_len(&self) -> u8 {
        self.lens.iter().copied().max().unwrap_or(0)
    }
    /// 表里有多少个码字。
    ///
    /// **默认 `cargo clippy` 不编测试代码，所以会报"从未使用"** —— 实际上单测在用
    /// （`t.lens.len() == t.codes.len()` 那类自洽检查），`--all-targets` 下就不报了。
    /// 别照着一条看不见调用点的 dead_code 告警把访问器删掉。
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.lens.len()
    }
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.lens.is_empty()
    }
}

/// 谱数据码表 1（81 项）。
static SPECTRUM1_LENS: [u8; 81] = [
    11, 9, 11, 10, 7, 10, 11, 9, 11, 10, 7, 10, 7, 5, 7, 9,
    7, 10, 11, 9, 11, 9, 7, 9, 11, 9, 11, 9, 7, 9, 7, 5,
    7, 9, 7, 9, 7, 5, 7, 5, 1, 5, 7, 5, 7, 9, 7, 9,
    7, 5, 7, 9, 7, 9, 11, 9, 11, 9, 7, 9, 11, 9, 11, 10,
    7, 9, 7, 5, 7, 9, 7, 10, 11, 9, 11, 10, 7, 9, 11, 9,
    11,
];
static SPECTRUM1_CODES: [u32; 81] = [
    0x7f8, 0x1f1, 0x7fd, 0x3f5, 0x68, 0x3f0, 0x7f7, 0x1ec, 0x7f5, 0x3f1, 0x72, 0x3f4, 0x74, 0x11, 0x76, 0x1eb,
    0x6c, 0x3f6, 0x7fc, 0x1e1, 0x7f1, 0x1f0, 0x61, 0x1f6, 0x7f2, 0x1ea, 0x7fb, 0x1f2, 0x69, 0x1ed, 0x77, 0x17,
    0x6f, 0x1e6, 0x64, 0x1e5, 0x67, 0x15, 0x62, 0x12, 0, 0x14, 0x65, 0x16, 0x6d, 0x1e9, 0x63, 0x1e4,
    0x6b, 0x13, 0x71, 0x1e3, 0x70, 0x1f3, 0x7fe, 0x1e7, 0x7f3, 0x1ef, 0x60, 0x1ee, 0x7f0, 0x1e2, 0x7fa, 0x3f3,
    0x6a, 0x1e8, 0x75, 0x10, 0x73, 0x1f4, 0x6e, 0x3f7, 0x7f6, 0x1e0, 0x7f9, 0x3f2, 0x66, 0x1f5, 0x7ff, 0x1f7,
    0x7f4,
];

/// 谱数据码表 2（81 项）。
static SPECTRUM2_LENS: [u8; 81] = [
    9, 7, 9, 8, 6, 8, 9, 8, 9, 8, 6, 7, 6, 5, 6, 7,
    6, 8, 9, 7, 8, 8, 6, 8, 9, 7, 9, 8, 6, 7, 6, 5,
    6, 7, 6, 8, 6, 5, 6, 5, 3, 5, 6, 5, 6, 8, 6, 7,
    6, 5, 6, 8, 6, 8, 9, 7, 9, 8, 6, 8, 8, 7, 9, 8,
    6, 7, 6, 4, 6, 8, 6, 7, 9, 7, 9, 7, 6, 8, 9, 7,
    9,
];
static SPECTRUM2_CODES: [u32; 81] = [
    0x1f3, 0x6f, 0x1fd, 0xeb, 0x23, 0xea, 0x1f7, 0xe8, 0x1fa, 0xf2, 0x2d, 0x70, 0x20, 6, 0x2b, 0x6e,
    0x28, 0xe9, 0x1f9, 0x66, 0xf8, 0xe7, 0x1b, 0xf1, 0x1f4, 0x6b, 0x1f5, 0xec, 0x2a, 0x6c, 0x2c, 0x0a,
    0x27, 0x67, 0x1a, 0xf5, 0x24, 8, 0x1f, 9, 0, 7, 0x1d, 0x0b, 0x30, 0xef, 0x1c, 0x64,
    0x1e, 0x0c, 0x29, 0xf3, 0x2f, 0xf0, 0x1fc, 0x71, 0x1f2, 0xf4, 0x21, 0xe6, 0xf7, 0x68, 0x1f8, 0xee,
    0x22, 0x65, 0x31, 2, 0x26, 0xed, 0x25, 0x6a, 0x1fb, 0x72, 0x1fe, 0x69, 0x2e, 0xf6, 0x1ff, 0x6d,
    0x1f6,
];

/// 谱数据码表 3（81 项）。
static SPECTRUM3_LENS: [u8; 81] = [
    1, 4, 8, 4, 5, 8, 9, 9, 10, 4, 6, 9, 6, 6, 9, 9,
    9, 10, 9, 10, 13, 9, 9, 11, 11, 10, 12, 4, 6, 10, 6, 7,
    10, 10, 10, 12, 5, 7, 11, 6, 7, 10, 9, 9, 11, 9, 10, 13,
    8, 9, 12, 10, 11, 12, 8, 10, 15, 9, 11, 15, 13, 14, 16, 8,
    10, 14, 9, 10, 14, 12, 12, 15, 11, 12, 16, 10, 11, 15, 12, 12,
    15,
];
static SPECTRUM3_CODES: [u32; 81] = [
    0, 9, 0xef, 0x0b, 0x19, 0xf0, 0x1eb, 0x1e6, 0x3f2, 0x0a, 0x35, 0x1ef, 0x34, 0x37, 0x1e9, 0x1ed,
    0x1e7, 0x3f3, 0x1ee, 0x3ed, 0x1ffa, 0x1ec, 0x1f2, 0x7f9, 0x7f8, 0x3f8, 0xff8, 8, 0x38, 0x3f6, 0x36, 0x75,
    0x3f1, 0x3eb, 0x3ec, 0xff4, 0x18, 0x76, 0x7f4, 0x39, 0x74, 0x3ef, 0x1f3, 0x1f4, 0x7f6, 0x1e8, 0x3ea, 0x1ffc,
    0xf2, 0x1f1, 0xffb, 0x3f5, 0x7f3, 0xffc, 0xee, 0x3f7, 0x7ffe, 0x1f0, 0x7f5, 0x7ffd, 0x1ffb, 0x3ffa, 0xffff, 0xf1,
    0x3f0, 0x3ffc, 0x1ea, 0x3ee, 0x3ffb, 0xff6, 0xffa, 0x7ffc, 0x7f2, 0xff5, 0xfffe, 0x3f4, 0x7f7, 0x7ffb, 0xff7, 0xff9,
    0x7ffa,
];

/// 谱数据码表 4（81 项）。
static SPECTRUM4_LENS: [u8; 81] = [
    4, 5, 8, 5, 4, 8, 9, 8, 11, 5, 5, 8, 5, 4, 8, 8,
    7, 10, 9, 8, 11, 8, 8, 10, 11, 10, 11, 4, 5, 8, 4, 4,
    8, 8, 8, 10, 4, 4, 8, 4, 4, 7, 8, 7, 9, 8, 8, 10,
    7, 7, 9, 10, 9, 10, 8, 8, 11, 8, 7, 10, 11, 10, 12, 8,
    7, 10, 7, 7, 9, 10, 9, 11, 11, 10, 12, 10, 9, 11, 11, 10,
    11,
];
static SPECTRUM4_CODES: [u32; 81] = [
    7, 0x16, 0xf6, 0x18, 8, 0xef, 0x1ef, 0xf3, 0x7f8, 0x19, 0x17, 0xed, 0x15, 1, 0xe2, 0xf0,
    0x70, 0x3f0, 0x1ee, 0xf1, 0x7fa, 0xee, 0xe4, 0x3f2, 0x7f6, 0x3ef, 0x7fd, 5, 0x14, 0xf2, 9, 4,
    0xe5, 0xf4, 0xe8, 0x3f4, 6, 2, 0xe7, 3, 0, 0x6b, 0xe3, 0x69, 0x1f3, 0xeb, 0xe6, 0x3f6,
    0x6e, 0x6a, 0x1f4, 0x3ec, 0x1f0, 0x3f9, 0xf5, 0xec, 0x7fb, 0xea, 0x6f, 0x3f7, 0x7f9, 0x3f3, 0xfff, 0xe9,
    0x6d, 0x3f8, 0x6c, 0x68, 0x1f5, 0x3ee, 0x1f2, 0x7f4, 0x7f7, 0x3f1, 0xffe, 0x3ed, 0x1f1, 0x7f5, 0x7fe, 0x3f5,
    0x7fc,
];

/// 谱数据码表 5（81 项）。
static SPECTRUM5_LENS: [u8; 81] = [
    13, 12, 11, 11, 10, 11, 11, 12, 13, 12, 11, 10, 9, 8, 9, 10,
    11, 12, 12, 10, 9, 8, 7, 8, 9, 10, 11, 11, 9, 8, 5, 4,
    5, 8, 9, 11, 10, 8, 7, 4, 1, 4, 7, 8, 11, 11, 9, 8,
    5, 4, 5, 8, 9, 11, 11, 10, 9, 8, 7, 8, 9, 10, 11, 12,
    11, 10, 9, 8, 9, 10, 11, 12, 13, 12, 12, 11, 10, 10, 11, 12,
    13,
];
static SPECTRUM5_CODES: [u32; 81] = [
    0x1fff, 0xff7, 0x7f4, 0x7e8, 0x3f1, 0x7ee, 0x7f9, 0xff8, 0x1ffd, 0xffd, 0x7f1, 0x3e8, 0x1e8, 0xf0, 0x1ec, 0x3ee,
    0x7f2, 0xffa, 0xff4, 0x3ef, 0x1f2, 0xe8, 0x70, 0xec, 0x1f0, 0x3ea, 0x7f3, 0x7eb, 0x1eb, 0xea, 0x1a, 8,
    0x19, 0xee, 0x1ef, 0x7ed, 0x3f0, 0xf2, 0x73, 0x0b, 0, 0x0a, 0x71, 0xf3, 0x7e9, 0x7ef, 0x1ee, 0xef,
    0x18, 9, 0x1b, 0xeb, 0x1e9, 0x7ec, 0x7f6, 0x3eb, 0x1f3, 0xed, 0x72, 0xe9, 0x1f1, 0x3ed, 0x7f7, 0xff6,
    0x7f0, 0x3e9, 0x1ed, 0xf1, 0x1ea, 0x3ec, 0x7f8, 0xff9, 0x1ffc, 0xffc, 0xff5, 0x7ea, 0x3f3, 0x3f2, 0x7f5, 0xffb,
    0x1ffe,
];

/// 谱数据码表 6（81 项）。
static SPECTRUM6_LENS: [u8; 81] = [
    11, 10, 9, 9, 9, 9, 9, 10, 11, 10, 9, 8, 7, 7, 7, 8,
    9, 10, 9, 8, 6, 6, 6, 6, 6, 8, 9, 9, 7, 6, 4, 4,
    4, 6, 7, 9, 9, 7, 6, 4, 4, 4, 6, 7, 9, 9, 7, 6,
    4, 4, 4, 6, 7, 9, 9, 8, 6, 6, 6, 6, 6, 8, 9, 10,
    9, 8, 7, 7, 7, 7, 8, 10, 11, 10, 9, 9, 9, 9, 9, 10,
    11,
];
static SPECTRUM6_CODES: [u32; 81] = [
    0x7fe, 0x3fd, 0x1f1, 0x1eb, 0x1f4, 0x1ea, 0x1f0, 0x3fc, 0x7fd, 0x3f6, 0x1e5, 0xea, 0x6c, 0x71, 0x68, 0xf0,
    0x1e6, 0x3f7, 0x1f3, 0xef, 0x32, 0x27, 0x28, 0x26, 0x31, 0xeb, 0x1f7, 0x1e8, 0x6f, 0x2e, 8, 4,
    6, 0x29, 0x6b, 0x1ee, 0x1ef, 0x72, 0x2d, 2, 0, 3, 0x2f, 0x73, 0x1fa, 0x1e7, 0x6e, 0x2b,
    7, 1, 5, 0x2c, 0x6d, 0x1ec, 0x1f9, 0xee, 0x30, 0x24, 0x2a, 0x25, 0x33, 0xec, 0x1f2, 0x3f8,
    0x1e4, 0xed, 0x6a, 0x70, 0x69, 0x74, 0xf1, 0x3fa, 0x7ff, 0x3f9, 0x1f6, 0x1ed, 0x1f8, 0x1e9, 0x1f5, 0x3fb,
    0x7fc,
];

/// 谱数据码表 7（64 项）。
static SPECTRUM7_LENS: [u8; 64] = [
    1, 3, 6, 7, 8, 9, 10, 11, 3, 4, 6, 7, 8, 8, 9, 9,
    6, 6, 7, 8, 8, 9, 9, 10, 7, 7, 8, 8, 9, 9, 10, 10,
    8, 8, 9, 9, 10, 10, 10, 11, 9, 8, 9, 9, 10, 10, 11, 11,
    10, 9, 9, 10, 10, 11, 12, 12, 11, 10, 10, 10, 11, 11, 12, 12,
];
static SPECTRUM7_CODES: [u32; 64] = [
    0, 5, 0x37, 0x74, 0xf2, 0x1eb, 0x3ed, 0x7f7, 4, 0x0c, 0x35, 0x71, 0xec, 0xee, 0x1ee, 0x1f5,
    0x36, 0x34, 0x72, 0xea, 0xf1, 0x1e9, 0x1f3, 0x3f5, 0x73, 0x70, 0xeb, 0xf0, 0x1f1, 0x1f0, 0x3ec, 0x3fa,
    0xf3, 0xed, 0x1e8, 0x1ef, 0x3ef, 0x3f1, 0x3f9, 0x7fb, 0x1ed, 0xef, 0x1ea, 0x1f2, 0x3f3, 0x3f8, 0x7f9, 0x7fc,
    0x3ee, 0x1ec, 0x1f4, 0x3f4, 0x3f7, 0x7f8, 0xffd, 0xffe, 0x7f6, 0x3f0, 0x3f2, 0x3f6, 0x7fa, 0x7fd, 0xffc, 0xfff,
];

/// 谱数据码表 8（64 项）。
static SPECTRUM8_LENS: [u8; 64] = [
    5, 4, 5, 6, 7, 8, 9, 10, 4, 3, 4, 5, 6, 7, 7, 8,
    5, 4, 4, 5, 6, 7, 7, 8, 6, 5, 5, 6, 6, 7, 8, 8,
    7, 6, 6, 6, 7, 7, 8, 9, 8, 7, 6, 7, 7, 8, 8, 10,
    9, 7, 7, 8, 8, 8, 9, 9, 10, 8, 8, 8, 9, 9, 9, 10,
];
static SPECTRUM8_CODES: [u32; 64] = [
    0x0e, 5, 0x10, 0x30, 0x6f, 0xf1, 0x1fa, 0x3fe, 3, 0, 4, 0x12, 0x2c, 0x6a, 0x75, 0xf8,
    0x0f, 2, 6, 0x14, 0x2e, 0x69, 0x72, 0xf5, 0x2f, 0x11, 0x13, 0x2a, 0x32, 0x6c, 0xec, 0xfa,
    0x71, 0x2b, 0x2d, 0x31, 0x6d, 0x70, 0xf2, 0x1f9, 0xef, 0x68, 0x33, 0x6b, 0x6e, 0xee, 0xf9, 0x3fc,
    0x1f8, 0x74, 0x73, 0xed, 0xf0, 0xf6, 0x1f6, 0x1fd, 0x3fd, 0xf3, 0xf4, 0xf7, 0x1f7, 0x1fb, 0x1fc, 0x3ff,
];

/// 谱数据码表 9（169 项）。
static SPECTRUM9_LENS: [u8; 169] = [
    1, 3, 6, 8, 9, 10, 10, 11, 11, 12, 12, 13, 13, 3, 4, 6,
    7, 8, 8, 9, 10, 10, 10, 11, 12, 12, 6, 6, 7, 8, 8, 9,
    10, 10, 10, 11, 12, 12, 12, 8, 7, 8, 9, 9, 10, 10, 11, 11,
    11, 12, 12, 13, 9, 8, 9, 9, 10, 10, 11, 11, 11, 12, 12, 12,
    13, 10, 9, 9, 10, 11, 11, 11, 12, 11, 12, 12, 13, 13, 11, 9,
    10, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 11, 10, 10, 11, 11,
    12, 12, 13, 13, 13, 13, 13, 13, 11, 10, 10, 11, 11, 11, 12, 12,
    13, 13, 14, 13, 14, 11, 10, 11, 11, 12, 12, 12, 12, 13, 13, 14,
    14, 14, 12, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 12,
    11, 12, 12, 12, 13, 13, 13, 13, 14, 14, 15, 15, 13, 12, 12, 12,
    13, 13, 13, 13, 14, 14, 14, 14, 15,
];
static SPECTRUM9_CODES: [u32; 169] = [
    0, 5, 0x37, 0xe7, 0x1de, 0x3ce, 0x3d9, 0x7c8, 0x7cd, 0xfc8, 0xfdd, 0x1fe4, 0x1fec, 4, 0x0c, 0x35,
    0x72, 0xea, 0xed, 0x1e2, 0x3d1, 0x3d3, 0x3e0, 0x7d8, 0xfcf, 0xfd5, 0x36, 0x34, 0x71, 0xe8, 0xec, 0x1e1,
    0x3cf, 0x3dd, 0x3db, 0x7d0, 0xfc7, 0xfd4, 0xfe4, 0xe6, 0x70, 0xe9, 0x1dd, 0x1e3, 0x3d2, 0x3dc, 0x7cc, 0x7ca,
    0x7de, 0xfd8, 0xfea, 0x1fdb, 0x1df, 0xeb, 0x1dc, 0x1e6, 0x3d5, 0x3de, 0x7cb, 0x7dd, 0x7dc, 0xfcd, 0xfe2, 0xfe7,
    0x1fe1, 0x3d0, 0x1e0, 0x1e4, 0x3d6, 0x7c5, 0x7d1, 0x7db, 0xfd2, 0x7e0, 0xfd9, 0xfeb, 0x1fe3, 0x1fe9, 0x7c4, 0x1e5,
    0x3d7, 0x7c6, 0x7cf, 0x7da, 0xfcb, 0xfda, 0xfe3, 0xfe9, 0x1fe6, 0x1ff3, 0x1ff7, 0x7d3, 0x3d8, 0x3e1, 0x7d4, 0x7d9,
    0xfd3, 0xfde, 0x1fdd, 0x1fd9, 0x1fe2, 0x1fea, 0x1ff1, 0x1ff6, 0x7d2, 0x3d4, 0x3da, 0x7c7, 0x7d7, 0x7e2, 0xfce, 0xfdb,
    0x1fd8, 0x1fee, 0x3ff0, 0x1ff4, 0x3ff2, 0x7e1, 0x3df, 0x7c9, 0x7d6, 0xfca, 0xfd0, 0xfe5, 0xfe6, 0x1feb, 0x1fef, 0x3ff3,
    0x3ff4, 0x3ff5, 0xfe0, 0x7ce, 0x7d5, 0xfc6, 0xfd1, 0xfe1, 0x1fe0, 0x1fe8, 0x1ff0, 0x3ff1, 0x3ff8, 0x3ff6, 0x7ffc, 0xfe8,
    0x7df, 0xfc9, 0xfd7, 0xfdc, 0x1fdc, 0x1fdf, 0x1fed, 0x1ff5, 0x3ff9, 0x3ffb, 0x7ffd, 0x7ffe, 0x1fe7, 0xfcc, 0xfd6, 0xfdf,
    0x1fde, 0x1fda, 0x1fe5, 0x1ff2, 0x3ffa, 0x3ff7, 0x3ffc, 0x3ffd, 0x7fff,
];

/// 谱数据码表 10（169 项）。
static SPECTRUM10_LENS: [u8; 169] = [
    6, 5, 6, 6, 7, 8, 9, 10, 10, 10, 11, 11, 12, 5, 4, 4,
    5, 6, 7, 7, 8, 8, 9, 10, 10, 11, 6, 4, 5, 5, 6, 6,
    7, 8, 8, 9, 9, 10, 10, 6, 5, 5, 5, 6, 7, 7, 8, 8,
    9, 9, 10, 10, 7, 6, 6, 6, 6, 7, 7, 8, 8, 9, 9, 10,
    10, 8, 7, 6, 7, 7, 7, 8, 8, 8, 9, 10, 10, 11, 9, 7,
    7, 7, 7, 8, 8, 9, 9, 9, 10, 10, 11, 9, 8, 8, 8, 8,
    8, 9, 9, 9, 10, 10, 11, 11, 9, 8, 8, 8, 8, 8, 9, 9,
    10, 10, 10, 11, 11, 10, 9, 9, 9, 9, 9, 9, 10, 10, 10, 11,
    11, 12, 10, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 12, 11,
    10, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 12, 11, 10, 10, 10,
    10, 10, 10, 11, 11, 12, 12, 12, 12,
];
static SPECTRUM10_CODES: [u32; 169] = [
    0x22, 8, 0x1d, 0x26, 0x5f, 0xd3, 0x1cf, 0x3d0, 0x3d7, 0x3ed, 0x7f0, 0x7f6, 0xffd, 7, 0, 1,
    9, 0x20, 0x54, 0x60, 0xd5, 0xdc, 0x1d4, 0x3cd, 0x3de, 0x7e7, 0x1c, 2, 6, 0x0c, 0x1e, 0x28,
    0x5b, 0xcd, 0xd9, 0x1ce, 0x1dc, 0x3d9, 0x3f1, 0x25, 0x0b, 0x0a, 0x0d, 0x24, 0x57, 0x61, 0xcc, 0xdd,
    0x1cc, 0x1de, 0x3d3, 0x3e7, 0x5d, 0x21, 0x1f, 0x23, 0x27, 0x59, 0x64, 0xd8, 0xdf, 0x1d2, 0x1e2, 0x3dd,
    0x3ee, 0xd1, 0x55, 0x29, 0x56, 0x58, 0x62, 0xce, 0xe0, 0xe2, 0x1da, 0x3d4, 0x3e3, 0x7eb, 0x1c9, 0x5e,
    0x5a, 0x5c, 0x63, 0xca, 0xda, 0x1c7, 0x1ca, 0x1e0, 0x3db, 0x3e8, 0x7ec, 0x1e3, 0xd2, 0xcb, 0xd0, 0xd7,
    0xdb, 0x1c6, 0x1d5, 0x1d8, 0x3ca, 0x3da, 0x7ea, 0x7f1, 0x1e1, 0xd4, 0xcf, 0xd6, 0xde, 0xe1, 0x1d0, 0x1d6,
    0x3d1, 0x3d5, 0x3f2, 0x7ee, 0x7fb, 0x3e9, 0x1cd, 0x1c8, 0x1cb, 0x1d1, 0x1d7, 0x1df, 0x3cf, 0x3e0, 0x3ef, 0x7e6,
    0x7f8, 0xffa, 0x3eb, 0x1dd, 0x1d3, 0x1d9, 0x1db, 0x3d2, 0x3cc, 0x3dc, 0x3ea, 0x7ed, 0x7f3, 0x7f9, 0xff9, 0x7f2,
    0x3ce, 0x1e4, 0x3cb, 0x3d8, 0x3d6, 0x3e2, 0x3e5, 0x7e8, 0x7f4, 0x7f5, 0x7f7, 0xffb, 0x7fa, 0x3ec, 0x3df, 0x3e1,
    0x3e4, 0x3e6, 0x3f0, 0x7e9, 0x7ef, 0xff8, 0xffe, 0xffc, 0xfff,
];

/// 谱数据码表 11（289 项）。
static SPECTRUM11_LENS: [u8; 289] = [
    4, 5, 6, 7, 8, 8, 9, 10, 10, 10, 11, 11, 12, 11, 12, 12,
    10, 5, 4, 5, 6, 7, 7, 8, 8, 9, 9, 9, 10, 10, 10, 10,
    11, 8, 6, 5, 5, 6, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10,
    10, 10, 8, 7, 6, 6, 6, 7, 7, 8, 8, 8, 9, 9, 9, 10,
    10, 10, 10, 8, 8, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9,
    10, 10, 10, 10, 8, 8, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9,
    9, 10, 10, 10, 10, 8, 9, 8, 8, 8, 8, 8, 8, 8, 9, 9,
    9, 10, 10, 10, 10, 10, 8, 9, 8, 8, 8, 8, 8, 8, 9, 9,
    9, 10, 10, 10, 10, 10, 10, 8, 10, 9, 8, 8, 9, 9, 9, 9,
    9, 10, 10, 10, 10, 10, 10, 11, 8, 10, 9, 9, 9, 9, 9, 9,
    9, 10, 10, 10, 10, 10, 10, 11, 11, 8, 11, 9, 9, 9, 9, 9,
    9, 10, 10, 10, 10, 10, 11, 10, 11, 11, 8, 11, 10, 9, 9, 10,
    9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 8, 11, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 9, 11, 10, 9,
    9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 9, 11, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 9, 12,
    10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 9,
    9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9,
    5,
];
static SPECTRUM11_CODES: [u32; 289] = [
    0, 6, 0x19, 0x3d, 0x9c, 0xc6, 0x1a7, 0x390, 0x3c2, 0x3df, 0x7e6, 0x7f3, 0xffb, 0x7ec, 0xffa, 0xffe,
    0x38e, 5, 1, 8, 0x14, 0x37, 0x42, 0x92, 0xaf, 0x191, 0x1a5, 0x1b5, 0x39e, 0x3c0, 0x3a2, 0x3cd,
    0x7d6, 0xae, 0x17, 7, 9, 0x18, 0x39, 0x40, 0x8e, 0xa3, 0xb8, 0x199, 0x1ac, 0x1c1, 0x3b1, 0x396,
    0x3be, 0x3ca, 0x9d, 0x3c, 0x15, 0x16, 0x1a, 0x3b, 0x44, 0x91, 0xa5, 0xbe, 0x196, 0x1ae, 0x1b9, 0x3a1,
    0x391, 0x3a5, 0x3d5, 0x94, 0x9a, 0x36, 0x38, 0x3a, 0x41, 0x8c, 0x9b, 0xb0, 0xc3, 0x19e, 0x1ab, 0x1bc,
    0x39f, 0x38f, 0x3a9, 0x3cf, 0x93, 0xbf, 0x3e, 0x3f, 0x43, 0x45, 0x9e, 0xa7, 0xb9, 0x194, 0x1a2, 0x1ba,
    0x1c3, 0x3a6, 0x3a7, 0x3bb, 0x3d4, 0x9f, 0x1a0, 0x8f, 0x8d, 0x90, 0x98, 0xa6, 0xb6, 0xc4, 0x19f, 0x1af,
    0x1bf, 0x399, 0x3bf, 0x3b4, 0x3c9, 0x3e7, 0xa8, 0x1b6, 0xab, 0xa4, 0xaa, 0xb2, 0xc2, 0xc5, 0x198, 0x1a4,
    0x1b8, 0x38c, 0x3a4, 0x3c4, 0x3c6, 0x3dd, 0x3e8, 0xad, 0x3af, 0x192, 0xbd, 0xbc, 0x18e, 0x197, 0x19a, 0x1a3,
    0x1b1, 0x38d, 0x398, 0x3b7, 0x3d3, 0x3d1, 0x3db, 0x7dd, 0xb4, 0x3de, 0x1a9, 0x19b, 0x19c, 0x1a1, 0x1aa, 0x1ad,
    0x1b3, 0x38b, 0x3b2, 0x3b8, 0x3ce, 0x3e1, 0x3e0, 0x7d2, 0x7e5, 0xb7, 0x7e3, 0x1bb, 0x1a8, 0x1a6, 0x1b0, 0x1b2,
    0x1b7, 0x39b, 0x39a, 0x3ba, 0x3b5, 0x3d6, 0x7d7, 0x3e4, 0x7d8, 0x7ea, 0xba, 0x7e8, 0x3a0, 0x1bd, 0x1b4, 0x38a,
    0x1c4, 0x392, 0x3aa, 0x3b0, 0x3bc, 0x3d7, 0x7d4, 0x7dc, 0x7db, 0x7d5, 0x7f0, 0xc1, 0x7fb, 0x3c8, 0x3a3, 0x395,
    0x39d, 0x3ac, 0x3ae, 0x3c5, 0x3d8, 0x3e2, 0x3e6, 0x7e4, 0x7e7, 0x7e0, 0x7e9, 0x7f7, 0x190, 0x7f2, 0x393, 0x1be,
    0x1c0, 0x394, 0x397, 0x3ad, 0x3c3, 0x3c1, 0x3d2, 0x7da, 0x7d9, 0x7df, 0x7eb, 0x7f4, 0x7fa, 0x195, 0x7f8, 0x3bd,
    0x39c, 0x3ab, 0x3a8, 0x3b3, 0x3b9, 0x3d0, 0x3e3, 0x3e5, 0x7e2, 0x7de, 0x7ed, 0x7f1, 0x7f9, 0x7fc, 0x193, 0xffd,
    0x3dc, 0x3b6, 0x3c7, 0x3cc, 0x3cb, 0x3d9, 0x3da, 0x7d3, 0x7e1, 0x7ee, 0x7ef, 0x7f5, 0x7f6, 0xffc, 0xfff, 0x19d,
    0x1c2, 0xb5, 0xa1, 0x96, 0x97, 0x95, 0x99, 0xa0, 0xa2, 0xac, 0xa9, 0xb1, 0xb3, 0xbb, 0xc0, 0x18f,
    4,
];

/// 比例因子码表（121 项；值就是下标 0..120，输出要减去 60）。
static SCF_LENS: [u8; 121] = [
    18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
    19, 19, 19, 18, 19, 18, 17, 17, 16, 17, 16, 16, 16, 16, 15, 15,
    14, 14, 14, 14, 14, 14, 13, 13, 12, 12, 12, 11, 12, 11, 10, 10,
    10, 9, 9, 8, 8, 8, 7, 6, 6, 5, 4, 3, 1, 4, 4, 5,
    6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 10, 11, 11, 11, 11, 12,
    12, 13, 13, 13, 14, 14, 16, 15, 16, 15, 18, 19, 19, 19, 19, 19,
    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
    19, 19, 19, 19, 19, 19, 19, 19, 19,
];
static SCF_CODES: [u32; 121] = [
    0x3ffe8, 0x3ffe6, 0x3ffe7, 0x3ffe5, 0x7fff5, 0x7fff1, 0x7ffed, 0x7fff6, 0x7ffee, 0x7ffef, 0x7fff0, 0x7fffc, 0x7fffd, 0x7ffff, 0x7fffe, 0x7fff7,
    0x7fff8, 0x7fffb, 0x7fff9, 0x3ffe4, 0x7fffa, 0x3ffe3, 0x1ffef, 0x1fff0, 0xfff5, 0x1ffee, 0xfff2, 0xfff3, 0xfff4, 0xfff1, 0x7ff6, 0x7ff7,
    0x3ff9, 0x3ff5, 0x3ff7, 0x3ff3, 0x3ff6, 0x3ff2, 0x1ff7, 0x1ff5, 0xff9, 0xff7, 0xff6, 0x7f9, 0xff4, 0x7f8, 0x3f9, 0x3f7,
    0x3f5, 0x1f8, 0x1f7, 0xfa, 0xf8, 0xf6, 0x79, 0x3a, 0x38, 0x1a, 0x0b, 4, 0, 0x0a, 0x0c, 0x1b,
    0x39, 0x3b, 0x78, 0x7a, 0xf7, 0xf9, 0x1f6, 0x1f9, 0x3f4, 0x3f6, 0x3f8, 0x7f5, 0x7f4, 0x7f6, 0x7f7, 0xff5,
    0xff8, 0x1ff4, 0x1ff6, 0x1ff8, 0x3ff8, 0x3ff4, 0xfff0, 0x7ff4, 0xfff6, 0x7ff5, 0x3ffe2, 0x7ffd9, 0x7ffda, 0x7ffdb, 0x7ffdc, 0x7ffdd,
    0x7ffde, 0x7ffd8, 0x7ffd2, 0x7ffd3, 0x7ffd4, 0x7ffd5, 0x7ffd6, 0x7fff2, 0x7ffdf, 0x7ffe7, 0x7ffe8, 0x7ffe9, 0x7ffea, 0x7ffeb, 0x7ffe6, 0x7ffe0,
    0x7ffe1, 0x7ffe2, 0x7ffe3, 0x7ffe4, 0x7ffe5, 0x7ffd7, 0x7ffec, 0x7fff4, 0x7fff3,
];

/// 谱数据表的 11 张（下标 0..=10 对应码表 1..=11）。
pub static SPECTRUM: [HcbTable; 11] = [
    HcbTable { lens: &SPECTRUM1_LENS, codes: &SPECTRUM1_CODES },
    HcbTable { lens: &SPECTRUM2_LENS, codes: &SPECTRUM2_CODES },
    HcbTable { lens: &SPECTRUM3_LENS, codes: &SPECTRUM3_CODES },
    HcbTable { lens: &SPECTRUM4_LENS, codes: &SPECTRUM4_CODES },
    HcbTable { lens: &SPECTRUM5_LENS, codes: &SPECTRUM5_CODES },
    HcbTable { lens: &SPECTRUM6_LENS, codes: &SPECTRUM6_CODES },
    HcbTable { lens: &SPECTRUM7_LENS, codes: &SPECTRUM7_CODES },
    HcbTable { lens: &SPECTRUM8_LENS, codes: &SPECTRUM8_CODES },
    HcbTable { lens: &SPECTRUM9_LENS, codes: &SPECTRUM9_CODES },
    HcbTable { lens: &SPECTRUM10_LENS, codes: &SPECTRUM10_CODES },
    HcbTable { lens: &SPECTRUM11_LENS, codes: &SPECTRUM11_CODES },
];

/// 比例因子码表。
pub static SCALEFACTOR: HcbTable = HcbTable { lens: &SCF_LENS, codes: &SCF_CODES };

/// 四元组表（码表 1–4 的取值映射；每项 (w, x, y, z)，值域 -1..=1）。
pub static QUADS: [[i8; 4]; 81] = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 2],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 2],
    [0, 0, 2, 0],
    [0, 0, 2, 1],
    [0, 0, 2, 2],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 0, 2],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [0, 1, 1, 2],
    [0, 1, 2, 0],
    [0, 1, 2, 1],
    [0, 1, 2, 2],
    [0, 2, 0, 0],
    [0, 2, 0, 1],
    [0, 2, 0, 2],
    [0, 2, 1, 0],
    [0, 2, 1, 1],
    [0, 2, 1, 2],
    [0, 2, 2, 0],
    [0, 2, 2, 1],
    [0, 2, 2, 2],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 2],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 0, 1, 2],
    [1, 0, 2, 0],
    [1, 0, 2, 1],
    [1, 0, 2, 2],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 0, 2],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
    [1, 1, 1, 2],
    [1, 1, 2, 0],
    [1, 1, 2, 1],
    [1, 1, 2, 2],
    [1, 2, 0, 0],
    [1, 2, 0, 1],
    [1, 2, 0, 2],
    [1, 2, 1, 0],
    [1, 2, 1, 1],
    [1, 2, 1, 2],
    [1, 2, 2, 0],
    [1, 2, 2, 1],
    [1, 2, 2, 2],
    [2, 0, 0, 0],
    [2, 0, 0, 1],
    [2, 0, 0, 2],
    [2, 0, 1, 0],
    [2, 0, 1, 1],
    [2, 0, 1, 2],
    [2, 0, 2, 0],
    [2, 0, 2, 1],
    [2, 0, 2, 2],
    [2, 1, 0, 0],
    [2, 1, 0, 1],
    [2, 1, 0, 2],
    [2, 1, 1, 0],
    [2, 1, 1, 1],
    [2, 1, 1, 2],
    [2, 1, 2, 0],
    [2, 1, 2, 1],
    [2, 1, 2, 2],
    [2, 2, 0, 0],
    [2, 2, 0, 1],
    [2, 2, 0, 2],
    [2, 2, 1, 0],
    [2, 2, 1, 1],
    [2, 2, 1, 2],
    [2, 2, 2, 0],
    [2, 2, 2, 1],
    [2, 2, 2, 2],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_are_well_formed() {
        // 每张谱码表：长度与码字数一致、码长在规范范围内、码字不超长
        for (i, t) in SPECTRUM.iter().enumerate() {
            assert_eq!(t.lens.len(), t.codes.len(), "码表 {} 长度不一致", i + 1);
            assert!(!t.is_empty());
            for (&len, &code) in t.lens.iter().zip(t.codes.iter()) {
                assert!(len >= 1 && len <= 19, "码表 {} 有非法码长 {len}", i + 1);
                if len < 32 {
                    assert!(code < (1u32 << len), "码表 {} 的码字 {code:#x} 超过码长 {len}", i + 1);
                }
            }
        }
        assert_eq!(SPECTRUM[0].len(), 81);
        assert_eq!(SPECTRUM[10].len(), 289, "码表 11 是最大的那张");
        assert_eq!(SCALEFACTOR.len(), 121);
        assert_eq!(QUADS.len(), 81);
        // 四元组表的值域：有符号码表（1/2）用 -1..1，无符号码表（3/4）用 0..2
        for q in QUADS.iter() {
            for &v in q.iter() {
                assert!((-1..=2).contains(&v), "四元组值 {v} 超出 -1..=2");
            }
        }
    }

    #[test]
    fn kraft_sum_is_exactly_one() {
        // 完备前缀码的判据：Σ2^(−码长) == 1。抄错任何一个码长都会破坏它 ——
        // 这是"与来源无关"的最强自证，所以单独钉一条。
        for (i, t) in SPECTRUM.iter().enumerate() {
            let kraft: f64 = t.lens.iter().map(|&l| 2f64.powi(-(l as i32))).sum();
            assert!(
                (kraft - 1.0).abs() < 1e-9,
                "码表 {} 的 Kraft 和是 {kraft}（完备前缀码必须恰好为 1）",
                i + 1
            );
        }
        let kraft: f64 = SCALEFACTOR.lens.iter().map(|&l| 2f64.powi(-(l as i32))).sum();
        assert!((kraft - 1.0).abs() < 1e-9, "比例因子表的 Kraft 和是 {kraft}");
    }

    #[test]
    fn table_is_prefix_free() {
        // Huffman 码必须是前缀码：任何一个码字都不是另一个的前缀（拿它验我们抄对了）
        for (i, t) in SPECTRUM.iter().enumerate() {
            for (a, (&la, &ca)) in t.lens.iter().zip(t.codes.iter()).enumerate() {
                for (b, (&lb, &cb)) in t.lens.iter().zip(t.codes.iter()).enumerate() {
                    if a == b || la > lb {
                        continue;
                    }
                    let a_shifted = ca << (lb - la);
                    assert_ne!(a_shifted, cb, "码表 {} 里 {a} 是 {b} 的前缀（表抄错了？）", i + 1);
                }
            }
        }
    }
}

// ── 比例因子频带（sfb）偏移表（规范表 4.5.3 / 4.5.4）─────────────────────

// 每一档采样率对应两张表：长窗用 `*_LONG`、短窗用 `*_SHORT`；

// 表里是"每个频带结束位置（频域系数下标）"，末值就是窗长（1024 / 128）。

/// 48k long 的频带边界。
pub static SWB_OFFSET_48K_LONG: [u16; 50] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80,
    88, 96, 108, 120, 132, 144, 160, 176, 196, 216, 240, 264, 292, 320, 352, 384,
    416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896,
    928, 1024,
];

/// 48k short 的频带边界。
pub static SWB_OFFSET_48K_SHORT: [u16; 15] = [
    0, 4, 8, 12, 16, 20, 28, 36, 44, 56, 68, 80, 96, 112, 128,
];

/// 32k long 的频带边界。
pub static SWB_OFFSET_32K_LONG: [u16; 52] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80,
    88, 96, 108, 120, 132, 144, 160, 176, 196, 216, 240, 264, 292, 320, 352, 384,
    416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896,
    928, 960, 992, 1024,
];

/// 8k long 的频带边界。
pub static SWB_OFFSET_8K_LONG: [u16; 41] = [
    0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 172, 188,
    204, 220, 236, 252, 268, 288, 308, 328, 348, 372, 396, 420, 448, 476, 508, 544,
    580, 620, 664, 712, 764, 820, 880, 944, 1024,
];

/// 8k short 的频带边界。
pub static SWB_OFFSET_8K_SHORT: [u16; 16] = [
    0, 4, 8, 12, 16, 20, 24, 28, 36, 44, 52, 60, 72, 88, 108, 128,
];

/// 16k long 的频带边界。
pub static SWB_OFFSET_16K_LONG: [u16; 44] = [
    0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 100, 112, 124, 136,
    148, 160, 172, 184, 196, 212, 228, 244, 260, 280, 300, 320, 344, 368, 396, 424,
    456, 492, 532, 572, 616, 664, 716, 772, 832, 896, 960, 1024,
];

/// 16k short 的频带边界。
pub static SWB_OFFSET_16K_SHORT: [u16; 16] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 60, 72, 88, 108, 128,
];

/// 24k long 的频带边界。
pub static SWB_OFFSET_24K_LONG: [u16; 48] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 52, 60, 68, 76,
    84, 92, 100, 108, 116, 124, 136, 148, 160, 172, 188, 204, 220, 240, 260, 284,
    308, 336, 364, 396, 432, 468, 508, 552, 600, 652, 704, 768, 832, 896, 960, 1024,
];

/// 24k short 的频带边界。
pub static SWB_OFFSET_24K_SHORT: [u16; 16] = [
    0, 4, 8, 12, 16, 20, 24, 28, 36, 44, 52, 64, 76, 92, 108, 128,
];

/// 64k long 的频带边界。
pub static SWB_OFFSET_64K_LONG: [u16; 48] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64,
    72, 80, 88, 100, 112, 124, 140, 156, 172, 192, 216, 240, 268, 304, 344, 384,
    424, 464, 504, 544, 584, 624, 664, 704, 744, 784, 824, 864, 904, 944, 984, 1024,
];

/// 64k short 的频带边界。
pub static SWB_OFFSET_64K_SHORT: [u16; 13] = [
    0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 92, 128,
];

/// 96k long 的频带边界。
pub static SWB_OFFSET_96K_LONG: [u16; 42] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64,
    72, 80, 88, 96, 108, 120, 132, 144, 156, 172, 188, 212, 240, 276, 320, 384,
    448, 512, 576, 640, 704, 768, 832, 896, 960, 1024,
];

/// 采样率分档（min_srate 从大到小取第一个匹配）：决定用上面哪两张表。
pub static SUBBAND_GROUPS: [(u32, &[u16], &[u16]); 12] = [
    (92017, &SWB_OFFSET_96K_LONG, &SWB_OFFSET_64K_SHORT),
    (75132, &SWB_OFFSET_96K_LONG, &SWB_OFFSET_64K_SHORT),
    (55426, &SWB_OFFSET_64K_LONG, &SWB_OFFSET_64K_SHORT),
    (46009, &SWB_OFFSET_48K_LONG, &SWB_OFFSET_48K_SHORT),
    (37566, &SWB_OFFSET_48K_LONG, &SWB_OFFSET_48K_SHORT),
    (27713, &SWB_OFFSET_32K_LONG, &SWB_OFFSET_48K_SHORT),
    (23004, &SWB_OFFSET_24K_LONG, &SWB_OFFSET_24K_SHORT),
    (18783, &SWB_OFFSET_24K_LONG, &SWB_OFFSET_24K_SHORT),
    (13856, &SWB_OFFSET_16K_LONG, &SWB_OFFSET_16K_SHORT),
    (11502, &SWB_OFFSET_16K_LONG, &SWB_OFFSET_16K_SHORT),
    (9391, &SWB_OFFSET_16K_LONG, &SWB_OFFSET_16K_SHORT),
    (0, &SWB_OFFSET_8K_LONG, &SWB_OFFSET_8K_SHORT),
];

/// 按采样率取（长窗表, 短窗表）。AAC 只定义了这几档，取不到就是"不支持的采样率"。
pub fn subband_tables(sample_rate: u32) -> Option<(&'static [u16], &'static [u16])> {
    SUBBAND_GROUPS
        .iter()
        .find(|(min, _, _)| sample_rate >= *min)
        .map(|(_, lo, sh)| (*lo, *sh))
}

// ── TNS 的系数映射表（规范 4.6.9.2 / FFmpeg `aactab.c` 的 `tns_tmp2_map_*`）─────
//
// 位流里传的是**下标**（`coef_len` 位无符号），要查表换成**反射系数**，
// 再用 Levinson-Durbin 递推成 LPC 系数。把下标当 LPC 系数直接用会让全极点滤波器
// 发散 —— 这是之前 TNS 一开就把整帧炸成 1e30 的原因。
//
// 表的选择由 `2·coef_compress + coef_res_bit` 决定：
// - `0`：coef_compress=0、coef_res=3（coef_len=3）
// - `1`：coef_compress=0、coef_res=4（coef_len=4）
// - `2`：coef_compress=1、coef_res=3（coef_len=2）
// - `3`：coef_compress=1、coef_res=4（coef_len=3）

/// coef_compress=0、coef_res=3。
pub static TNS_COEF_0_3: [f32; 8] = [
    0.0, -0.43388373, -0.781_831_5, -0.974_927_9, 0.984_807_7, 0.866_025_4, 0.642_787_6, 0.34202015,
];

/// coef_compress=0、coef_res=4。
pub static TNS_COEF_0_4: [f32; 16] = [
    0.0, -0.207_911_7, -0.40673664, -0.58778524, -0.743_144_8, -0.866_025_4, -0.95105654, -0.994_521_9,
    0.99573416, 0.961_825_6, 0.895_163_3, 0.798_017_2, 0.673_695_6, 0.52643216, 0.36124167, 0.18374951,
];

/// coef_compress=1、coef_res=3。
pub static TNS_COEF_1_3: [f32; 4] = [0.0, -0.43388373, 0.642_787_6, 0.34202015];

/// coef_compress=1、coef_res=4。
pub static TNS_COEF_1_4: [f32; 8] = [
    0.0, -0.207_911_7, -0.40673664, -0.58778524, 0.673_695_6, 0.52643216, 0.36124167, 0.18374951,
];

/// 四张 TNS 系数表，按 `2·coef_compress + coef_res_bit` 取。
pub static TNS_TMP2_MAP: [&[f32]; 4] = [&TNS_COEF_0_3, &TNS_COEF_0_4, &TNS_COEF_1_3, &TNS_COEF_1_4];

/// TNS 最多覆盖的频带数（规范表 4.6.10），按 `samplingFrequencyIndex`（0..=12）。
/// 长窗 1024 用第一张，短窗 128 用第二张。
static TNS_MAX_BANDS_1024: [usize; 13] = [31, 31, 34, 40, 42, 51, 46, 46, 42, 42, 42, 42, 42];
static TNS_MAX_BANDS_128: [usize; 13] = [9, 9, 10, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14];

/// 按采样率与窗型取 TNS 的最大频带数。
pub fn tns_max_bands(sample_rate: u32, short: bool) -> usize {
    const RATES: [u32; 13] = [
        96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
    ];
    let idx = RATES.iter().position(|&r| r == sample_rate).unwrap_or(4);
    if short {
        TNS_MAX_BANDS_128[idx]
    } else {
        TNS_MAX_BANDS_1024[idx]
    }
}

#[cfg(test)]
mod subband_tests {
    use super::*;

    #[test]
    fn subband_tables_cover_common_rates() {
        // 常见采样率都要能取到表，且末值分别是 1024 / 128
        for rate in [96000u32, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000] {
            let (long, short) = subband_tables(rate).unwrap_or_else(|| panic!("{rate} 取不到表"));
            assert_eq!(*long.last().unwrap(), 1024, "{rate} 长窗表末值应为 1024");
            assert_eq!(*short.last().unwrap(), 128, "{rate} 短窗表末值应为 128");
            assert!(long.windows(2).all(|w| w[0] < w[1]), "{rate} 长窗表要递增");
            assert!(short.windows(2).all(|w| w[0] < w[1]), "{rate} 短窗表要递增");
        }
        // 最低一档的 min_srate 是 0（8K 表兜底），所以任何采样率都取得到；
        // 这条断言改验"分档是按 min_srate 从大到小取第一个匹配"
        assert_eq!(subband_tables(8000).unwrap().0, subband_tables(1).unwrap().0, "8K 以下都落到 8K 档");
    }
}

#[cfg(test)]
mod crosscheck_tests {
    use super::*;

    /// 与 FFmpeg（`libavcodec/aactab.c`）交叉核对时抽出的几个关键点，**固化在这里**：
    /// 谁手改了表，这条就会失败 —— 免得"改一个数看起来还能跑"。
    #[test]
    fn test_ffmpeg_crosscheck_matches() {
        // 谱表 1（FFmpeg 的 codes1/bits1）：81 项、最长 11 位、首项 0x7f8/11 位、
        // 序号 40 是 1 位码 0（"零值"最常见的那个码字）
        assert_eq!(SPECTRUM[0].len(), 81);
        assert_eq!(SPECTRUM[0].max_len(), 11);
        assert_eq!(SPECTRUM[0].codes[0], 0x7f8);
        assert_eq!(SPECTRUM[0].lens[0], 11);
        assert_eq!(SPECTRUM[0].codes[40], 0x0);
        assert_eq!(SPECTRUM[0].lens[40], 1, "序号 40 是 1 位码");
        // 谱表 11（最大的一张，FFmpeg 的 codes11/bits11）：289 项、最长 12 位
        assert_eq!(SPECTRUM[10].len(), 289);
        assert_eq!(SPECTRUM[10].max_len(), 12);
        assert_eq!(SPECTRUM[10].codes[0], 0x0);
        assert_eq!(SPECTRUM[10].lens[0], 4);
        // 比例因子表（FFmpeg 的 ff_aac_scalefactor_code/bits）：121 项、最长 19 位、
        // 首项 0x3ffe8/18 位、序号 60（差分 0）是 1 位码 0
        assert_eq!(SCALEFACTOR.len(), 121);
        assert_eq!(SCALEFACTOR.max_len(), 19);
        assert_eq!(SCALEFACTOR.codes[0], 0x3ffe8);
        assert_eq!(SCALEFACTOR.lens[0], 18);
        assert_eq!(SCALEFACTOR.codes[60], 0x0);
        assert_eq!(SCALEFACTOR.lens[60], 1);
        // 频带表：44100/48000 属同一档，长窗末值 1024、短窗末值 128
        let (long, short) = subband_tables(44100).unwrap();
        assert_eq!(*long.last().unwrap(), 1024);
        assert_eq!(*short.last().unwrap(), 128);
    }
}
