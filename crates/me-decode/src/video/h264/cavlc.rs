//! **CAVLC：变换系数的变长解码**（ITU-T H.264 §9.2）。
//!
//! 这是 `entropy_coding_mode_flag == 0` 那条路（Baseline / 以及显式选了 CAVLC 的
//! 其它 profile）。CABAC 在 [`super::cabac`]。
//!
//! ## 入口与出口
//!
//! 入口是**切片数据**（`slice_data()` 的起点，位偏移由
//! [`crate::h264::SliceHeader::slice_data_bit_offset`] 给出），
//! 出口是"按扫描序排列的变换系数" —— 反扫描、反量化、反变换都在 [`super::transform`]。
//!
//! 上游（[`super::mb`]）负责算 `nC`：它要的是**左邻与上邻 4×4 块的非零系数个数**
//! （§9.2.1 第 5~7 步），那是宏块层的信息，不在这里。
//!
//! ## 四个子过程与它们各自的坑
//!
//! | 子过程 | 规范 | 坑 |
//! |---|---|---|
//! | `coeff_token` | §9.2.1 | 选哪张表由 `nC` 决定；`nC == -1`（4:2:0 色度 DC，只到 4）与 `nC == -2`（4:2:2 色度 DC，只到 8）**别按"−1 小 −2 大"想当然** |
//! | `level` | §9.2.2 | `levelCode` 的三处补偿（`prefix>=15 && suffixLength==0`、`prefix>=16`、第一个非 trailing 加 2）**一处都不能少**；`suffixLength` 的更新在**取出 levelVal 之后** |
//! | `run_before` | §9.2.3 | `zerosLeft == 0` 时**不读位**、直接给 0；`TotalCoeff == maxNumCoeff` 时 `total_zeros` 也不读 |
//! | 合并 | §9.2.4 | 从 `i = TotalCoeff-1` **倒着**放，`coeffNum` 从 −1 累加 —— 正着放整块系数会整体镜像 |
//!
//! ## 验收
//!
//! CAVLC 没有像 AV1 `exit_symbol()` 那样的绝对判据，但有一条**等价强度**的：
//! 把一张图的所有宏块解完之后，读位必须**恰好**停在 `rbsp_trailing_bits`
//! （最后一个字节是 `1` + 补零）上。多读一位少读一位都对不上 —— 见
//! `me-cli` 的 `h264_syntax` 那组用例。

use me_core::bits::BitReader;
use me_core::{MediaError, Result};

use super::cavlc_tables::*;

/// CAVLC 位读器：薄薄一层包装，只为了把"从 `slice_data()` 起点开始读"这件事说清楚。
///
/// 除了 §9.2 那四个残差子过程，切片数据里还有 `ue(v)` / `se(v)` / `u(n)` 这些
/// **普通位域语法**（`mb_type`、`mb_qp_delta`、`prev_intra4x4_pred_mode_flag`…）——
/// 它们用的是同一个位流，所以也挂在这里（CABAC 那一版会有另一套等价接口）。
pub struct Cavlc<'a> {
    r: BitReader<'a>,
    /// 原始缓冲（NAL 头 + 去竞争字节后的负载）——`more_rbsp_data()` 要靠它判断
    /// "剩下的到底是数据还是 `rbsp_trailing_bits`"。
    raw: &'a [u8],
}

impl<'a> Cavlc<'a> {
    /// `raw` = **NAL 头字节 + 去竞争字节后的负载**（与
    /// [`crate::h264::parse_slice_header`] 同一条流），`bit_offset` =
    /// [`crate::h264::SliceHeader::slice_data_bit_offset`]。
    pub fn new(raw: &'a [u8], bit_offset: usize) -> Result<Self> {
        let mut r = BitReader::new(raw);
        if bit_offset > r.remaining_bits() {
            return Err(MediaError::Bitstream(format!(
                "切片数据起点 {bit_offset} 超过了这条 NAL 的位数 {}",
                r.remaining_bits()
            )));
        }
        r.skip_bits(bit_offset)?;
        Ok(Self { r, raw })
    }

    /// 无符号指数哥伦布（§9.1 的 `ue(v)`）。
    pub fn ue(&mut self) -> Result<u32> {
        self.r.read_ue()
    }

    /// 有符号指数哥伦布（§9.1 的 `se(v)`）。
    pub fn se(&mut self) -> Result<i32> {
        self.r.read_se()
    }

    /// 定长无符号（`u(n)`）。
    pub fn u(&mut self, n: u8) -> Result<u32> {
        self.r.read_u32(n)
    }

    /// 当前是不是字节对齐（`I_PCM` 的 `pcm_alignment_zero_bit` 与 CABAC 的
    /// `cabac_alignment_one_bit` 都要先问这个）。
    pub fn aligned(&self) -> bool {
        self.r.position_bits().is_multiple_of(8)
    }

    /// 读到字节对齐，顺便校验那些**必须是 0** 的对齐位（`pcm_alignment_zero_bit`）。
    pub fn align_zero_bits(&mut self) -> Result<()> {
        while !self.aligned() {
            if self.r.read_bit()? != 0 {
                return Err(MediaError::Bitstream(
                    "对齐位不是 0（pcm_alignment_zero_bit 必须为 0）".into(),
                ));
            }
        }
        Ok(())
    }

    /// 读一个字节（`I_PCM` 的样本就是 8 位一读）。
    pub fn byte(&mut self) -> Result<u8> {
        self.r.read_u32(8).map(|v| v as u8)
    }

    /// §7.2 的 `more_rbsp_data()`：**当前位之后还有没有真正的数据**。
    ///
    /// 判法：`rbsp_trailing_bits` 是"一个 `1` + 补零到字节对齐"，所以
    /// **整条 RBSP 里最后一个为 1 的位就是它的起点**；当前位还没走到那儿就说明
    /// 后面还有数据。这条判据同时是切片数据循环的终止条件，也是"有没有多读/少读"
    /// 的第一道保险（真判据见 [`check_rbsp_trailing`]）。
    pub fn more_rbsp_data(&self) -> bool {
        match last_set_bit(self.raw) {
            Some(last) => self.r.position_bits() < last,
            None => false,
        }
    }

    /// 已经读掉的位数（**自 NAL 头起算、去竞争字节后**的口径）。
    pub fn position_bits(&self) -> usize {
        self.r.position_bits()
    }

    pub fn remaining_bits(&self) -> usize {
        self.r.remaining_bits()
    }

    /// 收尾校验：解完之后当前位必须**正好**停在 `rbsp_trailing_bits` 的那个 `1` 上。
    ///
    /// 这是 CAVLC 这条路上最硬的一道判据 —— 多读一位（多余的数据）或早停（少读）
    /// 都对不上，而且**不需要参考解码器**。见 `me-cli` 的 `h264_syntax` 用例。
    pub fn check_rbsp_trailing(&self) -> Result<()> {
        let Some(last) = last_set_bit(self.raw) else {
            return Err(MediaError::Bitstream("切片 NAL 里一个 1 位都没有".into()));
        };
        let pos = self.r.position_bits();
        if pos != last {
            return Err(MediaError::Bitstream(format!(
                "切片数据没有正好停在 rbsp_trailing_bits 上：停在 {pos} 位、收尾位在 {last} 位（{}）",
                if pos < last { "少读了" } else { "多读了" }
            )));
        }
        Ok(())
    }

    /// §9.2 的总入口：解一个残差块，返回**按扫描序**排列的系数。
    ///
    /// `max_num_coeff`：4x4 亮度 AC 是 15、普通 4x4/8x8 是 16、色度 DC（4:2:0）是 4。
    /// `n_c`：§9.2.1 的上下文选择量（可以小于 0）。
    ///
    /// 返回 `(系数数组, 有效长度)`；`total_coeff == 0` 时全零（调用方据此走"跳过块"）。
    pub fn residual_block(&mut self, max_num_coeff: usize, n_c: i32) -> Result<([i32; 16], usize)> {
        let mut coeff = [0i32; 16];
        let (trailing_ones, total_coeff) = self.coeff_token(n_c)?;
        if total_coeff == 0 {
            return Ok((coeff, max_num_coeff));
        }
        if total_coeff as usize > max_num_coeff {
            return Err(MediaError::Bitstream(format!(
                "CAVLC 的 TotalCoeff = {total_coeff} 超过了这个块的系数个数 {max_num_coeff}"
            )));
        }
        let levels = self.levels(trailing_ones as usize, total_coeff as usize)?;
        let runs = self.runs(total_coeff as usize, max_num_coeff)?;
        // §9.2.4：coeffNum 从 −1 起、i 从 TotalCoeff−1 **倒着**走
        let mut coeff_num: i32 = -1;
        for i in (0..total_coeff as usize).rev() {
            coeff_num += runs[i] as i32 + 1;
            if coeff_num as usize >= max_num_coeff {
                return Err(MediaError::Bitstream(format!(
                    "CAVLC 的 run 把系数放到了第 {coeff_num} 位（块只有 {max_num_coeff} 个）"
                )));
            }
            coeff[coeff_num as usize] = levels[i];
        }
        Ok((coeff, max_num_coeff))
    }

    /// §9.2.1 `coeff_token`：`(TrailingOnes, TotalCoeff)`。
    pub fn coeff_token(&mut self, n_c: i32) -> Result<(u8, u8)> {
        let table: &[(u8, u16, u8, u8)] = match n_c {
            i32::MIN..=-2 => COEFF_TOKEN_NC_M2,
            -1 => COEFF_TOKEN_NC_M1,
            0..=1 => COEFF_TOKEN_NC_LT2,
            2..=3 => COEFF_TOKEN_NC_2_4,
            4..=7 => COEFF_TOKEN_NC_4_8,
            _ => COEFF_TOKEN_NC_GE8,
        };
        let mut code = 0u32;
        for len in 1..=16u8 {
            code = (code << 1) | self.r.read_bit()? as u32;
            if let Ok(i) = table.binary_search_by_key(&(len, code as u16), |e| (e.0, e.1)) {
                return Ok((table[i].2, table[i].3));
            }
        }
        Err(MediaError::Bitstream(format!(
            "CAVLC 的 coeff_token 在 nC = {n_c} 的表里找不到匹配码字"
        )))
    }

    /// §9.2.2 `level`：返回 `levelVal[0..total_coeff]`（`levelVal[0]` 是**最高频**那个）。
    fn levels(&mut self, trailing_ones: usize, total_coeff: usize) -> Result<[i32; 16]> {
        let mut level_val = [0i32; 16];
        for item in level_val.iter_mut().take(trailing_ones) {
            // trailing_ones_sign_flag：0 → +1，1 → −1（注意是**反的**）
            *item = if self.r.read_bit()? == 0 { 1 } else { -1 };
        }
        let mut suffix_length: u32 = if total_coeff > 10 && trailing_ones < 3 { 1 } else { 0 };
        let mut i = trailing_ones;
        while i < total_coeff {
            let level_prefix = self.level_prefix()?;
            let level_suffix_size = if level_prefix == 14 && suffix_length == 0 {
                4
            } else if level_prefix >= 15 {
                level_prefix - 3
            } else {
                suffix_length
            };
            let level_suffix = if level_suffix_size > 0 {
                self.r.read_u32(level_suffix_size as u8)?
            } else {
                0
            };
            let mut level_code = (level_prefix.min(15) << suffix_length) + level_suffix;
            if level_prefix >= 15 && suffix_length == 0 {
                level_code += 15;
            }
            if level_prefix >= 16 {
                // `1 << (level_prefix - 3)` 可以到 1<<12 以上，用 u32 算完再折算
                level_code =
                    level_code.wrapping_add((1u32 << (level_prefix - 3)).wrapping_sub(4096));
            }
            if i == trailing_ones && trailing_ones < 3 {
                level_code += 2;
            }
            let level = if level_code.is_multiple_of(2) {
                ((level_code + 2) >> 1) as i32
            } else {
                // 规范写的是 `(−levelCode − 1) >> 1`（算术右移 = 向下取整）；
                // 这里 `levelCode` 恒为非负，等价于 `−((levelCode+1)>>1)`
                -(((level_code + 1) >> 1) as i32)
            };
            level_val[i] = level;
            if suffix_length == 0 {
                suffix_length = 1;
            }
            if level.unsigned_abs() > (3u32 << (suffix_length - 1)) && suffix_length < 6 {
                suffix_length += 1;
            }
            i += 1;
        }
        Ok(level_val)
    }

    /// §9.2.2.1 `level_prefix`：读到第一个 `1` 为止的**前导零个数**。
    fn level_prefix(&mut self) -> Result<u32> {
        let mut zeros = 0u32;
        loop {
            if zeros > 63 {
                return Err(MediaError::Bitstream(
                    "CAVLC 的 level_prefix 超过 63 位还没读到 1（位流错位或数据损坏）".into(),
                ));
            }
            if self.r.read_bit()? == 1 {
                return Ok(zeros);
            }
            zeros += 1;
        }
    }

    /// §9.2.3 `run_before` / `total_zeros`：返回 `runVal[0..total_coeff]`。
    fn runs(&mut self, total_coeff: usize, max_num_coeff: usize) -> Result<[u32; 16]> {
        let mut run_val = [0u32; 16];
        // TotalCoeff == maxNumCoeff 时 total_zeros **不读**，zerosLeft 直接是 0
        let mut zeros_left: u32 = if total_coeff == max_num_coeff {
            0
        } else if max_num_coeff == 4 {
            // 色度 DC 的 2x2（4:2:0）：表 9-9(a)，**不是**表 9-7
            self.total_zeros_chroma_dc(total_coeff)? as u32
        } else if max_num_coeff == 8 {
            return Err(MediaError::Unsupported(
                "CAVLC 走了 4:2:2 的色度 DC（maxNumCoeff = 8，表 9-9(b)），引擎只做 4:2:0".into(),
            ));
        } else {
            self.total_zeros(total_coeff)? as u32
        };
        for item in run_val.iter_mut().take(total_coeff.saturating_sub(1)) {
            let run = if zeros_left > 0 {
                let r = self.run_before(zeros_left)? as u32;
                if r > zeros_left {
                    return Err(MediaError::Bitstream(format!(
                        "CAVLC 的 run_before = {r} 超过了 zerosLeft = {zeros_left}"
                    )));
                }
                zeros_left -= r;
                r
            } else {
                0
            };
            *item = run;
        }
        run_val[total_coeff.saturating_sub(1)] = zeros_left;
        Ok(run_val)
    }

    /// §9.2.3 的 `total_zeros`。表按 `maxNumCoeff` 分三套。
    fn total_zeros(&mut self, total_coeff: usize) -> Result<u8> {
        let table: &[(u8, u16, u8)] = if total_coeff == 0 {
            return Err(MediaError::Bitstream("total_zeros 只对 TotalCoeff > 0 有意义".into()));
        } else if total_coeff <= 15 {
            TOTAL_ZEROS[total_coeff - 1]
        } else {
            return Err(MediaError::Bitstream(format!(
                "total_zeros 的 tzVlcIndex = {total_coeff} 越界（1..15）"
            )));
        };
        decode_vlc(&mut self.r, table, "total_zeros").map(|v| v as u8)
    }

    /// 色度 DC（4:2:0 的 2x2）专用：`maxNumCoeff == 4` 时用表 9-9(a)。
    fn total_zeros_chroma_dc(&mut self, total_coeff: usize) -> Result<u8> {
        let idx = total_coeff
            .checked_sub(1)
            .filter(|i| *i < 3)
            .ok_or_else(|| MediaError::Bitstream(format!("色度 DC 的 tzVlcIndex 越界：{total_coeff}")))?;
        decode_vlc(&mut self.r, CHROMA_DC_TOTAL_ZEROS[idx], "色度 DC total_zeros")
            .map(|v| v as u8)
    }

    /// §9.2.3 的 `run_before`（表 9-10，列由 `zerosLeft` 选）。
    fn run_before(&mut self, zeros_left: u32) -> Result<u8> {
        let idx = zeros_left.clamp(1, 7) as usize - 1;
        decode_vlc(&mut self.r, RUN_BEFORE[idx], "run_before").map(|v| v as u8)
    }
}

/// 缓冲里最后一个为 `1` 的位的下标（从缓冲起点、按位算）。
///
/// `rbsp_trailing_bits` 就是"一个 `1` + 补零"，所以这个下标**就是收尾位的起点**。
/// 用它做判据的好处：CABAC 那边的 `cabac_zero_word`（补的 0x0000）不会干扰
/// —— 尾部补零再多也改变不了"最后一个 1"的位置。
fn last_set_bit(buf: &[u8]) -> Option<usize> {
    for (i, b) in buf.iter().enumerate().rev() {
        if *b != 0 {
            return Some(i * 8 + (7 - b.trailing_zeros() as usize));
        }
    }
    None
}

/// `tzVlcIndex = 1..=15` 对应的表（`maxNumCoeff` 为 16 或 8 时用）。
static TOTAL_ZEROS: [&[(u8, u16, u8)]; 15] = [
    TOTAL_ZEROS_TZ1,
    TOTAL_ZEROS_TZ2,
    TOTAL_ZEROS_TZ3,
    TOTAL_ZEROS_TZ4,
    TOTAL_ZEROS_TZ5,
    TOTAL_ZEROS_TZ6,
    TOTAL_ZEROS_TZ7,
    TOTAL_ZEROS_TZ8,
    TOTAL_ZEROS_TZ9,
    TOTAL_ZEROS_TZ10,
    TOTAL_ZEROS_TZ11,
    TOTAL_ZEROS_TZ12,
    TOTAL_ZEROS_TZ13,
    TOTAL_ZEROS_TZ14,
    TOTAL_ZEROS_TZ15,
];

static CHROMA_DC_TOTAL_ZEROS: [&[(u8, u16, u8)]; 3] = [
    CHROMA_DC_TOTAL_ZEROS_TZ1,
    CHROMA_DC_TOTAL_ZEROS_TZ2,
    CHROMA_DC_TOTAL_ZEROS_TZ3,
];

/// `zerosLeft = 1..=6` 与 `>6`（第 7 项）对应的 `run_before` 表。
static RUN_BEFORE: [&[(u8, u16, u8)]; 7] = [
    RUN_BEFORE_ZL1,
    RUN_BEFORE_ZL2,
    RUN_BEFORE_ZL3,
    RUN_BEFORE_ZL4,
    RUN_BEFORE_ZL5,
    RUN_BEFORE_ZL6,
    RUN_BEFORE_ZL7,
];

/// 按"逐位读 + 二分查表"解一个 VLC（表必须按 `(位长, 码字)` 升序）。
fn decode_vlc(r: &mut BitReader<'_>, table: &[(u8, u16, u8)], name: &str) -> Result<u32> {
    let mut code = 0u32;
    for len in 1..=16u8 {
        code = (code << 1) | r.read_bit()? as u32;
        if let Ok(i) = table.binary_search_by_key(&(len, code as u16), |e| (e.0, e.1)) {
            return Ok(table[i].2 as u32);
        }
    }
    Err(MediaError::Bitstream(format!(
        "CAVLC 的 {name} 找不到匹配码字（位流错位或数据损坏）"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 把码字按位写进缓冲（大端位序），**并做防竞争字节插入** —— 与真实码流同形态。
    struct BitWriter {
        bits: Vec<u8>,
        pos: usize,
    }

    impl BitWriter {
        fn new() -> Self {
            Self { bits: Vec::new(), pos: 0 }
        }

        fn bit(&mut self, b: u8) -> &mut Self {
            if self.pos.is_multiple_of(8) {
                self.bits.push(0);
            }
            let last = self.bits.len() - 1;
            self.bits[last] |= (b & 1) << (7 - (self.pos % 8));
            self.pos += 1;
            self
        }

        fn bits(&mut self, value: u32, n: u32) -> &mut Self {
            for i in (0..n).rev() {
                self.bit(((value >> i) & 1) as u8);
            }
            self
        }

        /// 按"码字字面量"写：字符串里 1/0 的直接顺序。
        fn code(&mut self, s: &str) -> &mut Self {
            for ch in s.chars() {
                if ch == '0' || ch == '1' {
                    self.bit(ch as u8 - b'0');
                }
            }
            self
        }

        fn finish(&self) -> Vec<u8> {
            self.bits.clone()
        }
    }

    fn reader(bytes: &[u8]) -> Cavlc<'_> {
        Cavlc::new(bytes, 0).unwrap()
    }

    /// `nC` 把 6 张表选对 —— 选错表在真实码流上表现为"整片雪花"，很难查。
    #[test]
    fn coeff_token_picks_the_table_by_nc() {
        // 同一个 6 位码字 `000101` 在 0 <= nC < 2 那一列是 (TrailingOnes 0, TotalCoeff 1)，
        // 在 8 <= nC 那一列是 (1, 2)（那张表是纯 6 位码）
        let mut w = BitWriter::new();
        w.code("000101");
        let bytes = w.finish();
        assert_eq!(reader(&bytes).coeff_token(0).unwrap(), (0, 1), "nC ∈ [0,2)");
        assert_eq!(reader(&bytes).coeff_token(8).unwrap(), (1, 2), "nC >= 8");

        // nC == -2 的那一列（4:2:2 的色度 DC）：(0,1) 是 "0001 111"
        let mut w = BitWriter::new();
        w.code("0001111");
        let bytes = w.finish();
        assert_eq!(reader(&bytes).coeff_token(-2).unwrap(), (0, 1));

        // nC == -1 的那一列（4:2:0）：(1,3) 是 "0000 011"
        let mut w = BitWriter::new();
        w.code("0000011");
        let bytes = w.finish();
        assert_eq!(reader(&bytes).coeff_token(-1).unwrap(), (1, 3));
    }

    /// §9.2.2 的 `level_prefix` / `level_suffix`：手算一个"全是 ±1"的块。
    ///
    /// 这里顺带钉住两件容易搞反的事：
    /// 1. **`levelVal[0]` 是最高频那个**（§9.2.4 从 `TotalCoeff−1` 倒着放），
    ///    所以符号位的顺序与扫描序是**反的**；
    /// 2. 第一个非 trailing 的 level 要 **+2**，于是它**不可能是 ±1**
    ///    （±1 早就该被 trailing ones 占了，而 trailing ones 最多 3 个）。
    #[test]
    fn levels_of_trailing_ones_only() {
        // TotalCoeff = 3、TrailingOnes = 3、nC = 0 的码字是 "00011"（表里 (3,3)）
        // 之后跟 3 个符号位：0(→+1) 1(→−1) 0(→+1)
        let mut w = BitWriter::new();
        w.code("00011").bit(0).bit(1).bit(0);
        // total_zeros（T=3 → 表 9-7 第 3 列）：total_zeros = 1 → "111"
        w.code("111");
        // run_before：TotalCoeff−1 = 2 次，zerosLeft 从 1 开始
        //   i=0：zerosLeft=1 → run_before=0（表 9-10 zerosLeft=1 的 "1"）→ zerosLeft 仍 1
        //   i=1：再来一次 "1"
        //   末尾把剩下的 zerosLeft = 1 记给 runVal[2]
        w.code("1").code("1");
        let bytes = w.finish();
        let mut c = reader(&bytes);
        let (coeff, _) = c.residual_block(16, 0).unwrap();
        // runs = [0, 0, 1]；coeffNum 从 −1 起、i 倒着走：
        //   i=2 → −1+1+1 = 1 ⇒ coeff[1] = levelVal[2] = +1
        //   i=1 →  1+0+1 = 2 ⇒ coeff[2] = levelVal[1] = −1
        //   i=0 →  2+0+1 = 3 ⇒ coeff[3] = levelVal[0] = +1
        assert_eq!(coeff[0], 0, "DC 位置没有非零系数（run 把它让开了）");
        assert_eq!(coeff[1], 1, "levelVal[2]（最后解码的那个）落在最低的 coeffNum");
        assert_eq!(coeff[2], -1);
        assert_eq!(coeff[3], 1, "levelVal[0]（最先解码的）落在最高的 coeffNum");
        // 写进去一共 13 位（5 的码字 + 3 个符号 + 3 的 total_zeros + 2 个 run_before），
        // 位写器补齐到 16 —— 解码器只能吃掉那 13 位
        assert_eq!(c.position_bits(), 13, "正好读完，不多不少");
        assert_eq!(c.remaining_bits(), 3, "只剩位写器的补零");
    }

    /// `TotalCoeff == maxNumCoeff` 时 `total_zeros` **不许读**。
    /// 这一条错了会从这一块起整条流错位。
    #[test]
    fn total_zeros_is_not_read_when_the_block_is_full() {
        // nC == -2、maxNumCoeff = 4：(0,4) 的码字是 "0000 0011 0"（9 位）
        let mut w = BitWriter::new();
        w.code("000000110");
        // 4 个 level，全是非 trailing：
        //   i=0：prefix=0（"1"）、suffixLength=0 ⇒ 不读后缀 → levelCode 0 +2 → level = 2
        //   i=1..3：suffixLength 已是 1 ⇒ 每个 "1" + 1 位后缀 0 → levelCode 0 → level = 1
        w.code("1").code("1").bits(0, 1).code("1").bits(0, 1).code("1").bits(0, 1);
        // 紧接着放 4 个 0 位当"哨兵"：如果实现去读了 total_zeros，这几位会被吃掉
        w.bits(0, 4);
        let bytes = w.finish();
        let mut c = reader(&bytes);
        let (coeff, len) = c.residual_block(4, -2).unwrap();
        assert_eq!(len, 4);
        assert_eq!(coeff[..4], [1, 1, 1, 2], "levelVal 倒着放进 coeffNum");
        // 位写器把 20 位补齐到 24：解完 16 位后还剩 8 位 = 4 个哨兵 + 4 个补零
        assert_eq!(c.remaining_bits(), 8, "哨兵位一位都没被吃掉（被吃的会只剩 4 位）");
    }

    /// `nC` 的推导口径（§9.2.1 第 7 步）在 `residual_block` 之外，这里只钉
    /// "负 nC 能落到色度 DC 的表上"。
    #[test]
    fn negative_nc_uses_the_chroma_dc_tables() {
        // nC == -1 的 (0,1) → "0001 11"
        let mut w = BitWriter::new();
        w.code("000111");
        let bytes = w.finish();
        assert_eq!(reader(&bytes).coeff_token(-1).unwrap(), (0, 1));
    }

    #[test]
    fn empty_bits_is_an_error_not_a_panic() {
        let bytes = [0u8; 4]; // 全零：任何表的码字都不会是"永远读 0"
        let mut c = reader(&bytes);
        assert!(c.coeff_token(0).is_err());
    }
}
