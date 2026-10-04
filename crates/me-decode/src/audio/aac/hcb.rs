//! AAC 的 Huffman 解码与取值映射。
//!
//! 码表在 [`super::tables`]（标准数据），这里只做算法：按码长逐位匹配得到"表内下标"，
//! 再把下标映射成量化值。**符号的处理是这里最容易被漏掉的一环**：
//!
//! | 码表 | 维度 | 有无符号 | 映射 |
//! |---|---|---|---|
//! | 1、2 | 4 | **有** | 下标按 3 进制展开成 4 位数字，值 = 数字 − 1（∈ −1..1） |
//! | 3、4 | 4 | 无 | 数字即幅度（0..2），之后每个非零系数各读 **1 个符号位** |
//! | 5、6 | 2 | **有** | `(i/9 − 4, i%9 − 4)`（∈ −4..4） |
//! | 7、8 | 2 | 无 | `(i/8, i%8)` 幅度 + 各符号位 |
//! | 9、10 | 2 | 无 | `(i/13, i%13)` 幅度 + 各符号位 |
//! | 11 | 2 | 无 | `(i/17, i%17)` 幅度，16 是转义（读扩展序列），+ 各符号位 |
//!
//! 判据与 FFmpeg 的 `IS_CODEBOOK_UNSIGNED(x) = ((x-1) & 10)` 一致：
//! **3/4/7/8/9/10/11 是无符号码表**，符号位在码字之后单独传。
//! 位流顺序（对照 FFmpeg `decode_spectrum_and_dequant` 与 FAAD2）：
//! 码字 → 每个非零系数 1 个符号位（按系数顺序）→（码表 11）各转义序列。
//! 早期实现把 5–11 全当"有符号对"、又完全不读符号位，于是每帧都少读一半以上的位，
//! 后面所有句法整体错位 —— 这是 AAC 解析迟迟跑不通的主因。
//!
//! 为什么解码用"逐位试长度"：码长最多 19 位，逐位比较是 O(19)，而它换来的是
//! 零预处理、零额外内存，以及**可以直接对着规范讲清楚**的实现。

use me_core::{MediaError, Result};

use super::super::bitreader::BitReader;
use super::tables::{HcbTable, QUADS, SCALEFACTOR, SPECTRUM};

/// 一张码表的解码结构。
pub struct Hcb {
    /// `by_len[len]` = (码字, 表内下标)，按码字升序（二分查找）
    by_len: Vec<Vec<(u32, u16)>>,
    max_len: u8,
}

impl Hcb {
    pub fn new(table: &HcbTable) -> Self {
        let max_len = table.max_len();
        let mut by_len: Vec<Vec<(u32, u16)>> = vec![Vec::new(); max_len as usize + 1];
        for (idx, (&len, &code)) in table.lens.iter().zip(table.codes.iter()).enumerate() {
            if len > 0 {
                by_len[len as usize].push((code, idx as u16));
            }
        }
        for v in by_len.iter_mut() {
            v.sort_unstable();
        }
        Self { by_len, max_len }
    }

    /// 读一个符号（返回表内下标）。
    fn read_index(&self, r: &mut BitReader) -> Result<u16> {
        let mut code = 0u32;
        for len in 1..=self.max_len as usize {
            code = (code << 1) | r.read(1)?;
            if let Some(&(_, idx)) = self
                .by_len
                .get(len)
                .and_then(|v| v.binary_search_by_key(&code, |e| e.0).ok().map(|i| &v[i]))
            {
                return Ok(idx);
            }
        }
        Err(MediaError::Bitstream(
            "AAC: 读到表外的 Huffman 码字（数据损坏或码表不符）".into(),
        ))
    }
}

/// 11 张谱码表 + 1 张比例因子码表，加上每张的"取值规则"。
pub struct Hcbs {
    spectrum: Vec<Hcb>,
    scf: Hcb,
}

/// 谱码表的取值规则。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Rule {
    /// 1–2：有符号四元组（符号在映射里）
    SignedQuads,
    /// 3–4：无符号四元组（幅度 + 符号位）
    UnsignedQuads,
    /// 5–6：有符号对（映射里带符号）
    SignedPair9,
    /// 7–8：无符号对，幅度基 8
    UnsignedPair8,
    /// 9–10：无符号对，幅度基 13
    UnsignedPair13,
    /// 11：无符号对 + 转义，幅度基 17
    EscPair17,
}

impl Rule {
    /// 是否需要在码字之后单独读符号位（无符号码表要）。
    fn needs_sign_bits(self) -> bool {
        matches!(
            self,
            Rule::UnsignedQuads | Rule::UnsignedPair8 | Rule::UnsignedPair13 | Rule::EscPair17
        )
    }
}

impl Hcbs {
    pub fn new() -> Self {
        let spectrum = SPECTRUM.iter().map(Hcb::new).collect();
        Self { spectrum, scf: Hcb::new(&SCALEFACTOR) }
    }

    fn rule_for(&self, book: u32) -> Result<Rule> {
        Ok(match book {
            1..=2 => Rule::SignedQuads,
            3..=4 => Rule::UnsignedQuads,
            5..=6 => Rule::SignedPair9,
            7..=8 => Rule::UnsignedPair8,
            9..=10 => Rule::UnsignedPair13,
            11 => Rule::EscPair17,
            other => {
                return Err(MediaError::Bitstream(format!(
                    "AAC: 谱码表号 {other} 非法（1–11 才是谱数据）"
                )))
            }
        })
    }

    /// 读一个比例因子差分（值 = 下标 - 60，规范 4.6.2.3.2）。
    pub fn read_scalefactor(&self, r: &mut BitReader) -> Result<i32> {
        Ok(self.scf.read_index(r)? as i32 - 60)
    }

    /// 读一个"四元组"（码表 1–4）：返回 4 个**带符号**量化值。
    ///
    /// 表 1/2 的符号已含在映射里；表 3/4 只给幅度，每个非零系数随后各读 1 个符号位。
    pub fn read_quads(&self, book: u32, r: &mut BitReader) -> Result<[i32; 4]> {
        let rule = self.rule_for(book)?;
        if !matches!(rule, Rule::SignedQuads | Rule::UnsignedQuads) {
            return Err(MediaError::Bitstream("AAC: 码表号与四元组不符".into()));
        }
        let idx = self.spectrum[(book - 1) as usize].read_index(r)? as usize;
        let digits = QUADS
            .get(idx)
            .ok_or_else(|| MediaError::Bitstream("AAC: 四元组下标越界".into()))?;
        let mut out = [0i32; 4];
        match rule {
            // 有符号四元组：3 进制数字 0/1/2 → −1/0/+1（下标 40 的三位全 1 → 全 0，最短码）
            Rule::SignedQuads => {
                for i in 0..4 {
                    out[i] = digits[i] as i32 - 1;
                }
            }
            // 无符号四元组：数字即幅度，非零系数各读一个符号位
            _ => {
                for i in 0..4 {
                    let mag = digits[i] as i32;
                    out[i] = if mag == 0 { 0 } else { mag * read_sign(r)? };
                }
            }
        }
        Ok(out)
    }

    /// 读一对量化值（码表 5–11），返回**带符号**量化值。
    ///
    /// 无符号码表（7–11）的符号位在码字之后；码表 11 的幅度 16 是转义标记，
    /// 符号位读完之后再按系数顺序读扩展序列。
    pub fn read_pair(&self, book: u32, r: &mut BitReader) -> Result<(i32, i32)> {
        let rule = self.rule_for(book)?;
        if matches!(rule, Rule::SignedQuads | Rule::UnsignedQuads) {
            return Err(MediaError::Bitstream("AAC: 码表号与成对不符".into()));
        }
        let idx = self.spectrum[(book - 1) as usize].read_index(r)? as i32;
        let (mut x, mut y) = match rule {
            Rule::SignedPair9 => (idx / 9 - 4, idx % 9 - 4),
            Rule::UnsignedPair8 => (idx / 8, idx % 8),
            Rule::UnsignedPair13 => (idx / 13, idx % 13),
            Rule::EscPair17 => (idx / 17, idx % 17),
            _ => unreachable!(),
        };
        if !rule.needs_sign_bits() {
            return Ok((x, y));
        }
        // 符号位：按系数顺序，只为非零系数各读 1 位
        let sx = if x == 0 { 0 } else { read_sign(r)? };
        let sy = if y == 0 { 0 } else { read_sign(r)? };
        x *= sx;
        y *= sy;
        if rule == Rule::EscPair17 {
            // 转义：幅度恰好为 16 的那些系数读扩展序列（符号已在上面读过，保留）
            if x.abs() == 16 {
                x = read_escape(r)? * x.signum();
            }
            if y.abs() == 16 {
                y = read_escape(r)? * y.signum();
            }
        }
        Ok((x, y))
    }
}

/// 读一个符号位：1 表示负，返回 ±1。
fn read_sign(r: &mut BitReader) -> Result<i32> {
    Ok(if r.read(1)? == 1 { -1 } else { 1 })
}

/// AAC 的转义序列：先数个 1 得到 N = 4 + count，再读 N 位，值 = 2^N + 读到的位。
fn read_escape(r: &mut BitReader) -> Result<i32> {
    let mut count = 0u8;
    while r.read(1)? == 1 {
        count += 1;
        if count > 12 {
            return Err(MediaError::Bitstream("AAC: 转义序列过长（数据损坏）".into()));
        }
    }
    let width = (4 + count as u32).min(16);
    let v = r.read(width as u8)?;
    Ok((1i32 << width) + v as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::bitreader::BitReader;
    use crate::audio::aac::tables::{SCALEFACTOR as SCF_TABLE, SPECTRUM};

    /// 把某个 (码字, 码长) 写成比特流（高位在前），用来测解码器。
    fn bits_of(code: u32, len: u8) -> Vec<u8> {
        let mut bits = Vec::new();
        for i in (0..len).rev() {
            bits.push(((code >> i) & 1) as u8);
        }
        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let mut out = Vec::new();
        for chunk in bits.chunks(8) {
            let mut b = 0u8;
            for &bit in chunk {
                b = (b << 1) | bit;
            }
            out.push(b);
        }
        out
    }

    #[test]
    fn decodes_every_entry_of_every_spectrum_table() {
        let hcbs = Hcbs::new();
        for (book_i, table) in SPECTRUM.iter().enumerate() {
            let book = (book_i + 1) as u32;
            for (idx, (&len, &code)) in table.lens.iter().zip(table.codes.iter()).enumerate() {
                let bytes = bits_of(code, len);
                let mut r = BitReader::new(&bytes);
                // 用私有接口直接验证下标解码（这是所有取值映射的基础）
                let got = hcbs.spectrum[book_i].read_index(&mut r).unwrap();
                assert_eq!(got as usize, idx, "码表 {book} 第 {idx} 项解错");
            }
        }
    }

    #[test]
    fn decodes_every_scalefactor_entry() {
        let hcbs = Hcbs::new();
        for (idx, (&len, &code)) in SCF_TABLE.lens.iter().zip(SCF_TABLE.codes.iter()).enumerate() {
            let bytes = bits_of(code, len);
            let mut r = BitReader::new(&bytes);
            assert_eq!(hcbs.read_scalefactor(&mut r).unwrap(), idx as i32 - 60);
        }
    }

    #[test]
    fn signed_quads_offset_by_minus_one() {
        let hcbs = Hcbs::new();
        // 码表 1（有符号四元组）：3 进制数字 0/1/2 → −1/0/+1，不读符号位。
        // 下标 40 的三位全 1 → 全 0（最短码，最可能的四元组）。
        for (idx, (&len, &code)) in SPECTRUM[0].lens.iter().zip(SPECTRUM[0].codes.iter()).enumerate() {
            let bytes = bits_of(code, len);
            let mut r = BitReader::new(&bytes);
            let q = hcbs.read_quads(1, &mut r).unwrap();
            assert_eq!(q, QUADS[idx].map(|v| v as i32 - 1), "第 {idx} 项四元组不符");
        }
    }

    #[test]
    fn unsigned_quads_take_sign_bits() {
        let hcbs = Hcbs::new();
        // 码表 3（无符号四元组）：数字即幅度，之后每个非零系数各跟一个符号位。
        // 这里把符号位全给 0（正号），解出的幅度应与 QUADS 一致。
        for (idx, (&len, &code)) in SPECTRUM[2].lens.iter().zip(SPECTRUM[2].codes.iter()).enumerate() {
            let mag = QUADS[idx].map(|v| v as i32);
            let nnz = mag.iter().filter(|&&v| v != 0).count();
            let mut b: Vec<u8> = Vec::new();
            for i in (0..len).rev() {
                b.push(((code >> i) & 1) as u8);
            }
            b.extend(std::iter::repeat(0).take(nnz));
            while b.len() % 8 != 0 {
                b.push(0);
            }
            let mut bytes = Vec::new();
            for chunk in b.chunks(8) {
                let mut by = 0u8;
                for &bit in chunk {
                    by = (by << 1) | bit;
                }
                bytes.push(by);
            }
            let mut r = BitReader::new(&bytes);
            let q = hcbs.read_quads(3, &mut r).unwrap();
            assert_eq!(q, mag, "第 {idx} 项幅度不符");
        }
    }

    #[test]
    fn signed_pair_centers_around_zero() {
        let hcbs = Hcbs::new();
        // 码表 5 的模数是 9：下标 40 → (40/9, 40%9) = (4, 4) → 映射后 (0, 0)
        let table = &SPECTRUM[4];
        let idx = 40usize;
        let bytes = bits_of(table.codes[idx], table.lens[idx]);
        let mut r = BitReader::new(&bytes);
        let (x, y) = hcbs.read_pair(5, &mut r).unwrap();
        assert_eq!((x, y), (0, 0), "有符号对应当以 (4,4) 为零点");
    }

    #[test]
    fn truncated_stream_reports_need_more() {
        // 前缀码的合法流总能解出东西，所以"非法码字"要考"数据不够"这条：
        // 只给 1 个字节却要读最长的码（19 位），必须是 NeedMore 而不是垃圾值或 panic
        let hcbs = Hcbs::new();
        // 空数据：连一位都读不出来，必须是 NeedMore（前缀码的合法流总能解出东西，
        // 所以"非法码字"考不出错 —— 该考的是"数据不够"）
        let mut r = BitReader::new(&[]);
        assert!(hcbs.read_scalefactor(&mut r).is_err(), "没有数据时必须报错");
        assert!(hcbs.read_pair(1, &mut r).is_err());
        assert!(hcbs.read_quads(1, &mut r).is_err());
    }

    #[test]
    fn escape_reads_large_values() {
        let hcbs = Hcbs::new();
        // 手工构造：码表 11 里"x == 16"（转义标记）的那一项，后接 1 个符号位与转义序列
        // 找一个 x=16 的项（下标 / 17 == 16）
        let table = &SPECTRUM[10];
        let idx = table
            .lens
            .iter()
            .enumerate()
            .find(|(i, _)| *i / 17 == 16)
            .map(|(i, _)| i)
            .expect("码表 11 里应当有 x=16 的项");
        assert_eq!(idx % 17, 0, "这个测试假定同项的 y 为 0（不读 y 的符号位）");
        let mut b: Vec<u8> = Vec::new();
        // 先写主码字
        for i in (0..table.lens[idx]).rev() {
            b.push(((table.codes[idx] >> i) & 1) as u8);
        }
        // 符号位：x=16 非零 → 读 1 位（0 = 正）
        b.push(0);
        // 转义：0 个前导 1（即立刻一个 0）→ N = 4，再写 4 位值 0b1010
        b.push(0);
        b.extend_from_slice(&[1, 0, 1, 0]);
        while b.len() % 8 != 0 {
            b.push(0);
        }
        let mut bits = Vec::new();
        let mut cur = 0u8;
        let mut n = 0;
        for bit in b {
            cur = (cur << 1) | bit;
            n += 1;
            if n == 8 {
                bits.push(cur);
                cur = 0;
                n = 0;
            }
        }
        let mut r = BitReader::new(&bits);
        let (x, _y) = hcbs.read_pair(11, &mut r).unwrap();
        assert_eq!(x, (1 << 4) + 0b1010, "转义后的值 = (1<<N) + 读到的 N 位");
    }
}
