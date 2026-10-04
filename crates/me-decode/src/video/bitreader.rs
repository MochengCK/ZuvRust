//! AV1 位读取器：MSB-first，外加 AV1 自己的三个读取原语。
//!
//! 为什么不复用 `crate::audio::bitreader`：那个面向 FLAC/MP3/AAC 的
//! **码字**读取（Rice 残差、Huffman 前缀），每次读几个 bit 就完事；AV1 这边
//! 需要的是**位域 + 无符号指数哥伦布（`uvlc`）+ LEB128**，而且 `uvlc` 的
//! "先数前导 1"是变长循环、LEB128 要求字节对齐 —— 语义完全不同。
//! 硬凑在一起只会让两边的边界行为互相污染（那正是"读错一位满盘皆错"的来源）。
//!
//! 另一个必须自己持有的东西：**位下标**。AV1 的解析处处要"读到哪了"（帧内
//! 熵段的起点、OBU 载荷的边界、`byte_alignment` 的填充位），所以这里同时暴露
//! 位下标与字节下标。

use me_core::{MediaError, Result};

/// `FloorLog2(x)`：`x > 0` 时 x 的最高位下标（规范里 `ns()` / `tile_log2()` 要用）。
pub fn floor_log2(x: u32) -> u32 {
    31 - x.leading_zeros()
}

/// 按位读（大端位序：先读字节的最高位）。
pub struct BitReader<'a> {
    data: &'a [u8],
    /// 当前字节下标
    pos: usize,
    /// 当前字节里已经读掉多少位（0..=7）
    bit: u8,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0, bit: 0 }
    }

    /// 还剩多少位（越界的部分不计）。
    pub fn remaining_bits(&self) -> usize {
        if self.pos >= self.data.len() {
            return 0;
        }
        (self.data.len() - self.pos) * 8 - self.bit as usize
    }

    /// 已经读过的位数（诊断：帧内熵段的起点就是这么算出来的）。
    pub fn bits_read(&self) -> usize {
        self.pos * 8 + self.bit as usize
    }

    /// 当前是否停在字节边界上。
    pub fn byte_aligned(&self) -> bool {
        self.bit == 0
    }

    /// 读 `n`（≤32）位。
    ///
    /// 越界返回 `NeedMore`（而不是 0）：AV1 的解析里"读不到"和"读到 0"是完全
    /// 不同的结论 —— 前者说明这个 OBU 被截断了，后者可能是合法数据。
    pub fn f(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(MediaError::Bitstream(format!("AV1 一次最多读 32 位（要 {n}）")));
        }
        if self.remaining_bits() < n as usize {
            return Err(MediaError::NeedMore {
                need: n as usize,
                have: self.remaining_bits(),
            });
        }
        let mut out: u32 = 0;
        let mut left = n;
        while left > 0 {
            let byte = self.data[self.pos];
            let avail = 8 - self.bit;
            let take = avail.min(left as u8);
            let shift = avail - take;
            let mask = if take >= 8 { 0xFFu16 } else { (1u16 << take) - 1 } as u8;
            let chunk = (byte >> shift) & mask;
            out = (out << take) | chunk as u32;
            self.bit += take;
            if self.bit == 8 {
                self.bit = 0;
                self.pos += 1;
            }
            left -= take as u32;
        }
        Ok(out)
    }

    /// 读 1 位当布尔（AV1 里的 `f(1)` 遍地都是）。
    pub fn bit(&mut self) -> Result<bool> {
        Ok(self.f(1)? == 1)
    }

    /// `uvlc()`（AV1 规范 §4.10.3）：**前导 0 的个数**决定后面还要读几位。
    ///
    /// 编码形态（这是最容易写反的地方 —— 前导位是 0 不是 1）：
    ///
    /// | 值 | 位串 |
    /// |---|---|
    /// | 0 | `1` |
    /// | 1 | `01` + 1 位（`0`）|
    /// | 2 | `01` + 1 位（`1`）|
    /// | 3..6 | `001` + 2 位 |
    ///
    /// 即"`lz` 个 0 → 一个 1 → 再读 `lz` 位"，值 = `(1<<lz) - 1 + 读到的那几位`。
    /// 前导 0 到 32 个是规范里的越界形态，返回 `u32::MAX` 交给调用方判非法。
    pub fn uvlc(&mut self) -> Result<u32> {
        let mut leading_zeros = 0u32;
        loop {
            if self.bit()? {
                break;
            }
            leading_zeros += 1;
            if leading_zeros >= 32 {
                return Ok(u32::MAX);
            }
        }
        if leading_zeros == 0 {
            return Ok(0);
        }
        let more = self.f(leading_zeros)?;
        Ok((1u32 << leading_zeros) - 1 + more)
    }

    /// LEB128（AV1 的 OBU 长度字段用它）。**必须字节对齐**。
    pub fn leb128(&mut self) -> Result<u64> {
        if !self.byte_aligned() {
            return Err(MediaError::Bitstream("LEB128 只能在字节边界上读".into()));
        }
        let mut value: u64 = 0;
        let mut shift = 0u32;
        loop {
            if self.pos >= self.data.len() {
                return Err(MediaError::NeedMore { need: 1, have: 0 });
            }
            let byte = self.data[self.pos];
            self.pos += 1;
            value |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
            if shift >= 56 {
                return Err(MediaError::Bitstream("LEB128 太长（超过 8 字节）".into()));
            }
        }
    }

    /// `trailing_bits()`：一个 `1` 后面全 `0`（补到字节边界）。
    ///
    /// **这是抓"位偏移"最有效的一道检查**：AV1 的语法结构末尾都有它，少读/多读
    /// 一位，这个 `1` 就不会落在最后一位上。实测依据：真实 AV1 素材（B 站 1080p）
    /// 的序列头载荷 14 字节，按修正后的位序逐字段读完正好 111 位，剩下的那一位
    /// 就是 `1`（最后一个字节 `0x41` 的最低位）—— 一位不差。
    pub fn trailing_bits(&mut self) -> Result<()> {
        if self.remaining_bits() == 0 {
            return Err(MediaError::Bitstream(
                "载荷末尾缺少 trailing bit（多半是前面的字段多读了几位）".into(),
            ));
        }
        if !self.bit()? {
            return Err(MediaError::Bitstream(
                "载荷末尾的 trailing bit 不是 1（多半是前面的字段少读/多读了几位）".into(),
            ));
        }
        while self.remaining_bits() > 0 {
            if self.bit()? {
                return Err(MediaError::Bitstream(
                    "trailing bit 之后应当全是 0 填充".into(),
                ));
            }
        }
        Ok(())
    }

    /// `su(n)`（AV1 规范 §4.10.6）：`f(n)` 读到的值按**补码**解释成有符号数。
    ///
    /// 规范原文就三行：读 `n` 位 → 最高位当符号位 → 置位时减 `2 * signMask`。
    /// 它出现在 `read_delta_q()` 与分割特征里，**位宽由调用方算**（`su(1 + bits)`）。
    pub fn su(&mut self, n: u32) -> Result<i32> {
        if n == 0 {
            return Ok(0);
        }
        let value = self.f(n)?;
        let sign_mask = 1u32 << (n - 1);
        if value & sign_mask != 0 {
            Ok(value as i32 - 2 * sign_mask as i32)
        } else {
            Ok(value as i32)
        }
    }

    /// `ns(n)`（AV1 规范 §4.10.7，"non-symmetric"）：值域 `0..n`，**不定长**。
    ///
    /// 先读 `floor(log2(n))` 位；若落在"短码"区间（`v < m`）就直接是结果，
    /// 否则再读 1 位拼起来。它让 `0..n` 这种"非 2 的幂"的值域不浪费码字
    /// —— `tile_info()` 里瓦片宽高的增量就是用它编码的。
    pub fn ns(&mut self, n: u32) -> Result<u32> {
        if n <= 1 {
            return Ok(0);
        }
        let w = floor_log2(n) + 1;
        let m = (1u32 << w) - n;
        let v = self.f(w - 1)?;
        if v < m {
            return Ok(v);
        }
        let extra = self.f(1)?;
        Ok((v << 1) - m + extra)
    }

    /// `byte_alignment()`（AV1 规范 §4.10.8）：读 `0` 位直到字节边界。
    ///
    /// **它不是"跳过"**：填充位在规范里必须是 0。这条检查的价值在于"读少了位"
    /// 的形态 —— 那时第一个"填充位"其实是数据的下一位，有一半概率是 1，
    /// 于是逐帧叠加起来就成了很硬的一道位偏移检查（真素材 5.5 万帧全过，
    /// 说明非压缩头的位长没有系统性偏差）。
    pub fn byte_align(&mut self) -> Result<()> {
        while !self.byte_aligned() {
            if self.bit()? {
                return Err(MediaError::Bitstream(
                    "字节对齐位不是 0（非压缩头多半读少/读多了几位）".into(),
                ));
            }
        }
        Ok(())
    }

    /// 从当前字节边界上切走 `n` 个字节（OBU 载荷这种"整段搬"的场合用）。
    pub fn take_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if !self.byte_aligned() {
            return Err(MediaError::Bitstream("切字节要求先字节对齐".into()));
        }
        if self.pos + n > self.data.len() {
            return Err(MediaError::NeedMore { need: n, have: self.data.len() - self.pos });
        }
        let out = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_bitfields_msb_first() {
        // 0b1010_0110 0b1100_0000
        let mut r = BitReader::new(&[0xA6, 0xC0]);
        assert_eq!(r.f(1).unwrap(), 1, "最高位先出来");
        assert_eq!(r.f(3).unwrap(), 0b010);
        assert_eq!(r.f(4).unwrap(), 0b0110);
        assert_eq!(r.f(4).unwrap(), 0b1100);
        assert_eq!(r.f(4).unwrap(), 0b0000);
        assert_eq!(r.remaining_bits(), 0);
        assert!(r.f(1).is_err(), "读完了要报 NeedMore，不是给 0");
    }

    #[test]
    fn reads_across_byte_boundary() {
        let mut r = BitReader::new(&[0xFF, 0x0F]);
        assert_eq!(r.f(12).unwrap(), 0xFF0, "跨字节读要拼对");
        assert_eq!(r.f(4).unwrap(), 0xF);
    }

    #[test]
    fn uvlc_matches_spec_examples() {
        // 值 0 → 一个 '1' 位
        let mut r = BitReader::new(&[0b1000_0000]);
        assert_eq!(r.uvlc().unwrap(), 0);
        // 值 1 → "01" + 1 位 0
        let mut r = BitReader::new(&[0b0100_0000]);
        assert_eq!(r.uvlc().unwrap(), 1);
        // 值 2 → "01" + 1 位 1
        let mut r = BitReader::new(&[0b0110_0000]);
        assert_eq!(r.uvlc().unwrap(), 2);
        // 值 5 → "001" + 2 位 10 = 3 + 2
        let mut r = BitReader::new(&[0b0011_0000]);
        assert_eq!(r.uvlc().unwrap(), 5, "lz=2、more=0b10 → 3 + 2");
    }

    #[test]
    fn uvlc_overflows_to_u32_max() {
        // 33 个 0 位开头：规范里是"越界"的形态
        let mut r = BitReader::new(&[0x00, 0x00, 0x00, 0x00, 0x80]);
        assert_eq!(r.uvlc().unwrap(), u32::MAX);
    }

    #[test]
    fn leb128_reads_lengths() {
        let mut r = BitReader::new(&[0x00]);
        assert_eq!(r.leb128().unwrap(), 0);
        let mut r = BitReader::new(&[0x7F]);
        assert_eq!(r.leb128().unwrap(), 127, "单字节上限");
        let mut r = BitReader::new(&[0x80, 0x01]);
        assert_eq!(r.leb128().unwrap(), 128, "跨字节：低位 7 位 + 高 7 位");
        let mut r = BitReader::new(&[0xFF, 0x7F]);
        assert_eq!(r.leb128().unwrap(), 16383);
    }

    #[test]
    fn leb128_needs_byte_alignment() {
        let mut r = BitReader::new(&[0x00, 0x80, 0x01]);
        r.f(1).unwrap();
        assert!(r.leb128().is_err(), "不在字节边界上读 LEB128 要报错");
    }

    #[test]
    fn trailing_bits_must_be_one_then_zeros() {
        // 0b1010_1000：前 4 位是语法，后面 1 位 '1' + 3 位 0 = 合法的 trailing bits
        let mut r = BitReader::new(&[0b1010_1000]);
        r.f(4).unwrap();
        r.trailing_bits().unwrap();
        // trailing bit 不是 1
        let mut r = BitReader::new(&[0b1010_0000]);
        r.f(4).unwrap();
        assert!(r.trailing_bits().is_err());
        // 已经不是字节边界了（语法少读了）
        let mut r = BitReader::new(&[0b1010_1000]);
        r.f(5).unwrap();
        assert!(r.trailing_bits().is_err(), "填位不为 0");
        // 一个字节刚好读完（连 trailing bit 都没有）
        let mut r = BitReader::new(&[0b1010_1010]);
        r.f(8).unwrap();
        assert!(r.trailing_bits().is_err());
    }

    #[test]
    fn take_bytes_returns_slice_and_advances() {
        let mut r = BitReader::new(&[1, 2, 3, 4]);
        assert_eq!(r.take_bytes(2).unwrap(), &[1, 2]);
        assert_eq!(r.take_bytes(2).unwrap(), &[3, 4]);
        assert!(r.take_bytes(1).is_err());
    }

    #[test]
    fn floor_log2_matches_definition() {
        assert_eq!(floor_log2(1), 0);
        assert_eq!(floor_log2(2), 1);
        assert_eq!(floor_log2(3), 1);
        assert_eq!(floor_log2(4), 2);
        assert_eq!(floor_log2(4096), 12);
    }

    #[test]
    fn su_is_twos_complement() {
        // 3 位：0b000 → 0、0b111 → -1
        let mut r = BitReader::new(&[0b000_111_10]);
        assert_eq!(r.su(3).unwrap(), 0);
        assert_eq!(r.su(3).unwrap(), -1);
        // 0 位是合法调用（分割特征 j=6/7 的 bitsToRead = 0）
        assert_eq!(r.su(0).unwrap(), 0);
        // 1 位：1 → -1，0 → 0
        let mut r = BitReader::new(&[0b1_0_000000]);
        assert_eq!(r.su(1).unwrap(), -1);
        assert_eq!(r.su(1).unwrap(), 0);
        // 7 位：0b1111111 → 127 - 128 = -1
        let mut r = BitReader::new(&[0b1111111_0]);
        assert_eq!(r.su(7).unwrap(), -1);
        // 边界：最大值（0b0111111 = 63）与最小值（0b1000000 = -64）
        let mut r = BitReader::new(&[0b0111111_0]);
        assert_eq!(r.su(7).unwrap(), 63);
        let mut r = BitReader::new(&[0b1000000_0]);
        assert_eq!(r.su(7).unwrap(), -64);
    }

    #[test]
    fn ns_covers_zero_to_n_minus_1_exactly() {
        // n = 1：值域只有 0，**不读任何位**
        let mut r = BitReader::new(&[0b1010_0000]);
        assert_eq!(r.ns(1).unwrap(), 0);
        assert_eq!(r.bits_read(), 0);
        // n = 2：w = 2、m = 2 → 读 1 位，v < 2 恒成立 → 值 = v（1 位定长）
        let mut r = BitReader::new(&[0b0_1_000000]);
        assert_eq!(r.ns(2).unwrap(), 0);
        assert_eq!(r.ns(2).unwrap(), 1);
        // n = 3：w = 2、m = 1 → v=0（< m）直接是 0；v=1（>= m）再读 1 位 → 2*1-1+bit
        // 位串 0 1 0 1 1：v=0 → 0；v=1+extra0 → 1；v=1+extra1 → 2
        let mut r = BitReader::new(&[0b0101_1000]);
        assert_eq!(r.ns(3).unwrap(), 0, "v=0b0 < m=1");
        assert_eq!(r.ns(3).unwrap(), 1, "v=0b1、extra=0 → 2-1+0");
        assert_eq!(r.ns(3).unwrap(), 2, "v=0b1、extra=1 → 2-1+1");
        // 值域必须落在 0..n（每个 n 都试一遍，含 2 的幂与非 2 的幂）
        for n in 2..=17u32 {
            for byte in [0x00u8, 0x55, 0xAA, 0xFF] {
                let buf = [byte, byte, byte, byte];
                let mut r = BitReader::new(&buf);
                let v = r.ns(n).unwrap();
                assert!(v < n, "ns({n}) 用 {byte:#04x} 给出了越界值 {v}");
            }
        }
    }

    #[test]
    fn byte_align_rejects_nonzero_padding() {
        // 停在位 4，后面是 0 → 合法
        let mut r = BitReader::new(&[0b1010_0000]);
        r.f(4).unwrap();
        r.byte_align().unwrap();
        assert!(r.byte_aligned());
        // 停在位 4，填充位里出现 1 → 说明前面读少了
        let mut r = BitReader::new(&[0b1010_1000]);
        r.f(4).unwrap();
        assert!(r.byte_align().is_err());
        // 已经对齐：一个位都不读
        let mut r = BitReader::new(&[0xFF]);
        r.f(8).unwrap();
        r.byte_align().unwrap();
        assert_eq!(r.bits_read(), 8);
    }
}
