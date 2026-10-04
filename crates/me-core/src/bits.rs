//! 位级读取：H.264/H.265 的 SPS、AAC 的 AudioSpecificConfig 都按位打包。
//!
//! 这个读取器**只读不写**：引擎里没有任何"生成 H.264 码流"的需求
//! （合并只搬运，不重编码），所以写位流的代码没必要存在。
//!
//! 两个容易踩的点，这里都处理了：
//!
//! - **H.264 的 SPS 后面跟着填充位的 `1`**，读 Exp-Golomb 读到末尾若只剩填充位，
//!   会得到"语法错误"的假象。调用方要在读完之后自己 `byte_align()`，
//!   并把填充位当垃圾丢弃 —— 见 [`BitReader::read_ue`] 的越界处理（返回 `NeedMore`）。
//! - **起始码防竞争字节 `0x03`**：这不是位读的事，但读 SPS 前必须先去转义，
//!   否则 `0x00 0x00 0x03` 里的 `0x03` 会被当成真实的位数据。

use crate::error::{MediaError, Result};

/// 大端位序的位读取器（H.264/H.265/AAC 都是大端位序）。
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    buf: &'a [u8],
    /// 当前读到第几字节。
    byte: usize,
    /// 当前字节里已消耗的位数（0..8）。
    bit: u8,
}

impl<'a> BitReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, byte: 0, bit: 0 }
    }

    /// 还剩多少**位**。
    pub fn remaining_bits(&self) -> usize {
        (self.buf.len().saturating_sub(self.byte)) * 8 - self.bit as usize
    }

    /// 已经读了多少位。
    pub fn position_bits(&self) -> usize {
        self.byte * 8 + self.bit as usize
    }

    /// 读 1 位。
    pub fn read_bit(&mut self) -> Result<u8> {
        if self.byte >= self.buf.len() {
            return Err(MediaError::NeedMore { need: 1, have: 0 });
        }
        let b = (self.buf[self.byte] >> (7 - self.bit)) & 1;
        self.advance(1);
        Ok(b)
    }

    /// 读 n 位（n ≤ 64），大端位序。
    ///
    /// `n > 64` 直接报错而不是静默截断 —— 这里的调用点全是定长语法元素，
    /// 传大了必然是代码写错。
    pub fn read_bits(&mut self, n: u8) -> Result<u64> {
        if n == 0 {
            return Ok(0);
        }
        if n > 64 {
            return Err(MediaError::Bitstream(format!("一次最多读 64 位，请求 {n}")));
        }
        if self.remaining_bits() < n as usize {
            return Err(MediaError::NeedMore { need: n as usize, have: self.remaining_bits() });
        }
        let mut v: u64 = 0;
        for _ in 0..n {
            v = (v << 1) | self.read_bit()? as u64;
        }
        Ok(v)
    }

    /// 读 n 位当 `u32`（语法元素绝大多数不超过 32 位）。
    pub fn read_u32(&mut self, n: u8) -> Result<u32> {
        if n > 32 {
            return Err(MediaError::Bitstream(format!("read_u32 最多 32 位，请求 {n}")));
        }
        // n == 32 时 read_bits 返回的 u64 正常落在 u32 范围内，直接截断安全
        Ok(self.read_bits(n)? as u32)
    }

    /// 无符号指数哥伦布（H.264 7.2 / H.265 的 `ue(v)`）。
    ///
    /// 编码：`[M 个 0][1][M 位]` → 值 = `2^M - 1 + 后 M 位`。
    /// 领头零超过 32 个说明这不是合法码流（正常语法元素不会到那个量级）。
    pub fn read_ue(&mut self) -> Result<u32> {
        let mut zeros = 0u32;
        loop {
            let b = self.read_bit()?;
            if b == 1 {
                break;
            }
            zeros += 1;
            if zeros > 32 {
                return Err(MediaError::Bitstream("Exp-Golomb 前导零过多，码流非法".into()));
            }
        }
        if zeros == 0 {
            return Ok(0);
        }
        let rest = self.read_u32(zeros as u8)?;
        // 2^zeros - 1 + rest；zeros ≤ 32 时用 u64 算再落回 u32
        let v = ((1u64 << zeros) - 1 + rest as u64) as u32;
        Ok(v)
    }

    /// 有符号指数哥伦布 `se(v)`。
    ///
    /// 映射：`k=ue(v)` → `v = (k+1)/2`（k 为奇数时取负），见 H.264 9.1.1。
    pub fn read_se(&mut self) -> Result<i32> {
        let k = self.read_ue()? as i64;
        let v = if k % 2 == 0 { -(k / 2) } else { (k + 1) / 2 };
        Ok(v as i32)
    }

    /// 对齐到字节边界（丢弃当前字节里剩余的几个填充位）。
    pub fn byte_align(&mut self) {
        if self.bit != 0 {
            self.byte += 1;
            self.bit = 0;
        }
    }

    /// 跳过的位数（只前跳，不做回退）。
    pub fn skip_bits(&mut self, n: usize) -> Result<()> {
        if self.remaining_bits() < n {
            return Err(MediaError::NeedMore { need: n, have: self.remaining_bits() });
        }
        self.advance(n);
        Ok(())
    }

    /// 还剩几位"有效数据"（把尾部填充位也算进来 —— 供调用方诊断）。
    pub fn has_more_data(&self) -> bool {
        self.remaining_bits() > 0
    }

    fn advance(&mut self, n: usize) {
        let total = self.bit as usize + n;
        self.byte += total / 8;
        self.bit = (total % 8) as u8;
    }
}

/// 去掉 H.264/H.265 的起始码防竞争字节（`0x00 0x00 0x03` → `0x00 0x00`）。
///
/// 读 SPS/PPS 之前必须做：`0x03` 是**为了不出现伪起始码而插进来的**，
/// 留着它位读出的一切都是错的。
pub fn remove_emulation_prevention(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut zeros = 0u8;
    for &b in data {
        if zeros >= 2 && b == 0x03 {
            // 丢掉这个 0x03，并把它"消费"的零计数重置
            zeros = 0;
            continue;
        }
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
        out.push(b);
    }
    out
}

/// 施加防竞争字节（`0x00 0x00 0x00..0x03` → `0x00 0x00 0x03 0x00..0x03`）。
///
/// 只有"自己生成码流"时用得到（引擎里基本没有），但测试要用它构造样例。
pub fn add_emulation_prevention(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 8);
    let mut zeros = 0u8;
    for &b in data {
        if zeros >= 2 && b <= 0x03 {
            out.push(0x03);
            zeros = 0;
        }
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
        out.push(b);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_msb_first() {
        // 1011 0010
        let data = [0b1011_0010];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bit().unwrap(), 1);
        assert_eq!(r.read_bit().unwrap(), 0);
        assert_eq!(r.read_bits(3).unwrap(), 0b110);
        assert_eq!(r.read_bits(3).unwrap(), 0b010);
        assert_eq!(r.remaining_bits(), 0);
    }

    #[test]
    fn read_bits_spans_byte_boundary() {
        let data = [0xFF, 0x00];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(4).unwrap(), 0xF);
        assert_eq!(r.read_bits(8).unwrap(), 0xF0);
        assert_eq!(r.remaining_bits(), 4);
    }

    #[test]
    fn read_bits_zero_is_noop() {
        let data = [0xAA];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(0).unwrap(), 0);
        assert_eq!(r.position_bits(), 0);
    }

    #[test]
    fn read_bits_over_64_is_rejected() {
        let data = [0u8; 16];
        let mut r = BitReader::new(&data);
        assert!(matches!(r.read_bits(65), Err(MediaError::Bitstream(_))));
    }

    #[test]
    fn exhaustion_is_need_more() {
        let data = [0xFF];
        let mut r = BitReader::new(&data);
        r.read_bits(8).unwrap();
        let e = r.read_bit().unwrap_err();
        assert!(e.is_need_more(), "{e}");
    }

    #[test]
    fn ue_zero_is_single_one_bit() {
        // '1' → 0
        let data = [0b1000_0000];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_ue().unwrap(), 0);
        assert_eq!(r.position_bits(), 1);
    }

    #[test]
    fn ue_follows_the_spec_table() {
        // H.264 Table 9-1：1→0, 010→1, 011→2, 00100→3, 00101→4, 00110→5, 00111→6
        let cases: &[(&[u8], u32)] = &[
            (&[0b1000_0000], 0),
            (&[0b0100_0000], 1),
            (&[0b0110_0000], 2),
            (&[0b0010_0000], 3),
            (&[0b0010_1000], 4),
            (&[0b0011_0000], 5),
            (&[0b0011_1000], 6),
        ];
        for (bytes, want) in cases {
            let mut r = BitReader::new(bytes);
            assert_eq!(r.read_ue().unwrap(), *want, "样例 {bytes:?}");
        }
    }

    #[test]
    fn se_maps_signed_per_spec() {
        // 1→0, 010→1, 011→-1, 00100→2, 00101→-2
        let cases: &[(&[u8], i32)] = &[
            (&[0b1000_0000], 0),
            (&[0b0100_0000], 1),
            (&[0b0110_0000], -1),
            (&[0b0010_0000], 2),
            (&[0b0010_1000], -2),
        ];
        for (bytes, want) in cases {
            let mut r = BitReader::new(bytes);
            assert_eq!(r.read_se().unwrap(), *want, "样例 {bytes:?}");
        }
    }

    #[test]
    fn ue_rejects_all_zero_stream() {
        let data = [0u8; 8];
        let mut r = BitReader::new(&data);
        assert!(matches!(r.read_ue(), Err(MediaError::Bitstream(_))));
    }

    #[test]
    fn byte_align_discards_padding() {
        let data = [0b1010_0000, 0b1111_0000];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(3).unwrap(), 0b101);
        r.byte_align();
        assert_eq!(r.position_bits(), 8);
        assert_eq!(r.read_bits(4).unwrap(), 0b1111);
    }

    #[test]
    fn byte_align_on_boundary_is_noop() {
        let data = [0u8; 2];
        let mut r = BitReader::new(&data);
        r.read_bits(8).unwrap();
        r.byte_align();
        assert_eq!(r.position_bits(), 8);
    }

    #[test]
    fn emulation_prevention_roundtrip() {
        let clean = [0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x00, 0x00, 0x03];
        let escaped = add_emulation_prevention(&clean);
        // 每个危险点都要插一个 0x03
        assert_eq!(escaped.len(), clean.len() + 3);
        assert_eq!(remove_emulation_prevention(&escaped), clean);
    }

    #[test]
    fn emulation_prevention_removal_only_after_two_zeros() {
        // 单独的 0x03（前面不足两个零）必须保留
        let data = [0x01, 0x03, 0x00, 0x00];
        assert_eq!(remove_emulation_prevention(&data), data);
    }

    #[test]
    fn emulation_prevention_handles_long_zero_runs() {
        // 四个零 → 中间插一个 0x03；去掉后必须还原
        let clean = [0x00, 0x00, 0x00, 0x00, 0x40];
        let escaped = add_emulation_prevention(&clean);
        assert_eq!(remove_emulation_prevention(&escaped), clean);
    }

    #[test]
    fn sps_like_bitstream_reads_without_error() {
        // 手搓一个最小的"类 SPS"：profile_idc(8) + 一些 ue + 尾部对齐
        // 目的不是验语法，而是验"位读取器在真实形态上不 panic 且计数正确"
        let raw = [0x67u8, 0x42, 0x00, 0x1E];
        let clean = remove_emulation_prevention(&raw);
        let mut r = BitReader::new(&clean);
        assert_eq!(r.read_u32(8).unwrap(), 0x67);
        assert_eq!(r.read_u32(8).unwrap(), 0x42);
        assert_eq!(r.position_bits(), 16);
        r.byte_align();
        assert_eq!(r.position_bits(), 16);
    }
}