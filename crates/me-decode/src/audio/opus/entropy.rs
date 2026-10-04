//! Opus 的熵层：**区间解码器**（range decoder）与帧尾的**原始比特**。
//!
//! 全部照 RFC 6716 §4.1 实现。规范里的两个约束决定了这个文件的写法：
//!
//! - **所有运算必须位精确**（整数、不允许任何浮点或近似）——因为 CELT 的比特分配
//!   依赖 `ec_tell()` 的**上界**，编码器与解码器算出来的值只要差一位，后面的分配
//!   就会跟着错，最后解出一堆噪声（症状很难查）。所以这里逐行对着规范写。
//! - **区间解码器会"读到原始比特里去"**（§4.1.2.1）：解码过程中它可能已经读掉
//!   帧尾几个字节，而这些字节**同时**还要作为原始比特返回。这不是 bug，是格式设计。
//!
//! ## 与参考实现的对应关系（方便对拍）
//!
//! | 本文件 | 参考实现 | 规范 |
//! |---|---|---|
//! | [`RangeDecoder::new`] | `ec_dec_init` | §4.1.1 |
//! | [`RangeDecoder::decode`] + [`RangeDecoder::update`] | `ec_decode` / `ec_dec_update` | §4.1.2 |
//! | [`RangeDecoder::decode_bin`] | `ec_decode_bin` | §4.1.3.1 |
//! | [`RangeDecoder::dec_bit_logp`] | `ec_dec_bit_logp` | §4.1.3.2 |
//! | [`RangeDecoder::dec_icdf`] | `ec_dec_icdf` | §4.1.3.3 |
//! | [`RangeDecoder::dec_bits`] | `ec_dec_bits` | §4.1.4 |
//! | [`RangeDecoder::dec_uint`] | `ec_dec_uint` | §4.1.5 |
//! | [`RangeDecoder::tell`] / [`RangeDecoder::tell_frac`] | `ec_tell` / `ec_tell_frac` | §4.1.6 |

use me_core::{MediaError, Result};

/// 区间**大小**的下限：规范化要保证 `rng > 2^23`（RFC §4.1.2.1）。
const RNG_BOT: u32 = 1 << 23;

/// 区间"差值"的上限掩码（`val` 保持在 31 位内）。
const VAL_TOP_MASK: u32 = 0x7FFF_FFFF;

/// 区间解码器。
///
/// 生命周期：一个 Opus 包（帧）一个实例 —— 区间解码器的状态**不跨帧**，
/// 每帧都从 [`RangeDecoder::new`] 重新初始化（这也是丢包不会永久传播的原因）。
pub struct RangeDecoder<'a> {
    buf: &'a [u8],
    /// 下一个要读的字节下标。
    offs: usize,
    /// 上一个字节里"留给下一次规范化"的那一位（0 或 1）。
    rem: u32,
    /// 区间差值 `val`。
    val: u32,
    /// 区间大小 `rng`。
    rng: u32,
    /// 已经处理过的整位数（规范里的 `nbits_total`）。
    nbits_total: i32,
    /// 上一次 [`RangeDecoder::decode`] 算出的 `rng/ft`（[`RangeDecoder::update`] 要用）。
    ext: u32,
    /// 帧尾原始比特：已经从末尾往回读了多少字节。
    end_offs: u32,
    /// 原始比特的位窗口。
    end_window: u32,
    /// 窗口里还有多少位可用。
    nend_bits: u32,
    /// 从帧尾读原始比特时是否越过了帧首（见 [`RangeDecoder::over_read`]）。
    end_overrun: bool,
    /// 解均匀整数时是否夹过值（见 [`RangeDecoder::clamped_uint`]）。
    clamped: bool,
    /// 从**尾部**扣掉的字节数（参考实现的 `dec->storage -= n`）。
    ///
    /// 只有一处会用到：Opus 的"冗余帧"。SILK/CELT 主体数据在**前**、5 ms 的 CELT 冗余帧
    /// 贴在**后**，参考实现解完主体后把 `storage` 往前挪 `redundancy_bytes`，
    /// 于是"帧尾原始比特"再也不会读进冗余帧里、`len_bytes()` 也随之变小
    /// （CELT 的 `total_bits` 就是这么算出来的）。
    tail_limit: usize,
}

impl<'a> RangeDecoder<'a> {
    /// 初始化（RFC §4.1.1）。
    ///
    /// `rng` 从 128 起、`val` 取第一个字节的高 7 位、最低位留作下一次规范化的高比特；
    /// 然后立刻规范化一次（把 `rng` 拉过 `2^23`）。
    pub fn new(buf: &'a [u8]) -> Self {
        let b0 = buf.first().copied().unwrap_or(0) as u32;
        let mut d = Self {
            buf,
            // 第一个字节已经"读过"了（源里没有字节时 `read_byte` 不推进）
            offs: usize::from(!buf.is_empty()),
            rem: b0 & 1,
            val: 127 - (b0 >> 1),
            rng: 128,
            // 规范：初值 9（等价于"首次规范化之后是 33"）
            nbits_total: 9,
            ext: 0,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            end_overrun: false,
            clamped: false,
            tail_limit: 0,
        };
        d.normalize();
        d
    }

    /// 从尾部扣掉 `n` 个字节（参考实现的 `dec->storage -= n`）。
    ///
    /// 用于 Opus 的冗余帧：主体数据在前、冗余帧贴在后，扣掉之后 CELT 既读不到冗余帧、
    /// `len_bytes()` 也变成"主体的字节数"。
    pub fn truncate_tail(&mut self, n: usize) {
        debug_assert!(n <= self.buf.len());
        self.tail_limit = self.tail_limit.saturating_add(n).min(self.buf.len());
    }

    /// 取下一个输入字节；越界返回 0（规范：数据读完后一律当 0，且**不能**推进位置）。
    fn read_byte(&mut self) -> u32 {
        if self.offs < self.buf.len() {
            let b = self.buf[self.offs] as u32;
            self.offs += 1;
            b
        } else {
            0
        }
    }

    /// 从帧尾取下一个字节（原始比特是**从末尾往回**排的）。
    ///
    /// 超出帧首时**返回 0**（与参考实现一致：`ec_read_byte_from_end` 直接给 0），
    /// 同时把 `over_read` 标志立起来 —— 是不是"码流损坏"由上层决定
    /// （比如整帧的原始比特都读完了还在读，那才是坏；而"空载荷"这种退化情形
    /// 在规范里就是允许的：编码器一个字节都不写，解码器读 0）。
    fn read_byte_from_end(&mut self) -> u32 {
        let len = self.buf.len() - self.tail_limit;
        if (self.end_offs as usize) < len {
            self.end_offs += 1;
            let idx = len - self.end_offs as usize;
            self.buf[idx] as u32
        } else {
            self.end_overrun = true;
            0
        }
    }

    /// 规范化（RFC §4.1.2.1）：把 `rng` 拉回 `> 2^23`，同时从码流里补 `val`。
    fn normalize(&mut self) {
        while self.rng <= RNG_BOT {
            self.nbits_total += 8;
            self.rng <<= 8;
            let new_byte = self.read_byte();
            // sym 的高位是"上一个字节留下的那一位"，其余 7 位是新字节的高 7 位
            let sym = (self.rem << 7) | (new_byte >> 1);
            self.rem = new_byte & 1;
            // val = ((val<<8) + (255 - sym)) & 0x7fffffff（用 u64 算避免中间溢出）
            let v = ((self.val as u64) << 8) + (255 - sym) as u64;
            self.val = (v & VAL_TOP_MASK as u64) as u32;
        }
    }

    /// 第一步：求 `fs`（落在某个符号的频段里），随后必须调 [`RangeDecoder::update`]。
    ///
    /// `fs = ft - min(val / (rng/ft) + 1, ft)`（RFC §4.1.2）。
    /// 要求 `ft <= 2^16`（`fs` 就是 16 位量）且 `ft <= rng`（否则 `rng/ft` 为 0）——
    /// 这两条是规范对调用方的约束，给错了应当当场暴露，而不是静默算错。
    pub fn decode(&mut self, ft: u32) -> u32 {
        debug_assert!(ft > 0, "ft 必须 > 0");
        debug_assert!(ft <= (1 << 16), "ft 必须 <= 2^16（fs 是 16 位量）");
        debug_assert!(ft <= self.rng, "ft 不能大于当前区间");
        self.ext = self.rng / ft;
        let s = self.val / self.ext;
        ft - (s + 1).min(ft)
    }

    /// 第一步的变体：`ft = 1<<bits`（少一次除法，RFC §4.1.3.1）。
    pub fn decode_bin(&mut self, bits: u32) -> u32 {
        debug_assert!(bits <= 16, "ftb 不能超过 16（fs 是 16 位量）");
        self.ext = self.rng >> bits;
        let s = self.val / self.ext;
        let ft = 1u32 << bits;
        ft - (s + 1).min(ft)
    }

    /// 第二步：用符号 `k` 的三元组 `(fl, fh, ft)` 更新区间（RFC §4.1.2）。
    pub fn update(&mut self, fl: u32, fh: u32, ft: u32) {
        debug_assert!(fl < fh && fh <= ft, "(fl, fh, ft) 必须满足 fl < fh <= ft");
        let s = self.ext * (ft - fh);
        debug_assert!(s <= self.val, "更新前 val 必须 >= rng/ft*(ft-fh)");
        self.val -= s;
        // 符号 0 走另一条路：这样有限精度的截断误差全落在符号 0 上（规范刻意如此）
        self.rng = if fl > 0 { self.ext * (fh - fl) } else { self.rng - s };
        self.normalize();
    }

    /// 解一个二元符号：`logp` 是"1"的概率的以 2 为底对数的绝对值（RFC §4.1.3.2）。
    ///
    /// 等价于 `decode(1<<logp)` + `update`，但不做乘除法。
    pub fn dec_bit_logp(&mut self, logp: u32) -> u32 {
        let r = self.rng;
        let d = self.val;
        let s = r >> logp;
        let ret = u32::from(d < s);
        if ret == 0 {
            self.val = d - s;
        }
        self.rng = if ret != 0 { s } else { r - s };
        self.normalize();
        ret
    }

    /// 用**逆累积分布表**解一个符号（RFC §4.1.3.3）。
    ///
    /// `icdf[k] = (1<<ftb) - fh[k]`，表尾必须是 0。返回符号下标。
    /// 表不合法（没有 0 收尾／越界）时报错 —— 与其读越界，不如当场说这帧坏了。
    pub fn dec_icdf(&mut self, icdf: &[u8], ftb: u32) -> Result<u32> {
        let d = self.val;
        let mut s = self.rng;
        let r = s >> ftb;
        let mut ret: i32 = -1;
        loop {
            ret += 1;
            let Some(&entry) = icdf.get(ret as usize) else {
                return Err(MediaError::Bitstream(
                    "Opus: icdf 表没有以 0 收尾（表本身不合法或码流损坏）".into(),
                ));
            };
            let t = s;
            s = r * entry as u32;
            if d >= s {
                self.val = d - s;
                self.rng = t - s;
                break;
            }
        }
        self.normalize();
        Ok(ret as u32)
    }

    /// 读 `bits` 个**原始比特**（从帧尾、LSB 优先，RFC §4.1.4）。
    ///
    /// 它与区间解码器读的是同一段字节：区间解码器可能已经"读过头"进到原始比特区，
    /// 这是格式允许的，两边各自记自己的位置。读过了帧首一律当 0（见
    /// [`RangeDecoder::over_read`]）。
    pub fn dec_bits(&mut self, bits: u32) -> Result<u32> {
        debug_assert!(bits <= 25, "参考实现一次最多取 25 位");
        // 窗口里不够就先从帧尾补字节
        while self.nend_bits < bits {
            let b = self.read_byte_from_end();
            self.end_window |= b << self.nend_bits;
            self.nend_bits += 8;
        }
        let mask = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
        let ret = self.end_window & mask;
        self.end_window >>= bits;
        self.nend_bits -= bits;
        self.nbits_total += bits as i32;
        Ok(ret)
    }

    /// 解一个"均匀分布"的整数，取值 `[0, ft)`（RFC §4.1.5）。`ft` 可以大到 `2^32-1`。
    ///
    /// 高 8 位以内走区间解码、其余低位走原始比特。若解出来的值 `>= ft`，规范说
    /// "当前帧已损坏，应当**饱和到 ft-1** 或走 PLC" —— 参考实现就是返回 `ft-1`
    /// 并把 error 标志立起来。这里照做（返回夹住的值 + [`RangeDecoder::clamped_uint`]），
    /// 而不是当场报错：**大 ft（超过 2^32 回绕）的合法码流也会走到这条路**。
    pub fn dec_uint(&mut self, ft: u32) -> Result<u32> {
        debug_assert!(ft > 1, "ft 必须 > 1");
        let ft = ft - 1;
        let ftb = ilog(ft);
        if ftb > 8 {
            let ftb = ftb - 8;
            let ft_hi = (ft >> ftb) + 1;
            let s = self.decode(ft_hi);
            self.update(s, s + 1, ft_hi);
            let t = (s << ftb) | self.dec_bits(ftb)?;
            if t <= ft {
                Ok(t)
            } else {
                self.clamped = true;
                Ok(ft)
            }
        } else {
            let ft_hi = ft + 1;
            let s = self.decode(ft_hi);
            self.update(s, s + 1, ft_hi);
            Ok(s)
        }
    }

    /// 当前用掉的**整位**数（上界，RFC §4.1.6.1）。
    ///
    /// 比特分配必须与编码器算出完全一样的值，所以这条不能"差不多"。
    pub fn tell(&self) -> i32 {
        self.nbits_total - ilog(self.rng) as i32
    }

    /// 当前用掉的位数，**1/8 位精度**（RFC §4.1.6.2）。
    ///
    /// 保证与 `tell()` 的关系：`tell() == ceil(tell_frac() / 8.0)`。
    pub fn tell_frac(&self) -> u32 {
        let mut lg = ilog(self.rng) as i32;
        let mut r_q15 = self.rng >> (lg - 16);
        for _ in 0..3 {
            r_q15 = (r_q15 * r_q15) >> 15;
            lg = lg * 2 + (r_q15 >> 16) as i32;
            if r_q15 >> 16 == 1 {
                r_q15 >>= 1;
            }
        }
        (self.nbits_total * 8 - lg) as u32
    }

    /// 当前区间状态（`(val, rng)`），调试/对拍用。
    pub fn state(&self) -> (u32, u32) {
        (self.val, self.rng)
    }

    /// 这个包一共有多少字节（"还剩多少位"的判断要用它）。
    pub fn len_bytes(&self) -> usize {
        self.buf.len() - self.tail_limit
    }

    /// 把 `tell()` 直接抬到 `bits`（静音帧的处理：假装整帧都读过了）。
    ///
    /// 不能只改调用方的局部变量 —— `tell()` 是后面所有"还剩多少位"判断的依据，
    /// 而静音帧后面那些解码步骤（能量、TF、spread）都要看到"位已经用完"。
    pub fn force_tell(&mut self, bits: i32) {
        self.nbits_total += bits - self.tell();
    }

    /// 从帧尾往回读时是否越过了帧首（真越界说明这个帧的原始比特比帧里实际有的多）。
    pub fn over_read(&self) -> bool {
        self.end_overrun
    }

    /// 解均匀整数时是否解出了 `>= ft` 的值（已按规范夹到 `ft-1`）。
    pub fn clamped_uint(&self) -> bool {
        self.clamped
    }

    /// 帧尾还剩多少字节没被原始比特读过（调试用）。
    pub fn end_bytes_read(&self) -> u32 {
        self.end_offs
    }
}

/// `ilog(x)`：`x` 的二进制位宽（`x=0` 时按 0 计，与参考实现的 `EC_ILOG` 一致）。
fn ilog(x: u32) -> u32 {
    32 - x.leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 复用同一段字节、但各自独立地"解同一个符号"的辅助：
    /// 朴素路径 = `decode()` + `update()`（规范 §4.1.2 的定义）。
    fn naive_decode(d: &mut RangeDecoder, fl: &[u32], ft: u32) -> u32 {
        let fs = d.decode(ft);
        // 找出 fs 落在哪个符号的 [fl[k], fl[k+1]) 里
        let mut k = 0usize;
        while k + 1 < fl.len() && fs >= fl[k + 1] {
            k += 1;
        }
        let fh = fl[k + 1];
        d.update(fl[k], fh, ft);
        k as u32
    }

    #[test]
    fn fresh_decoder_reports_one_bit_used() {
        // 规范 §4.1.6.1：新初始化的解码器（还没解任何符号）报"用了 1 位"
        for data in [&[][..], &[0x00], &[0xFF], &[0x12, 0x34, 0x56]] {
            let d = RangeDecoder::new(data);
            assert_eq!(d.tell(), 1, "初始化后应当报 1 位（数据 {data:?}）");
            assert_eq!(d.tell_frac(), 8, "1/8 位精度下就是 8");
            assert!(d.state().1 > RNG_BOT, "规范化后 rng 必须 > 2^23");
        }
    }

    #[test]
    fn bit_logp_is_equivalent_to_decode_plus_update() {
        // 规范 §4.1.3.2 声称 ec_dec_bit_logp 与 decode(1<<logp)+update 等价 —— 逐个对
        let bytes: Vec<u8> = (0..64u16).map(|i| (i * 37 % 251) as u8).collect();
        for logp in [1u32, 2, 4, 8, 15] {
            let ft = 1u32 << logp;
            let fl = [0u32, ft - 1, ft];
            let mut a = RangeDecoder::new(&bytes);
            let mut b = RangeDecoder::new(&bytes);
            for _ in 0..40 {
                let bit_fast = a.dec_bit_logp(logp);
                // 朴素路径：fs < ft-1 就是 0，否则是 1
                let fs = b.decode(ft);
                let bit_naive = u32::from(fs >= ft - 1);
                b.update(if bit_naive == 0 { fl[0] } else { fl[1] },
                         if bit_naive == 0 { fl[1] } else { fl[2] }, ft);
                assert_eq!(bit_fast, bit_naive, "logp={logp} 时两条路不一致");
                assert_eq!(a.state(), b.state(), "状态也要一致");
                assert_eq!(a.tell_frac(), b.tell_frac());
            }
        }
    }

    #[test]
    fn icdf_is_equivalent_to_decode_plus_update() {
        // 均匀 4 符号 {4,4,4,4}/16 → icdf = [12, 8, 4, 0]
        let icdf = [12u8, 8, 4, 0];
        let fl = [0u32, 4, 8, 12, 16];
        let bytes: Vec<u8> = (0..128u16).map(|i| ((i * 91 + 7) % 256) as u8).collect();
        let mut a = RangeDecoder::new(&bytes);
        let mut b = RangeDecoder::new(&bytes);
        let mut seen = [0usize; 4];
        for _ in 0..60 {
            let sym = a.dec_icdf(&icdf, 4).unwrap();
            let sym_naive = naive_decode(&mut b, &fl, 16);
            assert_eq!(sym, sym_naive);
            assert_eq!(a.state(), b.state());
            seen[sym as usize] += 1;
        }
        // 均匀表下四个符号都该出现过（说明表确实被用起来了，而不是全落在 0 上）
        assert!(seen.iter().all(|&c| c > 0), "四个符号都应出现：{seen:?}");
    }

    #[test]
    fn icdf_table_without_zero_terminator_is_reported() {
        let bytes = [0x5Au8; 16];
        let mut d = RangeDecoder::new(&bytes);
        // 三项都是 255 → 每一步的 s 都远大于 val，循环一直走到表尾：
        // 没有 0 收尾就是表不合法，必须报错（而不是越界读或 panic）
        let err = d.dec_icdf(&[255, 255, 255], 8).unwrap_err();
        assert!(format!("{err}").contains("icdf"), "错误信息要说清原因：{err}");
    }

    #[test]
    fn decode_bin_matches_decode_with_power_of_two_ft() {
        let bytes: Vec<u8> = (0..64u16).map(|i| ((i * 53 + 11) % 256) as u8).collect();
        for bits in [1u32, 3, 8] {
            let mut a = RangeDecoder::new(&bytes);
            let mut b = RangeDecoder::new(&bytes);
            for _ in 0..40 {
                let fs_bin = a.decode_bin(bits);
                let fs = b.decode(1 << bits);
                assert_eq!(fs_bin, fs, "bits={bits}");
                // 用同一个 (fl,fh,ft) 更新，状态应当仍然一致
                a.update(fs_bin, fs_bin + 1, 1 << bits);
                b.update(fs, fs + 1, 1 << bits);
                assert_eq!(a.state(), b.state());
            }
        }
    }

    #[test]
    fn raw_bits_come_from_the_tail_lsb_first() {
        // 帧尾两字节 0xA5 0x03：**从最后一个字节的最低位往上填**（RFC §4.1.4）
        // → 先 5 位 = 0x03 的低 5 位 = 0b00011 = 3，再 3 位 = 高 3 位 = 0，
        // 再 8 位就轮到倒数第二个字节 0xA5。
        let data = [0x00u8, 0x00, 0x00, 0xA5, 0x03];
        let mut d = RangeDecoder::new(&data);
        assert_eq!(d.dec_bits(5).unwrap(), 0b00011, "末字节的低 5 位");
        assert_eq!(d.dec_bits(3).unwrap(), 0b000, "同一个字节的高 3 位");
        assert_eq!(d.dec_bits(8).unwrap(), 0xA5, "再往前一个字节");
        assert_eq!(d.end_bytes_read(), 2);
    }

    #[test]
    fn raw_bits_span_more_than_one_byte_lsb_first() {
        // 12 位：末字节全 8 位 + 倒数第二字节的低 4 位
        let data = [0x00u8, 0xF0, 0x34];
        let mut d = RangeDecoder::new(&data);
        assert_eq!(d.dec_bits(12).unwrap(), 0x034, "0x34 是低 8 位，再接 0xF0 的低 4 位 0");
        assert_eq!(d.dec_bits(4).unwrap(), 0xF, "0xF0 的高 4 位");
    }

    #[test]
    fn reading_more_raw_bits_than_the_frame_has_gives_zeros_and_a_flag() {
        // 参考实现在越界时返回 0（规范允许"空载荷"这种退化情形），并把 error 标志立起来；
        // 这里同样：给 0 + `over_read()` 为真，由上层决定要不要当损坏处理。
        let data = [0x11u8];
        let mut d = RangeDecoder::new(&data);
        assert_eq!(d.dec_bits(8).unwrap(), 0x11, "帧里唯一那个字节");
        assert!(!d.over_read(), "还在帧内不该立标志");
        assert_eq!(d.dec_bits(8).unwrap(), 0x00, "越过帧首之后一律给 0");
        assert!(d.over_read(), "越过帧首后标志要立起来");
        // 空载荷：一个字节都没有，读到的全是 0
        let mut empty = RangeDecoder::new(&[]);
        assert_eq!(empty.dec_bits(5).unwrap(), 0);
        assert!(empty.over_read());
    }

    #[test]
    fn uniform_ints_small_and_large() {
        let bytes: Vec<u8> = (0..512u16).map(|i| ((i * 31 + 5) % 256) as u8).collect();
        // 小 ft：只走区间解码
        let mut d = RangeDecoder::new(&bytes);
        for _ in 0..50 {
            assert!(d.dec_uint(3).unwrap() < 3);
        }
        // 大 ft：高位走区间解码 + 低位走原始比特
        let mut d = RangeDecoder::new(&bytes);
        let mut max = 0;
        for _ in 0..50 {
            let v = d.dec_uint(20_000).unwrap();
            assert!(v < 20_000);
            max = max.max(v);
        }
        assert!(max > 1_000, "大范围下应当取到明显大于 0 的值（拿到的最大 {max}）");
    }

    #[test]
    fn uniform_int_out_of_range_is_clamped_and_flagged() {
        // 全 0xFF 的帧尾会让"低位原始比特"全 1 → 很容易解出 >= ft 的值。
        // 规范要求此时**饱和**（参考实现返回 ft-1 并置 error 标志），这里照做。
        let bytes = [0xFFu8; 32];
        let mut d = RangeDecoder::new(&bytes);
        let mut clamped = 0;
        for _ in 0..40 {
            let v = d.dec_uint(257).unwrap();
            assert!(v < 257, "夹过之后必须 < ft（拿到 {v}）");
            if d.clamped_uint() {
                clamped += 1;
                break;
            }
        }
        assert!(clamped > 0, "这种构造下应当出现夹值");
    }

    #[test]
    fn tell_frac_and_tell_stay_consistent() {
        // 规范保证 tell() == ceil(tell_frac()/8)
        let bytes: Vec<u8> = (0..256u16).map(|i| ((i * 97 + 3) % 256) as u8).collect();
        let mut d = RangeDecoder::new(&bytes);
        let icdf = [100u8, 50, 20, 0];
        for i in 0..100 {
            let _ = d.dec_icdf(&icdf, 8).unwrap();
            let (t, tf) = (d.tell(), d.tell_frac());
            assert_eq!(t, (tf as i32 + 7) / 8, "第 {i} 步：tell={t} tell_frac={tf}");
            assert!(t >= 1);
        }
    }

    #[test]
    fn tell_grows_with_data_consumed() {
        // 8 个 0xFF 的输入：每规范化一次 tell 应当 +8
        let data = [0xFFu8; 16];
        let mut d = RangeDecoder::new(&data);
        let start = d.tell();
        let mut last = start;
        for _ in 0..5 {
            let _ = d.dec_bit_logp(1); // logp=1 → 每次几乎都消耗 1 位
            let t = d.tell();
            assert!(t >= last, "tell 只能单调不减（{last} → {t}）");
            last = t;
        }
        assert!(last > start, "解了几个符号之后 tell 应当增长");
    }

    #[test]
    fn silence_flag_of_a_real_opus_packet_style_frame() {
        // 真实包里第一个符号就是 CELT 的 silence 标志（{32767,1}/32768 = logp 15）。
        // 这里只保证"能解出 0/1 且状态合法"；真实包上的对拍在 tests/opus_entropy_ref.rs。
        for byte in [0x00u8, 0x01, 0x7F, 0x80, 0xFE, 0xFF] {
            let data = [byte, 0xAA, 0x55, 0x33];
            let mut d = RangeDecoder::new(&data);
            let bit = d.dec_bit_logp(15);
            assert!(bit <= 1);
            assert!(d.state().1 > RNG_BOT);
            assert!(d.tell() >= 1);
        }
    }
}
