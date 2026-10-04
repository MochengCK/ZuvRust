//! MSB-first 位读取器：FLAC / MP3 / AAC 都是这种位序，所以放在音频模块里共用。
//!
//! 为什么单独一层而不是在解码器里各自 `>>`：这几个解码器的位读取都夹在
//! "热循环里读几个 bit"的位置（Rice 残差、Huffman 前缀），读错一位就是满盘
//! 皆错，而错误现场离原因很远。收在一处之后，边界行为（越界、对齐、剩下的
//! 位数）可以单测钉住。

use me_core::{MediaError, Result};

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

    /// 还剩多少位。
    pub fn remaining_bits(&self) -> usize {
        if self.pos >= self.data.len() {
            return 0;
        }
        (self.data.len() - self.pos) * 8 - self.bit as usize
    }

    /// 读 `n`（≤32）位，返回无符号值。
    pub fn read(&mut self, n: u8) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(MediaError::Bitstream(format!("一次最多读 32 位（要 {n}）")));
        }
        if self.remaining_bits() < n as usize {
            return Err(MediaError::NeedMore { need: n as usize, have: self.remaining_bits() });
        }
        let mut out: u32 = 0;
        let mut left = n;
        while left > 0 {
            let cur = self.data[self.pos];
            let avail = 8 - self.bit;
            let take = avail.min(left);
            // 取当前字节里从 bit 开始的 take 位（大端序）
            let shift = avail - take;
            let mask = if take == 8 { 0xFF } else { ((1u16 << take) - 1) as u8 };
            let bits = (cur >> shift) & mask;
            out = (out << take) | bits as u32;
            self.bit += take;
            if self.bit == 8 {
                self.bit = 0;
                self.pos += 1;
            }
            left -= take;
        }
        Ok(out)
    }

    /// 读 `n`（≤32）位并按**二进制补码**解释为有符号。
    pub fn read_signed(&mut self, n: u8) -> Result<i32> {
        let v = self.read(n)?;
        if n == 0 {
            return Ok(0);
        }
        // 符号扩展
        let shift = 32 - n as u32;
        Ok(((v << shift) as i32) >> shift)
    }

    /// 读一元码：数"0 的个数直到遇到 1"（FLAC 的 Rice 商、UTF-8 前导都用它）。
    ///
    /// 这里**故意不设"长度上限"**：曾经按 `count > 剩余位数` 做过防疯判断，结果
    /// 误伤了合法码流 —— 一元码越长，读的过程中剩余位数掉得越快，判断式在
    /// "码本身还没读完、但已经读了超过一半数据"时就会成立。实测现场：FLAC 96 kHz
    /// 尾帧最后一个残差的商是 123（需要 124 位），读到时只剩 145 位，
    /// 读到第 74 个 0 就被判成"数据损坏"—— 位位置全对，帧尾 CRC-16 也能对上，
    /// 纯粹是这个上限算错了。真正需要保护的是**不会死循环**：位读空时下面的
    /// `read(1)` 会返回 `NeedMore`，循环必然结束，且消耗的位数不可能超过缓冲区大小。
    pub fn read_unary(&mut self) -> Result<u32> {
        let mut count: usize = 0;
        while self.read(1)? == 0 {
            count += 1;
        }
        // 一元码不可能比缓冲区还长（上面的循环最多消耗 total_bits 位），
        // 这里的转换错误只在 >4 Gbit 的输入上才可能出现，纯属兜底。
        u32::try_from(count)
            .map_err(|_| MediaError::Bitstream("一元码长度超出 32 位（输入异常）".into()))
    }

    /// 已经消耗的**位数**（相对数据起点）。调试/追踪用。
    pub fn pos_bits(&self) -> usize {
        self.pos * 8 + self.bit as usize
    }

    /// 数据总位数（判断"这个包有多少位可用"）。
    pub fn total_bits(&self) -> usize {
        self.data.len() * 8
    }

    /// 已经完整读过的字节数（CRC 校验要按"到某个字段为止的字节"算）。
    ///
    /// 语义：当前字节里已经读过 0 位 → 就是 `pos`；读过若干位 → 那个字节算"遇到过"，
    /// 返回 `pos + 1`（CRC 只在**字节对齐**的边界上算，所以这正是调用方要的值）。
    pub fn byte_pos(&self) -> usize {
        if self.bit == 0 {
            self.pos
        } else {
            self.pos + 1
        }
    }

    /// 对齐到字节边界（FLAC 的帧尾 CRC 之前要做）。
    pub fn align_to_byte(&mut self) {
        if self.bit != 0 {
            self.bit = 0;
            self.pos += 1;
        }
    }

    /// 把读指针挪到**绝对位偏移** `bits`（可前可后，允许超出当前数据末尾）。
    ///
    /// 为什么需要"能往后也能往前"：MP3 里每一对 (粒度, 声道) 在码流中占**固定长度**的一段
    /// （长度写在侧信息里），解完必须回到这一段末尾 —— 哈夫曼解码可能提前收工（数据不足），
    /// 也可能读过头（最后一组要丢掉）。位置不归位，**后面所有数据都会错位**。
    /// 只提供"读"的位读取器平时用不到这种操作，所以这条是给"按段切分"的格式准备的。
    pub fn seek_bits(&mut self, bits: usize) {
        self.pos = bits / 8;
        self.bit = (bits % 8) as u8;
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_bits_msb_first() {
        let data = [0b1011_0010u8, 0b0101_1111];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read(1).unwrap(), 1);
        assert_eq!(r.read(3).unwrap(), 0b011);
        assert_eq!(r.read(4).unwrap(), 0b0010, "跨过字节边界也要对");
        assert_eq!(r.read(8).unwrap(), 0b0101_1111);
        assert_eq!(r.remaining_bits(), 0);
    }

    #[test]
    fn signed_read_sign_extends() {
        let data = [0b1111_1111u8, 0b1000_0000];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_signed(4).unwrap(), -1, "1111 → -1");
        assert_eq!(r.read_signed(4).unwrap(), -1, "同一个字节的低半字节还是 1111");
        assert_eq!(r.read_signed(4).unwrap(), -8, "1000 → -8");
    }

    #[test]
    fn unary_counts_zeros_until_one() {
        // 0001 0000 → 3 个 0 然后 1
        let data = [0b0001_0000u8];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_unary().unwrap(), 3);
        assert_eq!(r.read(3).unwrap(), 0);
    }

    #[test]
    fn unary_long_code_survives_small_remainder() {
        // 回归：FLAC 96 kHz 尾帧最后一个残差的商是 123（124 位的一元码），
        // 读到时只剩 145 位。曾经的"count > 剩余位数 + 1"防疯判断会在第 74 个 0
        // 上误报"一元码过长（数据损坏）"，位位置其实全对。
        let mut data = [0u8; 16];
        data[0] = 0b1010_0000; // 先读 4 位，再开始一元码
        data[15] = 0b0000_0001; // 123 个 0 之后收尾的 1
        let mut r = BitReader::new(&data);
        assert_eq!(r.read(4).unwrap(), 0b1010);
        assert_eq!(r.read_unary().unwrap(), 123, "长一元码不能用剩余位数当上限");
        assert_eq!(r.remaining_bits(), 0, "正好读到末尾");
    }

    #[test]
    fn unary_truncated_reports_need_more() {
        // 全是 0：读到末尾必须给 NeedMore（调用方据此判"数据不足"），
        // 而不是编一个"码流过长"的结论 —— 二者对上层是"截断"与"损坏"的区别。
        let data = [0u8; 4];
        let mut r = BitReader::new(&data);
        assert!(matches!(r.read_unary(), Err(MediaError::NeedMore { .. })));
    }

    #[test]
    fn runs_out_as_need_more_not_garbage() {
        let data = [0xAB];
        let mut r = BitReader::new(&data);
        assert!(r.read(8).is_ok());
        assert!(matches!(r.read(1), Err(MediaError::NeedMore { .. })), "读空了要给 NeedMore");
    }

    #[test]
    fn aligns_to_byte() {
        let data = [0b1011_0010u8, 0xFF];
        let mut r = BitReader::new(&data);
        r.read(3).unwrap();
        r.align_to_byte();
        assert_eq!(r.read(8).unwrap(), 0xFF, "对齐后从下一个字节开始");
    }

    #[test]
    fn seeks_forward_and_backward() {
        let data = [0b1011_0010u8, 0b0101_1111, 0b1111_0000];
        let mut r = BitReader::new(&data);
        r.seek_bits(5);
        assert_eq!(r.pos_bits(), 5);
        assert_eq!(r.read(3).unwrap(), 0b010, "从第 5 位接着读");
        r.seek_bits(8);
        assert_eq!(r.read(4).unwrap(), 0b0101);
        r.seek_bits(4);
        assert_eq!(r.remaining_bits(), 20, "归位后剩余位数按新位置算");
        assert_eq!(r.read(4).unwrap(), 0b0010, "往回挪也要能读对");
    }
}
