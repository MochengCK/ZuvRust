//! **CABAC 算术解码器**（§9.3.1）与上下文初始化（§9.3.1.1）。
//!
//! ## 这一层只做"读 bin"，不懂语法
//!
//! 宏块层要的东西就四种：
//!
//! | 接口 | 规范里的名字 | 用在哪 |
//! |---|---|---|
//! | [`Cabac::decision`] | `decode_decision` | 绝大多数语法元素（带上下文）|
//! | [`Cabac::bypass`] | `decode_bypass` | 符号位、后缀、`coeff_abs_level_minus1` 的逃逸段 |
//! | [`Cabac::terminate`] | `decode_terminate` | `end_of_slice_flag`、`I_PCM` 的判定 |
//! | [`Cabac::ue_bypass`] | 旁路 Exp-Golomb | 少数 `ue(v)`（如 `mb_qp_delta` 的逃逸）|
//!
//! 状态机（`pStateIdx` / `valMPS`）与两张常量表（`rangeTabLPS`、状态转移）
//! 在 [`super::cabac_tables`]，那份是**按规范表生成**的。
//!
//! ## 位读法
//!
//! 规范里的算术解码器是"逐位"的：初始化读 9 位 `codIOffset`，之后每次
//! 重归一化读 1 位。这里就老老实实按位读 —— 解码器的吞吐瓶颈在反变换和
//! 运动补偿，位读这点开销不值得为它引入 32 位缓冲的边界 bug。
//!
//! ⚠️ **`cabac_alignment_one_bit` 必须校验**：切片数据起点要先对齐到字节，
//! 中间那些位规范写死是 `1`。不校验的话，`slice_data_bit_offset` 算错一位
//! 也能"解出东西来"，只是从第一个宏块起就是错的 —— 这类错最难查。

use me_core::{MediaError, Result};

use super::cabac_tables::{
    INIT_I, INIT_PB, NEXT_LPS, NEXT_MPS, RANGE_TAB_LPS, SWITCH_LPS,
};

/// 一个上下文的状态（§9.3.1.1 的 `pStateIdx` 与 `valMPS`）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Ctx {
    /// 0..63
    pub pstate: u8,
    /// 0 / 1
    pub mps: u8,
}

/// 上下文总数（`ctxIdx` 0..1023）。
pub const NUM_CTX: usize = 1024;

/// 按 §9.3.1.1 初始化一整组上下文。
///
/// `slice_qp_y` 是 `26 + pic_init_qp_minus26 + slice_qp_delta`（**不是**某个宏块
/// 的 QPY —— 宏块级的 `mb_qp_delta` 不参与初始化）。`idc` 是 `cabac_init_idc`
/// （I 切片不用）。
pub fn init_contexts(slice_qp_y: i32, idc: u32, is_i: bool) -> [Ctx; NUM_CTX] {
    let table: &[[i8; 2]; NUM_CTX] = if is_i {
        &INIT_I
    } else {
        &INIT_PB[(idc as usize).min(2)]
    };
    // §9.3.1.1：`SliceQPY` 取 0..51；`preCtxState = Clip3(1, 126, ((m * SliceQPY) >> 4) + n)`
    //
    // ⚠️ 是 **`m * SliceQPY` 再右移 4 位**，不是 `m * (SliceQPY − 26)` 右移 8 位。
    // 两者在 QP=26 时才碰巧接近，别的 QP 上差得很远：写错的表现是"前几个 bin 看着
    // 挺对，越往后越错，最后解不到 end_of_slice_flag"，极难从报错反推。
    let qp = slice_qp_y.clamp(0, 51);
    let mut out = [Ctx::default(); NUM_CTX];
    for (slot, &[m, n]) in out.iter_mut().zip(table.iter()) {
        let pre = (((m as i32) * qp) >> 4) + n as i32;
        let pre = pre.clamp(1, 126);
        if pre <= 63 {
            *slot = Ctx {
                pstate: (63 - pre) as u8,
                mps: 0,
            };
        } else {
            *slot = Ctx {
                pstate: (pre - 64) as u8,
                mps: 1,
            };
        }
    }
    out
}

/// 算术解码器本体。
pub struct Cabac<'a> {
    data: &'a [u8],
    /// 下一位的绝对位偏移（自 `data[0]` 起）
    bit: usize,
    range: u32,
    offset: u32,
}

impl<'a> Cabac<'a> {
    /// 从 `bit_offset` 起建解码器：跳过对齐位 → 读 9 位 `codIOffset`。
    pub fn new(data: &'a [u8], bit_offset: usize) -> Result<Self> {
        let mut c = Self {
            data,
            bit: bit_offset,
            range: 510,
            offset: 0,
        };
        // §9.3.1.2.1：`cabac_alignment_one_bit` 全是 1
        while c.bit % 8 != 0 {
            if c.read_bit() != 1 {
                return Err(MediaError::Bitstream(format!(
                    "CABAC 对齐位不是 1（位偏移 {}，字节 {}）",
                    c.bit,
                    c.bit / 8
                )));
            }
        }
        c.offset = c.read_bits(9)?;
        Ok(c)
    }

    #[inline]
    fn read_bit(&mut self) -> u32 {
        let byte = self.bit / 8;
        let v = if byte < self.data.len() {
            ((self.data[byte] >> (7 - (self.bit % 8))) & 1) as u32
        } else {
            // 越界按 0 读：CABAC 在最后一个 bin 之后可能还要重归一化几位，
            // 规范允许 `cabac_zero_word` 补零，这里等价处理。
            0
        };
        self.bit += 1;
        v
    }

    fn read_bits(&mut self, n: usize) -> Result<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.read_bit();
        }
        Ok(v)
    }

    /// §9.3.1.2.2 `decode_decision`：带上下文的 bin。
    #[inline]
    pub fn decision(&mut self, ctx: &mut Ctx) -> u32 {
        let st = ctx.pstate as usize;
        let lps = RANGE_TAB_LPS[st][((self.range >> 6) & 3) as usize] as u32;
        self.range -= lps;
        let bin;
        if self.offset >= self.range {
            // LPS 区间
            self.offset -= self.range;
            self.range = lps;
            bin = 1 - ctx.mps as u32;
            if SWITCH_LPS[st] {
                ctx.mps ^= 1;
            }
            ctx.pstate = NEXT_LPS[st];
        } else {
            bin = ctx.mps as u32;
            ctx.pstate = NEXT_MPS[st];
        }
        while self.range < 256 {
            self.range <<= 1;
            self.offset = (self.offset << 1) | self.read_bit();
        }
        bin
    }

    /// §9.3.1.2.3 `decode_bypass`：等概率 bin（不碰上下文，也不重归一化）。
    #[inline]
    pub fn bypass(&mut self) -> u32 {
        self.offset = (self.offset << 1) | self.read_bit();
        if self.offset >= self.range {
            self.offset -= self.range;
            1
        } else {
            0
        }
    }

    /// §9.3.1.2.4 `decode_terminate`：返回 `true` = `end_of_slice_flag` 为 1。
    #[inline]
    pub fn terminate(&mut self) -> bool {
        self.range -= 2;
        if self.offset >= self.range {
            return true;
        }
        while self.range < 256 {
            self.range <<= 1;
            self.offset = (self.offset << 1) | self.read_bit();
        }
        false
    }

    /// 旁路 Exp-Golomb（`ue(v)`）。CABAC 里只用于 `mb_qp_delta` 一类小值。
    pub fn ue_bypass(&mut self) -> u32 {
        let mut k = 0u32;
        while self.bypass() == 1 && k < 31 {
            k += 1;
        }
        let mut suffix = 0u32;
        for _ in 0..k {
            suffix = (suffix << 1) | self.bypass();
        }
        ((1u32 << k) - 1) + suffix
    }

    /// 当前位偏移（诊断用）。
    pub fn bit_position(&self) -> usize {
        self.bit
    }

    /// 对齐到字节边界（`I_PCM` 前后要用）。
    pub fn byte_align(&mut self) {
        self.bit = (self.bit + 7) & !7;
    }

    /// 取 `n` 个原始字节（`I_PCM` 的样本），并把读指针推到之后。
    pub fn take_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        debug_assert_eq!(self.bit % 8, 0);
        let start = self.bit / 8;
        let end = start.checked_add(n).ok_or_else(|| {
            MediaError::Bitstream("I_PCM 样本长度溢出".into())
        })?;
        if end > self.data.len() {
            return Err(MediaError::Bitstream(format!(
                "I_PCM 要 {n} 字节，但切片只剩 {} 字节",
                self.data.len().saturating_sub(start)
            )));
        }
        self.bit = end * 8;
        Ok(&self.data[start..end])
    }

    /// `I_PCM` 的样本读完以后，算术解码器从下一个字节**重新开始**（§9.3.1.2）。
    pub fn restart(&mut self) -> Result<()> {
        debug_assert_eq!(self.bit % 8, 0);
        self.range = 510;
        self.offset = self.read_bits(9)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_init_maps_to_the_documented_half_range() {
        // §9.3.1.1：preCtxState <= 63 → pStateIdx = 63 - pre、valMPS = 0；
        // 否则 pStateIdx = pre - 64、valMPS = 1。
        // I 切片 ctxIdx 0 的初值是 (20, -15)：QP=26 时 pre = 32 - 15 = 17
        // → pStateIdx = 46、MPS = 0。
        let c = init_contexts(26, 0, true);
        assert_eq!(c[0], Ctx { pstate: 46, mps: 0 });
        // ctxIdx 1 = (2, 54)：pre = 3 + 54 = 57 → pStateIdx = 6、MPS = 0
        assert_eq!(c[1], Ctx { pstate: 6, mps: 0 });
        // ctxIdx 6 = (-28, 127)：pre = -46 + 127 = 81 → pStateIdx = 17、MPS = 1
        assert_eq!(c[6], Ctx { pstate: 17, mps: 1 });
    }

    #[test]
    fn p_state_tables_are_the_spec_ones() {
        // 表 9-44（rangeTabLPS）的头两行与最后两行，肉眼可查
        assert_eq!(RANGE_TAB_LPS[0], [128, 176, 208, 240]);
        assert_eq!(RANGE_TAB_LPS[1], [128, 167, 197, 227]);
        assert_eq!(RANGE_TAB_LPS[62], [6, 7, 8, 9]);
        // ⚠️ 最后一行（pStateIdx 63）是规范里的特例：四个 range 档全是 2，
        // 不按上一行的递减规律走。ffmpeg 的 ff_h264_cabac_tables 也是 2,2,2,2
        // —— 曾经按"5,6,8,9"写成测试期望，把正确的表判成了红的。
        assert_eq!(RANGE_TAB_LPS[63], [2, 2, 2, 2]);
        assert_eq!(NEXT_MPS[0], 1);
        assert_eq!(NEXT_MPS[62], 62);
        assert_eq!(NEXT_LPS[0], 0);
        assert_eq!(NEXT_LPS[63], 63);
        assert!(SWITCH_LPS[0]);
        assert!(!SWITCH_LPS[1]);
    }
}
