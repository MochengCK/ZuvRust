//! SILK 的**重采样器**（参考实现 `silk/resampler.c` + `resampler_private_up2_HQ.c`
//! + `resampler_private_IIR_FIR.c`）。
//!
//! SILK 内部只有 8 / 12 / 16 kHz，而应用要的通常是 48 kHz，所以每条 SILK 轨道
//! 最后都要过一遍这里。参考实现按"输入率 × 输出率"列了一张方法矩阵
//! （`resampler.c` 顶部那张表）：
//!
//! ```text
//! 输入 = 8 / 12 / 16 kHz，输出 = 48 kHz 时：
//!   8 → 48 : 2 倍上采样（全通 IIR）+ FIR 分数插值   ← IIR_FIR
//!   12 → 48 : 同上
//!   16 → 48 : 同上
//! 输入 == 输出时走纯拷贝。
//! ```
//!
//! **下调（输出率 < 输入率）本引擎暂时不支持**：桌面端播放与 `decode --audio`
//! 都按 48 kHz 输出，用不到那条 `down_FIR` 分支，硬写一份没验证过的下来会违反
//! "不产噪声"的原则 —— 这里明确报 `Unsupported`。
//!
//! ## 为什么"照抄"这么重要
//!
//! 这里的每一级都是定点：2 倍上采样是**三级全通 IIR**（状态 Q10），
//! 分数插值是一张 12 相位的 8 抽头 FIR（系数 Q15，`frac_FIR_12`）。
//! 相位由 `invRatio_Q16` 累加决定，**这个值必须与参考实现逐位相同**，
//! 否则插值相位会逐样本漂移、听感上是音调微飘，对拍上表现为"整个文件都差一点点"。
//!
//! ## 两个描述上的坑
//!
//! 1. **`invRatio_Q16` 要"向上取整"**：初始化末尾那个 `while` 循环不是防御性代码，
//!    而是**定义的一部分**（比值必须刚好不小）；去掉它会在 16→48 上少一个 LSB。
//! 2. **`sFIR` 在参考实现里是 `opus_int32[36]`，但按 `RESAMPLER_ORDER_FIR_12 *
//!    sizeof(opus_int32)`（= 32 字节）做 memcpy** —— 实际只用到前 8 个 int16 当历史，
//!    多出来的 8 个样点位置立刻被 2 倍上采样覆盖。这里建模成 `[i16; 8]`，行为等价
//!    （写代码时别被那个 float 了）。

use me_core::{MediaError, Result};

use super::silk_fix as fx;

/// 参考实现 `RESAMPLER_ORDER_FIR_12`：分数插值的抽头数。
const FIR_ORDER_12: usize = 8;
/// 参考实现 `RESAMPLER_MAX_BATCH_SIZE_MS`。
const MAX_BATCH_SIZE_MS: usize = 10;
/// 内部分批缓冲（`RESAMPLER_MAX_BATCH_SIZE_IN`）。
const MAX_BATCH_SIZE_IN: usize = MAX_BATCH_SIZE_MS * 48;
/// 2 倍上采样三级全通的状态个数（`sIIR[6]`）。
const IIR_STATE_LEN: usize = 6;

/// 2 倍上采样：偶/奇输出各三级全通（`resampler_rom.c`）。
const UP2_HQ_0: [i32; 3] = [1746, 14986, 39083 - 65536];
const UP2_HQ_1: [i32; 3] = [6854, 25769, 55542 - 65536];

/// 12 相位的分数插值 FIR（`silk_resampler_frac_FIR_12`，Q15）。
const FRAC_FIR_12: [[i16; 4]; 12] = [
    [189, -600, 617, 30567],
    [117, -159, -1070, 29704],
    [52, 221, -2392, 28276],
    [-4, 529, -3350, 26341],
    [-48, 758, -3956, 23973],
    [-80, 905, -4235, 21254],
    [-99, 972, -4222, 18278],
    [-107, 967, -3957, 15143],
    [-103, 896, -3487, 11950],
    [-91, 773, -2865, 8798],
    [-71, 611, -2143, 5784],
    [-46, 425, -1375, 2996],
];

/// 解码方向的"延迟补偿"表（`delay_matrix_dec`）：`[rateID(in)][rateID(out)]`。
const DELAY_MATRIX_DEC: [[i32; 5]; 3] = [
    [4, 0, 2, 0, 0],
    [0, 9, 4, 7, 4],
    [0, 3, 12, 7, 7],
];

/// `rateID(R)`：把 `{8000, 12000, 16000, 24000, 48000}` 映射成 `{0,1,2,3,4}`。
///
/// 参考实现写成 `((((R>>12) - (R>16000)) >> (R>24000)) - 1)` 一串位移 ——
/// 等价于下面这张查表（那些位移只是"没有查表可用"年代的写法）。
fn rate_id(r: i32) -> Option<usize> {
    match r {
        8000 => Some(0),
        12000 => Some(1),
        16000 => Some(2),
        24000 => Some(3),
        48000 => Some(4),
        _ => None,
    }
}

/// 参考实现的 `silk_resampler_state_struct`。
#[derive(Debug, Clone)]
pub struct SilkResampler {
    /// 2 倍上采样的全通状态（Q10）。
    s_iir: [i32; IIR_STATE_LEN],
    /// 分数插值 FIR 的历史（参考实现那份 32 字节 memcpy 的有效部分）。
    s_fir: [i16; FIR_ORDER_12],
    /// 延迟缓冲：把上一帧末尾的 `input_delay` 个样点接到本帧前面。
    delay_buf: [i16; 48],
    inv_ratio_q16: i32,
    batch_size: usize,
    fs_in_khz: usize,
    fs_out_khz: usize,
    input_delay: usize,
    /// 走不走 2 倍上采样（输出率 > 输入率且不是 2:1 时为 true）。
    up2x: bool,
}

impl SilkResampler {
    /// `silk_resampler_init`（解码方向）。
    pub fn new(fs_hz_in: i32, fs_hz_out: i32) -> Result<Self> {
        if !matches!(fs_hz_in, 8000 | 12000 | 16000) {
            return Err(MediaError::Unsupported(format!(
                "SILK 重采样：输入采样率 {fs_hz_in} Hz 不支持（只有 8/12/16 kHz）"
            )));
        }
        if !matches!(fs_hz_out, 8000 | 12000 | 16000 | 24000 | 48000) {
            return Err(MediaError::Unsupported(format!(
                "SILK 重采样：输出采样率 {fs_hz_out} Hz 不支持（只有 8/12/16/24/48 kHz）"
            )));
        }
        let in_id = rate_id(fs_hz_in).unwrap();
        let out_id = rate_id(fs_hz_out).unwrap();
        let input_delay = DELAY_MATRIX_DEC[in_id][out_id] as usize;
        let fs_in_khz = (fs_hz_in / 1000) as usize;
        let fs_out_khz = (fs_hz_out / 1000) as usize;
        let batch_size = fs_in_khz * MAX_BATCH_SIZE_MS;

        let up2x;
        if fs_hz_out > fs_hz_in {
            // 2:1 时直接走 2 倍上采样；否则"先 2 倍上采样再 FIR 分数插值"
            up2x = fs_hz_out != fs_hz_in * 2;
        } else if fs_hz_out < fs_hz_in {
            return Err(MediaError::Unsupported(format!(
                "SILK 重采样：下调 {fs_hz_in} → {fs_hz_out} Hz 还没实现（桌面端固定 48 kHz 输出，用不到）"
            )));
        } else {
            up2x = false;
        }

        let mut s = Self {
            s_iir: [0; IIR_STATE_LEN],
            s_fir: [0; FIR_ORDER_12],
            delay_buf: [0; 48],
            inv_ratio_q16: 0,
            batch_size,
            fs_in_khz,
            fs_out_khz,
            input_delay,
            up2x,
        };
        // 参考实现：`LSHIFT32(DIV32(LSHIFT32(Fs_Hz_in, 14 + up2x), Fs_Hz_out), 2)`
        let numerator = (fs_hz_in as i64) << (14 + if up2x { 1 } else { 0 });
        s.inv_ratio_q16 = ((numerator / fs_hz_out as i64) << 2) as i32;
        // 比值必须"向上取整"——这是定义的一部分，不是防御
        let target = fs_hz_in << if up2x { 1 } else { 0 };
        while fx::smulww(s.inv_ratio_q16, fs_hz_out) < target {
            s.inv_ratio_q16 += 1;
        }
        Ok(s)
    }

    /// 采样率是否需要重采样（相同则走拷贝）。
    pub fn is_passthrough(&self) -> bool {
        self.fs_in_khz == self.fs_out_khz
    }

    /// 一批输入对应的输出样点数（调用方分配缓冲用）。
    pub fn out_len(&self, in_len: usize) -> usize {
        in_len * self.fs_out_khz / self.fs_in_khz
    }

    /// `silk_resampler`：把 `input`（`in_len` 个样点）重采样到输出。
    ///
    /// 输出写入 `out` 的开头 `out_len(in_len)` 个位置。
    pub fn process(&mut self, out: &mut [i16], input: &[i16], in_len: usize) {
        debug_assert!(in_len >= self.fs_in_khz, "至少要有 1 ms 的输入");
        debug_assert!(self.input_delay <= self.fs_in_khz);
        let n_samples = self.fs_in_khz - self.input_delay;
        debug_assert!(in_len >= n_samples);

        // 上一帧尾巴 + 本帧开头 → delayBuf
        self.delay_buf[self.input_delay..self.input_delay + n_samples]
            .copy_from_slice(&input[..n_samples]);

        let mut delay = [0i16; 48];
        delay[..self.fs_in_khz].copy_from_slice(&self.delay_buf[..self.fs_in_khz]);
        self.resample_into(out, &delay[..self.fs_in_khz], self.fs_in_khz);
        self.resample_into(
            &mut out[self.fs_out_khz..],
            &input[n_samples..],
            in_len - self.fs_in_khz,
        );

        if self.input_delay > 0 {
            self.delay_buf[..self.input_delay]
                .copy_from_slice(&input[in_len - self.input_delay..in_len]);
        }
    }

    fn resample_into(&mut self, out: &mut [i16], input: &[i16], in_len: usize) {
        if self.is_passthrough() || in_len == 0 {
            out[..in_len].copy_from_slice(&input[..in_len]);
            return;
        }
        if self.up2x {
            self.iir_fir(out, input, in_len);
        } else {
            self.up2_hq(out, input, in_len);
        }
    }

    /// 2 倍上采样（三级全通 IIR，`silk_resampler_private_up2_HQ`）。
    fn up2_hq(&mut self, out: &mut [i16], input: &[i16], in_len: usize) {
        for k in 0..in_len {
            let in32 = fx::lshift(input[k] as i32, 10);
            for (half, coef) in [(0usize, &UP2_HQ_0), (1usize, &UP2_HQ_1)] {
                let base = half * 3;
                // 三级全通，串联
                let y = in32.wrapping_sub(self.s_iir[base]);
                let x = fx::smulwb(y, coef[0]);
                let out32_1 = self.s_iir[base].wrapping_add(x);
                self.s_iir[base] = in32.wrapping_add(x);

                let y = out32_1.wrapping_sub(self.s_iir[base + 1]);
                let x = fx::smulwb(y, coef[1]);
                let out32_2 = self.s_iir[base + 1].wrapping_add(x);
                self.s_iir[base + 1] = out32_1.wrapping_add(x);

                let y = out32_2.wrapping_sub(self.s_iir[base + 2]);
                let x = fx::smlawb(y, y, coef[2]);
                let out32_1 = self.s_iir[base + 2].wrapping_add(x);
                self.s_iir[base + 2] = out32_2.wrapping_add(x);

                out[2 * k + half] = fx::sat16(fx::rshift_round(out32_1, 10));
            }
        }
    }

    /// 2 倍上采样 + 分数插值 FIR（`silk_resampler_private_IIR_FIR`）。
    fn iir_fir(&mut self, out: &mut [i16], input: &[i16], in_len: usize) {
        let mut buf = [0i16; MAX_BATCH_SIZE_IN + FIR_ORDER_12];
        buf[..FIR_ORDER_12].copy_from_slice(&self.s_fir);

        let index_increment_q16 = self.inv_ratio_q16;
        let mut pos = 0usize;
        let mut remaining = in_len;
        let mut out_pos = 0usize;
        loop {
            let n_samples_in = remaining.min(self.batch_size);
            self.up2_hq(
                &mut buf[FIR_ORDER_12..],
                &input[pos..pos + n_samples_in],
                n_samples_in,
            );
            // + 1 位是因为做过 2 倍上采样
            let max_index_q16 = ((n_samples_in as i32) << 17) as i64;
            let mut index_q16 = 0i64;
            let inc = index_increment_q16 as i64;
            while index_q16 < max_index_q16 {
                let table_index = (fx::smulwb((index_q16 & 0xFFFF) as i32, 12)) as usize;
                let b = (index_q16 >> 16) as usize;
                let c0 = &FRAC_FIR_12[table_index];
                let c1 = &FRAC_FIR_12[11 - table_index];
                let mut res_q15 = fx::smulbb(buf[b] as i32, c0[0] as i32);
                res_q15 = fx::smlabb(res_q15, buf[b + 1] as i32, c0[1] as i32);
                res_q15 = fx::smlabb(res_q15, buf[b + 2] as i32, c0[2] as i32);
                res_q15 = fx::smlabb(res_q15, buf[b + 3] as i32, c0[3] as i32);
                res_q15 = fx::smlabb(res_q15, buf[b + 4] as i32, c1[3] as i32);
                res_q15 = fx::smlabb(res_q15, buf[b + 5] as i32, c1[2] as i32);
                res_q15 = fx::smlabb(res_q15, buf[b + 6] as i32, c1[1] as i32);
                res_q15 = fx::smlabb(res_q15, buf[b + 7] as i32, c1[0] as i32);
                out[out_pos] = fx::sat16(fx::rshift_round(res_q15, 15));
                out_pos += 1;
                index_q16 += inc;
            }
            pos += n_samples_in;
            remaining -= n_samples_in;
            if remaining > 0 {
                buf.copy_within(n_samples_in << 1..(n_samples_in << 1) + FIR_ORDER_12, 0);
            } else {
                self.s_fir
                    .copy_from_slice(&buf[n_samples_in << 1..(n_samples_in << 1) + FIR_ORDER_12]);
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_id_covers_the_supported_rates() {
        assert_eq!(rate_id(8000), Some(0));
        assert_eq!(rate_id(12000), Some(1));
        assert_eq!(rate_id(16000), Some(2));
        assert_eq!(rate_id(24000), Some(3));
        assert_eq!(rate_id(48000), Some(4));
    }

    #[test]
    fn inv_ratio_rounds_up_so_that_a_full_frame_maps_to_the_full_output() {
        for (fin, fout) in [(8000, 48000), (12000, 48000), (16000, 48000), (8000, 16000)] {
            let r = SilkResampler::new(fin, fout).unwrap();
            let n = r.fs_in_khz; // 1 ms
            let produced = r.out_len(n);
            // 由 invRatio_Q16 推出来的批内输出数必须与按比例算的一致（不差 1）
            let max_index = (n as i64) << if r.up2x { 17 } else { 16 };
            let inc = r.inv_ratio_q16 as i64;
            let count = (max_index + inc - 1) / inc;
            assert_eq!(count as usize, produced, "{fin}→{fout}");
        }
    }

    #[test]
    fn downsampling_is_reported_as_unsupported() {
        assert!(SilkResampler::new(16000, 8000).is_err());
        assert!(SilkResampler::new(8000, 11025).is_err());
    }

    #[test]
    fn equal_rates_pass_through_with_the_delay_compensation() {
        // 采样率相同时也不做"原地拷贝"：参考实现照样过 delayBuf，把上一帧末尾的
        // `inputDelay` 个样点接到前面（16→16 的 inputDelay 是 12）。
        // 这个延迟是**故意的**——它让所有重采样组合的总延迟一致。
        let mut r = SilkResampler::new(16000, 16000).unwrap();
        assert!(r.is_passthrough());
        let input: Vec<i16> = (0..320).map(|i| (i % 100) as i16).collect();
        let mut out = vec![0i16; 320];
        r.process(&mut out, &input, 320);
        let d = r.input_delay;
        assert_eq!(d, 12);
        for i in 0..320 {
            let want = if i < d { 0 } else { input[i - d] };
            assert_eq!(out[i], want, "第 {i} 个样点");
        }
    }

    #[test]
    fn resampler_matches_the_reference_on_a_known_input() {
        // 期望值来自参考实现本身（独立探针 `/tmp/resamp_probe.c` 调 `silk_resampler`）：
        // 输入 320 个样点 `in[i] = (i*7919 % 30001) - 15000`，一次调用（内部会分成
        // 两个 batch，所以这条同时验到分批逻辑），输出 960 个样点。
        // 逐样点全比太长，取"每 60 个一个 + 16→48 那条的末尾 60 个"——
        // 这是一条带全局状态的 IIR+FIR 链，任何一处错了后面每个样点都会偏。
        let input: Vec<i16> = (0..320)
            .map(|i| ((i * 7919 % 30001) - 15000) as i16)
            .collect();
        let cases: [(i32, i32, usize, &[i16]); 3] = [
            (
                16000,
                43691,
                7,
                &[
                    0, -9437, 702, 13219, 723, -4625, 875, 5675, -14994, 2017, 9518, 1320, -6823,
                    1360, 3676, -21583,
                ],
            ),
            (
                12000,
                32768,
                4,
                &[
                    0, 10066, 2675, 737, 1366, 1568, 230, 1868, 5387, -3749, -5834, -6972, -9944,
                    -14353, -20490, -21182,
                ],
            ),
            (
                8000,
                21846,
                0,
                &[
                    0, -841, -384, 8688, -4369, 563, 1180, -9324, 1524, 542, -14970, 15787, 2314,
                    1071, 9825, 4178,
                ],
            ),
        ];
        for (fs_in, inv_ratio, delay, expected) in cases {
            let mut r = SilkResampler::new(fs_in, 48000).unwrap();
            assert_eq!(r.inv_ratio_q16, inv_ratio, "{fs_in}→48k 的 invRatio_Q16");
            assert_eq!(r.input_delay, delay, "{fs_in}→48k 的 inputDelay");
            let out_len = r.out_len(input.len());
            // 16/12/8 kHz 的 320 个样点分别是 20 / 26.67 / 40 ms → 960 / 1280 / 1920
            assert_eq!(out_len, [960, 1280, 1920][[16000, 12000, 8000].iter().position(|&v| v == fs_in).unwrap()]);
            let mut out = vec![0i16; out_len];
            r.process(&mut out, &input, input.len());
            for (i, want) in expected.iter().enumerate() {
                assert_eq!(out[i * 60], *want, "{fs_in}→48k 的第 {} 个输出样点", i * 60);
            }
        }

        // 16→48 的尾巴（含第二个 batch 的边界）
        let mut r = SilkResampler::new(16000, 48000).unwrap();
        let mut out = vec![0i16; 960];
        r.process(&mut out, &input, 320);
        let tail: [i16; 60] = [
            -21583, -14673, -3515, 4943, 7275, 5218, 2945, 2851, 3693, 2307, -2923, -10114,
            -14979, -14054, -7289, 1859, 8849, 11009, 8895, 5182, 2219, 256, -2133, -5985,
            -10102, -11437, -7560, 800, 9614, 14252, 12897, 7696, 2615, -45, -1207, -3255,
            -6749, -9104, -6759, 1053, 10718, 16659, 15778, 9863, 3677, 608, 45, -1067, -4216,
            -7123, -5818, 866, 8902, 11606, 5208, -7439, -18525, -21103, -14207, -3071,
        ];
        assert_eq!(&out[900..960], &tail[..]);
    }
}
