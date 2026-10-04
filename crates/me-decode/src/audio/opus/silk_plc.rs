//! SILK 的**丢包隐藏（PLC）与舒适噪声（CNG）**（参考实现 `silk/PLC.c` 与 `silk/CNG.c`）。
//!
//! 和 CELT 那边的 `celt_plc.rs` 是两套完全不同的东西：CELT 的 PLC 在**浮点频域**里干活，
//! SILK 的 PLC 全在**时域 + 定点**，而且可以做到**逐位相同**。
//!
//! ```text
//! 好帧（每帧都要跑一遍，用来"喂"PLC 的素材）：
//!   silk_PLC_update：记住上一帧的 LPC / LTP 系数、最后两个增益、pitch
//!   silk_CNG        ：估计噪声电平（NLSF 平滑 + 增益平滑 + 激励缓冲）
//! 丢帧：
//!   silk_PLC_conceal：把 LTP 状态用上一帧的 LPC 白化 → 按 pitch 周期做 LTP 预测合成，
//!                     叠上"随机激励"（每次丢包按子帧衰减 LTP 增益与随机增益、pitch 缓慢漂移）
//!                     → 再过一遍 LPC 合成 → 乘上一帧的增益
//!   silk_PLC_glue_frames：从"隐藏帧"切回"好帧"时按能量差做淡入
//!   silk_CNG        ：丢帧时叠一层舒适噪声（用平滑后的 NLSF 当频谱形状）
//! ```
//!
//! ## 几个必须照抄的点
//!
//! 1. **`silk_PLC_update` 在每一帧好帧上都要跑**（不是只在丢包前跑）—— 它负责维护
//!    `prevLPC_Q12` / `prevGain_Q16` / `pitchL_Q8` 这些"下次丢包要用的素材"。
//! 2. **`silk_bwexpander` 改的是 `prevLPC_Q12` 本体**，而 `silk_LPC_inverse_pred_gain`
//!    读的是**展开之后**的系数 —— 顺序不能反。
//! 3. `silk_PLC` 里 `psDec->lossCnt++` 发生在**隐藏之后**；`silk_CNG` 靠
//!    `lossCnt != 0` 判断"这一帧是藏的"，所以两个函数的调用顺序（`PLC` → `glue` → `CNG`）
//!    决定了 `lossCnt` 的取值，别调换。
//! 4. `silk_CNG` 的"正常帧"分支只在 `lossCnt == 0 && prevSignalType == NO_VOICE_ACTIVITY`
//!    时跑；`CNG_synth_state` 的清零只清 `LPC_order` 个（不是 `MAX_LPC_ORDER` 个）。
//! 5. pitch 的缓慢漂移是**乘性**的：`pitchL_Q8 += SMULWB(pitchL_Q8, 655)`（0.01），
//!    上限 `18 ms * fs_kHz`。

use super::silk::{
    SilkChannel, LTP_ORDER, MAX_FRAME_LENGTH, MAX_LPC_ORDER, MAX_NB_SUBFR, MAX_SUB_FRAME_LENGTH,
    TYPE_NO_VOICE_ACTIVITY, TYPE_VOICED,
};
use super::silk_fix as fx;
use super::silk_params::{bwexpander, lpc_inverse_pred_gain, nlsf2a};
use super::silk_synth::lpc_analysis_filter;

// ── 参考实现 `silk/PLC.h` 与 `silk/define.h` 里的常量 ──

/// `SILK_FIX_CONST(0.99, 16)`。
const BWE_COEF_Q16: i32 = 64881;
const V_PITCH_GAIN_START_MIN_Q14: i32 = 11469;
const V_PITCH_GAIN_START_MAX_Q14: i32 = 15565;
const MAX_PITCH_LAG_MS: i32 = 18;
const RAND_BUF_SIZE: usize = 128;
const RAND_BUF_MASK: i32 = RAND_BUF_SIZE as i32 - 1;
const LOG2_INV_LPC_GAIN_HIGH_THRES: i32 = 3;
const LOG2_INV_LPC_GAIN_LOW_THRES: i32 = 8;
const PITCH_DRIFT_FAC_Q16: i32 = 655;
/// `SILK_FIX_CONST(1, 16)`。
const ONE_Q16: i32 = 65536;

const HARM_ATT_Q15: [i32; 2] = [32440, 31130];
const PLC_RAND_ATTENUATE_V_Q15: [i32; 2] = [31130, 26214];
const PLC_RAND_ATTENUATE_UV_Q15: [i32; 2] = [32440, 29491];

/// `CNG_BUF_MASK_MAX`（`2^floor(log2(MAX_FRAME_LENGTH)) - 1`）。
const CNG_BUF_MASK_MAX: i32 = 255;
/// `CNG_GAIN_SMTH_Q16`（`0.25^(1/4)`）。
const CNG_GAIN_SMTH_Q16: i32 = 4634;
/// `CNG_NLSF_SMTH_Q16`（0.25）。
const CNG_NLSF_SMTH_Q16: i32 = 16348;

/// `silk_RAND(seed)`：LCG。**乘数与增量别写反** ——
/// `silk_MLA_ovflw(907633515, seed, 196314165)` 展开是 `907633515 + seed*196314165`
/// （常量在前）。写成 `seed*907633515 + 196314165` 看起来更眼熟，但那是错的：
/// `silk_RAND(0)` 必须是 907633515。直接复用 `decode_core` 里那一份，保证只有一处实现。
#[inline]
fn rand(seed: i32) -> i32 {
    super::silk_synth::rand(seed)
}

/// `silk_sum_sqr_shift`：算一段 int16 的平方和，并在溢出前右移、把移位数报出来。
///
/// "移位"这件事是**规范行为**：调用方拿 `energy/shift` 去比较不同段的能量，
/// 两边 shift 不同还要先对齐（`glue_frames` 就是这么干的）。
pub fn sum_sqr_shift(x: &[i16]) -> (i32, i32) {
    let mut len = x.len() as i32 - 1;
    let mut nrg: i32 = 0;
    let mut shft: i32 = 0;
    let mut i: i32 = 0;

    while i < len {
        nrg = fx::smlabb_ovflw(nrg, x[i as usize] as i32, x[i as usize] as i32);
        nrg = fx::smlabb_ovflw(nrg, x[i as usize + 1] as i32, x[i as usize + 1] as i32);
        if nrg < 0 {
            // 溢出 → 整体右移 2 位后从头继续（下面那个循环）
            nrg = ((nrg as u32) >> 2) as i32;
            shft = 2;
            break;
        }
        i += 2;
    }
    while i < len {
        let mut tmp = fx::smulbb(x[i as usize] as i32, x[i as usize] as i32);
        tmp = fx::smlabb_ovflw(tmp, x[i as usize + 1] as i32, x[i as usize + 1] as i32);
        nrg = nrg.wrapping_add(((tmp as u32) >> shft) as i32);
        if nrg < 0 {
            nrg = ((nrg as u32) >> 2) as i32;
            shft += 2;
        }
        i += 2;
    }
    if i == len {
        // 还剩最后一个样点
        let tmp = fx::smulbb(x[i as usize] as i32, x[i as usize] as i32);
        nrg = nrg.wrapping_add(((tmp as u32) >> shft) as i32);
    }
    // 保证至少有两个前导零
    if nrg & 0xC000_0000u32 as i32 != 0 {
        nrg = ((nrg as u32) >> 2) as i32;
        shft += 2;
    }
    let _ = &mut len;
    (nrg, shft)
}

/// PLC 的跨帧状态（参考实现 `silk_PLC_struct`）。
#[derive(Debug, Clone)]
pub struct SilkPlcState {
    /// 浊音隐藏要用的 pitch lag（Q8）。
    pub pitch_l_q8: i32,
    /// 浊音隐藏要用的 LTP 系数。
    pub ltp_coef_q14: [i16; LTP_ORDER],
    /// 上一帧的 LPC 系数（丢包时做带宽扩展后再用）。
    pub prev_lpc_q12: [i16; MAX_LPC_ORDER],
    /// 上一帧是不是"隐藏出来的"（`glue_frames` 要用）。
    pub last_frame_lost: bool,
    pub rand_seed: i32,
    pub rand_scale_q14: i16,
    /// 上一段隐藏残差的能量与它的移位（跨帧比较用）。
    pub conc_energy: i32,
    pub conc_energy_shift: i32,
    pub prev_ltp_scale_q14: i16,
    /// 最后两个子帧的增益（丢包时乘在 LPC 合成输出上）。
    pub prev_gain_q16: [i32; 2],
    pub fs_khz: u32,
    pub nb_subfr: usize,
    pub subfr_length: usize,
}

impl Default for SilkPlcState {
    fn default() -> Self {
        Self {
            pitch_l_q8: 0,
            ltp_coef_q14: [0; LTP_ORDER],
            prev_lpc_q12: [0; MAX_LPC_ORDER],
            last_frame_lost: false,
            rand_seed: 0,
            rand_scale_q14: 0,
            conc_energy: 0,
            conc_energy_shift: 0,
            prev_ltp_scale_q14: 0,
            prev_gain_q16: [0; 2],
            fs_khz: 0,
            nb_subfr: 0,
            subfr_length: 0,
        }
    }
}

impl SilkPlcState {
    /// `silk_PLC_Reset`。
    pub fn reset(&mut self, frame_length: usize) {
        self.pitch_l_q8 = (frame_length as i32) << 7;
        self.prev_gain_q16 = [ONE_Q16, ONE_Q16];
        self.subfr_length = 20;
        self.nb_subfr = 2;
    }
}

/// CNG 的跨帧状态（参考实现 `silk_CNG_struct`）。
#[derive(Debug, Clone)]
pub struct SilkCngState {
    pub exc_buf_q14: [i32; MAX_FRAME_LENGTH],
    pub smth_nlsf_q15: [i16; MAX_LPC_ORDER],
    pub synth_state: [i32; MAX_LPC_ORDER],
    pub smth_gain_q16: i32,
    pub rand_seed: i32,
    pub fs_khz: u32,
}

impl Default for SilkCngState {
    fn default() -> Self {
        Self {
            exc_buf_q14: [0; MAX_FRAME_LENGTH],
            smth_nlsf_q15: [0; MAX_LPC_ORDER],
            synth_state: [0; MAX_LPC_ORDER],
            smth_gain_q16: 0,
            rand_seed: 0,
            fs_khz: 0,
        }
    }
}

impl SilkCngState {
    /// `silk_CNG_Reset`。
    pub fn reset(&mut self, lpc_order: usize) {
        let step = 32767 / (lpc_order as i32 + 1);
        let mut acc = 0i32;
        for slot in self.smth_nlsf_q15.iter_mut().take(lpc_order) {
            acc += step;
            *slot = acc as i16;
        }
        self.smth_gain_q16 = 0;
        self.rand_seed = 3176576;
    }
}

/// `silk_PLC`：丢包时隐藏、好帧时更新状态。
///
/// `lost` 为真时把隐藏出来的信号写进 `frame`（`SilkChannel::frame_length` 个样点）。
pub fn plc(ch: &mut SilkChannel, ctrl: &mut super::silk_params::SilkDecoderControl, frame: &mut [i16], lost: bool) {
    if ch.fs_khz != ch.plc.fs_khz {
        let fl = ch.frame_length;
        ch.plc.reset(fl);
        ch.plc.fs_khz = ch.fs_khz;
    }
    if lost {
        plc_conceal(ch, ctrl, frame);
        ch.loss_cnt += 1;
    } else {
        plc_update(ch, ctrl);
    }
}

/// `silk_PLC_update`：把"这次好帧"的参数存下来，丢包时照着外推。
fn plc_update(ch: &mut SilkChannel, ctrl: &super::silk_params::SilkDecoderControl) {
    ch.prev_signal_type = ch.indices.signal_type as i32;

    let mut ltp_gain_q14 = 0i32;
    if ch.indices.signal_type as i32 == TYPE_VOICED {
        // 找一个"含 pitch 脉冲"的子帧当隐藏用的 LTP
        let mut j = 0usize;
        while (j * ch.subfr_length) < ctrl.pitch_l[ch.nb_subfr - 1] as usize {
            if j == ch.nb_subfr {
                break;
            }
            let base = (ch.nb_subfr - 1 - j) * LTP_ORDER;
            let mut tmp = 0i32;
            for i in 0..LTP_ORDER {
                tmp += ctrl.ltp_coef_q14[base + i] as i32;
            }
            if tmp > ltp_gain_q14 {
                ltp_gain_q14 = tmp;
                ch.plc.ltp_coef_q14.copy_from_slice(&ctrl.ltp_coef_q14[base..base + LTP_ORDER]);
                ch.plc.pitch_l_q8 = ctrl.pitch_l[ch.nb_subfr - 1 - j] << 8;
            }
            j += 1;
        }

        ch.plc.ltp_coef_q14 = [0; LTP_ORDER];
        ch.plc.ltp_coef_q14[LTP_ORDER / 2] = ltp_gain_q14 as i16;

        // 把 LTP 增益限制在 [0.7, 0.95]
        if ltp_gain_q14 < V_PITCH_GAIN_START_MIN_Q14 {
            let scale_q10 = fx::div32_16(V_PITCH_GAIN_START_MIN_Q14 << 10, fx::max(ltp_gain_q14, 1));
            for v in ch.plc.ltp_coef_q14.iter_mut() {
                *v = (fx::smulbb(*v as i32, scale_q10) >> 10) as i16;
            }
        } else if ltp_gain_q14 > V_PITCH_GAIN_START_MAX_Q14 {
            let scale_q14 = fx::div32_16(V_PITCH_GAIN_START_MAX_Q14 << 14, fx::max(ltp_gain_q14, 1));
            for v in ch.plc.ltp_coef_q14.iter_mut() {
                *v = (fx::smulbb(*v as i32, scale_q14) >> 14) as i16;
            }
        }
    } else {
        ch.plc.pitch_l_q8 = (ch.fs_khz as i32 * MAX_PITCH_LAG_MS) << 8;
        ch.plc.ltp_coef_q14 = [0; LTP_ORDER];
    }

    let order = ch.tables.lpc_order;
    ch.plc.prev_lpc_q12[..order].copy_from_slice(&ctrl.pred_coef_q12[1][..order]);
    ch.plc.prev_ltp_scale_q14 = ctrl.ltp_scale_q14 as i16;
    ch.plc.prev_gain_q16 = [ctrl.gains_q16[ch.nb_subfr - 2], ctrl.gains_q16[ch.nb_subfr - 1]];
    ch.plc.subfr_length = ch.subfr_length;
    ch.plc.nb_subfr = ch.nb_subfr;
}

/// `silk_PLC_conceal`：真正"藏"出一帧。
fn plc_conceal(ch: &mut SilkChannel, ctrl: &mut super::silk_params::SilkDecoderControl, frame: &mut [i16]) {
    let order = ch.tables.lpc_order;
    // `exc_buf` 那一段用 `psPLC->*`（上一帧存下来的），LTP/LPC 循环用 `psDec->*`（当前帧的）
    let plc_subfr_length = ch.plc.subfr_length;
    let plc_nb_subfr = ch.plc.nb_subfr;
    let subfr_length = ch.subfr_length;
    let nb_subfr = ch.nb_subfr;
    let ltp_mem_length = ch.ltp_mem_length;

    let prev_gain_q10 = [ch.plc.prev_gain_q16[0] >> 6, ch.plc.prev_gain_q16[1] >> 6];

    if ch.first_frame_after_reset {
        ch.plc.prev_lpc_q12 = [0; MAX_LPC_ORDER];
    }

    // ── 挑一段"能量最低的子帧"当随机激励的来源 ──
    let mut exc_buf = [0i16; 2 * MAX_SUB_FRAME_LENGTH];
    for k in 0..2 {
        for i in 0..plc_subfr_length {
            let raw = fx::smulww(
                ch.exc_q14[i + (k + plc_nb_subfr - 2) * plc_subfr_length],
                prev_gain_q10[k],
            );
            exc_buf[k * plc_subfr_length + i] = fx::sat16(fx::rshift(raw, 8));
        }
    }
    let (energy1, shift1) = sum_sqr_shift(&exc_buf[..plc_subfr_length]);
    let (energy2, shift2) = sum_sqr_shift(&exc_buf[plc_subfr_length..2 * plc_subfr_length]);
    let rand_off = if fx::rshift(energy1, shift2 as u32) < fx::rshift(energy2, shift1 as u32) {
        // 第一个子帧能量低
        fx::max(0, ((plc_nb_subfr - 1) * plc_subfr_length) as i32 - RAND_BUF_SIZE as i32) as usize
    } else {
        fx::max(0, (plc_nb_subfr * plc_subfr_length) as i32 - RAND_BUF_SIZE as i32) as usize
    };

    let mut b_q14 = ch.plc.ltp_coef_q14;
    let mut rand_scale_q14 = ch.plc.rand_scale_q14 as i32;

    let att_idx = usize::min(1, ch.loss_cnt.max(0) as usize);
    let harm_gain_q15 = HARM_ATT_Q15[att_idx];
    let mut rand_gain_q15 = if ch.prev_signal_type == TYPE_VOICED {
        PLC_RAND_ATTENUATE_V_Q15[att_idx]
    } else {
        PLC_RAND_ATTENUATE_UV_Q15[att_idx]
    };

    // LPC 隐藏：先给上一帧的 LPC 做带宽扩展（**就地改**），再拷到栈上
    bwexpander(&mut ch.plc.prev_lpc_q12[..order], BWE_COEF_Q16);
    let mut a_q12 = [0i16; MAX_LPC_ORDER];
    a_q12[..order].copy_from_slice(&ch.plc.prev_lpc_q12[..order]);

    // ── 第一帧丢包：定随机成分的强度 ──
    if ch.loss_cnt == 0 {
        rand_scale_q14 = 1 << 14;
        if ch.prev_signal_type == TYPE_VOICED {
            // 浊音：LTP 已经在重建波形了，随机成分要压下去
            for i in 0..LTP_ORDER {
                rand_scale_q14 -= b_q14[i] as i32;
            }
            rand_scale_q14 = fx::max(3277, rand_scale_q14);
            rand_scale_q14 = fx::smulbb(rand_scale_q14, ch.plc.prev_ltp_scale_q14 as i32) >> 14;
        } else {
            // 清音：LPC 增益越高，随机成分越要压（否则会有金属声）
            let inv_gain_q30 = lpc_inverse_pred_gain(&ch.plc.prev_lpc_q12[..order]);
            let mut down_scale_q30 = fx::min(1 << 30 >> LOG2_INV_LPC_GAIN_HIGH_THRES, inv_gain_q30);
            down_scale_q30 = fx::max(1 << 30 >> LOG2_INV_LPC_GAIN_LOW_THRES, down_scale_q30);
            down_scale_q30 <<= LOG2_INV_LPC_GAIN_HIGH_THRES;
            rand_gain_q15 = fx::rshift(fx::smulwb(down_scale_q30, rand_gain_q15), 14);
        }
    }

    let mut rand_seed = ch.plc.rand_seed;
    let mut lag = fx::rshift_round(ch.plc.pitch_l_q8, 8);
    let mut s_ltp_buf_idx = ltp_mem_length;

    // ── 把 LTP 历史重新白化成"激励"（用上一帧的 LPC 当白化滤波器）──
    let mut s_ltp = [0i16; MAX_FRAME_LENGTH];
    let mut s_ltp_q14 = [0i32; 2 * MAX_FRAME_LENGTH];
    let idx = ltp_mem_length as i32 - lag - order as i32 - (LTP_ORDER / 2) as i32;
    assert!(idx > 0, "SILK PLC：pitch 太大，白化窗口跑出缓冲");
    let idx = idx as usize;
    let blen = ltp_mem_length - idx;
    lpc_analysis_filter(&mut s_ltp[idx..idx + blen], &ch.out_buf[idx..idx + blen], &a_q12[..order]);

    // 再按"上一帧的增益"归一化回来
    let mut inv_gain_q30 = fx::inverse32_varq(ch.plc.prev_gain_q16[1], 46);
    inv_gain_q30 = fx::min(inv_gain_q30, i32::MAX >> 1);
    for i in (idx + order)..ltp_mem_length {
        s_ltp_q14[i] = fx::smulwb(inv_gain_q30, s_ltp[i] as i32);
    }

    // ── LTP 合成（逐子帧，增益与随机成分逐子帧衰减、pitch 缓慢漂移）──
    for _k in 0..nb_subfr {
        let mut pos = s_ltp_buf_idx as i32 - lag + (LTP_ORDER / 2) as i32;
        for _i in 0..subfr_length {
            let mut ltp_pred_q12 = 2;
            for j in 0..LTP_ORDER {
                ltp_pred_q12 = fx::smlawb(ltp_pred_q12, s_ltp_q14[(pos - j as i32) as usize], b_q14[j] as i32);
            }
            pos += 1;

            rand_seed = rand(rand_seed);
            let ridx = (fx::rshift(rand_seed, 25) & RAND_BUF_MASK) as usize;
            let rand_sample = ch.exc_q14[rand_off + ridx];
            s_ltp_q14[s_ltp_buf_idx] = fx::lshift(fx::smlawb(ltp_pred_q12, rand_sample, rand_scale_q14), 2);
            s_ltp_buf_idx += 1;
        }

        for j in 0..LTP_ORDER {
            b_q14[j] = (fx::smulbb(harm_gain_q15, b_q14[j] as i32) >> 15) as i16;
        }
        rand_scale_q14 = fx::smulbb(rand_scale_q14, rand_gain_q15) >> 15;

        ch.plc.pitch_l_q8 = fx::smlawb(ch.plc.pitch_l_q8, ch.plc.pitch_l_q8, PITCH_DRIFT_FAC_Q16);
        ch.plc.pitch_l_q8 = fx::min(
            ch.plc.pitch_l_q8,
            (MAX_PITCH_LAG_MS * ch.fs_khz as i32) << 8,
        );
        lag = fx::rshift_round(ch.plc.pitch_l_q8, 8);
    }

    // ── LPC 合成 ──
    let base = ltp_mem_length - MAX_LPC_ORDER;
    s_ltp_q14[base..base + MAX_LPC_ORDER].copy_from_slice(&ch.s_lpc_q14_buf);
    for i in 0..ch.frame_length {
        let at = base + MAX_LPC_ORDER + i;
        let mut lpc_pred_q10 = fx::rshift(order as i32, 1);
        for j in 0..order {
            lpc_pred_q10 = fx::smlawb(lpc_pred_q10, s_ltp_q14[at - j - 1], a_q12[j] as i32);
        }
        s_ltp_q14[at] = fx::add_lshift32(s_ltp_q14[at], lpc_pred_q10, 4);
        let v = fx::smulww(s_ltp_q14[at], prev_gain_q10[1]);
        frame[i] = fx::sat16(fx::sat16(fx::rshift_round(v, 8)) as i32);
    }
    ch.s_lpc_q14_buf
        .copy_from_slice(&s_ltp_q14[base + ch.frame_length..base + ch.frame_length + MAX_LPC_ORDER]);

    // ── 状态推进 ──
    // `B_Q14` 在参考实现里是**指回 `psPLC->LTPCoef_Q14` 的指针**：逐子帧的衰减会写回去，
    // 于是连续丢包时 LTP 增益是**累积**下降的。用局部副本却忘了写回，每一帧都会从
    // 原始增益重新开始衰减 —— 第二个丢失帧起就不对。
    ch.plc.ltp_coef_q14 = b_q14;
    ch.plc.rand_seed = rand_seed;
    ch.plc.rand_scale_q14 = rand_scale_q14 as i16;
    for i in 0..MAX_NB_SUBFR {
        ctrl.pitch_l[i] = lag;
    }
}

/// `silk_PLC_glue_frames`：从"隐藏帧"切回好帧时按能量差做淡入。
pub fn plc_glue_frames(ch: &mut SilkChannel, frame: &mut [i16], length: usize) {
    if ch.loss_cnt != 0 {
        // 记住隐藏残差的能量，下一帧好帧回来时按它对齐
        let (energy, shift) = sum_sqr_shift(&frame[..length]);
        ch.plc.conc_energy = energy;
        ch.plc.conc_energy_shift = shift;
        ch.plc.last_frame_lost = true;
        return;
    }
    if !ch.plc.last_frame_lost {
        return;
    }

    let (mut energy, energy_shift) = sum_sqr_shift(&frame[..length]);
    if energy_shift > ch.plc.conc_energy_shift {
        ch.plc.conc_energy = fx::rshift(ch.plc.conc_energy, (energy_shift - ch.plc.conc_energy_shift) as u32);
    } else if energy_shift < ch.plc.conc_energy_shift {
        energy = fx::rshift(energy, (ch.plc.conc_energy_shift - energy_shift) as u32);
    }

    if energy > ch.plc.conc_energy {
        let lz = fx::clz32(ch.plc.conc_energy) - 1;
        ch.plc.conc_energy = fx::lshift(ch.plc.conc_energy, lz as u32);
        energy = fx::rshift(energy, fx::max(24 - lz, 0) as u32);

        let frac_q24 = fx::div32_16(ch.plc.conc_energy, fx::max(energy, 1));
        let mut gain_q16 = fx::lshift(fx::sqrt_approx(frac_q24), 4);
        // 斜率放大 4 倍，避免 DTX 之后的起音被漏掉
        let slope_q16 = fx::div32_16((1 << 16) - gain_q16, length as i32) << 2;

        for i in 0..length {
            frame[i] = fx::sat16(fx::smulwb(gain_q16, frame[i] as i32));
            gain_q16 += slope_q16;
            if gain_q16 > 1 << 16 {
                break;
            }
        }
    }
    ch.plc.last_frame_lost = false;
}

/// `silk_CNG`：更新噪声估计，并在丢包/DTX 时叠上舒适噪声。
pub fn cng(
    ch: &mut SilkChannel,
    ctrl: &super::silk_params::SilkDecoderControl,
    frame: &mut [i16],
    length: usize,
) {
    let order = ch.tables.lpc_order;
    if ch.fs_khz != ch.cng.fs_khz {
        ch.cng.reset(order);
        ch.cng.fs_khz = ch.fs_khz;
    }

    if ch.loss_cnt == 0 && ch.prev_signal_type == TYPE_NO_VOICE_ACTIVITY {
        // 平滑 NLSF 与增益，并把"增益最大的子帧"的激励存进 CNG 激励缓冲
        for i in 0..order {
            let d = ch.prev_nlsf_q15[i] as i32 - ch.cng.smth_nlsf_q15[i] as i32;
            ch.cng.smth_nlsf_q15[i] =
                (ch.cng.smth_nlsf_q15[i] as i32 + fx::smulwb(d, CNG_NLSF_SMTH_Q16)) as i16;
        }
        let mut max_gain_q16 = 0i32;
        let mut subfr = 0usize;
        for i in 0..ch.nb_subfr {
            if ctrl.gains_q16[i] > max_gain_q16 {
                max_gain_q16 = ctrl.gains_q16[i];
                subfr = i;
            }
        }
        let subfr_length = ch.subfr_length;
        ch.cng
            .exc_buf_q14
            .copy_within(..(ch.nb_subfr - 1) * subfr_length, subfr_length);
        ch.cng.exc_buf_q14[..subfr_length]
            .copy_from_slice(&ch.exc_q14[subfr * subfr_length..(subfr + 1) * subfr_length]);
        for i in 0..ch.nb_subfr {
            let d = ctrl.gains_q16[i] - ch.cng.smth_gain_q16;
            ch.cng.smth_gain_q16 = fx::smlawb(ch.cng.smth_gain_q16, d, CNG_GAIN_SMTH_Q16);
        }
    }

    if ch.loss_cnt != 0 {
        // ① 造激励（从缓冲里随机取，乘上平滑增益）
        let mut cng_sig_q10 = [0i32; MAX_FRAME_LENGTH + MAX_LPC_ORDER];
        let mut exc_mask = CNG_BUF_MASK_MAX;
        while exc_mask > length as i32 {
            exc_mask = fx::rshift(exc_mask, 1);
        }
        let mut seed = ch.cng.rand_seed;
        for i in 0..length {
            seed = rand(seed);
            let idx = (fx::rshift(seed, 24) & exc_mask) as usize;
            cng_sig_q10[MAX_LPC_ORDER + i] =
                fx::sat16(fx::smulww(ch.cng.exc_buf_q14[idx], ch.cng.smth_gain_q16 >> 4)) as i32;
        }
        ch.cng.rand_seed = seed;

        // ② 平滑后的 NLSF → LPC
        let a_q12 = nlsf2a(&ch.cng.smth_nlsf_q15[..order], order);

        // ③ LPC 合成，叠到隐藏信号上
        cng_sig_q10[..MAX_LPC_ORDER].copy_from_slice(&ch.cng.synth_state);
        for i in 0..length {
            let at = MAX_LPC_ORDER + i;
            let mut sum_q6 = fx::rshift(order as i32, 1);
            for j in 0..order {
                sum_q6 = fx::smlawb(sum_q6, cng_sig_q10[at - j - 1], a_q12[j] as i32);
            }
            cng_sig_q10[at] = fx::add_lshift32(cng_sig_q10[at], sum_q6, 4);
            frame[i] = fx::add_sat16(frame[i], fx::rshift_round(sum_q6, 6));
        }
        ch.cng
            .synth_state
            .copy_from_slice(&cng_sig_q10[length..length + MAX_LPC_ORDER]);
    } else {
        // 好帧 → 清掉合成状态（只清 `LPC_order` 个，不是 `MAX_LPC_ORDER` 个）
        for slot in ch.cng.synth_state.iter_mut().take(order) {
            *slot = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_sqr_shift_matches_a_naive_energy_for_small_input() {
        let x: Vec<i16> = (0..64).map(|i| ((i % 7) - 3) as i16).collect();
        let want: i32 = x.iter().map(|&v| (v as i32) * (v as i32)).sum();
        let (energy, shift) = sum_sqr_shift(&x);
        assert_eq!(shift, 0, "小信号不该触发移位");
        assert_eq!(energy, want);
    }

    #[test]
    fn sum_sqr_shift_shifts_instead_of_overflowing() {
        // 满量程方波：不右移必然溢出，参考实现靠 `shft` 把它压下来
        let x = vec![30000i16; 320];
        let (energy, shift) = sum_sqr_shift(&x);
        assert!(shift >= 2, "应当发生过右移：shift = {shift}");
        assert!(energy > 0);
        // 右移之后能量仍在 int32 里（且带两个前导零）
        assert_eq!(energy & 0xC000_0000u32 as i32, 0);
    }

    #[test]
    fn plc_reset_uses_the_frame_length() {
        let mut st = SilkPlcState::default();
        st.reset(160);
        assert_eq!(st.pitch_l_q8, 160 << 7);
        assert_eq!(st.prev_gain_q16, [65536, 65536]);
        assert_eq!((st.subfr_length, st.nb_subfr), (20, 2));
    }

    #[test]
    fn cng_reset_spreads_the_nlsf_evenly() {
        let mut st = SilkCngState::default();
        st.reset(10);
        assert_eq!(st.smth_nlsf_q15[0], 32767 / 11);
        assert_eq!(st.rand_seed, 3176576);
        assert_eq!(st.smth_gain_q16, 0);
    }

    #[test]
    fn rand_advances_the_seed() {
        // `silk_RAND(0)` 的规范值：907633515 + 0
        assert_eq!(rand(0), 907633515);
        assert_eq!(rand(907633515), -1653660526);
    }
}
