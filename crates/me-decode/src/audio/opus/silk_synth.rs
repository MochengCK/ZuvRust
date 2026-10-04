//! SILK 的**核心合成**与立体声解混（参考实现 `silk/decode_core.c`、
//! `silk/LPC_analysis_filter.c`、`silk/stereo_MS_to_LR.c`）。
//!
//! 这一层把上一层的"解码器控制参数 + 激励脉冲"变成**内部采样率的 PCM**：
//!
//! ```text
//! 激励重建      脉冲 → Q14 激励（含量化偏移 + 种子决定的符号翻转）
//! LTP（浊音）   用 sLTP_Q15 里最近 lag 个样点做 5 抽头长时预测，加到激励上
//! LPC 合成      用 PredCoef_Q12（前半帧/后半帧各一套）做 10/16 阶短时预测
//! 增益          按子帧增益缩放到 Q0，饱和到 int16
//! ```
//!
//! ## 为什么这层也必须逐位相同
//!
//! 和参数层一样是**纯整数**（`SigProc_FIX.h` 的定点宏在浮点构建下也是同一份），
//! 所有运算要么是 Q 格式乘加、要么是带明确截断/取半的移位。参考实现对拍口径就是
//! **逐样本相同**（不是 CELT 那种"差几个量化步长"）。
//!
//! ## 三个最容易踩的地方
//!
//! 1. **`sLTP_Q15` 的"重白化"（re-whitening）**：每个子帧（浊音帧在 k==0，或者
//!    后半帧开始的 k==2 且这一帧允许 NLSF 插值时）都要用**当前的 LPC 系数**把
//!    `outBuf` 里的历史重新滤一遍 —— 因为历史是用旧系数合成的。索引
//!    `start_idx = ltp_mem_length - lag - LPC_order - LTP_ORDER/2` 很容易写错，
//!    而且 `sLTP_Q15` 是**逆序**填的（`sLTP_buf_idx - i - 1`）。
//! 2. **增益调整因子**：子帧增益和上一子帧不同时，`sLPC_Q14` 与 `sLTP_Q15` 的历史
//!    都要按 `gain_adj_Q16` 缩放 —— 否则增益跳变处会有咔哒声（这也是对拍最容易露馅的点）。
//! 3. **`LTP_scale_Q14` 只在 k==0 施加**（"降低包间依赖"），施加完立刻乘进
//!    `inv_gain_Q31`，**这个缩放后的值只用于那一次 LTP 历史填充**。
//!
//! ## 内部状态
//!
//! 跨帧状态全部挂在 [`SilkChannel`] 上：`out_buf`（LTP 历史）、`s_lpc_q14_buf`
//! （LPC 滤波器状态）、`lag_prev`、`prev_gain_q16`、`prev_signal_type`。
//! 立体声解混的 `sMid`/`sSide`/`pred_prev_Q13` 在 [`SilkStereoState`] 里。

use super::silk::{
    SilkChannel, MAX_FRAME_LENGTH, MAX_LPC_ORDER, MAX_SUB_FRAME_LENGTH, LTP_ORDER, TYPE_VOICED,
};
use super::silk_fix as fx;
use super::silk_params::SilkDecoderControl;

/// 量化偏移（`tables_other.c` 的 `silk_Quantization_Offsets_Q10`）：
/// 行 = `signalType >> 1`（0 清音 / 1 浊音），列 = 量化偏移类型。
const QUANTIZATION_OFFSETS_Q10: [[i32; 2]; 2] = [[100, 240], [32, 100]];
/// 非零脉冲的"电平微调"（`QUANT_LEVEL_ADJUST_Q10`）。
const QUANT_LEVEL_ADJUST_Q10: i32 = 80;
/// 立体声预测的内插长度（ms，`STEREO_INTERP_LEN_MS`）。
const STEREO_INTERP_LEN_MS: usize = 8;
/// 丢包后从浊音切到清音时给 LTP 的替代系数（`SILK_FIX_CONST(0.25, 14)`）。
const LTP_FALLBACK_COEF_Q14: i16 = 4096;
/// `silk_RAND`：种子推进（LCG，32 位回绕）。
///
/// `silk_MLA_ovflw(907633515, seed, 196314165)` = `907633515 + seed * 196314165`。
/// 激励的**符号**由它决定（每个样点先推一次种子，负数就翻转激励符号），
/// 所以这几位错了整帧会听起来像噪声、但能量是对的 —— 对拍时表现为"逐样本差但 rms 接近"。
#[inline]
pub(crate) fn rand(seed: i32) -> i32 {
    907633515i32.wrapping_add((seed as u32).wrapping_mul(196314165u32) as i32)
}

/// `silk_LPC_analysis_filter`：用 LPC 系数把信号"白化"（分析滤波）。
///
/// 参考实现明确"状态在内部、起点恒为零"，且**前 d 个输出恒为 0** ——
/// 这不是省略，而是"没有足够历史就不预测"的定义。
///
/// `input` 的长度就是 `len`；`out` 至少要 `len` 长。两个切片来自不同缓冲区
/// （`sLTP` 与 `outBuf`），所以不需要处理别名。
pub fn lpc_analysis_filter(out: &mut [i16], input: &[i16], b: &[i16]) {
    let len = input.len();
    let d = b.len();
    debug_assert!(out.len() >= len, "输出缓冲区比输入短");
    debug_assert!(d >= 6 && d % 2 == 0 && d <= len);
    for ix in d..len {
        // 前 6 个抽头参考实现是展开的，后面按 2 个一组；两者完全等价，这里统一成循环。
        // 关键是**加法允许回绕**（两次回绕会互相抵消，参考实现靠这个行为容忍非法码流）。
        let mut out32_q12 = fx::smulbb(input[ix - 1] as i32, b[0] as i32);
        for j in 1..d {
            out32_q12 = fx::smlabb_ovflw(out32_q12, input[ix - 1 - j] as i32, b[j] as i32);
        }
        let out32_q12 = fx::sub32_ovflw(fx::lshift(input[ix] as i32, 12), out32_q12);
        out[ix] = fx::sat16(fx::rshift_round(out32_q12, 12));
    }
    for v in out.iter_mut().take(d) {
        *v = 0;
    }
}

/// 立体声解混（MS → LR）的跨帧状态（参考实现 `stereo_dec_state`）。
#[derive(Debug, Clone, Copy, Default)]
pub struct SilkStereoState {
    /// 上一帧末尾用过的两个预测系数（本帧前 8 ms 要插值过渡过去）。
    pub pred_prev_q13: [i32; 2],
    /// 中间声道的最后两个样点（本帧解混要跟前一帧接上）。
    pub s_mid: [i16; 2],
    /// 侧声道的最后两个样点。
    pub s_side: [i16; 2],
}

/// `silk_stereo_MS_to_LR`：把"中间/侧"两条信号还原成左右声道。
///
/// `x1`/`x2` 是**带 2 个前导样点的缓冲区**（与参考实现一致）：
/// - 下标 0/1 在进入时被写成上一帧保存的 `sMid`/`sSide`；
/// - 解混结果写在 `x1[1..=frame_length]` / `x2[1..=frame_length]`；
/// - 离开时把 `x1[frame_length..frame_length+2]`（还没被覆盖的中间声道样点）
///   存进 `s_mid` 供下一帧用。
///
/// 第 1 个参数（`pred_q13`）是**本帧**的预测系数，前 8 ms 要从上一帧的值线性插值过去
/// —— 少了这段插值，帧边界会有可闻的"扑"声，对拍则在每帧头 8 ms 全错。
pub fn stereo_ms_to_lr(
    state: &mut SilkStereoState,
    x1: &mut [i16],
    x2: &mut [i16],
    pred_q13: [i32; 2],
    fs_khz: u32,
    frame_length: usize,
) {
    // 缓冲
    x1[0] = state.s_mid[0];
    x1[1] = state.s_mid[1];
    x2[0] = state.s_side[0];
    x2[1] = state.s_side[1];
    state.s_mid = [x1[frame_length], x1[frame_length + 1]];
    state.s_side = [x2[frame_length], x2[frame_length + 1]];

    // 预测系数：前 STEREO_INTERP_LEN_MS 毫秒从上一帧插值过来
    let mut pred0_q13 = state.pred_prev_q13[0];
    let mut pred1_q13 = state.pred_prev_q13[1];
    let interp_len = STEREO_INTERP_LEN_MS * fs_khz as usize;
    let denom_q16 = fx::div32_16(1 << 16, interp_len as i32);
    let delta0_q13 = fx::rshift_round(
        fx::smulbb(pred_q13[0] - state.pred_prev_q13[0], denom_q16),
        16,
    );
    let delta1_q13 = fx::rshift_round(
        fx::smulbb(pred_q13[1] - state.pred_prev_q13[1], denom_q16),
        16,
    );
    for n in 0..interp_len {
        pred0_q13 += delta0_q13;
        pred1_q13 += delta1_q13;
        let sum = fx::lshift(
            (x1[n] as i32 + x1[n + 2] as i32) + fx::lshift(x1[n + 1] as i32, 1),
            9,
        );
        let sum = fx::smlawb(fx::lshift(x2[n + 1] as i32, 8), sum, pred0_q13);
        let sum = fx::smlawb(sum, fx::lshift(x1[n + 1] as i32, 11), pred1_q13);
        x2[n + 1] = fx::sat16(fx::rshift_round(sum, 8));
    }
    let pred0_q13 = pred_q13[0];
    let pred1_q13 = pred_q13[1];
    for n in interp_len..frame_length {
        let sum = fx::lshift(
            (x1[n] as i32 + x1[n + 2] as i32) + fx::lshift(x1[n + 1] as i32, 1),
            9,
        );
        let sum = fx::smlawb(fx::lshift(x2[n + 1] as i32, 8), sum, pred0_q13);
        let sum = fx::smlawb(sum, fx::lshift(x1[n + 1] as i32, 11), pred1_q13);
        x2[n + 1] = fx::sat16(fx::rshift_round(sum, 8));
    }
    state.pred_prev_q13 = pred_q13;

    // 中间/侧 → 左/右
    for n in 0..frame_length {
        let sum = x1[n + 1] as i32 + x2[n + 1] as i32;
        let diff = x1[n + 1] as i32 - x2[n + 1] as i32;
        x1[n + 1] = fx::sat16(sum);
        x2[n + 1] = fx::sat16(diff);
    }
}

/// 若某一侧是"只编了中间声道"，参考实现会直接把中间声道复制给两侧。
pub fn stereo_mid_only_copy(state: &mut SilkStereoState, x1: &mut [i16], frame_length: usize) {
    // 参考实现（`dec_API.c` 的单声道/只中间声道分支）：同样维护两个前导样点，
    // 并把 x1[frame_length..frame_length+2] 当作新的 sMid。
    x1[0] = state.s_mid[0];
    x1[1] = state.s_mid[1];
    state.s_mid = [x1[frame_length], x1[frame_length + 1]];
}

/// `silk_decode_core`：激励重建 + LTP + LPC 合成 → 内部采样率 PCM。
///
/// `pulses` 是熵层解出的激励脉冲（长度 = `frame_length`），
/// `xq` 是输出（长度 = `frame_length`，Q0 的 int16）。
pub fn decode_core(
    ch: &mut SilkChannel,
    control: &mut SilkDecoderControl,
    pulses: &[i32],
    xq: &mut [i16],
) {
    let frame_length = ch.frame_length;
    let subfr_length = ch.subfr_length;
    let lpc_order = ch.tables.lpc_order;
    let ltp_mem_length = ch.ltp_mem_length;
    let nb_subfr = ch.nb_subfr;
    debug_assert!(pulses.len() >= frame_length && xq.len() >= frame_length);
    debug_assert!(ch.prev_gain_q16 != 0, "增益调整因子要拿它做分母");

    let offset_q10 = QUANTIZATION_OFFSETS_Q10[(ch.indices.signal_type >> 1) as usize]
        [ch.indices.quant_offset_type as usize];
    let nlsf_interpolation_flag = (ch.indices.nlsf_interp_coef_q2 as i32) < 4;

    // ① 激励：脉冲 → Q14，含"电平微调"、量化偏移与种子决定的符号
    let mut rand_seed = ch.indices.seed as i32;
    for i in 0..frame_length {
        rand_seed = rand(rand_seed);
        let mut e = fx::lshift(pulses[i], 14);
        if e > 0 {
            e -= QUANT_LEVEL_ADJUST_Q10 << 4;
        } else if e < 0 {
            e += QUANT_LEVEL_ADJUST_Q10 << 4;
        }
        e += offset_q10 << 4;
        if rand_seed < 0 {
            e = -e;
        }
        ch.exc_q14[i] = e;
        rand_seed = rand_seed.wrapping_add(pulses[i]);
    }

    // ② 逐子帧：LTP → LPC
    let mut s_lpc_q14 = [0i32; MAX_SUB_FRAME_LENGTH + MAX_LPC_ORDER];
    s_lpc_q14[..MAX_LPC_ORDER].copy_from_slice(&ch.s_lpc_q14_buf);
    let mut s_ltp = [0i16; MAX_FRAME_LENGTH];
    let mut s_ltp_q15 = [0i32; 2 * MAX_FRAME_LENGTH];
    let mut s_ltp_buf_idx = ltp_mem_length;
    let mut res_q14 = [0i32; MAX_SUB_FRAME_LENGTH];
    let mut lag = 0usize;

    for k in 0..nb_subfr {
        let a_q12 = control.pred_coef_q12[k >> 1];
        let mut b_q14 = [0i16; LTP_ORDER];
        b_q14.copy_from_slice(&control.ltp_coef_q14[k * LTP_ORDER..(k + 1) * LTP_ORDER]);
        let mut signal_type = ch.indices.signal_type as i32;

        let gain_q10 = fx::rshift(control.gains_q16[k], 6);
        let mut inv_gain_q31 = fx::inverse32_varq(control.gains_q16[k], 47);

        // 增益调整因子：子帧增益变化时，所有的 Q14 历史都要跟着缩放
        let gain_adj_q16 = if control.gains_q16[k] != ch.prev_gain_q16 {
            let adj = fx::div32_varq(ch.prev_gain_q16, control.gains_q16[k], 16);
            for v in s_lpc_q14.iter_mut().take(MAX_LPC_ORDER) {
                *v = fx::smulww(adj, *v);
            }
            adj
        } else {
            1 << 16
        };
        ch.prev_gain_q16 = control.gains_q16[k];

        // 丢包后"浊音 → 清音"的过渡：前两个子帧强制按浊音处理，避免突然断掉
        if ch.loss_cnt != 0
            && ch.prev_signal_type == TYPE_VOICED
            && signal_type != TYPE_VOICED
            && k < 4 /* MAX_NB_SUBFR */ / 2
        {
            b_q14 = [0; LTP_ORDER];
            b_q14[LTP_ORDER / 2] = LTP_FALLBACK_COEF_Q14;
            signal_type = TYPE_VOICED;
            control.pitch_l[k] = ch.lag_prev;
        }

        if signal_type == TYPE_VOICED {
            lag = control.pitch_l[k] as usize;

            if k == 0 || (k == 2 && nlsf_interpolation_flag) {
                // 重白化：用**当前**的 LPC 系数把 outBuf 里的历史重新滤一遍
                let start_idx = ch.ltp_mem_length - lag - lpc_order - LTP_ORDER / 2;
                debug_assert!(start_idx > 0, "重白化的起点越界（lag 太大）");
                if k == 2 {
                    // 后半帧的重白化要用到本帧已经合成出的前两个子帧
                    ch.out_buf[ltp_mem_length..ltp_mem_length + 2 * subfr_length]
                        .copy_from_slice(&xq[..2 * subfr_length]);
                }
                lpc_analysis_filter(
                    &mut s_ltp[start_idx..],
                    &ch.out_buf[start_idx + k * subfr_length
                        .. ltp_mem_length + k * subfr_length],
                    &a_q12[..lpc_order],
                );
                if k == 0 {
                    // LTP 降尺度：减少包间依赖（乘 LTP_scale 之后再左移 2 位）
                    inv_gain_q31 = fx::lshift(fx::smulwb(inv_gain_q31, control.ltp_scale_q14), 2);
                }
                // 注意是**逆序**填：sLTP_Q15 的最后 lag+2 个样点
                for i in 0..lag + LTP_ORDER / 2 {
                    s_ltp_q15[s_ltp_buf_idx - i - 1] =
                        fx::smulwb(inv_gain_q31, s_ltp[ltp_mem_length - i - 1] as i32);
                }
            } else if gain_adj_q16 != 1 << 16 {
                for i in 0..lag + LTP_ORDER / 2 {
                    s_ltp_q15[s_ltp_buf_idx - i - 1] =
                        fx::smulww(gain_adj_q16, s_ltp_q15[s_ltp_buf_idx - i - 1]);
                }
            }
        }

        // 长时（基音）预测 + 更新 sLTP 状态
        let pexc = k * subfr_length;
        if signal_type == TYPE_VOICED {
            let mut pred_lag_ptr = s_ltp_buf_idx - lag + LTP_ORDER / 2;
            for i in 0..subfr_length {
                // 初值 2 是参考实现刻意留的偏置：`silk_SMLAWB` 恒向下取整
                let mut ltp_pred_q13 = 2i32;
                for j in 0..LTP_ORDER {
                    ltp_pred_q13 =
                        fx::smlawb(ltp_pred_q13, s_ltp_q15[pred_lag_ptr - j], b_q14[j] as i32);
                }
                pred_lag_ptr += 1;
                let v = ch.exc_q14[pexc + i].wrapping_add(fx::lshift(ltp_pred_q13, 1));
                res_q14[i] = v;
                s_ltp_q15[s_ltp_buf_idx] = fx::lshift(v, 1);
                s_ltp_buf_idx += 1;
            }
        } else {
            res_q14[..subfr_length].copy_from_slice(&ch.exc_q14[pexc..pexc + subfr_length]);
        }

        // 短时（LPC）合成
        for i in 0..subfr_length {
            let mut lpc_pred_q10 = fx::rshift(lpc_order as i32, 1);
            for j in 0..lpc_order {
                lpc_pred_q10 = fx::smlawb(
                    lpc_pred_q10,
                    s_lpc_q14[MAX_LPC_ORDER + i - 1 - j],
                    a_q12[j] as i32,
                );
            }
            let s = res_q14[i].wrapping_add(fx::lshift(lpc_pred_q10, 4));
            s_lpc_q14[MAX_LPC_ORDER + i] = s;
            xq[k * subfr_length + i] =
                fx::sat16(fx::rshift_round(fx::smulww(s, gain_q10), 8));
        }

        // 滤波器状态前移一个子帧
        s_lpc_q14.copy_within(subfr_length..subfr_length + MAX_LPC_ORDER, 0);
    }

    // ③ 保存 LPC 状态
    ch.s_lpc_q14_buf.copy_from_slice(&s_lpc_q14[..MAX_LPC_ORDER]);
}

/// `silk_decode_frame` 的收尾：把本帧推进 `out_buf`（LTP 历史），并更新跨帧状态。
///
/// 参考实现里"移窗"这一步在 16 kHz 是空操作（`ltp_mem_length == frame_length`），
/// 但在 8 kHz 的 20 ms 帧下**真的会移动**（160 vs 320 的差别）—— 写死"直接覆盖"会在
/// NB 素材上从第二帧起全错。
///
/// `lagPrev` 取自**本帧最后一个子帧的 pitch lag**（清音帧在 `decode_parameters`
/// 里已被清零，与参考实现一致）。
///
/// **注意两件事**（都是参考实现 `silk_decode_frame` 的语义）：
///
/// 1. 推进 `outBuf` 用的是**衔接与舒适噪声之前**的信号 —— 也就是说这个函数必须
///    在 `plc_glue_frames` / `cng` **之前**把 `xq` 拷进历史，之后再让那两个函数
///    就地改 `xq`（它们改的是"送给输出"的那一份，不是 LTP 历史）。
/// 2. `loss_cnt` 归零发生在 `glue_frames` **之前**（正常帧），因为 `glue_frames`
///    正是靠它来区分"这一帧是藏的"还是"这一帧是接收到的"。
pub fn finish_frame(ch: &mut SilkChannel, control: &SilkDecoderControl, xq: &mut [i16], lost: bool) {
    let fl = ch.frame_length;
    let mv_len = ch.ltp_mem_length - fl;
    if mv_len > 0 {
        ch.out_buf.copy_within(fl..fl + mv_len, 0);
    }
    ch.out_buf[mv_len..mv_len + fl].copy_from_slice(&xq[..fl]);

    if !lost {
        ch.loss_cnt = 0;
    }

    // 隐藏帧 ↔ 好帧的能量衔接（就地改 xq）
    super::silk_plc::plc_glue_frames(ch, xq, fl);
    // 舒适噪声（丢帧/DTX 时叠加到 xq 上）
    super::silk_plc::cng(ch, control, xq, fl);

    ch.lag_prev = control.pitch_l[ch.nb_subfr - 1];
}

/// 立体声解混：把两条中间/侧声道缓冲区就地变成左右声道。
///
/// 这是一层薄封装，存在的意义只是让调用点读起来与参考实现的 `silk_Decode` 一致。
pub fn finish_stereo(
    state: &mut SilkStereoState,
    mid: &mut [i16],
    side: &mut [i16],
    pred_q13: [i32; 2],
    fs_khz: u32,
    frame_length: usize,
) {
    stereo_ms_to_lr(state, mid, side, pred_q13, fs_khz, frame_length);
}

#[cfg(test)]
mod tests {
    use super::super::silk::MAX_FRAME_LENGTH;
    use super::*;

    #[test]
    fn rand_matches_the_reference_lcg() {
        // silk_RAND(0) = 907633515 + 0 = 907633515
        assert_eq!(rand(0), 907633515);
        // 手工按 32 位回绕算第二步：907633515 + 907633515*196314165 (mod 2^32)
        let step2 = 907633515i64
            .wrapping_add((907633515i64.wrapping_mul(196314165i64)) & 0xFFFF_FFFF)
            & 0xFFFF_FFFF;
        assert_eq!(rand(907633515) as i64, step2 as i32 as i64);
    }

    #[test]
    fn lpc_analysis_filter_zeroes_the_first_order_samples() {
        // 阶数 6（<10 也允许，参考实现只要求 d>=6 且为偶数）
        let input: Vec<i16> = (0..32).map(|i| ((i * 100) % 1000) as i16).collect();
        let b = [100i16, -50, 25, -12, 6, -3];
        let mut out = [0i16; 32];
        lpc_analysis_filter(&mut out, &input, &b);
        assert_eq!(&out[..6], &[0; 6], "前 d 个样点必须是 0（没有历史就不预测）");
        // 全零系数 → 输出就是输入本身（只是延迟了 d 个样点，前 d 个为 0）
        let zero = [0i16; 6];
        let mut out2 = [0i16; 32];
        lpc_analysis_filter(&mut out2, &input, &zero);
        assert_eq!(&out2[6..], &input[6..]);
    }

    #[test]
    fn stereo_ms_to_lr_uses_the_interpolated_predictor() {
        // 用真实的 20 ms @ 16 kHz 帧长（320）—— 内插长度是 8 ms = 128 个样点，
        // 帧长比它短本来就不合法（这里第一次就是拿 32 试的，越界暴露了这点）。
        const L: usize = 320;
        let mut st = SilkStereoState::default();
        let mut x1 = [0i16; L + 2];
        let mut x2 = [0i16; L + 2];
        stereo_ms_to_lr(&mut st, &mut x1, &mut x2, [0, 0], 16, L);
        assert!(x1[1..=L].iter().all(|&v| v == 0) && x2[1..=L].iter().all(|&v| v == 0));
        assert_eq!(st.pred_prev_q13, [0, 0]);

        // 预测系数从 0 插值到 8192（Q13 的 1.0）：中间声道恒为 1000、侧声道为 0 时，
        // 解混出来的左声道（mid+side）应当从 ~1000 爬到 ~2000、右声道（mid-side）
        // 从 ~1000 降到 ~0。注意下标 0/1 是**上一帧留下的两个样点**（会被函数覆盖成
        // `s_mid`），真正的中间声道数据从下标 2 开始。
        let mut st = SilkStereoState::default();
        let mut x1 = [1000i16; L + 2];
        x1[..2].copy_from_slice(&[0, 0]);
        let mut x2 = [0i16; L + 2];
        stereo_ms_to_lr(&mut st, &mut x1, &mut x2, [8192, 0], 16, L);
        assert_eq!(st.pred_prev_q13, [8192, 0]);
        assert!(x1[3] < x1[L], "左声道帧头被插值削弱：{} vs {}", x1[3], x1[L]);
        assert!(x2[3] > x2[L], "右声道帧头被插值削弱：{} vs {}", x2[3], x2[L]);
        // 插值只作用于开头 8 ms（128 个样点），之后预测系数取满值
        assert_eq!(x1[128], x1[L], "8 ms 之后应当用满预测系数");
    }

    #[test]
    fn ltp_buffer_size_matches_the_reference_struct() {
        // outBuf 在参考实现里是 MAX_FRAME_LENGTH + 2*MAX_SUB_FRAME_LENGTH（480）
        let ch_len = MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH;
        assert_eq!(ch_len, 480);
    }
}
