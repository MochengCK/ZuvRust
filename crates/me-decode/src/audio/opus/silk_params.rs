//! SILK 的**参数重建**（参考实现 `silk_decode_parameters` 及其调用的一串函数）。
//!
//! 输入是熵层解出来的量化索引（[`super::silk::SideInfoIndices`]），输出一帧合成
//! 要用的解码器控制参数：
//!
//! ```text
//! GainsIndices  → silk_gains_dequant     → Gains_Q16[4]（子帧增益，对数域均匀量化）
//! NLSFIndices   → silk_NLSF_decode       → NLSF_Q15（一级码本 + 二级残差 + 权重）
//!               → silk_NLSF_stabilize    → 间隔合法、有序
//!               → silk_NLSF2A            → PredCoef_Q12（LPC 系数，含稳定性保护）
//! lagIndex/contourIndex → silk_decode_pitch → pitchL[4]
//! PERIndex/LTPIndex     → LTP 码本         → LTPCoef_Q14[20]
//! LTP_scaleIndex        → LTP_scale_Q14
//! ```
//!
//! ## 为什么这一层也要"逐位相同"
//!
//! 全是整数运算（近似也都写成整数多项式），所以对拍口径和熵层一样是逐位相同。
//! 两个容易写错的地方：
//!
//! - **NLSF → LPC 的稳定化**：`silk_NLSF2A` 在系数超界时会用带宽扩展（`bwexpander_32`）
//!   反复软化（最多 16 轮），每轮都要重算多项式系数 —— 少一轮，`PredCoef_Q12`
//!   就会有几位不一样。
//! - **增益的"双倍步长"**：增益增量在接近上限时步长会翻倍（为了能爬到最后一级），
//!   条件编码与非条件编码两条路径都要照抄。
//!
//! ## 与状态的关系
//!
//! 三个量是跨帧状态：`LastGainIndex`（增益增量的基准）、`prevNLSF_Q15`（插值用）、
//! `first_frame_after_reset`（复位后禁用 NLSF 插值）。采样率切换（`set_fs`）时
//! `LastGainIndex` 复位成 10、`first_frame_after_reset` 置 1。

use me_core::Result;

use super::silk::{SilkChannel, LTP_ORDER, MAX_LPC_ORDER, MAX_NB_SUBFR, TYPE_VOICED};
use super::silk_fix as fx;
use super::silk_tables as t;

/// 增益量化范围（参考实现 `define.h`）。
const MIN_QGAIN_DB: i32 = 2;
const MAX_QGAIN_DB: i32 = 88;
const N_LEVELS_QGAIN: i32 = 64;
const MIN_DELTA_GAIN_QUANT: i32 = -4;
const MAX_DELTA_GAIN_QUANT: i32 = 36;

/// `gain_quant.c` 里的三个换算常量（整数除法顺序照抄）。
const GAIN_OFFSET: i32 = (MIN_QGAIN_DB * 128) / 6 + 16 * 128;
const GAIN_INV_SCALE_Q16: i32 =
    (65536 * (((MAX_QGAIN_DB - MIN_QGAIN_DB) * 128) / 6)) / (N_LEVELS_QGAIN - 1);

/// NLSF 的权重指数（`define.h` 的 `NLSF_W_Q`）。
const NLSF_W_Q: i32 = 2;
/// `SILK_FIX_CONST(0.1, 10)`：二级残差的"电平微调"。
const NLSF_QUANT_LEVEL_ADJ_Q10: i32 = 102;
/// LSF 余弦表的大小（`LSF_COS_TAB_SZ_FIX`）。
const LSF_COS_TAB_SZ_FIX: i32 = 128;
/// NLSF→LPC 时多项式用的 Q 域（`NLSF2A.c` 里的 `QA`）。
const QA: u32 = 16;
/// `LPC_inv_pred_gain.c` 里**另有一个** `QA`，值是 **24**，与该文件无关。
///
/// 这两个 QA 同名但不同值，是参考实现里最容易踩的一处：稳定性判据全程在 Q24 域里做
/// （系数左移 12 位），把它错当成 NLSF2A 的 QA=16（左移 4 位）会让系数小 256 倍 ——
/// 于是"预测增益倒数"永远接近 1、**稳定化被整段跳过**，LPC 系数只差最后几位。
/// 症状极隐蔽：绝大多数帧都对，少数帧差 ±1（见 `nlsf2a` 的稳定化环）。
const QA_LPC_INV_GAIN: u32 = 24;
/// 稳定化最多迭代轮数（`MAX_LPC_STABILIZE_ITERATIONS`）。
const MAX_LPC_STABILIZE_ITERATIONS: i32 = 16;
/// `SILK_FIX_CONST(1.0 / MAX_PREDICTION_POWER_GAIN, 30)`（MAX_PREDICTION_POWER_GAIN = 1e4）。
const INV_PRED_GAIN_THRESHOLD_Q30: i32 = 107374;
/// `LPC_inv_pred_gain.c` 里的系数上限 `SILK_FIX_CONST(0.99975, 24)`。
const A_LIMIT_QA24: i32 = 16773022;
/// 丢包后对 LPC 做带宽扩展的强度 `BWE_AFTER_LOSS_Q16`。
const BWE_AFTER_LOSS_Q16: i32 = 63570;

/// NLSF→LPC 的系数重排（照抄参考实现注释：这个顺序数值精度最好）。
const ORDERING16: [usize; 16] = [0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1];
const ORDERING10: [usize; 10] = [0, 9, 6, 3, 4, 5, 8, 1, 2, 7];

/// 一帧重建出来的解码器控制参数（参考实现 `silk_decoder_control`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SilkDecoderControl {
    pub pitch_l: [i32; MAX_NB_SUBFR],
    pub gains_q16: [i32; MAX_NB_SUBFR],
    /// `[0]` 是前半帧（插值出来的），`[1]` 是后半帧（本帧的 NLSF）。
    pub pred_coef_q12: [[i16; MAX_LPC_ORDER]; 2],
    pub ltp_coef_q14: [i16; LTP_ORDER * MAX_NB_SUBFR],
    pub ltp_scale_q14: i32,
}

impl Default for SilkDecoderControl {
    fn default() -> Self {
        Self {
            pitch_l: [0; MAX_NB_SUBFR],
            gains_q16: [0; MAX_NB_SUBFR],
            pred_coef_q12: [[0; MAX_LPC_ORDER]; 2],
            ltp_coef_q14: [0; LTP_ORDER * MAX_NB_SUBFR],
            ltp_scale_q14: 0,
        }
    }
}

/// `silk_log2lin`：2^() 的定点近似（对数域 → 线性域，增益反量化用它）。
pub fn log2lin(in_log_q7: i32) -> i32 {
    if in_log_q7 < 0 {
        return 0;
    }
    let mut out = fx::lshift(1, fx::rshift(in_log_q7, 7) as u32);
    let frac_q7 = in_log_q7 & 0x7F;
    let curve = fx::smlawb(frac_q7, fx::smulbb(frac_q7, 128 - frac_q7), -174);
    if in_log_q7 < 2048 {
        // 小值：抛物线近似直接作用在输出上
        out = fx::lshift(out, 0).wrapping_add(out.wrapping_mul(curve).wrapping_shr(7));
    } else {
        out = out.wrapping_add((out >> 7).wrapping_mul(curve));
    }
    out
}

/// `silk_gains_dequant`：增益反量化（对数域均匀量化 + 相邻帧的滞回）。
///
/// `prev_ind` 是跨帧状态（`LastGainIndex`），会就地更新。
pub fn gains_dequant(
    gains_q16: &mut [i32],
    ind: &[i8],
    prev_ind: &mut i32,
    conditional: bool,
    nb_subfr: usize,
) {
    for k in 0..nb_subfr {
        if k == 0 && !conditional {
            // 独立编码：第一级的索引不允许比上一帧低超过 16 档
            *prev_ind = fx::max(ind[k] as i32, *prev_ind - 16);
        } else {
            let ind_tmp = ind[k] as i32 + MIN_DELTA_GAIN_QUANT;
            // 接近上限时步长翻倍（这样才够得着最后一级增益）
            let double_step_size_threshold = 2 * MAX_DELTA_GAIN_QUANT - N_LEVELS_QGAIN + *prev_ind;
            if ind_tmp > double_step_size_threshold {
                *prev_ind += fx::lshift(ind_tmp, 1) - double_step_size_threshold;
            } else {
                *prev_ind += ind_tmp;
            }
        }
        *prev_ind = fx::limit(*prev_ind, 0, N_LEVELS_QGAIN - 1);
        gains_q16[k] = log2lin(fx::min(
            fx::smulwb(GAIN_INV_SCALE_Q16, *prev_ind) + GAIN_OFFSET,
            3967, // 31 in Q7
        ));
    }
}

/// `silk_NLSF_residual_dequant`：二级残差的预测反量化（从高阶往低阶递推）。
fn nlsf_residual_dequant(
    x_q10: &mut [i16],
    indices: &[i8],
    pred_coef_q8: &[u8],
    quant_step_size_q16: i32,
) {
    let order = pred_coef_q8.len();
    let mut out_q10 = 0i32;
    for i in (0..order).rev() {
        let pred_q10 = fx::rshift(fx::smulbb(out_q10, pred_coef_q8[i] as i32), 8);
        let mut t = fx::lshift(indices[i] as i32, 10);
        if t > 0 {
            t -= NLSF_QUANT_LEVEL_ADJ_Q10;
        } else if t < 0 {
            t += NLSF_QUANT_LEVEL_ADJ_Q10;
        }
        out_q10 = fx::smlawb(pred_q10, t, quant_step_size_q16);
        x_q10[i] = out_q10 as i16;
    }
}

/// `silk_NLSF_VQ_weights_laroia`：NLSF 的权重（按相邻间隔的倒数）。
fn nlsf_vq_weights_laroia(nlsf_q15: &[i16]) -> [i16; MAX_LPC_ORDER] {
    let d = nlsf_q15.len();
    let mut w = [0i16; MAX_LPC_ORDER];
    let mut tmp2_int;
    let mut tmp1_int = fx::max(nlsf_q15[0] as i32, 1);
    tmp1_int = fx::div32_16(1 << (15 + NLSF_W_Q), tmp1_int);
    tmp2_int = fx::max(nlsf_q15[1] as i32 - nlsf_q15[0] as i32, 1);
    tmp2_int = fx::div32_16(1 << (15 + NLSF_W_Q), tmp2_int);
    w[0] = fx::min(tmp1_int + tmp2_int, i16::MAX as i32) as i16;

    let mut k = 1;
    while k < d - 1 {
        tmp1_int = fx::max(nlsf_q15[k + 1] as i32 - nlsf_q15[k] as i32, 1);
        tmp1_int = fx::div32_16(1 << (15 + NLSF_W_Q), tmp1_int);
        w[k] = fx::min(tmp1_int + tmp2_int, i16::MAX as i32) as i16;

        tmp2_int = fx::max(nlsf_q15[k + 2] as i32 - nlsf_q15[k + 1] as i32, 1);
        tmp2_int = fx::div32_16(1 << (15 + NLSF_W_Q), tmp2_int);
        w[k + 1] = fx::min(tmp1_int + tmp2_int, i16::MAX as i32) as i16;
        k += 2;
    }

    tmp1_int = fx::max((1 << 15) - nlsf_q15[d - 1] as i32, 1);
    tmp1_int = fx::div32_16(1 << (15 + NLSF_W_Q), tmp1_int);
    w[d - 1] = fx::min(tmp1_int + tmp2_int, i16::MAX as i32) as i16;
    w
}

/// `silk_NLSF_stabilize`：把 NLSF 拉开到合法间隔内（保持排序、尽量少动）。
pub fn nlsf_stabilize(nlsf_q15: &mut [i16], delta_min_q15: &[i16]) {
    let l = nlsf_q15.len();
    for _ in 0..20 {
        // 找出最"挤"的那一处（含两端与边界的距离）
        let mut min_diff = nlsf_q15[0] as i32 - delta_min_q15[0] as i32;
        let mut idx = 0usize;
        for i in 1..l {
            let diff = nlsf_q15[i] as i32 - (nlsf_q15[i - 1] as i32 + delta_min_q15[i] as i32);
            if diff < min_diff {
                min_diff = diff;
                idx = i;
            }
        }
        let diff = (1 << 15) - (nlsf_q15[l - 1] as i32 + delta_min_q15[l] as i32);
        if diff < min_diff {
            min_diff = diff;
            idx = l;
        }
        if min_diff >= 0 {
            return;
        }
        if idx == 0 {
            nlsf_q15[0] = delta_min_q15[0];
        } else if idx == l {
            nlsf_q15[l - 1] = ((1 << 15) - delta_min_q15[l] as i32) as i16;
        } else {
            // 保持"中心频率"不变，把相邻两个拉开
            let mut min_center = 0i32;
            for k in 0..idx {
                min_center += delta_min_q15[k] as i32;
            }
            min_center += fx::rshift(delta_min_q15[idx] as i32, 1);
            let mut max_center = 1 << 15;
            for k in (idx + 1..=l).rev() {
                max_center -= delta_min_q15[k] as i32;
            }
            max_center -= fx::rshift(delta_min_q15[idx] as i32, 1);
            let center = fx::limit(
                fx::rshift_round(nlsf_q15[idx - 1] as i32 + nlsf_q15[idx] as i32, 1),
                min_center,
                max_center,
            ) as i16;
            nlsf_q15[idx - 1] =
                (center as i32 - fx::rshift(delta_min_q15[idx] as i32, 1)) as i16;
            nlsf_q15[idx] =
                (nlsf_q15[idx - 1] as i32 + delta_min_q15[idx] as i32) as i16;
        }
    }
    // 兜底：排序 + 从两端往中间压间隔
    fx::insertion_sort_increasing_int16(nlsf_q15);
    nlsf_q15[0] = fx::max(nlsf_q15[0] as i32, delta_min_q15[0] as i32) as i16;
    for i in 1..l {
        nlsf_q15[i] = fx::max(
            nlsf_q15[i] as i32,
            nlsf_q15[i - 1] as i32 + delta_min_q15[i] as i32,
        ) as i16;
    }
    nlsf_q15[l - 1] = fx::min(
        nlsf_q15[l - 1] as i32,
        (1 << 15) - delta_min_q15[l] as i32,
    ) as i16;
    for i in (0..l - 1).rev() {
        nlsf_q15[i] = fx::min(
            nlsf_q15[i] as i32,
            nlsf_q15[i + 1] as i32 - delta_min_q15[i + 1] as i32,
        ) as i16;
    }
}

/// `silk_NLSF_decode`：一级码本 + 二级预测残差 + 权重 → NLSF（Q15）。
pub fn nlsf_decode(indices: &[i8], cb: &t::NlsfCodebook) -> [i16; MAX_LPC_ORDER] {
    let order = cb.order;
    let mut nlsf_q15 = [0i16; MAX_LPC_ORDER];
    // 一级码本（Q8 → Q15）
    let base = indices[0] as usize * order;
    for i in 0..order {
        nlsf_q15[i] = fx::lshift(cb.cb1_nlsf_q8[base + i] as i32, 7) as i16;
    }
    // 展开熵表下标与预测系数
    let mut ec_ix = [0i16; MAX_LPC_ORDER];
    let mut pred_q8 = [0u8; MAX_LPC_ORDER];
    super::silk::nlsf_unpack(cb, indices[0] as usize, &mut ec_ix, &mut pred_q8);
    let _ = ec_ix; // 熵表下标是解码侧用的；重建只用预测系数

    // 二级残差
    let mut res_q10 = [0i16; MAX_LPC_ORDER];
    nlsf_residual_dequant(
        &mut res_q10,
        &indices[1..],
        &pred_q8[..order],
        cb.quant_step_size_q16 as i32,
    );

    // 权重（拉罗亚）→ 反平方根 → 加到一级码本上
    let w_tmp_qw = nlsf_vq_weights_laroia(&nlsf_q15[..order]);
    for i in 0..order {
        let w_q9 = fx::sqrt_approx(fx::lshift(w_tmp_qw[i] as i32, (18 - NLSF_W_Q) as u32));
        let tmp = nlsf_q15[i] as i32
            + fx::div32_16(fx::lshift(res_q10[i] as i32, 14), w_q9);
        nlsf_q15[i] = fx::limit(tmp, 0, 32767) as i16;
    }
    nlsf_stabilize(&mut nlsf_q15[..order], &cb.delta_min_q15);
    nlsf_q15
}

/// `silk_NLSF2A_find_poly`：卷积生成 P/Q 多项式（QA 域）。
fn nlsf2a_find_poly(out: &mut [i32], c_lsf: &[i32], dd: usize) {
    out[0] = fx::lshift(1, QA);
    out[1] = -c_lsf[0];
    for k in 1..dd {
        let ftmp = c_lsf[2 * k];
        out[k + 1] =
            fx::lshift(out[k - 1], 1) - (fx::rshift_round64(fx::smull(ftmp, out[k]), QA) as i32);
        for n in (2..=k).rev() {
            out[n] += out[n - 2] - (fx::rshift_round64(fx::smull(ftmp, out[n - 1]), QA) as i32);
        }
        out[1] -= ftmp;
    }
}

/// `silk_bwexpander_32`：带宽扩展（chirp）—— 把系数朝"更稳"的方向软化。
pub fn bwexpander_32(ar: &mut [i32], chirp_q16: i32) {
    let d = ar.len();
    let chirp_minus_one_q16 = chirp_q16 - 65536;
    let mut chirp = chirp_q16;
    for a in ar.iter_mut().take(d - 1) {
        *a = fx::smulww(chirp, *a);
        chirp += fx::rshift_round(chirp.wrapping_mul(chirp_minus_one_q16), 16);
    }
    ar[d - 1] = fx::smulww(chirp, ar[d - 1]);
}

/// `silk_bwexpander`（16 位版，丢包后对 LPC 用）。
///
/// **注意 `chirp_minus_one_Q16` 只在进循环前算一次**（`chirp_Q16 - 65536`），
/// 循环里用的是这个常量、不是"当前 chirp 减 1.0" —— 后者会让扩张速度偏快，
/// 系数从第 3 个起就对不上（`A[0]`/`A[1]` 看不出来，因为前两步几乎一样）。
pub fn bwexpander(ar: &mut [i16], chirp_q16: i32) {
    let d = ar.len();
    let chirp_minus_one_q16 = chirp_q16 - 65536;
    let mut chirp = chirp_q16;
    // 参考实现特意注明：这里**不能**用 SMULWB（它的偏置会让滤波器不稳）
    for a in ar.iter_mut().take(d - 1) {
        *a = fx::rshift_round(chirp.wrapping_mul(*a as i32), 16) as i16;
        chirp += fx::rshift_round(chirp.wrapping_mul(chirp_minus_one_q16), 16);
    }
    ar[d - 1] = fx::rshift_round(chirp.wrapping_mul(ar[d - 1] as i32), 16) as i16;
}

/// `LPC_inverse_pred_gain_QA`：反射系数逐级消去，顺带检查稳定性。
fn lpc_inverse_pred_gain_qa(a_qa: &mut [[i32; MAX_LPC_ORDER]; 2], order: usize) -> i32 {
    let mut inv_gain_q30 = 1 << 30;
    let mut new_idx = order & 1;
    for k in (1..order).rev() {
        if a_qa[new_idx][k] > A_LIMIT_QA24 || a_qa[new_idx][k] < -A_LIMIT_QA24 {
            return 0;
        }
        let rc_q31 = -fx::lshift(a_qa[new_idx][k], 31 - 24);
        let rc_mult1_q30 = (1 << 30) - fx::smmul(rc_q31, rc_q31);
        let mult2q = 32 - fx::clz32(fx::abs(rc_mult1_q30));
        let rc_mult2 = fx::inverse32_varq(rc_mult1_q30, mult2q + 30);
        inv_gain_q30 = fx::lshift(fx::smmul(inv_gain_q30, rc_mult1_q30), 2);

        let old_idx = new_idx;
        new_idx = k & 1;
        for n in 0..k {
            let tmp =
                a_qa[old_idx][n] - fx::mul32_frac_q(a_qa[old_idx][k - n - 1], rc_q31, 31);
            a_qa[new_idx][n] = fx::mul32_frac_q(tmp, rc_mult2, mult2q as u32);
        }
    }
    if a_qa[new_idx][0] > A_LIMIT_QA24 || a_qa[new_idx][0] < -A_LIMIT_QA24 {
        return 0;
    }
    let rc_q31 = -fx::lshift(a_qa[new_idx][0], 31 - 24);
    let rc_mult1_q30 = (1 << 30) - fx::smmul(rc_q31, rc_q31);
    fx::lshift(fx::smmul(inv_gain_q30, rc_mult1_q30), 2)
}

/// `silk_LPC_inverse_pred_gain`：预测增益的倒数（Q30）；为 0 表示不稳定。
pub fn lpc_inverse_pred_gain(a_q12: &[i16]) -> i32 {
    let order = a_q12.len();
    let mut a_qa = [[0i32; MAX_LPC_ORDER]; 2];
    let new_idx = order & 1;
    let mut dc_resp = 0i32;
    for k in 0..order {
        dc_resp += a_q12[k] as i32;
        a_qa[new_idx][k] = fx::lshift(a_q12[k] as i32, QA_LPC_INV_GAIN - 12);
    }
    if dc_resp >= 4096 {
        // 直流不可稳，直接判死
        return 0;
    }
    lpc_inverse_pred_gain_qa(&mut a_qa, order)
}

/// `silk_NLSF2A`：NLSF（Q15）→ 单声道 LPC 系数（Q12）。
pub fn nlsf2a(nlsf: &[i16], d: usize) -> [i16; MAX_LPC_ORDER] {
    let ordering: &[usize] = if d == 16 { &ORDERING16 } else { &ORDERING10 };
    let mut cos_lsf_qa = [0i32; MAX_LPC_ORDER];
    // LSF → 2·cos(LSF)：查表 + 线性插值（表是 Q12，片内 128 点）
    for k in 0..d {
        let f_int = fx::rshift(nlsf[k] as i32, 15 - 7);
        let f_frac = nlsf[k] as i32 - fx::lshift(f_int, 15 - 7);
        debug_assert!((0..LSF_COS_TAB_SZ_FIX).contains(&f_int));
        let cos_val = t::LSF_COS_TAB_Q12[f_int as usize] as i32;
        let delta = t::LSF_COS_TAB_Q12[f_int as usize + 1] as i32 - cos_val;
        cos_lsf_qa[ordering[k]] = fx::rshift_round(
            fx::lshift(cos_val, 8) + delta.wrapping_mul(f_frac),
            20 - QA,
        );
    }

    let dd = d >> 1;
    let mut p = [0i32; MAX_LPC_ORDER / 2 + 1];
    let mut q = [0i32; MAX_LPC_ORDER / 2 + 1];
    nlsf2a_find_poly(&mut p, &cos_lsf_qa[0..], dd);
    nlsf2a_find_poly(&mut q, &cos_lsf_qa[1..], dd);

    let mut a32_qa1 = [0i32; MAX_LPC_ORDER];
    for k in 0..dd {
        let p_tmp = p[k + 1] + p[k];
        let q_tmp = q[k + 1] - q[k];
        a32_qa1[k] = -q_tmp - p_tmp; // QA+1
        a32_qa1[d - k - 1] = q_tmp - p_tmp;
    }

    let mut a_q12 = [0i16; MAX_LPC_ORDER];
    // 限制系数的绝对值（否则放不进 int16）：最多软化 10 轮
    let mut i = 0;
    while i < 10 {
        let mut maxabs = 0i32;
        let mut idx = 0usize;
        for k in 0..d {
            let absval = fx::abs(a32_qa1[k]);
            if absval > maxabs {
                maxabs = absval;
                idx = k;
            }
        }
        let maxabs = fx::rshift_round(maxabs, QA + 1 - 12);
        if maxabs > i16::MAX as i32 {
            let maxabs = fx::min(maxabs, 163838); // (int32_MAX >> 14) + int16_MAX
            let sc_q16 = 65470 // SILK_FIX_CONST(0.999, 16)
                - fx::div32_16(
                    fx::lshift(maxabs - i16::MAX as i32, 14),
                    fx::rshift(maxabs.wrapping_mul(idx as i32 + 1), 2),
                );
            bwexpander_32(&mut a32_qa1[..d], sc_q16);
        } else {
            break;
        }
        i += 1;
    }
    if i == 10 {
        // 跑了满 10 轮：钳位后再落 Q12，并把钳位结果写回去
        for k in 0..d {
            a_q12[k] = fx::sat16(fx::rshift_round(a32_qa1[k], QA + 1 - 12));
            a32_qa1[k] = fx::lshift(a_q12[k] as i32, QA + 1 - 12);
        }
    } else {
        for k in 0..d {
            a_q12[k] = fx::rshift_round(a32_qa1[k], QA + 1 - 12) as i16;
        }
    }

    // 稳定性保护：不稳就继续软化（最多 16 轮）
    for i in 0..MAX_LPC_STABILIZE_ITERATIONS {
        if lpc_inverse_pred_gain(&a_q12[..d]) < INV_PRED_GAIN_THRESHOLD_Q30 {
            bwexpander_32(&mut a32_qa1[..d], 65536 - fx::lshift(2, i as u32));
            for k in 0..d {
                a_q12[k] = fx::rshift_round(a32_qa1[k], QA + 1 - 12) as i16;
            }
        } else {
            break;
        }
    }
    a_q12
}

/// `silk_decode_pitch`：lag/contour 索引 → 每个子帧的 pitch lag。
pub fn decode_pitch(
    lag_index: i16,
    contour_index: i8,
    pitch_lags: &mut [i32],
    fs_khz: u32,
    nb_subfr: usize,
) {
    let (lag_cb, cbk_size): (&[i8], usize) = if fs_khz == 8 {
        if nb_subfr == MAX_NB_SUBFR {
            (&t::CB_LAGS_STAGE2, 11)
        } else {
            (&t::CB_LAGS_STAGE2_10MS, 3)
        }
    } else if nb_subfr == MAX_NB_SUBFR {
        (&t::CB_LAGS_STAGE3, 34)
    } else {
        (&t::CB_LAGS_STAGE3_10MS, 12)
    };

    let min_lag = 2 * fs_khz as i32; // PE_MIN_LAG_MS * fs
    let max_lag = 18 * fs_khz as i32; // PE_MAX_LAG_MS * fs
    let lag = min_lag + lag_index as i32;
    for k in 0..nb_subfr {
        let v = lag + lag_cb[k * cbk_size + contour_index as usize] as i32;
        pitch_lags[k] = fx::limit(v, min_lag, max_lag);
    }
}

/// `silk_decode_parameters`：把一帧的量化索引重建为解码器控制参数。
///
/// 会更新跨帧状态：`last_gain_index`、`prev_nlsf_q15`、`indices.nlsf_interp_coef_q2`
/// （非浊音帧还会把 `indices.per_index` 归零，和参考实现一致）。
pub fn decode_parameters(
    ch: &mut SilkChannel,
    control: &mut SilkDecoderControl,
    cond_coding: i32,
) -> Result<()> {
    let order = ch.tables.lpc_order;
    let nb_subfr = ch.nb_subfr;

    // ① 增益
    gains_dequant(
        &mut control.gains_q16,
        &ch.indices.gains_indices,
        &mut ch.last_gain_index,
        cond_coding == super::silk::CODE_CONDITIONALLY,
        nb_subfr,
    );

    // ② NLSF → LPC
    let cb = ch.tables.nlsf_cb;
    let nlsf = nlsf_decode(&ch.indices.nlsf_indices, cb);
    control.pred_coef_q12[1][..order].copy_from_slice(&nlsf2a(&nlsf[..order], order)[..order]);

    // 复位后的第一帧不允许插值（参考实现为了丢包后的衔接质量）
    if ch.first_frame_after_reset {
        ch.indices.nlsf_interp_coef_q2 = 4;
    }
    if (ch.indices.nlsf_interp_coef_q2 as i32) < 4 {
        // 前半帧用"上一帧 NLSF → 本帧 NLSF"的插值
        let mut nlsf0_q15 = [0i16; MAX_LPC_ORDER];
        for i in 0..order {
            nlsf0_q15[i] = (ch.prev_nlsf_q15[i] as i32
                + fx::rshift(
                    (ch.indices.nlsf_interp_coef_q2 as i32)
                        .wrapping_mul(nlsf[i] as i32 - ch.prev_nlsf_q15[i] as i32),
                    2,
                )) as i16;
        }
        control.pred_coef_q12[0][..order]
            .copy_from_slice(&nlsf2a(&nlsf0_q15[..order], order)[..order]);
    } else {
        control.pred_coef_q12[0] = control.pred_coef_q12[1];
    }
    ch.prev_nlsf_q15[..order].copy_from_slice(&nlsf[..order]);

    // 丢包后对 LPC 做带宽扩展（干净码流走不到，留着对齐语义）
    if ch.loss_cnt != 0 {
        bwexpander(&mut control.pred_coef_q12[0][..order], BWE_AFTER_LOSS_Q16);
        bwexpander(&mut control.pred_coef_q12[1][..order], BWE_AFTER_LOSS_Q16);
    }

    // ③ pitch / LTP
    if ch.indices.signal_type as i32 == TYPE_VOICED {
        decode_pitch(
            ch.indices.lag_index,
            ch.indices.contour_index,
            &mut control.pitch_l,
            ch.fs_khz,
            nb_subfr,
        );
        let per = ch.indices.per_index as usize;
        // 三个 PER 档的 VQ 码本是 8/16/32 个矢量（每个 5 抽头），在这里拍平成一张表
        let lo: usize = t::LTP_VQ_SIZES[..per]
            .iter()
            .map(|&s| s as usize * LTP_ORDER)
            .sum();
        for k in 0..nb_subfr {
            let ix = ch.indices.ltp_index[k] as usize * LTP_ORDER;
            for i in 0..LTP_ORDER {
                control.ltp_coef_q14[k * LTP_ORDER + i] =
                    fx::lshift(t::LTP_VQ_Q7[lo + ix + i] as i32, 7) as i16;
            }
        }
        control.ltp_scale_q14 = t::LTP_SCALES_Q14[ch.indices.ltp_scale_index as usize] as i32;
    } else {
        control.pitch_l = [0; MAX_NB_SUBFR];
        control.ltp_coef_q14 = [0; LTP_ORDER * MAX_NB_SUBFR];
        ch.indices.per_index = 0;
        control.ltp_scale_q14 = 0;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log2lin_is_monotone_and_saturates_at_zero() {
        assert_eq!(log2lin(-1), 0);
        assert_eq!(log2lin(0), 1 << 0, "2^0 = 1");
        assert!(log2lin(128) > log2lin(127));
        assert!((log2lin(128) - 2).abs() <= 1, "2^1 ≈ 2（近似）");
        assert!((log2lin(3967) as f64 / (1u64 << 31) as f64 - 1.0).abs() < 0.02);
    }

    #[test]
    fn gains_dequant_keeps_the_index_when_deltas_are_zero() {
        let mut gains = [0i32; MAX_NB_SUBFR];
        let mut prev = 10;
        // 条件编码 + 索引 4（= 增量 0）→ 索引不动、四个子帧增益相同
        // 注意：熵层给的索引要加 MIN_DELTA_GAIN_QUANT(-4) 才是真增量
        gains_dequant(&mut gains, &[4, 4, 4, 4], &mut prev, true, 4);
        assert_eq!(prev, 10);
        assert!(gains.iter().all(|&g| g == gains[0] && g > 0));
        // 独立编码：k=0 取绝对索引（且不能比上一帧低超过 16 档），其余子帧仍是增量
        let mut prev = 30;
        gains_dequant(&mut gains, &[2, 0, 0, 0], &mut prev, false, 4);
        assert_eq!(prev, 2, "14（绝对）+ 三个 -4 的增量");
        assert!(gains[0] > gains[1] && gains[1] > gains[2] && gains[2] > gains[3], "索引一路降 → 增益一路降");
    }

    #[test]
    fn nlsf_stabilize_enforces_the_minimum_distances() {
        // 三个挤在一起的 NLSF：稳定化后必须满足 delta_min 间隔
        // （delta_min 有 L+1 项：两端还各有一条到边界的最小距离）
        let delta_min: [i16; 5] = [100, 200, 300, 400, 500];
        let mut nlsf = [1000i16, 1050, 1100, 1150];
        nlsf_stabilize(&mut nlsf, &delta_min);
        let d = |a: i16, b: i16| a as i32 - b as i32;
        assert!(nlsf[0] >= delta_min[0]);
        assert!(d(nlsf[1], nlsf[0]) >= delta_min[1] as i32);
        assert!(d(nlsf[2], nlsf[1]) >= delta_min[2] as i32);
        assert!((1 << 15) - nlsf[3] as i32 >= delta_min[3] as i32);
        assert!(d(nlsf[3], nlsf[2]) >= delta_min[3] as i32, "相邻间隔也要够");
        // 本来就是合法的向量不该被动
        let mut ok = [2000i16, 4000, 6000, 8000];
        let before = ok;
        nlsf_stabilize(&mut ok, &delta_min);
        assert_eq!(ok, before);
    }

    #[test]
    fn lpc_inverse_pred_gain_is_max_for_zero_coefficients() {
        // 全零 LPC = 恒等滤波器 → 逆预测增益到顶（Q30 的 1<<30）
        let a = [0i16; 10];
        assert_eq!(lpc_inverse_pred_gain(&a), 1 << 30);
        // 系数大到越界 → 判为不稳定
        let mut a = [0i16; 10];
        a[9] = 32767;
        assert_eq!(lpc_inverse_pred_gain(&a), 0);
    }

    #[test]
    fn lpc_inverse_pred_gain_stays_in_the_q24_domain() {
        // 钉住 Q 域本身：这个向量必须落在"有效但不太稳"的区间（Q30 的六位数），
        // 而不是接近 1<<30。把 `QA_LPC_INV_GAIN` 写成 NLSF2A 的 16（左移 4 位而不是
        // 12 位）会让系数小 256 倍、增益恒接近 1<<30 —— 症状是**稳定化被整段跳过**，
        // LPC 系数只差最后一位（在官方向量 testvector02 的包 67 上才暴露）。
        let a: [i16; 10] = [7773, -3092, -502, -859, 238, -716, 1578, 234, -440, -334];
        assert_eq!(lpc_inverse_pred_gain(&a), 51268);
    }

    #[test]
    fn nlsf2a_matches_the_reference_on_a_known_vector() {
        // 期望值直接来自参考实现（官方向量 testvector02 包 67 的 NLSF，点 10）。
        // 它同时覆盖三件事：余弦表插值、find_poly、以及**两轮带宽扩展的稳定化**
        // （前两轮的逆预测增益 51268/10364 都低于阈值）。
        let nlsf: [i16; 10] = [1855, 1858, 4549, 5253, 11342, 15302, 18132, 22400, 25728, 28800];
        let a = nlsf2a(&nlsf, 10);
        assert_eq!(
            &a[..10],
            &[7772, -3091, -502, -859, 238, -715, 1577, 234, -439, -334]
        );
        assert_eq!(lpc_inverse_pred_gain(&a[..10]), 151692);
    }

    #[test]
    fn nlsf2a_produces_stable_coefficients() {
        // 均匀间隔的 NLSF（order 10）→ LPC 系数必须稳定（逆预测增益 > 阈值）
        let nlsf: [i16; 10] = [2000, 5000, 8000, 11000, 14000, 17000, 20000, 23000, 26000, 29000];
        let a = nlsf2a(&nlsf, 10);
        assert!(
            lpc_inverse_pred_gain(&a[..10]) >= INV_PRED_GAIN_THRESHOLD_Q30,
            "稳定化之后必须过阈值"
        );
        // 全部挤在低频端（初始不稳）也要被稳定化救回来
        let nlsf: [i16; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let a = nlsf2a(&nlsf, 10);
        assert!(lpc_inverse_pred_gain(&a[..10]) >= INV_PRED_GAIN_THRESHOLD_Q30);
    }

    #[test]
    fn decode_pitch_maps_to_the_codebook_and_clamps() {
        // 16 kHz、20 ms：lag = 32 + lagIndex，再叠 contour 码本的偏移
        let mut lags = [0i32; MAX_NB_SUBFR];
        decode_pitch(0, 0, &mut lags, 16, 4);
        assert!(lags.iter().all(|&l| l >= 32 && l <= 288));
        // 极端索引也不越界（会被钳到 [min_lag, max_lag]）
        decode_pitch(1000, 33, &mut lags, 16, 4);
        assert!(lags.iter().all(|&l| l == 288), "{lags:?}");
        decode_pitch(-1000, 0, &mut lags, 8, 2);
        assert!(lags[..2].iter().all(|&l| l == 16), "10 ms 帧只写前两个子帧：{lags:?}");
    }
}
