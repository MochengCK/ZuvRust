//! CELT 的**丢包隐藏**（参考实现 `celt/celt.c` 的 `celt_decode_lost`）。
//!
//! 两条完全不同的路径，由"丢了多久"和"起带"决定：
//!
//! ```text
//! loss_count < 5 且 start == 0  →  pitch 预测（pitch-based PLC）
//!   ① 在历史缓冲上做 2 倍下采样 + 4 阶 LPC 白化，再二分搜索出 pitch（第一帧才算）
//!   ② 用上一帧的 24 阶 LPC 白化激励，按 pitch 周期"循环复制"成新激励
//!   ③ 按前后两个周期的能量比推一个衰减系数，周期性复制时按段衰减
//!   ④ 用 LPC 合回来（IIR），再做能量校验（"爆炸"则清零，偏大则缩回来）
//!   ⑤ 与上一帧留下的 MDCT 尾巴做 TDAC 交叉混合，并过一遍后置滤波
//! 否则（丢了 ≥5 帧，或混合/SILK 侧的帧）→ 噪声填充（noise-based PLC / CNG）
//!   ④ 能量要么按 `backgroundLogE`（丢很久了）、要么按上一帧能量每帧衰减 1.5/0.5 dB
//!   ⑤ 用 LCG 造一段随机频谱、按频带整体归一化 → 反归一化 → 逆 MDCT
//! ```
//!
//! ## 几个必须照抄的点
//!
//! 1. **`C = st->channels`（API 声道数）**，不是 `stream_channels` —— 正常解码路径里
//!    这两个是分开的（`C` 用流声道、`CC` 用输出声道），丢包隐藏通篇只用输出声道数。
//! 2. **pitch 那条路径会给"历史窗"整体左移 N 个样本**（`out_mem[i] = out_mem[N+i]`），
//!    噪声那条**不移动** —— 它直接把新帧写到 `out_syn`。这不是笔误，是参考实现如此：
//!    两条路径后续被下一帧正常解码"接上"的方式不同。
//! 3. **衰减估计用 `E1/E2`（浮点下 `SHR32` 是空操作）**，不是 `E1/2/E2`；
//!    定点版本的 `SHR32(S1,1)` 在浮点构建里同样退化成原值。
//! 4. `loss_count` 在**帧尾**自增，正常解码路径（[`super::celt::CeltDecoder::decode_stream`]）
//!    在收尾处把它归零 —— 所以"丢一帧再收到好帧"下次又从 pitch 路径开始。

use super::bands;
use super::celt::{CeltState, NB_EBANDS};
use super::mdct::Mdct;
use super::synth::{self, SynthState, CHANNEL_BUF, DECODE_BUFFER_SIZE, LPC_ORDER, MAX_PERIOD};

/// `celt_lcg_rand`（参考实现 `bands.c`）。
#[inline]
fn lcg_rand(seed: u32) -> u32 {
    seed.wrapping_mul(1664525).wrapping_add(1013904223)
}

/// 丢包隐藏的主入口（对应 `celt_decode_lost`）。
///
/// `n` 是本帧的样本数（`(1<<lm) * 120`），`pcm` 长度必须是 `2*n`
/// —— 丢包隐藏固定输出 2 声道（参考实现里 `C = st->channels` 就是 API 声道数）。
#[allow(clippy::too_many_arguments)]
pub fn celt_decode_lost(
    state: &mut CeltState,
    synth: &mut SynthState,
    mdct: &Mdct,
    start: usize,
    end: usize,
    n: usize,
    lm: u32,
    pcm: &mut [i16],
) {
    let c = 2usize;
    let overlap = mdct.overlap();
    // `out_mem[c] = decode_mem[c] + DECODE_BUFFER_SIZE - MAX_PERIOD`
    let om = DECODE_BUFFER_SIZE - MAX_PERIOD;

    if synth.loss_count >= 5 || start != 0 {
        noise_based(state, synth, mdct, start, end, n, lm, overlap, om);
    } else {
        pitch_based(synth, mdct, n, overlap, om);
    }
    synth::deemphasis(synth, n, c, pcm);
    synth.loss_count += 1;
}

// ── pitch 预测那条路 ──

fn pitch_based(synth: &mut SynthState, mdct: &Mdct, n: usize, overlap: usize, om: usize) {
    let c = 2usize;
    let window = mdct.window();
    let len = n + overlap;

    // ① pitch 搜索（只在"刚丢第一帧"时做）—— 结果会一直用到丢包结束
    let pitch_index: i32;
    let mut fade = 1.0f32;
    if synth.loss_count == 0 {
        // 参考实现注释：这一段的"最长回看"是 720 个样点（对应最低 67 Hz）
        let poffset = 720usize;
        let mut lp = [0f32; DECODE_BUFFER_SIZE >> 1];
        {
            let (a, b) = (&synth.decode_mem, &mut lp);
            pitch_downsample(a, b, DECODE_BUFFER_SIZE, c);
        }
        let mut idx = 0i32;
        pitch_search(
            &lp[poffset >> 1..],
            &lp,
            DECODE_BUFFER_SIZE - poffset,
            poffset - 100,
            &mut idx,
        );
        pitch_index = poffset as i32 - idx;
        synth.last_pitch_index = pitch_index;
    } else {
        pitch_index = synth.last_pitch_index;
        fade = 0.8;
    }

    for ch in 0..c {
        let mut exc = [0f32; MAX_PERIOD];
        let mut e = vec![0f32; len + overlap];
        let mut ac = [0f32; LPC_ORDER + 1];
        let mut mem = [0f32; LPC_ORDER];

        // ② 用上一帧的 24 阶 LPC 把历史白化成激励
        let out_mem: [f32; MAX_PERIOD] = {
            let m = &synth.decode_mem[ch][om..om + MAX_PERIOD];
            let mut v = [0f32; MAX_PERIOD];
            v.copy_from_slice(m);
            v
        };
        exc.copy_from_slice(&out_mem);

        if synth.loss_count == 0 {
            autocorr(&exc, &mut ac, Some(window), overlap, LPC_ORDER, MAX_PERIOD);
            // 噪声底 -40 dB + 滞后窗
            ac[0] *= 1.0001;
            for i in 1..=LPC_ORDER {
                ac[i] -= ac[i] * (0.008 * i as f32) * (0.008 * i as f32);
            }
            lpc_coefs(&mut synth.lpc[ch], &ac, LPC_ORDER);
        }
        let lpc = synth.lpc[ch];

        for i in 0..LPC_ORDER {
            mem[i] = out_mem[MAX_PERIOD - 1 - i];
        }
        fir_inplace(&mut exc, &lpc, &mut mem, MAX_PERIOD, LPC_ORDER);

        // ③ 波形在衰减吗、衰减多快 —— 取前后两个周期的能量比
        let period = if pitch_index <= (MAX_PERIOD / 2) as i32 {
            pitch_index as usize
        } else {
            MAX_PERIOD / 2
        };
        let mut e1 = 1.0f32;
        let mut e2 = 1.0f32;
        for i in 0..period {
            let a = exc[MAX_PERIOD - period + i];
            let b = exc[MAX_PERIOD - 2 * period + i];
            e1 += a * a;
            e2 += b * b;
        }
        if e1 > e2 {
            e1 = e2;
        }
        let mut decay = (e1 / e2).sqrt();

        // ④ 周期性复制激励：每跨过一个 pitch 周期就把衰减再平方一次
        let mut s1 = 0.0f32;
        let mut pos = MAX_PERIOD as i32 - pitch_index;
        for i in 0..len + overlap {
            if pos >= MAX_PERIOD as i32 {
                pos -= pitch_index;
                decay *= decay;
            }
            debug_assert!(pos >= 0, "pitch 位置跑到缓冲区前面去了");
            e[i] = decay * exc[pos as usize];
            let tmp = out_mem[pos as usize];
            s1 += tmp * tmp;
            pos += 1;
        }

        for i in 0..LPC_ORDER {
            mem[i] = out_mem[MAX_PERIOD - 1 - i];
        }
        for v in e.iter_mut().take(len + overlap) {
            *v *= fade;
        }
        iir_inplace(&mut e, &lpc, &mut mem, len + overlap, LPC_ORDER);

        // ⑤ "爆炸"检测：合成出来的能量比原信号大太多就清零，稍大就缩回来
        let mut s2 = 0.0f32;
        for v in e.iter().take(len + overlap) {
            s2 += v * v;
        }
        if !(s1 > 0.2 * s2) {
            for v in e.iter_mut().take(len + overlap) {
                *v = 0.0;
            }
        } else if s1 < s2 {
            let ratio = ((s1 + 1.0) / (s2 + 1.0)).sqrt();
            for v in e.iter_mut().take(len + overlap) {
                *v *= ratio;
            }
        }

        // ⑥ 上一帧的 MDCT 尾巴（现在还在 overlap 区里）先过后置滤波
        let period_pf = synth.postfilter_period.max(synth::COMBFILTER_MINPERIOD) as usize;
        let gain_pf = synth.postfilter_gain;
        let tapset_pf = synth.postfilter_tapset;
        {
            let buf = &mut synth.decode_mem[ch];
            synth::comb_filter(
                buf,
                DECODE_BUFFER_SIZE,
                period_pf,
                period_pf,
                overlap,
                gain_pf,
                gain_pf,
                tapset_pf,
                tapset_pf,
                window,
                0,
            );
            // 历史窗整体左移 N
            for i in 0..MAX_PERIOD + overlap - n {
                buf[om + i] = buf[om + n + i];
            }
            // ⑦ TDAC：把新合成的前 overlap 个样本与尾巴交叉混合，写回 overlap 区
            for i in 0..overlap / 2 {
                let tmp =
                    window[i] * e[n + overlap - 1 - i] + window[overlap - i - 1] * e[n + i];
                buf[DECODE_BUFFER_SIZE + i] = window[overlap - i - 1] * tmp;
                buf[DECODE_BUFFER_SIZE + overlap - i - 1] = window[i] * tmp;
            }
            let out_start = om + MAX_PERIOD - n;
            buf[out_start..out_start + n].copy_from_slice(&e[..n]);
        }

        // ⑧ 给下一帧预备"预滤波"的尾巴（增益取负 —— 配对的后置滤波在下一帧做）
        {
            let mut tail = vec![0f32; overlap];
            synth::comb_filter_into(
                &mut tail,
                &synth.decode_mem[ch],
                DECODE_BUFFER_SIZE,
                period_pf,
                -gain_pf,
                tapset_pf,
                overlap,
            );
            synth.decode_mem[ch][DECODE_BUFFER_SIZE..DECODE_BUFFER_SIZE + overlap]
                .copy_from_slice(&tail);
        }
    }
}

// ── 噪声填充那条路（丢很久 / 混合与 SILK 侧）──

#[allow(clippy::too_many_arguments)]
fn noise_based(
    state: &mut CeltState,
    synth: &mut SynthState,
    mdct: &Mdct,
    start: usize,
    end: usize,
    n: usize,
    lm: u32,
    overlap: usize,
    om: usize,
) {
    let c = 2usize;
    let m = 1usize << lm;
    let eff_end = end.min(bands::EFF_EBANDS);
    let mut band_e = [[0f32; NB_EBANDS]; 2];

    if synth.loss_count >= 5 {
        // 丢太久了：按长期背景能量造噪声，不再跟着上一帧往下掉
        bands::log2_amp(&mut band_e, &state.background_log_e, start, end, c);
    } else {
        // 刚丢：上一帧能量每帧再掉一点（第一帧 1.5 dB，之后每帧 0.5 dB）
        let decay = if synth.loss_count == 0 { 1.5 } else { 0.5 };
        for ch in 0..c {
            for i in start..end {
                state.old_ebands[ch][i] -= decay;
            }
        }
        bands::log2_amp(&mut band_e, &state.old_ebands, start, end, c);
    }

    // LCG 造频谱：每个频带一段随机数，整段归一到单位范数
    let mut seed = state.rng;
    let mut x = vec![0f32; c * n];
    for ch in 0..c {
        for band in start..bands::EFF_EBANDS {
            let boffs = ch * n + (bands::band_edge(band) << lm);
            let blen = (bands::band_edge(band + 1) - bands::band_edge(band)) << lm;
            for j in 0..blen {
                seed = lcg_rand(seed);
                x[boffs + j] = ((seed as i32) >> 20) as f32;
            }
            renormalise_vector(&mut x[boffs..boffs + blen], blen, 1.0);
        }
        x[(ch * n + (bands::band_edge(end) << lm))..(ch * n + n)].fill(0.0);
    }
    state.rng = seed;

    let xv: Vec<Vec<f32>> = (0..c).map(|ch| x[ch * n..(ch + 1) * n].to_vec()).collect();
    let mut freq = bands::denormalise_bands(&xv, &band_e, bands::EFF_EBANDS, c, m);
    let bound = bands::band_edge(eff_end) << lm;
    for ch in 0..c {
        let low = bands::band_edge(start) << lm;
        freq[ch][..low].fill(0.0);
        freq[ch][bound..n].fill(0.0);
    }

    // 逆 MDCT + 重叠相加，直接落到 out_syn / overlap 区（**不动历史**）
    let out_start = om + MAX_PERIOD - n;
    for ch in 0..c {
        let mut tmp = vec![0f32; n + overlap];
        mdct.backward(&freq[ch], &mut tmp, mdct.max_shift() - lm as usize, 1);
        let buf = &mut synth.decode_mem[ch];
        for j in 0..overlap {
            buf[out_start + j] = tmp[j] + buf[DECODE_BUFFER_SIZE + j];
        }
        buf[out_start + overlap..out_start + n].copy_from_slice(&tmp[overlap..n]);
        buf[DECODE_BUFFER_SIZE..DECODE_BUFFER_SIZE + overlap]
            .copy_from_slice(&tmp[n..n + overlap]);
    }
}

// ── 参考实现里那一堆小工具（都是 float 构建下的形态，`SHR32`/`SHL32` 是空操作）──

/// `renormalise_vector(X, N, gain)`：把一段向量归一到单位范数再乘 `gain`。
fn renormalise_vector(x: &mut [f32], n: usize, gain: f32) {
    let mut e = 1e-15f32; // `EPSILON`（浮点构建下是 1e-15，不是 1）
    for v in x.iter().take(n) {
        e += v * v;
    }
    let g = gain / e.sqrt();
    for v in x.iter_mut().take(n) {
        *v *= g;
    }
}

/// `_celt_autocorr`：自相关（`window` 给定时对首尾各 `overlap` 个样本加窗）。
fn autocorr(x: &[f32], ac: &mut [f32], window: Option<&[f32]>, overlap: usize, lag: usize, n: usize) {
    let mut xx = x[..n].to_vec();
    if let Some(w) = window {
        for i in 0..overlap {
            xx[i] = x[i] * w[i];
            xx[n - i - 1] = x[n - i - 1] * w[i];
        }
    }
    for l in (0..=lag).rev() {
        let mut d = 0f32;
        for i in l..n {
            d += xx[i] * xx[i - l];
        }
        ac[l] = d;
    }
    ac[0] += 10.0;
}

/// `_celt_lpc`：Levinson-Durbin，从自相关解 LPC 系数（浮点版本直接迭代）。
fn lpc_coefs(lpc: &mut [f32], ac: &[f32], p: usize) {
    for v in lpc.iter_mut().take(p) {
        *v = 0.0;
    }
    if ac[0] == 0.0 {
        return;
    }
    let mut error = ac[0];
    for i in 0..p {
        let mut rr = 0f32;
        for j in 0..i {
            rr += lpc[j] * ac[i - j];
        }
        rr += ac[i + 1];
        let r = -rr / error;
        lpc[i] = r;
        for j in 0..(i + 1) >> 1 {
            let tmp1 = lpc[j];
            let tmp2 = lpc[i - 1 - j];
            lpc[j] = tmp1 + r * tmp2;
            lpc[i - 1 - j] = tmp2 + r * tmp1;
        }
        error -= (r * r) * error;
        if error < 0.001 * ac[0] {
            break;
        }
    }
}

/// `celt_fir`（就地版本，参考实现里 `x`/`y` 是同一块内存）。
fn fir_inplace(x: &mut [f32], num: &[f32], mem: &mut [f32], n: usize, ord: usize) {
    for i in 0..n {
        let mut sum = x[i];
        for j in 0..ord {
            sum += num[j] * mem[j];
        }
        for j in (1..ord).rev() {
            mem[j] = mem[j - 1];
        }
        mem[0] = x[i];
        x[i] = sum;
    }
}

/// `celt_iir`（就地版本）：LPC 合成滤波（单极点递归）。
fn iir_inplace(y: &mut [f32], den: &[f32], mem: &mut [f32], n: usize, ord: usize) {
    for i in 0..n {
        let mut sum = y[i];
        for j in 0..ord {
            sum -= den[j] * mem[j];
        }
        for j in (1..ord).rev() {
            mem[j] = mem[j - 1];
        }
        mem[0] = sum;
        y[i] = sum;
    }
}

/// `pitch_downsample`：2 倍下采样 + 4 阶 LPC 白化（+ 一阶 0.8 的高通）。
fn pitch_downsample(x: &[[f32; CHANNEL_BUF]; 2], lp: &mut [f32], len: usize, c: usize) {
    let n = len >> 1;
    for i in 1..n {
        lp[i] = 0.5 * (0.5 * (x[0][2 * i - 1] + x[0][2 * i + 1]) + x[0][2 * i]);
    }
    lp[0] = 0.5 * (0.5 * x[0][1] + x[0][0]);
    if c == 2 {
        for i in 1..n {
            lp[i] += 0.5 * (0.5 * (x[1][2 * i - 1] + x[1][2 * i + 1]) + x[1][2 * i]);
        }
        lp[0] += 0.5 * (0.5 * x[1][1] + x[1][0]);
    }

    let mut ac = [0f32; 5];
    autocorr(&lp[..n], &mut ac, None, 0, 4, n);
    ac[0] *= 1.0001;
    for i in 1..=4 {
        ac[i] -= ac[i] * (0.008 * i as f32) * (0.008 * i as f32);
    }
    let mut lpc = [0f32; 4];
    lpc_coefs(&mut lpc, &ac, 4);
    let mut tmp = 1.0f32;
    for v in lpc.iter_mut() {
        tmp *= 0.9;
        *v *= tmp;
    }
    let mut mem = [0f32; 4];
    fir_inplace(&mut lp[..n], &lpc, &mut mem, n, 4);
    mem[0] = 0.0;
    let one = [0.8f32];
    fir_inplace(&mut lp[..n], &one, &mut mem, n, 1);
}

/// `find_best_pitch`：在互相关表里挑两个最好的 pitch（用 `num²/(den·Syy)` 打分）。
fn find_best_pitch(xcorr: &[f32], y: &[f32], len: usize, max_pitch: usize, best: &mut [usize; 2]) {
    let mut syy = 1.0f32;
    let mut best_num = [-1.0f32; 2];
    let mut best_den = [0.0f32; 2];
    best[0] = 0;
    best[1] = 1;
    for v in y.iter().take(len) {
        syy += v * v;
    }
    for i in 0..max_pitch {
        if xcorr[i] > 0.0 {
            let num = xcorr[i] * xcorr[i];
            if num * best_den[1] > best_num[1] * syy {
                if num * best_den[0] > best_num[0] * syy {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best[1] = best[0];
                    best_num[0] = num;
                    best_den[0] = syy;
                    best[0] = i;
                } else {
                    best_num[1] = num;
                    best_den[1] = syy;
                    best[1] = i;
                }
            }
        }
        // `Syy` 是**滑动**窗能量（长度 len），每个 i 都要往前推一格 ——
        // 少了这一句，"打分"退化成 `argmax |xcorr|`，粗搜会锁到完全不同的周期上。
        // （浮点构建里 `SHR32(..., yshift)` 是空操作，`yshift` 根本没被求值。）
        let out_v = y[i + len];
        let in_v = y[i];
        syy += out_v * out_v - in_v * in_v;
        syy = syy.max(1.0);
    }
}

/// `pitch_search`：4 倍抽取粗搜 → 2 倍抽取细搜 → 伪插值细分。
fn pitch_search(x_lp: &[f32], y: &[f32], len: usize, max_pitch: usize, pitch: &mut i32) {
    let lag = len + max_pitch;
    let mut x_lp4 = vec![0f32; len >> 2];
    let mut y_lp4 = vec![0f32; lag >> 2];
    let mut xcorr = vec![0f32; max_pitch >> 1];
    for (j, v) in x_lp4.iter_mut().enumerate() {
        *v = x_lp[2 * j];
    }
    for (j, v) in y_lp4.iter_mut().enumerate() {
        *v = y[2 * j];
    }

    // 粗搜（4 倍抽取）
    for i in 0..max_pitch >> 2 {
        let mut sum = 0f32;
        for j in 0..len >> 2 {
            sum += x_lp4[j] * y_lp4[i + j];
        }
        xcorr[i] = sum.max(-1.0);
    }
    let mut best = [0usize; 2];
    find_best_pitch(&xcorr, &y_lp4, len >> 2, max_pitch >> 2, &mut best);

    // 细搜（2 倍抽取，只在粗搜命中的附近找）
    for i in 0..max_pitch >> 1 {
        xcorr[i] = 0.0;
        if (i as i32 - 2 * best[0] as i32).abs() > 2 && (i as i32 - 2 * best[1] as i32).abs() > 2 {
            continue;
        }
        let mut sum = 0f32;
        for j in 0..len >> 1 {
            sum += x_lp[j] * y[i + j];
        }
        xcorr[i] = sum.max(-1.0);
    }
    find_best_pitch(&xcorr, y, len >> 1, max_pitch >> 1, &mut best);

    // 伪插值：看左右两个邻居哪边更"像"
    let offset = if best[0] > 0 && best[0] < (max_pitch >> 1) - 1 {
        let a = xcorr[best[0] - 1];
        let b = xcorr[best[0]];
        let c = xcorr[best[0] + 1];
        if (c - a) > 0.7 * (b - a) {
            1
        } else if (a - c) > 0.7 * (b - c) {
            -1
        } else {
            0
        }
    } else {
        0
    };
    *pitch = 2 * best[0] as i32 - offset;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 金标准来自 `/tmp/opus_plc_units_probe.c`（直接调参考实现的 `_celt_autocorr`
    /// / `_celt_lpc` / `pitch_search` 打印出来的值）。
    fn close(a: f32, b: f32, what: &str) {
        let tol = 1e-4f32 * (1.0 + b.abs());
        assert!((a - b).abs() <= tol, "{what}: {a} vs {b}");
    }

    #[test]
    fn autocorr_matches_the_reference() {
        // x[i] = sin(0.03 i) * 1000 + (i*37 % 23) - 11，n = 256，lag = 24，不加窗
        let x: Vec<f32> = (0..256)
            .map(|i| (0.03 * i as f32).sin() * 1000.0 + ((i * 37) % 23) as f32 - 11.0)
            .collect();
        let mut ac = [0f32; 25];
        autocorr(&x, &mut ac, None, 0, 24, 256);
        const WANT: [f32; 5] = [
            124676928.0,
            124127968.0,
            123480344.0,
            122734040.0,
            121862888.0,
        ];
        for (i, w) in WANT.iter().enumerate() {
            close(ac[i], *w, &format!("ac[{i}]"));
        }
        // 尾巴也不能崩（单调缓降）
        assert!(ac[24] < ac[0] && ac[24] > 0.0);
    }

    #[test]
    fn lpc_matches_the_reference_on_a_pure_tone() {
        // x[i] = cos(2π·0.1·i)，512 个样点 → 参考实现的 24 阶 LPC
        let x: Vec<f32> = (0..512)
            .map(|i| (2.0 * std::f32::consts::PI * 0.1 * i as f32).cos())
            .collect();
        let mut ac = [0f32; 25];
        autocorr(&x, &mut ac, None, 0, 24, 512);
        let mut lpc = [0f32; LPC_ORDER];
        lpc_coefs(&mut lpc, &ac, LPC_ORDER);
        const WANT: [f32; 24] = [
            -0.308319, -0.048464, 0.134894, 0.207645, 0.184825, 0.108291, 0.022733, -0.039893,
            -0.066821, -0.061758, -0.037840, -0.009785, 0.011540, 0.021423, 0.020610, 0.013295,
            0.004129, -0.003134, -0.006845, -0.007017, -0.005011, -0.002124, 0.000419, 0.002098,
        ];
        for (i, w) in WANT.iter().enumerate() {
            close(lpc[i], *w, &format!("lpc[{i}]"));
        }
    }

    #[test]
    fn lpc_of_a_flat_autocorrelation_is_zero() {
        // 只有 ac[0] 有值（白噪声）→ LPC 系数全 0
        let mut ac = [0f32; 25];
        ac[0] = 1e9;
        let mut lpc = [1f32; LPC_ORDER];
        lpc_coefs(&mut lpc, &ac, LPC_ORDER);
        assert!(lpc.iter().all(|&v| v == 0.0), "{lpc:?}");
    }

    #[test]
    fn renormalise_vector_gives_unit_norm() {
        let mut x = [3f32, 4.0];
        renormalise_vector(&mut x, 2, 1.0);
        let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "范数 {norm}");
        assert!((x[0] - 0.6).abs() < 1e-6 && (x[1] - 0.8).abs() < 1e-6, "{x:?}");
    }

    #[test]
    fn pitch_search_matches_the_reference() {
        // 与探针完全相同的输入：y 是伪随机序列，x_lp 取 y[250..650]
        let mut y = vec![0f32; 400 + 300 + 8];
        for (i, v) in y.iter_mut().enumerate() {
            *v = ((i as u32).wrapping_mul(1103515245).wrapping_add(12345) % 2001) as f32 - 1000.0;
        }
        let x_lp: Vec<f32> = (0..400).map(|i| y[250 + i]).collect();
        let mut pitch = 0i32;
        pitch_search(&x_lp, &y, 400, 300, &mut pitch);
        assert_eq!(pitch, 172, "参考实现给出 172");
    }

    #[test]
    fn noise_plc_keeps_the_output_in_range_and_updates_state() {
        let mdct = Mdct::new_48k();
        let mut st = CeltState::default();
        let mut synth = SynthState::default();
        // 先让它"丢"够 6 帧，强制走噪声路径
        synth.loss_count = 6;
        st.old_ebands = [[-6.0; NB_EBANDS]; 2];
        st.background_log_e = [[-12.0; NB_EBANDS]; 2];
        let mut pcm = vec![0i16; 2 * 960];
        celt_decode_lost(&mut st, &mut synth, &mdct, 0, 21, 960, 3, &mut pcm);
        assert_eq!(synth.loss_count, 7);
        assert!(pcm.iter().any(|&v| v != 0), "背景能量不为 0 时应当填出噪声");
        // 噪声持续存在，但能量受 backgroundLogE 约束（不该打满量程）
        let peak = pcm.iter().map(|v| v.abs()).max().unwrap();
        assert!(peak < 32767, "不该削顶：{peak}");
        // 用掉的随机种子要推进（否则每帧造出同一段噪声）
        assert_ne!(st.rng, 0);
    }

    #[test]
    fn first_lost_frame_after_a_good_one_uses_the_pitch_path() {
        let mdct = Mdct::new_48k();
        let mut st = CeltState::default();
        let mut synth = SynthState::default();
        let mut pcm = vec![0i16; 2 * 960];
        celt_decode_lost(&mut st, &mut synth, &mdct, 0, 21, 960, 3, &mut pcm);
        assert_eq!(synth.loss_count, 1, "走的是 pitch 路径（不增到 5 以上）");
        assert!(synth.last_pitch_index > 0, "pitch 搜索应当给出一个周期");
    }
}
