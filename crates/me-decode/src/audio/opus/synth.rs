//! CELT 的**合成段**（参考实现 `celt.c` 里 `celt_decode_with_ec` 的收尾那一大段）。
//!
//! 顺序与参考实现逐句对应（顺序错了声音就对不上，别重排）：
//!
//! ```text
//! 反归一化频谱 → 边界清零 → 单/双声道适配
//!   → 每声道：搬历史 → IMDCT + 重叠相加（`mdct.rs`）→ 后置滤波（comb_filter）
//!   → 后置滤波状态推进 → 去加重（deemphasis）→ 转 i16 PCM
//! ```
//!
//! ## 几个必须照抄的语义
//!
//! 1. **历史缓冲是"后置滤波之后"的信号**：`decode_mem` 每帧左移 N 个样本，最新 N 个是
//!    本帧；`comb_filter` 就地递归（`y[i]` 会读到已经滤过的 `y[i-T]`）。
//! 2. **后置滤波参数有两组**：`_old` 是上一帧的，前 120 个样本从 old 过渡到当前帧；
//!    LM≠0 的帧在收尾时把 `_old` 直接设成当前帧（因为后半段本来就是用当前帧滤的），
//!    LM=0（2.5 ms）不这么设 —— 两种分支都要留。
//! 3. **重叠相加是 `x[j] + overlap_mem[j]`（前 `overlap` 个），之后把 `x[N..N+overlap]`
//!    存回 `overlap_mem`**。IMDCT 写进的 `x` 只有前 `overlap` 个样本需要预先清零
//!    （短块的交叠靠 `+=` 累加，参考实现里那句 `OPUS_CLEAR(x, overlap)` 就是干这个）。
//! 4. **去加重的输出换算**：参考实现 `deemphasis` 写的是 float PCM（`tmp/32768`），
//!    `opus_decode` 再调 `FLOAT2INT16` 转 i16 —— 本函数把两步合并：
//!    `clamp(±32768/32767)` 后 `floor(.5+x)`（`float2cast.h` 在没定义 `HAVE_LRINTF`
//!    时的兜底路径，参考实现的 Makefile 构建走的正是这条）。
//! 5. **预加重系数**是模式自带的 `preemph = [0.8500061035, 0, 1, 1]`（48 kHz 模式）。

use super::celt::{PostFilter, SHORT_MDCT};
use super::mdct::Mdct;

/// 每声道的历史缓冲长度（参考实现 `DECODE_BUFFER_SIZE`）。
pub(crate) const DECODE_BUFFER_SIZE: usize = 2048;
/// 每声道缓冲总长：历史 + 重叠加窗的尾巴。参考实现里 `overlap_mem` 就贴在
/// `_decode_mem` 后面（`overlap_mem[c] = decode_mem[c]+DECODE_BUFFER_SIZE`）——
/// 后置滤波周期为 0（滤波器关）时会读到尾巴那一段，读的位置必须合法。
pub(crate) const CHANNEL_BUF: usize = DECODE_BUFFER_SIZE + super::mdct::MODE_OVERLAP;
/// 后置滤波能回看的最长周期（参考实现 `MAX_PERIOD` / `COMBFILTER_MAXPERIOD`）。
pub(crate) const MAX_PERIOD: usize = 1024;
/// 丢包隐藏里 LPC 的阶数（参考实现 `celt_lpc.h` 的 `LPC_ORDER`）。
pub(crate) const LPC_ORDER: usize = 24;
/// 后置滤波的最短周期（参考实现 `COMBFILTER_MINPERIOD`）。
pub(crate) const COMBFILTER_MINPERIOD: i32 = 15;

/// 48 kHz 模式的预加重系数（参考实现 `opus_custom_mode_create` 的 `preemph`）。
/// 数值**照抄参考实现字面量**（`0.8500061035f` 等），不按 clippy 的短写法截断 ——
/// 这样和参考源码逐字对得上（值本身在 f32 里是精确的）。
#[allow(clippy::excessive_precision)]
const PREEMPH: [f32; 4] = [0.8500061035, 0.0, 1.0, 1.0];

/// 后置滤波的抽头增益表（参考实现 `comb_filter` 里的 `gains[3][3]`）。
/// 同样是照抄参考字面量，别按 clippy 建议改写。
#[allow(clippy::excessive_precision)]
const COMB_GAINS: [[f32; 3]; 3] = [
    [0.3066406250, 0.2170410156, 0.1296386719],
    [0.4638671875, 0.2680664062, 0.0],
    [0.7998046875, 0.1000976562, 0.0],
];

/// 合成的跨帧状态。
///
/// 初值全零 + 预加重记忆零 —— 参考实现 `OPUS_RESET_STATE` 就是这么清的
/// （`postfilter_*` 也会被清成 0，用的时候再钳到 `COMBFILTER_MINPERIOD`）。
pub struct SynthState {
    /// 每声道缓冲：`[0, DECODE_BUFFER_SIZE)` 是"历史 + 当前帧"，末尾 `overlap` 个是
    /// 上一帧 IMDCT 的尾巴（参考实现里 `overlap_mem` 就是这一段）。
    pub(crate) decode_mem: [[f32; CHANNEL_BUF]; 2],
    /// 去加重的上一样本记忆（每声道一个）。
    preemph_mem: [f32; 2],
    /// 本帧的后置滤波参数（丢包隐藏要拿它给"上一帧的 MDCT 尾巴"过一遍滤波，
    /// 所以对同 crate 可见）。
    pub(crate) postfilter_period: i32,
    pub(crate) postfilter_gain: f32,
    pub(crate) postfilter_tapset: u32,
    /// 上一帧的后置滤波参数（前 120 个样本从这里过渡）。
    postfilter_period_old: i32,
    postfilter_gain_old: f32,
    postfilter_tapset_old: u32,
    // ── 丢包隐藏（PLC）用的状态（参考实现 `celt_decode_lost` 读写的那些字段）──
    /// 每声道的 LPC 系数（`st->lpc`，`loss_count == 0` 的丢帧里重算一次）。
    pub(crate) lpc: [[f32; LPC_ORDER]; 2],
    /// 连续丢帧计数（`st->loss_count`）：0 = 刚丢第一帧、≥5 转噪声隐藏。
    pub(crate) loss_count: i32,
    /// 上一轮找到的 pitch 周期（`st->last_pitch_index`）。
    pub(crate) last_pitch_index: i32,
}

impl Default for SynthState {
    fn default() -> Self {
        Self {
            decode_mem: [[0.0; CHANNEL_BUF]; 2],
            preemph_mem: [0.0; 2],
            postfilter_period: 0,
            postfilter_gain: 0.0,
            postfilter_tapset: 0,
            postfilter_period_old: 0,
            postfilter_gain_old: 0.0,
            postfilter_tapset_old: 0,
            lpc: [[0.0; LPC_ORDER]; 2],
            loss_count: 0,
            last_pitch_index: 0,
        }
    }
}

impl SynthState {
    /// 当前帧的后置滤波参数（对拍/调试用）。
    pub fn postfilter(&self) -> (i32, f32, u32) {
        (self.postfilter_period, self.postfilter_gain, self.postfilter_tapset)
    }

    /// 正常解出一帧（或复位）之后把连续丢帧计数归零。
    pub(crate) fn reset_loss_count(&mut self) {
        self.loss_count = 0;
    }
}

/// 一帧合成的输入（对应参考实现 `celt_decode_with_ec` 里合成段用到的那些量）。
pub struct SynthInput<'a> {
    /// 流里的声道数（`C`，TOC 的 s 位决定）。
    pub stream_channels: usize,
    /// 解码器输出声道数（`CC`；Opus 解码器固定为 2，独立 CELT 时可以是 1）。
    pub out_channels: usize,
    /// 帧长指数（2.5 ms 是 0，20 ms 是 3）。
    pub lm: u32,
    /// 是否瞬态帧（决定走短块）。
    pub transient: bool,
    /// 起止带（CELT-only 从 0 带起；结束带由带宽决定）。
    pub start: usize,
    pub end: usize,
    /// 帧头解出的后置滤波参数。
    pub postfilter: PostFilter,
    /// 反归一化之后的频谱（`stream_channels` 个声道，每个 `M*120` 个系数）。
    pub freq: &'a [Vec<f32>],
}

/// 一帧合成：反归一化频谱 → PCM（i16 交错）。
///
/// `pcm` 的长度必须是 `out_channels * N`（N = `(1<<lm)*120`）。
pub fn synth_frame(
    state: &mut SynthState,
    mdct: &Mdct,
    inp: &SynthInput<'_>,
    pcm: &mut [i16],
) {
    let c = inp.stream_channels;
    let cc = inp.out_channels;
    debug_assert!(c == 1 || c == 2);
    debug_assert!(cc == 1 || cc == 2);
    debug_assert_eq!(inp.freq.len(), c);
    let m = 1usize << inp.lm;
    let n = m * SHORT_MDCT as usize;
    let overlap = mdct.overlap();
    debug_assert_eq!(pcm.len(), cc * n);
    // 回看的历史必须装得下最长周期（参考实现里 `decode_mem` 比 `MAX_PERIOD` 多留的那一截）
    debug_assert!(DECODE_BUFFER_SIZE - n >= MAX_PERIOD, "帧太长，历史不够回看");

    // ① 边界清零（参考实现里 denormalise 之后那两段循环）：
    //    denormalise 只填了 [0, M*eBands[end])，起带之前的那一段（混合模式 start=17）
    //    要显式清掉；倍数带之后本来就没被写，这里一并保证。
    let low = m * super::bands::band_edge(inp.start);
    let high = m * super::bands::band_edge(inp.end);
    let freq: Vec<Vec<f32>> = match (cc, c) {
        // 单声道帧 → 双声道输出：直接复制（参考实现 `CC==2 && C==1`）
        (2, 1) => {
            let mut f = inp.freq[0].clone();
            for v in &mut f[..low] {
                *v = 0.0;
            }
            for v in &mut f[high..] {
                *v = 0.0;
            }
            vec![f.clone(), f]
        }
        // 双声道帧 → 单声道输出：两声道取平均（参考实现 `CC==1 && C==2`，`HALF32(ADD32(a,b))`）
        (1, 2) => {
            let f: Vec<f32> = (0..n)
                .map(|j| 0.5 * (inp.freq[0][j] + inp.freq[1][j]))
                .collect();
            vec![f]
        }
        _ => {
            let mut f = inp.freq.to_vec();
            for v in &mut f[0][..low] {
                *v = 0.0;
            }
            for v in &mut f[0][high..] {
                *v = 0.0;
            }
            if cc == 2 {
                for v in &mut f[1][..low] {
                    *v = 0.0;
                }
                for v in &mut f[1][high..] {
                    *v = 0.0;
                }
            }
            f
        }
    };

    // ② 每声道：搬历史 → IMDCT + 重叠相加 → 后置滤波
    //
    //    参考实现先把 `decode_mem` 整体左移 N（`OPUS_MOVE`），末尾 N 个样本让给本帧。
    state.postfilter_period = state.postfilter_period.max(COMBFILTER_MINPERIOD);
    state.postfilter_period_old = state.postfilter_period_old.max(COMBFILTER_MINPERIOD);
    for (ch, fch) in freq.iter().enumerate().take(cc) {
        state.decode_mem[ch].copy_within(n..DECODE_BUFFER_SIZE, 0);
        let frame_start = DECODE_BUFFER_SIZE - n;

        // IMDCT + 重叠相加。`x` 只有前 `overlap` 个样本要清零：短块之间靠 `+=` 累加，
        // 长块的位置全都会被写出（参考实现 `OPUS_CLEAR(x, overlap)` 的语义）。
        let mut x = vec![0f32; n + overlap];
        if inp.transient {
            // 短块：B = M 个子块，系数在形状层就按块交错，所以 stride = B
            let b = m;
            let shift = mdct.max_shift();
            for k in 0..b {
                mdct.backward(&fch[k..], &mut x[SHORT_MDCT as usize * k..], shift, b);
            }
        } else {
            let shift = mdct.max_shift() - inp.lm as usize;
            mdct.backward(fch, &mut x, shift, 1);
        }
        let (hist, tail) = state.decode_mem[ch].split_at_mut(DECODE_BUFFER_SIZE);
        let out = &mut hist[frame_start..frame_start + n];
        for (j, o) in out.iter_mut().take(overlap).enumerate() {
            *o = x[j] + tail[j];
        }
        out[overlap..].copy_from_slice(&x[overlap..n]);
        tail[..overlap].copy_from_slice(&x[n..n + overlap]);

        // 后置滤波（两段：前 120 个样本从上一帧参数过渡到本帧参数；其余用本帧参数）。
        // T0/T1 传 `_old`/当前帧的周期（可能还没钳过 —— 参考实现里新解出的 pitch
        // 是下一帧才钳，所以这里可能为 0：增益也为 0，读到 x[i] 自身但乘 0，安全）。
        comb_filter(
            &mut state.decode_mem[ch],
            frame_start,
            state.postfilter_period_old as usize,
            state.postfilter_period as usize,
            SHORT_MDCT as usize,
            state.postfilter_gain_old,
            state.postfilter_gain,
            state.postfilter_tapset_old,
            state.postfilter_tapset,
            mdct.window(),
            overlap,
        );
        if inp.lm != 0 {
            comb_filter(
                &mut state.decode_mem[ch],
                frame_start + SHORT_MDCT as usize,
                state.postfilter_period as usize,
                inp.postfilter.pitch as usize,
                n - SHORT_MDCT as usize,
                state.postfilter_gain,
                inp.postfilter.gain,
                state.postfilter_tapset,
                inp.postfilter.tapset,
                mdct.window(),
                overlap,
            );
        }
    }

    // ③ 后置滤波状态推进（参考实现那段 `postfilter_*_old = ...`）
    state.postfilter_period_old = state.postfilter_period;
    state.postfilter_gain_old = state.postfilter_gain;
    state.postfilter_tapset_old = state.postfilter_tapset;
    state.postfilter_period = inp.postfilter.pitch;
    state.postfilter_gain = inp.postfilter.gain;
    state.postfilter_tapset = inp.postfilter.tapset;
    if inp.lm != 0 {
        state.postfilter_period_old = state.postfilter_period;
        state.postfilter_gain_old = state.postfilter_gain;
        state.postfilter_tapset_old = state.postfilter_tapset;
    }

    // ④ 去加重 + 转 i16
    deemphasis(state, n, cc, pcm);
}

/// 去加重（参考实现 `celt.c` 的 `deemphasis`）+ 转 i16。
///
/// 读的是 `out_syn[c] = out_mem[c] + MAX_PERIOD - N`，也就是 `decode_mem[c]` 的
/// 最后 `n` 个样本 —— 正常路径与丢包隐藏路径写进去的位置**恰好相同**
/// （`DECODE_BUFFER_SIZE - n`），所以这一份代码两处共用。
pub(crate) fn deemphasis(state: &mut SynthState, n: usize, out_channels: usize, pcm: &mut [i16]) {
    let frame_start = DECODE_BUFFER_SIZE - n;
    for ch in 0..out_channels {
        let mut mem = state.preemph_mem[ch];
        for j in 0..n {
            let x = state.decode_mem[ch][frame_start + j];
            let tmp = x + mem;
            mem = PREEMPH[0] * tmp - PREEMPH[1] * x;
            let tmp = PREEMPH[3] * tmp; // SHL32(...,2) 在浮点构建下是空操作
            pcm[j * out_channels + ch] = float_to_i16(tmp);
        }
        state.preemph_mem[ch] = mem;
    }
}

/// 参考实现 `deemphasis` 的输出换算（`SCALEOUT` + `FLOAT2INT16`）。
///
/// `SCALEOUT` 乘 1/32768、`FLOAT2INT16` 又乘回 32768（都是 2 的幂，逐位精确），
/// 中间夹一次 clamp；最后 `floor(.5+x)` 是参考构建（没定义 `HAVE_LRINTF`）用的取整。
pub(crate) fn float_to_i16(x: f32) -> i16 {
    let scaled = (x * (1.0 / 32768.0)) * 32768.0;
    let clamped = scaled.clamp(-32768.0, 32767.0);
    (0.5 + clamped).floor() as i32 as i16
}

/// 后置滤波（参考实现 `comb_filter`）：梳状滤波器，`y` 与 `x` 是同一块内存。
///
/// `buf[frame_start..]` 是本帧的 `n` 个样本；`x[i-T]` 式的回看直接落在历史里。
/// 前 `overlap` 个样本用窗做两套参数的交叉淡化（`f = window[i]²`），其余只用后者。
#[allow(clippy::too_many_arguments)]
pub(crate) fn comb_filter(
    buf: &mut [f32],
    frame_start: usize,
    t0: usize,
    t1: usize,
    n: usize,
    g0: f32,
    g1: f32,
    tapset0: u32,
    tapset1: u32,
    window: &[f32],
    overlap: usize,
) {
    let g00 = g0 * COMB_GAINS[tapset0 as usize][0];
    let g01 = g0 * COMB_GAINS[tapset0 as usize][1];
    let g02 = g0 * COMB_GAINS[tapset0 as usize][2];
    let g10 = g1 * COMB_GAINS[tapset1 as usize][0];
    let g11 = g1 * COMB_GAINS[tapset1 as usize][1];
    let g12 = g1 * COMB_GAINS[tapset1 as usize][2];
    let at = |i: isize| (frame_start as isize + i) as usize;
    for (i, &wi) in window.iter().enumerate().take(overlap) {
        let f = wi * wi;
        let i = i as isize;
        buf[at(i)] = buf[at(i)]
            + ((1.0 - f) * g00) * buf[at(i - t0 as isize)]
            + ((1.0 - f) * g01) * buf[at(i - t0 as isize - 1)]
            + ((1.0 - f) * g01) * buf[at(i - t0 as isize + 1)]
            + ((1.0 - f) * g02) * buf[at(i - t0 as isize - 2)]
            + ((1.0 - f) * g02) * buf[at(i - t0 as isize + 2)]
            + (f * g10) * buf[at(i - t1 as isize)]
            + (f * g11) * buf[at(i - t1 as isize - 1)]
            + (f * g11) * buf[at(i - t1 as isize + 1)]
            + (f * g12) * buf[at(i - t1 as isize - 2)]
            + (f * g12) * buf[at(i - t1 as isize + 2)];
    }
    for i in overlap..n {
        let i = i as isize;
        buf[at(i)] = buf[at(i)]
            + g10 * buf[at(i - t1 as isize)]
            + g11 * buf[at(i - t1 as isize - 1)]
            + g11 * buf[at(i - t1 as isize + 1)]
            + g12 * buf[at(i - t1 as isize - 2)]
            + g12 * buf[at(i - t1 as isize + 2)];
    }
}

/// `comb_filter` 的"输出写到另一块缓冲"的形态（参考实现里 `y` 与 `x` 不同的那一次调用）。
///
/// 丢包隐藏用它把"上一帧 MDCT 尾巴"过一遍后置滤波（注意增益是**负**的 —— 那是
/// 预滤波，与之配对的后置滤波在下一帧真正解码时做）：
///
/// ```text
/// comb_filter(e, out_mem + MAX_PERIOD, period, period, overlap,
///             -gain, -gain, tapset, tapset, NULL, 0)
/// ```
///
/// `window = NULL`、`overlap = 0` 意味着只走第二个循环（前 `overlap` 个样本不做过渡）。
pub(crate) fn comb_filter_into(
    y: &mut [f32],
    x: &[f32],
    x_start: usize,
    t1: usize,
    g1: f32,
    tapset1: u32,
    n: usize,
) {
    let g10 = g1 * COMB_GAINS[tapset1 as usize][0];
    let g11 = g1 * COMB_GAINS[tapset1 as usize][1];
    let g12 = g1 * COMB_GAINS[tapset1 as usize][2];
    let at = |i: isize| (x_start as isize + i) as usize;
    for i in 0..n {
        let i = i as isize;
        y[i as usize] = x[at(i)]
            + g10 * x[at(i - t1 as isize)]
            + g11 * x[at(i - t1 as isize - 1)]
            + g11 * x[at(i - t1 as isize + 1)]
            + g12 * x[at(i - t1 as isize - 2)]
            + g12 * x[at(i - t1 as isize + 2)];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::opus::mdct::Mdct;

    fn silent_input<'a>(freq: &'a [Vec<f32>], lm: u32) -> SynthInput<'a> {
        SynthInput {
            stream_channels: freq.len(),
            out_channels: freq.len(),
            lm,
            transient: false,
            start: 0,
            end: 21,
            postfilter: PostFilter { pitch: 0, gain: 0.0, tapset: 0 },
            freq,
        }
    }

    #[test]
    fn zero_input_stays_zero() {
        let mdct = Mdct::new_48k();
        let mut state = SynthState::default();
        let freq = vec![vec![0f32; 960], vec![0f32; 960]];
        let mut pcm = vec![0i16; 2 * 960];
        let inp = silent_input(&freq, 3);
        synth_frame(&mut state, &mdct, &inp, &mut pcm);
        assert!(pcm.iter().all(|&v| v == 0), "全零输入应当出全零 PCM");
        assert!(state.decode_mem[0][DECODE_BUFFER_SIZE..].iter().all(|&v| v == 0.0));
        assert!(state.preemph_mem.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn mono_frame_is_duplicated_to_both_channels() {
        let mdct = Mdct::new_48k();
        let mut state = SynthState::default();
        let mut freq = vec![vec![0f32; 240]];
        freq[0][10] = 1.0;
        let mut pcm = vec![0i16; 2 * 240];
        let inp = SynthInput { out_channels: 2, ..silent_input(&freq, 1) };
        synth_frame(&mut state, &mdct, &inp, &mut pcm);
        for j in 0..240 {
            assert_eq!(pcm[2 * j], pcm[2 * j + 1], "第 {j} 个样本双声道不一致");
        }
        assert!(pcm.iter().any(|&v| v != 0), "非零输入应当出非零 PCM");
    }

    /// 后置滤波是**就地递归**的：`y[i-T]` 读到的是已经滤过的值。
    ///
    /// 构造：把 `x[frame_start]` 放一个脉冲、周期 15、增益 1、tapset 2。
    /// `y[15] = gains[2][0] = 0.7998046875`（两组参数相同、f 项互相抵消），
    /// `y[30]` 只能来自"读到滤过的 x[15]"—— 非递归实现会得到 0。
    #[test]
    #[allow(clippy::excessive_precision)] // 0.7998046875 是参考表里的值（照抄）
    fn comb_filter_is_recursive_in_place() {
        let mdct = Mdct::new_48k();
        let window = mdct.window();
        let mut buf = vec![0f32; CHANNEL_BUF];
        let frame_start = DECODE_BUFFER_SIZE - 120;
        buf[frame_start] = 1.0;
        comb_filter(&mut buf, frame_start, 15, 15, 120, 1.0, 1.0, 2, 2, window, 120);
        assert!((buf[frame_start] - 1.0).abs() < 1e-6, "y[0] = {}", buf[frame_start]);
        assert!(
            (buf[frame_start + 15] - 0.7998046875).abs() < 1e-6,
            "y[15] = {}",
            buf[frame_start + 15]
        );
        assert!(
            buf[frame_start + 30] > 0.5,
            "y[30] = {}（若是 0，说明读的是未滤过的历史）",
            buf[frame_start + 30]
        );
    }

    #[test]
    fn postfilter_state_advances_like_the_reference() {
        let mdct = Mdct::new_48k();
        let mut state = SynthState::default();
        let freq = vec![vec![0f32; 960], vec![0f32; 960]];
        // 20 ms（LM=3）：收尾时 _old 也被设成本帧
        let inp = SynthInput {
            postfilter: PostFilter { pitch: 100, gain: 0.5, tapset: 1 },
            ..silent_input(&freq, 3)
        };
        synth_frame(&mut state, &mdct, &inp, &mut vec![0i16; 2 * 960]);
        assert_eq!(state.postfilter(), (100, 0.5, 1));
        // 再来一帧、参数清零：老的被钳到 15（0 会被钳），当前的仍是 100
        let inp = silent_input(&freq, 3);
        synth_frame(&mut state, &mdct, &inp, &mut vec![0i16; 2 * 960]);
        assert_eq!(state.postfilter(), (0, 0.0, 0));
        // LM=0（2.5 ms）：_old 保留上一帧的值（不跟当前帧走）
        let mut state = SynthState::default();
        let freq = vec![vec![0f32; 120]];
        let inp = SynthInput {
            lm: 0,
            postfilter: PostFilter { pitch: 40, gain: 0.3, tapset: 0 },
            ..silent_input(&freq, 0)
        };
        synth_frame(&mut state, &mdct, &inp, &mut vec![0i16; 120]);
        let inp = SynthInput {
            lm: 0,
            postfilter: PostFilter { pitch: 60, gain: 0.2, tapset: 0 },
            ..silent_input(&freq, 0)
        };
        synth_frame(&mut state, &mdct, &inp, &mut vec![0i16; 120]);
        // 第二帧之后：当前 = 60；_old = 40（LM=0 不覆盖）
        assert_eq!(state.postfilter(), (60, 0.2, 0));
        assert_eq!(state.postfilter_gain_old, 0.3);
        assert_eq!(state.postfilter_period_old, 40);
    }
}
