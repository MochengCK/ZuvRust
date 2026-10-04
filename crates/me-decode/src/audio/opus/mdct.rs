//! CELT 的**合成变换**（参考实现 `mdct.c` + `kiss_fft.c`）。
//!
//! 解码侧这条链是：归一化频谱 → 反归一化 → **IMDCT**（本文件的 [`Mdct::backward`]）
//! → 重叠相加（`synth.rs`）。参考实现的 IMDCT 走"N/4 点复数 FFT"这条路：
//!
//! ```text
//! 预旋转（把 N/2 个实数两两配成 N/4 个复数）
//!   → 未归一化的逆 FFT（N/4 点）
//!   → 后旋转 → 去洗牌 → 两侧镜像 + 加窗（TDAC）
//! ```
//!
//! ## 为什么 FFT 是自己写的
//!
//! 参考实现里的 kiss FFT 是混合基（radix 4/2/3/5）—— 48 kHz 模式下 N/4 = 480 = 2⁵·3·5，
//! **不是 2 的幂**，仓库里 AAC 那条 radix-2 的 FFT 顶不上。这里按标准的 Cooley-Tukey
//! 递归分解自己实现（因子 2/3/5 / 素数兜底），数学定义就是**未归一化的逆 DFT**
//! —— kiss 的 `opus_ifft` 明确不做 1/N 缩放，`mdct.c` 依赖这一点（注释原话
//! "This one should *not* downscale even in fixed-point"），所以我也不能顺手除 N。
//!
//! ## 三个表都按参考实现的公式在运行时算（不是抄数字）
//!
//! 精度路径逐个照抄，**三条路径互不相同，别混**：
//!
//! | 表 | 参考实现 | 精度路径 |
//! |---|---|---|
//! | IMDCT 的 `trig[i] = cos(2*PI*i/n)` | `clt_mdct_init` | 相位是 **float** 算的（`PI` 是 `3.141592653f`），`cos` 在 double，压 f32 |
//! | FFT 旋转因子 `W = e^{-2πi·i/n}` | `compute_twiddles` | `pi` 是 **double** 字面量，整条 double 算完压 f32 |
//! | 窗 `sin(.5*M_PI*sin(.5*M_PI*(i+.5)/overlap)²)` | `opus_custom_mode_create` | `M_PI` 是 double，整条 double |
//!
//! 单测里对着参考实现的静态表（`static_modes_float.h` 的 `mdct_twiddles960` /
//! `fft_twiddles48000_960` / `window120`）留了几个**回读**值：表算错时不用等到 PCM 对拍。
//!
//! ## 短块（瞬态帧）的交错
//!
//! 瞬态帧的系数在形状层就按"块交错"排好（`X[j*B + b]`），所以 `backward` 用 `stride=B`
//! 读输入、结果则连续写进 `out`（参考实现里输出指针不带 stride）。`shift` 决定块大小：
//! 瞬态帧所有子块都用 `maxLM`，非瞬态帧用 `maxLM-LM`。

use super::mathops::{PI_F32, PI_F64};

/// 48 kHz 标准模式的 MDCT 长度（`mode->mdct.n`）。
pub const MODE_N: usize = 1920;
/// 48 kHz 标准模式的 `maxLM`（短块最多 8 个 = 2^3）。
pub const MODE_MAX_SHIFT: usize = 3;
/// 48 kHz 标准模式的 `overlap`（也是窗的长度）。
pub const MODE_OVERLAP: usize = 120;

/// 复数：`[实部, 虚部]`（对应 kiss 的 `kiss_fft_cpx`）。
type Cpx = [f32; 2];

/// `n` 的最小质因子（本模块用到的长度只会是 2^a·3^b·5^c；素数走兜底分支）。
fn smallest_factor(n: usize) -> usize {
    let mut p = 2usize;
    while p * p <= n {
        if n.is_multiple_of(p) {
            return p;
        }
        p += if p == 2 { 1 } else { 2 };
    }
    n
}

/// 混合基 FFT —— 只提供**未归一化的逆变换**，因为解码侧（IMDCT）只用这一种。
struct Fft {
    n: usize,
    /// 旋转因子 `W_n^i = e^{-2πi·i/n}`，按 kiss 的口径（double 算、存 f32）。
    tw: Vec<Cpx>,
}

impl Fft {
    /// 按参考实现 `compute_twiddles` 的公式建表（double 路径）。
    fn new(n: usize) -> Self {
        let tw = (0..n)
            .map(|i| {
                let phase = ((-2.0 * PI_F64) / n as f64) * i as f64;
                [phase.cos() as f32, phase.sin() as f32]
            })
            .collect();
        Self { n, tw }
    }

    /// 从更大的表按整数步长取子表 —— 参考实现让子 FFT 复用 `kfft[0]` 的表并给索引
    /// 乘 `2^shift`，取出来的值一样（这里直接抽出来，省得每层都除）。
    fn derived(&self, n: usize) -> Self {
        debug_assert!(n > 0 && self.n.is_multiple_of(n), "子 FFT 长度必须整除父表");
        let step = self.n / n;
        Self { n, tw: (0..n).map(|i| self.tw[i * step]).collect() }
    }

    /// 未归一化逆变换（就地）：`X[k] = Σ_j x[j]·e^{+2πijk/n}`。
    fn inverse(&self, buf: &mut [Cpx]) {
        debug_assert_eq!(buf.len(), self.n);
        let mut out = vec![[0f32; 2]; self.n];
        self.rec(buf, 1, &mut out, self.n);
        buf.copy_from_slice(&out);
    }

    /// DIT 递归：从 `input` 里以 `stride` 为步长取 `n` 个点，结果连续写进 `out`。
    ///
    /// 分解式（p 是最小质因子、m = n/p，把序列按模 p 分成 p 条）：
    /// `X[k] = Σ_q W_n^{qk} · A_q[k mod m]`，其中 `A_q` 是第 q 条子序列的 m 点变换。
    fn rec(&self, input: &[Cpx], stride: usize, out: &mut [Cpx], n: usize) {
        if n == 1 {
            out[0] = input[0];
            return;
        }
        let p = smallest_factor(n);
        let m = n / p;
        let mut sub = vec![[0f32; 2]; n];
        for q in 0..p {
            self.rec(&input[q * stride..], stride * p, &mut sub[q * m..(q + 1) * m], m);
        }
        // 本层的 n 比计划长度小的时候，旋转因子在表里要隔 step 取
        let step = self.n / n;
        for k in 0..n {
            let (mut re, mut im) = (0f32, 0f32);
            for q in 0..p {
                let w = self.tw[((q * k) % n) * step];
                let a = sub[q * m + (k % m)];
                // 逆变换用旋转因子的共轭（参考实现里对应 `C_MULC`）
                re += a[0] * w[0] + a[1] * w[1];
                im += a[1] * w[0] - a[0] * w[1];
            }
            out[k] = [re, im];
        }
    }
}

/// CELT 模式的变换相关表：三角表、各级 FFT、窗。解码器整个生命周期只用 48 kHz 一套，
/// 但构造参数保留通用形式，单测能拿小尺寸验证。
pub struct Mdct {
    /// 完整 MDCT 长度（48 kHz 是 1920）。
    n: usize,
    /// 短块最多几分之一（`maxLM`，48 kHz 是 3）。
    max_shift: usize,
    overlap: usize,
    /// `cos(2*PI*i/n)`，i = 0..=n/4（参考实现 `clt_mdct_init` 的 `l->trig`）。
    trig: Vec<f32>,
    /// 每级 `shift` 的 FFT：点数 `n/4 >> shift`。
    ffts: Vec<Fft>,
    /// 窗（长度 = overlap）。
    window: Vec<f32>,
}

impl Mdct {
    /// 48 kHz 标准模式的表。
    pub fn new_48k() -> Self {
        Self::new(MODE_N, MODE_MAX_SHIFT, MODE_OVERLAP)
    }

    pub fn new(n: usize, max_shift: usize, overlap: usize) -> Self {
        // trig[i] = cos(2*PI*i/n)：相位在 **float** 里算（PI 是 3.141592653f），cos 在 double，
        // 最后压回 f32 —— 别改成"f64 里算整条"，最后一位会不一样。
        let n4 = n >> 2;
        let trig = (0..=n4)
            .map(|i| {
                let phase = ((2.0f32 * PI_F32) * i as f32) / n as f32;
                (phase as f64).cos() as f32
            })
            .collect();

        // 子 FFT 复用最大那张表（和参考实现 `opus_fft_alloc_twiddles(..., base)` 一致）。
        let base = Fft::new(n4);
        let ffts = (0..=max_shift).map(|s| base.derived(n4 >> s)).collect();

        // 窗：sin(.5*M_PI*sin(.5*M_PI*(i+.5)/overlap)²)。乘法顺序照抄 C（先乘 .5*M_PI、
        // 再乘 (i+.5)、再除 overlap），回合差一个 ulp 都算我输。
        let window = (0..overlap)
            .map(|i| {
                let s = (0.5 * PI_F64) * (i as f64 + 0.5) / overlap as f64;
                let s = s.sin();
                ((0.5 * PI_F64) * s * s).sin() as f32
            })
            .collect();

        Self { n, max_shift, overlap, trig, ffts, window }
    }

    /// `mode->overlap`（既是最长重叠，也是窗长）。
    pub fn overlap(&self) -> usize {
        self.overlap
    }

    /// `mode->window`（后置滤波的交叉淡化也要用）。
    pub fn window(&self) -> &[f32] {
        &self.window
    }

    /// 本模式允许的最大 `shift`（= `maxLM`）。
    pub fn max_shift(&self) -> usize {
        self.max_shift
    }

    /// 参考实现 `clt_mdct_backward`：反变换一个（子）块。
    ///
    /// - `shift`：瞬态帧传 `maxLM`（短块），非瞬态传 `maxLM - LM`；
    /// - `stride`：短块时是块数 B（输入按块交错），长块是 1；
    /// - `input`：本块要用的系数（短块时已经偏移到 `&X[b]`）；
    /// - `out`：写出的缓冲区（长度至少 `N/2 + overlap` —— 内部 N 是调用者帧长的两倍，
    ///   写入范围是 `[0, N/2 + overlap)`；参考实现里是 `x` 临时缓冲或 `x + 120b`）。
    ///
    /// 输出端的下标语义完全照抄参考实现：内部先把输出指针左移 `(N2-overlap)/2`，
    /// 再分左右两半做镜像；两侧的 `+=` 与 `=` 是刻意不同的（左半靠 `+=` 累加短块的
    /// 交叠、右半首次写入用 `=`），别"统一"成一种。
    pub fn backward(&self, input: &[f32], out: &mut [f32], shift: usize, stride: usize) {
        let n = self.n >> shift;
        let n2 = n >> 1;
        let n4 = n >> 2;
        debug_assert!(out.len() >= n2 + self.overlap, "输出缓冲区不足：{} < {}", out.len(), n2 + self.overlap);
        debug_assert!(input.len() > stride * (n2 - 1), "输入不够读：{}", input.len());
        // `sine = (float)2*PI*(.125f)/N` —— 又是 float 路径
        let sine = ((2.0f32 * PI_F32) * 0.125) / n as f32;
        let t = &self.trig;

        // ① 预旋转：in 的首尾两端各取 n4 个实数配成复数（等价于 N/2 点实序列的打包）
        let mut f = vec![[0f32; 2]; n4];
        for (i, slot) in f.iter_mut().enumerate() {
            let x1 = input[2 * stride * i];
            let x2 = input[stride * (n2 - 1 - 2 * i)];
            let c1 = t[i << shift];
            let c2 = t[(n4 - i) << shift];
            let yr = -x2 * c1 + x1 * c2;
            let yi = -x2 * c2 - x1 * c1;
            *slot = [yr - yi * sine, yi + yr * sine];
        }

        // ② 未归一化逆 FFT（注意：不除点数）
        self.ffts[shift].inverse(&mut f);

        // ③ 后旋转（就地）
        for i in 0..n4 {
            let (re, im) = (f[i][0], f[i][1]);
            let c1 = t[i << shift];
            let c2 = t[(n4 - i) << shift];
            let yr = re * c1 - im * c2;
            let yi = im * c1 + re * c2;
            f[i] = [yr - yi * sine, yi + yr * sine];
        }

        // ④ 去洗牌：n4 个复数摊成 n2 个实数（前段取负实部，后段倒序取虚部）
        let mut f2 = vec![0f32; n2];
        for i in 0..n4 {
            f2[2 * i] = -f[i][0];
            f2[2 * i + 1] = f[n4 - 1 - i][1];
        }

        // ⑤ 两侧镜像 + 加窗。参考实现里 `out` 指针先左移 `(N2-overlap)>>1`，下面所有
        //    下标都在那个坐标系里（可能为负的只是坐标原点；实际写入都落在 out[0..]）。
        //    `o` 负责把它换算回真实下标。
        let base = -(((n2 - self.overlap) >> 1) as isize);
        let o = |i: isize| (base + i) as usize;
        let half = self.overlap / 2;
        let w = &self.window;

        // 左半：前半段原样搬运，后半段一边镜像回绕、一边乘上升/降窗
        {
            let mut fp = n4 as isize - 1;
            let mut xp = n2 as isize - 1;
            let mut yp = n4 as isize - half as isize;
            for _ in 0..n4 - half {
                out[o(xp)] = f2[fp as usize];
                xp -= 1;
                fp -= 1;
            }
            for k in 0..half {
                let x1 = f2[fp as usize];
                out[o(yp)] += -w[k] * x1;
                yp += 1;
                out[o(xp)] += w[self.overlap - 1 - k] * x1;
                xp -= 1;
                fp -= 1;
            }
        }
        // 右半：镜像到尾部（这一半的写入是"赋值"而不是累加，照抄参考实现）
        {
            let mut fp = n4 as isize;
            let mut xp = n2 as isize;
            let mut yp = n as isize - 1 - (n4 as isize - half as isize);
            for _ in 0..n4 - half {
                out[o(xp)] = f2[fp as usize];
                xp += 1;
                fp += 1;
            }
            for k in 0..half {
                let x2 = f2[fp as usize];
                out[o(yp)] = w[k] * x2;
                yp -= 1;
                out[o(xp)] = w[self.overlap - 1 - k] * x2;
                xp += 1;
                fp += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 确定性伪随机（测试不需要真随机，只要能复现）。
    fn lcg(state: &mut u32) -> f32 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (*state >> 8) as f32 / (1 << 24) as f32 - 0.5
    }

    /// 朴素逆 DFT（double）—— 只用来给 FFT 当金标准。
    fn naive_inverse(input: &[Cpx]) -> Vec<Cpx> {
        let n = input.len();
        (0..n)
            .map(|k| {
                let mut re = 0f64;
                let mut im = 0f64;
                for (j, x) in input.iter().enumerate() {
                    let phase = 2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
                    let (s, c) = phase.sin_cos();
                    re += x[0] as f64 * c - x[1] as f64 * s;
                    im += x[0] as f64 * s + x[1] as f64 * c;
                }
                [re as f32, im as f32]
            })
            .collect()
    }

    /// 用"朴素 DFT"把 `clt_mdct_backward` 的整条流水线在 f64 里再走一遍 ——
    /// 验的是索引/交错/镜像这些**位与步长**，旋转因子的数值精度由表回读测试兜底。
    fn naive_backward(m: &Mdct, input: &[f32], shift: usize, stride: usize) -> Vec<f32> {
        let n = m.n >> shift;
        let (n2, n4) = (n >> 1, n >> 2);
        let sine = ((2.0f32 * PI_F32) * 0.125f32 / n as f32) as f64;
        let t: Vec<f64> = m.trig.iter().map(|&v| v as f64).collect();
        let mut f = vec![[0f64; 2]; n4];
        for i in 0..n4 {
            let x1 = input[2 * stride * i] as f64;
            let x2 = input[stride * (n2 - 1 - 2 * i)] as f64;
            let (c1, c2) = (t[i << shift], t[(n4 - i) << shift]);
            let yr = -x2 * c1 + x1 * c2;
            let yi = -x2 * c2 - x1 * c1;
            f[i] = [yr - yi * sine, yi + yr * sine];
        }
        // 逆 DFT（未归一化，且用共轭 → 指数取正号）
        let f = {
            let nf = &f;
            (0..n4)
                .map(|k| {
                    let mut re = 0f64;
                    let mut im = 0f64;
                    for (j, x) in nf.iter().enumerate() {
                        let phase = 2.0 * std::f64::consts::PI * (j * k) as f64 / n4 as f64;
                        let (s, c) = phase.sin_cos();
                        re += x[0] * c - x[1] * s;
                        im += x[0] * s + x[1] * c;
                    }
                    [re, im]
                })
                .collect::<Vec<_>>()
        };
        let mut f: Vec<[f64; 2]> = f;
        for i in 0..n4 {
            let (re, im) = (f[i][0], f[i][1]);
            let (c1, c2) = (t[i << shift], t[(n4 - i) << shift]);
            let yr = re * c1 - im * c2;
            let yi = im * c1 + re * c2;
            f[i] = [yr - yi * sine, yi + yr * sine];
        }
        let mut f2 = vec![0f64; n2];
        for i in 0..n4 {
            f2[2 * i] = -f[i][0];
            f2[2 * i + 1] = f[n4 - 1 - i][1];
        }
        let overlap = m.overlap;
        let base = -(((n2 - overlap) >> 1) as isize);
        let idx = |i: isize| (base + i) as usize;
        let mut out = vec![0f64; n];
        let w: Vec<f64> = m.window.iter().map(|&v| v as f64).collect();
        let half = overlap / 2;
        {
            let mut fp = n4 as isize - 1;
            let mut xp = n2 as isize - 1;
            let mut yp = n4 as isize - half as isize;
            for _ in 0..n4 - half {
                out[idx(xp)] = f2[fp as usize];
                xp -= 1;
                fp -= 1;
            }
            for k in 0..half {
                let x1 = f2[fp as usize];
                out[idx(yp)] += -w[k] * x1;
                yp += 1;
                out[idx(xp)] += w[overlap - 1 - k] * x1;
                xp -= 1;
                fp -= 1;
            }
        }
        {
            let mut fp = n4 as isize;
            let mut xp = n2 as isize;
            let mut yp = n as isize - 1 - (n4 as isize - half as isize);
            for _ in 0..n4 - half {
                out[idx(xp)] = f2[fp as usize];
                xp += 1;
                fp += 1;
            }
            for k in 0..half {
                let x2 = f2[fp as usize];
                out[idx(yp)] = w[k] * x2;
                yp -= 1;
                out[idx(xp)] = w[overlap - 1 - k] * x2;
                xp += 1;
                fp += 1;
            }
        }
        out.iter().map(|&v| v as f32).collect()
    }

    #[test]
    fn fft_matches_naive_inverse_dft() {
        for n in [15usize, 30, 60, 120, 240, 480] {
            let plan = Fft::new(n);
            let mut st = 12345u32;
            let input: Vec<Cpx> = (0..n).map(|_| [lcg(&mut st), lcg(&mut st)]).collect();
            let want = naive_inverse(&input);
            let mut got = input.clone();
            plan.inverse(&mut got);
            for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                let d = ((g[0] - w[0]).abs()).max((g[1] - w[1]).abs());
                assert!(d < 1e-4, "n={n} 第 {j} 点：{g:?} vs {w:?}（差 {d}）");
            }
        }
    }

    #[test]
    fn fft_twiddles_match_reference_table() {
        // 参考实现静态表 `fft_twiddles48000_960`（前几个值 + 半圈 + 最后一个）
        let f = Fft::new(480);
        assert!((f.tw[0][0] - 1.0).abs() < 1e-7 && f.tw[0][1].abs() < 1e-7);
        assert!((f.tw[1][0] - 0.99991433).abs() < 1e-6, "{:?}", f.tw[1]);
        assert!((f.tw[1][1] + 0.013089596).abs() < 1e-6, "{:?}", f.tw[1]);
        assert!((f.tw[240][0] + 1.0).abs() < 1e-5, "{:?}", f.tw[240]);
        assert!((f.tw[479][1] - 0.013089596).abs() < 1e-6, "{:?}", f.tw[479]);
    }

    #[test]
    #[allow(clippy::excessive_precision)] // 期望值是参考静态表原样抄下来的，不截断
    fn mdct_trig_matches_reference_table() {
        // 参考实现静态表 `mdct_twiddles960`（48000/960 模式）
        let m = Mdct::new_48k();
        assert_eq!(m.trig[0], 1.0);
        assert!((m.trig[1] - 0.99999465).abs() < 1e-7, "{}", m.trig[1]);
        assert!((m.trig[8] - 0.99965732).abs() < 1e-7, "{}", m.trig[8]);
        assert!((m.trig[120] - 0.92387953).abs() < 1e-7, "{}", m.trig[120]);
        assert!((m.trig[240] - 0.70710677).abs() < 1e-7, "{}", m.trig[240]);
        // cos(π/2) 在"float 算相位 + double 算 cos"下不是 0，参考表里就是 -4.371139e-08
        assert!((m.trig[480] + 4.371139e-08).abs() < 1e-9, "{}", m.trig[480]);
    }

    #[test]
    #[allow(clippy::excessive_precision)] // 同上
    fn window_matches_reference_table() {
        // 参考实现静态表 `window120`
        let m = Mdct::new_48k();
        assert!((m.window[0] - 6.7286966e-05).abs() < 1e-10, "{}", m.window[0]);
        assert!((m.window[1] - 0.00060551348).abs() < 1e-9, "{}", m.window[1]);
        assert!((m.window[59] - 0.6998001).abs() < 1e-6, "{}", m.window[59]);
        assert!((m.window[60] - 0.71433873).abs() < 1e-6, "{}", m.window[60]);
        assert!((m.window[119] - 1.0).abs() < 1e-6, "{}", m.window[119]);
        assert_eq!(m.overlap(), 120);
        assert_eq!(m.max_shift(), 3);
    }

    #[test]
    fn backward_matches_naive_pipeline() {
        // 三种典型调用：20 ms 长块（shift 0）、2.5 ms 长块（shift 3）、瞬态短块（stride=B）
        let m = Mdct::new_48k();
        let mut st = 987654321u32;
        for (shift, stride, len) in [(0usize, 1usize, 960usize), (3, 1, 120), (3, 8, 120)] {
            let input: Vec<f32> = (0..stride * (len - 1) + 1).map(|_| lcg(&mut st)).collect();
            let want = naive_backward(&m, &input, shift, stride);
            // 故意给大一截，末尾若被写到说明下标算错了
            let mut got = vec![0f32; 2 * len + 64];
            m.backward(&input, &mut got, shift, stride);
            let mut max_d = 0f32;
            for (g, w) in got.iter().zip(want.iter()) {
                max_d = max_d.max((g - w).abs());
            }
            assert!(max_d < 1e-3, "shift={shift} stride={stride}：最大差 {max_d}");
            for (i, v) in got.iter().enumerate().skip(2 * len) {
                assert_eq!(*v, 0.0, "越界写到了 out[{i}]");
            }
        }
    }
}
