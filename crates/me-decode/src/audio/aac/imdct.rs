//! AAC 的滤波器组：IMDCT + 窗 + 叠加。
//!
//! ## 为什么用"补零 FFT"这条路而不是教科书里的 N/4 复数 FFT
//!
//! 教科书写法（预旋转 + N/4 复数 FFT + 后旋转）系数极多，错一位要拿耳朵查。
//! 这里用一个**可以直接对上公式**的等价形式：
//!
//! ```text
//! y[n] = (2/N) Σ_{k<N/2} X[k]·cos(2π(n+n0)(k+½)/N)
//!      = 2·Re{ e^{iπ(n+n0)/N} · IDFT( X[k]·e^{i2πn0·k/N} 补零到 N )[n] }
//! ```
//!
//! 也就是：预旋转（N/2 次复数乘）→ 补零 → **N 点逆 FFT** → 后旋转取实部。
//! 代价是 N 点而不是 N/4 点 FFT，但每个系数都能和公式对上 —— 而单测里
//! 还放了一份 O(N²) 的直接公式实现做交叉验证。

/// 就地复数 FFT（radix-2，长度必须是 2 的幂）。
///
/// `inverse = true` 时做逆变换并**除以 N**（这样 `fft(fft(x, false), true) == x`）。
pub fn fft(re: &mut [f32], im: &mut [f32], inverse: bool) {
    let n = re.len();
    debug_assert_eq!(n, im.len());
    debug_assert!(n.is_power_of_two(), "FFT 长度必须是 2 的幂");
    if n <= 1 {
        return;
    }

    // 位反转置换
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // 蝶形
    let sign = if inverse { 1.0f32 } else { -1.0f32 };
    let mut len = 2usize;
    while len <= n {
        let ang = sign * 2.0 * std::f32::consts::PI / len as f32;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0usize;
        while i < n {
            let mut cur_r = 1.0f32;
            let mut cur_i = 0.0f32;
            for k in 0..len / 2 {
                let ur = re[i + k];
                let ui = im[i + k];
                let vr = re[i + k + len / 2] * cur_r - im[i + k + len / 2] * cur_i;
                let vi = re[i + k + len / 2] * cur_i + im[i + k + len / 2] * cur_r;
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                let nr = cur_r * wr - cur_i * wi;
                cur_i = cur_r * wi + cur_i * wr;
                cur_r = nr;
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let inv = 1.0 / n as f32;
        for i in 0..n {
            re[i] *= inv;
            im[i] *= inv;
        }
    }
}

/// 窗形状（0 = 正弦窗，1 = Kaiser-Bessel 派生窗）。
///
/// 这里只有"窗函数怎么算"，没有"窗序列" —— 序列（`OnlyLong`/`EightShort`/…）是语法层
/// 的概念，定义在 `aac` 模块的 `Sequence` 里，本文件不需要重复一份。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowShape {
    Sine,
    Kbd,
}

/// IMDCT：把 N/2 个频域系数变换成 N 个时域样本（**未加窗、未叠加**）。
///
/// `spec` 长度必须是 `N/2`，`out` 长度 `N`。
pub fn imdct(spec: &[f32], out: &mut [f32]) {
    let n = out.len();
    let half = n / 2;
    debug_assert_eq!(spec.len(), half);
    let n0 = (n / 4) as f32 + 0.5;

    // ① 预旋转：Z[k] = X[k]·e^{i2πn0·k/N}
    let mut re = vec![0.0f32; n];
    let mut im = vec![0.0f32; n];
    for k in 0..half {
        let ang = 2.0 * std::f32::consts::PI * n0 * k as f32 / n as f32;
        re[k] = spec[k] * ang.cos();
        im[k] = spec[k] * ang.sin();
    }
    // ② 补零到 N（后半已初始化为 0）后做 N 点逆 FFT
    fft(&mut re, &mut im, true);
    // ③ 后旋转：y[n] = 2·Re{ e^{iπ(n+n0)/N}·idft[n] }
    for i in 0..n {
        let ang = std::f32::consts::PI * (i as f32 + n0) / n as f32;
        let (s, c) = (ang.sin(), ang.cos());
        out[i] = 2.0 * (re[i] * c - im[i] * s);
    }
}

/// **直接按公式**算的 IMDCT（O(N²)）——只用于单测交叉验证，别在生产路径上用。
#[cfg(test)]
fn imdct_direct(spec: &[f32], out: &mut [f32]) {
    let n = out.len();
    let half = n / 2;
    let n0 = (n / 4) as f32 + 0.5;
    for (i, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (k, &x) in spec.iter().enumerate().take(half) {
            let ang = 2.0 * std::f32::consts::PI / n as f32
                * (i as f32 + n0)
                * (k as f32 + 0.5);
            acc += x * ang.cos();
        }
        *o = 2.0 / n as f32 * acc;
    }
}

/// 生成窗系数（长度 `n`，即整窗——左右各一半）。
///
/// - 正弦窗：`w[i] = sin(π/n·(i+0.5))`
/// - KBD：用 Kaiser 窗（β 由 `alpha` 定）做"派生"积分 —— 与规范 4.6.11.3.3 一致。
pub fn window (shape: WindowShape, n: usize, alpha: f32) -> Vec<f32> {
    let mut w = vec![0.0f32; n];
    let half = n / 2;
    match shape {
        WindowShape::Sine => {
            for (i, x) in w.iter_mut().enumerate() {
                *x = (std::f32::consts::PI / n as f32 * (i as f32 + 0.5)).sin();
            }
        }
        WindowShape::Kbd => {
            // Kaiser 窗（长度 n/2 + 1，规范里是 N/2 个点）→ 累加 → 归一 → 开方
            let m = half;
            let mut kaiser = vec![0.0f64; m + 1];
            let denom = bessel_i0(std::f64::consts::PI * alpha as f64);
            for (i, k) in kaiser.iter_mut().enumerate() {
                let r = 2.0 * i as f64 / m as f64 - 1.0;
                let t = 1.0 - r * r;
                *k = bessel_i0(std::f64::consts::PI * alpha as f64 * t.sqrt().max(0.0)) / denom;
            }
            let mut acc = 0.0f64;
            for k in kaiser.iter().take(m) {
                acc += k;
            }
            let mut running = 0.0f64;
            for i in 0..half {
                running += kaiser[i];
                let v = (running / acc).clamp(0.0, 1.0);
                w[i] = v.sqrt() as f32;
                w[n - 1 - i] = w[i];
            }
        }
    }
    w
}

/// 零阶修正贝塞尔函数（KBD 窗要用；级数展开到收敛）。
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let half_x = x / 2.0;
    for k in 1..64 {
        term *= (half_x / k as f64) * (half_x / k as f64);
        sum += term;
        if term < 1e-18 * sum {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_roundtrip() {
        let mut re = vec![1.0f32, 2.0, -0.5, 3.0, 0.25, -1.0, 0.0, 2.5];
        let mut im = vec![0.0f32; 8];
        let orig_re = re.clone();
        fft(&mut re, &mut im, false);
        // DC 分量应当是各元素之和
        assert!((re[0] - 7.25).abs() < 1e-4, "DC = {}, 期望 7.25", re[0]);
        fft(&mut re, &mut im, true);
        for i in 0..8 {
            assert!((re[i] - orig_re[i]).abs() < 1e-4, "往返第 {} 项不符", i);
            assert!(im[i].abs() < 1e-4);
        }
    }

    #[test]
    fn fft_matches_naive_dft() {
        // 与朴素 DFT 对比（长度 16，随机一些值）
        let n = 16usize;
        let mut re: Vec<f32> = (0..n).map(|i| ((i * 37 % 11) as f32) - 5.0).collect();
        let orig = re.clone();
        let mut im = vec![0.0f32; n];
        fft(&mut re, &mut im, false);
        for k in 0..n {
            let mut sr = 0.0f32;
            let mut si = 0.0f32;
            for i in 0..n {
                let ang = -2.0 * std::f32::consts::PI * (k * i) as f32 / n as f32;
                sr += orig[i] * ang.cos();
                si += orig[i] * ang.sin();
            }
            assert!((re[k] - sr).abs() < 1e-3 && (im[k] - si).abs() < 1e-3, "第 {k} 项不符");
        }
    }

    #[test]
    fn imdct_matches_direct_formula() {
        for n in [256usize, 2048] {
            let half = n / 2;
            // 用一组确定性的"谱"（避免随机性）
            let spec: Vec<f32> = (0..half)
                .map(|k| (((k * 13) % 29) as f32 - 14.0) / 7.0)
                .collect();
            let mut fast = vec![0.0f32; n];
            let mut slow = vec![0.0f32; n];
            imdct(&spec, &mut fast);
            imdct_direct(&spec, &mut slow);
            let mut worst = 0.0f32;
            for i in 0..n {
                worst = worst.max((fast[i] - slow[i]).abs());
            }
            assert!(worst < 1e-2, "N={n} 时与直接公式最大差 {worst}（应当只有浮点误差）");
        }
    }

    #[test]
    fn sine_window_is_symmetric_and_tapers() {
        let w = window(WindowShape::Sine, 2048, 4.0);
        assert_eq!(w.len(), 2048);
        assert!(w[0] > 0.0 && w[0] < 0.01, "正弦窗两端接近 0，实际 {}", w[0]);
        assert!((w[1023] - 1.0).abs() < 0.01, "中点接近 1，实际 {}", w[1023]);
        // 关于中心反对称（正弦窗特征）
        for i in 0..1024 {
            assert!((w[i] - w[2047 - i]).abs() < 1e-5);
        }
    }

    #[test]
    fn kbd_window_is_flat_in_the_middle() {
        let w = window(WindowShape::Kbd, 2048, 4.0);
        assert!(w[0] < 0.05, "KBD 窗起点接近 0，实际 {}", w[0]);
        // KBD 窗中间一段是 1（这是它与正弦窗的区别）
        for i in 900..1100 {
            assert!((w[i] - 1.0).abs() < 1e-3, "KBD 中点应当为 1（i={i} 是 {}）", w[i]);
        }
        for i in 0..1024 {
            assert!((w[i] - w[2047 - i]).abs() < 1e-5, "KBD 窗也要对称");
        }
    }

    #[test]
    fn bessel_i0_known_values() {
        // I0(0)=1、I0(1)≈1.2660658、I0(2)≈2.2795853
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-9);
        assert!((bessel_i0(1.0) - 1.2660658).abs() < 1e-6);
        assert!((bessel_i0(2.0) - 2.2795853).abs() < 1e-6);
    }
}
