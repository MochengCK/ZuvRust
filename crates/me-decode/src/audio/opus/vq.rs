//! CELT 的**矢量量化**原语（参考实现 `vq.c`）。
//!
//! 频带形状解码（[`super::bands::quant_band`]）调用的就是这里的几件东西：
//!
//! | 本文件 | 参考实现 | 干什么 |
//! |---|---|---|
//! | [`exp_rotation`] | `exp_rotation` | 把脉冲"铺开"（否则高能量带会听出金属味） |
//! | [`alg_unquant`] | `alg_unquant` | 解 PVQ 序号 → 归一化形状 + 扩散 + 折叠掩码 |
//! | [`renormalise_vector`] | `renormalise_vector` | 把矢量重新归一化到单位 L2 范数 |
//! | [`lcg_rand`] | `celt_lcg_rand` | 折叠/噪声填充用的伪随机数 |
//!
//! ## 为什么"扩散"要先归一化再旋转
//!
//! PVQ 解出来的是**整数脉冲矢量** `y`（L1 范数 = K）。直接拿它当形状，
//! 能量全挤在少数几个系数上，听起来是"嗡"的一声。参考实现的做法是：
//! 先把 `y` 归一化成单位 L2 范数的 `X`，再对 `X` 做一次**正交旋转**
//! （[`exp_rotation`]），把脉冲摊到整条带上。旋转的角度由
//! `len/(len + factor·K)` 决定 —— K 越大（脉冲越密）转得越多，
//! 于是"脉冲密集的带"会变得更像噪声，这正是 CELT 想要的听觉效果。
//!
//! ## 浮点构建下的几个关键化简（照抄参考实现，不要"优化"）
//!
//! 参考实现的浮点构建里 `SHR32/SHL32/PSHR32/VSHR32` 全是**空操作**、
//! `MULT16_16(a,b)`/`MULT16_16_Q15(a,b)` 就是普通乘法。所以：
//!
//! - `normalise_residual` 里那个"按位对齐"的 `k` 根本没被用到，
//!   实际就是 `X[i] = gain/√Ryy · y[i]`；
//! - `exp_rotation1` 里没有移位，`X[i] = c·x1 − s·x2`。
//!
//! 这些化简**不是**我们"数学等价地重写"，而是浮点构建本来就长这样 —— 照写才能对齐。

use me_core::Result;

use super::mathops::{celt_cos_norm, celt_rsqrt};

use super::cwrs::decode_pulses;
use super::entropy::RangeDecoder;

/// LCG（参考实现 `celt_lcg_rand`）。折叠和"没分到位"时的填补都靠它。
///
/// 它是**跨帧状态**：参考实现每帧结束时把解码器的最终区间状态写进 `st->rng`，
/// 下一帧从这里接着走 —— 所以对拍时种子必须逐帧一致，否则 X 从第一个折叠带就开始错。
#[inline]
pub fn lcg_rand(seed: u32) -> u32 {
    seed.wrapping_mul(1664525).wrapping_add(1013904223)
}

/// 扩散因子（参考实现 `SPREAD_FACTOR`）。spread 1/2/3 = LIGHT/NORMAL/AGGRESSIVE。
const SPREAD_FACTOR: [usize; 3] = [15, 10, 5];

/// 参考实现 `exp_rotation1` 的那一步（绕成环的两段写）。
///
/// 先把 `[0, len-stride)` 段整体转一遍，再从右往左转尾段 —— 两段拼起来才是一个
/// **首尾相接的循环旋转**（少了第二段，矢量两端会接不上，能量会漏出去）。
fn exp_rotation1(x: &mut [f32], len: usize, stride: usize, c: f32, s: f32) {
    for i in 0..len.saturating_sub(stride) {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 - s * x2;
    }
    // 第二段的循环次数恰好是 len-2*stride（起点 len-2*stride-1、终点 0，闭区间）
    for i in (0..len.saturating_sub(2 * stride)).rev() {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 - s * x2;
    }
}

/// 扩散：对归一化形状做一次正交旋转（参考实现 `exp_rotation`）。
///
/// `dir` 是旋转方向：**解码**用 `-1`（编码时用 `+1` 旋过去、解码旋回来）。
/// `stride = B`（短块数）是因为脉冲是按"时隙交错"排的，得逐条时隙分别旋转。
pub fn exp_rotation(x: &mut [f32], len_in: usize, dir: i32, stride: usize, k: usize, spread: u32) {
    // K 已经把整条带填满了（或根本没开扩散）就没什么可摊的
    if 2 * k >= len_in || spread == 0 {
        return;
    }
    let factor = SPREAD_FACTOR[(spread - 1) as usize];
    // celt_div(MULT16_16(Q15_ONE,len), len+factor*K) —— 浮点下就是普通除法
    let gain = len_in as f32 / (len_in + factor * k) as f32;
    let theta = 0.5 * gain * gain;
    // celt_cos_norm(t) = cos(π/2·t)，所以 s = cos(π/2·(1-t)) = sin(π/2·t)
    let c = celt_cos_norm(theta);
    let s = celt_cos_norm(1.0 - theta);

    // 带够长时再加一级"粗旋转"，等价于按 √(len/stride) 分组
    let mut stride2 = 0usize;
    if len_in >= 8 * stride {
        stride2 = 1;
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < len_in {
            stride2 += 1;
        }
    }

    let len = len_in / stride;
    for i in 0..stride {
        let seg = &mut x[i * len..i * len + len];
        if dir < 0 {
            if stride2 != 0 {
                exp_rotation1(seg, len, stride2, s, c);
            }
            exp_rotation1(seg, len, 1, c, s);
        } else {
            exp_rotation1(seg, len, 1, c, -s);
            if stride2 != 0 {
                exp_rotation1(seg, len, stride2, s, -c);
            }
        }
    }
}

/// 把单位范数的形状重新归一化到单位 L2 范数（参考实现 `renormalise_vector`）。
///
/// `gain` 一般是 `Q15ONE`（浮点构建下就是 1.0）。`EPSILON` 是为了让全零矢量
/// 不至于除出 `inf` —— 参考实现在浮点构建下取 `1e-15`。
pub fn renormalise_vector(x: &mut [f32], n: usize, gain: f32) {
    let mut e = 1e-15f32;
    for v in &x[..n] {
        e += v * v;
    }
    let g = celt_rsqrt(e) * gain;
    for v in x[..n].iter_mut() {
        *v *= g;
    }
}

/// 归一化残差（参考实现 `normalise_residual`，浮点构建）。
///
/// 浮点构建下 `VSHR32/PSHR32` 是空操作、那个"按位对齐"的 `k` 从未被真正用到，
/// 所以整件事就是 `X[i] = gain/√Ryy · y[i]`。
fn normalise_residual(iy: &[i32], x: &mut [f32], n: usize, ryy: f32, gain: f32) {
    let g = celt_rsqrt(ryy) * gain;
    for i in 0..n {
        x[i] = g * iy[i] as f32;
    }
}

/// 折叠掩码（参考实现 `extract_collapse_mask`）：第 i 位表示"第 i 个短块里还有非零系数"。
///
/// 它有两个用处：跨带折叠时用来算"能不能从这里折"，以及 `anti_collapse`
/// 用它判断哪些短块被量化成 0（塌陷）需要补噪。
fn extract_collapse_mask(iy: &[i32], n: usize, b: usize) -> u32 {
    if b <= 1 {
        return 1;
    }
    let n0 = n / b;
    let mut cm = 0u32;
    for i in 0..b {
        for j in 0..n0 {
            cm |= u32::from(iy[i * n0 + j] != 0) << i;
        }
    }
    cm
}

/// 解一条带的 PVQ 形状（参考实现 `alg_unquant`）。
///
/// `n` 是频带宽度（MDCT 系数个数），`k` 是脉冲数（来自比特分配），
/// `b` 是短块数，`spread` 是扩散决策，`gain` 是这一带的增益
/// （立体声分裂时 mid/side 各带自己的增益）。
/// 返回折叠掩码。
pub fn alg_unquant(
    x: &mut [f32],
    n: usize,
    k: usize,
    spread: u32,
    b: usize,
    dec: &mut RangeDecoder<'_>,
    gain: f32,
) -> Result<u32> {
    debug_assert!(k > 0, "alg_unquant 至少需要一个脉冲");
    debug_assert!(n > 1, "alg_unquant 的 N 至少是 2");
    let iy = decode_pulses(dec, n, k)?;
    let mut ryy = 0f32;
    for &v in &iy {
        ryy += (v * v) as f32;
    }
    normalise_residual(&iy, x, n, ryy, gain);
    // 解码侧是"旋回来"（编码侧旋过去的方向是 +1）
    exp_rotation(x, n, -1, b, k, spread);
    Ok(extract_collapse_mask(&iy, n, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg_is_the_reference_sequence() {
        // 参考实现的 1664525·seed + 1013904223（32 位回绕）
        assert_eq!(lcg_rand(0), 1013904223);
        assert_eq!(lcg_rand(1), 1664525 + 1013904223);
        assert_eq!(lcg_rand(0xFFFF_FFFF), 1664525u32.wrapping_mul(0xFFFF_FFFF).wrapping_add(1013904223));
    }

    #[test]
    fn collapse_mask_marks_nonzero_blocks() {
        // B=1 时恒为 1（没有短块可区分）
        assert_eq!(extract_collapse_mask(&[0, 0, 0, 0], 4, 1), 1);
        // B=4、N=4：每个短块 1 个系数
        assert_eq!(extract_collapse_mask(&[1, 0, 2, 0], 4, 4), 0b0101);
        assert_eq!(extract_collapse_mask(&[0, 0, 0, 0], 4, 4), 0);
        // B=2、N=4：短块 0 是前两个系数
        assert_eq!(extract_collapse_mask(&[0, 3, 0, 0], 4, 2), 0b01);
        assert_eq!(extract_collapse_mask(&[0, 3, 1, 0], 4, 2), 0b11);
    }

    #[test]
    fn renormalise_gives_unit_l2_norm() {
        let mut x = vec![3.0f32, 4.0, 0.0, 0.0];
        renormalise_vector(&mut x, 4, 1.0);
        let e: f32 = x[..4].iter().map(|v| v * v).sum();
        assert!((e - 1.0).abs() < 1e-6, "归一化之后 L2 范数应为 1，实际 {e}");

        // gain 会乘进去
        let mut y = vec![3.0f32, 4.0];
        renormalise_vector(&mut y, 2, 2.0);
        let e: f32 = y.iter().map(|v| v * v).sum();
        assert!((e - 4.0).abs() < 1e-5, "gain=2 时范数应为 2，实际 {}", e.sqrt());
    }

    #[test]
    fn rotation_preserves_energy() {
        // 正交旋转不改变 L2 范数 —— 这是"扩散不会动能量"的根据，
        // 也顺带证明两段拼接确实构成一个闭合的循环旋转
        let orig: Vec<f32> = (0..64).map(|i| ((i * 37 % 29) as f32) - 14.0).collect();
        for (stride, spread) in [(1usize, 2u32), (2, 2), (4, 3), (1, 1), (8, 2)] {
            let mut x = orig.clone();
            let before: f32 = x.iter().map(|v| v * v).sum();
            let len = x.len();
            exp_rotation(&mut x, len, -1, stride, 7, spread);
            let after: f32 = x.iter().map(|v| v * v).sum();
            assert!(
                (before - after).abs() < 1e-2 * before.max(1.0),
                "stride={stride} spread={spread}：能量从 {before} 变成 {after}"
            );
        }
    }

    #[test]
    fn rotation_is_a_noop_without_spread_or_when_full() {
        let orig: Vec<f32> = (0..32).map(|i| (i as f32) - 16.0).collect();
        // spread=0（SPREAD_NONE）不旋转
        let mut x = orig.clone();
        exp_rotation(&mut x, 32, -1, 1, 4, 0);
        assert_eq!(x, orig, "spread=0 时不该动");
        // 2K >= len 时也不旋转（脉冲已经够密，再摊也没意义）
        let mut x = orig.clone();
        exp_rotation(&mut x, 32, -1, 1, 16, 2);
        assert_eq!(x, orig, "2K>=len 时不该动");
    }

    #[test]
    fn rotation_and_its_inverse_cancel() {
        // 编码方向 (+1) 旋过去、解码方向 (-1) 旋回来应当还原
        // （注意两次调用是同一组 c/s，参考实现就是这样配的）
        let orig: Vec<f32> = (0..48).map(|i| ((i * 17 % 23) as f32) - 11.0).collect();
        for (stride, spread) in [(1usize, 2u32), (2, 3)] {
            let mut x = orig.clone();
            exp_rotation(&mut x, 48, 1, stride, 5, spread);
            exp_rotation(&mut x, 48, -1, stride, 5, spread);
            for (i, (a, b)) in x.iter().zip(orig.iter()).enumerate() {
                assert!((a - b).abs() < 1e-3, "第 {i} 个没还原：{a} vs {b}");
            }
        }
    }

    #[test]
    fn alg_unquant_produces_a_unit_shape_of_the_right_length() {
        let data: Vec<u8> = (0..256u16).map(|i| ((i * 71 + 3) % 256) as u8).collect();
        let mut dec = RangeDecoder::new(&data);
        let mut x = vec![0f32; 16];
        let cm = alg_unquant(&mut x, 16, 3, 2, 1, &mut dec, 1.0).unwrap();
        assert_eq!(cm, 1, "B=1 时折叠掩码恒为 1");
        let e: f32 = x.iter().map(|v| v * v).sum();
        assert!((e - 1.0).abs() < 1e-5, "形状应当是单位 L2 范数，实际 {e}");
    }
}
