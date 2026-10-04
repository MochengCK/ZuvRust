//! AV1 的 **warp**（仿射运动补偿）：规范 §7.11.3.5 – §7.11.3.8。
//!
//! ```text
//! 找候选          §7.10.4      邻居块的运动矢量（`find_warp_samples`，在 mv.rs）
//! 最小二乘拟合    §7.11.3.8    → LocalWarpParams + LocalValid
//! 分解成两次剪切  §7.11.3.6    → alpha / beta / gamma / delta + warpValid
//! 近似除法        §7.11.3.7    → divShift / divFactor（`Div_Lut`）
//! 逐 8×8 块滤波   §7.11.3.5    `Warped_Filters`（193×8）
//! ```
//!
//! ## 为什么 `warpValid` 里是 `4*|alpha| + 7*|beta|` 和 `4*|gamma| + 4*|delta|`
//!
//! 这两个系数**正好等于第二段滤波的最大索引幅度**（`i2 ∈ [-4,4)`、`i1 ∈ [-7,8)`）。
//! `offs = Round2(sx, 10) + 64` 必须落在 `0..=192`：
//! `sx = sx4 + alpha*i2 + beta*i1`，`sx4 ∈ [0, 65536)`，只要
//! `|alpha*i2 + beta*i1| < 65536` 就一定不越界 —— 而 `4|alpha| + 7|beta| < 65536`
//! 正是它的上界。所以那张表的 193 行**不是富余**，是刚好够。
//!
//! ## 三个容易搞错的地方
//!
//! 1. `Mv` 是 `[行, 列]`：`mvy = Mv[0][0]`、`mvx = Mv[0][1]`。
//! 2. `Block warp` 里的 `x`/`y` 是**平面**坐标，但 `srcX/srcY` 要 `<< subX/subY`
//!    折回亮度 —— 所以色度也能直接用亮度那套 `LocalWarpParams`。
//! 3. `useWarp` 的 `w < 8 || h < 8` 判的是**当前平面**的尺寸：4:2:0 下 8×8 的块
//!    色度只有 4×4 ⇒ 色度**不 warp**，走普通缩放帧间预测。

use super::flags;
use me_core::{MediaError, Result};

use super::block::{NUM_4X4_BLOCKS_HIGH, NUM_4X4_BLOCKS_WIDE};
use super::intra::{round2_signed, Plane};
use super::mv::Mv;
use super::warp_tables::{DIV_LUT, WARPED_FILTERS};

/// `WARPEDMODEL_PREC_BITS`（§6.10.15）：仿射参数的定点小数位数。
pub const WARPEDMODEL_PREC_BITS: u32 = 16;
/// `WARPEDDIFF_PREC_BITS`：滤波相位多出的精度（`offs` 的右移量）。
pub const WARPEDDIFF_PREC_BITS: u32 = 10;
/// `WARPEDPIXEL_PREC_SHIFTS`（`1 << 6`）。
pub const WARPEDPIXEL_PREC_SHIFTS: i32 = 1 << 6;
/// `WARP_PARAM_REDUCE_BITS`：剪切参数先按它取整再左移（丢掉无意义的低位）。
pub const WARP_PARAM_REDUCE_BITS: u32 = 6;
/// `DIV_LUT_BITS`：`Div_Lut` 的下标位数。
pub const DIV_LUT_BITS: u32 = 8;
/// `DIV_LUT_PREC_BITS`：`Div_Lut` 里每个值的小数位数。
pub const DIV_LUT_PREC_BITS: u32 = 14;
/// `LS_MV_MAX`：进最小二乘之前，`|s - d|` 的上限。
pub const LS_MV_MAX: i32 = 256;
/// `WARPEDMODEL_TRANS_CLAMP`（`1 << 23`）：平移分量的钳位。
pub const WARPEDMODEL_TRANS_CLAMP: i32 = 1 << 23;
/// `WARPEDMODEL_NONDIAGAFFINE_CLAMP`（`1 << 13`）：非对角分量的钳位。
pub const WARPEDMODEL_NONDIAGAFFINE_CLAMP: i32 = 1 << 13;

/// `setup shear` 的输出（§7.11.3.6）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shear {
    /// `warpValid`
    pub valid: bool,
    pub alpha: i32,
    pub beta: i32,
    pub gamma: i32,
    pub delta: i32,
}

/// `FloorLog2( x )`（对 i64）。`x <= 0` 时返回 0（规范只用在 `Abs(d)` 上）。
fn floor_log2_i64(x: i64) -> i32 {
    if x <= 1 {
        return 0;
    }
    63 - x.leading_zeros() as i32
}

/// `Round2Signed( x, n )`（对 i64；负数往远离零的方向进位）。
fn round2_signed_i64(x: i64, n: u32) -> i64 {
    if n == 0 {
        return x;
    }
    let r = 1i64 << (n - 1);
    if x >= 0 {
        (x + r) >> n
    } else {
        -((-x + r) >> n)
    }
}

/// `Resolve divisor process`（§7.11.3.7）：返回 `(divShift, divFactor)`。
///
/// `d` 用 `i64`：`warp estimation` 传进来的是**行列式**，可能远超 `i32`
/// （libaom 那边也是 64 位版本）。
pub fn resolve_divisor(d: i64) -> (u32, i64) {
    let ad = d.abs();
    let n = floor_log2_i64(ad);
    let e = ad - (1i64 << n);
    let f = if n > DIV_LUT_BITS as i32 {
        round2_signed_i64(e, (n - DIV_LUT_BITS as i32) as u32)
    } else {
        e << (DIV_LUT_BITS as i32 - n)
    };
    let div_shift = (n + DIV_LUT_PREC_BITS as i32) as u32;
    // `f` 落在 `0..=256`：`DIV_LUT_NUM = 257` 就是为了容下 `256`。
    let idx = f.clamp(0, DIV_LUT.len() as i64 - 1) as usize;
    let div_factor = if d < 0 {
        -i64::from(DIV_LUT[idx])
    } else {
        i64::from(DIV_LUT[idx])
    };
    (div_shift, div_factor)
}

/// `Setup shear process`（§7.11.3.6）：把仿射矩阵拆成两次一维剪切。
///
/// **中间乘积必须走 `i128`**：`v = warpParams[4] << 16` 已经在 2^32 量级，
/// 再乘 `divFactor`（≤ 2^14）就爆 `i32` 了（libaom 那边用 64 位）。
pub fn setup_shear(params: &[i32; 6]) -> Shear {
    let clip16 = |v: i64| v.clamp(-32768, 32767) as i32;
    let alpha0 = clip16(i64::from(params[2]) - (1i64 << WARPEDMODEL_PREC_BITS));
    let beta0 = clip16(i64::from(params[3]));

    let (div_shift, div_factor) = resolve_divisor(i64::from(params[2]));
    let mul = |v: i128| round2_signed_i128(v, div_shift) as i64;

    let v = i128::from(params[4]) << WARPEDMODEL_PREC_BITS;
    let gamma0 = clip16(mul(v * i128::from(div_factor)));

    let w = i128::from(params[3]) * i128::from(params[4]);
    let delta0 = clip16(
        i64::from(params[5]) - mul(w * i128::from(div_factor)) - (1i64 << WARPEDMODEL_PREC_BITS),
    );

    let reduce = |x: i32| round2_signed(x, WARP_PARAM_REDUCE_BITS) << WARP_PARAM_REDUCE_BITS;
    let (alpha, beta, gamma, delta) = (reduce(alpha0), reduce(beta0), reduce(gamma0), reduce(delta0));

    let prec = 1i32 << WARPEDMODEL_PREC_BITS;
    let valid = 4 * alpha.abs() + 7 * beta.abs() < prec && 4 * gamma.abs() + 4 * delta.abs() < prec;
    Shear { valid, alpha, beta, gamma, delta }
}

/// 对 `i128` 的 `Round2Signed`。
fn round2_signed_i128(x: i128, n: u32) -> i128 {
    if n == 0 {
        return x;
    }
    let r = 1i128 << (n - 1);
    if x >= 0 {
        (x + r) >> n
    } else {
        -((-x + r) >> n)
    }
}

/// `ls_product( a, b )`（§7.11.3.8）。
#[inline]
fn ls_product(a: i64, b: i64) -> i64 {
    ((a * b) >> 2) + (a + b)
}

/// `Warp estimation process`（§7.11.3.8）：由邻居候选做最小二乘拟合。
///
/// `cand[i] = [srcY(1/8), srcX(1/8), dstY(1/8), dstX(1/8)]`，都来自
/// `find_warp_samples`。返回 `(LocalValid, LocalWarpParams)`；
/// `LocalValid == 0` 时参数无意义（调用方不该用它）。
pub fn warp_estimate(
    mv: Mv,
    mi_row: i32,
    mi_col: i32,
    bsize: u8,
    num_samples: usize,
    cand: &[[i32; 4]],
) -> (bool, [i32; 6]) {
    let mut a = [[0i64; 2]; 2];
    let mut bx = [0i64; 2];
    let mut by = [0i64; 2];

    let w4 = i64::from(NUM_4X4_BLOCKS_WIDE[bsize as usize]);
    let h4 = i64::from(NUM_4X4_BLOCKS_HIGH[bsize as usize]);
    let mid_y = i64::from(mi_row) * 4 + h4 * 2 - 1;
    let mid_x = i64::from(mi_col) * 4 + w4 * 2 - 1;
    let suy = mid_y * 8;
    let sux = mid_x * 8;
    let duy = suy + i64::from(mv[0]);
    let dux = sux + i64::from(mv[1]);

    for c in cand.iter().take(num_samples) {
        let sy = i64::from(c[0]) - suy;
        let sx = i64::from(c[1]) - sux;
        let dy = i64::from(c[2]) - duy;
        let dx = i64::from(c[3]) - dux;
        if (sx - dx).abs() < i64::from(LS_MV_MAX) && (sy - dy).abs() < i64::from(LS_MV_MAX) {
            a[0][0] += ls_product(sx, sx) + 8;
            a[0][1] += ls_product(sx, sy) + 4;
            a[1][1] += ls_product(sy, sy) + 8;
            bx[0] += ls_product(sx, dx) + 8;
            bx[1] += ls_product(sy, dx) + 4;
            by[0] += ls_product(sx, dy) + 4;
            by[1] += ls_product(sy, dy) + 8;
        }
    }

    let det = a[0][0] * a[1][1] - a[0][1] * a[0][1];
    if det == 0 {
        return (false, [0; 6]);
    }

    let (div_shift, div_factor) = resolve_divisor(det);
    let shift = div_shift as i32 - WARPEDMODEL_PREC_BITS as i32;
    let (div_shift, div_factor) = if shift < 0 {
        (0u32, div_factor << (-shift))
    } else {
        (shift as u32, div_factor)
    };
    let apply = |v: i128| round2_signed_i128(v * i128::from(div_factor), div_shift) as i64;

    let prec = 1i64 << WARPEDMODEL_PREC_BITS;
    let diag = |v: i128| {
        apply(v).clamp(
            prec - i64::from(WARPEDMODEL_NONDIAGAFFINE_CLAMP) + 1,
            prec + i64::from(WARPEDMODEL_NONDIAGAFFINE_CLAMP) - 1,
        ) as i32
    };
    let nondiag = |v: i128| {
        apply(v).clamp(
            -i64::from(WARPEDMODEL_NONDIAGAFFINE_CLAMP) + 1,
            i64::from(WARPEDMODEL_NONDIAGAFFINE_CLAMP) - 1,
        ) as i32
    };

    let mut p = [0i32; 6];
    p[2] = diag(i128::from(a[1][1] * bx[0] - a[0][1] * bx[1]));
    p[3] = nondiag(i128::from(-a[0][1] * bx[0] + a[0][0] * bx[1]));
    p[4] = nondiag(i128::from(a[1][1] * by[0] - a[0][1] * by[1]));
    p[5] = diag(i128::from(-a[0][1] * by[0] + a[0][0] * by[1]));

    let mvy = i64::from(mv[0]);
    let mvx = i64::from(mv[1]);
    let vx = mvx * (1i64 << (WARPEDMODEL_PREC_BITS - 3))
        - (mid_x * (i64::from(p[2]) - prec) + mid_y * i64::from(p[3]));
    let vy = mvy * (1i64 << (WARPEDMODEL_PREC_BITS - 3))
        - (mid_x * i64::from(p[4]) + mid_y * (i64::from(p[5]) - prec));
    let tc = i64::from(WARPEDMODEL_TRANS_CLAMP);
    p[0] = vx.clamp(-tc, tc - 1) as i32;
    p[1] = vy.clamp(-tc, tc - 1) as i32;
    (true, p)
}

/// `Block warp process`（§7.11.3.5）：按 `warpParams` 把参考帧 8×8 一块一块地投过来。
///
/// 返回 `w×h` 的预测值，**精度与 `block_inter_prediction` 一致**（还要再按
/// `InterPostRound` 收尾），所以复合/非复合两条路都能直接用它。
#[allow(clippy::too_many_arguments)]
pub fn block_warp(
    dst: &mut [i32],
    ref_plane: &Plane,
    sub_x: u8,
    sub_y: u8,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    params: &[i32; 6],
    inter_round0: u32,
    inter_round1: u32,
) -> Result<()> {
    debug_assert_eq!(dst.len(), (w * h) as usize, "warp 的输出缓冲长度不对");
    let shear = setup_shear(params);
    debug_assert!(shear.valid, "block_warp 只应当在 warpValid 时被调用");
    let (alpha, beta, gamma, delta) = (shear.alpha, shear.beta, shear.gamma, shear.delta);
    if flags::trace_warp() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        eprintln!(
            "WARPTRACE#{n} x={x} y={y} w={w} h={h} sub={sub_x},{sub_y} \
             mat={},{},{},{},{},{} abgd={alpha},{beta},{gamma},{delta}",
            params[0], params[1], params[2], params[3], params[4], params[5]
        );
    }

    let last_x = ref_plane.w as i32 - 1;
    let last_y = ref_plane.h as i32 - 1;
    let prec_mask = (1i64 << WARPEDMODEL_PREC_BITS) - 1;
    // 规范正文（§7.11.3.5「Block warp process」）对 warp 的每一处取整都用
    // **`Round2`**（§4.7：加半再算术右移 —— 负数向下取整），`offs` 的滤波相位与
    // 两趟滤波和都是如此。dav1d（`warp_affine_8x8_c`：`(tmx + 512) >> 10` +
    // `FILTER_WARP_RND`）与之一致；libaom 的 C 源码多一步 `sx4 &= ~63`（见
    // `AV1_WARP_REFFMT`）。**默认就走规范/dav1d 这一套**。
    //
    // ⚠️ 2026-10-03 之前这里默认走的是 `Round2Signed`（远离零取整），依据是
    // 2026-10-01 的一次对拍（"46 vs 575"）—— 那次把"`Round2` 与 `Round2Signed`"
    // 和"抹不抹低位"两件事混在一起比了：真正明显变差的是 libaom C 源码那一步
    // `& ~63`，而不是 `Round2` 本身。改成规范语义后真素材 20 帧的亮度残差
    // 从 39 点降到 0（色度早在同一轮由 CFL/MaxLuma 修复清零）。
    let ref_fmt = flags::warp_reffmt();
    let signed = flags::warp_signed();
    let reduce_mask = !((1i64 << WARP_PARAM_REDUCE_BITS) - 1);
    let r2 = |x: i64, n: u32| -> i64 {
        if signed {
            round2_signed_i64(x, n)
        } else {
            round2_floor_i64(x, n)
        }
    };

    for i8 in 0..((h - 1) >> 3) + 1 {
        for j8 in 0..((w - 1) >> 3) + 1 {
            // `srcX/srcY` 是**亮度**坐标（平面坐标 << sub）
            let src_x = i64::from((x + j8 * 8 + 4) << sub_x);
            let src_y = i64::from((y + i8 * 8 + 4) << sub_y);
            let dst_x = i64::from(params[2]) * src_x + i64::from(params[3]) * src_y + i64::from(params[0]);
            let dst_y = i64::from(params[4]) * src_x + i64::from(params[5]) * src_y + i64::from(params[1]);

            let x4 = dst_x >> sub_x;
            let y4 = dst_y >> sub_y;
            let ix4 = x4 >> WARPEDMODEL_PREC_BITS;
            let iy4 = y4 >> WARPEDMODEL_PREC_BITS;
            let sx4 = if ref_fmt {
                (x4 & prec_mask) & reduce_mask
            } else {
                x4 & prec_mask
            };
            let sy4 = if ref_fmt {
                (y4 & prec_mask) & reduce_mask
            } else {
                y4 & prec_mask
            };

            // ① 横向：15×8 的中间结果
            //
            // **内点快路径**（2026-10-04）：取样窗口是 `[ix4-7, ix4+7] × [iy4-7, iy4+7]`，
            // 整段落在参考平面内时钳位全是恒等 ⇒ 省掉**每抽头一次 `clamp` + 一次带断言的
            // `get`**（原实现一个 8×8 块要付 7×120 次）。行指针也提到 `i2` 循环外
            // （它只随 `i1` 变）。值逐位相同：内点里 `clamp` 与 `get` 都是恒等映射。
            let mut intermediate = [[0i32; 8]; 15];
            let interior = ix4 - 7 >= 0
                && ix4 + 7 <= i64::from(last_x)
                && iy4 - 7 >= 0
                && iy4 + 7 <= i64::from(last_y);
            if interior {
                let data = ref_plane.px16();
                let stride = ref_plane.stride;
                let base_row = (iy4 - 7) as usize * stride;
                let base_col = (ix4 - 7) as usize;
                for i1 in -7i64..8 {
                    let row = &data[base_row + (i1 + 7) as usize * stride + base_col..];
                    for i2 in -4i64..4 {
                        let sx = sx4 + i64::from(alpha) * i2 + i64::from(beta) * i1;
                        let offs =
                            r2(sx, WARPEDDIFF_PREC_BITS) + i64::from(WARPEDPIXEL_PREC_SHIFTS);
                        debug_assert!((0..WARPED_FILTERS.len() as i64).contains(&offs));
                        let f = &WARPED_FILTERS[offs as usize];
                        let start = (i2 + 4) as usize;
                        let mut s = 0i64;
                        for (i3, tap) in f.iter().enumerate() {
                            s += i64::from(*tap) * i64::from(row[start + i3]);
                        }
                        intermediate[(i1 + 7) as usize][(i2 + 4) as usize] =
                            round2_i64_to_i32_with(&r2, s, inter_round0);
                    }
                }
            } else {
                for i1 in -7i64..8 {
                    let row = (iy4 + i1).clamp(0, i64::from(last_y)) as usize;
                    for i2 in -4i64..4 {
                        let sx = sx4 + i64::from(alpha) * i2 + i64::from(beta) * i1;
                        let offs =
                            r2(sx, WARPEDDIFF_PREC_BITS) + i64::from(WARPEDPIXEL_PREC_SHIFTS);
                        let f = filter_row(offs, sx, i1, i2)?;
                        let mut s = 0i64;
                        for (i3, tap) in f.iter().enumerate() {
                            let col =
                                (ix4 + i2 - 3 + i3 as i64).clamp(0, i64::from(last_x)) as usize;
                            s += i64::from(*tap) * i64::from(ref_plane.get(col, row));
                        }
                        intermediate[(i1 + 7) as usize][(i2 + 4) as usize] =
                            round2_i64_to_i32_with(&r2, s, inter_round0);
                    }
                }
            }

            // ② 纵向：只填这一块 8×8（到边界就裁）
            let i1_end = 4i64.min(i64::from(h) - i64::from(i8) * 8 - 4);
            let i2_end = 4i64.min(i64::from(w) - i64::from(j8) * 8 - 4);
            for i1 in -4i64..i1_end {
                for i2 in -4i64..i2_end {
                    let sy = sy4 + i64::from(gamma) * i2 + i64::from(delta) * i1;
                    let offs = r2(sy, WARPEDDIFF_PREC_BITS) + i64::from(WARPEDPIXEL_PREC_SHIFTS);
                    let f = filter_row(offs, sy, i1, i2)?;
                    let mut s = 0i64;
                    for (i3, tap) in f.iter().enumerate() {
                        s += i64::from(*tap)
                            * i64::from(intermediate[(i1 + i3 as i64 + 4) as usize][(i2 + 4) as usize]);
                    }
                    let ry = i8 * 8 + i1 as i32 + 4;
                    let rx = j8 * 8 + i2 as i32 + 4;
                    dst[(ry * w + rx) as usize] = round2_i64_to_i32_with(&r2, s, inter_round1);
                }
            }
        }
    }
    Ok(())
}

/// `offs` 只可能落在 `0..=192`（靠 `warpValid` 的那两个不等式保证）。
/// 越界就说明参数没先过 `setup_shear` 的校验 —— **明确报错、不静默钳**。
#[inline]
fn filter_row(offs: i64, s: i64, i1: i64, i2: i64) -> Result<&'static [i32; 8]> {
    let idx = usize::try_from(offs).ok().filter(|v| *v < WARPED_FILTERS.len());
    match idx {
        Some(i) => Ok(&WARPED_FILTERS[i]),
        None => Err(MediaError::Bitstream(format!(
            "AV1 的 warp 滤波器下标越界：offs = {offs}（s = {s}, i1 = {i1}, i2 = {i2}）"
        ))),
    }
}

/// `Round2( x, n )`（规范 §4）：`(x + (1 << (n-1))) >> n` —— 负数时**向下取整**
/// （与 libaom 的 `ROUND_POWER_OF_TWO` 一致）。与 `Round2Signed` 只在恰好半格时不同。
#[inline]
fn round2_floor_i64(x: i64, n: u32) -> i64 {
    if n == 0 {
        return x;
    }
    (x + (1 << (n - 1))) >> n
}

/// 按给定取整函数把 i64 收成 i32（`warp` 的两趟滤波用）。
#[inline]
fn round2_i64_to_i32_with(r2: &impl Fn(i64, u32) -> i64, x: i64, n: u32) -> i32 {
    r2(x, n).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_divisor_matches_the_spec_worked_example() {
        // d = 0 不是合法输入，但也不能崩
        let (s, f) = resolve_divisor(0);
        assert_eq!(s, DIV_LUT_PREC_BITS);
        assert_eq!(f, i64::from(DIV_LUT[0]));
        // d = 2^14 时 e = 0 ⇒ f = 0 ⇒ Div_Lut[0] = 2^14
        let (s, f) = resolve_divisor(1 << 14);
        assert_eq!(s, 14 + DIV_LUT_PREC_BITS);
        assert_eq!(f, 1 << DIV_LUT_PREC_BITS);
        // 负数取负
        let (_, fneg) = resolve_divisor(-(1 << 14));
        assert_eq!(fneg, -(1i64 << DIV_LUT_PREC_BITS));
    }

    #[test]
    fn resolve_divisor_is_an_approximate_reciprocal() {
        // **这是近似除法**：`Div_Lut` 只有 `DIV_LUT_BITS = 8` 位小数，相对误差到
        // 百分之几是正常的（规范原文就不保证精确）。`d = 7` 这种尾数偏大的输入
        // 实测就有 ~5.2% 的偏差，所以容差给到 8%，别收紧。
        for d in [1i64, 3, 7, 255, 256, 1000, 65535, 1 << 20, -999] {
            let (shift, factor) = resolve_divisor(d);
            let approx = factor as f64 / (1u64 << shift) as f64;
            let exact = 1.0 / d as f64;
            // 注意 `exact.abs()`：`d < 0` 时容差上限必须是正数
            let tol = exact.abs() * 0.08;
            assert!(
                (approx - exact).abs() <= tol,
                "d = {d}：{approx} 与 {exact} 相差超过 8%"
            );
        }
    }

    #[test]
    fn identity_matrix_is_a_valid_shear() {
        let mut p = [0i32; 6];
        p[2] = 1 << WARPEDMODEL_PREC_BITS;
        p[5] = 1 << WARPEDMODEL_PREC_BITS;
        let s = setup_shear(&p);
        assert!(s.valid);
        assert_eq!((s.alpha, s.beta, s.gamma, s.delta), (0, 0, 0, 0));
    }

    #[test]
    fn too_strong_shear_is_invalid() {
        // alpha 顶到 32767：4*|alpha| 已经超过 2^16 ⇒ warpValid = 0
        let p = [0, 0, (1 << WARPEDMODEL_PREC_BITS) + 32767, 0, 0, 1 << WARPEDMODEL_PREC_BITS];
        assert!(!setup_shear(&p).valid);
    }

    /// 最小二乘在"所有候选都给同一个平移"时必须退回纯平移：
    /// 对角系数是 1.0（`1 << 16`）、非对角是 0。
    ///
    /// **不能断言"恰好等于 `1 << 16`"**：`resolve_divisor` 是近似除法，
    /// 在 2^16 的尺度上偏 ±2 是正常的（6 位 `WARP_PARAM_REDUCE_BITS` 会把它抹掉）。
    #[test]
    fn warp_estimate_recovers_a_pure_translation() {
        let bsize = super::super::partition::BLOCK_16X16 as u8;
        let mv: Mv = [32, -48];
        let (mi_row, mi_col) = (4i32, 4i32);
        let w4 = NUM_4X4_BLOCKS_WIDE[bsize as usize] as i32;
        let h4 = NUM_4X4_BLOCKS_HIGH[bsize as usize] as i32;
        let mid_y = mi_row * 4 + h4 * 2 - 1;
        let mid_x = mi_col * 4 + w4 * 2 - 1;
        // 每个候选都是"同一个块中心 + 同一个 MV"
        let cand: Vec<[i32; 4]> = (0..4)
            .map(|_| [mid_y * 8, mid_x * 8, mid_y * 8 + mv[0], mid_x * 8 + mv[1]])
            .collect();
        let (valid, p) = warp_estimate(mv, mi_row, mi_col, bsize, cand.len(), &cand);
        assert!(valid, "纯平移必须拟合成功");
        let one = 1 << WARPEDMODEL_PREC_BITS;
        assert!((p[2] - one).abs() <= 8, "对角应当接近 1.0，拿到 {}", p[2]);
        assert!((p[5] - one).abs() <= 8, "对角应当接近 1.0，拿到 {}", p[5]);
        assert_eq!(p[3], 0, "非对角必须是 0");
        assert_eq!(p[4], 0, "非对角必须是 0");
        // 平移：`dst = src + mv`。`mv` 本身是 1/8 样本单位，而
        // `LocalWarpParams[0]` 是 `WARPEDMODEL_PREC_BITS` 精度，所以要左移 13 位。
        //
        // 容差要按 `mid_x` 放：`vx` 里带了 `mid_x * (p[2] - 2^16)` 这一项，
        // 而 `p[2]` 的 ±2 误差会被 `mid_x`（这里 23）放大成 ±46。
        let want = i64::from(mv[1]) << (WARPEDMODEL_PREC_BITS - 3);
        let tol = 8 * i64::from(mid_x.abs().max(mid_y.abs()));
        assert!(
            (i64::from(p[0]) - want).abs() <= tol,
            "水平平移应当是 {want}，拿到 {}（容差 {tol}）",
            p[0]
        );
        let s = setup_shear(&p);
        assert!(s.valid, "纯平移的剪切参数必须合法");
        assert_eq!((s.alpha, s.beta, s.gamma, s.delta), (0, 0, 0, 0));
    }

    /// 退化输入必须报 `LocalValid = 0`，绝不能让后面的除法炸掉。
    ///
    /// 注意**不能用"所有候选同一点"来构造退化**：`A` 的三个累加里都带 `+4`/`+8`
    /// 的常数项，同一点也算不出 `det == 0`。真正会退化的两种情形是"没有候选"
    /// 与"候选全被 `LS_MV_MAX` 滤掉"（`A` 留在全 0）。
    #[test]
    fn warp_estimate_rejects_degenerate_samples() {
        let bsize = super::super::partition::BLOCK_16X16 as u8;
        let mv: Mv = [0, 0];
        // ① 一个候选都没有：A 全 0 ⇒ det = 0
        let (valid, _) = warp_estimate(mv, 0, 0, bsize, 0, &[[0; 4]; 8]);
        assert!(!valid, "没有候选必须报 invalid");

        // ② 候选的落点离源点太远（`|sy - dy| >= LS_MV_MAX`）⇒ 全部被滤掉
        let cand = [[0i32, 0, 100_000, 0]; 4];
        let (valid, _) = warp_estimate(mv, 0, 0, bsize, 4, &cand);
        assert!(!valid, "候选全被滤掉必须报 invalid");
    }
}
