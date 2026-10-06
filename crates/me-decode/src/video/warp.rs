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
///
/// ## 两条路径（2026-10-06，第四十刀）
///
/// * [`block_warp_x86`]：AVX2 版（`vpmaddwd`，一次算 8 个输出）；
/// * [`block_warp_scalar`]：逐样本标量版 —— **既是回落路径，也是判据的参照系**
///   （`warp_simd_matches_scalar_bit_for_bit` 拿它逐值对拍）。
///
/// 走 SIMD 的条件（任何一条不满足就回落，结果**逐位相同**）：
/// `ME_NO_SIMD` 未开、AVX2 可用、平面是 `u16` 存储、`bit_depth <= 10`
/// （12bit 的中间量会超出 `i16`）、`InterRound0 >= 3`、且两个诊断开关
/// （`AV1_WARP_REFFMT` / `AV1_WARP_SIGNED`）都关着（它们是**改语义**的旋钮，
/// SIMD 只实现规范默认那一套）。
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
    #[cfg(target_arch = "x86_64")]
    {
        if !flags::no_simd()
            && !ref_plane.is_u8()
            && ref_plane.bit_depth <= 10
            && inter_round0 >= 3
            && !flags::warp_reffmt()
            && !flags::warp_signed()
            && super::simd::avx2_available()
        {
            let shear = setup_shear(params);
            debug_assert!(shear.valid, "block_warp 只应当在 warpValid 时被调用");
            // SAFETY: 已判 AVX2；`ref_plane.px16()` 的非 u8 已判；
            //   `shear` 已过 `warpValid` ⇒ 所有 `offs` 落在 `Warped_Filters` 内
            //   （SIMD 里另有一次整块范围校验，不成立就回落标量）。
            if unsafe {
                warp_x86::block_warp(
                    dst,
                    ref_plane.px16(),
                    ref_plane.stride,
                    ref_plane.w as i64 - 1,
                    ref_plane.h as i64 - 1,
                    x,
                    y,
                    w,
                    h,
                    sub_x,
                    sub_y,
                    params,
                    shear.alpha,
                    shear.beta,
                    shear.gamma,
                    shear.delta,
                    inter_round0,
                    inter_round1,
                )
            }? {
                return Ok(());
            }
        }
    }
    block_warp_scalar(
        dst,
        ref_plane,
        sub_x,
        sub_y,
        x,
        y,
        w,
        h,
        params,
        inter_round0,
        inter_round1,
    )
}

#[allow(clippy::too_many_arguments)]
fn block_warp_scalar(
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

// ─────────────────────────────────────────────────────────────────────────────
// §7.11.3.5 的 AVX2 实现（2026-10-06，第四十刀）
// ─────────────────────────────────────────────────────────────────────────────
//
// ## 标量版在算什么
//
// 每个 8×8 子块：横向 15 行 × 8 列、纵向 8 行 × 8 列，**每个输出各 8 个抽头**，
// 而 `Warped_Filters` 是按亚像素相位查表的 ⇒ **同一行里 8 个输出的滤波器互不
// 相同**。合计 1472 次 `imul` + 1472 次 `add`；`imul` 在这台机器上吞吐就是
// 1 条/周期 ⇒ 光乘法 ~1500 周期/块。另外每块还有 184 次 `filter_row()`
// （`try_from` + `filter` + `Result`）。
//
// ## 为什么 `vpmaddwd` 一次能吃掉 8 个抽头
//
// `vpmaddwd(a, b)` 把 16 个 `i16` 通道两两配对成 8 个 `i32`：
// `out[m] = a[2m]·b[2m] + a[2m+1]·b[2m+1]`。抽头 ∈ `[-127, 127]`、样本 ∈ `[0, 4095]`
// ⇒ 一个输出的 8 个乘积和 ≤ `4095·222 = 908,890`，`i32` 装得下。
//
// ## 关键：8 个**不同**的滤波器怎么共享一次 `pmaddwd`
//
// 输出 `j` 的源窗口是 `src[j .. j+8]` —— **相邻输出只差一个样本**。于是把两个输出
// `(2p, 2p+1)` 的 8 抽头系数放进 256 位寄存器的两个 128 位道（`set_m128i`），
// 源窗口用 `_mm256_loadu2_m128i(src + 2p + 1, src + 2p)` 一次载入两个错开一个样本
// 的窗口 —— `madd` 出的 8 个 `i32` 正好是「低道 4 个部分和 = 输出 2p、高道 4 个
// = 输出 2p+1」。4 组 `madd`（8 个输出）+ 3 次 `hadd` 把每组的 4 个部分和收成 1 个：
//
// ```text
//   t01 = hadd(m0, m1)   t23 = hadd(m2, m3)   u = hadd(t01, t23)
//   u = [ s0, s2, s4, s6,   s1, s3, s5, s7 ]     ← 偶数输出在低道、奇数在高道
// ```
//
// 取整（`Round2(·, r)`）在向量里做（`add` + `sra`），再 `packs_epi32` 收成 `i16`、
// `pshufb` 摆回 `[s0..s7]`，一次 16 字节存进中间缓冲。
//
// ## 纵向：先把中间缓冲转置
//
// 纵向第 `i1v` 个输出用 `inter[i1v + k][i2v]` —— 沿**行**方向取抽头，在行主序里跨
// `stride`。把 15×8 的中间缓冲转置成 `interT[列][行]`（两个 8×8 `i16` 转置）之后，
// 每一列的 8 个抽头就**连续**了 ⇒ 纵向原样复用横向那一套。
//
// ## 值域与前提（任何一条不满足 ⇒ 返回 `Ok(false)`，调用方回落标量）
//
// * 中间量 = `Round2(Σ c·x, r0)`，上界 `4095·222 >> r0`：**8bit 7076、10bit 28388**
//   ⇒ 装得进 `i16`；**12bit 113612 装不下**（由调用方的 `bit_depth <= 10` 挡住）。
// * 纵向的和 ≤ `222 · 28388 = 6.3e6`，`madd` 的 `i32` 结果够。
// * `offs` 的范围由「`sx`/`sy` 是线性的、极值在角点」推出 —— 越界就回落标量
//   （标量路径会给出原来那条精确的报错）。
//
// 判据：[`tests::warp_simd_matches_scalar_bit_for_bit`]。
#[cfg(target_arch = "x86_64")]
mod warp_x86 {
    use super::super::warp_tables::WARPED_FILTERS_I16;
    use super::{WARPEDMODEL_PREC_BITS, WARPEDPIXEL_PREC_SHIFTS};
    use me_core::Result;
    use std::arch::x86_64::*;

    /// `Warped_Filters` 的合法下标上界。
    const OFF_MAX: i64 = 192;

    /// 8×8 的 **`i16` 转置**（`x[i]` = 第 i 行 → 返回第 i 列）。转置自逆。
    ///
    /// 24 条 shuffle —— 与去块那次的 `dbl_transpose8` 同一套（那版已被逐位对拍验过）。
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn transpose8(x: [__m128i; 8]) -> [__m128i; 8] {
        let a0 = _mm_unpacklo_epi16(x[0], x[1]);
        let a1 = _mm_unpackhi_epi16(x[0], x[1]);
        let a2 = _mm_unpacklo_epi16(x[2], x[3]);
        let a3 = _mm_unpackhi_epi16(x[2], x[3]);
        let a4 = _mm_unpacklo_epi16(x[4], x[5]);
        let a5 = _mm_unpackhi_epi16(x[4], x[5]);
        let a6 = _mm_unpacklo_epi16(x[6], x[7]);
        let a7 = _mm_unpackhi_epi16(x[6], x[7]);
        let b0 = _mm_unpacklo_epi32(a0, a2);
        let b1 = _mm_unpackhi_epi32(a0, a2);
        let b2 = _mm_unpacklo_epi32(a1, a3);
        let b3 = _mm_unpackhi_epi32(a1, a3);
        let b4 = _mm_unpacklo_epi32(a4, a6);
        let b5 = _mm_unpackhi_epi32(a4, a6);
        let b6 = _mm_unpacklo_epi32(a5, a7);
        let b7 = _mm_unpackhi_epi32(a5, a7);
        [
            _mm_unpacklo_epi64(b0, b4),
            _mm_unpackhi_epi64(b0, b4),
            _mm_unpacklo_epi64(b1, b5),
            _mm_unpackhi_epi64(b1, b5),
            _mm_unpacklo_epi64(b2, b6),
            _mm_unpackhi_epi64(b2, b6),
            _mm_unpacklo_epi64(b3, b7),
            _mm_unpackhi_epi64(b3, b7),
        ]
    }

    /// 把 8 个滤波器（每个 8 个 `i16`）拼成 4 个 256 位寄存器：`p` 号寄存器装
    /// `[f_{2p} | f_{2p+1}]`（低道 = 偶数号输出）。
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn gather_filters(offs: &[usize; 8]) -> [__m256i; 4] {
        let f = |j: usize| -> __m128i {
            _mm_loadu_si128(WARPED_FILTERS_I16[offs[j]].as_ptr() as *const __m128i)
        };
        [
            _mm256_set_m128i(f(1), f(0)),
            _mm256_set_m128i(f(3), f(2)),
            _mm256_set_m128i(f(5), f(4)),
            _mm256_set_m128i(f(7), f(6)),
        ]
    }

    /// 4 组 `madd` + 3 次 `hadd`：`u = [s0,s2,s4,s6, s1,s3,s5,s7]`。
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dot8(c: &[__m256i; 4], a: &[__m256i; 4]) -> __m256i {
        let m0 = _mm256_madd_epi16(c[0], a[0]);
        let m1 = _mm256_madd_epi16(c[1], a[1]);
        let m2 = _mm256_madd_epi16(c[2], a[2]);
        let m3 = _mm256_madd_epi16(c[3], a[3]);
        _mm256_hadd_epi32(_mm256_hadd_epi32(m0, m1), _mm256_hadd_epi32(m2, m3))
    }

    /// `Block warp process` 的 AVX2 版。返回 `Ok(false)` = **这条流不能走 SIMD**
    /// （相位越出 `Warped_Filters`），调用方应当回落 [`super::block_warp_scalar`]。
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn block_warp(
        dst: &mut [i32],
        data: &[u16],
        stride: usize,
        last_x: i64,
        last_y: i64,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        sub_x: u8,
        sub_y: u8,
        params: &[i32; 6],
        alpha: i32,
        beta: i32,
        gamma: i32,
        delta: i32,
        inter_round0: u32,
        inter_round1: u32,
    ) -> Result<bool> {
        debug_assert!(inter_round0 >= 1 && inter_round0 <= 14);
        debug_assert!(inter_round1 >= 1 && inter_round1 <= 14);
        let prec_bits = WARPEDMODEL_PREC_BITS as i64;
        let prec_mask = (1i64 << prec_bits) - 1;
        let shifts = i64::from(WARPEDPIXEL_PREC_SHIFTS);
        let r0 = _mm_cvtsi32_si128(inter_round0 as i32);
        let r1 = _mm_cvtsi32_si128(inter_round1 as i32);
        let half0 = _mm256_set1_epi32(1i32 << (inter_round0 - 1));
        let half1 = _mm256_set1_epi32(1i32 << (inter_round1 - 1));
        // `packs_epi32` 之后 `i16` 通道顺序是 `[s0,s2,s4,s6,s1,s3,s5,s7]`，
        // 用 `pshufb` 摆成 `[s0..s7]`。
        let reorder = _mm_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);

        for i8 in 0..((h - 1) >> 3) + 1 {
            for j8 in 0..((w - 1) >> 3) + 1 {
                let src_x = i64::from((x + j8 * 8 + 4) << sub_x);
                let src_y = i64::from((y + i8 * 8 + 4) << sub_y);
                let dst_x = i64::from(params[2]) * src_x
                    + i64::from(params[3]) * src_y
                    + i64::from(params[0]);
                let dst_y = i64::from(params[4]) * src_x
                    + i64::from(params[5]) * src_y
                    + i64::from(params[1]);
                let x4 = dst_x >> sub_x;
                let y4 = dst_y >> sub_y;
                let ix4 = x4 >> prec_bits;
                let iy4 = y4 >> prec_bits;
                let sx4 = x4 & prec_mask;
                let sy4 = y4 & prec_mask;

                let i1_end = 4i64.min(i64::from(h) - i64::from(i8) * 8 - 4);
                let i2_end = 4i64.min(i64::from(w) - i64::from(j8) * 8 - 4);
                let nrow = (i1_end + 4).clamp(0, 8) as usize;
                let ncol = (i2_end + 4).clamp(0, 8) as usize;
                if nrow == 0 || ncol == 0 {
                    continue;
                }

                // ── 相位范围校验：`sx`/`sy` 对 (i1,i2) 是线性的 ⇒ 极值在角点 ──
                // 横向 `i2 ∈ [-4,3]`、`i1 ∈ [-7,7]`；纵向两者都 ∈ `[-4,3]`
                // （用满区间是**超集**，只会多回落、不会算错）。
                let ext = |s: i32, lo_coef: i64, hi_coef: i64| -> (i64, i64) {
                    let v = i64::from(s);
                    if s >= 0 {
                        (v * lo_coef, v * hi_coef)
                    } else {
                        (v * hi_coef, v * lo_coef)
                    }
                };
                let (amin, amax) = ext(alpha, -4, 3);
                let (bmin, bmax) = ext(beta, -7, 7);
                let (gmin, gmax) = ext(gamma, -4, 3);
                let (dmin, dmax) = ext(delta, -4, 3);
                let off_lo_h = ((sx4 + amin + bmin + 512) >> 10) + shifts;
                let off_hi_h = ((sx4 + amax + bmax + 512) >> 10) + shifts;
                let off_lo_v = ((sy4 + gmin + dmin + 512) >> 10) + shifts;
                let off_hi_v = ((sy4 + gmax + dmax + 512) >> 10) + shifts;
                if off_lo_h < 0 || off_hi_h > OFF_MAX || off_lo_v < 0 || off_hi_v > OFF_MAX {
                    return Ok(false);
                }

                // ── 横向：15 行 × 8 列（结果存 `i16`）────────────────────────
                // `inter[i1v]` 的第 `j` 个 `i16` = 输出列 `j` 的中间值。
                // 第 15 行只是转置时的填充（纵向最多取到下标 14）。
                let mut inter = [[0i16; 8]; 16];
                let mut win = [0u16; 16];
                let interior = ix4 - 7 >= 0
                    && ix4 + 7 <= last_x
                    && iy4 - 7 >= 0
                    && iy4 + 7 <= last_y;
                for i1v in 0..15i64 {
                    let mut offs = [0usize; 8];
                    let mut sx = sx4 + i64::from(alpha) * -4 + i64::from(beta) * (i1v - 7);
                    for o in offs.iter_mut() {
                        *o = (((sx + 512) >> 10) + shifts) as usize;
                        sx += i64::from(alpha);
                    }
                    let row: *const u16 = if interior {
                        data.as_ptr()
                            .add((iy4 - 7 + i1v) as usize * stride + (ix4 - 7) as usize)
                    } else {
                        let r = (iy4 + i1v - 7).clamp(0, last_y) as usize * stride;
                        for t in 0..16i64 {
                            let c = (ix4 - 7 + t).clamp(0, last_x) as usize;
                            win[t as usize] = *data.get_unchecked(r + c);
                        }
                        win.as_ptr()
                    };
                    let c = gather_filters(&offs);
                    let a = [
                        _mm256_loadu2_m128i(
                            row.add(1) as *const __m128i,
                            row as *const __m128i,
                        ),
                        _mm256_loadu2_m128i(
                            row.add(3) as *const __m128i,
                            row.add(2) as *const __m128i,
                        ),
                        _mm256_loadu2_m128i(
                            row.add(5) as *const __m128i,
                            row.add(4) as *const __m128i,
                        ),
                        _mm256_loadu2_m128i(
                            row.add(7) as *const __m128i,
                            row.add(6) as *const __m128i,
                        ),
                    ];
                    let v = _mm256_sra_epi32(_mm256_add_epi32(dot8(&c, &a), half0), r0);
                    let p16 = _mm_packs_epi32(
                        _mm256_castsi256_si128(v),
                        _mm256_extracti128_si256::<1>(v),
                    );
                    _mm_storeu_si128(
                        inter[i1v as usize].as_mut_ptr() as *mut __m128i,
                        _mm_shuffle_epi8(p16, reorder),
                    );
                }

                // ── 转置 15×8 → `interT[列][行]`（两个 8×8 转置）─────────────
                let mut inter_t = [[0i16; 16]; 8];
                for grp in 0..2usize {
                    let mut g = [_mm_setzero_si128(); 8];
                    for (r, slot) in g.iter_mut().enumerate() {
                        *slot = _mm_loadu_si128(
                            inter[grp * 8 + r].as_ptr() as *const __m128i
                        );
                    }
                    let t = transpose8(g);
                    for (c, col) in t.iter().enumerate() {
                        _mm_storeu_si128(
                            inter_t[c].as_mut_ptr().add(grp * 8) as *mut __m128i,
                            *col,
                        );
                    }
                }

                // ── 纵向：8 行 × 8 列 ────────────────────────────────────────
                let mut rowbuf = [0i32; 8];
                for i1v in 0..nrow {
                    let mut offs = [0usize; 8];
                    let mut sy = sy4 + i64::from(gamma) * -4 + i64::from(delta) * (i1v as i64 - 4);
                    for o in offs.iter_mut() {
                        *o = (((sy + 512) >> 10) + shifts) as usize;
                        sy += i64::from(gamma);
                    }
                    let c = gather_filters(&offs);
                    let a = [
                        _mm256_loadu2_m128i(
                            inter_t[1].as_ptr().add(i1v) as *const __m128i,
                            inter_t[0].as_ptr().add(i1v) as *const __m128i,
                        ),
                        _mm256_loadu2_m128i(
                            inter_t[3].as_ptr().add(i1v) as *const __m128i,
                            inter_t[2].as_ptr().add(i1v) as *const __m128i,
                        ),
                        _mm256_loadu2_m128i(
                            inter_t[5].as_ptr().add(i1v) as *const __m128i,
                            inter_t[4].as_ptr().add(i1v) as *const __m128i,
                        ),
                        _mm256_loadu2_m128i(
                            inter_t[7].as_ptr().add(i1v) as *const __m128i,
                            inter_t[6].as_ptr().add(i1v) as *const __m128i,
                        ),
                    ];
                    let v = _mm256_sra_epi32(_mm256_add_epi32(dot8(&c, &a), half1), r1);
                    let lo = _mm256_castsi256_si128(v);
                    let hi = _mm256_extracti128_si256::<1>(v);
                    _mm_storeu_si128(
                        rowbuf.as_mut_ptr() as *mut __m128i,
                        _mm_unpacklo_epi32(lo, hi),
                    );
                    _mm_storeu_si128(
                        rowbuf.as_mut_ptr().add(4) as *mut __m128i,
                        _mm_unpackhi_epi32(lo, hi),
                    );
                    let base = ((i8 * 8 + i1v as i32) * w + j8 * 8) as usize;
                    dst[base..base + ncol].copy_from_slice(&rowbuf[..ncol]);
                }
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AVX2 的 `block_warp` 与标量版**逐位对拍**（第四十刀）。
    ///
    /// 覆盖面：`bit_depth ∈ {8, 10}` × 平面尺寸 `{64×48, 16×16, 9×13}` × 6 组
    /// `warpParams`（恒等 / 纯平移 / 小剪切 / 对角剪切 / **大平移** —— 后者把取样
    /// 窗口推出帧外，逼出**非内点**那条逐样本钳位分支）× 3 组块位置与尺寸
    /// （含贴着右/下边界、以及 9×13 这种"`i8`/`j8` 两组都只走一半"的）× 3 档取整。
    ///
    /// 每组都断言 SIMD **真的接手了**（`Ok(true)`）—— 否则"对拍通过"可能只是
    /// 两边都没跑（回落标量时是自己跟自己比）。
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn warp_simd_matches_scalar_bit_for_bit() {
        if !super::super::simd::avx2_available() {
            return;
        }
        let param_sets: [[i32; 6]; 6] = [
            [0, 0, 1 << 16, 0, 0, 1 << 16],
            [37, -21, 1 << 16, 0, 0, 1 << 16],
            [0, 0, (1 << 16) + 8192, 0, 0, 1 << 16],
            [0, 0, (1 << 16) + 4096, 3072, -2048, 1 << 16],
            [-3_000_000, 2_000_000, 1 << 16, 0, 0, 1 << 16],
            [0, 0, 1 << 16, 0, 0, (1 << 16) + 2048],
        ];
        for &bd in &[8u8, 10] {
            let max = (1i32 << bd) - 1;
            let mut ran = 0usize;
            let mut shears_ok = 0usize;
            for &(pw, ph) in &[(64usize, 48usize), (16, 16), (9, 13)] {
                let mut plane = Plane::new(pw, ph, bd);
                if plane.is_u8() {
                    // u8 存储的迁移还没做完；SIMD 这一路只吃 u16 平面。
                    return;
                }
                {
                    let d = plane.px16_mut();
                    let mut s = 0x1234_5678u32;
                    for v in d.iter_mut() {
                        s ^= s << 13;
                        s ^= s >> 17;
                        s ^= s << 5;
                        *v = (s & max as u32) as u16;
                    }
                }
                for params in &param_sets {
                    let shear = setup_shear(params);
                    if !shear.valid {
                        continue;
                    }
                    shears_ok += 1;
                    for &(bx, by, bw, bh) in &[
                        (0i32, 0i32, 16i32, 16i32),
                        (16, 8, 32, 24),
                        (pw as i32 - 9, ph as i32 - 13, 9, 13),
                    ] {
                        if bw <= 0 || bh <= 0 || bx + bw > pw as i32 || by + bh > ph as i32 {
                            continue;
                        }
                        for &(r0, r1) in &[(3u32, 11u32), (3, 7), (5, 9)] {
                            let n = (bw * bh) as usize;
                            let mut a = vec![0i32; n];
                            let mut b = vec![0i32; n];
                            block_warp_scalar(&mut a, &plane, 0, 0, bx, by, bw, bh, params, r0, r1)
                                .unwrap();
                            // SAFETY: 已判 AVX2；参数与标量路径同一份（`shear` 已过
                            //   `warpValid`），相位范围校验不通过时函数会返回 `Ok(false)`
                            //   而不是算错。
                            let ok = unsafe {
                                warp_x86::block_warp(
                                    &mut b,
                                    plane.px16(),
                                    plane.stride,
                                    pw as i64 - 1,
                                    ph as i64 - 1,
                                    bx,
                                    by,
                                    bw,
                                    bh,
                                    0,
                                    0,
                                    params,
                                    shear.alpha,
                                    shear.beta,
                                    shear.gamma,
                                    shear.delta,
                                    r0,
                                    r1,
                                )
                            }
                            .unwrap();
                            assert!(ok, "SIMD 回落了（相位越界）：{params:?}");
                            ran += 1;
                            assert_eq!(
                                a, b,
                                "bd={bd} plane={pw}x{ph} params={params:?} \
                                 block=({bx},{by},{bw},{bh}) r=({r0},{r1})"
                            );
                        }
                    }
                }
            }
            // ⚠️ 防止"空跑"：`shear.valid` 全为假或块全被跳过时，上面的断言一次都
            // 不会执行、测试会**假绿**。这里把规模钉死。
            assert!(shears_ok >= 3, "参数集几乎都没过 `warpValid`：{shears_ok}");
            assert!(ran >= 40, "对拍组合太少（{ran}），测试可能在空跑");
        }
    }

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
