//! SILK 的**整数原语**（参考实现 `silk/macros.h` + `silk/SigProc_FIX.h`）。
//!
//! SILK 解码器是纯整数实现：这些宏看着像"普通乘法"，其实每个都规定了
//! **截断 / 移位 / 取哪一半**的确切语义 —— 必须逐位照抄，不能"数学等价"地重写。
//! 和 CELT 那边的 `mathops.rs` 分工不同：那边要对齐 f32 的精度路径，
//! 这边要对齐 int32/int16 的位模式。
//!
//! 参考实现里这些宏在定点/浮点两个构建下是**同一份**（`macros.h` 不带
//! `FIXED_POINT` 条件），所以 SILK 的对拍口径是**逐位相同**（不是"差几个量化步长"）。

/// `silk_SMULWB(a32, b32)`：`a32`（32 位）乘 `b32` 的低 16 位，结果 Q16 缩放。
///
/// 注意是**两段相加**：高 16 位先乘，低 16 位乘完再右移 16 —— 不能写成
/// `(a * b) >> 16`：那样在负数上会差 1（截断方向不同）。
#[inline]
pub fn smulwb(a32: i32, b32: i32) -> i32 {
    ((a32 >> 16) * (b32 as i16 as i32)) + (((a32 & 0xFFFF) * (b32 as i16 as i32)) >> 16)
}

/// `silk_SMULBB(a32, b32)`：两个数的**低 16 位**（符号扩展后）相乘。
#[inline]
pub fn smulbb(a32: i32, b32: i32) -> i32 {
    (a32 as i16 as i32) * (b32 as i16 as i32)
}

/// `silk_SMLABB(a32, b32, c32)` = `a32 + smulbb(b32, c32)`。
#[inline]
pub fn smlabb(a32: i32, b32: i32, c32: i32) -> i32 {
    a32 + smulbb(b32, c32)
}

/// `silk_SMLABB_ovflw(a32, b32, c32)`：与 [`smlabb`] 同式，但加法允许回绕。
///
/// 参考实现那句注释是关键：LPC 分析滤波里**允许回绕**，因为"两次回绕会互相抵消"——
/// 换成饱和或检查溢出会改变非法码流下的输出（合法码流里两者等价）。
#[inline]
pub fn smlabb_ovflw(a32: i32, b32: i32, c32: i32) -> i32 {
    a32.wrapping_add(smulbb(b32, c32))
}

/// `silk_DIV32_16(a32, b16)`：整除（C 语义：向零截断；Rust 的 `/` 一致）。
#[inline]
pub fn div32_16(a32: i32, b16: i32) -> i32 {
    a32 / b16
}

/// `silk_RSHIFT_ROUND(a, shift)`：四舍五入的算术右移。
///
/// 参考实现对 `shift == 1` 单独写了 `(a>>1) + (a&1)`，其余是
/// `((a >> (shift-1)) + 1) >> 1` —— 两条都要照抄（结果不完全等价）。
#[inline]
pub fn rshift_round(a: i32, shift: u32) -> i32 {
    if shift == 1 {
        (a >> 1) + (a & 1)
    } else {
        ((a >> (shift - 1)) + 1) >> 1
    }
}

/// `silk_LSHIFT(a, shift)`（shift >= 0）：左移（C 里是 int32 左移，回绕语义）。
#[inline]
pub fn lshift(a: i32, shift: u32) -> i32 {
    ((a as u32) << shift) as i32
}

/// `silk_RSHIFT(a, shift)`（shift >= 0）：算术右移。
#[inline]
pub fn rshift(a: i32, shift: u32) -> i32 {
    a >> shift
}

/// `silk_SMULL(a32, b32)`：32×32 → 64 位（不丢位）。
#[inline]
pub fn smull(a32: i32, b32: i32) -> i64 {
    a32 as i64 * b32 as i64
}

/// `silk_SMMUL(a32, b32)`：取 64 位乘积的高 32 位（Q 格式乘法的"只留整数部分"）。
#[inline]
pub fn smmul(a32: i32, b32: i32) -> i32 {
    (smull(a32, b32) >> 32) as i32
}

/// `silk_RSHIFT_ROUND64(a, shift)`：64 位的四舍五入右移（`shift == 1` 单列一条）。
#[inline]
pub fn rshift_round64(a: i64, shift: u32) -> i64 {
    if shift == 1 {
        (a >> 1) + (a & 1)
    } else {
        ((a >> (shift - 1)) + 1) >> 1
    }
}

/// `MUL32_FRAC_Q(a32, b32, Q)`（`LPC_inv_pred_gain.c` 里的私有宏）。
#[inline]
pub fn mul32_frac_q(a32: i32, b32: i32, q: u32) -> i32 {
    rshift_round64(smull(a32, b32), q) as i32
}

/// `silk_SMULWW(a32, b32)` = `SMULWB(a,b) + a·round(b>>16)`。
#[inline]
pub fn smulww(a32: i32, b32: i32) -> i32 {
    smulwb(a32, b32).wrapping_add(a32.wrapping_mul(rshift_round(b32, 16)))
}

/// `silk_SMLAWB(a32, b32, c32)` = `a + SMULWB(b, c)`。
#[inline]
pub fn smlawb(a32: i32, b32: i32, c32: i32) -> i32 {
    a32.wrapping_add(smulwb(b32, c32))
}

/// `silk_SMLAWW(a32, b32, c32)`。
#[inline]
pub fn smlaww(a32: i32, b32: i32, c32: i32) -> i32 {
    smlawb(a32, b32, c32).wrapping_add(b32.wrapping_mul(rshift_round(c32, 16)))
}

/// `silk_ADD_LSHIFT32(a, b, shift)` = `a + (b << shift)`。
///
/// 用在 SILK 的 LPC 递归里（`sLPC_Q14[n] = exc + (LPC_pred_Q10 << 4)`），
/// 左移允许回绕（C 里就是 `int32` 左移）。
#[inline]
pub fn add_lshift32(a: i32, b: i32, shift: u32) -> i32 {
    a.wrapping_add(((b as u32) << shift) as i32)
}

/// `silk_ADD_SAT16(a, b)`：先加再饱和到 int16。
#[inline]
pub fn add_sat16(a: i16, b: i32) -> i16 {
    sat16(a as i32 + b)
}

/// `silk_abs(a)`（参考实现注明：输入是 `int32_min` 时结果不对 —— 这里用回绕取负，一样）。
#[inline]
pub fn abs(a: i32) -> i32 {
    if a > 0 {
        a
    } else {
        a.wrapping_neg()
    }
}

/// `silk_CLZ32`：前导零个数（0 给 32）。
#[inline]
pub fn clz32(x: i32) -> i32 {
    (x as u32).leading_zeros() as i32
}

/// `silk_ROR32(a32, rot)`：32 位循环右移（rot 可以是负数 = 循环左移）。
#[inline]
pub fn ror32(a32: i32, rot: i32) -> i32 {
    let x = a32 as u32;
    if rot == 0 {
        a32
    } else if rot < 0 {
        let m = (-rot) as u32;
        ((x << m) | (x >> (32 - m))) as i32
    } else {
        let r = rot as u32;
        ((x << (32 - r)) | (x >> r)) as i32
    }
}

/// `silk_LIMIT(a, limit1, limit2)`：两端钳位（l1/l2 谁大谁小都行，参考实现是这一整条三元表达式）。
#[inline]
pub fn limit(a: i32, limit1: i32, limit2: i32) -> i32 {
    if limit1 > limit2 {
        if a > limit1 {
            limit1
        } else if a < limit2 {
            limit2
        } else {
            a
        }
    } else if a > limit2 {
        limit2
    } else if a < limit1 {
        limit1
    } else {
        a
    }
}

/// `silk_LSHIFT_SAT32(a, shift)`：先钳到移位后不溢出的范围再左移。
#[inline]
pub fn lshift_sat32(a: i32, shift: u32) -> i32 {
    lshift(limit(a, i32::MIN >> shift, i32::MAX >> shift), shift)
}

/// `silk_SUB32_ovflw` / `silk_LSHIFT_ovflw`：允许回绕的加减/左移。
#[inline]
pub fn sub32_ovflw(a: i32, b: i32) -> i32 {
    a.wrapping_sub(b)
}

#[inline]
pub fn lshift_ovflw(a: i32, shift: u32) -> i32 {
    ((a as u32) << shift) as i32
}

/// `silk_CLZ_FRAC`：前导零个数 + 有效位后面的 7 位小数（Q7）。
#[inline]
pub fn clz_frac(x: i32) -> (i32, i32) {
    let lz = clz32(x);
    (lz, ror32(x, 24 - lz) & 0x7f)
}

/// `silk_SQRT_APPROX`：整数平方根近似（误差 <10%，见参考实现注释）。
#[inline]
pub fn sqrt_approx(x: i32) -> i32 {
    if x <= 0 {
        return 0;
    }
    let (lz, frac_q7) = clz_frac(x);
    let mut y = if lz & 1 != 0 { 32768 } else { 46214 }; // 46214 = sqrt(2)·32768
    y >>= rshift(lz, 1);
    smlawb(y, y, smulbb(213, frac_q7))
}

/// `silk_INVERSE32_varQ(b32, Qres)`：`(1 << Qres) / b32` 的定点近似（牛顿迭代一次）。
#[inline]
pub fn inverse32_varq(b32: i32, qres: i32) -> i32 {
    let b_headrm = clz32(abs(b32)) - 1;
    let b32_nrm = lshift(b32, b_headrm as u32);
    let b32_inv = div32_16(i32::MAX >> 2, rshift(b32_nrm, 16));
    let mut result = lshift(b32_inv, 16);
    let err_q32 = lshift((1 << 29) - smulwb(b32_nrm, b32_inv), 3);
    result = smlaww(result, err_q32, b32_inv);
    let lsh = 61 - b_headrm - qres;
    if lsh <= 0 {
        lshift_sat32(result, (-lsh) as u32)
    } else if lsh < 32 {
        rshift(result, lsh as u32)
    } else {
        0
    }
}

/// `silk_SAT16(a)`：钳到 int16 范围。
#[inline]
pub fn sat16(a: i32) -> i16 {
    if a > i16::MAX as i32 {
        i16::MAX
    } else if a < i16::MIN as i32 {
        i16::MIN
    } else {
        a as i16
    }
}

/// `silk_DIV32_varQ(a32, b32, Qres)`：`(a32 << Qres) / b32` 的定点近似
/// （先归一化、用一次牛顿迭代求倒数、再做两次乘法）。
///
/// 与 [`inverse32_varq`] 是同一套套路，区别是它直接约掉了一个归一化因子 ——
/// 所以两条都要照抄，不能互推。`b32 == 0` 在参考实现里是断言失败（调用方保证非零）。
#[inline]
pub fn div32_varq(a32: i32, b32: i32, qres: i32) -> i32 {
    debug_assert!(b32 != 0);
    debug_assert!(qres >= 0);
    let a_headrm = clz32(abs(a32)) - 1;
    let mut a32_nrm = lshift(a32, a_headrm as u32);
    let b_headrm = clz32(abs(b32)) - 1;
    let b32_nrm = lshift(b32, b_headrm as u32);
    let b32_inv = div32_16(i32::MAX >> 2, rshift(b32_nrm, 16));
    let mut result = smulwb(a32_nrm, b32_inv);
    // 参考实现注明这里的减法**允许回绕**（a32_nrm 的最终值一定很小）
    a32_nrm = sub32_ovflw(a32_nrm, lshift_ovflw(smmul(b32_nrm, result), 3));
    result = smlawb(result, a32_nrm, b32_inv);
    let lsh = 29 + a_headrm - b_headrm - qres;
    if lsh < 0 {
        lshift_sat32(result, (-lsh) as u32)
    } else if lsh < 32 {
        rshift(result, lsh as u32)
    } else {
        0
    }
}

/// `silk_insertion_sort_increasing_all_values_int16`：插入排序（NLSF 稳定化的兜底路径）。
pub fn insertion_sort_increasing_int16(a: &mut [i16]) {
    for i in 1..a.len() {
        let value = a[i];
        let mut j = i as isize - 1;
        while j >= 0 && value < a[j as usize] {
            a[(j + 1) as usize] = a[j as usize];
            j -= 1;
        }
        a[(j + 1) as usize] = value;
    }
}

/// `silk_min` / `silk_max`（参考实现是宏，行为就是三元表达式）。
#[inline]
pub fn min(a: i32, b: i32) -> i32 {
    if a < b {
        a
    } else {
        b
    }
}

#[inline]
pub fn max(a: i32, b: i32) -> i32 {
    if a > b {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smulwb_splits_the_high_and_low_halves() {
        // 高半贡献：0x0001_0000 的高 16 位是 1，乘 int16 的 0x8000(= -32768)
        assert_eq!(smulwb(0x0001_0000, 0x8000), -32768);
        // 低半贡献：0x0000_8000 的高半是 0，低半 (32768 * 16384) >> 16 = 8192
        assert_eq!(smulwb(0x0000_8000, 0x4000), 8192);
        // 负的 a：高半 -1、低半 >>16 是 0
        assert_eq!(smulwb(-1, 1), -1);
        // 为什么不能写成 (a * b) >> 16：下面这组里 a*b ≈ 7.04e13 会撑爆 int32
        // （回绕后成负数）；两段写法各乘各的、再加起来，不溢出。
        // 高半 32767×32767，低半 (65535×32767)>>16 = 32766。
        assert_eq!(smulwb(0x7FFF_FFFF, 0x7FFF), 32767 * 32767 + 32766);
    }

    #[test]
    fn smulbb_and_smlabb_use_the_low_16_bits() {
        assert_eq!(smulbb(2, 3), 6);
        assert_eq!(smulbb(0x0001_0002, 3), 6, "高 16 位要被丢掉");
        assert_eq!(smulbb(-1, 1), -1);
        assert_eq!(smlabb(100, -2, 3), 94);
    }

    #[test]
    fn rshift_round_matches_the_two_reference_branches() {
        assert_eq!(rshift_round(3, 1), 2);
        assert_eq!(rshift_round(-3, 1), -1);
        assert_eq!(rshift_round(5, 2), 1);
        assert_eq!(rshift_round(-5, 2), -1);
        assert_eq!(rshift_round(6, 2), 2);
    }
}
