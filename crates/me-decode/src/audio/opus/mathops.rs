//! 参考实现 `mathops.h` 里那几个**浮点构建下**的数学助手。
//!
//! 单独一圈出来是因为它们**看着像 math 标准库、其实精度路径不一样**：
//!
//! | 参考实现 | 实际做的事 |
//! |---|---|
//! | `celt_sqrt(x)` | `(float)sqrt((double)x)` —— 先在 **double** 里开方再压回 float |
//! | `celt_rsqrt(x)` | `1.f / celt_sqrt(x)` |
//! | `celt_cos_norm(x)` | `(float)cos((.5f*PI)*(x))` —— 参数在 float 里算、`cos` 在 double 里算 |
//! | `celt_exp2(x)` | `(float)exp(0.6931471805599453094*(x))` —— 常数是 **double**，整条在 double 里 |
//!
//! 这不是吹毛求疵：`exp`/`cos` 只用 float 版本会在最后一位上差一点，而 `anti_collapse`
//! 填的噪声幅度、扩散的旋转角都是从这里出来的 —— 对拍时那一两位会顺着
//! "renormalise 之后还要逐样本比"放大成看得见的差。**照 C 的精度路径写**最省事。
//!
//! 下面那几个常数**故意照抄 C 的字面量**（`3.141592653`、`0.6931471805599453094`），
//! 所以整个模块放行 `clippy::approx_constant` —— 换成 `std::f32::consts::PI`
//! 虽然"更正确"，但就不是参考实现里那个只有 10 位有效数字的数了。
#![allow(clippy::approx_constant)]

/// 参考实现的 `PI`（float 字面量，只有 10 位有效数字 —— 别换成 `std::f32::consts::PI`）。
pub const PI_F32: f32 = 3.141592653;

/// 参考实现 `compute_twiddles`（FFT 旋转因子）里的 `pi`：**double** 字面量，
/// 精度和上面那个 float 的不是一回事，别混用。
pub const PI_F64: f64 = 3.14159265358979323846264338327;

/// `(float)sqrt((double)x)`。double 开方再压回 float 与"正确舍入的 float 开方"等价，
/// 这里写成 double 只是为了让意图和参考实现一致。
pub fn celt_sqrt(x: f32) -> f32 {
    (x as f64).sqrt() as f32
}

/// `1.f/celt_sqrt(x)`（参考实现 `celt_rsqrt`）。
pub fn celt_rsqrt(x: f32) -> f32 {
    1.0 / celt_sqrt(x)
}

/// `(float)cos((.5f*PI)*x)`（参考实现 `celt_cos_norm`）—— 注意实参是在 **float** 里乘出来的。
pub fn celt_cos_norm(x: f32) -> f32 {
    ((0.5f32 * PI_F32) * x).cos() as f32
}

/// `(float)exp(0.6931471805599453094*x)`（参考实现 `celt_exp2`，即以 2 为底的指数）。
pub fn celt_exp2(x: f32) -> f32 {
    (0.693_147_180_559_945_3f64 * x as f64).exp() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp2_is_base_two() {
        assert!((celt_exp2(0.0) - 1.0).abs() < 1e-6);
        assert!((celt_exp2(1.0) - 2.0).abs() < 1e-6);
        assert!((celt_exp2(3.0) - 8.0).abs() < 1e-5);
        assert!((celt_exp2(-3.0) - 0.125).abs() < 1e-6);
    }

    #[test]
    fn cos_norm_maps_quarter_turn() {
        // celt_cos_norm(0) = 1、celt_cos_norm(1) = cos(π/2) ≈ 0、celt_cos_norm(2) ≈ -1
        assert!((celt_cos_norm(0.0) - 1.0).abs() < 1e-6);
        assert!(celt_cos_norm(1.0).abs() < 1e-6);
        assert!((celt_cos_norm(2.0) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn rsqrt_is_the_reciprocal_square_root() {
        for v in [0.25f32, 1.0, 2.0, 100.0] {
            assert!((celt_rsqrt(v) - 1.0 / v.sqrt()).abs() < 1e-6, "v={v}");
        }
    }
}
