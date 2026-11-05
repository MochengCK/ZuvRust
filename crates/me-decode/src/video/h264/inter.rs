//! **帧间预测**（§8.4）：运动矢量预测、亚像素插值、加权预测。
//!
//! ## 这一层只做"像素"，不碰语法
//!
//! 与 [`super::intra`] 同一个分工：输入是参考面 + 运动矢量，输出是预测块，
//! 不读码流、不改状态。这样"6 抽头滤波抽错一个系数"能靠单测直接钉住，
//! 不用整帧对拍才能发现。
//!
//! ## 三个必须照着规范做、不能"看着差不多"的地方
//!
//! 1. **运动矢量单位是 1/4 亮度样本**。色度不是"把亮度 MV 除以 2"：
//!    4:2:0 下 1 个亮度样本 = 2 个色度样本，所以 1/4 亮度样本 = 1/8 色度样本 × 4
//!    —— 见 [`chroma_mv`]。除错一次，画面就是整体偏一到两个像素。
//! 2. **半样本位置用 6 抽头 `[1, -5, 20, 20, -5, 1] >> 5`，四分之一位置是两个
//!    最近半/整样本的平均**（`(a + b + 1) >> 1`）。把四分之一位置也直接套 6 抽头
//!    会得到"稍微太锐"的画面——对拍时表现为高频区域差几个样本值。
//! 3. **整样本位置不做任何滤波**（不是"6 抽头跑一遍结果不变"，中间要短路）。
//!
//! ## 越界
//!
//! §8.4.2.2.1 要求画面外的样本"由画面内最近的样本代替"，等价于坐标夹回画面内。
//! 这里一律走 [`Plane::get_clamped`] —— 帧缓冲是**紧的**（没有留白），
//! 所以夹取是唯一的正确做法，不能指望留白兜住。

use super::frame::Plane;

/// 运动矢量，单位 **1/4 亮度样本**（§8.4.1）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mv {
    pub x: i32,
    pub y: i32,
}

impl Mv {
    pub const ZERO: Mv = Mv { x: 0, y: 0 };

    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// 一个 **4×4 亮度块**的帧间信息。
///
/// 为什么按 4×4 存：运动矢量预测（§8.4.1.3）取邻居就是按 4×4 块取的
/// （`A` 是左邻 4×4、`B` 是上邻、`C` 是右上），去块滤波的 `bS` 也按 4×4 比。
/// 存成"每个分区一个 MV"会额外需要一套坐标换算，容易在子分区边界错一格。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MvInfo {
    pub mv: Mv,
    /// `-1` = 不可用（帧内宏块、或还没解到的邻居）
    pub ref_idx: i8,
}

impl Default for MvInfo {
    fn default() -> Self {
        Self {
            mv: Mv::ZERO,
            ref_idx: -1,
        }
    }
}

impl MvInfo {
    pub fn is_available(&self) -> bool {
        self.ref_idx >= 0
    }
}

/// §8.4.1.3.1 的运动矢量预测 —— **按"参考索引匹配个数"分三种情况**。
///
/// ```text
/// 三个邻居里 ref 与当前相同的个数 = 0：
///     B、C 都不可用而 A 可用 → 用 A
///     否则 → 三者取中位数（不可用的按 0 参与）
/// 个数 = 1 → 直接用在用的那个
/// 个数 >= 2 → 三者取中位数（"多数派"里挑中间值，不会被单个离群值带跑）
/// ```
///
/// ⚠️ 两个容易写错的地方：
/// 1. **不可用的邻居按 0 参与中位数**，而不是"从候选里去掉再算平均"。
///    去掉之后两两平均，实测会让画面整块偏 5～20 个像素（还不报错）。
/// 2. **判据是参考索引相同**，不是"有 MV 就行"。多参考帧的流上按后者写会
///    把不同参考帧的 MV 混进来。
pub fn predict_mv(ref_idx: i8, a: Option<MvInfo>, b: Option<MvInfo>, c: Option<MvInfo>) -> Mv {
    let matched = [a, b, c]
        .iter()
        .filter(|x| matches!(x, Some(m) if m.ref_idx == ref_idx))
        .count();
    let mv_or_zero = |x: Option<MvInfo>| x.map(|m| m.mv).unwrap_or(Mv::ZERO);
    match matched {
        0 => {
            // B、C 都不可用、只有 A 可用：直接用 A（规范的例外）
            if b.is_none() && c.is_none() && a.is_some() {
                mv_or_zero(a)
            } else {
                median3(mv_or_zero(a), mv_or_zero(b), mv_or_zero(c))
            }
        }
        1 => {
            if a.is_some_and(|m| m.ref_idx == ref_idx) {
                mv_or_zero(a)
            } else if b.is_some_and(|m| m.ref_idx == ref_idx) {
                mv_or_zero(b)
            } else {
                mv_or_zero(c)
            }
        }
        _ => median3(mv_or_zero(a), mv_or_zero(b), mv_or_zero(c)),
    }
}

#[inline]
fn median3(a: Mv, b: Mv, c: Mv) -> Mv {
    Mv::new(median3_i(a.x, b.x, c.x), median3_i(a.y, b.y, c.y))
}

#[inline]
fn median3_i(a: i32, b: i32, c: i32) -> i32 {
    a + b + c - a.min(b).min(c) - a.max(b).max(c)
}

/// 亮度 MV → 色度 MV（1/8 色度样本单位）。
///
/// 4:2:0：1 个亮度样本 = 2 个色度样本，而 MV 是按 1/4 亮度样本记的，
/// 于是 1/4 亮度样本 = 0.5 色度样本 = 4 个 1/8 色度样本。
#[inline]
pub fn chroma_mv(mv: Mv) -> Mv {
    Mv::new(mv.x * 4, mv.y * 4)
}

/// 加权预测的权重（§8.4.2.2.3 / 表 7-23）。
///
/// `x264` 默认开着 `weightp`，P 帧就会带 `pred_weight_table`。**不应用它**
/// 的症状是"整块颜色/亮度对但整体偏"，而且只出现在 P 帧、I 帧完全正常 ——
/// 很容易被误判成"帧间预测整体错了"。
#[derive(Clone, Copy, Debug)]
pub struct Weight {
    pub w: i32,
    pub o: i32,
    pub log_wd: u32,
}

impl Weight {
    /// 不做加权（`pred_weight_table` 里没这一项时的缺省值）。
    #[inline]
    pub fn identity() -> Self {
        Self {
            w: 1,
            o: 0,
            log_wd: 0,
        }
    }

    /// §8.4.2.2.3 式 8-274：`Clip1(((w * p + (1 << (logWD-1))) >> logWD) + o)`
    #[inline]
    fn apply(&self, p: i32) -> i32 {
        if self.log_wd == 0 {
            (self.w * p + self.o).clamp(0, 255)
        } else {
            let round = 1 << (self.log_wd - 1);
            (((self.w * p + round) >> self.log_wd) + self.o).clamp(0, 255)
        }
    }
}

/// 6 抽头滤波系数（§8.4.2.2.1 式 8-241）。
const TAP: [i32; 6] = [1, -5, 20, 20, -5, 1];

/// **半样本位置**取一个值：水平（`vertical == false`）或垂直方向做 6 抽头。
///
/// `(x, y)` 是**整样本**坐标，返回值是 `(x + 0.5, y)` 或 `(x, y + 0.5)` 处的样本。
/// 画面外按 §8.4.2.2.1 夹回画面内。
#[inline]
fn half_sample(plane: &Plane, x: i32, y: i32, vertical: bool) -> i32 {
    let mut acc = 0;
    for (i, t) in TAP.iter().enumerate() {
        let o = i as i32 - 2;
        let (px, py) = if vertical { (x, y + o) } else { (x + o, y) };
        acc += t * plane.get_clamped(px, py) as i32;
    }
    // 规范：`(sum + 16) >> 5`，结果再 Clip3(0, 255)
    ((acc + 16) >> 5).clamp(0, 255)
}

/// 半样本网格上的一个值（`x2`/`y2` 是**半样本**坐标）。
///
/// `x2` 偶 = 整样本列，奇 = 该列与下一列之间；`y2` 同理。
/// 两个方向都取半时先横后竖（可分离，顺序不影响结果）。
#[inline]
fn pel2(plane: &Plane, x2: i32, y2: i32) -> i32 {
    let (fx, fy) = (x2 & 1 == 1, y2 & 1 == 1);
    let (x, y) = (x2 >> 1, y2 >> 1);
    match (fx, fy) {
        (false, false) => plane.get_clamped(x, y) as i32,
        (true, false) => half_sample(plane, x, y, false),
        (false, true) => half_sample(plane, x, y, true),
        (true, true) => {
            // 竖向 6 抽头，每一行都先在水平方向插值 —— 与"先竖后横"等价
            let mut acc = 0;
            for (i, t) in TAP.iter().enumerate() {
                let o = i as i32 - 2;
                acc += t * half_sample(plane, x, y + o, false);
            }
            ((acc + 16) >> 5).clamp(0, 255)
        }
    }
}

/// 一个 4×4（或更大的）亮度预测块：把 `mv`（1/4 样本）作用在 `(x, y)` 上，
/// 写进 `out`（行优先，`w` 宽，`h` 高，stride = `w`）。
pub fn predict_luma(plane: &Plane, x: i32, y: i32, mv: Mv, w: usize, h: usize, out: &mut [u8]) {
    predict_luma_w(plane, x, y, mv, w, h, out, Weight::identity())
}

/// 带加权预测的亮度运动补偿。
pub fn predict_luma_w(
    plane: &Plane,
    x: i32,
    y: i32,
    mv: Mv,
    w: usize,
    h: usize,
    out: &mut [u8],
    weight: Weight,
) {
    debug_assert!(out.len() >= w * h);
    for j in 0..h {
        for i in 0..w {
            let sx = (x + i as i32) * 4 + mv.x;
            let sy = (y + j as i32) * 4 + mv.y;
            // 整样本/半样本网格坐标 + 是否需要在该轴上取平均
            let x2 = sx >> 1;
            let y2 = sy >> 1;
            let avg_x = sx & 1 == 1;
            let avg_y = sy & 1 == 1;
            let v = match (avg_x, avg_y) {
                (false, false) => pel2(plane, x2, y2),
                (true, false) => (pel2(plane, x2, y2) + pel2(plane, x2 + 1, y2) + 1) >> 1,
                (false, true) => (pel2(plane, x2, y2) + pel2(plane, x2, y2 + 1) + 1) >> 1,
                (true, true) => {
                    let a = pel2(plane, x2, y2);
                    let b = pel2(plane, x2 + 1, y2);
                    let c = pel2(plane, x2, y2 + 1);
                    let d = pel2(plane, x2 + 1, y2 + 1);
                    (a + b + c + d + 2) >> 2
                }
            };
            out[j * w + i] = weight.apply(v.clamp(0, 255)) as u8;
        }
    }
}

/// 一个色度预测块（§8.4.2.2.2 的双线性插值，1/8 样本精度）。
///
/// `mv` 传**亮度单位**的 MV，内部会换算（[`chroma_mv`]）；`(x, y)` 是色度坐标。
pub fn predict_chroma(plane: &Plane, x: i32, y: i32, mv: Mv, w: usize, h: usize, out: &mut [u8]) {
    predict_chroma_w(plane, x, y, mv, w, h, out, Weight::identity())
}

/// 带加权预测的色度运动补偿。
pub fn predict_chroma_w(
    plane: &Plane,
    x: i32,
    y: i32,
    mv: Mv,
    w: usize,
    h: usize,
    out: &mut [u8],
    weight: Weight,
) {
    let cmv = chroma_mv(mv);
    for j in 0..h {
        for i in 0..w {
            let sx = (x + i as i32) * 8 + cmv.x;
            let sy = (y + j as i32) * 8 + cmv.y;
            let (ix, fx) = (sx >> 3, sx & 7);
            let (iy, fy) = (sy >> 3, sy & 7);
            let v = if fx == 0 && fy == 0 {
                plane.get_clamped(ix, iy) as i32
            } else {
                // 规范式 8-268：四个角点按 (8-fx)*(8-fy) 之类加权，最后 +32 >> 6
                let a = plane.get_clamped(ix, iy) as i32;
                let b = plane.get_clamped(ix + 1, iy) as i32;
                let c = plane.get_clamped(ix, iy + 1) as i32;
                let d = plane.get_clamped(ix + 1, iy + 1) as i32;
                let wx1 = 8 - fx;
                let wy1 = 8 - fy;
                let val = a * wx1 * wy1 + b * fx * wy1 + c * wx1 * fy + d * fx * fy;
                (val + 32) >> 6
            };
            out[j * w + i] = weight.apply(v.clamp(0, 255)) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane_of(w: usize, h: usize, f: impl Fn(usize, usize) -> u8) -> Plane {
        let mut p = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                p.set(x, y, f(x, y));
            }
        }
        p
    }

    #[test]
    fn zero_mv_is_a_plain_copy() {
        // 整样本位置**不能**被滤波：6 抽头跑一遍在平坦区看不出来，
        // 在有纹理的地方会明显变糊 —— 这条用棋盘格钉住"逐样本相等"
        let p = plane_of(32, 32, |x, y| if (x + y) % 2 == 0 { 20 } else { 230 });
        let mut out = [0u8; 16];
        predict_luma(&p, 8, 8, Mv::ZERO, 4, 4, &mut out);
        for j in 0..4 {
            for i in 0..4 {
                assert_eq!(out[j * 4 + i], p.get(8 + i, 8 + j), "({i},{j}) 应逐样本相等");
            }
        }
    }

    #[test]
    fn half_pel_horizontal_matches_the_6_tap_formula() {
        // 取一段线性斜坡 y = 100 + 10x：6 抽头 [1,-5,20,20,-5,1]/32 对线性信号
        // 是"恒等"的（系数和为 32），所以半样本处应等于两整样本的平均 => 105 附近。
        let p = plane_of(32, 32, |x, _| (60 + 10 * x) as u8);
        let mut out = [0u8; 4];
        // mv.x = 2 表示偏移半个亮度样本
        predict_luma(&p, 4, 4, Mv::new(2, 0), 1, 1, &mut out);
        assert_eq!(out[0], 105, "线性斜坡上半样本 = 相邻两整样本平均");
    }

    #[test]
    fn quarter_pel_is_the_average_of_neighbours() {
        // 两整样本 100 / 200：1/4 位置 = (100 + 半样本)/2，半样本 = 150 → 125
        let mut p = Plane::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                p.set(x, y, if x >= 16 { 200 } else { 100 });
            }
        }
        let mut out = [0u8; 1];
        // 从 x=15 起偏移 1/4 样本 → 落在 15 与 15.5 之间
        predict_luma(&p, 15, 4, Mv::new(1, 0), 1, 1, &mut out);
        assert_eq!(out[0], 125);
    }

    #[test]
    fn chroma_mv_scales_by_four() {
        // 1 个亮度样本（mv = 4）在色度上是 2 个色度样本 = 16 个 1/8
        assert_eq!(chroma_mv(Mv::new(4, -4)), Mv::new(16, -16));
        assert_eq!(chroma_mv(Mv::new(-1, 3)), Mv::new(-4, 12));
    }

    fn mi(x: i32, y: i32, r: i8) -> MvInfo {
        MvInfo {
            mv: Mv::new(x, y),
            ref_idx: r,
        }
    }

    #[test]
    fn mv_prediction_follows_the_ref_match_count() {
        let a = mi(2, 0, 0);
        let b = mi(2, 4, 0);
        let c = mi(8, 4, 0);
        // 三个都同参考索引 → 取中位数（x: 2,2,8 → 2；y: 0,4,4 → 4）
        assert_eq!(predict_mv(0, Some(a), Some(b), Some(c)), Mv::new(2, 4));

        // 只有两个同索引 → 仍然是三取中位数，**不可用的 C 按 0 参与**
        // （y: median(0, 4, 0) = 0，不是 4 —— 这条容易想当然）
        assert_eq!(predict_mv(0, Some(a), Some(b), None), Mv::new(2, 0));

        // 只有一个同索引 → 直接用它
        let other = mi(30, 30, 1);
        assert_eq!(predict_mv(0, Some(other), Some(b), Some(other)), Mv::new(2, 4));

        // 左邻是帧内（ref = -2）：按 0 参与，仍是中位数
        let intra = mi(0, 0, -2);
        assert_eq!(predict_mv(0, Some(intra), Some(b), Some(c)), Mv::new(2, 4));

        // 全都不可用 → 0
        assert_eq!(predict_mv(0, None, None, None), Mv::ZERO);

        // 只有 A 可用（B、C 都没有）→ 用 A
        assert_eq!(predict_mv(0, Some(a), None, None), Mv::new(2, 0));
    }

    #[test]
    fn out_of_picture_reads_are_clamped() {
        let p = plane_of(16, 16, |x, _| x as u8);
        let mut out = [0u8; 4];
        // 从 x=0 起偏移 -8（两个整样本）：前 3 个落在画面外（按最左列 0 处理），
        // 第 4 个对应源 x = 1
        predict_luma(&p, 0, 0, Mv::new(-8, 0), 4, 1, &mut out);
        assert_eq!(&out[..], &[0, 0, 0, 1]);
    }
}
