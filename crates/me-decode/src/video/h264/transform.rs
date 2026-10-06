//! **反扫描 + 反量化 + 反变换**（ITU-T H.264 §8.5.6 ~ §8.5.13）。
//!
//! 入口是 [`super::mb::MbCoeffs`] 里那些**按扫描序**躺着的系数，出口是 4×4 / 8×8 的
//! **残差样本**。预测在 [`super::intra`] / [`super::inter`]，两者相加的结果写进
//! [`super::frame::FrameBuf`]。
//!
//! ## 索引约定（这一层最容易搞混的地方）
//!
//! 全部用 **`idx = i * n + j`** 的布局，其中 **`i` 是行（= 画面上的 y）**、
//! `j` 是列（= x）。规范里的 `c_ij` / `d_ij` 就是这个约定，别再翻一次。
//!
//! ## 三处"看起来该一样、其实不一样"的缩放
//!
//! | 对象 | 式号 | 形式 |
//! |---|---|---|
//! | 普通 4×4 系数 | 8-336/8-337 | `qP >= 24` 左移、否则右移（**24 是分界**）|
//! | 8×8 系数 | 8-356/8-357 | 分界是 **36** |
//! | `Intra_16x16` 的 16 个 DC | 8-321/8-322 | 分界 36，且**先做 Hadamard 再做缩放** |
//! | 色度 DC（2×2）| 8-326 | `((f * LS) << (qP/6)) >> 5` —— **左移再右移**，没有分界 |
//!
//! 还有一条只影响一个系数、但错了整块就偏的规则（§8.5.12.1 式 8-335）：
//! **`Intra_16x16` 的亮度块与所有色度块的 `d00` 直接等于 `c00`，不再缩放** ——
//! 因为那个位置放的是已经缩放过的 DC（见 [`intra16x16_dc`] / [`chroma_dc_2x2`]）。
//!
//! ## 反量化里的"缩放矩阵"
//!
//! 规范说 `LevelScale = weightScale * normAdjust`。`normAdjust` 是固定表
//! （式 8-315 / 8-318），`weightScale` 来自**量化矩阵**。码流默认不发量化矩阵
//! （`seq_scaling_matrix_present_flag == 0`），这时规范明确规定用
//! `Flat_4x4_16` / `Flat_8x8_16`（式 7-8/7-9，**全是 16**）—— 见
//! [`WeightScale`]。

use crate::h264::{Pps, Sps};

/// 4×4 的反 zig-zag 扫描（表 8-13）：`idx → (i, j)`，`i` 是行。
pub const ZIGZAG_4X4: [(usize, usize); 16] = [
    (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
    (2, 1), (3, 0), (3, 1), (2, 2), (1, 3), (2, 3), (3, 2), (3, 3),
];

/// 8×8 的反 zig-zag 扫描（表 8-14）：`idx → (i, j)`。
pub const ZIGZAG_8X8: [(usize, usize); 64] = [
    (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
    (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
    (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
    (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
    (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
    (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
    (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
    (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7),
];

/// 式 (8-315) 的矩阵 `v`（行 = `m = qP % 6`，列 = `v_m0 / v_m1 / v_m2`）。
///
/// 用哪一列由式 (8-314) 定：`(i%2, j%2) == (0,0)` 用第 0 列、`(1,1)` 用第 1 列、
/// 其余用第 2 列。**第 2 列不是"最大那一列"** —— `m = 0` 时 v_m0 = 10、
/// v_m2 = 13，`m = 4` 时 v_m0 = 16、v_m2 = 20，凭直觉填一定会填反。
///
/// ⚠️ 这两张表在规范里是**图片**（`v = [...]` 那个矩阵），文字层抽不出来，
/// 所以当初是凭印象写的，`m = 0` 整行倒着抄、`m = 4` 的第一项抄成了 14 ——
/// 症状是"平场整帧差 1、四象限里只有一个象限差几阶"，不像崩，但就是不对。
/// 现在由 [`tests::norm_adjust_matches_spec_8_315`] 逐项钉住。
const NORM_ADJUST_4X4: [[i32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];

/// 式 (8-318) 的矩阵 `v`（行 = `m = qP % 6`，列 = `v_m0 ..= v_m5`）。
/// 列的选用见式 (8-317)。
const NORM_ADJUST_8X8: [[i32; 6]; 6] = [
    [20, 18, 32, 19, 25, 24],
    [22, 19, 35, 21, 28, 26],
    [26, 23, 42, 24, 33, 31],
    [28, 25, 45, 26, 35, 33],
    [32, 28, 51, 30, 40, 38],
    [36, 32, 58, 34, 46, 43],
];

/// 表 8-15：`QP_C` 随 `qP_I` 的映射（`qP_I < 30` 时就是它自己）。
///
/// ⚠️ 入参是式 (8-311) 的 `qP_I = Clip3(-QpBdOffsetC, 51, QPY + qPOffset)` ——
/// **必须夹在 `[0, 51]`**（8 位时 `QpBdOffsetC == 0`，下界就是 0）。
/// 漏掉这个 Clip3 会很惨：`chroma_qp_index_offset` 是负的（x264 常写 −2）
/// 而 `QPY` 又很小时，`qP_I` 会是负数，落到 `qP % 6` 上就是负数取模 —— 8 位
/// 下表现为 `NORM_ADJUST_4X4[usize::MAX - 1]` 直接 panic（或者更坏：静默错帧）。
pub fn chroma_qp(qp_i: i32) -> i32 {
    const TABLE: [i32; 22] = [
        29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39,
    ];
    // 式 (8-311)：`qP_I = Clip3( −QpBdOffsetC, 51, QPY + qPOffset )`；
    // 8 位下 `QpBdOffsetC = 0`
    let qp_i = qp_i.clamp(0, 51);
    if qp_i < 30 {
        qp_i
    } else {
        TABLE[(qp_i - 30) as usize]
    }
}

/// §8.5.9 的 `LevelScale`：`weightScale * normAdjust`。
///
/// `weightScale` 是**量化矩阵**（4×4 时 16 项、8×8 时 64 项，**按扫描序**给，
/// 要反扫描回二维）。码流默认不发它，规范就规定用全 16 的 `Flat_*_16`。
#[derive(Debug, Clone)]
pub struct WeightScale {
    /// 4×4：`[iYCbCr + (mbIsInter ? 3 : 0)]`
    w4: [[i32; 16]; 6],
    /// 8×8：`[2 * iYCbCr + mbIsInter]`
    w8: [[i32; 64]; 2],
}

impl Default for WeightScale {
    fn default() -> Self {
        Self {
            w4: [[16; 16]; 6],
            w8: [[16; 64]; 2],
        }
    }
}

impl WeightScale {
    /// 从 SPS/PPS 里拿量化矩阵（没有的项用规范默认的 `Flat_*_16` = 全 16）。
    ///
    /// ⚠️ 规范里"某项没发"时还有一个 `Table 7-2` 的**回退链**（回退到
    /// `Default_4x4_Intra` 之类的非平坦矩阵）。引擎目前**只认平坦默认**：
    /// 真发过量化矩阵的流很少见（x264 / 各家硬件编码器默认都不发），
    /// 而回退链一旦猜错是"整帧偏色但不报错"。等碰到这样的素材再补 —— 见
    /// `h264/mod.rs` 抬头的"下一步"。
    pub fn new(sps: &Sps, pps: &Pps) -> Self {
        let mut s = Self::default();
        for (i, list) in pps.scaling4x4.iter().enumerate() {
            if let Some(l) = list {
                s.w4[i] = l.map(|v| v as i32);
            }
        }
        for (i, list) in pps.scaling8x8.iter().enumerate() {
            if let Some(l) = list {
                s.w8[i] = l.map(|v| v as i32);
            }
        }
        for (i, list) in sps.scaling4x4.iter().enumerate() {
            if let Some(l) = list {
                s.w4[i] = l.map(|v| v as i32);
            }
        }
        for (i, list) in sps.scaling8x8.iter().enumerate() {
            if let Some(l) = list {
                s.w8[i] = l.map(|v| v as i32);
            }
        }
        s
    }

    /// 4×4 的 `LevelScale4x4(m, i, j)`；`list_idx` = `iYCbCr + (inter ? 3 : 0)`。
    ///
    /// 列的选择照式 (8-314)：`(i%2, j%2)` 为 `(0,0)` / `(1,1)` / 其余 分别对应 `v_m0/1/2`。
    pub fn level_scale_4x4(&self, list_idx: usize, m: usize, i: usize, j: usize) -> i32 {
        // 量化矩阵按**扫描序**存：要按 zigzag 的位置查，不是按 (i,j) 直接查
        let pos = ZIGZAG_4X4
            .iter()
            .position(|&(zi, zj)| zi == i && zj == j)
            .unwrap_or(0);
        let w = self.w4[list_idx.min(5)][pos];
        w * match (i % 2, j % 2) {
            (0, 0) => NORM_ADJUST_4X4[m][0],
            (1, 1) => NORM_ADJUST_4X4[m][1],
            _ => NORM_ADJUST_4X4[m][2],
        }
    }

    /// 8×8 的 `LevelScale8x8(m, i, j)`；`list_idx` = `2 * iYCbCr + inter`。
    ///
    /// 列的选择**照式 (8-317) 原样写成 if-else 链**（顺序不能改，末尾是 `otherwise`）：
    ///
    /// ```text
    /// v_m0 for ( i % 4, j % 4 ) equal to (0,0),
    /// v_m1 for ( i % 2, j % 2 ) equal to (1,1),
    /// v_m2 for ( i % 4, j % 4 ) equal to (2,2),
    /// v_m3 for ( i % 4, j % 2 ) equal to (0,1)  or  ( i % 2, j % 4 ) equal to (1,0),
    /// v_m4 for ( i % 4, j % 4 ) equal to (0,2)  or  ( i % 4, j % 4 ) equal to (2,0),
    /// v_m5 otherwise;
    /// ```
    ///
    /// ⚠️ `v_m3` 的第一个条件是 **`j % 2`**、`v_m4` 的两个条件都是 `j % 4` ——
    /// 全都按 `j % 4` 写会把 `(0,5)`、`(4,1)` 这类位置判错（占 8×8 里 16/64 个位置），
    /// 症状是"8×8 用例整帧差一两阶"，小到很容易被当成舍入误差放过去。
    pub fn level_scale_8x8(&self, list_idx: usize, m: usize, i: usize, j: usize) -> i32 {
        let pos = ZIGZAG_8X8
            .iter()
            .position(|&(zi, zj)| zi == i && zj == j)
            .unwrap_or(0);
        let w = self.w8[list_idx.min(1)][pos];
        let v = if (i % 4, j % 4) == (0, 0) {
            NORM_ADJUST_8X8[m][0]
        } else if (i % 2, j % 2) == (1, 1) {
            NORM_ADJUST_8X8[m][1]
        } else if (i % 4, j % 4) == (2, 2) {
            NORM_ADJUST_8X8[m][2]
        } else if (i % 4, j % 2) == (0, 1) || (i % 2, j % 4) == (1, 0) {
            NORM_ADJUST_8X8[m][3]
        } else if (i % 4, j % 4) == (0, 2) || (i % 4, j % 4) == (2, 0) {
            NORM_ADJUST_8X8[m][4]
        } else {
            NORM_ADJUST_8X8[m][5]
        };
        w * v
    }
}

/// 把 `qP` 归一到规范允许的范围（8 位：`0..=51`）。
///
/// 规范保证 `QPY` 落在 `[-QpBdOffsetY, 51]`（8 位即 `0..=51`），但**畸形码流
/// 可以违反它**，而 `qP % 6` 一旦是负数，`usize` 转换就是 `usize::MAX - 1` 之类
/// —— 直接 panic 比"解出错帧"更糟（后者至少要过 `rbsp_trailing_bits`）。
#[inline]
fn norm_qp(qp: i32) -> i32 {
    qp.clamp(0, 51)
}

/// §8.5.6 反扫描：16 个"按扫描序"的值 → 4×4 二维数组。
///
/// `skip_dc = true` 时输入是 **15 个 AC**（`Intra_16x16` 的 AC 块、
/// 色度 AC 块），它们对应 zigzag 的**第 1~15 个位置**（第 0 个是 DC，另有其块）。
pub fn scan_4x4(vals: &[i32], skip_dc: bool) -> [i32; 16] {
    let mut c = [0i32; 16];
    if skip_dc {
        for (k, &v) in vals.iter().take(15).enumerate() {
            let (i, j) = ZIGZAG_4X4[k + 1];
            c[i * 4 + j] = v;
        }
    } else {
        for (k, &v) in vals.iter().take(16).enumerate() {
            let (i, j) = ZIGZAG_4X4[k];
            c[i * 4 + j] = v;
        }
    }
    c
}

/// §8.5.7 反扫描：64 个值 → 8×8。
pub fn scan_8x8(vals: &[i32; 64]) -> [i32; 64] {
    let mut c = [0i32; 64];
    for (k, &v) in vals.iter().enumerate() {
        let (i, j) = ZIGZAG_8X8[k];
        c[i * 8 + j] = v;
    }
    c
}

/// §8.5.12.1 反量化（普通 4×4 块）。`skip_dc` = `d00` 直接取 `c00`
/// （`Intra_16x16` 的亮度块、所有色度块 —— 见式 8-335）。
pub fn dequant_4x4(
    c: &[i32; 16],
    qp: i32,
    scale: &WeightScale,
    list_idx: usize,
    skip_dc: bool,
) -> [i32; 16] {
    let qp = norm_qp(qp);
    let m = (qp % 6) as usize;
    let mut d = [0i32; 16];
    for i in 0..4 {
        for j in 0..4 {
            if skip_dc && i == 0 && j == 0 {
                d[0] = c[0];
                continue;
            }
            let ls = scale.level_scale_4x4(list_idx, m, i, j);
            d[i * 4 + j] = if qp >= 24 {
                (c[i * 4 + j] * ls) << (qp / 6 - 4)
            } else {
                (c[i * 4 + j] * ls + (1 << (3 - qp / 6))) >> (4 - qp / 6)
            };
        }
    }
    d
}

/// §8.5.13.1 反量化（8×8 块）。
pub fn dequant_8x8(c: &[i32; 64], qp: i32, scale: &WeightScale, list_idx: usize) -> [i32; 64] {
    let qp = norm_qp(qp);
    let m = (qp % 6) as usize;
    let mut d = [0i32; 64];
    for i in 0..8 {
        for j in 0..8 {
            let ls = scale.level_scale_8x8(list_idx, m, i, j);
            d[i * 8 + j] = if qp >= 36 {
                (c[i * 8 + j] * ls) << (qp / 6 - 6)
            } else {
                (c[i * 8 + j] * ls + (1 << (5 - qp / 6))) >> (6 - qp / 6)
            };
        }
    }
    d
}

/// §8.5.12.2 反变换（4×4）：式 8-338 ~ 8-354。
pub fn inverse_4x4(d: &[i32; 16]) -> [i32; 16] {
    let mut e = [0i32; 16];
    for i in 0..4 {
        let (a0, a1, a2, a3) = (d[i * 4], d[i * 4 + 1], d[i * 4 + 2], d[i * 4 + 3]);
        e[i * 4] = a0 + a2;
        e[i * 4 + 1] = a0 - a2;
        e[i * 4 + 2] = (a1 >> 1) - a3;
        e[i * 4 + 3] = a1 + (a3 >> 1);
    }
    let mut f = [0i32; 16];
    for i in 0..4 {
        let (b0, b1, b2, b3) = (e[i * 4], e[i * 4 + 1], e[i * 4 + 2], e[i * 4 + 3]);
        f[i * 4] = b0 + b3;
        f[i * 4 + 1] = b1 + b2;
        f[i * 4 + 2] = b1 - b2;
        f[i * 4 + 3] = b0 - b3;
    }
    let mut g = [0i32; 16];
    for j in 0..4 {
        let (a0, a1, a2, a3) = (f[j], f[4 + j], f[8 + j], f[12 + j]);
        g[j] = a0 + a2;
        g[4 + j] = a0 - a2;
        g[8 + j] = (a1 >> 1) - a3;
        g[12 + j] = a1 + (a3 >> 1);
    }
    let mut h = [0i32; 16];
    for j in 0..4 {
        let (b0, b1, b2, b3) = (g[j], g[4 + j], g[8 + j], g[12 + j]);
        h[j] = b0 + b3;
        h[4 + j] = b1 + b2;
        h[8 + j] = b1 - b2;
        h[12 + j] = b0 - b3;
    }
    // 式 (8-354)：`r_ij = ( h_ij + 2^5 ) >> 6` —— 常数是 **32**。
    // ⚠️ 规范 PDF 的文字层把 `2^5` 抽成了 `25`，照文字层抄就会写成 `+ 25`：
    // 只在**有 AC 系数**时才显形（纯 DC 的块两种写法结果一样），
    // 症状是"整帧亮度偏低一点点、且误差会沿着帧内预测一路漂"。
    let mut r = [0i32; 16];
    for k in 0..16 {
        r[k] = (h[k] + 32) >> 6;
    }
    r
}

/// §8.5.13.2 反变换（8×8）：式 8-358 ~ 8-406（先横后竖，两趟一样）。
pub fn inverse_8x8(d: &[i32; 64]) -> [i32; 64] {
    fn pass(src: &[i32; 64], dst: &mut [i32; 64]) {
        for i in 0..8 {
            let o = i * 8;
            let d0 = src[o];
            let d1 = src[o + 1];
            let d2 = src[o + 2];
            let d3 = src[o + 3];
            let d4 = src[o + 4];
            let d5 = src[o + 5];
            let d6 = src[o + 6];
            let d7 = src[o + 7];
            let e0 = d0 + d4;
            let e1 = -d3 + d5 - d7 - (d7 >> 1);
            let e2 = d0 - d4;
            let e3 = d1 + d7 - d3 - (d3 >> 1);
            let e4 = (d2 >> 1) - d6;
            let e5 = -d1 + d7 + d5 + (d5 >> 1);
            let e6 = d2 + (d6 >> 1);
            let e7 = d3 + d5 + d1 + (d1 >> 1);
            let f0 = e0 + e6;
            let f1 = e1 + (e7 >> 2);
            let f2 = e2 + e4;
            let f3 = e3 + (e5 >> 2);
            let f4 = e2 - e4;
            let f5 = (e3 >> 2) - e5;
            let f6 = e0 - e6;
            let f7 = e7 - (e1 >> 2);
            dst[o] = f0 + f7;
            dst[o + 1] = f2 + f5;
            dst[o + 2] = f4 + f3;
            dst[o + 3] = f6 + f1;
            dst[o + 4] = f6 - f1;
            dst[o + 5] = f4 - f3;
            dst[o + 6] = f2 - f5;
            dst[o + 7] = f0 - f7;
        }
    }
    let mut g = [0i32; 64];
    pass(d, &mut g);
    let mut t = [0i32; 64];
    // 竖着走一趟：把列取出来当行做同样的变换
    let mut col = [0i32; 64];
    for i in 0..8 {
        for j in 0..8 {
            col[j * 8 + i] = g[i * 8 + j];
        }
    }
    pass(&col, &mut t);
    let mut r = [0i32; 64];
    for i in 0..8 {
        for j in 0..8 {
            r[i * 8 + j] = (t[j * 8 + i] + 32) >> 6;
        }
    }
    r
}

/// §8.5.10 `Intra_16x16` 的 16 个 ADC 系数：先 Hadamard（式 8-320），
/// 再缩放（式 8-321/8-322）。`c` 由 16 个 DC 电平反扫描而来。
pub fn intra16x16_dc(c: &[i32; 16], qp: i32) -> [i32; 16] {
    // H = [[1,1,1,1],[1,1,-1,-1],[1,-1,-1,1],[1,-1,1,-1]]，f = H c Hᵀ
    let mut b = [0i32; 16];
    for j in 0..4 {
        let (c0, c1, c2, c3) = (c[j], c[4 + j], c[8 + j], c[12 + j]);
        b[j] = c0 + c1 + c2 + c3;
        b[4 + j] = c0 + c1 - c2 - c3;
        b[8 + j] = c0 - c1 - c2 + c3;
        b[12 + j] = c0 - c1 + c2 - c3;
    }
    let mut f = [0i32; 16];
    for i in 0..4 {
        let (b0, b1, b2, b3) = (b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]);
        f[i * 4] = b0 + b1 + b2 + b3;
        f[i * 4 + 1] = b0 + b1 - b2 - b3;
        f[i * 4 + 2] = b0 - b1 - b2 + b3;
        f[i * 4 + 3] = b0 - b1 + b2 - b3;
    }
    // `LevelScale4x4(qP % 6, 0, 0)` = 16 * v_m0
    let qp = norm_qp(qp);
    let m = (qp % 6) as usize;
    let ls = 16 * NORM_ADJUST_4X4[m][0];
    let mut dc = [0i32; 16];
    for k in 0..16 {
        dc[k] = if qp >= 36 {
            (f[k] * ls) << (qp / 6 - 6)
        } else {
            (f[k] * ls + (1 << (5 - qp / 6))) >> (6 - qp / 6)
        };
    }
    dc
}

/// §8.5.11 色度 DC（4:2:0 的 2×2）：式 8-324 的变换 + 式 8-326 的缩放。
///
/// `c` 的 4 项按**光栅序**（`c[0]` = 左上）。返回的 `dc[blk]` 也是光栅序，
/// 对应 `chroma4x4BlkIdx = 0..3`（图 8-7a）。
pub fn chroma_dc_2x2(c: &[i32; 4], qp: i32) -> [i32; 4] {
    let (c00, c01, c10, c11) = (c[0], c[1], c[2], c[3]);
    let f = [
        c00 + c01 + c10 + c11,
        c00 - c01 + c10 - c11,
        c00 + c01 - c10 - c11,
        c00 - c01 - c10 + c11,
    ];
    let qp = norm_qp(qp);
    let m = (qp % 6) as usize;
    let ls = 16 * NORM_ADJUST_4X4[m][0];
    let mut dc = [0i32; 4];
    for k in 0..4 {
        dc[k] = ((f[k] * ls) << (qp / 6)) >> 5;
    }
    dc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **式 (8-315) 的矩阵 `v` 逐项钉死**（4×4）。
    ///
    /// 这张表在规范里是图片，抄错一位是**静默**的：解出来的画面不会崩，
    /// 只会"整体差一阶/某个区域差几阶"。真实翻车记录：`m = 0` 那行
    /// 抄成了 `[13, 16, 10]`（倒过来）、`m = 4` 抄成 `[14, 25, 20]`，
    /// 于是 qP % 6 == 0 与 == 4 的流全部偏掉。
    #[test]
    fn norm_adjust_matches_spec_8_315() {
        // 直接从 ITU-T H.264 (V15) (08/2024) p.187 的 (8-315) 抄下来 ——
        // 这行"抄自规范"是刻意的：测试与被测表必须来自**同一个外部来源**，
        // 否则两边一起错。
        let expect = [
            [10, 16, 13],
            [11, 18, 14],
            [13, 20, 16],
            [14, 23, 18],
            [16, 25, 20],
            [18, 29, 23],
        ];
        assert_eq!(NORM_ADJUST_4X4, expect);
    }

    /// **式 (8-318) 的矩阵 `v` 逐项钉死**（8×8）。
    #[test]
    fn norm_adjust_8x8_matches_spec_8_318() {
        let expect = [
            [20, 18, 32, 19, 25, 24],
            [22, 19, 35, 21, 28, 26],
            [26, 23, 42, 24, 33, 31],
            [28, 25, 45, 26, 35, 33],
            [32, 28, 51, 30, 40, 38],
            [36, 32, 58, 34, 46, 43],
        ];
        assert_eq!(NORM_ADJUST_8X8, expect);
    }

    /// 式 (8-314)：4×4 的列选择只看 `(i % 2, j % 2)`。
    ///
    /// 160 = 16（`Flat_4x4_16`）× 10 ⇒ `m = 0` 的 `(0, 0)` 类是 **10**，不是 13。
    #[test]
    fn level_scale_4x4_picks_the_column_by_parity() {
        let ws = WeightScale::default();
        for (m, row) in NORM_ADJUST_4X4.iter().enumerate() {
            assert_eq!(ws.level_scale_4x4(0, m, 0, 0), 16 * row[0], "m={m} (0,0)");
            assert_eq!(ws.level_scale_4x4(0, m, 1, 1), 16 * row[1], "m={m} (1,1)");
            assert_eq!(ws.level_scale_4x4(0, m, 0, 1), 16 * row[2], "m={m} (0,1)");
            assert_eq!(ws.level_scale_4x4(0, m, 1, 0), 16 * row[2], "m={m} (1,0)");
            assert_eq!(ws.level_scale_4x4(0, m, 2, 3), 16 * row[2], "m={m} (2,3)");
        }
        assert_eq!(ws.level_scale_4x4(0, 0, 0, 0), 160);
        assert_eq!(ws.level_scale_4x4(0, 4, 0, 0), 256);
    }

    /// 式 (8-317)：8×8 的列选择要看到 `i % 4` / `j % 4`。
    ///
    /// `(0, 3)` 与 `(0, 1)` 在 `(i%2, j%2)` 下**长得一样**（都是 `(0,1)`），
    /// 但列不同（`v_m5` vs `v_m3`）—— 用 `j % 2` 判就会把 `(0,3)` 判错。
    #[test]
    fn level_scale_8x8_distinguishes_j_mod_4() {
        let ws = WeightScale::default();
        let m = 0;
        let row = &NORM_ADJUST_8X8[m];
        assert_eq!(ws.level_scale_8x8(0, m, 0, 0), 16 * row[0], "(0,0)→v_m0");
        assert_eq!(ws.level_scale_8x8(0, m, 1, 1), 16 * row[1], "(1,1)→v_m1");
        assert_eq!(ws.level_scale_8x8(0, m, 2, 2), 16 * row[2], "(2,2)→v_m2");
        assert_eq!(ws.level_scale_8x8(0, m, 0, 1), 16 * row[3], "(0,1)→v_m3");
        assert_eq!(ws.level_scale_8x8(0, m, 0, 5), 16 * row[3], "(0,5)→v_m3（判据是 j%2）");
        assert_eq!(ws.level_scale_8x8(0, m, 1, 0), 16 * row[3], "(1,0)→v_m3");
        assert_eq!(ws.level_scale_8x8(0, m, 3, 4), 16 * row[3], "(3,4)→v_m3（i%2,j%4=(1,0)）");
        assert_eq!(ws.level_scale_8x8(0, m, 0, 2), 16 * row[4], "(0,2)→v_m4");
        assert_eq!(ws.level_scale_8x8(0, m, 2, 0), 16 * row[4], "(2,0)→v_m4");
        // `(0,3)` 的 `(i%4, j%2) = (0,1)` 成立 ⇒ 归 v_m3（写成 `j%4` 就会落到 v_m5）
        assert_eq!(ws.level_scale_8x8(0, m, 0, 3), 16 * row[3], "(0,3)→v_m3");
        // `(3,0)` 的 `(i%2, j%4) = (1,0)` ⇒ 归 v_m3 —— 第二个条件用的是 `i % 2`，不是 `i % 4`
        assert_eq!(ws.level_scale_8x8(0, m, 3, 0), 16 * row[3], "(3,0)→v_m3");
        // 真正落到 `otherwise`（v_m5）的是 (1,2)、(2,1)、(2,3)、(3,2) 这四类
        assert_eq!(ws.level_scale_8x8(0, m, 2, 1), 16 * row[5], "(2,1)→v_m5");
        assert_eq!(ws.level_scale_8x8(0, m, 3, 2), 16 * row[5], "(3,2)→v_m5");
        assert_eq!(ws.level_scale_8x8(0, m, 1, 2), 16 * row[5], "(1,2)→v_m5");
        assert_eq!(ws.level_scale_8x8(0, m, 2, 3), 16 * row[5], "(2,3)→v_m5");
    }

    /// 反扫描的**位置**必须与表 8-13 一致 —— 差一格整块就错位，
    /// 而且症状是"画面糊"不是"报错"。
    #[test]
    fn zigzag_4x4_matches_table_8_13() {        // 表 8-13 的行：zig-zag 是 c00 c01 c10 c20 c11 c02 c03 c12 c21 c30 c31 c22 c13 c23 c32 c33
        let expect = [
            (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0),
            (3, 1), (2, 2), (1, 3), (2, 3), (3, 2), (3, 3),
        ];
        for (k, &(i, j)) in expect.iter().enumerate() {
            assert_eq!(ZIGZAG_4X4[k], (i, j), "idx {k}");
        }
    }

    /// 8×8 的表也要逐项对上（表 8-14 的四段）。
    #[test]
    fn zigzag_8x8_matches_table_8_14() {
        let head = [
            (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
            (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
        ];
        assert_eq!(&ZIGZAG_8X8[..16], &head[..]);
        let tail = [
            (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
            (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7),
        ];
        assert_eq!(&ZIGZAG_8X8[48..], &tail[..]);
        // 每个位置恰好出现一次
        let mut seen = [false; 64];
        for &(i, j) in ZIGZAG_8X8.iter() {
            assert!(!seen[i * 8 + j], "({i},{j}) 出现了两次");
            seen[i * 8 + j] = true;
        }
    }

    /// 反变换对**只有 DC** 的块必须给出一个平坦块 —— 这是规范解，能手算。
    #[test]
    fn inverse_4x4_of_a_dc_only_block_is_flat() {
        let mut d = [0i32; 16];
        d[0] = 64; // 只有 DC
        let r = inverse_4x4(&d);
        // h 全是 64 ⇒ r = (64 + 32) >> 6 = 1
        assert!(r.iter().all(|&v| v == 1), "{r:?}");
    }

    /// 4×4 反变换是**可逆的线性变换**：把 DC 加一倍、结果也必须加一倍（整数上
    /// 只差取整）。用一个不产生取整歧义的值钉住。
    #[test]
    fn inverse_4x4_dc_is_linear_for_exact_values() {
        for dc in [64i32, 128, 256, -64] {
            let mut d = [0i32; 16];
            d[0] = dc;
            let r = inverse_4x4(&d);
            let expect = (dc + 32) >> 6;   // 式 (8-354)
            assert!(r.iter().all(|&v| v == expect), "dc={dc} → {r:?}");
        }
    }

    /// 8×8 的 DC-only 块同理：`r = (m + 32) >> 6`，`m` = DC 经过两趟的增益。
    /// 8×8 两趟的增益是 8×8 = 64（每趟 DC 通道是 ×8），所以 `m = 64 * dc`。
    #[test]
    fn inverse_8x8_of_a_dc_only_block_is_flat() {
        for dc in [1i32, 4, -4] {
            let mut d = [0i32; 64];
            d[0] = dc;
            let r = inverse_8x8(&d);
            // 8×8 反变换的 DC 增益也是 ×1（两趟各 ×8，最后 >>6）
            let expect = (dc + 32) >> 6;
            assert!(r.iter().all(|&v| v == expect), "dc={dc} → {r:?}");
        }
    }

    /// 表 8-15：`qP_I < 30` 时照抄，30 之后开始"折"。
    #[test]
    fn chroma_qp_follows_table_8_15() {
        for q in 0..30 {
            assert_eq!(chroma_qp(q), q);
        }
        assert_eq!(chroma_qp(30), 29);
        assert_eq!(chroma_qp(31), 30);
        assert_eq!(chroma_qp(34), 32);
        assert_eq!(chroma_qp(39), 35);
        assert_eq!(chroma_qp(45), 38);
        assert_eq!(chroma_qp(51), 39);
    }

    /// 色度 DC 的 2×2 Hadamard：只有 DC 时四个输出必须相等，
    /// 且 `dc = ((4*dc0 * ls) << (qP/6)) >> 5`。
    #[test]
    fn chroma_dc_2x2_dc_only_is_uniform() {
        let c = [10i32, 0, 0, 0];
        let out = chroma_dc_2x2(&c, 0);
        // `f` 四个元素都等于 `c00`（2×2 Hadamard 的增益同样是 ×1）；
        // 缩放取 `LevelScale4x4(qP%6 = 0, 0, 0) = 16 * v_00`，**从表里取**别写常数
        let ls = 16 * NORM_ADJUST_4X4[0][0];
        assert!(out.iter().all(|&v| v == (10 * ls) >> 5), "{out:?}");
        assert!(out[0] == out[1] && out[1] == out[2] && out[2] == out[3]);
    }

    /// `Intra_16x16` 的 DC：只有 DC 时 16 个输出相等。
    ///
    /// ⚠️ `H c Hᵀ` 对"只有 DC"的输入给出**每个元素都等于 `c[0]`**（增益 ×1），
    /// 不是 ×16 —— 我第一次就想当然写成 ×16 了。之后才按式 8-322 缩放。
    #[test]
    fn intra16x16_dc_dc_only_is_uniform() {
        let mut c = [0i32; 16];
        c[0] = 5;
        let out = intra16x16_dc(&c, 0);
        let ls = 16 * NORM_ADJUST_4X4[0][0]; // `LevelScale4x4(qP % 6 = 0, 0, 0)`
        assert!(out.iter().all(|&v| v == (5 * ls + 32) >> 6), "{out:?}");
    }

    /// 式 8-335：`skip_dc` 时 `d00 = c00`（不再缩放）。这一条错了
    /// `Intra_16x16` 与色度的整块亮度会整体偏。
    #[test]
    fn skip_dc_passes_the_dc_through_unscaled() {
        let mut c = [0i32; 16];
        c[0] = 1234;
        let ws = WeightScale::default();
        let d = dequant_4x4(&c, 30, &ws, 0, true);
        assert_eq!(d[0], 1234, "d00 必须原样透过");
        let d2 = dequant_4x4(&c, 30, &ws, 0, false);
        assert_ne!(d2[0], 1234, "不 skip 时要正常缩放");
    }

    /// 反量化有负数时的舍入要**向下取整**（算术右移），不是向零取整。
    #[test]
    fn dequant_rounds_toward_negative_infinity() {
        let mut c = [0i32; 16];
        c[1] = -1;
        let ws = WeightScale::default();
        let d = dequant_4x4(&c, 0, &ws, 0, false);
        // qP=0 且 (i,j) = (0,1)：`(i%2, j%2)` 落在"其它"那一支 ⇒ `normAdjust = v_m2`
        let raw = -(16 * NORM_ADJUST_4X4[0][2]) + (1 << 3);
        assert_eq!(d[1], raw >> 4);
        assert!(d[1] < 0);
    }
}
