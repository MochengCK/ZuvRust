//! **去块滤波**（ITU-T H.264 §8.7）：`bS` 推导、阈值、`bS < 4` 与 `bS == 4` 两套滤波器。
//!
//! ## 什么时候做
//!
//! 规范 §8.7 开头讲得很死：**"在整张图的重建完成之后、按宏块地址递增的顺序、
//! 逐宏块进行"**。所以帧内预测读到的永远是**去块前**的样本 —— 去块不参与预测，
//! 可以作为独立的一层单独验证（`tools/h264ref` 的 `--noflt` 口径就是拿
//! `ffmpeg -skip_loop_filter all` 的输出当"没去块"的参照）。
//!
//! ## 三条"看起来一样、其实不一样"的地方
//!
//! 1. **`filterOffsetA = slice_alpha_c0_offset_div2 × 2`**（不是除以 2）。
//! 2. **色度的 `bS` 用对应亮度边的 `bS`**；且 `chromaStyleFilteringFlag == 1`
//!    （4:2:0 / 4:2:2 都成立）会让 `bS == 4` 那一支**只改 p0/q0**，
//!    `p1/q1` 直接透传、`p2/q2` 也不动。
//! 3. **内部边的位置随 `transform_size_8x8_flag` 变**：4×4 变换时宏块内有
//!    x = 4/8/12 三条内部竖边；8×8 变换时只剩 x = 8（x = 4、12 落在 8×8 块内部，不是边）。
//!
//! ## `bS` 现在只覆盖帧内
//!
//! §8.7.2.1 的四档（4 / 3 / 2 / 1）里，前两档只要求"一侧是帧内编码"，后两档要看
//! 非零系数、运动矢量与参考索引。引擎目前只解 I 切片（见 [`super::mod`] 抬头那张表），
//! 所以 [`bs_for_edge`] 里后两档的分支先按"不可能发生"处理 —— **等 §8.4 接上帧间预测时
//! 必须补齐**，否则帧间流的去块强度会算小（表现为块边缘"不够糊"）。

use me_core::Result;

use super::frame::{FrameBuf, Plane};
use super::mb::PictureInfo;
use super::transform;
use crate::h264::{Pps, SliceHeader};

/// 表 8-16 的 `α′`（`indexA = 0..51`）。
///
/// 抄自 ITU-T H.264 (V15) (08/2024) p.209 的 Table 8-16（两段拼接，第二段是
/// `26..51`）。8 位下 `α = α′ << (BitDepthY − 8)` 恒等于 `α′`。
const ALPHA: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20, 22, 25, 28,
    32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226, 255, 255,
];

/// 表 8-16 的 `β′`（`indexB = 0..51`）。
const BETA: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8, 9, 9, 10,
    10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
];

/// 表 8-17 的 `t′C0`（行 = `bS = 1/2/3`，列 = `indexA = 0..51`）。
///
/// 8 位下 `tC0 = t′C0 << (BitDepth − 8)` 恒等于 `t′C0`。
const TC0: [[i32; 52]; 3] = [
    // bS = 1
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11, 13,
    ],
    // bS = 2
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
        2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 8, 10, 11, 12, 13, 15, 17,
    ],
    // bS = 3
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3,
        3, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 23, 25,
    ],
];

/// 对一张图做去块滤波（§8.7.1 的遍历顺序 + §8.7.2 的每条边）。
///
/// `info` 只用来取每宏块的 `QPY`（`I_PCM` 当 0）与 `transform_size_8x8_flag`。
pub fn deblock_picture(
    frame: &mut FrameBuf,
    info: &mut PictureInfo,
    pps: &Pps,
    sh: &SliceHeader,
) -> Result<()> {
    // `disable_deblocking_filter_idc == 1` ⇒ 整条流不过滤
    if sh.disable_deblocking_filter_idc == 1 {
        return Ok(());
    }
    // §8.7.2：filterOffsetA / filterOffsetB 是**偏移量 ×2**
    let off_a = sh.slice_alpha_c0_offset_div2 * 2;
    let off_b = sh.slice_beta_offset_div2 * 2;
    let mb_cols = frame.mb_cols();
    let mb_rows = frame.mb_rows();
    let cw = frame.u.w;
    let ch = frame.u.h;

    // `I_PCM` 宏块的 QP 当 0（§8.7.2 尾）
    let qp_of = |addr: usize| qp_for_deblock(info.mb[addr].is_pcm, info.mb[addr].qp_y);
    let qpc_of = |addr: usize| -> i32 {
        if info.mb[addr].is_pcm {
            transform::chroma_qp(0)
        } else {
            transform::chroma_qp(info.mb[addr].qp_y + pps.chroma_qp_index_offset)
        }
    };

    for addr in 0..mb_cols * mb_rows {
        let (mbx, mby) = (addr % mb_cols, addr / mb_cols);
        let (x0, y0) = (mbx * 16, mby * 16);
        let t8 = info.mb[addr].transform_8x8;
        let left_addr = if mbx > 0 { Some(addr - 1) } else { None };
        let top_addr = if mby > 0 { Some(addr - mb_cols) } else { None };

        // ── 亮度 ────────────────────────────────────────────────────────
        // 左边宏块边（x = x0）：bS = 4
        if let Some(l) = left_addr {
            filter_edge(&mut frame.y, 4, qp_of(l), qp_of(addr), false, off_a, off_b, x0, y0, 16, true);
        }
        // 内部竖边（x = 4/8/12；8×8 变换时只剩 8）：bS = 3
        for dx in [4usize, 8, 12] {
            if t8 && dx != 8 {
                continue;
            }
            filter_edge(&mut frame.y, 3, qp_of(addr), qp_of(addr), false, off_a, off_b, x0 + dx, y0, 16, true);
        }
        // 顶部宏块边（y = y0）：bS = 4
        if let Some(t) = top_addr {
            filter_edge(&mut frame.y, 4, qp_of(t), qp_of(addr), false, off_a, off_b, x0, y0, 16, false);
        }
        // 内部横边（y = 4/8/12）：bS = 3
        for dy in [4usize, 8, 12] {
            if t8 && dy != 8 {
                continue;
            }
            filter_edge(&mut frame.y, 3, qp_of(addr), qp_of(addr), false, off_a, off_b, x0, y0 + dy, 16, false);
        }

        // ── 色度（4:2:0 ⇒ 一个分量 8×8，只有 x = 0 与 x = 4 两条竖边） ──────
        // §8.7.2：色度的 bS 取**对应亮度边**的 bS（x=0 → 宏块边 4；x=4 → 内部边 3）
        let (cx0, cy0) = (mbx * 8, mby * 8);
        if cx0 == 0 || cx0 + 8 > cw || cy0 + 8 > ch {
            continue;
        }
        for idx in 0..2usize {
            if let Some(l) = left_addr {
                filter_edge(plane_mut(frame, idx), 4, qpc_of(l), qpc_of(addr), true, off_a, off_b, cx0, cy0, 8, true);
            }
            filter_edge(plane_mut(frame, idx), 3, qpc_of(addr), qpc_of(addr), true, off_a, off_b, cx0 + 4, cy0, 8, true);
            if let Some(t) = top_addr {
                filter_edge(plane_mut(frame, idx), 4, qpc_of(t), qpc_of(addr), true, off_a, off_b, cx0, cy0, 8, false);
            }
            filter_edge(plane_mut(frame, idx), 3, qpc_of(addr), qpc_of(addr), true, off_a, off_b, cx0, cy0 + 4, 8, false);
        }
    }
    Ok(())
}

fn plane_mut(f: &mut FrameBuf, idx: usize) -> &mut Plane {
    if idx == 0 {
        &mut f.u
    } else {
        &mut f.v
    }
}

/// 越界取 0（调用方保证不会真越界：图像边界那条边根本不会送进来）。
#[inline]
fn at(pl: &Plane, x: isize, y: isize) -> i32 {
    if x < 0 || y < 0 || x as usize >= pl.w || y as usize >= pl.h {
        0
    } else {
        pl.data[y as usize * pl.w + x as usize] as i32
    }
}

#[inline]
fn put(pl: &mut Plane, x: isize, y: isize, v: i32) {
    if x >= 0 && y >= 0 && (x as usize) < pl.w && (y as usize) < pl.h {
        pl.data[y as usize * pl.w + x as usize] = v.clamp(0, 255) as u8;
    }
}

/// 沿一条边**逐位置**滤波（式 8-442 ~ 8-449 + §8.7.2）。
///
/// 一条边跨 `n` 个位置（亮度 16、色度 8），每个位置各有 `p0..p3` / `q0..q3`。
/// ⚠️ 这里必须按位置循环 —— 一开始写成"只取边起点那一行"，结果每个内部边只滤了
/// 第 0 行，症状是"第 0 行全对、其余行差一两阶"，很容易被当成舍入误差。
#[allow(clippy::too_many_arguments)]
fn filter_edge(
    pl: &mut Plane,
    bs: u8,
    qp_p: i32,
    qp_q: i32,
    chroma: bool,
    off_a: i32,
    off_b: i32,
    x: usize,
    y: usize,
    n: usize,
    vertical: bool,
) {
    if bs == 0 {
        return;
    }
    // §8.7.2.2 的阈值只跟 QP 有关，与位置无关 ⇒ 提到循环外算一次
    let qp_av = (qp_p + qp_q + 1) >> 1;
    let index_a = (qp_av + off_a).clamp(0, 51) as usize;
    let index_b = (qp_av + off_b).clamp(0, 51) as usize;
    let alpha = ALPHA[index_a];
    let beta = BETA[index_b];
    // 4:2:0 / 4:2:2 ⇒ ChromaArrayType != 3 ⇒ 式 (8-450) 恒为 1
    let chroma_style = chroma;

    for k in 0..n {
        let (px, py) = if vertical {
            (x as isize, y as isize + k as isize)
        } else {
            (x as isize + k as isize, y as isize)
        };
        let mut p = [0i32; 4];
        let mut q = [0i32; 4];
        for i in 0..4isize {
            if vertical {
                p[i as usize] = at(pl, px - i - 1, py);
                q[i as usize] = at(pl, px + i, py);
            } else {
                p[i as usize] = at(pl, px, py - i - 1);
                q[i as usize] = at(pl, px, py + i);
            }
        }
        // 式 (8-460)
        if !((p[0] - q[0]).abs() < alpha && (p[1] - p[0]).abs() < beta && (q[1] - q[0]).abs() < beta) {
            continue;
        }

        let (np0, nq0, np1, nq1, np2, nq2);
        if bs == 4 {
            // §8.7.2.4
            let ap = (p[2] - p[0]).abs();
            let aq = (q[2] - q[0]).abs();
            let lim = (alpha >> 2) + 2;
            let strong_p = !chroma_style && ap < beta && (p[0] - q[0]).abs() < lim;
            let strong_q = !chroma_style && aq < beta && (p[0] - q[0]).abs() < lim;
            np0 = if strong_p {
                (p[2] + 2 * p[1] + 2 * p[0] + 2 * q[0] + q[1] + 4) >> 3
            } else {
                (2 * p[1] + p[0] + q[1] + 2) >> 2
            };
            np1 = if strong_p { (p[2] + p[1] + p[0] + q[0] + 2) >> 2 } else { p[1] };
            np2 = if strong_p {
                (2 * p[3] + 3 * p[2] + p[1] + p[0] + q[0] + 4) >> 3
            } else {
                p[2]
            };
            nq0 = if strong_q {
                (p[1] + 2 * p[0] + 2 * q[0] + 2 * q[1] + q[2] + 4) >> 3
            } else {
                (2 * q[1] + q[0] + p[1] + 2) >> 2
            };
            nq1 = if strong_q { (p[0] + q[0] + q[1] + q[2] + 2) >> 2 } else { q[1] };
            nq2 = if strong_q {
                (2 * q[3] + 3 * q[2] + q[1] + q[0] + p[0] + 4) >> 3
            } else {
                q[2]
            };
        } else {
            // §8.7.2.3
            let ap = (p[2] - p[0]).abs();
            let aq = (q[2] - q[0]).abs();
            let tc0 = TC0[(bs - 1) as usize][index_a];
            let tc = if chroma_style {
                tc0 + 1
            } else {
                tc0 + i32::from(ap < beta) + i32::from(aq < beta)
            };
            // 式 (8-467)
            let d = (((q[0] - p[0]) << 2) + (p[1] - q[1]) + 4) >> 3;
            let d = d.clamp(-tc, tc);
            np0 = p[0] + d;
            nq0 = q[0] - d;
            // 式 (8-470)/(8-472)：色度风格下 p1/q1 一律透传
            np1 = if !chroma_style && ap < beta {
                let d = (p[2] + ((p[0] + q[0] + 1) >> 1) - (p[1] << 1)) >> 1;
                p[1] + d.clamp(-tc0, tc0)
            } else {
                p[1]
            };
            nq1 = if !chroma_style && aq < beta {
                let d = (q[2] + ((p[0] + q[0] + 1) >> 1) - (q[1] << 1)) >> 1;
                q[1] + d.clamp(-tc0, tc0)
            } else {
                q[1]
            };
            np2 = p[2];
            nq2 = q[2];
        }

        // 写回（只改 p0..p2 / q0..q2）
        for (i, v) in [np0, np1, np2].into_iter().enumerate() {
            let i = i as isize;
            if vertical {
                put(pl, px - i - 1, py, v);
            } else {
                put(pl, px, py - i - 1, v);
            }
        }
        for (i, v) in [nq0, nq1, nq2].into_iter().enumerate() {
            let i = i as isize;
            if vertical {
                put(pl, px + i, py, v);
            } else {
                put(pl, px, py + i, v);
            }
        }
    }
}

/// §8.7.2.1 的 `bS`（当前只覆盖帧内；见模块抬头的说明）。
pub fn bs_for_edge(is_mb_edge: bool, intra_p: bool, intra_q: bool) -> u8 {
    // 帧内宏块边 ⇒ 4；帧内内部边 ⇒ 3；后面两档（2 / 1）要看非零系数与运动信息，
    // 等 §8.4 接上帧间预测时补。
    if intra_p || intra_q {
        if is_mb_edge {
            4
        } else {
            3
        }
    } else {
        0
    }
}

/// `I_PCM` 宏块在去块时 QP 当 0（§8.7.2 尾）—— 单独留一个函数是为了让调用点的意图显式。
pub fn qp_for_deblock(mb_is_pcm: bool, qp: i32) -> i32 {
    if mb_is_pcm {
        0
    } else {
        qp
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    /// **表 8-16 / 8-17 的结构性质**（值本身来自规范 p.209/p.210 的表，
    /// 由 `tools` 侧的抽取脚本从 PDF 文字层程序化取出）。
    ///
    /// 这两张表抄错一位不会崩、只会"滤波略强/略弱"，必须有机器判据。
    #[test]
    fn threshold_tables_are_well_formed() {
        assert_eq!(ALPHA.len(), 52);
        assert_eq!(BETA.len(), 52);
        assert_eq!(ALPHA[0], 0);
        assert_eq!(ALPHA[16], 4);
        assert_eq!(ALPHA[51], 255);
        assert_eq!(BETA[0], 0);
        assert_eq!(BETA[16], 2);
        assert_eq!(BETA[51], 18);
        for i in 1..52 {
            assert!(ALPHA[i] >= ALPHA[i - 1], "α′ 在 {i} 处不再单调");
            assert!(BETA[i] >= BETA[i - 1], "β′ 在 {i} 处不再单调");
        }
        for (bs, row) in TC0.iter().enumerate() {
            assert_eq!(row.len(), 52);
            for (i, w) in row.iter().enumerate().skip(1) {
                assert!(*w >= row[i - 1], "t′C0(bS={}) 在 {i} 处不再单调", bs + 1);
            }
        }
        for (i, idx) in (0..52).enumerate() {
            assert!(TC0[0][idx] <= TC0[1][idx], "bS=2 应不弱于 bS=1（indexA={i}）");
            assert!(TC0[1][idx] <= TC0[2][idx], "bS=3 应不弱于 bS=2（indexA={i}）");
        }
        assert_eq!([TC0[0][51], TC0[1][51], TC0[2][51]], [13, 17, 25]);
    }

    /// 一条完全平坦的边不该被改动。
    #[test]
    fn a_flat_edge_is_left_alone() {
        let mut pl = Plane::new(16, 16);
        pl.data.fill(100);
        filter_edge(&mut pl, 4, 20, 20, false, 0, 0, 8, 0, 16, true);
        assert!(pl.data.iter().all(|&v| v == 100));
    }

    /// `indexA = 0` ⇒ `α = 0` ⇒ 式 (8-460) 的 `|p0 − q0| < α` 恒不成立 ⇒ 一个样本都不动。
    #[test]
    fn zero_alpha_disables_the_whole_edge() {
        let mut pl = Plane::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                pl.data[y * 16 + x] = if x < 8 { 100 } else { 140 };
            }
        }
        filter_edge(&mut pl, 3, 0, 0, false, 0, 0, 8, 0, 16, true);
        assert!(pl
            .data
            .chunks(16)
            .all(|r| r[..8].iter().all(|&v| v == 100) && r[8..].iter().all(|&v| v == 140)));
    }

    /// `bS == 4` 的**非**强滤波分支（式 8-480/8-481/8-482）。
    ///
    /// 边在 x = 8：`p = [10, 11, 30, 40]`、`q = [11, 12, 31, 41]`，qp = 20
    /// ⇒ indexA = 20 ⇒ α = 7、β = 3。
    ///
    /// 式 (8-460) 的三个条件都成立（`|p0−q0| = 1 < 7`、`|p1−p0| = 1 < 3`、`|q1−q0| = 1 < 3`），
    /// 而 (8-476) 的 `ap = |30−10| = 20 ≥ β` ⇒ **不**走强滤波，走 (8-480)：
    /// `p0' = (2*11 + 10 + 12 + 2) >> 2 = 46 >> 2 = 11`，p1/p2 透传；
    /// `q0' = (2*12 + 11 + 11 + 2) >> 2 = 48 >> 2 = 12`。
    #[test]
    fn bs4_weak_branch_matches_equation_8_480() {
        let mut pl = Plane::new(16, 16);
        for y in 0..16 {
            // x = 4..7 → p3..p0；x = 8..11 → q0..q3
            for (x, v) in (4..12).zip([40, 30, 11, 10, 11, 12, 31, 41]) {
                pl.data[y * 16 + x] = v;
            }
        }
        filter_edge(&mut pl, 4, 20, 20, false, 0, 0, 8, 0, 16, true);
        for y in 0..16 {
            assert_eq!(pl.data[y * 16 + 7], 11, "p0' = (2*p1 + p0 + q1 + 2) >> 2");
            assert_eq!(pl.data[y * 16 + 6], 11, "p1 透传（8-481）");
            assert_eq!(pl.data[y * 16 + 5], 30, "p2 透传（8-482）");
            assert_eq!(pl.data[y * 16 + 8], 12, "q0' = (2*q1 + q0 + p1 + 2) >> 2");
            assert_eq!(pl.data[y * 16 + 9], 12, "q1 透传");
            assert_eq!(pl.data[y * 16 + 10], 31, "q2 透传");
        }
    }

    /// 色度风格（`chromaStyleFilteringFlag == 1`）下 `bS == 4` 只改 p0/q0。
    #[test]
    fn chroma_bs4_only_touches_p0_and_q0() {
        let mut pl = Plane::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                pl.data[y * 16 + x] = if x < 8 { 100 } else { 104 };
            }
        }
        filter_edge(&mut pl, 4, 20, 20, true, 0, 0, 8, 0, 16, true);
        // p0' = (2*100 + 100 + 104 + 2) >> 2 = 406 >> 2 = 101；q0' = (2*104 + 104 + 100 + 2) >> 2 = 414 >> 2 = 103
        assert_eq!(pl.data[7], 101);
        assert_eq!(pl.data[8], 103);
        assert_eq!(pl.data[6], 100, "色度风格下 p1 透传");
        assert_eq!(pl.data[5], 100, "色度风格下 p2 透传");
        assert_eq!(pl.data[9], 104);
        assert_eq!(pl.data[10], 104);
    }

    /// `bS < 4` 的 (8-467)~(8-472)：拿一条能手算的边。
    ///
    /// 边在 x = 8，`p = [31, 31, 31, 31]`、`q = [33, 33, 33, 33]`，qp = 18、偏移 0
    /// ⇒ indexA = indexB = 18 ⇒ α = 5、β = 2、t′C0(bS=3) = 1。
    /// `ap = aq = 0 < β` ⇒ tC = 3；`Δ = ((2 << 2) + 0 + 4) >> 3 = 1` ⇒ p0′ = 32、q0′ = 32。
    /// `p1′ = 31 + ((31 + 32 − 62) >> 1) = 31 + 0 = 31`；
    /// `q1′ = 33 + ((33 + 32 − 66) >> 1) = 33 + (−1) = 32`（`>>` 是向下取整）。
    #[test]
    fn bs3_branch_matches_equations_8_467_to_8_472() {
        let mut pl = Plane::new(16, 16);
        for y in 0..16 {
            for x in 0..8 {
                pl.data[y * 16 + x] = 31;
            }
            for x in 8..16 {
                pl.data[y * 16 + x] = 33;
            }
        }
        filter_edge(&mut pl, 3, 18, 18, false, 0, 0, 8, 0, 16, true);
        for y in 0..16 {
            assert_eq!(pl.data[y * 16 + 7], 32, "p0' = p0 + Δ");
            assert_eq!(pl.data[y * 16 + 8], 32, "q0' = q0 − Δ");
            assert_eq!(pl.data[y * 16 + 6], 31, "p1'");
            assert_eq!(pl.data[y * 16 + 9], 32, "q1'（算术右移向下取整）");
            assert_eq!(pl.data[y * 16 + 5], 31, "p2 透传（8-474）");
            assert_eq!(pl.data[y * 16 + 10], 33, "q2 透传（8-475）");
        }
    }

    /// **整条边跨 16 个位置都要滤到** —— 这条曾经写成"只滤边起点那一行"，
    /// 症状是"第 0 行全对、其余行差一两阶"。
    #[test]
    fn every_position_along_the_edge_is_filtered() {
        let mut pl = Plane::new(16, 16);
        // 只有第 7 行是"该被滤"的（左边 31、右边 33），其余行故意做成不满足条件
        for y in 0..16 {
            for x in 0..8 {
                pl.data[y * 16 + x] = if y == 7 { 31 } else { 90 };
            }
            for x in 8..16 {
                pl.data[y * 16 + x] = if y == 7 { 33 } else { 90 };
            }
        }
        filter_edge(&mut pl, 3, 18, 18, false, 0, 0, 8, 0, 16, true);
        assert_eq!(pl.data[7 * 16 + 7], 32, "第 7 行被滤了（p0 = s′[7][7]）");
        assert_eq!(pl.data[7], 90, "其余行左右相同，不满足 |p0−q0| < α，不该动");
        assert_eq!(pl.data[15 * 16 + 7], 90, "最后一行同样不该动 —— 这条边跨 16 个位置");
    }

    /// `bS` 的推导（§8.7.2.1）：目前只覆盖帧内，宏块边 4 / 内部边 3。
    #[test]
    fn bs_for_intra_edges() {
        assert_eq!(bs_for_edge(true, true, true), 4);
        assert_eq!(bs_for_edge(false, true, true), 3);
        assert_eq!(bs_for_edge(true, false, true), 4);
        assert_eq!(bs_for_edge(true, false, false), 0, "帧间边还没实现（§8.4 之后补）");
    }

    /// `I_PCM` 宏块在去块时 QP 当 0。
    #[test]
    fn ipcm_macroblock_counts_as_qp_zero() {
        assert_eq!(qp_for_deblock(true, 37), 0);
        assert_eq!(qp_for_deblock(false, 37), 37);
    }
}
