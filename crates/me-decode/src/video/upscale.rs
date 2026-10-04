//! AV1 **超分上采样**（规范 §7.16）。
//!
//! ## 它解决什么问题
//!
//! 开超分（`use_superres = 1`）时，编码器把一帧**横向压窄**（`FrameWidth` ⇥
//! `UpscaledWidth`）再编 —— 省的是横向的码率，代价是解码端得把这一帧重新拉宽。
//! 这个"拉宽"就是本节：**只有横向**（高度不变），八个抽头的规范滤波器，
//! 而且是**规范化（normative）**的 —— 滤波表、相位推进的全是整数定值，
//! 任何一处实现不同都会立刻体现在像素上，没有"设计自由"。
//!
//! ## 在管线里的位置（别搞错）
//!
//! `去块 → CDEF → 上采样(本节) → 环路恢复`。也就是说**环路恢复作用在拉宽之后的
//! 帧上**（[`super::lr`] 里的 `UpscaledWidth` / `PlaneEndX` 都是拉宽后的坐标）；
//! 而参考帧槽里存的也是拉宽后的 `LrFrame`（§7.20）。所以本节的输出尺寸必须
//! 一律按 `UpscaledWidth` 算，`CurrFrame` 与 `CdefFrame` 两份都要转。
//!
//! ## 规范怎么写的、这里就怎么抄
//!
//! §7.16 给了逐像素的循环：
//!
//! ```text
//! srcX        = -(1<<14) + initialSubpelX + x*stepX
//! srcXPx      = srcX >> 14          （算术右移 = 向下取整）
//! srcXSubpel  = (srcX & (2^14-1)) >> 8
//! sum         = Σ_{k=0..7} frame[plane][y][Clip3(0, maxX, srcXPx + k - 3)] * Upscale_Filter[srcXSubpel][k]
//! out[y][x]   = Clip1(Round2(sum, 7))
//! ```
//!
//! 常量：`SUPERRES_SCALE_BITS = 14`、`SUPERRES_EXTRA_BITS = 8`（=14−6，6 是滤波器
//! 相位精度）、`SUPERRES_FILTER_TAPS = 8`、`SUPERRES_FILTER_OFFSET = 3`、
//! `FILTER_BITS = 7`。滤波表就是规范里那张 `Upscale_Filter[64][8]`，见
//! [`UPSCALE_FILTER`]。
//!
//! ## 边界：规范写 `Clip3(0, maxX, ·)`，这里按**帧边复制**实现
//!
//! 规范把越界样本钳进 `[0, MiCols*MI_SIZE-1]`（`maxX` 是按 4×4 网格向上取整的
//! 帧宽减一）。参考实现 libaom 的做法等价于"把帧左右边界像素复制出去"
//! （`upscale_normative_rect()` 里的 `pad_left` / `pad_right`：首/末瓦片列各复制
//! 5 列），因为 `maxX` 超出真实帧宽的那几列在参考实现里正是被复制的边界像素。
//! 于是这里的钳位写成 `[0, plane.w - 1]` —— 与 libaom 一样，也与逐像素对拍
//! （dav1d）一致。**注意不能用平面的填充区**：`Plane` 的 padding 是中灰，
//! 不是边界像素，读进去会悄悄错。
//!
//! 另外 libaom 是"按瓦片列"分段上采样的（`av1_upscale_normative_rows`），但那只是
//! 为了同步推进各列相位；把 `x0_qn` 的增量累起来，与**整行一次扫完**完全等价
//! —— 所以这里直接整行扫，不需要知道瓦片怎么切。

use me_core::Result;

use super::frame_header::FrameHeader;
use super::intra::{round2, CurrFrame, Plane};
use super::sequence_header::SequenceHeader;

/// `SUPERRES_SCALE_BITS`：位置定点的分数位数。
const SCALE_BITS: u32 = 14;
/// `SUPERRES_SCALE_MASK`。
const SCALE_MASK: i32 = (1 << SCALE_BITS) - 1;
/// `SUPERRES_EXTRA_BITS`：位置定点 → 滤波器相位的多余精度（14 − 6）。
const EXTRA_BITS: u32 = 8;
/// `SUPERRES_FILTER_TAPS`。
const FILTER_TAPS: usize = 8;
/// `SUPERRES_FILTER_OFFSET`：抽头相对中心像素的偏移。
const FILTER_OFFSET: i32 = 3;
/// `FILTER_BITS`：滤波结果的定点位数（滤波表每行和为 `1 << 7`）。
const FILTER_BITS: u32 = 7;

/// 规范 §7.16 的 `Upscale_Filter[SUPERRES_FILTER_SHIFTS][SUPERRES_FILTER_TAPS]`
/// （= libaom `av1_resize_filter_normative[64][8]`，逐行照抄，每行和为 128）。
///
/// 它是**规范化**表：相位 0 行是纯复制（`{0,0,0,128,0,0,0,0}`），其余 63 个相位
/// 是越拉越宽的插值核。改一个数就会与参考解码器分叉，所以不做任何"优化"。
const UPSCALE_FILTER: [[i16; FILTER_TAPS]; 64] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, -1, 128, 2, -1, 0, 0],
    [0, 1, -3, 127, 4, -2, 1, 0],
    [0, 1, -4, 127, 6, -3, 1, 0],
    [0, 2, -6, 126, 8, -3, 1, 0],
    [0, 2, -7, 125, 11, -4, 1, 0],
    [-1, 2, -8, 125, 13, -5, 2, 0],
    [-1, 3, -9, 124, 15, -6, 2, 0],
    [-1, 3, -10, 123, 18, -6, 2, -1],
    [-1, 3, -11, 122, 20, -7, 3, -1],
    [-1, 4, -12, 121, 22, -8, 3, -1],
    [-1, 4, -13, 120, 25, -9, 3, -1],
    [-1, 4, -14, 118, 28, -9, 3, -1],
    [-1, 4, -15, 117, 30, -10, 4, -1],
    [-1, 5, -16, 116, 32, -11, 4, -1],
    [-1, 5, -16, 114, 35, -12, 4, -1],
    [-1, 5, -17, 112, 38, -12, 4, -1],
    [-1, 5, -18, 111, 40, -13, 5, -1],
    [-1, 5, -18, 109, 43, -14, 5, -1],
    [-1, 6, -19, 107, 45, -14, 5, -1],
    [-1, 6, -19, 105, 48, -15, 5, -1],
    [-1, 6, -19, 103, 51, -16, 5, -1],
    [-1, 6, -20, 101, 53, -16, 6, -1],
    [-1, 6, -20, 99, 56, -17, 6, -1],
    [-1, 6, -20, 97, 58, -17, 6, -1],
    [-1, 6, -20, 95, 61, -18, 6, -1],
    [-2, 7, -20, 93, 64, -18, 6, -2],
    [-2, 7, -20, 91, 66, -19, 6, -1],
    [-2, 7, -20, 88, 69, -19, 6, -1],
    [-2, 7, -20, 86, 71, -19, 6, -1],
    [-2, 7, -20, 84, 74, -20, 7, -2],
    [-2, 7, -20, 81, 76, -20, 7, -1],
    [-2, 7, -20, 79, 79, -20, 7, -2],
    [-1, 7, -20, 76, 81, -20, 7, -2],
    [-2, 7, -20, 74, 84, -20, 7, -2],
    [-1, 6, -19, 71, 86, -20, 7, -2],
    [-1, 6, -19, 69, 88, -20, 7, -2],
    [-1, 6, -19, 66, 91, -20, 7, -2],
    [-2, 6, -18, 64, 93, -20, 7, -2],
    [-1, 6, -18, 61, 95, -20, 6, -1],
    [-1, 6, -17, 58, 97, -20, 6, -1],
    [-1, 6, -17, 56, 99, -20, 6, -1],
    [-1, 6, -16, 53, 101, -20, 6, -1],
    [-1, 5, -16, 51, 103, -19, 6, -1],
    [-1, 5, -15, 48, 105, -19, 6, -1],
    [-1, 5, -14, 45, 107, -19, 6, -1],
    [-1, 5, -14, 43, 109, -18, 5, -1],
    [-1, 5, -13, 40, 111, -18, 5, -1],
    [-1, 4, -12, 38, 112, -17, 5, -1],
    [-1, 4, -12, 35, 114, -16, 5, -1],
    [-1, 4, -11, 32, 116, -16, 5, -1],
    [-1, 4, -10, 30, 117, -15, 4, -1],
    [-1, 3, -9, 28, 118, -14, 4, -1],
    [-1, 3, -9, 25, 120, -13, 4, -1],
    [-1, 3, -8, 22, 121, -12, 4, -1],
    [-1, 3, -7, 20, 122, -11, 3, -1],
    [-1, 2, -6, 18, 123, -10, 3, -1],
    [0, 2, -6, 15, 124, -9, 3, -1],
    [0, 2, -5, 13, 125, -8, 2, -1],
    [0, 1, -4, 11, 125, -7, 2, 0],
    [0, 1, -3, 8, 126, -6, 2, 0],
    [0, 1, -3, 6, 127, -4, 1, 0],
    [0, 1, -2, 4, 127, -3, 1, 0],
    [0, 0, -1, 2, 128, -1, 0, 0],
];

/// 把 `FrameWidth` 宽的帧横向放大到 `UpscaledWidth`（§7.16）。
///
/// `use_superres = 0` 时规范就是"原样返回"——这时直接克隆一份，调用方不用分叉。
pub fn apply(frame: &CurrFrame, fh: &FrameHeader, seq: &SequenceHeader) -> Result<CurrFrame> {
    if !fh.use_superres {
        return Ok(frame.clone());
    }
    let sub_x = seq.color.subsampling_x;
    let sub_y = seq.color.subsampling_y;
    // 输出帧按**拉宽后**的宽度开（`CurrFrame::new` 对色度用 `(w+1)>>1`，
    // 正是规范的 `Round2(UpscaledWidth, subX)`）。
    let mut out = CurrFrame::new(
        fh.upscaled_width as usize,
        fh.frame_height as usize,
        sub_x,
        sub_y,
        frame.num_planes,
        seq.color.bit_depth,
    );
    for p in 0..frame.num_planes {
        let (sx, sy) = if p == 0 { (false, false) } else { (sub_x, sub_y) };
        upscale_plane(
            &frame.planes[p],
            &mut out.planes[p],
            sx,
            sy,
            seq.color.bit_depth,
            fh,
        );
    }
    Ok(out)
}

/// 单个平面的横向上采样（§7.16 里 `plane` 那一层循环）。
fn upscale_plane(
    src: &Plane,
    dst: &mut Plane,
    sub_x: bool,
    sub_y: bool,
    bit_depth: u8,
    fh: &FrameHeader,
) {
    let (sub_x, sub_y) = (u32::from(sub_x), u32::from(sub_y));
    // `downscaledPlaneW` / `upscaledPlaneW` / `planeH` —— 都与 `Plane` 的 `w/h`
    // 同口径（`Plane::new` 用的就是同一套取整），可以直接照规范算。
    let down_w = round2(fh.frame_width as i32, sub_x);
    let up_w = round2(fh.upscaled_width as i32, sub_x);
    let plane_h = round2(fh.frame_height as i32, sub_y) as usize;

    // `stepX` / `err` / `initialSubpelX`（§7.16 三连）。全程 i32 —— 参考实现也是
    // `int32_t`，最大值（`down_w << 14`，8K 也才 ~1.3e8）离溢出还远。
    let step_x = ((down_w << SCALE_BITS) + up_w / 2) / up_w;
    let err = up_w * step_x - (down_w << SCALE_BITS);
    // 注意 `& SUPERRES_SCALE_MASK` 作用在**可能为负**的值上：C 与 Rust 的 `&`
    // 都是补码按位与，取到的就是低 14 位 —— 这里必须原样保留这个语义。
    let initial = ((-((up_w - down_w) << (SCALE_BITS - 1)) + up_w / 2) / up_w
        + (1 << (EXTRA_BITS - 1))
        - err / 2)
        & SCALE_MASK;

    // 边界钳到真实平面宽内=按帧边复制（见模块抬头）。`src.w` 就是
    // `Round2(FrameWidth, subX)`，与 `down_w` 相等。
    debug_assert_eq!(src.w as i32, down_w);
    let max_x = src.w as i32 - 1;
    let max_v = (1i32 << bit_depth) - 1;

    for y in 0..plane_h {
        let row = y * src.stride;
        let drow = y * dst.stride;
        // `srcX = -(1<<14) + initialSubpelX + x*stepX`：递推而不是每次重算乘法。
        let mut src_x = -(1 << SCALE_BITS) + initial;
        for x in 0..up_w as usize {
            let px = src_x >> SCALE_BITS;
            let phase = ((src_x & SCALE_MASK) >> EXTRA_BITS) as usize;
            let filt = &UPSCALE_FILTER[phase];
            let mut sum = 0i32;
            for (k, f) in filt.iter().enumerate() {
                // `Clip3(minX, maxX, px + k - SUPERRES_FILTER_OFFSET)`
                let s = (px + k as i32 - FILTER_OFFSET).clamp(0, max_x) as usize;
                sum += src.px16()[row + s] as i32 * *f as i32;
            }
            // `Clip1(Round2(sum, FILTER_BITS))`
            dst.px16_mut()[drow + x] = (((sum + (1 << (FILTER_BITS - 1))) >> FILTER_BITS)
                .clamp(0, max_v)) as u16;
            src_x += step_x;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::sequence_header::SequenceHeader;
    use crate::video::testbits::{write_reduced_still_picture_1080p, BitWriter};

    /// 滤波表是规范定的：每行必须恰好和为 `1 << FILTER_BITS`，否则亮度整体偏移。
    #[test]
    fn filter_rows_sum_to_one() {
        for (i, row) in UPSCALE_FILTER.iter().enumerate() {
            let s: i32 = row.iter().map(|v| *v as i32).sum();
            assert_eq!(s, 1 << FILTER_BITS, "第 {i} 行滤波系数和不是 128");
        }
        // 相位 0 必须是纯复制（不插值）。
        assert_eq!(UPSCALE_FILTER[0], [0, 0, 0, 128, 0, 0, 0, 0]);
    }

    /// 借 `testbits` 造一个真实的 8 位 4:2:0 序列头（走一遍解析器，不手搓结构体）。
    fn seq_420() -> SequenceHeader {
        let mut w = BitWriter::default();
        write_reduced_still_picture_1080p(&mut w);
        let sh = SequenceHeader::parse(w.finish()).expect("序列头应当能解析");
        assert!(sh.color.subsampling_x && sh.color.subsampling_y && sh.color.bit_depth == 8);
        sh
    }

    fn fh_superres(upscaled: u32, denom: u32) -> FrameHeader {
        let mut fh = FrameHeader::default();
        fh.use_superres = true;
        fh.superres_denom = denom;
        fh.upscaled_width = upscaled;
        fh.frame_width = (upscaled * 8 + denom / 2) / denom;
        fh.frame_height = 64;
        fh
    }

    /// `use_superres = 0` 必须原样返回（连尺寸都不动）。
    #[test]
    fn no_superres_is_identity() {
        let sh = seq_420();
        let mut fh = FrameHeader::default();
        fh.use_superres = false;
        fh.frame_width = 32;
        fh.upscaled_width = 32;
        fh.frame_height = 16;
        let f = CurrFrame::new(32, 16, true, true, 3, 8);
        let out = apply(&f, &fh, &sh).unwrap();
        assert_eq!(out.planes[0].w, 32);
        assert_eq!(out.planes[0].px16(), f.planes[0].px16());
    }

    /// **恒定输入 → 恒定输出**：滤波表每行和为 128，所以平坦块放大后必须还是同一个值
    /// （把"和不为 1"这类错误一次抓出来）。同时也验了"输出宽度 = UpscaledWidth"。
    #[test]
    fn flat_input_stays_flat() {
        let sh = seq_420();
        let fh = fh_superres(120, 12); // 120 ⇥ 80 宽
        assert_eq!(fh.frame_width, 80);
        let mut f = CurrFrame::new(80, 64, true, true, 3, 8);
        f.planes[0].px16_mut().iter_mut().for_each(|v| *v = 77);
        f.planes[1].px16_mut().iter_mut().for_each(|v| *v = 200);
        f.planes[2].px16_mut().iter_mut().for_each(|v| *v = 31);
        let out = apply(&f, &fh, &sh).unwrap();
        assert_eq!(out.planes[0].w, 120);
        assert_eq!(out.planes[1].w, 60);
        for y in 0..64 {
            for x in 0..120 {
                assert_eq!(out.planes[0].get(x, y), 77, "亮度 ({x},{y}) 被滤波改动了");
            }
        }
        for y in 0..32 {
            for x in 0..60 {
                assert_eq!(out.planes[1].get(x, y), 200);
                assert_eq!(out.planes[2].get(x, y), 31);
            }
        }
    }

    /// **绝对数值钉死**：8 → 16（分母 16 的恰好 2 倍）时，一行 `[10,20,…,80]`
    /// 的输出必须是 `[9,12,17,23,28,32,37,43,48,53,58,62,67,73,78,81]`。
    ///
    /// 这组数**不是**从本实现跑出来的，而是照 §7.16 的伪码**另写一遍**（用
    /// 截断除法 `tdiv`，与规范的 `/` 语义一致）算出来的 —— 所以它同时钉住了：
    /// 滤波表选择、`stepX`/`initialSubpelX` 的取整方向、边界钳位（前四个抽头
    /// 都被钳到源像素 0，这也是"照着公式算"才容易算对的地方）、以及
    /// `Round2(sum, 7)` 的舍入。
    #[test]
    fn hand_computed_2x_row_matches_the_spec() {
        let sh = seq_420();
        let fh = fh_superres(16, 16); // 16 ⇥ 8 宽
        assert_eq!(fh.frame_width, 8);
        let mut f = CurrFrame::new(8, 4, true, true, 3, 8);
        for x in 0..8 {
            f.planes[0].set(x, 0, (10 * (x + 1)) as u16);
        }
        let out = apply(&f, &fh, &sh).unwrap();
        let got: Vec<u16> = (0..16).map(|x| out.planes[0].get(x, 0)).collect();
        assert_eq!(
            got,
            vec![9, 12, 17, 23, 28, 32, 37, 43, 48, 53, 58, 62, 67, 73, 78, 81]
        );
    }

    /// **中心对齐 + 边界复制**：64 → 128 的竖直硬边（左半 0、右半 255）。
    ///
    /// - 最左/最右必须**恰好**是 0 / 255（边界样本被复制，也是钳位写法的回归保护：
    ///   若误读平面的中灰填充，这两处立刻变 128）；
    /// - 过渡落在**输出 63/64 之间**（规范是中心对齐采样，源的 x=32 边界映射到
    ///   输出的 64）。数值同样是照规范另算的。
    #[test]
    fn vertical_edge_is_center_aligned_and_clamped() {
        let sh = seq_420();
        let fh = fh_superres(128, 16); // 128 ⇥ 64 宽
        assert_eq!(fh.frame_width, 64);
        let mut f = CurrFrame::new(64, 8, true, true, 3, 8);
        for y in 0..8 {
            for x in 0..64 {
                f.planes[0].set(x, y, if x < 32 { 0 } else { 255 });
            }
        }
        let out = apply(&f, &fh, &sh).unwrap();
        for y in 0..8 {
            assert_eq!(out.planes[0].get(0, y), 0, "左边界没有复制源像素 0");
            assert_eq!(out.planes[0].get(127, y), 255, "右边界没有复制源像素 63");
            // 过渡位置（中心对齐）：63 还是暗侧、64 已到亮侧
            assert_eq!(out.planes[0].get(61, y), 0);
            assert_eq!(out.planes[0].get(63, y), 58);
            assert_eq!(out.planes[0].get(64, y), 197);
            assert_eq!(out.planes[0].get(65, y), 255);
        }
    }
}
