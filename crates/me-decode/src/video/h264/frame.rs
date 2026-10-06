//! H.264 软解的像素容器：三个平面 + 裁剪 + 紧凑 NV12 输出。
//!
//! ## 为什么不用 AV1 那套 [`crate::video::intra::Plane`]
//!
//! 两者的**边界语义**根本不同，混用迟早出"错一格"的帧：
//!
//! - AV1 的 `Plane` 带 `PLANE_PAD` 四周留白，且约定"可见区每样本都会被写"、
//!   读邻居读的是已解出的样本。它的留白是为了**分区超块伸出帧外**那一类访问。
//! - H.264 的帧内预测**从不读帧外**（规范 §8.3 对"不可用邻居"有专门的替代规则），
//!   而帧间预测读帧外样本时**每一处都要按 §8.4.2.2.1 夹回画面内**
//!   —— 夹的是**样本坐标**，不是"整个块"。留白反而会掩盖"夹错了"这件事。
//!
//! 所以这里的平面是**紧的**（`stride == w`，无留白），越界一律显式处理：
//! 帧内走"不可用邻居"规则，帧间走 [`Plane::get_clamped`]，去块走 §8.7 的边界条款。
//!
//! ## 8 位 4:2:0 是**写死的**
//!
//! 与 [`crate::h264`] 的支持口径一致（高 profile 的 10 位、4:2:2/4:4:4、
//! 独立色彩平面一律在 `check_supported` 就被拒）。存储用 `u8`，
//! 对外是引擎统一的**紧凑 NV12**（Y 平面 + CbCr 交织平面）。

use crate::h264::Sps;

/// 单平面样本缓冲（8 位，行优先，`stride == w`）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plane {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl Plane {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            data: vec![0u8; w * h],
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> u8 {
        debug_assert!(x < self.w && y < self.h, "平面坐标越界 ({x},{y}) / {}×{}", self.w, self.h);
        self.data[y * self.w + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: u8) {
        debug_assert!(x < self.w && y < self.h, "平面坐标越界 ({x},{y}) / {}×{}", self.w, self.h);
        self.data[y * self.w + x] = v;
    }

    /// 行切片（整行，含 `stride` 宽度）。
    #[inline]
    pub fn row(&self, y: usize) -> &[u8] {
        &self.data[y * self.w..y * self.w + self.w]
    }

    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [u8] {
        let w = self.w;
        &mut self.data[y * w..y * w + w]
    }

    /// **帧间预测专用的取样本**：坐标按 §8.4.2.2.1 夹回画面内。
    ///
    /// 规范原文的意思是"画面外的样本由画面内最近的样本代替"，等价于
    /// 对 `x`/`y` 各自夹到 `[0, w-1]` / `[0, h-1]`。
    #[inline]
    pub fn get_clamped(&self, x: i32, y: i32) -> u8 {
        let cx = x.clamp(0, self.w as i32 - 1) as usize;
        let cy = y.clamp(0, self.h as i32 - 1) as usize;
        self.data[cy * self.w + cx]
    }

    /// 整面填一个值（参考帧边界扩展、诊断用）。
    pub fn fill(&mut self, v: u8) {
        self.data.fill(v);
    }
}

/// 裁剪量（**样本**，不是 `frame_cropping` 里的原始单位）。
///
/// `frame_cropping` 的单位是"色度样本 × (2 - frame_mbs_only_flag)"，
/// 而引擎只支持逐行帧、4:2:0，所以单位就是色度样本：亮度要再乘 2。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Crop {
    pub left: u32,
    pub right: u32,
    pub top: u32,
    pub bottom: u32,
}

impl Crop {
    /// 从 SPS 的 `frame_cropping` 换算成亮度样本数。
    pub fn from_sps(sps: &Sps) -> Self {
        if !sps.frame_cropping_flag {
            return Self::default();
        }
        // 4:2:0 且逐行帧 ⇒ SubWidthC = SubHeightC = 2（`crate::h264::Sps::crop_unit`）
        Self {
            left: sps.crop_left * 2,
            right: sps.crop_right * 2,
            top: sps.crop_top * 2,
            bottom: sps.crop_bottom * 2,
        }
    }
}

/// 一张解码中的/已解码的图：三平面 + 尺寸 + 裁剪。
///
/// 尺寸有两套：`coded_*` 是**宏块对齐**的（内部一切寻址都用它，也是 §8.7 去块
/// 遍历的范围），`disp_*` 是**裁剪后**的（对外帧的大小）。360p 这类高度不是 16
/// 的整数倍的素材，两者差的就是裁剪掉的那几行 —— 输出时按 `disp_*` 取。
#[derive(Debug, Clone)]
pub struct FrameBuf {
    pub y: Plane,
    pub u: Plane,
    pub v: Plane,
    pub coded_w: usize,
    pub coded_h: usize,
    pub disp_w: usize,
    pub disp_h: usize,
    pub crop: Crop,
}

impl FrameBuf {
    /// 按 SPS 的**编码尺寸**开一张空图（同时算好显示尺寸与裁剪起点）。
    pub fn new(coded_w: usize, coded_h: usize, crop: Crop) -> Self {
        let cw = coded_w.div_ceil(2);
        let ch = coded_h.div_ceil(2);
        let disp_w = coded_w.saturating_sub((crop.left + crop.right) as usize).max(1);
        let disp_h = coded_h.saturating_sub((crop.top + crop.bottom) as usize).max(1);
        Self {
            y: Plane::new(coded_w, coded_h),
            u: Plane::new(cw, ch),
            v: Plane::new(cw, ch),
            coded_w,
            coded_h,
            disp_w,
            disp_h,
            crop,
        }
    }

    /// 从 SPS 开一张空图。
    pub fn for_sps(sps: &Sps) -> Self {
        Self::new(
            sps.coded_width() as usize,
            sps.coded_height() as usize,
            Crop::from_sps(sps),
        )
    }

    /// 宏块列数 / 行数（`PicWidthInMbs`、`FrameHeightInMbs`）。
    #[inline]
    pub fn mb_cols(&self) -> usize {
        self.coded_w / 16
    }

    #[inline]
    pub fn mb_rows(&self) -> usize {
        self.coded_h / 16
    }

    /// 4×4 亮度块列数 / 行数（去块与运动信息网格的尺寸）。
    #[inline]
    pub fn luma4_cols(&self) -> usize {
        self.coded_w / 4
    }

    #[inline]
    pub fn luma4_rows(&self) -> usize {
        self.coded_h / 4
    }

    /// 紧凑 NV12：`Y(disp_w×disp_h)` + `CbCr 交织((disp_w/2)×(disp_h/2))`。
    ///
    /// 与 [`crate::VideoFrame::expected_len`] 的口径一致：NV12 的第二个平面是
    /// **CbCr 交织**、宽度 = `disp_w/2 * 2`，所以长度就是 `w*h*3/2`。
    ///
    /// 宽高都取偶数（显示尺寸在 4:2:0 下必然是偶数：裁剪单位本来就是色度样本），
    /// 所以不会出现"半个色度样本"的取整问题。
    pub fn to_nv12(&self) -> Vec<u8> {
        let w = self.disp_w;
        let h = self.disp_h;
        let mut out = Vec::with_capacity(w * h * 3 / 2);
        let x0 = self.crop.left as usize;
        let y0 = self.crop.top as usize;
        for row in 0..h {
            let src = self.y.row(y0 + row);
            out.extend_from_slice(&src[x0..x0 + w]);
        }
        // CbCr 交织平面：Cb 在偶字节、Cr 在奇字节（NV12 = Y + UV）
        let cw = w / 2;
        let cx0 = x0 / 2;
        let cy0 = y0 / 2;
        for row in 0..h / 2 {
            let u = self.u.row(cy0 + row);
            let v = self.v.row(cy0 + row);
            for col in 0..cw {
                out.push(u[cx0 + col]);
                out.push(v[cx0 + col]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h264::tests_support::Bits;

    fn sps_of(width_mbs: u32, height_map_units: u32, crop_bottom: u32) -> Sps {
        let mut w = Bits::default();
        w.f(0x67, 8);
        w.f(77, 8);
        w.f(0, 8);
        w.f(30, 8);
        w.ue(0);
        w.ue(0); // log2_max_frame_num_minus4
        w.ue(0); // poc type 0
        w.ue(0); // log2_max_pic_order_cnt_lsb_minus4
        w.ue(1); // max_num_ref_frames
        w.bit(false);
        w.ue(width_mbs - 1);
        w.ue(height_map_units - 1);
        w.bit(true); // frame_mbs_only
        w.bit(true); // direct_8x8
        if crop_bottom > 0 {
            w.bit(true); // frame_cropping_flag
            w.ue(0);
            w.ue(0);
            w.ue(0);
            w.ue(crop_bottom);
        } else {
            w.bit(false);
        }
        w.bit(false); // vui
        w.f(0b1000_0000, 8);
        Sps::parse(&w.finish()).unwrap()
    }

    #[test]
    fn coded_size_is_macroblock_aligned() {
        let sps = sps_of(40, 23, 0);
        let f = FrameBuf::for_sps(&sps);
        assert_eq!((f.coded_w, f.coded_h), (640, 368));
        assert_eq!((f.disp_w, f.disp_h), (640, 368));
        assert_eq!(f.mb_cols(), 40);
        assert_eq!(f.mb_rows(), 23);
        assert_eq!(f.luma4_cols(), 160);
        assert_eq!(f.u.w, 320);
        assert_eq!(f.v.h, 184);
    }

    /// 裁剪按**色度样本**给，亮度要 ×2 —— 这条错了画面会整体错位 8 行。
    #[test]
    fn cropping_converts_to_luma_samples() {
        let sps = sps_of(40, 23, 4); // bottom = 4 个色度样本 = 8 行
        let f = FrameBuf::for_sps(&sps);
        assert_eq!(f.crop.bottom, 8);
        assert_eq!(f.coded_h, 368);
        assert_eq!(f.disp_h, 360, "368 - 8 = 360");
    }

    /// 平面是**紧的**：拿越界坐标直接取会 panic（debug 下），
    /// 帧间那条路必须走 `get_clamped`。
    #[test]
    fn clamped_fetch_replicates_the_edge() {
        let mut p = Plane::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                p.set(x, y, (y * 4 + x) as u8);
            }
        }
        assert_eq!(p.get_clamped(-5, 0), 0, "左边越界取最左");
        assert_eq!(p.get_clamped(9, 0), 3, "右边越界取最右");
        assert_eq!(p.get_clamped(0, -1), 0, "上边越界取最上");
        assert_eq!(p.get_clamped(0, 99), 12, "下边越界取最下");
        assert_eq!(p.get_clamped(-3, 100), 12, "两轴同时越界");
    }

    /// NV12 输出的**布局**：Y 之后是 CbCr 交织，长度正好 `w*h*3/2`。
    #[test]
    fn nv12_layout_is_y_then_interleaved_chroma() {
        let mut f = FrameBuf::new(16, 16, Crop::default());
        for y in 0..16 {
            for x in 0..16 {
                f.y.set(x, y, 100);
            }
        }
        for y in 0..8 {
            for x in 0..8 {
                f.u.set(x, y, 20);
                f.v.set(x, y, 30);
            }
        }
        let nv12 = f.to_nv12();
        assert_eq!(nv12.len(), 16 * 16 * 3 / 2);
        assert!(nv12[..256].iter().all(|&v| v == 100), "前 256 字节全是 Y");
        assert_eq!(&nv12[256..260], &[20, 30, 20, 30], "CbCr 交织：偶 Cb 奇 Cr");
    }

    /// 裁剪之后 NV12 取的是**裁剪窗口内的像素**，不是左上角那块。
    #[test]
    fn nv12_respects_the_crop_window() {
        let crop = Crop {
            left: 4,
            right: 0,
            top: 2,
            bottom: 0,
        };
        let mut f = FrameBuf::new(16, 16, crop);
        for y in 0..16usize {
            for x in 0..16usize {
                f.y.set(x, y, (y * 16 + x) as u8);
            }
        }
        let nv12 = f.to_nv12();
        assert_eq!(f.disp_w, 12);
        assert_eq!(f.disp_h, 14);
        assert_eq!(nv12.len(), 12 * 14 * 3 / 2, "NV12 = Y + CbCr 交织，长度是 w*h*3/2");
        // 第一行第一样本 = 原图 (4, 2) = 2*16+4 = 36
        assert_eq!(nv12[0], 36);
        // 第二行第一样本 = (4, 3) = 52
        assert_eq!(nv12[12], 52);
    }
}
