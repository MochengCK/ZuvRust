//! **把宏块变成像素**：取参考样本 → 预测 → 加残差 → 写进画面。
//!
//! 这一层把 [`super::mb`]（语法）、[`super::transform`]（残差）、
//! [`super::intra`]（预测）串起来，产出 §8.5.14 说的
//! "S′（去块滤波之前的构造样本）"。
//!
//! ## 可用性：**宏块级 + 块级**，两级都要看
//!
//! §6.4.8 说"宏块可用 ⟺ 地址 ≤ 当前宏块地址"（等于它在光栅序里已经解出来了）。
//! 但 `Intra_4x4` 还有一层**块级**的：§8.3.1.2 专门写了一条
//! "`x > 3` 且 `luma4x4BlkIdx == 3 / 11` 一律不可用"。
//!
//! 为什么要有这条 —— 因为 luma4x4 的块扫描序是**象限序**
//! （图 6-10：`0 1 4 5 / 2 3 6 7 / 8 9 12 13 / 10 11 14 15`），
//! 块 3（在 `(4,4)`）右上角那 4 个样本属于块 4（在 `(8,0)`），
//! **而块 4 还没解出来**。同理块 11 的右上角属于块 12。
//!
//! 所以这里把判据写全：**宏块在光栅序里已解出** + **同宏块内该 4×4 块在扫描序里已解出**。
//! 只按宏块判，这两处会读到还没重建的像素（画面局部花掉，而且不报错）。
//!
//! ## 块坐标的三个映射（别混）
//!
//! | 索引 | 公式 |
//! |---|---|
//! | `luma4x4BlkIdx` → 像素 `(x,y)` | `x = 8*(idx/4%2) + 4*(idx%4%2)`，`y = 8*(idx/4/2) + 4*(idx%4/2)` |
//! | 像素 `(x,y)` → `luma4x4BlkIdx` | 上面的逆（见 [`blk4_of`]）|
//! | `luma4x4BlkIdx` → `dcY` 下标 | `dcY` 按**光栅序**排块 ⇒ 见 [`dc_y_index`] |
//! | `luma8x8BlkIdx` → `(x,y)` | 光栅序：`x = 8*(idx%2)`，`y = 8*(idx/2)` |

use me_core::{MediaError, Result};

use super::frame::FrameBuf;
use super::intra::{self, Refs4, Refs8};
use super::mb::{IntraPred, MbCoeffs, MbInfo, PictureInfo};
use super::transform::{self, WeightScale};
use crate::h264::{Pps, Sps};

/// `luma4x4BlkIdx` → 宏块内的像素偏移（图 6-10 的象限序）。
#[inline]
pub fn blk4_xy(idx: usize) -> (usize, usize) {
    let q = idx / 4;
    let s = idx % 4;
    let gx = (q % 2) * 2 + (s % 2);
    let gy = (q / 2) * 2 + (s / 2);
    (gx * 4, gy * 4)
}

/// 宏块内像素 `(x,y)` → 它属于哪个 `luma4x4BlkIdx`（[`blk4_xy`] 的逆）。
#[inline]
pub fn blk4_of(x: i32, y: i32) -> usize {
    let (gx, gy) = ((x / 4) as usize, (y / 4) as usize);
    ((gy / 2) * 2 + (gx / 2)) * 4 + (gy % 2) * 2 + (gx % 2)
}

/// `luma4x4BlkIdx` → 它在 `dcY` 数组里的下标。
///
/// `dcY` 是**空间上按光栅序**排的 4×4 块网格（反 DC 变换的输出就是这么定义的），
/// 而 `luma4x4BlkIdx` 是象限序 —— 图 8-6 那张图专门画的就是这个错位。
#[inline]
pub fn dc_y_index(idx: usize) -> usize {
    let (x, y) = blk4_xy(idx);
    (y / 4) * 4 + (x / 4)
}

/// `luma8x8BlkIdx` → 宏块内的像素偏移（图 6-11，光栅序）。
#[inline]
pub fn blk8_xy(idx: usize) -> (usize, usize) {
    ((idx % 2) * 8, (idx / 2) * 8)
}

/// 宏块内像素 `(x,y)` → `luma8x8BlkIdx`。
#[inline]
pub fn blk8_of(x: i32, y: i32) -> usize {
    (y / 8) as usize * 2 + (x / 8) as usize
}

/// 重建一张图（按光栅序走宏块 —— 与单切片码流的解码序一致）。
pub fn reconstruct_picture(
    frame: &mut FrameBuf,
    info: &mut PictureInfo,
    sps: &Sps,
    pps: &Pps,
) -> Result<()> {
    let mut r = Recon {
        frame,
        info,
        ws: WeightScale::new(sps, pps),
        constrained: pps.constrained_intra_pred_flag,
        chroma_off: [pps.chroma_qp_index_offset, pps.second_chroma_qp_index_offset],
        modes4: vec![[0u8; 16]; sps.coded_width() as usize / 16 * (sps.coded_height() as usize / 16)],
        modes8: vec![[0u8; 4]; sps.coded_width() as usize / 16 * (sps.coded_height() as usize / 16)],
        mb_done: Vec::new(),
    };
    r.mb_done = vec![false; r.info.mb.len()];
    for addr in 0..r.info.mb.len() {
        r.reconstruct_mb(addr, sps, pps)?;
        r.mb_done[addr] = true;
    }
    Ok(())
}

struct Recon<'a> {
    frame: &'a mut FrameBuf,
    info: &'a mut PictureInfo,
    ws: WeightScale,
    constrained: bool,
    /// `chroma_qp_index_offset` / `second_chroma_qp_index_offset`（PPS 给的色度 QP 偏移）
    chroma_off: [i32; 2],
    modes4: Vec<[u8; 16]>,
    modes8: Vec<[u8; 4]>,
    mb_done: Vec<bool>,
}

impl Recon<'_> {
    fn mb_cols(&self) -> usize {
        self.info.mb_cols
    }

    /// 一个像素能不能当参考样本用（宏块级 + 块级可用性，见模块抬头）。
    fn avail_luma(&self, x: i32, y: i32, cur: usize, cur_blk4: usize) -> bool {
        let (w, h) = (self.frame.coded_w as i32, self.frame.coded_h as i32);
        if x < 0 || y < 0 || x >= w || y >= h {
            return false;
        }
        let n = (y as usize / 16) * self.mb_cols() + (x as usize / 16);
        if n > cur {
            return false;
        }
        if n == cur {
            // 同一个宏块里，只有**扫描序在前**的 4×4 块已经解出来
            if blk4_of(x % 16, y % 16) >= cur_blk4 {
                return false;
            }
        }
        if self.constrained && !self.mb_is_intra(n) {
            return false;
        }
        true
    }

    fn avail_luma8(&self, x: i32, y: i32, cur: usize, cur_blk8: usize) -> bool {
        let (w, h) = (self.frame.coded_w as i32, self.frame.coded_h as i32);
        if x < 0 || y < 0 || x >= w || y >= h {
            return false;
        }
        let n = (y as usize / 16) * self.mb_cols() + (x as usize / 16);
        if n > cur {
            return false;
        }
        if n == cur && blk8_of(x % 16, y % 16) > cur_blk8 {
            return false;
        }
        if self.constrained && !self.mb_is_intra(n) {
            return false;
        }
        true
    }

    fn avail_chroma(&self, x: i32, y: i32, cur: usize) -> bool {
        let (w, h) = ((self.frame.coded_w / 2) as i32, (self.frame.coded_h / 2) as i32);
        if x < 0 || y < 0 || x >= w || y >= h {
            return false;
        }
        let n = (y as usize / 8) * self.mb_cols() + (x as usize / 8);
        if n > cur {
            return false;
        }
        if self.constrained && n != cur && !self.mb_is_intra(n) {
            return false;
        }
        true
    }

    fn mb_is_intra(&self, addr: usize) -> bool {
        // 现在只有 I 切片（全是帧内）；帧间接上后要按 `intra_pred` 判
        self.info.mb.get(addr).is_some_and(|m| m.mb_type != u32::MAX)
    }

    fn reconstruct_mb(&mut self, addr: usize, sps: &Sps, pps: &Pps) -> Result<()> {
        let mb = self.info.mb[addr].clone();
        let mbx = addr % self.mb_cols();
        let mby = addr / self.mb_cols();
        let (x0, y0) = (mbx * 16, mby * 16);

        if mb.is_pcm {
            let pcm = &self.info.coeffs[addr].pcm;
            if pcm.len() < 256 + 128 {
                return Err(MediaError::Bitstream("I_PCM 宏块没留下样本".into()));
            }
            for (i, &v) in pcm[..256].iter().enumerate() {
                self.frame.y.set(x0 + i % 16, y0 + i / 16, v);
            }
            let (cw, ch) = (mbx * 8, mby * 8);
            for i in 0..64 {
                self.frame.u.set(cw + i % 8, ch + i / 8, pcm[256 + i]);
                self.frame.v.set(cw + i % 8, ch + i / 8, pcm[256 + 64 + i]);
            }
            return Ok(());
        }

        match mb.intra_pred {
            IntraPred::I16x16 => self.mb_intra16x16(addr, &mb, x0, y0)?,
            IntraPred::NxN4x4 => self.mb_intra4x4(addr, &mb, x0, y0)?,
            IntraPred::NxN8x8 => self.mb_intra8x8(addr, &mb, x0, y0)?,
            IntraPred::Pcm => unreachable!("上面已经处理"),
        }
        let _ = pps;
        self.mb_chroma(addr, &mb, sps, mbx, mby)?;
        Ok(())
    }

    // ── Intra_16x16 ────────────────────────────────────────────────────────

    fn mb_intra16x16(&mut self, addr: usize, mb: &MbInfo, x0: usize, y0: usize) -> Result<()> {
        let mut top = [0i32; 16];
        let mut left = [0i32; 16];
        let mut top_ok = true;
        let mut left_ok = true;
        for i in 0..16 {
            let ta = self.avail_luma(x0 as i32 + i as i32, y0 as i32 - 1, addr, 0);
            top[i] = if ta { self.frame.y.get(x0 + i, y0 - 1) as i32 } else { 0 };
            top_ok &= ta;
            let la = self.avail_luma(x0 as i32 - 1, y0 as i32 + i as i32, addr, 0);
            left[i] = if la { self.frame.y.get(x0 - 1, y0 + i) as i32 } else { 0 };
            left_ok &= la;
        }
        let tl_ok = self.avail_luma(x0 as i32 - 1, y0 as i32 - 1, addr, 0);
        let tl = if tl_ok { self.frame.y.get(x0 - 1, y0 - 1) as i32 } else { 0 };
        let pred = intra::predict_16x16(
            mb.intra16x16_pred_mode,
            &top,
            &left,
            tl,
            top_ok,
            left_ok,
            8,
        );

        let coeffs = self.info.coeffs[addr].clone();
        let qp = mb.qp_y;
        // 16 个 DC：反扫描 → Hadamard → 缩放
        let dc_c = transform::scan_4x4(&coeffs.luma_dc, false);
        let dcy = transform::intra16x16_dc(&dc_c, qp);
        let mut rmb = [0i32; 256];
        for blk in 0..16 {
            let ac: [i32; 15] = coeffs.luma4[blk][..15].try_into().unwrap();
            let mut c = transform::scan_4x4(&ac, true);
            // 式 8-335：`d00 = c00`，而这里放的正是**已经缩放过的** DC
            c[0] = dcy[dc_y_index(blk)];
            let d = transform::dequant_4x4(&c, qp, &self.ws, 0, true);
            let r = transform::inverse_4x4(&d);
            let (bx, by) = blk4_xy(blk);
            for i in 0..4 {
                for j in 0..4 {
                    rmb[(by + i) * 16 + bx + j] = r[i * 4 + j];
                }
            }
        }
        for y in 0..16 {
            for x in 0..16 {
                let v = pred[y * 16 + x] + rmb[y * 16 + x];
                self.frame.y.set(x0 + x, y0 + y, clip8(v));
            }
        }
        Ok(())
    }

    // ── Intra_4x4 ─────────────────────────────────────────────────────────

    fn mb_intra4x4(&mut self, addr: usize, mb: &MbInfo, x0: usize, y0: usize) -> Result<()> {
        let coeffs = self.info.coeffs[addr].clone();
        let qp = mb.qp_y;
        for blk in 0..16usize {
            let mode = self.derive_mode4(addr, blk, mb);
            self.modes4[addr][blk] = mode;
            let (bx, by) = blk4_xy(blk);
            let (px, py) = ((x0 + bx) as i32, (y0 + by) as i32);
            let r = self.refs4(addr, blk, px, py);
            let pred = intra::predict_4x4(mode, &r, 8);
            let c = transform::scan_4x4(&coeffs.luma4[blk], false);
            let d = transform::dequant_4x4(&c, qp, &self.ws, 0, false);
            let res = transform::inverse_4x4(&d);
            for i in 0..4usize {
                for j in 0..4usize {
                    let v = pred[i * 4 + j] + res[i * 4 + j];
                    self.frame.y.set(x0 + bx + j, y0 + by + i, clip8(v));
                }
            }
        }
        Ok(())
    }

    fn refs4(&self, addr: usize, blk: usize, px: i32, py: i32) -> Refs4 {
        let mut r = Refs4 {
            top: [0; 8],
            left: [0; 4],
            top_left: 0,
            top_avail: [false; 8],
            left_avail: [false; 4],
            top_left_avail: false,
        };
        for x in 0..8i32 {
            r.top_avail[x as usize] = self.avail_luma(px + x, py - 1, addr, blk);
            if r.top_avail[x as usize] {
                r.top[x as usize] = self.frame.y.get((px + x) as usize, (py - 1) as usize) as i32;
            }
        }
        for y in 0..4i32 {
            r.left_avail[y as usize] = self.avail_luma(px - 1, py + y, addr, blk);
            if r.left_avail[y as usize] {
                r.left[y as usize] = self.frame.y.get((px - 1) as usize, (py + y) as usize) as i32;
            }
        }
        r.top_left_avail = self.avail_luma(px - 1, py - 1, addr, blk);
        if r.top_left_avail {
            r.top_left = self.frame.y.get((px - 1) as usize, (py - 1) as usize) as i32;
        }
        // §8.3.1.2：右上角不可用而 p[3,-1] 可用时，用 p[3,-1] 顶替
        if r.top_avail[3] && !r.top_avail[4..].iter().all(|&a| a) {
            for x in 4..8 {
                if !r.top_avail[x] {
                    r.top[x] = if r.top_avail[3] { r.top[3] } else { 0 };
                    r.top_avail[x] = true;
                }
            }
        }
        r
    }

    /// §8.3.1.1：`Intra4x4PredMode` 要从邻居推。
    fn derive_mode4(&self, addr: usize, blk: usize, mb: &MbInfo) -> u8 {
        let (a_addr, a_blk) = self.neighbour4(addr, blk, true);
        let (b_addr, b_blk) = self.neighbour4(addr, blk, false);
        let dc_flag = a_addr.map(|n| n >= self.info.mb.len()).unwrap_or(true)
            || b_addr.map(|n| n >= self.info.mb.len()).unwrap_or(true)
            || a_addr.map(|n| self.constrained && !self.uses_4x4(n)).unwrap_or(false)
            || b_addr.map(|n| self.constrained && !self.uses_4x4(n)).unwrap_or(false);
        let get = |a: Option<usize>, b: Option<usize>| -> u8 {
            match a {
                Some(n) if !dc_flag && self.uses_4x4(n) => self.modes4[n][b.unwrap_or(0)],
                Some(n) if !dc_flag && self.mb_uses_8x8(n) => self.modes8[n][b.unwrap_or(0) >> 2],
                _ => 2,
            }
        };
        let ma = get(a_addr, a_blk);
        let mbb = get(b_addr, b_blk);
        let predv = ma.min(mbb);
        if mb.intra4x4_prev[blk] {
            predv
        } else {
            let rem = mb.intra4x4_rem[blk];
            if rem < predv {
                rem
            } else {
                rem + 1
            }
        }
    }

    fn uses_4x4(&self, addr: usize) -> bool {
        self.info.mb.get(addr).is_some_and(|m| m.intra_pred == IntraPred::NxN4x4)
    }

    fn mb_uses_8x8(&self, addr: usize) -> bool {
        self.info.mb.get(addr).is_some_and(|m| m.intra_pred == IntraPred::NxN8x8)
    }

    /// §6.4.11.4：`luma4x4BlkIdx` 的左右（A）/ 上下（B）邻居。
    ///
    /// 规范的做法是**按像素位置走**：把 `luma4x4BlkIdx` 反查成块坐标 `(x, y)`，
    /// 加上 `(xD, yD) = (−1, 0)` 或 `(0, −1)`，再反查"那个像素属于哪个宏块的哪个块"。
    /// 等价于：
    ///
    /// - A：`x > 0` → 同宏块的 `blk4_of(x−4, y)`；否则左宏块的 `blk4_of(12, y)`
    /// - B：`y > 0` → 同宏块的 `blk4_of(x, y−4)`；否则上宏块的 `blk4_of(x, 12)`
    ///
    /// ⚠️ **不能按索引硬算**。`luma4x4BlkIdx` 是象限序（图 6-10），
    /// "减 1 就是左边的块"只在每个象限内部成立：`blk = 1` 的左邻是 `blk 0`，
    /// 但 `blk = 4` 的左邻是**左宏块的 `blk 5`**、`blk = 4` 的上邻是**上宏块的
    /// `blk 14`**（因为 `blk 4` 在 `(8, 0)`，正上方已经没有本宏块的行了）。
    /// 按索引硬算会让 `Intra4x4PredMode` 的预测值歪掉 —— 而**位流长度不变**，
    /// 所以 `rbsp_trailing_bits` 那条判据抓不到，只有逐像素对拍能抓。
    fn neighbour4(&self, addr: usize, blk: usize, left: bool) -> (Option<usize>, Option<usize>) {
        let mbx = addr % self.mb_cols();
        let mby = addr / self.mb_cols();
        let (x, y) = blk4_xy(blk); // 像素偏移（宏块内）
        if left {
            if x >= 4 {
                (Some(addr), Some(blk4_of(x as i32 - 4, y as i32)))
            } else if mbx == 0 {
                (None, None)
            } else {
                (Some(addr - 1), Some(blk4_of(12, y as i32)))
            }
        } else if y >= 4 {
            (Some(addr), Some(blk4_of(x as i32, y as i32 - 4)))
        } else if mby == 0 {
            (None, None)
        } else {
            (Some(addr - self.mb_cols()), Some(blk4_of(x as i32, 12)))
        }
    }

    // ── Intra_8x8 ─────────────────────────────────────────────────────────

    fn mb_intra8x8(&mut self, addr: usize, mb: &MbInfo, x0: usize, y0: usize) -> Result<()> {
        let coeffs = self.info.coeffs[addr].clone();
        let qp = mb.qp_y;
        for blk in 0..4usize {
            let mode = self.derive_mode8(addr, blk, mb);
            self.modes8[addr][blk] = mode;
            let (bx, by) = blk8_xy(blk);
            let (px, py) = ((x0 + bx) as i32, (y0 + by) as i32);
            let r = self.refs8(addr, blk, px, py);
            let pred = intra::predict_8x8(mode, &r, 8);
            let c = transform::scan_8x8(&coeffs.luma8[blk]);
            let d = transform::dequant_8x8(&c, qp, &self.ws, 0);
            let res = transform::inverse_8x8(&d);
            for i in 0..8usize {
                for j in 0..8usize {
                    let v = pred[i * 8 + j] + res[i * 8 + j];
                    self.frame.y.set(x0 + bx + j, y0 + by + i, clip8(v));
                }
            }
        }
        Ok(())
    }

    fn refs8(&self, addr: usize, blk: usize, px: i32, py: i32) -> Refs8 {
        let mut r = Refs8 {
            top: [0; 16],
            left: [0; 8],
            top_left: 0,
            top_avail: [false; 16],
            left_avail: [false; 8],
            top_left_avail: false,
        };
        for x in 0..16i32 {
            r.top_avail[x as usize] = self.avail_luma8(px + x, py - 1, addr, blk);
            if r.top_avail[x as usize] {
                r.top[x as usize] = self.frame.y.get((px + x) as usize, (py - 1) as usize) as i32;
            }
        }
        for y in 0..8i32 {
            r.left_avail[y as usize] = self.avail_luma8(px - 1, py + y, addr, blk);
            if r.left_avail[y as usize] {
                r.left[y as usize] = self.frame.y.get((px - 1) as usize, (py + y) as usize) as i32;
            }
        }
        r.top_left_avail = self.avail_luma8(px - 1, py - 1, addr, blk);
        if r.top_left_avail {
            r.top_left = self.frame.y.get((px - 1) as usize, (py - 1) as usize) as i32;
        }
        if r.top_avail[7] && !r.top_avail[8..].iter().all(|&a| a) {
            for x in 8..16 {
                if !r.top_avail[x] {
                    r.top[x] = r.top[7];
                    r.top_avail[x] = true;
                }
            }
        }
        intra::filter_refs8(&r)
    }

    /// §8.3.2.1：`Intra8x8PredMode` 的推导。
    fn derive_mode8(&self, addr: usize, blk: usize, mb: &MbInfo) -> u8 {
        let (a_addr, a_blk) = self.neighbour8(addr, blk, true);
        let (b_addr, b_blk) = self.neighbour8(addr, blk, false);
        let dc_flag = a_addr.is_none()
            || b_addr.is_none()
            || a_addr.is_some_and(|n| self.constrained && !self.is_mxm(n))
            || b_addr.is_some_and(|n| self.constrained && !self.is_mxm(n));
        // 式 8-72：邻居是 Intra_4x4 时取 `luma8x8BlkIdxN * 4 + n`（A 侧 n=1、B 侧 n=2）
        let get = |a: Option<usize>, b: Option<usize>, n: usize| -> u8 {
            match a {
                Some(m) if !dc_flag && self.mb_uses_8x8(m) => self.modes8[m][b.unwrap_or(0)],
                Some(m) if !dc_flag && self.uses_4x4(m) => {
                    self.modes4[m][b.unwrap_or(0) * 4 + n]
                }
                _ => 2,
            }
        };
        let ma = get(a_addr, a_blk, 1);
        let mbb = get(b_addr, b_blk, 2);
        let predv = ma.min(mbb);
        if mb.intra8x8_prev[blk] {
            predv
        } else {
            let rem = mb.intra8x8_rem[blk];
            if rem < predv {
                rem
            } else {
                rem + 1
            }
        }
    }

    fn is_mxm(&self, addr: usize) -> bool {
        self.uses_4x4(addr) || self.mb_uses_8x8(addr)
    }

    /// §6.4.11.2：`luma8x8BlkIdx` 的左右/上下邻居。
    fn neighbour8(&self, addr: usize, blk: usize, left: bool) -> (Option<usize>, Option<usize>) {
        let mbx = addr % self.mb_cols();
        let mby = addr / self.mb_cols();
        if left {
            if blk.is_multiple_of(2) {
                if mbx == 0 {
                    (None, None)
                } else {
                    (Some(addr - 1), Some(blk + 1))
                }
            } else {
                (Some(addr), Some(blk - 1))
            }
        } else if blk < 2 {
            if mby == 0 {
                (None, None)
            } else {
                (Some(addr - self.mb_cols()), Some(blk + 2))
            }
        } else {
            (Some(addr), Some(blk - 2))
        }
    }

    // ── 色度 ───────────────────────────────────────────────────────────────

    fn mb_chroma(
        &mut self,
        addr: usize,
        mb: &MbInfo,
        _sps: &Sps,
        mbx: usize,
        mby: usize,
    ) -> Result<()> {
        let (cx, cy) = ((mbx * 8) as i32, (mby * 8) as i32);
        // ⚠️ **两个分量要各自取自己的参考样本**。之前这里只读 U 平面，
        // 于是 Cr 的预测是按 Cb 的邻居算的 —— 而所有"R=G=B"的测试内容
        // 恰好 Cb ≡ Cr，把这个错误完全遮住了（色度"全对"其实是对了一半）。
        let mut refs = [([0i32; 8], [0i32; 8]); 2];
        let mut top_avail = [false; 8];
        let mut left_avail = [false; 8];
        for i in 0..8i32 {
            top_avail[i as usize] = self.avail_chroma(cx + i, cy - 1, addr);
            left_avail[i as usize] = self.avail_chroma(cx - 1, cy + i, addr);
            if top_avail[i as usize] {
                refs[0].0[i as usize] =
                    self.frame.u.get((cx + i) as usize, (cy - 1) as usize) as i32;
                refs[1].0[i as usize] =
                    self.frame.v.get((cx + i) as usize, (cy - 1) as usize) as i32;
            }
            if left_avail[i as usize] {
                refs[0].1[i as usize] =
                    self.frame.u.get((cx - 1) as usize, (cy + i) as usize) as i32;
                refs[1].1[i as usize] =
                    self.frame.v.get((cx - 1) as usize, (cy + i) as usize) as i32;
            }
        }
        let mode = mb.intra_chroma_pred_mode;
        let coeffs = self.info.coeffs[addr].clone();
        // 色度用 `QP′C`（表 8-15 的映射；Cb 用 `chroma_qp_index_offset`、
        // Cr 用 `second_chroma_qp_index_offset`，见式 8-309/8-310）
        let qp_y = if mb.is_pcm { 0 } else { mb.qp_y };
        let qp_cb = transform::chroma_qp(qp_y + self.chroma_offset(0));
        let qp_cr = transform::chroma_qp(qp_y + self.chroma_offset(1));
        for (i, is_cb) in [(0usize, true), (1, false)] {
            self.chroma_plane(
                is_cb,
                addr,
                mode,
                &refs[i].0,
                &refs[i].1,
                top_avail,
                left_avail,
                &coeffs,
                if is_cb { qp_cb } else { qp_cr },
                cx,
                cy,
            )?;
        }
        Ok(())
    }

    /// `qPOffset`：Cb 用 `chroma_qp_index_offset`、Cr 用 `second_chroma_qp_index_offset`（式 8-309/8-310）。
    fn chroma_offset(&self, i: usize) -> i32 {
        self.chroma_off[i.min(1)]
    }

    #[allow(clippy::too_many_arguments)]
    fn chroma_plane(
        &mut self,
        is_cb: bool,
        addr: usize,
        mode: u8,
        top: &[i32; 8],
        left: &[i32; 8],
        top_avail: [bool; 8],
        left_avail: [bool; 8],
        coeffs: &MbCoeffs,
        qp: i32,
        cx: i32,
        cy: i32,
    ) -> Result<()> {
        let pred = intra::predict_chroma(mode, top, left, top_avail, left_avail, 8);
        let (dc_vals, ac) = if is_cb {
            (coeffs.cb_dc, coeffs.cb4)
        } else {
            (coeffs.cr_dc, coeffs.cr4)
        };
        let dcc = transform::chroma_dc_2x2(&dc_vals, qp);
        let mut res = [0i32; 64];
        for blk in 0..4usize {
            let a: [i32; 15] = ac[blk][..15].try_into().unwrap();
            let mut c = transform::scan_4x4(&a, true);
            c[0] = dcc[blk];
            let d = transform::dequant_4x4(&c, qp, &self.ws, if is_cb { 1 } else { 2 }, true);
            let r = transform::inverse_4x4(&d);
            let bx = (blk % 2) * 4;
            let by = (blk / 2) * 4;
            for i in 0..4 {
                for j in 0..4 {
                    res[(by + i) * 8 + bx + j] = r[i * 4 + j];
                }
            }
        }
        for y in 0..8usize {
            for x in 0..8usize {
                let v = clip8(pred[y * 8 + x] + res[y * 8 + x]);
                let xs = (cx + x as i32) as usize;
                let ys = (cy + y as i32) as usize;
                if is_cb {
                    self.frame.u.set(xs, ys, v);
                } else {
                    self.frame.v.set(xs, ys, v);
                }
            }
        }
        let _ = addr;
        Ok(())
    }
}

#[inline]
fn clip8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 图 6-10 的象限序：`0 1 4 5 / 2 3 6 7 / 8 9 12 13 / 10 11 14 15`。
    /// 这一条错了整个 4×4 的预测、残差、`nC` 邻居全跟着错，而且**不报错**。
    #[test]
    fn blk4_mapping_matches_figure_6_10() {
        let expect = [
            (0, 0), (4, 0), (0, 4), (4, 4),
            (8, 0), (12, 0), (8, 4), (12, 4),
            (0, 8), (4, 8), (0, 12), (4, 12),
            (8, 8), (12, 8), (8, 12), (12, 12),
        ];
        for (i, &(x, y)) in expect.iter().enumerate() {
            assert_eq!(blk4_xy(i), (x, y), "blk {i}");
            assert_eq!(blk4_of(x as i32, y as i32), i, "反查 blk {i}");
        }
    }

    /// `dcY` 的下标按**块的光栅序**排，而 `luma4x4BlkIdx` 是象限序 —— 图 8-6 画的就是这个错位。
    #[test]
    fn dc_y_index_follows_raster_block_order() {
        assert_eq!(dc_y_index(0), 0);
        assert_eq!(dc_y_index(1), 1);
        assert_eq!(dc_y_index(2), 4);
        assert_eq!(dc_y_index(3), 5);
        assert_eq!(dc_y_index(4), 2);
        assert_eq!(dc_y_index(5), 3);
        assert_eq!(dc_y_index(12), 10);
        // 双射
        let mut seen = [false; 16];
        for i in 0..16 {
            assert!(!seen[dc_y_index(i)]);
            seen[dc_y_index(i)] = true;
        }
    }

    /// §6.4.11.4：4×4 块的左右/上下邻居（**按像素位置走**，不按索引硬算）。
    ///
    /// 这里用一张 1×1 宏块的网格手工核对几个"反直觉"的：
    /// `blk 0` 的左邻是左宏块的 `blk 5`（那是 (12,0) 的位置）、
    /// `blk 4`（在 (8,0)）的上邻是上宏块的 `blk 14`（(8,12) 的位置）。
    #[test]
    fn neighbour4_follows_pixel_positions() {
        // 直接复用相同公式的纯函数版本（`Recon` 需要整帧，单测里拆不出来）
        let a = |blk: usize| {
            let (x, y) = blk4_xy(blk);
            if x >= 4 {
                (true, blk4_of(x as i32 - 4, y as i32))
            } else {
                (false, blk4_of(12, y as i32))
            }
        };
        let b = |blk: usize| {
            let (x, y) = blk4_xy(blk);
            if y >= 4 {
                (true, blk4_of(x as i32, y as i32 - 4))
            } else {
                (false, blk4_of(x as i32, 12))
            }
        };
        assert_eq!(a(0), (false, 5), "blk0 的左邻在左宏块，是它的 blk5");
        assert_eq!(a(1), (true, 0), "blk1 在 (4,0)，左邻是同宏块的 blk0");
        assert_eq!(a(2), (false, 7), "blk2 在 (0,4)，左邻在左宏块的 (12,4) = blk7");
        assert_eq!(a(4), (true, 1), "blk4 在 (8,0)，左邻是同宏块的 blk1");
        assert_eq!(a(5), (true, 4), "blk5 在 (12,0)，左邻是同宏块的 blk4");
        assert_eq!(a(8), (false, 13), "blk8 在 (0,8)，左邻在左宏块的 (12,8) = blk13");
        assert_eq!(b(0), (false, 10), "blk0 在 (0,0)，上邻在上宏块的 (0,12) = blk10");
        assert_eq!(b(1), (false, 11));
        assert_eq!(b(4), (false, 14), "blk4 在 (8,0)，上邻是上宏块的 (8,12) = blk14");
        assert_eq!(b(5), (false, 15));
        assert_eq!(b(10), (true, 8), "blk10 在 (0,12)，上邻是同宏块的 (0,8) = blk8");
        assert_eq!(b(12), (true, 6), "blk12 在 (8,8)，上邻是同宏块的 (8,4) = blk6");
    }

    /// 图 6-11：8×8 块是**光栅序**（0 1 / 2 3）。
    #[test]
    fn blk8_mapping_is_raster() {
        assert_eq!(blk8_xy(0), (0, 0));
        assert_eq!(blk8_xy(1), (8, 0));
        assert_eq!(blk8_xy(2), (0, 8));
        assert_eq!(blk8_xy(3), (8, 8));
        assert_eq!(blk8_of(8, 8), 3);
    }
}
