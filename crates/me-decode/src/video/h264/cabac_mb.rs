//! **CABAC 的宏块层**（§7.3.4 切片数据 / §7.3.5 宏块层，`entropy_coding_mode_flag == 1`）。
//!
//! 与 [`super::mb`] 那套 CAVLC 的宏块层是**平行实现**，不是替换：
//!
//! - 相同的部分：语法树的形状、表 7-11/7-13/7-14 的编号、预测与重建要用的
//!   [`MbInfo`] / [`MbCoeffs`] 布局 —— 所以重建/去块那一层完全不用改。
//! - 不同的部分：**每个语法元素怎么读**（上下文 + 算术解码 vs. VLC 码表），
//!   以及"跳过"的表示（CAVLC 是 `mb_skip_run` 批量，CABAC 是逐宏块 `mb_skip_flag`）。
//!
//! ## 上下文索引的来路
//!
//! `ctxIdx` = **语法元素的 `ctxIdxOffset`**（表 9-39 ~ 9-43）+ `ctxIdxInc`。
//! 这层代码里出现的裸数字（3、11、14、21、40、54、60、64、68、73、77、399、
//! 105/120/…/426）全部是 `ctxIdxOffset`，注释里标了对应的语法元素；
//! `ctxIdxInc` 则按 §9.3.3.1 的"邻居条件项"现算。
//!
//! ## 残差：三张表 + 一条反向扫描
//!
//! CABAC 的残差不是"`coeff_token` + 逐级 level"，而是：
//!
//! 1. `coded_block_flag`（每个块一次）；
//! 2. `significant_coeff_flag` / `last_significant_coeff_flag` 沿扫描序走一遍，
//!    得到"哪些位置有系数、最后一个是哪个"（**最后一个位置是隐含的**，不再发标志）；
//! 3. 从最后一个系数**倒着**读幅值：先一个 `coeff_abs_level_minus1 > 0` 的 bin，
//!    再一元前缀到 14，>= 15 走旁路 Exp-Golomb，最后一位符号。
//!
//! 幅值的上下文按"节点"推进（`node_ctx` 0..7），这是 §9.3.3.1.1.10 那张图：
//! 前面解出 1 和前面解出 >1，走向不同的上下文 —— 这层用三张小表照抄。
//!
//! ## 落盘布局与 CAVLC 一致（关键）
//!
//! [`MbCoeffs`] 里存的是**扫描序的 level**（不是光栅序）：
//! 4×4 块 16 个、I16 的 AC 15 个（跳过 DC 位置）、8×8 块 64 个、色度 DC 4 个。
//! 反扫描/反量化在 [`super::reconstruct`] 里做，这里只负责"摆对槽位"。
//! 摆错的症状是"画面糊成一片但语法照样走得下去"，很难从报错看出来。

use me_core::{MediaError, Result};

use super::cabac::{init_contexts, Cabac, Ctx};
use super::cabac_tables::{LAST_OFF_8X8, SIG_OFF_8X8};
use super::inter::{Mv, MvInfo};
use super::mb::{
    self, i16x16_parts, skip_macroblock, InterMb, IntraPred, MbCoeffs, MbInfo, PictureInfo, I_PCM,
    I_NXN,
};
use crate::h264::{Pps, SliceHeader, Sps};

// ── 块类别（表 9-43 的 ctxBlockCat）────────────────────────────────────────
const CAT_LUMA_DC: usize = 0;
const CAT_LUMA_AC: usize = 1;
const CAT_LUMA4: usize = 2;
const CAT_CHROMA_DC: usize = 3;
const CAT_CHROMA_AC: usize = 4;
const CAT_LUMA8: usize = 5;

/// 表 9-43：`significant_coeff_flag` 的 `ctxIdxOffset`。
const SIG_BASE: [usize; 6] = [105, 120, 134, 149, 152, 402];
/// 表 9-43：`last_significant_coeff_flag` 的 `ctxIdxOffset`。
const LAST_BASE: [usize; 6] = [166, 181, 195, 210, 213, 417];
/// 表 9-43：`coeff_abs_level_minus1` 的 `ctxIdxOffset`。
const ABS_BASE: [usize; 6] = [227, 237, 247, 257, 266, 426];
/// 表 9-43：`coded_block_flag` 的 `ctxIdxOffset`。
const CBF_BASE: [usize; 6] = [85, 89, 93, 97, 101, 460];

/// §9.3.3.1.1.10：节点上下文 → 第一层（`level == 1`）用哪个上下文。
const LEVEL1_CTX: [usize; 8] = [1, 2, 3, 4, 0, 0, 0, 0];
/// 节点上下文 → 第二层（`level > 1`）用哪个上下文。
const LEVELGT1_CTX: [usize; 8] = [5, 5, 5, 5, 6, 7, 8, 9];
/// 解出 `level == 1` 之后节点怎么走。
const NODE_AFTER_1: [usize; 8] = [1, 2, 3, 3, 4, 5, 6, 7];
/// 解出 `level > 1` 之后节点怎么走。
const NODE_AFTER_GT1: [usize; 8] = [4, 4, 4, 4, 5, 6, 7, 7];

/// CABAC 目前**默认不开**：I 切片已经逐样本对上 FFmpeg，但 P 切片还有个别宏块
/// 的偏差。真实素材必有 P/B 帧，半成品默认放出去用户看到的是局部花屏而不是报错，
/// 所以先用环境变量挡住，P 修好之后把这段删掉。
///
/// `JISHI_H264_CABAC=1` 可以强行启用（开发/验证用）。
fn cabac_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("JISHI_H264_CABAC").is_ok_and(|v| v != "0"))
}

fn ensure_enabled() -> Result<()> {
    if cabac_enabled() {
        Ok(())
    } else {
        Err(MediaError::Unsupported(
            "这条流用的是 CABAC。自研软解的 CABAC 仍在验证中（I 切片已逐样本对上，\
             P 切片还有偏差）—— 设 JISHI_H264_CABAC=1 可强行启用"
                .into(),
        ))
    }
}

/// 一条 CABAC 切片的解码现场。
struct SliceCabac<'a> {
    c: Cabac<'a>,
    ctx: [Ctx; 1024],
    /// 当前宏块之前的 `QPY`（宏块内 `mb_qp_delta` 会改它）
    qp_y: i32,
    /// 上一个宏块的 `mb_qp_delta` 是否非零（`mb_qp_delta` 第一个 bin 的上下文）
    delta_nonzero: bool,
    /// `H264_CABAC_TRACE=1`：逐 bin 打印（定位"从哪个语法元素起错"最快的一条）
    trace: bool,
}

impl<'a> SliceCabac<'a> {
    #[inline]
    fn bin(&mut self, ctx: usize) -> u32 {
        let v = self.c.decision(&mut self.ctx[ctx]);
        if self.trace {
            eprintln!("    bin ctx={ctx} -> {v} @bit {}", self.c.bit_position());
        }
        v
    }

    #[inline]
    fn flag(&mut self, ctx: usize) -> bool {
        self.bin(ctx) == 1
    }
}

// ── 邻居 ────────────────────────────────────────────────────────────────────

fn left_of(info: &PictureInfo, addr: usize) -> Option<usize> {
    if addr % info.mb_cols == 0 {
        return None;
    }
    let a = addr - 1;
    info.decoded[a].then_some(a)
}

fn top_of(info: &PictureInfo, addr: usize) -> Option<usize> {
    if addr < info.mb_cols {
        return None;
    }
    let b = addr - info.mb_cols;
    info.decoded[b].then_some(b)
}

/// 亮度 4×4 块（**全图坐标**，块单位）有没有系数。
///
/// ⚠️ **不可用（画面外 / 还没解到）按 1 算**。这条和 DC 块是同一个道理：实现里
/// 邻居的 `non_zero_count` 用 `-1` 表示不可用，`!!(-1) = 1`。按 0 算的话第一个
/// 宏块第一块的 `coded_block_flag` 上下文是 93，正确是 **96** —— 那一位错下去，
/// 显著性图从第二个系数起就会歪（`-25, -2` 会被解成 `-1, -1`）。
fn luma_nz(info: &PictureInfo, cur: usize, x4: i32, y4: i32) -> bool {
    if x4 < 0 || y4 < 0 {
        return true;
    }
    let (x4, y4) = (x4 as usize, y4 as usize);
    let (mbx, mby) = (x4 / 4, y4 / 4);
    if mbx >= info.mb_cols || mby >= info.mb_rows {
        return true;
    }
    let a = mby * info.mb_cols + mbx;
    // ⚠️ **当前宏块自己还没标 decoded**（要整块解完才标），不能把它当成"不可用"：
    // 同一宏块内的邻居直接读 `tc` 网格（按块顺序逐块写好），只有"别的宏块且还没解到"
    // 才回退成 1。写成 `!decoded[a] → true` 会让宏块内所有邻居恒为 1，
    // 第 4 个块（blk3）的上下文从 95 变成 96，从那里开始整片错。
    if a != cur && !info.decoded[a] {
        return true;
    }
    info.tc.luma_get(x4, y4) > 0
}

/// 色度 4×4 块（块单位，色度网格）有没有系数。`plane` 0=Cb、1=Cr。
/// 不可用同样按 1 算（与亮度一致）。
fn chroma_nz(info: &PictureInfo, cur: usize, plane: usize, x4: i32, y4: i32) -> bool {
    if x4 < 0 || y4 < 0 {
        return true;
    }
    let (x4, y4) = (x4 as usize, y4 as usize);
    let (mbx, mby) = (x4 / 2, y4 / 2);
    if mbx >= info.mb_cols || mby >= info.mb_rows {
        return true;
    }
    let a = mby * info.mb_cols + mbx;
    if a != cur && !info.decoded[a] {
        return true;
    }
    info.tc.chroma_get(plane, x4, y4) > 0
}

/// DC 块的"左/上邻有系数"标志。
///
/// ⚠️ **邻居不可用时按 1 算**（§9.3.3.1.1.9 的 `condTermFlag` 回退到 `b_intra`）：
/// x264 编码侧写得很直白 —— `i_cbp_left != -1 ? ... : b_intra`，而这里只会出现在
/// 帧内宏块上，所以恒为 1。按 0 算的话上下文是 85，正确是 88，第一个宏块的
/// cbf 就会读错，整条流从这里开始错位。
fn dc_luma_nz(info: &PictureInfo, addr: Option<usize>) -> bool {
    match addr {
        Some(a) => info.coeffs[a].luma_dc.iter().any(|&v| v != 0),
        None => true,
    }
}

fn dc_chroma_nz(info: &PictureInfo, addr: Option<usize>, plane: usize) -> bool {
    match addr {
        Some(a) => {
            let c = &info.coeffs[a];
            if plane == 0 {
                c.cb_dc.iter().any(|&v| v != 0)
            } else {
                c.cr_dc.iter().any(|&v| v != 0)
            }
        }
        None => true,
    }
}

// ── 一个块的残差 ────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct BlockLevels {
    levels: [i32; 64],
    nnz: u32,
}

/// 解一个残差块（§9.3.2 + §9.3.3.1.1.9 ~ 9.3.3.1.1.10）。
///
/// `nza`/`nzb` 是左邻/上邻的 "coded block flag"（不可用算 false）。
/// 返回的 `levels` 按**扫描序**排（索引 0 = 扫描位置 0）。
fn decode_block(
    s: &mut SliceCabac<'_>,
    cat: usize,
    max_coeff: usize,
    nza: bool,
    nzb: bool,
) -> Result<BlockLevels> {
    let mut out = BlockLevels {
        levels: [0; 64],
        nnz: 0,
    };
    // coded_block_flag
    let cbf = CBF_BASE[cat] + nza as usize + 2 * nzb as usize;
    if s.bin(cbf) == 0 {
        return Ok(out);
    }

    // ── 显著系数图：正着扫 ────────────────────────────────────────────────
    let mut pos = [0u8; 64];
    let mut count = 0usize;
    let mut found_last = false;
    if max_coeff == 64 {
        let mut i = 0usize;
        while i < 63 {
            if s.bin(SIG_BASE[cat] + SIG_OFF_8X8[i] as usize) == 1 {
                pos[count] = i as u8;
                count += 1;
                if s.bin(LAST_BASE[cat] + LAST_OFF_8X8[i] as usize) == 1 {
                    found_last = true;
                    break;
                }
            }
            i += 1;
        }
    } else {
        let mut i = 0usize;
        while i < max_coeff - 1 {
            if s.bin(SIG_BASE[cat] + i) == 1 {
                pos[count] = i as u8;
                count += 1;
                if s.bin(LAST_BASE[cat] + i) == 1 {
                    found_last = true;
                    break;
                }
            }
            i += 1;
        }
    }
    if !found_last {
        // §9.3.2.3：扫到倒数第二个位置还没有 last 标志，最后一个系数就是隐含的
        pos[count] = (max_coeff - 1) as u8;
        count += 1;
    }

    // ── 幅值：倒着读 ──────────────────────────────────────────────────────
    let mut node = 0usize;
    for k in (0..count).rev() {
        let slot = pos[k] as usize;
        let mut abs: u32;
        if s.bin(ABS_BASE[cat] + LEVEL1_CTX[node]) == 0 {
            abs = 1;
            node = NODE_AFTER_1[node];
        } else {
            let ctx_gt1 = ABS_BASE[cat] + LEVELGT1_CTX[node];
            node = NODE_AFTER_GT1[node];
            abs = 2;
            while abs < 15 && s.bin(ctx_gt1) == 1 {
                abs += 1;
            }
            if abs >= 15 {
                // 旁路 Exp-Golomb：j 个 1 + 1 个 0 + j 位后缀
                let mut j = 0u32;
                while s.c.bypass() == 1 && j < 23 {
                    j += 1;
                }
                let mut v = 1u32;
                for _ in 0..j {
                    v = v * 2 + s.c.bypass();
                }
                abs = v + 14;
            }
        }
        let negative = s.c.bypass() == 1;
        out.levels[slot] = if negative { -(abs as i32) } else { abs as i32 };
    }
    out.nnz = count as u32;
    Ok(out)
}

/// 写一个宏块的残差（布局与 CAVLC 那边完全一致）。
fn residual_cabac(
    s: &mut SliceCabac<'_>,
    info: &mut PictureInfo,
    addr: usize,
    mb: &MbInfo,
    is_i16: bool,
) -> Result<()> {
    let (mbx, mby) = (addr % info.mb_cols, addr / info.mb_cols);
    let (x4_0, y4_0) = (mbx * 4, mby * 4);
    let mut coeffs = MbCoeffs::default();

    if is_i16 {
        // `Intra16x16DCLevel`：左/上邻的 **DC 块**有没有系数（不是 cbp）
        let (la, ta) = (left_of(info, addr), top_of(info, addr));
        let b = decode_block(
            s,
            CAT_LUMA_DC,
            16,
            dc_luma_nz(info, la),
            dc_luma_nz(info, ta),
        )?;
        coeffs.luma_dc.copy_from_slice(&b.levels[..16]);
    }

    for i8x8 in 0..4usize {
        if mb.cbp_luma & (1 << i8x8) == 0 {
            for i4x4 in 0..4usize {
                let idx = i8x8 * 4 + i4x4;
                let (dx, dy) = super::reconstruct::blk4_xy(idx);
                info.tc.luma_set(x4_0 + dx / 4, y4_0 + dy / 4, 0);
            }
            continue;
        }
        if mb.transform_8x8 {
            // 8×8：一个块 64 个系数，上下文按"8×8 块左上角那个 4×4 的邻居"取
            let (dx, dy) = super::reconstruct::blk4_xy(i8x8 * 4);
            let (bx, by) = ((x4_0 + dx / 4) as i32, (y4_0 + dy / 4) as i32);
            let b = decode_block(
                s,
                CAT_LUMA8,
                64,
                luma_nz(info, addr, bx - 1, by),
                luma_nz(info, addr, bx, by - 1),
            )?;
            coeffs.luma8[i8x8] = b.levels;
            // 8×8 块的四块 4×4 共享一个"非零个数"（§9.3.2.3 的 NOTE）
            for i4x4 in 0..4usize {
                let idx = i8x8 * 4 + i4x4;
                let (dx, dy) = super::reconstruct::blk4_xy(idx);
                info.tc
                    .luma_set(x4_0 + dx / 4, y4_0 + dy / 4, b.nnz.min(255) as u8);
            }
        } else {
            for i4x4 in 0..4usize {
                let idx = i8x8 * 4 + i4x4;
                let (dx, dy) = super::reconstruct::blk4_xy(idx);
                let (bx, by) = (x4_0 + dx / 4, y4_0 + dy / 4);
                let (cat, max) = if is_i16 {
                    (CAT_LUMA_AC, 15)
                } else {
                    (CAT_LUMA4, 16)
                };
                let nza = luma_nz(info, addr, bx as i32 - 1, by as i32);
                let nzb = luma_nz(info, addr, bx as i32, by as i32 - 1);
                let b = decode_block(s, cat, max, nza, nzb)?;
                if is_i16 {
                    // AC：15 个，槽位 0..14 对应扫描位置 1..15
                    coeffs.luma4[idx][..15].copy_from_slice(&b.levels[..15]);
                } else {
                    coeffs.luma4[idx] = b.levels[..16].try_into().unwrap();
                }
                info.tc.luma_set(bx, by, b.nnz.min(255) as u8);
            }
        }
    }

    // ── 色度 ──────────────────────────────────────────────────────────────
    // ⚠️ 顺序是 §7.3.5.3 写死的 **Cb DC → Cr DC → Cb AC×4 → Cr AC×4**，
    // 不是"一个分量解完再下一个"（CAVLC 那边同一个坑，见 mb.rs 的注释）。
    let chroma_dc = mb.cbp_chroma != 0;
    let chroma_ac = mb.cbp_chroma & 2 != 0;
    let (cx0, cy0) = (mbx * 2, mby * 2);
    let (la, ta) = (left_of(info, addr), top_of(info, addr));
    for plane in 0..2usize {
        if !chroma_dc {
            continue;
        }
        let b = decode_block(
            s,
            CAT_CHROMA_DC,
            4,
            dc_chroma_nz(info, la, plane),
            dc_chroma_nz(info, ta, plane),
        )?;
        let mut dc = [0i32; 4];
        dc.copy_from_slice(&b.levels[..4]);
        if plane == 0 {
            coeffs.cb_dc = dc;
        } else {
            coeffs.cr_dc = dc;
        }
    }
    for plane in 0..2usize {
        for i4x4 in 0..4usize {
            let (bx, by) = (cx0 + i4x4 % 2, cy0 + i4x4 / 2);
            if !chroma_ac {
                info.tc.chroma_put(plane, bx, by, 0);
                continue;
            }
            let b = decode_block(
                s,
                CAT_CHROMA_AC,
                15,
                chroma_nz(info, addr, plane, bx as i32 - 1, by as i32),
                chroma_nz(info, addr, plane, bx as i32, by as i32 - 1),
            )?;
            let mut blk = [0i32; 16];
            blk[..15].copy_from_slice(&b.levels[..15]);
            if plane == 0 {
                coeffs.cb4[i4x4] = blk;
            } else {
                coeffs.cr4[i4x4] = blk;
            }
            info.tc.chroma_put(plane, bx, by, b.nnz.min(255) as u8);
        }
    }

    info.coeffs[addr] = coeffs;
    Ok(())
}

// ── 上下文推导（§9.3.3.1）──────────────────────────────────────────────────

/// `mb_skip_flag` 的 `ctxIdxInc`：左、上**各算 1**（不是 1 和 2）。
fn skip_flag_ctx(info: &PictureInfo, addr: usize) -> usize {
    let a = left_of(info, addr).is_some_and(|a| !info.mb[a].skipped);
    let b = top_of(info, addr).is_some_and(|b| !info.mb[b].skipped);
    a as usize + b as usize
}

fn i16_neighbour_ctx(info: &PictureInfo, addr: usize, intra_slice: bool) -> usize {
    if !intra_slice {
        return 0;
    }
    let is16 = |a: usize| {
        info.mb[a].intra_pred == IntraPred::I16x16 || info.mb[a].is_pcm
    };
    // 左、上各算 1（表 9-39 的 mb_type：A + B，不是 A + 2B）
    left_of(info, addr).is_some_and(is16) as usize
        + top_of(info, addr).is_some_and(is16) as usize
}

/// `transform_size_8x8_flag` 的 `ctxIdxInc`：同样是左、上各 1。
fn t8_ctx(info: &PictureInfo, addr: usize) -> usize {
    let a = left_of(info, addr).is_some_and(|a| info.mb[a].transform_8x8);
    let b = top_of(info, addr).is_some_and(|b| info.mb[b].transform_8x8);
    a as usize + b as usize
}

/// `intra_chroma_pred_mode` 的 `ctxIdxInc`：左、上各算 1。
fn chroma_pred_ctx(info: &PictureInfo, addr: usize) -> usize {
    let hit = |a: usize| info.mb[a].inter.is_none() && info.mb[a].intra_chroma_pred_mode != 0;
    left_of(info, addr).is_some_and(hit) as usize
        + top_of(info, addr).is_some_and(hit) as usize
}

fn cbp_luma_cabac(s: &mut SliceCabac<'_>, info: &PictureInfo, addr: usize) -> u32 {
    // ⚠️ 邻居**不可用**时按 `0xF`（四位全 1）算，不是 0。
    // x264/FFmpeg 内部把不可用邻居的 cbp 记成 `-1`，`!(cbp & bit)` 于是全为 0；
    // 按 0 算会把第一个宏块的上下文从 73 推到 76，`coded_block_pattern` 一开始就读错。
    let cbp_a = left_of(info, addr).map_or(0xF, |a| info.mb[a].cbp_luma);
    let cbp_b = top_of(info, addr).map_or(0xF, |b| info.mb[b].cbp_luma);
    let mut cbp = 0u32;
    // 73 = coded_block_pattern 的 ctxIdxOffset
    let ctx = ((cbp_a & 0x02) == 0) as usize + 2 * ((cbp_b & 0x04) == 0) as usize;
    cbp |= s.bin(73 + ctx);
    let ctx = ((cbp & 0x01) == 0) as usize + 2 * ((cbp_b & 0x08) == 0) as usize;
    cbp |= s.bin(73 + ctx) << 1;
    let ctx = ((cbp_a & 0x08) == 0) as usize + 2 * ((cbp & 0x01) == 0) as usize;
    cbp |= s.bin(73 + ctx) << 2;
    let ctx = ((cbp & 0x04) == 0) as usize + 2 * ((cbp & 0x02) == 0) as usize;
    cbp |= s.bin(73 + ctx) << 3;
    cbp
}

fn cbp_chroma_cabac(s: &mut SliceCabac<'_>, info: &PictureInfo, addr: usize) -> u32 {
    let cbp_a = left_of(info, addr).map_or(0, |a| info.mb[a].cbp_chroma);
    let cbp_b = top_of(info, addr).map_or(0, |b| info.mb[b].cbp_chroma);
    // 77 = coded_block_pattern 的色度段 ctxIdxOffset
    let ctx = (cbp_a > 0) as usize + 2 * (cbp_b > 0) as usize;
    if s.bin(77 + ctx) == 0 {
        return 0;
    }
    let ctx = 4 + (cbp_a == 2) as usize + 2 * (cbp_b == 2) as usize;
    1 + s.bin(77 + ctx)
}

// ── 宏块层 ──────────────────────────────────────────────────────────────────

/// `mb_qp_delta`（§9.3.3.1.1.11）：一元前缀，奇数正 / 偶数负。
fn qp_delta_cabac(s: &mut SliceCabac<'_>) -> Result<i32> {
    // 60 = mb_qp_delta 的 ctxIdxOffset；第一个 bin 的上文是"上一个 delta 是否为 0"
    if s.bin(60 + s.delta_nonzero as usize) == 0 {
        s.delta_nonzero = false;
        return Ok(0);
    }
    let mut val = 1i32;
    let mut ctx = 2usize;
    loop {
        if s.bin(60 + ctx) == 0 {
            break;
        }
        ctx = 3;
        val += 1;
        if val > 104 {
            return Err(MediaError::Bitstream(
                "mb_qp_delta 的一元前缀太长（> 104），多半是上下文错位".into(),
            ));
        }
    }
    s.delta_nonzero = true;
    Ok(if val & 1 == 1 {
        (val + 1) >> 1
    } else {
        -((val + 1) >> 1)
    })
}

/// 应用 `mb_qp_delta`：与 CAVLC 那边同口径（越界就按 52 回绕，不硬报错）。
fn apply_qp_delta(qp_y: i32, delta: i32) -> Result<i32> {
    let mut qp = qp_y + delta;
    if !(0..=51).contains(&qp) {
        if qp < 0 {
            qp += 52;
        } else {
            qp -= 52;
        }
    }
    if !(0..=51).contains(&qp) {
        return Err(MediaError::Bitstream(format!(
            "CABAC 的 QPY = {qp} 越界（0..51）—— 多半是语法错位"
        )));
    }
    Ok(qp)
}

fn read_i16_mb_type(
    s: &mut SliceCabac<'_>,
    info: &PictureInfo,
    addr: usize,
    intra_slice: bool,
) -> u32 {
    if intra_slice {
        let ctx = 3 + i16_neighbour_ctx(info, addr, true);
        if s.bin(ctx) == 0 {
            return 0; // I_NxN
        }
    } else if s.bin(17) == 0 {
        return 0; // P/B 切片里的 I_NxN
    }
    if s.c.terminate() {
        return I_PCM;
    }
    let mut t = 1u32;
    if intra_slice {
        // 5 = 表 7-11 段（cbp_luma 6 / 色度 7,8 / 预测模式 9,10）
        t += 12 * s.bin(6);
        if s.bin(7) == 1 {
            t += 4 + 4 * s.bin(8);
        }
        t += 2 * s.bin(9);
        t += s.bin(10);
    } else {
        t += 12 * s.bin(18);
        if s.bin(19) == 1 {
            t += 4 + 4 * s.bin(19);
        }
        t += 2 * s.bin(20);
        t += s.bin(20);
    }
    t
}

/// 宏块层里**帧内**那一段：`mb_type` 已经解出来了（表 7-11 编号）。
fn intra_body(
    s: &mut SliceCabac<'_>,
    sps: &Sps,
    pps: &Pps,
    info: &mut PictureInfo,
    addr: usize,
    mb_type: u32,
) -> Result<i32> {
    let (mbx, mby) = (addr % info.mb_cols, addr / info.mb_cols);
    let (x4_0, y4_0) = (mbx * 4, mby * 4);

    if mb_type == I_PCM {
        s.c.byte_align();
        let bytes = s.c.take_bytes(256 + 2 * 64)?;
        let mut mb = MbInfo {
            mb_type,
            intra_pred: IntraPred::Pcm,
            is_pcm: true,
            qp_y: 0,
            ..MbInfo::default()
        };
        mb.qp_y = s.qp_y;
        let mut coeffs = MbCoeffs::default();
        coeffs.pcm = bytes.to_vec();
        for dy in 0..4 {
            for dx in 0..4 {
                info.tc.luma_set(x4_0 + dx, y4_0 + dy, 16);
            }
        }
        for dy in 0..2 {
            for dx in 0..2 {
                info.tc.chroma_put(0, mbx * 2 + dx, mby * 2 + dy, 16);
                info.tc.chroma_put(1, mbx * 2 + dx, mby * 2 + dy, 16);
            }
        }
        for dy in 0..4 {
            for dx in 0..4 {
                info.mv_set(x4_0 + dx, y4_0 + dy, MvInfo {
                    mv: Mv::ZERO,
                    ref_idx: -2,
                });
            }
        }
        let (slot_mb, slot_c) = info.mb_at(addr);
        *slot_mb = mb;
        *slot_c = coeffs;
        s.c.restart()?;
        return Ok(s.qp_y);
    }

    let parts = i16x16_parts(mb_type);
    let mut mb = MbInfo {
        mb_type,
        qp_y: s.qp_y,
        ..MbInfo::default()
    };
    match parts {
        Some(p) => {
            mb.intra_pred = IntraPred::I16x16;
            mb.intra16x16_pred_mode = p.pred_mode;
            mb.cbp_luma = p.cbp_luma;
            mb.cbp_chroma = p.cbp_chroma;
        }
        None => {
            // I_NxN：先读 transform_size_8x8_flag（§7.3.5 的顺序），再读预测模式
            let t8 = pps.transform_8x8_mode_flag && s.flag(399 + t8_ctx(info, addr));
            mb.transform_8x8 = t8;
            mb.intra_pred = if t8 {
                IntraPred::NxN8x8
            } else {
                IntraPred::NxN4x4
            };
            if t8 {
                for i in 0..4usize {
                    mb.intra8x8_prev[i] = s.flag(68);
                    if !mb.intra8x8_prev[i] {
                        let mut m = 0u8;
                        m += s.bin(69) as u8;
                        m += (s.bin(69) as u8) << 1;
                        m += (s.bin(69) as u8) << 2;
                        mb.intra8x8_rem[i] = m;
                    }
                }
            } else {
                for i in 0..16usize {
                    mb.intra4x4_prev[i] = s.flag(68);
                    if !mb.intra4x4_prev[i] {
                        let mut m = 0u8;
                        m += s.bin(69) as u8;
                        m += (s.bin(69) as u8) << 1;
                        m += (s.bin(69) as u8) << 2;
                        mb.intra4x4_rem[i] = m;
                    }
                }
            }
        }
    }

    // intra_chroma_pred_mode：64 = 该语法元素的 ctxIdxOffset
    let ctx = 64 + chroma_pred_ctx(info, addr);
    mb.intra_chroma_pred_mode = if s.bin(ctx) == 0 {
        0
    } else if s.bin(67) == 0 {
        1
    } else if s.bin(67) == 0 {
        2
    } else {
        3
    };

    if parts.is_none() {
        mb.cbp_luma = cbp_luma_cabac(s, info, addr);
        mb.cbp_chroma = cbp_chroma_cabac(s, info, addr);
    }

    let need_residual = mb.cbp_luma > 0 || mb.cbp_chroma > 0 || parts.is_some();
    if need_residual {
        let delta = qp_delta_cabac(s)?;
        s.qp_y = apply_qp_delta(s.qp_y, delta)?;
        mb.qp_y = s.qp_y;
        residual_cabac(s, info, addr, &mb, parts.is_some())?;
    }

    if mb::trace_mb() {
        eprintln!(
            "MB[pic={}] {addr}: CABAC 帧内 type={mb_type} pred={:?} 模式16={} 色度={} cbp=({:x},{}) t8={} qp={}",
            info.pic_index,
            mb.intra_pred,
            mb.intra16x16_pred_mode,
            mb.intra_chroma_pred_mode,
            mb.cbp_luma,
            mb.cbp_chroma,
            mb.transform_8x8,
            mb.qp_y
        );
    }
    for dy in 0..4 {
        for dx in 0..4 {
            info.mv_set(x4_0 + dx, y4_0 + dy, MvInfo {
                mv: Mv::ZERO,
                ref_idx: -2,
            });
        }
    }
    let (slot_mb, _) = info.mb_at(addr);
    *slot_mb = mb;
    Ok(s.qp_y)
}

/// `ref_idx_l0`（§9.3.3.1.1.6）：`ctxIdxOffset = 54`。
fn ref_idx_cabac(
    s: &mut SliceCabac<'_>,
    info: &PictureInfo,
    x4: i32,
    y4: i32,
    nref: u32,
) -> Result<i8> {
    let a = info
        .mv_get(x4 - 1, y4)
        .is_some_and(|m| m.ref_idx > 0) as usize;
    let b = info
        .mv_get(x4, y4 - 1)
        .is_some_and(|m| m.ref_idx > 0) as usize;
    let mut ctx = a + 2 * b;
    let mut r = 0i8;
    while s.bin(54 + ctx) == 1 {
        r += 1;
        if r as u32 >= nref {
            return Err(MediaError::Bitstream(format!(
                "CABAC 的 ref_idx_l0 = {r} 越界（只有 {nref} 个参考帧）"
            )));
        }
        ctx = (ctx >> 2) + 4;
    }
    Ok(r)
}

/// `mvd_l0` 的一个分量（§9.3.3.1.1.7）：`ctxIdxOffset` 40（x）/ 47（y）。
fn mvd_component(s: &mut SliceCabac<'_>, base: usize, amvd: i32) -> Result<i32> {
    let ctx = base + 2 + (amvd > 2) as usize + (amvd > 32) as usize;
    if s.bin(ctx) == 0 {
        return Ok(0);
    }
    let mut mvd = 1i32;
    let mut c = base + 3;
    while mvd < 9 && s.bin(c) == 1 {
        if mvd < 4 {
            c += 1;
        }
        mvd += 1;
    }
    if mvd >= 9 {
        let mut k = 3u32;
        while s.c.bypass() == 1 {
            mvd += 1 << k;
            k += 1;
            if k > 24 {
                return Err(MediaError::Bitstream(
                    "mvd 的 Exp-Golomb 逃逸太长了".into(),
                ));
            }
        }
        while k > 0 {
            k -= 1;
            mvd += (s.c.bypass() as i32) << k;
        }
    }
    let negative = s.c.bypass() == 1;
    Ok(if negative { -mvd } else { mvd })
}

/// P 切片的宏块层（CABAC）。
fn p_mb(
    s: &mut SliceCabac<'_>,
    header: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    info: &mut PictureInfo,
    addr: usize,
) -> Result<i32> {
    // 11 = mb_skip_flag 的 ctxIdxOffset
    if s.flag(11 + skip_flag_ctx(info, addr)) {
        skip_macroblock(info, addr);
        return Ok(s.qp_y);
    }

    // 14 / 15 / 16 / 17 = mb_type 在 P 切片的 ctxIdxOffset 段
    if s.flag(14) {
        let t = read_i16_mb_type(s, info, addr, false);
        return intra_body(s, sps, pps, info, addr, t);
    }
    let mb_type = if !s.flag(15) {
        if s.flag(16) {
            3 // P_8x8
        } else {
            0 // P_L0_16x16
        }
    } else if s.flag(17) {
        1 // P_L0_L0_16x8
    } else {
        2 // P_L0_L0_8x16
    };

    let (mbx, mby) = (addr % info.mb_cols, addr / info.mb_cols);
    let mut parts: Vec<(usize, usize, usize, usize)> = Vec::new();
    match mb_type {
        0 => parts.push((0, 0, 16, 16)),
        1 => {
            parts.push((0, 0, 16, 8));
            parts.push((0, 8, 16, 8));
        }
        2 => {
            parts.push((0, 0, 8, 16));
            parts.push((8, 0, 8, 16));
        }
        _ => {
            for q in 0..4usize {
                // sub_mb_type：21/22/23 = 该语法元素的 ctxIdxOffset 段
                let st = if s.flag(21) {
                    0
                } else if !s.flag(22) {
                    1
                } else if s.flag(23) {
                    2
                } else {
                    3
                };
                let (qx, qy) = ((q % 2) * 8, (q / 2) * 8);
                match st {
                    0 => parts.push((qx, qy, 8, 8)),
                    1 => {
                        parts.push((qx, qy, 8, 4));
                        parts.push((qx, qy + 4, 8, 4));
                    }
                    2 => {
                        parts.push((qx, qy, 4, 8));
                        parts.push((qx + 4, qy, 4, 8));
                    }
                    _ => {
                        for k in 0..4 {
                            parts.push((qx + (k % 2) * 4, qy + (k / 2) * 4, 4, 4));
                        }
                    }
                }
            }
        }
    }

    let nref = header.num_ref_idx_l0_active;
    let mut inter = InterMb {
        parts: parts.len() as u8,
        ..InterMb::default()
    };

    // ── 先读所有分区的 ref_idx（§7.3.5 的顺序：mb_part_pred 里 ref 全在前）──
    let mut refs = Vec::with_capacity(parts.len());
    for (i, &(px, py, _, _)) in parts.iter().enumerate() {
        let (x4, y4) = ((mbx * 4 + px / 4) as i32, (mby * 4 + py / 4) as i32);
        let r = if nref > 1 {
            ref_idx_cabac(s, info, x4, y4, nref)?
        } else {
            0
        };
        refs.push(r);
        // 立刻铺进网格：下一个分区的 ref_idx 上下文要看左邻
        let slot = MvInfo {
            mv: Mv::ZERO,
            ref_idx: r,
        };
        for dy in 0..py_h(parts[i]) {
            for dx in 0..px_w(parts[i]) {
                info.mv_set(x4 as usize + dx, y4 as usize + dy, slot);
            }
        }
    }

    // ── 再读所有分区的 mvd ────────────────────────────────────────────────
    for (i, &(px, py, pw, ph)) in parts.iter().enumerate() {
        let (x4, y4) = (mbx * 4 + px / 4, mby * 4 + py / 4);
        let (xi, yi) = (x4 as i32, y4 as i32);
        let ref_idx = refs[i];
        let amvd_x = info.mvd_abs_get(xi - 1, yi)[0] + info.mvd_abs_get(xi, yi - 1)[0];
        let amvd_y = info.mvd_abs_get(xi - 1, yi)[1] + info.mvd_abs_get(xi, yi - 1)[1];
        let mvd_x = mvd_component(s, 40, amvd_x)?;
        let mvd_y = mvd_component(s, 47, amvd_y)?;
        let mvp = mb::predict_partition_mv(info, x4, y4, pw / 4, ref_idx);
        let mv = Mv::new(mvp.x + mvd_x, mvp.y + mvd_y);
        if mb::trace_mb() {
            eprintln!(
                "  MVP[pic={}] MB{addr} part=({px},{py},{pw},{ph}) ref={ref_idx} mvp={mvp:?} mvd=({mvd_x},{mvd_y}) → {mv:?}",
                info.pic_index
            );
        }
        let slot = MvInfo { mv, ref_idx };
        for dy in 0..ph / 4 {
            for dx in 0..pw / 4 {
                info.mv_set(x4 + dx, y4 + dy, slot);
                info.mvd_abs_set(x4 + dx, y4 + dy, [mvd_x.abs(), mvd_y.abs()]);
                inter.mv[py / 4 + dy][px / 4 + dx] = slot;
            }
        }
    }

    let mut mb = MbInfo {
        mb_type,
        inter: Some(inter),
        qp_y: s.qp_y,
        ..MbInfo::default()
    };
    mb.cbp_luma = cbp_luma_cabac(s, info, addr);
    mb.cbp_chroma = cbp_chroma_cabac(s, info, addr);

    // 8×8 变换标志只在"整宏块都是 8×8 分区且 cbp_luma 非零"时出现
    if pps.transform_8x8_mode_flag && mb.cbp_luma != 0 && mb_type == 3 && parts.len() == 4 {
        let all8x8 = parts.iter().all(|&(_, _, w, h)| w == 8 && h == 8);
        if all8x8 {
            mb.transform_8x8 = s.flag(399 + t8_ctx(info, addr));
        }
    }

    if mb.cbp_luma > 0 || mb.cbp_chroma > 0 {
        let delta = qp_delta_cabac(s)?;
        s.qp_y = apply_qp_delta(s.qp_y, delta)?;
        mb.qp_y = s.qp_y;
        residual_cabac(s, info, addr, &mb, false)?;
    }

    if mb::trace_mb() {
        eprintln!(
            "MB[pic={}] {addr}: CABAC 帧间 type={mb_type} 分区={} cbp=({:x},{}) t8={} qp={} mv={:?}",
            info.pic_index,
            mb.inter.as_ref().map_or(0, |i| i.parts),
            mb.cbp_luma,
            mb.cbp_chroma,
            mb.transform_8x8,
            mb.qp_y,
            mb.inter.as_ref().map(|i| i.mv[0][0].mv)
        );
    }
    let (slot_mb, _) = info.mb_at(addr);
    *slot_mb = mb;
    Ok(s.qp_y)
}

#[inline]
fn px_w(part: (usize, usize, usize, usize)) -> usize {
    part.2 / 4
}

#[inline]
fn py_h(part: (usize, usize, usize, usize)) -> usize {
    part.3 / 4
}

// ── 切片入口 ────────────────────────────────────────────────────────────────

fn new_slice<'a>(
    header: &SliceHeader,
    pps: &Pps,
    raw: &'a [u8],
    is_i: bool,
) -> Result<SliceCabac<'a>> {
    let qp = 26 + pps.pic_init_qp_minus26 + header.slice_qp_delta;
    if std::env::var("H264_CABAC_TRACE").is_ok_and(|v| v != "0") {
        let c = init_contexts(qp, header.cabac_init_idc, is_i);
        eprintln!(
            "CABAC 现场：slice_qp={qp}（pic_init={} slice_delta={}）offset={} ctx3={:?} ctx6={:?} ctx64={:?} ctx85={:?}",
            pps.pic_init_qp_minus26, header.slice_qp_delta, header.slice_data_bit_offset,
            c[3], c[6], c[64], c[85]
        );
    }
    Ok(SliceCabac {
        c: Cabac::new(raw, header.slice_data_bit_offset)?,
        ctx: init_contexts(qp, header.cabac_init_idc, is_i),
        qp_y: qp,
        delta_nonzero: false,
        trace: std::env::var("H264_CABAC_TRACE").is_ok_and(|v| v != "0"),
    })
}

/// 解一条 **I 切片**（CABAC）。
pub fn decode_i_slice(
    header: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    raw: &[u8],
    info: &mut PictureInfo,
) -> Result<()> {
    if !header.is_i() {
        return Err(MediaError::Unsupported(format!(
            "decode_i_slice(CABAC) 只接 I 切片（这条是 slice_type = {}）",
            header.slice_type
        )));
    }
    ensure_enabled()?;
    let mut s = new_slice(header, pps, raw, true)?;
    let total = info.mb_cols * info.mb_rows;
    let mut addr = header.first_mb_in_slice as usize;
    if addr >= total {
        return Err(MediaError::Bitstream(format!(
            "first_mb_in_slice = {} 超过了这张图的宏块数 {total}",
            header.first_mb_in_slice
        )));
    }
    if mb::trace_mb() {
        eprintln!("CABAC I 切片起点 first_mb={addr} 共 {total} 个宏块");
    }
    // 调试：只解前 N 个宏块（定位"从第几个宏块开始错"用）
    let max_mb = std::env::var("H264_CABAC_MAX_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    let mut done = 0usize;
    loop {
        let t = read_i16_mb_type(&mut s, info, addr, true);
        intra_body(&mut s, sps, pps, info, addr, t)
            .map_err(|e| MediaError::Bitstream(format!("I 切片(CABAC, MB {addr}): {e}")))?;
        info.decoded[addr] = true;
        done += 1;
        if done >= max_mb {
            break;
        }
        // `end_of_slice_flag` 在**图的最后一个宏块**之后允许缺省：x264 有些配置
        // 在画满整帧时不再写这一位，FFmpeg 也按 `eos || mb_y >= mb_height` 处理。
        let eos = s.c.terminate();
        if eos || addr + 1 >= total {
            break;
        }
        addr += 1;
        if addr >= total {
            return Err(MediaError::Bitstream(format!(
                "CABAC 的宏块还没解完就到了图末尾（已解 {addr} 个，图里只有 {total} 个）"
            )));
        }
    }
    Ok(())
}

/// 解一条 **P 切片**（CABAC）。
pub fn decode_p_slice(
    header: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    raw: &[u8],
    info: &mut PictureInfo,
) -> Result<()> {
    if !header.is_p() {
        return Err(MediaError::Unsupported(format!(
            "decode_p_slice(CABAC) 只接 P 切片（这条是 slice_type = {}）",
            header.slice_type
        )));
    }
    ensure_enabled()?;
    let mut s = new_slice(header, pps, raw, false)?;
    let total = info.mb_cols * info.mb_rows;
    let mut addr = header.first_mb_in_slice as usize;
    if addr >= total {
        return Err(MediaError::Bitstream(format!(
            "first_mb_in_slice = {} 超过了这张图的宏块数 {total}",
            header.first_mb_in_slice
        )));
    }
    if mb::trace_mb() {
        eprintln!(
            "CABAC P 切片起点 first_mb={addr} 共 {total} 个宏块，num_ref_l0={} cabac_init_idc={}",
            header.num_ref_idx_l0_active, header.cabac_init_idc
        );
    }
    loop {
        p_mb(&mut s, header, sps, pps, info, addr)
            .map_err(|e| MediaError::Bitstream(format!("P 切片(CABAC, MB {addr}): {e}")))?;
        info.decoded[addr] = true;
        let eos = s.c.terminate();
        if eos || addr + 1 >= total {
            break;
        }
        addr += 1;
        if addr >= total {
            return Err(MediaError::Bitstream(format!(
                "CABAC 的宏块还没解完就到了图末尾（已解 {addr} 个，图里只有 {total} 个）"
            )));
        }
    }
    Ok(())
}
