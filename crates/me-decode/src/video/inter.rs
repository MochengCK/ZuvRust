//! AV1 **帧间预测**（规范 §7.11.3）—— 自研软解的最后一大块。
//!
//! 分层照规范抄：
//!
//! ```text
//! compute_prediction( )        §7.11.2   逐平面、逐"预测块"驱动（本文件）
//!   └─ predict_inter( )        §7.11.3.1 每一步：舍入参数 → warp？→ 每个参考列表
//!        ├─ 运动矢量缩放        §7.11.3.3 把当前帧坐标 + MV 换算到参考帧坐标（含缩放）
//!        ├─ 块帧间预测          §7.11.3.4 8 抽头亚像素插值（先横后纵）
//!        ├─ 块 warp            §7.11.3.5 仿射（LOCALWARP / 全局运动）
//!        ├─ 复合               §7.11.3.14 平均/加权/掩码混合
//!        └─ OBMC               §7.11.3.9 边上与邻居预测的重叠混合
//! ```
//!
//! ## 性能（2026-10-02 第十轮）：写回路径去掉逐像素 `dst.set`
//!
//! 帧间预测的六条写回路径（单参考、`COMPOUND_AVERAGE`、`COMPOUND_DISTANCE`、
//! `blend_masked` 的楔形/差分、`blend_interintra_core`、OBMC 的
//! `obmc_overlap_predict`，外加 intrabc 的拷贝）原来都是**逐像素** `Plane::set`：
//! 每像素一次下标乘法 + 一次边界检查。现在统一改成**按行落盘**：
//! 单参考 / intrabc 这两条纯"钳位 + 收窄"的直接走
//! [`super::simd::store_row_clamped`]（`min` + `packus` + 一次 16 字节写，
//! 一次 8 个样本），其余几条先在行缓冲里按原式算完、再整行
//! `copy_from_slice`（memcpy）进平面。
//!
//! 效果（单线程 40 帧、两二进制交替多轮取 min）：9818 → 8821 ms（**1.11x**）；
//! 采样里 `compute_prediction` 从 ~11% 降到 ~6%。真素材 20 帧输出**逐字节一致**。
//!
//! ## 性能（2026-10-02 第十一轮）：MC 中间缓冲与 OBMC scratch 的复用
//!
//! 三处"每次调用都白花"的开销，都不改结果：
//! 1. `block_inter_prediction` 的横向中间量原本是 `[0i32; 4096]` 的**栈数组** ——
//!    **每次调用都 memset 16KB**，哪怕块只有 4 像素宽；更大的块还退化成逐块 `vec![]`。
//!    现在由调用方（`BlockCtx::pred_scratch[2]`）持有、`scratch_of` **只增不减**地复用。
//! 2. `obmc_blend` / `predict_intrabc` 各自 `[Vec<i32>; 2] = [Vec::new(), ...]` 的
//!    **逐块**局部 scratch（OBMC 一帧约 2 千块）—— 同样改用 `pred_scratch`。
//! 3. `mv::project_frame` 把 7 个目标帧的 `ref_to_dst` / 裁剪后的分子放进了**逐格**
//!    循环（1080p：一格 7 次 × 3.2 万格 × 5 个源帧）—— 提到循环外，另加一个
//!    "已裁剪"的 `mv_projection_clipped`。
//!
//! 效果（8 线程、两二进制交替 3 轮取 min）：9360 → 8442 ms（**1.11x**，≈211 ms/帧）；
//! 真素材 20 帧逐字节一致。
//!
//! ## 性能（2026-10-02 第十二轮）：横向 MC 换 16 位通道
//!
//! `mc_horiz16_8bpc`（[`super::simd`]）：i16 通道 + **偏置累加**（[`super::simd::HORIZ16_BIAS`]），
//! 一次算 **16 个输出**（i32 版 8 个），10/12bit 仍走 i32 版。
//! 效果：8 线程 ≈187 ms/帧（较上一轮 1.07x），逐字节一致。
//!
//! ⚠️ 这一轮最初**算错了值域**（按 `Σ|coef| = 128` 而非真实最坏的 240），
//! 造成过像素级回归；根因、症状与修复见 `simd.rs` 模块头。
//!
//! ## 三个容易混的口径
//!
//! 1. **`MV` 是 `[行, 列]`**（`mv[0]` 管竖直、`mv[1]` 管水平），单位 1/8 亮度样本；
//!    色度要 `>> subX/subY`（先乘 2 再右移，规范原文 `(2*mv[1]) >> subX`）。
//! 2. **两种坐标同时存在**：`(x, y)` 是**当前帧**里预测块的样本坐标（写回用它），
//!    `(startX, startY, stepX, stepY)` 是**参考帧**里 1/1024 样本精度的采样位置
//!    （插值用它）。`stepX/stepY` 只在参考帧尺寸不同时才不是 1024。
//! 3. **亚像素相位**：`(p >> 6) & 15`（SCALE_SUBPEL_BITS=10 → 4 位相位）；
//!    8 抽头的 `t-3` 偏移与 `Clip3(0, lastX/Y, …)` 的钳位都不能省。
//!
//! 进度：单参考 + 复合（平均 / 楔形 / 距离 / 差分）+ warp（局部仿射 §7.11.3.5-8、
//! 全局运动 `useWarp = 2`）已实现。
//!
//! **OBMC 的像素侧已落地**（2026-09-30 深夜，`obmc_blend` + `obmc_overlap_predict`）：
//! `motion_mode == OBMC` 的块在整块预测写完之后，沿上边/左边各叠一条带（用邻居的 MV
//! 再预测一次，按 `Obmc_Mask_*` 混合）。**不做不会报错、只是静默的画质偏差** —— 真实素材上
//! 帧间帧的像素差 6.58% → 4.39%，可见这一层确实被大量用到。
//!
//! **`interintra` 的像素侧已落地**（2026-10-01，`interintra_predict_plane` +
//! `predict_inter` 里的混合支）：`RefFrame[1] == INTRA_FRAME` 的块先用
//! `interintra_mode` 映射出的帧内模式**整块**预测一次，再与帧间预测按
//! `Ii_Weights_1d`（非楔形）或 `WedgeMasks`（楔形）混合（规范 §7.11.2 +
//! §7.11.3.13 + §7.11.3.14）。`intrabc` / `COMPOUND_*` 之外的缺项仍然明确报错。
//!
//! 判据是**对拍**：`p_simple.mp4`（单参考、无 OBMC/warp）逐样本比特精确；
//! interintra 是 `tools/av1ref` 的 `aom_720_ii_f6`（1280×720 × 6 帧、`--cpu-used=2`）
//! 与 dav1d **逐像素一致**，且同一条流置 `AV1_NO_INTERINTRA=1` 后会开始出现差异
//! （反证它真的用到了 interintra，楔形与 smooth 两条路都有）。

use super::flags;
use super::simd;
use me_core::{MediaError, Result};

use super::block::{
    block_height, block_width, BlockGrid, BlockInfo, BILINEAR, DC_PRED, H_PRED, II_H_PRED,
    II_SMOOTH_PRED, II_V_PRED, OBMC, SMOOTH_PRED, V_PRED, BLOCK_8X8, NUM_4X4_BLOCKS_HIGH,
    NUM_4X4_BLOCKS_WIDE,
};
use super::coeff::plane_residual_size;
use super::frame_header::FrameHeader;
use super::intra::{predict_intra, CurrFrame, IntraBlock, Plane};
use super::partition::MiGrid;
use super::mv::{Mv, INTRA_FRAME, LAST_FRAME, MAX_FRAME_DISTANCE, REF_SCALE_SHIFT};
use super::sequence_header::SequenceHeader;
use super::warp;
use super::wedge::{uses_wedge, wedge_mask};

/// `SUBPEL_BITS`（§7.11.3.4）：亚像素相位的位数。
pub const SUBPEL_BITS: u32 = 4;
/// `SUBPEL_MASK` = `(1 << SUBPEL_BITS) - 1`。
pub const SUBPEL_MASK: i32 = (1 << SUBPEL_BITS) - 1;
/// `SCALE_SUBPEL_BITS`：`startX/stepX` 的小数位数。
pub const SCALE_SUBPEL_BITS: u32 = 10;
/// `FILTER_BITS`（§7.11.3.2 的 InterPostRound 用到）。
pub const FILTER_BITS: u32 = 7;
/// `MAX_SB_SIZE`（§6.10.15）：`interintra` 的 `Ii_Weights_1d` 按它定标。
pub const MAX_SB_SIZE: usize = 128;

/// `Ii_Weights_1d[MAX_SB_SIZE]`（§7.11.3.13）：非楔形 `interintra` 的混合权重。
///
/// 它是**一条从 60 单调降到 1 的斜坡**，索引越靠后越偏向帧内预测那一半
/// （掩码 `m` 乘的是帧内、`64 - m` 乘的是帧间）。`sizeScale = 128 / max(w, h)`
/// 把块的像素位置映射到这条 128 项的斜坡上。
pub const II_WEIGHTS_1D: [i32; MAX_SB_SIZE] = [
    60, 58, 56, 54, 52, 50, 48, 47, 45, 44, 42, 41, 39, 38, 37, 35, 34, 33, 32, 31, 30, 29, 28, 27,
    26, 25, 24, 23, 22, 22, 21, 20, 19, 19, 18, 18, 17, 16, 16, 15, 15, 14, 14, 13, 13, 12, 12, 12,
    11, 11, 10, 10, 10, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 4, 4, 4, 4,
    4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

/// 4=小块的 REGULAR/SHARP 4 抽头、5=小块的 SMOOTH 4 抽头。
pub const SUBPEL_FILTERS: [[[i16; 8]; 16]; 6] = [
    [
        [0, 0, 0, 128, 0, 0, 0, 0],
        [0, 2, -6, 126, 8, -2, 0, 0],
        [0, 2, -10, 122, 18, -4, 0, 0],
        [0, 2, -12, 116, 28, -8, 2, 0],
        [0, 2, -14, 110, 38, -10, 2, 0],
        [0, 2, -14, 102, 48, -12, 2, 0],
        [0, 2, -16, 94, 58, -12, 2, 0],
        [0, 2, -14, 84, 66, -12, 2, 0],
        [0, 2, -14, 76, 76, -14, 2, 0],
        [0, 2, -12, 66, 84, -14, 2, 0],
        [0, 2, -12, 58, 94, -16, 2, 0],
        [0, 2, -12, 48, 102, -14, 2, 0],
        [0, 2, -10, 38, 110, -14, 2, 0],
        [0, 2, -8, 28, 116, -12, 2, 0],
        [0, 0, -4, 18, 122, -10, 2, 0],
        [0, 0, -2, 8, 126, -6, 2, 0],
    ],
    [
        [0, 0, 0, 128, 0, 0, 0, 0],
        [0, 2, 28, 62, 34, 2, 0, 0],
        [0, 0, 26, 62, 36, 4, 0, 0],
        [0, 0, 22, 62, 40, 4, 0, 0],
        [0, 0, 20, 60, 42, 6, 0, 0],
        [0, 0, 18, 58, 44, 8, 0, 0],
        [0, 0, 16, 56, 46, 10, 0, 0],
        [0, -2, 16, 54, 48, 12, 0, 0],
        [0, -2, 14, 52, 52, 14, -2, 0],
        [0, 0, 12, 48, 54, 16, -2, 0],
        [0, 0, 10, 46, 56, 16, 0, 0],
        [0, 0, 8, 44, 58, 18, 0, 0],
        [0, 0, 6, 42, 60, 20, 0, 0],
        [0, 0, 4, 40, 62, 22, 0, 0],
        [0, 0, 4, 36, 62, 26, 0, 0],
        [0, 0, 2, 34, 62, 28, 2, 0],
    ],
    [
        [0, 0, 0, 128, 0, 0, 0, 0],
        [-2, 2, -6, 126, 8, -2, 2, 0],
        [-2, 6, -12, 124, 16, -6, 4, -2],
        [-2, 8, -18, 120, 26, -10, 6, -2],
        [-4, 10, -22, 116, 38, -14, 6, -2],
        [-4, 10, -22, 108, 48, -18, 8, -2],
        [-4, 10, -24, 100, 60, -20, 8, -2],
        [-4, 10, -24, 90, 70, -22, 10, -2],
        [-4, 12, -24, 80, 80, -24, 12, -4],
        [-2, 10, -22, 70, 90, -24, 10, -4],
        [-2, 8, -20, 60, 100, -24, 10, -4],
        [-2, 8, -18, 48, 108, -22, 10, -4],
        [-2, 6, -14, 38, 116, -22, 10, -4],
        [-2, 6, -10, 26, 120, -18, 8, -2],
        [-2, 4, -6, 16, 124, -12, 6, -2],
        [0, 2, -2, 8, 126, -6, 2, -2],
    ],
    [
        [0, 0, 0, 128, 0, 0, 0, 0],
        [0, 0, 0, 120, 8, 0, 0, 0],
        [0, 0, 0, 112, 16, 0, 0, 0],
        [0, 0, 0, 104, 24, 0, 0, 0],
        [0, 0, 0, 96, 32, 0, 0, 0],
        [0, 0, 0, 88, 40, 0, 0, 0],
        [0, 0, 0, 80, 48, 0, 0, 0],
        [0, 0, 0, 72, 56, 0, 0, 0],
        [0, 0, 0, 64, 64, 0, 0, 0],
        [0, 0, 0, 56, 72, 0, 0, 0],
        [0, 0, 0, 48, 80, 0, 0, 0],
        [0, 0, 0, 40, 88, 0, 0, 0],
        [0, 0, 0, 32, 96, 0, 0, 0],
        [0, 0, 0, 24, 104, 0, 0, 0],
        [0, 0, 0, 16, 112, 0, 0, 0],
        [0, 0, 0, 8, 120, 0, 0, 0],
    ],
    [
        [0, 0, 0, 128, 0, 0, 0, 0],
        [0, 0, -4, 126, 8, -2, 0, 0],
        [0, 0, -8, 122, 18, -4, 0, 0],
        [0, 0, -10, 116, 28, -6, 0, 0],
        [0, 0, -12, 110, 38, -8, 0, 0],
        [0, 0, -12, 102, 48, -10, 0, 0],
        [0, 0, -14, 94, 58, -10, 0, 0],
        [0, 0, -12, 84, 66, -10, 0, 0],
        [0, 0, -12, 76, 76, -12, 0, 0],
        [0, 0, -10, 66, 84, -12, 0, 0],
        [0, 0, -10, 58, 94, -14, 0, 0],
        [0, 0, -10, 48, 102, -12, 0, 0],
        [0, 0, -8, 38, 110, -12, 0, 0],
        [0, 0, -6, 28, 116, -10, 0, 0],
        [0, 0, -4, 18, 122, -8, 0, 0],
        [0, 0, -2, 8, 126, -4, 0, 0],
    ],
    [
        [0, 0, 0, 128, 0, 0, 0, 0],
        [0, 0, 30, 62, 34, 2, 0, 0],
        [0, 0, 26, 62, 36, 4, 0, 0],
        [0, 0, 22, 62, 40, 4, 0, 0],
        [0, 0, 20, 60, 42, 6, 0, 0],
        [0, 0, 18, 58, 44, 8, 0, 0],
        [0, 0, 16, 56, 46, 10, 0, 0],
        [0, 0, 14, 54, 48, 12, 0, 0],
        [0, 0, 12, 52, 52, 12, 0, 0],
        [0, 0, 12, 48, 54, 14, 0, 0],
        [0, 0, 10, 46, 56, 16, 0, 0],
        [0, 0, 8, 44, 58, 18, 0, 0],
        [0, 0, 6, 42, 60, 20, 0, 0],
        [0, 0, 4, 40, 62, 22, 0, 0],
        [0, 0, 4, 36, 62, 26, 0, 0],
        [0, 0, 2, 34, 62, 30, 0, 0],
    ],
];

/// 预测阶段要用的 warp 计划（规范 §7.11.3.1 步骤 2–6 里算出来的 `useWarp`）。
///
/// 这些量在**读块级语法时定不下来**（要看 `is_scaled`、要跑最小二乘），所以由
/// `decode_block` 算好后传下来；`predict_inter` 只负责"当前平面尺寸够不够"那一支。
#[derive(Debug, Clone, Copy, Default)]
pub struct WarpPlan {
    /// 每个参考列表的 `useWarp`：0 = 不用、1 = 局部仿射、2 = 全局运动
    pub use_warp: [u8; 2],
    /// `LocalWarpParams`（`use_warp[0] == 1` 时才有意义）
    pub local: [i32; 6],
}

/// 一帧的"参考帧像素视图"（`predict_inter` 需要的那几样）。
pub struct RefFrame {
    pub frame: CurrFrame,
    /// `RefUpscaledWidth`（= 存储时的亮度宽度）
    pub upscaled_width: u32,
    pub frame_height: u32,
}

/// `interintra` 的帧内那一半要用的**帧内可用性 / 边滤波上下文**。
///
/// 这些量由 `decode_block` 算好（都是 `&self` 上的信息），`compute_prediction` 只负责用。
/// 只有 `interintra` 的块会读；普通帧间块完全用不到。
#[derive(Debug, Clone, Copy, Default)]
pub struct IntraContext {
    /// `AvailU`（§6.10.4）——块正上方在**本瓦片内**且有样本
    pub avail_u: bool,
    /// `AvailL`
    pub avail_l: bool,
    /// `get_filter_type( plane = 0 )`（§7.11.2.8）——方向帧内预测的边滤波强度
    pub filter_type_y: u8,
    /// `get_filter_type( plane = 1 / 2 )`
    pub filter_type_uv: u8,
    /// row-mt 回放：`interintra` 的帧内那一半要的两个 `BlockDecoded` 查询，
    /// 由 `RowMtJobs::precompute_dec_flags` 按记录序预先算好（见 `intra::DecFlags`）。
    /// 就地重建（非 row-mt）时是 `None` ⇒ 照旧读运行时网格。
    pub dec_flags: Option<[super::intra::DecFlags; 3]>,
}

/// `compute_prediction( )` 的帧间部分（§7.11.2）。
///
/// 只在 `is_inter` 时调用；帧内块的预测仍在 `transform_block` 里按变换块做。
#[allow(clippy::too_many_arguments)]
pub fn compute_prediction(
    frame: &mut CurrFrame,
    grid: &BlockGrid,
    mi_grid: &MiGrid,
    seq: &SequenceHeader,
    fh: &FrameHeader,
    refs: &[Option<std::sync::Arc<CurrFrame>>; 8],
    info: &BlockInfo,
    r: i32,
    c: i32,
    warp: &WarpPlan,
    intra_ctx: &IntraContext,
    // 两路预测值 + MC 中间量的**复用**落点（由 `BlockCtx` 持有、**每瓦片一份**，
    // 见那里的说明；`[2]` 是 MC 的横向中间量缓冲）
    pred_bufs: &mut PredScratch,
) -> Result<()> {
    if info.use_intrabc {
        return predict_intrabc(frame, seq, fh, info, r, c, pred_bufs);
    }
    let bsize = info.mi_size;
    let mi_row = r;
    let mi_col = c;
    let num_planes = if seq.color.mono_chrome { 1 } else { 3 };
    let _sub_x_seq = u8::from(seq.color.subsampling_x);
    let _sub_y_seq = u8::from(seq.color.subsampling_y);

    // `COMPOUND_DIFFWTD` 的掩码**只在 plane 0 生成**（规范 §7.11.3.1 的
    // "…and plane is equal to 0"），色度复用同一份**亮度分辨率**的掩码。
    // 2026-10-04：那份掩码落在 `PredScratch::mask`（**跨块复用**，只按需增长），
    // 平面循环的顺序保证"plane 0 先生成、色度再读"。

    for plane in 0..num_planes as u8 {
        let sub_x = u8::from(plane > 0 && seq.color.subsampling_x);
        let sub_y = u8::from(plane > 0 && seq.color.subsampling_y);
        let plane_sz = plane_residual_size(bsize, plane, seq.color.subsampling_x, seq.color.subsampling_y);
        let num4x4_w = NUM_4X4_BLOCKS_WIDE[plane_sz as usize] as i32;
        let num4x4_h = NUM_4X4_BLOCKS_HIGH[plane_sz as usize] as i32;
        let base_x = (mi_col >> sub_x) * 4;
        let base_y = (mi_row >> sub_y) * 4;
        let mut cand_row = (mi_row >> sub_y) << sub_y;
        let mut cand_col = (mi_col >> sub_x) << sub_x;

        // 子 8×8 的色度特例：块内有一个帧内块 ⇒ 整块的色度预测用"右下角亮度块"的
        // 模式信息，且按 4×4（平面单位）逐块预测（规范 compute_prediction）
        let mut pred_w = block_width(bsize) >> sub_x;
        let mut pred_h = block_height(bsize) >> sub_y;
        // ⚠️ `someUseIntra` **只可能出现在 `MiSize < BLOCK_8X8` 的块上**（规范
        // §7.11.3.1 的"子 8×8"那一支）：≥ 8×8 的块整块走普通路径，块内根本不存在
        // "某几个 4×4 是帧内"这回事。
        //
        // 原来**无条件**扫整块 —— 128×128 的块要扫 32×32 = **1024 个 4×4 单元 ×
        // 3 个平面**（每次 `grid.idx` 一次乘加 + 一次随机读）。真素材里 128×128 有
        // 602 次/8 帧、64×64 有 2649 次，**这是 MC 外层最大的一笔白工**
        // （1636 次调用/帧、每次最多上万周期）。
        let mut some_use_intra = false;
        if (bsize as usize) < super::partition::BLOCK_8X8 {
            'intra_scan: for rr in 0..(num4x4_h << sub_y) {
                for cc in 0..(num4x4_w << sub_x) {
                    let gi = grid.idx(cand_row + rr, cand_col + cc);
                    if grid.ref_frames[gi][0] == INTRA_FRAME {
                        some_use_intra = true;
                        break 'intra_scan;
                    }
                }
            }
        }
        if some_use_intra {
            pred_w = num4x4_w * 4;
            pred_h = num4x4_h * 4;
            cand_row = mi_row;
            cand_col = mi_col;
        }
        if pred_w <= 0 || pred_h <= 0 {
            return Err(MediaError::Bitstream(format!(
                "帧间预测的块尺寸非法：{pred_w}×{pred_h}"
            )));
        }

        // `interintra`：先把**帧内那一半**整块预测进 CurrFrame（规范 compute_prediction
        // 在 inter 之前的那一段）。下面的 `predict_inter` 再把它与帧间预测按掩码混合。
        // 顺序不能反 —— 混合时读的就是这里写下的帧内像素。
        // 隔离实验用：`AV1_NO_INTERINTRA=1` 时**跳过**这一整条 interintra 路径
        // （帧内那一半不预测、下面也不混合，块退化成纯帧间预测），与 `AV1_NO_OBMC`
        // 同一套路。interintra 是静默的画质偏差：不把它摘掉，"这条流用没用 interintra"
        // 与"用错了"在像素差上是分不清的；摘掉之后，与 dav1d 的差异图就正好圈出
        // interintra 块的位置。
        if info.interintra && !flags::no_interintra() {
            let ft = if plane == 0 { intra_ctx.filter_type_y } else { intra_ctx.filter_type_uv };
            // interintra 的块只可能是 8×8 … 32×32（`read_interintra_mode` 的条件），
            // 于是 §6.10.4 里 `AvailUChroma`/`AvailLChroma` 的两个特例（`bw4 == 1` /
            // `bh4 == 1`）都取不到 —— 色度的可用性与亮度一样，直接传同一份。
            interintra_predict_plane(
                frame, seq, info, plane, base_x, base_y, pred_w, pred_h,
                intra_ctx.avail_u, intra_ctx.avail_l, ft,
                intra_ctx.dec_flags.map(|f| f[plane as usize]),
            )?;
        }

        let mut rr = 0i32;
        let mut y = 0i32;
        while y < num4x4_h * 4 {
            let mut cc = 0i32;
            let mut x = 0i32;
            while x < num4x4_w * 4 {
                predict_inter(
                    frame, grid, seq, fh, refs, plane, base_x + x, base_y + y, pred_w, pred_h,
                    cand_row + rr, cand_col + cc, info, bsize, warp, pred_bufs,
                )?;
                cc += 1;
                x += pred_w;
            }
            rr += 1;
            y += pred_h;
        }

        // 规范 §7.11.3.1 的收尾：`motion_mode == OBMC` 的块要再叠加一次
        // 重叠运动补偿（§7.11.3.9）。**每个平面做完自己的整块预测之后再做**。
        // OBMC 只出现在单参考块上（`read_motion_mode` 已把复合/ref1 排除）。
        if info.motion_mode == OBMC && !flags::no_obmc() {
            let is_compound = info.ref_frame[1] > INTRA_FRAME;
            let (ir0, ir1, _) = rounding_vars(seq.color.bit_depth, is_compound);
            obmc_blend(
                frame, grid, mi_grid, seq, fh, refs, plane, pred_w, pred_h, bsize, mi_row, mi_col,
                ir0, ir1, pred_bufs,
            )?;
        }
    }
    Ok(())
}

/// `use_intrabc == 1` 的整块预测：**参考就是当前帧已重建的那部分**（规范
/// §7.11.3.1 里 `refIdx = -1`、§7.11.3.4 里 `ref = CurrFrame`）。
///
/// 与普通帧间的三处不同，都是规范专门开的口子：
///
/// 1. `mv_scale` 的参考尺寸取 `FrameWidth / FrameHeight / UpscaledWidth`
///    （`refIdx = -1` 那一档），于是缩放系数恒为 1，MV 就是纯粹的整像素位移；
/// 2. `block_inter_prediction` 的裁剪边界换成 `MiCols*MI_SIZE` / `MiRows*MI_SIZE`
///    —— 规范原话是"免得 intrabc 的预测被裁到帧边界"；
/// 3. **源与目的落在同一份像素里**，所以必须"先把整块算出来、再写回"。
///    边算边写会把还没读到的源像素覆盖掉。
///
/// 普通帧间的三种花活在这里都不存在：没有复合（`isCompound = 0`）、没有 OBMC
/// （`motion_mode = SIMPLE`）、没有 warp（帧内帧的 `force_integer_mv` 让
/// `warp_plan` 直接返回全 0）。
///
/// 隔离实验：`AV1_NO_INTRABC=1` 时**整块的拷贝都不做**（不报错），于是"这条流到底
/// 用没用 intrabc"就变成一次差异对拍 —— 与 `AV1_NO_OBMC` 同一套路。只看正着跑
/// 一致是分不清"没用"与"用对了"的。
fn predict_intrabc(
    frame: &mut CurrFrame,
    seq: &SequenceHeader,
    fh: &FrameHeader,
    info: &BlockInfo,
    mi_row: i32,
    mi_col: i32,
    // `[0]` = 预测值、`[2]` = MC 横向中间量（都由 `BlockCtx` 逐瓦片复用）
    pred_bufs: &mut PredScratch,
) -> Result<()> {
    if flags::no_intrabc() {
        return Ok(());
    }
    let bsize = info.mi_size;
    let subsampling_x = seq.color.subsampling_x;
    let subsampling_y = seq.color.subsampling_y;
    // 规范 `compute_prediction` 的平面循环上界是 `1 + HasChroma * 2`：`HasChroma = 0`
    // 的块（4:2:0 下 4×4 / 8×4 / 4×8 且落在 8×8 单元的"上/左半"）**不做色度** ——
    // 那个单元的色度整体归它右下角那个 4×4 块做（`candRow/candCol` 也因此指过去）。
    let bw4 = NUM_4X4_BLOCKS_WIDE[bsize as usize] as i32;
    let bh4 = NUM_4X4_BLOCKS_HIGH[bsize as usize] as i32;
    let has_chroma = frame.num_planes > 1
        && !(bh4 == 1 && subsampling_y && (mi_row & 1) == 0)
        && !(bw4 == 1 && subsampling_x && (mi_col & 1) == 0);
    let bit_depth = seq.color.bit_depth;
    let max = (1i32 << bit_depth) - 1;
    // 单参考、非复合 ⇒ 与普通帧间同一组舍入量
    let (ir0, ir1, _) = rounding_vars(bit_depth, false);
    let mv = info.mv[0];
    // 预测值的**复用**落点（三个平面依次复用同一块，见 `scratch_of`）
    let PredScratch { preds, mc: mc_scratch, .. } = pred_bufs;
    for plane in 0..if has_chroma { 3u8 } else { 1u8 } {
        let sub_x = sub_x_of(plane, seq);
        let sub_y = sub_y_of(plane, seq);
        // 尺寸按规范 `someUseIntra` 那一支来：intrabc 块在 `RefFrames` 网格里被记成
        // `INTRA_FRAME`（§5.11.5 起、`splat_intrabc_mv` 同），所以 `predW/predH`
        // 恒取 `plane_residual_size` 的 4×4 粒度（`num4x4W*4` / `num4x4H*4`），
        // **不是**"亮度尺寸 >> sub"：子 8×8 的块（4×4 / 8×4 / 4×8）色度被钳到
        // 4×4，整格由右下角那个块预测。用 `>> sub` 只会写左上 2×2，剩下的像素
        // 永远停在 128（未重建），而且会被后面的 intrabc 拷贝沿参考链放大。
        let plane_sz = plane_residual_size(bsize, plane, subsampling_x, subsampling_y);
        let w = NUM_4X4_BLOCKS_WIDE[plane_sz as usize] as i32 * 4;
        let h = NUM_4X4_BLOCKS_HIGH[plane_sz as usize] as i32 * 4;
        let x = (mi_col >> sub_x) * 4;
        let y = (mi_row >> sub_y) * 4;
        if w <= 0 || h <= 0 {
            continue;
        }
        // 规范里 intrabc 的裁剪边界是 `MiCols*MI_SIZE` / `MiRows*MI_SIZE`（按平面
        // 折一下），可以比 `FrameWidth` / `FrameHeight` **大**（最多差 7）。
        // 我们的当前帧平面只开到可见尺寸，没有那段按 MI 对齐的填充，所以再取一次
        // min —— 只有帧宽/高不是 8 的倍数时两者才不等（与 DEVELOPMENT.md 里"帧右边界超块
        // 伸出区"是同一类已知问题）。
        let spec_last_x = ((fh.mi_cols as i64) * 4 + i64::from(sub_x)) >> sub_x;
        let spec_last_y = ((fh.mi_rows as i64) * 4 + i64::from(sub_y)) >> sub_y;
        let bounds = (
            spec_last_x.min(frame.planes[plane as usize].w as i64) - 1,
            spec_last_y.min(frame.planes[plane as usize].h as i64) - 1,
        );
        let (start_x, start_y, step_x, step_y) =
            mv_scale(frame, fh, plane, x, y, mv, sub_x, sub_y)?;
        let out = scratch_of(&mut preds[0], (w * h) as usize);
        block_inter_prediction(
            out,
            &frame.planes[plane as usize], plane, start_x, start_y, step_x, step_y, w, h, BILINEAR,
            ir0, ir1, Some(bounds), mc_scratch,
        )?;
        let pred: &Vec<i32> = &preds[0];
        let dst = &mut frame.planes[plane as usize];
        // 按行落盘（同 `predict_inter` 的单参考路径）：`pred` 是先整块算好的
        // （源与目的同一份像素，边算边写会覆盖还没读的源）。
        let row_len = w as usize;
        let simd = super::simd::avx2_available();
        for i in 0..h as usize {
            let off = (y as usize + i) * dst.stride + x as usize;
            let row = &mut dst.px16_mut()[off..off + row_len];
            if simd {
                // SAFETY: 已判 AVX2；`row` 长 `w`、`pred` 至少 `w·h` 项。
                unsafe {
                    super::simd::store_row_clamped(row.as_mut_ptr(), &pred[i * row_len..], row_len, max)
                };
            } else {
                for (j, slot) in row.iter_mut().enumerate() {
                    *slot = pred[i * row_len + j].clamp(0, max) as u16;
                }
            }
        }
    }
    Ok(())
}

/// `predict_inter( plane, x, y, w, h, candRow, candCol )`（§7.11.3.1）。
#[allow(clippy::too_many_arguments)]
fn predict_inter(
    frame: &mut CurrFrame,
    grid: &BlockGrid,
    seq: &SequenceHeader,
    fh: &FrameHeader,
    refs: &[Option<std::sync::Arc<CurrFrame>>; 8],
    plane: u8,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    cand_row: i32,
    cand_col: i32,
    info: &BlockInfo,
    bsize: u8,
    warp: &WarpPlan,
    // 两路预测值 + MC 中间量的**复用**落点（由 `compute_prediction` 持有，见那里的说明）
    bufs: &mut PredScratch,
) -> Result<()> {
    let gi = grid.idx(cand_row, cand_col);
    let ref0 = grid.ref_frames[gi][0];
    if ref0 == INTRA_FRAME {
        // 子 8×8 的"某几个 4×4 是帧内"：那些 4×4 的预测已经在 transform_block 里
        // 做过（帧内路径），这里跳过。
        return Ok(());
    }
    let is_compound = grid.ref_frames[gi][1] > INTRA_FRAME;
    let bit_depth = seq.color.bit_depth;
    let (inter_round0, inter_round1, inter_post_round) = rounding_vars(bit_depth, is_compound);

    // 每个参考列表各做一遍块帧间预测（规范 §7.11.3.1 的步骤 5–13 重复两次）
    let lists = if is_compound { 2usize } else { 1usize };
    // 两路预测值落在**调用方给的复用缓冲**里（`compute_prediction` 持有、跨块复用）：
    // 每块一次 `vec![0i32; w*h]` 是 malloc + memset 两次开销，而 1080p 一帧几万块 × 1~2 路。
    let PredScratch { preds, mc: mc_scratch, mask: dw_mask } = bufs;
    let preds: &mut [Vec<i32>] = &mut preds[..lists];
    for list in 0..lists {
        let ref_frame_type = grid.ref_frames[gi][list];
        if ref_frame_type == INTRA_FRAME {
            return Err(MediaError::Bitstream(format!(
                "AV1 复合预测的第 {list} 个参考是帧内（不该出现在这里）"
            )));
        }
        let ref_idx = ref_slot(fh, ref_frame_type)?;
        let ref_frame = refs[ref_idx as usize].as_ref().ok_or_else(|| {
            MediaError::Bitstream(format!(
                "参考槽 {ref_idx} 还没有像素（解码顺序或 seek 问题）"
            ))
        })?;
        // 规范 §7.11.3.1 步骤 6 的第一支：**判的是当前平面的尺寸**。
        // 4:2:0 下 8×8 的块色度只有 4×4 ⇒ 色度不 warp、走普通缩放预测。
        let use_warp = if w < 8 || h < 8 || flags::no_warp() {
            0
        } else {
            // 细粒度证伪旋钮：`use_warp == 1` 是块级 LOCALWARP，`== 2` 是全局运动。
            match warp.use_warp[list] {
                1 if flags::no_localwarp() => 0,
                2 if flags::no_gmwarp() => 0,
                other => other,
            }
        };
        if use_warp != 0 {
            let params = if use_warp == 1 {
                &warp.local
            } else {
                &fh.params.global_motion.params[ref_frame_type as usize]
            };
            let out = scratch_of(&mut preds[list], (w * h) as usize);
            warp::block_warp(
                out,
                &ref_frame.planes[plane as usize],
                sub_x_of(plane, seq),
                sub_y_of(plane, seq),
                x,
                y,
                w,
                h,
                params,
                inter_round0,
                inter_round1,
            )?;
            continue;
        }
        let mv = grid.mvs[gi][list];
        let filter = grid.interp_filters[gi][list];
        let (start_x, start_y, step_x, step_y) = mv_scale(
            ref_frame, fh, plane, x, y, mv, sub_x_of(plane, seq), sub_y_of(plane, seq),
        )?;
        if plane == 0 && flags::trace_pred() {
            eprintln!(
                "PRED r={cand_row} c={cand_col} bs={bsize} mv=({},{}) ref0={} ref1={} filt={filter} w={w} h={h} start=({start_x},{start_y}) step=({step_x},{step_y})",
                mv[0], mv[1], ref0, grid.ref_frames[gi][1]
            );
        }
        let out = scratch_of(&mut preds[list], (w * h) as usize);
        block_inter_prediction(
            out,
            &ref_frame.planes[plane as usize], plane, start_x, start_y, step_x, step_y, w, h, filter,
            inter_round0, inter_round1, None, mc_scratch,
        )?;
        if plane == 0 && flags::trace_pred() {
            eprintln!(
                "POUT start=({},{}) phase=({},{}) wh=({},{}) out={}",
                start_x >> 10, start_y >> 10, (start_x >> 6) & 15, (start_y >> 6) & 15,
                w, h, preds[list][0].clamp(0, 255)
            );
        }
    }

    let dst = &mut frame.planes[plane as usize];
    let max = (1i32 << bit_depth) - 1;
    if !is_compound {
        if info.interintra && !flags::no_interintra() {
            // `interintra`：帧内那一半已经在 `compute_prediction` 里整块写好，
            // 这里按掩码把它与帧间预测混起来（§7.11.3.13 + §7.11.3.14）。
            blend_interintra_region(
                dst, &preds[0], grid, gi, seq, info, plane, x, y, w, h, inter_post_round, max,
            )?;
            return Ok(());
        }
        // 按行落盘（原来逐像素 `Plane::set`：每像素一次下标计算 + 边界检查）。
        // 内点行 ≥ 8 时走 AVX2 内核（`min` + `packus` + 一次 16 字节写）。
                let row_len = w as usize;
        let simd = super::simd::avx2_available();
        for i in 0..h as usize {
            let off = (y as usize + i) * dst.stride + x as usize;
            let row = &mut dst.px16_mut()[off..off + row_len];
            if simd {
                // SAFETY: 已判 AVX2；`row` 长 `w`、`preds[0]` 至少 `w·h` 项。
                unsafe {
                    super::simd::store_row_clamped(
                        row.as_mut_ptr(),
                        &preds[0][i * row_len..],
                        row_len,
                        max,
                    )
                };
            } else {
                for (j, slot) in row.iter_mut().enumerate() {
                    *slot = preds[0][i * row_len + j].clamp(0, max) as u16;
                }
            }
        }
        
        return Ok(());
    }

    // 复合：按 `compound_type` 混合两路预测（规范 §7.11.3.14）
    //
    // 注意读的是 `compound_types`（`COMPOUND_*`），**不是** `compound_idxs`
    // （那只是 0/1 的语法元素，取值 0 恰好撞上 `COMPOUND_WEDGE`）—— 2026-09-30
    // 拿真实素材才发现这里读错了字段。
    let compound_type = grid.compound_types[gi];
    // 诊断（`AV1_TRACE_CMP_AT=lx,ly`）：把**指定色度像素**所在复合块的两路预测值与
    // 混合结果打出来 —— 用来判断"差异出在预测(MC)还是混合"。
    let cmp_at: Option<(i32, i32)> = if plane > 0 { flags::trace_cmp_at() } else { None };
    let cmp_here = cmp_at.filter(|(tx, ty)| *tx >= x && *tx < x + w && *ty >= y && *ty < y + h);
    // 诊断（`AV1_TRACE_CMP16=1`）：把**每个复合色度块**的两路 d16 预测各自求和打出来。
    // 与 libaom 的 `aom_lowbd_blend_a64_d16_mask_c` 里同一条轨迹按**解码序**对齐 ——
    // 谁先不一致，就是色度 MC 出错的那个块。
    if plane > 0
        && is_compound
        && matches!(
            compound_type,
            super::block::COMPOUND_WEDGE | super::block::COMPOUND_DIFFWTD
        )
        && flags::trace_cmp16()
    {
        let sum = |v: &Vec<i32>| v.iter().map(|x| i64::from(*x)).sum::<i64>();
        eprintln!(
            "CMP16 oh={} p={plane} x={x} y={y} w={w} h={h} ct={compound_type} s0={} s1={}",
            fh.order_hint,
            sum(&preds[0]),
            preds.get(1).map(sum).unwrap_or(0)
        );
    }
    if let Some((tx, ty)) = cmp_here {
        let k = ((ty - y) * w + (tx - x)) as usize;
        eprintln!(
            "CMPAT-IN plane={plane} blk=({x},{y},{w}x{h}) ct={compound_type} k={k} p0={} p1={} ipr={inter_post_round}",
            preds[0][k],
            preds.get(1).map(|v| v[k]).unwrap_or(-1)
        );
    }
    // 掩码混合的取权重方式：掩码是**亮度分辨率**的，色度要按下采样规则折一下。
    let sub_x = sub_x_of(plane, seq);
    let sub_y = sub_y_of(plane, seq);
    match compound_type {
        super::block::COMPOUND_AVERAGE => {
            // `Clip1(Round2(p0 + p1, 1 + InterPostRound))`
            let shift = 1 + inter_post_round;
            let row_len = w as usize;
            let simd = super::simd::avx2_available();
            for i in 0..h as usize {
                let k0 = i * row_len;
                let off = (y as usize + i) * dst.stride + x as usize;
                if simd {
                    // SAFETY: 已判 AVX2；`preds` 至少 `w·h` 项、`dst.data` 这一行有 `w` 个可写样本
                    unsafe {
                        super::simd::blend_avg8(
                            dst.px16_mut().as_mut_ptr().add(off),
                            preds[0][k0..].as_ptr(),
                            preds[1][k0..].as_ptr(),
                            row_len,
                            shift,
                            max,
                        )
                    };
                } else {
                    let row = &mut dst.px16_mut()[off..off + row_len];
                    for (j, slot) in row.iter_mut().enumerate() {
                        let v = super::intra::round2(preds[0][k0 + j] + preds[1][k0 + j], shift)
                            .clamp(0, max);
                        *slot = v as u16;
                    }
                }
            }
        }
        super::block::COMPOUND_DISTANCE => {
            // 距离权重（§7.11.3.15）：按两个参考帧到当前帧的 `order_hint` 距离定权。
            // 权重的**和恒为 16**，所以 `>> (4 + InterPostRound)` 刚好除回像素范围。
            let (fwd, bck) = distance_weights(seq, fh, grid, gi);
            let shift = 4 + inter_post_round;
            let row_len = w as usize;
            let simd = super::simd::avx2_available();
            for i in 0..h as usize {
                let k0 = i * row_len;
                let off = (y as usize + i) * dst.stride + x as usize;
                if simd {
                    // SAFETY: 已判 AVX2；同 `COMPOUND_AVERAGE` 那一支
                    unsafe {
                        super::simd::blend_dist8(
                            dst.px16_mut().as_mut_ptr().add(off),
                            preds[0][k0..].as_ptr(),
                            preds[1][k0..].as_ptr(),
                            fwd,
                            bck,
                            row_len,
                            shift,
                            max,
                        )
                    };
                } else {
                    let row = &mut dst.px16_mut()[off..off + row_len];
                    for (j, slot) in row.iter_mut().enumerate() {
                        let v = super::intra::round2(
                            fwd * preds[0][k0 + j] + bck * preds[1][k0 + j],
                            shift,
                        )
                        .clamp(0, max);
                        *slot = v as u16;
                    }
                }
            }
        }
        super::block::COMPOUND_WEDGE => {
            if !uses_wedge(bsize as usize) {
                return Err(MediaError::Bitstream(format!(
                    "AV1 的 COMPOUND_WEDGE 用在 {bsize} 号块上（这个尺寸没有楔形掩码）"
                )));
            }
            let sign = grid.wedge_signs[gi];
            let wedge = grid.wedge_idxs[gi] as usize;
            let weight = |ly: usize, lx: usize| wedge_mask(bsize as usize, sign, wedge, ly, lx) as i32;
            blend_masked(dst, preds, x, y, w, h, sub_x, sub_y, inter_post_round, max, weight);
        }
        super::block::COMPOUND_DIFFWTD => {
            // 掩码只由**亮度**的预测差生成（§7.11.3.12 + §7.11.3.1 的 plane == 0）
            let luma_w = (w << sub_x) as usize;
            let luma_h = (h << sub_y) as usize;
            if plane == 0 {
                build_diffwtd_mask(
                    preds,
                    w,
                    h,
                    bit_depth,
                    inter_post_round,
                    grid.mask_types[gi],
                    dw_mask,
                );
                // 诊断（`AV1_TRACE_DIFFWTD_AT=lx,ly`，亮度坐标）：把该像素 2×2 的
                // `|Δ| / diff / 掩码` 打出来 —— 掩码是一个**台阶函数**，±1 的预测差
                // 在阈值附近就能让掩码跳 1（亮度看不出来，色度看得出来）。
                if let Some((lx, ly)) = flags::trace_diffwtd_at() {
                    if lx >= x && lx < x + w && ly >= y && ly < y + h {
                        let dm: &[i32] = dw_mask.as_slice();
                        for dy in 0..2i32 {
                            for dx in 0..2i32 {
                                let ly2 = ly + dy - (ly & 1);
                                let lx2 = lx + dx - (lx & 1);
                                if ly2 < y || ly2 >= y + h || lx2 < x || lx2 >= x + w {
                                    continue;
                                }
                                let k = ((ly2 - y) * w + (lx2 - x)) as usize;
                                let absd = (preds[0][k] - preds[1][k]).abs();
                                let d = round2_i64(absd as i64, inter_post_round);
                                eprintln!(
                                    "DIFFWTD luma=({lx2},{ly2}) p0={} p1={} |d|={absd} diff={d} m={}",
                                    preds[0][k], preds[1][k], dm[k]
                                );
                            }
                        }
                    }
                }
            }
            if dw_mask.len() != luma_w * luma_h {
                return Err(MediaError::Bitstream(format!(
                    "AV1 的 COMPOUND_DIFFWTD 掩码尺寸对不上：{} != {luma_w}×{luma_h}",
                    dw_mask.len()
                )));
            }
            let mask: &[i32] = dw_mask;
            let weight = |ly: usize, lx: usize| mask[ly * luma_w + lx];
            blend_masked(dst, preds, x, y, w, h, sub_x, sub_y, inter_post_round, max, weight);
        }
        other => {
            return Err(MediaError::Unsupported(format!(
                "AV1 的复合预测缺 COMPOUND_INTRA / interintra（这一块的 compound_type = {other}）"
            )));
        }
    }
    if let Some((tx, ty)) = cmp_here {
        eprintln!(
            "CMPAT-OUT plane={plane} ({tx},{ty}) val={}",
            dst.get(tx as usize, ty as usize)
        );
    }
    Ok(())
}

// ── 重叠运动补偿（OBMC，规范 §7.11.3.9 / §7.11.3.10）────────────────────
//
// `motion_mode == OBMC` 的块，在**自己的整块预测写完之后**，沿**上边**与**左边**
// 各叠一条带：用**邻居块的 MV** 再预测一次（`predict_overlap`），再按 `Obmc_Mask_*`
// 的权重把自己与邻居的预测混起来。
//
// 这一步不做**不会报错**，只是这些块的边缘少一次混合 —— 属**静默的画质偏差**。
// 真实素材（B 站 ASMR 1080p）上，修好 `read_motion_mode` 之后 OBMC 块开始大量出现
// （8 帧里约 2000 块），帧间帧的像素差从"只有环路恢复那点"涨到 6.6% 就是这么来的。

/// `Obmc_Mask_*`（§7.11.3.9）。权重满值是 64（= 1.0），越小越偏邻居的预测。
const OBMC_MASK_2: [u8; 2] = [45, 64];
const OBMC_MASK_4: [u8; 4] = [39, 50, 59, 64];
const OBMC_MASK_8: [u8; 8] = [36, 42, 48, 53, 57, 61, 64, 64];
const OBMC_MASK_16: [u8; 16] = [
    34, 37, 40, 43, 46, 49, 52, 54, 56, 58, 60, 61, 64, 64, 64, 64,
];
const OBMC_MASK_32: [u8; 32] = [
    33, 35, 36, 38, 40, 41, 43, 44, 45, 47, 48, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 60, 61, 62,
    64, 64, 64, 64, 64, 64, 64, 64,
];

/// `get_obmc_mask( length )`（§7.11.3.9）。
fn obmc_mask(length: i32) -> &'static [u8] {
    match length {
        2 => &OBMC_MASK_2,
        4 => &OBMC_MASK_4,
        16 => &OBMC_MASK_16,
        32 => &OBMC_MASK_32,
        _ if length <= 8 => &OBMC_MASK_8,
        _ => &OBMC_MASK_32,
    }
}

/// `Overlapped motion compensation process`（§7.11.3.9）。
#[allow(clippy::too_many_arguments)]
fn obmc_blend(
    frame: &mut CurrFrame,
    grid: &BlockGrid,
    mi_grid: &MiGrid,
    seq: &SequenceHeader,
    fh: &FrameHeader,
    refs: &[Option<std::sync::Arc<CurrFrame>>; 8],
    plane: u8,
    w: i32,
    h: i32,
    bsize: u8,
    mi_row: i32,
    mi_col: i32,
    inter_round0: u32,
    inter_round1: u32,
    pred_bufs: &mut PredScratch,
) -> Result<()> {
    // 规范 §7.11.3.9 开头这个 `get_plane_residual_size( MiSize, plane ) >= BLOCK_8X8` §7.11.3.9 开头这个 `get_plane_residual_size( MiSize, plane ) >= BLOCK_8X8`
    // **只挡\"上边\"那一支**（原文把它写在 `if ( AvailU )` 块内部），\"左边\"那一支无条件做。
    // libaom 的 `av1_skip_u4x4_pred_in_obmc( bsize, pd, dir )` 写得更清楚：
    // `BLOCK_4X4 / 8X4 / 4X8` 时返回 `dir == 0`，即**只有上边跳过**。
    // 这里曾经把它做成整体早退、两个方向都挡 —— 结果是子 8×8 的色度（平面块 4×4/4×8/8×4）
    // **左边那条整条漏叠**，真素材上表现为断断续续的 ±1~2。
    let plane_sz =
        plane_residual_size(bsize, plane, seq.color.subsampling_x, seq.color.subsampling_y);
    if (plane_sz as usize) >= super::partition::BLOCK_SIZES {
        // `BLOCK_INVALID`：OBMC 不会出现在这种块上（`is_motion_variation_allowed_bsize`）。
        return Ok(());
    }
    let above_ok = (plane_sz as usize) >= BLOCK_8X8;
    let sub_x = sub_x_of(plane, seq) as i32;
    let sub_y = sub_y_of(plane, seq) as i32;
    let w4 = NUM_4X4_BLOCKS_WIDE[bsize as usize] as i32;
    let h4 = NUM_4X4_BLOCKS_HIGH[bsize as usize] as i32;
    let mi_rows = fh.mi_rows as i32;
    let mi_cols = fh.mi_cols as i32;
    let max = (1i32 << seq.color.bit_depth) - 1;
    // 邻块预测的插值滤波器取的是**候选邻居自己的** `InterpFilters[candRow][candCol]`
    // （规范 §7.11.3.4：`block_inter_prediction` 的 `candRow/candCol` 就是用来查这两张表的，
    // OBMC 传进去的是邻居块坐标）。**不是**当前块的滤波器 —— 这里踩过坑：
    // 用当前块的滤波器时，被 SMOOTH 滤波器邻居覆盖的那几条边会差 ±1~2。
    // OBMC 只在单参考块上出现，取 list 0 即可。

    // 邻居那一路预测值的**复用**落点：OBMC 每个块最多沿两条边各取 4 个邻居，
    // 每块一次 `vec![]` 同样是白花（见 `scratch_of`）。直接用调用方的三块缓冲：
    // `[0]` 放邻居预测、`[2]` 当 MC 的横向中间量。
    let PredScratch { preds, mc: mc_scratch, .. } = pred_bufs;
    let neighbor_pred: &mut Vec<i32> = &mut preds[0];

    // pass 0 = 上边（`AvailU`）、pass 1 = 左边（`AvailL`）。
    for pass in 0..2u8 {
        if pass == 0 {
            // 上边：既要有上方邻居，又要过 `>= BLOCK_8X8` 那个门（见上）。
            if mi_row == 0 || !above_ok {
                continue;
            }
        } else if mi_col == 0 {
            continue;
        }
        let n_limit = if pass == 0 { w4.ilog2() as i32 } else { h4.ilog2() as i32 }.min(4);
        // 沿块边游走的坐标（上边那趟走列、左边那趟走行）
        let mut run = if pass == 0 { mi_col } else { mi_row };
        let run_end = if pass == 0 {
            (mi_col + w4).min(mi_cols)
        } else {
            (mi_row + h4).min(mi_rows)
        };
        let mut n_count = 0i32;
        while n_count < n_limit && run < run_end {
            let (cand_row, cand_col) =
                if pass == 0 { (mi_row - 1, run | 1) } else { (run | 1, mi_col - 1) };
            let cand_sz = mi_grid.get(cand_row, cand_col);
            // 邻居格没写过时 `MiGrid::get` 返回 `BLOCK_INVALID`（= `BLOCK_SIZES`），
            // 而块尺寸表只有 `BLOCK_SIZES` 项 —— 直接下标会越界 panic。
            if (cand_sz as usize) >= super::partition::BLOCK_SIZES {
                break;
            }
            let step4 = if pass == 0 {
                NUM_4X4_BLOCKS_WIDE[cand_sz as usize] as i32
            } else {
                NUM_4X4_BLOCKS_HIGH[cand_sz as usize] as i32
            }
            .clamp(2, 16);
            let gi = grid.idx(cand_row, cand_col);
            if grid.ref_frames[gi][0] > INTRA_FRAME {
                n_count += 1;
                let (pred_w, pred_h, mask) = if pass == 0 {
                    let ph = (h >> 1).min(32 >> sub_y);
                    (w.min((step4 * 4) >> sub_x), ph, obmc_mask(ph))
                } else {
                    let pw = (w >> 1).min(32 >> sub_x);
                    (pw, h.min((step4 * 4) >> sub_y), obmc_mask(pw))
                };
                // `predict_overlap()`：predX/predY 是**块**在该平面里的像素坐标
                let (pred_x, pred_y) = if pass == 0 {
                    ((run * 4) >> sub_x, (mi_row * 4) >> sub_y)
                } else {
                    ((mi_col * 4) >> sub_x, (run * 4) >> sub_y)
                };
                obmc_overlap_predict(
                    frame, grid, seq, fh, refs, plane, cand_row, cand_col, pred_x, pred_y, pred_w,
                    pred_h, mask, pass, grid.interp_filters[gi][0], inter_round0, inter_round1, max,
                    neighbor_pred, mc_scratch,
                )?;
            }
            run += step4;
        }
    }
    Ok(())
}

/// `predict_overlap()` + `Overlap blending process`（§7.11.3.9 收尾 / §7.11.3.10）。
#[allow(clippy::too_many_arguments)]
fn obmc_overlap_predict(
    frame: &mut CurrFrame,
    grid: &BlockGrid,
    seq: &SequenceHeader,
    fh: &FrameHeader,
    refs: &[Option<std::sync::Arc<CurrFrame>>; 8],
    plane: u8,
    cand_row: i32,
    cand_col: i32,
    pred_x: i32,
    pred_y: i32,
    pred_w: i32,
    pred_h: i32,
    mask: &[u8],
    pass: u8,
    filter: u8,
    inter_round0: u32,
    inter_round1: u32,
    max: i32,
    // 邻居那一路预测值的**复用**落点（`obmc_blend` 持有，沿边的每个邻居复用）
    neighbor_pred: &mut Vec<i32>,
    // MC 的横向中间量缓冲（同一份，逐块复用）
    mc_scratch: &mut Vec<i16>,
) -> Result<()> {
    let gi = grid.idx(cand_row, cand_col);
    let ref_frame_type = grid.ref_frames[gi][0];
    let ref_idx = ref_slot(fh, ref_frame_type)?;
    let ref_frame = refs[ref_idx as usize]
        .as_ref()
        .ok_or_else(|| MediaError::Bitstream(format!("OBMC 的参考槽 {ref_idx} 还没有像素")))?;
    let mv = grid.mvs[gi][0];
    let (start_x, start_y, step_x, step_y) = mv_scale(
        ref_frame,
        fh,
        plane,
        pred_x,
        pred_y,
        mv,
        sub_x_of(plane, seq),
        sub_y_of(plane, seq),
    )?;
    let out = scratch_of(neighbor_pred, (pred_w * pred_h) as usize);
    block_inter_prediction(
        out,
        &ref_frame.planes[plane as usize], plane, start_x, start_y, step_x, step_y, pred_w, pred_h,
        filter, inter_round0, inter_round1, None, mc_scratch,
    )?;
    let obmc_pred: &Vec<i32> = neighbor_pred;
    if plane == 0 && flags::trace_pred() {
        eprintln!(
            "OPRED start=({},{}) phase=({},{}) wh=({},{}) out={}",
            start_x >> 10, start_y >> 10, (start_x >> 6) & 15, (start_y >> 6) & 15,
            pred_w, pred_h, obmc_pred[0].clamp(0, 255)
        );
    }

    let dst = &mut frame.planes[plane as usize];
    let row_len = pred_w as usize;
    // 掩码按**行**铺开成 i32（pass 0 是一整行同一个值、pass 1 逐列），混合交给 AVX2
    // 内核（`simd::blend_masked_u16in8`：`p0` 就地取平面、`p1` 先钳到 `max`）。
    let mut mrow = [0i32; MAX_SB_SIZE];
    let simd = super::simd::avx2_available();
    for i in 0..pred_h as usize {
        let off = (pred_y as usize + i) * dst.stride + pred_x as usize;
        let m = if pass == 0 {
            mrow[..row_len].fill(i32::from(mask[i]));
            mrow[..row_len].as_ptr()
        } else {
            for (j, slot) in mrow[..row_len].iter_mut().enumerate() {
                *slot = i32::from(mask[j]);
            }
            mrow.as_ptr()
        };
        if simd {
            // SAFETY: 已判 AVX2；`dst.data` 这一行有 `pred_w` 个可读写样本、
            // `obmc_pred` 至少 `pred_w·pred_h` 项、`m` 指向 `pred_w` 个 i32。
            unsafe {
                super::simd::blend_masked_u16in8(
                    dst.px16_mut().as_mut_ptr().add(off),
                    dst.px16().as_ptr().add(off),
                    obmc_pred[i * row_len..].as_ptr(),
                    m,
                    row_len,
                    6,
                    max,
                    0,
                    max,
                )
            };
        } else {
            let row = &mut dst.px16_mut()[off..off + row_len];
            for (j, slot) in row.iter_mut().enumerate() {
                let mm = if pass == 0 { i32::from(mask[i]) } else { i32::from(mask[j]) };
                let cur = i32::from(*slot);
                let ob = obmc_pred[i * row_len + j].clamp(0, max);
                // `Round2( m * Cur + (64 - m) * obmcPred, 6 )`
                let v = ((mm * cur + (64 - mm) * ob) + 32) >> 6;
                *slot = v.clamp(0, max) as u16;
            }
        }
    }
    Ok(())
}

/// `Difference weight mask process`（§7.11.3.12）：按两路预测的差生成**亮度**掩码。
///
/// 差越大越信第一路（`m` 越大）；`mask_type` 为 1 时取补。
fn build_diffwtd_mask(
    preds: &[Vec<i32>],
    w: i32,
    h: i32,
    bit_depth: u8,
    inter_post_round: u32,
    mask_type: u8,
    mask: &mut Vec<i32>,
) {
    let shift = (bit_depth as u32).saturating_sub(8) + inter_post_round;
    let n = (w * h) as usize;
    // 只在**需要变长**时清零/扩张（`truncate` 不写内存）；下面两条路都会写满 `[0, n)`。
    if mask.len() < n {
        mask.resize(n, 0);
    }
    mask.truncate(n);
    if super::simd::avx2_available() {
        // SAFETY: 已判 AVX2；`preds[0]/[1]` 长度恰为 `w*h`（`scratch_of` 的契约）、
        // `mask` 长度恰为 `n`、`shift >= 1`（`inter_post_round >= 4`）。
        unsafe {
            super::simd::diffwtd_mask8(&preds[0][..n], &preds[1][..n], n, shift, mask_type, mask);
        }
        return;
    }
    for k in 0..n {
        let diff = super::intra::round2((preds[0][k] - preds[1][k]).abs(), shift);
        let m = (38 + diff / 16).clamp(0, 64);
        mask[k] = if mask_type != 0 { 64 - m } else { m };
    }
}

/// `Mask blend process`（§7.11.3.14 的非 interintra 分支）：
/// `Clip1(Round2(m * p0 + (64 - m) * p1, 6 + InterPostRound))`。
///
/// `weight(ly, lx)` 按**亮度**坐标给掩码值（楔形是现算、DIFFWTD 是查表）；
/// 色度由本函数按下采样规则折算：`subX && !subY` 取横向 2 点平均，否则取 2×2 平均。
#[allow(clippy::too_many_arguments)]
fn blend_masked(
    dst: &mut super::intra::Plane,
    preds: &[Vec<i32>],
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    sub_x: u8,
    sub_y: u8,
    inter_post_round: u32,
    max: i32,
    weight: impl Fn(usize, usize) -> i32,
) {
    let shift = 6 + inter_post_round;
    let at = |ly: usize, lx: usize| weight(ly, lx);
    let row_len = w as usize;
    // 掩码按**行**先算进小缓冲（色度要 2×2/2×1 折算，这一段必须逐像素、照规范），
    // 再把"乘加 + 取整 + 钳位 + 落盘"交给 AVX2 内核（`simd::blend_masked8`）。
    // 一半以上的复合块是 DIFFWTD/WEDGE，逐像素乘加原来占整帧 ~5 ms。
    let mut mrow = [0i32; MAX_SB_SIZE];
    let simd = super::simd::avx2_available();
    for i in 0..h as usize {
        for (j, slot) in mrow[..row_len].iter_mut().enumerate() {
            *slot = if sub_x == 0 && sub_y == 0 {
                at(i, j)
            } else if sub_x == 1 && sub_y == 0 {
                super::intra::round2(at(i, 2 * j) + at(i, 2 * j + 1), 1)
            } else {
                super::intra::round2(
                    at(2 * i, 2 * j)
                        + at(2 * i, 2 * j + 1)
                        + at(2 * i + 1, 2 * j)
                        + at(2 * i + 1, 2 * j + 1),
                    2,
                )
            };
        }
        let k0 = i * row_len;
        let off = (y as usize + i) * dst.stride + x as usize;
        if simd {
            // SAFETY: 已判 AVX2；`preds` 至少 `w·h` 项、`mrow` 至少 `w` 项、
            // `dst.data` 在这一行有 `w` 个可写样本（块在平面内）。
            unsafe {
                super::simd::blend_masked8(
                    dst.px16_mut().as_mut_ptr().add(off),
                    preds[0][k0..].as_ptr(),
                    preds[1][k0..].as_ptr(),
                    mrow.as_ptr(),
                    row_len,
                    shift,
                    max,
                )
            };
        } else {
            let row = &mut dst.px16_mut()[off..off + row_len];
            for (j, slot) in row.iter_mut().enumerate() {
                let m = mrow[j];
                let v = super::intra::round2(m * preds[0][k0 + j] + (64 - m) * preds[1][k0 + j], shift)
                    .clamp(0, max);
                *slot = v as u16;
            }
        }
    }
}

/// `interintra` 的**帧内那一半**：整块按 `interintra_mode` 映射出的帧内模式预测一次
/// （规范 §7.11.2 `compute_prediction` 里 `IsInterIntra` 那一段）。
///
/// 与 `intra::predict_intra_txb` 的差别：那条路是**按变换块**预测并自己推
/// `haveAbove`/`haveAboveRight`；这里是**整块**、可用性取自 `AvailU`/`AvailL`，
/// `haveAboveRight`/`haveBelowLeft` 直接用 `BlockDecoded` 查（规范原文那两个下标）。
#[allow(clippy::too_many_arguments)]
fn interintra_predict_plane(
    frame: &mut CurrFrame,
    seq: &SequenceHeader,
    info: &BlockInfo,
    plane: u8,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    avail_u: bool,
    avail_l: bool,
    filter_type: u8,
    // row-mt：预先算好的 `BlockDecoded` 两个查询（`None` = 照旧读运行时网格）
    flags: Option<super::intra::DecFlags>,
) -> Result<()> {
    // 规范 compute_prediction 的映射：II_DC→DC、II_V→V、II_H→H、其余→SMOOTH。
    let mode = match info.interintra_mode {
        II_V_PRED => V_PRED,
        II_H_PRED => H_PRED,
        II_SMOOTH_PRED => SMOOTH_PRED,
        _ => DC_PRED,
    };
    // `BlockDecoded[ plane ][ (subBlockMiRow >> subY) - 1 ][ (subBlockMiCol >> subX) + num4x4W ]`
    // 与它的下左对称式。`frame.decoded` 用的是**帧级平面 4×4 坐标**，就等价于这两个下标。
    let row4 = (y >> 2) as i64;
    let col4 = (x >> 2) as i64;
    let (have_above_right, have_below_left) = match flags {
        Some(f) => (f.above_right, f.below_left),
        None => (
            frame.decoded(plane, row4 - 1, col4 + (w >> 2) as i64),
            frame.decoded(plane, row4 + (h >> 2) as i64, col4 - 1),
        ),
    };

    let (max_x, max_y) = frame.plane_dims(plane);
    let bit_depth = frame.plane(plane).bit_depth;
    let ib = IntraBlock {
        plane,
        x: x as usize,
        y: y as usize,
        w: w as usize,
        h: h as usize,
        have_left: avail_l,
        have_above: avail_u,
        have_above_right,
        have_below_left,
        mode,
        // interintra 把 AngleDelta / filter_intra 都强制清零（见 `read_interintra_mode`）
        angle_delta: 0,
        use_filter_intra: false,
        filter_intra_mode: 0,
        enable_intra_edge_filter: seq.enable_intra_edge_filter,
        filter_type,
        max_x: max_x.saturating_sub(1),
        max_y: max_y.saturating_sub(1),
        bit_depth,
    };
    predict_intra(frame.plane_mut(plane), &ib)
}

/// `interintra` 的**混和**：判楔形 / 非楔形两路，各备一份掩码后交给
/// [`blend_interintra_core`]。
#[allow(clippy::too_many_arguments)]
fn blend_interintra_region(
    dst: &mut super::intra::Plane,
    inter: &[i32],
    grid: &BlockGrid,
    gi: usize,
    seq: &SequenceHeader,
    info: &BlockInfo,
    plane: u8,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    inter_post_round: u32,
    bit_max: i32,
) -> Result<()> {
    let sub_x = sub_x_of(plane, seq);
    let sub_y = sub_y_of(plane, seq);
    if info.wedge_interintra {
        // 楔形（`compound_type == COMPOUND_WEDGE`）：掩码是**亮度分辨率**的
        // （§7.11.3.11），色度按掩码混合的下采样规则折一下。
        let bsize = info.mi_size;
        if !uses_wedge(bsize as usize) {
            return Err(MediaError::Bitstream(format!(
                "AV1 的 wedge interintra 用在 {bsize} 号块上（这个尺寸没有楔形掩码）"
            )));
        }
        let sign = grid.wedge_signs[gi];
        let wedge = grid.wedge_idxs[gi] as usize;
        blend_interintra_core(
            dst, inter, x, y, w, h, sub_x, sub_y, false, inter_post_round, bit_max,
            |ly, lx| wedge_mask(bsize as usize, sign, wedge, ly, lx) as i32,
        );
    } else {
        // 非楔形（`compound_type == COMPOUND_INTRA`）：掩码按**当前平面**分辨率
        // （§7.11.3.13 的 `Ii_Weights_1d`）—— 注意这里的 `w`/`h` 已经是平面尺寸。
        let size_scale = MAX_SB_SIZE / (w.max(h) as usize);
        let mode = info.interintra_mode;
        blend_interintra_core(
            dst, inter, x, y, w, h, sub_x, sub_y, true, inter_post_round, bit_max,
            |i, j| match mode {
                II_V_PRED => II_WEIGHTS_1D[i * size_scale],
                II_H_PRED => II_WEIGHTS_1D[j * size_scale],
                II_SMOOTH_PRED => II_WEIGHTS_1D[i.min(j) * size_scale],
                _ => 32,
            },
        );
    }
    Ok(())
}

/// `Mask blend process` 的 `interintra` 支（规范 §7.11.3.14）：
///
/// ```text
/// pred0 = Clip1( Round2( preds[0][y][x], InterPostRound ) )   // 帧间那一半
/// pred1 = CurrFrame[plane][y+dy][x+dx]                        // 帧内那一半（已写好）
/// CurrFrame[...] = Round2( m * pred1 + (64 - m) * pred0, 6 )
/// ```
///
/// **注意这条公式与普通复合不同**：掩码 `m` 乘的是**帧内**预测，而且结果不再 `Clip1`
/// （加权值天然落在像素范围内）。`weight(ly, lx)` 给的是掩码值；`per_plane_mask` 为真
/// 表示掩码已经是**当前平面**分辨率的（非楔形），为假表示是亮度分辨率的（楔形），
/// 后者要按规范那个 `interintra && !wedge_interintra` 的例外规则给色度降采样。
#[allow(clippy::too_many_arguments)]
fn blend_interintra_core(
    dst: &mut super::intra::Plane,
    inter: &[i32],
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    sub_x: u8,
    sub_y: u8,
    per_plane_mask: bool,
    inter_post_round: u32,
    bit_max: i32,
    weight: impl Fn(usize, usize) -> i32,
) {
    let row_len = w as usize;
    // 掩码按行算好（色度要 2×2/2×1 折算），乘加交给 AVX2 内核：`p0` 就地取平面
    // （帧内那半已由 `interintra_predict_plane` 写好），`p1` 是帧间那半
    // （内核里先 `Round2(·, InterPostRound)` 再钳到 `bit_max`，与原来逐像素一致）。
    let mut mrow = [0i32; MAX_SB_SIZE];
    let simd = super::simd::avx2_available();
    for i in 0..h as usize {
        for (j, slot) in mrow[..row_len].iter_mut().enumerate() {
            *slot = if per_plane_mask || (sub_x == 0 && sub_y == 0) {
                weight(i, j)
            } else if sub_x == 1 && sub_y == 0 {
                super::intra::round2(weight(i, 2 * j) + weight(i, 2 * j + 1), 1)
            } else {
                super::intra::round2(
                    weight(2 * i, 2 * j)
                        + weight(2 * i, 2 * j + 1)
                        + weight(2 * i + 1, 2 * j)
                        + weight(2 * i + 1, 2 * j + 1),
                    2,
                )
            };
        }
        let off = (y as usize + i) * dst.stride + x as usize;
        let k0 = i * row_len;
        if simd {
            // SAFETY: 已判 AVX2；`dst.data` 这一行有 `w` 个可读写样本、
            // `inter` 至少 `w·h` 项、`mrow` 至少 `w` 项。
            unsafe {
                super::simd::blend_masked_u16in8(
                    dst.px16_mut().as_mut_ptr().add(off),
                    dst.px16().as_ptr().add(off),
                    inter[k0..].as_ptr(),
                    mrow.as_ptr(),
                    row_len,
                    6,
                    i32::MAX,
                    inter_post_round,
                    bit_max,
                )
            };
        } else {
            let row = &mut dst.px16_mut()[off..off + row_len];
            for (j, slot) in row.iter_mut().enumerate() {
                let m = mrow[j];
                let pred0 =
                    super::intra::round2(inter[k0 + j], inter_post_round).clamp(0, bit_max);
                let pred1 = i32::from(*slot);
                let v = super::intra::round2(m * pred1 + (64 - m) * pred0, 6);
                *slot = v as u16;
            }
        }
    }
}

/// `Rounding variables derivation process`（§7.11.3.2）：返回
/// `(InterRound0, InterRound1, InterPostRound)`。
///
/// **这三个数是"预测值带多少位小数"的唯一来源**，复合与否会换一整套：非复合时
/// `InterPostRound = 0`（所以单参考直接就是像素值），复合时 `InterRound1 = 7`
/// ⇒ 预测值多带 `InterPostRound` 位小数，混合公式必须把它除回去。
fn rounding_vars(bit_depth: u8, is_compound: bool) -> (u32, u32, u32) {
    let mut round0 = 3u32;
    let mut round1 = if is_compound { 7 } else { 11 };
    if bit_depth == 12 {
        round0 += 2;
        if !is_compound {
            round1 -= 2;
        }
    }
    (round0, round1, 2 * FILTER_BITS - (round0 + round1))
}

/// `Distance weights process`（§7.11.3.15）：返回 `(FwdWeight, BckWeight)`。
///
/// `dist[refList]` 是"参考帧的 `order_hint` 与当前帧 `order_hint` 的距离"（钳到 31）；
/// `order` 表示 `d0 = dist[1] <= d1 = dist[0]`。两张表的用法很别扭（`d0` 配 `dist[1]`），
/// 照抄就对了。
fn distance_weights(
    seq: &SequenceHeader,
    fh: &FrameHeader,
    grid: &BlockGrid,
    gi: usize,
) -> (i32, i32) {
    let mut dist = [0i32; 2];
    for list in 0..2 {
        let h = fh.order_hints[grid.ref_frames[gi][list] as usize];
        dist[list] = super::mv::get_relative_dist(seq, h, fh.order_hint)
            .abs()
            .clamp(0, MAX_FRAME_DISTANCE);
    }
    let (d0, d1) = (dist[1], dist[0]);
    dist_weights_for(d0, d1)
}

/// `Distance weights process` 里"由 `d0`/`d1` 选一行表"的那段（§7.11.3.15）。
fn dist_weights_for(d0: i32, d1: i32) -> (i32, i32) {
    let order = usize::from(d0 <= d1);
    let i = if d0 == 0 || d1 == 0 {
        3
    } else {
        let mut i = 0usize;
        while i < 3 {
            let c0 = QUANT_DIST_WEIGHT[i][order];
            let c1 = QUANT_DIST_WEIGHT[i][1 - order];
            if order == 1 {
                if d0 * c0 > d1 * c1 {
                    break;
                }
            } else if d0 * c0 < d1 * c1 {
                break;
            }
            i += 1;
        }
        i
    };
    (QUANT_DIST_LOOKUP[i][order], QUANT_DIST_LOOKUP[i][1 - order])
}

/// `Quant_Dist_Weight[4][2]`（§7.11.3.15）。第 4 行（`{1, 31}`）规范里没用到。
const QUANT_DIST_WEIGHT: [[i32; 2]; 4] = [[2, 3], [2, 5], [2, 7], [1, MAX_FRAME_DISTANCE]];
/// `Quant_Dist_Lookup[4][2]`（§7.11.3.15）：每一行的两个权重**和为 16**。
const QUANT_DIST_LOOKUP: [[i32; 2]; 4] = [[9, 7], [11, 5], [12, 4], [13, 3]];

/// 参考帧类型（LAST/GOLDEN/…）→ 参考槽号（`ref_frame_idx`）。
fn ref_slot(fh: &FrameHeader, ref_frame: i8) -> Result<u8> {
    if ref_frame < LAST_FRAME {
        return Err(MediaError::Bitstream("帧间块给了帧内参考".into()));
    }
    let idx = (ref_frame - LAST_FRAME) as usize;
    if idx >= fh.ref_frame_idx.len() {
        return Err(MediaError::Bitstream(format!("参考帧类型越界：{ref_frame}")));
    }
    Ok(fh.ref_frame_idx[idx])
}

fn sub_x_of(plane: u8, seq: &SequenceHeader) -> u8 {
    u8::from(plane > 0 && seq.color.subsampling_x)
}
fn sub_y_of(plane: u8, seq: &SequenceHeader) -> u8 {
    u8::from(plane > 0 && seq.color.subsampling_y)
}

/// `Motion vector scaling process`（§7.11.3.3）：返回 `(startX, startY, stepX, stepY)`。
#[allow(clippy::too_many_arguments)]
fn mv_scale(
    ref_frame: &CurrFrame,
    fh: &FrameHeader,
    plane: u8,
    x: i32,
    y: i32,
    mv: Mv,
    sub_x: u8,
    sub_y: u8,
) -> Result<(i64, i64, i64, i64)> {
    let ref_w = ref_frame.planes[0].w as i64;
    let ref_h = ref_frame.planes[0].h as i64;
    let frame_w = fh.upscaled_width as i64;
    let frame_h = fh.frame_height as i64;
    // **同尺寸快路径**（2026-10-04）：真素材里没有超分/尺寸差 ⇒ 两个 i64 除法
    // （每次调用都付）是白花。同尺寸时按规范式子代数上就是 `1 << REF_SCALE_SHIFT`：
    // `((w<<10) + w/2) / w = 1024 + (w/2)/w = 1024`（`w/2 < w` ⇒ 整数除法那项为 0）。
    let (x_scale, y_scale) = if ref_w == frame_w && ref_h == frame_h {
        (1i64 << REF_SCALE_SHIFT, 1i64 << REF_SCALE_SHIFT)
    } else {
        (
            ((ref_w << REF_SCALE_SHIFT) + (frame_w / 2)) / frame_w,
            ((ref_h << REF_SCALE_SHIFT) + (frame_h / 2)) / frame_h,
        )
    };
    let (sub_x, sub_y) = (sub_x as i32, sub_y as i32);
    let half_sample = 1i64 << (SUBPEL_BITS - 1);
    let orig_x = ((x as i64) << SUBPEL_BITS) + (((2 * mv[1]) as i64) >> sub_x) + half_sample;
    let orig_y = ((y as i64) << SUBPEL_BITS) + (((2 * mv[0]) as i64) >> sub_y) + half_sample;
    let base_x = orig_x * x_scale - (half_sample << REF_SCALE_SHIFT);
    let base_y = orig_y * y_scale - (half_sample << REF_SCALE_SHIFT);
    let off = (1i64 << (SCALE_SUBPEL_BITS - SUBPEL_BITS)) / 2;
    let shift = (REF_SCALE_SHIFT + SUBPEL_BITS - SCALE_SUBPEL_BITS) as i32;
    // Round2Signed（向远离零的方向取整）
    let round2s = |v: i64| -> i64 {
        if v < 0 { -((-v + (1 << (shift - 1))) >> shift) } else { (v + (1 << (shift - 1))) >> shift }
    };
    let start_x = round2s(base_x) + off;
    let start_y = round2s(base_y) + off;
    let step_shift = (REF_SCALE_SHIFT - SCALE_SUBPEL_BITS) as i32;
    let step2s = |v: i64| -> i64 {
        if v < 0 { -((-v + (1 << (step_shift - 1))) >> step_shift) } else { (v + (1 << (step_shift - 1))) >> step_shift }
    };
    let step_x = step2s(x_scale);
    let step_y = step2s(y_scale);
    let _ = plane;
    Ok((start_x, start_y, step_x, step_y))
}

/// 把一块**可复用**缓冲对齐到 `n` 个 `i32` 并交回切片。
///
/// **长度没变时一次分配、一次填充都不做**（连续同尺寸的块是常态）；变长时才
/// `clear + resize` —— 那一下的填充值是死数据（下面每条路径都会写满 `[0, n)`），
/// 但 `resize` 必须给个初值，所以写成 `resize(n, 0)`（一次 memset，比 malloc 便宜）。
///
/// 用它的地方都是"逐块调用、每块要一块 `w×h` 的预测值"：`block_inter_prediction`、
/// `warp::block_warp`、OBMC 的邻居预测。原来每处都 `vec![0i32; w*h]`，
/// 1080p 一帧几万块 × 1~2 路 ⇒ malloc 与 `bzero` 各几万次。
/// 帧间预测的**复用缓冲**（`BlockCtx::pred_scratch` 持有，生命周期 = 瓦片）。
///
/// 原来是 `[Vec<i32>; 3]`：`[0]/[1]` 是两路预测值、`[2]` 是 MC 的横向中间量。
/// 2026-10-04 把 `[2]` 拆出来改成 **i16** —— 8bit 的 `round2(Σc·s, inter_round0)`
/// 值域只有 ±4.5e3、12bit 最坏 30712，**都装得下 i16**；而 128 宽的块那一份中间量
/// 从 **69 KB 降到 34.5 KB**（i32 时超出 L1，是 MC 最大的单项开销）。
/// dav1d 的 `prep_8tap_c` 用的正是 `int16_t mid[128*135]`。
pub struct PredScratch {
    /// 两路预测值（i32；`compute_prediction` 按 `lists` 用前 1~2 个）
    pub preds: [Vec<i32>; 2],
    /// MC 的横向中间量（i16，见上面的说明）
    pub mc: Vec<i16>,
    /// `COMPOUND_DIFFWTD` 的**亮度**掩码（§7.11.3.12）。
    ///
    /// 2026-10-04 从"每块一次 `vec![0i32; w*h]`"改成**跨块复用**：真素材每帧
    /// 189 个 DIFFWTD 块、共 66 万像素 ⇒ 原来每帧 2.58 MB 的分配 + 清零。
    /// 只在**长度需要增长**时清零（`truncate` + 按需 `resize`），其余时候整段被覆写。
    pub mask: Vec<i32>,
}

impl Default for PredScratch {
    fn default() -> Self {
        Self {
            preds: [Vec::new(), Vec::new()],
            mc: Vec::new(),
            mask: Vec::new(),
        }
    }
}

#[inline]
fn scratch_of(v: &mut Vec<i32>, n: usize) -> &mut [i32] {
    // **只增不减**：变短时不清、变长时一次 `resize`（填充值是死数据，每条路径都会
    // 写满 `[0, n)`）；`resize` 只在**首次遇到更大的块**时发生，不再随块尺寸来回
    // memset。
    if v.len() < n {
        v.resize(n, 0);
    }
    &mut v[..n]
}

/// [`scratch_of`] 的 i16 版（MC 的横向中间量，见 [`PredScratch`]）。
#[inline]
fn scratch_of_i16(v: &mut Vec<i16>, n: usize) -> &mut [i16] {
    if v.len() < n {
        v.resize(n, 0);
    }
    &mut v[..n]
}

/// 纵向趟（`h × wu`）：`Σ_t fy[t] · inter[(r+t)·wu + c]` 再 `Round2(·, inter_round1)`。
///
/// 抽出来是因为**两条横向路径都要用它**（8bit 走整块内核、10/12bit 走逐行），
/// 复制一份迟早会漂。
///
/// ⚠️ `interp_rows` 是**显式传进来的**，不在这里 `intermediate.len() / wu` 现算：
/// 那是一次**变量除数**的整数除法（~25 周期），而本函数每个预测块调一次
/// （1080p 一帧 3 万多次 ⇒ 约 0.5 ms/帧）。调用方本来就知道这个值
/// （`interp_h`），直接给就行。
#[allow(clippy::too_many_arguments)]
fn mc_vertical_pass(
    intermediate: &[i16],
    dst: &mut [i32],
    wu: usize,
    h: usize,
    interp_rows: usize,
    fy: &[i16; 8],
    inter_round1: u32,
    simd: bool,
) -> Result<()> {
    let rows = interp_rows;
    // 原实现是逐行判 `r + 8 > rows`；`r` 最大 `h-1` ⇒ 等价于开头判一次 `h + 7 > rows`。
    if h + 7 > rows {
        return Err(MediaError::Bitstream("帧间预测的中间缓冲越界".into()));
    }
    if simd {
        debug_assert_eq!(dst.len(), wu * h);
        // SAFETY: 已判 AVX2；上面已保证 `intermediate` 至少 `(h+7) * wu` 项
        //（`mc_vert_rows_8bpc` 要读到第 `r+7` 行）、`dst` 恰好 `h * wu` 项。
        unsafe { simd::mc_vert_rows_8bpc(intermediate, wu, h, fy, inter_round1, dst.as_mut_ptr()) };
        return Ok(());
    }
    for r in 0..h {
        let out = &mut dst[r * wu..r * wu + wu];
        for (c, slot) in out.iter_mut().enumerate() {
            let mut s = 0i32;
            for t in 0..8usize {
                s += fy[t] as i32 * intermediate[(r + t) * wu + c] as i32;
            }
            *slot = round2_i32(s, inter_round1);
        }
    }
    Ok(())
}

/// `Block inter prediction process`（§7.11.3.4）：两级 8 抽头插值。
///
/// `bounds` 覆盖 `(lastX, lastY)`：规范算的是
/// `lastX = ((RefUpscaledWidth + subX) >> subX) - 1`、
/// `lastY = ((RefFrameHeight + subY) >> subY) - 1`，普通帧间的两个量正好等于
/// **参考平面自己的**行宽/行高减 1（所以传 `None` 由本函数照平面算），
/// 而 `use_intrabc` 时规范把这两个量改成了 `MiCols*MI_SIZE` / `MiRows*MI_SIZE`
/// （原话："免得 intrabc 的预测被裁到帧边界"）—— 那种情况由调用方传 `Some`。
#[allow(clippy::too_many_arguments)]
fn block_inter_prediction(
    dst: &mut [i32],
    plane_data: &Plane,
    _plane: u8,
    start_x: i64,
    start_y: i64,
    step_x: i64,
    step_y: i64,
    w: i32,
    h: i32,
    filter: u8,
    inter_round0: u32,
    inter_round1: u32,
    bounds: Option<(i64, i64)>,
    // 横向中间量的**复用**缓冲（由调用方持有，见 `scratch_of`）。
    // 原来这里是 `[0i32; 4096]` 的栈数组：**每次调用都 memset 16KB**，
    // 哪怕块只有 4 像素宽；超过 4096 项的大块还退化成逐块 `vec![]`。
    mc_scratch: &mut Vec<i16>,
) -> Result<()> {
        let (last_x, last_y) = bounds.unwrap_or((plane_data.w as i64 - 1, plane_data.h as i64 - 1));
    let clip = |v: i64, hi: i64| -> i64 { v.clamp(0, hi) };
    let stride = plane_data.stride;
    let data = plane_data.px16();
    let wu = w as usize;
    debug_assert_eq!(dst.len(), wu * h as usize, "帧间预测的输出缓冲长度不对");

    // ── 快路径 1：整数相位（两个方向都不缩放、相位都是 0）──────────────────
    //
    // 8 抽头滤波器在相位 0 就是 δ（`SUBPEL_FILTERS[..][0] == [0,0,0,128,0,0,0,0]`，
    // 见下面 `phase_zero_is_a_delta_for_every_filter` 这条单测）：横向
    // `(128·s) >> 3 = 16s`、纵向 `(128·16s + 1024) >> 11 = s`，两趟取整**恰好还原**
    // ⇒ 预测就是"按 MV 取源样本"。真素材里整像素 MV 占绝大多数，dav1d 也为这一档
    // 留了专门的拷贝路径（我们原来对每一块都跑完整的 8 抽头两趟卷积，纯属白算）。
    // 横向中间量的行数（`interp_h = ceil((h-1)·stepY / 1024) + 8`）
    let interp_h: usize = ((((h as i64 - 1) * step_y + (1 << SCALE_SUBPEL_BITS) - 1)
        >> SCALE_SUBPEL_BITS)
        + 8) as usize;
    let no_scale = step_x == (1 << SCALE_SUBPEL_BITS) && step_y == (1 << SCALE_SUBPEL_BITS);
    if no_scale {
        let x0 = start_x >> SCALE_SUBPEL_BITS;
        let y0 = start_y >> SCALE_SUBPEL_BITS;
        let phase_x = ((start_x >> 6) & SUBPEL_MASK as i64) as usize;
        let phase_y = ((start_y >> 6) & SUBPEL_MASK as i64) as usize;
        let filt_h = subpel_index(filter, w <= 4)?;
        let filt_v = subpel_index(filter, h <= 4)?;
        debug_assert_eq!(SUBPEL_FILTERS[filt_h][0], DELTA_FILTER);
        debug_assert_eq!(SUBPEL_FILTERS[filt_v][0], DELTA_FILTER);
        if phase_x == 0 && phase_y == 0 {
            // 整块都在帧内（没有 clip）时才能走 SIMD —— 边界块回落标量。
            let interior = x0 >= 0
                && y0 >= 0
                && x0 + wu as i64 - 1 <= last_x
                && y0 + h as i64 - 1 <= last_y;
            let simd = interior && simd::avx2_available();
            // 两趟 `Round2` 在本路径上**恒等于一次左移**（`src << post_round`，
            // 推导与全值域验证见 `simd::mc_shift8` 的说明）。用化简后的内核，
            // 每 8 个通道省掉两轮 `slli`+`add`+`srai`。
            let post_round = 2 * FILTER_BITS - (inter_round0 + inter_round1);
            // **整块一次调**（2026-10-05 第三十七轮）：原来是逐行调 `mc_shift_row`，
            // 而它是 `#[target_feature]` ⇒ 不能被这里内联 ⇒ 1080p 一帧约 20 万次
            // 真实函数调用。与第二十九轮 `mc_horiz_rows_8bpc` 同一个改法。
            let c_end = if simd {
                // SAFETY: 已判 AVX2；行下标由内核 `clip(y0 + r, last_y)` 现算
                // （每一行都在 `data` 内）、`dst` 恰好 `h * wu` 项。
                debug_assert!((h as usize) <= MAX_SB_SIZE, "整数相位快路径的块高超出最大块");
                unsafe {
                    simd::mc_shift_rows(
                        data,
                        stride,
                        y0,
                        last_y,
                        h as usize,
                        x0 as isize,
                        post_round,
                        wu,
                        dst.as_mut_ptr(),
                    )
                }
            } else {
                0
            };
            for r in 0..h as usize {
                let sy = clip(y0 + r as i64, last_y) as usize;
                let row = &data[sy * stride..];
                let out = &mut dst[r * wu..r * wu + wu];
                let mut c = c_end;
                while c < wu {
                    let sx = clip(x0 + c as i64, last_x) as usize;
                    // ⚠️ 这里**不能**直接写 `row[sx]`：δ 只保证"每趟只剩乘 128"，
                    // 两趟的取整量 `InterRound0/1` 是随位深与复合与否变的
                    // （8bit 非复合 3/11、8bit 复合 3/7、12bit 5/9 与 5/7）——
                    // 复合时纵向输出是 `16·src`（后面混合时再除回来）。
                    // 必须照原公式走两趟取整，才与一般路径逐位一致。
                    let h1 = round2_i32(128 * row[sx] as i32, inter_round0);
                    out[c] = round2_i32(128 * h1, inter_round1);
                    c += 1;
                }
            }
            
            return Ok(());
        }

        // ── 快路径 2：不缩放、但相位非 0 ────────────────────────────────
        //
        // 相位在两轴上都是**常数** ⇒ 系数只有一行/一列（不必逐像素查表），
        // 源坐标就是 `base + 下标`（`step == 1<<SCALE_SUBPEL_BITS` 时
        // `(start + 1024·i) >> 10 == base + i`）⇒ 里面的循环只剩乘加。
        // 标量这里用 i32 累加，够用：最坏 |Σ c·s| ≤ `Σ|c|·smax`，
        // 8bit `240·255 = 61200`、12bit `240·4095 ≈ 9.8e5`（**注意是 Σ|c| = 240，
        // 不是 Σc = 128** —— 这个混淆曾让 SIMD 分支踩坑，见 `simd.rs` 模块头）。
        let fx = &SUBPEL_FILTERS[filt_h][phase_x];
        let fy = &SUBPEL_FILTERS[filt_v][phase_y];
        let need = interp_h * wu;
        let intermediate = scratch_of_i16(mc_scratch, need);
                // SIMD 条件：横向每个 8/16 列一批要读到 `[x0+c-3, x0+c+11/20]`，
        // 所以整块窗口在水平方向必须落在帧内（竖向逐行钳位，不影响）。
        let interior_x = x0 >= 3 && x0 + wu as i64 + 3 <= last_x;
        let simd = simd::avx2_available();
        if simd && plane_data.bit_depth == 8 {
            // **整块**的横向趟（一次调用做完全部行，见 `simd::mc_horiz_rows_8bpc`）。
            // 行下标**由内核现算**（`clip(y0 + r - 3, last_y)`）—— 这里原先为它开一个
            // `[0usize; MAX_SB_SIZE + 16]` 的数组，等于每次调用先把 1.1 KB 清零
            // （1080p 一帧约 7.6 万次调用 ≈ 88 MB 白写，2026-10-05 第三十七轮去掉）。
            // SAFETY: 已判 AVX2；行下标由内核钳到 `last_y`、`intermediate`
            // 至少有 `interp_h * wu` 项（`need` 就是按它算的）。
            unsafe {
                simd::mc_horiz_rows_8bpc(
                    data,
                    stride,
                    y0,
                    last_y,
                    interp_h,
                    x0 as isize,
                    wu,
                    fx,
                    inter_round0,
                    interior_x,
                    last_x,
                    intermediate,
                )
            };
            
                        mc_vertical_pass(
                            intermediate,
                            dst,
                            wu,
                            h as usize,
                            interp_h,
                            fy,
                            inter_round1,
                            true,
                        )?;
            
            return Ok(());
        }
        // 10/12bit（或没有 AVX2）：横向仍逐行做 —— 8 位通道的收窄前提是
        // "取整后值域只有 ±4.5e3（12bit ±30712）"，10/12bit 的样本本身就要 i32 累加。
        for r in 0..interp_h {
            let sy = clip(y0 + r as i64 - 3, last_y) as usize;
            let row = &data[sy * stride..];
            let out = &mut intermediate[r * wu..r * wu + wu];
            let mut c = 0usize;
            if simd && c + 8 <= wu && interior_x {
                // i32 通道的 8 输出版（10/12bit 也走这条）
                while c + simd::LANES <= wu {
                    // SAFETY: 已判 AVX2；`interior_x` 保证 `[x0+c-3, x0+c+11]` 在帧内
                    unsafe {
                        simd::mc_horiz8(
                            row,
                            (x0 + c as i64) as isize,
                            fx,
                            inter_round0,
                            &mut out[c..],
                        )
                    };
                    c += simd::LANES;
                }
            }
            while c < wu {
                let bc = x0 + c as i64;
                let mut s = 0i32;
                for t in 0..8usize {
                    let sx = clip(bc + t as i64 - 3, last_x) as usize;
                    s += fx[t] as i32 * row[sx] as i32;
                }
                // i16 收窄：值域见 [`PredScratch`] 的说明（最坏 12bit 也只有 30712）
                out[c] = round2_i32(s, inter_round0) as i16;
                c += 1;
            }
        }
        
                mc_vertical_pass(
                    intermediate,
                    dst,
                    wu,
                    h as usize,
                    interp_h,
                    fy,
                    inter_round1,
                    simd,
                )?;
        
        return Ok(());
    }

    // ── 一般路径：参考帧被缩放（超分 / 尺寸不同）时相位逐像素变，保持原实现 ──
    // 这条路上建表不划算，且真素材里占比极低，所以只做最必要的整理。
    let filt_h = subpel_index(filter, w <= 4)?;
    let need = interp_h * wu;
    let intermediate = scratch_of_i16(mc_scratch, need);
    for r in 0..interp_h {
        let sy = clip((start_y >> SCALE_SUBPEL_BITS) + r as i64 - 3, last_y) as usize;
        let row = &data[sy * stride..];
        for c in 0..wu {
            let p = start_x + step_x * c as i64;
            let phase = ((p >> 6) & SUBPEL_MASK as i64) as usize;
            let f = &SUBPEL_FILTERS[filt_h][phase];
            let bc = p >> SCALE_SUBPEL_BITS;
            let mut s = 0i32;
            for t in 0..8usize {
                let sx = clip(bc + t as i64 - 3, last_x) as usize;
                s += f[t] as i32 * row[sx] as i32;
            }
            intermediate[r * wu + c] = round2_i32(s, inter_round0) as i16;
        }
    }
    // 纵向
    let filt_v = subpel_index(filter, h <= 4)?;
    let rows = intermediate.len() / wu;
    for r in 0..h as usize {
        let p = (start_y & 1023) + step_y * r as i64;
        let phase = ((p >> 6) & SUBPEL_MASK as i64) as usize;
        let f = &SUBPEL_FILTERS[filt_v][phase];
        let base = (p >> SCALE_SUBPEL_BITS) as usize;
        if base + 8 > rows {
            return Err(MediaError::Bitstream("帧间预测的中间缓冲越界".into()));
        }
        for c in 0..wu {
            let mut s = 0i32;
            for t in 0..8usize {
                s += f[t] as i32 * intermediate[(base + t) * wu + c] as i32;
            }
            dst[r * wu + c] = round2_i32(s, inter_round1);
        }
    }
    Ok(())
}

/// 相位为 0 时的 8 抽头滤波器（δ）：`SUBPEL_FILTERS[..][0]` 必须长这样，
/// 否则上面那条"整像素 MV 直接取样本"的快路径不成立。
const DELTA_FILTER: [i16; 8] = [0, 0, 0, 128, 0, 0, 0, 0];

/// `Round2( x, n )`（i32；插值累加值域安全，见 `block_inter_prediction` 的注释）。
#[inline]
fn round2_i32(x: i32, n: u32) -> i32 {
    if n == 0 {
        return x;
    }
    (x + (1 << (n - 1))) >> n
}

/// `Round2( x, n )`（对 i64；x 可能是负数，用算术右移的语义）。
#[inline]
fn round2_i64(x: i64, n: u32) -> i64 {
    if n == 0 {
        return x;
    }
    (x + (1 << (n - 1))) >> n
}

/// 插值滤波器下标（§7.11.3.4 的 `interpFilter` 重映射 + 表的行序）。
fn subpel_index(filter: u8, small: bool) -> Result<usize> {
    const EIGHTTAP: u8 = 0;
    const EIGHTTAP_SMOOTH: u8 = 1;
    const EIGHTTAP_SHARP: u8 = 2;
    const BILINEAR: u8 = 3;
    let idx = if small {
        match filter {
            EIGHTTAP | EIGHTTAP_SHARP => 4,
            EIGHTTAP_SMOOTH => 5,
            BILINEAR => 3,
            other => {
                return Err(MediaError::Bitstream(format!(
                    "帧间预测：块级插值滤波器不该是 {other}（SWITCHABLE 只出现在帧级）"
                )))
            }
        }
    } else {
        match filter {
            EIGHTTAP => 0,
            EIGHTTAP_SMOOTH => 1,
            EIGHTTAP_SHARP => 2,
            BILINEAR => 3,
            other => {
                return Err(MediaError::Bitstream(format!(
                    "帧间预测：块级插值滤波器不该是 {other}（SWITCHABLE 只出现在帧级）"
                )))
            }
        }
    };
    Ok(idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::block::II_DC_PRED;
    use crate::video::block::{BLOCK_16X16, BLOCK_32X32, BLOCK_8X8};
    use crate::video::intra::Plane;

    #[test]
    fn rounding_variables_match_the_spec_table() {
        // 规范 §7.11.3.2 的四行
        assert_eq!(rounding_vars(8, false), (3, 11, 0));
        assert_eq!(rounding_vars(8, true), (3, 7, 4));
        assert_eq!(rounding_vars(10, false), (3, 11, 0));
        assert_eq!(rounding_vars(10, true), (3, 7, 4));
        assert_eq!(rounding_vars(12, false), (5, 9, 0));
        assert_eq!(rounding_vars(12, true), (5, 7, 2));
        // 不变量：`InterRound0 + InterRound1 + InterPostRound == 2 * FILTER_BITS`
        for bd in [8u8, 10, 12] {
            for compound in [false, true] {
                let (a, b, c) = rounding_vars(bd, compound);
                assert_eq!(a + b + c, 2 * FILTER_BITS, "位深 {bd}、复合 {compound}");
            }
        }
    }

    /// 两个 `wedge_sign` 的掩码必须**逐点互补（和为 64）**——这是混合公式
    /// `m*p0 + (64-m)*p1` 能成立的前提。
    ///
    /// 注意：**色度下采样之后只能保证 `|和 - 64| <= 1`**。`Round2` 不是线性运算：
    /// `Round2(S,1) + Round2(128-S,1)` 在 `S` 为奇数时是 65。规范 §7.11.3.14 的
    /// 色度分支就是这么取的平均，所以这是**规范行为**、不是实现偏差——但别写出
    /// "色度掩码也严格互补"这种断言。
    #[test]
    fn wedge_masks_are_complements_including_chroma() {
        // 与 `blend_masked` 里折算色度权重的那段是同一套规则（这里直接照写一遍，
        // 保证测的是"亮度掩码 + 下采样规则"这个组合）。
        fn chroma_weight(
            bsize: usize,
            sign: u8,
            wedge: usize,
            sub_x: u8,
            sub_y: u8,
            y: usize,
            x: usize,
        ) -> i32 {
            let at = |ly: usize, lx: usize| wedge_mask(bsize, sign, wedge, ly, lx) as i32;
            if sub_x == 0 && sub_y == 0 {
                at(y, x)
            } else if sub_x == 1 && sub_y == 0 {
                crate::video::intra::round2(at(y, 2 * x) + at(y, 2 * x + 1), 1)
            } else {
                crate::video::intra::round2(
                    at(2 * y, 2 * x) + at(2 * y, 2 * x + 1) + at(2 * y + 1, 2 * x)
                        + at(2 * y + 1, 2 * x + 1),
                    2,
                )
            }
        }
        for bsize in [BLOCK_8X8 as usize, BLOCK_16X16 as usize, BLOCK_32X32 as usize] {
            let lw = block_width(bsize as u8) as usize;
            let lh = block_height(bsize as u8) as usize;
            for wedge in 0..super::super::wedge::WEDGE_TYPES {
                for (sub_x, sub_y) in [(0u8, 0u8), (1, 0), (1, 1)] {
                    let (w, h) = (lw >> sub_x, lh >> sub_y);
                    // 亮度（不下采样）必须严格互补
                    let tol = if sub_x == 0 && sub_y == 0 { 0 } else { 1 };
                    for y in 0..h {
                        for x in 0..w {
                            let a = chroma_weight(bsize, 0, wedge, sub_x, sub_y, y, x);
                            let b = chroma_weight(bsize, 1, wedge, sub_x, sub_y, y, x);
                            assert!(
                                (a + b - 64).abs() <= tol,
                                "bsize={bsize} wedge={wedge} sub=({sub_x},{sub_y}) \
                                 ({y},{x})：{a} + {b} != 64"
                            );
                            assert!((0..=64).contains(&a) && (0..=64).contains(&b));
                        }
                    }
                }
            }
        }
    }

    /// 混合公式的两个端点：`m == 64` 取第一路、`m == 0` 取第二路。
    #[test]
    fn wedge_blend_endpoints_pick_one_reference() {
        let bit_depth = 8u8;
        let (_, _, ipr) = rounding_vars(bit_depth, true);
        let shift = 6 + ipr;
        // 复合时预测值带 `ipr` 位小数：把像素值左移 `ipr` 位构造 preds
        let p0 = 200i32 << ipr;
        let p1 = 40i32 << ipr;
        let max = (1i32 << bit_depth) - 1;
        let blend = |m: i32| {
            super::super::intra::round2(m * p0 + (64 - m) * p1, shift).clamp(0, max)
        };
        assert_eq!(blend(64), 200);
        assert_eq!(blend(0), 40);
        assert_eq!(blend(32), 120);
    }

    /// `Difference weight mask process`（§7.11.3.12）：两路完全相同 ⇒ `diff = 0`
    /// ⇒ 权重恒 38（`mask_type = 1` 时 26）。
    #[test]
    fn diffwtd_mask_is_flat_when_predictions_agree() {
        let (_, _, ipr) = rounding_vars(8, true);
        let preds = [vec![100i32 << ipr; 16], vec![100i32 << ipr; 16]];
        let mut b = Vec::new();
        build_diffwtd_mask(&preds, 4, 4, 8, ipr, 0, &mut b);
        let m = &b;
        assert!(m.iter().all(|&v| v == 38), "两路相同时权重应当是 38：{m:?}");
        let mut b1 = Vec::new();
        build_diffwtd_mask(&preds, 4, 4, 8, ipr, 1, &mut b1);
        let m1 = &b1;
        assert!(m1.iter().all(|&v| v == 26), "mask_type = 1 应当是 64 - 38");

        // 差到顶：`38 + Round2(diff, ipr) / 16`。8 位下 `Round2(255 << 4, 4) = 255`，
        // 于是最大是 `38 + 15 = 53` —— **`Clip3(0, 64, …)` 的上限在 8 位下够不着**
        // （别按"差越大越接近 64"写断言）。
        let far = [vec![255i32 << ipr; 16], vec![0i32; 16]];
        let mut b2 = Vec::new();
        build_diffwtd_mask(&far, 4, 4, 8, ipr, 0, &mut b2);
        let m2 = &b2;
        assert!(m2.iter().all(|&v| v == 53), "差到顶应当是 53：{m2:?}");

        // 两个 `mask_type` 逐点互补（和为 64）；权重恒在 [0, 64]
        let mut b3 = Vec::new();
        build_diffwtd_mask(&far, 4, 4, 8, ipr, 1, &mut b3);
        let m3 = &b3;
        for (a, b) in m2.iter().zip(m3.iter()) {
            assert_eq!(a + b, 64, "两个 mask_type 不互补");
        }
        for v in m.iter().chain(m2.iter()).chain(m3.iter()) {
            assert!((0..=64).contains(v), "权重越界：{v}");
        }
    }

    /// 距离权重的两张表：`Quant_Dist_Lookup` **每一行的两个权重和恒为 16**
    /// （`>> (4 + InterPostRound)` 靠这个才刚好把复合精度除回像素范围）。
    /// `Quant_Dist_Weight` 是**递推用的乘数**、不是权重，行和不是 16。
    #[test]
    fn distance_weights_sum_to_16() {
        for r in QUANT_DIST_LOOKUP {
            assert_eq!(r[0] + r[1], 16, "Quant_Dist_Lookup 的一行和不为 16：{r:?}");
        }
        // 有一侧距离为 0 ⇒ 直接第 3 行 `{13, 3}`；`order = (d0 <= d1)` 决定谁配谁
        assert_eq!(dist_weights_for(0, 5), (3, 13));
        assert_eq!(dist_weights_for(5, 0), (13, 3));
        // 任何输入都必须落在合法行上、且两权重和为 16
        for d0 in 0..=31i32 {
            for d1 in 0..=31i32 {
                let (f, b) = dist_weights_for(d0, d1);
                assert_eq!(f + b, 16, "d0={d0} d1={d1} 得到 {f}/{b}");
            }
        }
    }

    /// 借 `testbits` 造一个真实的 8 位 4:2:0 序列头（4:2:0 才能碰到 interintra
    /// 掩码的色度降采样支）。
    fn seq_420() -> SequenceHeader {
        use crate::video::testbits::{write_reduced_still_picture_1080p, BitWriter};
        let mut w = BitWriter::default();
        write_reduced_still_picture_1080p(&mut w);
        let sh = SequenceHeader::parse(w.finish()).expect("序列头应当能解析");
        assert!(sh.color.subsampling_x && sh.color.subsampling_y && sh.color.bit_depth == 8);
        sh
    }

    /// `Ii_Weights_1d`（§7.11.3.13）：128 项、从 60 **单调不增**到 1，且恒在 [1, 64]。
    ///
    /// 最后那半段是**重复值**（1 一连串），所以别写"严格递减"。
    #[test]
    fn ii_weights_1d_match_the_spec_table() {
        assert_eq!(II_WEIGHTS_1D.len(), MAX_SB_SIZE);
        assert_eq!(II_WEIGHTS_1D[0], 60);
        assert_eq!(II_WEIGHTS_1D[16], 34);
        assert_eq!(II_WEIGHTS_1D[48], 11);
        assert_eq!(II_WEIGHTS_1D[127], 1);
        assert!(II_WEIGHTS_1D.iter().all(|&v| (1..=64).contains(&v)));
        for i in 0..MAX_SB_SIZE - 1 {
            assert!(II_WEIGHTS_1D[i] >= II_WEIGHTS_1D[i + 1], "第 {i} 项回升了");
        }
        // `sizeScale = MAX_SB_SIZE / max(w, h)`（§7.11.3.13）必须**整除**，
        // 且 interintra 的块（8×8 … 32×32，边长都是 2 的幂）索引不越界。
        for (w, h) in [(8usize, 8usize), (8, 16), (16, 8), (16, 16), (16, 32), (32, 32), (8, 32)] {
            let m = w.max(h);
            assert_eq!(MAX_SB_SIZE % m, 0, "{w}×{h} 的 sizeScale 不是整数");
            let last = (m - 1) * (MAX_SB_SIZE / m);
            assert!(last < MAX_SB_SIZE, "{w}×{h} 的掩码索引越界：{last}");
        }
    }

    /// 非楔形 `interintra`：DC 的掩码恒 32 ⇒ 结果正好落在帧内与帧间的**中点**。
    #[test]
    fn interintra_dc_blend_is_the_midpoint() {
        let seq = seq_420();
        let grid = BlockGrid::new(1, 1);
        let mut info = BlockInfo::default();
        info.mi_size = BLOCK_8X8 as u8;
        info.interintra = true;
        info.wedge_interintra = false;
        info.interintra_mode = II_DC_PRED;

        let mut plane = Plane::new(8, 8, 8);
        for y in 0..8 {
            for x in 0..8 {
                plane.set(x, y, 100); // 帧内那一半（`compute_prediction` 先写下的）
            }
        }
        let inter = vec![200i32; 64]; // 帧间那一半
        blend_interintra_region(&mut plane, &inter, &grid, 0, &seq, &info, 0, 0, 0, 8, 8, 0, 255)
            .expect("DC interintra 不该失败");
        // Round2(32 * 100 + 32 * 200, 6) = 150
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(plane.get(x, y), 150, "({x},{y}) 不是中点");
            }
        }
    }

    /// 掩码的**取值轴**：`II_H_PRED` 只看列 `j`、`II_V_PRED` 只看行 `i`（§7.11.3.13）。
    ///
    /// 把 `i`/`j` 写反了这个测试必挂 —— 用"帧内 0、帧间 64"把结果化成 `64 - m`，
    /// 于是可以直接断言掩码值本身。
    #[test]
    fn interintra_weight_axis_follows_the_mode() {
        let seq = seq_420();
        let grid = BlockGrid::new(1, 1);
        let (lw, lh) = (8usize, 16usize); // BLOCK_8X16，sizeScale = 128 / 16 = 8
        let scale = MAX_SB_SIZE / lw.max(lh);

        let run = |mode: u8| {
            let mut info = BlockInfo::default();
            info.mi_size = crate::video::block::BLOCK_8X16 as u8;
            info.interintra = true;
            info.interintra_mode = mode;
            let mut plane = Plane::new(lw, lh, 8);
            for y in 0..lh {
                for x in 0..lw {
                    plane.set(x, y, 0); // 帧内
                }
            }
            let inter = vec![64i32; lw * lh]; // 帧间
            blend_interintra_region(
                &mut plane, &inter, &grid, 0, &seq, &info, 0, 0, 0, lw as i32, lh as i32, 0, 255,
            )
            .expect("interintra 不该失败");
            plane
        };

        // 帧内 0 / 帧间 64 ⇒ Round2(m * 0 + (64 - m) * 64, 6) = 64 - m（m 是偶数时精确）
        let horiz = run(II_H_PRED);
        for j in 0..lw {
            let want = (64 - II_WEIGHTS_1D[j * scale]) as u16;
            for i in 0..lh {
                assert_eq!(horiz.get(j, i), want, "II_H 的列 {j} 在第 {i} 行不对");
            }
        }
        let vert = run(II_V_PRED);
        for i in 0..lh {
            let want = (64 - II_WEIGHTS_1D[i * scale]) as u16;
            for j in 0..lw {
                assert_eq!(vert.get(j, i), want, "II_V 的行 {i} 在第 {j} 列不对");
            }
        }
        // 两种模式的掩码必须不同（否则上面的断言说明不了轴的问题）
        assert_ne!(horiz.px16(), vert.px16());
    }

    /// 色度上的**楔形** interintra：掩码是亮度分辨率的，色度样本取
    /// 2×2（4:2:0/4:2:2 的竖直方向）或 1×2（4:2:2）的平均（§7.11.3.14 的取平均支）。
    #[test]
    fn interintra_wedge_mask_is_subsampled_for_chroma() {
        // 亮度掩码随便造一个有区分度的 8×8（只要不是常数，取平均就对不上别处）
        let luma: Vec<Vec<i32>> = (0..8).map(|y| (0..8).map(|x| ((y * 8 + x) % 65) as i32).collect()).collect();
        for (sub_x, sub_y) in [(1u8, 0u8), (1, 1)] {
            let (w, h) = (8usize >> sub_x, 8usize >> sub_y);
            let mut plane = Plane::new(w, h, 8);
            for y in 0..h {
                for x in 0..w {
                    plane.set(x, y, 64); // 帧内 64、帧间 0 ⇒ 结果 = Round2(m * 64, 6) = m
                }
            }
            let inter = vec![0i32; w * h];
            blend_interintra_core(
                &mut plane, &inter, 0, 0, w as i32, h as i32, sub_x, sub_y, false, 0, 255,
                |ly, lx| luma[ly][lx],
            );
            for y in 0..h {
                for x in 0..w {
                    let want = if sub_y == 0 {
                        crate::video::intra::round2(luma[y][2 * x] + luma[y][2 * x + 1], 1)
                    } else {
                        crate::video::intra::round2(
                            luma[2 * y][2 * x]
                                + luma[2 * y][2 * x + 1]
                                + luma[2 * y + 1][2 * x]
                                + luma[2 * y + 1][2 * x + 1],
                            2,
                        )
                    };
                    assert_eq!(
                        i32::from(plane.get(x, y)),
                        want,
                        "sub=({sub_x},{sub_y}) 的 ({x},{y}) 降采样不对"
                    );
                }
            }
        }
    }

    /// **微基准**：一次 `block_inter_prediction`（64×64 亮度、亚像素 MV）的
    /// **每像素耗时** —— 含横向趟、纵向趟与全部脚手架（这正是 `PHASE` 的 `mc=`
    /// 里最贵的一档）。用来在**同进程内**快速迭代 MC 的结构改造，不必受整帧
    /// 噪声的影响。
    ///
    /// 跑法：`cargo test --release -p me-decode --lib -- --ignored --nocapture bench_mc_block`
    #[test]
    #[ignore]
    fn bench_mc_block() {
        let w = 256usize;
        let h = 200usize;
        let mut plane = super::super::intra::Plane::new(w, h, 8);
        for (i, v) in plane.px16_mut().iter_mut().enumerate() {
            *v = ((i * 37 + 11) % 256) as u16;
        }
        let mut mc_scratch: Vec<i16> = Vec::new();
        let iters = 20_000usize;
        for (bw, bh) in [(64usize, 64usize), (32, 32), (16, 16)] {
            let mut dst = vec![0i32; bw * bh];
            // 相位 = (5, 3)（非 0 ⇒ 走两趟 8 抽头）
            let start_x = (10i64 << 10) + (5 << 6);
            let start_y = (10i64 << 10) + (3 << 6);
            let t = std::time::Instant::now();
            for i in 0..iters {
                let j = (i % 8) as i64;
                // SAFETY(逻辑)：块 + 3 抽头的窗口都在 256×200 的平面内
                block_inter_prediction(
                    &mut dst,
                    &plane,
                    0,
                    start_x + j * (1 << 10),
                    start_y,
                    1 << 10,
                    1 << 10,
                    bw as i32,
                    bh as i32,
                    0,
                    3,
                    7,
                    Some((w as i64 - 1, h as i64 - 1)),
                    &mut mc_scratch,
                )
                .unwrap();
                std::hint::black_box(&dst);
            }
            let ns = t.elapsed().as_secs_f64() * 1e9 / (iters as f64);
            eprintln!(
                "block_inter_prediction {bw}x{bh}: {ns:.0} ns/块 = {:.3} ns/像素",
                ns / (bw * bh) as f64
            );
        }
    }
}
