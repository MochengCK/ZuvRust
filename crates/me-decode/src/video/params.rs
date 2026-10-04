//! AV1 **非压缩头的第二部分**（自研软解的 M2.5 + M2.6）。
//!
//! 规范 §5.9.2 的 `frame_header_obu()` 在 `disable_frame_end_update_cdf` 之后还有一长串
//! 明文参数（**这一段仍然没有熵解码**，全是 `f(n)` / `su(n)` / `ns(n)` 位读）：
//!
//! ```text
//! tile_info()            瓦片网格（列/行起点、瓦片尺寸字段宽度）
//! quantization_params()  基础量化步长与四个 delta
//! segmentation_params()  8 段 × 8 个特征
//! delta_q_params()       qindex 的块级 delta
//! delta_lf_params()      环路滤波强度的块级 delta
//! loop_filter_params()   电平、锐度、参考/模式 delta
//! cdef_params()          阻尼与强度表
//! lr_params()            环路恢复类型与单元尺寸
//! read_tx_mode()         变换尺寸模式
//! frame_reference_mode() reference_select
//! skip_mode_params()     skip 模式的复合参考帧对
//! allow_warped_motion / reduced_tx_set
//! global_motion_params() 7 个参考帧的全局运动模型（含子指数哥伦布编码）
//! film_grain_params()    胶片颗粒
//! ```
//!
//! ## 为什么这一段值得单独做（而不是"等熵层一起做"）
//!
//! 因为 `frame_header_obu()` 结束后规范要求 **`byte_alignment()`**，而填充位必须是 0
//! —— 这把"非压缩头读了多少位"变成一条**可被真实码流证伪的硬约束**。序列头那边
//! 我们靠 `trailing_bits()` 抓出过一次整体偏移；非压缩头这一段更长（字段多、
//! 分支多），没有硬约束就只能靠"值看起来合理"来自我安慰。
//!
//! ## 跨帧继承的状态
//!
//! 有几个参数是"没写就沿用上一帧"的（分割特征、环路滤波 delta、全局运动参数），
//! 它们的来源是 `primary_ref_frame` 指向的那个槽（规范 §7.20/§7.21 的
//! `load_previous()`）。位读**不依赖**这些继承值（分支条件只看本帧读到的东西），
//! 但值对不上以后滤波/运动补偿就会错，所以这里照规范做了：
//! [`SavedFrameState`] 随参考帧刷新一起保存，解析时按 `primary_ref_frame` 加载。
//!
//! 逐字对照的规范出处写在各函数上方；**改动这里之前先把 `/tmp/av1-spec.txt`
//! 对应的伪代码翻出来**（这个项目已经吃过两次"凭记忆写位序"的亏）。

use me_core::{MediaError, Result};

use super::bitreader::BitReader;
use super::frame_header::{
    FrameHeader, FrameType, RefFrameStore, NUM_REF_FRAMES, PRIMARY_REF_NONE, REFS_PER_FRAME,
};
use super::sequence_header::SequenceHeader;

// ── 常量（规范 §3 / 常量表，取值都对照过原文）──────────────────────────────

/// 段数上限。
pub const MAX_SEGMENTS: usize = 8;
/// 每段的特征数。
pub const SEG_LVL_MAX: usize = 8;
/// `FRAME_LF_COUNT`：`DeltaLF[]` / 环路滤波 delta 的条数（Y 垂直/水平、U、V）
pub const FRAME_LF_COUNT: usize = 4;
/// 段特征的序号
pub const SEG_LVL_ALT_Q: usize = 0;
/// 分割特征里的去块强度偏移（§7.14.5：`feature = SEG_LVL_ALT_LF_Y_V + i`）
pub const SEG_LVL_ALT_LF_Y_V: usize = 1;
pub const SEG_LVL_ALT_LF_Y_H: usize = 2;
pub const SEG_LVL_ALT_LF_U: usize = 3;
pub const SEG_LVL_ALT_LF_V: usize = 4;
pub const SEG_LVL_SKIP: usize = 6;
pub const SEG_LVL_GLOBALMV: usize = 7;
/// 段特征的序号（从这一项起 `SegIdPreSkip = 1`）。
pub const SEG_LVL_REF_FRAME: usize = 5;
/// 参考帧类型数（含 `INTRA_FRAME`）。
pub const TOTAL_REFS_PER_FRAME: usize = 8;
/// 环路滤波电平上限。
pub const MAX_LOOP_FILTER: i32 = 63;
/// 单个瓦片的最大宽度（亮度样本）。
pub const MAX_TILE_WIDTH: u32 = 4096;
/// 单个瓦片的最大面积（亮度样本）。
pub const MAX_TILE_AREA: u32 = 4096 * 2304;
/// 瓦片列数上限。
pub const MAX_TILE_COLS: u32 = 64;
/// 瓦片行数上限。
pub const MAX_TILE_ROWS: u32 = 64;
/// 环路恢复单元的最大尺寸。
pub const RESTORATION_TILESIZE_MAX: u32 = 256;

/// 全局运动模型的内部精度。
pub const WARPEDMODEL_PREC_BITS: i32 = 16;
const GM_ABS_TRANS_BITS: u32 = 12;
const GM_ABS_TRANS_ONLY_BITS: u32 = 9;
const GM_ABS_ALPHA_BITS: u32 = 12;
const GM_ALPHA_PREC_BITS: i32 = 15;
const GM_TRANS_PREC_BITS: i32 = 6;
const GM_TRANS_ONLY_PREC_BITS: i32 = 3;

/// 全局运动模型类型（规范 §3）：IDENTITY / TRANSLATION / ROTZOOM / AFFINE。
pub const IDENTITY: u8 = 0;
pub const TRANSLATION: u8 = 1;
pub const ROTZOOM: u8 = 2;
pub const AFFINE: u8 = 3;

/// 变换尺寸模式。
pub const ONLY_4X4: u8 = 0;
pub const TX_MODE_LARGEST: u8 = 1;
pub const TX_MODE_SELECT: u8 = 2;

/// 环路恢复类型：0 不恢复、1 块级可选、2 Wiener、3 SGR。
pub const RESTORE_NONE: u8 = 0;
pub const RESTORE_SWITCHABLE: u8 = 1;
pub const RESTORE_WIENER: u8 = 2;
pub const RESTORE_SGRPROJ: u8 = 3;

/// `Segmentation_Feature_Bits`（规范 §5.9.8 后的常量表）。
const SEG_FEATURE_BITS: [u32; SEG_LVL_MAX] = [8, 6, 6, 6, 6, 3, 0, 0];
/// `Segmentation_Feature_Signed`。
const SEG_FEATURE_SIGNED: [bool; SEG_LVL_MAX] = [true, true, true, true, true, false, false, false];
/// `Segmentation_Feature_Max`（`MAX_LOOP_FILTER` = 63）。
const SEG_FEATURE_MAX: [i32; SEG_LVL_MAX] = [255, 63, 63, 63, 63, 7, 0, 0];

fn clip3(lo: i32, hi: i32, v: i32) -> i32 {
    v.clamp(lo, hi)
}

/// `tile_log2(blkSize, target)`（规范 §5.9.12）：最小的 k 使 `blkSize << k >= target`。
fn tile_log2(blk_size: u32, target: u32) -> u32 {
    let mut k = 0u32;
    while (blk_size << k) < target {
        k += 1;
        if k > 31 {
            break;
        }
    }
    k
}

// ── 跨帧继承的状态 ────────────────────────────────────────────────────────

/// 规范里"没写就沿用上一帧"的那几组参数（`SavedGmParams` / `SavedLoopFilterParams` /
/// `SavedSegmentationParams` 的合并体）。
///
/// 保存时机：参考帧更新过程（§7.20）；加载时机：`load_previous()`（§7.20），
/// 来源是 `ref_frame_idx[primary_ref_frame]`。`primary_ref_frame == PRIMARY_REF_NONE`
/// 时按 `setup_past_independence()` 的初值（规范 §7.20）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SavedFrameState {
    /// `loop_filter_ref_deltas[]`（索引是参考帧类型）
    pub lf_ref_deltas: [i32; TOTAL_REFS_PER_FRAME],
    /// `loop_filter_mode_deltas[2]`
    pub lf_mode_deltas: [i32; 2],
    /// `FeatureEnabled[8][8]`
    pub seg_feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    /// `FeatureData[8][8]`
    pub seg_feature_data: [[i32; SEG_LVL_MAX]; MAX_SEGMENTS],
    /// `GmParams[8][6]`（索引是参考帧类型）
    pub gm_params: [[i32; 6]; TOTAL_REFS_PER_FRAME],
}

impl SavedFrameState {
    /// `setup_past_independence()` 的初值（规范 §7.20 的逐条）。
    pub fn past_independence() -> Self {
        let mut lf_ref_deltas = [0i32; TOTAL_REFS_PER_FRAME];
        // INTRA_FRAME = 0 → 1；GOLDEN(4) / ALTREF2(6) / ALTREF(7) → -1
        lf_ref_deltas[0] = 1;
        lf_ref_deltas[4] = -1;
        lf_ref_deltas[6] = -1;
        lf_ref_deltas[7] = -1;
        let mut gm_params = [[0i32; 6]; TOTAL_REFS_PER_FRAME];
        for p in gm_params.iter_mut() {
            for (i, v) in p.iter_mut().enumerate() {
                // ((i % 3 == 2) ? 1 << WARPEDMODEL_PREC_BITS : 0)
                *v = if i % 3 == 2 { 1 << WARPEDMODEL_PREC_BITS } else { 0 };
            }
        }
        Self {
            lf_ref_deltas,
            lf_mode_deltas: [0; 2],
            seg_feature_enabled: [[false; SEG_LVL_MAX]; MAX_SEGMENTS],
            seg_feature_data: [[0; SEG_LVL_MAX]; MAX_SEGMENTS],
            gm_params,
        }
    }
}

impl Default for SavedFrameState {
    fn default() -> Self {
        Self::past_independence()
    }
}

// ── 各段参数的结构 ────────────────────────────────────────────────────────

/// `tile_info()`（规范 §5.9.12）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileInfo {
    pub uniform: bool,
    pub cols: u32,
    pub rows: u32,
    pub cols_log2: u32,
    pub rows_log2: u32,
    pub context_update_tile_id: u32,
    /// 每个非末瓦片的尺寸字段用几个字节（`TileSizeBytes`）
    pub tile_size_bytes: u32,
    /// 每个瓦片列起点的模式信息列（长度 = `cols + 1`）
    pub mi_col_starts: Vec<u32>,
    /// 每个瓦片行起点的模式信息行（长度 = `rows + 1`）
    pub mi_row_starts: Vec<u32>,
    pub sb_cols: u32,
    pub sb_rows: u32,
    pub sb_shift: u32,
}

impl Default for TileInfo {
    fn default() -> Self {
        Self {
            uniform: false,
            cols: 1,
            rows: 1,
            cols_log2: 0,
            rows_log2: 0,
            context_update_tile_id: 0,
            tile_size_bytes: 0,
            mi_col_starts: vec![0],
            mi_row_starts: vec![0],
            sb_cols: 0,
            sb_rows: 0,
            sb_shift: 4,
        }
    }
}

impl TileInfo {
    /// 瓦片总数（tile group 的 `NumTiles`）。
    pub fn num_tiles(&self) -> u32 {
        self.cols * self.rows
    }
}

/// `quantization_params()`（规范 §5.9.6）。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct QuantParams {
    pub base_q_idx: u8,
    pub delta_q_y_dc: i32,
    pub delta_q_u_dc: i32,
    pub delta_q_u_ac: i32,
    pub delta_q_v_dc: i32,
    pub delta_q_v_ac: i32,
    pub diff_uv_delta: bool,
    pub using_qmatrix: bool,
    pub qm_y: u8,
    pub qm_u: u8,
    pub qm_v: u8,
}

/// `segmentation_params()`（规范 §5.9.8）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentationParams {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal_update: bool,
    pub update_data: bool,
    pub feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub feature_data: [[i32; SEG_LVL_MAX]; MAX_SEGMENTS],
    /// 最后一个启用了特征的段号
    pub last_active_seg_id: u8,
    /// 段号在 skip 语法之前读（有 `SEG_LVL_REF_FRAME` 及以后的段特征时为 1）
    pub seg_id_pre_skip: bool,
    /// `SegQMLevel[plane][segment]`（`using_qmatrix` 时有意义）
    pub seg_qm_level: [[u8; MAX_SEGMENTS]; 3],
}

impl Default for SegmentationParams {
    fn default() -> Self {
        Self {
            enabled: false,
            update_map: false,
            temporal_update: false,
            update_data: false,
            feature_enabled: [[false; SEG_LVL_MAX]; MAX_SEGMENTS],
            feature_data: [[0; SEG_LVL_MAX]; MAX_SEGMENTS],
            last_active_seg_id: 0,
            seg_id_pre_skip: false,
            seg_qm_level: [[0; MAX_SEGMENTS]; 3],
        }
    }
}

impl SegmentationParams {
    /// `seg_feature_active_idx(segmentId, feature)`（规范 §5.11.3）。
    pub fn feature_active(&self, segment: usize, feature: usize) -> bool {
        self.enabled && self.feature_enabled[segment][feature]
    }
}

/// `delta_q_params()`（规范 §5.9.9）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeltaQParams {
    pub present: bool,
    pub res: u8,
}

/// `delta_lf_params()`（规范 §5.9.10）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeltaLfParams {
    pub present: bool,
    pub res: u8,
    pub multi: bool,
}

/// `loop_filter_params()`（规范 §5.9.4）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopFilterParams {
    /// 亮度竖直 / 亮度水平 / U / V
    pub level: [u8; 4],
    pub sharpness: u8,
    pub delta_enabled: bool,
    pub ref_deltas: [i32; TOTAL_REFS_PER_FRAME],
    pub mode_deltas: [i32; 2],
}

impl Default for LoopFilterParams {
    fn default() -> Self {
        let d = SavedFrameState::past_independence();
        Self {
            level: [0; 4],
            sharpness: 0,
            delta_enabled: true,
            ref_deltas: d.lf_ref_deltas,
            mode_deltas: [0; 2],
        }
    }
}

/// `cdef_params()`（规范 §5.9.5）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CdefParams {
    pub damping: u8,
    pub bits: u8,
    pub y_pri_strength: [u8; 8],
    pub y_sec_strength: [u8; 8],
    pub uv_pri_strength: [u8; 8],
    pub uv_sec_strength: [u8; 8],
}

impl Default for CdefParams {
    fn default() -> Self {
        Self {
            damping: 3,
            bits: 0,
            y_pri_strength: [0; 8],
            y_sec_strength: [0; 8],
            uv_pri_strength: [0; 8],
            uv_sec_strength: [0; 8],
        }
    }
}

/// `lr_params()`（规范 §5.9.7）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LrParams {
    /// `FrameRestorationType[plane]`
    pub frame_restoration_type: [u8; 3],
    pub uses_lr: bool,
    pub uses_chroma_lr: bool,
    pub unit_shift: u8,
    pub size: [u32; 3],
}

impl Default for LrParams {
    fn default() -> Self {
        Self {
            frame_restoration_type: [RESTORE_NONE; 3],
            uses_lr: false,
            uses_chroma_lr: false,
            unit_shift: 0,
            size: [0; 3],
        }
    }
}

/// `global_motion_params()`（规范 §5.9.13）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalMotion {
    /// `GmType[ref]`（索引是参考帧类型；`INTRA_FRAME` 位不用）
    pub gm_type: [u8; TOTAL_REFS_PER_FRAME],
    /// `gm_params[ref][6]`
    pub params: [[i32; 6]; TOTAL_REFS_PER_FRAME],
}

impl GlobalMotion {
    fn identity() -> Self {
        let mut params = [[0i32; 6]; TOTAL_REFS_PER_FRAME];
        for p in params.iter_mut() {
            for (i, v) in p.iter_mut().enumerate() {
                *v = if i % 3 == 2 { 1 << WARPEDMODEL_PREC_BITS } else { 0 };
            }
        }
        Self { gm_type: [IDENTITY; TOTAL_REFS_PER_FRAME], params }
    }
}

impl Default for GlobalMotion {
    fn default() -> Self {
        Self::identity()
    }
}

/// `film_grain_params()`（规范 §5.9.30）。
///
/// 这里存的是**合成颗粒所需的全部参数**（点表、自回归系数、色度系数、开关），
/// 因为胶片颗粒合成（§7.18.3）要用它们在输出帧上重建颗粒。跨帧继承（`update_grain
/// == 0` 时的 `load_grain_params`）也依赖这份结构逐槽保存（见
/// [`RefFrameStore`](super::frame_header::RefFrameStore) 的 `grain` 字段）。
///
/// 系数/乘子都按规范做了偏移还原：`ar_coeffs_*` 与 `uv_mult`/`uv_luma_mult`
/// 减 128，`uv_offset` 减 256。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FilmGrainParams {
    pub apply_grain: bool,
    pub grain_seed: u16,
    pub update_grain: bool,
    /// `update_grain == 0` 时被继承的参考槽号（`ref_idx`）
    pub ref_idx: u8,
    pub num_y_points: u8,
    /// `point_y_value` / `point_y_scaling`（`num_y_points` 项有效）
    pub y_points: [[u8; 2]; 14],
    pub chroma_scaling_from_luma: bool,
    /// `num_cb_points` / `num_cr_points`
    pub num_uv_points: [u8; 2],
    /// `point_cb_*` / `point_cr_*`（各 `num_*_points` 项有效）
    pub uv_points: [[[u8; 2]; 10]; 2],
    pub ar_coeff_lag: u8,
    /// `ar_coeffs_y_plus_128 - 128`
    pub ar_coeffs_y: [i8; 24],
    /// `ar_coeffs_cb/cr_plus_128 - 128`
    pub ar_coeffs_uv: [[i8; 25]; 2],
    pub ar_coeff_shift: u8,
    pub grain_scale_shift: u8,
    pub scaling_shift: u8,
    /// `cb_mult - 128` / `cr_mult - 128`
    pub uv_mult: [i16; 2],
    /// `cb_luma_mult - 128` / `cr_luma_mult - 128`
    pub uv_luma_mult: [i16; 2],
    /// `cb_offset - 256` / `cr_offset - 256`
    pub uv_offset: [i16; 2],
    pub overlap_flag: bool,
    pub clip_to_restricted_range: bool,
}

/// 非压缩头第二部分解析出来的全部参数。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FrameParams {
    pub tile_info: TileInfo,
    pub quant: QuantParams,
    pub segmentation: SegmentationParams,
    pub delta_q: DeltaQParams,
    pub delta_lf: DeltaLfParams,
    pub loop_filter: LoopFilterParams,
    pub cdef: CdefParams,
    pub lr: LrParams,
    pub tx_mode: u8,
    pub reference_select: bool,
    pub skip_mode_present: bool,
    pub skip_mode_frame: [u8; 2],
    pub allow_warped_motion: bool,
    pub reduced_tx_set: bool,
    pub global_motion: GlobalMotion,
    pub film_grain: FilmGrainParams,
    /// `CodedLossless`（所有段都无损）
    pub coded_lossless: bool,
    /// `AllLossless` = `CodedLossless && FrameWidth == UpscaledWidth`
    pub all_lossless: bool,
}

impl FrameParams {
    /// 按规范顺序解析 `tile_info()` → `film_grain_params()`。
    ///
    /// `fh` 是已经读好的非压缩头前半（身份/尺寸/参考帧）；`store` 提供跨帧继承的状态
    /// 与 `RefOrderHint[]`（`skip_mode_params` 要用）。
    pub fn parse(
        r: &mut BitReader<'_>,
        fh: &FrameHeader,
        seq: &SequenceHeader,
        store: &RefFrameStore,
    ) -> Result<Self> {
        let prev = Self::load_previous(fh, store);
        let tile_info = Self::parse_tile_info(r, fh, seq)?;
        let quant = Self::parse_quantization(r, seq)?;
        let mut segmentation = Self::parse_segmentation(r, fh, &prev)?;
        // CodedLossless / AllLossless / SegQMLevel（规范 §5.9.2 尾部，无位读）
        let (coded_lossless, all_lossless, seg_qm_level) =
            Self::lossless_arrays(&quant, &segmentation, fh);
        segmentation.seg_qm_level = seg_qm_level;
        let delta_q = Self::parse_delta_q(r, &quant)?;
        let delta_lf = Self::parse_delta_lf(r, fh, &delta_q)?;
        let loop_filter = Self::parse_loop_filter(r, fh, seq, &prev, coded_lossless)?;
        let cdef = Self::parse_cdef(r, fh, seq, coded_lossless)?;
        let lr = Self::parse_lr(r, fh, seq, all_lossless)?;
        let tx_mode = Self::parse_tx_mode(r, coded_lossless)?;
        let reference_select = Self::parse_reference_mode(r, fh)?;
        let (skip_mode_present, skip_mode_frame) =
            Self::parse_skip_mode(r, fh, seq, store, reference_select)?;
        let allow_warped_motion =
            if fh.is_intra || fh.error_resilient_mode || !seq.enable_warped_motion {
                false
            } else {
                r.bit()?
            };
        let reduced_tx_set = r.bit()?;
        let global_motion = Self::parse_global_motion(r, fh, &prev)?;
        let film_grain = Self::parse_film_grain(r, fh, seq, store)?;

        Ok(FrameParams {
            tile_info,
            quant,
            segmentation,
            delta_q,
            delta_lf,
            loop_filter,
            cdef,
            lr,
            tx_mode,
            reference_select,
            skip_mode_present,
            skip_mode_frame,
            allow_warped_motion,
            reduced_tx_set,
            global_motion,
            film_grain,
            coded_lossless,
            all_lossless,
        })
    }

    /// `load_previous()`（规范 §7.20）：`primary_ref_frame == PRIMARY_REF_NONE` 时
    /// 用 `setup_past_independence()` 的初值，否则取 `ref_frame_idx[primary_ref_frame]` 槽。
    fn load_previous(fh: &FrameHeader, store: &RefFrameStore) -> SavedFrameState {
        if fh.primary_ref_frame == PRIMARY_REF_NONE {
            return SavedFrameState::past_independence();
        }
        let slot = fh.ref_frame_idx[(fh.primary_ref_frame as usize).min(REFS_PER_FRAME - 1)] as usize;
        let slot = slot.min(NUM_REF_FRAMES - 1);
        store.saved[slot].clone()
    }

    /// `tile_info()`（规范 §5.9.12）。
    fn parse_tile_info(
        r: &mut BitReader<'_>,
        fh: &FrameHeader,
        seq: &SequenceHeader,
    ) -> Result<TileInfo> {
        let mi_cols = fh.mi_cols;
        let mi_rows = fh.mi_rows;
        let (sb_cols, sb_rows, sb_shift) = if seq.use_128x128_superblock {
            ((mi_cols + 31) >> 5, (mi_rows + 31) >> 5, 5u32)
        } else {
            ((mi_cols + 15) >> 4, (mi_rows + 15) >> 4, 4u32)
        };
        let sb_size = sb_shift + 2;
        let max_tile_width_sb = MAX_TILE_WIDTH >> sb_size;
        let mut max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size);
        let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
        let max_log2_tile_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS));
        let max_log2_tile_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS));
        let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));

        let mut ti = TileInfo {
            sb_cols,
            sb_rows,
            sb_shift,
            ..Default::default()
        };

        ti.uniform = r.bit()?;
        if ti.uniform {
            let mut cols_log2 = min_log2_tile_cols;
            while cols_log2 < max_log2_tile_cols {
                if r.bit()? {
                    cols_log2 += 1;
                } else {
                    break;
                }
            }
            ti.cols_log2 = cols_log2;
            let tile_width_sb = (sb_cols + (1 << cols_log2) - 1) >> cols_log2;
            ti.mi_col_starts.clear();
            let mut start_sb = 0;
            while start_sb < sb_cols {
                ti.mi_col_starts.push(start_sb << sb_shift);
                start_sb += tile_width_sb;
            }
            ti.mi_col_starts.push(mi_cols);
            ti.cols = (ti.mi_col_starts.len() - 1) as u32;

            let min_log2_tile_rows = (min_log2_tiles.saturating_sub(cols_log2)).max(0);
            let mut rows_log2 = min_log2_tile_rows;
            while rows_log2 < max_log2_tile_rows {
                if r.bit()? {
                    rows_log2 += 1;
                } else {
                    break;
                }
            }
            ti.rows_log2 = rows_log2;
            let tile_height_sb = (sb_rows + (1 << rows_log2) - 1) >> rows_log2;
            ti.mi_row_starts.clear();
            let mut start_sb = 0;
            while start_sb < sb_rows {
                ti.mi_row_starts.push(start_sb << sb_shift);
                start_sb += tile_height_sb;
            }
            ti.mi_row_starts.push(mi_rows);
            ti.rows = (ti.mi_row_starts.len() - 1) as u32;
        } else {
            let mut widest_tile_sb = 0u32;
            ti.mi_col_starts.clear();
            let mut start_sb = 0u32;
            while start_sb < sb_cols {
                ti.mi_col_starts.push(start_sb << sb_shift);
                let max_width = (sb_cols - start_sb).min(max_tile_width_sb).max(1);
                let size_sb = r.ns(max_width)? + 1;
                widest_tile_sb = widest_tile_sb.max(size_sb);
                start_sb += size_sb;
                // 防御畸形数据：列数有硬上限（规范要求 TileCols <= MAX_TILE_COLS）
                if ti.mi_col_starts.len() as u32 > MAX_TILE_COLS {
                    return Err(MediaError::Bitstream(
                        "tile_info：瓦片列数超过 MAX_TILE_COLS（数据畸形或位读错位）".into(),
                    ));
                }
            }
            ti.mi_col_starts.push(mi_cols);
            ti.cols = (ti.mi_col_starts.len() - 1) as u32;
            ti.cols_log2 = tile_log2(1, ti.cols);
            if min_log2_tiles > 0 {
                max_tile_area_sb = (sb_rows * sb_cols) >> (min_log2_tiles + 1);
            } else {
                max_tile_area_sb = sb_rows * sb_cols;
            }
            let max_tile_height_sb = (max_tile_area_sb / widest_tile_sb.max(1)).max(1);
            ti.mi_row_starts.clear();
            let mut start_sb = 0u32;
            while start_sb < sb_rows {
                ti.mi_row_starts.push(start_sb << sb_shift);
                let max_height = (sb_rows - start_sb).min(max_tile_height_sb).max(1);
                let size_sb = r.ns(max_height)? + 1;
                start_sb += size_sb;
                if ti.mi_row_starts.len() as u32 > MAX_TILE_ROWS {
                    return Err(MediaError::Bitstream(
                        "tile_info：瓦片行数超过 MAX_TILE_ROWS（数据畸形或位读错位）".into(),
                    ));
                }
            }
            ti.mi_row_starts.push(mi_rows);
            ti.rows = (ti.mi_row_starts.len() - 1) as u32;
            ti.rows_log2 = tile_log2(1, ti.rows);
        }

        if ti.cols_log2 > 0 || ti.rows_log2 > 0 {
            ti.context_update_tile_id = r.f(ti.rows_log2 + ti.cols_log2)?;
            ti.tile_size_bytes = r.f(2)? + 1;
        } else {
            ti.context_update_tile_id = 0;
            // 只有一个瓦片时没有尺寸字段（末瓦片的尺寸由 tile group 的总长推出）
            ti.tile_size_bytes = 0;
        }
        if ti.context_update_tile_id >= ti.num_tiles() {
            return Err(MediaError::Bitstream(format!(
                "tile_info：context_update_tile_id {} 越界（只有 {} 个瓦片）——多半是位读错位",
                ti.context_update_tile_id,
                ti.num_tiles()
            )));
        }
        Ok(ti)
    }

    /// `quantization_params()`（规范 §5.9.6）+ `read_delta_q()`（§5.9.19）。
    fn parse_quantization(r: &mut BitReader<'_>, seq: &SequenceHeader) -> Result<QuantParams> {
        let mut q = QuantParams {
            base_q_idx: r.f(8)? as u8,
            ..Default::default()
        };
        q.delta_q_y_dc = Self::read_delta_q(r)?;
        if !seq.color.mono_chrome {
            q.diff_uv_delta = if seq.color.separate_uv_delta_q { r.bit()? } else { false };
            q.delta_q_u_dc = Self::read_delta_q(r)?;
            q.delta_q_u_ac = Self::read_delta_q(r)?;
            if q.diff_uv_delta {
                q.delta_q_v_dc = Self::read_delta_q(r)?;
                q.delta_q_v_ac = Self::read_delta_q(r)?;
            } else {
                q.delta_q_v_dc = q.delta_q_u_dc;
                q.delta_q_v_ac = q.delta_q_u_ac;
            }
        }
        q.using_qmatrix = r.bit()?;
        if q.using_qmatrix {
            q.qm_y = r.f(4)? as u8;
            q.qm_u = r.f(4)? as u8;
            if seq.color.separate_uv_delta_q {
                q.qm_v = r.f(4)? as u8;
            } else {
                q.qm_v = q.qm_u;
            }
        }
        Ok(q)
    }

    /// `read_delta_q()`：`delta_coded` 一位，之后是 `su(7)`。
    fn read_delta_q(r: &mut BitReader<'_>) -> Result<i32> {
        if r.bit()? {
            r.su(1 + 6)
        } else {
            Ok(0)
        }
    }

    /// `segmentation_params()`（规范 §5.9.8）。
    fn parse_segmentation(
        r: &mut BitReader<'_>,
        fh: &FrameHeader,
        prev: &SavedFrameState,
    ) -> Result<SegmentationParams> {
        let mut seg = SegmentationParams::default();
        seg.enabled = r.bit()?;
        if seg.enabled {
            if fh.primary_ref_frame == PRIMARY_REF_NONE {
                seg.update_map = true;
                seg.temporal_update = false;
                seg.update_data = true;
            } else {
                seg.update_map = r.bit()?;
                if seg.update_map {
                    seg.temporal_update = r.bit()?;
                }
                seg.update_data = r.bit()?;
            }
            if seg.update_data {
                for i in 0..MAX_SEGMENTS {
                    for j in 0..SEG_LVL_MAX {
                        let enabled = r.bit()?;
                        seg.feature_enabled[i][j] = enabled;
                        let mut value = 0i32;
                        if enabled {
                            let bits = SEG_FEATURE_BITS[j];
                            let limit = SEG_FEATURE_MAX[j];
                            if SEG_FEATURE_SIGNED[j] {
                                value = clip3(-limit, limit, r.su(1 + bits)?);
                            } else {
                                value = clip3(0, limit, r.f(bits)? as i32);
                            }
                        }
                        seg.feature_data[i][j] = value;
                    }
                }
            } else {
                // 沿用上一帧（`load_segmentation_params`）
                seg.feature_enabled = prev.seg_feature_enabled;
                seg.feature_data = prev.seg_feature_data;
            }
        }
        // SegIdPreSkip / LastActiveSegId（规范 §5.9.8 尾部，无位读）
        seg.seg_id_pre_skip = false;
        seg.last_active_seg_id = 0;
        for i in 0..MAX_SEGMENTS {
            for j in 0..SEG_LVL_MAX {
                if seg.feature_enabled[i][j] {
                    seg.last_active_seg_id = i as u8;
                    if j >= SEG_LVL_REF_FRAME {
                        seg.seg_id_pre_skip = true;
                    }
                }
            }
        }
        Ok(seg)
    }

    /// `CodedLossless` / `AllLossless` / `SegQMLevel`（规范 §5.9.2 尾部，无位读）。
    fn lossless_arrays(
        quant: &QuantParams,
        seg: &SegmentationParams,
        fh: &FrameHeader,
    ) -> (bool, bool, [[u8; MAX_SEGMENTS]; 3]) {
        let mut coded_lossless = true;
        let mut seg_qm_level = [[0u8; MAX_SEGMENTS]; 3];
        for s in 0..MAX_SEGMENTS {
            // get_qindex(1, s)：ignoreDeltaQ = 1 → 只看 SEG_LVL_ALT_Q 的偏移
            let qindex = if seg.feature_active(s, SEG_LVL_ALT_Q) {
                clip3(0, 255, quant.base_q_idx as i32 + seg.feature_data[s][SEG_LVL_ALT_Q])
            } else {
                quant.base_q_idx as i32
            };
            let lossless = qindex == 0
                && quant.delta_q_y_dc == 0
                && quant.delta_q_u_dc == 0
                && quant.delta_q_u_ac == 0
                && quant.delta_q_v_dc == 0
                && quant.delta_q_v_ac == 0;
            if !lossless {
                coded_lossless = false;
            }
            if quant.using_qmatrix {
                let levels = if lossless {
                    [15u8, 15, 15]
                } else {
                    [quant.qm_y, quant.qm_u, quant.qm_v]
                };
                for p in 0..3 {
                    seg_qm_level[p][s] = levels[p];
                }
            }
        }
        let all_lossless = coded_lossless && fh.frame_width == fh.upscaled_width;
        (coded_lossless, all_lossless, seg_qm_level)
    }

    /// `delta_q_params()`（规范 §5.9.9）。
    fn parse_delta_q(r: &mut BitReader<'_>, quant: &QuantParams) -> Result<DeltaQParams> {
        let mut d = DeltaQParams::default();
        if quant.base_q_idx > 0 {
            d.present = r.bit()?;
        }
        if d.present {
            d.res = r.f(2)? as u8;
        }
        Ok(d)
    }

    /// `delta_lf_params()`（规范 §5.9.10）。
    fn parse_delta_lf(
        r: &mut BitReader<'_>,
        fh: &FrameHeader,
        delta_q: &DeltaQParams,
    ) -> Result<DeltaLfParams> {
        let mut d = DeltaLfParams::default();
        if delta_q.present {
            if !fh.allow_intrabc {
                d.present = r.bit()?;
            }
            if d.present {
                d.res = r.f(2)? as u8;
                d.multi = r.bit()?;
            }
        }
        Ok(d)
    }

    /// `loop_filter_params()`（规范 §5.9.4）。
    fn parse_loop_filter(
        r: &mut BitReader<'_>,
        fh: &FrameHeader,
        seq: &SequenceHeader,
        prev: &SavedFrameState,
        coded_lossless: bool,
    ) -> Result<LoopFilterParams> {
        let mut lf = LoopFilterParams::default();
        // 没写的项沿用上一帧（`load_loop_filter_params`）
        lf.ref_deltas = prev.lf_ref_deltas;
        lf.mode_deltas = prev.lf_mode_deltas;
        if coded_lossless || fh.allow_intrabc {
            // 电平全 0，delta 用默认值（规范里这一分支直接 return）
            return Ok(lf);
        }
        lf.level[0] = r.f(6)? as u8;
        lf.level[1] = r.f(6)? as u8;
        if !seq.color.mono_chrome {
            if lf.level[0] != 0 || lf.level[1] != 0 {
                lf.level[2] = r.f(6)? as u8;
                lf.level[3] = r.f(6)? as u8;
            }
        }
        lf.sharpness = r.f(3)? as u8;
        lf.delta_enabled = r.bit()?;
        if lf.delta_enabled {
            if r.bit()? {
                for i in 0..TOTAL_REFS_PER_FRAME {
                    if r.bit()? {
                        lf.ref_deltas[i] = r.su(1 + 6)?;
                    }
                }
                for i in 0..2 {
                    if r.bit()? {
                        lf.mode_deltas[i] = r.su(1 + 6)?;
                    }
                }
            }
        }
        Ok(lf)
    }

    /// `cdef_params()`（规范 §5.9.5）。
    fn parse_cdef(
        r: &mut BitReader<'_>,
        fh: &FrameHeader,
        seq: &SequenceHeader,
        coded_lossless: bool,
    ) -> Result<CdefParams> {
        let mut c = CdefParams::default();
        if coded_lossless || fh.allow_intrabc || !seq.enable_cdef {
            return Ok(c);
        }
        c.damping = r.f(2)? as u8 + 3;
        c.bits = r.f(2)? as u8;
        for i in 0..(1usize << c.bits) {
            c.y_pri_strength[i] = r.f(4)? as u8;
            let mut y_sec = r.f(2)? as u8;
            if y_sec == 3 {
                y_sec += 1;
            }
            c.y_sec_strength[i] = y_sec;
            if !seq.color.mono_chrome {
                c.uv_pri_strength[i] = r.f(4)? as u8;
                let mut uv_sec = r.f(2)? as u8;
                if uv_sec == 3 {
                    uv_sec += 1;
                }
                c.uv_sec_strength[i] = uv_sec;
            }
        }
        Ok(c)
    }

    /// `lr_params()`（规范 §5.9.7）。
    fn parse_lr(
        r: &mut BitReader<'_>,
        fh: &FrameHeader,
        seq: &SequenceHeader,
        all_lossless: bool,
    ) -> Result<LrParams> {
        let mut lr = LrParams::default();
        if all_lossless || fh.allow_intrabc || !seq.enable_restoration {
            return Ok(lr);
        }
        let num_planes = if seq.color.mono_chrome { 1 } else { 3 };
        let remap = [RESTORE_NONE, RESTORE_SWITCHABLE, RESTORE_WIENER, RESTORE_SGRPROJ];
        for i in 0..num_planes {
            let lr_type = r.f(2)? as usize;
            lr.frame_restoration_type[i] = remap[lr_type];
            if lr.frame_restoration_type[i] != RESTORE_NONE {
                lr.uses_lr = true;
                if i > 0 {
                    lr.uses_chroma_lr = true;
                }
            }
        }
        if lr.uses_lr {
            let mut unit_shift = r.f(1)? as u8;
            if seq.use_128x128_superblock {
                unit_shift += 1;
            } else if unit_shift != 0 {
                unit_shift += r.f(1)? as u8;
            }
            lr.unit_shift = unit_shift;
            lr.size[0] = RESTORATION_TILESIZE_MAX >> (2u32.saturating_sub(unit_shift as u32));
            let lr_uv_shift = if seq.color.subsampling_x
                && seq.color.subsampling_y
                && lr.uses_chroma_lr
            {
                r.f(1)? as u32
            } else {
                0
            };
            lr.size[1] = lr.size[0] >> lr_uv_shift;
            lr.size[2] = lr.size[0] >> lr_uv_shift;
        }
        Ok(lr)
    }

    /// `read_tx_mode()`（规范 §5.9.17）。
    fn parse_tx_mode(r: &mut BitReader<'_>, coded_lossless: bool) -> Result<u8> {
        if coded_lossless {
            return Ok(ONLY_4X4);
        }
        Ok(if r.bit()? { TX_MODE_SELECT } else { TX_MODE_LARGEST })
    }

    /// `frame_reference_mode()`（规范 §5.9.16）。
    fn parse_reference_mode(r: &mut BitReader<'_>, fh: &FrameHeader) -> Result<bool> {
        if fh.is_intra {
            return Ok(false);
        }
        Ok(r.bit()?)
    }

    /// `skip_mode_params()`（规范 §5.9.15）。返回 `(skip_mode_present, SkipModeFrame)`。
    fn parse_skip_mode(
        r: &mut BitReader<'_>,
        fh: &FrameHeader,
        seq: &SequenceHeader,
        store: &RefFrameStore,
        reference_select: bool,
    ) -> Result<(bool, [u8; 2])> {
        let mut allowed = false;
        let mut frames = [0u8; 2];
        if !(fh.is_intra || !reference_select || !seq.enable_order_hint) {
            let hint_of = |i: usize| store.order_hint[fh.ref_frame_idx[i] as usize];
            let mut forward: Option<usize> = None;
            let mut backward: Option<usize> = None;
            for i in 0..REFS_PER_FRAME {
                let ref_hint = hint_of(i);
                let d = super::frame_header::get_relative_dist(seq, ref_hint, fh.order_hint);
                if d < 0 {
                    let better = match forward {
                        None => true,
                        Some(idx) => {
                            super::frame_header::get_relative_dist(seq, ref_hint, hint_of(idx)) > 0
                        }
                    };
                    if better {
                        forward = Some(i);
                    }
                } else if d > 0 {
                    let better = match backward {
                        None => true,
                        Some(idx) => {
                            super::frame_header::get_relative_dist(seq, ref_hint, hint_of(idx)) < 0
                        }
                    };
                    if better {
                        backward = Some(i);
                    }
                }
            }
            match (forward, backward) {
                (Some(f), Some(b)) => {
                    allowed = true;
                    frames[0] = (1 + f.min(b)) as u8;
                    frames[1] = (1 + f.max(b)) as u8;
                }
                (Some(f), None) => {
                    // 只有前向：再找一个"次前向"（比 forward 更早的那个）
                    let forward_hint = hint_of(f);
                    let mut second: Option<usize> = None;
                    for i in 0..REFS_PER_FRAME {
                        let ref_hint = hint_of(i);
                        if super::frame_header::get_relative_dist(seq, ref_hint, forward_hint) < 0 {
                            let better = match second {
                                None => true,
                                Some(idx) => {
                                    super::frame_header::get_relative_dist(
                                        seq,
                                        ref_hint,
                                        hint_of(idx),
                                    ) > 0
                                }
                            };
                            if better {
                                second = Some(i);
                            }
                        }
                    }
                    if let Some(s) = second {
                        allowed = true;
                        frames[0] = (1 + f.min(s)) as u8;
                        frames[1] = (1 + f.max(s)) as u8;
                    }
                }
                _ => {}
            }
        }
        let present = if allowed { r.bit()? } else { false };
        Ok((present, frames))
    }

    /// `global_motion_params()`（规范 §5.9.13）。
    ///
    /// 返回值里的 `gm_params` 会被参考帧更新过程（§7.20）存进
    /// `SavedGmParams[slot]` —— 下一帧若以本帧为主参考帧，`read_global_param()`
    /// 的 recenter 基准就取它。
    fn parse_global_motion(
        r: &mut BitReader<'_>,
        fh: &FrameHeader,
        prev: &SavedFrameState,
    ) -> Result<GlobalMotion> {
        let mut gm = GlobalMotion::identity();
        if fh.is_intra {
            return Ok(gm);
        }
        // LAST_FRAME(1) .. ALTREF_FRAME(7)
        for ref_idx in 1..TOTAL_REFS_PER_FRAME {
            let is_global = r.bit()?;
            let ty = if is_global {
                if r.bit()? {
                    ROTZOOM
                } else if r.bit()? {
                    TRANSLATION
                } else {
                    AFFINE
                }
            } else {
                IDENTITY
            };
            gm.gm_type[ref_idx] = ty;
            if ty >= ROTZOOM {
                Self::read_global_param(r, fh, ty, ref_idx, 2, &mut gm, prev)?;
                Self::read_global_param(r, fh, ty, ref_idx, 3, &mut gm, prev)?;
                if ty == AFFINE {
                    Self::read_global_param(r, fh, ty, ref_idx, 4, &mut gm, prev)?;
                    Self::read_global_param(r, fh, ty, ref_idx, 5, &mut gm, prev)?;
                } else {
                    gm.params[ref_idx][4] = -gm.params[ref_idx][3];
                    gm.params[ref_idx][5] = gm.params[ref_idx][2];
                }
            }
            if ty >= TRANSLATION {
                Self::read_global_param(r, fh, ty, ref_idx, 0, &mut gm, prev)?;
                Self::read_global_param(r, fh, ty, ref_idx, 1, &mut gm, prev)?;
            }
        }
        Ok(gm)
    }

    /// `read_global_param(type, ref, idx)`（规范 §5.9.14）。
    fn read_global_param(
        r: &mut BitReader<'_>,
        fh: &FrameHeader,
        ty: u8,
        ref_idx: usize,
        idx: usize,
        gm: &mut GlobalMotion,
        prev: &SavedFrameState,
    ) -> Result<()> {
        let mut abs_bits = GM_ABS_ALPHA_BITS;
        let mut prec_bits = GM_ALPHA_PREC_BITS;
        if idx < 2 {
            if ty == TRANSLATION {
                let hp = !fh.allow_high_precision_mv as u32;
                abs_bits = GM_ABS_TRANS_ONLY_BITS - hp;
                prec_bits = GM_TRANS_ONLY_PREC_BITS - hp as i32;
            } else {
                abs_bits = GM_ABS_TRANS_BITS;
                prec_bits = GM_TRANS_PREC_BITS;
            }
        }
        let prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
        let round = if idx % 3 == 2 { 1 << WARPEDMODEL_PREC_BITS } else { 0 };
        let sub = if idx % 3 == 2 { 1 << prec_bits } else { 0 };
        let mx = 1i32 << abs_bits;
        let r_ref = (prev.gm_params[ref_idx][idx] >> prec_diff) - sub;
        let value = Self::decode_signed_subexp_with_ref(r, -mx, mx + 1, r_ref)?;
        gm.params[ref_idx][idx] = (value << prec_diff) + round;
        Ok(())
    }

    /// `decode_signed_subexp_with_ref(low, high, r)`（规范 §5.9.14 之后）。
    fn decode_signed_subexp_with_ref(
        r: &mut BitReader<'_>,
        low: i32,
        high: i32,
        value: i32,
    ) -> Result<i32> {
        let x = Self::decode_unsigned_subexp_with_ref(r, high - low, value - low)?;
        Ok(x + low)
    }

    /// `decode_unsigned_subexp_with_ref(mx, r)`。
    fn decode_unsigned_subexp_with_ref(
        r: &mut BitReader<'_>,
        mx: i32,
        value: i32,
    ) -> Result<i32> {
        let v = Self::decode_subexp(r, mx)?;
        if (value << 1) <= mx {
            Ok(inverse_recenter(value, v))
        } else {
            Ok(mx - 1 - inverse_recenter(mx - 1 - value, v))
        }
    }

    /// `decode_subexp(numSyms)`（规范 §5.9.14 之后）：子指数哥伦布。
    fn decode_subexp(r: &mut BitReader<'_>, num_syms: i32) -> Result<i32> {
        if num_syms <= 0 {
            return Ok(0);
        }
        let mut i = 0i32;
        let mut mk = 0i32;
        let k = 3i32;
        for _ in 0..32 {
            let b2 = if i != 0 { k + i - 1 } else { k };
            let a = 1i32 << b2;
            if num_syms <= mk + 3 * a {
                let bits = r.ns((num_syms - mk) as u32)? as i32;
                return Ok(bits + mk);
            }
            if r.bit()? {
                i += 1;
                mk += a;
            } else {
                let bits = r.f(b2 as u32)? as i32;
                return Ok(bits + mk);
            }
        }
        Err(MediaError::Bitstream("decode_subexp 超过 32 轮（位读错位）".into()))
    }

    /// `film_grain_params()`（规范 §5.9.30）。
    ///
    /// `update_grain == 0` 时走 `load_grain_params(ref_idx)`：参数从参考槽整体拷贝
    /// （不读位），只覆盖 `grain_seed`。参考槽必须属于本帧的参考列表且有效
    /// （对应规范里 `ref_idx` 要在 `ref_frame_idx[]` 中、且槽里有帧）。
    fn parse_film_grain(
        r: &mut BitReader<'_>,
        fh: &FrameHeader,
        seq: &SequenceHeader,
        store: &RefFrameStore,
    ) -> Result<FilmGrainParams> {
        let mut fg = FilmGrainParams::default();
        if !seq.film_grain_params_present || (!fh.show_frame && !fh.showable_frame) {
            return Ok(fg);
        }
        fg.apply_grain = r.bit()?;
        if !fg.apply_grain {
            return Ok(fg);
        }
        fg.grain_seed = r.f(16)? as u16;
        fg.update_grain = if fh.frame_type == FrameType::Inter { r.bit()? } else { true };
        if !fg.update_grain {
            let ref_idx = r.f(3)? as u8;
            if !fh.ref_frame_idx.contains(&ref_idx) || !store.valid[ref_idx as usize] {
                return Err(MediaError::Bitstream(format!(
                    "film_grain update_grain=0 引用槽 {ref_idx}，但该槽不是本帧的参考帧或无效"
                )));
            }
            // load_grain_params(ref_idx)：整体拷贝参考槽的颗粒参数，仅覆盖种子
            let mut loaded = store.grain[ref_idx as usize].clone();
            loaded.grain_seed = fg.grain_seed;
            loaded.ref_idx = ref_idx;
            loaded.update_grain = false;
            return Ok(loaded);
        }
        fg.num_y_points = r.f(4)? as u8;
        if fg.num_y_points > 14 {
            return Err(MediaError::Bitstream(format!(
                "num_y_points = {} 超过 14",
                fg.num_y_points
            )));
        }
        for i in 0..fg.num_y_points as usize {
            fg.y_points[i][0] = r.f(8)? as u8;
            fg.y_points[i][1] = r.f(8)? as u8;
            if i > 0 && fg.y_points[i - 1][0] >= fg.y_points[i][0] {
                return Err(MediaError::Bitstream("point_y_value 未严格递增".into()));
            }
        }
        fg.chroma_scaling_from_luma = if seq.color.mono_chrome { false } else { r.bit()? };
        let skip_chroma = seq.color.mono_chrome
            || fg.chroma_scaling_from_luma
            || (seq.color.subsampling_x && seq.color.subsampling_y && fg.num_y_points == 0);
        if skip_chroma {
            fg.num_uv_points = [0, 0];
        } else {
            for pl in 0..2usize {
                let n = r.f(4)? as u8;
                if n > 10 {
                    return Err(MediaError::Bitstream(format!(
                        "num_uv_points[{pl}] = {n} 超过 10"
                    )));
                }
                fg.num_uv_points[pl] = n;
                for i in 0..n as usize {
                    fg.uv_points[pl][i][0] = r.f(8)? as u8;
                    fg.uv_points[pl][i][1] = r.f(8)? as u8;
                    if i > 0 && fg.uv_points[pl][i - 1][0] >= fg.uv_points[pl][i][0] {
                        return Err(MediaError::Bitstream(format!(
                            "point_uv_value[{pl}] 未严格递增"
                        )));
                    }
                }
            }
            // 4:2:0 时两个色度平面要么都有点表、要么都没有
            if seq.color.subsampling_x
                && seq.color.subsampling_y
                && (fg.num_uv_points[0] > 0) != (fg.num_uv_points[1] > 0)
            {
                return Err(MediaError::Bitstream(
                    "4:2:0 下 Cb/Cr 点表存在性不一致".into(),
                ));
            }
        }
        fg.scaling_shift = r.f(2)? as u8 + 8;
        fg.ar_coeff_lag = r.f(2)? as u8;
        let num_pos_luma = 2 * fg.ar_coeff_lag as u32 * (fg.ar_coeff_lag as u32 + 1);
        let num_pos_chroma =
            if fg.num_y_points > 0 { num_pos_luma + 1 } else { num_pos_luma };
        if fg.num_y_points > 0 {
            for i in 0..num_pos_luma as usize {
                fg.ar_coeffs_y[i] = (r.f(8)? as i32 - 128) as i8;
            }
        }
        for pl in 0..2usize {
            if fg.chroma_scaling_from_luma || fg.num_uv_points[pl] > 0 {
                for i in 0..num_pos_chroma as usize {
                    fg.ar_coeffs_uv[pl][i] = (r.f(8)? as i32 - 128) as i8;
                }
                if fg.num_y_points == 0 {
                    fg.ar_coeffs_uv[pl][num_pos_chroma as usize] = 0;
                }
            }
        }
        fg.ar_coeff_shift = r.f(2)? as u8 + 6;
        fg.grain_scale_shift = r.f(2)? as u8;
        for pl in 0..2usize {
            if fg.num_uv_points[pl] > 0 {
                fg.uv_mult[pl] = (r.f(8)? as i32 - 128) as i16;
                fg.uv_luma_mult[pl] = (r.f(8)? as i32 - 128) as i16;
                fg.uv_offset[pl] = (r.f(9)? as i32 - 256) as i16;
            }
        }
        fg.overlap_flag = r.bit()?;
        fg.clip_to_restricted_range = r.bit()?;
        Ok(fg)
    }
}

/// `inverse_recenter(r, v)`（规范 §5.9.14 之后）。
pub(crate) fn inverse_recenter(r: i32, v: i32) -> i32 {
    if v > 2 * r {
        v
    } else if v & 1 != 0 {
        r - ((v + 1) >> 1)
    } else {
        r + (v >> 1)
    }
}

/// `tile_log2` 对外的只读版本（测试与诊断用）。
pub fn tile_log2_pub(blk_size: u32, target: u32) -> u32 {
    tile_log2(blk_size, target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_log2_matches_spec_examples() {
        // 4096 >> 6 = 64 → ceil(log2(64)) = 6
        assert_eq!(tile_log2_pub(64, 120), 1, "64 << 1 = 128 >= 120");
        assert_eq!(tile_log2_pub(1, 1), 0, "已经够大就是 0");
        assert_eq!(tile_log2_pub(1, 2), 1);
        assert_eq!(tile_log2_pub(1, 3), 2);
    }

    #[test]
    fn inverse_recenter_matches_spec() {
        // 规范 §5.9.14 之后的伪代码：
        //   v > 2r → v；v 为奇数 → r - ((v+1) >> 1)；否则 r + (v >> 1)
        assert_eq!(inverse_recenter(0, 0), 0, "0 > 0 不成立 → r + 0");
        assert_eq!(inverse_recenter(0, 1), 1, "1 > 0 → 直接是 v");
        assert_eq!(inverse_recenter(0, 2), 2);
        // r = 3 时以 3 为中心：v=1/2/3/4 分别落到 2/4/1/5
        assert_eq!(inverse_recenter(3, 1), 2, "奇数 → 3 - 1");
        assert_eq!(inverse_recenter(3, 2), 4, "偶数 → 3 + 1");
        assert_eq!(inverse_recenter(3, 3), 1, "奇数 → 3 - 2");
        assert_eq!(inverse_recenter(3, 4), 5, "偶数 → 3 + 2");
        // 超过 2r 直接透传
        assert_eq!(inverse_recenter(1, 5), 5);
    }

    #[test]
    fn saved_state_defaults_match_past_independence() {
        let s = SavedFrameState::past_independence();
        assert_eq!(s.lf_ref_deltas[0], 1, "INTRA_FRAME");
        assert_eq!(s.lf_ref_deltas[1], 0, "LAST_FRAME");
        assert_eq!(s.lf_ref_deltas[4], -1, "GOLDEN_FRAME");
        assert_eq!(s.lf_ref_deltas[7], -1, "ALTREF_FRAME");
        assert_eq!(s.lf_mode_deltas, [0, 0]);
        assert_eq!(s.gm_params[1][2], 1 << WARPEDMODEL_PREC_BITS, "第三列是 1.0");
        assert_eq!(s.gm_params[1][0], 0);
    }

    #[test]
    fn identity_global_motion_is_identity() {
        let gm = GlobalMotion::identity();
        assert!(gm.gm_type.iter().all(|&t| t == IDENTITY));
        for p in gm.params.iter() {
            assert_eq!(p[2], 1 << WARPEDMODEL_PREC_BITS);
            assert_eq!(p[5], 1 << WARPEDMODEL_PREC_BITS);
            assert_eq!(p[0], 0);
        }
    }

    #[test]
    fn segmentation_feature_active_requires_enabled() {
        let mut seg = SegmentationParams::default();
        seg.feature_enabled[3][SEG_LVL_ALT_Q] = true;
        assert!(!seg.feature_active(3, SEG_LVL_ALT_Q), "整体没开时不算激活");
        seg.enabled = true;
        assert!(seg.feature_active(3, SEG_LVL_ALT_Q));
        assert!(!seg.feature_active(3, 1));
    }
}
