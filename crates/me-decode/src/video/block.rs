//! AV1 **块级语法**（规范 §5.11.1 `decode_block` / §5.11.2 `mode_info`）—— 自研软解 M4b。
//!
//! 分区树给出叶子块之后，每个叶子走一遍块级语法；**只有把整条算术流按真实语法
//! 消耗干净，`exit_symbol()` 的 trailing 检查才会落在编码器写下的位置上** ——
//! 那是熵层的最终硬验收（`me-cli` 的 `real_av1_tests`）。
//!
//! ```text
//! decode_block( r, c, subSize )
//!   ├─ HasChroma / AvailU / AvailL / Avail*Chroma   （推导，不读位）
//!   ├─ mode_info()                                   ← 本文件
//!   ├─ palette_tokens()                              ← ✅ 屏幕内容流（§5.11.19）
//!   ├─ read_block_tx_size()                          ← 本文件
//!   ├─ if ( skip ) reset_block_context()             ← coeff.rs
//!   ├─ 写 YModes / UVModes / RefFrames / CompGroupIdxs / CompoundIdxs /
//!   │     InterpFilters / Mvs                        （每 4×4）
//!   ├─ compute_prediction()                          ← M5（不读位）
//!   ├─ residual() → transform_tree → coeffs          ← coeff.rs
//!   └─ 写 IsInters / SkipModes / Skips / TxSizes / MiSizes / SegmentIds …（每 4×4）
//! ```
//!
//! ## 做完的部分
//!
//! ```text
//! ✅ intra_frame_mode_info：段号 → skip → CDEF → delta_q/delta_lf → 模式 → 滤波
//! ✅ inter_frame_mode_info：跳过模式、段号、read_is_inter
//! ✅ inter_block_mode_info：read_ref_frames（单/复合、单向复合、双向复合）
//!      → find_mv_stack → 模式（compound_mode / new_mv / zero_mv / ref_mv）
//!      → read_drl_idx → assign_mv（read_mv）→ interintra → motion_mode
//!      → compound_type → interp_filter
//! ✅ intra_block_mode_info（帧间帧里的帧内块，用 `y_mode` 那套 CDF）
//! ✅ read_block_tx_size / read_tx_size / read_var_tx_size / txfm_split
//! ✅ 系数（[`super::coeff`]）
//! ✅ decode_tile：超块扫描 + 分区树 + 块级语法 + `exit_symbol()`
//! ✅ palette_mode_info / palette_tokens（§5.11.18/19）与 use_intrabc：屏幕内容流，
//!    2026-10-01 补齐；像素侧在 [`super::intra::predict_palette`] 与 [`super::inter`]
//! ✅ LOCALWARP：`warp_plan()` 在这里算 `useWarp`（要 `is_scaled` 与最小二乘，
//!    只有这一层拿得到），像素侧在 [`super::inter`] + [`super::warp`]
//! ✅ OBMC 的像素侧：`motion_mode` 读出来之后在 [`super::inter`] 里混合
//!    （§7.11.3.9 + §7.11.3.10，2026-09-30 夜补）
//! ```
//!
//! ## 三个必须记住的点
//!
//! 1. **帧内帧用 `intra_frame_y_mode`（邻居模式当上下文），帧间帧里的帧内块用
//!    `y_mode`（按块尺寸分组当上下文）** —— 两套 CDF，混用会整块错位。
//! 2. **`is_inside` 用瓦片范围**（`MiRowStart/End`、`MiColStart/End`），
//!    `hasRows/hasCols` 用帧范围 —— 与 `partition.rs` 同一个口径。
//! 3. **`skip` 的块不读系数，而且要把这一块覆盖到的 `Above/LeftLevelContext`
//!    清零**（`reset_block_context`，规范在 `read_block_tx_size` 之后立刻调）。

use super::flags;
use me_core::Result;

use super::cdf_tables as T;
use super::coeff::{
    CoeffCdfs, CoeffContexts, CoeffStats, ResidualConfig, ResidualCtx, TxTypeGrid, MI_SIZE,
};
use super::context::CdfContext;
use super::entropy::SymbolDecoder;
use super::frame_header::{FrameHeader, RefFrameStore};
use super::mv;
use super::mv::{
    get_mode, is_scaled, Mv, MvStackInput, RefMvStack, GLOBALMV, NEARMV, NEARESTMV,
    NEAREST_NEARESTMV, NEWMV, NEW_NEWMV, NONE,
};
use super::params::{
    FRAME_LF_COUNT, MAX_LOOP_FILTER, MAX_SEGMENTS, SEG_LVL_ALT_Q, SEG_LVL_GLOBALMV, SEG_LVL_REF_FRAME,
    SEG_LVL_SKIP, TX_MODE_SELECT,
};
use super::partition::{self, MiGrid};
/// 分区/块尺寸常量统一从 `block` 再导出，`coeff.rs` 与 `mv.rs` 只认这里。
pub use super::partition::{
    BLOCK_128X128, BLOCK_16X16, BLOCK_16X32, BLOCK_16X4, BLOCK_16X64, BLOCK_16X8, BLOCK_32X16,
    BLOCK_32X32, BLOCK_32X64, BLOCK_32X8, BLOCK_4X16, BLOCK_4X4, BLOCK_4X8, BLOCK_64X128,
    BLOCK_64X16, BLOCK_64X32, BLOCK_64X64, BLOCK_8X16, BLOCK_8X32, BLOCK_8X4, BLOCK_8X8,
    BLOCK_128X64, BLOCK_INVALID, BLOCK_SIZES, MI_HEIGHT_LOG2, MI_WIDTH_LOG2, NUM_4X4_BLOCKS_HIGH,
    NUM_4X4_BLOCKS_WIDE,
};
use super::sequence_header::SequenceHeader;
use super::tile::Tile;
use super::intra::CurrFrame;
use super::lr::LrState;
/// warp 的逻辑（`setup_shear` / `warp_estimate`）。起别名是因为本模块里也有
/// 叫 `warp` 的局部变量。
use super::warp as warping;

// ── 帧内预测模式（规范 §6.10.1）────────────────────────────────────────

pub const DC_PRED: u8 = 0;
pub const V_PRED: u8 = 1;
pub const H_PRED: u8 = 2;
pub const D45_PRED: u8 = 3;
pub const D135_PRED: u8 = 4;
pub const D113_PRED: u8 = 5;
pub const D157_PRED: u8 = 6;
pub const D203_PRED: u8 = 7;
pub const D67_PRED: u8 = 8;
pub const SMOOTH_PRED: u8 = 9;
pub const SMOOTH_V_PRED: u8 = 10;
pub const SMOOTH_H_PRED: u8 = 11;
pub const PAETH_PRED: u8 = 12;
/// `INTRA_MODES`
pub const INTRA_MODES: u8 = 13;
/// 色度专有的 CFL 模式（`uv_mode` 的第 14 个取值）
pub const UV_CFL_PRED: u8 = 13;

/// `MAX_ANGLE_DELTA`
pub const MAX_ANGLE_DELTA: i8 = 3;
/// `DIRECTIONAL_MODES`
pub const DIRECTIONAL_MODES: usize = 8;
/// `INTRA_MODE_CONTEXTS`
pub const INTRA_MODE_CONTEXTS: usize = 5;
/// `BLOCK_SIZE_GROUPS`
pub const BLOCK_SIZE_GROUPS: usize = 4;
/// `SEGMENT_ID_CONTEXTS`
pub const SEGMENT_ID_CONTEXTS: usize = 3;
/// `SEGMENT_ID_PREDICTED_CONTEXTS`
pub const SEGMENT_ID_PREDICTED_CONTEXTS: usize = 3;
/// `DELTA_Q_SMALL`
pub const DELTA_Q_SMALL: u32 = 3;
/// `DELTA_LF_SMALL`
pub const DELTA_LF_SMALL: u32 = 3;
/// `CFL_SIGN_ZERO / NEG`
pub const CFL_SIGN_ZERO: i32 = 0;
pub const CFL_SIGN_NEG: i32 = 1;
/// `CFL_ALPHA_CONTEXTS`
pub const CFL_ALPHA_CONTEXTS: usize = 6;

// ── 帧间相关的枚举 ──────────────────────────────────────────────────────

/// `comp_mode`
pub const SINGLE_REFERENCE: u8 = 0;
pub const COMPOUND_REFERENCE: u8 = 1;
/// `comp_ref_type`
pub const UNIDIR_COMP_REFERENCE: u8 = 0;
pub const BIDIR_COMP_REFERENCE: u8 = 1;
/// `compound_type`（规范 §6.10.6）
pub const COMPOUND_WEDGE: u8 = 0;
pub const COMPOUND_DIFFWTD: u8 = 1;
pub const COMPOUND_AVERAGE: u8 = 2;
pub const COMPOUND_INTRA: u8 = 3;
pub const COMPOUND_DISTANCE: u8 = 4;

/// `interintra_mode`（规范 §6.10.6 `read_interintra_mode`）。
///
/// **不是** `YMode`：`interintra_mode` 只有 4 个值，像素侧要把它映射成
/// `DC_PRED` / `V_PRED` / `H_PRED` / `SMOOTH_PRED`（见 [`super::inter`]）。
/// 值必须与符号取值一致（0..3），错一位在真实素材上就是整块像素错。
pub const II_DC_PRED: u8 = 0;
pub const II_V_PRED: u8 = 1;
pub const II_H_PRED: u8 = 2;
pub const II_SMOOTH_PRED: u8 = 3;

// ── 屏幕内容工具：palette（§5.11.18 / §5.11.19 / §7.11.4）───────────────
//
// palette 把一块限制到最多 8 种颜色，再逐样本编码"颜色索引"。颜色表在
// `palette_mode_info` 里读（并与上/左邻居的调色板合并成 cache），索引图在
// `palette_tokens` 里按 `get_palette_color_context` 的局部频率重排后解码，
// 重建时 `predict_palette` 直接拿索引查表填像素（规范 §7.11.4）。
//
// 这三步**必须一起做**：只读标志位不做 tokens 会让算术流错位，只做 tokens
// 不做 predict_palette 则像素全错。

/// `PALETTE_COLORS`：一块最多几种颜色。
pub const PALETTE_COLORS: usize = 8;
/// `PALETTE_SIZES`：`palette_size_*_minus_2` 的取值个数（尺寸 2..=8）。
pub const PALETTE_SIZES: usize = 7;
/// `PALETTE_NUM_NEIGHBORS`：`get_palette_color_context` 只给分数最高的前三名排序。
pub const PALETTE_NUM_NEIGHBORS: usize = 3;
/// `PALETTE_MAX_COLOR_CONTEXT_HASH`
pub const PALETTE_MAX_COLOR_CONTEXT_HASH: usize = 8;

/// `Palette_Color_Context` 的取值个数（0..=4）：颜色索引 CDF 的上下文数。
pub const PALETTE_COLOR_CONTEXTS: usize = 5;

/// `Palette_Color_Hash_Multipliers`（§9）：把邻域打分折成 `ColorContextHash`。
pub const PALETTE_COLOR_HASH_MULTIPLIERS: [i32; PALETTE_NUM_NEIGHBORS] = [1, 2, 2];

/// `Palette_Color_Context[ PALETTE_MAX_COLOR_CONTEXT_HASH + 1 ]`（§9）。
/// 负数表示"这个哈希值取不到"，规范注明"永远不会被访问"。
pub const PALETTE_COLOR_CONTEXT: [i8; PALETTE_MAX_COLOR_CONTEXT_HASH + 1] =
    [-1, -1, 0, -1, -1, 4, 3, 2, 1];

/// `COMPOUND_MODE_CONTEXTS`
pub const COMPOUND_MODE_CONTEXTS: usize = 8;
/// `COMP_NEWMV_CTXS`
pub const COMP_NEWMV_CTXS: usize = 5;
/// `interp_filter`（规范 §6.10.3）
pub const EIGHTTAP: u8 = 0;
pub const EIGHTTAP_SMOOTH: u8 = 1;
pub const EIGHTTAP_SHARP: u8 = 2;
pub const BILINEAR: u8 = 3;
pub const SWITCHABLE: u8 = 4;
/// `INTERP_FILTER_CONTEXTS`
pub const INTERP_FILTER_CONTEXTS: usize = 16;
/// `INTERP_FILTERS`：块级 `interp_filter` 符号的取值个数（`BILINEAR` 不在其中）
pub const INTERP_FILTERS: usize = 3;
/// `MOTION_MODES`
pub const SIMPLE: u8 = 0;
pub const OBMC: u8 = 1;
pub const LOCALWARP: u8 = 2;
/// `MV_JOINTS`
pub const MV_JOINTS: usize = 4;
/// `MV_CONTEXTS`
pub const MV_CONTEXTS: usize = 2;
/// `MV_OFFSET_BITS`
pub const MV_OFFSET_BITS: usize = 10;
/// `CLASS0_SIZE`
pub const CLASS0_SIZE: usize = 2;
/// `MV_JOINT` 的取值
pub const MV_JOINT_ZERO: u32 = 0;
pub const MV_JOINT_HNZVZ: u32 = 1;
pub const MV_JOINT_HZVNZ: u32 = 2;
pub const MV_JOINT_HNZVNZ: u32 = 3;
/// `MV_CLASS_0`
pub const MV_CLASS_0: u32 = 0;

/// `Wedge_Bits[ BLOCK_SIZES ]`（§9.5 原文：`0,0,0,4,4,4,4,4,4,4,0,` + `0,0,0,0,0,0,0,4,4,0,0`）。
///
/// **只有 8×8 – 32×32 那一族，外加 `BLOCK_8X32`(18) / `BLOCK_32X8`(19)**；
/// `4×16`(16) / `16×4`(17) 是 0（一边只有 4，太窄），`64×64` 及以上也是 0。
/// 这张表决定 `read_compound_type` 走 `n == 0 → COMPOUND_DIFFWTD` 还是读
/// `compound_type` 符号 —— **错一格就是整条算术流错位**。
pub const WEDGE_BITS: [u8; BLOCK_SIZES] =
    [0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0];

/// `Compound_Mode_Ctx_Map[ 3 ][ COMP_NEWMV_CTXS ]`
pub const COMPOUND_MODE_CTX_MAP: [[usize; COMP_NEWMV_CTXS]; 3] =
    [[0, 1, 1, 1, 1], [1, 2, 3, 4, 4], [4, 4, 5, 6, 7]];

// ── 变换尺寸 ────────────────────────────────────────────────────────────

pub const TX_4X4: u8 = 0;
pub const TX_8X8: u8 = 1;
pub const TX_16X16: u8 = 2;
pub const TX_32X32: u8 = 3;
pub const TX_64X64: u8 = 4;
pub const TX_4X8: u8 = 5;
pub const TX_8X4: u8 = 6;
pub const TX_8X16: u8 = 7;
pub const TX_16X8: u8 = 8;
pub const TX_16X32: u8 = 9;
pub const TX_32X16: u8 = 10;
pub const TX_32X64: u8 = 11;
pub const TX_64X32: u8 = 12;
pub const TX_4X16: u8 = 13;
pub const TX_16X4: u8 = 14;
pub const TX_8X32: u8 = 15;
pub const TX_32X8: u8 = 16;
pub const TX_16X64: u8 = 17;
pub const TX_64X16: u8 = 18;
/// `TX_SIZES_ALL`
pub const TX_SIZES_ALL: usize = 19;
/// `TX_SIZES`
pub const TX_SIZES: usize = 5;
/// `MAX_TX_DEPTH`
pub const MAX_TX_DEPTH: u32 = 2;
/// `MAX_VARTX_DEPTH`
pub const MAX_VARTX_DEPTH: u32 = 2;
/// `TX_SIZE_CONTEXTS`
pub const TX_SIZE_CONTEXTS: usize = 3;

/// `Tx_Width[ TX_SIZES_ALL ]`
pub const TX_WIDTH: [u16; TX_SIZES_ALL] =
    [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
/// `Tx_Height[ TX_SIZES_ALL ]`
pub const TX_HEIGHT: [u16; TX_SIZES_ALL] =
    [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];
/// `Tx_Width_Log2[ TX_SIZES_ALL ]`
pub const TX_WIDTH_LOG2: [u8; TX_SIZES_ALL] = [2, 3, 4, 5, 6, 2, 3, 3, 4, 4, 5, 5, 6, 2, 4, 3, 5, 4, 6];
/// `Tx_Height_Log2[ TX_SIZES_ALL ]`
pub const TX_HEIGHT_LOG2: [u8; TX_SIZES_ALL] = [2, 3, 4, 5, 6, 3, 2, 4, 3, 5, 4, 6, 5, 4, 2, 5, 3, 6, 4];
/// `Split_Tx_Size[ TX_SIZES_ALL ]`
pub const SPLIT_TX_SIZE: [u8; TX_SIZES_ALL] = [0, 0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3, 5, 6, 7, 8, 9, 10];
/// `Max_Tx_Size_Rect[ BLOCK_SIZES ]`
pub const MAX_TX_SIZE_RECT: [u8; BLOCK_SIZES] = [
    TX_4X4, TX_4X8, TX_8X4, TX_8X8, TX_8X16, TX_16X8, TX_16X16, TX_16X32, TX_32X16, TX_32X32,
    TX_32X64, TX_64X32, TX_64X64, TX_64X64, TX_64X64, TX_64X64, TX_4X16, TX_16X4, TX_8X32,
    TX_32X8, TX_16X64, TX_64X16,
];
/// `Max_Tx_Depth[ BLOCK_SIZES ]`
pub const MAX_TX_DEPTH_BY_BSIZE: [u8; BLOCK_SIZES] =
    [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 2, 2, 3, 3, 4, 4];

/// `Size_Group[ BLOCK_SIZES ]`
pub const SIZE_GROUP: [usize; BLOCK_SIZES] =
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 2, 2];

/// `Intra_Mode_Context[ INTRA_MODES ]`
pub const INTRA_MODE_CONTEXT: [usize; INTRA_MODES as usize] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];

/// 块宽度（像素）。越界给 0。
#[inline]
pub fn block_width(bsize: u8) -> i32 {
    if bsize as usize >= BLOCK_SIZES {
        return 0;
    }
    4i32 << partition::MI_WIDTH_LOG2[bsize as usize]
}

/// 块高度（像素）。
#[inline]
pub fn block_height(bsize: u8) -> i32 {
    if bsize as usize >= BLOCK_SIZES {
        return 0;
    }
    4i32 << partition::MI_HEIGHT_LOG2[bsize as usize]
}

/// `is_directional_mode( mode )`
#[inline]
pub fn is_directional_mode(mode: u8) -> bool {
    (V_PRED..=D67_PRED).contains(&mode)
}

/// `has_newmv( mode )`
#[inline]
pub fn has_newmv(mode: u8) -> bool {
    matches!(mode, NEWMV | NEW_NEWMV | mv::NEAR_NEWMV | mv::NEW_NEARMV | mv::NEAREST_NEWMV | mv::NEW_NEARESTMV)
}

/// `has_nearmv( )`
#[inline]
pub fn has_nearmv(ymode: u8) -> bool {
    matches!(ymode, NEARMV | mv::NEAR_NEARMV | mv::NEAR_NEWMV | mv::NEW_NEARMV)
}

// ── 瓦片级的模式 CDF ───────────────────────────────────────────────────

/// 一个瓦片的**模式信息 CDF 副本**（规范 `init_symbol` 时从 `Default_*` 拷一份）。
///
/// `mv_joint` 之类的表在规范里是 `MvJointCdf[i] is set to a copy of
/// Default_Mv_Joint_Cdf for i = 0..MV_CONTEXTS-1`（`Default_*` 本身没有
/// `MvCtx` 那一维，`init` 时复制）。
#[derive(Debug, Clone)]
pub struct BlockCdfs {
    pub intrabc: [u16; 3],
    /// 环路恢复的块级符号（§5.11.48）。三种帧级恢复类型**都会**用到：
    /// `RESTORE_WIENER` 读 `use_wiener`、`RESTORE_SGRPROJ` 读 `use_sgrproj`、
    /// `RESTORE_SWITCHABLE` 三个都读。**都是真 CDF 符号**（不是 `read_bool`）。
    pub lr_use_wiener: [u16; 3],
    pub lr_use_sgrproj: [u16; 3],
    pub lr_restoration_type: [u16; 4],
    pub intra_frame_y_mode: [[[u16; 14]; INTRA_MODE_CONTEXTS]; INTRA_MODE_CONTEXTS],
    pub y_mode: [[u16; 14]; BLOCK_SIZE_GROUPS],
    pub uv_mode_cfl_allowed: [[u16; 15]; INTRA_MODES as usize],
    pub uv_mode_cfl_not_allowed: [[u16; 14]; INTRA_MODES as usize],
    pub angle_delta: [[u16; 8]; DIRECTIONAL_MODES],
    pub filter_intra: [[u16; 3]; BLOCK_SIZES],
    pub filter_intra_mode: [u16; 6],
    pub skip: [[u16; 3]; 3],
    pub skip_mode: [[u16; 3]; 3],
    pub segment_id: [[u16; 9]; SEGMENT_ID_CONTEXTS],
    pub segment_id_predicted: [[u16; 3]; SEGMENT_ID_PREDICTED_CONTEXTS],
    pub delta_q: [u16; DELTA_Q_SMALL as usize + 2],
    pub delta_lf: [u16; DELTA_LF_SMALL as usize + 2],
    pub cfl_sign: [u16; 9],
    pub cfl_alpha: [[u16; 17]; CFL_ALPHA_CONTEXTS],
    /// 屏幕内容工具的 palette 标志位（`TilePaletteYModeCdf`）。
    /// 这两个每块都要读、每读一次都会自适应 —— 漏了它 CDF 上下文就与规范不一致。
    pub palette_y_mode: [[[u16; 3]; 3]; 7],
    pub palette_uv_mode: [[u16; 3]; 2],
    /// `palette_size_*_minus_2` 的 CDF（`TilePaletteYSizeCdf[ bsizeCtx ]`）：
    /// 7 个符号（尺寸 2..=8）+ 计数器。
    pub palette_y_size: [[u16; 8]; 7],
    pub palette_uv_size: [[u16; 8]; 7],
    /// `palette_color_idx_*`：按 `PaletteSize`（2..=8）分 7 档，每档 5 个
    /// `Palette_Color_Context`。规范里 7 张表长度是 3..9，这里统一补齐到 9 项
    /// （最大尺寸 8 + 计数器），读的时候按 `&mut cdf[..size + 1]` 切片。
    pub palette_color_y: [[[u16; 9]; PALETTE_COLOR_CONTEXTS]; PALETTE_SIZES],
    pub palette_color_uv: [[[u16; 9]; PALETTE_COLOR_CONTEXTS]; PALETTE_SIZES],
    pub tx8x8: [[u16; MAX_TX_DEPTH as usize + 1]; TX_SIZE_CONTEXTS],
    pub tx16x16: [[u16; MAX_TX_DEPTH as usize + 2]; TX_SIZE_CONTEXTS],
    pub tx32x32: [[u16; MAX_TX_DEPTH as usize + 2]; TX_SIZE_CONTEXTS],
    pub tx64x64: [[u16; MAX_TX_DEPTH as usize + 2]; TX_SIZE_CONTEXTS],
    pub txfm_split: [[u16; 3]; 21],
    pub is_inter: [[u16; 3]; 4],
    pub comp_mode: [[u16; 3]; 5],
    pub comp_ref: [[[u16; 3]; 3]; 3],
    pub comp_bwdref: [[[u16; 3]; 2]; 3],
    pub single_ref: [[[u16; 3]; 6]; 3],
    pub comp_ref_type: [[u16; 3]; 5],
    pub uni_comp_ref: [[[u16; 3]; 3]; 3],
    pub compound_mode: [[u16; 9]; COMPOUND_MODE_CONTEXTS],
    pub new_mv: [[u16; 3]; 6],
    pub zero_mv: [[u16; 3]; 2],
    pub ref_mv: [[u16; 3]; 6],
    pub drl_mode: [[u16; 3]; 3],
    pub interintra: [[u16; 3]; 3],
    pub interintra_mode: [[u16; 5]; 3],
    pub wedge_interintra: [[u16; 3]; BLOCK_SIZES],
    pub wedge_index: [[u16; 17]; BLOCK_SIZES],
    pub use_obmc: [[u16; 3]; BLOCK_SIZES],
    pub motion_mode: [[u16; 4]; BLOCK_SIZES],
    pub compound_type: [[u16; 3]; BLOCK_SIZES],
    pub comp_group_idx: [[u16; 3]; 6],
    pub compound_idx: [[u16; 3]; 6],
    pub interp_filter: [[u16; INTERP_FILTERS + 1]; INTERP_FILTER_CONTEXTS],
    pub mv_joint: [[u16; MV_JOINTS + 1]; MV_CONTEXTS],
    pub mv_class: [[[u16; 12]; 2]; MV_CONTEXTS],
    pub mv_class0_bit: [[[u16; 3]; 2]; MV_CONTEXTS],
    pub mv_class0_fr: [[[[u16; 5]; CLASS0_SIZE]; 2]; MV_CONTEXTS],
    pub mv_class0_hp: [[[u16; 3]; 2]; MV_CONTEXTS],
    pub mv_fr: [[[u16; 5]; 2]; MV_CONTEXTS],
    pub mv_hp: [[[u16; 3]; 2]; MV_CONTEXTS],
    pub mv_bit: [[[[u16; 3]; MV_OFFSET_BITS]; 2]; MV_CONTEXTS],
    pub mv_sign: [[[u16; 3]; 2]; MV_CONTEXTS],
}

/// 把"长度随尺寸变化"的颜色索引 CDF 补齐成定长 9（最大尺寸 8 + 计数器）。
///
/// 规范里那 7 张表长度分别是 3..9（`PaletteSize + 1`）。解码时按
/// `&mut cdf[..size + 1]` 切片，所以尾部补的 0 不会参与（[`SymbolDecoder::read_symbol`]
/// 只看切片本身）。不补的话 7 档就得写成 7 个不同类型的字段，索引起来很别扭。
fn palette_color_cdf<const N: usize>(src: [[u16; N]; 5]) -> [[u16; 9]; 5] {
    let mut out = [[0u16; 9]; 5];
    for (dst, s) in out.iter_mut().zip(src.iter()) {
        dst[..N].copy_from_slice(s);
    }
    out
}

/// `CeilLog2( x )`（规范 §4.7）：`x <= 1` 时为 0，否则是"包住 x 的最小位数"。
///
/// palette 的颜色增量用它收窄 `paletteBits`（`range` 已经小到不用那么多位了）。
fn ceil_log2(x: i32) -> u32 {
    if x <= 1 {
        0
    } else {
        32 - (x - 1).leading_zeros()
    }
}

impl Default for BlockCdfs {
    fn default() -> Self {
        Self {
            intrabc: T::DEFAULT_INTRABC_CDF,
            lr_use_wiener: T::DEFAULT_USE_WIENER_CDF,
            lr_use_sgrproj: T::DEFAULT_USE_SGRPROJ_CDF,
            lr_restoration_type: T::DEFAULT_RESTORATION_TYPE_CDF,
            intra_frame_y_mode: T::DEFAULT_INTRA_FRAME_Y_MODE_CDF,
            y_mode: T::DEFAULT_Y_MODE_CDF,
            uv_mode_cfl_allowed: T::DEFAULT_UV_MODE_CFL_ALLOWED_CDF,
            uv_mode_cfl_not_allowed: T::DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF,
            angle_delta: T::DEFAULT_ANGLE_DELTA_CDF,
            filter_intra: T::DEFAULT_FILTER_INTRA_CDF,
            filter_intra_mode: T::DEFAULT_FILTER_INTRA_MODE_CDF,
            skip: T::DEFAULT_SKIP_CDF,
            skip_mode: T::DEFAULT_SKIP_MODE_CDF,
            segment_id: T::DEFAULT_SEGMENT_ID_CDF,
            segment_id_predicted: T::DEFAULT_SEGMENT_ID_PREDICTED_CDF,
            delta_q: T::DEFAULT_DELTA_Q_CDF,
            delta_lf: T::DEFAULT_DELTA_LF_CDF,
            cfl_sign: T::DEFAULT_CFL_SIGN_CDF,
            cfl_alpha: T::DEFAULT_CFL_ALPHA_CDF,
            palette_y_mode: T::DEFAULT_PALETTE_Y_MODE_CDF,
            palette_uv_mode: T::DEFAULT_PALETTE_UV_MODE_CDF,
            palette_y_size: T::DEFAULT_PALETTE_Y_SIZE_CDF,
            palette_uv_size: T::DEFAULT_PALETTE_UV_SIZE_CDF,
            palette_color_y: [
                palette_color_cdf(T::DEFAULT_PALETTE_SIZE_2_Y_COLOR_CDF),
                palette_color_cdf(T::DEFAULT_PALETTE_SIZE_3_Y_COLOR_CDF),
                palette_color_cdf(T::DEFAULT_PALETTE_SIZE_4_Y_COLOR_CDF),
                palette_color_cdf(T::DEFAULT_PALETTE_SIZE_5_Y_COLOR_CDF),
                palette_color_cdf(T::DEFAULT_PALETTE_SIZE_6_Y_COLOR_CDF),
                palette_color_cdf(T::DEFAULT_PALETTE_SIZE_7_Y_COLOR_CDF),
                palette_color_cdf(T::DEFAULT_PALETTE_SIZE_8_Y_COLOR_CDF),
            ],
            palette_color_uv: [
                palette_color_cdf(T::DEFAULT_PALETTE_SIZE_2_UV_COLOR_CDF),
                palette_color_cdf(T::DEFAULT_PALETTE_SIZE_3_UV_COLOR_CDF),
                palette_color_cdf(T::DEFAULT_PALETTE_SIZE_4_UV_COLOR_CDF),
                palette_color_cdf(T::DEFAULT_PALETTE_SIZE_5_UV_COLOR_CDF),
                palette_color_cdf(T::DEFAULT_PALETTE_SIZE_6_UV_COLOR_CDF),
                palette_color_cdf(T::DEFAULT_PALETTE_SIZE_7_UV_COLOR_CDF),
                palette_color_cdf(T::DEFAULT_PALETTE_SIZE_8_UV_COLOR_CDF),
            ],
            tx8x8: T::DEFAULT_TX_8X8_CDF,
            tx16x16: T::DEFAULT_TX_16X16_CDF,
            tx32x32: T::DEFAULT_TX_32X32_CDF,
            tx64x64: T::DEFAULT_TX_64X64_CDF,
            txfm_split: T::DEFAULT_TXFM_SPLIT_CDF,
            is_inter: T::DEFAULT_IS_INTER_CDF,
            comp_mode: T::DEFAULT_COMP_MODE_CDF,
            comp_ref: T::DEFAULT_COMP_REF_CDF,
            comp_bwdref: T::DEFAULT_COMP_BWD_REF_CDF,
            single_ref: T::DEFAULT_SINGLE_REF_CDF,
            comp_ref_type: T::DEFAULT_COMP_REF_TYPE_CDF,
            uni_comp_ref: T::DEFAULT_UNI_COMP_REF_CDF,
            compound_mode: T::DEFAULT_COMPOUND_MODE_CDF,
            new_mv: T::DEFAULT_NEW_MV_CDF,
            zero_mv: T::DEFAULT_ZERO_MV_CDF,
            ref_mv: T::DEFAULT_REF_MV_CDF,
            drl_mode: T::DEFAULT_DRL_MODE_CDF,
            interintra: T::DEFAULT_INTER_INTRA_CDF,
            interintra_mode: T::DEFAULT_INTER_INTRA_MODE_CDF,
            wedge_interintra: T::DEFAULT_WEDGE_INTER_INTRA_CDF,
            wedge_index: T::DEFAULT_WEDGE_INDEX_CDF,
            use_obmc: T::DEFAULT_USE_OBMC_CDF,
            motion_mode: T::DEFAULT_MOTION_MODE_CDF,
            compound_type: T::DEFAULT_COMPOUND_TYPE_CDF,
            comp_group_idx: T::DEFAULT_COMP_GROUP_IDX_CDF,
            compound_idx: T::DEFAULT_COMPOUND_IDX_CDF,
            interp_filter: T::DEFAULT_INTERP_FILTER_CDF,
            mv_joint: [T::DEFAULT_MV_JOINT_CDF; MV_CONTEXTS],
            mv_class: [T::DEFAULT_MV_CLASS_CDF; MV_CONTEXTS],
            mv_class0_bit: [[T::DEFAULT_MV_CLASS0_BIT_CDF; 2]; MV_CONTEXTS],
            mv_class0_fr: [T::DEFAULT_MV_CLASS0_FR_CDF; MV_CONTEXTS],
            mv_class0_hp: [[T::DEFAULT_MV_CLASS0_HP_CDF; 2]; MV_CONTEXTS],
            mv_fr: [T::DEFAULT_MV_FR_CDF; MV_CONTEXTS],
            mv_hp: [[T::DEFAULT_MV_HP_CDF; 2]; MV_CONTEXTS],
            mv_bit: [[T::DEFAULT_MV_BIT_CDF; 2]; MV_CONTEXTS],
            mv_sign: [[T::DEFAULT_MV_SIGN_CDF; 2]; MV_CONTEXTS],
        }
    }
}

// ── 帧级 4×4 网格 ──────────────────────────────────────────────────────

/// 块级语法要维护的一整张**帧级 4×4 网格**。
///
/// AV1 的块级上下文是**按 4×4 位置**取的（一个块占很多格，邻居查的是"紧邻这一格"），
/// 所以规范里 `decode_block` 用 `bw4 × bh4` 的双重循环把值铺到每一格上。
#[derive(Debug, Clone)]
pub struct BlockGrid {
    rows: usize,
    cols: usize,
    pub y_modes: Vec<u8>,
    pub uv_modes: Vec<u8>,
    pub skips: Vec<u8>,
    pub skip_modes: Vec<u8>,
    pub is_inters: Vec<u8>,
    pub ref_frames: Vec<[i8; 2]>,
    pub mvs: Vec<[Mv; 2]>,
    pub comp_group_idxs: Vec<u8>,
    /// `CompoundIdx`（**语法元素**，只有 0/1）——`compound_idx_ctx()` 的上下文要用它。
    pub compound_idxs: Vec<u8>,
    /// `CompoundType`（`COMPOUND_WEDGE`/`DIFFWTD`/`AVERAGE`/`INTRA`/`DISTANCE`）。
    ///
    /// **与 `compound_idxs` 是两回事**：`compound_idx` 只是 `comp_group_idx == 0` 那一支
    /// 用来在 AVERAGE / DISTANCE 之间二选一的语法元素（0/1），而 `compound_type` 是
    /// 最终混合方式。预测阶段要的是后者。
    pub compound_types: Vec<u8>,
    /// `WedgeIndex`（`COMPOUND_WEDGE` 的掩码编号）
    pub wedge_idxs: Vec<u8>,
    /// `WedgeSign`（0 / 1）
    pub wedge_signs: Vec<u8>,
    /// `MaskType`（`COMPOUND_DIFFWTD` 的掩码方向）
    pub mask_types: Vec<u8>,
    pub interp_filters: Vec<[u8; 2]>,
    /// `InterTxSizes`
    pub inter_tx_sizes: Vec<u8>,
    /// `TxTypes`（亮度 4×4 坐标）
    pub tx_types: TxTypeGrid,
    /// `LoopfilterTxSizes[plane]`（去块滤波用；`transform_block` 结尾写）——
    /// 三个平面都按**亮度 4×4 坐标**开表，色度侧用 `row >> sub_y` 取。
    pub lf_tx_sizes: [Vec<u8>; 3],
    /// `DeltaLFs[row][col][k]`（去块强度用；`read_delta_lf` 的累积值按块铺开）。
    pub delta_lfs: Vec<[i32; FRAME_LF_COUNT]>,
    pub seg_ids: Vec<i8>,
    pub prev_seg_ids: Vec<i8>,
    pub palette_sizes: Vec<[u8; 2]>,
    /// `PaletteColors[ plane ][ i ][ j ][ k ]`（plane 只有 0/1 两路：Y 与 **U**）。
    ///
    /// 规范里 UV 的调色板缓存只用 U 分量（`get_palette_cache(1)` 读的就是它），
    /// V 分量单独存在 `BlockCtx` 里、只供预测查表，所以这里第二维只有 2。
    pub palette_colors: Vec<[[u16; PALETTE_COLORS]; 2]>,
    pub cdef_idx: Vec<i32>,
    /// `InterIntraMode`（`II_*`）：`interintra` 块的帧内那一半用哪种预测。
    pub interintra_modes: Vec<u8>,
    above_seg_pred_ctx: Vec<u8>,
    left_seg_pred_ctx: Vec<u8>,
    /// "这个位置的 `RefFrames` 本帧已经写过"
    written: Vec<bool>,
}

/// `clear()` + `resize(n, val)`：**复用已有容量**地重填。
///
/// 语义与 `vec![val; n]` 完全一致（都写满 `n` 个格点），差别只在容量够时
/// 不再走分配器 —— 见 [`BlockGrid::reset_all`] 里那笔账。
#[inline]
fn fill_set<T: Clone>(v: &mut Vec<T>, n: usize, val: T) {
    v.clear();
    v.resize(n, val);
}

/// 只在**长度变化**时分配并填一次初值（跨帧复用时一个格子都不写）。
///
/// 配 [`BlockGrid::reset_all`] 里"本帧必被写满、初值是纯白写"的那批数组用。
/// 首帧/尺寸变化时照样要分配 + 填一次 —— 否则读越界。
#[inline]
fn ensure_len<T: Clone>(v: &mut Vec<T>, n: usize, val: T) {
    if v.len() != n {
        v.clear();
        v.resize(n, val);
    }
}

impl BlockGrid {
    /// 空壳（所有 `Vec` 都没有容量）。配 [`BlockGrid::reset_all`] 用。
    fn empty() -> Self {
        Self {
            rows: 0,
            cols: 0,
            y_modes: Vec::new(),
            uv_modes: Vec::new(),
            skips: Vec::new(),
            skip_modes: Vec::new(),
            is_inters: Vec::new(),
            ref_frames: Vec::new(),
            mvs: Vec::new(),
            comp_group_idxs: Vec::new(),
            compound_idxs: Vec::new(),
            compound_types: Vec::new(),
            wedge_idxs: Vec::new(),
            wedge_signs: Vec::new(),
            mask_types: Vec::new(),
            interp_filters: Vec::new(),
            inter_tx_sizes: Vec::new(),
            tx_types: TxTypeGrid::empty(),
            lf_tx_sizes: [Vec::new(), Vec::new(), Vec::new()],
            delta_lfs: Vec::new(),
            seg_ids: Vec::new(),
            prev_seg_ids: Vec::new(),
            palette_sizes: Vec::new(),
            palette_colors: Vec::new(),
            cdef_idx: Vec::new(),
            interintra_modes: Vec::new(),
            above_seg_pred_ctx: Vec::new(),
            left_seg_pred_ctx: Vec::new(),
            written: Vec::new(),
        }
    }

    pub fn new(rows: usize, cols: usize) -> Self {
        let mut g = Self::empty();
        g.reset_all(rows, cols);
        g
    }

    /// 把整张网格按 `rows × cols` 重新初始化（**跨帧复用同一张网格时用**）。
    ///
    /// ⚠️ 这里是**真的重填**，一个格点都没省 —— 语义与 `new()` 完全一致，
    /// 省掉的只是"重新分配 ~11 MB + 每页首次写入的缺页中断 + 释放时的 `madvise`/
    /// `vm_deallocate`"（1080p 实测 `BlockGrid::new` + `MiGrid::new` 合计
    /// **3.07 ms/帧**，其中绝大部分是这笔分配账；见 `memory/2026-10-04`）。
    /// 想再往下省就得论证"某些数组本帧必被写满"，那是另一件事。
    /// 每帧重填（`av1.rs::new_pending` 每帧调一次，尺寸变了也由它调长度）。
    ///
    /// ## 哪些数组**不必**初值（2026-10-04，逐数组论证）
    ///
    /// 1080p 一帧这些数组合起来 ≈ **12 MB** 的写；其中下面这批是**每个 4×4 格
    /// 都会被本帧的块写到**的（`decode_block` 的两个收尾循环都是**无条件**的，
    /// 见那两处 `stamp`；块铺满整张网格 ⇒ 任一跨块读（上/左邻居、OBMC 候选、
    /// 调色板缓存……）落到的格子都已在本帧更早的 `decode_block` 里被盖过），
    /// 所以初值是纯白写 —— 只留 [`ensure_len`]（省 ≈ **6.1 MB/帧**）：
    ///
    /// * 第一段收尾循环（`residual()` 之前）：`y_modes`、`ref_frames`；
    /// * 第二段收尾循环：`skips`、`skip_modes`、`is_inters`、`inter_tx_sizes`、
    ///   `delta_lfs`、`palette_sizes`、`palette_colors`。
    ///
    /// ⚠️ **必须保留初值**的（读它们的地方不受"本格必被写"保护）：
    /// * `uv_modes` —— 子 8×8 的色度归"右下角那个 4×4"预测/铺格子，
    ///   其余三个 4×4 的格子可能一直没被写过，而 `get_uv_mode` 会读它们；
    /// * `mvs` / `comp_*` / `wedge_*` / `mask_types` / `interp_filters` /
    ///   `interintra_modes` —— 只在 `is_inter` 时写；读它们的地方（OBMC 候选、
    ///   复合上下文）虽然都有 `ref_frames` 门（见 `obmc_blend`：候选必须
    ///   `ref_frames[gi][0] > INTRA_FRAME` 才用），但"门与写同源"这条依赖链
    ///   太隐晦，留着初值（这批加起来才 ~2 MB 的写）；
    /// * `tx_types` / `lf_tx_sizes` / `cdef_idx` —— 分别按变换块/超块写，
    ///   且被帧级滤波与 CDEF 在**整帧解完之后**读；
    /// * `seg_ids` / `prev_seg_ids` —— `latch_prev_segment_ids` 在 `reset_all`
    ///   **之后**跑，`prev_seg_ids` 目前恒为全 0（等价 libaom 的
    ///   `last_frame_seg_map = NULL` 分支）。这是**语义**问题不是性能问题：
    ///   dav1d 取的是主参考槽存下来的 segmap（`c->refs[pri_ref].segmap`），
    ///   要改得先有"按参考槽存 segmap"这套设施，别在这里顺手改。
    ///   208 KB/帧，不值当省。
    pub fn reset_all(&mut self, rows: usize, cols: usize) {
        self.rows = rows;
        self.cols = cols;
        let n = rows * cols;
        ensure_len(&mut self.y_modes, n, DC_PRED);
        fill_set(&mut self.uv_modes, n, DC_PRED);
        ensure_len(&mut self.skips, n, 0);
        ensure_len(&mut self.skip_modes, n, 0);
        ensure_len(&mut self.is_inters, n, 0);
        ensure_len(&mut self.ref_frames, n, [mv::INTRA_FRAME, NONE]);
        fill_set(&mut self.mvs, n, [[0, 0]; 2]);
        fill_set(&mut self.comp_group_idxs, n, 0);
        fill_set(&mut self.compound_idxs, n, 1);
        fill_set(&mut self.compound_types, n, COMPOUND_AVERAGE);
        fill_set(&mut self.wedge_idxs, n, 0);
        fill_set(&mut self.wedge_signs, n, 0);
        fill_set(&mut self.mask_types, n, 0);
        fill_set(&mut self.interp_filters, n, [EIGHTTAP; 2]);
        ensure_len(&mut self.inter_tx_sizes, n, TX_4X4);
        self.tx_types.reset_all(rows, cols);
        for t in self.lf_tx_sizes.iter_mut() {
            fill_set(t, n, TX_4X4);
        }
        ensure_len(&mut self.delta_lfs, n, [0; FRAME_LF_COUNT]);
        fill_set(&mut self.seg_ids, n, 0);
        fill_set(&mut self.prev_seg_ids, n, 0);
        ensure_len(&mut self.palette_sizes, n, [0, 0]);
        ensure_len(&mut self.palette_colors, n, [[0u16; PALETTE_COLORS]; 2]);
        fill_set(&mut self.cdef_idx, n, -1);
        fill_set(&mut self.interintra_modes, n, II_DC_PRED);
        fill_set(&mut self.above_seg_pred_ctx, cols, 0);
        fill_set(&mut self.left_seg_pred_ctx, rows, 0);
        fill_set(&mut self.written, n, false);
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// 4×4 坐标 → 下标。越界返回 0（**所有调用方都先做过范围判断**）。
    #[inline]
    pub fn idx(&self, r: i32, c: i32) -> usize {
        if r < 0 || c < 0 || r as usize >= self.rows || c as usize >= self.cols {
            return 0;
        }
        r as usize * self.cols + c as usize
    }

    /// 每帧开头重置（`cdef_idx` 回 -1、分割预测上下文与"已写过"标记清零）。
    pub fn reset_frame(&mut self) {
        self.cdef_idx.fill(-1);
        for t in self.lf_tx_sizes.iter_mut() {
            t.fill(TX_4X4);
        }
        self.above_seg_pred_ctx.fill(0);
        self.left_seg_pred_ctx.fill(0);
        self.written.fill(false);
    }

    /// 把上一帧的 `SegmentIds` 收成 `PrevSegmentIds`。
    pub fn latch_prev_segment_ids(&mut self) {
        self.prev_seg_ids.copy_from_slice(&self.seg_ids);
    }

    #[inline]
    pub fn is_inter_written(&self, r: i32, c: i32) -> bool {
        if r < 0 || c < 0 || r as usize >= self.rows || c as usize >= self.cols {
            return false;
        }
        self.written[r as usize * self.cols + c as usize]
    }

    #[inline]
    /// `clear_cdef( r, c )`（规范 §5.11.1）：把一个超块覆盖的 `cdef_idx` 清回 -1。
    pub fn clear_cdef(&mut self, r: i32, c: i32, sb4: i32) {
        let rows = self.rows;
        let cols = self.cols;
        let mut i = r;
        while i < r + sb4 {
            let mut j = c;
            while j < c + sb4 {
                if i >= 0 && j >= 0 && (i as usize) < rows && (j as usize) < cols {
                    self.cdef_idx[i as usize * cols + j as usize] = -1;
                }
                j += 1;
            }
            i += 1;
        }
    }

    pub fn inter_tx_size(&self, r: i32, c: i32) -> u8 {
        if r < 0 || c < 0 || r as usize >= self.rows || c as usize >= self.cols {
            return TX_64X64;
        }
        self.inter_tx_sizes[r as usize * self.cols + c as usize]
    }

    #[inline]
    pub fn tx_type(&self, r: i32, c: i32) -> u8 {
        self.tx_types.get(r, c)
    }

    #[inline]
    pub fn set_tx_type(&mut self, r: i32, c: i32, v: u8) {
        self.tx_types.set(r, c, v)
    }

    /// `seg_id_predicted` 的上下文：`LeftSegPredContext[MiRow] + AboveSegPredContext[MiCol]`
    pub fn seg_pred_ctx(&self, r: i32, c: i32) -> usize {
        let l = if r >= 0 && (r as usize) < self.rows { self.left_seg_pred_ctx[r as usize] } else { 0 };
        let a = if c >= 0 && (c as usize) < self.cols { self.above_seg_pred_ctx[c as usize] } else { 0 };
        (l + a).min(2) as usize
    }

    /// 把 `seg_id_predicted` 铺到本块覆盖的行/列上
    pub fn set_seg_pred_ctx(&mut self, r: i32, c: i32, bw4: i32, bh4: i32, v: u8) {
        for i in 0..bw4 {
            let col = c + i;
            if col >= 0 && (col as usize) < self.cols {
                self.above_seg_pred_ctx[col as usize] = v;
            }
        }
        for i in 0..bh4 {
            let row = r + i;
            if row >= 0 && (row as usize) < self.rows {
                self.left_seg_pred_ctx[row as usize] = v;
            }
        }
    }

    /// 一个块覆盖到的每一格都写同样的值。
    fn stamp<F: FnMut(&mut Self, usize)>(&mut self, r: i32, c: i32, bsize: u8, mut f: F) {
        if bsize as usize >= BLOCK_SIZES {
            return;
        }
        let w = NUM_4X4_BLOCKS_WIDE[bsize as usize] as i32;
        let h = NUM_4X4_BLOCKS_HIGH[bsize as usize] as i32;
        for y in 0..h {
            for x in 0..w {
                let (rr, cc) = (r + y, c + x);
                if rr < 0 || cc < 0 || rr as usize >= self.rows || cc as usize >= self.cols {
                    continue;
                }
                let i = rr as usize * self.cols + cc as usize;
                f(self, i);
            }
        }
    }
}

// ── 解码结果与统计 ──────────────────────────────────────────────────────

/// 一个块的 `mode_info` 结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockInfo {
    pub mi_row: u32,
    pub mi_col: u32,
    pub mi_size: u8,
    pub skip: bool,
    pub skip_mode: bool,
    pub is_inter: bool,
    pub segment_id: u8,
    pub y_mode: u8,
    pub uv_mode: u8,
    pub angle_delta_y: i8,
    pub angle_delta_uv: i8,
    pub use_filter_intra: bool,
    pub filter_intra_mode: u8,
    pub palette_size_y: u8,
    pub palette_size_uv: u8,
    pub tx_size: u8,
    pub use_intrabc: bool,
    pub cfl_alpha_u: i8,
    pub cfl_alpha_v: i8,
    pub ref_frame: [i8; 2],
    pub mv: [Mv; 2],
    pub interp_filter: [u8; 2],
    pub motion_mode: u8,
    pub compound_type: u8,
    pub comp_group_idx: u8,
    pub compound_idx: u8,
    /// `wedge_index`（`COMPOUND_WEDGE` 才有；`read_compound_type` 里的 `S()` 符号）
    pub wedge_index: u8,
    /// `wedge_sign`（`L(1)`）
    pub wedge_sign: u8,
    /// `mask_type`（`COMPOUND_DIFFWTD` 才有；`L(1)`）
    pub mask_type: u8,
    pub interintra: bool,
    pub wedge_interintra: bool,
    /// `interintra_mode`（`II_*`；像素侧要映射成帧内预测模式）
    pub interintra_mode: u8,
}

impl Default for BlockInfo {
    fn default() -> Self {
        Self {
            mi_row: 0,
            mi_col: 0,
            mi_size: BLOCK_4X4 as u8,
            skip: false,
            skip_mode: false,
            is_inter: false,
            segment_id: 0,
            y_mode: DC_PRED,
            uv_mode: DC_PRED,
            angle_delta_y: 0,
            angle_delta_uv: 0,
            use_filter_intra: false,
            filter_intra_mode: 0,
            palette_size_y: 0,
            palette_size_uv: 0,
            tx_size: TX_4X4,
            use_intrabc: false,
            cfl_alpha_u: 0,
            cfl_alpha_v: 0,
            ref_frame: [mv::INTRA_FRAME, NONE],
            mv: [[0, 0]; 2],
            interp_filter: [EIGHTTAP; 2],
            motion_mode: SIMPLE,
            compound_type: COMPOUND_AVERAGE,
            comp_group_idx: 0,
            compound_idx: 1,
            wedge_index: 0,
            wedge_sign: 0,
            mask_type: 0,
            interintra: false,
            wedge_interintra: false,
            interintra_mode: II_DC_PRED,
        }
    }
}

/// 块级语法 + 系数的统计（诊断与验收）。
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct BlockStats {
    pub blocks: u64,
    pub symbols: u64,
    pub skip_blocks: u64,
    pub skip_mode_blocks: u64,
    pub inter_blocks: u64,
    pub compound_blocks: u64,
    pub cfl_blocks: u64,
    pub filter_intra_blocks: u64,
    /// 用了 Y / UV 调色板的块数（诊断：屏幕内容流是否真的走到 palette 分支）
    pub palette_y_blocks: u64,
    pub palette_uv_blocks: u64,
    /// `use_intrabc == 1` 的块数（诊断：屏幕内容流是否真的走到 intrabc 分支）
    pub intrabc_blocks: u64,
    /// `HasChroma == 0` 的块数（诊断：色度是否被正确跳过）
    pub no_chroma_blocks: u64,
    pub y_mode_histogram: [u64; INTRA_MODES as usize],
    pub uv_mode_histogram: [u64; 14],
    pub tx_size_histogram: [u64; TX_SIZES_ALL],
    pub segment_histogram: [u64; MAX_SEGMENTS],
    pub ref_frame_histogram: [u64; 8],
    pub motion_mode_histogram: [u64; 3],
    pub compound_type_histogram: [u64; 5],
    pub delta_q_nonzero: u64,
    /// 这些块一共吃掉了多少位（诊断）
    pub bits: u64,
    /// 其中 `mode_info` + `read_block_tx_size` 吃掉的位数（诊断）
    pub mode_bits: u64,
    pub coeff: CoeffStats,
}


/// 一趟瓦片的完整解码结果。
#[derive(Debug, Clone)]
pub struct TileDecodeResult {
    pub partition: partition::PartitionStats,
    pub blocks: BlockStats,
    /// `exit_symbol()` 通过了吗（瓦片末尾的 trailing bits）
    pub exit_ok: bool,
    /// 遍历完语法后算术流还剩多少真实位（诊断用）
    pub bits_left: i64,
    /// 覆盖检查的问题数（0 = 每个 4×4 位置都被覆盖）
    pub uncovered: usize,
    /// 重复写入的位置数（诊断：`uncovered == 0` 也可能有重叠）
    pub overlaps: usize,
    /// **每个超块**吃掉的位数（诊断）。算术解码的 `SymbolMaxBits` 不钳位，
    /// 差分就是该超块的真实消耗 —— 用来定位"多读"是均匀分布还是集中在某处。
    pub sb_bits: Vec<u32>,
    /// 每个超块里的块（分区叶子）数
    pub sb_leaves: Vec<u32>,
    /// 分区符号吃掉的真实位数（= 总消耗 - 块级消耗；诊断"多读在哪一层"）
    pub partition_bits: i64,
}

// ── 纯函数小工具 ────────────────────────────────────────────────────────

/// trace 开关（`AV1_TRACE_MODE`）。
/// 当前帧号（`AV1_TRACE_MODE` 的打印用；由驱动在每帧开头设置）
pub static TRACE_FRAME_NO: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 诊断（`AV1_PHASE_TIMING=1`）：`replay_one` 的三段累计纳秒（MC / 残差 / 簿记）。
/// **临时设施**，定位"重建为什么慢"用，定位完删除。
pub(crate) static DBG_MC_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static DBG_TXB_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static DBG_MARK_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 取走上一次重建的三段毫秒数并清零（诊断用）。
pub(crate) fn take_replay_dbg() -> (f64, f64, f64) {
    use std::sync::atomic::Ordering;
    (
        DBG_MC_NS.swap(0, Ordering::Relaxed) as f64 / 1e6,
        DBG_TXB_NS.swap(0, Ordering::Relaxed) as f64 / 1e6,
        DBG_MARK_NS.swap(0, Ordering::Relaxed) as f64 / 1e6,
    )
}

/// 当前"**带瓦片的帧**"序号（由驱动在每帧解瓦片**之前**设置）。
///
/// 计数口径与 `frames_parsed` 不同：`show_existing_frame` 不重建像素、不产生
/// 瓦片，所以不算在内 —— 这正是 libaom 插桩 `MARK tile frame=` 行的计数口径，
/// 两边对拍时按它对齐。
pub static TRACE_TILE_FRAME_NO: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 符号 trace 的**帧门控**（`AV1_TRACE_SYM_FRAME=<n>`）：只对第 `n` 个带瓦片的
/// 帧打开符号 trace。未设置时不门控（保持原来的"全帧都打"行为）。
///
/// 为什么需要它：真素材一帧有几千个符号，从第 0 帧追到出问题的那一帧会到
/// GB 级；门控后输出只剩目标帧，再和 libaom 的同帧逐符号对拍。
pub fn trace_sym_frame() -> Option<u64> {
    static F: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("AV1_TRACE_SYM_FRAME").ok().and_then(|v| v.parse().ok())
    })
}

/// 当前是否在**门控的目标帧**里（门控未设置时恒为真）。
pub fn in_traced_frame() -> bool {
    match trace_sym_frame() {
        None => true,
        Some(f) => TRACE_TILE_FRAME_NO.load(std::sync::atomic::Ordering::Relaxed) == f,
    }
}

fn trace_mode() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("AV1_TRACE_MODE").is_ok())
}

/// 诊断开关（`AV1_TRACE_LASTSB`）：逐块打"贴着帧边界"的块。
/// **必须缓存**：块级热路径上每次 `std::env::var` 都要分配一个 String。
fn trace_last_sb() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("AV1_TRACE_LASTSB").is_ok())
}

/// 诊断开关（`AV1_TRACE_SB`）：逐超块打位/叶子数。
fn trace_sb() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("AV1_TRACE_SB").is_ok())
}

/// 诊断开关（`AV1_TRACE_BLKATTR`）：**逐块打属性**（位置 / 尺寸 / 帧内帧间 / 参考帧 /
/// MV / 插值滤波 / 变换尺寸 / 运动模式 / 复合类型 / interintra）。
///
/// 为什么需要它：语法层已经与 libaom **逐符号一致** ⇒ **块边界一一对应**，
/// 所以两边的第 n 个块是同一个块。于是"像素差在哪"这个问题可以化成
/// "**第一个属性对不上的块是哪个、哪个属性**" —— 逐行 diff 即可。
/// libaom 那边对应 `LERXU_BLKATTR`（`av1/decoder/decodeframe.c` 的 `decode_block`）。
fn trace_blk_attr() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("AV1_TRACE_BLKATTR").is_ok())
}

/// `AV1_TRACE_BLKATTR` 的块序号（**不按帧重置**：两边从同一帧开跑，序号自然对齐）。
static BLK_ATTR_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `neg_deinterleave( diff, ref, max )`（规范 §5.11.9）
pub fn neg_deinterleave(diff: i32, reference: i32, max: i32) -> i32 {
    if reference == 0 {
        return diff;
    }
    if reference >= max - 1 {
        return max - diff - 1;
    }
    if 2 * reference < max {
        if diff <= 2 * reference {
            if diff & 1 == 1 {
                return reference + ((diff + 1) >> 1);
            }
            return reference - (diff >> 1);
        }
        return diff;
    }
    if diff <= 2 * (max - reference - 1) {
        if diff & 1 == 1 {
            return reference + ((diff + 1) >> 1);
        }
        return reference - (diff >> 1);
    }
    max - (diff + 1)
}

/// `LosslessArray[segmentId]`（规范里在帧初始化时算）
pub fn lossless_array(fh: &FrameHeader) -> [bool; MAX_SEGMENTS] {
    let q = &fh.params.quant;
    let seg = &fh.params.segmentation;
    let mut out = [false; MAX_SEGMENTS];
    let all_deltas_zero = q.delta_q_y_dc == 0
        && q.delta_q_u_ac == 0
        && q.delta_q_u_dc == 0
        && q.delta_q_v_ac == 0
        && q.delta_q_v_dc == 0;
    for (s, slot) in out.iter_mut().enumerate() {
        let alt_q = if seg.feature_active(s, SEG_LVL_ALT_Q) {
            seg.feature_data[s][SEG_LVL_ALT_Q]
        } else {
            0
        };
        let qindex = (q.base_q_idx as i32 + alt_q).clamp(0, 255);
        *slot = qindex == 0 && all_deltas_zero;
    }
    out
}

/// `ref_count_ctx( counts0, counts1 )`
#[inline]
pub fn ref_count_ctx(counts0: i32, counts1: i32) -> usize {
    if counts0 < counts1 {
        0
    } else if counts0 == counts1 {
        1
    } else {
        2
    }
}

/// `comp_mode` 的上下文（规范 §8.3.2）
#[allow(clippy::too_many_arguments)]
pub fn comp_mode_ctx(
    avail_u: bool,
    avail_l: bool,
    above0: i8,
    left0: i8,
    above_intra: bool,
    left_intra: bool,
    above_single: bool,
    left_single: bool,
) -> usize {
    let check_backward = |r: i8| (mv::BWDREF_FRAME..=mv::ALTREF_FRAME).contains(&r);
    if avail_u && avail_l {
        if above_single && left_single {
            usize::from(check_backward(above0) ^ check_backward(left0))
        } else if above_single {
            2 + usize::from(check_backward(above0) || above_intra)
        } else if left_single {
            2 + usize::from(check_backward(left0) || left_intra)
        } else {
            4
        }
    } else if avail_u {
        if above_single {
            usize::from(check_backward(above0))
        } else {
            3
        }
    } else if avail_l {
        if left_single {
            usize::from(check_backward(left0))
        } else {
            3
        }
    } else {
        1
    }
}

/// `is_samedir_ref_pair( ref0, ref1 )`
#[inline]
pub fn is_samedir_ref_pair(ref0: i8, ref1: i8) -> bool {
    (ref0 >= mv::BWDREF_FRAME) == (ref1 >= mv::BWDREF_FRAME)
}

/// `comp_ref_type` 的上下文（规范 §8.3.2）
#[allow(clippy::too_many_arguments)]
pub fn comp_ref_type_ctx(
    avail_u: bool,
    avail_l: bool,
    above0: i8,
    above1: i8,
    left0: i8,
    left1: i8,
    above_intra: bool,
    left_intra: bool,
    above_single: bool,
    left_single: bool,
) -> usize {
    let above_comp_inter = avail_u && !above_intra && !above_single;
    let left_comp_inter = avail_l && !left_intra && !left_single;
    let above_uni_comp = above_comp_inter && is_samedir_ref_pair(above0, above1);
    let left_uni_comp = left_comp_inter && is_samedir_ref_pair(left0, left1);
    if avail_u && !above_intra && avail_l && !left_intra {
        let samedir = is_samedir_ref_pair(above0, left0);
        if !above_comp_inter && !left_comp_inter {
            1 + 2 * usize::from(samedir)
        } else if !above_comp_inter {
            if !left_uni_comp {
                1
            } else {
                3 + usize::from(samedir)
            }
        } else if !left_comp_inter {
            if !above_uni_comp {
                1
            } else {
                3 + usize::from(samedir)
            }
        } else if !above_uni_comp && !left_uni_comp {
            0
        } else if !above_uni_comp || !left_uni_comp {
            2
        } else {
            3 + usize::from((above0 == mv::BWDREF_FRAME) == (left0 == mv::BWDREF_FRAME))
        }
    } else if avail_u && avail_l {
        if above_comp_inter {
            1 + 2 * usize::from(above_uni_comp)
        } else if left_comp_inter {
            1 + 2 * usize::from(left_uni_comp)
        } else {
            2
        }
    } else if above_comp_inter {
        4 * usize::from(above_uni_comp)
    } else if left_comp_inter {
        4 * usize::from(left_uni_comp)
    } else {
        2
    }
}

/// `needs_interp_filter( )`（规范 §5.11.14）
pub fn needs_interp_filter(
    fh: &FrameHeader,
    bsize: u8,
    skip_mode: bool,
    motion_mode: u8,
    ymode: u8,
    ref_frame: [i8; 2],
    large: bool,
) -> bool {
    if skip_mode || motion_mode == LOCALWARP {
        return false;
    }
    if large && ymode == GLOBALMV {
        return fh.params.global_motion.gm_type[ref_frame[0] as usize] == super::params::TRANSLATION;
    }
    if large && ymode == mv::GLOBAL_GLOBALMV {
        return fh.params.global_motion.gm_type[ref_frame[0] as usize]
            == super::params::TRANSLATION
            || fh.params.global_motion.gm_type[ref_frame[1] as usize] == super::params::TRANSLATION;
    }
    let _ = bsize;
    true
}

/// 读一个带 CDF 的符号（顺手记统计）。
#[inline]
/// 诊断（`AV1_STOP_SB=<n>`）：只解前 `n` 个超块就停。
///
/// 用来**二分定位**"这个瓦片从第几行超块开始跟参考对不上" —— 配合
/// `decode --count=<k>` 输出部分重建的帧，再与 dav1d 的同序号帧逐行比。
fn stop_sb_limit() -> Option<usize> {
    static ON: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AV1_STOP_SB").ok().and_then(|v| v.parse().ok()))
}

/// 诊断（`AV1_TRACE_INTER=1`）：帧间帧**前 40 个块**的分段位消耗。
///
/// 段序 = `[segPre, skipMode, skip, segPost, cdef, deltaQ, deltaLF, isInter, blockMode]`。
/// `blockMode` 里含 `ref_frame` / `inter_mode` / `DRL` / **运动矢量** / 运动模式，
/// 是"每块多读"最可能藏身的地方；这一列一眼就能看出多读出在哪一段。
fn trace_inter_mode(r: i32, c: i32, bsize: u8, info: &BlockInfo, seg: [i64; 9]) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static N: AtomicUsize = AtomicUsize::new(0);
    if !*ON.get_or_init(|| inter_trace_limit() > 0) {
        return;
    }
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n >= inter_trace_limit() {
        return;
    }
    eprintln!(
        "IMODE#{n} r={r} c={c} bs={bsize} skip={} skipm={} inter={} | segPre={} skipMode={} skipB={} segPost={} cdef={} dq={} dlf={} isInter={} blockMode={} | 合计 {}",
        info.skip as u8,
        info.skip_mode as u8,
        info.is_inter as u8,
        seg[0],
        seg[1],
        seg[2],
        seg[3],
        seg[4],
        seg[5],
        seg[6],
        seg[7],
        seg[8],
        seg.iter().sum::<i64>()
    );
}

/// `AV1_TRACE_INTER` 的打印条数上限：`=all` 时全打（用来统计整帧），否则前 40 个。
fn inter_trace_limit() -> usize {
    static L: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *L.get_or_init(|| match std::env::var("AV1_TRACE_INTER").ok().as_deref() {
        Some("all") => usize::MAX,
        Some(_) => 40,
        None => 0,
    })
}

/// 诊断（`AV1_TRACE_INTER=1|all`）：帧间块**内部 8 段**的位消耗。
///
/// 段序 = `[refFrames, yMode, drl, mv, interintra, motionMode, compoundType, interpFilter]`。
/// `mv` 那一列若异常大，说明 **运动矢量预测栈**（`find_mv_stack`）给错了预测值
/// —— 差分变大后 `uvlc` 会多读一大截位。
fn trace_inter_blk(r: i32, c: i32, bsize: u8, info: &BlockInfo, stack_num: usize, seg: [i64; 8]) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static N: AtomicUsize = AtomicUsize::new(0);
    if !*ON.get_or_init(|| inter_trace_limit() > 0) {
        return;
    }
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n >= inter_trace_limit() {
        return;
    }
    eprintln!(
        "IBLK#{n} r={r} c={c} bs={bsize} ymode={} ref1={} skip={} stack={} | refF={} yMode={} drl={} mv={} ii={} mm={} ct={} if={} | 合计 {}",
        info.y_mode,
        info.ref_frame[1] as i32,
        info.skip as u8,
        stack_num,
        seg[0],
        seg[1],
        seg[2],
        seg[3],
        seg[4],
        seg[5],
        seg[6],
        seg[7],
        seg.iter().sum::<i64>()
    );
}

#[track_caller]
fn read_sym(dec: &mut SymbolDecoder<'_>, stats: &mut BlockStats, cdf: &mut [u16]) -> Result<u32> {
    stats.symbols += 1;
    dec.read_symbol(cdf)
}

// ── 块级解码上下文 ──────────────────────────────────────────────────────

/// 块级解码的上下文。**一个瓦片一份**（CDF、`CurrentQIndex`、`DeltaLF` 都是瓦片局部的），
/// 帧级的网格由外部持有。
/// 语法阶段记录的一个变换块（重建阶段要用的一切）。
#[derive(Debug, Clone, Copy)]
pub(crate) struct TxbJob {
    pub plane: u8,
    pub start_x: i32,
    pub start_y: i32,
    pub tx_sz: u8,
    /// 规范 `transform_block` 的入参：txb 在块内的 4×4 偏移
    /// （帧内预测判 `haveLeft = AvailL || x > 0` 用）。
    pub x: i32,
    pub y: i32,
    /// 系数在 [`RowMtJobs::levels`] 里的起点与长度（系数格 `tw*th`）。
    pub levels_at: usize,
    pub levels_len: usize,
    /// `eob > 0` 才要加残差；帧内预测**无论如何**都要做。
    pub has_residual: bool,
    /// 两个 `BlockDecoded` 查询的**预计算值**（见 [`super::intra::DecFlags`]）；
    /// 回放前由 [`RowMtJobs::precompute_dec_flags`] 填。
    pub flags: super::intra::DecFlags,
}

/// 语法阶段记录的一个块（模式信息的快照 + 它的变换块区间）。
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockJob {
    pub info: BlockInfo,
    pub warp: super::inter::WarpPlan,
    pub intra_ctx: super::inter::IntraContext,
    /// 残差配置（`residual()` 建好后回填；重建阶段要用它做反量化/反变换）。
    pub cfg: Option<super::coeff::ResidualConfig>,
    /// `HasChroma`（`BlockDecoded` 标记要用；`replay` 时算不出来 ⇒ 记下来）
    pub has_chroma: bool,
    /// **本块要从上一行读到哪个 mi 列（不含）** —— row-mt 波前的等待上界。
    ///
    /// 0 = 本块不读上一行（纯帧间、没有 interintra ⇒ 只读参考帧）。
    /// 由 [`RowMtJobs::precompute_dec_flags`] 按**变换块**算出来（见 [`above_need_mi`]）：
    /// `have_above_right` 为真时上一行要读到 `x + 2w - 1`，是块宽的**两倍**。
    pub above_need: u32,
    pub txbs_at: usize,
    pub txbs_len: usize,
}

/// **row-mt 的工单 arena**（语法阶段写、重建阶段读；逐瓦片复位）。
///
/// `ME_ROW_MT=1` 时：`decode_block` 只读语法、把"重建要用的一切"记进来，
/// `decode_tile` 在超块循环结束后调 [`BlockCtx::replay_jobs`] 回放。
/// 判据是**回放出来的像素与现路径逐字节一致**（阶段 2 再把回放按 SB 行并行）。
#[derive(Default)]
pub struct RowMtJobs {
    pub(crate) blocks: Vec<BlockJob>,
    pub(crate) txbs: Vec<TxbJob>,
    pub(crate) levels: Vec<i32>,
    /// [`RowMtJobs::precompute_dec_flags`] 用的"解码顺序"网格（每平面一份，
    /// `mi_rows × mi_cols` 个 `u32`）。**只增不减地复用**（2026-10-04）：
    /// 1080p 是 480×270 ×4B = 518 KB/平面、三平面 **1.55 MB/帧**，原先每帧
    /// 新 `collect()` 一遍 —— 这么大的块会走 `mmap`，帧末 `free` 就归还，
    /// 下一帧再缺页重来（采样里 `bzero`/`memmove`/缺页合计约 11%）。
    pub(crate) dec_grids: Vec<Vec<u32>>,
    /// 语法阶段（`decode_block` 只记录、不写像素）。
    pub(crate) syntax_phase: bool,
    /// 当前块**就地重建**（调色板块：它要用"活的"调色板字段，而那是逐块覆盖的；
    /// 见 `decode_block` 里 flush 的说明）。
    pub(crate) inline_block: bool,
}

impl RowMtJobs {
    /// 开一个新块的记录，返回它的下标。
    pub(crate) fn start_block(
        &mut self,
        info: &BlockInfo,
        warp: super::inter::WarpPlan,
        intra_ctx: super::inter::IntraContext,
        has_chroma: bool,
    ) -> usize {
        let idx = self.blocks.len();
        self.blocks.push(BlockJob {
            info: *info,
            warp,
            intra_ctx,
            cfg: None,
            has_chroma,
            above_need: 0,
            txbs_at: self.txbs.len(),
            txbs_len: 0,
        });
        idx
    }

    /// 关掉当前块的记录（`txbs_len` 收口）。
    pub(crate) fn end_block(&mut self, idx: usize) {
        let at = self.blocks[idx].txbs_at;
        self.blocks[idx].txbs_len = self.txbs.len() - at;
    }

    /// 回放**之前**把每个单元的两个 `BlockDecoded` 查询算好（见 [`super::intra::DecFlags`]）。
    ///
    /// `BlockDecoded` 的语义是"那个 4×4 格子所属的单元在**解码顺序**里是否更早"。
    /// 解码顺序 = 工单记录序（划分树顺序，`VERT_4`/`HORZ_4` 下会跨行），这里就按记录序
    /// 给每个**标记事件**发一个递增下标：
    ///
    /// * 帧内块 = **每个变换块**一个事件（规范在 `transform_block` 结尾按 txb 标记）；
    /// * 帧间块 = 整块一个事件（规范的标记在 `decode_block` 收尾，见 `decode_block`
    ///   与 `replay_one` 里同一套坐标推导）。
    ///
    /// 下标铺进一份与 `CurrFrame::decoded` 同形的网格，查询就是
    /// "目标格子的下标 < 查询单元自己的下标"。网格初值取自 `frame.decoded` ——
    /// 那是**本批工单之前**（更早的 flush / 调色板块）写下的标记，用一个特殊小值
    /// 记下：对批内所有单元都可见；`0` 表示没标过，对谁也不可见。
    ///
    /// 有了它，回放（无论顺序还是并行）都不再依赖"谁先跑"，跨行的可见性也一致。
    pub(crate) fn precompute_dec_flags(&mut self, frame: &CurrFrame, seq: &SequenceHeader) {
        if flags::row_mt_live_flags() {
            // 调试用：不预计算，回放时照旧读运行时的 `BlockDecoded` 网格
            // （只有顺序回放才与记录序同序，此旋钮等价于改前的行为）。
            // 波前的等待上界还是要给（保守：整个块宽的两倍）——
            // 少了它并行回放会读到上一行还没写的像素。
            for b in self.blocks.iter_mut() {
                let info = b.info;
                let reads_above = !info.is_inter || (info.interintra && !flags::no_interintra());
                if reads_above {
                    let bw4 = (block_width(info.mi_size) >> 2).max(1);
                    b.above_need = (info.mi_col as i32 + 2 * bw4).max(0) as u32;
                }
            }
            return;
        }
        let (sx, sy) = (
            u8::from(seq.color.subsampling_x),
            u8::from(seq.color.subsampling_y),
        );
        let num_planes = if seq.color.mono_chrome { 1u8 } else { 3 };
        // 顺序网格：0 = 没标过；1 = 本批之前就标过；k ≥ 2 = 批内第 (k-2) 个标记事件。
        //
        // ⚠️ 网格**复用**（2026-10-04）：见 `RowMtJobs::dec_grids` 的说明 ——
        // 原先这里每帧都 `collect()` 出 1.55 MB 的新分配。
        if self.dec_grids.len() < num_planes as usize {
            self.dec_grids.resize_with(num_planes as usize, Vec::new);
        }
        for p in 0..num_planes as usize {
            let (d, _) = frame.decoded_grid(p as u8);
            let g = &mut self.dec_grids[p];
            g.clear();
            g.extend(d.iter().map(|&b| u32::from(b != 0)));
        }
        let cols: Vec<usize> = (0..num_planes).map(|p| frame.decoded_grid(p).1).collect();

        let grids = &mut self.dec_grids[..num_planes as usize];
        let blocks = &mut self.blocks;
        let txbs = &mut self.txbs;
        let mut event = 2u32;
        for i in 0..blocks.len() {
            let info = blocks[i].info;
            let has_chroma = blocks[i].has_chroma;
            let (r, c) = (info.mi_row as i32, info.mi_col as i32);
            let (w, h) = (block_width(info.mi_size), block_height(info.mi_size));
            if info.is_inter {
                let at = event;
                event += 1;
                for plane in 0..num_planes {
                    if plane > 0 && !has_chroma {
                        continue;
                    }
                    let (sub_x, sub_y) = (plane > 0 && sx != 0, plane > 0 && sy != 0);
                    let (pw, ph) = (if sub_x { w >> 1 } else { w }, if sub_y { h >> 1 } else { h });
                    let (r0, c0) = (if sub_y { r >> 1 } else { r }, if sub_x { c >> 1 } else { c });
                    order_stamp(
                        grids,
                        &cols,
                        plane,
                        r0 as i64,
                        c0 as i64,
                        (pw >> 2).max(1),
                        (ph >> 2).max(1),
                        at,
                    );
                }
                // `interintra` 的帧内那一半整块预测也要这两个查询（`interintra_predict_plane`）。
                // 查询坐标就是整块（该块只可能是 8×8 以上 ⇒ 没有"子 8×8 色度特例"，
                // 宽度就是 `block_width(bsize) >> sub`）。
                if info.interintra && !flags::no_interintra() {
                    let mut f = [super::intra::DecFlags::default(); 3];
                    for plane in 0..num_planes {
                        let (sub_x, sub_y) = (plane > 0 && sx != 0, plane > 0 && sy != 0);
                        let (pw, ph) =
                            (if sub_x { w >> 1 } else { w }, if sub_y { h >> 1 } else { h });
                        let (r0, c0) =
                            (if sub_y { r >> 1 } else { r }, if sub_x { c >> 1 } else { c });
                        f[plane as usize] = super::intra::DecFlags {
                            above_right: order_visible(
                                grids,
                                &cols,
                                plane,
                                r0 as i64 - 1,
                                c0 as i64 + (pw >> 2) as i64,
                                at,
                            ),
                            below_left: order_visible(
                                grids,
                                &cols,
                                plane,
                                r0 as i64 + (ph >> 2) as i64,
                                c0 as i64 - 1,
                                at,
                            ),
                        };
                    }
                    blocks[i].intra_ctx.dec_flags = Some(f);
                    // 帧内那一半走同一套 `derive_edges` ⇒ 也读上一行到 `x + 2w - 1`，
                    // 所以上界是"本块右边缘再往右一个块宽"（mi 列，不含）。
                    let bw4 = (w >> 2).max(1);
                    blocks[i].above_need = (c + 2 * bw4) as u32;
                }
            } else {
                let (at, len) = (blocks[i].txbs_at, blocks[i].txbs_len);
                for k in at..at + len {
                    let tj = txbs[k];
                    // 平面像素坐标 → 平面 4×4 坐标（`start_*` 本来就是**该平面**的像素；
                    // 与 `coeff.rs` 的 `(row >> sub_y)` 是同一个口径）
                    let r4 = tj.start_y >> 2;
                    let c4 = tj.start_x >> 2;
                    let step_x = (TX_WIDTH[tj.tx_sz as usize] as i32) >> 2;
                    let step_y = (TX_HEIGHT[tj.tx_sz as usize] as i32) >> 2;
                    let ev = event;
                    event += 1;
                    order_stamp(
                        grids,
                        &cols,
                        tj.plane,
                        r4 as i64,
                        c4 as i64,
                        step_x,
                        step_y,
                        ev,
                    );
                    txbs[k].flags = super::intra::DecFlags {
                        above_right: order_visible(
                            grids,
                            &cols,
                            tj.plane,
                            r4 as i64 - 1,
                            c4 as i64 + step_x as i64,
                            ev,
                        ),
                        below_left: flags::no_below_left()
                            == false
                            && order_visible(
                                grids,
                                &cols,
                                tj.plane,
                                r4 as i64 + step_y as i64,
                                c4 as i64 - 1,
                                ev,
                            ),
                    };
                    // 波前等待上界 = 本块所有变换块里"上一行读得最远"的那个
                    let need = above_need_mi(c4, step_x, tj.plane > 0 && sx != 0, txbs[k].flags.above_right);
                    blocks[i].above_need = blocks[i].above_need.max(need.max(0) as u32);
                }
            }
            if flags::trace_rowneed() {
                eprintln!(
                    "ROWB i={i} r={} c={} bs={} inter={} ii={} need={}",
                    info.mi_row,
                    info.mi_col,
                    info.mi_size,
                    u8::from(info.is_inter),
                    u8::from(info.interintra),
                    blocks[i].above_need
                );
            }
        }
    }

    /// 记一个变换块 + 它的系数。
    pub(crate) fn push_txb(&mut self, mut job: TxbJob, levels: &[i32]) {
        job.levels_at = self.levels.len();
        job.levels_len = levels.len();
        self.levels.extend_from_slice(levels);
        self.txbs.push(job);
    }

    /// **预留** `n` 个系数位（清零）并返回起始下标 —— 调用方拿到切片后**自己写**。
    ///
    /// 与 [`Self::push_txb`] 的分工：record 阶段本来有**两次**拷贝（`coeffs` 先写
    /// `TxScratch::levels`，`push_txb` 再整段拷进 arena）。用本方法把 arena 的
    /// 这段直接交给 `coeffs` 写，两次并一次 —— 省掉中间层与那趟整段拷贝
    /// （真素材上约 10% 帧时间的"记录开销"里的大头）。
    pub(crate) fn reserve_levels(&mut self, n: usize) -> usize {
        let at = self.levels.len();
        self.levels.resize(at + n, 0);
        at
    }

    /// 只推工单元数据（系数已由 [`Self::reserve_levels`] 直写进 arena）。
    pub(crate) fn push_txb_meta(&mut self, job: TxbJob) {
        debug_assert!(job.levels_at + job.levels_len <= self.levels.len());
        self.txbs.push(job);
    }

    fn reset(&mut self) {
        self.blocks.clear();
        self.txbs.clear();
        self.levels.clear();
    }
}

/// row-mt 回放的**共享上下文**（worker 之间只读）。
pub(crate) struct ReplayShared<'a> {
    /// ⚠️ **不变量**：同一时刻只有一个 worker 持有一个活的 `&mut *frame`，
    /// 且它只写自己那个 SB 行的像素 —— 行与行之间用 `AtomicBool`（Acquire/Release）
    /// 交接，放行下一行之前本 worker 已经不再碰帧。
    /// 与 `parallel::RowsSink` 是同一套口径（不变量写在类型上、release 不检查）。
    pub frame: *mut CurrFrame,
    pub grid: &'a BlockGrid,
    pub mi_grid: &'a MiGrid,
    pub store: &'a RefFrameStore,
    pub seq: &'a SequenceHeader,
    pub fh: &'a FrameHeader,
    pub blocks: &'a [BlockJob],
    pub txbs: &'a [TxbJob],
    pub levels: &'a [i32],
}

// SAFETY: 见 `frame` 的不变量；其余字段都是 `&`（`Sync`）。
unsafe impl Send for ReplayShared<'_> {}
unsafe impl Sync for ReplayShared<'_> {}

/// 回放时每个 worker 自己的一份状态（`BlockCtx` 里那份归语法线程）。
struct ReplayScratch {
    pred: super::inter::PredScratch,
    tx: super::coeff::TxScratch,
    /// 重建阶段用不到 `cdfs`/`ctxs`（`coeffs()` 只在语法阶段跑），但 `ResidualCtx`
    /// 要求这两个字段 ⇒ 每个 worker 备一份**最小**的（4×4 的表 + 一张 CDF）。
    cdfs: super::coeff::CoeffCdfs,
    ctxs: super::coeff::CoeffContexts,
    stats: super::coeff::CoeffStats,
}

impl ReplayScratch {
    fn new(qindex: u32) -> Self {
        Self {
            pred: Default::default(),
            tx: super::coeff::TxScratch::default(),
            cdfs: super::coeff::CoeffCdfs::new(qindex),
            ctxs: super::coeff::CoeffContexts::new(4, 4, true, true, 3),
            stats: super::coeff::CoeffStats::default(),
        }
    }
}

/// **诊断用的"已写"位图**（`ME_TRACE_RACE=1`）：按 4×4 格记账"这块像素被重建过没有"。
///
/// 用途是把并行回放里**漏掉的跨行依赖**直接抓出来：
///
/// - 顺序回放里，规范保证 `haveAbove` 为真时上面那一行**必然已重建** ⇒ 边推导**不该**
///   读到未写的格子（跑到零命中就是探针自检通过）；
/// - 并行回放里命中的那些读，就是"等错了/没等"的地方。
///
/// 只按**亮度**粒度记账：色度读换算成亮度格（4:2:0 时 ×2）再查，记账方也做同样换算。
/// 只在开关打开时才有开销（其余时候 [`is_written`] 直接返回 `true`）。
pub(crate) mod racemap {
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::{Mutex, OnceLock};

    struct Map {
        cols: usize,
        rows: usize,
        sx: bool,
        sy: bool,
        cells: Vec<AtomicU8>,
    }

    static M: OnceLock<Mutex<Option<Map>>> = OnceLock::new();

    fn slot() -> &'static Mutex<Option<Map>> {
        M.get_or_init(|| Mutex::new(None))
    }

    /// 每帧（每次回放）开始时调用：按亮度网格尺寸分配/清零。
    pub(crate) fn init(cols: usize, rows: usize, sx: bool, sy: bool) {
        let mut g = slot().lock().unwrap();
        match g.as_mut() {
            Some(v) if v.cols == cols && v.rows == rows => {
                for c in v.cells.iter() {
                    c.store(0, Ordering::Relaxed);
                }
                v.sx = sx;
                v.sy = sy;
            }
            _ => {
                let cells = (0..cols * rows).map(|_| AtomicU8::new(0)).collect();
                *g = Some(Map { cols, rows, sx, sy, cells });
            }
        }
    }

    /// 标记一块已经写了（亮度格坐标 + 尺寸；色度用 `chroma = true` 传入**色度**格坐标）。
    pub(crate) fn mark(r: i64, c: i64, w4: i64, h4: i64, chroma: bool) {
        let mut g = slot().lock().unwrap();
        let Some(v) = g.as_mut() else { return };
        // 色度格 → 亮度格：4:2:0 时 ×2
        let kx = if chroma && v.sx { 2 } else { 1 };
        let ky = if chroma && v.sy { 2 } else { 1 };
        for i in 0..(h4 * ky) {
            for j in 0..(w4 * kx) {
                let (rr, cc) = (r * ky + i, c * kx + j);
                if rr < 0 || cc < 0 {
                    continue;
                }
                let (rr, cc) = (rr as usize, cc as usize);
                if cc >= v.cols || rr >= v.rows {
                    continue;
                }
                v.cells[rr * v.cols + cc].store(1, Ordering::Release);
            }
        }
    }

    /// 该格写过没有（`plane > 0` 时传入色度格坐标，内部换算到亮度格）。
    pub(crate) fn is_written(r: i64, c: i64, chroma: bool) -> bool {
        let g = slot().lock().unwrap();
        let Some(v) = g.as_ref() else { return true };
        let kx = if chroma && v.sx { 2 } else { 1 };
        let ky = if chroma && v.sy { 2 } else { 1 };
        let (rr, cc) = (r * ky, c * kx);
        if rr < 0 || cc < 0 {
            return true;
        }
        let (rr, cc) = (rr as usize, cc as usize);
        if cc >= v.cols || rr >= v.rows {
            return true;
        }
        v.cells[rr * v.cols + cc].load(Ordering::Acquire) != 0
    }
}

/// row-mt 波前的等待：**先自旋、再让出**。
///
/// 纯 `spin_loop` 在"worker 数 ≈ 物理核数"时会互相偷 CPU（等待中的 worker 把核占满，
/// 正在干活的 worker 反而拿不到时间片）—— 实测 4 worker / 4 物理核上纯自旋比
/// 自旋+让出慢。前 64 次自旋覆盖"马上就绪"的常见情形（不付系统调用），之后就
/// `yield_now` 让出。
#[inline]
fn wait_until(cond: impl Fn() -> bool) {
    let timed = flags::trace_wait();
    let t0 = timed.then(std::time::Instant::now);
    let mut spins = 0u32;
    while !cond() {
        if spins < 64 {
            std::hint::spin_loop();
        } else {
            // `yield_now` 在"逻辑核数 > 物理核数"时仍然在烧核（MacBookAir 是 4 物理 /
            // 8 逻辑；8 个 worker 里等待的那几个把前端占满，干活的反而抢不到）——
            // 长等待直接睡一小段，把核让给真正在重建的 worker。
            std::thread::sleep(std::time::Duration::from_micros(20));
        }
        spins = spins.saturating_add(1);
    }
    if let Some(t0) = t0 {
        WAIT_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        WAIT_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// `AV1_TRACE_ROWNEED=1` 时记录"当前正在回放的块序号"，让 `ROWD2` 轨迹能精确归到块上
/// （顺序回放下严格正确；并行下仅供粗看）。见 `tools/av1ref/rowneed.py`。
pub(crate) static CUR_BLOCK_IDX: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

/// row-mt 波前空等统计（`ME_TRACE_WAIT=1`）。
static WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WAIT_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 取走并清零等待统计：`(总纳秒, 次数)`。
fn wait_stats_take() -> (u64, u64) {
    (
        WAIT_NS.swap(0, std::sync::atomic::Ordering::Relaxed),
        WAIT_N.swap(0, std::sync::atomic::Ordering::Relaxed),
    )
}

/// row-mt 作业的**发牌顺序键**：`(反对角线, 列号降序)`。
///
/// 这个键必须是作业依赖关系的**拓扑序**（依赖的键严格更小），否则会死锁。
/// 三条边的键都满足：
///
/// | 依赖 | 键 |
/// |---|---|
/// | 同带左邻列 `(band, col-1)` | `(d-1, ·)` |
/// | 上一带同列 `(band-1, col)` | `(d-1, ·)` |
/// | 上一带次列 `(band-1, col+1)` | `(d, MAX-(col+1))` **小于** `(d, MAX-col)` |
///
/// 见 `job_order_key_is_a_topological_order_of_the_dependencies`。
#[inline]
fn job_order_key(j: &impl JobKey) -> (u32, u32) {
    (j.diag(), u32::MAX - j.col_key())
}

/// 让 [`job_order_key`] 能同时吃内部 `Job` 与单测里的假作业（只依赖这两个字段）。
pub(crate) trait JobKey {
    fn diag(&self) -> u32;
    fn col_key(&self) -> u32;
}

/// 一个变换块要从**上一行**读到哪个 mi 列（**不含**）。
///
/// 依据 [`super::intra::derive_edges`]：`have_above_right` 为真时 `AboveRow` 的上右区要
/// 读到 `x + 2w - 1`，否则只到 `x + w - 1`。`c4` / `step` 都是**该平面**的 4×4 单位，
/// 换成 mi 列时：亮度 1 格 = 1 个 mi 列，色度 1 格 = 2 个 mi 列。
///
/// 这是 row-mt 波前**唯一**的跨行依赖（写只写自己那行、下方读留在同一 SB 行内），
/// 之前写死成"块右边缘 + 1"，块宽 > 8px 时就不够 —— 于是并行版会读到上一行
/// **还没重建**的像素（`ME_THREADS=1` 时不复现，因为整行是顺序的）。
#[inline]
fn above_need_mi(c4: i32, step: i32, subsampled: bool, above_right: bool) -> i32 {
    let k = if above_right { 2 } else { 1 };
    let need = c4 + k * step;
    if subsampled {
        2 * need
    } else {
        need
    }
}

/// 该块的**重建**是否要读**当前帧**的像素 —— 也就是并行回放的波前必须等它的
/// 左邻 / 上邻。
///
/// * 帧内块：`intra::derive_edges` 读上/左邻；
/// * `interintra`：帧内那一半走同一套 `derive_edges`；
/// * **纯帧间块：不读**。预测全部来自参考帧；`warp` 的样本只从**邻居的 MV** 估计
///   （[`super::mv::MotionField::find_warp_samples`]，语法阶段就填好了、从不读当前块），
///   OBMC 也只消费参考帧 + 邻居 MV。`intrabc` 是唯一的例外，但它在切作业时就把
///   整帧退回了顺序回放（见 [`BlockCtx::replay_jobs_parallel`]）。
///
/// 这是 [`BlockCtx::replay_jobs_parallel`] 判断"**同带左邻列**"这条边要不要画的依据：
/// 整列都是纯帧间块时那条边可以省 —— 一条带内就不再是串行链。
#[inline]
fn reads_cur_frame(b: &BlockJob) -> bool {
    !b.info.is_inter || (b.info.interintra && !flags::no_interintra())
}

/// 顺序网格的"标记"（越界规则照抄 [`CurrFrame::mark_decoded`]）。
#[allow(clippy::too_many_arguments)]
fn order_stamp(
    grids: &mut [Vec<u32>],
    cols: &[usize],
    plane: u8,
    r: i64,
    c: i64,
    w4: i32,
    h4: i32,
    v: u32,
) {
    let p = plane as usize;
    let w = cols[p];
    for i in 0..h4 {
        for j in 0..w4 {
            let (rr, cc) = (r + i as i64, c + j as i64);
            if rr < 0 || cc < 0 {
                continue;
            }
            let (rr, cc) = (rr as usize, cc as usize);
            if cc >= w || rr * w + cc >= grids[p].len() {
                continue;
            }
            grids[p][rr * w + cc] = v;
        }
    }
}

/// 顺序网格的"查询"（越界规则照抄 [`CurrFrame::decoded`]）：该格子已被**更早**的
/// 事件（下标 `< at`）标记，或在本批之前就标过（初值 1）。
fn order_visible(grids: &[Vec<u32>], cols: &[usize], plane: u8, r: i64, c: i64, at: u32) -> bool {
    let p = plane as usize;
    if r < 0 || c < 0 {
        return false;
    }
    let w = cols[p];
    let (r, c) = (r as usize, c as usize);
    if c >= w || r * w + c >= grids[p].len() {
        return false;
    }
    let v = grids[p][r * w + c];
    v != 0 && v < at
}

/// 顺序回放（自带 scratch；调色板 flush 与 `ME_ROW_MT=2` 走这条）。
fn replay_blocks(shared: &ReplayShared<'_>, range: std::ops::Range<usize>) -> Result<()> {
    let mut sc = ReplayScratch::new(u32::from(shared.fh.params.quant.base_q_idx));
    replay_blocks_with(shared, range, &mut sc)
}

/// **row-mt 回放**：把 `shared.blocks[range]` 按记录序做像素侧
/// （`compute_prediction` + 逐 txb 的帧内预测/残差叠加 + `BlockDecoded` 标记）。
///
/// 语法侧的一切（网格、上下文、统计）在语法阶段就写完了，这里一律不碰。
fn replay_blocks_with(
    shared: &ReplayShared<'_>,
    range: std::ops::Range<usize>,
    sc: &mut ReplayScratch,
) -> Result<()> {
    let trace = flags::trace_rowneed();
    for i in range {
        if trace {
            CUR_BLOCK_IDX.store(i, std::sync::atomic::Ordering::Relaxed);
        }
        replay_one(shared, &shared.blocks[i], sc)?;
    }
    Ok(())
}

/// 回放**一个块**（像素侧）。
fn replay_one(shared: &ReplayShared<'_>, bj: &BlockJob, sc: &mut ReplayScratch) -> Result<()> {
    {
        let t_a = std::time::Instant::now();
        let info = bj.info;
        let (r, c) = (info.mi_row as i32, info.mi_col as i32);
        let bsize = info.mi_size;
        // SAFETY: 见 `ReplayShared::frame` 的不变量（同一时刻单行独占）。
        let frame: &mut CurrFrame = unsafe { &mut *shared.frame };
        if info.is_inter {
            let refs: &[Option<std::sync::Arc<CurrFrame>>; 8] = &shared.store.pixels;
            super::inter::compute_prediction(
                frame, shared.grid, shared.mi_grid, shared.seq, shared.fh, refs, &info, r, c,
                &bj.warp, &bj.intra_ctx, &mut sc.pred,
            )?;
        }
        let t_b = std::time::Instant::now();
        let cfg = bj.cfg.expect("语法阶段必然写过 cfg");
        let txb_slice = &shared.txbs[bj.txbs_at..bj.txbs_at + bj.txbs_len];
        if !txb_slice.is_empty() {
            let mut ctx = ResidualCtx {
                cdfs: &mut sc.cdfs,
                ctxs: &mut sc.ctxs,
                grid: super::coeff::GridAccess::Read(shared.grid),
                stats: &mut sc.stats,
                cfg,
                frame: Some(&mut *frame),
                max_luma_w: 0,
                max_luma_h: 0,
                trace_count: 0,
                // 调色板块在语法阶段就地重建（flush）⇒ 回放里不会有它
                palette: None,
                record: None,
                // `replay_txbs` 逐个 txb 填（`TxbJob::flags`）
                flags: None,
            };
            ctx.replay_txbs(txb_slice, shared.levels, &mut sc.tx)?;
        }
        let t_c = std::time::Instant::now();
        if info.is_inter {
            // `BlockDecoded` 标记（与 `decode_block` 的收尾同一套坐标推导）
            let num_planes = shared.seq.color.mono_chrome.then_some(1).unwrap_or(3);
            let (sx, sy) = (
                u8::from(shared.seq.color.subsampling_x),
                u8::from(shared.seq.color.subsampling_y),
            );
            let has_chroma = bj.has_chroma;
            let (w, h) = (block_width(bsize), block_height(bsize));
            for plane in 0..num_planes {
                if plane > 0 && !has_chroma {
                    continue;
                }
                let (sub_x, sub_y) = (plane > 0 && sx != 0, plane > 0 && sy != 0);
                let (pw, ph) = (if sub_x { w >> 1 } else { w }, if sub_y { h >> 1 } else { h });
                let (r0, c0) = (if sub_y { r >> 1 } else { r }, if sub_x { c >> 1 } else { c });
                frame.mark_decoded(plane, r0 as i64, c0 as i64, (pw >> 2).max(1), (ph >> 2).max(1));
            }
        }
        // 诊断（`ME_TRACE_RACE=1`）：本块的像素已重建完 ⇒ 在"已写"位图上盖章。
        // ⚠️ 必须**在块跑完之后**盖 —— 块内读自己上面那一行是不可能的（上面那一行在块外），
        // 所以"块级盖章"对**上方/左侧**读的判定是精确的。
        if flags::trace_race() {
            let (sx, sy) = (
                shared.seq.color.subsampling_x,
                shared.seq.color.subsampling_y,
            );
            let (w, h) = (block_width(bsize), block_height(bsize));
            racemap::mark(
                r as i64,
                c as i64,
                (w >> 2).max(1) as i64,
                (h >> 2).max(1) as i64,
                false,
            );
            if bj.has_chroma {
                let (kx, ky) = (1 << u32::from(sx), 1 << u32::from(sy));
                racemap::mark(
                    (r >> u32::from(sy)) as i64,
                    (c >> u32::from(sx)) as i64,
                    ((w / kx) >> 2).max(1) as i64,
                    ((h / ky) >> 2).max(1) as i64,
                    true,
                );
            }
        }
        let t_d = std::time::Instant::now();
        {
            use std::sync::atomic::Ordering;
            DBG_MC_NS.fetch_add((t_b - t_a).as_nanos() as u64, Ordering::Relaxed);
            DBG_TXB_NS.fetch_add((t_c - t_b).as_nanos() as u64, Ordering::Relaxed);
            DBG_MARK_NS.fetch_add((t_d - t_c).as_nanos() as u64, Ordering::Relaxed);
        }
    }
    Ok(())
}

pub struct BlockCtx<'a> {
    pub grid: &'a mut BlockGrid,
    pub seq: &'a SequenceHeader,
    pub fh: &'a FrameHeader,
    pub store: &'a RefFrameStore,
    pub field: Option<&'a mv::MotionField>,
    /// 帧级 CDF 上下文（跨瓦片共享，见 [`super::context`]）里的这两组。
    pub cdfs: &'a mut BlockCdfs,
    pub coeff_cdfs: &'a mut CoeffCdfs,
    pub coeff_ctxs: &'a mut CoeffContexts,
    /// 瓦片范围（`is_inside` 用的就是它，不是帧的范围）
    pub mi_row_start: i32,
    pub mi_row_end: i32,
    pub mi_col_start: i32,
    pub mi_col_end: i32,
    /// 瓦片局部的量化/滤波状态（规范里 `decode_tile` 开头初始化）
    pub current_q_index: i32,
    pub delta_lfs: [i32; FRAME_LF_COUNT],
    pub lossless_array: [bool; MAX_SEGMENTS],
    pub stats: BlockStats,
    /// 帧级像素缓冲。`None` = 只读语法不做像素重建（真素材的语法验收走这条路）。
    pub frame: Option<&'a mut CurrFrame>,
    /// 帧间预测值的**复用落点**（`inter::scratch_of`）：`[0]`/`[1]` 是两路预测值、
    /// `[2]` 是 MC 的横向中间量缓冲。
    ///
    /// 放在这里是因为 `BlockCtx` 是**每瓦片**建一次、`decode_block` 每块调一次 ——
    /// 预测值缓冲正好该有"比块长、比帧短"的生命周期。塞进 `compute_prediction`
    /// 的局部量就等于每块新开（每块 malloc + memset 各 1~2 次，1080p 一帧几万块）。
    pub pred_scratch: super::inter::PredScratch,
    /// **row-mt 的工单**（`ME_ROW_MT=1` 时用；默认路径不碰它）。
    pub row_mt: RowMtJobs,
    /// 残差/反变换链路的复用缓冲（`coeff::TxScratch`）。理由同 `pred_scratch`：
    /// 逐块 `vec![]` 一次是纯浪费，缓冲的生命周期应该跟瓦片一样长。
    pub tx_scratch: super::coeff::TxScratch,
    /// palette 的临时状态：本块的颜色表与索引图。
    ///
    /// **只能放在这里，不能放进 `BlockInfo`**：`BlockInfo` 是 `Copy`（模式信息按值
    /// 传递），装不下 `Vec`。这些量只在"当前这一块"内有效，逐块整体覆盖。
    palette_colors_y: [u16; PALETTE_COLORS],
    palette_colors_u: [u16; PALETTE_COLORS],
    palette_colors_v: [u16; PALETTE_COLORS],
    /// `ColorMapY` / `ColorMapUV`（逐样本的颜色索引）。都按最大尺寸 64×64 开
    /// （4:4:4 时色度也是 64×64），行宽用实际块宽。
    color_map_y: Vec<u8>,
    color_map_uv: Vec<u8>,
    /// 规范里的全局 `segment_id`
    segment_id: u8,
    /// 规范里的 `ReadDeltas`：**每个超块**只在第一个块上读一次 delta_q / delta_lf
    read_deltas: bool,
    /// 上一次走过的**超块行**（`decode_tile` 每行要 `clear_left_context()`）
    last_sb_row: i32,
    cur_row: i32,
    cur_col: i32,
    cur_bsize: u8,
}

impl<'a> BlockCtx<'a> {
    pub fn new(
        grid: &'a mut BlockGrid,
        seq: &'a SequenceHeader,
        fh: &'a FrameHeader,
        store: &'a RefFrameStore,
        field: Option<&'a mv::MotionField>,
        coeff_ctxs: &'a mut CoeffContexts,
        tile: &Tile<'_>,
        cdfs: &'a mut BlockCdfs,
        coeff_cdfs: &'a mut CoeffCdfs,
        frame: Option<&'a mut CurrFrame>,
    ) -> Self {
        let base_q_idx = fh.params.quant.base_q_idx as u32;
        Self {
            grid,
            seq,
            fh,
            store,
            field,
            cdfs,
            coeff_cdfs,
            coeff_ctxs,
            mi_row_start: tile.mi_row_start as i32,
            mi_row_end: tile.mi_row_end as i32,
            mi_col_start: tile.mi_col_start as i32,
            mi_col_end: tile.mi_col_end as i32,
            current_q_index: base_q_idx as i32,
            delta_lfs: [0; FRAME_LF_COUNT],
            lossless_array: lossless_array(fh),
            stats: BlockStats::default(),
            frame,
            pred_scratch: Default::default(),
            tx_scratch: super::coeff::TxScratch::default(),
            row_mt: RowMtJobs::default(),
            palette_colors_y: [0; PALETTE_COLORS],
            palette_colors_u: [0; PALETTE_COLORS],
            palette_colors_v: [0; PALETTE_COLORS],
            color_map_y: vec![0; 64 * 64],
            color_map_uv: vec![0; 64 * 64],
            segment_id: 0,
            read_deltas: false,
            last_sb_row: -1,
            cur_row: 0,
            cur_col: 0,
            cur_bsize: BLOCK_4X4 as u8,
        }
    }

    /// `is_inside( r, c )`：**瓦片**范围。
    #[inline]
    fn is_inside(&self, r: i32, c: i32) -> bool {
        r >= self.mi_row_start && r < self.mi_row_end && c >= self.mi_col_start && c < self.mi_col_end
    }

    #[inline]
    fn num_planes(&self) -> u8 {
        if self.seq.color.mono_chrome {
            1
        } else {
            3
        }
    }

    fn sub_x(&self) -> bool {
        self.seq.color.subsampling_x
    }

    fn sub_y(&self) -> bool {
        self.seq.color.subsampling_y
    }

    /// `HasChroma`（规范 §5.11.2 开头）
    fn has_chroma(&self, r: i32, c: i32, bsize: u8) -> bool {
        let bw4 = NUM_4X4_BLOCKS_WIDE[bsize as usize] as i32;
        let bh4 = NUM_4X4_BLOCKS_HIGH[bsize as usize] as i32;
        if bh4 == 1 && self.sub_y() && (r & 1) == 0 {
            return false;
        }
        if bw4 == 1 && self.sub_x() && (c & 1) == 0 {
            return false;
        }
        self.num_planes() > 1
    }

    /// `decode_block( r, c, subSize )`
    pub fn decode_block(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        mi_grid: &MiGrid,
        r: i32,
        c: i32,
        bsize: u8,
    ) -> Result<()> {
        self.cur_row = r;
        self.cur_col = c;
        self.cur_bsize = bsize;
        // 诊断（`AV1_TRACE_LASTSB`）：只打"贴着帧右/下边界"的块 ——
        // 帧尺寸不是超块整数倍时，最后一条超块行/列最容易多解或少解块。
        if trace_last_sb() {
            let bh4 = NUM_4X4_BLOCKS_HIGH[bsize as usize] as i32;
            let bw4 = NUM_4X4_BLOCKS_WIDE[bsize as usize] as i32;
            if r + bh4 > self.fh.mi_rows as i32 - 32 || c + bw4 > self.fh.mi_cols as i32 - 32 {
                eprintln!(
                    "SB-BLK r={r} c={c} bsize={bsize} size={}x{} bits={} smb={}",
                    bw4 * 4,
                    bh4 * 4,
                    dec.consumed_bits(),
                    dec.symbol_max_bits()
                );
            }
        }
        // `decode_tile` 每个超块开头做 `ReadDeltas = delta_q_present`；
        // 第一个叶子块一定落在超块原点（分区树的四种形态都先解 (r_sb, c_sb)），
        // 所以这里用"位置正好是超块原点"来等价地判"本超块的第一个块"。
        let sb4 = if self.seq.use_128x128_superblock { 32 } else { 16 };
        if r % sb4 == 0 && c % sb4 == 0 {
            self.read_deltas = self.fh.params.delta_q.present;
            // 规范 `decode_tile`：**每个超块**开头 `clear_cdef( r, c )`
            self.grid.clear_cdef(r, c, sb4);
            // 规范 `decode_tile`：**每个超块行**开头 `clear_left_context( )`
            let sb_row = r / sb4;
            if sb_row != self.last_sb_row {
                self.last_sb_row = sb_row;
                self.coeff_ctxs.clear_left();
            }
        }
        let bw4 = NUM_4X4_BLOCKS_WIDE[bsize as usize] as i32;
        let bh4 = NUM_4X4_BLOCKS_HIGH[bsize as usize] as i32;
        let has_chroma = self.has_chroma(r, c, bsize);
        if !has_chroma {
            self.stats.no_chroma_blocks += 1;
        }

        let pos0 = dec.consumed_bits();
        // row-mt：本块在工单里的下标（语法阶段记、重建阶段回放）
        let mut job_idx: Option<usize> = None;
        let mut info = BlockInfo { mi_row: r as u32, mi_col: c as u32, mi_size: bsize, ..Default::default() };
        if self.fh.is_intra {
            self.intra_frame_mode_info(dec, mi_grid, r, c, bsize, &mut info)?;
        } else {
            self.inter_frame_mode_info(dec, mi_grid, r, c, bsize, &mut info)?;
        }
        // 规范 `decode_block`：`mode_info()` → `palette_tokens()` → `read_block_tx_size()`。
        self.palette_tokens(dec, &info)?;
        self.read_block_tx_size(dec, mi_grid, &mut info)?;
        // 逐块属性 trace：此刻 `info` 已经填全（含 MV / 插值滤波 / 变换尺寸）。
        if trace_blk_attr() {
            let n = BLK_ATTR_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            eprintln!(
                "BLKATTR#{n} r={r} c={c} bs={bsize} inter={} skip={} ref={},{} mv={},{},{},{} \
                 if={},{} tx={} mm={} ct={} cg={} ii={} oh={} sf={}",
                u8::from(info.is_inter),
                u8::from(info.skip),
                info.ref_frame[0],
                info.ref_frame[1],
                info.mv[0][0],
                info.mv[0][1],
                info.mv[1][0],
                info.mv[1][1],
                info.interp_filter[0],
                info.interp_filter[1],
                info.tx_size,
                info.motion_mode,
                info.compound_type,
                info.comp_group_idx,
                u8::from(info.interintra),
                self.fh.order_hint,
                u8::from(self.fh.show_frame),
            );
        }
        if info.skip {
            self.coeff_ctxs
                .reset_block(r, c, bw4, bh4, self.sub_x(), self.sub_y(), self.num_planes());
        }
        // row-mt：调色板块的预测要用**活的**调色板字段（`palette_colors_*` /
        // `color_map_*`），而那是逐块覆盖的、重建又被推迟了 ⇒ 先把已记录的工单
        // 回放掉（flush），本块**就地重建**。调色板块只在屏幕内容流里出现，罕见。
        if self.row_mt.syntax_phase
            && !self.row_mt.inline_block
            && (info.palette_size_y > 0 || info.palette_size_uv > 0)
        {
            self.replay_jobs(mi_grid)?;
            self.row_mt.inline_block = true;
        }

        // 规范：`residual()` **之前**先把模式信息铺进网格
        {
            let y = info.y_mode;
            let uv = info.uv_mode;
            let is_inter = info.is_inter;
            let refs = info.ref_frame;
            let cg = info.comp_group_idx;
            let ci = info.compound_idx;
            let ct = info.compound_type;
            let wix = info.wedge_index;
            let wsg = info.wedge_sign;
            let mty = info.mask_type;
            let ifs = info.interp_filter;
            let mvs = info.mv;
            let iim = info.interintra_mode;
            self.grid.stamp(r, c, bsize, move |g, i| {
                g.y_modes[i] = y;
                if refs[0] == mv::INTRA_FRAME && has_chroma {
                    g.uv_modes[i] = uv;
                }
                g.ref_frames[i] = refs;
                if is_inter {
                    g.comp_group_idxs[i] = cg;
                    g.compound_idxs[i] = ci;
                    g.compound_types[i] = ct;
                    g.wedge_idxs[i] = wix;
                    g.wedge_signs[i] = wsg;
                    g.mask_types[i] = mty;
                    g.interp_filters[i] = ifs;
                    g.mvs[i] = mvs;
                    g.interintra_modes[i] = iim;
                }
                g.written[i] = true;
            });
        }

        self.stats.mode_bits += (dec.consumed_bits() - pos0) as u64;
        // row-mt 语法阶段：本块只记录、不写像素（调色板块除外，见下面的 flush）。
        let recording = self.row_mt.syntax_phase && !self.row_mt.inline_block;
        if recording && !info.is_inter {
            // 帧内块的重建（逐变换块预测）在 `transform_block` 里，同样后置；
            // 这里先把模式信息记下来（`warp`/`intra_ctx` 只有帧间块用得到）。
            job_idx = Some(self.row_mt.start_block(
                &info,
                super::inter::WarpPlan::default(),
                super::inter::IntraContext::default(),
                has_chroma,
            ));
        }
        // compute_prediction()：像素预测（M5），不读位。
        // 帧内块在 `transform_block` 里按变换块预测；**帧间块在这里按"预测块"预测**
        // （规范 §7.11.2 就是这么分的：interintra 的帧内那一半也在这里做）。
        if info.is_inter {
            // `warp_plan` 借 `&self`，`frame` 借 `&mut self` ⇒ 先算计划再取帧
            let warp = self.warp_plan(mi_grid, &info, bsize);
            // `interintra` 的帧内那一半要的可用性 / 边滤波上下文（§7.11.2 的
            // `AvailL`/`AvailU` 与 §7.11.2.8 的 `filterType`）。与 `residual()` 里那份
            // 同源，这里先算好，避免在 `&mut self.frame` 借用期间再借 `&self`。
            let intra_ctx = super::inter::IntraContext {
                avail_u: self.is_inside(r - 1, c),
                avail_l: self.is_inside(r, c - 1),
                filter_type_y: self.filter_type_for(0, r, c, bsize),
                filter_type_uv: self.filter_type_for(1, r, c, bsize),
                // row-mt 语法阶段：回放前由 `precompute_dec_flags` 回填；
                // 就地重建（非 row-mt）时照旧读运行时网格
                dec_flags: None,
            };
            if recording {
                job_idx = Some(self.row_mt.start_block(&info, warp, intra_ctx, has_chroma));
            }
            // row-mt 语法阶段**不写像素**：工单里已经带上了 `warp` / `intra_ctx`，
            // 回放时 `replay_one` 会用同一份参数再算一次。这里若也跑一遍，
            // 就是**整帧预测算两遍**（第二遍覆盖第一遍，结果被丢弃）。
            //
            // 这正是 row-mt 一直比 inline 慢的根因：语法阶段本应只有 16~26 ms，
            // 实测却到 58~61 ms，差出来的 ~40 ms 就是这遍白跑的 MC。
            // 调色板块（`inline_block`）**不受影响** —— 它 recording=false，照旧就地预测。
            if let Some(frame) = self.frame.as_deref_mut().filter(|_| !recording) {
                let refs: &[Option<std::sync::Arc<CurrFrame>>; 8] = &*self.store.pixels;
                super::inter::compute_prediction(
                    frame, self.grid, mi_grid, self.seq, self.fh, refs, &info, r, c, &warp,
                    &intra_ctx, &mut self.pred_scratch,
                )?;
            }
        }
        self.residual(dec, &info, job_idx)?;
        if let Some(idx) = job_idx {
            self.row_mt.end_block(idx);
        }

        // 规范 `transform_block` 的收尾循环会把覆盖到的 4×4 标成"已重建"
        // （`BlockDecoded`）。帧内块在 `reconstruct_block` 里逐变换块标过了，**帧间块
        // （含 `use_intrabc` 的拷贝块）也必须标** —— 它们同样是"已经重建出来的像素"，
        // 后面任何帧内块（以及 `interintra` 的帧内那一半）算
        // `haveAboveRight` / `haveBelowLeft` 时都要看得见它们。
        // 漏标的症状很隐蔽：上右方正好是帧间块时，帧内预测的边沿会退化成"复制一整列"，
        // 真实的上右样本被丢掉 ⇒ 该块预测整体差一个常数（恰好是两处样本之差），
        // 再顺着 DC 预测向右/向下逐块衰减扩散（48 → 30 → 20 → … → 0）。
        if info.is_inter && !recording {
            let num_planes = self.num_planes();
            let has_chroma = self.has_chroma(r, c, bsize);
            let (sx, sy) = (self.sub_x(), self.sub_y());
            let (w, h) = (block_width(bsize), block_height(bsize));
            if let Some(frame) = self.frame.as_deref_mut() {
                for plane in 0..num_planes {
                    if plane > 0 && !has_chroma {
                        continue;
                    }
                    // 4×4 网格按**平面分辨率**分格；小块的色度可能不足一格，
                    // 但规范也是"每个变换块至少标一格"，这里同样取 `max(1)`。
                    let (sub_x, sub_y) = (plane > 0 && sx, plane > 0 && sy);
                    let (pw, ph) = (if sub_x { w >> 1 } else { w }, if sub_y { h >> 1 } else { h });
                    let (r0, c0) = (if sub_y { r >> 1 } else { r }, if sub_x { c >> 1 } else { c });
                    frame.mark_decoded(plane, r0 as i64, c0 as i64, (pw >> 2).max(1), (ph >> 2).max(1));
                }
            }
        }

        // 规范：`decode_block` 结尾的第二段双重循环
        {
            let ii = info.is_inter as u8;
            let sm = info.skip_mode as u8;
            let sk = info.skip as u8;
            let tx = info.tx_size;
            let seg = info.segment_id as i8;
            let ps = [info.palette_size_y, info.palette_size_uv];
            // `PaletteColors` 也要按 4×4 铺开：下一块的 `get_palette_cache` 就是
            // 从这里取"上/左调色板"的（UV 只存 U 分量，与规范一致）。
            let pc = [self.palette_colors_y, self.palette_colors_u];
            // `DeltaLF` 的累积值也要按块铺进网格（§7.14.4 去块强度要逐 4×4 读它）
            let dlf = self.delta_lfs;
            self.grid.stamp(r, c, bsize, move |g, i| {
                g.is_inters[i] = ii;
                g.skip_modes[i] = sm;
                g.skips[i] = sk;
                g.inter_tx_sizes[i] = tx;
                g.seg_ids[i] = seg;
                g.palette_sizes[i] = ps;
                g.palette_colors[i] = pc;
                g.delta_lfs[i] = dlf;
            });
        }

        // row-mt：本块（调色板）的就地重建结束，回到"只记录"模式
        self.row_mt.inline_block = false;
        self.stats.bits += (dec.consumed_bits() - pos0) as u64;
        self.stats.blocks += 1;
        self.stats.skip_blocks += info.skip as u64;
        self.stats.skip_mode_blocks += info.skip_mode as u64;
        self.stats.inter_blocks += info.is_inter as u64;
        self.stats.compound_blocks += (info.ref_frame[1] > mv::INTRA_FRAME) as u64;
        self.stats.y_mode_histogram[(info.y_mode as usize) % INTRA_MODES as usize] += 1;
        self.stats.uv_mode_histogram[(info.uv_mode as usize) % 14] += 1;
        self.stats.tx_size_histogram[info.tx_size as usize] += 1;
        self.stats.segment_histogram[(info.segment_id as usize).min(MAX_SEGMENTS - 1)] += 1;
        self.stats.motion_mode_histogram[(info.motion_mode as usize) % 3] += 1;
        self.stats.compound_type_histogram[(info.compound_type as usize) % 5] += 1;
        for rf in info.ref_frame {
            if rf > mv::INTRA_FRAME && (rf as usize) < 8 {
                self.stats.ref_frame_histogram[rf as usize] += 1;
            }
        }
        Ok(())
    }

    // ── mode_info ───────────────────────────────────────────────────────

    /// `intra_frame_mode_info( )`（规范 §5.11.3）
    fn intra_frame_mode_info(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        mi_grid: &MiGrid,
        r: i32,
        c: i32,
        bsize: u8,
        info: &mut BlockInfo,
    ) -> Result<()> {
        let tm0 = dec.consumed_bits();
        let t_skip0 = dec.consumed_bits();
        let seg_id_pre_skip = self.fh.params.segmentation.seg_id_pre_skip;
        if seg_id_pre_skip {
            info.segment_id = self.intra_segment_id(dec, r, c, false)?;
        }
        dec.label = "skip";
        info.skip = self.read_skip(dec, r, c, info.segment_id)?;
        let t_skip = dec.consumed_bits() - t_skip0;
        if !seg_id_pre_skip {
            info.segment_id = self.intra_segment_id(dec, r, c, info.skip)?;
        }
        let tm = trace_mode();
        let t_a = dec.consumed_bits();
        self.read_cdef(dec, r, c, bsize, info.skip)?;
        let t_b = dec.consumed_bits();
        self.read_delta_qindex(dec, r, c, bsize, info.skip)?;
        let t_c = dec.consumed_bits();
        self.read_delta_lf(dec, r, c, bsize, info.skip)?;
        let t_d = dec.consumed_bits();
        self.read_deltas = false;
        info.ref_frame = [mv::INTRA_FRAME, NONE];
        info.is_inter = false;

        if self.fh.allow_intrabc {
            dec.label = "intrabc";
            let s = read_sym(dec, &mut self.stats, &mut self.cdfs.intrabc)?;
            info.use_intrabc = s == 1;
        } else {
            info.use_intrabc = false;
        }
        if info.use_intrabc {
            // `use_intrabc == 1`：这一块**走帧间那一套**，只是"参考帧"换成了当前帧
            // 已经重建的那部分（规范 §5.11.5 的 `if ( use_intrabc )` 支）。
            //
            // 规范把这一批语法元素**全部写死、不读位**：YMode / UVMode = DC_PRED、
            // motion_mode = SIMPLE、compound_type = COMPOUND_AVERAGE、
            // PaletteSizeY/UV = 0、interp_filter = BILINEAR。所以下面那段帧内模式、
            // `palette_mode_info` 与 `filter_intra_mode_info` **都必须跳过** ——
            // 照读就是多读符号，整条算术流从这里开始错位。
            info.is_inter = true;
            info.y_mode = DC_PRED;
            info.uv_mode = DC_PRED;
            info.motion_mode = SIMPLE;
            info.compound_type = COMPOUND_AVERAGE;
            info.palette_size_y = 0;
            info.palette_size_uv = 0;
            info.interp_filter = [BILINEAR, BILINEAR];
            // `isCompound = 0`；`ref_frame` 仍是 `[INTRA_FRAME, NONE]` —— 这不是笔误，
            // 正是靠它，`add_ref_mv_candidate` 才会把"同样用 intrabc 的邻居"当成
            // 候选收进 `RefStackMv`（邻居的 `RefFrames[..][0]` 也是 INTRA_FRAME）。
            let stack = self.find_mv_stack(mi_grid, r, c, bsize, info.ref_frame, false);
            self.assign_mv(dec, &stack, NEWMV, false, 0, true, info)?;
            self.stats.intrabc_blocks += 1;
            // 诊断：`AV1_TRACE_IBC=1` 逐块打出块矢量。intrabc 是"静默的像素来源"，
            // 只看解码过不过分不清"这条流没用它"与"用对了"，所以需要一份块级证据
            // （与 `AV1_NO_INTRABC` 的差异对拍互为印证）。
            if flags::trace_ibc() {
                eprintln!(
                    "  IBC r={r} c={c} bs={bsize} stackN={} mv=({},{})",
                    stack.num, info.mv[0][0], info.mv[0][1]
                );
            }
            return Ok(());
        }

        let lossless = self.lossless_array[info.segment_id as usize];
        let (above_mode, left_mode) = self.neighbour_intra_context(r, c);
        dec.label = "intra_frame_y_mode";
        info.y_mode = read_sym(
            dec,
            &mut self.stats,
            &mut self.cdfs.intra_frame_y_mode[above_mode][left_mode],
        )? as u8;
        let t_e = dec.consumed_bits();
        info.angle_delta_y = self.intra_angle_info(dec, info.y_mode, bsize)?;
        if self.has_chroma(r, c, bsize) {
            info.uv_mode = self.read_uv_mode(dec, info.y_mode, bsize, lossless)?;
            if info.uv_mode == UV_CFL_PRED {
                let (u, v) = self.read_cfl_alphas(dec)?;
                info.cfl_alpha_u = u;
                info.cfl_alpha_v = v;
                self.stats.cfl_blocks += 1;
            }
            info.angle_delta_uv = self.intra_angle_info(dec, info.uv_mode, bsize)?;
        }
        info.palette_size_y = 0;
        info.palette_size_uv = 0;
        self.read_palette_mode_info(dec, r, c, bsize, info)?;
        let t_f = dec.consumed_bits();
        self.filter_intra_mode_info(dec, info, bsize)?;
        if tm && (self.stats.blocks < 6 || flags::trace_all()) {
            println!(
                "    F{} MODE r={} c={} bs={} skip={} nbr=({},{}) ymode={} | skip={} cdef={} dq={} dlf={} ymode={} ang={} uv={} fi={} pal=({},{})",
                TRACE_FRAME_NO.load(std::sync::atomic::Ordering::Relaxed),
                r,
                c,
                bsize,
                info.skip as u8,
                above_mode,
                left_mode,
                info.y_mode,
                t_a - tm0,
                t_b - t_a,
                t_c - t_b,
                t_d - t_c,
                t_e - t_d,
                t_f - t_e,
                dec.consumed_bits() - t_f,
                t_skip,
                info.palette_size_y,
                info.palette_size_uv
            );
        }
        Ok(())
    }

    /// `intra_block_mode_info( )`（帧间帧里的帧内块 —— 用 `y_mode` 那套 CDF）
    fn intra_block_mode_info(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        r: i32,
        c: i32,
        bsize: u8,
        info: &mut BlockInfo,
    ) -> Result<()> {
        info.ref_frame = [mv::INTRA_FRAME, NONE];
        let lossless = self.lossless_array[info.segment_id as usize];
        let group = SIZE_GROUP[bsize as usize];
        info.y_mode = read_sym(dec, &mut self.stats, &mut self.cdfs.y_mode[group])? as u8;
        info.angle_delta_y = self.intra_angle_info(dec, info.y_mode, bsize)?;
        if self.has_chroma(r, c, bsize) {
            info.uv_mode = self.read_uv_mode(dec, info.y_mode, bsize, lossless)?;
            if info.uv_mode == UV_CFL_PRED {
                let (u, v) = self.read_cfl_alphas(dec)?;
                info.cfl_alpha_u = u;
                info.cfl_alpha_v = v;
                self.stats.cfl_blocks += 1;
            }
            info.angle_delta_uv = self.intra_angle_info(dec, info.uv_mode, bsize)?;
        }
        info.palette_size_y = 0;
        info.palette_size_uv = 0;
        self.read_palette_mode_info(dec, r, c, bsize, info)?;
        self.filter_intra_mode_info(dec, info, bsize)?;
        Ok(())
    }

    /// `inter_frame_mode_info( )`（规范 §5.11.4）
    fn inter_frame_mode_info(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        mi_grid: &MiGrid,
        r: i32,
        c: i32,
        bsize: u8,
        info: &mut BlockInfo,
    ) -> Result<()> {
        info.use_intrabc = false;
        info.skip = false;
        let p_seg0 = dec.consumed_bits();
        self.inter_segment_id(dec, r, c, bsize, true, false)?;
        info.segment_id = self.segment_id;
        let p_seg1 = dec.consumed_bits();
        self.read_skip_mode(dec, r, c, info)?;
        let p_skm = dec.consumed_bits();
        if info.skip_mode {
            info.skip = true;
        } else {
            dec.label = "skip";
        info.skip = self.read_skip(dec, r, c, info.segment_id)?;
        }
        let p_sk = dec.consumed_bits();
        if !self.fh.params.segmentation.seg_id_pre_skip {
            self.inter_segment_id(dec, r, c, bsize, false, info.skip)?;
            info.segment_id = self.segment_id;
        }
        let t_a = dec.consumed_bits();
        self.read_cdef(dec, r, c, bsize, info.skip)?;
        let t_b = dec.consumed_bits();
        self.read_delta_qindex(dec, r, c, bsize, info.skip)?;
        let t_c = dec.consumed_bits();
        self.read_delta_lf(dec, r, c, bsize, info.skip)?;
        let t_d = dec.consumed_bits();
        self.read_deltas = false;
        info.is_inter = self.read_is_inter(dec, info.segment_id, info.skip_mode)?;
        let t_e = dec.consumed_bits();
        if info.is_inter {
            self.inter_block_mode_info(dec, mi_grid, r, c, bsize, info)?;
        } else {
            self.intra_block_mode_info(dec, r, c, bsize, info)?;
        }
        let t_f = dec.consumed_bits();
        trace_inter_mode(
            r,
            c,
            bsize,
            info,
            [
                p_seg1 - p_seg0,
                p_skm - p_seg1,
                p_sk - p_skm,
                t_a - p_sk,
                t_b - t_a,
                t_c - t_b,
                t_d - t_c,
                t_e - t_d,
                t_f - t_e,
            ],
        );
        Ok(())
    }

    /// `inter_segment_id( preSkip )`（规范 §5.11.5）
    fn inter_segment_id(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        r: i32,
        c: i32,
        bsize: u8,
        pre_skip: bool,
        skip: bool,
    ) -> Result<()> {
        let seg_enabled = self.fh.params.segmentation.enabled;
        let seg_update_map = self.fh.params.segmentation.update_map;
        let seg_pre_skip = self.fh.params.segmentation.seg_id_pre_skip;
        let seg_temporal_update = self.fh.params.segmentation.temporal_update;
        if !seg_enabled {
            self.segment_id = 0;
            return Ok(());
        }
        let bw4 = NUM_4X4_BLOCKS_WIDE[bsize as usize] as i32;
        let bh4 = NUM_4X4_BLOCKS_HIGH[bsize as usize] as i32;
        let predicted = self.get_segment_id(r, c, bsize);
        if !seg_update_map {
            self.segment_id = predicted as u8;
            return Ok(());
        }
        if pre_skip && !seg_pre_skip {
            self.segment_id = 0;
            return Ok(());
        }
        if !pre_skip && skip {
            self.grid.set_seg_pred_ctx(r, c, bw4, bh4, 0);
            self.segment_id = self.read_segment_id(dec, r, c, skip)?;
            return Ok(());
        }
        if seg_temporal_update {
            let ctx = self.grid.seg_pred_ctx(r, c);
            let s = read_sym(dec, &mut self.stats, &mut self.cdfs.segment_id_predicted[ctx])?;
            let predicted_flag = s as u8;
            if predicted_flag == 1 {
                self.segment_id = predicted as u8;
            } else {
                self.segment_id = self.read_segment_id(dec, r, c, skip)?;
            }
            self.grid.set_seg_pred_ctx(r, c, bw4, bh4, predicted_flag);
        } else {
            self.segment_id = self.read_segment_id(dec, r, c, skip)?;
        }
        Ok(())
    }

    /// `get_segment_id( )`
    fn get_segment_id(&self, r: i32, c: i32, bsize: u8) -> i32 {
        let bw4 = NUM_4X4_BLOCKS_WIDE[bsize as usize] as i32;
        let bh4 = NUM_4X4_BLOCKS_HIGH[bsize as usize] as i32;
        let x_mis = (self.fh.mi_cols as i32 - c).min(bw4);
        let y_mis = (self.fh.mi_rows as i32 - r).min(bh4);
        let mut seg = 7i32;
        for y in 0..y_mis {
            for x in 0..x_mis {
                let i = self.grid.idx(r + y, c + x);
                seg = seg.min(self.grid.prev_seg_ids[i] as i32);
            }
        }
        seg
    }

    fn intra_segment_id(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        r: i32,
        c: i32,
        skip: bool,
    ) -> Result<u8> {
        if self.fh.params.segmentation.enabled {
            self.read_segment_id(dec, r, c, skip)
        } else {
            Ok(0)
        }
    }

    /// `read_segment_id( )`（规范 §5.11.9）
    fn read_segment_id(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        r: i32,
        c: i32,
        skip: bool,
    ) -> Result<u8> {
        let avail_u = self.is_inside(r - 1, c);
        let avail_l = self.is_inside(r, c - 1);
        let get = |g: &BlockGrid, rr: i32, cc: i32| -> i32 { g.seg_ids[g.idx(rr, cc)] as i32 };
        let prev_ul = if avail_u && avail_l { get(self.grid, r - 1, c - 1) } else { -1 };
        let prev_u = if avail_u { get(self.grid, r - 1, c) } else { -1 };
        let prev_l = if avail_l { get(self.grid, r, c - 1) } else { -1 };
        let pred = if prev_u == -1 {
            if prev_l == -1 {
                0
            } else {
                prev_l
            }
        } else if prev_l == -1 {
            prev_u
        } else if prev_ul == prev_u {
            prev_u
        } else {
            prev_l
        };
        if skip {
            return Ok(pred.clamp(0, 7) as u8);
        }
        let ctx = if prev_ul < 0 {
            0
        } else if prev_ul == prev_u && prev_ul == prev_l {
            2
        } else if prev_ul == prev_u || prev_ul == prev_l || prev_u == prev_l {
            1
        } else {
            0
        };
        let last_active = self.fh.params.segmentation.last_active_seg_id as i32;
        let diff = read_sym(dec, &mut self.stats, &mut self.cdfs.segment_id[ctx])? as i32;
        Ok(neg_deinterleave(diff, pred, last_active + 1).clamp(0, 7) as u8)
    }

    /// `read_skip( )`
    fn read_skip(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        r: i32,
        c: i32,
        segment_id: u8,
    ) -> Result<bool> {
        if self.fh.params.segmentation.seg_id_pre_skip
            && self
                .fh
                .params
                .segmentation
                .feature_active(segment_id as usize, SEG_LVL_SKIP)
        {
            return Ok(true);
        }
        let mut ctx = 0usize;
        if self.is_inside(r - 1, c) {
            ctx += self.grid.skips[self.grid.idx(r - 1, c)] as usize;
        }
        if self.is_inside(r, c - 1) {
            ctx += self.grid.skips[self.grid.idx(r, c - 1)] as usize;
        }
        Ok(read_sym(dec, &mut self.stats, &mut self.cdfs.skip[ctx.min(2)])? == 1)
    }

    /// `read_skip_mode( )`
    fn read_skip_mode(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        r: i32,
        c: i32,
        info: &mut BlockInfo,
    ) -> Result<()> {
        info.skip_mode = false;
        // 规范 `read_skip_mode()`：**六个**条件里任一成立就 `skip_mode = 0`、**一个符号都不读**。
        // 旧实现只判了 `!skip_mode_present` —— 段级特征或"块太窄/太矮"时我们会**多读一个符号**，
        // 从那里起整条算术流错位。
        // 症状极隐蔽：前两帧逐符号全对，第 3 帧突然在 `skip_mode` 的 CDF 上"符号相同、CDF 不同"
        // （实际上是两边已经错开一格）。
        let seg = &self.fh.params.segmentation;
        let sid = info.segment_id as usize;
        let bsize = self.cur_bsize;
        if seg.feature_active(sid, SEG_LVL_SKIP)
            || seg.feature_active(sid, SEG_LVL_REF_FRAME)
            || seg.feature_active(sid, SEG_LVL_GLOBALMV)
            || !self.fh.params.skip_mode_present
            || block_width(bsize) < 8
            || block_height(bsize) < 8
        {
            return Ok(());
        }
        let mut ctx = 0usize;
        if self.is_inside(r - 1, c) {
            ctx += self.grid.skip_modes[self.grid.idx(r - 1, c)] as usize;
        }
        if self.is_inside(r, c - 1) {
            ctx += self.grid.skip_modes[self.grid.idx(r, c - 1)] as usize;
        }
        info.skip_mode = read_sym(dec, &mut self.stats, &mut self.cdfs.skip_mode[ctx.min(2)])? == 1;
        Ok(())
    }

    /// `read_is_inter( )`
    fn read_is_inter(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        segment_id: u8,
        skip_mode: bool,
    ) -> Result<bool> {
        if skip_mode {
            return Ok(true);
        }
        let has_ref = self
            .fh
            .params
            .segmentation
            .feature_active(segment_id as usize, SEG_LVL_REF_FRAME);
        if has_ref {
            let v = self.fh.params.segmentation.feature_data[segment_id as usize][SEG_LVL_REF_FRAME];
            return Ok(v != mv::INTRA_FRAME as i32);
        }
        if self
            .fh
            .params
            .segmentation
            .feature_active(segment_id as usize, SEG_LVL_GLOBALMV)
        {
            return Ok(true);
        }
        let (r, c) = (self.cur_row, self.cur_col);
        let avail_u = self.is_inside(r - 1, c);
        let avail_l = self.is_inside(r, c - 1);
        let above_intra = !avail_u || self.grid.is_inters[self.grid.idx(r - 1, c)] == 0;
        let left_intra = !avail_l || self.grid.is_inters[self.grid.idx(r, c - 1)] == 0;
        let ctx = if avail_u && avail_l {
            if left_intra && above_intra {
                3
            } else {
                usize::from(left_intra || above_intra)
            }
        } else if avail_u || avail_l {
            2 * usize::from(if avail_u { above_intra } else { left_intra })
        } else {
            0
        };
        Ok(read_sym(dec, &mut self.stats, &mut self.cdfs.is_inter[ctx.min(3)])? == 1)
    }

    /// `read_cdef( )`
    fn read_cdef(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        r: i32,
        c: i32,
        bsize: u8,
        skip: bool,
    ) -> Result<()> {
        if skip || self.fh.params.coded_lossless || !self.seq.enable_cdef || self.fh.allow_intrabc {
            return Ok(());
        }
        let cdef_size4 = NUM_4X4_BLOCKS_WIDE[BLOCK_64X64] as i32;
        let mask = !(cdef_size4 - 1);
        let (cr, cc) = (r & mask, c & mask);
        if cr < 0 || cc < 0 || cr as usize >= self.grid.rows() || cc as usize >= self.grid.cols() {
            return Ok(());
        }
        let idx = cr as usize * self.grid.cols() + cc as usize;
        if self.grid.cdef_idx[idx] == -1 {
            let bits = self.fh.params.cdef.bits as u32;
            dec.label = "var_tx_depth";
            let value = dec.read_literal(bits)? as i32;
            self.stats.symbols += 1;
            let w4 = NUM_4X4_BLOCKS_WIDE[bsize as usize] as i32;
            let h4 = NUM_4X4_BLOCKS_HIGH[bsize as usize] as i32;
            let rows = self.grid.rows();
            let cols = self.grid.cols();
            let mut i = cr;
            while i < cr + h4 {
                let mut j = cc;
                while j < cc + w4 {
                    if i >= 0 && j >= 0 && (i as usize) < rows && (j as usize) < cols {
                        self.grid.cdef_idx[i as usize * cols + j as usize] = value;
                    }
                    j += cdef_size4;
                }
                i += cdef_size4;
            }
        }
        Ok(())
    }

    fn read_deltas(&self) -> bool {
        self.read_deltas
    }

    /// `read_delta_qindex( )`
    fn read_delta_qindex(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        _r: i32,
        _c: i32,
        bsize: u8,
        skip: bool,
    ) -> Result<()> {
        let sb_size = if self.seq.use_128x128_superblock {
            BLOCK_128X128 as u8
        } else {
            BLOCK_64X64 as u8
        };
        if bsize == sb_size && skip {
            return Ok(());
        }
        if !self.read_deltas() {
            return Ok(());
        }
        let mut delta_q_abs = read_sym(dec, &mut self.stats, &mut self.cdfs.delta_q)?;
        if delta_q_abs == DELTA_Q_SMALL {
            let rem = dec.read_literal(3)? + 1;
            let bits = dec.read_literal(rem)?;
            self.stats.symbols += 2;
            delta_q_abs = bits + (1 << rem) + 1;
        }
        if delta_q_abs != 0 {
            let sign = dec.read_literal(1)?;
            self.stats.symbols += 1;
            let reduced = if sign == 1 { -(delta_q_abs as i32) } else { delta_q_abs as i32 };
            let res = self.fh.params.delta_q.res as i32;
            self.current_q_index = (self.current_q_index + (reduced << res)).clamp(1, 255);
            self.stats.delta_q_nonzero += 1;
        }
        Ok(())
    }

    /// `read_delta_lf( )`
    fn read_delta_lf(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        _r: i32,
        _c: i32,
        bsize: u8,
        skip: bool,
    ) -> Result<()> {
        let sb_size = if self.seq.use_128x128_superblock {
            BLOCK_128X128 as u8
        } else {
            BLOCK_64X64 as u8
        };
        if bsize == sb_size && skip {
            return Ok(());
        }
        let lf = self.fh.params.delta_lf;
        if !self.read_deltas() || !lf.present {
            return Ok(());
        }
        let count = if lf.multi {
            if self.num_planes() > 1 {
                FRAME_LF_COUNT
            } else {
                FRAME_LF_COUNT - 2
            }
        } else {
            1
        };
        for i in 0..count {
            let abs = read_sym(dec, &mut self.stats, &mut self.cdfs.delta_lf)?;
            let abs = if abs == DELTA_LF_SMALL {
                let rem = dec.read_literal(3)?;
                let n = rem + 1;
                let bits = dec.read_literal(n)?;
                self.stats.symbols += 2;
                bits + (1 << n) + 1
            } else {
                abs
            };
            if abs != 0 {
                let sign = dec.read_literal(1)?;
                self.stats.symbols += 1;
                let v = if sign == 1 { -(abs as i32) } else { abs as i32 };
                let res = lf.res as i32;
                self.delta_lfs[i] =
                    (self.delta_lfs[i] + (v << res)).clamp(-MAX_LOOP_FILTER, MAX_LOOP_FILTER);
            }
        }
        Ok(())
    }

    fn neighbour_intra_context(&self, r: i32, c: i32) -> (usize, usize) {
        let above = if self.is_inside(r - 1, c) {
            self.grid.y_modes[self.grid.idx(r - 1, c)]
        } else {
            DC_PRED
        };
        let left = if self.is_inside(r, c - 1) {
            self.grid.y_modes[self.grid.idx(r, c - 1)]
        } else {
            DC_PRED
        };
        (
            INTRA_MODE_CONTEXT[(above as usize) % INTRA_MODES as usize],
            INTRA_MODE_CONTEXT[(left as usize) % INTRA_MODES as usize],
        )
    }

    /// `intra_angle_info_y( )` / `intra_angle_info_uv( )`
    fn intra_angle_info(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        mode: u8,
        bsize: u8,
    ) -> Result<i8> {
        if bsize < BLOCK_8X8 as u8 || !is_directional_mode(mode) {
            return Ok(0);
        }
        dec.label = "angle_delta";
        let raw = read_sym(
            dec,
            &mut self.stats,
            &mut self.cdfs.angle_delta[(mode - V_PRED) as usize],
        )?;
        Ok(raw as i8 - MAX_ANGLE_DELTA)
    }

    /// `uv_mode`
    fn read_uv_mode(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        y_mode: u8,
        bsize: u8,
        lossless: bool,
    ) -> Result<u8> {
        dec.label = "uv_mode";
        // §8.3.2：无损时要看**色度残差块**是不是 4×4（不是"亮度块是 4×4"！）
        // —— 4:2:0 下亮度 8×8 的色度正是 4×4，所以 8×8 的块在无损流里也该走
        // `uv_mode_cfl_allowed` 那套 CDF。这条搞错只在无损流里显形（有损的判据
        // 是"最大边 ≤ 32"，与它无关），症状是 CFL 流整条位流错位。
        let cfl_allowed = if lossless {
            super::coeff::plane_residual_size(bsize, 1, self.sub_x(), self.sub_y()) == BLOCK_4X4 as u8
        } else {
            block_width(bsize).max(block_height(bsize)) <= 32
        };
        let key = (y_mode as usize) % INTRA_MODES as usize;
        if cfl_allowed {
            let mut c = self.cdfs.uv_mode_cfl_allowed[key];
            let s = read_sym(dec, &mut self.stats, &mut c)?;
            self.cdfs.uv_mode_cfl_allowed[key] = c;
            Ok(s as u8)
        } else {
            let mut c = self.cdfs.uv_mode_cfl_not_allowed[key];
            let s = read_sym(dec, &mut self.stats, &mut c)?;
            self.cdfs.uv_mode_cfl_not_allowed[key] = c;
            Ok(s as u8)
        }
    }

    /// `read_cfl_alphas( )`
    fn read_cfl_alphas(&mut self, dec: &mut SymbolDecoder<'_>) -> Result<(i8, i8)> {
        let signs = read_sym(dec, &mut self.stats, &mut self.cdfs.cfl_sign)? as i32;
        let sign_u = (signs + 1) / 3;
        let sign_v = (signs + 1) % 3;
        let ctx_u = if sign_u == CFL_SIGN_ZERO {
            None
        } else {
            Some(((sign_u - 1) * 3 + sign_v) as usize)
        };
        let ctx_v = if sign_v == CFL_SIGN_ZERO {
            None
        } else {
            Some(((sign_v - 1) * 3 + sign_u) as usize)
        };
        let mut alpha_u = 0i8;
        let mut alpha_v = 0i8;
        if let Some(ctx) = ctx_u {
            let mut c = self.cdfs.cfl_alpha[ctx];
            let s = read_sym(dec, &mut self.stats, &mut c)? as i8;
            self.cdfs.cfl_alpha[ctx] = c;
            alpha_u = 1 + s;
            if sign_u == CFL_SIGN_NEG {
                alpha_u = -alpha_u;
            }
        }
        if let Some(ctx) = ctx_v {
            let mut c = self.cdfs.cfl_alpha[ctx];
            let s = read_sym(dec, &mut self.stats, &mut c)? as i8;
            self.cdfs.cfl_alpha[ctx] = c;
            alpha_v = 1 + s;
            if sign_v == CFL_SIGN_NEG {
                alpha_v = -alpha_v;
            }
        }
        Ok((alpha_u, alpha_v))
    }

    /// `filter_intra_mode_info( )`
    fn filter_intra_mode_info(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        info: &mut BlockInfo,
        bsize: u8,
    ) -> Result<()> {
        info.use_filter_intra = false;
        let w = block_width(bsize);
        let h = block_height(bsize);
        if !self.seq.enable_filter_intra
            || info.y_mode != DC_PRED
            || info.palette_size_y != 0
            || w.max(h) > 32
        {
            return Ok(());
        }
        let s = read_sym(dec, &mut self.stats, &mut self.cdfs.filter_intra[bsize as usize])?;
        info.use_filter_intra = s == 1;
        if info.use_filter_intra {
            info.filter_intra_mode =
                read_sym(dec, &mut self.stats, &mut self.cdfs.filter_intra_mode)? as u8;
            self.stats.filter_intra_blocks += 1;
        }
        Ok(())
    }

    /// `get_palette_cache( plane )`（规范 §7.11.4）：把上/左邻居的调色板合并成
    /// 一份**升序去重**的缓存。
    ///
    /// 两份来源本身都是升序（颜色表读完就排过序），所以规范提示可以用**归并**
    /// 代替排序 —— 这里就是照归并写的。
    ///
    /// 上面那份只在 `( MiRow * MI_SIZE ) % 64` 非零时可用：超块行边界处不跨行取，
    /// 否则会读到"上一个超块行"的调色板（那些格子与本块不在同一个超块行里）。
    fn get_palette_cache(&self, plane: usize, r: i32, c: i32) -> Vec<u16> {
        let above_n = if (r * MI_SIZE) % 64 != 0 {
            self.grid.palette_sizes[self.grid.idx(r - 1, c)][plane] as usize
        } else {
            0
        };
        let left_n = if self.is_inside(r, c - 1) {
            self.grid.palette_sizes[self.grid.idx(r, c - 1)][plane] as usize
        } else {
            0
        };
        // `idx` 在越界时返回 0，但下面两个 `_n` 为 0 时不会去碰这份数据。
        let above = &self.grid.palette_colors[self.grid.idx(r - 1, c)][plane];
        let left = &self.grid.palette_colors[self.grid.idx(r, c - 1)][plane];
        let (mut ai, mut li) = (0usize, 0usize);
        let mut out: Vec<u16> = Vec::with_capacity(above_n + left_n);
        while ai < above_n && li < left_n {
            let (ac, lc) = (above[ai], left[li]);
            if lc < ac {
                if out.last() != Some(&lc) {
                    out.push(lc);
                }
                li += 1;
            } else {
                if out.last() != Some(&ac) {
                    out.push(ac);
                }
                ai += 1;
                if lc == ac {
                    li += 1;
                }
            }
        }
        while ai < above_n {
            let v = above[ai];
            ai += 1;
            if out.last() != Some(&v) {
                out.push(v);
            }
        }
        while li < left_n {
            let v = left[li];
            li += 1;
            if out.last() != Some(&v) {
                out.push(v);
            }
        }
        out
    }

    /// 读一张调色板的**颜色表**（`palette_mode_info` 里 Y 与 U 共用的那一段）。
    ///
    /// 三步：先按 `use_palette_color_cache_* L(1)` 逐条吃缓存；再用一个
    /// `L(BitDepth)` 的全位深字面量打底；剩下的按 `paletteBits` 变长增量累加，
    /// 且**每加一个就把 `range` 收窄一次**（越靠后越省位）。最后升序排序。
    ///
    /// `luma` —— **规范在 Y 与 U 上有两处不对称，必须照抄**（§5.11.18 原文）：
    ///
    /// | | Y | U |
    /// |---|---|---|
    /// | 增量 | `palette_delta_y++; colors = prev + delta` | `colors = prev + delta`（**不加 1**） |
    /// | `range` | `(1 << BitDepth) - color - 1` | `(1 << BitDepth) - color` |
    ///
    /// 增量那个 `++` 少抄一处，整块调色板的颜色就**整体偏高 1**：亮度可能因为
    /// 后续残差把它"吃掉"而看不出来，色度则会露出一片 `+1` 的差异。
    fn read_palette_colors(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        cache: &[u16],
        size: usize,
        bit_depth: u8,
        luma: bool,
        out: &mut [u16; PALETTE_COLORS],
    ) -> Result<()> {
        let mut idx = 0usize;
        for &entry in cache.iter() {
            if idx >= size {
                break;
            }
            if dec.read_bool()? {
                out[idx] = entry;
                idx += 1;
            }
        }
        if idx < size {
            out[idx] = dec.read_literal(bit_depth as u32)? as u16;
            idx += 1;
        }
        let mut palette_bits = 0u32;
        if idx < size {
            palette_bits = (bit_depth as u32 - 3) + dec.read_literal(2)?;
        }
        let max = 1i32 << bit_depth;
        while idx < size {
            let mut delta = dec.read_literal(palette_bits)? as i32;
            if luma {
                delta += 1;
            }
            out[idx] = super::intra::clip1(out[idx - 1] as i32 + delta, bit_depth);
            let mut range = max - out[idx] as i32;
            if luma {
                range -= 1;
            }
            palette_bits = palette_bits.min(ceil_log2(range));
            idx += 1;
        }
        out[..size].sort();
        Ok(())
    }

    /// `palette_mode_info( )`（规范 §5.11.18）—— 屏幕内容工具：读颜色表。
    ///
    /// **标志位必须照读**：`allow_screen_content_tools = 1` 的帧里每个 ≥8×8 的
    /// 帧内块都要读这两个 `S()`，少读一位后面全体错位（真实流里 libaom 默认就
    /// 把屏幕内容工具开着，哪怕根本没用 palette）。
    ///
    /// 颜色表本身只有"增量编码"这一条路：`palette_size_*` 定长度，随后
    /// 缓存 + 一个字面量 + 变长增量。**索引图不在这里读**，它归
    /// [`Self::palette_tokens`] —— 规范也是分成两个过程调用的。
    fn read_palette_mode_info(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        r: i32,
        c: i32,
        bsize: u8,
        info: &mut BlockInfo,
    ) -> Result<()> {
        let w = block_width(bsize);
        let h = block_height(bsize);
        if bsize < BLOCK_8X8 as u8 || w > 64 || h > 64 || !self.fh.allow_screen_content_tools {
            return Ok(());
        }
        let bit_depth = self.seq.color.bit_depth;
        let bsize_ctx = (MI_WIDTH_LOG2[bsize as usize] as usize
            + MI_HEIGHT_LOG2[bsize as usize] as usize)
            .saturating_sub(2)
            .min(6);
        if info.y_mode == DC_PRED {
            let mut ctx = 0usize;
            if self.is_inside(r - 1, c) && self.grid.palette_sizes[self.grid.idx(r - 1, c)][0] > 0 {
                ctx += 1;
            }
            if self.is_inside(r, c - 1) && self.grid.palette_sizes[self.grid.idx(r, c - 1)][0] > 0 {
                ctx += 1;
            }
            dec.label = "has_palette_y";
            let has = read_sym(dec, &mut self.stats, &mut self.cdfs.palette_y_mode[bsize_ctx][ctx])?;
            if has == 1 {
                dec.label = "palette_size_y";
                let size =
                    read_sym(dec, &mut self.stats, &mut self.cdfs.palette_y_size[bsize_ctx])? as usize
                        + 2;
                info.palette_size_y = size as u8;
                let cache = self.get_palette_cache(0, r, c);
                let mut colors = [0u16; PALETTE_COLORS];
                self.read_palette_colors(dec, &cache, size, bit_depth, true, &mut colors)?;
                self.palette_colors_y = colors;
                self.stats.palette_y_blocks += 1;
            }
        }
        if self.has_chroma(r, c, bsize) && info.uv_mode == DC_PRED {
            let ctx = usize::from(info.palette_size_y > 0);
            dec.label = "has_palette_uv";
            let has = read_sym(dec, &mut self.stats, &mut self.cdfs.palette_uv_mode[ctx])?;
            if has == 1 {
                dec.label = "palette_size_uv";
                let size =
                    read_sym(dec, &mut self.stats, &mut self.cdfs.palette_uv_size[bsize_ctx])? as usize
                        + 2;
                info.palette_size_uv = size as u8;
                let cache = self.get_palette_cache(1, r, c);
                let mut colors_u = [0u16; PALETTE_COLORS];
                self.read_palette_colors(dec, &cache, size, bit_depth, false, &mut colors_u)?;
                self.palette_colors_u = colors_u;
                // V 分量单独编码：要么逐个全位深字面量，要么"增量 + 符号位 + 环形回绕"。
                let mut colors_v = [0u16; PALETTE_COLORS];
                if dec.read_bool()? {
                    let max_val = 1i32 << bit_depth;
                    // 规范在这个分支里**不再**收窄 paletteBits（只有 U 那支才收窄）。
                    let v_bits = (bit_depth as u32 - 4) + dec.read_literal(2)?;
                    colors_v[0] = dec.read_literal(bit_depth as u32)? as u16;
                    for idx in 1..size {
                        let mut delta = dec.read_literal(v_bits)? as i32;
                        if delta != 0 && dec.read_bool()? {
                            delta = -delta;
                        }
                        let mut val = colors_v[idx - 1] as i32 + delta;
                        if val < 0 {
                            val += max_val;
                        }
                        if val >= max_val {
                            val -= max_val;
                        }
                        colors_v[idx] = super::intra::clip1(val, bit_depth);
                    }
                } else {
                    for idx in 0..size {
                        colors_v[idx] = dec.read_literal(bit_depth as u32)? as u16;
                    }
                }
                self.palette_colors_v = colors_v;
                self.stats.palette_uv_blocks += 1;
            }
        }
        Ok(())
    }

    /// `get_palette_color_context( colorMap, r, c, n )`（规范 §7.11.4）。
    ///
    /// 数的是**已经解码出来的**左 / 左上 / 上三个邻居的颜色出现次数（左 2 分、
    /// 左上 1 分、上 2 分），再把分数最高的前三名换到 `ColorOrder` 前面 ——
    /// 于是"当前样本最可能是哪个颜色"就排在最前，熵编码只要几个位。
    ///
    /// 返回 `(ctx, ColorOrder)`：`ctx` 是给 CDF 用的上下文
    /// （`Palette_Color_Context[ ColorContextHash ]`）。
    fn palette_color_context(
        map: &[u8],
        stride: usize,
        r: usize,
        c: usize,
        n: usize,
    ) -> (usize, [u8; PALETTE_COLORS]) {
        let mut scores = [0i32; PALETTE_COLORS];
        let mut order = [0u8, 1, 2, 3, 4, 5, 6, 7];
        if c > 0 {
            scores[map[r * stride + c - 1] as usize] += 2;
        }
        if r > 0 && c > 0 {
            scores[map[(r - 1) * stride + c - 1] as usize] += 1;
        }
        if r > 0 {
            scores[map[(r - 1) * stride + c] as usize] += 2;
        }
        // 稳定地只把前 `PALETTE_NUM_NEIGHBORS` 名挑出来（规范就是这种选择排序）
        for i in 0..PALETTE_NUM_NEIGHBORS {
            let mut max_score = scores[i];
            let mut max_idx = i;
            for j in i + 1..n {
                if scores[j] > max_score {
                    max_score = scores[j];
                    max_idx = j;
                }
            }
            if max_idx != i {
                let max_order = order[max_idx];
                let mut k = max_idx;
                while k > i {
                    scores[k] = scores[k - 1];
                    order[k] = order[k - 1];
                    k -= 1;
                }
                scores[i] = max_score;
                order[i] = max_order;
            }
        }
        let mut hash = 0i32;
        for i in 0..PALETTE_NUM_NEIGHBORS {
            hash += scores[i] * PALETTE_COLOR_HASH_MULTIPLIERS[i];
        }
        let ctx = PALETTE_COLOR_CONTEXT[hash as usize];
        debug_assert!(ctx >= 0, "Palette_Color_Context 的负值不该被访问");
        (ctx as usize, order)
    }

    /// 解一张颜色索引图（`palette_tokens` 里 Y / UV 共用的那一段）。
    ///
    /// 必须**沿反对角线**走：每个样本的上下文取自它已解码的左/左上/上邻居，
    /// 顺序一错就等于读了"还没解出来的样本"。走完之后，帧外区域按最右列 /
    /// 最下行复制填满（规范要求整块都有值，预测阶段会照块尺寸去取）。
    fn palette_color_map(
        dec: &mut SymbolDecoder<'_>,
        stats: &mut BlockStats,
        map: &mut [u8],
        block_w: usize,
        block_h: usize,
        onscreen_w: usize,
        onscreen_h: usize,
        size: usize,
        cdf: &mut [[[u16; 9]; 5]; 7],
    ) -> Result<()> {
        if onscreen_w == 0 || onscreen_h == 0 {
            return Ok(());
        }
        for i in 1..onscreen_h + onscreen_w - 1 {
            let j_from = i.min(onscreen_w - 1);
            let j_to = i.saturating_sub(onscreen_h - 1);
            // `j` 是 usize 且 `jTo` 可以是 0，所以必须用**闭区间倒序**而不是
            // `while j >= jTo { …; j -= 1 }` —— 后者会在 j=0 那一轮减溢出。
            for j in (j_to..=j_from).rev() {
                let (rr, cc) = (i - j, j);
                let (ctx, order) = Self::palette_color_context(map, block_w, rr, cc, size);
                dec.label = "palette_color_idx";
                let cdf_slice = &mut cdf[size - 2][ctx][..size + 1];
                let sym = read_sym(dec, stats, cdf_slice)? as usize;
                map[rr * block_w + cc] = order[sym];
            }
        }
        for i in 0..onscreen_h {
            for j in onscreen_w..block_w {
                map[i * block_w + j] = map[i * block_w + onscreen_w - 1];
            }
        }
        for i in onscreen_h..block_h {
            for j in 0..block_w {
                map[i * block_w + j] = map[(onscreen_h - 1) * block_w + j];
            }
        }
        Ok(())
    }

    /// `palette_tokens( )`（规范 §5.11.19）—— 屏幕内容工具：读颜色索引图。
    ///
    /// 位置很关键：`decode_block` 里排在 `mode_info()` **之后**、
    /// `read_block_tx_size()` 之前。放在别处（比如残差之后）就是整条算术流错位。
    fn palette_tokens(&mut self, dec: &mut SymbolDecoder<'_>, info: &BlockInfo) -> Result<()> {
        if info.palette_size_y == 0 && info.palette_size_uv == 0 {
            return Ok(());
        }
        let r = info.mi_row as i32;
        let c = info.mi_col as i32;
        let block_h = block_height(info.mi_size) as usize;
        let block_w = block_width(info.mi_size) as usize;
        // 帧外部分不进算术流（`onscreen*`），解完再靠复制补齐。
        let onscreen_h = block_h.min((self.fh.mi_rows as i32 - r).max(0) as usize * 4);
        let onscreen_w = block_w.min((self.fh.mi_cols as i32 - c).max(0) as usize * 4);
        if info.palette_size_y > 0 {
            let size = info.palette_size_y as usize;
            dec.label = "color_index_map_y";
            self.color_map_y[0] = dec.read_ns(size as u32)? as u8;
            Self::palette_color_map(
                dec,
                &mut self.stats,
                &mut self.color_map_y,
                block_w,
                block_h,
                onscreen_w,
                onscreen_h,
                size,
                &mut self.cdfs.palette_color_y,
            )?;
        }
        if info.palette_size_uv > 0 {
            let size = info.palette_size_uv as usize;
            dec.label = "color_index_map_uv";
            self.color_map_uv[0] = dec.read_ns(size as u32)? as u8;
            let (sub_x, sub_y) = (usize::from(self.sub_x()), usize::from(self.sub_y()));
            let (mut bw, mut bh) = (block_w >> sub_x, block_h >> sub_y);
            let (mut ow, mut oh) = (onscreen_w >> sub_x, onscreen_h >> sub_y);
            // 规范：色度块小于 4 时补到 4（4:2:0 下的 4 宽块会缩到 2）。
            if bw < 4 {
                bw += 2;
                ow += 2;
            }
            if bh < 4 {
                bh += 2;
                oh += 2;
            }
            Self::palette_color_map(
                dec,
                &mut self.stats,
                &mut self.color_map_uv,
                bw,
                bh,
                ow,
                oh,
                size,
                &mut self.cdfs.palette_color_uv,
            )?;
        }
        Ok(())
    }

    /// `read_ref_frames( )`（规范 §5.11.17）
    fn read_ref_frames(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        r: i32,
        c: i32,
        segment_id: u8,
        skip_mode: bool,
        info: &mut BlockInfo,
    ) -> Result<()> {
        if skip_mode {
            info.ref_frame = [
                self.fh.params.skip_mode_frame[0] as i8,
                self.fh.params.skip_mode_frame[1] as i8,
            ];
            return Ok(());
        }
        if self
            .fh
            .params
            .segmentation
            .feature_active(segment_id as usize, SEG_LVL_REF_FRAME)
        {
            let v = self.fh.params.segmentation.feature_data[segment_id as usize][SEG_LVL_REF_FRAME];
            info.ref_frame = [v as i8, NONE];
            return Ok(());
        }
        if self
            .fh
            .params
            .segmentation
            .feature_active(segment_id as usize, SEG_LVL_SKIP)
            || self
                .fh
                .params
                .segmentation
                .feature_active(segment_id as usize, SEG_LVL_GLOBALMV)
        {
            info.ref_frame = [mv::LAST_FRAME, NONE];
            return Ok(());
        }
        let avail_u = self.is_inside(r - 1, c);
        let avail_l = self.is_inside(r, c - 1);
        let above = if avail_u {
            self.grid.ref_frames[self.grid.idx(r - 1, c)]
        } else {
            [mv::INTRA_FRAME, NONE]
        };
        let left = if avail_l {
            self.grid.ref_frames[self.grid.idx(r, c - 1)]
        } else {
            [mv::INTRA_FRAME, NONE]
        };
        let above_intra = above[0] <= mv::INTRA_FRAME;
        let left_intra = left[0] <= mv::INTRA_FRAME;
        let above_single = above[1] <= mv::INTRA_FRAME;
        let left_single = left[1] <= mv::INTRA_FRAME;

        let bw4 = NUM_4X4_BLOCKS_WIDE[info.mi_size as usize] as i32;
        let bh4 = NUM_4X4_BLOCKS_HIGH[info.mi_size as usize] as i32;
        let comp_mode = if self.fh.params.reference_select && bw4.min(bh4) >= 2 {
            let ctx = comp_mode_ctx(
                avail_u, avail_l, above[0], left[0], above_intra, left_intra, above_single, left_single,
            );
            read_sym(dec, &mut self.stats, &mut self.cdfs.comp_mode[ctx.min(4)])? as u8
        } else {
            SINGLE_REFERENCE
        };
        let count_refs = |ref_frame: i8| -> i32 {
            let mut n = 0;
            if avail_u {
                if above[0] == ref_frame {
                    n += 1;
                }
                if above[1] == ref_frame {
                    n += 1;
                }
            }
            if avail_l {
                if left[0] == ref_frame {
                    n += 1;
                }
                if left[1] == ref_frame {
                    n += 1;
                }
            }
            n
        };
        if comp_mode == COMPOUND_REFERENCE {
            let ctx = comp_ref_type_ctx(
                avail_u, avail_l, above[0], above[1], left[0], left[1], above_intra, left_intra,
                above_single, left_single,
            );
            let comp_ref_type =
                read_sym(dec, &mut self.stats, &mut self.cdfs.comp_ref_type[ctx.min(4)])? as u8;
            if comp_ref_type == UNIDIR_COMP_REFERENCE {
                let uctx = ref_count_ctx(
                    count_refs(mv::LAST_FRAME)
                        + count_refs(mv::LAST2_FRAME)
                        + count_refs(mv::LAST3_FRAME)
                        + count_refs(mv::GOLDEN_FRAME),
                    count_refs(mv::BWDREF_FRAME)
                        + count_refs(mv::ALTREF2_FRAME)
                        + count_refs(mv::ALTREF_FRAME),
                );
                let u = read_sym(dec, &mut self.stats, &mut self.cdfs.uni_comp_ref[uctx.min(2)][0])?;
                if u == 1 {
                    info.ref_frame = [mv::BWDREF_FRAME, mv::ALTREF_FRAME];
                } else {
                    let p1 = ref_count_ctx(count_refs(mv::LAST2_FRAME), count_refs(mv::LAST3_FRAME) + count_refs(mv::GOLDEN_FRAME));
                    let a = read_sym(dec, &mut self.stats, &mut self.cdfs.uni_comp_ref[p1.min(2)][1])?;
                    if a == 1 {
                        let p2 = ref_count_ctx(count_refs(mv::LAST3_FRAME), count_refs(mv::GOLDEN_FRAME));
                        let b = read_sym(dec, &mut self.stats, &mut self.cdfs.uni_comp_ref[p2.min(2)][2])?;
                        if b == 1 {
                            info.ref_frame = [mv::LAST_FRAME, mv::GOLDEN_FRAME];
                        } else {
                            info.ref_frame = [mv::LAST_FRAME, mv::LAST3_FRAME];
                        }
                    } else {
                        info.ref_frame = [mv::LAST_FRAME, mv::LAST2_FRAME];
                    }
                }
            } else {
                let c0 = ref_count_ctx(
                    count_refs(mv::LAST_FRAME) + count_refs(mv::LAST2_FRAME),
                    count_refs(mv::LAST3_FRAME) + count_refs(mv::GOLDEN_FRAME),
                );
                let comp_ref = read_sym(dec, &mut self.stats, &mut self.cdfs.comp_ref[c0.min(2)][0])?;
                if comp_ref == 0 {
                    let c1 = ref_count_ctx(count_refs(mv::LAST_FRAME), count_refs(mv::LAST2_FRAME));
                    let p1 = read_sym(dec, &mut self.stats, &mut self.cdfs.comp_ref[c1.min(2)][1])?;
                    info.ref_frame[0] = if p1 == 1 { mv::LAST2_FRAME } else { mv::LAST_FRAME };
                } else {
                    let c2 = ref_count_ctx(count_refs(mv::LAST3_FRAME), count_refs(mv::GOLDEN_FRAME));
                    let p2 = read_sym(dec, &mut self.stats, &mut self.cdfs.comp_ref[c2.min(2)][2])?;
                    info.ref_frame[0] = if p2 == 1 { mv::GOLDEN_FRAME } else { mv::LAST3_FRAME };
                }
                let c3 = ref_count_ctx(
                    count_refs(mv::BWDREF_FRAME) + count_refs(mv::ALTREF2_FRAME),
                    count_refs(mv::ALTREF_FRAME),
                );
                let cb = read_sym(dec, &mut self.stats, &mut self.cdfs.comp_bwdref[c3.min(2)][0])?;
                if cb == 0 {
                    let c4 = ref_count_ctx(count_refs(mv::BWDREF_FRAME), count_refs(mv::ALTREF2_FRAME));
                    let p1 = read_sym(dec, &mut self.stats, &mut self.cdfs.comp_bwdref[c4.min(2)][1])?;
                    info.ref_frame[1] = if p1 == 1 { mv::ALTREF2_FRAME } else { mv::BWDREF_FRAME };
                } else {
                    info.ref_frame[1] = mv::ALTREF_FRAME;
                }
            }
        } else {
            let ctx = ref_count_ctx(
                count_refs(mv::LAST_FRAME)
                    + count_refs(mv::LAST2_FRAME)
                    + count_refs(mv::LAST3_FRAME)
                    + count_refs(mv::GOLDEN_FRAME),
                count_refs(mv::BWDREF_FRAME)
                    + count_refs(mv::ALTREF2_FRAME)
                    + count_refs(mv::ALTREF_FRAME),
            );
            let p1 = read_sym(dec, &mut self.stats, &mut self.cdfs.single_ref[ctx.min(2)][0])?;
            if p1 == 1 {
                let c1 = ref_count_ctx(
                    count_refs(mv::BWDREF_FRAME) + count_refs(mv::ALTREF2_FRAME),
                    count_refs(mv::ALTREF_FRAME),
                );
                let p2 = read_sym(dec, &mut self.stats, &mut self.cdfs.single_ref[c1.min(2)][1])?;
                if p2 == 0 {
                    let c2 = ref_count_ctx(count_refs(mv::BWDREF_FRAME), count_refs(mv::ALTREF2_FRAME));
                    let p6 = read_sym(dec, &mut self.stats, &mut self.cdfs.single_ref[c2.min(2)][5])?;
                    info.ref_frame[0] = if p6 == 1 { mv::ALTREF2_FRAME } else { mv::BWDREF_FRAME };
                } else {
                    info.ref_frame[0] = mv::ALTREF_FRAME;
                }
            } else {
                let c3 = ref_count_ctx(
                    count_refs(mv::LAST_FRAME) + count_refs(mv::LAST2_FRAME),
                    count_refs(mv::LAST3_FRAME) + count_refs(mv::GOLDEN_FRAME),
                );
                let p3 = read_sym(dec, &mut self.stats, &mut self.cdfs.single_ref[c3.min(2)][2])?;
                if p3 == 1 {
                    let c4 = ref_count_ctx(count_refs(mv::LAST3_FRAME), count_refs(mv::GOLDEN_FRAME));
                    let p5 = read_sym(dec, &mut self.stats, &mut self.cdfs.single_ref[c4.min(2)][4])?;
                    info.ref_frame[0] = if p5 == 1 { mv::GOLDEN_FRAME } else { mv::LAST3_FRAME };
                } else {
                    let c5 = ref_count_ctx(count_refs(mv::LAST_FRAME), count_refs(mv::LAST2_FRAME));
                    let p4 = read_sym(dec, &mut self.stats, &mut self.cdfs.single_ref[c5.min(2)][3])?;
                    info.ref_frame[0] = if p4 == 1 { mv::LAST2_FRAME } else { mv::LAST_FRAME };
                }
            }
            info.ref_frame[1] = NONE;
        }
        Ok(())
    }

    /// `inter_block_mode_info( )`（规范 §5.11.14）
    fn inter_block_mode_info(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        mi_grid: &MiGrid,
        r: i32,
        c: i32,
        bsize: u8,
        info: &mut BlockInfo,
    ) -> Result<()> {
        info.palette_size_y = 0;
        info.palette_size_uv = 0;
        let segment_id = info.segment_id;
        let skip_mode = info.skip_mode;
        let q_rf = dec.consumed_bits();
        self.read_ref_frames(dec, r, c, segment_id, skip_mode, info)?;
        let q_ymode0 = dec.consumed_bits();
        let is_compound = info.ref_frame[1] > mv::INTRA_FRAME;
        let stack = self.find_mv_stack(mi_grid, r, c, bsize, info.ref_frame, is_compound);

        let seg_skip = self
            .fh
            .params
            .segmentation
            .feature_active(segment_id as usize, SEG_LVL_SKIP);
        let seg_globalmv = self
            .fh
            .params
            .segmentation
            .feature_active(segment_id as usize, SEG_LVL_GLOBALMV);
        let y_mode = if skip_mode {
            NEAREST_NEARESTMV
        } else if seg_skip || seg_globalmv {
            GLOBALMV
        } else if is_compound {
            let ctx = COMPOUND_MODE_CTX_MAP[(stack.ref_mv_context >> 1).min(2)]
                [stack.new_mv_context.min(COMP_NEWMV_CTXS - 1)];
            let s = read_sym(dec, &mut self.stats, &mut self.cdfs.compound_mode[ctx.min(7)])? as u8;
            NEAREST_NEARESTMV + s
        } else {
            let new_mv = read_sym(dec, &mut self.stats, &mut self.cdfs.new_mv[stack.new_mv_context.min(5)])?;
            if new_mv == 0 {
                NEWMV
            } else {
                let zero_mv =
                    read_sym(dec, &mut self.stats, &mut self.cdfs.zero_mv[stack.zero_mv_context.min(1)])?;
                if zero_mv == 0 {
                    GLOBALMV
                } else {
                    let ref_mv =
                        read_sym(dec, &mut self.stats, &mut self.cdfs.ref_mv[stack.ref_mv_context.min(5)])?;
                    if ref_mv == 0 {
                        NEARESTMV
                    } else {
                        NEARMV
                    }
                }
            }
        };
        info.y_mode = y_mode;
        let q_drl0 = dec.consumed_bits();

        let mut ref_mv_idx = 0usize;
        if y_mode == NEWMV || y_mode == NEW_NEWMV {
            for idx in 0..2usize {
                if stack.num > idx + 1 {
                    let s = read_sym(dec, &mut self.stats, &mut self.cdfs.drl_mode[stack.drl_ctx[idx].min(2)])?;
                    if s == 0 {
                        ref_mv_idx = idx;
                        break;
                    }
                    ref_mv_idx = idx + 1;
                }
            }
        } else if has_nearmv(y_mode) {
            ref_mv_idx = 1;
            for idx in 1..3usize {
                if stack.num > idx + 1 {
                    let s = read_sym(dec, &mut self.stats, &mut self.cdfs.drl_mode[stack.drl_ctx[idx].min(2)])?;
                    if s == 0 {
                        ref_mv_idx = idx;
                        break;
                    }
                    ref_mv_idx = idx + 1;
                }
            }
        }

        let q_mv0 = dec.consumed_bits();
        self.assign_mv(dec, &stack, y_mode, is_compound, ref_mv_idx, false, info)?;
        let q_ii0 = dec.consumed_bits();
        self.read_interintra_mode(dec, is_compound, info)?;
        let q_mm0 = dec.consumed_bits();
        self.read_motion_mode(dec, mi_grid, bsize, is_compound, info)?;
        let q_ct0 = dec.consumed_bits();
        self.read_compound_type(dec, bsize, is_compound, skip_mode, info)?;
        let q_if0 = dec.consumed_bits();

        if self.fh.interpolation_filter == SWITCHABLE {
            let dirs = if self.seq.enable_dual_filter { 2 } else { 1 };
            for dir in 0..dirs {
                if needs_interp_filter_here(self.fh, skip_mode, info, bsize) {
                    let ctx = self.interp_filter_ctx(r, c, dir, info.ref_frame);
                    info.interp_filter[dir] =
                        read_sym(dec, &mut self.stats, &mut self.cdfs.interp_filter[ctx.min(15)])? as u8;
                } else {
                    info.interp_filter[dir] = EIGHTTAP;
                }
            }
            if !self.seq.enable_dual_filter {
                info.interp_filter[1] = info.interp_filter[0];
            }
        } else {
            info.interp_filter = [self.fh.interpolation_filter; 2];
        }
        trace_inter_blk(
            r,
            c,
            bsize,
            info,
            stack.num,
            [
                q_ymode0 - q_rf,
                q_drl0 - q_ymode0,
                q_mv0 - q_drl0,
                q_ii0 - q_mv0,
                q_mm0 - q_ii0,
                q_ct0 - q_mm0,
                q_if0 - q_ct0,
                dec.consumed_bits() - q_if0,
            ],
        );
        Ok(())
    }

    fn interp_filter_ctx(&self, r: i32, c: i32, dir: usize, ref_frame: [i8; 2]) -> usize {
        let mut ctx = ((dir & 1) * 2 + usize::from(ref_frame[1] > mv::INTRA_FRAME)) * 4;
        let mut left_type = 3usize;
        let mut above_type = 3usize;
        if self.is_inside(r, c - 1) {
            let rf = self.grid.ref_frames[self.grid.idx(r, c - 1)];
            if rf[0] == ref_frame[0] || rf[1] == ref_frame[0] {
                left_type = self.grid.interp_filters[self.grid.idx(r, c - 1)][dir] as usize;
            }
        }
        if self.is_inside(r - 1, c) {
            let rf = self.grid.ref_frames[self.grid.idx(r - 1, c)];
            if rf[0] == ref_frame[0] || rf[1] == ref_frame[0] {
                above_type = self.grid.interp_filters[self.grid.idx(r - 1, c)][dir] as usize;
            }
        }
        if left_type == above_type {
            ctx += left_type;
        } else if left_type == 3 {
            ctx += above_type;
        } else if above_type == 3 {
            ctx += left_type;
        } else {
            ctx += 3;
        }
        ctx
    }

    fn find_mv_stack(
        &self,
        mi_grid: &MiGrid,
        r: i32,
        c: i32,
        bsize: u8,
        ref_frame: [i8; 2],
        is_compound: bool,
    ) -> RefMvStack {
        MvStackInput {
            grid: self.grid,
            mi_grid,
            fh: self.fh,
            seq: self.seq,
            field: self.field,
            mi_row: r,
            mi_col: c,
            mi_size: bsize,
            ref_frame,
            mi_row_start: self.mi_row_start,
            mi_row_end: self.mi_row_end,
            mi_col_start: self.mi_col_start,
            mi_col_end: self.mi_col_end,
        }
        .find_mv_stack(is_compound)
    }

    /// `assign_mv( isCompound )`
    ///
    /// `intrabc` 为真时走规范里 `use_intrabc` 的那两支：`compMode` **恒为
    /// `NEWMV`**（不读位，所以不看 `y_mode`），`PredMv` 有自己的一条**回退链**。
    #[allow(clippy::too_many_arguments)]
    fn assign_mv(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        stack: &RefMvStack,
        y_mode: u8,
        is_compound: bool,
        ref_mv_idx: usize,
        intrabc: bool,
        info: &mut BlockInfo,
    ) -> Result<()> {
        let mv_ctx = if intrabc { mv::MV_INTRABC_CONTEXT } else { 0 };
        for i in 0..(1 + usize::from(is_compound)) {
            let comp_mode = if intrabc {
                NEWMV
            } else {
                get_mode(y_mode, i)
            };
            let mut pred_mv = if intrabc {
                // 规范 §5.11.13 `assign_mv` 的 `use_intrabc` 支：
                // 先拿栈里第 0 个候选，是零矢量就换第 1 个，还不行就用
                // "往上（或往左）挪一个超块再退 `INTRABC_DELAY_PIXELS`" 兜底。
                let mut p = stack.mv[0][0];
                if p == [0, 0] {
                    p = stack.mv[1][0];
                }
                if p == [0, 0] {
                    let sb_size = if self.seq.use_128x128_superblock {
                        BLOCK_128X128
                    } else {
                        BLOCK_64X64
                    };
                    let sb4 = NUM_4X4_BLOCKS_HIGH[sb_size as usize] as i32;
                    p = if info.mi_row as i32 - sb4 < self.mi_row_start {
                        // 上方超块不够用 ⇒ 改成往左，并且**再多退一个延迟量**
                        [0, -(sb4 * MI_SIZE as i32 + mv::INTRABC_DELAY_PIXELS) * 8]
                    } else {
                        [-(sb4 * MI_SIZE as i32 * 8), 0]
                    };
                }
                p
            } else if comp_mode == GLOBALMV {
                stack.global_mvs[i]
            } else {
                let mut pos = if comp_mode == NEARESTMV { 0 } else { ref_mv_idx };
                if comp_mode == NEWMV && stack.num <= 1 {
                    pos = 0;
                }
                stack.mv[pos.min(mv::MAX_REF_MV_STACK_SIZE - 1)][i]
            };
            if comp_mode == NEWMV {
                pred_mv = self.read_mv(dec, i, pred_mv, mv_ctx)?;
            }
            info.mv[i] = pred_mv;
        }
        Ok(())
    }

    /// `read_mv( ref )`
    ///
    /// `mv_ctx` 只有两个取值：普通帧间是 0，`use_intrabc` 是
    /// `MV_INTRABC_CONTEXT`（规范 §5.11.20 的 `MvCtx`）。整个 MV 的 CDF 表都按
    /// 这一维分开，抄错就会让后续每一位都错位。
    fn read_mv(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        _list: usize,
        pred: Mv,
        mv_ctx: usize,
    ) -> Result<Mv> {
        let mut diff = [0i32; 2];
        let joint = read_sym(dec, &mut self.stats, &mut self.cdfs.mv_joint[mv_ctx])?;
        if joint == MV_JOINT_HZVNZ || joint == MV_JOINT_HNZVNZ {
            diff[0] = self.read_mv_component(dec, mv_ctx, 0)?;
        }
        if joint == MV_JOINT_HNZVZ || joint == MV_JOINT_HNZVNZ {
            diff[1] = self.read_mv_component(dec, mv_ctx, 1)?;
        }
        Ok([pred[0] + diff[0], pred[1] + diff[1]])
    }

    /// `read_mv_component( comp )`
    fn read_mv_component(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        mv_ctx: usize,
        comp: usize,
    ) -> Result<i32> {
        let sign = read_sym(dec, &mut self.stats, &mut self.cdfs.mv_sign[mv_ctx][comp])?;
        let class = read_sym(dec, &mut self.stats, &mut self.cdfs.mv_class[mv_ctx][comp])?;
        let mag = if class == MV_CLASS_0 {
            let bit0 = read_sym(dec, &mut self.stats, &mut self.cdfs.mv_class0_bit[mv_ctx][comp])?;
            let fr = if self.fh.force_integer_mv {
                3
            } else {
                read_sym(
                    dec,
                    &mut self.stats,
                    &mut self.cdfs.mv_class0_fr[mv_ctx][comp][bit0 as usize],
                )?
            };
            let hp = if self.fh.allow_high_precision_mv {
                read_sym(dec, &mut self.stats, &mut self.cdfs.mv_class0_hp[mv_ctx][comp])?
            } else {
                1
            };
            ((bit0 << 3) | (fr << 1) | hp) + 1
        } else {
            let mut d = 0u32;
            for i in 0..class {
                let b = read_sym(
                    dec,
                    &mut self.stats,
                    &mut self.cdfs.mv_bit[mv_ctx][comp][i as usize],
                )?;
                d |= b << i;
            }
            let fr = if self.fh.force_integer_mv {
                3
            } else {
                read_sym(dec, &mut self.stats, &mut self.cdfs.mv_fr[mv_ctx][comp])?
            };
            let hp = if self.fh.allow_high_precision_mv {
                read_sym(dec, &mut self.stats, &mut self.cdfs.mv_hp[mv_ctx][comp])?
            } else {
                1
            };
            ((CLASS0_SIZE as u32) << (class + 2)) + ((d << 3) | (fr << 1) | hp) + 1
        };
        let mag = mag as i32;
        Ok(if sign == 1 { -mag } else { mag })
    }

    /// `read_interintra_mode( isCompound )`
    fn read_interintra_mode(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        is_compound: bool,
        info: &mut BlockInfo,
    ) -> Result<()> {
        info.interintra = false;
        info.wedge_interintra = false;
        let bsize = info.mi_size;
        if !info.skip_mode
            && self.seq.enable_interintra_compound
            && !is_compound
            && bsize >= BLOCK_8X8 as u8
            && bsize <= BLOCK_32X32 as u8
        {
            let ctx = SIZE_GROUP[bsize as usize].saturating_sub(1);
            let s = read_sym(dec, &mut self.stats, &mut self.cdfs.interintra[ctx.min(2)])?;
            if s == 1 {
                info.interintra = true;
                // 这两个值**必须在像素侧用到**：`interintra_mode` 决定帧内那一半的预测
                // 方向，`wedge_index` 决定楔形混合的掩码。早先读出来就丢掉了，
                // 于是像素侧只能"知道是 interintra 但不知道怎么混"。
                info.interintra_mode =
                    read_sym(dec, &mut self.stats, &mut self.cdfs.interintra_mode[ctx.min(2)])?
                        as u8;
                info.ref_frame[1] = mv::INTRA_FRAME;
                info.angle_delta_y = 0;
                info.angle_delta_uv = 0;
                info.use_filter_intra = false;
                let w = read_sym(dec, &mut self.stats, &mut self.cdfs.wedge_interintra[bsize as usize])?;
                info.wedge_interintra = w == 1;
                if info.wedge_interintra {
                    info.wedge_index =
                        read_sym(dec, &mut self.stats, &mut self.cdfs.wedge_index[bsize as usize])?
                            as u8;
                    // 规范：`wedge_sign = 0`（interintra 只有一种符号，不像复合那样再读）
                    info.wedge_sign = 0;
                }
            }
        }
        Ok(())
    }

    /// `read_motion_mode( isCompound )`
    fn read_motion_mode(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        mi_grid: &MiGrid,
        bsize: u8,
        is_compound: bool,
        info: &mut BlockInfo,
    ) -> Result<()> {
        info.motion_mode = SIMPLE;
        if info.skip_mode || !self.fh.is_motion_mode_switchable {
            return Ok(());
        }
        if block_width(bsize).min(block_height(bsize)) < 8 {
            return Ok(());
        }
        if !self.fh.force_integer_mv && (info.y_mode == GLOBALMV || info.y_mode == mv::GLOBAL_GLOBALMV) {
            if self.fh.params.global_motion.gm_type[info.ref_frame[0] as usize] > super::params::TRANSLATION {
                return Ok(());
            }
        }
        if is_compound || info.ref_frame[1] == mv::INTRA_FRAME {
            return Ok(());
        }
        if !self.has_overlappable(bsize) {
            return Ok(());
        }
        let num_samples = self.num_warp_samples(mi_grid, info.mv[0], bsize, info.ref_frame);
        if flags::trace_mm() {
            eprintln!(
                "MM bsize={bsize} force_integer_mv={} num_samples={num_samples} \
                 allow_warped_motion={} is_scaled={} is_compound={is_compound} \
                 y_mode={} ref0={} ref1={} mv0=({},{})",
                self.fh.force_integer_mv,
                self.fh.params.allow_warped_motion,
                is_scaled(self.store, self.fh, info.ref_frame[0]),
                info.y_mode,
                info.ref_frame[0],
                info.ref_frame[1],
                info.mv[0][0],
                info.mv[0][1]
            );
        }
        if self.fh.force_integer_mv
            || num_samples == 0
            || !self.fh.params.allow_warped_motion
            || is_scaled(self.store, self.fh, info.ref_frame[0])
        {
            let s = read_sym(dec, &mut self.stats, &mut self.cdfs.use_obmc[bsize as usize])?;
            info.motion_mode = if s == 1 { OBMC } else { SIMPLE };
        } else {
            info.motion_mode =
                read_sym(dec, &mut self.stats, &mut self.cdfs.motion_mode[bsize as usize])? as u8;
        }
        if info.motion_mode != SIMPLE && flags::trace_mm() {
            eprintln!("MMDEC bsize={bsize} motion_mode={}", info.motion_mode);
        }
        Ok(())
    }

    /// `has_overlappable_candidates( )`
    fn has_overlappable(&self, bsize: u8) -> bool {
        let bw4 = NUM_4X4_BLOCKS_WIDE[bsize as usize] as i32;
        let bh4 = NUM_4X4_BLOCKS_HIGH[bsize as usize] as i32;
        let (r, c) = (self.cur_row, self.cur_col);
        if self.is_inside(r - 1, c) {
            let mut x4 = c;
            while x4 < (c + bw4).min(self.fh.mi_cols as i32) {
                if self.grid.ref_frames[self.grid.idx(r - 1, x4 | 1)][0] > mv::INTRA_FRAME {
                    return true;
                }
                x4 += 2;
            }
        }
        if self.is_inside(r, c - 1) {
            let mut y4 = r;
            while y4 < (r + bh4).min(self.fh.mi_rows as i32) {
                if self.grid.ref_frames[self.grid.idx(y4 | 1, c - 1)][0] > mv::INTRA_FRAME {
                    return true;
                }
                y4 += 2;
            }
        }
        false
    }

    fn num_warp_samples(
        &self,
        mi_grid: &MiGrid,
        mv_value: Mv,
        bsize: u8,
        ref_frame: [i8; 2],
    ) -> usize {
        MvStackInput {
            grid: self.grid,
            mi_grid,
            fh: self.fh,
            seq: self.seq,
            field: None,
            mi_row: self.cur_row,
            mi_col: self.cur_col,
            mi_size: bsize,
            // 这里**必须**传当前块真正的参考帧：`add_sample`（规范 §7.10.4.2）里有
            // `RefFrames[mvRow][mvCol][0] != RefFrame[0] → return` —— 它拿邻居的
            // 参考帧跟当前块比。传 `INTRA_FRAME` 会让每个候选都提前返回，
            // `NumSamplesScanned` 停在 0，连"扫过候选就强制记 1"的兜底都救不回来，
            // 于是 `read_motion_mode` 永远走 `use_obmc` 分支（少一个符号 ⇒ 整条
            // 算术流从那里起错位）。
            ref_frame,
            mi_row_start: self.mi_row_start,
            mi_row_end: self.mi_row_end,
            mi_col_start: self.mi_col_start,
            mi_col_end: self.mi_col_end,
        }
        .num_warp_samples(mv_value)
    }

    /// `useWarp` 的推导（规范 §7.11.3.1 的步骤 2–6）。
    ///
    /// 放在这里而不是 `predict_inter` 里，是因为要 `is_scaled`（看参考帧槽）与
    /// 最小二乘拟合 —— 那两样只有 `decode_block` 这一层拿得到。
    /// **`w < 8 || h < 8` 那一支不在这里**：它判的是当前平面的尺寸，
    /// 只有 `predict_inter` 知道自己在哪个平面上。
    fn warp_plan(&self, mi_grid: &MiGrid, info: &BlockInfo, bsize: u8) -> super::inter::WarpPlan {
        let mut plan = super::inter::WarpPlan::default();
        if self.fh.force_integer_mv {
            return plan;
        }
        if info.motion_mode == LOCALWARP {
            // `read_motion_mode` 已经保证：非复合、`w/h >= 8`、`num_samples > 0`、
            // `allow_warped_motion`、`!is_scaled`。这里补规范步骤 2–3 的两步。
            let (num, cand) = MvStackInput {
                grid: self.grid,
                mi_grid,
                fh: self.fh,
                seq: self.seq,
                field: None,
                mi_row: self.cur_row,
                mi_col: self.cur_col,
                mi_size: bsize,
                ref_frame: info.ref_frame,
                mi_row_start: self.mi_row_start,
                mi_row_end: self.mi_row_end,
                mi_col_start: self.mi_col_start,
                mi_col_end: self.mi_col_end,
            }
            .find_warp_samples(info.mv[0]);
            if num > 0 {
                let (valid, params) =
                    warping::warp_estimate(info.mv[0], self.cur_row, self.cur_col, bsize, num, &cand);
                if valid && warping::setup_shear(&params).valid {
                    plan.use_warp[0] = 1;
                    plan.local = params;
                }
            }
            return plan;
        }
        if info.y_mode == mv::GLOBALMV || info.y_mode == mv::GLOBAL_GLOBALMV {
            for list in 0..2 {
                let rf = info.ref_frame[list];
                if rf <= mv::INTRA_FRAME {
                    continue;
                }
                if self.fh.params.global_motion.gm_type[rf as usize] <= super::params::TRANSLATION {
                    continue;
                }
                if mv::is_scaled(self.store, self.fh, rf) {
                    continue;
                }
                let params = self.fh.params.global_motion.params[rf as usize];
                if warping::setup_shear(&params).valid {
                    plan.use_warp[list] = 2;
                }
            }
        }
        plan
    }

    /// `read_compound_type( isCompound )`
    fn read_compound_type(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        bsize: u8,
        is_compound: bool,
        skip_mode: bool,
        info: &mut BlockInfo,
    ) -> Result<()> {
        info.compound_type = COMPOUND_AVERAGE;
        info.comp_group_idx = 0;
        info.compound_idx = 1;
        if skip_mode {
            return Ok(());
        }
        if is_compound {
            let n = WEDGE_BITS[bsize as usize] as u32;
            if self.seq.enable_masked_compound {
                let ctx = self.comp_group_idx_ctx();
                info.comp_group_idx =
                    read_sym(dec, &mut self.stats, &mut self.cdfs.comp_group_idx[ctx])? as u8;
            }
            if info.comp_group_idx == 0 {
                if self.seq.enable_jnt_comp {
                    let ctx = self.compound_idx_ctx(info);
                    info.compound_idx =
                        read_sym(dec, &mut self.stats, &mut self.cdfs.compound_idx[ctx])? as u8;
                    info.compound_type = if info.compound_idx == 1 {
                        COMPOUND_AVERAGE
                    } else {
                        COMPOUND_DISTANCE
                    };
                } else {
                    info.compound_type = COMPOUND_AVERAGE;
                }
            } else if n == 0 {
                info.compound_type = COMPOUND_DIFFWTD;
            } else {
                info.compound_type =
                    read_sym(dec, &mut self.stats, &mut self.cdfs.compound_type[bsize as usize])? as u8;
            }
            if info.compound_type == COMPOUND_WEDGE {
                info.wedge_index =
                    read_sym(dec, &mut self.stats, &mut self.cdfs.wedge_index[bsize as usize])?
                        as u8;
                info.wedge_sign = dec.read_literal(1)? as u8;
                self.stats.symbols += 1;
            } else if info.compound_type == COMPOUND_DIFFWTD {
                info.mask_type = dec.read_literal(1)? as u8;
                self.stats.symbols += 1;
            }
        } else if info.interintra {
            info.compound_type = if info.wedge_interintra { COMPOUND_WEDGE } else { COMPOUND_INTRA };
        }
        Ok(())
    }

    fn comp_group_idx_ctx(&self) -> usize {
        let (r, c) = (self.cur_row, self.cur_col);
        let avail_u = self.is_inside(r - 1, c);
        let avail_l = self.is_inside(r, c - 1);
        let above = if avail_u { self.grid.ref_frames[self.grid.idx(r - 1, c)] } else { [mv::INTRA_FRAME, NONE] };
        let left = if avail_l { self.grid.ref_frames[self.grid.idx(r, c - 1)] } else { [mv::INTRA_FRAME, NONE] };
        let mut ctx = 0usize;
        if avail_u {
            if above[1] > mv::INTRA_FRAME {
                ctx += self.grid.comp_group_idxs[self.grid.idx(r - 1, c)] as usize;
            } else if above[0] == mv::ALTREF_FRAME {
                ctx += 3;
            }
        }
        if avail_l {
            if left[1] > mv::INTRA_FRAME {
                ctx += self.grid.comp_group_idxs[self.grid.idx(r, c - 1)] as usize;
            } else if left[0] == mv::ALTREF_FRAME {
                ctx += 3;
            }
        }
        ctx.min(5)
    }

    fn compound_idx_ctx(&self, info: &BlockInfo) -> usize {
        let (r, c) = (self.cur_row, self.cur_col);
        let avail_u = self.is_inside(r - 1, c);
        let avail_l = self.is_inside(r, c - 1);
        let above = if avail_u { self.grid.ref_frames[self.grid.idx(r - 1, c)] } else { [mv::INTRA_FRAME, NONE] };
        let left = if avail_l { self.grid.ref_frames[self.grid.idx(r, c - 1)] } else { [mv::INTRA_FRAME, NONE] };
        let hints = self.fh.order_hints;
        let fwd = mv::get_relative_dist(self.seq, hints[info.ref_frame[0] as usize], self.fh.order_hint).abs();
        let bck = mv::get_relative_dist(self.seq, hints[info.ref_frame[1] as usize], self.fh.order_hint).abs();
        let mut ctx = if fwd == bck { 3 } else { 0 };
        if avail_u {
            if above[1] > mv::INTRA_FRAME {
                ctx += self.grid.compound_idxs[self.grid.idx(r - 1, c)] as usize;
            } else if above[0] == mv::ALTREF_FRAME {
                ctx += 1;
            }
        }
        if avail_l {
            if left[1] > mv::INTRA_FRAME {
                ctx += self.grid.compound_idxs[self.grid.idx(r, c - 1)] as usize;
            } else if left[0] == mv::ALTREF_FRAME {
                ctx += 1;
            }
        }
        ctx.min(5)
    }

    // ── 变换尺寸 ────────────────────────────────────────────────────────

    /// `read_block_tx_size( )`
    fn read_block_tx_size(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        mi_grid: &MiGrid,
        info: &mut BlockInfo,
    ) -> Result<()> {
        let bsize = info.mi_size;
        let bw4 = NUM_4X4_BLOCKS_WIDE[bsize as usize] as i32;
        let bh4 = NUM_4X4_BLOCKS_HIGH[bsize as usize] as i32;
        let lossless = self.lossless_array[info.segment_id as usize];
        let (r, c) = (info.mi_row as i32, info.mi_col as i32);
        if self.fh.params.tx_mode == TX_MODE_SELECT
            && bsize as usize > BLOCK_4X4
            && info.is_inter
            && !info.skip
            && !lossless
        {
            let max_tx_sz = MAX_TX_SIZE_RECT[bsize as usize];
            let tx_w4 = (TX_WIDTH[max_tx_sz as usize] as i32) / 4;
            let tx_h4 = (TX_HEIGHT[max_tx_sz as usize] as i32) / 4;
            let mut row = r;
            while row < r + bh4 {
                let mut col = c;
                while col < c + bw4 {
                    self.read_var_tx_size(dec, mi_grid, row, col, max_tx_sz, 0, info)?;
                    col += tx_w4;
                }
                row += tx_h4;
            }
        } else {
            let allow_select = !info.skip || !info.is_inter;
            self.read_tx_size(dec, mi_grid, bsize, lossless, allow_select, info)?;
            let tx = info.tx_size;
            self.grid.stamp(r, c, bsize, move |g, i| g.inter_tx_sizes[i] = tx);
        }
        Ok(())
    }

    /// `read_var_tx_size( row, col, txSz, depth )`
    fn read_var_tx_size(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        mi_grid: &MiGrid,
        row: i32,
        col: i32,
        tx_sz: u8,
        depth: u32,
        info: &mut BlockInfo,
    ) -> Result<()> {
        if row >= self.fh.mi_rows as i32 || col >= self.fh.mi_cols as i32 {
            return Ok(());
        }
        let txfm_split = if tx_sz == TX_4X4 || depth == MAX_VARTX_DEPTH {
            0
        } else {
            let ctx = self.txfm_split_ctx(mi_grid, row, col, tx_sz)?;
            read_sym(dec, &mut self.stats, &mut self.cdfs.txfm_split[ctx.min(20)])?
        };
        let w4 = (TX_WIDTH[tx_sz as usize] as i32) / 4;
        let h4 = (TX_HEIGHT[tx_sz as usize] as i32) / 4;
        if txfm_split == 1 {
            let sub = SPLIT_TX_SIZE[tx_sz as usize];
            let step_w = (TX_WIDTH[sub as usize] as i32) / 4;
            let step_h = (TX_HEIGHT[sub as usize] as i32) / 4;
            let mut i = 0;
            while i < h4 {
                let mut j = 0;
                while j < w4 {
                    self.read_var_tx_size(dec, mi_grid, row + i, col + j, sub, depth + 1, info)?;
                    j += step_w;
                }
                i += step_h;
            }
        } else {
            for i in 0..h4 {
                for j in 0..w4 {
                    let idx = self.grid.idx(row + i, col + j);
                    self.grid.inter_tx_sizes[idx] = tx_sz;
                }
            }
            info.tx_size = tx_sz;
        }
        Ok(())
    }

    /// `txfm_split` 的上下文（规范 §8.3.2）
    fn txfm_split_ctx(&self, mi_grid: &MiGrid, row: i32, col: i32, tx_sz: u8) -> Result<usize> {
        // `row`/`col` 是**当前子变换块的绝对 mi 坐标**（规范 `read_var_tx_size` 传的就是它，
        // libaom 侧是 `xd->above_txfm_context + blk_col` / `left_txfm_context + blk_row`）。
        // 别拿 `cur_row/cur_col`（块原点）顶替：深度 ≥1 的子块那样会去量**块外侧**的邻居，
        // 而不是同块内已经解过的左/上子块 ⇒ ctx 落到另一格 ⇒ `txfm_split` 读错 CDF。
        let above = self.get_above_tx_width(mi_grid, row, col) < TX_WIDTH[tx_sz as usize] as i32;
        let left = self.get_left_tx_height(mi_grid, row, col) < TX_HEIGHT[tx_sz as usize] as i32;
        let size = block_width(self.cur_bsize).max(block_height(self.cur_bsize)).min(64);
        let max_tx_sz = super::coeff::find_tx_size(size, size);
        let tx_sz_sqr_up = super::coeff::TX_SIZE_SQR_UP[tx_sz as usize];
        // libaom `txfm_partition_context`：
        //   category = (sqr_up != max && max > TX_8X8) + (TX_SIZES-1-max)*2
        //   ctx      = category*3 + above + left
        // **`max > TX_8X8` 这一项不能省**：`max == TX_8X8`（16×16 的块）时它恒为 0，
        // 少了它会把 ctx 从 18/19/20 顶到 21/22/23（再被 `.min(20)` 夹成 20），
        // 于是 `txfm_split` 读到另一份 CDF —— 只在 `above+left==3` 时偶然对上。
        let cat_extra = usize::from(tx_sz_sqr_up != max_tx_sz && max_tx_sz > TX_8X8);
        Ok(cat_extra * 3
            + (TX_SIZES - 1 - max_tx_sz as usize) * 6
            + usize::from(above)
            + usize::from(left))
    }

    /// `get_above_tx_width( row, col )`（规范 §8.3.2 的 `txfm_split` / `tx_depth` 条目）
    ///
    /// 注意 `Skips && IsInters` 那一支返回的是**邻居块的宽度**（不是 64）——
    /// 跳过的帧间块不写 `InterTxSizes`，所以要看块尺寸。
    fn get_above_tx_width(&self, mi_grid: &MiGrid, row: i32, col: i32) -> i32 {
        if row == self.cur_row {
            if !self.is_inside(row - 1, col) {
                return 64;
            }
            let i = self.grid.idx(row - 1, col);
            if self.grid.skips[i] != 0 && self.grid.is_inters[i] != 0 {
                return block_width(mi_grid.get(row - 1, col));
            }
        }
        TX_WIDTH[self.grid.inter_tx_sizes[self.grid.idx(row - 1, col)] as usize] as i32
    }

    /// `get_left_tx_height( row, col )`
    fn get_left_tx_height(&self, mi_grid: &MiGrid, row: i32, col: i32) -> i32 {
        if col == self.cur_col {
            if !self.is_inside(row, col - 1) {
                return 64;
            }
            let i = self.grid.idx(row, col - 1);
            if self.grid.skips[i] != 0 && self.grid.is_inters[i] != 0 {
                return block_height(mi_grid.get(row, col - 1));
            }
        }
        TX_HEIGHT[self.grid.inter_tx_sizes[self.grid.idx(row, col - 1)] as usize] as i32
    }

    /// `read_tx_size( allowSelect )`
    fn read_tx_size(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        mi_grid: &MiGrid,
        bsize: u8,
        lossless: bool,
        allow_select: bool,
        info: &mut BlockInfo,
    ) -> Result<()> {
        if lossless {
            info.tx_size = TX_4X4;
            return Ok(());
        }
        let max_rect = MAX_TX_SIZE_RECT[bsize as usize];
        let max_depth = MAX_TX_DEPTH_BY_BSIZE[bsize as usize];
        let mut tx_size = max_rect;
        if bsize as usize > BLOCK_4X4 && allow_select && self.fh.params.tx_mode == TX_MODE_SELECT {
            let ctx = self.tx_depth_ctx(mi_grid, bsize, info);
            let depth = match max_depth {
                4 => read_sym(dec, &mut self.stats, &mut self.cdfs.tx64x64[ctx.min(2)])?,
                3 => read_sym(dec, &mut self.stats, &mut self.cdfs.tx32x32[ctx.min(2)])?,
                2 => read_sym(dec, &mut self.stats, &mut self.cdfs.tx16x16[ctx.min(2)])?,
                _ => read_sym(dec, &mut self.stats, &mut self.cdfs.tx8x8[ctx.min(2)])?,
            };
            for _ in 0..depth {
                tx_size = SPLIT_TX_SIZE[tx_size as usize];
            }
        }
        info.tx_size = tx_size;
        Ok(())
    }

    /// `tx_depth` 的上下文（规范 §8.3.2）
    fn tx_depth_ctx(&self, mi_grid: &MiGrid, bsize: u8, info: &BlockInfo) -> usize {
        let (r, c) = (info.mi_row as i32, info.mi_col as i32);
        let max_rect = MAX_TX_SIZE_RECT[bsize as usize];
        let max_w = TX_WIDTH[max_rect as usize] as i32;
        let max_h = TX_HEIGHT[max_rect as usize] as i32;
        let avail_u = self.is_inside(r - 1, c);
        let avail_l = self.is_inside(r, c - 1);
        let above_w = if avail_u {
            let i = self.grid.idx(r - 1, c);
            if self.grid.is_inters[i] != 0 {
                block_width(mi_grid.get(r - 1, c))
            } else {
                self.get_above_tx_width(mi_grid, r, c)
            }
        } else {
            0
        };
        let left_h = if avail_l {
            let i = self.grid.idx(r, c - 1);
            if self.grid.is_inters[i] != 0 {
                block_height(mi_grid.get(r, c - 1))
            } else {
                self.get_left_tx_height(mi_grid, r, c)
            }
        } else {
            0
        };
        usize::from(above_w >= max_w) + usize::from(left_h >= max_h)
    }

    // ── 残差 ────────────────────────────────────────────────────────────

    /// **row-mt 重建阶段（顺序版）**：把语法阶段记下的工单按记录序回放。
    /// 调色板块的 flush 与 `ME_ROW_MT=2` 走这条；默认走并行版
    /// （[`Self::replay_jobs_parallel`]）。
    pub fn replay_jobs(&mut self, mi_grid: &MiGrid) -> Result<()> {
        if self.row_mt.blocks.is_empty() {
            return Ok(());
        }
        if let Some(frame) = self.frame.as_deref() {
            self.row_mt.precompute_dec_flags(frame, self.seq);
        }
        if flags::trace_race() {
            if let Some(frame) = self.frame.as_deref() {
                let (d, cols) = frame.decoded_grid(0);
                let rows = if cols == 0 { 0 } else { d.len() / cols };
                racemap::init(
                    cols,
                    rows,
                    self.seq.color.subsampling_x,
                    self.seq.color.subsampling_y,
                );
            }
        }
        let blocks = std::mem::take(&mut self.row_mt.blocks);
        let txbs = std::mem::take(&mut self.row_mt.txbs);
        let levels = std::mem::take(&mut self.row_mt.levels);
        let shared = ReplayShared {
            frame: self
                .frame
                .as_deref_mut()
                .map(|f| f as *mut CurrFrame)
                .unwrap_or(std::ptr::null_mut()),
            grid: &*self.grid,
            mi_grid,
            store: self.store,
            seq: self.seq,
            fh: self.fh,
            blocks: &blocks,
            txbs: &txbs,
            levels: &levels,
        };
        let r = replay_blocks(&shared, 0..blocks.len());
        self.row_mt.blocks = blocks;
        self.row_mt.txbs = txbs;
        self.row_mt.levels = levels;
        self.row_mt.reset();
        r
    }

    /// **row-mt 阶段 2**：把回放按 **(SB 行 × SB 列) 作业**并行。
    ///
    /// ## 为什么不是"每个 SB 行一个任务"
    ///
    /// 早先的版本是"一条 SB 行 = 一个任务、按 `行号 % worker 数` 分配"。真素材用的是
    /// **128×128 超块**，1080p 只有 **9 条带**、15 个超块列 ⇒ 4 个 worker 里每个要
    /// **串行**走 2~3 条整带，波前只在带与带的边界起作用。实测（`ME_TRACE_WAIT=1`）
    /// 忙时从单线程 199ms 膨胀到 8 线程 640ms、墙钟只快 1.07x。
    ///
    /// 正确粒度是 **libaom row-mt 的 job**：`(带, 超块列)`。一条带被切成 15 个作业，
    /// 总量 9×15 = 135 个；依赖只落在**上一带的同列 / 次列**（`above_need` 的上界，
    /// 见 [`above_need_mi`]）**以及同带左邻列**（仅当该作业要读当前帧像素时，
    /// 见 [`reads_cur_frame`]），于是可以按**反对角线** `d = 2*band + col` 调度：
    /// 同一条对角线内部的作业互不依赖（依赖要么在 `d-1`，要么在同 `d` 里**列号更大、
    /// 因而排在前面**的那个），天然并行。
    ///
    /// 调度 = 一条全局作业队列（`AtomicUsize` 取号）按反对角线序发牌 + 每个作业一个
    /// `AtomicBool` 完成位。**依赖在 `order` 里一定排在作业自己前面**，所以不会死锁；
    /// 队列取号天然做动态负载均衡，不需要"按行轮流"那种静态分配。
    ///
    /// 帧的写入仍靠 [`ReplayShared::frame`] 的裸指针 + **"同一时刻只有一行在写"** 的
    /// 不变量承载（作业按超块列切，行区间互不重叠）。
    ///
    /// 保守情形：`intrabc` 的 MV 能指向任何已解码区域 ⇒ 等上一带**整带**做完；
    /// `above_need` 超出两列时同样退化成整带等待（`ME_ROW_MT_FULLWAIT=1` 强制全体走这条，
    /// 用于二分诊断）。
    pub fn replay_jobs_parallel(&mut self, mi_grid: &MiGrid) -> Result<()> {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        /// 一个作业 = 一条带里**同一超块列**的那一段连续块。
        struct Job {
            a: usize,
            b: usize,
            band: u32,
            col: u32,
            /// 调度层 `2*band + col`：三条依赖（上一带同列 / 上一带次列 / 同带左邻列）
            /// 在这个量上都**严格更小** ⇒ 按它升序发牌就是一个合法拓扑序（无死锁）。
            /// ⚠️ 不能用 `band + col`：那时"上一带次列"与自己是**同一层**，
            /// 层内还是一条链，并行度会被吃掉（踩过）。
            diag: u32,
            /// 依赖作业：`[0]`=上一带同列、`[1]`=上一带次列、`[2]`=**同带左邻列**
            /// （`u32::MAX` = 无）。
            ///
            /// ⚠️ `[2]` 是**条件性**的：只有当该作业里**确实有要读当前帧像素的块**
            /// 时才画（判据见 [`reads_cur_frame`]）。纯帧间块的预测全部来自参考帧，
            /// 所以整列纯帧间时这条边可以省 —— 一条带内就不再是串行链。
            deps: [u32; 3],
        }

        impl JobKey for Job {
            fn diag(&self) -> u32 {
                self.diag
            }
            fn col_key(&self) -> u32 {
                self.col
            }
        }

        if self.row_mt.blocks.is_empty() {
            return Ok(());
        }
        if let Some(frame) = self.frame.as_deref() {
            self.row_mt.precompute_dec_flags(frame, self.seq);
        }
        if flags::trace_race() {
            if let Some(frame) = self.frame.as_deref() {
                let (d, cols) = frame.decoded_grid(0);
                let rows = if cols == 0 { 0 } else { d.len() / cols };
                racemap::init(
                    cols,
                    rows,
                    self.seq.color.subsampling_x,
                    self.seq.color.subsampling_y,
                );
            }
        }
        let sb4 = if self.seq.use_128x128_superblock { 32 } else { 16 };
        let blocks = std::mem::take(&mut self.row_mt.blocks);
        let txbs = std::mem::take(&mut self.row_mt.txbs);
        let levels = std::mem::take(&mut self.row_mt.levels);

        // ── ① 切带（SB 行）：记录序是光栅序 ⇒ `mi_row / sb4` 单调不减 ──
        let mut ranges: Vec<(usize, usize, i32)> = Vec::new();
        for (i, b) in blocks.iter().enumerate() {
            let row = b.info.mi_row as i32;
            match ranges.last_mut() {
                Some((_, end, r0)) if row / sb4 == *r0 / sb4 => *end = i + 1,
                _ => ranges.push((i, i + 1, row)),
            }
        }

        // ── ② 每条带按超块列切成作业，并解析"上一带同列/次列"的依赖 ──
        let mut jobs: Vec<Job> = Vec::new();
        let mut band_cols: Vec<Vec<(u32, u32)>> = Vec::with_capacity(ranges.len());
        for (band, &(a, b, _)) in ranges.iter().enumerate() {
            let mut cols_here: Vec<(u32, u32)> = Vec::new();
            let mut i = a;
            while i < b {
                let col = (blocks[i].info.mi_col as i32 / sb4).max(0) as u32;
                let mut j = i + 1;
                while j < b && (blocks[j].info.mi_col as i32 / sb4).max(0) as u32 == col {
                    j += 1;
                }
                // 本作业里所有块"上一行读得最远"的上界（mi 列，不含）
                let need = blocks[i..j]
                    .iter()
                    .map(|x| x.above_need as i64)
                    .max()
                    .unwrap_or(0)
                    + i64::from(flags::row_mt_slack().unwrap_or(0));
                let want_cols =
                    ((need + i64::from(sb4) - 1) / i64::from(sb4)).max(col as i64) as u32;
                // ⚠️ 有两种情形**表达不了**成上面那三条边：
                //   ① `intrabc` 的 MV 能指向**任意**已解码区域（不只是上一带）；
                //   ② `above_need` 超出"本列 + 次列"（正常不可能，防御性）。
                // 早先给它们做了"等上一带整条做完"的分支，但那在反对角线发牌顺序下**会死锁**
                // （要等的作业排在后面）。现在改成**整帧退回顺序回放**：这两种情形在真素材上
                // 基本不出现（屏幕内容才会用 intrabc），换来的是"并行路径无死角"。
                if blocks[i..j].iter().any(|x| x.info.use_intrabc)
                    || want_cols > col + 2
                    || flags::row_mt_fullwait()
                {
                    self.row_mt.blocks = blocks;
                    self.row_mt.txbs = txbs;
                    self.row_mt.levels = levels;
                    return self.replay_jobs(mi_grid);
                }
                let band = band as u32;
                jobs.push(Job {
                    a: i,
                    b: j,
                    band,
                    col,
                    diag: 2 * band + col,
                    deps: [u32::MAX; 3],
                });
                cols_here.push((col, (jobs.len() - 1) as u32));
                i = j;
            }
            band_cols.push(cols_here);
        }
        for k in 0..jobs.len() {
            let (band, col) = (jobs[k].band, jobs[k].col);
            // ③-a **同带左邻列**：左列预测（`LeftCol`）读的是左边那一列的像素。
            // 旧版"一条带 = 一个作业"天然保证带内按列顺序，切成列作业后必须显式画这条边。
            // 它在 `d-1` 上（`band + col - 1`），所以不破坏反对角线调度。
            // ⚠️ **只有真的要读当前帧像素的作业才画这条边**：纯帧间块的预测全部来自
            // 参考帧（`warp` 只读邻居的 **MV**，语法阶段就填好了），所以**整列都是
            // 纯帧间块时这条边可以省** —— 一条带内不再是串行链，有效并行度从
            // "反对角线上 ~8 个"放开到"一条带内的列数"。判据见 [`reads_cur_frame`]。
            if col > 0 && blocks[jobs[k].a..jobs[k].b].iter().any(reads_cur_frame) {
                // "最近的、列号更小的那一列"（正常情况下就是 `col - 1`）
                if let Some(&(_, idx)) = band_cols[band as usize]
                    .iter()
                    .filter(|(c, _)| *c < col)
                    .max_by_key(|(c, _)| *c)
                {
                    jobs[k].deps[2] = idx;
                }
            }
            // ③-b **上一带的同列 / 次列**（`AboveRow` 读到 `x + 2w - 1`）
            if band == 0 {
                continue;
            }
            let prev = (band - 1) as usize;
            let mut d = 0usize;
            for &(c, idx) in &band_cols[prev] {
                if (c == col || c == col + 1) && d < 2 {
                    jobs[k].deps[d] = idx;
                    d += 1;
                }
            }
        }

        // ── ③ 发牌顺序
        //
        // * **普通作业**：反对角线升序；同一条对角线内**列号大的在前** —— 这样它的三个
        //   依赖（`d-1` 的上一带同列 / 同带左邻列、同 `d` 的上一带次列）都排在它前面。
        // 表达不了的依赖（`intrabc` / 上界超两列 / `ME_ROW_MT_FULLWAIT`）在切作业时就
        // **整帧退回顺序回放**了（见上面），所以这里只剩普通作业这一种。
        let mut order: Vec<u32> = (0..jobs.len() as u32).collect();
        order.sort_by_key(|&k| job_order_key(&jobs[k as usize]));

        let shared = ReplayShared {
            frame: self
                .frame
                .as_deref_mut()
                .map(|f| f as *mut CurrFrame)
                .unwrap_or(std::ptr::null_mut()),
            grid: &*self.grid,
            mi_grid,
            store: self.store,
            seq: self.seq,
            fh: self.fh,
            blocks: &blocks,
            txbs: &txbs,
            levels: &levels,
        };
        let workers = super::parallel::worker_count().min(jobs.len().max(1)).max(1);
        let next = AtomicUsize::new(0);
        let job_done: Vec<AtomicBool> = (0..jobs.len()).map(|_| AtomicBool::new(false)).collect();
        let first_err: std::sync::Mutex<Option<me_core::MediaError>> = std::sync::Mutex::new(None);
        let t_scope = std::time::Instant::now();
        std::thread::scope(|scope| {
            for _ in 0..workers {
                let shared = &shared;
                let jobs = &jobs;
                let order = &order;
                let next = &next;
                let job_done = &job_done;
                let first_err = &first_err;
                scope.spawn(move || {
                    // 每个 worker 自己的一份状态（`BlockCtx` 的那份归语法线程）
                    let mut sc = ReplayScratch::new(u32::from(shared.fh.params.quant.base_q_idx));
                    // 单条全局队列按 `2*band + col` 发牌（动态负载均衡）。
                    // ⚠️ 依赖必须**无条件**等 —— 包括带 0（它的 `deps[2]` 是"同带左邻列"）。
                    // 早先把它写在 `if job.band > 0` 里面，带 0 的列 1..N 全不等左邻列，
                    // 帧内预测读到还没重建的像素：首个错点正好是 x=128（超块列 1 的起点）。
                    loop {
                        let n = next.fetch_add(1, Ordering::AcqRel);
                        if n >= order.len() {
                            break;
                        }
                        let k = order[n] as usize;
                        let job = &jobs[k];
                        for dep in job.deps {
                            if dep == u32::MAX {
                                continue;
                            }
                            wait_until(|| job_done[dep as usize].load(Ordering::Acquire));
                        }
                        let trace_idx = flags::trace_rowneed();
                        for i in job.a..job.b {
                            if trace_idx {
                                CUR_BLOCK_IDX.store(i, Ordering::Relaxed);
                            }
                            if let Err(e) = replay_one(shared, &shared.blocks[i], &mut sc) {
                                let mut g = first_err.lock().unwrap();
                                if g.is_none() {
                                    *g = Some(e);
                                }
                            }
                        }
                        job_done[k].store(true, Ordering::Release);
                    }
                });
            }
        });
        if flags::trace_wait() {
            let (ns, n) = wait_stats_take();
            eprintln!(
                "WAIT workers={workers} bands={} jobs={} blocks={} waits={n} \
                 wait_total={:.1}ms scope={:.1}ms",
                ranges.len(),
                jobs.len(),
                blocks.len(),
                ns as f64 / 1e6,
                t_scope.elapsed().as_secs_f64() * 1e3
            );
        }
        self.row_mt.blocks = blocks;
        self.row_mt.txbs = txbs;
        self.row_mt.levels = levels;
        self.row_mt.reset();
        match first_err.into_inner().unwrap() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn residual(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        info: &BlockInfo,
        job_idx: Option<usize>,
    ) -> Result<()> {
        let (r, c) = (info.mi_row as i32, info.mi_col as i32);
        let has_chroma = self.has_chroma(r, c, info.mi_size);
        // §7.12.3 的 `get_qindex( 0, segment_id )`
        let qp = &self.fh.params.quant;
        let seg = &self.fh.params.segmentation;
        let qindex = if seg.feature_active(info.segment_id as usize, SEG_LVL_ALT_Q) {
            let data = seg.feature_data[info.segment_id as usize][SEG_LVL_ALT_Q];
            if self.fh.params.delta_q.present {
                self.current_q_index + data
            } else {
                qp.base_q_idx as i32 + data
            }
        } else if self.fh.params.delta_q.present {
            self.current_q_index
        } else {
            qp.base_q_idx as i32
        };
        let qindex = qindex.clamp(0, 255);
        let bit_depth = self.seq.color.bit_depth;
        // 规范的不对称：`dc` 三个平面各加自己的 `DeltaQ*Dc`，
        // 而 `ac` 的 plane 0 **不加任何 delta**（§7.12.3 `get_ac_quant`）。
        let dc_quant = [
            super::transform::dc_q(qindex + qp.delta_q_y_dc, bit_depth),
            super::transform::dc_q(qindex + qp.delta_q_u_dc, bit_depth),
            super::transform::dc_q(qindex + qp.delta_q_v_dc, bit_depth),
        ];
        let ac_quant = [
            super::transform::ac_q(qindex, bit_depth),
            super::transform::ac_q(qindex + qp.delta_q_u_ac, bit_depth),
            super::transform::ac_q(qindex + qp.delta_q_v_ac, bit_depth),
        ];
        // §6.10.4：AvailU / AvailL / AvailUChroma / AvailLChroma
        let avail_u = self.is_inside(r - 1, c);
        let avail_l = self.is_inside(r, c - 1);
        let (mut avail_u_chroma, mut avail_l_chroma) = (avail_u, avail_l);
        if !has_chroma {
            avail_u_chroma = false;
            avail_l_chroma = false;
        } else {
            let bw4 = NUM_4X4_BLOCKS_WIDE[info.mi_size as usize] as i32;
            let bh4 = NUM_4X4_BLOCKS_HIGH[info.mi_size as usize] as i32;
            if self.sub_y() && bh4 == 1 {
                avail_u_chroma = self.is_inside(r - 2, c);
            }
            if self.sub_x() && bw4 == 1 {
                avail_l_chroma = self.is_inside(r, c - 2);
            }
        }
        let filter_type_y = self.filter_type_for(0, r, c, info.mi_size);
        let filter_type_uv = self.filter_type_for(1, r, c, info.mi_size);
        let cfg = ResidualConfig {
            mi_rows: self.fh.mi_rows as i32,
            mi_cols: self.fh.mi_cols as i32,
            subsampling_x: self.sub_x(),
            subsampling_y: self.sub_y(),
            num_planes: self.num_planes(),
            has_chroma,
            mi_size: info.mi_size,
            lossless: self.lossless_array[info.segment_id as usize],
            is_inter: info.is_inter,
            skip: info.skip,
            tx_size: info.tx_size,
            mi_row: info.mi_row as i32,
            mi_col: info.mi_col as i32,
            y_mode: info.y_mode,
            uv_mode: info.uv_mode,
            angle_delta_y: info.angle_delta_y,
            angle_delta_uv: info.angle_delta_uv,
            cfl_alpha_u: info.cfl_alpha_u,
            cfl_alpha_v: info.cfl_alpha_v,
            use_filter_intra: info.use_filter_intra,
            filter_intra_mode: info.filter_intra_mode,
            filter_type_y,
            filter_type_uv,
            enable_intra_edge_filter: self.seq.enable_intra_edge_filter,
            avail_u,
            avail_l,
            avail_u_chroma,
            avail_l_chroma,
            reduced_tx_set: self.fh.params.reduced_tx_set,
            qindex_nonzero: self.current_q_index > 0,
            dc_quant,
            ac_quant,
            quant_bit_depth: bit_depth,
            use_128x128_superblock: self.seq.use_128x128_superblock,
        };
        // palette 块的预测要"查颜色表"，所以把本块的调色板与索引图一并借给残差侧。
        // 索引图行宽按**平面样本**算：亮度是块宽，色度是右移后的块宽（规范在
        // `palette_tokens` 里对 <4 的色度块还有 +2 的调整，这里必须一致）。
        let palette = if info.palette_size_y > 0 || info.palette_size_uv > 0 {
            let bw_l = block_width(info.mi_size) as usize;
            let mut uw = bw_l >> usize::from(self.sub_x());
            if uw < 4 {
                uw += 2;
            }
            Some(super::coeff::PaletteRef {
                size_y: info.palette_size_y as usize,
                size_uv: info.palette_size_uv as usize,
                colors_y: &self.palette_colors_y,
                colors_u: &self.palette_colors_u,
                colors_v: &self.palette_colors_v,
                map_y: &self.color_map_y,
                map_uv: &self.color_map_uv,
                stride_y: bw_l,
                stride_uv: uw,
            })
        } else {
            None
        };
        // row-mt 语法阶段：把 cfg 回填进工单（重建阶段要拿它反量化/反变换）
        if let Some(idx) = job_idx {
            self.row_mt.blocks[idx].cfg = Some(cfg);
        }
        let recording = self.row_mt.syntax_phase && !self.row_mt.inline_block;
        let mut ctx = ResidualCtx {
            cdfs: &mut self.coeff_cdfs,
            ctxs: &mut *self.coeff_ctxs,
            grid: super::coeff::GridAccess::Write(&mut *self.grid),
            stats: &mut self.stats.coeff,
            cfg,
            frame: self.frame.as_deref_mut(),
            max_luma_w: 0,
            max_luma_h: 0,
            trace_count: 0,
            palette,
            record: if recording { Some(&mut self.row_mt) } else { None },
            // 就地重建（非 row-mt / 调色板 flush）照旧读运行时的 `BlockDecoded` 网格
            flags: None,
        };
        ctx.residual(dec, &mut self.tx_scratch)
    }

    /// `get_filter_type( plane )`（§7.11.2.8）—— 边滤波的强弱看"上/左邻居是不是
    /// 用了 smooth 模式"。
    ///
    /// 色度那边规范有一套坐标换算（`MiCol & 1` / `MiRow & 1` 的奇偶调整），
    /// 因为色度 4×4 只覆盖亮度的一半；邻居是帧间块时 `is_smooth` 直接返回 0
    /// （`RefFrames[row][col][0] > INTRA_FRAME`）。
    fn filter_type_for(&self, plane: u8, mi_row: i32, mi_col: i32, bsize: u8) -> u8 {
        let (sub_x, sub_y) = (self.sub_x(), self.sub_y());
        let has_chroma = self.has_chroma(mi_row, mi_col, bsize);
        let avail_u = self.is_inside(mi_row - 1, mi_col);
        let avail_l = self.is_inside(mi_row, mi_col - 1);
        let (mut avail_u_chroma, mut avail_l_chroma) = (avail_u, avail_l);
        if !has_chroma {
            avail_u_chroma = false;
            avail_l_chroma = false;
        } else {
            let bw4 = NUM_4X4_BLOCKS_WIDE[bsize as usize] as i32;
            let bh4 = NUM_4X4_BLOCKS_HIGH[bsize as usize] as i32;
            if sub_y && bh4 == 1 {
                avail_u_chroma = self.is_inside(mi_row - 2, mi_col);
            }
            if sub_x && bw4 == 1 {
                avail_l_chroma = self.is_inside(mi_row, mi_col - 2);
            }
        }
        let mut above_smooth = false;
        if if plane == 0 { avail_u } else { avail_u_chroma } {
            let (mut rr, mut cc) = (mi_row - 1, mi_col);
            if plane > 0 {
                if sub_x && mi_col & 1 == 0 {
                    cc += 1;
                }
                if sub_y && mi_row & 1 == 1 {
                    rr -= 1;
                }
            }
            above_smooth = self.is_smooth_mode_at(rr, cc, plane);
        }
        let mut left_smooth = false;
        if if plane == 0 { avail_l } else { avail_l_chroma } {
            let (mut rr, mut cc) = (mi_row, mi_col - 1);
            if plane > 0 {
                if sub_x && mi_col & 1 == 1 {
                    cc -= 1;
                }
                if sub_y && mi_row & 1 == 0 {
                    rr += 1;
                }
            }
            left_smooth = self.is_smooth_mode_at(rr, cc, plane);
        }
        u8::from(above_smooth || left_smooth)
    }

    /// 规范的 `is_smooth( row, col, plane )`：那个位置上的预测模式是不是三种 smooth。
    fn is_smooth_mode_at(&self, r: i32, c: i32, plane: u8) -> bool {
        if r < 0 || c < 0 || r as usize >= self.grid.rows() || c as usize >= self.grid.cols() {
            return false;
        }
        let idx = self.grid.idx(r, c);
        if plane == 0 {
            super::intra::is_smooth_mode(self.grid.y_modes[idx])
        } else {
            if self.grid.ref_frames[idx][0] > mv::INTRA_FRAME {
                return false;
            }
            super::intra::is_smooth_mode(self.grid.uv_modes[idx])
        }
    }
}

/// `needs_interp_filter( )`（用 `BlockInfo` 的版本）
pub fn needs_interp_filter_here(
    fh: &FrameHeader,
    skip_mode: bool,
    info: &BlockInfo,
    bsize: u8,
) -> bool {
    let large = block_width(bsize).min(block_height(bsize)) >= 8;
    needs_interp_filter(
        fh,
        bsize,
        skip_mode,
        info.motion_mode,
        info.y_mode,
        info.ref_frame,
        large,
    )
}

// ── 一趟瓦片：分区树 + 块级语法 + exit_symbol ───────────────────────────

/// 把一个瓦片完整解一遍（规范 §5.11.1 `decode_tile` + §8.2 的 `exit_symbol`）。
///
/// `mi_grid` 是**帧级**的 `MiSizes`（tile 之间共享坐标系），`block_grid` 与
/// `coeff_ctxs` 也是帧级的。**结尾一定调 `exit_symbol()`** —— 这是熵层的最终硬验收。
///
/// `frame` 为 `Some` 时顺带走 §7.11.2 的帧内预测把像素写出来（M5）；为 `None`
/// 时只读语法 —— 真素材的语法验收（`real_av1_tests`）走的是后者。
#[allow(clippy::too_many_arguments)]
pub fn decode_tile(
    tile: &Tile<'_>,
    seq: &SequenceHeader,
    fh: &FrameHeader,
    store: &RefFrameStore,
    field: Option<&mv::MotionField>,
    cdf_ctx: &mut CdfContext,
    mi_grid: &mut MiGrid,
    block_grid: &mut BlockGrid,
    coeff_ctxs: &mut CoeffContexts,
    lr: &mut LrState,
    frame: Option<&mut CurrFrame>,
) -> Result<TileDecodeResult> {
    let ti = &fh.params.tile_info;
    let mi_rows = *ti.mi_row_starts.last().unwrap_or(&0) as i32;
    let mi_cols = *ti.mi_col_starts.last().unwrap_or(&0) as i32;
    if trace_last_sb() {
        eprintln!(
            "TILE mi_rows={mi_rows} mi_cols={mi_cols} | row_starts={:?} col_starts={:?} | \
             tile r={}..{} c={}..{} sb128={} bytes={} | base_q_idx={} lossless={} \
             restoration={} is_intra={} frame_type={:?}",
            ti.mi_row_starts,
            ti.mi_col_starts,
            tile.mi_row_start,
            tile.mi_row_end,
            tile.mi_col_start,
            tile.mi_col_end,
            seq.use_128x128_superblock,
            tile.data.len(),
            fh.params.quant.base_q_idx,
            fh.params.quant.using_qmatrix,
            fh.params.lr.uses_lr,
            fh.is_intra,
            fh.frame_type
        );
    }
    // 帧门控对拍时的瓦片标记（格式对齐 libaom 插桩的 `MARK tile frame=…`，
    // 只是换个前缀便于在同一份输出里区分两边）。
    if trace_sym_frame().is_some() && in_traced_frame() {
        println!(
            "OURMARK tile frame={} tile=({},{})",
            TRACE_TILE_FRAME_NO.load(std::sync::atomic::Ordering::Relaxed),
            tile.row,
            tile.col
        );
    }
    let sb_size = if seq.use_128x128_superblock {
        BLOCK_128X128
    } else {
        BLOCK_64X64
    };
    let sb4 = NUM_4X4_BLOCKS_WIDE[sb_size] as i32;

    // 规范 `decode_tile` 开头：`RefLrWiener` / `RefSgrXqd` 重置成 `_Mid`（§5.11.3）
    lr.begin_tile();

    let mut dec = SymbolDecoder::new(tile.data, !fh.disable_cdf_update)?;
    // CDF 是**帧级**的（跨瓦片共享、跨帧 save/load）：三个字段分开借，
    // 分区那份给 `PartitionCtx`，其余两份给 `BlockCtx`。
    let CdfContext { partition: pcdfs, block: block_cdfs, coeff: coeff_cdfs, lr: lr_cdfs } =
        cdf_ctx;
    // 规范 `decode_tile` 开头只有 `clear_above_context()`；Left 是**每个超块行**清的
    // （在 `decode_block` 里判到新超块行时清）
    coeff_ctxs.clear_above();

    let mut ctx = BlockCtx::new(
        block_grid, seq, fh, store, field, coeff_ctxs, tile, block_cdfs, coeff_cdfs, frame,
    );
    // ── row-mt（`ME_ROW_MT=1`）：语法阶段只记工单、重建阶段回放 ──────────
    //
    // 判据：开与不开的输出**逐字节一致**（阶段 1 顺序回放；阶段 2 再把回放并行）。
    let row_mt = flags::row_mt();
    if row_mt {
        ctx.row_mt.syntax_phase = true;
    }
    let mut hook =
        |dec: &mut SymbolDecoder<'_>, r: i32, c: i32, bsize: u8, g: &MiGrid| -> Result<()> {
            ctx.decode_block(dec, g, r, c, bsize)
        };
    let exit_ok;
    let bits_left;
    let uncovered;
    let overlaps;
    let partition;
    let mut sb_bits: Vec<u32> = Vec::new();
    let mut sb_leaves: Vec<u32> = Vec::new();
    let t_syn = std::time::Instant::now();
    {
        let mut pctx = partition::PartitionCtx {
            dec: &mut dec,
            cdfs: pcdfs,
            grid: mi_grid,
            mi_rows,
            mi_cols,
            mi_row_start: tile.mi_row_start as i32,
            mi_row_end: tile.mi_row_end as i32,
            mi_col_start: tile.mi_col_start as i32,
            mi_col_end: tile.mi_col_end as i32,
            stats: partition::PartitionStats::default(),
            leaf_hook: Some(&mut hook),
        };
        let stop_sb = stop_sb_limit();
        let mut sb_seen = 0usize;
        let mut r = tile.mi_row_start as i32;
        while r < tile.mi_row_end as i32 && !stop_sb.is_some_and(|n| sb_seen >= n) {
            let mut c = tile.mi_col_start as i32;
            while c < tile.mi_col_end as i32 && !stop_sb.is_some_and(|n| sb_seen >= n) {
                sb_seen += 1;
                let smb0 = pctx.dec.consumed_bits();
                let leaves0 = pctx.stats.leaves;
                // §5.11.48：环路恢复的参数**每个超块读一次**，且必须在
                // `decode_partition` 之前（它和块级语法共用同一份自适应 CDF）。
                // 帧级类型全是 `RESTORE_NONE` 时这里一位都不读。
                super::lr::read_lr(pctx.dec, lr, lr_cdfs, seq, fh, r, c, sb_size)?;
                pctx.decode_partition(r, c, sb_size, 0)?;
                sb_bits.push((pctx.dec.consumed_bits() - smb0) as u32);
                sb_leaves.push((pctx.stats.leaves - leaves0) as u32);
                if trace_sb() {
                    eprintln!(
                        "SB r={r} c={c} bits={} leaves={} smb={} txbs={} all_zero={}",
                        pctx.dec.consumed_bits() - smb0,
                        pctx.stats.leaves - leaves0,
                        pctx.dec.symbol_max_bits(),
                        pctx.dec.dbg_txbs,
                        pctx.dec.dbg_all_zero
                    );
                }
                c += sb4;
            }
            r += sb4;
        }
        // 覆盖检查：瓦片范围内每个 4×4 位置都被写过
        let mut unc = 0usize;
        for rr in tile.mi_row_start..tile.mi_row_end {
            for cc in tile.mi_col_start..tile.mi_col_end {
                if !pctx.grid.is_filled(rr as usize, cc as usize) {
                    unc += 1;
                }
            }
        }
        uncovered = unc;
        overlaps = pctx.grid.overlaps();
        bits_left = pctx.dec.symbol_max_bits();
        exit_ok = pctx.dec.exit().is_ok();
        partition = std::mem::take(&mut pctx.stats);
        // `pctx`（同时借了 `dec` / `mi_grid` / `hook`）到这里结束，
        // 下面才能取 `ctx` 的统计与 `dec` 的最终位置
    }
    let syn_ms = t_syn.elapsed().as_secs_f64() * 1e3;
    if flags::trace_synms() {
        eprintln!(
            "SYNMS row_mt={} syntax={:.1}ms blocks={}",
            u8::from(row_mt),
            syn_ms,
            ctx.row_mt.blocks.len()
        );
    }
    // ── row-mt 重建阶段（阶段 1：整个瓦片语法跑完后顺序回放）─────────────
    if row_mt {
        let (n_blocks, n_txbs, n_levels) =
            (ctx.row_mt.blocks.len(), ctx.row_mt.txbs.len(), ctx.row_mt.levels.len());
        ctx.row_mt.syntax_phase = false;
        let t_rp = std::time::Instant::now();
        if flags::row_mt_par() {
            // 实验：SB 行列波前并行（**已知有竞态**，见 `flags::row_mt_par`）
            ctx.replay_jobs_parallel(mi_grid)?;
        } else {
            // 默认：顺序回放（与现路径逐字节一致）
            ctx.replay_jobs(mi_grid)?;
        }
        if flags::trace_wait() {
            eprintln!(
                "RMSTAGES syntax={:.1}ms replay_total={:.1}ms blocks={n_blocks} txbs={n_txbs} \
                 levels={n_levels} block_bytes={} txb_bytes={} level_bytes={}",
                syn_ms,
                t_rp.elapsed().as_secs_f64() * 1e3,
                n_blocks * std::mem::size_of::<BlockJob>(),
                n_txbs * std::mem::size_of::<TxbJob>(),
                n_levels * std::mem::size_of::<i32>(),
            );
        }
    }
    let blocks = std::mem::take(&mut ctx.stats);
    let total_consumed = dec.consumed_bits();
    let partition_bits = total_consumed - blocks.bits as i64;
    Ok(TileDecodeResult {
        partition,
        blocks,
        exit_ok,
        bits_left,
        uncovered,
        overlaps,
        sb_bits,
        sb_leaves,
        partition_bits,
    })
}

/// 叶子回调的类型别名（`partition.rs` 的 `PartitionCtx::leaf_hook` 用它）。
pub type LeafHook<'a, 'b> =
    &'a mut dyn FnMut(&mut SymbolDecoder<'b>, i32, i32, u8, &MiGrid) -> Result<()>;

#[cfg(test)]
mod tests {
    use super::*;

    /// 假作业：只带 `diag` / `col`（发牌顺序只看这两个）。
    struct FakeJob {
        diag: u32,
        col: u32,
    }
    impl JobKey for FakeJob {
        fn diag(&self) -> u32 {
            self.diag
        }
        fn col_key(&self) -> u32 {
            self.col
        }
    }
    fn key(band: u32, col: u32) -> (u32, u32) {
        job_order_key(&FakeJob { diag: band + col, col })
    }

    /// **row-mt 发牌顺序必须是依赖的拓扑序** —— 否则 worker 会互相等成死锁
    /// （2026-10-02 实测 `ME_ROW_MT_FULLWAIT=1` 就这么卡死过一版）。
    ///
    /// 三条依赖边：同带左邻列、上一带同列、上一带次列。
    #[test]
    fn job_order_key_is_a_topological_order_of_the_dependencies() {
        for band in 1..9u32 {
            for col in 0..15u32 {
                let me = key(band, col);
                // ① 同带左邻列
                if col > 0 {
                    assert!(key(band, col - 1) < me, "同带左邻列 ({band},{})", col - 1);
                }
                // ② 上一带同列
                assert!(key(band - 1, col) < me, "上一带同列 ({},{col})", band - 1);
                // ③ 上一带次列（同一条对角线，靠"列号降序"排在前面）
                assert!(key(band - 1, col + 1) < me, "上一带次列 ({},{})", band - 1, col + 1);
            }
        }
    }


    #[test]
    fn block_size_tables_agree_with_the_spec_shape() {
        for (i, b) in [BLOCK_4X4, BLOCK_8X8, BLOCK_16X16, BLOCK_32X32, BLOCK_64X64, BLOCK_128X128]
            .into_iter()
            .enumerate()
        {
            assert_eq!(partition::MI_WIDTH_LOG2[b], i as u8);
            assert_eq!(partition::MI_HEIGHT_LOG2[b], i as u8);
            assert_eq!(block_width(b as u8), 4 << i);
            assert_eq!(block_height(b as u8), 4 << i);
            let tx = MAX_TX_SIZE_RECT[b];
            assert!(TX_WIDTH[tx as usize] as i32 <= block_width(b as u8));
            assert!(TX_HEIGHT[tx as usize] as i32 <= block_height(b as u8));
        }
        for g in SIZE_GROUP {
            assert!(g < BLOCK_SIZE_GROUPS);
        }
        for c in INTRA_MODE_CONTEXT {
            assert!(c < INTRA_MODE_CONTEXTS);
        }
        for w in WEDGE_BITS {
            assert!(w == 0 || w == 4);
        }
        assert_eq!(MAX_TX_DEPTH_BY_BSIZE[BLOCK_4X4], 0);
        assert_eq!(MAX_TX_DEPTH_BY_BSIZE[BLOCK_64X64], 4);
    }

    /// `neg_deinterleave` 必须是 `[0, max)` 上的置换（分段号靠它还原）。
    #[test]
    fn neg_deinterleave_is_a_permutation() {
        for max in 1..=8 {
            for reference in 0..max {
                let mut seen = vec![false; max as usize];
                for diff in 0..max {
                    let v = neg_deinterleave(diff, reference, max);
                    assert!(
                        v >= 0 && v < max,
                        "neg_deinterleave({diff}, {reference}, {max}) = {v} 越界"
                    );
                    assert!(!seen[v as usize], "重复：max={max} ref={reference} diff={diff}");
                    seen[v as usize] = true;
                }
            }
        }
    }

    #[test]
    fn ref_count_ctx_orders_counts() {
        assert_eq!(ref_count_ctx(0, 1), 0);
        assert_eq!(ref_count_ctx(1, 1), 1);
        assert_eq!(ref_count_ctx(2, 1), 2);
    }

    /// `comp_mode` / `comp_ref_type` 的上下文必须落在 CDF 表的范围内。
    #[test]
    fn comp_contexts_stay_in_range() {
        for avail_u in [false, true] {
            for avail_l in [false, true] {
                for above in [mv::INTRA_FRAME, mv::LAST_FRAME, mv::ALTREF_FRAME] {
                    for left in [mv::INTRA_FRAME, mv::LAST_FRAME, mv::ALTREF_FRAME] {
                        let ctx = comp_mode_ctx(
                            avail_u,
                            avail_l,
                            above,
                            left,
                            above <= mv::INTRA_FRAME,
                            left <= mv::INTRA_FRAME,
                            above > mv::INTRA_FRAME,
                            left > mv::INTRA_FRAME,
                        );
                        assert!(ctx < 5, "comp_mode ctx {ctx}");
                        let c = comp_ref_type_ctx(
                            avail_u,
                            avail_l,
                            above,
                            mv::ALTREF_FRAME,
                            left,
                            mv::BWDREF_FRAME,
                            above <= mv::INTRA_FRAME,
                            left <= mv::INTRA_FRAME,
                            above > mv::INTRA_FRAME,
                            left > mv::INTRA_FRAME,
                        );
                        assert!(c < 5, "comp_ref_type ctx {c}");
                    }
                }
            }
        }
    }

    /// `LosslessArray[s]`：只有"量化步长 0 且四个 delta 全 0"的段才是无损。
    #[test]
    fn lossless_array_needs_zero_qindex_and_no_deltas() {
        use super::super::frame_header::FrameHeader;
        let mut fh = FrameHeader::default();
        fh.params.quant.base_q_idx = 0;
        assert!(lossless_array(&fh)[0], "qindex 0 且没有 delta → 无损");
        fh.params.quant.base_q_idx = 1;
        assert!(!lossless_array(&fh)[0], "有量化步长就不是无损");
        fh.params.quant.base_q_idx = 0;
        fh.params.quant.delta_q_u_ac = 1;
        assert!(!lossless_array(&fh)[0], "有 delta 也不是无损");
    }

    /// `has_newmv` / `has_nearmv` 只有 `NEAR_NEWMV` / `NEW_NEARMV` 两个交集。
    #[test]
    fn has_newmv_and_has_nearmv_overlap_exactly_on_the_two_mixed_modes() {
        let mut both = Vec::new();
        for mode in NEARESTMV..=NEW_NEWMV {
            if has_newmv(mode) && has_nearmv(mode) {
                both.push(mode);
            }
        }
        assert_eq!(both, vec![mv::NEAR_NEWMV, mv::NEW_NEARMV]);
        assert!(has_newmv(NEWMV) && !has_nearmv(NEWMV));
        assert!(has_nearmv(NEARMV) && !has_newmv(NEARMV));
    }
}
