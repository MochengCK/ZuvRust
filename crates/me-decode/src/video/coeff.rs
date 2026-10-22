//! AV1 **系数解码**（规范 §5.11.35/§5.11.38/§5.11.39 与 §8.3.2 的上下文推导）
//! —— 自研软解 M4b 的后半。
//!
//! 这个模块和 [`super::block`]（模式信息）合起来才是完整的块级语法：
//!
//! ```text
//! decode_block
//!   ├─ mode_info()            ← block.rs
//!   ├─ read_block_tx_size()   ← block.rs
//!   ├─ compute_prediction()   ← M5（不读位）
//!   └─ residual()             ← 本文件
//!        ├─ transform_tree()  反复对半拆，直到某个尺寸能塞进 InterTxSizes
//!        └─ transform_block()
//!             ├─ transform_type()   读 tx_type（intra_tx_type / inter_tx_type）
//!             └─ coeffs()           读「这个变换块有没有系数、系数是多少」
//! ```
//!
//! ## `coeffs()` 读的东西（每一层都是算术符号）
//!
//! ```text
//! all_zero          S()            整块是不是全 0
//! eob_pt_{16..1024} S()            EOB 落在哪一档（按 eobMultisize 选 7 张表之一）
//! eob_extra         S() + L(1)×k   EOB 的档内偏移
//! ── 从 EOB-1 往回走到扫描序开头 ──
//! coeff_base_eob    S()            最后一个非零系数的幅度 - 1
//! coeff_base        S()            幅度（0..2 是"基"层）
//! coeff_br          S() ×循环      幅度 > 2 时的余量（每次最多 +3）
//! ── 再从头走到 EOB（只为符号与 Exp-Golomb 尾）──
//! dc_sign           S()            DC 的符号（一个符号承载 ±，不是"正/负"两个符号）
//! sign_bit          L(1)          其余系数的符号
//! golomb_length_bit L(1) ×循环     幅度 > 14 时的 Exp-Golomb
//! golomb_data_bit   L(1) ×k
//! ```
//!
//! ## 四处最容易写错的地方
//!
//! 1. **`all_zero` 无条件读**（没有 `plane == 0` 之类的额外条件）；
//!    而 `coeffs()` 本身在 `skip` 的块上**根本不会被调用**（`transform_block` 里
//!    的 `if ( !skip )`）。
//! 2. **上下文用的是 `Adjusted_Tx_Size`**（64 宽的变换按 32 用），
//!    所以 `row = pos >> bwl` 里的 `bwl` 取自调整后的尺寸，不是原尺寸。
//! 3. **`AboveLevelContext` / `LeftLevelContext` 的坐标系是"平面 4×4"**
//!    （色度要右移 `subsampling_x/y`），`maxX4`/`maxY4` 用的是**帧**的 `MiCols/MiRows`。
//!    `reset_block_context` 在 `skip` 的块上把这一块覆盖到的槽位清零。
//! 4. **`Sig_Ref_Diff_Offset` / `Mag_Ref_Offset_With_Tx_Class` 的偏移是"系数格"**
//!    （`row`/`col` 也是），别自己乘 4。
//!
//! 这一层的验收口径（写在这里，免得下次又去"调参数"）：**`exit_symbol()`**。
//! 按上面的语法把整条算术流读干净，末尾位置才落在编码器写下的 trailing bits 上。
//! 见 `me-cli` 的 `real_av1_tests`。

use super::flags;
use me_core::{MediaError, Result};

use super::block::{
    block_height, block_width, BLOCK_128X128, BLOCK_128X64, BLOCK_16X16, BLOCK_16X32, BLOCK_16X4,
    BLOCK_16X64, BLOCK_16X8, BLOCK_32X16, BLOCK_32X32, BLOCK_32X64, BLOCK_32X8, BLOCK_4X16,
    BLOCK_4X4, BLOCK_4X8, BLOCK_64X128, BLOCK_64X16, BLOCK_64X32, BLOCK_64X64, BLOCK_8X16,
    BLOCK_8X32, BLOCK_8X4, BLOCK_8X8, BLOCK_INVALID, BLOCK_SIZES, NUM_4X4_BLOCKS_HIGH,
    NUM_4X4_BLOCKS_WIDE, PALETTE_COLORS, TX_16X16, TX_16X32, TX_16X4, TX_16X64, TX_16X8,
    TX_32X16, TX_32X32, TX_32X8, TX_4X16, TX_4X4, TX_4X8, TX_64X16, TX_64X64, TX_8X16, TX_8X32,
    TX_8X4, TX_8X8, TX_HEIGHT, TX_HEIGHT_LOG2, TX_SIZES_ALL, TX_WIDTH, TX_WIDTH_LOG2, V_PRED,
};
use super::cdf_tables as T;
use super::entropy::SymbolDecoder;

// ── 规范 §3 的常量 ─────────────────────────────────────────────────────

/// `MI_SIZE`
pub const MI_SIZE: i32 = 4;
/// `MI_SIZE_LOG2`
pub const MI_SIZE_LOG2: i32 = 2;

/// `NUM_BASE_LEVELS`：幅度 0..2 由 `coeff_base` 直接表示
pub const NUM_BASE_LEVELS: u32 = 2;
/// `COEFF_BASE_RANGE`：`coeff_br` 覆盖的余量（每次 +3，最多 4 次）
pub const COEFF_BASE_RANGE: u32 = 12;
/// `BR_CDF_SIZE`：`coeff_br` 的取值个数
pub const BR_CDF_SIZE: u32 = 4;
/// `SIG_COEF_CONTEXTS`：`coeff_base` 的上下文数
pub const SIG_COEF_CONTEXTS: usize = 42;
/// `SIG_COEF_CONTEXTS_EOB`：`coeff_base_eob` 的上下文数
pub const SIG_COEF_CONTEXTS_EOB: usize = 4;
/// `SIG_COEF_CONTEXTS_2D`
pub const SIG_COEF_CONTEXTS_2D: usize = 26;
/// `SIG_REF_DIFF_OFFSET_NUM`
pub const SIG_REF_DIFF_OFFSET_NUM: usize = 5;
/// `LEVEL_CONTEXTS`
pub const LEVEL_CONTEXTS: usize = 21;
/// `TXB_SKIP_CONTEXTS`
pub const TXB_SKIP_CONTEXTS: usize = 13;
/// `EOB_COEF_CONTEXTS`
pub const EOB_COEF_CONTEXTS: usize = 9;
/// `DC_SIGN_CONTEXTS`
pub const DC_SIGN_CONTEXTS: usize = 3;
/// 一个变换块最多多少个系数（64 宽的按 32 算，所以是 1024）
pub const MAX_TXB_COEFFS: usize = 1024;

// ── 变换类型（规范 §6.10.4 / §9）──────────────────────────────────────────

pub const DCT_DCT: u8 = 0;
pub const ADST_DCT: u8 = 1;
pub const DCT_ADST: u8 = 2;
pub const ADST_ADST: u8 = 3;
pub const FLIPADST_DCT: u8 = 4;
pub const DCT_FLIPADST: u8 = 5;
pub const FLIPADST_FLIPADST: u8 = 6;
pub const ADST_FLIPADST: u8 = 7;
pub const FLIPADST_ADST: u8 = 8;
pub const IDTX: u8 = 9;
pub const V_DCT: u8 = 10;
pub const H_DCT: u8 = 11;
pub const V_ADST: u8 = 12;
pub const H_ADST: u8 = 13;
pub const V_FLIPADST: u8 = 14;
pub const H_FLIPADST: u8 = 15;
/// `TX_TYPES`
pub const TX_TYPES: usize = 16;

/// `get_tx_class( txType )`
pub const TX_CLASS_2D: u8 = 0;
pub const TX_CLASS_HORIZ: u8 = 1;
pub const TX_CLASS_VERT: u8 = 2;

/// `get_tx_set( txSz )` 的取值
pub const TX_SET_DCTONLY: u8 = 0;
pub const TX_SET_INTRA_1: u8 = 1;
pub const TX_SET_INTRA_2: u8 = 2;
pub const TX_SET_INTER_1: u8 = 1;
pub const TX_SET_INTER_2: u8 = 2;
pub const TX_SET_INTER_3: u8 = 3;

/// `Tx_Type_Intra_Inv_Set1[7]`
pub const TX_TYPE_INTRA_INV_SET1: [u8; 7] =
    [IDTX, DCT_DCT, V_DCT, H_DCT, ADST_ADST, ADST_DCT, DCT_ADST];
/// `Tx_Type_Intra_Inv_Set2[5]`
pub const TX_TYPE_INTRA_INV_SET2: [u8; 5] = [IDTX, DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST];
/// `Tx_Type_Inter_Inv_Set1[16]`
pub const TX_TYPE_INTER_INV_SET1: [u8; 16] = [
    IDTX, V_DCT, H_DCT, V_ADST, H_ADST, V_FLIPADST, H_FLIPADST, DCT_DCT, ADST_DCT, DCT_ADST,
    FLIPADST_DCT, DCT_FLIPADST, ADST_ADST, FLIPADST_FLIPADST, ADST_FLIPADST, FLIPADST_ADST,
];
/// `Tx_Type_Inter_Inv_Set2[12]`
pub const TX_TYPE_INTER_INV_SET2: [u8; 12] = [
    IDTX, V_DCT, H_DCT, DCT_DCT, ADST_DCT, DCT_ADST, FLIPADST_DCT, DCT_FLIPADST, ADST_ADST,
    FLIPADST_FLIPADST, ADST_FLIPADST, FLIPADST_ADST,
];
/// `Tx_Type_Inter_Inv_Set3[2]`
pub const TX_TYPE_INTER_INV_SET3: [u8; 2] = [IDTX, DCT_DCT];

const D: bool = false;
const Y: bool = true;

/// `Tx_Type_In_Set_Intra[ TX_SET_TYPES_INTRA ][ TX_TYPES ]`
pub const TX_TYPE_IN_SET_INTRA: [[bool; TX_TYPES]; 3] = [
    [Y, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D],
    [Y, Y, Y, Y, D, D, D, D, D, Y, Y, Y, D, D, D, D],
    [Y, Y, Y, Y, D, D, D, D, D, Y, D, D, D, D, D, D],
];
/// `Tx_Type_In_Set_Inter[ TX_SET_TYPES_INTER ][ TX_TYPES ]`
pub const TX_TYPE_IN_SET_INTER: [[bool; TX_TYPES]; 4] = [
    [Y, D, D, D, D, D, D, D, D, D, D, D, D, D, D, D],
    [Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y],
    [Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, D, D, D, D],
    [Y, D, D, D, D, D, D, D, D, Y, D, D, D, D, D, D],
];

/// `Mode_To_Txfm[ UV_INTRA_MODES_CFL_ALLOWED ]`
pub const MODE_TO_TXFM: [u8; 14] = [
    DCT_DCT,   // DC_PRED
    ADST_DCT,  // V_PRED
    DCT_ADST,  // H_PRED
    DCT_DCT,   // D45_PRED
    ADST_ADST, // D135_PRED
    ADST_DCT,  // D113_PRED
    DCT_ADST,  // D157_PRED
    DCT_ADST,  // D203_PRED
    ADST_DCT,  // D67_PRED
    ADST_ADST, // SMOOTH_PRED
    ADST_DCT,  // SMOOTH_V_PRED
    DCT_ADST,  // SMOOTH_H_PRED
    ADST_ADST, // PAETH_PRED
    DCT_DCT,   // UV_CFL_PRED
];

/// `Filter_Intra_Mode_To_Intra_Dir[ INTRA_FILTER_MODES ]`
pub const FILTER_INTRA_MODE_TO_INTRA_DIR: [u8; 5] = [0, V_PRED, 2, 6, 0];

// ── 规范 §9 的几何表 ────────────────────────────────────────────────────

/// `Tx_Size_Sqr[ TX_SIZES_ALL ]`
pub const TX_SIZE_SQR: [u8; TX_SIZES_ALL] = [
    TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_64X64, TX_4X4, TX_4X4, TX_8X8, TX_8X8, TX_16X16,
    TX_16X16, TX_32X32, TX_32X32, TX_4X4, TX_4X4, TX_8X8, TX_8X8, TX_16X16, TX_16X16,
];
/// `Tx_Size_Sqr_Up[ TX_SIZES_ALL ]`
pub const TX_SIZE_SQR_UP: [u8; TX_SIZES_ALL] = [
    TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_64X64, TX_8X8, TX_8X8, TX_16X16, TX_16X16, TX_32X32,
    TX_32X32, TX_64X64, TX_64X64, TX_16X16, TX_16X16, TX_32X32, TX_32X32, TX_64X64, TX_64X64,
];
/// `Adjusted_Tx_Size[ TX_SIZES_ALL ]`
pub const ADJUSTED_TX_SIZE: [u8; TX_SIZES_ALL] = [
    TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_32X32, TX_4X8, TX_8X4, TX_8X16, TX_16X8, TX_16X32,
    TX_32X16, TX_32X32, TX_32X32, TX_4X16, TX_16X4, TX_8X32, TX_32X8, TX_16X32, TX_32X16,
];

/// `Subsampled_Size[ BLOCK_SIZES ][ 2 ][ 2 ]`（`BLOCK_INVALID` = 不适用）
pub const SUBSAMPLED_SIZE: [[[u8; 2]; 2]; BLOCK_SIZES] = {
    const I: u8 = BLOCK_INVALID;
    [
        [[BLOCK_4X4 as u8, BLOCK_4X4 as u8], [BLOCK_4X4 as u8, BLOCK_4X4 as u8]],
        [[BLOCK_4X8 as u8, BLOCK_4X4 as u8], [I, BLOCK_4X4 as u8]],
        [[BLOCK_8X4 as u8, I], [BLOCK_4X4 as u8, BLOCK_4X4 as u8]],
        [[BLOCK_8X8 as u8, BLOCK_8X4 as u8], [BLOCK_4X8 as u8, BLOCK_4X4 as u8]],
        [[BLOCK_8X16 as u8, BLOCK_8X8 as u8], [I, BLOCK_4X8 as u8]],
        [[BLOCK_16X8 as u8, I], [BLOCK_8X8 as u8, BLOCK_8X4 as u8]],
        [[BLOCK_16X16 as u8, BLOCK_16X8 as u8], [BLOCK_8X16 as u8, BLOCK_8X8 as u8]],
        [[BLOCK_16X32 as u8, BLOCK_16X16 as u8], [I, BLOCK_8X16 as u8]],
        [[BLOCK_32X16 as u8, I], [BLOCK_16X16 as u8, BLOCK_16X8 as u8]],
        [[BLOCK_32X32 as u8, BLOCK_32X16 as u8], [BLOCK_16X32 as u8, BLOCK_16X16 as u8]],
        [[BLOCK_32X64 as u8, BLOCK_32X32 as u8], [I, BLOCK_16X32 as u8]],
        [[BLOCK_64X32 as u8, I], [BLOCK_32X32 as u8, BLOCK_32X16 as u8]],
        [[BLOCK_64X64 as u8, BLOCK_64X32 as u8], [BLOCK_32X64 as u8, BLOCK_32X32 as u8]],
        [[BLOCK_64X128 as u8, BLOCK_64X64 as u8], [I, BLOCK_32X64 as u8]],
        [[BLOCK_128X64 as u8, I], [BLOCK_64X64 as u8, BLOCK_64X32 as u8]],
        [[BLOCK_128X128 as u8, BLOCK_128X64 as u8], [BLOCK_64X128 as u8, BLOCK_64X64 as u8]],
        [[BLOCK_4X16 as u8, BLOCK_4X8 as u8], [I, BLOCK_4X8 as u8]],
        [[BLOCK_16X4 as u8, I], [BLOCK_8X4 as u8, BLOCK_8X4 as u8]],
        [[BLOCK_8X32 as u8, BLOCK_8X16 as u8], [I, BLOCK_4X16 as u8]],
        [[BLOCK_32X8 as u8, I], [BLOCK_16X8 as u8, BLOCK_16X4 as u8]],
        [[BLOCK_16X64 as u8, BLOCK_16X32 as u8], [I, BLOCK_8X32 as u8]],
        [[BLOCK_64X16 as u8, I], [BLOCK_32X16 as u8, BLOCK_32X8 as u8]],
    ]
};

/// `Sig_Ref_Diff_Offset[ 3 ][ SIG_REF_DIFF_OFFSET_NUM ][ 2 ]`
pub const SIG_REF_DIFF_OFFSET: [[[i32; 2]; SIG_REF_DIFF_OFFSET_NUM]; 3] = [
    [[0, 1], [1, 0], [1, 1], [0, 2], [2, 0]],
    [[0, 1], [1, 0], [0, 2], [0, 3], [0, 4]],
    [[0, 1], [1, 0], [2, 0], [3, 0], [4, 0]],
];

/// `Coeff_Base_Ctx_Offset[ TX_SIZES_ALL ][ 5 ][ 5 ]`
pub const COEFF_BASE_CTX_OFFSET: [[[u8; 5]; 5]; TX_SIZES_ALL] = [
    [[0, 1, 6, 6, 0], [1, 6, 6, 21, 0], [6, 6, 21, 21, 0], [6, 21, 21, 21, 0], [0, 0, 0, 0, 0]],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [[0, 11, 11, 11, 0], [11, 11, 11, 11, 0], [6, 6, 21, 21, 0], [6, 21, 21, 21, 0], [21, 21, 21, 21, 0]],
    [[0, 16, 6, 6, 21], [16, 16, 6, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21], [0, 0, 0, 0, 0]],
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [[0, 16, 6, 6, 21], [16, 16, 6, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21]],
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [[0, 16, 6, 6, 21], [16, 16, 6, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21]],
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [[0, 16, 6, 6, 21], [16, 16, 6, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21]],
    [[0, 11, 11, 11, 0], [11, 11, 11, 11, 0], [6, 6, 21, 21, 0], [6, 21, 21, 21, 0], [21, 21, 21, 21, 0]],
    [[0, 16, 6, 6, 21], [16, 16, 6, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21], [0, 0, 0, 0, 0]],
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [[0, 16, 6, 6, 21], [16, 16, 6, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21]],
    [
        [0, 11, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [[0, 16, 6, 6, 21], [16, 16, 6, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21], [16, 16, 21, 21, 21]],
];

/// `Coeff_Base_Pos_Ctx_Offset[ 3 ]`
pub const COEFF_BASE_POS_CTX_OFFSET: [u8; 3] = [26, 31, 36];

/// `Mag_Ref_Offset_With_Tx_Class[ 3 ][ 3 ][ 2 ]`
pub const MAG_REF_OFFSET_WITH_TX_CLASS: [[[i32; 2]; 3]; 3] =
    [[[0, 1], [1, 0], [1, 1]], [[0, 1], [1, 0], [0, 2]], [[0, 1], [1, 0], [2, 0]]];

// ── 扫描顺序 ────────────────────────────────────────────────────────────

/// `get_default_scan( txSz )`
pub fn default_scan(tx_sz: u8) -> &'static [u16] {
    match tx_sz {
        TX_4X4 => &T::DEFAULT_SCAN_4X4,
        TX_4X8 => &T::DEFAULT_SCAN_4X8,
        TX_8X4 => &T::DEFAULT_SCAN_8X4,
        TX_8X8 => &T::DEFAULT_SCAN_8X8,
        TX_8X16 => &T::DEFAULT_SCAN_8X16,
        TX_16X8 => &T::DEFAULT_SCAN_16X8,
        TX_16X16 => &T::DEFAULT_SCAN_16X16,
        TX_16X32 => &T::DEFAULT_SCAN_16X32,
        TX_32X16 => &T::DEFAULT_SCAN_32X16,
        TX_4X16 => &T::DEFAULT_SCAN_4X16,
        TX_16X4 => &T::DEFAULT_SCAN_16X4,
        TX_8X32 => &T::DEFAULT_SCAN_8X32,
        TX_32X8 => &T::DEFAULT_SCAN_32X8,
        _ => &T::DEFAULT_SCAN_32X32,
    }
}

/// `get_mrow_scan( txSz )`：行优先（垂直变换用）。规范的最后一档是 `16x4`。
pub fn mrow_scan(tx_sz: u8) -> &'static [u16] {
    match tx_sz {
        TX_4X4 => &T::MROW_SCAN_4X4,
        TX_4X8 => &T::MROW_SCAN_4X8,
        TX_8X4 => &T::MROW_SCAN_8X4,
        TX_8X8 => &T::MROW_SCAN_8X8,
        TX_8X16 => &T::MROW_SCAN_8X16,
        TX_16X8 => &T::MROW_SCAN_16X8,
        TX_16X16 => &T::MROW_SCAN_16X16,
        TX_4X16 => &T::MROW_SCAN_4X16,
        _ => &T::MROW_SCAN_16X4,
    }
}

/// `get_mcol_scan( txSz )`：列优先（水平变换用）。
pub fn mcol_scan(tx_sz: u8) -> &'static [u16] {
    match tx_sz {
        TX_4X4 => &T::MCOL_SCAN_4X4,
        TX_4X8 => &T::MCOL_SCAN_4X8,
        TX_8X4 => &T::MCOL_SCAN_8X4,
        TX_8X8 => &T::MCOL_SCAN_8X8,
        TX_8X16 => &T::MCOL_SCAN_8X16,
        TX_16X8 => &T::MCOL_SCAN_16X8,
        TX_16X16 => &T::MCOL_SCAN_16X16,
        TX_4X16 => &T::MCOL_SCAN_4X16,
        _ => &T::MCOL_SCAN_16X4,
    }
}

/// `get_scan( txSz )`：`PlaneTxType` 决定用哪张扫描表。
pub fn scan_for(tx_sz: u8, plane_tx_type: u8) -> &'static [u16] {
    if tx_sz == TX_16X64 {
        return &T::DEFAULT_SCAN_16X32;
    }
    if tx_sz == TX_64X16 {
        return &T::DEFAULT_SCAN_32X16;
    }
    if TX_SIZE_SQR_UP[tx_sz as usize] == TX_64X64 {
        return &T::DEFAULT_SCAN_32X32;
    }
    if plane_tx_type == IDTX {
        return default_scan(tx_sz);
    }
    if matches!(plane_tx_type, V_DCT | V_ADST | V_FLIPADST) {
        mrow_scan(tx_sz)
    } else if matches!(plane_tx_type, H_DCT | H_ADST | H_FLIPADST) {
        mcol_scan(tx_sz)
    } else {
        default_scan(tx_sz)
    }
}

/// `get_tx_class( txType )`
pub fn tx_class(tx_type: u8) -> u8 {
    match tx_type {
        V_DCT | V_ADST | V_FLIPADST => TX_CLASS_VERT,
        H_DCT | H_ADST | H_FLIPADST => TX_CLASS_HORIZ,
        _ => TX_CLASS_2D,
    }
}

/// `get_plane_residual_size( subsize, plane )`（规范 §5.11.35）。
///
/// 极端组合（4:2:2 下的 4 宽块）规范表里是 `BLOCK_INVALID`；这里**兜底成 4×4**，
/// 免得索引越界。那种组合在 `decode_block` 的 `HasChroma` 规则下本来就不该出现，
/// 真出现了也是"这个块没有色度"。
pub fn plane_residual_size(bsize: u8, plane: u8, sub_x: bool, sub_y: bool) -> u8 {
    if bsize as usize >= BLOCK_SIZES {
        return bsize;
    }
    let sx = usize::from(plane > 0 && sub_x);
    let sy = usize::from(plane > 0 && sub_y);
    let sz = SUBSAMPLED_SIZE[bsize as usize][sx][sy];
    if sz as usize >= BLOCK_SIZES {
        BLOCK_4X4 as u8
    } else {
        sz
    }
}

/// `find_tx_size( w, h )`
pub fn find_tx_size(w: i32, h: i32) -> u8 {
    for tx_sz in 0..TX_SIZES_ALL as u8 {
        if TX_WIDTH[tx_sz as usize] as i32 == w && TX_HEIGHT[tx_sz as usize] as i32 == h {
            return tx_sz;
        }
    }
    TX_64X64
}

/// `get_tx_size( plane, txSz )`（规范 §5.11.35）
pub fn plane_tx_size(bsize: u8, plane: u8, tx_sz: u8, sub_x: bool, sub_y: bool) -> u8 {
    if plane == 0 {
        return tx_sz;
    }
    let uv_tx = super::block::MAX_TX_SIZE_RECT
        [plane_residual_size(bsize, plane, sub_x, sub_y) as usize];
    if TX_WIDTH[uv_tx as usize] == 64 || TX_HEIGHT[uv_tx as usize] == 64 {
        if TX_WIDTH[uv_tx as usize] == 16 {
            return TX_16X32;
        }
        if TX_HEIGHT[uv_tx as usize] == 16 {
            return TX_32X16;
        }
        return TX_32X32;
    }
    uv_tx
}

// ── 帧级的 `TxTypes` 网格 ───────────────────────────────────────────────

/// `TxTypes[ MiRows ][ MiCols ]`：每个**亮度 4×4**位置的变换类型。
///
/// 由 `transform_type()` 写、由 `compute_tx_type()` 读（帧间块的**色度**要看
/// 同位置亮度的 `TxTypes` 才能决定自己的变换类型）。
#[derive(Debug, Clone)]
pub struct TxTypeGrid {
    rows: usize,
    cols: usize,
    types: Vec<u8>,
}

impl TxTypeGrid {
    /// 空壳（配 [`TxTypeGrid::reset_all`] 用）。
    pub fn empty() -> Self {
        Self { rows: 0, cols: 0, types: Vec::new() }
    }

    pub fn new(rows: usize, cols: usize) -> Self {
        let mut g = Self::empty();
        g.reset_all(rows, cols);
        g
    }

    /// 按 `rows × cols` 重填（跨帧复用同一张网格时用；语义与 `new` 一致）。
    pub fn reset_all(&mut self, rows: usize, cols: usize) {
        self.rows = rows;
        self.cols = cols;
        self.types.clear();
        self.types.resize(rows * cols, DCT_DCT);
    }

    #[inline]
    pub fn get(&self, r: i32, c: i32) -> u8 {
        if r < 0 || c < 0 || r as usize >= self.rows || c as usize >= self.cols {
            return DCT_DCT;
        }
        self.types[r as usize * self.cols + c as usize]
    }

    #[inline]
    pub fn set(&mut self, r: i32, c: i32, v: u8) {
        if r < 0 || c < 0 || r as usize >= self.rows || c as usize >= self.cols {
            return;
        }
        self.types[r as usize * self.cols + c as usize] = v;
    }
}

// ── 帧级的系数上下文 ────────────────────────────────────────────────────

/// `AboveLevelContext` / `AboveDcContext` / `LeftLevelContext` / `LeftDcContext`
/// —— **帧级**、按平面分开、按"平面 4×4"索引。
#[derive(Debug, Clone)]
pub struct CoeffContexts {
    pub above_level: [Vec<u8>; 3],
    pub above_dc: [Vec<u8>; 3],
    pub left_level: [Vec<u8>; 3],
    pub left_dc: [Vec<u8>; 3],
}

impl CoeffContexts {
    pub fn new(mi_rows: usize, mi_cols: usize, sub_x: bool, sub_y: bool, num_planes: u8) -> Self {
        let make = |n: usize| [vec![0u8; n], vec![0u8; n], vec![0u8; n]];
        let mut above = make(mi_cols);
        let mut left = make(mi_rows);
        if num_planes > 1 {
            if sub_x {
                for p in 1..3 {
                    above[p] = vec![0u8; mi_cols.div_ceil(2)];
                }
            }
            if sub_y {
                for p in 1..3 {
                    left[p] = vec![0u8; mi_rows.div_ceil(2)];
                }
            }
        }
        Self {
            above_level: above.clone(),
            above_dc: above,
            left_level: left.clone(),
            left_dc: left,
        }
    }

    /// `clear_above_context( )`（规范 §5.11.1 开头）：**只清 Above**。
    ///
    /// 规范在 `decode_tile` 的**每个超块行**开头调 `clear_left_context()`，所以
    /// Left 是"逐行"清的、Above 只在瓦片开头清一次 —— 两者别揉成一个。
    pub fn clear_above(&mut self) {
        for p in 0..3 {
            self.above_level[p].fill(0);
            self.above_dc[p].fill(0);
        }
    }

    /// `clear_left_context( )`：每个**超块行**开头清一次。
    pub fn clear_left(&mut self) {
        for p in 0..3 {
            self.left_level[p].fill(0);
            self.left_dc[p].fill(0);
        }
    }

    /// 整张表清零（测试/兜底用）。
    pub fn clear(&mut self) {
        self.clear_above();
        self.clear_left();
    }

    /// `reset_block_context( bw4, bh4 )`（规范 §5.11.4 尾部）。
    ///
    /// 越界的槽位直接跳过 —— 规范里的循环会写到帧外（块可以伸出去），
    /// 而那些槽位永远读不到（读的时候也有 `maxX4/maxY4` 判断）。
    pub fn reset_block(
        &mut self,
        mi_row: i32,
        mi_col: i32,
        bw4: i32,
        bh4: i32,
        sub_x: bool,
        sub_y: bool,
        num_planes: u8,
    ) {
        // 与 libaom `av1_reset_entropy_context` 对齐的两条规则：
        //
        // ① 色度只在**色度参考块**上清 —— `is_chroma_reference(mi_row, mi_col, bsize, ss_x, ss_y)`：
        //    4:2:0 里 4×4 块只有 `mi_row`/`mi_col` 都为奇数（8×8 组的右下角）时才带色度。
        // ② 清的范围照规范原文：**`i = MiCol >> subX` 到 `( MiCol + bw4 ) >> subX`（左闭右开）**，
        //    也就是"**下界取整、上界取整**"，个数 = `((MiCol + bw4) >> subX) - (MiCol >> subX)`。
        //
        // ⚠️ 2026-10-01 修的**真 bug**（真素材第 34 帧逐符号对拍抓到的）：原来写成
        //    `if (!sub_x || bw4 % 2 == 0) { bw4 >> subX } else { 0 }` —— 即"宽度是奇数就一格都不清"。
        //    但对 `bw4` 为奇数、`MiCol` 也为奇数的块（如 4×4 跳过块落在 `mi_col = 311`），
        //    规范那两条边界算出来是 `[155, 156)` = **1 格**，我们却清了 0 格 ⇒ 该位置残留上一个
        //    变换块写下的 `level`（实测残留 1）⇒ 后面同一列上 `txb_skip` 的 `skip_ctx` 从 7 变成 8
        //    ⇒ CDF 选错 ⇒ 从那里起整条算术流错位（症状是"前 8841 个变换块逐符号全对，第 8842 个
        //    的 `txb_skip` 符号值相同但 CDF 不同"）。libaom 侧同位置会打
        //    `RSTL i=2 bsize=0 mi=(99,311) col=155 n=1`，是这条判据的铁证。
        let chroma_ref = ((mi_row & 1) != 0 || (bh4 & 1) == 0 || !sub_y)
            && ((mi_col & 1) != 0 || (bw4 & 1) == 0 || !sub_x);
        let planes = 1 + if num_planes > 1 && chroma_ref { 2 } else { 0 };
        for plane in 0..planes {
            let sx = usize::from(plane > 0 && sub_x);
            let sy = usize::from(plane > 0 && sub_y);
            let c0 = (mi_col >> sx).max(0);
            let c1 = ((mi_col + bw4) >> sx).max(0);
            let r0 = (mi_row >> sy).max(0);
            let r1 = ((mi_row + bh4) >> sy).max(0);
            if flags::trace_ctxw() {
                eprintln!(
                    "CTXW rst p={plane} dir=A idx={c0} n={} mi=({mi_row},{mi_col}) bw4={bw4} bh4={bh4}",
                    c1 - c0
                );
                eprintln!(
                    "CTXW rst p={plane} dir=L idx={r0} n={} mi=({mi_row},{mi_col}) bw4={bw4} bh4={bh4}",
                    r1 - r0
                );
            }
            for i in c0..c1 {
                if let Some(slot) = self.above_level[plane].get_mut(i as usize) {
                    *slot = 0;
                }
                if let Some(slot) = self.above_dc[plane].get_mut(i as usize) {
                    *slot = 0;
                }
            }
            for i in r0..r1 {
                if let Some(slot) = self.left_level[plane].get_mut(i as usize) {
                    *slot = 0;
                }
                if let Some(slot) = self.left_dc[plane].get_mut(i as usize) {
                    *slot = 0;
                }
            }
        }
    }
}

// ── 瓦片级的系数 CDF ────────────────────────────────────────────────────

/// 诊断（`AV1_TRACE_TB_COORD=1`）：逐变换块打印起点与帧边界。
fn trace_tb_coord() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("AV1_TRACE_TB_COORD").is_ok())
}

/// 一个瓦片的系数 CDF 副本（`init_coeff_cdfs`：按 `base_q_idx` 选一个
/// `COEFF_CDF_Q_CTXS` 档位，再拷一份出来就地自适应）。
#[derive(Debug, Clone)]
pub struct CoeffCdfs {
    pub txb_skip: [[[u16; 3]; TXB_SKIP_CONTEXTS]; 5],
    pub eob_pt_16: [[[u16; 6]; 2]; 2],
    pub eob_pt_32: [[[u16; 7]; 2]; 2],
    pub eob_pt_64: [[[u16; 8]; 2]; 2],
    pub eob_pt_128: [[[u16; 9]; 2]; 2],
    pub eob_pt_256: [[[u16; 10]; 2]; 2],
    pub eob_pt_512: [[u16; 11]; 2],
    pub eob_pt_1024: [[u16; 12]; 2],
    pub eob_extra: [[[[u16; 3]; EOB_COEF_CONTEXTS]; 2]; 5],
    pub dc_sign: [[[u16; 3]; DC_SIGN_CONTEXTS]; 2],
    pub coeff_base_eob: [[[[u16; 4]; SIG_COEF_CONTEXTS_EOB]; 2]; 5],
    pub coeff_base: [[[[u16; 5]; SIG_COEF_CONTEXTS]; 2]; 5],
    pub coeff_br: [[[[u16; 5]; LEVEL_CONTEXTS]; 2]; 5],
    // 变换类型（同样是瓦片级）
    pub intra_tx_type_set1: [[[u16; 8]; 13]; 2],
    pub intra_tx_type_set2: [[[u16; 6]; 13]; 3],
    pub inter_tx_type_set1: [[u16; 17]; 2],
    pub inter_tx_type_set2: [u16; 13],
    pub inter_tx_type_set3: [[u16; 3]; 4],
}

/// `COEFF_CDF_Q_CTXS`
pub const COEFF_CDF_Q_CTXS: usize = 4;

/// `init_coeff_cdfs` 用的档位选择。
pub fn coeff_cdf_q_idx(base_q_idx: u32) -> usize {
    if base_q_idx <= 20 {
        0
    } else if base_q_idx <= 60 {
        1
    } else if base_q_idx <= 120 {
        2
    } else {
        3
    }
}

impl Default for CoeffCdfs {
    fn default() -> Self {
        Self::new(0)
    }
}

impl CoeffCdfs {
    /// `init_coeff_cdfs( )`
    pub fn new(base_q_idx: u32) -> Self {
        let idx = coeff_cdf_q_idx(base_q_idx).min(COEFF_CDF_Q_CTXS - 1);
        Self {
            txb_skip: T::DEFAULT_TXB_SKIP_CDF[idx],
            eob_pt_16: T::DEFAULT_EOB_PT_16_CDF[idx],
            eob_pt_32: T::DEFAULT_EOB_PT_32_CDF[idx],
            eob_pt_64: T::DEFAULT_EOB_PT_64_CDF[idx],
            eob_pt_128: T::DEFAULT_EOB_PT_128_CDF[idx],
            eob_pt_256: T::DEFAULT_EOB_PT_256_CDF[idx],
            eob_pt_512: T::DEFAULT_EOB_PT_512_CDF[idx],
            eob_pt_1024: T::DEFAULT_EOB_PT_1024_CDF[idx],
            eob_extra: T::DEFAULT_EOB_EXTRA_CDF[idx],
            dc_sign: T::DEFAULT_DC_SIGN_CDF[idx],
            coeff_base_eob: T::DEFAULT_COEFF_BASE_EOB_CDF[idx],
            coeff_base: T::DEFAULT_COEFF_BASE_CDF[idx],
            coeff_br: T::DEFAULT_COEFF_BR_CDF[idx],
            intra_tx_type_set1: T::DEFAULT_INTRA_TX_TYPE_SET1_CDF,
            intra_tx_type_set2: T::DEFAULT_INTRA_TX_TYPE_SET2_CDF,
            inter_tx_type_set1: T::DEFAULT_INTER_TX_TYPE_SET1_CDF,
            inter_tx_type_set2: T::DEFAULT_INTER_TX_TYPE_SET2_CDF,
            inter_tx_type_set3: T::DEFAULT_INTER_TX_TYPE_SET3_CDF,
        }
    }
}

// ── 统计（诊断与验收）──────────────────────────────────────────────────

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CoeffStats {
    /// `coeffs()` 调用次数（走过的变换块数）
    pub txbs: u64,
    pub symbols: u64,
    pub literals: u64,
    /// `all_zero == 1` 的次数
    pub all_zero: u64,
    pub eob_sum: u64,
    pub eob_max: u32,
    pub nonzero_coeffs: u64,
    pub golomb: u64,
    pub tx_type_reads: u64,
    pub tx_type_histogram: [u64; TX_TYPES],
    /// 每个 `txSz` 上的变换块数与真实消耗位数（诊断：看有没有某类尺寸被多解/偏贵）
    pub txb_size_histogram: [[u64; 2]; TX_SIZES_ALL],
    /// 这些 `coeffs()` 一共吃掉了多少位（诊断：和瓦片字节数对账）
    pub bits: u64,
    /// `residual()` 自己的跨度（诊断：应当等于上面那个 `bits`）
    pub residual_bits: u64,
    /// 分段位记账（诊断）：eob 段 / base 段 / 符号段
    pub bits_eob: u64,
    pub bits_base: u64,
    pub bits_sign: u64,
    pub extra_bits: u64,
    pub sign_bits: u64,
}

/// trace 开关（`AV1_TRACE_TXB`）。
fn trace_txb() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("AV1_TRACE_TXB").is_ok())
}

/// trace 开关（`AV1_TRACE_CTXW`）：打印每个变换块的 `cul_level` / `dc_category`。
///
/// ⚠️ **必须缓存**（2026-10-04）：这两处分别在 `coeffs()` 尾部与 [`ResidualCtx::publish_levels`] 里，
/// 都是**逐变换块**的路径。裸 `std::env::var` 每次都要拿全局锁 + 线性扫描环境块 + 分配一个
/// `String` —— 实测占单线程总时间的 **1.7%**（`sample` 里 `__findenv_locked` 105 个样本，
/// 调用栈直指 `ResidualCtx::transform_block`）。
fn trace_ctxw() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("AV1_TRACE_CTXW").is_ok())
}

// ── residual：`transform_tree` / `transform_block` / `coeffs` ────────────

/// 解一个块的残差所需的"块级常量"（递归过程中不变）。
#[derive(Debug, Clone, Copy)]
pub struct ResidualConfig {
    pub mi_rows: i32,
    pub mi_cols: i32,    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub num_planes: u8,
    /// `HasChroma`（`decode_block` 开头推导出来的那个）
    pub has_chroma: bool,
    /// `MiSize`
    pub mi_size: u8,
    /// `Lossless = LosslessArray[ segment_id ]`
    pub lossless: bool,
    pub is_inter: bool,
    pub skip: bool,
    /// `TxSize`（块级）
    pub tx_size: u8,
    pub mi_row: i32,
    pub mi_col: i32,
    pub y_mode: u8,
    pub uv_mode: u8,
    pub angle_delta_y: i8,
    pub angle_delta_uv: i8,
    /// CFL 的 α（`CflAlphaU`/`CflAlphaV`；只有 `uv_mode == UV_CFL_PRED` 才用）
    pub cfl_alpha_u: i8,
    pub cfl_alpha_v: i8,
    pub use_filter_intra: bool,
    pub filter_intra_mode: u8,
    /// `get_filter_type(0)` / `get_filter_type(>0)`（§7.11.2.8，亮度/色度各一份）
    pub filter_type_y: u8,
    pub filter_type_uv: u8,
    /// `enable_intra_edge_filter`（序列头开关）
    pub enable_intra_edge_filter: bool,
    /// `AvailU` / `AvailL` / `AvailUChroma` / `AvailLChroma`（§6.10.4）
    pub avail_u: bool,
    pub avail_l: bool,
    pub avail_u_chroma: bool,
    pub avail_l_chroma: bool,
    pub reduced_tx_set: bool,
    /// `get_qindex( 1, segment_id ) > 0`
    pub qindex_nonzero: bool,
    /// §7.12.3 的三个平面量化步长（`dc` 每平面都加 `DeltaQ*Dc`；
    /// **`ac` 的 plane 0 不加任何 delta** —— 规范就是这么不对称）。
    pub dc_quant: [i32; 3],
    pub ac_quant: [i32; 3],
    /// 建这些量化步长用的位深（`dequantize` 要用）。
    pub quant_bit_depth: u8,
    pub use_128x128_superblock: bool,
}

/// palette 块重建要用的"颜色表 + 索引图"。
///
/// 归 [`super::block::BlockCtx`] 所有（那是逐块的临时状态），残差侧只借来看 ——
/// 所以这里全是共享引用。`stride_*` 是索引图的行宽（以**该平面的样本**为单位），
/// 因为 `predict_palette` 要拿"块内 4×4 偏移 × 4"去索引它。
pub struct PaletteRef<'a> {
    pub size_y: usize,
    pub size_uv: usize,
    pub colors_y: &'a [u16; PALETTE_COLORS],
    pub colors_u: &'a [u16; PALETTE_COLORS],
    pub colors_v: &'a [u16; PALETTE_COLORS],
    pub map_y: &'a [u8],
    pub map_uv: &'a [u8],
    pub stride_y: usize,
    pub stride_uv: usize,
}

/// `ResidualCtx` 对 `BlockGrid` 的访问句柄。
///
/// 语法阶段要**写**（`set_tx_type` / `LoopfilterTxSizes`），重建阶段只**读** ——
/// 分成两档之后，重建阶段就能拿着 `&BlockGrid` 跨线程共享（row-mt 阶段 2）。
pub enum GridAccess<'a> {
    Read(&'a super::block::BlockGrid),
    Write(&'a mut super::block::BlockGrid),
}

impl<'a> GridAccess<'a> {
    #[inline]
    pub(crate) fn ro(&self) -> &super::block::BlockGrid {
        match self {
            GridAccess::Read(g) => g,
            GridAccess::Write(g) => g,
        }
    }

    #[inline]
    pub(crate) fn rw(&mut self) -> &mut super::block::BlockGrid {
        match self {
            GridAccess::Write(g) => g,
            GridAccess::Read(_) => unreachable!("重建阶段不该写 BlockGrid"),
        }
    }
}

/// 残差链路的**复用缓冲**（逐瓦片一份，见 `BlockCtx::tx_scratch`）。
///
/// 每块 `vec![]` 一次在这里是纯浪费：这几块缓冲在连续的块之间**尺寸大概率相同**，
/// 长度没变就什么都不用做（与 [`super::inter::scratch_of`] 同一个套路）。清零规则：
/// * `levels`：`coeffs` 全量写（写入前 `quant` 的对应位置已清零，未编码位置就是 0）；
/// * `quant`：**唯一需要清零的一块**，但清零**由 `coeffs` 自己做、且只清
///   `[0, min(w*h, MAX_TXB_COEFFS))` 减去"这次真写到的位置"那个补集**
///   （2026-10-05 第三十四轮：原来由调用方每次 `fill(0)` 整块，`sample` 显示
///   独占整帧 2.1%，而 `all_zero` 的块那份完全是白写）；
/// * `t`：`dequantize` 写满系数格 `min(32,w)×min(32,h)`，而反变换**只读**这一片；
/// * `residual`：行趟把 `w·h` 全部写满；
/// * `pred`：每次整块从平面重填。
#[derive(Default)]
pub struct TxScratch {
    levels: Vec<i32>,
    t: Vec<i32>,
    pred: Vec<u16>,
    residual: Vec<i64>,
    /// 8bit/10bit 反变换的 i32 中间量（第四十五刀：8 路通道）；12bit 走 `residual`。
    residual32: Vec<i32>,
    /// `coeffs` 收集量化系数用的缓冲（**只清前 `min(w*h, MAX_TXB_COEFFS)` 个**）。
    ///
    /// 早先是 `coeffs` 里的局部栈数组 `[i32; MAX_TXB_COEFFS]`（4KB）——
    /// 那会让**每次调用都物化整个 1024 格**，而实际只有 `min(w*h,1024)`
    /// 个位置有效（真素材上平均不到 100）。实测在 row-mt 记录阶段每次都多花
    /// 约 2µs，一帧 3 万个变换块就是几十毫秒。复用缓冲 + 只清有效范围即可归零。
    quant: Vec<i32>,
}

/// 与 [`super::inter::scratch_of`] 同一个套路：长度没变就什么都不做，变长才重开。
fn scratch_len<T: Clone + Default>(v: &mut Vec<T>, n: usize) -> &mut [T] {
    if v.len() != n {
        v.clear();
        v.resize(n, T::default());
    }
    &mut v[..n]
}

/// 残差解码的上下文。`grid` 里的 `InterTxSizes` / `TxTypes` 是**帧级**的
/// （由 [`super::block`] 持有），`cdfs` 是**瓦片级**的。
pub struct ResidualCtx<'a> {
    pub cdfs: &'a mut CoeffCdfs,
    pub ctxs: &'a mut CoeffContexts,
    pub grid: GridAccess<'a>,
    pub stats: &'a mut CoeffStats,
    pub cfg: ResidualConfig,
    /// 帧级像素缓冲。`None` = **只读语法、不做像素重建**（真素材的语法验收就是这么跑的）；
    /// `Some` 时每个变换块都要走 §7.11.2 的帧内预测。
    pub frame: Option<&'a mut super::intra::CurrFrame>,
    /// `MaxLumaW` / `MaxLumaH`（§7.11.5 CFL 用）：**当前块已重建的亮度范围**，
    /// 每经过一个亮度变换块就更新（规范里它们是全局变量，只有亮度块会写）。
    pub max_luma_w: i32,
    pub max_luma_h: i32,
    /// 诊断计数（`AV1_TRACE_TX1`）
    pub trace_count: u32,
    /// 本块是 palette 块时才有：预测改走"查颜色表"，不走进帧内模式预测。
    pub palette: Option<PaletteRef<'a>>,
    /// **row-mt 语法阶段**：`Some` 时只记录工单（系数进 arena）、不写像素
    /// （见 `block::RowMtJobs`）。
    pub record: Option<&'a mut super::block::RowMtJobs>,
    /// **row-mt 回放**：当前变换块预先算好的 `BlockDecoded` 两个查询
    /// （`None` = 照旧读运行时网格；见 [`super::intra::DecFlags`]）。
    pub flags: Option<super::intra::DecFlags>,
}

impl ResidualCtx<'_> {
    /// `residual( )`（规范 §5.11.35）：块按 64×64 分片，逐平面走变换树/变换块。
    pub fn residual(&mut self, dec: &mut SymbolDecoder<'_>, scratch: &mut TxScratch) -> Result<()> {
        let r0 = dec.consumed_bits();
        let bw = block_width(self.cfg.mi_size);
        let bh = block_height(self.cfg.mi_size);
        let width_chunks = (bw >> 6).max(1);
        let height_chunks = (bh >> 6).max(1);
        let chunk_size = if width_chunks > 1 || height_chunks > 1 {
            BLOCK_64X64 as u8
        } else {
            self.cfg.mi_size
        };
        let chroma_planes = if self.chroma() { 2 } else { 0 };
        for chunk_y in 0..height_chunks {
            for chunk_x in 0..width_chunks {
                let mi_row_chunk = self.cfg.mi_row + (chunk_y << 4);
                let mi_col_chunk = self.cfg.mi_col + (chunk_x << 4);
                for plane in 0..(1 + chroma_planes) {
                    let tx_sz = if self.cfg.lossless {
                        TX_4X4
                    } else {
                        plane_tx_size(
                            self.cfg.mi_size,
                            plane,
                            self.cfg.tx_size,
                            self.cfg.subsampling_x,
                            self.cfg.subsampling_y,
                        )
                    };
                    let step_x = (TX_WIDTH[tx_sz as usize] as i32) >> 2;
                    let step_y = (TX_HEIGHT[tx_sz as usize] as i32) >> 2;
                    let plane_sz = plane_residual_size(
                        chunk_size,
                        plane,
                        self.cfg.subsampling_x,
                        self.cfg.subsampling_y,
                    );
                    let num_w = NUM_4X4_BLOCKS_WIDE[plane_sz as usize] as i32;
                    let num_h = NUM_4X4_BLOCKS_HIGH[plane_sz as usize] as i32;
                    let sub_x = u8::from(plane > 0 && self.cfg.subsampling_x);
                    let sub_y = u8::from(plane > 0 && self.cfg.subsampling_y);
                    let base_x = (mi_col_chunk >> sub_x) * MI_SIZE;
                    let base_y = (mi_row_chunk >> sub_y) * MI_SIZE;
                    if self.cfg.is_inter && !self.cfg.lossless && plane == 0 {
                        self.transform_tree(dec, base_x, base_y, num_w * 4, num_h * 4, scratch)?;
                    } else {
                        let base_x_block = (self.cfg.mi_col >> sub_x) * MI_SIZE;
                        let base_y_block = (self.cfg.mi_row >> sub_y) * MI_SIZE;
                        let mut y = 0;
                        while y < num_h {
                            let mut x = 0;
                            while x < num_w {
                                self.transform_block(
                                    dec,
                                    plane,
                                    base_x_block,
                                    base_y_block,
                                    tx_sz,
                                    x + ((chunk_x << 4) >> sub_x),
                                    y + ((chunk_y << 4) >> sub_y),
                                    scratch,
                                )?;
                                x += step_x;
                            }
                            y += step_y;
                        }
                    }
                }
            }
        }
        self.stats.residual_bits += (dec.consumed_bits() - r0) as u64;
        Ok(())
    }

    /// 这个块有没有色度（`HasChroma`）。
    fn chroma(&self) -> bool {
        self.cfg.num_planes > 1 && self.cfg.has_chroma
    }

    /// `transform_tree( startX, startY, w, h )`
    fn transform_tree(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        start_x: i32,
        start_y: i32,
        w: i32,
        h: i32,
        scratch: &mut TxScratch,
    ) -> Result<()> {
        let max_x = self.cfg.mi_cols * MI_SIZE;
        let max_y = self.cfg.mi_rows * MI_SIZE;
        if start_x >= max_x || start_y >= max_y {
            return Ok(());
        }
        let row = start_y >> MI_SIZE_LOG2;
        let col = start_x >> MI_SIZE_LOG2;
        let luma_tx_sz = self.grid.ro().inter_tx_size(row, col);
        let luma_w = TX_WIDTH[luma_tx_sz as usize] as i32;
        let luma_h = TX_HEIGHT[luma_tx_sz as usize] as i32;
        if w <= luma_w && h <= luma_h {
            let tx_sz = find_tx_size(w, h);
            self.transform_block(dec, 0, start_x, start_y, tx_sz, 0, 0, scratch)
        } else if w > h {
            self.transform_tree(dec, start_x, start_y, w / 2, h, scratch)?;
            self.transform_tree(dec, start_x + w / 2, start_y, w / 2, h, scratch)
        } else if w < h {
            self.transform_tree(dec, start_x, start_y, w, h / 2, scratch)?;
            self.transform_tree(dec, start_x, start_y + h / 2, w, h / 2, scratch)
        } else {
            self.transform_tree(dec, start_x, start_y, w / 2, h / 2, scratch)?;
            self.transform_tree(dec, start_x + w / 2, start_y, w / 2, h / 2, scratch)?;
            self.transform_tree(dec, start_x, start_y + h / 2, w / 2, h / 2, scratch)?;
            self.transform_tree(dec, start_x + w / 2, start_y + h / 2, w / 2, h / 2, scratch)
        }
    }

    /// `transform_block( plane, baseX, baseY, txSz, x, y )`。
    ///
    /// 顺序照规范 §7.11.35：**先帧内预测（不读位）、再读系数**，最后标记
    /// `BlockDecoded`。`frame` 为 `None` 时只读位（真素材的语法验收走这条路）。
    #[allow(clippy::too_many_arguments)]
    fn transform_block(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        plane: u8,
        base_x: i32,
        base_y: i32,
        tx_sz: u8,
        x: i32,
        y: i32,
        scratch: &mut TxScratch,
    ) -> Result<()> {
        let start_x = base_x + 4 * x;
        let start_y = base_y + 4 * y;
        let sub_x = u8::from(plane > 0 && self.cfg.subsampling_x);
        let sub_y = u8::from(plane > 0 && self.cfg.subsampling_y);
        let max_x = (self.cfg.mi_cols * MI_SIZE) >> sub_x;
        let max_y = (self.cfg.mi_rows * MI_SIZE) >> sub_y;
        if trace_tb_coord() {
            eprintln!(
                "TB p={} start=({},{}) max=({},{}) tx={} read={}",
                plane,
                start_x,
                start_y,
                max_x,
                max_y,
                tx_sz,
                !(start_x >= max_x || start_y >= max_y)
            );
        }
        if start_x >= max_x || start_y >= max_y {
            return Ok(());
        }

        // `MaxLumaW/MaxLumaH`：亮度变换块一重建完就更新（CFL 的钳位依据，§7.11.5）。
        //
        // ⚠️ 规范原文就是**裸的** `MaxLumaW = startX + stepX * 4`（**不钳帧**，
        // 已对着规范正文 §7.12.3 的 `transform_block` 原文核过）。AV1 的块尺寸
        // **不因帧边界缩小**：帧底那一行超块里的亮度变换块整个越出帧（例：1080
        // 高的帧里，(544,1056) 处仍有一个 32×32 的变换块 ⇒ `MaxLumaH = 1088`）。
        // 这些"帧外"样本**我们照样重建**（落在 `Plane` 的 `PLANE_PAD` 里），而
        // CFL 的 `Min(lumaY, MaxLumaH - 2)` 会读到它们、**累加进 `lumaAvg`** ——
        // 与 dav1d / libaom 的行为一致。2026-10-03 踩过：早先把 MaxLuma 钳到帧内
        // （1080），帧底色度块的 `lumaAvg` 因此偏小 4（Q3），**整个**色度块（连
        // 可见行一起）被拉偏 ±1 —— 真素材上表现为孤立的 ±1 单点、跨帧传染。
        if plane == 0 && self.record.is_none() {
            // row-mt 语法阶段不写（它是**重建侧**的量，回放时由 `replay_txbs` 补）
            self.update_max_luma(start_x, start_y, tx_sz);
        }

        // §5.11.35 `transform_block` 结尾：把这组 4×4 写进 `LoopfilterTxSizes`
        // （去块滤波靠它判断"这条边是不是变换块边界"；与像素重建无关，永远要写）。
        {
            let sub_x = i32::from(plane > 0 && self.cfg.subsampling_x);
            let sub_y = i32::from(plane > 0 && self.cfg.subsampling_y);
            let base_r = ((start_y << sub_y) >> MI_SIZE_LOG2) >> sub_y;
            let base_c = ((start_x << sub_x) >> MI_SIZE_LOG2) >> sub_x;
            let step_x = (TX_WIDTH[tx_sz as usize] >> 2) as usize;
            let step_y = (TX_HEIGHT[tx_sz as usize] >> 2) as usize;
            let cols = self.grid.ro().cols();
            let rows = self.grid.ro().rows();
            let table = &mut self.grid.rw().lf_tx_sizes[plane as usize];
            for i in 0..step_y {
                for j in 0..step_x {
                    let (rr, cc) = (base_r as usize + i, base_c as usize + j);
                    if rr < rows && cc < cols {
                        table[rr * cols + cc] = tx_sz;
                    }
                }
            }
        }

        // row-mt 语法阶段：只记录工单（系数抄进 arena），一切像素动作都推到重建阶段。
        let recording = self.record.is_some();
        let reconstructing = self.frame.is_some() && !recording;
        let dbg = !recording && flags::trace_tx1() && plane == 0 && self.trace_count < 40;
        let (mut v_pre, mut v_pred) = (0i32, 0i32);
        if dbg {
            v_pre = self.frame.as_ref().map(|f| f.plane(0).get(start_x as usize, start_y as usize) as i32).unwrap_or(-1);
        }
        if reconstructing {
            self.reconstruct_block(plane, start_x, start_y, tx_sz, x, y)?;
        }
        if dbg {
            v_pred = self.frame.as_ref().map(|f| f.plane(0).get(start_x as usize, start_y as usize) as i32).unwrap_or(-1);
            self.trace_count += 1;
        }

        if recording {
            // 语法阶段：照旧读系数（`levels_out` 必须开着），然后记工单。
            // ⚠️ `skip` 的块**也要记**：帧内预测按变换块做，跟有没有残差无关。
            let w = TX_WIDTH[tx_sz as usize] as usize;
            let h = TX_HEIGHT[tx_sz as usize] as usize;
            // 系数格只有 `min(w*h, MAX_TXB_COEFFS)` 是有效的：64 尺寸的变换
            // 只把系数放在左上 32×32（`coeffs` 的 `out[..n]` 正是这个口径），
            // 其余位置恒为 0。按 `w*h` 整段拷会白拷 3/4（对 64×64）。
            // `dequantize` 也只读 `min(w,32)×min(h,32)`，所以这个截断是安全的。
            let n_eff = (w * h).min(MAX_TXB_COEFFS);
            let n = if self.cfg.skip { 0 } else { n_eff };
            // **直写 arena**：先预留（清零）再把这段切片交给 `coeffs` 写 —— 本来有
            // 两次拷贝（`coeffs` 写 `TxScratch::levels`、`push_txb` 再整段拷进 arena），
            // 现在并成一次。`record` 先 `take()` 出来（否则它借的 `self` 与
            // `self.coeffs(&mut self, …)` 冲突），用完放回。
            let mut rec = self.record.take();
            let (at, out) = match rec {
                Some(ref mut r) => {
                    let at = r.reserve_levels(n);
                    (at, Some(&mut r.levels[at..at + n]))
                }
                None => (0, None),
            };
            let eob = if self.cfg.skip {
                0
            } else {
                // ⚠️ 这里**不再** `quant.fill(0)`（2026-10-05 第三十四轮）：清零已经
                // 挪进 `coeffs` 内部、且只在"真的编了系数"之后做 —— `all_zero` 的块
                // 一个字节都不该白写。
                let quant = scratch_len(&mut scratch.quant, n_eff);
                self.coeffs(dec, plane, start_x, start_y, tx_sz, out, quant)?
            };
            if let Some(r) = rec.as_deref_mut() {
                r.push_txb_meta(super::block::TxbJob {
                    plane,
                    start_x,
                    start_y,
                    tx_sz,
                    x,
                    y,
                    levels_at: at,
                    levels_len: n,
                    has_residual: eob > 0,
                    flags: Default::default(),
                });
            }
            self.record = rec;
            return Ok(());
        }

        if !self.cfg.skip {
            let w = TX_WIDTH[tx_sz as usize] as usize;
            let h = TX_HEIGHT[tx_sz as usize] as usize;
            // 复用缓冲：长度没变就什么都不做（见 [`TxScratch`]）
            let eob = {
                let n_eff = (w * h).min(MAX_TXB_COEFFS);
                let levels = scratch_len(&mut scratch.levels, n_eff);
                // 清零在 `coeffs` 内部做（同上），这里只负责把缓冲切到该长度。
                let quant = scratch_len(&mut scratch.quant, n_eff);
                self.coeffs(
                    dec,
                    plane,
                    start_x,
                    start_y,
                    tx_sz,
                    reconstructing.then_some(&mut *levels),
                    quant,
                )?
            };
            if reconstructing && eob > 0 {
                let TxScratch { levels, t, pred, residual, residual32, .. } = scratch;
                self.add_residual(
                    plane,
                    start_x,
                    start_y,
                    tx_sz,
                    &levels[..(w * h).min(MAX_TXB_COEFFS)],
                    t,
                    pred,
                    residual,
                    residual32,
                )?;
            }
            if dbg {
                let v_post = self.frame.as_ref().map(|f| f.plane(0).get(start_x as usize, start_y as usize) as i32).unwrap_or(-1);
                println!(
                    "BLK start=({start_x},{start_y}) tx={tx_sz} skip={} eob={eob} plane0: 预测前={v_pre} 预测后={v_pred} 残差后={v_post}",
                    self.cfg.skip
                );
            }
        } else if dbg {
            let v_post = self.frame.as_ref().map(|f| f.plane(0).get(start_x as usize, start_y as usize) as i32).unwrap_or(-1);
            println!(
                "BLK start=({start_x},{start_y}) tx={tx_sz} skip={} eob=skip块 plane0: 预测前={v_pre} 预测后={v_pred} 残差后={v_post}",
                self.cfg.skip
            );
        }
        Ok(())
    }

    /// 把残差加回预测值：§7.12.4 反量化 + §7.13 反变换。
    ///
    /// 预测值已经在 [`Self::reconstruct_block`] 里写进平面了，所以这里先把那一块
    /// 拷出来当 `pred`，再让反变换把 `pred + residual` 写回去（`Clip1` 在那边做）。
    /// `MaxLumaW/MaxLumaH`（§7.11.5，CFL 用）：**重建侧**的量 —— 每重建一个亮度
    /// 变换块更新一次（规范里它们是全局变量，只有亮度块会写）。
    ///
    /// **照规范裸算，不钳帧**（`MaxLumaW = startX + stepX * 4`）：块伸出帧底/帧右
    /// 时，伸出的那部分样本我们照样重建（落在 `Plane` 的 `PLANE_PAD` 里），CFL
    /// 要读到它们才能与 dav1d 对齐（理由见 `transform_block` 里那段长注释）。
    fn update_max_luma(&mut self, start_x: i32, start_y: i32, tx_sz: u8) {
        self.max_luma_w = start_x + TX_WIDTH[tx_sz as usize] as i32;
        self.max_luma_h = start_y + TX_HEIGHT[tx_sz as usize] as i32;
    }

    /// row-mt **重建阶段**：按顺序回放一个块的变换块（帧内预测 + `MaxLuma` 更新 +
    /// 残差叠加）。系数在语法阶段就抄进了 arena，这里只做像素侧。
    pub(crate) fn replay_txbs(
        &mut self,
        txbs: &[super::block::TxbJob],
        levels: &[i32],
        scratch: &mut TxScratch,
    ) -> Result<()> {
        for tj in txbs {
            // row-mt：这两个 `BlockDecoded` 查询按"划分树顺序"预先算好了（见 `DecFlags`）；
            // 诊断旋钮 `ME_ROW_MT_LIVE_FLAGS=1` 时改读运行时网格（只有顺序回放才与记录序同序）
            self.flags = if flags::row_mt_live_flags() { None } else { Some(tj.flags) };
            // 帧内：逐变换块预测（帧间块的 `reconstruct_block` 是 no-op）
            self.reconstruct_block(tj.plane, tj.start_x, tj.start_y, tj.tx_sz, tj.x, tj.y)?;
            if tj.plane == 0 {
                self.update_max_luma(tj.start_x, tj.start_y, tj.tx_sz);
            }
            if tj.has_residual {
                let lv = &levels[tj.levels_at..tj.levels_at + tj.levels_len];
                let TxScratch { t, pred, residual, residual32, .. } = scratch;
                self.add_residual(tj.plane, tj.start_x, tj.start_y, tj.tx_sz, lv, t, pred, residual, residual32)?;
            }
            // 诊断（`ME_TRACE_RACE=1`）：**逐变换块**盖章，而且必须等**残差也加完**才算"写完" ——
            // 早先在 `reconstruct_block` 之后就盖，读者会看到"已写但缺残差"的像素，
            // 探针于是假阴性（2026-10-02 踩过：并行输出大面积错而探针 0 命中）。
            if flags::trace_race() {
                super::block::racemap::mark(
                    (tj.start_y >> 2) as i64,
                    (tj.start_x >> 2) as i64,
                    (TX_WIDTH[tj.tx_sz as usize] >> 2).max(1) as i64,
                    (TX_HEIGHT[tj.tx_sz as usize] >> 2).max(1) as i64,
                    tj.plane > 0,
                );
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn add_residual(
        &mut self,
        plane: u8,
        start_x: i32,
        start_y: i32,
        tx_sz: u8,
        // 系数（系数格 `tw*th`）：普通路径是 `scratch.levels`，row-mt 回放是 arena 里那份
        src_levels: &[i32],
        t: &mut Vec<i32>,
        pred: &mut Vec<u16>,
        residual: &mut Vec<i64>,
        residual32: &mut Vec<i32>,
    ) -> Result<()> {
        let w = TX_WIDTH[tx_sz as usize] as usize;
        let h = TX_HEIGHT[tx_sz as usize] as usize;
        let tx_type = self.compute_tx_type(plane, tx_sz, start_x >> 2, start_y >> 2);
        let quant = super::transform::Quant {
            dc: self.cfg.dc_quant,
            ac: self.cfg.ac_quant,
            bit_depth: self.cfg.quant_bit_depth,
        };
        let t = scratch_len(t, w * h);
        super::transform::dequantize(src_levels, tx_sz, plane, &quant, self.cfg.lossless, t)?;

        let lossless = self.cfg.lossless;
        let frame = self.frame.as_mut().expect("调用前已判过 Some");
        let pred = scratch_len(pred, w * h);
        {
            // 逐像素 `Plane::get` 每次都要在 u8/u16 两种存储之间分支 + 做边界断言；
            // 这里按平面**分派一次**，然后整行切片拷贝。8bit 仍然要逐样本加宽到 u16。
            let p = frame.plane(plane);
            let stride = p.stride;
            let x0 = start_x as usize;
            let y0 = start_y as usize;
            if p.is_u8() {
                let data = p.px8();
                for i in 0..h {
                    let row = &data[(y0 + i) * stride + x0..][..w];
                    for (d, s) in pred[i * w..(i + 1) * w].iter_mut().zip(row) {
                        *d = u16::from(*s);
                    }
                }
            } else {
                let data = p.px16();
                for i in 0..h {
                    pred[i * w..(i + 1) * w]
                        .copy_from_slice(&data[(y0 + i) * stride + x0..][..w]);
                }
            }
        }
        let stride = frame.plane(plane).stride;
        let bit_depth = frame.plane(plane).bit_depth;
        if flags::trace_tx1() && start_y < 6 && start_x < 6 && plane == 0 {
            let _ = &frame;
            println!(
                "TX1 plane={plane} start=({start_x},{start_y}) tx_sz={tx_sz} tx_type={tx_type} \n  pred={:?}\n  levels={:?}\n  dequant_t={:?}",
                pred, src_levels, t
            );
            let res = super::transform::debug_inverse_only(t, tx_sz, tx_type, bit_depth, lossless);
            println!("  residual={res:?}");
        }
        // 写回的是**这个变换块**的位置：`dst` 要从 (start_x, start_y) 起算
        // （曾经整个平面从头传进去 → 残差全叠到左上角，画面上只剩"预测的平滑图"）。
        let base = start_y as usize * stride + start_x as usize;
        let plane_ref = &mut frame.planes[plane as usize];
        if plane_ref.is_u8() {
            let dst = &mut plane_ref.px8_mut()[base..];
            super::transform::inverse_transform_add8(
                &t,
                tx_sz,
                tx_type,
                pred,
                dst,
                stride,
                bit_depth,
                lossless,
                residual,
                residual32,
            )
        } else {
            let dst = &mut plane_ref.px16_mut()[base..];
            super::transform::inverse_transform_add(
                &t,
                tx_sz,
                tx_type,
                pred,
                dst,
                stride,
                bit_depth,
                lossless,
                residual,
                residual32,
            )
        }
    }

    /// 一个变换块的**像素重建**：帧内预测 + 标记 `BlockDecoded`。
    ///
    /// `x` / `y` 是它在块内的 4×4 偏移（规范 `transform_block` 的入参），
    /// 用来判 `haveLeft = AvailL || x > 0`、`haveAbove = AvailU || y > 0`。
    fn reconstruct_block(
        &mut self,
        plane: u8,
        start_x: i32,
        start_y: i32,
        tx_sz: u8,
        x: i32,
        y: i32,
    ) -> Result<()> {
        if self.cfg.is_inter {
            // 帧间块的预测已经在 `compute_prediction`（block.rs）里写完像素了；
            // 这里只负责"标记已重建"（规范 `transform_block` 的收尾在调用方做）。
            return Ok(());
        }
        let sub_x = i32::from(plane > 0 && self.cfg.subsampling_x);
        let sub_y = i32::from(plane > 0 && self.cfg.subsampling_y);
        // 规范 `transform_block` 开头：row/col 是"按平面 4×4 单位"的帧内坐标，
        // `subBlockMiRow/Col` 再取它在超块内的位置（`& sbMask`）。
        let sb_mask = if self.cfg.use_128x128_superblock { 31 } else { 15 };
        let row = (start_y << sub_y) >> MI_SIZE_LOG2;
        let col = (start_x << sub_x) >> MI_SIZE_LOG2;
        let sub_block_mi_row = row & sb_mask;
        let sub_block_mi_col = col & sb_mask;
        let step_x = (TX_WIDTH[tx_sz as usize] as i32) >> 2;
        let step_y = (TX_HEIGHT[tx_sz as usize] as i32) >> 2;

        // palette 块：预测是"拿索引图查颜色表"，**不走进帧内模式预测**（规范 §7.11.4
        // 的 `predict_palette`）。规范在 `transform_block` 里就是二选一：
        // `( plane == 0 && PaletteSizeY ) || ( plane != 0 && PaletteSizeUV )`。
        let palette_size = self
            .palette
            .as_ref()
            .map(|p| if plane == 0 { p.size_y } else { p.size_uv })
            .unwrap_or(0);
        if palette_size > 0 {
            let p = self.palette.as_ref().expect("上面已判过 Some");
            let (colors, map, stride) = match plane {
                0 => (p.colors_y, p.map_y, p.stride_y),
                1 => (p.colors_u, p.map_uv, p.stride_uv),
                _ => (p.colors_v, p.map_uv, p.stride_uv),
            };
            let frame = self.frame.as_mut().expect("调用前已判过 Some");
            super::intra::predict_palette(
                frame,
                plane,
                start_x as usize,
                start_y as usize,
                tx_sz,
                x,
                y,
                colors,
                map,
                stride,
            )?;
            let frame = self.frame.as_mut().expect("调用前已判过 Some");
            frame.mark_decoded(
                plane,
                (row >> sub_y) as i64,
                (col >> sub_x) as i64,
                step_x,
                step_y,
            );
            return Ok(());
        }

        // CFL：先做**普通 DC 预测**，随后 `predict_chroma_from_luma` 再加高频
        // （规范 `transform_block`：`mode = isCfl ? DC_PRED : UVMode`）
        let is_cfl = plane > 0 && self.cfg.uv_mode == super::block::UV_CFL_PRED;
        let (mode, angle_delta) = if plane == 0 {
            (self.cfg.y_mode, self.cfg.angle_delta_y)
        } else if is_cfl {
            (super::block::DC_PRED, 0)
        } else {
            (self.cfg.uv_mode, self.cfg.angle_delta_uv)
        };
        let (avail_l, avail_u, filter_type) = if plane == 0 {
            (self.cfg.avail_l, self.cfg.avail_u, self.cfg.filter_type_y)
        } else {
            (
                self.cfg.avail_l_chroma,
                self.cfg.avail_u_chroma,
                self.cfg.filter_type_uv,
            )
        };

        let frame = self.frame.as_mut().expect("调用前已判过 Some");
        super::intra::predict_intra_txb(
            frame,
            plane,
            start_x as usize,
            start_y as usize,
            tx_sz,
            sub_block_mi_row,
            sub_block_mi_col,
            step_x,
            step_y,
            x,
            y,
            avail_l,
            avail_u,
            mode,
            angle_delta,
            plane == 0 && self.cfg.use_filter_intra,
            self.cfg.filter_intra_mode,
            filter_type,
            self.cfg.enable_intra_edge_filter,
            self.flags,
        )?;

        if is_cfl {
            let alpha = if plane == 1 {
                self.cfg.cfl_alpha_u as i32
            } else {
                self.cfg.cfl_alpha_v as i32
            };
            let frame = self.frame.as_mut().expect("调用前已判过 Some");
            super::intra::predict_cfl(
                frame,
                plane,
                start_x as usize,
                start_y as usize,
                tx_sz,
                alpha,
                self.max_luma_w,
                self.max_luma_h,
            )?;
        }

        // 规范 `transform_block` 结尾：把这一块覆盖到的 4×4 全标成"已重建"
        let frame = self.frame.as_mut().expect("调用前已判过 Some");
        // 标记用**全局平面 4×4 坐标**（与 `CurrFrame::decoded` 的网格口径一致；
        // 原先误用了超块内局部坐标，跨超块会串台）
        frame.mark_decoded(
            plane,
            ((row >> sub_y)) as i64,
            ((col >> sub_x)) as i64,
            step_x,
            step_y,
        );
        Ok(())
    }

    /// `transform_type( x4, y4, txSz )`（规范 §5.11.38）
    fn transform_type(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        x4: i32,
        y4: i32,
        tx_sz: u8,
    ) -> Result<u8> {
        let set = self.tx_set(tx_sz);
        let tx_type = if set > 0 && self.cfg.qindex_nonzero {
            let sqr = TX_SIZE_SQR[tx_sz as usize] as usize;
            if self.cfg.is_inter {
                match set {
                    TX_SET_INTER_1 => {
                        let s = dec.read_symbol(&mut self.cdfs.inter_tx_type_set1[sqr])?;
                        self.stats.symbols += 1;
                        self.stats.tx_type_reads += 1;
                        TX_TYPE_INTER_INV_SET1[s as usize]
                    }
                    TX_SET_INTER_2 => {
                        let s = dec.read_symbol(&mut self.cdfs.inter_tx_type_set2)?;
                        self.stats.symbols += 1;
                        self.stats.tx_type_reads += 1;
                        TX_TYPE_INTER_INV_SET2[s as usize]
                    }
                    _ => {
                        let s = dec.read_symbol(&mut self.cdfs.inter_tx_type_set3[sqr])?;
                        self.stats.symbols += 1;
                        self.stats.tx_type_reads += 1;
                        TX_TYPE_INTER_INV_SET3[s as usize]
                    }
                }
            } else {
                let intra_dir = if self.cfg.use_filter_intra {
                    FILTER_INTRA_MODE_TO_INTRA_DIR[(self.cfg.filter_intra_mode as usize).min(4)]
                } else {
                    self.cfg.y_mode
                } as usize;
                if set == TX_SET_INTRA_1 {
                    let s = dec.read_symbol(&mut self.cdfs.intra_tx_type_set1[sqr][intra_dir])?;
                    self.stats.symbols += 1;
                    self.stats.tx_type_reads += 1;
                    TX_TYPE_INTRA_INV_SET1[s as usize]
                } else {
                    let s = dec.read_symbol(&mut self.cdfs.intra_tx_type_set2[sqr][intra_dir])?;
                    self.stats.symbols += 1;
                    self.stats.tx_type_reads += 1;
                    TX_TYPE_INTRA_INV_SET2[s as usize]
                }
            }
        } else {
            DCT_DCT
        };
        self.stats.tx_type_histogram[tx_type as usize] += 1;
        let w4 = (TX_WIDTH[tx_sz as usize] as i32) >> 2;
        let h4 = (TX_HEIGHT[tx_sz as usize] as i32) >> 2;
        for j in 0..h4 {
            for i in 0..w4 {
                self.grid.rw().set_tx_type(y4 + j, x4 + i, tx_type);
            }
        }
        Ok(tx_type)
    }

    /// `get_tx_set( txSz )`（规范 §5.11.38）
    fn tx_set(&self, tx_sz: u8) -> u8 {
        let sqr = TX_SIZE_SQR[tx_sz as usize];
        let sqr_up = TX_SIZE_SQR_UP[tx_sz as usize];
        if sqr_up > TX_32X32 {
            return TX_SET_DCTONLY;
        }
        if self.cfg.is_inter {
            if self.cfg.reduced_tx_set || sqr_up == TX_32X32 {
                TX_SET_INTER_3
            } else if sqr == TX_16X16 {
                TX_SET_INTER_2
            } else {
                TX_SET_INTER_1
            }
        } else if sqr_up == TX_32X32 {
            TX_SET_DCTONLY
        } else if self.cfg.reduced_tx_set || sqr == TX_16X16 {
            TX_SET_INTRA_2
        } else {
            TX_SET_INTRA_1
        }
    }

    /// `compute_tx_type( plane, txSz, blockX, blockY )`（规范 §5.11.38）
    fn compute_tx_type(&self, plane: u8, tx_sz: u8, block_x: i32, block_y: i32) -> u8 {
        let sqr_up = TX_SIZE_SQR_UP[tx_sz as usize];
        if self.cfg.lossless || sqr_up > TX_32X32 {
            return DCT_DCT;
        }
        if plane == 0 {
            return self.grid.ro().tx_type(block_y, block_x);
        }
        let set = self.tx_set(tx_sz);
        if self.cfg.is_inter {
            let x4 = self.cfg.mi_col.max(block_x << u8::from(self.cfg.subsampling_x));
            let y4 = self.cfg.mi_row.max(block_y << u8::from(self.cfg.subsampling_y));
            let ty = self.grid.ro().tx_type(y4, x4);
            if TX_TYPE_IN_SET_INTER[set as usize][ty as usize] {
                return ty;
            }
            return DCT_DCT;
        }
        let ty = MODE_TO_TXFM[(self.cfg.uv_mode as usize).min(13)];
        if TX_TYPE_IN_SET_INTRA[set as usize][ty as usize] {
            ty
        } else {
            DCT_DCT
        }
    }

    /// `all_zero` 的上下文（规范 §8.3.2）+ 对拍用中间量：`(ctx, top, left, bw, bh, w, h)`。
    ///
    /// top/left 是传播用的聚合值：`plane 0` 为邻域 `max`，色度为 OR 后的字节。
    fn txb_ctx_full(
        &self,
        plane: u8,
        x4: i32,
        y4: i32,
        w4: i32,
        h4: i32,
        tx_sz: u8,
    ) -> (usize, i32, i32, i32, i32, i32, i32) {
        let sub_x = u8::from(plane > 0 && self.cfg.subsampling_x);
        let sub_y = u8::from(plane > 0 && self.cfg.subsampling_y);
        let max_x4 = self.cfg.mi_cols >> sub_x;
        let max_y4 = self.cfg.mi_rows >> sub_y;
        let p = plane as usize;
        let w = TX_WIDTH[tx_sz as usize] as i32;
        let h = TX_HEIGHT[tx_sz as usize] as i32;
        let bsize =
            plane_residual_size(self.cfg.mi_size, plane, self.cfg.subsampling_x, self.cfg.subsampling_y);
        let bw = block_width(bsize);
        let bh = block_height(bsize);
        if plane == 0 {
            let mut top = 0i32;
            let mut left = 0i32;
            for k in 0..w4 {
                if x4 + k < max_x4 {
                    top = top.max(self.ctxs.above_level[0][(x4 + k).max(0) as usize] as i32);
                }
            }
            for k in 0..h4 {
                if y4 + k < max_y4 {
                    left = left.max(self.ctxs.left_level[0][(y4 + k).max(0) as usize] as i32);
                }
            }
            let ctx = if bw == w && bh == h {
                0
            } else if top == 0 && left == 0 {
                1
            } else if top == 0 || left == 0 {
                2 + usize::from(top.max(left) > 3)
            } else if top.max(left) <= 3 {
                4
            } else if top.min(left) <= 3 {
                5
            } else {
                6
            };
            (ctx, top, left, bw, bh, w, h)
        } else {
            let mut above = 0u8;
            let mut left = 0u8;
            for i in 0..w4 {
                if x4 + i < max_x4 {
                    let i = (x4 + i).max(0) as usize;
                    above |= self.ctxs.above_level[p][i];
                    above |= self.ctxs.above_dc[p][i];
                }
            }
            for i in 0..h4 {
                if y4 + i < max_y4 {
                    let i = (y4 + i).max(0) as usize;
                    left |= self.ctxs.left_level[p][i];
                    left |= self.ctxs.left_dc[p][i];
                }
            }
            let mut ctx = usize::from(above != 0) + usize::from(left != 0) + 7;
            if bw * bh > w * h {
                ctx += 3;
            }
            (ctx, above as i32, left as i32, bw, bh, w, h)
        }
    }

    /// `get_coeff_base_ctx( txSz, plane, blockX, blockY, pos, c, isEob )`（规范 §8.3.2）
    fn coeff_base_ctx(
        &self,
        plane: u8,
        x4: i32,
        y4: i32,
        tx_sz: u8,
        quant: &[i32],
        pos: usize,
        c: u32,
        is_eob: bool,
    ) -> usize {
        let adj = ADJUSTED_TX_SIZE[tx_sz as usize];
        let bwl = TX_WIDTH_LOG2[adj as usize] as u32;
        let width = 1usize << bwl;
        let height = TX_HEIGHT[adj as usize] as usize;
        let tx_type = self.compute_tx_type(plane, tx_sz, x4, y4);
        if is_eob {
            if c == 0 {
                return SIG_COEF_CONTEXTS - 4;
            }
            if (c as usize) <= (height << bwl) / 8 {
                return SIG_COEF_CONTEXTS - 3;
            }
            if (c as usize) <= (height << bwl) / 4 {
                return SIG_COEF_CONTEXTS - 2;
            }
            return SIG_COEF_CONTEXTS - 1;
        }
        let class = tx_class(tx_type) as usize;
        let row = (pos >> bwl) as i32;
        let col = (pos - ((row as usize) << bwl)) as i32;
        let mut mag = 0i32;
        for idx in 0..SIG_REF_DIFF_OFFSET_NUM {
            let rr = row + SIG_REF_DIFF_OFFSET[class][idx][0];
            let rc = col + SIG_REF_DIFF_OFFSET[class][idx][1];
            if rr >= 0 && rc >= 0 && (rr as usize) < height && (rc as usize) < width {
                mag += quant[((rr as usize) << bwl) + rc as usize].abs().min(3);
            }
        }
        let ctx = ((mag + 1) >> 1).min(4) as usize;
        if class == TX_CLASS_2D as usize {
            if row == 0 && col == 0 {
                return 0;
            }
            return ctx
                + COEFF_BASE_CTX_OFFSET[tx_sz as usize][row.min(4) as usize][col.min(4) as usize]
                    as usize;
        }
        let idx = if class == TX_CLASS_VERT as usize { row } else { col };
        ctx + COEFF_BASE_POS_CTX_OFFSET[idx.min(2) as usize] as usize
    }

    /// `coeff_br` 的上下文（规范 §8.3.2）
    fn coeff_br_ctx(
        &self,
        plane: u8,
        x4: i32,
        y4: i32,
        tx_sz: u8,
        quant: &[i32],
        pos: usize,
    ) -> usize {
        let adj = ADJUSTED_TX_SIZE[tx_sz as usize];
        let bwl = TX_WIDTH_LOG2[adj as usize] as u32;
        let txw = TX_WIDTH[adj as usize] as i32;
        let txh = TX_HEIGHT[adj as usize] as i32;
        let row = (pos >> bwl) as i32;
        let col = (pos - ((row as usize) << bwl)) as i32;
        let class = tx_class(self.compute_tx_type(plane, tx_sz, x4, y4)) as usize;
        let mut mag = 0i32;
        for idx in 0..3 {
            let rr = row + MAG_REF_OFFSET_WITH_TX_CLASS[class][idx][0];
            let rc = col + MAG_REF_OFFSET_WITH_TX_CLASS[class][idx][1];
            if rr >= 0 && rc >= 0 && rr < txh && rc < txw {
                mag += quant[(rr as usize) * (txw as usize) + rc as usize]
                    .min((COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1) as i32);
            }
        }
        let mag = ((mag + 1) >> 1).min(6) as usize;
        if pos == 0 {
            mag
        } else if class == TX_CLASS_2D as usize {
            if row < 2 && col < 2 {
                mag + 7
            } else {
                mag + 14
            }
        } else if class == TX_CLASS_HORIZ as usize {
            if col == 0 {
                mag + 7
            } else {
                mag + 14
            }
        } else if row == 0 {
            mag + 7
        } else {
            mag + 14
        }
    }

    /// `dc_sign` 的上下文（规范 §8.3.2）
    fn dc_sign_ctx(&self, plane: u8, x4: i32, y4: i32, w4: i32, h4: i32) -> usize {
        let sub_x = u8::from(plane > 0 && self.cfg.subsampling_x);
        let sub_y = u8::from(plane > 0 && self.cfg.subsampling_y);
        let max_x4 = self.cfg.mi_cols >> sub_x;
        let max_y4 = self.cfg.mi_rows >> sub_y;
        let p = plane as usize;
        let mut dc_sign = 0i32;
        for k in 0..w4 {
            if x4 + k < max_x4 {
                match self.ctxs.above_dc[p][(x4 + k).max(0) as usize] {
                    1 => dc_sign -= 1,
                    2 => dc_sign += 1,
                    _ => {}
                }
            }
        }
        for k in 0..h4 {
            if y4 + k < max_y4 {
                match self.ctxs.left_dc[p][(y4 + k).max(0) as usize] {
                    1 => dc_sign -= 1,
                    2 => dc_sign += 1,
                    _ => {}
                }
            }
        }
        if dc_sign < 0 {
            1
        } else if dc_sign > 0 {
            2
        } else {
            0
        }
    }

    /// `eob_pt_*` 的上下文：变换类不是 2D 时为 1
    fn eob_ctx(&self, plane: u8, tx_sz: u8, x4: i32, y4: i32) -> usize {
        usize::from(tx_class(self.compute_tx_type(plane, tx_sz, x4, y4)) != TX_CLASS_2D)
    }

    /// `coeffs( plane, startX, startY, txSz )`（规范 §5.11.39）—— 本模块的主角。
    fn coeffs(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        plane: u8,
        start_x: i32,
        start_y: i32,
        tx_sz: u8,
        levels_out: Option<&mut [i32]>,
        // 系数收集缓冲（长度 `min(w*h, MAX_TXB_COEFFS)`，**调用前已清零**）。
        quant: &mut [i32],
    ) -> Result<u32> {
        let x4 = start_x >> 2;
        let y4 = start_y >> 2;
        let w4 = (TX_WIDTH[tx_sz as usize] as i32) >> 2;
        let h4 = (TX_HEIGHT[tx_sz as usize] as i32) >> 2;
        let tx_sz_ctx =
            (TX_SIZE_SQR[tx_sz as usize] as usize + TX_SIZE_SQR_UP[tx_sz as usize] as usize + 1) >> 1;
        let ptype = usize::from(plane > 0);
        let seg_eob = if tx_sz == TX_16X64 || tx_sz == TX_64X16 {
            512usize
        } else {
            (TX_WIDTH[tx_sz as usize] as usize * TX_HEIGHT[tx_sz as usize] as usize).min(1024)
        };
        let pos0 = dec.consumed_bits();
        let eob_mark = self.stats.bits_eob;
        dec.tag = self.stats.txbs as u32;
        dec.dbg_plane = plane;
        dec.dbg_tx = tx_sz;
        dec.dbg_txbs += 1;
        let base_mark = self.stats.bits_base;
        let sign_mark = self.stats.bits_sign;
        self.stats.txbs += 1;
        self.stats.txb_size_histogram[tx_sz as usize][0] += 1;

        // all_zero S()（无条件读）
        let (ctx, top, left, bw, bh, w, h) = self.txb_ctx_full(plane, x4, y4, w4, h4, tx_sz);
        if super::block::in_traced_frame() && flags::trace_tbxctx() {
            let p = plane as usize;
            let g = |v: &[u8], i: i32| -> i32 { v.get(i.max(0) as usize).copied().unwrap_or(u8::MAX) as i32 };
            eprintln!(
                "TBXCTX frame={} txb={} plane={plane} tx_size={tx_sz} txs_ctx={tx_sz_ctx} skip_ctx={ctx} \
                 top={top} left={left} bw={bw} bh={bh} w={w} h={h} x4={x4} y4={y4} a0={} l0={}",
                super::block::TRACE_TILE_FRAME_NO.load(std::sync::atomic::Ordering::Relaxed),
                self.stats.txbs - 1,
                g(&self.ctxs.above_level[p], x4),
                g(&self.ctxs.left_level[p], y4)
            );
        }
        let sz0 = dec.consumed_bits();
        dec.label = "txb_skip";
        let all_zero = dec.read_symbol(&mut self.cdfs.txb_skip[tx_sz_ctx][ctx])? == 1;
        self.stats.symbols += 1;
        if trace_txb() && self.stats.txbs < 30 {
            println!(
                "      txb#{} p={} tx={} szctx={} ctx={} all_zero={} 花了 {} 位（smb={}）",
                self.stats.txbs,
                plane,
                tx_sz,
                tx_sz_ctx,
                ctx,
                all_zero as u8,
                dec.consumed_bits() - sz0,
                dec.symbol_max_bits()
            );
        }
        if all_zero {
            self.stats.all_zero += 1;
            dec.dbg_all_zero += 1;
            if trace_txb() && self.stats.txbs < 40 {
                println!(
                    "      txb#{:<3} p={} tx={} all_zero=1 合计 {} 位",
                    self.stats.txbs,
                    plane,
                    tx_sz,
                    dec.consumed_bits() - pos0
                );
            }
            if plane == 0 {
                for i in 0..w4 {
                    for j in 0..h4 {
                        self.grid.rw().set_tx_type(y4 + j, x4 + i, DCT_DCT);
                    }
                }
            }
            let used = (dec.consumed_bits() - pos0) as u64;
            self.stats.bits += used;
            self.stats.txb_size_histogram[tx_sz as usize][1] += used;
            if flags::trace_ctxw() {
                eprintln!("CTXW eob plane={plane} tx_size={tx_sz} eob=0 cul=0 dc=0");
            }
            self.publish_levels(plane, x4, y4, w4, h4, 0, 0);
            return Ok(0);
        }

        if plane == 0 {
            self.transform_type(dec, x4, y4, tx_sz)?;
        }
        let plane_tx_type = self.compute_tx_type(plane, tx_sz, x4, y4);
        let scan = scan_for(tx_sz, plane_tx_type);
        let eob_multisize = (TX_WIDTH_LOG2[tx_sz as usize] as i32).min(5)
            + (TX_HEIGHT_LOG2[tx_sz as usize] as i32).min(5)
            - 4;

        // eob_pt_* S()：按 eobMultisize 从 7 张表里选一张
        let eob_pt = match eob_multisize {
            5 | 6 => {
                dec.label = "eob_pt_512/1024";
                let s = if eob_multisize == 5 {
                    dec.read_symbol(&mut self.cdfs.eob_pt_512[ptype])?
                } else {
                    dec.read_symbol(&mut self.cdfs.eob_pt_1024[ptype])?
                };
                self.stats.symbols += 1;
                s + 1
            }
            _ => {
                dec.label = "eob_pt_16..256";
                let ctx = self.eob_ctx(plane, tx_sz, x4, y4);
                let s = match eob_multisize {
                    0 => dec.read_symbol(&mut self.cdfs.eob_pt_16[ptype][ctx])?,
                    1 => dec.read_symbol(&mut self.cdfs.eob_pt_32[ptype][ctx])?,
                    2 => dec.read_symbol(&mut self.cdfs.eob_pt_64[ptype][ctx])?,
                    3 => dec.read_symbol(&mut self.cdfs.eob_pt_128[ptype][ctx])?,
                    _ => dec.read_symbol(&mut self.cdfs.eob_pt_256[ptype][ctx])?,
                };
                self.stats.symbols += 1;
                s + 1
            }
        };

        let mut eob = if eob_pt < 2 { eob_pt } else { (1u32 << (eob_pt - 2)) + 1 };
        let mut eob_shift = (eob_pt as i32 - 3).max(-1);
        if eob_shift >= 0 {
            dec.label = "eob_extra";
            let s = dec.read_symbol(&mut self.cdfs.eob_extra[tx_sz_ctx][ptype][(eob_pt - 3) as usize])?;
            self.stats.symbols += 1;
            if s == 1 {
                eob += 1 << eob_shift;
            }
            for i in 1..(eob_pt as i32 - 2).max(0) {
                eob_shift = (eob_pt as i32 - 2).max(0) - 1 - i;
                dec.label = "eob_extra_bit";
                let bit = dec.read_literal(1)?;
                self.stats.literals += 1;
                self.stats.extra_bits += 1;
                if bit == 1 {
                    eob += 1 << eob_shift;
                }
            }
        }
        let eob_usize = eob as usize;
        if eob_usize == 0 || eob_usize > seg_eob || eob_usize > scan.len() {
            return Err(MediaError::Bitstream(format!(
                "AV1 系数 EOB 越界：eob = {eob}（eobPt = {eob_pt}、segEob = {seg_eob}、\
                 txSz = {tx_sz}、扫描表 {})",
                scan.len()
            )));
        }

        self.stats.bits_eob += (dec.consumed_bits() - pos0) as u64;
        // ── 清零改到"知道 eob 之后"（2026-10-05 第三十四轮）────────────────
        //
        // 原来由调用方在进 `coeffs` 之前 `quant[..n].fill(0)`。`sample` 显示这一处
        // 的 `bzero` 独占**整帧 2.1%**（30k+ 个变换块 × 平均不到 100 格）——
        // 而其中**一大半是白写的**：
        //
        // * `txb_skip == 1`（`all_zero`）的块在 §上面就 `return Ok(0)` 了，
        //   `eob == 0` ⇒ `add_residual` 根本不会被调用 ⇒ **`quant` 一个字节都没被读**；
        // * 剩下的块只需要"没写到的位置是 0"，而**已写到的位置马上就会被写掉**。
        //
        // 于是改成：只清 `[0, n)` 减去 `scan[0..eob)` 的那个补集。
        // `DEFAULT_SCAN_*` 是**对角扫描**（`0,1,4,8,5,2,…`，不单调），所以
        // 补集不能靠"顺序扫一遍"求；用一张 16 项的位图标 `scan[0..eob]`，
        // 再把没打标的位置清掉，最后把 `mx` 之后整段一次 `fill`。
        // 对角扫描下 `mx` 只比 `eob` 大一点 ⇒ 实际写入量是 `O(eob)` 而不是 `O(n)`。
        //
        // ⚠️ **最终状态与"先整块 `fill(0)`"逐位相同**：被清零的集合完全一样，
        // 而 `scan[0..eob]` 里的值随后由下面的幅度循环覆写（掩码里的洞也在
        // 同一个补集里）。这条由受控流 + dav1d 逐像素对拍钉住。
        {
            let n = quant.len();
            let mut mx = 0usize;
            for &p in &scan[..eob_usize] {
                mx = mx.max(usize::from(p));
            }
            // 空洞太多时位图不划算（`eob` 接近 `n` 的稠密块）——退回整块清。
            // `mx < n ≤ MAX_TXB_COEFFS = 1024` ⇒ 16 项位图一定够用。
            if mx + 1 > n / 2 {
                quant[..n].fill(0);
            } else {
                debug_assert!(mx < 1024, "系数量图必须放得下 16 × u64");
                let mut written = [0u64; 16];
                for &p in &scan[..eob_usize] {
                    let p = usize::from(p);
                    written[p >> 6] |= 1u64 << (p & 63);
                }
                for (i, slot) in quant[..=mx].iter_mut().enumerate() {
                    if written[i >> 6] & (1u64 << (i & 63)) == 0 {
                        *slot = 0;
                    }
                }
                quant[mx + 1..].fill(0);
            }
        }
        let p_eob = dec.consumed_bits();
        // 第一轮：从 EOB-1 往回读幅度
        for c in (0..eob_usize).rev() {
            let pos = scan[c] as usize;
            let k_sym: usize;
            let sym_val: u32;
            let mut level = if c == eob_usize - 1 {
                // 规范：`get_coeff_base_ctx(...,1) - SIG_COEF_CONTEXTS + SIG_COEF_CONTEXTS_EOB`
                // （前者是 38..41，所以要按**有符号**算，别在 usize 上减）
                let raw = self.coeff_base_ctx(plane, x4, y4, tx_sz, &quant, pos, c as u32, true);
                let k = (raw as i32 - SIG_COEF_CONTEXTS as i32 + SIG_COEF_CONTEXTS_EOB as i32)
                    as usize;
                dec.label = "coeff_base_eob";
                let s = dec.read_symbol(&mut self.cdfs.coeff_base_eob[tx_sz_ctx][ptype][k])?;
                self.stats.symbols += 1;
                k_sym = k;
                sym_val = s;
                s + 1
            } else {
                let k = self.coeff_base_ctx(plane, x4, y4, tx_sz, &quant, pos, c as u32, false);
                dec.label = "coeff_base";
                let s = dec.read_symbol(&mut self.cdfs.coeff_base[tx_sz_ctx][ptype][k])?;
                self.stats.symbols += 1;
                k_sym = k;
                sym_val = s;
                s
            };
            if trace_txb() {
                println!(
                    "        c={} pos={} k={} {}={} -> level={}",
                    c,
                    pos,
                    k_sym,
                    if c == eob_usize - 1 { "coeff_base_eob" } else { "coeff_base" },
                    sym_val,
                    level
                );
            }
            if level > NUM_BASE_LEVELS {
                let k = self.coeff_br_ctx(plane, x4, y4, tx_sz, &quant, pos);
                let br_sz_ctx = tx_sz_ctx.min(TX_32X32 as usize);
                for _ in 0..(COEFF_BASE_RANGE / (BR_CDF_SIZE - 1)) {
                    dec.label = "coeff_br";
                    let s = dec.read_symbol(&mut self.cdfs.coeff_br[br_sz_ctx][ptype][k])?;
                    self.stats.symbols += 1;
                    if trace_txb() {
                        println!("        coeff_br[{}][{}][{}] = {}", br_sz_ctx, ptype, k, s);
                    }
                    level += s;
                    if s < BR_CDF_SIZE - 1 {
                        break;
                    }
                }
            }
            quant[pos] = level as i32;
        }

        self.stats.bits_base += (dec.consumed_bits() - p_eob) as u64;
        let p_base = dec.consumed_bits();
        // 第二轮：符号 + Exp-Golomb 尾
        let dc_ctx = self.dc_sign_ctx(plane, x4, y4, w4, h4);
        let mut dc_category = 0i32;
        let mut cul_level = 0i32;
        for c in 0..eob_usize {
            let pos = scan[c] as usize;
            let sign = if quant[pos] != 0 {
                if c == 0 {
                    dec.label = "dc_sign";
                    if trace_txb() {
                        println!(
                            "      >> dc_sign 读取：txb#{} p={} eob={} c={} dc_ctx={}",
                            self.stats.txbs, plane, eob, c, dc_ctx
                        );
                    }
                    let s = dec.read_symbol(&mut self.cdfs.dc_sign[ptype][dc_ctx])?;
                    self.stats.symbols += 1;
                    s
                } else {
                    dec.label = "sign_bit";
                    let b = dec.read_literal(1)?;
                    self.stats.literals += 1;
                    self.stats.sign_bits += 1;
                    b
                }
            } else {
                0
            };
            if quant[pos] > (NUM_BASE_LEVELS + COEFF_BASE_RANGE) as i32 {
                // Exp-Golomb 的**长度码是一元码、且以 `1` 收尾**：
                // 规范写的是 `do { length++; bit } while ( !bit )` ——
                // 即"位是 0 就继续读"，所以**终止位是 1**。
                // （这里曾经写成 `if bit == 0 { break }`，极性反了 —— 只在残差大到
                //   进 golomb 时才会显形，所以全零残差的流看不出来。）
                let mut length = 0u32;
                loop {
                    length += 1;
                    dec.label = "golomb_len";
                    let b = dec.read_literal(1)?;
                    self.stats.literals += 1;
                    if b == 1 {
                        break;
                    }
                    // **硬上限**：越界后归位补的全是虚拟 0 位，这个循环会永远等不到
                    // 终止位 ⇒ 不加这道闸就是死循环（2026-09-30 真素材上原地挂了 18 分钟）。
                    // 合法码流里 `length` 远小于 32（系数幅度不可能那么大）。
                    if length >= 32 {
                        return Err(MediaError::Bitstream(
                            "AV1 的 Exp-Golomb 长度码读了 32 位还没遇到终止位（算术流已经越界）"
                                .into(),
                        ));
                    }
                }
                let mut x = 1u32;
                let mut i = length as i32 - 2;
                while i >= 0 {
                    dec.label = "golomb_data";
                    let b = dec.read_literal(1)?;
                    self.stats.literals += 1;
                    x = (x << 1) | b;
                    i -= 1;
                }
                self.stats.golomb += 1;
                if trace_txb() {
                    println!(
                        "        GOLOMB c={} pos={} level={} length={} x={} -> {}",
                        c,
                        pos,
                        quant[pos],
                        length,
                        x,
                        x + COEFF_BASE_RANGE + NUM_BASE_LEVELS
                    );
                }
                quant[pos] = (x + COEFF_BASE_RANGE + NUM_BASE_LEVELS) as i32;
            }
            if pos == 0 && quant[pos] > 0 {
                dc_category = if sign == 1 { 1 } else { 2 };
            }
            quant[pos] &= 0xF_FFFF;
            if flags::trace_ctxw() && self.stats.txbs == 1 {
                eprintln!("CTXL c={c} pos={pos} level={}", quant[pos]);
            }
            cul_level += quant[pos];
            if sign == 1 {
                quant[pos] = -quant[pos];
            }
        }
        let cul_level = cul_level.min(63);
        self.stats.bits_sign += (dec.consumed_bits() - p_base) as u64;
        self.stats.eob_sum += eob as u64;
        self.stats.eob_max = self.stats.eob_max.max(eob);
        self.stats.nonzero_coeffs += eob as u64;
        let used = (dec.consumed_bits() - pos0) as u64;
        self.stats.bits += used;
        self.stats.txb_size_histogram[tx_sz as usize][1] += used;
        if trace_txb() && self.stats.txbs < 40 {
            println!(
                "      txb#{:<3} p={} tx={} all_zero=0 eob={} 合计 {} 位（eob 段 {}、base 段 {}、符号段 {}）",
                self.stats.txbs,
                plane,
                tx_sz,
                eob,
                used,
                self.stats.bits_eob - eob_mark,
                self.stats.bits_base - base_mark,
                self.stats.bits_sign - sign_mark
            );
        }
        if trace_ctxw() {
            eprintln!("CTXW eob plane={plane} tx_size={tx_sz} eob={} cul={cul_level} dc={dc_category}", eob_usize);
        }
        self.publish_levels(plane, x4, y4, w4, h4, cul_level, dc_category);
        // 把量化系数交给调用方（重建时要用它做反量化 + 反变换）。
        // `quant` 的下标就是系数格 `pos`，所以按行优先 `w×h` 拷出去正好。
        if let Some(out) = levels_out {
            let n = (TX_WIDTH[tx_sz as usize] as usize * TX_HEIGHT[tx_sz as usize] as usize)
                .min(MAX_TXB_COEFFS)
                .min(out.len());
            out[..n].copy_from_slice(&quant[..n]);
        }
        Ok(eob)
    }

    /// `coeffs()` 结尾把 `culLevel` / `dcCategory` 铺到四个上下文数组上。
    fn publish_levels(
        &mut self,
        plane: u8,
        x4: i32,
        y4: i32,
        w4: i32,
        h4: i32,
        cul_level: i32,
        dc_category: i32,
    ) {
        let p = plane as usize;
        if trace_ctxw() {
            let sign = match dc_category {
                1 => 64,
                2 => 128,
                _ => 0,
            };
            eprintln!(
                "CTXW pub p={p} txb={} dir=A idx={} n={w4} val={}",
                self.stats.txbs.wrapping_sub(1),
                x4.max(0),
                cul_level + sign
            );
            eprintln!(
                "CTXW pub p={p} txb={} dir=L idx={} n={h4} val={}",
                self.stats.txbs.wrapping_sub(1),
                y4.max(0),
                cul_level + sign
            );
        }
        for i in 0..w4 {
            if let Some(slot) = self.ctxs.above_level[p].get_mut((x4 + i).max(0) as usize) {
                *slot = cul_level as u8;
            }
            if let Some(slot) = self.ctxs.above_dc[p].get_mut((x4 + i).max(0) as usize) {
                *slot = dc_category as u8;
            }
        }
        for i in 0..h4 {
            if let Some(slot) = self.ctxs.left_level[p].get_mut((y4 + i).max(0) as usize) {
                *slot = cul_level as u8;
            }
            if let Some(slot) = self.ctxs.left_dc[p].get_mut((y4 + i).max(0) as usize) {
                *slot = dc_category as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::block::{BlockGrid, DC_PRED};

    /// `reset_block_context` 的**色度清范围**必须照规范原文
    /// `i = MiCol >> subX .. (MiCol + bw4) >> subX`（下界取整、上界取整）。
    ///
    /// 这是 2026-10-01 用真素材逐符号对拍抓出来的真 bug 的回归钉子：旧实现把
    /// "宽度是奇数"的块直接判成"一格都不清"，而规范在 `mi_col` 也是奇数时**要清 1 格**
    /// （`[311>>1, (311+1)>>1) = [155,156)`）。漏清 ⇒ 上一个变换块写下的 `level` 残留
    /// ⇒ 同一列的 `txb_skip` 选错上下文 ⇒ 整条算术流错位（症状：前 8841 个变换块逐符号
    /// 全对，第 8842 个符号值相同但 CDF 不同）。
    #[test]
    fn reset_block_clears_the_chroma_slots_the_spec_range_covers() {
        // 1080p 4:2:0：mi 270×480，色度上下文 240×135
        let mut c = CoeffContexts::new(270, 480, true, true, 3);
        // ① 4×4 跳过块落在 mi=(99,311)（行、列都奇数 ⇒ 是色度参考块）
        c.above_level[2][155] = 1;
        c.above_dc[2][155] = 1;
        c.left_level[2][49] = 1;
        c.left_dc[2][49] = 1;
        c.reset_block(99, 311, 1, 1, true, true, 3);
        assert_eq!(c.above_level[2][155], 0, "色度 above 列 155 必须被清");
        assert_eq!(c.above_dc[2][155], 0);
        assert_eq!(c.left_level[2][49], 0, "色度 left 行 49 必须被清");
        assert_eq!(c.left_dc[2][49], 0);

        // ② 同样 4×4 但落在**偶数**列（mi=310，且不是色度参考块）⇒ 规范的范围是空的
        let mut c2 = CoeffContexts::new(270, 480, true, true, 3);
        c2.above_level[2][155] = 1;
        c2.reset_block(99, 310, 1, 1, true, true, 3);
        assert_eq!(c2.above_level[2][155], 1, "偶数列上 4×4 不该清色度（范围为空）");

        // ③ 8×8（bw4=2）落在奇数 mi_col：范围 [155, 156) 仍是 1 格
        let mut c3 = CoeffContexts::new(270, 480, true, true, 3);
        c3.above_level[2][155] = 3;
        c3.above_level[2][156] = 3;
        c3.reset_block(98, 311, 2, 2, true, true, 3);
        assert_eq!(c3.above_level[2][155], 0);
        assert_eq!(c3.above_level[2][156], 3, "范围右界是开的：156 不该被清");
    }

    /// `get_scan` 的长度必须等于 `coeffs` 的 `segEob`（否则 EOB 会落空）。
    #[test]
    fn scan_tables_cover_seg_eob_for_every_tx_size() {
        for tx_sz in 0..TX_SIZES_ALL as u8 {
            let seg_eob = if tx_sz == TX_16X64 || tx_sz == TX_64X16 {
                512usize
            } else {
                (TX_WIDTH[tx_sz as usize] as usize * TX_HEIGHT[tx_sz as usize] as usize).min(1024)
            };
            // 默认（2D / IDTX）扫描对每个尺寸都必须够长
            for ty in [DCT_DCT, IDTX] {
                let scan = scan_for(tx_sz, ty);
                assert!(
                    scan.len() >= seg_eob,
                    "txSz {tx_sz}（txType {ty}）的扫描表只有 {} 项，需要 {seg_eob}",
                    scan.len()
                );
            }
            // 行/列优先的扫描表只有 9 个尺寸有（规范的 `get_mrow_scan` 最后一档是 16x4）；
            // V/H 变换类型也只在这 9 个尺寸上可达（`get_tx_set` 会把更大的尺寸收成 DCTONLY）。
            if TX_SIZE_SQR_UP[tx_sz as usize] <= TX_16X16 {
                for ty in [V_DCT, H_DCT] {
                    let scan = scan_for(tx_sz, ty);
                    assert!(
                        scan.len() >= seg_eob,
                        "txSz {tx_sz}（txType {ty}）的扫描表只有 {} 项，需要 {seg_eob}",
                        scan.len()
                    );
                }
            }
        }
    }

    /// `Adjusted_Tx_Size` 的几何：宽高都得是 2 的幂（`pos >> bwl` 要用）。
    #[test]
    fn adjusted_tx_size_is_square_ish_and_power_of_two() {
        for tx_sz in 0..TX_SIZES_ALL {
            let adj = ADJUSTED_TX_SIZE[tx_sz];
            let w = TX_WIDTH[adj as usize];
            let h = TX_HEIGHT[adj as usize];
            assert!(w.is_power_of_two() && h.is_power_of_two(), "txSz {tx_sz}");
            assert!(w <= 32 && h <= 32, "调整后的尺寸不该超过 32：txSz {tx_sz} → {w}x{h}");
            assert_eq!(TX_WIDTH_LOG2[adj as usize], w.trailing_zeros() as u8);
            assert_eq!(TX_HEIGHT_LOG2[adj as usize], h.trailing_zeros() as u8);
        }
    }

    /// `init_coeff_cdfs` 的档位边界（`base_q_idx` 21/61/121 要换档）。
    #[test]
    fn coeff_cdf_class_boundaries() {
        assert_eq!(coeff_cdf_q_idx(0), 0);
        assert_eq!(coeff_cdf_q_idx(20), 0);
        assert_eq!(coeff_cdf_q_idx(21), 1);
        assert_eq!(coeff_cdf_q_idx(60), 1);
        assert_eq!(coeff_cdf_q_idx(61), 2);
        assert_eq!(coeff_cdf_q_idx(120), 2);
        assert_eq!(coeff_cdf_q_idx(121), 3);
        assert_eq!(coeff_cdf_q_idx(255), 3);
    }

    /// `get_tx_set` 的三条分支（规范 §5.11.38）。
    #[test]
    fn tx_set_follows_the_tx_size_and_flags() {
        // 帧内 + 非 reduced + 16×16 → SET_INTRA_1；reduced 时为 SET_INTRA_2；
        // 32×32 以上一律 DCTONLY
        let cfg = |is_inter: bool, reduced: bool| ResidualConfig {
            cfl_alpha_u: 0,
            cfl_alpha_v: 0,
            mi_rows: 64,
            mi_cols: 64,
            subsampling_x: true,
            subsampling_y: true,
            num_planes: 3,
            has_chroma: true,
            mi_size: BLOCK_8X8 as u8,
            lossless: false,
            is_inter,
            skip: false,
            tx_size: TX_8X8,
            mi_row: 0,
            mi_col: 0,
            y_mode: DC_PRED,
            uv_mode: DC_PRED,
            angle_delta_y: 0,
            angle_delta_uv: 0,
            use_filter_intra: false,
            filter_intra_mode: 0,
            filter_type_y: 0,
            filter_type_uv: 0,
            enable_intra_edge_filter: false,
            avail_u: false,
            avail_l: false,
            avail_u_chroma: false,
            avail_l_chroma: false,
            reduced_tx_set: reduced,
            qindex_nonzero: true,
            dc_quant: [4; 3],
            ac_quant: [4; 3],
            quant_bit_depth: 8,
            use_128x128_superblock: false,
        };
        let mut cdfs = CoeffCdfs::new(100);
        let mut ctxs = CoeffContexts::new(64, 64, true, true, 3);
        let mut grid = BlockGrid::new(64, 64);
        let mut stats = CoeffStats::default();
        let mut probe = ResidualCtx {
            max_luma_w: 0,
            max_luma_h: 0,
            trace_count: 0,
            cdfs: &mut cdfs,
            ctxs: &mut ctxs,
            grid: GridAccess::Write(&mut grid),
            stats: &mut stats,
            cfg: cfg(false, false),
            frame: None,
            palette: None,
            record: None,
            flags: None,
        };
        // 帧内：8×8 是 SET_INTRA_1，**16×16 是 SET_INTRA_2**（规范里专门有一条），
        // 32×32 及以上一律 DCTONLY
        assert_eq!(probe.tx_set(TX_8X8), TX_SET_INTRA_1);
        assert_eq!(probe.tx_set(TX_16X16), TX_SET_INTRA_2);
        assert_eq!(probe.tx_set(TX_32X32), TX_SET_DCTONLY);
        probe.cfg = cfg(false, true);
        assert_eq!(probe.tx_set(TX_8X8), TX_SET_INTRA_2);
        probe.cfg = cfg(true, false);
        assert_eq!(probe.tx_set(TX_16X16), TX_SET_INTER_2);
        assert_eq!(probe.tx_set(TX_8X8), TX_SET_INTER_1);
        assert_eq!(probe.tx_set(TX_32X32), TX_SET_INTER_3);
    }

    /// 4:2:0 下的色度残差尺寸（`Subsampled_Size` 的常见格子）。
    #[test]
    fn plane_residual_size_for_420() {
        assert_eq!(plane_residual_size(BLOCK_8X8 as u8, 1, true, true), BLOCK_4X4 as u8);
        assert_eq!(plane_residual_size(BLOCK_16X16 as u8, 1, true, true), BLOCK_8X8 as u8);
        assert_eq!(plane_residual_size(BLOCK_64X64 as u8, 1, true, true), BLOCK_32X32 as u8);
        assert_eq!(plane_residual_size(BLOCK_128X128 as u8, 1, true, true), BLOCK_64X64 as u8);
        // 亮度不变
        assert_eq!(plane_residual_size(BLOCK_32X32 as u8, 0, true, true), BLOCK_32X32 as u8);
        // 2 的幂
        for b in 0..BLOCK_SIZES as u8 {
            for plane in [0u8, 1, 2] {
                let sz = plane_residual_size(b, plane, true, true);
                assert!(sz as usize <= BLOCK_SIZES);
            }
        }
    }
}
