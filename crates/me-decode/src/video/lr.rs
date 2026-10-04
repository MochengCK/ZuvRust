//! AV1 **环路恢复**（规范 §7.17）：块级语法（§5.11.48）+ 滤波过程。
//!
//! ```text
//! 帧级：lr_params()（§5.9.7）—— FrameRestorationType[plane] / LoopRestorationSize[plane]
//! 块级：read_lr( r, c, sbSize )         §5.11.48  **每个超块一次**（在 decode_partition 之前）
//!          └─ read_lr_unit(plane, unitRow, unitCol)
//!                ├─ RESTORE_WIENER   → read_bool + 3 个系数（子指数哥伦布）
//!                ├─ RESTORE_SGRPROJ  → read_bool + sgr set + 2 个投影权重
//!                └─ RESTORE_SWITCHABLE → 3 个真 CDF 符号（use_wiener / use_sgrproj / type）
//! 像素：loop_restore_block( plane, row, col )  §7.17.1
//!          ├─ RESTORE_WIENER  → §7.17.4（水平 + 垂直两遍 7 抽头）
//!          └─ RESTORE_SGRPROJ → §7.17.2 + box filter（§7.17.3）
//! ```
//!
//! ## 性能（2026-10-02，单线程忙时从 ~25% 压到 ~11%，整帧 1.07x）
//!
//! 这一层是**逐 4×4 块**调用的（1080p 一帧几十万个块），原来最大的浪费全在
//! "每块都重算"和"每样本都重查"上。三样都不改结果、只改落点：
//!
//! 1. **`a2` 的整数除法 → 256 项常量表**（[`X_BY_XPLUS1`]，按规范式子 const 求值）：
//!    `box_filter` 每个 4×4 块要算 36 次 `((z<<8)+(z>>1))/(z+1)`，而 `z` 只有 256 档。
//!    ⚠️ `z` 在真实素材上会**为负**（`p * s_factor` 在 i32 里溢出，pass 1 的小 eps 档
//!    `s_factor` 到几千、`p` 到千万量级）—— 那是改动前就有的行为，负档走原式子兜底，
//!    **不要"顺手修"**（会改像素）。
//! 2. **`src.get` 的两次钳位 + stripe 判定：逐样本 → 逐行/逐列**。`get_source_sample`
//!    每个样本要钳 x、钳 y、判 stripe 两侧；而一个块只用到 ~10 行 × ~11 列 ——
//!    两趟解析表把这块全省掉（[`SourceSample::get`] 仍留着当规范参照）。
//! 3. **恢复单元几何每平面算一次**：`RestorationUnitSize ∈ {64,128,256}` 是 2 的幂，
//!    `/ unitSize` 换 `>> shift`；`count_units_in_frame` 与两个除法原先每 4×4 块算一遍。
//!    另外 `RESTORE_NONE` 的逐像素 `set`、以及三个滤波器的输出都改成按行写
//!    （`RowsSink::set_row`）。
//!
//! 验收：真素材 20 帧输出与优化前**逐字节一致**；计时"两二进制交替多轮取 min"
//! （这台机器噪声大，单次不可信）。
//!
//! ## 三个容易搞错的点
//!
//! 1. **`RefLrWiener` / `RefSgrXqd` 是"瓦片级"的**：规范在 `decode_tile` 开头
//!    （`clear_above_context()` 之后）把它们重置成 `Wiener_Taps_Mid` /
//!    `Sgrproj_Xqd_Mid`。它们是 `decode_signed_subexp_with_ref` 的**预测基准**
//!    （让相邻 unit 的系数编码更省），不是"跨帧继承的参数"。
//! 2. **`unitRowStart` / `unitColStart` 是向上取整**（`(x + unitSize - 1) / unitSize`），
//!    所以每个 restoration unit 只在它**结束**的那个超块里被读一次 ——
//!    写成"覆盖到的每个 unit 都读"会多读一大堆符号。
//! 3. **方向/顺序**：规范里列 `for (plane)` 在外、`unitRow` 再 `unitCol` 在内。
//!    SGR 的 `RefSgrXqd` 被 `i = 0, 1` 两个 pass 依次更新，**顺序不能换**。

use super::flags;
use me_core::{MediaError, Result};

use super::block::{NUM_4X4_BLOCKS_HIGH, NUM_4X4_BLOCKS_WIDE};
use super::entropy::SymbolDecoder;
use super::frame_header::FrameHeader;
use super::intra::CurrFrame;
use super::params::{
    LrParams, RESTORE_NONE, RESTORE_SGRPROJ, RESTORE_SWITCHABLE, RESTORE_WIENER,
};
use super::sequence_header::SequenceHeader;

/// `MI_SIZE`：一个 4×4 块的像素边长。
const MI_SIZE: i32 = 4;
/// `SUPERRES_NUM`（§5.9.2.10）：超分只在这个分母下才有意义。
const SUPERRES_NUM: i32 = 8;
/// `SGRPROJ_PARAMS_BITS`（§6.10.15）：`lr_sgr_set` 的位数。
pub const SGRPROJ_PARAMS_BITS: u32 = 4;
/// `SGRPROJ_PRJ_SUBEXP_K`（§6.10.15）。
const SGRPROJ_PRJ_SUBEXP_K: i32 = 4;

/// `Wiener_Taps_Min[3]`（§5.11.48）。
const WIENER_TAPS_MIN: [i32; 3] = [-5, -23, -17];
/// `Wiener_Taps_Max[3]`。
const WIENER_TAPS_MAX: [i32; 3] = [10, 8, 46];
/// `Wiener_Taps_K[3]`。
const WIENER_TAPS_K: [i32; 3] = [1, 2, 3];
/// `Wiener_Taps_Mid[3]`（§5.11.3）：`RefLrWiener` 每瓦片重置到它。
const WIENER_TAPS_MID: [i32; 3] = [3, -7, 15];
/// `Sgrproj_Xqd_Min[2]`。
const SGRPROJ_XQD_MIN: [i32; 2] = [-96, -32];
/// `Sgrproj_Xqd_Max[2]`。
const SGRPROJ_XQD_MAX: [i32; 2] = [31, 95];
/// `Sgrproj_Xqd_Mid[2]`（§5.11.3）。
const SGRPROJ_XQD_MID: [i32; 2] = [-32, 31];

/// `Sgr_Params[16][4]`（§7.17.3）：`[radius0, eps0, radius1, eps1]`。
pub const SGR_PARAMS: [[i32; 4]; 16] = [
    [2, 12, 1, 4],
    [2, 15, 1, 6],
    [2, 18, 1, 8],
    [2, 21, 1, 9],
    [2, 24, 1, 10],
    [2, 29, 1, 11],
    [2, 36, 1, 12],
    [2, 45, 1, 13],
    [2, 56, 1, 14],
    [2, 68, 1, 15],
    [0, 0, 1, 5],
    [0, 0, 1, 8],
    [0, 0, 1, 11],
    [0, 0, 1, 14],
    [2, 30, 0, 0],
    [2, 75, 0, 0],
];

/// 环路恢复的 3 个块级 CDF（§5.11.48）：三种帧级恢复类型都会用到
/// （`RESTORE_WIENER` 用第一个、`RESTORE_SGRPROJ` 用第二个、`RESTORE_SWITCHABLE` 三个都用）。
///
/// 它们**必须是帧级 CDF 的一部分**（随 `save_cdfs` / `load_cdfs` 在帧间继承），
/// 否则符号的概率模型会和编码器对不上 → 位错位。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LrCdfs {
    /// `Use_Wiener_Cdf`
    pub use_wiener: [u16; 3],
    /// `Use_Sgrproj_Cdf`
    pub use_sgrproj: [u16; 3],
    /// `Restoration_Type_Cdf`
    pub restoration_type: [u16; 4],
}

impl LrCdfs {
    /// 默认表（与 `cdf_tables.rs` 里的 `Default_*` 同源）。
    pub fn defaults() -> Self {
        Self {
            use_wiener: super::cdf_tables::DEFAULT_USE_WIENER_CDF,
            use_sgrproj: super::cdf_tables::DEFAULT_USE_SGRPROJ_CDF,
            restoration_type: super::cdf_tables::DEFAULT_RESTORATION_TYPE_CDF,
        }
    }

    /// `save_cdfs` 快照用：把每个 CDF 的**计数器**（数组最后一项）清零。
    ///
    /// 规范是在 `load_cdfs` 时清零的；我们放在"存快照"时做，两者等价。
    /// **漏掉这个就会出大问题**：计数器决定自适应速率，带着旧计数跨帧继承 ⇒
    /// 概率模型与编码器分道扬镳 ⇒ 真实素材上表现为"前两帧逐符号全对、第三帧起
    /// 在 LR 的 `restoration_type` 上莫名其妙地对不上"。
    pub fn reset_counts(&mut self) {
        let n = self.use_wiener.len() - 1;
        self.use_wiener[n] = 0;
        let n = self.use_sgrproj.len() - 1;
        self.use_sgrproj[n] = 0;
        let n = self.restoration_type.len() - 1;
        self.restoration_type[n] = 0;
    }
}

impl Default for LrCdfs {
    fn default() -> Self {
        Self::defaults()
    }
}

/// 帧级环路恢复状态（规范里的 `LrType` / `LrSgrSet` / `LrSgrXqd` / `LrWiener`）。
///
/// 按 **restoration unit** 索引（不是按像素/块），跨瓦片共享 —— 滤波阶段（§7.17）
/// 要遍历整个帧去查。两个 `Ref*` 是**瓦片级**的（每瓦片重置成 `_Mid`）。
#[derive(Debug, Clone)]
pub struct LrState {
    /// `LrType[plane][unitRow][unitCol]`
    pub lr_type: [Vec<u8>; 3],
    /// `LrSgrSet[plane][unitRow][unitCol]`
    pub sgr_set: [Vec<u8>; 3],
    /// `LrSgrXqd[plane][unitRow][unitCol][i]`
    pub sgr_xqd: [Vec<[i32; 2]>; 3],
    /// `LrWiener[plane][unitRow][unitCol][pass][j]`
    pub wiener: [Vec<[[i32; 3]; 2]>; 3],
    pub unit_rows: [usize; 3],
    pub unit_cols: [usize; 3],
    /// 瓦片级：`RefLrWiener[plane][pass][j]`
    ref_wiener: [[[i32; 3]; 2]; 3],
    /// 瓦片级：`RefSgrXqd[plane][pass]`
    ref_sgr_xqd: [[i32; 2]; 3],
}

impl LrState {
    /// 按帧头给的 `LoopRestorationSize[plane]` 建帧级数组。
    ///
    /// `count_units_in_frame(unitSize, frameSize) = Max((frameSize + unitSize/2) / unitSize, 1)`
    /// ——注意它是**四舍五入**而不是向上取整。
    pub fn new(
        seq: &SequenceHeader,
        lr: &LrParams,
        frame_height: u32,
        upscaled_width: u32,
    ) -> Self {
        let num_planes = if seq.color.mono_chrome { 1 } else { 3 };
        let mut unit_rows = [0usize; 3];
        let mut unit_cols = [0usize; 3];
        let mut lr_type = [Vec::new(), Vec::new(), Vec::new()];
        let mut sgr_set = [Vec::new(), Vec::new(), Vec::new()];
        let mut sgr_xqd = [Vec::new(), Vec::new(), Vec::new()];
        let mut wiener = [Vec::new(), Vec::new(), Vec::new()];
        for plane in 0..num_planes {
            // **不做恢复的 plane 直接留空**：`enable_restoration = 0`（或该 plane 是
            // `RESTORE_NONE`）时 `LoopRestorationSize[plane]` 根本没被赋值、还是 0，
            // 拿它去除就是除零。`read_lr` 对 `RESTORE_NONE` 本来就会 `continue`，
            // 所以空数组不会有人去查。
            if lr.frame_restoration_type[plane] == RESTORE_NONE {
                continue;
            }
            let (sub_x, sub_y) = plane_subsampling(plane, seq);
            let unit_size = (lr.size[plane] as i32).max(1);
            let rows = count_units_in_frame(unit_size, round2(frame_height as i32, sub_y));
            let cols = count_units_in_frame(unit_size, round2(upscaled_width as i32, sub_x));
            unit_rows[plane] = rows as usize;
            unit_cols[plane] = cols as usize;
            let n = (rows * cols) as usize;
            lr_type[plane] = vec![RESTORE_NONE; n];
            sgr_set[plane] = vec![0; n];
            sgr_xqd[plane] = vec![[0; 2]; n];
            wiener[plane] = vec![[[0; 3]; 2]; n];
        }
        Self {
            lr_type,
            sgr_set,
            sgr_xqd,
            wiener,
            unit_rows,
            unit_cols,
            ref_wiener: [[WIENER_TAPS_MID; 2]; 3],
            ref_sgr_xqd: [SGRPROJ_XQD_MID; 3],
        }
    }

    /// `decode_tile` 开头：`RefLrWiener` / `RefSgrXqd` 重置为 `_Mid`（规范 §5.11.3 之后）。
    pub fn begin_tile(&mut self) {
        self.ref_wiener = [[WIENER_TAPS_MID; 2]; 3];
        self.ref_sgr_xqd = [SGRPROJ_XQD_MID; 3];
    }

    /// 这一帧有没有任何 plane 真的要用恢复滤波（`UsesLr`）。
    pub fn any_lr(&self) -> bool {
        (0..3).any(|p| self.lr_type[p].iter().any(|t| *t != RESTORE_NONE))
    }

    /// `LrType[plane][unitRow][unitCol]`（滤波阶段用）。
    pub fn lr_type_at(&self, plane: usize, unit_row: usize, unit_col: usize) -> u8 {
        let idx = unit_row * self.unit_cols[plane] + unit_col;
        self.lr_type[plane].get(idx).copied().unwrap_or(RESTORE_NONE)
    }
}

/// `read_lr( r, c, bSize )`（规范 §5.11.48）：**每个超块一次**，
/// 在 `decode_tile` 的超块循环里、`decode_partition` **之前**调用。
///
/// `r` / `c` 是超块左上角的 4×4 坐标，`bsize` 是超块尺寸。
pub fn read_lr(
    dec: &mut SymbolDecoder<'_>,
    state: &mut LrState,
    cdfs: &mut LrCdfs,
    seq: &SequenceHeader,
    fh: &FrameHeader,
    r: i32,
    c: i32,
    bsize: usize,
) -> Result<()> {
    if trace_lr_frame_once() {
        eprintln!(
            "[LR] 帧级恢复类型 = {:?}，LoopRestorationSize = {:?}",
            fh.params.lr.frame_restoration_type, fh.params.lr.size
        );
    }
    if fh.allow_intrabc {
        return Ok(());
    }
    let num_planes = if seq.color.mono_chrome { 1 } else { 3 };
    let w = NUM_4X4_BLOCKS_WIDE[bsize] as i32;
    let h = NUM_4X4_BLOCKS_HIGH[bsize] as i32;
    for plane in 0..num_planes {
        let rtype = fh.params.lr.frame_restoration_type[plane];
        if rtype == RESTORE_NONE {
            continue;
        }
        let (sub_x, sub_y) = plane_subsampling(plane, seq);
        let unit_size = fh.params.lr.size[plane] as i32;
        if unit_size <= 0 {
            return Err(MediaError::Bitstream(format!(
                "AV1 环路恢复：plane {plane} 的 LoopRestorationSize 是 {unit_size}"
            )));
        }
        let unit_rows = state.unit_rows[plane] as i32;
        let unit_cols = state.unit_cols[plane] as i32;

        let row_scale = MI_SIZE >> sub_y;
        let unit_row_start = (r * row_scale + unit_size - 1) / unit_size;
        let unit_row_end = unit_rows.min(((r + h) * row_scale + unit_size - 1) / unit_size);

        // `use_superres` 时列的换算要带上 `SuperresDenom`（§5.11.48）
        let (numerator, denominator) = if fh.use_superres {
            ((MI_SIZE >> sub_x) * fh.superres_denom as i32, unit_size * SUPERRES_NUM)
        } else {
            (MI_SIZE >> sub_x, unit_size)
        };
        let unit_col_start = (c * numerator + denominator - 1) / denominator;
        let unit_col_end = unit_cols.min(((c + w) * numerator + denominator - 1) / denominator);

        for unit_row in unit_row_start..unit_row_end {
            for unit_col in unit_col_start..unit_col_end {
                read_lr_unit(
                    dec,
                    state,
                    cdfs,
                    plane,
                    rtype,
                    unit_row as usize,
                    unit_col as usize,
                )?;
            }
        }
    }
    Ok(())
}

/// `restoration_type` 的**原始取值**（规范 §6.10.15 的编号：
/// 0 = `RESTORE_NONE` / 1 = `RESTORE_WIENER` / 2 = `RESTORE_SGRPROJ`）
/// → **本模块内部编号**（`params.rs` 的 `RESTORE_*`：NONE=0 / SWITCHABLE=1 /
/// WIENER=2 / SGRPROJ=3）。
///
/// 两套编号长得像但**不是一回事** —— 直接拿原始值当内部值，`SGRPROJ`(2) 会被
/// 当成 `RESTORE_WIENER`、`WIENER`(1) 会被当成 `RESTORE_SWITCHABLE`。
const RESTORATION_TYPE_REMAP: [u8; 3] = [RESTORE_NONE, RESTORE_WIENER, RESTORE_SGRPROJ];

/// 诊断（`AV1_TRACE_LR=1`）：只在第一个超块打一次帧级 LR 参数。
fn trace_lr_frame_once() -> bool {
    use std::sync::atomic::{AtomicBool, Ordering};
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static DONE: AtomicBool = AtomicBool::new(false);
    if !*ON.get_or_init(|| std::env::var("AV1_TRACE_LR").is_ok()) {
        return false;
    }
    !DONE.swap(true, Ordering::Relaxed)
}

/// `read_lr_unit( plane, unitRow, unitCol )`（规范 §5.11.48）。
fn read_lr_unit(
    dec: &mut SymbolDecoder<'_>,
    state: &mut LrState,
    cdfs: &mut LrCdfs,
    plane: usize,
    frame_rtype: u8,
    unit_row: usize,
    unit_col: usize,
) -> Result<()> {
    let slot = unit_row * state.unit_cols[plane] + unit_col;
    if slot >= state.lr_type[plane].len() {
        return Err(MediaError::Bitstream(format!(
            "AV1 环路恢复：unit ({unit_row},{unit_col}) 落在 plane {plane} 的数组之外"
        )));
    }

    let restoration_type = if frame_rtype == RESTORE_WIENER {
        // ⚠️ 规范这里写的是 `use_wiener  S()` —— **`S()` 是读 CDF 符号，不是 `read_bool`**。
        // 三种帧级类型各自有对应的 CDF（`Use_Wiener_Cdf` / `Use_Sgrproj_Cdf` /
        // `Restoration_Type_Cdf`），全都走算术解码器。
        // 用 `read_bool()`（等概率伪原始位）读会**吃掉完全不同的概率模型**，
        // 于是从这一位开始整条算术流错位 —— 真实素材上表现为"瓦片读满能过、
        // 块级语法全崩"。
        if dec.read_symbol(&mut cdfs.use_wiener)? == 1 {
            RESTORE_WIENER
        } else {
            RESTORE_NONE
        }
    } else if frame_rtype == RESTORE_SGRPROJ {
        if dec.read_symbol(&mut cdfs.use_sgrproj)? == 1 {
            RESTORE_SGRPROJ
        } else {
            RESTORE_NONE
        }
    } else if frame_rtype == RESTORE_SWITCHABLE {
        // `RESTORE_SWITCHABLE` 下**只读一个符号**（规范 §5.11.48）：
        // `Default_Restoration_Type_Cdf`（= `cdfs.restoration_type`，3 个取值，
        // 就是 libaom 的 `switchable_restore_cdf = AOM_CDF3(9413, 22581)`）。
        //
        // ⚠️ **不是**"先读 `use_wiener`、再读 `use_sgrproj`、再读 `restoration_type`"
        // —— `Use_Wiener_Cdf` / `Use_Sgrproj_Cdf` 那两个 2 值 CDF 只属于
        // 帧级类型就是 `RESTORE_WIENER` / `RESTORE_SGRPROJ` 的两个分支。
        // 把三个符号串起来读会让**每个恢复单元多读 2 个符号** ⇒ 从第一个超块起
        // 整条算术流错位（真实 1080p 素材上表现为"每超块多读约 4 倍位、只解完
        // 2 行超块就把整片瓦片的位吃光"，且像素与参考完全对不上）。
        //
        // 取到的是**规范 §6.10.15 的编号**（0/1/2 = NONE/WIENER/SGRPROJ），
        // 必须过 [`RESTORATION_TYPE_REMAP`] 换成内部编号 —— 直接返回原始值会让
        // SGRPROJ 被当成 WIENER（真实素材报过 "restoration_type 越界：2"）。
        let t = dec.read_symbol(&mut cdfs.restoration_type)? as usize;
        if t >= RESTORATION_TYPE_REMAP.len() {
            return Err(MediaError::Bitstream(format!(
                "AV1 环路恢复的 restoration_type 越界：{t}"
            )));
        }
        RESTORATION_TYPE_REMAP[t]
    } else {
        return Err(MediaError::Bitstream(format!(
            "AV1 环路恢复不认识帧级 FrameRestorationType = {frame_rtype}"
        )));
    };
    state.lr_type[plane][slot] = restoration_type;

    if restoration_type == RESTORE_WIENER {
        for pass in 0..2usize {
            // 色度没有最外侧那对系数（规范：`coeff[0]` 恒为 0）
            let first_coeff = usize::from(plane > 0);
            if plane > 0 {
                state.wiener[plane][slot][pass][0] = 0;
            }
            for j in first_coeff..3 {
                let v = decode_signed_subexp_with_ref_bool(
                    dec,
                    WIENER_TAPS_MIN[j],
                    WIENER_TAPS_MAX[j] + 1,
                    WIENER_TAPS_K[j],
                    state.ref_wiener[plane][pass][j],
                )?;
                state.wiener[plane][slot][pass][j] = v;
                state.ref_wiener[plane][pass][j] = v;
            }
        }
    } else if restoration_type == RESTORE_SGRPROJ {
        let set = dec.read_literal(SGRPROJ_PARAMS_BITS)? as usize;
        if set >= SGR_PARAMS.len() {
            return Err(MediaError::Bitstream(format!(
                "AV1 环路恢复的 lr_sgr_set 越界：{set}"
            )));
        }
        state.sgr_set[plane][slot] = set as u8;
        for i in 0..2usize {
            let radius = SGR_PARAMS[set][i * 2];
            let (min, max) = (SGRPROJ_XQD_MIN[i], SGRPROJ_XQD_MAX[i]);
            let v = if radius != 0 {
                decode_signed_subexp_with_ref_bool(
                    dec,
                    min,
                    max + 1,
                    SGRPROJ_PRJ_SUBEXP_K,
                    state.ref_sgr_xqd[plane][i],
                )?
            } else if i == 1 {
                // radius == 0 ⇒ 第二路退化成"补足到 SGRPROJ_PRJ_BITS"
                let want = (1 << SGRPROJ_PRJ_BITS) - state.ref_sgr_xqd[plane][0];
                want.max(min).min(max)
            } else {
                0
            };
            state.sgr_xqd[plane][slot][i] = v;
            state.ref_sgr_xqd[plane][i] = v;
        }
    }
    Ok(())
}

/// `SGRPROJ_PRJ_BITS`（§6.10.15）：投影权重的精度。
const SGRPROJ_PRJ_BITS: u32 = 7;

/// `decode_signed_subexp_with_ref_bool( low, high, k, r )`（规范 §5.11.48 之后）。
fn decode_signed_subexp_with_ref_bool(
    dec: &mut SymbolDecoder<'_>,
    low: i32,
    high: i32,
    k: i32,
    r: i32,
) -> Result<i32> {
    let x = decode_unsigned_subexp_with_ref_bool(dec, high - low, k, r - low)?;
    Ok(x + low)
}

/// `decode_unsigned_subexp_with_ref_bool( mx, k, r )`。
fn decode_unsigned_subexp_with_ref_bool(
    dec: &mut SymbolDecoder<'_>,
    mx: i32,
    k: i32,
    r: i32,
) -> Result<i32> {
    let v = decode_subexp_bool(dec, mx, k)?;
    if (r << 1) <= mx {
        Ok(inverse_recenter(r, v))
    } else {
        Ok(mx - 1 - inverse_recenter(mx - 1 - r, v))
    }
}

/// `decode_subexp_bool( numSyms, k )`（规范 §5.11.48 之后）：子指数哥伦布的
/// **符号解码器版本**（`L()` / `NS()` 都走算术解码器）。
///
/// 与 `params.rs` 里那个位读器版本（`BitReader::f` / `ns`）逻辑相同，
/// 区别只在"位从哪来"，以及这里的 `k` 是参数（位读器版本里写死 3）。
fn decode_subexp_bool(dec: &mut SymbolDecoder<'_>, num_syms: i32, k: i32) -> Result<i32> {
    if num_syms <= 0 {
        return Ok(0);
    }
    let mut i = 0i32;
    let mut mk = 0i32;
    for _ in 0..32 {
        let b2 = if i != 0 { k + i - 1 } else { k };
        if b2 < 0 || b2 > 30 {
            return Err(MediaError::Bitstream(format!(
                "AV1 decode_subexp_bool：b2 = {b2} 越界（位读错位）"
            )));
        }
        let a = 1i32 << b2;
        if num_syms <= mk + 3 * a {
            let bits = dec.read_ns((num_syms - mk) as u32)? as i32;
            return Ok(bits + mk);
        }
        if dec.read_bool()? {
            i += 1;
            mk += a;
        } else {
            let bits = dec.read_literal(b2 as u32)? as i32;
            return Ok(bits + mk);
        }
    }
    Err(MediaError::Bitstream(
        "AV1 decode_subexp_bool 超过 32 轮（位读错位）".into(),
    ))
}

/// `inverse_recenter( r, v )` 直接用 `params.rs` 里那一份（`global_motion_params`
/// 的位读器路径与这里共用同一个函数）—— **绝不在这里再抄一份**。
use super::params::inverse_recenter;

/// `count_units_in_frame( unitSize, frameSize )`（规范 §5.11.48 之后）。
///
/// 注意是 `(frameSize + unitSize/2) / unitSize` —— **四舍五入**，不是向上取整，
/// 而且至少要 1 个。
fn count_units_in_frame(unit_size: i32, frame_size: i32) -> i32 {
    ((frame_size + (unit_size >> 1)) / unit_size).max(1)
}

/// `Round2( x, n )`。
fn round2(x: i32, n: u32) -> i32 {
    if n == 0 {
        x
    } else {
        (x + (1 << (n - 1))) >> n
    }
}

/// plane 的色度抽样位数（亮度是 `(0, 0)`）。
fn plane_subsampling(plane: usize, seq: &SequenceHeader) -> (u32, u32) {
    if plane == 0 {
        (0, 0)
    } else {
        (
            u32::from(seq.color.subsampling_x),
            u32::from(seq.color.subsampling_y),
        )
    }
}

/// `SGRPROJ_RST_BITS`（§6.10.15）。
const SGRPROJ_RST_BITS: u32 = 4;
/// `SGRPROJ_MTABLE_BITS`。
const SGRPROJ_MTABLE_BITS: u32 = 20;
/// `SGRPROJ_RECIP_BITS`。
const SGRPROJ_RECIP_BITS: u32 = 12;
/// 环路恢复的**处理档宽**（亮度像素）。
///
/// 2026-10-03：`restore_block` 原先按 **4×4 块**调用，`box_filter` 的 A/B 滑动和
/// 因此把同一批样本重复算 4~6 倍（半径 2 时一块 16 个输出要铺 10×10 的 A/B 窗口）、
/// 每 4 个像素就要付一次函数调用与定长缓冲清零。现在**按 64 像素一档**处理
/// （两轴都是 2 的幂、恢复单元尺寸 ≥ 64 且起点对齐 ⇒ 一档必定落在**同一个恢复单元**
/// 内，`lr_type_at` 的查表结果不变；`get_source_sample` 的钳位是**逐样本**语义，
/// 与处理档的划分无关 ⇒ 逐位不变）。
///
/// 64 的取法：再宽（128/256）就要把下面的定长缓冲换成按 worker 的堆缓冲（~两轴
/// 一起放大会到几百 KB），收益递减；64 时各缓冲最大 ~70 KB，仍可留在栈上。
///
/// ## 竖直方向也按档（2026-10-04）
///
/// 水平按档只解决了 `box_filter` 的**水平**相位；竖直相位原先仍按 `MI_SIZE`（亮度
/// 4 行、色度 2 行）走，于是每 4 行就把整个恢复块重来一次：`box_filter` 要重新累加
/// 初始的 `2r+1` 行窗口、`sgr_filter`/`wiener_filter` 要重新 `fill(0)` 上万项的
/// 中间表、`restore_block` 要重算一遍 stripe/unit。
///
/// 竖直档可以安全取到 **64 行**（亮度），理由是三条边界**同源**：
///
/// * `stripe_num = (lumaY + 8) / 64`（§7.17.1 的"64 行一条、向上偏 8"）；
/// * `unit_row = ((lumaY + 8) >> sub_y) >> log2(unit_size)`，而 `RestorationUnitSize`
///   只有 **64 / 128 / 256** —— 都是 64 的倍数；
///
/// ⇒ 只要档不跨 `(lumaY + 8)` 的 64 倍数，`stripe_start_y` / `stripe_end_y` /
/// `unit_row` / `rtype` **在档内全部恒定**。`get_source_sample` 的钳位又是逐样本
/// 语义（`box_filter` / `wiener_filter` 的行解析表本来就逐行判 stripe），
/// 所以档的划分不影响任何取值 ⇒ **逐位不变**。
///
/// ⚠️ 并行回放的 worker **行区间必须吸附到档边界**（`16m - 2` 个 MI 行），
/// 否则一个档被切在 worker 交界处、后半段会误用前一个 `unit_row` 的 `rtype`。
/// 见 [`apply`] 里的切分。
const LR_CHUNK: i32 = 64;
/// `box_filter` 的竖直和缓冲列数上限：`64 + 2 + 2*2 = 70`（留余量写 72）。
const SGR_BOX_COLS_MAX: usize = 72;
/// `sgr_filter` 的 a/b 中间表上限：`(64+2) × (64+2) = 4356`（留余量写 4400）。
const SGR_AB_MAX: usize = 4400;
/// `box_filter` 竖直相位的**行解析表**上限：`h + 2r + 2 ≤ 64 + 4 + 2 = 70`（留余量写 72）。
const SGR_BOX_ROWS_MAX: usize = 72;
/// `wiener_filter` 的列解析表上限：`w + 7 ≤ 71`（留余量写 72）。
const WIENER_COLS_MAX: usize = 72;
/// `wiener_filter` 的行解析表上限：`h + 6 ≤ 70`（留余量写 72）。
const WIENER_ROWS_MAX: usize = 72;

/// `x_by_xplus1[ z ]`（§7.17.3 的 `a2`）：`((z << 8) + (z >> 1)) / (z + 1)`。
///
/// 规范把 `a2` 写成一个**除法**，但它只依赖 `z ∈ [0, 254]` —— libaom 用同一张
/// 256 项表把它换成查表（`av1/common/restoration.c` 的 `x_by_xplus1`）。
/// 我们**自己按公式算出来**（`const` 求值，编译期完成），与逐样本做除法**逐位同值**：
/// `z == 0 → 1`、`z ≥ 255 → 256` 是规范的两个特例，表里 `t[0] = 1`，`≥255` 由调用方走分支。
const X_BY_XPLUS1: [i32; 256] = {
    let mut t = [0i32; 256];
    t[0] = 1;
    let mut z = 1usize;
    while z < 256 {
        t[z] = (((z as i32) << SGRPROJ_SGR_BITS) + (z as i32) / 2) / (z as i32 + 1);
        z += 1;
    }
    t
};
/// `sgr_filter` 的 F 上限（`64 × 64`，见 [`LR_CHUNK`]）。
const SGR_F_MAX: usize = 4096;
/// `wiener_filter` 中间量的栈上上限（`64 × (64+6)`）。
const WIENER_MAX_CELLS: usize = 4480;
/// `SGRPROJ_SGR_BITS`。
const SGRPROJ_SGR_BITS: u32 = 8;
/// `FILTER_BITS`（§7.11.3.2）：Wiener 与插值共用。
const FILTER_BITS: u32 = 7;

/// `LrFrame = LR(UpscaledCurrFrame, UpscaledCdefFrame)`（规范 §7.17）。
///
/// 逐 **4×4 块**遍历（`x += 4` / `y += 4`），每个块查自己所属 **restoration unit**
/// 的类型与系数。`UsesLr = 0` 时规范就是"整份拷贝 `CdefFrame`"。
///
/// ## 三个关键点
///
/// 1. **stripe（64 行一条，向上偏 8 行）决定取样来源**：stripe **内**取
///    `UpscaledCdefFrame`（去过 CDEF），stripe **外**取 `UpscaledCurrFrame`
///    （只去过块）。这是"环路恢复不需要额外行缓冲"的实现方式。
/// 2. **`get_source_sample` 的坐标要钳两次**：先钳到 `[0, PlaneEnd]`，
///    再按所在侧钳到 `[StripeStart-2, StripeEnd+2]`（第三条线拿第二条的副本）。
/// 3. **SGR 的 `w0 + w1 + w2 == 1 << SGRPROJ_PRJ_BITS`**，且 `radius == 0` 的那一路
///    退化成 `u`（不滤波）—— 见 §7.17.2 的 `if ( r0 ) ... else ...`。
/// 诊断开关（`AV1_TRACE_LR=1`）：每帧把"几何 + 量化 + LR 参数"打一行到 stderr。
fn trace_lr() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("AV1_TRACE_LR").is_ok())
}

/// 诊断（`AV1_LR_DUMP=<目录>`）：把 LR 的**输入**（`UpscaledCdefFrame`）与**输出**
/// （`LrFrame`）按平面落盘，用来与 libaom 的同一阶段逐步比对 ——
/// "像素差到底由环路恢复引入、还是滤波之前就有"一眼可判，省掉大量猜测。
///
/// 文件格式：`ours_{prelr,postlr}_<帧号>.yuv`，**planar**（Y、U、V 依次紧密排）。
pub(crate) fn lr_dump(frame: &CurrFrame, dir: &str, tag: &str, n: u64) {
    use std::io::Write;
    let path = format!("{dir}/ours_{tag}_{n:03}.yuv");
    let Ok(mut f) = std::fs::File::create(&path) else {
        return;
    };
    for p in &frame.planes {
        let mut buf = Vec::with_capacity((p.w as usize) * (p.h as usize));
        for y in 0..p.h {
            for x in 0..p.w {
                buf.push(p.get(x, y).clamp(0, 255) as u8);
            }
        }
        let _ = f.write_all(&buf);
    }
}

/// ⚠️ `up_curr` / `up_cdef` 是**按值**传进来的（2026-10-02 改）：所有权进来之后
/// "不做环路恢复"那一档就能把 `CdefFrame` **直接移动**成 `LrFrame`，
/// 零拷贝 —— 而真素材上这一档是绝大多数帧（ASMR 1080p 实测：8 帧里 6 帧
/// `uses_lr=false`）。`sample` 归因显示那次 `up_cdef.clone()` 的 memmove
/// 占 **3.07%**（整帧 6 MB），是本函数最大的单项开销。
///
/// 需要真的滤波时（`uses_lr == true`）仍然要开第三份 `out`：环路恢复的输出
/// 既不回灌 `CdefFrame` 也不回灌 `CurrFrame`（规范 §7.17 就是这么定的），
/// 两份源都必须**同时**可读。这时 `up_curr`/`up_cdef` 退化成只读借用。
pub fn apply(
    up_curr: CurrFrame,
    up_cdef: Option<CurrFrame>,
    fh: &FrameHeader,
    seq: &SequenceHeader,
    lr: &LrState,
) -> Result<CurrFrame> {
    // 诊断（`AV1_TRACE_LR`）：一行回答"这一帧到底用了什么" —— 排查真实素材时最省时间的一条。
    if trace_lr() {
        eprintln!(
            "[FRAME] {}x{} sb128={} q={} | LR uses={} chroma={} types={:?} size={:?} \
             unit_shift={}",
            up_cdef.as_ref().unwrap_or(&up_curr).planes[0].w,
            up_cdef.as_ref().unwrap_or(&up_curr).planes[0].h,
            seq.use_128x128_superblock,
            fh.params.quant.base_q_idx,
            fh.params.lr.uses_lr,
            fh.params.lr.uses_chroma_lr,
            fh.params.lr.frame_restoration_type,
            fh.params.lr.size,
            fh.params.lr.unit_shift
        );
    }
    let dump_dir = flags::lr_dump_dir();
    let dump_n = super::block::TRACE_FRAME_NO.load(std::sync::atomic::Ordering::Relaxed);
    if let Some(d) = &dump_dir {
        lr_dump(up_cdef.as_ref().unwrap_or(&up_curr), d, "prelr", dump_n);
    }
    if !fh.params.lr.uses_lr || flags::no_lr() {
        // 这一档的输出**就是** `CdefFrame` 本身 —— 直接移动，不再 `clone()`。
        // 上一层没做 CDEF 时（`None`）它就是 `CurrFrame`，连"两个缓冲"都不用 ——
        // `up_curr` 直接当输出，零拷贝、零分配。
        let out = up_cdef.unwrap_or(up_curr);
        if let Some(d) = &dump_dir {
            lr_dump(&out, d, "postlr", dump_n);
        }
        return Ok(out);
    }
    // 需要真滤波：两份源都得可读 ⇒ 从所有权降级成借用（`out` 是第三份）。
    // 没做 CDEF 时 `CdefFrame ≡ CurrFrame`，直接借用同一份（规范语义允许）。
    let up_curr = &up_curr;
    let up_cdef: &CurrFrame = up_cdef.as_ref().unwrap_or(up_curr);
    // `out` 的**每一个**像素都会被下面的恢复块写一遍 —— `RESTORE_NONE` 那一档
    // 也按块从 `CdefFrame` 把样本搬过来（见 `restore_block`）。所以**不需要**先
    // `up_cdef.clone()` 整份拷贝：1080p 一帧省掉 6 MB 的 memmove，而且
    // `CurrFrame::new` 里那次填充是 memset（resident 内存），比新 mmap 一页页
    // 缺页 + 内核清零便宜得多。
    // ⚠️ 前提是"覆盖满"：x/y 两个循环按 `MI_SIZE` 走满整帧，`w`/`h` 已按平面
    //    边界裁剪，三档（wiener / sgr / none）都写。改这段循环时务必保持这条。
    let mut out = CurrFrame::new(
        up_cdef.planes[0].w,
        up_cdef.planes[0].h,
        up_cdef.sub_x,
        up_cdef.sub_y,
        up_cdef.num_planes,
        up_cdef.planes[0].bit_depth,
    );
    let upscaled_width = up_cdef.planes[0].w as i32;
    let frame_height = up_cdef.planes[0].h as i32;
    let num_planes = up_cdef.num_planes;

    // `frame_restoration_type[plane] == RESTORE_NONE` 的平面**整帧都不滤波**：
    // 规范里它的输出就是 `CdefFrame` 的一份拷贝。这里按**平面**整块搬（连 padding
    // 一起 `copy_from_slice`，与"预先 clone 整帧"的旧行为逐字节一致），
    // 而不是把整帧都拷一遍 —— 每个平面只需要一次连续 memcpy。
    for plane in 0..num_planes {
        if fh.params.lr.frame_restoration_type[plane] == RESTORE_NONE {
            out.planes[plane].px16_mut().copy_from_slice(up_cdef.planes[plane].px16());
        }
    }

    // 每个平面的**恢复单元几何是常量**：`RestorationUnitSize ∈ {64,128,256}`（2 的幂
    // ⇒ 除法换移位）、`count_units_in_frame` 与两个除法原先在 `restore_block` 里
    // **每 4×4 块各算一次**（1080p 一帧几十万次）。这里每平面算一次。
    // 定长数组（`num_planes ≤ 3`）：元组是 `Copy`，move 闭包里不会牵出所有权问题。
    let mut lr_unit_geom = [(0i32, 0i32, 0u32); 3];
    for plane in 0..num_planes {
        let (sx, sy) = plane_subsampling(plane, seq);
        let unit_size = (fh.params.lr.size[plane] as i32).max(1);
        let shift = unit_size.trailing_zeros();
        debug_assert_eq!(1 << shift, unit_size, "RestorationUnitSize 应当是 2 的幂");
        let rows = count_units_in_frame(unit_size, round2(frame_height, sy));
        let cols = count_units_in_frame(unit_size, round2(upscaled_width, sx));
        lr_unit_geom[plane] = (rows, cols, shift);
    }

    // 每个恢复块只读 `CurrFrame` / `CdefFrame`、只写**第三份** `out`（规范就是这么
    // 定的：环路恢复的输出既不回灌 CDEF 帧，也不回灌当前帧），所以按 **MI 行**切分
    // 就能并行 —— 各 worker 写的像素行互不重叠。
    let rows = (frame_height as usize).div_ceil(MI_SIZE as usize);
    let workers = super::parallel::worker_count().min(rows.max(1)).max(1);
    let per = rows.div_ceil(workers).max(1);
    let sub_y_seq = i32::from(seq.color.subsampling_y);
    let mut ranges: Vec<Vec<usize>> = Vec::new();
    let mut row_chunks: Vec<(usize, usize)> = Vec::new();
    let mut a = 0usize;
    while a < rows {
        let b0 = (a + per).min(rows);
        // ⚠️ 把切分点**吸附到竖直档边界**（`lumaY = 64m - 8` ⇒ MI 行 `16m - 2`）。
        // 竖直档内 `unit_row` 恒定，一个档若被切在 worker 交界处，后半段会拿前
        // 一个 `unit_row` 的 `rtype` —— 那是错的。吸附只会让 `b` 变大（`t ≥ b0`），
        // 所以仍严格大于 `a`，循环不会卡住。
        let b = if b0 < rows {
            let t = (b0 + 2).div_ceil(16) * 16 - 2;
            t.min(rows)
        } else {
            rows
        };
        let mut v = Vec::with_capacity(out.planes.len());
        for (p, plane) in out.planes.iter().enumerate() {
            let sy = if p == 0 { 0 } else { sub_y_seq };
            let end = (((b * MI_SIZE as usize) as i32) >> sy).max(0) as usize;
            let _ = plane;
            v.push(end);
        }
        ranges.push(v);
        row_chunks.push((a, b));
        a = b;
    }
    let sinks = super::parallel::split_frame_rows(&mut out.planes, &ranges);

    let pool: Vec<_> = row_chunks
        .iter()
        .zip(sinks.iter())
        .map(|(&(y0, y1), sink)| {
            move || -> Result<()> {
                let mut y = (y0 * MI_SIZE as usize) as i32;
                let y_end = (y1 * MI_SIZE as usize) as i32;
                while y < y_end {
                    // 竖直按档（见 [`LR_CHUNK`]）：推进到下一个 `(lumaY + 8)` 的
                    // 64 倍数边界。切分已吸附到同一族边界，所以正常情况下
                    // `next ≥ y_end` 不会真的截断；`.min(y_end - y)` 只是兜底。
                    let next = (((y + 8) >> 6) + 1) * 64 - 8;
                    let chunk_rows = ((next - y).min(y_end - y) >> 2).max(1);
                    let mut x = 0i32;
                    while x < upscaled_width {
                        for plane in 0..num_planes {
                            if fh.params.lr.frame_restoration_type[plane] == RESTORE_NONE {
                                continue;
                            }
                            let (unit_rows, unit_cols, unit_shift) = lr_unit_geom[plane];
                            restore_block(
                                up_curr,
                                up_cdef,
                                &sink[plane],
                                seq,
                                lr,
                                plane,
                                y >> 2, // lumaY = row * MI_SIZE ⇒ row = y >> 2
                                x >> 2,
                                LR_CHUNK,
                                chunk_rows,
                                upscaled_width,
                                frame_height,
                                unit_rows,
                                unit_cols,
                                unit_shift,
                            )?;
                        }
                        x += LR_CHUNK;
                    }
                    y += chunk_rows * MI_SIZE;
                }
                Ok(())
            }
        })
        .collect();

    if pool.len() <= 1 {
        for f in pool {
            f()?;
        }
    } else {
        let mut first_err: Option<MediaError> = None;
        std::thread::scope(|scope| {
            let handles: Vec<_> = pool.into_iter().map(|f| scope.spawn(f)).collect();
            for h in handles {
                match h.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        if first_err.is_none() {
                            first_err = Some(e);
                        }
                    }
                    Err(_) => {
                        if first_err.is_none() {
                            first_err = Some(MediaError::Bitstream(
                                "环路恢复并行的某个线程 panic 了".into(),
                            ));
                        }
                    }
                }
            }
        });
        if let Some(e) = first_err {
            return Err(e);
        }
    }
    if let Some(d) = &dump_dir {
        lr_dump(&out, d, "postlr", dump_n);
    }
    Ok(out)
}

/// `loop_restore_block( plane, row, col )`（规范 §7.17.1）。
#[allow(clippy::too_many_arguments)]
fn restore_block(
    up_curr: &CurrFrame,
    up_cdef: &CurrFrame,
    out: &super::parallel::RowsSink,
    seq: &SequenceHeader,
    lr: &LrState,
    plane: usize,
    row: i32,
    col: i32,
    // 本档的**亮度**宽度（见 [`LR_CHUNK`]）：`row`/`col` 只是这一档左上角的亮度
    // 4×4 坐标，挡宽由调用方给（`w` 从它折算并钳到平面右边界）。
    chunk: i32,
    // 本档的**亮度**高度（MI 行数 ≥ 1，见 [`LR_CHUNK`]）：`h` 从它折算并钳到平面
    // 下边界。竖直档保证档内 `stripe_start_y` / `stripe_end_y` / `unit_row` 恒定
    // （三条边界同源于 `(lumaY + 8)` 的 64 倍数），所以这里只算一次 `rtype`。
    chunk_rows: i32,
    upscaled_width: i32,
    frame_height: i32,
    unit_rows: i32,
    unit_cols: i32,
    unit_shift: u32,
) -> Result<()> {
    let (sub_x, sub_y) = plane_subsampling(plane, seq);
    let luma_y = row * MI_SIZE;
    // stripe 是"64 行一条、向上偏 8 行"（LumaY + 8）
    let stripe_num = (luma_y + 8) / 64;
    let stripe_start_y = (-8 + stripe_num * 64) >> sub_y;
    let stripe_end_y = stripe_start_y + (64 >> sub_y) - 1;

    // `unit_rows` / `unit_cols` / `unit_shift`（= log2(unit_size)）由 `apply` 每平面
    // 预计算（`RestorationUnitSize` 是 2 的幂 ⇒ `/ unit_size` 换成 `>> unit_shift`）。
    //
    // 规范 §7.17.1：上界是 **`unitRows - 1`**（与 `unitCols - 1`）——最后一个恢复单元
    // 吸收帧底/帧右的余量。写成 `Min(unitRows, …)` 会让帧底那一带取到**越界索引**：
    // `LrType` 每个 plane 只有 `unitRows * unitCols` 项，`lr_type_at` 用 `.get()` 取，
    // 于是**静默返回 `RESTORE_NONE`** ⇒ 这 64 行完全不滤波，也不会走到
    // `wiener_filter`/`sgr_filter`（那里才有越界报错），所以**一路无声**。
    // 症状极隐蔽：1920×1080 的帧只差最底 64 行（y ∈ [1016, 1079]）、0.36% 像素、
    // 幅度 1~2 —— 很容易被当成"舍入误差"放过去。
    let unit_row = (unit_rows - 1).min(((row * MI_SIZE + 8) >> sub_y) >> unit_shift);
    let unit_col = (unit_cols - 1).min(((col * MI_SIZE) >> sub_x) >> unit_shift);

    let plane_end_x = round2(upscaled_width, sub_x) - 1;
    let plane_end_y = round2(frame_height, sub_y) - 1;
    let x = (col * MI_SIZE) >> sub_x;
    let y = (row * MI_SIZE) >> sub_y;
    let w = (chunk >> sub_x).min(plane_end_x - x + 1);
    let h = ((chunk_rows * MI_SIZE) >> sub_y).min(plane_end_y - y + 1);
    if w <= 0 || h <= 0 {
        return Ok(());
    }

    let rtype = lr.lr_type_at(plane, unit_row as usize, unit_col as usize);
    let src = SourceSample {
        up_curr,
        up_cdef,
        plane,
        stripe_start_y,
        stripe_end_y,
        plane_end_x,
        plane_end_y,
    };
    if rtype == RESTORE_WIENER {
        wiener_filter(
            &src, out, lr, plane, unit_row as usize, unit_col as usize, x, y, w, h, seq,
        );
    } else if rtype == RESTORE_SGRPROJ {
        sgr_filter(
            &src, out, lr, plane, unit_row as usize, unit_col as usize, x, y, w, h, seq,
        )?;
    } else {
        // `rtype == RESTORE_NONE`：这一档不滤波，输出就是 `CdefFrame` 的对应样本。
        // 以前靠"`out` 预先整份 clone `up_cdef`"来实现，现在**按块搬**——
        // 这是省掉整帧拷贝的前提，别退回去。
        let s = &up_cdef.planes[plane];
        let data = s.px16();
        let stride = s.stride;
        // 按行搬：一次 `set_row` 写完一整行的 `w` 个样本（逐像素 `set` 要 4 倍
        // 次数的函数调用 + 边界检查；CDEF 那条路早就这么做了）。
        let mut row = [0i32; LR_CHUNK as usize];
        for i in 0..h {
            let off = (y + i) as usize * stride + x as usize;
            for (j, slot) in row[..w as usize].iter_mut().enumerate() {
                *slot = i32::from(data[off + j]);
            }
            out.set_row(x as usize, (y + i) as usize, &row[..w as usize]);
        }
    }
    Ok(())
}

/// `get_source_sample`（§7.17.6）的上下文。
struct SourceSample<'a> {
    up_curr: &'a CurrFrame,
    up_cdef: &'a CurrFrame,
    plane: usize,
    stripe_start_y: i32,
    stripe_end_y: i32,
    plane_end_x: i32,
    plane_end_y: i32,
}

impl SourceSample<'_> {
    /// `get_source_sample( plane, x, y )`：stripe 内取 `CdefFrame`、外取 `CurrFrame`。
    ///
    /// ⚠️ **内核不再直接调它**（2026-10-02 起 `box_filter` / `wiener_filter` 用
    /// 预解析的"逐行/逐列"表，把两次钳位与 stripe 判定从逐样本搬到逐行），
    /// 但保留这个方法当**规范参照**：那两张表就是把这里的逻辑拆开实现的。
    #[allow(dead_code)]
    fn get(&self, x: i32, y: i32) -> i32 {
        let x = x.clamp(0, self.plane_end_x);
        let y = y.clamp(0, self.plane_end_y);
        let (y, from_curr) = if y < self.stripe_start_y {
            ((self.stripe_start_y - 2).max(y), true)
        } else if y > self.stripe_end_y {
            ((self.stripe_end_y + 2).min(y), true)
        } else {
            (y, false)
        };
        let frame = if from_curr { self.up_curr } else { self.up_cdef };
        i32::from(frame.planes[self.plane].get(x as usize, y as usize))
    }
}

/// `Wiener filter process`（规范 §7.17.4）+ `Wiener coefficient process`（§7.17.5）。
#[allow(clippy::too_many_arguments)]
fn wiener_filter(
    src: &SourceSample<'_>,
    out: &super::parallel::RowsSink,
    lr: &LrState,
    plane: usize,
    unit_row: usize,
    unit_col: usize,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    seq: &SequenceHeader,
) {
    let slot = unit_row * lr.unit_cols[plane] + unit_col;
    let coeffs = lr.wiener[plane][slot];
    let bit_depth = seq.color.bit_depth as u32;
    // Wiener 的 `isCompound = 0`
    let inter_round0 = if bit_depth == 12 { 5 } else { 3 };
    let inter_round1 = if bit_depth == 12 { 9 } else { 11 };

    let vfilter = wiener_coeffs(coeffs[0]);
    let hfilter = wiener_coeffs(coeffs[1]);

    let offset = 1i32 << (bit_depth + FILTER_BITS - inter_round0 - 1);
    let limit = (1i32 << (bit_depth + 1 + FILTER_BITS - inter_round0)) - 1;
    // 中间量也放**栈上**：竖直按档后 `w`/`h` ≤ 64，所以最多
    // `(64 + 6) × 64 = 4480` 项。
    let mut interm_buf = [0i32; WIENER_MAX_CELLS];
    {
        let need = ((h + 6) * w) as usize;
        debug_assert!(need <= WIENER_MAX_CELLS);
        // `interm` **不需要预清零**：下面的双重循环把 `0..(h+6) × 0..w` 全量写满
        // （`need` 正是这个尺寸）—— 竖直按档后这份缓冲到了 ~18 KB，每次调用
        // `fill(0)` 在真素材上一帧要白写十几 MB。
        let interm = &mut interm_buf[..need];
        // 源解析表（与 `box_filter` 同一个套路）：水平窗口用到 `x-3 .. x+w+3`（w+7 列）、
        // 竖直 `y-3 .. y+h+2`（h+6 行）。逐样本 `src.get` 每次要两次钳位 + stripe 判定，
        // 而这里一个块要取 (h+6)·w·7 = 280 次 —— 解析表把这块全省掉。
        let stride = src.up_curr.planes[src.plane].stride;
        let ncols = (w + 7) as usize;
        let nrows = (h + 6) as usize;
        debug_assert!(ncols <= WIENER_COLS_MAX && nrows <= WIENER_ROWS_MAX);
        let mut xs = [0usize; WIENER_COLS_MAX];
        for (k, slot) in xs[..ncols].iter_mut().enumerate() {
            *slot = (x - 3 + k as i32).clamp(0, src.plane_end_x) as usize;
        }
        let mut rows_ptr = [std::ptr::null::<u16>(); WIENER_ROWS_MAX];
        let mut rows_off = [0usize; WIENER_ROWS_MAX];
        for k in 0..nrows {
            let yy = (y - 3 + k as i32).clamp(0, src.plane_end_y);
            let (yy, from_curr) = if yy < src.stripe_start_y {
                ((src.stripe_start_y - 2).max(yy), true)
            } else if yy > src.stripe_end_y {
                ((src.stripe_end_y + 2).min(yy), true)
            } else {
                (yy, false)
            };
            let frame = if from_curr { src.up_curr } else { src.up_cdef };
            rows_ptr[k] = frame.planes[src.plane].px16().as_ptr();
            rows_off[k] = (yy as usize) * stride;
        }
        // `read(k_col, k_row)` ≡ `src.get(x - 3 + k_col, y - 3 + k_row)`（已解析）
        let read = |kc: usize, kr: usize| -> i32 {
            // SAFETY: 坐标已按 `get` 的规则钳到帧内；两份源在本函数里都只读。
            unsafe { *rows_ptr[kr].add(rows_off[kr] + xs[kc]) as i32 }
        };
        // 列窗口不触边 ⇒ 列解析表等差（`xs[c] = x - 3 + c`），横向趟可以 8 列一组连续读。
        let interior_x = x - 3 >= 0 && x + w + 3 <= src.plane_end_x;
        let simd_h = interior_x && w >= 8 && super::simd::avx2_available();
        let simd_v = w >= 8 && super::simd::avx2_available();
        if simd_h {
            // SAFETY: 已确认 AVX2；行表按 `get` 的规则解析过（窗口都在平面缓冲内）、
            // 列窗口不触边 ⇒ 读取范围全在缓冲内；`interm` 恰好 `(h + 6) × w`。
            unsafe {
                super::simd::wiener_horiz_rows(
                    &rows_ptr[..nrows],
                    &rows_off[..nrows],
                    nrows,
                    (x - 3) as isize,
                    w as usize,
                    &hfilter,
                    inter_round0,
                    -offset,
                    limit - offset,
                    interm,
                );
            }
        } else {
            for r in 0..(h + 6) {
                for c in 0..w {
                    let mut s = 0i32;
                    for t in 0..7 {
                        s += hfilter[t as usize] * read((c + t) as usize, r as usize);
                    }
                    let v = round2(s, inter_round0);
                    interm[(r * w + c) as usize] = v.clamp(-offset, limit - offset);
                }
            }
        }
        let mut row = [0i32; LR_CHUNK as usize];
        if simd_v {
            let max = (1i32 << bit_depth) - 1;
            for r in 0..h {
                // SAFETY: 已确认 AVX2；`interm` 有 `(h + 6) × w` 项，本行读到 `(r + 6) × w + w`。
                unsafe {
                    super::simd::wiener_vert_row(
                        interm,
                        w as usize,
                        r as usize,
                        &vfilter,
                        inter_round1,
                        max,
                        &mut row[..w as usize],
                    );
                }
                out.set_row(x as usize, (y + r) as usize, &row[..w as usize]);
            }
        } else {
            for r in 0..h {
                for (c, slot) in row[..w as usize].iter_mut().enumerate() {
                    let mut s = 0i32;
                    for t in 0..7 {
                        s += vfilter[t as usize] * interm[((r + t) * w + c as i32) as usize];
                    }
                    let v = round2(s, inter_round1);
                    *slot = i32::from(super::intra::clip1(v, bit_depth as u8));
                }
                out.set_row(x as usize, (y + r) as usize, &row[..w as usize]);
            }
        }
    }
}

/// `Wiener coefficient process`（§7.17.5）：3 个系数 → 7 抽头（对称、DC 增益为 1）。
fn wiener_coeffs(coeff: [i32; 3]) -> [i32; 7] {
    let mut filter = [0i32; 7];
    filter[3] = 128;
    for i in 0..3 {
        let c = coeff[i];
        filter[i] = c;
        filter[6 - i] = c;
        filter[3] -= 2 * c;
    }
    filter
}

/// `Self guided filter process`（规范 §7.17.2）+ `Box filter process`（§7.17.3）。
#[allow(clippy::too_many_arguments)]
fn sgr_filter(
    src: &SourceSample<'_>,
    out: &super::parallel::RowsSink,
    lr: &LrState,
    plane: usize,
    unit_row: usize,
    unit_col: usize,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    seq: &SequenceHeader,
) -> Result<()> {
    let slot = unit_row * lr.unit_cols[plane] + unit_col;
    let set = lr.sgr_set[plane][slot] as usize;
    if set >= SGR_PARAMS.len() {
        return Err(MediaError::Bitstream(format!(
            "AV1 SGR 的 lr_sgr_set 越界：{set}"
        )));
    }
    let [w0, w1] = lr.sgr_xqd[plane][slot];
    let w2 = (1 << SGRPROJ_PRJ_BITS) - w0 - w1;
    let r0 = SGR_PARAMS[set][0];
    let r1 = SGR_PARAMS[set][2];
    // 四块中间缓冲**全部放在栈上**：竖直按档后一档最大 64×64，于是 a/b 上限
    // `66×66 = 4356` 项、F 上限 `64×64 = 4096` 项 —— 定长数组就够，既不用 `vec![]`
    // （一帧几十万次分配），也不用穿线程局部（TLS 访问本身在 1080p 上是可观开销，
    // 采样里一度占到 ~24%）。
    let mut a = [0i32; SGR_AB_MAX];
    let mut b = [0i32; SGR_AB_MAX];
    let mut f0 = [0i32; SGR_F_MAX];
    let mut f1 = [0i32; SGR_F_MAX];
    let need_f = (w * h) as usize;
    let need_ab = ((w + 2) * (h + 2)) as usize;
    debug_assert!(need_ab <= SGR_AB_MAX && need_f <= SGR_F_MAX);
    // `f0` / `f1` **不需要预清零**：`box_filter` 三条出口（`r == 0` 时自己
    // `fill(0)`、否则末段按行写满 `w × h`）都会覆盖 `need_f` 全部。
    let (f0s, f1s) = (&mut f0[..need_f], &mut f1[..need_f]);
    box_filter(src, set, 0, x, y, w, h, seq, f0s, &mut a[..need_ab], &mut b[..need_ab])?;
    box_filter(src, set, 1, x, y, w, h, seq, f1s, &mut a[..need_ab], &mut b[..need_ab])?;
    let mut row = [0i32; LR_CHUNK as usize];
    for i in 0..h {
        for j in 0..w {
            let u = i32::from(
                src.up_cdef.planes[plane].get((x + j) as usize, (y + i) as usize),
            ) << SGRPROJ_RST_BITS;
            let idx = (i * w + j) as usize;
            let mut v = w1 * u;
            v += if r0 != 0 { w0 * f0[idx] } else { w0 * u };
            v += if r1 != 0 { w2 * f1[idx] } else { w2 * u };
            let s = round2(v, SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS);
            row[j as usize] = i32::from(super::intra::clip1(s, seq.color.bit_depth));
        }
        out.set_row(x as usize, (y + i) as usize, &row[..w as usize]);
    }
    Ok(())
}

/// `Box filter process`（规范 §7.17.3）：`pass` 为 0 / 1。
///
/// 输出 `F[i][j]`（尺寸 `w × h`，写进调用方的 `f`），中间表 `A` / `B`
/// （要多算一圈，尺寸 `(w+2) × (h+2)`，写进 `a_buf` / `b_buf`）。
///
/// ## 为什么不让本函数自己 `vec![]`
///
/// 环路恢复一帧要调几十万次（竖直按档前；按档后仍有数千次），每次都新分配三块
/// 数组会把分配器打爆（`sample` 采样里 `malloc`/`free` 合计约 9%）。所以三块缓冲
/// 由调用方（`sgr_filter` 的线程局部池）提供、只增不减地复用。
///
/// ## 为什么不能"每个位置扫 25 抽头"
///
/// 规范把 `A`/`B` 写成"对每个 (i,j) 累加 `(2r+1)²` 的邻域"。照着抄是每像素
/// 25 次取样 × 2 个累加器，而那个邻域是**可分离**的：先沿列做竖直滑动和
/// （进一行、出一行），再沿行做水平滑动和（进一列、出一列），每像素摊到常数次
/// 加减。**值完全一样**（整数加法，没有舍入参与），只是不再重复读同一批样本。
/// 这是本模块最大的单点开销（`box_filter` 一度占采样 18%）。
#[allow(clippy::too_many_arguments)]
fn box_filter(
    src: &SourceSample<'_>,
    set: usize,
    pass: usize,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    seq: &SequenceHeader,
    f: &mut [i32],
    a_buf: &mut [i32],
    b_buf: &mut [i32],
) -> Result<()> {
    let bit_depth = seq.color.bit_depth as u32;
    let r = SGR_PARAMS[set][pass * 2];
    let f_len = (w * h) as usize;
    debug_assert!(f.len() >= f_len);
    if r == 0 {
        // 规范：`r == 0` 时本过程直接返回（`F` 不参与后续求和）
        f[..f_len].fill(0);
        return Ok(());
    }
    let eps = SGR_PARAMS[set][pass * 2 + 1];
    let n = (2 * r + 1) * (2 * r + 1);
    let n2e = n * n * eps;
    let s_factor = (((1i32 << SGRPROJ_MTABLE_BITS) + n2e / 2) / n2e).max(1);
    let one_over_n = ((1i32 << SGRPROJ_RECIP_BITS) + (n / 2)) / n;

    let aw = (w + 2) as usize;
    let ab_len = aw * (h + 2) as usize;
    debug_assert!(a_buf.len() >= ab_len && b_buf.len() >= ab_len);

    // 竖直和：`cs`/`cs2` 是"当前 i 的行窗口"在每个**列**上的和。
    // 列范围是 `x + (-1-r) .. x + (w+r)`，共 `w + 2 + 2r` 列；块的宽度最多 4
    // （环路恢复按 4×4 走），所以定长数组就够了，不必再分配。
    let cols = (w + 2 + 2 * r) as usize;
    debug_assert!(cols <= SGR_BOX_COLS_MAX, "SGR box filter 的列数超出定长缓冲");
    let mut cs = [0i32; SGR_BOX_COLS_MAX];
    let mut cs2 = [0i32; SGR_BOX_COLS_MAX];
    let col0 = x - 1 - r; // 第 0 个缓冲列对应的平面 x

    // ── 源解析表（2026-10-02）：把 `src.get` 的两次钳位 + stripe 判定从**逐样本**
    // 搬到**逐行/逐列**。竖直相位一个 4×4 块要取样约 140 次，而列只有
    // `w+2+2r ≤ 10` 个、行只有 `h+2r+2 ≤ 10` 个 —— 各解析一次就够。
    // `get` 的钳位按轴独立（x、y 互不影响）⇒ 拆开与逐样本**逐位等价**。
    let stride = src.up_curr.planes[src.plane].stride;
    let mut xs = [0usize; SGR_BOX_COLS_MAX];
    for (k, slot) in xs[..cols].iter_mut().enumerate() {
        *slot = (col0 + k as i32).clamp(0, src.plane_end_x) as usize;
    }
    let row_lo = y - 1 - r;
    let nrows = (h + 2 * r + 2) as usize;
    debug_assert!(nrows <= SGR_BOX_ROWS_MAX, "SGR box filter 的行数超出定长缓冲");
    let mut rows_ptr = [std::ptr::null::<u16>(); SGR_BOX_ROWS_MAX];
    let mut rows_off = [0usize; SGR_BOX_ROWS_MAX];
    for k in 0..nrows {
        let yy = (row_lo + k as i32).clamp(0, src.plane_end_y);
        let (yy, from_curr) = if yy < src.stripe_start_y {
            ((src.stripe_start_y - 2).max(yy), true)
        } else if yy > src.stripe_end_y {
            ((src.stripe_end_y + 2).min(yy), true)
        } else {
            (yy, false)
        };
        let frame = if from_curr { src.up_curr } else { src.up_cdef };
        rows_ptr[k] = frame.planes[src.plane].px16().as_ptr();
        rows_off[k] = (yy as usize) * stride;
    }
    // `read(k_col, k_row)` ≡ `src.get(col0 + k_col, row_lo + k_row)`（已解析）
    let read = |kc: usize, kr: usize| -> i32 {
        // SAFETY: 坐标已按 `get` 的规则钳到帧内；两份源在本函数里都只读。
        unsafe { *rows_ptr[kr].add(rows_off[kr] + xs[kc]) as i32 }
    };

    // 竖直滑动和的两条路径：**列连续**（`xs[k] == col0 + k`，即窗口整段落在帧内）
    // 时走 [`super::simd::sgr_cols_update8`]（一次 8 列），否则回落逐列标量。
    // 值逐位相同（纯 i32 整数加/减与乘方），见 `simd::tests`。
    let avx2 = super::simd::avx2_available();
    let interior_cols = col0 >= 0 && col0 + cols as i32 - 1 <= src.plane_end_x;
    let simd_cols = avx2 && interior_cols;
    // 该列窗口第 0 列在行指针上的偏移（`read(0, kr)` 的地址）
    let col_off = |kr: usize| unsafe { rows_ptr[kr].add(rows_off[kr] + col0 as usize) };

    // i = -1 时先把行窗口 [y-1-r, y-1+r] 累起来
    if simd_cols {
        cs[..cols].fill(0);
        cs2[..cols].fill(0);
        for dy in 0..=2 * r {
            // SAFETY: 已判 AVX2 且列连续；`cs`/`cs2` 各 >= cols 项，行指针可读 cols 个样本。
            unsafe {
                super::simd::sgr_cols_update8(
                    col_off(dy as usize),
                    std::ptr::null(),
                    cs.as_mut_ptr(),
                    cs2.as_mut_ptr(),
                    cols,
                )
            };
        }
    } else {
        for k in 0..cols {
            let mut a = 0i32;
            let mut b = 0i32;
            for dy in 0..=2 * r {
                let c = read(k, dy as usize);
                a += c * c;
                b += c;
            }
            cs[k] = b;
            cs2[k] = a;
        }
    }

    let two_r_plus_1 = (2 * r + 1) as usize;
    for i in -1..(h + 1) {
        if i > -1 {
            // 行窗口整体下移一行：进 `y+i+r`（表行 `i+2r+1`）、出 `y+i-r-1`（表行 `i`）
            if simd_cols {
                // SAFETY: 同上；两行指针都可读 `cols` 个样本。
                unsafe {
                    super::simd::sgr_cols_update8(
                        col_off((i + 2 * r + 1) as usize),
                        col_off(i as usize),
                        cs.as_mut_ptr(),
                        cs2.as_mut_ptr(),
                        cols,
                    )
                };
            } else {
                for k in 0..cols {
                    let add = read(k, (i + 2 * r + 1) as usize);
                    let sub = read(k, i as usize);
                    cs[k] += add - sub;
                    cs2[k] += add * add - sub * sub;
                }
            }
        }
        // ── 前缀和 + 8 路向量化（2026-10-04）────────────────────────────────
        //
        // 规范把 `Σ c` / `Σ c²` 写成"以 (i,j) 为中心的窗口和"。原实现用**水平滑动**
        // 增量维护（串行依赖 —— 8 列就得串 8 步，向量化不了）。改成**前缀和相减**
        // 之后，每个输出只依赖两张已经算好的表 ⇒ 8 个相邻列完全独立，
        // [`super::simd::sgr_ab_8`] 就能一次算 8 个（`a2` 的查表仍是逐通道标量）。
        //
        // j 的窗口（缓冲下标）是 `[j+1, j+2r+1]`；记 `jj = j+1 ∈ [0, w+1]`，
        // 则窗口和 `= pre[jj + 2r + 1] - pre[jj]`。
        let mut pre_s = [0i32; SGR_BOX_COLS_MAX + 2];
        let mut pre_s2 = [0i32; SGR_BOX_COLS_MAX + 2];
        for k in 0..cols {
            pre_s[k + 1] = pre_s[k] + cs[k];
            pre_s2[k + 1] = pre_s2[k] + cs2[k];
        }
        let jj_n = (w + 2) as usize;
        let row_base = (i + 1) as usize * aw;
        let mut jj = 0usize;
        if avx2 {
            while jj + 8 <= jj_n {
                // SAFETY: 已判 AVX2。`jj + 8 <= w+2` ⇒ 前缀和最大读到
                // `pre_s[jj + 2r + 1 + 7]`，`jj ≤ w-6` ⇒ 下标 ≤ `w + 2r + 2 = cols`
                // （`pre_s` 填到 `cols`）；`a_buf`/`b_buf` 的 8 个落点也都在
                // `[row_base, row_base + aw)` 内（`jj ≤ w+1`）。
                unsafe {
                    super::simd::sgr_ab_8(
                        &pre_s,
                        &pre_s2,
                        jj,
                        jj + two_r_plus_1,
                        n,
                        s_factor,
                        one_over_n,
                        2 * (bit_depth - 8),
                        bit_depth - 8,
                        SGRPROJ_MTABLE_BITS,
                        SGRPROJ_RECIP_BITS,
                        &X_BY_XPLUS1,
                        &mut a_buf[row_base + jj..row_base + jj + 8],
                        &mut b_buf[row_base + jj..row_base + jj + 8],
                    );
                }
                jj += 8;
            }
        }
        while jj < jj_n {
            let sb = pre_s[jj + two_r_plus_1] - pre_s[jj];
            let sa = pre_s2[jj + two_r_plus_1] - pre_s2[jj];
            let a = round2(sa, 2 * (bit_depth - 8));
            let d = round2(sb, bit_depth - 8);
            let p = 0i32.max(a * n - d * d);
            let z = round2(p * s_factor, SGRPROJ_MTABLE_BITS);
            // `z` 在真实素材上会出现**负值**：`p * s_factor` 在 i32 里溢出
            // （pass 1 的小 eps 档 `s_factor` 能到几千，p 到千万量级）。
            // 改动前就是让最后那条式子兜住的，这里**保持逐位一致**，别"顺手修"。
            let a2 = if z >= 255 {
                256
            } else if z > 0 {
                X_BY_XPLUS1[z as usize]
            } else if z == 0 {
                1
            } else {
                ((z << SGRPROJ_SGR_BITS) + (z / 2)) / (z + 1)
            };
            let b2 = ((1i32 << SGRPROJ_SGR_BITS) - a2) * sb * one_over_n;
            a_buf[row_base + jj] = a2;
            b_buf[row_base + jj] = round2(b2, SGRPROJ_RECIP_BITS);
            jj += 1;
        }
    }

    // 末段（§7.17.3 的 3×3 加权）：**权重只跟 (pass, i 的奇偶) 有关** ⇒ 每个 i 先算
    // 一张 3×3 权重表，内层按表展开、不再逐样本判 `weight == 0` 的分支；
    // `a_buf` 的索引用 `(i+dyi)*aw + j + dxi`（dyi/dxi ∈ 0..3）—— 同一行的三个
    // 邻居就是连续内存；结果先写进行缓冲再整行拷进 `f`。
    // 末段（§7.17.3 的 3×3 加权）：**权重只跟 (pass, i 的奇偶) 有关** ⇒ 每个 i 先算
    // 一张 3×3 权重表，内层按表展开、不再逐样本判 `weight == 0` 的分支；
    // `a_buf` 的索引用 `(i+dyi)*aw + j + dxi`（dyi/dxi ∈ 0..3）—— 同一行的三个
    // 邻居就是连续内存；结果先写进行缓冲再整行拷进 `f`。
    //
    // 这一段的算术对 8 个相邻列**完全独立**（`A`/`B` 同行相邻、`u` 同行相邻）
    // ⇒ 8 路向量化（[`super::simd::sgr_final_8`]），不足 8 列的尾巴仍走标量。
    // 两条路**逐位一致**（`ME_NO_SIMD` 开/关逐字节验证过）。
    let cdef = &src.up_cdef.planes[plane_of(src)];
    let cdef_stride = cdef.stride;
    for i in 0..h {
        let mut shift = 5u32;
        if pass == 0 && (i & 1) != 0 {
            shift = 4;
        }
        let mut wt = [[0i32; 3]; 3];
        for (dyi, wrow) in wt.iter_mut().enumerate() {
            for (dxi, w) in wrow.iter_mut().enumerate() {
                let (dy, dx) = (dyi as i32 - 1, dxi as i32 - 1);
                *w = if pass == 0 {
                    if (i + dy) & 1 != 0 {
                        if dx == 0 { 6 } else { 5 }
                    } else {
                        0
                    }
                } else if dx == 0 || dy == 0 {
                    4
                } else {
                    3
                };
            }
        }
        let sgr_shift = SGRPROJ_SGR_BITS + shift - SGRPROJ_RST_BITS;
        let cdef_off = (y + i) as usize * cdef_stride + x as usize;
        // ⚠️ **直接写进 `f`**，不再"先攒 `row` 再整行 `copy_from_slice`"（2026-10-04）：
        // 那一趟是每行一次 `w` 个 `i32` 的 memcpy，一档 64 行、一帧上千档 ——
        // 采样里 `platform_memmove` 一直排在前十，这是其中一份。
        let frow = &mut f[i as usize * w as usize..][..w as usize];
        let mut j = 0i32;
        if avx2 {
            while j + 8 <= w {
                // SAFETY: 已判 AVX2；`j + 8 <= w` ⇒ 这一批 8 列的
                // `a_buf[base + dyi*aw + dxi .. +8]` 全部落在 `a_buf[..need_ab]` 内
                // （行内偏移最多到 `w + 1 < aw`），`u` 的 8 个 `u16` 也都在 `cdef` 行内。
                unsafe {
                    super::simd::sgr_final_8(
                        &a_buf[..ab_len],
                        &b_buf[..ab_len],
                        aw,
                        i as usize * aw + j as usize,
                        &wt,
                        cdef.px16().as_ptr().add(cdef_off + j as usize),
                        sgr_shift,
                        &mut frow[j as usize..j as usize + 8],
                    );
                }
                j += 8;
            }
        }
        while j < w {
            let mut a = 0i32;
            let mut b = 0i32;
            for dyi in 0..3usize {
                let base = (i as usize + dyi) * aw + j as usize;
                for dxi in 0..3usize {
                    let ww = wt[dyi][dxi];
                    a += ww * a_buf[base + dxi];
                    b += ww * b_buf[base + dxi];
                }
            }
            let u = i32::from(cdef.get((x + j) as usize, (y + i) as usize));
            let v = a * u + b;
            frow[j as usize] = round2(v, sgr_shift);
            j += 1;
        }
    }
    Ok(())
}

/// `box_filter` 里取 `UpscaledCdefFrame` 用（plane 由 `SourceSample` 携带）。
fn plane_of(s: &SourceSample<'_>) -> usize {
    s.plane
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_units_in_frame_rounds_to_nearest_and_never_zero() {
        // `(1080 + 64/2) / 64 = 17`
        assert_eq!(count_units_in_frame(64, 1080), 17);
        // `(1920 + 64/2) / 64 = 30`
        assert_eq!(count_units_in_frame(64, 1920), 30);
        // 帧比 unit 还小 ⇒ 至少 1
        assert_eq!(count_units_in_frame(256, 8), 1);
        // 恰好整除：`(64 + 128/2)/128 = 1`
        assert_eq!(count_units_in_frame(128, 64), 1);
        // 超过半个 unit 才进位：`(193 + 64)/128 = 2`
        assert_eq!(count_units_in_frame(128, 193), 2);
        // 卡在"刚好一半"上不进位：`(128 + 64)/128 = 1`
        assert_eq!(count_units_in_frame(128, 128), 1);
    }

    #[test]
    fn inverse_recenter_matches_the_spec_branches() {
        // v > 2r ⇒ 原样
        assert_eq!(inverse_recenter(3, 10), 10);
        // 奇数 ⇒ 往左
        assert_eq!(inverse_recenter(10, 3), 8);
        // 偶数 ⇒ 往右
        assert_eq!(inverse_recenter(10, 4), 12);
        assert_eq!(inverse_recenter(0, 0), 0);
        // `v > 2r` 直接返回 v ⇒ `r = 0` 时任何 `v >= 1` 都走这一支
        assert_eq!(inverse_recenter(0, 1), 1);
    }

    #[test]
    fn sgr_params_table_shape_matches_the_spec() {
        // 规范 §7.17.3 的 16 行：`radius` 只可能是 0 或 2，`eps` 与 `radius` 配套
        for (i, row) in SGR_PARAMS.iter().enumerate() {
            assert!(
                row[0] == 0 || row[0] == 2,
                "第 {i} 行的 radius0 = {} 不该出现",
                row[0]
            );
            assert!(
                row[2] == 0 || row[2] == 1,
                "第 {i} 行的 radius1 = {} 不该出现",
                row[2]
            );
            if row[0] == 0 {
                assert_eq!(row[1], 0, "第 {i} 行 radius0 = 0 时 eps0 必须是 0");
            }
        }
        // 15 号那一行两路都是"关掉"，是规范里唯一 `eps1 = 0` 的行
        assert_eq!(SGR_PARAMS[14][2], 0);
        assert_eq!(SGR_PARAMS[15][2], 0);
    }

    #[test]
    fn wiener_taps_bounds_are_ordered_and_chroma_first_coeff_is_zero() {
        for j in 0..3 {
            assert!(
                WIENER_TAPS_MIN[j] < WIENER_TAPS_MAX[j],
                "第 {j} 个 Wiener 系数的上下界反了"
            );
            assert!(
                WIENER_TAPS_MID[j] >= WIENER_TAPS_MIN[j]
                    && WIENER_TAPS_MID[j] <= WIENER_TAPS_MAX[j],
                "第 {j} 个 Wiener 系数的 Mid 值 {} 不在 [{}, {}] 内",
                WIENER_TAPS_MID[j],
                WIENER_TAPS_MIN[j],
                WIENER_TAPS_MAX[j]
            );
        }
        for i in 0..2 {
            assert!(SGRPROJ_XQD_MIN[i] < SGRPROJ_XQD_MAX[i]);
            assert!(SGRPROJ_XQD_MID[i] >= SGRPROJ_XQD_MIN[i]);
            assert!(SGRPROJ_XQD_MID[i] <= SGRPROJ_XQD_MAX[i]);
        }
    }
}
