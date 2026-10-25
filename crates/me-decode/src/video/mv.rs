//! AV1 **运动矢量预测**（规范 §7.9 运动场 + §7.10 MV 栈）—— 自研软解 M4b 的帧间那一半。
//!
//! 帧间块的位读有一半靠这里：`find_mv_stack` 建出的 `RefStackMv` 不直接读位，
//! 但它决定 **`NumMvFound`**（`drl_mode` 读几个符号）与 **`DrlCtxStack` / `NewMvContext` /
//! `RefMvContext` / `ZeroMvContext`**（这些符号用哪张 CDF）——
//! 栈里少一个候选、权重算错一档，位读就错。所以这一层和熵层一样，**必须逐字照抄**。
//!
//! ## 两张"表"要分清
//!
//! - **`SavedMvs` / `SavedRefFrames` / `SavedOrderHints`**：每个参考帧槽存一份
//!   "上一帧解码完之后的运动矢量场"（§7.20 参考帧更新过程里的
//!   *Motion field motion vector storage process*，只保留 `dist < 0` 的那些）。
//! - **`MotionFieldMvs`**：当前帧在**读非压缩头时**（`use_ref_frame_mvs == 1`）
//!   用线性投影算出来的"未来帧的运动矢量场"（§7.9），供 §7.10.2.6 的
//!   *temporal scan* 查表。
//!
//! ## 一个容易踩的坑
//!
//! `motion_field_estimation` 在规范的**非压缩头里**调用（`use_ref_frame_mvs == 1`
//! 时，在 `tile_info()` **之前**）—— 它用的是**参考帧**的运动矢量，不是本帧的。
//! 顺序错了不会报错，只会让 temporal scan 的候选莫名其妙。

use super::flags;
use super::block::{
    block_height, block_width, has_newmv, BLOCK_SIZES, NUM_4X4_BLOCKS_HIGH, NUM_4X4_BLOCKS_WIDE,
};
pub use super::frame_header::get_relative_dist;
use super::frame_header::{FrameHeader, RefFrameStore, NUM_REF_FRAMES};
use super::params::{AFFINE, IDENTITY, ROTZOOM, TRANSLATION, TOTAL_REFS_PER_FRAME};
use super::partition::MiGrid;
use super::sequence_header::SequenceHeader;

// ── 参考帧类型（规范 §6.10.2）────────────────────────────────────────────

/// `INTRA_FRAME`
pub const INTRA_FRAME: i8 = 0;
pub const LAST_FRAME: i8 = 1;
pub const LAST2_FRAME: i8 = 2;
pub const LAST3_FRAME: i8 = 3;
pub const GOLDEN_FRAME: i8 = 4;
pub const BWDREF_FRAME: i8 = 5;
pub const ALTREF2_FRAME: i8 = 6;
pub const ALTREF_FRAME: i8 = 7;
/// `NONE`（`RefFrame[1]` 的"没有第二个参考帧"）
pub const NONE: i8 = -1;

// ── 常量 ────────────────────────────────────────────────────────────────

/// `MV_INTRABC_CONTEXT`
pub const MV_INTRABC_CONTEXT: usize = 1;
/// `REF_CAT_LEVEL`：近邻候选的权重加成
pub const REF_CAT_LEVEL: i32 = 640;
/// `MAX_REF_MV_STACK_SIZE`
pub const MAX_REF_MV_STACK_SIZE: usize = 8;
/// `MFMV_STACK_SIZE`
pub const MFMV_STACK_SIZE: usize = 3;
/// `MAX_FRAME_DISTANCE`
pub const MAX_FRAME_DISTANCE: i32 = 31;
/// `MV_BORDER`
pub const MV_BORDER: i32 = 128;
/// `REFMVS_LIMIT`
pub const REFMVS_LIMIT: i32 = (1 << 12) - 1;
/// `INTRABC_DELAY_PIXELS`
pub const INTRABC_DELAY_PIXELS: i32 = 256;
/// `INTRABC_DELAY_SB64`：intrabc 的源必须至少落后这么多**栅格序 64×64 块**。
///
/// 规范用 `is_mv_valid`（§7.10.4 前的那段）把它写成"码流必须满足"的约束，解码器
/// 本身不检查；但 `assign_mv` 在栈里没有候选时给的那个**默认预测 MV** 要用到它。
pub const INTRABC_DELAY_SB64: i32 = 4;
/// `MAX_OFFSET_WIDTH`
pub const MAX_OFFSET_WIDTH: i32 = 8;
/// `MAX_OFFSET_HEIGHT`
pub const MAX_OFFSET_HEIGHT: i32 = 0;
/// `LEAST_SQUARES_SAMPLES_MAX`
pub const LEAST_SQUARES_SAMPLES_MAX: usize = 8;
/// `WARPEDMODEL_PREC_BITS`
pub const WARPEDMODEL_PREC_BITS: i32 = 16;
/// `GM_TRANS_ONLY_PREC_BITS`
pub const GM_TRANS_ONLY_PREC_BITS: i32 = 3;
/// `REF_SCALE_SHIFT`
pub const REF_SCALE_SHIFT: u32 = 14;
/// `MV_CLASSES`
pub const MV_CLASSES: usize = 11;
/// `CLASS0_SIZE`
pub const CLASS0_SIZE: usize = 2;

/// `Div_Mult[32]`（§7.9.3）
pub const DIV_MULT: [i32; 32] = [
    0, 16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638, 1489, 1365, 1260, 1170, 1092,
    1024, 963, 910, 862, 819, 780, 744, 712, 682, 655, 630, 606, 585, 564, 546, 528,
];

/// 一个运动矢量：`[0]` 是行（y），`[1]` 是列（x）—— **规范的顺序**，别弄反。
pub type Mv = [i32; 2];

/// `Round2( x, n )`
#[inline]
pub fn round2(x: i32, n: i32) -> i32 {
    if n == 0 {
        return x;
    }
    (x + (1 << (n - 1))) >> n
}

/// `Round2Signed( x, n )`（64 位版；`mv_projection` 的中间量会超 i32）
#[inline]
pub fn round2_signed_i64(x: i64, n: i32) -> i64 {
    if x >= 0 {
        round2_i64(x, n)
    } else {
        -round2_i64(-x, n)
    }
}

/// `Round2( x, n )`（64 位版）
#[inline]
pub fn round2_i64(x: i64, n: i32) -> i64 {
    if n == 0 {
        return x;
    }
    (x + (1 << (n - 1))) >> n
}

/// `Round2Signed( x, n )`
#[inline]
pub fn round2_signed(x: i32, n: i32) -> i32 {
    if x >= 0 {
        round2(x, n)
    } else {
        -round2(-x, n)
    }
}

/// `lower_mv_precision( mv )`（规范 §7.10.2.10）
pub fn lower_mv_precision(mv: &mut Mv, allow_high_precision_mv: bool, force_integer_mv: bool) {
    if allow_high_precision_mv {
        return;
    }
    for i in 0..2 {
        if force_integer_mv {
            let a = mv[i].abs();
            let a_int = (a + 3) >> 3;
            mv[i] = if mv[i] > 0 { a_int << 3 } else { -(a_int << 3) };
        } else if mv[i] & 1 != 0 {
            if mv[i] > 0 {
                mv[i] -= 1;
            } else {
                mv[i] += 1;
            }
        }
    }
}

// ── 每个参考帧槽的运动矢量存储 ──────────────────────────────────────────

/// `SavedRefFrames` / `SavedMvs` / `SavedOrderHints` 的合并体（一个参考帧槽一份）。
#[derive(Debug, Clone)]
pub struct SavedRefMvs {
    pub mi_rows: usize,
    pub mi_cols: usize,
    pub valid: bool,
    /// `RefFrameType[slot]` 是不是 KEY / INTRA_ONLY（`motion_field_projection` 要检查）
    pub is_intra_only: bool,
    /// `SavedRefFrames[row][col]`
    pub ref_frames: Vec<i8>,
    /// `SavedMvs[row][col]`
    pub mvs: Vec<Mv>,
    /// `SavedOrderHints[slot][ref]`
    pub order_hints: [u8; TOTAL_REFS_PER_FRAME],
}

impl Default for SavedRefMvs {
    fn default() -> Self {
        Self {
            mi_rows: 0,
            mi_cols: 0,
            valid: false,
            is_intra_only: true,
            ref_frames: Vec::new(),
            mvs: Vec::new(),
            order_hints: [0; TOTAL_REFS_PER_FRAME],
        }
    }
}

/// 8 个参考帧槽的运动矢量存储。
#[derive(Debug, Clone, Default)]
pub struct MotionFieldStore {
    pub slots: [SavedRefMvs; NUM_REF_FRAMES],
}

impl MotionFieldStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// 把一个刷新了的槽写成当前帧的运动矢量场。
    pub fn save(
        &mut self,
        slot: usize,
        mi_rows: usize,
        mi_cols: usize,
        is_intra_only: bool,
        ref_frames: Vec<i8>,
        mvs: Vec<Mv>,
        order_hints: [u8; TOTAL_REFS_PER_FRAME],
    ) {
        self.slots[slot] = SavedRefMvs {
            mi_rows,
            mi_cols,
            valid: true,
            is_intra_only,
            ref_frames,
            mvs,
            order_hints,
        };
    }
}

/// "这一格没有投影"的哨兵值（规范里 `MotionFieldMvs` 的初值）。
pub const MF_INVALID: Mv = [-1 << 15, -1 << 15];

/// 运动场每格存几个目标参考（`LAST_FRAME..=ALTREF_FRAME`，即 7 个）。
///
/// 规范写的是 `MotionFieldMvs[ref][y8][x8]`，但**访问模式**是"投影一格就把 7 个
/// 目标一次性写满"，而 `[ref]` 在前意味着那 7 次写落在**相隔 `h8·w8·8` 字节**
/// （1080p 约 259 KB）的 7 个不同平面上 —— 每次都是独立的缓存行，实测
/// `AV1_SYNTAX_ONLY` 采样里 `project_frame` 占**语法阶段的 13%**。
///
/// 改成 `[y8][x8][ref]` 之后那 7 次写是**一段连续的 56 字节**（`base` 对齐到 7 个
/// `Mv`），缓存行触碰从 7 条降到 1~2 条。读取方只有 [`MotionField::get`]，
/// 索引一次就够。内存也从 8 槽降到 7 槽（2.07 MB → 1.81 MB）。
///
/// ⚠️ `INTRA_FRAME` 那一槽**不存在**（规范从不写它，读到时返回哨兵）——
/// 老布局里它恒为 `MF_INVALID`，两种布局**可观测行为相同**。
pub const MF_SRC_SLOTS: usize = (ALTREF_FRAME - LAST_FRAME + 1) as usize;

/// `MotionFieldMvs[y8][x8][ref - LAST_FRAME]`（见 [`MF_SRC_SLOTS`] 的说明）。
#[derive(Debug, Clone, Default)]
pub struct MotionField {
    pub w8: usize,
    pub h8: usize,
    /// 展平：`((y8) * w8 + x8) * MF_SRC_SLOTS + (ref - LAST_FRAME)`
    pub mvs: Vec<Mv>,
}

impl MotionField {
    /// ⚠️ 生产路径**不再**用它：1080p 一份 ≈ 2 MB，每帧 `vec![哨兵; n]` 要走
    /// `mmap` + 写入时 512 次缺页（实测 `mf` 段 4.14 ms 里相当一部分是它）。
    /// 现在由 [`MotionField::reset`] 复用同一块缓冲。留作测试/冷路径。
    pub fn new(mi_rows: usize, mi_cols: usize) -> Self {
        let mut f = Self::default();
        f.reset(mi_rows, mi_cols);
        f
    }

    /// 复用同一块缓冲并**重置成哨兵**（尺寸变了才重新分配）。
    ///
    /// `fill` 仍要跑（没被投影到的格子必须回到哨兵，否则会读到上一帧的值），
    /// 但省掉了 `mmap` 与随之而来的缺页。
    pub fn reset(&mut self, mi_rows: usize, mi_cols: usize) {
        self.w8 = mi_cols >> 1;
        self.h8 = mi_rows >> 1;
        let n = MF_SRC_SLOTS * self.h8 * self.w8;
        if self.mvs.len() != n {
            self.mvs.clear();
            self.mvs.resize(n, MF_INVALID);
        } else {
            self.mvs.fill(MF_INVALID);
        }
    }

    #[inline]
    pub fn get(&self, r: i8, y8: usize, x8: usize) -> Mv {
        // `r` 只可能是 `LAST_FRAME..=ALTREF_FRAME`（`INTRA_FRAME` 那一槽不存在，
        // 老布局里它恒为哨兵 ⇒ 这里显式返回哨兵，行为不变）。
        if r < LAST_FRAME || r > ALTREF_FRAME || y8 >= self.h8 || x8 >= self.w8 {
            return MF_INVALID;
        }
        let slot = (r as usize) - (LAST_FRAME as usize);
        self.mvs[(y8 * self.w8 + x8) * MF_SRC_SLOTS + slot]
    }
}

/// `get_mv_projection( mv, numerator, denominator )`（规范 §7.9.3）
///
/// ⚠️ 生产路径已换成 [`project_k`]（乘子提前算好，见 [`project_frame`]）；
/// 这一对留作**参考实现**，由 `project_k_matches_the_reference_projection` 逐值对拍。
#[cfg(test)]
fn mv_projection(mv: Mv, numerator: i32, denominator: i32) -> Mv {
    let clipped_denominator = denominator.min(MAX_FRAME_DISTANCE).max(1);
    let clipped_numerator = numerator.clamp(-MAX_FRAME_DISTANCE, MAX_FRAME_DISTANCE);
    mv_projection_clipped(mv, clipped_numerator, clipped_denominator)
}

/// [`mv_projection`] 的**已裁剪**版本（参考实现，见上）。
#[cfg(test)]
fn mv_projection_clipped(mv: Mv, clipped_numerator: i32, clipped_denominator: i32) -> Mv {
    let mut out = [0i32; 2];
    for i in 0..2 {
        // **必须用 i64**：真实流里 `mv` 可以达到 ±8800（帧边界的钳位值），
        // `mv * numerator * Div_Mult` 会到 4e9 量级，i32 会溢出。
        // 规范是数学式定义，i64 之后夹到 ±(1<<14 - 1) 才是本意。
        let wide = mv[i] as i64 * clipped_numerator as i64
            * DIV_MULT[clipped_denominator as usize] as i64;
        let scaled = round2_signed_i64(wide, 14);
        out[i] = scaled.clamp(-(1 << 14) + 1, (1 << 14) - 1) as i32;
    }
    out
}

/// `mv_projection` 的**乘子已提好**版本：`round2_signed( mv · k, 14 )` 再夹到 ±(2^14-1)。
///
/// 等价于 `mv_projection_clipped( mv, num, den )`（`k = num · Div_Mult[den]`，见
/// [`project_frame`] 里的推导）。**必须用 i64**：真实流里 `mv` 可达 ±8800，
/// 乘出来会到 4e9 量级，i32 会溢出（规范是数学式定义，i64 之后夹到 ±(2^14-1) 才是本意）。
#[inline]
fn project_k(mv: Mv, k: i64) -> Mv {
    let mut out = [0i32; 2];
    for i in 0..2 {
        let wide = mv[i] as i64 * k;
        let scaled = round2_signed_i64(wide, 14);
        out[i] = scaled.clamp(-(1 << 14) + 1, (1 << 14) - 1) as i32;
    }
    out
}

/// `get_block_position( x8, y8, dstSign, projMv )`（规范 §7.9.4）
fn block_position(x8: i32, y8: i32, dst_sign: i32, proj_mv: Mv, mi_rows: i32, mi_cols: i32) -> Option<(i32, i32)> {
    fn project(v8: i32, delta: i32, dst_sign: i32, max8: i32, max_off8: i32) -> (i32, bool) {
        let base8 = (v8 >> 3) << 3;
        let offset8 = if delta >= 0 {
            delta >> (3 + 1 + 2)
        } else {
            -((-delta) >> (3 + 1 + 2))
        };
        let v8 = v8 + dst_sign * offset8;
        let valid = !(v8 < 0
            || v8 >= max8
            || v8 < base8 - max_off8
            || v8 >= base8 + 8 + max_off8);
        (v8, valid)
    }
    let (pos_y8, ok_y) = project(y8, proj_mv[0], dst_sign, mi_rows >> 1, MAX_OFFSET_HEIGHT);
    let (pos_x8, ok_x) = project(x8, proj_mv[1], dst_sign, mi_cols >> 1, MAX_OFFSET_WIDTH);
    if ok_y && ok_x {
        Some((pos_y8, pos_x8))
    } else {
        None
    }
}

/// `motion_field_estimation( )`（规范 §7.9）—— 在非压缩头里调用。
pub fn motion_field_estimation(
    field: &mut MotionField,
    fh: &FrameHeader,
    seq: &SequenceHeader,
    store: &MotionFieldStore,
) {
    field.reset(fh.mi_rows as usize, fh.mi_cols as usize);
    let w8 = field.w8;
    let h8 = field.h8;

    // `OrderHints[ref]`（本帧的）
    let order_hints = frame_order_hints(fh, seq);

    let last_idx = fh.ref_frame_idx[0];
    let cur_gold_order_hint = order_hints[GOLDEN_FRAME as usize];
    let last_alt_order_hint = if (last_idx as usize) < NUM_REF_FRAMES {
        store.slots[last_idx as usize].order_hints[ALTREF_FRAME as usize]
    } else {
        0
    };
    let use_last = last_alt_order_hint != cur_gold_order_hint;

    if use_last {
        project_frame(field, store, seq, order_hints, fh, LAST_FRAME, -1, w8, h8);
    }
    let mut ref_stamp = (MFMV_STACK_SIZE - 2) as i32;

    let use_bwd = get_relative_dist(seq, order_hints[BWDREF_FRAME as usize], fh.order_hint) > 0;
    if use_bwd
        && project_frame(field, store, seq, order_hints, fh, BWDREF_FRAME, 1, w8, h8)
    {
        ref_stamp -= 1;
    }
    let use_alt2 = get_relative_dist(seq, order_hints[ALTREF2_FRAME as usize], fh.order_hint) > 0;
    if use_alt2
        && project_frame(field, store, seq, order_hints, fh, ALTREF2_FRAME, 1, w8, h8)
    {
        ref_stamp -= 1;
    }
    let use_alt = get_relative_dist(seq, order_hints[ALTREF_FRAME as usize], fh.order_hint) > 0;
    if use_alt && ref_stamp >= 0 {
        if project_frame(field, store, seq, order_hints, fh, ALTREF_FRAME, 1, w8, h8) {
            ref_stamp -= 1;
        }
    }
    if ref_stamp >= 0 {
        project_frame(field, store, seq, order_hints, fh, LAST2_FRAME, -1, w8, h8);
    }
}

/// 本帧的 `OrderHints[ref]`（规范在非压缩头里算好，存在 `FrameHeader` 上）。
fn frame_order_hints(fh: &FrameHeader, seq: &SequenceHeader) -> [u8; TOTAL_REFS_PER_FRAME] {
    let _ = seq;
    fh.order_hints
}

/// `motion_field_projection( src, dstSign )`（规范 §7.9.2）——返回"这个源帧有效吗"。
#[allow(clippy::too_many_arguments)]
fn project_frame(
    field: &mut MotionField,
    store: &MotionFieldStore,
    seq: &SequenceHeader,
    order_hints: [u8; TOTAL_REFS_PER_FRAME],
    fh: &FrameHeader,
    src: i8,
    dst_sign: i32,
    w8: usize,
    h8: usize,
) -> bool {
    let src_idx = fh.ref_frame_idx[(src - LAST_FRAME) as usize] as usize;
    if src_idx >= NUM_REF_FRAMES {
        return false;
    }
    let slot = &store.slots[src_idx];
    if !slot.valid || slot.mi_rows != fh.mi_rows as usize || slot.mi_cols != fh.mi_cols as usize {
        return false;
    }
    if slot.is_intra_only {
        return false;
    }
    // 提到逐格循环外（1080p：一格 7 个目标帧 × 3.2 万格 × 5 个源帧都要重算一遍）：
    // `ref_to_cur` 只跟源帧有关；`ref_to_dst` / 裁剪后的分子只跟 (src, dst) 有关。
    let ref_to_cur = get_relative_dist(seq, order_hints[src as usize], fh.order_hint);
    let mut num_clip = [0i32; TOTAL_REFS_PER_FRAME];
    for dst in LAST_FRAME..=ALTREF_FRAME {
        let d = get_relative_dist(seq, fh.order_hint, order_hints[dst as usize]);
        num_clip[dst as usize] = d.clamp(-MAX_FRAME_DISTANCE, MAX_FRAME_DISTANCE);
    }
    // 再往下提一层（2026-10-04）：`ref_offset` 与 `pos_valid` **只跟 `src_ref` 有关**，
    // 而格子里的 `src_ref` 只有 8 种取值 —— 原来每个格子都要重算一遍
    // `get_relative_dist`（几个分支 + 移位），现在一小张表查完。
    let mut ref_offset_tab = [0i32; TOTAL_REFS_PER_FRAME];
    let mut pos_valid_tab = [false; TOTAL_REFS_PER_FRAME];
    for sr in (INTRA_FRAME + 1)..=ALTREF_FRAME {
        let ro = get_relative_dist(
            seq,
            order_hints[src as usize],
            slot.order_hints[sr as usize],
        );
        ref_offset_tab[sr as usize] = ro;
        pos_valid_tab[sr as usize] = ref_to_cur.abs() <= MAX_FRAME_DISTANCE
            && ro.abs() <= MAX_FRAME_DISTANCE
            && ro > 0;
    }
    // ── 再提一层（2026-10-05）：把**乘子**也提前算好 ──────────────────────────
    //
    // `mv_projection_clipped( mv, num, den )` 展开是
    //   `round2_signed( mv[i] * num * Div_Mult[den], 14 )`
    // —— 每格每目标帧**两次 i64 乘**（1080p 实测 `mf` 段 4.14 ms/帧，是 `setup` 的八成）。
    // 而 `num` 只跟 (src, dst) 有关、`den` 只跟 `src` 有关 ⇒ `k = num · Div_Mult[den]`
    // 是一张 8×8 的小表，**每帧每源帧算一次**。整数乘可结合 ⇒ `mv * k` 与
    // `mv * num * Div_Mult[den]` **逐位相同**（值域远在 i64 内）。
    let mut k_tab = [[0i64; TOTAL_REFS_PER_FRAME]; TOTAL_REFS_PER_FRAME];
    let num_cur = (ref_to_cur * dst_sign).clamp(-MAX_FRAME_DISTANCE, MAX_FRAME_DISTANCE);
    let mut k_cur = [0i64; TOTAL_REFS_PER_FRAME];
    for sr in (INTRA_FRAME + 1)..=ALTREF_FRAME {
        if !pos_valid_tab[sr as usize] {
            continue;
        }
        let dm = DIV_MULT[ref_offset_tab[sr as usize].max(1) as usize] as i64;
        k_cur[sr as usize] = num_cur as i64 * dm;
        for dst in LAST_FRAME..=ALTREF_FRAME {
            k_tab[sr as usize][dst as usize] = num_clip[dst as usize] as i64 * dm;
        }
    }
    // `field.set` 的边界检查与目标帧平面步长也提到循环外：原来一格要判 7 次
    // （每目标帧一次），且**越界时 7 次投影全是白算** —— 现在整格先判一次，
    // 越界直接跳过（结果与"算了再丢弃"逐位相同）。
    //
    // 2026-10-05：布局改成 `[y8][x8][ref]` ⇒ 一格的 7 个目标在**一段连续的 56 字节**
    // 里（`base` 是 7 的倍数、每格间隔 7 个 `Mv`，互不重叠），写出去的缓存行从 7 条
    // 降到 1~2 条。索引推导见 [`MF_SRC_SLOTS`]。
    for y8 in 0..h8 {
        for x8 in 0..w8 {
            let row = 2 * y8 + 1;
            let col = 2 * x8 + 1;
            let src_ref = slot.ref_frames[row * slot.mi_cols + col];
            if src_ref <= INTRA_FRAME {
                continue;
            }
            let sr = src_ref as usize;
            if !pos_valid_tab[sr] {
                continue;
            }
            let mv = slot.mvs[row * slot.mi_cols + col];
            // **零 MV 快路径**：`project_k(0, k) = 0` 对任何 `k` 都成立 ⇒ 7 个
            // 目标帧的投影全是 `(0,0)`、落点也全一样。静止内容里这种格子很多，
            // 省掉 16 次 i64 乘法与 8 次取整/钳位。
            if mv[0] == 0 && mv[1] == 0 && !flags::no_mv0_fast() {
                let Some((py, px)) = block_position(
                    x8 as i32,
                    y8 as i32,
                    dst_sign,
                    [0, 0],
                    fh.mi_rows as i32,
                    fh.mi_cols as i32,
                ) else {
                    continue;
                };
                let (py, px) = (py as usize, px as usize);
                if py >= h8 || px >= w8 {
                    continue;
                }
                let base = (py * w8 + px) * MF_SRC_SLOTS;
                field.mvs[base..base + MF_SRC_SLOTS].fill([0, 0]);
                continue;
            }
            let proj_mv = project_k(mv, k_cur[sr]);
            let Some((pos_y8, pos_x8)) = block_position(
                x8 as i32,
                y8 as i32,
                dst_sign,
                proj_mv,
                fh.mi_rows as i32,
                fh.mi_cols as i32,
            ) else {
                continue;
            };
            let (py, px) = (pos_y8 as usize, pos_x8 as usize);
            if py >= h8 || px >= w8 {
                continue;
            }
            // `pos_valid` 已保证 `|ref_offset| ≤ MAX_FRAME_DISTANCE` ⇒ 这等价于
            // 原来的 `.min(MAX).max(1)`。
            let base = (py * w8 + px) * MF_SRC_SLOTS;
            let krow = &k_tab[sr];
            // 连续 7 个 `Mv`（`dst - LAST_FRAME` 正好是 0..=6）
            let out = &mut field.mvs[base..base + MF_SRC_SLOTS];
            for (slot, o) in out.iter_mut().enumerate() {
                *o = project_k(mv, krow[slot + LAST_FRAME as usize]);
            }
        }
    }
    true
}

/// *Motion field motion vector storage process*（规范 §7.20）：
/// 把本帧的运动矢量过滤/重排之后，作为参考帧的 `SavedMvs` / `SavedRefFrames` 存起来。
pub fn motion_field_storage(
    grid: &super::block::BlockGrid,
    fh: &FrameHeader,
    seq: &SequenceHeader,
    store: &RefFrameStore,
) -> (Vec<i8>, Vec<Mv>) {
    let rows = fh.mi_rows as usize;
    let cols = fh.mi_cols as usize;
    motion_field_storage_into(
        grid,
        fh,
        seq,
        store,
        vec![NONE; rows * cols],
        vec![[0i32; 2]; rows * cols],
    )
}

/// 同 [`motion_field_storage`]，但**由调用方给缓冲**（跨帧复用容量：1080p 一帧
/// 1.5 MB 的 `vec![]` 是"新页 + 首次写入缺页"，而槽里的旧内容在本帧解码结束后
/// 已经死了 ⇒ 直接拿它当落点）。
pub fn motion_field_storage_into(
    grid: &super::block::BlockGrid,
    fh: &FrameHeader,
    seq: &SequenceHeader,
    store: &RefFrameStore,
    mut ref_frames: Vec<i8>,
    mut mvs: Vec<Mv>,
) -> (Vec<i8>, Vec<Mv>) {
    let rows = fh.mi_rows as usize;
    let cols = fh.mi_cols as usize;
    ref_frames.clear();
    ref_frames.resize(rows * cols, NONE);
    mvs.clear();
    mvs.resize(rows * cols, [0, 0]);
    let hints = fh.order_hints;
    // "这个参考帧是否在过去"只跟**参考帧类型**（LAST..ALTREF 七种）有关 ——
    // 原来每格每列表都重算一次 `get_relative_dist`（1080p 一帧 6.5 万次）。
    // 先做成 8 项表，格子里只剩一次查表。（`r` 即参考帧类型，`usable` 直接按它索引。）
    let mut usable = [false; NUM_REF_FRAMES];
    for rf in LAST_FRAME..=ALTREF_FRAME {
        let ref_idx =
            (fh.ref_frame_idx[(rf - LAST_FRAME) as usize] as usize).min(NUM_REF_FRAMES - 1);
        usable[rf as usize] =
            get_relative_dist(seq, store.order_hint[ref_idx], fh.order_hint) < 0;
    }
    // 按**行切片**走（2026-10-24）：原来每格每列表都要 `row*cols+col` 的乘法 +
    // 4 次 `Vec` 越界判断（1080p 一帧 13 万格 × 2 列表）。
    let names = &grid.ref_frames;
    let all_mvs = &grid.mvs;
    for row in 0..rows {
        let base = row * cols;
        let rf = &names[base..base + cols];
        let mvs_in = &all_mvs[base..base + cols];
        let out_rf = &mut ref_frames[base..base + cols];
        let out_mv = &mut mvs[base..base + cols];
        for col in 0..cols {
            for list in 0..2 {
                let r = rf[col][list];
                let ok = if flags::no_mf_table() {
                    if r > INTRA_FRAME {
                        let ref_idx = (fh.ref_frame_idx[(r - LAST_FRAME) as usize] as usize)
                            .min(NUM_REF_FRAMES - 1);
                        get_relative_dist(seq, store.order_hint[ref_idx], fh.order_hint) < 0
                    } else {
                        false
                    }
                } else {
                    r > INTRA_FRAME && usable[r as usize]
                };
                if ok {
                    let mv = mvs_in[col][list];
                    if mv[0].abs() <= REFMVS_LIMIT && mv[1].abs() <= REFMVS_LIMIT {
                        out_rf[col] = r;
                        out_mv[col] = mv;
                    }
                }
            }
        }
    }
    let _ = hints;
    (ref_frames, mvs)
}

// ── MV 栈（规范 §7.10.2）────────────────────────────────────────────────

/// `find_mv_stack` 的产物。
#[derive(Debug, Clone)]
pub struct RefMvStack {
    /// `RefStackMv[idx][list]`
    pub mv: [[Mv; 2]; MAX_REF_MV_STACK_SIZE],
    pub weight: [i32; MAX_REF_MV_STACK_SIZE],
    pub num: usize,
    /// `DrlCtxStack[idx]`
    pub drl_ctx: [usize; MAX_REF_MV_STACK_SIZE],
    pub ref_mv_context: usize,
    pub new_mv_context: usize,
    pub zero_mv_context: usize,
    pub global_mvs: [Mv; 2],
}

/// `setup_global_mv( refList )`（规范 §7.10.2.1）
pub fn setup_global_mv(
    fh: &FrameHeader,
    ref_frame: i8,
    ref_list: usize,
    mi_row: i32,
    mi_col: i32,
    mi_size: u8,
) -> Mv {
    let ref_frame = if ref_list == 0 {
        ref_frame
    } else {
        ref_frame
    };
    let mut mv = [0i32; 2];
    if ref_frame == INTRA_FRAME {
        return mv;
    }
    let typ = fh.params.global_motion.gm_type[ref_frame as usize];
    let bw = block_width(mi_size);
    let bh = block_height(mi_size);
    if typ == IDENTITY {
        mv = [0, 0];
    } else if typ == TRANSLATION {
        let p = &fh.params.global_motion.params[ref_frame as usize];
        mv[0] = p[0] >> (WARPEDMODEL_PREC_BITS - 3);
        mv[1] = p[1] >> (WARPEDMODEL_PREC_BITS - 3);
    } else {
        let p = &fh.params.global_motion.params[ref_frame as usize];
        let x = mi_col * 4 + bw / 2 - 1;
        let y = mi_row * 4 + bh / 2 - 1;
        let xc = (p[2] - (1 << WARPEDMODEL_PREC_BITS)) * x + p[3] * y + p[0];
        let yc = p[4] * x + (p[5] - (1 << WARPEDMODEL_PREC_BITS)) * y + p[1];
        if fh.allow_high_precision_mv {
            mv[0] = round2_signed(yc, WARPEDMODEL_PREC_BITS - 3);
            mv[1] = round2_signed(xc, WARPEDMODEL_PREC_BITS - 3);
        } else {
            mv[0] = round2_signed(yc, WARPEDMODEL_PREC_BITS - 2) * 2;
            mv[1] = round2_signed(xc, WARPEDMODEL_PREC_BITS - 2) * 2;
        }
    }
    lower_mv_precision(&mut mv, fh.allow_high_precision_mv, fh.force_integer_mv);
    mv
}

/// `find_mv_stack( isCompound )` 的全部输入（打包成一个结构，免得二十个参数）。
pub struct MvStackInput<'a> {
    pub grid: &'a super::block::BlockGrid,
    pub mi_grid: &'a MiGrid,
    pub fh: &'a FrameHeader,
    pub seq: &'a SequenceHeader,
    pub field: Option<&'a MotionField>,
    pub mi_row: i32,
    pub mi_col: i32,
    pub mi_size: u8,
    pub ref_frame: [i8; 2],
    pub mi_row_start: i32,
    pub mi_row_end: i32,
    pub mi_col_start: i32,
    pub mi_col_end: i32,
}

struct StackState {
    mv: [[Mv; 2]; MAX_REF_MV_STACK_SIZE],
    weight: [i32; MAX_REF_MV_STACK_SIZE],
    num: usize,
    new_mv_count: i32,
    found_match: bool,
    zero_mv_context: i32,
    global_mvs: [Mv; 2],
    ref_id_mvs: [[Mv; 2]; 2],
    ref_id_count: [usize; 2],
    ref_diff_mvs: [[Mv; 2]; 2],
    ref_diff_count: [usize; 2],
}

impl StackState {
    fn new() -> Self {
        Self {
            mv: [[[0, 0]; 2]; MAX_REF_MV_STACK_SIZE],
            weight: [0; MAX_REF_MV_STACK_SIZE],
            num: 0,
            new_mv_count: 0,
            found_match: false,
            zero_mv_context: 0,
            global_mvs: [[0, 0]; 2],
            ref_id_mvs: [[[0, 0]; 2]; 2],
            ref_id_count: [0; 2],
            ref_diff_mvs: [[[0, 0]; 2]; 2],
            ref_diff_count: [0; 2],
        }
    }

    /// 栈里找同一个候选：找到就加权，否则压栈（复合时要求两个分量都对上）。
    fn push_or_weight(&mut self, cand: Mv, weight: i32, compound_pair: Option<Mv>) {
        for idx in 0..self.num {
            let single_match = self.mv[idx][0] == cand
                && match compound_pair {
                    Some(p) => self.mv[idx][1] == p,
                    None => true,
                };
            if single_match {
                self.weight[idx] += weight;
                return;
            }
        }
        if self.num < MAX_REF_MV_STACK_SIZE {
            self.mv[self.num][0] = cand;
            if let Some(p) = compound_pair {
                self.mv[self.num][1] = p;
            }
            self.weight[self.num] = weight;
            self.num += 1;
        }
    }
}

impl MvStackInput<'_> {
    #[inline]
    fn is_inside(&self, r: i32, c: i32) -> bool {
        r >= self.mi_row_start && r < self.mi_row_end && c >= self.mi_col_start && c < self.mi_col_end
    }

    #[inline]
    fn cand_size(&self, r: i32, c: i32) -> u8 {
        self.mi_grid.get(r, c)
    }

    #[inline]
    fn is_inter_written(&self, r: i32, c: i32) -> bool {
        self.grid.is_inter_written(r, c)
    }

    /// *Add reference motion vector process*（§7.10.2.7）
    fn add_ref_mv_candidate(&self, st: &mut StackState, mv_row: i32, mv_col: i32, is_compound: bool, weight: i32) {
        if !self.is_inter_written(mv_row, mv_col) {
            return;
        }
        let i = self.grid.idx(mv_row, mv_col);
        if self.grid.is_inters[i] == 0 {
            return;
        }
        if !is_compound {
            for cand_list in 0..2 {
                if self.grid.ref_frames[i][cand_list] == self.ref_frame[0] {
                    self.search_stack(st, mv_row, mv_col, weight, cand_list);
                }
            }
        } else if self.grid.ref_frames[i][0] == self.ref_frame[0]
            && self.grid.ref_frames[i][1] == self.ref_frame[1]
        {
            self.compound_search_stack(st, mv_row, mv_col, weight);
        }
    }

    /// *Search stack process*（§7.10.2.8）
    fn search_stack(&self, st: &mut StackState, mv_row: i32, mv_col: i32, weight: i32, cand_list: usize) {
        let i = self.grid.idx(mv_row, mv_col);
        let cand_mode = self.grid.y_modes[i];
        let cand_size = self.cand_size(mv_row, mv_col);
        let large = block_width(cand_size).min(block_height(cand_size)) >= 8;
        let gm_type = self.fh.params.global_motion.gm_type[self.ref_frame[0] as usize];
        let mut cand_mv = if (cand_mode == GLOBALMV || cand_mode == GLOBAL_GLOBALMV)
            && gm_type > TRANSLATION
            && large
        {
            st.global_mvs[0]
        } else {
            self.grid.mvs[i][cand_list]
        };
        lower_mv_precision(&mut cand_mv, self.fh.allow_high_precision_mv, self.fh.force_integer_mv);
        if has_newmv(cand_mode) {
            st.new_mv_count += 1;
        }
        st.found_match = true;
        st.push_or_weight(cand_mv, weight, None);
    }

    /// *Compound search stack process*（§7.10.2.9）
    fn compound_search_stack(&self, st: &mut StackState, mv_row: i32, mv_col: i32, weight: i32) {
        let i = self.grid.idx(mv_row, mv_col);
        let mut cand_mvs = self.grid.mvs[i];
        let cand_mode = self.grid.y_modes[i];
        if cand_mode == GLOBAL_GLOBALMV {
            for ref_list in 0..2 {
                let gm = self.fh.params.global_motion.gm_type[self.ref_frame[ref_list] as usize];
                if gm > TRANSLATION {
                    cand_mvs[ref_list] = st.global_mvs[ref_list];
                }
            }
        }
        for m in cand_mvs.iter_mut() {
            lower_mv_precision(m, self.fh.allow_high_precision_mv, self.fh.force_integer_mv);
        }
        st.found_match = true;
        st.push_or_weight(cand_mvs[0], weight, Some(cand_mvs[1]));
        if has_newmv(cand_mode) {
            st.new_mv_count += 1;
        }
    }

    /// *Scan row process*（§7.10.2.2）
    fn scan_row(&self, st: &mut StackState, delta_row: i32, is_compound: bool) {
        let bw4 = NUM_4X4_BLOCKS_WIDE[self.mi_size as usize] as i32;
        let end4 = bw4.min(self.fh.mi_cols as i32 - self.mi_col).min(16);
        let mut delta_col = 0i32;
        let use_step16 = bw4 >= 16;
        let mut delta_row = delta_row;
        if delta_row.abs() > 1 {
            delta_row += self.mi_row & 1;
            delta_col = 1 - (self.mi_col & 1);
        }
        let mut i = 0i32;
        while i < end4 {
            let mv_row = self.mi_row + delta_row;
            let mv_col = self.mi_col + delta_col + i;
            if !self.is_inside(mv_row, mv_col) {
                break;
            }
            let sz = self.cand_size(mv_row, mv_col);
            let mut len = bw4.min(NUM_4X4_BLOCKS_WIDE[sz as usize] as i32);
            if delta_row.abs() > 1 {
                len = len.max(2);
            }
            if use_step16 {
                len = len.max(4);
            }
            self.add_ref_mv_candidate(st, mv_row, mv_col, is_compound, len * 2);
            i += len;
        }
    }

    /// *Scan col process*（§7.10.2.3）
    fn scan_col(&self, st: &mut StackState, delta_col: i32, is_compound: bool) {
        let bh4 = NUM_4X4_BLOCKS_HIGH[self.mi_size as usize] as i32;
        let end4 = bh4.min(self.fh.mi_rows as i32 - self.mi_row).min(16);
        let mut delta_row = 0i32;
        let use_step16 = bh4 >= 16;
        let mut delta_col = delta_col;
        if delta_col.abs() > 1 {
            delta_row = 1 - (self.mi_row & 1);
            delta_col += self.mi_col & 1;
        }
        let mut i = 0i32;
        while i < end4 {
            let mv_row = self.mi_row + delta_row + i;
            let mv_col = self.mi_col + delta_col;
            if !self.is_inside(mv_row, mv_col) {
                break;
            }
            let sz = self.cand_size(mv_row, mv_col);
            let mut len = bh4.min(NUM_4X4_BLOCKS_HIGH[sz as usize] as i32);
            if delta_col.abs() > 1 {
                len = len.max(2);
            }
            if use_step16 {
                len = len.max(4);
            }
            self.add_ref_mv_candidate(st, mv_row, mv_col, is_compound, len * 2);
            i += len;
        }
    }

    /// *Scan point process*（§7.10.2.4）
    fn scan_point(&self, st: &mut StackState, delta_row: i32, delta_col: i32, is_compound: bool) {
        let mv_row = self.mi_row + delta_row;
        let mv_col = self.mi_col + delta_col;
        if self.is_inside(mv_row, mv_col) && self.is_inter_written(mv_row, mv_col) {
            self.add_ref_mv_candidate(st, mv_row, mv_col, is_compound, 4);
        }
    }

    /// *Temporal scan process*（§7.10.2.5）
    fn temporal_scan(&self, st: &mut StackState, is_compound: bool) {
        let bw4 = NUM_4X4_BLOCKS_WIDE[self.mi_size as usize] as i32;
        let bh4 = NUM_4X4_BLOCKS_HIGH[self.mi_size as usize] as i32;
        let step_w4 = if bw4 >= 16 { 4 } else { 2 };
        let step_h4 = if bh4 >= 16 { 4 } else { 2 };
        let mut delta_row = 0;
        while delta_row < bh4.min(16) {
            let mut delta_col = 0;
            while delta_col < bw4.min(16) {
                self.add_tpl_ref_mv(st, delta_row, delta_col, is_compound);
                delta_col += step_w4;
            }
            delta_row += step_h4;
        }
        let allow_extension = bh4 >= NUM_4X4_BLOCKS_HIGH[super::partition::BLOCK_8X8] as i32
            && bh4 < NUM_4X4_BLOCKS_HIGH[super::partition::BLOCK_64X64] as i32
            && bw4 >= NUM_4X4_BLOCKS_WIDE[super::partition::BLOCK_8X8] as i32
            && bw4 < NUM_4X4_BLOCKS_WIDE[super::partition::BLOCK_64X64] as i32;
        if allow_extension {
            let tpl = [[bh4, -2], [bh4, bw4], [bh4 - 2, bw4]];
            for [dr, dc] in tpl {
                let row = (self.mi_row & 15) + dr;
                let col = (self.mi_col & 15) + dc;
                if row >= 0 && row < 16 && col >= 0 && col < 16 {
                    self.add_tpl_ref_mv(st, dr, dc, is_compound);
                }
            }
        }
    }

    /// *Temporal sample process*（§7.10.2.6）
    fn add_tpl_ref_mv(&self, st: &mut StackState, delta_row: i32, delta_col: i32, is_compound: bool) {
        let Some(field) = self.field else { return };
        let mv_row = (self.mi_row + delta_row) | 1;
        let mv_col = (self.mi_col + delta_col) | 1;
        if !self.is_inside(mv_row, mv_col) {
            return;
        }
        let x8 = (mv_col >> 1) as usize;
        let y8 = (mv_row >> 1) as usize;
        if delta_row == 0 && delta_col == 0 {
            st.zero_mv_context = 1;
        }
        if !is_compound {
            let mut cand_mv = field.get(self.ref_frame[0], y8, x8);
            if cand_mv[0] == -1 << 15 {
                return;
            }
            lower_mv_precision(&mut cand_mv, self.fh.allow_high_precision_mv, self.fh.force_integer_mv);
            if delta_row == 0 && delta_col == 0 {
                if (cand_mv[0] - st.global_mvs[0][0]).abs() >= 16
                    || (cand_mv[1] - st.global_mvs[0][1]).abs() >= 16
                {
                    st.zero_mv_context = 1;
                } else {
                    st.zero_mv_context = 0;
                }
            }
            st.push_or_weight(cand_mv, 2, None);
        } else {
            let mut cand0 = field.get(self.ref_frame[0], y8, x8);
            if cand0[0] == -1 << 15 {
                return;
            }
            let mut cand1 = field.get(self.ref_frame[1], y8, x8);
            if cand1[0] == -1 << 15 {
                return;
            }
            lower_mv_precision(&mut cand0, self.fh.allow_high_precision_mv, self.fh.force_integer_mv);
            lower_mv_precision(&mut cand1, self.fh.allow_high_precision_mv, self.fh.force_integer_mv);
            if delta_row == 0 && delta_col == 0 {
                if (cand0[0] - st.global_mvs[0][0]).abs() >= 16
                    || (cand0[1] - st.global_mvs[0][1]).abs() >= 16
                    || (cand1[0] - st.global_mvs[1][0]).abs() >= 16
                    || (cand1[1] - st.global_mvs[1][1]).abs() >= 16
                {
                    st.zero_mv_context = 1;
                } else {
                    st.zero_mv_context = 0;
                }
            }
            st.push_or_weight(cand0, 2, Some(cand1));
        }
    }

    /// *Add extra MV candidate process*（§7.10.2.13）
    fn add_extra_mv_candidate(&self, st: &mut StackState, mv_row: i32, mv_col: i32, is_compound: bool) {
        let i = self.grid.idx(mv_row, mv_col);
        if is_compound {
            for cand_list in 0..2 {
                let cand_ref = self.grid.ref_frames[i][cand_list];
                if cand_ref > INTRA_FRAME {
                    for list in 0..2 {
                        let mut cand_mv = self.grid.mvs[i][cand_list];
                        if cand_ref == self.ref_frame[list] && st.ref_id_count[list] < 2 {
                            st.ref_id_mvs[list][st.ref_id_count[list]] = cand_mv;
                            st.ref_id_count[list] += 1;
                        } else if st.ref_diff_count[list] < 2 {
                            if self.fh.ref_frame_sign_bias[cand_ref as usize]
                                != self.fh.ref_frame_sign_bias[self.ref_frame[list] as usize]
                            {
                                cand_mv[0] *= -1;
                                cand_mv[1] *= -1;
                            }
                            st.ref_diff_mvs[list][st.ref_diff_count[list]] = cand_mv;
                            st.ref_diff_count[list] += 1;
                        }
                    }
                }
            }
        } else {
            for cand_list in 0..2 {
                let cand_ref = self.grid.ref_frames[i][cand_list];
                if cand_ref > INTRA_FRAME {
                    let mut cand_mv = self.grid.mvs[i][cand_list];
                    if self.fh.ref_frame_sign_bias[cand_ref as usize]
                        != self.fh.ref_frame_sign_bias[self.ref_frame[0] as usize]
                    {
                        cand_mv[0] *= -1;
                        cand_mv[1] *= -1;
                    }
                    let mut found = false;
                    for idx in 0..st.num {
                        if st.mv[idx][0] == cand_mv {
                            found = true;
                            break;
                        }
                    }
                    if !found && st.num < MAX_REF_MV_STACK_SIZE {
                        st.mv[st.num][0] = cand_mv;
                        st.weight[st.num] = 2;
                        st.num += 1;
                    }
                }
            }
        }
    }

    /// *Sorting process*（§7.10.2.11）：权重大的排前面（**稳定**）。
    fn sort_stack(&self, st: &mut StackState, start: usize, end: usize) {
        let mut end = end;
        while end > start {
            let mut new_end = start;
            for idx in (start + 1)..end {
                if st.weight[idx - 1] < st.weight[idx] {
                    st.weight.swap(idx - 1, idx);
                    st.mv.swap(idx - 1, idx);
                    new_end = idx;
                }
            }
            end = new_end;
        }
    }

    /// *Extra search process*（§7.10.2.12）
    fn extra_search(&self, st: &mut StackState, is_compound: bool) {
        st.ref_id_count = [0; 2];
        st.ref_diff_count = [0; 2];
        let mut w4 = NUM_4X4_BLOCKS_WIDE[self.mi_size as usize] as i32;
        let mut h4 = NUM_4X4_BLOCKS_HIGH[self.mi_size as usize] as i32;
        w4 = w4.min(16).min(self.fh.mi_cols as i32 - self.mi_col);
        h4 = h4.min(16).min(self.fh.mi_rows as i32 - self.mi_row);
        let num4x4 = w4.min(h4);
        for pass in 0..2 {
            let mut idx = 0;
            while idx < num4x4 && st.num < 2 {
                let (mv_row, mv_col) = if pass == 0 {
                    (self.mi_row - 1, self.mi_col + idx)
                } else {
                    (self.mi_row + idx, self.mi_col - 1)
                };
                if !self.is_inside(mv_row, mv_col) {
                    break;
                }
                self.add_extra_mv_candidate(st, mv_row, mv_col, is_compound);
                let sz = self.cand_size(mv_row, mv_col);
                idx += if pass == 0 {
                    NUM_4X4_BLOCKS_WIDE[sz as usize] as i32
                } else {
                    NUM_4X4_BLOCKS_HIGH[sz as usize] as i32
                };
            }
        }
        if is_compound {
            let mut combined = [[[0i32; 2]; 2]; 2];
            for list in 0..2 {
                let mut comp_count = 0;
                for idx in 0..st.ref_id_count[list] {
                    combined[comp_count][list] = st.ref_id_mvs[list][idx];
                    comp_count += 1;
                }
                let mut idx = 0;
                while idx < st.ref_diff_count[list] && comp_count < 2 {
                    combined[comp_count][list] = st.ref_diff_mvs[list][idx];
                    comp_count += 1;
                    idx += 1;
                }
                while comp_count < 2 {
                    combined[comp_count][list] = st.global_mvs[list];
                    comp_count += 1;
                }
            }
            if st.num == 1 {
                let next = if combined[0][0] == st.mv[0][0] && combined[0][1] == st.mv[0][1] {
                    combined[1]
                } else {
                    combined[0]
                };
                if st.num < MAX_REF_MV_STACK_SIZE {
                    st.mv[st.num] = next;
                    st.weight[st.num] = 2;
                    st.num += 1;
                }
            } else {
                for idx in 0..2 {
                    if st.num >= MAX_REF_MV_STACK_SIZE {
                        break;
                    }
                    st.mv[st.num] = combined[idx];
                    st.weight[st.num] = 2;
                    st.num += 1;
                }
            }
        } else {
            for idx in st.num..2 {
                st.mv[idx][0] = st.global_mvs[0];
            }
        }
    }

    /// `clamp_mv_row( mvec, border )`（规范 §7.10.2.14 引用的 §5.x 定义）
    fn clamp_mv_row(&self, mvec: i32, border: i32) -> i32 {
        let bh4 = NUM_4X4_BLOCKS_HIGH[self.mi_size as usize] as i32;
        let to_top = -((self.mi_row * 4) * 8);
        let to_bottom = ((self.fh.mi_rows as i32 - bh4 - self.mi_row) * 4) * 8;
        mvec.clamp(to_top - border, to_bottom + border)
    }

    /// `clamp_mv_col( mvec, border )`
    fn clamp_mv_col(&self, mvec: i32, border: i32) -> i32 {
        let bw4 = NUM_4X4_BLOCKS_WIDE[self.mi_size as usize] as i32;
        let to_left = -((self.mi_col * 4) * 8);
        let to_right = ((self.fh.mi_cols as i32 - bw4 - self.mi_col) * 4) * 8;
        mvec.clamp(to_left - border, to_right + border)
    }

    /// *Find MV stack process*（§7.10.2）
    pub fn find_mv_stack(&self, is_compound: bool) -> RefMvStack {
        let mut st = StackState::new();
        st.global_mvs[0] =
            setup_global_mv(self.fh, self.ref_frame[0], 0, self.mi_row, self.mi_col, self.mi_size);
        if is_compound {
            st.global_mvs[1] =
                setup_global_mv(self.fh, self.ref_frame[1], 1, self.mi_row, self.mi_col, self.mi_size);
        }

        st.found_match = false;
        self.scan_row(&mut st, -1, is_compound);
        let mut found_above_match = st.found_match;
        st.found_match = false;
        self.scan_col(&mut st, -1, is_compound);
        let mut found_left_match = st.found_match;
        st.found_match = false;

        let bw4 = NUM_4X4_BLOCKS_WIDE[self.mi_size as usize] as i32;
        let bh4 = NUM_4X4_BLOCKS_HIGH[self.mi_size as usize] as i32;
        if bw4.max(bh4) <= 16 {
            self.scan_point(&mut st, -1, bw4, is_compound);
        }
        if st.found_match {
            found_above_match = true;
        }
        let close_matches = usize::from(found_above_match) + usize::from(found_left_match);
        let num_nearest = st.num;
        let num_new = st.new_mv_count;
        if num_nearest > 0 {
            for idx in 0..num_nearest {
                st.weight[idx] += REF_CAT_LEVEL;
            }
        }
        st.zero_mv_context = 0;
        if self.fh.use_ref_frame_mvs {
            self.temporal_scan(&mut st, is_compound);
        }
        st.found_match = false;
        self.scan_point(&mut st, -1, -1, is_compound);
        if st.found_match {
            found_above_match = true;
        }
        st.found_match = false;
        self.scan_row(&mut st, -3, is_compound);
        if st.found_match {
            found_above_match = true;
        }
        st.found_match = false;
        self.scan_col(&mut st, -3, is_compound);
        if st.found_match {
            found_left_match = true;
        }
        st.found_match = false;
        if bh4 > 1 {
            self.scan_row(&mut st, -5, is_compound);
            if st.found_match {
                found_above_match = true;
            }
        }
        st.found_match = false;
        if bw4 > 1 {
            self.scan_col(&mut st, -5, is_compound);
            if st.found_match {
                found_left_match = true;
            }
        }
        let total_matches = usize::from(found_above_match) + usize::from(found_left_match);

        self.sort_stack(&mut st, 0, num_nearest);
        let total = st.num;
        self.sort_stack(&mut st, num_nearest, total);
        if st.num < 2 {
            self.extra_search(&mut st, is_compound);
        }

        // 上下文与钳位
        let mut drl_ctx = [0usize; MAX_REF_MV_STACK_SIZE];
        for idx in 0..st.num {
            let mut z = 0usize;
            if idx + 1 < st.num {
                let w0 = st.weight[idx];
                let w1 = st.weight[idx + 1];
                if w0 >= REF_CAT_LEVEL {
                    if w1 < REF_CAT_LEVEL {
                        z = 1;
                    }
                } else {
                    z = 2;
                }
            }
            drl_ctx[idx] = z;
        }
        let num_lists = if is_compound { 2 } else { 1 };
        for list in 0..num_lists {
            for idx in 0..st.num {
                let mut m = st.mv[idx][list];
                m[0] = self.clamp_mv_row(m[0], MV_BORDER + block_height(self.mi_size) * 8);
                m[1] = self.clamp_mv_col(m[1], MV_BORDER + block_width(self.mi_size) * 8);
                st.mv[idx][list] = m;
            }
        }
        let (new_mv_context, ref_mv_context) = if close_matches == 0 {
            ((total_matches.min(1)) as usize, total_matches)
        } else if close_matches == 1 {
            ((3 - (num_new.min(1) as i32)) as usize, 2 + total_matches)
        } else {
            ((5 - (num_new.min(1) as i32)) as usize, 5)
        };

        RefMvStack {
            mv: st.mv,
            weight: st.weight,
            num: st.num,
            drl_ctx,
            ref_mv_context,
            new_mv_context,
            zero_mv_context: st.zero_mv_context as usize,
            global_mvs: st.global_mvs,
        }
    }

    /// `has_overlappable_candidates( )`（规范 §7.10.3）
    pub fn has_overlappable_candidates(&self) -> bool {
        let bw4 = NUM_4X4_BLOCKS_WIDE[self.mi_size as usize] as i32;
        let bh4 = NUM_4X4_BLOCKS_HIGH[self.mi_size as usize] as i32;
        if self.is_inside(self.mi_row - 1, self.mi_col) {
            let mut x4 = self.mi_col;
            while x4 < (self.mi_col + bw4).min(self.fh.mi_cols as i32) {
                if self.grid.ref_frames[self.grid.idx(self.mi_row - 1, x4 | 1)][0] > INTRA_FRAME {
                    return true;
                }
                x4 += 2;
            }
        }
        if self.is_inside(self.mi_row, self.mi_col - 1) {
            let mut y4 = self.mi_row;
            while y4 < (self.mi_row + bh4).min(self.fh.mi_rows as i32) {
                if self.grid.ref_frames[self.grid.idx(y4 | 1, self.mi_col - 1)][0] > INTRA_FRAME {
                    return true;
                }
                y4 += 2;
            }
        }
        false
    }

    /// `find_warp_samples( )`（规范 §7.10.4）：只需要 `NumSamples` 是不是 0
    /// （它决定 `read_motion_mode` 读 `use_obmc` 还是 `motion_mode`）。
    pub fn num_warp_samples(&self, mv: Mv) -> usize {
        self.find_warp_samples(mv).0
    }

    /// `find_warp_samples( )`（规范 §7.10.4）：**完整的 `CandList`**。
    ///
    /// 预测阶段（`warp estimation` §7.11.3.8）要的是候选坐标本身，不只是个数。
    /// 返回 `(NumSamples, CandList)`，`CandList[i] = [srcY, srcX, dstY, dstX]`，
    /// 单位都是 **1/8 亮度样本**。
    ///
    /// 调用时机无所谓：`add_sample` 只看邻居（`(-1,0)` / `(0,-1)` / `(-1,-1)` /
    /// `(-1,w4)`），从不读当前块，所以读模式信息时和写回网格之后算出来的候选
    /// 完全一样。
    pub fn find_warp_samples(&self, mv: Mv) -> (usize, [[i32; 4]; LEAST_SQUARES_SAMPLES_MAX]) {
        let w4 = NUM_4X4_BLOCKS_WIDE[self.mi_size as usize] as i32;
        let h4 = NUM_4X4_BLOCKS_HIGH[self.mi_size as usize] as i32;
        let mut out = [[0i32; 4]; LEAST_SQUARES_SAMPLES_MAX];
        let mut num_samples = 0usize;
        let mut num_scanned = 0usize;
        let mut do_top_left = true;
        let mut do_top_right = true;

        if self.is_inside(self.mi_row - 1, self.mi_col) {
            let src_size = self.cand_size(self.mi_row - 1, self.mi_col);
            let src_w = NUM_4X4_BLOCKS_WIDE[src_size as usize] as i32;
            if w4 <= src_w {
                let col_offset = -(self.mi_col & (src_w - 1));
                if col_offset < 0 {
                    do_top_left = false;
                }
                if col_offset + src_w > w4 {
                    do_top_right = false;
                }
                self.add_sample(mv, -1, 0, &mut num_samples, &mut num_scanned, &mut out);
            } else {
                let mut i = 0;
                let mut mi_step;
                while i < w4.min(self.fh.mi_cols as i32 - self.mi_col) {
                    let src_size = self.cand_size(self.mi_row - 1, self.mi_col + i);
                    let src_w = NUM_4X4_BLOCKS_WIDE[src_size as usize] as i32;
                    mi_step = w4.min(src_w);
                    self.add_sample(mv, -1, i, &mut num_samples, &mut num_scanned, &mut out);
                    i += mi_step;
                }
            }
        }
        if self.is_inside(self.mi_row, self.mi_col - 1) {
            let src_size = self.cand_size(self.mi_row, self.mi_col - 1);
            let src_h = NUM_4X4_BLOCKS_HIGH[src_size as usize] as i32;
            if h4 <= src_h {
                let row_offset = -(self.mi_row & (src_h - 1));
                if row_offset < 0 {
                    do_top_left = false;
                }
                self.add_sample(mv, 0, -1, &mut num_samples, &mut num_scanned, &mut out);
            } else {
                let mut i = 0;
                let mut mi_step;
                while i < h4.min(self.fh.mi_rows as i32 - self.mi_row) {
                    let src_size = self.cand_size(self.mi_row + i, self.mi_col - 1);
                    let src_h = NUM_4X4_BLOCKS_HIGH[src_size as usize] as i32;
                    mi_step = h4.min(src_h);
                    self.add_sample(mv, i, -1, &mut num_samples, &mut num_scanned, &mut out);
                    i += mi_step;
                }
            }
        }
        if do_top_left {
            self.add_sample(mv, -1, -1, &mut num_samples, &mut num_scanned, &mut out);
        }
        if do_top_right && w4.max(h4) <= 16 {
            self.add_sample(mv, -1, w4, &mut num_samples, &mut num_scanned, &mut out);
        }
        if num_samples == 0 && num_scanned > 0 {
            num_samples = 1;
        }
        (num_samples, out)
    }

    /// *Add sample process*（规范 §7.10.4.2）
    fn add_sample(
        &self,
        mv: Mv,
        delta_row: i32,
        delta_col: i32,
        num_samples: &mut usize,
        num_scanned: &mut usize,
        cand_list: &mut [[i32; 4]; LEAST_SQUARES_SAMPLES_MAX],
    ) {
        if *num_scanned >= LEAST_SQUARES_SAMPLES_MAX {
            return;
        }
        let mv_row = self.mi_row + delta_row;
        let mv_col = self.mi_col + delta_col;
        if !self.is_inside(mv_row, mv_col) {
            return;
        }
        if !self.is_inter_written(mv_row, mv_col) {
            return;
        }
        let i = self.grid.idx(mv_row, mv_col);
        if self.grid.ref_frames[i][0] != self.ref_frame[0] {
            return;
        }
        if self.grid.ref_frames[i][1] != NONE {
            return;
        }
        let cand_sz = self.cand_size(mv_row, mv_col);
        let cand_w4 = NUM_4X4_BLOCKS_WIDE[cand_sz as usize] as i32;
        let cand_h4 = NUM_4X4_BLOCKS_HIGH[cand_sz as usize] as i32;
        let cand_row = mv_row & !(cand_h4 - 1);
        let cand_col = mv_col & !(cand_w4 - 1);
        let cand_mv = self.grid.mvs[self.grid.idx(cand_row, cand_col)][0];
        let threshold = block_width(self.mi_size)
            .max(block_height(self.mi_size))
            .clamp(16, 112);
        let mv_diff = (cand_mv[0] - mv[0]).abs() + (cand_mv[1] - mv[1]).abs();
        let valid = mv_diff <= threshold;
        *num_scanned += 1;
        if !valid && *num_scanned > 1 {
            return;
        }
        // 候选坐标：块中心（1/8 样本）与其加过 MV 的落点
        let mid_y = cand_row * 4 + cand_h4 * 2 - 1;
        let mid_x = cand_col * 4 + cand_w4 * 2 - 1;
        cand_list[*num_samples] = [mid_y * 8, mid_x * 8, mid_y * 8 + cand_mv[0], mid_x * 8 + cand_mv[1]];
        if valid {
            *num_samples += 1;
        }
    }
}

/// `GLOBALMV` / `GLOBAL_GLOBALMV` 的取值（`NEAREST_NEARESTMV` 起算的 `YMode` 编号）。
pub const NEARESTMV: u8 = 13;
pub const NEARMV: u8 = 14;
pub const GLOBALMV: u8 = 15;
pub const NEWMV: u8 = 16;
pub const NEAREST_NEARESTMV: u8 = 17;
pub const NEAR_NEARMV: u8 = 18;
pub const NEAREST_NEWMV: u8 = 19;
pub const NEW_NEARESTMV: u8 = 20;
pub const NEAR_NEWMV: u8 = 21;
pub const NEW_NEARMV: u8 = 22;
pub const GLOBAL_GLOBALMV: u8 = 23;
pub const NEW_NEWMV: u8 = 24;
/// `MB_MODE_COUNT`
pub const MB_MODE_COUNT: u8 = 25;

/// `CompMode` 的取值
pub const NEWMV_MODE: u8 = 0;
pub const NEARESTMV_MODE: u8 = 1;
pub const NEARMV_MODE: u8 = 2;
pub const GLOBALMV_MODE: u8 = 3;

/// `get_mode( refList )`（规范 §5.11.22）
pub fn get_mode(ymode: u8, ref_list: usize) -> u8 {
    if ref_list == 0 {
        if ymode < NEAREST_NEARESTMV {
            ymode
        } else if matches!(ymode, NEW_NEWMV | NEW_NEARESTMV | NEW_NEARMV) {
            NEWMV
        } else if matches!(ymode, NEAREST_NEARESTMV | NEAREST_NEWMV) {
            NEARESTMV
        } else if matches!(ymode, NEAR_NEARMV | NEAR_NEWMV) {
            NEARMV
        } else {
            GLOBALMV
        }
    } else if matches!(ymode, NEW_NEWMV | NEAREST_NEWMV | NEAR_NEWMV) {
        NEWMV
    } else if matches!(ymode, NEAREST_NEARESTMV | NEW_NEARESTMV) {
        NEARESTMV
    } else if matches!(ymode, NEAR_NEARMV | NEW_NEARMV) {
        NEARMV
    } else {
        GLOBALMV
    }
}

/// `is_scaled( refFrame )`（规范 §5.11.20）
pub fn is_scaled(store: &RefFrameStore, fh: &FrameHeader, ref_frame: i8) -> bool {
    if ref_frame <= INTRA_FRAME {
        return false;
    }
    let ref_idx = fh.ref_frame_idx[(ref_frame - LAST_FRAME) as usize] as usize;
    if ref_idx >= NUM_REF_FRAMES {
        return false;
    }
    let w = fh.frame_width.max(1) as i64;
    let h = fh.frame_height.max(1) as i64;
    let x_scale = ((store.upscaled_width[ref_idx] as i64) << REF_SCALE_SHIFT) + w / 2;
    let y_scale = ((store.frame_height[ref_idx] as i64) << REF_SCALE_SHIFT) + h / 2;
    let no_scale = 1i64 << REF_SCALE_SHIFT;
    (x_scale / w) != no_scale || (y_scale / h) != no_scale
}

/// `is_tx_type_in_set` 用不到，但 `AFFINE` 之类的常量要在这里引用一次，
/// 免得被 clippy 当成未使用（它们表达的是 `GmType` 的取值范围）。
pub const GM_TYPE_MAX: u8 = AFFINE;
/// `GmType` 的非平凡取值下限（`> TRANSLATION` 表示"带形变"）
pub const GM_TYPE_WARPED_MIN: u8 = ROTZOOM;
/// 让 `IDENTITY` 也被引用（`setup_global_mv` 里用到，这里再声明一次语义）
pub const GM_TYPE_IDENTITY: u8 = IDENTITY;
/// `BLOCK_SIZES` 的再导出（`mv.rs` 里查表要用）
pub const BLOCK_SIZE_COUNT: usize = BLOCK_SIZES;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round2_signed_symmetry() {
        for n in 0..16 {
            for x in [-1000i32, -7, -1, 0, 1, 7, 1000] {
                assert_eq!(round2_signed(x, n), -round2_signed(-x, n), "x={x} n={n}");
            }
        }
        assert_eq!(round2_signed(0, 5), 0);
        assert_eq!(round2_signed(16, 4), 1);
        assert_eq!(round2_signed(-16, 4), -1);
    }

    /// `Div_Mult` 是 `(1 << 14) / i` 的定点近似（`i = 1..31`），第 0 项是 0。
    #[test]
    fn div_mult_matches_the_reciprocal() {
        assert_eq!(DIV_MULT[0], 0);
        for i in 1..32 {
            let want = (1u32 << 14) / i as u32;
            let got = DIV_MULT[i] as i64;
            assert!(
                (got - want as i64).abs() <= 1,
                "Div_Mult[{i}] = {got}，应当约等于 {want}"
            );
        }
    }

    /// `project_k( mv, num · Div_Mult[den] )` 必须与参考实现 `mv_projection` **逐值相同**。
    ///
    /// 这是把"两次 i64 乘并成一次"的安全性论证钉死的那条单测：整数乘可结合，
    /// 但**夹取与取整的位置**必须一模一样，所以对全部 `den`（1..=MAX_FRAME_DISTANCE）
    /// × 全部 `num`（-MAX..=MAX）× 一组覆盖值域的 `mv` 全扫一遍。
    #[test]
    fn project_k_matches_the_reference_projection() {
        let mvs: [Mv; 11] = [
            [0, 0],
            [1, -1],
            [100, -100],
            [200, -200],
            [1 << 14, 0],
            [-(1 << 14), 0],
            [8800, -8800],
            [-8800, 8800],
            [4095, 4095],
            [12345, -54321],
            [REFMVS_LIMIT, -REFMVS_LIMIT],
        ];
        for den in 1..=MAX_FRAME_DISTANCE {
            let dm = DIV_MULT[den as usize] as i64;
            for num in -MAX_FRAME_DISTANCE..=MAX_FRAME_DISTANCE {
                let k = num as i64 * dm;
                for mv in mvs {
                    assert_eq!(
                        project_k(mv, k),
                        mv_projection(mv, num, den),
                        "mv={mv:?} num={num} den={den}"
                    );
                }
            }
        }
    }

    fn mv_projection_clamps_and_scales() {
        // `numerator == denominator` 时是恒等变换（定点除法回到原值）
        let mv = [100, -100];
        assert_eq!(mv_projection(mv, 1, 1), [100, -100]);
        assert_eq!(mv_projection(mv, 4, 4), [100, -100]);
        // 分母 2、分子 1 → 减半
        assert_eq!(mv_projection([200, -200], 1, 2), [100, -100]);
        // 极端放大被夹住
        let p = mv_projection([1 << 14, 0], 31, 1);
        assert_eq!(p[0], (1 << 14) - 1);
    }

    #[test]
    fn lower_precision_keeps_sign_and_rounds_away_from_zero() {
        let mut mv = [3, -3];
        lower_mv_precision(&mut mv, false, false);
        assert_eq!(mv, [2, -2]);
        let mut mv = [9, -9];
        lower_mv_precision(&mut mv, false, true);
        assert_eq!(mv, [8, -8]);
        let mut mv = [9, -9];
        lower_mv_precision(&mut mv, true, false);
        assert_eq!(mv, [9, -9], "高精度时原样返回");
    }

    #[test]
    fn get_mode_maps_compound_modes_to_single_ones() {
        assert_eq!(get_mode(NEW_NEWMV, 0), NEWMV);
        assert_eq!(get_mode(NEW_NEARESTMV, 0), NEWMV);
        assert_eq!(get_mode(NEAREST_NEWMV, 0), NEARESTMV);
        assert_eq!(get_mode(NEAR_NEWMV, 0), NEARMV);
        assert_eq!(get_mode(GLOBAL_GLOBALMV, 0), GLOBALMV);
        assert_eq!(get_mode(NEW_NEWMV, 1), NEWMV);
        assert_eq!(get_mode(NEW_NEARESTMV, 1), NEARESTMV);
        assert_eq!(get_mode(NEW_NEARMV, 1), NEARMV);
        assert_eq!(get_mode(GLOBAL_GLOBALMV, 1), GLOBALMV);
        // 单参考时，前 4 个值原样返回
        for m in [NEARESTMV, NEARMV, GLOBALMV, NEWMV] {
            assert_eq!(get_mode(m, 0), m);
        }
    }
}
