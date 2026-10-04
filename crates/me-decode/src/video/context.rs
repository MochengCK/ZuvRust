//! AV1 的**帧级 CDF 上下文**（规范 §7.20 的 `save_cdfs` / `load_cdfs`）。
//!
//! 为什么要单独一层：CDF 是**跨瓦片、跨帧**自适应的 ——
//! 一个帧里所有瓦片共用同一份 CDF（瓦片会就地更新它），而帧与帧之间，
//! 只要 `primary_ref_frame != PRIMARY_REF_NONE`，就要把 CDF 从**参考帧槽**里
//! 整份拷出来接着用（`load_cdfs(ref_frame_idx[primary_ref_frame])`）。
//! 不做这件事，帧间帧（P/B）的所有语法都会用错概率 → 位流解不通
//! （症状是 `exit_symbol` 失败或分区树乱）。
//!
//! ## 规范里的两条硬规则（都容易漏）
//!
//! 1. **`load_cdfs` 会把每个 CDF 的计数器（数组最后一项）重置为 0**。
//!    计数器只影响自适应速率，但它是"这个 CDF 被用过多少次"的记录，
//!    不重置会让新加载的上下文在头几次使用时"学得太慢"。
//! 2. **`save_cdfs(i)` 是每个被刷新的参考槽各存一份**（`refresh_frame_flags`
//!    的每一位），不是"一个帧一份"。所以上下文存储的索引 = **参考槽号**，
//!    没有独立的 `frame_context_idx` 这种语法元素。
//!
//! 因为上下文只会经 `save → load` 往返，`load` 时重置计数器与
//! **`save` 时把计数器清零**完全等价 —— 本模块选后者（[`CdfContext::snapshot`]），
//! 这样存进来的每一份都是"干净"的，读出去直接用。

use super::block::{BlockCdfs, PALETTE_COLOR_CONTEXTS, PALETTE_SIZES};
use super::coeff::CoeffCdfs;
use super::lr::LrCdfs;
use super::partition::PartitionCdfs;

/// 一帧用到的全部 CDF（`init_non_coeff_cdfs` + `init_coeff_cdfs` 的并集）。
///
/// 三个成员分别对应三组语法：分区树 / 块级模式 / 系数与变换类型。
#[derive(Debug, Clone)]
pub struct CdfContext {
    pub partition: PartitionCdfs,
    pub block: BlockCdfs,
    pub coeff: CoeffCdfs,
    /// 环路恢复的块级 CDF（§5.11.48）。**必须放在这里**而不是解码驱动里 ——
    /// 它要随 `save_cdfs` / `load_cdfs` 在帧间继承，否则符号的概率模型与编码器
    /// 对不上，读出来的位就整体错位（比"少读几位"隐蔽得多）。
    pub lr: LrCdfs,
}

impl CdfContext {
    /// 帧级默认值（`primary_ref_frame == PRIMARY_REF_NONE` 时用）：
    /// 非系数 CDF 用规范默认表，系数 CDF 按 `base_q_idx` 选档。
    pub fn defaults(base_q_idx: u32) -> Self {
        Self {
            partition: PartitionCdfs::default(),
            block: BlockCdfs::default(),
            coeff: CoeffCdfs::new(base_q_idx),
            lr: LrCdfs::defaults(),
        }
    }

    /// `save_cdfs` 的落盘副本：所有计数器清零（等价于规范在 load 时重置）。
    pub fn snapshot(&self) -> Self {
        let mut c = self.clone();
        c.partition.reset_counts();
        c.block.reset_counts();
        c.coeff.reset_counts();
        c.lr.reset_counts();
        c
    }
}

// ── 计数器清零：所有 CDF 数组的**最后一项**是"用过多少次" ──────────────
//
// 三种深度的数组都有（一维 / 二维 / 三维 / 四维），用 const 泛型写四个
// 递归形态的小函数，字段那边逐个调用即可。

fn z1<const N: usize>(a: &mut [u16; N]) {
    a[N - 1] = 0;
}

fn z2<const N: usize, const M: usize>(a: &mut [[u16; N]; M]) {
    for r in a.iter_mut() {
        z1(r);
    }
}

fn z3<const N: usize, const M: usize, const K: usize>(a: &mut [[[u16; N]; M]; K]) {
    for r in a.iter_mut() {
        z2(r);
    }
}

fn z4<const N: usize, const M: usize, const K: usize, const L: usize>(
    a: &mut [[[[u16; N]; M]; K]; L],
) {
    for r in a.iter_mut() {
        z3(r);
    }
}

impl PartitionCdfs {
    /// 把五张分区表的计数器清零。
    pub fn reset_counts(&mut self) {
        z2(&mut self.w8);
        z2(&mut self.w16);
        z2(&mut self.w32);
        z2(&mut self.w64);
        z2(&mut self.w128);
    }
}

impl BlockCdfs {
    /// 把块级所有 CDF 的计数器清零（字段顺序照结构体定义）。
    pub fn reset_counts(&mut self) {
        z1(&mut self.intrabc);
        z3(&mut self.intra_frame_y_mode);
        z2(&mut self.y_mode);
        z2(&mut self.uv_mode_cfl_allowed);
        z2(&mut self.uv_mode_cfl_not_allowed);
        z2(&mut self.angle_delta);
        z2(&mut self.filter_intra);
        z1(&mut self.filter_intra_mode);
        z2(&mut self.skip);
        z2(&mut self.skip_mode);
        z2(&mut self.segment_id);
        z2(&mut self.segment_id_predicted);
        z1(&mut self.delta_q);
        z1(&mut self.delta_lf);
        z1(&mut self.cfl_sign);
        z2(&mut self.cfl_alpha);
        z3(&mut self.palette_y_mode);
        z2(&mut self.palette_uv_mode);
        z2(&mut self.palette_y_size);
        z2(&mut self.palette_uv_size);
        // 颜色索引表**不能**用 z1/z2：解码时按 `&mut cdf[..size + 1]` 切片，
        // 指的是 `size` 个符号 ⇒ 计数器落在下标 `size`（= 档号 k + 2），不是末项。
        for k in 0..PALETTE_SIZES {
            for c in 0..PALETTE_COLOR_CONTEXTS {
                self.palette_color_y[k][c][k + 2] = 0;
                self.palette_color_uv[k][c][k + 2] = 0;
            }
        }
        z2(&mut self.tx8x8);
        z2(&mut self.tx16x16);
        z2(&mut self.tx32x32);
        z2(&mut self.tx64x64);
        z2(&mut self.txfm_split);
        z2(&mut self.is_inter);
        z2(&mut self.comp_mode);
        z3(&mut self.comp_ref);
        z3(&mut self.comp_bwdref);
        z3(&mut self.single_ref);
        z2(&mut self.comp_ref_type);
        z3(&mut self.uni_comp_ref);
        z2(&mut self.compound_mode);
        z2(&mut self.new_mv);
        z2(&mut self.zero_mv);
        z2(&mut self.ref_mv);
        z2(&mut self.drl_mode);
        z2(&mut self.interintra);
        z2(&mut self.interintra_mode);
        z2(&mut self.wedge_interintra);
        z2(&mut self.wedge_index);
        z2(&mut self.use_obmc);
        z2(&mut self.motion_mode);
        z2(&mut self.compound_type);
        z2(&mut self.comp_group_idx);
        z2(&mut self.compound_idx);
        z2(&mut self.interp_filter);
        z2(&mut self.mv_joint);
        z3(&mut self.mv_class);
        z3(&mut self.mv_class0_bit);
        z4(&mut self.mv_class0_fr);
        z3(&mut self.mv_class0_hp);
        z3(&mut self.mv_fr);
        z3(&mut self.mv_hp);
        z4(&mut self.mv_bit);
        z3(&mut self.mv_sign);
    }
}

impl CoeffCdfs {
    /// 把系数类 CDF 的计数器清零（含变换类型那几张）。
    pub fn reset_counts(&mut self) {
        z3(&mut self.txb_skip);
        z3(&mut self.eob_pt_16);
        z3(&mut self.eob_pt_32);
        z3(&mut self.eob_pt_64);
        z3(&mut self.eob_pt_128);
        z3(&mut self.eob_pt_256);
        z2(&mut self.eob_pt_512);
        z2(&mut self.eob_pt_1024);
        z4(&mut self.eob_extra);
        z3(&mut self.dc_sign);
        z4(&mut self.coeff_base_eob);
        z4(&mut self.coeff_base);
        z4(&mut self.coeff_br);
        z3(&mut self.intra_tx_type_set1);
        z3(&mut self.intra_tx_type_set2);
        z2(&mut self.inter_tx_type_set1);
        z1(&mut self.inter_tx_type_set2);
        z2(&mut self.inter_tx_type_set3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_zeroes_every_counter_but_keeps_the_distribution() {
        let mut ctx = CdfContext::defaults(100);
        // 把几份 CDF 弄成"用过很多次"的状态（计数器涨、分布变形）
        ctx.block.skip[0][2] = 17;
        ctx.block.skip[0][0] = 12345;
        ctx.partition.w8[0][4] = 7;
        ctx.coeff.txb_skip[0][0][2] = 31;
        ctx.coeff.inter_tx_type_set2[12] = 5;

        let snap = ctx.snapshot();
        assert_eq!(snap.block.skip[0][2], 0, "块级计数器要清零");
        assert_eq!(snap.partition.w8[0][4], 0, "分区计数器要清零");
        assert_eq!(snap.coeff.txb_skip[0][0][2], 0, "系数计数器要清零");
        assert_eq!(snap.coeff.inter_tx_type_set2[12], 0, "一维表也要清");
        // 分布本体必须原样保留（save 存的是自适应之后的 CDF）
        assert_eq!(snap.block.skip[0][0], 12345, "分布不该被 snapshot 改动");
    }

    #[test]
    fn defaults_pick_the_coeff_tier_by_base_q() {
        // init_coeff_cdfs 按 base_q_idx 选 4 档；用两张不同档的表验证选档确实生效
        let low = CdfContext::defaults(10);
        let high = CdfContext::defaults(200);
        assert_ne!(
            low.coeff.txb_skip[0][0][0], high.coeff.txb_skip[0][0][0],
            "不同 base_q 档位的系数 CDF 应当不同"
        );
    }
}
