//! 测试用的极简**位写入器**（只在 `cfg(test)` 下编译）。
//!
//! 为什么要它：AV1 的序列头是纯位域，用十六进制常量写测试等于"把解析器再手算
//! 一遍"，改一个字段就要重算整段字节。按规范顺序写位可就地读、可就地改。
//!
//! 它的局限必须在测试里说清楚：**它与解析器出自同一份理解**，所以只能证明
//! "位序自洽"，不能证明"规范理解没错"。真正的验收要拿真实码流的预期值
//! （见 `video/mod.rs` 进度表里"真实码流验收"那条）。

use super::sequence_header::SequenceHeader;

/// 按位写（大端位序，与 [`super::bitreader::BitReader`] 对称）。
#[derive(Default)]
pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    bit: u8,
}

impl BitWriter {
    /// 写 `bits` 位（低位对齐的值）。
    pub(crate) fn f(&mut self, value: u32, bits: u32) -> &mut Self {
        for i in (0..bits).rev() {
            let b = ((value >> i) & 1) as u8;
            if self.bit == 0 {
                self.bytes.push(0);
            }
            let last = self.bytes.len() - 1;
            self.bytes[last] |= b << (7 - self.bit);
            self.bit = (self.bit + 1) % 8;
        }
        self
    }

    pub(crate) fn bit(&mut self, v: bool) -> &mut Self {
        self.f(v as u32, 1)
    }

    /// 语法结构末尾的 `trailing_bits()`：一个 `1`，后面交给 `finish()` 补 0。
    ///
    /// 每个造序列头的测试都要以它收尾 —— 解析器那边会**校验**这一位
    /// （那是抓位偏移最有效的检查，详见 `bitreader::BitReader::trailing_bits`）。
    pub(crate) fn trailing(&mut self) -> &mut Self {
        self.f(1, 1)
    }

    /// `byte_alignment()`：补 0 到字节边界（非压缩头末尾要求这个）。
    pub(crate) fn byte_align(&mut self) -> &mut Self {
        while self.bit != 0 {
            self.f(0, 1);
        }
        self
    }

    /// 已经写出来的字节（未对齐的尾巴按 0 补齐）。
    pub(crate) fn finish(&self) -> &[u8] {
        &self.bytes
    }
}

/// 造一个"1920×1080、8 位 4:2:0、简化静图模式"的序列头载荷。
///
/// 选简化模式是因为**它最短**：没有 timing/decoder model/operating point 循环，
/// 工具开关整段省略 —— 用它当"能跑通 OBU + 序列头两层"的样本最省事。
pub(crate) fn write_reduced_still_picture_1080p(w: &mut BitWriter) {
    w.f(0, 3); // seq_profile = 0（4:2:0 8/10 位）
    w.bit(true); // still_picture
    w.bit(true); // reduced_still_picture_header
    w.f(8, 5); // seq_level_idx = 8（4.0）
    w.f(10, 4); // frame_width_bits_minus_1 → 11 位
    w.f(10, 4); // frame_height_bits_minus_1 → 11 位
    w.f(1919, 11); // max_frame_width_minus_1 → 1920
    w.f(1079, 11); // max_frame_height_minus_1 → 1080
    w.bit(false); // use_128x128_superblock
    w.bit(false); // enable_filter_intra
    w.bit(false); // enable_intra_edge_filter
    w.bit(false); // enable_superres
    w.bit(true); // enable_cdef
    w.bit(false); // enable_restoration
    w.bit(false); // high_bitdepth → 8 位
    w.bit(false); // mono_chrome
    w.bit(false); // color_description_present_flag
    w.bit(false); // color_range
    w.f(0, 2); // chroma_sample_position
    w.bit(false); // separate_uv_delta_q
    w.bit(false); // film_grain_params_present
    w.trailing(); // trailing_bits
}

/// 按给定的序列头，写一段**最简的非压缩头参数段**（M2.5/M2.6）。
///
/// "最简"= 每个可选分支都走"不用读更多位"的那一支：
/// 单个瓦片、所有 delta 为 0、无分割、无环路恢复、无全局运动、无胶片颗粒。
/// 它让所有造合成帧头的测试都能一路读到 `byte_alignment()` —— 也就是说，
/// **合成测试同样受那道位偏移硬校验约束**（写少一位就会在末尾暴露）。
///
/// 因为依赖序列头的属性，所以签名直接收 `&SequenceHeader`：
/// 换了序列头（128 超级块、mono、enable_restoration 开关…）不需要改这里。
pub(crate) fn write_min_params(w: &mut BitWriter, seq: &SequenceHeader, is_intra: bool) {
    use super::params::{
        MAX_TILE_AREA, MAX_TILE_COLS, MAX_TILE_ROWS, MAX_TILE_WIDTH, TOTAL_REFS_PER_FRAME,
    };

    // ── tile_info()：uniform = 1，之后两个 while 立刻 break ──
    let mi_cols = 2 * ((seq.max_frame_width + 7) >> 3);
    let mi_rows = 2 * ((seq.max_frame_height + 7) >> 3);
    let (sb_cols, sb_rows, sb_shift) = if seq.use_128x128_superblock {
        ((mi_cols + 31) >> 5, (mi_rows + 31) >> 5, 5u32)
    } else {
        ((mi_cols + 15) >> 4, (mi_rows + 15) >> 4, 4u32)
    };
    let sb_size = sb_shift + 2;
    let max_tile_width_sb = MAX_TILE_WIDTH >> sb_size;
    let max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size);
    let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
    let max_log2_tile_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS));
    let max_log2_tile_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS));
    let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));
    let cols_log2 = min_log2_tile_cols;
    let min_log2_tile_rows = min_log2_tiles.saturating_sub(cols_log2);

    w.bit(true); // uniform_tile_spacing_flag
    // 解析器的 `while` 在读到第一个 0 就 break —— 所以这里也只写一个 0，
    // 而不是把 `max - min` 个都写满（写满会让后面的字段整体错位）。
    if max_log2_tile_cols > cols_log2 {
        w.bit(false); // increment_tile_cols_log2 = 0 → break
    }
    if max_log2_tile_rows > min_log2_tile_rows {
        w.bit(false); // increment_tile_rows_log2 = 0 → break
    }
    // 单瓦片 → 不写 context_update_tile_id / tile_size_bytes

    // ── quantization_params() ──
    // base_q_idx 取非 0：这样 CodedLossless = 0，才会走到 cdef/lr/tx_mode 的读位分支
    w.f(100, 8);
    w.bit(false); // DeltaQYDc：delta_coded = 0
    if !seq.color.mono_chrome {
        if seq.color.separate_uv_delta_q {
            w.bit(false); // diff_uv_delta
        }
        w.bit(false); // DeltaQUDc
        w.bit(false); // DeltaQUAc
    }
    w.bit(false); // using_qmatrix

    // ── segmentation_params() ──
    w.bit(false); // segmentation_enabled

    // ── delta_q_params() / delta_lf_params() ──
    w.bit(false); // delta_q_present = 0（base_q_idx > 0 才读）

    // ── loop_filter_params()：level 全 0 → 不读 level[2]/[3] ──
    w.f(0, 6); // loop_filter_level[0]
    w.f(0, 6); // loop_filter_level[1]
    w.f(0, 3); // loop_filter_sharpness
    w.bit(false); // loop_filter_delta_enabled

    // ── cdef_params() ──
    if seq.enable_cdef {
        w.f(0, 2); // cdef_damping_minus_3
        w.f(0, 2); // cdef_bits = 0 → 一组
        w.f(0, 4); // cdef_y_pri_strength
        w.f(0, 2); // cdef_y_sec_strength
        if !seq.color.mono_chrome {
            w.f(0, 4); // cdef_uv_pri_strength
            w.f(0, 2); // cdef_uv_sec_strength
        }
    }

    // ── lr_params()：三个平面都是 RESTORE_NONE → UsesLr = 0 ──
    if seq.enable_restoration {
        let planes = if seq.color.mono_chrome { 1 } else { 3 };
        for _ in 0..planes {
            w.f(0, 2); // lr_type = RESTORE_NONE
        }
    }

    // ── read_tx_mode() ──
    w.bit(false); // tx_mode_select = 0 → TX_MODE_LARGEST

    // ── frame_reference_mode() ──
    if !is_intra {
        w.bit(false); // reference_select = 0
    }

    // ── skip_mode_params()：reference_select = 0 → skipModeAllowed = 0，不读位 ──

    // ── allow_warped_motion / reduced_tx_set ──
    if !is_intra && seq.enable_warped_motion {
        w.bit(false); // allow_warped_motion
    }
    w.bit(false); // reduced_tx_set

    // ── global_motion_params()：7 个参考帧 is_global = 0 ──
    if !is_intra {
        for _ in 1..TOTAL_REFS_PER_FRAME {
            w.bit(false);
        }
    }

    // ── film_grain_params()：film_grain_params_present = false → 不读位 ──
}

/// `tile_log2`（规范 §5.9.12）的测试侧副本。
fn tile_log2(blk_size: u32, target: u32) -> u32 {
    let mut k = 0u32;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}
