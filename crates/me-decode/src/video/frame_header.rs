//! AV1 **帧头（非压缩头）** —— 自研软解的 M2（规范 §5.9.2 与 §6.5 的语法）。
//!
//! ## 这一层解析到哪、到哪停
//!
//! 非压缩头很长，这里按规范顺序**读到 `byte_alignment()` 为止**（即整段非压缩头）：
//!
//! ```text
//! ✅ show_existing_frame / frame_type / show_frame / showable_frame / error_resilient_mode
//! ✅ disable_cdf_update / 屏幕内容工具 / force_integer_mv / current_frame_id
//! ✅ frame_size_override_flag / order_hint / primary_ref_frame / buffer_removal_time
//! ✅ refresh_frame_flags / 参考帧的 ref_order_hint 校验
//! ✅ 帧尺寸：frame_size() / render_size() / frame_size_with_refs() + 超分（§6.5.6~6.5.10）
//! ✅ 参考帧索引（含 frame_refs_short_signaling 的 set_frame_refs 全过程）
//! ✅ allow_high_precision_mv / interpolation_filter / is_motion_mode_switchable / use_ref_frame_mvs
//! ✅ disable_frame_end_update_cdf
//! ✅ 参数段（M2.5/M2.6，见 [`super::params`]）：tile_info → quantization → segmentation
//!    → delta_q / delta_lf → loop_filter → cdef → lr → tx_mode → frame_reference_mode
//!    → skip_mode → allow_warped_motion / reduced_tx_set → global_motion → film_grain
//! ✅ byte_alignment()：填充位必须是 0（**位偏移的硬校验**）
//! ⛔ 再往下是瓦片数据（熵层，M3）
//! ```
//!
//! `bits_consumed` 给的是**对齐之后**的位置，也就是瓦片数据的起点（对 `OBU_FRAME`
//! 而言）；`payload_bits_after_header` 给的是该 OBU 载荷里剩下多少位 ——
//! `OBU_FRAME_HEADER` 应当剩 0，`OBU_FRAME` 剩的是整个 tile group。
//!
//! ## 参考帧状态
//!
//! 帧头里有几处**必须知道"之前解过的帧"**才能读对：
//!
//! - `show_existing_frame` 时 `frame_type = RefFrameType[idx]`；
//! - 帧尺寸可能从参考帧继承（`frame_size_with_refs`）；
//! - `frame_refs_short_signaling` 要用 `RefOrderHint[]` 现算 `ref_frame_idx[]`；
//! - `skip_mode_params()` 要用 `RefOrderHint[ref_frame_idx[i]]`。
//!
//! 所以这一层带一个 [`RefFrameStore`]：既存非压缩头用得上的那几个字段（有效性、
//! order hint、帧类型、尺寸），也存"没写就沿用上一帧"的那几组参数
//! （[`SavedFrameState`]：环路滤波 delta / 分割特征 / 全局运动）。它按规范的
//! "参考帧更新过程"（§7.20）在每帧解析完之后刷新 —— 像素本身不在这里（M6 才有帧存储）。

use me_core::{MediaError, Result};

use super::bitreader::BitReader;
use super::context::CdfContext;
use super::intra::CurrFrame;
use super::params::{FilmGrainParams, FrameParams, SavedFrameState};
use super::sequence_header::{SequenceHeader, SELECT_INTEGER_MV, SELECT_SCREEN_CONTENT_TOOLS};

/// `REFS_PER_FRAME`：可用于帧间预测的参考帧类型数（LAST/LAST2/LAST3/GOLDEN/BWDREF/ALTREF2/ALTREF）。
pub const REFS_PER_FRAME: usize = 7;
/// `NUM_REF_FRAMES`：参考帧槽位数。
pub const NUM_REF_FRAMES: usize = 8;
/// `PRIMARY_REF_NONE`：`primary_ref_frame` 的特殊值（表示没有主参考帧）。
pub const PRIMARY_REF_NONE: u8 = 7;
/// `allFrames` = `(1 << NUM_REF_FRAMES) - 1`。
pub const ALL_FRAMES: u8 = 0xFF;

/// `SUPERRES_NUM`：超分比的分母（不超分时 `SuperresDenom` 就是它）。
pub const SUPERRES_NUM: u32 = 8;
/// `SUPERRES_DENOM_MIN`：最小超分分母。
pub const SUPERRES_DENOM_MIN: u32 = 9;
/// `SUPERRES_DENOM_BITS`：`coded_denom` 的位数。
pub const SUPERRES_DENOM_BITS: u32 = 3;

/// 帧类型（`frame_type` 的取值，语义见规范 §6.5.2 的表）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// 0：关键帧（重置解码过程）
    Key,
    /// 1：帧间帧
    Inter,
    /// 2：只帧内（不重置解码过程）
    IntraOnly,
    /// 3：切换帧
    Switch,
}

impl FrameType {
    fn from_code(code: u32) -> Self {
        match code {
            0 => FrameType::Key,
            1 => FrameType::Inter,
            2 => FrameType::IntraOnly,
            _ => FrameType::Switch,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            FrameType::Key => "key",
            FrameType::Inter => "inter",
            FrameType::IntraOnly => "intra-only",
            FrameType::Switch => "switch",
        }
    }

    /// 帧内帧（KEY / INTRA_ONLY）：只用帧内预测，不引用之前的帧。
    pub fn is_intra(self) -> bool {
        matches!(self, FrameType::Key | FrameType::IntraOnly)
    }
}

/// OBU 层的层级标识（帧头的 `buffer_removal_time` 循环要用它判断"这个
/// operating point 是否属于本包所在的时空层"）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ObuLayerIds {
    pub temporal_id: u8,
    pub spatial_id: u8,
}

/// 非压缩头之后那一段"收尾"的写法 —— **两种 OBU 不一样**（规范 §5.9.1）：
///
/// | 承载它的 OBU | 收尾 | 为什么 |
/// |---|---|---|
/// | `OBU_FRAME_HEADER` | `trailing_bits()`（一个 `1` + 补 `0` 到载荷末尾）| 载荷到此结束 |
/// | `OBU_FRAME` | `byte_alignment()`（补 `0` 到字节边界）| 后面还接**字节对齐**的瓦片组 |
///
/// 真实素材给出的铁证：这条流的 `show_existing_frame` 帧载荷是 1 字节 `0xc8`
/// = `1`（show_existing）+ `100`（槽 4）+ `1000` —— 最后四位是 `1` 加三个 `0`，
/// 正是 `trailing_bits`。当成 `byte_alignment` 处理会报"对齐位不是 0"。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderTail {
    /// `OBU_FRAME_HEADER`：`trailing_bits()`
    TrailingBits,
    /// `OBU_FRAME`：`byte_alignment()`，之后是瓦片组
    ByteAlignment,
}

/// 解析出来的帧头（M2 覆盖的部分，见模块抬头）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameHeader {
    /// 这一帧只是"把某个已解过的参考帧再显示一次"，没有新的压缩数据
    pub show_existing_frame: bool,
    pub frame_to_show_map_idx: u8,
    pub frame_type: FrameType,
    /// `FrameIsIntra`：帧内帧（KEY / INTRA_ONLY）
    pub is_intra: bool,
    pub show_frame: bool,
    pub showable_frame: bool,
    pub error_resilient_mode: bool,
    pub disable_cdf_update: bool,
    pub allow_screen_content_tools: bool,
    pub force_integer_mv: bool,
    /// 只有 `frame_id_numbers_present` 时才有意义（否则规范给 0）
    pub current_frame_id: u32,
    pub frame_size_override: bool,
    pub order_hint: u8,
    pub primary_ref_frame: u8,
    pub refresh_frame_flags: u8,

    /// 超分之前（= 编码时的宽度）；`FrameWidth` 是它的显示宽度
    pub upscaled_width: u32,
    /// 超分之后的宽度（不超分时等于 `upscaled_width`）
    pub frame_width: u32,
    pub frame_height: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub use_superres: bool,
    pub superres_denom: u32,
    /// 模式信息单元列数（`2 * ((FrameWidth + 7) >> 3)`）
    pub mi_cols: u32,
    pub mi_rows: u32,

    pub allow_intrabc: bool,
    /// 帧间帧才有意义
    pub frame_refs_short_signaling: bool,
    pub ref_frame_idx: [u8; REFS_PER_FRAME],
    /// `OrderHints[ref]`（`LAST_FRAME..ALTREF_FRAME`；`INTRA_FRAME` 位不用）
    pub order_hints: [u8; 8],
    /// `RefFrameSignBias[ref]`（`get_relative_dist(hint, OrderHint) > 0`）
    pub ref_frame_sign_bias: [bool; 8],
    /// 尺寸是从参考帧继承来的（`frame_size_with_refs` 里 `found_ref == 1`）
    pub size_from_ref: bool,
    pub allow_high_precision_mv: bool,
    /// `interpolation_filter`（`read_interpolation_filter()`）
    pub interpolation_filter: u8,
    pub is_motion_mode_switchable: bool,
    pub use_ref_frame_mvs: bool,
    pub disable_frame_end_update_cdf: bool,

    /// 非压缩头的参数段（M2.5/M2.6，规范 §5.9.2 的 `tile_info()` 起）。
    /// `show_existing_frame` 的帧没有这一段（规范在读到这里之前就 return 了）。
    pub params: FrameParams,

    /// 非压缩头 + `byte_alignment()` 读到第几位（= 瓦片数据的起点）
    pub bits_consumed: usize,
    /// 对齐之后这个 OBU 载荷里还剩多少位（`OBU_FRAME_HEADER` 应当剩 0）
    pub payload_bits_after_header: usize,
}

impl Default for FrameHeader {
    fn default() -> Self {
        Self {
            show_existing_frame: false,
            frame_to_show_map_idx: 0,
            frame_type: FrameType::Key,
            is_intra: true,
            show_frame: false,
            showable_frame: false,
            error_resilient_mode: false,
            disable_cdf_update: false,
            allow_screen_content_tools: false,
            force_integer_mv: false,
            current_frame_id: 0,
            frame_size_override: false,
            order_hint: 0,
            primary_ref_frame: PRIMARY_REF_NONE,
            refresh_frame_flags: 0,
            upscaled_width: 0,
            frame_width: 0,
            frame_height: 0,
            render_width: 0,
            render_height: 0,
            use_superres: false,
            superres_denom: SUPERRES_NUM,
            mi_cols: 0,
            mi_rows: 0,
            allow_intrabc: false,
            frame_refs_short_signaling: false,
            ref_frame_idx: [0; REFS_PER_FRAME],
            order_hints: [0; 8],
            ref_frame_sign_bias: [false; 8],
            size_from_ref: false,
            allow_high_precision_mv: false,
            interpolation_filter: 0,
            is_motion_mode_switchable: false,
            use_ref_frame_mvs: false,
            disable_frame_end_update_cdf: false,
            params: FrameParams::default(),
            bits_consumed: 0,
            payload_bits_after_header: 0,
        }
    }
}

impl FrameHeader {
    /// 解析一帧的非压缩头。
    ///
    /// `store` 是参考帧状态：**读的时候要用它**（继承尺寸 / `show_existing_frame`
    /// 的帧类型 / `set_frame_refs`），所以调用方要在解析完这一帧之后调
    /// [`RefFrameStore::refresh`] 把它推进到"这一帧已解码"的状态。
    pub fn parse(
        payload: &[u8],
        seq: &SequenceHeader,
        store: &mut RefFrameStore,
        obu: ObuLayerIds,
        tail: HeaderTail,
    ) -> Result<Self> {
        let mut r = BitReader::new(payload);
        let mut fh = FrameHeader::default();
        let id_len = if seq.frame_id_numbers_present { seq.frame_id_length as u32 } else { 0 };

        if seq.reduced_still_picture_header {
            // 简化静图模式：身份与显示行为全部写死，连 error_resilient_mode 都不读
            fh.show_existing_frame = false;
            fh.frame_type = FrameType::Key;
            fh.is_intra = true;
            fh.show_frame = true;
            fh.showable_frame = false;
        } else {
            fh.show_existing_frame = r.bit()?;
            if fh.show_existing_frame {
                fh.frame_to_show_map_idx = r.f(3)? as u8;
                if seq.decoder_model_info_present && !seq.equal_picture_interval() {
                    Self::parse_temporal_point_info(&mut r, seq)?;
                }
                fh.refresh_frame_flags = 0;
                if seq.frame_id_numbers_present {
                    let _display_frame_id = r.f(id_len)?;
                }
                fh.frame_type = store.frame_type(fh.frame_to_show_map_idx);
                fh.is_intra = fh.frame_type.is_intra();
                if fh.frame_type == FrameType::Key {
                    fh.refresh_frame_flags = ALL_FRAMES;
                }
                // film_grain_params_present 时规范调 load_grain_params()（只影响输出，
                // 不读位）；这里不需要知道，M2 不解释它
                let slot = fh.frame_to_show_map_idx;
                // 这一帧的"有效 order hint"是**被显示的那一帧的**（规范不在这里读位，
                // 显示顺序取 RefOrderHint[slot]）。记下来之后，"显示顺序"就能被验证：
                // 真实素材里这一帧负责把隐藏帧按 0,1,2,3… 逐个显示出来。
                fh.order_hint = store.order_hint[(slot as usize).min(NUM_REF_FRAMES - 1)];
                Self::adopt_ref_size(&mut fh, store, slot);
                // 这一条路规范在读到这里就 return 了：没有 `tile_info()` 及之后的一切。
                // 但 OBU 层面仍有收尾（真实素材里这种帧的载荷正好 1 字节）。
                Self::parse_tail(&mut r, tail)?;
                fh.bits_consumed = r.bits_read();
                fh.payload_bits_after_header = r.remaining_bits();
                return Ok(fh);
            }

            fh.frame_type = FrameType::from_code(r.f(2)?);
            fh.is_intra = fh.frame_type.is_intra();
            fh.show_frame = r.bit()?;
            if fh.show_frame && seq.decoder_model_info_present && !seq.equal_picture_interval() {
                Self::parse_temporal_point_info(&mut r, seq)?;
            }
            if fh.show_frame {
                fh.showable_frame = fh.frame_type != FrameType::Key;
            } else {
                fh.showable_frame = r.bit()?;
            }
            if fh.frame_type == FrameType::Switch
                || (fh.frame_type == FrameType::Key && fh.show_frame)
            {
                fh.error_resilient_mode = true;
            } else {
                fh.error_resilient_mode = r.bit()?;
            }
        }

        // 关键帧 + 显示：参考帧全部失效（旧状态一律丢掉）
        // （规范在这里只清 RefValid/RefOrderHint，M2 的 store 只存这几个字段）

        fh.disable_cdf_update = r.bit()?;

        fh.allow_screen_content_tools = if seq.seq_force_screen_content_tools
            == SELECT_SCREEN_CONTENT_TOOLS
        {
            r.bit()?
        } else {
            seq.seq_force_screen_content_tools != 0
        };
        fh.force_integer_mv = if fh.allow_screen_content_tools {
            if seq.seq_force_integer_mv == SELECT_INTEGER_MV {
                r.bit()?
            } else {
                seq.seq_force_integer_mv != 0
            }
        } else {
            false
        };
        if fh.is_intra {
            fh.force_integer_mv = true;
        }

        fh.current_frame_id = if seq.frame_id_numbers_present { r.f(id_len)? } else { 0 };

        fh.frame_size_override = if fh.frame_type == FrameType::Switch {
            true
        } else if seq.reduced_still_picture_header {
            false
        } else {
            r.bit()?
        };
        fh.order_hint = r.f(seq.order_hint_bits as u32)? as u8;

        fh.primary_ref_frame = if fh.is_intra || fh.error_resilient_mode {
            PRIMARY_REF_NONE
        } else {
            r.f(3)? as u8
        };

        if seq.decoder_model_info_present {
            let present = r.bit()?;
            if present {
                Self::parse_buffer_removal_times(&mut r, seq, obu)?;
            }
        }

        fh.refresh_frame_flags = if fh.frame_type == FrameType::Switch
            || (fh.frame_type == FrameType::Key && fh.show_frame)
        {
            ALL_FRAMES
        } else {
            r.f(8)? as u8
        };

        if (!fh.is_intra || fh.refresh_frame_flags != ALL_FRAMES)
            && fh.error_resilient_mode
            && seq.enable_order_hint
        {
            for slot in 0..NUM_REF_FRAMES {
                let hint = r.f(seq.order_hint_bits as u32)? as u8;
                if hint != store.order_hint[slot] {
                    store.valid[slot] = false;
                }
            }
        }

        if fh.is_intra {
            Self::parse_frame_size(&mut r, seq, &mut fh)?;
            Self::parse_render_size(&mut r, &mut fh)?;
            if fh.allow_screen_content_tools && fh.upscaled_width == fh.frame_width {
                fh.allow_intrabc = r.bit()?;
            }
        } else {
            if !seq.enable_order_hint {
                fh.frame_refs_short_signaling = false;
            } else {
                fh.frame_refs_short_signaling = r.bit()?;
                if fh.frame_refs_short_signaling {
                    let last_frame_idx = r.f(3)? as u8;
                    let gold_frame_idx = r.f(3)? as u8;
                    fh.ref_frame_idx = set_frame_refs(
                        store,
                        seq,
                        fh.order_hint,
                        last_frame_idx,
                        gold_frame_idx,
                    );
                }
            }
            for i in 0..REFS_PER_FRAME {
                if !fh.frame_refs_short_signaling {
                    fh.ref_frame_idx[i] = r.f(3)? as u8;
                }
                if seq.frame_id_numbers_present {
                    // delta_frame_id：只用于一致性检查，M2 读完即丢
                    let n = seq.delta_frame_id_length_minus_2 as u32 + 2;
                    let _delta_frame_id_minus_1 = r.f(n)?;
                }
            }
            if fh.frame_size_override && !fh.error_resilient_mode {
                Self::parse_frame_size_with_refs(&mut r, seq, store, &mut fh)?;
            } else {
                Self::parse_frame_size(&mut r, seq, &mut fh)?;
                Self::parse_render_size(&mut r, &mut fh)?;
            }
            if fh.force_integer_mv {
                fh.allow_high_precision_mv = false;
            } else {
                fh.allow_high_precision_mv = r.bit()?;
            }
            fh.interpolation_filter = Self::parse_interpolation_filter(&mut r)?;
            fh.is_motion_mode_switchable = r.bit()?;
            fh.use_ref_frame_mvs = if fh.error_resilient_mode || !seq.enable_ref_frame_mvs {
                false
            } else {
                r.bit()?
            };
        }
        // `OrderHints[ref]` 与 `RefFrameSignBias[ref]`（规范 §5.9.2；不读位）
        for i in 0..REFS_PER_FRAME {
            let ref_frame = 1 + i; // LAST_FRAME .. ALTREF_FRAME
            let hint = store.order_hint
                [(fh.ref_frame_idx[i] as usize).min(NUM_REF_FRAMES - 1)];
            fh.order_hints[ref_frame] = hint;
            fh.ref_frame_sign_bias[ref_frame] = if !seq.enable_order_hint {
                false
            } else {
                get_relative_dist(seq, hint, fh.order_hint) > 0
            };
        }

        fh.disable_frame_end_update_cdf =
            if seq.reduced_still_picture_header || fh.disable_cdf_update {
                true
            } else {
                r.bit()?
            };

        // ── 参数段（M2.5 / M2.6，规范 §5.9.2 的续写）────────────────────────
        let params = FrameParams::parse(&mut r, &fh, seq, store)?;
        fh.params = params;

        // 收尾（规范 §5.9.1）：`OBU_FRAME_HEADER` 是 `trailing_bits()`、
        // `OBU_FRAME` 是 `byte_alignment()` —— 两者都要求"填充位为 0"，
        // 但前者还额外要求"载荷到此结束"（多出的位会让 trailing 检查报错）。
        // 这是整段非压缩头最有效的一道位偏移校验：真素材 5.5 万帧全过 ⇒
        // 每一层的位长都没有系统性偏差。
        Self::parse_tail(&mut r, tail)?;
        fh.bits_consumed = r.bits_read();
        fh.payload_bits_after_header = r.remaining_bits();
        Ok(fh)
    }

    /// 非压缩头之后的收尾（见 [`HeaderTail`]）。
    fn parse_tail(r: &mut BitReader<'_>, tail: HeaderTail) -> Result<()> {
        match tail {
            HeaderTail::TrailingBits => r.trailing_bits(),
            HeaderTail::ByteAlignment => r.byte_align(),
        }
    }

    /// 从参考帧槽复制尺寸（`show_existing_frame` 那条路：这一帧没有新的尺寸信息）。
    fn adopt_ref_size(fh: &mut FrameHeader, store: &RefFrameStore, slot: u8) {
        let i = (slot as usize).min(NUM_REF_FRAMES - 1);
        fh.upscaled_width = store.upscaled_width[i];
        fh.frame_width = store.upscaled_width[i];
        fh.frame_height = store.frame_height[i];
        fh.render_width = store.render_width[i];
        fh.render_height = store.render_height[i];
        fh.superres_denom = SUPERRES_NUM;
        fh.mi_cols = 2 * ((fh.frame_width + 7) >> 3);
        fh.mi_rows = 2 * ((fh.frame_height + 7) >> 3);
    }

    /// `frame_size()`（§6.5.6）+ `superres_params()`（§6.5.9）+ `compute_image_size()`（§6.5.10）。
    fn parse_frame_size(
        r: &mut BitReader<'_>,
        seq: &SequenceHeader,
        fh: &mut FrameHeader,
    ) -> Result<()> {
        if fh.frame_size_override {
            let width_minus_1 = r.f(seq.frame_width_bits as u32)?;
            let height_minus_1 = r.f(seq.frame_height_bits as u32)?;
            fh.upscaled_width = width_minus_1 + 1;
            fh.frame_height = height_minus_1 + 1;
        } else {
            fh.upscaled_width = seq.max_frame_width;
            fh.frame_height = seq.max_frame_height;
        }
        Self::parse_superres_params(r, seq, fh)?;
        fh.mi_cols = 2 * ((fh.frame_width + 7) >> 3);
        fh.mi_rows = 2 * ((fh.frame_height + 7) >> 3);
        Ok(())
    }

    /// `frame_size_with_refs()`（§6.5.8）：尺寸可能整段继承某个参考帧。
    fn parse_frame_size_with_refs(
        r: &mut BitReader<'_>,
        seq: &SequenceHeader,
        store: &RefFrameStore,
        fh: &mut FrameHeader,
    ) -> Result<()> {
        let mut found = false;
        for i in 0..REFS_PER_FRAME {
            let found_ref = r.bit()?;
            if found_ref {
                let slot = (fh.ref_frame_idx[i] as usize).min(NUM_REF_FRAMES - 1);
                fh.upscaled_width = store.upscaled_width[slot];
                fh.frame_width = store.upscaled_width[slot];
                fh.frame_height = store.frame_height[slot];
                fh.render_width = store.render_width[slot];
                fh.render_height = store.render_height[slot];
                fh.size_from_ref = true;
                found = true;
                break;
            }
        }
        if found {
            // 规范：继承参考帧尺寸时**不再读** render_size，只补算超分与模式网格
            Self::parse_superres_params(r, seq, fh)?;
            fh.mi_cols = 2 * ((fh.frame_width + 7) >> 3);
            fh.mi_rows = 2 * ((fh.frame_height + 7) >> 3);
        } else {
            Self::parse_frame_size(r, seq, fh)?;
            Self::parse_render_size(r, fh)?;
        }
        Ok(())
    }

    /// `superres_params()`（§6.5.9）。
    fn parse_superres_params(
        r: &mut BitReader<'_>,
        seq: &SequenceHeader,
        fh: &mut FrameHeader,
    ) -> Result<()> {
        fh.use_superres = if seq.enable_superres { r.bit()? } else { false };
        fh.superres_denom = if fh.use_superres {
            r.f(SUPERRES_DENOM_BITS)? + SUPERRES_DENOM_MIN
        } else {
            SUPERRES_NUM
        };
        let upscaled = fh.upscaled_width;
        fh.frame_width = (upscaled * SUPERRES_NUM + (fh.superres_denom / 2)) / fh.superres_denom;
        Ok(())
    }

    /// `render_size()`（§6.5.7）。
    fn parse_render_size(r: &mut BitReader<'_>, fh: &mut FrameHeader) -> Result<()> {
        let different = r.bit()?;
        if different {
            fh.render_width = r.f(16)? + 1;
            fh.render_height = r.f(16)? + 1;
        } else {
            fh.render_width = fh.upscaled_width;
            fh.render_height = fh.frame_height;
        }
        Ok(())
    }

    /// `read_interpolation_filter()`（§6.5.11）。
    fn parse_interpolation_filter(r: &mut BitReader<'_>) -> Result<u8> {
        let switchable = r.bit()?;
        if switchable {
            // SWITCHABLE 的取值在规范里是 4（滤波在块级决定）
            Ok(4)
        } else {
            Ok(r.f(2)? as u8)
        }
    }

    /// `temporal_point_info()`（§5.9.2 / §6.5.29）：显示时刻，只影响输出时机。
    fn parse_temporal_point_info(r: &mut BitReader<'_>, seq: &SequenceHeader) -> Result<()> {
        let n = seq
            .decoder_model_info
            .map(|d| d.frame_presentation_time_length_minus_1 as u32 + 1)
            .unwrap_or(1);
        let _frame_presentation_time = r.f(n)?;
        Ok(())
    }

    /// 帧头里的 `buffer_removal_time` 循环（§5.9.2）：每个 operating point 各一位域。
    ///
    /// 这一段平时不出现（要 `decoder_model_info_present`），但**必须把位读对** ——
    /// 少读一位后面全错。所以按规范把"是否属于本包的时空层"也照抄了。
    fn parse_buffer_removal_times(
        r: &mut BitReader<'_>,
        seq: &SequenceHeader,
        obu: ObuLayerIds,
    ) -> Result<()> {
        let n = seq
            .decoder_model_info
            .map(|d| d.buffer_removal_time_length_minus_1 as u32 + 1)
            .unwrap_or(1);
        for op in &seq.operating_points {
            if !op.decoder_model_present {
                continue;
            }
            let op_idc = op.idc as u32;
            let in_temporal = (op_idc >> obu.temporal_id) & 1;
            let in_spatial = (op_idc >> (obu.spatial_id + 8)) & 1;
            if op_idc == 0 || (in_temporal == 1 && in_spatial == 1) {
                let _buffer_removal_time = r.f(n)?;
            }
        }
        Ok(())
    }
}

/// 参考帧状态（非压缩头用得上的那几个字段 + 跨帧继承的参数）。
///
/// 为什么现在就要它：帧头有三处"必须知道之前解过什么"（继承尺寸 /
/// `show_existing_frame` 的帧类型 / `set_frame_refs` 的排序 / `skip_mode` 的
/// `RefOrderHint`），没有它连位都读不对。另外"没写就沿用上一帧"的那几组参数
/// （[`SavedFrameState`]）也挂在这里 —— 位读不依赖它们，但值对不上以后
/// 滤波与运动补偿就会错。像素不在这里 —— 那是 M6（帧存储）的事。
#[derive(Debug, Clone)]
pub struct RefFrameStore {
    pub valid: [bool; NUM_REF_FRAMES],
    pub order_hint: [u8; NUM_REF_FRAMES],
    pub frame_type: [FrameType; NUM_REF_FRAMES],
    pub upscaled_width: [u32; NUM_REF_FRAMES],
    pub frame_height: [u32; NUM_REF_FRAMES],
    pub render_width: [u32; NUM_REF_FRAMES],
    pub render_height: [u32; NUM_REF_FRAMES],
    /// 每个槽的"可从上一帧继承的参数"（`SavedGmParams` / `SavedLoopFilterParams` /
    /// `SavedSegmentationParams` 的合并体，规范 §7.20）
    pub saved: Box<[SavedFrameState; NUM_REF_FRAMES]>,
    /// 每个槽的**像素**（M6 帧存储；规范 §7.20 的 `FrameStore[i]`）。
    /// 存的是**上采样之后**的帧（`LrFrame`，亮度尺寸 = `UpscaledWidth`）。
    /// `None` = 这个槽没有像素（纯语法验收路径就是这样 —— 不重建像素也不存帧）。
    ///
    /// ⚠️ 是 `Arc<CurrFrame>` 而**不是** `CurrFrame`（2026-10-04 改）：一帧 1080p
    /// 约 6.2 MB，而 `refresh_frame_flags` 经常一次刷两个槽（KEY + ARF）——
    /// 按值存就是每帧 6~12 MB 的 `memmove`，加每份新缓冲的缺页。
    /// 参考帧是**只读**的（预测/OBMC/warp 都只读），所以共享一份即可，零拷贝。
    pub pixels: Box<[Option<std::sync::Arc<CurrFrame>>; NUM_REF_FRAMES]>,
    /// 每个槽的 CDF 上下文（规范 §7.20：`save_cdfs(i)` / `load_cdfs(ctx)`，
    /// 下标就是参考槽号）。
    pub cdfs: Box<[CdfContext; NUM_REF_FRAMES]>,
    /// 每个槽保存的胶片颗粒参数（§5.9.30 的 `load_grain_params(ref_idx)` 从这里取）。
    /// 参考帧本身存的是**未加颗粒**的像素，颗粒参数随帧头一起存槽。
    pub grain: Box<[FilmGrainParams; NUM_REF_FRAMES]>,
}

impl Default for RefFrameStore {
    fn default() -> Self {
        Self {
            valid: [false; NUM_REF_FRAMES],
            order_hint: [0; NUM_REF_FRAMES],
            frame_type: [FrameType::Key; NUM_REF_FRAMES],
            upscaled_width: [0; NUM_REF_FRAMES],
            frame_height: [0; NUM_REF_FRAMES],
            render_width: [0; NUM_REF_FRAMES],
            render_height: [0; NUM_REF_FRAMES],
            saved: Box::new(std::array::from_fn(|_| SavedFrameState::past_independence())),
            pixels: Box::new(std::array::from_fn(|_| None)),
            // 默认上下文（base_q = 0 档）—— 真正的帧级初始化在驱动里按
            // `primary_ref_frame` 决定"用默认还是 load_cdfs"。
            cdfs: Box::new(std::array::from_fn(|_| CdfContext::defaults(0))),
            grain: Box::new(std::array::from_fn(|_| FilmGrainParams::default())),
        }
    }
}

impl RefFrameStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// 某个槽的帧类型（`show_existing_frame` 要用）。
    pub fn frame_type(&self, slot: u8) -> FrameType {
        self.frame_type[(slot as usize).min(NUM_REF_FRAMES - 1)]
    }

    /// 关键帧重置：所有参考帧失效（规范在"关键帧 + 显示"时清 `RefValid`/`RefOrderHint`）。
    pub fn reset(&mut self) {
        self.valid = [false; NUM_REF_FRAMES];
        self.order_hint = [0; NUM_REF_FRAMES];
        // 参考帧作废后其颗粒参数也不该再被 `load_grain_params` 取到
        for g in self.grain.iter_mut() {
            *g = FilmGrainParams::default();
        }
    }

    /// "参考帧更新过程"（§7.20）：把这一帧存进 `refresh_frame_flags` 指定的槽。
    ///
    /// 调用时机：**这一帧解析完之后**（`frame_size_with_refs` 读的是"这一帧之前"的
    /// 参考帧尺寸，所以不能提前刷新）。
    ///
    /// `show_existing_frame` 的帧只做"显示"，不产生新的参数 —— 它刷新的槽里本来就
    /// 是那一帧的状态，所以这里不覆盖 `saved`（规范里 `load_grain_params` 那类
    /// 也是从槽里读）。
    pub fn refresh(&mut self, fh: &FrameHeader) {
        if fh.show_existing_frame && fh.refresh_frame_flags == 0 {
            return;
        }
        for i in 0..NUM_REF_FRAMES {
            if (fh.refresh_frame_flags >> i) & 1 == 0 {
                continue;
            }
            self.valid[i] = true;
            self.order_hint[i] = fh.order_hint;
            self.frame_type[i] = fh.frame_type;
            self.upscaled_width[i] = fh.upscaled_width;
            self.frame_height[i] = fh.frame_height;
            self.render_width[i] = fh.render_width;
            self.render_height[i] = fh.render_height;
            if !fh.show_existing_frame {
                let st = &mut self.saved[i];
                st.lf_ref_deltas = fh.params.loop_filter.ref_deltas;
                st.lf_mode_deltas = fh.params.loop_filter.mode_deltas;
                st.seg_feature_enabled = fh.params.segmentation.feature_enabled;
                st.seg_feature_data = fh.params.segmentation.feature_data;
                st.gm_params = fh.params.global_motion.params;
                // 颗粒参数随帧头一起存槽（`load_grain_params` 从这里取）
                self.grain[i] = fh.params.film_grain.clone();
            }
        }
    }
}

impl RefFrameStore {
    /// 把这一帧的**像素**存进 `refresh_frame_flags` 指定的槽
    /// （§7.20 的 `FrameStore[i] = LrFrame`；像素要先经环路滤波与超分上采样）。
    /// **按引用共享**存进被刷新的每个槽（零拷贝，见 [`RefFrameStore::pixels`] 的说明）。
    pub fn store_pixels(&mut self, refresh_flags: u8, frame: std::sync::Arc<CurrFrame>) {
        for i in 0..NUM_REF_FRAMES {
            if (refresh_flags >> i) & 1 == 1 {
                self.pixels[i] = Some(std::sync::Arc::clone(&frame));
            }
        }
    }

    /// `save_cdfs(i)`（§7.20）：把当前帧的 CDF 上下文存进被刷新的每个槽。
    /// 存进去的是**计数器已清零**的快照（与规范在 `load_cdfs` 时重置等价）。
    pub fn save_cdfs(&mut self, refresh_flags: u8, ctx: &CdfContext) {
        let snap = ctx.snapshot();
        for i in 0..NUM_REF_FRAMES {
            if (refresh_flags >> i) & 1 == 1 {
                self.cdfs[i] = snap.clone();
            }
        }
    }

    /// `load_cdfs(ref_frame_idx[primary_ref_frame])`（§7.20）：整份拷出。
    pub fn load_cdfs(&self, slot: u8) -> CdfContext {
        self.cdfs[(slot as usize).min(NUM_REF_FRAMES - 1)].clone()
    }
}

/// `get_relative_dist(a, b)`（§6.5.3）：把 order hint 的差做符号扩展。
pub fn get_relative_dist(seq: &SequenceHeader, a: u8, b: u8) -> i32 {
    if !seq.enable_order_hint {
        return 0;
    }
    let diff = a as i32 - b as i32;
    let m = 1i32 << (seq.order_hint_bits as i32 - 1);
    (diff & (m - 1)) - (diff & m)
}

/// `set_frame_refs()`（§7.8）：`frame_refs_short_signaling` 时**不读位**，
/// 由 `last_frame_idx` / `gold_frame_idx` 与各参考槽的 order hint 现算 `ref_frame_idx[]`。
///
/// 为什么必须实现它：这条路一个位都不读，算错了 `ref_frame_idx` 就会把
/// `frame_size_with_refs` 的尺寸继承接到错的槽上（症状是尺寸莫名其妙地变），
/// 而且**报不出错**。
pub fn set_frame_refs(
    store: &RefFrameStore,
    seq: &SequenceHeader,
    order_hint: u8,
    last_frame_idx: u8,
    gold_frame_idx: u8,
) -> [u8; REFS_PER_FRAME] {
    // 参考帧类型在 ref_frame_idx 里的顺序：LAST(0) LAST2(1) LAST3(2) GOLDEN(3)
    // BWDREF(4) ALTREF2(5) ALTREF(6)
    const LAST: usize = 0;
    const LAST2: usize = 1;
    const LAST3: usize = 2;
    const GOLDEN: usize = 3;
    const BWDREF: usize = 4;
    const ALTREF2: usize = 5;
    const ALTREF: usize = 6;

    let mut idx = [0u8; REFS_PER_FRAME];
    let mut assigned = [false; REFS_PER_FRAME];
    let mut used = [false; NUM_REF_FRAMES];
    idx[LAST] = last_frame_idx;
    assigned[LAST] = true;
    idx[GOLDEN] = gold_frame_idx;
    assigned[GOLDEN] = true;
    used[last_frame_idx as usize] = true;
    used[gold_frame_idx as usize] = true;

    // 以"当前帧的 order hint"为基准把各槽的期望输出顺序平移过来
    let cur_frame_hint = 1i32 << (seq.order_hint_bits as i32 - 1);
    let shifted: Vec<i32> = (0..NUM_REF_FRAMES)
        .map(|i| cur_frame_hint + get_relative_dist(seq, store.order_hint[i], order_hint))
        .collect();

    // 在"未使用"的槽里找：hint >= 基准 的是后向参考（取最晚 / 最早），hint < 基准 的是前向参考
    let find_backward = |latest: bool, used: &[bool; NUM_REF_FRAMES]| -> Option<usize> {
        let mut best: Option<usize> = None;
        let mut best_hint = 0i32;
        for i in 0..NUM_REF_FRAMES {
            let hint = shifted[i];
            if used[i] || hint < cur_frame_hint {
                continue;
            }
            if best.is_none()
                || (latest && hint >= best_hint)
                || (!latest && hint < best_hint)
            {
                best = Some(i);
                best_hint = hint;
            }
        }
        best
    };
    let find_forward = |used: &[bool; NUM_REF_FRAMES]| -> Option<usize> {
        let mut best: Option<usize> = None;
        let mut best_hint = 0i32;
        for i in 0..NUM_REF_FRAMES {
            let hint = shifted[i];
            if used[i] || hint >= cur_frame_hint {
                continue;
            }
            if best.is_none() || hint >= best_hint {
                best = Some(i);
                best_hint = hint;
            }
        }
        best
    };

    if let Some(slot) = find_backward(true, &used) {
        idx[ALTREF] = slot as u8;
        assigned[ALTREF] = true;
        used[slot] = true;
    }
    if let Some(slot) = find_backward(false, &used) {
        idx[BWDREF] = slot as u8;
        assigned[BWDREF] = true;
        used[slot] = true;
    }
    if let Some(slot) = find_backward(false, &used) {
        idx[ALTREF2] = slot as u8;
        assigned[ALTREF2] = true;
        used[slot] = true;
    }
    // 剩下的按 Ref_Frame_List 的顺序填前向参考（最近优先）
    for slot_idx in [LAST2, LAST3, BWDREF, ALTREF2, ALTREF] {
        if assigned[slot_idx] {
            continue;
        }
        if let Some(slot) = find_forward(&used) {
            idx[slot_idx] = slot as u8;
            assigned[slot_idx] = true;
            used[slot] = true;
        }
    }
    // 还有没填上的：用"输出顺序最小"的那个槽兜底（规范 §7.8 最后一段）
    let mut earliest: Option<usize> = None;
    let mut earliest_hint = 0i32;
    for i in 0..NUM_REF_FRAMES {
        if earliest.is_none() || shifted[i] < earliest_hint {
            earliest = Some(i);
            earliest_hint = shifted[i];
        }
    }
    let fallback = earliest.unwrap_or(0) as u8;
    for i in 0..REFS_PER_FRAME {
        if !assigned[i] {
            idx[i] = fallback;
        }
    }
    idx
}

/// 解析失败时的统一文案（把"读到哪一位"带出来，便于对着码流排查）。
pub fn frame_header_error(context: &str, e: MediaError, bits: usize) -> MediaError {
    MediaError::Bitstream(format!("{context}（已读 {bits} 位）：{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::testbits::{write_min_params, write_reduced_still_picture_1080p, BitWriter};
    use crate::video::params::{TX_MODE_LARGEST, TX_MODE_SELECT};

    /// 造一个"普通 1080p 视频"的序列头（与 `sequence_header` 的测试同源），
    /// 让帧头的测试有真实的 `seq` 可用。
    fn seq_1080p() -> SequenceHeader {
        // 简化静图模式的序列头已经由 testbits 提供了；这里直接拿它，
        // 但**把它当普通序列头用**是不对的（工具开关不同），所以另造一个最小的普通头。
        let mut w = BitWriter::default();
        w.f(0, 3); // profile 0
        w.bit(false); // still_picture
        w.bit(false); // reduced_still_picture_header
        w.bit(false); // timing_info_present_flag
        w.bit(false); // initial_display_delay_present_flag
        w.f(0, 5); // operating_points_cnt_minus_1
        w.f(0, 12); // operating_point_idc
        w.f(4, 5); // seq_level_idx = 4（≤7，不带 tier）
        w.f(10, 4); // frame_width_bits_minus_1 → 11 位
        w.f(10, 4); // frame_height_bits_minus_1 → 11 位
        w.f(1919, 11); // max_frame_width_minus_1
        w.f(1079, 11); // max_frame_height_minus_1
        w.bit(false); // frame_id_numbers_present_flag
        w.bit(false); // use_128x128_superblock
        w.bit(false); // enable_filter_intra
        w.bit(false); // enable_intra_edge_filter
        w.bit(true); // enable_interintra_compound
        w.bit(true); // enable_masked_compound
        w.bit(true); // enable_warped_motion
        w.bit(true); // enable_dual_filter
        w.bit(true); // enable_order_hint
        w.bit(true); // enable_jnt_comp
        w.bit(true); // enable_ref_frame_mvs
        w.bit(true); // seq_choose_screen_content_tools → SELECT（不读 force 位）
        w.bit(true); // seq_choose_integer_mv → SELECT
        w.f(6, 3); // order_hint_bits_minus_1 = 6 → 7 位
        w.bit(false); // enable_superres
        w.bit(true); // enable_cdef
        w.bit(true); // enable_restoration
        w.bit(false); // high_bitdepth → 8 位
        w.bit(false); // mono_chrome
        w.bit(false); // color_description_present_flag
        w.bit(false); // color_range
        w.f(0, 2); // chroma_sample_position
        w.bit(false); // separate_uv_delta_q
        w.bit(false); // film_grain_params_present
        w.trailing();
        SequenceHeader::parse(w.finish()).unwrap()
    }

    /// 造一帧**关键帧**的非压缩头（`show_frame = 1`，尺寸沿用序列头的最大值），
    /// 含参数段并补到字节边界 —— 也就是"一整段可被 `byte_alignment` 校验的头"。
    fn keyframe_bits(w: &mut BitWriter, seq: &SequenceHeader) {
        w.bit(false); // show_existing_frame = 0
        w.f(0, 2); // frame_type = KEY_FRAME
        w.bit(true); // show_frame
        // showable_frame 由推导得出（KEY_FRAME → 0），不读位
        // error_resilient_mode 由推导得出（KEY + show → 1），不读位
        w.bit(false); // disable_cdf_update
        w.bit(false); // allow_screen_content_tools
        // force_integer_mv：帧内帧强制 1，不读位
        w.bit(false); // frame_size_override_flag
        w.f(0, 7); // order_hint（7 位）
        // primary_ref_frame：帧内 → PRIMARY_REF_NONE，不读位
        // refresh_frame_flags：KEY+show → allFrames，不读位
        // 关键帧 + 显示：规格里那段"清参考帧"不读位
        w.bit(false); // render_and_frame_size_different
        // allow_intrabc：allow_screen_content_tools = 0 → 不读
        // disable_frame_end_update_cdf：disable_cdf_update = 0 → 读一位
        w.bit(false);
        write_min_params(w, seq, true);
        w.byte_align();
    }

    #[test]
    fn parses_a_key_frame() {
        let seq = seq_1080p();
        let mut store = RefFrameStore::new();
        let mut w = BitWriter::default();
        keyframe_bits(&mut w, &seq);
        let bits = w.finish().to_vec();
        let fh = FrameHeader::parse(&bits, &seq, &mut store, ObuLayerIds::default(), HeaderTail::ByteAlignment).unwrap();
        assert_eq!(fh.frame_type, FrameType::Key);
        assert!(fh.is_intra);
        assert!(fh.show_frame);
        assert!(!fh.showable_frame, "关键帧的 showable_frame 推导为 0");
        assert!(fh.error_resilient_mode, "KEY + show → 强制错误弹性");
        assert_eq!(fh.refresh_frame_flags, ALL_FRAMES, "关键帧刷新全部参考槽");
        assert_eq!(fh.primary_ref_frame, PRIMARY_REF_NONE);
        assert!(fh.force_integer_mv, "帧内帧强制整数运动矢量");
        assert_eq!(fh.upscaled_width, 1920, "没有 frame_size_override → 用序列头的最大值");
        assert_eq!(fh.frame_height, 1080);
        assert_eq!(fh.render_width, 1920);
        assert_eq!(fh.mi_cols, 2 * ((1920 + 7) >> 3));
        assert!(!fh.size_from_ref);
        assert!(fh.bits_consumed > 0);
        // 这一段合成头（M2 的字段 + 参数段 + 对齐）应当在一百来位内
        assert!(fh.bits_consumed <= 160, "读多了：{} 位", fh.bits_consumed);
        assert_eq!(fh.bits_consumed % 8, 0, "读到末尾必须是字节对齐的");
        // 参数段的关键结论
        let p = &fh.params;
        assert_eq!(p.tile_info.cols, 1);
        assert_eq!(p.tile_info.rows, 1);
        assert_eq!(p.tile_info.num_tiles(), 1);
        assert_eq!(p.tile_info.tile_size_bytes, 0, "单瓦片没有尺寸字段");
        assert_eq!(p.quant.base_q_idx, 100);
        assert!(!p.segmentation.enabled);
        assert!(!p.coded_lossless, "base_q_idx = 100 → 不是无损");
        assert_eq!(p.loop_filter.level, [0, 0, 0, 0]);
        assert_eq!(p.cdef.damping, 3);
        assert!(!p.lr.uses_lr);
        assert_eq!(p.tx_mode, TX_MODE_LARGEST, "tx_mode_select = 0 → TX_MODE_LARGEST");
        assert_eq!(p.global_motion.gm_type[1], 0, "帧内帧没有全局运动");
        assert!(!p.film_grain.apply_grain, "序列头没开胶片颗粒");
    }

    #[test]
    fn a_bit_slip_is_caught_by_byte_alignment() {
        // 参数段末尾**多写一个值为 1 的位**（等价于"某一层少读了一位"）：
        // 解析器读完参数段后，那个 1 就落进了 `byte_alignment()` 的填充位里，
        // 于是对齐检查必须报错。这就是"整段非压缩头"的位序保险。
        let seq = seq_1080p();
        let mut store = RefFrameStore::new();
        let mut w = BitWriter::default();
        w.bit(false); // show_existing_frame
        w.f(0, 2); // KEY_FRAME
        w.bit(true); // show_frame
        w.bit(false); // disable_cdf_update
        w.bit(false); // allow_screen_content_tools
        w.bit(false); // frame_size_override
        w.f(0, 7); // order_hint
        w.bit(false); // render_and_frame_size_different
        w.bit(false); // disable_frame_end_update_cdf
        write_min_params(&mut w, &seq, true);
        w.bit(true); // ← 多余的位（真实码流里不该有）
        w.byte_align();
        let bits = w.finish().to_vec();
        let e = FrameHeader::parse(&bits, &seq, &mut store, ObuLayerIds::default(), HeaderTail::ByteAlignment);
        assert!(e.is_err(), "多出来的那一位必须被 byte_alignment 抓住");
        let text = format!("{}", e.unwrap_err());
        assert!(text.contains("字节对齐位"), "报错要指出是对齐位：{text}");

        // 对照：不多那一位就一切正常
        let mut w = BitWriter::default();
        w.bit(false);
        w.f(0, 2);
        w.bit(true);
        w.bit(false);
        w.bit(false);
        w.bit(false);
        w.f(0, 7);
        w.bit(false);
        w.bit(false);
        write_min_params(&mut w, &seq, true);
        w.byte_align();
        let mut store = RefFrameStore::new();
        let ok = FrameHeader::parse(&w.finish().to_vec(), &seq, &mut store, ObuLayerIds::default(), HeaderTail::ByteAlignment);
        assert!(ok.is_ok(), "同样的头去掉那一位就该通过：{ok:?}");
    }

    #[test]
    fn parses_an_inter_frame_and_inherits_size_from_refs() {
        let seq = seq_1080p();
        let mut store = RefFrameStore::new();
        // 先来一个关键帧把参考槽填上（所有槽都变成 1920×1080）
        let mut w = BitWriter::default();
        keyframe_bits(&mut w, &seq);
        let kf = FrameHeader::parse(&w.finish().to_vec(), &seq, &mut store, ObuLayerIds::default(), HeaderTail::ByteAlignment)
            .unwrap();
        store.refresh(&kf);

        let mut w = BitWriter::default();
        w.bit(false); // show_existing_frame
        w.f(1, 2); // frame_type = INTER_FRAME
        w.bit(true); // show_frame
        // showable_frame 推导（INTER → 1）
        w.bit(false); // error_resilient_mode
        w.bit(false); // disable_cdf_update
        w.bit(true); // allow_screen_content_tools（开了才会去读 force_integer_mv）
        w.bit(false); // force_integer_mv（非帧内才读；1 会让 allow_high_precision_mv 不再读）
        w.bit(true); // frame_size_override_flag = 1
        w.f(1, 7); // order_hint = 1
        w.f(0, 3); // primary_ref_frame = 0
        w.f(0x01, 8); // refresh_frame_flags（只刷新槽 0）
        // frame_refs_short_signaling = 0 → 逐个读 7 个 ref_frame_idx
        w.bit(false); // frame_refs_short_signaling
        for i in 0..REFS_PER_FRAME as u8 {
            w.f(i as u32 % 8, 3); // ref_frame_idx[i] = i
        }
        // frame_size_override && !error_resilient → frame_size_with_refs
        w.bit(true); // found_ref（第一个参考帧就把尺寸继承了）
        // 继承尺寸 → 不读 render_size；enable_superres = 0 → 不读 superres 位
        w.bit(true); // allow_high_precision_mv（force_integer_mv = 0 才读）
        w.bit(true); // interpolation_filter: is_filter_switchable = 1 → SWITCHABLE
        w.bit(true); // is_motion_mode_switchable
        w.bit(false); // use_ref_frame_mvs
        w.bit(false); // disable_frame_end_update_cdf
        write_min_params(&mut w, &seq, false);
        w.byte_align();
        let fh = FrameHeader::parse(&w.finish().to_vec(), &seq, &mut store, ObuLayerIds::default(), HeaderTail::ByteAlignment)
            .unwrap();
        assert_eq!(fh.frame_type, FrameType::Inter);
        assert!(!fh.is_intra);
        assert!(!fh.error_resilient_mode);
        assert_eq!(fh.order_hint, 1);
        assert_eq!(fh.refresh_frame_flags, 0x01, "只有槽 0 被刷新");
        assert!(fh.size_from_ref, "尺寸应当从参考帧继承");
        assert_eq!(fh.upscaled_width, 1920);
        assert_eq!(fh.frame_height, 1080);
        assert_eq!(fh.render_width, 1920, "继承来的 render 尺寸");
        assert!(fh.allow_high_precision_mv);
        assert_eq!(fh.interpolation_filter, 4, "SWITCHABLE");
        assert!(fh.is_motion_mode_switchable);
        assert!(!fh.use_ref_frame_mvs);
        // 帧间帧的参数段：reference_select = 0 → skip 模式不参与；全局运动 7 个都读成 identity
        assert!(!fh.params.reference_select);
        assert!(!fh.params.skip_mode_present);
        assert!(fh.params.global_motion.gm_type[1..].iter().all(|&t| t == 0));
        assert_eq!(fh.params.tx_mode, TX_MODE_LARGEST);
        assert_ne!(fh.params.tx_mode, TX_MODE_SELECT, "这里写的是 tx_mode_select = 0");
        assert_eq!(fh.bits_consumed % 8, 0, "读到末尾必须字节对齐");
    }

    #[test]
    fn parses_show_existing_frame() {
        let seq = seq_1080p();
        let mut store = RefFrameStore::new();
        // 先解码一个关键帧（填满参考槽）
        let mut w = BitWriter::default();
        keyframe_bits(&mut w, &seq);
        let kf = FrameHeader::parse(
            &w.finish().to_vec(),
            &seq,
            &mut store,
            ObuLayerIds::default(),
            HeaderTail::ByteAlignment,
        )
        .unwrap();
        store.refresh(&kf);

        // `show_existing_frame` 的帧在真实码流里走 `OBU_FRAME_HEADER`，
        // 所以收尾是 `trailing_bits()`（一个 `1` + 补 `0`）
        let mut w = BitWriter::default();
        w.bit(true); // show_existing_frame
        w.f(3, 3); // frame_to_show_map_idx = 3
        w.f(1, 1); // trailing_bits
        let bits = w.finish().to_vec();
        let fh = FrameHeader::parse(
            &bits,
            &seq,
            &mut store,
            ObuLayerIds::default(),
            HeaderTail::TrailingBits,
        )
        .unwrap();
        assert!(fh.show_existing_frame);
        assert_eq!(fh.frame_to_show_map_idx, 3);
        assert_eq!(fh.frame_type, FrameType::Key, "槽里存的是关键帧");
        assert_eq!(fh.refresh_frame_flags, ALL_FRAMES, "显示的关键帧 → 刷新全部槽");
        assert_eq!(fh.upscaled_width, 1920, "尺寸从槽里取");
        assert_eq!(fh.bits_consumed, 8, "4 位内容 + trailing_bits（1 + 3 个 0）");
        assert_eq!(fh.payload_bits_after_header, 0, "载荷到 trailing 为止恰好用完");
        assert_eq!(fh.params.tile_info.num_tiles(), 1, "这条路没有参数段，用默认值");
    }

    /// 真实素材的教训：`OBU_FRAME_HEADER` 与 `OBU_FRAME` 的收尾**不一样**。
    ///
    /// B 站 1080p 那份素材里 `show_existing_frame` 的载荷是 1 字节 `0xc8`
    /// = `1`（show_existing）+ `100`（槽 4）+ `1000`（`trailing_bits`）。
    /// 把收尾一律当成 `byte_alignment`（要求填充位全 0）会在这里报错。
    #[test]
    fn obu_frame_header_uses_trailing_bits_not_byte_alignment() {
        let seq = seq_1080p();
        let mut store = RefFrameStore::new();
        let mut w = BitWriter::default();
        keyframe_bits(&mut w, &seq);
        let kf = FrameHeader::parse(
            &w.finish().to_vec(),
            &seq,
            &mut store,
            ObuLayerIds::default(),
            HeaderTail::ByteAlignment,
        )
        .unwrap();
        store.refresh(&kf);

        let payload = [0xc8u8];
        // OBU_FRAME_HEADER：trailing_bits → 通过
        let fh = FrameHeader::parse(
            &payload,
            &seq,
            &mut store,
            ObuLayerIds::default(),
            HeaderTail::TrailingBits,
        )
        .expect("真实素材里这种载荷是合法的 OBU_FRAME_HEADER");
        assert!(fh.show_existing_frame);
        assert_eq!(fh.frame_to_show_map_idx, 4, "中间三位就是槽号");
        assert_eq!(fh.bits_consumed, 8);
        assert_eq!(fh.payload_bits_after_header, 0);
        // 同一段载荷按 OBU_FRAME 读（要求 byte_alignment）→ 必须报错，
        // 因为最后四位里的那个 1 不是合法的填充位
        let e = FrameHeader::parse(
            &payload,
            &seq,
            &mut store,
            ObuLayerIds::default(),
            HeaderTail::ByteAlignment,
        );
        assert!(e.is_err(), "OBU_FRAME 的收尾是 byte_alignment，不该接受这段载荷");
    }

    #[test]
    fn parses_reduced_still_picture_frame_header() {
        // 简化静图模式：身份与显示行为全部写死，连 frame_type 都不读
        let mut sw = BitWriter::default();
        write_reduced_still_picture_1080p(&mut sw);
        let seq = SequenceHeader::parse(sw.finish()).unwrap();
        let mut store = RefFrameStore::new();
        let mut w = BitWriter::default();
        w.bit(false); // disable_cdf_update
        w.bit(false); // allow_screen_content_tools
        w.bit(false); // render_and_frame_size_different
        // disable_frame_end_update_cdf：简化模式强制 1，**不读位**
        write_min_params(&mut w, &seq, true);
        w.byte_align();
        let fh = FrameHeader::parse(&w.finish().to_vec(), &seq, &mut store, ObuLayerIds::default(), HeaderTail::ByteAlignment)
            .unwrap();
        assert_eq!(fh.frame_type, FrameType::Key);
        assert!(fh.show_frame);
        assert!(!fh.show_existing_frame);
        assert_eq!(fh.upscaled_width, 1920);
        assert_eq!(fh.frame_height, 1080);
        assert!(fh.disable_frame_end_update_cdf, "简化模式强制为 1");
    }

    #[test]
    fn set_frame_refs_orders_refs_by_output_order() {
        let seq = seq_1080p();
        let mut store = RefFrameStore::new();
        // 造 8 个槽：order hint 分别是 100..107（比当前帧 110 都小 → 都是前向参考）
        for i in 0..NUM_REF_FRAMES {
            store.valid[i] = true;
            store.order_hint[i] = 100 + i as u8;
        }
        let idx = set_frame_refs(&store, &seq, 110, 7, 6);
        assert_eq!(idx[0], 7, "LAST 用显式给的 last_frame_idx");
        assert_eq!(idx[3], 6, "GOLDEN 用显式给的 gold_frame_idx");
        // 其余前向参考按"输出顺序最晚优先"填：107、105、104、103、102（107 已被 GOLDEN 占用）
        assert_eq!(idx[1], 5, "LAST2 = 剩下最晚的（hint 105）");
        assert_eq!(idx[2], 4, "LAST3 = 次晚（hint 104）");
        assert!(idx.iter().all(|&i| (i as usize) < NUM_REF_FRAMES));
        assert_ne!(idx[1], idx[3], "同一个槽不该被两个参考类型占用");
    }

    #[test]
    fn get_relative_dist_handles_wraparound() {
        let seq = seq_1080p();
        // order_hint_bits = 7 → 模 128 的有符号差
        assert_eq!(get_relative_dist(&seq, 10, 5), 5);
        assert_eq!(get_relative_dist(&seq, 5, 10), -5);
        assert_eq!(get_relative_dist(&seq, 0, 127), 1, "回绕：0 在 127 之后");
    }

    #[test]
    fn truncated_frame_header_is_an_error() {
        let seq = seq_1080p();
        let mut store = RefFrameStore::new();
        let mut w = BitWriter::default();
        keyframe_bits(&mut w, &seq);
        let full = w.finish().to_vec();
        // 只给 1 个字节：必须报错，不能把"读到 0"当合法值
        assert!(
            FrameHeader::parse(&full[..1], &seq, &mut store, ObuLayerIds::default(), HeaderTail::ByteAlignment).is_err(),
            "截断到 1 字节时应当报错"
        );
        assert!(full.len() > 8, "合成头远不止 1 个字节（含参数段）：{}", full.len());
    }

    #[test]
    fn refresh_only_touches_selected_slots() {
        let mut store = RefFrameStore::new();
        let mut fh = FrameHeader::default();
        fh.frame_type = FrameType::Inter;
        fh.order_hint = 42;
        fh.upscaled_width = 1280;
        fh.frame_height = 720;
        fh.refresh_frame_flags = 0b0000_0101; // 槽 0 与槽 2
        store.refresh(&fh);
        assert!(store.valid[0] && store.valid[2]);
        assert_eq!(store.order_hint[2], 42);
        assert_eq!(store.upscaled_width[0], 1280);
        assert!(!store.valid[1], "没被刷新的槽不该变");
        assert_eq!(store.frame_type[2], FrameType::Inter);

        // 关键帧重置：全失效
        store.reset();
        assert!(store.valid.iter().all(|v| !v));
    }
}
