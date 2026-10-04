//! AV1 **序列头**（规范 §5.5.1）+ 颜色配置（§5.5.2）+ 定时信息（§5.5.3/§5.5.4）。
//!
//! 序列头是整条流的全局契约：profile、位深、色度采样、最大尺寸、工具开关
//! （CDEF / 环路恢复 / 屏幕内容 / order hint……）。后面每一层都要读它：
//! 帧头要靠它决定"有没有 frame_id"，瓦片头要靠它决定超块大小（64 还是 128），
//! 重建像素要靠它决定位深与色度格式。
//!
//! 所以这一层是自研软解里**第一个"必须完全正确"的解析层**：读错一位，
//! 后面全盘皆错，而且错误现场离原因很远（典型的"解码器写对了但画面是雪花"）。
//! 对策与 Opus 那边一样：位域顺序严格照规范，且每个"跳过/省略"的分支都在
//! 注释里写明它的条件（省略条件写错的后果就是整体错位一个位）。

use me_core::{MediaError, Result};

use super::bitreader::BitReader;

/// `seq_force_screen_content_tools` / `seq_force_integer_mv` 的"交给帧头决定"哨兵值。
pub const SELECT_SCREEN_CONTENT_TOOLS: u8 = 2;
pub const SELECT_INTEGER_MV: u8 = 2;

/// 颜色配置（`color_config()`，规范 §5.5.2）。
///
/// 为什么把 `bit_depth` 与色度采样单独拎出来说明：**输出像素的形态由它们决定**
/// （8/10/12 位、4:2:0 / 4:4:4 / 单色），而引擎对外只出 **8 位 4:2:0 NV12** ——
/// 高位深与 4:4:4 的转换是"重建之后"的事，必须在解码前就知道，不能边解边猜。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorConfig {
    /// 8 / 10 / 12
    pub bit_depth: u8,
    pub mono_chrome: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    /// 色度采样位置（`subsampling_x && subsampling_y` 时才有意义）
    pub chroma_sample_position: u8,
    /// 0 = studio swing（有限范围），1 = full swing
    pub color_range: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    /// 色度分量是否分开写 delta q（影响 delta_q 的读取次数）
    pub separate_uv_delta_q: bool,
}

/// 一个 operating point（多码率分层用；单层码流通常只有一个）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperatingPoint {
    pub idc: u16,
    pub seq_level_idx: u8,
    pub seq_tier: u8,
    pub initial_display_delay_minus_1: u8,
    /// `decoder_model_present_for_this_op[i]`（帧头的 buffer_removal_time 循环要看它）
    pub decoder_model_present: bool,
}

/// `decoder_model_info()`（规范 §5.5.4）：解码器模型的缓冲/时钟长度字段。
///
/// 播放（按显示顺序输出）不需要它，但**帧头要用**：`buffer_removal_time` 的位宽
/// 与 `temporal_point_info` 的位宽分别取自这里的两个 `*_length_minus_1`。
/// 少存一个字段，读帧头时就会错位 —— 所以它在序列头里必须留下来。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderModelInfo {
    pub buffer_delay_length_minus_1: u8,
    pub num_units_in_decoding_tick: u32,
    pub buffer_removal_time_length_minus_1: u8,
    pub frame_presentation_time_length_minus_1: u8,
}

/// `timing_info()`（规范 §5.5.3）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimingInfo {
    pub num_units_in_display_tick: u32,
    pub time_scale: u32,
    /// 每帧的 tick 数恒定吗（恒定才读得到 `num_ticks_per_picture`）
    pub equal_picture_interval: bool,
    pub num_ticks_per_picture_minus_1: u32,
}

/// 解析后的序列头。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceHeader {
    pub seq_profile: u8,
    /// 整条流只有一张静图（AVIF 那类）
    pub still_picture: bool,
    /// 静图简化模式：工具开关全部取默认、没有 timing/operating point 循环
    pub reduced_still_picture_header: bool,
    /// `max_frame_width_minus_1` 的位数
    pub frame_width_bits: u8,
    pub frame_height_bits: u8,
    pub max_frame_width: u32,
    pub max_frame_height: u32,
    pub frame_id_numbers_present: bool,
    /// `delta_frame_id_length_minus_2`（`frame_id_numbers_present` 时才有意义）
    pub delta_frame_id_length_minus_2: u8,
    /// `additional_frame_id_length_minus_1`（同上）
    pub additional_frame_id_length_minus_1: u8,
    /// 超块是 128×128（否则 64×64）—— 瓦片层要靠它算列数
    pub use_128x128_superblock: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_warped_motion: bool,
    pub enable_dual_filter: bool,
    pub enable_order_hint: bool,
    pub enable_jnt_comp: bool,
    pub enable_ref_frame_mvs: bool,
    pub seq_force_screen_content_tools: u8,
    pub seq_force_integer_mv: u8,
    pub order_hint_bits: u8,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    pub color: ColorConfig,
    pub film_grain_params_present: bool,
    pub operating_points: Vec<OperatingPoint>,
    pub timing: Option<TimingInfo>,
    /// `decoder_model_info_present_flag`（帧头要靠它决定读不读 buffer_removal_time 等）
    pub decoder_model_info_present: bool,
    pub decoder_model_info: Option<DecoderModelInfo>,
    /// 帧头的 `frame_id_numbers` 用到的 `idLen`
    pub frame_id_length: u8,
}

impl SequenceHeader {
    /// 解析一个序列头 OBU 的**载荷**（不含 OBU 头/长度字段）。
    pub fn parse(payload: &[u8]) -> Result<Self> {
        let mut r = BitReader::new(payload);
        let seq_profile = r.f(3)? as u8;
        let still_picture = r.bit()?;
        let reduced_still_picture_header = r.bit()?;

        let mut timing = None;
        let mut operating_points = Vec::new();
        let mut decoder_model_info_present = false;
        let mut decoder_model_info = None;

        if reduced_still_picture_header {
            // 简化模式：没有 timing / decoder model / 显示延迟，工具开关全默认
            operating_points.push(OperatingPoint {
                idc: 0,
                seq_level_idx: r.f(5)? as u8,
                seq_tier: 0,
                initial_display_delay_minus_1: 0,
                decoder_model_present: false,
            });
        } else {
            let timing_info_present = r.bit()?;
            if timing_info_present {
                timing = Some(Self::parse_timing_info(&mut r)?);
                decoder_model_info_present = r.bit()?;
                if decoder_model_info_present {
                    decoder_model_info = Some(Self::parse_decoder_model_info(&mut r)?);
                }
            }
            // **注意：这一位无条件读**（不在 timing_info 的 if 里）。
            // 踩过的坑：写在 if 里面时，真实码流（timing_info_present = 0）会整体
            // 少读 1 位 → 后面所有字段全部错位。症状很有迷惑性：报出来的是
            // "profile 0、10bit mono、最大 6×1792"（而文件其实是 1920×1080 8bit 4:2:0），
            // 因为错位后的位串照样能"自洽地"解析出一堆值。合成测试发现不了它
            // ——写测试的那个人与写解析器的是同一个理解，两边一起错。
            // 真码流回归见 `parses_the_real_bitstream_sequence_header`。
            let initial_display_delay_present = r.bit()?;
            let count = r.f(5)? as usize + 1;
            for _ in 0..count {
                let idc = r.f(12)? as u16;
                let seq_level_idx = r.f(5)? as u8;
                // 只有 level 4.0（序号 8）及以上才有 tier 位
                let seq_tier = if seq_level_idx > 7 { r.f(1)? as u8 } else { 0 };
                let mut decoder_model_present = false;
                if decoder_model_info_present {
                    decoder_model_present = r.bit()?;
                    if decoder_model_present {
                        let delay_bits = decoder_model_info
                            .as_ref()
                            .map(|d| d.buffer_delay_length_minus_1 as u32 + 1)
                            .unwrap_or(1);
                        Self::parse_operating_parameters_info(&mut r, delay_bits)?;
                    }
                }
                let initial_display_delay_minus_1 = if initial_display_delay_present {
                    if r.bit()? {
                        r.f(4)? as u8
                    } else {
                        0
                    }
                } else {
                    0
                };
                operating_points.push(OperatingPoint {
                    idc,
                    seq_level_idx,
                    seq_tier,
                    initial_display_delay_minus_1,
                    decoder_model_present,
                });
            }
        }

        let frame_width_bits = r.f(4)? as u8 + 1;
        let frame_height_bits = r.f(4)? as u8 + 1;
        let max_frame_width = r.f(frame_width_bits as u32)? + 1;
        let max_frame_height = r.f(frame_height_bits as u32)? + 1;

        // 简化模式（静图）没有 frame_id 这一类跨帧状态
        let frame_id_numbers_present = if reduced_still_picture_header {
            false
        } else {
            r.bit()?
        };
        let mut delta_frame_id_length_minus_2 = 0u8;
        let mut additional_frame_id_length_minus_1 = 0u8;
        if frame_id_numbers_present {
            delta_frame_id_length_minus_2 = r.f(4)? as u8;
            additional_frame_id_length_minus_1 = r.f(3)? as u8;
        }

        let use_128x128_superblock = r.bit()?;
        let enable_filter_intra = r.bit()?;
        let enable_intra_edge_filter = r.bit()?;

        let (
            enable_interintra_compound,
            enable_masked_compound,
            enable_warped_motion,
            enable_dual_filter,
            enable_order_hint,
            enable_jnt_comp,
            enable_ref_frame_mvs,
            seq_force_screen_content_tools,
            seq_force_integer_mv,
            order_hint_bits,
        ) = if reduced_still_picture_header {
            // 简化模式：工具开关全 0，屏幕内容/整数运动矢量交给帧头
            (false, false, false, false, false, false, false, SELECT_SCREEN_CONTENT_TOOLS, SELECT_INTEGER_MV, 0)
        } else {
            let interintra = r.bit()?;
            let masked = r.bit()?;
            let warped = r.bit()?;
            let dual = r.bit()?;
            let order_hint = r.bit()?;
            let (jnt, ref_mvs) = if order_hint {
                (r.bit()?, r.bit()?)
            } else {
                (false, false)
            };
            let choose_sct = r.bit()?;
            let force_sct = if choose_sct {
                SELECT_SCREEN_CONTENT_TOOLS
            } else {
                r.f(1)? as u8
            };
            let force_int_mv = if force_sct > 0 {
                if r.bit()? {
                    SELECT_INTEGER_MV
                } else {
                    r.f(1)? as u8
                }
            } else {
                SELECT_INTEGER_MV
            };
            let order_hint_bits = if order_hint { r.f(3)? as u8 + 1 } else { 0 };
            (
                interintra, masked, warped, dual, order_hint, jnt, ref_mvs, force_sct, force_int_mv,
                order_hint_bits,
            )
        };

        let enable_superres = r.bit()?;
        let enable_cdef = r.bit()?;
        let enable_restoration = r.bit()?;
        let color = Self::parse_color_config(&mut r, seq_profile)?;
        let film_grain_params_present = r.bit()?;
        // 语法结构末尾的 trailing_bits（一个 1 + 补 0）：**位偏移的硬校验**。
        // 少读/多读一位都会在这里被抓住，而不是产出一堆"看着像样"的错字段。
        r.trailing_bits()?;

        Ok(Self {
            seq_profile,
            still_picture,
            reduced_still_picture_header,
            frame_width_bits,
            frame_height_bits,
            max_frame_width,
            max_frame_height,
            frame_id_numbers_present,
            delta_frame_id_length_minus_2,
            additional_frame_id_length_minus_1,
            use_128x128_superblock,
            enable_filter_intra,
            enable_intra_edge_filter,
            enable_interintra_compound,
            enable_masked_compound,
            enable_warped_motion,
            enable_dual_filter,
            enable_order_hint,
            enable_jnt_comp,
            enable_ref_frame_mvs,
            seq_force_screen_content_tools,
            seq_force_integer_mv,
            order_hint_bits,
            enable_superres,
            enable_cdef,
            enable_restoration,
            color,
            film_grain_params_present,
            operating_points,
            timing,
            decoder_model_info_present,
            decoder_model_info,
            frame_id_length: additional_frame_id_length_minus_1
                .saturating_add(delta_frame_id_length_minus_2)
                .saturating_add(3),
        })
    }

    /// `timing_info()`：显示时钟与每帧的 tick 数。
    fn parse_timing_info(r: &mut BitReader<'_>) -> Result<TimingInfo> {
        let num_units_in_display_tick = r.f(32)?;
        let time_scale = r.f(32)?;
        if num_units_in_display_tick == 0 || time_scale == 0 {
            return Err(MediaError::Bitstream("timing_info 里 tick/时标不能为 0".into()));
        }
        let equal_picture_interval = r.bit()?;
        let num_ticks_per_picture_minus_1 =
            if equal_picture_interval { r.uvlc()? } else { 0 };
        Ok(TimingInfo {
            num_units_in_display_tick,
            time_scale,
            equal_picture_interval,
            num_ticks_per_picture_minus_1,
        })
    }

    /// `decoder_model_info()`：缓冲延迟/解码时钟的长度字段。
    ///
    /// 这一层只影响"解码器模型"的时序推导，**不影响像素重建**，但必须把位读对
    /// —— 少读一位，后面所有字段整体错位。**而且帧头要用它的两个长度字段**
    /// （`buffer_removal_time` / `temporal_point_info` 的位宽），所以值要留下来。
    fn parse_decoder_model_info(r: &mut BitReader<'_>) -> Result<DecoderModelInfo> {
        let buffer_delay_length_minus_1 = r.f(5)? as u8;
        let num_units_in_decoding_tick = r.f(32)?;
        let buffer_removal_time_length_minus_1 = r.f(5)? as u8;
        let frame_presentation_time_length_minus_1 = r.f(5)? as u8;
        Ok(DecoderModelInfo {
            buffer_delay_length_minus_1,
            num_units_in_decoding_tick,
            buffer_removal_time_length_minus_1,
            frame_presentation_time_length_minus_1,
        })
    }

    /// `operating_parameters_info(op)`（规范 §5.5.4）。
    ///
    /// 两个延迟字段的位宽是 `buffer_delay_length_minus_1 + 1`（由调用方传进来），
    /// 不是固定值 —— 写死 24 位的话，这种码流（专业编码器产物）会整体错位。
    fn parse_operating_parameters_info(r: &mut BitReader<'_>, delay_bits: u32) -> Result<()> {
        let _decoder_buffer_delay = r.f(delay_bits)?;
        let _encoder_buffer_delay = r.f(delay_bits)?;
        let _low_delay_mode_flag = r.bit()?;
        Ok(())
    }

    /// `color_config()`：位深、色度格式、色彩描述。
    fn parse_color_config(r: &mut BitReader<'_>, seq_profile: u8) -> Result<ColorConfig> {
        let high_bitdepth = r.bit()?;
        let twelve_bit = if seq_profile == 2 && high_bitdepth { r.bit()? } else { false };
        let bit_depth = if twelve_bit {
            12
        } else if high_bitdepth {
            10
        } else {
            8
        };
        // profile 1 只允许 4:4:4，规范因此不写 mono_chrome 位
        let mono_chrome = if seq_profile == 1 { false } else { r.bit()? };
        let color_description_present = r.bit()?;
        let (color_primaries, transfer_characteristics, matrix_coefficients) =
            if color_description_present {
                (r.f(8)? as u8, r.f(8)? as u8, r.f(8)? as u8)
            } else {
                // 2 = unspecified（规范里的默认）
                (2u8, 2u8, 2u8)
            };

        // 特例：BT.709 + sRGB + identity 被当作"RGB，不做色度采样"（屏幕内容常用）
        const CP_BT_709: u8 = 1;
        const TC_SRGB: u8 = 13;
        const MC_IDENTITY: u8 = 0;

        if mono_chrome {
            // 单色分支在规范里是 **return**：后面的 separate_uv_delta_q 不读
            let color_range = r.bit()?;
            return Ok(ColorConfig {
                bit_depth,
                mono_chrome,
                subsampling_x: true,
                subsampling_y: true,
                chroma_sample_position: 0,
                color_range,
                color_primaries,
                transfer_characteristics,
                matrix_coefficients,
                separate_uv_delta_q: false,
            });
        }

        // 特例：BT.709 + sRGB + identity 被当作"RGB，不做色度采样"（屏幕内容常用）。
        // **注意它不 return**：`color_range` 直接赋值 1、色度采样直接 (0,0)，
        // 但末尾的 `separate_uv_delta_q` 仍然要读一位（漏读 = 后面全部错位）。
        let rgb_like = color_primaries == CP_BT_709
            && transfer_characteristics == TC_SRGB
            && matrix_coefficients == MC_IDENTITY;

        let (color_range, subsampling_x, subsampling_y) = if rgb_like {
            (true, false, false)
        } else {
            let color_range = r.bit()?;
            let (sx, sy) = match seq_profile {
                0 => (true, true),
                1 => (false, false),
                // profile 2：只有 12 位才两个位都写；否则规范直接给 (1, 0) 且**不读位**
                _ => {
                    if bit_depth == 12 {
                        let sx = r.bit()?;
                        let sy = if sx { r.bit()? } else { false };
                        (sx, sy)
                    } else {
                        (true, false)
                    }
                }
            };
            (color_range, sx, sy)
        };
        let chroma_sample_position = if subsampling_x && subsampling_y { r.f(2)? as u8 } else { 0 };
        let separate_uv_delta_q = r.bit()?;
        Ok(ColorConfig {
            bit_depth,
            mono_chrome,
            subsampling_x,
            subsampling_y,
            chroma_sample_position,
            color_range,
            color_primaries,
            transfer_characteristics,
            matrix_coefficients,
            separate_uv_delta_q,
        })
    }

    /// `equal_picture_interval`（没有 `timing_info` 时为 false）。
    ///
    /// 帧头要用它：只有"每帧 tick 数恒定"时才会去读 `temporal_point_info`。
    pub fn equal_picture_interval(&self) -> bool {
        self.timing.map(|t| t.equal_picture_interval).unwrap_or(false)
    }

    /// `operating_points_cnt_minus_1`。
    pub fn operating_points_cnt_minus_1(&self) -> usize {
        self.operating_points.len().saturating_sub(1)
    }

    /// 输出像素形态的一句话描述（日志/错误文案用）。
    pub fn pixel_format_text(&self) -> String {
        let chroma = if self.color.mono_chrome {
            "mono".to_string()
        } else if self.color.subsampling_x && self.color.subsampling_y {
            "4:2:0".to_string()
        } else if self.color.subsampling_x {
            "4:2:2".to_string()
        } else {
            "4:4:4".to_string()
        };
        format!("{}bit {chroma}", self.color.bit_depth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::testbits::{write_reduced_still_picture_1080p, BitWriter};

    /// 造一个"普通视频"的序列头（非简化模式、无 timing、一个 operating point）。
    fn build_typical_sequence_header() -> Vec<u8> {
        let mut w = BitWriter::default();
        w.f(0, 3); // seq_profile = 0（4:2:0 8/10 位）
        w.bit(false); // still_picture
        w.bit(false); // reduced_still_picture_header
        w.bit(false); // timing_info_present_flag
        w.bit(false); // initial_display_delay_present_flag（**无条件读**，不在 timing 分支里）
        w.f(0, 5); // operating_points_cnt_minus_1 = 0
        w.f(0, 12); // operating_point_idc
        w.f(8, 5); // seq_level_idx = 8（4.0）
        w.bit(true); // seq_tier（level > 7 才有）
        w.f(14, 4); // frame_width_bits_minus_1 = 14 → 15 位（够 1920）
        w.f(14, 4); // frame_height_bits_minus_1
        w.f(1919, 15); // max_frame_width_minus_1
        w.f(1079, 15); // max_frame_height_minus_1
        w.bit(true); // frame_id_numbers_present_flag
        w.f(1, 4); // delta_frame_id_length_minus_2
        w.f(3, 3); // additional_frame_id_length_minus_1
        w.bit(false); // use_128x128_superblock
        w.bit(true); // enable_filter_intra
        w.bit(true); // enable_intra_edge_filter
        w.bit(false); // enable_interintra_compound
        w.bit(true); // enable_masked_compound
        w.bit(true); // enable_warped_motion
        w.bit(false); // enable_dual_filter
        w.bit(true); // enable_order_hint
        w.bit(true); // enable_jnt_comp
        w.bit(true); // enable_ref_frame_mvs
        w.bit(true); // seq_choose_screen_content_tools
        w.bit(false); // seq_choose_integer_mv → 读 1 位
        w.f(1, 1); // seq_force_integer_mv = 1
        w.f(6, 3); // order_hint_bits_minus_1 = 6 → 7 位
        w.bit(true); // enable_superres
        w.bit(true); // enable_cdef
        w.bit(false); // enable_restoration
        // color_config：profile 0 → high_bitdepth 决定 8/10 位
        w.bit(false); // high_bitdepth = 0 → 8 位
        w.bit(false); // mono_chrome（profile 0 要读）
        w.bit(true); // color_description_present_flag
        w.f(1, 8); // color_primaries = BT.709
        w.f(1, 8); // transfer_characteristics
        w.f(1, 8); // matrix_coefficients
        w.bit(false); // color_range = studio swing
        w.f(0, 2); // chroma_sample_position（4:2:0 才读这两位）
        w.bit(false); // separate_uv_delta_q
        w.bit(true); // film_grain_params_present
        w.trailing(); // trailing_bits
        w.finish().to_vec()
    }

    #[test]
    fn parses_typical_sequence_header() {
        let sh = SequenceHeader::parse(&build_typical_sequence_header()).unwrap();
        assert_eq!(sh.seq_profile, 0);
        assert!(!sh.still_picture);
        assert!(!sh.reduced_still_picture_header);
        assert_eq!(sh.frame_width_bits, 15);
        assert_eq!(sh.frame_height_bits, 15);
        assert_eq!(sh.max_frame_width, 1920);
        assert_eq!(sh.max_frame_height, 1080);
        assert!(sh.frame_id_numbers_present);
        assert!(!sh.use_128x128_superblock, "64×64 超块");
        assert!(sh.enable_filter_intra);
        assert!(sh.enable_order_hint);
        assert!(sh.enable_jnt_comp, "order hint 开着才读这两位");
        assert_eq!(sh.order_hint_bits, 7, "读的是 minus_1");
        assert_eq!(sh.seq_force_integer_mv, 1);
        assert!(sh.enable_superres);
        assert!(sh.enable_cdef);
        assert!(!sh.enable_restoration);
        assert_eq!(sh.color.bit_depth, 8);
        assert_eq!(sh.color.color_primaries, 1);
        assert!(!sh.color.color_range);
        assert_eq!(sh.pixel_format_text(), "8bit 4:2:0");
        assert!(sh.film_grain_params_present);
        assert_eq!(sh.operating_points.len(), 1);
        assert_eq!(sh.operating_points[0].seq_level_idx, 8);
        assert_eq!(sh.operating_points[0].seq_tier, 1);
        assert!(sh.timing.is_none(), "没有 timing_info");
    }

    #[test]
    fn parses_reduced_still_picture_header() {
        let mut w = BitWriter::default();
        write_reduced_still_picture_1080p(&mut w);
        let sh = SequenceHeader::parse(w.finish()).unwrap();
        assert!(sh.still_picture);
        assert!(sh.reduced_still_picture_header);
        assert!(!sh.frame_id_numbers_present, "简化模式没有 frame_id");
        assert_eq!(sh.max_frame_width, 1920);
        assert_eq!(sh.max_frame_height, 1080);
        assert_eq!(sh.frame_width_bits, 11, "1920 需要 11 位");
        assert_eq!(sh.seq_force_screen_content_tools, SELECT_SCREEN_CONTENT_TOOLS);
        assert!(!sh.enable_order_hint);
        assert_eq!(sh.order_hint_bits, 0, "order hint 关着就是 0");
        assert_eq!(sh.color.color_primaries, 2, "没写色彩描述时是 unspecified");
        assert!(!sh.color.color_range);
        assert_eq!(sh.color.chroma_sample_position, 0);
        assert!(sh.enable_cdef);
        assert!(!sh.enable_restoration);
        assert_eq!(sh.pixel_format_text(), "8bit 4:2:0");
    }

    #[test]
    fn parses_ten_bit_and_mono_chrome() {
        let mut w = BitWriter::default();
        w.f(2, 3); // profile 2
        w.bit(false);
        w.bit(false);
        w.bit(false); // 无 timing
        w.bit(false); // initial_display_delay_present_flag
        w.f(0, 5);
        w.f(0, 12);
        w.f(5, 5); // seq_level_idx = 5（不带 tier）
        w.f(0, 4);
        w.f(0, 4); // 位宽 1 位
        w.f(0, 1); // max_frame_width_minus_1 = 0 → 1
        w.f(0, 1); // max_frame_height_minus_1 = 0 → 1
        w.bit(false); // frame_id_numbers_present_flag
        w.bit(false); // use_128x128_superblock
        w.bit(false); // enable_filter_intra
        w.bit(false); // enable_intra_edge_filter
        w.bit(false); // interintra
        w.bit(false); // masked
        w.bit(false); // warped
        w.bit(false); // dual filter
        w.bit(false); // order hint 关
        w.bit(true); // choose screen content tools
        w.bit(true); // choose integer mv
        w.bit(false); // enable_superres
        w.bit(false); // enable_cdef
        w.bit(false); // enable_restoration
        w.bit(true); // high_bitdepth → 10 位
        // profile 2 + high_bitdepth → 读 twelve_bit
        w.bit(false); // twelve_bit = 0
        w.bit(true); // mono_chrome
        w.bit(false); // color_description_present
        w.bit(true); // color_range
        w.bit(false); // film_grain (mono 分支里没有 separate_uv_delta_q)
        w.trailing();
        let sh = SequenceHeader::parse(w.finish()).unwrap();
        assert_eq!(sh.seq_profile, 2);
        assert_eq!(sh.color.bit_depth, 10);
        assert!(sh.color.mono_chrome);
        assert_eq!(sh.pixel_format_text(), "10bit mono");
        assert_eq!(sh.operating_points[0].seq_tier, 0, "level ≤ 7 不带 tier");
        assert!(sh.operating_points[0].initial_display_delay_minus_1 == 0);
    }

    /// 真实素材（B 站 1080p AV1）的序列头载荷，见
    /// `parses_the_real_bitstream_sequence_header` 的说明。
    const REAL_SEQUENCE_HEADER: [u8; 14] = [
        0x00, 0x00, 0x00, 0x42, 0xab, 0xbf, 0xc3, 0x77, 0xee, 0x66, 0x40, 0x40, 0x40, 0x41,
    ];

    #[test]
    fn parses_the_real_bitstream_sequence_header() {
        // 这段字节来自真实素材（B 站 1080p AV1 + Hi-Res 音轨那条 ASMR）：它在文件里
        // 出现两次 —— moov 的 `av1C` configOBUs 里一次、第一个采样的序列头 OBU 里
        // 一次，两处逐字节相同。
        //
        // 为什么必须有这条：**合成测试只能证明自洽**（写测试的人与写解析器的人出自
        // 同一份理解，会一起错）。事实上第一版解析器就栽在这里：少读一位，这段字节
        // 被解成 "profile 0、10bit mono、最大 6×1792"，而文件其实是 1920×1080 8bit 4:2:0。
        let sh = SequenceHeader::parse(&REAL_SEQUENCE_HEADER).unwrap();
        // 外部真值：av1C 记录（81 08 0C 00）= profile 0 / level 8 / 8bit 4:2:0；
        // probe 报这一轨是 1920×1080。序列头必须与之一致。
        assert_eq!(sh.seq_profile, 0);
        assert_eq!(sh.operating_points.len(), 1);
        assert_eq!(sh.operating_points[0].seq_level_idx, 8, "level 4.0");
        assert_eq!(sh.operating_points[0].seq_tier, 0);
        assert!(!sh.frame_id_numbers_present);
        assert_eq!(sh.frame_width_bits, 11);
        assert_eq!(sh.frame_height_bits, 11);
        assert_eq!(sh.max_frame_width, 1920);
        assert_eq!(sh.max_frame_height, 1080);
        assert_eq!(sh.pixel_format_text(), "8bit 4:2:0");
        assert!(sh.timing.is_none());

        // 下面这些没有外部真值可比，但它们是"111 位语法 + 1 位 trailing bit = 14 字节
        // 正好读完"的结果 —— 逐位对齐的证据就在这里，所以一并钉死：
        // 以后任何人改解析器（哪怕只是调一个字段的顺序）都会立刻被这几条打回来。
        assert!(sh.use_128x128_superblock);
        assert!(sh.enable_filter_intra);
        assert!(sh.enable_intra_edge_filter);
        assert!(sh.enable_warped_motion);
        assert!(!sh.enable_dual_filter);
        assert!(sh.enable_order_hint);
        assert!(sh.enable_jnt_comp);
        assert!(sh.enable_ref_frame_mvs);
        assert_eq!(sh.order_hint_bits, 7);
        assert_eq!(sh.seq_force_screen_content_tools, 0);
        assert!(!sh.enable_superres);
        assert!(sh.enable_cdef);
        assert!(sh.enable_restoration);
        assert_eq!(sh.color.color_primaries, 1, "BT.709");
        assert_eq!(sh.color.transfer_characteristics, 1);
        assert_eq!(sh.color.matrix_coefficients, 1);
        assert!(!sh.color.color_range, "studio swing");
        assert_eq!(sh.color.chroma_sample_position, 0);
        assert!(!sh.color.separate_uv_delta_q);
        assert!(!sh.film_grain_params_present);
    }

    #[test]
    fn a_bit_shift_is_caught_by_trailing_bits() {
        // 抹掉真实载荷最后一位（那个 trailing bit，0x41 → 0x40）：等价于"字段多读/
        // 少读了一位"，必须被抓住，而不是照旧产出一堆看着像样的错字段。
        let mut payload = REAL_SEQUENCE_HEADER;
        payload[13] &= 0xFE;
        let err = SequenceHeader::parse(&payload).expect_err("trailing bit 丢了就该报错");
        assert!(format!("{err}").contains("trailing"), "{err}");
    }

    #[test]
    fn profile2_without_twelve_bit_is_four_two_two_without_extra_bits() {
        // 规范（§5.5.2）：profile 2 且位深不是 12 → 色度采样**直接给 (1, 0)**，
        // 不读任何位。第一版按"读一位"写，是这次两处 color_config 错误之一。
        let mut w = BitWriter::default();
        w.f(2, 3); // profile 2
        w.bit(false); // still_picture
        w.bit(false); // reduced
        w.bit(false); // timing_info_present
        w.bit(false); // initial_display_delay_present
        w.f(0, 5); // operating_points_cnt_minus_1
        w.f(0, 12); // operating_point_idc
        w.f(5, 5); // seq_level_idx = 5（≤7，不带 tier）
        w.f(10, 4); // frame_width_bits_minus_1
        w.f(10, 4); // frame_height_bits_minus_1
        w.f(1919, 11);
        w.f(1079, 11);
        w.bit(false); // frame_id_numbers_present_flag
        w.bit(false); // use_128x128_superblock
        w.bit(false); // enable_filter_intra
        w.bit(false); // enable_intra_edge_filter
        w.bit(false); // enable_interintra_compound
        w.bit(false); // enable_masked_compound
        w.bit(false); // enable_warped_motion
        w.bit(false); // enable_dual_filter
        w.bit(false); // enable_order_hint 关 → 不读 jnt/ref_mvs，也不读 order_hint_bits
        w.bit(true); // seq_choose_screen_content_tools → force = SELECT（不读 force 位）
        w.bit(true); // seq_choose_integer_mv → force = SELECT
        w.bit(false); // enable_superres
        w.bit(true); // enable_cdef
        w.bit(false); // enable_restoration
        w.bit(true); // high_bitdepth → 10 位
        w.bit(false); // twelve_bit = 0
        w.bit(false); // mono_chrome
        w.bit(false); // color_description_present_flag
        w.bit(false); // color_range
        // 色度采样：**这里一位都不写**（规范直接给 (1, 0)）
        w.bit(false); // separate_uv_delta_q
        w.bit(false); // film_grain_params_present
        w.trailing();
        let sh = SequenceHeader::parse(w.finish()).unwrap();
        assert_eq!(sh.color.bit_depth, 10);
        assert!(sh.color.subsampling_x && !sh.color.subsampling_y, "4:2:2");
        assert_eq!(sh.pixel_format_text(), "10bit 4:2:2");
        assert_eq!(sh.max_frame_width, 1920);
        assert_eq!(sh.max_frame_height, 1080);
    }

    #[test]
    fn truncated_sequence_header_is_an_error() {        let full = build_typical_sequence_header();
        for cut in [1usize, 4, 8] {
            assert!(
                SequenceHeader::parse(&full[..full.len() - cut]).is_err(),
                "截断的序列头必须报错（少 {cut} 字节）"
            );
        }
    }

    #[test]
    fn rejects_zero_timing_info() {
        let mut w = BitWriter::default();
        w.f(0, 3);
        w.bit(false);
        w.bit(false);
        w.bit(true); // timing_info_present
        w.f(0, 32); // num_units_in_display_tick = 0（非法）
        w.f(30, 32);
        assert!(SequenceHeader::parse(w.finish()).is_err());
    }
}
