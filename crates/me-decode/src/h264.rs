//! H.264 码流解析：**三个平台的硬解后端共用的"共识层"**。
//!
//! ## 为什么需要它
//!
//! 硬解后端对"谁解析码流"的分工不一样：
//!
//! - **VideoToolbox（macOS）/ Media Foundation（Windows）**：码流从头到尾由系统
//!   解码器自己解析，调用方只要按格式把 NAL 喂进去。这条路上本模块干两件事：
//!   从 `avcC` 里**提取参数集**，以及把**不支持的构造提前说清楚** —— 否则用户
//!   看到的是一句无从下手的 OSStatus / HRESULT。
//! - **VA-API（Linux）**：它是 **VLD（变长解码）接口** —— 驱动只做熵解码往后的
//!   部分，SPS/PPS/slice 头必须由调用方解析成结构体喂进去，**参考帧列表**也得
//!   调用方自己按 H.264 8.2.4 构造。这条路上本模块是必需品，不是优化。
//!
//! ## 支持口径（超出即明确报 `Unsupported`）
//!
//! **8 位 4:2:0 的逐行帧 + 只做滑动窗口参考帧标记**：这是现实里绝大多数 H.264
//! 素材的形态，也是三个后端能给出**一致行为**的最小交集。隔行/场、10 位色深、
//! FMO、MMCO、长期参考、SP/SI 切片一律明确报错 —— "猜着解"出来的错色/错位画面
//! 比一句清楚的不支持难查得多（这是引擎的硬约定）。
//!
//! 规范引用都指 **ITU-T H.264（ISO/IEC 14496-10）**：语法见 7.3、推导见 8.2。

use me_core::bits::{remove_emulation_prevention, BitReader};
use me_core::{MediaError, Result};

// ── NAL 单元类型（7.4.1，这里只列后端会遇到的） ──────────────────────────────

pub const NAL_SLICE: u8 = 1;
pub const NAL_SLICE_IDR: u8 = 5;
pub const NAL_SEI: u8 = 6;
pub const NAL_SPS: u8 = 7;
pub const NAL_PPS: u8 = 8;
pub const NAL_AUD: u8 = 9;

/// 这个 NAL 类型是不是"带画面数据的"（VCL：1~5）。
pub fn is_vcl(nal_type: u8) -> bool {
    (1..=5).contains(&nal_type)
}

/// 一组 SPS / PPS：`avcC` 里拆出来的形态，每项是一段**完整 NAL**（含 NAL 头字节）。
///
/// 起别名不只是为了压 clippy 的类型复杂度告警 —— `(Vec<Vec<u8>>, Vec<Vec<u8>>)`
/// 光看签名分不清哪个是 SPS、哪个是 PPS，起了名字调用点才好读。
pub type ParameterSets = (Vec<Vec<u8>>, Vec<Vec<u8>>);

/// 从 `avcC` 里拆出 SPS / PPS 列表。
///
/// avcC 结构（ISO 14496-15 §5.2.4.1.1）：
/// `[0]=1` `[1..4]=profile/compat/level` `[4]=0xFC|lengthSizeMinusOne`
/// `[5]=0xE0|numSPS` 之后是 `u16 长度 + 内容`，再 `numPPS` + 同样的序列。
pub fn parse_avcc(avcc: &[u8]) -> Result<ParameterSets> {
    if avcc.len() < 7 || avcc[0] != 1 {
        return Err(MediaError::Bitstream("avcC 头不合法".into()));
    }
    let mut pos = 5usize;
    let num_sps = (avcc[pos] & 0x1F) as usize;
    pos += 1;
    let mut sps = Vec::new();
    for _ in 0..num_sps {
        let len = read_u16(avcc, pos)?;
        pos += 2;
        if pos + len > avcc.len() {
            return Err(MediaError::Bitstream("avcC 里的 SPS 越界".into()));
        }
        sps.push(avcc[pos..pos + len].to_vec());
        pos += len;
    }
    if pos >= avcc.len() {
        return Err(MediaError::Bitstream("avcC 里没有 PPS".into()));
    }
    let num_pps = avcc[pos] as usize;
    pos += 1;
    let mut pps = Vec::new();
    for _ in 0..num_pps {
        let len = read_u16(avcc, pos)?;
        pos += 2;
        if pos + len > avcc.len() {
            return Err(MediaError::Bitstream("avcC 里的 PPS 越界".into()));
        }
        pps.push(avcc[pos..pos + len].to_vec());
        pos += len;
    }
    Ok((sps, pps))
}

/// `avcC` 里声明的 NAL 长度前缀字节数（`lengthSizeMinusOne + 1`）。
///
/// 引擎内部统一按 4 字节喂（`me-mux::convert` 转换时就写死 4），但这里如实读出来：
/// 外部给我们的 `avcC` 未必是 4，读错了整条采样都会错位。
pub fn nal_length_size(avcc: &[u8]) -> usize {
    if avcc.len() >= 5 {
        ((avcc[4] & 0x03) as usize) + 1
    } else {
        4
    }
}

fn read_u16(buf: &[u8], pos: usize) -> Result<usize> {
    if pos + 2 > buf.len() {
        return Err(MediaError::Bitstream("avcC 读长度越界".into()));
    }
    Ok(u16::from_be_bytes([buf[pos], buf[pos + 1]]) as usize)
}

/// 把一个 AVCC 采样（`[长度][NAL]…` 重复）拆成 NAL 单元（每项含 NAL 头字节）。
pub fn split_avcc_sample(sample: &[u8], length_size: usize) -> Result<Vec<&[u8]>> {
    let len_size = length_size.clamp(1, 4);
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < sample.len() {
        if pos + len_size > sample.len() {
            return Err(MediaError::Bitstream(format!(
                "采样尾部只剩 {} 字节，放不下 {len_size} 字节的长度前缀",
                sample.len() - pos
            )));
        }
        let mut len = 0usize;
        for i in 0..len_size {
            len = (len << 8) | sample[pos + i] as usize;
        }
        pos += len_size;
        if pos + len > sample.len() {
            return Err(MediaError::Bitstream(format!(
                "采样里的 NAL 长度 {len} 越界（还剩 {} 字节）",
                sample.len() - pos
            )));
        }
        if len > 0 {
            out.push(&sample[pos..pos + len]);
        }
        pos += len;
    }
    Ok(out)
}

/// AVCC → Annex-B（把长度前缀换成起始码，字节内容一字不改）。
///
/// Windows 的 Media Foundation 解码器吃的是 Annex-B（带起始码的字节流），
/// 而引擎内部统一是 AVCC —— 这一步只在 Windows 后端用得到。
pub fn avcc_to_annexb(sample: &[u8], length_size: usize) -> Result<Vec<u8>> {
    let nals = split_avcc_sample(sample, length_size)?;
    let mut out = Vec::with_capacity(sample.len() + nals.len() * 4);
    for n in nals {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(n);
    }
    Ok(out)
}

/// 高profile 集合（7.3.2.1.1 里"要读 chroma_format_idc 那一段"的 profile_idc 全集）。
///
/// 这张表还有第二个用处：VA-API 的 `seq_fields.bits.MinLumaBiPredSize8x8`
/// 就是拿它判断的（见 `linux.rs`）—— 高 profile 的最小明暗块是 8x8 而不是 4x4。
pub fn is_high_profile(profile_idc: u8) -> bool {
    matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134
    )
}

// ── 序列参数集 ───────────────────────────────────────────────────────────────

/// VUI 里与本引擎有关的部分（其余字段只为"读到位"而解析）。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Vui {
    /// `timing_info_present_flag` 那三个值：`(num_units_in_tick, time_scale, fixed_frame_rate)`。
    pub timing: Option<(u32, u32, bool)>,
    pub bitstream_restriction: Option<BitstreamRestriction>,
}

/// `bitstream_restriction_flag` 段（输出顺序缓冲的规格上限都在这里）。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BitstreamRestriction {
    pub max_num_reorder_frames: u32,
    pub max_dec_frame_buffering: u32,
}

/// 序列参数集（只保留后端与校验用得到的字段）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sps {
    pub profile_idc: u8,
    pub constraint_flags: u8,
    pub level_idc: u8,
    pub id: u8,
    pub chroma_format_idc: u8,
    pub separate_colour_plane_flag: bool,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    /// `log2_max_frame_num`（已加回 4）
    pub log2_max_frame_num: u8,
    pub pic_order_cnt_type: u8,
    /// `log2_max_pic_order_cnt_lsb`（已加回 4）
    pub log2_max_pic_order_cnt_lsb: u8,
    pub delta_pic_order_always_zero_flag: bool,
    pub offset_for_non_ref_pic: i32,
    pub offset_for_top_to_bottom_field: i32,
    pub offset_for_ref_frame: Vec<i32>,
    pub max_num_ref_frames: u32,
    pub gaps_in_frame_num_value_allowed_flag: bool,
    pub pic_width_in_mbs: u32,
    pub pic_height_in_map_units: u32,
    pub frame_mbs_only_flag: bool,
    pub mb_adaptive_frame_field_flag: bool,
    pub direct_8x8_inference_flag: bool,
    pub frame_cropping_flag: bool,
    pub crop_left: u32,
    pub crop_right: u32,
    pub crop_top: u32,
    pub crop_bottom: u32,
    /// 4x4 的 6 张（i 0..5）与 8x8 的 2 张（i 6..7）；`None` = 该表没在这里给出。
    pub scaling4x4: [Option<[u8; 16]>; 6],
    pub scaling8x8: [Option<[u8; 64]>; 2],
    pub vui: Option<Vui>,
}

impl Default for Sps {
    fn default() -> Self {
        Self {
            profile_idc: 0,
            constraint_flags: 0,
            level_idc: 0,
            id: 0,
            // 非高 profile 的默认值（7.3.2.1.1）
            chroma_format_idc: 1,
            separate_colour_plane_flag: false,
            bit_depth_luma_minus8: 0,
            bit_depth_chroma_minus8: 0,
            log2_max_frame_num: 4,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb: 4,
            delta_pic_order_always_zero_flag: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            offset_for_ref_frame: Vec::new(),
            max_num_ref_frames: 1,
            gaps_in_frame_num_value_allowed_flag: false,
            pic_width_in_mbs: 1,
            pic_height_in_map_units: 1,
            frame_mbs_only_flag: true,
            mb_adaptive_frame_field_flag: false,
            direct_8x8_inference_flag: false,
            frame_cropping_flag: false,
            crop_left: 0,
            crop_right: 0,
            crop_top: 0,
            crop_bottom: 0,
            scaling4x4: [None; 6],
            scaling8x8: [None; 2],
            vui: None,
        }
    }
}

impl Sps {
    /// 解析一个 SPS（入参是**完整 NAL**，含头字节）。
    pub fn parse(nal: &[u8]) -> Result<Sps> {
        if nal.len() < 4 {
            return Err(MediaError::Bitstream("SPS 太短".into()));
        }
        // 位读出前必须先去防竞争字节（0x00 0x00 0x03），否则位全错
        let rbsp = remove_emulation_prevention(&nal[1..]);
        let mut r = BitReader::new(&rbsp);
        let mut sps = Sps {
            profile_idc: r.read_u32(8)? as u8,
            constraint_flags: r.read_u32(8)? as u8,
            level_idc: r.read_u32(8)? as u8,
            id: r.read_ue()? as u8,
            ..Sps::default()
        };
        if is_high_profile(sps.profile_idc) {
            sps.chroma_format_idc = r.read_ue()? as u8;
            if sps.chroma_format_idc == 3 {
                sps.separate_colour_plane_flag = r.read_bit()? == 1;
            }
            sps.bit_depth_luma_minus8 = r.read_ue()? as u8;
            sps.bit_depth_chroma_minus8 = r.read_ue()? as u8;
            let _qpprime_y_zero_transform_bypass_flag = r.read_bit()?;
            if r.read_bit()? == 1 {
                // seq_scaling_matrix_present_flag：4:4:4 是 12 张表，其余 8 张
                let count = if sps.chroma_format_idc != 3 { 8 } else { 12 };
                let mut lists: Vec<Option<[u8; 64]>> = Vec::with_capacity(count);
                for i in 0..count {
                    if r.read_bit()? == 1 {
                        let size = if i < 6 { 16 } else { 64 };
                        lists.push(Some(read_scaling_list(&mut r, size)?));
                    } else {
                        lists.push(None);
                    }
                }
                // 4:4:4 的 8..11 张在 4:2:0 上没有意义；这里只留用得上的
                for i in 0..6.min(lists.len()) {
                    sps.scaling4x4[i] = lists[i].map(|l| {
                        let mut out = [16u8; 16];
                        out.copy_from_slice(&l[..16]);
                        out
                    });
                }
                for i in 0..2 {
                    if let Some(Some(l)) = lists.get(6 + i) {
                        sps.scaling8x8[i] = Some(*l);
                    }
                }
            }
        }
        sps.log2_max_frame_num = u8::try_from(r.read_ue()?)
            .ok()
            .and_then(|v| v.checked_add(4))
            .filter(|v| *v <= 16)
            .ok_or_else(|| MediaError::Bitstream("SPS 的 log2_max_frame_num_minus4 不合法".into()))?;
        sps.pic_order_cnt_type = u8::try_from(r.read_ue()?)
            .map_err(|_| MediaError::Bitstream("SPS 的 pic_order_cnt_type 不合法".into()))?;
        match sps.pic_order_cnt_type {
            0 => {
                sps.log2_max_pic_order_cnt_lsb = u8::try_from(r.read_ue()?)
                    .ok()
                    .and_then(|v| v.checked_add(4))
                    .filter(|v| *v <= 16)
                    .ok_or_else(|| {
                        MediaError::Bitstream("SPS 的 log2_max_pic_order_cnt_lsb_minus4 不合法".into())
                    })?;
            }
            1 => {
                sps.delta_pic_order_always_zero_flag = r.read_bit()? == 1;
                sps.offset_for_non_ref_pic = r.read_se()?;
                sps.offset_for_top_to_bottom_field = r.read_se()?;
                let n = r.read_ue()?;
                if n > 255 {
                    return Err(MediaError::Bitstream(format!(
                        "SPS 的 num_ref_frames_in_pic_order_cnt_cycle = {n}（规范上限 255）"
                    )));
                }
                for _ in 0..n {
                    sps.offset_for_ref_frame.push(r.read_se()?);
                }
            }
            2 => {}
            other => {
                return Err(MediaError::Bitstream(format!(
                    "SPS 的 pic_order_cnt_type = {other} 不在规范里"
                )))
            }
        }
        sps.max_num_ref_frames = r.read_ue()?;
        sps.gaps_in_frame_num_value_allowed_flag = r.read_bit()? == 1;
        sps.pic_width_in_mbs = read_ue_plus1(&mut r, "pic_width_in_mbs_minus1")?;
        sps.pic_height_in_map_units = read_ue_plus1(&mut r, "pic_height_in_map_units_minus1")?;
        sps.frame_mbs_only_flag = r.read_bit()? == 1;
        if !sps.frame_mbs_only_flag {
            sps.mb_adaptive_frame_field_flag = r.read_bit()? == 1;
        }
        sps.direct_8x8_inference_flag = r.read_bit()? == 1;
        sps.frame_cropping_flag = r.read_bit()? == 1;
        if sps.frame_cropping_flag {
            sps.crop_left = r.read_ue()?;
            sps.crop_right = r.read_ue()?;
            sps.crop_top = r.read_ue()?;
            sps.crop_bottom = r.read_ue()?;
        }
        if r.read_bit()? == 1 {
            sps.vui = Some(parse_vui(&mut r)?);
        }
        Ok(sps)
    }

    /// **编码尺寸**（宏块对齐后）：解码器内部的面/纹理就是这个大小。
    pub fn coded_width(&self) -> u32 {
        self.pic_width_in_mbs * 16
    }

    pub fn coded_height(&self) -> u32 {
        self.pic_height_in_map_units * 16 * (2 - self.frame_mbs_only_flag as u32)
    }

    fn crop_unit(&self) -> (u32, u32) {
        // 只支持 4:2:0 8 位（check_supported 会拦下别的），此时 SubWidthC = SubHeightC = 2
        let unit_x = if self.chroma_format_idc == 0 || self.separate_colour_plane_flag {
            1
        } else {
            2
        };
        let unit_y = unit_x * (2 - self.frame_mbs_only_flag as u32);
        (unit_x, unit_y)
    }

    /// **显示尺寸**（裁掉 `frame_cropping` 之后）—— 对外帧的大小就是它。
    ///
    /// 编码尺寸按 16 对齐（`pic_height_in_map_units` 的单位是宏块行），
    /// 而 360p 这类高度不是 16 的整数倍，多出来的部分靠裁剪去掉。
    pub fn width(&self) -> u32 {
        let (ux, _) = self.crop_unit();
        self.coded_width()
            .saturating_sub((self.crop_left + self.crop_right) * ux)
    }

    pub fn height(&self) -> u32 {
        let (_, uy) = self.crop_unit();
        self.coded_height()
            .saturating_sub((self.crop_top + self.crop_bottom) * uy)
    }

    /// 规范隐含的 `MaxDpbFrames`（VUI 没写时的代替值）：
    /// `min(MaxDpbMbs / (PicWidthInMbs * FrameHeightInMbs), 16)`。
    ///
    /// `MaxDpbMbs` 随 level 变，这里取 5.1 档的 16×1024（最大档）：
    /// 取大不取小 —— 它只影响"我们愿意缓冲多少帧"，取大了顶多多占几帧内存，
    /// 取小了会把合法的输出顺序缓冲算少，画面顺序就错了。
    pub fn implied_max_dpb_frames(&self) -> u32 {
        const MAX_DPB_MBS: u32 = 16 * 1024;
        let mbs = (self.pic_width_in_mbs * self.pic_height_in_map_units).max(1);
        (MAX_DPB_MBS / mbs).min(16)
    }

    /// 输出顺序缓冲的深度：重排窗口多大，解码器就得压住多少帧才能按显示顺序吐出。
    pub fn max_num_reorder_frames(&self) -> u32 {
        self.vui
            .as_ref()
            .and_then(|v| v.bitstream_restriction.as_ref())
            .map(|b| b.max_num_reorder_frames)
            .unwrap_or_else(|| self.implied_max_dpb_frames())
    }

    pub fn max_dec_frame_buffering(&self) -> u32 {
        self.vui
            .as_ref()
            .and_then(|v| v.bitstream_restriction.as_ref())
            .map(|b| b.max_dec_frame_buffering)
            .unwrap_or_else(|| self.implied_max_dpb_frames())
    }

    /// 这一版引擎不支持的构造在这里**明确报出来**（绝不静默出错帧）。
    pub fn check_supported(&self) -> Result<()> {
        if self.separate_colour_plane_flag {
            return Err(MediaError::Unsupported(
                "H.264 用了独立色彩平面（separate_colour_plane_flag=1）".into(),
            ));
        }
        if self.chroma_format_idc != 1 {
            return Err(MediaError::Unsupported(format!(
                "H.264 是 {} 色度采样，引擎只支持 4:2:0",
                match self.chroma_format_idc {
                    0 => "单色（4:0:0）",
                    2 => "4:2:2",
                    _ => "4:4:4",
                }
            )));
        }
        if self.bit_depth_luma_minus8 != 0 || self.bit_depth_chroma_minus8 != 0 {
            return Err(MediaError::Unsupported(format!(
                "H.264 是 {}-位色深，引擎对外只有 8 位 NV12",
                self.bit_depth_luma_minus8 as u32 + 8
            )));
        }
        if !self.frame_mbs_only_flag {
            return Err(MediaError::Unsupported(
                "H.264 是隔行（场）编码（frame_mbs_only_flag=0），引擎只支持逐行帧".into(),
            ));
        }
        if self.mb_adaptive_frame_field_flag {
            return Err(MediaError::Unsupported(
                "H.264 开了宏块级自适应帧场（MBaff），引擎只支持逐行帧".into(),
            ));
        }
        if self.gaps_in_frame_num_value_allowed_flag {
            return Err(MediaError::Unsupported(
                "H.264 声明允许 frame_num 跳号（gaps_in_frame_num_value_allowed_flag=1），\
                 这一版不实现\"不存在的帧\"补齐"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// 读 `ue(v)` 并 +1（尺寸类字段都是 minus1 形态），溢出/为 0 都当非法。
fn read_ue_plus1(r: &mut BitReader, name: &str) -> Result<u32> {
    let v = r.read_ue()?;
    v.checked_add(1)
        .ok_or_else(|| MediaError::Bitstream(format!("SPS 的 {name} 溢出")))
}

/// `scaling_list()`（7.3.2.1.1.1）。
///
/// 返回**有效矩阵**：`useDefaultScalingMatrixFlag`（第 0 项就把 nextScale 归零）
/// 表示"这张表用缺省"，而缺省就是 Flat（全 16）—— 注意此时按公式逐项算出来的是
/// 全 8，与规范要求的"回到缺省"不是一回事，必须显式区分。
fn read_scaling_list(r: &mut BitReader, size: usize) -> Result<[u8; 64]> {
    let mut list = [16u8; 64];
    let mut last = 8i32;
    let mut next = 8i32;
    for j in 0..size {
        if next != 0 {
            let delta = r.read_se()?;
            next = (last + delta + 256) % 256;
            if j == 0 && next == 0 {
                return Ok([16u8; 64]);
            }
        }
        let v = if next == 0 { last } else { next };
        list[j] = v as u8;
        last = v;
    }
    Ok(list)
}

/// `vui_parameters()`（7.3.2.1.1）。
///
/// 本引擎只用得上 `timing_info`（记着）与 `bitstream_restriction`（**必需**：
/// `max_num_reorder_frames` 决定输出顺序缓冲的深度）。其余字段照规范读过去，
/// 因为不读就会位错位 —— VUI 是个"可选字段串成的链"，跳过任何一环后面都对不上。
fn parse_vui(r: &mut BitReader) -> Result<Vui> {
    let mut vui = Vui::default();
    if r.read_bit()? == 1 {
        let aspect_ratio_idc = r.read_u32(8)?;
        if aspect_ratio_idc == 255 {
            let _sar_width = r.read_u32(16)?;
            let _sar_height = r.read_u32(16)?;
        }
    }
    if r.read_bit()? == 1 {
        let _overscan_appropriate_flag = r.read_bit()?;
    }
    if r.read_bit()? == 1 {
        let _video_format = r.read_u32(3)?;
        let _video_full_range_flag = r.read_bit()?;
        if r.read_bit()? == 1 {
            let _colour_primaries = r.read_u32(8)?;
            let _transfer_characteristics = r.read_u32(8)?;
            let _matrix_coefficients = r.read_u32(8)?;
        }
    }
    if r.read_bit()? == 1 {
        let _chroma_sample_loc_type_top_field = r.read_ue()?;
        let _chroma_sample_loc_type_bottom_field = r.read_ue()?;
    }
    if r.read_bit()? == 1 {
        let num_units_in_tick = r.read_u32(32)?;
        let time_scale = r.read_u32(32)?;
        let fixed_frame_rate = r.read_bit()? == 1;
        vui.timing = Some((num_units_in_tick, time_scale, fixed_frame_rate));
    }
    let nal_hrd = r.read_bit()? == 1;
    if nal_hrd {
        skip_hrd_parameters(r)?;
    }
    let vcl_hrd = r.read_bit()? == 1;
    if vcl_hrd {
        skip_hrd_parameters(r)?;
    }
    if nal_hrd || vcl_hrd {
        let _low_delay_hrd_flag = r.read_bit()?;
    }
    let _pic_struct_present_flag = r.read_bit()?;
    if r.read_bit()? == 1 {
        let _motion_vectors_over_pic_boundaries_flag = r.read_bit()?;
        let _max_bytes_per_pic_denom = r.read_ue()?;
        let _max_bits_per_mb_denom = r.read_ue()?;
        let _log2_max_mv_length_horizontal = r.read_ue()?;
        let _log2_max_mv_length_vertical = r.read_ue()?;
        vui.bitstream_restriction = Some(BitstreamRestriction {
            max_num_reorder_frames: r.read_ue()?,
            max_dec_frame_buffering: r.read_ue()?,
        });
    }
    Ok(vui)
}

/// `hrd_parameters()`（E.1.2）：只为读到位而走一遍。
fn skip_hrd_parameters(r: &mut BitReader) -> Result<()> {
    let cpb_cnt_minus1 = r.read_ue()?;
    if cpb_cnt_minus1 > 31 {
        return Err(MediaError::Bitstream(format!(
            "VUI 的 cpb_cnt_minus1 = {cpb_cnt_minus1}（规范上限 31）"
        )));
    }
    let _bit_rate_scale = r.read_u32(4)?;
    let _cpb_size_scale = r.read_u32(4)?;
    for _ in 0..=cpb_cnt_minus1 {
        let _bit_rate_value_minus1 = r.read_ue()?;
        let _cpb_size_value_minus1 = r.read_ue()?;
        let _cbr_flag = r.read_bit()?;
    }
    let _initial_cpb_removal_delay_length_minus1 = r.read_u32(5)?;
    let _cpb_removal_delay_length_minus1 = r.read_u32(5)?;
    let _dpb_output_delay_length_minus1 = r.read_u32(5)?;
    let _time_offset_length = r.read_u32(5)?;
    Ok(())
}

// ── 图像参数集 ───────────────────────────────────────────────────────────────

/// 图像参数集（同样只留后端与校验用得到的字段）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pps {
    pub id: u8,
    pub sps_id: u8,
    pub entropy_coding_mode_flag: bool,
    pub bottom_field_pic_order_in_frame_present_flag: bool,
    pub num_slice_groups_minus1: u32,
    pub num_ref_idx_l0_default_active_minus1: u32,
    pub num_ref_idx_l1_default_active_minus1: u32,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_idc: u8,
    pub pic_init_qp_minus26: i32,
    pub pic_init_qs_minus26: i32,
    pub chroma_qp_index_offset: i32,
    pub deblocking_filter_control_present_flag: bool,
    pub constrained_intra_pred_flag: bool,
    pub redundant_pic_cnt_present_flag: bool,
    pub transform_8x8_mode_flag: bool,
    /// PPS 自己带的量化矩阵（`None` = 沿用 SPS 的）
    pub scaling4x4: [Option<[u8; 16]>; 6],
    pub scaling8x8: [Option<[u8; 64]>; 2],
    pub second_chroma_qp_index_offset: i32,
}

impl Default for Pps {
    fn default() -> Self {
        Self {
            id: 0,
            sps_id: 0,
            entropy_coding_mode_flag: false,
            bottom_field_pic_order_in_frame_present_flag: false,
            num_slice_groups_minus1: 0,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            weighted_pred_flag: false,
            weighted_bipred_idc: 0,
            pic_init_qp_minus26: 0,
            pic_init_qs_minus26: 0,
            chroma_qp_index_offset: 0,
            deblocking_filter_control_present_flag: false,
            constrained_intra_pred_flag: false,
            redundant_pic_cnt_present_flag: false,
            transform_8x8_mode_flag: false,
            scaling4x4: [None; 6],
            scaling8x8: [None; 2],
            second_chroma_qp_index_offset: 0,
        }
    }
}

/// §7.2 的 `more_rbsp_data()`：给定 RBSP 与当前位位置，**后面还有没有真正的数据**。
///
/// 判法：`rbsp_trailing_bits` 是"一个 `1` + 补零到字节对齐"，所以从后往前找到的
/// 第一个 `1` 就是它的起点；当前位还没走到那儿就说明后面还有语法元素。
/// （与 [`crate::video::h264::cavlc`] 里那条同源 —— 那里用在切片数据上。）
fn more_rbsp_data(rbsp: &[u8], pos_bits: usize) -> bool {
    for (i, b) in rbsp.iter().enumerate().rev() {
        if *b != 0 {
            let last = i * 8 + (7 - b.trailing_zeros() as usize);
            return pos_bits < last;
        }
    }
    false
}

impl Pps {
    /// 解析一个 PPS（入参是**完整 NAL**）。
    ///
    /// 需要 SPS 的两个字段：`chroma_format_idc` 与 `transform_8x8_mode_flag`
    /// 一起决定"后面跟几张量化矩阵表"（7.3.2.2 的循环上界）。
    pub fn parse(nal: &[u8], sps: &Sps) -> Result<Pps> {
        if nal.len() < 2 {
            return Err(MediaError::Bitstream("PPS 太短".into()));
        }
        let rbsp = remove_emulation_prevention(&nal[1..]);
        let mut r = BitReader::new(&rbsp);
        let mut pps = Pps {
            id: r.read_ue()? as u8,
            sps_id: r.read_ue()? as u8,
            ..Pps::default()
        };
        pps.entropy_coding_mode_flag = r.read_bit()? == 1;
        pps.bottom_field_pic_order_in_frame_present_flag = r.read_bit()? == 1;
        pps.num_slice_groups_minus1 = r.read_ue()?;
        if pps.num_slice_groups_minus1 > 0 {
            // FMO 的 map 语法这里不完整实现：真要支持得连 slice_group_map 一起读，
            // 而 VA-API 的对应字段早被标成 deprecated（驱动也不认）—— 直接拒绝。
            return Err(MediaError::Unsupported(
                "H.264 用了 FMO（num_slice_groups_minus1 > 0），引擎不支持".into(),
            ));
        }
        pps.num_ref_idx_l0_default_active_minus1 = r.read_ue()?;
        pps.num_ref_idx_l1_default_active_minus1 = r.read_ue()?;
        pps.weighted_pred_flag = r.read_bit()? == 1;
        pps.weighted_bipred_idc = r.read_u32(2)? as u8;
        pps.pic_init_qp_minus26 = r.read_se()?;
        pps.pic_init_qs_minus26 = r.read_se()?;
        pps.chroma_qp_index_offset = r.read_se()?;
        // ⚠️ 规范（§7.4.2.2 的 `second_chroma_qp_index_offset`）：**这个语法元素
        // 不存在时推断为等于 `chroma_qp_index_offset`**。Baseline/Main 等 profile
        // 的 PPS 里根本没有它 —— 留成结构体默认值 0 就会让 **Cr 的 `QP′C` 差一截**，
        // 症状是"Cb 逐样本一致、Cr 整体偏几阶"（R=G=B 的内容还会完全看不出来）。
        pps.second_chroma_qp_index_offset = pps.chroma_qp_index_offset;
        pps.deblocking_filter_control_present_flag = r.read_bit()? == 1;
        pps.constrained_intra_pred_flag = r.read_bit()? == 1;
        pps.redundant_pic_cnt_present_flag = r.read_bit()? == 1;
        // 可选尾巴：**只有这几个 profile 的 PPS 才可能有**（§7.3.2.2）。
        // 判据是"还剩真正的数据"（`more_rbsp_data()`）—— 不能拿"剩余位数"凑，
        // 那样会把 `rbsp_trailing_bits` 当数据读进来。
        let high_profile = matches!(sps.profile_idc, 100 | 110 | 122 | 244 | 44);
        if high_profile && more_rbsp_data(&rbsp, r.position_bits()) {
            pps.transform_8x8_mode_flag = r.read_bit()? == 1;
            if r.read_bit()? == 1 {
                // 表的张数（7.3.2.2）：4x4 六张，8x8 只在开了 8x8 变换时才有；
                // 4:2:0 少两张（那两张属于 4:4:4）
                let chroma_tables = if sps.chroma_format_idc != 3 { 2 } else { 6 };
                let count = 6 + chroma_tables * pps.transform_8x8_mode_flag as usize;
                for i in 0..count {
                    if r.read_bit()? == 1 {
                        let size = if i < 6 { 16 } else { 64 };
                        let list = read_scaling_list(&mut r, size)?;
                        if i < 6 {
                            let mut out = [16u8; 16];
                            out.copy_from_slice(&list[..16]);
                            pps.scaling4x4[i] = Some(out);
                        } else if i < 8 {
                            pps.scaling8x8[i - 6] = Some(list);
                        }
                    }
                }
            }
            pps.second_chroma_qp_index_offset = r.read_se()?;
        }
        Ok(pps)
    }

    /// 这一版引擎不支持的构造。
    pub fn check_supported(&self) -> Result<()> {
        if self.num_slice_groups_minus1 > 0 {
            return Err(MediaError::Unsupported(
                "H.264 用了 FMO（num_slice_groups_minus1 > 0），引擎不支持".into(),
            ));
        }
        Ok(())
    }

    /// 送进硬件的**有效量化矩阵**（4x4 六张 + 8x8 两张）。
    ///
    /// 取值链（7.4.2.1.1.1）：PPS 自己带了就用 PPS 的 → 否则用 SPS 的 →
    /// 两处都没有就是缺省矩阵（Flat，全 16）。
    pub fn effective_scaling(&self, sps: &Sps) -> ([[u8; 16]; 6], [[u8; 64]; 2]) {
        let mut l4 = [[16u8; 16]; 6];
        for i in 0..6 {
            if let Some(m) = self.scaling4x4[i].or(sps.scaling4x4[i]) {
                l4[i] = m;
            }
        }
        let mut l8 = [[16u8; 64]; 2];
        for i in 0..2 {
            if let Some(m) = self.scaling8x8[i].or(sps.scaling8x8[i]) {
                l8[i] = m;
            }
        }
        (l4, l8)
    }

    /// 这个 PPS 到底带没带量化矩阵（VA-API 只在"带了"的时候才要那张 IQ 矩阵缓冲）。
    pub fn has_scaling_matrix(&self, sps: &Sps) -> bool {
        self.scaling4x4.iter().any(|m| m.is_some())
            || self.scaling8x8.iter().any(|m| m.is_some())
            || sps.scaling4x4.iter().any(|m| m.is_some())
            || sps.scaling8x8.iter().any(|m| m.is_some())
    }
}

// ── 切片头 ───────────────────────────────────────────────────────────────────

/// 参考帧列表重排指令（7.3.3.1）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefListMod {
    /// `modification_of_pic_nums_idc` 0/1：相对当前帧往前/往后数第 `abs_diff + 1` 帧
    ShortTerm { negative: bool, abs_diff: u32 },
    /// `modification_of_pic_nums_idc` 2：长期参考帧（引擎不支持，留着是为了把位读完）
    LongTerm { long_term_pic_num: u32 },
}

/// MMCO 指令（7.3.3.3）—— 引擎不支持，留着是为了遇到时报一句准确的话。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mmco {
    /// 记下 `difference_of_pic_nums_minus1`（1/3）
    MarkUnused { difference_of_pic_nums_minus1: u32 },
    /// 长期参考帧相关（2/3/4/6）
    LongTerm { code: u32 },
}

/// 显式加权预测表（7.3.3.2）。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PredWeight {
    pub luma_log2_weight_denom: u32,
    pub chroma_log2_weight_denom: u32,
    pub l0: Vec<WeightEntry>,
    pub l1: Vec<WeightEntry>,
}

/// 单个参考帧的权重：`None` 表示"没显式给"，按缺省（权重 `1 << denom`、偏移 0）用。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WeightEntry {
    pub luma: Option<(i32, i32)>,
    pub chroma: Option<[(i32, i32); 2]>,
}

/// 参考帧标记（7.3.3.3）：IDR 与非 IDR 的字段完全不同。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DecRefPicMarking {
    /// IDR：`no_output_of_prior_pics_flag`
    pub no_output_of_prior_pics: bool,
    /// IDR：`long_term_reference_flag`
    pub long_term_reference: bool,
    /// 非 IDR：`adaptive_ref_pic_marking_mode_flag`（= 有 MMCO 跟在后头）
    pub adaptive: bool,
    pub ops: Vec<Mmco>,
}

/// 切片头（只到 `slice_data()` 之前）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceHeader {
    pub nal_type: u8,
    pub nal_ref_idc: u8,
    pub idr: bool,
    pub first_mb_in_slice: u32,
    /// 原始 `slice_type`（0..9；5..9 表示"整张图都是这一种"）
    pub slice_type: u8,
    pub pps_id: u8,
    pub frame_num: u32,
    pub field_pic_flag: bool,
    pub bottom_field_flag: bool,
    pub idr_pic_id: u32,
    pub pic_order_cnt_lsb: u32,
    pub delta_pic_order_cnt_bottom: i32,
    pub delta_pic_order_cnt: [i32; 2],
    pub redundant_pic_cnt: u32,
    pub direct_spatial_mv_pred_flag: bool,
    /// 生效的 `num_ref_idx_l0_active`（已加回 1）
    pub num_ref_idx_l0_active: u32,
    pub num_ref_idx_l1_active: u32,
    pub l0_mod: Vec<RefListMod>,
    pub l1_mod: Vec<RefListMod>,
    pub pred_weight: Option<PredWeight>,
    pub dec_ref_pic_marking: DecRefPicMarking,
    pub cabac_init_idc: u32,
    pub slice_qp_delta: i32,
    pub disable_deblocking_filter_idc: u32,
    pub slice_alpha_c0_offset_div2: i32,
    pub slice_beta_offset_div2: i32,
    /// `slice_data()` 的起始位偏移：**去防竞争字节后、自 NAL 头字节起算**。
    ///
    /// 这正是 VA-API `slice_data_bit_offset` 要的口径（va.h 的注释写得很死）；
    /// 而送进驱动的**切片数据缓冲仍是原样带防竞争字节的字节流** —— 两者一个按
    /// 去转义后的位算、一个按原始字节给，别弄混。
    pub slice_data_bit_offset: usize,
}

impl SliceHeader {
    /// `slice_type mod 5`：0=P 1=B 2=I 3=SP 4=SI
    pub fn kind(&self) -> u8 {
        self.slice_type % 5
    }

    pub fn is_i(&self) -> bool {
        self.kind() == 2
    }

    pub fn is_p(&self) -> bool {
        self.kind() == 0
    }

    pub fn is_b(&self) -> bool {
        self.kind() == 1
    }

    pub fn is_reference(&self) -> bool {
        self.nal_ref_idc != 0
    }

    /// 这一版引擎不支持的构造。
    pub fn check_supported(&self) -> Result<()> {
        if self.field_pic_flag {
            return Err(MediaError::Unsupported(
                "H.264 用了场编码（field_pic_flag=1），引擎只支持逐行帧".into(),
            ));
        }
        if self.kind() >= 3 {
            return Err(MediaError::Unsupported(
                "H.264 用了 SP/SI 切片，引擎不支持".into(),
            ));
        }
        if self.redundant_pic_cnt != 0 {
            return Err(MediaError::Unsupported(
                "H.264 用了冗余切片（redundant_pic_cnt > 0），引擎不支持".into(),
            ));
        }
        if !self.dec_ref_pic_marking.ops.is_empty() {
            return Err(MediaError::Unsupported(
                "H.264 用了自适应参考帧标记（MMCO），引擎只支持滑动窗口标记".into(),
            ));
        }
        if self.dec_ref_pic_marking.long_term_reference {
            return Err(MediaError::Unsupported(
                "H.264 把 IDR 标成了长期参考帧，引擎只支持短期参考".into(),
            ));
        }
        Ok(())
    }
}

/// 解析切片头（入参是**完整 NAL**，需要它所属的 SPS/PPS 才知道字段位宽）。
pub fn parse_slice_header(nal: &[u8], sps: &Sps, pps: &Pps) -> Result<SliceHeader> {
    if nal.len() < 2 {
        return Err(MediaError::Bitstream("切片 NAL 太短".into()));
    }
    let nal_type = nal[0] & 0x1F;
    if !is_vcl(nal_type) {
        return Err(MediaError::Bitstream(format!(
            "NAL 类型 {nal_type} 不是切片（要 1..5）"
        )));
    }
    // 关键：把"NAL 头 + 去转义后的负载"拼成一条流来读 —— 这样 `position_bits()`
    // 天然就是"自 NAL 头起算、且已去掉防竞争字节"的偏移（VA-API 要的口径）。
    let mut raw = Vec::with_capacity(nal.len());
    raw.push(nal[0]);
    raw.extend_from_slice(&remove_emulation_prevention(&nal[1..]));
    let mut r = BitReader::new(&raw);
    let _header = r.read_u32(8)?;
    let nal_ref_idc = (nal[0] >> 5) & 0x03;
    let idr = nal_type == NAL_SLICE_IDR;

    let first_mb_in_slice = r.read_ue()?;
    let slice_type = r.read_ue()?;
    if slice_type > 9 {
        return Err(MediaError::Bitstream(format!(
            "切片头的 slice_type = {slice_type} 不在规范里"
        )));
    }
    let pps_id = u8::try_from(r.read_ue()?)
        .map_err(|_| MediaError::Bitstream("切片头的 pic_parameter_set_id 越界".into()))?;
    let mut sh = SliceHeader {
        nal_type,
        nal_ref_idc,
        idr,
        first_mb_in_slice,
        slice_type: slice_type as u8,
        pps_id,
        frame_num: 0,
        field_pic_flag: false,
        bottom_field_flag: false,
        idr_pic_id: 0,
        pic_order_cnt_lsb: 0,
        delta_pic_order_cnt_bottom: 0,
        delta_pic_order_cnt: [0, 0],
        redundant_pic_cnt: 0,
        direct_spatial_mv_pred_flag: false,
        num_ref_idx_l0_active: 1,
        num_ref_idx_l1_active: 1,
        l0_mod: Vec::new(),
        l1_mod: Vec::new(),
        pred_weight: None,
        dec_ref_pic_marking: DecRefPicMarking::default(),
        cabac_init_idc: 0,
        slice_qp_delta: 0,
        disable_deblocking_filter_idc: 0,
        slice_alpha_c0_offset_div2: 0,
        slice_beta_offset_div2: 0,
        slice_data_bit_offset: 0,
    };

    sh.frame_num = r.read_u32(sps.log2_max_frame_num)?;
    if !sps.frame_mbs_only_flag {
        sh.field_pic_flag = r.read_bit()? == 1;
        if sh.field_pic_flag {
            sh.bottom_field_flag = r.read_bit()? == 1;
        }
    }
    if idr {
        sh.idr_pic_id = r.read_ue()?;
    }
    match sps.pic_order_cnt_type {
        0 => {
            sh.pic_order_cnt_lsb = r.read_u32(sps.log2_max_pic_order_cnt_lsb)?;
            if pps.bottom_field_pic_order_in_frame_present_flag && !sh.field_pic_flag {
                sh.delta_pic_order_cnt_bottom = r.read_se()?;
            }
        }
        1 if !sps.delta_pic_order_always_zero_flag => {
            sh.delta_pic_order_cnt[0] = r.read_se()?;
            if pps.bottom_field_pic_order_in_frame_present_flag && !sh.field_pic_flag {
                sh.delta_pic_order_cnt[1] = r.read_se()?;
            }
        }
        _ => {}
    }
    if pps.redundant_pic_cnt_present_flag {
        sh.redundant_pic_cnt = r.read_ue()?;
    }
    if sh.is_b() {
        sh.direct_spatial_mv_pred_flag = r.read_bit()? == 1;
    }
    if sh.is_p() || sh.is_b() {
        let override_flag = r.read_bit()? == 1;
        if override_flag {
            sh.num_ref_idx_l0_active = read_ue_plus1(&mut r, "num_ref_idx_l0_active_minus1")?;
            if sh.is_b() {
                sh.num_ref_idx_l1_active = read_ue_plus1(&mut r, "num_ref_idx_l1_active_minus1")?;
            }
        } else {
            sh.num_ref_idx_l0_active = pps.num_ref_idx_l0_default_active_minus1 + 1;
            sh.num_ref_idx_l1_active = pps.num_ref_idx_l1_default_active_minus1 + 1;
        }
    }
    // 参考帧列表重排（I/SI 没有）
    if sh.kind() <= 1 {
        if r.read_bit()? == 1 {
            sh.l0_mod = read_ref_list_mod(&mut r)?;
        }
        if sh.is_b() && r.read_bit()? == 1 {
            sh.l1_mod = read_ref_list_mod(&mut r)?;
        }
    }
    if (pps.weighted_pred_flag && (sh.is_p() || sh.kind() == 3))
        || (pps.weighted_bipred_idc == 1 && sh.is_b())
    {
        sh.pred_weight = Some(read_pred_weight(
            &mut r,
            sh.num_ref_idx_l0_active,
            sh.num_ref_idx_l1_active,
            sh.is_b(),
        )?);
    }
    if sh.is_reference() {
        if idr {
            sh.dec_ref_pic_marking.no_output_of_prior_pics = r.read_bit()? == 1;
            sh.dec_ref_pic_marking.long_term_reference = r.read_bit()? == 1;
        } else {
            sh.dec_ref_pic_marking.adaptive = r.read_bit()? == 1;
            if sh.dec_ref_pic_marking.adaptive {
                sh.dec_ref_pic_marking.ops = read_mmco_ops(&mut r)?;
            }
        }
    }
    if pps.entropy_coding_mode_flag && !sh.is_i() && sh.kind() != 4 {
        sh.cabac_init_idc = r.read_ue()?;
    }
    sh.slice_qp_delta = r.read_se()?;
    if sh.kind() == 3 {
        let _sp_for_switch_flag = r.read_bit()?;
    }
    if sh.kind() >= 3 {
        let _slice_qs_delta = r.read_se()?;
    }
    if pps.deblocking_filter_control_present_flag {
        sh.disable_deblocking_filter_idc = r.read_ue()?;
        if sh.disable_deblocking_filter_idc != 1 {
            sh.slice_alpha_c0_offset_div2 = r.read_se()?;
            sh.slice_beta_offset_div2 = r.read_se()?;
        }
    }
    if pps.num_slice_groups_minus1 > 0 {
        // check_supported 已经拒了 FMO，这里防御性兜底（真进来说明上游漏了检查）
        return Err(MediaError::Unsupported(
            "H.264 用了 FMO（num_slice_groups_minus1 > 0），引擎不支持".into(),
        ));
    }
    sh.slice_data_bit_offset = r.position_bits();
    Ok(sh)
}

fn read_ref_list_mod(r: &mut BitReader) -> Result<Vec<RefListMod>> {
    let mut out = Vec::new();
    loop {
        let idc = r.read_ue()?;
        match idc {
            0 | 1 => out.push(RefListMod::ShortTerm {
                negative: idc == 0,
                abs_diff: r.read_ue()?,
            }),
            2 => out.push(RefListMod::LongTerm {
                long_term_pic_num: r.read_ue()?,
            }),
            3 => break,
            other => {
                return Err(MediaError::Bitstream(format!(
                    "参考帧列表重排的 modification_of_pic_nums_idc = {other} 不在规范里"
                )))
            }
        }
    }
    Ok(out)
}

fn read_mmco_ops(r: &mut BitReader) -> Result<Vec<Mmco>> {
    let mut out = Vec::new();
    loop {
        let code = r.read_ue()?;
        if code == 0 {
            break;
        }
        match code {
            1 | 3 => out.push(Mmco::MarkUnused {
                difference_of_pic_nums_minus1: r.read_ue()?,
            }),
            2 => {
                let _long_term_pic_num = r.read_ue()?;
                out.push(Mmco::LongTerm { code });
            }
            4 => {
                let _max_long_term_frame_idx_plus1 = r.read_ue()?;
                out.push(Mmco::LongTerm { code });
            }
            5 => out.push(Mmco::LongTerm { code }),
            6 => {
                let _long_term_frame_idx = r.read_ue()?;
                out.push(Mmco::LongTerm { code });
            }
            other => {
                return Err(MediaError::Bitstream(format!(
                    "MMCO 指令 {other} 不在规范里"
                )))
            }
        }
    }
    Ok(out)
}

fn read_pred_weight(
    r: &mut BitReader,
    l0_active: u32,
    l1_active: u32,
    is_b: bool,
) -> Result<PredWeight> {
    let mut w = PredWeight {
        luma_log2_weight_denom: r.read_ue()?,
        chroma_log2_weight_denom: r.read_ue()?,
        ..PredWeight::default()
    };
    // 4:2:0 一定有色度分量（check_supported 保证），所以 chroma_* 一定在
    for _ in 0..l0_active {
        w.l0.push(read_weight_entry(r)?);
    }
    if is_b {
        for _ in 0..l1_active {
            w.l1.push(read_weight_entry(r)?);
        }
    }
    Ok(w)
}

fn read_weight_entry(r: &mut BitReader) -> Result<WeightEntry> {
    let mut e = WeightEntry::default();
    if r.read_bit()? == 1 {
        e.luma = Some((r.read_se()?, r.read_se()?));
    }
    if r.read_bit()? == 1 {
        let mut c = [(0, 0); 2];
        for item in c.iter_mut() {
            *item = (r.read_se()?, r.read_se()?);
        }
        e.chroma = Some(c);
    }
    Ok(e)
}

// ── POC（画面顺序计数，8.2.1） ───────────────────────────────────────────────

/// POC 推导的状态机。
///
/// **一张图只算一次**：多 slice 的图重复调用会把 `prev_*` 状态带偏（规范里
/// `prevPicOrderCnt*` 是"上一个参考图"的量，与 slice 无关）。
#[derive(Debug, Clone, Default)]
pub struct PocState {
    prev_poc_msb: i32,
    prev_poc_lsb: u32,
    prev_frame_num: u32,
    prev_frame_num_offset: u32,
}

impl PocState {
    pub fn new() -> Self {
        Self::default()
    }

    /// IDR 之后状态清零（8.2.1.1/2/3 都以"IDR 则 prev* = 0"开头）。
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// 算这一张图的 POC（按 `sps.pic_order_cnt_type` 走三条路之一）。
    pub fn compute(&mut self, sps: &Sps, sh: &SliceHeader) -> Result<i32> {
        let max_frame_num = 1u32 << sps.log2_max_frame_num;
        if sh.idr {
            self.reset();
        }
        match sps.pic_order_cnt_type {
            0 => {
                let max_lsb = 1i32 << sps.log2_max_pic_order_cnt_lsb;
                let lsb = sh.pic_order_cnt_lsb as i32;
                if !sh.idr {
                    let prev_lsb = self.prev_poc_lsb as i32;
                    if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
                        self.prev_poc_msb += max_lsb;
                    } else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
                        self.prev_poc_msb -= max_lsb;
                    }
                }
                let top = self.prev_poc_msb + lsb;
                let bottom = top + sh.delta_pic_order_cnt_bottom;
                if sh.is_reference() {
                    self.prev_poc_lsb = sh.pic_order_cnt_lsb;
                    self.prev_poc_msb = top - lsb;
                }
                Ok(top.min(bottom))
            }
            1 => {
                let frame_num_offset = self.frame_num_offset(sh, max_frame_num);
                let cycle = sps.offset_for_ref_frame.len() as u32;
                let mut abs_frame_num = if cycle != 0 {
                    frame_num_offset + sh.frame_num
                } else {
                    0
                };
                if !sh.is_reference() && abs_frame_num > 0 {
                    abs_frame_num -= 1;
                }
                let mut expected = 0i32;
                if abs_frame_num > 0 {
                    let cnt = (abs_frame_num - 1) / cycle;
                    let in_cycle = (abs_frame_num - 1) % cycle;
                    let per_cycle: i32 = sps.offset_for_ref_frame.iter().sum();
                    expected = cnt as i32 * per_cycle;
                    for i in 0..=in_cycle as usize {
                        expected += sps.offset_for_ref_frame[i];
                    }
                }
                if !sh.is_reference() {
                    expected += sps.offset_for_non_ref_pic;
                }
                let top = expected + sh.delta_pic_order_cnt[0];
                let bottom = top + sps.offset_for_top_to_bottom_field;
                self.update_frame_num_state(sh, frame_num_offset);
                Ok(top.min(bottom))
            }
            2 => {
                let frame_num_offset = self.frame_num_offset(sh, max_frame_num);
                let temp = 2 * (frame_num_offset as i32 + sh.frame_num as i32)
                    - if sh.is_reference() { 0 } else { 1 };
                self.update_frame_num_state(sh, frame_num_offset);
                Ok(temp)
            }
            other => Err(MediaError::Bitstream(format!(
                "POC 类型 {other} 不在规范里"
            ))),
        }
    }

    fn frame_num_offset(&self, sh: &SliceHeader, max_frame_num: u32) -> u32 {
        if sh.idr {
            0
        } else if self.prev_frame_num > sh.frame_num {
            self.prev_frame_num_offset + max_frame_num
        } else {
            self.prev_frame_num_offset
        }
    }

    fn update_frame_num_state(&mut self, sh: &SliceHeader, frame_num_offset: u32) {
        if sh.is_reference() {
            self.prev_frame_num = sh.frame_num;
            self.prev_frame_num_offset = frame_num_offset;
        }
    }
}

// ── DPB 与参考帧列表（8.2.4 / 8.2.5） ────────────────────────────────────────

/// DPB 里的一个**短期参考帧**。
///
/// `slot` 是后端的句柄（VA-API 是 surfacce id），本模块不解释它。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefPic {
    pub slot: u32,
    pub frame_num: u32,
    pub poc: i32,
}

/// 参考帧列表（DPB）—— 只支持**滑动窗口标记**：没有 MMCO、没有长期参考。
///
/// 这是"硬件负责参考、我们负责顺序"的分工：参考帧该不该留着由这份表说话，
/// 具体怎么拿它做帧间预测是驱动/系统解码器的事。表错了（留早了、删晚了），
/// 驱动要么拿到已释放的面，要么被喂不存在的参考帧 —— 都会静默出错帧。
#[derive(Debug, Clone, Default)]
pub struct Dpb {
    refs: Vec<RefPic>,
}

impl Dpb {
    pub fn clear(&mut self) {
        self.refs.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.refs.is_empty()
    }

    pub fn len(&self) -> usize {
        self.refs.len()
    }

    pub fn refs(&self) -> &[RefPic] {
        &self.refs
    }

    /// 滑动窗口标记（8.2.5.3）：参考帧已满就淘汰 `FrameNumWrap` 最小的那个
    /// （同值再比 POC），返回被淘汰的帧（调用方据此决定能不能复用它的面）。
    ///
    /// 在**加入新参考帧之前**调用：`curr_frame_num` 是当前图的 `frame_num`
    /// （FrameNumWrap 是"相对当前图"的编号，这个参数不能省）。
    pub fn mark_sliding_window(
        &mut self,
        max_num_ref_frames: u32,
        curr_frame_num: u32,
        max_frame_num: u32,
    ) -> Vec<RefPic> {
        let cap = max_num_ref_frames.max(1) as usize;
        let mut removed = Vec::new();
        while self.refs.len() >= cap {
            let Some(idx) = (0..self.refs.len()).min_by_key(|&i| {
                let r = &self.refs[i];
                (frame_num_wrap(r.frame_num, curr_frame_num, max_frame_num), r.poc)
            }) else {
                break;
            };
            removed.push(self.refs.remove(idx));
        }
        removed
    }

    pub fn push(&mut self, pic: RefPic) {
        self.refs.push(pic);
    }

    /// 默认参考帧列表（8.2.4.1：帧、无长期参考、未用过 MMCO 的情形）：
    /// 短期参考帧按 `FrameNumWrap` 降序，同值再按 POC 降序。
    pub fn default_list(&self, list1: bool, curr_frame_num: u32, max_frame_num: u32) -> Vec<RefPic> {
        let mut list = self.refs.clone();
        list.sort_by_key(|r| {
            let wrap = frame_num_wrap(r.frame_num, curr_frame_num, max_frame_num);
            if list1 {
                (wrap, r.poc) // list1：升序
            } else {
                (-wrap, -r.poc) // list0：降序
            }
        });
        list
    }

    /// 最终参考帧列表：默认顺序 + 切片头里的重排指令（8.2.4.3.1）。
    ///
    /// `list` 里的**重复项是允许的**（同一条参考帧可以占两个位置，规范如此），
    /// 所以重排是"插入 + 丢掉最后一位"，不是"移动去重"。
    pub fn ref_list(
        &self,
        list1: bool,
        active: u32,
        mods: &[RefListMod],
        curr_frame_num: u32,
        max_frame_num: u32,
    ) -> Result<Vec<RefPic>> {
        let mut list = self.default_list(list1, curr_frame_num, max_frame_num);
        let active = active.max(1) as usize;
        if list.len() < active {
            return Err(MediaError::Bitstream(format!(
                "参考帧不够：切片要 {} 个，DPB 里只有 {} 个",
                active,
                list.len()
            )));
        }
        list.truncate(active);
        let curr = curr_frame_num as i32;
        let max_pic_num = max_frame_num as i32;
        let mut pred = curr;
        let mut idx = 0usize;
        for m in mods {
            match *m {
                RefListMod::ShortTerm { negative, abs_diff } => {
                    let step = abs_diff as i64 + 1;
                    let no_wrap = if negative {
                        pred as i64 - step
                    } else {
                        pred as i64 + step
                    };
                    pred = no_wrap as i32;
                    let pic_num = if no_wrap > curr as i64 {
                        (no_wrap - max_pic_num as i64) as i32
                    } else if no_wrap < 0 {
                        (no_wrap + max_pic_num as i64) as i32
                    } else {
                        no_wrap as i32
                    };
                    let target = self
                        .refs
                        .iter()
                        .find(|r| frame_num_wrap(r.frame_num, curr_frame_num, max_frame_num) == pic_num)
                        .copied()
                        .ok_or_else(|| {
                            MediaError::Bitstream(format!(
                                "参考帧列表重排指向了不存在的帧（PicNum {pic_num}）"
                            ))
                        })?;
                    let at = idx.min(list.len().saturating_sub(1));
                    list.pop();
                    list.insert(at, target);
                    idx = at + 1;
                }
                RefListMod::LongTerm { .. } => {
                    return Err(MediaError::Unsupported(
                        "H.264 的参考帧列表重排里引用了长期参考帧，引擎不支持".into(),
                    ));
                }
            }
        }
        Ok(list)
    }
}

/// `FrameNumWrap`（8.2.2.1）：把绝对帧号折成"相对当前图"的编号。
fn frame_num_wrap(frame_num: u32, curr_frame_num: u32, max_frame_num: u32) -> i32 {
    if frame_num > curr_frame_num {
        frame_num as i32 - max_frame_num as i32
    } else {
        frame_num as i32
    }
}

// ── avcC 的"已解析"形态（三个后端开解码器的共同起点） ────────────────────────

/// 一个 H.264 流的初始化配置：`avcC` 原文 + 第一组参数集的解析结果。
///
/// 为什么在这里就解析并校验：三个后端都需要"编码尺寸/重排深度/切片头字段位宽"，
/// 而**不支持的构造要在这里就说清楚** —— 等到驱动返回一个裸错误码再说，
/// 用户看到的是一句看不懂的 HRESULT/VAStatus。
#[derive(Debug, Clone)]
pub struct Config {
    pub avcc: Vec<u8>,
    pub nal_length_size: usize,
    pub sps: Sps,
    pub pps: Pps,
}

impl Config {
    pub fn parse(avcc: &[u8]) -> Result<Config> {
        let (sps_list, pps_list) = parse_avcc(avcc)?;
        let sps_nal = sps_list
            .first()
            .ok_or_else(|| MediaError::Bitstream("avcC 里没有 SPS".into()))?;
        let sps = Sps::parse(sps_nal)?;
        sps.check_supported()?;
        let pps_nal = pps_list
            .first()
            .ok_or_else(|| MediaError::Bitstream("avcC 里没有 PPS".into()))?;
        let pps = Pps::parse(pps_nal, &sps)?;
        pps.check_supported()?;
        if pps.sps_id != sps.id {
            return Err(MediaError::Bitstream(format!(
                "avcC 里第一组参数集对不上（PPS 指向 SPS {}，拿到的是 SPS {}）",
                pps.sps_id, sps.id
            )));
        }
        Ok(Config {
            avcc: avcc.to_vec(),
            nal_length_size: nal_length_size(avcc),
            sps,
            pps,
        })
    }
}

/// 测试用的共享夹具（`linux.rs` / `windows.rs` 的测试也要同一组参数集）。
///
/// 放在这里而不是各测各的：三个后端的测试都要"一组能过 `Config::parse` 的
/// avcC"，各造一份迟早会出现"某一家的夹具根本不是合法码流"的假绿灯。
#[cfg(test)]
pub(crate) mod tests_support {
    use super::*;

    /// 按位写（大端位序），**并记录位偏移** —— `slice_data_bit_offset` 的断言
    /// 要拿"写位流时的真实位置"当基准，不能靠人肉数位。
    #[derive(Default)]
    pub(crate) struct Bits {
        pub(crate) bytes: Vec<u8>,
        pub(crate) pos: usize,
    }

    impl Bits {
        pub(crate) fn f(&mut self, value: u32, bits: u32) -> &mut Self {
            for i in (0..bits).rev() {
                let b = ((value >> i) & 1) as u8;
                if self.pos % 8 == 0 {
                    self.bytes.push(0);
                }
                let last = self.bytes.len() - 1;
                self.bytes[last] |= b << (7 - (self.pos % 8));
                self.pos += 1;
            }
            self
        }

        pub(crate) fn bit(&mut self, v: bool) -> &mut Self {
            self.f(v as u32, 1)
        }

        pub(crate) fn ue(&mut self, v: u32) -> &mut Self {
            let m = v as u64 + 1;
            let bits = 64 - m.leading_zeros();
            self.f(0, bits - 1).f(m as u32, bits)
        }

        pub(crate) fn se(&mut self, v: i32) -> &mut Self {
            let k = if v <= 0 { (-2 * v) as u32 } else { (2 * v - 1) as u32 };
            self.ue(k)
        }

        pub(crate) fn finish(&self) -> Vec<u8> {
            self.bytes.clone()
        }
    }

    /// 造一个逐行帧 SPS（8 位 4:2:0）的公共部分。
    pub(crate) fn build_sps(w: &mut Bits, profile_idc: u8, width_mbs: u32, height_map_units: u32) {
        w.f(profile_idc as u32, 8);
        w.f(0, 8); // constraint flags
        w.f(30, 8); // level_idc = 3.0
        w.ue(0); // sps id
        if is_high_profile(profile_idc) {
            w.ue(1); // chroma_format_idc = 4:2:0
            w.ue(0); // bit_depth_luma_minus8
            w.ue(0); // bit_depth_chroma_minus8
            w.bit(false); // qpprime_y_zero_transform_bypass_flag
            w.bit(false); // seq_scaling_matrix_present_flag
        }
        w.ue(0); // log2_max_frame_num_minus4 → 4
        w.ue(0); // pic_order_cnt_type = 0
        w.ue(0); // log2_max_pic_order_cnt_lsb_minus4 → 4
        w.ue(2); // max_num_ref_frames
        w.bit(false); // gaps_in_frame_num_value_allowed_flag
        w.ue(width_mbs - 1);
        w.ue(height_map_units - 1);
        w.bit(true); // frame_mbs_only_flag
        w.bit(true); // direct_8x8_inference_flag
        w.bit(false); // frame_cropping_flag
        w.bit(false); // vui_parameters_present_flag
        w.f(0b1000_0000, 8); // rbsp_trailing_bits（对齐一下，方便直接当 NAL 用）
    }

    /// 造一个最简 PPS（无重排、无加权、无 8x8），**不含 NAL 头**。
    pub(crate) fn build_pps(w: &mut Bits) {
        w.ue(0); // pps id
        w.ue(0); // sps id
        w.bit(false); // entropy_coding_mode_flag
        w.bit(false); // bottom_field_pic_order_in_frame_present_flag
        w.ue(0); // num_slice_groups_minus1
        w.ue(0); // num_ref_idx_l0_default_active_minus1
        w.ue(0); // num_ref_idx_l1_default_active_minus1
        w.bit(false); // weighted_pred_flag
        w.f(0, 2); // weighted_bipred_idc
        w.se(0); // pic_init_qp_minus26
        w.se(0); // pic_init_qs_minus26
        w.se(0); // chroma_qp_index_offset
        w.bit(true); // deblocking_filter_control_present_flag
        w.bit(false); // constrained_intra_pred_flag
        w.bit(false); // redundant_pic_cnt_present_flag
        w.f(0b1000_0000, 8); // 收尾（解析器只要求还有位可读）
    }

    /// 一组"字段都最普通"的 SPS/PPS（640×368 编码、无裁剪、VUI 也不写）。
    fn sps_pps_pair() -> (Vec<u8>, Vec<u8>) {
        let mut sw = Bits::default();
        sw.f(0x67, 8);
        build_sps(&mut sw, 77, 40, 23);
        let mut pw = Bits::default();
        pw.f(0x68, 8);
        build_pps(&mut pw);
        (sw.finish(), pw.finish())
    }

    /// 一个能过 [`Config::parse`] 的 avcC（640×368 编码尺寸）。
    pub(crate) fn mini_avcc() -> Vec<u8> {
        let (sps, pps) = sps_pps_pair();
        let mut avcc = vec![1u8, 77, 0x00, 30, 0xFF, 0xE1];
        avcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(&sps);
        avcc.push(0x01);
        avcc.extend_from_slice(&(pps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(&pps);
        avcc
    }

    /// 一个最简 I 切片头（字段随意，够测试当底子改）。
    pub(crate) fn minimal_slice() -> SliceHeader {
        let sps = Sps {
            log2_max_frame_num: 4,
            log2_max_pic_order_cnt_lsb: 4,
            ..Sps::default()
        };
        let mut w = Bits::default();
        w.f(0x41, 8); // nal_ref_idc = 2, type = 1
        w.ue(0); // first_mb_in_slice
        w.ue(2); // slice_type = I
        w.ue(0); // pps id
        w.f(0, 4); // frame_num
        w.f(0, 4); // pic_order_cnt_lsb
        w.bit(false); // adaptive_ref_pic_marking_mode_flag
        w.se(0); // slice_qp_delta
        parse_slice_header(&w.finish(), &sps, &Pps::default()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::*;
    use super::*;

    #[test]
    fn parses_sps_and_pps_from_avcc() {
        let mut avcc = vec![1u8, 0x64, 0x00, 0x28, 0xFF, 0xE1];
        avcc.extend_from_slice(&3u16.to_be_bytes());
        avcc.extend_from_slice(&[0x67, 0x64, 0x00]);
        avcc.push(0x01);
        avcc.extend_from_slice(&2u16.to_be_bytes());
        avcc.extend_from_slice(&[0x68, 0xEE]);
        let (sps, pps) = parse_avcc(&avcc).unwrap();
        assert_eq!(sps, vec![vec![0x67, 0x64, 0x00]]);
        assert_eq!(pps, vec![vec![0x68, 0xEE]]);
        assert_eq!(nal_length_size(&avcc), 4);
    }

    #[test]
    fn rejects_garbage_avcc() {
        assert!(parse_avcc(&[]).is_err());
        assert!(parse_avcc(&[0u8; 8]).is_err(), "version 不是 1 要拒绝");
        let bad = vec![1u8, 0x64, 0x00, 0x28, 0xFF, 0xE1, 0xFF, 0xFF, 0x67];
        assert!(parse_avcc(&bad).is_err(), "长度越界要拒绝");
        // 1 字节长度前缀的 avcC
        let one_byte = vec![1u8, 0x64, 0x00, 0x28, 0xFC, 0xE1, 0x00, 0x03, 0x67, 0x64, 0x00];
        assert_eq!(nal_length_size(&one_byte), 1);
    }

    #[test]
    fn sps_roundtrip_gives_display_size() {
        // 640×368 编码 / 裁剪到 640×360（360 = 22 个宏块行 352 + 半行 ⇒ 用 cropping 收）
        let mut w2 = Bits::default();
        w2.f(0x67, 8);
        w2.f(77, 8);
        w2.f(0, 8);
        w2.f(30, 8);
        w2.ue(0);
        w2.ue(0); // log2_max_frame_num_minus4
        w2.ue(0); // poc type
        w2.ue(0); // log2_max_pic_order_cnt_lsb_minus4
        w2.ue(2); // max_num_ref_frames
        w2.bit(false);
        w2.ue(39); // width_mbs 40
        w2.ue(22); // height_map_units 23 → 368
        w2.bit(true); // frame_mbs_only
        w2.bit(true); // direct_8x8
        w2.bit(true); // frame_cropping_flag
        w2.ue(0); // left
        w2.ue(0); // right
        w2.ue(0); // top
        w2.ue(4); // bottom：4 个色度单位 = 8 行 → 368 - 8 = 360
        w2.bit(false); // vui
        w2.f(0b1000_0000, 8);
        let sps = Sps::parse(&w2.finish()).unwrap();
        assert_eq!(sps.profile_idc, 77);
        assert_eq!(sps.coded_width(), 640);
        assert_eq!(sps.coded_height(), 368);
        assert_eq!(sps.width(), 640);
        assert_eq!(sps.height(), 360);
        assert_eq!(sps.max_num_ref_frames, 2);
        assert_eq!(sps.pic_order_cnt_type, 0);
        assert_eq!(sps.log2_max_frame_num, 4);
        sps.check_supported().unwrap();
    }

    #[test]
    fn sps_high_profile_reads_vui_restriction() {
        let mut w = Bits::default();
        w.f(0x67, 8);
        w.f(100, 8); // High
        w.f(0, 8);
        w.f(31, 8);
        w.ue(0);
        w.ue(1); // chroma_format_idc
        w.ue(0); // bit_depth_luma_minus8
        w.ue(0); // bit_depth_chroma_minus8
        w.bit(false); // qpprime
        w.bit(true); // seq_scaling_matrix_present_flag
        // 第 0 张（4x4）用缺省（nextScale 直接归零）
        w.bit(true);
        w.se(-8);
        // 第 1 张：显式给出 16 个 delta（全 0 → 值保持 8/16 的递增序列）
        w.bit(true);
        for _ in 0..16 {
            w.se(0);
        }
        for _ in 2..8 {
            w.bit(false); // 其余表不在 SPS 里给
        }
        w.ue(0); // log2_max_frame_num_minus4
        w.ue(2); // pic_order_cnt_type = 2
        w.ue(4); // max_num_ref_frames
        w.bit(false); // gaps
        w.ue(19); // width 320
        w.ue(17); // height_map_units 18 → 288
        w.bit(true); // frame_mbs_only
        w.bit(true); // direct_8x8
        w.bit(false); // cropping
        w.bit(true); // vui_parameters_present_flag
        w.bit(false); // aspect_ratio_info_present
        w.bit(false); // overscan
        w.bit(false); // video_signal_type
        w.bit(false); // chroma_loc
        w.bit(true); // timing_info_present
        w.f(1001, 32);
        w.f(60000, 32);
        w.bit(true); // fixed_frame_rate
        w.bit(false); // nal_hrd
        w.bit(false); // vcl_hrd
        w.bit(false); // pic_struct_present
        w.bit(true); // bitstream_restriction
        w.bit(true); // motion_vectors_over_pic_boundaries
        w.ue(0); // max_bytes_per_pic_denom
        w.ue(0); // max_bits_per_mb_denom
        w.ue(8); // log2_max_mv_length_horizontal
        w.ue(8); // log2_max_mv_length_vertical
        w.ue(2); // max_num_reorder_frames
        w.ue(4); // max_dec_frame_buffering
        w.f(0b1000_0000, 8);
        let sps = Sps::parse(&w.finish()).unwrap();
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.pic_order_cnt_type, 2);
        assert_eq!(sps.scaling4x4[0], Some([16u8; 16]), "useDefault 表示回到 Flat");
        assert!(sps.scaling4x4[1].is_some(), "显式给出的表要留下来");
        assert_eq!(sps.scaling4x4[1].unwrap()[0], 8);
        assert_eq!(sps.scaling8x8[0], None);
        assert_eq!(sps.max_num_reorder_frames(), 2);
        assert_eq!(sps.max_dec_frame_buffering(), 4);
        assert_eq!(sps.vui.as_ref().unwrap().timing, Some((1001, 60000, true)));
        assert_eq!(sps.width(), 320);
        assert_eq!(sps.height(), 288);
        sps.check_supported().unwrap();
    }

    #[test]
    fn sps_absent_vui_falls_back_to_implied_dpb() {
        let mut w = Bits::default();
        w.f(0x67, 8);
        build_sps(&mut w, 66, 40, 23);
        let sps = Sps::parse(&w.finish()).unwrap();
        // 640×368：MaxDpbMbs(5.1 档) / (40*23) = 16384/920 = 17 → 上限 16
        assert_eq!(sps.max_num_reorder_frames(), 16);
        sps.check_supported().unwrap();
    }

    #[test]
    fn sps_rejects_unsupported_constructs() {
        // 隔行：frame_mbs_only_flag = 0
        let mut w = Bits::default();
        w.f(0x67, 8);
        w.f(77, 8);
        w.f(0, 8);
        w.f(30, 8);
        w.ue(0);
        w.ue(0);
        w.ue(0);
        w.ue(0);
        w.ue(1);
        w.bit(false);
        w.ue(39);
        w.ue(22);
        w.bit(false); // frame_mbs_only_flag = 0
        w.bit(false); // mb_adaptive_frame_field_flag
        w.bit(true);
        w.bit(false);
        w.bit(false);
        w.f(0b1000_0000, 8);
        let sps = Sps::parse(&w.finish()).unwrap();
        let text = format!("{}", sps.check_supported().unwrap_err());
        assert!(text.contains("隔行"), "隔行的报错要说人话：{text}");

        // 10 位：High 10 profile 的 bit_depth_luma_minus8 = 2
        let mut w = Bits::default();
        w.f(0x67, 8);
        w.f(110, 8); // High 10
        w.f(0, 8);
        w.f(31, 8);
        w.ue(0);
        w.ue(1); // chroma 4:2:0
        w.ue(2); // 10 位
        w.ue(2);
        w.bit(false);
        w.bit(false);
        w.ue(0);
        w.ue(0);
        w.ue(0);
        w.ue(1);
        w.bit(false);
        w.ue(19);
        w.ue(17);
        w.bit(true);
        w.bit(true);
        w.bit(false);
        w.bit(false);
        w.f(0b1000_0000, 8);
        let sps = Sps::parse(&w.finish()).unwrap();
        assert!(format!("{}", sps.check_supported().unwrap_err()).contains("10-位"));

        // gaps：只改一个位
        let mut w = Bits::default();
        w.f(0x67, 8);
        w.f(77, 8);
        w.f(0, 8);
        w.f(30, 8);
        w.ue(0);
        w.ue(0);
        w.ue(0);
        w.ue(0);
        w.ue(1);
        w.bit(true); // gaps_in_frame_num_value_allowed_flag = 1
        w.ue(39);
        w.ue(22);
        w.bit(true);
        w.bit(true);
        w.bit(false);
        w.bit(false);
        w.f(0b1000_0000, 8);
        let sps = Sps::parse(&w.finish()).unwrap();
        assert!(format!("{}", sps.check_supported().unwrap_err()).contains("跳号"));
    }

    #[test]
    fn pps_parses_and_merges_scaling_lists() {
        let mut w = Bits::default();
        w.f(0x68, 8);
        w.ue(0); // pps id
        w.ue(0); // sps id
        w.bit(true); // entropy_coding_mode_flag（CABAC）
        w.bit(true); // bottom_field_pic_order_in_frame_present_flag
        w.ue(0); // num_slice_groups_minus1
        w.ue(2); // l0 default active - 1 → 3
        w.ue(0); // l1
        w.bit(true); // weighted_pred_flag
        w.f(1, 2); // weighted_bipred_idc
        w.se(-2); // pic_init_qp_minus26
        w.se(0);
        w.se(1); // chroma_qp_index_offset
        w.bit(true); // deblocking_filter_control_present_flag
        w.bit(true); // constrained_intra_pred_flag
        w.bit(true); // redundant_pic_cnt_present_flag
        w.bit(true); // transform_8x8_mode_flag
        w.bit(true); // pic_scaling_matrix_present_flag
        w.bit(false); // 4x4 #0 不给 → 沿用 SPS / 缺省
        w.bit(true); // 4x4 #1 给
        for _ in 0..16 {
            w.se(0);
        }
        for _ in 2..8 {
            w.bit(false);
        }
        w.se(-1); // second_chroma_qp_index_offset
        w.f(0b1000_0000, 8);

        // ⚠️ 这个尾巴**只在 High 系 profile 里存在**（§7.3.2.2），
        // 拿 `Sps::default()`（Baseline）去解析会（正确地）跳过它
        let sps = Sps {
            profile_idc: 100,
            ..Sps::default()
        };
        let pps = Pps::parse(&w.finish(), &sps).unwrap();
        assert!(pps.entropy_coding_mode_flag);
        assert!(pps.bottom_field_pic_order_in_frame_present_flag);
        assert_eq!(pps.num_ref_idx_l0_default_active_minus1, 2);
        assert!(pps.weighted_pred_flag);
        assert_eq!(pps.weighted_bipred_idc, 1);
        assert_eq!(pps.pic_init_qp_minus26, -2);
        assert_eq!(pps.chroma_qp_index_offset, 1);
        assert!(pps.transform_8x8_mode_flag);
        assert_eq!(pps.second_chroma_qp_index_offset, -1);
        assert!(pps.scaling4x4[0].is_none(), "没给的表要保留 None");
        let (l4, l8) = pps.effective_scaling(&sps);
        assert_eq!(l4[0], [16u8; 16], "两处都没有 → 缺省 Flat");
        assert_eq!(l4[1][0], 8, "PPS 自己带的表生效");
        assert_eq!(l8[0], [16u8; 64]);
        assert!(pps.has_scaling_matrix(&sps));
    }

    /// §7.4.2.2：`second_chroma_qp_index_offset` **不在位流里时推断为等于
    /// `chroma_qp_index_offset`**。
    ///
    /// 这条错了的症状特别隐蔽：**Cb 逐样本全对、Cr 整体偏几阶**，
    /// 而且 R=G=B 的测试内容完全看不出来（Cb ≡ Cr）。
    #[test]
    fn baseline_pps_infers_second_chroma_offset_from_the_first() {
        let mut w = Bits::default();
        w.f(0x68, 8); // NAL 头
        w.ue(0); // pps id
        w.ue(0); // sps id
        w.f(0, 2); // entropy_coding_mode_flag + bottom_field_pic_order_in_frame_present_flag
        w.ue(0); // num_slice_groups_minus1
        w.ue(0); // num_ref_idx_l0_default_active_minus1
        w.ue(0); // num_ref_idx_l1_default_active_minus1
        w.f(0, 3); // weighted_pred_flag + weighted_bipred_idc(2)
        w.se(0); // pic_init_qp_minus26
        w.se(0); // pic_init_qs_minus26
        w.se(-3); // chroma_qp_index_offset
        w.f(0, 3); // deblocking / constrained_intra / redundant_pic_cnt
        w.f(0b1000_0000, 8); // rbsp_trailing_bits

        // Baseline（66）：PPS 里**没有**可选尾巴
        let pps = Pps::parse(&w.finish(), &Sps::default()).unwrap();
        assert_eq!(pps.chroma_qp_index_offset, -3);
        assert_eq!(
            pps.second_chroma_qp_index_offset, -3,
            "不在位流里时必须推断为等于 chroma_qp_index_offset"
        );
        assert!(!pps.transform_8x8_mode_flag);
    }

    #[test]
    fn pps_rejects_fmo() {
        let mut w = Bits::default();
        w.f(0x68, 8);
        w.ue(0);
        w.ue(0);
        w.bit(false);
        w.bit(false);
        w.ue(1); // num_slice_groups_minus1 = 1 → FMO
        let e = Pps::parse(&w.finish(), &Sps::default()).unwrap_err();
        assert!(matches!(e, MediaError::Unsupported(_)), "{e}");
        assert!(format!("{e}").contains("FMO"));
    }

    /// 一片 P 切片：带参考帧重排 + 加权预测 + 去块参数，并核对
    /// `slice_data_bit_offset` 就是"写完切片头那一刻的位偏移"。
    #[test]
    fn slice_header_reports_slice_data_offset() {
        let sps = Sps {
            log2_max_frame_num: 4,
            log2_max_pic_order_cnt_lsb: 4,
            ..Sps::default()
        };
        let pps = Pps {
            weighted_pred_flag: true,
            deblocking_filter_control_present_flag: true,
            ..Pps::default()
        };
        let mut w = Bits::default();
        w.f(0x21, 8); // nal_ref_idc = 1, type = 1（非 IDR 参考切片）
        w.ue(0); // first_mb_in_slice
        w.ue(0); // slice_type = 0（P）
        w.ue(0); // pps id
        w.f(7, 4); // frame_num = 7
        w.f(3, 4); // pic_order_cnt_lsb = 3
        w.bit(true); // num_ref_idx_active_override_flag
        w.ue(1); // num_ref_idx_l0_active = 2
        w.bit(true); // ref_pic_list_modification_flag_l0
        w.ue(0); // idc 0
        w.ue(0); // abs_diff_pic_num_minus1 = 0 → 往前 1 帧
        w.ue(3); // 结束
        w.ue(0); // luma_log2_weight_denom
        w.ue(0); // chroma_log2_weight_denom
        w.bit(true); // luma_weight_l0_flag
        w.se(1); // luma_weight
        w.se(0); // luma_offset
        w.bit(false); // chroma_weight_l0_flag
        w.bit(false); // luma_weight_l0_flag（第 2 个参考帧）
        w.bit(false); // chroma_weight_l0_flag
        w.bit(false); // adaptive_ref_pic_marking_mode_flag = 0（滑动窗口）
        w.se(2); // slice_qp_delta
        w.ue(0); // disable_deblocking_filter_idc
        w.se(-1); // slice_alpha_c0_offset_div2
        w.se(1); // slice_beta_offset_div2
        let header_bits = w.pos;

        // 数据段：随便写几字节，只要能被"切片数据"接上（解析器读到这就停）
        w.f(0xDEAD_BEEF, 32);
        let raw_nal = w.finish();

        let sh = parse_slice_header(&raw_nal, &sps, &pps).unwrap();
        assert_eq!(sh.nal_ref_idc, 1);
        assert!(!sh.idr);
        assert!(sh.is_p());
        assert!(sh.is_reference());
        assert_eq!(sh.frame_num, 7);
        assert_eq!(sh.pic_order_cnt_lsb, 3);
        assert_eq!(sh.num_ref_idx_l0_active, 2);
        assert_eq!(
            sh.l0_mod,
            vec![RefListMod::ShortTerm {
                negative: true,
                abs_diff: 0
            }]
        );
        let pw = sh.pred_weight.clone().unwrap();
        assert_eq!(pw.l0.len(), 2);
        assert_eq!(pw.l0[0].luma, Some((1, 0)));
        assert_eq!(pw.l0[1].luma, None);
        assert!(!sh.dec_ref_pic_marking.adaptive);
        assert_eq!(sh.slice_qp_delta, 2);
        assert_eq!(sh.disable_deblocking_filter_idc, 0);
        assert_eq!(sh.slice_alpha_c0_offset_div2, -1);
        assert_eq!(sh.slice_beta_offset_div2, 1);
        assert_eq!(sh.slice_data_bit_offset, header_bits, "偏移就是切片头写完的位置");
        sh.check_supported().unwrap();
    }

    /// 防竞争字节**不改变** `slice_data_bit_offset`。
    ///
    /// 这是 VA-API 那条路上最容易踩的坑：偏移按"去转义后的位"算，而真正送进驱动
    /// 的切片数据缓冲仍是**原样带 0x03 的字节流**（见 va.h 对
    /// `slice_data_bit_offset` 的注释）。这条测试造一个真的会触发插入的头。
    #[test]
    fn slice_data_offset_ignores_emulation_prevention_bytes() {
        let sps = Sps {
            log2_max_frame_num: 4,
            log2_max_pic_order_cnt_lsb: 4,
            ..Sps::default()
        };
        let mut w = Bits::default();
        w.f(0x21, 8); // 非 IDR 参考切片
        // first_mb_in_slice 取一个大值：它的 ue 编码带 22 个前导零 ⇒ 字节流里出现
        // `00 00 02`，正是"要插防竞争字节"的位置（0x00 0x00 后面跟着 ≤ 0x03）
        w.ue((1 << 22) - 1);
        w.ue(2); // I 切片
        w.ue(0); // pps id
        w.f(0, 4); // frame_num
        w.f(0, 4); // pic_order_cnt_lsb
        w.bit(false); // adaptive_ref_pic_marking_mode_flag
        w.se(0); // slice_qp_delta
        let header_bits = w.pos;
        w.f(0xDEAD_BEEF, 32);
        let raw = w.finish();

        let escaped = me_core::bits::add_emulation_prevention(&raw);
        assert_ne!(escaped, raw, "这个头必须真的触发过一次插入，否则测试是空转");

        // 解码器拿到的永远是**带防竞争字节**的字节流；解析器先去转义再读位，
        // 所以这里解析的是转义后的形态，而断言里的位偏移是"去转义后"的口径。
        let sh = parse_slice_header(&escaped, &sps, &Pps::default()).unwrap();
        assert_eq!(sh.first_mb_in_slice, (1 << 22) - 1);
        assert_eq!(sh.slice_data_bit_offset, header_bits);
        assert_eq!(
            me_core::bits::remove_emulation_prevention(&escaped),
            raw,
            "去转义必须一字不差地还原"
        );
    }

    #[test]
    fn slice_header_rejects_mmco() {
        let sps = Sps::default();
        let pps = Pps::default();
        let mut w = Bits::default();
        w.f(0x21, 8); // nal_ref_idc = 1, type = 1：参考切片
        w.ue(0); // first_mb
        w.ue(2); // I 切片
        w.ue(0); // pps
        w.f(1, 4); // frame_num
        w.f(0, 4); // poc lsb
        w.bit(true); // adaptive_ref_pic_marking_mode_flag = 1 → MMCO
        w.ue(1); // mmco 1
        w.ue(0); // difference_of_pic_nums_minus1
        w.ue(0); // 结束
        w.se(0); // slice_qp_delta
        let nal = w.finish();
        let sh = parse_slice_header(&nal, &sps, &pps).unwrap();
        assert_eq!(sh.dec_ref_pic_marking.ops.len(), 1);
        let e = sh.check_supported().unwrap_err();
        assert!(matches!(e, MediaError::Unsupported(_)));
        assert!(format!("{e}").contains("MMCO"), "{e}");
    }

    /// 拿一个最简 I 切片头当底子（POC 测试只关心少数几个字段，其余随意）。
    fn base_slice() -> SliceHeader {
        minimal_slice()
    }

    #[test]
    fn poc_type0_orders_b_frames() {
        let sps = Sps {
            log2_max_frame_num: 4,
            log2_max_pic_order_cnt_lsb: 4,
            ..Sps::default()
        };
        let mk = |idr: bool, lsb: u32, reference: bool| SliceHeader {
            idr,
            nal_ref_idc: if reference { 1 } else { 0 },
            pic_order_cnt_lsb: lsb,
            ..base_slice()
        };
        let mut st = PocState::new();
        // 解码序 I0 P4 B2 B1 B3（POC 0,4,2,1,3）
        assert_eq!(st.compute(&sps, &mk(true, 0, true)).unwrap(), 0);
        assert_eq!(st.compute(&sps, &mk(false, 4, true)).unwrap(), 4);
        assert_eq!(st.compute(&sps, &mk(false, 2, false)).unwrap(), 2);
        assert_eq!(st.compute(&sps, &mk(false, 1, false)).unwrap(), 1);
        assert_eq!(st.compute(&sps, &mk(false, 3, false)).unwrap(), 3);
        // 非参考帧不动 prev_*（prev 是"上一个参考图"的量），所以这里 POC = 5 而不是负数
        assert_eq!(st.compute(&sps, &mk(false, 5, true)).unwrap(), 5);
    }

    #[test]
    fn poc_type2_counts_non_reference_frames() {
        let sps = Sps {
            log2_max_frame_num: 4,
            pic_order_cnt_type: 2,
            ..Sps::default()
        };
        let mk = |frame_num: u32, reference: bool, idr: bool| SliceHeader {
            idr,
            nal_ref_idc: if reference { 1 } else { 0 },
            frame_num,
            ..base_slice()
        };
        let mut st = PocState::new();
        assert_eq!(st.compute(&sps, &mk(0, true, true)).unwrap(), 0);
        assert_eq!(st.compute(&sps, &mk(1, true, false)).unwrap(), 2);
        assert_eq!(st.compute(&sps, &mk(2, false, false)).unwrap(), 3); // 非参考 −1
        assert_eq!(st.compute(&sps, &mk(3, true, false)).unwrap(), 6);
    }

    #[test]
    fn poc_type1_adds_cycle_offsets() {
        // 每轮两个参考帧的偏移是 +4、+6；非参考帧额外 −2
        let sps = Sps {
            log2_max_frame_num: 4,
            pic_order_cnt_type: 1,
            offset_for_non_ref_pic: -2,
            offset_for_top_to_bottom_field: 0,
            offset_for_ref_frame: vec![4, 6],
            ..Sps::default()
        };
        let mk = |frame_num: u32, reference: bool| SliceHeader {
            frame_num,
            nal_ref_idc: if reference { 1 } else { 0 },
            ..base_slice()
        };
        let mut st = PocState::new();
        assert_eq!(st.compute(&sps, &mk(0, true)).unwrap(), 0); // absFrameNum = 0
        assert_eq!(st.compute(&sps, &mk(1, true)).unwrap(), 4); // 第 0 个偏移
        assert_eq!(st.compute(&sps, &mk(2, true)).unwrap(), 10); // 4 + 6
        assert_eq!(st.compute(&sps, &mk(3, false)).unwrap(), 8); // absFrame 3−1=2 → 10，再 −2
        assert_eq!(st.compute(&sps, &mk(4, true)).unwrap(), 20); // 第二轮：10 + 4 + 6
    }

    #[test]
    fn dpb_sliding_window_and_list_order() {
        let mut dpb = Dpb::default();
        // 参考帧 1,2,3（frame_num/POC 都对上：I0 P1 P2 P3）
        dpb.push(RefPic { slot: 10, frame_num: 1, poc: 1 });
        dpb.push(RefPic { slot: 11, frame_num: 2, poc: 2 });
        dpb.push(RefPic { slot: 12, frame_num: 3, poc: 3 });
        // 当前图 frame_num = 4，最多留 3 个 → 淘汰 frame_num=1
        let removed = dpb.mark_sliding_window(3, 4, 16);
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].slot, 10);

        let l0 = dpb.default_list(false, 4, 16);
        assert_eq!(l0.iter().map(|r| r.frame_num).collect::<Vec<_>>(), vec![3, 2]);
        let l1 = dpb.default_list(true, 4, 16);
        assert_eq!(l1.iter().map(|r| r.frame_num).collect::<Vec<_>>(), vec![2, 3]);

        // 重排：list0 的第 0 位换成 frame_num=2（往前 2 帧）→ 2,3
        let mods = vec![RefListMod::ShortTerm {
            negative: true,
            abs_diff: 1,
        }];
        let reordered = dpb.ref_list(false, 2, &mods, 4, 16).unwrap();
        assert_eq!(
            reordered.iter().map(|r| r.frame_num).collect::<Vec<_>>(),
            vec![2, 3]
        );

        // 要 4 个参考帧但只有 2 个 → 明确报错
        let e = dpb.ref_list(false, 4, &[], 4, 16).unwrap_err();
        assert!(format!("{e}").contains("参考帧不够"), "{e}");

        // 重排指向不存在的帧 → 明确报错
        let bad = vec![RefListMod::ShortTerm {
            negative: true,
            abs_diff: 5,
        }];
        assert!(dpb.ref_list(false, 2, &bad, 4, 16).is_err());
    }

    #[test]
    fn avcc_samples_split_and_convert() {
        let sample = vec![
            0, 0, 0, 3, 0x65, 0xAA, 0xBB, //
            0, 0, 0, 2, 0x41, 0xCC,
        ];
        let nals = split_avcc_sample(&sample, 4).unwrap();
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0], &[0x65, 0xAA, 0xBB]);
        assert_eq!(nals[1], &[0x41, 0xCC]);
        let annexb = avcc_to_annexb(&sample, 4).unwrap();
        assert_eq!(
            annexb,
            vec![0, 0, 0, 1, 0x65, 0xAA, 0xBB, 0, 0, 0, 1, 0x41, 0xCC]
        );
        // 长度越界要拒绝，不能截断着继续
        assert!(split_avcc_sample(&[0, 0, 0, 9, 0x41], 4).is_err());
        assert!(split_avcc_sample(&[0, 0], 4).is_err());
    }

    #[test]
    fn config_parses_a_full_avcc() {
        // 自己造一组**几何可信**的参数集：640×368 编码、裁到 640×360。
        //
        // 讲一下为什么不用仓库里那串现成的 SPS/PPS 字节（`macos.rs` 与 CLI 契约
        // 测试都在用）：那是**合成夹具**，只为"让 probe 认出这是 H.264 轨"，
        // 按规范解出来其实是隔行、320 宽 —— 那里不需要几何正确，这里需要。
        let mut sw = Bits::default();
        sw.f(0x67, 8);
        sw.f(100, 8); // High
        sw.f(0, 8);
        sw.f(30, 8);
        sw.ue(0);
        sw.ue(1); // chroma_format_idc = 4:2:0
        sw.ue(0);
        sw.ue(0);
        sw.bit(false); // qpprime
        sw.bit(false); // 无量化矩阵
        sw.ue(0); // log2_max_frame_num_minus4
        sw.ue(0); // pic_order_cnt_type = 0
        sw.ue(0); // log2_max_pic_order_cnt_lsb_minus4
        sw.ue(2); // max_num_ref_frames
        sw.bit(false); // gaps
        sw.ue(39); // 640
        sw.ue(22); // 368
        sw.bit(true); // frame_mbs_only_flag
        sw.bit(true); // direct_8x8_inference
        sw.bit(true); // frame_cropping_flag
        sw.ue(0);
        sw.ue(0);
        sw.ue(0);
        sw.ue(4); // bottom 4 → 368 − 8 = 360
        sw.bit(true); // vui_parameters_present_flag
        sw.bit(false); // aspect
        sw.bit(false); // overscan
        sw.bit(false); // video_signal_type
        sw.bit(false); // chroma_loc
        sw.bit(false); // timing
        sw.bit(false); // nal_hrd
        sw.bit(false); // vcl_hrd
        sw.bit(false); // pic_struct
        sw.bit(true); // bitstream_restriction
        sw.bit(true); // motion_vectors_over_pic_boundaries
        sw.ue(0);
        sw.ue(0);
        sw.ue(8);
        sw.ue(8);
        sw.ue(2); // max_num_reorder_frames
        sw.ue(4); // max_dec_frame_buffering
        sw.f(0b1000_0000, 8);
        let sps = sw.finish();

        let mut pw = Bits::default();
        pw.f(0x68, 8);
        build_pps(&mut pw);
        let pps = pw.finish();

        let mut avcc = vec![1u8, 0x64, 0x00, 0x1E, 0xFF, 0xE1];
        avcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(&sps);
        avcc.push(0x01);
        avcc.extend_from_slice(&(pps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(&pps);

        let cfg = Config::parse(&avcc).expect("这组参数集应当能解析");
        assert_eq!(cfg.sps.profile_idc, 100);
        assert_eq!((cfg.sps.width(), cfg.sps.height()), (640, 360));
        assert_eq!(cfg.sps.coded_height(), 368, "360 要凑到 16 的整数倍再裁");
        assert_eq!(cfg.sps.max_num_reorder_frames(), 2);
        assert_eq!(cfg.pps.id, 0);
        assert_eq!(cfg.nal_length_size, 4);
    }
}
