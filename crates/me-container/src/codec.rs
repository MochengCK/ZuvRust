//! 码流辅助解析：Annex-B 拆包、NAL 类型、SPS 里的图像尺寸。
//!
//! 这一层**只读语法、不解码**。它存在的唯一目的是让 `probe` 能给出
//! "1920x1080 / 25fps" 这种用户真正想看的信息 —— 没有它，探测结果里
//! 分辨率只能写 0，宿主界面就只能显示"未知分辨率"。
//!
//! ## extradata 的两种形态（**全引擎最重要的约定之一**）
//!
//! 同一个 H.264 轨道，从不同容器里解析出来，`Track::extradata` 的**字节形态是不同的**：
//!
//! - **从 MPEG-TS 解析出来**：Annex-B 形态 —— `SPS` 与 `PPS` 各自带起始码拼接起来。
//!   TS 里根本没有 `avcC` 这个东西，参数集就是当作普通 NAL 混在码流里传的。
//! - **从 ISO BMFF 解析出来**：`avcC` box 的**载荷**（不含 box 头与长度），
//!   里面是"version + profile + lengthSizeMinusOne + SPS 数组 + PPS 数组"。
//!
//! 这不是不统一，而是**如实反映来源**：硬把它们在解析时就统一，
//! 会让"原样透传"这条路径（TS→TS 重封装）被迫做一次无谓的形态转换，
//! 而转换过程中一旦出错就是黑屏。所以约定是：
//! **解析层如实报告，转换由 muxer 负责**（见 `me-mux` 的 `annexb_to_avcc` / `avcc_to_annexb`）。

use me_core::bits::{remove_emulation_prevention, BitReader};

/// SPS 里能拿到的关键信息。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpsInfo {
    pub width: u32,
    pub height: u32,
    /// 采样宽高比（VUI 里给的，未给则 None）。
    pub sar: Option<(u32, u32)>,
    pub profile: u8,
    pub level: u8,
}

/// 拆出 Annex-B 里的每个 NAL（**返回的切片含起始码**）。
///
/// 长度前缀布局（`avcC` / `hvcC` 那种「[4 字节长度][NAL]…」）→ Annex-B。
///
/// **唯一实现**：fMP4 解封装（`mp4.rs`）与 Matroska 解封装（`mkv.rs`）共用。
/// Matroska 的 H.264/H.265 采样就是长度前缀布局，与 fMP4 一模一样 —— 两处各写
/// 一份的话，将来修"长度越界"这类边界只会修到一边。
///
/// `nal_len == 0` 表示这条轨道**不是**长度前缀布局（音频、VP8/VP9、AV1 的 OBU），
/// 原样返回 —— **不做"顺手也加个起始码"这种事**：那会把裸 AAC 帧变成噪声。
pub fn length_prefixed_to_annexb(data: &[u8], nal_len: u8) -> Vec<u8> {
    if nal_len == 0 {
        return data.to_vec();
    }
    let n = nal_len as usize;
    let mut out = Vec::with_capacity(data.len() + 32);
    let mut pos = 0usize;
    while pos + n <= data.len() {
        let mut len = 0usize;
        for i in 0..n {
            len = (len << 8) | data[pos + i] as usize;
        }
        pos += n;
        if len == 0 {
            continue;
        }
        if pos + len > data.len() {
            // 长度越界：这个采样坏了。把剩下的字节当最后一个 NAL 带上，
            // 让解码器自己丢 —— 因为一个坏采样就让整条流报错，代价太大。
            // 注意 `pos` 本身也可能已经越界（长度前缀都不完整），
            // 那时切片会 panic，所以必须先判一下
            if pos < data.len() {
                out.extend_from_slice(&[0, 0, 0, 1]);
                out.extend_from_slice(&data[pos..]);
            }
            break;
        }
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&data[pos..pos + len]);
        pos += len;
    }
    if out.is_empty() && !data.is_empty() {
        // 全是零长度（畸形）：宁可原样交出去，也不要把数据变成空包
        return data.to_vec();
    }
    out
}

/// 为什么连起始码一起返回：重封装时要原样搬运，把起始码剥掉再拼回去
/// 只是多两次分配；而"要不要剥"是个策略问题，交给用的人决定。
///
/// 起始码有 3 字节（`00 00 01`）与 4 字节（`00 00 00 01`）两种形态，
/// **必须都认**：H.264 里 4 字节形态常用于 SPS/PPS 与 IDR 之前，
/// 只认 3 字节会把 NAL 头切错一个字节，解析出来的 profile/level 全乱。
pub fn split_annexb(data: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut starts: Vec<(usize, usize)> = Vec::new(); // (起始码长度, NAL 起点)
    let mut i = 0usize;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            let sc = if i > 0 && data[i - 1] == 0 { 4 } else { 3 };
            let nal_start = i + 3;
            starts.push((sc, nal_start));
            i = nal_start;
        } else {
            i += 1;
        }
    }
    for (idx, &(_sc, nal_start)) in starts.iter().enumerate() {
        let end = starts.get(idx + 1).map(|&(sc, s)| s - sc).unwrap_or(data.len());
        if end > nal_start {
            out.push(&data[nal_start..end]);
        }
    }
    out
}

/// 带起始码的 NAL 切片（给"原样搬运"用）。
pub fn split_annexb_with_start_codes(data: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut starts: Vec<(usize, usize)> = Vec::new();
    let mut i = 0usize;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            let sc = if i > 0 && data[i - 1] == 0 { 4 } else { 3 };
            starts.push((i + 3 - sc, i + 3));
            i += 3;
        } else {
            i += 1;
        }
    }
    for (idx, &(sc_start, _nal_start)) in starts.iter().enumerate() {
        let end = starts.get(idx + 1).map(|&(s, _)| s).unwrap_or(data.len());
        if end > sc_start {
            out.push(&data[sc_start..end]);
        }
    }
    out
}

/// H.264 的 NAL 类型（取头字节低 5 位）。
pub fn h264_nal_type(nal: &[u8]) -> Option<u8> {
    // NAL 头是 1 字节；只给出空切片时无从判断
    nal.first().map(|b| b & 0x1F)
}

/// H.265 的 NAL 类型（两字节头里的 6 位）。
pub fn h265_nal_type(nal: &[u8]) -> Option<u8> {
    nal.first().map(|b| (b >> 1) & 0x3F)
}

/// 这个 NAL 是不是"能从这个点开始解"的随机访问点。
///
/// - H.264：`IDR(5)` 必然是；`SPS(7)` 也当随机访问点（序列参数变了，必须从这里重来）。
/// - H.265：`BLA_W_LP..CRA_NUT(16..=21)` 是；`VPS(32)/SPS(33)/PPS(34)` 同理。
///
/// 不把"任何 I 帧"当随机访问点是刻意的：非 IDR 的 I 帧后面可能跟着
/// 引用更早帧的 B/P 帧，从那里切入会花屏。
pub fn is_random_access_nal(codec_hevc: bool, nal: &[u8]) -> bool {
    if codec_hevc {
        matches!(h265_nal_type(nal), Some(16..=21) | Some(32) | Some(33) | Some(34))
    } else {
        matches!(h264_nal_type(nal), Some(5) | Some(7))
    }
}

/// 解析 H.264 的 SPS（`nal` 含 1 字节 NAL 头）。
///
/// 只在需要"宽高 / profile / SAR"时用；解析失败返回 `None` ——
/// **探测阶段不该因为一个畸形 SPS 就整条流失败**，那只会让本来能播的片子探不出来。
pub fn parse_h264_sps(nal: &[u8]) -> Option<SpsInfo> {
    if h264_nal_type(nal)? != 7 {
        return None;
    }
    let body = remove_emulation_prevention(nal.get(1..)?);
    let mut r = BitReader::new(&body);

    let profile = r.read_u32(8).ok()? as u8;
    r.read_u32(8).ok()?; // constraint flags + reserved
    let level = r.read_u32(8).ok()? as u8;
    r.read_ue().ok()?; // seq_parameter_set_id

    let mut chroma_format_idc = 1u32;
    let mut separate_colour_plane = false;
    if matches!(
        profile,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    ) {
        chroma_format_idc = r.read_ue().ok()?;
        if chroma_format_idc == 3 {
            separate_colour_plane = r.read_bits(1).ok()? == 1;
        }
        r.read_ue().ok()?; // bit_depth_luma_minus8
        r.read_ue().ok()?; // bit_depth_chroma_minus8
        r.read_bits(1).ok()?; // qpprime_y_zero_transform_bypass_flag
        let scaling_present = r.read_bits(1).ok()? == 1;
        if scaling_present {
            let count = if chroma_format_idc == 3 { 12 } else { 8 };
            for i in 0..count {
                let present = r.read_bits(1).ok()? == 1;
                if present {
                    let size = if i < 6 { 16 } else { 64 };
                    skip_scaling_list(&mut r, size)?;
                }
            }
        }
    }

    r.read_ue().ok()?; // log2_max_frame_num_minus4
    let poc_type = r.read_ue().ok()?;
    if poc_type == 0 {
        r.read_ue().ok()?; // log2_max_pic_order_cnt_lsb_minus4
    } else if poc_type == 1 {
        r.read_bits(1).ok()?; // delta_pic_order_always_zero_flag
        r.read_se().ok()?; // offset_for_non_ref_pic
        r.read_se().ok()?; // offset_for_top_to_bottom_field
        let n = r.read_ue().ok()?;
        // 一个畸形流可能给出天文数字的循环次数；设个上限防"解析一条坏 SPS 卡死"
        if n > 256 {
            return None;
        }
        for _ in 0..n {
            r.read_se().ok()?;
        }
    }

    r.read_ue().ok()?; // max_num_ref_frames
    r.read_bits(1).ok()?; // gaps_in_frame_num_value_allowed_flag
    let width_mbs = r.read_ue().ok()? + 1;
    let height_map_units = r.read_ue().ok()? + 1;
    let frame_mbs_only = r.read_bits(1).ok()? == 1;
    if !frame_mbs_only {
        r.read_bits(1).ok()?; // mb_adaptive_frame_field_flag
    }
    r.read_bits(1).ok()?; // direct_8x8_inference_flag

    let mut crop = (0u32, 0u32, 0u32, 0u32);
    if r.read_bits(1).ok()? == 1 {
        crop.0 = r.read_ue().ok()?;
        crop.1 = r.read_ue().ok()?;
        crop.2 = r.read_ue().ok()?;
        crop.3 = r.read_ue().ok()?;
    }

    let sar = if r.read_bits(1).ok()? == 1 {
        parse_vui_aspect_ratio(&mut r)
    } else {
        None
    };

    // 4:2:0 的裁剪单位是 (2,2)，4:2:2 是 (2,1)，4:4:4 与单色是 (1,1)
    let crop_unit_x = if chroma_format_idc == 0 || separate_colour_plane || chroma_format_idc == 3 {
        1
    } else {
        2
    };
    let crop_unit_y = if chroma_format_idc == 0 || separate_colour_plane {
        1
    } else if chroma_format_idc == 1 {
        2
    } else {
        1
    };

    let width = (width_mbs * 16).saturating_sub((crop.0 + crop.1) * crop_unit_x);
    let height = ((2 - frame_mbs_only as u32) * height_map_units * 16)
        .saturating_sub((crop.2 + crop.3) * crop_unit_y);

    if width == 0 || height == 0 {
        return None;
    }
    Some(SpsInfo { width, height, sar, profile, level })
}

/// 解析 H.265 的 SPS（`nal` 含 2 字节 NAL 头）。
pub fn parse_h265_sps(nal: &[u8]) -> Option<SpsInfo> {
    if h265_nal_type(nal)? != 33 {
        return None;
    }
    let body = remove_emulation_prevention(nal.get(2..)?);
    let mut r = BitReader::new(&body);

    let profile = {
        // 先把前 8 位读了才能知道 general_profile_idc —— 但它是"先读 4 位 + 3 位 + 1 位"的
        // 结构，所以按语法顺序读，顺手把 profile_idc 记下来
        let _vps_id = r.read_u32(4).ok()?;
        let max_sub_layers_minus1 = r.read_u32(3).ok()?;
        r.read_bits(1).ok()?; // temporal_id_nesting_flag
        parse_h265_profile_tier_level(&mut r, max_sub_layers_minus1)?
    };

    r.read_ue().ok()?; // sps_seq_parameter_set_id
    let chroma_format_idc = r.read_ue().ok()?;
    let separate_colour_plane = if chroma_format_idc == 3 {
        r.read_bits(1).ok()? == 1
    } else {
        false
    };
    let width = r.read_ue().ok()?;
    let height = r.read_ue().ok()?;
    let mut crop = (0u32, 0u32, 0u32, 0u32);
    if r.read_bits(1).ok()? == 1 {
        crop.0 = r.read_ue().ok()?;
        crop.1 = r.read_ue().ok()?;
        crop.2 = r.read_ue().ok()?;
        crop.3 = r.read_ue().ok()?;
    }

    let (sub_w, sub_h) = if separate_colour_plane {
        (1, 1)
    } else {
        match chroma_format_idc {
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        }
    };
    let width = width.saturating_sub((crop.0 + crop.1) * sub_w);
    let height = height.saturating_sub((crop.2 + crop.3) * sub_h);
    if width == 0 || height == 0 {
        return None;
    }
    Some(SpsInfo { width, height, sar: None, profile, level: 0 })
}

/// H.265 的 `profile_tier_level()`；返回 `general_profile_idc`。
fn parse_h265_profile_tier_level(r: &mut BitReader, max_sub_layers_minus1: u32) -> Option<u8> {
    if max_sub_layers_minus1 > 6 {
        return None; // 规范上最多 7 层
    }
    r.read_bits(2).ok()?; // general_profile_space
    r.read_bits(1).ok()?; // general_tier_flag
    let profile_idc = r.read_u32(5).ok()? as u8;
    r.read_bits(32).ok()?; // general_profile_compatibility_flag[32]
    // 约束标志：4 个 source/interlaced/non_packed/frame_only 标志 + 44 位保留 = 48 位
    r.read_bits(32).ok()?;
    r.read_bits(16).ok()?;

    let mut profile_present = [false; 7];
    let mut level_present = [false; 7];
    for i in 0..max_sub_layers_minus1 as usize {
        profile_present[i] = r.read_bits(1).ok()? == 1;
        level_present[i] = r.read_bits(1).ok()? == 1;
    }
    if max_sub_layers_minus1 > 0 {
        for _ in max_sub_layers_minus1..8 {
            r.read_bits(2).ok()?; // reserved_zero_2bits
        }
    }
    for i in 0..max_sub_layers_minus1 as usize {
        if profile_present[i] {
            // sub_layer_profile_space(2)+tier(1)+profile_idc(5)+compat(32)+constraints(48) = 88
            r.read_bits(32).ok()?;
            r.read_bits(32).ok()?;
            r.read_bits(24).ok()?;
        }
        if level_present[i] {
            r.read_bits(8).ok()?;
        }
    }
    Some(profile_idc)
}

/// 只读到 `aspect_ratio_info` 为止（后面的 overscan / timing / HRD 与我们无关）。
fn parse_vui_aspect_ratio(r: &mut BitReader) -> Option<(u32, u32)> {
    if r.read_bits(1).ok()? == 0 {
        return None;
    }
    let idc = r.read_u32(8).ok()?;
    // 规范表 E-1：1..16 是预定义比例，255 表示后面显式给宽高
    let table = [
        (1u32, 1u32),
        (1, 1),
        (12, 11),
        (10, 11),
        (16, 11),
        (40, 33),
        (24, 11),
        (20, 11),
        (32, 11),
        (80, 33),
        (18, 11),
        (15, 11),
        (64, 33),
        (160, 99),
        (4, 3),
        (3, 2),
        (2, 1),
    ];
    if idc == 255 {
        let w = r.read_bits(16).ok()? as u32;
        let h = r.read_bits(16).ok()? as u32;
        if w > 0 && h > 0 {
            return Some((w, h));
        }
        return None;
    }
    table.get(idc as usize).copied()
}

/// 跳过一段 scaling list（H.264 `scaling_list()`）。
fn skip_scaling_list(r: &mut BitReader, size: usize) -> Option<()> {
    let mut last_scale = 8i32;
    let mut next_scale = 8i32;
    for _ in 0..size {
        if next_scale != 0 {
            let delta = r.read_se().ok()?;
            next_scale = (last_scale + delta + 256) % 256;
        }
        last_scale = if next_scale == 0 { last_scale } else { next_scale };
    }
    Some(())
}

/// 求最大公约数（把帧率化简成最简分数）。
pub fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        return a;
    }
    gcd(b, a % b)
}

/// 化简分数；分母为 0 时返回 `None`。
pub fn reduce(mut a: u64, mut b: u64) -> Option<(u32, u32)> {
    if b == 0 || a == 0 {
        return None;
    }
    let g = gcd(a, b);
    a /= g;
    b /= g;
    if a > u32::MAX as u64 || b > u32::MAX as u64 {
        return None;
    }
    Some((a as u32, b as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 用最小的位写入器手搓 SPS —— 期望值全部按 H.264 规范逐字段算出来，
    /// 而不是"跑出来是什么就写什么"。
    struct BitWriter {
        bits: Vec<u8>,
    }
    impl BitWriter {
        fn new() -> Self {
            Self { bits: Vec::new() }
        }
        fn u(&mut self, v: u64, n: u32) {
            for i in (0..n).rev() {
                let b = ((v >> i) & 1) as u8;
                self.bits.push(b);
            }
        }
        fn ue(&mut self, v: u32) {
            // Exp-Golomb：值 v → 先写 (n-1) 个 0，再把 (v+1) 写成 n 位，
            // 其中 n = bit_length(v+1)。v=0 时 v1=1、n=1，只写一个 1。
            let v1 = v as u64 + 1;
            let n = 64 - v1.leading_zeros();
            self.u(0, n - 1);
            self.u(v1, n);
        }
        fn bytes(&self) -> Vec<u8> {
            let mut out = Vec::new();
            let mut cur = 0u8;
            let mut n = 0;
            for b in &self.bits {
                cur = (cur << 1) | b;
                n += 1;
                if n == 8 {
                    out.push(cur);
                    cur = 0;
                    n = 0;
                }
            }
            if n > 0 {
                out.push(cur << (8 - n));
            }
            out
        }
    }

    fn build_sps_baseline(width_mbs: u32, height_map_units: u32, crop: (u32, u32, u32, u32)) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.u(66, 8); // profile_idc = 66 (Baseline，不带 chroma 扩展字段)
        w.u(0, 8); // constraint flags
        w.u(30, 8); // level_idc = 3.0
        w.ue(0); // sps_id
        w.ue(0); // log2_max_frame_num_minus4
        w.ue(0); // pic_order_cnt_type = 0
        w.ue(0); // log2_max_pic_order_cnt_lsb_minus4
        w.ue(1); // max_num_ref_frames
        w.u(0, 1); // gaps_in_frame_num_value_allowed_flag
        w.ue(width_mbs - 1);
        w.ue(height_map_units - 1);
        w.u(1, 1); // frame_mbs_only_flag = 1
        w.u(1, 1); // direct_8x8_inference_flag
        if crop == (0, 0, 0, 0) {
            w.u(0, 1); // frame_cropping_flag
        } else {
            w.u(1, 1);
            w.ue(crop.0);
            w.ue(crop.1);
            w.ue(crop.2);
            w.ue(crop.3);
        }
        w.u(0, 1); // vui_parameters_present_flag
        let mut out = vec![0x67u8]; // NAL 头：forbidden=0, ref_idc=3, type=7
        out.extend_from_slice(&w.bytes());
        out
    }

    #[test]
    fn splits_three_and_four_byte_start_codes() {
        let data = [
            0, 0, 0, 1, 0x67, 0xAA, // 4 字节起始码
            0, 0, 1, 0x68, 0xBB, // 3 字节起始码
            0, 0, 1, 0x65, 0xCC, // IDR
        ];
        let nals = split_annexb(&data);
        assert_eq!(nals.len(), 3);
        assert_eq!(nals[0], &[0x67, 0xAA]);
        assert_eq!(nals[1], &[0x68, 0xBB]);
        assert_eq!(nals[2], &[0x65, 0xCC]);
    }

    #[test]
    fn split_keeps_nothing_for_trailing_zeros() {
        // 末尾连续的零之后没有 NAL 头（填充），不该产出一个空 NAL
        let data = [0, 0, 1, 0x67, 0xAA, 0, 0, 0];
        assert_eq!(split_annexb(&data).len(), 1);
    }

    #[test]
    fn with_start_codes_returns_original_spans() {
        let data = [0, 0, 0, 1, 0x67, 0xAA, 0, 0, 1, 0x68, 0xBB];
        let nals = split_annexb_with_start_codes(&data);
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0], &[0, 0, 0, 1, 0x67, 0xAA]);
        assert_eq!(nals[1], &[0, 0, 1, 0x68, 0xBB]);
    }

    #[test]
    fn nal_types_and_random_access() {
        assert_eq!(h264_nal_type(&[0x65]), Some(5));
        assert_eq!(h264_nal_type(&[0x41]), Some(1));
        assert!(is_random_access_nal(false, &[0x65]));
        assert!(is_random_access_nal(false, &[0x67]));
        assert!(!is_random_access_nal(false, &[0x41]), "非 IDR 的 I 帧不算随机访问点");
        assert!(is_random_access_nal(true, &[0x26, 0x01]), "H.265 IDR_W_RADL");
        assert!(!is_random_access_nal(true, &[0x02, 0x01]), "H.265 TRAIL_R");
    }

    #[test]
    fn parses_1080p_sps() {
        // 1920x1080：宽 120 个宏块，高 68 个宏块（1088），需要裁掉底部 4 行（8 像素）
        let nal = build_sps_baseline(120, 68, (0, 0, 0, 4));
        let sps = parse_h264_sps(&nal).unwrap();
        assert_eq!(sps.width, 1920);
        assert_eq!(sps.height, 1080);
        assert_eq!(sps.profile, 66);
        assert_eq!(sps.level, 30);
    }

    #[test]
    fn parses_720p_without_crop() {
        let nal = build_sps_baseline(80, 45, (0, 0, 0, 0));
        let sps = parse_h264_sps(&nal).unwrap();
        assert_eq!((sps.width, sps.height), (1280, 720));
    }

    #[test]
    fn rejects_wrong_nal_type_and_truncated() {
        assert!(parse_h264_sps(&[0x41, 0x00]).is_none(), "P 帧不该被当 SPS");
        assert!(parse_h264_sps(&[0x67]).is_none(), "截断的 SPS 返回 None 而不是 panic");
        assert!(parse_h264_sps(&[]).is_none());
        assert!(parse_h265_sps(&[0x40, 0x01]).is_none());
    }

    #[test]
    fn h265_sps_gives_dimensions() {
        // 手搓 H.265 SPS：vps_id=0, max_sub_layers_minus1=0,
        // PTL(88 位)、sps_id=0、chroma=1、宽 1920、高 1080、无裁剪窗
        let mut w = BitWriter::new();
        w.u(0, 4); // sps_video_parameter_set_id
        w.u(0, 3); // sps_max_sub_layers_minus1
        w.u(1, 1); // sps_temporal_id_nesting_flag
        w.u(0, 2); // general_profile_space
        w.u(0, 1); // general_tier_flag
        w.u(1, 5); // general_profile_idc = Main
        w.u(0, 32); // compatibility flags
        w.u(0, 32); // constraint flags 前半
        w.u(0, 16); // constraint flags 后半（合计 48 位）
        w.ue(0); // sps_seq_parameter_set_id
        w.ue(1); // chroma_format_idc = 4:2:0
        w.ue(1920); // pic_width_in_luma_samples
        w.ue(1080); // pic_height_in_luma_samples
        w.u(0, 1); // conformance_window_flag
        let mut nal = vec![0x42u8, 0x01]; // nal_type = 33 (SPS)
        nal.extend_from_slice(&w.bytes());
        let sps = parse_h265_sps(&nal).unwrap();
        assert_eq!((sps.width, sps.height), (1920, 1080));
        assert_eq!(sps.profile, 1);
    }

    #[test]
    fn h265_sps_applies_conformance_window() {
        let mut w = BitWriter::new();
        w.u(0, 4);
        w.u(0, 3);
        w.u(1, 1);
        w.u(0, 2);
        w.u(0, 1);
        w.u(1, 5);
        w.u(0, 32);
        w.u(0, 32);
        w.u(0, 16);
        w.ue(0);
        w.ue(1);
        w.ue(1920);
        w.ue(1088);
        w.u(1, 1); // conformance_window_flag
        w.ue(0); // left
        w.ue(0); // right
        w.ue(0); // top
        w.ue(4); // bottom → 4 * 2 = 8 行
        let mut nal = vec![0x42u8, 0x01];
        nal.extend_from_slice(&w.bytes());
        let sps = parse_h265_sps(&nal).unwrap();
        assert_eq!((sps.width, sps.height), (1920, 1080));
    }

    #[test]
    fn reduces_frame_rate_fractions() {
        assert_eq!(reduce(50_000, 1001), Some((50_000, 1001)));
        assert_eq!(reduce(60_000, 2_000), Some((30, 1)));
        assert_eq!(reduce(0, 100), None);
        assert_eq!(reduce(100, 0), None);
        assert_eq!(gcd(12, 18), 6);
    }

    #[test]
    fn aspect_ratio_table_is_reachable() {
        // SAR 走的是 VUI 分支，这里验一下表本身没写歪（16:9 用的是 idc=16 → 2:1? 不，
        // H.264 表 E-1 里 idc=1 才是 1:1，idc=16 是 2:1；这里断言几个确定的）
        let mut w = BitWriter::new();
        w.u(66, 8);
        w.u(0, 8);
        w.u(30, 8);
        w.ue(0);
        w.ue(0);
        w.ue(0);
        w.ue(0);
        w.ue(1);
        w.u(0, 1);
        w.ue(119);
        w.ue(44);
        w.u(1, 1);
        w.u(1, 1);
        w.u(0, 1); // frame_cropping_flag = 0
        w.u(1, 1); // vui_parameters_present_flag
        w.u(1, 1); // aspect_ratio_info_present_flag
        w.u(1, 8); // aspect_ratio_idc = 1 → 1:1
        let mut nal = vec![0x67u8];
        nal.extend_from_slice(&w.bytes());
        let sps = parse_h264_sps(&nal).unwrap();
        assert_eq!(sps.sar, Some((1, 1)));
    }
}