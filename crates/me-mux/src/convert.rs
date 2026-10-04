//! 码流形态转换：Annex-B ↔ 长度前缀（AVCC），以及 `extradata` 的两种形态互转。
//!
//! 这一层存在的唯一理由是 **`extradata` 与采样数据的形态随来源不同**：
//!
//! | 来源 | 采样数据 | `extradata` |
//! |---|---|---|
//! | MPEG-TS | Annex-B（含起始码） | Annex-B 参数集（SPS/PPS 带起始码） |
//! | ISO BMFF | 长度前缀（每 NAL 前 4 字节长度） | `avcC` / `hvcC` 的载荷 |
//!
//! 全引擎的约定是：**解析层如实报告来源形态，转换由 muxer 负责**。
//! 把"形态"当隐式约定往上传，任何一层都可能猜错，而猜错的后果是黑屏 ——
//! 所以转换集中在这里，并且**判断来源形态用"能不能按该形态解析"而不是猜**。

use me_core::{CodecId, MediaError, Result};

use me_container::codec;

/// 判断"这段 extradata 是不是 `avcC` / `hvcC` / `av1C` 载荷"。
///
/// 判据是**能不能解析出配置**，不是看头几个字节：
/// - `avcC`：第 0 字节必须是 1（configurationVersion），第 5 字节是 SPS 个数，
///   并且第一段 SPS 的长度字段要落在缓冲里；
/// - `hvcC`：第 0 字节必须是 1，第 22 字节是数组个数，第一个数组的类型要合法；
/// - `av1C`：第 0 字节必须是 `0x81`（marker=1 + version=1）—— AV1 没有 Annex-B
///   形态，这个 marker 是唯一能把它和"任意一段字节"分开的东西。
///
/// 这样判断比"看有没有起始码"更稳：Annex-B 参数集里出现 `01` 开头是完全可能的。
pub fn looks_like_avcc(codec: &CodecId, extradata: &[u8]) -> bool {
    match codec {
        CodecId::H264 => {
            if extradata.len() < 7 || extradata[0] != 1 {
                return false;
            }
            let num_sps = extradata[5] as usize;
            if num_sps == 0 {
                return false;
            }
            let len = u16::from_be_bytes([extradata[6], extradata[7]]) as usize;
            8 + len <= extradata.len()
        }
        CodecId::H265 => {
            if extradata.len() < 24 || extradata[0] != 1 {
                return false;
            }
            let arrays = extradata[22] as usize;
            if arrays == 0 {
                return false;
            }
            // 第一个数组：类型(1) + numNalus(2)，类型必须在 32..=39 之间
            let ty = extradata[23] & 0x3F;
            (32..=39).contains(&ty)
        }
        CodecId::Av1 => is_av1c(extradata),
        _ => false,
    }
}

/// 这段字节是不是 `av1C`（AV1CodecConfigurationRecord，ISO/IEC 14496-15）。
///
/// `0x81` = marker(1) + version(7)。长度至少 4 字节（marker/profile/位深/保留）。
pub fn is_av1c(extradata: &[u8]) -> bool {
    extradata.len() >= 4 && extradata[0] == 0x81
}

/// 参数集（Annex-B 形态，含起始码）→ `avcC` / `hvcC` 载荷。
///
/// `annexb` 里可以是 SPS+PPS（H.264）或 VPS+SPS+PPS（H.265），顺序不限 ——
/// 按 NAL 类型分拣，而不是假设"第一个是 SPS"。
pub fn annexb_to_avcc_extradata(codec: &CodecId, annexb: &[u8], nal_len: u8) -> Result<Vec<u8>> {
    let nals = codec::split_annexb(annexb);
    if nals.is_empty() {
        return Err(MediaError::Bitstream("extradata 里没有任何 NAL".into()));
    }
    let nal_len = nal_len.clamp(1, 4);
    match codec {
        CodecId::H264 => {
            let mut sps: Vec<&[u8]> = Vec::new();
            let mut pps: Vec<&[u8]> = Vec::new();
            for n in &nals {
                match codec::h264_nal_type(n) {
                    Some(7) => sps.push(n),
                    Some(8) => pps.push(n),
                    _ => {}
                }
            }
            if sps.is_empty() || pps.is_empty() {
                return Err(MediaError::Bitstream(
                    "H.264 的 extradata 必须同时含 SPS(7) 与 PPS(8)".into(),
                ));
            }
            let head = &sps[0];
            let mut out = Vec::with_capacity(annexb.len() + 16);
            out.push(1); // configurationVersion
            out.push(*head.get(1).unwrap_or(&0x42)); // AVCProfileIndication
            out.push(*head.get(2).unwrap_or(&0)); // profile_compatibility
            out.push(*head.get(3).unwrap_or(&0)); // AVCLevelIndication
            out.push(0xFC | (nal_len - 1)); // lengthSizeMinusOne
            out.push(0xE0 | (sps.len().min(31) as u8)); // numOfSequenceParameterSets
            for s in sps.iter().take(31) {
                out.extend_from_slice(&(s.len() as u16).to_be_bytes());
                out.extend_from_slice(s);
            }
            out.push(pps.len().min(255) as u8);
            for p in pps.iter().take(255) {
                out.extend_from_slice(&(p.len() as u16).to_be_bytes());
                out.extend_from_slice(p);
            }
            Ok(out)
        }
        CodecId::H265 => {
            let mut vps: Vec<&[u8]> = Vec::new();
            let mut sps: Vec<&[u8]> = Vec::new();
            let mut pps: Vec<&[u8]> = Vec::new();
            for n in &nals {
                match codec::h265_nal_type(n) {
                    Some(32) => vps.push(n),
                    Some(33) => sps.push(n),
                    Some(34) => pps.push(n),
                    _ => {}
                }
            }
            if sps.is_empty() || pps.is_empty() {
                return Err(MediaError::Bitstream(
                    "H.265 的 extradata 至少要含 SPS(33) 与 PPS(34)".into(),
                ));
            }
            let mut out = Vec::with_capacity(annexb.len() + 32);
            out.push(1); // configurationVersion
            out.push(0x01); // general_profile_space/tier/profile_idc
            out.extend_from_slice(&[0, 0, 0, 0]); // compatibility flags
            out.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // constraint flags
            out.push(0); // general_level_idc
            out.extend_from_slice(&0xF000u16.to_be_bytes()); // min_spatial_segmentation_idc
            out.push(0xFC); // parallelismType
            out.push(0xFC | 1); // chromaFormat = 1 (4:2:0)
            out.push(0xF8); // bitDepthLumaMinus8 = 0
            out.push(0xF8); // bitDepthChromaMinus8 = 0
            out.extend_from_slice(&0u16.to_be_bytes()); // avgFrameRate
            out.push(0x0F | (nal_len - 1)); // constantFrameRate/numTemporalLayers/temporalIdNested/lengthSizeMinusOne
            let arrays: Vec<(u8, &Vec<&[u8]>)> =
                vec![(32, &vps), (33, &sps), (34, &pps)];
            let present: Vec<&(u8, &Vec<&[u8]>)> = arrays.iter().filter(|(_, l)| !l.is_empty()).collect();
            out.push(present.len() as u8);
            for (ty, list) in present {
                out.push(0x80 | ty); // array_completeness = 1
                out.extend_from_slice(&(list.len() as u16).to_be_bytes());
                for n in list.iter() {
                    out.extend_from_slice(&(n.len() as u16).to_be_bytes());
                    out.extend_from_slice(n);
                }
            }
            Ok(out)
        }
        other => Err(MediaError::Unsupported(format!(
            "{} 没有 avcC/hvcC 形态的 extradata",
            other.name()
        ))),
    }
}

/// `avcC` / `hvcC` 载荷 → 参数集（Annex-B 形态，含 4 字节起始码）。
pub fn avcc_to_annexb_extradata(codec: &CodecId, extradata: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(extradata.len() + 32);
    match codec {
        CodecId::H264 => {
            if extradata.len() < 7 {
                return Err(MediaError::Bitstream("avcC 太短".into()));
            }
            let mut pos = 5usize;
            let num_sps = (extradata[pos] & 0x1F) as usize;
            pos += 1;
            for _ in 0..num_sps {
                let len = read_u16(extradata, pos)?;
                pos += 2;
                let n = extradata.get(pos..pos + len).ok_or_else(|| MediaError::Bitstream("avcC 越界".into()))?;
                out.extend_from_slice(&[0, 0, 0, 1]);
                out.extend_from_slice(n);
                pos += len;
            }
            let num_pps = *extradata.get(pos).ok_or_else(|| MediaError::Bitstream("avcC 缺 PPS".into()))? as usize;
            pos += 1;
            for _ in 0..num_pps {
                let len = read_u16(extradata, pos)?;
                pos += 2;
                let n = extradata.get(pos..pos + len).ok_or_else(|| MediaError::Bitstream("avcC 越界".into()))?;
                out.extend_from_slice(&[0, 0, 0, 1]);
                out.extend_from_slice(n);
                pos += len;
            }
        }
        CodecId::H265 => {
            if extradata.len() < 23 {
                return Err(MediaError::Bitstream("hvcC 太短".into()));
            }
            let arrays = extradata[22] as usize;
            let mut pos = 23usize;
            for _ in 0..arrays {
                let ty = *extradata.get(pos).ok_or_else(|| MediaError::Bitstream("hvcC 越界".into()))? & 0x3F;
                let count = read_u16(extradata, pos + 1)?;
                pos += 3;
                // 只取参数集（VPS/SPS/PPS），别的 NAL 类型（SEI 之类）不要
                let keep = matches!(ty, 32..=34);
                for _ in 0..count {
                    let len = read_u16(extradata, pos)?;
                    pos += 2;
                    let n = extradata.get(pos..pos + len).ok_or_else(|| MediaError::Bitstream("hvcC 越界".into()))?;
                    if keep {
                        out.extend_from_slice(&[0, 0, 0, 1]);
                        out.extend_from_slice(n);
                    }
                    pos += len;
                }
            }
        }
        other => {
            return Err(MediaError::Unsupported(format!(
                "{} 没有 avcC/hvcC 形态的 extradata",
                other.name()
            )))
        }
    }
    if out.is_empty() {
        return Err(MediaError::Bitstream("extradata 里没解出任何参数集".into()));
    }
    Ok(out)
}

/// `extradata` → Annex-B 参数集（**自动识别来源形态**）。
pub fn extradata_to_annexb(codec: &CodecId, extradata: &[u8]) -> Result<Vec<u8>> {
    if extradata.is_empty() {
        return Err(MediaError::Bitstream("extradata 为空".into()));
    }
    if looks_like_avcc(codec, extradata) {
        return avcc_to_annexb_extradata(codec, extradata);
    }
    if codec::split_annexb(extradata).is_empty() {
        return Err(MediaError::Bitstream(
            "extradata 既不是 Annex-B 参数集，也不是 avcC/hvcC".into(),
        ));
    }
    Ok(extradata.to_vec())
}

/// `extradata` → `avcC`/`hvcC`/`av1C` 载荷（**自动识别来源形态**）。
pub fn extradata_to_avcc(codec: &CodecId, extradata: &[u8], nal_len: u8) -> Result<Vec<u8>> {
    // AV1 没有 Annex-B 形态：extradata 本来就是 `av1C`，原样用它。
    // 不像 `av1C` 就**明确报错**，不要掉进"按 NAL 拆"那条路 —— 那条路对 OBU 流
    // 只会给出看不懂的错误（"找不到起始码"）。
    if matches!(codec, CodecId::Av1) {
        if !is_av1c(extradata) {
            let head: Vec<String> = extradata
                .iter()
                .take(4)
                .map(|b| format!("{b:02X}"))
                .collect();
            return Err(MediaError::Bitstream(format!(
                "AV1 的 extradata 必须是 av1C（首字节 0x81），读到 [{}]（共 {} 字节）",
                head.join(" "),
                extradata.len()
            )));
        }
        return Ok(extradata.to_vec());
    }
    if looks_like_avcc(codec, extradata) {
        return Ok(extradata.to_vec());
    }
    annexb_to_avcc_extradata(codec, extradata, nal_len)
}

/// 从 `extradata` 里取 NALU 长度前缀的字节数（`avcC`/`hvcC` 里的 `lengthSizeMinusOne`）。
pub fn nal_length_size(codec: &CodecId, extradata: &[u8]) -> u8 {
    match codec {
        CodecId::H264 => extradata.get(4).map(|b| (b & 0x03) + 1).unwrap_or(4),
        CodecId::H265 => extradata.get(21).map(|b| (b & 0x03) + 1).unwrap_or(4),
        _ => 4,
    }
}

/// 引擎里的采样 → **解码层要的形态**。
///
/// - H.264 / H.265：引擎里统一是 Annex-B，解码层要长度前缀（AVCC）；
/// - **AV1 / VP9 / VP8：原样透传** —— 它们的采样自带长度信息（AV1 的每个 OBU
///   带 `leb128` 长度），根本没有 Annex-B 形态。
///
/// 为什么这条要在**采样的每一个消费点**都走它：按 NAL 去拆 AV1 的 OBU 流会
/// 拆不出任何 NAL，于是"采样变成空的" → 调用方 `continue` 跳过 → 整条流
/// **一帧都不解、进程还退出 0**（实测踩到：`decode --out` 报 `frames: 0`，
/// 没有任何报错，比失败更难发现）。
pub fn to_decoder_sample(codec: &CodecId, engine_sample: &[u8], nal_len: u8) -> Vec<u8> {
    match codec {
        CodecId::Av1 | CodecId::Vp9 | CodecId::Vp8 => engine_sample.to_vec(),
        _ => annexb_to_avcc(engine_sample, nal_len),
    }
}

/// Annex-B 码流 → 长度前缀布局。
pub fn annexb_to_avcc(data: &[u8], nal_len: u8) -> Vec<u8> {
    let nal_len = nal_len.clamp(1, 4) as usize;
    let nals = codec::split_annexb(data);
    let mut out = Vec::with_capacity(data.len() + nals.len() * nal_len);
    for n in nals {
        let len = n.len();
        for i in (0..nal_len).rev() {
            out.push(((len >> (8 * i)) & 0xFF) as u8);
        }
        out.extend_from_slice(n);
    }
    out
}

/// 长度前缀布局 → Annex-B 码流。
pub fn avcc_to_annexb(data: &[u8], nal_len: u8) -> Vec<u8> {
    let nal_len = nal_len.clamp(1, 4) as usize;
    let mut out = Vec::with_capacity(data.len() + 8);
    let mut pos = 0usize;
    while pos + nal_len <= data.len() {
        let mut len = 0usize;
        for i in 0..nal_len {
            len = (len << 8) | data[pos + i] as usize;
        }
        pos += nal_len;
        if len == 0 {
            continue;
        }
        if pos + len > data.len() {
            // 长度越界：把剩下的当最后一个 NAL 带上（坏采样丢帧，不丢整条流）
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(&data[pos..]);
            return out;
        }
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&data[pos..pos + len]);
        pos += len;
    }
    if out.is_empty() && !data.is_empty() {
        return data.to_vec();
    }
    out
}

/// 这段 Annex-B 码流里有没有参数集（SPS/PPS/VPS）。
pub fn has_parameter_sets(codec: &CodecId, annexb: &[u8]) -> bool {
    let hevc = matches!(codec, CodecId::H265);
    codec::split_annexb(annexb).iter().any(|n| {
        if hevc {
            matches!(codec::h265_nal_type(n), Some(32) | Some(33) | Some(34))
        } else {
            matches!(codec::h264_nal_type(n), Some(7) | Some(8))
        }
    })
}

fn read_u16(buf: &[u8], pos: usize) -> Result<usize> {
    let a = *buf.get(pos).ok_or_else(|| MediaError::Bitstream("描述符越界".into()))?;
    let b = *buf.get(pos + 1).ok_or_else(|| MediaError::Bitstream("描述符越界".into()))?;
    Ok(u16::from_be_bytes([a, b]) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 造一个 1920x1080 的 SPS（与 me-container 测试里用的是同一套位序列）。
    fn sps_1080p() -> Vec<u8> {
        fn push(bits: &mut Vec<u8>, v: u64, n: u32) {
            for i in (0..n).rev() {
                bits.push(((v >> i) & 1) as u8);
            }
        }
        fn ue(bits: &mut Vec<u8>, v: u32) {
            let v1 = v as u64 + 1;
            let n = 64 - v1.leading_zeros();
            for _ in 0..n - 1 {
                bits.push(0);
            }
            for i in (0..n).rev() {
                bits.push(((v1 >> i) & 1) as u8);
            }
        }
        let mut bits = Vec::new();
        push(&mut bits, 66, 8);
        push(&mut bits, 0, 8);
        push(&mut bits, 30, 8);
        ue(&mut bits, 0);
        ue(&mut bits, 0);
        ue(&mut bits, 0);
        ue(&mut bits, 0);
        ue(&mut bits, 1);
        push(&mut bits, 0, 1);
        ue(&mut bits, 119);
        ue(&mut bits, 67);
        push(&mut bits, 1, 1);
        push(&mut bits, 1, 1);
        push(&mut bits, 1, 1);
        ue(&mut bits, 0);
        ue(&mut bits, 0);
        ue(&mut bits, 0);
        ue(&mut bits, 4);
        push(&mut bits, 0, 1);
        let mut bytes = vec![0x67u8];
        let mut cur = 0u8;
        let mut n = 0;
        for b in &bits {
            cur = (cur << 1) | b;
            n += 1;
            if n == 8 {
                bytes.push(cur);
                cur = 0;
                n = 0;
            }
        }
        if n > 0 {
            bytes.push(cur << (8 - n));
        }
        bytes
    }

    fn annexb_params() -> Vec<u8> {
        let mut v = vec![0, 0, 0, 1];
        v.extend_from_slice(&sps_1080p());
        v.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE, 0x3C, 0x80]); // PPS
        v
    }

    #[test]
    fn annexb_and_avcc_roundtrip() {
        let annexb = {
            let mut v = annexb_params();
            v.extend_from_slice(&[0, 0, 0, 1, 0x65, 0x11, 0x22]);
            v.extend_from_slice(&[0, 0, 1, 0x41, 0x33]); // 3 字节起始码
            v
        };
        let avcc = annexb_to_avcc(&annexb, 4);
        let back = avcc_to_annexb(&avcc, 4);
        // 3 字节起始码会被规范成 4 字节，所以比的是 NAL 序列
        let a: Vec<&[u8]> = codec::split_annexb(&annexb);
        let b: Vec<&[u8]> = codec::split_annexb(&back);
        assert_eq!(a, b, "往返之后 NAL 序列必须一致");
    }

    #[test]
    fn avcc_length_prefix_uses_the_requested_width() {
        let annexb = vec![0, 0, 0, 1, 0x65, 0xAA, 0xBB];
        let two = annexb_to_avcc(&annexb, 2);
        assert_eq!(two, vec![0x00, 0x03, 0x65, 0xAA, 0xBB]);
        let four = annexb_to_avcc(&annexb, 4);
        assert_eq!(four, vec![0x00, 0x00, 0x00, 0x03, 0x65, 0xAA, 0xBB]);
    }

    #[test]
    fn extradata_converts_both_ways_and_detects_form() {
        let annexb = annexb_params();
        let avcc = annexb_to_avcc_extradata(&CodecId::H264, &annexb, 4).unwrap();
        assert!(looks_like_avcc(&CodecId::H264, &avcc), "产出的必须是合法 avcC");
        assert!(!looks_like_avcc(&CodecId::H264, &annexb), "Annex-B 参数集不该被认成 avcC");
        assert_eq!(nal_length_size(&CodecId::H264, &avcc), 4);

        let back = avcc_to_annexb_extradata(&CodecId::H264, &avcc).unwrap();
        let a: Vec<&[u8]> = codec::split_annexb(&annexb);
        let b: Vec<&[u8]> = codec::split_annexb(&back);
        assert_eq!(a, b, "参数集往返必须一致");

        // 自动识别：两种形态都能转成另一种
        assert_eq!(extradata_to_avcc(&CodecId::H264, &annexb, 4).unwrap(), avcc);
        assert_eq!(extradata_to_annexb(&CodecId::H264, &avcc).unwrap(), back);
    }

    #[test]
    fn extradata_errors_are_explicit_not_panics() {
        assert!(annexb_to_avcc_extradata(&CodecId::H264, &[], 4).is_err(), "空参数集要报错");
        // 只有 SPS 没有 PPS
        let mut only_sps = vec![0, 0, 0, 1];
        only_sps.extend_from_slice(&sps_1080p());
        assert!(annexb_to_avcc_extradata(&CodecId::H264, &only_sps, 4).is_err());
        // 垃圾字节
        assert!(extradata_to_annexb(&CodecId::H264, &[0x11; 16]).is_err());
        assert!(extradata_to_annexb(&CodecId::H264, &[]).is_err());
    }

    #[test]
    fn has_parameter_sets_detects_them() {
        assert!(has_parameter_sets(&CodecId::H264, &annexb_params()));
        assert!(!has_parameter_sets(&CodecId::H264, &[0, 0, 0, 1, 0x65, 0x11]));
    }

    #[test]
    fn av1_extradata_is_passed_through_untouched() {
        // av1C：marker(0x81) + profile/level + 位深/色度 + 保留
        let av1c = vec![0x81u8, 0x00, 0x0C, 0x00];
        assert!(is_av1c(&av1c));
        assert!(looks_like_avcc(&CodecId::Av1, &av1c), "av1C 本身就是配置盒的形态");
        // 关键：**原样透传**。AV1 没有 Annex-B 形态，走"按 NAL 拆"那条路会
        // 把 OBU 流拆坏（或者报"找不到起始码"）
        assert_eq!(extradata_to_avcc(&CodecId::Av1, &av1c, 4).unwrap(), av1c);
    }

    #[test]
    fn av1_extradata_that_is_not_av1c_says_why() {
        let err = extradata_to_avcc(&CodecId::Av1, &[0x00, 0x00, 0x00, 0x01, 0x67], 4);
        let text = format!("{}", err.expect_err("不是 av1C 就该报错"));
        assert!(text.contains("av1C"), "错误里要指名 av1C：{text}");
        assert!(
            !text.contains("起始码"),
            "不能掉进 NAL 拆分那条路（那会给出看不懂的错误）：{text}"
        );
        assert!(!is_av1c(&[0x81, 0x00]), "太短的也不算 av1C");
    }

    #[test]
    fn avcc_to_annexb_survives_broken_lengths() {
        let data = vec![0, 0, 0, 99, 0x65, 0x11];
        let out = avcc_to_annexb(&data, 4);
        assert_eq!(&out[..4], &[0, 0, 0, 1]);
        assert!(out.ends_with(&[0x65, 0x11]));
        // nal_len = 0 会被夹到 1，不能 panic
        let _ = avcc_to_annexb(&data, 0);
        let _ = annexb_to_avcc(&data, 0);
    }
}
