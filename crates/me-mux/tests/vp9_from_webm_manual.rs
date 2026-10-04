//! **手工验证**（默认 `#[ignore]`）：把一段真实 VP9 的 WebM 帧写进 fMP4，
//! 交给外部播放器（macOS 的 AVFoundation / `afinfo`）检查能不能解。
//!
//! 为什么要它：VP9 这条链在本仓库里**没有别的输入源**（引擎还没有 WebM 解封装，
//! 而 VP9 in fMP4 的配置盒 `vpcC` 只能来自 fMP4 或 Matroska 的 `CodecPrivate`），
//! 于是"自己写、自己读"的往返就成了唯一证据 —— 那证明不了**别人的播放器**认不认。
//! 这个测试用真实码流补上这一点：
//!
//! ```sh
//! curl -o /tmp/vp9.webm \
//!   https://test-videos.co.uk/vids/bigbuckbunny/webm/vp9/360/Big_Buck_Bunny_360_10s_1MB.webm
//! WEBM_FILE=/tmp/vp9.webm VP9_OUT=/tmp/vp9.mp4 \
//!   cargo test -p me-mux --test vp9_from_webm_manual -- --ignored --nocapture
//! ```
//!
//! 然后用系统框架核实（本引擎不依赖它们，它们只是"局外裁判"）：
//!
//! ```sh
//! afinfo /tmp/vp9.mp4                       # 应当报 640x360 vp09
//! swift /tmp/avcheck.swift /tmp/vp9.mp4     # AVAsset 读出时长与 subtype=vp09
//! ```
//!
//! ## 两个来自真实素材的事实（顺手记下来）
//!
//! 1. **Matroska 的 VP9 `CodecPrivate` 就是 `vpcC` 的载荷**（不含 box 头）——
//!    与 fMP4 里的配置盒载荷**同一串字节**，所以将来做 WebM 解封装时这条链是零转换的。
//! 2. **常见 WebM 根本没有 `CodecPrivate`**（本测试用的这版就没有，libvpx 产出的文件
//!    大多如此）。这时只能从**第一帧的帧头**读出 profile 与位深，其余（色度采样、
//!    色彩三要素）按 VP9 的默认值填。本测试就带着这个兜底实现 —— 它同时说明了一件事：
//!    "VP9 轨缺 vpcC 就报错"这条纪律对 **fMP4 → fMP4** 是对的（配置盒本该原样带着），
//!    但对将来的 WebM 输入必须配一个"从帧头推导"的转换，否则真实文件全都进不来。

use me_core::{CodecId, MediaKind, Packet, Timebase, Track};
use me_mux::Mp4Muxer;

/// VP9 帧所在的 WebM 文件（`WEBM_FILE`）。
fn webm_path() -> std::path::PathBuf {
    std::env::var("WEBM_FILE").expect("要设 WEBM_FILE=/path/to.webm").into()
}

/// 产物路径（`VP9_OUT`）。
fn out_path() -> std::path::PathBuf {
    std::env::var("VP9_OUT").expect("要设 VP9_OUT=/path/out.mp4").into()
}

// ── 极简 EBML 读取（只够读 WebM 的这几层） ────────────────────────────────

/// 读元素 ID：**长度位要保留**（它就是 ID 的一部分）。
fn read_id(buf: &[u8], pos: usize) -> Option<(u64, usize)> {
    let first = *buf.get(pos)?;
    if first == 0 {
        return None;
    }
    let len = first.leading_zeros() as usize + 1;
    if len > 8 || pos + len > buf.len() {
        return None;
    }
    let mut v = 0u64;
    for i in 0..len {
        v = (v << 8) | buf[pos + i] as u64;
    }
    Some((v, pos + len))
}

/// 读长度：剥掉长度标记；`unknown` 表示"长度未知"（**所有值位全 1**，常见于整段 Segment）。
///
/// 判据必须是"值位全 1"而不是"等于某个字节掩码"：长度字段可以是 1~8 字节，
/// 8 字节时值位有 56 位 —— 按首字节比会把任何长度都判成"未知"，然后解析器一路读到文件尾。
fn read_size(buf: &[u8], pos: usize) -> Option<(u64, usize, bool)> {
    let first = *buf.get(pos)?;
    if first == 0 {
        return None;
    }
    let len = first.leading_zeros() as usize + 1;
    if len > 8 || pos + len > buf.len() {
        return None;
    }
    let mut v = 0u64;
    for i in 0..len {
        v = (v << 8) | buf[pos + i] as u64;
    }
    let value_bits = 7 * len;
    let mask = (1u64 << value_bits) - 1;
    v &= mask;
    Some((v, pos + len, v == mask))
}

/// 一层里的所有子元素：`(id, 载荷起, 载荷止)`。
fn children(buf: &[u8], start: usize, end: usize) -> Vec<(u64, usize, usize)> {
    let mut out = Vec::new();
    let mut p = start;
    while p < end {
        let Some((id, after_id)) = read_id(buf, p) else { break };
        let Some((size, after_size, unknown)) = read_size(buf, after_id) else { break };
        // 长度未知（值位全 1）表示"到本层末尾为止"；其余按声明长度走，并夹在本层内
        let stop = if unknown { end } else { (after_size + size as usize).min(end) };
        out.push((id, after_size, stop));
        // 空元素（size = 0）是合法的，游标照样前进 —— 这里只防"游标不动"的死循环
        if stop <= p {
            break;
        }
        p = stop;
    }
    out
}

/// 在指定层里找第一个 id 命中的元素的载荷。
fn find(buf: &[u8], start: usize, end: usize, want: u64) -> Option<&[u8]> {
    children(buf, start, end)
        .into_iter()
        .find(|(id, _, _)| *id == want)
        .map(|(_, s, e)| &buf[s..e])
}

/// WebM 里我们关心的东西。
struct Webm {
    codec_id: String,
    /// Matroska 的 `CodecPrivate`（VP9 的话就是 vpcC 的载荷，常常缺失）。
    codec_private: Option<Vec<u8>>,
    width: u32,
    height: u32,
    /// 时间戳单位（纳秒）。
    timescale_ns: u64,
    /// `(时间戳, 是否关键帧, 帧字节)`
    frames: Vec<(i64, bool, Vec<u8>)>,
}

const ID_SEGMENT: u64 = 0x1853_8067;
const ID_INFO: u64 = 0x1549_A966;
const ID_TIMESTAMP_SCALE: u64 = 0x2A_D7B1;
const ID_TRACKS: u64 = 0x1654_AE6B;
const ID_TRACK_ENTRY: u64 = 0xAE;
const ID_CODEC_ID: u64 = 0x86;
const ID_CODEC_PRIVATE: u64 = 0x63A2;
const ID_VIDEO: u64 = 0xE0;
const ID_PIXEL_WIDTH: u64 = 0xB0;
const ID_PIXEL_HEIGHT: u64 = 0xBA;
const ID_CLUSTER: u64 = 0x1F43_B675;
const ID_CLUSTER_TIME: u64 = 0xE7;
const ID_SIMPLE_BLOCK: u64 = 0xA3;

fn u64_of(b: &[u8]) -> u64 {
    b.iter().fold(0u64, |a, &x| (a << 8) | x as u64)
}

fn read_webm(buf: &[u8]) -> Webm {
    let (seg_s, seg_e) = children(buf, 0, buf.len())
        .into_iter()
        .find(|(id, _, _)| *id == ID_SEGMENT)
        .map(|(_, s, e)| (s, e))
        .expect("EBML 里要有 Segment");

    let mut w = Webm {
        codec_id: String::new(),
        codec_private: None,
        width: 0,
        height: 0,
        timescale_ns: 1_000_000, // 默认 1 ms
        frames: Vec::new(),
    };

    if let Some(info) = find(buf, seg_s, seg_e, ID_INFO) {
        let base = info.as_ptr() as usize - buf.as_ptr() as usize;
        if let Some(ts) = find(buf, base, base + info.len(), ID_TIMESTAMP_SCALE) {
            if !ts.is_empty() {
                w.timescale_ns = u64_of(ts);
            }
        }
    }

    if let Some(tracks) = find(buf, seg_s, seg_e, ID_TRACKS) {
        let base = tracks.as_ptr() as usize - buf.as_ptr() as usize;
        for (id, s, e) in children(buf, base, base + tracks.len()) {
            if id != ID_TRACK_ENTRY {
                continue;
            }
            for (k, ks, ke) in children(buf, s, e) {
                let payload = &buf[ks..ke];
                match k {
                    ID_CODEC_ID => w.codec_id = String::from_utf8_lossy(payload).into_owned(),
                    ID_CODEC_PRIVATE => w.codec_private = Some(payload.to_vec()),
                    // 宽高在 `Video`(0xE0) 里，不是 TrackEntry 的直接子元素
                    ID_VIDEO => {
                        for (vk, vs, ve) in children(buf, ks, ke) {
                            match vk {
                                ID_PIXEL_WIDTH => w.width = u64_of(&buf[vs..ve]) as u32,
                                ID_PIXEL_HEIGHT => w.height = u64_of(&buf[vs..ve]) as u32,
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    let mut cluster_time = 0i64;
    for (id, s, e) in children(buf, seg_s, seg_e) {
        if id != ID_CLUSTER {
            continue;
        }
        for (k, ks, ke) in children(buf, s, e) {
            match k {
                ID_CLUSTER_TIME => cluster_time = u64_of(&buf[ks..ke]) as i64,
                ID_SIMPLE_BLOCK => {
                    // SimpleBlock = track number(vint) + 时间戳(i16 BE) + flags + 帧
                    let Some((track, after, _)) = read_size(buf, ks) else { continue };
                    if track != 1 || after + 3 > ke {
                        continue;
                    }
                    let ts = i16::from_be_bytes([buf[after], buf[after + 1]]) as i64;
                    let flags = buf[after + 2];
                    let data = buf[after + 3..ke].to_vec();
                    if data.is_empty() {
                        continue;
                    }
                    w.frames.push((cluster_time + ts, flags & 0x80 != 0, data));
                }
                _ => {}
            }
        }
    }
    w
}

/// 从 VP9 帧头读出 `(profile, bit_depth)`（只读到 color_config 的位深那一步）。
///
/// VP9 的帧头是**按位**的（MSB 优先）：`frame_marker(2) = 0b10`、
/// `profile_low_bit`、`profile_high_bit`（profile 3 时多一个保留位），
/// 然后 `show_existing_frame`；关键帧的 color_config 里才有位深
/// （profile ≥ 2 时 1 位区分 10/12 位，profile ≤ 1 恒为 8 位）。
fn vp9_profile_and_depth(frame: &[u8]) -> Option<(u8, u8)> {
    let mut r = BitCursor { buf: frame, bit: 0 };
    if r.take(2)? != 0b10 {
        return None; // frame_marker 不对：这不是 VP9 帧
    }
    let low = r.take(1)? as u8;
    let high = r.take(1)? as u8;
    let profile = (high << 1) | low;
    if profile == 3 {
        r.take(1)?; // reserved_zero
    }
    if r.take(1)? == 1 {
        // show_existing_frame：这帧只是"再显示一遍已有的帧"，没有自己的帧头
        return Some((profile, 8));
    }
    let frame_type = r.take(1)?; // 0 = key
    let show_frame = r.take(1)?;
    r.take(1)?; // error_resilient_mode
    if frame_type != 0 || show_frame != 1 {
        // 非关键帧不带 color_config：位深只能按 profile 推断
        return Some((profile, if profile >= 2 { 10 } else { 8 }));
    }
    let depth = if profile >= 2 { if r.take(1)? == 1 { 12 } else { 10 } } else { 8 };
    Some((profile, depth))
}

struct BitCursor<'a> {
    buf: &'a [u8],
    bit: usize,
}

impl BitCursor<'_> {
    fn take(&mut self, n: usize) -> Option<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            let byte = *self.buf.get(self.bit / 8)?;
            v = (v << 1) | ((byte >> (7 - (self.bit % 8))) & 1) as u32;
            self.bit += 1;
        }
        Some(v)
    }
}

/// 造一个 `vpcC` 的载荷（version 1，11 字节）。
///
/// 色度采样与色彩三要素在帧头里要到 color_config 更深处才有，这里按 VP9 的常见默认写：
/// 4:2:0（与亮度同位）、有限范围、BT.709；`level` 未知写 0（表示"未指明"）。
fn vpcc(profile: u8, bit_depth: u8) -> Vec<u8> {
    let mut v = vec![1u8, 0, 0, 0]; // version 1 + flags
    v.push(profile);
    v.push(0); // level：未指明
    v.push((bit_depth << 4) | 1); // bitDepth(4) | chromaSubsampling(4)，1 = 4:2:0 与亮度同位
    // 接下来是 10 个位，**不是字节对齐的**：
    //   videoFullRangeFlag(1) | colourPrimaries(3) | transferCharacteristics(3) | matrixCoefficients(3)
    // 按位流拼成一个 10 位数，再写成"高 8 位 + 低 2 位（补 6 个 0）"两个字节。
    // 最高那位是 videoFullRangeFlag：这里要的是**有限范围**，所以它是 0、不参与或运算。
    let bits: u16 = (1 << 6) | (1 << 3) | 1; // primaries / transfer / matrix 都取 BT.709(1)
    v.push((bits >> 2) as u8);
    v.push(((bits & 0x03) << 6) as u8);
    v.extend_from_slice(&0u16.to_be_bytes()); // codecInitializationDataSize = 0
    v
}

#[test]
#[ignore = "手工验证：需要一个真实的 VP9 WebM（WEBM_FILE）"]
fn write_real_vp9_webm_frames_into_fmp4() {
    let webm = read_webm(&std::fs::read(webm_path()).expect("读不到 WEBM_FILE"));
    assert_eq!(webm.codec_id, "V_VP9", "这个测试只处理 VP9 的 WebM");
    println!(
        "WebM：{}x{} · timescale {} ns · {} 帧 · CodecPrivate {}",
        webm.width,
        webm.height,
        webm.timescale_ns,
        webm.frames.len(),
        webm.codec_private.as_ref().map(|c| c.len()).unwrap_or(0)
    );
    assert!(webm.frames.len() > 10, "素材太小，看不出问题");

    let extradata = match &webm.codec_private {
        Some(c) => {
            println!("用 Matroska 的 CodecPrivate 当 vpcC（{} 字节）", c.len());
            c.clone()
        }
        None => {
            let (profile, depth) =
                vp9_profile_and_depth(&webm.frames[0].2).expect("第一帧不是 VP9 帧");
            println!("WebM 没有 CodecPrivate：从帧头读出 profile={profile} bitDepth={depth}");
            vpcc(profile, depth)
        }
    };

    // 时基用微秒：VP9 没有固定的帧率，时间戳直接用容器给的
    let mut track = Track::new(1, MediaKind::Video, CodecId::Vp9, Timebase::new(1_000_000));
    track.width = webm.width;
    track.height = webm.height;
    track.extradata = extradata;

    let file = std::fs::File::create(out_path()).expect("建不了 VP9_OUT");
    let mut m = Mp4Muxer::with_writer(Box::new(std::io::BufWriter::new(file)));
    m.set_tracks(&[track]).expect("VP9 轨应当能写进 fMP4");

    let to_us = |ts: i64| ts * webm.timescale_ns as i64 / 1000;
    let n = webm.frames.len();
    for i in 0..n {
        let (ts, key, data) = &webm.frames[i];
        let dts = to_us(*ts);
        let duration = if i + 1 < n {
            (to_us(webm.frames[i + 1].0) - dts).max(1) as u32
        } else {
            // 最后一帧：沿用上一帧的间隔
            (dts - to_us(webm.frames[i - 1].0)).max(1) as u32
        };
        let mut p = Packet::new(1, dts, dts, duration, data.clone());
        p.keyframe = *key;
        m.write_packet(&p).expect("写采样");
    }
    m.finish().expect("收尾");
    let size = std::fs::metadata(out_path()).expect("产物不见了").len();
    println!("写出 {} （{} 字节，{} 帧）", out_path().display(), size, n);
}
