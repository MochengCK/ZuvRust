//! FLV（`.flv`）封装：`FLV header + [Tag]*`，H.264 画面 + AAC / MP3 声音。
//!
//! ## 为什么补它
//!
//! 合并产物此前只有 MP4 / MKV / WebM / TS 这几家。FLV 是**老站点与老资源**的
//! 常见形态（早期 B 站、优酷、大量网课/录屏工具直接产 FLV），下载下来要"画面 +
//! 声音合成一个文件"时，没有 FLV 就只能改扩展名 —— 那不算支持。
//!
//! ## 必须做对的几件事（错一条就是"能生成但播不了"）
//!
//! 1. **每个 Tag 末尾的 `PreviousTagSize` 要如实写**（= 11 + DataSize），
//!    解析器靠它跳下一个 Tag；写错一个，后面的全乱。
//! 2. **时间戳是 32 位毫秒**（低 24 位 + 高 8 位分开存），单位与 MP4 的刻度不同，
//!    必须从时基换算；**视频 Tag 的时间戳是 DTS**，`CompositionTime`（SI24 毫秒）
//!    才是 `pts - dts`。
//! 3. **H.264 的采样要长度前缀布局**（与 fMP4 一样），而引擎内部是 Annex-B ⇒
//!    转换走 [`crate::convert`]；SPS/PPS 只放在 sequence header（`avcC`）里，
//!    采样里的参数集 NAL 要**剔掉**（重复的 SPS/PPS 会让部分解码器报错）。
//! 4. **AAC 的 sequence header 是 `AudioSpecificConfig`**（extradata 原样），
//!    采样是**裸帧**（从 TS 来的 `ADTS` 头要剥掉）；MP3 没有 sequence header，
//!    每一帧自带帧头，首字节只描述格式/采样率/声道。
//! 5. **`onMetaData` 里的 duration 只能来自调用方的声明时长**（分片输入的
//!    `mvhd.duration` 常是 0）。没有声明就不写这一项 —— 写 0 会让播放器显示 0 秒。

use std::io::Write;

use me_core::{CodecId, MediaError, MediaKind, Packet, Result, Track, Timebase};

use crate::convert;
use crate::sink::Sink;

/// Tag 类型。
const TAG_AUDIO: u8 = 8;
const TAG_VIDEO: u8 = 9;
const TAG_SCRIPT: u8 = 18;

/// 画面配置：`avcC` 载荷 + NAL 长度前缀宽度。
struct VideoCfg {
    track_id: u32,
    avcc: Vec<u8>,
    nal_len: usize,
    width: u32,
    height: u32,
    timebase: Timebase,
}

/// 声音配置。
enum AudioKind {
    /// `AudioSpecificConfig`（AAC 的 extradata）。
    Aac(Vec<u8>),
    /// MP3：首字节要写采样率档位与声道数。
    Mp3 { rate_index: u8, stereo: bool },
}

struct AudioCfg {
    track_id: u32,
    kind: AudioKind,
    sample_rate: u32,
    channels: u16,
    timebase: Timebase,
}

/// FLV 封装器（API 与 [`crate::Mp4Muxer`] 对齐）。
pub struct FlvMuxer {
    sink: Sink,
    video: Option<VideoCfg>,
    audio: Option<AudioCfg>,
    /// 全片共用的时间轴零点（微秒）：只用第一条采样的 dts 定，**不按轨各定一个**
    /// —— 各定一个会把音视频之间的真实偏移抹掉。
    base_us: Option<i64>,
    duration_hint_us: Option<i64>,
    declared: bool,
    packets_written: u64,
}

impl Default for FlvMuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl FlvMuxer {
    /// 内存落点（测试与小产物）。
    pub fn new() -> Self {
        Self::with_sink(Sink::mem())
    }

    /// 直写目标：产物**不驻留内存**。
    pub fn with_writer(w: Box<dyn Write + Send>) -> Self {
        Self::with_sink(Sink::writer(w))
    }

    fn with_sink(sink: Sink) -> Self {
        Self {
            sink,
            video: None,
            audio: None,
            base_us: None,
            duration_hint_us: None,
            declared: false,
            packets_written: 0,
        }
    }

    /// 声明总时长（微秒），只用于 `onMetaData` 的 `duration`。
    ///
    /// 必须在 [`Self::set_tracks`] **之前**调用：元数据 Tag 在那一刻就写出去了。
    pub fn set_duration_hint_us(&mut self, us: Option<i64>) {
        self.duration_hint_us = us.filter(|v| *v > 0);
    }

    /// 声明轨道（最多一条 H.264 画面 + 一条 AAC/MP3 声音），并写出文件头、
    /// 元数据与两条 sequence header。
    pub fn set_tracks(&mut self, tracks: &[Track]) -> Result<()> {
        if tracks.is_empty() {
            return Err(MediaError::Container("FLV 至少需要一条轨道".into()));
        }
        if tracks.len() > 2 {
            return Err(MediaError::Unsupported(format!(
                "FLV 最多一条画面 + 一条声音，这里给了 {} 条",
                tracks.len()
            )));
        }
        self.video = None;
        self.audio = None;
        for t in tracks {
            match t.kind {
                MediaKind::Video => {
                    if self.video.is_some() {
                        return Err(MediaError::Unsupported("FLV 只能装一条画面轨".into()));
                    }
                    if t.codec != CodecId::H264 {
                        return Err(MediaError::Unsupported(format!(
                            "FLV 的画面只支持 H.264，这里是 {}（引擎不转码；换 .mkv / .mp4）",
                            t.codec.name()
                        )));
                    }
                    if t.extradata.is_empty() {
                        return Err(MediaError::Container(
                            "H.264 轨缺 extradata（没有 SPS/PPS 就写不出 sequence header）".into(),
                        ));
                    }
                    let avcc = convert::extradata_to_avcc(&CodecId::H264, &t.extradata, 4)?;
                    let nal_len = convert::nal_length_size(&CodecId::H264, &avcc) as usize;
                    self.video = Some(VideoCfg {
                        track_id: t.id,
                        avcc,
                        nal_len,
                        width: t.width,
                        height: t.height,
                        timebase: t.timebase,
                    });
                }
                MediaKind::Audio => {
                    if self.audio.is_some() {
                        return Err(MediaError::Unsupported("FLV 只能装一条声音轨".into()));
                    }
                    let kind = match &t.codec {
                        CodecId::Aac => {
                            if t.extradata.len() < 2 {
                                return Err(MediaError::Container(
                                    "AAC 轨缺 AudioSpecificConfig（写不出 sequence header）".into(),
                                ));
                            }
                            AudioKind::Aac(t.extradata.clone())
                        }
                        CodecId::Mp3 => AudioKind::Mp3 {
                            rate_index: mp3_rate_index(t.sample_rate),
                            stereo: t.channels > 1,
                        },
                        other => {                            return Err(MediaError::Unsupported(format!(
                                "FLV 的声音只支持 AAC / MP3，这里是 {}（换 .mkv / .mp4）",
                                other.name()
                            )))
                        }
                    };
                    self.audio = Some(AudioCfg {
                        track_id: t.id,
                        kind,
                        sample_rate: t.sample_rate,
                        channels: t.channels,
                        timebase: t.timebase,
                    });
                }
                other => {
                    return Err(MediaError::Unsupported(format!(
                        "FLV 装不下 {:?} 轨",
                        other
                    )))
                }
            }
        }
        self.declared = true;

        // ── 文件头 ──
        let flags = if self.audio.is_some() { 0x01 } else { 0 } | if self.video.is_some() { 0x04 } else { 0 };
        self.sink.write_all(b"FLV")?;
        self.sink.write_all(&[1, flags])?;
        self.sink.write_all(&9u32.to_be_bytes())?; // DataOffset
        self.sink.write_all(&0u32.to_be_bytes())?; // PreviousTagSize0

        // ── 元数据（时长/分辨率/编码）──
        let meta = self.metadata_payload();
        self.write_tag(TAG_SCRIPT, 0, &meta)?;

        // ── sequence header：解码器靠它初始化，必须排在第一条采样之前 ──
        if let Some(v) = &self.video {
            let mut body = vec![0x17, 0x00, 0, 0, 0];
            body.extend_from_slice(&v.avcc);
            self.write_tag(TAG_VIDEO, 0, &body)?;
        }
        if let Some(AudioCfg { kind: AudioKind::Aac(asc), .. }) = &self.audio {
            let mut body = vec![0xAF, 0x00];
            body.extend_from_slice(asc);
            self.write_tag(TAG_AUDIO, 0, &body)?;
        }
        Ok(())
    }

    /// 写一条采样。
    pub fn write_packet(&mut self, p: &Packet) -> Result<()> {
        if !self.declared {
            return Err(MediaError::Container("写采样之前必须先 set_tracks".into()));
        }
        let is_video = self.video.as_ref().is_some_and(|v| v.track_id == p.track_id);
        let is_audio = self.audio.as_ref().is_some_and(|a| a.track_id == p.track_id);
        if !is_video && !is_audio {
            return Err(MediaError::Container(format!("轨道 {} 不在 Tracks 里", p.track_id)));
        }
        let tb = self.timebase_of(p.track_id)?;
        let dts_us = tb.ticks_to_us(p.dts);
        let pts_us = tb.ticks_to_us(p.pts);
        let base = *self.base_us.get_or_insert(dts_us);
        let ts_ms = ((dts_us - base).max(0) / 1000) as u32;

        if is_video {
            let (nal_len, keyframe) = {
                let v = self.video.as_ref().expect("刚判过");
                (v.nal_len, p.keyframe)
            };
            let payload = avcc_payload(&p.data, nal_len, p.track_id)?;
            // CTS 是 SI24 毫秒；超出 ±8.3 秒只能报错（正常素材不会）
            let cts = ((pts_us - dts_us) as f64 / 1000.0).round() as i64;
            if !(-8_388_608..=8_388_607).contains(&cts) {
                return Err(MediaError::Bitstream(format!(
                    "轨道 {} 的合成时间偏移 {cts} ms 超出 FLV 的 24 位有符号范围",
                    p.track_id
                )));
            }
            let mut body = Vec::with_capacity(5 + payload.len());
            body.push(if keyframe { 0x17 } else { 0x27 }); // FrameType | CodecID=AVC
            body.push(1); // AVCPacketType = NALU
            body.extend_from_slice(&(cts as i32).to_be_bytes()[1..]);
            body.extend_from_slice(&payload);
            self.write_tag(TAG_VIDEO, ts_ms, &body)?;
            self.packets_written += 1;
            return Ok(());
        }

        // 声音
        let body = match &self.audio.as_ref().expect("刚判过").kind {
            AudioKind::Aac(_) => {
                let raw = crate::mp4::strip_adts(&p.data);
                let mut body = Vec::with_capacity(2 + raw.len());
                body.push(0xAF); // AAC / 44 kHz / 16 bit / stereo（AAC 恒为 3 档）
                body.push(1); // AACPacketType = raw
                body.extend_from_slice(&raw);
                body
            }
            AudioKind::Mp3 { rate_index, stereo } => {
                let mut body = Vec::with_capacity(1 + p.data.len());
                body.push(0x20 | (*rate_index << 2) | 0x02 | u8::from(*stereo));
                body.extend_from_slice(&p.data);
                body
            }
        };
        self.write_tag(TAG_AUDIO, ts_ms, &body)?;
        self.packets_written += 1;
        Ok(())
    }

    /// 收尾（FLV 没有尾元素），返回整份文件（内存模式）。
    pub fn finish(&mut self) -> Result<Vec<u8>> {
        self.sink.flush()?;
        Ok(self.sink.take_bytes())
    }

    /// 已经写出的字节数（进度上报用）。
    pub fn bytes_written(&self) -> u64 {
        self.sink.bytes_written()
    }

    pub fn packets_written(&self) -> u64 {
        self.packets_written
    }

    fn timebase_of(&self, track_id: u32) -> Result<Timebase> {
        if let Some(v) = &self.video {
            if v.track_id == track_id {
                return Ok(v.timebase);
            }
        }
        if let Some(a) = &self.audio {
            if a.track_id == track_id {
                return Ok(a.timebase);
            }
        }
        Err(MediaError::Container(format!("轨道 {track_id} 不在 Tracks 里")))
    }

    /// 写一个 Tag（含 `PreviousTagSize`）。
    fn write_tag(&mut self, tag_type: u8, ts_ms: u32, body: &[u8]) -> Result<()> {
        if body.len() > 0xFF_FFFF {
            return Err(MediaError::Unsupported(format!(
                "FLV 单个 Tag 最大 16 MiB，这里是 {} 字节",
                body.len()
            )));
        }
        let size = body.len();
        let mut tag = Vec::with_capacity(11 + size + 4);
        tag.push(tag_type);
        tag.extend_from_slice(&[(size >> 16) as u8, (size >> 8) as u8, size as u8]);
        // 时间戳：低 24 位是**大端**（UI24），高 8 位单独一个字节
        tag.extend_from_slice(&[
            ((ts_ms >> 16) & 0xFF) as u8,
            ((ts_ms >> 8) & 0xFF) as u8,
            (ts_ms & 0xFF) as u8,
            ((ts_ms >> 24) & 0xFF) as u8,
        ]);
        tag.extend_from_slice(&[0, 0, 0]); // StreamID
        tag.extend_from_slice(body);
        tag.extend_from_slice(&((11 + size) as u32).to_be_bytes());
        self.sink.write_all(&tag)?;
        Ok(())
    }

    /// `onMetaData`：播放器靠它显示时长与分辨率（没有它，时长要扫全文件才知道）。
    fn metadata_payload(&self) -> Vec<u8> {
        let mut props: Vec<(&str, Amf)> = Vec::new();
        if let Some(us) = self.duration_hint_us {
            props.push(("duration", Amf::Number(us as f64 / 1_000_000.0)));
        }
        if let Some(v) = &self.video {
            if v.width > 0 && v.height > 0 {
                props.push(("width", Amf::Number(v.width as f64)));
                props.push(("height", Amf::Number(v.height as f64)));
            }
            props.push(("videocodecid", Amf::Number(7.0)));
        }
        if let Some(a) = &self.audio {
            let id = match &a.kind {
                AudioKind::Aac(_) => 10.0,
                AudioKind::Mp3 { .. } => 2.0,
            };
            props.push(("audiocodecid", Amf::Number(id)));
            if a.sample_rate > 0 {
                props.push(("audiosamplerate", Amf::Number(a.sample_rate as f64)));
            }
            props.push(("audiosamplesize", Amf::Number(16.0)));
            props.push(("stereo", Amf::Bool(a.channels > 1)));
        }
        props.push(("encoder", Amf::Str("ZuvRust")));

        let mut v = amf_str("onMetaData");
        v.push(0x08); // ECMA array
        v.extend_from_slice(&(props.len() as u32).to_be_bytes());
        for (k, val) in props {
            v.extend_from_slice(&amf_str(k));
            match val {
                Amf::Number(n) => {
                    v.push(0x00);
                    v.extend_from_slice(&n.to_be_bytes());
                }
                Amf::Bool(b) => {
                    v.push(0x01);
                    v.push(u8::from(b));
                }
                Amf::Str(s) => v.extend_from_slice(&amf_str(s)),
            }
        }
        v.extend_from_slice(&[0, 0, 0x09]); // object end
        v
    }
}

enum Amf {
    Number(f64),
    Bool(bool),
    Str(&'static str),
}

/// AMF0 字符串。
fn amf_str(s: &str) -> Vec<u8> {
    let bytes = s.as_bytes();
    let mut v = Vec::with_capacity(3 + bytes.len());
    v.push(0x02);
    v.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
    v.extend_from_slice(bytes);
    v
}

/// MP3 的采样率档位（2 位）。AAC 在 FLV 里恒为 3 档，不走这里。
///
/// 48 kHz 没有独立档位（档位只到 44 kHz），按惯例写 3 —— FLV 的这两
/// 位只是"提示"，真实采样率在 MP3 帧头里。
fn mp3_rate_index(sample_rate: u32) -> u8 {
    match sample_rate {
        48_000 | 44_100 => 3,
        32_000 | 24_000 | 22_050 => 2,
        16_000 | 12_000 | 11_025 => 1,
        _ => 0,
    }
}

/// Annex-B 采样 → FLV 的长度前缀载荷（剔掉参数集，它们只在 sequence header 里）。
fn avcc_payload(data: &[u8], nal_len: usize, track_id: u32) -> Result<Vec<u8>> {
    let nals = me_container::codec::split_annexb(data);
    if nals.is_empty() {
        return Err(MediaError::Bitstream(format!(
            "轨道 {track_id} 的视频采样不是 Annex-B（找不到起始码）"
        )));
    }
    let mut out = Vec::with_capacity(data.len() + nals.len() * nal_len);
    for n in nals {
        // SPS/PPS 已经在 avcC（sequence header）里；采样里再带一遍会让部分解码器
        // 以"参数集不一致"报错。AUD 之类保留（无害）。
        if matches!(me_container::codec::h264_nal_type(n), Some(7) | Some(8)) {
            continue;
        }
        if n.len() >= 1usize << (8 * nal_len) {
            return Err(MediaError::Bitstream(format!(
                "轨道 {track_id} 的 NAL 有 {} 字节，超过 {nal_len} 字节长度前缀的上限",
                n.len()
            )));
        }
        let len = n.len();
        for i in (0..nal_len).rev() {
            out.push(((len >> (8 * i)) & 0xFF) as u8);
        }
        out.extend_from_slice(n);
    }
    if out.is_empty() {
        return Err(MediaError::Bitstream(format!(
            "轨道 {track_id} 的采样里只有参数集，没有可写的 NAL"
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn video_track() -> Track {
        let mut t = Track::new(1, MediaKind::Video, CodecId::H264, Timebase::new(30_000));
        t.width = 640;
        t.height = 360;
        // Annex-B 形态的 SPS + PPS：封装器要自己转成 avcC
        let mut ed = vec![0u8, 0, 0, 1];
        ed.extend_from_slice(&[
            0x67, 0x42, 0x00, 0x1E, 0x95, 0xA8, 0x28, 0x0F, 0x00, 0x44, 0xFC, 0xB8, 0x08, 0x80,
            0x00, 0x03, 0x00, 0x08, 0x00, 0x00, 0x03, 0x01, 0xE0, 0x78, 0xB1, 0x6C, 0x90,
        ]);
        ed.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE, 0x3C, 0x80]);
        t.extradata = ed;
        t
    }

    fn audio_track() -> Track {
        let mut t = Track::new(2, MediaKind::Audio, CodecId::Aac, Timebase::new(44_100));
        t.sample_rate = 44_100;
        t.channels = 2;
        t.extradata = vec![0x12, 0x10];
        t
    }

    fn video_packet(dts: i64, pts: i64, keyframe: bool) -> Packet {
        let mut v = vec![0u8, 0, 0, 1];
        v.push(if keyframe { 0x65 } else { 0x41 });
        v.extend_from_slice(&[0x11, 0x22, 0x33, 0x44]);
        let mut p = Packet::new(1, pts, dts, 1000, v);
        p.keyframe = keyframe;
        p
    }

    fn audio_packet(dts: i64) -> Packet {
        Packet::new(2, dts, dts, 1024, vec![0x21, 0x10, 0x05, 0x55])
    }

    struct Tag {
        ty: u8,
        ts: u32,
        body: Vec<u8>,
    }

    fn u24(b: &[u8]) -> u32 {
        ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32
    }

    /// 按 FLV 的结构逐 Tag 走一遍（顺带断言每个 `PreviousTagSize` 都对得上）。
    fn parse_tags(data: &[u8]) -> (u8, Vec<Tag>) {
        assert_eq!(&data[..3], b"FLV");
        assert_eq!(data[3], 1, "版本必须是 1");
        let flags = data[4];
        assert_eq!(u32::from_be_bytes(data[5..9].try_into().unwrap()), 9, "DataOffset 必须是 9");
        assert_eq!(u32::from_be_bytes(data[9..13].try_into().unwrap()), 0, "PreviousTagSize0 必须是 0");
        let mut pos = 13usize;
        let mut out = Vec::new();
        while pos < data.len() {
            let ty = data[pos];
            let size = u24(&data[pos + 1..]) as usize;
            let ts = u24(&data[pos + 4..]) | ((data[pos + 7] as u32) << 24);
            assert!(pos + 11 + size + 4 <= data.len(), "Tag 越界");
            let body = data[pos + 11..pos + 11 + size].to_vec();
            let prev = u32::from_be_bytes(
                data[pos + 11 + size..pos + 15 + size].try_into().unwrap(),
            );
            assert_eq!(prev as usize, 11 + size, "PreviousTagSize 必须 = 11 + DataSize");
            out.push(Tag { ty, ts, body });
            pos += 11 + size + 4;
        }
        (flags, out)
    }

    #[test]
    fn writes_header_metadata_sequence_headers_and_media_tags() {
        let mut m = FlvMuxer::new();
        m.set_duration_hint_us(Some(1_000_000));
        m.set_tracks(&[video_track(), audio_track()]).unwrap();
        m.write_packet(&video_packet(0, 1000, true)).unwrap();
        m.write_packet(&audio_packet(0)).unwrap();
        m.write_packet(&video_packet(1000, 2000, false)).unwrap();
        m.write_packet(&audio_packet(1024)).unwrap();
        let data = m.finish().unwrap();
        assert_eq!(m.packets_written(), 4, "只数采样，不数元数据/头 Tag");

        let (flags, tags) = parse_tags(&data);
        assert_eq!(flags, 0x05, "音视频都要在");
        assert_eq!(tags.len(), 7, "元数据 + 两条 sequence header + 四条采样");

        // 元数据：时长与编码标识
        assert_eq!(tags[0].ty, TAG_SCRIPT);
        let meta = String::from_utf8_lossy(&tags[0].body);
        assert!(meta.contains("onMetaData"), "脚本 Tag 必须叫 onMetaData");
        assert!(meta.contains("duration"), "有声明时长就要写到元数据里");
        assert!(meta.contains("videocodecid") && meta.contains("audiocodecid"));

        // 画面 sequence header：avcC 原样
        assert_eq!(tags[1].ty, TAG_VIDEO);
        assert_eq!(&tags[1].body[..5], &[0x17, 0x00, 0, 0, 0]);
        let avcc = convert::extradata_to_avcc(&CodecId::H264, &video_track().extradata, 4).unwrap();
        assert_eq!(&tags[1].body[5..], &avcc[..], "sequence header 必须是 avcC");

        // 声音 sequence header：AudioSpecificConfig 原样
        assert_eq!(tags[2].ty, TAG_AUDIO);
        assert_eq!(&tags[2].body, &[0xAF, 0x00, 0x12, 0x10]);

        // 第一条画面（关键帧）：DTS 0、CTS = 33 ms、载荷是长度前缀
        assert_eq!(tags[3].ty, TAG_VIDEO);
        assert_eq!(tags[3].ts, 0);
        assert_eq!(&tags[3].body[..2], &[0x17, 0x01], "关键帧 + AVCPacketType=NALU");
        assert_eq!(&tags[3].body[2..5], &[0x00, 0x00, 33], "1000 刻度 @30k = 33 ms");
        assert_eq!(&tags[3].body[5..], &[0, 0, 0, 5, 0x65, 0x11, 0x22, 0x33, 0x44]);

        // 第一条声音：裸帧 + 0xAF
        assert_eq!(tags[4].ty, TAG_AUDIO);
        assert_eq!(&tags[4].body, &[0xAF, 0x01, 0x21, 0x10, 0x05, 0x55]);

        // 第二条画面（非关键帧）：0x27，时间戳 33 ms
        assert_eq!(tags[5].body[0], 0x27);
        assert_eq!(tags[5].ts, 33);

        // 第二条声音：dts 1024 @44.1k = 23 ms
        assert_eq!(tags[6].ts, 23);
    }

    #[test]
    fn parameter_sets_in_samples_are_stripped_they_live_in_the_sequence_header() {
        let mut m = FlvMuxer::new();
        m.set_tracks(&[video_track()]).unwrap();
        // 一个"参数集 + IDR"的采样（有些站点会把 SPS/PPS 塞在关键帧前）
        let mut v = vec![0u8, 0, 0, 1, 0x67, 0x42, 0x00, 0x1E];
        v.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE, 0x3C, 0x80]);
        v.extend_from_slice(&[0, 0, 0, 1, 0x65, 0xAA, 0xBB]);
        let mut p = Packet::new(1, 0, 0, 1000, v);
        p.keyframe = true;
        m.write_packet(&p).unwrap();
        let data = m.finish().unwrap();
        let (flags, tags) = parse_tags(&data);
        assert_eq!(flags, 0x04, "只有画面时 audio 标志位必须是 0");
        let last = tags.last().expect("要有画面 Tag");
        assert_eq!(&last.body[5..], &[0, 0, 0, 3, 0x65, 0xAA, 0xBB], "SPS/PPS 不该重复出现");
    }

    #[test]
    fn mp3_audio_uses_the_mp3_sound_header() {
        let mut t = Track::new(2, MediaKind::Audio, CodecId::Mp3, Timebase::new(44_100));
        t.sample_rate = 44_100;
        t.channels = 2;
        let mut m = FlvMuxer::new();
        m.set_tracks(&[t]).unwrap();
        let mut frame = vec![0xFF, 0xFB, 0x90, 0x00];
        frame.resize(417, 0x22);
        m.write_packet(&Packet::new(2, 0, 0, 1152, frame.clone())).unwrap();
        let data = m.finish().unwrap();
        let (flags, tags) = parse_tags(&data);
        assert_eq!(flags, 0x01);
        let audio = tags.last().unwrap();
        assert_eq!(audio.ty, TAG_AUDIO);
        // MP3 / 44 kHz / 16 bit / stereo ⇒ 0x2F；MP3 没有 sequence header
        assert_eq!(audio.body[0], 0x2F);
        assert_eq!(&audio.body[1..], &frame[..], "MP3 帧原样搬");
    }

    #[test]
    fn rejects_unsupported_tracks() {
        let mut h265 = Track::new(1, MediaKind::Video, CodecId::H265, Timebase::new(30_000));
        h265.extradata = video_track().extradata;
        let mut m = FlvMuxer::new();
        let err = m.set_tracks(&[h265]).unwrap_err();
        assert!(format!("{err}").contains("H.264"), "{err}");

        let mut aac = audio_track();
        aac.extradata.clear();
        let mut m = FlvMuxer::new();
        let err = m.set_tracks(&[aac]).unwrap_err();
        assert!(format!("{err}").contains("AudioSpecificConfig"), "{err}");

        let mut m = FlvMuxer::new();
        let err = m
            .set_tracks(&[video_track(), video_track()])
            .expect_err("两条画面轨必须被拒");
        assert!(format!("{err}").contains("一条画面"), "{err}");

        let sub = Track::new(3, MediaKind::Subtitle, CodecId::Unknown(0), Timebase::new(1000));
        let mut m = FlvMuxer::new();
        let err = m.set_tracks(&[sub]).unwrap_err();
        assert!(format!("{err}").contains("装不下"), "{err}");
    }
}
