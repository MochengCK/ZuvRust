//! 媒体数据形状：轨道、采样、编解码器标识。

use serde::{Deserialize, Serialize};

use crate::time::Timebase;

/// 轨道类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MediaKind {
    Video,
    Audio,
    Subtitle,
    /// 认得出但不归类（数据流、时码轨……）。
    Other,
}

impl MediaKind {
    /// 与 C API / CLI 约定的整数编码（宿主侧不用去解析字符串）。
    pub fn as_i32(self) -> i32 {
        match self {
            MediaKind::Video => 0,
            MediaKind::Audio => 1,
            MediaKind::Subtitle => 2,
            MediaKind::Other => 3,
        }
    }

    pub fn from_i32(v: i32) -> Self {
        match v {
            0 => MediaKind::Video,
            1 => MediaKind::Audio,
            2 => MediaKind::Subtitle,
            _ => MediaKind::Other,
        }
    }
}

/// 编解码器标识。
///
/// **只描述"是什么"**：容器层与合并层只需要知道"这条轨是 H.264 还是 AAC"，
/// 用来写 `stsd`/`PMT` 里的编码标识与决定 extradata 的形状。
/// 真正的解码器在另一层（平台解码器），不在这里。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "name", rename_all = "lowercase")]
pub enum CodecId {
    H264,
    H265,
    Av1,
    Vp9,
    Vp8,
    Aac,
    Mp3,
    Opus,
    Flac,
    Ac3,
    Eac3,
    /// 容器里认得出但引擎没归类（保留原始标识，便于诊断）。
    Unknown(u32),
}

impl CodecId {
    /// 简短名字（CLI / 日志）。
    pub fn name(&self) -> String {
        match self {
            CodecId::H264 => "h264".into(),
            CodecId::H265 => "h265".into(),
            CodecId::Av1 => "av1".into(),
            CodecId::Vp9 => "vp9".into(),
            CodecId::Vp8 => "vp8".into(),
            CodecId::Aac => "aac".into(),
            CodecId::Mp3 => "mp3".into(),
            CodecId::Opus => "opus".into(),
            CodecId::Flac => "flac".into(),
            CodecId::Ac3 => "ac3".into(),
            CodecId::Eac3 => "eac3".into(),
            CodecId::Unknown(v) => format!("unknown({v})"),
        }
    }

    /// MPEG-TS 的 `stream_type`（PMT 里那个字节）。
    ///
    /// 合并成 TS 时**必须**写对：播放器就是靠它找解码器的。
    pub fn ts_stream_type(&self) -> u8 {
        match self {
            CodecId::H264 => 0x1B,
            CodecId::H265 => 0x24,
            CodecId::Aac => 0x0F, // ADTS
            CodecId::Mp3 => 0x03,
            CodecId::Ac3 => 0x81,
            CodecId::Eac3 => 0x87,
            CodecId::Vp9 | CodecId::Vp8 | CodecId::Av1 | CodecId::Opus | CodecId::Flac => 0x06,
            CodecId::Unknown(_) => 0x06,
        }
    }

    /// 从 MPEG-TS 的 `stream_type` 反推。
    pub fn from_ts_stream_type(v: u8) -> Self {
        match v {
            0x1B => CodecId::H264,
            0x24 => CodecId::H265,
            0x0F | 0x11 => CodecId::Aac,
            0x03 | 0x04 => CodecId::Mp3,
            0x81 => CodecId::Ac3,
            0x87 => CodecId::Eac3,
            other => CodecId::Unknown(other as u32),
        }
    }

    /// 是不是视频。
    pub fn is_video(&self) -> bool {
        matches!(self, CodecId::H264 | CodecId::H265 | CodecId::Av1 | CodecId::Vp9 | CodecId::Vp8)
    }

    /// 是不是音频。
    pub fn is_audio(&self) -> bool {
        matches!(
            self,
            CodecId::Aac | CodecId::Mp3 | CodecId::Opus | CodecId::Flac | CodecId::Ac3 | CodecId::Eac3
        )
    }
}

/// 一条轨道。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Track {
    /// 轨道号。**同一次解析内唯一**；合并时由 muxer 重新分配。
    pub id: u32,
    pub kind: MediaKind,
    pub codec: CodecId,
    /// 时基：`Packet::pts` 的单位。
    pub timebase: Timebase,

    /// 解码器初始化数据（H.264 的 `avcC`、H.265 的 `hvcC`、AAC 的 `AudioSpecificConfig`……）。
    ///
    /// 合并时**必须原样带过去**：丢了它，解码器起不来，产物是黑屏/静音。
    pub extradata: Vec<u8>,

    // ── 视频 ──
    pub width: u32,
    pub height: u32,
    /// 帧率（分子/分母），未知为 None。
    pub frame_rate: Option<(u32, u32)>,

    // ── 音频 ──
    pub sample_rate: u32,
    pub channels: u16,

    // ── 通用 ──
    pub language: Option<String>,
    pub title: Option<String>,
}

impl Track {
    /// 最小可用轨道（各字段取中性值，避免每个构造点都写一长串 `Default`）。
    pub fn new(id: u32, kind: MediaKind, codec: CodecId, timebase: Timebase) -> Self {
        Self {
            id,
            kind,
            codec,
            timebase,
            extradata: Vec::new(),
            width: 0,
            height: 0,
            frame_rate: None,
            sample_rate: 0,
            channels: 0,
            language: None,
            title: None,
        }
    }

    /// 刻度 → 微秒。
    pub fn pts_to_us(&self, pts: i64) -> i64 {
        self.timebase.ticks_to_us(pts)
    }
}

/// 一个采样（视频一帧 / 音频一帧 / 字幕一条）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Packet {
    /// 属于哪条轨道（对应 [`Track::id`]）。
    pub track_id: u32,
    /// 显示时间戳（刻度）。
    pub pts: i64,
    /// 解码时间戳（刻度）。没有 B 帧的流里等于 `pts`。
    pub dts: i64,
    /// 这一条覆盖的时长（刻度）。0 = 未知。
    pub duration: u32,
    /// 码流数据。
    ///
    /// **视频一律是 Annex-B 字节流**（含起始码），音频按容器原生形态
    /// （TS 里 AAC 是 ADTS，fMP4 里是裸 AAC）。这条约定贯穿全引擎：
    /// 容器之间转换时由 muxer 负责加/去 ADTS 头，解析器不做隐式转换。
    pub data: Vec<u8>,
    /// 是否关键帧（随机访问点）。
    pub keyframe: bool,
}

impl Packet {
    pub fn new(track_id: u32, pts: i64, dts: i64, duration: u32, data: Vec<u8>) -> Self {
        Self { track_id, pts, dts, duration, data, keyframe: false }
    }

    /// 结束时间（pts + duration）。
    pub fn end(&self) -> i64 {
        self.pts.saturating_add(self.duration as i64)
    }
}

/// 一份媒体的总体信息（探测结果的摘要）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MediaInfo {
    /// 容器名字（`"mpeg-ts"` / `"iso-bmff"` / `"mpeg-audio"` / `"hls"` / `"bt"`）。
    pub container: String,
    /// 总时长（微秒）；未知为 None（直播、BT 未取到元数据）。
    pub duration_us: Option<i64>,
    /// 总码率（bit/s）。
    pub bitrate: Option<u64>,
    pub tracks: Vec<Track>,
    /// 源标识（本地路径 / URL / magnet），供上层回显。
    pub source: String,
}

impl MediaInfo {
    /// 第一条视频轨。
    pub fn video(&self) -> Option<&Track> {
        self.tracks.iter().find(|t| t.kind == MediaKind::Video)
    }

    /// 第一条音频轨。
    pub fn audio(&self) -> Option<&Track> {
        self.tracks.iter().find(|t| t.kind == MediaKind::Audio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ts_stream_type_roundtrip() {
        for c in [
            CodecId::H264,
            CodecId::H265,
            CodecId::Aac,
            CodecId::Mp3,
            CodecId::Ac3,
            CodecId::Eac3,
        ] {
            let st = c.ts_stream_type();
            assert_eq!(CodecId::from_ts_stream_type(st), c, "stream_type 0x{st:02X}");
        }
    }

    #[test]
    fn codec_classification() {
        assert!(CodecId::H264.is_video());
        assert!(!CodecId::H264.is_audio());
        assert!(CodecId::Aac.is_audio());
        assert!(!CodecId::Aac.is_video());
        assert!(!CodecId::Unknown(0x06).is_video());
        assert_eq!(CodecId::Unknown(0x99).name(), "unknown(153)");
    }

    #[test]
    fn media_kind_int_mapping_is_stable() {
        // 这组数字是 CLI/C API 的对外契约，不许改
        assert_eq!(MediaKind::Video.as_i32(), 0);
        assert_eq!(MediaKind::Audio.as_i32(), 1);
        assert_eq!(MediaKind::Subtitle.as_i32(), 2);
        for k in [MediaKind::Video, MediaKind::Audio, MediaKind::Subtitle, MediaKind::Other] {
            assert_eq!(MediaKind::from_i32(k.as_i32()), k);
        }
    }

    #[test]
    fn packet_end_saturates() {
        let p = Packet::new(1, i64::MAX - 1, 0, 100, vec![]);
        assert_eq!(p.end(), i64::MAX);
    }

    #[test]
    fn info_helpers_pick_first_of_kind() {
        let mut info = MediaInfo::default();
        info.tracks.push(Track::new(1, MediaKind::Audio, CodecId::Aac, Timebase::new(44100)));
        info.tracks.push(Track::new(2, MediaKind::Video, CodecId::H264, Timebase::MPEG_TS));
        assert_eq!(info.video().unwrap().id, 2);
        assert_eq!(info.audio().unwrap().id, 1);
    }
}
