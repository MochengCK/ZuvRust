//! Matroska（`.mkv`）封装：`EBML + Segment{Info, Tracks, Cluster*}`，**流式直写**。
//!
//! ## 为什么值得有
//!
//! 合并/转封装的产物以前只有 fMP4 与 MPEG-TS 两种。MKV 是"什么都能装"的那一类：
//! 几乎所有编码组合、以后要加的字幕/多音轨都放得下，用户也更常拿它当合并产物。
//!
//! ## 几个必须做对、错了就"能生成但播不了"的点
//!
//! 1. **`Segment` 的长度写"未知"**（8 字节全 1 的 vint = `0x01FFFFFFFFFFFFFF`）：
//!    流式直写时不能回头改长度，而未做索引的 MKV 本来就不需要知道 Segment 有多长。
//!    但 **`Cluster` 是我们先在内存里组好、长度确定**的 ⇒ 照常写确定长度。
//! 2. **时间戳单位是 `TimestampScale`**（这里写 1 ms）：`Cluster` 与 `SimpleBlock`
//!    里的时间戳都是刻度，块内的相对时间戳是**有符号 16 位毫秒** ⇒ 一个 Cluster
//!    最多覆盖 ±32.7 秒。我们按 2 秒切（[`DEFAULT_CLUSTER_US`]），并在临近 30 秒时
//!    提前切（[`MAX_CLUSTER_US`]）。
//! 3. **块里的时间戳是"呈现时间"（pts）** —— 与 fMP4 的 `trun`（dts + cto 偏移）不同。
//! 4. **H.264 / H.265 的采样是长度前缀布局**（`CodecPrivate` = `avcC`/`hvcC` 载荷），
//!    与 fMP4 同一条转换；AV1（OBU）与 VP9 原样搬；AAC 要剥 `ADTS`。
//! 5. **FLAC 明确拒绝**：Matroska 要求 `CodecPrivate` 是 `fLaC` 头 + 元数据块序列，
//!    而引擎内部是 fMP4 形态的 `dfLa` 载荷 —— 形态转换还没做。宁可报错，也不产出一个
//!    "有数据但解不出来"的文件（我们的读取侧同样明确拒绝 FLAC-in-MKV）。
//! 6. **关键帧位（`SimpleBlock` 的 0x80）要如实写**：视频按 `Packet::keyframe`，
//!    音频全部置位（音频帧都是关键帧）。与 fMP4 的 `sidx` 同一道理，**不许无条件
//!    写"是关键帧"** —— 那会让 seek 到 Cluster 中间直接解码出花屏。

use std::io::Write;

use me_core::{CodecId, MediaError, MediaKind, Packet, Result, Track};

use crate::convert;
use crate::sink::Sink;

/// 一个 Cluster 的目标时长（微秒）。2 秒是常见取值（播放器一跳就是一簇）。
pub const DEFAULT_CLUSTER_US: i64 = 2_000_000;

/// `TimestampScale`：1 ms（单位是纳秒）。块的相对时间戳是 i16 毫秒，别改大。
const TIMESTAMP_SCALE_NS: u64 = 1_000_000;

/// 一个 Cluster 的**安全上限**（微秒）：块内相对时间戳是 i16 毫秒，
/// 越过 ±32767 ms 就写不进去 —— 取 30 秒留足余量。
const MAX_CLUSTER_US: i64 = 30_000_000;

/// Opus 的 `SeekPreRoll`（纳秒）：规范给的固定值（80 ms）。
const OPUS_SEEK_PRE_ROLL_NS: u64 = 80_000_000;

// ── EBML ID（常量本身已带长度标记位，写盘时按"最短大端字节序"落）──────

const ID_EBML: u32 = 0x1A45_DFA3;
const ID_EBML_VERSION: u32 = 0x4286;
const ID_EBML_READ_VERSION: u32 = 0x42F7;
const ID_EBML_MAX_ID_LENGTH: u32 = 0x42F2;
const ID_EBML_MAX_SIZE_LENGTH: u32 = 0x42F3;
const ID_DOC_TYPE: u32 = 0x4282;
const ID_DOC_TYPE_VERSION: u32 = 0x4287;
const ID_DOC_TYPE_READ_VERSION: u32 = 0x4285;
const ID_SEGMENT: u32 = 0x1853_8067;
const ID_INFO: u32 = 0x1549_A966;
const ID_TIMESTAMP_SCALE: u32 = 0x2AD7_B1;
const ID_MUXING_APP: u32 = 0x4D80;
const ID_WRITING_APP: u32 = 0x5741;
const ID_TRACKS: u32 = 0x1654_AE6B;
const ID_TRACK_ENTRY: u32 = 0xAE;
const ID_TRACK_NUMBER: u32 = 0xD7;
const ID_TRACK_UID: u32 = 0x73C5;
const ID_TRACK_TYPE: u32 = 0x83;
const ID_FLAG_LACING: u32 = 0x9C;
const ID_CODEC_ID: u32 = 0x86;
const ID_CODEC_PRIVATE: u32 = 0x63A2;
const ID_CODEC_DELAY: u32 = 0x56AA;
const ID_SEEK_PRE_ROLL: u32 = 0x56BB;
const ID_DEFAULT_DURATION: u32 = 0x23E3_83;
const ID_LANGUAGE: u32 = 0x22B5_9C;
const ID_VIDEO: u32 = 0xE0;
const ID_PIXEL_WIDTH: u32 = 0xB0;
const ID_PIXEL_HEIGHT: u32 = 0xBA;
const ID_AUDIO: u32 = 0xE1;
const ID_SAMPLING_FREQ: u32 = 0xB5;
const ID_CHANNELS: u32 = 0x9F;
const ID_DURATION: u32 = 0x4489;
const ID_CLUSTER: u32 = 0x1F43_B675;
const ID_CLUSTER_TS: u32 = 0xE7;
const ID_SIMPLE_BLOCK: u32 = 0xA3;

/// EBML ID → 最短大端字节（常量里已经带长度标记位）。
fn id_bytes(id: u32) -> Vec<u8> {
    let b = id.to_be_bytes();
    let lead = b.iter().take_while(|&&x| x == 0).count();
    b[lead..].to_vec()
}

/// EBML 长度字段（vint）：最短能装下的长度，带标记位。`None` = "未知长度"（8 字节全 1）。
fn vint(value: Option<u64>) -> Vec<u8> {
    let Some(v) = value else {
        return vec![0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
    };
    let mut n = 1usize;
    while n < 8 && v >= (1u64 << (7 * n)) - 1 {
        n += 1;
    }
    let mut out = vec![0u8; n];
    for (i, slot) in out.iter_mut().enumerate() {
        let shift = 8 * (n - 1 - i);
        *slot = ((v >> shift) & 0xFF) as u8;
    }
    out[0] |= 1 << (8 - n); // 长度标记位
    out
}

/// 最小长度的无符号整数（EBML 的 `uint`：省前导零字节，至少 1 字节）。
fn uint_bytes(v: u64) -> Vec<u8> {
    if v == 0 {
        return vec![0];
    }
    let b = v.to_be_bytes();
    let lead = b.iter().take_while(|&&x| x == 0).count();
    b[lead..].to_vec()
}

/// 一个 EBML 元素：ID + 确定长度 + 体。
fn el(id: u32, body: &[u8]) -> Vec<u8> {
    let mut out = id_bytes(id);
    out.extend_from_slice(&vint(Some(body.len() as u64)));
    out.extend_from_slice(body);
    out
}

/// 一个"未知长度"的 EBML 元素（只给 `Segment` 用，见模块抬头）。
fn el_unknown(id: u32, body: &[u8]) -> Vec<u8> {
    let mut out = id_bytes(id);
    out.extend_from_slice(&vint(None));
    out.extend_from_slice(body);
    out
}

fn el_uint(id: u32, v: u64) -> Vec<u8> {
    el(id, &uint_bytes(v))
}

fn el_str(id: u32, s: &str) -> Vec<u8> {
    el(id, s.as_bytes())
}

fn el_float(id: u32, v: f64) -> Vec<u8> {
    el(id, &v.to_be_bytes())
}

/// 轨道的编解码标识（Matroska 的 `CodecID` 字符串）。
fn codec_id_of(t: &Track) -> Result<&'static str> {
    Ok(match (t.kind, t.codec.clone()) {
        (MediaKind::Video, CodecId::H264) => "V_MPEG4/ISO/AVC",
        (MediaKind::Video, CodecId::H265) => "V_MPEGH/ISO/HEVC",
        (MediaKind::Video, CodecId::Av1) => "V_AV1",
        (MediaKind::Video, CodecId::Vp9) => "V_VP9",
        (MediaKind::Audio, CodecId::Aac) => "A_AAC",
        (MediaKind::Audio, CodecId::Opus) => "A_OPUS",
        (MediaKind::Audio, CodecId::Mp3) => "A_MPEG/L3",
        (MediaKind::Audio, CodecId::Ac3) => "A_AC3",
        (MediaKind::Audio, CodecId::Eac3) => "A_EAC3",
        (MediaKind::Audio, CodecId::Flac) => {
            return Err(MediaError::Unsupported(
                "Matroska 里的 FLAC 需要 `fLaC` 头形态的 CodecPrivate（引擎内部是 fMP4 的 \
                 `dfLa` 载荷），这个转换还没做 —— 换 fMP4/TS 产物，或先转成别的容器"
                    .into(),
            ))
        }
        (kind, codec) => {
            return Err(MediaError::Unsupported(format!(
                "Matroska 封装暂不支持 {kind:?} 的 {}",
                codec.name()
            )))
        }
    })
}

/// `OpusHead` 的 pre-skip（采样数）→ `CodecDelay`（纳秒）。
///
/// Matroska 要求 Opus 轨给出 `CodecDelay`，否则开头会多出 pre-skip 个样本的错位。
/// `OpusHead` 的第 10~11 字节是小端 u16 的 pre-skip。
fn opus_codec_delay_ns(extradata: &[u8]) -> Option<u64> {
    if extradata.len() < 12 || &extradata[..8] != b"OpusHead" {
        return None;
    }
    let pre_skip = u16::from_le_bytes([extradata[10], extradata[11]]) as u64;
    Some(pre_skip * 1_000_000_000 / 48_000)
}

/// 一条轨攒着的一条采样。Matroska 不写逐条时长：块时长由"下一条的时间戳"推，
/// 末条用 `DefaultDuration` 兜底（与 `me-container` 的读取侧一致）。
#[derive(Debug, Clone)]
struct Sample {
    /// 呈现时间（微秒，已经过时间轴归零）。
    pts_us: i64,
    keyframe: bool,
    data: Vec<u8>,
}

/// 一条轨道的封装状态。
#[derive(Debug)]
struct EsTrack {
    track: Track,
    codec_id: &'static str,
    nal_len: u8,
    samples: Vec<Sample>,
    last_pts_us: Option<i64>,
}

/// Matroska 封装器（API 与 [`crate::Mp4Muxer`] 对齐，CLI 里两个容器可以互换）。
pub struct MkvMuxer {
    tracks: Vec<EsTrack>,
    sink: Sink,
    cluster_us: i64,
    header_written: bool,
    packets_written: u64,
    clusters_written: u32,
    /// 时间轴基准（微秒）：`None` = 由第一条采样决定（合并时每段从头归零）。
    base_us: Option<i64>,
    /// 当前 Cluster 的起点（微秒）。`i64::MAX` = 空簇。
    cluster_start_us: i64,
    /// `Info.Duration` 的**提示值**（微秒）。
    ///
    /// 流式直写时总时长本来是不知道的（与 fMP4 的 `mvhd.duration = 0` 同一个道理），
    /// 但**合并/转封装这一步是知道的** —— 调用方把"输入声明时长之和"喂进来，
    /// 播放器/探针就能立刻报出时长，而不是先扫全文件。给 `None` 就不写这个元素
    /// （诚实：不知道就别编）。
    duration_hint_us: Option<i64>,
}

impl Default for MkvMuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl MkvMuxer {
    /// 内存落点（测试与小产物）。
    pub fn new() -> Self {
        Self::with_sink(Sink::mem())
    }

    /// 直写目标：产物**不驻留内存**（大文件合并走这条）。
    pub fn with_writer(w: Box<dyn Write + Send>) -> Self {
        Self::with_sink(Sink::writer(w))
    }

    fn with_sink(sink: Sink) -> Self {
        Self {
            tracks: Vec::new(),
            sink,
            cluster_us: DEFAULT_CLUSTER_US,
            header_written: false,
            packets_written: 0,
            clusters_written: 0,
            base_us: None,
            cluster_start_us: i64::MAX,
            duration_hint_us: None,
        }
    }

    /// 给 `Info.Duration` 一个提示值（微秒，见字段说明）。
    pub fn set_duration_hint_us(&mut self, us: Option<i64>) {
        self.duration_hint_us = us.filter(|v| *v > 0);
    }

    /// 设置一个 Cluster 的目标时长（微秒）。
    pub fn set_cluster_duration_us(&mut self, us: i64) {
        self.cluster_us = us.clamp(1, MAX_CLUSTER_US);
    }

    /// 固定时间轴基准（微秒）：`0` = 从头归零，`landing_us` = 保留绝对时间。
    pub fn set_timeline_base_us(&mut self, us: i64) {
        self.base_us = Some(us);
    }

    /// 已经写出的 Cluster 数。
    pub fn fragments_written(&self) -> u32 {
        self.clusters_written
    }

    /// 把已经写出的字节刷到目标。
    pub fn flush_output(&mut self) -> Result<()> {
        self.sink.flush()
    }

    /// 声明轨道，写出 `EBML + Segment{Info, Tracks}`（Segment 长度未知，见模块抬头）。
    pub fn set_tracks(&mut self, tracks: &[Track]) -> Result<()> {
        if tracks.is_empty() {
            return Err(MediaError::Container("Matroska 至少需要一条轨道".into()));
        }
        self.tracks.clear();
        for t in tracks {
            let codec_id = codec_id_of(t)?;
            let nal_len = if matches!(t.kind, MediaKind::Video) && !t.extradata.is_empty() {
                convert::nal_length_size(&t.codec, &t.extradata)
            } else {
                4
            };
            self.tracks.push(EsTrack {
                track: t.clone(),
                codec_id,
                nal_len,
                samples: Vec::new(),
                last_pts_us: None,
            });
        }
        self.write_header()
    }

    /// 写一条采样。
    pub fn write_packet(&mut self, p: &Packet) -> Result<()> {
        if self.tracks.is_empty() {
            return Err(MediaError::Container("写采样之前必须先 set_tracks".into()));
        }
        let idx = self
            .tracks
            .iter()
            .position(|t| t.track.id == p.track_id)
            .ok_or_else(|| MediaError::Container(format!("轨道 {} 不在 Tracks 里", p.track_id)))?;
        let tb = self.tracks[idx].track.timebase;
        // 时间轴基准：与 fMP4 同一条规矩（按微秒存，再换算回本轨刻度）。
        let base = match self.base_us {
            Some(us) => tb.us_to_ticks(us),
            None => {
                let us = tb.ticks_to_us(p.dts);
                self.base_us = Some(us);
                tb.us_to_ticks(us)
            }
        };
        let pts_us = tb.ticks_to_us(p.pts - base);
        // 时间**严格后退**会让块时间戳乱掉：当场报错，别产出坏文件。
        if let Some(prev) = self.tracks[idx].last_pts_us {
            if pts_us < prev {
                return Err(MediaError::Container(format!(
                    "轨道 {} 的时间戳倒流了（{} < {}）：调用方必须先交织好再写",
                    p.track_id, pts_us, prev
                )));
            }
        }
        // 本簇装不下这条（超目标时长，或临近 i16 毫秒上限）⇒ 先把已有的一簇落盘
        if self.cluster_start_us != i64::MAX {
            let span = pts_us - self.cluster_start_us;
            if span >= self.cluster_us || span >= MAX_CLUSTER_US {
                self.flush_cluster()?;
            }
        }
        // 空簇：起点由这一条决定（**当前簇的起点**，不是全片起点）
        if self.cluster_start_us == i64::MAX {
            self.cluster_start_us = pts_us;
        }
        let data = self.convert_sample(idx, p)?;
        let t = &mut self.tracks[idx];
        t.samples.push(Sample { pts_us, keyframe: p.keyframe, data });
        t.last_pts_us = Some(pts_us);
        self.packets_written += 1;
        Ok(())
    }

    /// 收尾：把最后一簇写完，并（内存模式下）交出整份文件。
    pub fn finish(&mut self) -> Result<Vec<u8>> {
        self.flush_cluster()?;
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

    /// 采样数据形态转换（与 fMP4 同一条规矩，见模块抬头第 4 条）。
    fn convert_sample(&self, idx: usize, p: &Packet) -> Result<Vec<u8>> {
        let kind = self.tracks[idx].track.kind;
        let codec = self.tracks[idx].track.codec.clone();
        match kind {
            MediaKind::Video if matches!(codec, CodecId::Av1 | CodecId::Vp9) => Ok(p.data.clone()),
            MediaKind::Video => {
                let nal_len = self.tracks[idx].nal_len;
                if me_container::codec::split_annexb(&p.data).is_empty() {
                    return Err(MediaError::Bitstream(format!(
                        "轨道 {} 的视频采样不是 Annex-B（找不到起始码）",
                        p.track_id
                    )));
                }
                Ok(convert::annexb_to_avcc(&p.data, nal_len))
            }
            MediaKind::Audio if codec == CodecId::Aac => Ok(crate::mp4::strip_adts(&p.data)),
            _ => Ok(p.data.clone()),
        }
    }

    /// 写出 `EBML + Segment{Info, Tracks}`。
    fn write_header(&mut self) -> Result<()> {
        let mut ebml = el_uint(ID_EBML_VERSION, 1);
        ebml.extend_from_slice(&el_uint(ID_EBML_READ_VERSION, 1));
        ebml.extend_from_slice(&el_uint(ID_EBML_MAX_ID_LENGTH, 4));
        ebml.extend_from_slice(&el_uint(ID_EBML_MAX_SIZE_LENGTH, 8));
        ebml.extend_from_slice(&el_str(ID_DOC_TYPE, "matroska"));
        ebml.extend_from_slice(&el_uint(ID_DOC_TYPE_VERSION, 4));
        ebml.extend_from_slice(&el_uint(ID_DOC_TYPE_READ_VERSION, 2));
        let mut out = el(ID_EBML, &ebml);

        let mut info = el_uint(ID_TIMESTAMP_SCALE, TIMESTAMP_SCALE_NS);
        // `Duration` 是浮点（单位 = TimestampScale 刻度，这里就是毫秒）
        if let Some(us) = self.duration_hint_us {
            info.extend_from_slice(&el_float(ID_DURATION, us as f64 / 1000.0));
        }
        info.extend_from_slice(&el_str(ID_MUXING_APP, "zuvrust"));
        info.extend_from_slice(&el_str(ID_WRITING_APP, "zuvrust"));

        let mut tracks = Vec::new();
        for t in &self.tracks {
            tracks.extend_from_slice(&el(ID_TRACK_ENTRY, &track_entry(t)));
        }

        let mut segment = el(ID_INFO, &info);
        segment.extend_from_slice(&el(ID_TRACKS, &tracks));
        out.extend_from_slice(&el_unknown(ID_SEGMENT, &segment));

        self.sink.write_all(&out)?;
        self.header_written = true;
        Ok(())
    }

    /// 把攒着的采样写成一个 `Cluster`（长度确定：先在内存里组好）。
    fn flush_cluster(&mut self) -> Result<()> {
        if !self.header_written {
            return Ok(());
        }
        if self.tracks.iter().all(|t| t.samples.is_empty()) {
            self.cluster_start_us = i64::MAX;
            return Ok(());
        }
        // 簇基准（毫秒，向下取整；负值夹到 0 —— 时间轴已归零，负的只可能是重排余量）
        let cluster_ts_ms = self.cluster_start_us.div_euclid(1000).max(0);
        let mut body = el_uint(ID_CLUSTER_TS, cluster_ts_ms as u64);

        // 块按 (pts, 轨道) 归并 —— 播放器喜欢簇内大致按时间排
        let mut order: Vec<(i64, usize, usize)> = Vec::new();
        for (ti, t) in self.tracks.iter().enumerate() {
            for si in 0..t.samples.len() {
                order.push((t.samples[si].pts_us, ti, si));
            }
        }
        order.sort_by_key(|&(pts, ti, si)| (pts, ti, si));

        for (pts_us, ti, si) in order {
            let t = &self.tracks[ti];
            let s = &t.samples[si];
            let rel_ms = pts_us.div_euclid(1000) - cluster_ts_ms;
            debug_assert!(
                (-32_768..=32_767).contains(&rel_ms),
                "块内相对时间戳越出 i16（{rel_ms} ms）：Cluster 该提前切"
            );
            let mut blk = vint(Some(t.track.id as u64));
            blk.extend_from_slice(&(rel_ms as i16).to_be_bytes());
            // 关键帧位：视频如实写、音频恒为关键帧（见模块抬头第 6 条）
            let key = s.keyframe || t.track.kind == MediaKind::Audio;
            blk.push(if key { 0x80 } else { 0x00 });
            blk.extend_from_slice(&s.data);
            body.extend_from_slice(&el(ID_SIMPLE_BLOCK, &blk));
        }
        self.sink.write_all(&el(ID_CLUSTER, &body))?;

        for t in &mut self.tracks {
            t.samples.clear();
        }
        self.cluster_start_us = i64::MAX;
        self.clusters_written += 1;
        Ok(())
    }
}

/// 一个 `TrackEntry` 的体。
fn track_entry(t: &EsTrack) -> Vec<u8> {
    let track = &t.track;
    let mut e = el_uint(ID_TRACK_NUMBER, track.id as u64);
    e.extend_from_slice(&el_uint(ID_TRACK_UID, track.id as u64));
    let track_type = if track.kind == MediaKind::Video { 1 } else { 2 };
    e.extend_from_slice(&el_uint(ID_TRACK_TYPE, track_type));
    e.extend_from_slice(&el_uint(ID_FLAG_LACING, 0));
    e.extend_from_slice(&el_str(ID_CODEC_ID, t.codec_id));
    if !track.extradata.is_empty() {
        e.extend_from_slice(&el(ID_CODEC_PRIVATE, &track.extradata));
    }
    if track.codec == CodecId::Opus {
        if let Some(delay) = opus_codec_delay_ns(&track.extradata) {
            e.extend_from_slice(&el_uint(ID_CODEC_DELAY, delay));
        }
        e.extend_from_slice(&el_uint(ID_SEEK_PRE_ROLL, OPUS_SEEK_PRE_ROLL_NS));
    }
    if let Some((num, den)) = track.frame_rate {
        if num > 0 && den > 0 {
            // 一帧的标称时长（纳秒）
            let ns = 1_000_000_000u64 * den as u64 / num as u64;
            e.extend_from_slice(&el_uint(ID_DEFAULT_DURATION, ns));
        }
    }
    if let Some(lang) = &track.language {
        if !lang.is_empty() {
            e.extend_from_slice(&el_str(ID_LANGUAGE, lang));
        }
    }
    if track.kind == MediaKind::Video {
        let mut v = el_uint(ID_PIXEL_WIDTH, track.width as u64);
        v.extend_from_slice(&el_uint(ID_PIXEL_HEIGHT, track.height as u64));
        e.extend_from_slice(&el(ID_VIDEO, &v));
    } else {
        let rate = if track.sample_rate > 0 { track.sample_rate as f64 } else { 48_000.0 };
        let mut a = el_float(ID_SAMPLING_FREQ, rate);
        a.extend_from_slice(&el_uint(ID_CHANNELS, track.channels.max(1) as u64));
        e.extend_from_slice(&el(ID_AUDIO, &a));
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use me_container::{Demuxer, MkvDemuxer};
    use me_core::{Cursor, ReadAt, Timebase};
    use std::sync::Arc;

    fn video_track() -> Track {
        let mut t = Track::new(1, MediaKind::Video, CodecId::H264, Timebase::new(1000));
        t.width = 640;
        t.height = 360;
        t.frame_rate = Some((25, 1));
        // avcC 样的 extradata：`[4] & 3 == 3` ⇒ 采样是 4 字节长度前缀
        t.extradata = vec![
            1, 0x42, 0x00, 0x1E, 0xFF, 0xE1, 0x00, 0x02, 0x67, 0x42, 0x01, 0x00, 0x02, 0x68,
            0xCE,
        ];
        t
    }

    fn audio_track() -> Track {
        let mut t = Track::new(2, MediaKind::Audio, CodecId::Aac, Timebase::new(1000));
        t.sample_rate = 48_000;
        t.channels = 2;
        t.language = Some("jpn".into());
        t.extradata = vec![0x11, 0x90];
        t
    }

    fn flac_track() -> Track {
        let mut t = Track::new(3, MediaKind::Audio, CodecId::Flac, Timebase::new(48_000));
        t.sample_rate = 48_000;
        t.channels = 2;
        t
    }

    /// 一条视频采样（Annex-B，起始码 + NAL 类型 + 载荷）。
    fn video_packet(dts: i64, key: bool) -> Packet {
        let mut v = vec![0u8, 0, 0, 1];
        v.push(if key { 0x65 } else { 0x41 });
        v.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
        let mut p = Packet::new(1, dts, dts, 40, v);
        p.keyframe = key;
        p
    }

    /// 一条 **ADTS 封装**的 AAC 采样（muxer 必须把 7 字节头剥掉）。
    fn audio_payload() -> Vec<u8> {
        vec![0x21, 0x10, 0x05, 0x55]
    }

    fn audio_packet(dts: i64) -> Packet {
        let mut v = vec![0xFF, 0xF1, 0x4C, 0x80, 0x23, 0x1F, 0xFC];
        v.extend_from_slice(&audio_payload());
        Packet::new(2, dts, dts, 21, v)
    }

    /// 写一段 4 秒的素材，返回 `(产物, 音频包数)`。
    fn muxed(cluster_us: i64) -> (Vec<u8>, usize) {
        let mut m = MkvMuxer::new();
        m.set_cluster_duration_us(cluster_us);
        m.set_tracks(&[video_track(), audio_track()]).unwrap();
        let mut a = 0i64;
        let mut audio_packets = 0usize;
        for i in 0..100 {
            let dts = i as i64 * 40; // 25 fps，1 刻度 = 1 ms
            m.write_packet(&video_packet(dts, i % 25 == 0)).unwrap();
            // 音频 21 ms 一包：时间上与视频同步推进
            while a <= dts {
                m.write_packet(&audio_packet(a)).unwrap();
                a += 21;
                audio_packets += 1;
            }
        }
        (m.finish().unwrap(), audio_packets)
    }

    fn open(bytes: Vec<u8>) -> MkvDemuxer {
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(bytes));
        MkvDemuxer::new(src, "mem").expect("应当能打开自己封装的 MKV")
    }

    #[test]
    fn roundtrips_through_our_own_demuxer() {
        let (data, audio_written) = muxed(DEFAULT_CLUSTER_US);
        assert_eq!(&data[..4], &[0x1A, 0x45, 0xDF, 0xA3], "EBML 魔数");
        assert!(
            data.windows(8).any(|w| w == b"matroska"),
            "DocType 要是 matroska"
        );

        let mut d = open(data);
        let info = d.info();
        assert_eq!(info.tracks.len(), 2);
        let v = info.video().expect("要有视频轨");
        assert_eq!(v.codec, CodecId::H264);
        assert_eq!((v.width, v.height), (640, 360));
        assert_eq!(
            v.timebase.ticks_per_second, 1000,
            "TimestampScale = 1ms ⇒ 时基就是 1000"
        );
        let a = info.audio().expect("要有音频轨");
        assert_eq!(a.codec, CodecId::Aac);
        assert_eq!(a.sample_rate, 48_000);
        assert_eq!(a.channels, 2);
        assert_eq!(a.language.as_deref(), Some("jpn"));

        let mut video = Vec::new();
        let mut audio = Vec::new();
        while let Some(p) = d.next_packet().expect("读采样") {
            if p.track_id == 1 {
                video.push(p);
            } else {
                audio.push(p);
            }
        }
        assert_eq!(video.len(), 100, "100 帧视频一条不少");
        assert_eq!(audio.len(), audio_written, "音频包一条不少");
        for (i, p) in video.iter().enumerate() {
            assert_eq!(p.pts, i as i64 * 40, "第 {i} 帧的呈现时间（ms）");
            assert_eq!(p.keyframe, i % 25 == 0, "关键帧标志必须还原");
            assert_eq!(&p.data[..4], &[0, 0, 0, 1], "视频要还原成 Annex-B");
            assert_eq!(p.data[4], if i % 25 == 0 { 0x65 } else { 0x41 });
            assert_eq!(p.data.len(), 9, "起始码(4) + NAL 头 + 4 字节载荷");
        }
        for (i, p) in audio.iter().enumerate() {
            assert_eq!(p.pts, i as i64 * 21, "第 {i} 包音频的呈现时间");
            assert_eq!(p.keyframe, true, "音频帧都是关键帧");
            assert_eq!(p.data, audio_payload(), "ADTS 头必须被剥掉");
        }
    }

    #[test]
    fn cluster_duration_controls_the_cluster_count() {
        // 4 秒素材：1 秒一簇 ⇒ 4 簇；2 秒一簇 ⇒ 2 簇
        let mut m = MkvMuxer::new();
        m.set_cluster_duration_us(1_000_000);
        m.set_tracks(&[video_track()]).unwrap();
        for i in 0..100 {
            m.write_packet(&video_packet(i as i64 * 40, i % 25 == 0)).unwrap();
        }
        let data = m.finish().unwrap();
        assert_eq!(m.fragments_written(), 4, "4 秒 / 1 秒一簇");
        // 两秒一簇时簇数减半（用同一个素材重跑一遍）
        let mut m2 = MkvMuxer::new();
        m2.set_cluster_duration_us(2_000_000);
        m2.set_tracks(&[video_track()]).unwrap();
        for i in 0..100 {
            m2.write_packet(&video_packet(i as i64 * 40, i % 25 == 0)).unwrap();
        }
        m2.finish().unwrap();
        assert_eq!(m2.fragments_written(), 2, "4 秒 / 2 秒一簇");
        // 两种切法都要能被我们的解封装器读完
        for bytes in [data, {
            let mut m = MkvMuxer::new();
            m.set_tracks(&[video_track()]).unwrap();
            for i in 0..10 {
                m.write_packet(&video_packet(i as i64 * 40, i == 0)).unwrap();
            }
            m.finish().unwrap()
        }] {
            let mut d = open(bytes);
            let mut n = 0;
            while d.next_packet().expect("读采样").is_some() {
                n += 1;
            }
            assert!(n >= 10, "读回来的采样数不对（{n}）");
        }
    }

    #[test]
    fn audio_only_output_works() {
        let mut m = MkvMuxer::new();
        m.set_tracks(&[audio_track()]).unwrap();
        for i in 0..10 {
            m.write_packet(&audio_packet(i * 21)).unwrap();
        }
        let mut d = open(m.finish().unwrap());
        assert_eq!(d.info().tracks.len(), 1);
        let mut n = 0;
        while d.next_packet().expect("读采样").is_some() {
            n += 1;
        }
        assert_eq!(n, 10);
    }

    #[test]
    fn flac_is_rejected_with_a_clear_message() {
        let err = MkvMuxer::new()
            .set_tracks(&[flac_track()])
            .expect_err("FLAC 在 MKV 里还没做，必须明确拒绝");
        let msg = format!("{err}");
        assert!(msg.contains("FLAC"), "要说清是哪种编码：{msg}");
        assert!(msg.contains("CodecPrivate"), "要说清缺的是什么：{msg}");
    }

    #[test]
    fn timestamps_going_backwards_are_rejected() {
        let mut m = MkvMuxer::new();
        m.set_tracks(&[video_track()]).unwrap();
        m.write_packet(&video_packet(40, true)).unwrap();
        let err = m
            .write_packet(&video_packet(0, true))
            .expect_err("时间戳倒流必须当场报错");
        assert!(format!("{err}").contains("倒流"));
    }
}
