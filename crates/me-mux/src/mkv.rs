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
//! 5. **FLAC 的 `CodecPrivate` 是 `fLaC` 头**，而引擎内部存的是 `dfLa` 载荷 ⇒
//!    写之前先转换（`me_container::flac_config`）；帧数据原样搬。
//! 6. **`.webm` 用同一个封装器**，只换 `DocType` 与编码白名单（VP8/VP9/AV1 +
//!    Opus），见 [`DocType`] 与 [`ensure_webm_codec`]。
//! 7. **关键帧位（`SimpleBlock` 的 0x80）要如实写**：视频按 `Packet::keyframe`，
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
        (MediaKind::Video, CodecId::Vp8) => "V_VP8",
        (MediaKind::Audio, CodecId::Aac) => "A_AAC",
        (MediaKind::Audio, CodecId::Opus) => "A_OPUS",
        (MediaKind::Audio, CodecId::Mp3) => "A_MPEG/L3",
        (MediaKind::Audio, CodecId::Ac3) => "A_AC3",
        (MediaKind::Audio, CodecId::Eac3) => "A_EAC3",
        (MediaKind::Audio, CodecId::Flac) => "A_FLAC",
        (kind, codec) => {
            return Err(MediaError::Unsupported(format!(
                "Matroska 封装暂不支持 {kind:?} 的 {}",
                codec.name()
            )))
        }
    })
}

/// WebM 是 Matroska 的**受限子集**：只认 VP8 / VP9 / AV1 画面与 Opus / Vorbis 声音。
///
/// 不在这里拦一道的话，`.webm` 会静默产出"H.264 + AAC"的文件 —— 扩展名与
/// DocType 都说 WebM，浏览器却不认里面的编码，症状是"能打开但黑屏/没声音"。
fn ensure_webm_codec(t: &Track) -> Result<()> {
    let ok = match t.kind {
        MediaKind::Video => matches!(t.codec, CodecId::Vp8 | CodecId::Vp9 | CodecId::Av1),
        MediaKind::Audio => matches!(t.codec, CodecId::Opus),
        _ => false,
    };
    if ok {
        return Ok(());
    }
    Err(MediaError::Unsupported(format!(
        "WebM 装不下 {} / {:?}：WebM 只允许 VP8 / VP9 / AV1 画面与 Opus 声音。\
         换成 `.mkv` 或 `.mp4`（它们什么都能装）",
        t.codec.name(),
        t.kind
    )))
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
    /// 要写进 `CodecPrivate` 的字节。
    ///
    /// 大多数编码就是 `track.extradata` 原样；**FLAC 例外**：引擎内部是 `dfLa`
    /// 载荷，而 Matroska 要 `fLaC` 头（见 `me_container::flac_config`）。
    codec_private: Vec<u8>,
    nal_len: u8,
    samples: Vec<Sample>,
    /// 上一包的**解码时间**（微秒，相对时间轴基准）。
    ///
    /// 判"倒流"只能用 dts：B 帧的呈现时间天然是来回跳的（0,3,1,2…），拿 pts
    /// 判会把几乎所有带 B 帧的 H.264 素材当成坏文件（表现就是"换成 mkv 一律
    /// 失败"，而换成 mp4 / ts 都正常）。
    last_dts_us: Option<i64>,
}

/// Matroska 的「文档类型」：`.mkv` 与 `.webm` 用的是**同一个封装器**，
/// 差别只有 `DocType` 字符串与"允许哪些编码"（WebM 是 Matroska 的受限子集）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocType {
    /// `.mkv` / `.mka`：什么都能装。
    Matroska,
    /// `.webm`：只允许 VP8 / VP9 / AV1 + Opus / Vorbis（本引擎没有 Vorbis）。
    WebM,
}

impl DocType {
    fn name(self) -> &'static str {
        match self {
            DocType::Matroska => "matroska",
            DocType::WebM => "webm",
        }
    }

    /// (DocTypeVersion, DocTypeReadVersion)。
    ///
    /// WebM 规范钉的是 2/2；Matroska 这边我们用 4/2（4 是从 `DefaultDuration`
    /// 那一代开始的版本号，写 4 更符合实际用到的元素）。
    fn versions(self) -> (u64, u64) {
        match self {
            DocType::Matroska => (4, 2),
            DocType::WebM => (2, 2),
        }
    }
}

/// Matroska 封装器（API 与 [`crate::Mp4Muxer`] 对齐，CLI 里两个容器可以互换）。
pub struct MkvMuxer {
    tracks: Vec<EsTrack>,
    sink: Sink,
    doc_type: DocType,
    cluster_us: i64,
    header_written: bool,
    packets_written: u64,
    clusters_written: u32,
    /// 时间轴基准（微秒）：`None` = 由第一条采样决定（合并时每段从头归零）。
    base_us: Option<i64>,
    /// 当前 Cluster 的起点（微秒）。`i64::MAX` = 空簇。
    ///
    /// ⚠️ 它是簇内**最早**的 pts，不是"第一条采样"的 pts —— 采样可能乱序到达
    /// （见 `write_packet` 的注释），若拿第一条当基准，晚到的采样会算出负偏移。
    cluster_start_us: i64,
    /// 当前 Cluster 里最晚的 pts（微秒）。与 `cluster_start_us` 一起决定"该不该切"。
    cluster_end_us: i64,
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

    /// 内存落点的 `.webm` 封装器（测试用）。
    pub fn new_webm() -> Self {
        let mut m = Self::new();
        m.set_doc_type(DocType::WebM);
        m
    }

    /// 直写目标：产物**不驻留内存**（大文件合并走这条）。
    pub fn with_writer(w: Box<dyn Write + Send>) -> Self {
        Self::with_sink(Sink::writer(w))
    }

    fn with_sink(sink: Sink) -> Self {
        Self {
            tracks: Vec::new(),
            sink,
            doc_type: DocType::Matroska,
            cluster_us: DEFAULT_CLUSTER_US,
            header_written: false,
            packets_written: 0,
            clusters_written: 0,
            base_us: None,
            cluster_start_us: i64::MAX,
            cluster_end_us: i64::MIN,
            duration_hint_us: None,
        }
    }

    /// 切换文档类型（`.mkv` ↔ `.webm`）。**必须在 `set_tracks` 之前调用**：
    /// `DocType` 是 `EBML` 头的一部分，写完就定型了。
    pub fn set_doc_type(&mut self, doc_type: DocType) {
        self.doc_type = doc_type;
    }

    /// `.webm` 封装器（受限的 Matroska，见 [`DocType::WebM`]）。
    pub fn with_writer_webm(w: Box<dyn Write + Send>) -> Self {
        let mut m = Self::with_writer(w);
        m.set_doc_type(DocType::WebM);
        m
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
            if self.doc_type == DocType::WebM {
                ensure_webm_codec(t)?;
            }
            // FLAC：`dfLa` 载荷 → `fLaC` 头（Matroska/WebM 的 CodecPrivate 要后者）
            let codec_private = if t.codec == CodecId::Flac {
                me_container::flac_config::dfla_to_flac_header(&t.extradata)?
            } else {
                t.extradata.clone()
            };
            let nal_len = if matches!(t.kind, MediaKind::Video) && !t.extradata.is_empty() {
                convert::nal_length_size(&t.codec, &t.extradata)
            } else {
                4
            };
            self.tracks.push(EsTrack {
                track: t.clone(),
                codec_id,
                codec_private,
                nal_len,
                samples: Vec::new(),
                last_dts_us: None,
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
        let dts_us = tb.ticks_to_us(p.dts - base);
        // 时间**严格后退**会让块时间戳乱掉：当场报错，别产出坏文件。
        //
        // ⚠️ 判据是 **dts**（解码顺序），不是 pts：B 帧的 pts 天然会来回跳
        // （0,3,1,2…），用 pts 判会把所有带 B 帧的 H.264 素材都拒掉 ——
        // 症状就是"换成 mkv 一律失败，换 mp4 / ts 都正常"（用户报的形态）。
        if let Some(prev) = self.tracks[idx].last_dts_us {
            if dts_us < prev {
                return Err(MediaError::Container(format!(
                    "轨道 {} 的解码时间倒流了（{} < {}）：调用方必须先交织好再写",
                    p.track_id, dts_us, prev
                )));
            }
        }
        // 本簇装不下这条 ⇒ 先把已有的一簇落盘。
        //
        // ⚠️ 判据必须**双向**：既看"往后超了目标时长"，也看"这条比簇里最早的还早"。
        // 后者不是理论情况 —— 真实素材里出现过解封装器一口气吐出 768 秒音频
        // （源文件最后一个分片是畸形巨片），音频于是比画面晚到 30 秒；
        // 只判"往后"的话这些晚到的采样会被塞进旧簇，块内相对时间戳变成负数。
        if self.cluster_start_us != i64::MAX {
            let span = (pts_us.max(self.cluster_end_us) - pts_us.min(self.cluster_start_us)).abs();
            if span >= self.cluster_us || span >= MAX_CLUSTER_US {
                self.flush_cluster()?;
            }
        }
        // 空簇：起点由这一条决定（**当前簇的起点**，不是全片起点）
        if self.cluster_start_us == i64::MAX {
            self.cluster_start_us = pts_us;
            self.cluster_end_us = pts_us;
        }
        self.cluster_start_us = self.cluster_start_us.min(pts_us);
        self.cluster_end_us = self.cluster_end_us.max(pts_us);
        let data = self.convert_sample(idx, p)?;
        let t = &mut self.tracks[idx];
        t.samples.push(Sample { pts_us, keyframe: p.keyframe, data });
        t.last_dts_us = Some(dts_us);
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
            MediaKind::Video
                if matches!(codec, CodecId::Av1 | CodecId::Vp9 | CodecId::Vp8) =>
            {
                Ok(p.data.clone())
            }
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
        ebml.extend_from_slice(&el_str(ID_DOC_TYPE, self.doc_type.name()));
        let (doc_version, read_version) = self.doc_type.versions();
        ebml.extend_from_slice(&el_uint(ID_DOC_TYPE_VERSION, doc_version));
        ebml.extend_from_slice(&el_uint(ID_DOC_TYPE_READ_VERSION, read_version));
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
            self.cluster_end_us = i64::MIN;
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
            // 超出 i16 就**报错**，不能靠 `as i16` 截断：那会把时间戳静默变成
            // 一个看似合法的大负数，产物能生成但播放器一跳到这段就错位。
            // 切簇判据已经把跨度压到 cluster_us（2 秒）以内，走到这里说明调用方
            // 送进来的时间轴本身有问题（比如两条轨的时基差了一个数量级）。
            if !(-32_768..=32_767).contains(&rel_ms) {
                return Err(MediaError::Container(format!(
                    "簇内相对时间戳 {rel_ms} ms 超出 i16（簇基准 {cluster_ts_ms} ms，样本 {pts_us} us）：\
                     时间轴异常，拒绝产出坏文件"
                )));
            }
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
        self.cluster_end_us = i64::MIN;
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
    if !t.codec_private.is_empty() {
        e.extend_from_slice(&el(ID_CODEC_PRIVATE, &t.codec_private));
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
        t.extradata = flac_dfla();
        t
    }

    /// `dfLa` 载荷：version/flags + STREAMINFO 块（48 kHz / 2 声道 / 16 bit）。
    fn flac_dfla() -> Vec<u8> {
        let mut si = vec![0u8; 34];
        let w = (48_000u32 << 12) | (1u32 << 9) | (15u32 << 4);
        si[10..14].copy_from_slice(&w.to_be_bytes());
        let mut v = vec![0, 0, 0, 0];
        v.extend_from_slice(&[0x80, 0x00, 0x00, 34]);
        v.extend_from_slice(&si);
        v
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
    fn webm_uses_its_own_doc_type_and_rejects_foreign_codecs() {
        // 同一个封装器，DocType 不同：`.webm` 必须写 "webm"，Matroska 写 "matroska"
        let mut m = MkvMuxer::new_webm();
        let mut t = video_track();
        t.codec = CodecId::Vp9; // WebM 允许的编码
        t.extradata.clear();
        m.set_tracks(&[t]).expect("VP9 应当能进 WebM");
        m.write_packet(&video_packet(0, true)).unwrap();
        let bytes = m.finish().unwrap();
        assert!(
            bytes.windows(4).any(|w| w == b"webm"),
            "DocType 必须是 webm"
        );

        // H.264 不能进 WebM：要在 set_tracks 就报错，而不是写出一个浏览器打不开的文件
        let err = MkvMuxer::new_webm()
            .set_tracks(&[video_track()])
            .expect_err("H.264 + WebM 必须明确拒绝");
        let msg = format!("{err}");
        assert!(msg.contains("WebM") && msg.contains("VP8"), "{msg}");
    }

    #[test]
    fn a_track_that_arrives_late_does_not_break_the_cluster() {
        // 真实素材踩出来的：源文件最后一个分片是畸形巨片（579 KB 的 moof 装着
        // 768 秒音频），解封装器一口气吐出三万条音频 ⇒ 音频比画面晚到几十秒。
        // 旧的切簇判据只看"往后超了没"，这些晚到的采样被塞进旧簇，块内相对
        // 时间戳变成 -33 秒（debug 构建 panic、release 构建静默截断成垃圾）。
        let mut m = MkvMuxer::new();
        m.set_tracks(&[video_track(), audio_track()]).unwrap();
        // 先灌 40 秒画面（没有任何音频）
        for i in 0..40 * 25 {
            m.write_packet(&video_packet(i as i64 * 40, i % 25 == 0)).unwrap();
        }
        // 再灌 40 秒音频（迟到的那一路）
        for i in 0..40 * 47 {
            m.write_packet(&Packet::new(2, i * 21, i * 21, 21, vec![0x11, 0x22])).unwrap();
        }
        let bytes = m.finish().expect("晚到的轨道不该让封装失败");

        let mut d = open(bytes);
        let mut v = 0;
        let mut a = 0;
        while let Some(p) = d.next_packet().expect("读采样") {
            match p.track_id {
                1 => v += 1,
                2 => a += 1,
                _ => {}
            }
        }
        assert_eq!(v, 40 * 25, "画面一条都不能丢");
        assert_eq!(a, 40 * 47, "晚到的音频也要一条不少地写进去");
    }

    #[test]
    fn flac_round_trips_through_matroska() {
        // FLAC in MKV：`CodecPrivate` 要 `fLaC` 头形态，而引擎内部是 `dfLa` 载荷。
        // 写的时候转出去、读回来再转回来，两个形态必须逐字节对齐 —— 否则
        // 产物"能生成但解不出来"。
        let mut m = MkvMuxer::new();
        m.set_tracks(&[flac_track()]).unwrap();
        for i in 0..5 {
            let frame = vec![0xFF, 0xF8, 0x69, 0x08, 0x00, i as u8, 0xAA];
            m.write_packet(&Packet::new(3, i * 4096, i * 4096, 4096, frame)).unwrap();
        }
        let bytes = m.finish().unwrap();

        // 产物里应当出现 A_FLAC 与 fLaC 头
        assert!(
            bytes.windows(6).any(|w| w == b"A_FLAC"),
            "CodecID 必须是 A_FLAC"
        );
        assert!(
            bytes.windows(4).any(|w| w == b"fLaC"),
            "CodecPrivate 必须是 fLaC 头（不是 dfLa 载荷）"
        );

        // 读回来：extradata 必须重新变成 dfLa 载荷（引擎内部形态）
        let mut d = open(bytes);
        let info = d.info();
        assert_eq!(info.tracks.len(), 1);
        assert_eq!(info.tracks[0].codec, CodecId::Flac);
        assert_eq!(info.tracks[0].sample_rate, 48_000);
        assert_eq!(info.tracks[0].channels, 2);
        assert_eq!(
            info.tracks[0].extradata,
            flac_dfla(),
            "读回来要还原成 dfLa 载荷（否则解码器拿到的形态就错了）"
        );

        let mut n = 0;
        while d.next_packet().expect("读采样").is_some() {
            n += 1;
        }
        assert_eq!(n, 5, "FLAC 帧要原样搬过去");
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

    /// B 帧：**pts 倒退、但 dts 递增**是完全合法的，不能当成坏文件拒掉。
    ///
    /// 曾经这里判的是 pts 单调，于是"把普通 H.264 视频换成 mkv"全部失败，
    /// 而换成 mp4 / ts 都正常 —— 症状看起来像"选了 MKV 没生效"。
    #[test]
    fn b_frames_with_pts_jumping_back_are_allowed() {
        let mut m = MkvMuxer::new();
        m.set_tracks(&[video_track()]).unwrap();
        let mk = |dts: i64, pts: i64| {
            let mut p = video_packet(dts, dts == 0);
            p.pts = pts;
            p
        };
        // 典型的 I-B-B-P 呈现/解码时间关系
        m.write_packet(&mk(0, 0)).expect("I 帧");
        m.write_packet(&mk(40, 120)).expect("P 帧（pts 在后）");
        m.write_packet(&mk(80, 40)).expect("B 帧（pts 回跳）");
        m.write_packet(&mk(120, 80)).expect("B 帧（pts 仍在前面之后）");
        m.finish().expect("收尾");
    }

    /// 但 **dts** 真倒退仍然是坏输入，必须当场报错。
    #[test]
    fn dts_going_backwards_is_still_rejected() {
        let mut m = MkvMuxer::new();
        m.set_tracks(&[video_track()]).unwrap();
        m.write_packet(&video_packet(80, true)).unwrap();
        let err = m
            .write_packet(&video_packet(40, false))
            .expect_err("dts 倒流必须报错");
        assert!(format!("{err}").contains("倒流"));
    }
}
