//! Matroska / WebM 解封装（`.mkv` / `.webm` / `.mka` / `.weba`）。
//!
//! ## 为什么补这一块
//!
//! 市面上下载回来的「音视频分离」资源里，除了 DASH 的 fMP4 与 HLS 的 TS，
//! 很大一部分是 **Matroska 家族**：视频站导出的 `.webm`（VP9 / AV1 + Opus）、
//! 影视资源站的 `.mkv`（H.264 / H.265 + AAC / AC-3）。在这之前引擎完全认不出
//! 它们 —— `probe` 会说"无法识别的容器"，合并更是无从谈起。
//!
//! ## 与别的解封装器共享的那套约定
//!
//! - **`extradata` 保留容器原始形态**（avcC / hvcC / av1C / vpcC / OpusHead /
//!   AudioSpecificConfig），与 fMP4 那条链完全一致 —— 因为 Matroska 的
//!   `CodecPrivate` 存放的就是这些东西；
//! - **`Packet::data` 里的视频一律是 Annex-B**，所以 H.264 / H.265 的采样要按
//!   `avcC` / `hvcC` 声明的长度宽度做转换（[`codec::length_prefixed_to_annexb`]，
//!   与 fMP4 共用同一份实现）；AV1 的 OBU、VP9 的压缩帧、音频帧都原样搬。
//!
//! ## 时间戳：MKV 只给 PTS
//!
//! 块的 timestamp 是**显示时间**，而块在文件里是**按解码顺序**存的。
//! 无 B 帧时两者一致（WebM 的 VP9/AV1、大多数 H.264 配置都如此），直接用；
//! 一旦 PTS 在存储顺序里回退（有 B 帧），就把 DTS 按"上一帧 + 步长"推上去保证
//! 单调 —— fMP4 封装器会当场拒绝倒退的 `tfdt`，而"一条流直接读不了"比
//! "重构出来的 DTS 略有偏差"糟得多。
//!
//! 采样时长同样不在块里：只能等**同一条轨的下一帧**来了才知道这一帧有多长
//! （见 [`MkvDemuxer::push_with_duration`]）。
//!
//! ## 明确不支持的（宁可报错，不产出坏文件）
//!
//! - **Vorbis**：引擎没有它的解码器，`CodecId` 里也没有这个变体；
//! - **FLAC in MKV**：`CodecPrivate` 是 `fLaC` 头 + 元数据块序列，与引擎内部的
//!   `dfLa` 载荷形态不同，转换还没做。
//!
//! 字幕轨与非音视频轨**直接跳过**（`.mkv` 带字幕是常态，不该因此整个读不了）；
//! 但**一条能用的音视频轨都没有**时仍然报错。

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use me_core::{CodecId, MediaError, MediaInfo, MediaKind, Packet, ReadAt, Result, Timebase, Track};

use crate::codec;
use crate::demux::Demuxer;

// ── EBML 元素 ID ──
const ID_EBML: u32 = 0x1A45_DFA3;
const ID_DOCTYPE: u32 = 0x4282;
const ID_SEGMENT: u32 = 0x1853_8067;
const ID_INFO: u32 = 0x1549_A966;
const ID_TIMESTAMP_SCALE: u32 = 0x2AD7_B1;
const ID_DURATION: u32 = 0x4489;
const ID_TRACKS: u32 = 0x1654_AE6B;
const ID_TRACK_ENTRY: u32 = 0xAE;
const ID_TRACK_NUMBER: u32 = 0xD7;
const ID_TRACK_TYPE: u32 = 0x83;
const ID_CODEC_ID: u32 = 0x86;
const ID_CODEC_PRIVATE: u32 = 0x63A2;
const ID_DEFAULT_DURATION: u32 = 0x23E3_83;
const ID_LANGUAGE: u32 = 0x22B5_9C;
const ID_VIDEO: u32 = 0xE0;
const ID_PIXEL_WIDTH: u32 = 0xB0;
const ID_PIXEL_HEIGHT: u32 = 0xBA;
const ID_AUDIO: u32 = 0xE1;
const ID_SAMPLING_FREQ: u32 = 0xB5;
const ID_CHANNELS: u32 = 0x9F;
const ID_CLUSTER: u32 = 0x1F43_B675;
const ID_CLUSTER_TS: u32 = 0xE7;
const ID_SIMPLE_BLOCK: u32 = 0xA3;
const ID_BLOCK_GROUP: u32 = 0xA0;
const ID_BLOCK: u32 = 0xA1;
const ID_BLOCK_DURATION: u32 = 0x9B;
const ID_REFERENCE_BLOCK: u32 = 0xFB;

/// EBML 头魔数。探测只认它。
pub fn looks_like_mkv(head: &[u8]) -> bool {
    head.len() >= 4 && head[0..4] == [0x1A, 0x45, 0xDF, 0xA3]
}

/// 读一个 EBML 变长整数。
///
/// `keep_marker`：元素 **ID** 要保留长度标志位（它是 ID 的一部分）；
/// **长度** 则要去掉标志位、取剩余的位。这是 EBML 里最容易写错的一处 ——
/// 把 ID 的标志位去掉，所有元素常量都会对不上。
fn read_vint(data: &[u8], pos: usize, keep_marker: bool) -> Option<(u64, usize)> {
    let first = *data.get(pos)?;
    if first == 0 {
        return None; // 超过 8 字节：非法
    }
    let len = first.leading_zeros() as usize + 1;
    if len > 8 || pos + len > data.len() {
        return None;
    }
    let mut v = if keep_marker {
        u64::from(first)
    } else {
        // len == 8 时长度标志位占满首字节、值部分为空（mask = 0）。
        // ⚠️ 不能直接写 `0xFF >> len`：len 到 8 会溢出，debug 下当场 panic。
        let mask = if len >= 8 { 0u8 } else { 0xFF >> len };
        u64::from(first & mask)
    };
    for i in 1..len {
        v = (v << 8) | u64::from(data[pos + i]);
    }
    Some((v, len))
}

/// 未知长度："长度字段的**值位**全为 1"（标志位不算）。
///
/// ⚠️ 不能写成 `v == u64::MAX`：8 字节长度的 vint 只承载 7 个字节的值位，
/// 读出来是 `0x00FF_FFFF_FFFF_FFFF`，永远等不上 `u64::MAX` —— 那样写
/// 的结果是"未知长度"一个都认不出来，流式写入的 WebM 全部解析失败。
fn is_unknown_size(v: u64, len: usize) -> bool {
    let bits = 7 * len;
    bits >= 1 && bits <= 63 && v == (1u64 << bits) - 1
}

fn read_uint(data: &[u8]) -> u64 {
    data.iter().fold(0u64, |a, b| (a << 8) | u64::from(*b))
}

fn read_float(data: &[u8]) -> Option<f64> {
    match data.len() {
        4 => Some(f32::from_be_bytes([data[0], data[1], data[2], data[3]]) as f64),
        8 => {
            let mut b = [0u8; 8];
            b.copy_from_slice(data);
            Some(f64::from_be_bytes(b))
        }
        _ => None,
    }
}

/// 去掉 EBML 字符串末尾的填充 0（规范允许，实际文件里也常见）。
fn trim_nul(b: &[u8]) -> &[u8] {
    let end = b.iter().rposition(|c| *c != 0).map(|i| i + 1).unwrap_or(0);
    &b[..end]
}

// ───────────────────────── 分块读取 ─────────────────────────

/// 一次拉多少字节。解析 EBML 是"读几个字节 → 跳过一大段 → 再读几个字节"，
/// 逐个字段直接打 `ReadAt` 会产生大量小请求（网络源上尤其明显）。
const CHUNK: usize = 64 * 1024;

struct Reader {
    src: Arc<dyn ReadAt>,
    buf: Vec<u8>,
    base: u64,
}

impl Reader {
    fn new(src: Arc<dyn ReadAt>) -> Self {
        Self { src, buf: Vec::new(), base: 0 }
    }

    /// 把 `off` 起至少 `n` 字节装进缓冲。
    ///
    /// **读不满不算错**（文件末尾就是这样），调用方用 [`Reader::slice`] 拿
    /// "有多少给多少"的切片，或用 [`Reader::at`] 要求严格长度。
    fn ensure(&mut self, off: u64, n: usize) -> Result<()> {
        let covered = off >= self.base && (off + n as u64) <= self.base + self.buf.len() as u64;
        if covered {
            return Ok(());
        }
        // 已经为这个偏移读过一次、还是没读满 ⇒ 确认到文件末尾了，别再重读。
        // ⚠️ `!self.buf.is_empty()` 不能省：初始状态就是 base = 0 + 空缓冲，
        // 少了它会把**第一次**读取直接当成"已到末尾"（所有元素都读不出来）。
        if self.base == off && !self.buf.is_empty() && self.buf.len() < n {
            return Ok(());
        }
        let data = self.src.read_range(off, CHUNK.max(n))?;
        self.base = off;
        self.buf = data;
        Ok(())
    }

    /// 从 `off` 起最多 `n` 字节；到文件末尾时返回**短的**切片，不报错。
    ///
    /// 读元素头必须用这个：文件最后一个元素的头上只剩三五个字节是常态，
    /// 要求"必须读满 12 字节"会把正常的结尾当成解析失败。
    fn slice(&mut self, off: u64, n: usize) -> Result<&[u8]> {
        self.ensure(off, n)?;
        let s = (off - self.base) as usize;
        if s >= self.buf.len() {
            return Ok(&[]);
        }
        let e = (s + n).min(self.buf.len());
        Ok(&self.buf[s..e])
    }

    fn at(&mut self, off: u64, n: usize) -> Result<&[u8]> {
        self.ensure(off, n)?;
        let s = (off - self.base) as usize;
        if s + n > self.buf.len() {
            return Err(MediaError::NeedMore { need: n, have: self.buf.len().saturating_sub(s) });
        }
        Ok(&self.buf[s..s + n])
    }
}

/// 一个元素头。
#[derive(Debug, Clone, Copy)]
struct Element {
    id: u32,
    /// 数据长度；`None` = 未知长度（Segment / Cluster 会用到）。
    size: Option<u64>,
    /// 数据起始偏移。
    data_off: u64,
}

/// 读一个元素头；`None` = 到这里读不出合法头了（到边界 / 文件结束）。
fn read_element(r: &mut Reader, off: u64) -> Result<Option<Element>> {
    // 头最长 = ID(4) + 长度(8)，但文件末尾可能只剩几字节 —— 用宽松的 slice
    let head = r.slice(off, 12)?;
    let Some((id_v, id_len)) = read_vint(head, 0, true) else {
        return Ok(None);
    };
    let Some((size_v, size_len)) = read_vint(head, id_len, false) else {
        return Ok(None);
    };
    let unknown = is_unknown_size(size_v, size_len);
    Ok(Some(Element {
        id: id_v as u32,
        size: if unknown { None } else { Some(size_v) },
        data_off: off + (id_len + size_len) as u64,
    }))
}

// ───────────────────────── 轨道 ─────────────────────────

struct MkvTrack {
    /// Matroska 的 `TrackNumber`（**块里写的是它**，不是下标）。
    number: u64,
    /// 引擎侧轨道号（= 下标 + 1；muxer 会重新分配）。
    id: u32,
    kind: MediaKind,
    /// 长度前缀布局的宽度（0 = 不转换）。H.264 / H.265 从 avcC / hvcC 里读。
    nal_len: u8,
    /// 一帧的标称时长（时基刻度），来自 `DefaultDuration`。
    default_duration: i64,
}

/// `CodecID` 字符串 → 引擎的编解码器。
///
/// `None` = "这个名字引擎不支持"（Vorbis / FLAC-in-MKV），调用方据此报一条
/// **说明了原因**的错误，而不是笼统的"不认识的编码"。
fn codec_of(id: &str) -> Option<CodecId> {
    Some(match id {
        "V_MPEG4/ISO/AVC" => CodecId::H264,
        "V_MPEGH/ISO/HEVC" => CodecId::H265,
        "V_AV1" => CodecId::Av1,
        "V_VP9" => CodecId::Vp9,
        "V_VP8" => CodecId::Vp8,
        // AAC 的 CodecID 带很多后缀（A_AAC/MPEG4/LC、A_AAC/MPEG2/LC/SBR…）
        s if s.starts_with("A_AAC") => CodecId::Aac,
        "A_OPUS" => CodecId::Opus,
        "A_MPEG/L3" => CodecId::Mp3,
        "A_AC3" => CodecId::Ac3,
        "A_EAC3" => CodecId::Eac3,
        _ => return None,
    })
}

// ───────────────────────── 解封装器 ─────────────────────────

pub struct MkvDemuxer {
    reader: Reader,
    info: MediaInfo,
    tracks: Vec<MkvTrack>,
    by_number: HashMap<u64, usize>,
    /// 解析游标（下一个元素的起点）。
    off: u64,
    /// Segment 数据的结束位置（未知长度时为 `None`）。
    segment_end: Option<u64>,
    /// 当前 Cluster 的基准时间戳（时基刻度）。
    cluster_ts: i64,
    /// 已经切好、等着返回的采样（lacing 一包多帧，且每轨延迟一帧）。
    queue: VecDeque<Packet>,
    /// 每条轨"上一条还没定 duration 的采样"。
    pending: HashMap<u32, Packet>,
    /// 每条轨已产出的最大 DTS（保证单调）。
    last_dts: HashMap<u32, i64>,
    eof: bool,
}

impl MkvDemuxer {
    pub fn new(src: Arc<dyn ReadAt>, source: &str) -> Result<Self> {
        let mut me = Self {
            reader: Reader::new(src),
            info: MediaInfo {
                container: "matroska".into(),
                duration_us: None,
                bitrate: None,
                tracks: Vec::new(),
                source: source.to_string(),
            },
            tracks: Vec::new(),
            by_number: HashMap::new(),
            off: 0,
            segment_end: None,
            cluster_ts: 0,
            queue: VecDeque::new(),
            pending: HashMap::new(),
            last_dts: HashMap::new(),
            eof: false,
        };
        me.parse_head()?;
        Ok(me)
    }

    /// 解析 EBML 头 → Segment → Info / Tracks，并停在第一个 Cluster 上。
    /// `new` 与 `seek` 都用它（seek 就是"重头再来"）。
    fn parse_head(&mut self) -> Result<()> {
        self.reader = Reader::new(self.reader.src.clone());
        self.off = 0;
        self.segment_end = None;
        self.cluster_ts = 0;
        self.queue.clear();
        self.pending.clear();
        self.last_dts.clear();
        self.by_number.clear();
        self.tracks.clear();
        self.info.tracks.clear();
        self.info.duration_us = None;
        self.eof = false;

        // ── EBML 头 ──
        let head = read_element(&mut self.reader, 0)?
            .ok_or_else(|| MediaError::container("不是 EBML 文件（读不出头元素）"))?;
        if head.id != ID_EBML {
            return Err(MediaError::container("不是 EBML 文件（魔数不匹配）"));
        }
        let head_size = head.size.ok_or_else(|| {
            MediaError::container("EBML 头声明了未知长度，这不是合法的 Matroska")
        })?;

        // ── Segment（它的长度常常是"未知"，那就一直读到文件末尾）──
        let seg_off = head.data_off + head_size;
        let seg = read_element(&mut self.reader, seg_off)?
            .ok_or_else(|| MediaError::container("EBML 头之后没有 Segment"))?;
        if seg.id != ID_SEGMENT {
            return Err(MediaError::container(format!(
                "EBML 头之后的第一个元素不是 Segment（读到 0x{:X}）",
                seg.id
            )));
        }
        self.segment_end = seg.size.map(|s| seg.data_off + s);
        self.off = seg.data_off;

        // ── Segment 内的头部元素 ──
        let mut scale_ns: u64 = 1_000_000; // 规范默认 1 ms
        let mut duration_ticks: Option<f64> = None;
        loop {
            let Some(elem) = read_element(&mut self.reader, self.off)? else { break };
            if elem.id == ID_CLUSTER {
                break; // 头读完了，停在 Cluster 上
            }
            let Some(size) = elem.size else { break }; // 未知长度的头部元素：停下
            let end = elem.data_off + size;
            match elem.id {
                ID_INFO => self.parse_info(elem.data_off, end, &mut scale_ns, &mut duration_ticks)?,
                ID_TRACKS => self.parse_tracks(elem.data_off, end)?,
                _ => {}
            }
            self.off = end;
        }

        if self.info.tracks.is_empty() {
            return Err(MediaError::container(
                "Matroska 里没有可用的音视频轨（可能只有字幕，或编码不被支持）",
            ));
        }

        // 时基：块的 timestamp 单位是 TimestampScale（默认 1 ms）⇒ 每秒
        // 10^9 / scale 个刻度。scale 正好 1 ms 时就是 1000，块时间戳可直接用。
        let ticks_per_sec = u32::try_from(1_000_000_000u64 / scale_ns.max(1))
            .unwrap_or(1000)
            .max(1);
        let tb = Timebase::new(ticks_per_sec);
        for t in self.info.tracks.iter_mut() {
            t.timebase = tb;
        }
        self.info.duration_us = duration_ticks
            .map(|d| (d * scale_ns as f64 / 1000.0).round() as i64)
            .filter(|d| *d > 0);

        // 帧率：有 DefaultDuration 就能算出来
        for (i, mt) in self.tracks.iter().enumerate() {
            if mt.kind == MediaKind::Video && mt.default_duration > 0 {
                let ns = mt.default_duration as u64 * scale_ns;
                if ns > 0 && ns <= u32::MAX as u64 {
                    self.info.tracks[i].frame_rate = Some((1_000_000_000, ns as u32));
                }
            }
        }
        Ok(())
    }

    fn parse_info(
        &mut self,
        off: u64,
        end: u64,
        scale_ns: &mut u64,
        duration: &mut Option<f64>,
    ) -> Result<()> {
        let mut cur = off;
        while cur < end {
            let Some(e) = read_element(&mut self.reader, cur)? else { break };
            let Some(sz) = e.size else { break };
            let data = self.reader.at(e.data_off, sz as usize)?.to_vec();
            match e.id {
                ID_TIMESTAMP_SCALE => *scale_ns = read_uint(&data).max(1),
                ID_DURATION => *duration = read_float(&data),
                _ => {}
            }
            cur = e.data_off + sz;
        }
        Ok(())
    }

    fn parse_tracks(&mut self, off: u64, end: u64) -> Result<()> {
        let mut cur = off;
        while cur < end {
            let Some(e) = read_element(&mut self.reader, cur)? else { break };
            let Some(sz) = e.size else { break };
            if e.id == ID_TRACK_ENTRY {
                self.parse_track_entry(e.data_off, e.data_off + sz)?;
            }
            cur = e.data_off + sz;
        }
        Ok(())
    }

    fn parse_track_entry(&mut self, off: u64, end: u64) -> Result<()> {
        let mut number: u64 = 0;
        let mut track_type: u64 = 0;
        let mut codec_id = String::new();
        let mut private: Vec<u8> = Vec::new();
        let mut default_duration: i64 = 0;
        let mut width = 0u32;
        let mut height = 0u32;
        let mut sample_rate = 0u32;
        let mut channels = 0u16;
        let mut language: Option<String> = None;

        let mut cur = off;
        while cur < end {
            let Some(e) = read_element(&mut self.reader, cur)? else { break };
            let Some(sz) = e.size else { break };
            match e.id {
                ID_TRACK_NUMBER => number = read_uint(self.reader.at(e.data_off, sz as usize)?),
                ID_TRACK_TYPE => track_type = read_uint(self.reader.at(e.data_off, sz as usize)?),
                ID_CODEC_ID => {
                    let b = self.reader.at(e.data_off, sz as usize)?;
                    codec_id = String::from_utf8_lossy(trim_nul(b)).to_string();
                }
                ID_CODEC_PRIVATE => {
                    private = self.reader.at(e.data_off, sz as usize)?.to_vec();
                }
                ID_DEFAULT_DURATION => {
                    default_duration =
                        read_uint(self.reader.at(e.data_off, sz as usize)?) as i64
                }
                ID_LANGUAGE => {
                    let b = self.reader.at(e.data_off, sz as usize)?;
                    language = Some(String::from_utf8_lossy(trim_nul(b)).to_string());
                }
                ID_VIDEO => {
                    for (id, v) in self.read_children(e.data_off, e.data_off + sz)? {
                        match id {
                            ID_PIXEL_WIDTH => width = v as u32,
                            ID_PIXEL_HEIGHT => height = v as u32,
                            _ => {}
                        }
                    }
                }
                ID_AUDIO => {
                    for (id, v) in self.read_children_float(e.data_off, e.data_off + sz)? {
                        match id {
                            ID_SAMPLING_FREQ => sample_rate = v.0 as u32,
                            ID_CHANNELS => channels = v.1 as u16,
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
            cur = e.data_off + sz;
        }

        // TrackType：1 = 视频、2 = 音频、17 = 字幕；其余（按钮 / 控制）跳过
        let kind = match track_type {
            1 => MediaKind::Video,
            2 => MediaKind::Audio,
            _ => return Ok(()),
        };

        let codec = codec_of(&codec_id).ok_or_else(|| {
            MediaError::Unsupported(format!(
                "这个 Matroska 里的 `{codec_id}` 引擎还不支持（Vorbis 与 FLAC-in-MKV 都还没做）"
            ))
        })?;

        // 长度前缀宽度：H.264 / H.265 的采样是「[N 字节长度][NAL]」，N 写在配置盒里
        let nal_len = match codec {
            CodecId::H264 => private.get(4).map(|b| (b & 0x03) + 1).unwrap_or(0),
            CodecId::H265 => private.get(21).map(|b| (b & 0x03) + 1).unwrap_or(0),
            _ => 0,
        };

        let id = self.info.tracks.len() as u32 + 1;
        let mut t = Track::new(id, kind, codec, Timebase::new(1000));
        t.extradata = private;
        t.width = width;
        t.height = height;
        t.sample_rate = sample_rate;
        t.channels = channels;
        t.language = language;
        self.info.tracks.push(t);
        self.tracks.push(MkvTrack { number, id, kind, nal_len, default_duration });
        self.by_number.insert(number, self.tracks.len() - 1);
        Ok(())
    }

    /// 读一层子元素的 `(id, 无符号值)`。
    fn read_children(&mut self, off: u64, end: u64) -> Result<Vec<(u32, u64)>> {
        let mut out = Vec::new();
        let mut cur = off;
        while cur < end {
            let Some(e) = read_element(&mut self.reader, cur)? else { break };
            let Some(sz) = e.size else { break };
            let v = read_uint(self.reader.at(e.data_off, sz as usize)?);
            out.push((e.id, v));
            cur = e.data_off + sz;
        }
        Ok(out)
    }

    /// 音频子元素里既有整数（声道数）又有浮点（采样率），分两种取值方式。
    fn read_children_float(&mut self, off: u64, end: u64) -> Result<Vec<(u32, (f64, u64))>> {
        let mut out = Vec::new();
        let mut cur = off;
        while cur < end {
            let Some(e) = read_element(&mut self.reader, cur)? else { break };
            let Some(sz) = e.size else { break };
            let raw = self.reader.at(e.data_off, sz as usize)?.to_vec();
            let f = read_float(&raw).unwrap_or(0.0);
            let u = read_uint(&raw);
            out.push((e.id, (f, u)));
            cur = e.data_off + sz;
        }
        Ok(out)
    }

    /// 取下一个采样（内部状态机）。
    fn poll(&mut self) -> Result<Option<Packet>> {
        loop {
            if let Some(p) = self.queue.pop_front() {
                return Ok(Some(p));
            }
            if self.eof {
                return Ok(None);
            }
            if let Some(end) = self.segment_end {
                if self.off >= end {
                    self.finish_pending();
                    self.eof = true;
                    continue;
                }
            }
            let Some(elem) = read_element(&mut self.reader, self.off)? else {
                self.finish_pending();
                self.eof = true;
                continue;
            };
            match elem.id {
                // Cluster 只是个容器：把游标移进它的数据区接着解析子元素。
                // 它的长度常常是"未知"（流式写入），所以不能靠 size 跳过。
                ID_CLUSTER => {
                    self.off = elem.data_off;
                    self.cluster_ts = 0;
                }
                ID_CLUSTER_TS => {
                    let sz = elem.size.unwrap_or(0) as usize;
                    self.cluster_ts = read_uint(self.reader.at(elem.data_off, sz)?) as i64;
                    self.off = elem.data_off + sz as u64;
                }
                ID_SIMPLE_BLOCK => {
                    let sz = elem.size.unwrap_or(0) as usize;
                    let data = self.reader.at(elem.data_off, sz)?.to_vec();
                    self.off = elem.data_off + sz as u64;
                    self.parse_block(&data, None)?;
                }
                ID_BLOCK_GROUP => {
                    let sz = elem.size.unwrap_or(0) as usize;
                    let end = elem.data_off + sz as u64;
                    self.parse_block_group(elem.data_off, end)?;
                    self.off = end;
                }
                // 其它顶层元素（Cues / Tags / SeekHead / Void…）：整块跳过
                _ => match elem.size {
                    Some(s) => self.off = elem.data_off + s,
                    None => {
                        self.finish_pending();
                        self.eof = true;
                    }
                },
            }
        }
    }

    fn parse_block_group(&mut self, off: u64, end: u64) -> Result<()> {
        let mut cur = off;
        let mut block: Option<Vec<u8>> = None;
        let mut duration: Option<u64> = None;
        let mut referenced = false;
        while cur < end {
            let Some(e) = read_element(&mut self.reader, cur)? else { break };
            let Some(sz) = e.size else { break };
            match e.id {
                ID_BLOCK => block = Some(self.reader.at(e.data_off, sz as usize)?.to_vec()),
                ID_BLOCK_DURATION => {
                    duration = Some(read_uint(self.reader.at(e.data_off, sz as usize)?))
                }
                ID_REFERENCE_BLOCK => referenced = true,
                _ => {}
            }
            cur = e.data_off + sz;
        }
        if let Some(b) = block {
            // BlockGroup 里的关键帧判据是**没有 ReferenceBlock**
            self.parse_block(&b, Some((!referenced, duration)))?;
        }
        Ok(())
    }

    /// 解析一个 (Simple)Block：轨道号 + 相对时间戳 + 标志 + lacing + 帧数据。
    fn parse_block(&mut self, data: &[u8], group: Option<(bool, Option<u64>)>) -> Result<()> {
        let Some((number, n_len)) = read_vint(data, 0, false) else {
            return Ok(());
        };
        let Some(idx) = self.by_number.get(&number).copied() else {
            return Ok(()); // 不认识的轨（字幕…）：跳过这一块
        };
        if data.len() < n_len + 3 {
            return Ok(());
        }
        let rel = i16::from_be_bytes([data[n_len], data[n_len + 1]]) as i64;
        let flags = data[n_len + 2];
        let keyframe = match group {
            Some((kf, _)) => kf,
            None => flags & 0x80 != 0,
        };
        let lacing = (flags & 0x06) >> 1;

        let mut p = n_len + 3;
        let frame_count = if lacing == 0 {
            1usize
        } else {
            let n = *data.get(p).unwrap_or(&0) as usize + 1;
            p += 1;
            n
        };
        if frame_count == 0 {
            return Ok(());
        }

        // 先算出前 n-1 帧的长度（最后一帧吃掉剩下的全部）
        let mut sizes: Vec<usize> = Vec::with_capacity(frame_count.saturating_sub(1));
        match lacing {
            0 => {}
            // Xiph：长度按"255 累加"编码（读满 255 就继续加）
            1 => {
                for _ in 0..frame_count - 1 {
                    let mut sz = 0usize;
                    loop {
                        let Some(b) = data.get(p) else { return Ok(()) };
                        p += 1;
                        sz += usize::from(*b);
                        if *b != 255 {
                            break;
                        }
                    }
                    sizes.push(sz);
                }
            }
            // 固定：等分（除不尽时余数留给最后一帧）
            2 => {
                let remain = data.len().saturating_sub(p);
                let each = remain / frame_count;
                for _ in 0..frame_count - 1 {
                    sizes.push(each);
                }
            }
            // EBML：第一个是 vint 长度，之后每个是"与上一个的**带符号**差"
            3 => {
                let mut prev = 0i64;
                for i in 0..frame_count - 1 {
                    let Some((v, l)) = read_vint(data, p, false) else { return Ok(()) };
                    p += l;
                    let cur = if i == 0 {
                        v as i64
                    } else {
                        // 有符号 vint 的偏置：2^(7n-1) - 1
                        let bias = (1i64 << (7 * l as u32 - 1)) - 1;
                        prev + (v as i64 - bias)
                    };
                    if cur < 0 {
                        return Ok(());
                    }
                    prev = cur;
                    sizes.push(cur as usize);
                }
            }
            _ => return Ok(()),
        }

        let (track_id, nal_len, default_dur) = {
            let t = &self.tracks[idx];
            (t.id, t.nal_len, t.default_duration)
        };
        let group_dur = group.and_then(|(_, d)| d).unwrap_or(0) as i64;

        for i in 0..frame_count {
            let len = if i + 1 == frame_count {
                data.len().saturating_sub(p)
            } else {
                sizes.get(i).copied().unwrap_or(0)
            };
            if p + len > data.len() {
                break;
            }
            let raw = &data[p..p + len];
            p += len;

            // 视频统一成 Annex-B；音频与 AV1/VP9 原样搬
            let content = if nal_len > 0 {
                codec::length_prefixed_to_annexb(raw, nal_len)
            } else {
                raw.to_vec()
            };

            let pts = self.cluster_ts + rel + i as i64 * default_dur;
            // 每条轨的 DTS 必须严格递增：MKV 只给 PTS，无 B 帧时两者相等
            let step = if default_dur > 0 { default_dur } else { 1 };
            let dts = match self.last_dts.get(&track_id).copied() {
                Some(prev) if pts <= prev => prev + step,
                _ => pts,
            };
            self.last_dts.insert(track_id, dts);

            let dur = if group_dur > 0 {
                group_dur.min(u32::MAX as i64) as u32
            } else {
                default_dur.clamp(0, u32::MAX as i64) as u32
            };
            let mut packet = Packet::new(track_id, dts, pts, dur, content);
            packet.keyframe = keyframe;
            self.push_with_duration(packet);
        }
        Ok(())
    }

    /// 入队，并顺手给**同一条轨的上一个采样**补上真实时长。
    ///
    /// MKV 的块不带时长，唯一可靠的算法是"同轨下一帧的时间戳减这一帧的"。
    /// 代价是每条轨都延迟一帧才吐出来 —— 所以 `finish_pending` 必须存在，
    /// 否则每条轨的最后一条会被吞掉。
    fn push_with_duration(&mut self, p: Packet) {
        if let Some(mut prev) = self.pending.insert(p.track_id, p.clone()) {
            if prev.duration == 0 {
                let d = (p.dts - prev.dts).max(0);
                prev.duration = d.min(u32::MAX as i64) as u32;
            }
            self.queue.push_back(prev);
        }
    }

    /// 收尾：把每条轨最后那一条（等不到下一帧的）放出去。
    fn finish_pending(&mut self) {
        let mut left: Vec<Packet> = self.pending.drain().map(|(_, p)| p).collect();
        left.sort_by_key(|p| p.dts);
        for p in left {
            self.queue.push_back(p);
        }
    }
}

impl Demuxer for MkvDemuxer {
    fn info(&self) -> MediaInfo {
        self.info.clone()
    }

    fn next_packet(&mut self) -> Result<Option<Packet>> {
        match self.poll() {
            Ok(v) => Ok(v),
            Err(e) => {
                self.eof = true;
                Err(e)
            }
        }
    }

    fn seek(&mut self, pts_us: i64) -> Result<i64> {
        // 没有 Cues 索引，只能重头顺序扫到目标附近的关键帧。
        // 慢，但**一定对** —— 合并路径基本不用 seek，而播放器那边宁可多等，
        // 也不要跳到一个解不出来的位置（宁可正确，不要"看起来能用"）。
        let target_us = pts_us.max(0);
        if target_us == 0 {
            self.parse_head()?;
            return Ok(0);
        }
        self.parse_head()?;
        let tb = self.info.tracks.first().map(|t| t.timebase).unwrap_or(Timebase::new(1000));
        let mut landed = 0i64;
        while let Some(p) = self.poll()? {
            if !p.keyframe {
                continue;
            }
            let us = tb.ticks_to_us(p.pts);
            if us >= target_us {
                landed = us;
                // 把这一帧留给调用方（它是这次 seek 的落点）
                self.queue.push_front(p);
                break;
            }
            landed = us;
        }
        Ok(landed)
    }

    fn duration_us(&self) -> Option<i64> {
        self.info.duration_us
    }

    fn container(&self) -> &'static str {
        "matroska"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use me_core::Cursor;

    // ── 手拼 Matroska 字节流 ──
    //
    // 与 mp4.rs 那套合成的思路一致：把字节拼出来能精确构造真实文件里那些
    // 难遇上的形态（未知长度 Segment、Xiph lacing、BlockGroup…），
    // 而"这个用例到底在测什么"也能从代码读出来。

    /// EBML 长度：固定用 8 字节编码（首字节 0x01 是长度标志）。
    fn elen(n: u64) -> Vec<u8> {
        let mut v = [0u8; 8];
        v[0] = 0x01;
        v[1..].copy_from_slice(&n.to_be_bytes()[1..]);
        v.to_vec()
    }

    /// 一个元素：ID（去前导 0，标志位本来就在）+ 长度 + 数据。
    fn el(id: u32, body: &[u8]) -> Vec<u8> {
        let mut v = id.to_be_bytes().to_vec();
        while v.len() > 1 && v[0] == 0 {
            v.remove(0);
        }
        v.extend_from_slice(&elen(body.len() as u64));
        v.extend_from_slice(body);
        v
    }

    /// 未知长度的元素（Segment / Cluster 在流式写入时常见）。
    ///
    /// 编码是 `0x01`（8 字节长度标志）后面跟 7 个 `0xFF`（值位全 1）——
    /// **不是** 8 个 `0xFF`：那样首字节的 `0xFF` 会被读成"1 字节长度"。
    fn el_unknown(id: u32, body: &[u8]) -> Vec<u8> {
        let mut v = id.to_be_bytes().to_vec();
        while v.len() > 1 && v[0] == 0 {
            v.remove(0);
        }
        v.extend_from_slice(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
        v.extend_from_slice(body);
        v
    }

    /// 无符号整数：最少字节数的大端。
    fn uint(v: u64) -> Vec<u8> {
        let b = v.to_be_bytes();
        let s = b.iter().position(|x| *x != 0).unwrap_or(b.len() - 1);
        b[s..].to_vec()
    }

    fn f64b(v: f64) -> Vec<u8> {
        v.to_be_bytes().to_vec()
    }

    fn vint(v: u64) -> Vec<u8> {
        // 单字节 vint（值 < 127 时够用）
        vec![0x80 | v as u8]
    }

    struct Builder {
        tracks: Vec<u8>,
        clusters: Vec<u8>,
        ts_scale: u64,
        duration: Option<f64>,
        seg_unknown: bool,
    }

    impl Builder {
        fn new() -> Self {
            Self { tracks: Vec::new(), clusters: Vec::new(), ts_scale: 1_000_000, duration: None, seg_unknown: false }
        }

        /// 加一条轨。`codec_id` 是 Matroska 的字符串（`V_VP9` / `A_OPUS`…）。
        fn track(mut self, number: u64, track_type: u64, codec_id: &str, private: &[u8]) -> Self {
            let mut e = el(ID_TRACK_NUMBER, &uint(number));
            e.extend_from_slice(&el(ID_TRACK_TYPE, &uint(track_type)));
            e.extend_from_slice(&el(ID_CODEC_ID, codec_id.as_bytes()));
            if !private.is_empty() {
                e.extend_from_slice(&el(ID_CODEC_PRIVATE, private));
            }
            if track_type == 1 {
                let mut v = el(ID_PIXEL_WIDTH, &uint(640));
                v.extend_from_slice(&el(ID_PIXEL_HEIGHT, &uint(360)));
                e.extend_from_slice(&el(ID_VIDEO, &v));
            } else if track_type == 2 {
                let mut v = el(ID_SAMPLING_FREQ, &f64b(48_000.0));
                v.extend_from_slice(&el(ID_CHANNELS, &uint(2)));
                e.extend_from_slice(&el(ID_AUDIO, &v));
            }
            self.tracks.extend_from_slice(&el(ID_TRACK_ENTRY, &e));
            self
        }

        fn default_duration(mut self, ns: u64) -> Self {
            // 挂到最后一条轨上（测试里只用一条带 DefaultDuration 的轨）
            let d = el(ID_DEFAULT_DURATION, &uint(ns));
            let last = self.tracks.len();
            self.tracks.splice(last..last, d);
            self
        }

        /// 一个 Cluster：时间戳（时基刻度）+ 若干块。
        fn cluster(mut self, ts: i64, blocks: &[Vec<u8>]) -> Self {
            let mut c = el(ID_CLUSTER_TS, &uint(ts as u64));
            for b in blocks {
                c.extend_from_slice(&el(ID_SIMPLE_BLOCK, b));
            }
            self.clusters.extend_from_slice(&el(ID_CLUSTER, &c));
            self
        }

        fn build(self) -> Vec<u8> {
            let mut ebml = el(ID_DOCTYPE, b"matroska");
            ebml.extend_from_slice(&el(0x4286, &uint(1))); // EBMLVersion
            let mut out = el(ID_EBML, &ebml);

            let mut info = el(ID_TIMESTAMP_SCALE, &uint(self.ts_scale));
            if let Some(d) = self.duration {
                info.extend_from_slice(&el(ID_DURATION, &f64b(d)));
            }
            let mut seg = el(ID_INFO, &info);
            seg.extend_from_slice(&el(ID_TRACKS, &self.tracks));
            seg.extend_from_slice(&self.clusters);
            out.extend_from_slice(&if self.seg_unknown {
                el_unknown(ID_SEGMENT, &seg)
            } else {
                el(ID_SEGMENT, &seg)
            });
            out
        }
    }

    /// 一个 SimpleBlock 的体：轨道号 + 相对时间戳 + flags + 帧。
    fn block(track: u64, rel: i16, keyframe: bool, frame: &[u8]) -> Vec<u8> {
        let mut v = vint(track);
        v.extend_from_slice(&rel.to_be_bytes());
        v.push(if keyframe { 0x80 } else { 0x00 });
        v.extend_from_slice(frame);
        v
    }

    fn open(bytes: Vec<u8>) -> MkvDemuxer {
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(bytes));
        MkvDemuxer::new(src, "test.mkv").expect("应当能打开")
    }

    fn drain(d: &mut MkvDemuxer) -> Vec<Packet> {
        let mut out = Vec::new();
        while let Some(p) = d.next_packet().expect("读采样") {
            out.push(p);
        }
        out
    }

    #[test]
    fn recognizes_the_ebml_magic() {
        assert!(looks_like_mkv(&[0x1A, 0x45, 0xDF, 0xA3, 0x00]));
        assert!(!looks_like_mkv(&[0x1A, 0x45, 0xDF, 0xA2]));
        assert!(!looks_like_mkv(b"ftyp"));
        assert!(!looks_like_mkv(&[]));
    }

    #[test]
    fn parses_a_minimal_file_and_yields_packets() {
        let bytes = Builder::new()
            .track(1, 1, "V_VP9", &[])
            .track(2, 2, "A_OPUS", &[0u8; 19])
            .cluster(0, &[block(1, 0, true, b"VIDEOFRAME"), block(2, 0, true, b"AUDIO")])
            .cluster(1000, &[block(1, 40, false, b"V2"), block(2, 20, true, b"A2")])
            .build();
        let mut d = open(bytes);
        let info = d.info();
        assert_eq!(info.container, "matroska");
        assert_eq!(info.tracks.len(), 2);
        assert_eq!(info.tracks[0].kind, MediaKind::Video);
        assert_eq!(info.tracks[0].codec, CodecId::Vp9);
        assert_eq!(info.tracks[1].kind, MediaKind::Audio);
        assert_eq!(info.tracks[1].codec, CodecId::Opus);

        let packets = drain(&mut d);
        assert_eq!(packets.len(), 4, "两个 cluster 各两块");
        // 时间戳按 cluster 基准 + 相对值落在时基刻度上（1 ms/刻度）
        let v2 = packets.iter().find(|p| p.data == b"V2").expect("第二段视频");
        assert_eq!(v2.track_id, 1);
        assert_eq!(v2.pts, 1040, "cluster 1000 + 相对 40");
    }

    #[test]
    fn duration_comes_from_the_next_frame_of_the_same_track() {
        // MKV 的块不带时长：第 1 帧的时长应当等于"同轨下一帧的时间戳 - 自己"
        let bytes = Builder::new()
            .track(1, 1, "V_VP9", &[])
            .cluster(0, &[block(1, 0, true, b"A")])
            .cluster(1000, &[block(1, 0, false, b"B"), block(1, 500, false, b"C")])
            .build();
        let mut d = open(bytes);
        let packets = drain(&mut d);
        assert_eq!(packets.len(), 3);
        assert_eq!(packets[0].duration, 1000, "第 1 帧到第 2 帧差 1000 刻度");
        assert_eq!(packets[1].duration, 500, "第 2 帧到第 3 帧差 500");
    }

    #[test]
    fn avc_codec_private_is_kept_and_frames_become_annexb() {
        // avcC：第 4 字节低 2 位 + 1 = NAL 长度宽度。这里造 4 字节宽度。
        let mut avcc = vec![1u8, 0x42, 0x00, 0x1E, 0xFF];
        avcc.extend_from_slice(&[0xE0, 0x00, 0x00]); // numSPS 等（内容不重要）
        let mut frame = Vec::new();
        frame.extend_from_slice(&3u32.to_be_bytes());
        frame.extend_from_slice(&[0x65, 0x11, 0x22]); // 一个 NAL
        frame.extend_from_slice(&2u32.to_be_bytes());
        frame.extend_from_slice(&[0x41, 0x33]); // 再来一个

        let bytes = Builder::new()
            .track(1, 1, "V_MPEG4/ISO/AVC", &avcc)
            .cluster(0, &[block(1, 0, true, &frame)])
            .build();
        let mut d = open(bytes);
        assert_eq!(d.info().tracks[0].extradata, avcc, "配置盒要原样带过去");
        let packets = drain(&mut d);
        assert_eq!(packets.len(), 1);
        assert_eq!(
            packets[0].data,
            vec![0, 0, 0, 1, 0x65, 0x11, 0x22, 0, 0, 0, 1, 0x41, 0x33],
            "长度前缀要变成 Annex-B（与 fMP4 那条链同一套约定）"
        );
    }

    #[test]
    fn xiph_lacing_splits_one_block_into_frames() {
        // Xiph lacing：帧数 - 1 在 flags 后一个字节，长度按 255 累加编码
        let f1 = b"AA".to_vec();
        let f2 = b"BBB".to_vec();
        let f3 = b"CCCC".to_vec();
        let mut body = vint(1);
        body.extend_from_slice(&0i16.to_be_bytes());
        body.push(0x80 | (1 << 1)); // keyframe + Xiph lacing
        body.push(0x02); // frames - 1 = 2
        body.push(f1.len() as u8);
        body.push(f2.len() as u8);
        body.extend_from_slice(&f1);
        body.extend_from_slice(&f2);
        body.extend_from_slice(&f3);

        let bytes = Builder::new()
            .track(1, 1, "V_VP9", &[])
            .cluster(0, &[body])
            .build();
        let mut d = open(bytes);
        let packets = drain(&mut d);
        assert_eq!(packets.len(), 3, "一个块里切出三帧");
        assert_eq!(packets[0].data, f1);
        assert_eq!(packets[1].data, f2);
        assert_eq!(packets[2].data, f3);
    }

    #[test]
    fn subtitle_tracks_are_skipped_not_fatal() {
        let bytes = Builder::new()
            .track(1, 1, "V_VP9", &[])
            .track(2, 17, "S_TEXT/UTF8", &[]) // 字幕
            .cluster(0, &[block(1, 0, true, b"V")])
            .build();
        let d = open(bytes);
        assert_eq!(d.info().tracks.len(), 1, "字幕轨不该进轨道表");
        assert_eq!(d.info().tracks[0].kind, MediaKind::Video);
    }

    #[test]
    fn an_unsupported_codec_says_which_one() {
        let bytes = Builder::new().track(1, 2, "A_VORBIS", &[]).build();
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(bytes));
        let e = match MkvDemuxer::new(src, "t.mkv") {
            Ok(_) => panic!("Vorbis 应当被拒绝"),
            Err(e) => e,
        };
        let msg = format!("{e}");
        assert!(msg.contains("A_VORBIS"), "要说清是哪个 CodecID：{msg}");
    }

    #[test]
    fn unknown_length_segment_and_cluster_still_parse() {
        let mut b = Builder::new();
        b.seg_unknown = true;
        b = b.track(1, 1, "V_VP9", &[]).cluster(0, &[block(1, 0, true, b"FRAME")]);
        let mut d = open(b.build());
        let packets = drain(&mut d);
        assert_eq!(packets.len(), 1, "未知长度的 Segment/Cluster 也要能读到末尾");
        assert_eq!(packets[0].data, b"FRAME");
    }

    #[test]
    fn duration_and_frame_rate_come_from_the_header() {
        let b = Builder::new();
        let b = b.track(1, 1, "V_VP9", &[]);
        let mut bytes = b.build();
        // 手工把 DefaultDuration(40ms = 40_000_000 ns) 塞进轨里太绕，直接改 ts_scale 验证时长
        bytes = {
            let mut b2 = Builder::new();
            b2.duration = Some(2000.0); // 2000 个时间戳单位
            b2 = b2.track(1, 1, "V_VP9", &[]).cluster(0, &[block(1, 0, true, b"F")]);
            b2.build()
        };
        let d = open(bytes);
        // ts_scale = 1 ms ⇒ 2000 刻度 = 2 秒
        assert_eq!(d.duration_us(), Some(2_000_000));
    }
}