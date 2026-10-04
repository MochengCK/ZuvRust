//! MPEG-TS 解封装（ISO/IEC 13818-1）。
//!
//! 这是引擎里**最吃力不讨好**的一层：TS 没有任何索引，所有信息都得顺着
//! 188 字节的包一个个扫出来，而且真实世界的流普遍不合规（时间戳跳、包丢失、
//! 一个 PES 里塞好几帧、PAT/PMT 几分钟才重复一次）。所以这里的每一处"宽容"
//! 都是被真实片子打出来的。
//!
//! ## 三条包长都要认
//!
//! | 形态 | 包长 | TS 包在包内的偏移 | 出处 |
//! |---|---|---|---|
//! | 裸 TS | 188 | 0 | 绝大多数 HLS 分片 |
//! | M2TS | 192 | 4 | 蓝光、部分广播 |
//! | DVB | 204 | 0（后 16 字节是 RS 校验） | 卫星录制 |
//!
//! ## 输出粒度
//!
//! - **视频**：一个 PES 一个 [`Packet`]（绝大多数 TS 是一帧一 PES）。
//!   TS 里没有"帧"这个概念，硬要拆帧就得解析 slice header 的 `first_mb_in_slice`，
//!   那是解码器的工作而不是封装层的工作；重封装到 TS 时 PES 粒度完全够用。
//! - **音频（AAC）**：一个 ADTS 帧一个 [`Packet`]。
//!   音频必须拆到帧：一个 PES 里常常塞十几帧，重封装到 fMP4 时
//!   "一个 sample 一个 frame"是硬要求，不拆就会写出一堆跨帧的畸形 sample。
//!
//! ## 时间戳
//!
//! PTS/DTS 是 90kHz 的 33 位值，**会回绕**（约 26.5 小时一轮）。
//! 这里的处理是：记录第一个 PTS 作为原点，之后每个时间戳都做一次
//! "相对上一条不后退太多"的展开（见 [`unwrap_pts`]）。不做展开的话，
//! 长时间录制会在回绕点突然跳到 0，交织器会以为"时间倒流"而乱序输出。

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use me_core::{CodecId, MediaError, MediaInfo, MediaKind, Packet, ReadAt, Result, Timebase, Track};

use crate::codec;
use crate::probe::ContainerFormat;

/// 裸 TS 的包长（规范固定，其余包长都是它的变体）。
pub const TS_PACKET_LEN: usize = 188;
/// 同步字。
pub const SYNC_BYTE: u8 = 0x47;
/// 一次最多缓冲多少字节的输入（64 KiB ≈ 348 个包）。
const CHUNK: usize = 64 * 1024;
/// 探测时最多扫多少个 TS 包（≈750 KB，足够拿到 PAT/PMT/SPS 与若干帧）。
const PROBE_PACKET_BUDGET: usize = 4096;
/// 探测时"值得扫到底"的最大源大小。
///
/// 扫到底才能用时长的**精确值**（最后时间戳 − 第一个时间戳）；超过这个大小就退回
/// "按实测码率外推"（有误差，但探测不能为了时长去读几个 GB）。
/// 8 MB 对本地文件只是十几毫秒的 IO，却能覆盖 HLS 分片、短片、录屏片段这些
/// 最常被探测的对象。
const PROBE_SCAN_BYTES_MAX: u64 = 8 * 1024 * 1024;
/// 33 位 PTS 的回绕周期。
const PTS_WRAP: i64 = 1 << 33;

/// 认包长：返回整包长度（188 / 192 / 204）。
///
/// 判据是"**同步点占比**"，而不是"前 4 个同步点全对上"：
///
/// - 只看 1 个同步字太松：随机数据有 1/256 的概率命中，那样的"成功探测"比失败更糟；
/// - 只看前 4 个又太紧：真实流丢一个包是常态（丢包、插广告、录制被打断），
///   而前 4 个包里坏一个就直接判成"这不是 TS"，等于整条流白给。
///
/// 所以这里在头部按候选包长**数够多的点**，要求至少 4 个命中且占比 ≥ 70%。
/// 随机数据命中 4 个点的概率约 (1/256)^4，可以忽略；真实流丢一两个包照样过。
pub fn packet_size_of(head: &[u8]) -> Option<usize> {
    /// 最多数多少个同步点（多了对判定没有增益，只是白烧 CPU）。
    const MAX_POINTS: usize = 64;
    for &size in &[188usize, 192, 204] {
        let base = if size == 192 { 4 } else { 0 };
        if head.len() <= base || head[base] != SYNC_BYTE {
            continue;
        }
        let mut checked = 0usize;
        let mut hits = 0usize;
        let mut pos = base;
        while pos < head.len() && checked < MAX_POINTS {
            checked += 1;
            if head[pos] == SYNC_BYTE {
                hits += 1;
            }
            pos += size;
        }
        if checked >= 4 && hits >= 4 && hits * 10 >= checked * 7 {
            return Some(size);
        }
    }
    None
}

/// 展开回绕的 PTS：让时间戳相对 `reference` 落在 `±PTS_WRAP/2` 之内。
fn unwrap_pts(raw: i64, reference: i64) -> i64 {
    let mut v = raw;
    while v - reference > PTS_WRAP / 2 {
        v -= PTS_WRAP;
    }
    while reference - v > PTS_WRAP / 2 {
        v += PTS_WRAP;
    }
    v
}

/// 5 字节的 PTS/DTS 编码（33 位，穿插标记位）。
pub fn read_timestamp(b: &[u8]) -> i64 {
    let b0 = b[0] as i64;
    let b1 = b[1] as i64;
    let b2 = b[2] as i64;
    let b3 = b[3] as i64;
    let b4 = b[4] as i64;
    ((b0 & 0x0E) << 29) | (b1 << 22) | ((b2 & 0xFE) << 14) | (b3 << 7) | ((b4 & 0xFE) >> 1)
}

/// ADTS 头里我们关心的字段。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdtsFrame {
    pub header_len: usize,
    pub frame_len: usize,
    pub sample_rate: u32,
    pub channels: u16,
    /// ADTS 的 `profile`（0=Main, 1=LC, 2=SSR）。LC 是最常见的。
    pub profile: u8,
    pub sf_index: u8,
}

const ADTS_SAMPLE_RATES: [u32; 13] = [
    96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
];

/// ADTS 帧的同步字判定（12 位同步 + layer=00）。
///
/// 单独拿出来是因为"重新对齐"要用它：丢包或从流中间接入时，开头几个字节
/// 往往是半个帧，得先找到下一个同步字才能继续。
fn is_adts_sync(data: &[u8]) -> bool {
    data.len() >= 2 && data[0] == 0xFF && (data[1] & 0xF0) == 0xF0
}

/// 从 `data` 里找下一个 ADTS 同步字的偏移。
fn find_adts_sync(data: &[u8]) -> Option<usize> {
    if data.len() < 2 {
        return None;
    }
    let mut i = 1usize;
    while i + 1 < data.len() {
        if data[i] == 0xFF && (data[i + 1] & 0xF0) == 0xF0 {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// 解析一个 ADTS 帧头。
pub fn parse_adts(data: &[u8]) -> Option<AdtsFrame> {
    if data.len() < 7 {
        return None;
    }
    if !is_adts_sync(data) {
        return None;
    }
    let protection_absent = data[1] & 0x01 == 1;
    let profile = (data[2] >> 6) & 0x03;
    let sf_index = (data[2] >> 2) & 0x0F;
    if sf_index as usize >= ADTS_SAMPLE_RATES.len() {
        return None;
    }
    let channels = ((data[2] & 0x01) << 2) | ((data[3] >> 6) & 0x03);
    let frame_len = ((data[3] as usize & 0x03) << 11) | ((data[4] as usize) << 3) | ((data[5] as usize) >> 5);
    let header_len = if protection_absent { 7 } else { 9 };
    if frame_len < header_len {
        return None;
    }
    Some(AdtsFrame {
        header_len,
        frame_len,
        sample_rate: ADTS_SAMPLE_RATES[sf_index as usize],
        channels: channels as u16,
        profile,
        sf_index,
    })
}

/// 由 ADTS 头合成 fMP4 要用的 `AudioSpecificConfig`（2 字节）。
pub fn adts_to_asc(a: &AdtsFrame) -> Vec<u8> {
    // audioObjectType(5) | samplingFrequencyIndex(4) | channelConfiguration(4) | 3 位 0
    let aot = (a.profile as u32 + 1) & 0x1F;
    let bits: u32 = (aot << 11) | ((a.sf_index as u32) << 7) | ((a.channels as u32) << 3);
    vec![(bits >> 8) as u8, (bits & 0xFF) as u8]
}

/// 一条轨道的解析态。
#[derive(Debug, Clone)]
struct TrackState {
    id: u32,
    codec: CodecId,
    kind: MediaKind,
    /// 已从码流里攒到的参数集（H.264/H.265 的 SPS+PPS，Annex-B 形态）。
    extradata: Vec<u8>,
    /// AAC 的 AudioSpecificConfig（首次见到 ADTS 时合成）。
    asc: Option<Vec<u8>>,
    sample_rate: u32,
    channels: u16,
    width: u32,
    height: u32,
    language: Option<String>,
    /// 视频帧计数（诊断用；帧率不用它算，见 `deltas`）。
    frames: u64,
    /// 最近若干条视频 PES 的时间戳间隔（90kHz 刻度）。
    ///
    /// 帧率取这些间隔的**中位数**，而不是"帧数 ÷ 跨度"：
    /// 后者要求跨度恰好等于帧数 × 每帧时长，而真实流开头结尾的间隔常常不齐
    /// （实测一条 29.97fps 的流被算成 31.2fps）；而且**重复扫描同一段数据时
    /// 帧数会翻倍、跨度不变**，估出来的帧率直接错一倍。中位数对这两件事都免疫。
    deltas: VecDeque<i64>,
    /// 上一条视频 PES 的**解码**时间戳（用来算帧间隔）。
    ///
    /// 用 dts 而不是 pts：B 帧存在时，pts 在解码顺序里会往回跳
    /// （显示顺序 ≠ 解码顺序），拿 pts 的差值估帧率会估出"10fps"这种荒唐值；
    /// dts 在解码顺序里单调递增，差值就是每帧时长。
    last_dts: Option<i64>,
    first_pts: Option<i64>,
    last_end: i64,
}

/// 跨 PES 的 AAC 残帧。
///
/// 除了字节，还要记住**这一帧是从哪条 PES 开始的**：帧的显示时间戳属于它开始
/// 的那条 PES，用"续上来的那条 PES"的 PTS 会让整条音轨的时间戳逐段往后飘。
#[derive(Debug, Clone)]
struct AdtsCarry {
    data: Vec<u8>,
    /// 这条残帧所属 PES 的 PTS。
    pts: i64,
    /// 在那条 PES 里，这帧之前已经吐出了几帧（用来算这帧的 pts）。
    frames_before: i64,
}

/// 每个 PID 的 PES 重组器。
#[derive(Debug, Default)]
struct PesAssembler {
    buf: Vec<u8>,
    /// PES 头里声明的载荷长度（0 = 不定长，视频常见）。
    expected: usize,
    /// 载荷在 `buf` 里的起点（PES 头长度是变的）。
    es_start: usize,
    pts: Option<i64>,
    dts: Option<i64>,
    /// 是否已经读到过 PES 头（没读到之前的数据要丢掉 —— 说明我们是从中间接进来的）。
    started: bool,
    /// 这一串 PES 的起点带随机访问标志。
    random_access: bool,
}

/// MPEG-TS 解封装器。
pub struct TsDemuxer {
    src: Arc<dyn ReadAt>,
    packet_size: usize,
    /// 下一个要读的绝对字节偏移（按整包对齐）。
    file_pos: u64,
    buf: Vec<u8>,
    buf_len: usize,
    buf_off: usize,
    eof: bool,

    programs: HashMap<u16, u16>, // program_number → PMT PID
    tracks: HashMap<u16, TrackState>, // elementary PID → 轨道
    order: Vec<u16>,             // 保持 PID 出现顺序（轨道号要稳定，不能随 HashMap 迭代顺序变）
    next_track_id: u32,
    pmt_pids: Vec<u16>,
    assemblers: HashMap<u16, PesAssembler>,
    queue: VecDeque<Packet>,
    /// seek 之后要"补吐"的第一条（两遍扫描的实现细节，见 `seek_us`）。
    pending: Option<Packet>,
    pcr_pid: Option<u16>,
    seen_pmt: bool,

    source_name: String,
    bitrate: Option<u64>,
    first_pcr: Option<(u64, u64)>, // (字节位置, PCR 27MHz)
    last_pcr: Option<(u64, u64)>,
    total_bytes: Option<u64>,
    /// 是否已经读到源的末尾（与"缓冲里还有没有数据"是两件事）。
    src_eof: bool,
    /// **曾经**读到过源的末尾（`reset()` 不会清掉它）。
    ///
    /// 与 `eof` 分开是必须的：`reset()` 会把游标拨回开头（seek 要用），
    /// 而"整条流扫完了吗"是另一个事实 —— 时长能不能用"最后 PTS − 第一个 PTS"
    /// 来算，取决于后者。
    scanned_to_eof: bool,
    /// AAC 的**跨 PES 残帧**（按 PID 存）。
    ///
    /// 为什么不放在 [`PesAssembler`] 里：PES 装够长度后会被 `remove` 掉再重建，
    /// 残帧挂在它身上会**跟着一起被丢掉** —— 表现就是"每条 PES 的第一帧都少半截"，
    /// 一条十秒的音频只剩几个包。
    adts_carry: HashMap<u16, AdtsCarry>,
    /// PTS 展开用的参考值（最后一条已输出的时间戳）。
    unwrap_ref: i64,
    started_ts: bool,
    first_pts: Option<i64>,
    last_end: i64,
}

impl TsDemuxer {
    pub fn new(src: Arc<dyn ReadAt>, source_name: impl Into<String>) -> Result<Self> {
        let head = src.read_range(0, 4096)?;
        let packet_size = packet_size_of(&head)
            .ok_or_else(|| MediaError::container("不是 MPEG-TS：4 个同步点对不上（188/192/204 都试过）"))?;
        let mut d = Self {
            total_bytes: src.len(),
            src,
            packet_size,
            file_pos: 0,
            buf: Vec::with_capacity(CHUNK),
            buf_len: 0,
            buf_off: 0,
            eof: false,
            src_eof: false,
            scanned_to_eof: false,
            programs: HashMap::new(),
            tracks: HashMap::new(),
            order: Vec::new(),
            next_track_id: 1,
            pmt_pids: Vec::new(),
            assemblers: HashMap::new(),
            adts_carry: HashMap::new(),
            queue: VecDeque::new(),
            pending: None,
            pcr_pid: None,
            seen_pmt: false,
            source_name: source_name.into(),
            bitrate: None,
            first_pcr: None,
            last_pcr: None,
            unwrap_ref: 0,
            started_ts: false,
            first_pts: None,
            last_end: 0,
        };
        // 先扫一段把轨道信息建立起来，再回到开头。
        //
        // 为什么必须这么做：[`Demuxer::info`] 的契约是"**建立之后不可变**"
        // —— 合并器要先读它才知道该写几条轨、怎么写头。如果 open 之后交出的
        // 是一份空轨道表，合并器只能写出一条空的 PAT/PMT，产物是废的。
        // TS 没有索引，轨道信息只能从 PAT/PMT 里扫出来（在流的头部），
        // 所以这一步是"打开"的固有代价。
        d.scan_head()?;
        d.reset();
        Ok(d)
    }

    /// 扫到"轨道信息够用"为止（有界：最多 [`PROBE_PACKET_BUDGET`] 个包）。
    ///
    /// 例外：**源比 [`PROBE_SCAN_BYTES_MAX`] 还小就直接扫到底**。这样时长能用
    /// "最后时间戳 − 第一个时间戳"精确算出来，而代价只是把本来就有权读的
    /// 几 MB 读完；大文件才退回"按码率外推"（有误差，但探测不能为了时长
    /// 去读几个 GB）。
    fn scan_head(&mut self) -> Result<()> {
        let small_source = matches!(self.total_bytes, Some(l) if l <= PROBE_SCAN_BYTES_MAX);
        let max_packets = if small_source {
            // 扫到底，但仍然有硬上限（防止"长度撒谎"的源把我们拖住）
            (PROBE_SCAN_BYTES_MAX / self.packet_size as u64 + 1) as usize
        } else {
            PROBE_PACKET_BUDGET
        };
        for _ in 0..max_packets {
            if self.eof {
                break;
            }
            self.pump_one()?;
            if self.seen_pmt {
                // 探测不关心数据本身：攒够轨道信息就把队列清掉，
                // 否则整条流会在内存里堆出几千个 Packet
                self.queue.clear();
                if self.tracks_ready() && !small_source {
                    break;
                }
            }
        }
        Ok(())
    }

    pub fn packet_size(&self) -> usize {
        self.packet_size
    }

    /// PCR 所在 PID（诊断用：宿主界面可以显示"这条流的时钟挂在哪条轨上"）。
    pub fn pcr_pid(&self) -> Option<u16> {
        self.pcr_pid
    }

    /// 是否已经读完（调用方据此判断"到头了"而不是"暂时没数据"）。
    pub fn eof(&self) -> bool {
        self.eof && self.queue.is_empty()
    }

    /// 建立轨道的元信息（可能还不完整，`next_packet` 会继续补）。
    pub fn info(&self) -> MediaInfo {
        let mut tracks: Vec<Track> = Vec::new();
        for pid in &self.order {
            let st = &self.tracks[pid];
            let mut t = Track::new(st.id, st.kind, st.codec.clone(), Timebase::MPEG_TS);
            t.extradata = if let Some(asc) = &st.asc { asc.clone() } else { st.extradata.clone() };
            t.width = st.width;
            t.height = st.height;
            t.sample_rate = st.sample_rate;
            t.channels = st.channels;
            t.language = st.language.clone();
            if st.kind == MediaKind::Video {
                t.frame_rate = frame_rate_from_deltas(&st.deltas);
            }
            tracks.push(t);
        }
        // 推翻"帧率靠帧计数推"的兜底：帧太少（刚开播）时算出来的帧率是噪声
        for t in &mut tracks {
            if let Some((n, d)) = t.frame_rate {
                if n == 0 || d == 0 || (n as f64 / d as f64) > 240.0 {
                    t.frame_rate = None;
                }
            }
        }
        MediaInfo {
            container: "mpeg-ts".into(),
            duration_us: self.duration_us(),
            bitrate: self.bitrate,
            tracks,
            source: self.source_name.clone(),
        }
    }

    /// 时长（微秒）。
    ///
    /// 优先用"整条扫完后的最后 PTS"，其次用 PCR 外推（PCR 是 TS 里唯一
    /// 自带墙钟含义的字段，拿它算平均码率再乘总字节数是标准做法），
    /// 都没有就返回 `None`（直播流本来就该没有时长）。
    ///
    /// 判据用的是 `scanned_to_eof`（**曾经读到过源的末尾**）而不是 `eof`
    /// （当前游标在末尾）：`reset()` 会把游标拨回开头，如果拿 `eof` 当判据，
    /// "扫描过整条流 → 回到开头 → 问时长"就会退化成 PCR 外推，
    /// 实测把一条 10 秒的流报成 18 秒。
    pub fn duration_us(&self) -> Option<i64> {
        if let (Some(first), true) = (self.first_pts, self.scanned_to_eof) {
            if self.last_end > first {
                return Some(Timebase::MPEG_TS.ticks_to_us(self.last_end - first));
            }
        }
        let (pos1, pcr1) = self.first_pcr?;
        let (pos2, pcr2) = self.last_pcr?;
        if pcr2 <= pcr1 || pos2 <= pos1 {
            return None;
        }
        // 27MHz 的 PCR → bit/s。窗口太短时**不估**：宁可报"未知"，
        // 也不要报一个误差两倍的假时长（宿主界面会拿它画进度条）
        let secs = (pcr2 - pcr1) as f64 / 27_000_000.0;
        if secs < 0.5 {
            return None;
        }
        let bitrate = self.bitrate?;
        if bitrate == 0 {
            return None;
        }
        let total = self.total_bytes?;
        if total <= pos1 {
            return None;
        }
        // 剩下没扫到的字节数按实测码率折算成时间 = 整条流大致有多长。
        //
        // **不要再加 `first_pts`**：媒体时间轴不保证从 0 开始
        // （真实样例：Apple 的 TS 分片第一条 PTS 就是 10 秒），
        // 加了会把 10 秒的流报成 18 秒。
        let remaining_secs = (total - pos1) as f64 * 8.0 / bitrate as f64;
        Some((remaining_secs * 1_000_000.0) as i64)
    }

    /// 平均码率（bit/s），只在拿到两次 PCR 后才有值。
    pub fn bitrate(&self) -> Option<u64> {
        self.bitrate
    }

    /// 探测：只扫有限的包，给出"这是什么 + 有哪些轨 + 分辨率/帧率"。
    ///
    /// `src` 要 `Arc` 是因为解析器需要**长期持有**它（`next_packet` 是拉取式的）。
    pub fn probe_reader(src: Arc<dyn ReadAt>, source: &str) -> Result<(ContainerFormat, MediaInfo)> {
        // `new` 已经扫过一段把轨道建立起来了（否则 `info()` 会交出空轨道表），
        // 这里只需要读它的结果
        let d = Self::new(src, source)?;
        Ok((ContainerFormat::MpegTs, d.info()))
    }

    /// 所有轨道是否都已经有可报告的参数。
    ///
    /// 为什么不能只等视频：AAC 的采样率/声道数是**从 ADTS 头里现学的**，
    /// 而音频 PES 常常排在视频后面。只等视频就收工的话，探测结果里音频的
    /// `sample_rate` 永远是 0 —— 宿主界面就只能显示"未知采样率"。
    fn tracks_ready(&self) -> bool {
        let mut any = false;
        for st in self.tracks.values() {
            match st.kind {
                MediaKind::Video => {
                    any = true;
                    if st.width == 0 || st.height == 0 || st.frames < 2 {
                        return false;
                    }
                }
                MediaKind::Audio => {
                    any = true;
                    if st.sample_rate == 0 {
                        return false;
                    }
                }
                _ => {}
            }
        }
        any
    }

    /// 一直读到末尾（本地文件算精确时长用；网络流别调它）。
    pub fn scan_to_eof(&mut self) -> Result<()> {
        while !self.eof {
            self.pump_one()?;
            self.queue.clear();
        }
        Ok(())
    }

    /// 取下一个采样；`Ok(None)` = 真的到头了。
    pub fn next_packet(&mut self) -> Result<Option<Packet>> {
        if let Some(p) = self.pending.take() {
            return Ok(Some(p));
        }
        loop {
            if let Some(p) = self.queue.pop_front() {
                return Ok(Some(p));
            }
            if self.eof {
                return Ok(None);
            }
            self.pump_one()?;
        }
    }

    /// 回到流的开头（清空 PAT/PMT 之外的所有状态）。
    ///
    /// `seek` 要靠它做"两遍扫描"：TS 没有索引，只能重头扫到目标关键帧。
    pub fn reset(&mut self) {
        self.file_pos = 0;
        self.buf.clear();
        self.buf_len = 0;
        self.buf_off = 0;
        self.eof = false;
        self.src_eof = false;
        self.queue.clear();
        self.pending = None;
        self.assemblers.clear();
        // **ADTS 残帧也要清**：它记录着"上一段没拼完的帧从哪个时间戳开始"，
        // 而 seek 之后的那条 PES 与 seek 之前的残帧根本不是连续的字节。
        // 不清的症状：seek 之后第一条音频 PES 会沿用旧帧的 PTS，
        // 整条音轨一下子跳到别处（实测跳到 19.5 秒，而视频在 14.8 秒）。
        self.adts_carry.clear();
        // 轨道表与 PAT/PMT 保留：它们的 PID 不会因为 seek 而变，
        // 重扫一遍反而要等 PAT 重复出现（真实流里可能几十秒）
        for st in self.tracks.values_mut() {
            st.frames = 0;
            st.first_pts = None;
            st.last_end = 0;
            // 清掉"上一条 PTS"：复位后第一条 PES 与复位前那条之间的间隔是
            // 一次跳变（seek 的产物），不能当帧率样本
            st.last_dts = None;
        }
        self.started_ts = false;
        self.unwrap_ref = 0;
    }

    /// 跳到 `pts_us` 之前最近的关键帧。
    ///
    /// **这是 O(n) 的两遍扫描**（TS 没有索引，只能这样）。
    /// 对本地文件与"BT 已下完整段"的场景可接受；直播/流式下别调它。
    /// 返回实际落点（微秒）。
    ///
    /// ## 只认**视频**关键帧
    ///
    /// 音频在"能不能独立解码"这个意义上**每条都是关键帧**，拿它当落点会让视频
    /// 停在原处：实测 seek 到 15 秒后音频从 15.06 秒开始吐，而视频还在 10.13 秒处
    /// —— 产物里视频时间戳被整段压成同一刻（播放器上就是"卡在第一帧"）。
    /// 没有视频轨时才退化成"按任意轨定位"。
    pub fn seek_us(&mut self, pts_us: i64) -> Result<i64> {
        let target = Timebase::MPEG_TS.us_to_ticks(pts_us);
        let primary: Vec<u32> = self
            .tracks
            .values()
            .filter(|st| st.kind == MediaKind::Video)
            .map(|st| st.id)
            .collect();
        let is_primary = |id: u32| primary.is_empty() || primary.contains(&id);

        // 第一遍：找到 <= target 的最后一个关键帧
        let mut best: Option<i64> = None;
        while let Some(p) = self.next_packet()? {
            if p.keyframe && is_primary(p.track_id) && p.pts <= target {
                best = Some(p.pts);
            }
            if p.pts > target {
                break;
            }
        }
        let Some(best) = best else {
            self.reset();
            return Ok(self
                .first_pts
                .map(|p| Timebase::MPEG_TS.ticks_to_us(p))
                .unwrap_or(0));
        };
        // 第二遍：重头扫到那个关键帧，把它挂成 pending
        self.reset();
        while let Some(p) = self.next_packet()? {
            if p.pts >= best && p.keyframe && is_primary(p.track_id) {
                self.pending = Some(p);
                break;
            }
        }
        // 落点之前读出来的包**留在队列里也不能要**：它们属于 seek 之前的时间段，
        // 混进输出会让时间戳倒流（封装器只能报错，或者把它们全压成同一刻）
        self.queue.clear();
        Ok(Timebase::MPEG_TS.ticks_to_us(best))
    }

    /// 读一个 TS 包并处理。返回是否真的读到了一个**有效**包。
    ///
    /// 注意"读到了但没有载荷"（坏同步字、纯 adaptation field）返回 `Ok(true)`：
    /// 那不是"流到头了"，只是这个包没用，继续往下扫就对了。
    fn pump_one(&mut self) -> Result<bool> {
        let Some(pkt) = self.next_ts_packet()? else {
            // 两种"读不到包"必须分清：
            // - 缓冲里还有**整包** → 刚才那个包丢了同步字（丢包、坏字节），
            //   继续扫后面的包就对了，绝不能判成"流到头"；
            // - 缓冲里只剩零头、源里也没有了 → 真的到头了。
            //
            // 早期版本在这里去问源"还有没有字节"，只要源读空就判 EOF ——
            // 于是"整个文件已经在缓冲里 + 中间坏了一个同步字"会被误判成结束，
            // 整条流一个包都出不来。
            if self.buf_off + self.packet_size <= self.buf_len {
                return Ok(true);
            }
            self.eof = true;
            // "读到过末尾"要单独记：`reset()` 会把游标拨回开头，但
            // "整条流扫完了"这件事不会因此改变（时长计算依赖它）
            self.scanned_to_eof = true;
            self.finish_assemblers();
            return Ok(false);
        };
        self.handle_ts_packet(pkt)?;
        Ok(true)
    }

    /// 把缓冲补齐到"至少有一个整包"。
    ///
    /// 不变量：`buf.len() == buf_len`，缓冲里始终是源上**连续**的一段字节
    /// （所以某个包在文件里的绝对位置 = `file_pos - buf_len + 包内偏移`）。
    fn refill(&mut self) -> Result<()> {
        if self.buf_off + self.packet_size <= self.buf_len {
            return Ok(());
        }
        // 把已经消费掉的零头丢掉，再从源里接着补
        if self.buf_off > 0 {
            self.buf.drain(..self.buf_off);
            self.buf_len = self.buf.len();
            self.buf_off = 0;
        }
        let mut tmp = vec![0u8; CHUNK];
        while self.buf_len < self.packet_size && !self.src_eof {
            let got = self.src.read_at(self.file_pos, &mut tmp)?;
            if got == 0 {
                // 源读到 0 = 真的没有更多字节了（不是"暂时没有"：
                // ReadAt 的约定是"等不到就阻塞或报 NeedMore"）
                self.src_eof = true;
                break;
            }
            self.file_pos += got as u64;
            self.buf.extend_from_slice(&tmp[..got]);
            self.buf_len = self.buf.len();
        }
        Ok(())
    }

    /// 取出下一个 TS 包的 188 字节内容（**已按包长剥掉前缀/后缀**）。
    fn next_ts_packet(&mut self) -> Result<Option<Vec<u8>>> {
        self.refill()?;
        if self.buf_off + self.packet_size > self.buf_len {
            return Ok(None);
        }
        let base = if self.packet_size == 192 { 4 } else { 0 };
        let start = self.buf_off + base;
        self.buf_off += self.packet_size;
        if self.buf[start] != SYNC_BYTE {
            // 丢同步：直接丢掉这个包继续扫（后面的包多是好的）。
            // 真实的流会丢包/插广告，为了一个坏包就整条失败等于"什么都不给放"。
            return Ok(None);
        }
        Ok(Some(self.buf[start..start + TS_PACKET_LEN].to_vec()))
    }

    fn handle_ts_packet(&mut self, pkt: Vec<u8>) -> Result<()> {
        let pusi = pkt[1] & 0x40 != 0;
        let pid = (((pkt[1] & 0x1F) as u16) << 8) | pkt[2] as u16;
        let afc = (pkt[3] >> 4) & 0x03;
        let mut payload_at = 4usize;
        let mut random_access = false;

        if afc == 0 {
            return Ok(()); // 无效的 afc，规范上不允许；跳过
        }
        if afc >= 2 {
            // 有 adaptation field
            let af_len = pkt[4] as usize;
            if af_len > 0 && af_len + 5 <= TS_PACKET_LEN {
                let flags = pkt[5];
                if flags & 0x10 != 0 && af_len >= 7 {
                    // PCR：33 位基值 + 6 位保留 + 9 位扩展 = 42 位 @27MHz
                    let p = &pkt[6..12];
                    let base = ((p[0] as u64) << 25)
                        | ((p[1] as u64) << 17)
                        | ((p[2] as u64) << 9)
                        | ((p[3] as u64) << 1)
                        | ((p[4] as u64) >> 7);
                    let ext = (((p[4] as u64) & 0x01) << 8) | p[5] as u64;
                    let pcr = base * 300 + ext;
                    // 包在文件里的绝对位置（refill 之后 file_pos 指向"已读进缓冲的末尾"）
                    let pos = self
                        .file_pos
                        .saturating_sub(self.buf_len as u64)
                        .saturating_add(self.buf_off as u64);
                    if self.first_pcr.is_none() {
                        self.first_pcr = Some((pos, pcr));
                    }
                    self.last_pcr = Some((pos, pcr));
                }
                random_access = flags & 0x40 != 0;
            }
            payload_at = 5 + af_len;
        }
        if afc == 2 {
            return Ok(()); // 只有 adaptation field，没有载荷
        }
        if payload_at >= TS_PACKET_LEN {
            return Ok(());
        }

        // PAT
        if pid == 0 {
            self.parse_pat(&pkt[payload_at..]);
            return Ok(());
        }
        // 我们关心的 PMT
        if self.pmt_pids.contains(&pid) {
            self.parse_pmt(&pkt[payload_at..]);
            return Ok(());
        }
        if !self.tracks.contains_key(&pid) {
            return Ok(());
        }
        let payload = pkt[payload_at..].to_vec();
        self.push_es(pid, pusi, payload, random_access);
        Ok(())
    }

    /// PAT：`program_number → PMT PID` 的映射表。
    fn parse_pat(&mut self, payload: &[u8]) {
        if payload.is_empty() {
            return;
        }
        let pointer = payload[0] as usize;
        let body = &payload[1 + pointer..];
        if body.len() < 8 || body[0] != 0x00 {
            return; // 不是 PAT
        }
        let section_len = ((((body[1] & 0x0F) as usize) << 8) | body[2] as usize).min(body.len() - 3);
        let entries = &body[8..3 + section_len.saturating_sub(4)];
        let mut i = 0;
        while i + 4 <= entries.len() {
            let program_number = ((entries[i] as u16) << 8) | entries[i + 1] as u16;
            let pid = (((entries[i + 2] & 0x1F) as u16) << 8) | entries[i + 3] as u16;
            if program_number != 0 {
                self.programs.insert(program_number, pid);
                if !self.pmt_pids.contains(&pid) {
                    self.pmt_pids.push(pid);
                }
            }
            i += 4;
        }
    }

    /// PMT：这张表决定了"这个节目里有哪几条基本流、分别是什么编码"。
    fn parse_pmt(&mut self, payload: &[u8]) {
        if payload.is_empty() {
            return;
        }
        let pointer = payload[0] as usize;
        let body = &payload[1 + pointer..];
        if body.len() < 12 || body[0] != 0x02 {
            return;
        }
        let section_len = ((((body[1] & 0x0F) as usize) << 8) | body[2] as usize).min(body.len() - 3);
        let pcr_pid = (((body[8] & 0x1F) as u16) << 8) | body[9] as u16;
        self.pcr_pid = Some(pcr_pid);
        let program_info_len = (((body[10] & 0x0F) as usize) << 8) | body[11] as usize;
        let mut i = 12 + program_info_len;
        let end = 3 + section_len.saturating_sub(4);
        while i + 5 <= end.min(body.len()) {
            let stream_type = body[i];
            let pid = (((body[i + 1] & 0x1F) as u16) << 8) | body[i + 2] as u16;
            let es_info_len = (((body[i + 3] & 0x0F) as usize) << 8) | body[i + 4] as usize;
            let desc_start = i + 5;
            let desc_end = (desc_start + es_info_len).min(body.len());
            let language = find_language_descriptor(&body[desc_start..desc_end]);
            self.add_track(stream_type, pid, language);
            i = desc_start + es_info_len;
        }
        self.seen_pmt = true;
    }

    fn add_track(&mut self, stream_type: u8, pid: u16, language: Option<String>) {
        if self.tracks.contains_key(&pid) {
            return;
        }
        let codec = CodecId::from_ts_stream_type(stream_type);
        // 认得出是视频/音频就归类，其余（含私有流 0x06）统一算"其它"：
        // 引擎不猜私有流里装的是什么，宿主界面按"其它"显示
        let kind = if codec.is_video() {
            MediaKind::Video
        } else if codec.is_audio() {
            MediaKind::Audio
        } else {
            MediaKind::Other
        };
        let id = self.next_track_id;
        self.next_track_id += 1;
        self.order.push(pid);
        self.tracks.insert(
            pid,
            TrackState {
                id,
                codec,
                kind,
                extradata: Vec::new(),
                asc: None,
                sample_rate: 0,
                channels: 0,
                width: 0,
                height: 0,
                language,
                frames: 0,
                deltas: VecDeque::new(),
                last_dts: None,
                first_pts: None,
                last_end: 0,
            },
        );
    }

    fn push_es(&mut self, pid: u16, pusi: bool, payload: Vec<u8>, random_access: bool) {
        if pusi {
            // 上一个 PES 到此为止
            let pending = self.assemblers.get(&pid).map(|a| a.started && !a.buf.is_empty()).unwrap_or(false);
            if pending {
                if let Some(done) = self.assemblers.remove(&pid) {
                    self.flush_pes(pid, done);
                }
            }
            // 解析新的 PES 头
            if payload.len() < 9 || payload[0] != 0x00 || payload[1] != 0x00 || payload[2] != 0x01 {
                // 不是 PES 起点（可能是 section 数据），丢掉直到下一个 PUSI
                if let Some(asm) = self.assemblers.get_mut(&pid) {
                    asm.buf.clear();
                    asm.started = false;
                }
                return;
            }
            let pes_len = ((payload[4] as usize) << 8) | payload[5] as usize;
            let pts_dts_flags = (payload[7] >> 6) & 0x03;
            let header_data_len = payload[8] as usize;
            let mut pts = None;
            let mut dts = None;
            if pts_dts_flags >= 2 && payload.len() >= 14 && header_data_len >= 5 {
                pts = Some(read_timestamp(&payload[9..14]));
            }
            if pts_dts_flags == 3 && payload.len() >= 19 && header_data_len >= 10 {
                dts = Some(read_timestamp(&payload[14..19]));
            }
            let es_start = 9 + header_data_len;
            if es_start > payload.len() {
                if let Some(asm) = self.assemblers.get_mut(&pid) {
                    asm.buf.clear();
                    asm.started = false;
                }
                return;
            }
            let asm = self.assemblers.entry(pid).or_default();
            asm.buf.clear();
            asm.buf.extend_from_slice(&payload[es_start..]);
            asm.es_start = 0;
            asm.expected = if pes_len == 0 { 0 } else { pes_len.saturating_sub(3 + header_data_len) };
            asm.pts = pts;
            asm.dts = dts;
            asm.started = true;
            asm.random_access = random_access;
        } else {
            let asm = self.assemblers.entry(pid).or_default();
            if !asm.started {
                return; // 从中间接进来的半截数据，丢掉
            }
            asm.buf.extend_from_slice(&payload);
        }

        // 定长 PES 够了就立刻吐出去（视频的 pes_len 通常是 0，走 PUSI 那条路）
        let ready = {
            let asm = self.assemblers.get(&pid).unwrap();
            asm.started && asm.expected > 0 && asm.buf.len() >= asm.expected
        };
        if ready {
            if let Some(asm) = self.assemblers.remove(&pid) {
                self.flush_pes(pid, asm);
                // remove 之后留一个空的占位，免得 `get_mut` 那条路径要重新判断
                self.assemblers.entry(pid).or_default();
            }
        }
    }

    /// 一个完整的 PES 到手：按轨道类型拆成若干 [`Packet`]。
    fn flush_pes(&mut self, pid: u16, asm: PesAssembler) {
        if !asm.started || asm.buf.is_empty() {
            return;
        }
        let Some(state) = self.tracks.get(&pid).cloned() else {
            return;
        };
        let raw_pts = asm.pts.unwrap_or(self.unwrap_ref);
        let pts = if self.started_ts { unwrap_pts(raw_pts, self.unwrap_ref) } else { raw_pts };
        self.started_ts = true;
        self.unwrap_ref = pts + 90_000; // 参考值往前推一秒，避免相邻时间戳被判定成回绕
        let dts = asm.dts.map(|d| unwrap_pts(d, self.unwrap_ref - 90_000)).unwrap_or(pts);

        if self.first_pts.is_none() {
            self.first_pts = Some(pts);
        }

        match state.kind {
            MediaKind::Video => {
                let keyframe = {
                    let nals = codec::split_annexb(&asm.buf);
                    let hevc = state.codec == CodecId::H265;
                    let is_ra = nals.iter().any(|n| codec::is_random_access_nal(hevc, n));
                    if is_ra || asm.random_access {
                        // 顺手把参数集攒进 extradata（TS 里参数集是跟着码流走的）
                        self.harvest_parameter_sets(pid, &nals);
                    }
                    is_ra || asm.random_access
                };
                if let Some(st) = self.tracks.get_mut(&pid) {
                    st.frames += 1;
                    // 帧率靠"最近若干条 PES 的解码间隔中位数"估（见 `deltas` 的注释）
                    if let Some(prev) = st.last_dts {
                        let d = dts - prev;
                        if d > 0 && d < 900_000 {
                            st.deltas.push_back(d);
                            while st.deltas.len() > 64 {
                                st.deltas.pop_front();
                            }
                        }
                    }
                    st.last_dts = Some(dts);
                    if st.first_pts.is_none() {
                        st.first_pts = Some(pts);
                    }
                    st.last_end = st.last_end.max(pts);
                }
                let mut p = Packet::new(state.id, pts, dts, 0, asm.buf);
                p.keyframe = keyframe;
                self.queue.push_back(p);
            }
            MediaKind::Audio => {
                if state.codec == CodecId::Aac {
                    self.split_adts(pid, state.id, pts, &asm.buf);
                } else {
                    // 非 AAC 音频（MP3 / AC3 / EAC3）：一条 PES 就是一条采样 ——
                    // 这些编码自己带帧长，不需要像 ADTS 那样再切一刀，
                    // 下游解码器也收得下"一个包里有多帧"。
                    //
                    // MP3 的参数在 TS 里没有描述符可读（不像 AAC 有 ADTS 头），
                    // 只在每一帧的帧头里。顺手解第一帧填回轨道信息，
                    // 否则 `probe` 会报 0 Hz / 0 声道（界面上就是"未知"）。
                    if state.codec == CodecId::Mp3 && state.sample_rate == 0 {
                        if let Some(h) = crate::mpeg_audio::first_frame_header(&asm.buf) {
                            if let Some(st) = self.tracks.get_mut(&pid) {
                                st.sample_rate = h.sample_rate;
                                st.channels = h.channels as u16;
                            }
                        }
                    }
                    if let Some(st) = self.tracks.get_mut(&pid) {
                        st.frames += 1;
                        st.first_pts.get_or_insert(pts);
                        st.last_end = st.last_end.max(pts);
                    }
                    let mut p = Packet::new(state.id, pts, dts, 0, asm.buf);
                    p.keyframe = true;
                    self.queue.push_back(p);
                }
            }
            _ => {
                if let Some(st) = self.tracks.get_mut(&pid) {
                    st.last_end = st.last_end.max(pts);
                }
                let mut p = Packet::new(state.id, pts, dts, 0, asm.buf);
                p.keyframe = true;
                self.queue.push_back(p);
            }
        }

        let end = self.queue.back().map(|p| p.end()).unwrap_or(pts);
        self.last_end = self.last_end.max(end.max(pts));
        self.refresh_bitrate();
    }

    /// 用两次 PCR 之间的字节数与时间差算平均码率（TS 里唯一可靠的总长依据）。
    ///
    /// 两个要点：
    ///
    /// 1. **窗口太短就不算**。PCR 每 100ms 才来一个，拿相邻两个 PCR 算出来的
    ///    码率噪声极大（实测把 234kbps 的流算成 2.4Mbps，时长因此虚增到近两倍）。
    ///    要求窗口 ≥ 0.5 秒才认这个数。
    /// 2. **允许反复更新**。扫描是渐进的，窗口会越来越长、估计越来越准，
    ///    算一次就锁死（早期版本就是这样）等于永远用最差的那个估计。
    fn refresh_bitrate(&mut self) {
        let (Some((pos1, pcr1)), Some((pos2, pcr2))) = (self.first_pcr, self.last_pcr) else {
            return;
        };
        if pos2 <= pos1 || pcr2 <= pcr1 {
            return;
        }
        let secs = (pcr2 - pcr1) as f64 / 27_000_000.0;
        if secs < 0.5 {
            return;
        }
        let bits = (pos2 - pos1) as f64 * 8.0;
        if secs > 0.0 {
            self.bitrate = Some((bits / secs) as u64);
        }
    }

    /// 一个 PES 里的多个 ADTS 帧要拆开，各自带自己的时间戳。
    ///
    /// 两个真实世界的事实决定了这里的写法：
    ///
    /// 1. **编码器切 PES 时不管 ADTS 帧边界** —— 一帧被切成两半是常态。
    ///    残下的字节必须留到下一段前面续上（见 [`TsDemuxer::adts_carry`]），
    ///    否则每条 PES 都要丢一帧，一条十秒的音频只剩几个包。
    /// 2. **流可能从中间接进来**（丢包、seek 到一半、录制被打断），
    ///    这时开头几个字节是半个帧，帧头对不上。所以要**找同步字重新对齐**，
    ///    而不是一遇到对不上的字节就把整段丢掉。
    ///
    /// 时间戳：第 k 帧的 pts = 本段 PTS + k × 每帧刻度（1024 / 采样率）。
    fn split_adts(&mut self, pid: u16, track_id: u32, base_pts: i64, data: &[u8]) {
        // 时间戳基准：PES 头里的 PTS **只在它不落后于已吐出的帧时**才采信。
        //
        // 为什么：一个 PES 里塞多少帧是编码器定的，而帧时长是"1024 / 采样率"
        // 四舍五入来的；真实流里 PES 声明的 PTS 偶尔与帧序列对不上
        // （实测 Apple 的分片流就会出现），采信它就会得到"时间戳倒流"，
        // 下游的交织器/封装器只能报错 —— 而音频时间戳本该是**连续推进**的。
        let base_pts = match self.tracks.get(&pid).map(|st| st.last_end) {
            Some(end) if end > base_pts => end,
            _ => base_pts,
        };
        // 有残帧时，帧是从**上一条 PES**开始的：时间戳基准要用那一条的
        let carry = self.adts_carry.remove(&pid);
        let (mut buf, base_pts, mut idx) = match carry {
            Some(c) => (c.data, c.pts, c.frames_before),
            None => (Vec::new(), base_pts, 0),
        };
        buf.extend_from_slice(data);

        let mut off = 0usize;
        let mut ticks = 0i64;
        let mut resynced = false;
        while off + 7 <= buf.len() {
            if !is_adts_sync(&buf[off..]) {
                match find_adts_sync(&buf[off..]) {
                    Some(k) => {
                        off += k;
                        if !resynced {
                            resynced = true;
                            // 丢掉的是"上一帧的尾巴"，所以这一条 PES 的第一帧
                            // 在序号上要往后挪一格（否则整条音轨的时间戳会逐段提前）
                            idx += 1;
                        }
                        continue;
                    }
                    None => {
                        off = buf.len();
                        break;
                    }
                }
            }
            let Some(f) = parse_adts(&buf[off..]) else {
                // 同步字在但头不合法（多半是数据里的巧合），往后挪一格继续找
                off += 1;
                continue;
            };
            if off + f.frame_len > buf.len() {
                break; // 尾巴装不下整帧：留给下一次续上
            }
            ticks = (1024i64 * 90_000) / f.sample_rate.max(1) as i64;
            if idx == 0 {
                if let Some(st) = self.tracks.get_mut(&pid) {
                    if st.asc.is_none() {
                        st.asc = Some(adts_to_asc(&f));
                    }
                    st.sample_rate = f.sample_rate;
                    st.channels = f.channels;
                    if st.first_pts.is_none() {
                        st.first_pts = Some(base_pts);
                    }
                }
            }
            let pts = base_pts + idx * ticks;
            let mut p = Packet::new(
                track_id,
                pts,
                pts,
                ticks as u32,
                buf[off..off + f.frame_len].to_vec(),
            );
            p.keyframe = true;
            self.queue.push_back(p);
            self.last_end = self.last_end.max(pts + ticks);
            off += f.frame_len;
            idx += 1;
        }

        if idx > 0 {
            if let Some(st) = self.tracks.get_mut(&pid) {
                st.frames += idx as u64;
                st.last_end = st.last_end.max(base_pts + idx * ticks);
            }
        }
        // 残帧留到下一段，但**只续"看起来真是半截帧"的那部分**。
        //
        // 判据是"这段尾巴能解析出一个合法的 ADTS 头"：真实流里 PES 尾部
        // 可能是编码器填的 0xFF（不少编码器这么干），把填充当残帧续下去
        // 会让下一段整体错位一格 —— 实测后果是"整条音轨一个包都出不来"，
        // 比丢掉半帧严重得多。代价是"PES 恰好切在帧头前 1~6 字节"时会丢一帧，
        // 这种情形罕见且有界。
        if off < buf.len() && buf.len() - off <= 8192 && parse_adts(&buf[off..]).is_some() {
            self.adts_carry.insert(
                pid,
                AdtsCarry { data: buf[off..].to_vec(), pts: base_pts, frames_before: idx },
            );
        }
    }

    /// 从视频 NAL 里把 SPS/PPS 存下来当 extradata（TS 没有 `avcC`）。
    ///
    /// 顺手解一次分辨率：**这一步不能提前返回**。早期版本"刚加了参数集就 return"，
    /// 于是第一次关键帧拿不到分辨率，得等下一个关键帧 —— 而关键帧可能隔 8 秒
    /// （实测一个真实文件的第二个关键帧在 8.3 秒处），探测要么白扫几千个包，
    /// 要么因为预算用尽而报"分辨率未知"。
    fn harvest_parameter_sets(&mut self, pid: u16, nals: &[&[u8]]) {
        let Some(st) = self.tracks.get_mut(&pid) else {
            return;
        };
        let hevc = st.codec == CodecId::H265;
        for n in nals {
            let want = if hevc {
                matches!(codec::h265_nal_type(n), Some(32) | Some(33) | Some(34))
            } else {
                matches!(codec::h264_nal_type(n), Some(7) | Some(8))
            };
            if !want {
                continue;
            }
            if st.extradata.is_empty() || !contains_subslice(&st.extradata, n) {
                st.extradata.extend_from_slice(&[0, 0, 0, 1]);
                st.extradata.extend_from_slice(n);
            }
        }
        // 把分辨率解出来（还没有的时候才做，SPS 一般不会变）
        if st.kind == MediaKind::Video && st.width == 0 && st.extradata.len() >= 5 {
            for n in codec::split_annexb(&st.extradata) {
                let sps = if hevc { codec::parse_h265_sps(n) } else { codec::parse_h264_sps(n) };
                if let Some(s) = sps {
                    st.width = s.width;
                    st.height = s.height;
                    break;
                }
            }
        }
    }

    fn finish_assemblers(&mut self) {
        let pids: Vec<u16> = self.assemblers.keys().copied().collect();
        for pid in pids {
            if let Some(asm) = self.assemblers.remove(&pid) {
                self.flush_pes(pid, asm);
            }
        }
    }
}

/// 把 TS 解封装器接到统一接口上。
///
/// 这里全是**转发**，一行逻辑都不加：上层（流层/合并层/CLI）只认 [`Demuxer`]，
/// 换容器时它们不该改一行代码；而 TS 自己的扩展能力（`packet_size`/`pcr_pid`/
/// `scan_to_eof`）留在固有方法上，谁需要精确控制谁直接拿具体类型用。
impl crate::demux::Demuxer for TsDemuxer {
    fn info(&self) -> MediaInfo {
        TsDemuxer::info(self)
    }

    fn next_packet(&mut self) -> Result<Option<Packet>> {
        TsDemuxer::next_packet(self)
    }

    fn seek(&mut self, pts_us: i64) -> Result<i64> {
        self.seek_us(pts_us)
    }

    fn duration_us(&self) -> Option<i64> {
        TsDemuxer::duration_us(self)
    }

    fn container(&self) -> &'static str {
        // 与 `ContainerFormat::MpegTs.name()` 必须一致：宿主拿这个字符串做分发
        ContainerFormat::MpegTs.name()
    }
}

/// 从视频 PES 的间隔里取中位数当每帧时长，再换算成帧率。
///
/// 为什么不用"帧数 ÷ 跨度"：
/// - 真实流的首尾间隔常常不齐，估出来会偏（实测把 29.97fps 算成 31.2fps）；
/// - 更重要的是**同一段数据被扫描两次时帧数会翻倍、跨度不变**，
///   估出来的帧率直接错一倍 —— 而"打开时先扫一段建立轨道信息"正好会造成这种重复扫描。
///
/// 中位数对这两件事都免疫。
fn frame_rate_from_deltas(deltas: &VecDeque<i64>) -> Option<(u32, u32)> {
    if deltas.len() < 2 {
        return None;
    }
    let mut v: Vec<i64> = deltas.iter().copied().collect();
    v.sort_unstable();
    let median = v[v.len() / 2];
    if median <= 0 {
        return None;
    }
    let (n, d) = codec::reduce(Timebase::MPEG_TS.ticks_per_second as u64, median as u64)?;
    if n == 0 || d == 0 || (n as f64 / d as f64) > 240.0 {
        return None;
    }
    Some((n, d))
}

/// 有没有 `needle` 这个子串（参数集去重用；参数集通常只有几十字节，朴素匹配足够）。
fn contains_subslice(hay: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || needle.len() > hay.len() {
        return false;
    }
    hay.windows(needle.len()).any(|w| w == needle)
}

/// ISO_639_language_descriptor（tag 0x0A）：3 字节语言码 + 1 字节 audio_type。
fn find_language_descriptor(descs: &[u8]) -> Option<String> {
    let mut i = 0;
    while i + 2 <= descs.len() {
        let tag = descs[i];
        let len = descs[i + 1] as usize;
        if i + 2 + len > descs.len() {
            return None;
        }
        if tag == 0x0A && len >= 3 {
            let lang = &descs[i + 2..i + 5];
            if lang.iter().all(|c| c.is_ascii_alphabetic()) {
                return Some(String::from_utf8_lossy(lang).to_ascii_lowercase());
            }
        }
        i += 2 + len;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use me_core::Cursor;

    /// 极简 TS 构造器：只造我们解析器认得的那几样。
    struct TsBuilder {
        packets: Vec<u8>,
        cc: HashMap<u16, u8>,
    }

    impl TsBuilder {
        fn new() -> Self {
            Self { packets: Vec::new(), cc: HashMap::new() }
        }

        fn cc_of(&mut self, pid: u16) -> u8 {
            let c = self.cc.entry(pid).or_insert(0);
            let v = *c;
            *c = (v + 1) & 0x0F;
            v
        }

        /// 一个带载荷的 TS 包。
        ///
        /// 载荷不足 184 字节时**用 adaptation field 填充**（`afc=3`），而不是
        /// 拿 0xFF 把尾巴补齐 —— 真实编码器就是这么做的，而"用垃圾字节填满"
        /// 会把解析器（尤其是音频拆帧）带到坑里去。
        fn packet(&mut self, pid: u16, pusi: bool, payload: &[u8]) {
            let cc = self.cc_of(pid);
            let mut p = vec![0xFFu8; TS_PACKET_LEN];
            p[0] = SYNC_BYTE;
            p[1] = if pusi { 0x40 } else { 0 } | ((pid >> 8) as u8 & 0x1F);
            p[2] = (pid & 0xFF) as u8;
            let n = payload.len().min(TS_PACKET_LEN - 4);
            let stuffing = TS_PACKET_LEN - 4 - n;
            let (afc, payload_at) = if stuffing > 0 {
                // adaptation field：长度字节 + 填充（长度字段里不含它自己）
                p[3] = 0x30 | cc; // afc = 3
                p[4] = (stuffing - 1) as u8;
                for b in &mut p[5..4 + stuffing] {
                    *b = 0xFF;
                }
                (3u8, 4 + stuffing)
            } else {
                p[3] = 0x10 | cc; // afc = 1
                (1u8, 4usize)
            };
            let _ = afc;
            p[payload_at..payload_at + n].copy_from_slice(&payload[..n]);
            self.packets.extend_from_slice(&p);
        }

        /// 把一段数据切成若干 TS 包发出去。
        fn payload_packets(&mut self, pid: u16, data: &[u8]) {
            let mut off = 0;
            while off < data.len() {
                let n = (data.len() - off).min(TS_PACKET_LEN - 4);
                let pusi = off == 0;
                self.packet(pid, pusi, &data[off..off + n]);
                off += n;
            }
        }

        fn pat(&mut self, pmt_pid: u16) {
            let mut body = vec![0x00u8]; // pointer_field
            let mut section = vec![0x00u8, 0xB0, 0x0D]; // table_id, section_length(占位)
            section.extend_from_slice(&[0x00, 0x01]); // transport_stream_id
            section.extend_from_slice(&[0xC1, 0x00, 0x00]); // version/current, section_number, last
            section.extend_from_slice(&[0x00, 0x01]); // program_number = 1
            section.push(0xE0 | ((pmt_pid >> 8) as u8 & 0x1F));
            section.push((pmt_pid & 0xFF) as u8);
            let sec_len = section.len() - 3 + 4; // + CRC32
            section[1] = 0xB0 | ((sec_len >> 8) as u8 & 0x0F);
            section[2] = (sec_len & 0xFF) as u8;
            section.extend_from_slice(&[0, 0, 0, 0]); // CRC32（我们不解）
            body.extend_from_slice(&section);
            self.payload_packets(0, &body);
        }

        fn pmt(&mut self, pid: u16, pcr_pid: u16, streams: &[(u8, u16, Option<&'static str>)]) {
            let mut body = vec![0x00u8];
            let mut section = vec![0x02u8, 0xB0, 0x00];
            section.extend_from_slice(&[0x00, 0x01]); // program_number
            section.extend_from_slice(&[0xC1, 0x00, 0x00]);
            section.push(0xE0 | ((pcr_pid >> 8) as u8 & 0x1F));
            section.push((pcr_pid & 0xFF) as u8);
            section.extend_from_slice(&[0xF0, 0x00]); // program_info_length = 0
            for (st, epid, lang) in streams {
                section.push(*st);
                section.push(0xE0 | ((epid >> 8) as u8 & 0x1F));
                section.push((epid & 0xFF) as u8);
                if let Some(l) = lang {
                    let mut d = vec![0x0A, 0x04];
                    d.extend_from_slice(l.as_bytes());
                    d.push(0x00);
                    section.push(0xF0 | ((d.len() >> 8) as u8 & 0x0F));
                    section.push((d.len() & 0xFF) as u8);
                    section.extend_from_slice(&d);
                } else {
                    section.extend_from_slice(&[0xF0, 0x00]);
                }
            }
            let sec_len = section.len() - 3 + 4;
            section[1] = 0xB0 | ((sec_len >> 8) as u8 & 0x0F);
            section[2] = (sec_len & 0xFF) as u8;
            section.extend_from_slice(&[0, 0, 0, 0]);
            body.extend_from_slice(&section);
            self.payload_packets(pid, &body);
        }

        fn pes(&mut self, pid: u16, pts: i64, dts: Option<i64>, es: &[u8], stream_id: u8) {
            let mut header = vec![0x00u8, 0x00, 0x01, stream_id];
            let mut opts: Vec<u8> = Vec::new();
            let pts_bytes = encode_ts(pts);
            match dts {
                Some(d) => {
                    opts.push(0xC0); // PTS_DTS_flags = 3
                    opts.push(10);
                    opts.extend_from_slice(&pts_bytes);
                    opts.extend_from_slice(&encode_ts(d));
                }
                None => {
                    opts.push(0x80); // PTS_DTS_flags = 2
                    opts.push(5);
                    opts.extend_from_slice(&pts_bytes);
                }
            }
            let pes_len = 3 + opts.len() + es.len();
            header.push((pes_len >> 8) as u8);
            header.push((pes_len & 0xFF) as u8);
            // PES 头的第 6 字节是 `10` 标记位 + 加扰/优先级等标志（这里全 0）。
            // **这一字节不能省**：省掉之后 PTS 标志与 header_data_length 全部错位一格，
            // 解析器会把 ES 的起点切错，症状是"视频没时间戳、音频一帧都出不来"
            header.push(0x80);
            header.extend_from_slice(&opts);
            header.extend_from_slice(es);
            self.payload_packets(pid, &header);
        }

        fn finish(&self) -> Vec<u8> {
            self.packets.clone()
        }

        /// 造一条"声明长度比实际数据长"的 PES（尾部是填充字节）。
        ///
        /// 真实编码器会这么干（填充 0xFF 到某个对齐边界），解析器必须容忍：
        /// 填充既不能被当成音频数据，也不能把下一条 PES 顶歪。
        fn pes_with_padding(&mut self, pid: u16, pts: i64, es: &[u8], pad: usize) {
            let mut header = vec![0x00u8, 0x00, 0x01, 0xC0];
            let opts = {
                let mut o = vec![0x80u8, 5];
                o.extend_from_slice(&encode_ts(pts));
                o
            };
            // 声明长度 = 3 + opts + es + pad
            let pes_len = 3 + opts.len() + es.len() + pad;
            header.push((pes_len >> 8) as u8);
            header.push((pes_len & 0xFF) as u8);
            header.push(0x80);
            header.extend_from_slice(&opts);
            header.extend_from_slice(es);
            header.extend(std::iter::repeat(0xFF).take(pad));
            self.payload_packets(pid, &header);
        }
    }

    fn encode_ts(v: i64) -> [u8; 5] {
        [
            // b0 = 0b0010_xxx_1，xxx 是 PTS[32..30]。
            // 注意这里必须写成 `0x21 | ((v >> 29) & 0x0E)`：
            // 写成 `0x02 | ((v >> 30) & 0x0E)` 会把 3 个高位比特放到错的位置，
            // 解出来的时间戳会凭空多出 2^30（症状是"时间戳大得离谱"）
            0x21 | ((v >> 29) as u8 & 0x0E),
            (v >> 22) as u8,
            0x01 | ((v >> 14) as u8 & 0xFE),
            (v >> 7) as u8,
            0x01 | ((v << 1) as u8 & 0xFE),
        ]
    }

    fn adts_frame(sample_rate_index: u8, channels: u8, payload_len: usize) -> Vec<u8> {
        let frame_len = 7 + payload_len;
        let mut f = vec![
            0xFF,
            0xF1, // MPEG-4, layer 0, no CRC
            ((1u8) << 6) | (sample_rate_index << 2) | ((channels >> 2) & 0x01),
            (((channels & 0x03) as u8) << 6) | ((frame_len >> 11) as u8 & 0x03),
            ((frame_len >> 3) & 0xFF) as u8,
            (((frame_len & 0x07) as u8) << 5) | 0x1F,
            0xFC,
        ];
        f.extend(std::iter::repeat(0x55).take(payload_len));
        f
    }

    fn h264_annexb_1080p() -> Vec<u8> {
        // SPS（120×68 宏块 + 底部裁 4 行 = 1920x1080） + PPS + IDR 占位
        let mut sps = vec![0x67u8, 0x42, 0x00, 0x1E];
        // 手工位序列：ue(0),ue(0),ue(0),ue(0),ue(1),u(0),ue(119),ue(67),u(1),u(1),u(1),ue(0),ue(0),ue(0),ue(4),u(0)
        // 直接用 codec 测试里的构造思路太长，这里用一个"能让解析器算出 1920x1080 的最短序列"
        // —— 见 codec.rs 的 build_sps_baseline，此处复用它的输出
        sps.clear();
        sps.extend_from_slice(&sps_1080p_bytes());
        let mut out = vec![0, 0, 0, 1];
        out.extend_from_slice(&sps);
        out.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE, 0x3C, 0x80]); // PPS
        out.extend_from_slice(&[0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00]); // IDR
        out
    }

    /// 与 codec.rs 测试里 `build_sps_baseline(120, 68, (0,0,0,4))` 等价的字节序列。
    fn sps_1080p_bytes() -> Vec<u8> {
        // 位写入用**内部函数**而不是闭包：两个闭包同时捕获 `bits` 的可变借用
        // 在 Rust 里是编译错误（借用检查器不允许），改成显式传 `&mut Vec` 最直白
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
        // 用手写位序生成：profile=66, level=30, 然后各 ue
        let mut bits: Vec<u8> = Vec::new();
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
        push(&mut bits, 1, 1); // frame_cropping_flag
        ue(&mut bits, 0);
        ue(&mut bits, 0);
        ue(&mut bits, 0);
        ue(&mut bits, 4);
        push(&mut bits, 0, 1); // vui
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

    fn build_stream() -> Vec<u8> {
        let mut b = TsBuilder::new();
        b.pat(0x100);
        b.pmt(0x100, 0x101, &[(0x1B, 0x101, None), (0x0F, 0x102, Some("eng"))]);
        // 两帧视频 + 一个音频 PES（里面塞 2 个 ADTS 帧，真实流就是这样）
        b.pes(0x101, 90_000, Some(90_000), &h264_annexb_1080p(), 0xE0);
        b.pes(0x101, 93_600, Some(93_600), &h264_annexb_1080p(), 0xE0);
        let mut audio = adts_frame(4, 2, 32); // 44100Hz 立体声
        audio.extend_from_slice(&adts_frame(4, 2, 32));
        b.pes(0x102, 90_000, None, &audio, 0xC0);
        b.finish()
    }

    #[test]
    fn detects_packet_size_188() {
        let data = build_stream();
        assert_eq!(packet_size_of(&data), Some(188));
    }

    #[test]
    fn rejects_data_that_is_not_ts() {
        let junk = vec![0x11u8; 4096];
        assert_eq!(packet_size_of(&junk), None);
        // 单个同步字不算（误判概率 1/256，不能认）
        let mut one = vec![0u8; 4096];
        one[0] = SYNC_BYTE;
        assert_eq!(packet_size_of(&one), None);
    }

    #[test]
    fn demodulates_tracks_and_packets() {
        let data = build_stream();
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
        let mut d = TsDemuxer::new(src, "test.ts").unwrap();
        let mut video = 0;
        let mut audio = 0;
        let mut video_pts = Vec::new();
        let mut audio_durations = Vec::new();
        loop {
            match d.next_packet().unwrap() {
                Some(p) => {
                    let kind = d.tracks.values().find(|t| t.id == p.track_id).map(|t| t.kind);
                    match kind {
                        Some(MediaKind::Video) => {
                            video += 1;
                            video_pts.push(p.pts);
                            assert!(p.keyframe, "IDR 帧必须被标成关键帧");
                        }
                        Some(MediaKind::Audio) => {
                            audio += 1;
                            audio_durations.push(p.duration);
                        }
                        _ => {}
                    }
                }
                None => break,
            }
        }
        assert_eq!(video, 2);
        assert_eq!(audio, 2, "一个 PES 里的 2 个 ADTS 帧必须拆成 2 条");
        assert_eq!(video_pts, vec![90_000, 93_600]);
        // 44100Hz 下 1024 采样 ≈ 2089.8 个 90kHz 刻度
        assert!(audio_durations.iter().all(|&d| d == 2089), "{audio_durations:?}");
    }

    #[test]
    fn audio_frame_timestamps_are_incremented_within_a_pes() {
        let data = build_stream();
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
        let mut d = TsDemuxer::new(src, "t").unwrap();
        let mut audio_pts = Vec::new();
        while let Some(p) = d.next_packet().unwrap() {
            let is_audio = d.tracks.values().any(|t| t.id == p.track_id && t.kind == MediaKind::Audio);
            if is_audio {
                audio_pts.push(p.pts);
            }
        }
        assert_eq!(audio_pts, vec![90_000, 90_000 + 2089]);
    }

    /// 一个 MPEG-1 44.1k / 128 kbps / 立体声的 MP3 帧（帧长 417）。
    fn mp3_frame() -> Vec<u8> {
        let mut v = vec![0xFFu8, 0xFB, 0x90, 0x00];
        v.resize(417, 0);
        v
    }

    /// 造一条"视频 + MP3 音频"的 TS：**一条 PES 一帧**（引擎自己的 TS 封装就是这样）。
    fn build_mp3_stream() -> Vec<u8> {
        let mut b = TsBuilder::new();
        b.pat(0x100);
        // 0x03 / 0x04 = MPEG-1 / MPEG-2 音频；0x1B = H.264
        b.pmt(0x100, 0x101, &[(0x1B, 0x101, None), (0x03, 0x102, None)]);
        b.pes(0x101, 90_000, Some(90_000), &h264_annexb_1080p(), 0xE0);
        b.pes(0x101, 93_600, Some(93_600), &h264_annexb_1080p(), 0xE0);
        b.pes(0x102, 90_000, None, &mp3_frame(), 0xC0);
        b.pes(0x102, 93_840, None, &mp3_frame(), 0xC0);
        b.finish()
    }

    /// TS 里的 MP3 轨必须报出真实的采样率与声道数。
    ///
    /// MP3 在 TS 里**没有描述符可读**（不像 AAC 有 ADTS 头，也不像 H.264 有 SPS），
    /// 参数只存在于每一帧的帧头里。不专门解一下第一帧，`probe` 就永远报 0 Hz / 0 声道，
    /// 界面上那条音轨只能显示"未知"。而且 `tracks_ready()` 又要求音频的采样率非 0，
    /// 于是探测会一直读到文件末尾才收工（大文件上就是"探测很慢"）。
    #[test]
    fn mp3_track_gets_rate_and_channels_from_the_frame_header() {
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(build_mp3_stream()));
        let (_fmt, info) = TsDemuxer::probe_reader(src, "mp3.ts").unwrap();
        let a = info.audio().expect("应认出音频轨");
        assert_eq!(a.codec, CodecId::Mp3);
        assert_eq!(a.sample_rate, 44100, "MP3 的参数只能从帧头里学");
        assert_eq!(a.channels, 2);
        // 一条 PES 一帧 → 两条采样
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(build_mp3_stream()));
        let mut d = TsDemuxer::new(src, "mp3.ts").unwrap();
        let mut audio = 0;
        while let Some(p) = d.next_packet().unwrap() {
            if d.tracks.values().any(|t| t.id == p.track_id && t.kind == MediaKind::Audio) {
                audio += 1;
                assert_eq!(p.data.len(), 417, "载荷是整帧");
            }
        }
        assert_eq!(audio, 2);
    }

    #[test]
    fn probe_reports_resolution_codec_and_language() {
        let data = build_stream();
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
        let (_fmt, info) = TsDemuxer::probe_reader(src, "test.ts").unwrap();
        assert_eq!(info.container, "mpeg-ts");
        let v = info.video().expect("应认出视频轨");
        assert_eq!(v.codec, CodecId::H264);
        assert_eq!((v.width, v.height), (1920, 1080));
        assert!(!v.extradata.is_empty(), "SPS/PPS 必须被攒进 extradata");
        let a = info.audio().expect("应认出音频轨");
        assert_eq!(a.codec, CodecId::Aac);
        assert_eq!(a.sample_rate, 44100);
        assert_eq!(a.channels, 2);
        assert_eq!(a.language.as_deref(), Some("eng"));
        assert!(a.extradata.len() >= 2, "AAC 必须给出 AudioSpecificConfig");
    }

    #[test]
    fn adts_parsing_and_asc() {
        let f = adts_frame(4, 2, 100);
        let a = parse_adts(&f).unwrap();
        assert_eq!(a.frame_len, 107);
        assert_eq!(a.header_len, 7);
        assert_eq!(a.sample_rate, 44100);
        assert_eq!(a.channels, 2);
        let asc = adts_to_asc(&a);
        // AOT=2(LC), sf_index=4, channels=2 → 0x12 0x10
        assert_eq!(asc, vec![0x12, 0x10]);

        assert!(parse_adts(&[0u8; 7]).is_none(), "没有同步字就不是 ADTS");
        assert!(parse_adts(&f[..4]).is_none(), "头不全返回 None");
    }

    #[test]
    fn adts_frame_length_below_header_is_rejected() {
        let mut f = adts_frame(4, 2, 0);
        f[3] = 0; // frame_len = 0
        f[4] = 0;
        f[5] = 0;
        assert!(parse_adts(&f).is_none());
    }

    #[test]
    fn pts_unwrap_handles_wraparound() {
        // 临近回绕：raw 很小，参考值很大 → 应该展开成"继续往前"
        let near_end = PTS_WRAP - 1000;
        let raw = 500i64; // 回绕后的值
        let v = unwrap_pts(raw, near_end);
        assert!(v > near_end, "回绕后必须继续前进，而不是跳回 0：{v}");
        assert_eq!(v, raw + PTS_WRAP);
        // 正常范围内不该被改动
        assert_eq!(unwrap_pts(90_000, 89_000), 90_000);
    }

    #[test]
    fn corrupted_sync_does_not_kill_the_stream() {
        // 把一个包的首字节改坏，后面的包必须还能解出来
        let mut data = build_stream();
        // 第 3 个包（跳过 PAT/PMT 之后的第一批）
        data[188 * 2] = 0x00;
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
        let mut d = TsDemuxer::new(src, "t").unwrap();
        let mut count = 0;
        while let Some(_p) = d.next_packet().unwrap() {
            count += 1;
        }
        assert!(count > 0, "坏掉一个包不该让整条流变成空");
    }

    #[test]
    fn truncation_ends_cleanly_not_with_error() {
        let mut data = build_stream();
        data.truncate(188 * 3 + 40); // 从包中间截断
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
        let mut d = TsDemuxer::new(src, "t").unwrap();
        loop {
            if d.next_packet().unwrap().is_none() {
                break;
            }
        }
        assert!(d.eof());
    }

    #[test]
    fn adts_frame_split_across_pes_is_reassembled() {
        // 真实流里非常常见：编码器切 PES 时不管 ADTS 帧边界，一帧被切在两段里。
        // 早期版本会把"装不下的尾巴"直接丢掉，于是每条 PES 都丢一帧
        // —— 一条十秒的音频只剩几个包（实测真实样本只出 11 个包）。
        let mut b = TsBuilder::new();
        b.pat(0x100);
        b.pmt(0x100, 0x101, &[(0x0F, 0x102, Some("eng"))]);
        let mut audio = adts_frame(4, 2, 32);
        audio.extend_from_slice(&adts_frame(4, 2, 32));
        let cut = 20usize; // 第一帧（39 字节）被切成 20 + 19
        b.pes(0x102, 90_000, None, &audio[..cut], 0xC0);
        b.pes(0x102, 90_000 + 2089, None, &audio[cut..], 0xC0);

        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(b.finish()));
        let mut d = TsDemuxer::new(src, "t").unwrap();
        let mut packets = Vec::new();
        while let Some(p) = d.next_packet().unwrap() {
            packets.push(p);
        }
        assert_eq!(packets.len(), 2, "两帧音频必须都被吐出来，不能因为跨 PES 就丢");
        // 拼回去必须与原数据逐字节一致（半帧拼接不能错位）
        let mut joined = packets[0].data.clone();
        joined.extend_from_slice(&packets[1].data);
        assert_eq!(joined, audio, "拼回来的音频必须与原数据一致");
        assert_eq!(packets[0].pts, 90_000);
        assert_eq!(packets[1].pts, 90_000 + 2089);
        assert!(packets.iter().all(|p| p.keyframe), "音频每帧都是随机访问点");
    }

    #[test]
    fn padding_bytes_do_not_poison_the_next_pes() {
        // PES 尾部被编码器填了 0xFF：不能把这堆填充当"残帧"续到下一段，
        // 否则下一段整体错位一格 —— 实测后果是整条音轨一个包都出不来。
        let mut b = TsBuilder::new();
        b.pat(0x100);
        b.pmt(0x100, 0x101, &[(0x0F, 0x102, None)]);
        let audio = adts_frame(4, 2, 32);
        // 声明长度 = 实际 ES 长度 + 32 字节填充
        b.pes_with_padding(0x102, 90_000, &audio, 32);
        b.pes(0x102, 90_000 + 2089, None, &audio, 0xC0);

        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(b.finish()));
        let mut d = TsDemuxer::new(src, "t").unwrap();
        let mut packets = Vec::new();
        while let Some(p) = d.next_packet().unwrap() {
            packets.push(p);
        }
        assert_eq!(packets.len(), 2, "两段各一帧，填充不该让第二段错位");
        assert_eq!(packets[0].data, audio);
        assert_eq!(packets[1].data, audio);
    }

    #[test]
    fn duration_comes_from_pts_span_when_the_whole_stream_was_scanned() {
        // 这条合成流只有两帧视频（90000 / 93600）+ 两帧音频，没有 PCR。
        // 头部扫描已经把整条流读完了，所以时长可以直接用"最后时间戳 − 第一个时间戳"
        // —— 这比"没有 PCR 就报未知"更诚实：我们确实量到了 40 多毫秒的内容。
        let data = build_stream();
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
        let (_f, info) = TsDemuxer::probe_reader(src, "t").unwrap();
        let dur = info.duration_us.expect("扫完整条流后应该给出时长");
        assert!((30_000..60_000).contains(&dur), "两帧之间的跨度应该是几十毫秒，实际 {dur}us");
    }

    #[test]
    fn duration_is_none_when_nothing_was_measured() {
        // 空源：没有任何时间戳、也没有 PCR → 必须报"未知"而不是 0
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(Vec::new()));
        let d = TsDemuxer::new(src, "t");
        assert!(d.is_err(), "空源连 TS 都认不出来，应该报错而不是给一个 0 时长的结果");
    }

    #[test]
    fn language_descriptor_parsing() {
        let d = [0x0A, 0x04, b'e', b'n', b'g', 0x00];
        assert_eq!(find_language_descriptor(&d).as_deref(), Some("eng"));
        // 长度超出实际字节 → None，不能越界
        let bad = [0x0A, 0x40, b'e', b'n'];
        assert_eq!(find_language_descriptor(&bad), None);
        assert_eq!(find_language_descriptor(&[]), None);
        // 非字母的"语言"是垃圾数据，不认
        let junk = [0x0A, 0x04, 0x01, 0x02, 0x03, 0x00];
        assert_eq!(find_language_descriptor(&junk), None);
    }

    #[test]
    fn contains_subslice_basic() {
        assert!(contains_subslice(b"abcdef", b"cd"));
        assert!(!contains_subslice(b"abcdef", b"zz"));
        assert!(!contains_subslice(b"ab", b"abc"));
        assert!(!contains_subslice(b"abc", b""));
    }

    #[test]
    fn m2ts_packet_size_detected() {
        // 造一段 192 字节包（前 4 字节是时码）
        let inner = build_stream();
        let mut m2ts = Vec::new();
        for chunk in inner.chunks(188) {
            m2ts.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // ATC 时码
            m2ts.extend_from_slice(chunk);
        }
        let head = &m2ts[..m2ts.len().min(4096)];
        assert_eq!(packet_size_of(head), Some(192));
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(m2ts));
        let d = TsDemuxer::new(src, "t").unwrap();
        assert_eq!(d.packet_size(), 192);
    }
}