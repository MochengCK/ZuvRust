//! 裸 ADTS AAC 流（`.aac`）—— "裸流"这一族的第三个成员
//! （前两个是 [`crate::mpeg_audio`] 的 `.mp3` 与 [`crate::flac_stream`] 的 `.flac`）。
//!
//! ## 为什么它最好做
//!
//! ADTS 头（7 或 9 字节）里**直接写着这一帧有多长**（13 位 `frame_length`），
//! 所以帧边界是算出来的：不需要像 FLAC 那样"找下一个同步字 + 拿上一帧的 CRC-16 确认"
//! （FLAC 帧头只有 14 位同步码，熵编码数据里到处是像帧头的字节），也不需要像 MP3
//! 那样"由比特率反算帧长"。一帧一取，走到底。
//!
//! ## 但它有两个必须自己解决的麻烦
//!
//! 1. **没有 `esds`**：fMP4 / TS 里的 AAC 参数来自容器的配置盒或描述符，裸 ADTS 流里
//!    只有每帧头里的 2 位 profile + 4 位采样率索引 + 3 位声道配置。解码器要的是
//!    `AudioSpecificConfig`，所以这里**从第一帧的头反推合成**（[`asc_from_header`]）——
//!    与裸 FLAC 那条"从 STREAMINFO 合成 dfLa 载荷"同源。
//! 2. **时长只能估**：ADTS 头里既没有总帧数、也没有码率字段（MP3 的帧头有码率、
//!    FLAC 的 STREAMINFO 有总样本数，所以那两个能算）。这里在整段上取**几个采样点**
//!    量平均帧长（[`estimate_avg_frame_len`]），只读几十 KiB，`probe` 仍是毫秒级。
//!
//! ## 认哪些、不认哪些
//!
//! | 情况 | 处理 |
//! |---|---|
//! | 7 字节头（`protection_absent = 1`）与 9 字节头（带 CRC） | ✅ 都认；CRC 只跳过不校验 |
//! | 前面顶一个 ID3v2 标签 | ✅ 跳过（有些抓轨/下载工具会写） |
//! | AAC-LC 以外的 profile（Main / SSR / LTP） | ✅ 解封装照做（轨道参数照报），**解码器**会明确报不支持 |
//! | 一个 ADTS 帧装多个 `raw_data_block` | ❌ 明确报 `Unsupported`（理由见 [`raw_blocks_unsupported`]） |
//! | `channel_configuration = 0`（声道布局写在 PCE 里） | ❌ 明确报 `Unsupported` |
//! | 采样率索引 13/14（保留）、15（转义） | ❌ 这一帧的头非法，不认 |
//!
//! [`Demuxer`]: crate::demux::Demuxer

use std::sync::Arc;

use me_core::{CodecId, MediaError, MediaInfo, MediaKind, Packet, ReadAt, Result, Timebase, Track};

use crate::probe::ContainerFormat;

/// 探测窗口（与另外两种裸流一致）。
const SNIFF: usize = 4096;

/// 兜底扫描上限：既没标签、4 KiB 里也找不到帧时，最多再往后找这么多。
const FIND_FIRST_SCAN: usize = 64 * 1024;

/// 对不齐时最多往后扫多少字节找下一个同步字。
const RESYNC_SCAN: usize = 64 * 1024;

/// 估平均帧长时每个采样点读多少字节。
///
/// 32 KiB 装得下一两百帧（实测 44.1k 单声道 AAC 平均帧长 90 字节、
/// 48k 立体声 176 字节），样本量足够把"逐帧长度抖动"抹平。
const DURATION_SAMPLE_BYTES: usize = 32 * 1024;

/// 整段不超过这个字节数就**全量走一遍**（帧头挨个跳，字节全在手里）——
/// 这样短音频的时长是精确值，而不是估的。256 KiB 对 `probe` 来说仍然是一次
/// 可以忽略的本地读（对网络源就是一次 range 请求）。
const DURATION_FULL_SCAN_MAX: u64 = 256 * 1024;

/// 估平均帧长取几个采样点（段首 / 段中 / 段尾）。
///
/// 为什么要多点：AAC 帧长逐帧都在变，只看开头会偏 —— 而段首那一帧往往还是个
/// **空帧**（实测 `afconvert` 的第一帧只有 11 字节，真正的音频从第二帧开始）。
const DURATION_REGIONS: u64 = 3;

/// 规范表：`sampling_frequency_index` → 采样率。
///
/// 13 / 14 是保留值、15 表示"后面再给 24 位显式采样率"（ADTS 头里没有那个位置，
/// 所以 15 在 ADTS 里也是非法的）。三个都当"这不是帧头"处理。
const SAMPLING_RATES: [u32; 13] = [
    96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
];

/// 一个 ADTS 帧头解出来的字段。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdtsFrameHeader {
    /// MPEG-4 音频对象类型 = 头里的 2 位 profile + 1（1 = Main、**2 = LC**、3 = SSR、4 = LTP）。
    ///
    /// 为什么存 `object_type` 而不是原始的 `profile`：合成 `AudioSpecificConfig` 时
    /// 要的是对象类型，"加一"这步藏在解析里，调用方就不会忘。
    pub object_type: u8,
    pub sample_rate_index: u8,
    pub sample_rate: u32,
    /// `channel_configuration`（1..=7 是具体布局；0 = 写在 PCE 里，本引擎不支持）。
    pub channels: u8,
    /// 这一帧在码流里占多少字节（**含**帧头）。
    pub frame_len: usize,
    /// 帧头多少字节：7（无 CRC）或 9（`protection_absent = 0`）。
    pub header_len: usize,
    /// 这一帧每声道的样本数 = 1024 × `raw_blocks`。
    pub samples: u32,
    /// `number_of_raw_data_blocks_in_frame`（1 = 常规，>1 = 一帧装多块）。
    pub raw_blocks: u8,
}

impl AdtsFrameHeader {
    /// 每帧样本数的基数：AAC 的 1024 点帧（960 点帧由 `frameLengthFlag` 表示，
    /// 那个标志在 `AudioSpecificConfig` 里、ADTS 头里没有 —— 裸流只能是 1024）。
    pub const SAMPLES_PER_BLOCK: u32 = 1024;
}

/// 解析 `b` 开头的 ADTS 帧头；不是合法帧头就返回 `None`。
///
/// "合法"的判据是**规范里能唯一确定帧长与轨道参数**：12 位同步字、`layer == 0`、
/// 采样率索引 < 13、`frame_len >= header_len`。刻意**不**在这里挡
/// `channel_configuration = 0` 与多 `raw_data_block` —— 那两个是"这道流本引擎不支持"，
/// 属于打开阶段要报 `Unsupported` 的事，不该混进"这是不是帧头"的判断里
/// （混进来会让它们在 `next_packet` 里被当成坏数据**静默跳过**）。
pub fn parse_frame_header(b: &[u8]) -> Option<AdtsFrameHeader> {
    if b.len() < 7 {
        return None;
    }
    // 12 位同步字：字节 0 全 1，字节 1 高 4 位全 1
    if b[0] != 0xFF || (b[1] & 0xF0) != 0xF0 {
        return None;
    }
    // 层固定 00（这个字段是从 MPEG 音频那边继承来的历史包袱）
    if (b[1] >> 1) & 0x03 != 0 {
        return None;
    }
    let protection_absent = b[1] & 0x01 == 1;
    let header_len = if protection_absent { 7 } else { 9 };
    let sample_rate_index = (b[2] >> 2) & 0x0F;
    let sample_rate = *SAMPLING_RATES.get(sample_rate_index as usize)?;
    // frame_length 是 13 位，横跨 byte3 低 2 位 + byte4 + byte5 高 3 位
    let frame_len = (((b[3] & 0x03) as usize) << 11) | ((b[4] as usize) << 3) | ((b[5] as usize) >> 5);
    if frame_len < header_len {
        return None;
    }
    let object_type = ((b[2] >> 6) & 0x03) + 1;
    let channels = ((b[2] & 0x01) << 2) | ((b[3] >> 6) & 0x03);
    let raw_blocks = (b[6] & 0x03) + 1;
    Some(AdtsFrameHeader {
        object_type,
        sample_rate_index,
        sample_rate,
        channels,
        frame_len,
        header_len,
        samples: AdtsFrameHeader::SAMPLES_PER_BLOCK * raw_blocks as u32,
        raw_blocks,
    })
}

/// `off` 处是不是"一对自洽的帧"。
///
/// **为什么要看两帧**：单个 ADTS 头只有 12 位同步字，在二进制数据里撞上的概率不低
/// （`0xFF 0xFx` 这种字节对很常见）。要求"第二帧正好落在第一帧 `frame_length`
/// 算出来的位置上、且采样率/声道/对象类型都一致"之后，误判率降到可忽略 ——
/// 这条也是把它排在"裸 MPEG 音频"前面的底气（那边只有 11 位同步字，还得外加码率表校验）。
///
/// 与 MP3 那边刻意**不**要求声道模式一致相反：ADTS 的参数是"整条流一份"，
/// 逐帧变参数的流不存在（真变了就该换配置了），所以要求一致不会误伤。
pub fn frame_pair_at(bytes: &[u8], off: usize) -> bool {
    let Some(first) = bytes.get(off..).and_then(parse_frame_header) else {
        return false;
    };
    let Some(second) = bytes.get(off + first.frame_len..).and_then(parse_frame_header) else {
        return false;
    };
    first.sample_rate_index == second.sample_rate_index
        && first.channels == second.channels
        && first.object_type == second.object_type
}

/// 在 `buf` 里找第一个合法帧头（`require_pair` = 是否要求"成对"）。
///
/// 两道闸的分工（与裸 MP3 那套完全一致）：
/// - **探测**要求一对自洽的帧 —— 那时还不知道这是什么文件，必须严；
/// - **解封装器打开**只要求一个合法帧头 —— 走到那里说明探测已经确认过是 ADTS 了，
///   再卡"必须有第二帧"会把**只截到一帧的文件**判成打不开（明明能解）。
fn first_frame_offset(buf: &[u8], require_pair: bool) -> Option<usize> {
    let start = crate::mpeg_audio::id3v2_len(buf).unwrap_or(0);
    let scan = buf.len().min(start + FIND_FIRST_SCAN);
    (start..scan.saturating_sub(7)).find(|&off| {
        parse_frame_header(&buf[off..]).is_some() && (!require_pair || frame_pair_at(buf, off))
    })
}

/// 只按头部字节判断"是不是裸 ADTS 流"（探测用）。
///
/// 只认"4 KiB 里就有一对自洽的帧"这一种情况，**不做**"ID3v2 标签长过窗口就当它是
/// ADTS"的兜底（裸 MP3 那边有这个兜底，因为 ID3v2 只会长在 MPEG 音频前面）。
/// 理由：`.aac` 与 `.mp3` 都可能顶着 ID3v2 与几 MB 的专辑封面，光凭 `ID3` 签名分不出
/// 是哪一个 —— 谁先判都会把对方的文件"认成自己再解析失败"。宁可少认（那种文件
/// 可以显式改名/换工具），也不要互相抢。
pub fn looks_like_adts(head: &[u8]) -> bool {
    first_frame_offset(head, true).is_some()
}

/// 定位第一帧：返回（含第一帧的缓冲、第一帧的**绝对**偏移、它在缓冲里的下标）。
///
/// 分三步是为了**别在边下边播时多读**：绝大多数文件第一帧就在开头 4 KiB 里；
/// 只有"ID3v2 长过窗口"（专辑封面）才跳到标签末尾再读一次（而不是傻读 64 KiB ——
/// 对几 MB 的标签来说，`read_range` 会一直等到那些字节下完）；两者都不行才放大窗口。
fn locate_first_frame(src: &Arc<dyn ReadAt>) -> Result<(Vec<u8>, u64, usize)> {
    let head = src.read_range(0, SNIFF)?;
    if let Some(off) = first_frame_offset(&head, false) {
        return Ok((head, off as u64, off));
    }
    if let Some(skip) = crate::mpeg_audio::id3v2_len(&head) {
        let tail = src.read_range(skip as u64, SNIFF)?;
        if let Some(off) = first_frame_offset(&tail, false) {
            return Ok((tail, skip as u64 + off as u64, off));
        }
    }
    let wide = src.read_range(0, FIND_FIRST_SCAN)?;
    match first_frame_offset(&wide, false) {
        Some(off) => Ok((wide, off as u64, off)),
        None => Err(MediaError::container(
            "不是裸 ADTS AAC 流：找不到合法的 ADTS 帧头（12 位同步字 + layer=00）",
        )),
    }
}

/// 从第一帧的头合成 `AudioSpecificConfig`（解码器的初始化数据）。
///
/// 位布局（ISO 14496-3）：5 位对象类型 | 4 位采样率索引 | 4 位声道配置 |
/// GASpecificConfig 的 `frameLengthFlag` / `dependsOnCoreCoder` / `extensionFlag`
/// 各 1 位（都是 0）→ 正好 2 字节。AAC-LC 44.1k 立体声就是经典的 `12 10`。
///
/// 索引 15（显式采样率）在这里不可能出现：ADTS 的索引只能 0..=12，解析时已经挡住了。
fn asc_from_header(h: &AdtsFrameHeader) -> Vec<u8> {
    let index = h.sample_rate_index & 0x0F;
    let channels = h.channels & 0x0F;
    let b0 = (h.object_type << 3) | (index >> 1);
    let b1 = ((index & 0x01) << 7) | (channels << 3);
    vec![b0, b1]
}

/// 一帧装多个 `raw_data_block` 时报的错。
///
/// 为什么不"就把整帧发出去、让解码器自己看着办"：那种帧里每块前面还有自己的 CRC，
/// 要切开就得解析块结构（那是解码器的事，容器层不该依赖解码器）。发出去的结果是
/// 解码器只解第一块、**剩下的静默丢掉** —— 时长与内容都不对，属于"看着成功、其实错"。
fn raw_blocks_unsupported(n: u8) -> MediaError {
    MediaError::Unsupported(format!(
        "ADTS: 这一帧里装了 {n} 个 raw_data_block（一帧多块），本引擎还没接"
    ))
}

// ---------------------------------------------------------------- 解封装器

/// 裸 ADTS AAC 流的解封装器：一条音轨，一条采样 = 一帧。
pub struct AdtsDemuxer {
    src: Arc<dyn ReadAt>,
    source: String,
    /// 第一个音频帧在文件里的偏移（跳过 ID3v2）。
    audio_start: u64,
    /// 下一个要读的帧的偏移。
    pos: u64,
    /// 第一帧的头（轨道参数由它定；ADTS 的参数整条流一份）。
    first: AdtsFrameHeader,
    /// 已经吐出的样本数。时基就是采样率，所以它同时是下一条的 pts。
    samples_out: u64,
    total_bytes: Option<u64>,
    /// 估出来的总时长（微秒）；连一段都量不到就是 None。
    duration_us: Option<i64>,
    /// 估出来的平均码率（bit/s）。
    avg_bitrate: Option<u64>,
    eof: bool,
}

impl AdtsDemuxer {
    pub fn new(src: Arc<dyn ReadAt>, source: impl Into<String>) -> Result<Self> {
        let source = source.into();
        let (buf, abs, off) = locate_first_frame(&src)?;
        let first = parse_frame_header(&buf[off..])
            .ok_or_else(|| MediaError::container("第一个 ADTS 帧头解析失败"))?;
        if first.channels == 0 {
            return Err(MediaError::Unsupported(
                "ADTS: channel_configuration = 0（声道布局写在 PCE 里），本引擎还没接".into(),
            ));
        }
        if first.raw_blocks > 1 {
            return Err(raw_blocks_unsupported(first.raw_blocks));
        }
        let total_bytes = src.len();
        let span = total_bytes.map(|t| t.saturating_sub(abs)).unwrap_or(0);
        let avg_frame_len = if span > 0 { estimate_avg_frame_len(&src, abs, span)? } else { None };
        let duration_us = avg_frame_len.map(|avg| {
            let frames = span as f64 / avg;
            (frames * first.samples as f64 / first.sample_rate as f64 * 1_000_000.0) as i64
        });
        let avg_bitrate = avg_frame_len.map(|avg| {
            (avg * 8.0 * first.sample_rate as f64 / first.samples as f64) as u64
        });
        Ok(Self {
            pos: abs,
            audio_start: abs,
            first,
            samples_out: 0,
            total_bytes,
            duration_us,
            avg_bitrate,
            eof: false,
            src,
            source,
        })
    }

    /// 探测 + 返回容器类型与信息（与另外两个裸流解封装器同形）。
    pub fn probe_reader(src: Arc<dyn ReadAt>, source: &str) -> Result<(ContainerFormat, MediaInfo)> {
        let d = Self::new(src, source)?;
        Ok((ContainerFormat::Adts, d.info()))
    }

    pub fn info(&self) -> MediaInfo {
        let mut track =
            Track::new(1, MediaKind::Audio, CodecId::Aac, Timebase::new(self.first.sample_rate));
        track.sample_rate = self.first.sample_rate;
        track.channels = self.first.channels as u16;
        // 解码器要 AudioSpecificConfig：裸流里没有 `esds`，从帧头合成一份
        track.extradata = asc_from_header(&self.first);
        MediaInfo {
            container: ContainerFormat::Adts.name().to_string(),
            duration_us: self.duration_us,
            bitrate: self.avg_bitrate,
            tracks: vec![track],
            source: self.source.clone(),
        }
    }

    pub fn duration_us(&self) -> Option<i64> {
        self.duration_us
    }

    /// 第一帧的头（**不支持**的判断也基于它：参数整条流一份）。
    pub fn first_header(&self) -> AdtsFrameHeader {
        self.first
    }

    /// 取下一帧。
    ///
    /// 载荷是**完整的一帧（含 ADTS 头）**：自研 AAC 解码器自己会剥头
    /// （`me-decode` 的 `strip_adts`，TS 那条路也是这么喂的），这里不做隐式转换。
    pub fn next_packet(&mut self) -> Result<Option<Packet>> {
        loop {
            if self.eof {
                return Ok(None);
            }
            let head = self.src.read_range(self.pos, 7)?;
            if head.len() < 7 {
                self.eof = true;
                return Ok(None);
            }
            let Some(h) = parse_frame_header(&head) else {
                // 对不齐了（末尾的 ID3v1、填充、半截帧）：往后找下一个真帧头
                match self.resync_from(self.pos + 1)? {
                    Some(p) => {
                        self.pos = p;
                        continue;
                    }
                    None => {
                        self.eof = true;
                        return Ok(None);
                    }
                }
            };
            if h.raw_blocks > 1 {
                return Err(raw_blocks_unsupported(h.raw_blocks));
            }
            let data = self.src.read_range(self.pos, h.frame_len)?;
            if data.len() < h.frame_len {
                // 短读 = 真的到末尾了（`ReadAt` 会自己等"暂时没有"的数据，不拿短读糊弄）。
                // 半截帧丢掉：交给解码器只会在尾巴上报错。
                self.eof = true;
                return Ok(None);
            }
            let pts = self.samples_out as i64;
            self.samples_out += h.samples as u64;
            self.pos += h.frame_len as u64;
            let mut pkt = Packet::new(1, pts, pts, h.samples, data);
            // AAC 帧不受别的帧影响（不像 MP3 有位水库要往回借），随时可作起点
            pkt.keyframe = true;
            return Ok(Some(pkt));
        }
    }

    /// 跳到 `pts_us` 附近，返回实际落点（微秒）。
    ///
    /// 与另外两种裸流同样的处境：**没有任何索引**，只能按"时间比例 ≈ 字节比例"插值
    /// （时长本身就是估的，所以这条比 FLAC/MP3 更粗）。估出来的偏移通常不是帧首，
    /// 落点之后一定要往后同步到真帧头 —— 因此给出的位置**只会晚、不会早**。
    ///
    /// 另外记两条"看起来像 bug、其实是常识"的事（免得以后重复排查）：
    /// - **落点那一帧解出来会与"从头解"不同**：AAC 是跨帧叠加的（IMDCT 50% 重叠），
    ///   从中间某一帧开始解时解码器手里没有上一帧的尾巴，于是**只**影响那一帧
    ///   （实测第一帧最大差 1.6 万，**第二帧起逐样本一致**）。想避免就得往前多解
    ///   一帧再丢掉，那样返回的落点会早于请求位置，播放器与进度条都得额外处理 —— 不值。
    pub fn seek(&mut self, pts_us: i64) -> Result<i64> {
        let want = pts_us.max(0);
        let span = self.total_bytes.map(|t| t.saturating_sub(self.audio_start)).unwrap_or(0);
        let pos = match self.duration_us {
            Some(dur) if dur > 0 && span > 0 => {
                let off = span as i128 * want as i128 / dur as i128;
                self.audio_start + off.clamp(0, span as i128) as u64
            }
            // 没有时长（直播/流式）：定位不了，从头开始 —— 返回 0 让调用方知道
            _ => self.audio_start,
        };
        let pos = match self.total_bytes {
            Some(t) => pos.min(t),
            None => pos,
        };
        self.pos = match self.resync_from(pos)? {
            Some(p) => p,
            None => pos,
        };
        self.eof = false;
        // 落点对应的样本数：同样按比例反推（能估出时长就一定估得出样本数）
        self.samples_out = match (self.duration_us, span) {
            (Some(dur), _) if dur > 0 && span > 0 => {
                let done = self.pos.saturating_sub(self.audio_start).min(span);
                (done as u128 * (dur as u128 * self.first.sample_rate as u128 / 1_000_000)
                    / span as u128) as u64
            }
            _ => 0,
        };
        Ok((self.samples_out as i128 * 1_000_000 / self.first.sample_rate as i128) as i64)
    }

    /// 从 `from` 往后找下一个真帧首。
    ///
    /// 两级判据：先找**一对自洽的帧**（落点之后是要接着解的，选错就喂垃圾给解码器）；
    /// 窗口里凑不出成对、**而且这个窗口已经够到文件末尾**时，退到"这个帧头声明的帧长
    /// 正好吃到文件末尾" —— 那正是"垃圾之后只剩最后一帧"这种真实形态。
    ///
    /// 为什么要卡"够到文件末尾"：`frame_length` 只要凑巧落在窗口末尾也能通过
    /// "正好吃到末尾"这条，而窗口里有几百个候选时这种巧合并不罕见 —— 拿它当落点
    /// 就是往解码器里喂垃圾。卡住之后这条只可能命中"真的最后一帧"。
    fn resync_from(&self, from: u64) -> Result<Option<u64>> {
        let scan = self.src.read_range(from, RESYNC_SCAN)?;
        let scan_end = from + scan.len() as u64;
        let last = scan.len().saturating_sub(7);
        if let Some(off) = (0..last).find(|&off| frame_pair_at(&scan, off)) {
            return Ok(Some(from + off as u64));
        }
        // 文件长度未知（直播）时把窗口末尾当末尾：那种场景本来就没有"最后一帧"可言
        let reaches_eof = self.total_bytes.map(|t| scan_end >= t).unwrap_or(true);
        if !reaches_eof {
            return Ok(None);
        }
        Ok((0..last)
            .find(|&off| {
                parse_frame_header(&scan[off..])
                    .is_some_and(|h| from + off as u64 + h.frame_len as u64 == scan_end)
            })
            .map(|off| from + off as u64))
    }
}

/// 在几个采样点上量平均帧长（估时长与估码率都用它）。
///
/// 为什么要采样而不是"整段扫一遍"：那要把每一帧的头都读一遍（大文件就是整个文件
/// 读一次），与 `probe` "只看头部、毫秒级"的约定冲突。取 3 个点各 32 KiB 已经能把
/// "逐帧长度抖动"与"VBR 的逐段差异"抹平到肉眼看不出 —— 而 `.aac` 本来也不是需要
/// 精确时长的场合（真要精确，解一遍比扫一遍更值）。
///
/// **例外**：整段不超过 [`DURATION_FULL_SCAN_MAX`] 时干脆全量走一遍 ——
/// 短音频（几十秒以内）的时长因此是**精确值**。这一条很重要：短文件里一个采样点
/// 只装得下几帧，"平均帧长"被逐帧抖动带偏 5% 都很正常（实测 0.5 秒的素材就偏了 6%）。
fn estimate_avg_frame_len(src: &Arc<dyn ReadAt>, start: u64, span: u64) -> Result<Option<f64>> {
    if span == 0 {
        return Ok(None);
    }
    if span <= DURATION_FULL_SCAN_MAX {
        let buf = src.read_range(start, span as usize)?;
        let (frames, bytes) = measure_frames(&buf);
        return Ok((frames > 0).then(|| bytes as f64 / frames as f64));
    }
    let mut frames = 0u64;
    let mut bytes = 0u64;
    for k in 0..DURATION_REGIONS {
        let off = start + span * k / DURATION_REGIONS;
        let win = (start + span - off).min(DURATION_SAMPLE_BYTES as u64) as usize;
        let buf = src.read_range(off, win)?;
        let (n, b) = measure_frames(&buf);
        frames += n;
        bytes += b;
    }
    if frames == 0 {
        return Ok(None);
    }
    Ok(Some(bytes as f64 / frames as f64))
}

/// 在一个窗口里"走"帧：返回（帧数, 这些帧的字节数）。
///
/// 两个坑都要绕开：
/// - **窗口末尾的半帧不能算**（只数放得下的完整帧），否则平均帧长被那半个帧拉低；
/// - **载荷里偶然撞出来的假帧头不能算**：AAC 的载荷是熵编码字节，`0xFF 0xFx` 这种
///   字节对平均每 8 KB 就撞上一次，而假帧头会让人**从那里开始整段错位**（数出来的
///   帧数、字节数全歪）。判据是"这一帧的下一帧位置上也得是个合法帧头"（真帧的下一帧
///   一定在）—— 加上这一条，假帧头几乎不可能成立；万一还是在别处断了，
///   就从断点继续找下一个"像真帧首"的位置接着数，**不因为一次意外丢掉整段样本**。
fn measure_frames(buf: &[u8]) -> (u64, u64) {
    let mut off = 0usize;
    let mut frames = 0u64;
    let mut bytes = 0u64;
    while off + 7 <= buf.len() {
        let Some(start) = (off..buf.len().saturating_sub(7)).find(|&o| looks_like_frame_start(buf, o))
        else {
            break;
        };
        let h = parse_frame_header(&buf[start..]).expect("looks_like_frame_start 已经解析过");
        if start + h.frame_len > buf.len() {
            break; // 窗口里放不下完整的一帧：到此为止
        }
        frames += 1;
        bytes += h.frame_len as u64;
        off = start + h.frame_len;
    }
    (frames, bytes)
}

/// `off` 处能不能当"真帧首"：帧头合法，**且下一帧的位置上也是合法帧头**
/// （已经到窗口末尾、没有下一帧可看时，就采信这一帧 —— 每窗最多放宽这一次）。
fn looks_like_frame_start(buf: &[u8], off: usize) -> bool {
    let Some(h) = parse_frame_header(&buf[off..]) else {
        return false;
    };
    let next = off + h.frame_len;
    next + 7 > buf.len() || parse_frame_header(&buf[next..]).is_some()
}

// ---------------------------------------------------------------- Demuxer

impl crate::demux::Demuxer for AdtsDemuxer {
    fn info(&self) -> MediaInfo {
        AdtsDemuxer::info(self)
    }

    fn next_packet(&mut self) -> Result<Option<Packet>> {
        AdtsDemuxer::next_packet(self)
    }

    fn seek(&mut self, pts_us: i64) -> Result<i64> {
        AdtsDemuxer::seek(self, pts_us)
    }

    fn duration_us(&self) -> Option<i64> {
        self.duration_us
    }

    fn container(&self) -> &'static str {
        // 与 `ContainerFormat::Adts.name()` 必须一致：宿主拿这个字符串做分发
        ContainerFormat::Adts.name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demux::Demuxer;
    use me_core::Cursor;

    /// 造一个 ADTS 帧头。
    ///
    /// 参数化采样率索引/声道数/块数，单测里才能造出"规范里合法但本引擎不支持"的组合
    /// （`channel_configuration = 0`、一帧多块）；`with_crc` 造 9 字节头。
    fn header_with(
        frame_len: usize,
        sf_index: u8,
        channels: u8,
        raw_blocks: u8,
        with_crc: bool,
    ) -> Vec<u8> {
        assert!((7..1 << 13).contains(&frame_len), "帧长是 13 位");
        let profile = 1u8; // AAC-LC：对象类型 2 = profile + 1
        let mut h = vec![0u8; if with_crc { 9 } else { 7 }];
        h[0] = 0xFF;
        // 12 位同步字 + ID=0（MPEG-4）+ layer=00 + protection_absent
        h[1] = 0xF0 | if with_crc { 0x00 } else { 0x01 };
        h[2] = (profile << 6) | (sf_index << 2) | ((channels >> 2) & 0x01);
        h[3] = ((channels & 0x03) << 6) | ((frame_len >> 11) & 0x03) as u8;
        h[4] = ((frame_len >> 3) & 0xFF) as u8;
        // buffer_fullness 填 0x7FF（"码率可变"的写法）
        h[5] = ((frame_len & 0x07) << 5) as u8 | 0x1F;
        h[6] = 0xFC | ((raw_blocks - 1) & 0x03);
        h
    }

    fn header(frame_len: usize, sf_index: u8, channels: u8, raw_blocks: u8) -> Vec<u8> {
        header_with(frame_len, sf_index, channels, raw_blocks, false)
    }

    /// 造一个完整帧：头 + `body` 字节载荷（帧长 = 头长 + 载荷）。
    fn frame_with(body: usize, sf_index: u8, channels: u8, raw_blocks: u8, with_crc: bool) -> Vec<u8> {
        let head_len = if with_crc { 9 } else { 7 };
        let mut v = header_with(head_len + body, sf_index, channels, raw_blocks, with_crc);
        v.extend(std::iter::repeat_n(0xA5u8, body));
        v
    }

    /// 常规帧：AAC-LC / 44.1k / 立体声 / 一块 / 无 CRC。
    fn frame(body: usize) -> Vec<u8> {
        frame_with(body, 4, 2, 1, false)
    }

    fn stream(frames: &[Vec<u8>]) -> Vec<u8> {
        frames.concat()
    }

    fn open(data: Vec<u8>) -> AdtsDemuxer {
        AdtsDemuxer::new(Arc::new(Cursor::new(data)), "a.aac").expect("打开 ADTS")
    }

    #[test]
    fn parses_header_fields() {
        let h = parse_frame_header(&frame_with(100, 3, 6, 1, false)).unwrap();
        assert_eq!(h.object_type, 2, "profile = 1 → AAC-LC（对象类型 2）");
        assert_eq!(h.sample_rate_index, 3);
        assert_eq!(h.sample_rate, 48_000);
        assert_eq!(h.channels, 6, "5.1 的声道配置是 6（解封装只管报，解码器会拒绝）");
        assert_eq!(h.header_len, 7);
        assert_eq!(h.frame_len, 107);
        assert_eq!(h.samples, 1024);
        assert_eq!(h.raw_blocks, 1);

        // 带 CRC（protection_absent = 0）时头是 9 字节，帧长把这两个字节也算进去
        let crc = parse_frame_header(&frame_with(100, 4, 1, 1, true)).unwrap();
        assert_eq!(crc.header_len, 9);
        assert_eq!(crc.frame_len, 109);
        assert_eq!(crc.channels, 1);

        // 一帧多块：每块 1024 个样本
        let multi = parse_frame_header(&frame_with(100, 4, 2, 3, false)).unwrap();
        assert_eq!(multi.raw_blocks, 3);
        assert_eq!(multi.samples, 3072);

        // 声道配置 0（PCE）：头本身合法（"这是不是帧头"与"支不支持"分开判）
        assert_eq!(parse_frame_header(&frame_with(50, 4, 0, 1, false)).unwrap().channels, 0);
    }

    #[test]
    fn rejects_illegal_headers() {
        // 同步字不对
        let mut bad = frame(50);
        bad[0] = 0xFE;
        assert!(parse_frame_header(&bad).is_none());
        let mut bad = frame(50);
        bad[1] = 0xE1; // 高 4 位不是 1111
        assert!(parse_frame_header(&bad).is_none());
        // layer 不是 00（这是从 MPEG 音频继承来的字段，ADTS 要求固定 0）
        let mut bad = frame(50);
        bad[1] |= 0b0000_0010;
        assert!(parse_frame_header(&bad).is_none());
        // 采样率索引 13/14 保留、15 转义
        for index in [13u8, 14, 15] {
            assert!(parse_frame_header(&frame_with(50, index, 2, 1, false)).is_none());
        }
        // 帧长比头还短：13 位字段写 0
        let mut bad = frame(50);
        bad[3] &= 0xFC;
        bad[4] = 0;
        bad[5] &= 0x1F;
        assert!(parse_frame_header(&bad).is_none());
        // 连头都不够长
        assert!(parse_frame_header(&[0xFF, 0xF1, 0x60]).is_none());
    }

    #[test]
    fn frame_pair_requires_matching_parameters() {
        let data = stream(&[frame(50), frame(50)]);
        assert!(frame_pair_at(&data, 0));
        // 只有一帧时"成对"不成立（第二帧的位置上是空的）
        assert!(!frame_pair_at(&data[..57], 0));
        // 参数不一致（把第二帧的采样率索引改成 8 = 16 kHz）就不算一对
        let mut mixed = data.clone();
        let second = 57;
        let channels: u8 = 2;
        mixed[second + 2] = (1 << 6) | (8 << 2) | ((channels >> 2) & 0x01);
        assert!(!frame_pair_at(&mixed, 0));
        // 帧长错位（第二帧不在第一帧算出来的位置上）也不算
        let mut shifted = stream(&[frame(50)]);
        shifted.push(0x00); // 中间多一个字节，第二帧就落不到声明的偏移上了
        shifted.extend_from_slice(&frame(51));
        assert!(!frame_pair_at(&shifted, 0));
    }

    #[test]
    fn recognises_adts_and_skips_id3v2() {
        let data = stream(&[frame(100), frame(120), frame(90)]);
        assert!(looks_like_adts(&data));

        // 前面顶一个 ID3v2 标签（10 字节头 + 20 字节正文）
        let mut tagged = b"ID3\x04\x00\x00\x00\x00\x00\x14".to_vec();
        tagged.extend(std::iter::repeat_n(0u8, 20));
        tagged.extend_from_slice(&data);
        assert!(looks_like_adts(&tagged), "标签之后是帧就得认");
        let d = open(tagged);
        assert_eq!(d.first_header().frame_len, 107, "定位到的应当是标签后的第一帧");

        // 随机字节（含"看着像 ADTS"的字节对，但凑不出成对的帧）不认
        let mut noise = vec![0xFFu8, 0xF1, 0x60, 0x00, 0x00, 0x00, 0x00];
        noise.extend(std::iter::repeat_n(0x77u8, 3000));
        assert!(!looks_like_adts(&noise));
        // 只有一帧也认不出来：那更可能是巧合
        assert!(!looks_like_adts(&frame(100)));
    }

    #[test]
    fn demuxer_walks_frames_and_pts() {
        let data = stream(&[frame(100), frame(200), frame(150)]);
        let mut d = open(data);
        let info = d.info();
        assert_eq!(info.container, "adts");
        assert_eq!(info.tracks.len(), 1);
        let t = &info.tracks[0];
        assert_eq!(t.codec, CodecId::Aac);
        assert_eq!(t.sample_rate, 44_100, "采样率直接从 ADTS 头来");
        assert_eq!(t.channels, 2);
        assert_eq!(t.extradata, vec![0x12, 0x10], "AAC-LC 44.1k 立体声的配置");

        for (i, want_len) in [107usize, 207, 157].into_iter().enumerate() {
            let p = d.next_packet().unwrap().expect("应当有第 {i} 帧");
            assert_eq!(p.track_id, 1);
            assert_eq!(p.pts, i as i64 * 1024, "pts 按每帧 1024 个样本累加");
            assert_eq!(p.duration, 1024);
            assert_eq!(p.data.len(), want_len);
            assert!(p.keyframe);
        }
        assert!(d.next_packet().unwrap().is_none(), "到末尾了");
    }

    #[test]
    fn extradata_covers_the_three_common_shapes() {
        // 48 kHz 单声道：对象类型 2、索引 3、声道 1
        let d = open(stream(&[frame_with(80, 3, 1, 1, false), frame_with(80, 3, 1, 1, false)]));
        assert_eq!(d.info().tracks[0].extradata, vec![0x11, 0x88]);
        // 22.05 kHz 立体声：索引 7
        let d = open(stream(&[frame_with(80, 7, 2, 1, false), frame_with(80, 7, 2, 1, false)]));
        assert_eq!(d.info().tracks[0].extradata, vec![0x13, 0x90]);
        assert_eq!(d.info().tracks[0].sample_rate, 22_050);
    }

    #[test]
    fn measurement_ignores_fake_sync_words_inside_payloads() {
        // 载荷里塞满"看着像 ADTS 帧头"的字节（12 位同步字 + layer=00 + 合法采样率索引 +
        // 声称帧长 20）。平均每 8 KB 载荷就会撞上一次这种巧合，如果"走帧"只看单个帧头，
        // 撞上就从那里整段错位 —— 数出来的帧数、字节数全歪，时长也就跟着歪。
        // 整段 342 KB > 256 KB，走的是"三个采样点"那条路（采样点会落在帧中间，
        // 正好考"从帧中间起步"的判断）。
        let body = 120usize;
        // 假帧头放在载荷**中间**（不是开头）：这样"从帧中间起步"的那两个采样点
        // 会先撞上它 —— 放在开头的话，扫到它之前会先扫到下一帧的真帧首，
        // 这条测试就测不到东西了。
        let fake_at = 50usize;
        let mut payload = vec![0x00u8; fake_at];
        payload.extend_from_slice(&[0xFF, 0xF1, 0x60, 0x40, 0x02, 0x9F, 0xFC]);
        payload.resize(body, 0x00);
        assert!(
            parse_frame_header(&payload[fake_at..]).is_some(),
            "这串字节确实像个合法帧头"
        );

        let one = {
            // 手工拼一帧：头 + "载荷里带着假帧头"的体
            let mut f = header_with(7 + payload.len(), 4, 2, 1, false);
            f.extend_from_slice(&payload);
            f
        };
        assert_eq!(one.len(), 7 + body);
        let n = 3200usize;
        let data: Vec<u8> = (0..n).flat_map(|_| one.clone()).collect();
        assert!(data.len() as u64 > DURATION_FULL_SCAN_MAX, "这一条要走采样那条路");

        let d = open(data);
        let want = (n as f64 * 1024.0 / 44_100.0 * 1_000_000.0) as i64;
        let got = d.duration_us().unwrap();
        assert!(
            (got - want).abs() <= 1_000,
            "载荷里的假帧头不该影响平均帧长（估 {got} µs，精确 {want} µs）"
        );
    }

    #[test]
    fn duration_is_estimated_from_average_frame_length() {
        // 每帧等长：平均帧长量出来就是精确值，于是帧数、时长都是精确的
        let body = 100;
        let n = 200usize;
        let one = frame(body);
        let data: Vec<u8> = (0..n).flat_map(|_| one.clone()).collect();
        let span = (n * one.len()) as f64;
        let d = open(data);
        let want = (span / one.len() as f64 * 1024.0 / 44_100.0 * 1_000_000.0) as i64;
        let got = d.duration_us().unwrap();
        assert!(
            (got - want).abs() <= 1_000,
            "等长帧估出来的时长应当准（估 {got} µs，精确 {want} µs）"
        );
        // 码率也应当对得上：帧长 107 字节 / 1024 样本
        let bitrate = d.info().bitrate.unwrap();
        let want_bitrate = 107.0 * 8.0 * 44_100.0 / 1024.0;
        assert!((bitrate as f64 - want_bitrate).abs() < 1.0, "估的码率 {bitrate}");
    }

    #[test]
    fn garbage_between_frames_is_resynced() {
        let mut data = stream(&[frame(100), frame(100)]);
        // 中间塞一段垃圾（含半截 ADTS 头：只有 4 个字节，凑不出合法帧）
        data.extend_from_slice(&[0xFF, 0xF1, 0x60, 0x00]);
        data.extend(std::iter::repeat_n(0x00u8, 40));
        data.extend_from_slice(&frame(120));
        let mut d = open(data);
        assert_eq!(d.next_packet().unwrap().unwrap().data.len(), 107);
        assert_eq!(d.next_packet().unwrap().unwrap().data.len(), 107);
        let third = d.next_packet().unwrap().unwrap();
        assert_eq!(third.data.len(), 127, "跳过垃圾后接上第三帧");
        assert_eq!(third.pts, 2048, "pts 仍然按解出来的帧累加");
        assert!(d.next_packet().unwrap().is_none());
    }

    #[test]
    fn truncated_last_frame_is_dropped() {
        let mut data = stream(&[frame(100), frame(100)]);
        data.extend_from_slice(&header(107, 4, 2, 1)); // 只有头，声称还有 100 字节载荷
        data.extend(std::iter::repeat_n(0u8, 30));
        let mut d = open(data);
        assert_eq!(d.next_packet().unwrap().unwrap().data.len(), 107);
        assert_eq!(d.next_packet().unwrap().unwrap().data.len(), 107);
        assert!(d.next_packet().unwrap().is_none(), "半截帧丢掉，不当成完整帧发出去");
    }

    #[test]
    fn seek_lands_on_a_real_frame() {
        let n = 300usize;
        let one = frame(100);
        let data: Vec<u8> = (0..n).flat_map(|_| one.clone()).collect();
        let mut d = open(data);
        assert_eq!(d.seek(0).unwrap(), 0);
        let p = d.next_packet().unwrap().unwrap();
        assert_eq!(p.pts, 0);
        // 到大约 1/3 处：落点必须是真帧首（ADTS 头能解析出来）
        let landed = d.seek(d.duration_us().unwrap() / 3).unwrap();
        assert!(landed > 0, "落点应当落在段中");
        let p = d.next_packet().unwrap().unwrap();
        assert_eq!(p.data.len(), 107);
        assert!(parse_frame_header(&p.data).is_some());
        // 落点的样本数要对得上"字节比例"（每帧等长，所以是准的）
        let want_samples = (landed as f64 * 44_100.0 / 1_000_000.0) as i64;
        assert!(
            (p.pts - want_samples).abs() <= 2048,
            "落点样本数 {} 与按落点时间算的 {want_samples} 差得太多",
            p.pts
        );
    }

    /// 打开应当失败，并返回错误（`AdtsDemuxer` 没实现 `Debug`，所以不能用 `unwrap_err`）。
    fn open_err(data: Vec<u8>) -> MediaError {
        match AdtsDemuxer::new(Arc::new(Cursor::new(data)), "x.aac") {
            Ok(_) => panic!("这道流不该被接受"),
            Err(e) => e,
        }
    }

    #[test]
    fn unsupported_shapes_are_reported_not_skipped() {
        // 一帧多块
        let multi = stream(&[frame_with(100, 4, 2, 2, false), frame_with(100, 4, 2, 1, false)]);
        assert!(
            matches!(open_err(multi), MediaError::Unsupported(_)),
            "一帧多块要报 Unsupported"
        );

        // channel_configuration = 0（布局在 PCE 里）
        let pce = stream(&[frame_with(100, 4, 0, 1, false), frame_with(100, 4, 0, 1, false)]);
        assert!(
            matches!(open_err(pce), MediaError::Unsupported(_)),
            "PCE 声道布局要报 Unsupported"
        );

        // 这两种流在"走路"时也不能被静默跳过：多块出现在第二帧上照样当场报错
        let mut late = stream(&[frame(100), frame(100)]);
        late.extend_from_slice(&frame_with(100, 4, 2, 2, false));
        let mut d = open(late);
        d.next_packet().unwrap().unwrap();
        d.next_packet().unwrap().unwrap();
        assert!(matches!(d.next_packet(), Err(MediaError::Unsupported(_))));
    }

    #[test]
    fn demuxer_interface_reports_the_same_as_inherent_methods() {
        let mut d = open(stream(&[frame(100), frame(100)]));
        assert_eq!(Demuxer::container(&d), "adts");
        assert_eq!(Demuxer::duration_us(&d), AdtsDemuxer::duration_us(&d));
        assert_eq!(Demuxer::info(&d).container, "adts");
        assert!(Demuxer::next_packet(&mut d).unwrap().is_some());
        assert_eq!(Demuxer::seek(&mut d, 0).unwrap(), 0);
    }
}
