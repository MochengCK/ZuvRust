//! Ogg 容器（`.opus` / `.ogg`）—— 目前只认里面装 **Opus** 的逻辑流。
//!
//! `.opus` 不是"裸流"：它是 Ogg 页的序列（每页 27 字节头 + 段表 + 正文），
//! 一个包可以跨页。所以这里做的是**真正的容器解析**，不是 ADTS/FLAC 那种"一帧接一帧"。
//!
//! ## 页结构（RFC 3533）
//!
//! ```text
//! 0..27   "OggS" | version(1) | header_type(1) | granule(8, 小端, 有符号) |
//!         serial(4) | page_seq(4) | crc(4) | page_segments(1)     ← 除魔数外全小端
//! 27..    段表：`page_segments` 个 lacing 值，每个 1 字节
//! 之后    正文：长度 = 各 lacing 值之和
//! ```
//!
//! - lacing < 255：一个包在这一页**结束**；= 255：包还没完，续到下一页。
//!   因此"包长正好是 255 的整数倍"要靠一个 0 lacing 收尾。
//! - `header_type`：0x01 = 续包、0x02 = 流开始（BOS）、0x04 = 流结束（EOS）。
//! - **CRC-32**（多项式 `0x04c11db7`、**不反射**、初值 0、无最终异或）覆盖整页，
//!   算的时候 CRC 字段先置 0。这是 Ogg 唯一的完整性校验，而"这一页有多长"也来自
//!   段表 —— 所以校验不过时连长度都不可信：只能丢页、往后重找 `OggS`。
//!
//! ## 只认 Opus；别的编码明确报 `Unsupported`
//!
//! Ogg 是通用容器：现实里 `.ogg` 多数装 Vorbis、`.opus` 装 Opus，靠**第一个包**的
//! 魔数区分（`OpusHead` / `\x01vorbis` / `\x7fFLAC` / `Speex   `）。本引擎的 `CodecId`
//! 里没有这些（解码器也没写），所以那些流在打开时**明确报 `Unsupported`**，而不是塞一个
//! 假的编码名进轨道 —— 与 TS 里遇到没接的 `stream_type` 是同一条纪律。
//!
//! ## 时间戳：granule 是权威，TOC 用来分配
//!
//! Opus 的 granule position 单位是 **48 kHz 采样数**（与 `OpusHead` 里那个
//! "输入采样率"无关，那个只是元数据；RFC 7845 §4）。一页的 granule =
//! "在这一页完成的最后一个包"的**末尾**位置。于是：
//!
//! - **包时长**只看包的第一个字节（TOC）：config(5 位) 定"每帧多少毫秒"、c(2 位) 定
//!   "包里几帧"（code 3 时第 2 字节低 6 位给帧数）。见 [`toc_duration`]。
//!   **不需要**解析帧长（那是解码器的事）。
//! - 一页里各包的 pts 由"该页 granule − 这些包时长之和"往前推。正常文件里这个起点
//!   与"上一页末尾"严格相等（实测一个 231 页的真实文件逐页相等）；不等就是丢了页，
//!   这时**以 granule 为准**重新对齐。
//! - **EOS 页是例外**：它的 granule 可能小于"解出来的样本数"，表示末尾有 end trimming
//!   （RFC 7845 §4.4）。这时不能反推起点（会让时间戳倒退），只能继续按 TOC 累加，
//!   把差值体现在总时长上。
//!
//! ## pts 的口径（写下来，免得以后对不上）
//!
//! `Packet::pts` 是**解码样本序号**（从 0 开始、单位 48 kHz、**含** pre-skip 那段预热），
//! 与 AAC / MP3 两条路一致（那两边的解码输出也都带着编码器的预热样本）。
//! pre-skip、输入采样率、输出增益、映射族都在 extradata 里交给解码器 —— extradata 是
//! `OpusHead`（**全引擎统一的 Opus 配置形态**：fMP4 那边读到的 `dOps` 会在容器层
//! 就地规范化成 `OpusHead`，见 [`crate::opus_config`]）；
//! 而 `duration_us` 报的是**可播时长**（已扣起始偏移、pre-skip 与末尾 trim）。
//!
//! ## 已知的取舍
//!
//! - **多路复用**：只跟第一条 `OpusHead` 的逻辑流（按 serial 认），别的 serial 一律跳过。
//! - **链接（chaining）**：第一条链 EOS 之后出现新的 BOS 就收工（FFmpeg / 浏览器默认如此），
//!   不做"多段连播"。
//! - **seek** 对 granule 做二分（RFC 7845 §4.6 的做法），落在"目标所在的那一页"的**页首**，
//!   于是天然带上了页面长度的内容当预热（比规范要求的 80 ms 更宽裕）。

use std::collections::VecDeque;
use std::sync::Arc;

use me_core::{CodecId, MediaError, MediaInfo, MediaKind, Packet, ReadAt, Result, Timebase, Track};

use crate::probe::ContainerFormat;

/// Ogg 页头固定部分长度。
const HEADER_LEN: usize = 27;

/// 一页最多能有多大：27 + 255 段表 + 255×255 正文 = 65307。
/// 读窗口按它来，保证"任意一页都装得下一次"。
const MAX_PAGE_LEN: usize = HEADER_LEN + 255 + 255 * 255;

/// Opus 解码采样率恒为 48 kHz（granule 也是这个单位）。
pub const OPUS_RATE: u32 = 48_000;

/// CRC-32 多项式（Ogg 用**不反射**的写法）。
const CRC_POLY: u32 = 0x04c1_1db7;

/// Ogg 页的 CRC-32：多项式 `0x04c11db7`、不反射、初值 0、无最终异或。
///
/// 已知答案：`crc32_ogg(b"123456789") == 0x89a1_897f`。
/// （CRC-32/MPEG-2 用同一个多项式，但初值 0xFFFFFFFF 且最后取反，所以值不同；
/// 这个常量是拿真实 `.opus` 文件逐页核对出来的。）
pub fn crc32_ogg(data: &[u8]) -> u32 {
    let mut crc = 0u32;
    for &b in data {
        crc ^= (b as u32) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 { (crc << 1) ^ CRC_POLY } else { crc << 1 };
        }
    }
    crc
}

/// 页头里我们关心的字段。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageHeader {
    pub flags: u8,
    /// granule（`None` = 规范里的 -1："这一页没有完成任何包"）。
    pub granule: Option<u64>,
    pub serial: u32,
    pub page_sequence: u32,
    pub segments: u8,
    /// 页头总长 = 27 + 段表项个数。
    pub header_len: usize,
    /// 正文长度（各 lacing 值之和）。
    pub body_len: usize,
    /// 本页最后一个包是否续到下一页（最后一个 lacing 是 255）。
    pub open_ended: bool,
}

impl PageHeader {
    pub fn page_len(&self) -> usize {
        self.header_len + self.body_len
    }
    pub fn is_continued(&self) -> bool {
        self.flags & 0x01 != 0
    }
    pub fn is_bos(&self) -> bool {
        self.flags & 0x02 != 0
    }
    pub fn is_eos(&self) -> bool {
        self.flags & 0x04 != 0
    }
}

/// 解析页头。`buf` 至少要含 `27 + page_segments` 字节（多余的无所谓）。
///
/// **只看结构、不校验 CRC**：校验要用整页，而"这一页多长"恰恰来自段表，
/// 所以顺序必然是"先解头部 → 算出页长 → 读满整页 → 再校验"。
pub fn parse_page_header(buf: &[u8]) -> Option<PageHeader> {
    if buf.len() < HEADER_LEN || &buf[..4] != b"OggS" || buf[4] != 0 {
        return None;
    }
    let flags = buf[5];
    // granule 是有符号 64 位；负数（-1）当成"没有"
    let raw = i64::from_le_bytes(buf[6..14].try_into().ok()?);
    let granule = u64::try_from(raw).ok();
    let serial = u32::from_le_bytes(buf[14..18].try_into().ok()?);
    let page_sequence = u32::from_le_bytes(buf[18..22].try_into().ok()?);
    let segments = buf[26];
    let header_len = HEADER_LEN + segments as usize;
    if buf.len() < header_len {
        return None;
    }
    let mut body_len = 0usize;
    let mut open_ended = false;
    for &lacing in &buf[HEADER_LEN..header_len] {
        body_len += lacing as usize;
        open_ended = lacing == 255;
    }
    Some(PageHeader { flags, granule, serial, page_sequence, segments, header_len, body_len, open_ended })
}

/// 校验一整页（`buf` 必须含完整的一页，长度 = `header.page_len()`）。
pub fn page_crc_ok(buf: &[u8]) -> bool {
    if buf.len() < HEADER_LEN {
        return false;
    }
    let want = u32::from_le_bytes([buf[22], buf[23], buf[24], buf[25]]);
    let mut crc = 0u32;
    for (i, &b) in buf.iter().enumerate() {
        // CRC 字段自身按 0 参与计算
        let byte = if (22..26).contains(&i) { 0 } else { b };
        crc ^= (byte as u32) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 { (crc << 1) ^ CRC_POLY } else { crc << 1 };
        }
    }
    crc == want
}

/// 头部字节看起来是不是 Ogg（探测用）。
///
/// `OggS` + 版本 0 已经是很强的特征；如果整页都在窗口里，再顺手校验一次 CRC ——
/// 连"随机数据里撞上 OggS"都排除掉。
pub fn looks_like_ogg(head: &[u8]) -> bool {
    let Some(h) = parse_page_header(head) else {
        return false;
    };
    if h.page_len() <= head.len() {
        page_crc_ok(&head[..h.page_len()])
    } else {
        true // 第一页比探测窗口还大：先认魔数，进解封装器再逐页校验
    }
}

/// 一个 Opus 包的时长（48 kHz 采样数）与帧数，只看 TOC 字节。
///
/// RFC 6716 §3.1 / Table 2：config 高 5 位给出"模式 + 带宽 + 每帧毫秒数"；c 低 2 位
/// 给出帧数（code 3 时帧数写在第二个字节的低 6 位）。**不需要**解析每帧多少字节。
pub fn toc_duration(packet: &[u8]) -> Option<(u32, u8)> {
    let toc = *packet.first()?;
    let config = toc >> 3;
    let c = toc & 0x03;
    // 每帧毫秒数，用"1/100 毫秒"为单位算，避开浮点
    let frame_ms100: u32 = match config {
        0..=11 => [1000, 2000, 4000, 6000][(config & 3) as usize],
        12..=15 => [1000, 2000][(config & 1) as usize],
        _ => [250, 500, 1000, 2000][(config & 3) as usize],
    };
    let frames: u32 = match c {
        0 => 1,
        1 | 2 => 2,
        _ => {
            let n = (*packet.get(1)? & 0x3F) as u32;
            if n == 0 || n > 48 {
                return None; // 规范：一包最多 48 帧（2.5 ms × 48 = 120 ms）
            }
            n
        }
    };
    // 1/100 毫秒 × 48 kHz：`frame_ms100 * 48000 / 100000`，化成整数写作 ×48/100
    let samples = frame_ms100 * 48 / 100;
    (samples as u64 * frames as u64).try_into().ok().map(|s| (s, frames as u8))
}

/// 头部包的种类（第一个包决定这条逻辑流装的是什么）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OggCodec {
    Opus,
    Vorbis,
    Flac,
    Speex,
    Theora,
    Unknown,
}

impl OggCodec {
    pub fn name(self) -> &'static str {
        match self {
            OggCodec::Opus => "opus",
            OggCodec::Vorbis => "vorbis",
            OggCodec::Flac => "flac",
            OggCodec::Speex => "speex",
            OggCodec::Theora => "theora",
            OggCodec::Unknown => "未知编码",
        }
    }
}

/// 按第一个包的魔数判断这条逻辑流装的是什么。
pub fn sniff_codec(first_packet: &[u8]) -> OggCodec {
    if first_packet.starts_with(b"OpusHead") {
        OggCodec::Opus
    } else if first_packet.starts_with(b"\x01vorbis") {
        OggCodec::Vorbis
    } else if first_packet.starts_with(b"\x7fFLAC") {
        OggCodec::Flac
    } else if first_packet.starts_with(b"Speex   ") {
        OggCodec::Speex
    } else if first_packet.starts_with(b"\x80theora") {
        OggCodec::Theora
    } else {
        OggCodec::Unknown
    }
}

/// `OpusHead` 里我们关心的字段（同时原样留着当 extradata）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpusHead {
    pub channels: u8,
    pub pre_skip: u16,
    /// 编码前的原始采样率（**元数据**，不是解码采样率）。
    pub input_sample_rate: u32,
    /// 输出增益 Q7.8（dB）。
    pub output_gain_q78: i16,
    pub mapping_family: u8,
    pub stream_count: u8,
    pub coupled_count: u8,
    /// 声道映射表（family != 0 时才有）。
    pub mapping: Vec<u8>,
    /// 原始字节（给解码器当 extradata）。
    pub raw: Vec<u8>,
}

/// 解析 `OpusHead`。版本高 4 位不认识就报 `Unsupported`（规范：≥16 不兼容）。
///
/// 字段解析统一在 [`crate::opus_config`]（那里同时管着 fMP4 的 `dOps` 形态），
/// 这里只负责把结果装进带 `raw` 的结构 —— `raw` 要原样交给解码器当 extradata。
pub fn parse_opus_head(p: &[u8]) -> Result<OpusHead> {
    let c = crate::opus_config::OpusConfig::parse_head(p)?;
    Ok(OpusHead {
        channels: c.channels,
        pre_skip: c.pre_skip,
        input_sample_rate: c.input_sample_rate,
        output_gain_q78: c.output_gain_q78,
        mapping_family: c.mapping_family,
        stream_count: c.stream_count,
        coupled_count: c.coupled_count,
        mapping: c.channel_mapping,
        raw: p.to_vec(),
    })
}

// ---------------------------------------------------------------- 页与包

/// 一页（含段表）。
struct RawPage {
    start: u64,
    header: PageHeader,
    lacings: Vec<u8>,
    body: Vec<u8>,
}

impl RawPage {
    /// 切出"在这一页结束"的包（跨页未完成的半包留在 `carry` 里），返回每个包的时长。
    ///
    /// `skip_open`：页首是"续包但手里没有前半段"（丢页 / 从中间接入）—— 规范要求
    /// **不许解**这个包，于是把它的字节整段丢掉。丢掉它不影响我们手里这些包的 pts：
    /// 它们是按"该页 granule 往前推"摆放的，缺的那个包在它们前面。
    fn split(&self, carry: &mut Vec<u8>, skip_open: bool) -> Result<(Vec<Vec<u8>>, Vec<u64>)> {
        let mut data = Vec::new();
        let mut spans = Vec::new();
        let mut pos = 0usize;
        let mut skipping = skip_open;
        for &lacing in &self.lacings {
            let chunk = &self.body[pos..pos + lacing as usize];
            pos += lacing as usize;
            if skipping {
                if lacing < 255 {
                    skipping = false; // 缺前半段的那个包到此结束，已丢弃
                }
                continue;
            }
            carry.extend_from_slice(chunk);
            if lacing < 255 {
                let whole = std::mem::take(carry);
                let (samples, _) = toc_duration(&whole).ok_or_else(|| {
                    MediaError::Bitstream("Ogg Opus: 包开头的 TOC 解不出时长".into())
                })?;
                spans.push(samples as u64);
                data.push(whole);
            }
        }
        if skipping {
            carry.clear(); // 整页都是那个坏包的续段
        }
        Ok((data, spans))
    }

    /// 这一页完成的各包时长之和（反推"流的起始 granule"用）。
    ///
    /// `skip_open` 要与真正取包时一致：页首那个"缺前半段"的续包**不算** ——
    /// 它不属于我们手里的这段流，把它算进去会让起始 granule 偏小。
    fn span(&self, skip_open: bool) -> Result<u64> {
        let mut scratch = Vec::new();
        let (_, spans) = self.split(&mut scratch, skip_open)?;
        Ok(spans.iter().sum())
    }
}

// ---------------------------------------------------------------- 解封装器

/// Ogg 容器里的 Opus 解封装器：一条音轨，一条采样 = 一个 Ogg 包。
pub struct OggOpusDemuxer {
    src: Arc<dyn ReadAt>,
    source: String,
    /// 跟定的逻辑流序号（多路复用文件里别的 serial 全部跳过）。
    serial: u32,
    head: OpusHead,
    /// 流的起始 granule（把第一条音频页往前推得到；正常文件是 0）。
    initial_granule: u64,
    /// 下一条要吐的包在 granule 域的起点。
    next_granule: u64,
    /// 第一条音频页的偏移（seek 的下界）。
    audio_start: u64,
    pos: u64,
    carry: Vec<u8>,
    last_seq: Option<u32>,
    /// 头两个包（OpusHead / OpusTags）还没跳过完。
    headers_left: u8,
    pending: VecDeque<Packet>,
    total_bytes: Option<u64>,
    duration_us: Option<i64>,
    eof: bool,
}

impl OggOpusDemuxer {
    pub fn new(src: Arc<dyn ReadAt>, source: impl Into<String>) -> Result<Self> {
        let total_bytes = src.len();
        let mut d = Self {
            src,
            source: source.into(),
            serial: 0,
            head: OpusHead {
                channels: 0,
                pre_skip: 0,
                input_sample_rate: 0,
                output_gain_q78: 0,
                mapping_family: 0,
                stream_count: 1,
                coupled_count: 0,
                mapping: Vec::new(),
                raw: Vec::new(),
            },
            initial_granule: 0,
            next_granule: 0,
            audio_start: 0,
            pos: 0,
            carry: Vec::new(),
            last_seq: None,
            headers_left: 2,
            pending: VecDeque::new(),
            total_bytes,
            duration_us: None,
            eof: false,
        };
        d.read_headers()?;
        d.duration_us = d.measure_duration()?;
        Ok(d)
    }

    pub fn probe_reader(src: Arc<dyn ReadAt>, source: &str) -> Result<(ContainerFormat, MediaInfo)> {
        let d = Self::new(src, source)?;
        Ok((ContainerFormat::Ogg, d.info()))
    }

    /// 读开头：确认是 Ogg + Opus，拿到 `OpusHead`，并定位第一条音频页。
    fn read_headers(&mut self) -> Result<()> {
        let first = self
            .read_page()?
            .ok_or_else(|| MediaError::container("不是 Ogg 流：开头找不到完整的 Ogg 页"))?;
        if !first.header.is_bos() {
            return Err(MediaError::container(
                "Ogg 第一页没置「流开始」标志（不是一个逻辑流的开头）",
            ));
        }
        let codec = sniff_codec(&first.body);
        if codec != OggCodec::Opus {
            return Err(MediaError::Unsupported(format!(
                "Ogg 里装的是 {}，本引擎只接了 Opus（{} 还没有解码器）",
                codec.name(),
                codec.name()
            )));
        }
        self.serial = first.header.serial;
        let mut scratch = Vec::new();
        let (head_packets, _) = first.split(&mut scratch, false)?;
        let first_packet = head_packets
            .into_iter()
            .next()
            .ok_or_else(|| MediaError::container("Ogg 第一页里没有完整的包"))?;
        self.head = parse_opus_head(&first_packet)?;
        if self.head.mapping_family != 0 {
            return Err(MediaError::Unsupported(format!(
                "Ogg Opus 的声道映射族 {}（多声道/环绕），本引擎只做单声道与立体声",
                self.head.mapping_family
            )));
        }
        // 头两个包是 OpusHead 与 OpusTags，已经用掉一个。注意 **`OpusTags` 可以跨页**
        // （带封面时几 MB 很常见），所以必须按"包"数、带着 carry，不能按"页"数。
        self.headers_left = 1;
        self.carry.clear();
        loop {
            let Some(page) = self.read_page()? else {
                return Ok(()); // 只有头、没有音频：也开得起来（时长按 0 报）
            };
            if page.header.serial != self.serial {
                continue;
            }
            if self.headers_left > 0 {
                let (data, _) = page.split(&mut self.carry, false)?;
                self.headers_left = self.headers_left.saturating_sub(data.len() as u8);
                continue;
            }
            // 第一条音频页：定位 + 反推流的起始 granule，然后**这一页就地吐出去**
            // （已经在手里了，不能重读 —— 重读会把跨页的 carry 丢掉）
            // 这一页页首若是续包，前半段一定不在手里（头包刚跳完、carry 是空的）——
            // 规范要求这种情况**不许解**那个包（拼起来的是残缺数据）
            let skip_open = page.header.is_continued();
            self.audio_start = page.start;
            if let Some(g) = page.header.granule {
                self.initial_granule = g.saturating_sub(page.span(skip_open)?);
            }
            self.next_granule = self.initial_granule;
            self.last_seq = Some(page.header.page_sequence);
            self.consume_page(&page, skip_open)?;
            if page.header.is_eos() {
                self.eof = true;
            }
            return Ok(());
        }
    }

    /// 把一页的包（已切好、已定好 pts）塞进 `pending`。
    ///
    /// pts 的摆放：非 EOS 页**从该页 granule 往前推** —— 这样即使页首那个续包因为
    /// 丢页被丢掉（时长未知），我们手里这些包的位置依然准确（它们由"后面的包时长"
    /// 决定）；EOS 页则从 `next_granule` 往前累加（末尾可能有 end trimming，
    /// 反推会让时间戳倒退）。
    fn consume_page(&mut self, page: &RawPage, skip_open: bool) -> Result<()> {
        let (data, spans) = page.split(&mut self.carry, skip_open)?;
        if data.is_empty() {
            return Ok(());
        }
        let total: u64 = spans.iter().sum();
        let mut at = match page.header.granule {
            Some(g) if !page.header.is_eos() => g.saturating_sub(total),
            _ => self.next_granule,
        };
        for (i, d) in data.into_iter().enumerate() {
            let samples = spans[i] as u32;
            let pts = at.saturating_sub(self.initial_granule) as i64;
            at += spans[i];
            let mut pkt = Packet::new(1, pts, pts, samples, d);
            // Opus 每个包都能独立解（不需要前序包也能出正确样本），随时可作起点
            pkt.keyframe = true;
            self.pending.push_back(pkt);
        }
        self.next_granule = at;
        Ok(())
    }

    /// 读一页（含正文 + CRC）；校验不过就往后重找同步字。
    fn read_page(&mut self) -> Result<Option<RawPage>> {
        loop {
            if let Some(total) = self.total_bytes {
                if self.pos >= total {
                    return Ok(None);
                }
            }
            let head = self.src.read_range(self.pos, HEADER_LEN + 255)?;
            let Some(header) = parse_page_header(&head) else {
                match self.find_next_page(self.pos + 1)? {
                    Some(next) => {
                        self.pos = next;
                        continue;
                    }
                    None => return Ok(None),
                }
            };
            let page_len = header.page_len();
            let raw = self.src.read_range(self.pos, page_len)?;
            if raw.len() < page_len {
                return Ok(None); // 半截页 = 文件被截断
            }
            if !page_crc_ok(&raw) {
                match self.find_next_page(self.pos + 1)? {
                    Some(next) => {
                        self.pos = next;
                        continue;
                    }
                    None => return Ok(None),
                }
            }
            let start = self.pos;
            let lacings = raw[HEADER_LEN..header.header_len].to_vec();
            self.pos += page_len as u64;
            return Ok(Some(RawPage { start, header, lacings, body: raw[header.header_len..].to_vec() }));
        }
    }

    /// 从 `from` 往后找第一个 **CRC 通过**的页，返回它的偏移。
    fn find_next_page(&self, from: u64) -> Result<Option<u64>> {
        let mut at = from;
        loop {
            let window = self.src.read_range(at, MAX_PAGE_LEN)?;
            if window.len() < HEADER_LEN {
                return Ok(None);
            }
            for off in 0..window.len().saturating_sub(HEADER_LEN) {
                if &window[off..off + 4] != b"OggS" {
                    continue;
                }
                let Some(h) = parse_page_header(&window[off..]) else {
                    continue;
                };
                let len = h.page_len();
                let start = at + off as u64;
                if off + len <= window.len() {
                    if page_crc_ok(&window[off..off + len]) {
                        return Ok(Some(start));
                    }
                } else {
                    let full = self.src.read_range(start, len)?;
                    if full.len() == len && page_crc_ok(&full) {
                        return Ok(Some(start));
                    }
                }
            }
            if window.len() < MAX_PAGE_LEN {
                return Ok(None);
            }
            at += (MAX_PAGE_LEN - 4) as u64; // 留 3 字节重叠，防止同步字被切开
        }
    }

    /// 找"偏移 `at` 处或之后"的第一个合法页。
    fn page_at_or_after(&self, at: u64) -> Result<Option<(u64, u64, Option<u64>)>> {
        Ok(self.find_next_page(at)?.map(|start| {
            let head = self.src.read_range(start, HEADER_LEN + 255).unwrap_or_default();
            let h = parse_page_header(&head);
            let end = start + h.map(|h| h.page_len()).unwrap_or(0) as u64;
            (start, end, h.and_then(|h| h.granule))
        }))
    }

    /// 量可播时长：取**文件末尾**最后一页的 granule（Ogg 页最大 65307 字节，
    /// 所以末尾 128 KiB 里一定含有最后一页）。
    fn measure_duration(&self) -> Result<Option<i64>> {
        let total = match self.total_bytes {
            Some(t) if t > 0 => t,
            _ => return Ok(None),
        };
        let tail_len = (total as usize).min(128 * 1024);
        let tail = self.src.read_range(total - tail_len as u64, tail_len)?;
        let mut last_granule = None;
        for off in (0..tail.len().saturating_sub(HEADER_LEN)).rev() {
            if &tail[off..off + 4] != b"OggS" {
                continue;
            }
            let Some(h) = parse_page_header(&tail[off..]) else {
                continue;
            };
            if h.serial != self.serial || off + h.page_len() > tail.len() {
                continue;
            }
            if !page_crc_ok(&tail[off..off + h.page_len()]) {
                continue;
            }
            last_granule = h.granule;
            break;
        }
        let Some(last) = last_granule else {
            return Ok(None);
        };
        // 可播时长 = 末尾 granule − 起始 granule − pre-skip
        //（末尾 granule 已经体现了 end trimming，不用另算）
        let playable = last
            .saturating_sub(self.initial_granule)
            .saturating_sub(self.head.pre_skip as u64);
        Ok(Some((playable as i128 * 1_000_000 / OPUS_RATE as i128) as i64))
    }

    pub fn info(&self) -> MediaInfo {
        let mut track = Track::new(1, MediaKind::Audio, CodecId::Opus, Timebase::new(OPUS_RATE));
        // 解码采样率恒为 48 kHz；OpusHead 里那个"输入采样率"只是元数据
        track.sample_rate = OPUS_RATE;
        track.channels = self.head.channels as u16;
        track.extradata = self.head.raw.clone();
        let bitrate = match (self.total_bytes, self.duration_us) {
            (Some(bytes), Some(us)) if us > 0 => {
                Some((bytes as u128 * 8 * 1_000_000 / us as u128) as u64)
            }
            _ => None,
        };
        MediaInfo {
            container: ContainerFormat::Ogg.name().to_string(),
            duration_us: self.duration_us,
            bitrate,
            tracks: vec![track],
            source: self.source.clone(),
        }
    }

    pub fn duration_us(&self) -> Option<i64> {
        self.duration_us
    }

    /// `OpusHead` 的解析结果（pre-skip / 增益这些给解码器用）。
    pub fn opus_head(&self) -> &OpusHead {
        &self.head
    }

    pub fn next_packet(&mut self) -> Result<Option<Packet>> {
        loop {
            if let Some(p) = self.pending.pop_front() {
                return Ok(Some(p));
            }
            if self.eof {
                return Ok(None);
            }
            let Some(page) = self.read_page()? else {
                self.eof = true;
                return Ok(None);
            };
            // 别的逻辑流：跳过（多路复用文件里很常见）
            if page.header.serial != self.serial {
                continue;
            }
            // 第一条链 EOS 之后又出现 BOS = 换链：播完第一条就收工
            if page.header.is_bos() && self.last_seq.is_some() {
                self.eof = true;
                return Ok(None);
            }
            // 页序不连续 = 中间丢了页：跨页拼起来的半个包不可信，丢掉
            let seq_gap = matches!(self.last_seq, Some(prev)
                if page.header.page_sequence != prev.wrapping_add(1));
            if seq_gap {
                self.carry.clear();
            }
            self.last_seq = Some(page.header.page_sequence);
            // 页首是续包、但手里没有前半段（丢页 / 从中间接入）：规范要求不许解它
            let skip_open = page.header.is_continued() && self.carry.is_empty();
            self.consume_page(&page, skip_open)?;
            if page.header.is_eos() {
                self.eof = true;
            }
        }
    }

    /// 跳到 `pts_us` 附近，返回落点（微秒）。
    ///
    /// granule 就是 48 kHz 采样数，所以"按时间找页"= **对 granule 做二分**
    /// （RFC 7845 §4.6），不像裸流那样按字节比例插值 —— 定位不受码率波动影响。
    /// 落点取"granule ≤ 目标"的**最后一页的页首**，于是天然带上了整页音频当预热。
    pub fn seek(&mut self, pts_us: i64) -> Result<i64> {
        let want = (pts_us.max(0) as i128 * OPUS_RATE as i128 / 1_000_000) as u64;
        let target = self.initial_granule + want;
        let mut lo = self.audio_start;
        let mut hi = self.total_bytes.unwrap_or(self.audio_start).max(self.audio_start);
        let mut best: Option<u64> = None;
        // 二分：hi − lo 收敛到一个最大页长以内，剩下的线性扫
        while hi.saturating_sub(lo) > MAX_PAGE_LEN as u64 {
            let mid = lo + (hi - lo) / 2;
            match self.page_at_or_after(mid)? {
                Some((start, end, granule)) if start < hi && end > start => {
                    if granule.unwrap_or(u64::MAX) <= target {
                        best = Some(start);
                        lo = end;
                    } else {
                        hi = start;
                    }
                }
                _ => hi = mid,
            }
        }
        let mut at = best.unwrap_or(lo);
        loop {
            match self.page_at_or_after(at)? {
                Some((start, end, granule)) if end > start && start < hi => {
                    if granule.unwrap_or(u64::MAX) <= target {
                        best = Some(start);
                    } else {
                        break;
                    }
                    at = end;
                }
                _ => break,
            }
        }
        let landing_page = best.unwrap_or(self.audio_start);
        // 落点页的第一个包的样本位置 = 页面 granule − 该页包时长之和
        let mut landing_granule = self.initial_granule;
        if let Some(page) = self.read_page_at(landing_page)? {
            if let Some(g) = page.header.granule {
                if !page.header.is_eos() {
                    // 与真正取包时同一个口径（页首缺前半段的续包不算）
                    landing_granule = g.saturating_sub(page.span(page.header.is_continued())?);
                }
            }
        }
        self.pos = landing_page;
        self.carry.clear();
        self.last_seq = None;
        self.headers_left = 0;
        self.pending.clear();
        self.next_granule = landing_granule;
        self.eof = false;
        Ok((landing_granule.saturating_sub(self.initial_granule) as i128 * 1_000_000
            / OPUS_RATE as i128) as i64)
    }

    /// 读指定偏移处的那一页（不移动读指针）。
    fn read_page_at(&self, at: u64) -> Result<Option<RawPage>> {
        let head = self.src.read_range(at, HEADER_LEN + 255)?;
        let Some(header) = parse_page_header(&head) else {
            return Ok(None);
        };
        let raw = self.src.read_range(at, header.page_len())?;
        if raw.len() < header.page_len() || !page_crc_ok(&raw) {
            return Ok(None);
        }
        let lacings = raw[HEADER_LEN..header.header_len].to_vec();
        Ok(Some(RawPage {
            start: at,
            header,
            lacings,
            body: raw[header.header_len..].to_vec(),
        }))
    }
}

impl crate::demux::Demuxer for OggOpusDemuxer {
    fn info(&self) -> MediaInfo {
        OggOpusDemuxer::info(self)
    }
    fn next_packet(&mut self) -> Result<Option<Packet>> {
        OggOpusDemuxer::next_packet(self)
    }
    fn seek(&mut self, pts_us: i64) -> Result<i64> {
        OggOpusDemuxer::seek(self, pts_us)
    }
    fn duration_us(&self) -> Option<i64> {
        self.duration_us
    }
    fn container(&self) -> &'static str {
        ContainerFormat::Ogg.name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demux::Demuxer;
    use me_core::Cursor;

    /// 造一页。`segs` 每项是（字节, 这个包/片段是否在这一页结束）。
    ///
    /// **不收尾的片段长度必须是 255 的整数倍** —— 这是 Ogg 的段表语义决定的：
    /// lacing = 255 表示"包还没完、续到下一页"，只有 < 255 才是"包到此为止"。
    /// 想表达"一个包跨页"，就得让前一页的最后一个 lacing 正好是 255。
    fn page_raw(
        flags: u8,
        granule: i64,
        serial: u32,
        seq: u32,
        segs: &[(Vec<u8>, bool)],
    ) -> Vec<u8> {
        let mut lacings: Vec<u8> = Vec::new();
        let mut body: Vec<u8> = Vec::new();
        for (data, ends) in segs {
            if !ends {
                assert!(
                    !data.is_empty() && data.len() % 255 == 0,
                    "不收尾的片段长度得是 255 的整数倍（Ogg 段表语义）"
                );
                lacings.extend(std::iter::repeat_n(255u8, data.len() / 255));
                body.extend_from_slice(data);
                continue;
            }
            let mut left = data.len();
            let mut off = 0usize;
            loop {
                let take = left.min(255);
                lacings.push(take as u8);
                body.extend_from_slice(&data[off..off + take]);
                off += take;
                left -= take;
                if take < 255 {
                    break;
                }
            }
        }
        assert!(lacings.len() <= 255, "段表最多 255 项");
        let mut p = Vec::new();
        p.extend_from_slice(b"OggS");
        p.push(0); // version
        p.push(flags);
        p.extend_from_slice(&granule.to_le_bytes());
        p.extend_from_slice(&serial.to_le_bytes());
        p.extend_from_slice(&seq.to_le_bytes());
        p.extend_from_slice(&[0, 0, 0, 0]); // CRC 占位
        p.push(lacings.len() as u8);
        p.extend_from_slice(&lacings);
        p.extend_from_slice(&body);
        let crc = crc32_ogg(&p);
        p[22..26].copy_from_slice(&crc.to_le_bytes());
        p
    }

    /// 造一页，每个元素都是**完整**的包。
    fn page(flags: u8, granule: i64, serial: u32, seq: u32, packets: &[Vec<u8>]) -> Vec<u8> {
        let segs: Vec<(Vec<u8>, bool)> =
            packets.iter().map(|p| (p.clone(), true)).collect();
        page_raw(flags, granule, serial, seq, &segs)
    }

    /// 造一个 Opus 包：`config`（高 5 位）+ `c`（低 2 位），后面补 `extra` 个 0 字节。
    fn opus_packet(config: u8, c: u8, extra: usize) -> Vec<u8> {
        let mut v = vec![(config << 3) | c];
        v.extend(std::iter::repeat_n(0u8, extra.max(1)));
        v
    }

    fn opus_head(channels: u8, pre_skip: u16, family: u8) -> Vec<u8> {
        let mut v = b"OpusHead".to_vec();
        v.push(1);
        v.push(channels);
        v.extend_from_slice(&pre_skip.to_le_bytes());
        v.extend_from_slice(&44100u32.to_le_bytes());
        v.extend_from_slice(&0i16.to_le_bytes());
        v.push(family);
        if family != 0 {
            v.push(1); // stream count
            v.push(u8::from(channels > 1));
            v.extend(std::iter::repeat_n(0u8, channels as usize));
        }
        v
    }

    fn tags() -> Vec<u8> {
        let mut v = b"OpusTags".to_vec();
        v.extend_from_slice(&4u32.to_le_bytes());
        v.extend_from_slice(b"test");
        v.extend_from_slice(&0u32.to_le_bytes());
        v
    }

    /// 一条最小可用的 `.opus`：OpusHead / OpusTags 各一页，然后音频页。
    /// `pages` 给出音频页：每项是（granule, 每包毫秒对应的 config, 包数, flags）。
    fn build_file(pre_skip: u16, audio_pages: &[(i64, u8, usize, u8)]) -> Vec<u8> {
        let serial = 0x1234_5678;
        let mut f = page(0x02, 0, serial, 0, &[opus_head(2, pre_skip, 0)]);
        f.extend_from_slice(&page(0x00, 0, serial, 1, &[tags()]));
        for (i, (granule, config, count, flags)) in audio_pages.iter().enumerate() {
            let packets: Vec<Vec<u8>> =
                (0..*count).map(|_| opus_packet(*config, 0, 20)).collect();
            f.extend_from_slice(&page(
                *flags,
                *granule,
                serial,
                2 + i as u32,
                &packets,
            ));
        }
        f
    }

    fn open(data: Vec<u8>) -> OggOpusDemuxer {
        OggOpusDemuxer::new(Arc::new(Cursor::new(data)), "x.opus")
            .unwrap_or_else(|e| panic!("打开 Ogg Opus 失败：{e}"))
    }

    /// 打开应当失败，返回错误（解封装器没实现 `Debug`，不能用 `unwrap_err`）。
    fn open_err(data: Vec<u8>) -> MediaError {
        match OggOpusDemuxer::new(Arc::new(Cursor::new(data)), "x.opus") {
            Ok(_) => panic!("这条流不该被接受"),
            Err(e) => e,
        }
    }

    #[test]
    fn crc_has_a_known_answer() {
        // 拿真实 .opus 文件逐页核对过的值（多项式 0x04c11db7、不反射、初值 0、不取反）
        assert_eq!(crc32_ogg(b"123456789"), 0x89a1_897f);
    }

    #[test]
    fn parses_page_header_and_checks_crc() {
        let p = page(0x02, 0, 0xdead_beef, 7, &[vec![1, 2, 3], vec![4; 300]]);
        let h = parse_page_header(&p).unwrap();
        assert_eq!(h.serial, 0xdead_beef);
        assert_eq!(h.page_sequence, 7);
        assert_eq!(h.granule, Some(0));
        assert!(h.is_bos() && !h.is_eos() && !h.is_continued());
        // 段表：3 字节的包（1 段）+ 300 字节的包（255 + 45 两段）= 3 段
        assert_eq!(h.segments, 3);
        assert_eq!(h.body_len, 303);
        assert_eq!(h.page_len(), p.len());
        assert!(!h.open_ended);
        assert!(page_crc_ok(&p));

        // granule = -1 → None（"这一页没有完成的包"）
        let q = page(0, -1, 1, 0, &[vec![9; 10]]);
        assert_eq!(parse_page_header(&q).unwrap().granule, None);

        // 正文改一个字节 → CRC 必须不过
        let mut bad = p.clone();
        let last = bad.len() - 1;
        bad[last] ^= 0xFF;
        assert!(!page_crc_ok(&bad), "CRC 要能抓出篡改");
        // 版本不是 0 就不认
        let mut v = p.clone();
        v[4] = 1;
        assert!(parse_page_header(&v).is_none());
        // 魔数不对
        assert!(parse_page_header(b"OggX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00").is_none());
    }

    #[test]
    fn toc_durations_follow_table_2() {
        // SILK-only 10/20/40/60 ms（config 0..3）
        assert_eq!(toc_duration(&[0 << 3]).unwrap().0, 480);
        assert_eq!(toc_duration(&[1 << 3]).unwrap().0, 960);
        assert_eq!(toc_duration(&[2 << 3]).unwrap().0, 1920);
        assert_eq!(toc_duration(&[3 << 3]).unwrap().0, 2880);
        // Hybrid 10/20 ms（config 12..15）
        assert_eq!(toc_duration(&[12 << 3]).unwrap().0, 480);
        assert_eq!(toc_duration(&[15 << 3]).unwrap().0, 960);
        // CELT 2.5/5/10/20 ms（config 16..19 与 28..31）
        assert_eq!(toc_duration(&[16 << 3]).unwrap().0, 120);
        assert_eq!(toc_duration(&[17 << 3]).unwrap().0, 240);
        assert_eq!(toc_duration(&[18 << 3]).unwrap().0, 480);
        assert_eq!(toc_duration(&[31 << 3]).unwrap().0, 960);
        // 帧数：c=0 → 1 帧；c=1/2 → 2 帧；c=3 → 第二个字节低 6 位（下面都是 config 0）
        assert_eq!(toc_duration(&[0x00, 0]).unwrap().0, 480, "c=0 一帧");
        assert_eq!(toc_duration(&[0x01, 0, 0]).unwrap().0, 960, "c=1 两帧等长");
        assert_eq!(toc_duration(&[0x02, 0, 0]).unwrap().0, 960, "c=2 两帧不等长");
        let mut code3 = vec![0x03u8, 6]; // c=3：第二个字节低 6 位 = 6 帧
        code3.resize(32, 0);
        assert_eq!(toc_duration(&code3).unwrap(), (6 * 480, 6));
        // 非法：code 3 但帧数 0 或 > 48
        assert!(toc_duration(&[0x03, 0, 0]).is_none());
        assert!(toc_duration(&[0x03, 49, 0]).is_none());
        // 空包
        assert!(toc_duration(&[]).is_none());
    }

    #[test]
    fn opus_head_fields_and_rejections() {
        let h = parse_opus_head(&opus_head(2, 356, 0)).unwrap();
        assert_eq!(h.channels, 2);
        assert_eq!(h.pre_skip, 356);
        assert_eq!(h.input_sample_rate, 44100);
        assert_eq!(h.mapping_family, 0);
        assert_eq!(h.stream_count, 1);
        assert_eq!(h.coupled_count, 1, "立体声在 family 0 下是耦合的一路");
        assert_eq!(h.raw.len(), 19, "family 0 的 OpusHead 就是 19 字节");

        // 版本 16 以上不认识（规范说不兼容）
        let mut v = opus_head(2, 0, 0);
        v[8] = 16;
        assert!(matches!(
            parse_opus_head(&v).unwrap_err(),
            MediaError::Unsupported(_)
        ));
        // 版本 15 属于"可兼容的小版本"，要认
        v[8] = 15;
        assert!(parse_opus_head(&v).is_ok());
        // 声道数 0 非法
        assert!(parse_opus_head(&opus_head(0, 0, 0)).is_err());
        // 映射表被截断
        let mut cut = opus_head(2, 0, 1);
        cut.truncate(20);
        assert!(parse_opus_head(&cut).is_err());
    }

    #[test]
    fn walks_packets_and_derives_pts_from_granule() {
        // 两页音频：20 ms/包（960 样本），每页两包
        let data = build_file(356, &[(1920, 1, 2, 0), (3840, 1, 2, 0x04)]);
        let mut d = open(data);
        let info = d.info();
        assert_eq!(info.container, "ogg");
        assert_eq!(info.tracks.len(), 1);
        let t = &info.tracks[0];
        assert_eq!(t.codec, CodecId::Opus);
        assert_eq!(t.sample_rate, 48_000, "Opus 解码恒为 48 kHz");
        assert_eq!(t.channels, 2);
        assert_eq!(t.extradata.len(), 19);

        let mut got = Vec::new();
        while let Some(p) = d.next_packet().unwrap() {
            got.push((p.pts, p.duration));
        }
        assert_eq!(
            got,
            vec![(0, 960), (960, 960), (1920, 960), (2880, 960)],
            "pts 按 granule 反推，包时长来自 TOC"
        );
        assert_eq!(d.duration_us().unwrap(), 3484 * 1_000_000 / 48_000);
    }

    #[test]
    fn end_trimming_shows_up_in_duration_not_in_pts() {
        // 第二页 granule 比"解出来的样本数"小 400 → 末尾被裁掉（RFC 7845 §4.4）
        let data = build_file(0, &[(1920, 1, 2, 0), (3440, 1, 2, 0x04)]);
        let mut d = open(data);
        assert_eq!(d.duration_us().unwrap(), 3440 * 1_000_000 / 48_000);
        let mut pts = Vec::new();
        while let Some(p) = d.next_packet().unwrap() {
            pts.push(p.pts);
        }
        assert_eq!(pts, vec![0, 960, 1920, 2880], "pts 不能因为末尾 trim 而倒退");
    }

    #[test]
    fn pre_skip_is_excluded_from_duration() {
        // 两页共 3840 样本，pre-skip 356 → 可播 3484 样本
        let data = build_file(356, &[(1920, 1, 2, 0), (3840, 1, 2, 0x04)]);
        let d = open(data);
        assert_eq!(d.duration_us().unwrap(), 3484 * 1_000_000 / 48_000);
        assert_eq!(d.opus_head().pre_skip, 356);
    }

    #[test]
    fn packet_spanning_two_pages_is_rejoined() {
        let serial = 7;
        // 真实形态：第 1 页只有 OpusHead、第 2 页是 OpusTags，第 3/4 页装一个跨页的音频包
        // （前 255 字节的最后一个 lacing = 255 = 续页，后 145 字节在第 4 页收尾）
        let mut whole = vec![0x08u8]; // TOC：config 1 / c=0 → SILK-only 20 ms = 960 样本
        whole.resize(400, 0x5A);
        let mut first = page(0x02, 0, serial, 0, &[opus_head(1, 0, 0)]);
        first.extend_from_slice(&page(0x00, 0, serial, 1, &[tags()]));
        first.extend_from_slice(&page_raw(0x00, -1, serial, 2, &[(whole[..255].to_vec(), false)]));
        first.extend_from_slice(&page_raw(
            0x01 | 0x04,
            1920,
            serial,
            3,
            &[(whole[255..].to_vec(), true)],
        ));
        let mut d = open(first);
        let p = d.next_packet().unwrap().expect("跨页的包要拼起来");
        assert_eq!(p.data.len(), 400, "跨页包必须拼回完整长度");
        assert_eq!(p.data[399], 0x5A);
        assert_eq!((p.data[0] >> 3), 1, "TOC 还是原来的（没有被拼接破坏）");
        assert_eq!(p.duration, 960, "时长来自 TOC，与跨页无关");
        assert_eq!(p.pts, 0, "起始 granule = 1920 − 一个包 960 = 960，所以 pts 从 0 起");
    }

    #[test]
    fn vorbis_inside_ogg_is_unsupported_not_misread() {
        let mut head = b"\x01vorbis".to_vec();
        head.extend(std::iter::repeat_n(0u8, 22));
        let data = page(0x02, 0, 3, 0, &[head]);
        match open_err(data) {
            MediaError::Unsupported(m) => assert!(m.contains("vorbis"), "要点明是什么：{m}"),
            other => panic!("应当报 Unsupported，得到 {other:?}"),
        }
    }

    #[test]
    fn multichannel_mapping_family_is_unsupported() {
        let data = page(0x02, 0, 3, 0, &[opus_head(6, 0, 1)]);
        match open_err(data) {
            MediaError::Unsupported(m) => assert!(m.contains("映射族"), "要说清原因：{m}"),
            other => panic!("应当报 Unsupported，得到 {other:?}"),
        }
    }

    #[test]
    fn recognises_ogg_and_rejects_other_things() {
        let ok = page(0x02, 0, 1, 0, &[opus_head(1, 0, 0)]);
        assert!(looks_like_ogg(&ok));
        // CRC 坏的页不认（探测阶段就能排除"随机数据里撞上 OggS"）
        let mut bad = ok.clone();
        let last = bad.len() - 1;
        bad[last] ^= 1;
        assert!(!looks_like_ogg(&bad));
        assert!(!looks_like_ogg(b"not an ogg file"));
        assert!(!looks_like_ogg(&[]));
    }

    #[test]
    fn seek_lands_on_the_page_holding_the_target() {
        // 10 页 × 4 包 × 960 样本 = 3840 样本/页
        let pages: Vec<(i64, u8, usize, u8)> = (0..10)
            .map(|i| {
                let last = if i == 9 { 0x04 } else { 0 };
                ((i + 1) as i64 * 3840, 1u8, 4usize, last)
            })
            .collect();
        let data = build_file(0, &pages);
        let mut d = open(data);
        // 目标落在第 5 页（样本 19200）里：落点应当是那一页的页首（样本 15360）
        let landed = d.seek(19_500 * 1_000_000 / 48_000).unwrap();
        let want = 15_360 * 1_000_000 / 48_000;
        assert!(
            (landed - want).abs() <= 1000,
            "落点应当是「granule ≤ 目标」的那一页页首：得到 {landed} µs，期望约 {want} µs"
        );
        let p = d.next_packet().unwrap().unwrap();
        assert_eq!(p.pts, 15_360, "落点页的第一个包");
        // 落点之后照常往前吐
        let p2 = d.next_packet().unwrap().unwrap();
        assert_eq!(p2.pts, 15_360 + 960);
    }

    #[test]
    fn page_sequence_gap_drops_the_partial_packet() {
        let serial = 9;
        let p0 = page(0x02, 0, serial, 0, &[opus_head(1, 0, 0)]);
        let p1 = page(0x00, 0, serial, 1, &[tags()]);
        // 第 3 页只装"半个包"（granule = -1：本页没有完成的包）
        let p2 = page_raw(0x00, -1, serial, 2, &[(vec![7u8; 255], false)]);
        // 第 4 页是那个包的续段，但**页序号跳了**（模拟丢页）→ 规范要求不许解它
        let p3 = page_raw(0x01, 1920, serial, 5, &[(vec![7u8; 21], true)]);
        let p4 = page(0x04, 2880, serial, 6, &[opus_packet(1, 0, 20)]);
        let data = [p0, p1, p2, p3, p4].concat();
        let mut d = open(data);
        let first = d.next_packet().unwrap().unwrap();
        assert_eq!(first.data.len(), 21, "丢页时不能把拼了一半的包吐出来（p3 整页丢掉）");
        assert_eq!(first.pts, 0, "首音频页 granule 是 -1（没有完成的包），起始只能按 0 算");
        assert_eq!(first.duration, 960);
        assert!(d.next_packet().unwrap().is_none());
    }

    #[test]
    fn continued_page_without_the_first_half_is_not_decoded() {
        // 从"流中间"接入：第一页就带续包标志、手里没有前半段
        let serial = 11;
        let p0 = page(0x02, 0, serial, 0, &[opus_head(1, 0, 0)]);
        let p1 = page(0x00, 0, serial, 1, &[tags()]);
        // 第 3 页：页首是续段（不许解），后面跟一个完整包
        let p2 = page_raw(
            0x01,
            2880,
            serial,
            2,
            &[(vec![0xEEu8; 40], true), (opus_packet(1, 0, 20), true)],
        );
        let data = [p0, p1, p2].concat();
        let mut d = open(data);
        let first = d.next_packet().unwrap().unwrap();
        assert_eq!(first.data.len(), 21, "续段丢掉，只吐完整包");
        assert_eq!(
            first.pts, 0,
            "起始 granule 由第一页反推（2880 − 960 = 1920），第一个**可解**包就是 pts 0"
        );
        assert_eq!(first.duration, 960);
    }

    #[test]
    fn demuxer_interface_matches_inherent_methods() {
        let mut d = open(build_file(0, &[(960, 0, 1, 0x04)]));
        assert_eq!(Demuxer::container(&d), "ogg");
        assert_eq!(Demuxer::duration_us(&d), OggOpusDemuxer::duration_us(&d));
        assert_eq!(Demuxer::info(&d).container, "ogg");
        assert!(Demuxer::next_packet(&mut d).unwrap().is_some());
        assert_eq!(Demuxer::seek(&mut d, 0).unwrap(), 0);
    }
}
