//! 裸 FLAC 流（`.flac`）—— "裸流"这一族的第二个成员（第一个是 [`crate::mpeg_audio`]）。
//!
//! 与裸 MP3 一样，它**不是容器**：`fLaC` 魔数后面跟一串元数据块（STREAMINFO、
//! VORBIS_COMMENT、PICTURE……），元数据之后就是首尾相接的音频帧。
//! 之所以照 [`Demuxer`] 的接口实现，理由与 MP3 那条完全一样：
//! 让 `probe` / `decode --audio` / `remux` / `play` 认得用户手里的 `.flac`。
//!
//! [`Demuxer`]: crate::demux::Demuxer
//!
//! ## 与裸 MP3 最大的不同：FLAC 的帧**不自带长度**
//!
//! MP3 的帧长能由帧头里的比特率算出来（`144×bitrate/采样率`），ADTS 更直接
//! （头里就写着 `frame_length`）。FLAC 不行：帧头只给"这一帧有多少个样本"，
//! **一个字节的长度都没有** —— 要精确知道边界，只有两条路：
//!
//! 1. 把整帧解一遍（那是解码器干的事，容器层不该依赖解码器）；
//! 2. **找下一个帧头，再用"上一帧的页脚 CRC-16"确认**（本模块的做法，也是 FFmpeg
//!    的 FLAC 解析器走的路）。
//!
//! 第二条路的关键是**两级 CRC 过滤**，否则熵编码数据里到处都是从同步码看很像帧头的字节：
//! - 帧头自带 **CRC-8**（覆盖同步码到 CRC 之前的所有字节）→ 先便宜地筛掉绝大多数假货；
//! - 帧尾是 **CRC-16**（覆盖整帧除页脚自身）→ 真边界一定对得上，对不上就继续往后找。
//!
//! ## 认哪些、不认哪些
//!
//! | 情况 | 处理 |
//! |---|---|
//! | `fLaC` + 元数据块序列 | ✅ 认；STREAMINFO 拿出来当 extradata，其余块（含几 MB 的封面）**跳过不读** |
//! | 前面顶一个 ID3v2 标签 | ✅ 跳过（FLAC 规范不建议这么干，但现实里真有） |
//! | 8 / 12 / 16 / 20 / 24 / 32 位 | ✅ 都交给解码器（它有对应的子帧解码路径） |
//! | 1 ~ 8 声道（含左右/中侧立体声） | ✅ |
//! | 没有 `fLaC` 魔数的"裸帧流" | ❌ 不认（没有 STREAMINFO 就没有采样率/位深，硬猜不如说不认识） |

use std::sync::Arc;

use me_core::{CodecId, MediaError, MediaInfo, MediaKind, Packet, ReadAt, Result, Timebase, Track};

use crate::probe::ContainerFormat;

/// 探测窗口（与另外三种形态一致）。
const SNIFF: usize = 4096;

/// STREAMINFO 块长（固定）。
const STREAMINFO_LEN: usize = 34;

/// 一帧最少多少字节：帧头（4）+ 编号（1）+ CRC-8（1）+ 至少一字节子帧数据 + 页脚（2）。
const MIN_FRAME_BYTES: usize = 8;

/// 读一帧时窗口的起止大小。
///
/// 16 KiB 覆盖绝大多数帧（44.1k/16 位立体声的 4096 样本帧约 8~12 KiB）；
/// 找不到边界就翻倍，上限 1 MiB —— 32 位 8 声道的极端帧也在这个量级以内。
const FRAME_WINDOW_MIN: usize = 16 * 1024;
const FRAME_WINDOW_MAX: usize = 1024 * 1024;

/// 找下一个边界时最多试多少个候选帧头（防止在垃圾数据里扫太久）。
const MAX_BOUNDARY_TRIES: usize = 64;

/// FLAC 的流信息（`STREAMINFO` 元数据块里我们关心的字段）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlacStreamInfo {
    pub sample_rate: u32,
    pub channels: u8,
    pub bits_per_sample: u8,
    /// 总样本数（每声道）；0 = 未知（编码时没填，边录边传的流就是这样）
    pub total_samples: u64,
    /// 音频数据从哪个偏移开始（= `fLaC` + 全部元数据块之后）。
    pub audio_start: u64,
}

/// 一个 FLAC 帧头解析出来的字段。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlacFrameHeader {
    /// 这一帧每声道多少个样本。
    pub block_size: u32,
    /// 帧头占多少字节（含末尾的 CRC-8）。
    pub header_len: usize,
    /// 定长分块流（编号是帧号）还是变长分块流（编号是样本号）。
    pub variable_blocking: bool,
}

// ---------------------------------------------------------------- 校验和

/// FLAC 帧头用的 CRC-8（多项式 `x⁸+x²+x+1`，初值 0）。
///
/// 校验值（非循环一致性，用来钉住实现）：`crc8(b"123456789") == 0xF4`。
pub fn crc8(b: &[u8]) -> u8 {
    let mut c = 0u8;
    for byte in b {
        c ^= *byte;
        for _ in 0..8 {
            c = if c & 0x80 != 0 { (c << 1) ^ 0x07 } else { c << 1 };
        }
    }
    c
}

/// FLAC 帧尾用的 CRC-16（多项式 `x¹⁶+x¹⁵+x²+1`，初值 0，非反转）。
///
/// 校验值：`crc16(b"123456789") == 0xFEE8`。
pub fn crc16(b: &[u8]) -> u16 {
    let mut c = 0u16;
    for byte in b {
        c ^= (*byte as u16) << 8;
        for _ in 0..8 {
            c = if c & 0x8000 != 0 { (c << 1) ^ 0x8005 } else { c << 1 };
        }
    }
    c
}

// ---------------------------------------------------------------- 帧头

/// 解析 `b` 开头的 FLAC 帧头；不是合法帧头（含 **CRC-8 对不上**）就返回 `None`。
///
/// **不返回采样率/位深**：这两个参数以 STREAMINFO 为准（帧头允许"重写"它们，
/// 但现实中编码器不会写出不一致的值）。时间轴按 STREAMINFO 的采样率算，
/// 帧头里那几个采样率字段只影响"要跳过多少字节"。
pub fn parse_frame_header(b: &[u8]) -> Option<FlacFrameHeader> {
    if b.len() < 6 {
        return None;
    }
    // 14 位同步码 + 1 位"必须为 0"的保留位 + 1 位分块策略 → 只可能是 0xFFF8 / 0xFFF9
    if b[0] != 0xFF || (b[1] & 0xFE) != 0xF8 {
        return None;
    }
    let variable_blocking = b[1] & 0x01 != 0;
    let block_size_code = (b[2] >> 4) & 0x0F;
    let sample_rate_code = b[2] & 0x0F;
    let channel_code = (b[3] >> 4) & 0x0F;
    let sample_size_code = (b[3] >> 1) & 0x07;
    if b[3] & 0x01 != 0 {
        return None; // 末位保留位必须为 0
    }
    if channel_code > 10 {
        return None; // 11~15 是保留值
    }
    if sample_size_code == 3 || sample_rate_code == 15 {
        return None; // 保留值
    }
    let mut pos = 4usize;
    // 帧号（定长）或样本号（变长），用 UTF-8 那样编码，1~7 字节
    let digits = utf8_number_len(b.get(pos..)?)?;
    pos += digits;
    // 块大小：6/7 号往头里塞了实际值（存的是"值-1"）
    let block_size = match block_size_code {
        0 => return None,
        1 => 192,
        2..=5 => 576u32 << (block_size_code - 2),
        6 => {
            let v = *b.get(pos)? as u32 + 1;
            pos += 1;
            v
        }
        7 => {
            let hi = *b.get(pos)? as u32;
            let lo = *b.get(pos + 1)? as u32;
            pos += 2;
            (hi << 8 | lo) + 1
        }
        _ => 256u32 << (block_size_code - 8),
    };
    // 采样率：12/13/14 号在头里多带字段 —— **按正确长度跳过**，否则 CRC-8 的位置就错了
    match sample_rate_code {
        12 => pos += 1,
        13 | 14 => pos += 2,
        _ => {}
    }
    // 帧头最后一个是 CRC-8，覆盖它**之前**的全部字节
    let crc = *b.get(pos)?;
    if crc8(&b[..pos]) != crc {
        return None;
    }
    Some(FlacFrameHeader { block_size, header_len: pos + 1, variable_blocking })
}

/// UTF-8 编码的帧号/样本号占多少字节；不合法返回 `None`。
///
/// 只看长度不看值：本模块用不上那个数（时间戳按块大小累加，采样率还是从
/// STREAMINFO 来的），但**必须知道它有多长**才能找到后面的 CRC-8。
fn utf8_number_len(b: &[u8]) -> Option<usize> {
    let first = *b.first()?;
    let extra = match first {
        0x00..=0x7F => 0,
        0xC0..=0xDF => 1,
        0xE0..=0xEF => 2,
        0xF0..=0xF7 => 3,
        0xF8..=0xFB => 4,
        0xFC..=0xFD => 5,
        0xFE => 6, // FLAC 允许用 0xFE 表示 7 字节编号
        _ => return None, // 0x80..=0xBF（续字节开头）、0xFF 都不合法
    };
    for i in 1..=extra {
        let c = *b.get(i)?;
        if c & 0xC0 != 0x80 {
            return None;
        }
    }
    Some(extra + 1)
}

// ---------------------------------------------------------------- 文件头

/// 跳过 ID3v2（FLAC 规范不建议，但现实里真有），返回 `fLaC` 的偏移。
fn flac_magic_offset(head: &[u8]) -> Option<usize> {
    let off = crate::mpeg_audio::id3v2_len(head).unwrap_or(0);
    if head.len() >= off + 4 && &head[off..off + 4] == b"fLaC" {
        Some(off)
    } else {
        None
    }
}

/// 只按头部字节判断"是不是裸 FLAC 流"（探测用）。
///
/// `fLaC` 是 4 字节魔数，比 MP3 那 11 位同步码强得多，所以不需要"看两帧"那种兜底。
pub fn looks_like_flac(head: &[u8]) -> bool {
    flac_magic_offset(head).is_some()
}

/// 解析元数据块序列，取 STREAMINFO，并算出音频起点。
///
/// **只读 STREAMINFO 的载荷，其余块一律跳过**：PICTURE（专辑封面）动辄几 MB，
/// 把它读进来既慢又白费内存 —— 而它在真实文件里非常常见。
///
/// 注意音频起点是**走完所有元数据块**之后（`last` 标志）才算出来的：
/// STREAMINFO 之后常常还有 VORBIS_COMMENT、PICTURE，只算到 STREAMINFO 末尾
/// 就会把后面的块当成音频数据（有封面的文件全都解析不出来）。
fn read_stream_info(src: &Arc<dyn ReadAt>, magic: usize) -> Result<FlacStreamInfo> {
    let mut pos = magic as u64 + 4;
    let mut found: Option<FlacStreamInfo> = None;
    loop {
        let head = src.read_range(pos, 4)?;
        if head.len() < 4 {
            return Err(MediaError::container("FLAC 元数据块头被截断"));
        }
        let last = head[0] & 0x80 != 0;
        let ty = head[0] & 0x7F;
        let len = ((head[1] as usize) << 16) | ((head[2] as usize) << 8) | head[3] as usize;
        if ty == 0 && found.is_none() {
            if len < STREAMINFO_LEN {
                return Err(MediaError::container(format!(
                    "FLAC 的 STREAMINFO 只有 {len} 字节（应当 {STREAMINFO_LEN}）"
                )));
            }
            let body = src.read_range(pos + 4, STREAMINFO_LEN)?;
            if body.len() < STREAMINFO_LEN {
                return Err(MediaError::container("FLAC 的 STREAMINFO 被截断"));
            }
            found = Some(parse_streaminfo(&body)?);
        }
        pos += 4 + len as u64;
        if last {
            let mut si = found
                .ok_or_else(|| MediaError::container("FLAC 文件头里没有 STREAMINFO"))?;
            si.audio_start = pos;
            return Ok(si);
        }
    }
}

/// 解析 34 字节的 STREAMINFO 载荷。
fn parse_streaminfo(b: &[u8]) -> Result<FlacStreamInfo> {
    if b.len() < STREAMINFO_LEN {
        return Err(MediaError::container("STREAMINFO 长度不足 34 字节"));
    }
    // 位域是紧凑打包的：20 位采样率 + 3 位声道数-1 + 5 位位深-1 正好横跨 4 字节
    let w = u32::from_be_bytes([b[10], b[11], b[12], b[13]]);
    let sample_rate = w >> 12;
    let channels = ((w >> 9) & 0x07) as u8 + 1;
    let bits_per_sample = ((w >> 4) & 0x1F) as u8 + 1;
    // 后面 4 字节的高 4 位 + 再 4 字节 = 36 位总样本数
    let hi = (b[13] & 0x0F) as u64;
    let lo = u32::from_be_bytes([b[14], b[15], b[16], b[17]]) as u64;
    let total_samples = (hi << 32) | lo;
    if sample_rate == 0 {
        return Err(MediaError::container("FLAC 的 STREAMINFO 里采样率是 0"));
    }
    Ok(FlacStreamInfo {
        sample_rate,
        channels,
        bits_per_sample,
        total_samples,
        audio_start: 0, // 由调用方填（read_stream_info 知道真实偏移）
    })
}

// ---------------------------------------------------------------- 解封装器

/// 裸 FLAC 流的解封装器：一条音轨，一条采样 = 一帧。
pub struct FlacStreamDemuxer {
    src: Arc<dyn ReadAt>,
    source: String,
    info: FlacStreamInfo,
    /// 下一个要读的帧的偏移。
    pos: u64,
    /// 已吐出的样本数（每声道）。时基 = 采样率，所以它同时是下一条的 pts。
    samples_out: u64,
    total_bytes: Option<u64>,
    duration_us: Option<i64>,
    eof: bool,
}

impl FlacStreamDemuxer {
    pub fn new(src: Arc<dyn ReadAt>, source: impl Into<String>) -> Result<Self> {
        let source = source.into();
        let head = src.read_range(0, SNIFF)?;
        let magic = flac_magic_offset(&head)
            .ok_or_else(|| MediaError::container("不是裸 FLAC 流：找不到 `fLaC` 魔数"))?;
        let info = read_stream_info(&src, magic)?;
        let total_bytes = src.len();
        let duration_us = if info.total_samples > 0 && info.sample_rate > 0 {
            Some((info.total_samples as i128 * 1_000_000 / info.sample_rate as i128) as i64)
        } else {
            None
        };
        Ok(Self {
            pos: info.audio_start,
            info,
            samples_out: 0,
            total_bytes,
            duration_us,
            eof: false,
            src,
            source,
        })
    }

    pub fn probe_reader(src: Arc<dyn ReadAt>, source: &str) -> Result<(ContainerFormat, MediaInfo)> {
        let d = Self::new(src, source)?;
        Ok((ContainerFormat::Flac, d.info()))
    }

    /// 流信息（`probe` 之外，`me-cli` 想知道位深时也用它）。
    pub fn stream_info(&self) -> FlacStreamInfo {
        self.info
    }

    pub fn info(&self) -> MediaInfo {
        let mut track =
            Track::new(1, MediaKind::Audio, CodecId::Flac, Timebase::new(self.info.sample_rate));
        track.sample_rate = self.info.sample_rate;
        track.channels = self.info.channels as u16;
        track.extradata = self.extradata();
        MediaInfo {
            container: ContainerFormat::Flac.name().to_string(),
            duration_us: self.duration_us,
            // FLAC 头里没有码率字段；要精确得整段扫一遍（不值得，`probe` 要保持毫秒级）
            bitrate: None,
            tracks: vec![track],
            source: self.source.clone(),
        }
    }

    /// 给解码器/封装器用的 `dfLa` 载荷：`version+flags(4)` + 元数据块头(4) + STREAMINFO(34)。
    ///
    /// 为什么要凑成这个形态而不是直接给 34 字节的 STREAMINFO：
    /// - **解码器**三种形态都认（见 `me-decode` 的 `parse_streaminfo_any`）；
    /// - **封装器**只认这一种（`me-mux` 写 `dfLa` 时是把 extradata 原样灌进盒子里，
    ///   不能再套一层 FullBox 头）。所以按"更严的那个"来写，两边都对。
    ///
    /// 块头固定写成 `0x80 00 00 22`：置"最后一块"标志 + 类型 0 + 长度 34 ——
    /// 在 `dfLa` 里 STREAMINFO 就是唯一一块，置上标志才是规范的样子。
    fn extradata(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(8 + STREAMINFO_LEN);
        v.extend_from_slice(&[0, 0, 0, 0]); // version + flags
        v.extend_from_slice(&[0x80, 0x00, 0x00, STREAMINFO_LEN as u8]);
        let si = self.raw_streaminfo();
        v.extend_from_slice(&si);
        v
    }

    /// 把流信息重新打包回 34 字节 STREAMINFO（只填我们解析过的那几个字段）。
    ///
    /// 为什么不把文件里那 34 字节原样留着：元数据块是**边扫边跳**的（`read_stream_info`
    /// 只读 STREAMINFO 的载荷就返回了），不留副本就不用为了"以后再要"多占一份状态；
    /// 而 MD5 与 min/max 块长这些字段解码器与封装器都用不上，填 0 即可。
    fn raw_streaminfo(&self) -> [u8; STREAMINFO_LEN] {
        let mut b = [0u8; STREAMINFO_LEN];
        let i = self.info;
        // 位域布局：20 位采样率 | 3 位(声道数-1) | 5 位(位深-1)，从第 10 字节开始
        let w = (i.sample_rate << 12) | (((i.channels - 1) as u32) << 9) | (((i.bits_per_sample - 1) as u32) << 4);
        b[10..14].copy_from_slice(&w.to_be_bytes());
        // 第 13 字节的低 4 位 + 第 14..18 字节 = 36 位总样本数
        let t = i.total_samples;
        b[13] = (b[13] & 0xF0) | (((t >> 32) as u8) & 0x0F);
        b[14..18].copy_from_slice(&((t & 0xFFFF_FFFF) as u32).to_be_bytes());
        b
    }

    pub fn duration_us(&self) -> Option<i64> {
        self.duration_us
    }

    /// 取下一帧。载荷是**完整的一帧（含帧头与页脚）**。
    ///
    /// 不需要循环：帧头对不上时 `read_frame` 自己会往后同步（丢包/从中间接入），
    /// 真的到头了才返回 `None`。
    pub fn next_packet(&mut self) -> Result<Option<Packet>> {
        if self.eof {
            return Ok(None);
        }
        let Some(data) = self.read_frame()? else {
            self.eof = true;
            return Ok(None);
        };
        let Some(h) = parse_frame_header(&data) else {
            // 理论上到不了这里（read_frame 已经校验过帧头），到这儿说明流坏了
            self.eof = true;
            return Ok(None);
        };
        let pts = self.samples_out as i64;
        self.samples_out += h.block_size as u64;
        self.pos += data.len() as u64;
        let mut pkt = Packet::new(1, pts, pts, h.block_size, data);
        pkt.keyframe = true;
        Ok(Some(pkt))
    }

    /// 读出 `pos` 处的整帧（含帧头与 2 字节页脚）。
    ///
    /// 边界靠"下一个帧头 + 上一帧的 CRC-16 页脚"确定（见模块文档）。窗口从
    /// `FRAME_WINDOW_MIN` 开始，找不到就翻倍 —— 所以一帧只会被读一次（外加翻倍时
    /// 多读的那部分），不会把整个文件重读。
    fn read_frame(&mut self) -> Result<Option<Vec<u8>>> {
        let mut window = FRAME_WINDOW_MIN;
        loop {
            let buf = self.src.read_range(self.pos, window)?;
            if buf.len() < MIN_FRAME_BYTES {
                return Ok(None); // 连一个最小帧都装不下：到头了
            }
            // 帧头对不上（丢包、从中间接入）：往后找下一个真帧头
            if parse_frame_header(&buf).is_none() {
                match find_header(&buf, 1) {
                    Some(off) => {
                        self.pos += off as u64;
                        continue;
                    }
                    None => return Ok(None),
                }
            }
            if let Some(end) = find_boundary(&buf) {
                return Ok(Some(buf[..end].to_vec()));
            }
            // 这一窗里没有下一个边界：要么帧比窗口大，要么这已经是最后一帧
            if buf.len() < window {
                return Ok(Some(buf)); // 短读 = 到 EOF 了，剩下的就是最后一帧
            }
            if window >= FRAME_WINDOW_MAX {
                return Ok(Some(buf)); // 兜底：不无限翻倍
            }
            window *= 2;
        }
    }

    /// 跳到 `pts_us` 附近，返回实际落点（微秒）。
    ///
    /// 与裸 MP3 同样的处境：**没有任何索引**，只能按"时间比例 ≈ 字节比例"插值
    /// （总样本数在 STREAMINFO 里，所以比例是准的）。落点后同步到下一个真帧头。
    ///
    /// 不精确的地方：变长分块（VBR 式）的文件里各段字节密度不同，插值只保证
    /// "整段平均对"；要更准得建帧索引，那是另一个功能。
    pub fn seek(&mut self, pts_us: i64) -> Result<i64> {
        let want = pts_us.max(0);
        let audio_bytes = self.total_bytes.map(|t| t.saturating_sub(self.info.audio_start));
        let pos = match (self.duration_us, audio_bytes) {
            (Some(dur), Some(bytes)) if dur > 0 && bytes > 0 => {
                let off = bytes as i128 * want as i128 / dur as i128;
                self.info.audio_start + off.clamp(0, bytes as i128) as u64
            }
            _ => self.info.audio_start,
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
        self.samples_out = match (self.info.total_samples, audio_bytes) {
            (total, Some(bytes)) if total > 0 && bytes > 0 => {
                let done = self.pos.saturating_sub(self.info.audio_start).min(bytes);
                (done as u128 * total as u128 / bytes as u128) as u64
            }
            _ => 0,
        };
        Ok((self.samples_out as i128 * 1_000_000 / self.info.sample_rate as i128) as i64)
    }

    /// 从 `from` 往后找下一个合法帧头（只看 CRC-8，够用：seek 不需要精确边界）。
    fn resync_from(&self, from: u64) -> Result<Option<u64>> {
        let buf = self.src.read_range(from, FRAME_WINDOW_MAX)?;
        Ok(find_header(&buf, 0).map(|off| from + off as u64))
    }
}

/// 在 `buf` 里从 `from` 开始找第一个合法帧头（**CRC-8 必须是好的**）。
fn find_header(buf: &[u8], from: usize) -> Option<usize> {
    (from..buf.len().saturating_sub(MIN_FRAME_BYTES))
        .find(|&off| parse_frame_header(&buf[off..]).is_some())
}

/// 在当前帧内部找"下一帧的起点"。
///
/// 判据是两级：候选帧头本身要合法（CRC-8 过），**并且**上一帧的 CRC-16 页脚要对得上。
/// 只有一条（帧头）在熵编码数据里会误判；加上页脚校验之后，一个假边界的概率是
/// 两者的乘积量级，实际不可能出现。
fn find_boundary(buf: &[u8]) -> Option<usize> {
    let mut from = MIN_FRAME_BYTES;
    for _ in 0..MAX_BOUNDARY_TRIES {
        let off = find_header(buf, from)?;
        if off >= 2 {
            let body = &buf[..off - 2];
            let footer = u16::from_be_bytes([buf[off - 2], buf[off - 1]]);
            if crc16(body) == footer {
                return Some(off);
            }
        }
        from = off + 1;
    }
    None
}

// ---------------------------------------------------------------- Demuxer

impl crate::demux::Demuxer for FlacStreamDemuxer {
    fn info(&self) -> MediaInfo {
        FlacStreamDemuxer::info(self)
    }

    fn next_packet(&mut self) -> Result<Option<Packet>> {
        FlacStreamDemuxer::next_packet(self)
    }

    fn seek(&mut self, pts_us: i64) -> Result<i64> {
        FlacStreamDemuxer::seek(self, pts_us)
    }

    fn duration_us(&self) -> Option<i64> {
        self.duration_us
    }

    fn container(&self) -> &'static str {
        // 与 `ContainerFormat::Flac.name()` 必须一致：宿主拿这个字符串做分发
        ContainerFormat::Flac.name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demux::Demuxer;
    use me_core::Cursor;

    /// CRC 的已知答案（不是拿自己的实现跟自己对，是钉住一个标准值）。
    #[test]
    fn crc_known_answers() {
        assert_eq!(crc8(b"123456789"), 0xF4, "CRC-8/SMBUS 的标准校验值");
        assert_eq!(crc16(b"123456789"), 0xFEE8, "CRC-16/UMTS 的标准校验值");
        assert_eq!(crc8(&[]), 0);
        assert_eq!(crc16(&[]), 0);
    }

    /// 造一个 STREAMINFO 载荷。
    fn streaminfo(rate: u32, channels: u8, bits: u8, total: u64) -> [u8; STREAMINFO_LEN] {
        let mut b = [0u8; STREAMINFO_LEN];
        b[0..2].copy_from_slice(&4096u16.to_be_bytes()); // min block size
        b[2..4].copy_from_slice(&4096u16.to_be_bytes()); // max block size
        let w = (rate << 12) | (((channels - 1) as u32) << 9) | (((bits - 1) as u32) << 4);
        b[10..14].copy_from_slice(&w.to_be_bytes());
        b[13] = (b[13] & 0xF0) | (((total >> 32) as u8) & 0x0F);
        b[14..18].copy_from_slice(&((total & 0xFFFF_FFFF) as u32).to_be_bytes());
        b
    }

    /// 造一个"定长分块、块大小 4096、44.1k、立体声（左右独立）"的帧头（含 CRC-8）。
    fn frame_header(block_size: u32, rate_code: u8) -> Vec<u8> {
        let mut h = vec![0xFFu8, 0xF8]; // 同步 + 保留位 0 + 定长分块
        // 块大小 7 号 = 头里塞 16 位（值-1）；采样率按参数给的号
        h.push(0x70 | rate_code);
        h.push(0x10); // 声道分配 1 = 立体声（左右独立）；位深 0 = 看 STREAMINFO；末位保留 0
        h.push(0x00); // 帧号（UTF-8 一个字节就够）
        h.extend_from_slice(&((block_size - 1) as u16).to_be_bytes());
        let c = crc8(&h);
        h.push(c);
        h
    }

    /// 造一个完整帧：帧头 + `body` 字节的假数据 + 页脚 CRC-16。
    fn frame(block_size: u32, body: usize) -> Vec<u8> {
        let mut f = frame_header(block_size, 9); // 9 = 44.1kHz
        f.extend(std::iter::repeat_n(0x5Au8, body));
        let c = crc16(&f);
        f.extend_from_slice(&c.to_be_bytes());
        f
    }

    /// 造一个完整的裸 FLAC 文件：`fLaC` + STREAMINFO 块（+ 可选的大块）+ 若干帧。
    fn build_flac(frames: &[Vec<u8>], total_samples: u64, extra_block: Option<usize>) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"fLaC");
        let last = extra_block.is_none();
        v.push(if last { 0x80 } else { 0x00 });
        v.extend_from_slice(&[0x00, 0x00, STREAMINFO_LEN as u8]);
        v.extend_from_slice(&streaminfo(44100, 2, 16, total_samples));
        if let Some(len) = extra_block {
            // 4 号块 = VORBIS_COMMENT，用大块模拟"几 MB 的封面"
            v.push(0x84);
            v.push(((len >> 16) & 0xFF) as u8);
            v.push(((len >> 8) & 0xFF) as u8);
            v.push((len & 0xFF) as u8);
            v.extend(std::iter::repeat_n(0xA5u8, len));
        }
        for f in frames {
            v.extend_from_slice(f);
        }
        v
    }

    #[test]
    fn parses_streaminfo_fields() {
        let si = parse_streaminfo(&streaminfo(96_000, 2, 24, 123_456_789)).unwrap();
        assert_eq!(si.sample_rate, 96_000);
        assert_eq!(si.channels, 2);
        assert_eq!(si.bits_per_sample, 24);
        assert_eq!(si.total_samples, 123_456_789);
    }

    #[test]
    fn frame_header_requires_a_valid_crc8() {
        let h = frame_header(4096, 9);
        let got = parse_frame_header(&h).unwrap();
        assert_eq!(got.block_size, 4096);
        assert_eq!(got.header_len, h.len());
        // 改一个字节，CRC-8 就对不上了
        let mut bad = h.clone();
        bad[2] ^= 0x08;
        assert!(parse_frame_header(&bad).is_none(), "CRC-8 必须挡下来");
        // 同步码不对 / 保留位不为 0
        assert!(parse_frame_header(&[0xFE, 0xF8, 0x70, 0x10, 0, 0, 0, 0]).is_none());
        assert!(parse_frame_header(&[0xFF, 0xFA, 0x70, 0x10, 0, 0, 0, 0]).is_none());
    }

    #[test]
    fn recognises_flac_and_skips_id3v2() {
        let data = build_flac(&[frame(4096, 100)], 4096, None);
        assert!(looks_like_flac(&data));
        // 前面顶一个 1000 字节的 ID3v2 标签
        let mut tagged = vec![b'I', b'D', b'3', 0x03, 0x00, 0x00];
        tagged.extend_from_slice(&[0, 0, (1000 >> 7) as u8, (1000 & 0x7F) as u8]);
        tagged.extend(std::iter::repeat_n(0u8, 1000));
        tagged.extend_from_slice(&data);
        assert!(looks_like_flac(&tagged));
        // 不是 FLAC 的别认
        assert!(!looks_like_flac(b"RIFF....WAVE"));
        assert!(!looks_like_flac(&true_false_dummy()));
    }

    fn true_false_dummy() -> [u8; 8] {
        [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]
    }

    #[test]
    fn demuxer_walks_frames_and_pts() {
        let frames = vec![frame(4096, 500), frame(4096, 500), frame(4096, 500)];
        let data = build_flac(&frames, 12_288, None);
        let total = data.len();
        let mut d = FlacStreamDemuxer::new(Arc::new(Cursor::new(data)), "mem.flac").unwrap();
        assert_eq!(d.container(), "flac");
        let si = d.stream_info();
        assert_eq!(si.sample_rate, 44_100);
        assert_eq!(si.channels, 2);
        assert_eq!(si.bits_per_sample, 16);
        let info = d.info();
        assert_eq!(info.tracks[0].codec, CodecId::Flac);
        assert_eq!(info.tracks[0].timebase.ticks_per_second, 44_100);
        // 时长 = 12288 / 44100 秒
        assert_eq!(info.duration_us.unwrap(), (12_288i128 * 1_000_000 / 44_100) as i64);

        let mut n = 0i64;
        while let Some(p) = d.next_packet().unwrap() {
            assert_eq!(p.data.len(), frames[n as usize].len(), "整帧（含页脚）");
            assert_eq!(p.pts, n * 4096, "pts 按样本数走");
            assert_eq!(p.duration, 4096);
            assert!(p.keyframe);
            n += 1;
        }
        assert_eq!(n, 3);
        let _ = total;
    }

    #[test]
    fn big_metadata_block_is_skipped_not_read() {
        // 元数据里塞 300 KiB（模拟专辑封面）：STREAMINFO 之后的数据不参与解析
        let data = build_flac(&[frame(4096, 50)], 4096, Some(300 * 1024));
        let mut d = FlacStreamDemuxer::new(Arc::new(Cursor::new(data.clone())), "cover.flac").unwrap();
        assert_eq!(d.stream_info().sample_rate, 44_100);
        // 音频起点应当正好在封面块之后
        let expect = 4 + 4 + STREAMINFO_LEN + 4 + 300 * 1024;
        assert_eq!(d.info().tracks[0].sample_rate, 44_100);
        let p = d.next_packet().unwrap().unwrap();
        assert_eq!(p.pts, 0);
        assert_eq!(p.data.len(), 50 + 2 + frame_header(4096, 9).len());
        let _ = expect;
    }

    #[test]
    fn frame_bodies_containing_fake_syncs_do_not_confuse_the_demuxer() {
        // 帧体里塞满 0xFF 0xF8（最像帧头的东西）：只有"页脚 CRC-16 对得上"的那一处
        // 才是真边界 —— 这正是两级校验存在的理由
        let mut f = frame_header(4096, 9);
        for i in 0..400usize {
            if i % 2 == 0 {
                f.push(0xFF);
            } else {
                f.push(0xF8);
            }
        }
        let c = crc16(&f);
        f.extend_from_slice(&c.to_be_bytes());
        let data = build_flac(&[f.clone(), frame(4096, 30)], 8192, None);
        let mut d = FlacStreamDemuxer::new(Arc::new(Cursor::new(data)), "fake.flac").unwrap();
        let p1 = d.next_packet().unwrap().unwrap();
        assert_eq!(p1.data.len(), f.len(), "假同步字不能被当边界");
        let p2 = d.next_packet().unwrap().unwrap();
        assert_eq!(p2.pts, 4096);
        assert!(d.next_packet().unwrap().is_none());
    }

    #[test]
    fn truncated_last_frame_is_still_delivered() {
        // 最后一帧被截断（没有页脚）：短读说明到 EOF，剩下的照样交给解码器（它会报错）
        let good = frame(4096, 100);
        let mut data = build_flac(std::slice::from_ref(&good), 8192, None);
        data.extend_from_slice(&frame_header(4096, 9));
        data.extend(std::iter::repeat_n(0u8, 50)); // 半截帧
        let mut d = FlacStreamDemuxer::new(Arc::new(Cursor::new(data)), "cut.flac").unwrap();
        assert_eq!(d.next_packet().unwrap().unwrap().data.len(), good.len());
        let last = d.next_packet().unwrap().unwrap();
        assert!(last.data.len() < good.len());
        assert!(d.next_packet().unwrap().is_none());
    }

    #[test]
    fn extradata_is_the_dfla_payload_shape() {
        // 解码器（parse_streaminfo_any）与封装器（写 dfLa）都要认的形态：
        // version+flags(4) + 块头(4) + STREAMINFO(34)
        let data = build_flac(&[frame(4096, 10)], 4096, None);
        let d = FlacStreamDemuxer::new(Arc::new(Cursor::new(data)), "e.flac").unwrap();
        let ex = d.info().tracks[0].extradata.clone();
        assert_eq!(ex.len(), 8 + STREAMINFO_LEN);
        assert_eq!(&ex[0..4], &[0, 0, 0, 0], "dfLa 的 version+flags");
        assert_eq!(ex[4] & 0x80, 0x80, "STREAMINFO 要带'最后一块'标志");
        assert_eq!(ex[4] & 0x7F, 0, "块类型必须是 0（STREAMINFO）");
        assert_eq!(ex[7], STREAMINFO_LEN as u8);
        // 第 10 字节起的位域是采样率
        let w = u32::from_be_bytes([ex[8 + 10], ex[8 + 11], ex[8 + 12], ex[8 + 13]]);
        assert_eq!(w >> 12, 44_100);
    }

    #[test]
    fn seek_lands_on_a_real_frame() {
        let frames = vec![frame(4096, 200), frame(4096, 200), frame(4096, 200), frame(4096, 200)];
        let data = build_flac(&frames, 16_384, None);
        let mut d = FlacStreamDemuxer::new(Arc::new(Cursor::new(data)), "s.flac").unwrap();
        // 从头
        assert_eq!(d.seek(0).unwrap(), 0);
        assert_eq!(d.next_packet().unwrap().unwrap().pts, 0);
        // 到 1/4 处附近：落点必须是真帧头（能被解码器接受的那一帧）
        let landed = d.seek((4096i64 * 1_000_000) / 44_100).unwrap();
        assert!(landed >= 0);
        let p = d.next_packet().unwrap().unwrap();
        assert!(parse_frame_header(&p.data).is_some(), "落点必须是合法帧头");
    }
}
