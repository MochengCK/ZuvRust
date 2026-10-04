//! 裸 MPEG 音频流（`.mp3`）—— 引擎认的第三种输入形态。
//!
//! 严格说它**不是容器**：没有轨道表、没有索引，就是一串首尾相接的音频帧，
//! 前面顶着一个可有可无的 ID3v2 标签。之所以还给它一个容器位、照 [`Demuxer`] 的
//! 接口实现，是因为**现实里绝大多数音频文件就是这个样子**：一个 `.mp3` 要能被
//! `probe` / `decode --audio` / `play` 认下来，必须有人告诉上层"这是码流、
//! 一条采样是一帧、轨道参数是多少、总时长大概多久"。与其让每个调用点各写一套，
//! 不如在这里收敛成"只有一条音轨的解封装器"。
//!
//! ## 为什么裸流也值得做
//!
//! 没有它，自研 MP3 解码器在真实链路里**几乎够不着**：TS / fMP4 里出现 MP3 轨的
//! 场合很少（HLS/DASH 基本只用 AAC），而 `.mp3` 恰恰是用户手里最常见的音频文件。
//! 于是 `zuvrust decode foo.mp3 --audio` 会直接死在"无法识别的容器"上。
//!
//! ## 认哪些、不认哪些
//!
//! | 情况 | 处理 |
//! |---|---|
//! | Layer III（MPEG-1 / 2 / 2.5） | ✅ 认，码流交给 `me-decode` 的自研 MP3 解码器 |
//! | Layer I / Layer II | ❌ 明确拒绝（与解码器口径一致：那是另一套帧格式） |
//! | ID3v2 标签（含 footer） | ✅ 按 synchsafe 长度整段跳过 |
//! | Xing / Info 头 | ✅ 用它拿精确的帧数与字节数 → 精确时长（VBR 文件必需） |
//! | ID3v1（末尾 128 字节） | 不用管：走到最后一帧时帧长不够，自然停下 |
//! | free format（`bitrate_index == 0`） | ❌ 帧长不由码流字段决定，猜不出来 |
//!
//! [`Demuxer`]: crate::demux::Demuxer

use std::sync::Arc;

use me_core::{CodecId, MediaError, MediaInfo, MediaKind, Packet, ReadAt, Result, Timebase, Track};

use crate::probe::ContainerFormat;

/// Layer III 的比特率表（kbps，下标 = 帧头里的 4 位 `bitrate_index`）。
///
/// 下标 0 是 free format、15 是非法值，都留 0 当哨兵：**调用方必须显式拒绝**
/// （0 在表里是合法位置，不能靠"查表得到 0"来当解析失败）。
const BITRATES_V1: [u32; 16] = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0];
const BITRATES_LSF: [u32; 16] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0];

/// 采样率：三档版本 × 3 个档位（MPEG-2/2.5 是在 MPEG-1 的基础上折半、再折半）。
const SAMPLE_RATES: [[u32; 3]; 3] = [
    [44100, 48000, 32000], // MPEG-1
    [22050, 24000, 16000], // MPEG-2
    [11025, 12000, 8000],  // MPEG-2.5
];

/// 探测窗口：定位第一帧时先读这么多。
///
/// 4 KiB 与容器探测（[`crate::probe`]）用同一个数量级 —— 边下边播时多读一点就是多等一点，
/// 而正常文件的第一帧要么在偏移 0、要么就贴在 ID3v2 标签后面。
const SNIFF: usize = 4096;

/// 兜底扫描上限：既没标签、4 KiB 里也找不到帧时，最多再往后找这么多。
const FIND_FIRST_SCAN: usize = 64 * 1024;

/// 对不齐时最多往后扫多少字节找下一个同步字。
const RESYNC_SCAN: usize = 64 * 1024;

/// 一个 Layer III 帧头解出来的字段。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mp3FrameHeader {
    /// MPEG-2 / 2.5（LSF）：每帧 576 个样本而不是 1152。
    pub lsf: bool,
    pub sample_rate: u32,
    pub bitrate_kbps: u32,
    /// 1 = 单声道，2 = 立体声（含联合立体声与双声道）。
    pub channels: u8,
    /// 这一帧在码流里占多少字节（含 4 字节帧头、含 padding）。
    pub frame_len: usize,
    /// 这一帧每声道的样本数。
    pub samples: u32,
    /// 侧信息有多少字节（Xing 头就贴在它后面）。
    pub side_info_len: usize,
}

impl Mp3FrameHeader {
    /// Xing / Info 头相对帧首的偏移 = 4 + CRC + 侧信息。
    pub fn xing_offset(&self, crc: usize) -> usize {
        4 + crc + self.side_info_len
    }
}

/// 帧头后面有没有 2 字节 CRC：`protection_bit == 0` 表示有。
fn crc_len(header_byte1: u8) -> usize {
    if header_byte1 & 1 == 0 {
        2
    } else {
        0
    }
}

/// 解析一个 Layer III 帧头；不是合法帧头就返回 `None`。
///
/// **只认 Layer III**：与自研解码器口径一致，Layer I/II 是另一套帧格式，
/// 认了也解不出来（不如在探测阶段就说不认识，让调用方看到真实原因）。
pub fn parse_frame_header(b: &[u8]) -> Option<Mp3FrameHeader> {
    if b.len() < 4 {
        return None;
    }
    // 11 位同步字：0xFF + 高 3 位为 111
    if b[0] != 0xFF || (b[1] & 0xE0) != 0xE0 {
        return None;
    }
    let version_bits = (b[1] >> 3) & 0x3;
    if version_bits == 0b01 {
        return None; // 保留值
    }
    // 00 = MPEG-2.5、10 = MPEG-2、11 = MPEG-1
    let ver_row = match version_bits {
        0b00 => 2,
        0b10 => 1,
        _ => 0,
    };
    let lsf = ver_row != 0;
    if (b[1] >> 1) & 0x3 != 0b01 {
        return None; // 不是 Layer III
    }
    let bitrate_index = ((b[2] >> 4) & 0x0F) as usize;
    let sample_rate_index = ((b[2] >> 2) & 0x3) as usize;
    if sample_rate_index == 3 {
        return None; // 保留值
    }
    let bitrate_kbps = if lsf { BITRATES_LSF[bitrate_index] } else { BITRATES_V1[bitrate_index] };
    if bitrate_kbps == 0 {
        return None; // 0 = free format、15 = 非法
    }
    let sample_rate = SAMPLE_RATES[ver_row][sample_rate_index];
    let padding = (b[2] >> 1) & 1;
    let channels: u8 = if (b[3] >> 6) & 0x3 == 0b11 { 1 } else { 2 };
    // 帧长：MPEG-1 一帧 1152 个样本（144×比特率/采样率），LSF 是 576（72×）；
    // padding 位表示这一帧多垫一个字节，让平均帧长贴合码率
    let mult: u32 = if lsf { 72 } else { 144 };
    let frame_len = (mult * bitrate_kbps * 1000 / sample_rate) as usize + padding as usize;
    if frame_len < 4 {
        return None;
    }
    let samples = if lsf { 576 } else { 1152 };
    let side_info_len = match (lsf, channels == 1) {
        (false, true) => 17,
        (false, false) => 32,
        (true, true) => 9,
        (true, false) => 17,
    };
    Some(Mp3FrameHeader {
        lsf,
        sample_rate,
        bitrate_kbps,
        channels,
        frame_len,
        samples,
        side_info_len,
    })
}

/// 在一段"可能从任意位置开始"的字节里找第一个帧头。
///
/// TS 的 PES 载荷就用它：MP3 在 TS 里没有任何描述符可读（不像 AAC 有 ADTS 头），
/// 采样率与声道数只存在于**每一帧的帧头**里。顺手解一下第一帧，`probe` 才能报出
/// 真实参数 —— 否则界面上那条音轨会显示 0 Hz / 0 声道。
///
/// 两道判据，因为载荷形态有两种：
/// 1. **一个包里有多帧** → 找"一对自洽的帧"（最稳，不靠猜）；
/// 2. **包里只有一帧**（引擎自己的 TS 封装就是"一条 PES 一帧"，实测每包 960 字节）
///    → 第二帧根本不存在，"成对"永远不成立，于是换成"这一帧的帧长正好把剩下的
///    字节用完"。这条约束同样能挡掉音频数据里那些恰好长得像帧头的字节。
pub(crate) fn first_frame_header(buf: &[u8]) -> Option<Mp3FrameHeader> {
    for off in 0..buf.len().saturating_sub(4) {
        if frame_pair_at(buf, off) {
            return parse_frame_header(&buf[off..]);
        }
    }
    for off in 0..buf.len().saturating_sub(4) {
        if let Some(h) = parse_frame_header(&buf[off..]) {
            if buf.len() - off == h.frame_len {
                return Some(h);
            }
        }
    }
    None
}

/// `off` 处是不是"一对自洽的帧"。
///
/// **为什么要看两帧**：单个帧头只有 11 位同步字加几个合法字段，随机数据撞上的概率
/// 不低（`0xFF Ex` 这种字节对在二进制文件里很常见）。要求"第二帧正好落在第一帧帧长
/// 算出来的位置上、且版本与采样率一致"之后，误判率降到可忽略 —— 探测阶段敢把
/// MPEG 音频放在最后兜底，靠的就是这一条。
///
/// 刻意**不要求声道模式一致**：真的存在逐帧切换声道模式的码流（`l3-he_mode` 就是
/// 单声道/立体声/联合立体声/双声道混着来），卡这一条会把它当成垃圾。
pub fn frame_pair_at(bytes: &[u8], off: usize) -> bool {
    let Some(first) = bytes.get(off..).and_then(parse_frame_header) else {
        return false;
    };
    let Some(second) = bytes.get(off + first.frame_len..).and_then(parse_frame_header) else {
        return false;
    };
    first.lsf == second.lsf && first.sample_rate == second.sample_rate
}

/// 头部是不是 ID3v2 标签；是就返回整个标签占多少字节。
///
/// 长度字段是 **synchsafe**：4 个字节各只用低 7 位（最高位恒 0，专门用来避免标签
/// 长度里出现 `0xFF` 而破坏音频同步字）。按普通大端读会得到一个约 2 倍偏大的值，
/// 于是"跳过标签"变成"跳过标签 + 一大截音频"——症状是第一帧永远对不上，
/// 整个文件被判成不认识。
///
/// `pub(crate)`：ID3v2 是 MP3 世界的标签，但规范允许它出现在 `fLaC` 之前
/// （现实中真有这种文件），裸 FLAC 那条路也要用它。
pub(crate) fn id3v2_len(head: &[u8]) -> Option<usize> {
    if head.len() < 10 || &head[0..3] != b"ID3" {
        return None;
    }
    if !(2..=4).contains(&head[3]) {
        return None; // 版本字段不是 2/3/4，不当标签处理
    }
    let flags = head[5];
    if head[6..10].iter().any(|b| b & 0x80 != 0) {
        return None; // 不是 synchsafe，说明这 4 个字节不是长度
    }
    let size = ((head[6] as usize) << 21)
        | ((head[7] as usize) << 14)
        | ((head[8] as usize) << 7)
        | head[9] as usize;
    // 10 字节标签头 + 正文；flag bit4 表示后面还有 10 字节 footer
    Some(10 + size + if flags & 0x10 != 0 { 10 } else { 0 })
}

/// 在 `buf` 里找音频帧，返回第一帧在 `buf` 里的偏移。
///
/// 起点是"标签之后"：ID3v2 正文里是文本与图片，与音频帧无关，
/// 在里面扫同步字只是白费力气（还可能撞上巧合）。
///
/// `require_pair` 是**两道闸的分工**：
/// - 探测（[`looks_like_mpeg_audio`]）要求一对自洽的帧 —— 那时还不知道这是什么文件，
///   必须严；单个帧头太容易在随机数据里撞上；
/// - 解封装器打开时只要求一个合法帧头 —— 走到这里说明探测已经确认过是 MPEG 音频了，
///   再卡"必须有第二帧"会把**只截到一帧的文件**判成打不开（明明能解）。
fn first_frame_offset(buf: &[u8], require_pair: bool) -> Option<usize> {
    let start = id3v2_len(buf).unwrap_or(0);
    let scan = buf.len().min(start + FIND_FIRST_SCAN);
    (start..scan.saturating_sub(4)).find(|&off| {
        let has_header = parse_frame_header(&buf[off..]).is_some();
        if require_pair {
            has_header && frame_pair_at(buf, off)
        } else {
            has_header
        }
    })
}

/// 只按头部字节判断"是不是裸 MPEG 音频流"（探测用）。
///
/// 两种情况都算：
/// 1. 开头（或 ID3v2 之后）就是一对自洽的帧 —— 强证据；
/// 2. 开头是 ID3v2 标签、但**标签长到 4 KiB 的探测窗口里看不到第一帧**（专辑封面
///    几 MB 很常见）—— 这时只能靠签名认。`ID3` + 版本合法已经足够强：ID3v2 标签
///    只长在 MPEG 音频前面，撞上别的格式的概率可以忽略。
pub fn looks_like_mpeg_audio(head: &[u8]) -> bool {
    if first_frame_offset(head, true).is_some() {
        return true;
    }
    matches!(id3v2_len(head), Some(len) if len >= head.len())
}

/// 定位第一帧：返回（含第一帧的缓冲、第一帧的**绝对**偏移、它在缓冲里的下标）。
///
/// 分三步是为了**别在边下边播时多读**：绝大多数文件第一帧就在开头 4 KiB 里；
/// 只有"ID3v2 长过窗口"（专辑封面）才跳到标签末尾再读 4 KiB（而不是傻读 64 KiB ——
/// 对几 MB 的标签来说，`read_range` 会一直等到那些字节下完）；
/// 两者都不成立时（没标签、开头有杂物）才把窗口放大。
fn locate_first_frame(src: &Arc<dyn ReadAt>) -> Result<(Vec<u8>, u64, usize)> {
    let head = src.read_range(0, SNIFF)?;
    if let Some(off) = first_frame_offset(&head, false) {
        return Ok((head, off as u64, off));
    }
    if let Some(skip) = id3v2_len(&head) {
        // 标签长过窗口：直接跳到标签末尾（而不是把标签整个读进来）
        let tail = src.read_range(skip as u64, SNIFF)?;
        if let Some(off) = first_frame_offset(&tail, false) {
            return Ok((tail, skip as u64 + off as u64, off));
        }
    }
    let wide = src.read_range(0, FIND_FIRST_SCAN)?;
    match first_frame_offset(&wide, false) {
        Some(off) => Ok((wide, off as u64, off)),
        None => Err(MediaError::container(
            "不是裸 MPEG 音频流：找不到合法的 Layer III 帧头\
             （Layer I/II 与 free format 按约定都不认）",
        )),
    }
}

// ---------------------------------------------------------------- 解封装器

/// 裸 MPEG 音频流的解封装器：一条音轨，一条采样 = 一帧。
pub struct MpegAudioDemuxer {
    src: Arc<dyn ReadAt>,
    source: String,
    /// 第一个音频帧在文件里的偏移（跳过 ID3v2）。
    audio_start: u64,
    /// 下一个要读的帧的偏移。
    pos: u64,
    /// 第一条帧头（轨道参数与"每帧多少样本"由它定）。
    first: Mp3FrameHeader,
    /// 已经吐出的样本数。时基就是采样率，所以它同时是下一条的 pts。
    samples_out: u64,
    /// 总字节数（`ReadAt` 不知道就是 None，直播/流式）。
    total_bytes: Option<u64>,
    /// 总样本数（Xing 的确切帧数，或按码率估的）；未知为 None。
    total_samples: Option<u64>,
    /// 平均码率（bit/s）：`info().bitrate` 与 seek 的换算都用它。
    avg_bitrate: u64,
    duration_us: Option<i64>,
    eof: bool,
}

impl MpegAudioDemuxer {
    pub fn new(src: Arc<dyn ReadAt>, source: impl Into<String>) -> Result<Self> {
        let source = source.into();
        let (buf, abs, off) = locate_first_frame(&src)?;
        let first = parse_frame_header(&buf[off..])
            .ok_or_else(|| MediaError::container("第一个音频帧头解析失败"))?;

        // Xing / Info：贴在第一条帧的侧信息后面（那一帧本身是个"空帧"，只装元信息）
        let fs = &buf[off..];
        let crc = crc_len(fs[1]);
        let xing = parse_xing(fs.get(first.xing_offset(crc)..).unwrap_or(&[]));

        let total_bytes = src.len();
        let audio_bytes = total_bytes.map(|t| t.saturating_sub(abs));
        let duration_us = match (xing.frames, xing.bytes.or(audio_bytes)) {
            (Some(frames), _) if frames > 0 => {
                let samples = frames.saturating_mul(first.samples as u64);
                Some(samples as i128 * 1_000_000 / first.sample_rate as i128)
            }
            _ => audio_bytes.map(|bytes| {
                bytes as i128 * 8 * 1_000_000 / (first.bitrate_kbps as i128 * 1000)
            }),
        }
        .map(|d| d as i64);

        let total_samples = match (xing.frames, duration_us) {
            (Some(frames), _) if frames > 0 => Some(frames.saturating_mul(first.samples as u64)),
            (_, Some(us)) if us > 0 => {
                Some((us as i128 * first.sample_rate as i128 / 1_000_000) as u64)
            }
            _ => None,
        };

        // 平均码率：有字节数与时长就用"字节 ÷ 时长"（VBR 文件的真平均），否则用第一帧的
        let avg_bitrate = match (xing.bytes.or(audio_bytes), duration_us) {
            (Some(bytes), Some(us)) if us > 0 => (bytes as u128 * 8 * 1_000_000 / us as u128) as u64,
            _ => first.bitrate_kbps as u64 * 1000,
        };

        Ok(Self {
            audio_start: abs,
            pos: abs,
            first,
            samples_out: 0,
            total_bytes,
            total_samples,
            avg_bitrate,
            duration_us,
            eof: false,
            src,
            source,
        })
    }

    /// 探测 + 返回容器类型与信息（与 TS / fMP4 两个解封装器同形）。
    pub fn probe_reader(src: Arc<dyn ReadAt>, source: &str) -> Result<(ContainerFormat, MediaInfo)> {
        let d = Self::new(src, source)?;
        Ok((ContainerFormat::MpegAudio, d.info()))
    }

    pub fn info(&self) -> MediaInfo {
        let mut track =
            Track::new(1, MediaKind::Audio, CodecId::Mp3, Timebase::new(self.first.sample_rate));
        track.sample_rate = self.first.sample_rate;
        track.channels = self.first.channels as u16;
        // MP3 没有 extradata：解码器自己从每帧的帧头取参数
        MediaInfo {
            container: ContainerFormat::MpegAudio.name().to_string(),
            duration_us: self.duration_us,
            bitrate: Some(self.avg_bitrate),
            tracks: vec![track],
            source: self.source.clone(),
        }
    }

    pub fn duration_us(&self) -> Option<i64> {
        self.duration_us
    }

    /// 取下一帧。
    ///
    /// 载荷是**完整的一帧（含 4 字节帧头）**：自研解码器收的是"包里可能有半帧或多帧"
    /// 的字节流，给它完整的帧它自己会找同步字；这里不做任何隐式转换。
    pub fn next_packet(&mut self) -> Result<Option<Packet>> {
        loop {
            if self.eof {
                return Ok(None);
            }
            let mut head = [0u8; 4];
            let got = self.src.read_at(self.pos, &mut head)?;
            if got < 4 {
                self.eof = true;
                return Ok(None);
            }
            let Some(h) = parse_frame_header(&head) else {
                // 对不齐了（末尾的 ID3v1、填充、半截帧）：往后找下一个同步字
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
            let data = self.src.read_range(self.pos, h.frame_len)?;
            if data.len() < h.frame_len {
                // 短读 = 真的到末尾了（`ReadAt` 会自己等"暂时没有"的数据，不会拿短读糊弄）。
                // 半截帧丢掉：交给解码器只会在尾巴上报错。
                self.eof = true;
                return Ok(None);
            }
            let pts = self.samples_out as i64;
            self.samples_out += h.samples as u64;
            self.pos += h.frame_len as u64;
            let mut pkt = Packet::new(1, pts, pts, h.samples, data);
            // 音频帧不受别的帧影响（位水库最多往回借 511 字节，见 seek 的说明），随时可作起点
            pkt.keyframe = true;
            return Ok(Some(pkt));
        }
    }

    /// 跳到 `pts_us` 附近，返回实际落点（微秒）。
    ///
    /// 裸流里**一个索引字节都没有**，所以只能按"时间比例 ≈ 字节比例"估：
    /// 有时长（Xing 或按码率算出）就按整段线性插值，否则按平均帧长折算。
    /// 估出来的偏移**通常不是帧首**，所以落点之后一定要同步到下一个真帧首。
    ///
    /// 已知不精确的地方，写下来免得以后当 bug 查：
    /// - VBR 文件各段码率不同，线性插值只保证"整段平均对"，落点会偏零点几秒
    ///   （Xing 的 TOC 表能把误差压到一帧以内，但那是另一个功能，先不做）；
    /// - 位水库（`main_data_begin`）最多往回借 511 字节，跳到任意帧时那段历史不在手里，
    ///   解码器的做法是补 0（几毫秒的高频略有差别）。要完全避免得往前多解几帧热身，
    ///   但那样返回的落点会早于请求位置，播放器与进度条都要额外处理 —— 不值。
    pub fn seek(&mut self, pts_us: i64) -> Result<i64> {
        let want = pts_us.max(0);
        let audio_bytes = self.total_bytes.map(|t| t.saturating_sub(self.audio_start));
        let pos = match (self.duration_us, audio_bytes) {
            (Some(dur), Some(bytes)) if dur > 0 && bytes > 0 => {
                let off = bytes as i128 * want as i128 / dur as i128;
                self.audio_start + off.clamp(0, bytes as i128) as u64
            }
            _ => {
                // 没有时长（直播/流式）：按"平均一帧多少字节"折
                let frames = want as u64 * self.first.sample_rate as u64 / 1_000_000
                    / self.first.samples as u64;
                self.audio_start + frames * self.avg_frame_len()
            }
        };
        let pos = match self.total_bytes {
            Some(t) => pos.min(t),
            None => pos,
        };
        self.pos = self.resync_from(pos)?.unwrap_or(pos);
        self.eof = false;
        // 落点对应的样本数：同样按比例反推（字节偏移 ÷ 总字节 ≈ 已播比例）
        self.samples_out = match (self.total_samples, audio_bytes) {
            (Some(total), Some(bytes)) if bytes > 0 => {
                let done = self.pos.saturating_sub(self.audio_start).min(bytes);
                (done as u128 * total as u128 / bytes as u128) as u64
            }
            _ => {
                let frames = self.pos.saturating_sub(self.audio_start) / self.avg_frame_len();
                frames * self.first.samples as u64
            }
        };
        Ok((self.samples_out as i128 * 1_000_000 / self.first.sample_rate as i128) as i64)
    }

    /// 平均一帧多少字节（没有时长时的 seek 折算用）。
    fn avg_frame_len(&self) -> u64 {
        (self.avg_bitrate * self.first.samples as u64 / 8 / self.first.sample_rate as u64).max(1)
    }

    /// 从 `from` 往后找一对自洽的帧，返回第一帧的偏移。
    fn resync_from(&self, from: u64) -> Result<Option<u64>> {
        let scan = self.src.read_range(from, RESYNC_SCAN)?;
        for off in 0..scan.len().saturating_sub(4) {
            if frame_pair_at(&scan, off) {
                return Ok(Some(from + off as u64));
            }
        }
        Ok(None)
    }
}

/// Xing / Info 头（VBR 文件靠它拿精确的帧数与字节数）。
///
/// 两个标签名其实是同一套字段：VBR 编码器写 `Xing`、CBR 编码器写 `Info`。
/// 都按它读 —— CBR 下算出来的时长与"字节 ÷ 码率"一致，没有副作用。
#[derive(Debug, Default, Clone, Copy)]
struct XingHeader {
    frames: Option<u64>,
    bytes: Option<u64>,
}

fn parse_xing(b: &[u8]) -> XingHeader {
    if b.len() < 8 || (&b[0..4] != b"Xing" && &b[0..4] != b"Info") {
        return XingHeader::default();
    }
    let flags = u32::from_be_bytes([b[4], b[5], b[6], b[7]]);
    let mut pos = 8usize;
    let mut out = XingHeader::default();
    // 闭包只读 `b`、偏移靠参数传入，所以是 `Fn`：绑定不需要 `mut`（clippy 会指出这一点）
    let take_u32 = |pos: &mut usize| -> Option<u32> {
        let v = b.get(*pos..*pos + 4)?;
        *pos += 4;
        Some(u32::from_be_bytes([v[0], v[1], v[2], v[3]]))
    };
    if flags & 0x1 != 0 {
        out.frames = take_u32(&mut pos).map(u64::from);
    }
    if flags & 0x2 != 0 {
        out.bytes = take_u32(&mut pos).map(u64::from);
    }
    // 0x4 = TOC（100 字节）、0x8 = quality：本实现不用，但要知道它们存在 ——
    // 谁以后要加 LAME 的 delay/padding，得从这两个字段之后接着数
    out
}

// ---------------------------------------------------------------- Demuxer

impl crate::demux::Demuxer for MpegAudioDemuxer {
    fn info(&self) -> MediaInfo {
        MpegAudioDemuxer::info(self)
    }

    fn next_packet(&mut self) -> Result<Option<Packet>> {
        MpegAudioDemuxer::next_packet(self)
    }

    fn seek(&mut self, pts_us: i64) -> Result<i64> {
        MpegAudioDemuxer::seek(self, pts_us)
    }

    fn duration_us(&self) -> Option<i64> {
        self.duration_us
    }

    fn container(&self) -> &'static str {
        // 与 `ContainerFormat::MpegAudio.name()` 必须一致：宿主拿这个字符串做分发
        ContainerFormat::MpegAudio.name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demux::Demuxer;
    use me_core::Cursor;

    /// 造一个 MPEG-1 44.1k / 128 kbps / 立体声的帧头（帧长 417 字节）。
    fn v1_header() -> [u8; 4] {
        [0xFF, 0xFB, 0x90, 0x00]
    }

    /// 造 `n` 帧首尾相接的码流；`pad_every > 0` 时每隔这么多帧带一次 padding。
    fn stream(n: usize, pad_every: usize) -> Vec<u8> {
        let mut v = Vec::new();
        for i in 0..n {
            let mut h = v1_header();
            let padded = pad_every > 0 && i % pad_every == 0;
            if padded {
                h[2] |= 0b10; // padding 位
            }
            let frame_len = if padded { 418 } else { 417 };
            v.extend_from_slice(&h);
            v.extend(std::iter::repeat_n(0u8, frame_len - 4));
        }
        v
    }

    /// 造一个正文 `size` 字节的 ID3v2.3 标签头 + 占位正文。
    fn id3v2(size: usize) -> Vec<u8> {
        let mut v = vec![b'I', b'D', b'3', 0x03, 0x00, 0x00];
        // synchsafe：7 位一段
        v.push(((size >> 21) & 0x7F) as u8);
        v.push(((size >> 14) & 0x7F) as u8);
        v.push(((size >> 7) & 0x7F) as u8);
        v.push((size & 0x7F) as u8);
        v.extend(std::iter::repeat_n(0u8, size));
        v
    }

    #[test]
    fn parses_mpeg1_layer3_header() {
        let h = parse_frame_header(&v1_header()).unwrap();
        assert!(!h.lsf);
        assert_eq!(h.sample_rate, 44100);
        assert_eq!(h.bitrate_kbps, 128);
        assert_eq!(h.channels, 2);
        assert_eq!(h.samples, 1152);
        assert_eq!(h.side_info_len, 32);
        // 144 * 128000 / 44100 = 417.9 → 417
        assert_eq!(h.frame_len, 417);
    }

    #[test]
    fn parses_lsf_and_mono() {
        // MPEG-2（10）+ Layer III，24 kbps（bitrate_index = 3），22.05 kHz（index 0），单声道（11）
        let h = parse_frame_header(&[0xFF, 0xF3, 0x30, 0xC0]).unwrap();
        assert!(h.lsf);
        assert_eq!(h.sample_rate, 22050);
        assert_eq!(h.bitrate_kbps, 24);
        assert_eq!(h.channels, 1);
        assert_eq!(h.samples, 576);
        assert_eq!(h.side_info_len, 9);
        // 72 * 24000 / 22050 = 78.3 → 78
        assert_eq!(h.frame_len, 78);
    }

    #[test]
    fn rejects_layers_we_do_not_decode() {
        // Layer II / Layer I 与保留版本：与自研解码器口径一致，一律不认
        assert!(parse_frame_header(&[0xFF, 0xFD, 0x90, 0x00]).is_none(), "Layer II");
        assert!(parse_frame_header(&[0xFF, 0xFF, 0x90, 0x00]).is_none(), "Layer I");
        assert!(parse_frame_header(&[0xFF, 0xEB, 0x90, 0x00]).is_none(), "保留版本");
        // free format（bitrate_index = 0）与非法码率（15）
        assert!(parse_frame_header(&[0xFF, 0xFB, 0x00, 0x00]).is_none(), "free format");
        assert!(parse_frame_header(&[0xFF, 0xFB, 0xF0, 0x00]).is_none(), "bitrate 15");
        // 保留采样率（index 3）
        assert!(parse_frame_header(&[0xFF, 0xFB, 0x9C, 0x00]).is_none(), "采样率保留值");
        // 同步字不对 / 不足 4 字节
        assert!(parse_frame_header(&[0x47, 0xFB, 0x90, 0x00]).is_none());
        assert!(parse_frame_header(&[0xFF]).is_none());
    }

    #[test]
    fn frame_pair_is_stricter_than_a_single_header() {
        let data = stream(3, 0);
        assert!(frame_pair_at(&data, 0));
        // 单帧头成立、但帧长被改坏 → 第二帧接不上，不算一对
        let mut broken = data.clone();
        broken[2] = 0x70; // 改成 24 kbps，帧长变小
        assert!(!frame_pair_at(&broken, 0));
    }

    #[test]
    fn skips_id3v2_tag() {
        let tag = id3v2(1000);
        let tag_len = tag.len();
        let mut v = tag;
        v.extend_from_slice(&stream(3, 0));
        assert_eq!(id3v2_len(&v), Some(tag_len));
        assert_eq!(first_frame_offset(&v, true), Some(tag_len), "起点要从标签之后算");
        assert!(looks_like_mpeg_audio(&v));
    }

    #[test]
    fn very_long_id3v2_is_recognised_by_signature() {
        // 正文 1 MiB，远超探测窗口：只能靠"ID3 开头 + 版本合法"认
        let mut v = id3v2(1 << 20);
        v.truncate(SNIFF);
        assert_eq!(first_frame_offset(&v, true), None, "窗口里看不到第一帧");
        assert!(looks_like_mpeg_audio(&v), "标签签名足够强");
    }

    #[test]
    fn random_bytes_are_not_mpeg_audio() {
        assert!(!looks_like_mpeg_audio(&[0x11, 0x22, 0x33, 0x44]));
        assert!(!looks_like_mpeg_audio(&[]));
        assert!(!looks_like_mpeg_audio(b"ID3"), "连标签头都不全");
        // 只有单个同步字、后面接不上第二帧的，不算
        assert!(!looks_like_mpeg_audio(&[0xFF, 0xFB, 0x90, 0x00, 0, 0, 0, 0]));
    }

    #[test]
    fn demuxer_walks_every_frame_with_pts() {
        let data = stream(5, 0);
        let total = data.len();
        let mut d = MpegAudioDemuxer::new(Arc::new(Cursor::new(data)), "mem.mp3").unwrap();
        assert_eq!(d.container(), "mpeg-audio");
        let info = d.info();
        assert_eq!(info.tracks.len(), 1);
        assert_eq!(info.tracks[0].codec, CodecId::Mp3);
        assert_eq!(info.tracks[0].sample_rate, 44100);
        assert_eq!(info.tracks[0].channels, 2);
        assert_eq!(info.tracks[0].timebase.ticks_per_second, 44100);
        assert!(info.tracks[0].extradata.is_empty(), "MP3 不需要 extradata");

        let mut frames = 0i64;
        while let Some(p) = d.next_packet().unwrap() {
            assert_eq!(p.data.len(), 417, "载荷是完整的一帧（含帧头）");
            assert_eq!(p.track_id, 1);
            assert_eq!(p.pts, frames * 1152, "pts 按样本数走");
            assert_eq!(p.end(), (frames + 1) * 1152);
            assert!(p.keyframe);
            frames += 1;
        }
        assert_eq!(frames, 5);
        // 没有 Xing 时按"字节 ÷ 码率"估时长
        let expect = (total as i128 * 8 * 1_000_000 / 128_000) as i64;
        assert!((d.duration_us().unwrap() - expect).abs() <= 1000, "{:?}", d.duration_us());
        assert_eq!(d.seek(0).unwrap(), 0, "从头 seek 要落在第一帧");
        assert_eq!(d.next_packet().unwrap().unwrap().pts, 0);
    }

    #[test]
    fn padding_bit_is_counted_in_frame_length() {
        // 每隔一帧带 padding（帧长 418）：算错就从第二帧起错位并提前结束
        let mut d =
            MpegAudioDemuxer::new(Arc::new(Cursor::new(stream(6, 2))), "padded.mp3").unwrap();
        let mut n = 0;
        while d.next_packet().unwrap().is_some() {
            n += 1;
        }
        assert_eq!(n, 6);
    }

    #[test]
    fn garbage_between_frames_is_resynced() {
        let mut data = stream(2, 0);
        data.extend_from_slice(&[0u8; 64]); // 中间插一段垃圾
        data.extend_from_slice(&stream(2, 0));
        let mut d = MpegAudioDemuxer::new(Arc::new(Cursor::new(data)), "junk.mp3").unwrap();
        let mut n = 0;
        while let Some(p) = d.next_packet().unwrap() {
            assert_eq!(p.data.len(), 417);
            n += 1;
        }
        assert_eq!(n, 4, "垃圾段要能跳过去");
    }

    #[test]
    fn truncated_tail_frame_is_dropped() {
        let mut data = stream(2, 0);
        data.truncate(data.len() - 100); // 最后一帧只剩一半
        let mut d = MpegAudioDemuxer::new(Arc::new(Cursor::new(data)), "trunc.mp3").unwrap();
        let mut n = 0;
        while d.next_packet().unwrap().is_some() {
            n += 1;
        }
        assert_eq!(n, 1, "半截帧丢掉，不往上抛错误");
    }

    /// 只截到一帧的文件也必须能打开。
    ///
    /// 探测那一步要求"一对自洽的帧"，但**打开**时不能再卡这一条：截断的文件
    /// （下载到一半、或者录音被切断）就是只有一个帧头，卡了它就等于说"解不了"。
    #[test]
    fn single_frame_file_still_opens() {
        let data = stream(1, 0);
        assert!(
            !looks_like_mpeg_audio(&data),
            "前提：探测这一步看不见第二帧，认不出来（要靠上层已经确认过格式）"
        );
        let mut d = MpegAudioDemuxer::new(Arc::new(Cursor::new(data)), "one-frame.mp3").unwrap();
        assert_eq!(d.next_packet().unwrap().unwrap().pts, 0);
        assert!(d.next_packet().unwrap().is_none());
    }

    #[test]
    fn xing_header_gives_exact_duration() {
        let mut data = stream(1, 0);
        let off = 4 + 32; // 帧头 + 立体声侧信息
        data[off..off + 4].copy_from_slice(b"Xing");
        data[off + 4..off + 8].copy_from_slice(&0x3u32.to_be_bytes()); // frames + bytes
        data[off + 8..off + 12].copy_from_slice(&20u32.to_be_bytes());
        data[off + 12..off + 16].copy_from_slice(&(417u32 * 20).to_be_bytes());
        let d = MpegAudioDemuxer::new(Arc::new(Cursor::new(data)), "xing.mp3").unwrap();
        let expect = (20i128 * 1152 * 1_000_000 / 44100) as i64;
        assert_eq!(d.duration_us().unwrap(), expect);
    }

    #[test]
    fn first_frame_header_handles_single_frame_payloads() {
        // 一个包里有多帧：走"成对"判据
        assert_eq!(first_frame_header(&stream(3, 0)).unwrap().frame_len, 417);
        // 一个包里**只一帧**（引擎自己的 TS 封装就是这样）：成对判据永远不成立，
        // 得靠"帧长正好把剩下的字节用完"这条兜住
        assert_eq!(first_frame_header(&stream(1, 0)).unwrap().frame_len, 417);
        // 前面带半帧残渣（真实 PES 很常见）：要能跳过残渣找到真帧头
        let mut partial = vec![0u8; 100];
        partial.extend_from_slice(&stream(1, 0));
        assert_eq!(first_frame_header(&partial).unwrap().sample_rate, 44100);
        // 纯垃圾：一个都不能认
        assert!(first_frame_header(&[0u8; 64]).is_none());
        assert!(first_frame_header(&[0xFF, 0xFB, 0x90, 0x00, 0, 0]).is_none(), "帧长对不上");
    }

    #[test]
    fn long_id3v2_tag_is_jumped_over_when_opening() {
        // 标签 200 KiB：`locate_first_frame` 必须跳到标签末尾，而不是把标签读进来
        let tag = id3v2(200 * 1024);
        let tag_len = tag.len();
        let mut v = tag;
        v.extend_from_slice(&stream(4, 0));
        let mut d = MpegAudioDemuxer::new(Arc::new(Cursor::new(v)), "big-tag.mp3").unwrap();
        assert_eq!(d.audio_start, tag_len as u64);
        let p = d.next_packet().unwrap().unwrap();
        assert_eq!(p.pts, 0, "跳过标签后第一帧的 pts 从 0 开始");
        assert_eq!(d.info().tracks[0].sample_rate, 44100);
    }
}
