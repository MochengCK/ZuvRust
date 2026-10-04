//! 容器探测：只看头部若干字节判断"这是什么容器"，再交给对应解析器。
//!
//! 探测与解析**分开**是刻意的：探测必须极快且**绝不能读太多**——
//! 边下边播时，磁力还没拿到元信息、BT 第一个片还没下完，
//! 唯一能确定的就是"开头这几个字节长什么样"。读多了就会卡在
//! "等远端把整段送过来"，播放起不来。
//!
//! 引擎现在认六种输入形态（按判定顺序，**强特征在前、弱特征兜底**）：
//!
//! | 输入形态 | 同步特征 | 来源 |
//! |---|---|---|
//! | MPEG-TS | 每 188 字节一个 `0x47` | HLS 的 `.ts` 分片、DVB 录制 |
//! | ISO BMFF | 第 4..8 字节是 `ftyp`/`styp`/`moov`/`moof` | MP4、fMP4（HLS 的 `.m4s`、DASH） |
//! | Ogg | 开头是 `OggS` 魔数 + 版本 0（页 CRC 也验） | `.opus` / `.ogg`（Opus） |
//! | 裸 FLAC | 开头（ID3v2 之后）是 `fLaC` 魔数 | `.flac` 无损音频 |
//! | 裸 ADTS AAC | 开头（ID3v2 之后）是**一对自洽**的 ADTS 帧头 | `.aac`（含 iTunes 导出的） |
//! | 裸 MPEG 音频 | 开头（ID3v2 之后）是一对自洽的 Layer III 帧头 | `.mp3` 文件、音频 CD 抓轨 |
//!
//! 判定顺序是有讲究的：**裸 MPEG 音频的信号最弱**（只有 11 位同步字加几个字段，
//! 所以还要拿"第二帧正好落在第一帧帧长算出的位置"来加固，见
//! [`mpeg_audio::frame_pair_at`]），所以放最后兜底；前面几个都有强特征，
//! 先摘掉再谈 MP3。ADTS 排在 MP3 前面是因为它的证据更强（12 位同步字 +
//! `layer == 00` + 帧长字段自洽 + 两帧参数一致），而且它自己**不做**"ID3v2 长过
//! 窗口就当它是 ADTS"的兜底（那种文件 `.aac` 与 `.mp3` 长得一样，谁先判都会抢，
//! 见 [`adts::looks_like_adts`]）。Ogg 排得靠前是因为它有 4 字节魔数 + 版本 + 页 CRC
//! 三重证据 —— 只是它装了 Vorbis 时本引擎还没有解码器，会在打开时明确报 `Unsupported`。
//!
//! ## 为什么给 TS 留了"包长不是 188"的余地
//!
//! 真实世界里 TS 有三种排布：裸的 188 字节包、加了 4 字节时码的 192 字节包
//! （M2TS，蓝光与部分广播流）、加了 16 字节 RS 校验的 204 字节包（DVB）。
//! 只认 188 会在遇到这两种时"整条流全是垃圾" —— 同步字永远对不上。

use std::sync::Arc;

use me_core::{MediaError, MediaInfo, ReadAt, Result};

/// 引擎认得的输入形态（对外叫"容器"，其中"裸流"三种其实没有容器壳）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerFormat {
    /// MPEG-TS（188 / 192 / 204 字节三种包长）。
    MpegTs,
    /// ISO BMFF（普通 MP4 与分段 fMP4）。
    IsoBmff,
    /// 裸 FLAC 流（`fLaC` + 元数据块 + 音频帧）。
    Flac,
    /// Ogg 容器（页序列；本引擎目前只认里面装 Opus 的逻辑流）。
    Ogg,
    /// 裸 ADTS AAC 流（`.aac`）：一串首尾相接的 ADTS 帧，帧长写在每帧头里。
    Adts,
    /// 裸 MPEG 音频流（Layer III 的 `.mp3`）：其实没有容器，只有一串首尾相接的帧。
    MpegAudio,
    /// 认不出来。
    Unknown,
}

impl ContainerFormat {
    /// 对外名字（CLI / C API 会把它写进 JSON，改名等于破坏契约）。
    pub fn name(self) -> &'static str {
        match self {
            ContainerFormat::MpegTs => "mpeg-ts",
            ContainerFormat::IsoBmff => "iso-bmff",
            ContainerFormat::Flac => "flac",
            ContainerFormat::Ogg => "ogg",
            ContainerFormat::Adts => "adts",
            ContainerFormat::MpegAudio => "mpeg-audio",
            ContainerFormat::Unknown => "unknown",
        }
    }
}

/// 探测时最多看多少字节。
///
/// 4 KiB 足够覆盖：MP4 的 `ftyp` 在头 8 字节，TS 要确认包长得看 4 个同步点
/// （204 包长时 3×204+1 = 613），MP3 的第一帧要么在偏移 0、要么贴在 ID3v2 标签后面。
/// 再往上加除了拖慢探测没有任何收益。
const SNIFF_BYTES: usize = 4096;

/// ISO BMFF 里可以作为顶层第一个 box 的类型。
///
/// 不把这些列全就会漏：录屏软件产出的 MP4 第一个 box 往往是 `free`/`skip`，
/// 而 fMP4 分片（`styp` 开头）根本没有 `ftyp`。
/// 类型写成 `&[u8]` 而不是 `&[u8; 4]`：手头拿到的 `kind` 是切片（`&head[4..8]`），
/// 用定长数组做元素就得每处都 `try_into()` 一次，纯属给自己找麻烦。
const BMFF_TOP_LEVEL: [&[u8]; 9] = [
    b"ftyp", b"styp", b"moov", b"moof", b"mdat", b"free", b"skip", b"wide", b"sidx",
];

/// 只看头部，判断容器类型。
pub fn sniff(src: &dyn ReadAt) -> Result<ContainerFormat> {
    let head = src.read_range(0, SNIFF_BYTES)?;
    Ok(sniff_bytes(&head))
}

/// 纯函数版的探测（测试与"数据已在手"的场景用它）。
pub fn sniff_bytes(head: &[u8]) -> ContainerFormat {
    if looks_like_bmff(head) {
        return ContainerFormat::IsoBmff;
    }
    if ts::packet_size_of(head).is_some() {
        return ContainerFormat::MpegTs;
    }
    // Ogg：4 字节魔数 + 版本 0（能读满一页时再验 CRC），排在最前几个强特征里
    if ogg::looks_like_ogg(head) {
        return ContainerFormat::Ogg;
    }
    // 裸 FLAC：`fLaC` 是 4 字节魔数，比 MP3 那 11 位同步码强，排在 MP3 前面
    if flac_stream::looks_like_flac(head) {
        return ContainerFormat::Flac;
    }
    // 裸 ADTS AAC：12 位同步字 + `layer == 00` + 帧长自洽，还要求"成对"（证据比 MP3 强）
    if adts::looks_like_adts(head) {
        return ContainerFormat::Adts;
    }
    // 最后才试裸 MPEG 音频：它的特征最弱，前面几个用强特征先摘掉
    if mpeg_audio::looks_like_mpeg_audio(head) {
        return ContainerFormat::MpegAudio;
    }
    ContainerFormat::Unknown
}

fn looks_like_bmff(head: &[u8]) -> bool {
    if head.len() < 8 {
        return false;
    }
    // box 头：4 字节长度 + 4 字节类型。长度 1 表示后面还有 8 字节 largesize，
    // 0 表示"一直到文件末尾"，这两种都不是畸形数据，都要认。
    let size = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
    let kind = &head[4..8];
    if size == 0 || size == 1 {
        return BMFF_TOP_LEVEL.contains(&kind);
    }
    if size < 8 {
        return false;
    }
    BMFF_TOP_LEVEL.contains(&kind)
}

/// 完整探测：识别容器并解析出轨道信息。
///
/// **不解析码流本身**（不解码、不读 `mdat`），所以对一个大文件也是毫秒级。
pub fn probe(src: Arc<dyn ReadAt>, source: &str) -> Result<MediaInfo> {
    match sniff(&*src)? {
        ContainerFormat::MpegTs => Ok(ts::TsDemuxer::probe_reader(src, source)?.1),
        ContainerFormat::IsoBmff => Ok(mp4::Mp4Demuxer::probe_reader(src, source)?.1),
        ContainerFormat::Flac => Ok(flac_stream::FlacStreamDemuxer::probe_reader(src, source)?.1),
        ContainerFormat::Ogg => Ok(ogg::OggOpusDemuxer::probe_reader(src, source)?.1),
        ContainerFormat::Adts => Ok(adts::AdtsDemuxer::probe_reader(src, source)?.1),
        ContainerFormat::MpegAudio => Ok(mpeg_audio::MpegAudioDemuxer::probe_reader(src, source)?.1),
        ContainerFormat::Unknown => Err(MediaError::container(format!(
            "无法识别的容器（看了 {} 字节，既不是 MPEG-TS、ISO BMFF、Ogg，也不是裸 FLAC / ADTS AAC / MPEG 音频）",
            src.read_range(0, SNIFF_BYTES)?.len()
        ))),
    }
}

// 让 `sniff_bytes` 与 `probe` 能复用子模块里的实现，又不把内部细节暴露到 crate 外。
use crate::{adts, flac_stream, mp4, mpeg_audio, ogg, ts};

#[cfg(test)]
mod tests {
    use super::*;
    use me_core::Cursor;

    fn ts_head(packet_size: usize) -> Vec<u8> {
        let mut v = vec![0u8; SNIFF_BYTES];
        // 每个包的首字节是同步字，其余填成非同步字（0x00），
        // 免得"碰巧连续对上"把探测算法测宽松了
        for i in (0..SNIFF_BYTES).step_by(packet_size) {
            v[i] = 0x47;
        }
        v
    }

    /// 裸 MP3 的头部：`n` 帧 MPEG-1 44.1k / 128 kbps / 立体声（帧长 417）。
    fn mp3_head(n: usize, pad_every: usize) -> Vec<u8> {
        let mut v = Vec::new();
        for i in 0..n {
            let mut h = [0xFFu8, 0xFB, 0x90, 0x00];
            let padded = pad_every > 0 && i % pad_every == 0;
            if padded {
                h[2] |= 0b10;
            }
            let frame_len = if padded { 418 } else { 417 };
            v.extend_from_slice(&h);
            v.extend(std::iter::repeat_n(0u8, frame_len - 4));
        }
        v.resize(SNIFF_BYTES, 0);
        v
    }

    #[test]
    fn detects_plain_mpeg_ts() {
        assert_eq!(sniff_bytes(&ts_head(188)), ContainerFormat::MpegTs);
    }

    #[test]
    fn detects_192_byte_ts_and_204_byte_ts() {
        // M2TS：4 字节时码前缀 + 188 字节包
        let mut m2ts = ts_head(192);
        for i in (0..SNIFF_BYTES).step_by(192) {
            m2ts[i + 4] = 0x47;
            m2ts[1] = 0x00; // 时码前缀里不该有同步字
        }
        // 上面那段循环把 i 位置的 0x47 也留下了，去掉它（真实 M2TS 没有）
        for i in (0..SNIFF_BYTES).step_by(192) {
            m2ts[i] = 0x00;
        }
        assert_eq!(sniff_bytes(&m2ts), ContainerFormat::MpegTs);

        // DVB：188 字节包 + 16 字节 RS 校验
        assert_eq!(sniff_bytes(&ts_head(204)), ContainerFormat::MpegTs);
    }

    #[test]
    fn detects_mp4_variants() {
        let mut ftyp = Vec::new();
        ftyp.extend_from_slice(&32u32.to_be_bytes());
        ftyp.extend_from_slice(b"ftyp");
        ftyp.extend_from_slice(b"isom");
        ftyp.extend_from_slice(&[0u8; 20]);
        assert_eq!(sniff_bytes(&ftyp), ContainerFormat::IsoBmff);

        // fMP4 分片：styp 开头，没有 ftyp
        let mut styp = Vec::new();
        styp.extend_from_slice(&16u32.to_be_bytes());
        styp.extend_from_slice(b"styp");
        styp.extend_from_slice(&[0u8; 8]);
        assert_eq!(sniff_bytes(&styp), ContainerFormat::IsoBmff);

        // 录屏软件常见的 free 开头
        let mut free = Vec::new();
        free.extend_from_slice(&24u32.to_be_bytes());
        free.extend_from_slice(b"free");
        free.extend_from_slice(&[0u8; 16]);
        assert_eq!(sniff_bytes(&free), ContainerFormat::IsoBmff);
    }

    #[test]
    fn detects_bare_mpeg_audio() {
        assert_eq!(sniff_bytes(&mp3_head(2, 0)), ContainerFormat::MpegAudio);
        // ID3v2 标签在前（正文 1000 字节）
        let mut tagged = vec![b'I', b'D', b'3', 0x03, 0x00, 0x00];
        tagged.extend_from_slice(&[0, 0, (1000 >> 7) as u8, (1000 & 0x7F) as u8]);
        tagged.extend(std::iter::repeat_n(0u8, 1000));
        tagged.extend_from_slice(&mp3_head(2, 0));
        assert_eq!(sniff_bytes(&tagged), ContainerFormat::MpegAudio);
    }

    /// 只有一帧、后面接不上第二帧：**不算** MPEG 音频。
    ///
    /// 这一条是"敢把裸 MP3 放进探测"的前提：`0xFF Ex` 这种字节对在二进制里太常见，
    /// 放松到单个帧头就会把各种文件错认成 mp3（症状是 `probe` 报一条假的音轨）。
    #[test]
    fn a_lone_sync_word_is_not_mpeg_audio() {
        let mut v = vec![0xFFu8, 0xFB, 0x90, 0x00];
        v.resize(SNIFF_BYTES, 0);
        assert_eq!(sniff_bytes(&v), ContainerFormat::Unknown);
        // 随机数据同理
        assert_eq!(sniff_bytes(&[0x11, 0x22, 0x33, 0x44]), ContainerFormat::Unknown);
    }

    /// 造一条 ADTS AAC：`n` 帧、44.1k 立体声、每帧 `body + 7` 字节。
    fn adts_head(n: usize, body: usize) -> Vec<u8> {
        let frame_len = 7 + body;
        let mut v = Vec::new();
        for _ in 0..n {
            let mut h = [0u8; 7];
            let channels: u8 = 2; // 立体声
            h[0] = 0xFF;
            h[1] = 0xF1; // 12 位同步字 + layer=00 + 无 CRC
            h[2] = (1 << 6) | (4 << 2) | ((channels >> 2) & 0x01); // AAC-LC / 44.1k / 立体声
            h[3] = (channels & 0x03) << 6 | ((frame_len >> 11) & 0x03) as u8;
            h[4] = ((frame_len >> 3) & 0xFF) as u8;
            h[5] = ((frame_len & 0x07) << 5) as u8 | 0x1F;
            h[6] = 0xFC;
            v.extend_from_slice(&h);
            v.extend(std::iter::repeat_n(0x00u8, body));
        }
        v.resize(SNIFF_BYTES, 0);
        v
    }

    #[test]
    fn detects_bare_adts_aac() {
        assert_eq!(sniff_bytes(&adts_head(2, 100)), ContainerFormat::Adts);
        // ID3v2 标签在前（正文 1000 字节）
        let mut tagged = vec![b'I', b'D', b'3', 0x04, 0x00, 0x00];
        tagged.extend_from_slice(&[0, 0, (1000 >> 7) as u8, (1000 & 0x7F) as u8]);
        tagged.extend(std::iter::repeat_n(0u8, 1000));
        tagged.extend_from_slice(&adts_head(3, 80));
        assert_eq!(sniff_bytes(&tagged), ContainerFormat::Adts);
    }

    /// 单个 ADTS 帧头（后面接不上第二帧）**不算** ADTS：与裸 MP3 同一条纪律。
    ///
    /// 这一条比 MP3 那侧更要紧：ADTS 与 MP3 都在"ID3v2 之后"，一旦放松成"单个头就算"，
    /// 两种格式就会互相错认（谁先判都抢），`probe` 会报出假的格式名。
    #[test]
    fn a_lone_adts_header_is_not_adts() {
        let mut v = adts_head(1, 100);
        v.resize(SNIFF_BYTES, 0x00); // 抹掉后面那些帧（只留第一帧）
        let mut only = adts_head(1, 100);
        only.truncate(107);
        only.resize(SNIFF_BYTES, 0x00);
        assert_eq!(sniff_bytes(&only), ContainerFormat::Unknown);
        // 反过来：真的 mp3 不会被判成 adts（同步字/layer 不同）
        assert_eq!(sniff_bytes(&mp3_head(2, 0)), ContainerFormat::MpegAudio);
    }

    #[test]
    fn random_bytes_are_unknown() {
        assert_eq!(sniff_bytes(&[0x11, 0x22, 0x33, 0x44]), ContainerFormat::Unknown);
        assert_eq!(sniff_bytes(b"not a media file at all"), ContainerFormat::Unknown);
        assert_eq!(sniff_bytes(&[]), ContainerFormat::Unknown);
    }

    #[test]
    fn short_buffer_does_not_panic() {
        // 只有 2 字节：两种探测都读不到特征，必须是 Unknown 而不是 panic
        assert_eq!(sniff_bytes(&[0x47, 0x47]), ContainerFormat::Unknown);
    }

    #[test]
    fn sniff_uses_source_range() {
        let c = Cursor::new(ts_head(188));
        assert_eq!(sniff(&c).unwrap(), ContainerFormat::MpegTs);
    }

    #[test]
    fn format_names_are_the_external_contract() {
        assert_eq!(ContainerFormat::MpegTs.name(), "mpeg-ts");
        assert_eq!(ContainerFormat::IsoBmff.name(), "iso-bmff");
        assert_eq!(ContainerFormat::Flac.name(), "flac");
        assert_eq!(ContainerFormat::Adts.name(), "adts");
        assert_eq!(ContainerFormat::MpegAudio.name(), "mpeg-audio");
        assert_eq!(ContainerFormat::Unknown.name(), "unknown");
    }
}