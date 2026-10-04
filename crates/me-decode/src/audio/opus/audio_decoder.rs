//! Opus 的 [`AudioDecoder`] 适配层：把 [`OpusDecoder`]（一包一段 PCM）接到
//! 引擎统一的"喂包 → 收 [`PcmChunk`]"接口上。
//!
//! 这一层只做三件事，都跟"解码"本身无关：
//!
//! 1. **读 `OpusHead`**：声道数与 `pre_skip`（有 19 字节的头就够，不为此引入对容器层的依赖）；
//! 2. **丢掉前导样点（pre-skip）**：Opus 规范要求解码器丢掉开头的 `pre_skip` 个样点
//!    （那是编码器/重采样链的固有延迟，典型 312 个 = 6.5 ms）。丢了它，WAV 的时间轴
//!    才与"容器报的时间戳"对齐 —— 容器侧的时间戳本身已经按 pre-skip 修正过了
//!    （见 `me-container/src/ogg.rs` 的时长计算）。
//! 3. **不做增益**：`OpusHead` 里的 `output_gain` 是"输出侧"的事（就像位深缩放），
//!    参考解码器 `opus_decode` 默认也不应用它；这里保持一致，数值上可与参考逐样本对拍。
//!
//! ## 为什么输出固定 48 kHz / 16 bit
//!
//! Opus 的**解码采样率按定义就是 48 kHz**（`OpusHead` 里那个"输入采样率"只是编码前的
//! 元数据）。参考解码器也一样：要别的采样率得在**输出侧**再重采样。位深是 16：
//! Opus 的有损输出就是 16 位精度，多给的位只会是零。

use me_core::{MediaError, Result};

use super::opus_decoder::OpusDecoder;
use crate::audio::{AudioDecoder, PcmChunk};

/// `OpusHead` 里我们需要的两个字段（其余字段本层用不到）。
#[derive(Debug)]
struct HeadInfo {
    channels: u8,
    pre_skip: usize,
}

/// 解析 `OpusHead`（与 `me-container::ogg::parse_opus_head` 同一套字段，
/// 但那一边在容器层，本层不依赖它 —— 头只有 19 字节）。
fn parse_head(p: &[u8]) -> Result<HeadInfo> {
    if p.len() < 19 || !p.starts_with(b"OpusHead") {
        return Err(MediaError::Unsupported(
            "Opus: extradata 不是 OpusHead（长度不足或魔数不对）".into(),
        ));
    }
    let version = p[8];
    if version >> 4 >= 1 {
        return Err(MediaError::Unsupported(format!(
            "Opus: 封装版本 {version} 不认识（本引擎认 1）"
        )));
    }
    let channels = p[9];
    if channels == 0 {
        return Err(MediaError::Bitstream("Opus: OpusHead 的声道数是 0".into()));
    }
    let mapping_family = p[18];
    if mapping_family != 0 {
        return Err(MediaError::Unsupported(format!(
            "Opus: 声道映射族 {mapping_family}（环绕）还没实现"
        )));
    }
    if channels > 2 {
        return Err(MediaError::Unsupported(format!(
            "Opus: {channels} 声道还没实现（本引擎只做 1/2 声道）"
        )));
    }
    Ok(HeadInfo { channels, pre_skip: u16::from_le_bytes([p[10], p[11]]) as usize })
}

/// Opus 音频解码器（引擎侧）。
pub struct OpusAudioDecoder {
    dec: OpusDecoder,
    channels: u8,
    /// 还有多少个前导样点要丢（丢到 0 为止）。
    pre_skip_left: usize,
    /// 复用缓冲：`decode_packet` 的输出。
    raw: Vec<i16>,
}

impl OpusAudioDecoder {
    /// 用 `OpusHead` 开一个解码器。
    pub fn open(extradata: &[u8]) -> Result<Self> {
        let head = parse_head(extradata)?;
        Ok(Self {
            dec: OpusDecoder::new(head.channels as usize)?,
            channels: head.channels,
            pre_skip_left: head.pre_skip,
            raw: Vec::new(),
        })
    }
}

impl AudioDecoder for OpusAudioDecoder {
    fn decode(&mut self, packet: &[u8], pts_us: i64) -> Result<Vec<PcmChunk>> {
        let n = self.dec.decode_packet(packet, &mut self.raw)?; // 每声道样点数
        let ch = self.channels as usize;
        // 丢掉前导样点（只可能发生在开头几个包里）
        let mut start = 0usize;
        if self.pre_skip_left > 0 {
            let drop = self.pre_skip_left.min(n);
            self.pre_skip_left -= drop;
            start = drop * ch;
        }
        if start >= self.raw.len() {
            return Ok(Vec::new());
        }
        Ok(vec![PcmChunk {
            sample_rate: 48000,
            channels: self.channels,
            bits_per_sample: 16,
            pts_us,
            samples: self.raw[start..].iter().map(|&v| v as i32).collect(),
        }])
    }

    fn flush(&mut self) -> Result<Vec<PcmChunk>> {
        Ok(Vec::new())
    }

    fn stream_info(&self) -> (u32, u8, u8) {
        (48000, self.channels, 16)
    }

    fn backend(&self) -> &'static str {
        "self-opus"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn head(channels: u8, pre_skip: u16, family: u8) -> Vec<u8> {
        let mut v = b"OpusHead".to_vec();
        v.push(1);
        v.push(channels);
        v.extend_from_slice(&pre_skip.to_le_bytes());
        v.extend_from_slice(&48000u32.to_le_bytes());
        v.extend_from_slice(&0i16.to_le_bytes());
        v.push(family);
        v
    }

    #[test]
    fn head_gives_channels_and_pre_skip() {
        let h = parse_head(&head(2, 356, 0)).unwrap();
        assert_eq!(h.channels, 2);
        assert_eq!(h.pre_skip, 356);
        assert_eq!(OpusAudioDecoder::open(&head(1, 312, 0)).unwrap().stream_info(), (48000, 1, 16));
    }

    #[test]
    fn unsupported_heads_say_what_is_missing() {
        // 环绕映射族
        let e = parse_head(&head(6, 0, 1)).unwrap_err();
        assert!(format!("{e}").contains("映射族"), "{e}");
        // 多于 2 声道
        assert!(parse_head(&head(3, 0, 0)).is_err());
        // 不是 OpusHead
        assert!(parse_head(b"OggS").is_err());
        // 版本不兼容（高 4 位 >= 1）
        let mut v = head(2, 0, 0);
        v[8] = 0x10;
        assert!(parse_head(&v).is_err());
    }

    #[test]
    fn pre_skip_samples_are_dropped_once() {
        // 用一个"能正常解出 960 样点/声道"的 CELT-only 包，把 pre_skip 设成 100
        let mut d = OpusAudioDecoder::open(&head(2, 100, 0)).unwrap();
        let pkt = [0xFCu8, 0x00, 0x00];
        let chunks = d.decode(&pkt, 0).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].samples.len(), (960 - 100) * 2, "前导样点应当只丢一次");
        assert_eq!(chunks[0].sample_rate, 48000);
        assert_eq!(chunks[0].bits_per_sample, 16);
        let chunks2 = d.decode(&pkt, 20_000).unwrap();
        assert_eq!(chunks2[0].samples.len(), 960 * 2, "后面的包不该再丢");
    }
}
