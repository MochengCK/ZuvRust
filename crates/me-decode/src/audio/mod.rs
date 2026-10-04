//! 音频解码层：把压缩音频解成 PCM。
//!
//! 与 [`crate::VideoDecoder`] 同一套思路（引擎持有解码器，前端只收结果），
//! 但音频这一侧**后端只有自研软解**：音频解码的计算量小（相对视频），没有
//! 硬解的必要，而"完全自研"这条底线在音频上最该先立住。
//!
//! 已接：**FLAC**（无损，可以逐样本 bit-exact 校验 —— 拿系统解码器当参考，
//! 一个 bit 都不差才算对）、**AAC-LC**、**MP3**（Layer III）。接下来按同一套接口补 Opus。
//!
//! ## 输出形态
//!
//! [`PcmChunk`]：交织的 i32 采样（**按原始位深**，不做缩放）+ 采样率/声道数 +
//! 时间戳。为什么用 i32 而不是 i16：24-bit / 32-bit 的 FLAC 装不进 i16，缩放会
//! 丢掉"无损"这个前提；要不要降位是**播出去那一步**的事（由输出侧决定）。

pub mod aac;
mod bitreader;
pub mod flac;
pub mod mp3;
pub mod opus;

use me_core::{CodecId, MediaError, Result};

/// 一段解好的 PCM。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcmChunk {
    pub sample_rate: u32,
    pub channels: u8,
    /// 原始位深（8/16/20/24/32）
    pub bits_per_sample: u8,
    /// 这段第一个采样对应的媒体时间（微秒）
    pub pts_us: i64,
    /// 交织采样（长度 = 帧样本数 × 声道数），数值按 `bits_per_sample` 位深
    pub samples: Vec<i32>,
}

/// 音频解码器：喂一个压缩包，收 PCM。
pub trait AudioDecoder: Send {
    /// 解一个压缩包（FLAC 里一个包 = 一帧）。
    fn decode(&mut self, packet: &[u8], pts_us: i64) -> Result<Vec<PcmChunk>>;

    /// 收尾（把内部残留的帧吐出来）。
    fn flush(&mut self) -> Result<Vec<PcmChunk>>;

    /// 声道数 / 采样率 / 位深（来自流信息，解第一帧之前就可用）。
    fn stream_info(&self) -> (u32, u8, u8);

    /// 后端名字（日志用）。
    fn backend(&self) -> &'static str {
        "self"
    }
}

/// 打开一个音频解码器（目前有自研 FLAC / AAC-LC / MP3）。
///
/// `extradata` 是该编码的解码器初始化数据：FLAC 是 `dfLa` 载荷（MP4）或
/// STREAMINFO（裸流）；AAC 是 AudioSpecificConfig；**MP3 没有这种东西**
/// （采样率/声道/比特率都在每帧的帧头里），传什么都行。
pub fn open_audio_decoder(codec: CodecId, extradata: &[u8]) -> Result<Box<dyn AudioDecoder>> {
    match codec {
        CodecId::Flac => Ok(Box::new(flac::FlacDecoder::open(extradata)?)),
        CodecId::Aac => Ok(Box::new(aac::AacDecoder::open(extradata)?)),
        CodecId::Mp3 => Ok(Box::new(mp3::Mp3Decoder::open(extradata)?)),
        CodecId::Opus => Ok(Box::new(opus::audio_decoder::OpusAudioDecoder::open(extradata)?)),
        other => Err(MediaError::Unsupported(format!(
            "音频解码还没接这个编码：{}（已接：FLAC、AAC-LC、MP3、Opus）",
            other.name()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_codec_says_what_is_missing() {
        // AC-3 本引擎没有解码器：错误里必须点名
        let e = match open_audio_decoder(CodecId::Ac3, &[]) {
            Ok(_) => panic!("AC-3 不该成功"),
            Err(e) => e,
        };
        let text = format!("{e}");
        assert!(text.contains("ac3"), "错误里要有编码名：{text}");
    }

    #[test]
    fn opus_opens_with_a_minimal_head() {
        let mut head = b"OpusHead".to_vec();
        head.push(1); // 版本
        head.push(2); // 2 声道
        head.extend_from_slice(&312u16.to_le_bytes()); // pre-skip
        head.extend_from_slice(&48000u32.to_le_bytes());
        head.extend_from_slice(&0i16.to_le_bytes());
        head.push(0); // 映射族 0
        let d = open_audio_decoder(CodecId::Opus, &head).unwrap();
        assert_eq!(d.stream_info(), (48000, 2, 16));
        assert_eq!(d.backend(), "self-opus");
    }
}
