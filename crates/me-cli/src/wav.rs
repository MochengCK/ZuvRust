//! 最小 WAV 写出（PCM）。
//!
//! 为什么自己写：这是**校验解码器**用的（引擎解出的 PCM 存成 WAV，就能和系统
//! 解码器的产物逐样本比对）。不为了写 WAV 引一个库 —— 头就 44 字节。

use std::io::Write;

use me_core::{MediaError, Result};

/// 把交织 PCM 写成 WAV（`bits` 支持 16/24/32）。
///
/// - `samples` 的数值按 `bits` 位深（与 [me_decode::audio::PcmChunk] 的约定一致）；
/// - 24 位按 3 字节小端打包（`S24_3LE`），16 位按 `S16_LE`，32 位按 `S32_LE`。
pub fn write_wav(
    path: &str,
    sample_rate: u32,
    channels: u8,
    bits: u8,
    samples: &[i32],
) -> Result<()> {
    if !matches!(bits, 16 | 24 | 32) {
        return Err(MediaError::Unsupported(format!(
            "写 WAV 只支持 16/24/32 位（这条流是 {bits} 位）"
        )));
    }
    if sample_rate == 0 || channels == 0 {
        return Err(MediaError::Io("采样率/声道数不合法，写不了 WAV".into()));
    }
    let bytes_per_sample = (bits / 8) as usize;
    let data_len = samples.len() * bytes_per_sample;
    let mut f = std::fs::File::create(path)?;
    let byte_rate = sample_rate * channels as u32 * bytes_per_sample as u32;
    let block_align = channels as u16 * bytes_per_sample as u16;
    let mut head = Vec::with_capacity(44);
    head.extend_from_slice(b"RIFF");
    head.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    head.extend_from_slice(b"WAVE");
    head.extend_from_slice(b"fmt ");
    head.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk 长度
    head.extend_from_slice(&1u16.to_le_bytes()); // 1 = PCM
    head.extend_from_slice(&(channels as u16).to_le_bytes());
    head.extend_from_slice(&sample_rate.to_le_bytes());
    head.extend_from_slice(&byte_rate.to_le_bytes());
    head.extend_from_slice(&block_align.to_le_bytes());
    head.extend_from_slice(&(bits as u16).to_le_bytes());
    head.extend_from_slice(b"data");
    head.extend_from_slice(&(data_len as u32).to_le_bytes());
    f.write_all(&head)?;

    // 逐样本打包（24 位要自己拆字节）
    let mut buf = Vec::with_capacity(data_len.min(1 << 20));
    for s in samples {
        match bits {
            16 => buf.extend_from_slice(&(*s as i16).to_le_bytes()),
            24 => buf.extend_from_slice(&s.to_le_bytes()[..3]),
            _ => buf.extend_from_slice(&s.to_le_bytes()),
        }
        if buf.len() >= 1 << 20 {
            f.write_all(&buf)?;
            buf.clear();
        }
    }
    f.write_all(&buf)?;
    f.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_a_valid_header_and_payload() {
        let dir = std::env::temp_dir().join(format!("me-wav-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.wav");
        let p = path.to_str().unwrap();
        write_wav(p, 44100, 2, 16, &[1, -1, 32767, -32768]).unwrap();
        let got = std::fs::read(&path).unwrap();
        assert_eq!(&got[0..4], b"RIFF");
        assert_eq!(&got[8..12], b"WAVE");
        assert_eq!(&got[36..40], b"data");
        assert_eq!(
            u32::from_le_bytes([got[4], got[5], got[6], got[7]]) as usize,
            got.len() - 8
        );
        assert_eq!(u32::from_le_bytes([got[40], got[41], got[42], got[43]]), 8);
        // 采样值按小端原样落盘（bit-exact 校验就靠这个）
        assert_eq!(i16::from_le_bytes([got[44], got[45]]), 1);
        assert_eq!(i16::from_le_bytes([got[46], got[47]]), -1);
        assert_eq!(i16::from_le_bytes([got[48], got[49]]), 32767);
        assert_eq!(i16::from_le_bytes([got[50], got[51]]), -32768);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_unsupported_depths() {
        assert!(write_wav("/tmp/x.wav", 44100, 2, 12, &[]).is_err());
        assert!(write_wav("/tmp/x.wav", 0, 2, 16, &[]).is_err());
    }
}
