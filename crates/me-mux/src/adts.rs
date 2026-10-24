//! 裸 ADTS（`.aac`）封装：给每条 AAC 裸帧补一个 7 字节 ADTS 头直写。
//!
//! ADTS 是**音频基本流的帧包装**（不是容器）：每帧自带同步字与长度，所以
//! 播放器/解码器可以从中任意位置接入。`.aac` 文件就是"一帧接一帧"。
//!
//! 与 [`crate::ts`] 的关系：MPEG-TS 里的 AAC 也要套 ADTS 头，两处用的是**同一份**
//! 头生成函数（[`adts_header`]）—— 差别只在 TS 还要再包一层 PES/TS 包。
//!
//! 只接受**一条 AAC 音频轨**：裸流没有轨的概念。

use std::io::Write;

use me_core::{CodecId, MediaError, MediaKind, Packet, Result, Track};

use crate::sink::Sink;

/// 这段数据是不是已经是 ADTS 帧（带 12 位同步字）。
pub(crate) fn is_adts(data: &[u8]) -> bool {
    data.len() >= 2 && data[0] == 0xFF && (data[1] & 0xF0) == 0xF0
}

/// 由 `AudioSpecificConfig` 合成 ADTS 头（7 字节，无 CRC）。
///
/// 返回 `None` 表示 ASC 不合法（或者这一帧大到 13 位长度字段装不下）——
/// 调用方要据此**报错**，绝不能写一个头字段错误的帧出去（那会让解码器
/// 从这一帧开始整段失步）。
pub(crate) fn adts_header(asc: &[u8], payload_len: usize) -> Option<Vec<u8>> {
    if asc.len() < 2 {
        return None;
    }
    let aot = (asc[0] >> 3) & 0x1F;
    let sf_index = ((asc[0] & 0x07) << 1) | (asc[1] >> 7);
    let channels = (asc[1] >> 3) & 0x0F;
    if aot == 0 || sf_index >= 13 {
        return None;
    }
    let frame_len = 7 + payload_len;
    if frame_len > 0x1FFF {
        return None; // ADTS 的长度字段只有 13 位
    }
    let profile = (aot - 1) & 0x03;
    Some(vec![
        0xFF,
        0xF1, // MPEG-4, layer 0, protection_absent
        (profile << 6) | (sf_index << 2) | ((channels >> 2) & 0x01),
        ((channels & 0x03) << 6) | ((frame_len >> 11) as u8 & 0x03),
        ((frame_len >> 3) & 0xFF) as u8,
        (((frame_len & 0x07) as u8) << 5) | 0x1F,
        0xFC,
    ])
}

/// 裸 ADTS 封装器（API 与 [`crate::Mp4Muxer`] 对齐）。
pub struct AdtsMuxer {
    sink: Sink,
    /// `AudioSpecificConfig`（AAC 的 extradata）。
    asc: Vec<u8>,
    declared: bool,
    packets_written: u64,
}

impl Default for AdtsMuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl AdtsMuxer {
    /// 内存落点（测试与小产物）。
    pub fn new() -> Self {
        Self::with_sink(Sink::mem())
    }

    /// 直写目标：产物**不驻留内存**。
    pub fn with_writer(w: Box<dyn Write + Send>) -> Self {
        Self::with_sink(Sink::writer(w))
    }

    fn with_sink(sink: Sink) -> Self {
        Self { sink, asc: Vec::new(), declared: false, packets_written: 0 }
    }

    /// 声明轨道：必须**恰好一条 AAC 音频轨**。
    pub fn set_tracks(&mut self, tracks: &[Track]) -> Result<()> {
        let t = match tracks {
            [one] => one,
            [] => return Err(MediaError::Container("裸 ADTS 至少需要一条轨道".into())),
            many => {
                return Err(MediaError::Unsupported(format!(
                    "裸 ADTS（`.aac`）只能装一条声音轨，这里给了 {} 条",
                    many.len()
                )))
            }
        };
        if t.kind != MediaKind::Audio || t.codec != CodecId::Aac {
            return Err(MediaError::Unsupported(format!(
                "裸 ADTS 的轨道必须是 AAC 声音，这里是 {} / {:?}（引擎不转码）",
                t.codec.name(),
                t.kind
            )));
        }
        if t.extradata.len() < 2 {
            return Err(MediaError::Container(
                "AAC 轨缺 AudioSpecificConfig（没有它算不出 ADTS 头里的采样率与声道）".into(),
            ));
        }
        self.asc = t.extradata.clone();
        self.declared = true;
        Ok(())
    }

    /// 写一条采样：裸帧补 ADTS 头；已经是 ADTS 的原样写出（不重复包）。
    pub fn write_packet(&mut self, p: &Packet) -> Result<()> {
        if !self.declared {
            return Err(MediaError::Container("写采样之前必须先 set_tracks".into()));
        }
        if is_adts(&p.data) {
            self.sink.write_all(&p.data)?;
            self.packets_written += 1;
            return Ok(());
        }
        let header = adts_header(&self.asc, p.data.len()).ok_or_else(|| {
            MediaError::Bitstream(format!(
                "算不出 ADTS 头：AudioSpecificConfig 是 {:02X?}、这一帧 {} 字节",
                &self.asc[..self.asc.len().min(4)],
                p.data.len()
            ))
        })?;
        self.sink.write_all(&header)?;
        self.sink.write_all(&p.data)?;
        self.packets_written += 1;
        Ok(())
    }

    /// 收尾（裸流没有尾元素），返回整份文件（内存模式）。
    pub fn finish(&mut self) -> Result<Vec<u8>> {
        self.sink.flush()?;
        Ok(self.sink.take_bytes())
    }

    /// 已经写出的字节数（进度上报用）。
    pub fn bytes_written(&self) -> u64 {
        self.sink.bytes_written()
    }

    pub fn packets_written(&self) -> u64 {
        self.packets_written
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use me_core::Timebase;

    fn aac_track() -> Track {
        let mut t = Track::new(1, MediaKind::Audio, CodecId::Aac, Timebase::new(44_100));
        t.sample_rate = 44_100;
        t.channels = 2;
        t.extradata = vec![0x12, 0x10]; // AAC-LC / 44.1k / 2ch
        t
    }

    #[test]
    fn wraps_raw_frames_with_adts_headers() {
        let mut m = AdtsMuxer::new();
        m.set_tracks(&[aac_track()]).unwrap();
        m.write_packet(&Packet::new(1, 0, 0, 1024, vec![0xDE, 0xAD, 0xBE, 0xEF])).unwrap();
        let data = m.finish().unwrap();
        assert_eq!(data.len(), 7 + 4, "一帧 = 7 字节头 + 载荷");

        let f = me_container::adts::parse_frame_header(&data).expect("必须是一帧合法 ADTS");
        assert_eq!(f.sample_rate, 44_100);
        assert_eq!(f.channels, 2);
        assert_eq!(f.frame_len, 11);
        assert_eq!(&data[f.header_len..], &[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn already_adts_input_is_not_double_wrapped() {
        let mut m = AdtsMuxer::new();
        m.set_tracks(&[aac_track()]).unwrap();
        let mut frame = adts_header(&[0x12, 0x10], 4).unwrap();
        frame.extend_from_slice(&[1, 2, 3, 4]);
        m.write_packet(&Packet::new(1, 0, 0, 1024, frame.clone())).unwrap();
        let data = m.finish().unwrap();
        assert_eq!(data, frame, "已经带头的输入要原样写出");
    }

    #[test]
    fn rejects_non_aac() {
        let mut t = aac_track();
        t.codec = CodecId::Opus;
        let mut m = AdtsMuxer::new();
        let err = m.set_tracks(&[t]).unwrap_err();
        assert!(format!("{err}").contains("AAC"), "{err}");
    }

    #[test]
    fn rejects_missing_config() {
        let mut t = aac_track();
        t.extradata.clear();
        let mut m = AdtsMuxer::new();
        let err = m.set_tracks(&[t]).unwrap_err();
        assert!(format!("{err}").contains("AudioSpecificConfig"), "{err}");
    }

    #[test]
    fn header_layout_matches_the_spec() {
        // AAC-LC(2) / 44.1k(index 4) / 2ch，载荷 4 字节 ⇒ 帧长 11
        let h = adts_header(&[0x12, 0x10], 4).unwrap();
        assert_eq!(h.len(), 7);
        assert_eq!(h[0], 0xFF);
        assert_eq!(h[1] & 0xF0, 0xF0, "同步字");
        assert_eq!((h[2] >> 6) & 0x03, 1, "profile = AOT-1 = 1（LC）");
        assert_eq!((h[2] >> 2) & 0x0F, 4, "采样率索引 4 = 44100");
        let frame_len = (((h[3] & 0x03) as usize) << 11) | ((h[4] as usize) << 3) | ((h[5] >> 5) as usize);
        assert_eq!(frame_len, 11);
    }
}
