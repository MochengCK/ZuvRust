//! 裸 MPEG 音频（`.mp3`）封装：把每帧原样写出去。
//!
//! 与 `.aac`（ADTS）/ `.flac` 同属"裸音频流"：没有容器、没有轨表，产物就是一串
//! 首尾相接的帧。写它不是为了新功能，而是**读写对称** —— 引擎早就读得进 `.mp3`
//! （`me-container/src/mpeg_audio.rs`），却一直没有写侧，于是"从视频里提取 MP3
//! 音轨"只能绕成 `.m4a` / `.mkv`。
//!
//! 刻意不做两件事：
//! - **不写 ID3v2 标签**：引擎不解析标签，写一个自己都不认识的标签只会让产物与
//!   输入对不上；调用方要标签自己加。
//! - **不重编码、不改帧头**：产物就是输入帧的字节串接（测试按"逐字节相同"断言）。
//!
//! 代价是**只能装 MP3**：源音频是 AAC / Opus / FLAC 时明确报错（换 `.m4a` /
//! `.opus` / `.flac`），绝不"改个扩展名发出去"。

use std::io::Write;

use me_core::{CodecId, MediaError, MediaKind, Packet, Result, Track};

use crate::sink::Sink;

/// 裸 MPEG 音频封装器（API 与 [`crate::Mp4Muxer`] 对齐）。
pub struct Mp3Muxer {
    sink: Sink,
    declared: bool,
    packets_written: u64,
}

impl Default for Mp3Muxer {
    fn default() -> Self {
        Self::new()
    }
}

impl Mp3Muxer {
    /// 内存落点（测试与小产物）。
    pub fn new() -> Self {
        Self::with_sink(Sink::mem())
    }

    /// 直写目标：产物**不驻留内存**。
    pub fn with_writer(w: Box<dyn Write + Send>) -> Self {
        Self::with_sink(Sink::writer(w))
    }

    fn with_sink(sink: Sink) -> Self {
        Self { sink, declared: false, packets_written: 0 }
    }

    /// 声明轨道：必须**恰好一条 MP3 音频轨**。
    pub fn set_tracks(&mut self, tracks: &[Track]) -> Result<()> {
        let t = match tracks {
            [one] => one,
            [] => return Err(MediaError::Container("裸 MP3 至少需要一条轨道".into())),
            many => {
                return Err(MediaError::Unsupported(format!(
                    "裸 MP3（`.mp3`）只能装一条声音轨，这里给了 {} 条",
                    many.len()
                )))
            }
        };
        if t.kind != MediaKind::Audio || t.codec != CodecId::Mp3 {
            return Err(MediaError::Unsupported(format!(
                "裸 MP3 的轨道必须是 MP3 声音，这里是 {} / {:?}（引擎不转码）",
                t.codec.name(),
                t.kind
            )));
        }
        self.declared = true;
        Ok(())
    }

    /// 写一条采样（一个 MPEG Layer III 帧）。
    ///
    /// 逐帧校验帧头：写一个"帧头对不上"的字节串出去，解码器会从那一帧开始失步，
    /// 症状是"能播但全是噪声"，比直接报错难查得多。
    pub fn write_packet(&mut self, p: &Packet) -> Result<()> {
        if !self.declared {
            return Err(MediaError::Container("写采样之前必须先 set_tracks".into()));
        }
        if me_container::mpeg_audio::parse_frame_header(&p.data).is_none() {
            return Err(MediaError::Bitstream(format!(
                "轨道 {} 的采样不是合法的 MPEG Layer III 帧（前 4 字节 {:02X?}）",
                p.track_id,
                &p.data[..p.data.len().min(4)]
            )));
        }
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
    use std::sync::Arc;

    fn mp3_track() -> Track {
        let mut t = Track::new(1, MediaKind::Audio, CodecId::Mp3, Timebase::new(44_100));
        t.sample_rate = 44_100;
        t.channels = 2;
        t
    }

    /// 一个 128 kbps / 44.1 kHz 的 MPEG-1 Layer III 帧（417 字节）。
    fn frame(fill: u8) -> Vec<u8> {
        let mut v = vec![0xFF, 0xFB, 0x90, 0x00];
        v.resize(417, fill);
        v
    }

    #[test]
    fn writes_frames_verbatim_and_reads_them_back() {
        let mut m = Mp3Muxer::new();
        m.set_tracks(&[mp3_track()]).unwrap();
        let mut want = Vec::new();
        for i in 0..10u8 {
            let f = frame(i);
            want.extend_from_slice(&f);
            m.write_packet(&Packet::new(1, i as i64 * 1152, i as i64 * 1152, 1152, f)).unwrap();
        }
        let data = m.finish().unwrap();
        assert_eq!(data, want, "裸 MP3 的产物就是帧的字节串接，一格都不能变");

        // 用引擎自己的解封装器回读
        let mut d = me_container::demux::open(
            Arc::new(me_core::Cursor::new(data)),
            "x.mp3",
        )
        .expect("产物必须能被自己的解封装器打开");
        let info = d.info();
        assert_eq!(info.tracks.len(), 1);
        assert_eq!(info.tracks[0].codec, CodecId::Mp3);
        assert_eq!(info.tracks[0].sample_rate, 44_100);
        assert_eq!(info.tracks[0].channels, 2);
        let mut n = 0;
        while let Some(p) = d.next_packet().unwrap() {
            assert_eq!(p.pts, n * 1152);
            n += 1;
        }
        assert_eq!(n, 10, "十帧要一帧不少地读回来");
    }

    #[test]
    fn refuses_wrong_codec_and_garbage_frames() {
        let mut aac = Track::new(1, MediaKind::Audio, CodecId::Aac, Timebase::new(44_100));
        aac.extradata = vec![0x12, 0x10];
        let mut m = Mp3Muxer::new();
        let err = m.set_tracks(&[aac]).unwrap_err();
        assert!(format!("{err}").contains("MP3"), "{err}");

        let mut m = Mp3Muxer::new();
        let err = m.set_tracks(&[mp3_track(), mp3_track()]).unwrap_err();
        assert!(format!("{err}").contains("一条"), "{err}");

        let mut m = Mp3Muxer::new();
        m.set_tracks(&[mp3_track()]).unwrap();
        let err = m.write_packet(&Packet::new(1, 0, 0, 1152, vec![0x21, 0x10, 0x05, 0x55])).unwrap_err();
        assert!(format!("{err}").contains("MPEG Layer III"), "{err}");
    }
}
