//! 裸 FLAC（`.flac`）封装：`fLaC` 头 + FLAC 帧直写。
//!
//! `.flac` 不是容器：`fLaC` 魔数后面跟一串元数据块（`STREAMINFO` 必须有），
//! 之后就是**连续的 FLAC 帧**。所以这个封装器只做两件事：
//!
//! 1. 开头写一次 `fLaC` + 元数据块（由引擎内部的 `dfLa` 载荷转出来，
//!    见 [`me_container::flac_config`]）；
//! 2. 每条采样原样写出（FLAC 帧自带同步码与 CRC，不需要任何包装）。
//!
//! 只接受**一条 FLAC 音频轨**：裸流没有"轨"的概念，塞两条轨就是坏文件。

use std::io::Write;

use me_core::{CodecId, MediaError, MediaKind, Packet, Result, Track};

use crate::sink::Sink;

/// 裸 FLAC 封装器（API 与 [`crate::Mp4Muxer`] 对齐）。
pub struct FlacMuxer {
    sink: Sink,
    /// `fLaC` + 元数据块的完整头（`set_tracks` 时算好）。
    header: Vec<u8>,
    header_written: bool,
    packets_written: u64,
}

impl Default for FlacMuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl FlacMuxer {
    /// 内存落点（测试与小产物）。
    pub fn new() -> Self {
        Self::with_sink(Sink::mem())
    }

    /// 直写目标：产物**不驻留内存**。
    pub fn with_writer(w: Box<dyn Write + Send>) -> Self {
        Self::with_sink(Sink::writer(w))
    }

    fn with_sink(sink: Sink) -> Self {
        Self { sink, header: Vec::new(), header_written: false, packets_written: 0 }
    }

    /// 声明轨道：必须**恰好一条 FLAC 音频轨**。
    pub fn set_tracks(&mut self, tracks: &[Track]) -> Result<()> {
        let t = match tracks {
            [one] => one,
            [] => return Err(MediaError::Container("裸 FLAC 至少需要一条轨道".into())),
            many => {
                return Err(MediaError::Unsupported(format!(
                    "裸 FLAC（`.flac`）只能装一条声音轨，这里给了 {} 条",
                    many.len()
                )))
            }
        };
        if t.kind != MediaKind::Audio {
            return Err(MediaError::Unsupported(format!(
                "裸 FLAC 只装声音，装不下 {:?} 轨（先只保留声音再导出）",
                t.kind
            )));
        }
        if t.codec != CodecId::Flac {
            return Err(MediaError::Unsupported(format!(
                "裸 FLAC 的轨道必须是 FLAC，这里是 {}（引擎不转码）",
                t.codec.name()
            )));
        }
        self.header = me_container::flac_config::dfla_to_flac_header(&t.extradata)?;
        Ok(())
    }

    /// 写一条采样（FLAC 帧原样）。
    pub fn write_packet(&mut self, p: &Packet) -> Result<()> {
        if self.header.is_empty() {
            return Err(MediaError::Container("写采样之前必须先 set_tracks".into()));
        }
        if !self.header_written {
            self.sink.write_all(&self.header.clone())?;
            self.header_written = true;
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

    fn streaminfo() -> Vec<u8> {
        let mut b = vec![0u8; 34];
        let w = (44_100u32 << 12) | (1 << 9) | (15 << 4); // 44.1k / 2ch / 16bit
        b[10..14].copy_from_slice(&w.to_be_bytes());
        b
    }

    fn dfla() -> Vec<u8> {
        let mut v = vec![0, 0, 0, 0];
        v.extend_from_slice(&[0x80, 0x00, 0x00, 34]);
        v.extend_from_slice(&streaminfo());
        v
    }

    fn flac_track() -> Track {
        let mut t = Track::new(1, MediaKind::Audio, CodecId::Flac, Timebase::new(44_100));
        t.sample_rate = 44_100;
        t.channels = 2;
        t.extradata = dfla();
        t
    }

    /// 一个最小的"FLAC 帧"：同步码 + 一点载荷（封装器不解析帧）。
    fn frame() -> Vec<u8> {
        vec![0xFF, 0xF8, 0x69, 0x08, 0x00, 0x01, 0x02, 0x03]
    }

    #[test]
    fn writes_magic_streaminfo_and_frames() {
        let mut m = FlacMuxer::new();
        m.set_tracks(&[flac_track()]).unwrap();
        m.write_packet(&Packet::new(1, 0, 0, 4096, frame())).unwrap();
        m.write_packet(&Packet::new(1, 4096, 4096, 4096, frame())).unwrap();
        let data = m.finish().unwrap();

        assert_eq!(&data[..4], b"fLaC", "必须以 fLaC 魔数开头");
        assert_eq!(data[4] & 0x7F, 0x00, "第一个块必须是 STREAMINFO");
        assert_eq!(data[4] & 0x80, 0x80, "只有一块时要带 last 标志");
        assert_eq!(m.packets_written(), 2);

        // 头部之后应当正好是两帧
        let body = &data[4 + 4 + 34..];
        assert_eq!(body.len(), frame().len() * 2, "帧要原样写出，不许多也不许少");
        assert_eq!(&body[..frame().len()], &frame()[..]);
    }

    #[test]
    fn header_is_written_once() {
        let mut m = FlacMuxer::new();
        m.set_tracks(&[flac_track()]).unwrap();
        for i in 0..3 {
            m.write_packet(&Packet::new(1, i * 4096, i * 4096, 4096, frame())).unwrap();
        }
        let data = m.finish().unwrap();
        // 只有一处 fLaC 魔数
        let count = data.windows(4).filter(|w| *w == b"fLaC").count();
        assert_eq!(count, 1, "头只能写一次");
    }

    #[test]
    fn rejects_aac_track() {
        let mut t = flac_track();
        t.codec = CodecId::Aac;
        let mut m = FlacMuxer::new();
        let err = m.set_tracks(&[t]).unwrap_err();
        assert!(format!("{err}").contains("FLAC"), "错误要说清是编码不对：{err}");
    }

    #[test]
    fn rejects_two_tracks_and_video() {
        let mut m = FlacMuxer::new();
        let err = m.set_tracks(&[flac_track(), flac_track()]).unwrap_err();
        assert!(format!("{err}").contains("一条声音轨"), "{err}");

        let mut v = flac_track();
        v.kind = MediaKind::Video;
        let err = m.set_tracks(&[v]).unwrap_err();
        assert!(format!("{err}").contains("只装声音"), "{err}");
    }

    #[test]
    fn works_with_a_writer_sink() {
        // 直写路径（大文件合并走这条）：产物必须落进调用方给的 writer，
        // 且报告出来的字节数与"头 + 帧"一致
        let sink: Box<dyn Write + Send> = Box::new(std::io::Cursor::new(Vec::new()));
        let mut m = FlacMuxer::with_writer(sink);
        m.set_tracks(&[flac_track()]).unwrap();
        m.write_packet(&Packet::new(1, 0, 0, 4096, frame())).unwrap();
        let written = m.bytes_written();
        m.finish().unwrap();
        assert_eq!(
            written as usize,
            4 + 4 + 34 + frame().len(),
            "直写路径写出的字节数必须等于 头 + 一帧"
        );
    }
}
