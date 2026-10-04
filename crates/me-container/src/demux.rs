//! 解封装器的统一接口。
//!
//! 上层（`me-stream` 的播放器、`me-mux` 的合并器、`me-cli`）**只认这个 trait**：
//! 它不知道也不该知道数据是 TS 还是 fMP4、来自本地文件还是 BT 正在下的字节。
//! 这条边界是"引擎解耦"的落点 —— 换容器、换协议都不用动上层一行。

use std::sync::Arc;

use me_core::{MediaInfo, Packet, ReadAt, Result};

use crate::probe::{sniff, ContainerFormat};

/// 解封装器：把字节流变成带完整时序的采样流。
pub trait Demuxer {
    /// 轨道与总览信息。**建立之后不可变**（合并器会先读它再决定怎么写头）。
    fn info(&self) -> MediaInfo;

    /// 取下一个采样；`Ok(None)` = 到头了。
    ///
    /// 还没读到的数据（BT 未下完、直播还没到）由 [`ReadAt`] 自己去等，
    /// **不会**以 `NeedMore` 的形式冒到这一层 —— 否则每个调用点都要处理"暂时没有"。
    fn next_packet(&mut self) -> Result<Option<Packet>>;

    /// 跳到 `pts_us` 附近的关键帧，返回实际落点（微秒）。
    fn seek(&mut self, pts_us: i64) -> Result<i64>;

    /// 总时长（微秒）；未知返回 `None`（直播、BT 未取到元信息）。
    fn duration_us(&self) -> Option<i64>;

    /// 容器名字（`"mpeg-ts"` / `"iso-bmff"` / `"mpeg-audio"`）。
    fn container(&self) -> &'static str;
}

/// 按探测结果开一个解封装器。
pub fn open(src: Arc<dyn ReadAt>, source: &str) -> Result<Box<dyn Demuxer>> {
    match sniff(&*src)? {
        ContainerFormat::MpegTs => Ok(Box::new(crate::ts::TsDemuxer::new(src, source)?)),
        ContainerFormat::IsoBmff => Ok(Box::new(crate::mp4::Mp4Demuxer::new(src, source)?)),
        ContainerFormat::Flac => Ok(Box::new(crate::flac_stream::FlacStreamDemuxer::new(src, source)?)),
        ContainerFormat::Ogg => Ok(Box::new(crate::ogg::OggOpusDemuxer::new(src, source)?)),
        ContainerFormat::Adts => Ok(Box::new(crate::adts::AdtsDemuxer::new(src, source)?)),
        ContainerFormat::MpegAudio => {
            Ok(Box::new(crate::mpeg_audio::MpegAudioDemuxer::new(src, source)?))
        }
        ContainerFormat::Unknown => Err(me_core::MediaError::container(
            "无法识别的容器，既不是 MPEG-TS、ISO BMFF、Ogg，也不是裸 FLAC / ADTS AAC / MPEG 音频",
        )),
    }
}
