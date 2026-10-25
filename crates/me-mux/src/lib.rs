//! me-mux —— 合并层（多轨交织 + MPEG-TS / fMP4 封装）。
//!
//! 这一层**只搬字节，不重编码**：
//!
//! - [`interleave`]：把多条轨道的采样按时间交织成一条流（容器要求数据大致按时间排列）；
//! - [`convert`]：`Annex-B ↔ 长度前缀`、`extradata` 的两种形态互转；
//! - [`ts`]：MPEG-TS 封装（PAT/PMT/PCR 周期、AAC 补 ADTS、关键帧重发参数集）；
//! - [`mkv`]：Matroska 封装（`EBML + Segment{Info, Tracks, Cluster*}`）；
//! - [`mp4`]：分片 fMP4 封装（`moov` + `[sidx + moof/mdat]*`）。
//!
//! 约定（与全引擎一致）：
//! - 时间一律**整数刻度 + 时基**，跨时基比较先换算成微秒；
//! - `Packet::data` 里**视频一律是 Annex-B**，音频按来源形态（TS 里是 ADTS、
//!   fMP4 里是裸帧），转换由这里的封装器负责；
//! - 产物要么是**正确的**，要么**报错** —— 绝不静默产出坏文件；
//! - 封装器**直写落点**（见 [`sink`]）：产物不攒在内存里，内存占用与产物大小无关。
//!
//! `convert` 永远可用；其余封装器由 `mux` feature 门控（默认开启）。只做解码/
//! 播放的构建可以关掉它，省下整个写方向。

pub mod convert;

#[cfg(feature = "mux")]
pub mod adts;
#[cfg(feature = "mux")]
pub mod flac;
#[cfg(feature = "mux")]
pub mod interleave;
#[cfg(feature = "mux")]
pub mod mkv;
#[cfg(feature = "mux")]
pub mod mp4;
#[cfg(feature = "mux")]
pub mod sink;
#[cfg(feature = "mux")]
pub mod ts;

#[cfg(feature = "mux")]
pub use adts::AdtsMuxer;
pub use convert::{
    annexb_to_avcc, annexb_to_avcc_extradata, avcc_to_annexb, avcc_to_annexb_extradata,
    extradata_to_annexb, extradata_to_avcc, looks_like_avcc, nal_length_size,
};
#[cfg(feature = "mux")]
pub use flac::FlacMuxer;
#[cfg(feature = "mux")]
pub use interleave::{interleave_all, Interleaver};
#[cfg(feature = "mux")]
pub use mkv::{DocType, MkvMuxer};
#[cfg(feature = "mux")]
pub use mp4::Mp4Muxer;
#[cfg(feature = "mux")]
pub use sink::Sink;
#[cfg(feature = "mux")]
pub use ts::TsMuxer;
