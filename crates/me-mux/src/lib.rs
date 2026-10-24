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

pub mod adts;
pub mod convert;
pub mod flac;
pub mod interleave;
pub mod mkv;
pub mod mp4;
pub mod sink;
pub mod ts;

pub use adts::AdtsMuxer;
pub use convert::{
    annexb_to_avcc, annexb_to_avcc_extradata, avcc_to_annexb, avcc_to_annexb_extradata,
    extradata_to_annexb, extradata_to_avcc, looks_like_avcc, nal_length_size,
};
pub use flac::FlacMuxer;
pub use interleave::{interleave_all, Interleaver};
pub use mkv::{DocType, MkvMuxer};
pub use mp4::Mp4Muxer;
pub use sink::Sink;
pub use ts::TsMuxer;
