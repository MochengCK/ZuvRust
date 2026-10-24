//! me-container —— 容器层（MPEG-TS / fMP4 / Ogg / 裸 ADTS AAC / 裸 FLAC / 裸 MPEG 音频）。

pub mod ac3;
pub mod adts;
pub mod codec;
pub mod demux;
pub mod flac_config;
pub mod flac_stream;
pub mod mkv;
pub mod mp4;
pub mod mpeg_audio;
pub mod ogg;
pub mod opus_config;
pub mod probe;
pub mod ts;

pub use ac3::{parse_dac3, Dac3};
pub use adts::AdtsDemuxer;
pub use codec::{is_random_access_nal, parse_h264_sps, parse_h265_sps, split_annexb, SpsInfo};
pub use demux::{open, Demuxer};
pub use flac_config::{dfla_to_flac_header, flac_header_to_dfla, streaminfo_of_dfla, FLAC_MAGIC};
pub use flac_stream::FlacStreamDemuxer;
pub use mkv::MkvDemuxer;
pub use mp4::Mp4Demuxer;
pub use mpeg_audio::MpegAudioDemuxer;
pub use ogg::OggOpusDemuxer;
pub use opus_config::{dops_to_opus_head, opus_head_to_dops, OpusConfig};
pub use probe::{probe, ContainerFormat};
pub use ts::TsDemuxer;
