//! me-stream —— 流层（把协议/文件的字节接进容器，提供播放缓冲与寻址）。
//!
//! 三块：
//!
//! - [`source`]：把"文件 / 还在增长的下载文件 / HTTP / HLS / DASH"统一成
//!   [`me_core::ReadAt`]，并提供取消标志；
//! - [`playback`]：播放器（有界缓冲、寻址、**播放头上报**）；
//! - 依赖方向：只依赖 me-core / me-container / me-net。
//!
//! ## 与下载引擎的边界
//!
//! **BT 下载由下载引擎（XferRust）负责，媒体引擎只负责播放。**
//! 所以这里没有 BT 网络会话；"边下边播"靠 [`source::GrowingFileSource`]
//! 读另一个进程正在写的文件，靠 [`playback::PlayheadHint`] 把播放位置报出去。

pub mod avail;
pub mod playback;
pub mod source;

// 只给测试用：一个极小的本地 HTTP 服务器（起真 socket、说真 HTTP），
// 用来验证"取清单 → 拼分片"整条链路走的是真协议，而不是被 mock 掉
#[cfg(test)]
mod testutil;

pub use playback::{Player, PlayerConfig, PlayerStats, PlayheadHint};
pub use source::{
    classify, default_client, open_dash_tracks, open_hls, open_input, CancelFlag, DashTrack,
    GrowingFileSource, GrowingOptions, HlsSource, InputKind, OpenOptions, OpenedSource, RangeSource,
};
