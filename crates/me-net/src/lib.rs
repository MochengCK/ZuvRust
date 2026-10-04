//! me-net —— 多协议层（HLS / DASH / HTTP / URL / 自研加密）。
//!
//! 这一层只做「把远端的分片变成可随机访问的字节」这件事，刻意**不碰容器解析**：
//!
//! - [`url`]：自研 URL 解析与 RFC 3986 引用解析（HLS 相对 URI 的生死线）；
//! - [`http`]：基于 ureq 3 的取流，统一请求头过滤 + Range 语义判定；
//! - [`crypto`]：自研 AES（128/192/256，CBC/CTR/ECB）+ Base64 + PKCS7，
//!   给 HLS 的 `EXT-X-KEY` 用；
//! - [`hls`]：媒体/主清单解析与分片计划；
//! - [`dash`]：MPD 解析（含自研 XML）与分片 URL 展开。
//!
//! 之所以全部自研而不引入 `m3u8-rs` / `dash-mpd` / `aes` / `url`：
//! 本工程的硬性约束是「不依赖任何老牌媒体引擎与重型协议库」，
//! 而且 HLS 方言极多，第三方库"严格校验"的行为反而会让我们放不了很多源。

pub mod crypto;
pub mod dash;
pub mod hls;
pub mod http;
pub mod url;
pub mod xml;

// 关键类型在 crate 根 re-export：上层的 me-stream / me-cli 只想写
// `me_net::MediaPlaylist`，不该被迫记住每个类型躺在哪个子模块里。
// 子模块路径照旧可用（`me_net::hls::parse` 与 `me_net::parse_hls` 等价）。
pub use dash::{
    pick_representation, AdaptationSet, Mpd, Period, Representation, SegmentList,
    SegmentTemplate, TimelineEntry,
};
pub use hls::{
    decrypt, parse as parse_hls, pick_variant, ByteRange, InitMap, KeyInfo, KeyMethod,
    MasterPlaylist, MediaPlaylist, Part, PartInf, Playlist, Segment, ServerControl, Variant,
};
pub use http::{HttpClient, HttpConfig, HttpHead, RangeResponse};
pub use url::{percent_decode, percent_encode, resolve as resolve_url, Url};
pub use xml::XmlNode;

/// 解析 DASH MPD（短名字；`dash::parse_mpd` 是同一个函数）。
pub use dash::parse_mpd as parse_dash;
