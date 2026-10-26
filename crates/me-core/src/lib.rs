//! me-core —— 媒体引擎的**基础类型层**。
//!
//! 这一层刻意只有"数据形状"，没有任何算法与 IO：
//! 上层的容器解析（TS / fMP4）、合并、协议、BT 全都依赖这里的类型，
//! 所以它必须是**稳定的**、**可序列化的**、**不依赖任何第三方媒体库**的。
//!
//! 设计上的三条硬约定：
//!
//! 1. **时间一律用「整数刻度 + 时基」表达**，绝不用浮点秒。
//!    浮点在长片子上会累积误差（一部 2 小时的片子，微秒级误差足够让音画错开一帧）。
//!    [`Track::timescale`] 是这一条轨道每秒的刻度数，[`Packet::pts`] 是刻度值；
//!    需要"秒"的时候显式调 [`Track::pts_to_us`]。
//! 2. **采样（[`Packet`]）自带完整时序**（pts/dts/duration），不靠"上一条推下一条"。
//!    合并两条轨道时要在时间轴上交织，靠推算出来的时间必然错。
//! 3. **编解码器只描述"是什么"（[`CodecId`]），不含解码器**。
//!    引擎的容器/合并/协议层完全不需要解码器；解码是另一层的事
//!    （走平台解码器，见 DEVELOPMENT.md 的分层说明）。

/// 引擎名称（CLI / JSON-RPC / SDK 展示用）。
pub const ENGINE_NAME: &str = "ZuvRust";

/// 引擎版本。与工作区 `Cargo.toml` 的 `version` 保持同步；
/// `version --json`、JSON-RPC `initialize`、CLI 测试都只读这一个常量。
pub const ENGINE_VERSION: &str = "0.2.1";

pub mod bits;
pub mod bytes;
pub mod error;
pub mod media;
pub mod source;
pub mod time;

pub use bits::{add_emulation_prevention, remove_emulation_prevention, BitReader};
pub use bytes::{ByteReader, ByteWriter};
pub use error::{MediaError, Result};
pub use media::{CodecId, MediaInfo, MediaKind, Packet, Track};
pub use source::{ChainedSource, Cursor, FileSource, ReadAt};
pub use time::{Rational, Timebase};

#[cfg(test)]
mod version_tests {
    use super::*;

    /// `ENGINE_VERSION` 是手写常量，必须与 Cargo 工作区版本严格一致。
    ///
    /// 两者一旦漂移，CLI、JSON-RPC、SDK 与发布说明会对外宣称不同版本号；
    /// 这条测试让版本升级时至少有一处会失败。
    #[test]
    fn engine_version_matches_cargo_version() {
        assert_eq!(
            ENGINE_VERSION,
            env!("CARGO_PKG_VERSION"),
            "me-core::ENGINE_VERSION 与工作区 Cargo.toml 的 version 不一致"
        );
    }
}
