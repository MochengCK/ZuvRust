//! me-bt —— BT 层（含「边下边播」所需的按播放位置优先调度）。
//!
//! 分五块，各自只干一件事，互相之间只通过数据（不是控制流）耦合：
//!
//! - [`bencode`]：BT 的通用序列化格式，`info_hash` 的"原文切片"坑在这里处理；
//! - [`metainfo`]：`.torrent` / 磁力 → 结构化的种子描述；
//! - [`storage`]：片 → 可随机访问的字节流（实现 [`me_core::source::ReadAt`]）；
//! - [`picker`]：**按播放头位置决定先下哪片**，这是边下边播与"顺序下载"的分水岭；
//! - [`peer`] / [`tracker`]：线协议与 tracker 交互，只做编解码与调度，
//!   **不做任何网络 IO**（TCP/UDP 的会话管理是上层的事）。

pub mod bencode;
pub mod metainfo;
pub mod peer;
pub mod picker;
pub mod storage;
pub mod tracker;

pub use metainfo::MetaInfo;
pub use picker::{PiecePicker, PickerConfig, PickerStats, Prio};

/// SHA-1（BT 用它算 `info_hash` 与每片的哈希）。
///
/// 放在 crate 根而不是某个模块里：`metainfo`（算 info_hash）与 `storage`（校验片）
/// 都要用，谁都不该依赖谁。参数与返回都刻意用"裸切片 / 定长数组"，
/// 不暴露 `sha1` crate 的 `GenericArray` —— 否则这个第三方类型会顺着
/// `piece_hashes: Vec<[u8; 20]>` 这样的字段漏进整个 crate 的公开 API。
pub(crate) fn sha1(data: &[u8]) -> [u8; 20] {
    use sha1::{Digest, Sha1};
    let mut hasher = Sha1::new();
    hasher.update(data);
    let digest = hasher.finalize();
    let mut out = [0u8; 20];
    out.copy_from_slice(&digest);
    out
}
