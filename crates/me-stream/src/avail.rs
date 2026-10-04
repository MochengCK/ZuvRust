//! 可用性图：**"这个文件的哪些字节已经真的下到了"**。
//!
//! ## 为什么必须由宿主告诉引擎
//!
//! 下载引擎先**扩文件**、后填数据 —— 还没下到的位置读出来是零。零字节与真实数据
//! 在磁盘上无法区分，于是容器/码流层会把空洞当成损坏（实测：H.264 报
//! `采样不是 Annex-B`，播放直接终止）。真相只有**下载引擎的片位图**知道，
//! 而位图在宿主手里（它管着下载任务），所以由宿主翻译成一个小 JSON 文件、
//! 秒级刷新，引擎在等待期间反复读它。
//!
//! ## 文件格式（宿主写、引擎读）
//!
//! ```json
//! {"fileOffset": 0, "pieceLength": 4194304, "total": 4294967296, "bitfield": "a1b2…"}
//! ```
//!
//! - `fileOffset`：这个文件在**种子总数据**里的起点（多文件种子必须算对，
//!   位图是按整份种子编的）；
//! - `bitfield`：种子片位图，十六进制，**每片 1 bit、字节内高位在前**（wire 语义，
//!   与下载引擎 `bitfield` 字段一致）；
//! - 文件不存在或过期（宿主不再刷新）**一律按"不知道"处理** —— 这时退回
//!   老行为（照读），绝不因为图没了就卡住播放。

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// 图过期阈值：宿主每秒刷一次，超过这么久没动静说明它不在了（播放已停/应用退出）。
pub const STALE_AFTER: Duration = Duration::from_secs(10);

/// 解析出来的可用性信息（纯数据，不碰文件系统）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Avail {
    /// 本文件在种子总数据里的起点。
    pub file_offset: u64,
    /// 片长（种子的 piece length）。
    pub piece_length: u64,
    /// 种子总长度（用于片数上限推断；0 = 未知，按位图长度推断）。
    pub total: u64,
    /// 片位图（原始字节）。
    pub bits: Vec<u8>,
}

impl Avail {
    /// 从宿主写的 JSON 解析。任何字段不对都返回 `None`（调用方按"不知道"处理）。
    pub fn parse(text: &str) -> Option<Avail> {
        let v: serde_json::Value = serde_json::from_str(text).ok()?;
        let file_offset = v.get("fileOffset")?.as_u64()?;
        let piece_length = v.get("pieceLength")?.as_u64()?;
        if piece_length == 0 {
            return None;
        }
        let total = v.get("total").and_then(|t| t.as_u64()).unwrap_or(0);
        let hex = v.get("bitfield")?.as_str()?;
        let bits = decode_hex(hex)?;
        if bits.is_empty() {
            return None;
        }
        Some(Avail { file_offset, piece_length, total, bits })
    }

    /// 片号是否已下载。
    pub fn has_piece(&self, index: u64) -> bool {
        let byte = (index / 8) as usize;
        if byte >= self.bits.len() {
            // 位图不够长：按"没有"处理（保守 —— 宁可多等一会儿）
            return false;
        }
        self.bits[byte] >> (7 - (index % 8)) & 1 == 1
    }

    /// **文件内**区间 `[lo, hi)` 是否已全部可用（半开区间；`hi <= lo` 视为可用）。
    pub fn covers(&self, lo: u64, hi: u64) -> bool {
        if hi <= lo {
            return true;
        }
        let first = (self.file_offset + lo) / self.piece_length;
        let last = (self.file_offset + hi - 1) / self.piece_length;
        (first..=last).all(|p| self.has_piece(p))
    }
}

/// 一个"随文件刷新"的可用性图（带路径与读入时刻）。
#[derive(Debug, Clone)]
pub struct AvailFile {
    pub avail: Avail,
    /// 读到这份内容的时刻（用于判过期 —— 用读取时刻而不是文件 mtime：
    /// 宿主用"临时文件 + rename"原子替换，mtime 语义在跨文件系统时不可靠）。
    pub loaded_at: SystemTime,
    pub path: PathBuf,
}

impl AvailFile {
    /// 读一份（解析失败/不新鲜都返回 `None`）。
    pub fn load(path: &Path) -> Option<AvailFile> {
        let text = std::fs::read_to_string(path).ok()?;
        let avail = Avail::parse(&text)?;
        Some(AvailFile { avail, loaded_at: SystemTime::now(), path: path.to_path_buf() })
    }

    pub fn is_fresh(&self) -> bool {
        self.loaded_at.elapsed().map(|d| d < STALE_AFTER).unwrap_or(false)
    }
}

fn decode_hex(s: &str) -> Option<Vec<u8>> {
    let s = s.trim();
    if s.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let hi = (bytes[i] as char).to_digit(16)?;
        let lo = (bytes[i + 1] as char).to_digit(16)?;
        out.push((hi * 16 + lo) as u8);
        i += 2;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4 片、每片 1 KiB、文件从第 2 片（offset 2048）开始的两片（= 文件长 2048）。
    fn sample(bits: &str) -> Avail {
        Avail::parse(&format!(
            r#"{{"fileOffset":2048,"pieceLength":1024,"total":4096,"bitfield":"{bits}"}}"#
        ))
        .expect("样例要能解析")
    }

    #[test]
    fn parses_host_written_json() {
        let a = sample("c0"); // 1100 0000 → 片 0、1 有
        assert_eq!(a.file_offset, 2048);
        assert_eq!(a.piece_length, 1024);
        assert!(a.has_piece(0) && a.has_piece(1));
        assert!(!a.has_piece(2) && !a.has_piece(3));
    }

    #[test]
    fn garbage_is_treated_as_unknown() {
        assert!(Avail::parse("").is_none());
        assert!(Avail::parse("not json").is_none());
        assert!(Avail::parse(r#"{"fileOffset":0,"pieceLength":0,"bitfield":"ff"}"#).is_none());
        assert!(Avail::parse(r#"{"fileOffset":0,"pieceLength":1,"bitfield":"f"}"#).is_none(), "奇数长度 hex 要拒绝");
        assert!(Avail::parse(r#"{"fileOffset":0,"pieceLength":1,"bitfield":""}"#).is_none());
    }

    #[test]
    fn covers_checks_every_piece_the_range_touches() {
        // 文件从片 2 开始（fileOffset=2048），文件内偏移 0 就是片 2
        let a = sample("c0"); // 片 0、1 有 → 文件的两片（片 2、3）都没有
        assert!(!a.covers(0, 1), "第一片就没下到");
        let b = sample("f0"); // 片 0..3 都有
        assert!(b.covers(0, 2048), "整段都在图里");
        assert!(b.covers(1024, 2048), "跨片也算");
        let c = sample("e0"); // 只有片 0、1、2 → 文件第二片（片 3）没有
        assert!(c.covers(0, 1024), "文件第一片（片 2）有");
        assert!(!c.covers(1024, 1025), "文件第二片（片 3）没有");
        assert!(c.covers(1024, 1024), "空区间视为可用");
        // 位图不够长 → 保守地当作没有
        let short = sample("80");
        assert!(!short.covers(1024, 1025));
    }

    #[test]
    fn loads_from_a_file_and_notices_age() {
        let dir = std::env::temp_dir().join(format!("me-avail-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("map.json");
        std::fs::write(&path, r#"{"fileOffset":0,"pieceLength":1024,"bitfield":"80"}"#).unwrap();
        let f = AvailFile::load(&path).expect("要能读出来");
        assert!(f.is_fresh(), "刚写的图是新鲜的");
        assert!(f.avail.has_piece(0));
        std::fs::write(&path, "broken").unwrap();
        assert!(AvailFile::load(&path).is_none(), "坏内容按不知道处理");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
