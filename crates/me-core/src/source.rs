//! 随机访问字节源：**上层所有解析只认这个接口**。
//!
//! 为什么是全引擎的关键抽象：同一个 MPEG-TS 解析器要能跑在
//! 本地文件、HLS 分片串、BT 正在下载的字节上 —— 三者唯一的共同点是
//! "我可以按偏移量给你一段字节"。把它们统一成 [`ReadAt`] 之后：
//!
//! - 容器层完全不知道数据从哪来（不依赖任何网络/磁盘代码）；
//! - 「边下边播」变成**纯调度问题**：播放器读哪里，BT 层就把哪里的片提到最前；
//! - 测试可以用 [`Cursor`] 直接把内存里的字节喂进去，不起网络不起盘。
//!
//! 关键语义：**`read_at` 允许阻塞**（等 BT 把这一段下下来），
//! 允许"只给了一部分"（返回实际字节数），但**不允许返回错误来表示"暂时没有"** ——
//! 等不到就由源自己决定超时并返回 `MediaError::NeedMore`。

use std::path::Path;
use std::sync::Arc;

use crate::error::{MediaError, Result};

/// 可按偏移量读取的字节源。
pub trait ReadAt: Send + Sync {
    /// 总长度；`None` = 未知（直播、磁力还没拿到元信息）。
    fn len(&self) -> Option<u64>;

    /// 从 `offset` 读最多 `buf.len()` 字节，返回实际读到的字节数。
    ///
    /// 返回 `0` 表示真的到末尾了（不是"暂时没有"）。
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize>;

    /// 是否为空源。
    fn is_empty(&self) -> bool {
        matches!(self.len(), Some(0))
    }

    /// 读满 `n` 字节，读不满就是 `NeedMore`。
    fn read_exact_at(&self, offset: u64, n: usize) -> Result<Vec<u8>> {
        let mut v = vec![0u8; n];
        let got = self.read_at(offset, &mut v)?;
        if got < n {
            v.truncate(got);
            return Err(MediaError::NeedMore { need: n, have: got });
        }
        Ok(v)
    }

    /// 读 `n` 字节，允许短读（返回实际长度）。
    fn read_range(&self, offset: u64, n: usize) -> Result<Vec<u8>> {
        let mut v = vec![0u8; n];
        let got = self.read_at(offset, &mut v)?;
        v.truncate(got);
        Ok(v)
    }
}

impl ReadAt for Arc<dyn ReadAt> {
    fn len(&self) -> Option<u64> {
        (**self).len()
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        (**self).read_at(offset, buf)
    }
}

/// 内存源：测试、HLS 分片缓冲、小文件全都用它。
#[derive(Debug, Clone, Default)]
pub struct Cursor {
    data: Vec<u8>,
}

impl Cursor {
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    pub fn from_slice(data: &[u8]) -> Self {
        Self { data: data.to_vec() }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
}

impl ReadAt for Cursor {
    fn len(&self) -> Option<u64> {
        Some(self.data.len() as u64)
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        if offset >= self.data.len() as u64 {
            return Ok(0);
        }
        let start = offset as usize;
        let n = buf.len().min(self.data.len() - start);
        buf[..n].copy_from_slice(&self.data[start..start + n]);
        Ok(n)
    }
}

/// 本地文件源（`pread`，无游标状态，可多线程共用）。
#[derive(Debug)]
pub struct FileSource {
    file: std::fs::File,
    len: Option<u64>,
}

impl FileSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())?;
        let len = file.metadata().ok().map(|m| m.len());
        Ok(Self { file, len })
    }

    pub fn from_file(file: std::fs::File) -> Result<Self> {
        let len = file.metadata().ok().map(|m| m.len());
        Ok(Self { file, len })
    }
}

impl ReadAt for FileSource {
    fn len(&self) -> Option<u64> {
        self.len
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        if let Some(len) = self.len {
            if offset >= len {
                return Ok(0);
            }
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            if offset > i64::MAX as u64 {
                return Err(MediaError::Io(format!("偏移量超出平台能力: {offset}")));
            }
            Ok(self.file.read_at(buf, offset)?)
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::FileExt;
            Ok(self.file.seek_read(buf, offset)?)
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = offset;
            Err(MediaError::Unsupported("当前平台缺少 pread 实现".into()))
        }
    }
}

/// 把若干段字节源**在逻辑上拼成一整条**（HLS 分片串的标准接法）。
///
/// 每段自带 `logical_start`（它在拼接结果里的起始偏移）与长度。
/// 段必须按 `logical_start` 升序且**首尾相接**（HLS 的字节范围分片与
/// 普通分片混用时会有重叠，重叠部分由**先出现的那段**提供）。
#[derive(Clone, Default)]
pub struct ChainedSource {
    segments: Vec<ChainSegment>,
}

#[derive(Clone)]
struct ChainSegment {
    logical_start: u64,
    len: u64,
    source: Arc<dyn ReadAt>,
}

impl ChainedSource {
    pub fn new() -> Self {
        Self { segments: Vec::new() }
    }

    /// 追加一段。`len` 与 `source.len()` 不一致时以显式传入的 `len` 为准
    /// （BT 源在下载完成前长度不可信）。
    pub fn push(&mut self, logical_start: u64, len: u64, source: Arc<dyn ReadAt>) {
        self.segments.push(ChainSegment { logical_start, len, source });
    }

    /// 按顺序追加一段（起始偏移自动接在现有末尾）。
    pub fn append(&mut self, len: u64, source: Arc<dyn ReadAt>) {
        let start = self.len().unwrap_or(0);
        self.push(start, len, source);
    }

    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// 找到覆盖 `offset` 的段（二分）。
    fn locate(&self, offset: u64) -> Option<usize> {
        if self.segments.is_empty() {
            return None;
        }
        let idx = match self
            .segments
            .binary_search_by(|s| s.logical_start.cmp(&offset))
        {
            Ok(i) => i,
            Err(0) => return None,
            Err(i) => i - 1,
        };
        let seg = &self.segments[idx];
        if offset < seg.logical_start + seg.len {
            Some(idx)
        } else {
            None
        }
    }
}

impl ReadAt for ChainedSource {
    fn len(&self) -> Option<u64> {
        // 空链是"长 0"，不是"长度未知"：拼接链的长度永远是可数的，
        // 返回 None 会让上层的 `is_empty()` 判断失灵（没有段却不算空）。
        match self.segments.last() {
            Some(s) => Some(s.logical_start + s.len),
            None => Some(0),
        }
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        let total = match self.len() {
            Some(t) => t,
            None => return Ok(0),
        };
        if offset >= total || buf.is_empty() {
            return Ok(0);
        }
        let mut done = 0usize;
        let mut at = offset;
        while done < buf.len() && at < total {
            let Some(idx) = self.locate(at) else {
                // 段之间有空隙（HLS 的字节范围没接上）：就地停住，
                // 让调用方知道"这里读不到"，而不是把后面的数据挪过来冒充。
                break;
            };
            let seg = &self.segments[idx];
            let in_seg = at - seg.logical_start;
            let avail = (seg.len - in_seg) as usize;
            let want = (buf.len() - done).min(avail);
            let got = seg.source.read_at(in_seg, &mut buf[done..done + want])?;
            if got == 0 {
                break;
            }
            done += got;
            at += got as u64;
        }
        Ok(done)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_reads_and_stops_at_end() {
        let c = Cursor::from_slice(b"hello world");
        assert_eq!(c.len(), Some(11));
        let mut buf = [0u8; 5];
        assert_eq!(c.read_at(6, &mut buf).unwrap(), 5);
        assert_eq!(&buf, b"world");
        // 越界读返回 0 而不是错误
        assert_eq!(c.read_at(11, &mut buf).unwrap(), 0);
        assert_eq!(c.read_at(100, &mut buf).unwrap(), 0);
    }

    #[test]
    fn cursor_short_read_near_end() {
        let c = Cursor::from_slice(b"abc");
        let mut buf = [0u8; 8];
        assert_eq!(c.read_at(2, &mut buf).unwrap(), 1);
        assert_eq!(buf[0], b'c');
    }

    #[test]
    fn read_exact_at_reports_need_more() {
        let c = Cursor::from_slice(b"abc");
        assert!(c.read_exact_at(0, 3).is_ok());
        let e = c.read_exact_at(0, 4).unwrap_err();
        assert!(e.is_need_more());
        match e {
            MediaError::NeedMore { need, have } => {
                assert_eq!(need, 4);
                assert_eq!(have, 3);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn chain_stitches_segments_in_order() {
        let mut c = ChainedSource::new();
        c.push(0, 3, Arc::new(Cursor::from_slice(b"abc")));
        c.push(3, 3, Arc::new(Cursor::from_slice(b"def")));
        assert_eq!(c.len(), Some(6));
        assert_eq!(c.read_exact_at(0, 6).unwrap(), b"abcdef");
        // 跨段读（起点在上一段、终点在下一段）
        assert_eq!(c.read_exact_at(2, 3).unwrap(), b"cde");
    }

    #[test]
    fn chain_append_uses_running_length() {
        let mut c = ChainedSource::new();
        c.append(2, Arc::new(Cursor::from_slice(b"ab")));
        c.append(2, Arc::new(Cursor::from_slice(b"cd")));
        assert_eq!(c.len(), Some(4));
        assert_eq!(c.read_exact_at(1, 2).unwrap(), b"bc");
        assert_eq!(c.segment_count(), 2);
    }

    #[test]
    fn chain_gap_is_not_papered_over() {
        // 第二段从 5 开始，3..5 是空洞：读到这里必须停住，
        // 不能把第二段的数据挪到前面来（那会让解析器读到错位的字节）
        let mut c = ChainedSource::new();
        c.push(0, 3, Arc::new(Cursor::from_slice(b"abc")));
        c.push(5, 2, Arc::new(Cursor::from_slice(b"xy")));
        assert_eq!(c.len(), Some(7));
        let mut buf = [0u8; 6];
        assert_eq!(c.read_at(0, &mut buf).unwrap(), 3, "只应返回第一段");
        assert_eq!(&buf[..3], b"abc");
    }

    #[test]
    fn chain_tolerates_source_longer_than_declared() {
        // BT 源长度不可信：声明 2 字节但底层有 5 字节，只能给 2
        let mut c = ChainedSource::new();
        c.push(0, 2, Arc::new(Cursor::from_slice(b"abcde")));
        assert_eq!(c.len(), Some(2));
        let mut buf = [0u8; 8];
        assert_eq!(c.read_at(0, &mut buf).unwrap(), 2);
    }

    #[test]
    fn chain_empty_is_inert() {
        let c = ChainedSource::new();
        assert_eq!(c.len(), Some(0));
        assert!(c.is_empty());
        let mut buf = [0u8; 4];
        assert_eq!(c.read_at(0, &mut buf).unwrap(), 0);
    }

    #[test]
    fn arc_dyn_read_at_delegates() {
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::from_slice(b"xyz"));
        assert_eq!(src.len(), Some(3));
        assert_eq!(src.read_exact_at(0, 3).unwrap(), b"xyz");
    }

    #[test]
    fn file_source_roundtrip() {
        let dir = std::env::temp_dir().join("me-core-src-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.bin");
        std::fs::write(&p, b"0123456789").unwrap();
        let f = FileSource::open(&p).unwrap();
        assert_eq!(f.len(), Some(10));
        assert_eq!(f.read_exact_at(4, 3).unwrap(), b"456");
        assert_eq!(f.read_at(10, &mut [0u8; 4]).unwrap(), 0);
        std::fs::remove_file(&p).ok();
    }
}