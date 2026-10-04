//! 产物的落点。
//!
//! ## 为什么要有这一层
//!
//! 封装器（[`crate::mp4`] / [`crate::ts`]）**必须能直写目标**，不能把产物攒在内存里：
//! 合并一个 2 GB 的视频就会要 2 GB 常驻内存，而"只搬字节"的任务本来只需要
//! 一个分片的缓冲。之前的实现是 `Vec<u8>` + 收尾时 `fs::write` 一次落盘 ——
//! 功能对，但资源占用与产物大小成正比，这是不能接受的。
//!
//! 分片 fMP4 与 MPEG-TS 都是**自包含**的：`moov` 写完之后，每个 `moof+mdat`（或每个
//! 188 字节的 TS 包）的大小在写入的那一刻就确定了，**没有任何字段需要回头改**。
//! 所以流式写盘在这里是天然可行的，不需要"先写占位再回填"。
//!
//! ## 两种落点
//!
//! - [`Sink::mem`]：攒在内存里，收尾时把整份产物交出来。单测与小产物用它；
//! - [`Sink::writer`]：直写目标（通常是 `BufWriter<File>`），内存占用与产物大小无关。
//!
//! 两者共用同一套写入路径，所以"单测验过的字节"与"真实落盘的字节"必然一致 ——
//! 如果测试走内存、生产走流式，那就是两套行为，测试就失去意义了。

use std::io::Write;

use me_core::{MediaError, Result};

/// 产物落点。
pub struct Sink {
    kind: SinkKind,
    /// 已写出的字节数（流式模式下没法问目标要，只能自己数）。
    written: u64,
}

enum SinkKind {
    Mem(Vec<u8>),
    Writer(Box<dyn Write + Send>),
}

impl Sink {
    /// 落点 = 内存缓冲。
    pub fn mem() -> Self {
        Self { kind: SinkKind::Mem(Vec::new()), written: 0 }
    }

    /// 落点 = 目标写入器（已在调用侧包好缓冲，见 `BufWriter`）。
    pub fn writer(w: Box<dyn Write + Send>) -> Self {
        Self { kind: SinkKind::Writer(w), written: 0 }
    }

    /// 预定容量的内存落点（小产物少几次扩容）。
    pub fn mem_with_capacity(cap: usize) -> Self {
        Self { kind: SinkKind::Mem(Vec::with_capacity(cap)), written: 0 }
    }

    /// 写一段字节。
    ///
    /// 写失败是**真错误**（磁盘满、管道断），必须冒到调用方：静默吞掉会产出
    /// 一个"看起来成功但内容缺失"的文件，这是最坏的结果。
    pub fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        match &mut self.kind {
            SinkKind::Mem(v) => v.extend_from_slice(bytes),
            SinkKind::Writer(w) => w.write_all(bytes).map_err(io_err)?,
        }
        self.written += bytes.len() as u64;
        Ok(())
    }

    /// 已写出的字节数。
    pub fn bytes_written(&self) -> u64 {
        self.written
    }

    /// 落点是不是内存（收尾时只有内存模式能交出产物）。
    pub fn is_mem(&self) -> bool {
        matches!(self.kind, SinkKind::Mem(_))
    }

    /// 收尾：把缓冲刷到目标。**流式模式必须调用**，否则最后一批数据留在缓冲里。
    pub fn flush(&mut self) -> Result<()> {
        if let SinkKind::Writer(w) = &mut self.kind {
            w.flush().map_err(io_err)?;
        }
        Ok(())
    }

    /// 取出内存里攒的产物（流式模式返回空 —— 字节已经在目标里了）。
    pub fn take_bytes(&mut self) -> Vec<u8> {
        match &mut self.kind {
            SinkKind::Mem(v) => std::mem::take(v),
            SinkKind::Writer(_) => Vec::new(),
        }
    }
}

fn io_err(e: std::io::Error) -> MediaError {
    MediaError::Io(format!("写产物失败：{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 一个"写进去就看得见"的假目标：用来断言流式路径写出的字节与内存路径一致。
    #[derive(Clone, Default)]
    struct SharedBuf(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl Write for SharedBuf {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn mem_sink_hands_back_what_was_written() {
        let mut s = Sink::mem();
        s.write_all(b"abc").unwrap();
        s.write_all(b"de").unwrap();
        assert_eq!(s.bytes_written(), 5);
        assert!(s.is_mem());
        assert_eq!(s.take_bytes(), b"abcde");
        // 取走之后计数仍是"写给目标的总量"（进度上报要的是这个）
        assert_eq!(s.bytes_written(), 5);
    }

    #[test]
    fn writer_sink_counts_without_buffering_the_product() {
        let shared = SharedBuf::default();
        let mut s = Sink::writer(Box::new(shared.clone()));
        s.write_all(b"hello").unwrap();
        s.write_all(b" world").unwrap();
        s.flush().unwrap();
        assert_eq!(s.bytes_written(), 11);
        assert!(!s.is_mem());
        // 流式模式不该再持有产物
        assert!(s.take_bytes().is_empty());
        assert_eq!(&*shared.0.lock().unwrap(), b"hello world");
    }
}
