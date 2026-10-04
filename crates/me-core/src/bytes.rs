//! 字节级读写：所有容器解析的公共底座。
//!
//! 两个刻意的设计：
//!
//! 1. **读越界返回 [`MediaError::NeedMore`] 而不是 panic**。
//!    容器的解析器是增量喂数据的（网络流、BT 边下边播），"这段还没到"
//!    是**正常路径**，不是 bug。返回 `NeedMore` 让调用方知道"再喂一段重来"。
//! 2. **读写都区分大小端**，方法名里带 `_be`/`_le`。
//!    MPEG-TS 与 fMP4 都是大端，RIFF 家族是小端 —— 名字不写清楚，
//!    过两个月没人记得哪个是哪个。

use crate::error::{MediaError, Result};

/// 大端/小端通吃的只读游标。
#[derive(Debug, Clone)]
pub struct ByteReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// 从 `pos` 开始读（越界会把游标钉在末尾）。
    pub fn at(buf: &'a [u8], pos: usize) -> Self {
        let mut r = Self::new(buf);
        r.pos = pos.min(buf.len());
        r
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    /// 当前游标是否已经到末尾。
    pub fn is_empty(&self) -> bool {
        self.pos >= self.buf.len()
    }

    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    /// 底层缓冲（相对整个切片，不受游标影响）。
    pub fn buffer(&self) -> &'a [u8] {
        self.buf
    }

    /// "还要 n 字节"的检查。**所有读方法的唯一越界入口**。
    fn need(&self, n: usize) -> Result<()> {
        if self.remaining() < n {
            return Err(MediaError::NeedMore { need: n, have: self.remaining() });
        }
        Ok(())
    }

    pub fn u8(&mut self) -> Result<u8> {
        self.need(1)?;
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }

    pub fn i8(&mut self) -> Result<i8> {
        Ok(self.u8()? as i8)
    }

    pub fn u16_be(&mut self) -> Result<u16> {
        let b = self.read(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    pub fn u16_le(&mut self) -> Result<u16> {
        let b = self.read(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    /// 24 位大端（fMP4 的 box 长度、TS 里也有）。
    pub fn u24_be(&mut self) -> Result<u32> {
        let b = self.read(3)?;
        Ok(u32::from_be_bytes([0, b[0], b[1], b[2]]))
    }

    pub fn u32_be(&mut self) -> Result<u32> {
        let b = self.read(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn u32_le(&mut self) -> Result<u32> {
        let b = self.read(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn u64_be(&mut self) -> Result<u64> {
        let b = self.read(8)?;
        Ok(u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    pub fn u64_le(&mut self) -> Result<u64> {
        let b = self.read(8)?;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    /// 读 n 字节并前进。
    pub fn read(&mut self, n: usize) -> Result<&'a [u8]> {
        self.need(n)?;
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    /// 看一眼接下来 n 字节，**不动游标**。
    pub fn peek(&mut self, n: usize) -> Result<&'a [u8]> {
        self.need(n)?;
        Ok(&self.buf[self.pos..self.pos + n])
    }

    /// 剩下的全部（并走到末尾）。
    pub fn take_rest(&mut self) -> &'a [u8] {
        let s = &self.buf[self.pos..];
        self.pos = self.buf.len();
        s
    }

    pub fn skip(&mut self, n: usize) -> Result<()> {
        self.need(n)?;
        self.pos += n;
        Ok(())
    }

    /// 绝对定位（越界报错，不静默钳制 —— 定位错了必须立刻暴露）。
    pub fn seek(&mut self, pos: usize) -> Result<()> {
        if pos > self.buf.len() {
            return Err(MediaError::NeedMore { need: pos, have: self.buf.len() });
        }
        self.pos = pos;
        Ok(())
    }

    /// 以 4 字节对齐跳过（fMP4 的 `stsd` 之类）。
    pub fn align4(&mut self) -> Result<()> {
        let pad = (4 - (self.pos % 4)) % 4;
        self.skip(pad)
    }
}

/// 只写不读的缓冲（算长度时回填）。
///
/// fMP4 的 box 必须先把长度写进去 —— 于是流程固定是
/// "写占位长度 → 写子内容 → 回来 patch"，所以 patch 是必备能力。
#[derive(Debug, Default, Clone)]
pub struct ByteWriter {
    buf: Vec<u8>,
}

impl ByteWriter {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn with_capacity(n: usize) -> Self {
        Self { buf: Vec::with_capacity(n) }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.buf
    }

    pub fn into_vec(self) -> Vec<u8> {
        self.buf
    }

    pub fn clear(&mut self) {
        self.buf.clear();
    }

    pub fn truncate(&mut self, n: usize) {
        self.buf.truncate(n);
    }

    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub fn u16_be(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    pub fn u16_le(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u24_be(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_be_bytes()[1..]);
    }

    pub fn u32_be(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    pub fn u32_le(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u64_be(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    pub fn u64_le(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn bytes(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }

    pub fn zeros(&mut self, n: usize) {
        self.buf.resize(self.buf.len() + n, 0);
    }

    /// 回填 4 字节大端（box 长度）。
    pub fn patch_u32_be(&mut self, offset: usize, v: u32) -> Result<()> {
        if offset + 4 > self.buf.len() {
            return Err(MediaError::container(format!(
                "回填越界：offset={offset} len={}",
                self.buf.len()
            )));
        }
        self.buf[offset..offset + 4].copy_from_slice(&v.to_be_bytes());
        Ok(())
    }

    /// 回填 2 字节大端。
    pub fn patch_u16_be(&mut self, offset: usize, v: u16) -> Result<()> {
        if offset + 2 > self.buf.len() {
            return Err(MediaError::container(format!(
                "回填越界：offset={offset} len={}",
                self.buf.len()
            )));
        }
        self.buf[offset..offset + 2].copy_from_slice(&v.to_be_bytes());
        Ok(())
    }

    /// 回填 1 字节。
    pub fn patch_u8(&mut self, offset: usize, v: u8) -> Result<()> {
        if offset >= self.buf.len() {
            return Err(MediaError::container(format!(
                "回填越界：offset={offset} len={}",
                self.buf.len()
            )));
        }
        self.buf[offset] = v;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_big_and_little_endian() {
        let data = [0x12, 0x34, 0x56, 0x78];
        let mut r = ByteReader::new(&data);
        assert_eq!(r.u16_be().unwrap(), 0x1234);
        assert_eq!(r.u16_be().unwrap(), 0x5678);

        let mut r2 = ByteReader::new(&data);
        assert_eq!(r2.u16_le().unwrap(), 0x3412);
    }

    #[test]
    fn u24_is_unsigned_not_sign_extended() {
        // 0xFF 开头是最容易写错的那一档：u24 必须当无符号
        let data = [0xFF, 0xFF, 0xFF];
        let mut r = ByteReader::new(&data);
        assert_eq!(r.u24_be().unwrap(), 0xFF_FFFF);
    }

    #[test]
    fn exhaustion_reports_need_more_not_panic() {
        let data = [1u8, 2];
        let mut r = ByteReader::new(&data);
        assert_eq!(r.u16_be().unwrap(), 0x0102);
        let e = r.u32_be().unwrap_err();
        assert!(e.is_need_more(), "越界必须是 NeedMore：{e}");
        match e {
            MediaError::NeedMore { need, have } => {
                assert_eq!(need, 4);
                assert_eq!(have, 0);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn peek_does_not_advance() {
        let data = [9u8, 8, 7, 6];
        let mut r = ByteReader::new(&data);
        assert_eq!(r.peek(2).unwrap(), &[9, 8]);
        assert_eq!(r.pos(), 0);
        assert_eq!(r.u16_be().unwrap(), 0x0908);
    }

    #[test]
    fn take_rest_empties_reader() {
        let data = [1u8, 2, 3, 4];
        let mut r = ByteReader::new(&data);
        assert_eq!(r.u8().unwrap(), 1);
        assert_eq!(r.take_rest(), &[2, 3, 4]);
        assert!(r.is_empty());
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn seek_out_of_range_is_an_error_not_a_clamp() {
        let data = [0u8; 4];
        let mut r = ByteReader::new(&data);
        assert!(r.seek(4).is_ok());
        assert!(r.seek(5).is_err());
        // 失败的 seek 不该改动游标
        assert_eq!(r.pos(), 4);
    }

    #[test]
    fn align4_skips_to_boundary() {
        let data = [0u8; 8];
        let mut r = ByteReader::new(&data);
        r.skip(1).unwrap();
        r.align4().unwrap();
        assert_eq!(r.pos(), 4);
        r.align4().unwrap();
        assert_eq!(r.pos(), 4, "已在边界上时不应前进");
    }

    #[test]
    fn writer_patches_box_size_back() {
        // 模拟 fMP4 写 box：先占位长度，写完回填
        let mut w = ByteWriter::new();
        w.u32_be(0); // 占位
        w.bytes(b"ftyp");
        w.bytes(b"isom");
        let total = w.len() as u32;
        w.patch_u32_be(0, total).unwrap();
        assert_eq!(&w.as_slice()[0..4], &total.to_be_bytes());
        assert_eq!(&w.as_slice()[4..8], b"ftyp");
    }

    #[test]
    fn writer_patch_out_of_range_errors() {
        let mut w = ByteWriter::new();
        w.u32_be(1);
        assert!(w.patch_u32_be(1, 0).is_err());
        assert!(w.patch_u8(4, 0).is_err());
    }

    #[test]
    fn writer_endianness_and_zeros() {
        let mut w = ByteWriter::new();
        w.u16_le(0x1234);
        w.zeros(2);
        w.u64_be(1);
        assert_eq!(w.as_slice(), &[0x34, 0x12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
    }
}