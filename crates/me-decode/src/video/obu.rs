//! AV1 的 OBU 层（规范 §5.3 / §6.2）：把一个压缩采样切成一个个"数据单元"。
//!
//! 这是自研 AV1 软解的第一站，也是**唯一不依赖任何上下文的一站**：OBU 的头
//! 与长度字段自己就说明了"这一段是什么、有多长"，不需要序列头、不需要参考帧、
//! 不需要 CDF 状态。所以它能单独验证（头部位域、扩展头、缺长度字段三种形态）。
//!
//! 一个必须记住的形态差异：**AV1 的采样没有 Annex-B 那种起始码**，每个 OBU
//! 自带 `leb128` 长度（`obu_has_size_field = 1`）。ISO BMFF 里规范明确要求
//! 必须带长度字段；不带的情况只可能是"整段就是最后一个 OBU"，此时长度由缓冲
//! 边界给出 —— 解析层如实区分这两件事，而不是猜。
//!
//! 之所以既认识 `OBU_FRAME`（帧头 + 瓦片挤在一个 OBU 里，最常见）也认识
//! `OBU_FRAME_HEADER` + `OBU_TILE_GROUP`（拆开写）：真实码流里编码器两种都用。

use me_core::{MediaError, Result};

use super::bitreader::BitReader;

/// OBU 类型（规范 §6.2.2 的表）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObuType {
    /// 保留（不该出现在合规码流里）
    Reserved,
    /// 序列头：整条流的全局参数（profile、位深、工具开关……）
    SequenceHeader,
    /// 时间分隔符：一帧的开始标记，不含参数
    TemporalDelimiter,
    /// 帧头（与瓦片分开写时）
    FrameHeader,
    /// 瓦片组（与帧头分开写时）
    TileGroup,
    /// 元数据（HDR 静元数据、缩放等）
    Metadata,
    /// 帧：帧头 + 瓦片挤在同一个 OBU 里
    Frame,
    /// 冗余帧头（用于丢包恢复）
    RedundantFrameHeader,
    /// 瓦片列表
    TileList,
    /// 填充
    Padding,
}

impl ObuType {
    fn from_code(code: u8) -> Self {
        match code {
            1 => ObuType::SequenceHeader,
            2 => ObuType::TemporalDelimiter,
            3 => ObuType::FrameHeader,
            4 => ObuType::TileGroup,
            5 => ObuType::Metadata,
            6 => ObuType::Frame,
            7 => ObuType::RedundantFrameHeader,
            8 => ObuType::TileList,
            15 => ObuType::Padding,
            _ => ObuType::Reserved,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ObuType::Reserved => "reserved",
            ObuType::SequenceHeader => "sequence-header",
            ObuType::TemporalDelimiter => "temporal-delimiter",
            ObuType::FrameHeader => "frame-header",
            ObuType::TileGroup => "tile-group",
            ObuType::Metadata => "metadata",
            ObuType::Frame => "frame",
            ObuType::RedundantFrameHeader => "redundant-frame-header",
            ObuType::TileList => "tile-list",
            ObuType::Padding => "padding",
        }
    }

    /// 这个 OBU 携带压缩数据（而不只是参数/分隔符）。
    ///
    /// 用途：诊断与统计（"这条流有多少个真正带像素的 OBU"）。
    pub fn carries_bitstream(self) -> bool {
        matches!(
            self,
            ObuType::Frame | ObuType::FrameHeader | ObuType::TileGroup | ObuType::TileList
        )
    }
}

/// OBU 头（解析后的形态）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObuHeader {
    pub obu_type: ObuType,
    /// 时间层（可分层编码用；没有扩展头时为 0）
    pub temporal_id: u8,
    /// 空间层（同上）
    pub spatial_id: u8,
    /// 载荷前有没有 `leb128` 长度字段
    pub has_size_field: bool,
}

/// 一个 OBU：头 + 载荷切片（**零拷贝**，切片指向原始采样缓冲）。
#[derive(Debug, Clone, Copy)]
pub struct Obu<'a> {
    pub header: ObuHeader,
    pub payload: &'a [u8],
}

/// 顺序遍历一段采样里的 OBU。
///
/// 为什么是迭代器而不是"先解析成一堆结构"：播放路径上一个采样就是几 MB，
/// 多一次全量解析就多一份拷贝；而解析层要的正是"挨个处理"。
pub struct ObuIter<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ObuIter<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// 取下一个 OBU；`Ok(None)` = 这段数据看完了。
    pub fn next_obu(&mut self) -> Result<Option<Obu<'a>>> {
        if self.pos >= self.data.len() {
            return Ok(None);
        }
        let rest = &self.data[self.pos..];
        let mut r = BitReader::new(rest);
        let header = parse_header(&mut r)?;
        // 头部永远是整数个字节（含扩展头），所以这里必然对齐
        if !r.byte_aligned() {
            return Err(MediaError::Bitstream("OBU 头没有落在字节边界上".into()));
        }
        if header.has_size_field {
            let size = r.leb128()? as usize;
            // `take_bytes` 自带越界检查：截断的 OBU 报 NeedMore，而不是给半截载荷
            let payload = r.take_bytes(size)?;
            // 位读器已经吃到了载荷之后（头 + 长度字段 + 载荷）
            self.pos += r.bits_read() / 8;
            Ok(Some(Obu { header, payload }))
        } else {
            // 没有长度字段：载荷一直到这段缓冲末尾（只在最后一个 OBU 上合法）
            let consumed = r.bits_read() / 8;
            let payload = &rest[consumed..];
            self.pos += rest.len();
            Ok(Some(Obu { header, payload }))
        }
    }
}

/// 从头里解析出类型/层标识（读完后位读器停在"头之后、长度字段之前"）。
fn parse_header(r: &mut BitReader<'_>) -> Result<ObuHeader> {
    if r.f(1)? != 0 {
        return Err(MediaError::Bitstream("OBU 头的 forbidden 位必须是 0".into()));
    }
    let obu_type = ObuType::from_code(r.f(4)? as u8);
    let extension = r.bit()?;
    let has_size_field = r.bit()?;
    if r.f(1)? != 0 {
        return Err(MediaError::Bitstream("OBU 头的保留位必须是 0".into()));
    }
    let mut temporal_id = 0u8;
    let mut spatial_id = 0u8;
    if extension {
        temporal_id = r.f(3)? as u8;
        spatial_id = r.f(2)? as u8;
        if r.f(3)? != 0 {
            return Err(MediaError::Bitstream("OBU 扩展头的保留位必须是 0".into()));
        }
    }
    Ok(ObuHeader { obu_type, temporal_id, spatial_id, has_size_field })
}

/// 一段采样里的 OBU 数（诊断/统计用；坏数据在第一个错误处停下并报出来）。
pub fn count_obus(data: &[u8]) -> Result<usize> {
    let mut it = ObuIter::new(data);
    let mut n = 0;
    while it.next_obu()?.is_some() {
        n += 1;
    }
    Ok(n)
}

/// 找出第一个序列头 OBU 的载荷（"取 AV1 序列头"只有这一处）。
pub fn first_sequence_header(data: &[u8]) -> Option<&[u8]> {
    let mut it = ObuIter::new(data);
    while let Ok(Some(obu)) = it.next_obu() {
        if obu.header.obu_type == ObuType::SequenceHeader {
            return Some(obu.payload);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 造一个带长度字段的 OBU：`[(type<<3)|has_size, size, payload…]`
    fn obu_with_size(obu_type: u8, payload: &[u8]) -> Vec<u8> {
        let mut v = vec![(obu_type << 3) | 0b0000_0010];
        v.push(payload.len() as u8);
        v.extend_from_slice(payload);
        v
    }

    #[test]
    fn parses_header_fields() {
        // 0b0010_1010：forbidden=0, type=5(Metadata), ext=0, has_size=1, reserved=0
        let mut r = BitReader::new(&[0b0010_1010]);
        let h = parse_header(&mut r).unwrap();
        assert_eq!(h.obu_type, ObuType::Metadata);
        assert!(h.has_size_field);
        assert_eq!(h.temporal_id, 0, "没有扩展头时时间层是 0");
        assert_eq!(r.bits_read(), 8, "一个头字节");
    }

    #[test]
    fn parses_extension_header() {
        // 0b0001_1110：forbidden=0, type=3(FrameHeader), ext=1, has_size=1, reserved=0
        // 扩展字节：temporal_id(3)=0b010, spatial_id(2)=0b11, 保留 3 位=0 → 0b0101_1000
        let mut r = BitReader::new(&[0b0001_1110, 0b0101_1000]);
        let h = parse_header(&mut r).unwrap();
        assert_eq!(h.obu_type, ObuType::FrameHeader);
        assert_eq!(h.temporal_id, 0b010);
        assert_eq!(h.spatial_id, 0b11);
        assert_eq!(r.bits_read(), 16, "头 + 扩展头");
    }

    #[test]
    fn rejects_bad_header_bits() {
        let mut r = BitReader::new(&[0b1000_0000]);
        assert!(parse_header(&mut r).is_err(), "forbidden 位必须是 0");
        let mut r = BitReader::new(&[0b0010_1011]);
        assert!(parse_header(&mut r).is_err(), "保留位必须是 0");
        // 扩展头的保留 3 位里有一位是 1
        let mut r = BitReader::new(&[0b0001_1110, 0b0000_0001]);
        assert!(parse_header(&mut r).is_err(), "扩展头保留位必须是 0");
    }

    #[test]
    fn walks_a_typical_sample() {
        // 真实采样的典型形态：时间分隔符 → 序列头 → 帧
        let mut sample = obu_with_size(2, &[]);
        sample.extend(obu_with_size(1, &[0xAA, 0xBB]));
        sample.extend(obu_with_size(6, &[0x01, 0x02, 0x03]));
        let mut it = ObuIter::new(&sample);
        let mut obus = Vec::new();
        while let Some(o) = it.next_obu().unwrap() {
            obus.push(o);
        }
        assert_eq!(obus.len(), 3);
        assert_eq!(obus[0].header.obu_type, ObuType::TemporalDelimiter);
        assert!(obus[0].payload.is_empty());
        assert_eq!(obus[1].header.obu_type, ObuType::SequenceHeader);
        assert_eq!(obus[1].payload, &[0xAA, 0xBB]);
        assert_eq!(obus[2].header.obu_type, ObuType::Frame);
        assert_eq!(obus[2].payload, &[0x01, 0x02, 0x03]);
        assert!(obus[2].header.obu_type.carries_bitstream());
        assert!(!obus[1].header.obu_type.carries_bitstream());
        assert_eq!(count_obus(&sample).unwrap(), 3);
    }

    #[test]
    fn last_obu_may_omit_size_field() {
        let mut sample = obu_with_size(2, &[]);
        sample.push(0b0001_1000); // type=3, ext=0, has_size=0
        sample.extend_from_slice(&[9, 8, 7]);
        let mut it = ObuIter::new(&sample);
        assert_eq!(
            it.next_obu().unwrap().unwrap().header.obu_type,
            ObuType::TemporalDelimiter
        );
        let last = it.next_obu().unwrap().unwrap();
        assert_eq!(last.header.obu_type, ObuType::FrameHeader);
        assert!(!last.header.has_size_field);
        assert_eq!(last.payload, &[9, 8, 7], "长度由缓冲边界给出");
        assert!(it.next_obu().unwrap().is_none());
    }

    #[test]
    fn truncated_size_field_is_an_error() {
        // 长度字段说要 9 字节，但缓冲里只剩 2 字节：必须报错，
        // 不能把短的载荷当成完整载荷（那会让后面所有解析整体错位）
        let sample = vec![0b0001_1010, 9, 1, 2];
        let mut it = ObuIter::new(&sample);
        assert!(it.next_obu().is_err());
    }

    #[test]
    fn finds_first_sequence_header() {
        let mut sample = obu_with_size(2, &[]);
        sample.extend(obu_with_size(6, &[0xFF]));
        sample.extend(obu_with_size(1, &[0x81, 0x00]));
        assert_eq!(first_sequence_header(&sample), Some(&[0x81u8, 0x00][..]));
        assert_eq!(first_sequence_header(&obu_with_size(2, &[])), None);
    }
}
