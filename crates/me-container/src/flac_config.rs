//! FLAC 的**配置数据**在两种容器里是两种形态，本模块是唯一的转换处：
//!
//! | 容器 | 形态 | 开头 |
//! |---|---|---|
//! | ISOBMFF（`.mp4`） | `dfLa` 载荷（ISO/IEC 14496-3） | 4 字节 `version+flags`，然后是元数据块序列 |
//! | Matroska / 裸流（`.mkv` / `.flac`） | `fLaC` 头（FLAC 规范） | 魔数 `fLaC` + 元数据块序列 |
//!
//! 两者的**元数据块字节完全相同**（块头 4 字节 + 载荷），差别只有"前面有没有魔数、
//! 有没有 `version+flags`"。所以转换就是加/去这 4 个字节的包装。
//!
//! ## 为什么全引擎统一用 `dfLa` 载荷形态
//!
//! 与 Opus 同一条纪律（见 [`crate::opus_config`]）：FLAC 的 `extradata` **只**是
//! 解码器初始化数据，采样数据（一个 FLAC 帧）在两种容器里是同一串字节。所以让
//! Matroska 读入时就地规范化成 `dfLa` 载荷，解码器于是只有一条路径。
//!
//! ## 两个必须做对的点
//!
//! 1. **`dfLa` 是 FullBox**：它的载荷以 `version+flags`（4 字节）开头，而 `fLaC`
//!    头没有。写 `dfLa` 盒时是"把 extradata 原样灌进盒子"，所以 extradata 必须
//!    **已经包含**那 4 字节 —— 多写一层、少写一层都会让整条轨解不出来。
//! 2. **"最后一块"标志要如实落到最后一个块上**：FLAC 的解析是"一直读块直到
//!    last-flag"，标志没置位就会把后面的音频帧当成元数据块头继续解析，
//!    症状是"文件能打开但时长/采样率为 0"。本模块在输出时统一把标志重新算一遍，
//!    不信任输入里的那一位。

use me_core::{MediaError, Result};

/// FLAC 的流魔数。
pub const FLAC_MAGIC: [u8; 4] = *b"fLaC";

/// `STREAMINFO` 块的固定长度（FLAC 规范）。
pub const STREAMINFO_LEN: usize = 34;

/// 元数据块类型：`STREAMINFO`。
const BLOCK_STREAMINFO: u8 = 0;

/// 一个元数据块（`last` 是规范里的"最后一块"标志）。
#[derive(Debug, Clone, PartialEq, Eq)]
struct Block {
    kind: u8,
    last: bool,
    payload: Vec<u8>,
}

/// 解析从 `start` 开始的一串元数据块。
///
/// 两种情况都算正常结束：**遇到 last-flag**，或**正好读到缓冲区末尾**
/// （`dfLa` 载荷与 Matroska 的 `CodecPrivate` 都只装头部，本来就没有音频帧；
/// 有的封装器不给 last-flag —— 那种输入不该被拒）。
fn parse_blocks(data: &[u8], start: usize, what: &str) -> Result<Vec<Block>> {
    let mut out = Vec::new();
    let mut off = start;
    while off < data.len() {
        if off + 4 > data.len() {
            return Err(MediaError::Container(format!(
                "{what} 的 FLAC 元数据块在 {off} 处被截断（块头需要 4 字节，只剩 {}）",
                data.len().saturating_sub(off)
            )));
        }
        let head = data[off];
        let last = head & 0x80 != 0;
        let kind = head & 0x7F;
        let len = ((data[off + 1] as usize) << 16)
            | ((data[off + 2] as usize) << 8)
            | (data[off + 3] as usize);
        let body_off = off + 4;
        if body_off + len > data.len() {
            return Err(MediaError::Container(format!(
                "{what} 的 FLAC 元数据块（类型 {kind}）声明 {len} 字节，但只剩 {}",
                data.len().saturating_sub(body_off)
            )));
        }
        out.push(Block { kind, last, payload: data[body_off..body_off + len].to_vec() });
        off = body_off + len;
        if last {
            break;
        }
    }
    Ok(out)
}

/// 把块序列拼回去，并**重新计算**"最后一块"标志（见模块抬头第 2 条）。
fn emit_blocks(blocks: &[Block]) -> Vec<u8> {
    let mut out = Vec::with_capacity(blocks.iter().map(|b| b.payload.len() + 4).sum());
    for (i, b) in blocks.iter().enumerate() {
        let last = i + 1 == blocks.len();
        out.push((if last { 0x80 } else { 0x00 }) | (b.kind & 0x7F));
        let len = b.payload.len() as u32;
        out.extend_from_slice(&len.to_be_bytes()[1..]);
        out.extend_from_slice(&b.payload);
    }
    out
}

/// 块序列里必须有一个合法的 `STREAMINFO`，否则整条轨没有采样率/声道/位深。
fn ensure_streaminfo(blocks: &[Block], what: &str) -> Result<()> {
    let Some(si) = blocks.iter().find(|b| b.kind == BLOCK_STREAMINFO) else {
        return Err(MediaError::Container(format!(
            "{what} 的 FLAC 配置里没有 STREAMINFO 块 —— 没有它就没有采样率与声道数"
        )));
    };
    if si.payload.len() != STREAMINFO_LEN {
        return Err(MediaError::Container(format!(
            "{what} 的 STREAMINFO 是 {} 字节（规范固定 {STREAMINFO_LEN}）",
            si.payload.len()
        )));
    }
    Ok(())
}

/// `dfLa` 载荷 → `fLaC` 头（`fLaC` 魔数 + 元数据块序列）。
///
/// 用于写 Matroska 的 `CodecPrivate` 与裸 `.flac` 文件头。
pub fn dfla_to_flac_header(dfla: &[u8]) -> Result<Vec<u8>> {
    // 只有 34 字节 STREAMINFO 的"裸块"形态：有些实现会这么给，兜住它
    let blocks = if dfla.len() == STREAMINFO_LEN {
        vec![Block { kind: BLOCK_STREAMINFO, last: true, payload: dfla.to_vec() }]
    } else {
        if dfla.len() < 4 {
            return Err(MediaError::Container(format!(
                "FLAC 的 dfLa 载荷只有 {} 字节（至少要 4 字节 version+flags）",
                dfla.len()
            )));
        }
        parse_blocks(dfla, 4, "dfLa")?
    };
    ensure_streaminfo(&blocks, "dfLa")?;
    let mut out = Vec::with_capacity(4 + dfla.len());
    out.extend_from_slice(&FLAC_MAGIC);
    out.extend_from_slice(&emit_blocks(&blocks));
    Ok(out)
}

/// `fLaC` 头（Matroska `CodecPrivate` / 裸 `.flac` 文件头）→ `dfLa` 载荷。
pub fn flac_header_to_dfla(header: &[u8]) -> Result<Vec<u8>> {
    if header.len() < 4 || header[..4] != FLAC_MAGIC {
        return Err(MediaError::Container(
            "Matroska 里的 FLAC 轨缺少 `fLaC` 魔术开头（CodecPrivate 不是 FLAC 头）".into(),
        ));
    }
    let blocks = parse_blocks(header, 4, "fLaC")?;
    ensure_streaminfo(&blocks, "fLaC")?;
    let mut out = Vec::with_capacity(4 + header.len());
    out.extend_from_slice(&[0, 0, 0, 0]); // dfLa 的 version+flags
    out.extend_from_slice(&emit_blocks(&blocks));
    Ok(out)
}

/// 从 `dfLa` 载荷里取 `STREAMINFO` 的 34 字节载荷（诊断/校验用）。
///
/// 直接扫原字节、不建中间 Vec：返回的是**借用的切片**（调用方常用来对拍）。
pub fn streaminfo_of_dfla(dfla: &[u8]) -> Option<&[u8]> {
    if dfla.len() == STREAMINFO_LEN {
        return Some(dfla);
    }
    let mut off = 4;
    loop {
        if off + 4 > dfla.len() {
            return None;
        }
        let head = dfla[off];
        let len = ((dfla[off + 1] as usize) << 16)
            | ((dfla[off + 2] as usize) << 8)
            | (dfla[off + 3] as usize);
        let body = off + 4;
        if body + len > dfla.len() {
            return None;
        }
        if head & 0x7F == BLOCK_STREAMINFO {
            return Some(&dfla[body..body + len]);
        }
        if head & 0x80 != 0 {
            return None; // 已经是最后一块了，没有 STREAMINFO
        }
        off = body + len;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 一个 34 字节的 STREAMINFO：48 kHz / 2 声道 / 16 bit。
    fn streaminfo() -> Vec<u8> {
        let mut b = vec![0u8; STREAMINFO_LEN];
        let sample_rate = 48_000u32;
        let channels = 2u32;
        let bits = 16u32;
        let w = (sample_rate << 12) | ((channels - 1) << 9) | ((bits - 1) << 4);
        b[10..14].copy_from_slice(&w.to_be_bytes());
        b
    }

    fn dfla() -> Vec<u8> {
        let mut v = vec![0, 0, 0, 0];
        v.extend_from_slice(&[0x80, 0x00, 0x00, STREAMINFO_LEN as u8]);
        v.extend_from_slice(&streaminfo());
        v
    }

    #[test]
    fn roundtrips_between_the_two_forms() {
        let header = dfla_to_flac_header(&dfla()).expect("转 fLaC");
        assert_eq!(&header[..4], b"fLaC");
        let back = flac_header_to_dfla(&header).expect("转回来");
        assert_eq!(back, dfla(), "两种形态必须无损往返");
    }

    #[test]
    fn last_block_flag_is_recomputed() {
        // 输入里**没有**置 last 标志（有的封装器就是不给），输出必须补上，
        // 否则解析器会把后面的音频帧当成元数据块继续读
        let mut bad = vec![0, 0, 0, 0];
        bad.extend_from_slice(&[0x00, 0x00, 0x00, STREAMINFO_LEN as u8]);
        bad.extend_from_slice(&streaminfo());
        let header = dfla_to_flac_header(&bad).expect("转 fLaC");
        assert_eq!(header[4] & 0x80, 0x80, "输出必须补上 last-flag");
    }

    #[test]
    fn keeps_extra_metadata_blocks_but_marks_the_real_last() {
        let mut v = dfla();
        v[4] &= 0x7F; // 把 STREAMINFO 的 last 抹掉，再追加一个 VORBIS_COMMENT（类型 4）
        v.extend_from_slice(&[0x04, 0x00, 0x00, 0x06, b'a', b'b', b'c', b'd', b'e', b'f']);
        let header = dfla_to_flac_header(&v).expect("转 fLaC");
        // 块头依次是 0x00（STREAMINFO）与 0x84（last + 注释）
        assert_eq!(header[4] & 0x7F, 0x00);
        assert_eq!(header[4] & 0x80, 0x00, "不是最后一块就不该有 last 标志");
        let second = 4 + 4 + STREAMINFO_LEN;
        assert_eq!(header[second], 0x84, "最后一块要带 last 标志");
    }

    #[test]
    fn rejects_missing_magic() {
        let err = flac_header_to_dfla(&[0, 1, 2, 3, 4, 5]).unwrap_err();
        assert!(format!("{err}").contains("fLaC"), "错误要说清是缺魔数：{err}");
    }

    #[test]
    fn rejects_missing_streaminfo() {
        // 只有一个注释块，没有 STREAMINFO
        let header = b"fLaC\x84\x00\x00\x02ab".to_vec();
        let err = flac_header_to_dfla(&header).unwrap_err();
        assert!(format!("{err}").contains("STREAMINFO"), "错误要指向 STREAMINFO：{err}");
    }

    #[test]
    fn accepts_bare_streaminfo_block() {
        // 有的实现只给 34 字节的 STREAMINFO（没有块头），也要能转
        let header = dfla_to_flac_header(&streaminfo()).expect("裸 STREAMINFO 也要认");
        assert_eq!(&header[..4], b"fLaC");
        assert_eq!(header[4] & 0x80, 0x80);
    }
}
