//! ISO BMFF（MP4 / fMP4）解封装。
//!
//! MP4 与 TS 是**两个极端**：TS 什么索引都没有，全靠顺着字节扫；
//! MP4 则把索引明明白白写在 `moov` 里（`stbl` 的几张表）。
//! 所以这一层的难点不在"猜"，而在**盒子里套盒子**的结构与几十种方言：
//!
//! | 形态 | 索引在哪 | 典型来源 |
//! |---|---|---|
//! | 渐进式 MP4 | `moov/trak/mdia/minf/stbl`（采样表） | 本地文件、录屏、下载完成的片子 |
//! | 分段 fMP4 | `moov`（只有轨道信息，采样表是空的）+ 每个分片的 `moof/traf/trun` | HLS 的 `.m4s`、DASH |
//!
//! ## 三条刻意的设计决定
//!
//! 1. **探测只读 `moov`，绝不碰 `mdat`**。一个 4 GB 的片子，探测耗时与一个
//!    1 MB 的片子一样。这是"边下边播"能起来的前提：BT 场景下第一个片里通常
//!    就有 `ftyp + moov`，读多了就会卡在"等远端把整段送过来"。
//! 2. **视频采样一律转成 Annex-B 再交出去**。fMP4 里视频采样是"长度前缀"布局
//!    （`avcC` 里的 `lengthSizeMinusOne` 说了每个 NAL 前有几个字节的长度），
//!    而全引擎对 [`Packet::data`] 的约定是 **Annex-B（含起始码）**。
//!    转换放在这里而不是 muxer，是因为只有解析器知道 `lengthSizeMinusOne` 是多少
//!    —— 把它当"隐式约定"往上传，muxer 只能靠猜。
//! 3. **`extradata` 如实报告来源形态**：MP4 出来的是 `avcC` / `hvcC` / `esds`
//!    的**载荷原字节**，不做任何转换（TS 出来的是 Annex-B 参数集）。
//!    转换是 muxer 的事，见 `me-mux` 的 `annexb_to_avcc` / `avcc_to_annexb`。
//!
//! ## 宽容度
//!
//! 真实文件很脏：`moov` 在文件末尾（没做 faststart）、盒子长度比文件还大、
//! `stsz` 里声明 40 亿条采样、`stsd` 里的 `AudioSampleEntry` 是 QuickTime 的
//! v1 变体（比标准多 16 字节）。这里的策略统一是：
//! **能在结构上自洽就继续解析，结构上不可能成立就报错**，绝不因为一处畸形
//! 就 panic —— 解析器面对的是全网最脏的字节。

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, OnceLock};

use me_core::{
    BitReader, ByteReader, CodecId, MediaError, MediaInfo, MediaKind, Packet, ReadAt, Result,
    Timebase, Track,
};

use crate::codec;
use crate::demux::Demuxer;
use crate::probe::ContainerFormat;

/// 单个盒子载荷的解析上限（`moov` 正常几十 KB，`mdat` 我们不读）。
const MAX_BOX_PAYLOAD: u64 = 32 * 1024 * 1024;
/// 一个父盒子里最多认多少个子盒子（防"盒子套娃"把内存吃光）。
const MAX_CHILDREN: usize = 4096;
/// 一条轨道最多多少采样。超过这个数说明 `stsz` 是坏的（20M ≈ 60fps 跑 92 小时）。
const MAX_SAMPLES: u32 = 20_000_000;
/// 扫顶层盒子（找最后一个 `moof` 时）的上限。
const MAX_TOP_LEVEL_SCAN: u64 = 200_000;

/// 采样条目里认得出的子盒子（`avcC` 这类）。用来在"方言偏移"下把它捞出来。
const KNOWN_SAMPLE_ENTRY_BOXES: [&[u8; 4]; 12] = [
    b"avcC", b"hvcC", b"av1C", b"vpcC", b"esds", b"dOps", b"dac3", b"dec3", b"dfLa", b"alac",
    b"btrt", b"pasp",
];

// ---------------------------------------------------------------------------
// 盒子遍历
// ---------------------------------------------------------------------------

/// 一个盒子在文件里的位置。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BoxHeader {
    kind: [u8; 4],
    /// 盒子起点（含 8/16 字节头）。
    start: u64,
    /// 载荷起点。
    payload_start: u64,
    /// 盒子终点（不含）。
    end: u64,
}

impl BoxHeader {
    fn payload_len(&self) -> u64 {
        self.end.saturating_sub(self.payload_start)
    }

    fn name(&self) -> String {
        String::from_utf8_lossy(&self.kind).into_owned()
    }
}

/// 读一个盒子头。
///
/// 返回 `None` 的三种情况都**不是错误**：偏移已经到 `limit`、尾部不足 8 字节
/// （文件被截断）、长度字段非法（< 8 或 largesize 不合法）。
/// 把"这里没有盒子了"与"这里有个坏盒子"都归成"停"，
/// 是因为上层唯一的反应都是"停止遍历"—— 而文件末尾少几个字节是常态
/// （录制被打断、BT 最后一个片还没下完）。
fn read_header(src: &dyn ReadAt, offset: u64, limit: u64) -> Result<Option<BoxHeader>> {
    if offset.saturating_add(8) > limit {
        return Ok(None);
    }
    let mut hdr = [0u8; 8];
    if src.read_at(offset, &mut hdr)? < 8 {
        return Ok(None);
    }
    let size32 = u32::from_be_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
    let kind = [hdr[4], hdr[5], hdr[6], hdr[7]];
    let (payload_start, end) = match size32 {
        // size = 0：这个盒子一直到文件末尾（`mdat` 常见）
        0 => (offset + 8, limit),
        // size = 1：真正的长度在后面 8 字节里
        1 => {
            if offset.saturating_add(16) > limit {
                return Ok(None);
            }
            let mut big = [0u8; 8];
            if src.read_at(offset + 8, &mut big)? < 8 {
                return Ok(None);
            }
            let size = u64::from_be_bytes(big);
            if size < 16 {
                return Ok(None);
            }
            // 声明长度超出可用范围时**按可用长度算**：截断的文件不该让整条流解析失败，
            // 该失败的地方（真读不到采样数据）自然会在读采样时报 NeedMore
            (offset + 16, offset.saturating_add(size).min(limit))
        }
        n if n < 8 => return Ok(None),
        n => (offset + 8, offset.saturating_add(n as u64).min(limit)),
    };
    Ok(Some(BoxHeader { kind, start: offset, payload_start, end }))
}

/// 读盒子的全部载荷。
fn read_payload(src: &dyn ReadAt, b: &BoxHeader) -> Result<Vec<u8>> {
    if b.payload_len() > MAX_BOX_PAYLOAD {
        return Err(MediaError::container(format!(
            "盒子 `{}` 的载荷有 {} 字节，超出解析上限（多半是长度字段坏了）",
            b.name(),
            b.payload_len()
        )));
    }
    src.read_exact_at(b.payload_start, b.payload_len() as usize)
}

/// 列出 `[start, end)` 范围内的直接子盒子。
///
/// 循环不变量：`pos` 永远是"下一个盒子的起点"。`pos = b.end` 与 push 必须成对出现，
/// 少一处就是死循环（而且是在解析脏文件时才触发，最难查）。
fn children(src: &dyn ReadAt, start: u64, end: u64) -> Result<Vec<BoxHeader>> {
    let mut out = Vec::new();
    let mut pos = start;
    while pos.saturating_add(8) <= end {
        let Some(b) = read_header(src, pos, end)? else { break };
        if b.end <= b.start {
            break;
        }
        pos = b.end;
        out.push(b);
        if out.len() >= MAX_CHILDREN {
            break;
        }
    }
    Ok(out)
}

/// 找直接子盒子。
fn child_of(src: &dyn ReadAt, parent: &BoxHeader, kind: &[u8; 4]) -> Result<Option<BoxHeader>> {
    Ok(children(src, parent.payload_start, parent.end)?
        .into_iter()
        .find(|b| &b.kind == kind))
}

/// 从 `from` 开始找一个顶层盒子（`from` 必须是盒子边界）。
fn next_top_level(src: &dyn ReadAt, from: u64) -> Result<Option<BoxHeader>> {
    let limit = src.len().unwrap_or(u64::MAX);
    if from.saturating_add(8) > limit {
        return Ok(None);
    }
    match read_header(src, from, limit)? {
        Some(b) if b.end > b.start => Ok(Some(b)),
        _ => Ok(None),
    }
}

/// 从 `from` 开始往后找第一个指定类型的顶层盒子。
fn find_top_level(src: &dyn ReadAt, kind: &[u8; 4], from: u64) -> Result<Option<BoxHeader>> {
    let mut pos = from;
    let mut scanned = 0u64;
    while scanned < MAX_TOP_LEVEL_SCAN {
        let Some(b) = next_top_level(src, pos)? else { return Ok(None) };
        if &b.kind == kind {
            return Ok(Some(b));
        }
        pos = b.end;
        scanned += 1;
    }
    Ok(None)
}

/// 在**已经在内存里**的缓冲里扫子盒子，返回 `(类型, 载荷起, 载荷止)`（下标相对该缓冲）。
///
/// 采样条目（`stsd` 里的那一条）的子盒子直接在同一个缓冲里，没必要再走一次
/// `read_at`：BT 场景下每一次 `read_at` 都可能是一次"等远端送数据"。
fn scan_sub_boxes(buf: &[u8], from: usize, to: usize) -> Vec<([u8; 4], usize, usize)> {
    let mut out = Vec::new();
    let mut pos = from;
    let end = to.min(buf.len());
    while pos + 8 <= end {
        let size = u32::from_be_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]) as usize;
        let kind = [buf[pos + 4], buf[pos + 5], buf[pos + 6], buf[pos + 7]];
        let (hdr, total) = if size == 1 {
            if pos + 16 > end {
                break;
            }
            let big = u64::from_be_bytes([
                buf[pos + 8],
                buf[pos + 9],
                buf[pos + 10],
                buf[pos + 11],
                buf[pos + 12],
                buf[pos + 13],
                buf[pos + 14],
                buf[pos + 15],
            ]) as usize;
            (16usize, big)
        } else if size == 0 {
            (8usize, end - pos)
        } else if size < 8 {
            break;
        } else {
            (8usize, size)
        };
        if total < hdr || pos + total > end {
            break;
        }
        out.push((kind, pos + hdr, pos + total));
        pos += total;
    }
    out
}

/// 在采样条目里捞一个已知子盒子（`avcC` / `esds` …）。
///
/// 为什么要"捞"而不是"按固定偏移读"：`AudioSampleEntry` 有 v0/v1/v2 三种
/// 布局（QuickTime 的 v1 比标准多 16 字节、v2 多 36 字节），而现实里
/// 标着 v0 却按 v1 排布的文件并不罕见。按固定偏移读会在这些文件上
/// 读出"空 extradata"，症状是**解码器起不来、画面全黑**，且极难排查。
/// 按"已知类型 + 长度自洽"去捞，能让这些文件正常播。
fn hunt_sub_box<'a>(
    buf: &'a [u8],
    entry_body_start: usize,
    entry_body_end: usize,
    want: &[u8; 4],
) -> Option<&'a [u8]> {
    for offset in [entry_body_start, entry_body_start + 16, entry_body_start + 36] {
        if offset >= entry_body_end {
            continue;
        }
        for (kind, s, e) in scan_sub_boxes(buf, offset, entry_body_end) {
            if &kind == want {
                return buf.get(s..e);
            }
        }
    }
    // 兜底：整段里找第一个"类型命中且长度自洽"的
    for (kind, s, e) in scan_sub_boxes(buf, entry_body_start, entry_body_end) {
        if KNOWN_SAMPLE_ENTRY_BOXES.contains(&&kind) {
            let _ = want;
            return buf.get(s..e);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// 采样表
// ---------------------------------------------------------------------------

/// 渐进式 MP4 的采样表（`stbl` 里那几张表的原样收集）。
#[derive(Debug, Default, Clone)]
struct SampleTable {
    /// `stts`：(采样条数, 每条的时长)
    stts: Vec<(u32, u32)>,
    /// `ctts`：(采样条数, 合成偏移)；空 = 没有 B 帧重排
    ctts: Vec<(u32, i32)>,
    /// `stsc`：(首个 chunk 号，1 基；该 chunk 里多少采样；采样描述下标)
    stsc: Vec<(u32, u32, u32)>,
    /// `stco` / `co64`：每个 chunk 的绝对文件偏移
    chunk_offsets: Vec<u64>,
    /// `stsz` / `stz2` 展开后的每条采样大小
    sizes: Vec<u32>,
    /// `stss`：同步采样号（1 基，已排序）。`None` = 全部都是同步点（规范默认）
    stss: Option<Vec<u32>>,
    sample_count: u32,
}

/// 采样游标：把"第 N 条采样在哪、时间是多少"这件事**增量**算出来。
///
/// 为什么不用"一次性展开成 Vec<Sample>"：一部两小时的片子有二十多万条采样，
/// 展开就是几 MB 常驻内存，而播放时其实是**顺序**消费的。增量游标的内存是 O(1)，
/// 且天然支持 seek（从头空转一遍，纯 CPU 无 IO）。
#[derive(Debug, Clone)]
struct SampleCursor {
    /// 下一条采样号（1 基）
    next: u32,
    /// 当前 chunk 下标（0 基）
    chunk: usize,
    /// 当前 chunk 内已经取走几条
    in_chunk: u32,
    /// 下一条采样在文件里的偏移
    offset: u64,
    /// 下一条采样的解码时间（轨道时基）
    dts: i64,
    stts_i: usize,
    stts_left: u32,
    ctts_i: usize,
    ctts_left: u32,
}

/// 游标吐出的一条采样：`(偏移, 字节数, dts, 时长, 合成偏移, 是否关键帧)`。
type SampleTuple = (u64, u32, i64, u32, i32, bool);

impl SampleCursor {
    fn new(t: &SampleTable) -> Self {
        Self {
            next: 1,
            chunk: 0,
            in_chunk: 0,
            offset: t.chunk_offsets.first().copied().unwrap_or(0),
            dts: 0,
            stts_i: 0,
            stts_left: 0,
            ctts_i: 0,
            ctts_left: 0,
        }
    }

    /// `stsc` 是游程表：给出第 `chunk0` 个 chunk 里有几条采样。
    fn samples_in_chunk(t: &SampleTable, chunk0: usize) -> u32 {
        let chunk_no = chunk0 as u32 + 1;
        let mut spc = 0u32;
        for &(first, n, _) in &t.stsc {
            if first <= chunk_no {
                spc = n;
            } else {
                break;
            }
        }
        spc
    }

    fn next_stts(t: &SampleTable, i: &mut usize, left: &mut u32) -> u32 {
        if *left == 0 {
            while *i < t.stts.len() {
                if t.stts[*i].0 > 0 {
                    *left = t.stts[*i].0;
                    break;
                }
                *i += 1;
            }
        }
        if *left == 0 {
            return 0; // 表用完了（采样数比 stts 声明的多）：如实返回 0，不猜
        }
        *left -= 1;
        let v = t.stts[*i].1;
        if *left == 0 {
            // **这一项取完了就要挪下标**：不挪的话下一次会重新读到同一项，
            // 症状是"整条流的时长都等于第一帧的时长"，而且极难发现
            *i += 1;
        }
        v
    }

    fn next_ctts(t: &SampleTable, i: &mut usize, left: &mut u32) -> i32 {
        if *left == 0 {
            while *i < t.ctts.len() {
                if t.ctts[*i].0 > 0 {
                    *left = t.ctts[*i].0;
                    break;
                }
                *i += 1;
            }
        }
        if *left == 0 {
            return 0;
        }
        *left -= 1;
        let v = t.ctts[*i].1;
        if *left == 0 {
            *i += 1;
        }
        v
    }

    /// 取一条采样并前进。
    fn take(&mut self, t: &SampleTable) -> Option<SampleTuple> {
        if self.next > t.sample_count {
            return None;
        }
        let idx = (self.next - 1) as usize;
        let size = t.sizes.get(idx).copied().unwrap_or(0);
        let delta = Self::next_stts(t, &mut self.stts_i, &mut self.stts_left);
        let cts = if t.ctts.is_empty() {
            0
        } else {
            Self::next_ctts(t, &mut self.ctts_i, &mut self.ctts_left)
        };
        let key = match &t.stss {
            Some(list) => list.binary_search(&self.next).is_ok(),
            None => true,
        };
        let out = (self.offset, size, self.dts, delta, cts, key);

        self.offset = self.offset.saturating_add(size as u64);
        self.dts += delta as i64;
        self.next += 1;
        self.in_chunk += 1;
        let spc = Self::samples_in_chunk(t, self.chunk);
        if spc == 0 || self.in_chunk >= spc {
            self.chunk += 1;
            self.in_chunk = 0;
            if let Some(&o) = t.chunk_offsets.get(self.chunk) {
                self.offset = o;
            }
        }
        Some(out)
    }
}

/// `mvex/trex`：分片模式下的默认值（`trun` 里没写的字段从这里取）。
#[derive(Debug, Clone, Copy)]
struct Trex {
    default_sample_duration: u32,
    default_sample_size: u32,
    default_sample_flags: u32,
}

/// 一条轨道的解析结果。
#[derive(Debug, Clone)]
struct Mp4Track {
    /// 文件里的 `track_ID`（沿用不重新编号：`tfhd` 就是靠它找轨道的）
    id: u32,
    kind: MediaKind,
    codec: CodecId,
    /// 媒体时基（`mdhd` 的 timescale）
    timescale: u32,
    /// `mdhd` 里的时长（媒体时基）；`None` = 未知
    duration_ticks: Option<i64>,
    /// 解码器初始化数据：MP4 出来的是 `avcC`/`hvcC`/`esds` 的载荷原字节
    extradata: Vec<u8>,
    width: u32,
    height: u32,
    sample_rate: u32,
    channels: u16,
    language: Option<String>,
    /// 视频采样的"长度前缀"字节数（`avcC`/`hvcC` 给的）；0 = 不做 Annex-B 转换
    nal_length_size: u8,
    /// 编辑列表带来的时间平移（轨道时基刻度，直接加在 pts/dts 上）
    edit_shift: i64,
    /// 渐进式的采样表
    stbl: Option<SampleTable>,
    /// 分片模式的默认值
    trex: Option<Trex>,
}

impl Mp4Track {
    fn timebase(&self) -> Timebase {
        Timebase::new(self.timescale.max(1))
    }
}

// ---------------------------------------------------------------------------
// moov 解析
// ---------------------------------------------------------------------------

/// `moov` 的解析结果。
struct Moov {
    movie_timescale: u32,
    duration_ticks: Option<i64>,
    tracks: Vec<Mp4Track>,
    fragmented: bool,
}

fn parse_mvhd(p: &[u8]) -> Result<(u32, Option<i64>)> {
    let mut r = ByteReader::new(p);
    let version = r.u8()?;
    r.skip(3)?;
    if version == 1 {
        r.skip(16)?;
    } else {
        r.skip(8)?;
    }
    let timescale = r.u32_be()?;
    let duration = if version == 1 { r.u64_be()? } else { r.u32_be()? as u64 };
    // 0 与全 1 都表示"未知"（fMP4 的 mvhd 时长就是 0，分片自己带时间）
    let dur = if duration == 0 || duration == u32::MAX as u64 || duration == u64::MAX {
        None
    } else {
        Some(duration as i64)
    };
    Ok((timescale, dur))
}

fn parse_tkhd(p: &[u8]) -> Option<(u32, u32, u32)> {
    let mut r = ByteReader::new(p);
    let version = r.u8().ok()?;
    r.skip(3).ok()?;
    if version == 1 {
        r.skip(16).ok()?;
    } else {
        r.skip(8).ok()?;
    }
    let id = r.u32_be().ok()?;
    r.skip(4).ok()?;
    if version == 1 {
        r.skip(8).ok()?;
    } else {
        r.skip(4).ok()?;
    }
    r.skip(8 + 2 + 2 + 2 + 2 + 36).ok()?; // reserved + layer + group + volume + reserved + matrix
    let w = r.u32_be().ok()? >> 16; // 16.16 定点
    let h = r.u32_be().ok()? >> 16;
    Some((id, w, h))
}

fn parse_mdhd(p: &[u8]) -> Result<(u32, Option<i64>, Option<String>)> {
    let mut r = ByteReader::new(p);
    let version = r.u8()?;
    r.skip(3)?;
    if version == 1 {
        r.skip(16)?;
    } else {
        r.skip(8)?;
    }
    let timescale = r.u32_be()?;
    let duration = if version == 1 { r.u64_be()? } else { r.u32_be()? as u64 };
    let dur = if duration == 0 || duration == u32::MAX as u64 || duration == u64::MAX {
        None
    } else {
        Some(duration as i64)
    };
    // 语言：3 个 5 位字母，各 +0x60；全 0 与 "und" 都算"没说"
    let lang = r.u16_be().ok().and_then(|v| {
        if v == 0 {
            return None;
        }
        let c = |shift: u16| ((v >> shift) & 0x1F) as u8;
        let s: String = [c(10), c(5), c(0)]
            .iter()
            .map(|&x| if x == 0 { ' ' } else { (x + 0x60) as char })
            .collect();
        let s = s.trim().to_ascii_lowercase();
        if s.len() == 3 && s != "und" && s.chars().all(|c| c.is_ascii_alphabetic()) {
            Some(s)
        } else {
            None
        }
    });
    Ok((timescale, dur, lang))
}

fn parse_hdlr(p: &[u8]) -> Option<[u8; 4]> {
    let mut r = ByteReader::new(p);
    r.u8().ok()?; // version
    r.skip(3).ok()?; // flags
    r.skip(4).ok()?; // pre_defined
    let h = r.read(4).ok()?;
    Some([h[0], h[1], h[2], h[3]])
}

/// `elst`：编辑列表 → **时间平移刻度**（按媒体时基）。
///
/// 编辑列表是 MP4 里最容易被忽略、又最容易造成"音画错开"的东西：
///
/// - 音频为了填满解码器的预热采样，开头常有一段 `media_time = -1` 的**空编辑**，
///   它的时长表示"这段时间什么都不播"。忽略它，音频会比视频**早**开始若干毫秒。
/// - `media_time > 0` 表示"从媒体的这个时间点开始播"。
///
/// 这里**只做平移、不丢采样**：`media_time` 之前的那几条采样可能包含**关键帧**
/// （实测一个真实 MP4 的编辑列表裁掉的两帧里就有唯一的关键帧），丢掉它们会让
/// 解码器从非关键帧开始解 —— 花屏或者干脆不出画。代价是这些采样的时间戳会变成
/// 负数，由调用方按容器规则处理（`me-mux` 会把负值夹到 0）。
fn parse_elst(p: &[u8], movie_timescale: u32, media_timescale: u32) -> Option<i64> {
    let mut r = ByteReader::new(p);
    let version = r.u8().ok()?;
    r.skip(3).ok()?;
    let count = r.u32_be().ok()?;
    if movie_timescale == 0 || media_timescale == 0 {
        return None;
    }
    let mut empty_ticks = 0i64; // 空编辑累计（换算到媒体时基）
    let mut media_time0: Option<i64> = None;
    for _ in 0..count.min(64) {
        let seg_dur = if version == 1 { r.u64_be().ok()? } else { r.u32_be().ok()? as u64 };
        let media_time = if version == 1 {
            r.u64_be().ok()? as i64
        } else {
            r.u32_be().ok()? as i32 as i64
        };
        r.skip(4).ok()?; // media_rate_integer + media_rate_fraction
        if media_time < 0 {
            // 空编辑：时长按**电影时基**给，要换算到媒体时基
            let in_media = (seg_dur as i128) * (media_timescale as i128) / (movie_timescale as i128);
            empty_ticks += in_media as i64;
        } else if media_time0.is_none() {
            media_time0 = Some(media_time);
        }
    }
    let start = media_time0.unwrap_or(0);
    Some(empty_ticks - start)
}

/// 解析 `stsd` 的第一条采样描述，得到编解码器与初始化数据。
struct SampleEntry {
    codec: CodecId,
    extradata: Vec<u8>,
    nal_length_size: u8,
    width: u32,
    height: u32,
    sample_rate: u32,
    channels: u16,
}

fn parse_stsd(p: &[u8]) -> Option<SampleEntry> {
    if p.len() < 16 {
        return None;
    }
    let entry_count = u32::from_be_bytes([p[4], p[5], p[6], p[7]]);
    if entry_count == 0 {
        return None;
    }
    let start = 8usize;
    let size = u32::from_be_bytes([p[start], p[start + 1], p[start + 2], p[start + 3]]) as usize;
    if size < 16 {
        return None;
    }
    let format = [p[start + 4], p[start + 5], p[start + 6], p[start + 7]];
    let entry_end = (start + size).min(p.len());
    let body = start + 16; // 跳过 (size, format, reserved(6), data_reference_index(2))
    if body > entry_end {
        return None;
    }

    let mut e = SampleEntry {
        codec: CodecId::Unknown(u32::from_be_bytes(format)),
        extradata: Vec::new(),
        nal_length_size: 0,
        width: 0,
        height: 0,
        sample_rate: 0,
        channels: 0,
    };

    match &format {
        b"avc1" | b"avc3" | b"hvc1" | b"hev1" | b"av01" | b"vp09" | b"vp08" | b"mp4v" => {
            // VisualSampleEntry：70 字节固定字段（在 16 字节条目头之后）
            if body + 70 <= entry_end {
                e.width = u16::from_be_bytes([p[body + 16], p[body + 17]]) as u32;
                e.height = u16::from_be_bytes([p[body + 18], p[body + 19]]) as u32;
            }
            let sub_start = (body + 70).min(entry_end);
            match &format {
                b"avc1" | b"avc3" => {
                    e.codec = CodecId::H264;
                    if let Some(avcc) = hunt_sub_box(p, sub_start, entry_end, b"avcC") {
                        e.extradata = avcc.to_vec();
                        // avcC[4] 的低 2 位 = lengthSizeMinusOne
                        e.nal_length_size = avcc.get(4).map(|b| (b & 0x03) + 1).unwrap_or(4);
                    }
                }
                b"hvc1" | b"hev1" => {
                    e.codec = CodecId::H265;
                    if let Some(hvcc) = hunt_sub_box(p, sub_start, entry_end, b"hvcC") {
                        e.extradata = hvcc.to_vec();
                        // hvcC[21] 的低 2 位 = lengthSizeMinusOne
                        e.nal_length_size = hvcc.get(21).map(|b| (b & 0x03) + 1).unwrap_or(4);
                    }
                }
                b"av01" => {
                    e.codec = CodecId::Av1;
                    // AV1 的采样是 OBU（自带 leb128 长度），**没有 Annex-B 形态**，
                    // 所以这里不做转换，原样透传
                    if let Some(c) = hunt_sub_box(p, sub_start, entry_end, b"av1C") {
                        e.extradata = c.to_vec();
                    }
                }
                b"vp09" => {
                    e.codec = CodecId::Vp9;
                    if let Some(c) = hunt_sub_box(p, sub_start, entry_end, b"vpcC") {
                        e.extradata = c.to_vec();
                    }
                }
                b"vp08" => e.codec = CodecId::Vp8,
                _ => e.codec = CodecId::Unknown(u32::from_be_bytes(format)),
            }
        }
        b"mp4a" | b"ac-3" | b"ec-3" | b"Opus" | b"fLaC" | b"alac" | b"mp3 " | b"samr" => {
            // AudioSampleEntry：20 字节固定字段
            if body + 20 <= entry_end {
                e.channels = u16::from_be_bytes([p[body + 8], p[body + 9]]);
                // 采样率是 16.16 定点
                e.sample_rate = u32::from_be_bytes([p[body + 16], p[body + 17], p[body + 18], p[body + 19]]) >> 16;
            }
            let sub_start = (body + 20).min(entry_end);
            match &format {
                b"mp4a" => {
                    if let Some(esds) = hunt_sub_box(p, sub_start, entry_end, b"esds") {
                        if let Some((oti, asc, rate, ch)) = parse_esds(esds) {
                            e.codec = match oti {
                                0x40 | 0x66 | 0x67 | 0x68 => CodecId::Aac,
                                0x69 | 0x6B => CodecId::Mp3,
                                0xA5 => CodecId::Ac3,
                                0xA6 => CodecId::Eac3,
                                _ => CodecId::Aac, // mp4a 下最可能还是 AAC
                            };
                            if let Some(rate) = rate {
                                e.sample_rate = rate;
                            }
                            if let Some(ch) = ch {
                                if ch > 0 {
                                    e.channels = ch;
                                }
                            }
                            e.extradata = asc;
                        } else {
                            e.codec = CodecId::Aac;
                        }
                    } else {
                        e.codec = CodecId::Aac;
                    }
                }
                b"ac-3" => {
                    e.codec = CodecId::Ac3;
                    if let Some(c) = hunt_sub_box(p, sub_start, entry_end, b"dac3") {
                        e.extradata = c.to_vec();
                    }
                }
                b"ec-3" => {
                    e.codec = CodecId::Eac3;
                    if let Some(c) = hunt_sub_box(p, sub_start, entry_end, b"dec3") {
                        e.extradata = c.to_vec();
                    }
                }
                b"Opus" => {
                    e.codec = CodecId::Opus;
                    // `dOps` 是 OpusHead 的**另一种字节形态**（大端、没有魔数），
                    // 在这里就地规范化成引擎统一的 `OpusHead`：采样数据（Opus 包）
                    // 在两种容器里本来就是同一串字节，extradata 只给解码器用，
                    // 没必要让解码器认识两种形态（见 `crate::opus_config`）。
                    //
                    // 盒子里有声道数：它比 `AudioSampleEntry` 的 channelcount 权威
                    // （后者在真实文件里时常与码流不一致）。
                    if let Some(c) = hunt_sub_box(p, sub_start, entry_end, b"dOps") {
                        if let Ok(cfg) = crate::opus_config::OpusConfig::parse_dops(c) {
                            e.channels = cfg.channels as u16;
                            e.extradata = cfg.to_head();
                        }
                    }
                    // 解码采样率恒为 48 kHz（`dOps` 里的 input_sample_rate 是**编码前**
                    // 的采样率，不是解码采样率）；AudioSampleEntry 那个 16.16 字段
                    // 有些封装器写 0，兜底补上 48 kHz。
                    if e.sample_rate == 0 {
                        e.sample_rate = crate::opus_config::OPUS_RATE;
                    }
                }
                b"fLaC" | b"alac" => {
                    e.codec = CodecId::Flac;
                    if let Some(c) = hunt_sub_box(p, sub_start, entry_end, b"dfLa") {
                        e.extradata = c.to_vec();
                        // 真实采样率在 STREAMINFO 里：AudioSampleEntry 那个 16.16 字段的
                        // 整数部分只有 16 位，Hi-Res 的 96k/192k 装不下（很多文件就写 0），
                        // 所以这里以 STREAMINFO 为准
                        if let Some(rate) = flac_sample_rate(c) {
                            e.sample_rate = rate;
                        }
                    }
                }
                _ => {
                    e.codec = CodecId::Mp3;
                }
            }
        }
        _ => {
            e.codec = CodecId::Unknown(u32::from_be_bytes(format));
        }
    }
    Some(e)
}

/// 描述符里的变长长度（1~4 字节，最高位是"还有后续"）。
fn read_desc_len(buf: &[u8], pos: &mut usize) -> Option<usize> {
    let mut len = 0usize;
    for _ in 0..4 {
        let b = *buf.get(*pos)?;
        *pos += 1;
        len = (len << 7) | (b & 0x7F) as usize;
        if b & 0x80 == 0 {
            return Some(len);
        }
    }
    Some(len)
}

/// 在 MPEG-4 描述符树里找 `want`，返回它的载荷区间。
fn walk_desc(buf: &[u8], start: usize, end: usize, want: u8) -> Option<(usize, usize)> {
    let end = end.min(buf.len());
    let mut pos = start;
    while pos < end {
        let tag = buf[pos];
        let mut p = pos + 1;
        let len = read_desc_len(buf, &mut p)?;
        let payload_end = p.checked_add(len)?;
        if payload_end > end {
            return None;
        }
        if tag == want {
            return Some((p, payload_end));
        }
        if tag == 0x03 {
            // ES_Descriptor：ES_ID(2) + flags(1) + 可选字段
            if p + 3 > payload_end {
                return None;
            }
            let flags = buf[p + 2];
            let mut q = p + 3;
            if flags & 0x80 != 0 {
                q += 2;
            }
            if flags & 0x40 != 0 {
                q += 1 + *buf.get(q)? as usize;
            }
            if flags & 0x20 != 0 {
                q += 2;
            }
            if q < payload_end {
                if let Some(r) = walk_desc(buf, q, payload_end, want) {
                    return Some(r);
                }
            }
        } else if tag == 0x04 {
            // DecoderConfigDescriptor：objectType(1)+streamType(1)+bufferSize(3)+max(4)+avg(4)
            let q = p + 13;
            if q < payload_end {
                if let Some(r) = walk_desc(buf, q, payload_end, want) {
                    return Some(r);
                }
            }
        }
        pos = payload_end;
    }
    None
}

/// `esds` → `(objectTypeIndication, AudioSpecificConfig, 采样率, 声道数)`。
type EsdsInfo = (u8, Vec<u8>, Option<u32>, Option<u16>);

fn parse_esds(p: &[u8]) -> Option<EsdsInfo> {
    if p.len() < 4 {
        return None;
    }
    let (dc_start, dc_end) = walk_desc(p, 4, p.len(), 0x04)?;
    let oti = *p.get(dc_start)?;
    let asc = match walk_desc(p, dc_start + 13, dc_end, 0x05) {
        Some((s, e)) => p.get(s..e)?.to_vec(),
        None => Vec::new(),
    };
    let (rate, ch) = parse_asc(&asc).map(|(r, c)| (Some(r), Some(c))).unwrap_or((None, None));
    Some((oti, asc, rate, ch))
}

/// AudioSpecificConfig → `(采样率, 声道数)`。
fn parse_asc(asc: &[u8]) -> Option<(u32, u16)> {
    const RATES: [u32; 13] = [
        96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
    ];
    let mut r = BitReader::new(asc);
    let _object_type = r.read_bits(5).ok()?;
    let freq_idx = r.read_bits(4).ok()?;
    let chan = r.read_bits(4).ok()?;
    let rate = if freq_idx == 15 {
        r.read_bits(24).ok()? as u32
    } else {
        *RATES.get(freq_idx as usize)?
    };
    // chan == 0 表示"声道配置在后面的 GASpecificConfig 里"，这里如实报 0（未知）
    Some((rate, chan as u16))
}

/// 从 `dfLa` 的载荷里读 FLAC 的采样率。
///
/// `dfLa` 载荷 = `version+flags(4)` + `metadata block 头(4: last+type, 长度 24)` +
/// STREAMINFO(34)。采样率是 STREAMINFO 里的 **20 位**字段，位于 STREAMINFO 偏移 10
/// —— 所以取 `extradata[18..21]`。
fn flac_sample_rate(dfla_payload: &[u8]) -> Option<u32> {
    // STREAMINFO 的载荷从第 8 字节开始（4 字节 version/flags + 4 字节 metadata block 头），
    // 采样率在 STREAMINFO 偏移 10 → 也就是载荷的 18..21
    let b = dfla_payload.get(18..21)?;
    let rate = ((b[0] as u32) << 12) | ((b[1] as u32) << 4) | ((b[2] as u32) >> 4);
    if rate == 0 || rate > 1_000_000 {
        return None;
    }
    Some(rate)
}

/// 从 `avcC` / `hvcC` 的参数集里补出分辨率。
///
/// 有些文件的 `tkhd` 宽高是 0（编码器偷懒），但 `avcC`/`hvcC` 里一定带着 SPS ——
/// 不从这里补，探测结果就只能显示"未知分辨率"。
///
/// 为什么不手写"跳过 profile_tier_level 直接读宽高"：H.265 的 PTL 长度随
/// `max_sub_layers_minus1` 变化，还夹着一堆条件字段，手写一遍就是在维护第二份
/// 容易漂移的解析器。这里改成**把 SPS 拼成带 NAL 头的形态交给 `codec`** ——
/// 解析逻辑只有一份。
fn size_from_extradata(codec: &CodecId, extradata: &[u8]) -> Option<(u32, u32)> {
    match codec {
        CodecId::H264 => {
            // avcC: version(1) profile(1) compat(1) level(1) lengthSizeMinusOne(1)
            //       numOfSPS(1) [ len(2) + sps ]...
            let num = *extradata.get(5)? as usize;
            if num == 0 {
                return None;
            }
            let len = u16::from_be_bytes([*extradata.get(6)?, *extradata.get(7)?]) as usize;
            let sps = extradata.get(8..8 + len)?;
            let mut nal = Vec::with_capacity(len + 1);
            nal.push(0x67); // H.264 SPS 的 NAL 头（type = 7）
            nal.extend_from_slice(sps);
            let info = codec::parse_h264_sps(&nal)?;
            if info.width > 0 && info.height > 0 {
                Some((info.width, info.height))
            } else {
                None
            }
        }
        CodecId::H265 => {
            // hvcC: 22 字节固定头 → numOfArrays(1)
            //       每个数组：[ 类型(1) numNalus(2) [ len(2) + nalu ]... ]
            let arrays = *extradata.get(22)? as usize;
            let mut pos = 23usize;
            for _ in 0..arrays.min(8) {
                let nal_type = *extradata.get(pos)? & 0x3F;
                let count = u16::from_be_bytes([*extradata.get(pos + 1)?, *extradata.get(pos + 2)?]) as usize;
                pos += 3;
                for _ in 0..count {
                    let len = u16::from_be_bytes([*extradata.get(pos)?, *extradata.get(pos + 1)?]) as usize;
                    pos += 2;
                    let nalu = extradata.get(pos..pos + len)?;
                    pos += len;
                    // 33 = SPS（H.265 的 NAL 类型在头字节的 6 位里）
                    if nal_type == 33 {
                        let mut nal = Vec::with_capacity(len + 2);
                        nal.extend_from_slice(&[0x42, 0x01]); // type = 33
                        nal.extend_from_slice(nalu);
                        let info = codec::parse_h265_sps(&nal)?;
                        if info.width > 0 && info.height > 0 {
                            return Some((info.width, info.height));
                        }
                    }
                }
            }
            None
        }
        _ => None,
    }
}

fn parse_stts(p: &[u8]) -> Result<Vec<(u32, u32)>> {
    let mut r = ByteReader::new(p);
    r.skip(4)?;
    let count = r.u32_be()?;
    let mut out = Vec::with_capacity(count.min(4096) as usize);
    for _ in 0..count {
        let n = r.u32_be()?;
        let d = r.u32_be()?;
        out.push((n, d));
    }
    Ok(out)
}

fn parse_ctts(p: &[u8]) -> Result<Vec<(u32, i32)>> {
    let mut r = ByteReader::new(p);
    r.skip(4)?;
    let count = r.u32_be()?;
    let mut out = Vec::with_capacity(count.min(4096) as usize);
    for _ in 0..count {
        let n = r.u32_be()?;
        let off = r.u32_be()? as i32;
        out.push((n, off));
    }
    Ok(out)
}

fn parse_stsc(p: &[u8]) -> Result<Vec<(u32, u32, u32)>> {
    let mut r = ByteReader::new(p);
    r.skip(4)?;
    let count = r.u32_be()?;
    let mut out = Vec::with_capacity(count.min(4096) as usize);
    for _ in 0..count {
        out.push((r.u32_be()?, r.u32_be()?, r.u32_be()?));
    }
    Ok(out)
}

fn parse_stco(p: &[u8], wide: bool) -> Result<Vec<u64>> {
    let mut r = ByteReader::new(p);
    r.skip(4)?;
    let count = r.u32_be()?;
    let mut out = Vec::with_capacity(count.min(1 << 20) as usize);
    for _ in 0..count {
        out.push(if wide { r.u64_be()? } else { r.u32_be()? as u64 });
    }
    Ok(out)
}

fn parse_stsz(p: &[u8]) -> Result<Vec<u32>> {
    let mut r = ByteReader::new(p);
    r.skip(4)?;
    let uniform = r.u32_be()?;
    let count = r.u32_be()?;
    if count > MAX_SAMPLES {
        return Err(MediaError::container(format!(
            "stsz 声明了 {count} 条采样，明显是坏文件（上限 {MAX_SAMPLES}）"
        )));
    }
    if uniform != 0 {
        return Ok(vec![uniform; count as usize]);
    }
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        out.push(r.u32_be()?);
    }
    Ok(out)
}

/// `stz2`：紧凑采样大小表（4/8/16 位打包）。
fn parse_stz2(p: &[u8]) -> Result<Vec<u32>> {
    let mut r = ByteReader::new(p);
    r.skip(4)?;
    r.skip(3)?; // reserved
    let field_size = r.u8()?;
    let count = r.u32_be()?;
    if count > MAX_SAMPLES {
        return Err(MediaError::container(format!("stz2 声明了 {count} 条采样，明显是坏文件")));
    }
    let mut out = Vec::with_capacity(count as usize);
    match field_size {
        4 => {
            while out.len() < count as usize {
                let b = r.u8()?;
                out.push((b >> 4) as u32);
                if out.len() < count as usize {
                    out.push((b & 0x0F) as u32);
                }
            }
        }
        8 => {
            for _ in 0..count {
                out.push(r.u8()? as u32);
            }
        }
        16 => {
            for _ in 0..count {
                out.push(r.u16_be()? as u32);
            }
        }
        _ => return Err(MediaError::container(format!("stz2 的字段宽度 {field_size} 非法"))),
    }
    Ok(out)
}

fn parse_stss(p: &[u8]) -> Result<Vec<u32>> {
    let mut r = ByteReader::new(p);
    r.skip(4)?;
    let count = r.u32_be()?;
    let mut out = Vec::with_capacity(count.min(1 << 20) as usize);
    for _ in 0..count {
        out.push(r.u32_be()?);
    }
    // 二分查找要求有序，坏文件里的乱序表会让我们"找不到关键帧"（seek 失效但不崩）
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

fn parse_trex(p: &[u8]) -> Option<(u32, Trex)> {
    let mut r = ByteReader::new(p);
    r.skip(4).ok()?;
    let track_id = r.u32_be().ok()?;
    let _desc_idx = r.u32_be().ok()?;
    let duration = r.u32_be().ok()?;
    let size = r.u32_be().ok()?;
    let flags = r.u32_be().ok()?;
    Some((track_id, Trex { default_sample_duration: duration, default_sample_size: size, default_sample_flags: flags }))
}

fn handler_kind(h: &[u8; 4]) -> MediaKind {
    match h {
        b"vide" => MediaKind::Video,
        b"soun" => MediaKind::Audio,
        b"sbtl" | b"subt" | b"text" | b"clcp" => MediaKind::Subtitle,
        _ => MediaKind::Other,
    }
}

/// 解析一条 `trak`。
fn parse_trak(
    src: &dyn ReadAt,
    trak: &BoxHeader,
    movie_timescale: u32,
    trex: &HashMap<u32, Trex>,
) -> Result<Option<Mp4Track>> {
    let mut id = 0u32;
    let mut width = 0u32;
    let mut height = 0u32;
    if let Some(b) = child_of(src, trak, b"tkhd")? {
        let p = read_payload(src, &b)?;
        if let Some((i, w, h)) = parse_tkhd(&p) {
            id = i;
            width = w;
            height = h;
        }
    }
    if id == 0 {
        // 没有 track_ID 的 trak 无法与 tfhd 对应，跳过比瞎猜安全
        return Ok(None);
    }

    let Some(mdia) = child_of(src, trak, b"mdia")? else {
        return Ok(None);
    };

    let mut timescale = 0u32;
    let mut duration_ticks = None;
    let mut language = None;
    if let Some(b) = child_of(src, &mdia, b"mdhd")? {
        let p = read_payload(src, &b)?;
        if let Ok((ts, dur, lang)) = parse_mdhd(&p) {
            timescale = ts;
            duration_ticks = dur;
            language = lang;
        }
    }
    if timescale == 0 {
        // 时基为 0 会让所有换算失去意义（且会除以 0）。给个安全值并继续：
        // 轨道本身还能读，只是时间戳不准，总比整条流失败强
        timescale = movie_timescale.max(1);
    }

    let kind = match child_of(src, &mdia, b"hdlr")? {
        Some(b) => {
            let p = read_payload(src, &b)?;
            parse_hdlr(&p).map(|h| handler_kind(&h)).unwrap_or(MediaKind::Other)
        }
        None => MediaKind::Other,
    };

    let mut stbl_box = None;
    if let Some(minf) = child_of(src, &mdia, b"minf")? {
        stbl_box = child_of(src, &minf, b"stbl")?;
    }

    let mut stbl = None;
    let mut entry = None;
    if let Some(stbl_b) = stbl_box {
        if let Some(b) = child_of(src, &stbl_b, b"stsd")? {
            let p = read_payload(src, &b)?;
            entry = parse_stsd(&p);
        }
        let mut t = SampleTable::default();
        if let Some(b) = child_of(src, &stbl_b, b"stts")? {
            t.stts = parse_stts(&read_payload(src, &b)?)?;
        }
        if let Some(b) = child_of(src, &stbl_b, b"ctts")? {
            t.ctts = parse_ctts(&read_payload(src, &b)?)?;
        }
        if let Some(b) = child_of(src, &stbl_b, b"stsc")? {
            t.stsc = parse_stsc(&read_payload(src, &b)?)?;
        }
        if let Some(b) = child_of(src, &stbl_b, b"stco")? {
            t.chunk_offsets = parse_stco(&read_payload(src, &b)?, false)?;
        } else if let Some(b) = child_of(src, &stbl_b, b"co64")? {
            t.chunk_offsets = parse_stco(&read_payload(src, &b)?, true)?;
        }
        if let Some(b) = child_of(src, &stbl_b, b"stsz")? {
            t.sizes = parse_stsz(&read_payload(src, &b)?)?;
        } else if let Some(b) = child_of(src, &stbl_b, b"stz2")? {
            t.sizes = parse_stz2(&read_payload(src, &b)?)?;
        }
        if let Some(b) = child_of(src, &stbl_b, b"stss")? {
            t.stss = Some(parse_stss(&read_payload(src, &b)?)?);
        }
        t.sample_count = t.sizes.len() as u32;
        // 采样表是空的（fMP4 的 moov 就是这样）就当作"没有采样表"
        if t.sample_count > 0 && !t.chunk_offsets.is_empty() {
            stbl = Some(t);
        }
    }

    // 编辑列表：**只取时间平移**（不丢采样，见 `parse_elst` 的说明）
    let mut edit_shift = 0i64;
    if let Some(edts) = child_of(src, trak, b"edts")? {
        if let Some(b) = child_of(src, &edts, b"elst")? {
            let p = read_payload(src, &b)?;
            if let Some(shift) = parse_elst(&p, movie_timescale, timescale) {
                edit_shift = shift;
            }
        }
    }

    let mut codec = CodecId::Unknown(0);
    let mut extradata = Vec::new();
    let mut nal_length_size = 0u8;
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    if let Some(e) = entry {
        codec = e.codec;
        extradata = e.extradata;
        nal_length_size = e.nal_length_size;
        sample_rate = e.sample_rate;
        channels = e.channels;
        if e.width > 0 && e.height > 0 {
            width = e.width;
            height = e.height;
        }
    }
    // 宽高还是 0 就从 SPS 里补
    if (width == 0 || height == 0) && matches!(kind, MediaKind::Video) {
        if let Some((w, h)) = size_from_extradata(&codec, &extradata) {
            width = w;
            height = h;
        }
    }

    Ok(Some(Mp4Track {
        id,
        kind,
        codec,
        timescale,
        duration_ticks,
        extradata,
        width,
        height,
        sample_rate,
        channels,
        language,
        nal_length_size,
        edit_shift,
        stbl,
        trex: trex.get(&id).copied(),
    }))
}

fn parse_moov(src: &dyn ReadAt, moov: &BoxHeader) -> Result<Moov> {
    let mut movie_timescale = 1000u32;
    let mut duration_ticks = None;
    if let Some(b) = child_of(src, moov, b"mvhd")? {
        let p = read_payload(src, &b)?;
        let (ts, dur) = parse_mvhd(&p)?;
        if ts > 0 {
            movie_timescale = ts;
        }
        duration_ticks = dur;
    }

    let mut trex: HashMap<u32, Trex> = HashMap::new();
    let mut fragmented = false;
    let mut trak_boxes = Vec::new();
    for b in children(src, moov.payload_start, moov.end)? {
        match &b.kind {
            b"trak" => trak_boxes.push(b),
            b"mvex" => {
                fragmented = true;
                for tb in children(src, b.payload_start, b.end)? {
                    if &tb.kind == b"trex" {
                        let p = read_payload(src, &tb)?;
                        if let Some((id, t)) = parse_trex(&p) {
                            trex.insert(id, t);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let mut tracks = Vec::new();
    for b in &trak_boxes {
        if let Some(t) = parse_trak(src, b, movie_timescale, &trex)? {
            tracks.push(t);
        }
    }

    // 有 mvex **且**采样表是空的，才算分片文件。
    // 两者都有（混合文件）时按渐进式读：`stbl` 里的采样是确定存在的，
    // 而分片可能只覆盖后面一部分，先给能给的。
    let has_samples = tracks.iter().any(|t| t.stbl.is_some());
    Ok(Moov { movie_timescale, duration_ticks, tracks, fragmented: fragmented && !has_samples })
}

// ---------------------------------------------------------------------------
// moof（fMP4 分片）解析
// ---------------------------------------------------------------------------

/// 分片里解出来的一条采样。
#[derive(Debug, Clone, Copy)]
struct FragmentSample {
    track_index: usize,
    offset: u64,
    size: u32,
    duration: u32,
    /// 轨道时基的解码时间
    decode_time: i64,
    composition_offset: i32,
    keyframe: bool,
}

/// `tfhd` 里我们关心的字段。
#[derive(Debug, Default, Clone, Copy)]
struct Tfhd {
    track_id: u32,
    base_data_offset: Option<u64>,
    default_sample_duration: Option<u32>,
    default_sample_size: Option<u32>,
    default_sample_flags: Option<u32>,
    /// `default-base-is-moof`（flags 位 0x020000）：没有 `base_data_offset` 时，
    /// 采样数据的基准是**这个 moof 的起点**（B 站 / ffmpeg 都是这么写的）。
    default_base_is_moof: bool,
}

fn parse_tfhd(p: &[u8]) -> Option<Tfhd> {
    let mut r = ByteReader::new(p);
    r.u8().ok()?;
    let flags = r.u24_be().ok()?;
    let mut h = Tfhd { track_id: r.u32_be().ok()?, ..Default::default() };
    if flags & 0x01 != 0 {
        h.base_data_offset = Some(r.u64_be().ok()?);
    }
    if flags & 0x02 != 0 {
        r.skip(4).ok()?; // sample_description_index
    }
    if flags & 0x08 != 0 {
        h.default_sample_duration = Some(r.u32_be().ok()?);
    }
    if flags & 0x10 != 0 {
        h.default_sample_size = Some(r.u32_be().ok()?);
    }
    if flags & 0x20 != 0 {
        h.default_sample_flags = Some(r.u32_be().ok()?);
    }
    h.default_base_is_moof = flags & 0x02_0000 != 0;
    Some(h)
}

fn parse_tfdt(p: &[u8]) -> Option<i64> {
    let mut r = ByteReader::new(p);
    let version = r.u8().ok()?;
    r.skip(3).ok()?;
    let v = if version == 1 { r.u64_be().ok()? as i64 } else { r.u32_be().ok()? as i64 };
    Some(v)
}

/// 采样标志 → 是否随机访问点。
///
/// 权威判据是 `sample_is_non_sync_sample`（bit 16）。`sample_depends_on == 1`
/// 表示"这一帧依赖别的帧"，那当然不能从它切入。两者取"与"是为了**保守**：
/// 少认一个关键帧只是 seek 落点偏早一点（多解几帧），多认一个就是花屏。
fn sample_is_sync(kind: MediaKind, flags: u32) -> bool {
    if !matches!(kind, MediaKind::Video) {
        // 音频采样每条都能独立解码
        return true;
    }
    let non_sync = (flags >> 16) & 0x01 != 0;
    let depends_on = (flags >> 24) & 0x03;
    !non_sync && depends_on != 1
}

/// 解析一个 `moof`，把里面所有 `traf` 的采样摊平。
fn parse_moof(
    src: &dyn ReadAt,
    moof: &BoxHeader,
    tracks: &[Mp4Track],
) -> Result<Vec<FragmentSample>> {
    let mut out = Vec::new();
    // `trun.data_offset` 的相对基准（ISO 14496-12 §8.8.8），按下面的顺序确定：
    //   1. `tfhd.base_data_offset` 写了 → 用它（绝对文件偏移）；
    //   2. 否则，若 `default-base-is-moof` 或它是**本 moof 的第一个 traf**
    //      → **这个 moof 的起点**；
    //   3. 否则 → 前一个 traf 数据的结束位置。
    //
    // 现实里绝大多数文件走第 2 条：`data_offset = moof 长度 + 8`，
    // 数值上正好落在紧随其后的 mdat 载荷起点。**把它当成"相对 moof 末尾"是错的** ——
    // 那样每个分片的数据都会整整错位一个 moof 的长度（自产自销时读写一致看不出来，
    // 一读 B 站 / ffmpeg 的文件就全是垃圾，最后一个分片还会直接读越界）。
    let mut base = moof.start;
    let mut first_traf = true;
    for traf in children(src, moof.payload_start, moof.end)? {
        if &traf.kind != b"traf" {
            continue;
        }
        let mut h = Tfhd::default();
        let mut track_index = None;
        let mut decode_time = 0i64;
        for b in children(src, traf.payload_start, traf.end)? {
            match &b.kind {
                b"tfhd" => {
                    let p = read_payload(src, &b)?;
                    let Some(parsed) = parse_tfhd(&p) else { continue };
                    h = parsed;
                    track_index = tracks.iter().position(|t| t.id == h.track_id);
                    base = match h.base_data_offset {
                        Some(v) => v,
                        None if h.default_base_is_moof || first_traf => moof.start,
                        // 多 traf 且没声明默认基准：接着上一个 traf 的数据往后排
                        None => base,
                    };
                }
                b"tfdt" => {
                    let p = read_payload(src, &b)?;
                    decode_time = parse_tfdt(&p).unwrap_or(0);
                }
                b"trun" => {
                    let Some(ti) = track_index else { continue };
                    let t = &tracks[ti];
                    let p = read_payload(src, &b)?;
                    // 返回值 = 本条 traf 数据的结束位置，供后面（可能的）下一个 traf 当基准
                    base = parse_trun(&p, &h, t, ti, base, &mut decode_time, &mut out)?;
                }
                _ => {}
            }
        }
        first_traf = false;
    }
    Ok(out)
}

/// 解析一个 `trun`，把采样追加到 `out`。
///
/// 返回**本条 traf 数据的结束位置**（下一个 traf 若没声明自己的基准就接着它排）。
fn parse_trun(
    p: &[u8],
    tfhd: &Tfhd,
    track: &Mp4Track,
    track_index: usize,
    base: u64,
    decode_time: &mut i64,
    out: &mut Vec<FragmentSample>,
) -> Result<u64> {
    let mut r = ByteReader::new(p);
    let version = r.u8()?;
    let flags = r.u24_be()?;
    let count = r.u32_be()?;
    let data_offset = if flags & 0x01 != 0 { Some(r.u32_be()? as i32) } else { None };
    let first_sample_flags = if flags & 0x04 != 0 { Some(r.u32_be()?) } else { None };

    // 默认值优先级：trun 里写的 > tfhd 里的 > trex 里的
    let def_dur = tfhd.default_sample_duration.or(track.trex.map(|t| t.default_sample_duration)).unwrap_or(0);
    let def_size = tfhd.default_sample_size.or(track.trex.map(|t| t.default_sample_size)).unwrap_or(0);
    let def_flags = tfhd.default_sample_flags.or(track.trex.map(|t| t.default_sample_flags)).unwrap_or(0);

    let mut off = match data_offset {
        Some(d) => {
            let v = base as i64 + d as i64;
            if v < 0 {
                0
            } else {
                v as u64
            }
        }
        None => base,
    };

    for i in 0..count {
        let dur = if flags & 0x100 != 0 { r.u32_be()? } else { def_dur };
        let size = if flags & 0x200 != 0 { r.u32_be()? } else { def_size };
        let sf = if flags & 0x400 != 0 {
            r.u32_be()?
        } else if i == 0 {
            first_sample_flags.unwrap_or(def_flags)
        } else {
            def_flags
        };
        let cts = if flags & 0x800 != 0 {
            // version 0 是无符号、version 1 是有符号；实际数值都很小，直接按位转即可
            r.u32_be()? as i32
        } else {
            0
        };
        let _ = version;
        out.push(FragmentSample {
            track_index,
            offset: off,
            size,
            duration: dur,
            decode_time: *decode_time,
            composition_offset: cts,
            keyframe: sample_is_sync(track.kind, sf),
        });
        off = off.saturating_add(size as u64);
        *decode_time += dur as i64;
    }
    Ok(off)
}

// ---------------------------------------------------------------------------
// 采样数据形态转换
// ---------------------------------------------------------------------------

/// 长度前缀布局 → Annex-B。
///
/// 实现在 [`codec::length_prefixed_to_annexb`]（**唯一实现**，Matroska 那条链
/// 也要用它）；这里保留这层薄转发只是为了不改动本文件里已有的调用点。
fn to_annexb(data: &[u8], nal_len: u8) -> Vec<u8> {
    codec::length_prefixed_to_annexb(data, nal_len)
}

/// Annex-B 里有没有随机访问点（表格/标志说"不是"时的兜底判据）。
///
/// 有些编码器写的 `stss` 是错的（比如整条流一个关键帧都没标），
/// 只信表格会让 seek 永远落在开头。这里再扫一遍码流确认。
fn annexb_has_random_access(codec: &CodecId, data: &[u8]) -> bool {
    let hevc = match codec {
        CodecId::H265 => true,
        CodecId::H264 => false,
        _ => return false,
    };
    codec::split_annexb(data).iter().any(|nal| codec::is_random_access_nal(hevc, nal))
}

/// 从 `stts` / `trex` 推帧率（只在"每帧时长都一样"时给值）。
fn frame_rate_of(t: &Mp4Track) -> Option<(u32, u32)> {
    if !matches!(t.kind, MediaKind::Video) {
        return None;
    }
    let delta = match t.stbl.as_ref().and_then(|s| s.stts.first()) {
        Some(&(count, d)) if count >= 2 && d > 0 => d,
        _ => t.trex.map(|x| x.default_sample_duration).filter(|d| *d > 0)?,
    };
    let (n, d) = codec::reduce(t.timescale as u64, delta as u64)?;
    if n == 0 || d == 0 || (n as f64 / d as f64) > 240.0 {
        return None;
    }
    Some((n, d))
}

// ---------------------------------------------------------------------------
// 解封装器
// ---------------------------------------------------------------------------

/// ISO BMFF 解封装器。
pub struct Mp4Demuxer {
    src: Arc<dyn ReadAt>,
    source_name: String,
    /// `ftyp` 里的主品牌（诊断用：宿主可以显示"这是 isom 还是 iso6"）
    brand: String,
    tracks: Vec<Mp4Track>,
    track_index: HashMap<u32, usize>,
    movie_timescale: u32,
    movie_duration_ticks: Option<i64>,
    fragmented: bool,
    /// 渐进式的每轨游标
    cursors: HashMap<u32, SampleCursor>,
    /// 每轨"已经取出来但还没轮到输出"的一条采样（跨轨按时间交织要用）
    pending: HashMap<u32, SampleTuple>,
    /// 待输出的采样（分片模式一次解析一个 moof，会攒一小批）
    queue: VecDeque<Packet>,
    /// 下一个要看的顶层盒子位置
    next_box: u64,
    finished: bool,
    /// seek 之后要丢掉"早于这个时间"的采样（微秒）—— 关键帧之前的 B 帧
    skip_before_us: Option<i64>,
    /// 时长缓存（分片模式要扫一遍分片才知道，别每次都扫）
    duration_cache: OnceLock<Option<i64>>,
    /// 第一帧的媒体时间（微秒）。
    ///
    /// seek 的目标早于第一个关键帧时要**如实返回它**，而不是返回 0 ——
    /// 源的时间轴不一定从 0 开始（实测 Apple 的分片从 10 秒起），
    /// 报 0 会让上层的"起点归零"算错，表现是进度条长度不对、一开播就显示结束。
    first_media_cache: OnceLock<Option<i64>>,
}

impl Mp4Demuxer {
    /// 打开一个 ISO BMFF 源（只读 `moov`，不碰 `mdat`）。
    pub fn new(src: Arc<dyn ReadAt>, source: impl Into<String>) -> Result<Self> {
        let source_name = source.into();

        let mut brand = String::new();
        if let Some(ftyp) = find_top_level(&*src, b"ftyp", 0)? {
            let p = read_payload(&*src, &ftyp)?;
            if p.len() >= 4 {
                brand = String::from_utf8_lossy(&p[..4]).into_owned();
            }
        }

        let Some(moov) = find_top_level(&*src, b"moov", 0)? else {
            return Err(MediaError::container(
                "ISO BMFF 里找不到 moov：文件被截断，或者 moov 还没下到（边下边播时它是第一优先要的盒子）",
            ));
        };
        let m = parse_moov(&*src, &moov)?;
        if m.tracks.is_empty() {
            return Err(MediaError::container("moov 里没有任何可用轨道（trak 全被跳过）"));
        }

        let mut track_index = HashMap::new();
        for (i, t) in m.tracks.iter().enumerate() {
            track_index.insert(t.id, i);
        }
        let mut cursors = HashMap::new();
        for t in &m.tracks {
            if let Some(table) = &t.stbl {
                cursors.insert(t.id, SampleCursor::new(table));
            }
        }

        Ok(Self {
            src,
            source_name,
            brand,
            tracks: m.tracks,
            track_index,
            movie_timescale: m.movie_timescale,
            movie_duration_ticks: m.duration_ticks,
            fragmented: m.fragmented,
            cursors,
            pending: HashMap::new(),
            queue: VecDeque::new(),
            next_box: 0,
            finished: false,
            skip_before_us: None,
            duration_cache: OnceLock::new(),
            first_media_cache: OnceLock::new(),
        })
    }

    /// 探测：只读 `moov`，所以对一个 4 GB 的文件也是毫秒级。
    pub fn probe_reader(src: Arc<dyn ReadAt>, source: &str) -> Result<(ContainerFormat, MediaInfo)> {
        let d = Self::new(src, source)?;
        Ok((ContainerFormat::IsoBmff, d.info()))
    }

    /// 主品牌（`ftyp` 的前 4 字节）。
    pub fn brand(&self) -> &str {
        &self.brand
    }

    /// 是不是分片（fMP4）文件。
    pub fn is_fragmented(&self) -> bool {
        self.fragmented
    }

    /// 轨道与总览信息。
    pub fn info(&self) -> MediaInfo {
        let tracks: Vec<Track> = self
            .tracks
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let mut tr = Track::new(t.id, t.kind, t.codec.clone(), t.timebase());
                tr.extradata = t.extradata.clone();
                tr.width = t.width;
                tr.height = t.height;
                tr.sample_rate = t.sample_rate;
                tr.channels = t.channels;
                tr.language = t.language.clone();
                // 渐进式看 stts、分片看第一个 moof —— 两处都不看就永远报"未知帧率"
                tr.frame_rate = frame_rate_of(t).or_else(|| self.fragment_frame_rate(i));
                tr
            })
            .collect();
        let duration_us = self.duration_us();
        let bitrate = match (self.src.len(), duration_us) {
            (Some(len), Some(dur)) if dur > 0 => {
                Some(((len as u128) * 8 * 1_000_000 / dur as u128) as u64)
            }
            _ => None,
        };
        MediaInfo {
            container: ContainerFormat::IsoBmff.name().to_string(),
            duration_us,
            bitrate,
            tracks,
            source: self.source_name.clone(),
        }
    }

    /// 时长（微秒）。
    ///
    /// 优先 `mvhd`（电影时基），其次各轨 `mdhd`，最后（分片文件才有必要）
    /// 扫一遍分片取最后一个 `tfdt + 时长`。分片那次扫描会缓存，
    /// 不会因为调用方多问几次就反复扫。
    pub fn duration_us(&self) -> Option<i64> {
        *self.duration_cache.get_or_init(|| self.compute_duration())
    }

    fn compute_duration(&self) -> Option<i64> {
        if let Some(d) = self.movie_duration_ticks {
            if d > 0 {
                return Some(Timebase::new(self.movie_timescale.max(1)).ticks_to_us(d));
            }
        }
        let by_track = self
            .tracks
            .iter()
            .filter_map(|t| t.duration_ticks.filter(|d| *d > 0).map(|d| t.timebase().ticks_to_us(d)))
            .max();
        if by_track.is_some() {
            return by_track;
        }
        if self.fragmented {
            return self.fragmented_duration_us();
        }
        None
    }

    /// 扫一遍分片，用"最后一个分片的结束 − 第一个分片的开始"当时长。
    ///
    /// 两点刻意为之：
    ///
    /// - **算的是跨度，不是绝对结束时间**。fMP4 的媒体时间轴不保证从 0 开始
    ///   （真实样例：Apple 的 bipbop fMP4 第一条 `tfdt` 是 10 秒，整段只有 6 秒，
    ///   直接报结束时间就会把 6 秒的片子报成 16 秒）。
    /// - 只在 `mvhd`/`mdhd` 都没给时长时才走这条路（fMP4 的 `mvhd` 时长就是 0）。
    ///   代价是每个顶层盒子一次 8~16 字节的读：一个两小时、每 2 秒一个分片的片子
    ///   是几千次读，对本地文件与已下载完的数据可以接受；直播时 `mvhd` 本来
    ///   就不该有时长，这条路径不会走到。
    fn fragmented_duration_us(&self) -> Option<i64> {
        let mut pos = 0u64;
        let mut first_moof = None;
        let mut last_moof = None;
        let mut scanned = 0u64;
        while scanned < MAX_TOP_LEVEL_SCAN {
            let Some(b) = next_top_level(&*self.src, pos).ok()? else { break };
            if &b.kind == b"moof" {
                if first_moof.is_none() {
                    first_moof = Some(b);
                }
                last_moof = Some(b);
            }
            pos = b.end;
            scanned += 1;
        }
        let mut start_us = i64::MAX;
        let mut end_us = i64::MIN;
        for moof in [first_moof, last_moof].into_iter().flatten() {
            let Ok(samples) = parse_moof(&*self.src, &moof, &self.tracks) else { continue };
            for s in &samples {
                let Some(t) = self.tracks.get(s.track_index) else { continue };
                let base = s.decode_time + t.edit_shift;
                start_us = start_us.min(t.timebase().ticks_to_us(base + s.composition_offset as i64));
                end_us = end_us.max(t.timebase().ticks_to_us(base + s.duration as i64));
            }
        }
        if start_us == i64::MAX || end_us == i64::MIN || end_us <= start_us {
            return None;
        }
        Some(end_us - start_us)
    }

    /// 分片文件的帧率要从**分片头**里取。
    ///
    /// `trex` 的 `default_sample_duration` 经常是 0（真实样例：Apple 的 fMP4
    /// 只在 `tfhd`/`trun` 里带时长），光看 `trex` 会让所有 fMP4 源都报"帧率未知"。
    /// 只读第一个 `moof`（几百字节，且不读采样数据），代价可以忽略。
    fn fragment_frame_rate(&self, track_index: usize) -> Option<(u32, u32)> {
        let t = self.tracks.get(track_index)?;
        if !matches!(t.kind, MediaKind::Video) {
            return None;
        }
        let moof = find_top_level(&*self.src, b"moof", 0).ok()??;
        // 边下边播时这个 moof 可能还没下到：探测**不能因此阻塞**，直接放弃
        if let Some(len) = self.src.len() {
            if moof.end > len {
                return None;
            }
        }
        let samples = parse_moof(&*self.src, &moof, &self.tracks).ok()?;
        let dur = samples.iter().find(|s| s.track_index == track_index)?.duration;
        if dur == 0 {
            return None;
        }
        let (n, d) = codec::reduce(t.timescale as u64, dur as u64)?;
        if n == 0 || d == 0 || (n as f64 / d as f64) > 240.0 {
            return None;
        }
        Some((n, d))
    }

    /// 取下一个采样；`Ok(None)` = 到头了。
    pub fn next_packet(&mut self) -> Result<Option<Packet>> {
        loop {
            if let Some(p) = self.queue.pop_front() {
                return Ok(Some(p));
            }
            if self.finished {
                return Ok(None);
            }
            if self.fragmented {
                self.pump_fragment()?;
            } else {
                self.pump_progressive()?;
            }
        }
    }

    /// 渐进式：每轨取一条待输出，选解码时间最早的那条交出去。
    ///
    /// 为什么要"跨轨挑最早"：文件里的 chunk 是**交织**存放的，但轨道之间
    /// 谁先谁后没有保证。播放器与 muxer 都希望拿到的是"大致按时间递增"的流，
    /// 直接把所有视频轨读完再读音频轨会让下游的缓冲策略彻底失效。
    fn pump_progressive(&mut self) -> Result<()> {
        for i in 0..self.tracks.len() {
            let t = &self.tracks[i];
            if t.stbl.is_none() {
                continue;
            }
            if self.pending.contains_key(&t.id) {
                continue;
            }
            let table = t.stbl.as_ref().expect("上面刚判过");
            self.cursors
                .entry(t.id)
                .or_insert_with(|| SampleCursor::new(table));
            let cur = self.cursors.get_mut(&t.id).expect("刚插入");
            // 编辑列表**不丢采样**（见 `parse_elst`）：只平移时间戳，
            // 前置采样的负时间戳在封装层被夹到 0
            let got = cur.take(table);
            if let Some(s) = got {
                self.pending.insert(t.id, s);
            }
        }

        if self.pending.is_empty() {
            self.finished = true;
            return Ok(());
        }
        // 取 dts 最小的那条；dts 相同时按轨道号定序。
        //
        // 这里的"平局"不是罕见情况：音频与视频的第一条采样都在 0。
        // 不定序的话，输出顺序会随 `HashMap` 的迭代顺序变化（每进程不同），
        // 上层拿到的流会时而"视频在前"时而"音频在前" —— 那是最难复现的一类 bug。
        //
        // ⚠️ 比较前必须把 dts 换算到**公共时间轴**（见 `pump_fragment` 里那段长注释：
        // 各轨时基不同，直接比原始 tick 会把音频推到 1/3 速、播放端音频饿死）。
        let mut best: Option<(i64, u32)> = None;
        for (id, s) in &self.pending {
            let us = match self.track_index.get(id) {
                Some(&ti) => self.tracks[ti].timebase().ticks_to_us(s.2),
                None => s.2,
            };
            let key = (us, *id);
            if best.map(|(d, i)| key < (d, i)).unwrap_or(true) {
                best = Some(key);
            }
        }
        let Some((_, best_id)) = best else {
            self.finished = true;
            return Ok(());
        };
        let s = self.pending.remove(&best_id).expect("刚判过非空");
        let ti = match self.track_index.get(&best_id) {
            Some(i) => *i,
            None => return Ok(()),
        };
        let pkt = self.build_packet(ti, s.0, s.1, s.2, s.3, s.4, s.5)?;
        self.push_packet(pkt);
        Ok(())
    }

    /// 分片：往后找 `moof`，把里面的采样按解码时间排好塞进队列。
    fn pump_fragment(&mut self) -> Result<()> {
        loop {
            let Some(b) = next_top_level(&*self.src, self.next_box)? else {
                self.finished = true;
                return Ok(());
            };
            self.next_box = b.end;
            if &b.kind != b"moof" {
                continue;
            }
            let samples = parse_moof(&*self.src, &b, &self.tracks)?;
            if samples.is_empty() {
                continue;
            }
            let mut sorted = samples;
            // ⚠️ 跨轨排序必须先把 `decode_time` 换算到**公共时间轴**（微秒）。
            //
            // 各轨的 `mdhd` 时基**通常不同**（这条素材：视频 16000、音频 48000），
            // 而 `tfdt`/`trun` 出来的 dts 是**各自轨道时基**的 tick —— 直接拿原始
            // tick 比大小，等于按"tick 数"对齐：时基大的那一轨（音频，48000）
            // 在输出序里被推到 `16000/48000 = 1/3` 的速度、成串爆发。
            // 播放端（`stream --video=frames --audio=pcm`）按输出序解码，于是
            // 每次都先解完一整段视频才收到一小段音频 ⇒ **音频队列饿死 = 听感上的
            // "跳帧/断音"**（2026-10-06 定位；H.264 硬解因为快而不明显，AV1 软解
            // 每帧 ~80ms 就压不住了）。
            //
            // 同轨之间用原始 tick 兜底，保证**轨内顺序严格不变**；
            // 跨轨同一微秒时按轨道号定序（确定性，与 `pump_progressive` 的约定一致）。
            sorted.sort_by_key(|s| {
                let us = self.tracks[s.track_index]
                    .timebase()
                    .ticks_to_us(s.decode_time);
                (us, s.track_index, s.decode_time)
            });
            for s in sorted {
                let pkt = self.build_packet(
                    s.track_index,
                    s.offset,
                    s.size,
                    s.decode_time,
                    s.duration,
                    s.composition_offset,
                    s.keyframe,
                )?;
                self.push_packet(pkt);
            }
            if !self.queue.is_empty() {
                return Ok(());
            }
        }
    }

    /// 读采样数据并组装成 [`Packet`]。
    ///
    /// 参数确实多，但它们都是"一条采样的描述"，拆成结构体反而要多写一层
    /// 中转（调用点就在下面两处、字段全部来自 `SampleTuple`）。
    #[allow(clippy::too_many_arguments)]
    fn build_packet(
        &self,
        track_index: usize,
        offset: u64,
        size: u32,
        dts: i64,
        duration: u32,
        cts: i32,
        key: bool,
    ) -> Result<Packet> {
        let t = &self.tracks[track_index];
        let raw = self.src.read_exact_at(offset, size as usize)?;
        let data = if matches!(t.kind, MediaKind::Video) {
            to_annexb(&raw, t.nal_length_size)
        } else {
            raw
        };
        let keyframe = if key {
            true
        } else {
            // 表格/标志说"不是关键帧"时再扫一遍码流确认（有些编码器标错了）
            matches!(t.kind, MediaKind::Video) && annexb_has_random_access(&t.codec, &data)
        };
        // 编辑列表的平移可能让开头几条采样的时间戳变成负数（那几条可能是关键帧，
        // **不能丢**）。这里如实报出来，由 muxer 把整条流的起点归零 ——
        // 在解析层夹取会破坏采样之间的间距（前几帧的时间戳会撞在一起），
        // 而 muxer 归零是"整条流一起平移"，间距与音视频相对偏移都能保住。
        let shifted_dts = dts + t.edit_shift;
        let shifted_pts = shifted_dts + cts as i64;
        let mut p = Packet::new(t.id, shifted_pts, shifted_dts, duration, data);
        p.keyframe = keyframe;
        Ok(p)
    }

    /// 入队（并处理 seek 之后"落点之前的数据不输出"）。
    ///
    /// 过滤**只对非视频轨**生效：音频/字幕按时间切很自然（切在哪儿都能解），
    /// 而视频必须从关键帧整段往后放 —— 把关键帧之后、显示时间早于落点的
    /// 那几个 B 帧丢掉，会让解码器缺少参考帧而花屏。宁可多解几帧。
    fn push_packet(&mut self, p: Packet) {
        if let Some(skip) = self.skip_before_us {
            if let Some(&ti) = self.track_index.get(&p.track_id) {
                let t = &self.tracks[ti];
                if !matches!(t.kind, MediaKind::Video) {
                    let end_us = t.timebase().ticks_to_us(p.pts + p.duration as i64);
                    if end_us <= skip {
                        return;
                    }
                }
            }
        }
        self.queue.push_back(p);
    }

    /// 跳到 `pts_us` 之前最近的关键帧，返回实际落点（微秒）。
    ///
    /// 定位完全靠 `stbl` 里的索引：**没有任何 IO**（不像 TS 那层要重扫字节）。
    /// 落点由主视频轨的关键帧决定，其余轨道按时间对齐到同一点。
    pub fn seek_us(&mut self, pts_us: i64) -> Result<i64> {
        if self.fragmented {
            return self.seek_fragmented(pts_us);
        }
        // 主轨：优先视频，没有视频就用第一条有采样表的轨道
        let primary = self
            .tracks
            .iter()
            .position(|t| t.stbl.is_some() && matches!(t.kind, MediaKind::Video))
            .or_else(|| self.tracks.iter().position(|t| t.stbl.is_some()));
        let Some(pi) = primary else {
            self.reset_stream();
            return Ok(0);
        };

        // 第一遍：顺着采样表找 `<= target` 的最后一个关键帧，同时记下它的下标
        let mut landing: Option<(i64, u32)> = None; // (pts 刻度, 采样下标 0 基)
        {
            let t = &self.tracks[pi];
            let table = match t.stbl.as_ref() {
                Some(x) => x,
                None => {
                    self.reset_stream();
                    return Ok(0);
                }
            };
            let target = t.timebase().us_to_ticks(pts_us);
            let is_video = matches!(t.kind, MediaKind::Video);
            let mut cur = SampleCursor::new(table);
            while cur.next <= table.sample_count {
                // 采样下标必须在 `take` **之前**算：`take` 会推进 `next`，
                // 之后再减 1 得到的是"下一条"的下标（off-by-one，
                // 症状是 seek 落点偏后一个采样 —— 视频上就是"多丢一帧"）
                let here = cur.next - 1;
                let Some(s) = cur.take(table) else { break };
                let pts = s.2 + s.4 as i64;
                if pts > target {
                    break;
                }
                if !is_video || s.5 {
                    landing = Some((pts, here));
                }
            }
        }

        let Some((pts_ticks, key_index)) = landing else {
            // 找不到（目标在第一个关键帧之前）：回到开头 —— **开头是源的第一个
            // 媒体时间，不一定是 0**（见 `first_media_us`），如实返回它
            self.reset_stream();
            return Ok(self.first_media_us().unwrap_or(0));
        };

        let landing_us = self.tracks[pi].timebase().ticks_to_us(pts_ticks);
        let primary_id = self.tracks[pi].id;

        // 第二遍：重置各轨游标并空转到落点
        self.reset_stream();
        for i in 0..self.tracks.len() {
            let t = &self.tracks[i];
            let Some(table) = t.stbl.as_ref() else { continue };
            let start_index = if t.id == primary_id {
                key_index
            } else {
                // 其余轨道：落到第一条**结束时间不早于**落点的采样。
                // 默认值取"采样总数"= 这条轨道在落点之后没有内容了，那就什么都不吐，
                // 免得把一条已经播完的轨道硬拖到落点之后。
                let target = t.timebase().us_to_ticks(landing_us);
                let mut idx = table.sample_count;
                let mut cur = SampleCursor::new(table);
                while cur.next <= table.sample_count {
                    let here = cur.next - 1;
                    let Some(s) = cur.take(table) else { break };
                    if s.2 + s.3 as i64 >= target {
                        idx = here;
                        break;
                    }
                }
                idx
            };
            let mut cur = SampleCursor::new(table);
            while cur.next <= start_index {
                if cur.take(table).is_none() {
                    break;
                }
            }
            self.cursors.insert(t.id, cur);
        }
        self.skip_before_us = Some(landing_us);
        Ok(landing_us)
    }

    /// 源的第一个媒体时间（微秒）。
    ///
    /// 为什么要有它：源的时间轴**不一定从 0 开始**（Apple 的 HLS 分片从 10 秒起、
    /// DASH 段常常从 10 秒起）。seek 的目标早于第一个关键帧时，正确行为是
    /// "从头开始"，而"头"就是这里返回的值 —— 返回 0 会让上层的起点归零算错。
    fn first_media_us(&self) -> Option<i64> {
        *self.first_media_cache.get_or_init(|| self.compute_first_media_us())
    }

    fn compute_first_media_us(&self) -> Option<i64> {
        if self.fragmented {
            let mut pos = 0u64;
            let mut scanned = 0u64;
            while scanned < MAX_TOP_LEVEL_SCAN {
                let b = next_top_level(&*self.src, pos).ok()??;
                pos = b.end;
                scanned += 1;
                if &b.kind != b"moof" {
                    continue;
                }
                let samples = parse_moof(&*self.src, &b, &self.tracks).ok()?;
                let mut min = i64::MAX;
                for s in &samples {
                    let t = self.tracks.get(s.track_index)?;
                    let us = t.timebase().ticks_to_us(
                        s.decode_time + s.composition_offset as i64 + t.edit_shift,
                    );
                    min = min.min(us);
                }
                if min != i64::MAX {
                    return Some(min);
                }
            }
            None
        } else {
            // 渐进式：第一条采样（任一有采样表的轨道）
            let idx = self.tracks.iter().position(|t| t.stbl.is_some())?;
            let t = &self.tracks[idx];
            let table = t.stbl.as_ref()?;
            let mut cur = SampleCursor::new(table);
            let s = cur.take(table)?;
            Some(t.timebase().ticks_to_us(s.2 + s.4 as i64 + t.edit_shift))
        }
    }

    /// 分片模式下的 seek：两遍扫分片（第一遍定位，第二遍从头放）。
    ///
    /// **只认视频的关键帧**：音频采样每条都"能独立解码"（`sample_is_sync`
    /// 对音频恒为真），拿它当落点会让"报出来的落点"落在音频时间轴上，
    /// 而实际吐出的第一条视频却在下一个分片 —— 实测差整整一个分片（2 秒）。
    /// 没有视频轨时才退化成"按任意轨定位"。
    fn seek_fragmented(&mut self, pts_us: i64) -> Result<i64> {
        let has_video = self.tracks.iter().any(|t| matches!(t.kind, MediaKind::Video));
        let mut best: Option<(u64, i64)> = None; // (moof 起点, 关键帧时间 us)
        let mut pos = 0u64;
        let mut scanned = 0u64;
        while scanned < MAX_TOP_LEVEL_SCAN {
            let Some(b) = next_top_level(&*self.src, pos)? else { break };
            pos = b.end;
            scanned += 1;
            if &b.kind != b"moof" {
                continue;
            }
            let samples = parse_moof(&*self.src, &b, &self.tracks)?;
            let mut min_us = i64::MAX;
            for s in &samples {
                let Some(t) = self.tracks.get(s.track_index) else { continue };
                let is_video = matches!(t.kind, MediaKind::Video);
                let us = t.timebase().ticks_to_us(s.decode_time + s.composition_offset as i64 + t.edit_shift);
                // 定位只看视频轨（见上面的说明）
                if has_video && !is_video {
                    continue;
                }
                min_us = min_us.min(us);
                if s.keyframe && us <= pts_us {
                    let better = best.map(|(_, bp)| us > bp).unwrap_or(true);
                    if better {
                        best = Some((b.start, us));
                    }
                }
            }
            // 分片是按时间递增排的：这个分片的最早时间已经超过目标，后面不用看了
            if min_us != i64::MAX && min_us > pts_us {
                break;
            }
        }

        let Some((moof_start, landing)) = best else {
            // 目标在第一个关键帧之前：从头开始（源的开头不一定是 0）
            self.reset_stream();
            return Ok(self.first_media_us().unwrap_or(0));
        };
        self.reset_stream();
        self.next_box = moof_start;
        self.skip_before_us = Some(landing);
        Ok(landing)
    }

    /// 回到流的开头（保留轨道信息）。
    pub fn reset_stream(&mut self) {
        self.queue.clear();
        self.pending.clear();
        self.cursors.clear();
        for i in 0..self.tracks.len() {
            if let Some(table) = self.tracks[i].stbl.as_ref() {
                self.cursors.insert(self.tracks[i].id, SampleCursor::new(table));
            }
        }
        self.next_box = 0;
        self.finished = false;
        self.skip_before_us = None;
    }
}

impl Demuxer for Mp4Demuxer {
    fn info(&self) -> MediaInfo {
        Mp4Demuxer::info(self)
    }

    fn next_packet(&mut self) -> Result<Option<Packet>> {
        Mp4Demuxer::next_packet(self)
    }

    fn seek(&mut self, pts_us: i64) -> Result<i64> {
        self.seek_us(pts_us)
    }

    fn duration_us(&self) -> Option<i64> {
        Mp4Demuxer::duration_us(self)
    }

    fn container(&self) -> &'static str {
        ContainerFormat::IsoBmff.name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use me_core::Cursor;

    // ── 合成文件用的构造器 ─────────────────────────────────────────────
    //
    // 为什么自己拼字节而不放一个样例文件进仓库：
    // 样例文件是二进制，改不动也看不清；而"这个测试到底在测什么"必须能从代码读出来。
    // 手拼还有一个好处：可以精确造出真实文件里那些脏形态（stsz 统一尺寸、
    // 多 chunk、elst 空编辑、分片 trun 省略字段）。

    fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut v = ((body.len() + 8) as u32).to_be_bytes().to_vec();
        v.extend_from_slice(kind);
        v.extend_from_slice(body);
        v
    }

    fn full(kind: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
        let mut b = vec![version];
        b.extend_from_slice(&flags.to_be_bytes()[1..]);
        b.extend_from_slice(body);
        bx(kind, &b)
    }

    fn cat(parts: &[Vec<u8>]) -> Vec<u8> {
        parts.concat()
    }

    fn mvhd(timescale: u32, duration: u32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&0u32.to_be_bytes()); // creation_time
        b.extend_from_slice(&0u32.to_be_bytes()); // modification_time
        b.extend_from_slice(&timescale.to_be_bytes());
        b.extend_from_slice(&duration.to_be_bytes());
        b.extend_from_slice(&[0u8; 80]); // rate/volume/matrix/next_track_id
        full(b"mvhd", 0, 0, &b)
    }

    fn tkhd(id: u32, w: u32, h: u32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&0u32.to_be_bytes());
        b.extend_from_slice(&0u32.to_be_bytes());
        b.extend_from_slice(&id.to_be_bytes());
        b.extend_from_slice(&0u32.to_be_bytes()); // reserved
        b.extend_from_slice(&0u32.to_be_bytes()); // duration
        b.extend_from_slice(&[0u8; 8]); // reserved
        b.extend_from_slice(&0u16.to_be_bytes()); // layer
        b.extend_from_slice(&0u16.to_be_bytes()); // alternate_group
        b.extend_from_slice(&0u16.to_be_bytes()); // volume
        b.extend_from_slice(&0u16.to_be_bytes()); // reserved
        b.extend_from_slice(&[0u8; 36]); // matrix
        b.extend_from_slice(&(w << 16).to_be_bytes()); // 16.16 定点
        b.extend_from_slice(&(h << 16).to_be_bytes());
        full(b"tkhd", 0, 7, &b)
    }

    fn mdhd(timescale: u32, duration: u32, lang: &[u8; 3]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&0u32.to_be_bytes());
        b.extend_from_slice(&0u32.to_be_bytes());
        b.extend_from_slice(&timescale.to_be_bytes());
        b.extend_from_slice(&duration.to_be_bytes());
        let l = (((lang[0] - 0x60) as u16) << 10)
            | (((lang[1] - 0x60) as u16) << 5)
            | ((lang[2] - 0x60) as u16);
        b.extend_from_slice(&l.to_be_bytes());
        b.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
        full(b"mdhd", 0, 0, &b)
    }

    fn hdlr(kind: &[u8; 4]) -> Vec<u8> {
        let mut b = vec![0u8; 4];
        b.extend_from_slice(kind);
        b.extend_from_slice(&[0u8; 12]);
        b.push(0);
        full(b"hdlr", 0, 0, &b)
    }

    fn stts(entries: &[(u32, u32)]) -> Vec<u8> {
        let mut b = (entries.len() as u32).to_be_bytes().to_vec();
        for &(n, d) in entries {
            b.extend_from_slice(&n.to_be_bytes());
            b.extend_from_slice(&d.to_be_bytes());
        }
        full(b"stts", 0, 0, &b)
    }

    fn ctts(entries: &[(u32, i32)]) -> Vec<u8> {
        let mut b = (entries.len() as u32).to_be_bytes().to_vec();
        for &(n, o) in entries {
            b.extend_from_slice(&n.to_be_bytes());
            b.extend_from_slice(&o.to_be_bytes());
        }
        full(b"ctts", 0, 0, &b)
    }

    fn stsc(entries: &[(u32, u32, u32)]) -> Vec<u8> {
        let mut b = (entries.len() as u32).to_be_bytes().to_vec();
        for &(f, n, d) in entries {
            b.extend_from_slice(&f.to_be_bytes());
            b.extend_from_slice(&n.to_be_bytes());
            b.extend_from_slice(&d.to_be_bytes());
        }
        full(b"stsc", 0, 0, &b)
    }

    fn stsz_uniform(size: u32, count: u32) -> Vec<u8> {
        let mut b = size.to_be_bytes().to_vec();
        b.extend_from_slice(&count.to_be_bytes());
        full(b"stsz", 0, 0, &b)
    }

    fn stsz_sizes(sizes: &[u32]) -> Vec<u8> {
        let mut b = 0u32.to_be_bytes().to_vec();
        b.extend_from_slice(&(sizes.len() as u32).to_be_bytes());
        for s in sizes {
            b.extend_from_slice(&s.to_be_bytes());
        }
        full(b"stsz", 0, 0, &b)
    }

    fn stco(offsets: &[u32]) -> Vec<u8> {
        let mut b = (offsets.len() as u32).to_be_bytes().to_vec();
        for o in offsets {
            b.extend_from_slice(&o.to_be_bytes());
        }
        full(b"stco", 0, 0, &b)
    }

    fn stss(list: &[u32]) -> Vec<u8> {
        let mut b = (list.len() as u32).to_be_bytes().to_vec();
        for s in list {
            b.extend_from_slice(&s.to_be_bytes());
        }
        full(b"stss", 0, 0, &b)
    }

    /// 1920x1080 的 SPS（含 NAL 头 0x67），与 ts.rs 的测试用的是同一套位序列。
    fn sps_1080p() -> Vec<u8> {
        // 位写入用内部函数：两个闭包同时可变借用 `bits` 会被借用检查器拒绝
        fn push(bits: &mut Vec<u8>, v: u64, n: u32) {
            for i in (0..n).rev() {
                bits.push(((v >> i) & 1) as u8);
            }
        }
        fn ue(bits: &mut Vec<u8>, v: u32) {
            let v1 = v as u64 + 1;
            let n = 64 - v1.leading_zeros();
            for _ in 0..n - 1 {
                bits.push(0);
            }
            for i in (0..n).rev() {
                bits.push(((v1 >> i) & 1) as u8);
            }
        }
        let mut bits: Vec<u8> = Vec::new();
        push(&mut bits, 66, 8); // profile_idc
        push(&mut bits, 0, 8); // constraint flags
        push(&mut bits, 30, 8); // level_idc
        ue(&mut bits, 0); // sps_id
        ue(&mut bits, 0); // log2_max_frame_num_minus4
        ue(&mut bits, 0); // pic_order_cnt_type
        ue(&mut bits, 0); // log2_max_pic_order_cnt_lsb_minus4
        ue(&mut bits, 1); // max_num_ref_frames
        push(&mut bits, 0, 1); // gaps_in_frame_num_value_allowed_flag
        ue(&mut bits, 119); // pic_width_in_mbs_minus1 → 120*16 = 1920
        ue(&mut bits, 67); // pic_height_in_map_units_minus1 → 68*16 = 1088
        push(&mut bits, 1, 1); // frame_mbs_only_flag
        push(&mut bits, 1, 1); // direct_8x8_inference_flag
        push(&mut bits, 1, 1); // frame_cropping_flag
        ue(&mut bits, 0); // crop_left
        ue(&mut bits, 0); // crop_right
        ue(&mut bits, 0); // crop_top
        ue(&mut bits, 4); // crop_bottom → 1088 - 4*2 = 1080
        push(&mut bits, 0, 1); // vui_parameters_present_flag
        let mut bytes = vec![0x67u8];
        let mut cur = 0u8;
        let mut n = 0;
        for b in &bits {
            cur = (cur << 1) | b;
            n += 1;
            if n == 8 {
                bytes.push(cur);
                cur = 0;
                n = 0;
            }
        }
        if n > 0 {
            bytes.push(cur << (8 - n));
        }
        bytes
    }

    fn avcc(nal_len: u8) -> Vec<u8> {
        let sps = sps_1080p();
        let pps = vec![0x68u8, 0xCE, 0x3C, 0x80];
        let mut b = vec![1u8, sps[1], sps[2], sps[3], 0xFC | (nal_len - 1), 1];
        b.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        b.extend_from_slice(&sps);
        b.push(1);
        b.extend_from_slice(&(pps.len() as u16).to_be_bytes());
        b.extend_from_slice(&pps);
        b
    }

    fn esds_asc(asc: &[u8]) -> Vec<u8> {
        // DecoderSpecificInfo(0x05) → DecoderConfigDescriptor(0x04) → ES_Descriptor(0x03)
        let mut dsi = vec![0x05u8, asc.len() as u8];
        dsi.extend_from_slice(asc);
        let mut dc = vec![0x04u8, (13 + dsi.len()) as u8];
        dc.extend_from_slice(&[0x40, 0x15]); // objectType=AAC, streamType=audio
        dc.extend_from_slice(&[0, 0, 0]); // bufferSizeDB
        dc.extend_from_slice(&0u32.to_be_bytes()); // maxBitrate
        dc.extend_from_slice(&0u32.to_be_bytes()); // avgBitrate
        dc.extend_from_slice(&dsi);
        let mut es = vec![0x03u8, (3 + dc.len()) as u8];
        es.extend_from_slice(&0u16.to_be_bytes()); // ES_ID
        es.push(0); // flags
        es.extend_from_slice(&dc);
        full(b"esds", 0, 0, &es)
    }

    fn visual_entry(format: &[u8; 4], w: u16, h: u16, subs: &[Vec<u8>]) -> Vec<u8> {
        let mut b = vec![0u8; 6]; // reserved
        b.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
        b.extend_from_slice(&[0u8; 16]); // pre_defined + reserved + pre_defined
        b.extend_from_slice(&w.to_be_bytes());
        b.extend_from_slice(&h.to_be_bytes());
        b.extend_from_slice(&0x0048_0000u32.to_be_bytes());
        b.extend_from_slice(&0x0048_0000u32.to_be_bytes());
        b.extend_from_slice(&0u32.to_be_bytes());
        b.extend_from_slice(&1u16.to_be_bytes()); // frame_count
        b.extend_from_slice(&[0u8; 32]); // compressorname
        b.extend_from_slice(&0x0018u16.to_be_bytes()); // depth
        b.extend_from_slice(&0xFFFFu16.to_be_bytes()); // pre_defined
        for s in subs {
            b.extend_from_slice(s);
        }
        bx(format, &b)
    }

    fn audio_entry(format: &[u8; 4], channels: u16, rate: u32, subs: &[Vec<u8>]) -> Vec<u8> {
        let mut b = vec![0u8; 6]; // reserved
        b.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
        b.extend_from_slice(&[0u8; 8]); // reserved
        b.extend_from_slice(&channels.to_be_bytes());
        b.extend_from_slice(&16u16.to_be_bytes()); // sample_size
        b.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
        b.extend_from_slice(&0u16.to_be_bytes()); // reserved
        b.extend_from_slice(&(rate << 16).to_be_bytes()); // 16.16 定点
        for s in subs {
            b.extend_from_slice(s);
        }
        bx(format, &b)
    }

    fn stsd(entry: &[u8]) -> Vec<u8> {
        let mut b = 1u32.to_be_bytes().to_vec(); // entry_count
        b.extend_from_slice(entry);
        full(b"stsd", 0, 0, &b)
    }

    /// 造一条 `trak`（测试用的拼装器，参数多是它的本质：每条表都要显式给）。
    #[allow(clippy::too_many_arguments)]
    fn trak(
        id: u32,
        handler: &[u8; 4],
        timescale: u32,
        duration: u32,
        lang: &[u8; 3],
        w: u32,
        h: u32,
        stsd_box: Vec<u8>,
        tables: Vec<Vec<u8>>,
        elst: Option<Vec<u8>>,
    ) -> Vec<u8> {
        let mut stbl_parts = vec![stsd_box];
        stbl_parts.extend(tables);
        let stbl = bx(b"stbl", &cat(&stbl_parts));
        let minf = bx(b"minf", &stbl);
        let mdia = bx(b"mdia", &cat(&[mdhd(timescale, duration, lang), hdlr(handler), minf]));
        let mut parts = vec![tkhd(id, w, h)];
        if let Some(e) = elst {
            parts.push(bx(b"edts", &e));
        }
        parts.push(mdia);
        bx(b"trak", &cat(&parts))
    }

    /// 构造器状态：一边拼盒子一边记下"mdat 载荷起点"。
    struct FileBuilder {
        head: Vec<u8>,
        body: Vec<u8>,
        tracks: Vec<Vec<u8>>,
    }

    impl FileBuilder {
        fn new() -> Self {
            Self { head: Vec::new(), body: Vec::new(), tracks: Vec::new() }
        }

        /// 当前（还没拼出最终文件时）文件长度：用来算绝对偏移。
        fn body_offset(&self) -> u32 {
            self.head.len() as u32
        }

        /// `body` 内某个偏移在**最终文件**里的绝对位置。
        ///
        /// 这里有个绕不开的自指：`stco` 里要写 `mdat` 的绝对偏移，而 `mdat`
        /// 在 `moov` 之后 —— 也就是说偏移值取决于 `moov` 的长度，`moov` 的长度
        /// 又取决于 `stco` 里写的值。真实编码器是靠"写完 moov 再回填"或
        /// "先算长度"解决的；测试里用同样便宜的办法：**先按占位值建一遍量长度**。
        /// `stco` 的条目数不随偏移值变化，所以长度是稳定的。
        fn absolute(&self, mvex: Option<Vec<u8>>, body_off: usize) -> u32 {
            let moov_len = self.build(mvex).len() - self.head.len() - self.body.len();
            (self.head.len() + moov_len + body_off) as u32
        }

        fn push_body(&mut self, bytes: &[u8]) {
            self.body.extend_from_slice(bytes);
        }

        fn build(&self, mvex: Option<Vec<u8>>) -> Vec<u8> {
            // 分片文件的 mvhd 时长就是 0（真实 fMP4 都这样）：时长由分片自己带
            let movie_duration = if mvex.is_some() { 0 } else { 10_000 };
            let mut moov_parts = vec![mvhd(1000, movie_duration)];
            if let Some(m) = &mvex {
                moov_parts.push(m.clone());
            }
            moov_parts.extend(self.tracks.iter().cloned());
            let moov = bx(b"moov", &cat(&moov_parts));
            cat(&[self.head.clone(), moov, self.body.clone()])
        }
    }

    fn ftyp() -> Vec<u8> {
        let mut b = b"isom".to_vec();
        b.extend_from_slice(&512u32.to_be_bytes());
        b.extend_from_slice(b"isomiso2avc1mp41");
        bx(b"ftyp", &b)
    }

    // ── 渐进式 MP4 ────────────────────────────────────────────────────

    /// 造一个"视频 + 音频"的渐进式 MP4：
    /// - 视频：`timescale = 30000`，3 帧（关键帧只有第 1 帧），时长 1000 刻度 ≈ 33.3ms
    /// - 音频：`timescale = 44100`，4 帧，每帧 1024 刻度
    /// - `mdat` 里视频在前、音频在后（真实的交织文件）
    fn progressive_mp4(nal_len: u8) -> Vec<u8> {
        let mut fb = FileBuilder::new();
        fb.head = ftyp();

        // 视频采样：长度前缀布局（AVCC）
        let mut v_data = Vec::new();
        let nals: Vec<Vec<u8>> = vec![
            vec![0x65, 0x11, 0x22], // IDR
            vec![0x41, 0x33],       // 非 IDR
            vec![0x41, 0x44],
        ];
        let mut v_sizes = Vec::new();
        for n in nals.iter() {
            let mut s = Vec::new();
            s.extend_from_slice(&(n.len() as u32).to_be_bytes()[4 - nal_len as usize..]);
            s.extend_from_slice(n);
            v_sizes.push(s.len() as u32);
            v_data.extend_from_slice(&s);
        }
        // 音频采样：裸 AAC（不做任何转换）
        let mut a_data = Vec::new();
        let mut a_sizes = Vec::new();
        for i in 0..4u8 {
            let f = vec![0x21u8, 0x10, i, 0x55];
            a_sizes.push(f.len() as u32);
            a_data.extend_from_slice(&f);
        }

        let mut mdat_body = Vec::new();
        mdat_body.extend_from_slice(&v_data);
        mdat_body.extend_from_slice(&a_data);
        fb.push_body(&bx(b"mdat", &mdat_body));

        let make_v = |off: u32| {
            trak(
                1,
                b"vide",
                30_000,
                3000,
                b"und",
                1920,
                1080,
                stsd(&visual_entry(b"avc1", 1920, 1080, &[bx(b"avcC", &avcc(nal_len))])),
                vec![
                    stts(&[(3, 1000)]),
                    stsc(&[(1, 3, 1)]),
                    stsz_sizes(&v_sizes),
                    stco(&[off]),
                    stss(&[1]),
                ],
                None,
            )
        };
        let make_a = |off: u32| {
            trak(
                2,
                b"soun",
                44_100,
                4096,
                b"eng",
                0,
                0,
                stsd(&audio_entry(b"mp4a", 2, 44_100, &[esds_asc(&[0x12, 0x10])])),
                vec![
                    stts(&[(4, 1024)]),
                    stsc(&[(1, 4, 1)]),
                    stsz_sizes(&a_sizes),
                    stco(&[off]),
                ],
                None,
            )
        };

        // 两遍构建：先用占位偏移量出 moov 长度（条目数不变，长度稳定），
        // 再回填 `mdat` 载荷的真实绝对偏移
        fb.tracks = vec![make_v(0), make_a(0)];
        let v_off = fb.absolute(None, 8); // body 内偏移 8 = 跳过 mdat 的 8 字节头
        let a_off = v_off + v_data.len() as u32;
        fb.tracks = vec![make_v(v_off), make_a(a_off)];
        fb.build(None)
    }

    fn demux(data: Vec<u8>) -> Mp4Demuxer {
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
        Mp4Demuxer::new(src, "mem").unwrap()
    }

    #[test]
    fn detects_tracks_codec_and_geometry() {
        let d = demux(progressive_mp4(4));
        let info = d.info();
        assert_eq!(info.container, "iso-bmff");
        assert_eq!(info.tracks.len(), 2);

        let v = info.video().expect("要有视频轨");
        assert_eq!(v.codec, CodecId::H264);
        assert_eq!(v.width, 1920);
        assert_eq!(v.height, 1080);
        assert_eq!(v.timebase.ticks_per_second, 30_000);
        // extradata 必须是 avcC 的载荷（不是 Annex-B）：第一条是 version=1
        assert_eq!(v.extradata.first(), Some(&1u8));
        assert_eq!(v.frame_rate, Some((30, 1)));

        let a = info.audio().expect("要有音频轨");
        assert_eq!(a.codec, CodecId::Aac);
        assert_eq!(a.sample_rate, 44_100);
        assert_eq!(a.channels, 2);
        assert_eq!(a.language.as_deref(), Some("eng"));
        // extradata 是 AudioSpecificConfig（esds 里的 DecoderSpecificInfo）
        assert_eq!(a.extradata, vec![0x12, 0x10]);
    }

    #[test]
    fn samples_come_out_interleaved_with_correct_timing() {
        let mut d = demux(progressive_mp4(4));
        let mut video = Vec::new();
        let mut audio = Vec::new();
        while let Some(p) = d.next_packet().unwrap() {
            if p.track_id == 1 {
                video.push(p);
            } else {
                audio.push(p);
            }
        }
        assert_eq!(video.len(), 3);
        assert_eq!(audio.len(), 4);

        // 视频：dts 按 stts 递增，pts == dts（没有 ctts）
        assert_eq!(video[0].dts, 0);
        assert_eq!(video[1].dts, 1000);
        assert_eq!(video[2].dts, 2000);
        assert!(video[0].keyframe, "stss 标了第 1 帧是关键帧");
        assert!(!video[1].keyframe);
        assert!(!video[2].keyframe);

        // 视频数据必须已经转成 Annex-B（含 00 00 00 01 起始码）
        assert_eq!(&video[0].data[..4], &[0, 0, 0, 1]);
        assert_eq!(video[0].data[4], 0x65, "IDR 的 NAL 头要保留");
        assert_eq!(video[1].data[4], 0x41);

        // 音频：裸 AAC 原样，不许多出起始码
        assert_eq!(audio[0].data[0], 0x21);
        assert_eq!(audio[0].dts, 0);
        assert_eq!(audio[1].dts, 1024);
        assert_eq!(audio[1].duration, 1024);

        // 交织：视频第一帧(0) 必须排在音频第四帧(3072) 之前
        let mut d2 = demux(progressive_mp4(4));
        let first = d2.next_packet().unwrap().unwrap();
        assert_eq!(first.track_id, 1, "时间最早的采样要先出来");
    }

    #[test]
    fn different_nal_length_size_is_honoured() {
        // 2 字节长度前缀：转换必须按 avcC 说的来，不能写死 4
        let mut d = demux(progressive_mp4(2));
        let p = d.next_packet().unwrap().unwrap();
        assert_eq!(&p.data[..4], &[0, 0, 0, 1]);
        assert_eq!(p.data[4], 0x65);
        // 转换后应该是 "起始码 + 3 字节 NAL"，而不是把长度前缀当数据留下
        assert_eq!(p.data.len(), 4 + 3);
    }

    #[test]
    fn duration_uses_movie_timescale() {
        let d = demux(progressive_mp4(4));
        // mvhd: 1000 刻度/秒，10000 刻度 = 10 秒
        assert_eq!(d.duration_us(), Some(10_000_000));
    }

    #[test]
    fn seek_lands_on_keyframe_and_drops_earlier_audio() {
        let mut d = demux(progressive_mp4(4));
        // 目标 60ms：视频只有第 1 帧(0ms)是关键帧 → 落点 0
        let landing = d.seek_us(60_000).unwrap();
        assert_eq!(landing, 0);
        let p = d.next_packet().unwrap().unwrap();
        assert_eq!(p.track_id, 1);
        assert!(p.keyframe);

        // 目标 10 秒（超过所有采样）→ 仍落到最后一个关键帧，不能报错
        let mut d2 = demux(progressive_mp4(4));
        let landing2 = d2.seek_us(10_000_000).unwrap();
        assert_eq!(landing2, 0, "只有第 1 帧是关键帧，落点只能是它");
        assert!(d2.next_packet().unwrap().is_some());
    }

    #[test]
    fn probe_only_reads_moov_not_mdat() {
        // 把 mdat 做成 1MB，探测必须照样秒回（这里用"能探出来"间接验证：
        // 如果探测去读 mdat，下面的断言在超大文件上会明显变慢，且更容易触发
        // 读越界错误）
        let mut data = progressive_mp4(4);
        // 直接在 mdat 里塞垃圾：`moov` 在 mdat 之后，解析必须不受影响
        let mdat_pos = data.windows(4).position(|w| w == b"mdat").unwrap();
        let insert_at = mdat_pos + 4;
        data.splice(insert_at..insert_at, std::iter::repeat(0xAAu8).take(1024));
        // 插入了字节 → stco 里的偏移失效，但探测（只读 moov）依然要成功
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
        let info = Mp4Demuxer::probe_reader(src, "mem").unwrap().1;
        assert_eq!(info.tracks.len(), 2);
        assert_eq!(info.video().unwrap().width, 1920);
    }

    // ── 分片 fMP4 ─────────────────────────────────────────────────────

    /// 造一个 moof。
    ///
    /// `data_offset` 按**规范**写成"moof 长度 + 8"，并置 `default-base-is-moof`
    /// —— 基准是 moof 的起点，这正是 B 站 / ffmpeg 的写法（也是引擎曾经读错、
    /// 导致整个分片数据错位一个 moof 长度的那种文件）。
    fn moof(track_id: u32, decode_time: u64, samples: &[(u32, u32)]) -> Vec<u8> {
        let build = |data_offset: i32| {
            // tfhd：track_ID + default_sample_duration + default-base-is-moof
            let mut tfhd_body = track_id.to_be_bytes().to_vec();
            tfhd_body.extend_from_slice(&1000u32.to_be_bytes()); // default_sample_duration
            let tfhd = full(b"tfhd", 0, 0x08 | 0x02_0000, &tfhd_body);

            let tfdt_body = (decode_time as u32).to_be_bytes().to_vec();
            let tfdt = full(b"tfdt", 0, 0, &tfdt_body);

            // trun：data_offset + 每条采样的 size / flags（第一个带 first_sample_flags）
            let mut trun_body = (samples.len() as u32).to_be_bytes().to_vec();
            trun_body.extend_from_slice(&data_offset.to_be_bytes());
            trun_body.extend_from_slice(&0u32.to_be_bytes()); // first_sample_flags
            for (size, flags) in samples {
                trun_body.extend_from_slice(&size.to_be_bytes());
                trun_body.extend_from_slice(&flags.to_be_bytes());
            }
            let trun = full(b"trun", 0, 0x01 | 0x04 | 0x200 | 0x400, &trun_body);

            let traf = bx(b"traf", &cat(&[tfhd, tfdt, trun]));
            bx(b"moof", &cat(&[full(b"mfhd", 0, 0, &1u32.to_be_bytes()), traf]))
        };
        // data_offset 是定长 4 字节：先按 0 构建一次拿到长度，再写正确值
        let moof_len = build(0).len() as i32;
        build(moof_len + 8)
    }

    /// 造一个分片 fMP4：moov（空采样表 + mvex/trex）+ 三个分片。
    fn fragmented_mp4() -> Vec<u8> {
        let mut fb = FileBuilder::new();
        fb.head = ftyp();

        // moov：stbl 全空（fMP4 的常态）
        let v_track = trak(
            1,
            b"vide",
            30_000,
            0,
            b"und",
            1920,
            1080,
            stsd(&visual_entry(b"avc1", 1920, 1080, &[bx(b"avcC", &avcc(4))])),
            vec![stts(&[]), stsc(&[]), stsz_sizes(&[]), stco(&[])],
            None,
        );
        let a_track = trak(
            2,
            b"soun",
            44_100,
            0,
            b"und",
            0,
            0,
            stsd(&audio_entry(b"mp4a", 2, 44_100, &[esds_asc(&[0x12, 0x10])])),
            vec![stts(&[]), stsc(&[]), stsz_sizes(&[]), stco(&[])],
            None,
        );
        fb.tracks = vec![v_track, a_track];

        let mut mvex_body = Vec::new();
        for (id, dur, size) in [(1u32, 1000u32, 0u32), (2, 1024, 4)] {
            let mut t = id.to_be_bytes().to_vec();
            t.extend_from_slice(&1u32.to_be_bytes());
            t.extend_from_slice(&dur.to_be_bytes());
            t.extend_from_slice(&size.to_be_bytes());
            t.extend_from_slice(&0u32.to_be_bytes());
            mvex_body.extend_from_slice(&bx(b"trex", &t));
        }
        let mvex = bx(b"mvex", &mvex_body);

        // 采样标志：bit16 = sample_is_non_sync，bit24-25 = sample_depends_on
        let key_flags = 0x0200_0000u32; // 同步点
        let non_key = 0x0101_0000u32; // depends_on=1 + non_sync

        // 采样数据紧跟 moof 之后，data_offset 由 `moof()` 按规范算（moof 长度 + 8）

        // 分片 0：视频 2 帧
        let v0: Vec<Vec<u8>> = vec![
            {
                let mut s = 4u32.to_be_bytes().to_vec();
                s.extend_from_slice(&[0x65, 0x01]);
                s
            },
            {
                let mut s = 4u32.to_be_bytes().to_vec();
                s.extend_from_slice(&[0x41, 0x02]);
                s
            },
        ];
        let m0 = moof(1, 0, &[(6, key_flags), (6, non_key)]);
        let mut v_payload = Vec::new();
        for s in &v0 {
            v_payload.extend_from_slice(s);
        }
        let frag0 = cat(&[m0, bx(b"mdat", &v_payload)]);

        // 分片 1：音频 1 帧
        let a0 = vec![0x21u8, 0x10, 0x01, 0x55];
        let m1 = moof(2, 0, &[(4, key_flags)]);
        let frag1 = cat(&[m1, bx(b"mdat", &a0)]);

        // 分片 2：视频 1 帧（解码时间接着上一个分片）
        let v1 = {
            let mut s = 4u32.to_be_bytes().to_vec();
            s.extend_from_slice(&[0x65, 0x03]);
            s
        };
        let m2 = moof(1, 2000, &[(6, key_flags)]);
        let frag2 = cat(&[m2, bx(b"mdat", &v1)]);

        fb.body = cat(&[frag0, frag1, frag2]);
        fb.build(Some(mvex))
    }

    #[test]
    fn fragmented_mp4_is_recognised() {
        let d = demux(fragmented_mp4());
        assert!(d.is_fragmented(), "有 mvex 且采样表为空 = 分片文件");
        let info = d.info();
        assert_eq!(info.tracks.len(), 2);
        assert_eq!(info.video().unwrap().width, 1920);
    }

    #[test]
    fn fragmented_samples_are_read_from_trun() {
        let mut d = demux(fragmented_mp4());
        let mut all = Vec::new();
        while let Some(p) = d.next_packet().unwrap() {
            all.push(p);
        }
        assert_eq!(all.len(), 4, "2 + 1 + 1 条采样");
        // 第一个分片的两条视频
        assert_eq!(all[0].track_id, 1);
        assert_eq!(all[0].dts, 0);
        assert!(all[0].keyframe);
        assert_eq!(&all[0].data[..4], &[0, 0, 0, 1]);
        assert_eq!(all[0].data[4], 0x65);
        // 第二条：non_sync 标志 + depends_on=1 → 不是关键帧
        assert_eq!(all[1].track_id, 1);
        assert!(!all[1].keyframe, "sample_flags 说它不是同步点");
        // 音频
        assert_eq!(all[2].track_id, 2);
        assert_eq!(all[2].data[0], 0x21, "音频不做 Annex-B 转换");
        // 第二个视频分片：tfdt 给的 baseMediaDecodeTime = 2000
        assert_eq!(all[3].track_id, 1);
        assert_eq!(all[3].dts, 2000);
    }

    #[test]
    fn fragmented_duration_comes_from_last_fragment() {
        let d = demux(fragmented_mp4());
        // 最后一个分片：tfdt = 2000 刻度 + 1000 刻度 = 3000 刻度 / 30000 = 100ms
        assert_eq!(d.duration_us(), Some(100_000));
    }

    #[test]
    fn fragmented_seek_moves_to_matching_fragment() {
        let mut d = demux(fragmented_mp4());
        let landing = d.seek_us(50_000).unwrap();
        assert_eq!(landing, 0, "50ms 落在第一个分片");
        assert!(d.next_packet().unwrap().is_some());

        let mut d2 = demux(fragmented_mp4());
        let landing2 = d2.seek_us(80_000).unwrap();
        assert_eq!(landing2, 66_666, "80ms 应该落到第二个分片(2000 刻度 ≈ 66.6ms)");
        let p = d2.next_packet().unwrap().unwrap();
        assert_eq!(p.dts, 2000);
    }

    // ── 编辑列表 ──────────────────────────────────────────────────────

    #[test]
    fn empty_edit_shifts_timestamps() {
        // elst：一条空编辑（media_time = -1，时长 100 电影刻度 = 100ms），
        // 然后一条正常编辑从 0 开始。音频的时间戳必须整体后移 100ms。
        let mut fb = FileBuilder::new();
        fb.head = ftyp();
        let mut elst_body = 1u32.to_be_bytes().to_vec();
        elst_body.extend_from_slice(&100u32.to_be_bytes()); // segment_duration（电影时基 1000）
        elst_body.extend_from_slice(&(-1i32).to_be_bytes()); // media_time = -1 → 空编辑
        elst_body.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // rate 1.0
        elst_body.extend_from_slice(&0u32.to_be_bytes());
        elst_body.extend_from_slice(&9000u32.to_be_bytes());
        elst_body.extend_from_slice(&0i32.to_be_bytes());
        elst_body.extend_from_slice(&0x0001_0000u32.to_be_bytes());
        elst_body.extend_from_slice(&0u32.to_be_bytes());
        let elst = full(b"elst", 0, 0, &elst_body);

        let data = vec![0x21u8, 0x10, 0x01, 0x55];
        let mut fb_body = Vec::new();
        fb_body.extend_from_slice(&bx(b"mdat", &data));
        fb.push_body(&fb_body);
        let off = fb.body_offset();
        let a = trak(
            2,
            b"soun",
            44_100,
            1024,
            b"eng",
            0,
            0,
            stsd(&audio_entry(b"mp4a", 2, 44_100, &[esds_asc(&[0x12, 0x10])])),
            vec![stts(&[(1, 1024)]), stsc(&[(1, 1, 1)]), stsz_sizes(&[4]), stco(&[off])],
            Some(elst),
        );
        fb.tracks = vec![a];
        let bytes = fb.build(None);

        let mut d = demux(bytes);
        let p = d.next_packet().unwrap().unwrap();
        // 100ms 换算到 44100 时基 = 4410 刻度
        assert_eq!(p.pts, 4410, "空编辑要把时间戳整体后移");
        assert_eq!(p.dts, 4410);
    }

    // ── 脏数据 ────────────────────────────────────────────────────────

    #[test]
    fn garbage_does_not_panic() {
        for data in [
            vec![],
            vec![0u8; 16],
            b"not a media file".to_vec(),
            {
                // 长度字段比文件大得多
                let mut v = 0xFFFF_FFFFu32.to_be_bytes().to_vec();
                v.extend_from_slice(b"ftyp");
                v.extend_from_slice(&[0u8; 32]);
                v
            },
            {
                // 盒子套娃到很深
                let mut inner = bx(b"moov", &[]);
                for _ in 0..200 {
                    inner = bx(b"trak", &inner);
                }
                cat(&[ftyp(), inner])
            },
        ] {
            let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
            // 只要求"不 panic"，报错是允许的（而且大多数情况就该报错）
            let _ = Mp4Demuxer::new(src, "mem");
        }
    }

    #[test]
    fn truncated_file_reports_error_not_panic() {
        let full = progressive_mp4(4);
        // 砍掉 moov 的后半段
        let cut = full.len() / 2;
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(full[..cut].to_vec()));
        let r = Mp4Demuxer::new(src, "mem");
        assert!(r.is_err(), "moov 不完整时必须报错，而不是给出一堆空轨道");
    }

    #[test]
    fn missing_moov_is_a_clear_error() {
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(ftyp()));
        // 用 `err()` 而不是 `unwrap_err()`：后者要求 `Ok` 类型实现 Debug，
        // 而解封装器里有 `Arc<dyn ReadAt>`，为测试硬加一个 Debug 实现不值得
        let e = Mp4Demuxer::new(src, "mem").err().expect("只有 ftyp 必须报错");
        assert!(e.to_string().contains("moov"), "错误信息要指出缺什么：{e}");
    }

    #[test]
    fn to_annexb_handles_broken_lengths() {
        // 长度越界：剩下的字节当最后一个 NAL 带上，不能丢数据也不能 panic
        let data = vec![0, 0, 0, 99, 0x65, 0x11];
        let out = to_annexb(&data, 4);
        assert_eq!(&out[..4], &[0, 0, 0, 1]);
        assert!(out.ends_with(&[0x65, 0x11]));

        // nal_len = 0（音频/VP9/AV1）：原样返回
        let raw = vec![1u8, 2, 3];
        assert_eq!(to_annexb(&raw, 0), raw);

        // 全是零长度：不能返回空包（那会让上层以为"这条采样没有数据"）
        let zeros = vec![0u8; 8];
        assert_eq!(to_annexb(&zeros, 4), zeros);
    }

    #[test]
    fn uniform_stsz_and_multiple_chunks() {
        // stsz 用"统一尺寸"形态 + 采样分散在多个 chunk 里：
        // 游标必须跟着 stsc 换 chunk 偏移，否则第二条采样会从错的地方读
        let mut fb = FileBuilder::new();
        fb.head = ftyp();
        // 采样用 4 字节长度前缀布局（和 avcC 里的 nal_len 一致）
        let s1 = vec![0u8, 0, 0, 2, 0x65, 0xAA];
        let s2 = vec![0u8, 0, 0, 2, 0x41, 0xBB];
        let mut mdat = Vec::new();
        mdat.extend_from_slice(&s1);
        mdat.extend_from_slice(&[0xFFu8; 16]); // chunk 之间的填充
        mdat.extend_from_slice(&s2);
        fb.push_body(&bx(b"mdat", &mdat));
        let make = |c1: u32, c2: u32| {
            trak(
                1,
                b"vide",
                30_000,
                2000,
                b"und",
                1920,
                1080,
                stsd(&visual_entry(b"avc1", 1920, 1080, &[bx(b"avcC", &avcc(4))])),
                vec![
                    stts(&[(2, 1000)]),
                    stsc(&[(1, 1, 1), (2, 1, 1)]),
                    stsz_uniform(6, 2),
                    stco(&[c1, c2]),
                    stss(&[1]),
                ],
                None,
            )
        };
        fb.tracks = vec![make(0, 0)];
        let base = fb.absolute(None, 8); // mdat 载荷起点
        fb.tracks = vec![make(base, base + 6 + 16)];
        let mut d = demux(fb.build(None));
        let p1 = d.next_packet().unwrap().unwrap();
        let p2 = d.next_packet().unwrap().unwrap();
        assert_eq!(p1.data[4], 0x65);
        assert_eq!(p2.data[4], 0x41, "第二个 chunk 的偏移要按 stco 走");
        assert!(d.next_packet().unwrap().is_none());
    }

    #[test]
    fn resolution_falls_back_to_sps_when_tkhd_is_zero() {
        // tkhd 宽高写 0（编码器偷懒），必须从 avcC 里的 SPS 补出来
        let mut fb = FileBuilder::new();
        fb.head = ftyp();
        let s = vec![0u8, 0, 0, 2, 0x65, 0xAA];
        fb.push_body(&bx(b"mdat", &s));
        let make = |off: u32| {
            trak(
                1,
                b"vide",
                30_000,
                1000,
                b"und",
                0,
                0,
                stsd(&visual_entry(b"avc1", 0, 0, &[bx(b"avcC", &avcc(4))])),
                vec![stts(&[(1, 1000)]), stsc(&[(1, 1, 1)]), stsz_sizes(&[6]), stco(&[off])],
                None,
            )
        };
        fb.tracks = vec![make(0)];
        let off = fb.absolute(None, 8);
        fb.tracks = vec![make(off)];
        let d = demux(fb.build(None));
        let t = d.info().video().cloned().unwrap();
        assert_eq!((t.width, t.height), (1920, 1080));
    }

    #[test]
    fn composition_offset_becomes_pts_difference() {
        // ctts：B 帧重排。pts 必须 = dts + ctts 偏移，dts 保持解码顺序
        let mut fb = FileBuilder::new();
        fb.head = ftyp();
        let s = vec![0u8, 0, 0, 2, 0x41, 0xAA];
        let mut mdat = Vec::new();
        mdat.extend_from_slice(&s);
        mdat.extend_from_slice(&s); // 两条采样，各 6 字节
        fb.push_body(&bx(b"mdat", &mdat));
        let make = |off: u32| {
            trak(
                1,
                b"vide",
                30_000,
                2000,
                b"und",
                1920,
                1080,
                stsd(&visual_entry(b"avc1", 1920, 1080, &[bx(b"avcC", &avcc(4))])),
                vec![
                    stts(&[(2, 1000)]),
                    ctts(&[(1, 1000), (1, 0)]),
                    stsc(&[(1, 2, 1)]),
                    stsz_sizes(&[6, 6]),
                    stco(&[off]),
                    stss(&[1]),
                ],
                None,
            )
        };
        fb.tracks = vec![make(0)];
        let off = fb.absolute(None, 8);
        fb.tracks = vec![make(off)];
        let mut d = demux(fb.build(None));
        let p1 = d.next_packet().unwrap().unwrap();
        let p2 = d.next_packet().unwrap().unwrap();
        assert_eq!((p1.dts, p1.pts), (0, 1000));
        assert_eq!((p2.dts, p2.pts), (1000, 1000));
    }
}
