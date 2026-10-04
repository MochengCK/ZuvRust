//! 播放器：把"字节源 + 解封装 + 缓冲 + 寻址"包成一件东西，供 CLI 的 `play` 用。
//!
//! ## 三条设计取舍
//!
//! 1. **单线程、拉取式**（调用方要一个包，我们就按需解一个）。
//!    不自己开预读线程：解封装器是 `&mut` 的，多线程要加锁；而"按需解"
//!    天然就让读位置领先播放位置 —— 边下边播要的正是"播放头往前推"这个信号，
//!    不需要额外的预读线程去制造它。
//! 2. **缓冲有上限**（按微秒 + 条数双重封顶）。不封顶的话，一个两小时的片子
//!    被"边解边攒"就能把内存吃光；封顶之后，解到上限就停，等调用方消费。
//! 3. **播放头要能被上报**。宿主把它转达给下载引擎（XferRust）去做
//!    "按播放位置优先下载"。这是媒体引擎与下载引擎之间唯一的协作接口 ——
//!    媒体引擎里**没有** BT 调度逻辑。

use std::collections::VecDeque;
use std::sync::Arc;

use me_container::demux::{self, Demuxer};
use me_core::{MediaError, MediaInfo, Packet, ReadAt, Result};

/// 播放头提示：宿主拿它去告诉下载引擎"先下这一段"。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayheadHint {
    /// 当前读到的字节偏移。
    pub offset_bytes: u64,
    /// 当前播放时间（微秒）。
    pub time_us: i64,
}

/// 播放统计。
#[derive(Debug, Clone, Copy, Default)]
pub struct PlayerStats {
    /// 已经吐出去的包数。
    pub emitted_packets: u64,
    /// 缓冲区里还有多少包。
    pub buffered_packets: usize,
    /// 缓冲区覆盖的媒体时长（微秒）。
    pub buffered_us: i64,
    /// 当前播放时间（微秒）。
    pub playhead_us: i64,
    /// 源上的读位置（字节）。
    pub source_read_offset: u64,
}

/// 播放器配置。
#[derive(Debug, Clone)]
pub struct PlayerConfig {
    /// 缓冲上限（微秒）：攒够这么久就不再预读。
    pub max_buffer_us: i64,
    /// 缓冲上限（包数）：兜底，防止"每包时长都是 0"时按微秒算不出来。
    pub max_buffer_packets: usize,
}

impl Default for PlayerConfig {
    fn default() -> Self {
        Self { max_buffer_us: 2_000_000, max_buffer_packets: 4096 }
    }
}

pub struct Player {
    demuxer: Box<dyn Demuxer>,
    /// 字节源：留着是为了报告"源有多长"（进度条与"下完了没有"的判断要用）。
    source: Arc<dyn ReadAt>,
    /// 已经解出来、等着被消费的包。
    buffer: VecDeque<Packet>,
    cfg: PlayerConfig,
    playhead_us: i64,
    emitted: u64,
    /// 源上"读到哪里"（用最后一条解出来的包的字节位置近似：
    /// 容器层没有暴露精确偏移，而播放头只需要**单调前进**就够下载引擎用了）。
    read_offset: u64,
    /// 缓冲区里第一条采样的开始时间（微秒）；空缓冲为 None。
    first_buffered_us: Option<i64>,
    /// 缓冲区里最后一条采样的结束时间（微秒）。
    last_buffered_end_us: i64,
}

impl Player {
    /// 打开一个源。
    pub fn open(source: Arc<dyn ReadAt>, name: &str, cfg: PlayerConfig) -> Result<Self> {
        let demuxer = demux::open(source.clone(), name)?;
        Ok(Self {
            demuxer,
            source,
            buffer: VecDeque::new(),
            cfg,
            playhead_us: 0,
            emitted: 0,
            read_offset: 0,
            first_buffered_us: None,
            last_buffered_end_us: 0,
        })
    }

    /// 轨道与总览信息（合并器/宿主都会先读它）。
    pub fn info(&self) -> MediaInfo {
        self.demuxer.info()
    }

    /// 容器名字。
    pub fn container(&self) -> &'static str {
        self.demuxer.container()
    }

    /// 总时长（微秒）；未知返回 None。
    pub fn duration_us(&self) -> Option<i64> {
        self.demuxer.duration_us()
    }

    /// 源的总字节数（未知返回 None；边下边播时可能是"下载引擎声明的最终大小"）。
    pub fn source_len(&self) -> Option<u64> {
        self.source.len()
    }

    /// 取下一个采样；`Ok(None)` = 真的到头了。
    ///
    /// 会先把缓冲区里已有的吐出去；缓冲区空了才继续解。
    pub fn next_packet(&mut self) -> Result<Option<Packet>> {
        if let Some(p) = self.buffer.pop_front() {
            self.emitted += 1;
            self.playhead_us = self.playhead_us.max(p.pts_us_approx(&self.info()));
            self.refresh_buffer_window();
            return Ok(Some(p));
        }
        self.fill_buffer()?;
        if let Some(p) = self.buffer.pop_front() {
            self.emitted += 1;
            self.playhead_us = self.playhead_us.max(p.pts_us_approx(&self.info()));
            self.refresh_buffer_window();
            return Ok(Some(p));
        }
        Ok(None)
    }

    /// 缓冲窗口的首尾：取走一条之后要重新对齐（否则"缓冲时长"会一直虚高）。
    fn refresh_buffer_window(&mut self) {
        match (self.buffer.front(), self.buffer.back()) {
            (Some(front), Some(back)) => {
                self.first_buffered_us = Some(front.pts_us_approx(&self.info()));
                let (_, end) = self.span_of(back);
                self.last_buffered_end_us = end;
            }
            _ => {
                self.first_buffered_us = None;
                self.last_buffered_end_us = 0;
            }
        }
    }

    /// 跳到 `us` 附近的关键帧，返回实际落点（微秒）。清空缓冲。
    pub fn seek_us(&mut self, us: i64) -> Result<i64> {
        let landing = self.demuxer.seek(us)?;
        self.buffer.clear();
        self.first_buffered_us = None;
        self.last_buffered_end_us = 0;
        self.playhead_us = landing;
        Ok(landing)
    }

    /// 播放头提示（宿主转达给下载引擎）。
    pub fn playhead_hint(&self) -> PlayheadHint {
        PlayheadHint { offset_bytes: self.read_offset, time_us: self.playhead_us }
    }

    pub fn stats(&self) -> PlayerStats {
        PlayerStats {
            emitted_packets: self.emitted,
            buffered_packets: self.buffer.len(),
            buffered_us: self.buffered_us(),
            playhead_us: self.playhead_us,
            source_read_offset: self.read_offset,
        }
    }

    /// 把缓冲填到上限（或解到末尾）。
    fn fill_buffer(&mut self) -> Result<()> {
        loop {
            if self.buffer.len() >= self.cfg.max_buffer_packets {
                return Ok(());
            }
            // 缓冲时长**增量维护**（见 `buffered_us` 字段）：每次重新累加整个缓冲
            // 会让填充变成 O(n²) —— 4096 条缓冲、每条都扫一遍是几百万次无谓加法
            if self.buffered_us() >= self.cfg.max_buffer_us {
                return Ok(());
            }
            match self.demuxer.next_packet()? {
                Some(p) => {
                    // 读位置随"解出来的数据量"前进：下载引擎只需要一个单调信号
                    self.read_offset = self.read_offset.saturating_add(p.data.len() as u64);
                    let (start, end) = self.span_of(&p);
                    if self.first_buffered_us.is_none() {
                        self.first_buffered_us = Some(start);
                    }
                    self.last_buffered_end_us = self.last_buffered_end_us.max(end);
                    self.buffer.push_back(p);
                }
                None => return Ok(()),
            }
        }
    }

    /// 一条采样覆盖的时间区间（微秒）。
    fn span_of(&self, p: &Packet) -> (i64, i64) {
        let start = p.pts_us_approx(&self.info());
        let end = match self.info().tracks.iter().find(|t| t.id == p.track_id) {
            Some(t) => t.pts_to_us(p.pts + p.duration as i64),
            None => start,
        };
        (start, end)
    }

    /// 缓冲覆盖的媒体时长（微秒）。空缓冲为 0。
    fn buffered_us(&self) -> i64 {
        match self.first_buffered_us {
            Some(first) => (self.last_buffered_end_us - first).max(0),
            None => 0,
        }
    }
}

/// `Packet` 的显示时间（微秒）—— 需要知道轨道时基，所以单独实现一个辅助 trait。
trait PacketTime {
    fn pts_us_approx(&self, info: &MediaInfo) -> i64;
}

impl PacketTime for Packet {
    fn pts_us_approx(&self, info: &MediaInfo) -> i64 {
        match info.tracks.iter().find(|t| t.id == self.track_id) {
            Some(t) => t.pts_to_us(self.pts),
            // 找不到轨道（不该发生）：按 MPEG-TS 的 90kHz 兜底，至少单调
            None => self.pts * 1_000_000 / 90_000,
        }
    }
}

/// 打开失败时的统一错误（把"源打不开"与"容器认不出"分开，宿主好分类）。
pub fn open_error(e: MediaError, name: &str) -> MediaError {
    match e {
        MediaError::Container(m) => MediaError::Container(format!("{name}: {m}")),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use me_core::Cursor;

    /// 手拼一个最小 MP4（视频轨 + 3 帧），与 me-container 的测试同一套思路。
    fn tiny_mp4() -> Vec<u8> {
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
        let ftyp = {
            let mut b = b"isom".to_vec();
            b.extend_from_slice(&512u32.to_be_bytes());
            b.extend_from_slice(b"isomiso2avc1mp41");
            bx(b"ftyp", &b)
        };
        let mvhd = full(b"mvhd", 0, 0, &{
            let mut b = Vec::new();
            b.extend_from_slice(&0u32.to_be_bytes());
            b.extend_from_slice(&0u32.to_be_bytes());
            b.extend_from_slice(&1000u32.to_be_bytes());
            b.extend_from_slice(&3000u32.to_be_bytes());
            b.extend_from_slice(&[0u8; 80]);
            b
        });
        let tkhd = full(b"tkhd", 0, 7, &{
            let mut b = Vec::new();
            b.extend_from_slice(&0u32.to_be_bytes());
            b.extend_from_slice(&0u32.to_be_bytes());
            b.extend_from_slice(&1u32.to_be_bytes());
            b.extend_from_slice(&0u32.to_be_bytes());
            b.extend_from_slice(&0u32.to_be_bytes());
            b.extend_from_slice(&[0u8; 8]);
            b.extend_from_slice(&[0u8; 8]);
            b.extend_from_slice(&[0u8; 36]);
            b.extend_from_slice(&(640u32 << 16).to_be_bytes());
            b.extend_from_slice(&(360u32 << 16).to_be_bytes());
            b
        });
        let mdhd = full(b"mdhd", 0, 0, &{
            let mut b = Vec::new();
            b.extend_from_slice(&0u32.to_be_bytes());
            b.extend_from_slice(&0u32.to_be_bytes());
            b.extend_from_slice(&30_000u32.to_be_bytes());
            b.extend_from_slice(&3000u32.to_be_bytes());
            b.extend_from_slice(&[0u8; 4]);
            b
        });
        let hdlr = full(b"hdlr", 0, 0, &{
            let mut b = vec![0u8; 4];
            b.extend_from_slice(b"vide");
            b.extend_from_slice(&[0u8; 12]);
            b.push(0);
            b
        });
        // 一个能解析出参数集的 avcC
        let avcc = {
            let mut b = vec![1u8, 0x42, 0x00, 0x1E, 0xFF, 1];
            b.extend_from_slice(&3u16.to_be_bytes());
            b.extend_from_slice(&[0x67, 0x42, 0x00]);
            b.push(1);
            b.extend_from_slice(&2u16.to_be_bytes());
            b.extend_from_slice(&[0x68, 0xCE]);
            b
        };
        let avc1 = {
            let mut b = vec![0u8; 6];
            b.extend_from_slice(&1u16.to_be_bytes());
            b.extend_from_slice(&[0u8; 16]);
            b.extend_from_slice(&640u16.to_be_bytes());
            b.extend_from_slice(&360u16.to_be_bytes());
            b.extend_from_slice(&0x0048_0000u32.to_be_bytes());
            b.extend_from_slice(&0x0048_0000u32.to_be_bytes());
            b.extend_from_slice(&0u32.to_be_bytes());
            b.extend_from_slice(&1u16.to_be_bytes());
            b.extend_from_slice(&[0u8; 32]);
            b.extend_from_slice(&0x0018u16.to_be_bytes());
            b.extend_from_slice(&0xFFFFu16.to_be_bytes());
            b.extend_from_slice(&bx(b"avcC", &avcc));
            bx(b"avc1", &b)
        };
        let stsd = full(b"stsd", 0, 0, &{
            let mut b = 1u32.to_be_bytes().to_vec();
            b.extend_from_slice(&avc1);
            b
        });
        let stts = full(b"stts", 0, 0, &{
            let mut b = 1u32.to_be_bytes().to_vec();
            b.extend_from_slice(&3u32.to_be_bytes());
            b.extend_from_slice(&1000u32.to_be_bytes());
            b
        });
        let stsc = full(b"stsc", 0, 0, &{
            let mut b = 1u32.to_be_bytes().to_vec();
            b.extend_from_slice(&1u32.to_be_bytes());
            b.extend_from_slice(&3u32.to_be_bytes());
            b.extend_from_slice(&1u32.to_be_bytes());
            b
        });
        let stsz = full(b"stsz", 0, 0, &{
            let mut b = 0u32.to_be_bytes().to_vec();
            b.extend_from_slice(&3u32.to_be_bytes());
            for _ in 0..3 {
                b.extend_from_slice(&6u32.to_be_bytes());
            }
            b
        });
        let stco = full(b"stco", 0, 0, &{
            let mut b = 1u32.to_be_bytes().to_vec();
            b.extend_from_slice(&0u32.to_be_bytes()); // 稍后回填
            b
        });
        let stss = full(b"stss", 0, 0, &{
            let mut b = 1u32.to_be_bytes().to_vec();
            b.extend_from_slice(&1u32.to_be_bytes());
            b
        });
        let stbl = bx(b"stbl", &[stsd, stts, stsc, stsz, stco, stss].concat());
        let minf = bx(b"minf", &stbl);
        let mdia = bx(b"mdia", &[mdhd, hdlr, minf].concat());
        let trak = bx(b"trak", &[tkhd, mdia].concat());
        let moov = bx(b"moov", &[mvhd, trak].concat());

        // 三条采样（长度前缀布局）
        let mut mdat_body = Vec::new();
        for i in 0..3u8 {
            mdat_body.extend_from_slice(&2u32.to_be_bytes());
            mdat_body.push(if i == 0 { 0x65 } else { 0x41 });
            mdat_body.push(i);
        }
        let mdat = bx(b"mdat", &mdat_body);

        let head_len = ftyp.len() + moov.len();
        let mdat_payload = (head_len + 8) as u32;
        // 回填 stco
        let mut moov_fixed = moov.clone();
        let pos = moov_fixed
            .windows(4)
            .position(|w| w == b"stco")
            .expect("要有 stco");
        let entry = pos + 4 + 4 + 4; // box头 + version/flags + entry_count
        moov_fixed[entry..entry + 4].copy_from_slice(&mdat_payload.to_be_bytes());
        let mut out = ftyp;
        out.extend_from_slice(&moov_fixed);
        out.extend_from_slice(&mdat);
        out
    }

    fn open_tiny() -> Player {
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(tiny_mp4()));
        Player::open(src, "mem", PlayerConfig::default()).unwrap()
    }

    #[test]
    fn opens_and_reports_tracks() {
        let p = open_tiny();
        assert_eq!(p.container(), "iso-bmff");
        let info = p.info();
        assert_eq!(info.tracks.len(), 1);
        assert_eq!(info.video().unwrap().width, 640);
        assert_eq!(p.duration_us(), Some(3_000_000));
    }

    #[test]
    fn reads_packets_in_order_until_eof() {
        let mut p = open_tiny();
        let mut pts = Vec::new();
        while let Some(pkt) = p.next_packet().unwrap() {
            pts.push(pkt.pts);
        }
        assert_eq!(pts, vec![0, 1000, 2000]);
        assert!(p.next_packet().unwrap().is_none(), "到头之后必须一直是 None");
        assert_eq!(p.stats().emitted_packets, 3, "统计要如实反映吐出去的包数");
    }

    #[test]
    fn buffer_is_bounded() {
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(tiny_mp4()));
        let cfg = PlayerConfig { max_buffer_us: 1, max_buffer_packets: 2 };
        let mut p = Player::open(src, "mem", cfg).unwrap();
        // 消费一条：内部会把缓冲填到上限（2 条）
        let _ = p.next_packet().unwrap().unwrap();
        assert!(p.stats().buffered_packets <= 2, "缓冲必须封顶，实际 {}", p.stats().buffered_packets);
    }

    #[test]
    fn seek_lands_on_a_keyframe_and_clears_the_buffer() {
        let mut p = open_tiny();
        let _ = p.next_packet().unwrap();
        let landing = p.seek_us(0).unwrap();
        assert_eq!(landing, 0);
        let pkt = p.next_packet().unwrap().unwrap();
        assert!(pkt.keyframe, "seek 之后第一条必须是关键帧");
    }

    #[test]
    fn playhead_hint_advances() {
        let mut p = open_tiny();
        assert!(p.source_len().unwrap_or(0) > 0, "源长度要能报出来（进度条要用）");
        let before = p.playhead_hint();
        while p.next_packet().unwrap().is_some() {}
        let after = p.playhead_hint();
        assert!(
            after.offset_bytes >= before.offset_bytes,
            "读位置必须单调前进（下载引擎靠它决定先下哪片）"
        );
        assert!(after.time_us >= before.time_us);
    }

    #[test]
    fn garbage_input_is_a_clear_error() {
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(vec![0x11u8; 64]));
        let e = Player::open(src, "mem", PlayerConfig::default());
        assert!(e.is_err());
    }
}
