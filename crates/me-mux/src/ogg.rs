//! Ogg（`.ogg` / `.opus`）封装：一条 Opus 逻辑流的页写入。
//!
//! ## 为什么值得有
//!
//! 引擎一直**读得进** Ogg Opus（`me-container/src/ogg.rs`），却**写不出来** ——
//! 用户想把视频里的 Opus 音轨单独导出成 `.opus` 时，只能在 `.mka` / `.mp4` 里绕。
//! 补上写侧之后读写两边才对称（这是本仓库对格式缺口的一贯判据）。
//!
//! ## 几个必须做对、错了就"能生成但播不了"的点
//!
//! 1. **页 CRC 是 Ogg 自己的 CRC-32**：多项式 `0x04C11DB7`、**不反射**、初值 0、
//!    不取反（与 zlib 的 CRC-32 不是一回事）。CRC 字段自身按 0 参与计算。
//!    实现复用读侧的 [`me_container::ogg::crc32_ogg`] —— 只有一份，别另写。
//! 2. **lacing 决定包边界**：每个包按 `255` 的整倍数切段，**末段必须 < 255**；
//!    包长正好是 255 的整数倍时要补一个 `0` 段收尾。少这个 0，解码器会把这个包
//!    和下一个包看成一个。
//! 3. **granule 是"这一页最后一个包结束时的累计 48 kHz 采样数"**，从 0 起累计
//!    （Opus 的解码采样率恒为 48 kHz，与 `OpusHead` 里的"输入采样率"无关）。
//!    包的时长只看 TOC 字节（[`me_container::ogg::toc_duration`]），不看
//!    `Packet::duration` —— 后者在跨容器重封装时可能已被换算取整。
//! 4. **页序与标志**：第一页 `BOS`（0x02）装 `OpusHead`，第二页装 `OpusTags`，
//!    最后一页必须带 `EOS`（0x04）。`page_sequence` 从 0 连续递增。
//! 5. **最后一页要等 `finish` 才写**（[`OggOpusMuxer::ready`]）：EOS 只能落在
//!    "写完才知道是最后一页"的那一页上，所以永远把上一页攒着、下一批数据到了
//!    再落盘。内存里最多压两页（≈8 KiB），与产物大小无关。

use std::io::Write;

use me_core::{CodecId, MediaError, MediaKind, Packet, Result, Track};

use crate::sink::Sink;

/// Opus 的解码采样率（也是 granule 的单位）。只给测试算期望值用。
#[cfg(test)]
const OPUS_RATE: u64 = 48_000;

/// 一页攒到这么多字节就封页。4 KiB 是常见取值（libogg 默认也是 4 KiB 量级）。
const PAGE_TARGET: usize = 4096;

/// 一页正文的上限：段表最多 255 项、每项最大 255 字节。
const MAX_PAGE_BODY: usize = 255 * 255;

/// 逻辑流的序列号。单流文件里取什么都行，固定值让产物可复现。
const SERIAL: u32 = 0x5A17_0C2E;

/// 正在组的一页。
#[derive(Default)]
struct Page {
    packets: Vec<Vec<u8>>,
    body_len: usize,
    /// 这一页最后一个包结束时的累计采样数（写页头 granule 用）。
    granule: u64,
}

/// Ogg Opus 封装器（API 与 [`crate::Mp4Muxer`] 对齐）。
pub struct OggOpusMuxer {
    sink: Sink,
    /// `OpusHead`（extradata 原样，19 字节起）。
    head: Vec<u8>,
    seq: u32,
    /// 正在攒的页。
    cur: Page,
    /// 上一页：等下一批数据到了才写（把 EOS 留给真正最后一页）。
    ready: Option<Page>,
    /// 已收下的采样总数（48 kHz）。
    total_samples: u64,
    declared: bool,
    packets_written: u64,
}

impl Default for OggOpusMuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl OggOpusMuxer {
    /// 内存落点（测试与小产物）。
    pub fn new() -> Self {
        Self::with_sink(Sink::mem())
    }

    /// 直写目标：产物**不驻留内存**。
    pub fn with_writer(w: Box<dyn Write + Send>) -> Self {
        Self::with_sink(Sink::writer(w))
    }

    fn with_sink(sink: Sink) -> Self {
        Self {
            sink,
            head: Vec::new(),
            seq: 0,
            cur: Page::default(),
            ready: None,
            total_samples: 0,
            declared: false,
            packets_written: 0,
        }
    }

    /// 声明轨道：必须**恰好一条 Opus 音频轨**，并立刻写出两页头。
    pub fn set_tracks(&mut self, tracks: &[Track]) -> Result<()> {
        let t = match tracks {
            [one] => one,
            [] => return Err(MediaError::Container("Ogg 至少需要一条轨道".into())),
            many => {
                return Err(MediaError::Unsupported(format!(
                    "Ogg（`.ogg` / `.opus`）只能装一条 Opus 声音轨，这里给了 {} 条",
                    many.len()
                )))
            }
        };
        if t.kind != MediaKind::Audio || t.codec != CodecId::Opus {
            return Err(MediaError::Unsupported(format!(
                "Ogg 产物的轨道必须是 Opus 声音，这里是 {} / {:?}（引擎不转码）",
                t.codec.name(),
                t.kind
            )));
        }
        if t.extradata.is_empty() {
            return Err(MediaError::Container(
                "Opus 轨缺 OpusHead（没有它写不出第一页，也定不出 pre-skip）".into(),
            ));
        }
        // 用读侧的解析器校验并取出规范形态：写出去的必须是一个能读回来的 OpusHead。
        let head = me_container::ogg::parse_opus_head(&t.extradata)?;
        // 多声道映射族（family != 0）本引擎读不回来（解封装器明确拒绝），
        // 写出去就是"自己能生成、自己读不了"——按本仓库的对称性判据直接拒绝。
        if head.mapping_family != 0 {
            return Err(MediaError::Unsupported(format!(
                "OpusHead 的声道映射族 {}（多声道/环绕），引擎只支持单声道与立体声",
                head.mapping_family
            )));
        }
        self.head = head.raw;
        self.declared = true;
        // 两页头：BOS(OpusHead) + OpusTags。granule 都是 0（这一页没有音频包）。
        self.write_page(0x02, 0, &[self.head.clone()])?;
        self.write_page(0, 0, &[opus_tags()])?;
        Ok(())
    }

    /// 写一条采样（一个 Ogg 包 = 一帧完整的 Opus 包）。
    pub fn write_packet(&mut self, p: &Packet) -> Result<()> {
        if !self.declared {
            return Err(MediaError::Container("写采样之前必须先 set_tracks".into()));
        }
        // 时长以 TOC 为准：跨容器重封装时 `duration` 可能已被换算取整。
        let (samples, _) = me_container::ogg::toc_duration(&p.data).ok_or_else(|| {
            MediaError::Bitstream(format!(
                "轨道 {} 的采样不是合法的 Opus 包（TOC 解不出时长，前 2 字节 {:02X?}）",
                p.track_id,
                &p.data[..p.data.len().min(2)]
            ))
        })?;
        // 一个包必须能装进一页（255 段 × 255 字节）。
        if p.data.len() > MAX_PAGE_BODY - 255 {
            return Err(MediaError::Unsupported(format!(
                "Opus 包 {} 字节，超过 Ogg 单页能装下的上限（{} 字节）",
                p.data.len(),
                MAX_PAGE_BODY - 255
            )));
        }
        self.total_samples += samples as u64;
        self.cur.body_len += p.data.len();
        self.cur.packets.push(p.data.clone());
        self.cur.granule = self.total_samples;
        self.packets_written += 1;

        if self.cur.body_len >= PAGE_TARGET {
            // 上一页现在可以落盘了；当前页接替"待写"的位置。
            if let Some(prev) = self.ready.take() {
                self.write_page(0, prev.granule, &prev.packets)?;
            }
            self.ready = Some(std::mem::take(&mut self.cur));
        }
        Ok(())
    }

    /// 收尾：把压着的最后一页写上 EOS，返回整份文件（内存模式）。
    pub fn finish(&mut self) -> Result<Vec<u8>> {
        if let Some(prev) = self.ready.take() {
            let flags = if self.cur.packets.is_empty() { 0x04 } else { 0 };
            self.write_page(flags, prev.granule, &prev.packets)?;
        }
        if !self.cur.packets.is_empty() {
            let last = std::mem::take(&mut self.cur);
            self.write_page(0x04, last.granule, &last.packets)?;
        }
        self.sink.flush()?;
        Ok(self.sink.take_bytes())
    }

    /// 已经写出的字节数（进度上报用）。
    pub fn bytes_written(&self) -> u64 {
        self.sink.bytes_written()
    }

    pub fn packets_written(&self) -> u64 {
        self.packets_written
    }

    /// 写一整页。`granule` = 本页最后一个包结束时的累计采样数。
    fn write_page(&mut self, flags: u8, granule: u64, packets: &[Vec<u8>]) -> Result<()> {
        let mut table = Vec::new();
        let mut body = Vec::new();
        for p in packets {
            let mut left = p.len();
            loop {
                let seg = left.min(255);
                table.push(seg as u8);
                if seg < 255 {
                    break;
                }
                left -= 255;
                // 包长正好是 255 的整数倍：补一个 0 段收尾，否则会和下一个包粘住
                if left == 0 {
                    table.push(0);
                    break;
                }
            }
            body.extend_from_slice(p);
        }
        if table.len() > 255 {
            return Err(MediaError::Unsupported(format!(
                "一页最多 255 个段，这一页需要 {} 个（包太大或攒得太多）",
                table.len()
            )));
        }
        let mut page = Vec::with_capacity(27 + table.len() + body.len());
        page.extend_from_slice(b"OggS");
        page.push(0); // 版本
        page.push(flags);
        page.extend_from_slice(&granule.to_le_bytes());
        page.extend_from_slice(&SERIAL.to_le_bytes());
        page.extend_from_slice(&self.seq.to_le_bytes());
        page.extend_from_slice(&[0, 0, 0, 0]); // CRC 占位：计算时按 0 参与
        page.push(table.len() as u8);
        page.extend_from_slice(&table);
        page.extend_from_slice(&body);
        let crc = me_container::ogg::crc32_ogg(&page);
        page[22..26].copy_from_slice(&crc.to_le_bytes());
        self.sink.write_all(&page)?;
        self.seq = self.seq.wrapping_add(1);
        Ok(())
    }
}

/// `OpusTags` 包：厂商串 + 0 条用户注释。
fn opus_tags() -> Vec<u8> {
    let vendor = b"ZuvRust";
    let mut v = b"OpusTags".to_vec();
    v.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    v.extend_from_slice(vendor);
    v.extend_from_slice(&0u32.to_le_bytes());
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use me_core::Timebase;
    use me_container::ogg::{page_crc_ok, parse_page_header};
    use std::sync::Arc;

    fn head(pre_skip: u16) -> Vec<u8> {
        let mut v = b"OpusHead".to_vec();
        v.push(1); // version
        v.push(2); // channels
        v.extend_from_slice(&pre_skip.to_le_bytes());
        v.extend_from_slice(&(OPUS_RATE as u32).to_le_bytes());
        v.extend_from_slice(&0i16.to_le_bytes());
        v.push(0); // mapping family
        v
    }

    fn opus_track(pre_skip: u16) -> Track {
        let mut t = Track::new(1, MediaKind::Audio, CodecId::Opus, Timebase::new(OPUS_RATE as u32));
        t.sample_rate = OPUS_RATE as u32;
        t.channels = 2;
        t.extradata = head(pre_skip);
        t
    }

    /// 20 ms 的 Opus 包（TOC：config 1 = SILK 20 ms，c = 0 单帧）。
    fn packet(dts: i64) -> Packet {
        Packet::new(1, dts, dts, 960, vec![0x08, 0x11, 0x22, 0x33])
    }

    /// 逐页走一遍，返回（flags, granule, seq, 页长）。
    fn pages(data: &[u8]) -> Vec<(u8, Option<u64>, u32)> {
        let mut out = Vec::new();
        let mut pos = 0usize;
        while pos < data.len() {
            let h = parse_page_header(&data[pos..]).unwrap_or_else(|| panic!("第 {pos} 字节不是合法页头"));
            let len = h.page_len();
            assert!(pos + len <= data.len(), "页越长出文件");
            assert!(page_crc_ok(&data[pos..pos + len]), "第 {} 页 CRC 不对", out.len());
            out.push((h.flags, h.granule, h.page_sequence));
            pos += len;
        }
        out
    }

    fn open(data: Vec<u8>) -> me_container::ogg::OggOpusDemuxer {
        me_container::ogg::OggOpusDemuxer::new(
            Arc::new(me_core::Cursor::new(data)),
            "x.opus",
        )
        .expect("写出来的 Ogg 必须能被自己的解封装器打开")
    }

    #[test]
    fn headers_are_bos_tags_and_granule_counts_samples() {
        let mut m = OggOpusMuxer::new();
        m.set_tracks(&[opus_track(312)]).unwrap();
        for i in 0..10 {
            m.write_packet(&packet(i * 960)).unwrap();
        }
        let data = m.finish().unwrap();

        let p = pages(&data);
        assert!(p.len() >= 3, "至少三页（BOS + Tags + 音频）");
        assert_eq!(p[0].0 & 0x02, 0x02, "第一页必须是 BOS");
        assert_eq!(p[0].1, Some(0), "头页没有音频包，granule 必须是 0");
        assert_eq!(p[0].2, 0);
        assert_eq!(p[1].1, Some(0), "OpusTags 页 granule 也是 0");
        assert_eq!(p[1].2, 1);
        assert_eq!(p.last().unwrap().0 & 0x04, 0x04, "最后一页必须带 EOS");
        assert_eq!(p.last().unwrap().1, Some(9600), "末页 granule = 累计采样数");
        // 页序连续
        for (i, (_, _, seq)) in p.iter().enumerate() {
            assert_eq!(*seq, i as u32, "页序必须从 0 连续递增");
        }
        // 头两页的包就是 OpusHead / OpusTags
        let h = parse_page_header(&data).unwrap();
        assert!(data[h.header_len..h.page_len()].starts_with(b"OpusHead"));
    }

    #[test]
    fn roundtrips_through_our_own_demuxer() {
        let mut m = OggOpusMuxer::new();
        m.set_tracks(&[opus_track(312)]).unwrap();
        let n = 100;
        for i in 0..n {
            m.write_packet(&packet(i * 960)).unwrap();
        }
        let data = m.finish().unwrap();
        assert_eq!(m.packets_written(), n as u64);

        let mut d = open(data);
        let info = d.info();
        assert_eq!(info.tracks.len(), 1);
        assert_eq!(info.tracks[0].codec, CodecId::Opus);
        assert_eq!(info.tracks[0].sample_rate, 48_000);
        assert_eq!(info.tracks[0].channels, 2);
        assert_eq!(info.tracks[0].extradata, head(312), "OpusHead 要原样往返");

        let mut pts = Vec::new();
        while let Some(p) = d.next_packet().unwrap() {
            pts.push(p.pts);
        }
        assert_eq!(pts.len(), n as usize, "一条采样都不能少");
        assert_eq!(pts[0], 0, "时间轴必须从 0 起");
        assert!(pts.windows(2).all(|w| w[1] > w[0]), "pts 必须严格递增：{pts:?}");
        // 可播时长 = 累计采样 − pre-skip
        let us = info.duration_us.unwrap_or(0);
        let want = ((n as u64 * 960 - 312) * 1_000_000 / OPUS_RATE) as i64;
        assert_eq!(us, want, "时长口径必须与读侧一致");
    }

    #[test]
    fn pages_hold_many_packets_and_stay_under_the_lacing_limit() {
        // 40 个 200 字节的包 ≈ 8 KiB ⇒ 至少两页音频
        let mut m = OggOpusMuxer::new();
        m.set_tracks(&[opus_track(0)]).unwrap();
        for i in 0..200 {
            let mut p = packet(i * 960);
            p.data.resize(200, 0xAA);
            m.write_packet(&p).unwrap();
        }
        let data = m.finish().unwrap();
        let p = pages(&data);
        let audio_pages = p.len() - 2;
        assert!(audio_pages >= 2, "应当切成多页，实际 {audio_pages}");
    }

    #[test]
    fn rejects_wrong_tracks_and_bad_packets() {
        let mut video = Track::new(1, MediaKind::Video, CodecId::H264, Timebase::new(30_000));
        video.extradata = vec![0, 0, 0, 1, 0x67];
        let mut m = OggOpusMuxer::new();
        let err = m.set_tracks(&[video]).unwrap_err();
        assert!(format!("{err}").contains("Opus"), "{err}");

        let mut m = OggOpusMuxer::new();
        let err = m
            .set_tracks(&[opus_track(0), opus_track(0)])
            .expect_err("两条轨必须被拒");
        assert!(format!("{err}").contains("一条"), "{err}");

        let mut t = opus_track(0);
        t.extradata.clear();
        let mut m = OggOpusMuxer::new();
        let err = m.set_tracks(&[t]).unwrap_err();
        assert!(format!("{err}").contains("OpusHead"), "{err}");

        let mut m = OggOpusMuxer::new();
        m.set_tracks(&[opus_track(0)]).unwrap();
        let err = m.write_packet(&Packet::new(1, 0, 0, 0, vec![0x03])).unwrap_err();
        assert!(format!("{err}").contains("TOC"), "{err}");
    }
}
