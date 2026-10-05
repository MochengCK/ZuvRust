//! 分片 fMP4 封装（`ftyp + moov(mvex) + [moof + mdat]*`）。
//!
//! ## 为什么产出分片形态而不是普通 MP4
//!
//! - 分片是 HLS/DASH 与"边下边播"要的形态：一个分片一到就能播，不用等整片写完；
//! - 普通 MP4 要把所有采样的索引（`stbl`）攒在 `moov` 里，**必须等全部写完
//!   才能落盘**，中途断电就是一个废文件。
//!
//! ## 几个必须做对、错了就"能生成但播不了"的点
//!
//! 1. **视频采样必须是长度前缀布局**（`avcC` 的 `lengthSizeMinusOne` 说了宽度）。
//!    引擎内部约定视频是 Annex-B，转换在这里做。**AV1 是例外**：它的采样本来就是
//!    OBU 流（每个 OBU 自带 leb128 长度），没有 Annex-B 这种形态，原样搬即可。
//! 2. **音频在 fMP4 里是裸帧**（不带 ADTS 头）。输入是 ADTS 时要剥掉。
//! 3. **`tfdt` 跨分片必须连续递增**。它是"这个分片从媒体的哪一刻开始"，
//!    写错会让播放器在切分片时跳时间（音画错开、进度条乱跳）。
//! 4. **`trun.data_offset` 的基准**（ISO 14496-12 §8.8.8）：写了
//!    `tfhd.base_data_offset` 就用它；否则（`default-base-is-moof`，或本 moof 的
//!    **第一个** traf）基准是**这个 moof 的起点**。所以我们写
//!    `default-base-is-moof`，偏移写成 `moof 长度 + 8`，正好指向紧随其后的
//!    mdat 载荷 —— 这也是 B 站 / ffmpeg 的写法，双方互相都读得对。
//!    把它当成"相对 moof 末尾"会让**每个分片的数据整整错位一个 moof 长度**：
//!    自产自销时看不出来（读写两边一致），一读别人的文件就全是垃圾。
//! 5. **`stsd` 里的 `esds` 长度字段用变长编码**（1~4 字节，最高位表示续接）。
//!    写死 1 字节在描述符稍大时就溢出，解析器读到的是一段错位的垃圾。
//! 6. **Opus 的配置盒是 `dOps`**（`OpusHead` 的另一种字节形态：大端、没有魔数，
//!    而且**不是 FullBox**）。引擎内部统一用 `OpusHead` 形态，转换在
//!    [`me_container::opus_config`]；Opus 的采样时长必须是 **48 kHz 刻度**，
//!    所以时基不是 48000 时这里直接报错，而不是写出一条越播越漂的时间轴。
//! 7. **VP9 的采样不做任何转换**（与 AV1 一样）：它没有起始码/长度前缀可言，
//!    `.webm` 与 fMP4 里的字节完全一致，走 H.264/H.265 那条路会以
//!    "找不到起始码"报错。

use std::io::Write;

use me_core::{ByteWriter, CodecId, MediaError, MediaKind, Packet, Result, Timebase, Track};

use crate::convert;
use crate::sink::Sink;

/// 一个分片的目标时长（微秒）。2 秒是 HLS/DASH 的常见取值。
pub const DEFAULT_FRAGMENT_US: i64 = 2_000_000;

/// 一条轨道攒着的一条采样。
#[derive(Debug, Clone)]
struct Sample {
    data: Vec<u8>,
    duration: u32,
    /// 合成偏移（`pts − dts`），B 帧重排用。
    cts: i32,
    keyframe: bool,
}

/// 一条轨道的封装状态。
#[derive(Debug)]
struct EsTrack {
    track: Track,
    timescale: u32,
    nal_len: u8,
    /// 本分片已攒的采样。
    samples: Vec<Sample>,
    /// 与 `samples` 一一对应的 dts（补时长要用）。
    sample_dts: Vec<i64>,
    /// 本分片第一条采样的 dts（写 `tfdt` 用）。
    frag_start_dts: Option<i64>,
    /// 本分片最后一条采样的结束时间（刻度）。分片边界按它算。
    frag_end_dts: i64,
    /// 上一片结束时"下一条采样"的 dts：用来检查调用方有没有把时间倒回去。
    next_dts: Option<i64>,
    /// 上一条采样的 dts（用来给"时长为 0"的采样补时长）。
    last_dts: Option<i64>,
    /// 最近一次见到的非零时长（给分片最后一条采样兜底）。
    last_duration: u32,
}

impl EsTrack {
    fn is_video(&self) -> bool {
        matches!(self.track.kind, MediaKind::Video)
    }

    /// 新采样到来时**只补上一条**的时长。
    ///
    /// 为什么不是每次全量推导：刚 push 进来的那一条还不知道自己的时长，
    /// 全量推导会给它填上兜底值（1），而下一次推送时它已经有"非零时长"，
    /// 就再也不会被正确修正 —— 实测后果是整条视频轨的时长全是 1 个刻度，
    /// 播放器的时间轴直接坏掉。
    fn fix_previous_duration(&mut self) {
        let n = self.samples.len();
        if n < 2 {
            return;
        }
        let i = n - 2;
        if self.samples[i].duration == 0 {
            let d = self.sample_dts[n - 1] - self.sample_dts[i];
            if d > 0 {
                self.samples[i].duration = d as u32;
            }
        }
        if self.samples[i].duration != 0 {
            self.last_duration = self.samples[i].duration;
        }
    }

    /// 落盘前把还缺时长的采样补齐（主要是本分片的最后一条）。
    ///
    /// 视频采样常常没有时长（TS 里的视频 PES 就没有），而 fMP4 的 `trun`/`stts`
    /// 必须给出每条的时长 —— 不补的话播放器的时间轴是坏的（时长 0、进度条不动）。
    fn infer_durations(&mut self) {
        for i in 0..self.samples.len() {
            if self.samples[i].duration != 0 {
                self.last_duration = self.samples[i].duration;
                continue;
            }
            // 后面还有采样 → 用"下一条的 dts − 这一条的 dts"
            if i + 1 < self.samples.len() {
                let d = self.sample_dts[i + 1] - self.sample_dts[i];
                if d > 0 {
                    self.samples[i].duration = d as u32;
                    self.last_duration = self.samples[i].duration;
                    continue;
                }
            }
            // 最后一条（或时间戳没前进）：沿用上一条的时长；
            // 一条参考都没有时给 1 个刻度 —— 0 会让时间轴退化成 0
            self.samples[i].duration = if self.last_duration > 0 { self.last_duration } else { 1 };
            self.last_duration = self.samples[i].duration;
        }
    }
}

/// 分片 fMP4 封装器。
pub struct Mp4Muxer {
    tracks: Vec<EsTrack>,
    /// 产物落点：内存（测试）或直写目标（生产）。
    ///
    /// 为什么不是 `Vec<u8>`：分片 fMP4 的每个 `moof+mdat` 都是自包含的，
    /// 大小在写入时就定了，**没有任何字段需要回头改**，所以可以边写边落盘。
    /// 攒在内存里会让内存占用与产物大小成正比（2 GB 的视频要 2 GB 内存）。
    sink: Sink,
    fragment_us: i64,
    /// 已写出的分片数（也是 `mfhd` 的序号）。
    fragment_seq: u32,
    /// **时间轴基准（微秒）**。
    ///
    /// - `None`（默认）：由第一条写入的采样决定（合并时每一段都从头归零）；
    /// - `Some`：由 [`Mp4Muxer::set_timeline_base_us`] 给定 —— 播放要的是
    ///   **绝对时间轴**（seek 到 300 秒之后，片段的时间戳就该从 300 秒开始，
    ///   否则播放器的进度条与 `currentTime` 全是错的）。
    base_us: Option<i64>,
    /// `ftyp + moov` 是否已经写出。
    header_written: bool,
    packets_written: u64,
}

impl Default for Mp4Muxer {
    fn default() -> Self {
        Self::new()
    }
}

impl Mp4Muxer {
    /// 内存落点（测试与小产物）。
    pub fn new() -> Self {
        Self::with_sink(Sink::mem())
    }

    /// 直写目标：产物**不驻留内存**（大文件合并走这条）。
    ///
    /// 调用方负责包缓冲（`BufWriter`），因为"要不要缓冲、缓冲多大"取决于
    /// 目标的性质，封装器不该替它决定。
    pub fn with_writer(w: Box<dyn Write + Send>) -> Self {
        Self::with_sink(Sink::writer(w))
    }

    fn with_sink(sink: Sink) -> Self {
        Self {
            tracks: Vec::new(),
            sink,
            fragment_us: DEFAULT_FRAGMENT_US,
            fragment_seq: 0,
            base_us: None,
            header_written: false,
            packets_written: 0,
        }
    }

    /// 设置分片目标时长（微秒）。
    pub fn set_fragment_duration_us(&mut self, us: i64) {
        self.fragment_us = us.max(1);
    }

    /// 固定时间轴基准（微秒）：`0` = 从头归零，`landing_us` = 保留绝对时间。
    ///
    /// 播放（`stream`）要用绝对值：seek 到 300 秒后片段的时间戳从 300 秒开始，
    /// 播放器的 `currentTime` 与进度条才是媒体真实位置。
    pub fn set_timeline_base_us(&mut self, us: i64) {
        self.base_us = Some(us);
    }

    /// 已经落盘的分片数（流式播放靠它判断"该把管道刷出去"了）。
    pub fn fragments_written(&self) -> u32 {
        self.fragment_seq
    }

    /// 把已经写出的字节刷到目标。
    ///
    /// 流式播放（`stream`）每凑够一个分片就必须调它：产物写向的是**管道**，
    /// 而管道那一头是等着喂数据的播放器 —— 不刷的话分片会留在用户态缓冲里，
    /// 播放器那边表现为"一直没有数据"（而进程其实在正常干活，最难查的一类问题）。
    pub fn flush_output(&mut self) -> Result<()> {
        self.sink.flush()
    }

    /// 声明轨道并写出 `ftyp + moov`。
    pub fn set_tracks(&mut self, tracks: &[Track]) -> Result<()> {
        if tracks.is_empty() {
            return Err(MediaError::Container("fMP4 至少需要一条轨道".into()));
        }
        self.tracks.clear();
        for t in tracks {
            let timescale = t.timebase.ticks_per_second.max(1);
            let nal_len = if matches!(t.kind, MediaKind::Video) && !t.extradata.is_empty() {
                convert::nal_length_size(&t.codec, &t.extradata)
            } else {
                4
            };
            // 视频必须能给出 avcC / hvcC / av1C / vpcC —— 给不出就是"解不出来"，
            // 当场报错比产出黑屏强
            if matches!(t.kind, MediaKind::Video)
                && !matches!(t.codec, CodecId::H264 | CodecId::H265 | CodecId::Av1 | CodecId::Vp9)
            {
                return Err(MediaError::Unsupported(format!(
                    "fMP4 封装暂不支持 {} 视频（只有 H.264 / H.265 / AV1 / VP9 有 avcC/hvcC/av1C/vpcC 形态）",
                    t.codec.name()
                )));
            }
            self.tracks.push(EsTrack {
                track: t.clone(),
                timescale,
                nal_len,
                samples: Vec::new(),
                sample_dts: Vec::new(),
                frag_start_dts: None,
                frag_end_dts: 0,
                next_dts: None,
                last_dts: None,
                last_duration: 0,
            });
        }
        self.write_header()?;
        Ok(())
    }

    /// 写一条采样（攒够一个分片就自动落盘）。
    ///
    /// **起点归零**：第一条采样建立基准，之后所有时间戳都按它平移。
    /// 为什么在这里做：解析层如实报告的 dts 可能是负的（MP4 的编辑列表会平移出
    /// 负值），而容器的 `tfdt` 是无符号的。整条流一起平移能保住采样间距与
    /// 音视频相对偏移；在解析层夹取会让前几帧的时间戳撞在一起。
    pub fn write_packet(&mut self, p: &Packet) -> Result<()> {
        if self.tracks.is_empty() {
            return Err(MediaError::Container("写采样之前必须先 set_tracks".into()));
        }
        let idx = self
            .tracks
            .iter()
            .position(|t| t.track.id == p.track_id)
            .ok_or_else(|| MediaError::Container(format!("轨道 {} 不在 moov 里", p.track_id)))?;
        let tb = self.tracks[idx].track.timebase;
        // 时间轴基准**按微秒存**，再换算回本轨的刻度。
        //
        // 为什么不能直接存某个轨的刻度：各轨时基不同（fMP4 视频常见 15360、
        // 音频是采样率），拿视频轨的刻度去减音频轨的 dts 是**在减两个不同单位**。
        // 起点恰好都是 0 时看不出来，一旦有偏移（编辑列表 / seek 到中间）就会
        // 让音视频错开一段 —— 这类错误在播放时表现为"声音比画面早/晚一点"，极难定位。
        let base = match self.base_us {
            Some(us) => tb.us_to_ticks(us),
            None => {
                let us = tb.ticks_to_us(p.dts);
                self.base_us = Some(us);
                tb.us_to_ticks(us)
            }
        };
        let dts = p.dts - base;
        let pts = p.pts - base;

        // 时间**严格后退**会让 tfdt 乱掉、播放器跳时间：当场报错，别产出坏文件。
        // 注意判据是"后退"而不是"重叠"：编辑列表平移后的前几帧允许时间戳相同
        // （真实文件就是这样），要求严格递增会把合法文件判成坏的。
        if let Some(prev) = self.tracks[idx].last_dts {
            if dts < prev {
                return Err(MediaError::Container(format!(
                    "轨道 {} 的时间戳倒流了（{} < {}）：调用方必须先交织好再写",
                    p.track_id, dts, prev
                )));
            }
        }

        let data = self.convert_sample(idx, p)?;
        let cts = (pts - dts).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        let t = &mut self.tracks[idx];
        if t.frag_start_dts.is_none() {
            t.frag_start_dts = Some(dts);
        }
        t.samples.push(Sample { data, duration: p.duration, cts, keyframe: p.keyframe });
        t.sample_dts.push(dts);
        let end = dts + p.duration as i64;
        t.frag_end_dts = t.frag_end_dts.max(end).max(dts);
        t.next_dts = Some(end);
        t.last_dts = Some(dts);
        // 补上一条的时长（视频常常没有时长，见 `fix_previous_duration`）
        t.fix_previous_duration();
        self.packets_written += 1;

        // 到分片边界就落盘。判据用**时间跨度**而不是"时长之和"：
        // 视频采样的时长可能是补出来的、甚至补不出来（首片只有一条），
        // 拿跨度判断在任何情况下都成立。
        let has_video = self.tracks.iter().any(|t| t.is_video());
        let should_flush = self
            .tracks
            .iter()
            .filter(|t| !has_video || t.is_video())
            .any(|t| {
                let start = t.frag_start_dts.unwrap_or(t.frag_end_dts);
                Timebase::new(t.timescale).ticks_to_us(t.frag_end_dts - start) >= self.fragment_us
            });
        if should_flush {
            self.flush_fragment()?;
        }
        Ok(())
    }

    /// 收尾：把最后半个分片写完，并（内存模式下）交出整份文件。
    ///
    /// 流式模式下返回值是**空的** —— 字节已经在目标里了；用
    /// [`Mp4Muxer::bytes_written`] 取总量。返回 `Result` 是因为落盘可能失败，
    /// 而"写不进去"绝不能当成成功（否则用户拿到一个截断的文件却以为合并好了）。
    pub fn finish(&mut self) -> Result<Vec<u8>> {
        self.flush_fragment()?;
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

    /// 采样数据形态转换（视频 → 长度前缀；AAC → 剥掉 ADTS）。
    fn convert_sample(&mut self, idx: usize, p: &Packet) -> Result<Vec<u8>> {
        let kind = self.tracks[idx].track.kind;
        let codec = self.tracks[idx].track.codec.clone();
        match kind {
            // AV1 没有 Annex-B 这种形态：fMP4 里的采样本来就是 OBU 流
            // （每个 OBU 自带 leb128 长度），与目标形态完全一致 —— **原样搬，不做转换**。
            // 走下面那条 H.264/H.265 的路会以"找不到起始码"报错。
            // VP9 同理：它的压缩帧没有任何起始码/长度前缀可言，源与目标字节完全一致
            // （`.webm` 与 fMP4 里的 VP9 采样是同一串字节）。
            MediaKind::Video if matches!(codec, CodecId::Av1 | CodecId::Vp9) => Ok(p.data.clone()),
            MediaKind::Video => {
                let nal_len = self.tracks[idx].nal_len;
                // 输入必须是 Annex-B（全引擎约定）。判断"是不是长度前缀"用
                // "按 Annex-B 拆出来有没有 NAL" —— 拆不出来说明形态不对，报错而不是瞎转
                if me_container::codec::split_annexb(&p.data).is_empty() {
                    return Err(MediaError::Bitstream(format!(
                        "轨道 {} 的视频采样不是 Annex-B（找不到起始码）",
                        p.track_id
                    )));
                }
                Ok(convert::annexb_to_avcc(&p.data, nal_len))
            }
            MediaKind::Audio if codec == CodecId::Aac => Ok(strip_adts(&p.data)),
            _ => Ok(p.data.clone()),
        }
    }

    /// 写出 `ftyp + moov`。
    fn write_header(&mut self) -> Result<()> {
        let mut w = ByteWriter::with_capacity(2048);
        // ftyp
        let at = box_open(&mut w, b"ftyp");
        w.bytes(b"iso5"); // major_brand：分片 MP4
        w.u32_be(512);
        w.bytes(b"iso5iso6mp41");
        box_close(&mut w, at);

        // moov
        let moov_at = box_open(&mut w, b"moov");
        write_mvhd(&mut w);
        for (i, t) in self.tracks.iter().enumerate() {
            write_trak(&mut w, t, (i + 1) as u32)?;
        }
        // mvex/trex：分片模式必须要有，否则播放器不知道分片里采样描述的下标
        let mvex_at = box_open(&mut w, b"mvex");
        for i in 0..self.tracks.len() {
            let trex_at = full_box_open(&mut w, b"trex", 0, 0);
            w.u32_be((i + 1) as u32); // track_ID
            w.u32_be(1); // default_sample_description_index
            w.u32_be(0); // default_sample_duration（分片里逐条写，这里给 0）
            w.u32_be(0); // default_sample_size
            w.u32_be(0); // default_sample_flags
            box_close(&mut w, trex_at);
        }
        box_close(&mut w, mvex_at);
        box_close(&mut w, moov_at);

        self.sink.write_all(&w.into_vec())?;
        self.header_written = true;
        Ok(())
    }

    /// 写一个分片（`moof` + `mdat`）。
    fn flush_fragment(&mut self) -> Result<()> {
        if !self.header_written {
            return Ok(());
        }
        if self.tracks.iter().all(|t| t.samples.is_empty()) {
            return Ok(());
        }
        self.fragment_seq += 1;
        // 落盘前把最后一条的时长补上（视频的时长常常是补出来的）
        for t in &mut self.tracks {
            t.infer_durations();
        }

        // `trun.data_offset` 的基准是**这个 moof 的起点**（我们写了
        // `default-base-is-moof`）。所以偏移 = moof 长度 + 8（mdat 头）
        // + 前面那些轨的数据长度 —— 数据紧接着 moof 之后，第一轨的偏移正好
        // 落在 mdat 的载荷起点上。
        //
        // `data_offset` 是**定长 4 字节**，它的数值不改变 moof 的总长度，
        // 所以可以先构建一次拿到长度、再用正确的偏移重建（两次都很便宜）。
        let sizes: Vec<i64> = self
            .tracks
            .iter()
            .map(|t| t.samples.iter().map(|s| s.data.len() as i64).sum())
            .collect();
        let moof_len = build_moof(&self.tracks, self.fragment_seq, |_| 0).len() as i64;
        let offsets: Vec<i64> = (0..self.tracks.len())
            .map(|i| moof_len + 8 + sizes[..i].iter().sum::<i64>())
            .collect();

        let moof = build_moof(&self.tracks, self.fragment_seq, |i| offsets[i]);
        debug_assert_eq!(
            moof.len() as i64,
            moof_len,
            "data_offset 定长，重建后 moof 长度不该变"
        );

        let mut mdat = ByteWriter::with_capacity(1024);
        let mdat_at = box_open(&mut mdat, b"mdat");
        for t in &self.tracks {
            for s in &t.samples {
                mdat.bytes(&s.data);
            }
        }
        box_close(&mut mdat, mdat_at);

        // 分片一写完就落盘：**内存占用只与一个分片有关，与产物总大小无关**。
        // `moof` 与 `mdat` 的大小在这里都已确定（没有任何回头要改的字段），
        // 所以不需要"先占位再回填"，也就不需要把产物攒在内存里。
        self.sink.write_all(&moof)?;
        self.sink.write_all(&mdat.into_vec())?;

        // 清空各轨缓冲，准备下一个分片（`frag_start_dts` 也要清：下一片的 tfdt
        // 由它自己的第一条采样决定）
        for t in &mut self.tracks {
            t.samples.clear();
            t.sample_dts.clear();
            t.frag_start_dts = None;
            t.frag_end_dts = 0;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// box 拼装
// ---------------------------------------------------------------------------

fn box_open(w: &mut ByteWriter, kind: &[u8; 4]) -> usize {
    let at = w.len();
    w.u32_be(0);
    w.bytes(kind);
    at
}

fn box_close(w: &mut ByteWriter, at: usize) {
    let size = (w.len() - at) as u32;
    let _ = w.patch_u32_be(at, size);
}

fn full_box_open(w: &mut ByteWriter, kind: &[u8; 4], version: u8, flags: u32) -> usize {
    let at = box_open(w, kind);
    w.u8(version);
    w.bytes(&flags.to_be_bytes()[1..]);
    at
}

/// 单位矩阵（3×3 的 16.16 定点，按 [a,b,u,c,d,v,x,y,w] 顺序存）。
const UNITY_MATRIX: [u32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000];

fn write_mvhd(w: &mut ByteWriter) {
    let at = full_box_open(w, b"mvhd", 0, 0);
    w.u32_be(0); // creation_time
    w.u32_be(0); // modification_time
    w.u32_be(1000); // timescale（电影层用 1000 就够，分片的时长由 tfdt 决定）
    w.u32_be(0); // duration：分片文件的 mvhd 时长必须是 0
    w.u32_be(0x0001_0000); // rate = 1.0
    w.u16_be(0x0100); // volume = 1.0
    w.u16_be(0); // reserved
    w.u32_be(0);
    w.u32_be(0);
    for v in UNITY_MATRIX {
        w.u32_be(v);
    }
    for _ in 0..6 {
        w.u32_be(0); // pre_defined
    }
    w.u32_be(0xFFFF_FFFF); // next_track_ID
    box_close(w, at);
}

fn write_trak(w: &mut ByteWriter, t: &EsTrack, track_id: u32) -> Result<()> {
    let trak_at = box_open(w, b"trak");
    write_tkhd(w, t, track_id);
    let mdia_at = box_open(w, b"mdia");
    write_mdhd(w, t);
    write_hdlr(w, t);
    let minf_at = box_open(w, b"minf");
    // 媒体头：视频要 vmhd、音频要 smhd，其它用 nmhd
    match t.track.kind {
        MediaKind::Video => {
            let at = full_box_open(w, b"vmhd", 0, 1);
            w.u16_be(0); // graphicsmode
            w.u16_be(0);
            w.u16_be(0);
            w.u16_be(0);
            box_close(w, at);
        }
        MediaKind::Audio => {
            let at = full_box_open(w, b"smhd", 0, 0);
            w.u16_be(0); // balance
            w.u16_be(0); // reserved
            box_close(w, at);
        }
        _ => {
            let at = full_box_open(w, b"nmhd", 0, 0);
            box_close(w, at);
        }
    }
    // dinf/dref：声明"数据就在这个文件里"
    let dinf_at = box_open(w, b"dinf");
    let dref_at = full_box_open(w, b"dref", 0, 0);
    w.u32_be(1); // entry_count
    let url_at = full_box_open(w, b"url ", 0, 1); // flags = 1：self-contained
    box_close(w, url_at);
    box_close(w, dref_at);
    box_close(w, dinf_at);

    let stbl_at = box_open(w, b"stbl");
    write_stsd(w, t)?;
    // 分片文件的 stbl 是空的：采样索引在 moof/trun 里
    for kind in [b"stts", b"stsc", b"stsz", b"stco"] {
        let at = full_box_open(w, kind, 0, 0);
        if kind == b"stsz" {
            w.u32_be(0); // sample_size
        }
        w.u32_be(0); // entry_count
        box_close(w, at);
    }
    box_close(w, stbl_at);
    box_close(w, minf_at);
    box_close(w, mdia_at);
    box_close(w, trak_at);
    Ok(())
}

fn write_tkhd(w: &mut ByteWriter, t: &EsTrack, track_id: u32) {
    let at = full_box_open(w, b"tkhd", 0, 0x7); // enabled | in_movie | in_preview
    w.u32_be(0); // creation_time
    w.u32_be(0); // modification_time
    w.u32_be(track_id);
    w.u32_be(0); // reserved
    w.u32_be(0); // duration：分片文件里为 0
    w.u32_be(0);
    w.u32_be(0);
    w.u16_be(0); // layer
    w.u16_be(0); // alternate_group
    // volume：音频轨要有值，视频轨为 0
    w.u16_be(if matches!(t.track.kind, MediaKind::Audio) { 0x0100 } else { 0 });
    w.u16_be(0); // reserved
    for v in UNITY_MATRIX {
        w.u32_be(v);
    }
    w.u32_be(t.track.width << 16); // 16.16 定点
    w.u32_be(t.track.height << 16);
    box_close(w, at);
}

fn write_mdhd(w: &mut ByteWriter, t: &EsTrack) {
    let at = full_box_open(w, b"mdhd", 0, 0);
    w.u32_be(0);
    w.u32_be(0);
    w.u32_be(t.timescale);
    w.u32_be(0); // duration：分片文件里为 0
    // 语言：3 个 5 位字母，各 -0x60
    let lang = t.track.language.as_deref().unwrap_or("und");
    let b = lang.as_bytes();
    let packed: u16 = if b.len() == 3 {
        (((b[0].saturating_sub(0x60)) as u16) << 10)
            | (((b[1].saturating_sub(0x60)) as u16) << 5)
            | ((b[2].saturating_sub(0x60)) as u16)
    } else {
        // "und"
        (21 << 10) | (14 << 5) | 4
    };
    w.u16_be(packed);
    w.u16_be(0); // pre_defined
    box_close(w, at);
}

fn write_hdlr(w: &mut ByteWriter, t: &EsTrack) {
    let at = full_box_open(w, b"hdlr", 0, 0);
    w.u32_be(0); // pre_defined
    w.bytes(match t.track.kind {
        MediaKind::Video => b"vide",
        MediaKind::Audio => b"soun",
        MediaKind::Subtitle => b"sbtl",
        MediaKind::Other => b"meta",
    });
    for _ in 0..3 {
        w.u32_be(0); // reserved
    }
    // name：以 0 结尾的字符串（要含结尾的 0）
    let name = b"ZuvRust";
    w.bytes(name);
    w.u8(0);
    box_close(w, at);
}

/// `stsd`：采样描述。这里决定了"用什么解码器解、参数集长什么样"。
fn write_stsd(w: &mut ByteWriter, t: &EsTrack) -> Result<()> {
    let stsd_at = full_box_open(w, b"stsd", 0, 0);
    w.u32_be(1); // entry_count
    match t.track.kind {
        MediaKind::Video => write_visual_entry(w, t)?,
        MediaKind::Audio => write_audio_entry(w, t)?,
        _ => {
            return Err(MediaError::Unsupported(format!(
                "轨道 {} 的类型（{:?}）暂时不能写进 fMP4",
                t.track.id, t.track.kind
            )))
        }
    }
    box_close(w, stsd_at);
    Ok(())
}

fn write_visual_entry(w: &mut ByteWriter, t: &EsTrack) -> Result<()> {
    let (format, extra_kind): (&[u8; 4], &[u8; 4]) = match &t.track.codec {
        CodecId::H264 => (b"avc1", b"avcC"),
        CodecId::H265 => (b"hvc1", b"hvcC"),
        // AV1（B 站等站点已经在用的编码）：采样条目是 `av01`，
        // 配置盒是 `av1C`（4 字节头 + sequence header OBU 的载荷）
        CodecId::Av1 => (b"av01", b"av1C"),
        // VP9：采样条目是 `vp09`，配置盒是 `vpcC`（FullBox，version 1）
        CodecId::Vp9 => (b"vp09", b"vpcC"),
        other => {
            return Err(MediaError::Unsupported(format!(
                "{} 暂时不能写进 fMP4 的视频轨",
                other.name()
            )))
        }
    };
    let entry_at = box_open(w, format);
    w.bytes(&[0u8; 6]); // reserved
    w.u16_be(1); // data_reference_index
    w.u16_be(0); // pre_defined
    w.u16_be(0); // reserved
    for _ in 0..3 {
        w.u32_be(0); // pre_defined
    }
    w.u16_be(t.track.width.min(0xFFFF) as u16);
    w.u16_be(t.track.height.min(0xFFFF) as u16);
    w.u32_be(0x0048_0000); // horizresolution 72dpi
    w.u32_be(0x0048_0000); // vertresolution
    w.u32_be(0); // reserved
    w.u16_be(1); // frame_count
    let mut name = [0u8; 32];
    let src = b"ZuvRust";
    name[..src.len()].copy_from_slice(src);
    name[0] = src.len() as u8; // 长度前缀
    w.bytes(&name);
    w.u16_be(0x0018); // depth = 24
    w.u16_be(0xFFFF); // pre_defined
    // avcC / hvcC / av1C / vpcC：
    //   · H.264/H.265 的 extradata 可能是 Annex-B 参数集，由 convert 转成配置盒载荷；
    //   · AV1 的 extradata **就是** av1C 的载荷本身（解封装给什么就写什么，
    //     它没有任何"另一种形态"可转）；
    //   · VP9 同理：`vpcC` 描述的是 profile / 位深 / 色彩三要素，与采样排列无关，
    //     从哪里读到的就写回哪里。
    let payload = if matches!(t.track.codec, CodecId::Av1 | CodecId::Vp9) {
        let who = if matches!(t.track.codec, CodecId::Av1) { "av1C" } else { "vpcC" };
        if t.track.extradata.is_empty() {
            return Err(MediaError::Bitstream(format!(
                "{} 轨 {} 缺少 {who} 配置（extradata 为空），写出来的画面解不了",
                t.track.codec.name(),
                t.track.id
            )));
        }
        if matches!(t.track.codec, CodecId::Vp9) {
            // vpcC 是定长结构（+ 可选的初始化数据）：version 0 是 9 字节，
            // version 1 是 11 字节（多一个 `codecInitializationDataSize`）。
            // 长度不够说明配置盒本身是坏的 —— 报错，别写一个解不了的文件。
            let vpc = &t.track.extradata;
            let min = match vpc[0] {
                0 => 9usize,
                1 => 11,
                other => {
                    return Err(MediaError::Unsupported(format!(
                        "VP9 轨 {} 的 vpcC 版本 {other} 不认识（本引擎认 0 与 1）",
                        t.track.id
                    )))
                }
            };
            if vpc.len() < min {
                return Err(MediaError::Bitstream(format!(
                    "VP9 轨 {} 的 vpcC 只有 {} 字节（版本 {} 至少要 {min} 字节），配置不完整",
                    t.track.id,
                    vpc.len(),
                    vpc[0]
                )));
            }
            // version 1 还声明了一段初始化数据：它的长度必须真的落在盒子里
            if vpc[0] == 1 {
                let n = u16::from_be_bytes([vpc[9], vpc[10]]) as usize;
                if vpc.len() < min + n {
                    return Err(MediaError::Bitstream(format!(
                        "VP9 轨 {} 的 vpcC 声明了 {n} 字节初始化数据，但盒子里只剩 {} 字节",
                        t.track.id,
                        vpc.len() - min
                    )));
                }
            }
        }
        t.track.extradata.clone()
    } else {
        convert::extradata_to_avcc(&t.track.codec, &t.track.extradata, t.nal_len)?
    };
    let sub_at = box_open(w, extra_kind);
    w.bytes(&payload);
    box_close(w, sub_at);
    box_close(w, entry_at);
    Ok(())
}

/// 音频采样条目：AAC / MP3（`mp4a` + `esds`）、FLAC（`fLaC` + `dfLa`）、
/// Opus（`Opus` + `dOps`）与 AC-3（`ac-3` + `dac3`）。
///
/// 为什么 FLAC 也得支持：B 站的"Hi-Res 无损音质"就是 FLAC 装在 fMP4 里。
/// 它与 AAC 的差别**只在配置盒** —— 采样数据都是"裸帧"，合并时原样搬、不重编码。
/// Opus 同理（`.opus` 与 fMP4 里的 Opus 包是同一串字节），它多两条自己的规矩：
///
/// - **时基必须是 48 kHz**：fMP4 里 Opus 的采样时长就是 48 kHz 刻度，输入轨时基
///   不是 48000 时 `duration` 会被按别的单位解释，时间轴整体漂；
/// - **配置盒是 `dOps` 而不是 `OpusHead`**：两者只差魔数与字节序，转换在
///   [`me_container::opus_config`]（引擎内部统一用 `OpusHead` 形态）。
///
/// **MP3 与 AAC 共用 `mp4a` 采样条目**，靠 `esds` 里的 objectTypeIndication 区分
/// （0x40 = MPEG-4 Audio、0x6B = MPEG-1 Audio）。MP3 **没有解码配置**
/// （参数都在每帧的帧头里），所以它的 esds 里不带 DecoderSpecificInfo ——
/// 这也是它与 AAC 在写侧唯一的实质差别。读侧（`me-container/src/mp4.rs`）本来
/// 就按 OTI 把 `mp4a` 分成 AAC/MP3/AC-3，写侧补齐后两边才对称。
///
/// **AC-3 的配置盒是 `dac3`**：TS 与裸流里没有它，参数只能从 syncframe 头里读
/// （见 [`me_container::ac3`]）；`mux` 层在写 moov 之前会把缺的补进 extradata。
fn write_audio_entry(w: &mut ByteWriter, t: &EsTrack) -> Result<()> {
    let entry_kind: &[u8; 4] = match &t.track.codec {
        CodecId::Aac | CodecId::Mp3 => b"mp4a",
        CodecId::Flac => b"fLaC",
        CodecId::Opus => b"Opus",
        CodecId::Ac3 => b"ac-3",
        other => {
            return Err(MediaError::Unsupported(format!(
                "{} 暂时不能写进 fMP4 的音频轨（目前支持 AAC / MP3 / FLAC / Opus / AC-3）",
                other.name()
            )))
        }
    };
    // MP3 没有解码配置：它的采样率/声道都在每一帧的帧头里，extradata 为空是**正常**的
    let needs_config = !matches!(t.track.codec, CodecId::Mp3);
    if needs_config && t.track.extradata.is_empty() {
        // 说清缺的是哪样东西：AAC 缺的是 AudioSpecificConfig、FLAC 缺的是 STREAMINFO
        let what = match &t.track.codec {
            CodecId::Aac => "AudioSpecificConfig",
            CodecId::Flac => "STREAMINFO（dfLa）",
            CodecId::Opus => "OpusHead（dOps 的来源）",
            CodecId::Ac3 => "dac3（应当由 mux 层从第一帧的 syncframe 里补出来）",
            _ => "解码配置",
        };
        return Err(MediaError::Bitstream(format!(
            "{} 轨 {} 缺少 {what}（extradata 为空），写出来解不了",
            t.track.codec.name(),
            t.track.id
        )));
    }
    // Opus 的配置盒要先解析出来：声道数**以它为准**（`AudioSampleEntry` 里那个
    // channelcount 在真实文件里时常与码流不一致），时基也要在这里就查掉。
    let opus_cfg = if matches!(t.track.codec, CodecId::Opus) {
        let cfg = me_container::opus_config::OpusConfig::parse_head(&t.track.extradata)
            .map_err(|e| {
                MediaError::Bitstream(format!(
                    "Opus 轨 {} 的 extradata 不是可用的 OpusHead：{e}",
                    t.track.id
                ))
            })?;
        if t.timescale != me_container::opus_config::OPUS_RATE {
            return Err(MediaError::Unsupported(format!(
                "Opus 轨 {} 的时基是 {}，但 fMP4 里 Opus 的采样时长必须是 48 kHz 刻度（48000）",
                t.track.id, t.timescale
            )));
        }
        Some(cfg)
    } else {
        None
    };

    let entry_at = box_open(w, entry_kind);
    w.bytes(&[0u8; 6]); // reserved
    w.u16_be(1); // data_reference_index
    w.u32_be(0); // reserved
    w.u32_be(0); // reserved
    let channels = opus_cfg.as_ref().map(|c| c.channels as u16).unwrap_or(t.track.channels);
    w.u16_be(channels.max(1));
    w.u16_be(16); // sample_size
    w.u16_be(0); // pre_defined
    w.u16_be(0); // reserved
    // 采样率是 16.16 定点，但**整数部分只有 16 位**：Hi-Res 的 96k / 192k 左移 16 位
    // 会直接溢出 u32，写出一个荒唐的采样率。装不下时写 0（语义是"看解码配置"），
    // 真实采样率在 esds / dfLa 里，解码器认那个。
    // Opus 例外：它的**解码采样率按定义恒为 48 kHz**（`OpusHead`/`dOps` 里那个
    // "输入采样率"只是编码前的元数据），所以这里直接写规范值，不看来源轨字段。
    let rate =
        if opus_cfg.is_some() { me_container::opus_config::OPUS_RATE } else { t.track.sample_rate };
    w.u32_be(if rate > 0xFFFF { 0 } else { rate << 16 });

    match &t.track.codec {
        // 同为 `mp4a`，靠 objectTypeIndication 区分：0x40 = MPEG-4 Audio、
        // 0x6B = MPEG-1 Audio（MP3）。MP3 不带 DecoderSpecificInfo。
        CodecId::Aac => write_esds(w, &t.track.extradata, 0x40)?,
        CodecId::Mp3 => write_esds(w, &[], 0x6B)?,
        CodecId::Flac => {
            // `dfLa` 是 FullBox，而解封装交给我们的 extradata **已经是它的载荷**
            // （含 version + flags）—— 所以这里用普通 `box_open`，不能再套一层 FullBox 头，
            // 否则版本字段会被写两遍（解析器读到的 STREAMINFO 就错位了）。
            let at = box_open(w, b"dfLa");
            w.bytes(&t.track.extradata);
            box_close(w, at);
        }
        CodecId::Opus => {
            // `dOps` 与 `dfLa` 一样是"载荷自带全部内容"的盒子。注意它**不是 FullBox**：
            // 第 0 字节就是规范里的 `Version`（值 0）。当成 FullBox 写会让后面每个字段
            // 整体错位 4 字节 —— 而错位后读出来的采样率/声道数都还是"像样的数"，
            // 属于那种"文件能生成、播放器解不了"的错误。
            let cfg = opus_cfg.as_ref().expect("上面已经解析过 Opus 的配置盒");
            let at = box_open(w, b"dOps");
            w.bytes(&cfg.to_dops());
            box_close(w, at);
        }
        CodecId::Ac3 => {
            // `dac3` 是定长 3 字节：长度不对说明 mux 层补进来的不是 dac3 载荷，
            // 直接拒绝 —— 写一个字段错位的配置盒出去，症状是"能生成但解不了"。
            if t.track.extradata.len() != 3 {
                return Err(MediaError::Bitstream(format!(
                    "AC-3 轨 {} 的 dac3 载荷应当是 3 字节，实际 {}",
                    t.track.id,
                    t.track.extradata.len()
                )));
            }
            let at = box_open(w, b"dac3");
            w.bytes(&t.track.extradata);
            box_close(w, at);
        }
        _ => unreachable!("上面已经拦过不支持的编码"),
    }
    box_close(w, entry_at);
    Ok(())
}

/// `esds`：把解码配置放进 DecoderSpecificInfo。
///
/// `object_type` 是 `objectTypeIndication`：0x40 = MPEG-4 Audio（AAC）、
/// 0x6B = MPEG-1 Audio（MP3）。MP3 的参数都在帧头里，`config` 传空时
/// **整个 DecoderSpecificInfo 都不写**（写一个长度 0 的 DSI 虽然合法，
/// 但会让一部分老播放器在解析描述符链时提前收尾）。
fn write_esds(w: &mut ByteWriter, asc: &[u8], object_type: u8) -> Result<()> {
    let esds_at = full_box_open(w, b"esds", 0, 0);
    // 描述符长度是**变长编码**（1~4 字节，最高位表示续接），所以先把内层拼进
    // Vec 里、拿到长度再写外层 —— 写死一字节在描述符稍大时就溢出
    let mut dsi = Vec::new();
    if !asc.is_empty() {
        dsi.push(0x05); // DecoderSpecificInfo
        write_desc_len(&mut dsi, asc.len());
        dsi.extend_from_slice(asc);
    }

    let mut dc = Vec::new();
    dc.push(0x04); // DecoderConfigDescriptor
    write_desc_len(&mut dc, 13 + dsi.len());
    dc.push(object_type); // objectTypeIndication
    dc.push(0x15); // streamType = audio
    dc.extend_from_slice(&[0, 0, 0]); // bufferSizeDB
    dc.extend_from_slice(&0u32.to_be_bytes()); // maxBitrate
    dc.extend_from_slice(&0u32.to_be_bytes()); // avgBitrate
    dc.extend_from_slice(&dsi);

    let mut es = Vec::new();
    es.extend_from_slice(&0u16.to_be_bytes()); // ES_ID
    es.push(0); // flags
    es.extend_from_slice(&dc);

    let mut payload = Vec::new();
    payload.push(0x03); // ES_Descriptor
    write_desc_len(&mut payload, es.len());
    payload.extend_from_slice(&es);
    w.bytes(&payload);
    box_close(w, esds_at);
    Ok(())
}

/// 描述符长度：1~4 字节变长编码（最高位表示"后面还有"）。
fn write_desc_len(out: &mut Vec<u8>, len: usize) {
    let mut bytes = [0u8; 4];
    let mut n = 0usize;
    let mut v = len;
    loop {
        bytes[n] = (v & 0x7F) as u8;
        n += 1;
        v >>= 7;
        if v == 0 || n == 4 {
            break;
        }
    }
    for i in (0..n).rev() {
        let more = if i == 0 { 0 } else { 0x80 };
        out.push(bytes[i] | more);
    }
}

// ---------------------------------------------------------------------------
// moof
// ---------------------------------------------------------------------------

/// 采样标志：`sample_is_non_sync_sample` 是 bit 16。
fn sample_flags(keyframe: bool, is_video: bool) -> u32 {
    if !is_video {
        return 0x0200_0000; // 音频每条都能独立解码
    }
    if keyframe {
        0x0200_0000 // depends_on = 2（不依赖别人）、non_sync = 0
    } else {
        0x0101_0000 // depends_on = 1 + non_sync = 1
    }
}
/// 拼 `moof`。`data_offset_of(i)` 给出第 i 条轨道的 `trun.data_offset`。
/// `tfhd` 的 `default-base-is-moof` 标志（采样数据以 moof 起点为基准）。
const TFHD_DEFAULT_BASE_IS_MOOF: u32 = 0x02_0000;

fn build_moof<F: Fn(usize) -> i64>(tracks: &[EsTrack], seq: u32, data_offset_of: F) -> Vec<u8> {
    let mut w = ByteWriter::with_capacity(1024);
    let moof_at = box_open(&mut w, b"moof");
    let mfhd_at = full_box_open(&mut w, b"mfhd", 0, 0);
    w.u32_be(seq);
    box_close(&mut w, mfhd_at);

    for (i, t) in tracks.iter().enumerate() {
        if t.samples.is_empty() {
            continue;
        }
        let traf_at = box_open(&mut w, b"traf");
        // tfhd：track_ID + default-base-is-moof
        //（声明"采样数据的基准是这个 moof 的起点"，与 trun 里的偏移配套）
        let tfhd_at = full_box_open(&mut w, b"tfhd", 0, TFHD_DEFAULT_BASE_IS_MOOF);
        w.u32_be((i + 1) as u32);
        box_close(&mut w, tfhd_at);
        // tfdt：这个分片从媒体的哪一刻开始（跨分片必须连续）
        let tfdt_at = full_box_open(&mut w, b"tfdt", 0, 0);
        w.u32_be(t.frag_start_dts.unwrap_or(0).max(0) as u32);
        box_close(&mut w, tfdt_at);
        // trun：逐条写时长/大小/标志/合成偏移
        let trun_at = full_box_open(&mut w, b"trun", 0, 0x0001 | 0x0100 | 0x0200 | 0x0400 | 0x0800);
        w.u32_be(t.samples.len() as u32);
        w.u32_be(data_offset_of(i) as u32);
        for s in &t.samples {
            w.u32_be(s.duration);
            w.u32_be(s.data.len() as u32);
            w.u32_be(sample_flags(s.keyframe, t.is_video()));
            w.u32_be(s.cts as u32);
        }
        box_close(&mut w, trun_at);
        box_close(&mut w, traf_at);
    }
    box_close(&mut w, moof_at);
    w.into_vec()
}

/// 剥掉 ADTS 头（fMP4 里的 AAC 是裸帧）。
fn strip_adts(data: &[u8]) -> Vec<u8> {
    if data.len() >= 7 && data[0] == 0xFF && (data[1] & 0xF0) == 0xF0 {
        let header_len = if data[1] & 0x01 == 1 { 7 } else { 9 };
        if data.len() > header_len {
            return data[header_len..].to_vec();
        }
    }
    data.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use me_container::Mp4Demuxer;
    use me_core::{Cursor, ReadAt};
    use std::collections::HashMap;
    use std::sync::Arc;

    fn sps() -> Vec<u8> {
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
        let mut bits = Vec::new();
        push(&mut bits, 66, 8);
        push(&mut bits, 0, 8);
        push(&mut bits, 30, 8);
        ue(&mut bits, 0);
        ue(&mut bits, 0);
        ue(&mut bits, 0);
        ue(&mut bits, 0);
        ue(&mut bits, 1);
        push(&mut bits, 0, 1);
        ue(&mut bits, 119);
        ue(&mut bits, 67);
        push(&mut bits, 1, 1);
        push(&mut bits, 1, 1);
        push(&mut bits, 1, 1);
        ue(&mut bits, 0);
        ue(&mut bits, 0);
        ue(&mut bits, 0);
        ue(&mut bits, 4);
        push(&mut bits, 0, 1);
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

    fn video_track() -> Track {
        let mut t = Track::new(1, MediaKind::Video, CodecId::H264, Timebase::new(30_000));
        t.width = 1920;
        t.height = 1080;
        let mut ed = vec![0, 0, 0, 1];
        ed.extend_from_slice(&sps());
        ed.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE, 0x3C, 0x80]);
        t.extradata = ed;
        t
    }

    fn audio_track() -> Track {
        let mut t = Track::new(2, MediaKind::Audio, CodecId::Aac, Timebase::new(44_100));
        t.sample_rate = 44_100;
        t.channels = 2;
        t.language = Some("eng".into());
        t.extradata = vec![0x12, 0x10];
        t
    }

    fn video_packet(dts: i64, keyframe: bool) -> Packet {
        let mut v = vec![0u8, 0, 0, 1];
        v.push(if keyframe { 0x65 } else { 0x41 });
        v.extend_from_slice(&[0x11, 0x22, 0x33]);
        let mut p = Packet::new(1, dts, dts, 1000, v);
        p.keyframe = keyframe;
        p
    }

    fn audio_packet(dts: i64) -> Packet {
        // 裸 AAC（4 字节）
        Packet::new(2, dts, dts, 1024, vec![0x21, 0x10, 0x05, 0x55])
    }

    fn mux_av(fragments: usize) -> Vec<u8> {
        let mut m = Mp4Muxer::new();
        m.set_fragment_duration_us(1_000_000); // 1 秒一片
        m.set_tracks(&[video_track(), audio_track()]).unwrap();
        for f in 0..fragments {
            for i in 0..30 {
                let dts = (f * 30 + i) as i64 * 1000;
                m.write_packet(&video_packet(dts, i == 0)).unwrap();
            }
            for i in 0..43 {
                let dts = (f * 43 + i) as i64 * 1024;
                m.write_packet(&audio_packet(dts)).unwrap();
            }
        }
        m.finish().unwrap()
    }

    #[test]
    fn roundtrip_through_our_own_demuxer() {
        let data = mux_av(3);
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
        let d = Mp4Demuxer::new(src, "mem").unwrap();
        assert!(d.is_fragmented(), "产出必须是分片形态");
        let info = d.info();
        assert_eq!(info.tracks.len(), 2);
        let v = info.video().expect("要有视频轨");
        assert_eq!(v.codec, CodecId::H264);
        assert_eq!((v.width, v.height), (1920, 1080));
        assert_eq!(v.timebase.ticks_per_second, 30_000);
        let a = info.audio().expect("要有音频轨");
        assert_eq!(a.codec, CodecId::Aac);
        assert_eq!(a.sample_rate, 44_100);
        assert_eq!(a.channels, 2);
        assert_eq!(a.language.as_deref(), Some("eng"));
    }

    #[test]
    fn samples_roundtrip_with_continuous_timestamps() {
        let data = mux_av(3);
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
        let mut d = Mp4Demuxer::new(src, "mem").unwrap();
        let mut video_dts = Vec::new();
        let mut audio_dts = Vec::new();
        let mut video_data = Vec::new();
        while let Some(p) = d.next_packet().unwrap() {
            if p.track_id == 1 {
                video_dts.push(p.dts);
                video_data.push(p.data.clone());
                assert!(p.keyframe == (p.dts % 30_000 == 0), "关键帧标志必须还原");
            } else {
                audio_dts.push(p.dts);
            }
        }
        assert_eq!(video_dts.len(), 90);
        assert_eq!(audio_dts.len(), 129);
        // 视频 dts 必须是 0,1000,2000,...（跨分片连续，tfdt 没写错）
        for (i, d) in video_dts.iter().enumerate() {
            assert_eq!(*d, i as i64 * 1000, "第 {i} 条视频采样的 dts 不对（tfdt 跨分片断了？）");
        }
        for (i, d) in audio_dts.iter().enumerate() {
            assert_eq!(*d, i as i64 * 1024, "第 {i} 条音频采样的 dts 不对");
        }
        // 视频读回来必须是 Annex-B，且与原 NAL 一致
        assert_eq!(&video_data[0][..4], &[0, 0, 0, 1]);
        assert_eq!(video_data[0][4], 0x65);
        assert_eq!(video_data[1][4], 0x41);
    }

    #[test]
    fn fragments_are_split_on_the_target_duration() {
        let data = mux_av(3);
        let mut moofs = 0;
        let mut pos = 0usize;
        while pos + 8 <= data.len() {
            let size = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
            if size < 8 || pos + size > data.len() {
                break;
            }
            if &data[pos + 4..pos + 8] == b"moof" {
                moofs += 1;
            }
            pos += size;
        }
        assert!(moofs >= 3, "1 秒一片 + 3 秒素材应该至少有 3 个分片，实际 {moofs}");
    }

    #[test]
    fn timeline_base_is_per_track_not_shared_ticks() {
        // 两条轨时基不同（视频 15360、音频 44100），且都**不从 0 开始**。
        // 基准必须按微秒存再换算回各轨刻度：直接拿"第一条采样的 dts"
        // 去减另一条轨的 dts 等于在减两个不同单位，会让音视频错开一段。
        let mut v = video_track();
        v.timebase = Timebase::new(15_360); // 0.1s = 1536 刻度
        let mut a = audio_track();
        a.timebase = Timebase::new(44_100); // 0.1s = 4410 刻度

        let mut m = Mp4Muxer::new();
        m.set_tracks(&[v, a]).unwrap();
        for i in 0..4 {
            let dts = 1536 + i * 512;
            let mut p = video_packet(dts, i == 0);
            p.pts = dts;
            p.dts = dts;
            m.write_packet(&p).unwrap();

            let adts = 4410 + i as i64 * 1024;
            m.write_packet(&Packet::new(2, adts, adts, 1024, vec![0x21, 0x10, 0x05, 0x55]))
                .unwrap();
        }
        let data = m.finish().unwrap();

        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
        let mut d = Mp4Demuxer::new(src, "mem").unwrap();
        let mut first: HashMap<u32, i64> = HashMap::new();
        while let Some(p) = d.next_packet().unwrap() {
            first.entry(p.track_id).or_insert(p.dts);
        }
        assert_eq!(first.get(&1).copied(), Some(0), "视频第一条要从 0 开始（按微秒定的基准）");
        assert_eq!(first.get(&2).copied(), Some(0), "音频第一条也要从 0 开始");
    }

    #[test]
    fn explicit_timeline_base_keeps_absolute_positions() {
        // `--timeline` 的用途：seek 到 300 秒后播放器要能显示真实位置。
        // 这里用 5 秒做同样的验证：片段的时间戳应当保留"媒体真实时刻"。
        let mut m = Mp4Muxer::new();
        m.set_tracks(&[video_track()]).unwrap();
        m.set_timeline_base_us(0); // = 绝对值模式（把 5s 处的 5s 原样保留）
        for i in 0..3 {
            let dts = 150_000 + i * 1000; // 5.000s 起，30fps
            let mut p = video_packet(dts, i == 0);
            p.pts = dts;
            p.dts = dts;
            m.write_packet(&p).unwrap();
        }
        let data = m.finish().unwrap();

        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
        let mut d = Mp4Demuxer::new(src, "mem").unwrap();
        let p = d.next_packet().unwrap().unwrap();
        // 30000 时基下 5 秒 = 150000 刻度
        assert_eq!(p.dts, 150_000, "显式基准 0 = 保留绝对时间戳");
    }

    #[test]
    fn audio_adts_is_stripped_to_raw_frames() {
        let mut m = Mp4Muxer::new();
        m.set_tracks(&[audio_track()]).unwrap();
        // 造一个带 ADTS 头的帧
        let mut frame = vec![0xFFu8, 0xF1, 0x50, 0x80, 0x01, 0xBF, 0xFC];
        frame.extend_from_slice(&[0x21, 0x10, 0x05, 0x55]);
        m.write_packet(&Packet::new(2, 0, 0, 1024, frame)).unwrap();
        let data = m.finish().unwrap();

        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
        let mut d = Mp4Demuxer::new(src, "mem").unwrap();
        let p = d.next_packet().unwrap().unwrap();
        assert_eq!(p.data, vec![0x21, 0x10, 0x05, 0x55], "fMP4 里必须是裸 AAC 帧");
    }

    #[test]
    fn esds_lengths_use_variable_encoding() {
        // 描述符长度 > 127 时必须是多字节变长编码，写死一字节会溢出
        let mut m = Mp4Muxer::new();
        let mut a = audio_track();
        let mut asc = vec![0x12u8, 0x10];
        asc.extend(std::iter::repeat(0xAA).take(200)); // 202 字节 → 需要 2 字节长度
        a.extradata = asc;
        m.set_tracks(&[a]).unwrap();
        let data = m.finish().unwrap();
        let pos = data.windows(4).position(|w| w == b"esds").expect("要有 esds");
        let body = pos + 4; // 跳过 box 头
        assert_eq!(data[body + 4], 0x03, "第一个描述符必须是 ES_Descriptor");
        // 长度字段（可能多字节）解析出来必须等于后面的实际字节数
        let mut i = body + 5;
        let mut len = 0usize;
        let mut bytes = 0;
        loop {
            let b = data[i];
            i += 1;
            bytes += 1;
            len = (len << 7) | (b & 0x7F) as usize;
            if b & 0x80 == 0 {
                break;
            }
        }
        assert!(bytes >= 2, "202 字节的描述符必须用多字节长度（实际 {bytes} 字节）");
        let box_size = u32::from_be_bytes([data[pos - 4], data[pos - 3], data[pos - 2], data[pos - 1]]) as usize;
        let box_end = pos - 4 + box_size;
        assert_eq!(i + len, box_end, "描述符长度必须与实际字节数一致");
    }

    #[test]
    fn video_must_be_annexb() {
        let mut m = Mp4Muxer::new();
        m.set_tracks(&[video_track()]).unwrap();
        // 长度前缀形态（不是 Annex-B）：必须报错，而不是瞎转
        let bad = Packet::new(1, 0, 0, 1000, vec![0, 0, 0, 3, 0x65, 0x11, 0x22]);
        let e = m.write_packet(&bad);
        assert!(e.is_err());
        assert!(e.unwrap_err().to_string().contains("Annex-B"));
    }

    #[test]
    fn timestamp_going_backwards_is_an_error() {
        let mut m = Mp4Muxer::new();
        m.set_tracks(&[video_track()]).unwrap();
        m.write_packet(&video_packet(1000, true)).unwrap();
        let e = m.write_packet(&video_packet(500, false));
        assert!(e.is_err(), "时间倒流会写出坏的 tfdt，必须报错");
        assert!(e.unwrap_err().to_string().contains("倒流"));
    }

    #[test]
    fn errors_are_explicit_for_unsupported_inputs() {
        let mut m = Mp4Muxer::new();
        assert!(m.set_tracks(&[]).is_err());
        assert!(m.write_packet(&video_packet(0, true)).is_err());

        // 缺少 AudioSpecificConfig 的 AAC 轨：连 moov 都写不出来（写不出可解的 esds），
        // 必须在 set_tracks 就失败 —— 越早失败越不可能产出坏文件
        let mut m2 = Mp4Muxer::new();
        let mut a = audio_track();
        a.extradata.clear();
        let e = m2.set_tracks(&[a]);
        assert!(e.is_err());
        assert!(e.unwrap_err().to_string().contains("AudioSpecificConfig"));
    }

    #[test]
    fn empty_muxer_still_produces_a_valid_header() {
        // 一条采样都没有时也要产出"能被解析出轨道"的文件（合并失败的边界场景）
        let mut m = Mp4Muxer::new();
        m.set_tracks(&[video_track()]).unwrap();
        let data = m.finish().unwrap();
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
        let d = Mp4Demuxer::new(src, "mem").unwrap();
        let info = d.info();
        assert_eq!(info.tracks.len(), 1);
        assert_eq!(info.video().unwrap().width, 1920);
    }

    #[test]
    fn nal_length_size_follows_the_extradata() {
        // avcC 形态的 extradata 里写的是 2 字节长度前缀，产物必须照它写
        let mut t = video_track();
        t.extradata = convert::extradata_to_avcc(&CodecId::H264, &t.extradata, 2).unwrap();
        assert_eq!(convert::nal_length_size(&CodecId::H264, &t.extradata), 2);

        let mut m = Mp4Muxer::new();
        m.set_tracks(&[t]).unwrap();
        m.write_packet(&video_packet(0, true)).unwrap();
        let data = m.finish().unwrap();

        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
        let mut d = Mp4Demuxer::new(src, "mem").unwrap();
        let p = d.next_packet().unwrap().unwrap();
        // 解析器按 avcC 说的宽度解回来，应当还原成 Annex-B
        assert_eq!(&p.data[..4], &[0, 0, 0, 1]);
        assert_eq!(p.data[4], 0x65);
    }

    // ── AV1（B 站等站点在用的编码）─────────────────────────────────────────

    /// av1C 的载荷：4 字节头（marker/version + seq_profile/level/tier + 色度信息）
    /// 后跟 sequence header OBU 的载荷。
    fn av1_config() -> Vec<u8> {
        vec![0x81, 0x00, 0x0C, 0x00, 0x0A, 0x0B, 0x0C, 0x0D]
    }

    fn av1_track() -> Track {
        let mut t = Track::new(1, MediaKind::Video, CodecId::Av1, Timebase::new(30_000));
        t.width = 1920;
        t.height = 1080;
        t.extradata = av1_config();
        t
    }

    /// 一条 AV1 采样：temporal delimiter OBU + 一个（假的）frame OBU。
    /// 形态就是 fMP4 要求的"OBU 流"，**没有起始码**。
    fn av1_packet(dts: i64, keyframe: bool) -> Packet {
        let mut v = vec![0x12, 0x00]; // OBU_TEMPORAL_DELIMITER，长度 0
        v.extend_from_slice(&[0x32, 0x05]); // OBU_FRAME（假）长度 5
        v.extend_from_slice(if keyframe { &[0x11, 0x22, 0x33, 0x44, 0x55] } else { &[0xAA; 5] });
        let mut p = Packet::new(1, dts, dts, 1000, v);
        p.keyframe = keyframe;
        p
    }

    fn has(hay: &[u8], needle: &[u8]) -> bool {
        hay.windows(needle.len()).any(|w| w == needle)
    }

    #[test]
    fn av1_samples_and_config_roundtrip() {
        let mut m = Mp4Muxer::new();
        m.set_tracks(&[av1_track()]).unwrap();
        for i in 0..10 {
            m.write_packet(&av1_packet(i as i64 * 1000, i == 0)).unwrap();
        }
        let data = m.finish().unwrap();

        // 采样条目与配置盒都必须是 AV1 的形态
        assert!(has(&data, b"av01"), "采样条目应当是 av01");
        assert!(has(&data, b"av1C"), "配置盒应当是 av1C");

        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
        let mut d = Mp4Demuxer::new(src, "mem").unwrap();
        let info = d.info();
        let v = info.video().expect("要有视频轨");
        assert_eq!(v.codec, CodecId::Av1);
        // av1C 的载荷必须**原样**过去（不是 Annex-B 转出来的东西）
        assert_eq!(v.extradata, av1_config());

        let first = d.next_packet().unwrap().unwrap();
        assert_eq!(
            first.data,
            av1_packet(0, true).data,
            "AV1 的 OBU 流必须原样搬（按 Annex-B 转会把采样的长度前缀搞坏）"
        );
        assert!(first.keyframe, "关键帧标志要还原");
    }

    #[test]
    fn av1_without_config_is_rejected_not_written_blindly() {
        // 没有 av1C 就写不出能解的画面：宁可报错，也不要产出一个播放器打不开的文件
        let mut t = av1_track();
        t.extradata.clear();
        let mut m = Mp4Muxer::new();
        let e = m.set_tracks(&[t]);
        assert!(e.is_err(), "缺 av1C 时必须在开工前就拒绝");
        assert!(e.unwrap_err().to_string().contains("av1C"));
    }

    // ── VP9 ───────────────────────────────────────────────────────────────

    /// `vpcC` 的载荷：version 1 + profile/level/位深色度/色彩三要素 +
    /// `codecInitializationDataSize`（0）。
    fn vp9_config() -> Vec<u8> {
        vec![1, 0, 0, 0, 0x00, 0x0A, 0x81, 0x01, 0x02, 0x00, 0x00, 0x00]
    }

    fn vp9_track() -> Track {
        let mut t = Track::new(1, MediaKind::Video, CodecId::Vp9, Timebase::new(30_000));
        t.width = 640;
        t.height = 360;
        t.extradata = vp9_config();
        t
    }

    /// 一条 VP9 采样：压缩帧就是"一整帧字节"，没有起始码也没有长度前缀。
    fn vp9_packet(dts: i64, keyframe: bool) -> Packet {
        let mut v = vec![0x82, 0x49, 0x83, 0x42]; // 假的帧头（uncompressed header 开头）
        v.extend_from_slice(if keyframe { &[0x11; 6] } else { &[0xAA; 6] });
        let mut p = Packet::new(1, dts, dts, 1000, v);
        p.keyframe = keyframe;
        p
    }

    #[test]
    fn vp9_samples_and_config_roundtrip() {
        let mut m = Mp4Muxer::new();
        m.set_tracks(&[vp9_track()]).unwrap();
        for i in 0..10 {
            m.write_packet(&vp9_packet(i as i64 * 1000, i == 0)).unwrap();
        }
        let data = m.finish().unwrap();

        assert!(has(&data, b"vp09"), "采样条目应当是 vp09");
        assert!(has(&data, b"vpcC"), "配置盒应当是 vpcC");

        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
        let mut d = Mp4Demuxer::new(src, "mem").unwrap();
        let v = d.info().video().expect("要有视频轨").clone();
        assert_eq!(v.codec, CodecId::Vp9);
        assert_eq!(v.extradata, vp9_config(), "vpcC 的载荷必须原样过去");
        assert_eq!((v.width, v.height), (640, 360));

        let first = d.next_packet().unwrap().unwrap();
        assert_eq!(
            first.data,
            vp9_packet(0, true).data,
            "VP9 的压缩帧必须原样搬（按 Annex-B 转会以找不到起始码报错）"
        );
        assert!(first.keyframe, "关键帧标志要还原");
    }

    #[test]
    fn vp9_without_config_is_rejected_not_written_blindly() {
        // 没有 vpcC，播放器就不知道 profile/位深/色度采样：宁可报错
        let mut t = vp9_track();
        t.extradata.clear();
        let mut m = Mp4Muxer::new();
        let e = m.set_tracks(&[t]).unwrap_err();
        assert!(e.to_string().contains("vpcC"), "错误要点名 vpcC：{e}");
    }

    #[test]
    fn vp9_with_broken_config_is_rejected() {
        // 版本不认识
        let mut t = vp9_track();
        t.extradata[0] = 7;
        let e = Mp4Muxer::new().set_tracks(&[t]).unwrap_err();
        assert!(e.to_string().contains("版本"), "{e}");

        // 版本 1 至少 11 字节：短的说明盒子被截断了
        let mut t = vp9_track();
        t.extradata.truncate(9);
        let e = Mp4Muxer::new().set_tracks(&[t]).unwrap_err();
        assert!(e.to_string().contains("11"), "{e}");

        // 声明了初始化数据却没有那一段字节
        let mut t = vp9_track();
        t.extradata[9] = 0;
        t.extradata[10] = 4;
        let e = Mp4Muxer::new().set_tracks(&[t]).unwrap_err();
        assert!(e.to_string().contains("初始化数据"), "{e}");
    }

    // ── Opus（`.opus` 与 fMP4 里的 Opus 包是同一串字节）────────────────────

    /// `OpusHead`（Ogg 形态，小端）：引擎内部的统一形态。
    fn opus_head_cfg(channels: u8, pre_skip: u16) -> Vec<u8> {
        let mut v = b"OpusHead".to_vec();
        v.push(1); // version
        v.push(channels);
        v.extend_from_slice(&pre_skip.to_le_bytes());
        v.extend_from_slice(&48_000u32.to_le_bytes());
        v.extend_from_slice(&0i16.to_le_bytes());
        v.push(0); // 映射族 0
        v
    }

    fn opus_track() -> Track {
        let mut t = Track::new(1, MediaKind::Audio, CodecId::Opus, Timebase::new(48_000));
        t.sample_rate = 48_000;
        t.channels = 2;
        t.extradata = opus_head_cfg(2, 312);
        t
    }

    /// 一个 20 ms 的 Opus 包（960 个 48 kHz 样点）：TOC 0x78 = CELT 全带 20 ms 单帧。
    fn opus_packet(dts: i64) -> Packet {
        let mut v = vec![0x78];
        v.extend_from_slice(&[0xAA; 40]);
        Packet::new(1, dts, dts, 960, v)
    }

    #[test]
    fn opus_samples_and_config_roundtrip() {
        let mut m = Mp4Muxer::new();
        m.set_tracks(&[opus_track()]).unwrap();
        for i in 0..12 {
            m.write_packet(&opus_packet(i * 960)).unwrap();
        }
        let data = m.finish().unwrap();

        assert!(has(&data, b"Opus"), "采样条目应当是 Opus");
        assert!(has(&data, b"dOps"), "配置盒应当是 dOps");

        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
        let mut d = Mp4Demuxer::new(src, "mem").unwrap();
        let a = d.info().audio().expect("要有音频轨").clone();
        assert_eq!(a.codec, CodecId::Opus);
        assert_eq!(a.sample_rate, 48_000, "Opus 的解码采样率恒为 48 kHz");
        assert_eq!(a.channels, 2);
        // 引擎内部的统一形态是 OpusHead（Ogg 那边也是它），所以读回来应当**逐字节相同**
        assert_eq!(a.extradata, opus_head_cfg(2, 312), "dOps 必须被规范化成 OpusHead");

        let first = d.next_packet().unwrap().unwrap();
        assert_eq!(first.data, opus_packet(0).data, "Opus 包原样搬");
        assert_eq!(first.duration, 960, "采样时长就是 48 kHz 刻度下的样点数");
        assert_eq!(first.pts, 0);
    }

    #[test]
    fn opus_dops_is_not_a_full_box() {
        // `dOps` 写成 FullBox（多 4 字节 version+flags）是最容易犯的错：字段整体错位，
        // 但错位后读出来的每个数值都"像样的"。这里直接按字节核对盒子内容。
        let mut m = Mp4Muxer::new();
        m.set_tracks(&[opus_track()]).unwrap();
        m.write_packet(&opus_packet(0)).unwrap();
        let data = m.finish().unwrap();

        let at = data.windows(4).position(|w| w == b"dOps").expect("moov 里要有 dOps");
        // 类型字段之前 4 字节是盒子长度，之后 4 字节就是载荷本身
        let size = u32::from_be_bytes([data[at - 4], data[at - 3], data[at - 2], data[at - 1]]);
        assert_eq!(size, 8 + 11, "映射族 0 的 dOps 载荷是 11 字节");
        let p = &data[at + 4..at + 4 + 11];
        assert_eq!(p[0], 0, "第 0 字节是 Version（0），不是 FullBox 的 version+flags");
        assert_eq!(p[1], 2, "声道数");
        assert_eq!(u16::from_be_bytes([p[2], p[3]]), 312, "pre-skip 是**大端**");
        assert_eq!(u32::from_be_bytes([p[4], p[5], p[6], p[7]]), 48_000);
        assert_eq!(p[10], 0, "声道映射族");
    }

    #[test]
    fn opus_without_config_is_rejected_not_written_blindly() {
        let mut t = opus_track();
        t.extradata.clear();
        let e = Mp4Muxer::new().set_tracks(&[t]).unwrap_err();
        assert!(e.to_string().contains("OpusHead"), "错误要说清缺的是 OpusHead：{e}");
    }

    #[test]
    fn opus_timescale_must_be_48k() {
        // 时基不是 48 kHz 时，`duration`（样点数）会被按别的单位解释，
        // 时间轴越播越漂 —— 必须在开工前拦住
        let mut t = opus_track();
        t.timebase = Timebase::new(44_100);
        t.sample_rate = 44_100;
        let e = Mp4Muxer::new().set_tracks(&[t]).unwrap_err();
        assert!(e.to_string().contains("48000"), "{e}");
    }

    // ── FLAC（B 站的"Hi-Res 无损音质"就是它）──────────────────────────────

    /// `dfLa` 的载荷：version/flags + 一个 STREAMINFO metadata block。
    fn flac_config(sample_rate: u32, channels: u8) -> Vec<u8> {
        let mut v = vec![0, 0, 0, 0]; // version + flags
        v.extend_from_slice(&[0x80, 0, 0, 34]); // last-metadata-block + STREAMINFO(34)
        let mut si = vec![0u8; 34];
        // 采样率是 STREAMINFO 里的 20 位字段（偏移 10），后面紧跟 channels-1
        si[10] = ((sample_rate >> 12) & 0xFF) as u8;
        si[11] = ((sample_rate >> 4) & 0xFF) as u8;
        si[12] = (((sample_rate & 0x0F) << 4) as u8) | (((channels - 1) & 0x07) << 1);
        v.extend_from_slice(&si);
        v
    }

    fn flac_track(sample_rate: u32) -> Track {
        let mut t = Track::new(1, MediaKind::Audio, CodecId::Flac, Timebase::new(sample_rate));
        t.sample_rate = sample_rate;
        t.channels = 2;
        t.extradata = flac_config(sample_rate, 2);
        t
    }

    fn flac_packet(dts: i64) -> Packet {
        // 裸 FLAC 帧（帧头 + 假子帧数据），fMP4 里就是这种形态
        Packet::new(1, dts, dts, 4096, vec![0xFF, 0xF8, 0x69, 0x08, 0x00, 0x11, 0x22])
    }

    #[test]
    fn flac_roundtrips_including_hires_sample_rates() {
        // 96k / 192k 是 Hi-Res 的常态。AudioSampleEntry 的采样率字段是 16.16 定点、
        // 整数部分只有 16 位 —— 直接 `rate << 16` 会溢出 u32，写出一个荒唐的采样率。
        for rate in [44_100u32, 96_000, 192_000] {
            let mut m = Mp4Muxer::new();
            m.set_tracks(&[flac_track(rate)]).unwrap();
            for i in 0..5 {
                m.write_packet(&flac_packet(i * 4096)).unwrap();
            }
            let data = m.finish().unwrap();
            assert!(has(&data, b"fLaC"), "采样条目应当是 fLaC（rate={rate}）");
            assert!(has(&data, b"dfLa"), "配置盒应当是 dfLa（rate={rate}）");

            let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
            let mut d = Mp4Demuxer::new(src, "mem").unwrap();
            let info = d.info();
            let a = info.audio().expect("要有音频轨");
            assert_eq!(a.codec, CodecId::Flac);
            assert_eq!(
                a.extradata,
                flac_config(rate, 2),
                "dfLa 载荷要原样往返（不能多写一层 FullBox 头，rate={rate}）"
            );
            assert_eq!(a.sample_rate, rate, "采样率要从 STREAMINFO 读回来（rate={rate}）");
            let p = d.next_packet().unwrap().unwrap();
            assert_eq!(p.data, flac_packet(0).data, "FLAC 裸帧必须原样搬");
        }
    }
}
