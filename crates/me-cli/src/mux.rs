//! 合并/重封装：把若干条输入流合成为一个文件。
//!
//! 这一层是**前端最关心的那条路径**：以前前端调 `ffmpeg -c copy` 合并
//! 视频分片与音频分片，现在改成调 `zuvrust mux`。所以这里刻意做到：
//!
//! - **谁来当画面、谁来当声音由引擎探测决定**，不靠调用方传 `-map`，
//!   也不靠文件名猜（`xxx-1-30080.m4s` 与 `xxx_1.m4s` 都得能用）；
//! - **输出容器由文件扩展名决定**（`.ts` / `.mp4`），不支持就报错；
//! - 进度按"已写字节 / 输入总字节"上报（只搬字节的任务里这是最准的即时进度），
//!   同时带上媒体时间，宿主画进度条用得上；
//! - 只搬字节，**不重编码**：输入编解码器不被目标容器支持时报错，
//!   绝不静默产出一个坏文件。
//!
//! ## 输入怎么变成一条连续的流
//!
//! 一次合并的输入可能长这样（都来自真实下载场景）：
//!
//! | 输入 | 谁提供的 |
//! |---|---|
//! | `video.m4s`（只有画面） | 画面轨 |
//! | `audio.m4s`（只有声音） | 声音轨 |
//! | `标题-1-30080.m4s`、`标题-2-30080.m4s`（一个视频被切成的两段） | 画面轨，**按顺序接起来** |
//! | `整片.mp4`（音视频都有） | 两条轨都来自它 |
//!
//! 规则只有一条，**由探测结果决定、与文件名无关**：
//! 含视频轨的输入按传入顺序组成"画面序列"，含音频轨的输入按传入顺序组成"声音序列"；
//! 两个序列都按**段**对齐（第 i 段 = 画面序列的第 i 个 + 声音序列的第 i 个），
//! 段内做多轨交织，段与段之间把时间轴平移到上一段的末尾。
//!
//! 为什么按"段"而不是"把所有输入一起交织"：段与段的时间轴平移量必须**等这一段读完
//! 才知道**（分片式 fMP4 的 `mvhd` 时长通常是 0，探不出时长），而交织要求同时拿到
//! 两条轨的数据。按段处理让"平移量"在该段结束时自然确定，也顺手保住了
//! `tfdt` 必须连续递增这条硬要求。

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use me_container::Demuxer;
use me_core::{CodecId, MediaError, MediaInfo, MediaKind, Packet, ReadAt, Result, Timebase, Track};
use me_mux::{AdtsMuxer, FlacMuxer, Interleaver, MkvMuxer, Mp4Muxer, TsMuxer};

use crate::progress::Progress;

/// 目标容器。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputContainer {
    MpegTs,
    Fmp4,
    Matroska,
    /// `.webm`：Matroska 的受限子集（VP8/VP9/AV1 + Opus）。
    WebM,
    /// `.flac`：裸 FLAC，只能装声音。
    Flac,
    /// `.aac`：裸 ADTS，只能装 AAC 声音。
    Adts,
}

impl OutputContainer {
    /// 按扩展名判断（宿主传的就是文件路径，不再让调用方多传一个参数）。
    pub fn from_path(path: &str) -> Result<Self> {
        let lower = path.to_ascii_lowercase();
        let ext = lower.rsplit('.').next().unwrap_or("");
        match ext {
            "ts" | "m2ts" | "mts" => Ok(OutputContainer::MpegTs),
            "mp4" | "m4s" | "m4v" | "mov" | "m4a" => Ok(OutputContainer::Fmp4),
            // `.mka` = 只有声音的 Matroska，容器规则完全一样
            "mkv" | "mka" => Ok(OutputContainer::Matroska),
            "webm" => Ok(OutputContainer::WebM),
            "flac" => Ok(OutputContainer::Flac),
            "aac" | "adts" => Ok(OutputContainer::Adts),
            other => Err(MediaError::Unsupported(format!(
                "不认识的输出容器 `.{other}`（支持 .ts / .mp4 / .m4s / .mov / .mkv / .mka / \
                 .webm / .flac / .aac）"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            OutputContainer::MpegTs => "mpeg-ts",
            OutputContainer::Fmp4 => "fmp4",
            OutputContainer::Matroska => "matroska",
            OutputContainer::WebM => "webm",
            OutputContainer::Flac => "flac",
            OutputContainer::Adts => "adts",
        }
    }

    /// 这个容器本身**只能装声音**（`.flac` / `.aac`）。
    ///
    /// 裸音频流没有"轨"的概念，画面根本无处安放 ⇒ 遇到这类产物自动丢掉画面，
    /// 而不是要求调用方先传 `--audio-only`（宿主并没有暴露那个开关）。
    fn audio_only(self) -> bool {
        matches!(self, OutputContainer::Flac | OutputContainer::Adts)
    }
}

/// 合并结果（写进 NDJSON 的 `result`）。
pub struct MuxOutcome {
    pub output: String,
    pub container: &'static str,
    pub bytes: u64,
    pub packets: u64,
    pub duration_us: Option<i64>,
    pub tracks: Vec<Track>,
  /// 参与合并的输入个数（诊断用）。
    pub inputs: usize,
}

/// 取一条输入里的某类轨道（**只认探测结果**，与文件名无关）。
fn pick_track(info: &MediaInfo, kind: MediaKind) -> Option<&Track> {
    info.tracks.iter().find(|t| t.kind == kind)
}

/// 打开一个输入、读**目标轨**的第一帧，从 AC-3 的 syncframe 头里拼出 `dac3` 载荷。
///
/// 只有 AC-3 需要走这条路（别的编码的配置都在容器里）。读不出来就让调用**失败**：
/// 写一个字段全 0 的 `dac3` 出去，播放器会按错误的声道布局去解，症状是
/// "文件能生成但声音不对"，比直接报错难查得多。
///
/// ⚠️ **必须按轨道号过滤**：解封装器吐出的第一条采样是**按解码时间交织**的，
/// 大多时候是画面那条轨 —— 不看 `track_id` 就会拿一帧视频去解 AC-3 同步字，
/// 报出"第一帧不是 AC-3"这种指向完全错误的结论。
fn read_dac3_from_first_frame(
    src: Arc<dyn ReadAt>,
    name: &str,
    want_track: u32,
) -> Result<Vec<u8>> {
    let mut d = me_container::demux::open(src, name)?;
    // 上限只是防御性的：正常文件里音频第一帧就在最前面几十条内
    for _ in 0..1000 {
        let Some(p) = d.next_packet()? else { break };
        if p.track_id != want_track {
            continue;
        }
        let dac3 = me_container::parse_dac3(&p.data).ok_or_else(|| {
            MediaError::Bitstream(format!(
                "{name} 轨 {want_track} 的第一帧不是 AC-3 syncframe（同步字不是 0x0B77），取不出 dac3"
            ))
        })?;
        return Ok(dac3.to_payload().to_vec());
    }
    Err(MediaError::Container(format!(
        "{name} 里读不到轨 {want_track} 的采样，取不出 AC-3 的 dac3"
    )))
}

/// 某一段里、某个输入要搬运的轨道：`源轨 id → 输出轨 id`。
struct SegInput {
    input_index: usize,
    map: Vec<(u32, u32)>,
}

/// 片段裁剪窗口（微秒）：只把 `[start, end)` 这一段写进产物。
///
/// 起点是**帧级**的：视频从切点之前最近的关键帧开始解码（丢帧会花屏），
/// 但容器会把「呈现起点」挪到切点 —— fMP4 写 `edts/elst`，所以看到的
/// 第一帧就是切点那一帧；mkv / ts 没有编辑列表，只能从那个关键帧开始
/// （多出最多一个 GOP 的画面）。终点在所有容器上都是帧级的（按显示时间丢采样）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClipRange {
    /// 切点（微秒，相对媒体起点）。
    pub start_us: i64,
    /// 切尾（微秒）；`None` = 一直到结尾。
    pub end_us: Option<i64>,
}

/// 合并的可选项（宿主「视频」设置项映射到这里）。
#[derive(Debug, Clone, Copy, Default)]
pub struct MergeOptions {
    /// 分片/簇的目标时长（微秒）。`None` = 用容器默认值（都是 2 秒）。
    pub fragment_us: Option<i64>,
    /// 要哪些内容（默认画面与声音都要）。
    pub tracks: TrackChoice,
    /// 只保留某一段（默认整条）。
    pub clip: Option<ClipRange>,
}

/// 合并产物保留哪些内容。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrackChoice {
    /// 画面 + 声音（默认）。
    #[default]
    Both,
    /// 只留声音（"提取音频"）。
    AudioOnly,
    /// 只留画面。
    VideoOnly,
}

/// 把若干输入合并成一个文件。
pub fn mux_to_file(
    output: &str,
    inputs: &[(&str, Arc<dyn ReadAt>, MediaInfo)],
    opts: MergeOptions,
    progress: bool,
    cancel: &me_stream::source::CancelFlag,
) -> Result<MuxOutcome> {
    match run(output, inputs, opts, progress, cancel) {
        Ok(o) => Ok(o),
        Err(e) => {
            // 半截产物必须清掉：留着它，用户会以为"合并好了"，
            // 而且下一次重试还会撞上"输出已存在"。
            let _ = std::fs::remove_file(output);
            Err(e)
        }
    }
}

fn run(
    output: &str,
    inputs: &[(&str, Arc<dyn ReadAt>, MediaInfo)],
    opts: MergeOptions,
    progress: bool,
    cancel: &me_stream::source::CancelFlag,
) -> Result<MuxOutcome> {
    let container = OutputContainer::from_path(output)?;

    // ── 1. 角色分组：含视频轨的进画面序列、含音频轨的进声音序列（按传入顺序） ──
    let mut video_inputs: Vec<usize> = Vec::new();
    let mut audio_inputs: Vec<usize> = Vec::new();
    for (i, (_, _, info)) in inputs.iter().enumerate() {
        if pick_track(info, MediaKind::Video).is_some() {
            video_inputs.push(i);
        }
        if pick_track(info, MediaKind::Audio).is_some() {
            audio_inputs.push(i);
        }
    }
    // 「只留声音 / 只留画面」：把不要的那一路整个摘掉 —— 下游（输出轨、分段、
    // 采样映射）都是按这两个列表推的，摘掉即生效；未被映射的采样会被自然跳过。
    //
    // `.flac` / `.aac` 这类**裸音频流**容器本身装不下画面，这里直接按"只留声音"处理：
    // 不这么做，用户选个 .aac 就会被 check_codec_support 以"装不下 video"拒掉，
    // 而"提取音轨"恰恰是他想要的。
    let tracks = if container.audio_only() {
        TrackChoice::AudioOnly
    } else {
        opts.tracks
    };
    match tracks {
        TrackChoice::Both => {}
        TrackChoice::AudioOnly => video_inputs.clear(),
        TrackChoice::VideoOnly => audio_inputs.clear(),
    }
    if video_inputs.is_empty() && audio_inputs.is_empty() {
        return Err(MediaError::Container(match tracks {
            TrackChoice::Both => "输入里既没有视频轨也没有音频轨，没有可合并的内容".into(),
            TrackChoice::AudioOnly => "输入里没有音频轨，没得可提取".into(),
            TrackChoice::VideoOnly => "输入里没有视频轨，没得可提取".into(),
        }));
    }

    // ── 2. 输出轨：每个角色一条，沿用第一条源轨的时基与编解码参数 ──
    let mut out_tracks: Vec<Track> = Vec::new();
    let mut next_id = 1u32;
    let mut video_out: Option<u32> = None;
    let mut audio_out: Option<u32> = None;
    if let Some(&i) = video_inputs.first() {
        if let Some(t) = pick_track(&inputs[i].2, MediaKind::Video) {
            let mut ot = t.clone();
            ot.id = next_id;
            next_id += 1;
            video_out = Some(ot.id);
            out_tracks.push(ot);
        }
    }
    if let Some(&i) = audio_inputs.first() {
        if let Some(t) = pick_track(&inputs[i].2, MediaKind::Audio) {
            let mut ot = t.clone();
            // 音频是最后一条轨道，到此不再需要新号
            ot.id = next_id;
            audio_out = Some(ot.id);
            out_tracks.push(ot);
        }
    }
    // ── 2.5 AC-3 的 `dac3` 补齐 ──
    //
    // TS 与裸流里的 AC-3 **没有配置盒**（采样率码、码率码、声道配置全都写在每个
    // syncframe 的头里），而 fMP4 产物必须有 `dac3`。所以只能在这里从**码流第一帧**
    // 读出来 —— 而且必须在 `Muxer::with_writer` 之前做：moov 里的 `stsd` 是最终
    // 产物的一部分，写完就定型了，之后再补也进不去。
    if matches!(container, OutputContainer::Fmp4) {
        if let (Some(&ai), Some(aid)) = (audio_inputs.first(), audio_out) {
            let needs_fill = out_tracks
                .iter()
                .any(|t| t.id == aid && t.codec == CodecId::Ac3 && t.extradata.is_empty());
            if needs_fill {
                let (name, src, info) = &inputs[ai];
                let src_track = pick_track(info, MediaKind::Audio)
                    .map(|t| t.id)
                    .ok_or_else(|| {
                        MediaError::Container(format!("{name} 里没有音频轨，取不出 AC-3 的 dac3"))
                    })?;
                let dac3 = read_dac3_from_first_frame(src.clone(), name, src_track)?;
                if let Some(t) = out_tracks.iter_mut().find(|t| t.id == aid) {
                    t.extradata = dac3;
                }
            }
        }
    }

    check_codec_support(container, &out_tracks)?;

    // 输出轨的时基（平移时要用）
    let out_timebases: HashMap<u32, Timebase> =
        out_tracks.iter().map(|t| (t.id, t.timebase)).collect();
    // 源轨的时基：`(输入下标, 源轨 id) → 时基`
    let mut src_timebases: HashMap<(usize, u32), Timebase> = HashMap::new();
    for (i, (_, _, info)) in inputs.iter().enumerate() {
        for t in &info.tracks {
            src_timebases.insert((i, t.id), t.timebase);
        }
    }

    // ── 3. 分段：第 i 段 = 画面序列的第 i 个 + 声音序列的第 i 个 ──
    let seg_count = video_inputs.len().max(audio_inputs.len());
    let mut segments: Vec<Vec<SegInput>> = Vec::with_capacity(seg_count);
    for s in 0..seg_count {
        let mut seg: Vec<SegInput> = Vec::new();
        if let (Some(&i), Some(out_id)) = (video_inputs.get(s), video_out) {
            if let Some(t) = pick_track(&inputs[i].2, MediaKind::Video) {
                seg.push(SegInput { input_index: i, map: vec![(t.id, out_id)] });
            }
        }
        if let (Some(&i), Some(out_id)) = (audio_inputs.get(s), audio_out) {
            if let Some(t) = pick_track(&inputs[i].2, MediaKind::Audio) {
                // 同一个输入同时供画面与声音（完整 mp4 / 单文件重封装）：
                // **并进同一条 SegInput**，否则这个输入会被拉两遍（第二遍已经空了）
                match seg.iter_mut().find(|x| x.input_index == i) {
                    Some(x) => {
                        // 容器保证同一个文件里轨道号唯一。真撞上了说明探测结果自相矛盾，
                        // 这时候"按号找轨"会把一路的采样写进另一路的轨道 —— 宁可报错。
                        if x.map.iter().any(|(src_id, _)| *src_id == t.id) {
                            return Err(MediaError::Container(format!(
                                "输入 {} 的画面轨与声音轨用了同一个轨道号 {}，无法区分",
                                inputs[i].0, t.id
                            )));
                        }
                        x.map.push((t.id, out_id));
                    }
                    None => seg.push(SegInput { input_index: i, map: vec![(t.id, out_id)] }),
                }
            }
        }
        segments.push(seg);
    }

    // ── 3.5 片段裁剪：归一化窗口 + 先把落点问出来 ──
    //
    // 必须赶在**建 muxer 之前**：mp4 的呈现起点（`edts/elst`）是 `moov` 的一部分，
    // 而 `moov` 在 `Muxer::with_writer` 里就写出去了，之后再改不进产物。
    let clip = opts.clip.and_then(|c| {
        let start = c.start_us.max(0);
        let end = c.end_us.filter(|e| *e > start);
        (start > 0 || end.is_some()).then_some(ClipRange { start_us: start, end_us: end })
    });
    if clip.is_some() && inputs.len() != 1 {
        return Err(MediaError::Unsupported(
            "片段裁剪只支持单个输入：多输入串接时「第几秒」没有唯一定义".into(),
        ));
    }

    // 每个输入一个解封装器（懒开：没被任何一段用到的输入不必打开）
    let mut demuxers: Vec<Option<Box<dyn Demuxer>>> = (0..inputs.len()).map(|_| None).collect();
    // 裁剪起点落到「切点之前最近的关键帧」，这个落点就是铺垫段（解码必需）的起点
    let mut clip_landing_us = 0i64;
    if let Some(c) = clip {
        if c.start_us > 0 {
            let (name, src, _) = &inputs[0];
            let mut d = me_container::demux::open(src.clone(), name)?;
            clip_landing_us = d.seek(c.start_us)?;
            demuxers[0] = Some(d);
        }
    }
    // 呈现起点 = 切点相对铺垫起点的偏移（只有 mp4 能表达，见 `set_edit_onset_us`）
    let edit_onset_us = clip.map_or(0, |c| (c.start_us - clip_landing_us).max(0));

    // 声明总时长：各输入 duration 之和（**不是 min**：多段串接的总时长是加起来）。
    // 它只作参考（分片源常常没有声明时长），但 MKV 的 `Info.Duration` 可以先用它 ——
    // 播放器/探针就不用先扫全文件才知道有多长（fMP4 那边靠每段的 `sidx` 解决）。
    let mut declared_us: i64 = inputs.iter().filter_map(|(_, _, i)| i.duration_us).sum();
    // 裁剪后声明时长也要跟着缩，否则进度条按整条视频算，永远走不到 100%
    if let (Some(c), true) = (clip, declared_us > 0) {
        let end = c.end_us.unwrap_or(declared_us).min(declared_us);
        declared_us = (end - c.start_us).max(0);
    }

    // ── 4. 产物直写目标（不攒内存） ──
    let file = std::fs::File::create(output)?;
    // 1 MiB 缓冲：合并是顺序写，缓冲再大收益有限，而 1 MiB 已经能把 syscall 压到可忽略
    let writer = Box::new(std::io::BufWriter::with_capacity(1 << 20, file));
    let mut muxer = Muxer::with_writer(
        container,
        &out_tracks,
        (declared_us > 0).then_some(declared_us),
        edit_onset_us,
        writer,
    )?;
    if let Some(us) = opts.fragment_us {
        muxer.set_fragment_us(us);
    }

    // ── 5. 进度口径 ──
    // 输入总字节：只搬字节的任务里，它是最准的分母（媒体的声明时长常常是 0）。
    let input_bytes: u64 = inputs.iter().map(|(_, s, _)| s.len().unwrap_or(0)).sum();
    let mut reporter = Progress::new(progress, "mux");
    let mut packets_out = 0u64;
    // 时间轴：**每条输出轨各自**的"下一段起点"与"已见末尾"（微秒）。
    //
    // ⚠️ 平移量必须**按轨**算，不能全片共用一个起点。段数不等时（一整条画面 +
    // 三段声音、或画面 3 段 + 声音 2 段）用全片最大末尾会把少的那一方的后续段
    // 推到对方的末尾之后 —— 产物里那一轨中间整整一段是空的（用户听到的是
    // "合并完中间有几分钟没声音"）。按轨平移下，每条轨都接在自己的上一段末尾，
    // 各自的 `tfdt` 仍然严格递增（fMP4 的硬要求是**每轨**递增，不是跨轨）。
    let mut track_start_us: HashMap<u32, i64> = HashMap::new();
    let mut track_end_us: HashMap<u32, i64> = HashMap::new();

    for seg in &segments {
        cancel.check()?;
        let mut interleaver = Interleaver::new(&out_tracks);

        // 每条源轨各自的零点（同一个输入里音视频的相对偏移是真实信息，不能抹掉）。
        // key 带上输入下标：段里两个输入的轨道号可能相同。
        let mut first_dts: HashMap<(usize, u32), i64> = HashMap::new();
        let mut done: Vec<bool> = vec![false; seg.len()];

        // 段内**轮流**从每个输入各取一条采样 —— 这是让多条轨的时间轴同步推进的唯一办法。
        // 千万别写成"把一个输入读完再读下一个"：那样后读的那条轨（通常是音频）的
        // 全部采样都会被堆进同一个分片，产物变成"视频全在前面、音频全在文件末尾"，
        // 播放时前大半段没有声音（而且那个分片会大到离谱）。
        loop {
            cancel.check()?;
            let mut any = false;
            for (k, si) in seg.iter().enumerate() {
                if done[k] {
                    continue;
                }
                if demuxers[si.input_index].is_none() {
                    let (name, src, _) = &inputs[si.input_index];
                    demuxers[si.input_index] =
                        Some(me_container::demux::open(src.clone(), name)?);
                }
                let d = demuxers[si.input_index].as_mut().expect("刚判过");
                match d.next_packet()? {
                    Some(p) => {
                        any = true;
                        let Some(&(_, out_id)) =
                            si.map.iter().find(|(s_id, _)| *s_id == p.track_id)
                        else {
                            continue;
                        };
                        let Some(src_tb) =
                            src_timebases.get(&(si.input_index, p.track_id)).copied()
                        else {
                            continue;
                        };
                        let out_tb = out_timebases.get(&out_id).copied().unwrap_or(src_tb);

                        // ── 片段裁剪：丢掉落点之前的「预滚帧」 ──
                        //
                        // 分片输入（fMP4 / HLS）的 seek 只能定位到**含该关键帧的
                        // 分片开头**，那个分片里落在关键帧之前的帧既不是关键帧、
                        // 也不会被后面引用 —— 留着只会让文件起点比落点更早，
                        // `elst` 的 `media_time` 于是偏掉（最多一个分片，实测 2 秒）。
                        // 渐进式 mp4 的 seek 直接定位到关键帧下标，这条对它是空操作。
                        if let Some(c) = clip {
                            if c.start_us > 0 {
                                let seg_end_us = src_tb.ticks_to_us(p.pts)
                                    + src_tb.ticks_to_us(p.duration as i64);
                                // 关键帧留 100ms 余量：`elst`/`cts` 会让落点关键帧的
                                // pts 比 seek 报回来的值早一点点，别把它一起丢了
                                let todo = if p.keyframe {
                                    seg_end_us + 100_000 <= clip_landing_us
                                } else {
                                    seg_end_us <= clip_landing_us
                                };
                                if todo {
                                    continue;
                                }
                            }
                        }

                        // ── 片段裁剪：切尾（帧级） ──
                        //
                        // 判据用 **pts**（显示时间）而不是 dts：B 帧的 dts 早于 pts，
                        // 按 dts 丢会把该留的帧丢掉，结尾会闪一下窗口外的画面。
                        //
                        // 起点那边的处理见上：落点关键帧之后的采样一条都不能丢
                        // （丢了就缺参考帧），落点之前的预滚帧可以丢 —— 容器把
                        // 剩下的铺垫藏进 mp4 的 edts/elst，mkv/ts 则如实保留。
                        if let Some(end_us) = clip.and_then(|c| c.end_us) {
                            if src_tb.ticks_to_us(p.pts) >= end_us {
                                continue;
                            }
                            // 解码顺序已经明显越过窗口 → 这个输入读完了。
                            // 留 1 秒余量：B 帧重排下 dts 落后于 pts，丢早了会少帧
                            if src_tb.ticks_to_us(p.dts) >= end_us + 1_000_000 {
                                done[k] = true;
                                for (_, out_id) in &si.map {
                                    interleaver.finish_track(*out_id);
                                }
                                continue;
                            }
                        }

                        let base = *first_dts
                            .entry((si.input_index, p.track_id))
                            .or_insert(p.dts);
                        // 这一段里**这条轨**的起点（别的轨走到哪了与它无关）
                        let seg_start_us = *track_start_us.get(&out_id).unwrap_or(&0);

                        let mut q = p;
                        q.track_id = out_id;
                        q.dts = shift_ticks(src_tb, out_tb, q.dts, base, seg_start_us);
                        q.pts = shift_ticks(src_tb, out_tb, q.pts, base, seg_start_us);
                        q.duration = if src_tb == out_tb {
                            q.duration
                        } else {
                            out_tb
                                .us_to_ticks(src_tb.ticks_to_us(q.duration as i64))
                                .max(0)
                                .min(u32::MAX as i64) as u32
                        };

                        // 记这一段的末尾（段间平移量靠它）；时长可能是 0，所以用"最大末端"
                        let end_us =
                            out_tb.ticks_to_us(q.dts) + out_tb.ticks_to_us(q.duration as i64);
                        let e = track_end_us.entry(out_id).or_insert(i64::MIN);
                        if end_us > *e {
                            *e = end_us;
                        }

                        interleaver.push(q)?;
                    }
                    None => {
                        done[k] = true;
                        // 这条输入到头了：告诉交织器别再等它的轨
                        for (_, out_id) in &si.map {
                            interleaver.finish_track(*out_id);
                        }
                    }
                }
            }

            // 边推边取：缓冲只保留"窗口"那么大的量（2 秒媒体数据），
            // 不随产物增长 —— 这是"内存占用与文件大小无关"的另一半
            while let Some(p) = interleaver.pop() {
                muxer.write(&p)?;
                packets_out += 1;
                let media_us = track_end_us.values().copied().max().unwrap_or(0);
                reporter.maybe_report(&progress_line(
                    media_us,
                    declared_us,
                    muxer.bytes_written(),
                    input_bytes,
                ));
            }

            if !any {
                break;
            }
        }

        // 段末把剩下的一次性取完，然后**把每条轨各自推到自己的末尾**：
        // 这一段的每一轨都从自己上一段的末尾接着走，`tfdt` 因此严格递增
        // （倒退会被 fMP4 封装器当场拒绝），而段数不等的那一方也不会被推走。
        interleaver.flush();
        while let Some(p) = interleaver.pop() {
            muxer.write(&p)?;
            packets_out += 1;
        }
        for t in &out_tracks {
            if let Some(end) = track_end_us.get(&t.id) {
                let cur = *track_start_us.get(&t.id).unwrap_or(&0);
                track_start_us.insert(t.id, (*end).max(cur));
            }
        }
    }

    // ── 6. 收尾 ──
    let bytes = muxer.finish()?;
    // 呈现时长要扣掉铺垫段：mp4 把铺垫藏在编辑列表里（从切点开始播），
    // 不扣的话「裁出来的 30 秒」会报成 32 秒
    let presented_us = if matches!(container, OutputContainer::Fmp4) { edit_onset_us } else { 0 };
    let written_end_us = track_end_us
        .values()
        .copied()
        .filter(|v| *v > 0)
        .max()
        .map(|v| (v - presented_us).max(0));
    // 时长以实际写出的媒体时间为准（声明时长只作参考：分片常常没有声明时长）
    let duration_us = written_end_us.or(if declared_us > 0 { Some(declared_us) } else { None });
    reporter.finish(&progress_line(
        duration_us.unwrap_or(0),
        declared_us,
        bytes,
        input_bytes,
    ));

    Ok(MuxOutcome {
        output: output.to_string(),
        container: container.name(),
        bytes,
        packets: packets_out,
        duration_us,
        tracks: out_tracks,
        inputs: inputs.len(),
    })
}

/// 把源轨的一个刻度平移到输出时间轴（段起点 + 段内相对时间）。
///
/// 时基相同时走**纯刻度**路径（最常见的路径：分片与产物都是 fMP4，
/// 时基一模一样），一点精度都不丢；不同时基才经微秒换算。
fn shift_ticks(
    src_tb: Timebase,
    out_tb: Timebase,
    ticks: i64,
    base: i64,
    seg_start_us: i64,
) -> i64 {
    if src_tb == out_tb {
        let start = out_tb.us_to_ticks(seg_start_us);
        start + (ticks - base)
    } else {
        out_tb.us_to_ticks(seg_start_us + src_tb.ticks_to_us(ticks - base))
    }
}

/// 目标容器能不能装下这些编解码器。
///
/// 提前检查比"写到一半才发现"好：后者会留下一个半截的坏文件。
fn check_codec_support(container: OutputContainer, tracks: &[Track]) -> Result<()> {
    for t in tracks {
        let ok = match container {
            // TS 里能"装"的只有经典那几种：AV1/VP9 没有标准 stream_type，
            // FLAC/Opus 要各自的描述符 —— 都没实现，宁可在这里明确拒绝，
            // 也不产出一个"有数据但解不出来"的文件
            OutputContainer::MpegTs => match t.kind {
                MediaKind::Video => {
                    !matches!(t.codec, CodecId::Av1 | CodecId::Vp9 | CodecId::Vp8 | CodecId::Unknown(_))
                }
                MediaKind::Audio => matches!(
                    t.codec,
                    CodecId::Aac | CodecId::Mp3 | CodecId::Ac3 | CodecId::Eac3
                ),
                _ => false,
            },
            // fMP4：有对应配置盒的都能装
            // （avcC / hvcC / av1C / vpcC / esds / dfLa / dOps / dac3）
            OutputContainer::Fmp4 => match t.kind {
                MediaKind::Video => {
                    matches!(t.codec, CodecId::H264 | CodecId::H265 | CodecId::Av1 | CodecId::Vp9)
                }
                // MP3 与 AAC 共用 `mp4a` 采样条目（靠 esds 的 OTI 区分），
                // AC-3 用 `ac-3` + `dac3`。读侧（me-container/src/mp4.rs）本来就在认这三种，
                // 写侧补齐后"能读出来的也能写回去"。
                MediaKind::Audio => matches!(
                    t.codec,
                    CodecId::Aac | CodecId::Mp3 | CodecId::Flac | CodecId::Opus | CodecId::Ac3
                ),
                _ => false,
            },
            // Matroska：视频四种都认；音频除 Vorbis 都认
            //（FLAC 的 `fLaC` 头由封装器从 `dfLa` 载荷转出来，见 me_container::flac_config）
            OutputContainer::Matroska => match t.kind {
                MediaKind::Video => {
                    matches!(t.codec, CodecId::H264 | CodecId::H265 | CodecId::Av1 | CodecId::Vp9)
                }
                MediaKind::Audio => matches!(
                    t.codec,
                    CodecId::Aac | CodecId::Mp3 | CodecId::Opus | CodecId::Ac3 | CodecId::Eac3
                        | CodecId::Flac
                ),
                _ => false,
            },
            // WebM 是 Matroska 的受限子集：VP8/VP9/AV1 + Opus（引擎没有 Vorbis）
            OutputContainer::WebM => match t.kind {
                MediaKind::Video => {
                    matches!(t.codec, CodecId::Vp8 | CodecId::Vp9 | CodecId::Av1)
                }
                MediaKind::Audio => matches!(t.codec, CodecId::Opus),
                _ => false,
            },
            // 裸流：只能装一条，且编码必须完全对得上
            OutputContainer::Flac => t.kind == MediaKind::Audio && t.codec == CodecId::Flac,
            OutputContainer::Adts => t.kind == MediaKind::Audio && t.codec == CodecId::Aac,
        };
        if !ok {
            return Err(MediaError::Unsupported(format!(
                "{} 装不下这条轨道：{} / {:?}{}",
                container.name(),
                t.codec.name(),
                t.kind,
                match container {
                    OutputContainer::WebM =>
                        "（WebM 只允许 VP8 / VP9 / AV1 画面与 Opus 声音，换 .mkv 或 .mp4）",
                    OutputContainer::Flac => "（裸 FLAC 只装 FLAC 声音，换 .mkv / .mp4 / .mka）",
                    OutputContainer::Adts => "（裸 ADTS 只装 AAC 声音，换 .m4a / .mkv）",
                    _ => "",
                }
            )));
        }
    }
    // 裸流装不下第二条轨（产物里没有"轨"的概念，第二条只会是坏文件）
    if matches!(container, OutputContainer::Flac | OutputContainer::Adts) && tracks.len() > 1 {
        return Err(MediaError::Unsupported(format!(
            "{} 只能装一条声音轨，这里有 {} 条",
            container.name(),
            tracks.len()
        )));
    }
    Ok(())
}

/// 封装器的几种实现（对外只暴露"写一条采样"和"收尾"）。
///
/// `.mkv` 与 `.webm` 共用一个变体：它们是同一个封装器，只差 `DocType`
/// （在 [`Muxer::with_writer`] 里按容器设置）。
enum Muxer {
    Ts(TsMuxer),
    Mp4(Mp4Muxer),
    Mkv(MkvMuxer),
    Flac(FlacMuxer),
    Adts(AdtsMuxer),
}

impl Muxer {
    fn with_writer(
        container: OutputContainer,
        tracks: &[Track],
        duration_hint_us: Option<i64>,
        edit_onset_us: i64,
        writer: Box<dyn Write + Send>,
    ) -> Result<Self> {
        match container {
            OutputContainer::MpegTs => {
                let mut m = TsMuxer::with_writer(writer);
                m.set_tracks(tracks)?;
                Ok(Muxer::Ts(m))
            }
            OutputContainer::Fmp4 => {
                let mut m = Mp4Muxer::with_writer(writer);
                // ⚠️ 必须在 `set_tracks`（写 moov）之前设：编辑列表是 moov 的一部分
                m.set_edit_onset_us(edit_onset_us);
                m.set_tracks(tracks)?;
                Ok(Muxer::Mp4(m))
            }
            OutputContainer::Matroska | OutputContainer::WebM => {
                let mut m = if container == OutputContainer::WebM {
                    MkvMuxer::with_writer_webm(writer)
                } else {
                    MkvMuxer::with_writer(writer)
                };
                // ⚠️ 必须在 `set_tracks`（它会把 `Info` 写进产物）**之前**设好
                m.set_duration_hint_us(duration_hint_us);
                m.set_tracks(tracks)?;
                Ok(Muxer::Mkv(m))
            }
            OutputContainer::Flac => {
                let mut m = FlacMuxer::with_writer(writer);
                m.set_tracks(tracks)?;
                Ok(Muxer::Flac(m))
            }
            OutputContainer::Adts => {
                let mut m = AdtsMuxer::with_writer(writer);
                m.set_tracks(tracks)?;
                Ok(Muxer::Adts(m))
            }
        }
    }

    /// 分片/簇的目标时长（微秒）。TS 与裸音频流没有分片概念，忽略。
    fn set_fragment_us(&mut self, us: i64) {
        match self {
            Muxer::Ts(_) | Muxer::Flac(_) | Muxer::Adts(_) => {}
            Muxer::Mp4(m) => m.set_fragment_duration_us(us),
            Muxer::Mkv(m) => m.set_cluster_duration_us(us),
        }
    }

    fn write(&mut self, p: &Packet) -> Result<()> {
        match self {
            Muxer::Ts(m) => m.write_packet(p),
            Muxer::Mp4(m) => m.write_packet(p),
            Muxer::Mkv(m) => m.write_packet(p),
            Muxer::Flac(m) => m.write_packet(p),
            Muxer::Adts(m) => m.write_packet(p),
        }
    }

    /// 已经写出的字节数（进度上报用；含已落盘的部分）。
    fn bytes_written(&self) -> u64 {
        match self {
            Muxer::Ts(m) => m.bytes_written(),
            Muxer::Mp4(m) => m.bytes_written(),
            Muxer::Mkv(m) => m.bytes_written(),
            Muxer::Flac(m) => m.bytes_written(),
            Muxer::Adts(m) => m.bytes_written(),
        }
    }

    /// 收尾并返回写出的总字节数。
    fn finish(self) -> Result<u64> {
        match self {
            Muxer::Ts(mut m) => {
                m.finish()?;
                Ok(m.bytes_written())
            }
            Muxer::Mp4(mut m) => {
                m.finish()?;
                Ok(m.bytes_written())
            }
            Muxer::Mkv(mut m) => {
                m.finish()?;
                Ok(m.bytes_written())
            }
            Muxer::Flac(mut m) => {
                m.finish()?;
                Ok(m.bytes_written())
            }
            Muxer::Adts(mut m) => {
                m.finish()?;
                Ok(m.bytes_written())
            }
        }
    }
}

/// 进度行：`percent` + 已写字节 + 输入总字节 + 媒体时间。
///
/// 分母优先用**输入总字节**：合并是"只搬字节"，产物字节量与输入字节量线性相关，
/// 而媒体时长在分片式 fMP4 上常常探不出来（`mvhd.duration = 0`）——
/// 用时长当分母会让进度条永远停在 0。
fn progress_line(
    media_us: i64,
    declared_us: i64,
    written: u64,
    input_bytes: u64,
) -> serde_json::Value {
    let percent = if input_bytes > 0 {
        (written as f64 / input_bytes as f64) * 100.0
    } else if declared_us > 0 {
        (media_us as f64 / declared_us as f64) * 100.0
    } else {
        0.0
    };
    serde_json::json!({
        "percent": percent.clamp(0.0, 100.0),
        "outputBytes": written,
        "inputBytes": input_bytes,
        "mediaTimeUs": media_us,
        "durationUs": if declared_us > 0 { serde_json::json!(declared_us) } else { serde_json::Value::Null },
    })
}

pub fn outcome_json(o: &MuxOutcome) -> serde_json::Value {
    serde_json::json!({
        "output": o.output,
        "container": o.container,
        "bytes": o.bytes,
        "packets": o.packets,
        "inputs": o.inputs,
        "durationUs": o.duration_us,
        "durationSecs": o.duration_us.map(|u| u as f64 / 1_000_000.0),
        "tracks": o.tracks.iter().map(|t| serde_json::json!({
            "id": t.id,
            "kind": t.kind.as_i32(),
            "kindName": match t.kind {
                MediaKind::Video => "video",
                MediaKind::Audio => "audio",
                MediaKind::Subtitle => "subtitle",
                MediaKind::Other => "other",
            },
            "codec": t.codec.name(),
            "width": t.width,
            "height": t.height,
            "sampleRate": t.sample_rate,
            "channels": t.channels,
            "language": t.language,
        })).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use me_core::{CodecId, FileSource};
    use me_stream::CancelFlag;
    use std::path::Path;

    // ─────────────────────────────────────────────────────────────
    // 测试媒体：用引擎自己的封装器造 fMP4 输入，再让合并器去合。
    // 刻意走**真实的文件 + 真实的解封装**（而不是内存里喂 Packet），
    // 因为"多输入怎么变成一条流"这件事只在真实链路上才成立。
    // ─────────────────────────────────────────────────────────────

    fn video_track() -> Track {
        let mut t = Track::new(1, MediaKind::Video, CodecId::H264, Timebase::new(30_000));
        t.width = 640;
        t.height = 360;
        // Annex-B 形态的 extradata（SPS + PPS），封装器会转成 avcC
        let mut ed = vec![0u8, 0, 0, 1];
        ed.extend_from_slice(&[
            0x67, 0x42, 0x00, 0x1E, 0x95, 0xA8, 0x28, 0x0F, 0x00, 0x44, 0xFC, 0xB8, 0x08, 0x80,
            0x00, 0x03, 0x00, 0x08, 0x00, 0x00, 0x03, 0x01, 0xE0, 0x78, 0xB1, 0x6C, 0x90,
        ]);
        ed.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE, 0x3C, 0x80]);
        t.extradata = ed;
        t
    }

    fn audio_track() -> Track {
        // 轨道号 2：同一个容器里两条轨的号必须不同（容器保证唯一）
        let mut t = Track::new(2, MediaKind::Audio, CodecId::Aac, Timebase::new(44_100));
        t.sample_rate = 44_100;
        t.channels = 2;
        t.extradata = vec![0x12, 0x10];
        t
    }

    fn video_packet(dts: i64, keyframe: bool) -> Packet {
        let mut v = vec![0u8, 0, 0, 1];
        v.push(if keyframe { 0x65 } else { 0x41 });
        v.extend_from_slice(&[0x11, 0x22, 0x33, 0x44]);
        let mut p = Packet::new(1, dts, dts, 1000, v);
        p.keyframe = keyframe;
        p
    }

    fn audio_packet(dts: i64) -> Packet {
        Packet::new(2, dts, dts, 1024, vec![0x21, 0x10, 0x05, 0x55])
    }

    /// 只含画面的一路（`seconds` 秒 / 30fps）。
    fn write_video_only(path: &Path, seconds: usize) {
        let mut m = Mp4Muxer::new();
        m.set_fragment_duration_us(1_000_000);
        m.set_tracks(&[video_track()]).unwrap();
        for i in 0..seconds * 30 {
            m.write_packet(&video_packet(i as i64 * 1000, i % 30 == 0)).unwrap();
        }
        std::fs::write(path, m.finish().unwrap()).unwrap();
    }

    /// 只含声音的一路（`seconds` 秒 / 43 帧每秒）。
    fn write_audio_only(path: &Path, seconds: usize) {
        let mut m = Mp4Muxer::new();
        m.set_fragment_duration_us(1_000_000);
        m.set_tracks(&[audio_track()]).unwrap();
        for i in 0..seconds * 43 {
            m.write_packet(&audio_packet(i as i64 * 1024)).unwrap();
        }
        std::fs::write(path, m.finish().unwrap()).unwrap();
    }

    /// 一个**音视频都有**的单文件（重封装场景）。
    fn write_av(path: &Path, seconds: usize) {
        let mut m = Mp4Muxer::new();
        m.set_fragment_duration_us(1_000_000);
        m.set_tracks(&[video_track(), audio_track()]).unwrap();
        for f in 0..seconds {
            for i in 0..30 {
                let dts = (f * 30 + i) as i64 * 1000;
                m.write_packet(&video_packet(dts, i == 0)).unwrap();
            }
            for i in 0..43 {
                m.write_packet(&audio_packet((f * 43 + i) as i64 * 1024)).unwrap();
            }
        }
        std::fs::write(path, m.finish().unwrap()).unwrap();
    }

    /// AV1 的 av1C 载荷：4 字节头 + sequence header OBU 的载荷。
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

    /// 一条 AV1 采样：OBU 流（**没有起始码**，这是它和 H.264/H.265 的关键区别）。
    fn av1_packet(dts: i64, keyframe: bool) -> Packet {
        let mut v = vec![0x12, 0x00, 0x32, 0x05];
        v.extend_from_slice(if keyframe { &[0x11, 0x22, 0x33, 0x44, 0x55] } else { &[0xAA; 5] });
        let mut p = Packet::new(1, dts, dts, 1000, v);
        p.keyframe = keyframe;
        p
    }

    /// 只含 AV1 画面的一路（B 站等站点已经在用的编码）。
    fn write_av1_only(path: &Path, seconds: usize) {
        let mut m = Mp4Muxer::new();
        m.set_fragment_duration_us(1_000_000);
        m.set_tracks(&[av1_track()]).unwrap();
        for i in 0..seconds * 30 {
            m.write_packet(&av1_packet(i as i64 * 1000, i % 30 == 0)).unwrap();
        }
        std::fs::write(path, m.finish().unwrap()).unwrap();
    }

    /// FLAC 的 `dfLa` 载荷（version/flags + STREAMINFO metadata block）。
    fn flac_config(sample_rate: u32) -> Vec<u8> {
        let mut v = vec![0, 0, 0, 0];
        v.extend_from_slice(&[0x80, 0, 0, 34]); // last + STREAMINFO(34)
        let mut si = vec![0u8; 34];
        si[10] = ((sample_rate >> 12) & 0xFF) as u8;
        si[11] = ((sample_rate >> 4) & 0xFF) as u8;
        si[12] = (((sample_rate & 0x0F) << 4) as u8) | (1 << 1); // channels-1 = 1
        v.extend_from_slice(&si);
        v
    }

    fn flac_track(sample_rate: u32) -> Track {
        let mut t = Track::new(2, MediaKind::Audio, CodecId::Flac, Timebase::new(sample_rate));
        t.sample_rate = sample_rate;
        t.channels = 2;
        t.extradata = flac_config(sample_rate);
        t
    }

    fn flac_packet(dts: i64) -> Packet {
        Packet::new(2, dts, dts, 4096, vec![0xFF, 0xF8, 0x69, 0x08, 0x00, 0x11, 0x22])
    }

    /// 只含 FLAC（B 站"Hi-Res 无损音质"）声音的一路。
    fn write_flac_only(path: &Path, seconds: usize, sample_rate: u32) {
        let mut m = Mp4Muxer::new();
        m.set_fragment_duration_us(1_000_000);
        m.set_tracks(&[flac_track(sample_rate)]).unwrap();
        let frames = (seconds as u64 * sample_rate as u64 / 4096) as usize;
        for i in 0..frames {
            m.write_packet(&flac_packet(i as i64 * 4096)).unwrap();
        }
        std::fs::write(path, m.finish().unwrap()).unwrap();
    }

    fn open_src(path: &Path) -> Arc<dyn ReadAt> {
        Arc::new(FileSource::open(path).expect("打开测试输入"))
    }

    fn info_of(path: &Path) -> MediaInfo {
        let src = open_src(path);
        me_container::probe(src, &path.to_string_lossy()).expect("探测测试输入")
    }

    /// 每个用例一个独立临时目录（进程号 + 纳秒，避免并行测试互相踩）。
    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir()
            .join(format!("zuvrust-mux-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("建临时目录");
        dir
    }

    fn merge(dir: &Path, name: &str, inputs: &[&Path]) -> MuxOutcome {
        let owned: Vec<(String, Arc<dyn ReadAt>, MediaInfo)> = inputs
            .iter()
            .map(|p| (p.to_string_lossy().to_string(), open_src(p), info_of(p)))
            .collect();
        let refs: Vec<(&str, Arc<dyn ReadAt>, MediaInfo)> = owned
            .iter()
            .map(|(n, s, i)| (n.as_str(), s.clone(), i.clone()))
            .collect();
        let output = dir.join(name);
        mux_to_file(&output.to_string_lossy(), &refs, MergeOptions::default(), false, &CancelFlag::new())
            .expect("合并应当成功")
    }

    /// 读回产物的某条轨的 dts 序列（按输出轨号过滤）。
    fn dts_of(path: &Path, track_id: u32) -> Vec<i64> {
        let src = open_src(path);
        let mut d = me_container::demux::open(src, &path.to_string_lossy()).expect("打开产物");
        let mut out = Vec::new();
        while let Some(p) = d.next_packet().expect("读产物采样") {
            if p.track_id == track_id {
                out.push(p.dts);
            }
        }
        out
    }

    #[test]
    fn merges_audio_and_video_from_separate_inputs() {
        let dir = tmp_dir("pair");
        let v = dir.join("v.mp4");
        let a = dir.join("a.mp4");
        write_video_only(&v, 2);
        write_audio_only(&a, 2);

        let out = merge(&dir, "out.mp4", &[&v, &a]);
        assert_eq!(out.container, "fmp4");
        assert_eq!(out.tracks.len(), 2, "产物必须同时有画面与声音");
        assert_eq!(out.inputs, 2);

        let info = info_of(&dir.join("out.mp4"));
        let d = info.duration_us.unwrap_or(0);
        assert!((d - 2_000_000).abs() < 200_000, "时长应当在 2 秒附近，实际 {d}");
        // 两条轨都要能被重新解出来（不是只写了轨头）
        assert!(!dts_of(&dir.join("out.mp4"), 1).is_empty());
        assert!(!dts_of(&dir.join("out.mp4"), 2).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn concatenates_multiple_segments_into_one_timeline() {
        let dir = tmp_dir("concat");
        let (v1, v2) = (dir.join("v1.mp4"), dir.join("v2.mp4"));
        let (a1, a2) = (dir.join("a1.mp4"), dir.join("a2.mp4"));
        write_video_only(&v1, 1);
        write_video_only(&v2, 1);
        write_audio_only(&a1, 1);
        write_audio_only(&a2, 1);

        // 画面序列 = [v1, v2]，声音序列 = [a1, a2]
        let out = merge(&dir, "joined.mp4", &[&v1, &v2, &a1, &a2]);
        assert_eq!(out.tracks.len(), 2);

        let path = dir.join("joined.mp4");
        let vd = dts_of(&path, 1);
        let ad = dts_of(&path, 2);
        // 关键断言：第二段必须**接在第一段之后**，而不是各自从 0 重来
        assert!(
            vd.windows(2).all(|w| w[1] > w[0]),
            "画面时间戳必须严格递增（分段串接后不许回到 0）：{vd:?}"
        );
        assert!(
            ad.windows(2).all(|w| w[1] > w[0]),
            "声音时间戳必须严格递增：{ad:?}"
        );
        // 2 秒 = 60000 刻度（30kHz 时基），末尾应当落在它附近
        let v_last = *vd.last().expect("画面不能为空");
        assert!(v_last >= 58_000, "画面末尾必须到 2 秒附近，实际 {v_last}");
        let a_last = *ad.last().expect("声音不能为空");
        assert!(a_last >= 44_100 * 2 - 2048, "声音末尾必须到 2 秒附近，实际 {a_last}");

        let info = info_of(&path);
        let d = info.duration_us.unwrap_or(0);
        assert!((d - 2_000_000).abs() < 200_000, "串接后总时长应当在 2 秒附近，实际 {d}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn single_input_remux_keeps_both_tracks() {
        let dir = tmp_dir("remux");
        let src = dir.join("in.mp4");
        write_av(&src, 2);

        // 单输入：两条轨都来自同一个文件（不能把它读两遍导致声音丢失）
        let out = merge(&dir, "out.mp4", &[&src]);
        assert_eq!(out.tracks.len(), 2, "单文件重封装必须保住两条轨");
        let path = dir.join("out.mp4");
        assert_eq!(dts_of(&path, 1).len(), 60, "画面采样一条都不能丢");
        assert_eq!(dts_of(&path, 2).len(), 86, "声音采样一条都不能丢");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn av1_video_merges_with_aac_audio() {
        // B 站常见组合：AV1 画面 + AAC 声音。以前这里会以
        // "fmp4 装不下这条轨道：av1 / Video" 直接失败 —— 用户实测报的就是这个。
        let dir = tmp_dir("av1");
        let v = dir.join("v.mp4");
        let a = dir.join("a.mp4");
        write_av1_only(&v, 2);
        write_audio_only(&a, 2);

        let out = merge(&dir, "out.mp4", &[&v, &a]);
        assert_eq!(out.tracks.len(), 2, "AV1 画面与 AAC 声音都要在");

        let path = dir.join("out.mp4");
        let info = info_of(&path);
        let video = info
            .tracks
            .iter()
            .find(|t| t.kind == MediaKind::Video)
            .expect("产物要有画面轨");
        assert_eq!(video.codec, CodecId::Av1, "AV1 必须原样保留（引擎不转码）");
        assert_eq!(video.extradata, av1_config(), "av1C 配置要原样过去");
        assert_eq!((video.width, video.height), (1920, 1080));
        assert_eq!(dts_of(&path, 1).len(), 60, "AV1 采样一条都不能丢");

        let d = info.duration_us.unwrap_or(0);
        assert!((d - 2_000_000).abs() < 200_000, "时长应当在 2 秒附近，实际 {d}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn av1_segments_concatenate_like_any_other_video() {
        // 串接逻辑与编码无关：两个 AV1 分片接起来，时间轴必须连续
        let dir = tmp_dir("av1-concat");
        let (v1, v2) = (dir.join("v1.mp4"), dir.join("v2.mp4"));
        let (a1, a2) = (dir.join("a1.mp4"), dir.join("a2.mp4"));
        write_av1_only(&v1, 1);
        write_av1_only(&v2, 1);
        write_audio_only(&a1, 1);
        write_audio_only(&a2, 1);

        let out = merge(&dir, "joined.mp4", &[&v1, &v2, &a1, &a2]);
        assert_eq!(out.tracks.len(), 2);
        let path = dir.join("joined.mp4");
        let vd = dts_of(&path, 1);
        assert!(vd.windows(2).all(|w| w[1] > w[0]), "AV1 时间戳必须严格递增：{vd:?}");
        assert_eq!(vd.len(), 60, "两段各 30 帧，一条都不能少");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn av1_with_hires_flac_audio_merges() {
        // 用户实测那条 B 站视频的真实组合：AV1 画面 + Hi-Res 无损音质（FLAC 192k）。
        // 修 AV1 之前这里会卡在 "fmp4 装不下这条轨道：av1 / Video"。
        let dir = tmp_dir("av1-flac");
        let v = dir.join("v.mp4");
        let a = dir.join("a.mp4");
        write_av1_only(&v, 2);
        write_flac_only(&a, 2, 192_000);

        let out = merge(&dir, "out.mp4", &[&v, &a]);
        assert_eq!(out.tracks.len(), 2, "AV1 画面与 FLAC 声音都要在");

        let info = info_of(&dir.join("out.mp4"));
        let video = info
            .tracks
            .iter()
            .find(|t| t.kind == MediaKind::Video)
            .expect("要有画面轨");
        assert_eq!(video.codec, CodecId::Av1, "AV1 原样保留");
        let audio = info
            .tracks
            .iter()
            .find(|t| t.kind == MediaKind::Audio)
            .expect("要有声音轨");
        assert_eq!(audio.codec, CodecId::Flac, "FLAC 原样保留");
        // Hi-Res 采样率靠 dfLa 的 STREAMINFO：AudioSampleEntry 那个字段装不下 192k
        assert_eq!(audio.sample_rate, 192_000, "Hi-Res 采样率必须保住（不能溢出成怪值）");
        assert_eq!(audio.extradata, flac_config(192_000), "dfLa 配置要原样往返");
        assert_eq!(audio.channels, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn audio_and_video_are_interleaved_not_stacked() {
        // 回归保护：段内曾经写成"先把视频输入读完、再读音频输入"，结果音频的
        // **全部**采样被堆进最后一个分片 —— 产物变成"视频全在前、音频全在文件末尾"，
        // 播放时前面大半段没有声音。判据：产物开头的采样里两条轨都得出现。
        let dir = tmp_dir("interleave");
        let v = dir.join("v.mp4");
        let a = dir.join("a.mp4");
        write_video_only(&v, 3);
        write_audio_only(&a, 3);
        let out = merge(&dir, "out.mp4", &[&v, &a]);
        assert_eq!(out.tracks.len(), 2);

        let path = dir.join("out.mp4");
        let src = open_src(&path);
        let mut d = me_container::demux::open(src, &path.to_string_lossy()).expect("打开产物");
        let mut seen: Vec<u32> = Vec::new();
        while let Some(p) = d.next_packet().expect("读产物采样") {
            seen.push(p.track_id);
            if seen.len() >= 200 {
                break;
            }
        }
        let first_audio = seen.iter().position(|t| *t == 2);
        assert!(
            seen.iter().any(|t| *t == 1),
            "开头必须有画面采样：{:?}",
            &seen[..seen.len().min(20)]
        );
        assert!(
            first_audio.map(|i| i < 100).unwrap_or(false),
            "音频必须很早就出现（被堆到文件末尾了？），实际首次出现在第 {:?} 条",
            first_audio
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn container_is_decided_by_extension() {
        assert_eq!(OutputContainer::from_path("a.mp4").unwrap(), OutputContainer::Fmp4);
        assert_eq!(OutputContainer::from_path("a.M4S").unwrap(), OutputContainer::Fmp4);
        assert_eq!(OutputContainer::from_path("a.ts").unwrap(), OutputContainer::MpegTs);
        assert!(OutputContainer::from_path("a.avi").is_err());
    }

    #[test]
    fn percent_falls_back_to_media_time_when_input_size_is_unknown() {
        // 输入大小未知（BT 边下边播的常态）：不能因此永远显示 0
        let v = progress_line(5_000_000, 10_000_000, 1024, 0);
        assert_eq!(v["percent"], 50.0);
        // 输入大小已知时以它为分母：只搬字节的任务里它最准
        let v = progress_line(5_000_000, 10_000_000, 50, 200);
        assert_eq!(v["percent"], 25.0);
        assert_eq!(v["inputBytes"], 200);
    }

    #[test]
    fn percent_never_exceeds_100() {
        // 产物比输入大是正常现象（fMP4 的 moof/索引开销），进度条不能因此爆表
        let v = progress_line(0, 0, 120, 100);
        assert_eq!(v["percent"], 100.0);
    }

    #[test]
    fn shift_is_lossless_when_timebases_match() {
        let tb = Timebase::new(15_360);
        // 段起点 2 秒、段内第 3 条采样（相对首条 2 个刻度）
        let got = shift_ticks(tb, tb, 3_000 + 2, 3_000, 2_000_000);
        assert_eq!(got, 2 * 15_360 + 2, "同基必须走纯刻度路径，不能经微秒取整");
    }

    #[test]
    fn shift_converts_across_timebases() {
        let src = Timebase::new(1_000);
        let out = Timebase::new(90_000);
        // 段起点 1 秒 + 段内 0.5 秒 → 输出刻度 135000
        assert_eq!(shift_ticks(src, out, 500, 0, 1_000_000), 135_000);
    }
}
