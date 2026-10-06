//! 流式播放：把一条输入变成**边读边吐的分片 fMP4**，交给宿主播放器的解码器。
//!
//! ## 为什么播放内核必须是引擎，而不是浏览器的 `<video>`
//!
//! 直接让 `<video>` 去啃文件（`media.src = http://…/裸文件`）只在一种情况下能用：
//! 文件完整、容器与编码都在 Chromium 的支持面里。真实场景几乎都落在外面：
//!
//! | 场景 | 直接丢给 `<video>` 的结果 |
//! |---|---|
//! | HLS 的 `.ts` 分片 | 认不出容器 → 直接报"无法播放" |
//! | fMP4 分片（`.m4s`，B 站 / DASH） | `moov` 在别处，单看分片是一段无索引的字节 |
//! | 还在下载的文件 | 中间有空洞，播放器把空洞当文件末尾 |
//! | `moov` 在文件末尾的 MP4 | 边下边播时读不到索引 |
//!
//! 所以分工改成：**引擎负责容器解析 / 边等边读 / 寻址 / 转封装成 fMP4，
//! 浏览器的解码器只负责解码**。解码本就是引擎之外的一层（见 `me-core` 的
//! `CodecId` 约定：它只描述"是什么"，不含解码器）。
//!
//! ## 输出协议（与宿主的契约）
//!
//! ```text
//! stdout: {"ok":true,"type":"stream-meta","data":{…}} \n    ← 恰好一行 JSON
//!         [ ftyp + moov ][ moof + mdat ][ moof + mdat ]…     ← 之后全是 fMP4 字节
//! stderr: {"ok":true,"type":"playhead","data":{…}} \n        ← 播放头/缓冲，NDJSON
//! ```
//!
//! 元信息走 stdout 的第一行（而不是 stderr）：宿主读一行就能拿到
//! `codecs` 去建 `SourceBuffer`，之后**同一个管道里剩下的全是媒体字节**，
//! 不用第二个请求、也没有"元信息还没到就先 append 了"的竞态。
//!
//! ## 时间轴
//!
//! 每条流内部**从 0 开始**（`--seek` 之后也一样），媒体真实起点由元信息里的
//! `timelineOffsetUs` 告诉宿主，宿主用 `SourceBuffer.timestampOffset` 映射回去。
//! 这样引擎侧永远不需要关心"播放器的进度条要显示什么"，
//! 而宿主换来的是**正确的 `currentTime` 与进度条**。

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use me_core::{CodecId, MediaError, MediaKind, ReadAt, Result, Timebase, Track};
use me_mux::convert;
use me_mux::Mp4Muxer;
use me_stream::{OpenOptions, OpenedSource, Player, PlayerConfig};

use crate::json;
use serde_json::json;

/// 一次流式播放的选项。
pub struct StreamOptions {
    /// **出帧模式**：不出 fMP4，改成把视频轨解成 NV12 帧写 stdout。
    ///
    /// 为什么放在同一条子命令里而不是新开一条：宿主侧的通路（本地 HTTP →
    /// 前端页面）完全一样 —— 前端读元信息里的 `mime`（`video/x-nv12`）就知道
    /// 该用帧渲染器，传输层一行都不用改。
    pub video_frames: bool,
    /// **出音频**：把音频轨解成 PCM（S16LE）与画面同一条流里发出去。
    ///
    /// 与 `video_frames` 一起用：前端拿到的是"带类型标签的包流"（见 `KIND_*`），
    /// 画面画到 canvas、PCM 喂给音频输出 —— 引擎自己解码，前端只负责放。
    pub audio_pcm: bool,
    /// 从哪一刻开始（微秒）。`None` = 从能找到的第一个关键帧开始。
    pub seek_us: Option<i64>,
    /// 分片目标时长（毫秒）。太大则出画慢，太小则文件头开销占比高。
    pub fragment_ms: u64,
    pub open: OpenOptions,
    /// 是否往 stderr 写播放头（宿主用它驱动"优先下载播放位置附近"）。
    pub progress: bool,
}

/// 流式播放到 stdout，直到输入读完（或宿主关闭管道）。
pub fn stream_to_stdout(input: &str, opts: StreamOptions) -> Result<u8> {
    if opts.video_frames {
        return frames_to_stdout(input, opts);
    }
    // 本地文件必须先存在。
    //
    // 为什么单独判：边下边播用的「增长文件源」**故意容忍文件还不存在**
    // （下载还没建文件时它是合法的等待状态）。但"路径写错了"与"文件还没建"
    // 在宿主看来完全不同 —— 前者要立刻报错说清原因，否则用户看到的是
    // "无法识别的容器"这种没用的信息。
    if matches!(me_stream::classify(input), me_stream::InputKind::File)
        && !std::path::Path::new(input).exists()
    {
        return Err(MediaError::Io(format!("文件不存在：{input}")));
    }
    let client = me_stream::default_client()?;
    let opened = me_stream::open_input(&client, input, &opts.open)?;
    let (src, dash_note) = pick_source(opened)?;

    // 播放缓冲给得小一些：流式播放的"缓冲"是管道对端的 MSE，
    // 引擎这边多攒一秒数据只会让内存白占、延迟变高
    let cfg = PlayerConfig { max_buffer_us: 1_000_000, max_buffer_packets: 1024 };
    let mut player = Player::open(src, input, cfg)?;
    let info = player.info();

    // ── 选轨：一条画面 + 一条声音（与 `mux` 同一套判据） ──
    let mut out_tracks: Vec<Track> = Vec::new();
    let mut map: HashMap<u32, u32> = HashMap::new();
    let mut next_id = 1u32;
    if let Some(t) = info.tracks.iter().find(|t| t.kind == MediaKind::Video) {
        let mut ot = t.clone();
        ot.id = next_id;
        next_id += 1;
        map.insert(t.id, ot.id);
        out_tracks.push(ot);
    }
    if let Some(t) = info.tracks.iter().find(|t| t.kind == MediaKind::Audio) {
        let mut ot = t.clone();
        ot.id = next_id;
        map.insert(t.id, ot.id);
        out_tracks.push(ot);
    }
    if out_tracks.is_empty() {
        return Err(MediaError::Container(format!(
            "{input}: 这条输入里没有可播的轨道"
        )));
    }

    // ── 能不能流式（不能就报错，让宿主回退到直接播放） ──
    let mime = check_and_build_mime(&out_tracks).map_err(|e| match e {
        MediaError::Unsupported(m) => MediaError::Unsupported(format!("{input}: {m}")),
        other => other,
    })?;

    // 声音装不进 fMP4（MP3 / Opus / FLAC…）时**只丢声音、保住画面**：
    // 能看画面总比整个播不了强，而且这件事要如实报给宿主。
    let dropped_audio = out_tracks
        .iter()
        .find(|t| t.kind == MediaKind::Audio && t.codec != CodecId::Aac)
        .map(|t| t.codec.name());
    if dropped_audio.is_some() {
        let audio_out = out_tracks.iter().find(|t| t.kind == MediaKind::Audio).map(|t| t.id);
        if let Some(id) = audio_out {
            map.retain(|_, v| *v != id);
        }
        out_tracks.retain(|t| t.kind != MediaKind::Audio);
    }

    let duration_us = player.duration_us().or(info.duration_us);
    // 源轨时基：把采样的 dts 换算成"绝对媒体时间"要用它
    let src_timebases: HashMap<u32, Timebase> =
        info.tracks.iter().map(|t| (t.id, t.timebase)).collect();

    // ── 这条流的起点（绝对媒体时间） ──
    //
    // **不能假定它从 0 开始**：真实流的第一个时间戳常常不是 0（实测 Apple 的
    // 分片就从 10 秒起）。流内部会统一归零，而这个"起点"必须如实告诉宿主 ——
    // 它用 `SourceBuffer.timestampOffset` 把流映射回真正的媒体时间轴。
    // 报告错了的表现是"进度条长度不对 + 一开播就显示结束"，很难查。
    let mut peeked: Option<me_core::Packet> = None;
    let start_us = match opts.seek_us {
        Some(us) => player.seek_us(us)?,
        None => match player.next_packet()? {
            Some(p) => {
                let us = src_timebases
                    .get(&p.track_id)
                    .map(|tb| tb.ticks_to_us(p.dts))
                    .unwrap_or(0);
                peeked = Some(p);
                us
            }
            None => 0,
        },
    };

    // 元信息行：**必须在 ftyp 之前**吐出去
    let meta = meta_json(&info, &out_tracks, &mime, duration_us, start_us, opts.fragment_ms, dash_note, dropped_audio);
    println!("{}", json::progress_line("stream-meta", meta));
    std::io::stdout().flush()?;

    // ── 产物直写 stdout ──
    // 用 `Stdout`（行缓冲）而不是自己开 fd：每次分片落盘后会显式 flush，
    // 所以缓冲不会让数据滞留；而 stdout 是 `Send` 的，正好满足 `Sink` 的要求。
    let mut muxer = Mp4Muxer::with_writer(Box::new(std::io::stdout()));
    muxer.set_fragment_duration_us((opts.fragment_ms.max(20) as i64) * 1000);
    // 时间轴从 0 起（真实起点已写进元信息）
    muxer.set_timeline_base_us(0);
    muxer.set_tracks(&out_tracks)?;
    muxer.flush_output()?;

    let mut fragments = muxer.fragments_written();
    let mut packets = 0u64;
    let mut need_more_retries = 0u32;
    let stderr = std::io::stderr();

    loop {
        let p = match match peeked.take() {
            Some(p) => Ok(Some(p)),
            None => player.next_packet(),
        } {
            Ok(Some(p)) => p,
            Ok(None) => break,
            // 数据还没到（增长文件等超时 / 网络卡住）：如实报"缓冲中"并重试，
            // 而不是把整条流判死 —— 这是边下边播的正常路径
            Err(e @ MediaError::NeedMore { .. }) => {
                need_more_retries += 1;
                if need_more_retries > NEED_MORE_MAX_RETRIES {
                    return Err(e);
                }
                report_playhead(&stderr, opts.progress, &player, packets, muxer.bytes_written(), "buffering");
                std::thread::sleep(std::time::Duration::from_millis(500));
                continue;
            }
            Err(e) => return Err(e),
        };
        need_more_retries = 0;

        let Some(&out_id) = map.get(&p.track_id) else {
            continue; // 被丢掉的轨（声音装不进 fMP4）与字幕等
        };
        let mut q = p;
        q.track_id = out_id;
        // 时间轴归零：以这条流的起点为 0（宿主用 timestampOffset 映射回媒体时间）
        if start_us != 0 {
            let tb = out_tracks
                .iter()
                .find(|t| t.id == out_id)
                .map(|t| t.timebase)
                .unwrap_or(Timebase::MPEG_TS);
            let shift = tb.us_to_ticks(start_us);
            q.pts -= shift;
            q.dts -= shift;
        }

        if let Err(e) = muxer.write_packet(&q) {
            // 宿主主动关掉管道（换了 seek 位置 / 关窗口）：正常结束，不是错误
            if is_broken_pipe(&e) {
                return Ok(0);
            }
            return Err(e);
        }
        packets += 1;

        // 凑够一个分片就刷出去 —— 这是"边读边播"的关键：
        // 不刷的话数据留在缓冲里，播放器那边就是"一直黑屏"
        if muxer.fragments_written() != fragments {
            fragments = muxer.fragments_written();
            if let Err(e) = muxer.flush_output() {
                if is_broken_pipe(&e) {
                    return Ok(0);
                }
                return Err(e);
            }
            report_playhead(&stderr, opts.progress, &player, packets, muxer.bytes_written(), "playing");
        }
    }

    // 收尾：最后半个分片也要吐出去，否则尾部一小段播不到
    muxer.finish()?;
    report_playhead(&stderr, opts.progress, &player, packets, muxer.bytes_written(), "eof");
    Ok(0)
}

/// 数据没到时的最大重试次数（每次间隔 500ms）。
///
/// 有上限是必须的：宿主若已经放弃（进程没被杀但没人读了），无限重试会变成
/// 一个永远不退出的僵尸进程。3 分钟足够覆盖"下载卡一下"的正常情形。
const NEED_MORE_MAX_RETRIES: u32 = 360;

/// 把打开结果收敛成"一条字节流 + 需要告知宿主的备注"。
pub(crate) fn pick_source(opened: OpenedSource) -> Result<(Arc<dyn ReadAt>, Option<String>)> {
    match opened {
        OpenedSource::Single(s) => Ok((s, None)),
        OpenedSource::Hls(h) => {
            let note = if h.is_live {
                Some("直播流：本版按当前快照播放，不做时移".to_string())
            } else {
                None
            };
            Ok((h.source, note))
        }
        OpenedSource::DashTracks(list) => {
            // DASH 的视频与音频是两条独立字节流，播放窗一次只播一条。
            if list.len() == 1 {
                let t = list.into_iter().next().expect("刚判过");
                return Ok((t.source, None));
            }
            let has_video = list.iter().any(|t| t.content_type == "video");
            let note = format!(
                "这份 DASH 有 {} 条轨（视频与音频分开），本流只含{}；要带声音请先用 mux 合并",
                list.len(),
                if has_video { "视频" } else { "音频" }
            );
            // 有画面优先：画面没声音还能看，只有声音算怎么回事
            let pick = list.iter().position(|t| t.content_type == "video").unwrap_or(0);
            let mut list = list;
            let t = list.swap_remove(pick);
            Ok((t.source, Some(note)))
        }
    }
}

/// 流式播放只支持 fMP4 装得下的编解码器；顺便拼出给 `SourceBuffer` 的 MIME。
///
/// 装不下就返回 `Unsupported`（宿主据此回退到直接播放），**绝不硬凑**：
/// 硬把 MP3 写成 fMP4 的 `mp4a` 只会得到一条解不出声的轨道。
fn check_and_build_mime(tracks: &[Track]) -> Result<String> {
    let mut codecs: Vec<String> = Vec::new();
    let mut has_video = false;
    let mut has_audio = false;
    for t in tracks {
        match t.kind {
            MediaKind::Video => {
                has_video = true;
                codecs.push(video_codec_string(&t.codec, &t.extradata)?);
            }
            MediaKind::Audio => {
                if t.codec != CodecId::Aac {
                    // 由调用方决定"丢声音保画面"，这里不报错
                    continue;
                }
                has_audio = true;
                codecs.push(aac_codec_string(&t.extradata));
            }
            _ => {}
        }
    }
    if codecs.is_empty() {
        return Err(MediaError::Unsupported(
            "没有能装进 fMP4 的轨道（视频限 H.264/H.265/AV1，声音限 AAC）".into(),
        ));
    }
    let base = if has_video {
        "video/mp4"
    } else if has_audio {
        "audio/mp4"
    } else {
        "video/mp4"
    };
    Ok(format!("{base}; codecs=\"{}\"", codecs.join(",")))
}

/// 视频编解码器字符串（`SourceBuffer` 的 `codecs` 参数）。
///
/// 这几个字符串是**浏览器的契约**：写错了 `addSourceBuffer` 直接抛
/// `NotSupportedError`，播放不成立。所以每个字节都从 `extradata` 里取，
/// 不猜、不省略。
fn video_codec_string(codec: &CodecId, extradata: &[u8]) -> Result<String> {
    match codec {
        CodecId::H264 => {
            // avcC[1..4] = profile / compatibility / level
            let c = convert::extradata_to_avcc(codec, extradata, 4)?;
            if c.len() < 4 {
                return Err(MediaError::Bitstream("avcC 太短，取不到 profile/level".into()));
            }
            Ok(format!("avc1.{:02X}{:02X}{:02X}", c[1], c[2], c[3]))
        }
        CodecId::H265 => {
            let c = convert::extradata_to_avcc(codec, extradata, 4)?;
            if c.len() < 13 {
                return Err(MediaError::Bitstream("hvcC 太短，取不到 profile/level".into()));
            }
            let b = c[1];
            let space = b >> 6;
            let tier = (b >> 5) & 0x01;
            let idc = b & 0x1F;
            let compat = u32::from_be_bytes([c[2], c[3], c[4], c[5]]);
            let level = c[12];
            // 约束标志（48 位）按规范写在最后，去掉尾部多余的 0 更短更常见
            let constraints: Vec<u8> = c[6..12].to_vec();
            let mut tail = constraints.as_slice();
            while let Some((&last, rest)) = tail.split_last() {
                if last == 0 {
                    tail = rest;
                } else {
                    break;
                }
            }
            let constraint_hex: String = tail.iter().map(|b| format!("{b:02X}")).collect();
            let t = if tier == 1 { 'H' } else { 'L' };
            let base = format!("hvc1.{space}{idc}.{compat:X}.{t}{level}");
            Ok(if constraint_hex.is_empty() {
                base
            } else {
                format!("{base}.{constraint_hex}")
            })
        }
        CodecId::Av1 => {
            // av1C: [1] = marker/version, [2] = seq_profile(3) | seq_level_idx(5),
            //       [3] = seq_tier(1) | high_bitdepth(1) | twelve_bit(1) | …
            if extradata.len() < 4 || extradata[0] != 0x81 {
                return Err(MediaError::Bitstream("av1C 不合法，取不到 profile/level".into()));
            }
            let profile = extradata[2] >> 5;
            let level = extradata[2] & 0x1F;
            let tier = if (extradata[3] >> 7) & 1 == 1 { 'H' } else { 'M' };
            let high = (extradata[3] >> 6) & 1 == 1;
            let twelve = (extradata[3] >> 5) & 1 == 1;
            let depth = if twelve {
                12
            } else if high {
                10
            } else {
                8
            };
            Ok(format!("av01.{profile}.{level:02}{tier}.{depth:02}"))
        }
        other => Err(MediaError::Unsupported(format!(
            "{} 装不进 fMP4（只有 H.264 / H.265 / AV1 有对应的采样描述）",
            other.name()
        ))),
    }
}

/// AAC 的编解码器字符串：`mp4a.40.<objectType>`。
///
/// `objectType` 从 `AudioSpecificConfig` 的前 5 位读；读不到就按最通用的
/// AAC-LC（2）报 —— 这是绝大多数 AAC 的形态，报错反而让本来能播的流播不了。
fn aac_codec_string(extradata: &[u8]) -> String {
    let aot = extradata.first().map(|b| b >> 3).unwrap_or(2);
    if aot == 0 || aot == 31 {
        "mp4a.40.2".to_string()
    } else {
        format!("mp4a.40.{aot}")
    }
}

/// 元信息行（宿主拿它建 `SourceBuffer`、设时长与时间轴偏移）。
#[allow(clippy::too_many_arguments)]
fn meta_json(
    info: &me_core::MediaInfo,
    out_tracks: &[Track],
    mime: &str,
    duration_us: Option<i64>,
    start_us: i64,
    fragment_ms: u64,
    note: Option<String>,
    dropped_audio: Option<String>,
) -> serde_json::Value {
    let tracks: Vec<serde_json::Value> = out_tracks
        .iter()
        .map(|t| {
            serde_json::json!({
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
            })
        })
        .collect();
    serde_json::json!({
        "mime": mime,
        "container": info.container,
        "durationUs": duration_us,
        "durationSecs": duration_us.map(|u| u as f64 / 1_000_000.0),
        // 这条流的 0 对应媒体的哪一刻：宿主用 timestampOffset 映射回真实时间轴
        "timelineOffsetUs": start_us,
        "fragmentMs": fragment_ms,
        "tracks": tracks,
        "note": note,
        // 声音装不进 fMP4 被丢掉时如实上报（宿主可以提示用户）
        "droppedAudio": dropped_audio,
    })
}

/// 播放头上报（stderr，NDJSON）。
///
/// 宿主拿 `sourceOffset` 去告诉下载引擎"先下这一带"，拿 `bufferedUs`
/// 画缓冲条 —— 这些都是**引擎量的**，比宿主按码率猜准得多。
fn report_playhead(
    stderr: &std::io::Stderr,
    enabled: bool,
    player: &Player,
    packets: u64,
    bytes_out: u64,
    state: &str,
) {
    if !enabled {
        return;
    }
    let hint = player.playhead_hint();
    let stats = player.stats();
    let line = json::progress_line(
        "playhead",
        serde_json::json!({
            "state": state,
            "packets": packets,
            "playheadUs": stats.playhead_us,
            "sourceOffset": hint.offset_bytes,
            "bufferedPackets": stats.buffered_packets,
            "outputBytes": bytes_out,
        }),
    );
    let mut lock = stderr.lock();
    let _ = writeln!(lock, "{line}");
    let _ = lock.flush();
}

/// 管道被对端关掉（宿主换了 seek 位置 / 关了窗口）。
fn is_broken_pipe(e: &MediaError) -> bool {
    match e {
        MediaError::Io(m) => {
            m.contains("Broken pipe") || m.contains("broken pipe") || m.contains("os error 32")
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track(kind: MediaKind, codec: CodecId, extradata: Vec<u8>) -> Track {
        let mut t = Track::new(1, kind, codec, Timebase::MPEG_TS);
        t.extradata = extradata;
        t
    }

    /// 一个最小的 avcC（profile 0x64 / compat 0x00 / level 0x1F）。
    fn avcc() -> Vec<u8> {
        let mut v = vec![1u8, 0x64, 0x00, 0x1F, 0xFF, 1];
        v.extend_from_slice(&2u16.to_be_bytes());
        v.extend_from_slice(&[0x67, 0x64]);
        v.push(1);
        v.extend_from_slice(&2u16.to_be_bytes());
        v.extend_from_slice(&[0x68, 0xCE]);
        v
    }

    /// Annex-B 形态的参数集（TS 出来的 extradata 就长这样）。
    fn annexb_params() -> Vec<u8> {
        // 复用 mux 里的转换：Annex-B → avcC，再手工拆回 Annex-B 没有意义，
        // 这里直接给一段"能被 split_annexb 拆出 SPS/PPS"的字节
        let mut v = vec![0, 0, 0, 1, 0x67, 0x64, 0x00, 0x1F];
        v.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE, 0x3C, 0x80]);
        v
    }

    #[test]
    fn h264_mime_comes_from_the_avcc_bytes() {
        let s = video_codec_string(&CodecId::H264, &avcc()).unwrap();
        assert_eq!(s, "avc1.64001F");
        // TS 出来的 Annex-B extradata 也要能拼出同样的串（形态靠"能否解析"判定）
        let s2 = video_codec_string(&CodecId::H264, &annexb_params()).unwrap();
        assert!(s2.starts_with("avc1."), "{s2}");
    }

    #[test]
    fn unsupported_video_codec_is_rejected_not_guessed() {
        let e = video_codec_string(&CodecId::Vp9, &[0x01, 0x00]).unwrap_err();
        assert!(matches!(e, MediaError::Unsupported(_)));
        assert!(e.to_string().contains("装不进 fMP4"), "{e}");
    }

    #[test]
    fn aac_object_type_comes_from_asc_and_falls_back() {
        // AAC-LC：aot=2 → asc[0] = 00010xxx
        assert_eq!(aac_codec_string(&[0x12, 0x10]), "mp4a.40.2");
        // aot=5（SBR）：asc[0] = 00101xxx
        assert_eq!(aac_codec_string(&[0x2B, 0x92]), "mp4a.40.5");
        // 空的 / 转义值：回落到 LC，而不是报错（能播总比报错强）
        assert_eq!(aac_codec_string(&[]), "mp4a.40.2");
        assert_eq!(aac_codec_string(&[0xF8, 0x00]), "mp4a.40.2");
    }

    #[test]
    fn mime_joins_video_and_audio_codecs() {
        let tracks = vec![
            track(MediaKind::Video, CodecId::H264, avcc()),
            track(MediaKind::Audio, CodecId::Aac, vec![0x12, 0x10]),
        ];
        let mime = check_and_build_mime(&tracks).unwrap();
        assert_eq!(mime, "video/mp4; codecs=\"avc1.64001F,mp4a.40.2\"");
    }

    #[test]
    fn audio_only_input_gets_an_audio_mime() {
        let tracks = vec![track(MediaKind::Audio, CodecId::Aac, vec![0x12, 0x10])];
        let mime = check_and_build_mime(&tracks).unwrap();
        assert!(mime.starts_with("audio/mp4"), "{mime}");
    }

    #[test]
    fn non_aac_audio_does_not_block_the_video() {
        // MP3 声音装不进 fMP4：不能因此让整条流失败 —— 只丢声音
        let tracks = vec![
            track(MediaKind::Video, CodecId::H264, avcc()),
            track(MediaKind::Audio, CodecId::Mp3, vec![0x00]),
        ];
        let mime = check_and_build_mime(&tracks).unwrap();
        assert_eq!(mime, "video/mp4; codecs=\"avc1.64001F\"");
    }

    #[test]
    fn nothing_streamable_is_an_explicit_error() {
        let tracks = vec![track(MediaKind::Audio, CodecId::Mp3, vec![])];
        let e = check_and_build_mime(&tracks).unwrap_err();
        assert!(matches!(e, MediaError::Unsupported(_)));
    }

    #[test]
    fn broken_pipe_is_recognised() {
        assert!(is_broken_pipe(&MediaError::Io("写产物失败：Broken pipe (os error 32)".into())));
        assert!(!is_broken_pipe(&MediaError::Io("No space left".into())));
        assert!(!is_broken_pipe(&MediaError::container("x")));
    }
}

/// 播放包流：`[28 字节头 + 载荷]` 重复。
///
/// 头是**定长小端**：
/// `magic "MEV1"`(4) `kind`(u32) `a`(u32) `b`(u32) `pts_us`(i64) `len`(u32)。
///
/// - `kind = 0`（画面）：`a` = 宽、`b` = 高，载荷是紧凑 NV12；
/// - `kind = 1`（声音）：`a` = 采样率、`b` = 声道数，载荷是交织 **S16LE**。
///
/// 为什么定长小端：前端（JS）用 DataView 两行就解出来，不需要任何解析库；
/// 为什么带 pts：时钟长在包自己的时间戳上（音频当主时钟、画面跟随）。
pub const FRAME_HEADER_BYTES: usize = 28;
/// 播放包魔数（前端拿它校验自己没跟丢字节）。
pub const FRAME_MAGIC: &[u8; 4] = b"MEV1";
/// 包类型：画面（NV12）。
pub const KIND_VIDEO: u32 = 0;
/// 包类型：声音（交织 S16LE PCM）。
pub const KIND_AUDIO: u32 = 1;

fn frames_to_stdout(input: &str, opts: StreamOptions) -> Result<u8> {
    let client = me_stream::default_client()?;
    let opened = me_stream::open_input(&client, input, &opts.open)?;
    let (src, dash_note) = pick_source(opened)?;
    // ⚠️ 缓冲窗口给到 **6 秒**（原来 1 秒）：见下面"音频优先"的长注释 ——
    // 宿主最多愿意把音频预读到播放头前 4 秒（`MAX_AUDIO_AHEAD_SEC`），
    // 引擎要能把这段音频"读出来"就得先读到它后面的视频采样。1 秒的窗口
    // 会让音频最多只能跑到视频解码进度前 ~1 秒，宿主的 4 秒余量永远填不满。
    // 代价：多缓存 ~6 秒的采样（这条素材 ~0.8 MB）。
    let cfg = PlayerConfig { max_buffer_us: 6_000_000, max_buffer_packets: 4096 };
    let mut player = Player::open(src, input, cfg)?;
    let info = player.info();
    // 视频（可选）：没有视频轨时只发声音 —— "有什么发什么"比"必须有画面"更合用
    let video_track = info.tracks.iter().find(|t| t.kind == MediaKind::Video).cloned();
    let (mut decoder, video_track_id, video_timebase) = match &video_track {
        Some(_) => {
            let (tr, config) = crate::decode::video_track_and_config(&info, input)?;
            (Some(me_decode::open_video_decoder(tr.codec.clone(), &config)?), tr.id, tr.timebase)
        }
        None => (None, 0, me_core::Timebase::new(1_000_000)),
    };

    // 音频（可选）：同一条流里发，前端按 kind 分流
    let audio_track = if opts.audio_pcm {
        info.tracks.iter().find(|t| t.kind == MediaKind::Audio).cloned()
    } else {
        None
    };
    // 音频解码器起不来（编码还没接）**不该带走画面**：降级成"只出画面"，
    // 并在 meta.note 里说清楚少的是什么 —— 否则开了出帧模式放普通视频会整条流失败。
    let mut audio_note: Option<String> = None;
    let mut audio_decoder = match &audio_track {
        Some(t) => match me_decode::audio::open_audio_decoder(t.codec.clone(), &t.extradata) {
            Ok(d) => Some(d),
            Err(e) => {
                audio_note = Some(format!("音频（{}）还不能自研解码，这条流只有画面：{e}", t.codec.name()));
                None
            }
        },
        None => None,
    };
    if video_track.is_none() && audio_decoder.is_none() {
        return Err(MediaError::Container(match &audio_note {
            Some(n) => format!("{input}: 没有可播的轨道（{n}）"),
            None => format!("{input}: 这条输入里没有可播的轨道"),
        }));
    }

    let out_tracks: Vec<Track> = video_track.iter().cloned().collect();
    let mime = "video/x-nv12";
    let duration_us = player.duration_us().or(info.duration_us);
    let start_us = match opts.seek_us {
        Some(us) => player.seek_us(us)?,
        None => 0,
    };
    let meta = meta_json(
        &info,
        &out_tracks,
        mime,
        duration_us,
        start_us,
        opts.fragment_ms,
        dash_note,
        None,
    );
    let mut meta = meta;
    if let Some(obj) = meta.as_object_mut() {
        obj.insert(
            "backend".into(),
            json!(decoder.as_ref().map(|d| d.backend()).unwrap_or("none")),
        );
        obj.insert("packetHeaderBytes".into(), json!(FRAME_HEADER_BYTES));
        obj.insert("decodeMode".into(), json!("frames"));
        if video_track.is_none() {
            obj.insert("note".into(), json!("这条流只有声音（输入没有视频轨）"));
        }
        match &audio_decoder {
            Some(dec) => {
                let (rate, channels, bits) = dec.stream_info();
                obj.insert(
                    "audio".into(),
                    json!({
                        "codec": "pcm_s16le",
                        "sourceBitsPerSample": bits,
                        "sampleRate": rate,
                        "channels": channels,
                        "kind": KIND_AUDIO,
                    }),
                );
                if video_track.is_some() {
                    obj.insert("videoKind".into(), json!(KIND_VIDEO));
                }
            }
            None => {
                let note = match &audio_note {
                    Some(n) => format!("画面由引擎解码（NV12 帧流）；{n}"),
                    None => "画面由引擎解码（NV12 帧流）；这条流没有音频轨道".to_string(),
                };
                obj.insert("note".into(), json!(note));
            }
        }
    }
    println!("{}", json::progress_line("stream-meta", meta));
    std::io::stdout().flush()?;

    // 采样形态转换要**按编码**走：H.264/H.265 在引擎里是 Annex-B（转长度前缀），
    // AV1 是 OBU 流（原样透传）。按 NAL 去拆 OBU 流会拆不出东西 → 采样变空 →
    // 整条流一帧都不发、进程还退出 0（这条踩过）。
    let video_codec = video_track
        .as_ref()
        .map(|t| t.codec.clone())
        .unwrap_or(me_core::CodecId::Unknown(0));

    let stderr = std::io::stderr();
    let mut frames = 0u64;
    let mut audio_chunks = 0u64;
    // ⚠️ **音频优先**（2026-10-06，修"AV1+AAC 播放时音频跳帧"）：
    //
    // 解码是这条循环里最慢的一环（1080p AV1 软解 ~80ms/帧，比 33ms 的实时
    // 需求慢一倍多）。而采样在文件里是**按时间交织**的：视频采样夹在音频
    // 采样中间。原来"读一个解一个"，于是音频的出厂速度被视频解码**卡在同一个
    // 节拍上** —— 宿主那点音频余量（几十毫秒）一眨眼就耗干，听感上就是
    // 断断续续 / 跳帧。
    //
    // 现在把视频采样先攒进 `pending_video`（只读不解），**音频到了立刻解、立刻发**：
    // 读包本身很便宜，所以音频可以很快跑到"宿主愿意预读的上限"（4 秒），
    // 之后宿主自带背压（`canReadMore`）会让我们自然慢下来。等攒够上限再解最老的
    // 视频帧 —— 视频解码慢只影响**画面**的及时性，不再连累声音。
    //
    // 上限取"够填满宿主 4 秒音频余量 + 一点抖动余量"：150 帧 ≈ 5 秒视频，
    // 24 MB 是硬顶（这条素材 5 秒 ≈ 0.6 MB，离顶很远）。
    //
    // 没有活跃音频时**不进队列**（照旧"读一个解一个"）—— 纯视频流没有要抢路的
    // 东西，白攒只会给慢源（边下边播）添无用停顿。
    const PENDING_VIDEO_MAX_FRAMES: usize = 150;
    const PENDING_VIDEO_MAX_BYTES: usize = 24 << 20;
    let mut pending_video: std::collections::VecDeque<(i64, Vec<u8>)> =
        std::collections::VecDeque::new();
    let mut pending_bytes = 0usize;
    let stdout = std::io::stdout();
    loop {
        let pkt = match player.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            // 数据还没到（增长文件的正常路径）：如实报"缓冲中"再重试
            Err(MediaError::NeedMore { .. }) => {
                report_playhead(&stderr, opts.progress, &player, frames, 0, "buffering");
                std::thread::sleep(std::time::Duration::from_millis(500));
                continue;
            }
            Err(e) => return Err(e),
        };
        // 音频包：解成 S16LE 直接发（与画面同一条流，按容器的顺序交错）
        if let (Some(at), Some(dec)) = (&audio_track, audio_decoder.as_mut()) {
            if pkt.track_id == at.id {
                let pts_us = at.timebase.ticks_to_us(pkt.pts);
                let chunks = match dec.decode(&pkt.data, pts_us) {
                    Ok(v) => v,
                    // 音频解不了**不能带走画面**（AAC 还没接完就是这种情形）：
                    // 停用音频、往 stderr 说清原因，画面继续流。
                    Err(e) => {
                        eprintln!(
                            "{}",
                            crate::json::error_line(&MediaError::Unsupported(format!(
                                "音频解码失败，已停用音频只出画面：{e}"
                            )))
                        );
                        audio_decoder = None;
                        continue;
                    }
                };
                for chunk in chunks {
                    let pcm = to_s16le(&chunk.samples, chunk.bits_per_sample);
                    let mut head = [0u8; FRAME_HEADER_BYTES];
                    head[0..4].copy_from_slice(FRAME_MAGIC);
                    head[4..8].copy_from_slice(&KIND_AUDIO.to_le_bytes());
                    head[8..12].copy_from_slice(&chunk.sample_rate.to_le_bytes());
                    head[12..16].copy_from_slice(&(chunk.channels as u32).to_le_bytes());
                    head[16..24].copy_from_slice(&chunk.pts_us.to_le_bytes());
                    head[24..28].copy_from_slice(&(pcm.len() as u32).to_le_bytes());
                    let mut lock = stdout.lock();
                    if lock.write_all(&head).is_err() || lock.write_all(&pcm).is_err() {
                        return Ok(crate::exit_code::OK);
                    }
                    let _ = lock.flush();
                    audio_chunks += 1;
                }
                continue;
            }
        }
        let Some(vdec) = decoder.as_mut() else {
            continue;
        };
        if pkt.track_id != video_track_id {
            continue;
        }
        let sample = convert::to_decoder_sample(&video_codec, &pkt.data, 4);
        if sample.is_empty() {
            continue;
        }
        let pts_us = video_timebase.ticks_to_us(pkt.pts);
        // 音频还在跟的时候，视频先攒着（见上面"音频优先"的长注释）。
        let audio_waiting = audio_track.is_some() && audio_decoder.is_some();
        if !audio_waiting {
            // 没有音频要抢路：把攒着的先清掉，再照旧"读一个解一个"
            while let Some((pts, s)) = pending_video.pop_front() {
                pending_bytes -= s.len();
                if !decode_and_emit_video(vdec.as_mut(), &s, pts, &mut frames)? {
                    return Ok(crate::exit_code::OK);
                }
                if frames.is_multiple_of(30) {
                    report_playhead(&stderr, opts.progress, &player, frames, audio_chunks, "playing");
                }
            }
        }
        pending_bytes += sample.len();
        pending_video.push_back((pts_us, sample));
        while pending_video.len() > PENDING_VIDEO_MAX_FRAMES
            || pending_bytes > PENDING_VIDEO_MAX_BYTES
        {
            let Some((pts, s)) = pending_video.pop_front() else {
                break;
            };
            pending_bytes -= s.len();
            if !decode_and_emit_video(vdec.as_mut(), &s, pts, &mut frames)? {
                return Ok(crate::exit_code::OK);
            }
            if frames.is_multiple_of(30) {
                report_playhead(&stderr, opts.progress, &player, frames, audio_chunks, "playing");
            }
        }
    }
    // 收尾：先把攒着的视频采样解完，再让解码器冲刷内部排队的帧/段
    if let Some(vdec) = decoder.as_mut() {
        while let Some((pts, s)) = pending_video.pop_front() {
            pending_bytes -= s.len();
            if !decode_and_emit_video(vdec.as_mut(), &s, pts, &mut frames)? {
                return Ok(crate::exit_code::OK);
            }
        }
    }
    for frame in decoder.as_mut().map(|d| d.flush()).transpose()?.unwrap_or_default() {
        let mut head = [0u8; FRAME_HEADER_BYTES];
        head[0..4].copy_from_slice(FRAME_MAGIC);
        head[4..8].copy_from_slice(&KIND_VIDEO.to_le_bytes());
        head[8..12].copy_from_slice(&frame.width.to_le_bytes());
        head[12..16].copy_from_slice(&frame.height.to_le_bytes());
        head[16..24].copy_from_slice(&frame.pts_us.to_le_bytes());
        head[24..28].copy_from_slice(&(frame.nv12.len() as u32).to_le_bytes());
        let mut lock = std::io::stdout().lock();
        if lock.write_all(&head).is_err() || lock.write_all(&frame.nv12).is_err() {
            return Ok(crate::exit_code::OK);
        }
        let _ = lock.flush();
        frames += 1;
    }
    if let Some(dec) = audio_decoder.as_mut() {
        for chunk in dec.flush()? {
            let pcm = to_s16le(&chunk.samples, chunk.bits_per_sample);
            let mut head = [0u8; FRAME_HEADER_BYTES];
            head[0..4].copy_from_slice(FRAME_MAGIC);
            head[4..8].copy_from_slice(&KIND_AUDIO.to_le_bytes());
            head[8..12].copy_from_slice(&chunk.sample_rate.to_le_bytes());
            head[12..16].copy_from_slice(&(chunk.channels as u32).to_le_bytes());
            head[16..24].copy_from_slice(&chunk.pts_us.to_le_bytes());
            head[24..28].copy_from_slice(&(pcm.len() as u32).to_le_bytes());
            let mut lock = std::io::stdout().lock();
            if lock.write_all(&head).is_err() || lock.write_all(&pcm).is_err() {
                return Ok(crate::exit_code::OK);
            }
            let _ = lock.flush();
            audio_chunks += 1;
        }
    }
    report_playhead(&stderr, opts.progress, &player, frames, audio_chunks, "ended");
    Ok(crate::exit_code::OK)
}

/// 解一个视频采样并把解出的帧写进 stdout；`frames` 是累计计数（调用方按它报播放头）。
///
/// 返回 `false` = 对端把管道关了（前端换 seek / 关窗）—— 调用方按**正常收尾**处理
/// （与 `stream` 的 broken pipe 语义一致），不是错误。
fn decode_and_emit_video(
    vdec: &mut dyn me_decode::VideoDecoder,
    sample: &[u8],
    pts_us: i64,
    frames: &mut u64,
) -> Result<bool> {
    let decoded = match vdec.decode(sample, pts_us) {
        Ok(v) => v,
        Err(e) if is_broken_pipe(&e) => return Ok(false),
        Err(e) => return Err(e),
    };
    for frame in decoded {
        let mut head = [0u8; FRAME_HEADER_BYTES];
        head[0..4].copy_from_slice(FRAME_MAGIC);
        head[4..8].copy_from_slice(&KIND_VIDEO.to_le_bytes());
        head[8..12].copy_from_slice(&frame.width.to_le_bytes());
        head[12..16].copy_from_slice(&frame.height.to_le_bytes());
        head[16..24].copy_from_slice(&frame.pts_us.to_le_bytes());
        head[24..28].copy_from_slice(&(frame.nv12.len() as u32).to_le_bytes());
        let mut lock = std::io::stdout().lock();
        if lock.write_all(&head).is_err() || lock.write_all(&frame.nv12).is_err() {
            return Ok(false);
        }
        let _ = lock.flush();
        drop(lock);
        *frames += 1;
    }
    Ok(true)
}

/// 原始位深的交织采样 → S16LE（前端音频输出用）。
///
/// 只做位移、不做抖动/重采样：解码器给多少位深，这里就按位对齐折算 ——
/// 16 位原样、24 位丢最低 8 位、32 位丢最低 16 位、8 位补到高 8 位。
fn to_s16le(samples: &[i32], bits: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        let v: i16 = match bits {
            8 => (*s as i16) << 8,
            16 => *s as i16,
            24 => (*s >> 8) as i16,
            32 => (*s >> 16) as i16,
            _ => *s as i16,
        };
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}
