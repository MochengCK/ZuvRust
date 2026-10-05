//! 子命令实现。
//!
//! 每个子命令都是"参数 → 干活 → 吐 NDJSON 行"的同一套路子。刻意**不做参数解析库**：
//! 引擎只认 `--k=v` 与位置参数两种形态，够用且不会因为某个库的版本变化而改行为。
//! （宿主那边传参也是照这个约定生成的。）

use std::sync::Arc;

use me_core::{MediaError, MediaInfo, ReadAt, Result};

use crate::exit_code;
use crate::json;
use crate::mux;
use crate::progress::Progress;

/// 分发子命令。
pub fn dispatch(args: &[String]) -> Result<u8> {
    let (cmd, rest) = args.split_first().expect("main 已保证非空");
    match cmd.as_str() {
        "version" => cmd_version(rest),
        "probe" => cmd_probe(rest),
        "mux" => cmd_mux(rest),
        "remux" => cmd_remux(rest),
        "download" => cmd_download(rest),
        "play" => cmd_play(rest),
        "stream" => cmd_stream(rest),
        "decode" => cmd_decode(rest),
        other => Err(MediaError::Unsupported(format!("未知子命令 `{other}`"))),
    }
}

/// 错误 → 退出码。
///
/// 判据只有一条：**宿主重试这个错误有没有意义**。
/// `Io` 归到"输入不可用"而不是"引擎内部错误"是刻意的：本地 IO 失败几乎都是
/// 路径不存在 / 权限 / 磁盘满，同一个输入重试永远不会成功 —— 报成 5（内部错误）
/// 会让宿主以为是引擎的 bug，界面上给用户一个完全错误的方向。
/// 真正值得重试的临时失败走 `Protocol`（网络）与 `NeedMore`（数据还没到）。
pub fn exit_code_of(e: &MediaError) -> u8 {
    match e {
        MediaError::Cancelled => exit_code::CANCELLED,
        MediaError::Unsupported(_) => exit_code::USAGE,
        MediaError::Container(_) | MediaError::Bitstream(_) | MediaError::Io(_) => exit_code::BAD_INPUT,
        MediaError::Protocol(_) | MediaError::NeedMore { .. } => exit_code::TRANSIENT,
    }
}

/// 取 `--k=v` 形式的选项值。
fn opt<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    let prefix = format!("--{key}=");
    args.iter().find_map(|a| a.strip_prefix(prefix.as_str()))
}

/// 位置参数（跳过所有 `--` 开头的）。
fn positional(args: &[String]) -> Vec<&str> {
    args.iter().filter(|a| !a.starts_with("--")).map(|s| s.as_str()).collect()
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn cmd_version(args: &[String]) -> Result<u8> {
    let v = serde_json::json!({
        "name": "ZuvRust",
        "version": env!("CARGO_PKG_VERSION"),
        // 这两个列表是宿主"能力探测"的依据：界面据此决定哪些按钮可点。
        // `flac` / `adts` / `mpeg-audio` 是"裸流"（没有容器壳，只有一串首尾相接的音频帧）；
        // `ogg` 是真正的容器（页序列，`.opus` 就是它装 Opus）；
        // `matroska` 是 EBML 家族（`.mkv` / `.webm` / `.mka`）。
        "containers": ["mpeg-ts", "iso-bmff", "matroska", "ogg", "flac", "adts", "mpeg-audio"],
        "protocols": ["http", "hls", "dash"],
        // `concat` = mux 能把同角色的多个输入（分片/分段）按顺序接成一条连续时间轴
        "features": ["probe", "remux", "mux", "concat", "download", "play", "growing-file"],
        // 边下边播的契约：媒体引擎读"下载引擎正在写的文件"，并上报播放头。
        // BT 下载本身由下载引擎（XferRust）负责，不在本引擎里。
        "streaming": {
            "mode": "growing-file",
            "downloadEngine": "external",
            "playheadHint": true
        }
    });
    emit(args, &v);
    Ok(exit_code::OK)
}

fn cmd_probe(args: &[String]) -> Result<u8> {
    let pos = positional(args);
    let Some(input) = pos.first() else {
        return Err(MediaError::Unsupported("probe 需要一个输入（文件路径或 URL）".into()));
    };
    let cancel = me_stream::CancelFlag::install();
    let client = me_stream::default_client()?;
    let opts = me_stream::OpenOptions {
        grow: has_flag(args, "--grow"),
        expected_len: opt(args, "size").and_then(|v| v.parse::<u64>().ok()),
        max_height: opt(args, "max-height").and_then(|v| v.parse::<u32>().ok()),
        max_bandwidth: opt(args, "max-bandwidth").and_then(|v| v.parse::<u64>().ok()),
        cancel,
        ..Default::default()
    };
    let opened = me_stream::open_input(&client, input, &opts)?;

    // 三种打开结果各自的探测方式
    let value = match opened {
        me_stream::OpenedSource::Single(src) => {
            let info = me_container::probe(src, input)?;
            info_json(&info)
        }
        me_stream::OpenedSource::Hls(hls) => {
            let info = me_container::probe(hls.source.clone(), input)?;
            let mut v = info_json(&info);
            // HLS 的时长以清单为准（比从第一条分片里扫出来的准）
            if let Some(d) = hls.duration_us {
                v["durationUs"] = serde_json::json!(d);
                v["durationSecs"] = serde_json::json!(d as f64 / 1_000_000.0);
            }
            v["hls"] = serde_json::json!({
                "segments": hls.segment_count,
                "live": hls.is_live,
                "bandwidth": hls.variant_bandwidth,
                "resolution": hls.variant_resolution.map(|(w, h)| format!("{w}x{h}")),
            });
            v
        }
        me_stream::OpenedSource::DashTracks(tracks) => {
            // DASH 是"按轨开"：每条轨单独探测（视频与音频本来就是两条流）
            let mut list = Vec::new();
            for t in &tracks {
                let info = me_container::probe(t.source.clone(), input)?;
                let mut v = info_json(&info);
                v["dash"] = serde_json::json!({
                    "contentType": t.content_type,
                    "representationId": t.representation_id,
                    "codec": t.codec,
                    "mime": t.mime,
                    "segments": t.segment_count,
                });
                list.push(v);
            }
            serde_json::json!({
                "container": "dash",
                "source": input,
                "tracks": list,
            })
        }
    };
    emit(args, &value);
    Ok(exit_code::OK)
}

fn info_json(info: &MediaInfo) -> serde_json::Value {
    let tracks: Vec<serde_json::Value> = info
        .tracks
        .iter()
        .map(|t| {
            serde_json::json!({
                "id": t.id,
                // kind 用整数：这是 C API / CLI 的对外契约（见 MediaKind::as_i32）
                "kind": t.kind.as_i32(),
                "kindName": kind_name(t.kind),
                "codec": t.codec.name(),
                "timescale": t.timebase.ticks_per_second,
                "width": t.width,
                "height": t.height,
                "sampleRate": t.sample_rate,
                "channels": t.channels,
                "frameRate": t.frame_rate.map(|(n, d)| format!("{n}/{d}")),
                "language": t.language,
                "extradataBytes": t.extradata.len(),
            })
        })
        .collect();
    serde_json::json!({
        "container": info.container,
        "durationUs": info.duration_us,
        "durationSecs": info.duration_us.map(|u| u as f64 / 1_000_000.0),
        "bitrate": info.bitrate,
        "source": info.source,
        "tracks": tracks,
    })
}

fn kind_name(k: me_core::MediaKind) -> &'static str {
    match k {
        me_core::MediaKind::Video => "video",
        me_core::MediaKind::Audio => "audio",
        me_core::MediaKind::Subtitle => "subtitle",
        me_core::MediaKind::Other => "other",
    }
}

/// `mux <输出> <输入>...`：把多条流合并成一个文件。
///
/// 输入个数**不设上限也不假设含义**：谁是画面、谁是声音由引擎探测，同角色的
/// 多个输入按传入顺序接起来（见 [`crate::mux`]）。所以一个输入也合法（重封装）。
fn cmd_mux(args: &[String]) -> Result<u8> {
    let pos = positional(args);
    if pos.len() < 2 {
        return Err(MediaError::Unsupported(
            "mux 需要一个输出与至少一个输入：zuvrust mux <输出> <输入>...".into(),
        ));
    }
    let output = pos[0];
    let inputs = &pos[1..];
    mux_files(args, output, inputs)
}

/// `remux <输出> <输入>`：换容器重新封装（不重编码）。
fn cmd_remux(args: &[String]) -> Result<u8> {
    let pos = positional(args);
    if pos.len() < 2 {
        return Err(MediaError::Unsupported(
            "remux 需要一个输出与一个输入：zuvrust remux <输出> <输入>".into(),
        ));
    }
    mux_files(args, pos[0], &pos[1..2])
}

fn mux_files(args: &[String], output: &str, inputs: &[&str]) -> Result<u8> {
    let cancel = me_stream::CancelFlag::install();
    let client = me_stream::default_client()?;
    let opts = me_stream::OpenOptions { cancel: cancel.clone(), ..Default::default() };

    // 先把每个输入打开并探测：谁是画面、谁是声音由**引擎**判断，不靠调用方传
    let mut opened: Vec<(String, Arc<dyn ReadAt>, MediaInfo)> = Vec::new();
    for input in inputs {
        let src: Arc<dyn ReadAt> = match me_stream::open_input(&client, input, &opts)? {
            me_stream::OpenedSource::Single(s) => s,
            me_stream::OpenedSource::Hls(h) => h.source,
            me_stream::OpenedSource::DashTracks(t) => {
                // DASH 的每条轨是独立字节流：多给几条输入就等于"把轨拆开再合并"
                if t.len() == 1 {
                    t.into_iter().next().map(|x| x.source).expect("刚判过")
                } else {
                    return Err(MediaError::Unsupported(format!(
                        "DASH 输入（{input}）里有 {} 条轨，请分别下载视频轨与音频轨后再 mux",
                        t.len()
                    )));
                }
            }
        };
        let info = me_container::probe(src.clone(), input)?;
        opened.push((input.to_string(), src, info));
    }

    let refs: Vec<(&str, Arc<dyn ReadAt>, MediaInfo)> = opened
        .iter()
        .map(|(n, s, i)| (n.as_str(), s.clone(), i.clone()))
        .collect();
    let outcome = mux::mux_to_file(output, &refs, has_flag(args, "--progress"), &cancel)?;
    emit(args, &mux::outcome_json(&outcome));
    Ok(exit_code::OK)
}

/// `download <地址> [--out=路径]`：把 HTTP / HLS / DASH 拉到本地。
///
/// **磁力会被明确拒绝**：BT 下载由下载引擎（XferRust）负责，媒体引擎只播放。
fn cmd_download(args: &[String]) -> Result<u8> {
    let pos = positional(args);
    let Some(input) = pos.first() else {
        return Err(MediaError::Unsupported("download 需要一个地址（http/https 或清单）".into()));
    };
    let cancel = me_stream::CancelFlag::install();
    let client = me_stream::default_client()?;
    let opts = me_stream::OpenOptions {
        cancel: cancel.clone(),
        max_height: opt(args, "max-height").and_then(|v| v.parse::<u32>().ok()),
        max_bandwidth: opt(args, "max-bandwidth").and_then(|v| v.parse::<u64>().ok()),
        ..Default::default()
    };
    let out_path = opt(args, "out").map(|s| s.to_string()).unwrap_or_else(|| default_out_name(input));
    let progress = has_flag(args, "--progress");

    let opened = me_stream::open_input(&client, input, &opts)?;
    let mut reporter = Progress::new(progress, "download");
    let (bytes, tracks) = match opened {
        me_stream::OpenedSource::Single(src) => {
            let n = copy_to_file(&src, &out_path, &mut reporter, None, &cancel)?;
            (n, 1usize)
        }
        me_stream::OpenedSource::Hls(hls) => {
            let n = copy_to_file(&hls.source, &out_path, &mut reporter, hls.duration_us, &cancel)?;
            (n, 1usize)
        }
        me_stream::OpenedSource::DashTracks(list) => {
            // DASH 是"按轨开"：一条轨一个文件（视频/音频分开下，再由 mux 合并）
            let mut total = 0u64;
            let mut count = 0usize;
            for t in &list {
                let path = with_suffix(&out_path, &t.content_type);
                total += copy_to_file(&t.source, &path, &mut reporter, t.duration_us, &cancel)?;
                count += 1;
            }
            (total, count)
        }
    };
    reporter.finish(&serde_json::json!({ "percent": 100.0, "outputBytes": bytes }));
    emit(
        args,
        &serde_json::json!({
            "output": out_path,
            "bytes": bytes,
            "tracks": tracks,
            "source": input,
        }),
    );
    Ok(exit_code::OK)
}

/// `play <输入> [--seek=秒]`：把采样流喂给宿主（边下边播时同时上报播放头）。
fn cmd_play(args: &[String]) -> Result<u8> {
    let pos = positional(args);
    let Some(input) = pos.first() else {
        return Err(MediaError::Unsupported("play 需要一个输入（文件路径或 URL）".into()));
    };
    let cancel = me_stream::CancelFlag::install();
    let client = me_stream::default_client()?;
    let opts = me_stream::OpenOptions {
        // 默认按"可能还在下载"的方式读本地文件：边下边播是常态，
        // 文件已下完时这个源照样能读完（多几次 stat 而已）
        grow: !has_flag(args, "--no-grow"),
        expected_len: opt(args, "size").and_then(|v| v.parse::<u64>().ok()),
        cancel: cancel.clone(),
        ..Default::default()
    };
    let opened = me_stream::open_input(&client, input, &opts)?;
    let (src, name) = match opened {
        me_stream::OpenedSource::Single(s) => (s, input.to_string()),
        me_stream::OpenedSource::Hls(h) => (h.source, input.to_string()),
        me_stream::OpenedSource::DashTracks(list) => {
            let Some(first) = list.into_iter().next() else {
                return Err(MediaError::Protocol("DASH 里没有可播的轨道".into()));
            };
            (first.source, input.to_string())
        }
    };

    let mut player = me_stream::Player::open(src, &name, me_stream::PlayerConfig::default())?;
    let info = player.info();
    emit(args, &info_json(&info));

    if let Some(secs) = opt(args, "seek").and_then(|v| v.parse::<f64>().ok()) {
        let landing = player.seek_us((secs * 1_000_000.0) as i64)?;
        emit(args, &serde_json::json!({ "seekedToUs": landing }));
    }

    let mut reporter = Progress::new(true, "playhead");
    let duration = player.duration_us();
    let mut count = 0u64;
    while let Some(p) = player.next_packet()? {
        cancel.check()?;
        count += 1;
        // 每吐一条采样就上报一次播放头：宿主把它转达给下载引擎
        // （"读到哪里就优先下哪里"）—— 这是两个引擎之间唯一的协作接口
        if reporter.due() {
            let hint = player.playhead_hint();
            let stats = player.stats();
            reporter.report(&serde_json::json!({
                "percent": duration
                    .filter(|d| *d > 0)
                    .map(|d| ((stats.playhead_us as f64 / d as f64) * 100.0).clamp(0.0, 100.0))
                    .unwrap_or(0.0),
                "packets": count,
                "playheadUs": stats.playhead_us,
                "sourceOffset": hint.offset_bytes,
                "bufferedPackets": stats.buffered_packets,
                "durationUs": duration,
                "trackId": p.track_id,
                "keyframe": p.keyframe,
            }));
        }
    }
    reporter.finish(&serde_json::json!({
        "percent": 100.0,
        "packets": count,
        "playheadUs": player.stats().playhead_us,
        "sourceOffset": player.playhead_hint().offset_bytes,
        "durationUs": duration,
    }));
    Ok(exit_code::OK)
}

/// `stream <输入>`：**流式播放** —— stdout 吐"一行元信息 + 分片 fMP4 字节"，
/// stderr 吐播放头（NDJSON）。宿主播放器把字节喂给 MSE，只负责解码。
fn cmd_decode(args: &[String]) -> Result<u8> {
    let pos = positional(args);
    let Some(input) = pos.first() else {
        return Err(MediaError::Unsupported(
            "decode 需要一个输入（文件路径或 URL）".into(),
        ));
    };
    let cancel = me_stream::CancelFlag::install();
    let opts = crate::decode::DecodeOptions {
        audio: has_flag(args, "--audio"),
        out_dir: opt(args, "out").map(|s| s.to_string()),
        max_frames: opt(args, "count").and_then(|v| v.parse::<u64>().ok()),
        seek_us: opt(args, "seek")
            .and_then(|v| v.parse::<f64>().ok())
            .map(|s| (s * 1_000_000.0) as i64),
        open: me_stream::OpenOptions {
            grow: !has_flag(args, "--no-grow"),
            cancel,
            ..Default::default()
        },
    };
    crate::decode::decode_to_frames(input, opts)
}

fn cmd_stream(args: &[String]) -> Result<u8> {
    let pos = positional(args);
    let Some(input) = pos.first() else {
        return Err(MediaError::Unsupported(
            "stream 需要一个输入（文件路径或 URL）".into(),
        ));
    };
    let cancel = me_stream::CancelFlag::install();
    let mut opts = crate::stream::StreamOptions {
        // `--video=frames`：不出 fMP4，出解码后的 NV12 帧流（前端用帧渲染器上屏）
        video_frames: opt(args, "video").map(|v| v == "frames").unwrap_or(false),
        // `--audio=pcm`：把音频也解成 PCM 一起发（引擎解码，前端放）
        audio_pcm: opt(args, "audio").map(|v| v == "pcm").unwrap_or(false),
        seek_us: opt(args, "seek")
            .and_then(|v| v.parse::<f64>().ok())
            .map(|s| (s * 1_000_000.0) as i64),
        fragment_ms: opt(args, "fragment-ms")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(500)
            .clamp(20, 10_000),
        // 播放头默认开：宿主靠它做"优先下载播放位置附近"
        progress: !has_flag(args, "--no-progress"),
        open: me_stream::OpenOptions {
            // 与 `play` 一致：本地文件默认按"可能还在下载"的方式读
            grow: !has_flag(args, "--no-grow"),
            expected_len: opt(args, "size").and_then(|v| v.parse::<u64>().ok()),
            max_height: opt(args, "max-height").and_then(|v| v.parse::<u32>().ok()),
            max_bandwidth: opt(args, "max-bandwidth").and_then(|v| v.parse::<u64>().ok()),
            cancel,
            ..Default::default()
        },
    };
    // `--avail=<文件>`：宿主给的「哪些字节已经下到了」（见 me_stream::avail）。
    // 给了它，空洞就是「等」而不是「损坏」—— BT 边下边播必用。
    if let Some(path) = opt(args, "avail") {
        opts.open.grow_opts.availability = Some(std::path::PathBuf::from(path));
    }
    // `--wait-timeout=<毫秒>`：等下载引擎补数据的单次上限（默认 30s）。
    // 宿主在"网络已经断了"时需要更快放弃（好提示用户重试），CI 与测试也需要它
    // 把一次"必然等满"的行为压到秒级 —— 没有它，任何覆盖 avail 等待路径的测试
    // 都得等满 30 秒。
    if let Some(ms) = opt(args, "wait-timeout").and_then(|v| v.parse::<u64>().ok()) {
        opts.open.grow_opts.wait_timeout = std::time::Duration::from_millis(ms);
    }
    crate::stream::stream_to_stdout(input, opts)
}

/// 把一个字节源整段写进文件（带进度）。
fn copy_to_file(
    src: &Arc<dyn ReadAt>,
    path: &str,
    reporter: &mut Progress,
    duration_us: Option<i64>,
    cancel: &me_stream::CancelFlag,
) -> Result<u64> {
    use std::io::Write;
    let total = src.len();
    let mut file = std::fs::File::create(path)?;
    let mut buf = vec![0u8; 256 * 1024];
    let mut off = 0u64;
    loop {
        cancel.check()?;
        let n = src.read_at(off, &mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        off += n as u64;
        let percent = match total {
            Some(t) if t > 0 => (off as f64 / t as f64) * 100.0,
            _ => 0.0,
        };
        reporter.maybe_report(&serde_json::json!({
            "percent": percent.clamp(0.0, 100.0),
            "outputBytes": off,
            "totalBytes": total,
            "durationUs": duration_us,
        }));
    }
    file.flush()?;
    Ok(off)
}

/// 默认输出文件名：从地址里取最后一段（去掉查询串），没有就用 `download.bin`。
fn default_out_name(input: &str) -> String {
    let no_query = input.split(['?', '#']).next().unwrap_or(input);
    let last = no_query.rsplit('/').next().unwrap_or("");
    let name = last.trim();
    if name.is_empty() || name.contains(':') {
        "download.bin".to_string()
    } else {
        name.to_string()
    }
}

/// 给输出名加后缀（DASH 分轨下载用）：`a.mp4` + `video` → `a.video.mp4`。
fn with_suffix(path: &str, suffix: &str) -> String {
    match path.rsplit_once('.') {
        Some((stem, ext)) => format!("{stem}.{suffix}.{ext}"),
        None => format!("{path}.{suffix}"),
    }
}

/// `--json` 决定输出形态。
///
/// 不加 `--json` 时给**人**看：多行缩进 JSON（调试时能直接读）。
/// 加了就压成一行（宿主按行解析）。两种形态都由引擎决定，
/// 宿主不需要自己格式化 —— 那是重复实现。
fn emit(args: &[String], v: &serde_json::Value) {
    if args.iter().any(|a| a == "--json") {
        println!("{}", json::ok_line(v));
    } else {
        println!("{}", serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn version_reports_capabilities() {
        let code = dispatch(&s(&["version", "--json"])).unwrap();
        assert_eq!(code, exit_code::OK);
    }

    #[test]
    fn unknown_subcommand_is_a_usage_error() {
        let e = dispatch(&s(&["frobnicate"])).unwrap_err();
        assert_eq!(exit_code_of(&e), exit_code::USAGE);
    }

    #[test]
    fn subcommands_validate_their_arguments() {
        for cmd in ["mux", "remux", "download", "play", "probe"] {
            let e = dispatch(&s(&[cmd])).unwrap_err();
            assert!(matches!(e, MediaError::Unsupported(_)), "{cmd} 缺参数时该报用法错误");
            assert_eq!(exit_code_of(&e), exit_code::USAGE, "{cmd}");
        }
        // mux 至少要有一个输入（只给输出不够）
        let e = dispatch(&s(&["mux", "out.mp4"])).unwrap_err();
        assert!(matches!(e, MediaError::Unsupported(_)));
        // 一个输入是合法的（重封装/单文件换容器），不该在参数校验就拦下
        let e = dispatch(&s(&["mux", "out.mp4", "/definitely/not/here.ts"])).unwrap_err();
        assert!(!matches!(e, MediaError::Unsupported(_)), "一个输入不该被判成用法错误：{e}");
    }

    #[test]
    fn probe_without_input_is_an_error() {
        let e = dispatch(&s(&["probe"])).unwrap_err();
        assert_eq!(exit_code_of(&e), exit_code::USAGE);
    }

    #[test]
    fn magnet_is_rejected_with_a_pointer_to_the_download_engine() {
        let e = dispatch(&s(&["probe", "magnet:?xt=urn:btih:abc", "--json"])).unwrap_err();
        assert!(matches!(e, MediaError::Unsupported(_)));
        assert!(e.to_string().contains("下载引擎"), "{e}");
    }

    #[test]
    fn option_and_positional_extraction() {
        let a = s(&["probe", "/tmp/a.ts", "--json", "--log=debug"]);
        assert_eq!(positional(&a), vec!["probe", "/tmp/a.ts"]);
        assert_eq!(opt(&a, "log"), Some("debug"));
        assert_eq!(opt(&a, "missing"), None);
        assert!(has_flag(&a, "--json"));
        assert!(!has_flag(&a, "--progress"));
    }

    #[test]
    fn exit_codes_distinguish_retryable_from_fatal() {
        // 宿主靠这个决定"要不要重试"，分类错了会让它无限重试一个坏文件
        assert_eq!(exit_code_of(&MediaError::container("bad")), exit_code::BAD_INPUT);
        assert_eq!(exit_code_of(&MediaError::protocol("timeout")), exit_code::TRANSIENT);
        assert_eq!(exit_code_of(&MediaError::Cancelled), exit_code::CANCELLED);
        // 本地 IO 失败（路径不存在/权限）与容器坏掉同类：重试无用
        assert_eq!(exit_code_of(&MediaError::Io("no such file".into())), exit_code::BAD_INPUT);
        assert!(!json::is_retryable(&MediaError::Io("no such file".into())));
        // "数据还没到"是可以重试的（BT 边下边播的常态）
        assert_eq!(exit_code_of(&MediaError::NeedMore { need: 10, have: 0 }), exit_code::TRANSIENT);
    }

    #[test]
    fn default_out_name_strips_query_and_handles_weird_inputs() {
        assert_eq!(default_out_name("https://a.com/v/x.mp4?token=1"), "x.mp4");
        assert_eq!(default_out_name("https://a.com/"), "download.bin");
        assert_eq!(default_out_name("magnet:?xt=urn:btih:x"), "download.bin");
    }

    #[test]
    fn with_suffix_inserts_before_the_extension() {
        assert_eq!(with_suffix("a.mp4", "video"), "a.video.mp4");
        assert_eq!(with_suffix("a", "audio"), "a.audio");
    }
}
