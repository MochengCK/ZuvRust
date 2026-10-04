//! CLI 层的契约测试：**真的启动一次 `zuvrust` 进程**，走完整的
//! 「命令行参数 → 干活 → NDJSON 输出」。
//!
//! 为什么单测不够：单测直接调函数，跳过了"宿主怎么给参数、引擎怎么把进度吐出来"
//! 这一段 —— 而桌面端恰恰**只依赖命令行与输出格式**（它不链接引擎的任何代码）。
//! 所以"合并时会实时上报进度"这句话，只有在这里才能被真正验证。

use std::path::Path;
use std::process::Command;

use me_core::{CodecId, MediaKind, Packet, Timebase, Track};
use me_mux::Mp4Muxer;

/// 引擎二进制（Cargo 为 bin target 注入的路径）。
const BIN: &str = env!("CARGO_BIN_EXE_zuvrust");

fn video_track() -> Track {
    let mut t = Track::new(1, MediaKind::Video, CodecId::H264, Timebase::new(30_000));
    t.width = 640;
    t.height = 360;
    let mut ed = vec![0u8, 0, 0, 1];
    ed.extend_from_slice(&[
        0x67, 0x42, 0x00, 0x1E, 0x95, 0xA8, 0x28, 0x0F, 0x00, 0x44, 0xFC, 0xB8, 0x08, 0x80, 0x00,
        0x03, 0x00, 0x08, 0x00, 0x00, 0x03, 0x01, 0xE0, 0x78, 0xB1, 0x6C, 0x90,
    ]);
    ed.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE, 0x3C, 0x80]);
    t.extradata = ed;
    t
}

fn audio_track() -> Track {
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

fn write_video_only(path: &Path, seconds: usize) {
    let mut m = Mp4Muxer::new();
    m.set_fragment_duration_us(1_000_000);
    m.set_tracks(&[video_track()]).unwrap();
    for i in 0..seconds * 30 {
        m.write_packet(&video_packet(i as i64 * 1000, i % 30 == 0)).unwrap();
    }
    std::fs::write(path, m.finish().unwrap()).unwrap();
}

fn write_audio_only(path: &Path, seconds: usize) {
    let mut m = Mp4Muxer::new();
    m.set_fragment_duration_us(1_000_000);
    m.set_tracks(&[audio_track()]).unwrap();
    for i in 0..seconds * 43 {
        m.write_packet(&audio_packet(i as i64 * 1024)).unwrap();
    }
    std::fs::write(path, m.finish().unwrap()).unwrap();
}

/// AV1 的画面（B 站等站点在用的编码）：采样是 OBU 流，配置是 av1C。
fn av1_track() -> Track {
    let mut t = Track::new(1, MediaKind::Video, CodecId::Av1, Timebase::new(30_000));
    t.width = 1920;
    t.height = 1080;
    t.extradata = vec![0x81, 0x00, 0x0C, 0x00, 0x0A, 0x0B, 0x0C, 0x0D];
    t
}

fn av1_packet(dts: i64, keyframe: bool) -> Packet {
    let mut v = vec![0x12, 0x00, 0x32, 0x05];
    v.extend_from_slice(if keyframe { &[0x11, 0x22, 0x33, 0x44, 0x55] } else { &[0xAA; 5] });
    let mut p = Packet::new(1, dts, dts, 1000, v);
    p.keyframe = keyframe;
    p
}

fn write_av1_only(path: &Path, seconds: usize) {
    let mut m = Mp4Muxer::new();
    m.set_fragment_duration_us(1_000_000);
    m.set_tracks(&[av1_track()]).unwrap();
    for i in 0..seconds * 30 {
        m.write_packet(&av1_packet(i as i64 * 1000, i % 30 == 0)).unwrap();
    }
    std::fs::write(path, m.finish().unwrap()).unwrap();
}

/// FLAC 的声音（B 站"Hi-Res 无损音质"）。
fn flac_track(sample_rate: u32) -> Track {
    let mut t = Track::new(2, MediaKind::Audio, CodecId::Flac, Timebase::new(sample_rate));
    t.sample_rate = sample_rate;
    t.channels = 2;
    // dfLa 载荷：version/flags + STREAMINFO（采样率是其中的 20 位字段）
    let mut ed = vec![0, 0, 0, 0];
    ed.extend_from_slice(&[0x80, 0, 0, 34]);
    let mut si = vec![0u8; 34];
    si[10] = ((sample_rate >> 12) & 0xFF) as u8;
    si[11] = ((sample_rate >> 4) & 0xFF) as u8;
    si[12] = (((sample_rate & 0x0F) << 4) as u8) | (1 << 1);
    ed.extend_from_slice(&si);
    t.extradata = ed;
    t
}

fn write_flac_only(path: &Path, seconds: usize, sample_rate: u32) {
    let mut m = Mp4Muxer::new();
    m.set_fragment_duration_us(1_000_000);
    m.set_tracks(&[flac_track(sample_rate)]).unwrap();
    let frames = (seconds as u64 * sample_rate as u64 / 4096) as usize;
    for i in 0..frames {
        m.write_packet(&Packet::new(
            2,
            i as i64 * 4096,
            i as i64 * 4096,
            4096,
            vec![0xFF, 0xF8, 0x69, 0x08, 0x00, 0x11, 0x22],
        ))
        .unwrap();
    }
    std::fs::write(path, m.finish().unwrap()).unwrap();
}

fn tmp_dir(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir()
        .join(format!("zuvrust-cli-{tag}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("建临时目录");
    dir
}

fn run(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(BIN).args(args).output().expect("启动引擎");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

#[test]
fn mux_reports_progress_lines_then_a_result() {
    let dir = tmp_dir("progress");
    let v = dir.join("v.m4s");
    let a = dir.join("a.m4s");
    write_video_only(&v, 3);
    write_audio_only(&a, 3);
    let out = dir.join("out.mp4");

    let (ok, stdout, stderr) = run(&[
        "mux",
        out.to_str().unwrap(),
        v.to_str().unwrap(),
        a.to_str().unwrap(),
        "--json",
        "--progress",
    ]);
    assert!(ok, "合并应当成功，stderr={stderr}");

    let lines: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();
    // 宿主画进度条靠的就是这种行（`type` 是任务名，数据在 `data` 里）
    let progress: Vec<&&str> = lines.iter().filter(|l| l.contains("\"type\":\"mux\"")).collect();
    assert!(!progress.is_empty(), "合并过程中必须上报进度行，实际输出：{stdout}");
    assert!(
        progress.iter().any(|l| l.contains("\"percent\"")),
        "进度行必须带 percent（没有它宿主画不出进度条）：{stdout}"
    );
    assert!(
        progress.iter().any(|l| l.contains("\"inputBytes\"")),
        "进度行必须带 inputBytes（宿主显示「已写 / 总量」要用）：{stdout}"
    );

    // 最后一行永远是结果行（宿主靠它拿产出信息）
    let last = lines.last().expect("不能没有任何输出");
    assert!(last.contains("\"ok\":true"), "最后一行必须是结果行：{last}");
    assert!(last.contains("\"result\""), "结果行要带 result：{last}");
    assert!(last.contains("\"durationUs\""), "结果行要带时长：{last}");
    assert!(out.exists(), "产物必须真的落在磁盘上");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn concat_through_cli_keeps_the_whole_timeline() {
    let dir = tmp_dir("concat");
    let (v1, v2) = (dir.join("v1.mp4"), dir.join("v2.mp4"));
    let (a1, a2) = (dir.join("a1.mp4"), dir.join("a2.mp4"));
    write_video_only(&v1, 1);
    write_video_only(&v2, 1);
    write_audio_only(&a1, 1);
    write_audio_only(&a2, 1);
    let out = dir.join("joined.mp4");

    let (ok, stdout, stderr) = run(&[
        "mux",
        out.to_str().unwrap(),
        v1.to_str().unwrap(),
        v2.to_str().unwrap(),
        a1.to_str().unwrap(),
        a2.to_str().unwrap(),
        "--json",
        "--progress",
    ]);
    assert!(ok, "多输入合并应当成功，stderr={stderr}");
    assert!(stdout.contains("\"result\""), "要有结果行：{stdout}");

    // 用 probe 独立校验产物：两条轨都在，时长约为两段之和
    let (ok2, probe, stderr2) = run(&["probe", out.to_str().unwrap(), "--json"]);
    assert!(ok2, "probe 产物应当成功，stderr={stderr2}");
    assert!(probe.contains("\"kindName\":\"video\""), "产物缺画面轨：{probe}");
    assert!(probe.contains("\"kindName\":\"audio\""), "产物缺声音轨：{probe}");
    // 两段各 1 秒 → 总时长应当在 2 秒附近（不是 1 秒：那说明第二段没接上）
    let dur = probe
        .split("\"durationUs\":")
        .nth(1)
        .and_then(|s| s.split([',', '}']).next())
        .and_then(|s| s.trim().parse::<i64>().ok())
        .unwrap_or(0);
    assert!(
        (dur - 2_000_000).abs() < 300_000,
        "两段 1 秒的分片接起来总时长应当是 2 秒附近，实际 {dur}（第二段没接上？）"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn av1_with_flac_hires_merges_through_the_cli() {
    // 用户实际撞上的那条：B 站 AV1 画面 + "Hi-Res 无损音质"（FLAC 192k）。
    // 以前 CLI 会以 "fmp4 装不下这条轨道：av1 / Video" 失败。
    let dir = tmp_dir("av1-flac");
    let v = dir.join("v.mp4");
    let a = dir.join("a.m4a");
    write_av1_only(&v, 2);
    write_flac_only(&a, 2, 192_000);
    let out = dir.join("out.mp4");

    let (ok, stdout, stderr) = run(&[
        "mux",
        out.to_str().unwrap(),
        v.to_str().unwrap(),
        a.to_str().unwrap(),
        "--json",
        "--progress",
    ]);
    assert!(ok, "AV1 + FLAC 合并应当成功，stderr={stderr}");
    assert!(stdout.contains("\"type\":\"mux\""), "要有进度行：{stdout}");

    let (ok2, probe, stderr2) = run(&["probe", out.to_str().unwrap(), "--json"]);
    assert!(ok2, "probe 产物应当成功，stderr={stderr2}");
    assert!(probe.contains("\"codec\":\"av1\""), "画面应当是 AV1：{probe}");
    assert!(probe.contains("\"codec\":\"flac\""), "声音应当是 FLAC：{probe}");
    assert!(probe.contains("\"sampleRate\":192000"), "Hi-Res 采样率要保住：{probe}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_single_input_is_a_valid_remux() {
    let dir = tmp_dir("remux");
    let v = dir.join("v.mp4");
    write_video_only(&v, 1);
    let out = dir.join("out.ts");

    // 一个输入 + 换容器（.ts）也是合法用法：参数校验不该把它拦下
    let (ok, stdout, stderr) = run(&["remux", out.to_str().unwrap(), v.to_str().unwrap(), "--json"]);
    assert!(ok, "单输入换容器应当成功，stderr={stderr}");
    assert!(stdout.contains("\"container\":\"mpeg-ts\""), "目标容器应当是 mpeg-ts：{stdout}");

    let _ = std::fs::remove_dir_all(&dir);
}
