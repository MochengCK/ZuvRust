//! 合并能力的**多场景 / 复杂场景**回归。
//!
//! 已有的两个测试文件各管一段：`mux.rs` 的内嵌单测管"分段与时间轴平移"的纯逻辑，
//! `mux_cli.rs` 管"CLI 出口的进度行与结果行"。**这一份补的是场景覆盖**：合并会
//! 被真实下载里各种奇怪的输入组合撞上，而每一种的表现都不一样。
//!
//! 覆盖矩阵（`✓` = 本文件钉住，`—` = 已有文件覆盖）：
//!
//! | 场景 | 谁是画面/声音 | 段数 | 容器 | 期望 |
//! |---|---|---|---|---|
//! | 只有画面（用户只下了视频流） | 探测 | 1 | fMP4→fMP4 | 成功，产物只有一条轨 |
//! | 只有声音 | 探测 | 1 | fMP4→fMP4 | 成功，产物只有一条轨 |
//! | 画面 3 段 + 声音 2 段 | 探测 | 不等 | fMP4→fMP4 | 成功，画面时长按自己三段算 |
//! | 画面 1 段 + 声音 3 段 | 探测 | 不等 | fMP4→fMP4 | 成功，**画面不被声音拖长** |
//! | TS 分片（老站点 / HLS）| 探测 | 1 | **TS→fMP4** | 成功，两条轨都在 |
//! | fMP4 分片 → TS 产物 | 探测 | 1 | **fMP4→TS** | 成功，AAC 补回 ADTS |
//! | AV1 画面 + AAC 声音（B 站最常见） | 探测 | 1 | fMP4→fMP4 | 成功 |
//! | MP3 音频 + MP4 画面 | 探测 | 1 | mp3→fMP4 | 成功（`mp4a` + esds OTI 0x6B） |
//! | AC-3（来自 TS，没有配置盒） | 探测 | 1 | **TS→fMP4** | 成功，`dac3` 由首帧补出 |
//! | AV1 → TS | 探测 | 1 | fMP4→**TS** | **明确拒绝**，不留半截产物 |
//! | 不认识的输出扩展名 | 探测 | 1 | → `.avi` | **明确拒绝** |
//! | 输入损坏 / 不是媒体 | — | 1 | → fMP4 | 报错，不留半截产物 |
//! | 输出目录不存在 | 探测 | 1 | → fMP4 | 报错 |
//! | 24 段串接 | 探测 | 24 | fMP4→fMP4 | 时间轴严格递增、总时长正确 |
//! | 输入 DTS 不从 0 开始 | 探测 | 1 | fMP4→fMP4 | 起点归零 |
//! | 整片（音视频都有）+ 额外一条声音 | 探测 | 2 | fMP4→fMP4 | 成功，画面不被拖长 |
//!
//! ⚠️ **素材是引擎自己造的**（`Mp4Muxer` / `TsMuxer` 写出来再喂回去）。这能覆盖
//! "组合与边界"，但**证明不了"能读别人的文件"** —— 那一条靠 `lossless.rs` 的
//! 真实素材用例（`LERXU_VERIFY_*`，默认 ignore）与宿主现场。

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use me_container::demux;
use me_core::{CodecId, FileSource, MediaKind, Packet, ReadAt, Timebase, Track};
use me_mux::{Mp4Muxer, TsMuxer};

const BIN: &str = env!("CARGO_BIN_EXE_zuvrust");

// ───────────────────────── 素材 ─────────────────────────

fn h264_track(id: u32) -> Track {
    let mut t = Track::new(id, MediaKind::Video, CodecId::H264, Timebase::new(30_000));
    t.width = 640;
    t.height = 360;
    // Annex-B 形态的 extradata（SPS + PPS）：封装器自己会转成 avcC
    let mut ed = vec![0u8, 0, 0, 1];
    ed.extend_from_slice(&[
        0x67, 0x42, 0x00, 0x1E, 0x95, 0xA8, 0x28, 0x0F, 0x00, 0x44, 0xFC, 0xB8, 0x08, 0x80, 0x00,
        0x03, 0x00, 0x08, 0x00, 0x00, 0x03, 0x01, 0xE0, 0x78, 0xB1, 0x6C, 0x90,
    ]);
    ed.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE, 0x3C, 0x80]);
    t.extradata = ed;
    t
}

fn aac_track(id: u32) -> Track {
    let mut t = Track::new(id, MediaKind::Audio, CodecId::Aac, Timebase::new(44_100));
    t.sample_rate = 44_100;
    t.channels = 2;
    t.extradata = vec![0x12, 0x10];
    t
}

fn av1_track(id: u32) -> Track {
    let mut t = Track::new(id, MediaKind::Video, CodecId::Av1, Timebase::new(30_000));
    t.width = 1920;
    t.height = 1080;
    t.extradata = vec![0x81, 0x00, 0x0C, 0x00, 0x0A, 0x0B, 0x0C, 0x0D];
    t
}

fn video_packet(id: u32, dts: i64, keyframe: bool) -> Packet {
    let mut v = vec![0u8, 0, 0, 1];
    v.push(if keyframe { 0x65 } else { 0x41 });
    v.extend_from_slice(&[0x11, 0x22, 0x33, 0x44, 0x55]);
    let mut p = Packet::new(id, dts, dts, 1000, v);
    p.keyframe = keyframe;
    p
}

fn av1_packet(id: u32, dts: i64, keyframe: bool) -> Packet {
    // AV1 是 OBU 流（**没有起始码**），与 H.264/H.265 的关键区别
    let mut v = vec![0x12, 0x00, 0x32, 0x05];
    v.extend_from_slice(if keyframe { &[0x11, 0x22, 0x33, 0x44, 0x55] } else { &[0xAA; 5] });
    let mut p = Packet::new(id, dts, dts, 1000, v);
    p.keyframe = keyframe;
    p
}

fn audio_packet(id: u32, dts: i64) -> Packet {
    Packet::new(id, dts, dts, 1024, vec![0x21, 0x10, 0x05, 0x55, 0x66])
}

/// `seconds` 秒 / 30fps 的画面采样。`offset` 给一个非零起点（验证起始归一化）。
fn video_packets(id: u32, seconds: f64, offset: i64) -> Vec<Packet> {
    let n = (seconds * 30.0).round() as usize;
    (0..n)
        .map(|i| video_packet(id, offset + i as i64 * 1000, i % 30 == 0))
        .collect()
}

/// `seconds` 秒 / 约 43 帧每秒的声音采样。
fn audio_packets(id: u32, seconds: f64, offset: i64) -> Vec<Packet> {
    let n = (seconds * 43.0).round() as usize;
    (0..n).map(|i| audio_packet(id, offset + i as i64 * 1024)).collect()
}

fn write_fmp4(path: &Path, tracks: &[Track], packets: &[Packet]) {
    let mut m = Mp4Muxer::new();
    m.set_fragment_duration_us(1_000_000);
    m.set_tracks(tracks).expect("声明轨道");
    for p in packets {
        m.write_packet(p).expect("写采样");
    }
    std::fs::write(path, m.finish().expect("收尾")).expect("落盘");
}

/// 造一个 **MPEG-TS** 输入：老站点与 HLS 的分片就是这个容器。
fn write_ts(path: &Path, tracks: &[Track], packets: &[Packet]) {
    let mut m = TsMuxer::new();
    m.set_tracks(tracks).expect("声明轨道");
    for p in packets {
        m.write_packet(p).expect("写采样");
    }
    let bytes = m.finish().expect("收尾");
    std::fs::write(path, bytes).expect("落盘");
}

/// 造一个**裸 MP3**（`.mp3`）：MPEG-1 Layer III / 44.1 kHz / 128 kbps / 立体声。
///
/// 帧长 417 字节、每帧 1152 采样 —— 这是 `me-container` 的裸 MP3 探测与解封装
/// 认的那套头（`0xFF 0xFB 0x90 0x00`），不是随便凑的字节。
fn write_mp3(path: &Path, seconds: f64) {
    let frames = (seconds * 44_100.0 / 1152.0).round() as usize;
    let mut v = Vec::with_capacity(frames * 417);
    for _ in 0..frames {
        v.extend_from_slice(&[0xFF, 0xFB, 0x90, 0x00]);
        v.extend(std::iter::repeat_n(0x11u8, 417 - 4));
    }
    std::fs::write(path, v).expect("落盘");
}

fn ac3_track(id: u32) -> Track {
    // AC-3 的 TS `stream_type` 是 0x81；它在 TS / 裸流里**没有配置盒**
    let mut t = Track::new(id, MediaKind::Audio, CodecId::Ac3, Timebase::new(48_000));
    t.sample_rate = 48_000;
    t.channels = 6;
    t
}

/// 一条 AC-3 采样：`syncword + crc1` 之后是字段（这里取 5.1：acmod=7、LFE 开）。
///
/// 字段按位拼（`fscod(2) frmsizecod(6) bsid(5) bsmod(3) acmod(3)
/// cmixlev(2) surmixlev(2) lfeon(1)`），后面对齐到字节再补填充。
fn ac3_packet(id: u32, dts: i64) -> Packet {
    let mut bits: Vec<u8> = Vec::new();
    let mut push = |v: u32, n: u32| {
        for i in (0..n).rev() {
            bits.push(((v >> i) & 1) as u8);
        }
    };
    push(0, 2); // fscod = 48 kHz
    push(38, 6); // frmsizecod → bit_rate_code = 19
    push(8, 5); // bsid = 8（AC-3）
    push(0, 3); // bsmod
    push(7, 3); // acmod = 7（5.1：cmixlev + surmixlev 都要跳）
    push(0, 2);
    push(0, 2);
    push(1, 1); // lfeon = 1

    let mut data = vec![0x0B, 0x77, 0x00, 0x00];
    let mut byte = 0u8;
    let mut n = 0;
    for b in bits {
        byte = (byte << 1) | b;
        n += 1;
        if n == 8 {
            data.push(byte);
            byte = 0;
            n = 0;
        }
    }
    if n > 0 {
        data.push(byte << (8 - n));
    }
    data.resize(512, 0);

    Packet::new(id, dts, dts, 1536, data)
}

fn tmp_dir(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("zuvrust-scn-{tag}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("建临时目录");
    dir
}

// ───────────────────────── CLI 驱动 ─────────────────────────

fn run(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(BIN).args(args).output().expect("启动引擎");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

/// 合并并断言成功，返回 stdout。
fn mux_ok(out: &Path, inputs: &[&Path]) -> String {
    let mut args: Vec<String> = vec!["mux".into(), out.to_string_lossy().to_string()];
    for p in inputs {
        args.push(p.to_string_lossy().to_string());
    }
    args.push("--json".into());
    let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let (ok, stdout, stderr) = run(&refs);
    assert!(
        ok,
        "合并应当成功：out={} inputs={:?}\nstderr={stderr}",
        out.display(),
        inputs
    );
    stdout
}

/// 合并并断言失败，返回 stderr。
fn mux_err(out: &Path, inputs: &[&Path]) -> String {
    let mut args: Vec<String> = vec!["mux".into(), out.to_string_lossy().to_string()];
    for p in inputs {
        args.push(p.to_string_lossy().to_string());
    }
    args.push("--json".into());
    let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let (ok, stdout, stderr) = run(&refs);
    assert!(
        !ok,
        "这一步**应当**失败：out={} inputs={:?}\nstdout={stdout}",
        out.display(),
        inputs
    );
    stderr
}

/// `probe` 产物的 JSON（断言探测成功）。
fn probe_json(path: &Path) -> String {
    let (ok, stdout, stderr) = run(&["probe", path.to_string_lossy().as_ref(), "--json"]);
    assert!(ok, "probe 应当成功：{}\nstderr={stderr}", path.display());
    stdout
}

fn duration_us(probe: &str) -> i64 {
    probe
        .split("\"durationUs\":")
        .nth(1)
        .and_then(|s| s.split([',', '}']).next())
        .and_then(|s| s.trim().parse::<i64>().ok())
        .unwrap_or(0)
}

fn track_count(probe: &str, kind: &str) -> usize {
    probe.matches(&format!("\"kindName\":\"{kind}\"")).count()
}

/// 产物里每条采样的 `(输出轨号, dts)`，按读出顺序。
fn read_dts(path: &Path) -> Vec<(u32, i64)> {
    let src: Arc<dyn ReadAt> = Arc::new(FileSource::open(path).expect("打开产物"));
    let mut d = demux::open(src, &path.to_string_lossy()).expect("开解封装器");
    let mut out = Vec::new();
    while let Some(p) = d.next_packet().expect("读采样") {
        out.push((p.track_id, p.dts));
    }
    out
}

fn assert_close(actual: i64, expect: i64, tol: i64, label: &str) {
    assert!(
        (actual - expect).abs() <= tol,
        "{label}：期望 {expect} 左右（容差 {tol}），实际 {actual}"
    );
}

// ───────────────────────── 1. 单角色 ─────────────────────────

#[test]
fn video_only_merges_into_a_single_track() {
    let dir = tmp_dir("v-only");
    let v = dir.join("v.mp4");
    write_fmp4(&v, &[h264_track(1)], &video_packets(1, 2.0, 0));
    let out = dir.join("out.mp4");

    mux_ok(&out, &[&v]);
    let probe = probe_json(&out);
    assert_eq!(track_count(&probe, "video"), 1, "只应有一条画面轨：{probe}");
    assert_eq!(track_count(&probe, "audio"), 0, "不该凭空多出声音轨：{probe}");
    assert_close(duration_us(&probe), 2_000_000, 300_000, "只有画面的产物时长");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn audio_only_merges_into_a_single_track() {
    let dir = tmp_dir("a-only");
    let a = dir.join("a.m4a");
    write_fmp4(&a, &[aac_track(2)], &audio_packets(2, 2.0, 0));
    let out = dir.join("out.mp4");

    mux_ok(&out, &[&a]);
    let probe = probe_json(&out);
    assert_eq!(track_count(&probe, "audio"), 1, "只应有一条声音轨：{probe}");
    assert_eq!(track_count(&probe, "video"), 0, "不该凭空多出画面轨：{probe}");

    let _ = std::fs::remove_dir_all(&dir);
}

// ───────────────────────── 2. 段数不等 ─────────────────────────

/// 画面 3 段、声音 2 段（真实下载里两边的分片数确实可能不同）。
#[test]
fn more_video_segments_than_audio_segments() {
    let dir = tmp_dir("v3a2");
    let (v1, v2, v3) = (dir.join("v1.mp4"), dir.join("v2.mp4"), dir.join("v3.mp4"));
    let (a1, a2) = (dir.join("a1.mp4"), dir.join("a2.mp4"));
    for p in [&v1, &v2, &v3] {
        write_fmp4(p, &[h264_track(1)], &video_packets(1, 1.0, 0));
    }
    for p in [&a1, &a2] {
        write_fmp4(p, &[aac_track(2)], &audio_packets(2, 1.0, 0));
    }
    let out = dir.join("out.mp4");

    mux_ok(&out, &[&v1, &v2, &v3, &a1, &a2]);
    let probe = probe_json(&out);
    assert_eq!(track_count(&probe, "video"), 1, "画面应当是一整条轨：{probe}");
    assert_eq!(track_count(&probe, "audio"), 1, "声音应当是一整条轨：{probe}");
    // 画面三段各 1 秒 ⇒ 约 3 秒。**不能被声音的段数影响**。
    assert_close(duration_us(&probe), 3_000_000, 400_000, "画面三段接起来的总时长");

    let _ = std::fs::remove_dir_all(&dir);
}

/// 画面 1 段、声音 3 段 —— 这是**最容易错位**的一种：
/// 按"第 i 段 = 画面第 i 个 + 声音第 i 个"的规则，第 2、3 段只有声音，
/// 它们会被平移接到第一段的**末尾**上。这里钉住两条底线：
/// 画面那条轨**不能**被声音拖长，声音也不能整段落到画面之外。
#[test]
fn more_audio_segments_than_video_segments() {
    let dir = tmp_dir("v1a3");
    let v = dir.join("v.mp4");
    let (a1, a2, a3) = (dir.join("a1.mp4"), dir.join("a2.mp4"), dir.join("a3.mp4"));
    write_fmp4(&v, &[h264_track(1)], &video_packets(1, 3.0, 0));
    for p in [&a1, &a2, &a3] {
        write_fmp4(p, &[aac_track(2)], &audio_packets(2, 1.0, 0));
    }
    let out = dir.join("out.mp4");

    mux_ok(&out, &[&v, &a1, &a2, &a3]);
    let probe = probe_json(&out);
    assert_eq!(track_count(&probe, "video"), 1, "画面应当是一整条轨：{probe}");
    assert_eq!(track_count(&probe, "audio"), 1, "声音应当是一整条轨：{probe}");

    // 画面只有一段 3 秒 —— 它的 dts 不能被声音（合计 3 秒）往后推。
    let packets = read_dts(&out);
    let video_first = packets.iter().find(|(id, _)| *id == 1).map(|(_, d)| *d).expect("画面轨");
    let video_last = packets
        .iter()
        .filter(|(id, _)| *id == 1)
        .map(|(_, d)| *d)
        .max()
        .expect("画面轨");
    assert_eq!(video_first, 0, "画面起点应当归零，实际 {video_first}");
    // 画面轨时基是 30_000 ⇒ 3 秒 = 90_000 刻度（这里比的是刻度，不是微秒）
    assert!(
        video_last <= 90_000 + 3_000,
        "画面只有 3 秒，dts 不该被别的轨推得更远，实际末尾 {video_last} 刻度"
    );

    // 声音三段必须**首尾相接**：第 2、3 段接在第 1 段末尾，中间不能出现空洞。
    // 用"相邻采样间隔"判 —— 音频一帧是 1024 刻度（44.1kHz），
    // 段间若按"全片最大末尾"平移，a2 会被推到画面结束（3 秒）之后，断层就是 2 秒。
    let audio: Vec<i64> = packets
        .iter()
        .filter(|(id, _)| *id == 2)
        .map(|(_, d)| *d)
        .collect();
    assert!(audio.len() > 100, "声音采样太少（{} 条），素材有问题", audio.len());
    let gap = audio.windows(2).find(|w| w[1] - w[0] > 4_096);
    assert!(
        gap.is_none(),
        "声音中间出现了断层（{} → {}）：第 2/3 段被推到画面之后了",
        gap.map(|w| w[0]).unwrap_or(0),
        gap.map(|w| w[1]).unwrap_or(0)
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ───────────────────────── 3. 跨容器 ─────────────────────────

/// 老站点与 HLS 的分片是 **TS**：这是"别人的文件"里最常见的一种。
#[test]
fn ts_input_merges_into_mp4() {
    let dir = tmp_dir("ts-in");
    let v = dir.join("v.ts");
    let a = dir.join("a.ts");
    write_ts(&v, &[h264_track(1)], &video_packets(1, 1.0, 0));
    write_ts(&a, &[aac_track(2)], &audio_packets(2, 1.0, 0));
    let out = dir.join("out.mp4");

    mux_ok(&out, &[&v, &a]);
    let probe = probe_json(&out);
    assert!(probe.contains("\"codec\":\"h264\""), "画面应当仍是 H.264：{probe}");
    assert!(probe.contains("\"codec\":\"aac\""), "声音应当仍是 AAC：{probe}");
    assert_eq!(track_count(&probe, "video"), 1);
    assert_eq!(track_count(&probe, "audio"), 1);

    let _ = std::fs::remove_dir_all(&dir);
}

/// fMP4 分片 → **TS 产物**：AAC 要从"裸帧"补回 ADTS 才能被 TS 播放器认出来。
#[test]
fn fmp4_input_merges_into_ts() {
    let dir = tmp_dir("ts-out");
    let v = dir.join("v.mp4");
    let a = dir.join("a.mp4");
    write_fmp4(&v, &[h264_track(1)], &video_packets(1, 1.0, 0));
    write_fmp4(&a, &[aac_track(2)], &audio_packets(2, 1.0, 0));
    let out = dir.join("out.ts");

    let stdout = mux_ok(&out, &[&v, &a]);
    assert!(stdout.contains("\"container\":\"mpeg-ts\""), "目标容器应当是 TS：{stdout}");

    // 产物要能被自己读回来（ADTS 往返的闭环）
    let probe = probe_json(&out);
    assert!(probe.contains("\"codec\":\"h264\""), "TS 产物里的画面：{probe}");
    assert!(probe.contains("\"codec\":\"aac\""), "TS 产物里的声音：{probe}");
    assert_eq!(track_count(&probe, "audio"), 1, "AAC 必须能读回来（ADTS 写对了）：{probe}");

    let _ = std::fs::remove_dir_all(&dir);
}

// ───────────────────────── 4. 编解码器组合 ─────────────────────────

/// B 站现在最常见的组合：**AV1 画面 + AAC 声音**（Hi-Res 那批才是 FLAC）。
#[test]
fn av1_with_aac_merges() {
    let dir = tmp_dir("av1-aac");
    let v = dir.join("v.mp4");
    let a = dir.join("a.m4a");
    write_fmp4(&v, &[av1_track(1)], &{
        let n = 60;
        (0..n).map(|i| av1_packet(1, i as i64 * 1000, i % 30 == 0)).collect::<Vec<_>>()
    });
    write_fmp4(&a, &[aac_track(2)], &audio_packets(2, 2.0, 0));
    let out = dir.join("out.mp4");

    mux_ok(&out, &[&v, &a]);
    let probe = probe_json(&out);
    assert!(probe.contains("\"codec\":\"av1\""), "画面应当是 AV1：{probe}");
    assert!(probe.contains("\"codec\":\"aac\""), "声音应当是 AAC：{probe}");

    let _ = std::fs::remove_dir_all(&dir);
}

/// **MP3 音频 + MP4 画面** —— 市面很常见的一种组合。
///
/// 修之前 fMP4 的允许列表里没有 MP3，`mux` 会以"fmp4 装不下这条轨道：mp3 / Audio"
/// 直接拒绝 —— 而**读侧本来就在认** `mp4a` + `esds`(OTI 0x6B) 的 MP3
/// （`me-container/src/mp4.rs`）。写侧补齐之后两边才对称：
/// 引擎读得出来的，也能原样写回去。
#[test]
fn mp3_audio_merges_into_mp4() {
    let dir = tmp_dir("mp3");
    let v = dir.join("v.mp4");
    let a = dir.join("a.mp3");
    write_fmp4(&v, &[h264_track(1)], &video_packets(1, 1.0, 0));
    write_mp3(&a, 1.0);
    let out = dir.join("out.mp4");

    mux_ok(&out, &[&v, &a]);
    let probe = probe_json(&out);
    assert!(probe.contains("\"codec\":\"mp3\""), "声音应当被认成 MP3：{probe}");
    assert_eq!(track_count(&probe, "audio"), 1, "{probe}");
    assert_eq!(track_count(&probe, "video"), 1, "{probe}");

    let _ = std::fs::remove_dir_all(&dir);
}

/// **AC-3（来自 TS）合并进 MP4** —— 关键在 `dac3`。
///
/// TS 里的 AC-3 没有配置盒，参数只写在 syncframe 头里；而 fMP4 必须有 `dac3`。
/// `mux` 层会在写 moov **之前**从第一帧补出来，所以这里除"能出 AC-3 轨"之外，
/// 还要断言**配置盒真的在产物里**（`extradataBytes == 3`）——
/// 补 `dac3` 那一步一旦漏掉，产物会是一条"读得出 ac3、配置却为空"的轨：
/// 播放器按默认声道布局解，声音是错的但文件看着没毛病。
#[test]
fn ac3_audio_from_ts_merges_into_mp4() {
    let dir = tmp_dir("ac3");
    let src = dir.join("in.ts");
    let mut packets = video_packets(1, 1.0, 0);
    packets.extend((0..20).map(|i| ac3_packet(2, i * 1536)));
    write_ts(&src, &[h264_track(1), ac3_track(2)], &packets);
    let out = dir.join("out.mp4");

    mux_ok(&out, &[&src]);
    let probe = probe_json(&out);
    assert!(probe.contains("\"codec\":\"ac3\""), "声音应当被认成 AC-3：{probe}");
    assert_eq!(track_count(&probe, "audio"), 1, "{probe}");

    let ac3_bytes = probe
        .split("\"codec\":\"ac3\"")
        .nth(1)
        .and_then(|s| s.split("\"extradataBytes\":").nth(1))
        .and_then(|s| s.split([',', '}']).next())
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(0);
    assert_eq!(ac3_bytes, 3, "dac3 应当是 3 字节，实际 {ac3_bytes}（{probe}）");

    let _ = std::fs::remove_dir_all(&dir);
}

// ───────────────────────── 5. 拒绝路径 ─────────────────────────

/// AV1 没有标准的 TS `stream_type`：**必须在开工前拒绝，且不留半截产物**。
#[test]
fn av1_into_ts_is_rejected_without_leaving_a_partial_file() {
    let dir = tmp_dir("av1-ts");
    let v = dir.join("v.mp4");
    write_fmp4(&v, &[av1_track(1)], &{
        let n = 30;
        (0..n).map(|i| av1_packet(1, i as i64 * 1000, i % 30 == 0)).collect::<Vec<_>>()
    });
    let out = dir.join("out.ts");

    let stderr = mux_err(&out, &[&v]);
    assert!(
        stderr.contains("装不下") || stderr.contains("Unsupported"),
        "应当明确说明「这个容器装不下这条轨」：{stderr}"
    );
    assert!(!out.exists(), "被拒绝的合并**不能**留下半截产物");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn unknown_output_extension_is_rejected() {
    let dir = tmp_dir("bad-ext");
    let v = dir.join("v.mp4");
    write_fmp4(&v, &[h264_track(1)], &video_packets(1, 1.0, 0));
    let out = dir.join("out.avi");

    let stderr = mux_err(&out, &[&v]);
    assert!(
        stderr.contains("不认识的输出容器"),
        "应当说清是扩展名不认识：{stderr}"
    );
    assert!(!out.exists(), "不能留下产物");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_corrupt_input_fails_and_leaves_no_output() {
    let dir = tmp_dir("corrupt");
    let junk = dir.join("junk.mp4");
    std::fs::write(&junk, vec![0x5A; 4096]).expect("写垃圾文件");
    let out = dir.join("out.mp4");

    let _ = mux_err(&out, &[&junk]);
    assert!(!out.exists(), "输入都读不了，绝不能留下产物");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_missing_output_directory_is_rejected() {
    let dir = tmp_dir("no-dir");
    let v = dir.join("v.mp4");
    write_fmp4(&v, &[h264_track(1)], &video_packets(1, 1.0, 0));
    let out = dir.join("nope").join("out.mp4");

    let _ = mux_err(&out, &[&v]);
    assert!(!out.exists());

    let _ = std::fs::remove_dir_all(&dir);
}

// ───────────────────────── 6. 复杂时间轴 ─────────────────────────

/// 24 段串接：时间轴必须严格递增，总时长是各段之和。
///
/// 这条把"段间平移"从两个段放大到很多段 —— 只要有**一次**平移算错，
/// 后面所有段的 `tfdt` 都会整体偏移，或者 fMP4 封装器直接拒绝（它要求单调）。
#[test]
fn many_segments_concat_and_keep_a_monotonic_timeline() {
    let dir = tmp_dir("many");
    let segments = 24usize;
    let mut videos = Vec::new();
    let mut audios = Vec::new();
    for i in 0..segments {
        let v = dir.join(format!("v{i}.mp4"));
        let a = dir.join(format!("a{i}.mp4"));
        write_fmp4(&v, &[h264_track(1)], &video_packets(1, 0.5, 0));
        write_fmp4(&a, &[aac_track(2)], &audio_packets(2, 0.5, 0));
        videos.push(v);
        audios.push(a);
    }
    let out = dir.join("out.mp4");

    let mut inputs: Vec<&Path> = Vec::new();
    for i in 0..segments {
        inputs.push(&videos[i]);
        inputs.push(&audios[i]);
    }
    mux_ok(&out, &inputs);

    let probe = probe_json(&out);
    // 24 × 0.5 秒 = 12 秒
    assert_close(duration_us(&probe), 12_000_000, 600_000, "24 段串接的总时长");

    // 每条轨的 dts 严格递增（解码序不允许回退）
    let packets = read_dts(&out);
    for id in [1u32, 2u32] {
        let mut prev: Option<i64> = None;
        for (tid, dts) in packets.iter().filter(|(t, _)| *t == id) {
            let _ = tid;
            if let Some(p) = prev {
                assert!(*dts > p, "轨 {id} 的 dts 回退了：{p} → {dts}");
            }
            prev = Some(*dts);
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// 输入自己的 DTS 不从 0 开始（下载分片常常带着累计时间戳）：
/// 合并必须把起点归零，而不是把 1 秒的空洞带进产物。
#[test]
fn a_nonzero_input_start_is_normalized_to_zero() {
    let dir = tmp_dir("offset");
    let v = dir.join("v.mp4");
    let a = dir.join("a.mp4");
    // 画面从 30_000 刻度（1 秒）起步，声音从 44_100（1 秒）起步
    write_fmp4(&v, &[h264_track(1)], &video_packets(1, 1.0, 30_000));
    write_fmp4(&a, &[aac_track(2)], &audio_packets(2, 1.0, 44_100));
    let out = dir.join("out.mp4");

    mux_ok(&out, &[&v, &a]);
    let packets = read_dts(&out);
    for id in [1u32, 2u32] {
        let first = packets.iter().find(|(t, _)| *t == id).map(|(_, d)| *d);
        assert_eq!(first, Some(0), "轨 {id} 的起点应当归零，实际 {first:?}");
    }

    let probe = probe_json(&out);
    assert_close(duration_us(&probe), 1_000_000, 300_000, "归零后的时长应当是 1 秒");

    let _ = std::fs::remove_dir_all(&dir);
}

// ───────────────────────── 7. 混合角色 ─────────────────────────

/// 一个**音视频都有**的输入 + 一条只有声音的输入（"给整片补一段音轨"的用法）。
///
/// 覆盖 `mux.rs` 里"同一个输入同时供画面与声音 ⇒ 并进同一条 `SegInput`"那条分支：
/// 不并的话那个输入会被拉两遍 —— 第二遍已经读到末尾，产物会缺一段声音。
#[test]
fn a_combined_input_plus_an_extra_audio_input() {
    let dir = tmp_dir("mixed");
    let av = dir.join("av.mp4");
    let extra = dir.join("extra.m4a");

    // 整片：2 秒，音视频都有（轨道号 1 = 画面 / 2 = 声音）
    {
        let mut m = Mp4Muxer::new();
        m.set_fragment_duration_us(1_000_000);
        m.set_tracks(&[h264_track(1), aac_track(2)]).expect("声明轨道");
        for p in video_packets(1, 2.0, 0) {
            m.write_packet(&p).expect("写画面");
        }
        for p in audio_packets(2, 2.0, 0) {
            m.write_packet(&p).expect("写声音");
        }
        std::fs::write(&av, m.finish().expect("收尾")).expect("落盘");
    }
    write_fmp4(&extra, &[aac_track(2)], &audio_packets(2, 1.0, 0));
    let out = dir.join("out.mp4");

    mux_ok(&out, &[&av, &extra]);
    let probe = probe_json(&out);
    assert_eq!(track_count(&probe, "video"), 1, "画面只应有一条轨：{probe}");
    assert_eq!(track_count(&probe, "audio"), 1, "声音只应有一条轨：{probe}");

    // 画面那条轨仍然是 2 秒（= 60_000 刻度），不能被追加的声音拖长
    let packets = read_dts(&out);
    let v_last = packets
        .iter()
        .filter(|(id, _)| *id == 1)
        .map(|(_, d)| *d)
        .max()
        .unwrap_or(0);
    assert!(
        v_last <= 60_000 + 3_000,
        "画面只有 2 秒（60_000 刻度），实际末尾 {v_last}"
    );

    // 追加的声音必须接在原声音末尾（约 2 秒 = 88_200 刻度）之后、且中间没有洞
    let audio: Vec<i64> = packets.iter().filter(|(id, _)| *id == 2).map(|(_, d)| *d).collect();
    let gap = audio.windows(2).find(|w| w[1] - w[0] > 4_096);
    assert!(
        gap.is_none(),
        "声音中间出现断层（{} → {}）：追加的那段没接上",
        gap.map(|w| w[0]).unwrap_or(0),
        gap.map(|w| w[1]).unwrap_or(0)
    );

    let _ = std::fs::remove_dir_all(&dir);
}
