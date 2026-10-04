//! "只搬字节、不重编码"的**证据**。
//!
//! 用户的原话是"确保合并出来的视频音频是无损的"。这件事不能靠"我觉得没转码"，
//! 要有可复现的比对：把输入与产物的采样**逐条逐字节**过一遍。
//!
//! 比较口径（三件事决定了可以直接比字节）：
//! - 容器里视频是**长度前缀**布局、引擎内部统一成 **Annex-B**，而解封装读回来的
//!   采样**一律归一化到 Annex-B** —— 所以两边直接逐字节比就是对的（那次转换是
//!   纯字节搬运、可逆，不是解码重编码）；
//! - 音频在 fMP4 里本来就是裸帧，逐字节比；
//! - 产物会重编轨号（1 = 画面、2 = 声音），所以**按角色对齐**而不是按轨号。

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use me_container::demux;
use me_core::{CodecId, FileSource, MediaKind, Packet, ReadAt, Timebase, Track};
use me_mux::Mp4Muxer;

const BIN: &str = env!("CARGO_BIN_EXE_zuvrust");

/// 按角色取一个文件里的全部采样（形态已归一化）。
fn samples_by_kind(path: &Path) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let src: Arc<dyn ReadAt> = Arc::new(FileSource::open(path).expect("打开输入"));
    let info = me_container::probe(src.clone(), &path.to_string_lossy()).expect("探测");
    let kinds: Vec<(u32, MediaKind)> = info.tracks.iter().map(|t| (t.id, t.kind)).collect();
    let mut d = demux::open(src, &path.to_string_lossy()).expect("开解封装器");
    let (mut video, mut audio) = (Vec::new(), Vec::new());
    while let Some(p) = d.next_packet().expect("读采样") {
        match kinds.iter().find(|(id, _)| *id == p.track_id).map(|(_, k)| *k) {
            Some(MediaKind::Video) => video.push(p.data),
            Some(MediaKind::Audio) => audio.push(p.data),
            _ => {}
        }
    }
    (video, audio)
}

/// 逐条比两串采样，返回总字节数；不一致当场报错。
fn assert_sample_exact(label: &str, input: &[Vec<u8>], output: &[Vec<u8>]) -> u64 {
    assert_eq!(
        input.len(),
        output.len(),
        "{label}：采样**条数**都不一致（{} vs {}）—— 合并丢了或多了数据",
        input.len(),
        output.len()
    );
    let mut bytes = 0u64;
    for (i, (a, b)) in input.iter().zip(output.iter()).enumerate() {
        assert_eq!(
            a, b,
            "{label}：第 {i} 条采样（{} 字节 vs {} 字节）被改动了 —— 合并必须只搬字节",
            a.len(),
            b.len()
        );
        bytes += a.len() as u64;
    }
    bytes
}

fn run_mux(out: &Path, inputs: &[&Path]) {
    let mut args: Vec<String> = vec!["mux".into(), out.to_string_lossy().to_string()];
    for p in inputs {
        args.push(p.to_string_lossy().to_string());
    }
    args.push("--json".into());
    let r = Command::new(BIN).args(&args).output().expect("启动引擎");
    assert!(
        r.status.success(),
        "合并失败：{}",
        String::from_utf8_lossy(&r.stderr)
    );
}

fn tmp_dir(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("lerxu-{tag}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("建临时目录");
    dir
}

// ─────────────────────── 合成媒体（常规测试） ───────────────────────

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

/// 一条视频采样：**4 字节起始码** + 假 NAL（容器写出去是长度前缀，读回来应当还原）。
fn video_packet(dts: i64, keyframe: bool) -> Packet {
    let mut v = vec![0u8, 0, 0, 1];
    v.push(if keyframe { 0x65 } else { 0x41 });
    v.extend_from_slice(&[0x11, 0x22, 0x33, 0x44, 0x55]);
    let mut p = Packet::new(1, dts, dts, 1000, v);
    p.keyframe = keyframe;
    p
}

fn audio_packet(dts: i64) -> Packet {
    Packet::new(2, dts, dts, 1024, vec![0x21, 0x10, 0x05, 0x55, 0x66])
}

fn write_inputs(dir: &Path) -> (PathBuf, PathBuf) {
    let v = dir.join("v.mp4");
    let mut m = Mp4Muxer::new();
    m.set_fragment_duration_us(1_000_000);
    m.set_tracks(&[video_track()]).unwrap();
    for i in 0..90 {
        m.write_packet(&video_packet(i * 1000, i % 30 == 0)).unwrap();
    }
    std::fs::write(&v, m.finish().unwrap()).unwrap();

    let a = dir.join("a.mp4");
    let mut m = Mp4Muxer::new();
    m.set_fragment_duration_us(1_000_000);
    m.set_tracks(&[audio_track()]).unwrap();
    for i in 0..129 {
        m.write_packet(&audio_packet(i * 1024)).unwrap();
    }
    std::fs::write(&a, m.finish().unwrap()).unwrap();
    (v, a)
}

#[test]
fn merged_output_carries_the_original_samples_byte_for_byte() {
    let dir = tmp_dir("lossless");
    let (v, a) = write_inputs(&dir);
    let out = dir.join("out.mp4");
    run_mux(&out, &[&v, &a]);

    let (iv, _) = samples_by_kind(&v);
    let (_, ia) = samples_by_kind(&a);
    let (ov, oa) = samples_by_kind(&out);
    let vb = assert_sample_exact("视频", &iv, &ov);
    let ab = assert_sample_exact("音频", &ia, &oa);

    assert!(vb > 0 && ab > 0, "两条轨都要有数据");
    println!("无损校验通过：视频 {} 条 / {} 字节，音频 {} 条 / {} 字节", iv.len(), vb, ia.len(), ab);
    let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────── 真实文件（手动触发） ───────────────────────

#[test]
#[ignore = "需要 LERXU_VERIFY_SINGLE 指向一个真实的音视频文件（单文件换容器）"]
fn real_single_file_remux_is_lossless() {
    let src = PathBuf::from(std::env::var("LERXU_VERIFY_SINGLE").expect("LERXU_VERIFY_SINGLE"));
    let out = std::env::temp_dir().join(format!("lerxu-lossless-single-{}.mp4", std::process::id()));

    let t0 = std::time::Instant::now();
    run_mux(&out, &[&src]);
    let merge_ms = t0.elapsed().as_millis();

    let (iv, ia) = samples_by_kind(&src);
    let (ov, oa) = samples_by_kind(&out);
    let vb = assert_sample_exact("视频", &iv, &ov);
    let ab = assert_sample_exact("音频", &ia, &oa);

    println!(
        "真实单文件无损校验通过（{} ms）：视频 {} 条 / {} 字节，音频 {} 条 / {} 字节",
        merge_ms, iv.len(), vb, ia.len(), ab
    );
    let _ = std::fs::remove_file(&out);
}

/// 拿一对真实文件做无损校验。默认不跑（要指定文件），手动触发：
///
/// ```sh
/// LERXU_VERIFY_VIDEO=/path/video.mp4 LERXU_VERIFY_AUDIO=/path/audio.m4a \
///   cargo test -p me-cli --test lossless -- --ignored --nocapture
/// ```
#[test]
#[ignore = "需要 LERXU_VERIFY_VIDEO / LERXU_VERIFY_AUDIO 指向一对真实音视频"]
fn real_files_are_merged_losslessly() {
    let v = PathBuf::from(std::env::var("LERXU_VERIFY_VIDEO").expect("LERXU_VERIFY_VIDEO"));
    let a = PathBuf::from(std::env::var("LERXU_VERIFY_AUDIO").expect("LERXU_VERIFY_AUDIO"));
    let out = std::env::temp_dir().join(format!("lerxu-lossless-{}.mp4", std::process::id()));

    let t0 = std::time::Instant::now();
    run_mux(&out, &[&v, &a]);
    let merge_ms = t0.elapsed().as_millis();

    let (iv, _) = samples_by_kind(&v);
    let (_, ia) = samples_by_kind(&a);
    let (ov, oa) = samples_by_kind(&out);
    let vb = assert_sample_exact("视频", &iv, &ov);
    let ab = assert_sample_exact("音频", &ia, &oa);
    let out_bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);

    println!(
        "真实文件无损校验通过（{} ms）：\n  视频 {} 条采样 / {} 字节\n  音频 {} 条采样 / {} 字节\n  产物 {} 字节",
        merge_ms, iv.len(), vb, ia.len(), ab, out_bytes
    );
    let _ = std::fs::remove_file(&out);
}
