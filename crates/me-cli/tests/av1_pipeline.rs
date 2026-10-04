//! AV1 播放链路的集成测试：**容器里的 av1C → 解码器选择 → 自研软解的第一层**。
//!
//! 与 `mux_cli.rs` 同一个约定：真的启动一次 `zuvrust` 进程，走完整的
//! 「命令行 → 干活 → NDJSON」。为什么这条要单独钉住：
//!
//! - 用户实测过 AV1 素材（B 站已经在用），而引擎以前在解码这一档直接报
//!   "解码目前只接了 H.264"；
//! - 现在 AV1 的路径是"**有硬解就用硬解，没有就落到自研软解**"，两条路的
//!   判定条件（`VTIsHardwareDecodeSupported`）随机器而变 —— 这条测试在
//!   Intel 机与 M3 机上都要给出**可读的**结果，而不是一句含糊的失败。

use std::path::Path;
use std::process::Command;

use me_core::{CodecId, MediaKind, Packet, Timebase, Track};
use me_mux::Mp4Muxer;

/// 引擎二进制（Cargo 为 bin target 注入的路径）。
const BIN: &str = env!("CARGO_BIN_EXE_zuvrust");

// ── 造一个"像真的"AV1 样本 ──────────────────────────────────────────
//
// 真实 AV1 采样长这样：时间分隔符 OBU → 序列头 OBU → 帧 OBU。这里按规范位序
// 手写一个 1920×1080 / 8bit 4:2:0 的序列头（简化静图模式，最短的形态）。
// 位写入器与 `me-decode` 的测试专用版本同序 —— 这里复制一份是因为那个是
// `cfg(test)` 内部的，跨 crate 拿不到；位序的权威说明在
// `me-decode/src/video/sequence_header.rs`。

#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    bit: u8,
}

impl BitWriter {
    fn f(&mut self, value: u32, bits: u32) -> &mut Self {
        for i in (0..bits).rev() {
            let b = ((value >> i) & 1) as u8;
            if self.bit == 0 {
                self.bytes.push(0);
            }
            let last = self.bytes.len() - 1;
            self.bytes[last] |= b << (7 - self.bit);
            self.bit = (self.bit + 1) % 8;
        }
        self
    }

    fn bit(&mut self, v: bool) -> &mut Self {
        self.f(v as u32, 1)
    }
}

/// 带长度字段的 OBU：`[(type<<3)|has_size, leb128 长度, 载荷…]`（这里长度都 < 128）。
fn obu(obu_type: u8, payload: &[u8]) -> Vec<u8> {
    let mut v = vec![(obu_type << 3) | 0b0000_0010, payload.len() as u8];
    v.extend_from_slice(payload);
    v
}

/// 1920×1080 / 8 位 4:2:0 的序列头载荷（简化静图模式）。
fn sequence_header_1080p() -> Vec<u8> {
    let mut w = BitWriter::default();
    w.f(0, 3); // seq_profile = 0
    w.bit(true); // still_picture
    w.bit(true); // reduced_still_picture_header
    w.f(8, 5); // seq_level_idx
    w.f(10, 4); // frame_width_bits_minus_1 → 11 位
    w.f(10, 4); // frame_height_bits_minus_1
    w.f(1919, 11); // max_frame_width_minus_1 → 1920
    w.f(1079, 11); // max_frame_height_minus_1 → 1080
    w.bit(false); // use_128x128_superblock
    w.bit(false); // enable_filter_intra
    w.bit(false); // enable_intra_edge_filter
    w.bit(false); // enable_superres
    w.bit(true); // enable_cdef
    w.bit(false); // enable_restoration
    w.bit(false); // high_bitdepth → 8 位
    w.bit(false); // mono_chrome
    w.bit(false); // color_description_present_flag
    w.bit(false); // color_range
    w.f(0, 2); // chroma_sample_position
    w.bit(false); // separate_uv_delta_q
    w.bit(false); // film_grain_params_present
    w.f(1, 1); // trailing_bits（解析器会校验这一位）
    w.bytes
}

/// 一个 AV1 采样：时间分隔符 + 序列头 + 帧。
///
/// 帧的载荷是**一段完整的合法非压缩头**（含 `tile_info`/量化/分割/滤波参数，
/// 并以 `byte_alignment()` 收尾）—— 引擎现在会一路读到末尾并校验对齐，
/// 所以"随手三个字节当帧载荷"那种假样本已经过不了。
///
/// 它仍然不是真码流：帧里没有瓦片数据（没有熵编码的残差），所以解码器会在
/// "像素重建还没做"这一步停下 —— 这正是这条链路测试要确认的位置。
fn av1_sample() -> Vec<u8> {
    let mut v = obu(2, &[]); // 时间分隔符
    v.extend(obu(1, &sequence_header_1080p())); // 序列头
    v.extend(obu(6, &key_frame_header_reduced())); // 帧
    v
}

/// 关键帧的完整非压缩头（配合本文件那份**简化静图模式**的序列头）。
///
/// 属性取自 `sequence_header_1080p()`：1920×1080、64 像素超级块、CDEF 开、
/// 环路恢复关、非 mono、`separate_uv_delta_q = 0`、无超分、无 order hint。
fn key_frame_header_reduced() -> Vec<u8> {
    let mut w = BitWriter::default();
    // 简化静图模式：show_existing_frame / frame_type / show_frame / 尺寸都写死，
    // 连 frame_size_override 与 order_hint 都不读
    w.bit(false); // disable_cdf_update
    w.bit(false); // allow_screen_content_tools
    w.bit(false); // render_and_frame_size_different
    // allow_intrabc：allow_screen_content_tools = 0 → 不读
    // disable_frame_end_update_cdf：简化模式强制 1 → 不读
    write_min_params(&mut w);
    while w.bit != 0 {
        w.f(0, 1); // byte_alignment()：补 0 到字节边界
    }
    // 单瓦片流：帧头之后就是那唯一瓦片的算术流。这里补一段确定性字节当瓦片数据
    // —— 不是真的算术流（真流要 M4 才解得开），但够 `init_symbol` 把状态建起来。
    let mut out = w.bytes;
    out.extend((0..64u32).map(|i| ((i * 37 + 11) % 251) as u8));
    out
}

/// 非压缩头的参数段（`tile_info()` → `film_grain_params()`）：每一位都走
/// "不需要读更多位"的那一支 —— 单瓦片、无 delta、无分割、无环路恢复、
/// 无全局运动、无胶片颗粒。
///
/// 位序与 `me-decode/src/video/params.rs` 里的解析顺序一一对应。之所以在这里
/// 再写一遍：那份合成工具是 `cfg(test)` 内部的，跨 crate 拿不到。
/// **改解析器时要跟着看这里** —— 它是"合成样本还能不能被引擎接受"的唯一依据。
fn write_min_params(w: &mut BitWriter) {
    // tile_info()：uniform = 1，两个 while 各读一位 0 就 break；
    // 单瓦片 → 不读 context_update_tile_id / tile_size_bytes
    w.bit(true); // uniform_tile_spacing_flag
    w.bit(false); // increment_tile_cols_log2 = 0
    w.bit(false); // increment_tile_rows_log2 = 0
    // quantization_params()：base_q_idx 取非 0 → 不走无损分支
    w.f(100, 8); // base_q_idx
    w.bit(false); // DeltaQYDc：delta_coded = 0
    w.bit(false); // DeltaQUDc
    w.bit(false); // DeltaQUAc
    w.bit(false); // using_qmatrix
    // segmentation_params()
    w.bit(false); // segmentation_enabled
    // delta_q_params()（delta_lf_params 在 delta_q_present = 0 时不读位）
    w.bit(false); // delta_q_present = 0
    // loop_filter_params()：level 全 0 → 不读 level[2]/[3]
    w.f(0, 6); // loop_filter_level[0]
    w.f(0, 6); // loop_filter_level[1]
    w.f(0, 3); // loop_filter_sharpness
    w.bit(false); // loop_filter_delta_enabled
    // cdef_params()（enable_cdef = 1）
    w.f(0, 2); // cdef_damping_minus_3
    w.f(0, 2); // cdef_bits = 0 → 一组强度
    w.f(0, 4); // cdef_y_pri_strength
    w.f(0, 2); // cdef_y_sec_strength
    w.f(0, 4); // cdef_uv_pri_strength
    w.f(0, 2); // cdef_uv_sec_strength
    // lr_params()：enable_restoration = 0 → 不读
    // read_tx_mode()
    w.bit(false); // tx_mode_select = 0 → TX_MODE_LARGEST
    // frame_reference_mode / skip_mode / allow_warped_motion / global_motion：
    // 帧内帧一律不读
    // reduced_tx_set
    w.bit(false);
    // film_grain_params()：film_grain_params_present = 0 → 不读
}

fn av1_track() -> Track {
    let mut t = Track::new(1, MediaKind::Video, CodecId::Av1, Timebase::new(30_000));
    t.width = 1920;
    t.height = 1080;
    // av1C：marker(0x81) + seq_profile/level + seq_tier/位深/色度 + 保留
    t.extradata = vec![0x81, 0x00, 0x0C, 0x00];
    t
}

fn write_av1_file(path: &Path, seconds: usize) {
    write_av1_file_frames(path, seconds * 30);
}

/// 只写 `n` 帧。
///
/// 为什么需要它：自研软解现在**真的会解码**（帧内预测 → 反变换 → 去块 → CDEF），
/// 一帧 1080p 在 debug 构建下要十几秒；照 30 帧写会把整条测试拖到十分钟。
/// 链路测试只关心"能不能走通"，两帧足够。
fn write_av1_file_frames(path: &Path, frames: usize) {
    let mut m = Mp4Muxer::new();
    m.set_fragment_duration_us(1_000_000);
    m.set_tracks(&[av1_track()]).unwrap();
    for i in 0..frames {
        let dts = i as i64 * 1000;
        let keyframe = i % 30 == 0;
        let mut p = Packet::new(1, dts, dts, 1000, av1_sample());
        p.keyframe = keyframe;
        m.write_packet(&p).unwrap();
    }
    std::fs::write(path, m.finish().unwrap()).unwrap();
}

fn run(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(BIN).args(args).output().expect("启动引擎失败");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

fn tmp_dir(name: &str) -> std::path::PathBuf {
    // 留在一个固定路径上便于人工复现（`zuvrust decode /tmp/lerxu-av1-<name>/av1.mp4 …`）
    let dir = std::env::temp_dir().join(format!("lerxu-av1-{name}"));
    // 现场留在 /tmp/lerxu-av1-*/ 里，便于人工复现（见 tmp_dir 的说明）
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn av1_file_is_recognised_as_av1_by_the_engine() {
    let dir = tmp_dir("probe");
    let f = dir.join("av1.mp4");
    write_av1_file(&f, 1);

    let (ok, stdout, stderr) = run(&["probe", f.to_str().unwrap(), "--json"]);
    assert!(ok, "probe 应当成功，stderr={stderr}");
    assert!(stdout.contains("\"codec\":\"av1\""), "这条轨应当是 AV1：{stdout}");
    assert!(stdout.contains("\"width\":1920"), "画面尺寸要读出来：{stdout}");

    // 现场留在 /tmp/lerxu-av1-*/ 里，便于人工复现（见 tmp_dir 的说明）
}

#[test]
fn av1_decoding_runs_the_self_decoder_all_the_way() {
    let dir = tmp_dir("decode");
    let f = dir.join("av1.mp4");
    let out = dir.join("frames");
    write_av1_file_frames(&f, 2);

    let (ok, stdout, stderr) = run(&[
        "decode",
        f.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--json",
    ]);
    let all = format!("{stdout}\n{stderr}");
    // 方便人工复现（现场不删）：直接照这行命令再跑一次就能看到引擎的原话
    eprintln!("样本：{}", f.display());

    // 这条链路测试验证的是**形态**，不是像素值：样本的瓦片是"凑数的字节"，
    // 不是真算术流，所以它解出什么画面没有意义。要确认的是——
    // ② 引擎必须走完"自研软解"这条路，**不能再出现"还没实现"**（这正是
    //    这条测试原先的断言在防的事，只是断言方向反了）；
    // ① 报错（如果有）必须指名 AV1，不能是一句含糊的"不支持"。
    assert!(
        !all.contains("像素重建还没做") && !all.contains("还没做到像素重建"),
        "自研软解的**像素重建**不该再是缺口：{all}"
    );
    assert!(all.contains("av1") || all.contains("AV1") || ok, "要指名编码：{all}");

    let engine_has_av1_hardware = me_decode::hardware_decode_available(&CodecId::Av1);
    if engine_has_av1_hardware {
        assert!(
            all.contains("拒绝") || all.contains("VideoToolbox") || ok,
            "有 AV1 硬解时应当走到硬解那一条：{all}"
        );
    } else {
        // Intel 机没有 AV1 硬解 ⇒ 必定落到自研软解。合成样本不是真码流，
        // 所以这里可能成功（语法层能读通、像素是垃圾）也可能在数据层如实失败；
        // 但**能走到像素重建**这件事本身就是要验的（"这一步还没做"已经被上面挡掉了）。
        assert!(
            ok || all.contains("自研软解") || all.contains("码流错误"),
            "没有硬解时应当落到自研软解并说明进度：{all}"
        );
        if ok {
            // 成功：从容器/码流里读到的尺寸必须拿到（1920×1080 的序列头）
            assert!(
                all.contains("\"width\":1920") && all.contains("\"height\":1080"),
                "成功时结果里要带上从码流里读到的尺寸：{all}"
            );
        } else {
            // 失败也行，但**必须是自研软解在数据层如实失败**（指名字节层面），
            // 不能是含糊的"不支持"。
            //
            // 注意：合成样本的瓦片是"凑数字节"，算术解码器碰到它会一路产出乱码符号，
            // 于是很快就**越出瓦片末尾** —— 熵层现在有一条规范判据
            // （§8.2.6 要求 `SymbolMaxBits >= -14`）会当场把它按住并报出来。
            // 这条错误信息里没有"自研软解"字样（它是熵层的判据，比那一层更靠下），
            // 所以这里同时接受它。
            assert!(
                all.contains("自研软解") || all.contains("越过了瓦片末尾"),
                "失败时要说清走的是自研软解（或熵层如实报越界）：{all}"
            );
        }
    }

    // 现场留在 /tmp/lerxu-av1-*/ 里，便于人工复现（见 tmp_dir 的说明）
}

#[test]
fn av1_stream_mode_reports_the_backend() {
    let dir = tmp_dir("stream");
    let f = dir.join("av1.mp4");
    write_av1_file_frames(&f, 2);

    // `stream --video=frames` 是桌面播放器实际用的那条路：元信息行里带 `backend`
    // 字段，宿主据此就能回答"这次播放用的什么内核"。
    let (ok, stdout, stderr) = run(&["stream", f.to_str().unwrap(), "--video=frames", "--json"]);
    let all = format!("{stdout}\n{stderr}");
    // 成功时 stdout 里还跟着 NV12 帧数据（二进制），截一段出来再进断言消息
    let head: String = all.chars().take(600).collect();
    if ok {
        assert!(
            stdout.contains("\"backend\":\"videotoolbox\"")
                || stdout.contains("\"backend\":\"self-av1\""),
            "成功时后端名字必须出现在元信息里（M3+ 是硬解、Intel 机是自研软解）：{head}"
        );
    } else {
        assert!(
            all.contains("自研软解")
                || all.contains("VideoToolbox")
                || all.contains("拒绝")
                || all.contains("码流错误"),
            "失败时也要说清走的是哪条路：{head}"
        );
    }

    // 现场留在 /tmp/lerxu-av1-*/ 里，便于人工复现（见 tmp_dir 的说明）
}
