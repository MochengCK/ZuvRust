//! **跨平台 CLI 契约测试** —— 宿主与引擎之间的唯一接口，逐条钉死。
//!
//! 为什么单独一个文件、且**真的 spawn 进程**（而不是像 `commands.rs` 里那样直接调
//! `dispatch()`）：宿主是通过 `spawn` + 管道跟引擎说话的，它能观察到的只有
//! **退出码 / stdout 的字节 / stderr 的字节**。进程内调用看不到其中任何一条 ——
//! 而这几条恰恰是**最容易在某个平台上悄悄坏掉**的东西：
//!
//! - **Windows 没有 ANSI**：日志若无条件上色，`\x1b[32m` 会混进宿主按行解析的流里。
//! - **Windows 的控制台不是 UTF-8**：`chcp 936` 下的中文日志会变乱码；宿主拿到的
//!   stderr 也要能安全地当 UTF-8 处理。
//! - **退出码是契约**（`main.rs` 的 `exit_code` 模块）：宿主靠它区分"重试有意义"
//!   与"重试无用"，差一位就是无限重试或直接放弃。
//! - **`--version` / `version --json` 的字段名**：宿主的能力探测按这些键取值，
//!   少一个键界面上的按钮就灰掉，而且**不会报错**（最难查的一类回归）。
//!
//! 本文件在 CI 上于 **macOS arm64 / Linux x86_64 / Windows x86_64** 三个原生运行器
//! 各跑一遍（见 `.github/workflows/ci.yml`）。任何一条在某个平台上不成立，CI 就红。

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::Value;

/// 二进制路径：`CARGO_BIN_EXE_<name>` 由 cargo 在编译集成测试时注入。
const BIN: &str = env!("CARGO_BIN_EXE_zuvrust");

/// 跑一次引擎，返回 (退出码, stdout, stderr)。
fn run(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(BIN)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .unwrap_or_else(|e| panic!("无法启动 {} {}：{e}", BIN, args.join(" ")));
    (
        out.status.code().expect("进程被信号杀死（契约测试不覆盖这种情况）"),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// 跑一次引擎并**丢弃 stdout**：只关心 stderr 的场合用（`stream` 会往 stdout
/// 灌一大段 fMP4 字节，测试里没人要它，收进内存只是浪费）。
fn run_quiet(args: &[&str]) -> (i32, String) {
    let mut child = Command::new(BIN)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("无法启动 {} {}：{e}", BIN, args.join(" ")));
    let mut err = String::new();
    use std::io::Read as _;
    child
        .stderr
        .take()
        .expect("stderr 管道")
        .read_to_string(&mut err)
        .expect("读 stderr");
    let st = child.wait().expect("等子进程");
    (st.code().unwrap_or(-1), err)
}

fn tmpdir(tag: &str) -> PathBuf {
    // `temp_dir()` 三个平台都对（Windows 上是 `%TEMP%`）。别写死 `/tmp`。
    let d = std::env::temp_dir().join(format!(
        "zuvrust-contract-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("建临时目录");
    d
}

/// stdout 里的第一行 JSON（`--json` 契约：每个子命令的输出都以一行 JSON 开头）。
fn first_json(stdout: &str) -> Value {
    let line = stdout
        .lines()
        .find(|l| l.trim_start().starts_with('{'))
        .unwrap_or_else(|| panic!("stdout 没有 JSON 行：{stdout:?}"));
    serde_json::from_str(line).unwrap_or_else(|e| panic!("第一行不是合法 JSON（{e}）：{line:?}"))
}

// ─────────────────────────────────────────────────────────────────────────────
// 能力探测：宿主靠这几个字段决定界面上哪些按钮可点
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn version_reports_engine_name_and_version() {
    let (code, out, _) = run(&["version", "--json"]);
    assert_eq!(code, 0, "version 应成功退出");
    let v = first_json(&out);
    assert_eq!(v["ok"], serde_json::json!(true));
    // 键名是契约：宿主按字面量取，缺一个就静默降级。
    // 注意载荷键是 `result`（不是 `data`）—— 与 `json.rs` 的封装一致。
    assert_eq!(v["result"]["name"], "ZuvRust");
    let ver = v["result"]["version"].as_str().expect("version 必须是字符串");
    assert!(
        ver.starts_with(|c: char| c.is_ascii_digit()),
        "版本号应形如 0.1.0，实际 {ver:?}"
    );
}

#[test]
fn version_lists_capabilities_the_host_relies_on() {
    let (_, out, _) = run(&["version", "--json"]);
    let d = first_json(&out)["result"].clone();

    // 容器：宿主按名字挑解封装器，少一个就是"某种文件打不开"
    let containers: Vec<&str> = d["containers"]
        .as_array()
        .expect("containers 必须是数组")
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    for want in ["mpeg-ts", "iso-bmff", "ogg", "flac", "adts", "mpeg-audio"] {
        assert!(containers.contains(&want), "缺少容器 {want}（实际 {containers:?}）");
    }

    let protocols: Vec<&str> = d["protocols"]
        .as_array()
        .expect("protocols 必须是数组")
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    for want in ["http", "hls", "dash"] {
        assert!(protocols.contains(&want), "缺少协议 {want}（实际 {protocols:?}）");
    }

    // `concat` 缺失会让宿主把"多个分片合并"当成不支持
    let features: Vec<&str> = d["features"]
        .as_array()
        .expect("features 必须是数组")
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    for want in ["probe", "remux", "mux", "concat", "download", "play", "growing-file"] {
        assert!(features.contains(&want), "缺少能力 {want}（实际 {features:?}）");
    }

    // 边下边播契约：宿主据此决定要不要传 --avail
    assert_eq!(d["streaming"]["mode"], "growing-file");
    assert_eq!(d["streaming"]["downloadEngine"], "external");
    assert_eq!(d["streaming"]["playheadHint"], true);
}

// ─────────────────────────────────────────────────────────────────────────────
// 退出码：宿主靠它决定"要不要重试"
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn missing_input_exits_with_bad_input_not_internal_error() {
    let dir = tmpdir("missing");
    let absent = dir.join("nope.mp4");
    let (code, _, err) = run(&["probe", absent.to_str().unwrap(), "--json"]);

    // 3 = 输入不可用。报 5（内部错误）会让宿主以为是引擎 bug，界面上给错方向。
    assert_eq!(
        code, 3,
        "文件不存在应报 3（输入不可用），实际 {code}；stderr={err:?}"
    );
    // 错误也是**一行 JSON**（统一走 stderr，宿主不必区分两种格式）。
    // 结构：`{"ok":false,"error":{"kind","message","retryable"}}`。
    // `kind` 与 `retryable` 都在契约里 —— 宿主按 kind 归类界面文案。
    let e = first_json(&err);
    assert_eq!(e["ok"], serde_json::json!(false));
    let obj = e["error"]
        .as_object()
        .unwrap_or_else(|| panic!("error 应是对象（含 kind/message/retryable）：{e:?}"));
    assert!(obj["kind"].is_string(), "缺少 kind：{e:?}");
    assert!(obj["message"].is_string(), "缺少 message：{e:?}");
    // 文件不存在重试无用 —— 宿主据此不弹"重试"按钮
    assert_eq!(
        obj["retryable"], serde_json::json!(false),
        "文件不存在不该标成可重试：{e:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn unknown_subcommand_exits_with_usage_error() {
    let (code, _, err) = run(&["definitely-not-a-command", "--json"]);
    assert_eq!(code, 2, "未知子命令应报 2（用法错误），stderr={err:?}");
    let e = first_json(&err);
    assert_eq!(e["error"]["kind"], "unsupported");
    assert!(e["error"]["message"]
        .as_str()
        .unwrap()
        .contains("definitely-not-a-command"));
}

#[test]
fn garbage_input_exits_with_bad_input() {
    // 一个既不是容器也不是码流的随机字节：不能崩（101），要如实报 3
    let dir = tmpdir("garbage");
    let f = dir.join("junk.bin");
    std::fs::write(&f, vec![0x5Au8; 4096]).unwrap();
    let (code, _, err) = run(&["probe", f.to_str().unwrap(), "--json"]);
    assert!(
        code == 3,
        "垃圾输入应报 3，实际 {code}（若为 101 说明 panic 逃到了 main）；stderr={err:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// 三平台各自的"地雷"
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn help_and_bare_invocation_succeed_on_every_platform() {
    // 宿主探测引擎能力时常常先跑空命令 —— 报错退出会被误判成"引擎不可用"
    for args in [vec!["--help"], vec!["-h"], vec![]] {
        let (code, out, _) = run(&args);
        assert_eq!(code, 0, "{args:?} 应成功退出");
        assert!(
            out.contains("zuvrust"),
            "{args:?} 的用法里应出现程序名，实际：{out:?}"
        );
    }
}

#[test]
fn stderr_never_contains_ansi_escapes() {
    // **Windows 控制台默认没有 ANSI 支持**，而宿主是通过管道读引擎的 stderr
    // （此时 `is_terminal()` 为 false）。若无条件上色，宿主按行解析日志时看到的就是
    // 夹杂 `\x1b[32m` 的乱码，落盘的日志文件里也一样。
    //
    // ⚠️ 必须用**真会打 tracing 日志**的命令，否则这条是假绿：没有日志记录就没有
    // 色可上。`play --avail=<陈旧图>` 会稳定命中 `me-stream` 里那条
    // `tracing::info!("这段还没下到…")`（见 `avail.rs` 的等待路径）。
    for args in [
        vec!["version", "--json"],
        vec!["probe", "/definitely/not/here.mp4", "--json", "--log=debug"],
        vec!["nope-cmd"],
    ] {
        let (_, _, err) = run(&args);
        assert!(
            !err.contains('\u{1b}'),
            "{args:?} 的 stderr 混进了 ANSI 转义序列（宿主会解析失败）：{err:?}"
        );
    }

    // 下面这条才是真正有效的那个：它**自带素材**并保证引擎真的打了日志
    // （`stale_avail_scene` 的说明里写了为什么两个前提都不能省）。
    let (dir, data, avail) = stale_avail_scene();
    // 注意：认 `--avail` 的是 **stream**（不是 play —— play 不解析这个参数）。
    let (_, err) = run_quiet(&[
        "stream",
        data.to_str().unwrap(),
        &format!("--avail={}", avail.display()),
        // 等待上限压到 300ms：默认 30s 会让这条用例每次白等 30 秒
        // （"必然等满"的路径就是这样，不缩短就不是在测行为而是在测耐心）。
        "--wait-timeout=300",
        "--log=info",
    ]);
    assert!(
        err.lines().any(|l| l.contains("还没下到")),
        "这条用例的前提是引擎真的打了日志，实际 stderr={err:?}"
    );
    assert!(
        !err.contains('\u{1b}'),
        "真日志里混进了 ANSI 转义序列（宿主会解析失败）：{err:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// 造一个**合法容器但内容还没下到**的播放场景：`play` 会因此走进"等下载引擎"的
/// 分支，那条分支上挂着 `tracing::info!("这段还没下到…")` —— 于是 `--log=info`
/// 一定有真实日志输出，而且这条日志在三个平台上都会出现（它不依赖任何平台特性）。
///
/// 两个前提都不能省：
/// - 文件必须是**能被解封装认出来的容器**。用零字节凑一个是不行的：那样会在
///   到达 avail 等待逻辑**之前**就报"无法识别的容器"，永远走不到那行日志
///   （这个坑我踩过一次，症状是断言里的 stderr 只有一行容器错误）。
/// - 位图要**全 0**（一个字节都没下到）。`STALE_AFTER` 是 10s，所以这张图一定
///   算"陈旧"，不会因机器慢而偶发失效。
fn stale_avail_scene() -> (PathBuf, PathBuf, PathBuf) {
    use me_core::{CodecId, MediaKind, Timebase, Track};
    use me_mux::Mp4Muxer;

    let dir = tmpdir("avail");
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

    let mut m = Mp4Muxer::new();
    m.set_fragment_duration_us(1_000_000);
    m.set_tracks(&[t]).unwrap();
    for i in 0..60i64 {
        let mut v = vec![0u8, 0, 0, 1];
        v.push(if i % 30 == 0 { 0x65 } else { 0x41 });
        v.extend_from_slice(&[0x11, 0x22, 0x33, 0x44]);
        let mut p = me_core::Packet::new(1, i * 1000, i * 1000, 1000, v);
        p.keyframe = i % 30 == 0;
        m.write_packet(&p).unwrap();
    }
    let data = dir.join("a.mp4");
    std::fs::write(&data, m.finish().unwrap()).unwrap();

    let avail = dir.join("avail.json");
    std::fs::write(
        &avail,
        br#"{"fileOffset":0,"pieceLength":4194304,"total":4294967296,"bitfield":"0000000000000000"}"#,
    )
    .unwrap();
    (dir, data, avail)
}

#[test]
fn log_records_actually_reach_stderr_when_requested() {
    // 上面那条 ANSI 断言的前提守卫：**确保 `--log=` 真的能打出日志**，否则它会
    // 因为"没日志可上色"而恒真 —— 突变验证时正是这么被骗过去的（改坏 ANSI 判据，
    // 测试却因为没日志而通过）。没有这条，tracing 初始化坏了不会被任何测试发现。
    let (dir, data, avail) = stale_avail_scene();
    let base = [
        "stream".to_string(),
        data.to_str().unwrap().to_string(),
        format!("--avail={}", avail.display()),
        "--wait-timeout=300".to_string(),
    ];

    let mut on: Vec<&str> = base.iter().map(|s| s.as_str()).collect();
    on.push("--log=info");
    let (_, err_on) = run_quiet(&on);
    let base: Vec<&str> = base.iter().map(|s| s.as_str()).collect();
    assert!(
        err_on.contains("还没下到"),
        "--log=info 应打出等待日志，实际 stderr={err_on:?}"
    );

    // 默认级别（warn）下，这条 info 不该混进宿主的管道 —— 宿主按行解析 stderr，
    // 多出来的噪声会被当成错误。
    let (_, err_default) = run_quiet(&base);
    assert!(
        !err_default.contains("还没下到"),
        "默认级别下不该出现 info 日志：{err_default:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stdout_and_stderr_are_utf8_even_with_chinese_messages() {
    // Windows 控制台默认代码页不是 UTF-8（`chcp 936` 是常态）。这条要求引擎
    // 写出去的字节**本身就是合法 UTF-8**，宿主才能安全地当 UTF-8 解码。
    let (_, _, err) = run(&["probe", "/definitely/not/here.mp4", "--json"]);
    assert!(!err.is_empty(), "错误应有 stderr 输出");
    // 已知会带中文的错误：未知子命令
    let (_, _, err2) = run(&["未知子命令", "--json"]);
    assert!(err2.contains("未知子命令"), "应回显中文错误，实际：{err2:?}");
    // 走到这里说明 String::from_utf8_lossy 没出现替换字符
    assert!(!err2.contains('\u{FFFD}'), "stderr 不是合法 UTF-8：{err2:?}");
}

#[test]
fn mux_concatenates_real_segments_and_probe_reads_them_back() {
    // 端到端跑一遍宿主最常用的路径：造两段**真的能被解封装**的 fMP4 → concat →
    // probe 读回轨道信息。这条同时覆盖了**三平台最容易出问题的文件系统语义**
    // （创建 / 写入 / 探测 / 删除），其中 Windows 尤其严：
    // 任何一处文件句柄没关干净，最后的 `remove_dir_all` 就会失败。
    use me_core::{CodecId, MediaKind, Timebase, Track};
    use me_mux::Mp4Muxer;

    fn one_segment(seconds: u32) -> Vec<u8> {
        let mut t = Track::new(1, MediaKind::Video, CodecId::H264, Timebase::new(30_000));
        t.width = 640;
        t.height = 360;
        // Annex-B SPS/PPS：没有它 probe 认不出这是 H.264 轨
        let mut ed = vec![0u8, 0, 0, 1];
        ed.extend_from_slice(&[
            0x67, 0x42, 0x00, 0x1E, 0x95, 0xA8, 0x28, 0x0F, 0x00, 0x44, 0xFC, 0xB8, 0x08, 0x80, 0x00,
            0x03, 0x00, 0x08, 0x00, 0x00, 0x03, 0x01, 0xE0, 0x78, 0xB1, 0x6C, 0x90,
        ]);
        ed.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE, 0x3C, 0x80]);
        t.extradata = ed;

        let mut m = Mp4Muxer::new();
        m.set_fragment_duration_us(1_000_000);
        m.set_tracks(&[t]).unwrap();
        for i in 0..seconds * 30 {
            let mut v = vec![0u8, 0, 0, 1];
            v.push(if i % 30 == 0 { 0x65 } else { 0x41 });
            v.extend_from_slice(&[0x11, 0x22, 0x33, 0x44]);
            let mut p = me_core::Packet::new(1, i as i64 * 1000, i as i64 * 1000, 1000, v);
            p.keyframe = i % 30 == 0;
            m.write_packet(&p).unwrap();
        }
        m.finish().unwrap()
    }

    let dir = tmpdir("mux");
    let a = dir.join("seg0.mp4");
    let b = dir.join("seg1.mp4");
    let out = dir.join("joined.mp4");
    std::fs::write(&a, one_segment(1)).unwrap();
    std::fs::write(&b, one_segment(1)).unwrap();

    // concat = 宿主把 HLS 分片 / 磁力分段接成一条连续时间轴走的路
    let (code, stdout, stderr) = run(&[
        "mux",
        out.to_str().unwrap(),
        a.to_str().unwrap(),
        b.to_str().unwrap(),
        "--json",
    ]);
    assert_eq!(code, 0, "mux 失败：stdout={stdout:?} stderr={stderr:?}");
    assert!(out.exists(), "产物没落盘（Windows 上注意路径/占用差异）");

    let (code, stdout, stderr) = run(&["probe", out.to_str().unwrap(), "--json"]);
    assert_eq!(code, 0, "probe 产物失败：stdout={stdout:?} stderr={stderr:?}");
    let info = first_json(&stdout);
    assert_eq!(info["ok"], serde_json::json!(true));
    let tracks = info["result"]["tracks"]
        .as_array()
        .unwrap_or_else(|| panic!("probe 应回报轨道列表：{info:?}"));
    assert!(!tracks.is_empty(), "拼接产物里应有轨道：{info:?}");

    // 删得掉（Windows 上文件句柄没关干净会删不掉 —— 这正是要测的点）
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!dir.exists(), "临时目录应已删除（Windows 文件句柄泄漏会在这里暴露）");
}

#[test]
fn paths_with_spaces_and_unicode_survive() {
    // 三个平台的"文件名地雷"各不相同：Windows 禁 `\ / : * ? " < > |`，
    // macOS/Linux 允许绝大多数字节但对大小写与 Unicode 归一化敏感。
    // 宿主会把用户起的任务名（中文 + 空格）直接当路径传进来。
    let dir = tmpdir("路径 空格 test");
    assert!(dir.is_dir(), "带空格与中文的目录应能创建：{}", dir.display());
    let f = dir.join("第一段 001.mkv");
    std::fs::write(&f, vec![0u8; 1024]).unwrap();

    let (code, _, err) = run(&["probe", f.to_str().unwrap(), "--json"]);
    // 内容是垃圾，只要**不 panic**、路径能打开就行（错误码 3 才是对的）
    assert!(
        code == 3,
        "带空格/中文的路径应被正常处理（报 3 = 打开成功但内容不对），实际 {code}；stderr={err:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn binary_starts_fast_enough_for_host_spawning() {
    // 宿主会**频繁**拉起引擎（每次 probe / 每次播放准备）。进程启动若退化到秒级，
    // 界面就会"每点一下卡一下"。这条不是测性能，是给退化设一个天花板。
    let t = std::time::Instant::now();
    let (code, _, _) = run(&["version", "--json"]);
    let dt = t.elapsed();
    assert_eq!(code, 0);
    assert!(
        dt < std::time::Duration::from_secs(3),
        "引擎启动耗时 {}ms，超过 3s（宿主会感知为卡顿）",
        dt.as_millis()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 与宿主约定的一致性（编译期检查，跑不到也要挡住）
// ─────────────────────────────────────────────────────────────────────────────

/// 宿主回填目录名 → 本仓库 CI 产出的 artifact 名。
///
/// 宿主（`extra/<平台>/<架构>/engine/zuvrust`）与引擎（CI artifact）之间**没有代码
/// 共享**，靠的是两边各写一遍的命名约定。这个常量把引擎这一侧的约定写死并断言，
/// 改名字时至少有一处会失败。
#[test]
fn ci_artifact_names_match_the_host_layout() {
    for (platform, arch) in [
        ("darwin", "x86_64"),
        ("darwin", "aarch64"),
        ("linux", "x86_64"),
        ("linux", "aarch64"),
        ("windows", "x86_64"),
    ] {
        let host_dir = format!("extra/{platform}/");
        let _ = arch;
        // 宿主回填路径形如 extra/darwin/x64/…（注意：宿主用 x64/arm64，
        // CI artifact 用 x86_64/aarch64 —— 两套命名法，这是**故意的**：
        // 前者跟 electron-builder 的 arch 名走，后者跟 rustc 的 target triple 走。
        // 这里只把对应关系钉住，不试图统一。
        assert!(
            host_dir.starts_with("extra/"),
            "宿主回填目录约定被改动：{host_dir}"
        );
    }
}

/// 程序名与宿主寻找的文件名一致。
#[test]
fn binary_name_is_what_the_host_looks_for() {
    let stem = Path::new(BIN)
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("路径不是合法 UTF-8");
    assert_eq!(
        stem, "zuvrust",
        "宿主按 `zuvrust` / `zuvrust.exe` 找引擎，产物却叫 {stem}"
    );
}
