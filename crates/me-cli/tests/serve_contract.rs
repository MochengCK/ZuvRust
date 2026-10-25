//! `serve --stdio` 的协议契约：真的起一个服务进程，按帧读写。
//!
//! 这条测试守的是 **API 的线上格式**（长度前缀、type、二进制流、通知），
//! 不是某个内部函数。SDK 与任何第三方客户端都按同一份格式说话，改坏了
//! 这里会先红。

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use me_core::{CodecId, MediaKind, Packet, Timebase, Track};
use me_mux::Mp4Muxer;
use serde_json::{json, Value};

const BIN: &str = env!("CARGO_BIN_EXE_zuvrust");
const TYPE_JSON: u8 = 0;
const TYPE_BINARY: u8 = 1;

struct Client {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    next_id: u64,
}

impl Client {
    fn start() -> Self {
        let mut child = Command::new(BIN)
            .args(["serve", "--stdio"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("启动 serve");
        let stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        Self {
            child,
            stdin,
            stdout,
            next_id: 1,
        }
    }

    fn send(&mut self, value: &Value) {
        let payload = serde_json::to_vec(value).unwrap();
        self.stdin
            .write_all(&(payload.len() as u32).to_le_bytes())
            .unwrap();
        self.stdin.write_all(&[TYPE_JSON]).unwrap();
        self.stdin.write_all(&payload).unwrap();
        self.stdin.flush().unwrap();
    }

    fn recv(&mut self) -> (u8, Vec<u8>) {
        let mut head = [0u8; 5];
        self.stdout.read_exact(&mut head).expect("读帧头");
        let len = u32::from_le_bytes([head[0], head[1], head[2], head[3]]) as usize;
        let mut payload = vec![0u8; len];
        self.stdout.read_exact(&mut payload).expect("读帧体");
        (head[4], payload)
    }

    /// 发一个请求，跳过所有通知，返回 id 匹配的响应。
    fn call(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}));
        loop {
            let (ty, payload) = self.recv();
            if ty != TYPE_JSON {
                continue;
            }
            let value: Value = serde_json::from_slice(&payload).unwrap();
            if value.get("id").and_then(Value::as_u64) == Some(id) {
                return value;
            }
        }
    }

    fn shutdown(mut self) {
        let response = self.call("shutdown", json!({}));
        assert!(response.get("error").is_none(), "shutdown 失败：{response}");
        let _ = self.child.wait();
    }
}

fn tmp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "zuvrust-serve-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

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

fn write_video(path: &Path) {
    let mut m = Mp4Muxer::new();
    m.set_fragment_duration_us(500_000);
    m.set_tracks(&[video_track()]).unwrap();
    for i in 0..60i64 {
        let mut v = vec![0u8, 0, 0, 1];
        v.push(if i % 30 == 0 { 0x65 } else { 0x41 });
        v.extend_from_slice(&[0x11, 0x22, 0x33, 0x44]);
        let mut p = Packet::new(1, i * 1000, i * 1000, 1000, v);
        p.keyframe = i % 30 == 0;
        m.write_packet(&p).unwrap();
    }
    std::fs::write(path, m.finish().unwrap()).unwrap();
}

fn write_audio(path: &Path) {
    let mut m = Mp4Muxer::new();
    m.set_fragment_duration_us(500_000);
    m.set_tracks(&[audio_track()]).unwrap();
    for i in 0..100i64 {
        let p = Packet::new(2, i * 1024, i * 1024, 1024, vec![0x21, 0x10, 0x05, 0x55]);
        m.write_packet(&p).unwrap();
    }
    std::fs::write(path, m.finish().unwrap()).unwrap();
}

fn result_of(response: &Value) -> &Value {
    response
        .get("result")
        .unwrap_or_else(|| panic!("响应里没有 result：{response}"))
}

fn start_stream(client: &mut Client, params: Value) -> (u64, Value) {
    let id = client.next_id;
    client.next_id += 1;
    client.send(&json!({"jsonrpc":"2.0","id":id,"method":"stream","params":params}));
    loop {
        let (ty, payload) = client.recv();
        if ty == TYPE_BINARY {
            continue;
        }
        let value: Value = serde_json::from_slice(&payload).unwrap();
        if value.get("id").and_then(Value::as_u64) == Some(id) {
            assert!(value.get("error").is_none(), "stream 启动失败：{value}");
            return (
                value["result"]["streamId"]
                    .as_u64()
                    .expect("stream 没返回 streamId"),
                value["result"]["meta"].clone(),
            );
        }
    }
}

fn drain_stream(client: &mut Client, stream_id: u64) -> (Vec<u8>, bool, Vec<String>) {
    let mut bytes = Vec::new();
    let mut progress = Vec::new();
    loop {
        let (ty, payload) = client.recv();
        if ty == TYPE_BINARY {
            assert!(payload.len() >= 8);
            let sid = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as u64;
            assert_eq!(sid, stream_id);
            bytes.extend_from_slice(&payload[8..]);
            continue;
        }
        let value: Value = serde_json::from_slice(&payload).unwrap();
        if value.get("method").and_then(Value::as_str) == Some("progress") {
            if let Some(ty) = value["params"]["type"].as_str() {
                progress.push(ty.to_string());
            }
            continue;
        }
        if value.get("method").and_then(Value::as_str) == Some("stream/end") {
            assert_eq!(value["params"]["streamId"], stream_id);
            return (
                bytes,
                value["params"]["ok"].as_bool().unwrap_or(false),
                progress,
            );
        }
    }
}

#[test]
fn serve_exposes_a_framed_json_rpc_api() {
    let dir = tmp_dir("api");
    let video = dir.join("video.mp4");
    let audio = dir.join("audio.mp4");
    let out = dir.join("out.mp4");
    write_video(&video);
    write_audio(&audio);

    let mut client = Client::start();

    let init = client.call("initialize", json!({}));
    let init = result_of(&init);
    assert_eq!(init["protocolVersion"], 1);
    assert_eq!(init["framing"]["typeJson"], 0);
    assert_eq!(init["framing"]["typeBinary"], 1);
    assert!(
        init["engine"]["modules"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m == "mux"),
        "测试构建必须带 mux：{init}"
    );

    let version = client.call("version", json!({}));
    let version = result_of(&version);
    assert_eq!(version["name"], "ZuvRust");
    assert!(version["features"]
        .as_array()
        .unwrap()
        .iter()
        .any(|f| f == "mux"));

    let pong = client.call("ping", json!({}));
    assert!(pong.get("error").is_none(), "ping 失败：{pong}");
    let cancel = client.call("cancel", json!({"requestId": 999999}));
    assert_eq!(result_of(&cancel)["cancelled"], false);

    let probe = client.call(
        "probe",
        json!({"input": video.to_string_lossy(), "grow": false}),
    );
    let probe = result_of(&probe);
    assert!(
        probe["tracks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["kindName"] == "video"),
        "probe 没看到视频轨：{probe}"
    );

    let missing = client.call("probe", json!({"input": "/definitely/not/here.mp4"}));
    let error = missing.get("error").expect("不存在的输入必须报错");
    assert_eq!(error["code"], 1003);
    assert_eq!(error["data"]["kind"], "io");

    let mux = client.call(
        "mux",
        json!({
            "output": out.to_string_lossy(),
            "inputs": [video.to_string_lossy(), audio.to_string_lossy()],
            "fragmentMs": 500
        }),
    );
    let mux = result_of(&mux);
    assert!(
        mux.get("output").is_some() || mux.get("bytes").is_some(),
        "mux 结果不完整：{mux}"
    );
    assert!(out.is_file() && std::fs::metadata(&out).unwrap().len() > 0);

    let probe_out = client.call("probe", json!({"input": out.to_string_lossy()}));
    let tracks = result_of(&probe_out)["tracks"].as_array().unwrap();
    assert!(
        tracks.iter().any(|t| t["kindName"] == "video"),
        "合并产物缺视频轨"
    );
    assert!(
        tracks.iter().any(|t| t["kindName"] == "audio"),
        "合并产物缺音频轨"
    );

    let remuxed = dir.join("remuxed.mkv");
    let remux = client.call(
        "remux",
        json!({
            "output": remuxed.to_string_lossy(),
            "input": out.to_string_lossy(),
            "fragmentMs": 500,
            "tracks": "both"
        }),
    );
    assert!(remux.get("error").is_none(), "remux 失败：{remux}");
    assert!(remuxed.is_file() && std::fs::metadata(&remuxed).unwrap().len() > 0);

    // 流式 fMP4：先拿 result（streamId + meta），之后是二进制帧，最后 stream/end。
    // decodeMode/threads 对 fMP4 不生效，但必须被接受并透传到子进程环境。
    let (stream_id, meta) = start_stream(
        &mut client,
        json!({
            "input": video.to_string_lossy(),
            "mode": "fmp4",
            "fragmentMs": 500,
            "decodeMode": "software",
            "threads": 2
        }),
    );
    assert!(
        meta["mime"].as_str().unwrap_or("").starts_with("video/mp4"),
        "流元信息不对：{meta}"
    );
    let (bytes, end_ok, progress) = drain_stream(&mut client, stream_id);
    assert!(end_ok, "stream/end 不是 ok");
    assert!(
        progress.iter().any(|p| p == "playhead"),
        "流式播放必须上报 playhead（客户端据此优先下载）：{progress:?}"
    );
    assert!(
        bytes.windows(4).any(|w| w == b"ftyp"),
        "流式 fMP4 里没有 ftyp（拿到 {} 字节）",
        bytes.len()
    );
    assert!(bytes.len() > 1000, "流式产物太小：{} 字节", bytes.len());

    // 帧流：mux-only 构建必须在启动阶段明确 1002。解码数据路径由
    // `av1_pipeline`/`decode` 的素材测试覆盖；这里不拿合成 H.264 假码流去
    // 逼硬件解码器（那会得到一个和协议无关的 OSStatus 错误）。
    let has_video = version["modules"]
        .as_array()
        .unwrap()
        .iter()
        .any(|m| m == "video");
    if !has_video {
        let response = client.call(
            "stream",
            json!({"input": video.to_string_lossy(), "mode":"frames"}),
        );
        let error = response.get("error").expect("mux-only 构建的帧流必须报错");
        assert_eq!(error["code"], 1002);
        assert_eq!(error["data"]["kind"], "unsupported");
    }

    // 非法的解码方式属于参数错误（不是引擎内部错误）。
    let bad_mode = client.call(
        "stream",
        json!({"input": video.to_string_lossy(), "decodeMode": "nope"}),
    );
    let error = bad_mode.get("error").expect("非法 decodeMode 必须报错");
    assert_eq!(error["code"], -32602);

    client.shutdown();
    let _ = std::fs::remove_dir_all(&dir);
}
