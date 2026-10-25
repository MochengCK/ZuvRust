//! `serve --stdio`：给宿主用的常驻 JSON-RPC 2.0 接口。
//!
//! 设计取舍见 `API.md`；这里只记实现上的两条硬约定：
//!
//! 1. **任务仍然跑在子进程里**。服务进程只做协议、并发与生命周期管理 ——
//!    这样某个输入把容器解析打崩时，宿主收到的是一个 JSON-RPC 错误，而不是
//!    整条应用一起消失。代价是每次任务多一次 `zuvrust` 进程启动（毫秒级）。
//! 2. **stdout 只走分帧协议**。JSON-RPC 消息用 type=0，流式字节用 type=1；
//!    所有写入都经过同一把锁，多个并发任务不会把帧写花。
//!
//! 帧格式（小端）：`u32 payload_len` + `u8 type` + `payload`。
//! type=1 的 payload 再带 `u32 stream_id` + `u32 kind` + 原始字节。
//! 详细定义与错误码见 `API.md`。

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use me_core::{MediaError, Result};
use serde_json::{json, Value};

const PROTOCOL_VERSION: u32 = 1;
const TYPE_JSON: u8 = 0;
const TYPE_BINARY: u8 = 1;
const MAX_JSON_BYTES: usize = 8 << 20;
const MAX_BINARY_BYTES: usize = 64 << 20;

/// 二进制帧里的 `kind`：fMP4 字节流。
const BIN_FMP4: u32 = 0;
/// 二进制帧里的 `kind`：`stream --video=frames` 的 `MEV1` 包流。
const BIN_FRAMES: u32 = 1;

/// 引擎错误的 JSON-RPC 码（与 CLI 退出码一一对应，见 `API.md`）。
const E_USAGE: i64 = 1002;
const E_BAD_INPUT: i64 = 1003;
const E_TRANSIENT: i64 = 1004;
const E_INTERNAL: i64 = 1005;
const E_CANCELLED: i64 = 1006;

/// 服务入口。只支持 `--stdio`：不开放网络端口，避免鉴权/防火墙那套问题。
pub fn run(args: &[String]) -> Result<u8> {
    if !args.iter().any(|a| a == "--stdio") {
        return Err(MediaError::Unsupported(
            "serve 只支持 `--stdio`（JSON-RPC over stdin/stdout）".into(),
        ));
    }

    let writer = Arc::new(FrameWriter::new());
    let server = Arc::new(Server::new(writer));
    let mut stdin = std::io::stdin();

    loop {
        if server.shutting_down.load(Ordering::SeqCst) {
            break;
        }
        let frame = match read_frame(&mut stdin) {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => {
                tracing::warn!("serve 读 stdin 失败：{e}");
                break;
            }
        };
        if frame.0 != TYPE_JSON {
            let _ = server.writer.json(&error_response(
                &Value::Null,
                &RpcError::parse("stdin only accepts JSON frames (type 0)"),
            ));
            continue;
        }
        let msg: Value = match serde_json::from_slice(&frame.1) {
            Ok(v) => v,
            Err(e) => {
                let _ = server.writer.json(&error_response(
                    &Value::Null,
                    &RpcError::parse(format!("invalid JSON: {e}")),
                ));
                continue;
            }
        };
        if msg.is_array() {
            let _ = server.writer.json(&error_response(
                &Value::Null,
                &RpcError::invalid_request("batch requests are not supported"),
            ));
            continue;
        }
        server.handle_message(msg);
        if server.shutting_down.load(Ordering::SeqCst) {
            break;
        }
    }

    server.shutdown_all();
    Ok(0)
}

// ---------------------------------------------------------------------------
// 帧读写
// ---------------------------------------------------------------------------

struct FrameWriter {
    out: Mutex<std::io::Stdout>,
}

impl FrameWriter {
    fn new() -> Self {
        Self {
            out: Mutex::new(std::io::stdout()),
        }
    }

    fn write_frame(&self, ty: u8, payload: &[u8]) -> std::io::Result<()> {
        if payload.len() > u32::MAX as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "frame too large",
            ));
        }
        let mut out = self.out.lock().unwrap_or_else(|e| e.into_inner());
        out.write_all(&(payload.len() as u32).to_le_bytes())?;
        out.write_all(&[ty])?;
        out.write_all(payload)?;
        out.flush()
    }

    fn json(&self, value: &Value) -> std::io::Result<()> {
        let bytes = serde_json::to_vec(value)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        if bytes.len() > MAX_JSON_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "JSON frame too large",
            ));
        }
        self.write_frame(TYPE_JSON, &bytes)
    }

    fn binary(&self, stream_id: u32, kind: u32, data: &[u8]) -> std::io::Result<()> {
        let mut payload = Vec::with_capacity(8 + data.len());
        payload.extend_from_slice(&stream_id.to_le_bytes());
        payload.extend_from_slice(&kind.to_le_bytes());
        payload.extend_from_slice(data);
        self.write_frame(TYPE_BINARY, &payload)
    }
}

fn read_frame<R: Read>(reader: &mut R) -> std::io::Result<Option<(u8, Vec<u8>)>> {
    let mut first = [0u8; 1];
    if reader.read(&mut first)? == 0 {
        return Ok(None);
    }
    let mut head = [0u8; 5];
    head[0] = first[0];
    reader.read_exact(&mut head[1..])?;
    let len = u32::from_le_bytes([head[0], head[1], head[2], head[3]]) as usize;
    let ty = head[4];
    let max = if ty == TYPE_JSON {
        MAX_JSON_BYTES
    } else {
        MAX_BINARY_BYTES
    };
    if len > max {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("frame length {len} exceeds limit {max}"),
        ));
    }
    let mut payload = vec![0u8; len];
    reader.read_exact(&mut payload)?;
    Ok(Some((ty, payload)))
}

// ---------------------------------------------------------------------------
// 服务状态
// ---------------------------------------------------------------------------

struct Server {
    writer: Arc<FrameWriter>,
    jobs: Arc<Mutex<HashMap<String, Arc<Job>>>>,
    streams: Arc<Mutex<HashMap<u32, String>>>,
    next_stream: Arc<AtomicU32>,
    shutting_down: AtomicBool,
}

impl Server {
    fn new(writer: Arc<FrameWriter>) -> Self {
        Self {
            writer,
            jobs: Arc::new(Mutex::new(HashMap::new())),
            streams: Arc::new(Mutex::new(HashMap::new())),
            next_stream: Arc::new(AtomicU32::new(0)),
            shutting_down: AtomicBool::new(false),
        }
    }

    fn handle_message(&self, msg: Value) {
        let id = msg.get("id").cloned();
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
        if method.is_empty() {
            if let Some(id) = id {
                let _ = self.writer.json(&error_response(
                    &id,
                    &RpcError::invalid_request("missing method"),
                ));
            }
            return;
        }
        let params = msg.get("params").cloned().unwrap_or_else(|| json!({}));

        match method {
            "initialize" => {
                if let Some(id) = id {
                    let _ = self.writer.json(&json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": self.initialize_result(),
                    }));
                }
            }
            "version" => {
                if let Some(id) = id {
                    let _ = self.writer.json(&json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": crate::commands::version_value(),
                    }));
                }
            }
            "ping" => {
                if let Some(id) = id {
                    let _ = self
                        .writer
                        .json(&json!({"jsonrpc":"2.0","id":id,"result":{}}));
                }
            }
            "cancel" => {
                let found = self.cancel(&params);
                if let Some(id) = id {
                    let _ = self.writer.json(&json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {"cancelled": found},
                    }));
                }
            }
            "shutdown" => {
                if let Some(id) = id {
                    let _ = self
                        .writer
                        .json(&json!({"jsonrpc":"2.0","id":id,"result":{}}));
                }
                self.shutting_down.store(true, Ordering::SeqCst);
            }
            "probe" | "mux" | "remux" | "download" | "decode" | "stream" => {
                let Some(id) = id else {
                    // JSON-RPC 通知：不接受长任务，忽略。
                    return;
                };
                if let Err(e) = self.spawn_job(id.clone(), method.to_string(), params) {
                    let _ = self.writer.json(&error_response(&id, &e));
                }
            }
            other => {
                if let Some(id) = id {
                    let _ = self
                        .writer
                        .json(&error_response(&id, &RpcError::method_not_found(other)));
                }
            }
        }
    }

    fn initialize_result(&self) -> Value {
        json!({
            "protocolVersion": PROTOCOL_VERSION,
            "engine": crate::commands::version_value(),
            "framing": {
                "header": "u32le payload_len + u8 type",
                "typeJson": TYPE_JSON,
                "typeBinary": TYPE_BINARY,
                "maxJsonBytes": MAX_JSON_BYTES,
                "maxBinaryBytes": MAX_BINARY_BYTES,
                "binaryPayload": "u32le streamId + u32le kind + bytes",
            },
            "binaryKinds": {"fmp4": BIN_FMP4, "frames": BIN_FRAMES},
            "methods": [
                "initialize", "version", "ping", "probe", "mux", "remux",
                "download", "decode", "stream", "cancel", "shutdown"
            ],
            "notifications": ["progress", "stream/end"],
            "isolation": "one child process per job",
        })
    }

    fn cancel(&self, params: &Value) -> bool {
        let by_id = params.get("requestId").map(id_key);
        let by_stream = params
            .get("streamId")
            .and_then(Value::as_u64)
            .and_then(|id| {
                self.streams
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(&(id as u32))
                    .cloned()
            });
        let key = by_id.or(by_stream);
        let Some(key) = key else { return false };
        let job = self
            .jobs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            .cloned();
        if let Some(job) = job {
            job.cancel();
            true
        } else {
            false
        }
    }

    fn spawn_job(
        &self,
        id: Value,
        method: String,
        params: Value,
    ) -> std::result::Result<(), RpcError> {
        let key = id_key(&id);
        {
            let mut jobs = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
            if jobs.contains_key(&key) {
                return Err(RpcError::invalid_request("duplicate request id"));
            }
            jobs.insert(
                key.clone(),
                Arc::new(Job {
                    request_id: id.clone(),
                    method: method.clone(),
                    params: params.clone(),
                    child: Mutex::new(None),
                    cancelled: AtomicBool::new(false),
                    finished: AtomicBool::new(false),
                }),
            );
        }
        let job = self
            .jobs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            .cloned()
            .expect("刚插入");
        let writer = self.writer.clone();
        let jobs = self.jobs.clone();
        let streams = self.streams.clone();
        let next_stream = self.next_stream.clone();
        thread::spawn(move || {
            if let Err(e) = execute_job(&job, &writer, &streams, &next_stream, &key) {
                let _ = writer.json(&error_response(&job.request_id, &e));
            }
            if job.cancelled.load(Ordering::SeqCst) && !job.finished.load(Ordering::SeqCst) {
                job.cleanup_partial();
            }
            jobs.lock().unwrap_or_else(|e| e.into_inner()).remove(&key);
        });
        Ok(())
    }

    fn shutdown_all(&self) {
        self.shutting_down.store(true, Ordering::SeqCst);
        let jobs: Vec<_> = {
            let map = self.jobs.lock().unwrap_or_else(|e| e.into_inner());
            map.values().cloned().collect()
        };
        for job in &jobs {
            job.cancel();
        }
        // 给子进程一点时间走完取消清理（mux 会删半截产物）。
        for _ in 0..20 {
            if self
                .jobs
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty()
            {
                break;
            }
            thread::sleep(Duration::from_millis(100));
        }
    }
}

struct Job {
    request_id: Value,
    method: String,
    params: Value,
    child: Mutex<Option<Child>>,
    cancelled: AtomicBool,
    finished: AtomicBool,
}

impl Job {
    fn set_child(&self, child: Child) {
        *self.child.lock().unwrap_or_else(|e| e.into_inner()) = Some(child);
        if self.cancelled.load(Ordering::SeqCst) {
            self.terminate();
        }
    }

    fn take_child(&self) -> Option<Child> {
        self.child.lock().unwrap_or_else(|e| e.into_inner()).take()
    }

    fn terminate(&self) {
        if let Some(child) = self
            .child
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_mut()
        {
            terminate(child);
        }
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        self.terminate();
    }

    fn cleanup_partial(&self) {
        if self.method != "mux" {
            return;
        }
        if let Some(output) = self.params.get("output").and_then(Value::as_str) {
            let _ = std::fs::remove_file(output);
        }
    }
}

// ---------------------------------------------------------------------------
// 任务执行
// ---------------------------------------------------------------------------

fn execute_job(
    job: &Arc<Job>,
    writer: &Arc<FrameWriter>,
    streams: &Arc<Mutex<HashMap<u32, String>>>,
    next_stream: &Arc<AtomicU32>,
    request_key: &str,
) -> std::result::Result<(), RpcError> {
    if job.method == "stream" {
        run_stream(job, writer, streams, next_stream, request_key)
    } else {
        run_oneshot(job, writer)
    }
}

fn run_oneshot(job: &Arc<Job>, writer: &Arc<FrameWriter>) -> std::result::Result<(), RpcError> {
    let (args, env) = build_args(&job.method, &job.params)?;
    let mut child = spawn_engine(&args, &env)?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| RpcError::internal("child stdout unavailable"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| RpcError::internal("child stderr unavailable"))?;
    job.set_child(child);

    let stderr_writer = writer.clone();
    let request_id = job.request_id.clone();
    let stderr_thread =
        thread::spawn(move || read_stderr(stderr, stderr_writer, request_id, false));

    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    let mut result: Option<Value> = None;
    let mut engine_error: Option<Value> = None;
    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| RpcError::internal(format!("read child stdout: {e}")))?;
        if n == 0 {
            break;
        }
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(text) else {
            continue;
        };
        if value.get("ok").and_then(Value::as_bool) == Some(true) {
            if let (Some(ty), Some(data)) =
                (value.get("type").and_then(Value::as_str), value.get("data"))
            {
                let _ = writer.json(&progress_notification(&job.request_id, ty, data));
            } else if let Some(r) = value.get("result") {
                result = Some(r.clone());
            }
        } else if value.get("ok").and_then(Value::as_bool) == Some(false) {
            engine_error = value.get("error").cloned();
        }
    }

    let status = job
        .take_child()
        .map(|mut c| c.wait())
        .transpose()
        .map_err(|e| RpcError::internal(format!("wait child: {e}")))?
        .ok_or_else(|| RpcError::internal("child handle missing"))?;
    let stderr_text = stderr_thread.join().unwrap_or_default();

    if job.cancelled.load(Ordering::SeqCst) {
        return Err(RpcError::cancelled());
    }
    if let Some(result) = result {
        writer
            .json(&json!({"jsonrpc":"2.0","id":job.request_id,"result":result}))
            .map_err(|e| RpcError::internal(format!("write response: {e}")))?;
        job.finished.store(true, Ordering::SeqCst);
        return Ok(());
    }
    let engine_error = match engine_error {
        Some(error) => EngineError {
            error: Some(error),
            fallback: stderr_text,
            exit: exit_code(Some(status)),
        },
        None => engine_error_from_text(&stderr_text, exit_code(Some(status))),
    };
    Err(engine_error.into_rpc_error())
}

fn run_stream(
    job: &Arc<Job>,
    writer: &Arc<FrameWriter>,
    streams: &Arc<Mutex<HashMap<u32, String>>>,
    next_stream: &Arc<AtomicU32>,
    request_key: &str,
) -> std::result::Result<(), RpcError> {
    let (args, kind, env) = build_stream_args(&job.params)?;
    let mut child = spawn_engine(&args, &env)?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| RpcError::internal("child stdout unavailable"))?;
    let mut stderr = Some(
        child
            .stderr
            .take()
            .ok_or_else(|| RpcError::internal("child stderr unavailable"))?,
    );
    job.set_child(child);

    let mut reader = BufReader::new(stdout);
    let mut meta_line = String::new();
    let n = reader
        .read_line(&mut meta_line)
        .map_err(|e| RpcError::internal(format!("read stream meta: {e}")))?;
    if n == 0 {
        job.terminate();
        let status = job
            .take_child()
            .map(|mut c| c.wait())
            .transpose()
            .ok()
            .flatten();
        let stderr_text = stderr
            .take()
            .map(|s| read_stderr(s, writer.clone(), job.request_id.clone(), true))
            .unwrap_or_default();
        if job.cancelled.load(Ordering::SeqCst) {
            return Err(RpcError::cancelled());
        }
        return Err(engine_error_from_text(&stderr_text, exit_code(status)).into_rpc_error());
    }
    let meta_msg: Value = serde_json::from_str(meta_line.trim())
        .map_err(|e| RpcError::internal(format!("invalid stream meta JSON: {e}")))?;
    if meta_msg.get("ok").and_then(Value::as_bool) != Some(true)
        || meta_msg.get("type").and_then(Value::as_str) != Some("stream-meta")
    {
        job.terminate();
        let status = job
            .take_child()
            .map(|mut c| c.wait())
            .transpose()
            .ok()
            .flatten();
        let stderr_text = stderr
            .take()
            .map(|s| read_stderr(s, writer.clone(), job.request_id.clone(), true))
            .unwrap_or_default();
        return Err(engine_error_from_text(&stderr_text, exit_code(status)).into_rpc_error());
    }
    let meta = meta_msg.get("data").cloned().unwrap_or_else(|| json!({}));
    let stream_id = next_stream.fetch_add(1, Ordering::SeqCst).wrapping_add(1);
    streams
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(stream_id, request_key.to_string());

    if writer
        .json(&json!({
            "jsonrpc": "2.0",
            "id": job.request_id,
            "result": {"streamId": stream_id, "meta": meta},
        }))
        .is_err()
    {
        job.terminate();
        let _ = job.take_child().map(|mut c| c.wait());
        let _ = stderr
            .take()
            .map(|s| read_stderr(s, writer.clone(), job.request_id.clone(), true));
        streams
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&stream_id);
        return Ok(());
    }

    // 进度必须在 result 之后才开始转发：客户端按 requestId 把 progress 归到流上，
    // result 还没到就发 progress，客户端没有登记这条流，只能丢弃。
    let stderr_thread = stderr.take().map(|s| {
        let stderr_writer = writer.clone();
        let request_id = job.request_id.clone();
        thread::spawn(move || read_stderr(s, stderr_writer, request_id, true))
    });

    let mut buf = vec![0u8; 256 * 1024];
    let mut write_failed = false;
    loop {
        let n = match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => {
                tracing::warn!("serve 读流式 stdout 失败：{e}");
                break;
            }
        };
        if writer.binary(stream_id, kind, &buf[..n]).is_err() {
            write_failed = true;
            job.terminate();
            break;
        }
    }
    let status = job
        .take_child()
        .map(|mut c| c.wait())
        .transpose()
        .ok()
        .flatten();
    let stderr_text = stderr_thread
        .map(|h| h.join().unwrap_or_default())
        .unwrap_or_default();
    streams
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&stream_id);

    if write_failed {
        return Ok(());
    }
    if job.cancelled.load(Ordering::SeqCst) {
        let _ = writer.json(&stream_end_error(stream_id, &RpcError::cancelled()));
        return Ok(());
    }
    if exit_code(status) == Some(0) {
        let _ = writer.json(&json!({
            "jsonrpc": "2.0",
            "method": "stream/end",
            "params": {"streamId": stream_id, "ok": true},
        }));
    } else {
        let err = engine_error_from_text(&stderr_text, exit_code(status)).into_rpc_error();
        let _ = writer.json(&stream_end_error(stream_id, &err));
    }
    Ok(())
}

fn progress_notification(request_id: &Value, ty: &str, data: &Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "method": "progress",
        "params": {"requestId": request_id, "type": ty, "data": data},
    })
}

fn stream_end_error(stream_id: u32, err: &RpcError) -> Value {
    json!({
        "jsonrpc": "2.0",
        "method": "stream/end",
        "params": {
            "streamId": stream_id,
            "ok": false,
            "error": {"code": err.code, "message": err.message, "data": err.data},
        },
    })
}

fn read_stderr<R: Read>(
    stderr: R,
    writer: Arc<FrameWriter>,
    request_id: Value,
    forward_progress: bool,
) -> String {
    let mut reader = BufReader::new(stderr);
    let mut all = String::new();
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        all.push_str(&line);
        if !forward_progress {
            // 即使不转发进度，也把非 JSON 的日志行交给客户端 —— 否则适配层
            // 只能看到"任务失败"，看不到引擎自己说了什么。
            let text = line.trim();
            if !text.is_empty() && serde_json::from_str::<Value>(text).is_err() {
                let _ = writer.json(&log_notification(&request_id, text));
            }
            continue;
        }
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(text) {
            Ok(value) => {
                if value.get("ok").and_then(Value::as_bool) == Some(true) {
                    if let (Some(ty), Some(data)) =
                        (value.get("type").and_then(Value::as_str), value.get("data"))
                    {
                        let _ = writer.json(&progress_notification(&request_id, ty, data));
                    }
                }
            }
            Err(_) => {
                let _ = writer.json(&log_notification(&request_id, text));
            }
        }
    }
    all
}

fn log_notification(request_id: &Value, line: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "method": "log",
        "params": {"requestId": request_id, "line": line},
    })
}

// ---------------------------------------------------------------------------
// 子进程
// ---------------------------------------------------------------------------

fn spawn_engine(args: &[String], env: &[(String, String)]) -> std::result::Result<Child, RpcError> {
    let exe = std::env::current_exe()
        .map_err(|e| RpcError::internal(format!("locate engine binary: {e}")))?;
    let mut cmd = Command::new(exe);
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in env {
        cmd.env(key, value);
    }
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    cmd.spawn()
        .map_err(|e| RpcError::internal(format!("spawn engine: {e}")))
}

#[cfg(unix)]
fn terminate(child: &mut Child) {
    unsafe extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }
    let pid = child.id() as i32;
    // 先 SIGTERM：引擎会走 CancelFlag 的清理路径（mux 会删半截产物）。
    unsafe {
        let _ = kill(pid, 15);
    }
    let pid2 = pid;
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(1));
        unsafe {
            let _ = kill(pid2, 9);
        }
    });
}

#[cfg(not(unix))]
fn terminate(child: &mut Child) {
    let _ = child.kill();
}

fn exit_code(status: Option<std::process::ExitStatus>) -> Option<i32> {
    status.and_then(|s| s.code())
}

fn engine_code(exit: Option<i32>) -> i64 {
    match exit {
        Some(2) => E_USAGE,
        Some(3) => E_BAD_INPUT,
        Some(4) => E_TRANSIENT,
        Some(6) => E_CANCELLED,
        _ => E_INTERNAL,
    }
}

fn engine_error_from_text(stderr: &str, exit: Option<i32>) -> EngineError {
    let last = stderr.lines().rev().find_map(|line| {
        let value: Value = serde_json::from_str(line.trim()).ok()?;
        if value.get("ok").and_then(Value::as_bool) == Some(false) {
            value.get("error").cloned()
        } else {
            None
        }
    });
    let fallback = stderr
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.trim().to_string())
        .unwrap_or_else(|| "engine failed".to_string());
    EngineError {
        error: last,
        fallback,
        exit,
    }
}

struct EngineError {
    error: Option<Value>,
    fallback: String,
    exit: Option<i32>,
}

impl EngineError {
    fn into_rpc_error(self) -> RpcError {
        if let Some(error) = self.error {
            let kind = error
                .get("kind")
                .and_then(Value::as_str)
                .unwrap_or("internal");
            let retryable = error
                .get("retryable")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or(&self.fallback)
                .to_string();
            let code = if kind == "cancelled" {
                E_CANCELLED
            } else {
                engine_code(self.exit)
            };
            RpcError {
                code,
                message,
                data: Some(json!({
                    "kind": kind,
                    "retryable": retryable,
                    "exitCode": self.exit,
                })),
            }
        } else {
            RpcError {
                code: engine_code(self.exit),
                message: self.fallback,
                data: Some(json!({"kind": "internal", "retryable": false, "exitCode": self.exit})),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 参数 → CLI 参数
// ---------------------------------------------------------------------------

fn build_args(
    method: &str,
    params: &Value,
) -> std::result::Result<(Vec<String>, Vec<(String, String)>), RpcError> {
    let mut args = vec![method.to_string()];
    let mut env = Vec::new();
    match method {
        "probe" => {
            args.push(need_str(params, "input", "probe")?);
            if opt_bool(params, "grow") == Some(true) {
                args.push("--grow".into());
            }
            push_num(&mut args, "size", opt_u64(params, "size"));
            push_num(&mut args, "max-height", opt_u64(params, "maxHeight"));
            push_num(&mut args, "max-bandwidth", opt_u64(params, "maxBandwidth"));
        }
        "mux" => {
            args.push(need_str(params, "output", "mux")?);
            push_inputs(&mut args, params)?;
            push_num(&mut args, "fragment-ms", opt_u64(params, "fragmentMs"));
            push_tracks(&mut args, params)?;
            args.push("--progress".into());
        }
        "remux" => {
            args.push(need_str(params, "output", "remux")?);
            args.push(need_str(params, "input", "remux")?);
            push_num(&mut args, "fragment-ms", opt_u64(params, "fragmentMs"));
            // 片段裁剪（秒）：帧级切点，见 mux::ClipRange
            push_secs(&mut args, "clip-start", opt_f64(params, "clipStart"));
            push_secs(&mut args, "clip-end", opt_f64(params, "clipEnd"));
            push_tracks(&mut args, params)?;
            args.push("--progress".into());
        }
        "download" => {
            args.push(need_str(params, "input", "download")?);
            if let Some(output) = opt_str(params, "output") {
                args.push(format!("--out={output}"));
            }
            push_num(&mut args, "max-height", opt_u64(params, "maxHeight"));
            push_num(&mut args, "max-bandwidth", opt_u64(params, "maxBandwidth"));
            args.push("--progress".into());
        }
        "decode" => {
            args.push(need_str(params, "input", "decode")?);
            if opt_bool(params, "audio") == Some(true) {
                args.push("--audio".into());
            }
            if let Some(out) = opt_str(params, "out") {
                args.push(format!("--out={out}"));
            }
            push_num(&mut args, "count", opt_u64(params, "count"));
            if let Some(seek) = opt_f64(params, "seek") {
                args.push(format!("--seek={seek}"));
            }
            if opt_bool(params, "noGrow") == Some(true) {
                args.push("--no-grow".into());
            }
            env = decode_env(params)?;
        }
        "stream" => return build_stream_args(params).map(|(a, _, e)| (a, e)),
        other => return Err(RpcError::method_not_found(other)),
    }
    args.push("--json".into());
    Ok((args, env))
}

fn build_stream_args(
    params: &Value,
) -> std::result::Result<(Vec<String>, u32, Vec<(String, String)>), RpcError> {
    let mut args = vec!["stream".to_string(), need_str(params, "input", "stream")?];
    let mode = params.get("mode").and_then(Value::as_str).unwrap_or("fmp4");
    let kind = match mode {
        "fmp4" => BIN_FMP4,
        "frames" => {
            args.push("--video=frames".into());
            BIN_FRAMES
        }
        _ => return Err(RpcError::invalid_params("`mode` must be fmp4|frames")),
    };
    if opt_bool(params, "audioPcm") == Some(true) {
        args.push("--audio=pcm".into());
    }
    if let Some(seek) = opt_f64(params, "seek") {
        args.push(format!("--seek={seek}"));
    }
    push_num(&mut args, "fragment-ms", opt_u64(params, "fragmentMs"));
    push_num(&mut args, "size", opt_u64(params, "size"));
    push_num(&mut args, "max-height", opt_u64(params, "maxHeight"));
    push_num(&mut args, "max-bandwidth", opt_u64(params, "maxBandwidth"));
    push_num(&mut args, "wait-timeout", opt_u64(params, "waitTimeoutMs"));
    if let Some(avail) = opt_str(params, "avail") {
        args.push(format!("--avail={avail}"));
    }
    if opt_bool(params, "grow") == Some(false) {
        args.push("--no-grow".into());
    }
    args.push("--json".into());
    let env = decode_env(params)?;
    Ok((args, kind, env))
}

/// `decodeMode` / `threads` → 子进程环境变量。只影响解码任务。
fn decode_env(params: &Value) -> std::result::Result<Vec<(String, String)>, RpcError> {
    let mut env = Vec::new();
    if let Some(mode) = params.get("decodeMode").and_then(Value::as_str) {
        match mode {
            "auto" => {}
            "hardware" => env.push(("ME_DECODE".to_string(), "hardware".to_string())),
            "software" => env.push(("ME_DECODE".to_string(), "software".to_string())),
            _ => {
                return Err(RpcError::invalid_params(
                    "`decodeMode` must be auto|hardware|software",
                ))
            }
        }
    }
    if let Some(threads) = opt_u64(params, "threads") {
        if threads > 0 {
            env.push(("ME_THREADS".to_string(), threads.min(256).to_string()));
        }
    }
    Ok(env)
}

fn push_inputs(args: &mut Vec<String>, params: &Value) -> std::result::Result<(), RpcError> {
    let inputs = params
        .get("inputs")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
        .ok_or_else(|| RpcError::invalid_params("`inputs` must be a non-empty array"))?;
    for input in inputs {
        let s = input
            .as_str()
            .ok_or_else(|| RpcError::invalid_params("`inputs` must contain strings"))?;
        args.push(s.to_string());
    }
    Ok(())
}

fn push_tracks(args: &mut Vec<String>, params: &Value) -> std::result::Result<(), RpcError> {
    match params.get("tracks").and_then(Value::as_str) {
        None | Some("both") => {}
        Some("audio") => args.push("--audio-only".into()),
        Some("video") => args.push("--video-only".into()),
        Some(_) => {
            return Err(RpcError::invalid_params(
                "`tracks` must be both|audio|video",
            ))
        }
    }
    Ok(())
}

fn push_num(args: &mut Vec<String>, name: &str, value: Option<u64>) {
    if let Some(v) = value {
        args.push(format!("--{name}={v}"));
    }
}

/// 带小数的秒数（`clipStart` / `clipEnd` 这类）。NaN/Inf 直接忽略。
fn push_secs(args: &mut Vec<String>, name: &str, value: Option<f64>) {
    if let Some(v) = value.filter(|v| v.is_finite()) {
        args.push(format!("--{name}={v}"));
    }
}

fn need_str(params: &Value, key: &str, method: &str) -> std::result::Result<String, RpcError> {
    params
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| RpcError::invalid_params(format!("{method}: `{key}` must be a string")))
}

fn opt_str(params: &Value, key: &str) -> Option<String> {
    params.get(key).and_then(Value::as_str).map(str::to_string)
}

fn opt_bool(params: &Value, key: &str) -> Option<bool> {
    params.get(key).and_then(Value::as_bool)
}

fn opt_u64(params: &Value, key: &str) -> Option<u64> {
    params.get(key).and_then(Value::as_u64)
}

fn opt_f64(params: &Value, key: &str) -> Option<f64> {
    params.get(key).and_then(Value::as_f64)
}

fn id_key(id: &Value) -> String {
    serde_json::to_string(id).unwrap_or_else(|_| "null".to_string())
}

// ---------------------------------------------------------------------------
// JSON-RPC 错误
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct RpcError {
    code: i64,
    message: String,
    data: Option<Value>,
}

impl RpcError {
    fn parse(message: impl Into<String>) -> Self {
        Self {
            code: -32700,
            message: message.into(),
            data: None,
        }
    }

    fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            code: -32600,
            message: message.into(),
            data: None,
        }
    }

    fn invalid_params(message: impl Into<String>) -> Self {
        Self {
            code: -32602,
            message: message.into(),
            data: None,
        }
    }

    fn method_not_found(method: &str) -> Self {
        Self {
            code: -32601,
            message: format!("method not found: {method}"),
            data: None,
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            code: E_INTERNAL,
            message: message.into(),
            data: None,
        }
    }

    fn cancelled() -> Self {
        Self {
            code: E_CANCELLED,
            message: "cancelled".into(),
            data: Some(json!({"kind":"cancelled","retryable":false,"exitCode":6})),
        }
    }
}

fn error_response(id: &Value, err: &RpcError) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": err.code,
            "message": err.message,
            "data": err.data,
        }
    })
}
