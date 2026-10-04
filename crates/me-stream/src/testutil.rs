//! 测试用的极小 HTTP 服务器。
//!
//! 为什么不用测试框架或 mock 库：本工程的硬约束是"不引入第三方依赖"，
//! 而"起一个真 socket、说真 HTTP"恰恰是最能暴露问题的方式 ——
//! ureq 的连接池、keep-alive、Range 语义都会被真实走一遍，
//! mock 掉这些就等于把最容易出错的地方测掉了。
//!
//! 刻意**不碰外网**：只绑 `127.0.0.1:0`（端口由内核分配），测试之间不会互相干扰。

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

/// 一条路由。
#[derive(Debug, Clone)]
pub struct Route {
    pub body: Vec<u8>,
    pub content_type: String,
    /// 是否支持 Range。真实 CDN 大多数支持，但**不支持 Range 回 200 全量**也是真实行为，
    /// 两种都要能造出来（上层对这两种的处置完全不同）。
    pub supports_range: bool,
}

impl Route {
    pub fn new(body: impl Into<Vec<u8>>) -> Self {
        Self {
            body: body.into(),
            content_type: "application/octet-stream".to_string(),
            supports_range: true,
        }
    }

    pub fn text(body: impl Into<String>) -> Self {
        Self {
            body: body.into().into_bytes(),
            content_type: "application/vnd.apple.mpegurl".to_string(),
            supports_range: true,
        }
    }

    pub fn no_range(mut self) -> Self {
        self.supports_range = false;
        self
    }
}

/// 测试服务器。`Drop` 时停止接受新连接。
pub struct TestServer {
    base: String,
    requests: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    accept: Option<JoinHandle<()>>,
}

impl TestServer {
    /// 起一个服务 `path → Route` 的服务器。`path` 必须带前导 `/`。
    pub fn serve(routes: Vec<(&str, Route)>) -> Self {
        let table: HashMap<String, Route> =
            routes.into_iter().map(|(p, r)| (p.to_string(), r)).collect();
        let table = Arc::new(table);
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));

        let listener = TcpListener::bind("127.0.0.1:0").expect("绑定本地端口");
        let addr = listener.local_addr().expect("取本地地址");
        listener.set_nonblocking(true).expect("设为非阻塞");

        let stop2 = stop.clone();
        let requests2 = requests.clone();
        let accept = std::thread::spawn(move || {
            while !stop2.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((sock, _)) => {
                        // 每条连接一个线程：ureq 的 Agent 会复用连接（keep-alive），
                        // 单线程串行处理会在"上一条连接还开着"时把下一条请求堵死。
                        let table = table.clone();
                        let requests = requests2.clone();
                        let stop = stop2.clone();
                        std::thread::spawn(move || handle_conn(sock, &table, &requests, &stop));
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });

        Self {
            base: format!("http://{addr}"),
            requests,
            stop,
            accept: Some(accept),
        }
    }

    pub fn url(&self, path: &str) -> String {
        format!("{}{}", self.base, path)
    }

    /// 收到过的请求行（`METHOD path`），按顺序。用来断言"该拉的拉了、不该拉的没拉"。
    pub fn requests(&self) -> Vec<String> {
        self.requests.lock().expect("请求记录锁").clone()
    }

    pub fn request_count(&self, needle: &str) -> usize {
        self.requests()
            .iter()
            .filter(|r| r.contains(needle))
            .count()
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.accept.take() {
            let _ = h.join();
        }
    }
}
fn handle_conn(
    mut sock: TcpStream,
    table: &HashMap<String, Route>,
    requests: &Mutex<Vec<String>>,
    stop: &AtomicBool,
) {
    // accept() 在 macOS/BSD 上会让新 socket **继承 listener 的非阻塞标志**，
    // 于是 `read` 会随机返回 WouldBlock —— 表现是"并发跑测试时偶发 Peer disconnected"。
    // 这里显式改回阻塞模式（读超时仍然保留，避免卡死）。
    let _ = sock.set_nonblocking(false);
    let _ = sock.set_read_timeout(Some(Duration::from_secs(10)));
    let _ = sock.set_nodelay(true);
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        // 读到请求头结束（\r\n\r\n）为止。我们只处理无 body 的 GET/HEAD。
        let head_end = loop {
            if let Some(p) = find_subslice(&buf, b"\r\n\r\n") {
                break p + 4;
            }
            match sock.read(&mut tmp) {
                Ok(0) => return,
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
                Err(_) => return,
            }
        };
        let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
        buf.drain(..head_end);

        let mut lines = head.split("\r\n");
        let request_line = lines.next().unwrap_or_default().to_string();
        let mut parts = request_line.split_whitespace();
        let method = parts.next().unwrap_or("GET").to_string();
        let target = parts.next().unwrap_or("/").to_string();
        let range = lines.find_map(|l| {
            let (k, v) = l.split_once(':')?;
            if k.eq_ignore_ascii_case("range") {
                Some(v.trim().to_string())
            } else {
                None
            }
        });
        requests
            .lock()
            .expect("请求记录锁")
            .push(format!("{method} {target}"));

        // 查询串不参与路由匹配（HLS 常把 token 放在查询串里）。
        let path = target.split(['?', '#']).next().unwrap_or("/").to_string();
        let Some(route) = table.get(&path) else {
            let body = b"not found";
            let _ = write_head(&mut sock, 404, "Not Found", body.len(), None, None);
            let _ = sock.write_all(body);
            return;
        };

        let total = route.body.len() as u64;
        let slice = match (&range, route.supports_range) {
            (Some(spec), true) => parse_range(spec, total),
            _ => None,
        };

        match slice {
            Some((start, end)) => {
                let part = &route.body[start as usize..=(end as usize)];
                let cr = format!("bytes {start}-{end}/{total}");
                let _ = write_head(
                    &mut sock,
                    206,
                    "Partial Content",
                    part.len(),
                    Some(&route.content_type),
                    Some(&cr),
                );
                if method != "HEAD" {
                    let _ = sock.write_all(part);
                }
            }
            None => {
                let _ = write_head(
                    &mut sock,
                    200,
                    "OK",
                    route.body.len(),
                    Some(&route.content_type),
                    None,
                );
                if method != "HEAD" {
                    let _ = sock.write_all(&route.body);
                }
            }
        }
        let _ = sock.flush();
        // **不能在这里 return**：响应头里写了 `Connection: keep-alive`，
        // 关掉连接会让客户端（ureq 的连接池）在下一次复用这条连接时报
        // "Peer disconnected"。继续循环、等下一个请求即可。
    }
}

fn write_head(
    sock: &mut TcpStream,
    code: u16,
    reason: &str,
    len: usize,
    ctype: Option<&str>,
    content_range: Option<&str>,
) -> std::io::Result<()> {
    let mut head = format!("HTTP/1.1 {code} {reason}\r\nContent-Length: {len}\r\n");
    head.push_str(&format!(
        "Content-Type: {}\r\n",
        ctype.unwrap_or("application/octet-stream")
    ));
    if let Some(cr) = content_range {
        head.push_str(&format!("Content-Range: {cr}\r\n"));
    }
    head.push_str("Accept-Ranges: bytes\r\nConnection: keep-alive\r\n\r\n");
    sock.write_all(head.as_bytes())
}

/// 解析 `bytes=start-end` / `bytes=start-`，返回**闭区间** `(start, end)`。
fn parse_range(spec: &str, total: u64) -> Option<(u64, u64)> {
    let spec = spec.trim().strip_prefix("bytes=")?;
    let (a, b) = spec.split_once('-')?;
    let start: u64 = a.trim().parse().ok()?;
    if start >= total {
        return None;
    }
    let end = match b.trim() {
        "" => total - 1,
        v => v.parse::<u64>().ok()?.min(total - 1),
    };
    Some((start, end))
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}
