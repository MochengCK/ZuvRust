//! HTTP 取流层（ureq 3）：清单 / 密钥的整段拉取，分片的 Range 拉取，以及
//! 把一条远端 URL 包装成引擎统一的 [`ReadAt`]。
//!
//! ## 为什么这一层要自己再包一层，而不让上层直接用 ureq
//!
//! 1. **请求头必须由引擎统一决定**。HLS 的很多源对 `Origin` 极敏感：
//!    浏览器会自动带 `Origin`，而引擎不该带 —— 带了会被判定成"跨域盗链"直接 403。
//!    所以这里**永不注入 `Origin`**，并且把调用方传进来的危险头（`Host` /
//!    `Content-Length` / `Range` / `Connection` / `Accept-Encoding`）全部丢掉：
//!    这些头要么由传输层自己管，要么会让 Range 语义错乱。
//! 2. **超时必须分阶段，不能一个大而全的 30 秒**。一条僵死的 TCP 连接会一直挂着，
//!    如果只有一个全局超时，它会把整条"按序写盘"的流水线一起顶住。
//!    这里：等响应头 15 秒、连上 10 秒、DNS 5 秒，响应体按**读空闲**计（见下）。
//! 3. **服务器无视 Range 回 200 全量** 是真实存在的行为（CDN 回源失败、防盗链网关等）。
//!    这时**绝不能把整段当成"你要的那一段"返回** —— 那会让分片字节整体错位，
//!    上层拼出来的 TS 流会莫名其妙地花屏。所以用 [`RangeResponse::partial`]
//!    如实表达"这不是你要的那一段"，由调用方自己决定怎么切片。
//! 4. **响应体用"读空闲超时"而不是"总预算"**。ureq 3 的 `timeout_recv_body`
//!    是**从收到响应头开始的总预算**，对一个大分片（几 MB）来说 10 秒总预算会在
//!    慢速移动网络上误杀；而"10 秒内一个字节都没来"才是真正的僵死。
//!    所以这里用独立线程读 socket + 带超时的通道，实现真正的空闲判定。

use std::io::Read;
use std::sync::Arc;
use std::time::Duration;

// `get_uri()`（重定向后的最终 URL）在 ureq 3 里是扩展 trait 的方法，必须显式引入。
use ureq::ResponseExt;

use me_core::error::{MediaError, Result};
use me_core::source::ReadAt;

/// 等响应头（TTFB）的上限。首字节慢是常态，给 15 秒。
const HEADER_TIMEOUT_SECS: u64 = 15;
/// 连接建立（含 TLS 握手）上限。
const CONNECT_TIMEOUT_SECS: u64 = 10;
/// DNS 解析上限。
const RESOLVE_TIMEOUT_SECS: u64 = 5;
/// 响应体**读空闲**上限：连续这么久读不到任何字节就判定为僵死连接。
pub const READ_IDLE_TIMEOUT_SECS: u64 = 10;

/// 调用方**不许**设置的头。理由分两类：
/// - `host` / `content-length` / `connection` / `transfer-encoding`：属于传输层，
///   让上层乱写会直接造出非法请求；
/// - `range`：Range 是引擎用来做分片取流的语义，必须由 [`HttpClient::get_range`] 独占控制；
/// - `accept-encoding`：一旦协商出 gzip，`Content-Range` 的偏移就对不上了；
/// - `upgrade` / `proxy-connection`：隧道类头，不该由业务层注入。
const FORBIDDEN_HEADERS: &[&str] = &[
    "host",
    "content-length",
    "range",
    "connection",
    "accept-encoding",
    "transfer-encoding",
    "upgrade",
    "proxy-connection",
];

/// HTTP 取流配置。
#[derive(Debug, Clone)]
pub struct HttpConfig {
    /// User-Agent。默认 `ZuvRust/0.1`。
    pub user_agent: String,
    /// Referer（不少 CDN 靠它做防盗链）。
    pub referer: Option<String>,
    /// 额外请求头（危险头会被丢弃，见 [`FORBIDDEN_HEADERS`]）。
    pub headers: Vec<(String, String)>,
    /// **整条请求的总预算**，只用于 [`HttpClient::get`]（清单/密钥，体积小、一次性）。
    /// 分片下载刻意不用它 —— 大分片被总预算掐死是线上最常见的误伤。
    pub timeout_secs: u64,
    /// 最大重定向次数。
    pub max_redirects: u32,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            user_agent: "ZuvRust/0.1".to_string(),
            referer: None,
            headers: Vec::new(),
            timeout_secs: 30,
            max_redirects: 10,
        }
    }
}

/// HEAD 响应里我们关心的部分。
#[derive(Debug, Clone, Default)]
pub struct HttpHead {
    pub content_length: Option<u64>,
    pub content_type: Option<String>,
    /// `Accept-Ranges` 的原文（`bytes` / `none`），调用方自己判断要不要走 Range。
    pub accept_ranges: Option<String>,
    /// 跟随重定向之后的最终 URL。
    pub final_url: String,
}

/// Range 请求的结果。
///
/// 关键设计：**用 `partial` 如实表达服务器到底认没认 Range**，而不是"看起来拿到了
/// 一段字节就当分片用"。服务器无视 Range 回整段时 `data` 是整个资源，
/// 调用方必须自己按偏移切（或者干脆改用整段缓存），否则拼出来的流会整体错位。
#[derive(Debug, Clone)]
pub struct RangeResponse {
    pub data: Vec<u8>,
    /// `true` = 服务器真的按 Range 回了（206，`data` 就是请求的那一段）；
    /// `false` = 服务器无视 Range 回了全量（200，`data` 是**整个资源**）。
    pub partial: bool,
    /// 响应体的 `Content-Type`（HLS 分片多为 `video/mp2t` / `video/mp4`）。
    pub content_type: Option<String>,
    /// `Content-Range: bytes 0-1023/4096` → `(0, Some(4096))`。
    /// 第一项是**本次响应的起始偏移**，第二项是**资源总长度**（`*` 时为 `None`）。
    /// 服务器回 200 时这个头通常不存在，于是它是 `None`。
    pub content_range: Option<(u64, Option<u64>)>,
    /// 跟随重定向之后的最终 URL。
    pub final_url: String,
}

/// HTTP 客户端。内部持有一个连接池（`ureq::Agent` 自带），可跨线程共享。
///
/// 内部用 `Arc<HttpInner>` 而不是裸字段：`get_source(&self, ..)` 要能把
/// "客户端句柄"塞进 [`HttpSource`] 里长期持有，而 `&self` 是调用方最自然的用法
/// （不必先把客户端装进 `Arc`）。`Clone` 是廉价的引用计数。
#[derive(Clone)]
pub struct HttpClient {
    inner: Arc<HttpInner>,
}

struct HttpInner {
    agent: ureq::Agent,
    /// 归一化后的自定义头：名字小写、同名后者覆盖前者、危险头与含 CR/LF 的已剔除。
    headers: Vec<(String, String)>,
    referer: Option<String>,
    timeout_secs: u64,
}

impl HttpClient {
    pub fn new(cfg: HttpConfig) -> Result<Self> {
        let headers = sanitize_headers(&cfg.headers);
        let timeout_secs = cfg.timeout_secs.max(1);

        let agent_cfg = ureq::Agent::config_builder()
            .user_agent(cfg.user_agent.as_str())
            .max_redirects(cfg.max_redirects)
            .timeout_resolve(Some(Duration::from_secs(RESOLVE_TIMEOUT_SECS)))
            .timeout_connect(Some(Duration::from_secs(CONNECT_TIMEOUT_SECS)))
            // 注意：这里**不设** timeout_global —— 全局超时会连大分片一起掐死。
            .timeout_recv_response(Some(Duration::from_secs(HEADER_TIMEOUT_SECS)))
            // 给 ureq 自己的兜底预算留一个宽松值；真正生效的是我们的读空闲判定。
            .timeout_recv_body(Some(Duration::from_secs(
                READ_IDLE_TIMEOUT_SECS * 30,
            )))
            .build();

        Ok(Self {
            inner: Arc::new(HttpInner {
                agent: ureq::Agent::new_with_config(agent_cfg),
                headers,
                referer: cfg.referer,
                timeout_secs,
            }),
        })
    }

    /// 整段 GET（清单、密钥、小文件）。带 `timeout_secs` 的总预算。
    pub fn get(&self, url: &str) -> Result<Vec<u8>> {
        let req = self.apply(self.inner.agent.get(url));
        let req = req
            .config()
            .timeout_global(Some(Duration::from_secs(self.inner.timeout_secs)))
            .build();
        let resp = req.call().map_err(map_ureq_err)?;
        let status = resp.status().as_u16();
        if status >= 400 {
            return Err(MediaError::protocol(format!("HTTP {status}（GET {url}）")));
        }
        let mut body = resp.into_body();
        body.read_to_vec()
            .map_err(|e| MediaError::protocol(format!("读取响应体失败: {e}")))
    }

    /// Range GET。
    ///
    /// `end` 为 `None` 时是"从 start 到结尾"的开区间请求。
    /// 服务器回 200（无视 Range）时不报错，而是把 [`RangeResponse::partial`] 置假，
    /// 让调用方自己切 —— 静默当分片返回是绝对不能做的。
    pub fn get_range(&self, url: &str, start: u64, end: Option<u64>) -> Result<RangeResponse> {
        let range_value = match end {
            Some(e) => format!("bytes={start}-{e}"),
            None => format!("bytes={start}-"),
        };
        let req = self
            .apply(self.inner.agent.get(url))
            .header("Range", range_value.as_str());
        let resp = req.call().map_err(map_ureq_err)?;
        let status = resp.status().as_u16();
        let h = resp.headers();
        let content_type = h
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim().to_string());
        let content_range = h
            .get("content-range")
            .and_then(|v| v.to_str().ok())
            .and_then(parse_content_range);
        let final_url = resp.get_uri().to_string();

        match status {
            206 => {
                let data = read_body_with_idle_timeout(resp.into_body(), READ_IDLE_TIMEOUT_SECS)?;
                Ok(RangeResponse {
                    data,
                    partial: true,
                    content_type,
                    content_range,
                    final_url,
                })
            }
            200 => {
                // 服务器无视了 Range。可能是它不支持，也可能是中间网关把请求"归一化"了。
                let data = read_body_with_idle_timeout(resp.into_body(), READ_IDLE_TIMEOUT_SECS)?;
                Ok(RangeResponse {
                    data,
                    partial: false,
                    content_type,
                    content_range,
                    final_url,
                })
            }
            s if s >= 400 => Err(MediaError::protocol(format!(
                "HTTP {s}（Range GET {url} {range_value}）"
            ))),
            s => Err(MediaError::protocol(format!(
                "Range 请求返回了意外状态码 {s}（{url}）"
            ))),
        }
    }

    /// HEAD：拿长度、类型、是否支持 Range。
    pub fn head(&self, url: &str) -> Result<HttpHead> {
        let req = self.apply(self.inner.agent.head(url));
        let resp = req.call().map_err(map_ureq_err)?;
        let status = resp.status().as_u16();
        if status >= 400 {
            return Err(MediaError::protocol(format!(
                "HTTP {status}（HEAD {url}）"
            )));
        }
        let h = resp.headers();
        let get = |name: &str| -> Option<String> {
            h.get(name)
                .and_then(|v| v.to_str().ok())
                .map(|s| s.trim().to_string())
        };
        Ok(HttpHead {
            content_length: get("content-length").and_then(|s| s.parse().ok()),
            content_type: get("content-type"),
            accept_ranges: get("accept-ranges"),
            final_url: resp.get_uri().to_string(),
        })
    }

    /// 把一条 URL 包装成随机访问源。长度用 HEAD 探测；探不到就是 `None`（长度未知）。
    ///
    /// 这里刻意**不做任何缓存**：缓存策略属于上层调度（谁该被留在内存里，
    /// 只有知道播放位置的调度器说了算），协议层擅自缓存只会和上层抢内存。
    pub fn get_source(&self, url: &str) -> Result<Arc<dyn ReadAt>> {
        let len = self.head(url).ok().and_then(|h| h.content_length);
        Ok(Arc::new(HttpSource {
            client: self.clone(),
            url: url.to_string(),
            len,
        }))
    }

    /// 给请求挂上 UA/Referer/自定义头。
    ///
    /// UA 走 agent 级配置（`AutoHeaderValue`），这里只管 Referer 与自定义头。
    fn apply(
        &self,
        req: ureq::RequestBuilder<ureq::typestate::WithoutBody>,
    ) -> ureq::RequestBuilder<ureq::typestate::WithoutBody> {
        let mut req = req;
        if let Some(r) = &self.inner.referer {
            if !contains_crlf(r) {
                req = req.header("Referer", r.as_str());
            }
        }
        for (k, v) in &self.inner.headers {
            req = req.header(k.as_str(), v.as_str());
        }
        req
    }
}

/// 一条远端 URL 的随机访问视图：每次 `read_at` 发一个 Range 请求。
///
/// 为什么不做块缓存：见 [`HttpClient::get_source`] 的说明。
pub struct HttpSource {
    client: HttpClient,
    url: String,
    len: Option<u64>,
}

impl ReadAt for HttpSource {
    fn len(&self) -> Option<u64> {
        self.len
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        if let Some(len) = self.len {
            if offset >= len {
                return Ok(0);
            }
        }
        let want_end = offset + buf.len() as u64 - 1;
        let end = match self.len {
            Some(len) => want_end.min(len - 1),
            None => want_end,
        };
        let resp = self.client.get_range(&self.url, offset, Some(end))?;

        if !resp.partial {
            // 服务器给了整段：从里面按偏移切出我们要的那一段。
            // 正确性优先 —— 切成空片或者错位才是真正的灾难。
            let start = offset as usize;
            if start >= resp.data.len() {
                return Ok(0);
            }
            let n = buf.len().min(resp.data.len() - start);
            buf[..n].copy_from_slice(&resp.data[start..start + n]);
            return Ok(n);
        }

        let n = buf.len().min(resp.data.len());
        buf[..n].copy_from_slice(&resp.data[..n]);
        Ok(n)
    }
}

// ---------------------------------------------------------------------------
// 内部工具
// ---------------------------------------------------------------------------

/// 归一化自定义头：
/// - 名字小写（HTTP 头名大小写不敏感，但我们要靠它去查重）；
/// - 丢弃 [`FORBIDDEN_HEADERS`]；
/// - 丢弃名字或值里含 CR/LF 的（防**请求头注入**：`"x-a\r\nHost: evil"` 这种值
///   如果被原样写进请求行，就能伪造出任意头）；
/// - 同名后者覆盖前者（调用方后写的意图更明确）。
fn sanitize_headers(raw: &[(String, String)]) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for (name, value) in raw {
        let name = name.trim().to_ascii_lowercase();
        if name.is_empty() || FORBIDDEN_HEADERS.contains(&name.as_str()) {
            continue;
        }
        if contains_crlf(&name) || contains_crlf(value) {
            continue;
        }
        let value = value.trim().to_string();
        match out.iter_mut().find(|(k, _)| *k == name) {
            Some(slot) => slot.1 = value,
            None => out.push((name, value)),
        }
    }
    out
}

fn contains_crlf(s: &str) -> bool {
    s.contains('\r') || s.contains('\n')
}

/// 解析 `Content-Range: bytes 0-1023/4096` → `(0, Some(4096))`；`/*` 时第二项为 `None`。
fn parse_content_range(s: &str) -> Option<(u64, Option<u64>)> {
    let s = s.trim();
    let rest = s.strip_prefix("bytes")?.trim_start();
    let (range, total) = rest.split_once('/')?;
    let (start, _end) = range.trim().split_once('-')?;
    let start: u64 = start.trim().parse().ok()?;
    let total = match total.trim() {
        "*" => None,
        t => Some(t.parse().ok()?),
    };
    Some((start, total))
}

/// 把响应体读完，用**读空闲超时**而不是总预算。
///
/// 实现方式：拿一个独立线程去阻塞读 socket，主线程用 `recv_timeout` 收块。
/// 只要 `idle` 内一块都没收到就判定僵死并返回错误。
///
/// 代价要说清楚：判定超时后，那个读线程还挂在 socket 上，要等服务器发数据或关连接
/// 才会退出。这是"能主动打断阻塞读"的必然代价（ureq 3 没有暴露 socket 级的
/// `SO_RCVTIMEO`），而且只发生在真正僵死的连接上，不会在正常路径上泄漏。
fn read_body_with_idle_timeout(body: ureq::Body, idle_secs: u64) -> Result<Vec<u8>> {
    read_to_end_with_idle_timeout(body.into_reader(), Duration::from_secs(idle_secs))
}

/// [`read_body_with_idle_timeout`] 的泛型内核，抽出来是为了能用假 reader 单测。
fn read_to_end_with_idle_timeout(
    mut reader: impl Read + Send + 'static,
    idle: Duration,
) -> Result<Vec<u8>> {
    let (tx, rx) = std::sync::mpsc::channel::<std::io::Result<Vec<u8>>>();
    std::thread::Builder::new()
        .name("me-net-body".to_string())
        .spawn(move || {
            let mut buf = vec![0u8; 64 * 1024];
            loop {
                match reader.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        if tx.send(Ok(buf[..n].to_vec())).is_err() {
                            // 主线程已经放弃（超时/错误），别再往里灌数据了
                            break;
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        break;
                    }
                }
            }
            // 正常结束：直接 drop 掉 sender，主线程会收到 Disconnected 并收尾
        })
        .map_err(|e| MediaError::protocol(format!("无法创建下载线程: {e}")))?;

    let mut out = Vec::new();
    loop {
        match rx.recv_timeout(idle) {
            Ok(Ok(chunk)) => out.extend_from_slice(&chunk),
            Ok(Err(e)) => {
                return Err(MediaError::protocol(format!("读取响应体失败: {e}")));
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                return Err(MediaError::protocol(format!(
                    "读取响应体空闲超时：{} 秒内没有收到任何数据（判定为僵死连接）",
                    idle.as_secs()
                )));
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    Ok(out)
}

fn map_ureq_err(e: ureq::Error) -> MediaError {
    // 保留 ureq 的原文：它会区分 "timeout: receive response" / "connect" 等阶段，
    // 排查线上问题时这几个字比一个笼统的 "网络错误" 值钱得多。
    MediaError::protocol(format!("HTTP 请求失败: {e}"))
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_engine_ua() {
        let c = HttpConfig::default();
        assert_eq!(c.user_agent, "ZuvRust/0.1");
        assert_eq!(c.max_redirects, 10);
        assert!(c.referer.is_none());
        assert!(c.headers.is_empty());
        // 构造不能失败（不联网，只建 agent）
        assert!(HttpClient::new(c).is_ok());
    }

    #[test]
    fn sanitize_drops_forbidden_headers() {
        let raw = vec![
            ("Host".into(), "evil.example".into()),
            ("Content-Length".into(), "999".into()),
            ("Range".into(), "bytes=0-1".into()),
            ("Connection".into(), "keep-alive".into()),
            ("Accept-Encoding".into(), "gzip".into()),
            ("X-Keep".into(), "yes".into()),
        ];
        let out = sanitize_headers(&raw);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], ("x-keep".to_string(), "yes".to_string()));
    }

    #[test]
    fn sanitize_rejects_header_injection() {
        // 值里塞 CRLF：如果不拦，就能在请求里凭空多出一个 Host 头
        let raw = vec![("X-A".into(), "v\r\nHost: evil".into())];
        assert!(sanitize_headers(&raw).is_empty());
        // 名字里塞 CRLF
        let raw = vec![("X-A\r\nHost".into(), "v".into())];
        assert!(sanitize_headers(&raw).is_empty());
    }

    #[test]
    fn sanitize_is_case_insensitive_and_last_wins() {
        let raw = vec![
            ("X-Token".into(), "first".into()),
            ("x-token".into(), "second".into()),
            ("X-OTHER".into(), "  padded  ".into()),
        ];
        let out = sanitize_headers(&raw);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], ("x-token".to_string(), "second".to_string()));
        // 值两端空白会被裁掉（HTTP 头值不允许首尾空白）
        assert_eq!(out[1], ("x-other".to_string(), "padded".to_string()));
    }

    #[test]
    fn no_origin_is_ever_injected() {
        // 配置里没有 Origin 时，sanitize 的结果里也不会凭空出现一个
        let out = sanitize_headers(&HttpConfig::default().headers);
        assert!(!out.iter().any(|(k, _)| k == "origin"));
        // 调用方显式给了就按调用方的来（我们只是"不注入"，不是"禁止"）
        let raw = vec![("Origin".into(), "https://a.example".into())];
        assert_eq!(sanitize_headers(&raw).len(), 1);
    }

    #[test]
    fn parse_content_range_forms() {
        assert_eq!(
            parse_content_range("bytes 0-1023/4096"),
            Some((0, Some(4096)))
        );
        assert_eq!(
            parse_content_range("bytes 2048-4095/4096"),
            Some((2048, Some(4096)))
        );
        // 总长未知
        assert_eq!(parse_content_range("bytes 100-199/*"), Some((100, None)));
        // 垃圾输入不能 panic，只返回 None
        assert_eq!(parse_content_range("items 0-1/2"), None);
        assert_eq!(parse_content_range("bytes abc"), None);
        assert_eq!(parse_content_range(""), None);
    }

    // ---- 读空闲超时 ----

    /// 立即返回数据的假 reader。
    struct InstantReader {
        chunks: Vec<Vec<u8>>,
    }

    impl Read for InstantReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.chunks.is_empty() {
                return Ok(0);
            }
            let c = self.chunks.remove(0);
            let n = c.len().min(buf.len());
            buf[..n].copy_from_slice(&c[..n]);
            Ok(n)
        }
    }

    #[test]
    fn idle_reader_collects_all_chunks() {
        let r = InstantReader {
            chunks: vec![b"abc".to_vec(), b"defg".to_vec()],
        };
        let got = read_to_end_with_idle_timeout(r, Duration::from_secs(5)).unwrap();
        assert_eq!(got, b"abcdefg");
    }

    /// 读之前先睡很久的假 reader，用来模拟"连接建上了但服务器再也不发数据"。
    struct StallingReader {
        stall: Duration,
    }

    impl Read for StallingReader {
        fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
            std::thread::sleep(self.stall);
            Ok(0)
        }
    }

    #[test]
    fn idle_timeout_fires_on_stalled_connection() {
        let r = StallingReader {
            stall: Duration::from_millis(600),
        };
        let err = read_to_end_with_idle_timeout(r, Duration::from_millis(50)).unwrap_err();
        assert!(
            err.to_string().contains("空闲超时"),
            "应当是空闲超时错误，实际: {err}"
        );
    }

    /// 慢但持续有数据：只要每块之间不超过 idle，就不该被判超时。
    /// 这正是"读空闲"和"总预算"的区别 —— 总预算会在这里误杀。
    struct DripReader {
        left: usize,
        gap: Duration,
    }

    impl Read for DripReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.left == 0 {
                return Ok(0);
            }
            std::thread::sleep(self.gap);
            let n = self.left.min(buf.len()).min(4);
            for b in buf[..n].iter_mut() {
                *b = b'x';
            }
            self.left -= n;
            Ok(n)
        }
    }

    #[test]
    fn slow_but_alive_transfer_survives_idle_timeout() {
        let r = DripReader {
            left: 40,
            gap: Duration::from_millis(10),
        };
        // 总耗时 100ms > idle(50ms)，但每块间隔 10ms < idle → 必须成功
        let got = read_to_end_with_idle_timeout(r, Duration::from_millis(50)).unwrap();
        assert_eq!(got.len(), 40);
    }

    /// 读线程报错要冒泡出来，不能静默当成"读完了"。
    struct FailingReader;

    impl Read for FailingReader {
        fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                "peer reset",
            ))
        }
    }

    #[test]
    fn reader_error_is_propagated() {
        let err = read_to_end_with_idle_timeout(FailingReader, Duration::from_secs(5)).unwrap_err();
        assert!(err.to_string().contains("peer reset"), "实际: {err}");
    }

    // ---- HttpSource 的偏移语义（用假 client 不现实，这里只验证长度探测失败时的行为） ----

    #[test]
    fn http_source_reports_unknown_len_when_head_fails() {
        // 用一个必然连不上的地址，验证 get_source 不 panic 且长度未知
        let client = Arc::new(
            HttpClient::new(HttpConfig {
                timeout_secs: 1,
                ..HttpConfig::default()
            })
            .unwrap(),
        );
        let src = client
            .get_source("http://127.0.0.1:1/never")
            .expect("HEAD 失败也不该让 get_source 失败，长度未知即可");
        assert_eq!(src.len(), None);
        // 空缓冲不触发任何网络请求
        assert_eq!(src.read_at(0, &mut []).unwrap(), 0);
    }
}
