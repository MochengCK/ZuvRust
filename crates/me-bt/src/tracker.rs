//! tracker 交互（**只做协议与调度，不做网络传输**）。
//!
//! 传输由上层注入：HTTP 走 [`HttpGet`] trait，UDP 只提供"打包 / 解包"的纯函数。
//! 这么切的原因很实际 —— 本层一旦自己起 `TcpStream`，就没法在单元测试里
//! 断言"query 里 info_hash 的编码方式"，而**这恰恰是 tracker 协议最容易错的地方**：
//!
//! > HTTP announce 的 `info_hash` / `peer_id` 是 **20 个原始字节**做百分号编码，
//! > 不是它们的十六进制字符串。把 `[0x00, 0x20, 0x25, …]` 编成
//! > `"002025…"`（hex）而不是 `"%00%20%25…"`，tracker 会算出另一个 info_hash，
//! > 于是"tracker 返回 0 个 peer，但也不报错"—— 一种几乎无法从日志里看出来的失败。
//!
//! 另外两个刻意的决定：
//!
//! - **compact（BEP 23）优先且是默认**：`peers` 是 6 字节/条（IPv4 大端 + 端口大端），
//!   `peers6` 是 18 字节/条（IPv6 大端 + 端口大端）。**IPv6 必须按 16 字节还原**，
//!   任何"把 peers6 也当成 4 字节 IP"的写法都会得到一堆 `0.0.0.0`。
//! - **失败响应是错误**：tracker 用 `failure reason` 表示拒绝（种子被封禁、
//!   passkey 失效……）。把它当成"没有 peer 的成功响应"会让上层一直重试，
//!   而真正的原因（那个字符串）被丢掉。

use std::net::{IpAddr, Ipv6Addr, SocketAddr};

use me_core::error::{MediaError, Result};

use crate::bencode::BValue;

/// HTTP tracker 的传输抽象。**刻意不在本层实现**（见模块开头）。
///
/// `&dyn` 而不是泛型：调用方在运行时才决定用哪个 HTTP 客户端，
/// 而且 trait object 让测试里塞一个 mock 的成本降到最低。
pub trait HttpGet: Send + Sync {
    /// GET 一个完整 URL（已含 query），返回响应体。
    fn get(&self, url: &str) -> Result<Vec<u8>>;
}

/// announce 事件。`None` 是"周期性汇报"，其余是状态变化时的一次性事件。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    None,
    Started,
    Completed,
    Stopped,
}

impl Event {
    /// HTTP query 里的取值。`None` 表示**不带 `event` 参数**（而不是带一个空值）。
    pub fn as_http_str(&self) -> Option<&'static str> {
        match self {
            Event::None => None,
            Event::Started => Some("started"),
            Event::Completed => Some("completed"),
            Event::Stopped => Some("stopped"),
        }
    }

    /// BEP 15 的数字编码。注意**与消息顺序无关**：0 none / 1 completed / 2 started / 3 stopped。
    pub fn udp_code(&self) -> u32 {
        match self {
            Event::None => 0,
            Event::Completed => 1,
            Event::Started => 2,
            Event::Stopped => 3,
        }
    }
}

/// 一次 announce 请求。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnounceRequest {
    pub info_hash: [u8; 20],
    pub peer_id: [u8; 20],
    /// 我们监听（或对外宣称）的端口。
    pub port: u16,
    pub uploaded: u64,
    pub downloaded: u64,
    /// 还差多少字节。**0 表示"我下完了"**，tracker 据此把我们从 leecher 挪到 seeder。
    pub left: u64,
    pub event: Event,
    /// 想要多少个 peer。-1 表示用 tracker 的默认值（此时 query 里不带 `numwant`）。
    pub num_want: i32,
    /// 会话标识，用来让 tracker 认出"同一台机器的重复 announce"。
    pub key: Option<u32>,
    pub compact: bool,
}

impl AnnounceRequest {
    /// 拼 query 串（不含 `?`）。
    ///
    /// `info_hash` / `peer_id` 用**原始 20 字节**做百分号编码 —— 见模块开头。
    pub fn to_query(&self) -> String {
        let mut parts: Vec<String> = Vec::with_capacity(10);
        parts.push(format!("info_hash={}", percent_encode(&self.info_hash)));
        parts.push(format!("peer_id={}", percent_encode(&self.peer_id)));
        parts.push(format!("port={}", self.port));
        parts.push(format!("uploaded={}", self.uploaded));
        parts.push(format!("downloaded={}", self.downloaded));
        parts.push(format!("left={}", self.left));
        if self.compact {
            // 不显式要 compact 的话，很多 tracker 会回一个 list of dict，
            // 体积大一个量级（announce 是每次都要发的，这个开销不划算）。
            parts.push("compact=1".to_string());
        }
        if let Some(e) = self.event.as_http_str() {
            parts.push(format!("event={e}"));
        }
        if self.num_want >= 0 {
            parts.push(format!("numwant={}", self.num_want));
        }
        if let Some(k) = self.key {
            parts.push(format!("key={k}"));
        }
        parts.join("&")
    }
}

/// tracker 的 announce 响应。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnounceResponse {
    /// 常规汇报间隔（秒）。
    pub interval: u32,
    pub min_interval: Option<u32>,
    /// 已完成（seeder）数量。
    pub complete: u32,
    /// 未完成（leecher）数量。
    pub incomplete: u32,
    pub peers: Vec<SocketAddr>,
    pub peers6: Vec<SocketAddr>,
    pub warning: Option<String>,
    pub tracker_id: Option<String>,
}

/// 解析 HTTP tracker 的 announce 响应体（bencode）。
pub fn parse_announce_response(body: &[u8]) -> Result<AnnounceResponse> {
    let (v, _) = BValue::parse(body)?;
    if v.as_dict().is_none() {
        return Err(MediaError::protocol("tracker 响应不是 bencode 字典"));
    }
    // `failure reason` 是 tracker 明确拒绝：必须当成错误抛出去，
    // 否则上层会把"种子被封"当成"暂时没有 peer"而无限重试。
    if let Some(reason) = v.get_str("failure reason") {
        return Err(MediaError::protocol(format!("tracker 拒绝: {reason}")));
    }

    let peers = match v.get("peers") {
        Some(BValue::Bytes(b)) => parse_compact_v4(b),
        Some(other) => parse_peer_list(other, false),
        None => Vec::new(),
    };
    let peers6 = match v.get("peers6") {
        Some(BValue::Bytes(b)) => parse_compact_v6(b),
        Some(other) => parse_peer_list(other, true),
        None => Vec::new(),
    };

    Ok(AnnounceResponse {
        interval: v.get_int("interval").unwrap_or(0).max(0) as u32,
        min_interval: v.get_int("min interval").map(|x| x.max(0) as u32),
        complete: v.get_int("complete").unwrap_or(0).max(0) as u32,
        incomplete: v.get_int("incomplete").unwrap_or(0).max(0) as u32,
        peers,
        peers6,
        warning: v.get_str("warning message"),
        tracker_id: v.get_str("tracker id"),
    })
}

/// 拼 URL → 注入的 HTTP 客户端取回 → 解析。
///
/// 已经带 query 的 URL（少见但合法）用 `&` 接，不要拼出两个 `?`。
pub fn announce_http(
    http: &dyn HttpGet,
    url: &str,
    req: &AnnounceRequest,
) -> Result<AnnounceResponse> {
    let sep = if url.contains('?') { '&' } else { '?' };
    let full = format!("{url}{sep}{}", req.to_query());
    let body = http.get(&full)?;
    parse_announce_response(&body)
}

/// BEP 15 的 UDP tracker 打包/解包（纯函数，不含 socket）。
pub struct UdpAnnounce;

/// BEP 15 的协议魔法数。放在 connect 请求最前面，用来让 tracker 认出这是 BT 客户端。
const UDP_PROTOCOL_MAGIC: u64 = 0x0000_0417_2710_1980;
/// action 编码：0 connect / 1 announce / 2 scrape / 3 error。
const ACTION_CONNECT: u32 = 0;
const ACTION_ANNOUNCE: u32 = 1;
const ACTION_ERROR: u32 = 3;

impl UdpAnnounce {
    /// connect 请求：`magic(8) + action(4) + transaction_id(4)`，共 16 字节。
    pub fn build_connect_request(transaction_id: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(&UDP_PROTOCOL_MAGIC.to_be_bytes());
        out.extend_from_slice(&ACTION_CONNECT.to_be_bytes());
        out.extend_from_slice(&transaction_id.to_be_bytes());
        out
    }

    /// announce 请求，共 98 字节。字段顺序严格按 BEP 15：
    /// `connection_id / action / transaction_id / info_hash / peer_id /
    ///  downloaded / left / uploaded / event / ip / key / num_want / port`。
    ///
    /// 注意 `key`（4 字节）**不能漏**：漏了之后整个包会短 4 字节，
    /// 而 tracker 会照着它自己的字段偏移去读 `port`，得到的是 `key` 的高 2 字节 ——
    /// 表现是"tracker 回给我们一批端口全错的 peer"。
    pub fn build_announce_request(
        connection_id: u64,
        transaction_id: u32,
        req: &AnnounceRequest,
    ) -> Vec<u8> {
        let mut out = Vec::with_capacity(98);
        out.extend_from_slice(&connection_id.to_be_bytes());
        out.extend_from_slice(&ACTION_ANNOUNCE.to_be_bytes());
        out.extend_from_slice(&transaction_id.to_be_bytes());
        out.extend_from_slice(&req.info_hash);
        out.extend_from_slice(&req.peer_id);
        out.extend_from_slice(&req.downloaded.to_be_bytes());
        out.extend_from_slice(&req.left.to_be_bytes());
        out.extend_from_slice(&req.uploaded.to_be_bytes());
        out.extend_from_slice(&req.event.udp_code().to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes()); // ip = 0：让 tracker 用数据包的源地址
        out.extend_from_slice(&req.key.unwrap_or(0).to_be_bytes());
        out.extend_from_slice(&req.num_want.to_be_bytes());
        out.extend_from_slice(&req.port.to_be_bytes());
        debug_assert_eq!(out.len(), 98);
        out
    }

    /// 解析 connect 响应，返回 `(transaction_id, connection_id)`。
    pub fn parse_connect_response(b: &[u8]) -> Result<(u32, u64)> {
        if b.len() < 16 {
            return Err(MediaError::NeedMore {
                need: 16,
                have: b.len(),
            });
        }
        let action = be_u32(&b[0..4]);
        let transaction_id = be_u32(&b[4..8]);
        if action == ACTION_ERROR {
            return Err(udp_error(&b[8..]));
        }
        if action != ACTION_CONNECT {
            return Err(MediaError::protocol(format!(
                "UDP connect 响应的 action 应为 {ACTION_CONNECT}，实得 {action}"
            )));
        }
        let mut cid = [0u8; 8];
        cid.copy_from_slice(&b[8..16]);
        Ok((transaction_id, u64::from_be_bytes(cid)))
    }

    /// 解析 announce 响应，返回 `(transaction_id, 响应)`。
    pub fn parse_announce_response(b: &[u8]) -> Result<(u32, AnnounceResponse)> {
        if b.len() < 20 {
            return Err(MediaError::NeedMore {
                need: 20,
                have: b.len(),
            });
        }
        let action = be_u32(&b[0..4]);
        let transaction_id = be_u32(&b[4..8]);
        if action == ACTION_ERROR {
            return Err(udp_error(&b[8..]));
        }
        if action != ACTION_ANNOUNCE {
            return Err(MediaError::protocol(format!(
                "UDP announce 响应的 action 应为 {ACTION_ANNOUNCE}，实得 {action}"
            )));
        }
        let interval = be_u32(&b[8..12]);
        let leechers = be_u32(&b[12..16]);
        let seeders = be_u32(&b[16..20]);
        // 20 字节之后是 compact IPv4 peer 列表（6 字节/条）。UDP 协议没有 peers6，
        // 想拿 IPv6 peer 得走 HTTP 的 peers6 字段。
        Ok((
            transaction_id,
            AnnounceResponse {
                interval,
                min_interval: None,
                complete: seeders,
                incomplete: leechers,
                peers: parse_compact_v4(&b[20..]),
                peers6: Vec::new(),
                warning: None,
                tracker_id: None,
            },
        ))
    }
}

/// BEP 15 的错误响应：action=3，之后是一段 UTF-8 错误信息。
fn udp_error(body: &[u8]) -> MediaError {
    MediaError::protocol(format!(
        "UDP tracker 错误: {}",
        String::from_utf8_lossy(body).trim()
    ))
}

fn be_u32(p: &[u8]) -> u32 {
    u32::from_be_bytes([p[0], p[1], p[2], p[3]])
}

/// BEP 23 compact IPv4：每 6 字节 = 4 字节 IP（大端）+ 2 字节端口（大端）。
///
/// 末尾不足 6 字节的残片**直接忽略**：与其为一个残缺条目报错，
/// 不如把已经拿到的 peer 用上（这些 peer 全都要经过后续的握手校验）。
fn parse_compact_v4(b: &[u8]) -> Vec<SocketAddr> {
    b.chunks_exact(6)
        .map(|c| {
            let ip = [c[0], c[1], c[2], c[3]];
            let port = u16::from_be_bytes([c[4], c[5]]);
            SocketAddr::from((ip, port))
        })
        .collect()
}

/// BEP 23 compact IPv6：每 18 字节 = 16 字节 IP + 2 字节端口。
///
/// **必须按 16 字节还原成 `Ipv6Addr`**。这里最常见的 bug 是沿用 v4 的
/// 4 字节读法，结果每个 IPv6 peer 都被读成 `0.0.0.0`（外加一个错位的端口）。
fn parse_compact_v6(b: &[u8]) -> Vec<SocketAddr> {
    b.chunks_exact(18)
        .map(|c| {
            let mut ip = [0u8; 16];
            ip.copy_from_slice(&c[..16]);
            let port = u16::from_be_bytes([c[16], c[17]]);
            SocketAddr::new(IpAddr::V6(Ipv6Addr::from(ip)), port)
        })
        .collect()
}

/// 非 compact 形态：`peers` 是一个 list of dict（`ip` / `port`）。
///
/// compact 是默认，但真实世界里确实有 tracker 无视 `compact=1` 回这个形态，
/// 所以也认；`want_v6` 用来把 v4 / v6 两种条目分到不同字段里。
fn parse_peer_list(v: &BValue, want_v6: bool) -> Vec<SocketAddr> {
    let Some(list) = v.as_list() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for it in list {
        let (Some(ip_s), Some(port)) = (it.get_str("ip"), it.get_int("port")) else {
            continue;
        };
        let Ok(ip) = ip_s.parse::<IpAddr>() else {
            continue;
        };
        if ip.is_ipv6() != want_v6 || !(0..=65535).contains(&port) {
            continue;
        }
        out.push(SocketAddr::new(ip, port as u16));
    }
    out
}

/// 百分号编码，**按字节**处理（`info_hash` 是二进制，不是文本）。
///
/// 只保留 RFC 3986 的 unreserved 字符（`A-Za-z0-9-._~`），其余一律 `%XX`（大写十六进制）。
/// 这就是 tracker 期望的形态：`0x00` → `%00`、`0x20` → `%20`、`0x25` → `%25`。
fn percent_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(bytes.len() * 3);
    for &b in bytes {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0x0f) as usize] as char);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    fn req() -> AnnounceRequest {
        AnnounceRequest {
            info_hash: [0xAB; 20],
            peer_id: *b"-ME0001-abcdefghijkl",
            port: 6881,
            uploaded: 100,
            downloaded: 200,
            left: 300,
            event: Event::Started,
            num_want: 50,
            key: Some(0x1234_5678),
            compact: true,
        }
    }

    /// 关键用例：`info_hash` 必须是**原始 20 字节的百分号编码**，不是 hex。
    #[test]
    fn info_hash_is_raw_percent_encoded_not_hex() {
        // 故意包含 0x00 / 0x20 / 0x25（`%` 本身）以及非 ASCII 字节
        let info_hash = [
            0x00, 0x20, 0x25, 0xde, 0xad, 0xbe, 0xef, 0x7f, 0x80, 0xff, 0x0a, 0x0d, 0x2b, 0x2f,
            0x3a, 0x3f, 0x23, 0x26, 0x3d, 0x41,
        ];
        let r = AnnounceRequest {
            info_hash,
            ..req()
        };
        let q = r.to_query();

        // 必须出现这些转义（0x00→%00，0x20→%20，0x25→%25）
        assert!(q.contains("%00"), "0x00 必须编成 %00：{q}");
        assert!(q.contains("%20"), "0x20 必须编成 %20：{q}");
        assert!(q.contains("%25"), "0x25 必须编成 %25：{q}");
        assert!(
            q.contains("info_hash=%00%20%25%DE%AD%BE%EF"),
            "必须是原始字节的百分号编码：{q}"
        );
        // 反证：绝不是 info_hash 的十六进制字符串
        assert!(
            !q.contains("002025deadbeef"),
            "绝不能把 info_hash 编成 hex 字符串：{q}"
        );
        assert!(!q.contains("deadbeef"), "hex 形态不应出现：{q}");

        // 0x2b('+') / 0x2f / 0x3a / 0x3f / 0x23 / 0x26 / 0x3d 这些也不能被当成安全字符
        assert!(q.contains("%2B%2F%3A%3F%23%26%3D"), "保留字符必须转义：{q}");
        // 只有 0x41('A') 这种 unreserved 才原样出现。info_hash 是第一个参数、
        // 后面还接着 peer_id 等，所以用 starts_with 而不是 ends_with。
        assert!(
            q.starts_with(
                "info_hash=%00%20%25%DE%AD%BE%EF%7F%80%FF%0A%0D%2B%2F%3A%3F%23%26%3DA&"
            ),
            "可打印的 unreserved 字符原样保留：{q}"
        );
    }

    #[test]
    fn peer_id_is_also_percent_encoded_by_bytes() {
        let r = AnnounceRequest {
            peer_id: *b"-ME0001-\x00\x01\x02\xff\xfe\xfd\x7f\x20\x25\x2b\x00\x00",
            ..req()
        };
        let q = r.to_query();
        assert!(q.contains("peer_id=-ME0001-%00%01%02%FF%FE%FD%7F%20%25%2B%00%00"), "{q}");
        // 前缀是 unreserved（字母、数字、'-'），必须原样
        assert!(!q.contains("peer_id=%2D"), "连字符是 unreserved，不该被转义：{q}");
    }

    #[test]
    fn query_has_all_expected_params() {
        let q = req().to_query();
        assert!(q.contains("port=6881"));
        assert!(q.contains("uploaded=100"));
        assert!(q.contains("downloaded=200"));
        assert!(q.contains("left=300"));
        assert!(q.contains("compact=1"));
        assert!(q.contains("event=started"));
        assert!(q.contains("numwant=50"));
        assert!(q.contains("key=305419896"), "key 以十进制出现：{q}");

        // event=None 时不带 event 参数；num_want=-1 时不带 numwant；compact=false 时不带 compact
        let r = AnnounceRequest {
            event: Event::None,
            num_want: -1,
            compact: false,
            key: None,
            ..req()
        };
        let q = r.to_query();
        assert!(!q.contains("event="), "{q}");
        assert!(!q.contains("numwant="), "{q}");
        assert!(!q.contains("compact="), "{q}");
        assert!(!q.contains("key="), "{q}");
        // 其余字段仍在
        assert!(q.contains("info_hash=") && q.contains("port=6881"));

        // 事件取值
        assert_eq!(Event::Started.as_http_str(), Some("started"));
        assert_eq!(Event::Completed.as_http_str(), Some("completed"));
        assert_eq!(Event::Stopped.as_http_str(), Some("stopped"));
        assert_eq!(Event::None.as_http_str(), None);
    }

    fn compact_v4(peers: &[([u8; 4], u16)]) -> Vec<u8> {
        peers
            .iter()
            .flat_map(|(ip, port)| {
                let mut v = ip.to_vec();
                v.extend_from_slice(&port.to_be_bytes());
                v
            })
            .collect()
    }

    fn compact_v6(peers: &[([u8; 16], u16)]) -> Vec<u8> {
        peers
            .iter()
            .flat_map(|(ip, port)| {
                let mut v = ip.to_vec();
                v.extend_from_slice(&port.to_be_bytes());
                v
            })
            .collect()
    }

    #[test]
    fn parse_http_response_compact_peers() {
        let v4 = compact_v4(&[([192, 168, 1, 10], 6881), ([10, 0, 0, 1], 80)]);
        let v6 = compact_v6(&[(
            [0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            6881,
        )]);
        let mut d = BTreeMap::new();
        d.insert(b"interval".to_vec(), BValue::Int(1800));
        d.insert(b"min interval".to_vec(), BValue::Int(60));
        d.insert(b"complete".to_vec(), BValue::Int(5));
        d.insert(b"incomplete".to_vec(), BValue::Int(7));
        d.insert(b"peers".to_vec(), BValue::Bytes(v4));
        d.insert(b"peers6".to_vec(), BValue::Bytes(v6));
        d.insert(b"warning message".to_vec(), BValue::Bytes(b"slow".to_vec()));
        d.insert(b"tracker id".to_vec(), BValue::Bytes(b"tid-1".to_vec()));

        let r = parse_announce_response(&BValue::Dict(d).encode()).unwrap();
        assert_eq!(r.interval, 1800);
        assert_eq!(r.min_interval, Some(60));
        assert_eq!(r.complete, 5);
        assert_eq!(r.incomplete, 7);
        assert_eq!(
            r.peers,
            vec![
                "192.168.1.10:6881".parse().unwrap(),
                "10.0.0.1:80".parse().unwrap()
            ]
        );
        // IPv6 必须还原成真正的 v6 地址，绝不能是 0.0.0.0
        assert_eq!(r.peers6.len(), 1);
        assert!(r.peers6[0].is_ipv6());
        assert_eq!(r.peers6[0], "[2001:db8::1]:6881".parse().unwrap());
        assert_ne!(r.peers6[0].ip().to_string(), "0.0.0.0");
        assert_eq!(r.warning.as_deref(), Some("slow"));
        assert_eq!(r.tracker_id.as_deref(), Some("tid-1"));
    }

    #[test]
    fn parse_http_response_tolerates_missing_and_non_compact() {
        // 最小响应：只有 interval
        let mut d = BTreeMap::new();
        d.insert(b"interval".to_vec(), BValue::Int(60));
        let r = parse_announce_response(&BValue::Dict(d).encode()).unwrap();
        assert_eq!(r.interval, 60);
        assert!(r.peers.is_empty() && r.peers6.is_empty());
        assert_eq!(r.complete, 0);

        // 非 compact：list of dict
        let mut d = BTreeMap::new();
        d.insert(b"interval".to_vec(), BValue::Int(60));
        d.insert(
            b"peers".to_vec(),
            BValue::List(vec![
                BValue::Dict(BTreeMap::from([
                    (b"ip".to_vec(), BValue::Bytes(b"1.2.3.4".to_vec())),
                    (b"port".to_vec(), BValue::Int(1234)),
                    (b"peer id".to_vec(), BValue::Bytes(b"-XX0001-abcdefghijkl".to_vec())),
                ])),
                // 坏条目：端口非法 / ip 非法 —— 应被跳过而不是让整条响应失败
                BValue::Dict(BTreeMap::from([
                    (b"ip".to_vec(), BValue::Bytes(b"1.2.3.4".to_vec())),
                    (b"port".to_vec(), BValue::Int(70000)),
                ])),
                BValue::Dict(BTreeMap::from([
                    (b"ip".to_vec(), BValue::Bytes(b"not-an-ip".to_vec())),
                    (b"port".to_vec(), BValue::Int(1)),
                ])),
            ]),
        );
        let r = parse_announce_response(&BValue::Dict(d).encode()).unwrap();
        assert_eq!(r.peers, vec!["1.2.3.4:1234".parse().unwrap()]);
        assert!(r.peers6.is_empty());
    }

    #[test]
    fn failure_reason_is_an_error() {
        let mut d = BTreeMap::new();
        d.insert(b"failure reason".to_vec(), BValue::Bytes(b"torrent not registered".to_vec()));
        let e = parse_announce_response(&BValue::Dict(d).encode()).unwrap_err();
        match e {
            MediaError::Protocol(msg) => assert!(msg.contains("not registered"), "{msg}"),
            other => panic!("应当是 Protocol 错误，实得 {other:?}"),
        }
        // 不是字典
        assert!(parse_announce_response(b"i1e").is_err());
    }

    /// 记录最后一次请求 URL 的 mock HTTP 客户端。
    struct MockHttp {
        body: Vec<u8>,
        last_url: Mutex<Option<String>>,
    }

    impl HttpGet for MockHttp {
        fn get(&self, url: &str) -> Result<Vec<u8>> {
            *self.last_url.lock().unwrap() = Some(url.to_string());
            Ok(self.body.clone())
        }
    }

    #[test]
    fn announce_http_builds_url_and_parses() {
        let body = BValue::Dict(BTreeMap::from([
            (b"interval".to_vec(), BValue::Int(900)),
            (
                b"peers".to_vec(),
                BValue::Bytes(compact_v4(&[([127, 0, 0, 1], 51413)])),
            ),
        ]))
        .encode();
        let http = MockHttp {
            body,
            last_url: Mutex::new(None),
        };
        let r = announce_http(&http, "http://tracker.example/announce", &req()).unwrap();
        assert_eq!(r.interval, 900);
        assert_eq!(r.peers, vec!["127.0.0.1:51413".parse().unwrap()]);
        let url = http.last_url.lock().unwrap().clone().unwrap();
        assert!(url.starts_with("http://tracker.example/announce?"), "{url}");
        assert!(url.contains("info_hash=%AB%AB"), "{url}");
        assert!(url.contains("event=started"));

        // 已带 query 的 URL 用 & 接
        let http = MockHttp {
            body: BValue::Dict(BTreeMap::from([(b"interval".to_vec(), BValue::Int(1))])).encode(),
            last_url: Mutex::new(None),
        };
        announce_http(&http, "http://t/a?passkey=x", &req()).unwrap();
        let url = http.last_url.lock().unwrap().clone().unwrap();
        assert_eq!(url.matches('?').count(), 1, "不能拼出两个问号：{url}");
        assert!(url.contains("?passkey=x&info_hash="), "{url}");
    }

    #[test]
    fn udp_connect_request_layout() {
        let b = UdpAnnounce::build_connect_request(0xDEAD_BEEF);
        assert_eq!(b.len(), 16);
        assert_eq!(&b[0..8], &UDP_PROTOCOL_MAGIC.to_be_bytes());
        assert_eq!(UDP_PROTOCOL_MAGIC, 0x41727101980);
        assert_eq!(&b[8..12], &0u32.to_be_bytes(), "action = connect(0)");
        assert_eq!(&b[12..16], &0xDEAD_BEEFu32.to_be_bytes());
    }

    #[test]
    fn udp_announce_request_layout_is_98_bytes() {
        let r = req();
        let b = UdpAnnounce::build_announce_request(0x0102_0304_0506_0708, 0xCAFE_BABE, &r);
        assert_eq!(b.len(), 98, "BEP 15 的 announce 请求正好 98 字节");
        assert_eq!(&b[0..8], &0x0102030405060708u64.to_be_bytes());
        assert_eq!(&b[8..12], &1u32.to_be_bytes(), "action = announce(1)");
        assert_eq!(&b[12..16], &0xCAFEBABEu32.to_be_bytes());
        assert_eq!(&b[16..36], &r.info_hash, "info_hash 紧跟 transaction_id");
        assert_eq!(&b[36..56], &r.peer_id);
        assert_eq!(&b[56..64], &r.downloaded.to_be_bytes());
        assert_eq!(&b[64..72], &r.left.to_be_bytes());
        assert_eq!(&b[72..80], &r.uploaded.to_be_bytes());
        assert_eq!(&b[80..84], &r.event.udp_code().to_be_bytes());
        assert_eq!(&b[80..84], &2u32.to_be_bytes(), "started 在 UDP 里是 2");
        assert_eq!(&b[84..88], &0u32.to_be_bytes(), "ip 交给 tracker 从数据包取");
        assert_eq!(&b[88..92], &0x12345678u32.to_be_bytes(), "key 不能漏（漏了端口就错位）");
        assert_eq!(&b[92..96], &50u32.to_be_bytes(), "num_want");
        assert_eq!(&b[96..98], &6881u16.to_be_bytes(), "port 在最后 2 字节");
    }

    #[test]
    fn udp_parse_connect_response() {
        let mut b = Vec::new();
        b.extend_from_slice(&0u32.to_be_bytes());
        b.extend_from_slice(&0x1122_3344u32.to_be_bytes());
        b.extend_from_slice(&0xAABB_CCDD_EEFF_0011u64.to_be_bytes());
        let (tx, cid) = UdpAnnounce::parse_connect_response(&b).unwrap();
        assert_eq!(tx, 0x1122_3344);
        assert_eq!(cid, 0xAABB_CCDD_EEFF_0011);

        // 截断 → NeedMore
        assert!(UdpAnnounce::parse_connect_response(&b[..8]).unwrap_err().is_need_more());
        // 错误响应（action = 3）
        let mut e = Vec::new();
        e.extend_from_slice(&3u32.to_be_bytes());
        e.extend_from_slice(&0x1122_3344u32.to_be_bytes());
        e.extend_from_slice(b"bad connection id");
        assert!(matches!(
            UdpAnnounce::parse_connect_response(&e).unwrap_err(),
            MediaError::Protocol(_)
        ));
        // action 不对
        let mut w = Vec::new();
        w.extend_from_slice(&1u32.to_be_bytes());
        w.extend_from_slice(&0u32.to_be_bytes());
        w.extend_from_slice(&0u64.to_be_bytes());
        assert!(UdpAnnounce::parse_connect_response(&w).is_err());
    }

    #[test]
    fn udp_parse_announce_response() {
        let mut b = Vec::new();
        b.extend_from_slice(&1u32.to_be_bytes()); // action = announce
        b.extend_from_slice(&0x00A1_B2C3u32.to_be_bytes()); // transaction id
        b.extend_from_slice(&1800u32.to_be_bytes()); // interval
        b.extend_from_slice(&9u32.to_be_bytes()); // leechers
        b.extend_from_slice(&4u32.to_be_bytes()); // seeders
        b.extend_from_slice(&compact_v4(&[([1, 2, 3, 4], 6881), ([5, 6, 7, 8], 80)]));

        let (tx, r) = UdpAnnounce::parse_announce_response(&b).unwrap();
        assert_eq!(tx, 0x00A1_B2C3);
        assert_eq!(r.interval, 1800);
        assert_eq!(r.complete, 4, "seeders → complete");
        assert_eq!(r.incomplete, 9, "leechers → incomplete");
        assert_eq!(
            r.peers,
            vec![
                "1.2.3.4:6881".parse().unwrap(),
                "5.6.7.8:80".parse().unwrap()
            ]
        );
        assert!(r.peers6.is_empty(), "UDP 协议不带 IPv6 peer");

        // 截断 → NeedMore
        assert!(UdpAnnounce::parse_announce_response(&b[..19])
            .unwrap_err()
            .is_need_more());
        // 错误响应
        let mut e = Vec::new();
        e.extend_from_slice(&3u32.to_be_bytes());
        e.extend_from_slice(&1u32.to_be_bytes());
        e.extend_from_slice(b"nope");
        assert!(UdpAnnounce::parse_announce_response(&e).is_err());
        // 尾部残片（不是 6 的倍数）要被忽略，而不是 panic
        let mut partial = b.clone();
        partial.extend_from_slice(&[1, 2, 3]);
        assert_eq!(UdpAnnounce::parse_announce_response(&partial).unwrap().1.peers.len(), 2);
    }
}
