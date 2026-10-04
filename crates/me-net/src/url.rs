//! 自研 URL 处理。
//!
//! 为什么不用 `url` crate：本工程硬性要求「协议层不引入第三方依赖」，
//! 而且我们真正需要的只是 **HLS/DASH 里相对 URI 的拼接** 与 **查询串/百分号编解码**，
//! 一个完整 URL 库的 IDNA、punycode、序列化规则反而会带来"输入合法但输出被改写"
//! 的惊喜。自己写一份，行为边界完全可控。
//!
//! 踩过的坑，先说清楚：
//!
//! 1. **`resolve` 是 HLS 能不能播的生死线**。清单里大量出现 `seg1.ts`、`../v1/a.ts`、
//!    `?token=xxx` 这类相对引用；只要有一条拼错，取回来的就是 404 或者别家的分片。
//!    所以这里严格按 RFC 3986 §5 实现，并对每一类引用都写了测试。
//! 2. **base 自带查询串时，相对路径必须丢掉 base 的查询串**。`http://h/a/b.m3u8?t=1`
//!    拼 `c.ts` 应该是 `http://h/a/c.ts` 而不是 `http://h/a/c.ts?t=1` ——
//!    否则会把上一级清单的 token 带到分片请求上（有些站点会因此 403）。
//! 3. **只有 `?query` / `#frag` 时才继承 base 的路径**，其它相对引用一律"替换最后一段"。
//! 4. 百分号编码只做最朴素的规则：非 unreserved 且不在 `keep` 集合里的一律转义。
//!    解码则**宽容**：遇到非法 `%XX` 原样保留，绝不因为一个坏字节把整条 URL 丢掉
//!    （HLS 方言里出现裸 `%` 并不罕见）。

use me_core::error::{MediaError, Result};

/// 解析后的绝对 URL。
///
/// 刻意**不做百分号解码**：`path`/`query` 保持原文，因为我们要拿它去发请求，
/// 解码后再编码很容易把 `%2F` 变成 `/` 从而改变语义。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Url {
    pub scheme: String,
    pub host: String,
    pub port: Option<u16>,
    pub path: String,
    pub query: Option<String>,
    pub fragment: Option<String>,
}

impl Url {
    /// 解析一个**带 scheme 的绝对 URL**（`scheme://host[:port]/path?query#fragment`）。
    ///
    /// 相对引用不走这里，走 [`resolve`]。
    pub fn parse(s: &str) -> Result<Url> {
        let s = s.trim();
        if s.is_empty() {
            return Err(MediaError::protocol("URL 为空"));
        }
        // scheme 必须以 "://" 结束；没有就说明是相对引用（或畸形输入）。
        let sep = s
            .find("://")
            .ok_or_else(|| MediaError::protocol(format!("URL 缺少 scheme://: {s}")))?;
        let scheme = s[..sep].to_ascii_lowercase();
        if scheme.is_empty() || !is_valid_scheme(&scheme) {
            return Err(MediaError::protocol(format!("非法 scheme: {scheme}")));
        }
        let rest = &s[sep + 3..];
        // authority 到第一个 '/'、'?'、'#' 为止。
        let auth_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        let authority = &rest[..auth_end];
        let tail = &rest[auth_end..];

        // 丢掉 userinfo（我们不需要认证信息，且带 userinfo 的 URL 发出去容易泄漏）。
        let hostport = match authority.rfind('@') {
            Some(i) => &authority[i + 1..],
            None => authority,
        };

        let (host, port) = split_host_port(hostport)?;

        let (path_q, fragment) = split_once_char(tail, '#');
        let (path, query) = split_once_char(path_q, '?');

        Ok(Url {
            scheme,
            host,
            port,
            path: if path.is_empty() { "/".into() } else { path.into() },
            query: query.map(|q| q.to_string()),
            fragment: fragment.map(|f| f.to_string()),
        })
    }

    /// `scheme://host[:port]`（不含路径）。
    pub fn origin(&self) -> String {
        match self.port {
            Some(p) => format!("{}://{}:{}", self.scheme, host_display(&self.host), p),
            None => format!("{}://{}", self.scheme, host_display(&self.host)),
        }
    }

    /// 还原成完整字符串。IPv6 主机会自动补回方括号。
    pub fn as_string(&self) -> String {
        let mut s = self.origin();
        s.push_str(&self.path);
        if let Some(q) = &self.query {
            s.push('?');
            s.push_str(q);
        }
        if let Some(f) = &self.fragment {
            s.push('#');
            s.push_str(f);
        }
        s
    }

    /// 是不是 http/https。
    pub fn is_http(&self) -> bool {
        self.scheme == "http" || self.scheme == "https"
    }

    /// 换掉查询串，返回完整字符串（`q` 传 `""` 表示只留一个空 `?`）。
    pub fn with_query(&self, q: &str) -> String {
        let mut s = self.origin();
        s.push_str(&self.path);
        s.push('?');
        s.push_str(q);
        if let Some(f) = &self.fragment {
            s.push('#');
            s.push_str(f);
        }
        s
    }
}

/// 解析 `host[:port]`，IPv6 允许 `[::1]:8080` 形式。
fn split_host_port(s: &str) -> Result<(String, Option<u16>)> {
    if let Some(rest) = s.strip_prefix('[') {
        let close = rest
            .find(']')
            .ok_or_else(|| MediaError::protocol(format!("IPv6 字面量缺少 ']': {s}")))?;
        let host = &rest[..close];
        let after = &rest[close + 1..];
        let port = match after.strip_prefix(':') {
            Some(p) if !p.is_empty() => Some(parse_port(p)?),
            _ => None,
        };
        return Ok((host.to_string(), port));
    }
    match s.rfind(':') {
        Some(i) => {
            let port = parse_port(&s[i + 1..])?;
            Ok((s[..i].to_string(), Some(port)))
        }
        None => Ok((s.to_string(), None)),
    }
}

fn parse_port(p: &str) -> Result<u16> {
    p.parse::<u16>()
        .map_err(|_| MediaError::protocol(format!("非法端口: {p}")))
}

/// 输出主机时，含 ':' 的一律当作 IPv6 补上方括号。
fn host_display(host: &str) -> String {
    if host.contains(':') {
        format!("[{host}]")
    } else {
        host.to_string()
    }
}

fn is_valid_scheme(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.')
}

/// 在第一个 `sep` 处切成两半，分隔符本身被丢弃。
fn split_once_char(s: &str, sep: char) -> (&str, Option<&str>) {
    match s.find(sep) {
        Some(i) => (&s[..i], Some(&s[i + sep.len_utf8()..])),
        None => (s, None),
    }
}

// ---------------------------------------------------------------------------
// RFC 3986 §5 引用解析
// ---------------------------------------------------------------------------

/// 把 `reference`（相对或绝对）相对 `base` 解析成绝对 URL 字符串。
///
/// 覆盖的形态（每一种都有对应测试）：
/// 绝对 URL、`//host/path`、`/abs/path`、`rel/path`、`../rel`、`./rel`、
/// `?query`、`#frag`、空串。
pub fn resolve(base: &str, reference: &str) -> String {
    let reference = reference.trim();

    // 1) 空引用 = base 去掉 fragment。
    if reference.is_empty() {
        return strip_fragment(base).to_string();
    }

    // 2) 网络路径引用（协议相对）：换 authority，继承 base 的 scheme。
    if let Some(rest) = reference.strip_prefix("//") {
        let scheme = scheme_of(base);
        let (_, tail) = split_authority(rest);
        return format!("{scheme}://{}{}", authority_of(rest), normalize_tail(tail));
    }

    // 3) 绝对路径引用：继承 scheme + authority。
    if reference.starts_with('/') {
        let scheme = scheme_of(base);
        let auth = authority_of(after_scheme(base));
        return format!("{scheme}://{auth}{}", normalize_tail(reference));
    }

    // 4) 只有查询串：继承 base 的路径，丢掉 base 的查询串。
    if reference.starts_with('?') {
        let head = strip_query_and_fragment(base);
        return format!("{head}{reference}");
    }

    // 5) 只有锚点：继承 base 的查询串，换掉锚点。
    if reference.starts_with('#') {
        let head = strip_fragment(base);
        return format!("{head}{reference}");
    }

    // 6) 带 scheme 的绝对引用：原样返回（RFC 5.2.2）。
    if has_scheme(reference) {
        return reference.to_string();
    }

    // 7) 相对路径：丢掉 base 的查询串/锚点，替换最后一段后做点段归一。
    let base_no_qf = strip_query_and_fragment(base);
    let prefix = merge_prefix(base_no_qf);
    let merged = format!("{prefix}{reference}");
    // 点段归一**只能作用在路径上**：整条 URL 一起归一的话，`split('/')` 会把
    // `scheme://` 里的空段也吃掉，拼出 `http:/h/c.ts` 这种少一个斜杠的地址 ——
    // 发出去必然 404，而且是"看起来像对的"那种错，最难查。
    let (head, tail) = split_authority_end(&merged);
    format!("{head}{}", normalize_tail(tail))
}

/// 把绝对 URL 切成 `scheme://authority` 与 `路径[?查询][#锚点]` 两段。
fn split_authority_end(url: &str) -> (&str, &str) {
    let start = match url.find("://") {
        Some(i) => i + 3,
        None => 0,
    };
    match url[start..].find(['/', '?', '#']) {
        Some(i) => url.split_at(start + i),
        None => (url, ""),
    }
}

/// 相对路径合并时的前缀：base 里最后一个 '/' 及其之前的部分。
fn merge_prefix(base: &str) -> String {
    let after = after_scheme(base);
    let auth_len = authority_of(after).len();
    // authority 结束的绝对下标（"scheme://" 的长度 + authority 长度）。
    let auth_end = base.len() - after.len() + auth_len;
    match base.rfind('/') {
        // 只有当 '/' 落在 authority 之后才算路径里的 '/'。
        Some(i) if i >= auth_end => base[..=i].to_string(),
        _ => format!("{base}/"),
    }
}

/// 取 `scheme://` 的 scheme 部分；没有 scheme 时按 http 兜底（调用方保证 base 是绝对的）。
fn scheme_of(s: &str) -> &str {
    match s.find("://") {
        Some(i) => &s[..i],
        None => "http",
    }
}

/// 去掉 `scheme://` 前缀。
fn after_scheme(s: &str) -> &str {
    match s.find("://") {
        Some(i) => &s[i + 3..],
        None => s,
    }
}

/// 从 `host[:port]/tail...` 里切出 authority（不含 userinfo）。
fn authority_of(after_scheme: &str) -> &str {
    let end = after_scheme
        .find(['/', '?', '#'])
        .unwrap_or(after_scheme.len());
    let a = &after_scheme[..end];
    match a.rfind('@') {
        Some(i) => &a[i + 1..],
        None => a,
    }
}

/// 从 authority 开头切出 (authority, tail)。
fn split_authority(s: &str) -> (&str, &str) {
    let end = s.find(['/', '?', '#']).unwrap_or(s.len());
    (&s[..end], &s[end..])
}

fn strip_fragment(s: &str) -> &str {
    match s.find('#') {
        Some(i) => &s[..i],
        None => s,
    }
}

fn strip_query_and_fragment(s: &str) -> &str {
    match s.find(['?', '#']) {
        Some(i) => &s[..i],
        None => s,
    }
}

/// 判断字符串是否以 `scheme:` 开头（RFC 认为这就是绝对引用）。
fn has_scheme(s: &str) -> bool {
    let end = s.find(['/', '?', '#']).unwrap_or(s.len());
    let Some(colon) = s[..end].find(':') else {
        return false;
    };
    colon > 0 && is_valid_scheme(&s[..colon])
}

/// 只对 URL 的路径部分做点段归一，查询串与锚点原样保留。
fn normalize_tail(tail: &str) -> String {
    let (path, suffix) = match tail.find(['?', '#']) {
        Some(i) => (&tail[..i], &tail[i..]),
        None => (tail, ""),
    };
    format!("{}{suffix}", remove_dot_segments(path))
}

/// RFC 3986 §5.2.4：去掉 `.` 与 `..` 段。
fn remove_dot_segments(path: &str) -> String {
    let starts_slash = path.starts_with('/');
    let trailing_slash =
        path.ends_with('/') || path.ends_with("/.") || path.ends_with("/..");
    let mut stack: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                stack.pop();
            }
            other => stack.push(other),
        }
    }
    let mut out = String::new();
    if starts_slash {
        out.push('/');
    }
    out.push_str(&stack.join("/"));
    if trailing_slash && !out.ends_with('/') {
        out.push('/');
    }
    out
}

// ---------------------------------------------------------------------------
// 百分号编解码
// ---------------------------------------------------------------------------

/// 百分号编码。`keep` 里出现的字符不转义（unreserved 永远不转义）。
///
/// 例：`percent_encode("a b/c", "/")` → `"a%20b/c"`。
pub fn percent_encode(s: &str, keep: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        let c = b as char;
        let unreserved = c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '~');
        if unreserved || (keep.contains(c) && b < 0x80) {
            out.push(c);
        } else {
            out.push('%');
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0x0f) as usize] as char);
        }
    }
    out
}

/// 百分号解码。**宽容**：非法的 `%XX` 原样保留，不报错也不吞字符。
///
/// 注意不把 `+` 当成空格 —— 那是 form-urlencoded 的规则，路径里 `+` 就是 `+`。
pub fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2])) {
                out.push((h << 4) | l);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// 取 URL 的扩展名，**含 `.`**；没有后缀返回 `""`。
///
/// 只看最后一个路径段里的最后一个 `.`，并忽略查询串/锚点。
/// 形如 `.../v1.2/seg` 这种"点在前面的目录里"不会被误判。
pub fn extension_of(url: &str) -> &str {
    let end = url.find(['?', '#']).unwrap_or(url.len());
    let s = &url[..end];
    let last_slash = s.rfind('/').map(|i| i + 1).unwrap_or(0);
    let seg = &s[last_slash..];
    match seg.rfind('.') {
        Some(i) if i > 0 => &seg[i..],
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic() {
        let u = Url::parse("https://cdn.example.com/v1/index.m3u8?token=ab#frag").unwrap();
        assert_eq!(u.scheme, "https");
        assert_eq!(u.host, "cdn.example.com");
        assert_eq!(u.port, None);
        assert_eq!(u.path, "/v1/index.m3u8");
        assert_eq!(u.query.as_deref(), Some("token=ab"));
        assert_eq!(u.fragment.as_deref(), Some("frag"));
        assert!(u.is_http());
        assert_eq!(
            u.as_string(),
            "https://cdn.example.com/v1/index.m3u8?token=ab#frag"
        );
        assert_eq!(u.origin(), "https://cdn.example.com");
    }

    #[test]
    fn parse_port_and_default_path() {
        let u = Url::parse("http://h:8080").unwrap();
        assert_eq!(u.port, Some(8080));
        assert_eq!(u.path, "/");
        assert_eq!(u.origin(), "http://h:8080");
        assert_eq!(u.as_string(), "http://h:8080/");
    }

    #[test]
    fn parse_ipv6_roundtrip() {
        let u = Url::parse("http://[::1]:9000/a/b.ts").unwrap();
        assert_eq!(u.host, "::1");
        assert_eq!(u.port, Some(9000));
        assert_eq!(u.as_string(), "http://[::1]:9000/a/b.ts");
    }

    #[test]
    fn parse_rejects_relative_and_bad_port() {
        assert!(Url::parse("seg1.ts").is_err());
        assert!(Url::parse("").is_err());
        assert!(Url::parse("http://h:notaport/x").is_err());
    }

    #[test]
    fn with_query_replaces() {
        let u = Url::parse("http://h/a.m3u8?old=1").unwrap();
        assert_eq!(u.with_query("new=2"), "http://h/a.m3u8?new=2");
    }

    #[test]
    fn resolve_absolute_passthrough() {
        assert_eq!(
            resolve("http://h/a/b.m3u8", "https://other/x.ts"),
            "https://other/x.ts"
        );
    }

    #[test]
    fn resolve_scheme_relative() {
        assert_eq!(
            resolve("https://h/a/b.m3u8", "//cdn2/x.ts"),
            "https://cdn2/x.ts"
        );
    }

    #[test]
    fn resolve_absolute_path() {
        assert_eq!(resolve("https://h/a/b/c.m3u8", "/x/y.ts"), "https://h/x/y.ts");
        assert_eq!(
            resolve("https://h:8443/a/b/c.m3u8", "/x.ts"),
            "https://h:8443/x.ts"
        );
    }

    #[test]
    fn resolve_relative_replaces_last_segment() {
        assert_eq!(resolve("http://h/a/b.m3u8", "c.ts"), "http://h/a/c.ts");
        // base 无路径时补 '/'
        assert_eq!(resolve("http://h", "c.ts"), "http://h/c.ts");
    }

    #[test]
    fn resolve_directory_semantics() {
        // base 以 '/' 结尾 = 目录，直接拼
        assert_eq!(resolve("http://h/a/", "b.ts"), "http://h/a/b.ts");
    }

    #[test]
    fn resolve_dot_segments() {
        assert_eq!(resolve("http://h/a/b.m3u8", "../c.ts"), "http://h/c.ts");
        assert_eq!(resolve("http://h/a/b.m3u8", "./c.ts"), "http://h/a/c.ts");
        assert_eq!(
            resolve("http://h/a/b/c.m3u8", "../../x.ts"),
            "http://h/x.ts"
        );
        // 归一化不能把 authority 吃掉
        assert_eq!(
            resolve("http://h/a/b/c.m3u8", "../../../x.ts"),
            "http://h/x.ts"
        );
    }

    #[test]
    fn resolve_query_only_inherits_path() {
        assert_eq!(
            resolve("http://h/a/b.m3u8?t=1", "?t=2"),
            "http://h/a/b.m3u8?t=2"
        );
    }

    #[test]
    fn resolve_fragment_only_inherits_query() {
        assert_eq!(
            resolve("http://h/a/b.m3u8?t=1", "#frag"),
            "http://h/a/b.m3u8?t=1#frag"
        );
        assert_eq!(resolve("http://h/a/b.m3u8", "#frag"), "http://h/a/b.m3u8#frag");
    }

    #[test]
    fn resolve_empty_returns_base_without_fragment() {
        assert_eq!(resolve("http://h/a/b.m3u8?t=1#x", ""), "http://h/a/b.m3u8?t=1");
    }

    #[test]
    fn resolve_relative_drops_base_query() {
        // 关键用例：base 带 token，相对分片不能继承它
        assert_eq!(
            resolve("http://h/a/b.m3u8?token=secret", "c.ts"),
            "http://h/a/c.ts"
        );
        assert_eq!(
            resolve("http://h/a/b.m3u8?token=secret", "sub/c.ts"),
            "http://h/a/sub/c.ts"
        );
    }

    #[test]
    fn resolve_relative_keeps_reference_query_and_fragment() {
        assert_eq!(
            resolve("http://h/a/b.m3u8", "c.ts?x=1#f"),
            "http://h/a/c.ts?x=1#f"
        );
    }

    #[test]
    fn percent_roundtrip() {
        assert_eq!(percent_encode("a b/c", "/"), "a%20b/c");
        assert_eq!(percent_decode("a%20b/c"), "a b/c");
        // 保留集合里的字符不转义
        assert_eq!(percent_encode("k=v&x=1", "=&"), "k=v&x=1");
        // 非法转义原样保留
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%zz"), "%zz");
        // 中文（UTF-8）编解码
        let enc = percent_encode("中文", "");
        assert_eq!(percent_decode(&enc), "中文");
    }

    #[test]
    fn extension_extraction() {
        assert_eq!(extension_of("http://h/a/seg1.ts"), ".ts");
        assert_eq!(extension_of("http://h/a/seg1.ts?x=1"), ".ts");
        assert_eq!(extension_of("http://h/a/seg1"), "");
        assert_eq!(extension_of("http://h/v1.2/seg"), "");
        assert_eq!(extension_of("http://h/a/seg.m4s#f"), ".m4s");
    }
}
