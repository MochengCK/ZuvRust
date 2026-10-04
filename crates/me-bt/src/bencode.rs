//! bencode 编解码（BEP 3）。
//!
//! 为什么自己写而不是找现成库：本 crate 被约束为**零第三方依赖**，
//! 而且 bencode 里有两个和"媒体引擎"直接相关的坑，用库反而更容易踩：
//!
//! 1. **info_hash 必须对 `info` 字典在原始 .torrent 文件里的那段字节做 SHA-1**，
//!    不是"解析出来再 encode 一遍"。因为 bencode 的规范虽然要求字典键按字节序排列，
//!    但真实世界里的种子（尤其是老种子工具生成的）经常：
//!    - 键的顺序不是字典序；
//!    - 整数写成 `i05e`、`i-0e` 这种非规范形式。
//!    对"重新编码后的串"算 SHA-1 会得到另一个值，于是**连不上任何 peer、tracker 也认不出你**
//!    —— 这是 BT 客户端最经典的一类 bug，且几乎不可能靠肉眼发现（hash 值看起来"很正常"）。
//!    所以这里除了 [`BValue::parse`]，还提供 [`BValue::parse_with_span`]，
//!    把"这个值在原文里占了哪些字节"一并返回。
//! 2. **流式解析**：磁力链接拉元信息（ut_metadata）时，一个 bencode 值可能跨多个
//!    TCP 包到达。截断时返回 [`MediaError::NeedMore`] 而不是 `Protocol`，
//!    调用方才能"再读一段重来"，而不是误判成"对面发来的元信息是坏的"。
//!
//! 另外两个刻意的决定：
//!
//! - **字典用 `BTreeMap<Vec<u8>, BValue>`**：键是**字节串**不是 `String`。
//!   BT 里有大量非 UTF-8 的键/值（中文文件名、Windows 的 GBK 路径），
//!   键用 `String` 就得在解析阶段做一次有损转换，两个本不相同的键可能变成同一个。
//!   `BTreeMap` 按 `Vec<u8>` 的字典序迭代，这恰好就是 BEP 3 要求的"按字节序"，
//!   编码时直接顺序输出即可，不必再排序。
//! - **深度上限 100**：bencode 是递归语法，一个 `llllll...` 的恶意文件能直接爆栈。
//!   Rust 的栈溢出是**直接 abort**（没法 catch），进程整个没，所以只能在解析器里挡住。

use std::collections::BTreeMap;
use std::ops::Range;

use me_core::error::{MediaError, Result};

/// bencode 的最大嵌套深度。
///
/// 100 是"远超真实种子、又远低于爆栈"的经验值：真实种子里最深的也就是
/// `announce-list`（list of list of string，2 层）。恶意种子想用深度打我们，
/// 到 100 就报 `Protocol` 错误。
pub const MAX_DEPTH: usize = 100;

/// 一个 bencode 值。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BValue {
    /// 整数（`i42e`，支持负数）。
    Int(i64),
    /// 字节串（`4:spam`）。**长度是字节数不是字符数**。
    Bytes(Vec<u8>),
    /// 列表（`l...e`）。
    List(Vec<BValue>),
    /// 字典（`d...e`），键按字节序排列。
    Dict(BTreeMap<Vec<u8>, BValue>),
}

impl BValue {
    /// 解析一个值，返回 `(值, 消耗的字节数)`。
    ///
    /// 允许开头之后还有别的数据（这正是顶层字典 + `info` 的关系：我们只关心消耗了多少）。
    pub fn parse(data: &[u8]) -> Result<(BValue, usize)> {
        let (v, _pos, span) = Self::parse_with_span(data)?;
        Ok((v, span.end))
    }

    /// 解析一个值，并额外返回它**在原始输入里占用的字节区间**。
    ///
    /// `info_hash` 就得用这个区间去切原文做 SHA-1，见本模块开头的说明。
    pub fn parse_with_span(data: &[u8]) -> Result<(BValue, usize, Range<usize>)> {
        let mut pos = 0usize;
        let (v, span) = parse_value(data, &mut pos, 0)?;
        Ok((v, pos, span))
    }

    /// 编码。字典键按**字节序**输出（`BTreeMap` 的迭代顺序天然满足 BEP 3）。
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode_into(&mut out);
        out
    }

    fn encode_into(&self, out: &mut Vec<u8>) {
        match self {
            BValue::Int(i) => {
                out.push(b'i');
                out.extend_from_slice(i.to_string().as_bytes());
                out.push(b'e');
            }
            BValue::Bytes(b) => {
                out.extend_from_slice(b.len().to_string().as_bytes());
                out.push(b':');
                out.extend_from_slice(b);
            }
            BValue::List(l) => {
                out.push(b'l');
                for item in l {
                    item.encode_into(out);
                }
                out.push(b'e');
            }
            BValue::Dict(d) => {
                out.push(b'd');
                for (k, v) in d {
                    out.extend_from_slice(k.len().to_string().as_bytes());
                    out.push(b':');
                    out.extend_from_slice(k);
                    v.encode_into(out);
                }
                out.push(b'e');
            }
        }
    }

    /// 这个值是整数吗（只有 `Int` 才算，字节串不做隐式转换）。
    pub fn as_int(&self) -> Option<i64> {
        match self {
            BValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            BValue::Bytes(b) => Some(b),
            _ => None,
        }
    }

    /// 当字符串看。
    ///
    /// **非法 UTF-8 用 lossy 而不是报错**：中文文件名在 Windows 上常是 GBK，
    /// 报错会让整个种子打不开，而 lossy 至少能显示（且 info_hash 与其它字段都不受影响）。
    pub fn as_str(&self) -> Option<String> {
        self.as_bytes().map(|b| String::from_utf8_lossy(b).into_owned())
    }

    pub fn as_list(&self) -> Option<&[BValue]> {
        match self {
            BValue::List(l) => Some(l),
            _ => None,
        }
    }

    pub fn as_dict(&self) -> Option<&BTreeMap<Vec<u8>, BValue>> {
        match self {
            BValue::Dict(d) => Some(d),
            _ => None,
        }
    }

    /// 取字典里的某个键（非字典、键不存在都返回 `None`）。
    pub fn get(&self, key: &str) -> Option<&BValue> {
        self.as_dict()?.get(key.as_bytes())
    }

    pub fn get_str(&self, key: &str) -> Option<String> {
        self.get(key).and_then(|v| v.as_str())
    }

    pub fn get_int(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(|v| v.as_int())
    }
}

/// 越界时的统一返回。`have` 是"当前手上有多少字节"，调用方据此决定还要读多少。
fn need_more(need: usize, have: usize) -> MediaError {
    MediaError::NeedMore { need, have }
}

/// 看一眼第 `pos` 个字节；越界是 `NeedMore`（流式场景的正常路径）。
fn peek(buf: &[u8], pos: usize) -> Result<u8> {
    buf.get(pos).copied().ok_or_else(|| need_more(pos + 1, buf.len()))
}

/// 递归解析的核心。
///
/// 返回 `(值, 该值在 `buf` 里的原始字节区间)`。`pos` 是输入输出参数：
/// 进入时指向值的第一个字节，返回时指向值之后的那个字节。
pub(crate) fn parse_value(buf: &[u8], pos: &mut usize, depth: usize) -> Result<(BValue, Range<usize>)> {
    if depth > MAX_DEPTH {
        // 报错而不是继续递归：这里再往下走一层的代价是爆栈，而爆栈是 abort，
        // 调用方连"这个种子是坏的"都报不出来。
        return Err(MediaError::protocol(format!(
            "bencode 嵌套过深（超过 {MAX_DEPTH} 层）"
        )));
    }
    let start = *pos;
    let c = peek(buf, *pos)?;
    let value = match c {
        b'i' => parse_int(buf, pos)?,
        b'l' => parse_list(buf, pos, depth)?,
        b'd' => parse_dict(buf, pos, depth)?,
        b'0'..=b'9' => parse_bytes(buf, pos)?,
        other => {
            return Err(MediaError::protocol(format!(
                "非法的 bencode 起始字节 0x{other:02x}（位置 {pos}）"
            )))
        }
    };
    Ok((value, start..*pos))
}

/// `i<十进制>e`。刻意**接受** `i05e` / `i-0e` 这类非规范写法：
/// 真实种子里有，拒绝它们等于少连上一批 peer。
fn parse_int(buf: &[u8], pos: &mut usize) -> Result<BValue> {
    *pos += 1; // 吃掉 'i'
    let digits_start = *pos;
    loop {
        let c = peek(buf, *pos)?;
        if c == b'e' {
            break;
        }
        let is_sign = c == b'-' && *pos == digits_start;
        if !(c.is_ascii_digit() || is_sign) {
            return Err(MediaError::protocol(format!(
                "bencode 整数里有非法字符 0x{c:02x}（位置 {pos}）"
            )));
        }
        *pos += 1;
        // i64 最长 20 个字符（含符号），留 1 个余量；不设限的话 `i000...0e` 能吃掉整个文件。
        if *pos - digits_start > 21 {
            return Err(MediaError::protocol("bencode 整数过长"));
        }
    }
    if *pos == digits_start || (*pos == digits_start + 1 && buf[digits_start] == b'-') {
        return Err(MediaError::protocol("bencode 整数没有数字"));
    }
    let text = std::str::from_utf8(&buf[digits_start..*pos])
        .map_err(|e| MediaError::protocol(format!("bencode 整数不是 ASCII: {e}")))?;
    let v: i64 = text
        .parse()
        .map_err(|e| MediaError::protocol(format!("bencode 整数 `{text}` 溢出或非法: {e}")))?;
    *pos += 1; // 吃掉 'e'
    Ok(BValue::Int(v))
}

/// `<长度>:<字节>`，长度是**字节数**（中文名一个字 3 字节，按字符数读必然错位）。
fn parse_bytes(buf: &[u8], pos: &mut usize) -> Result<BValue> {
    let len_start = *pos;
    loop {
        let c = peek(buf, *pos)?;
        if c == b':' {
            break;
        }
        if !c.is_ascii_digit() {
            return Err(MediaError::protocol(format!(
                "bencode 字符串长度里有非法字符 0x{c:02x}（位置 {pos}）"
            )));
        }
        *pos += 1;
        if *pos - len_start > 19 {
            return Err(MediaError::protocol("bencode 字符串长度过长"));
        }
    }
    if *pos == len_start {
        return Err(MediaError::protocol("bencode 字符串长度为空的写法 `:`"));
    }
    let text = std::str::from_utf8(&buf[len_start..*pos])
        .map_err(|e| MediaError::protocol(format!("bencode 长度不是 ASCII: {e}")))?;
    let len: usize = text
        .parse()
        .map_err(|e| MediaError::protocol(format!("bencode 长度 `{text}` 溢出: {e}")))?;
    *pos += 1; // 吃掉 ':'
    let data_start = *pos;
    let end = data_start
        .checked_add(len)
        .ok_or_else(|| MediaError::protocol("bencode 字符串长度溢出 usize"))?;
    if buf.len() < end {
        // 关键：这里必须是 NeedMore。流式拉元信息时"字符串还没收全"是正常路径。
        return Err(need_more(end, buf.len()));
    }
    *pos = end;
    Ok(BValue::Bytes(buf[data_start..end].to_vec()))
}

/// `l...e`
fn parse_list(buf: &[u8], pos: &mut usize, depth: usize) -> Result<BValue> {
    *pos += 1; // 吃掉 'l'
    let mut items = Vec::new();
    loop {
        let c = peek(buf, *pos)?;
        if c == b'e' {
            *pos += 1;
            break;
        }
        let (item, _) = parse_value(buf, pos, depth + 1)?;
        items.push(item);
    }
    Ok(BValue::List(items))
}

/// `d...e`。**不要求键已经排好序**（解析要宽容，编码才严格按 BEP 3 排序）。
fn parse_dict(buf: &[u8], pos: &mut usize, depth: usize) -> Result<BValue> {
    *pos += 1; // 吃掉 'd'
    let mut map = BTreeMap::new();
    loop {
        let c = peek(buf, *pos)?;
        if c == b'e' {
            *pos += 1;
            break;
        }
        if !c.is_ascii_digit() {
            return Err(MediaError::protocol(format!(
                "bencode 字典的键必须是字符串，见到 0x{c:02x}（位置 {pos}）"
            )));
        }
        let key = parse_bytes(buf, pos)?;
        let key = match key {
            BValue::Bytes(b) => b,
            _ => unreachable!("parse_bytes 只会返回 Bytes"),
        };
        let (value, _) = parse_value(buf, pos, depth + 1)?;
        // 重复键：后者覆盖前者。BEP 3 说键唯一，真实种子里出现重复时
        // 用最后一个和用第一个都是"猜"，不额外报错（报错会让整个种子打不开）。
        map.insert(key, value);
    }
    Ok(BValue::Dict(map))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scalars() {
        assert_eq!(BValue::parse(b"i42e").unwrap(), (BValue::Int(42), 4));
        assert_eq!(BValue::parse(b"i-13e").unwrap().0, BValue::Int(-13));
        // i-0e 是合法的 0（BEP 3 里 -0 与 0 同义，非规范但真实存在）
        assert_eq!(BValue::parse(b"i-0e").unwrap().0, BValue::Int(0));
        // i05e 也是非规范写法，必须能吃下
        assert_eq!(BValue::parse(b"i05e").unwrap().0, BValue::Int(5));
        assert_eq!(
            BValue::parse(b"4:spam").unwrap().0,
            BValue::Bytes(b"spam".to_vec())
        );
        // 空字符串 / 空列表 / 空字典
        assert_eq!(BValue::parse(b"0:").unwrap().0, BValue::Bytes(vec![]));
        assert_eq!(BValue::parse(b"le").unwrap().0, BValue::List(vec![]));
        assert_eq!(BValue::parse(b"de").unwrap().0, BValue::Dict(BTreeMap::new()));
    }

    #[test]
    fn string_length_counts_bytes_not_chars() {
        // "中文" 是 6 字节 / 2 字符：按字符数写 2: 会解析出一个残缺的串
        let raw = "6:中文".as_bytes();
        let v = BValue::parse(raw).unwrap().0;
        assert_eq!(v.as_str().as_deref(), Some("中文"));
        assert_eq!(v.as_bytes().unwrap().len(), 6);
    }

    #[test]
    fn nested_roundtrip() {
        let raw = b"d8:announce14:http://a/b/c/d3:bar5:hello3:fooli1ei2eee";
        // 上面这个串的字典键顺序是 announce < bar < foo，恰好是字节序
        let (v, used) = BValue::parse(raw).unwrap();
        assert_eq!(used, raw.len());
        assert_eq!(v.get_str("announce").as_deref(), Some("http://a/b/c/d"));
        assert_eq!(v.get_str("bar").as_deref(), Some("hello"));
        assert_eq!(v.get("foo").unwrap().as_list().unwrap().len(), 2);
        assert_eq!(v.get("foo").unwrap().as_list().unwrap()[1], BValue::Int(2));
        assert_eq!(v.encode(), raw.to_vec(), "编码-解析往返必须回到原字节");
    }

    #[test]
    fn encode_sorts_dict_keys_by_byte_order() {
        let mut d = BTreeMap::new();
        // 故意乱序插入
        d.insert(b"pieces".to_vec(), BValue::Bytes(b"pp".to_vec()));
        d.insert(b"name".to_vec(), BValue::Bytes(b"n".to_vec()));
        d.insert(b"piece length".to_vec(), BValue::Int(16384));
        d.insert(b"length".to_vec(), BValue::Int(7));
        let out = BValue::Dict(d).encode();
        let text = String::from_utf8_lossy(&out);
        // BEP 3：键必须按字节序。' ' (0x20) < 's'，所以 "piece length" 在 "pieces" 之前
        assert_eq!(text, "d6:lengthi7e4:name1:n12:piece lengthi16384e6:pieces2:ppe");
    }

    #[test]
    fn lossy_utf8_is_not_an_error() {
        // 非法 UTF-8（GBK 的"中"是 0xD6 0xD0）
        let v = BValue::Bytes(vec![0xD6, 0xD0]);
        let s = v.as_str().unwrap();
        assert!(!s.is_empty(), "lossy 转换不应失败，只是内容不可读");
    }

    #[test]
    fn span_points_at_raw_bytes() {
        // 兄弟姐妹键乱序，且整数非规范：重编码一定与原文不同
        let raw = b"d4:infod6:lengthi05e4:name1:a6:pieces0:e4:junk3:xyze";
        let (root, used, span) = BValue::parse_with_span(raw).unwrap();
        assert_eq!(used, raw.len());
        // 顶层 span 是整串
        assert_eq!(span, 0..raw.len());
        // 手工定位 info 的那段
        let info_text = String::from_utf8_lossy(&raw[span.start..span.end]).to_string();
        let info_start = info_text.find("4:infod").unwrap() + "4:info".len();
        let info_raw = b"d6:lengthi05e4:name1:a6:pieces0:e";
        assert_eq!(&raw[info_start..info_start + info_raw.len()], info_raw, "原文不变");
        // 重编码后的 info 与原文不同（key 顺序被排过、i05e 变成 i5e）
        let info = root.get("info").unwrap();
        assert_ne!(info.encode(), info_raw.to_vec());
        assert_eq!(info.get_int("length"), Some(5));
    }

    #[test]
    fn truncated_input_reports_need_more() {
        // 真实运行里最常见的一类：TCP 包切在中间
        for cut in 1..b"d3:fooi42ee".len() {
            let e = BValue::parse(&b"d3:fooi42ee"[..cut]).unwrap_err();
            assert!(e.is_need_more(), "截断到 {cut} 字节时应当是 NeedMore，实得 {e:?}");
        }
        // 字符串长度说 4 但只给了 2 字节
        let e = BValue::parse(b"4:ab").unwrap_err();
        match e {
            MediaError::NeedMore { need, have } => {
                assert_eq!(need, 6);
                assert_eq!(have, 4);
            }
            other => panic!("应当是 NeedMore，实得 {other:?}"),
        }
        // 只差最后一个 'e'
        let e = BValue::parse(b"i42").unwrap_err();
        assert!(e.is_need_more());
    }

    #[test]
    fn deep_nesting_is_rejected_not_crashed() {
        let mut deep = Vec::new();
        deep.extend(std::iter::repeat(b'l').take(500));
        deep.extend(std::iter::repeat(b'e').take(500));
        let e = BValue::parse(&deep).unwrap_err();
        match e {
            MediaError::Protocol(msg) => assert!(msg.contains("嵌套过深"), "{msg}"),
            other => panic!("应当是 Protocol 错误，实得 {other:?}"),
        }
        // 正好在限制内是合法的
        let mut ok = Vec::new();
        ok.extend(std::iter::repeat(b'l').take(MAX_DEPTH));
        ok.extend(std::iter::repeat(b'e').take(MAX_DEPTH));
        assert!(BValue::parse(&ok).is_ok());
    }

    #[test]
    fn malformed_is_protocol_error() {
        assert!(matches!(
            BValue::parse(b"x42e").unwrap_err(),
            MediaError::Protocol(_)
        ));
        assert!(matches!(BValue::parse(b"ie").unwrap_err(), MediaError::Protocol(_)));
        assert!(matches!(BValue::parse(b"i1.5e").unwrap_err(), MediaError::Protocol(_)));
        assert!(matches!(BValue::parse(b":").unwrap_err(), MediaError::Protocol(_)));
        // 字典的键必须是字符串：`di1ei2ee` 是错的
        assert!(matches!(
            BValue::parse(b"di1ei2ee").unwrap_err(),
            MediaError::Protocol(_)
        ));
        // 整数溢出
        assert!(matches!(
            BValue::parse(b"i99999999999999999999e").unwrap_err(),
            MediaError::Protocol(_)
        ));
    }

    #[test]
    fn numbers_are_not_coerced_to_strings() {
        // 严格类型：拿 Int 当字符串取会得到 None，逼调用方显式处理
        let v = BValue::Int(3);
        assert!(v.as_str().is_none());
        assert!(v.as_bytes().is_none());
        assert_eq!(v.as_int(), Some(3));
        assert!(v.get("any").is_none());
        assert!(v.get_int("any").is_none());
    }

    #[test]
    fn duplicate_dict_keys_do_not_fail() {
        let (v, _) = BValue::parse(b"d1:ai1e1:ai2ee").unwrap();
        assert_eq!(v.get_int("a"), Some(2));
    }

    #[test]
    fn trailing_data_is_left_to_caller() {
        let (v, used) = BValue::parse(b"i1eXXX").unwrap();
        assert_eq!(v, BValue::Int(1));
        assert_eq!(used, 3);
    }
}
