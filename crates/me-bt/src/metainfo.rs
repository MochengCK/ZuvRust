//! 种子元信息：`.torrent` 解析、磁力链接解析、info_hash。
//!
//! 这一层的三个"必须想清楚"的地方：
//!
//! 1. **info_hash 用原文切片算**。见 [`MetaInfo::from_torrent`] 里的注释 —— 它决定了
//!    你能不能和别的客户端对上话（tracker 与 peer 都拿它当种子的唯一 ID）。
//! 2. **单文件与多文件统一成一张 `files` 表**。上层（片存储、落盘、边下边播）
//!    如果对单文件走一条路、多文件走另一条路，那么"跨文件读一段字节"就要写两遍，
//!    而且必然在"最后一个文件的最后一片"这种边界上出错。这里把单文件也变成
//!    一项 `FileEntry`，上层只认"偏移量 → 文件"这一种映射。
//! 3. **磁力链接里的百分号编码是原始字节级的**，不是"文本转义"。
//!    `dn` 里的中文是 UTF-8 的 `%E4%B8%AD`，而 `xt` 的 info_hash 是 20 个
//!    **二进制字节**，解码时必须按字节走，不能先转成 `String` 再处理。

use me_core::error::{MediaError, Result};

use crate::bencode::{parse_value, BValue};

/// 种子里的一个文件。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    /// 相对于种子根目录的路径。
    ///
    /// 单文件种子就是文件名本身；多文件种子是 `path` 列表用 `/` 拼起来的相对路径
    /// （BT 的 `path` 天然是分段的，用 `/` 拼接在 Windows 上落盘时再换分隔符即可 ——
    /// 在这一层保留"平台无关"的形式，别提前引入 `\`）。
    pub path: String,
    /// 文件长度（字节）。
    pub length: u64,
    /// 该文件在整个种子**逻辑字节流**里的起始偏移。
    ///
    /// 边下边播要的就是这个：播放器说"我要第 1GB"，我们得知道它落在哪个文件里。
    pub offset: u64,
}

/// 解析好的种子元信息。
#[derive(Debug, Clone)]
pub struct MetaInfo {
    /// 对 `info` 字典**原始字节**做的 SHA-1。
    pub info_hash: [u8; 20],
    /// 种子名（单文件种子它是文件名，多文件种子它是根目录名）。
    pub name: String,
    /// 每片的字节数。
    pub piece_length: u64,
    /// 每片的 SHA-1。
    pub piece_hashes: Vec<[u8; 20]>,
    /// 文件表（单文件种子也是 1 项）。
    pub files: Vec<FileEntry>,
    /// 所有文件长度之和。
    pub total_length: u64,
    pub is_multi_file: bool,
    /// BEP 27：`private=1` 表示禁止 DHT/PEX，只走 tracker。
    pub private: bool,
    /// 去重后的 tracker 列表（只保留 http/https/udp）。
    pub announce: Vec<String>,
    pub comment: Option<String>,
    pub created_by: Option<String>,
    pub creation_date: Option<i64>,
}

impl MetaInfo {
    /// 从 `.torrent` 的原始字节解析。
    pub fn from_torrent(data: &[u8]) -> Result<Self> {
        let (root, _used) = BValue::parse(data)?;
        root.as_dict()
            .ok_or_else(|| MediaError::protocol("种子文件的顶层不是 bencode 字典"))?;

        // ---- info_hash：**必须**用 info 字典在原文里的那一段 ----
        // 先按 bencode 语法"走"一遍顶层字典，记下每个值的字节区间，找到 info 的那一段。
        // 为什么不直接 `root.get("info").encode()`：见 bencode 模块开头的说明 ——
        // 键顺序、`i05e` 这类非规范写法都会让重编码结果与原文不同，
        // 算出来的 hash 会是一个"看着很正常但谁也认不出"的值。
        let (info_start, info_end) = find_info_span(data)?;
        let info_hash = crate::sha1(&data[info_start..info_end]);
        let info = root
            .get("info")
            .ok_or_else(|| MediaError::protocol("种子文件缺少 info 字典"))?;

        let name = info.get_str("name").unwrap_or_default();

        let piece_length = info
            .get_int("piece length")
            .ok_or_else(|| MediaError::protocol("info 缺少 piece length"))?;
        if piece_length <= 0 {
            return Err(MediaError::protocol("piece length 必须为正数"));
        }
        let piece_length = piece_length as u64;

        let pieces_raw = info
            .get("pieces")
            .and_then(|v| v.as_bytes())
            .ok_or_else(|| MediaError::protocol("info 缺少 pieces（片哈希串）"))?;
        if pieces_raw.len() % 20 != 0 {
            return Err(MediaError::protocol(format!(
                "pieces 长度 {} 不是 20 的整数倍",
                pieces_raw.len()
            )));
        }
        let piece_hashes: Vec<[u8; 20]> = pieces_raw
            .chunks_exact(20)
            .map(|c| {
                let mut h = [0u8; 20];
                h.copy_from_slice(c);
                h
            })
            .collect();

        // ---- 文件表：单文件 / 多文件统一处理 ----
        let mut files = Vec::new();
        let mut total_length = 0u64;
        let is_multi_file = if let Some(list) = info.get("files").and_then(|v| v.as_list()) {
            if list.is_empty() {
                return Err(MediaError::protocol("info.files 是空列表"));
            }
            for (i, f) in list.iter().enumerate() {
                let length = f
                    .get_int("length")
                    .ok_or_else(|| MediaError::protocol(format!("files[{i}] 缺少 length")))?;
                if length < 0 {
                    return Err(MediaError::protocol(format!("files[{i}] 的长度为负")));
                }
                let parts = f
                    .get("path")
                    .and_then(|v| v.as_list())
                    .ok_or_else(|| MediaError::protocol(format!("files[{i}] 缺少 path")))?;
                if parts.is_empty() {
                    return Err(MediaError::protocol(format!("files[{i}] 的 path 是空列表")));
                }
                let path = parts
                    .iter()
                    .map(|p| p.as_str().unwrap_or_default())
                    .collect::<Vec<_>>()
                    .join("/");
                files.push(FileEntry {
                    path,
                    length: length as u64,
                    offset: total_length,
                });
                total_length += length as u64;
            }
            true
        } else if let Some(length) = info.get_int("length") {
            if length < 0 {
                return Err(MediaError::protocol("info.length 为负"));
            }
            files.push(FileEntry {
                path: name.clone(),
                length: length as u64,
                offset: 0,
            });
            total_length = length as u64;
            false
        } else {
            return Err(MediaError::protocol("info 里既没有 length 也没有 files"));
        };

        // 片数与总长必须自洽：不自洽说明 pieces 字段被截断了，
        // 这时候最危险的做法是"按现有 hash 数量凑合"——那样每个片的边界都会错位。
        let expect_pieces = total_length.div_ceil(piece_length);
        if piece_hashes.len() as u64 != expect_pieces {
            return Err(MediaError::protocol(format!(
                "片数不自洽：总长 {total_length} / 片长 {piece_length} 应为 {expect_pieces} 片，实际 {} 片",
                piece_hashes.len()
            )));
        }

        // ---- tracker 列表 ----
        let mut announce: Vec<String> = Vec::new();
        let mut push_tracker = |raw: String| {
            let url = raw.trim();
            if !is_tracker_url(url) {
                return;
            }
            if !announce.iter().any(|x| x == url) {
                announce.push(url.to_string());
            }
        };
        if let Some(a) = root.get_str("announce") {
            push_tracker(a);
        }
        if let Some(tiers) = root.get("announce-list").and_then(|v| v.as_list()) {
            for tier in tiers {
                // 标准形态是 list of list of string，但也见过直接给字符串的
                let items: Vec<&BValue> = match tier.as_list() {
                    Some(l) => l.iter().collect(),
                    None => vec![tier],
                };
                for it in items {
                    if let Some(u) = it.as_str() {
                        push_tracker(u);
                    }
                }
            }
        }
        drop(push_tracker);

        Ok(MetaInfo {
            info_hash,
            name,
            piece_length,
            piece_hashes,
            files,
            total_length,
            is_multi_file,
            private: info.get_int("private") == Some(1),
            announce,
            comment: root.get_str("comment"),
            created_by: root.get_str("created by"),
            creation_date: root.get_int("creation date"),
        })
    }

    pub fn piece_length(&self) -> u64 {
        self.piece_length
    }

    pub fn num_pieces(&self) -> u32 {
        self.piece_hashes.len() as u32
    }

    /// 第 `index` 片的真实长度。
    ///
    /// **最后一片通常短于 `piece_length`**（总长几乎不可能刚好是片长的整数倍），
    /// 拿 `piece_length` 当最后一片的长度去校验/读取是 BT 实现里最经典的 off-by-one，
    /// 症状是"整片文件下到最后一片总是校验失败"。
    pub fn piece_size(&self, index: u32) -> u32 {
        if index >= self.num_pieces() {
            return 0;
        }
        let start = index as u64 * self.piece_length;
        let remain = self.total_length.saturating_sub(start);
        remain.min(self.piece_length).min(u32::MAX as u64) as u32
    }

    /// 某个逻辑字节偏移落在哪个文件里。
    pub fn file_of_offset(&self, offset: u64) -> Option<&FileEntry> {
        self.files
            .iter()
            .find(|f| offset >= f.offset && offset < f.offset.saturating_add(f.length))
    }

    /// 落盘用的相对路径。
    pub fn relative_path(&self, f: &FileEntry) -> String {
        if self.is_multi_file {
            if self.name.is_empty() {
                f.path.clone()
            } else {
                format!("{}/{}", self.name, f.path)
            }
        } else {
            // 单文件种子的"相对路径"就是文件名，前面**不要**再套一层目录名，
            // 否则会和绝大多数客户端（以及用户的预期）不一致。
            f.path.clone()
        }
    }

    /// 给通信层用的 JSON 形态（Android / Electron 侧展示种子信息）。
    pub fn to_json(&self) -> serde_json::Value {
        let files: Vec<serde_json::Value> = self
            .files
            .iter()
            .map(|f| {
                serde_json::json!({
                    "path": f.path,
                    "relative_path": self.relative_path(f),
                    "length": f.length,
                    "offset": f.offset,
                })
            })
            .collect();
        serde_json::json!({
            "info_hash": infohash_hex(&self.info_hash),
            "name": self.name,
            "piece_length": self.piece_length,
            "num_pieces": self.num_pieces(),
            "total_length": self.total_length,
            "is_multi_file": self.is_multi_file,
            "private": self.private,
            "announce": self.announce,
            "comment": self.comment,
            "created_by": self.created_by,
            "creation_date": self.creation_date,
            "files": files,
        })
    }
}

/// 磁力链接（BEP 9 的 `xt=urn:btih:` 形态）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Magnet {
    pub info_hash: [u8; 20],
    pub name: Option<String>,
    pub trackers: Vec<String>,
    pub web_seeds: Vec<String>,
}

/// 解析磁力链接。
pub fn parse_magnet(uri: &str) -> Result<Magnet> {
    let trimmed = uri.trim();
    let rest = split_scheme(trimmed)
        .ok_or_else(|| MediaError::protocol("不是 magnet: 链接"))?;

    let mut info_hash: Option<[u8; 20]> = None;
    let mut name = None;
    let mut trackers = Vec::new();
    let mut web_seeds = Vec::new();

    for pair in rest.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (raw_key, raw_val) = match pair.split_once('=') {
            Some(kv) => kv,
            None => continue,
        };
        // 键名大小写按标准是小写，但有些工具会输出大写，宽容一点。
        if raw_key.eq_ignore_ascii_case("xt") {
            let v = percent_decode(raw_val);
            // urn:btih:<40 hex> 或 <32 base32>；btmh（多哈希）本版不支持，直接跳过
            let hash_part = match strip_prefix_ci(&v, "urn:btih:") {
                Some(h) => h,
                None => continue,
            };
            let h = if hash_part.len() == 40 {
                infohash_from_hex(&hash_part)?
            } else if hash_part.len() == 32 {
                base32_decode(&hash_part).ok_or_else(|| {
                    MediaError::protocol(format!("磁力链接的 base32 info_hash 非法: {hash_part}"))
                })?
            } else {
                return Err(MediaError::protocol(format!(
                    "磁力链接的 info_hash 长度应为 40(hex) 或 32(base32)，实际 {}",
                    hash_part.len()
                )));
            };
            if let Some(prev) = info_hash {
                if prev != h {
                    return Err(MediaError::protocol("磁力链接里有多个互相矛盾的 xt"));
                }
            }
            info_hash = Some(h);
        } else if raw_key.eq_ignore_ascii_case("dn") {
            let v = percent_decode(raw_val);
            if !v.is_empty() {
                name = Some(v);
            }
        } else if raw_key.eq_ignore_ascii_case("tr") {
            let v = percent_decode(raw_val);
            if !v.is_empty() {
                trackers.push(v);
            }
        } else if raw_key.eq_ignore_ascii_case("ws") {
            let v = percent_decode(raw_val);
            if !v.is_empty() {
                web_seeds.push(v);
            }
        }
    }

    Ok(Magnet {
        info_hash: info_hash.ok_or_else(|| MediaError::protocol("磁力链接里没有 xt=urn:btih:"))?,
        name,
        trackers,
        web_seeds,
    })
}

/// 20 字节 info_hash 的十六进制表示（40 个字符，小写）。
///
/// 固定小写：这个串会被拿去当文件名、URL 参数、日志字段，
/// 大小写不统一会在"按字符串比较"的地方产生极难查的重复。
pub fn infohash_hex(h: &[u8; 20]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(40);
    for b in h {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

/// 从 40 个十六进制字符解析 info_hash（大小写都接受）。
pub fn infohash_from_hex(s: &str) -> Result<[u8; 20]> {
    let b = s.as_bytes();
    if b.len() != 40 {
        return Err(MediaError::protocol(format!(
            "info_hash 的 hex 长度应为 40，实际 {}",
            b.len()
        )));
    }
    let mut out = [0u8; 20];
    for i in 0..20 {
        let hi = hex_val(b[i * 2]).ok_or_else(|| {
            MediaError::protocol(format!("info_hash 的 hex 里有非法字符: {}", s))
        })?;
        let lo = hex_val(b[i * 2 + 1]).ok_or_else(|| {
            MediaError::protocol(format!("info_hash 的 hex 里有非法字符: {}", s))
        })?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// 跳过 `magnet:` 与 `?`，返回 query 部分。用 `get()` 切片以避免切在 UTF-8 边界中间 panic。
fn split_scheme(s: &str) -> Option<&str> {
    let head = s.get(..7)?;
    if !head.eq_ignore_ascii_case("magnet:") {
        return None;
    }
    let rest = &s[7..];
    let rest = rest.strip_prefix('?').unwrap_or(rest);
    Some(rest)
}

fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    if s.len() >= prefix.len() && s[..prefix.len()].eq_ignore_ascii_case(prefix) {
        Some(&s[prefix.len()..])
    } else {
        None
    }
}

/// 百分号解码，**按字节处理**。
///
/// `plus_as_space`：`+` 当空格是 `application/x-www-form-urlencoded` 的惯例
/// （tracker 的 query 里就是这个规矩），磁力链接的 `dn`/`tr` 也照这个处理
/// —— 实际数据里确实有工具用 `+` 表示空格。
///
/// 非法转义（`%` 后面不是两位 hex）**原样保留**：宁可让用户看到一个带 `%` 的名字，
/// 也不要因为一个坏字节丢掉整个磁力链接。
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                match (hex_val(bytes[i + 1]), hex_val(bytes[i + 2])) {
                    (Some(hi), Some(lo)) => {
                        out.push((hi << 4) | lo);
                        i += 3;
                    }
                    _ => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    // 中文名字解码出来是 UTF-8 字节，正常；真碰上非 UTF-8（老工具用 GBK 编码的 dn）
    // 就 lossy 一次，不报错 —— 名字只是个提示，info_hash 才是身份。
    String::from_utf8_lossy(&out).into_owned()
}

/// Base32（RFC 4648 字母表，`A-Z2-7`）解码，只用于解析 32 字符的 `urn:btih`。
///
/// 不引第三方 base32 库的原因和别的模块一样：本 crate 零依赖。
/// 尾部 `=` 补位容忍（有些工具会带），其余字符一律视为非法。
fn base32_decode(s: &str) -> Option<[u8; 20]> {
    let mut out: Vec<u8> = Vec::with_capacity(20);
    let mut acc: u32 = 0;
    let mut bits: u32 = 0;
    for c in s.chars() {
        if c == '=' {
            continue;
        }
        let v = match c {
            'A'..='Z' => c as u32 - 'A' as u32,
            'a'..='z' => c as u32 - 'a' as u32,
            '2'..='7' => c as u32 - '2' as u32 + 26,
            _ => return None,
        };
        acc = (acc << 5) | v;
        bits += 5;
        if bits >= 8 {
            bits -= 8;
            out.push(((acc >> bits) & 0xff) as u8);
        }
    }
    if out.len() != 20 {
        return None;
    }
    let mut h = [0u8; 20];
    h.copy_from_slice(&out);
    Some(h)
}

fn is_tracker_url(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://") || lower.starts_with("udp://")
}

/// 找出顶层字典里 `info` 值的**原始字节区间**。
///
/// 做法是"按语法重走一遍顶层字典"，而不是在 [`BValue`] 上找 ——
/// 因为树上不保留位置信息，而位置恰恰是 info_hash 唯一需要的东西。
/// 只走顶层（不递归进 info 内部），代价是 O(文件大小) 的一次线性扫描，
/// 对种子文件（通常几百 KB）完全可接受。
pub(crate) fn find_info_span(data: &[u8]) -> Result<(usize, usize)> {
    if data.first() != Some(&b'd') {
        return Err(MediaError::protocol("种子文件的顶层不是 bencode 字典"));
    }
    let mut pos = 1usize;
    loop {
        match data.get(pos) {
            None => {
                // 顶层字典没闭合：和 bencode 的约定保持一致，算"数据不够"
                return Err(MediaError::NeedMore {
                    need: pos + 1,
                    have: data.len(),
                })
            }
            Some(b'e') => break,
            Some(_) => {}
        }
        let (key, _) = parse_value(data, &mut pos, 0)?;
        let key = match key {
            BValue::Bytes(b) => b,
            _ => return Err(MediaError::protocol("种子顶层字典的键不是字符串")),
        };
        let value_start = pos;
        let (_v, _) = parse_value(data, &mut pos, 0)?;
        if key == b"info" {
            return Ok((value_start, pos));
        }
    }
    Err(MediaError::protocol("种子文件缺少 info 字典"))
}

/// 从 `BValue` 形态的 info 字典算出 info_hash —— **只用于测试**：
/// 它走的是"重编码"路径，和真实路径（原文切片）会有差异，
/// 存在的意义就是让测试能断言"两条路径的结果确实不同"。
#[cfg(test)]
pub(crate) fn infohash_by_reencode(info: &BValue) -> [u8; 20] {
    crate::sha1(&info.encode())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bencode::BValue;
    use std::collections::BTreeMap;

    fn fake_pieces(n: usize) -> Vec<u8> {
        (0..n).flat_map(|i| [i as u8; 20]).collect()
    }

    /// 用 `BValue` 结构化地造一个 .torrent。
    /// 键交给 `encode()` 排序，省得手写长度数错（手写长度是这类测试最常见的翻车点）。
    fn make_torrent(
        announce: &str,
        extra_top: Vec<(&str, BValue)>,
        info_extra: Vec<(&str, BValue)>,
        pieces: &[u8],
    ) -> Vec<u8> {
        let mut info = BTreeMap::new();
        for (k, v) in info_extra {
            info.insert(k.as_bytes().to_vec(), v);
        }
        info.insert(b"pieces".to_vec(), BValue::Bytes(pieces.to_vec()));
        let mut top = BTreeMap::new();
        top.insert(b"announce".to_vec(), bytes(announce));
        top.insert(b"info".to_vec(), BValue::Dict(info));
        for (k, v) in extra_top {
            top.insert(k.as_bytes().to_vec(), v);
        }
        BValue::Dict(top).encode()
    }

    /// 顶层字典里塞一段**原始的** info 字节（用来构造"非规范的 info"）。
    fn wrap_raw_info(announce: &str, info_raw: &[u8]) -> Vec<u8> {
        let mut raw = Vec::new();
        raw.extend_from_slice(b"d8:announce");
        raw.extend_from_slice(format!("{}:{}", announce.len(), announce).as_bytes());
        raw.extend_from_slice(b"4:info");
        raw.extend_from_slice(info_raw);
        raw.push(b'e');
        raw
    }

    fn bytes(s: &str) -> BValue {
        BValue::Bytes(s.as_bytes().to_vec())
    }

    fn file_entry(length: u64, path: &[&str]) -> BValue {
        let mut d = BTreeMap::new();
        d.insert(b"length".to_vec(), BValue::Int(length as i64));
        d.insert(
            b"path".to_vec(),
            BValue::List(path.iter().map(|p| bytes(p)).collect()),
        );
        BValue::Dict(d)
    }

    #[test]
    fn single_file_torrent() {
        let pieces = fake_pieces(2);
        let raw = make_torrent(
            "http://tracker.example/announce",
            vec![],
            vec![
                ("length", BValue::Int(100)),
                ("name", bytes("test.bin")),
                ("piece length", BValue::Int(64)),
            ],
            &pieces,
        );

        let m = MetaInfo::from_torrent(&raw).unwrap();
        assert_eq!(m.name, "test.bin");
        assert_eq!(m.piece_length(), 64);
        assert_eq!(m.num_pieces(), 2);
        assert_eq!(m.total_length, 100);
        assert!(!m.is_multi_file);
        assert_eq!(m.files.len(), 1, "单文件种子也必须有 1 项 files");
        assert_eq!(m.files[0].path, "test.bin");
        assert_eq!(m.files[0].offset, 0);
        assert_eq!(m.files[0].length, 100);
        // 最后一片是短的：100 = 64 + 36
        assert_eq!(m.piece_size(0), 64);
        assert_eq!(m.piece_size(1), 36);
        assert_eq!(m.piece_size(2), 0, "越界的片长度为 0");
        assert_eq!(m.announce, vec!["http://tracker.example/announce"]);
        assert_eq!(m.relative_path(&m.files[0]), "test.bin");
        assert_eq!(m.piece_hashes[1], [1u8; 20]);
    }

    #[test]
    fn info_hash_uses_raw_bytes_not_reencode() {
        // 手工拼一个**非规范**的 info：键不是字节序、整数带前导零。
        // 这两种写法在真实种子里都有，而且重编码后字节一定不同。
        let pieces = fake_pieces(2);
        let mut info_raw = Vec::new();
        info_raw.extend_from_slice(b"d4:name8:test.bin6:lengthi0100e12:piece lengthi64e6:pieces");
        info_raw.extend_from_slice(format!("{}:", pieces.len()).as_bytes());
        info_raw.extend_from_slice(&pieces);
        info_raw.push(b'e');
        let raw = wrap_raw_info("http://t/announce", &info_raw);

        let m = MetaInfo::from_torrent(&raw).unwrap();
        let (s, e) = find_info_span(&raw).unwrap();
        assert_eq!(&raw[s..e], &info_raw[..], "span 必须正好框住原始 info 字节");
        assert_eq!(
            m.info_hash,
            crate::sha1(&info_raw),
            "info_hash 必须等于对原文切片做的 SHA-1"
        );

        // 反证：重编码后的 info 与被切片的那段**字节不同**，
        // 因此"重编码再 hash"会得到另一个值（这就是要防的 bug）
        let info_value = BValue::parse(&info_raw).unwrap().0;
        assert_ne!(info_value.encode(), info_raw.clone());
        assert_ne!(infohash_by_reencode(&info_value), m.info_hash);

        // 解析结果不受非规范写法影响
        assert_eq!(m.total_length, 100);
        assert_eq!(m.name, "test.bin");
    }

    #[test]
    fn multi_file_torrent_offsets_and_paths() {
        // 三个文件：64 + 10 + 200 = 274 字节，片长 64 → 5 片（最后一片 18 字节）
        let pieces = fake_pieces(5);
        let raw = make_torrent(
            "http://t/announce",
            vec![],
            vec![
                (
                    "files",
                    BValue::List(vec![
                        file_entry(64, &["a.bin"]),
                        file_entry(10, &["dir", "b.bin"]),
                        file_entry(200, &["c"]),
                    ]),
                ),
                ("name", bytes("root")),
                ("piece length", BValue::Int(64)),
            ],
            &pieces,
        );

        let m = MetaInfo::from_torrent(&raw).unwrap();
        assert!(m.is_multi_file);
        assert_eq!(m.name, "root");
        assert_eq!(m.total_length, 274);
        assert_eq!(m.num_pieces(), 5);
        assert_eq!(m.files.len(), 3);
        assert_eq!(m.files[0].path, "a.bin");
        assert_eq!(m.files[0].offset, 0);
        assert_eq!(m.files[1].path, "dir/b.bin", "多级目录用 / 拼");
        assert_eq!(m.files[1].offset, 64);
        assert_eq!(m.files[2].path, "c");
        assert_eq!(m.files[2].offset, 74);
        assert_eq!(m.files[2].length, 200);
        // 最后一片：274 - 4*64 = 18
        assert_eq!(m.piece_size(4), 18);
        assert_eq!(
            m.relative_path(&m.files[1]),
            "root/dir/b.bin",
            "多文件种子落盘要带根目录"
        );
        // 偏移量 → 文件（跨文件边界的偏移最容易写错）
        assert_eq!(m.file_of_offset(0).unwrap().path, "a.bin");
        assert_eq!(m.file_of_offset(63).unwrap().path, "a.bin");
        assert_eq!(m.file_of_offset(64).unwrap().path, "dir/b.bin");
        assert_eq!(m.file_of_offset(73).unwrap().path, "dir/b.bin");
        assert_eq!(m.file_of_offset(74).unwrap().path, "c");
        assert_eq!(m.file_of_offset(273).unwrap().path, "c");
        assert!(m.file_of_offset(274).is_none());
    }

    #[test]
    fn pieces_count_must_match_total_length() {
        // 总长 100 / 片长 64 → 应 2 片，这里只给 1 片
        let raw = make_torrent(
            "http://t/a",
            vec![],
            vec![
                ("length", BValue::Int(100)),
                ("name", bytes("a")),
                ("piece length", BValue::Int(64)),
            ],
            &fake_pieces(1),
        );
        let e = MetaInfo::from_torrent(&raw).unwrap_err();
        assert!(matches!(e, MediaError::Protocol(_)), "{e:?}");

        // pieces 长度不是 20 的倍数（字段被截断了）
        let raw = make_torrent(
            "http://t/a",
            vec![],
            vec![
                ("length", BValue::Int(64)),
                ("name", bytes("a")),
                ("piece length", BValue::Int(64)),
            ],
            &[0u8; 19],
        );
        let e = MetaInfo::from_torrent(&raw).unwrap_err();
        assert!(matches!(e, MediaError::Protocol(_)), "{e:?}");

        // 既没有 length 也没有 files
        let raw = make_torrent(
            "http://t/a",
            vec![],
            vec![("name", bytes("a")), ("piece length", BValue::Int(64))],
            &fake_pieces(1),
        );
        assert!(MetaInfo::from_torrent(&raw).is_err());

        // 顶层不是字典
        assert!(MetaInfo::from_torrent(b"i1e").is_err());
        // 缺 info
        assert!(MetaInfo::from_torrent(b"d8:announce3:abce").is_err());
    }

    #[test]
    fn trackers_are_deduped_and_scheme_filtered() {
        let raw = make_torrent(
            "http://t1.example/announce",
            vec![(
                "announce-list",
                BValue::List(vec![
                    BValue::List(vec![
                        bytes("http://t1.example/announce"),
                        bytes("udp://t2.example:6868"),
                    ]),
                    BValue::List(vec![bytes("https://t3.example")]),
                    BValue::List(vec![bytes("wss://not-a-tracker")]),
                ]),
            )],
            vec![
                ("length", BValue::Int(64)),
                ("name", bytes("a")),
                ("piece length", BValue::Int(64)),
            ],
            &fake_pieces(1),
        );
        let m = MetaInfo::from_torrent(&raw).unwrap();
        assert_eq!(
            m.announce,
            vec![
                "http://t1.example/announce",
                "udp://t2.example:6868",
                "https://t3.example"
            ],
            "重复的 announce 只留一份，非 http/https/udp 的丢掉"
        );
    }

    #[test]
    fn private_flag_and_metadata_fields() {
        let raw = make_torrent(
            "http://t/a",
            vec![
                ("comment", bytes("hello there")),
                ("created by", bytes("me-bt 0.1")),
                ("creation date", BValue::Int(1700000000)),
            ],
            vec![
                ("length", BValue::Int(64)),
                ("name", bytes("a")),
                ("piece length", BValue::Int(64)),
                ("private", BValue::Int(1)),
            ],
            &fake_pieces(1),
        );
        let m = MetaInfo::from_torrent(&raw).unwrap();
        assert!(m.private);
        assert_eq!(m.created_by.as_deref(), Some("me-bt 0.1"));
        assert_eq!(m.creation_date, Some(1700000000));
        assert_eq!(m.comment.as_deref(), Some("hello there"));

        let j = m.to_json();
        assert_eq!(j["num_pieces"], 1);
        assert_eq!(j["info_hash"].as_str().unwrap().len(), 40);
        assert_eq!(j["is_multi_file"], false);
        assert_eq!(j["files"][0]["relative_path"], "a");
        assert_eq!(j["files"][0]["length"], 64);

        // private 缺省为 false
        let raw = make_torrent(
            "http://t/a",
            vec![],
            vec![
                ("length", BValue::Int(64)),
                ("name", bytes("a")),
                ("piece length", BValue::Int(64)),
            ],
            &fake_pieces(1),
        );
        assert!(!MetaInfo::from_torrent(&raw).unwrap().private);
    }

    #[test]
    fn utf8_name_survives() {
        let raw = make_torrent(
            "http://t/a",
            vec![],
            vec![
                ("length", BValue::Int(64)),
                ("name", bytes("中文名.bin")),
                ("piece length", BValue::Int(64)),
            ],
            &fake_pieces(1),
        );
        let m = MetaInfo::from_torrent(&raw).unwrap();
        assert_eq!(m.name, "中文名.bin");
        assert_eq!(m.files[0].path, "中文名.bin");
    }

    #[test]
    fn infohash_hex_roundtrip() {
        let h = [0x00u8, 0x20, 0x25, 0xff, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let hex = infohash_hex(&h);
        assert_eq!(hex.len(), 40);
        assert!(hex.chars().all(|c| !c.is_ascii_uppercase()), "必须是小写");
        assert_eq!(infohash_from_hex(&hex).unwrap(), h);
        // 大写也能解析
        assert_eq!(infohash_from_hex(&hex.to_uppercase()).unwrap(), h);
        assert!(infohash_from_hex("abc").is_err());
        assert!(infohash_from_hex(&"z".repeat(40)).is_err());
    }

    #[test]
    fn magnet_hex_form() {
        let hex = "0123456789abcdef0123456789abcdef01234567";
        let uri = format!(
            "magnet:?xt=urn:btih:{hex}&dn=My%20Movie%202024&tr=http%3A%2F%2Ft1%2Fannounce&tr=udp%3A%2F%2Ft2%3A80&ws=http%3A%2F%2Fw%2Ff.bin"
        );
        let m = parse_magnet(&uri).unwrap();
        assert_eq!(infohash_hex(&m.info_hash), hex);
        assert_eq!(m.name.as_deref(), Some("My Movie 2024"));
        assert_eq!(m.trackers, vec!["http://t1/announce", "udp://t2:80"]);
        assert_eq!(m.web_seeds, vec!["http://w/f.bin"]);
    }

    #[test]
    fn magnet_base32_form() {
        // 20 字节全 0 的 base32 是 32 个 'A'
        let uri = format!("magnet:?xt=urn:btih:{}", "A".repeat(32));
        let m = parse_magnet(&uri).unwrap();
        assert_eq!(m.info_hash, [0u8; 20]);

        // 与 hex 形态必须得到同一个 hash
        let h = [0xdeu8, 0xad, 0xbe, 0xef, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
            0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00];
        let b32 = base32_encode_for_test(&h);
        assert_eq!(b32.len(), 32);
        let m2 = parse_magnet(&format!("magnet:?xt=urn:btih:{b32}")).unwrap();
        assert_eq!(m2.info_hash, h);
        let m3 = parse_magnet(&format!("magnet:?xt=urn:btih:{}", infohash_hex(&h))).unwrap();
        assert_eq!(m3.info_hash, h);
    }

    /// 测试用的 base32 编码（只为了让上面的用例能造出合法的 32 字符串）
    fn base32_encode_for_test(data: &[u8]) -> String {
        const ALPHABET: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
        let mut out = String::new();
        let mut acc: u32 = 0;
        let mut bits: u32 = 0;
        for b in data {
            acc = (acc << 8) | *b as u32;
            bits += 8;
            while bits >= 5 {
                bits -= 5;
                out.push(ALPHABET[((acc >> bits) & 0x1f) as usize] as char);
            }
        }
        if bits > 0 {
            out.push(ALPHABET[((acc << (5 - bits)) & 0x1f) as usize] as char);
        }
        out
    }

    #[test]
    fn magnet_chinese_name_and_plus_as_space() {
        // "测试.mp4" 的 UTF-8 百分号编码
        let uri = "magnet:?xt=urn:btih:0123456789abcdef0123456789abcdef01234567&dn=%E6%B5%8B%E8%AF%95.mp4&tr=udp%3A%2F%2Ft%2Fannounce";
        let m = parse_magnet(uri).unwrap();
        assert_eq!(m.name.as_deref(), Some("测试.mp4"));
        assert_eq!(m.trackers, vec!["udp://t/announce"]);

        // `+` 当空格（tracker 惯例）
        let m2 = parse_magnet(
            "magnet:?xt=urn:btih:0123456789abcdef0123456789abcdef01234567&dn=a+b%2Bc",
        )
        .unwrap();
        assert_eq!(m2.name.as_deref(), Some("a b+c"));
    }

    #[test]
    fn magnet_missing_xt_is_error() {
        assert!(parse_magnet("magnet:?dn=no-hash").is_err());
        assert!(parse_magnet("http://not-a-magnet").is_err());
        assert!(parse_magnet("magnet:?xt=urn:btih:tooshort").is_err());
        // btmh（多哈希）本版不支持，应当明确报"没有 btih"而不是崩
        assert!(parse_magnet("magnet:?xt=urn:btmh:1220deadbeef").is_err());
    }

    #[test]
    fn percent_decode_keeps_bad_escapes() {
        assert_eq!(percent_decode("a%2Gb"), "a%2Gb");
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%25"), "%");
        assert_eq!(percent_decode("%00"), "\u{0}");
    }

    #[test]
    fn base32_rejects_bad_chars() {
        assert!(base32_decode("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!").is_none());
        // 长度对但含非法字符
        assert!(base32_decode("1AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA").is_none());
    }
}
