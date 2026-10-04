//! peer wire 协议（BEP 3 基础 / BEP 6 Fast / BEP 9 元信息 / BEP 10 扩展）。
//!
//! 这一层只做"字节 ↔ 结构体"的翻译，**不碰 socket**：谁连谁、连上了发什么，
//! 是上层会话管理的事。把协议编解码独立出来的好处是它能被完整地单测 ——
//! 没有网络也能断言"截断的包返回 NeedMore，而不是解析出一个看着正常其实是坏的消息"。
//!
//! 几个真实世界里一定会踩的坑，都在这里挡住：
//!
//! 1. **握手先看 `pstrlen`**。`BitTorrent protocol` 这 19 个字节前面还有一个长度字节。
//!    对面发来的是别的协议（或干脆是随机字节）时不校验 `pstrlen`，就会把它的
//!    第 20 个字节当成保留位、之后全部错位 —— 症状是"连上了，但一个片也下不到"，
//!    而且看日志（连接建立成功）完全查不出来。
//! 2. **消息长度前缀必须有上限**。长度前缀是 `u32`，对面可以声明 4GB。
//!    照着它分配缓冲就是一次远程 OOM。这里限 [`MAX_MESSAGE_SIZE`]（1 MiB，
//!    远大于正常最大的 Piece 块），超了就报 `Protocol`。
//! 3. **扩展位与 Fast 位在 `reserved` 里位置不同**，见 [`Handshake::new`]：
//!    `reserved[5] & 0x10` 是扩展协议（BEP 10），`reserved[7] & 0x04` 是 Fast（BEP 6）。
//!    记混了的表现是"对面明明支持 ut_metadata，我们却永远拉不到元信息"。

use std::collections::BTreeMap;

use me_core::error::{MediaError, Result};

use crate::storage::Bitfield;

/// 请求块大小。16 KiB 是 BEP 3 的惯例值：太小则请求数爆炸、太大则丢一个块就浪费很多带宽。
pub const BLOCK_SIZE: u32 = 16 * 1024;

/// 握手协议串。**前面还要加一个长度字节**（19），见 [`Handshake::encode`]。
pub const PROTOCOL: &str = "BitTorrent protocol";

/// 握手固定 68 字节 = 1(pstrlen) + 19(pstr) + 8(reserved) + 20(info_hash) + 20(peer_id)。
pub const HANDSHAKE_LEN: usize = 68;

/// 单条消息（含 4 字节长度前缀）的上限。见模块开头第 2 条。
pub const MAX_MESSAGE_SIZE: usize = 1024 * 1024;

/// ut_metadata（BEP 9）每块的字节数。
pub const METADATA_PIECE_SIZE: u32 = 16 * 1024;

/// 握手。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Handshake {
    pub info_hash: [u8; 20],
    pub peer_id: [u8; 20],
    /// 8 个能力位。只有我们关心的两位会被读写，其余原样保留。
    pub reserved: [u8; 8],
}

impl Handshake {
    /// 构造握手，并在 `reserved` 里置上**扩展协议位**与**Fast 位**。
    ///
    /// 两个位的位置是不同 BEP 各自规定的，容易记混：
    /// - BEP 10（扩展协议）：`reserved[5] & 0x10`；
    /// - BEP 6（Fast extension）：`reserved[7] & 0x04`。
    pub fn new(info_hash: [u8; 20], peer_id: [u8; 20]) -> Self {
        let mut reserved = [0u8; 8];
        reserved[5] |= 0x10; // BEP 10 扩展协议
        reserved[7] |= 0x04; // BEP 6 Fast extension
        Self {
            info_hash,
            peer_id,
            reserved,
        }
    }

    /// 编码成 68 字节。
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HANDSHAKE_LEN);
        out.push(PROTOCOL.len() as u8);
        out.extend_from_slice(PROTOCOL.as_bytes());
        out.extend_from_slice(&self.reserved);
        out.extend_from_slice(&self.info_hash);
        out.extend_from_slice(&self.peer_id);
        debug_assert_eq!(out.len(), HANDSHAKE_LEN);
        out
    }

    /// 从字节还原。不足 68 字节返回 [`MediaError::NeedMore`]（流式读取的正常路径），
    /// 而 `pstrlen` 或协议串不对则是**真正的协议错误**，两者必须分开。
    pub fn decode(b: &[u8]) -> Result<Self> {
        if b.len() < HANDSHAKE_LEN {
            return Err(MediaError::NeedMore {
                need: HANDSHAKE_LEN,
                have: b.len(),
            });
        }
        // pstrlen：对面可能根本不是 BitTorrent（见模块开头第 1 条）。
        let pstrlen = b[0] as usize;
        if pstrlen != PROTOCOL.len() {
            return Err(MediaError::protocol(format!(
                "握手 pstrlen 应为 {}，实得 {pstrlen}",
                PROTOCOL.len()
            )));
        }
        if &b[1..1 + PROTOCOL.len()] != PROTOCOL.as_bytes() {
            return Err(MediaError::protocol("握手协议串不是 BitTorrent protocol"));
        }
        let mut reserved = [0u8; 8];
        reserved.copy_from_slice(&b[20..28]);
        let mut info_hash = [0u8; 20];
        info_hash.copy_from_slice(&b[28..48]);
        let mut peer_id = [0u8; 20];
        peer_id.copy_from_slice(&b[48..68]);
        Ok(Self {
            info_hash,
            peer_id,
            reserved,
        })
    }

    /// 对面是否支持扩展协议（BEP 10）。
    pub fn supports_extended(&self) -> bool {
        self.reserved[5] & 0x10 != 0
    }

    /// 对面是否支持 Fast extension（BEP 6）。
    pub fn supports_fast(&self) -> bool {
        self.reserved[7] & 0x04 != 0
    }
}

/// 生成 peer_id：固定 8 字节前缀 `-ME0001-` + 12 字节随机。
///
/// 前缀里带版本号是为了在 tracker 的日志里能认出"这是我们的客户端"；
/// 后面必须随机，否则同一台机器起两个实例会撞成同一个 peer_id，
/// 被 tracker 判成重复、也让我们在 swarm 里被当成自己。
pub fn generate_peer_id() -> [u8; 20] {
    const PREFIX: &[u8; 8] = b"-ME0001-";
    let mut id = [0u8; 20];
    id[..8].copy_from_slice(PREFIX);
    // getrandom 失败（极罕见）时退化成固定值也没关系：peer_id 不是安全边界，
    // 只要求"不同实例不同"，不值得为此让构造函数变成 Result。
    if getrandom::fill(&mut id[8..]).is_err() {
        tracing::warn!("获取随机数失败，peer_id 的后半段退化为固定值");
        for (i, b) in id[8..].iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(37).wrapping_add(11);
        }
    }
    id
}

/// 一条 peer 消息。变体顺序与消息 id 对应，便于对照 BEP。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message {
    /// 长度前缀为 0，**没有 id 字节**。
    KeepAlive,
    Choke,
    Unchoke,
    Interested,
    NotInterested,
    Have(u32),
    Bitfield(Bitfield),
    Request { index: u32, begin: u32, length: u32 },
    Piece { index: u32, begin: u32, data: Vec<u8> },
    Cancel { index: u32, begin: u32, length: u32 },
    Port(u16),
    // ---- BEP 6 Fast extension ----
    HaveAll,
    HaveNone,
    Suggest(u32),
    RejectRequest { index: u32, begin: u32, length: u32 },
    AllowedFast(u32),
    // ---- BEP 10 扩展协议 ----
    Extended { id: u8, payload: Vec<u8> },
    /// 我们不认识的消息 id。**原样保留**而不是报错：BT 扩展很多，
    /// 因为对面多发一种消息就断连是得不偿失的。
    Unknown { id: u8, payload: Vec<u8> },
}

/// [`Message::KeepAlive`] 的伪 id。KeepAlive 在线上没有 id 字节，
/// 用 255（即 `-1` 的 u8 形态）当哨兵；所有真实消息 id 都 ≤ 20，不会冲突。
pub const KEEPALIVE_ID: u8 = 255;

impl Message {
    /// 编码，含 4 字节大端长度前缀。`KeepAlive` 就是 4 个 0。
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Message::KeepAlive => vec![0, 0, 0, 0],
            Message::Choke => frame(0, &[]),
            Message::Unchoke => frame(1, &[]),
            Message::Interested => frame(2, &[]),
            Message::NotInterested => frame(3, &[]),
            Message::Have(i) => frame(4, &i.to_be_bytes()),
            Message::Bitfield(bf) => frame(5, &bf.to_bytes()),
            Message::Request {
                index,
                begin,
                length,
            } => frame(6, &block_args(*index, *begin, *length)),
            Message::Piece {
                index,
                begin,
                data,
            } => {
                let mut p = Vec::with_capacity(8 + data.len());
                p.extend_from_slice(&index.to_be_bytes());
                p.extend_from_slice(&begin.to_be_bytes());
                p.extend_from_slice(data);
                frame(7, &p)
            }
            Message::Cancel {
                index,
                begin,
                length,
            } => frame(8, &block_args(*index, *begin, *length)),
            Message::Port(p) => frame(9, &p.to_be_bytes()),
            Message::Suggest(i) => frame(13, &i.to_be_bytes()),
            Message::HaveAll => frame(14, &[]),
            Message::HaveNone => frame(15, &[]),
            Message::RejectRequest {
                index,
                begin,
                length,
            } => frame(16, &block_args(*index, *begin, *length)),
            Message::AllowedFast(i) => frame(17, &i.to_be_bytes()),
            Message::Extended { id, payload } => {
                let mut p = Vec::with_capacity(1 + payload.len());
                p.push(*id);
                p.extend_from_slice(payload);
                frame(20, &p)
            }
            Message::Unknown { id, payload } => frame(*id, payload),
        }
    }

    /// 解码，返回 `(消息, 消耗的字节数)`。数据不够时返回 [`MediaError::NeedMore`]，
    /// 调用方保留剩余字节、再读一段后重来。
    pub fn decode(buf: &[u8]) -> Result<(Message, usize)> {
        if buf.len() < 4 {
            return Err(MediaError::NeedMore { need: 4, have: buf.len() });
        }
        let len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        if len > MAX_MESSAGE_SIZE {
            return Err(MediaError::protocol(format!(
                "peer 消息长度 {len} 超过上限 {MAX_MESSAGE_SIZE}"
            )));
        }
        let total = 4 + len;
        if buf.len() < total {
            return Err(MediaError::NeedMore {
                need: total,
                have: buf.len(),
            });
        }
        if len == 0 {
            return Ok((Message::KeepAlive, 4));
        }
        let id = buf[4];
        let payload = &buf[5..total];
        let msg = match id {
            0 => {
                expect_empty(payload, "choke")?;
                Message::Choke
            }
            1 => {
                expect_empty(payload, "unchoke")?;
                Message::Unchoke
            }
            2 => {
                expect_empty(payload, "interested")?;
                Message::Interested
            }
            3 => {
                expect_empty(payload, "not interested")?;
                Message::NotInterested
            }
            4 => {
                expect_len(payload, 4, "have")?;
                Message::Have(be_u32(payload))
            }
            5 => {
                // 线层**不知道种子的真实片数**，只能按"字节数 × 8"暂存；
                // 末字节的补齐位因此暂时保留。调用方拿到真实片数后应当用
                // `Bitfield::from_bytes(raw, true_count)` 重解析一次，
                // 那一步才会把补齐位清掉（storage 层已有对应测试）。
                let num_pieces = (payload.len() * 8) as u32;
                Message::Bitfield(Bitfield::from_bytes(payload, num_pieces)?)
            }
            6 => {
                let (index, begin, length) = parse_block_args(payload, "request")?;
                Message::Request {
                    index,
                    begin,
                    length,
                }
            }
            7 => {
                if payload.len() < 8 {
                    return Err(MediaError::protocol(format!(
                        "piece 消息负载至少 8 字节，实得 {}",
                        payload.len()
                    )));
                }
                Message::Piece {
                    index: be_u32(payload),
                    begin: be_u32(&payload[4..]),
                    data: payload[8..].to_vec(),
                }
            }
            8 => {
                let (index, begin, length) = parse_block_args(payload, "cancel")?;
                Message::Cancel {
                    index,
                    begin,
                    length,
                }
            }
            9 => {
                expect_len(payload, 2, "port")?;
                Message::Port(be_u16(payload))
            }
            13 => {
                expect_len(payload, 4, "suggest")?;
                Message::Suggest(be_u32(payload))
            }
            14 => {
                expect_empty(payload, "have all")?;
                Message::HaveAll
            }
            15 => {
                expect_empty(payload, "have none")?;
                Message::HaveNone
            }
            16 => {
                let (index, begin, length) = parse_block_args(payload, "reject request")?;
                Message::RejectRequest {
                    index,
                    begin,
                    length,
                }
            }
            17 => {
                expect_len(payload, 4, "allowed fast")?;
                Message::AllowedFast(be_u32(payload))
            }
            20 => {
                if payload.is_empty() {
                    return Err(MediaError::protocol("extended 消息缺少扩展 id"));
                }
                Message::Extended {
                    id: payload[0],
                    payload: payload[1..].to_vec(),
                }
            }
            other => Message::Unknown {
                id: other,
                payload: payload.to_vec(),
            },
        };
        Ok((msg, total))
    }

    /// 消息 id。`KeepAlive` 返回 [`KEEPALIVE_ID`]（它在线上没有 id 字节）。
    pub fn id(&self) -> u8 {
        match self {
            Message::KeepAlive => KEEPALIVE_ID,
            Message::Choke => 0,
            Message::Unchoke => 1,
            Message::Interested => 2,
            Message::NotInterested => 3,
            Message::Have(_) => 4,
            Message::Bitfield(_) => 5,
            Message::Request { .. } => 6,
            Message::Piece { .. } => 7,
            Message::Cancel { .. } => 8,
            Message::Port(_) => 9,
            Message::Suggest(_) => 13,
            Message::HaveAll => 14,
            Message::HaveNone => 15,
            Message::RejectRequest { .. } => 16,
            Message::AllowedFast(_) => 17,
            Message::Extended { .. } => 20,
            Message::Unknown { id, .. } => *id,
        }
    }
}

/// 加 4 字节大端长度前缀（长度含 id 字节本身）。
fn frame(id: u8, payload: &[u8]) -> Vec<u8> {
    let len = 1 + payload.len();
    let mut out = Vec::with_capacity(4 + len);
    out.extend_from_slice(&(len as u32).to_be_bytes());
    out.push(id);
    out.extend_from_slice(payload);
    out
}

/// `Request` / `Cancel` / `RejectRequest` 共用的 12 字节负载。
fn block_args(index: u32, begin: u32, length: u32) -> [u8; 12] {
    let mut p = [0u8; 12];
    p[0..4].copy_from_slice(&index.to_be_bytes());
    p[4..8].copy_from_slice(&begin.to_be_bytes());
    p[8..12].copy_from_slice(&length.to_be_bytes());
    p
}

fn parse_block_args(payload: &[u8], what: &str) -> Result<(u32, u32, u32)> {
    expect_len(payload, 12, what)?;
    Ok((
        be_u32(payload),
        be_u32(&payload[4..]),
        be_u32(&payload[8..]),
    ))
}

/// 只在长度已经校验过之后调用（越界会 panic，这是**编程错误**不是输入错误）。
fn be_u32(p: &[u8]) -> u32 {
    u32::from_be_bytes([p[0], p[1], p[2], p[3]])
}

fn be_u16(p: &[u8]) -> u16 {
    u16::from_be_bytes([p[0], p[1]])
}

fn expect_empty(payload: &[u8], what: &str) -> Result<()> {
    if payload.is_empty() {
        Ok(())
    } else {
        Err(MediaError::protocol(format!(
            "{what} 消息不该带负载（实得 {} 字节）",
            payload.len()
        )))
    }
}

fn expect_len(payload: &[u8], n: usize, what: &str) -> Result<()> {
    if payload.len() == n {
        Ok(())
    } else {
        Err(MediaError::protocol(format!(
            "{what} 消息负载应为 {n} 字节，实得 {}",
            payload.len()
        )))
    }
}

/// 扩展握手（BEP 10 的 `extended id = 0` 消息的负载，本身是一段 bencode 字典）。
///
/// 我们只关心其中三项：`m`（扩展名 → 扩展 id 的映射）、`metadata_size`
/// （ut_metadata 要用它算分块数）、`reqq`（对面的请求队列深度，用来限流）。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ExtendedHandshake {
    /// 扩展名 → 对面给它的 id。发 ut_metadata 请求时要查这张表拿 id。
    pub m: BTreeMap<String, u8>,
    pub metadata_size: Option<u32>,
    pub reqq: Option<u32>,
    /// 对面的客户端名（`v` 字段），只用于日志/诊断。
    pub v: Option<String>,
}

impl ExtendedHandshake {
    pub fn parse(b: &[u8]) -> Result<Self> {
        let (v, _) = crate::bencode::BValue::parse(b)?;
        let mut m = BTreeMap::new();
        if let Some(d) = v.get("m").and_then(|x| x.as_dict()) {
            for (k, val) in d {
                // 扩展名一定是 ASCII；非法 UTF-8 的键直接丢掉（不认识它也用不了）。
                if let (Ok(name), Some(n)) = (String::from_utf8(k.clone()), val.as_int()) {
                    m.insert(name, n.clamp(0, 255) as u8);
                }
            }
        }
        Ok(Self {
            m,
            metadata_size: v.get_int("metadata_size").map(|x| x.max(0) as u32),
            reqq: v.get_int("reqq").map(|x| x.max(0) as u32),
            v: v.get_str("v"),
        })
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut d = BTreeMap::new();
        let mut m = BTreeMap::new();
        for (k, id) in &self.m {
            m.insert(k.as_bytes().to_vec(), crate::bencode::BValue::Int(*id as i64));
        }
        d.insert(b"m".to_vec(), crate::bencode::BValue::Dict(m));
        if let Some(s) = self.metadata_size {
            d.insert(
                b"metadata_size".to_vec(),
                crate::bencode::BValue::Int(s as i64),
            );
        }
        if let Some(s) = self.reqq {
            d.insert(b"reqq".to_vec(), crate::bencode::BValue::Int(s as i64));
        }
        if let Some(s) = &self.v {
            d.insert(
                b"v".to_vec(),
                crate::bencode::BValue::Bytes(s.as_bytes().to_vec()),
            );
        }
        crate::bencode::BValue::Dict(d).encode()
    }
}

/// ut_metadata（BEP 9）消息。**负载是一段 bencode 字典**，
/// 而 `Data` 还要在字典**之后**直接跟一段原始字节（元信息分块）。
///
/// 这个"字典 + 裸数据"的拼法是最容易实现错的地方：把 `data` 塞进字典
/// （当成 bencode 字符串）会让对面解析出长度前缀、多一个 `n:` 前缀，整块元信息就废了。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetadataMessage {
    /// `msg_type = 0`
    Request(u32),
    /// `msg_type = 1`：字典之后跟着这一块的真实字节。
    Data {
        piece: u32,
        total_size: u32,
        data: Vec<u8>,
    },
    /// `msg_type = 2`
    Reject(u32),
}

impl MetadataMessage {
    pub fn parse(b: &[u8]) -> Result<Self> {
        let (v, used) = crate::bencode::BValue::parse(b)?;
        let msg_type = v
            .get_int("msg_type")
            .ok_or_else(|| MediaError::protocol("ut_metadata 消息缺少 msg_type"))?;
        let piece = v
            .get_int("piece")
            .ok_or_else(|| MediaError::protocol("ut_metadata 消息缺少 piece"))?;
        if piece < 0 {
            return Err(MediaError::protocol("ut_metadata 的 piece 为负"));
        }
        let piece = piece as u32;
        match msg_type {
            0 => Ok(MetadataMessage::Request(piece)),
            1 => {
                let total_size = v
                    .get_int("total_size")
                    .ok_or_else(|| MediaError::protocol("ut_metadata data 缺少 total_size"))?;
                if total_size < 0 {
                    return Err(MediaError::protocol("ut_metadata 的 total_size 为负"));
                }
                Ok(MetadataMessage::Data {
                    piece,
                    total_size: total_size as u32,
                    data: b[used..].to_vec(),
                })
            }
            2 => Ok(MetadataMessage::Reject(piece)),
            other => Err(MediaError::protocol(format!(
                "未知的 ut_metadata msg_type: {other}"
            ))),
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        match self {
            MetadataMessage::Request(piece) => meta_dict(0, *piece, None),
            MetadataMessage::Reject(piece) => meta_dict(2, *piece, None),
            MetadataMessage::Data {
                piece,
                total_size,
                data,
            } => {
                let mut out = meta_dict(1, *piece, Some(*total_size));
                out.extend_from_slice(data);
                out
            }
        }
    }
}

fn meta_dict(msg_type: i64, piece: u32, total_size: Option<u32>) -> Vec<u8> {
    use crate::bencode::BValue;
    let mut d = BTreeMap::new();
    d.insert(b"msg_type".to_vec(), BValue::Int(msg_type));
    d.insert(b"piece".to_vec(), BValue::Int(piece as i64));
    if let Some(t) = total_size {
        d.insert(b"total_size".to_vec(), BValue::Int(t as i64));
    }
    BValue::Dict(d).encode()
}

/// 元信息要分成几块（每块 [`METADATA_PIECE_SIZE`] 字节，向上取整）。
pub fn metadata_piece_count(metadata_size: u32) -> u32 {
    metadata_size.div_ceil(METADATA_PIECE_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handshake_roundtrip_and_capability_bits() {
        let ih = [0x11u8; 20];
        let pid = generate_peer_id();
        let h = Handshake::new(ih, pid);
        assert!(h.supports_extended(), "BEP 10 扩展位必须置上");
        assert!(h.supports_fast(), "BEP 6 Fast 位必须置上");
        assert_eq!(h.reserved[5], 0x10);
        assert_eq!(h.reserved[7], 0x04);

        let raw = h.encode();
        assert_eq!(raw.len(), HANDSHAKE_LEN, "握手必须正好 68 字节");
        assert_eq!(raw[0], 19, "pstrlen 是协议串长度");
        assert_eq!(&raw[1..20], PROTOCOL.as_bytes());

        let back = Handshake::decode(&raw).unwrap();
        assert_eq!(back, h);
        assert!(back.supports_extended() && back.supports_fast());

        // 不置位时两个能力都应为 false（确认不是恒真）
        let plain = Handshake {
            info_hash: ih,
            peer_id: pid,
            reserved: [0u8; 8],
        };
        assert!(!plain.supports_extended());
        assert!(!plain.supports_fast());
        // 只有扩展位、没有 fast 位
        let mut r = [0u8; 8];
        r[5] = 0x10;
        assert!(Handshake { info_hash: ih, peer_id: pid, reserved: r }.supports_extended());
        assert!(!Handshake { info_hash: ih, peer_id: pid, reserved: r }.supports_fast());
    }

    #[test]
    fn handshake_rejects_bad_pstrlen_and_protocol() {
        let mut raw = Handshake::new([0u8; 20], [0u8; 20]).encode();
        // 错的 pstrlen：必须报 Protocol，而不是把后面的字节错位解读
        raw[0] = 18;
        let e = Handshake::decode(&raw).unwrap_err();
        assert!(matches!(e, MediaError::Protocol(_)), "{e:?}");

        // pstrlen 对但协议串不对（比如对面是别的协议）
        let mut raw = Handshake::new([0u8; 20], [0u8; 20]).encode();
        raw[1] = b'X';
        let e = Handshake::decode(&raw).unwrap_err();
        assert!(matches!(e, MediaError::Protocol(_)), "{e:?}");
    }

    #[test]
    fn handshake_truncated_reports_need_more() {
        let raw = Handshake::new([0u8; 20], [0u8; 20]).encode();
        for cut in 0..HANDSHAKE_LEN {
            let e = Handshake::decode(&raw[..cut]).unwrap_err();
            assert!(e.is_need_more(), "截断到 {cut} 字节应当是 NeedMore，实得 {e:?}");
        }
        assert!(Handshake::decode(&raw).is_ok());
        // 多出来的字节不影响解析（后面可能已经跟了别的消息）
        let mut longer = raw.clone();
        longer.extend_from_slice(b"junk");
        assert!(Handshake::decode(&longer).is_ok());
    }

    #[test]
    fn peer_id_has_prefix_and_is_20_bytes() {
        let a = generate_peer_id();
        let b = generate_peer_id();
        assert_eq!(a.len(), 20);
        assert_eq!(&a[..8], b"-ME0001-");
        // 两次生成必须不同（后 12 字节随机）
        assert_ne!(a, b, "peer_id 后 12 字节必须随机");
    }

    fn all_variants() -> Vec<Message> {
        let mut bf = Bitfield::new(16);
        bf.set(0, true);
        bf.set(15, true);
        vec![
            Message::KeepAlive,
            Message::Choke,
            Message::Unchoke,
            Message::Interested,
            Message::NotInterested,
            Message::Have(12345),
            Message::Bitfield(bf),
            Message::Request {
                index: 1,
                begin: 16384,
                length: BLOCK_SIZE,
            },
            Message::Piece {
                index: 2,
                begin: 0,
                data: vec![0xAB; BLOCK_SIZE as usize],
            },
            Message::Cancel {
                index: 3,
                begin: 32768,
                length: BLOCK_SIZE,
            },
            Message::Port(6881),
            Message::HaveAll,
            Message::HaveNone,
            Message::Suggest(7),
            Message::RejectRequest {
                index: 4,
                begin: 0,
                length: BLOCK_SIZE,
            },
            Message::AllowedFast(9),
            Message::Extended {
                id: 1,
                payload: b"hello".to_vec(),
            },
            Message::Unknown {
                id: 200,
                payload: vec![1, 2, 3],
            },
        ]
    }

    #[test]
    fn message_roundtrip_all_variants() {
        for m in all_variants() {
            let enc = m.encode();
            let (dec, used) = Message::decode(&enc).unwrap();
            assert_eq!(used, enc.len(), "{m:?} 的消耗长度必须等于整包长度");
            assert_eq!(dec, m, "编解码往返必须一致: {m:?}");
            assert_eq!(dec.id(), m.id());
        }
    }

    #[test]
    fn keepalive_is_four_zero_bytes_without_id() {
        assert_eq!(Message::KeepAlive.encode(), vec![0, 0, 0, 0]);
        let (m, used) = Message::decode(&[0, 0, 0, 0]).unwrap();
        assert_eq!(m, Message::KeepAlive);
        assert_eq!(used, 4);
        assert_eq!(m.id(), KEEPALIVE_ID);
        // KeepAlive 之后跟着别的消息时，消耗长度只算 4
        let mut buf = vec![0, 0, 0, 0];
        buf.extend_from_slice(&Message::Unchoke.encode());
        let (m, used) = Message::decode(&buf).unwrap();
        assert_eq!(m, Message::KeepAlive);
        assert_eq!(used, 4);
    }

    #[test]
    fn truncated_message_reports_need_more() {
        // 空 buffer：连长度前缀都不够
        let e = Message::decode(&[]).unwrap_err();
        assert!(e.is_need_more());
        let e = Message::decode(&[0, 0, 0]).unwrap_err();
        assert!(e.is_need_more());

        // 只有长度前缀、没有正文
        let e = Message::decode(&[0, 0, 0, 5]).unwrap_err();
        match e {
            MediaError::NeedMore { need, have } => {
                assert_eq!(need, 9);
                assert_eq!(have, 4);
            }
            other => panic!("应当是 NeedMore，实得 {other:?}"),
        }

        // 逐字节截断一条 Request（17 字节）
        let enc = Message::Request {
            index: 1,
            begin: 2,
            length: 3,
        }
        .encode();
        assert_eq!(enc.len(), 17);
        for cut in 0..enc.len() {
            let e = Message::decode(&enc[..cut]).unwrap_err();
            assert!(e.is_need_more(), "截断到 {cut} 字节应当是 NeedMore，实得 {e:?}");
        }
        // 多读一个字节就够
        assert!(Message::decode(&enc).is_ok());
    }

    #[test]
    fn oversized_length_prefix_is_rejected_not_allocated() {
        // 声明 1 MiB + 1 字节：必须报 Protocol，绝不能照着分配缓冲
        let buf = (MAX_MESSAGE_SIZE as u32 + 1).to_be_bytes();
        let e = Message::decode(&buf).unwrap_err();
        assert!(matches!(e, MediaError::Protocol(_)), "{e:?}");
        // 正好 1 MiB 是允许的（只是数据不够，所以是 NeedMore）
        let buf = (MAX_MESSAGE_SIZE as u32).to_be_bytes();
        assert!(Message::decode(&buf).unwrap_err().is_need_more());
    }

    #[test]
    fn malformed_payloads_are_protocol_errors() {
        // have 少了 1 字节（长度前缀说 4，但 id + 负载只有 4 字节 → 负载 3）
        let bad = frame(4, &[0, 0, 0]);
        assert!(matches!(
            Message::decode(&bad).unwrap_err(),
            MediaError::Protocol(_)
        ));
        // choke 带了负载
        let bad = frame(0, &[1]);
        assert!(matches!(
            Message::decode(&bad).unwrap_err(),
            MediaError::Protocol(_)
        ));
        // piece 负载不足 8 字节
        let bad = frame(7, &[0, 0, 0, 1]);
        assert!(matches!(
            Message::decode(&bad).unwrap_err(),
            MediaError::Protocol(_)
        ));
        // extended 缺扩展 id（只有消息 id 20，没有负载）
        let bad = frame(20, &[]);
        assert!(matches!(
            Message::decode(&bad).unwrap_err(),
            MediaError::Protocol(_)
        ));
    }

    #[test]
    fn unknown_message_id_is_kept_verbatim() {
        let enc = frame(200, b"whatever");
        let (m, used) = Message::decode(&enc).unwrap();
        assert_eq!(used, enc.len());
        assert_eq!(
            m,
            Message::Unknown {
                id: 200,
                payload: b"whatever".to_vec()
            }
        );
        // 往返不变
        assert_eq!(m.encode(), enc);
    }

    #[test]
    fn bitfield_message_spare_bits_need_the_real_piece_count() {
        // 3 片：线上是 1 字节，末 5 位是补齐位。置上第 0、2 片 → 0b1010_0000
        let mut bf = Bitfield::new(3);
        bf.set(0, true);
        bf.set(2, true);
        let raw = bf.to_bytes();
        assert_eq!(raw, vec![0xa0]);

        let enc = Message::Bitfield(bf).encode();
        let (m, used) = Message::decode(&enc).unwrap();
        assert_eq!(used, enc.len());
        let Message::Bitfield(wire_bf) = m else {
            panic!("应当是 Bitfield 消息");
        };
        // 线层不知道真实片数，按"字节数 × 8"暂存，字节原样保留
        assert_eq!(wire_bf.to_bytes(), raw);
        assert_eq!(wire_bf.num_pieces(), 8, "线层只能按字节数推片数");

        // 调用方拿真实片数重解析：补齐位被清掉，不会被当成 have
        let real = Bitfield::from_bytes(&wire_bf.to_bytes(), 3).unwrap();
        assert_eq!(real.count(), 2);
        assert!(real.get(0) && real.get(2));
        assert!(!real.get(3), "补齐位不能被当成 have");
        assert_eq!(real.num_pieces(), 3);
        assert_eq!(real.to_bytes(), vec![0xa0], "重编码也要抹掉补齐位");
    }

    #[test]
    fn empty_bitfield_message_is_fine() {
        // 对面一片都没有时可能发一个空位图
        let enc = frame(5, &[]);
        let (m, used) = Message::decode(&enc).unwrap();
        assert_eq!(used, enc.len());
        let Message::Bitfield(bf) = m else {
            panic!("应当是 Bitfield 消息");
        };
        assert_eq!(bf.count(), 0);
        assert_eq!(bf.num_pieces(), 0);
    }

    #[test]
    fn extended_handshake_roundtrip() {
        let hs = ExtendedHandshake {
            m: BTreeMap::from([
                ("ut_metadata".to_string(), 3u8),
                ("ut_pex".to_string(), 1u8),
            ]),
            metadata_size: Some(40000),
            reqq: Some(250),
            v: Some("me-bt 0.1".to_string()),
        };
        let enc = hs.encode();
        let back = ExtendedHandshake::parse(&enc).unwrap();
        assert_eq!(back, hs);
        assert_eq!(back.m["ut_metadata"], 3);
        assert_eq!(back.metadata_size, Some(40000));
        assert_eq!(back.reqq, Some(250));
        assert_eq!(back.v.as_deref(), Some("me-bt 0.1"));

        // 只有 m、别的字段缺省
        let hs = ExtendedHandshake {
            m: BTreeMap::from([("ut_metadata".to_string(), 2u8)]),
            ..Default::default()
        };
        let back = ExtendedHandshake::parse(&hs.encode()).unwrap();
        assert_eq!(back, hs);
        assert_eq!(back.metadata_size, None);
        assert_eq!(back.reqq, None);
        assert_eq!(back.v, None);
    }

    #[test]
    fn extended_handshake_ignores_garbage_fields() {
        use crate::bencode::BValue;
        let mut d = BTreeMap::new();
        // m 里混入非整数值、非 UTF-8 的键：都要被安静丢弃，不能整个解析失败
        let mut m = BTreeMap::new();
        m.insert(b"ut_metadata".to_vec(), BValue::Int(3));
        m.insert(b"broken".to_vec(), BValue::Bytes(b"not-an-int".to_vec()));
        m.insert(vec![0xff, 0xfe], BValue::Int(9));
        d.insert(b"m".to_vec(), BValue::Dict(m));
        d.insert(b"metadata_size".to_vec(), BValue::Int(-5)); // 负数钳成 0
        let back = ExtendedHandshake::parse(&BValue::Dict(d).encode()).unwrap();
        assert_eq!(back.m.len(), 1);
        assert_eq!(back.m["ut_metadata"], 3);
        assert_eq!(back.metadata_size, Some(0));
    }

    #[test]
    fn metadata_message_roundtrip() {
        assert_eq!(
            MetadataMessage::parse(&MetadataMessage::Request(3).encode()).unwrap(),
            MetadataMessage::Request(3)
        );
        assert_eq!(
            MetadataMessage::parse(&MetadataMessage::Reject(2).encode()).unwrap(),
            MetadataMessage::Reject(2)
        );

        let data = vec![0xCDu8; 16384];
        let dm = MetadataMessage::Data {
            piece: 1,
            total_size: 40000,
            data: data.clone(),
        };
        let enc = dm.encode();
        let back = MetadataMessage::parse(&enc).unwrap();
        assert_eq!(back, dm);
        // 关键：裸数据必须**接在 bencode 字典之后**，不能被塞进字典当字符串
        let (_, used) = crate::bencode::BValue::parse(&enc).unwrap();
        assert_eq!(&enc[used..], &data[..], "字典之后必须紧跟原始元信息字节");

        // 未知 msg_type / 缺字段
        assert!(MetadataMessage::parse(b"d8:msg_typei9e5:piecei0ee").is_err());
        assert!(MetadataMessage::parse(b"d5:piecei0ee").is_err());
        assert!(MetadataMessage::parse(b"d8:msg_typei0ee").is_err());
        // piece 为负
        assert!(MetadataMessage::parse(b"d8:msg_typei0e5:piecei-1ee").is_err());
    }

    #[test]
    fn metadata_piece_count_rounds_up() {
        assert_eq!(metadata_piece_count(0), 0);
        assert_eq!(metadata_piece_count(1), 1);
        assert_eq!(metadata_piece_count(METADATA_PIECE_SIZE), 1);
        assert_eq!(metadata_piece_count(METADATA_PIECE_SIZE + 1), 2);
        assert_eq!(metadata_piece_count(40000), 3, "40000 / 16384 向上取整是 3");
    }
}
