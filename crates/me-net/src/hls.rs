//! HLS（RFC 8216 + LL-HLS）清单解析与分片计划。
//!
//! 这一层只负责"把清单文本变成结构化的分片计划"：取流交给 [`crate::http`]，
//! 解密交给 [`crate::crypto`]。几条刻意的决定：
//!
//! 1. **时间一律整数微秒**。`#EXTINF:4.008,` 是规范里唯一一处小数秒，
//!    本模块在**解析的那一行**就把它换算成 `i64` 微秒，对外只暴露微秒整数。
//!    全引擎的时间运算都不碰浮点（见 me-core 的 crate 文档），这里若放行 f64，
//!    后面累加几百个分片就会攒出肉眼可见的音画偏移。
//! 2. **容错优先，但不静默产出错数据**。HLS 方言极多：`#EXTINF` 后面少逗号、
//!    `#EXT-X-KEY` 属性顺序乱掉、私有标签满天飞，这些一律照收；但**结构性错误**
//!    （缺 `#EXTM3U`、`#EXTINF` 后面没有 URI、媒体清单缺 `#EXT-X-TARGETDURATION`）
//!    必须报错 —— 静默当成"时长 0 的分片"只会让播放器拿着一份错的时长表跑到死。
//! 3. **状态型标签（KEY / MAP / DISCONTINUITY）在遇到分片时"快照"进该分片**。
//!    规范里它们作用于后续分片，解析时必须自己维护状态；否则会出现"密钥变了但
//!    每个分片都说自己是新密钥"这种错。
//! 4. **主清单与媒体清单自动判定**：出现 `#EXT-X-STREAM-INF` 或 `#EXT-X-MEDIA`
//!    就当主清单。只看 `#EXTM3U` 是分不出来的 —— 两者开头一模一样。
//! 5. **LL-HLS 的 `#EXT-X-PART` 归属于"它后面那个 `#EXTINF`"**。
//!    这是 LL-HLS 的排版约定：一个分片的部分分片写在它的 `#EXTINF` **之前**，
//!    于是清单末尾那些还没等到 `#EXTINF` 的部分分片，正是"正在生成、尚未完整发布"
//!    的那个分片 —— 它们收在 [`MediaPlaylist::pending_parts`] 里，
//!    低延迟起播要靠它，绝不能悄悄丢掉。

use me_core::error::{MediaError, Result};

use crate::crypto::{self, AesKey};
use crate::http::HttpClient;
use crate::url;

/// 一份 HLS 清单：要么是主清单（变体列表），要么是媒体清单（分片列表）。
#[derive(Debug, Clone, PartialEq)]
pub enum Playlist {
    Master(MasterPlaylist),
    Media(MediaPlaylist),
}

/// 主清单：一组可选变体（清晰度/码率）与全局属性。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MasterPlaylist {
    pub variants: Vec<Variant>,
    /// `#EXT-X-INDEPENDENT-SEGMENTS`：每个分片都能独立解码（可任意起播/切流）。
    pub independent_segments: bool,
    /// `#EXT-X-SESSION-DATA` 的 `(DATA-ID, 值)`。值取 `VALUE=`，没有则取绝对化后的 `URI=`。
    pub session_data: Vec<(String, String)>,
}

/// 一个变体（清晰度/码率）。
#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    /// **已解析成绝对 URL** 的媒体清单地址。
    pub uri: String,
    pub bandwidth: u64,
    pub average_bandwidth: Option<u64>,
    pub resolution: Option<(u32, u32)>,
    pub codecs: Option<String>,
    /// 帧率原样保留：规范允许 `25` 与 `30000/1001` 两种写法，
    /// 想算数值的调用方自己按 `/` 拆（本层不数值化，免得精度在这里丢一次）。
    pub frame_rate: Option<String>,
}

/// 媒体清单：分片计划。
#[derive(Debug, Clone, PartialEq)]
pub struct MediaPlaylist {
    pub version: Option<u32>,
    /// `#EXT-X-TARGETDURATION`（秒 → 微秒）。缺失时为 0。
    pub target_duration_us: i64,
    pub media_sequence: u64,
    pub discontinuity_sequence: u64,
    pub end_list: bool,
    /// 是否直播：没有 `#EXT-X-ENDLIST` 就是直播；`#EXT-X-PLAYLIST-TYPE:VOD` 视为点播。
    pub is_live: bool,
    pub segments: Vec<Segment>,
    /// 清单级 `#EXT-X-MAP`（第一个出现的那个）。
    pub init_map: Option<InitMap>,
    pub server_control: Option<ServerControl>,
    pub part_inf: Option<PartInf>,
    /// 已发布、但**所属分片还没在清单里完整出现**的部分分片（LL-HLS 里正在生成的那一个）。
    /// 它们没有可用的完整分片 URI，所以不能塞进 [`MediaPlaylist::segments`]。
    pub pending_parts: Vec<Part>,
}

impl MediaPlaylist {
    /// 已完成分片的时长之和（微秒）。直播清单会随时间增长，这里只是"当前这份"的总长。
    pub fn total_duration_us(&self) -> i64 {
        self.segments.iter().map(|s| s.duration_us).sum()
    }
}

/// 一个媒体分片。
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// 已解析成绝对 URL。
    pub uri: String,
    pub duration_us: i64,
    /// `#EXTINF` 逗号后的标题。
    pub title: Option<String>,
    pub byte_range: Option<ByteRange>,
    pub key: Option<KeyInfo>,
    /// 本分片生效的 `#EXT-X-MAP`（分片级覆盖清单级）。
    pub map: Option<InitMap>,
    /// 本分片之前是否有 `#EXT-X-DISCONTINUITY`。
    pub discontinuity: bool,
    /// 媒体序号 = `media_sequence + 下标`。IV 缺省时要靠它推导，必须固化在分片上。
    pub sequence: u64,
    /// LL-HLS 的部分分片（`#EXT-X-PART`），可为空。
    pub parts: Vec<Part>,
}

/// 字节区间。`end` 是**闭区间**的末字节偏移。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    pub start: u64,
    pub end: Option<u64>,
}

impl ByteRange {
    /// 闭区间长度；区间畸形（`end < start`）时为 `None`。
    pub fn len(&self) -> Option<u64> {
        self.end
            .and_then(|e| e.checked_sub(self.start))
            .map(|d| d + 1)
    }

    /// 是否为空区间（`end < start`，只可能来自畸形清单）。
    pub fn is_empty(&self) -> bool {
        matches!(self.end, Some(e) if e < self.start)
    }
}

/// `#EXT-X-KEY` 的加密方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyMethod {
    /// `METHOD=NONE`（或该分片无加密）。
    None,
    /// `METHOD=AES-128`：整段 AES-128-CBC + PKCS7。
    Aes128,
    /// `METHOD=SAMPLE-AES`：只加密样本，本版不支持（见 [`decrypt`]）。
    SampleAes,
}

/// `#EXT-X-KEY`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyInfo {
    pub method: KeyMethod,
    /// 密钥地址（已绝对化）。
    pub uri: Option<String>,
    /// IV。显式 IV 用清单里写的；没写时由解析器按分片序号的大端 16 字节补上
    /// （见 [`KeyInfo::for_sequence`]）—— 这样 `Segment.key` 拿到手就能直接解密。
    pub iv: Option<[u8; 16]>,
    /// `KEYFORMAT`；`None` 表示默认的 `identity`（即我们支持的那种）。
    pub key_format: Option<String>,
}

impl KeyInfo {
    /// 按分片序号固化 IV。
    ///
    /// RFC 8216 §5.2：`#EXT-X-KEY` 没写 IV 时，IV 就是**媒体序号的大端 16 字节**。
    /// 序号随分片而变，所以这一步必须在生成分片时做（快照），而不是等到解密时 ——
    /// 解密函数只拿得到一个 [`KeyInfo`]，它无从知道这是第几个分片。
    /// 用错 IV 不会报错、只会解出乱码，所以这个推导必须严格照规范来。
    pub fn for_sequence(&self, sequence: u64) -> KeyInfo {
        if self.method == KeyMethod::Aes128 && self.iv.is_none() {
            let mut iv = [0u8; 16];
            iv[8..].copy_from_slice(&sequence.to_be_bytes());
            return KeyInfo {
                iv: Some(iv),
                ..self.clone()
            };
        }
        self.clone()
    }
}

/// `#EXT-X-MAP`：初始化段（fMP4 的 `init` 分片）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InitMap {
    pub uri: String,
    pub byte_range: Option<ByteRange>,
}

/// `#EXT-X-PART`：LL-HLS 的部分分片。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Part {
    pub uri: String,
    pub duration_us: i64,
    pub byte_range: Option<ByteRange>,
    /// 该部分分片是否可独立解码（`INDEPENDENT=YES`）—— 低延迟起播只能从它开始。
    pub independent: bool,
    /// `GAP=YES`：这一片在源侧缺失，播放器应跳过而不是卡住。
    pub gap: bool,
}

/// `#EXT-X-PART-INF`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartInf {
    pub part_target_us: i64,
}

/// `#EXT-X-SERVER-CONTROL`：LL-HLS 的服务器行为协商。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ServerControl {
    /// 是否允许阻塞式重载（长轮询新分片）。
    pub can_block_reload: bool,
    pub hold_back_us: Option<i64>,
    pub part_hold_back_us: Option<i64>,
    /// 服务端承诺"这个时间之前的内容不会被移除"，据此才能安全做时间跳转。
    pub can_skip_until_us: Option<i64>,
    pub can_skip_dateranges: bool,
}

// ---------------------------------------------------------------------------
// 解析入口
// ---------------------------------------------------------------------------

/// 解析一份 HLS 清单，自动判定主/媒体清单。
///
/// `base_url` 是这份清单自己的 URL，用来把清单里的相对 URI 拼成绝对 URL。
///
/// ```
/// use me_net::hls::{self, Playlist};
///
/// let text = "#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXTINF:4.0,\nseg1.ts\n#EXT-X-ENDLIST\n";
/// let pl = match hls::parse(text, "http://cdn/live/index.m3u8").unwrap() {
///     Playlist::Media(m) => m,
///     Playlist::Master(_) => unreachable!("这份清单是媒体清单"),
/// };
/// assert_eq!(pl.segments[0].uri, "http://cdn/live/seg1.ts");
/// assert_eq!(pl.total_duration_us(), 4_000_000);
/// assert!(!pl.is_live);
/// ```
pub fn parse(text: &str, base_url: &str) -> Result<Playlist> {
    let mut saw_header = false;
    let mut is_master = false;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "#EXTM3U" {
            saw_header = true;
        } else if line.starts_with("#EXT-X-STREAM-INF:") || line.starts_with("#EXT-X-MEDIA:") {
            is_master = true;
            break;
        }
    }

    if !saw_header {
        return Err(MediaError::protocol(
            "不是 HLS 清单：缺少 #EXTM3U 首行（不少源会把 HTML 错误页当清单返回，必须早报错）",
        ));
    }

    if is_master {
        Ok(Playlist::Master(parse_master(text, base_url)?))
    } else {
        Ok(Playlist::Media(parse_media(text, base_url)?))
    }
}

fn parse_master(text: &str, base_url: &str) -> Result<MasterPlaylist> {
    let mut master = MasterPlaylist::default();
    // 上一行 `#EXT-X-STREAM-INF` 的属性，等着下一行 URI 来配对。
    let mut pending: Option<Vec<(String, String)>> = None;

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix('#') {
            let (tag, value) = match rest.split_once(':') {
                Some((t, v)) => (t, Some(v.trim())),
                None => (rest, None),
            };
            match tag {
                // `#EXT-X-MEDIA` 也可能带可取流的 URI（纯音频源的备用音轨），
                // 所以顺手纳入候选。
                "EXT-X-MEDIA" => {
                    // 悬着的 STREAM-INF 被别的标签打断就作废：它已经失去了自己的 URI，
                    // 拿标签行当 URI 会凭空造出一条指向 "…/EXT-X-MEDIA:…" 的变体。
                    pending = None;
                    record_media(&mut master, &parse_attributes(value.unwrap_or("")), base_url);
                }
                "EXT-X-STREAM-INF" => {
                    pending = Some(parse_attributes(value.unwrap_or("")));
                }
                "EXT-X-INDEPENDENT-SEGMENTS" => master.independent_segments = true,
                "EXT-X-SESSION-DATA" => {
                    let attrs = parse_attributes(value.unwrap_or(""));
                    let id = attr_get(&attrs, "DATA-ID").unwrap_or_default();
                    let value = attr_get(&attrs, "VALUE")
                        .or_else(|| attr_get(&attrs, "URI").map(|u| url::resolve(base_url, &u)));
                    if let (false, Some(v)) = (id.is_empty(), value) {
                        master.session_data.push((id, v));
                    }
                }
                _ => {}
            }
            continue;
        }

        // URI 行：只有紧跟 `#EXT-X-STREAM-INF` 才有意义。
        if let Some(attrs) = pending.take() {
            record_stream_inf(&mut master, &attrs, line, base_url)?;
        }
        // 没有配对 STREAM-INF 的裸 URI 行在主清单里没有意义，忽略。
        // （有些私有标签会留下裸行，为它拒绝整份清单不值得。）
    }

    if master.variants.is_empty() {
        return Err(MediaError::protocol(
            "主清单里没有任何可用的 #EXT-X-STREAM-INF 变体",
        ));
    }
    Ok(master)
}

fn record_stream_inf(
    master: &mut MasterPlaylist,
    attrs: &[(String, String)],
    line: &str,
    base_url: &str,
) -> Result<()> {
    let bandwidth = attr_get(attrs, "BANDWIDTH")
        .and_then(|v| v.parse::<u64>().ok())
        .ok_or_else(|| {
            MediaError::protocol(format!("#EXT-X-STREAM-INF 缺少合法的 BANDWIDTH: {line}"))
        })?;
    master.variants.push(Variant {
        uri: url::resolve(base_url, line),
        bandwidth,
        average_bandwidth: attr_get(attrs, "AVERAGE-BANDWIDTH").and_then(|v| v.parse().ok()),
        resolution: attr_get(attrs, "RESOLUTION").and_then(|v| parse_resolution(&v)),
        codecs: attr_get(attrs, "CODECS"),
        frame_rate: attr_get(attrs, "FRAME-RATE"),
    });
    Ok(())
}

/// 把 `#EXT-X-MEDIA` 的 `URI` 也纳入变体候选。
///
/// 为什么这么做：纯音频源（只有 `TYPE=AUDIO` 的备用音轨、没有任何 STREAM-INF）
/// 在体育直播里很常见，主清单里唯一的可取流地址就是 MEDIA 的 URI。
/// 带宽未知时按 0 记，选流时自然排到最后。
fn record_media(master: &mut MasterPlaylist, attrs: &[(String, String)], base_url: &str) {
    if let Some(uri) = attr_get(attrs, "URI") {
        master.variants.push(Variant {
            uri: url::resolve(base_url, &uri),
            bandwidth: attr_get(attrs, "BANDWIDTH")
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0),
            average_bandwidth: None,
            resolution: None,
            codecs: None,
            frame_rate: None,
        });
    }
}

fn parse_media(text: &str, base_url: &str) -> Result<MediaPlaylist> {
    let mut pl = MediaPlaylist {
        version: None,
        target_duration_us: 0,
        media_sequence: 0,
        discontinuity_sequence: 0,
        end_list: false,
        is_live: true,
        segments: Vec::new(),
        init_map: None,
        server_control: None,
        part_inf: None,
        pending_parts: Vec::new(),
    };

    // `#EXTINF` / `#EXT-X-BYTERANGE` 等着下一行 URI 来配对。
    let mut pending_inf: Option<(i64, Option<String>)> = None;
    let mut pending_range: Option<String> = None;
    // 状态型标签：作用于"后续分片"，遇到分片时快照进 Segment。
    let mut current_key: Option<KeyInfo> = None;
    let mut current_map: Option<InitMap> = None;
    let mut pending_discontinuity = false;
    let mut playlist_type_vod = false;
    let mut iframe_only = false;
    // 已发布但还没等到所属分片的部分分片（见模块文档第 5 条）。
    let mut pending_parts: Vec<Part> = Vec::new();

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix('#') {
            let (tag, value) = match rest.split_once(':') {
                Some((t, v)) => (t, Some(v.trim())),
                None => (rest, None),
            };
            match tag {
                "EXT-X-VERSION" => pl.version = value.and_then(|v| v.parse().ok()),
                "EXT-X-TARGETDURATION" => {
                    let secs = value
                        .and_then(|v| v.parse::<f64>().ok())
                        .and_then(secs_to_us)
                        .ok_or_else(|| {
                            MediaError::protocol(format!(
                                "#EXT-X-TARGETDURATION 不是合法秒数: {}",
                                value.unwrap_or("")
                            ))
                        })?;
                    pl.target_duration_us = secs;
                }
                "EXT-X-MEDIA-SEQUENCE" => {
                    pl.media_sequence = value.and_then(|v| v.parse().ok()).unwrap_or(0);
                }
                "EXT-X-DISCONTINUITY-SEQUENCE" => {
                    pl.discontinuity_sequence = value.and_then(|v| v.parse().ok()).unwrap_or(0);
                }
                "EXTINF" => {
                    let v = value
                        .ok_or_else(|| MediaError::protocol("#EXTINF 缺少时长"))?;
                    // 逗号后的标题原样保留（有些源拿它放时间码，对不上号时很有用）
                    let (dur, title) = match v.split_once(',') {
                        Some((d, t)) => (d.trim(), Some(t.to_string())),
                        None => (v.trim(), None),
                    };
                    let us = dur
                        .parse::<f64>()
                        .ok()
                        .and_then(secs_to_us)
                        .ok_or_else(|| {
                            MediaError::protocol(format!("#EXTINF 的时长不是合法秒数: {dur}"))
                        })?;
                    pending_inf = Some((us, title.filter(|t| !t.is_empty())));
                }
                "EXT-X-BYTERANGE" => pending_range = Some(value.unwrap_or("").to_string()),
                "EXT-X-DISCONTINUITY" => pending_discontinuity = true,
                "EXT-X-KEY" => {
                    let attrs = parse_attributes(value.unwrap_or(""));
                    current_key = parse_key(&attrs, base_url)?;
                }
                "EXT-X-MAP" => {
                    let attrs = parse_attributes(value.unwrap_or(""));
                    current_map =
                        parse_map(&attrs, base_url, prev_range_end(&pl.segments))?;
                    if pl.init_map.is_none() {
                        pl.init_map = current_map.clone();
                    }
                }
                "EXT-X-ENDLIST" => pl.end_list = true,
                "EXT-X-PLAYLIST-TYPE" => {
                    if value.map(|v| v.eq_ignore_ascii_case("VOD")).unwrap_or(false) {
                        playlist_type_vod = true;
                    }
                }
                "EXT-X-I-FRAMES-ONLY" => iframe_only = true,
                "EXT-X-SERVER-CONTROL" => {
                    let attrs = parse_attributes(value.unwrap_or(""));
                    pl.server_control = Some(ServerControl {
                        can_block_reload: attr_yes(&attrs, "CAN-BLOCK-RELOAD"),
                        hold_back_us: attr_secs_us(&attrs, "HOLD-BACK"),
                        part_hold_back_us: attr_secs_us(&attrs, "PART-HOLD-BACK"),
                        can_skip_until_us: attr_secs_us(&attrs, "CAN-SKIP-UNTIL"),
                        can_skip_dateranges: attr_yes(&attrs, "CAN-SKIP-DATERANGES"),
                    });
                }
                "EXT-X-PART-INF" => {
                    let attrs = parse_attributes(value.unwrap_or(""));
                    // PART-TARGET 只是提示（"部分分片大概多长"），
                    // 解析不出来就不设，不影响分片计划。
                    if let Some(t) = attr_secs_us(&attrs, "PART-TARGET") {
                        pl.part_inf = Some(PartInf {
                            part_target_us: t,
                        });
                    }
                }
                "EXT-X-PART" => {
                    let attrs = parse_attributes(value.unwrap_or(""));
                    let uri = attr_get(&attrs, "URI")
                        .ok_or_else(|| MediaError::protocol("#EXT-X-PART 缺少 URI"))?;
                    let duration_us = attr_secs_us(&attrs, "DURATION").ok_or_else(|| {
                        MediaError::protocol(format!("#EXT-X-PART 缺少合法的 DURATION: {uri}"))
                    })?;
                    // 规范里 PART 的 BYTERANGE 省略起始偏移时，起点是"上一个分片
                    // **或**上一个部分分片"的结尾之后，所以先看部分分片、再看完整分片。
                    let prev = pending_parts
                        .last()
                        .and_then(|p| p.byte_range)
                        .and_then(|r| r.end)
                        .or_else(|| prev_range_end(&pl.segments));
                    let byte_range = match attr_get(&attrs, "BYTERANGE") {
                        Some(v) => Some(parse_byte_range(&v, prev, None)?),
                        None => None,
                    };
                    pending_parts.push(Part {
                        uri: url::resolve(base_url, &uri),
                        duration_us,
                        byte_range,
                        independent: attr_yes(&attrs, "INDEPENDENT"),
                        gap: attr_yes(&attrs, "GAP"),
                    });
                }
                // 其余标签解析出来也不影响播放计划，认得但有意忽略：
                // START / PROGRAM-DATE-TIME 是绝对时间（播放器只需要相对时长），
                // DATERANGE / PRELOAD-HINT / RENDITION-REPORT / SKIP / ALLOW-CACHE
                // 都是提示类信息。
                _ => {}
            }
            continue;
        }

        // ---- URI 行：一个分片 ----
        let (duration_us, title) = pending_inf.take().ok_or_else(|| {
            MediaError::protocol(format!(
                "分片 URI 之前没有 #EXTINF（结构性错误，会让总时长表整体错位）: {line}"
            ))
        })?;
        let byte_range = match pending_range.take() {
            Some(v) => Some(parse_byte_range(&v, prev_range_end(&pl.segments), None)?),
            None => None,
        };
        let sequence = pl.media_sequence + pl.segments.len() as u64;
        pl.segments.push(Segment {
            uri: url::resolve(base_url, line),
            duration_us,
            title,
            byte_range,
            // 缺省 IV 在这里按序号固化，解密时就不用再关心"这是第几片"。
            key: current_key.as_ref().map(|k| k.for_sequence(sequence)),
            map: current_map.clone(),
            discontinuity: pending_discontinuity,
            sequence,
            // 部分分片写在所属分片的 `#EXTINF` 之前，到这里正好收口。
            parts: std::mem::take(&mut pending_parts),
        });
        pending_discontinuity = false;
    }

    if let Some((dur, _)) = pending_inf {
        // 悬空的 #EXTINF：规范里它描述的是"下一个分片"，没有下一个就是清单被截断了。
        return Err(MediaError::protocol(format!(
            "#EXTINF（{dur}us）后面没有分片 URI，清单可能被截断"
        )));
    }
    pl.pending_parts = pending_parts;

    if pl.target_duration_us <= 0 {
        return Err(MediaError::protocol(
            "媒体清单缺少 #EXT-X-TARGETDURATION（播放器要靠它决定预缓冲深度）",
        ));
    }

    // I-FRAMES-ONLY 清单里只有关键帧：它的"分片时长"是帧间隔而不是播放时长，
    // 拿它算总时长会得到完全错误的结果，所以明确拒绝而不是当成普通媒体清单。
    if iframe_only {
        return Err(MediaError::Unsupported(
            "I-FRAMES-ONLY 清单只含关键帧，本版不把它当作可播放媒体清单（仅可用于拖拽预览）"
                .to_string(),
        ));
    }

    pl.is_live = !pl.end_list && !playlist_type_vod;
    Ok(pl)
}

// ---------------------------------------------------------------------------
// 标签细节
// ---------------------------------------------------------------------------

fn parse_key(attrs: &[(String, String)], base_url: &str) -> Result<Option<KeyInfo>> {
    let method_raw = attr_get(attrs, "METHOD").unwrap_or_else(|| "NONE".to_string());
    let method = match method_raw.to_ascii_uppercase().as_str() {
        "NONE" => return Ok(None),
        "AES-128" => KeyMethod::Aes128,
        "SAMPLE-AES" | "SAMPLE-AES-CTR" => KeyMethod::SampleAes,
        other => {
            return Err(MediaError::Unsupported(format!(
                "不认识的 #EXT-X-KEY METHOD={other}"
            )))
        }
    };
    let uri = attr_get(attrs, "URI").map(|u| url::resolve(base_url, &u));
    if uri.is_none() {
        return Err(MediaError::protocol(
            "#EXT-X-KEY 声明了加密却没有 URI（拿不到密钥就取不了流）",
        ));
    }
    Ok(Some(KeyInfo {
        method,
        uri,
        iv: attr_get(attrs, "IV").map(|v| parse_iv(&v)).transpose()?,
        key_format: attr_get(attrs, "KEYFORMAT").filter(|f| !f.eq_ignore_ascii_case("identity")),
    }))
}

fn parse_map(
    attrs: &[(String, String)],
    base_url: &str,
    prev_end: Option<u64>,
) -> Result<Option<InitMap>> {
    let Some(uri) = attr_get(attrs, "URI") else {
        return Ok(None);
    };
    let byte_range = match attr_get(attrs, "BYTERANGE") {
        // 初始化段省略起始偏移时按"接着上一段"理解；它前面什么都没有就按文件开头，
        // 因为 init 段本来就在文件最前面（这和普通分片不同，后者缺偏移是清单坏了）。
        Some(v) => Some(parse_byte_range(&v, prev_end, Some(0))?),
        None => None,
    };
    Ok(Some(InitMap {
        uri: url::resolve(base_url, &uri),
        byte_range,
    }))
}

/// 解析 `#EXT-X-BYTERANGE:len[@start]`。
///
/// `start` 缺省时**按规范累加到上一段结尾之后**（RFC 8216 §4.3.2.2）：
/// 省略起始偏移的意思就是"接着上一段继续"。不累加会让整份清单的分片都指向同一个
/// 偏移，播出来是同一段画面反复放 —— 这种错很难从现象反推，所以这里必须对。
///
/// `missing_start` 是"既没有 `@`、前面也没有上一段"时的兜底起点（`None` = 报错）。
fn parse_byte_range(s: &str, prev_end: Option<u64>, missing_start: Option<u64>) -> Result<ByteRange> {
    let s = s.trim();
    let (len_s, start_s) = match s.split_once('@') {
        Some((a, b)) => (a.trim(), Some(b.trim())),
        None => (s, None),
    };
    let len: u64 = len_s
        .parse()
        .map_err(|_| MediaError::protocol(format!("#EXT-X-BYTERANGE 的长度非法: {s}")))?;
    if len == 0 {
        return Err(MediaError::protocol(format!(
            "#EXT-X-BYTERANGE 长度为 0: {s}"
        )));
    }
    let start = match start_s {
        Some(v) => v
            .parse()
            .map_err(|_| MediaError::protocol(format!("#EXT-X-BYTERANGE 的起始偏移非法: {s}")))?,
        None => match prev_end {
            Some(e) => e + 1,
            None => missing_start.ok_or_else(|| {
                MediaError::protocol(format!(
                    "#EXT-X-BYTERANGE 省略了起始偏移，但它前面没有分片可续: {s}"
                ))
            })?,
        },
    };
    let end = start
        .checked_add(len - 1)
        .ok_or_else(|| MediaError::protocol(format!("#EXT-X-BYTERANGE 区间溢出: {s}")))?;
    Ok(ByteRange {
        start,
        end: Some(end),
    })
}

/// 上一段的结束偏移（用于 BYTERANGE 累加）。
fn prev_range_end(segments: &[Segment]) -> Option<u64> {
    segments.last().and_then(|s| s.byte_range).and_then(|r| r.end)
}

/// 解析 `0x...` 形式的 128 位 IV。
///
/// 容忍前导零被省略（有些编码器写 `0x1`），按左补零处理；但**超过 32 个 hex 位
/// 一律报错** —— 那说明这根本不是 IV，猜下去只会解出乱码。
fn parse_iv(s: &str) -> Result<[u8; 16]> {
    let t = s.trim();
    let hex = t
        .strip_prefix("0x")
        .or_else(|| t.strip_prefix("0X"))
        .ok_or_else(|| MediaError::protocol(format!("#EXT-X-KEY 的 IV 必须以 0x 开头: {t}")))?;
    if hex.is_empty() || hex.len() > 32 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(MediaError::protocol(format!(
            "#EXT-X-KEY 的 IV 非法（应为 0x + 最多 32 个 hex 位）: {t}"
        )));
    }
    let padded = format!("{hex:0>32}");
    let mut out = [0u8; 16];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&padded[i * 2..i * 2 + 2], 16)
            .map_err(|_| MediaError::protocol(format!("IV 解析失败: {t}")))?;
    }
    Ok(out)
}

fn parse_resolution(s: &str) -> Option<(u32, u32)> {
    let (w, h) = s.trim().split_once(['x', 'X'])?;
    Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
}

/// 小数秒 → 整数微秒。
///
/// 这是本模块唯一接触浮点的地方，而且**立刻收口**：规范里的 `#EXTINF` 就是小数写法，
/// 用 f64 读进来之后马上变成整数微秒，此后全引擎的时间运算都不再出现浮点。
/// 四舍五入到最近的微秒，避免 `4.008` 因为二进制表示变成 `4007999`。
/// 非有限值或负数返回 `None`（由调用方决定报错还是忽略），绝不静默当成 0。
fn secs_to_us(secs: f64) -> Option<i64> {
    if !secs.is_finite() || secs < 0.0 {
        return None;
    }
    Some((secs * 1_000_000.0).round() as i64)
}

// ---------------------------------------------------------------------------
// 属性行
// ---------------------------------------------------------------------------

/// 解析 `KEY=VALUE` 形式的属性列表（`#EXT-X-STREAM-INF:` / `#EXT-X-KEY:` 等）。
///
/// 必须处理三件事，缺一个都会踩坑：
/// 1. **引号里的逗号不是分隔符**。`CODECS="avc1.4d401f,mp4a.40.2"` 是最常见的形态，
///    用 `split(',')` 会把它切成两半，于是 codecs 只剩前半段、后半段变成垃圾属性。
/// 2. **引号本身要去掉**，否则拿去拼 URL 会带上引号。
/// 3. **值两侧空白去掉**，`BANDWIDTH= 1000` 这种写法也认。
fn parse_attributes(s: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut key = String::new();
    let mut value = String::new();
    let mut in_quotes = false;
    let mut in_value = false;

    for c in s.chars() {
        match c {
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                push_attr(&mut out, &key, &value);
                key.clear();
                value.clear();
                in_value = false;
            }
            '=' if !in_quotes && !in_value => in_value = true,
            _ => {
                if in_value {
                    value.push(c);
                } else {
                    key.push(c);
                }
            }
        }
    }
    push_attr(&mut out, &key, &value);
    out
}

fn push_attr(out: &mut Vec<(String, String)>, key: &str, value: &str) {
    let k = key.trim();
    if k.is_empty() {
        return;
    }
    out.push((k.to_string(), value.trim().to_string()));
}

/// 按名字取属性值。属性名规范里全大写，这里大小写不敏感地找，
/// 免得遇到小写方言就被当成"没有这个属性"。
fn attr_get(attrs: &[(String, String)], name: &str) -> Option<String> {
    attrs
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.clone())
}

/// 属性里的小数秒 → 微秒。
fn attr_secs_us(attrs: &[(String, String)], name: &str) -> Option<i64> {
    attr_get(attrs, name)
        .and_then(|s| s.parse::<f64>().ok())
        .and_then(secs_to_us)
}

/// `YES` 判定。缺失按 `NO`（规范默认值）。
fn attr_yes(attrs: &[(String, String)], name: &str) -> bool {
    attr_get(attrs, name)
        .map(|v| v.eq_ignore_ascii_case("YES"))
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// 解密
// ---------------------------------------------------------------------------

/// 按 `#EXT-X-KEY` 解密一个分片的字节。
///
/// - `None`：原样返回（不加密的源占绝大多数）；
/// - `Aes128`：拉密钥 → AES-128-CBC 解密 + PKCS7 去填充。IV 取 [`KeyInfo::iv`]；
///   解析器已经在生成分片时按媒体序号固化过缺省 IV（[`KeyInfo::for_sequence`]），
///   所以正常路径下这里一定有值。手工构造的 `KeyInfo` 若没写 IV，按全零处理
///   （全零正是"序号 0"的取值），不会静默产出一份看着合理的错明文；
/// - `SampleAes`：直接报 `Unsupported`。SAMPLE-AES 只加密样本（TS 里是加密的
///   载荷片段，fMP4 里是 `cbcs` 模式），与整段 CBC 完全是两回事；
///   硬按 CBC 解会得到"能跑但花屏"的结果，比报错难排查得多。
pub fn decrypt(key: &KeyInfo, client: &HttpClient, data: Vec<u8>) -> Result<Vec<u8>> {
    match key.method {
        KeyMethod::None => Ok(data),
        KeyMethod::SampleAes => Err(MediaError::Unsupported(
            "SAMPLE-AES 只加密样本（TS 的载荷片段 / fMP4 的 cbcs），与整段 AES-128-CBC 不是一回事，本版不实现"
                .to_string(),
        )),
        KeyMethod::Aes128 => {
            // 非 identity 的 KEYFORMAT 说明密钥本身被 DRM 包了一层（Widevine / PlayReady
            // 的 `METHOD=AES-128` 都会这么写）。此时 URI 拿回来的不是裸密钥，
            // 硬按裸密钥解会得到"能跑但花屏"的结果，所以直接拒绝。
            if let Some(fmt) = &key.key_format {
                return Err(MediaError::Unsupported(format!(
                    "KEYFORMAT={fmt} 的 AES-128 密钥是被 DRM 包装过的，本版只支持 identity"
                )));
            }
            let uri = key
                .uri
                .as_ref()
                .ok_or_else(|| MediaError::protocol("#EXT-X-KEY 没有 URI，无法取密钥"))?;
            let body = client.get(uri)?;
            let raw = crypto::decode_key_body(&body, 16)?;
            let aes = AesKey::from_bytes(&raw)?;
            let iv = key.iv.unwrap_or([0u8; 16]);
            crypto::aes_cbc_decrypt(&aes, &iv, &data)
        }
    }
}

// ---------------------------------------------------------------------------
// 选流
// ---------------------------------------------------------------------------

/// 在不超过上限的前提下选**带宽最高**的变体。
///
/// 没有任何变体满足上限时**返回最低带宽的那个而不是 `None`**：
/// 返回 `None` 调用方只能报"没有可用流"，可实际上有流可放（只是画质/码率超了
/// 用户设定）—— 能放总比报错强，真实播放器也是这么做的。
pub fn pick_variant(
    master: &MasterPlaylist,
    max_height: Option<u32>,
    max_bandwidth: Option<u64>,
) -> Option<&Variant> {
    let fits = |v: &Variant| {
        max_height
            .map(|h| v.resolution.map(|(_, vh)| vh <= h).unwrap_or(true))
            .unwrap_or(true)
            && max_bandwidth.map(|b| v.bandwidth <= b).unwrap_or(true)
    };

    let mut best: Option<&Variant> = None;
    let mut lowest: Option<&Variant> = None;
    for v in &master.variants {
        if fits(v) && best.map(|b| v.bandwidth > b.bandwidth).unwrap_or(true) {
            best = Some(v);
        }
        if lowest.map(|l| v.bandwidth < l.bandwidth).unwrap_or(true) {
            lowest = Some(v);
        }
    }
    best.or(lowest)
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::HttpConfig;

    const BASE: &str = "http://cdn.example.com/live/stream/index.m3u8";

    fn media(text: &str) -> MediaPlaylist {
        match parse(text, BASE).expect("清单应当解析成功") {
            Playlist::Media(m) => m,
            Playlist::Master(_) => panic!("这份清单应当被判定为媒体清单"),
        }
    }

    fn master(text: &str) -> MasterPlaylist {
        match parse(text, BASE).expect("清单应当解析成功") {
            Playlist::Master(m) => m,
            Playlist::Media(_) => panic!("这份清单应当被判定为主清单"),
        }
    }

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
            .collect()
    }

    fn arr16(s: &str) -> [u8; 16] {
        let mut a = [0u8; 16];
        a.copy_from_slice(&hex(s));
        a
    }

    // ---- 相对 URI ----

    #[test]
    fn relative_uris_are_resolved_against_the_playlist_url() {
        let pl = media(
            "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:4\n\
             #EXTINF:4.0,\nseg1.ts\n\
             #EXTINF:4.0,\n../a/b.ts\n\
             #EXTINF:4.0,\n?token=x\n\
             #EXT-X-ENDLIST\n",
        );
        assert_eq!(pl.segments.len(), 3);
        assert_eq!(
            pl.segments[0].uri,
            "http://cdn.example.com/live/stream/seg1.ts"
        );
        // `..` 要真的回到上一级
        assert_eq!(pl.segments[1].uri, "http://cdn.example.com/live/a/b.ts");
        // 只有查询串的相对引用继承清单路径（不能丢掉 index.m3u8）
        assert_eq!(
            pl.segments[2].uri,
            "http://cdn.example.com/live/stream/index.m3u8?token=x"
        );
        assert_eq!(pl.target_duration_us, 4_000_000);
        assert_eq!(pl.total_duration_us(), 12_000_000);
    }

    #[test]
    fn absolute_uri_is_kept_as_is() {
        let pl = media(
            "#EXTM3U\n#EXT-X-TARGETDURATION:6\n#EXTINF:6,\nhttp://other.h/s.ts\n#EXT-X-ENDLIST\n",
        );
        assert_eq!(pl.segments[0].uri, "http://other.h/s.ts");
    }

    // ---- BYTERANGE ----

    #[test]
    fn byterange_accumulates_when_start_is_omitted() {
        let pl = media(
            "#EXTM3U\n#EXT-X-VERSION:4\n#EXT-X-TARGETDURATION:10\n\
             #EXT-X-BYTERANGE:1000@0\n#EXTINF:10,\nall.ts\n\
             #EXT-X-BYTERANGE:2000\n#EXTINF:10,\nall.ts\n\
             #EXT-X-BYTERANGE:500\n#EXTINF:10,\nall.ts\n\
             #EXT-X-ENDLIST\n",
        );
        let ranges: Vec<_> = pl.segments.iter().map(|s| s.byte_range.unwrap()).collect();
        assert_eq!(ranges[0], ByteRange { start: 0, end: Some(999) });
        // 第二段从上一段结尾之后开始：1000..=2999
        assert_eq!(ranges[1], ByteRange { start: 1000, end: Some(2999) });
        assert_eq!(ranges[2], ByteRange { start: 3000, end: Some(3499) });
        assert_eq!(ranges[1].len(), Some(2000));
        assert!(!ranges[1].is_empty());
    }

    #[test]
    fn byterange_without_start_and_without_previous_segment_is_an_error() {
        let e = parse(
            "#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXT-X-BYTERANGE:1000\n#EXTINF:10,\na.ts\n#EXT-X-ENDLIST\n",
            BASE,
        )
        .unwrap_err();
        assert!(e.to_string().contains("起始偏移"), "实际: {e}");
    }

    #[test]
    fn byterange_length_zero_is_rejected() {
        let e = parse(
            "#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXT-X-BYTERANGE:0@0\n#EXTINF:10,\na.ts\n#EXT-X-ENDLIST\n",
            BASE,
        )
        .unwrap_err();
        assert!(e.to_string().contains("长度为 0"), "实际: {e}");
    }

    // ---- 属性行 ----

    #[test]
    fn key_attributes_allow_commas_inside_quotes() {
        let pl = media(
            "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:10\n\
             #EXT-X-KEY:METHOD=AES-128,URI=\"http://cdn.example.com/key.bin?a=1,b=2\",\
             IV=0x000102030405060708090a0b0c0d0e0f,KEYFORMAT=\"identity\"\n\
             #EXTINF:10,\nseg0.ts\n#EXT-X-ENDLIST\n",
        );
        let key = pl.segments[0].key.as_ref().expect("分片应当带上密钥信息");
        assert_eq!(key.method, KeyMethod::Aes128);
        // 引号里的逗号不能被当成分隔符，引号本身也要去掉
        assert_eq!(key.uri.as_deref(), Some("http://cdn.example.com/key.bin?a=1,b=2"));
        assert_eq!(key.iv, Some(arr16("000102030405060708090a0b0c0d0e0f")));
        // identity 是默认值，不该留着
        assert_eq!(key.key_format, None);
    }

    #[test]
    fn key_method_none_clears_encryption_and_default_iv_uses_sequence() {
        let pl = media(
            "#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXT-X-MEDIA-SEQUENCE:7\n\
             #EXT-X-KEY:METHOD=AES-128,URI=\"key.bin\"\n\
             #EXTINF:10,\na0.ts\n\
             #EXTINF:10,\na1.ts\n\
             #EXT-X-KEY:METHOD=NONE\n\
             #EXTINF:10,\na2.ts\n#EXT-X-ENDLIST\n",
        );
        // 第 7 片的缺省 IV = 序号 7 的大端 16 字节
        let k0 = pl.segments[0].key.as_ref().unwrap();
        assert_eq!(k0.uri.as_deref(), Some("http://cdn.example.com/live/stream/key.bin"));
        assert_eq!(k0.iv, Some([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7]));
        assert_eq!(pl.segments[0].sequence, 7);
        assert_eq!(pl.segments[1].sequence, 8);
        assert_eq!(
            pl.segments[1].key.as_ref().unwrap().iv,
            Some([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8])
        );
        // METHOD=NONE 之后不再加密
        assert!(pl.segments[2].key.is_none());
    }

    #[test]
    fn sample_aes_is_recognized_but_decrypt_refuses_it() {
        let pl = media(
            "#EXTM3U\n#EXT-X-TARGETDURATION:10\n\
             #EXT-X-KEY:METHOD=SAMPLE-AES,URI=\"skd://key\",KEYFORMAT=\"com.apple.streamingkeydelivery\"\n\
             #EXTINF:10,\na.ts\n#EXT-X-ENDLIST\n",
        );
        let key = pl.segments[0].key.as_ref().unwrap();
        assert_eq!(key.method, KeyMethod::SampleAes);
        assert_eq!(
            key.key_format.as_deref(),
            Some("com.apple.streamingkeydelivery")
        );

        let client = HttpClient::new(HttpConfig::default()).unwrap();
        let e = decrypt(key, &client, vec![0u8; 16]).unwrap_err();
        assert!(matches!(e, MediaError::Unsupported(_)), "实际: {e}");
    }

    #[test]
    fn aes128_with_drm_keyformat_is_refused() {
        let pl = media(
            "#EXTM3U\n#EXT-X-TARGETDURATION:10\n\
             #EXT-X-KEY:METHOD=AES-128,URI=\"skd://key\",KEYFORMAT=\"com.widevine\"\n\
             #EXTINF:10,\na.ts\n#EXT-X-ENDLIST\n",
        );
        let key = pl.segments[0].key.as_ref().unwrap();
        assert_eq!(key.key_format.as_deref(), Some("com.widevine"));
        let client = HttpClient::new(HttpConfig::default()).unwrap();
        let e = decrypt(key, &client, vec![0u8; 16]).unwrap_err();
        assert!(matches!(e, MediaError::Unsupported(_)), "实际: {e}");
    }

    #[test]
    fn unknown_key_method_is_rejected() {
        let e = parse(
            "#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXT-X-KEY:METHOD=AES-256,URI=\"k\"\n#EXTINF:10,\na.ts\n#EXT-X-ENDLIST\n",
            BASE,
        )
        .unwrap_err();
        assert!(matches!(e, MediaError::Unsupported(_)), "实际: {e}");
    }

    #[test]
    fn quoted_comma_survives_in_codecs() {
        let m = master(
            "#EXTM3U\n\
             #EXT-X-STREAM-INF:BANDWIDTH=1280000,RESOLUTION=640x360,CODECS=\"avc1.4d401f,mp4a.40.2\",FRAME-RATE=25\n\
             low/index.m3u8\n",
        );
        assert_eq!(m.variants[0].codecs.as_deref(), Some("avc1.4d401f,mp4a.40.2"));
        assert_eq!(m.variants[0].resolution, Some((640, 360)));
        assert_eq!(m.variants[0].frame_rate.as_deref(), Some("25"));
    }

    // ---- 直播/点播判定 ----

    #[test]
    fn live_and_vod_detection() {
        // 没有 ENDLIST 就是直播
        let live = media("#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXTINF:4,\na.ts\n");
        assert!(live.is_live);
        assert!(!live.end_list);

        let vod = media(
            "#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXTINF:4,\na.ts\n#EXT-X-ENDLIST\n",
        );
        assert!(!vod.is_live);
        assert!(vod.end_list);

        // 没有 ENDLIST 但声明了 PLAYLIST-TYPE:VOD，也算点播
        let typed = media(
            "#EXTM3U\n#EXT-X-PLAYLIST-TYPE:VOD\n#EXT-X-TARGETDURATION:4\n#EXTINF:4,\na.ts\n",
        );
        assert!(!typed.is_live);
        assert!(!typed.end_list);

        // EVENT 不是点播
        let event = media(
            "#EXTM3U\n#EXT-X-PLAYLIST-TYPE:EVENT\n#EXT-X-TARGETDURATION:4\n#EXTINF:4,\na.ts\n",
        );
        assert!(event.is_live);
    }

    // ---- LL-HLS ----

    #[test]
    fn ll_hls_parts_attach_to_the_segment_that_follows_them() {
        let pl = media(
            "#EXTM3U\n#EXT-X-VERSION:9\n#EXT-X-TARGETDURATION:4\n\
             #EXT-X-PART-INF:PART-TARGET=1.0\n\
             #EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,PART-HOLD-BACK=3.0,CAN-SKIP-UNTIL=12.0\n\
             #EXT-X-MEDIA-SEQUENCE:266\n\
             #EXTINF:4.000,\nfileSequence266.mp4\n\
             #EXT-X-PART:DURATION=1.000,URI=\"part267.0.mp4\",INDEPENDENT=YES\n\
             #EXT-X-PART:DURATION=1.000,URI=\"part267.1.mp4\",GAP=YES\n\
             #EXTINF:4.000,\nfileSequence267.mp4\n\
             #EXT-X-PART:DURATION=1.000,URI=\"part268.0.mp4\",INDEPENDENT=YES\n\
             #EXT-X-PRELOAD-HINT:TYPE=PART,URI=\"part268.1.mp4\"\n",
        );

        assert_eq!(pl.part_inf, Some(PartInf { part_target_us: 1_000_000 }));
        let sc = pl.server_control.expect("SERVER-CONTROL 应当被解析");
        assert!(sc.can_block_reload);
        assert_eq!(sc.part_hold_back_us, Some(3_000_000));
        assert_eq!(sc.can_skip_until_us, Some(12_000_000));

        // 第一个分片没有部分分片，第二个有两个
        assert!(pl.segments[0].parts.is_empty());
        assert_eq!(pl.segments[1].parts.len(), 2);
        assert_eq!(pl.segments[1].parts[0].duration_us, 1_000_000);
        assert!(pl.segments[1].parts[0].independent);
        assert!(pl.segments[1].parts[1].gap);
        assert_eq!(
            pl.segments[1].parts[0].uri,
            "http://cdn.example.com/live/stream/part267.0.mp4"
        );
        // 清单末尾那个还没完整发布的分片收在 pending_parts 里，不能丢
        assert_eq!(pl.pending_parts.len(), 1);
        assert_eq!(
            pl.pending_parts[0].uri,
            "http://cdn.example.com/live/stream/part268.0.mp4"
        );
        // 它不算进已完成分片的总时长
        assert_eq!(pl.total_duration_us(), 8_000_000);
    }

    #[test]
    fn part_byterange_accumulates_within_the_segment() {
        let pl = media(
            "#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXT-X-MEDIA-SEQUENCE:1\n\
             #EXT-X-PART:DURATION=1,URI=\"s.ts\",BYTERANGE=\"100@0\"\n\
             #EXT-X-PART:DURATION=1,URI=\"s.ts\",BYTERANGE=\"200\"\n\
             #EXTINF:4,\ns.ts\n#EXT-X-ENDLIST\n",
        );
        let parts = &pl.segments[0].parts;
        assert_eq!(parts[0].byte_range, Some(ByteRange { start: 0, end: Some(99) }));
        assert_eq!(parts[1].byte_range, Some(ByteRange { start: 100, end: Some(299) }));
    }

    // ---- EXT-X-MAP / 不连续 ----

    #[test]
    fn map_key_and_discontinuity_are_snapshotted_into_segments() {
        let pl = media(
            "#EXTM3U\n#EXT-X-VERSION:7\n#EXT-X-TARGETDURATION:4\n\
             #EXT-X-MAP:URI=\"init.mp4\",BYTERANGE=\"720@0\"\n\
             #EXTINF:4,\na0.m4s\n\
             #EXT-X-DISCONTINUITY\n\
             #EXT-X-MAP:URI=\"init2.mp4\"\n\
             #EXTINF:4,\na1.m4s\n#EXT-X-ENDLIST\n",
        );
        assert_eq!(
            pl.init_map.as_ref().map(|m| m.uri.as_str()),
            Some("http://cdn.example.com/live/stream/init.mp4")
        );
        assert_eq!(
            pl.init_map.as_ref().and_then(|m| m.byte_range),
            Some(ByteRange { start: 0, end: Some(719) })
        );
        assert_eq!(pl.segments[0].map.as_ref().unwrap().uri, pl.init_map.as_ref().unwrap().uri);
        assert!(!pl.segments[0].discontinuity);
        assert!(pl.segments[1].discontinuity);
        assert_eq!(
            pl.segments[1].map.as_ref().map(|m| m.uri.as_str()),
            Some("http://cdn.example.com/live/stream/init2.mp4")
        );
    }

    #[test]
    fn extinf_title_and_fractional_duration_become_integer_micros() {
        let pl = media(
            "#EXTM3U\n#EXT-X-TARGETDURATION:5\n\
             #EXTINF:4.008,title here\n a.ts\n\
             #EXTINF:4.00008,\nb.ts\n#EXT-X-ENDLIST\n",
        );
        assert_eq!(pl.segments[0].duration_us, 4_008_000);
        assert_eq!(pl.segments[0].title.as_deref(), Some("title here"));
        // 五万分之一秒的写法也要收成整数微秒，不能变成 4007999
        assert_eq!(pl.segments[1].duration_us, 4_000_080);
        assert_eq!(pl.segments[1].title, None);
    }

    // ---- 主清单 ----

    #[test]
    fn master_playlist_variants_and_selection() {
        let m = master(
            "#EXTM3U\n#EXT-X-INDEPENDENT-SEGMENTS\n\
             #EXT-X-SESSION-DATA:DATA-ID=\"com.example.title\",VALUE=\"示例片\"\n\
             #EXT-X-STREAM-INF:BANDWIDTH=800000,AVERAGE-BANDWIDTH=700000,RESOLUTION=426x240\n\
             240p/index.m3u8\n\
             #EXT-X-STREAM-INF:BANDWIDTH=2400000,RESOLUTION=1280x720,CODECS=\"avc1.640028\"\n\
             720p/index.m3u8\n\
             #EXT-X-STREAM-INF:BANDWIDTH=5000000,RESOLUTION=1920x1080\n\
             1080p/index.m3u8\n",
        );
        assert!(m.independent_segments);
        assert_eq!(m.variants.len(), 3);
        assert_eq!(m.session_data.len(), 1);
        assert_eq!(m.session_data[0].0, "com.example.title");
        assert_eq!(m.session_data[0].1, "示例片");
        assert_eq!(m.variants[1].uri, "http://cdn.example.com/live/stream/720p/index.m3u8");
        assert_eq!(m.variants[1].average_bandwidth, None);
        assert_eq!(m.variants[0].average_bandwidth, Some(700_000));

        // 不设上限：取带宽最高的
        assert_eq!(pick_variant(&m, None, None).unwrap().bandwidth, 5_000_000);
        // 限制高度 720：在 240p/720p 里取带宽最高的（720p）
        assert_eq!(pick_variant(&m, Some(720), None).unwrap().bandwidth, 2_400_000);
        // 再限制带宽：只能落回 240p
        assert_eq!(
            pick_variant(&m, Some(720), Some(1_000_000)).unwrap().bandwidth,
            800_000
        );
        // 上限谁都满足不了 → 退而取最低带宽的，而不是 None
        let picked = pick_variant(&m, Some(120), Some(1000)).unwrap();
        assert_eq!(picked.bandwidth, 800_000);
    }

    #[test]
    fn master_playlist_with_only_media_uri_still_yields_a_variant() {
        let m = master(
            "#EXTM3U\n#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"a\",NAME=\"中文\",URI=\"audio/index.m3u8\"\n",
        );
        assert_eq!(m.variants.len(), 1);
        assert_eq!(
            m.variants[0].uri,
            "http://cdn.example.com/live/stream/audio/index.m3u8"
        );
    }

    #[test]
    fn stream_inf_without_bandwidth_is_an_error() {
        let e = parse(
            "#EXTM3U\n#EXT-X-STREAM-INF:RESOLUTION=640x360\nlow.m3u8\n",
            BASE,
        )
        .unwrap_err();
        assert!(e.to_string().contains("BANDWIDTH"), "实际: {e}");
    }

    // ---- 坏清单 ----

    #[test]
    fn missing_extm3u_is_an_error() {
        let e = parse("<html>404 not found</html>", BASE).unwrap_err();
        assert!(e.to_string().contains("#EXTM3U"), "实际: {e}");
    }

    #[test]
    fn extinf_without_uri_is_an_error_not_a_panic() {
        // 悬空的 EXTINF（清单被截断）
        let e = parse(
            "#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXTINF:4,\n",
            BASE,
        )
        .unwrap_err();
        assert!(e.to_string().contains("没有分片 URI"), "实际: {e}");

        // 反过来：URI 行前面没有 EXTINF
        let e = parse(
            "#EXTM3U\n#EXT-X-TARGETDURATION:4\na.ts\n#EXT-X-ENDLIST\n",
            BASE,
        )
        .unwrap_err();
        assert!(e.to_string().contains("#EXTINF"), "实际: {e}");
    }

    #[test]
    fn media_playlist_without_target_duration_is_an_error() {
        let e = parse("#EXTM3U\n#EXTINF:4,\na.ts\n#EXT-X-ENDLIST\n", BASE).unwrap_err();
        assert!(e.to_string().contains("TARGETDURATION"), "实际: {e}");
    }

    #[test]
    fn iframe_only_playlist_is_refused_explicitly() {
        let e = parse(
            "#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXT-X-I-FRAMES-ONLY\n#EXTINF:4,\na.ts\n#EXT-X-ENDLIST\n",
            BASE,
        )
        .unwrap_err();
        assert!(matches!(e, MediaError::Unsupported(_)), "实际: {e}");
    }

    #[test]
    fn empty_master_playlist_is_an_error() {
        // 有 MEDIA 标签但没有任何可取流的 URI，也没有 STREAM-INF
        let e = parse(
            "#EXTM3U\n#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"a\",NAME=\"x\"\n",
            BASE,
        )
        .unwrap_err();
        assert!(e.to_string().contains("变体"), "实际: {e}");
    }

    // ---- 解密 ----

    #[test]
    fn decrypt_none_is_identity() {
        let client = HttpClient::new(HttpConfig::default()).unwrap();
        let key = KeyInfo {
            method: KeyMethod::None,
            uri: None,
            iv: None,
            key_format: None,
        };
        assert_eq!(decrypt(&key, &client, vec![1, 2, 3]).unwrap(), vec![1, 2, 3]);
    }

    /// NIST SP 800-38A F.2.1 的 CBC-AES128 密钥与 IV。
    const NIST_KEY: &str = "2b7e151628aed2a6abf7158809cf4f3c";
    const NIST_IV: &str = "000102030405060708090a0b0c0d0e0f";
    const NIST_PT0: &str = "6bc1bee22e409f96e93d7e117393172a";

    /// 用 [`crypto::AesKey::encrypt_block`] 手搓 CBC 加密（只用于造测试密文）。
    fn cbc_encrypt(key_hex: &str, iv_hex: &str, plain: &[u8]) -> Vec<u8> {
        let key = AesKey::from_bytes(&hex(key_hex)).unwrap();
        let mut prev = arr16(iv_hex);
        let mut out = Vec::with_capacity(plain.len());
        for chunk in plain.chunks_exact(16) {
            let mut block = [0u8; 16];
            for i in 0..16 {
                block[i] = chunk[i] ^ prev[i];
            }
            key.encrypt_block(&mut block);
            out.extend_from_slice(&block);
            prev = block;
        }
        out
    }

    /// PKCS7 填充（密文长度必须是 16 的倍数）。
    fn pkcs7_pad(mut data: Vec<u8>) -> Vec<u8> {
        let pad = 16 - (data.len() % 16);
        data.extend(std::iter::repeat_n(pad as u8, pad));
        data
    }

    /// 一个只服务一次请求的极简 HTTP 服务器，用来验证"拉密钥 → 解密"整条链路。
    /// 返回 `(url, JoinHandle)`；不引入任何测试框架依赖。
    fn serve_once(body: Vec<u8>) -> (String, std::thread::JoinHandle<()>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                let mut buf = [0u8; 1024];
                // 读到请求头结束就够了（我们只发 GET）
                let _ = sock.read(&mut buf);
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/octet-stream\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = sock.write_all(head.as_bytes());
                let _ = sock.write_all(&body);
                let _ = sock.flush();
            }
        });
        (format!("http://{addr}/key.bin"), handle)
    }

    #[test]
    fn aes128_decrypt_pulls_key_and_decrypts_real_vector() {
        // 明文用 NIST 的第一块；再按 PKCS7 补一整块 0x10，
        // 于是密文 = 官方向量第一块 + 填充块
        let plain = hex(NIST_PT0);
        let padded = pkcs7_pad(plain.clone());
        let cipher = cbc_encrypt(NIST_KEY, NIST_IV, &padded);
        // 前 16 字节必须等于 NIST F.2.1 给出的密文，确认我们造的是真向量
        assert_eq!(&cipher[..16], &hex("7649abac8119b246cee98e9b12e9197d")[..]);

        let (url, handle) = serve_once(hex(NIST_KEY));
        let client = HttpClient::new(HttpConfig {
            timeout_secs: 10,
            ..HttpConfig::default()
        })
        .unwrap();
        let key = KeyInfo {
            method: KeyMethod::Aes128,
            uri: Some(url),
            iv: Some(arr16(NIST_IV)),
            key_format: None,
        };
        let got = decrypt(&key, &client, cipher).expect("解密应当成功");
        assert_eq!(got, plain);
        handle.join().unwrap();
    }

    #[test]
    fn aes128_decrypt_derives_iv_from_segment_sequence_end_to_end() {
        // 走完整链路：清单里不写 IV → 解析器按序号固化 → 用该 IV 造的密文能解开
        let pl = media(
            "#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXT-X-MEDIA-SEQUENCE:9\n\
             #EXT-X-KEY:METHOD=AES-128,URI=\"key.bin\"\n\
             #EXTINF:4,\na.ts\n#EXT-X-ENDLIST\n",
        );
        let seg = &pl.segments[0];
        let key = seg.key.as_ref().unwrap();
        let iv = key.iv.unwrap();
        assert_eq!(iv, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9]);

        let plain = hex(NIST_PT0);
        let iv_hex: String = iv.iter().map(|b| format!("{b:02x}")).collect();
        let cipher = cbc_encrypt(NIST_KEY, &iv_hex, &pkcs7_pad(plain.clone()));

        let (url, handle) = serve_once(hex(NIST_KEY));
        let mut key = key.clone();
        key.uri = Some(url);
        let client = HttpClient::new(HttpConfig {
            timeout_secs: 10,
            ..HttpConfig::default()
        })
        .unwrap();
        assert_eq!(decrypt(&key, &client, cipher).unwrap(), plain);
        handle.join().unwrap();
    }

    #[test]
    fn aes128_decrypt_with_wrong_key_reports_padding_error() {
        let cipher = cbc_encrypt(NIST_KEY, NIST_IV, &pkcs7_pad(hex(NIST_PT0)));
        // 换一个密钥，PKCS7 去填充必然失败 —— 必须报错而不是给出乱码
        let (url, handle) = serve_once(hex("000102030405060708090a0b0c0d0e0f"));
        let client = HttpClient::new(HttpConfig {
            timeout_secs: 10,
            ..HttpConfig::default()
        })
        .unwrap();
        let key = KeyInfo {
            method: KeyMethod::Aes128,
            uri: Some(url),
            iv: Some(arr16(NIST_IV)),
            key_format: None,
        };
        let e = decrypt(&key, &client, cipher).unwrap_err();
        assert!(e.to_string().contains("去填充"), "实际: {e}");
        handle.join().unwrap();
    }

    // ---- 端到端 ----

    #[test]
    fn end_to_end_media_playlist_plan() {
        let text = "#EXTM3U\n\
                    #EXT-X-VERSION:6\n\
                    #EXT-X-TARGETDURATION:6\n\
                    #EXT-X-MEDIA-SEQUENCE:100\n\
                    #EXT-X-DISCONTINUITY-SEQUENCE:1\n\
                    #EXT-X-MAP:URI=\"init.mp4\"\n\
                    #EXT-X-PLAYLIST-TYPE:VOD\n\
                    #EXTINF:5.76,first\n\
                    seg100.m4s\n\
                    #EXTINF:5.76,\n\
                    seg101.m4s\n\
                    #EXTINF:5.76,\n\
                    seg102.m4s\n\
                    #EXTINF:2.5,last\n\
                    seg103.m4s\n\
                    #EXT-X-ENDLIST\n";
        let pl = media(text);
        assert_eq!(pl.version, Some(6));
        assert_eq!(pl.media_sequence, 100);
        assert_eq!(pl.discontinuity_sequence, 1);
        assert!(!pl.is_live);
        assert_eq!(pl.segments.len(), 4);
        assert_eq!(pl.segments[0].uri, "http://cdn.example.com/live/stream/seg100.m4s");
        assert_eq!(pl.segments[3].uri, "http://cdn.example.com/live/stream/seg103.m4s");
        assert_eq!(pl.segments[0].sequence, 100);
        assert_eq!(pl.segments[3].sequence, 103);
        // 5.76*3 + 2.5 = 19.78 秒
        assert_eq!(pl.total_duration_us(), 19_780_000);
        assert_eq!(pl.segments[3].title.as_deref(), Some("last"));
        assert!(pl.segments.iter().all(|s| s.map.is_some()));
    }
}
