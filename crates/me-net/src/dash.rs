//! DASH：MPD 解析与分片 URL 展开（XML 解析器见 [`crate::xml`]）。
//!
//! 这一层只做"从 MPD 文本算出分片地址与时间"，取流交给 [`crate::http`]。
//! 几条刻意的决定：
//!
//! 1. **两种寻址方式都要支持**：`SegmentTemplate`（用 `$Number$` / `$Time$` 模板算地址）
//!    与 `SegmentList`（把地址逐条列出来）。前者是直播与主流点播的默认形态，
//!    后者常见于老编码器与单文件切片，缺任何一个都会有一大批源放不了。
//! 2. **时间用整数刻度 + 时基**（`timescale`）。`SegmentTemplate` 的 `duration` /
//!    `presentationTimeOffset` / `SegmentTimeline` 的 `t`/`d` 全是**刻度**，
//!    本模块原样保留刻度，只在必须给"微秒"的地方（[`SegmentList::duration_us`]）
//!    才换算一次。用浮点秒做中间量会在长片子上累积误差（见 me-core 的 crate 文档）。
//! 3. **`SegmentTimeline` 的 `r` 必须真展开**。`r=-1` 表示"重复到 Period 结束"，
//!    这是直播 MPD 的常见写法；不展开就等于丢掉整段时间轴。
//!    但**推不出重复次数时按"不重复"处理**（保留该段自身明确的 `d`，绝不猜一个次数）：
//!    猜错会让后续分片的时间戳整体错位，比少几段更糟。理由见
//!    [`SegmentTemplate::expand_timeline`] 的文档。
//! 4. **URL 逐层继承**：`BaseURL` 与 `xml:base` 会在 MPD → Period → AdaptationSet →
//!    Representation 逐层覆盖，任何一层算错都会取到 404（甚至别的源的地址）。
//!    继承规则集中在 `apply_base` 一个函数里，不散落各处。
//! 5. **`type="dynamic"` 就是直播**，缺省是点播（规范默认 `static`）。

use me_core::error::{MediaError, Result};

use crate::url;
use crate::xml::{self, XmlNode};

// ---------------------------------------------------------------------------
// 数据模型
// ---------------------------------------------------------------------------

/// 一份 MPD。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mpd {
    /// `MPD@mediaPresentationDuration`（微秒）。
    pub duration_us: Option<i64>,
    pub periods: Vec<Period>,
    /// MPD 层的 `BaseURL`（已绝对化）。
    pub base_urls: Vec<String>,
    /// 是否直播：`type="dynamic"`。
    pub is_live: bool,
    /// `type` 属性是否为 `dynamic`（原样保留，便于上层区分 `dynamic` 与 `static`）。
    pub type_is_dynamic: bool,
    /// `MPD@minimumUpdatePeriod`（微秒）：直播时上层靠它决定多久重载一次。
    pub min_update_period_us: Option<i64>,
}

/// 一个 Period。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Period {
    pub id: Option<String>,
    /// `Period@start`（微秒）。缺省 0：规范允许省略（隐含接着上一个 Period）。
    pub start_us: i64,
    /// `Period@duration`（微秒）。
    pub duration_us: Option<i64>,
    pub base_urls: Vec<String>,
    pub adaptations: Vec<AdaptationSet>,
}

/// 一个 AdaptationSet（一组内容相同、编码不同的 Representation）。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AdaptationSet {
    pub id: Option<String>,
    /// `video` / `audio` / `text`。由 `contentType` 或 `mimeType` 推断。
    pub content_type: Option<String>,
    pub mime: Option<String>,
    pub codecs: Option<String>,
    pub lang: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub representations: Vec<Representation>,
    pub segment_template: Option<SegmentTemplate>,
    pub segment_list: Option<SegmentList>,
}

/// 一个 Representation（一种具体编码，最终取流就是按它取）。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Representation {
    pub id: String,
    pub bandwidth: u64,
    pub codecs: Option<String>,
    pub mime: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub audio_sampling_rate: Option<u32>,
    pub base_urls: Vec<String>,
    pub segment_template: Option<SegmentTemplate>,
    pub segment_list: Option<SegmentList>,
}

/// `SegmentTemplate`：用模板算分片地址。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SegmentTemplate {
    /// `media` 模板（`$Number$` / `$Time$` / `$RepresentationID$` / `$Bandwidth$`）。
    pub media: Option<String>,
    /// `initialization` 模板（不含 `$Number$` / `$Time$`，否则就是清单写错了）。
    pub initialization: Option<String>,
    /// `startNumber`，缺省 1（规范默认值，不是 0）。
    pub start_number: u64,
    /// 时基：每秒多少刻度。
    pub timescale: u64,
    /// 每个分片的时长（刻度）。与 `timeline` 二选一。
    pub duration: Option<u64>,
    /// `presentationTimeOffset`（刻度）。
    pub presentation_time_offset: u64,
    /// `SegmentTimeline`（与 `duration` 二选一）。
    pub timeline: Vec<TimelineEntry>,
}

/// `SegmentTimeline` 里的一个 `<S>`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimelineEntry {
    /// 起始时间（刻度）。缺省时按"上一段结尾"续。
    pub start: Option<u64>,
    /// 每段时长（刻度）。
    pub duration: u64,
    /// 重复次数：`0` = 只此一段；`n` = 再来 `n` 段；`-1` = 重复到 Period 结束。
    pub repeat: i64,
}

/// `SegmentList`：把分片地址逐条列出来。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SegmentList {
    /// `SegmentList@duration`（微秒）。规范里它是 `@duration` 属性（秒的小数写法），
    /// 有些清单把它写在 `<SegmentDuration>` 子元素里，两种都认。
    pub duration_us: Option<i64>,
    pub timescale: u64,
    pub urls: Vec<String>,
    pub init: Option<String>,
}

// ---------------------------------------------------------------------------
// URL 继承与时间解析
// ---------------------------------------------------------------------------

/// 把一层 `BaseURL` 作用到上一层的基准上。
///
/// `xml:base` 与 `BaseURL` 一起参与继承，顺序按"属性先于子元素"（属性描述的是
/// 这个元素自己的基准，子元素再在其上叠加）。
fn apply_base(prev: &str, node: &XmlNode, texts: &[String]) -> String {
    let mut base = prev.to_string();
    if let Some(xb) = node.attr("xml:base") {
        base = url::resolve(&base, xb);
    }
    for t in texts {
        base = url::resolve(&base, t);
    }
    base
}

/// 收集一个元素的 `BaseURL` 子元素文本（去空白、忽略空串）。
fn base_url_texts(node: &XmlNode) -> Vec<String> {
    node.children_named("BaseURL")
        .map(|c| c.text_trimmed().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

/// 解析 ISO 8601 时长（`PT1H2M3.5S`）成微秒。
///
/// 只认 DASH 里真实会出现的形态：`P[nD][T[nH][nM][nS]]`。`Y`/`M`（年月）在
/// 媒体时长里没有意义，直接报错而不是瞎算成 30 天 —— 静默算错会让整条时间轴偏移。
fn parse_duration_us(s: &str) -> Result<i64> {
    let t = s.trim();
    let rest = t
        .strip_prefix('P')
        .or_else(|| t.strip_prefix('p'))
        .ok_or_else(|| MediaError::protocol(format!("ISO 8601 时长必须以 P 开头: {t}")))?;

    let mut total_us: i64 = 0;
    let mut num = String::new();
    let mut in_time = false;
    let mut saw_any = false;

    for c in rest.chars() {
        match c {
            'T' | 't' => {
                if !num.is_empty() {
                    return Err(MediaError::protocol(format!("ISO 8601 时长里 T 的位置不对: {t}")));
                }
                in_time = true;
            }
            '0'..='9' | '.' | ',' => num.push(if c == ',' { '.' } else { c }),
            'D' | 'd' => {
                total_us += whole_days(&num, t)?;
                num.clear();
                saw_any = true;
            }
            'H' | 'h' if in_time => {
                total_us += scale_us(&num, 3_600_000_000, t)?;
                num.clear();
                saw_any = true;
            }
            'M' | 'm' if in_time => {
                total_us += scale_us(&num, 60_000_000, t)?;
                num.clear();
                saw_any = true;
            }
            'S' | 's' if in_time => {
                total_us += scale_us(&num, 1_000_000, t)?;
                num.clear();
                saw_any = true;
            }
            'Y' | 'y' | 'M' | 'm' => {
                return Err(MediaError::Unsupported(format!(
                    "ISO 8601 时长里的年/月无法换算成精确秒数: {t}"
                )))
            }
            other => {
                return Err(MediaError::protocol(format!(
                    "ISO 8601 时长里出现无法识别的内容 '{other}': {t}"
                )))
            }
        }
    }
    if !num.is_empty() {
        return Err(MediaError::protocol(format!(
            "ISO 8601 时长结尾多了数字（缺少单位）: {t}"
        )));
    }
    if !saw_any {
        return Err(MediaError::protocol(format!("ISO 8601 时长里没有任何时间量: {t}")));
    }
    Ok(total_us)
}

fn whole_days(num: &str, full: &str) -> Result<i64> {
    let d: f64 = num
        .parse()
        .map_err(|_| MediaError::protocol(format!("ISO 8601 时长里的天数非法: {full}")))?;
    Ok((d * 86_400_000_000.0).round() as i64)
}

/// `num`（小数秒形式的数值）× 单位微秒。这里只做一次浮点乘法，结果立刻取整，
/// 不会把浮点带进后续的时间运算。
fn scale_us(num: &str, unit_us: i64, full: &str) -> Result<i64> {
    if num.is_empty() {
        return Err(MediaError::protocol(format!(
            "ISO 8601 时长里缺少数值: {full}"
        )));
    }
    let v: f64 = num
        .parse()
        .map_err(|_| MediaError::protocol(format!("ISO 8601 时长里的数值非法: {full}")))?;
    Ok((v * unit_us as f64).round() as i64)
}

/// 解析属性里的整数（刻度、时基、带宽都用它）。
fn attr_u64(node: &XmlNode, name: &str) -> Option<u64> {
    node.attr(name).and_then(|v| v.trim().parse::<u64>().ok())
}

fn attr_u32(node: &XmlNode, name: &str) -> Option<u32> {
    node.attr(name).and_then(|v| v.trim().parse::<u32>().ok())
}

/// 由 `mimeType` / `contentType` / `codecs` 推断内容类型。
///
/// 三个信号按可靠性排序：
/// 1. `contentType` 是清单作者显式声明的，最可靠；
/// 2. `mimeType` 的前缀（`video/` `audio/` `text/`）；
/// 3. `codecs` 的首个编码串 —— 真实 MPD 里 `mimeType="application/mp4"` 极常见
///    （fMP4 音视频都是这个 mime），此时只能靠 `mp4a` / `avc1` 这类前缀来分；
///    最后才用"有没有宽高"兜底。
fn infer_content_type(
    declared: Option<&str>,
    mime: Option<&str>,
    codecs: Option<&str>,
    has_dimensions: bool,
) -> Option<String> {
    if let Some(c) = declared {
        let c = c.trim().to_ascii_lowercase();
        if !c.is_empty() {
            return Some(c);
        }
    }
    if let Some(m) = mime {
        let m = m.trim().to_ascii_lowercase();
        if m.starts_with("video/") {
            return Some("video".to_string());
        }
        if m.starts_with("audio/") {
            return Some("audio".to_string());
        }
        if m.starts_with("text/") || m.starts_with("application/ttml") {
            return Some("text".to_string());
        }
    }
    if let Some(cs) = codecs {
        let first = cs.split(',').next().unwrap_or("").trim().to_ascii_lowercase();
        if ["mp4a", "ac-3", "ec-3", "opus", "dts", "flac"]
            .iter()
            .any(|p| first.starts_with(p))
        {
            return Some("audio".to_string());
        }
        if ["avc", "hvc", "hev", "dvh", "vp", "av01"]
            .iter()
            .any(|p| first.starts_with(p))
        {
            return Some("video".to_string());
        }
        if ["stpp", "wvtt", "ttml"].iter().any(|p| first.starts_with(p)) {
            return Some("text".to_string());
        }
    }
    if has_dimensions {
        Some("video".to_string())
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// SegmentTemplate
// ---------------------------------------------------------------------------

impl SegmentTemplate {
    /// 展开分片地址与时间轴，返回 `(URL, 起始刻度, 时长刻度)`。
    ///
    /// 返回的 URL **可能是相对地址**（模板里本来就是相对写法），调用方要拿
    /// [`Representation::base_urls`] 的第一个元素去 [`crate::url::resolve`]；
    /// 模板里直接写了绝对地址时原样返回。
    ///
    /// 起始刻度是**媒体时间轴上的刻度**（`$Time$` 取的就是它）。
    /// `presentation_time_offset` 只用于把媒体时间换算成 Period 时间，
    /// 而这里给出的正是媒体时间，所以不做偏移 —— 上层若要 Period 时间，
    /// 减去 `presentation_time_offset` 即可（这也是规范里的定义）。
    ///
    /// `period_duration_ticks` 只在 `SegmentTimeline` 里出现 `r="-1"`、
    /// 或模板只给了 `duration`（要靠总时长算段数）时才需要。
    ///
    /// 两种寻址方式：
    /// - 有 `timeline`：按时间轴逐段展开，`$Number$` 从 `startNumber` 顺次递增
    ///   （`$Number$` 与 `$Time$` 同时出现时两者都要对）；
    /// - 只有 `duration`：从 0 开始按 `duration` 步进。
    ///
    /// 算不出任何一段（没有 `media`、没有 `timeline` 也没有 `duration`、
    /// 或 Period 时长未知）时返回**空表**，而不是编一个假的时长。
    pub fn media_urls(
        &self,
        rep_id: &str,
        bandwidth: u64,
        period_duration_ticks: Option<u64>,
    ) -> Vec<(String, u64, u64)> {
        let Some(media) = self.media.as_deref() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let mut number = self.start_number;

        if !self.timeline.is_empty() {
            for (start, duration) in self.expand_timeline(period_duration_ticks) {
                out.push((
                    fill_template(media, rep_id, bandwidth, Some(number), Some(start)),
                    start,
                    duration,
                ));
                number += 1;
            }
            return out;
        }

        let Some(dur) = self.duration else {
            // 既没有 timeline 又没有 duration：算不出任何一段。返回空表比编一个
            // 假的时长好 —— 上层看到空表就能明确报"这份清单没法播"。
            return Vec::new();
        };
        if dur == 0 {
            return Vec::new();
        }
        // 用 SegmentTemplate@duration 时必须知道总时长才能知道有几段。
        // Period 时长未知（直播常见）时按 0 段返回，等清单更新后再来。
        let Some(total) = period_duration_ticks else {
            return Vec::new();
        };
        let count = total / dur;
        for i in 0..count {
            let start = i * dur;
            out.push((
                fill_template(media, rep_id, bandwidth, Some(self.start_number + i), Some(start)),
                start,
                dur,
            ));
        }
        out
    }

    /// 初始化段地址（`$RepresentationID$` / `$Bandwidth$`；`$Number$` 与 `$Time$` 在这里没有意义）。
    pub fn initialization_url(&self, rep_id: &str, bandwidth: u64) -> Option<String> {
        let init = self.initialization.as_deref()?;
        Some(fill_template(init, rep_id, bandwidth, None, None))
    }

    /// 把 `SegmentTimeline` 展开成 `(起始刻度, 时长刻度)` 列表。
    ///
    /// `r="-1"` 表示"重复到 Period 结束"，只有在 Period 时长已知且 `d > 0` 时才能
    /// 推算次数；推不出来时按"不重复"处理 —— 这一段自身的 `d` 是明确的，
    /// 丢掉它反而丢信息，但**绝不猜一个次数**：猜错会让后面所有时间戳整体错位。
    pub fn expand_timeline(&self, period_duration_ticks: Option<u64>) -> Vec<(u64, u64)> {
        let mut out = Vec::new();
        let mut cursor: u64 = 0;
        for entry in &self.timeline {
            if entry.duration == 0 {
                // 时长为 0 的段推进不了时间轴，继续下去会死循环，直接停在这里。
                break;
            }
            // 显式 `t` 优先（规范允许它与上一段结尾不连续，比如中间有空洞）。
            let mut start = entry.start.unwrap_or(cursor);
            let reps = if entry.repeat < 0 {
                match period_duration_ticks {
                    Some(total) if total > start => (total - start) / entry.duration,
                    // Period 时长未知（或已经走到末尾）：推不出次数就按"不重复"处理。
                    // 这一段自身的 `d` 是明确的，丢掉它反而丢信息；
                    // 但绝不猜一个次数 —— 猜错会让后面所有时间戳整体错位。
                    _ => 1,
                }
            } else {
                entry.repeat as u64 + 1
            };
            for _ in 0..reps {
                out.push((start, entry.duration));
                start += entry.duration;
            }
            cursor = start;
        }
        out
    }
}

/// 展开 `$...$` 模板。
///
/// 支持 `$RepresentationID$`、`$Bandwidth$`、`$Number$`、`$Time$`，
/// 以及带格式的 `$Number%05d$` / `$Time%05d$`（DASH 的 `%0[width]d` 写法）。
/// 认不出来的 `$...$` 原样保留：宁可让上层看到明显的模板残留，
/// 也不静默替换成空串产出一条"看起来正常"的错 URL。
fn fill_template(
    template: &str,
    rep_id: &str,
    bandwidth: u64,
    number: Option<u64>,
    time: Option<u64>,
) -> String {
    let mut out = String::with_capacity(template.len() + 16);
    let bytes = template.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'$' {
            let start = i;
            while i < bytes.len() && bytes[i] != b'$' {
                i += 1;
            }
            out.push_str(&template[start..i]);
            continue;
        }
        // 找到成对的 `$`；`$$` 是规范里的转义写法，表示一个字面 `$`。
        let Some(close) = template[i + 1..].find('$').map(|p| i + 1 + p) else {
            out.push_str(&template[i..]);
            return out;
        };
        let inner = &template[i + 1..close];
        if inner.is_empty() {
            out.push('$');
            i = close + 1;
            continue;
        }
        match expand_variable(inner, rep_id, bandwidth, number, time) {
            Some(v) => out.push_str(&v),
            None => out.push_str(&template[i..close + 1]),
        }
        i = close + 1;
    }
    out
}

/// 展开单个 `$...$` 里的内容。`None` = 认不出来（原样保留）。
fn expand_variable(
    inner: &str,
    rep_id: &str,
    bandwidth: u64,
    number: Option<u64>,
    time: Option<u64>,
) -> Option<String> {
    // 先把 `%0Nd` 之类的格式说明符拆出来
    let (name, fmt) = match inner.split_once('%') {
        Some((n, f)) => (n, Some(f)),
        None => (inner, None),
    };
    match name {
        // ID 是字符串，带格式说明符对它没有意义
        "RepresentationID" => match fmt {
            None => Some(rep_id.to_string()),
            Some(_) => None,
        },
        "Bandwidth" => format_value(bandwidth, fmt),
        "Number" => format_value(number?, fmt),
        "Time" => format_value(time?, fmt),
        _ => None,
    }
}

/// 数值变量 → 字符串。`fmt` 为 `Some("05d")` 这类写法时按它补零。
fn format_value(value: u64, fmt: Option<&str>) -> Option<String> {
    match fmt {
        None => Some(value.to_string()),
        Some(f) => format_number(value, f),
    }
}

/// 按 `0Nd` 形式格式化数字（DASH 只允许这种）。
fn format_number(value: u64, fmt: &str) -> Option<String> {
    let digits = fmt.strip_prefix('0')?.strip_suffix('d')?;
    let width: usize = digits.parse().ok()?;
    // 上限是防御性的：畸形清单写 `$Number%099999999d$` 会让 format! 尝试分配
    // 几百 MB 的字符串，一个坏清单就能把进程拖死。
    if width > 32 {
        return None;
    }
    Some(format!("{value:0width$}"))
}

// ---------------------------------------------------------------------------
// 解析入口
// ---------------------------------------------------------------------------

/// 解析一份 MPD。
///
/// `base_url` 是 MPD 自己的 URL，用来把清单里的相对地址逐层拼成绝对地址。
///
/// ```
/// use me_net::{dash, url};
///
/// let text = r#"<MPD type="static" mediaPresentationDuration="PT4S"><Period>
///   <AdaptationSet contentType="video">
///     <SegmentTemplate timescale="1000" duration="2000" media="v/$Number$.m4s"/>
///     <Representation id="v" bandwidth="800000" height="360"/>
///   </AdaptationSet></Period></MPD>"#;
/// let mpd = dash::parse_mpd(text, "http://cdn/m.mpd").unwrap();
/// let period = mpd.first_period().unwrap();
/// let rep = dash::pick_representation(&period.adaptations, "video", None, None).unwrap();
/// let t = rep.segment_template.as_ref().unwrap();
/// let ticks = mpd.period_duration_ticks(period, t.timescale).unwrap();
/// let segs = t.media_urls(&rep.id, rep.bandwidth, Some(ticks));
/// assert_eq!(segs.len(), 2);
/// // 模板给的是相对地址，要拿 Representation 继承来的 BaseURL 拼成绝对地址
/// assert_eq!(url::resolve(&rep.base_urls[0], &segs[0].0), "http://cdn/v/1.m4s");
/// ```
pub fn parse_mpd(text: &str, base_url: &str) -> Result<Mpd> {
    let root = xml::parse(text)?;
    if root.name != "MPD" {
        return Err(MediaError::protocol(format!(
            "MPD 根元素必须是 <MPD>，实际是 <{}>",
            root.name
        )));
    }

    let type_is_dynamic = root
        .attr("type")
        .map(|t| t.trim().eq_ignore_ascii_case("dynamic"))
        .unwrap_or(false);

    let mut mpd = Mpd {
        duration_us: match root.attr("mediaPresentationDuration") {
            Some(v) => Some(parse_duration_us(v)?),
            None => None,
        },
        periods: Vec::new(),
        base_urls: Vec::new(),
        is_live: type_is_dynamic,
        type_is_dynamic,
        min_update_period_us: match root.attr("minimumUpdatePeriod") {
            Some(v) => Some(parse_duration_us(v)?),
            None => None,
        },
    };

    let mpd_base = apply_base(base_url, &root, &base_url_texts(&root));
    mpd.base_urls = vec![mpd_base.clone()];

    // Period 没有显式 @start 时隐含"接着上一个 Period 结束"，所以要顺序推进。
    let mut cursor_us: i64 = 0;
    for p in root.children_named("Period") {
        let start_us = match p.attr("start") {
            Some(v) => parse_duration_us(v)?,
            None => cursor_us,
        };
        let duration_us = match p.attr("duration") {
            Some(v) => Some(parse_duration_us(v)?),
            None => None,
        };
        cursor_us = start_us + duration_us.unwrap_or(0);

        let period_base = apply_base(&mpd_base, p, &base_url_texts(p));
        let mut period = Period {
            id: p.attr("id").map(|s| s.to_string()),
            start_us,
            duration_us,
            base_urls: vec![period_base.clone()],
            adaptations: Vec::new(),
        };
        for a in p.children_named("AdaptationSet") {
            period
                .adaptations
                .push(parse_adaptation_set(a, &period_base)?);
        }
        mpd.periods.push(period);
    }

    // 直播清单一个 Period 都没有是正常的（还没生成），点播则是坏清单。
    if mpd.periods.is_empty() && !type_is_dynamic {
        return Err(MediaError::protocol("MPD 里没有任何 <Period>"));
    }
    Ok(mpd)
}

fn parse_adaptation_set(node: &XmlNode, base: &str) -> Result<AdaptationSet> {
    let base = apply_base(base, node, &base_url_texts(node));
    let declared = node.attr("contentType");
    let mime = node.attr("mimeType").map(|s| s.to_string());
    // 尺寸可能在 AdaptationSet 上，也可能只在 Representation 上；两处都看一眼，
    // 否则纯音频集会被误判成视频。
    let has_dimensions = node.attr("height").is_some()
        || node.attr("width").is_some()
        || node
            .children_named("Representation")
            .any(|r| r.attr("height").is_some() || r.attr("width").is_some());

    let mut set = AdaptationSet {
        id: node.attr("id").map(|s| s.to_string()),
        content_type: infer_content_type(
            declared,
            mime.as_deref(),
            node.attr("codecs"),
            has_dimensions,
        ),
        mime,
        codecs: node.attr("codecs").map(|s| s.to_string()),
        lang: node.attr("lang").map(|s| s.to_string()),
        width: attr_u32(node, "width"),
        height: attr_u32(node, "height"),
        representations: Vec::new(),
        segment_template: parse_segment_template(node)?,
        segment_list: parse_segment_list(node, &base)?,
    };

    for r in node.children_named("Representation") {
        let rep_base = apply_base(&base, r, &base_url_texts(r));
        // 子层没写 SegmentTemplate 时继承 AdaptationSet 的（DASH 的常规写法：
        // 模板写在集上，每个 Representation 只给 id/bandwidth）。
        let template = parse_segment_template(r)?.or_else(|| set.segment_template.clone());
        let list = parse_segment_list(r, &rep_base)?.or_else(|| set.segment_list.clone());
        let rep_mime = r.attr("mimeType").map(|s| s.to_string()).or_else(|| set.mime.clone());
        let codecs = r.attr("codecs").map(|s| s.to_string()).or_else(|| set.codecs.clone());
        let rep = Representation {
            id: r.attr("id").map(|s| s.to_string()).unwrap_or_default(),
            bandwidth: attr_u64(r, "bandwidth").unwrap_or(0),
            codecs,
            mime: rep_mime,
            width: attr_u32(r, "width").or(set.width),
            height: attr_u32(r, "height").or(set.height),
            audio_sampling_rate: attr_u32(r, "audioSamplingRate")
                .or_else(|| attr_u32(node, "audioSamplingRate")),
            base_urls: vec![rep_base],
            segment_template: template,
            segment_list: list,
        };
        // 集里没写 content_type 时，用 Representation 的信息补一次推断。
        if set.content_type.is_none() {
            set.content_type = infer_content_type(
                None,
                rep.mime.as_deref(),
                rep.codecs.as_deref(),
                rep.height.is_some() || rep.width.is_some(),
            );
        }
        set.representations.push(rep);
    }

    Ok(set)
}

fn parse_segment_template(node: &XmlNode) -> Result<Option<SegmentTemplate>> {
    let Some(t) = node.child("SegmentTemplate") else {
        return Ok(None);
    };
    let mut timeline = Vec::new();
    if let Some(tl) = t.child("SegmentTimeline") {
        for s in tl.children_named("S") {
            let duration = attr_u64(s, "d").unwrap_or(0);
            // `r` 缺省是 0（只此一段）；解析不出来也按 0 处理，不影响这一段本身。
            let repeat = s.attr("r").and_then(|v| v.trim().parse::<i64>().ok()).unwrap_or(0);
            timeline.push(TimelineEntry {
                start: attr_u64(s, "t"),
                duration,
                repeat,
            });
        }
    }
    Ok(Some(SegmentTemplate {
        media: t.attr("media").map(|s| s.to_string()),
        initialization: t.attr("initialization").map(|s| s.to_string()),
        start_number: attr_u64(t, "startNumber").unwrap_or(1),
        timescale: attr_u64(t, "timescale").unwrap_or(1).max(1),
        duration: attr_u64(t, "duration"),
        presentation_time_offset: attr_u64(t, "presentationTimeOffset").unwrap_or(0),
        timeline,
    }))
}

fn parse_segment_list(node: &XmlNode, base: &str) -> Result<Option<SegmentList>> {
    let Some(l) = node.child("SegmentList") else {
        return Ok(None);
    };
    let timescale = attr_u64(l, "timescale").unwrap_or(1).max(1);
    let duration_us = match l.attr("duration") {
        Some(v) => Some(parse_duration_us(v)?),
        // 有些清单把每段时长写成 <SegmentDuration> 子元素（老编码器的写法）
        None => match l.child_text("SegmentDuration") {
            Some(v) => Some(parse_duration_us(v)?),
            None => None,
        },
    };
    let mut urls = Vec::new();
    for u in l.children_named("SegmentURL") {
        if let Some(media) = u.attr("media") {
            urls.push(url::resolve(base, media));
        }
    }
    let init = l
        .child("Initialization")
        .and_then(|i| i.attr("sourceURL"))
        .map(|s| url::resolve(base, s));
    Ok(Some(SegmentList {
        duration_us,
        timescale,
        urls,
        init,
    }))
}

impl SegmentList {
    /// 每段时长（刻度）；`duration_us` 未知时按 `timescale` 反推。
    pub fn segment_duration_ticks(&self) -> Option<u64> {
        self.duration_us
            .map(|us| (us as u128 * self.timescale as u128 / 1_000_000) as u64)
    }

    /// 把所有分片按时长串成 `(URL, 起始刻度, 时长刻度)`。
    pub fn media_urls(&self) -> Vec<(String, u64, u64)> {
        let Some(dur) = self.segment_duration_ticks() else {
            return Vec::new();
        };
        self.urls
            .iter()
            .enumerate()
            .map(|(i, u)| (u.clone(), i as u64 * dur, dur))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// 选流
// ---------------------------------------------------------------------------

/// 在给定内容类型（`video` / `audio` / `text`）里选一个 Representation。
///
/// 选择规则与 HLS 的 [`crate::hls::pick_variant`] 保持一致：
/// 不超过上限时取带宽最高者；没有任何一个满足上限时退而取最低带宽者
/// （能放总比报"没有可用流"强）。
pub fn pick_representation<'a>(
    adaptations: &'a [AdaptationSet],
    content_type: &str,
    max_height: Option<u32>,
    max_bandwidth: Option<u64>,
) -> Option<&'a Representation> {
    let mut best: Option<&Representation> = None;
    let mut lowest: Option<&Representation> = None;
    for set in adaptations {
        let want = content_type.to_ascii_lowercase();
        let matches = match set.content_type.as_deref() {
            Some(c) => c.eq_ignore_ascii_case(content_type),
            // 集上没标类型时按 mime 前缀再兜一次：
            // 否则"清单没写 contentType"就一个流都选不出来。
            None => set
                .mime
                .as_deref()
                .map(|m| m.to_ascii_lowercase().starts_with(&format!("{want}/")))
                .unwrap_or(false),
        };
        if !matches {
            continue;
        }
        for rep in &set.representations {
            let fits = max_height
                .map(|h| rep.height.map(|rh| rh <= h).unwrap_or(true))
                .unwrap_or(true)
                && max_bandwidth
                    .map(|b| rep.bandwidth <= b)
                    .unwrap_or(true);
            if fits && best.map(|b| rep.bandwidth > b.bandwidth).unwrap_or(true) {
                best = Some(rep);
            }
            if lowest.map(|l| rep.bandwidth < l.bandwidth).unwrap_or(true) {
                lowest = Some(rep);
            }
        }
    }
    best.or(lowest)
}

/// 取 MPD 里第一个 Period（点播清单通常只有一个）。
impl Mpd {
    pub fn first_period(&self) -> Option<&Period> {
        self.periods.first()
    }

    /// Period 的时长（换算成 `timescale` 刻度）。
    ///
    /// 给 [`SegmentTemplate::media_urls`] 用：`SegmentTemplate@duration` 寻址与
    /// `SegmentTimeline` 的 `r="-1"` 都要靠它才能算出段数。
    /// Period 没写 `@duration` 时退用 MPD 的 `mediaPresentationDuration`（点播常见写法）。
    pub fn period_duration_ticks(&self, period: &Period, timescale: u64) -> Option<u64> {
        let us = period.duration_us.or(self.duration_us)?;
        Some((us as u128 * timescale.max(1) as u128 / 1_000_000) as u64)
    }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = "http://origin.example.com/vod/manifest.mpd";

    fn rep_of<'a>(mpd: &'a Mpd, content_type: &str, id: &str) -> &'a Representation {
        mpd.periods[0]
            .adaptations
            .iter()
            .flat_map(|a| a.representations.iter())
            .find(|r| r.id == id)
            .unwrap_or_else(|| panic!("找不到 Representation {id}（{content_type}）"))
    }

    /// 一份形态真实的点播 MPD：模板写在 AdaptationSet 上、BaseURL 指向 CDN。
    const VOD_MPD: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" type="static"
     mediaPresentationDuration="PT30S" minBufferTime="PT2S">
  <BaseURL>http://cdn.example.com/vod/</BaseURL>
  <Period id="p0" start="PT0S" duration="PT30S">
    <AdaptationSet id="1" contentType="video" mimeType="video/mp4" codecs="avc1.640028">
      <SegmentTemplate timescale="1000" duration="2000" startNumber="1"
                       initialization="$RepresentationID$/init.mp4"
                       media="$RepresentationID$/seg-$Number%05d$.m4s"/>
      <Representation id="v0" bandwidth="800000" width="640" height="360"/>
      <Representation id="v1" bandwidth="2400000" width="1280" height="720"/>
    </AdaptationSet>
    <AdaptationSet id="2" contentType="audio" mimeType="audio/mp4" codecs="mp4a.40.2" lang="zh">
      <SegmentTemplate timescale="48000" duration="96000" startNumber="1"
                       initialization="a/init.mp4" media="a/seg-$Number$.m4s"/>
      <Representation id="a0" bandwidth="128000" audioSamplingRate="48000"/>
    </AdaptationSet>
  </Period>
</MPD>"#;

    // ---- 整体解析 ----

    #[test]
    fn parses_vod_mpd_with_base_url_inheritance() {
        let mpd = parse_mpd(VOD_MPD, BASE).unwrap();
        assert!(!mpd.is_live);
        assert!(!mpd.type_is_dynamic);
        assert_eq!(mpd.duration_us, Some(30_000_000));
        assert_eq!(mpd.base_urls, vec!["http://cdn.example.com/vod/".to_string()]);
        assert_eq!(mpd.periods.len(), 1);

        let p = &mpd.periods[0];
        assert_eq!(p.id.as_deref(), Some("p0"));
        assert_eq!(p.start_us, 0);
        assert_eq!(p.duration_us, Some(30_000_000));
        // Period 没有自己的 BaseURL，应当继承 MPD 的
        assert_eq!(p.base_urls, vec!["http://cdn.example.com/vod/".to_string()]);
        assert_eq!(p.adaptations.len(), 2);

        let v = &p.adaptations[0];
        assert_eq!(v.content_type.as_deref(), Some("video"));
        assert_eq!(v.mime.as_deref(), Some("video/mp4"));
        assert_eq!(v.codecs.as_deref(), Some("avc1.640028"));
        let a = &p.adaptations[1];
        assert_eq!(a.content_type.as_deref(), Some("audio"));
        assert_eq!(a.lang.as_deref(), Some("zh"));

        // Representation 继承了 AdaptationSet 的尺寸与模板
        let v0 = rep_of(&mpd, "video", "v0");
        assert_eq!(v0.height, Some(360));
        assert_eq!(v0.bandwidth, 800_000);
        let a0 = rep_of(&mpd, "audio", "a0");
        assert_eq!(a0.audio_sampling_rate, Some(48_000));
        assert_eq!(a0.width, None);
    }

    #[test]
    fn segment_template_number_addressing() {
        let mpd = parse_mpd(VOD_MPD, BASE).unwrap();
        let p = &mpd.periods[0];
        let v1 = rep_of(&mpd, "video", "v1");
        let t = v1.segment_template.as_ref().unwrap();
        assert_eq!(t.timescale, 1000);
        assert_eq!(t.duration, Some(2000));
        assert_eq!(t.start_number, 1);

        let ticks = mpd.period_duration_ticks(p, t.timescale).unwrap();
        assert_eq!(ticks, 30_000);
        let urls = t.media_urls(&v1.id, v1.bandwidth, Some(ticks));
        // 30 秒 / 2 秒 = 15 段
        assert_eq!(urls.len(), 15);
        assert_eq!(urls[0].0, "v1/seg-00001.m4s");
        assert_eq!(urls[0].1, 0);
        assert_eq!(urls[0].2, 2000);
        assert_eq!(urls[14].0, "v1/seg-00015.m4s");
        assert_eq!(urls[14].1, 28_000);

        // 拼上继承来的 BaseURL 就是最终可取流的绝对地址
        let abs = url::resolve(&v1.base_urls[0], &urls[0].0);
        assert_eq!(abs, "http://cdn.example.com/vod/v1/seg-00001.m4s");
        assert_eq!(
            t.initialization_url(&v1.id, v1.bandwidth).unwrap(),
            "v1/init.mp4"
        );
    }

    #[test]
    fn segment_template_without_period_duration_yields_no_segments() {
        let mpd = parse_mpd(
            r#"<MPD type="dynamic"><Period>
                 <AdaptationSet contentType="video">
                   <SegmentTemplate timescale="1000" duration="2000" media="$Number$.m4s"/>
                   <Representation id="v" bandwidth="1"/>
                 </AdaptationSet>
               </Period></MPD>"#,
            BASE,
        )
        .unwrap();
        let v = &mpd.periods[0].adaptations[0].representations[0];
        let t = v.segment_template.as_ref().unwrap();
        // 直播清单没有总时长，段数算不出来：返回空表，而不是编一个假的
        assert!(t.media_urls(&v.id, v.bandwidth, None).is_empty());
    }

    #[test]
    fn segment_template_time_addressing_with_timeline() {
        let mpd = parse_mpd(
            r#"<MPD type="static" mediaPresentationDuration="PT12S">
                 <Period duration="PT12S">
                   <AdaptationSet contentType="video" mimeType="video/mp4">
                     <SegmentTemplate timescale="90000" initialization="init.mp4"
                                      media="chunk-$Time$.m4s">
                       <SegmentTimeline>
                         <S t="0" d="180000" r="2"/>
                         <S d="90000"/>
                         <S t="900000" d="45000" r="-1"/>
                       </SegmentTimeline>
                     </SegmentTemplate>
                     <Representation id="v" bandwidth="1000000" height="720"/>
                   </AdaptationSet>
                 </Period>
               </MPD>"#,
            BASE,
        )
        .unwrap();
        let p = &mpd.periods[0];
        let v = &p.adaptations[0].representations[0];
        let t = v.segment_template.as_ref().unwrap();
        assert_eq!(t.timeline.len(), 3);

        let ticks = mpd.period_duration_ticks(p, t.timescale).unwrap();
        assert_eq!(ticks, 1_080_000);
        let urls = t.media_urls(&v.id, v.bandwidth, Some(ticks));

        // r=2 → 3 段；随后一段；t=900000 起 r=-1 → (1080000-900000)/45000 = 4 段
        let expect_starts = [
            0u64, 180_000, 360_000, 540_000, 900_000, 945_000, 990_000, 1_035_000,
        ];
        assert_eq!(urls.len(), expect_starts.len());
        for (got, want) in urls.iter().zip(expect_starts.iter()) {
            assert_eq!(got.1, *want, "$Time$ 展开的起始刻度不对");
            assert_eq!(got.0, format!("chunk-{want}.m4s"));
        }
        assert_eq!(urls[0].2, 180_000);
        assert_eq!(urls[3].2, 90_000);
        assert_eq!(urls[7].2, 45_000);
    }

    #[test]
    fn timeline_repeat_minus_one_without_period_duration_keeps_one_entry() {
        let t = SegmentTemplate {
            media: Some("c-$Time$.m4s".to_string()),
            timescale: 1000,
            timeline: vec![TimelineEntry {
                start: Some(500),
                duration: 1000,
                repeat: -1,
            }],
            ..SegmentTemplate::default()
        };
        // 推不出次数：保留这一段（它自己的 d 是明确的），但绝不猜次数
        let got = t.expand_timeline(None);
        assert_eq!(got, vec![(500, 1000)]);
        // 给了 Period 时长就能算：(5000-500)/1000 = 4 段
        let got = t.expand_timeline(Some(5000));
        assert_eq!(got, vec![(500, 1000), (1500, 1000), (2500, 1000), (3500, 1000)]);
    }

    #[test]
    fn timeline_zero_duration_stops_instead_of_looping_forever() {
        let t = SegmentTemplate {
            timescale: 1,
            timeline: vec![
                TimelineEntry { start: None, duration: 100, repeat: 1 },
                TimelineEntry { start: None, duration: 0, repeat: -1 },
            ],
            ..SegmentTemplate::default()
        };
        assert_eq!(t.expand_timeline(Some(10_000)), vec![(0, 100), (100, 100)]);
    }

    #[test]
    fn unknown_template_variables_are_left_untouched() {
        let t = SegmentTemplate {
            media: Some("$RepresentationID$/$Bandwidth$-$Number$-$Foo$.m4s".to_string()),
            timescale: 1000,
            duration: Some(1000),
            start_number: 3,
            ..SegmentTemplate::default()
        };
        let urls = t.media_urls("v0", 1234, Some(1000));
        // 认不出来的 $Foo$ 原样保留：上层看到模板残留就能立刻发现清单不兼容，
        // 比静默产出一条"看起来正常"的错 URL 强
        assert_eq!(urls[0].0, "v0/1234-3-$Foo$.m4s");
    }

    #[test]
    fn dollar_escape_and_string_id_are_handled() {
        let t = SegmentTemplate {
            media: Some("v$$$RepresentationID$-$Number$.m4s".to_string()),
            timescale: 1000,
            duration: Some(1000),
            start_number: 1,
            ..SegmentTemplate::default()
        };
        assert_eq!(t.media_urls("r1", 0, Some(1000))[0].0, "v$r1-1.m4s");
    }

    // ---- SegmentList ----

    #[test]
    fn segment_list_is_expanded_with_inherited_base() {
        let mpd = parse_mpd(
            r#"<MPD type="static" mediaPresentationDuration="PT8S">
                 <Period>
                   <AdaptationSet contentType="video" mimeType="video/mp4">
                     <BaseURL>sub/</BaseURL>
                     <SegmentList timescale="1000" duration="PT4S">
                       <Initialization sourceURL="init.mp4"/>
                       <SegmentURL media="s0.m4s"/>
                       <SegmentURL media="s1.m4s"/>
                     </SegmentList>
                     <Representation id="v" bandwidth="500000" height="480"/>
                   </AdaptationSet>
                 </Period>
               </MPD>"#,
            BASE,
        )
        .unwrap();
        let v = &mpd.periods[0].adaptations[0].representations[0];
        let list = v.segment_list.as_ref().unwrap();
        assert_eq!(list.duration_us, Some(4_000_000));
        assert_eq!(list.timescale, 1000);
        assert_eq!(list.segment_duration_ticks(), Some(4000));
        assert_eq!(
            list.urls,
            vec![
                "http://origin.example.com/vod/sub/s0.m4s".to_string(),
                "http://origin.example.com/vod/sub/s1.m4s".to_string()
            ]
        );
        assert_eq!(
            list.init.as_deref(),
            Some("http://origin.example.com/vod/sub/init.mp4")
        );
        let expanded = list.media_urls();
        assert_eq!(expanded.len(), 2);
        assert_eq!(expanded[1].1, 4000);
        assert_eq!(expanded[1].2, 4000);
    }

    // ---- BaseURL / xml:base ----

    #[test]
    fn xml_base_and_nested_base_urls_stack() {
        let mpd = parse_mpd(
            r#"<MPD type="static" mediaPresentationDuration="PT4S" xml:base="http://a.example.com/root/">
                 <BaseURL>level1/</BaseURL>
                 <Period xml:base="../p0/">
                   <AdaptationSet>
                     <BaseURL>l2/</BaseURL>
                     <SegmentTemplate timescale="1" duration="1" media="$Number$.m4s"/>
                     <Representation id="v" bandwidth="1">
                       <BaseURL>l3/</BaseURL>
                     </Representation>
                   </AdaptationSet>
                 </Period>
               </MPD>"#,
            BASE,
        )
        .unwrap();
        // xml:base 先作用，再叠加 BaseURL 子元素
        assert_eq!(mpd.base_urls[0], "http://a.example.com/root/level1/");
        // Period 的 `../p0/` 是从 `root/level1/` 往上一级
        assert_eq!(mpd.periods[0].base_urls[0], "http://a.example.com/root/p0/");
        let v = &mpd.periods[0].adaptations[0].representations[0];
        // AdaptationSet 的 `l2/` 与 Representation 的 `l3/` 逐层叠加
        assert_eq!(
            v.base_urls,
            vec!["http://a.example.com/root/p0/l2/l3/".to_string()]
        );
        // 最终地址由 Representation 的基准拼出来
        assert_eq!(
            url::resolve(&v.base_urls[0], "7.m4s"),
            "http://a.example.com/root/p0/l2/l3/7.m4s"
        );
    }

    // ---- 直播 ----

    #[test]
    fn dynamic_type_means_live() {
        let mpd = parse_mpd(
            r#"<MPD type="dynamic" minimumUpdatePeriod="PT2S" timeShiftBufferDepth="PT30S">
                 <Period id="live0" start="PT0S">
                   <AdaptationSet contentType="video" mimeType="video/mp4">
                     <SegmentTemplate timescale="90000" media="$Number$.m4s"/>
                     <Representation id="v" bandwidth="1"/>
                   </AdaptationSet>
                 </Period>
               </MPD>"#,
            BASE,
        )
        .unwrap();
        assert!(mpd.is_live);
        assert!(mpd.type_is_dynamic);
        assert_eq!(mpd.min_update_period_us, Some(2_000_000));
        assert_eq!(mpd.duration_us, None);
    }

    #[test]
    fn dynamic_mpd_without_period_is_not_an_error() {
        // 直播清单刚生成时可能还没有 Period，这不该被判成坏清单
        let mpd = parse_mpd(r#"<MPD type="dynamic"/>"#, BASE).unwrap();
        assert!(mpd.periods.is_empty());
        assert!(mpd.is_live);

        // 点播清单没有 Period 就是坏清单
        let e = parse_mpd(r#"<MPD type="static"/>"#, BASE).unwrap_err();
        assert!(e.to_string().contains("Period"), "实际: {e}");
    }

    // ---- 内容类型推断 ----

    #[test]
    fn content_type_is_inferred_from_mime_and_codecs() {
        let mpd = parse_mpd(
            r#"<MPD type="static" mediaPresentationDuration="PT4S">
                 <Period>
                   <AdaptationSet mimeType="application/mp4" codecs="mp4a.40.2">
                     <Representation id="a" bandwidth="1"/>
                   </AdaptationSet>
                   <AdaptationSet mimeType="application/mp4" codecs="avc1.4d401f">
                     <Representation id="v" bandwidth="2"/>
                   </AdaptationSet>
                   <AdaptationSet>
                     <Representation id="s" bandwidth="3" mimeType="text/vtt"/>
                   </AdaptationSet>
                 </Period>
               </MPD>"#,
            BASE,
        )
        .unwrap();
        let sets = &mpd.periods[0].adaptations;
        assert_eq!(sets[0].content_type.as_deref(), Some("audio"));
        assert_eq!(sets[1].content_type.as_deref(), Some("video"));
        assert_eq!(sets[2].content_type.as_deref(), Some("text"));
    }

    // ---- 选流 ----

    #[test]
    fn pick_representation_prefers_highest_within_limits() {
        let mpd = parse_mpd(VOD_MPD, BASE).unwrap();
        let sets = &mpd.periods[0].adaptations;

        // 不设上限：视频里带宽最高的是 v1
        assert_eq!(
            pick_representation(sets, "video", None, None).unwrap().id,
            "v1"
        );
        assert_eq!(
            pick_representation(sets, "audio", None, None).unwrap().id,
            "a0"
        );
        // 限制高度 360：只剩 v0
        assert_eq!(
            pick_representation(sets, "video", Some(360), None).unwrap().id,
            "v0"
        );
        // 上限谁都满足不了 → 取最低带宽者，而不是 None
        assert_eq!(
            pick_representation(sets, "video", Some(100), Some(1000)).unwrap().id,
            "v0"
        );
        // 类型对不上就是 None
        assert!(pick_representation(sets, "text", None, None).is_none());
    }

    // ---- 坏清单 ----

    #[test]
    fn malformed_mpd_reports_errors() {
        // 根元素不对
        let e = parse_mpd("<Manifest/>", BASE).unwrap_err();
        assert!(e.to_string().contains("MPD"), "实际: {e}");
        // XML 本身坏了
        assert!(parse_mpd("<MPD><Period></MPD>", BASE).is_err());
        // 时长写法非法
        let e = parse_mpd(
            r#"<MPD type="static" mediaPresentationDuration="30s"><Period/></MPD>"#,
            BASE,
        )
        .unwrap_err();
        assert!(e.to_string().contains("ISO 8601"), "实际: {e}");
        // 年/月无法换算成精确秒数，必须明确拒绝而不是瞎算
        let e = parse_mpd(
            r#"<MPD type="static" mediaPresentationDuration="P1M"><Period/></MPD>"#,
            BASE,
        )
        .unwrap_err();
        assert!(matches!(e, MediaError::Unsupported(_)), "实际: {e}");
    }

    #[test]
    fn iso8601_durations() {
        assert_eq!(parse_duration_us("PT30S").unwrap(), 30_000_000);
        assert_eq!(parse_duration_us("PT2S").unwrap(), 2_000_000);
        assert_eq!(parse_duration_us("PT1H2M3S").unwrap(), 3_723_000_000);
        assert_eq!(parse_duration_us("PT1.5S").unwrap(), 1_500_000);
        assert_eq!(parse_duration_us("PT0.33334S").unwrap(), 333_340);
        assert_eq!(parse_duration_us("P1DT1S").unwrap(), 86_401_000_000);
        assert_eq!(parse_duration_us("PT1M").unwrap(), 60_000_000);
        // 只有 P、没有时间量
        assert!(parse_duration_us("PT").is_err());
        assert!(parse_duration_us("PT3").is_err());
        assert!(parse_duration_us("T3S").is_err());
    }

    #[test]
    fn period_start_is_inherited_when_omitted() {
        let mpd = parse_mpd(
            r#"<MPD type="static">
                 <Period duration="PT10S"/>
                 <Period duration="PT5S"/>
               </MPD>"#,
            BASE,
        )
        .unwrap();
        assert_eq!(mpd.periods[0].start_us, 0);
        // 第二个 Period 没写 start：应当接着上一个的结束
        assert_eq!(mpd.periods[1].start_us, 10_000_000);
    }
}
