//! 字幕：把容器里的文本字幕解析成"按时间查询的 cue 表"。
//!
//! 只做**文本**字幕（SRT / ASS / WebVTT / mov_text）。PGS、VobSub 是位图，
//! 需要另一条渲染链，容器层已经把它们跳过了。
//!
//! 分工：容器给"这条字幕包"，时间轴（pts/duration）也来自容器；这里只负责
//! 把包里的字节变成人能读的文本，并把一条轨的所有包攒成一张按时间排好的表。

use me_core::CodecId;

/// 一条字幕：`[start_us, end_us)` 期间显示 `text`。
#[derive(Clone, Debug)]
pub struct Cue {
    pub start_us: i64,
    pub end_us: i64,
    pub text: String,
}

/// 把一条字幕包解析成纯文本。时间来自容器，不从文本里抠。
///
/// - `S_TEXT/UTF8`（我们映射成 `CodecId::SUBRIP`）：包体就是字幕文本
/// - `S_TEXT/ASS`：包体是 `ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text`，
///   Text 前面 8 个逗号，Text 自身还可能带逗号
/// - WebVTT：包体是 cue 文本
/// - `tx3g`：2 字节长度 + UTF-8
pub fn packet_text(codec: &CodecId, data: &[u8]) -> Option<String> {
    let text = match *codec {
        CodecId::MOVTEXT => {
            if data.len() < 2 {
                return None;
            }
            let len = u16::from_be_bytes([data[0], data[1]]) as usize;
            let end = (2 + len).min(data.len());
            String::from_utf8_lossy(&data[2..end]).to_string()
        }
        CodecId::ASS => {
            let s = String::from_utf8_lossy(data);
            // 前 8 个字段用逗号分隔，剩下全是正文
            let mut rest = s.as_ref();
            for _ in 0..8 {
                match rest.find(',') {
                    Some(i) => rest = &rest[i + 1..],
                    None => return None,
                }
            }
            rest.to_string()
        }
        _ => String::from_utf8_lossy(data).to_string(),
    };
    let cleaned = cleanup(&text);
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

/// 去掉排版标记，只留下能直接显示的文本。
///
/// 为什么不保留标签交给富文本渲染：ASS 的 `{\pos(...)}`、`{\an8}` 这类覆盖码
/// 语义完全不同，Qt 的富文本不认；直接丢给 Text 会原样显示成一堆花括号。
fn cleanup(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            // {\...} ASS 覆盖码
            '{' => {
                for n in chars.by_ref() {
                    if n == '}' {
                        break;
                    }
                }
            }
            // <i> / </i> / <font ...> 之类
            '<' => {
                for n in chars.by_ref() {
                    if n == '>' {
                        break;
                    }
                }
            }
            '\\' => match chars.peek() {
                Some('N') | Some('n') => {
                    chars.next();
                    out.push('\n');
                }
                Some('h') => {
                    chars.next();
                    out.push(' ');
                }
                _ => out.push('\\'),
            },
            '\r' => {}
            _ => out.push(c),
        }
    }
    // 逐行裁掉首尾空白，再把连续空行压掉
    let lines: Vec<&str> = out.split('\n').map(|l| l.trim()).collect();
    let joined = lines.join("\n");
    joined.trim().trim_matches('\n').to_string()
}

/// 外部 `.srt` 文件解析（sidecar 字幕用）。
pub fn parse_srt(data: &str) -> Vec<Cue> {
    let mut cues = Vec::new();
    let text = data.replace("\r\n", "\n").replace('\r', "\n");
    for block in text.split("\n\n") {
        let mut lines = block.lines().filter(|l| !l.trim().is_empty());
        let Some(first) = lines.next() else { continue };
        // 时间行：`00:00:01,000 --> 00:00:02,500`（首行可能是序号）
        let time_line = if first.contains("-->") {
            first
        } else {
            match lines.next() {
                Some(l) if l.contains("-->") => l,
                _ => continue,
            }
        };
        let Some((a, b)) = time_line.split_once("-->") else { continue };
        let (Some(start), Some(end)) = (parse_srt_time(a), parse_srt_time(b)) else {
            continue;
        };
        let body = lines.collect::<Vec<_>>().join("\n");
        let cleaned = cleanup(&body);
        if cleaned.is_empty() {
            continue;
        }
        cues.push(Cue { start_us: start, end_us: end, text: cleaned });
    }
    cues.sort_by_key(|c| c.start_us);
    cues
}

/// 外部字幕文件：按内容判断格式再解析（sidecar 可能是 srt / ass / vtt）。
pub fn parse_text(data: &str) -> Vec<Cue> {
    let head = data.trim_start();
    // ASS/SSA 一定带 `[Script Info]`；有些精简文件只有 Dialogue 行
    if head.starts_with("[Script Info]") || head.contains("\nDialogue:") || head.starts_with("Dialogue:") {
        parse_ass_file(data)
    } else {
        parse_srt(data)
    }
}

/// 外部 `.ass` / `.ssa` 文件解析。
///
/// 行格式：`Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text`
/// —— 前 9 个字段逗号分隔，第 10 个字段是正文（正文里还能有逗号）。
pub fn parse_ass_file(data: &str) -> Vec<Cue> {
    let mut cues = Vec::new();
    for raw in data.lines() {
        let line = raw.trim_start();
        let Some(rest) = line.strip_prefix("Dialogue:") else { continue };
        let mut parts = rest.splitn(10, ',');
        let _layer = parts.next();
        let Some(start) = parts.next().and_then(parse_ass_time) else { continue };
        let Some(end) = parts.next().and_then(parse_ass_time) else { continue };
        // Style / Name / MarginL / MarginR / MarginV / Effect 共 6 个
        for _ in 0..6 {
            parts.next();
        }
        let Some(body) = parts.next() else { continue };
        let cleaned = cleanup(body);
        if cleaned.is_empty() {
            continue;
        }
        cues.push(Cue { start_us: start, end_us: end, text: cleaned });
    }
    cues.sort_by_key(|c| c.start_us);
    cues
}

/// `0:00:01.50` → 微秒（ASS 的时间精度是**百分秒**，不是毫秒）
fn parse_ass_time(s: &str) -> Option<i64> {
    let s = s.trim();
    let (hms, frac) = s.rsplit_once('.').or_else(|| s.rsplit_once(','))?;
    let mut it = hms.split(':');
    let h: i64 = it.next()?.trim().parse().ok()?;
    let m: i64 = it.next()?.trim().parse().ok()?;
    let sec: i64 = it.next()?.trim().parse().ok()?;
    let frac = frac.trim();
    let value: i64 = frac.parse().ok()?;
    // 按位数缩放：2 位 = 百分秒，3 位 = 毫秒
    let scale = match frac.len() {
        1 => 100_000,
        2 => 10_000,
        3 => 1_000,
        _ => return None,
    };
    Some((h * 3600 + m * 60 + sec) * 1_000_000 + value * scale)
}

/// `00:01:02,345` / `00:01:02.345` → 微秒
fn parse_srt_time(s: &str) -> Option<i64> {
    let s = s.trim();
    let (hms, ms) = match s.rsplit_once([',', '.']) {
        Some((a, b)) => (a, b),
        None => (s, "0"),
    };
    let mut it = hms.split(':');
    let h: i64 = it.next()?.trim().parse().ok()?;
    let m: i64 = it.next()?.trim().parse().ok()?;
    let sec: f64 = it.next()?.trim().parse().ok()?;
    let ms: i64 = ms.trim().parse().ok()?;
    Some(((h * 3600 + m * 60) as f64 * 1e6 + sec * 1e6 + ms as f64 * 1000.0) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srt_packet_keeps_text_and_drops_tags() {
        let t = packet_text(&CodecId::SUBRIP, "<i>你好</i>\n世界".as_bytes()).unwrap();
        assert_eq!(t, "你好\n世界");
    }

    #[test]
    fn ass_packet_takes_the_tenth_field() {
        let line = "1,0,Default,,0,0,0,,{\\an8}第一行\\N第二行";
        let t = packet_text(&CodecId::ASS, line.as_bytes()).unwrap();
        assert_eq!(t, "第一行\n第二行", "覆盖码要去掉，\\N 要变成换行");
    }

    #[test]
    fn ass_text_may_contain_commas() {
        let line = "1,0,Default,,0,0,0,,a, b, c";
        assert_eq!(packet_text(&CodecId::ASS, line.as_bytes()).unwrap(), "a, b, c");
    }

    #[test]
    fn mov_text_has_a_length_prefix() {
        let payload = b"hi";
        let mut data = (payload.len() as u16).to_be_bytes().to_vec();
        data.extend_from_slice(payload);
        assert_eq!(packet_text(&CodecId::MOVTEXT, &data).unwrap(), "hi");
    }

    #[test]
    fn external_srt_parses_timing_and_multiline_text() {
        let srt = "1\n00:00:01,000 --> 00:00:03,500\n第一行\n第二行\n\n2\n00:00:05.000 --> 00:00:06.000\n<i>斜体</i>\n";
        let cues = parse_srt(srt);
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].start_us, 1_000_000);
        assert_eq!(cues[0].end_us, 3_500_000);
        assert_eq!(cues[0].text, "第一行\n第二行");
        assert_eq!(cues[1].text, "斜体");
    }

    #[test]
    fn ass_file_parses_dialogue_lines() {
        let ass = r#"[Script Info]
Title: x

[Events]
Dialogue: 0,0:00:01.50,0:00:03.00,Default,,0,0,0,,{\an8}你好\N世界
Dialogue: 0,0:00:05.00,0:00:06.25,Default,,0,0,0,,带,逗号,正文
"#;
        let cues = parse_text(ass);
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].start_us, 1_500_000, "百分秒");
        assert_eq!(cues[0].end_us, 3_000_000);
        assert_eq!(cues[0].text, "你好\n世界");
        assert_eq!(cues[1].start_us, 5_000_000);
        assert_eq!(cues[1].end_us, 6_250_000);
        assert_eq!(cues[1].text, "带,逗号,正文", "正文里的逗号不能被切开");
    }

    #[test]
    fn srt_still_goes_through_parse_text() {
        let cues = parse_text("1\n00:00:01,000 --> 00:00:02,000\n一行\n");
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "一行");
    }

    #[test]
    fn empty_packet_is_ignored() {
        assert!(packet_text(&CodecId::SUBRIP, b"   \n  ").is_none());
    }
}
