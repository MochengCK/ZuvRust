//! 自研 XML 解析器（只覆盖 MPD 真正用到的那一小撮语法）。
//!
//! 为什么自己写：本工程硬性禁止第三方依赖，而 MPD 用到的 XML 语法其实极窄 ——
//! 元素、属性、文本、注释、CDATA、实体转义。真正难缠的是**命名空间前缀**
//! （`cenc:pssh`、`xml:base`）与**属性值里的引号**，这两点必须在解析阶段就处理干净，
//! 否则上层 MPD 代码到处都要做字符串兜底。手写状态机还顺带省掉了通用 XML 库的
//! DTD / 外部实体 / schema 这些我们完全用不到、却最容易出问题的那部分。
//!
//! 几条刻意的决定：
//!
//! 1. **元素名只留 local name**（`cenc:pssh` → `pssh`）。MPD 里我们关心的元素
//!    （Period/AdaptationSet/Representation/SegmentTemplate…）都不带前缀，
//!    留前缀只会让 `child("pssh")` 这类查询变得不可预测。
//! 2. **属性名保留原文**（含前缀）。因为 `xml:base` 是唯一带前缀且语义重要的属性，
//!    它的 local name 恰好是 `base`；若也去前缀就会和别处可能出现的 `base` 混淆。
//!    [`XmlNode::attr`] 允许用 local name 兜底匹配带前缀的属性，单向匹配不会产生歧义。
//! 3. **文本原样保留**（不 trim、不做行尾归一）。MPD 里元素之间的缩进空白也是文本，
//!    解析器擅自 trim 就把"排版空白"和"真正的文本"混成了一件事；
//!    需要干净文本的调用方调 [`XmlNode::text_trimmed`]。
//! 4. **宽容但不静默**：裸 `&`（URL 里忘了转义）原样保留，但**看起来是实体却写坏了**的
//!    （带分号的未知实体 `&foo;`、非法数字实体 `&#xZZ;`）一律报错 ——
//!    静默当普通文本会让 BaseURL 里悄悄出现错字符，排查成本极高。
//! 5. **只取第一个根元素**，其后的内容忽略。MPD 只有一个根，后面的东西对播放没有意义，
//!    为它拒绝整份清单不值得。

use me_core::error::{MediaError, Result};

/// 一个 XML 元素节点。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct XmlNode {
    /// 元素名，**已去掉命名空间前缀**（`cenc:pssh` → `pssh`）。
    pub name: String,
    /// 属性，顺序与原文一致；属性名**保留前缀**（见模块文档第 2 条）。
    pub attrs: Vec<(String, String)>,
    /// 子元素。文本节点不会出现在这里：元素内部的文本一律拼进 [`XmlNode::text`]。
    pub children: Vec<XmlNode>,
    /// 元素内部所有文本片段（含 CDATA）按出现顺序拼接的结果，未做任何修剪。
    pub text: String,
}

impl XmlNode {
    /// 按属性名取值。
    ///
    /// 先按原文精确匹配；没命中再用 local name 兜底匹配带前缀的属性，
    /// 于是 `xml:base` 既可以用 `"xml:base"` 取，也可以用 `"base"` 取。
    /// 反向不会命中（无前缀属性不可能匹配到带前缀的查询），所以不存在歧义。
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(k, _)| k == name)
            .or_else(|| self.attrs.iter().find(|(k, _)| local_name(k) == name))
            .map(|(_, v)| v.as_str())
    }

    /// 第一个同名子元素。
    pub fn child(&self, name: &str) -> Option<&XmlNode> {
        self.children.iter().find(|c| c.name == name)
    }

    /// 所有同名子元素（保持原文顺序）。
    pub fn children_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a XmlNode> + 'a {
        self.children.iter().filter(move |c| c.name == name)
    }

    /// 去掉首尾空白后的文本。
    pub fn text_trimmed(&self) -> &str {
        self.text.trim()
    }

    /// 同名子元素的文本（去掉首尾空白）。MPD 里 `<BaseURL>` 这种"只放文本"的元素用它最省事。
    pub fn child_text(&self, name: &str) -> Option<&str> {
        self.child(name).map(|c| c.text_trimmed())
    }
}

/// 去掉命名空间前缀：`cenc:pssh` → `pssh`；无前缀原样返回。
fn local_name(name: &str) -> &str {
    match name.split_once(':') {
        Some((_, local)) => local,
        None => name,
    }
}

/// 解析一份 XML 文本，返回第一个根元素。
pub fn parse(text: &str) -> Result<XmlNode> {
    let mut p = Parser { s: text, pos: 0 };
    // 跳过 BOM：不少 MPD 是 Windows 上的工具生成的，BOM 会让首字节判定全错。
    if p.s.starts_with('\u{feff}') {
        p.pos = '\u{feff}'.len_utf8();
    }
    loop {
        p.skip_ws();
        if p.eof() {
            return Err(MediaError::protocol("XML 里没有任何元素"));
        }
        if p.eat("<?") {
            p.skip_until("?>")?;
            continue;
        }
        if p.eat("<!--") {
            p.skip_until("-->")?;
            continue;
        }
        if p.eat("<!") {
            p.skip_decl()?;
            continue;
        }
        if p.peek() == Some(b'<') {
            return p.parse_element();
        }
        return Err(MediaError::protocol(format!(
            "XML 起始处出现意外内容（{}）",
            p.describe()
        )));
    }
}

// ---------------------------------------------------------------------------
// 手写状态机
// ---------------------------------------------------------------------------

struct Parser<'a> {
    s: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn eof(&self) -> bool {
        self.pos >= self.s.len()
    }

    fn peek(&self) -> Option<u8> {
        self.s.as_bytes().get(self.pos).copied()
    }

    /// 当前位置起是否以 `pat` 开头；是则消费掉它。
    fn eat(&mut self, pat: &str) -> bool {
        if self.s[self.pos..].starts_with(pat) {
            self.pos += pat.len();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, b: u8) -> Result<()> {
        if self.peek() == Some(b) {
            self.pos += 1;
            Ok(())
        } else {
            Err(MediaError::protocol(format!(
                "XML 期望 '{}'（{}）",
                b as char,
                self.describe()
            )))
        }
    }

    /// 跳过空白；返回是否真的跳过了（属性之间必须靠空白分隔，靠它来判定）。
    fn skip_ws(&mut self) -> bool {
        let start = self.pos;
        while matches!(self.peek(), Some(b) if b.is_ascii_whitespace()) {
            self.pos += 1;
        }
        self.pos > start
    }

    /// 出错位置附近的一小段原文，用来定位坏清单。
    fn describe(&self) -> String {
        // 这是字节级状态机，`pos` 有可能落在多字节字符中间；切片前必须退到字符边界，
        // 否则"格式化一条错误信息"本身就会 panic —— 报错路径上再崩一次最要命。
        let mut start = self.pos.min(self.s.len());
        while start > 0 && !self.s.is_char_boundary(start) {
            start -= 1;
        }
        let mut end = (start + 24).min(self.s.len());
        while end > start && !self.s.is_char_boundary(end) {
            end -= 1;
        }
        format!("第 {} 字节 '{}'", self.pos, &self.s[start..end])
    }

    fn skip_until(&mut self, pat: &str) -> Result<()> {
        match self.s[self.pos..].find(pat) {
            Some(i) => {
                self.pos += i + pat.len();
                Ok(())
            }
            None => Err(MediaError::protocol(format!(
                "XML 缺少 '{pat}' 结束标记（{}）",
                self.describe()
            ))),
        }
    }

    /// 跳过 `<!DOCTYPE ...>` 这类声明。DOCTYPE 可能带内部子集 `[...]`，
    /// 里面的 `>` 不算结束，所以要记方括号深度。
    fn skip_decl(&mut self) -> Result<()> {
        let mut depth = 0i32;
        while let Some(b) = self.peek() {
            self.pos += 1;
            match b {
                b'[' => depth += 1,
                b']' => depth -= 1,
                b'>' if depth <= 0 => return Ok(()),
                _ => {}
            }
        }
        Err(MediaError::protocol(format!(
            "XML 声明没有结束的 '>'（{}）",
            self.describe()
        )))
    }

    /// 读一个名字（元素名或属性名）。非 ASCII 字节一律算作名字字符，
    /// 这样 UTF-8 的多字节名字不会被截断。
    fn read_name(&mut self) -> Option<String> {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if is_name_char(b) {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            None
        } else {
            Some(self.s[start..self.pos].to_string())
        }
    }

    fn parse_element(&mut self) -> Result<XmlNode> {
        self.expect(b'<')?;
        let raw_name = self
            .read_name()
            .ok_or_else(|| MediaError::protocol(format!("XML 元素名缺失（{}）", self.describe())))?;
        let name = local_name(&raw_name).to_string();

        let mut attrs: Vec<(String, String)> = Vec::new();
        loop {
            let had_ws = self.skip_ws();
            match self.peek() {
                Some(b'/') => {
                    self.pos += 1;
                    self.expect(b'>')?;
                    return Ok(XmlNode {
                        name,
                        attrs,
                        children: Vec::new(),
                        text: String::new(),
                    });
                }
                Some(b'>') => {
                    self.pos += 1;
                    break;
                }
                Some(_) => {
                    if !had_ws {
                        return Err(MediaError::protocol(format!(
                            "XML 元素 <{raw_name}> 的属性之间缺少空白（{}）",
                            self.describe()
                        )));
                    }
                    let (k, v) = self.read_attr()?;
                    attrs.push((k, v));
                }
                None => {
                    return Err(MediaError::protocol(format!(
                        "XML 元素 <{raw_name}> 未闭合（{}）",
                        self.describe()
                    )))
                }
            }
        }

        let mut node = XmlNode {
            name,
            attrs,
            children: Vec::new(),
            text: String::new(),
        };
        loop {
            if self.eof() {
                return Err(MediaError::protocol(format!(
                    "XML 元素 <{raw_name}> 缺少结束标签"
                )));
            }
            if self.eat("</") {
                let close = self.read_name().ok_or_else(|| {
                    MediaError::protocol(format!("XML 结束标签缺名字（{}）", self.describe()))
                })?;
                self.skip_ws();
                self.expect(b'>')?;
                if local_name(&close) != node.name {
                    return Err(MediaError::protocol(format!(
                        "XML 结束标签不匹配: <{}> 对 </{}>",
                        node.name, close
                    )));
                }
                return Ok(node);
            }
            if self.eat("<!--") {
                self.skip_until("-->")?;
                continue;
            }
            if self.eat("<![CDATA[") {
                let start = self.pos;
                let end = self.s[start..].find("]]>").ok_or_else(|| {
                    MediaError::protocol(format!("CDATA 没有结束的 ']]>'（{}）", self.describe()))
                })?;
                node.text.push_str(&self.s[start..start + end]);
                self.pos = start + end + 3;
                continue;
            }
            if self.eat("<?") {
                self.skip_until("?>")?;
                continue;
            }
            if self.eat("<!") {
                self.skip_decl()?;
                continue;
            }
            if self.peek() == Some(b'<') {
                let child = self.parse_element()?;
                node.children.push(child);
                continue;
            }
            // 文本：一路读到下一个 '<'
            let start = self.pos;
            while !self.eof() && self.peek() != Some(b'<') {
                self.pos += 1;
            }
            node.text.push_str(&decode_entities(&self.s[start..self.pos])?);
        }
    }

    fn read_attr(&mut self) -> Result<(String, String)> {
        let name = self
            .read_name()
            .ok_or_else(|| MediaError::protocol(format!("XML 属性名缺失（{}）", self.describe())))?;
        self.skip_ws();
        self.expect(b'=')?;
        self.skip_ws();
        let quote = self.peek().ok_or_else(|| {
            MediaError::protocol(format!("XML 属性 {name} 缺少值（{}）", self.describe()))
        })?;
        if quote != b'"' && quote != b'\'' {
            return Err(MediaError::protocol(format!(
                "XML 属性 {name} 的值没有引号（{}）",
                self.describe()
            )));
        }
        self.pos += 1;
        let start = self.pos;
        while !self.eof() && self.peek() != Some(quote) {
            self.pos += 1;
        }
        if self.eof() {
            return Err(MediaError::protocol(format!(
                "XML 属性 {name} 的引号未闭合（{}）",
                self.describe()
            )));
        }
        let raw = &self.s[start..self.pos];
        self.pos += 1; // 吃掉收尾引号
        Ok((name, decode_entities(raw)?))
    }
}

/// 名字字符：字母、数字、`_`、`-`、`.`、`:`，以及所有非 ASCII 字节。
fn is_name_char(b: u8) -> bool {
    b >= 0x80 || b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b':')
}

// ---------------------------------------------------------------------------
// 实体
// ---------------------------------------------------------------------------

/// 解开 XML 实体（`&amp;` `&lt;` `&gt;` `&quot;` `&apos;` 与十进制/十六进制数字实体）。
fn decode_entities(s: &str) -> Result<String> {
    if !s.contains('&') {
        return Ok(s.to_string());
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        match entity_at(rest)? {
            Some((decoded, consumed)) => {
                out.push_str(&decoded);
                rest = &rest[consumed..];
            }
            None => {
                // 不像实体的裸 `&`（URL 里没转义很常见），原样保留。
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    Ok(out)
}

/// 从以 `&` 开头的字符串里识别一个实体。
///
/// 返回 `Ok(None)` = 这个 `&` 不像实体（原样保留）；`Err` = 看起来是实体却写坏了。
/// 判据是**有没有分号**：有分号说明作者本来就想写实体，此时名字非法/未知一律报错；
/// 没有分号（`AT&T`、`?a=1&b=2`）就当普通 `&`。
fn entity_at(s: &str) -> Result<Option<(String, usize)>> {
    let b = s.as_bytes();
    let mut i = 1;
    while i < b.len() {
        let c = b[i];
        if c == b';' {
            break;
        }
        let ok = c.is_ascii_alphanumeric() || matches!(c, b'#' | b'_' | b'.' | b':' | b'-');
        if !ok {
            return Ok(None);
        }
        i += 1;
        if i > 34 {
            // 长到离谱，肯定不是实体名
            return Ok(None);
        }
    }
    if i >= b.len() || b[i] != b';' {
        return Ok(None);
    }
    let name = &s[1..i];
    let consumed = i + 1;
    let decoded = match name {
        "amp" => "&".to_string(),
        "lt" => "<".to_string(),
        "gt" => ">".to_string(),
        "quot" => "\"".to_string(),
        "apos" => "'".to_string(),
        _ => {
            let num = name.strip_prefix('#').ok_or_else(|| {
                MediaError::protocol(format!("XML 里出现未定义的实体 '&{name};'"))
            })?;
            let cp = if let Some(hex) = num.strip_prefix('x').or_else(|| num.strip_prefix('X')) {
                u32::from_str_radix(hex, 16)
            } else {
                num.parse::<u32>()
            }
            .map_err(|_| MediaError::protocol(format!("XML 数字实体非法: '&{name};'")))?;
            let ch = char::from_u32(cp).filter(|c| *c != '\0').ok_or_else(|| {
                MediaError::protocol(format!("XML 数字实体超出合法码点: '&{name};'"))
            })?;
            ch.to_string()
        }
    };
    Ok(Some((decoded, consumed)))
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declaration_comments_and_self_closing() {
        let xml = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
                   <!-- 这是注释，里面的 <MPD> 不该被当成元素 -->\n\
                   <MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\" minBufferTime=\"PT2S\">\n\
                     <Period id=\"p0\"/>\n\
                     <Period id=\"p1\"></Period>\n\
                   </MPD>\n";
        let root = parse(xml).unwrap();
        assert_eq!(root.name, "MPD");
        assert_eq!(root.attr("minBufferTime"), Some("PT2S"));
        assert_eq!(root.attr("xmlns"), Some("urn:mpeg:dash:schema:mpd:2011"));
        assert_eq!(root.children_named("Period").count(), 2);
        assert_eq!(root.children[1].attr("id"), Some("p1"));
        // 自闭合元素没有子节点
        assert!(root.children[0].children.is_empty());
    }

    #[test]
    fn cdata_and_entities_in_text() {
        let xml = "<BaseURL><![CDATA[http://h/a&b?x=1]]></BaseURL>";
        let n = parse(xml).unwrap();
        assert_eq!(n.text_trimmed(), "http://h/a&b?x=1");

        let n = parse("<t>a&amp;b &lt;c&gt; &quot;d&quot; &apos;e&apos;</t>").unwrap();
        assert_eq!(n.text_trimmed(), "a&b <c> \"d\" 'e'");
    }

    #[test]
    fn numeric_entities() {
        let n = parse("<t>&#65;&#x42;&#x4e2d;</t>").unwrap();
        assert_eq!(n.text_trimmed(), "AB中");
    }

    #[test]
    fn bare_ampersand_is_kept_but_broken_entity_errors() {
        // URL 里没转义的 & 很常见，必须原样留下
        let n = parse("<BaseURL>http://h/a?x=1&y=2</BaseURL>").unwrap();
        assert_eq!(n.text_trimmed(), "http://h/a?x=1&y=2");
        let n = parse("<t>AT&T</t>").unwrap();
        assert_eq!(n.text_trimmed(), "AT&T");

        // 写了分号说明本来就是实体，名字不认识必须报错而不是静默
        let e = parse("<t>&foo;</t>").unwrap_err();
        assert!(e.to_string().contains("&foo;"), "实际: {e}");
        assert!(parse("<t>&#xZZ;</t>").is_err());
    }

    #[test]
    fn namespaces_and_prefixed_attributes() {
        let xml = "<Period xml:base=\"http://cdn/p/\" cenc:default_KID=\"abc\">\
                     <cenc:pssh>AAAA</cenc:pssh>\
                   </Period>";
        let n = parse(xml).unwrap();
        assert_eq!(n.name, "Period");
        // 元素名去前缀
        assert!(n.child("pssh").is_some());
        assert!(n.child("cenc:pssh").is_none());
        // 属性名保留原文，且能用 local name 兜底取到
        assert_eq!(n.attr("xml:base"), Some("http://cdn/p/"));
        assert_eq!(n.attr("base"), Some("http://cdn/p/"));
        assert_eq!(n.attr("cenc:default_KID"), Some("abc"));
        assert_eq!(n.attr("default_KID"), Some("abc"));
        assert_eq!(n.attr("nope"), None);
    }

    #[test]
    fn single_quoted_attrs_and_attr_entities() {
        let n = parse("<a b='x y' c=\"p&amp;q\"/>").unwrap();
        assert_eq!(n.attr("b"), Some("x y"));
        assert_eq!(n.attr("c"), Some("p&q"));
    }

    #[test]
    fn text_keeps_whitespace_but_trimmed_helper_cleans_it() {
        let n = parse("<BaseURL>\n  http://cdn/x/\n</BaseURL>").unwrap();
        assert_eq!(n.text, "\n  http://cdn/x/\n");
        assert_eq!(n.text_trimmed(), "http://cdn/x/");
        assert_eq!(n.child_text("BaseURL"), None);
    }

    #[test]
    fn malformed_xml_reports_error_instead_of_panicking() {
        assert!(parse("<a><b></a>").is_err());
        assert!(parse("<a").is_err());
        assert!(parse("<a b=c/>").is_err());
        assert!(parse("<a><!-- 没关").is_err());
        assert!(parse("<a><![CDATA[没关</a>").is_err());
        assert!(parse("纯文本没有元素").is_err());
        assert!(parse("").is_err());
    }

    #[test]
    fn doctype_with_internal_subset_is_skipped() {
        let xml = "<!DOCTYPE MPD [ <!ENTITY x \"y\"> ]>\n<MPD/>";
        let n = parse(xml).unwrap();
        assert_eq!(n.name, "MPD");
    }

    #[test]
    fn bom_and_leading_whitespace() {
        let n = parse("\u{feff}\n  <MPD id=\"1\"/>").unwrap();
        assert_eq!(n.attr("id"), Some("1"));
    }

    #[test]
    fn first_root_wins_and_trailing_junk_is_ignored() {
        let n = parse("<MPD/><extra/>").unwrap();
        assert_eq!(n.name, "MPD");
    }
}
