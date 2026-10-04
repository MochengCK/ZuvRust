//! 输出格式化：**一行一条 JSON**，宿主按行读。
//!
//! 为什么不用"一个完整 JSON 文档"：下载/播放是**流式**的，进度要实时吐给宿主。
//! 攒成一个文档就得等到任务结束才输出，那宿主界面上的进度条就没法动。
//! NDJSON（newline-delimited JSON）让"随时追加一行"变成天然支持。
//!
//! 为什么不用纯文本：宿主两端（Electron 的 JS 与 Android 的 Kotlin）解析 JSON
//! 都是一行代码，而解析自定义文本格式要各写一套正则 —— 那是两份会漂移的实现。

use me_core::MediaError;

/// 成功的结果行。
pub fn ok_line(value: &serde_json::Value) -> String {
    serde_json::json!({ "ok": true, "result": value }).to_string()
}

/// 进度行（流式任务用；宿主按 `type` 字段分发）。
pub fn progress_line(kind: &str, value: serde_json::Value) -> String {
    serde_json::json!({ "ok": true, "type": kind, "data": value }).to_string()
}

/// 错误行。
///
/// `kind` 是**机器可读的分类**（宿主靠它决定要不要重试），`message` 是给人看的。
/// 只给 message 的话，宿主只能靠字符串匹配判断"这是网络问题还是格式问题"，
/// 而引擎的文案一改宿主就失灵。
pub fn error_line(e: &MediaError) -> String {
    serde_json::json!({
        "ok": false,
        "error": {
            "kind": error_kind(e),
            "message": e.to_string(),
            "retryable": is_retryable(e),
        }
    })
    .to_string()
}

/// 错误分类（对外契约，改名字等于破坏宿主）。
pub fn error_kind(e: &MediaError) -> &'static str {
    match e {
        MediaError::Io(_) => "io",
        MediaError::Container(_) => "container",
        MediaError::Bitstream(_) => "bitstream",
        MediaError::Protocol(_) => "protocol",
        MediaError::Unsupported(_) => "unsupported",
        MediaError::NeedMore { .. } => "need-more",
        MediaError::Cancelled => "cancelled",
    }
}

/// 这个错误值不值得重试。
pub fn is_retryable(e: &MediaError) -> bool {
    match e {
        // 网络抖动、BT 还没下到：重试是对的
        MediaError::Protocol(_) => true,
        MediaError::NeedMore { .. } => true,
        // 文件坏了、格式不支持、路径打不开：重试永远不会成功
        MediaError::Container(_) | MediaError::Bitstream(_) | MediaError::Unsupported(_) => false,
        // 本地 IO 失败同样重试无用（路径不存在 / 权限 / 磁盘满）：
        // 可重试的临时失败统一走 Protocol 与 NeedMore 这两条
        MediaError::Io(_) => false,
        MediaError::Cancelled => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok_line_is_single_line_json() {
        let s = ok_line(&serde_json::json!({"a": 1}));
        assert!(!s.contains('\n'), "必须是单行：{s}");
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["result"]["a"], 1);
    }

    #[test]
    fn progress_line_carries_type() {
        let s = progress_line("progress", serde_json::json!({"percent": 42}));
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["type"], "progress");
        assert_eq!(v["data"]["percent"], 42);
        assert_eq!(v["ok"], true);
    }

    #[test]
    fn error_line_classifies_and_flags_retryable() {
        let e = MediaError::container("bad bytes");
        let s = error_line(&e);
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["ok"], false);
        assert_eq!(v["error"]["kind"], "container");
        assert_eq!(v["error"]["retryable"], false);

        let t = MediaError::protocol("timeout");
        let v2: serde_json::Value = serde_json::from_str(&error_line(&t)).unwrap();
        assert_eq!(v2["error"]["kind"], "protocol");
        assert_eq!(v2["error"]["retryable"], true);

        // 本地 IO 失败：分类是 io，但**不可重试**（同一个路径重试永远不会成功）
        let io = MediaError::Io("no such file".into());
        let v3: serde_json::Value = serde_json::from_str(&error_line(&io)).unwrap();
        assert_eq!(v3["error"]["kind"], "io");
        assert_eq!(v3["error"]["retryable"], false);
    }

    #[test]
    fn all_error_kinds_have_a_name() {
        // 新增变体时必须同步这里，否则宿主会收到 "unknown"
        for e in [
            MediaError::Io("x".into()),
            MediaError::Container("x".into()),
            MediaError::Bitstream("x".into()),
            MediaError::Protocol("x".into()),
            MediaError::Unsupported("x".into()),
            MediaError::NeedMore { need: 1, have: 0 },
            MediaError::Cancelled,
        ] {
            let k = error_kind(&e);
            assert!(!k.is_empty());
            assert_ne!(k, "unknown", "{e} 没有分类");
        }
    }
}
