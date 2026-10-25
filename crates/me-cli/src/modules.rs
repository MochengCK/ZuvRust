//! 模块清单与"这个构建没编这个模块"的统一报错。
//!
//! feature 是 Cargo 的编译期概念；宿主看到的是运行时的 `version --json` 与命令
//! 报错，所以两处必须由同一份清单驱动，避免"version 说有、命令却说没有"。

use me_core::MediaError;

/// 当前构建编译进来的模块（顺序固定：audio / video / mux）。
pub(crate) fn compiled_modules() -> Vec<&'static str> {
    #[allow(unused_mut)]
    let mut out = Vec::new();
    #[cfg(feature = "audio")]
    out.push("audio");
    #[cfg(feature = "video")]
    out.push("video");
    #[cfg(feature = "mux")]
    out.push("mux");
    out
}

/// 命令需要某个模块、但当前构建没编它时的错误。
///
/// 用 `Unsupported`（退出码 2 = 用法错误）而不是内部错误：这不是引擎坏了，
/// 是这个二进制本来就没带这块能力，重试没有意义。
pub(crate) fn disabled(module: &str, feature: &str, command: &str) -> MediaError {
    MediaError::Unsupported(format!(
        "`{command}` 需要{module}模块，但当前构建未包含它（用 `--features {feature}` 重新构建）"
    ))
}
