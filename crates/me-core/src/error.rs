//! 引擎统一错误类型。

/// ZuvRust 的错误。
///
/// 刻意**按来源分类**而不是一个大枚举：容器解析失败与网络失败的处理方式完全不同
/// （前者要报给用户"这个文件坏了"，后者要重试），混在一起调用方只能靠字符串匹配。
#[derive(Debug, thiserror::Error)]
pub enum MediaError {
    /// 本地 IO（读写文件、管道）。
    #[error("IO 错误: {0}")]
    Io(String),

    /// 容器层错误：字节流不符合该容器的规范（找不到同步字、box 长度越界……）。
    #[error("容器格式错误: {0}")]
    Container(String),

    /// 码流层错误：比特流语法不对（Exp-Golomb 越界、NAL 长度非法……）。
    #[error("码流错误: {0}")]
    Bitstream(String),

    /// 协议层错误：清单解析、HTTP、BT 线协议。
    #[error("协议错误: {0}")]
    Protocol(String),

    /// 语法合法但这一版引擎还没实现。
    #[error("暂不支持: {0}")]
    Unsupported(String),

    /// 数据还不够（增量解析的常见返回）：**不是错误**，调用方应继续喂数据。
    ///
    /// 单独一个变体是为了让"再读一段"这条路径不落进错误分支 ——
    /// 否则解析器每读一个 box 都要靠字符串判断"是不是要更多数据"。
    #[error("需要更多数据（还需 {need} 字节，当前 {have}）")]
    NeedMore { need: usize, have: usize },

    /// 调用方主动取消。
    #[error("已取消")]
    Cancelled,
}

/// 引擎统一 `Result`。
pub type Result<T> = std::result::Result<T, MediaError>;

impl From<std::io::Error> for MediaError {
    fn from(e: std::io::Error) -> Self {
        MediaError::Io(e.to_string())
    }
}

impl MediaError {
    /// 便捷构造：把任意 `Display` 变成容器错误。
    pub fn container(msg: impl std::fmt::Display) -> Self {
        MediaError::Container(msg.to_string())
    }

    /// 便捷构造：协议错误。
    pub fn protocol(msg: impl std::fmt::Display) -> Self {
        MediaError::Protocol(msg.to_string())
    }

    /// 这一条是不是"数据不够，继续读"。
    pub fn is_need_more(&self) -> bool {
        matches!(self, MediaError::NeedMore { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn need_more_is_not_a_failure() {
        let e = MediaError::NeedMore { need: 16, have: 4 };
        assert!(e.is_need_more());
        // 其余变体都不是"继续读"
        assert!(!MediaError::Cancelled.is_need_more());
        assert!(!MediaError::container("x").is_need_more());
    }

    #[test]
    fn io_error_converts() {
        let e: MediaError = std::io::Error::new(std::io::ErrorKind::NotFound, "nope").into();
        assert!(matches!(e, MediaError::Io(_)));
        assert!(e.to_string().contains("nope"));
    }
}
