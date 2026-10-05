//! ZuvRust 的**通信层端点**。
//!
//! 这个二进制是整个引擎与外界唯一的接口。宿主（Electron 桌面端 / Android / 任何
//! 前端）**不链接引擎的任何 Rust 代码**，只做两件事：
//!
//! 1. 用命令行子命令做一次性任务（`probe` / `mux` / `download`）；
//! 2. 用 `--json` 让每条输出都是一行 JSON，宿主按行解析（不用写解析器）。
//!
//! 为什么是"进程 + 文本协议"而不是 `cdylib` + C ABI：
//!
//! - **崩溃隔离**：容器解析面对的是全网最脏的字节，进程内崩一次宿主就跟着崩；
//!   独立进程崩了宿主只是收到一个非零退出码。
//! - **零耦合**：宿主不需要 Rust 工具链、不需要匹配 ABI、不需要为每个平台重新链接。
//!   引擎升级 = 换一个可执行文件。
//! - **可测**：任何人拿 `curl` 与 `zuvrust` 就能复现问题，不用起整个宿主。
//!
//! 代价是进程启动开销（毫秒级）与大文件不能共享内存 —— 对"下载/合并"这类
//! 本来就是磁盘 IO 主导的任务，这个代价可以忽略。

use std::process::ExitCode;

mod commands;
mod decode;
mod json;
mod mux;
mod progress;
mod stream;
mod wav;

/// 退出码是**对外契约**的一部分：宿主靠它区分"地址有问题"与"引擎自己坏了"。
///
/// 只用一个非零码的话，宿主没法决定"要不要重试" —— 重试一个格式不支持的文件
/// 永远不会成功，而重试一次网络超时是对的。
pub mod exit_code {
    /// 成功。
    pub const OK: u8 = 0;
    /// 参数用法错误（宿主把命令拼错了，重试无用）。
    pub const USAGE: u8 = 2;
    /// 输入不可用：地址打不开、容器认不出、码流非法（重试无用）。
    pub const BAD_INPUT: u8 = 3;
    /// 临时失败：网络超时、BT 还没下到那一段（**可以重试**）。
    pub const TRANSIENT: u8 = 4;
    /// 引擎内部错误（bug，或环境缺东西）。
    pub const INTERNAL: u8 = 5;
    /// 调用方取消。
    pub const CANCELLED: u8 = 6;
}

fn main() -> ExitCode {
    // **不能用 `std::env::args()`**：参数里只要有一个不是合法 UTF-8 它就**直接 panic**
    // （Linux 上文件名可以是任意字节序列；Windows 上是未配对的 UTF-16 代理对）。
    // panic 的退出码 101 不在对外契约里 —— 宿主只会看到"引擎自己崩了"，
    // 连一句能看懂的错误都拿不到。这里改成：转不出来就按"输入不可用"如实报错。
    let args: Vec<String> = match args_utf8() {
        Ok(a) => a,
        Err(bad) => {
            let e = me_core::MediaError::Io(format!(
                "命令行参数含非 UTF-8 字节，无法作为路径使用：{}",
                bad.to_string_lossy()
            ));
            eprintln!("{}", json::error_line(&e));
            return ExitCode::from(exit_code::BAD_INPUT);
        }
    };

    // `--help` / 无参数都要给出用法，而不是报错退出 —— 宿主在探测引擎能力时
    // 常常就是先跑一次空命令看它认哪些子命令。
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return ExitCode::from(exit_code::OK);
    }

    // `--version` / `-V`：与 `version` 子命令等价。
    //
    // 为什么必须认这两种写法：几乎每个命令行工具都同时提供两者（GNU 惯例），
    // 而**冒烟脚本与 CI 会本能地写 `--version`** —— 让人以为"工具坏了"的成本
    // 远高于多认两个参数。判据是"**第一个**参数就是它"，而不是"参数里含它"：
    // 后者会把 `probe clip--version.mkv` 这类**文件名里带 --version** 的输入
    // 误当成版本查询。
    //
    // ⚠️ 关键点：只换掉第一个参数，**其余参数原样带过去** ——
    // `--version --json` 必须仍然是 `--json` 模式。丢掉它们的话调用方会拿到
    // 没有 `{ok, result}` 包装的裸载荷，而它是按 `result.tracks` 这类路径读结果的
    // —— 少一层包装就是"字段全是 null"，且**不报任何错**（最难查的一类回归）。
    if matches!(args.first().map(String::as_str), Some("--version") | Some("-V")) {
        let mut rest: Vec<String> = args[1..].to_vec();
        rest.insert(0, "version".to_string());
        init_tracing(&rest);
        return dispatch_exit(&rest);
    }

    init_tracing(&args);
    dispatch_exit(&args)
}

/// 执行分发并把结果翻译成退出码（`--version` 与常规路径共用这一段）。
fn dispatch_exit(args: &[String]) -> ExitCode {
    match commands::dispatch(args) {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            // 错误统一走 stderr，且是**一行 JSON**（宿主不必区分"正常输出"与"错误输出"
            // 的格式差异）；`--json` 只是显式声明这个意图。
            eprintln!("{}", json::error_line(&e));
            ExitCode::from(commands::exit_code_of(&e))
        }
    }
}

/// 收命令行参数（不含 argv[0]），全部转成 UTF-8 字符串；有转不出来的就把那个
/// 原始参数（`OsString`）回给调用方，由它给出一条符合契约的错误。
///
/// 代价说清楚：**非 UTF-8 的路径目前不支持**（转不出来就没法按路径打开），
/// 但它会**报错而不是崩** —— 这条底线比"支持任意字节路径"重要得多。
fn args_utf8() -> std::result::Result<Vec<String>, std::ffi::OsString> {
    std::env::args_os().skip(1).map(|a| a.into_string()).collect()
}

/// 日志要不要上色。**独立成函数是为了能测**（见 `tests/cli_contract.rs`）。
///
/// 判据是"stderr 是不是终端"，而不是"平台是不是 Windows"：
/// 宿主（Electron / 任何前端）永远是通过**管道**读引擎的 stderr，
/// 那种情况下 `is_terminal()` 为 false；而人在终端里手跑引擎时为 true，
/// 想看彩色日志。这条同时覆盖了两个诉求，不用按平台分支。
fn ansi_enabled() -> bool {
    use std::io::IsTerminal;
    std::io::stderr().is_terminal()
}

/// `--log=<level>` 打开日志（默认关闭：宿主不该被引擎的内部日志污染 stdout）。
fn init_tracing(args: &[String]) {
    let level = args
        .iter()
        .find_map(|a| a.strip_prefix("--log="))
        .unwrap_or("warn");
    let filter = tracing_subscriber::EnvFilter::try_new(level)
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(ansi_enabled())
        .with_writer(std::io::stderr)
        .try_init();
}

fn print_usage() {
    println!(
        r#"zuvrust —— 自研媒体引擎（独立进程，通过命令行通信）

用法: zuvrust <子命令> [参数...] [--json]

子命令:
  probe <输入>                 探测容器与轨道信息
  mux   <输出> <输入>...       合并成一个文件（角色由探测决定，同角色按顺序串接）
  remux <输出> <输入>          换个容器重新封装（不重编码）
  download <地址> [--out=路径]  下载（HTTP / HLS / DASH；磁力由下载引擎负责）
  play  <输入> [--seek=秒]      播放数据源（边下边播时给宿主喂采样）
  stream <输入> [--seek=秒]     流式播放：一行元信息 + 分片 fMP4 字节（宿主喂 MSE）
                                stream 支持 --avail=<文件>：可用性图（哪些字节已下到，空洞会等）
                                stream 支持 --wait-timeout=<毫秒>：等数据的上限（默认 30000）
  decode <输入> [--count=N]     把视频轨解成帧（播放内核的"解码"这块；--out=目录 落盘）
                               --audio 解音频轨（自研软解）写成 WAV：--audio --out=x.wav
  version                      打印版本与支持的容器/协议

通用选项:
  --version, -V                与 `version` 子命令等价（GNU 惯例）
  --json                       所有输出为一行 JSON（宿主按行解析）
  --log=<级别>                 打开日志到 stderr（默认 warn）
  --help                       显示这份说明

退出码:
  0 成功   2 用法错误   3 输入不可用（重试无用）
  4 临时失败（可重试）   5 引擎内部错误   6 已取消
"#
    );
}
