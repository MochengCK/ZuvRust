//! 解码层：**播放内核的第一块** —— 把压缩采样解成帧。
//!
//! ## 为什么要有这一层（而不是交给浏览器）
//!
//! 引擎的目标是"任何前端都能直接调它播放"：前端只负责窗口与 UI，数据和解码
//! 都在引擎里。所以解码器必须由引擎持有，而不是靠 `<video>`/MSE 背后的
//! Chromium 解码器 —— 后者把"能播什么"绑死在浏览器上，换个前端就没了。
//!
//! ## 后端策略：**硬解优先，自研软解兜底**
//!
//! 具体走哪条路由 [`DecodeMode`] 说话（宿主「设置 → 视频 → 解码方式」那一档）：
//! **自适应**（默认，硬解起不来落自研软解）/ **仅硬解**（起不来就明确报错，
//! 不静默掉到慢路）/ **仅软解**（不碰硬件解码器）。由环境变量
//! `ME_DECODE=auto|hardware|software` 给出，旧名 `ME_NO_HW` 仍等价于"仅软解"。
//!
//! | 后端 | 状态 | 说明 |
//! |---|---|---|
//! | 平台硬解（VideoToolbox / VA-API / Media Foundation） | **三个平台都已接 H.264**（另：macOS 还有 AV1） | 用**操作系统**的解码器：不是 FFmpeg/GStreamer 那类第三方媒体引擎，而是每台机器本来就有的能力；功耗与性能最好 |
//! | 自研软解 | **AV1 已完成**（见 [`video`]）；**H.264 在建**（见 [`video::h264`]，参数集/切片头/成帧已就绪）| 兜底：硬件不支持的编码也能播，且保证"完全自研"这条底线在任何平台都成立 |
//!
//! 三条硬解通路的**分工不一样**，这也是 [`h264`] 存在的理由：
//! macOS 的 VideoToolbox 与 Windows 的 Media Foundation 是**整解码器**
//! （切片头、参考帧、输出顺序它们自己管），Linux 的 VA-API 是 **VLD 接口**
//! （这些全得我们自己做）。三家的差异都收敛在各自的 `macos.rs` / `windows.rs` /
//! `linux.rs` 里，解析层的共识在 [`h264`]。
//!
//! AV1 是这条策略最典型的例子：Apple 只从 **M3 及以后**提供 AV1 解码器，Intel
//! 机的 VideoToolbox 连软件兜底都没有（实测 `VTDecompressionSessionCreate`
//! 返回 `-12906 kVTCouldNotFindVideoDecoderErr`）。所以"引擎支持 AV1"这件事
//! 必须靠自研软解 —— 硬解只是"有就用"的那一半。
//!
//! 调用方只认 [`VideoDecoder`] 这个 trait（`open_video_decoder` 负责选后端），
//! 换后端不影响上层一行代码。
//!
//! ## 数据形态的约定
//!
//! - 输入采样：**AVCC**（4 字节大端长度前缀），与容器解出来的 H.264 一致；
//!   引擎里 `Packet::data` 统一是 Annex-B，转换由调用方用 `me-mux::convert` 做
//!   —— 解码层不认识容器，也不认识 Annex-B。**AV1 例外**：它的采样是 OBU 流
//!   （每个 OBU 自带 `leb128` 长度），原样送进来，不做任何形态转换。
//! - 输出帧：**紧凑 NV12**（Y 平面 w×h，CbCr 交织平面 (w/2)×h/2 行优先），
//!   加上显示时间戳（微秒）。渲染层拿到它就能上屏（WebGL / 原生纹理都一样）。

use me_core::{CodecId, MediaError, Result};

pub mod audio;
pub mod h264;

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "windows")]
mod windows;

pub mod video;

/// 一帧解码结果。
///
/// NV12 是刻意选的：它是所有平台解码器的"母语"（VideoToolbox / MediaCodec /
/// D3D11 都直接产出），上屏时也正好是 GPU 最容易吃的格式 —— 引擎里不做
/// 颜色空间转换，把这一步留给渲染层（那里是 GPU 的活）。
#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    /// 显示时间戳（微秒，媒体时间轴）
    pub pts_us: i64,
    /// 紧凑 NV12：`w*h` 字节 Y + `w*(h/2)` 字节 CbCr（CbCr 交错，宽度 = w）
    pub nv12: Vec<u8>,
}

impl Drop for VideoFrame {
    fn drop(&mut self) {
        // 紧凑 NV12 是每帧 3 MB 的大缓冲，解码器每条路径都会**写满**它 ⇒
        // 还给 `video::pool` 复用，省掉下一帧的分配 + 清零（`vec![0u8; n]`）。
        let v = std::mem::take(&mut self.nv12);
        crate::video::pool::give_u8(v);
    }
}

impl VideoFrame {
    /// 紧凑布局下 NV12 应有的字节数（诊断/校验用）。
    pub fn expected_len(width: u32, height: u32) -> usize {
        (width as usize) * (height as usize) * 3 / 2
    }
}

/// 视频解码器：喂采样，收帧。
///
/// 同步接口：`decode` 返回这次调用**已经解出来**的帧（可能为空 —— 有 B 帧时
/// 输入在先、输出在后，这是解码器的固有延迟，不是错误）。
pub trait VideoDecoder: Send {
    /// 喂一个采样（AVCC），返回此次新解出的帧。
    fn decode(&mut self, sample_avcc: &[u8], pts_us: i64) -> Result<Vec<VideoFrame>>;

    /// 冲刷：把解码器里排队的帧全部取出（输入读完之后调用）。
    fn flush(&mut self) -> Result<Vec<VideoFrame>>;

    /// 后端名字（日志/诊断用）。
    fn backend(&self) -> &'static str;
}

/// 解码后端偏好：宿主「设置 → 视频 → 解码方式」那一档映射过来的三选一。
///
/// 为什么要有"仅硬解"这一档：硬解在少数素材上有兼容问题（花屏），而软解又慢得多。
/// 三档把选择权交给用户：自适应（默认，能硬解就硬解）、仅硬解（**起不来就报错**，
/// 不静默掉到慢路）、仅软解（完全不碰硬件解码器，用于排查"是不是硬解的锅"）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeMode {
    /// 自适应：优先硬解，起不来落自研软解
    Auto,
    /// 仅硬解：硬解起不来就明确报错（不落软解）
    Hardware,
    /// 仅软解：不碰硬件解码器（只有自研软解的编码照常走自研软解）
    Software,
}

/// 解码模式的来源：环境变量。
///
/// - `ME_DECODE=auto|hardware|software`（也认 `hw` / `sw` 简写与中文）；
/// - `ME_NO_HW`（旧名，值任意）等价于 `ME_DECODE=software` —— 留着是为了不打破
///   已经在用它的脚本与文档；两个都设时以旧名为准（它更"强"）。
/// - 认不出来的值按自适应走并**告警**，不静默按某个方向解。
pub fn decode_mode() -> DecodeMode {
    parse_decode_mode(
        std::env::var("ME_DECODE").ok().as_deref(),
        std::env::var_os("ME_NO_HW").is_some(),
    )
}

/// [`decode_mode`] 的纯函数部分（不碰进程环境，好写好测）。
fn parse_decode_mode(me_decode: Option<&str>, no_hw: bool) -> DecodeMode {
    if no_hw {
        return DecodeMode::Software;
    }
    let value = me_decode.map(|v| v.trim().to_ascii_lowercase()).unwrap_or_default();
    match value.as_str() {
        "" | "auto" => DecodeMode::Auto,
        "hardware" | "hw" | "硬解" => DecodeMode::Hardware,
        "software" | "sw" | "软解" => DecodeMode::Software,
        other => {
            tracing::warn!("ME_DECODE={other} 认不出来（只认 auto / hardware / software），按自适应走");
            DecodeMode::Auto
        }
    }
}

/// 打开一个视频解码器（按 [`decode_mode`] 决定硬解 / 软解 / 自适应）。
///
/// `config` 是该编码的**解码器初始化数据**（H.264 = `avcC`、AV1 = `av1C`），
/// 由容器层给出、`me-mux::convert` 归一化成各自的标准形态（AV1 本来就是
/// `av1C`，没有 Annex-B 形态，所以是原样透传）。
pub fn open_video_decoder(codec: CodecId, config: &[u8]) -> Result<Box<dyn VideoDecoder>> {
    open_with_mode(decode_mode(), codec, config)
}

/// [`open_video_decoder`] 的实现（模式显式传入 —— 测试要能点名三档各自的行为）。
fn open_with_mode(mode: DecodeMode, codec: CodecId, config: &[u8]) -> Result<Box<dyn VideoDecoder>> {
    let use_hardware = mode != DecodeMode::Software;
    match codec {
        CodecId::H264 => {
            // 硬解优先：三个平台各有一条**系统自带**的解码通路。
            // 记下最后一个失败原因 —— 硬解起不来时要把"为什么"告诉用户，
            // 而不是丢一句"不支持 H.264"（那是错的：机器可能只是没有硬件解码器）。
            let mut failure: Option<String> = None;
            #[cfg(target_os = "macos")]
            if use_hardware {
                match macos::VtDecoder::open_h264(config) {
                    Ok(d) => return Ok(Box::new(d)),
                    Err(e) => {
                        tracing::warn!("VideoToolbox 硬解不可用：{e}");
                        failure = Some(format!("{e}"));
                    }
                }
            }
            #[cfg(target_os = "linux")]
            if use_hardware {
                match linux::VaDecoder::open(config) {
                    Ok(d) => return Ok(Box::new(d)),
                    Err(e) => {
                        tracing::warn!("VA-API 硬解不可用：{e}");
                        failure = Some(format!("{e}"));
                    }
                }
            }
            #[cfg(target_os = "windows")]
            if use_hardware {
                match windows::MfDecoder::open(config) {
                    Ok(d) => return Ok(Box::new(d)),
                    Err(e) => {
                        tracing::warn!("Media Foundation 硬解不可用：{e}");
                        failure = Some(format!("{e}"));
                    }
                }
            }
            // 到这里说明硬解没打开。**有两条完全不同的原因**，文案必须分开说：
            // 「仅硬解」是用户点名不要软解（那就该报错，而不是偷偷落到慢路）；
            // 其余情况（自适应 / 仅软解）落到自研软解兜底。
            if mode == DecodeMode::Hardware {
                return Err(MediaError::Unsupported(match failure {
                    Some(reason) => format!(
                        "解码方式是「仅硬解」，但 H.264 硬解（{}）起不来：{reason}",
                        hardware_backend()
                    ),
                    None => format!(
                        "解码方式是「仅硬解」，但这台机器上没有可用的 H.264 硬解（{}）",
                        hardware_backend()
                    ),
                }));
            }
            // 自研软解：进度见 `video::h264` 的分层表。**打开成功**不代表能出画面 ——
            // 它会在第一个真正带帧的采样上给出"进度到哪了"的明确错误，而不是产一帧
            // 黑屏把缺口藏起来（与 AV1 同一条约定）。
            Ok(Box::new(video::h264::H264Decoder::open(config)?))
        }
        CodecId::Av1 => {
            // 硬解优先，但**先问能力再建会话**：Apple 的 AV1 解码器只存在于硬件里
            // （M3 及以后；Intel 机的 VideoToolbox 连软件兜底都没有 —— 实测
            // `VTDecompressionSessionCreate` 返回 -12906）。所以 Intel 机上"先建会话
            // 再看报错"只会让用户看到一句 VideoToolbox 的 OSStatus，却看不到自研软解
            // 的真实进度；这里直接用能力探测当门，把话说清楚。
            //
            // 若哪天 Apple 在 VideoToolbox 里加了**软件** AV1 解码器（那种情况下
            // 这个探测会返回 false 而会话其实建得起来），这道门要改成"先试建会话"。
            let mut failure: Option<String> = None;
            #[cfg(target_os = "macos")]
            if use_hardware {
                if macos::hardware_decode_available(&CodecId::Av1) {
                    match macos::VtDecoder::open_av1(config) {
                        Ok(d) => {
                            tracing::info!(
                                "AV1 走 VideoToolbox 硬解（{}）",
                                video::av1::decode_outlook(config)
                            );
                            return Ok(Box::new(d));
                        }
                        Err(e) => {
                            tracing::warn!("AV1 硬解建会话失败：{e}");
                            failure = Some(format!("{e}"));
                        }
                    }
                } else {
                    tracing::info!("这台机器没有 AV1 硬解（Apple 只从 M3 起提供）");
                }
            }
            if mode == DecodeMode::Hardware {
                // 「仅硬解」就不该偷偷落到自研软解（那正是用户想避免的慢路）
                return Err(MediaError::Unsupported(match failure {
                    Some(reason) => {
                        format!("解码方式是「仅硬解」，但 AV1 硬解建不起来：{reason}")
                    }
                    None => "解码方式是「仅硬解」，但这台机器上 AV1 没有可用的硬解".into(),
                }));
            }
            // 自研软解：进度见 `video` 模块的分层表。**打开成功**不代表能出画面 ——
            // 它会在第一个真正带帧的采样上给出"进度到哪了"的明确错误（含已经读到的
            // 序列头），而不是产一帧黑屏把缺口藏起来。
            tracing::info!("AV1 走自研软解（{}）", video::av1::decode_outlook(config));
            Ok(Box::new(video::av1::Av1Decoder::open(config)?))
        }
        other => Err(MediaError::Unsupported(format!(
            "解码器还没接这个编码：{}（已接：H.264 硬解、AV1 硬解 + 自研软解）",
            other.name()
        ))),
    }
}

/// 当前平台可用的硬解后端名字（诊断/日志）。
pub fn hardware_backend() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "videotoolbox"
    }
    #[cfg(target_os = "linux")]
    {
        "vaapi"
    }
    #[cfg(target_os = "windows")]
    {
        "mediafoundation"
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        "none"
    }
}

/// 这个编码在这台机器上有没有**平台硬解**。
///
/// 与 [`open_video_decoder`] 的分工：这个回答"有没有硬件解码器"（决定"能不能
/// 硬解"），后者负责"实际打开一个解码器"（可能是硬解，也可能是自研软解）。
/// 日志与错误文案拿它来说清"为什么这台机器播不了 AV1"。
pub fn hardware_decode_available(codec: &CodecId) -> bool {
    #[cfg(target_os = "macos")]
    {
        macos::hardware_decode_available(codec)
    }
    #[cfg(target_os = "linux")]
    {
        linux::hardware_decode_available(codec)
    }
    #[cfg(target_os = "windows")]
    {
        windows::hardware_decode_available(codec)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        let _ = codec;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nv12_len_matches_layout() {
        assert_eq!(VideoFrame::expected_len(1920, 1080), 1920 * 1080 * 3 / 2);
        assert_eq!(VideoFrame::expected_len(640, 360), 640 * 360 * 3 / 2);
    }

    #[test]
    fn av1_opens_a_decoder_with_a_readable_status() {
        // AV1 现在**能打开**（硬解或自研软解的壳），打开后立刻能从后端名字看出
        // 走的是哪条路 —— 这正是"改完之后要能回答'到底用的什么内核'"。
        let av1c = [0x81u8, 0x00, 0x0C, 0x00];
        let dec = open_video_decoder(CodecId::Av1, &av1c).expect("AV1 应当能打开一个解码器");
        let backend = dec.backend();
        assert!(
            backend == "videotoolbox" || backend == "self-av1",
            "后端只能是硬解或自研软解：{backend}"
        );
        if !hardware_decode_available(&CodecId::Av1) {
            assert_eq!(backend, "self-av1", "没有硬解时必须落到自研软解");
        }
    }

    #[test]
    fn av1_without_config_is_rejected() {
        assert!(open_video_decoder(CodecId::Av1, &[]).is_err(), "没有 av1C 建不了解码器");
    }

    /// H.264：要么真的开出一个硬解解码器，要么落到**自研软解**，要么给一句能读懂的原因。
    ///
    /// 这条测试在三个平台都成立：有硬解的机器上它走硬解；没有硬解的机器
    /// （CI 容器、无 GPU 的服务器）**落到自研软解**（不再是"没得播"）；
    /// 只有「仅硬解」那一档才会真的报错。
    #[test]
    fn h264_open_ends_in_a_decoder_or_a_readable_reason() {
        let avcc = h264::tests_support::mini_avcc();
        match open_video_decoder(CodecId::H264, &avcc) {
            Ok(d) => {
                let backend = d.backend();
                assert!(
                    matches!(backend, "videotoolbox" | "vaapi" | "mediafoundation" | "self-h264"),
                    "H.264 的后端只能是硬解或自研软解：{backend}"
                );
                if backend == "self-h264" {
                    assert!(
                        !hardware_decode_available(&CodecId::H264),
                        "有硬解时不该落到自研软解"
                    );
                } else {
                    assert_eq!(backend, hardware_backend(), "后端名要与平台一致");
                }
            }
            Err(e) => {
                let text = format!("{e}");
                assert!(text.contains("H.264"), "失败文案要说清是哪个编码：{text}");
                assert!(text.contains("仅硬解"), "只有「仅硬解」这一档该失败：{text}");
                assert!(matches!(e, MediaError::Unsupported(_)), "{text}");
            }
        }
    }

    #[test]
    fn unsupported_codec_says_what_is_missing() {
        // 用引擎真正没接的编码（VP9）来钉"明确报错"这条约定
        let e = match open_video_decoder(CodecId::Vp9, &[]) {
            Ok(_) => panic!("VP9 还没接，不该成功"),
            Err(e) => e,
        };
        let text = format!("{e}");
        assert!(text.contains("vp9"), "错误里要有编码名：{text}");
    }

    /// 三档解码模式由两个环境变量翻出来（纯函数部分，不碰进程环境）。
    #[test]
    fn decode_mode_comes_from_env_values() {
        assert_eq!(parse_decode_mode(None, false), DecodeMode::Auto, "什么都不设 = 自适应");
        assert_eq!(parse_decode_mode(Some(""), false), DecodeMode::Auto);
        assert_eq!(parse_decode_mode(Some("auto"), false), DecodeMode::Auto);
        assert_eq!(parse_decode_mode(Some("Hardware"), false), DecodeMode::Hardware, "大小写不敏感");
        assert_eq!(parse_decode_mode(Some(" hw "), false), DecodeMode::Hardware);
        assert_eq!(parse_decode_mode(Some("software"), false), DecodeMode::Software);
        assert_eq!(parse_decode_mode(Some("sw"), false), DecodeMode::Software);
        assert_eq!(parse_decode_mode(Some("???"), false), DecodeMode::Auto, "认不出来按自适应");
        // 旧名优先（它更"强"）：两个都设时按仅软解
        assert_eq!(parse_decode_mode(Some("hardware"), true), DecodeMode::Software);
    }

    /// 「仅软解」= **完全不碰**硬件解码器。
    ///
    /// 两个编码现在都有自研软解可走：AV1 那台最典型（即使机器有 AV1 硬解也落到
    /// 自研软解），H.264 同理 —— 用户选这一档就是为了"排除硬解的锅"。
    #[test]
    fn software_mode_never_touches_hardware() {
        let av1c = [0x81u8, 0x00, 0x0C, 0x00];
        let dec = open_with_mode(DecodeMode::Software, CodecId::Av1, &av1c)
            .expect("仅软解下 AV1 应当由自研软解接管");
        assert_eq!(dec.backend(), "self-av1", "仅软解必须走自研软解，哪怕机器有硬解");

        // H.264 现在也有自研软解了：这一档必须开出 `self-h264`，**不碰 VideoToolbox**
        let avcc = h264::tests_support::mini_avcc();
        let dec = open_with_mode(DecodeMode::Software, CodecId::H264, &avcc)
            .expect("仅软解下 H.264 应当由自研软解接管");
        assert_eq!(dec.backend(), "self-h264", "仅软解必须走自研软解，哪怕机器有硬解");
    }

    /// 「仅硬解」= 硬解起不来就**报错**，不静默落到自研软解。
    ///
    /// 两种机器都成立：有 AV1 硬解的机器上它开出硬解；没有的（Intel Mac、
    /// CI 容器）走 Err，且文案必须点明"仅硬解"。
    #[test]
    fn hardware_mode_does_not_fall_back_to_software() {
        let av1c = [0x81u8, 0x00, 0x0C, 0x00];
        match open_with_mode(DecodeMode::Hardware, CodecId::Av1, &av1c) {
            Ok(d) => {
                assert!(
                    matches!(d.backend(), "videotoolbox" | "vaapi" | "mediafoundation"),
                    "仅硬解成功时只能是硬解后端：{}",
                    d.backend()
                );
                assert!(hardware_decode_available(&CodecId::Av1), "能开出硬解就说明探测也该说'有'");
            }
            Err(e) => {
                let text = format!("{e}");
                assert!(text.contains("仅硬解"), "文案要点明是解码方式选的：{text}");
                assert!(!hardware_decode_available(&CodecId::Av1), "没有 AV1 硬解才会走到这里");
            }
        }
    }
}
