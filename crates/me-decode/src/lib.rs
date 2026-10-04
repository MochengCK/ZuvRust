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
//! | 后端 | 状态 | 说明 |
//! |---|---|---|
//! | 平台硬解（VideoToolbox / …） | macOS 已接（H.264、**AV1**） | 用**操作系统**的解码器：不是 FFmpeg/GStreamer 那类第三方媒体引擎，而是每台机器本来就有的能力；功耗与性能最好 |
//! | 自研软解 | **AV1 进行中**（OBU/序列头已落地，见 [`video`]）；H.264 待做 | 兜底：硬件不支持的编码也能播，且保证"完全自研"这条底线在任何平台都成立 |
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

#[cfg(target_os = "macos")]
mod macos;

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

/// 打开一个视频解码器（硬解优先，自研软解兜底）。
///
/// `config` 是该编码的**解码器初始化数据**（H.264 = `avcC`、AV1 = `av1C`），
/// 由容器层给出、`me-mux::convert` 归一化成各自的标准形态（AV1 本来就是
/// `av1C`，没有 Annex-B 形态，所以是原样透传）。
pub fn open_video_decoder(codec: CodecId, config: &[u8]) -> Result<Box<dyn VideoDecoder>> {
    match codec {
        CodecId::H264 => {
            #[cfg(target_os = "macos")]
            {
                // 硬解优先：这台机器有 VideoToolbox 就用它。
                match macos::VtDecoder::open_h264(config) {
                    Ok(d) => return Ok(Box::new(d)),
                    // 硬解起不来（配置缺失等）不该直接判死：软解兜底接上。
                    // 软解还没实现时，把两个原因都写清楚，别让人以为"引擎不支持 H.264"。
                    Err(e) => tracing::warn!("VideoToolbox 硬解不可用：{e}（将尝试软解兜底）"),
                }
            }
            Err(MediaError::Unsupported(
                "H.264 软解兜底还没实现（硬解目前只在 macOS 上接了 VideoToolbox）".into(),
            ))
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
            #[cfg(target_os = "macos")]
            {
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
                            tracing::warn!("AV1 硬解建会话失败：{e}（落到自研软解）");
                        }
                    }
                } else {
                    tracing::info!("这台机器没有 AV1 硬解（Apple 只从 M3 起提供）：走自研软解");
                }
            }
            // 自研软解：进度见 `video` 模块的分层表。**打开成功**不代表能出画面 ——
            // 它会在第一个真正带帧的采样上给出"进度到哪了"的明确错误（含已经读到的
            // 序列头），而不是产一帧黑屏把缺口藏起来。
            tracing::info!("AV1 走自研软解（{}）", video::av1::decode_outlook(config));
            Ok(Box::new(video::av1::Av1Decoder::open(config)?))
        }
        other => Err(MediaError::Unsupported(format!(
            "解码器还没接这个编码：{}（已接：H.264 硬解、AV1 硬解 + 自研软解进行中）",
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
    #[cfg(not(target_os = "macos"))]
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
    #[cfg(not(target_os = "macos"))]
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
}
