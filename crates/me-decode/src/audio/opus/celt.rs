//! CELT 层——本轮落地的是**帧头前半段**：TOC 解析、帧头标志（silence / 后置滤波 /
//! 瞬态 / intra）、**粗能量**、时频（TF）变化、扩散（spread）。
//!
//! ## 为什么规范这一层"以参考实现为准"
//!
//! RFC 6716 明文写着：CELT 的**浮点参考实现的输出就是规范性定义**（§4.3 把概率模型、
//! 频带布局、TF 表、能量预测系数都指向参考实现里的表与函数，而不是在正文里列出）。
//! 所以这一层的做法是：**照参考实现的算法与表用 Rust 重写**，再用逐帧对拍证明一致
//! （`tests/opus_celt_ref.rs`）。这不是"抄"，而是规范给出的定义方式。
//!
//! ## 这一层在码流里的位置（顺序错一位后面全错）
//!
//! ```text
//! TOC(字节 1) → silence → 后置滤波 → 瞬态 → intra → 粗能量 → TF 变化 → spread
//!            → （下一轮）dynalloc 提升 → alloc_trim → 比特分配 → 细能量 → PVQ 形状 …
//! ```
//!
//! 本轮做到 `spread` 为止：这一段是**连续且可独立验证**的（对拍时比的就是到这里为止的
//! 帧参数与 `tell`）。`tell` 尤其重要 —— 后面的比特分配全靠它，而它必须与参考实现
//! **逐位相同**。
//!
//! ## 几个必须记住的点（都对着参考实现核过）
//!
//! 1. **粗能量是"跨帧状态"**：`old_ebands[ch][band]` 会带到下一帧做预测，而且
//!    silence 帧**不解码能量**（位数不够时按 `qi = -1` 走完），帧尾再把整张表置成
//!    **-28**。所以静音段可以让后面的帧能量一路衰减下去 —— 这是刻意设计的。
//! 2. 预测前要把上一帧的能量**下限钳到 -9**（`max(-9, old)`），否则长期静音会让
//!    预测项无界地负下去。
//! 3. 粗能量的三种降级路径（位数不够时）：`>=15` 位走 Laplace 概率模型、
//!    `>=2` 位走 2 位 icdf + 符号折叠、`>=1` 位走 1 位 logp，一位都没有就取 `-1`。
//!    静音帧走的就是最后一条。
//! 4. 通道循环在**带内**（`for band { for ch }`），因为 `prev[ch]` 是逐通道累加的
//!    频率预测项。

use me_core::{MediaError, Result};

use super::bands::{self, ShapeInput};
use super::celt_plc;
use super::entropy::RangeDecoder;
use super::mdct::Mdct;
use super::rate;
use super::synth::{self, SynthInput, SynthState};

/// 每个帧长对应的帧大小（48 kHz 采样）。CELT 的短块固定 120 个样本（2.5 ms）。
pub const SHORT_MDCT: u32 = 120;

/// 普通 CELT 的频带数（Opus Custom 可以不同，这里不涉及）。
pub const NB_EBANDS: usize = 21;

// ── 表（数值照参考实现；浮点构建下的移位宏是空操作，所以公式就是下面注释里的样子）──

/// 帧间预测系数 `pred_coef[LM]`（参考实现里是 `29440/32768.` 这种，浮点构建直接是小数）。
const PRED_COEF: [f32; 4] = [29440.0 / 32768.0, 26112.0 / 32768.0, 21248.0 / 32768.0, 16384.0 / 32768.0];
/// 频率方向预测的记忆衰减 `beta_coef[LM]`。
const BETA_COEF: [f32; 4] = [30147.0 / 32768.0, 22282.0 / 32768.0, 12124.0 / 32768.0, 6554.0 / 32768.0];
/// intra 帧用的衰减（`alpha = 0`）。
const BETA_INTRA: f32 = 4915.0 / 32768.0;

/// 位数不够时用的 2 位能量表。
const SMALL_ENERGY_ICDF: [u8; 3] = [2, 1, 0];
/// 扩散决策：NONE 21.875% / LIGHT 6.25% / NORMAL 65.625% / AGGRESSIVE 6.25%。
const SPREAD_ICDF: [u8; 4] = [25, 23, 2, 0];
/// 后置滤波器的抽头集。
const TAPSET_ICDF: [u8; 3] = [2, 1, 0];

/// `tf_select_table[LM][4*isTransient + 2*tf_select + tf_change]`。
const TF_SELECT_TABLE: [[i8; 8]; 4] = [
    [0, -1, 0, -1, 0, -1, 0, -1],
    [0, -1, 0, -2, 1, 0, 1, -1],
    [0, -2, 0, -3, 2, 0, 1, -1],
    [0, -2, 0, -3, 3, 0, 1, -1],
];

/// 粗能量的概率模型 `e_prob_model[LM][intra][2*min(band,20) + {0,1}]`
/// （每对是"取 0 的概率"和"衰减率"，都是 Q8）。
const E_PROB_MODEL: [[[u8; 42]; 2]; 4] = [
    [
        [
            72, 127, 65, 129, 66, 128,
            65, 128, 64, 128, 62, 128,
            64, 128, 64, 128, 92, 78,
            92, 79, 92, 78, 90, 79,
            116, 41, 115, 40, 114, 40,
            132, 26, 132, 26, 145, 17,
            161, 12, 176, 10, 177, 11,
        ],
        [
            24, 179, 48, 138, 54, 135,
            54, 132, 53, 134, 56, 133,
            55, 132, 55, 132, 61, 114,
            70, 96, 74, 88, 75, 88,
            87, 74, 89, 66, 91, 67,
            100, 59, 108, 50, 120, 40,
            122, 37, 97, 43, 78, 50,
        ],
    ],
    [
        [
            83, 78, 84, 81, 88, 75,
            86, 74, 87, 71, 90, 73,
            93, 74, 93, 74, 109, 40,
            114, 36, 117, 34, 117, 34,
            143, 17, 145, 18, 146, 19,
            162, 12, 165, 10, 178, 7,
            189, 6, 190, 8, 177, 9,
        ],
        [
            23, 178, 54, 115, 63, 102,
            66, 98, 69, 99, 74, 89,
            71, 91, 73, 91, 78, 89,
            86, 80, 92, 66, 93, 64,
            102, 59, 103, 60, 104, 60,
            117, 52, 123, 44, 138, 35,
            133, 31, 97, 38, 77, 45,
        ],
    ],
    [
        [
            61, 90, 93, 60, 105, 42,
            107, 41, 110, 45, 116, 38,
            113, 38, 112, 38, 124, 26,
            132, 27, 136, 19, 140, 20,
            155, 14, 159, 16, 158, 18,
            170, 13, 177, 10, 187, 8,
            192, 6, 175, 9, 159, 10,
        ],
        [
            21, 178, 59, 110, 71, 86,
            75, 85, 84, 83, 91, 66,
            88, 73, 87, 72, 92, 75,
            98, 72, 105, 58, 107, 54,
            115, 52, 114, 55, 112, 56,
            129, 51, 132, 40, 150, 33,
            140, 29, 98, 35, 77, 42,
        ],
    ],
    [
        [
            42, 121, 96, 66, 108, 43,
            111, 40, 117, 44, 123, 32,
            120, 36, 119, 33, 127, 33,
            134, 34, 139, 21, 147, 23,
            152, 20, 158, 25, 154, 26,
            166, 21, 173, 16, 184, 13,
            184, 10, 150, 13, 139, 15,
        ],
        [
            22, 178, 63, 114, 74, 82,
            84, 83, 92, 82, 103, 62,
            96, 72, 96, 67, 101, 73,
            107, 72, 113, 55, 118, 52,
            125, 52, 118, 52, 117, 55,
            135, 49, 137, 39, 157, 32,
            145, 29, 97, 33, 77, 40,
        ],
    ],
];

// ── TOC（RFC 6716 §3.1 Table 2）──

/// 编码模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusMode {
    SilkOnly,
    Hybrid,
    CeltOnly,
}

/// 音频带宽。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bandwidth {
    Narrow,
    Medium,
    Wide,
    SuperWide,
    Full,
}

impl Bandwidth {
    /// CELT 编码到第几带为止（参考实现里 `endband`；NB 13 / MB·WB 17 / SWB 19 / FB 21）。
    pub fn end_band(self) -> usize {
        match self {
            Bandwidth::Narrow => 13,
            Bandwidth::Medium | Bandwidth::Wide => 17,
            Bandwidth::SuperWide => 19,
            Bandwidth::Full => 21,
        }
    }
}

/// config（TOC 高 5 位）→（模式, 带宽）。RFC 6716 Table 2 的分组。
fn mode_and_bandwidth(config: u8) -> (OpusMode, Bandwidth) {
    match config {
        0..=3 => (OpusMode::SilkOnly, Bandwidth::Narrow),
        4..=7 => (OpusMode::SilkOnly, Bandwidth::Medium),
        8..=11 => (OpusMode::SilkOnly, Bandwidth::Wide),
        12..=13 => (OpusMode::Hybrid, Bandwidth::SuperWide),
        14..=15 => (OpusMode::Hybrid, Bandwidth::Full),
        16..=19 => (OpusMode::CeltOnly, Bandwidth::Narrow),
        20..=23 => (OpusMode::CeltOnly, Bandwidth::Wide),
        24..=27 => (OpusMode::CeltOnly, Bandwidth::SuperWide),
        _ => (OpusMode::CeltOnly, Bandwidth::Full),
    }
}

/// 一个包的 TOC 解析结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Toc {
    pub config: u8,
    pub mode: OpusMode,
    pub bandwidth: Bandwidth,
    /// 每帧的样本数（48 kHz）。
    pub frame_size: u32,
    pub stereo: bool,
    /// 这个包里有多少帧。
    pub frames: u32,
}

impl Toc {
    /// `LM`：帧长相对 2.5 ms 的 2 的幂（CELT 用）。
    pub fn lm(self) -> u32 {
        (self.frame_size / SHORT_MDCT).trailing_zeros()
    }

    /// 数一数帧数（code 3 时要看第二个字节）。
    fn frames_of(toc: u8, packet: &[u8], frame_size: u32) -> Result<u32> {
        Ok(match toc & 0x03 {
            0 => 1,
            1 | 2 => 2,
            _ => {
                let n = (*packet.get(1).ok_or_else(|| {
                    MediaError::Bitstream("Opus: code 3 的包没有帧数字节".into())
                })? & 0x3F) as u32;
                // 一包最多 120 ms（2.5 ms × 48）
                if n == 0 || n * frame_size > 5760 {
                    return Err(MediaError::Bitstream(format!(
                        "Opus: 包里的帧数 {n} 非法（1..=48，且总长不超过 120 ms）"
                    )));
                }
                n
            }
        })
    }

    pub fn parse(packet: &[u8]) -> Result<Toc> {
        let toc = *packet
            .first()
            .ok_or_else(|| MediaError::Bitstream("Opus: 空包没有 TOC".into()))?;
        let config = toc >> 3;
        let (mode, bandwidth) = mode_and_bandwidth(config);
        let frame_size = config_frame_size(toc);
        Ok(Toc {
            config,
            mode,
            bandwidth,
            frame_size,
            stereo: toc & 0x04 != 0,
            frames: Self::frames_of(toc, packet, frame_size)?,
        })
    }
}

// ── 帧状态与解析结果 ──

/// CELT 解码器的跨帧状态。
///
/// 三样东西都是"上一帧留下的"：
///
/// - `old_ebands`：频带对数能量（粗能量做帧间预测的底子）；
/// - `old_log_e` / `old_log_e2`：再往前两帧的能量，只有 [`anti_collapse`] 用
///   （它靠"当前能量比历史高多少"判断该填多响的噪声）；
/// - `rng`：折叠/填噪用的 LCG 种子。参考实现**每帧末**把它设成区间解码器的最终
///   区间状态 —— 看着莫名其妙，但那正是"解码器与编码器用同一串随机数"的同步方式。
///
/// [`anti_collapse`]: super::bands::anti_collapse
pub struct CeltState {
    /// `old_ebands[ch][band]`：上一帧（量化后）的频带对数能量。
    pub old_ebands: [[f32; NB_EBANDS]; 2],
    /// `oldLogE`：上一帧的能量（`anti_collapse` 用）。
    pub old_log_e: [[f32; NB_EBANDS]; 2],
    /// `oldLogE2`：上上帧的能量（同上）。
    pub old_log_e2: [[f32; NB_EBANDS]; 2],
    /// `backgroundLogE`：长期背景能量（丢包隐藏的噪声填充用）。
    ///
    /// 每帧慢慢往上爬（`+= M * 0.001`）、永远不超过当前能量；只有非瞬态帧更新。
    /// 正常解码完全用不到它 —— 但没有它就没法做"丢了很久之后按背景噪声填"。
    pub background_log_e: [[f32; NB_EBANDS]; 2],
    /// 折叠/填噪的 LCG 种子。
    pub rng: u32,
}

impl Default for CeltState {
    fn default() -> Self {
        Self {
            old_ebands: [[0.0; NB_EBANDS]; 2],
            // 参考实现里历史能量从 -28 dB 起步（"极安静"），不是 0
            old_log_e: [[-28.0; NB_EBANDS]; 2],
            old_log_e2: [[-28.0; NB_EBANDS]; 2],
            // `OPUS_RESET_STATE` 把这一块整体清零（它不在 -28 的显式赋值里）
            background_log_e: [[0.0; NB_EBANDS]; 2],
            rng: 0,
        }
    }
}

impl CeltState {
    /// silence 帧处理完之后要把整张能量表置成 -28（参考实现如此，别改成 0）。
    fn reset_after_silence(&mut self, channels: usize) {
        for ch in 0..channels {
            self.old_ebands[ch] = [-28.0; NB_EBANDS];
        }
    }
}

/// 后置滤波器参数（解码出来的"逆滤波器"）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PostFilter {
    pub pitch: i32,
    pub gain: f32,
    pub tapset: u32,
}

/// 本轮的解析结果（帧头 + 粗能量 + TF + spread），供对拍与后续层使用。
#[derive(Debug, Clone, PartialEq)]
pub struct CeltFront {
    pub silence: bool,
    pub transient: bool,
    pub intra: bool,
    pub postfilter: PostFilter,
    /// 后置滤波器是否开启（`gain > 0`）。
    pub postfilter_on: bool,
    /// 每带的时频变化（TF 调整量）。
    pub tf: [i32; NB_EBANDS],
    /// 扩散决策（0..=3）。
    pub spread: u32,
    /// 粗能量（解码后的 `old_ebands`，每声道每带）。
    pub coarse_energy: [[f32; NB_EBANDS]; 2],
    /// 解析到 spread 之后的 `tell()`（后面比特分配全靠它）。
    pub tell: i32,
    pub start: usize,
    pub end: usize,
}

/// TOC → 每帧的样本数（48 kHz）。表里的数字直接是样本数（10 ms = 480、2.5 ms = 120），
/// 免得再乘一次 48 —— 之前就是这里乘错，把 2.5 ms 算成了 10 倍。
///
/// SILK/hybrid/CELT 三组的帧长集合不同，组内按 config 的低位取。
fn config_frame_size(toc: u8) -> u32 {
    let config = toc >> 3;
    let sizes: &[u32] = match config {
        0..=11 => &[480, 960, 1920, 2880],
        12..=15 => &[480, 960],
        _ => &[120, 240, 480, 960],
    };
    sizes[(config as usize) & (sizes.len() - 1)]
}

/// 读一个"帧长"字段：`1..=251` 是 1 字节；`252..=255` 表示还有第二字节，
/// 总长 = `second*4 + first`（RFC 6716 §3.2.1，**注意字节顺序**：第二字节在高位）。
fn parse_size(packet: &[u8], at: usize, remaining: usize) -> Result<(usize, usize)> {
    if remaining < 1 || at >= packet.len() {
        return Err(MediaError::Bitstream("Opus: 帧长字段被截断".into()));
    }
    let first = packet[at] as usize;
    if first < 252 {
        Ok((first, 1))
    } else if remaining < 2 {
        Err(MediaError::Bitstream("Opus: 两字节帧长字段被截断".into()))
    } else {
        Ok((4 * packet[at + 1] as usize + first, 2))
    }
}

/// 按 RFC 6716 §3.2.5 把包拆成各帧，返回每帧的（偏移, 长度）。
///
/// 这一段就是"一个包里装几帧、每帧多少字节"：code 0/1/2/3 四种打包方式，
/// 其中 code 3 还带 VBR 标志与填充计数。**CELT/SILK 解码器拿到的是帧**（不是包），
/// 所以必须先切开，交给解码器的字节里也不含 TOC 与这些打包字段。
pub fn frame_layout(packet: &[u8]) -> Result<Vec<(usize, usize)>> {
    let toc = *packet
        .first()
        .ok_or_else(|| MediaError::Bitstream("Opus: 空包".into()))?;
    let mut data = 1usize;
    let mut len = packet.len() - 1;
    let mut sizes: Vec<usize> = Vec::new();
    let count: usize;
    let last_size: usize;
    match toc & 0x03 {
        0 => {
            count = 1;
            last_size = len;
        }
        1 => {
            count = 2;
            if len % 2 != 0 {
                return Err(MediaError::Bitstream("Opus: 两帧 CBR 的载荷长度是奇数".into()));
            }
            sizes.push(len / 2);
            last_size = len / 2;
        }
        2 => {
            count = 2;
            let (sz, bytes) = parse_size(packet, data, len)?;
            data += bytes;
            len -= bytes;
            if sz > len {
                return Err(MediaError::Bitstream("Opus: 两帧 VBR 的第一帧长度越界".into()));
            }
            sizes.push(sz);
            last_size = len - sz;
        }
        _ => {
            if len < 1 {
                return Err(MediaError::Bitstream("Opus: code 3 的包没有计数字节".into()));
            }
            let ch = packet[data];
            data += 1;
            len -= 1;
            count = (ch & 0x3F) as usize;
            // 一包最多 120 ms（2.5 ms × 48）
            if count == 0 || count * (config_frame_size(toc) as usize) > 5760 {
                return Err(MediaError::Bitstream(format!(
                    "Opus: 包里的帧数 {count} 非法"
                )));
            }
            if ch & 0x40 != 0 {
                // 填充计数：每个 255 表示再加 254 字节
                let mut padding = 0usize;
                loop {
                    if len == 0 {
                        return Err(MediaError::Bitstream("Opus: 填充计数被截断".into()));
                    }
                    let p = packet[data];
                    data += 1;
                    len -= 1;
                    padding += if p == 255 { 254 } else { p as usize };
                    if p != 255 {
                        break;
                    }
                }
                if padding > len {
                    return Err(MediaError::Bitstream("Opus: 填充长度超过了包长".into()));
                }
                len -= padding;
            }
            // 第 7 位是"VBR 标志"：为 1 表示每帧长度单独写，为 0 表示各帧等长
            let vbr = ch & 0x80 != 0;
            if vbr {
                let mut rest = len as i64;
                for _ in 0..count - 1 {
                    let (sz, bytes) = parse_size(packet, data, len)?;
                    data += bytes;
                    len -= bytes;
                    if sz > len {
                        return Err(MediaError::Bitstream("Opus: VBR 帧长越界".into()));
                    }
                    sizes.push(sz);
                    rest -= (bytes + sz) as i64;
                }
                if rest < 0 {
                    return Err(MediaError::Bitstream("Opus: VBR 帧长之和超过了包长".into()));
                }
                last_size = rest as usize;
            } else {
                if count == 0 || len % count != 0 {
                    return Err(MediaError::Bitstream("Opus: CBR 帧长不能整除".into()));
                }
                last_size = len / count;
                for _ in 0..count - 1 {
                    sizes.push(last_size);
                }
            }
        }
    }
    if last_size > 1275 {
        return Err(MediaError::Bitstream(format!(
            "Opus: 最后一帧 {last_size} 字节，超过规范上限 1275"
        )));
    }
    sizes.push(last_size);
    debug_assert_eq!(sizes.len(), count);
    let mut out = Vec::with_capacity(count);
    let mut off = data;
    for sz in sizes {
        out.push((off, sz));
        off += sz;
    }
    Ok(out)
}

/// 取出包里第 `idx` 帧的**载荷**。
///
/// 注意：**TOC 与打包字段都不属于编解码器码流**（参考实现里 `opus_packet_parse_impl`
/// 先拆帧，再把 `data+offset` 交给 CELT/SILK 解码器）。区间解码器从载荷的第一个字节
/// 起算 —— 弄错一个字节，后面每个符号都会错位。
pub fn frame_payload(packet: &[u8], toc: Toc, idx: u32) -> Result<&[u8]> {
    let layout = frame_layout(packet)?;
    let (off, len) = *layout.get(idx as usize).ok_or_else(|| {
        MediaError::Bitstream(format!(
            "Opus: 要第 {idx} 帧，但这个包里只有 {} 帧",
            layout.len()
        ))
    })?;
    debug_assert_eq!(layout.len(), toc.frames as usize, "帧数与 TOC 记录的不一致");
    Ok(&packet[off..off + len])
}

/// 解出 CELT 帧的前半段（到 spread 为止），并更新跨帧状态。
///
/// `channels` 是**当前帧实际编码的声道数**（单声道帧即使流是立体声也要按 1 算）。
/// `start` / `end` 是起止带：CELT-only 从第 0 带起；**混合模式是 17**
/// （0–8 kHz 归 SILK、8 kHz 以上才归 CELT）。
pub fn decode_front(
    dec: &mut RangeDecoder<'_>,
    lm: u32,
    end: usize,
    start: usize,
    channels: usize,
    state: &mut CeltState,
) -> Result<CeltFront> {
    if channels == 0 || channels > 2 {
        return Err(MediaError::Bitstream(format!("Opus: 声道数 {channels} 非法")));
    }
    // 单声道帧：先把第二声道的能量并到第一声道（取 max）再解码。
    // 不这么做的话，"立体声 → 单声道 → 立体声"时第一声道的预测底子会缺一块。
    // （逐帧播种的对拍测试感受不到这一条：dump 里的帧前能量是**合并之后**的值。）
    if channels == 1 {
        for i in 0..NB_EBANDS {
            state.old_ebands[0][i] = state.old_ebands[0][i].max(state.old_ebands[1][i]);
        }
    }
    let total_bits = (dec.len_bytes() * 8) as i32;

    // ① silence：`tell == 1` 才是"还没解过任何符号"，这时才有这一位
    let mut tell = dec.tell();
    let silence = if tell >= total_bits {
        true
    } else if tell == 1 {
        dec.dec_bit_logp(15) == 1
    } else {
        false
    };
    if silence {
        // 假装整帧都读过了（否则后面那些"还剩多少位"的判断会以为还有富余）
        dec.force_tell(total_bits);
        tell = total_bits;
    }

    // ② 后置滤波器：**只有从第 0 带起编码时才有这一位**
    //    （混合模式下 CELT 只负责 8 kHz 以上，后置滤波是宽带操作，没它的事）
    let mut postfilter = PostFilter { pitch: 0, gain: 0.0, tapset: 0 };
    if start == 0 && tell + 16 <= total_bits {
        if dec.dec_bit_logp(1) == 1 {
            let octave = dec.dec_uint(6)?;
            let pitch = (16i32 << octave) + dec.dec_bits(4 + octave)? as i32 - 1;
            let qg = dec.dec_bits(3)? as i32;
            // 抽头集只在"还有余量"时才解（规范如此：它是可选的一位两符号）
            let tapset = if dec.tell() + 2 <= total_bits { dec.dec_icdf(&TAPSET_ICDF, 2)? } else { 0 };
            postfilter = PostFilter { pitch, gain: 0.09375 * (qg as f32 + 1.0), tapset };
        }
        tell = dec.tell();
    }

    // ③ 瞬态（LM=0 时没有这一位）
    let transient = if lm > 0 && tell + 3 <= total_bits {
        let t = dec.dec_bit_logp(3) == 1;
        tell = dec.tell();
        t
    } else {
        false
    };

    // ④ intra：能量不做帧间预测
    let intra = if tell + 3 <= total_bits { dec.dec_bit_logp(3) == 1 } else { false };

    // ⑤ 粗能量（会更新跨帧状态）。**注意**：对拍要的是"刚解完粗能量"的值 ——
    // 静音帧在帧尾会把整张表置成 -28，那是后面的事，别把这两件事混在一起。
    decode_coarse_energy(dec, start, end, channels, lm, intra, state)?;
    let coarse_energy = state.old_ebands;

    // ⑥ TF 变化
    let tf = tf_decode(dec, start, end, transient, lm);

    // ⑦ 扩散
    tell = dec.tell();
    let spread = if tell + 4 <= total_bits { dec.dec_icdf(&SPREAD_ICDF, 5)? } else { 2 };

    Ok(CeltFront {
        silence,
        transient,
        intra,
        postfilter_on: postfilter.gain > 0.0,
        postfilter,
        tf,
        spread,
        coarse_energy,
        tell: dec.tell(),
        start,
        end,
    })
}

/// 一帧**解码完之后**的状态更新（参考实现把这些散在帧尾几处，这里集中成一个函数）。
///
/// 两件事：
/// 1. **静音帧**要把整张能量表置成 -28（参考实现如此）。注意它发生在**帧尾**：
///    帧中间那些步骤（细能量、形状）看到的还是解码出来的值 —— 对拍时这一点很关键。
/// 2. **单声道帧**要把第一声道的能量回填给第二声道，否则"立体声 → 单声道 → 立体声"
///    时第二声道会停在很久以前的值上。
pub fn finish_frame(state: &mut CeltState, channels: usize, silence: bool) {
    if silence {
        state.reset_after_silence(channels);
    }
    if channels == 1 {
        state.old_ebands[1] = state.old_ebands[0];
    }
}

/// 帧尾**历史能量**的更新（参考实现 `celt_decode_with_ec` 收尾那几段）。
///
/// 两条规则：
///
/// - **非瞬态帧**：历史整体前推一格（`old_log_e2 ← old_log_e ← old_ebands`）；
/// - **瞬态帧**：只把历史调**低**（`old_log_e = min(old_log_e, old_ebands)`）——
///   因为瞬态之后能量会掉下来，如果照搬，`anti_collapse` 会以为"一直很响"
///   而填出过大的噪声。
///
/// 最后把 `[start, end)` 之外的能量清零并复位历史 —— 参考实现是为了"start/end
/// 会变"（比如带宽切换）时不留下陈旧值。
///
/// **调用顺序**：必须在静音帧复位（[`finish_frame`]）与帧内所有解码之后调用。
///
/// `lm` 只用来推进背景能量（`backgroundLogE += M * 0.001`，`M = 1<<lm`）——
/// 它只在非瞬态帧更新，正是"背景噪声电平"该有的慢变化。
pub fn update_history(state: &mut CeltState, transient: bool, start: usize, end: usize, lm: u32) {
    if !transient {
        state.old_log_e2 = state.old_log_e;
        state.old_log_e = state.old_ebands;
        let step = (1u32 << lm) as f32 * 0.001;
        for ch in 0..2 {
            for i in 0..NB_EBANDS {
                state.background_log_e[ch][i] =
                    (state.background_log_e[ch][i] + step).min(state.old_ebands[ch][i]);
            }
        }
    } else {
        for ch in 0..2 {
            for i in 0..NB_EBANDS {
                state.old_log_e[ch][i] = state.old_log_e[ch][i].min(state.old_ebands[ch][i]);
            }
        }
    }
    for ch in 0..2 {
        for i in 0..start {
            state.old_ebands[ch][i] = 0.0;
            state.old_log_e[ch][i] = -28.0;
            state.old_log_e2[ch][i] = -28.0;
        }
        for i in end..NB_EBANDS {
            state.old_ebands[ch][i] = 0.0;
            state.old_log_e[ch][i] = -28.0;
            state.old_log_e2[ch][i] = -28.0;
        }
    }
}

// ── 整链入口（CELT-only）──

/// CELT-only 的单帧解码器：把一帧载荷解成 PCM（i16 交错），状态跨帧自然传承。
///
/// 这是 CELT 整条链的接线处 —— [`decode_front`] → 比特分配 → 形状 → 合成，
/// 顺序与参考实现 `celt_decode_with_ec` 逐句对应。**状态自然传承**这一点很关键：
/// 之前的对拍都是逐帧播种（每帧从 dump 恢复帧前状态），只有顺序解码才会暴露
/// "传承语义"的错误（单声道帧的能量合并、静音帧的 -28 复位、LCG 种子的取回）。
///
/// 输出**固定 2 声道**（与 Opus 解码器一致）：单声道帧会复制到两个声道。
/// `pcm` 的长度必须是 `2 * (1<<lm) * 120`。SILK / 混合模式的帧会明确报 `Unsupported`。
pub struct CeltDecoder {
    /// 跨帧状态（能量、历史、LCG 种子）。解完一帧后 `state.rng` 就是该帧的**最终区间状态**
    /// （RFC §6.1 要求它与编码器返回的值一致）。
    pub state: CeltState,
    /// 合成状态（重叠缓冲、后置滤波记忆、去加重记忆、PLC 状态）。
    synth: SynthState,
    mdct: Mdct,
    /// 当前起带（参考实现 `st->start`，由 `CELT_SET_START_BAND` 设置）。
    ///
    /// 它**不是**"这一帧的起带"那么简单：丢包隐藏要按它选路径（`start != 0` 走噪声填充），
    /// 而参考实现是在 `opus_decode_frame` 的**不同位置**改它的（SILK-only 的帧也会被设成 17），
    /// 所以这里单独存一份、由调用方在对应位置显式设。
    start: usize,
    /// 当前结束带（参考实现 `st->end`，由 `CELT_SET_END_BAND` 设置）。
    end: usize,
}

impl Default for CeltDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl CeltDecoder {
    pub fn new() -> Self {
        Self {
            state: CeltState::default(),
            synth: SynthState::default(),
            mdct: Mdct::new_48k(),
            // `celt_decoder_init`：`st->start = 0; st->end = st->mode->nbEBands;`
            start: 0,
            end: NB_EBANDS,
        }
    }

    /// 复位（对应参考实现的 `OPUS_RESET_STATE`）。
    ///
    /// 模式切换（CELT ↔ 非 CELT）时参考实现会整体清一遍状态 —— 注意它清的是
    /// "从 `rng` 往后"的全部字段，**包括合成的重叠历史、去加重记忆与 PLC 的 `lpc`
    /// / `loss_count` / `last_pitch_index`**，所以这里直接重建，`old_log_e` 由
    /// `CeltState::default()` 置成 -28（不是 0）。`start`/`end` **不在**被清之列
    /// （参考实现的 `ENCODER_RESET_START` 在它们之前）。
    pub fn reset(&mut self) {
        let (start, end) = (self.start, self.end);
        *self = Self::new();
        self.start = start;
        self.end = end;
    }

    /// `CELT_SET_START_BAND`（参考实现里校验 `0 <= v < nbEBands`）。
    pub fn set_start_band(&mut self, start: usize) {
        debug_assert!(start < NB_EBANDS);
        self.start = start;
    }

    /// `CELT_SET_END_BAND`（参考实现里校验 `1 <= v <= nbEBands`）。
    pub fn set_end_band(&mut self, end: usize) {
        debug_assert!(end >= 1 && end <= NB_EBANDS);
        self.end = end;
    }

    /// 解一帧 CELT-only 的帧：自己开一个区间解码器，从第 0 带起。
    pub fn decode_frame(&mut self, payload: &[u8], toc: Toc, pcm: &mut [i16]) -> Result<()> {
        let channels = if toc.stereo { 2 } else { 1 };
        self.set_start_band(0);
        self.set_end_band(toc.bandwidth.end_band());
        let mut dec = RangeDecoder::new(payload);
        self.decode_stream(&mut dec, payload.len(), channels, toc.lm(), pcm)
    }

    /// 解一帧"丢包"（对应参考实现 `celt_decode_lost`）。
    ///
    /// `lm` 决定帧长（`N = (1<<lm) * 120`），输出 `2*N` 个 i16。
    pub fn decode_lost(&mut self, lm: u32, pcm: &mut [i16]) -> Result<()> {
        let n = (1usize << lm) * SHORT_MDCT as usize;
        if pcm.len() < 2 * n {
            return Err(MediaError::Bitstream(format!(
                "Opus: 单帧解码需要 {} 个 i16 输出，给了 {}",
                2 * n,
                pcm.len()
            )));
        }
        celt_plc::celt_decode_lost(
            &mut self.state,
            &mut self.synth,
            &self.mdct,
            self.start,
            self.end,
            n,
            lm,
            &mut pcm[..2 * n],
        );
        Ok(())
    }

    /// 用**外部**区间解码器解一帧 CELT（混合模式与冗余帧都要走这条）。
    ///
    /// `dec` 是"这一帧的比特流"上的解码器；混合模式下它已经被 SILK 用过一段，
    /// 所以 `total_bits` 与 `tell` 的比较口径必须用整帧的字节数 `len` ——
    /// 这正是参考实现把 `ec_dec*` 传进来的原因（CELT 的 silence 判定看的是
    /// "整帧还剩多少位"，不是"CELT 那一段还剩多少位"）。
    ///
    /// 起止带取自 [`CeltDecoder::set_start_band`] / [`CeltDecoder::set_end_band`]
    /// 设下的当前值（与参考实现的 `st->start` / `st->end` 一致）；
    /// `channels` 是**流**的声道数（TOC 的 s 位），`lm` 是帧长指数。
    pub fn decode_stream(
        &mut self,
        dec: &mut RangeDecoder<'_>,
        len: usize,
        channels: usize,
        lm: u32,
        pcm: &mut [i16],
    ) -> Result<()> {
        let (start, end) = (self.start, self.end);
        let m = 1usize << lm;
        let n = m * SHORT_MDCT as usize;
        if pcm.len() < 2 * n {
            return Err(MediaError::Bitstream(format!(
                "Opus: 单帧解码需要 {} 个 i16 输出，给了 {}",
                2 * n,
                pcm.len()
            )));
        }
        if len <= 1 {
            // 参考实现 `celt_decode_with_ec`：`data == NULL || len<=1` → `celt_decode_lost`
            return self.decode_lost(lm, pcm);
        }
        let pcm = &mut pcm[..2 * n];

        let front = decode_front(dec, lm, end, start, channels, &mut self.state)?;
        let alloc = rate::decode_allocation(dec, lm, channels, front.transient, start, end)?;
        bands::unquant_fine_energy(dec, &alloc, channels, start, end, &mut self.state)?;

        let total_bits = len as i32 * (8 << rate::BITRES) - alloc.anti_collapse_rsv;
        let shape_in = ShapeInput {
            pulses: &alloc.pulses,
            tf: &front.tf,
            start,
            end,
            lm,
            transient: front.transient,
            spread: front.spread,
            dual_stereo: alloc.dual_stereo,
            intensity: alloc.intensity,
            coded_bands: alloc.coded_bands,
            total_bits,
            balance: alloc.balance,
            seed: self.state.rng,
        };
        let mut shape = bands::quant_all_bands(dec, &shape_in, channels)?;

        let ac_on = if alloc.anti_collapse_rsv > 0 {
            dec.dec_bits(1)? != 0
        } else {
            false
        };
        let bits_left = len as i32 * 8 - dec.tell();
        bands::unquant_energy_finalise(dec, &alloc, channels, start, end, bits_left, &mut self.state)?;
        if ac_on {
            bands::anti_collapse(
                &mut shape.x,
                &shape.collapse_masks,
                lm,
                channels,
                start,
                end,
                &self.state.old_ebands,
                &self.state.old_log_e,
                &self.state.old_log_e2,
                &alloc.pulses,
                shape.seed,
            );
        }

        // 能量：对数 → 线性。静音帧要在合成**之前**把 bandE 清 0（dump 的 BAND 行是清零前的值，
        // 逐帧播种的对拍测试在这里要自己补这一刀）。
        let mut band_e = [[0f32; NB_EBANDS]; 2];
        bands::log2_amp(&mut band_e, &self.state.old_ebands, start, end, channels);
        if front.silence {
            for e in band_e.iter_mut().take(channels) {
                *e = [0.0; NB_EBANDS];
            }
        }
        finish_frame(&mut self.state, channels, front.silence);

        let freq = bands::denormalise_bands(&shape.x, &band_e, end, channels, m);
        let synth_in = SynthInput {
            stream_channels: channels,
            out_channels: 2,
            lm,
            transient: front.transient,
            start,
            end,
            postfilter: front.postfilter,
            freq: &freq,
        };
        synth::synth_frame(&mut self.synth, &self.mdct, &synth_in, pcm);

        // 帧尾：历史能量前推 + 区间状态取回（参考实现那句 `st->rng = dec->rng`）
        update_history(&mut self.state, front.transient, start, end, lm);
        self.state.rng = dec.state().1;
        // 正常解出一帧 → 连续丢帧计数归零（参考实现 `st->loss_count = 0` 在收尾处）
        self.synth.reset_loss_count();
        Ok(())
    }

    /// 2.5 ms 的"静音帧"（参考实现里 hybrid → SILK 过渡时用来做淡出的那两字节 `FF FF`）。
    ///
    /// 结束带跟**当前包**的带宽一致（参考实现是先 `CELT_SET_END_BAND(endband)` 再解这个
    /// 静音帧、并在解之前把它 `CELT_SET_START_BAND(0)` —— 所以解完之后 `st->start`
    /// 会停在 0，后面的丢包隐藏就按"不带限"走）。
    pub fn decode_silence_2ms5(&mut self, end: usize, pcm: &mut [i16]) -> Result<()> {
        self.set_start_band(0);
        self.set_end_band(end);
        let silence = [0xFFu8, 0xFF];
        let mut dec = RangeDecoder::new(&silence);
        self.decode_stream(&mut dec, 2, 2, 0, pcm)
    }

    /// 帧长对应的 `lm`（参考实现 `celt_decode_with_ec` 开头那段"从 frame_size 反推 LM"）。
    pub fn lm_for_frame_size(frame_size: usize) -> Result<u32> {
        let mut lm = 0u32;
        while lm <= 3 {
            if (SHORT_MDCT as usize) << lm == frame_size {
                return Ok(lm);
            }
            lm += 1;
        }
        Err(MediaError::Bitstream(format!(
            "Opus: CELT 帧长 {frame_size} 不是 2.5 ms 的整数倍"
        )))
    }

    /// 模式的窗表（`mode->window`，长度 = `overlap`）。
    ///
    /// Opus 的 `smooth_fade` 直接用它（取平方当权重），所以必须与合成用的是同一张表。
    pub fn window(&self) -> &[f32] {
        self.mdct.window()
    }

    /// 连续丢帧计数（`st->loss_count`，测试用）。
    pub fn loss_count(&self) -> i32 {
        self.synth.loss_count
    }
}

/// 粗能量解码（参考实现 `unquant_coarse_energy`）。
fn decode_coarse_energy(
    dec: &mut RangeDecoder<'_>,
    start: usize,
    end: usize,
    channels: usize,
    lm: u32,
    intra: bool,
    state: &mut CeltState,
) -> Result<()> {
    let prob_model = &E_PROB_MODEL[lm as usize][usize::from(intra)];
    let (coef, beta) = if intra {
        (0.0, BETA_INTRA)
    } else {
        (PRED_COEF[lm as usize], BETA_COEF[lm as usize])
    };
    let budget = (dec.len_bytes() * 8) as i32;
    let mut prev = [0f32; 2];
    for i in start..end {
        for ch in 0..channels {
            let tell = dec.tell();
            let qi: i32 = if budget - tell >= 15 {
                let pi = 2 * i.min(20);
                laplace_decode(dec, (prob_model[pi] as u32) << 7, (prob_model[pi + 1] as i32) << 6)?
            } else if budget - tell >= 2 {
                // 2 位 icdf + 符号折叠（`qi = (qi>>1) ^ -(qi&1)`）
                let v = dec.dec_icdf(&SMALL_ENERGY_ICDF, 2)? as i32;
                (v >> 1) ^ -(v & 1)
            } else if budget - tell >= 1 {
                -(dec.dec_bit_logp(1) as i32)
            } else {
                -1
            };
            let q = qi as f32;
            // 预测项：`coef*max(-9, 上一帧)` + 频率方向的记忆 + 本次的粗量化值
            let e = state.old_ebands[ch][i].max(-9.0);
            state.old_ebands[ch][i] = coef * e + prev[ch] + q;
            prev[ch] += q * (1.0 - beta);
        }
    }
    Ok(())
}

/// 时频变化解码（参考实现 `tf_decode`）。
///
/// 每带一位 logp 符号（"相对于上一带是否翻转"），最后按 `tf_select` 查表换算成调整量。
/// `tf_select` 只在"它真的会改变结果"时才解码 —— 这是编码器与解码器都必须一致的一点。
fn tf_decode(
    dec: &mut RangeDecoder<'_>,
    start: usize,
    end: usize,
    transient: bool,
    lm: u32,
) -> [i32; NB_EBANDS] {
    let mut tf = [0i32; NB_EBANDS];
    let budget_full = (dec.len_bytes() * 8) as i32;
    let mut tell = dec.tell();
    let mut logp = if transient { 2u32 } else { 4u32 };
    let tf_select_rsv = i32::from(lm > 0 && tell + logp as i32 + 1 <= budget_full);
    let budget = budget_full - tf_select_rsv;
    let mut tf_changed = 0i32;
    let mut curr = 0i32;
    for i in start..end {
        if tell + logp as i32 <= budget {
            curr ^= dec.dec_bit_logp(logp) as i32;
            tell = dec.tell();
            tf_changed |= curr;
        }
        tf[i] = curr;
        logp = if transient { 4 } else { 5 };
    }
    let t = usize::from(transient);
    let mut tf_select = 0usize;
    if tf_select_rsv == 1
        && TF_SELECT_TABLE[lm as usize][4 * t + tf_changed as usize]
            != TF_SELECT_TABLE[lm as usize][4 * t + 2 + tf_changed as usize]
    {
        tf_select = dec.dec_bit_logp(1) as usize;
    }
    for i in start..end {
        tf[i] = TF_SELECT_TABLE[lm as usize][4 * t + 2 * tf_select + tf[i] as usize] as i32;
    }
    tf
}

// ── Laplace 解码（参考实现 laplace.c）──

const LAPLACE_LOG_MINP: u32 = 0;
const LAPLACE_MINP: u32 = 1 << LAPLACE_LOG_MINP;
const LAPLACE_NMIN: u32 = 16;

/// 衰减段下一个频段的大小。
fn laplace_get_freq1(fs0: u32, decay: i32) -> u32 {
    let ft = 32768 - LAPLACE_MINP * (2 * LAPLACE_NMIN) - fs0;
    ((ft as i32 * (16384 - decay)) >> 15) as u32
}

/// 解一个"Laplace 分布"的整数（粗能量的预测误差用的就是它）。
///
/// `fs` 是"取 0 的概率"（Q8 域的 32768 分之一），`decay` 是尾部衰减率。
pub fn laplace_decode(dec: &mut RangeDecoder<'_>, fs0: u32, decay: i32) -> Result<i32> {
    let mut val = 0i32;
    let mut fl = 0u32;
    let fm = dec.decode_bin(15);
    let mut fs = fs0;
    if fm >= fs {
        val += 1;
        fl = fs;
        fs = laplace_get_freq1(fs, decay) + LAPLACE_MINP;
        // 尾部：每一段的宽度按 decay 递减，直到比最小概率还小
        while fs > LAPLACE_MINP && fm >= fl + 2 * fs {
            fs *= 2;
            fl += fs;
            fs = (((fs - 2 * LAPLACE_MINP) as i32 * decay) >> 15) as u32 + LAPLACE_MINP;
            val += 1;
        }
        // 再往外是等概率的尾巴，直接用除法算偏移
        if fs <= LAPLACE_MINP {
            let di = (fm - fl) >> (LAPLACE_LOG_MINP + 1);
            val += di as i32;
            fl += 2 * di * LAPLACE_MINP;
        }
        if fm < fl + fs {
            val = -val; // 负半边
        } else {
            fl += fs;
        }
    }
    dec.update(fl, (fl + fs).min(32768), 32768);
    Ok(val)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toc_of(config: u8, stereo: bool) -> u8 {
        (config << 3) | if stereo { 0x04 } else { 0 } | 1 // c=1 → 两帧等长（够用即可）
    }

    /// CELT（和混合）的帧长是 120 样本的 2 的幂倍；SILK 的 40/60 ms 不是（也没有 LM 概念）。
    fn frame_size_is_celt_like(size: u32) -> bool {
        size >= SHORT_MDCT && size.count_ones() == 1
    }

    #[test]
    fn parses_all_32_configs() {
        // RFC 6716 Table 2 的模式/带宽/帧长（每 4 个 config 一组，组内按顺序取帧长）
        for config in 0u8..32 {
            let toc = Toc::parse(&[toc_of(config, false), 0]).unwrap();
            let (mode, bw, size) = match config {
                0..=3 => (OpusMode::SilkOnly, Bandwidth::Narrow, [480u32, 960, 1920, 2880][config as usize]),
                4..=7 => (OpusMode::SilkOnly, Bandwidth::Medium, [480, 960, 1920, 2880][(config - 4) as usize]),
                8..=11 => (OpusMode::SilkOnly, Bandwidth::Wide, [480, 960, 1920, 2880][(config - 8) as usize]),
                12..=13 => (OpusMode::Hybrid, Bandwidth::SuperWide, [480, 960][(config - 12) as usize]),
                14..=15 => (OpusMode::Hybrid, Bandwidth::Full, [480, 960][(config - 14) as usize]),
                16..=19 => (OpusMode::CeltOnly, Bandwidth::Narrow, [120, 240, 480, 960][(config - 16) as usize]),
                20..=23 => (OpusMode::CeltOnly, Bandwidth::Wide, [120, 240, 480, 960][(config - 20) as usize]),
                24..=27 => (OpusMode::CeltOnly, Bandwidth::SuperWide, [120, 240, 480, 960][(config - 24) as usize]),
                _ => (OpusMode::CeltOnly, Bandwidth::Full, [120, 240, 480, 960][(config - 28) as usize]),
            };
            assert_eq!(toc.config, config);
            assert_eq!(toc.mode, mode, "config {config} 的模式");
            assert_eq!(toc.bandwidth, bw, "config {config} 的带宽");
            assert_eq!(toc.frame_size, size, "config {config} 的帧长");
            assert_eq!(toc.frames, 2, "c=1 → 两帧");
            // LM 只对 CELT（和混合模式）有意义：帧长必须是 120 样本的 2 的幂倍
            if frame_size_is_celt_like(size) {
                assert_eq!(toc.frame_size, SHORT_MDCT << toc.lm(), "config {config} 的 LM");
            }
        }
        // s 位：单声道/立体声
        assert!(!Toc::parse(&[toc_of(16, false), 0]).unwrap().stereo);
        assert!(Toc::parse(&[toc_of(16, true), 0]).unwrap().stereo);
    }

    #[test]
    fn toc_frame_counts() {
        // c=0 → 1 帧；c=1/2 → 2 帧；c=3 → 第二个字节低 6 位
        assert_eq!(Toc::parse(&[16 << 3, 0]).unwrap().frames, 1);
        assert_eq!(Toc::parse(&[(16 << 3) | 1, 0]).unwrap().frames, 2);
        assert_eq!(Toc::parse(&[(16 << 3) | 2, 0]).unwrap().frames, 2);
        assert_eq!(Toc::parse(&[(16 << 3) | 3, 7]).unwrap().frames, 7);
        // 非法帧数
        assert!(Toc::parse(&[(16 << 3) | 3, 0]).is_err());
        assert!(Toc::parse(&[(16 << 3) | 3, 49]).is_err());
        assert!(Toc::parse(&[]).is_err(), "空包没有 TOC");
    }

    #[test]
    fn band_ends_follow_bandwidth() {
        assert_eq!(Bandwidth::Narrow.end_band(), 13);
        assert_eq!(Bandwidth::Medium.end_band(), 17);
        assert_eq!(Bandwidth::Wide.end_band(), 17);
        assert_eq!(Bandwidth::SuperWide.end_band(), 19);
        assert_eq!(Bandwidth::Full.end_band(), 21);
    }

    #[test]
    fn silence_frame_quantises_energy_to_minus_one_per_band() {
        // 位数耗尽时（静音帧就是这样）每个带的粗量化值都是 -1，
        // 于是能量按预测公式衰减：e[i] = coef*max(-9, e_prev) + prev + (-1)
        // 20 ms（LM=3）：coef = 0.5、beta = 0.2 → 首帧是 -1, -1.8, -2.6 …
        let data = [0u8; 2];
        let mut dec = RangeDecoder::new(&data);
        // 静音帧的处理：先把 tell 抬到帧尾，这样"还剩多少位"才和参考实现一致
        dec.force_tell(16);
        let mut state = CeltState::default();
        decode_coarse_energy(&mut dec, 0, 4, 1, 3, false, &mut state).unwrap();
        for (i, want) in [-1.0f32, -1.8, -2.6, -3.4].iter().enumerate() {
            assert!(
                (state.old_ebands[0][i] - want).abs() < 1e-4,
                "第 {i} 带：得到 {}，期望 {want}",
                state.old_ebands[0][i]
            );
        }
        // 帧尾的静音处理：整张能量表被置成 -28（参考实现如此）——
        // 所以下一帧从 -9（钳位后）起算，得到 -5.5, -6.3 …
        state.reset_after_silence(1);
        decode_coarse_energy(&mut dec, 0, 3, 1, 3, false, &mut state).unwrap();
        assert!((state.old_ebands[0][0] + 5.5).abs() < 1e-4, "{}", state.old_ebands[0][0]);
        assert!((state.old_ebands[0][1] + 6.3).abs() < 1e-4);
    }

    #[test]
    fn energy_prediction_uses_the_previous_frame_when_not_reset() {
        // 不重置时（普通帧）预测直接从上一帧的能量出发：e = 0.5*(-1) + 0 + (-1) = -1.5
        let data = [0u8; 2];
        let mut dec = RangeDecoder::new(&data);
        dec.force_tell(16);
        let mut state = CeltState::default();
        decode_coarse_energy(&mut dec, 0, 1, 1, 3, false, &mut state).unwrap();
        assert!((state.old_ebands[0][0] + 1.0).abs() < 1e-4);
        decode_coarse_energy(&mut dec, 0, 1, 1, 3, false, &mut state).unwrap();
        assert!(
            (state.old_ebands[0][0] + 1.5).abs() < 1e-4,
            "不重置时应从上一帧继续预测：{}",
            state.old_ebands[0][0]
        );
    }

    #[test]
    fn silence_resets_the_energy_table_to_minus_28() {
        let mut state = CeltState::default();
        state.old_ebands[0][3] = 12.0;
        state.old_ebands[1][3] = 12.0;
        state.reset_after_silence(2);
        assert!(state.old_ebands[0].iter().all(|&v| v == -28.0));
        assert!(state.old_ebands[1].iter().all(|&v| v == -28.0));
    }

    #[test]
    fn tiny_payloads_go_through_the_packet_loss_concealer() {
        // 参考实现里 `len <= 1` 一律走 `celt_decode_lost`（丢包隐藏）。
        // 这里只钉"确实走了 PLC"这一个事实（数值由 PLC 的对拍测试负责）：
        // 没解过任何帧 → 历史全零 → 隐藏出来的也是全零，但 `loss_count` 必须自增。
        let toc = Toc::parse(&[0xFC]).unwrap(); // config 31：CELT-only / 全带 / 20 ms / 立体声
        let mut dec = CeltDecoder::new();
        let mut pcm = vec![0i16; 2 * 960];
        dec.decode_frame(&[], toc, &mut pcm).unwrap();
        assert_eq!(dec.loss_count(), 1);
        assert!(pcm.iter().all(|&v| v == 0), "历史全零时隐藏不出东西");
        dec.decode_frame(&[0u8], toc, &mut pcm).unwrap();
        assert_eq!(dec.loss_count(), 2, "一字节载荷同样算丢包");
    }

    #[test]
    fn a_silence_frame_zeroes_the_energy_table() {
        // 2.5 ms 的静音帧（两字节 FF FF）—— 参考实现里 hybrid → SILK 的过渡用它做淡出。
        let mut dec = CeltDecoder::new();
        let mut pcm = vec![0i16; 2 * 120];
        dec.decode_silence_2ms5(21, &mut pcm).unwrap();
        assert!(dec.state.old_ebands[0].iter().all(|&v| v == -28.0), "静音帧后能量表应置 -28");
        assert!(dec.state.old_ebands[1].iter().all(|&v| v == -28.0));
    }

    #[test]
    fn frame_decoding_checks_the_output_length() {
        let toc = Toc::parse(&[0xFC]).unwrap();
        let mut dec = CeltDecoder::new();
        let err = dec.decode_frame(&[], toc, &mut vec![0i16; 100]).unwrap_err();
        assert!(matches!(err, MediaError::Bitstream(_)), "{err:?}");
    }

    #[test]
    fn lm_for_frame_size_maps_the_standard_lengths() {
        // CELT 的帧长只能是 2.5 ms 的整数倍（120/240/480/960 @48 kHz）
        assert_eq!(CeltDecoder::lm_for_frame_size(120).unwrap(), 0);
        assert_eq!(CeltDecoder::lm_for_frame_size(240).unwrap(), 1);
        assert_eq!(CeltDecoder::lm_for_frame_size(480).unwrap(), 2);
        assert_eq!(CeltDecoder::lm_for_frame_size(960).unwrap(), 3);
        assert!(CeltDecoder::lm_for_frame_size(300).is_err());
    }

    #[test]
    fn hybrid_frames_skip_the_postfilter_flag() {
        // 混合模式下 CELT 只负责 8 kHz 以上（`start == 17`），而后置滤波是宽带操作 ——
        // 参考实现用 `st->start==0` 把这一位挡掉。这里用同一串字节在两种 start 下各解一次：
        // start=17 不可能比 start=0 读得多（少读一位标志、少读它的参数）。
        let data = [0x5Au8; 16];
        let mut tells = [0i32; 2];
        for (i, start) in [0usize, 17].into_iter().enumerate() {
            let mut dec = RangeDecoder::new(&data);
            let mut state = CeltState::default();
            let front = decode_front(&mut dec, 3, 21, start, 2, &mut state).unwrap();
            tells[i] = front.tell;
        }
        assert!(
            tells[1] <= tells[0],
            "start=17 应当不比 start=0 读得多：{} vs {}",
            tells[1],
            tells[0]
        );
    }

    #[test]
    fn splits_multi_frame_packets() {
        // code 0：单帧
        let p = vec![16 << 3, 0xAA, 0xBB, 0xCC];
        assert_eq!(frame_layout(&p).unwrap(), vec![(1, 3)]);

        // code 1：两帧 CBR（载荷平分）
        let p = vec![(16 << 3) | 1, 1, 2, 3, 4];
        assert_eq!(frame_layout(&p).unwrap(), vec![(1, 2), (3, 2)]);
        // 奇数长度 → 不合法
        assert!(frame_layout(&[(16 << 3) | 1, 1, 2, 3]).is_err());

        // code 2：两帧 VBR（第一帧长度写在前面，1 字节形式）
        let p = vec![(16 << 3) | 2, 3, 1, 2, 3, 7, 7];
        assert_eq!(frame_layout(&p).unwrap(), vec![(2, 3), (5, 2)]);
        // 两字节形式：first=255、second=63 → 4*63+255 = 507（真实码流里就是这么编的）
        let mut p = vec![(16 << 3) | 2, 255, 63];
        p.resize(2 + 2 + 507 + 5, 0);
        let layout = frame_layout(&p).unwrap();
        assert_eq!(layout[0], (3, 507), "TOC(1) + 两字节帧长(2) 之后才是第一帧");
        assert_eq!(layout[1].1, p.len() - 3 - 507);

        // code 3 CBR：计数 3、VBR 位为 0 → 三帧平分
        let p = vec![(16 << 3) | 3, 3, 1, 2, 3, 4, 5, 6];
        assert_eq!(frame_layout(&p).unwrap(), vec![(2, 2), (4, 2), (6, 2)]);

        // code 3 VBR：计数 2、VBR 位为 1 → 第一帧长度 3（1 字节形式），其余 2
        let p = vec![(16 << 3) | 3, 0x80 | 2, 3, 1, 2, 3, 4, 5];
        assert_eq!(frame_layout(&p).unwrap(), vec![(3, 3), (6, 2)]);

        // code 3 + 填充：填充计数 1（=1 字节）
        let p = vec![(16 << 3) | 3, 0x40 | 2, 1, 1, 2, 3, 4, 0];
        assert_eq!(frame_layout(&p).unwrap(), vec![(3, 2), (5, 2)]);
    }

    #[test]
    fn rejects_impossible_frame_layouts() {
        // 帧数 0
        assert!(frame_layout(&[(16 << 3) | 3, 0]).is_err());
        // 帧数过多（2.5 ms × 49 > 120 ms）
        assert!(frame_layout(&[16 << 3 | 3, 49]).is_err());
        // 空包
        assert!(frame_layout(&[]).is_err());
        // VBR 帧长之和超过包长
        assert!(frame_layout(&[(16 << 3) | 3, 0x80 | 2, 200, 1, 2, 3]).is_err());
        // 两字节形式的帧长（507）超过包长
        assert!(frame_layout(&[(16 << 3) | 2, 255, 63, 0, 0]).is_err());
        // 最后一帧超过 1275 字节
        let mut big = vec![16 << 3];
        big.resize(1300, 0);
        assert!(frame_layout(&big).is_err());
    }

    #[test]
    fn laplace_decode_stays_in_range_and_updates_state() {
        // 随便一段字节：解出来的值应当有正有负、且区间状态合法推进
        let data: Vec<u8> = (0..64u16).map(|i| ((i * 89 + 13) % 256) as u8).collect();
        let mut dec = RangeDecoder::new(&data);
        let mut seen_positive = false;
        let mut seen_negative = false;
        for _ in 0..40 {
            let v = laplace_decode(&mut dec, 72u32 << 7, 127 << 6).unwrap();
            assert!(v.abs() < 1000, "解出来的值不该离谱：{v}");
            seen_positive |= v > 0;
            seen_negative |= v < 0;
            let (_, rng) = dec.state();
            assert!(rng > 0);
        }
        assert!(seen_positive && seen_negative, "两个方向都该出现");
    }

    #[test]
    fn tf_table_applies_select_and_transient() {
        // LM=3、非瞬态：tf_select=0 一组、tf_select=1 一组（照参考实现的表）
        assert_eq!(TF_SELECT_TABLE[3][0], 0);
        assert_eq!(TF_SELECT_TABLE[3][4], 3, "非瞬态 + tf_select=1 + 变化 0");
        assert_eq!(TF_SELECT_TABLE[3][5], 0);
        assert_eq!(TF_SELECT_TABLE[3][6], 1);
        assert_eq!(TF_SELECT_TABLE[3][7], -1);
        // 瞬态（isTransient=1）时同下标取另一组
        assert_eq!(TF_SELECT_TABLE[3][1], -2);
        assert_eq!(TF_SELECT_TABLE[3][3], -3);
    }
}
