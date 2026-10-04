//! **自研 AAC-LC 解码器**。
//!
//! ## 已经就绪并单测通过的部分
//!
//! | 部分 | 位置 | 状态 |
//! |---|---|---|
//! | 标准码表（11 张谱表 + 比例因子表 + 四元组表 + sfb 频带表） | [`tables`] | ✅ 含"前缀码"自校验 |
//! | Huffman 解码与取值映射（四元组/成对/转义） | [`hcb`] | ✅ 逐项遍历全表验证 |
//! | 滤波器组（FFT、IMDCT、正弦/KBD 窗） | [`imdct`] | ✅ 与 O(N²) 直接公式互验 |
//! | 解码器配置（AudioSpecificConfig）解析 | 本文件 | ✅ |
//! | ADTS 头剥离、`raw_data_block` 逐句法解析 | 本文件 | ✅ |
//! | ICS（窗信息/分段/比例因子/脉冲/谱数据）、TNS | 本文件 | ✅ |
//! | PNS、M/S 与强度立体声 | 本文件 | ✅ |
//! | IMDCT → 加窗 → 跨帧叠加 → 交织 PCM | 本文件 | ✅ |
//!
//! 端到端验证：用系统 `afconvert` 编出的 AAC 与参考解码结果比 —— 对齐编码器的
//! 2112 样本预热后，**稳态帧逐样本最大差 ≤ ±110**、整段 rms 差 0.1%~0.8%
//! （甲乙两个独立解码器之间的正常差距；`tone` 那条我们与**源**的差是 rms 0.828%，
//! 系统解码器与源的差是 0.823% —— 同一水平）。
//!
//! **验证口径要按"逐帧 rms + 最大差"来**，不能只看"逐频率分量误差 <1%"：
//! 2026-09-27 就是靠逐帧对比才抓到"瞬态处整段 448 个样本被清零"这种错
//! （频谱上不显眼、听感上像卡了一下）。滤波器组本身另有一条**完美重建**自校验
//! 单测（`filterbank_*`，纯合成频谱、不依赖样例文件），改动窗口几何时会当场变红。
//!
//! ## 还没接的部分（有意不做，会给出明确的 Unsupported）
//!
//! SBR/PS（HE-AAC）、960 点帧、CCE（耦合声道）、PCE、Main profile 的长时预测、
//! 增益控制（SSR）、4.0 声道以上。这些都**明确报错**，不会产出听起来像对、
//! 实际是噪声的音频。
//!
//! ## 几个已经把坑写下来的地方（实现时按这个来）
//!
//! 1. **短窗序列的时域位置**：8 个 256 点短块叠加后放在 2048 点缓冲的
//!    **448 偏移**处（每块间隔 128）—— 不是从 0 开始。放错声音会抖、长/短窗切换处爆音。
//!    （它带来的那两段"没有短块供能"的区间由过渡窗负责，见第 4 条。）
//! 2. **比例因子是从 `global_gain` 起的差分**：第一个比例因子取 global_gain，
//!    之后每个都是"上一个 + 差分"，**跨窗组连续**；每组重新起算会整体走音。
//! 3. **`sect_cb` 没有"码表转义"**：15 是合法的强度立体声码表（`INTENSITY_HCB`），
//!    4 位里只有 12 是保留值。把它当转义读会让"短窗 + 立体声"的帧整体错位。
//! 4. **长↔短窗过渡窗的"平坦段"必须是 1.0**（2026-09-27 修的真 bug，就是本文件
//!    `seq_window`）：短序列只占 `[448,1600)`，所以 `LONG_START` 的 `[1024,1472)` 与
//!    `LONG_STOP` 的 `[576,1024)` 这两段**没有任何短块提供能量**，必须由这一帧把信号
//!    原样送过去（窗口值 1.0）；写成 0 的表现是**固定 448 个样本被清零** ——
//!    正弦的开头与结尾各静一小段，听感像卡了一下，而"逐频率分量误差"这种统计口径
//!    看不出来。同理 `LONG_START` 的尾巴 `[1472,1600)` 要用短窗的**下降段**
//!    （短窗是对称的，写成 `short[255-i]` 等于用了上升段，一样会砸掉 448 个样本）。

mod hcb;
mod imdct;
mod tables;

use me_core::{MediaError, Result};

use super::bitreader::BitReader;
use super::{AudioDecoder, PcmChunk};
use hcb::Hcbs;
use imdct::WindowShape;

/// 规范里的采样率表（`samplingFrequencyIndex` 0..=12）。
const SAMPLING_RATES: [u32; 13] = [
    96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
];

/// AudioSpecificConfig（`esds` 里的解码器配置）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioSpecificConfig {
    /// 2 = AAC-LC（本解码器只做它）
    pub object_type: u8,
    pub sample_rate: u32,
    pub channels: u8,
}

impl AudioSpecificConfig {
    /// 解析配置。兼容两种形态：裸 ASC，以及带 `DecoderSpecificInfo`（0x05 + 变长长度）头的形态。
    pub fn parse(data: &[u8]) -> Result<AudioSpecificConfig> {
        let body = strip_dsi_header(data).unwrap_or(data);
        let mut r = BitReader::new(body);
        let mut object_type = r.read(5)? as u8;
        if object_type == 31 {
            object_type = 32 + r.read(6)? as u8;
        }
        let index = r.read(4)? as usize;
        let sample_rate = if index == 15 {
            r.read(24)?
        } else {
            *SAMPLING_RATES
                .get(index)
                .ok_or_else(|| MediaError::Bitstream(format!("AAC: 采样率索引 {index} 非法")))?
        };
        let channels = r.read(4)? as u8;
        // GASpecificConfig：frameLengthFlag = 1 表示 960 点帧
        let frame_length_flag = r.read(1)?;
        if frame_length_flag != 0 {
            return Err(MediaError::Unsupported(
                "AAC: 960 点帧（frameLengthFlag=1）还没接".into(),
            ));
        }
        Ok(AudioSpecificConfig { object_type, sample_rate, channels })
    }
}

/// 剥掉 `DecoderSpecificInfo` 的标签与长度（0x05 + 变长长度）。
fn strip_dsi_header(data: &[u8]) -> Option<&[u8]> {
    if data.first() != Some(&0x05) {
        return None;
    }
    let mut pos = 1usize;
    let mut len = 0usize;
    let mut bytes = 0;
    while pos < data.len() && bytes < 4 {
        let b = data[pos];
        len = (len << 7) | (b & 0x7F) as usize;
        pos += 1;
        bytes += 1;
        if b & 0x80 == 0 {
            break;
        }
    }
    if len > 0 && pos + len <= data.len() {
        Some(&data[pos..pos + len])
    } else {
        None
    }
}

/// 剥掉 ADTS 头（TS 源里的 AAC 是带 ADTS 的；fMP4 里是裸帧）。
///
/// 判据是规范的同步字：12 位全 1。保护位（CRC）存在时头是 9 字节，否则 7 字节。
fn strip_adts(packet: &[u8]) -> &[u8] {
    if packet.len() >= 7 && packet[0] == 0xFF && (packet[1] & 0xF0) == 0xF0 {
        let protection_absent = packet[1] & 0x01 == 1;
        let header = if protection_absent { 7 } else { 9 };
        if packet.len() > header {
            return &packet[header..];
        }
    }
    packet
}

/// ICS 的窗信息（`CPE` 的 `common_window` 会让两个声道共用一份）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IcsInfo {
    pub sequence: Sequence,
    pub shape: Shape,
    pub max_sfb: usize,
    /// 窗的个数（1 或 8）
    pub num_windows: usize,
    /// 窗组个数与每组包含几个窗
    pub group_len: Vec<usize>,
    /// 长窗 / 短窗的频带边界（来自 `tables::subband_tables`）
    pub long_swb: &'static [u16],
    pub short_swb: &'static [u16],
}

impl IcsInfo {
    /// 一个窗的频域系数个数（长窗 1024 / 短窗 128）。
    fn window_len(&self) -> usize {
        if self.num_windows == 8 {
            128
        } else {
            1024
        }
    }

    /// 频带 `sfb` 在窗内的起始下标。
    fn sfb_start(&self, sfb: usize) -> usize {
        let swb = if self.num_windows == 8 { self.short_swb } else { self.long_swb };
        swb.get(sfb).copied().unwrap_or(0) as usize
    }

    /// 频带 `sfb` 一共有多少个系数。
    fn sfb_len(&self, sfb: usize) -> usize {
        let swb = if self.num_windows == 8 { self.short_swb } else { self.long_swb };
        let hi = swb.get(sfb + 1).copied().unwrap_or(0) as usize;
        let lo = swb.get(sfb).copied().unwrap_or(0) as usize;
        hi.saturating_sub(lo)
    }
}

/// 窗序列。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sequence {
    OnlyLong,
    LongStart,
    EightShort,
    LongStop,
}

impl Sequence {
    fn from_bits(v: u32) -> Sequence {
        match v {
            0 => Sequence::OnlyLong,
            1 => Sequence::LongStart,
            2 => Sequence::EightShort,
            _ => Sequence::LongStop,
        }
    }
    pub fn num_windows(&self) -> usize {
        if *self == Sequence::EightShort {
            8
        } else {
            1
        }
    }
}

/// 窗形状（0 = 正弦，1 = KBD）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Sine,
    Kbd,
}

impl Shape {
    fn from_bit(v: u32) -> Shape {
        if v == 0 {
            Shape::Sine
        } else {
            Shape::Kbd
        }
    }
}

/// 某个窗组里一个 sfb 范围的**分段**：码表号 + 该段覆盖的 sfb 个数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub codebook: u8,
    /// 这一段覆盖的 sfb 个数
    pub sfb_count: usize,
}

/// 分段数据：`sections[group]`。
///
/// **分段是按窗组传的，不是按窗**：一组里的几个窗共用同一份码表安排
/// （短窗成组的意义就在于此）。对照 FFmpeg 的 `decode_band_types` —
/// 它的外层循环是 `for (g = 0; g < ics->num_window_groups; g++)`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sections {
    pub groups: Vec<Vec<Section>>,
}

impl AacDecoder {
    /// 读窗信息（`ics_info()`）。
    fn read_ics_info(&self, r: &mut BitReader) -> Result<IcsInfo> {
        let trace = std::env::var("ME_AAC_TRACE").is_ok();
        let reserved_ok = r.read(1)? == 0;
        let sequence = Sequence::from_bits(r.read(2)?);
        let shape = Shape::from_bit(r.read(1)?);
        if trace {
            eprintln!("  ics_info: 保留位{} seq={sequence:?} shape={shape:?} @位{}", if reserved_ok { "0" } else { "1(非法)" }, r.pos_bits());
        }
        if !reserved_ok {
            return Err(MediaError::Bitstream(format!(
                "AAC: ics_info 的保留位不是 0（seq={sequence:?} shape={shape:?} @位{}）",
                r.pos_bits()
            )));
        }
        let num_windows = sequence.num_windows();
        let max_sfb = if num_windows == 8 {
            r.read(4)? as usize
        } else {
            let max_sfb = r.read(6)? as usize;
            // 长窗的 ics_info 里还有"长时预测"标志（Main profile 的工具）。
            // 只做 LC：这一位必须读（不读就整体错 1 位），但必须是 0。
            let pred = r.read(1)?;
            if trace {
                eprintln!("  ics_info(长窗): max_sfb={max_sfb} 预测位={pred} @位{}", r.pos_bits());
            }
            if pred != 0 {
                return Err(MediaError::Unsupported(
                    "AAC: 出现了长时预测（Main profile 工具），本解码器只做 LC".into(),
                ));
            }
            max_sfb
        };
        // 短窗：scale_factor_grouping 决定 8 个窗怎么分组（连续 1 表示"同一组"）
        let mut group_len = Vec::new();
        if num_windows == 8 {
            let grouping = r.read(7)?;
            let mut len = 1usize;
            for i in 0..7 {
                if grouping & (1 << (6 - i)) != 0 {
                    len += 1;
                } else {
                    group_len.push(len);
                    len = 1;
                }
            }
            group_len.push(len);
        } else {
            group_len.push(1);
        }
        let (long_swb, short_swb) = tables::subband_tables(self.cfg.sample_rate)
            .ok_or_else(|| MediaError::Unsupported(format!(
                "AAC: 采样率 {} 没有对应的频带表",
                self.cfg.sample_rate
            )))?;
        if num_windows == 8 {
            if max_sfb + 1 > short_swb.len() {
                return Err(MediaError::Bitstream(format!(
                    "AAC: max_sfb={max_sfb} 超出短窗频带表（{} 段）",
                    short_swb.len() - 1
                )));
            }
        } else if max_sfb + 1 > long_swb.len() {
            return Err(MediaError::Bitstream(format!(
                "AAC: max_sfb={max_sfb} 超出长窗频带表（{} 段）",
                long_swb.len() - 1
            )));
        }
        Ok(IcsInfo { sequence, shape, max_sfb, num_windows, group_len, long_swb, short_swb })
    }

    /// 读分段数据（`section_data()`）——**每个窗组一份**。
    fn read_sections(&self, r: &mut BitReader, info: &IcsInfo) -> Result<Sections> {
        let short = info.num_windows == 8;
        let mut groups = Vec::with_capacity(info.group_len.len());
        for _ in 0..info.group_len.len() {
            let mut sections: Vec<Section> = Vec::new();
            let mut sfb = 0usize;
            while sfb < info.max_sfb {
                // 码表号：4 位。**这里没有"码表转义"** —— 15 是合法的
                // `INTENSITY_HCB`（强度立体声），不是转义标记；4 位里只有 12 是保留值。
                //
                // 曾经的写法是 `while cb == 15 { cb += read(4) }`（照 `ESC_HCB` 的转义
                // 语义）。它在本目录的立体声测试文件上就翻车：那帧右声道第 2 段正好是
                // `15`，按转义读会变成 `15 + 6 = 21`（LC 里根本不存在这个码表），整帧报废。
                // 编码器（Apple 的 afconvert）确实会把 15 当强度码表用，FFmpeg 也只拒绝
                // 12 —— 转义在 LC 里是死代码（转义出来的值只会 ≥15，与 15 本身撞车）。
                let cb = r.read(4)?;
                if cb == 12 {
                    return Err(MediaError::Bitstream(
                        "AAC: 码表号 12 非法（保留值；13=PNS、14/15=强度）".into(),
                    ));
                }
                // 段长：短窗 3 位 / 长窗 5 位，全 1 是转义（继续加）。
                // **注意语义**：覆盖的 sfb 数 = 各增量的**和**（不加 1）——
                // 这一点对着 FFmpeg 的 `decode_band_types`（`sect_end += sect_len_incr`）
                // 校正过；写成"和+1"会让整帧从分段起就错位。
                let (bits, escape) = if short { (3u8, 7u32) } else { (5u8, 31u32) };
                let mut count = 0usize;
                loop {
                    let incr = r.read(bits)?;
                    count += incr as usize;
                    if incr != escape {
                        break;
                    }
                }
                if sfb + count > info.max_sfb {
                    return Err(MediaError::Bitstream(format!(
                        "AAC: 段长越界（sfb {sfb}+{count} > max_sfb {}）",
                        info.max_sfb
                    )));
                }
                sections.push(Section { codebook: cb as u8, sfb_count: count });
                sfb += count;
            }
            groups.push(sections);
        }
        if std::env::var("ME_AAC_TRACE").is_ok() {
            // 码表分布也打出来：`cb13` 是 PNS（噪声是**随机**的，与系统解码器对不齐是
            // 正常现象）、`cb14/15` 是强度立体声。排查"某几帧与参考差得多"时，
            // 先看这几帧是不是用了这些工具 —— 免得去查本来就没问题的滤波器组。
            let per_group: Vec<String> = groups
                .iter()
                .map(|g| {
                    g.iter()
                        .map(|s| format!("cb{}×{}", s.codebook, s.sfb_count))
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .collect();
            eprintln!("  分段: {}", per_group.join(" | "));
        }
        Ok(Sections { groups })
    }

    /// 读比例因子（`scale_factor_data()`）：**每组只有一组比例因子** ——
    /// 返回 `sf[group][sfb]`，同组的各个窗共用（FFmpeg 的 `sfo[g*max_sfb+sfb]`）。
    fn read_scalefactors(
        &self,
        r: &mut BitReader,
        info: &IcsInfo,
        sections: &Sections,
        global_gain: i32,
    ) -> Result<Vec<Vec<i32>>> {
        // **三条独立的累加链**（对照 FFmpeg 的 `decode_scalefactors` 校正）：
        //   ① 普通谱段：从 global_gain 起差分；
        //   ② 噪声道（PNS）：从 global_gain − 90 起差分，且**第一个噪声段的第一个值
        //      是 9 位原值 − 256**（不是 Huffman 差分）；
        //   ③ 强度立体声：从 0 起差分。
        // 三条链互不干扰：混成一条会让后续所有比例因子整体偏移。
        const NOISE_OFFSET: i32 = 90;
        const NOISE_PRE_BITS: u8 = 9;
        const NOISE_PRE: i32 = 256;
        let mut offset_main = global_gain;
        let mut offset_noise = global_gain - NOISE_OFFSET;
        let mut offset_intensity = 0i32;
        let mut noise_flag = true;
        let mut out: Vec<Vec<i32>> = Vec::with_capacity(sections.groups.len());
        for group in sections.groups.iter() {
            let mut sfs = vec![0i32; info.max_sfb];
            let mut sfb = 0usize;
            for section in group.iter() {
                for _ in 0..section.sfb_count {
                    match section.codebook {
                        // 零码表：没有比例因子
                        0 => {}
                        // 强度立体声（14/15）：绝对位置-100
                        14 | 15 => {
                            offset_intensity += self.hcbs.read_scalefactor(r)?;
                            let clipped = offset_intensity.clamp(-155, 100);
                            sfs[sfb] = clipped - 100;
                        }
                        // PNS（13）：首段首值 9 位原值 − 256，之后走 Huffman 差分
                        13 => {
                            if noise_flag {
                                noise_flag = false;
                                offset_noise += r.read(NOISE_PRE_BITS)? as i32 - NOISE_PRE;
                            } else {
                                offset_noise += self.hcbs.read_scalefactor(r)?;
                            }
                            sfs[sfb] = offset_noise.clamp(-100, 155);
                        }
                        // 普通谱段（1..=11）
                        _ => {
                            offset_main += self.hcbs.read_scalefactor(r)?;
                            if offset_main > 255 {
                                return Err(MediaError::Bitstream(format!(
                                    "AAC: 比例因子越界（{offset_main} > 255）"
                                )));
                            }
                            sfs[sfb] = offset_main;
                        }
                    }
                    sfb += 1;
                }
            }
            out.push(sfs);
        }
        Ok(out)
    }
}

/// 谱数据的量化值（按窗组 → 窗 → 系数展开）。
#[derive(Debug, Clone)]
pub struct Spectral {
    /// `[group][window][coef]`
    pub q: Vec<Vec<Vec<i32>>>,
}

impl AacDecoder {
    /// 读谱数据（`spectral_data()`）：按分段表逐段用对应码表解量化值。
    ///
    /// **位流顺序**（对照 FFmpeg 的 `decode_spectrum_and_dequant`，它的循环是
    /// `for g { for sfb { for window-in-group } }`）：同一个 sfb 的**各个窗的谱系数
    /// 是连着传的**。按"一个窗读完所有 sfb 再读下一个窗"解析会让多窗组（短窗）
    /// 从第一段起就错位 —— 这是 element_instance_tag 之后的第二个错位源。
    fn read_spectral(
        &self,
        r: &mut BitReader,
        info: &IcsInfo,
        sections: &Sections,
    ) -> Result<Spectral> {
        let win_len = info.window_len();
        let mut groups = Vec::with_capacity(sections.groups.len());
        for (g, group) in sections.groups.iter().enumerate() {
            let g_len = info.group_len[g];
            let mut windows = vec![vec![0i32; win_len]; g_len];
            let mut sfb = 0usize;
            for section in group.iter() {
                let cb = section.codebook as u32;
                for _ in 0..section.sfb_count {
                    let (start, len) = (info.sfb_start(sfb), info.sfb_len(sfb));
                    debug_assert!(start + len <= win_len);
                    match cb {
                        // 零码表：整段是 0（不用读位）
                        0 => {}
                        // 四元组：一次解 4 个
                        1..=4 => {
                            if len % 4 != 0 {
                                return Err(MediaError::Bitstream(
                                    "AAC: 四元组码表的频带长度不是 4 的倍数".into(),
                                ));
                            }
                            for plane in windows.iter_mut() {
                                let mut i = start;
                                while i < start + len {
                                    let v = self.hcbs.read_quads(cb, r)?;
                                    plane[i..i + 4].copy_from_slice(&v);
                                    i += 4;
                                }
                            }
                        }
                        // 成对（含码表 11 的转义）
                        5..=11 => {
                            if len % 2 != 0 {
                                return Err(MediaError::Bitstream(
                                    "AAC: 成对码表的频带长度不是 2 的倍数".into(),
                                ));
                            }
                            for plane in windows.iter_mut() {
                                let mut i = start;
                                while i < start + len {
                                    let (x, y) = self.hcbs.read_pair(cb, r)?;
                                    plane[i] = x;
                                    plane[i + 1] = y;
                                    i += 2;
                                }
                            }
                        }
                        // PNS / 强度立体声：值在别处生成（这里不读位）
                        13..=15 => {}
                        other => {
                            return Err(MediaError::Bitstream(format!(
                                "AAC: 分段里出现非法码表号 {other}"
                            )))
                        }
                    }
                    sfb += 1;
                }
            }
            groups.push(windows);
        }
        Ok(Spectral { q: groups })
    }

    /// 读 TNS 数据（`tns_data()`）。
    fn read_tns(&self, r: &mut BitReader, r_present: bool, info: &IcsInfo) -> Result<Option<TnsData>> {
        if !r_present {
            return Ok(None);
        }
        let short = info.num_windows == 8;
        let n_windows = info.num_windows;
        let mut windows = Vec::with_capacity(n_windows);
        // n_filt / coef_res 都是**按窗**的：
        // - 长窗（1 个窗）读 2 位 n_filt；短窗（8 个窗）每窗读 1 位；
        // - `coef_res` 紧跟在该窗的 `n_filt` 之后，**只在 n_filt>0 时出现**，且**每个窗各读一次**。
        //   之前把它当成"整帧读一次"（`any` 之后读 1 位），短窗里第二个有滤波器的窗起
        //   就整体错位 —— 后面 gain_control / spectral 全乱。
        for _ in 0..n_windows {
            let n = if short { r.read(1)? as usize } else { r.read(2)? as usize };
            let mut filters = Vec::with_capacity(n);
            if n > 0 {
                // `coef_res` 是一个**位**（0/1，表示 3 或 4），不是"1 位 + 3"后的值；
                // 位宽与查表下标都由它和 `coef_compress` 决定。
                let coef_res_bit = r.read(1)? as usize;
                for _ in 0..n {
                    let length = r.read(if short { 4 } else { 6 })? as usize;
                    let order = r.read(if short { 3 } else { 5 })? as usize;
                    let mut refl = Vec::new();
                    let mut up = true;
                    if order > 0 {
                        up = r.read(1)? == 1;
                        let compress = r.read(1)? as usize;
                        let coef_len = coef_res_bit + 3 - compress;
                        let map = tables::TNS_TMP2_MAP[2 * compress + coef_res_bit];
                        for _ in 0..order {
                            // 传的是**下标**（无符号）→ 查表得反射系数
                            let raw = r.read(coef_len as u8)? as usize;
                            let v = map.get(raw).copied().ok_or_else(|| {
                                MediaError::Bitstream("AAC: TNS 系数下标越界".into())
                            })?;
                            refl.push(v);
                        }
                    }
                    filters.push(TnsFilter { length, order, up, refl });
                }
            }
            windows.push(filters);
        }
        Ok(Some(TnsData { windows }))
    }
}

/// TNS 数据（按窗）。
#[derive(Debug, Clone)]
pub struct TnsData {
    pub windows: Vec<Vec<TnsFilter>>,
}

/// 一个 TNS 滤波器。
#[derive(Debug, Clone)]
pub struct TnsFilter {
    /// 覆盖的**频带数**（从高频端往下数，规范如此；不是系数个数）
    pub length: usize,
    pub order: usize,
    /// true = 从低频往高频扫
    pub up: bool,
    /// **反射系数**（已按 `TNS_TMP2_MAP` 由位流下标映射而来）
    pub refl: Vec<f32>,
}

/// 反量化 + 立体声工具 + 合成。
impl AacDecoder {
    /// 反量化一个声道（含 PNS 噪声段），按 `[group][window][coef]` 展开。
    ///
    /// 普通谱段：`x = sign(q)·|q|^(4/3)·2^(0.25·(sf−100))`（规范 4.6.2.3.3）。
    /// PNS 段：填随机谱再把**整段能量**归一到 `2^(0.5·noise_sf)` ——
    /// 与 FFmpeg 的 `scale = sf/√(band_energy)`（`sf = 2^(0.25·noise_sf)`）等价；
    /// 只按系数乘幅度会差一个"带宽"因子，噪声电平会随频带宽度变化。
    fn dequant_channel(state: &mut ChannelState, parsed: &IcsParsed) -> Vec<Vec<Vec<f32>>> {
        let info = &parsed.info;
        let mut out: Vec<Vec<Vec<f32>>> = Vec::with_capacity(parsed.spectral.q.len());
        for (g, windows) in parsed.spectral.q.iter().enumerate() {
            let mut per_window = Vec::with_capacity(windows.len());
            for q in windows.iter() {
                let mut x = vec![0f32; q.len()];
                for sfb in 0..info.max_sfb {
                    let (start, len) = (info.sfb_start(sfb), info.sfb_len(sfb));
                    let end = (start + len).min(x.len());
                    let sfv = parsed.sf[g][sfb] as f32;
                    match parsed.cb[g][sfb] {
                        // PNS：随机谱 + 能量归一
                        13 => {
                            let target = 2f32.powf(0.5 * sfv);
                            let mut energy = 0f32;
                            for slot in &mut x[start..end] {
                                let v = state.noise_value();
                                *slot = v;
                                energy += v * v;
                            }
                            if energy > 0.0 {
                                let scale = (target / energy).sqrt();
                                for slot in &mut x[start..end] {
                                    *slot *= scale;
                                }
                            }
                        }
                        // 强度立体声段：值在 CPE 层从另一声道拷贝（这里留 0）
                        14 | 15 => {}
                        _ => {
                            let gain = 2f32.powf(0.25 * (sfv - 100.0));
                            for i in start..end {
                                let qi = q[i];
                                if qi != 0 {
                                    let mag = (qi.abs() as f32).powf(4.0 / 3.0);
                                    x[i] = if qi < 0 { -mag } else { mag } * gain;
                                }
                            }
                        }
                    }
                }
                per_window.push(x);
            }
            out.push(per_window);
        }
        out
    }

    /// M/S 立体声（规范 4.6.8.1）：`L = M + S`、`R = M − S`，在反量化后的频域做。
    ///
    /// 逐窗平面摊平后按下标 `[group][window][coef]` 访问，而 `ms_mask` 是
    /// **按窗组**的（`ms_mask[g*max_sfb+sfb]`）。只对真正的谱段（码表 < 13）做：
    /// PNS / 强度段不走这个蝴蝶变换（对照 FFmpeg 的 `apply_mid_side_stereo`）。
    fn apply_mid_side(
        left: &mut [Vec<f32>],
        right: &mut [Vec<f32>],
        info: &IcsInfo,
        l_cb: &[Vec<u8>],
        r_cb: &[Vec<u8>],
        ms_mask: &[bool],
    ) {
        let mut w0 = 0usize;
        for (g, &g_len) in info.group_len.iter().enumerate() {
            for sfb in 0..info.max_sfb {
                let idx = g * info.max_sfb + sfb;
                let on = ms_mask.get(idx).copied().unwrap_or(false);
                let spectral = l_cb[g][sfb] < 13 && r_cb[g][sfb] < 13;
                if on && spectral {
                    let (start, len) = (info.sfb_start(sfb), info.sfb_len(sfb));
                    for w in 0..g_len {
                        let wi = w0 + w;
                        if wi >= left.len() || wi >= right.len() {
                            break;
                        }
                        for i in start..(start + len).min(left[wi].len()) {
                            let m = left[wi][i];
                            let s = right[wi][i];
                            left[wi][i] = m + s;
                            right[wi][i] = m - s;
                        }
                    }
                }
            }
            w0 += g_len;
        }
    }

    /// 强度立体声（规范 4.6.8.2）：右声道的强度段**直接取左声道的谱**再乘一个尺度。
    ///
    /// 尺度的**幅度**是 `2^(−is_pos/4)`（`is_pos` 是那条从 0 起差分的链上的值，
    /// 0 ⇒ 1.0、越大越衰减）—— 这是 FFmpeg `ff_aac_pow2sf_tab[-sfo-100+POW_SF2_ZERO]`
    /// 的等价写法；符号由码表号（14 ⇒ −、15 ⇒ +）与 `ms_mask` 共同决定。
    fn apply_intensity(
        left: &[Vec<f32>],
        right: &mut [Vec<f32>],
        r_info: &IcsInfo,
        r_cb: &[Vec<u8>],
        r_sf: &[Vec<i32>],
        ms_present: u8,
        ms_mask: &[bool],
    ) {
        let mut w0 = 0usize;
        for (g, &g_len) in r_info.group_len.iter().enumerate() {
            for sfb in 0..r_info.max_sfb {
                let bt = r_cb[g][sfb];
                if bt != 14 && bt != 15 {
                    continue;
                }
                let mut c = -1.0f32 + 2.0 * (bt as f32 - 14.0);
                if ms_present != 0 && ms_mask.get(g * r_info.max_sfb + sfb).copied().unwrap_or(false) {
                    c = -c;
                }
                let is_pos = r_sf[g][sfb] as f32 + 100.0;
                let scale = c * 2f32.powf(-0.25 * is_pos);
                let (start, len) = (r_info.sfb_start(sfb), r_info.sfb_len(sfb));
                for w in 0..g_len {
                    let wi = w0 + w;
                    if wi >= right.len() || wi >= left.len() {
                        break;
                    }
                    for i in start..(start + len).min(right[wi].len()) {
                        right[wi][i] = left[wi][i] * scale;
                    }
                }
            }
            w0 += g_len;
        }
    }

    /// TNS：在频域做全极点滤波（规范 4.6.9.3）。
    ///
    /// 位流里的 `length` 是**从高频端数起的频带个数**，滤波器覆盖的系数范围由 swb 偏移表
    /// 换算出来 —— 不是"从下标 0 起 length 个系数"。系数还要经过"位流下标 → 反射系数
    /// （查表）→ LPC 系数（Levinson-Durbin）"三步；少了任何一步都会让全极点滤波器发散。
    fn apply_tns(plane: &mut [f32], tns: Option<&TnsData>, info: &IcsInfo, w: usize, max_bands: usize) {
        let Some(data) = tns else { return };
        let Some(filters) = data.windows.get(w) else { return };
        if filters.is_empty() {
            return;
        }
        let short = info.num_windows == 8;
        let swb: &[u16] = if short { info.short_swb } else { info.long_swb };
        let num_swb = swb.len().saturating_sub(1);
        let mmm = max_bands.min(info.max_sfb);
        if mmm == 0 {
            return;
        }
        let n = plane.len();
        let mut bottom = num_swb;
        for f in filters {
            let top = bottom;
            bottom = top.saturating_sub(f.length);
            if f.order == 0 || f.refl.len() < f.order {
                continue;
            }
            let lpc = levinson_lpc(&f.refl, f.order);
            let start = swb.get(bottom.min(mmm)).copied().unwrap_or(0) as usize;
            let end = swb.get(top.min(mmm)).copied().unwrap_or(0) as usize;
            let size = end.saturating_sub(start);
            if size == 0 {
                continue;
            }
            let (mut idx, inc) = if f.up { (start, 1isize) } else { (end - 1, -1isize) };
            for m in 0..size {
                if idx >= n {
                    break;
                }
                let mut y = plane[idx];
                for i in 1..=m.min(f.order) {
                    let j = (idx as isize - i as isize * inc) as usize;
                    if j >= n {
                        break;
                    }
                    y -= lpc[i - 1] * plane[j];
                }
                plane[idx] = y;
                idx = (idx as isize + inc) as usize;
            }
        }
    }

    /// 合成一个声道：IMDCT → 加窗 → 放进 2048 缓冲（短窗从 448 起）→ 与上一帧尾巴相加。
    fn synthesize(
        state: &mut ChannelState,
        info: &IcsInfo,
        x: &[Vec<f32>],
        long_windows: &[(Shape, Vec<f32>); 2],
        short_windows: &[(Shape, Vec<f32>); 2],
        out: &mut Vec<f32>,
    ) {
        let mut buf = vec![0f32; 2048];
        let long_win = &long_windows[usize::from(info.shape == Shape::Kbd)].1;
        match info.sequence {
            Sequence::OnlyLong | Sequence::LongStart | Sequence::LongStop => {
                let mut tmp = vec![0f32; 2048];
                imdct::imdct(&x[0], &mut tmp);
                let win = seq_window(info.sequence, long_win, info.shape, short_windows);
                for i in 0..2048 {
                    buf[i] = tmp[i] * win[i];
                }
            }
            Sequence::EightShort => {
                let short_win = &short_windows[usize::from(info.shape == Shape::Kbd)].1;
                for (w, plane) in x.iter().enumerate() {
                    if plane.len() != 128 {
                        break;
                    }
                    let mut tmp = vec![0f32; 256];
                    imdct::imdct(plane, &mut tmp);
                    let base = 448 + 128 * w;
                    for i in 0..256 {
                        buf[base + i] += tmp[i] * short_win[i];
                    }
                }
            }
        }
        out.clear();
        out.extend(buf[..1024].iter().zip(state.overlap.iter()).map(|(a, b)| a + b));
        state.overlap.copy_from_slice(&buf[1024..2048]);
    }
}

/// 反射系数 → LPC 系数（Levinson-Durbin 递推）。
///
/// 与 FFmpeg `compute_lpc_coefs(..., normalize=0)` 等价：每次迭代用反射系数 `r = -k_i`
/// 就地更新已求出的低阶系数。TNS 的全极点滤波器用的是这里得到的 `lpc`，不是位流里的
/// 原始下标，也不是反射系数本身。
fn levinson_lpc(refl: &[f32], order: usize) -> Vec<f32> {
    let mut lpc = vec![0f32; order];
    for i in 0..order {
        let r = -refl[i];
        lpc[i] = r;
        for j in 0..i.div_ceil(2) {
            let f = lpc[j];
            let b = lpc[i - 1 - j];
            lpc[j] = f + r * b;
            lpc[i - 1 - j] = b + r * f;
        }
    }
    lpc
}

/// 序列对应的 2048 点合成窗。
///
/// 几何关系（推导一遍，别再错）：短序列的 8 个块放在 **[448, 1600)**（每块 256 点、
/// 间隔 128），所以：
/// - 它**够不着** [0,448) 与 [1600,2048) 这两段；
/// - 它盖住 [448,1600)，其中与前/后一帧真窗重叠的是 [448,576) 与 [1472,1600) 两段。
///
/// 于是过渡窗是这么拼的（`w_short` 是 256 点短窗，前半 0..128 上升、后半 128..256 下降）：
///
/// - `LONG_START`（长→短）：左半照常；**[1024,1472) 必须是 1.0**（短序列够不着这里，
///   这一段要靠这一帧自己原样送过去，置 0 就是把声音抹掉）；`[1472,1600)` 用短窗的
///   **下降段**（与短序列第一块的上升段互补：`w²+(1-w²)=1`）；`[1600,2048)` 为 0。
/// - `LONG_STOP`（短→长）：`[0,448)` 为 0；`[448,576)` 用短窗的**上升段**（与短序列
///   最后一块的下降段互补）；**[576,1024) 必须是 1.0**（接住短序列送过来的那一段）；
///   右半照常。
///
/// 平坦段与"短窗半段"的取法错任何一处，症状都是**固定 448 个样本被清零或重复**
/// （2026-09-27 踩过：正弦开头与结尾各静一小段）。自校验测试见
/// [`tests::filterbank_reconstructs_the_signal_across_window_sequences`]。
fn seq_window(seq: Sequence, long_win: &[f32], shape: Shape, short_windows: &[(Shape, Vec<f32>); 2]) -> Vec<f32> {
    let mut w = vec![0f32; 2048];
    let short = &short_windows[usize::from(shape == Shape::Kbd)].1;
    match seq {
        Sequence::OnlyLong => w.copy_from_slice(&long_win[..2048]),
        Sequence::LongStart => {
            w[..1024].copy_from_slice(&long_win[..1024]);
            w[1024..1472].fill(1.0);
            // 尾巴用短窗的**下降段**（不是把上升段反着放 —— 短窗是对称的，反着放等于上升段）
            w[1472..1600].copy_from_slice(&short[128..]);
        }
        Sequence::LongStop => {
            w[448..576].copy_from_slice(&short[..128]);
            w[576..1024].fill(1.0);
            w[1024..].copy_from_slice(&long_win[1024..2048]);
        }
        Sequence::EightShort => unreachable!("短序列用短窗，不走这里"),
    }
    w
}

/// 单条声道的跨帧状态。
struct ChannelState {
    /// 上一帧留下的时域后半段（1024 点，叠加用）
    overlap: Vec<f32>,
    /// PNS 用的确定性伪随机（同一文件每次解出来一样，方便对拍）
    rng: u32,
}

impl ChannelState {
    fn new() -> Self {
        Self { overlap: vec![0.0; 1024], rng: 0x1234_5678 }
    }
    /// PNS 的随机谱值：零均值、确定性（同一文件每次解出来一样，方便对拍）。
    fn noise_value(&mut self) -> f32 {
        // xorshift32 → [−1, 1)
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        (self.rng as f32 / 2147483648.0) - 1.0
    }
}

/// ICS 的完整解析结果（一条声道一帧）。
#[derive(Debug, Clone)]
pub struct IcsParsed {
    pub info: IcsInfo,
    /// `sf[group][sfb]`（同组的窗共用）
    pub sf: Vec<Vec<i32>>,
    /// `cb[group][sfb]`：每段的码表号（判断 PNS=13 / 强度=14,15 用）
    pub cb: Vec<Vec<u8>>,
    pub spectral: Spectral,
    pub tns: Option<TnsData>,
}

/// 自研 AAC-LC 解码器。
pub struct AacDecoder {
    cfg: AudioSpecificConfig,
    /// 码表在 `open` 时就建好：等于开播前把全部码字校验一遍，表错了当场暴露。
    hcbs: Hcbs,
    /// 两种窗形的窗：长窗 2048、短窗 256（每帧都要用，建一次）
    long_windows: [(Shape, Vec<f32>); 2],
    short_windows: [(Shape, Vec<f32>); 2],
    /// 各声道的跨帧状态（按元素里出现的顺序）
    states: Vec<ChannelState>,
}

impl AacDecoder {
    pub fn open(extradata: &[u8]) -> Result<Self> {
        let cfg = AudioSpecificConfig::parse(extradata)?;
        if cfg.object_type != 2 {
            return Err(MediaError::Unsupported(format!(
                "AAC: object type {} 还不是 LC（2）—— SBR/PS 等还没接",
                cfg.object_type
            )));
        }
        if cfg.channels == 0 || cfg.channels > 2 {
            return Err(MediaError::Unsupported(format!(
                "AAC: 声道配置 {} 暂不支持（本解码器做单声道/立体声）",
                cfg.channels
            )));
        }
        Ok(Self {
            cfg,
            hcbs: Hcbs::new(),
            // 窗：长窗 2048（KBD alpha = 4）、短窗 256（KBD alpha = 6）——
            // 规范给这两种长度定的 KBD 参数不同，短窗用 4.0 会让瞬态窗形偏斜。
            long_windows: [
                (Shape::Sine, imdct::window(WindowShape::Sine, 2048, 4.0)),
                (Shape::Kbd, imdct::window(WindowShape::Kbd, 2048, 4.0)),
            ],
            short_windows: [
                (Shape::Sine, imdct::window(WindowShape::Sine, 256, 6.0)),
                (Shape::Kbd, imdct::window(WindowShape::Kbd, 256, 6.0)),
            ],
            states: Vec::new(),
        })
    }

    pub fn config(&self) -> AudioSpecificConfig {
        self.cfg
    }

    fn state(&mut self, index: usize) -> &mut ChannelState {
        while self.states.len() <= index {
            self.states.push(ChannelState::new());
        }
        &mut self.states[index]
    }

    /// 读一条 ICS（`individual_channel_stream()`）。
    fn read_ics(&mut self, r: &mut BitReader, shared: Option<&IcsInfo>) -> Result<IcsParsed> {
        let trace = std::env::var("ME_AAC_TRACE").is_ok();
        let t0 = r.pos_bits();
        let global_gain = r.read(8)? as i32;
        let t1 = r.pos_bits();
        let info = match shared {
            Some(i) => i.clone(),
            None => self.read_ics_info(r)?,
        };
        let t2 = r.pos_bits();
        let sections = self.read_sections(r, &info)?;
        let t3 = r.pos_bits();
        let sf = self.read_scalefactors(r, &info, &sections, global_gain)?;
        let t4 = r.pos_bits();
        if trace {
            eprintln!(
                "  ICS gain={global_gain} seq={:?} shape={:?} max_sfb={} 组={:?} 段={} | 位: gg={} info+{} 段+{} sf+{} 位置={}",
                info.sequence,
                info.shape,
                info.max_sfb,
                info.group_len,
                sections.groups.iter().map(|g| g.len()).sum::<usize>(),
                t1 - t0,
                t2 - t1,
                t3 - t2,
                t4 - t3,
                t4,
            );
        }

        // 脉冲数据（罕见但合法）：会改谱系数的幅度
        let pulse_present = r.read(1)? == 1;
        let mut pulses = Vec::new();
        if pulse_present {
            let count = r.read(2)? as usize + 1;
            let start_sfb = r.read(6)? as usize;
            for _ in 0..count {
                let offset = r.read(5)? as usize;
                let amp = r.read(4)? as usize;
                pulses.push((start_sfb, offset, amp));
            }
        }
        let tns_present = r.read(1)? == 1;
        let tns = self.read_tns(r, tns_present, &info)?;
        if trace {
            eprintln!("  脉冲={pulse_present} TNS={tns_present} 位={}", r.pos_bits());
        }
        // 增益控制数据：那是 SSR profile 的工具，LC 里这一位必须是 0。
        if r.read(1)? != 0 {
            return Err(MediaError::Unsupported(
                "AAC: 出现了增益控制（gain_control_data），LC 不该有".into(),
            ));
        }
        let t5 = r.pos_bits();
        let spectral = self.read_spectral(r, &info, &sections)?;
        if trace {
            eprintln!(
                "  谱数据 {} 位 → 累计 {} / 共 {} 位",
                r.pos_bits() - t5,
                r.pos_bits(),
                r.total_bits()
            );
        }

        // 逐 sfb 的分段码表号（PNS / 强度判断要用）——按窗组
        let mut cb: Vec<Vec<u8>> = Vec::with_capacity(sections.groups.len());
        for group in sections.groups.iter() {
            let mut row = vec![0u8; info.max_sfb];
            let mut sfb = 0usize;
            for sec in group.iter() {
                for _ in 0..sec.sfb_count {
                    row[sfb] = sec.codebook;
                    sfb += 1;
                }
            }
            cb.push(row);
        }

        let mut parsed = IcsParsed { info, sf, cb, spectral, tns };
        Self::apply_pulses(&mut parsed, &pulses)?;
        Ok(parsed)
    }

    /// 脉冲数据：在指定 sfb 的谱系数上加幅度（规范 4.6.3.3）。
    fn apply_pulses(parsed: &mut IcsParsed, pulses: &[(usize, usize, usize)]) -> Result<()> {
        if pulses.is_empty() {
            return Ok(());
        }
        let win_len = parsed.info.window_len();
        for (g, windows) in parsed.spectral.q.iter_mut().enumerate() {
            for (w, plane) in windows.iter_mut().enumerate() {
                let _ = (g, w);
                for &(start_sfb, offset, amp) in pulses {
                    let base = parsed.info.sfb_start(start_sfb);
                    let idx = base + offset;
                    if idx >= win_len || idx >= plane.len() {
                        return Err(MediaError::Bitstream("AAC: 脉冲位置越界".into()));
                    }
                    let cur = plane[idx];
                    plane[idx] = if cur > 0 { cur + amp as i32 } else { cur - amp as i32 };
                }
            }
        }
        Ok(())
    }

    /// 一个声道元素（SCE/LFE）→ 时域 PCM。
    fn decode_sce(&mut self, r: &mut BitReader, channel: usize) -> Result<Vec<f32>> {
        let parsed = self.read_ics(r, None)?;
        self.render(channel, parsed)
    }

    /// 反量化 + 摊平成"逐窗"平面（分组只影响解码时的组织，TNS 与合成按窗走）。
    fn planes_of(&mut self, channel: usize, parsed: &IcsParsed) -> Vec<Vec<f32>> {
        let x = {
            let state = self.state(channel);
            Self::dequant_channel(state, parsed)
        };
        let mut planes: Vec<Vec<f32>> = Vec::new();
        for windows in x.into_iter() {
            for plane in windows.into_iter() {
                planes.push(plane);
            }
        }
        planes
    }

    /// TNS（逐窗，在频域）→ 合成 1024 点时域样本。
    fn synth_from_planes(
        &mut self,
        channel: usize,
        info: &IcsInfo,
        tns: Option<&TnsData>,
        planes: &mut [Vec<f32>],
    ) -> Result<Vec<f32>> {
        let short = info.num_windows == 8;
        let max_bands = tables::tns_max_bands(self.cfg.sample_rate, short);
        for (w, plane) in planes.iter_mut().enumerate() {
            Self::apply_tns(plane, tns, info, w, max_bands);
        }
        let mut out = Vec::with_capacity(1024);
        // 窗表拷到局部：`state()` 要可变借用 self，窗表同时被借用会冲突（表很小，拷贝无妨）
        let long_windows = self.long_windows.clone();
        let short_windows = self.short_windows.clone();
        {
            let state = self.state(channel);
            Self::synthesize(
                state,
                info,
                planes,
                &long_windows,
                &short_windows,
                &mut out,
            );
        }
        Ok(out)
    }

    /// 把一条 ICS 渲染成 1024 点时域样本（反量化 → TNS → 合成）。
    fn render(&mut self, channel: usize, parsed: IcsParsed) -> Result<Vec<f32>> {
        let mut planes = self.planes_of(channel, &parsed);
        self.synth_from_planes(channel, &parsed.info, parsed.tns.as_ref(), &mut planes)
    }

    /// 解一个 `raw_data_block`。
    fn decode_frame(&mut self, packet: &[u8], pts_us: i64) -> Result<PcmChunk> {
        let body = strip_adts(packet);
        let mut r = BitReader::new(body);
        let mut channels: Vec<Vec<f32>> = Vec::new();
        let trace = std::env::var("ME_AAC_TRACE").is_ok();
        let mut ele = 0;
        loop {
            let at = r.pos_bits();
            let id = r.read(3)?;
            // **ID_END 没有后续的 4 位字段**：它只有那 3 位，读到这里就结束。
            // 之前无条件读 4 位 "tag"，会在每个 `raw_data_block` 的末尾多吞 4 位 ——
            // 数据刚好够时直接报 NeedMore（静音帧就是这样全军覆没的）。
            if id == 7 {
                if trace {
                    eprintln!("元素#{ele} id=7 (ID_END) @位{at}");
                }
                break;
            }
            // 其余元素的 id 之后紧跟 4 位：SCE/CPE/CCE/DSE/PCE 是
            // **element_instance_tag**，FIL 是填充计数（同一个 4 位字段，语义不同）。
            // **漏读这 4 位会让后面所有字段整体错位 4 位**。
            let tag = r.read(4)?;
            if trace {
                eprintln!("元素#{ele} id={id} tag={tag} @位{at}");
            }
            ele += 1;
            match id {
                // SCE / LFE：单声道元素
                0 | 3 => {
                    let ch = channels.len();
                    let pcm = self.decode_sce(&mut r, ch)?;
                    channels.push(pcm);
                }
                // CPE：一对声道（可能共用窗信息、可能有 M/S 与强度立体声）
                1 => {
                    let common = r.read(1)? == 1;
                    let shared = if common { Some(self.read_ics_info(&mut r)?) } else { None };
                    let mut ms_mask: Vec<bool> = Vec::new();
                    // ms_mask_present 要带出 `if common` 作用域：强度立体声要看它决定符号
                    let mut ms_present = 0u8;
                    if common {
                        let info = shared.as_ref().unwrap();
                        let present = r.read(2)?;
                        ms_present = present as u8;
                        // `ms_used` 是 **每个窗组一份**（`num_window_groups × max_sfb`），
                        // 不是单个窗组的大小 —— 短窗（多组）时少读会整体错位。
                        // 下游 `apply_mid_side` / `apply_intensity` 也按 `g*max_sfb+sfb` 索引。
                        ms_mask = vec![false; info.group_len.len() * info.max_sfb];
                        match present {
                            0 => {}
                            1 => {
                                for m in ms_mask.iter_mut() {
                                    *m = r.read(1)? == 1;
                                }
                            }
                            2 => ms_mask.iter_mut().for_each(|m| *m = true),
                            other => {
                                return Err(MediaError::Bitstream(format!(
                                    "AAC: ms_mask_present={other} 是保留值"
                                )))
                            }
                        }
                    }
                    let left = self.read_ics(&mut r, shared.as_ref())?;
                    let right = self.read_ics(&mut r, shared.as_ref())?;
                    let li = channels.len();
                    let ri = li + 1;
                    // 立体声工具的**顺序**：反量化 → M/S → 强度 → TNS → 合成
                    // （对照 FFmpeg：`decode_cpe` 里做 M/S 与强度，
                    //  `spectral_to_sample` 里才 apply_tns 再 IMDCT）。
                    let mut planes_l = self.planes_of(li, &left);
                    let mut planes_r = self.planes_of(ri, &right);
                    if common {
                        Self::apply_mid_side(
                            &mut planes_l,
                            &mut planes_r,
                            &left.info,
                            &left.cb,
                            &right.cb,
                            &ms_mask,
                        );
                    }
                    Self::apply_intensity(
                        &planes_l,
                        &mut planes_r,
                        &right.info,
                        &right.cb,
                        &right.sf,
                        ms_present,
                        &ms_mask,
                    );
                    let raw_l =
                        self.synth_from_planes(li, &left.info, left.tns.as_ref(), &mut planes_l)?;
                    let raw_r =
                        self.synth_from_planes(ri, &right.info, right.tns.as_ref(), &mut planes_r)?;
                    channels.push(raw_l);
                    channels.push(raw_r);
                }
                // CCE / PCE：不支持
                2 | 5 => {
                    return Err(MediaError::Unsupported(
                        "AAC: 耦合声道（CCE）与声道配置元素（PCE）还没接".into(),
                    ))
                }
                // DSE：数据元素，跳过（标签已在循环里读过）
                4 => {
                    if r.read(1)? == 1 {
                        r.align_to_byte();
                    }
                    let count = r.read(8)? as usize;
                    for _ in 0..count {
                        r.read(8)?;
                    }
                }
                // FIL：填充，跳过（那 4 位计数就是循环里读到的 tag）
                6 => {
                    let total = if tag == 15 { r.read(8)? as usize + 14 } else { tag as usize };
                    for _ in 0..total {
                        r.read(8)?;
                    }
                }
                other => {
                    return Err(MediaError::Bitstream(format!("AAC: 未知句法元素 id={other}")))
                }
            }
        }
        if channels.is_empty() {
            return Err(MediaError::Bitstream("AAC: 这一帧里没有声道数据".into()));
        }
        // 交织 + 取整。**合成结果已经在 i16 量级上**（`2/N` 的 IMDCT 配上规范的比例因子，
        // 满量程信号落在 ±32768 附近），不需要再乘 32768 —— 再乘一次就会整体削顶成方波。
        let n = channels[0].len();
        let nch = channels.len().min(2);
        let mut samples = Vec::with_capacity(n * nch);
        for i in 0..n {
            for plane in channels.iter().take(nch) {
                let v = plane.get(i).copied().unwrap_or(0.0);
                let s = v.round().clamp(-32768.0, 32767.0) as i32;
                samples.push(s);
            }
        }
        Ok(PcmChunk {
            sample_rate: self.cfg.sample_rate,
            channels: nch as u8,
            bits_per_sample: 16,
            pts_us,
            samples,
        })
    }
}

impl AudioDecoder for AacDecoder {
    fn decode(&mut self, packet: &[u8], pts_us: i64) -> Result<Vec<PcmChunk>> {
        if packet.is_empty() {
            return Ok(Vec::new());
        }
        Ok(vec![self.decode_frame(packet, pts_us)?])
    }

    fn flush(&mut self) -> Result<Vec<PcmChunk>> {
        Ok(Vec::new())
    }

    fn stream_info(&self) -> (u32, u8, u8) {
        (self.cfg.sample_rate, self.cfg.channels, 16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_audio_specific_config() {
        // LC / 44100（索引 4）/ 立体声（配置 2）：00010 0100 0010 0 → 0x12 0x10
        let asc = [0x12u8, 0x10];
        let cfg = AudioSpecificConfig::parse(&asc).unwrap();
        assert_eq!(cfg.object_type, 2);
        assert_eq!(cfg.sample_rate, 44100);
        assert_eq!(cfg.channels, 2);
    }

    #[test]
    fn parses_asc_with_dsi_header() {
        let with_header = [0x05u8, 0x02, 0x12, 0x10];
        let cfg = AudioSpecificConfig::parse(&with_header).unwrap();
        assert_eq!(cfg.sample_rate, 44100);
        assert_eq!(cfg.channels, 2);
    }

    #[test]
    fn rejects_unsupported_profiles() {
        // 配置解析本身宽容（object type 原样报出来），由解码器 open 时拒绝
        let asc_sbr = [0b00101_010, 0b0100_0_000];
        assert_eq!(AudioSpecificConfig::parse(&asc_sbr).unwrap().object_type, 5);
        let e = match AacDecoder::open(&asc_sbr) {
            Ok(_) => panic!("SBR 还没接，不该成功"),
            Err(e) => e,
        };
        let text = format!("{e}");
        assert!(text.contains("SBR") || text.contains("LC"), "错误里要点明是什么没接：{text}");
        // 5.1（声道配置 6）
        let asc5 = [0x12u8, 0x18];
        assert!(AacDecoder::open(&asc5).is_err(), "多声道还没接，要明确报错");
    }

    #[test]
    fn strips_adts_header() {
        // 7 字节 ADTS（protection_absent = 1）+ 载荷
        let mut p = vec![0xFFu8, 0xF1, 0x50, 0x80, 0x00, 0x1F, 0xFC];
        p.extend_from_slice(&[0xAA, 0xBB]);
        assert_eq!(strip_adts(&p), &[0xAA, 0xBB]);
        // 9 字节 ADTS（带 CRC）
        let mut q = vec![0xFFu8, 0xF0, 0x50, 0x80, 0x00, 0x1F, 0xFC, 0x11, 0x22];
        q.extend_from_slice(&[0xCC]);
        assert_eq!(strip_adts(&q), &[0xCC]);
        // 不是 ADTS 就原样返回（fMP4 的裸帧）
        let raw = [0x01u8, 0x02, 0x03];
        assert_eq!(strip_adts(&raw), &raw);
    }

    /// 测试用的位写入器（按 MSB-first 写，与 BitReader 对称）。
    struct BitWriter {
        bits: Vec<u8>,
    }

    impl BitWriter {
        fn new() -> Self {
            Self { bits: Vec::new() }
        }
        fn push(&mut self, value: u32, count: u8) -> &mut Self {
            for i in (0..count).rev() {
                self.bits.push(((value >> i) & 1) as u8);
            }
            self
        }
        fn bytes(&self) -> Vec<u8> {
            let mut out = Vec::new();
            let mut cur = 0u8;
            let mut n = 0;
            for &b in &self.bits {
                cur = (cur << 1) | b;
                n += 1;
                if n == 8 {
                    out.push(cur);
                    cur = 0;
                    n = 0;
                }
            }
            if n > 0 {
                out.push(cur << (8 - n));
            }
            out
        }
    }

    fn lc_decoder() -> AacDecoder {
        // LC / 44100 / 立体声
        AacDecoder::open(&[0x12, 0x10]).unwrap()
    }

    #[test]
    fn parses_long_window_ics_info() {
        let d = lc_decoder();
        let mut w = BitWriter::new();
        let p = &mut w;
        let b = p.push(0, 1) // 保留位
            .push(0, 2)     // ONLY_LONG
            .push(0, 1)     // 正弦窗
            .push(40, 6)    // max_sfb
            .push(0, 1)     // 无长时预测
            .bytes();
        let mut r = BitReader::new(&b);
        let info = d.read_ics_info(&mut r).unwrap();
        assert_eq!(info.sequence, Sequence::OnlyLong);
        assert_eq!(info.shape, Shape::Sine);
        assert_eq!(info.num_windows, 1);
        assert_eq!(info.max_sfb, 40);
        assert_eq!(info.group_len, vec![1]);
        // 44100 属于"48K 档"（频带边界末值 1024）
        assert_eq!(*info.long_swb.last().unwrap(), 1024);
        assert_eq!(*info.short_swb.last().unwrap(), 128);
    }

    #[test]
    fn parses_eight_short_ics_info_with_grouping() {
        let d = lc_decoder();
        // 分组位 1011000：bit i 表示"窗 i+1 并入窗 i 所在的组"
        //   b0=1 → w1 并入 w0；b1=0 → w2 另起；b2=1 → w3 并入；b3=1 → w4 并入；
        //   b4..b6=0 → w5/w6/w7 各自成组  ⇒ 组长度 [2,3,1,1,1]
        let mut w = BitWriter::new();
        let b = w
            .push(0, 1)
            .push(2, 2) // EIGHT_SHORT
            .push(1, 1) // KBD 窗
            .push(12, 4) // max_sfb
            .push(0b1011000, 7)
            .bytes();
        let mut r = BitReader::new(&b);
        let info = d.read_ics_info(&mut r).unwrap();
        assert_eq!(info.sequence, Sequence::EightShort);
        assert_eq!(info.shape, Shape::Kbd);
        assert_eq!(info.num_windows, 8);
        assert_eq!(info.max_sfb, 12);
        assert_eq!(info.group_len, vec![2, 3, 1, 1, 1], "分组位要按 AAC 的规则切开");
        assert_eq!(info.group_len.iter().sum::<usize>(), 8, "8 个窗都要有归属");
    }

    #[test]
    fn rejects_long_term_prediction() {
        let d = lc_decoder();
        let mut w = BitWriter::new();
        let b = w
            .push(0, 1)
            .push(0, 2)
            .push(0, 1)
            .push(20, 6)
            .push(1, 1) // 有长时预测 → LC 不该出现
            .bytes();
        let mut r = BitReader::new(&b);
        let e = d.read_ics_info(&mut r).unwrap_err();
        assert!(format!("{e}").contains("长时预测"), "要明确说清是什么没接：{e}");
    }

    #[test]
    fn parses_sections_with_escapes() {
        let d = lc_decoder();
        let info = IcsInfo {
            sequence: Sequence::OnlyLong,
            shape: Shape::Sine,
            max_sfb: 6,
            num_windows: 1,
            group_len: vec![1],
            long_swb: tables::subband_tables(44100).unwrap().0,
            short_swb: tables::subband_tables(44100).unwrap().1,
        };
        // 两段（段长字段就是"覆盖的 sfb 数"，**不加 1** —— 对照 FFmpeg 校正过）：
        // 第一段 cb=1 覆盖 2 个 sfb，第二段 cb=11 覆盖 4 个 sfb
        let mut w = BitWriter::new();
        let b = w
            .push(1, 4)
            .push(2, 5)
            .push(11, 4)
            .push(4, 5)
            .bytes();
        let mut r = BitReader::new(&b);
        let sections = d.read_sections(&mut r, &info).unwrap();
        let first = &sections.groups[0];
        assert_eq!(first.len(), 2);
        assert_eq!(first[0], Section { codebook: 1, sfb_count: 2 });
        assert_eq!(first[1], Section { codebook: 11, sfb_count: 4 });
    }

    #[test]
    fn section_escape_extends_length() {
        let d = lc_decoder();
        let info = IcsInfo {
            sequence: Sequence::OnlyLong,
            shape: Shape::Sine,
            max_sfb: 40,
            num_windows: 1,
            group_len: vec![1],
            long_swb: tables::subband_tables(44100).unwrap().0,
            short_swb: tables::subband_tables(44100).unwrap().1,
        };
        // 段长字段 31（全 1）→ 继续读并累加：31 + 9 = 40（覆盖 40 个 sfb，正好 max_sfb）
        let mut w = BitWriter::new();
        let b = w.push(3, 4).push(31, 5).push(9, 5).bytes();
        let mut r = BitReader::new(&b);
        let sections = d.read_sections(&mut r, &info).unwrap();
        assert_eq!(sections.groups[0][0].sfb_count, 40, "转义要把增量累加起来");
    }

    #[test]
    fn section_codebook_15_is_intensity_not_escape() {
        // 15 是 `INTENSITY_HCB`（强度立体声），**不是"码表转义"**。
        // 曾经按 `while cb == 15 { cb += read(4) }` 读：真机上一帧右声道的第 2 段正好
        // 以 15 开头，于是被拼成 15+6=21（LC 里不存在）→ 整帧报废。
        let d = lc_decoder();
        let info = IcsInfo {
            sequence: Sequence::OnlyLong,
            shape: Shape::Sine,
            max_sfb: 2,
            num_windows: 1,
            group_len: vec![1],
            long_swb: tables::subband_tables(44100).unwrap().0,
            short_swb: tables::subband_tables(44100).unwrap().1,
        };
        // 段 1：cb=15 覆盖 1 个 sfb；段 2：cb=14 覆盖 1 个 sfb（都合法）
        let mut w = BitWriter::new();
        let b = w.push(15, 4).push(1, 5).push(14, 4).push(1, 5).bytes();
        let mut r = BitReader::new(&b);
        let secs = d.read_sections(&mut r, &info).unwrap();
        assert_eq!(secs.groups[0][0], Section { codebook: 15, sfb_count: 1 });
        assert_eq!(secs.groups[0][1], Section { codebook: 14, sfb_count: 1 });
        // 12 是保留值 → 必须明确报错（别静默当 0 用）
        let mut w2 = BitWriter::new();
        let b2 = w2.push(12, 4).push(1, 5).bytes();
        let mut r2 = BitReader::new(&b2);
        let e = d.read_sections(&mut r2, &info).unwrap_err();
        assert!(format!("{e}").contains("12"), "错误里要点明码表号 12 非法：{e}");
    }

    #[test]
    fn scalefactors_accumulate_from_global_gain() {
        let d = lc_decoder();
        let info = IcsInfo {
            sequence: Sequence::OnlyLong,
            shape: Shape::Sine,
            max_sfb: 3,
            num_windows: 1,
            group_len: vec![1],
            long_swb: tables::subband_tables(44100).unwrap().0,
            short_swb: tables::subband_tables(44100).unwrap().1,
        };
        let sections = Sections {
            groups: vec![vec![Section { codebook: 1, sfb_count: 3 }]],
        };
        // 三个比例因子的差分：+1、-2、+3（用比例因子码表把"下标 = 值 + 60"写成位流）
        let (lens, codes) = (tables::SCALEFACTOR.lens, tables::SCALEFACTOR.codes);
        let mut w = BitWriter::new();
        for delta in [1i32, -2, 3] {
            let idx = (delta + 60) as usize;
            w.push(codes[idx], lens[idx]);
        }
        let bytes = w.bytes();
        let mut r = BitReader::new(&bytes);
        let sf = d.read_scalefactors(&mut r, &info, &sections, 100).unwrap();
        // global_gain=100 → 101、99、102
        assert_eq!(sf[0][0], 101);
        assert_eq!(sf[0][1], 99);
        assert_eq!(sf[0][2], 102);
    }

    #[test]
    fn build_decoder_exposes_stream_info() {
        let d = AacDecoder::open(&[0x12, 0x10]).unwrap();
        assert_eq!(d.stream_info(), (44100, 2, 16));
    }

    /// 正向 MDCT（与 `imdct` 互为转置）—— 只给下面的完美重建测试用。
    ///
    /// **必须带因子 2**：`(2/N)·M·Mᵀ` 的对角线是 0.5 而不是 1，所以只有
    /// "分析 ×2、合成 2/N"这一对才满足 `Σ_k cos·cos = δ` 的归一化
    /// （AAC 规范里正向 MDCT 就是 `2·Σ x·cos(...)`）。
    /// 少这个 2 的症状是"整条链路增益只有一半"，很好认。
    ///
    /// 为什么要自己写"分析"这一侧：滤波器组对不对**没法靠读代码确认**（长/短窗的
    /// 平坦段与短窗半段是几块拼出来的）。只有"分析 → 合成 → 跨帧叠加 == 原信号"
    /// 能一票定音，而且它不需要任何样例文件。
    fn mdct_forward(block: &[f32], out: &mut [f32]) {
        let n = block.len();
        let n0 = (n / 4) as f32 + 0.5;
        for (k, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (i, &x) in block.iter().enumerate() {
                let ang =
                    2.0 * std::f32::consts::PI / n as f32 * (i as f32 + n0) * (k as f32 + 0.5);
                acc += x * ang.cos();
            }
            *o = 2.0 * acc;
        }
    }

    /// 测试用的窗表（与 `AacDecoder::open` 里建的那两张一模一样）。
    fn test_windows() -> ([(Shape, Vec<f32>); 2], [(Shape, Vec<f32>); 2]) {
        (
            [
                (Shape::Sine, imdct::window(WindowShape::Sine, 2048, 4.0)),
                (Shape::Kbd, imdct::window(WindowShape::Kbd, 2048, 4.0)),
            ],
            [
                (Shape::Sine, imdct::window(WindowShape::Sine, 256, 6.0)),
                (Shape::Kbd, imdct::window(WindowShape::Kbd, 256, 6.0)),
            ],
        )
    }

    /// 走一遍"分析 → 合成 → 交叉叠加"，返回（最大误差, 出错位置）。
    /// `seqs` 是每个窗序列；`only_long_sanity` 那种对照也复用它。
    fn reconstruction_error(seqs: &[Sequence]) -> (f32, usize) {
        let (long_windows, short_windows) = test_windows();
        let (long_swb, short_swb) = tables::subband_tables(44100).unwrap();
        let long_win = &long_windows[0].1;
        let short_win = &short_windows[0].1;

        // 确定性的伪随机信号 + 一个瞬态：第 2~4 帧突然变响（正是编码器会用短窗的地方）
        let mut signal = vec![0f32; (seqs.len() + 1) * 1024 + 2048];
        let mut rng = 0x2468_1357u32;
        for (i, s) in signal.iter_mut().enumerate() {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let amp = if (2 * 1024..4 * 1024).contains(&i) { 0.5 } else { 0.05 };
            *s = (rng as f32 / 2147483648.0 - 1.0) * amp;
        }

        let mut state = ChannelState::new();
        let mut got = vec![0f32; seqs.len() * 1024];
        for (k, seq) in seqs.iter().enumerate() {
            let info = IcsInfo {
                sequence: *seq,
                shape: Shape::Sine,
                max_sfb: 0,
                num_windows: seq.num_windows(),
                group_len: if *seq == Sequence::EightShort { vec![1; 8] } else { vec![1] },
                long_swb,
                short_swb,
            };
            let x = &signal[k * 1024..k * 1024 + 2048];
            let planes: Vec<Vec<f32>> = if *seq == Sequence::EightShort {
                (0..8)
                    .map(|m| {
                        let base = 448 + 128 * m;
                        let windowed: Vec<f32> =
                            (0..256).map(|i| x[base + i] * short_win[i]).collect();
                        let mut coef = vec![0f32; 128];
                        mdct_forward(&windowed, &mut coef);
                        coef
                    })
                    .collect()
            } else {
                let w = seq_window(*seq, long_win, Shape::Sine, &short_windows);
                let windowed: Vec<f32> = (0..2048).map(|i| x[i] * w[i]).collect();
                let mut coef = vec![0f32; 1024];
                mdct_forward(&windowed, &mut coef);
                vec![coef]
            };
            let mut rendered = Vec::new();
            AacDecoder::synthesize(
                &mut state,
                &info,
                &planes,
                &long_windows,
                &short_windows,
                &mut rendered,
            );
            assert_eq!(rendered.len(), 1024);
            got[k * 1024..(k + 1) * 1024].copy_from_slice(&rendered);
        }

        // 首尾各有一半窗没被"上一帧 / 下一帧"补齐，只比中间那些帧
        let mut worst = 0f32;
        let mut worst_at = 0usize;
        for i in 2 * 1024..(seqs.len() - 1) * 1024 {
            let e = (got[i] - signal[i]).abs();
            if e > worst {
                worst = e;
                worst_at = i;
            }
        }
        (worst, worst_at)
    }

    #[test]
    fn transition_windows_keep_their_flat_segments() {
        // 过渡窗的几何（对照规范 4.6.11.2.3 的窗定义）：短序列只占 `[448,1600)`，
        // 于是两侧长窗各有一段"没有任何短块供能"的平坦区，必须**原样放行**（1.0）。
        // 这两段被写成 0，就是 2026-09-27 那个"瞬态丢 448 个样本"的 bug；
        // 短窗半段取错（上升/下降）症状一模一样，所以这里逐段钉住。
        let (long_windows, short_windows) = test_windows();
        let long = &long_windows[0].1;
        let short = &short_windows[0].1;

        let start = seq_window(Sequence::LongStart, long, Shape::Sine, &short_windows);
        assert_eq!(&start[..1024], &long[..1024], "左半还是普通长窗");
        assert!(
            start[1024..1472].iter().all(|&v| v == 1.0),
            "LONG_START 的平坦段必须是 1.0（短序列够不着这里）"
        );
        assert_eq!(&start[1472..1600], &short[128..], "尾巴要用短窗的下降段");
        assert!(start[1600..].iter().all(|&v| v == 0.0), "再往后要留给短序列");

        let stop = seq_window(Sequence::LongStop, long, Shape::Sine, &short_windows);
        assert!(stop[..448].iter().all(|&v| v == 0.0), "开头要留给短序列");
        assert_eq!(&stop[448..576], &short[..128], "开头要用短窗的上升段");
        assert!(
            stop[576..1024].iter().all(|&v| v == 1.0),
            "LONG_STOP 的平坦段必须是 1.0（接住短序列送过来的那一段）"
        );
        assert_eq!(&stop[1024..], &long[1024..], "右半还是普通长窗");
    }

    #[test]
    fn filterbank_is_transparent_for_steady_long_windows() {
        // 对照：全程长窗（没有任何过渡）。这一条要是也不过，说明测试口径本身有问题
        // （比如分析侧的公式与 `imdct` 不是一对），先修测试再谈窗序列。
        let (worst, at) = reconstruction_error(&[Sequence::OnlyLong; 8]);
        assert!(worst < 1e-3, "稳态长窗应当完美重建：第 {at} 个样本差 {worst:.6}");
    }

    #[test]
    fn filterbank_reconstructs_the_signal_across_window_sequences() {
        // 滤波器组的完美重建自校验（不依赖任何样例文件）：把一段**已知信号**按 AAC 的
        // 分析侧处理（按窗序列取 2048 点 → 加窗 → MDCT；短序列切成 8 个 256 点窗），
        // 再喂给生产代码 `synthesize`（IMDCT → 加窗 → 跨帧叠加），输出必须还原原信号。
        //
        // 为什么值得专门写这一条：长↔短窗过渡的窗口是几块拼出来的（平坦段、短窗的
        // 上升/下降段、短序列的 448 偏移），**写错任何一段的症状都是"某 448 个样本
        // 被清零或重复"** —— 2026-09-27 就在真实码流上踩到：正弦的开头与结尾各静了
        // 一小段（听感上像卡了一下），而"逐频率分量误差 <1%"这种统计口径看不出来。
        let seqs = [
            Sequence::OnlyLong,
            Sequence::OnlyLong,
            Sequence::LongStart,
            Sequence::EightShort,
            Sequence::EightShort,
            Sequence::LongStop,
            Sequence::OnlyLong,
            Sequence::OnlyLong,
        ];
        let (worst, at) = reconstruction_error(&seqs);
        assert!(
            worst < 1e-3,
            "长↔短窗过渡没能还原信号：第 {at} 个样本差 {worst:.6}（看一看是不是 448 的整数倍附近）"
        );

        // 单个短序列帧（前后都是长窗）也要能还原：这是"只有一次瞬态"的常见形态
        let (worst, at) = reconstruction_error(&[
            Sequence::OnlyLong,
            Sequence::LongStart,
            Sequence::EightShort,
            Sequence::LongStop,
            Sequence::OnlyLong,
        ]);
        assert!(worst < 1e-3, "单帧短序列没能还原：第 {at} 个样本差 {worst:.6}");
    }
}
