//! Opus 的**配置盒**在两种容器里是两种字节形态，本模块是唯一的转换处：
//!
//! | 容器 | 形态 | 字节序 | 开头 |
//! |---|---|---|---|
//! | Ogg（`.opus`） | `OpusHead`（RFC 7845 §5.1） | **小端** | 魔数 `OpusHead` |
//! | ISOBMFF（`.mp4`） | `dOps`（Opus in ISOBMFF） | **大端** | 版本字节 |
//!
//! 两者的字段**完全一样**（版本、声道数、pre-skip、输入采样率、输出增益、声道映射族
//! 与映射表），差别只有"魔数 + 字节序"。所以转换就是换字节序、补/去魔数。
//!
//! ## 为什么全引擎统一用 `OpusHead` 形态（这一点与视频不同）
//!
//! 视频的 `extradata` 有形态之分（Annex-B / `avcC`），因为它同时决定**采样数据**
//! 的排列（要不要带长度前缀），所以解析层如实报告来源形态、由 muxer 转换
//! （见 `me-mux::convert` 的约定）。Opus 不一样：
//!
//! - `extradata` **只**是解码器的初始化数据；
//! - 采样数据（一个 Opus 包）在两种容器里是**同一串字节**，没有任何形态可言。
//!
//! 所以我们把 `dOps` 在读入时**就地规范化成 `OpusHead`**（Ogg 侧原生就是它），
//! 解码器（`me-decode`）于是只有一条路径、不需要认识容器。
//!
//! ## 两个容易踩的点
//!
//! 1. **pre-skip 与输入采样率都是"编码前的元数据"**：`input_sample_rate` 是编码器
//!    输入端的采样率（可能是 44100），**不是解码采样率** —— Opus 的解码采样率
//!    按定义恒为 48 kHz（见 [`OPUS_RATE`]）。
//! 2. **`dOps` 不是 FullBox**：它没有 `version+flags` 那 4 个字节，第 0 字节就是
//!    规范里的 `Version` 字段（值为 0）。当成 FullBox 写会让后面所有字段整体错位 4 字节，
//!    而 `dOps` 的字段又都是"数值看起来都合法"的那种（错位后仍能读出一个像样的
//!    采样率与声道数），是典型的"能写出来但解码器起不来"。

use me_core::{MediaError, Result};

/// `OpusHead` 的最小长度：魔数 8 + 版本 1 + 声道 1 + pre-skip 2 + 输入采样率 4 +
/// 输出增益 2 + 映射族 1。
pub const OPUS_HEAD_MIN: usize = 19;

/// `dOps` 载荷的最小长度：版本 1 + 声道 1 + pre-skip 2 + 输入采样率 4 +
/// 输出增益 2 + 映射族 1（**不含** 8 字节 box 头）。
pub const DOPS_MIN: usize = 11;

/// Opus 的**解码**采样率：按规范恒为 48 kHz，与 `input_sample_rate` 无关。
pub const OPUS_RATE: u32 = 48_000;

/// Opus 的初始化参数（`OpusHead` 与 `dOps` 的共同内容）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpusConfig {
    /// 声道数（1 / 2 之外的取值只在映射族非 0 时合法）。
    pub channels: u8,
    /// 解码时要丢掉的前导样点数（编码器/重采样链的固有延迟，典型 312）。
    pub pre_skip: u16,
    /// 编码前的原始采样率（**元数据**，不是解码采样率）。
    pub input_sample_rate: u32,
    /// 输出增益 Q7.8（dB）。
    pub output_gain_q78: i16,
    /// 声道映射族：0 = 单声道/立体声，其余取值带映射表。
    pub mapping_family: u8,
    /// 编码器输入端的流数（映射族 0 时隐含 1）。
    pub stream_count: u8,
    /// 耦合（立体声）流数（映射族 0 时隐含"双声道即 1"）。
    pub coupled_count: u8,
    /// 声道映射表（映射族非 0 时才有，长度 = 声道数）。
    pub channel_mapping: Vec<u8>,
}

impl OpusConfig {
    /// 解析 `OpusHead`（Ogg 形态，小端）。
    pub fn parse_head(p: &[u8]) -> Result<Self> {
        if p.len() < OPUS_HEAD_MIN || !p.starts_with(b"OpusHead") {
            return Err(MediaError::container("不是 OpusHead（长度不足或魔数不对）"));
        }
        let version = p[8];
        if version >> 4 >= 1 {
            return Err(MediaError::Unsupported(format!(
                "Opus 封装版本 {version} 不认识（本引擎认 1）"
            )));
        }
        let channels = p[9];
        if channels == 0 {
            return Err(MediaError::container("OpusHead 的声道数是 0（规范要求非 0）"));
        }
        let pre_skip = u16::from_le_bytes([p[10], p[11]]);
        let input_sample_rate = u32::from_le_bytes([p[12], p[13], p[14], p[15]]);
        let output_gain_q78 = i16::from_le_bytes([p[16], p[17]]);
        let mapping_family = p[18];
        let (stream_count, coupled_count, channel_mapping) = if mapping_family == 0 {
            // 映射族 0 时规范不写这三个字段，它们是**隐含值**
            (1u8, u8::from(channels > 1), Vec::new())
        } else {
            if p.len() < 21 + channels as usize {
                return Err(MediaError::container("OpusHead 的声道映射表被截断"));
            }
            (p[19], p[20], p[21..21 + channels as usize].to_vec())
        };
        Ok(Self {
            channels,
            pre_skip,
            input_sample_rate,
            output_gain_q78,
            mapping_family,
            stream_count,
            coupled_count,
            channel_mapping,
        })
    }

    /// 生成 `OpusHead`（Ogg 形态，小端）。
    ///
    /// 版本写 **1**（RFC 7845 的当前版本；v1 与早期 v0 完全兼容，本引擎两侧都认）。
    pub fn to_head(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(OPUS_HEAD_MIN + self.channel_mapping.len());
        v.extend_from_slice(b"OpusHead");
        v.push(1); // version
        v.push(self.channels);
        v.extend_from_slice(&self.pre_skip.to_le_bytes());
        v.extend_from_slice(&self.input_sample_rate.to_le_bytes());
        v.extend_from_slice(&self.output_gain_q78.to_le_bytes());
        v.push(self.mapping_family);
        if self.mapping_family != 0 {
            v.push(self.stream_count);
            v.push(self.coupled_count);
            v.extend_from_slice(&self.channel_mapping);
        }
        v
    }

    /// 解析 `dOps` 载荷（ISOBMFF 形态，大端；**不含 box 头**）。
    pub fn parse_dops(p: &[u8]) -> Result<Self> {
        if p.len() < DOPS_MIN {
            return Err(MediaError::container(format!(
                "dOps 只有 {} 字节（至少 {DOPS_MIN} 字节）",
                p.len()
            )));
        }
        if p[0] != 0 {
            return Err(MediaError::Unsupported(format!(
                "dOps 的版本 {} 不认识（本引擎认 0）",
                p[0]
            )));
        }
        let channels = p[1];
        if channels == 0 {
            return Err(MediaError::container("dOps 的声道数是 0（规范要求非 0）"));
        }
        let pre_skip = u16::from_be_bytes([p[2], p[3]]);
        let input_sample_rate = u32::from_be_bytes([p[4], p[5], p[6], p[7]]);
        let output_gain_q78 = i16::from_be_bytes([p[8], p[9]]);
        let mapping_family = p[10];
        let (stream_count, coupled_count, channel_mapping) = if mapping_family == 0 {
            (1u8, u8::from(channels > 1), Vec::new())
        } else {
            if p.len() < 13 + channels as usize {
                return Err(MediaError::container("dOps 的声道映射表被截断"));
            }
            (p[11], p[12], p[13..13 + channels as usize].to_vec())
        };
        Ok(Self {
            channels,
            pre_skip,
            input_sample_rate,
            output_gain_q78,
            mapping_family,
            stream_count,
            coupled_count,
            channel_mapping,
        })
    }

    /// 生成 `dOps` 载荷（ISOBMFF 形态，大端；**不含 box 头**）。
    ///
    /// 版本写 **0**：规范对 `dOps` 的 `Version` 明确要求 0（与 `OpusHead` 的
    /// 版本字段含义不同，它标记的是**这个盒子的布局**）。
    pub fn to_dops(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(DOPS_MIN + self.channel_mapping.len());
        v.push(0); // Version
        v.push(self.channels);
        v.extend_from_slice(&self.pre_skip.to_be_bytes());
        v.extend_from_slice(&self.input_sample_rate.to_be_bytes());
        v.extend_from_slice(&self.output_gain_q78.to_be_bytes());
        v.push(self.mapping_family);
        if self.mapping_family != 0 {
            v.push(self.stream_count);
            v.push(self.coupled_count);
            v.extend_from_slice(&self.channel_mapping);
        }
        v
    }
}

/// `dOps` 载荷 → `OpusHead`（引擎统一的 Opus 配置形态）。
pub fn dops_to_opus_head(dops: &[u8]) -> Result<Vec<u8>> {
    Ok(OpusConfig::parse_dops(dops)?.to_head())
}

/// `OpusHead` → `dOps` 载荷（写进 fMP4 的采样条目）。
pub fn opus_head_to_dops(head: &[u8]) -> Result<Vec<u8>> {
    Ok(OpusConfig::parse_head(head)?.to_dops())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 造一份 `OpusHead`（小端、带魔数）——**不是** dOps。
    fn opus_head_bytes(channels: u8, pre_skip: u16, family: u8) -> Vec<u8> {
        let mut v = b"OpusHead".to_vec();
        v.push(1);
        v.push(channels);
        v.extend_from_slice(&pre_skip.to_le_bytes());
        v.extend_from_slice(&48_000u32.to_le_bytes());
        v.extend_from_slice(&0i16.to_le_bytes());
        v.push(family);
        if family != 0 {
            v.push(1); // stream count
            v.push(u8::from(channels > 1)); // coupled count
            for i in 0..channels {
                v.push(i); // 映射表
            }
        }
        v
    }

    #[test]
    fn opus_head_round_trips_through_dops() {
        // 单声道与立体声：dOps 再读回来必须与 OpusHead 逐字节相同
        for channels in [1u8, 2] {
            let h = opus_head_bytes(channels, 356, 0);
            let d = opus_head_to_dops(&h).expect("OpusHead 应当能转成 dOps");
            assert_eq!(d.len(), DOPS_MIN, "映射族 0 的 dOps 就是 11 字节");
            assert_eq!(d[0], 0, "dOps 的第 0 字节是版本（规范要求 0）");
            assert_eq!(d[1], channels);
            let back = dops_to_opus_head(&d).expect("dOps 应当能转回 OpusHead");
            assert_eq!(back, h, "OpusHead → dOps → OpusHead 必须无损失");
        }
    }

    #[test]
    fn dops_pre_skip_and_gain_are_big_endian() {
        // pre-skip 与增益在 dOps 里是大端：写成小端只会让 pre-skip 变成一个
        // "看起来也合理"的别的数值（比如 356 变成 57601），极难发现
        let d = opus_head_to_dops(&opus_head_bytes(2, 356, 0)).unwrap();
        assert_eq!(u16::from_be_bytes([d[2], d[3]]), 356);
        assert_eq!(u32::from_be_bytes([d[4], d[5], d[6], d[7]]), 48_000);

        let mut with_gain = opus_head_bytes(2, 312, 0);
        with_gain[16] = 0x02; // output gain = 0x0102 (LE) = 258
        with_gain[17] = 0x01;
        let d = opus_head_to_dops(&with_gain).unwrap();
        assert_eq!(i16::from_be_bytes([d[8], d[9]]), 258, "增益也要换成大端");
    }

    #[test]
    fn channel_mapping_survives_the_round_trip() {
        // 映射族非 0 时还要带 stream/coupled 与映射表
        let h = opus_head_bytes(4, 120, 1);
        let d = opus_head_to_dops(&h).unwrap();
        assert_eq!(d.len(), DOPS_MIN + 2 + 4, "多出 stream/coupled 两个字节与映射表");
        assert_eq!(d[10], 1, "映射族要跟着过去");
        assert_eq!(&d[11..13], &[1, 1], "stream / coupled 计数");
        assert_eq!(&d[13..], &[0, 1, 2, 3], "映射表");
        assert_eq!(dops_to_opus_head(&d).unwrap(), h);
    }

    #[test]
    fn truncated_dops_is_an_error_not_a_silent_default() {
        let mut d = opus_head_to_dops(&opus_head_bytes(2, 356, 0)).unwrap();
        d.truncate(DOPS_MIN - 1);
        assert!(OpusConfig::parse_dops(&d).is_err());

        // 映射族非 0 但映射表缺一半：必须报错（不能当成立体声放过去）
        let d = opus_head_to_dops(&opus_head_bytes(4, 120, 1)).unwrap();
        let e = OpusConfig::parse_dops(&d[..16]).unwrap_err();
        assert!(e.to_string().contains("映射表"), "错误要说清缺的是映射表：{e}");
    }

    #[test]
    fn unknown_dops_version_is_refused() {
        let mut d = opus_head_to_dops(&opus_head_bytes(2, 356, 0)).unwrap();
        d[0] = 7;
        let e = OpusConfig::parse_dops(&d).unwrap_err();
        assert!(e.to_string().contains("版本"), "{e}");
    }

    #[test]
    fn head_errors_stay_specific() {
        assert!(OpusConfig::parse_head(b"NotOpusHead........").is_err());
        let mut h = opus_head_bytes(2, 356, 0);
        h[9] = 0;
        assert!(OpusConfig::parse_head(&h).unwrap_err().to_string().contains("声道数是 0"));
        let mut h = opus_head_bytes(2, 356, 0);
        h[8] = 0x20; // 版本 32：高 4 位 >= 1 表示不兼容
        assert!(OpusConfig::parse_head(&h).unwrap_err().to_string().contains("版本"));
    }

    #[test]
    fn output_gain_is_signed() {
        // 增益是有符号的：负增益（衰减）不能被当成 65400 这样的大正数
        let mut h = opus_head_bytes(1, 0, 0);
        h[16] = 0x00;
        h[17] = 0xFF; // -256 (LE)
        let cfg = OpusConfig::parse_head(&h).unwrap();
        assert_eq!(cfg.output_gain_q78, -256);
        assert_eq!(cfg.to_dops()[8..10], (-256i16).to_be_bytes());
    }
}
