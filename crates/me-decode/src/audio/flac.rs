//! **自研 FLAC 解码器**（无损音频）。
//!
//! 为什么从 FLAC 先做：它是无损格式，能拿系统解码器当参考做**逐样本 bit-exact**
//! 校验 —— 一个 bit 都不差，才敢说解码器是对的（有损格式只能比统计量，
//! 错了也看不出来）。AAC/MP3/Opus 按同一套接口跟在后面。
//!
//! ## 覆盖范围
//!
//! 覆盖 FLAC 规范里**实际会遇到的**全部子帧类型：constant / verbatim / fixed
//! 0..4 阶 / LPC 1..32 阶、Rice 分区（方法 0/1，含 escape 原样编码）、
//! 独立立体声与 left/right/mid-side 三种声道去相关、wasted bits（位深对齐）、
//! 帧头 CRC-8 与帧尾 CRC-16 校验。保留字段（子帧类型 2..7、13..31）按规范报错。
//!
//! ## 一个必须记住的细节
//!
//! FLAC 的 **Rice 残差是"折叠"有符号**：`v = (u >> 1) ^ -(u & 1)`（u 为无符号读到的值）。
//! 少了这一步，残差全变正数，解出来的声音是刺啦声 —— 而且音量、时长、帧结构
//! 看起来都"正常"，非常难查。

use me_core::{MediaError, Result};

use super::bitreader::BitReader;
use super::{AudioDecoder, PcmChunk};

/// STREAMINFO 的载荷长度（固定 34 字节）。
const STREAMINFO_LEN: usize = 34;

/// 流信息（`STREAMINFO` 元数据块）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamInfo {
    pub min_block_size: u16,
    pub max_block_size: u16,
    pub sample_rate: u32,
    pub channels: u8,
    pub bits_per_sample: u8,
    /// 总样本数（每声道）；0 = 未知
    pub total_samples: u64,
}

/// 自研 FLAC 解码器。
pub struct FlacDecoder {
    info: StreamInfo,
    /// 已解出的样本总数（每声道）—— 给"包里没有时间戳"的调用方兜底算时间
    decoded_samples: u64,
}

impl FlacDecoder {
    /// 用解码器初始化数据建解码器。
    ///
    /// 三种形态都要认（都是真会遇到的）：
    /// - `fLaC` 开头的**裸 FLAC 文件头**（元数据块序列）；
    /// - **`dfLa` 载荷**（MP4：`version+flags(4)` + 元数据块头(4) + STREAMINFO(34)）；
    /// - **裸 STREAMINFO**（34 字节，某些封装/中间表示）。
    pub fn open(extradata: &[u8]) -> Result<Self> {
        let si = parse_streaminfo_any(extradata)?;
        if si.sample_rate == 0 {
            return Err(MediaError::Bitstream("FLAC 的采样率是 0（STREAMINFO 没读到）".into()));
        }
        if si.channels == 0 || si.channels > 8 {
            return Err(MediaError::Bitstream(format!("FLAC 声道数不合法：{}", si.channels)));
        }
        Ok(Self { info: si, decoded_samples: 0 })
    }

    pub fn stream_info_fields(&self) -> StreamInfo {
        self.info
    }

    /// 解一帧（FLAC 的一个 packet 就是一帧）。
    fn decode_frame(&mut self, packet: &[u8], pts_us: i64) -> Result<PcmChunk> {
        let mut r = BitReader::new(packet);
        let sync = r.read(14)?;
        if sync != 0x3FFE {
            return Err(MediaError::Bitstream(format!(
                "FLAC 帧同步码不对（读到 {sync:#x}，应当是 0x3ffe）"
            )));
        }
        if r.read(1)? != 0 {
            return Err(MediaError::Bitstream("FLAC 帧头的保留位不是 0".into()));
        }
        let variable_blocking = r.read(1)? == 1;
        let block_size_code = r.read(4)?;
        let sample_rate_code = r.read(4)?;
        let channel_code = r.read(4)?;
        let sample_size_code = r.read(3)?;
        if r.read(1)? != 0 {
            return Err(MediaError::Bitstream("FLAC 帧头的保留位（末尾）不是 0".into()));
        }

        // 帧号（定长块）或样本号（变长块）：UTF-8 风格编码，最长 31/36 位
        let _number = if variable_blocking {
            read_utf8_number(&mut r, 36)?
        } else {
            read_utf8_number(&mut r, 31)?
        };

        // 帧头 CRC-8：到 CRC 字段之前的字节算出来要对得上
        // **块长与采样率的扩展字段也在 CRC-8 之前**（规范里帧头字段就是这个顺序）：
        // 块长码 7 → 16 位字面块长；采样率码 12/13/14 → 8/16 位扩展。少读或多读
        // 都会让 CRC 的字节范围错位（实测踩过：最后一帧块长码是 7）。
        let block_size = decode_block_size(&mut r, block_size_code)?;
        let sample_rate = decode_sample_rate(&mut r, sample_rate_code, self.info.sample_rate)?;

        let crc8_at = r.byte_pos();
        let got_crc8 = r.read(8)? as u8;
        let want_crc8 = crc8(&packet[..crc8_at]);
        if got_crc8 != want_crc8 {
            return Err(MediaError::Bitstream(format!(
                "FLAC 帧头 CRC-8 不符（读到 {got_crc8:#x}，算出 {want_crc8:#x}）"
            )));
        }

        let channels = match channel_code {
            0..=7 => (channel_code + 1) as usize,
            8..=10 => 2,
            other => {
                return Err(MediaError::Bitstream(format!(
                    "FLAC 声道分配码 {other} 是保留值"
                )))
            }
        };
        let bps = match sample_size_code {
            0 => self.info.bits_per_sample,
            1 => 8,
            2 => 12,
            4 => 16,
            5 => 20,
            6 => 24,
            7 => 32,
            other => {
                return Err(MediaError::Bitstream(format!(
                    "FLAC 位深码 {other} 是保留值"
                )))
            }
        };
        if bps == 0 || bps > 32 {
            return Err(MediaError::Bitstream(format!("FLAC 位深不合法：{bps}")));
        }
        if block_size == 0 {
            return Err(MediaError::Bitstream("FLAC 块长为 0".into()));
        }

        // 逐声道解子帧
        let mut planes: Vec<Vec<i32>> = Vec::with_capacity(channels);
        for _ in 0..channels {
            planes.push(decode_subframe(&mut r, block_size, bps)?);
        }

        let crc16_at = {
            r.align_to_byte();
            r.byte_pos()
        };
        // 帧尾 CRC-16：只在"帧正好到这里结束"时校验（MP4 里可能有填充字节）
        if crc16_at + 2 == packet.len() {
            let got = u16::from_be_bytes([packet[crc16_at], packet[crc16_at + 1]]);
            let want = crc16(&packet[..crc16_at]);
            if got != want {
                return Err(MediaError::Bitstream(format!(
                    "FLAC 帧尾 CRC-16 不符（读到 {got:#x}，算出 {want:#x}）"
                )));
            }
        }

        // 声道去相关 → 交织
        let samples = join_channels(&mut planes, channel_code, block_size)?;

        // 时间戳：容器给了就用它（准）；没给就按已解样本数推算
        let pts = if pts_us > 0 {
            pts_us
        } else if sample_rate > 0 {
            (self.decoded_samples * 1_000_000 / sample_rate as u64) as i64
        } else {
            0
        };
        self.decoded_samples += block_size as u64;

        Ok(PcmChunk {
            sample_rate,
            channels: channels as u8,
            bits_per_sample: bps,
            pts_us: pts,
            samples,
        })
    }
}

impl AudioDecoder for FlacDecoder {
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
        (self.info.sample_rate, self.info.channels, self.info.bits_per_sample)
    }
}

// ── 帧头字段 ─────────────────────────────────────────────────────────

fn decode_block_size(r: &mut BitReader, code: u32) -> Result<usize> {
    Ok(match code {
        0 => return Err(MediaError::Bitstream("FLAC 块长码 0 是保留值".into())),
        1 => 192,
        2..=5 => 576 << (code - 2),
        6 => r.read(8)? as usize + 1,
        7 => r.read(16)? as usize + 1,
        8..=15 => 256 << (code - 8),
        _ => unreachable!(),
    })
}

fn decode_sample_rate(r: &mut BitReader, code: u32, from_stream_info: u32) -> Result<u32> {
    Ok(match code {
        0 => from_stream_info,
        1 => 88200,
        2 => 176400,
        3 => 192000,
        4 => 8000,
        5 => 16000,
        6 => 22050,
        7 => 24000,
        8 => 32000,
        9 => 44100,
        10 => 48000,
        11 => 96000,
        12 => r.read(8)? * 1000,
        13 => r.read(16)?,
        14 => r.read(16)? * 10,
        other => {
            return Err(MediaError::Bitstream(format!(
                "FLAC 采样率码 {other} 是保留值"
            )))
        }
    })
}

/// FLAC 的"UTF-8 风格"整数（1..7 字节，首字节的高位 1 的个数 = 字节数）。
fn read_utf8_number(r: &mut BitReader, max_bits: u32) -> Result<u64> {
    let first = r.read(8)?;
    let (mut value, extra) = if first & 0x80 == 0 {
        (first as u64, 0u32)
    } else if first & 0xE0 == 0xC0 {
        ((first & 0x1F) as u64, 1)
    } else if first & 0xF0 == 0xE0 {
        ((first & 0x0F) as u64, 2)
    } else if first & 0xF8 == 0xF0 {
        ((first & 0x07) as u64, 3)
    } else if first & 0xFC == 0xF8 {
        ((first & 0x03) as u64, 4)
    } else if first & 0xFE == 0xFC {
        ((first & 0x01) as u64, 5)
    } else if first == 0xFE {
        (0u64, 6)
    } else {
        return Err(MediaError::Bitstream(format!(
            "FLAC 的帧号/样本号前缀不合法（{first:#x}）"
        )));
    };
    for _ in 0..extra {
        let b = r.read(8)?;
        if b & 0xC0 != 0x80 {
            return Err(MediaError::Bitstream("FLAC 的帧号/样本号续字节不合法".into()));
        }
        value = (value << 6) | (b & 0x3F) as u64;
    }
    if (64 - value.leading_zeros()) > max_bits {
        return Err(MediaError::Bitstream(format!(
            "FLAC 的帧号/样本号超过 {max_bits} 位（数据损坏？）"
        )));
    }
    Ok(value)
}

// ── 子帧 ─────────────────────────────────────────────────────────────

fn decode_subframe(r: &mut BitReader, block_size: usize, bps: u8) -> Result<Vec<i32>> {
    if r.read(1)? != 0 {
        return Err(MediaError::Bitstream("FLAC 子帧头的填充位不是 0".into()));
    }
    let ty = r.read(6)?;
    let wasted = if r.read(1)? == 1 {
        r.read_unary()? + 1
    } else {
        0
    };
    let eff_bps = bps
        .checked_sub(wasted as u8)
        .ok_or_else(|| MediaError::Bitstream(format!("FLAC wasted bits 比位深还大（{wasted} > {bps}）")))?;
    if eff_bps == 0 {
        return Err(MediaError::Bitstream("FLAC 子帧的有效位深为 0".into()));
    }
    let mut out = match ty {
        // constant：整块同一个值
        0 => vec![r.read_signed(eff_bps)?; block_size],
        // verbatim：原样
        1 => {
            let mut v = Vec::with_capacity(block_size);
            for _ in 0..block_size {
                v.push(r.read_signed(eff_bps)?);
            }
            v
        }
        // fixed：0..4 阶定长预测
        8..=12 => decode_fixed(r, block_size, eff_bps, (ty - 8) as usize)?,
        // LPC：1..32 阶线性预测
        t if t >= 32 => decode_lpc(r, block_size, eff_bps, (t - 31) as usize)?,
        other => {
            return Err(MediaError::Bitstream(format!(
                "FLAC 子帧类型 {other} 是保留值（规范未定义）"
            )))
        }
    };
    if wasted > 0 {
        // wasted bits：低位对齐的样本要左移回去
        for s in out.iter_mut() {
            *s <<= wasted;
        }
    }
    Ok(out)
}

fn decode_fixed(r: &mut BitReader, n: usize, bps: u8, order: usize) -> Result<Vec<i32>> {
    let mut out: Vec<i32> = Vec::with_capacity(n);
    for _ in 0..order {
        out.push(r.read_signed(bps)?);
    }
    let residual = decode_residual(r, n, order)?;
    for i in order..n {
        let pred: i64 = match order {
            0 => 0,
            1 => out[i - 1] as i64,
            2 => 2 * out[i - 1] as i64 - out[i - 2] as i64,
            3 => 3 * out[i - 1] as i64 - 3 * out[i - 2] as i64 + out[i - 3] as i64,
            4 => {
                4 * out[i - 1] as i64 - 6 * out[i - 2] as i64 + 4 * out[i - 3] as i64
                    - out[i - 4] as i64
            }
            _ => unreachable!("fixed 只有 0..=4 阶"),
        };
        out.push((pred as i32).wrapping_add(residual[i - order]));
    }
    Ok(out)
}

fn decode_lpc(r: &mut BitReader, n: usize, bps: u8, order: usize) -> Result<Vec<i32>> {
    let mut out: Vec<i32> = Vec::with_capacity(n);
    for _ in 0..order {
        out.push(r.read_signed(bps)?);
    }
    let precision = r.read(4)? as u8 + 1;
    if precision == 16 {
        return Err(MediaError::Bitstream("FLAC LPC 精度 15（保留值）".into()));
    }
    // 移位是**有符号**的：负值表示左移（规范允许）
    let shift = r.read_signed(5)?;
    let mut coef = Vec::with_capacity(order);
    for _ in 0..order {
        coef.push(r.read_signed(precision)? as i64);
    }
    let residual = decode_residual(r, n, order)?;
    for i in order..n {
        let mut acc: i64 = 0;
        for (j, c) in coef.iter().enumerate() {
            acc += c * out[i - 1 - j] as i64;
        }
        let pred = if shift >= 0 { acc >> shift } else { acc << (-shift) };
        out.push((pred as i32).wrapping_add(residual[i - order]));
    }
    Ok(out)
}

/// Rice 分区残差（方法 0/1，含 escape）。
fn decode_residual(r: &mut BitReader, block_size: usize, predictor_order: usize) -> Result<Vec<i32>> {
    let method = r.read(2)?;
    let param_bits: u8 = match method {
        0 => 4,
        1 => 5,
        other => {
            return Err(MediaError::Bitstream(format!(
                "FLAC 残差编码方法 {other} 是保留值"
            )))
        }
    };
    let escape = (1u32 << param_bits) - 1;
    let partition_order = r.read(4)? as usize;
    let partitions = 1usize << partition_order;
    if !block_size.is_multiple_of(partitions) {
        return Err(MediaError::Bitstream(format!(
            "FLAC 块长 {block_size} 不是分区数 {partitions} 的整数倍"
        )));
    }
    let per_partition = block_size / partitions;
    if per_partition < predictor_order {
        return Err(MediaError::Bitstream(
            "FLAC 分区样本数比预测阶数还少（数据损坏）".into(),
        ));
    }
    let mut out: Vec<i32> = Vec::with_capacity(block_size.saturating_sub(predictor_order));
    for p in 0..partitions {
        let count = if p == 0 { per_partition - predictor_order } else { per_partition };
        let k = r.read(param_bits)?;
        if k == escape {
            // escape：这一段残差**原样**存（每样本 5 位给出的位宽）
            let raw_bits = r.read(5)? as u8;
            for _ in 0..count {
                out.push(r.read_signed(raw_bits)?);
            }
        } else {
            for _ in 0..count {
                let q = r.read_unary()? as u64;
                let rem = if k > 0 { r.read(k as u8)? as u64 } else { 0 };
                let u = (q << k) | rem;
                // 折叠有符号：最低位是符号
                out.push(((u >> 1) as i32) ^ -((u & 1) as i32));
            }
        }
    }
    Ok(out)
}

/// 声道去相关 → 交织采样。
fn join_channels(planes: &mut [Vec<i32>], channel_code: u32, block_size: usize) -> Result<Vec<i32>> {
    let mut samples = vec![0i32; block_size * planes.len()];
    match channel_code {
        0..=7 => {
            for (c, plane) in planes.iter().enumerate() {
                for (i, s) in plane.iter().enumerate() {
                    samples[i * planes.len() + c] = *s;
                }
            }
        }
        8 => {
            // left/side：ch0 = 左，ch1 = 差
            for i in 0..block_size {
                let l = planes[0][i];
                let side = planes[1][i];
                samples[i * 2] = l;
                samples[i * 2 + 1] = l.wrapping_sub(side);
            }
        }
        9 => {
            // right/side：ch0 = 差，ch1 = 右
            for i in 0..block_size {
                let side = planes[0][i];
                let rr = planes[1][i];
                samples[i * 2] = rr.wrapping_add(side);
                samples[i * 2 + 1] = rr;
            }
        }
        10 => {
            // mid/side：ch0 = 中，ch1 = 差
            for i in 0..block_size {
                let mid = planes[0][i];
                let side = planes[1][i];
                // 中声道多存了一位：先把最低位从差声道借回来
                let mid2 = (mid << 1) | (side & 1);
                samples[i * 2] = (mid2.wrapping_add(side)) >> 1;
                samples[i * 2 + 1] = (mid2.wrapping_sub(side)) >> 1;
            }
        }
        other => {
            return Err(MediaError::Bitstream(format!(
                "FLAC 声道分配码 {other} 不支持"
            )))
        }
    }
    Ok(samples)
}

// ── STREAMINFO ───────────────────────────────────────────────────────

fn parse_streaminfo_any(data: &[u8]) -> Result<StreamInfo> {
    if data.len() >= 4 && &data[..4] == b"fLaC" {
        // 裸 FLAC：元数据块序列，第一个必须是 STREAMINFO(type 0)
        let mut pos = 4usize;
        while pos + 4 <= data.len() {
            let header = data[pos];
            let last = header & 0x80 != 0;
            let ty = header & 0x7F;
            let len = ((data[pos + 1] as usize) << 16)
                | ((data[pos + 2] as usize) << 8)
                | data[pos + 3] as usize;
            let body = data.get(pos + 4..pos + 4 + len).ok_or_else(|| {
                MediaError::Bitstream("FLAC 元数据块长度越界".to_string())
            })?;
            if ty == 0 {
                return parse_streaminfo(body);
            }
            if last {
                break;
            }
            pos += 4 + len;
        }
        return Err(MediaError::Bitstream("FLAC 文件头里没有 STREAMINFO".into()));
    }
    // dfLa 载荷（MP4）：version+flags(4) + 元数据块头(4) + STREAMINFO(34)
    if data.len() >= 8 + STREAMINFO_LEN && data[0..4] == [0, 0, 0, 0] && data[4] & 0x7F == 0 {
        let len = ((data[5] as usize) << 16) | ((data[6] as usize) << 8) | data[7] as usize;
        let body = data.get(8..8 + len).ok_or_else(|| {
            MediaError::Bitstream("dfLa 里的 STREAMINFO 长度越界".to_string())
        })?;
        return parse_streaminfo(body);
    }
    // 裸 STREAMINFO
    parse_streaminfo(data)
}

fn parse_streaminfo(b: &[u8]) -> Result<StreamInfo> {
    if b.len() < STREAMINFO_LEN {
        return Err(MediaError::Bitstream(format!(
            "STREAMINFO 只有 {} 字节（应当 {STREAMINFO_LEN}）",
            b.len()
        )));
    }
    let min_block_size = u16::from_be_bytes([b[0], b[1]]);
    let max_block_size = u16::from_be_bytes([b[2], b[3]]);
    let sample_rate = ((b[10] as u32) << 12) | ((b[11] as u32) << 4) | ((b[12] as u32) >> 4);
    let channels = ((b[12] >> 1) & 0x07) + 1;
    let bits_per_sample = (((b[12] & 0x01) << 4) | (b[13] >> 4)) + 1;
    let total_samples = (((b[13] & 0x0F) as u64) << 32)
        | ((b[14] as u64) << 24)
        | ((b[15] as u64) << 16)
        | ((b[16] as u64) << 8)
        | (b[17] as u64);
    Ok(StreamInfo {
        min_block_size,
        max_block_size,
        sample_rate,
        channels,
        bits_per_sample,
        total_samples,
    })
}

// ── CRC ──────────────────────────────────────────────────────────────
//
// FLAC 用两个标准 CRC：帧头 CRC-8（x^8+x^2+x+1 = 0x07）、帧尾 CRC-16
// （x^16+x^15+x^2+1 = 0x8005）。**不是**常见的 CRC-32/CCITT 那两个多项式。

fn crc8(data: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &b in data {
        crc ^= b;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 { (crc << 1) ^ 0x07 } else { crc << 1 };
        }
    }
    crc
}

fn crc16(data: &[u8]) -> u16 {
    let mut crc = 0u16;
    for &b in data {
        crc ^= (b as u16) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 { (crc << 1) ^ 0x8005 } else { crc << 1 };
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 与被测实现**互相独立**的逐位参考（同多项式、不同写法），用来交叉验证。
    fn ref_crc8(data: &[u8]) -> u8 {
        let mut reg = 0u8;
        for byte in data {
            for i in (0..8).rev() {
                let bit = (byte >> i) & 1;
                let top = (reg >> 7) & 1;
                reg <<= 1;
                if top ^ bit == 1 {
                    reg ^= 0x07;
                }
            }
        }
        reg
    }

    fn ref_crc16(data: &[u8]) -> u16 {
        let mut reg = 0u16;
        for byte in data {
            for i in (0..8).rev() {
                let bit = ((byte >> i) & 1) as u16;
                let top = (reg >> 15) & 1;
                reg <<= 1;
                if top ^ bit == 1 {
                    reg ^= 0x8005;
                }
            }
        }
        reg
    }

    #[test]
    fn crc_matches_bitwise_reference() {
        let cases: Vec<Vec<u8>> = vec![
            vec![],
            vec![0x00],
            vec![0xFF],
            vec![0x12, 0x34, 0x56, 0x78],
            (0u8..64).collect(),
        ];
        for c in cases {
            assert_eq!(crc8(&c), ref_crc8(&c), "CRC-8 不一致：{c:?}");
            assert_eq!(crc16(&c), ref_crc16(&c), "CRC-16 不一致：{c:?}");
        }
    }

    #[test]
    fn parses_streaminfo_from_raw_and_dfla() {
        // 手造 STREAMINFO：44100 Hz / 2 声道 / 16 位 / 总样本 123456
        let mut si = [0u8; STREAMINFO_LEN];
        si[0..2].copy_from_slice(&4096u16.to_be_bytes());
        si[2..4].copy_from_slice(&4096u16.to_be_bytes());
        let rate = 44100u32;
        si[10] = (rate >> 12) as u8;
        si[11] = (rate >> 4) as u8;
        // si[12] 里塞了三样：采样率低 4 位、声道数-1（3 位）、位深最高位（1 位）
        si[12] = ((rate & 0x0F) as u8) << 4 | (1 << 1);
        // 位深 16 → 低 4 位写 15；总样本数 123456 只有 32 位，高 4 位为 0
        let total_samples: u64 = 123456;
        si[13] = (15 << 4) | ((total_samples >> 32) as u8 & 0x0F);
        let total_low = total_samples as u32;
        si[14..18].copy_from_slice(&total_low.to_be_bytes());

        let raw = parse_streaminfo_any(&si).unwrap();
        assert_eq!(raw.sample_rate, 44100);
        assert_eq!(raw.channels, 2);
        assert_eq!(raw.bits_per_sample, 16);
        assert_eq!(raw.total_samples, 123456);
        assert_eq!(raw.min_block_size, 4096);

        // dfLa 载荷形态（version+flags + 块头 + STREAMINFO）
        let mut dfla = vec![0u8, 0, 0, 0, 0x80, 0, 0, STREAMINFO_LEN as u8];
        dfla.extend_from_slice(&si);
        let from_dfla = parse_streaminfo_any(&dfla).unwrap();
        assert_eq!(from_dfla, raw, "两种形态要解出同样的流信息");

        // 裸 FLAC 文件头形态
        let mut file = b"fLaC".to_vec();
        file.push(0x80); // last + type 0
        file.extend_from_slice(&[0, 0, STREAMINFO_LEN as u8]);
        file.extend_from_slice(&si);
        assert_eq!(parse_streaminfo_any(&file).unwrap(), raw);

        // 坏数据
        assert!(parse_streaminfo_any(&[]).is_err());
        assert!(parse_streaminfo_any(&si[..10]).is_err());
    }

    #[test]
    fn utf8_style_frame_numbers() {
        // 0x00 → 0；0xC2 0x80 → 0x80；0xE0 0xA0 0x80 → 0x800
        let cases: Vec<(Vec<u8>, u64)> = vec![
            (vec![0x00], 0),
            (vec![0x7F], 0x7F),
            (vec![0xC2, 0x80], 0x80),
            (vec![0xE0, 0xA0, 0x80], 0x800),
            (vec![0xF0, 0x90, 0x80, 0x80], 0x10000),
        ];
        for (bytes, want) in cases {
            let mut r = BitReader::new(&bytes);
            assert_eq!(read_utf8_number(&mut r, 36).unwrap(), want, "{bytes:?}");
        }
        // 续字节前缀不对要报错
        let mut r = BitReader::new(&[0xC2, 0x40]);
        assert!(read_utf8_number(&mut r, 36).is_err());
    }

    /// 把 `"0101"` 这样的位串按 MSB 优先打包成字节（尾字节左对齐补 0）。
    ///
    /// 手写位串比手算十六进制更不容易错：残差编码的"方法/分区阶/参数/商/余数"是
    /// 一串变长字段，写成十六进制之后再想核对就很难了。
    fn bits_to_bytes(bitstring: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut cur = 0u8;
        let mut n = 0;
        for ch in bitstring.chars() {
            cur = (cur << 1) | u8::from(ch == '1');
            n += 1;
            if n == 8 {
                bytes.push(cur);
                cur = 0;
                n = 0;
            }
        }
        if n > 0 {
            bytes.push(cur << (8 - n));
        }
        bytes
    }

    #[test]
    fn rice_residual_is_folded_signed() {
        // 方法 0（4 位参数）、分区阶 0、参数 k=2：写两个残差 +1 与 -1
        //   u=2 → +1（0b10）；u=1 → -1（0b01）
        let bitstring = "00"             // 方法 = 0（4 位参数）
            .to_string()
            + "0000"                     // 分区阶 = 0（1 个分区）
            + "0010"                     // k = 2
            + "1" + "10"                 // 残差 1：一元 0 个 0 + 2 位余数 10 → u=2 → +1
            + "1" + "01";                // 残差 2：一元 0 个 0 + 2 位余数 01 → u=1 → -1
        let bytes = bits_to_bytes(&bitstring);
        let mut r = BitReader::new(&bytes);
        let out = decode_residual(&mut r, 2, 0).unwrap();
        assert_eq!(out, vec![1, -1], "折叠有符号：u=2→+1、u=1→-1");
    }

    #[test]
    fn rice_accepts_long_quotients_near_buffer_end() {
        // 回归（96 kHz 尾帧踩过的坑）：商可以很大 —— 那个残差的商是 123，
        // 需要 124 位的一元码，而**读到它时缓冲区里只剩 145 位**。
        // 位读取器曾经按"剩余位数"给一元码设上限，把这种合法长码判成数据损坏。
        // u = (123 << 2) | 3 = 495（奇数）→ v = (495 >> 1) ^ -1 = -248
        let bitstring = "00".to_string()
            + "0000"                     // 分区阶 = 0
            + "0010"                     // k = 2
            + &"0".repeat(123) + "1"     // 商 = 123
            + "11";                      // 余数 = 3
        let bytes = bits_to_bytes(&bitstring);
        let mut r = BitReader::new(&bytes);
        let out = decode_residual(&mut r, 1, 0).unwrap();
        assert_eq!(out, vec![-248], "长商必须按折叠有符号解出来，不能报损坏");
    }

    #[test]
    fn mid_side_stereo_reconstruction() {
        // 左=10、右=4 → mid=(10+4)>>1=7、side=10-4=6；解码侧要能原样还原
        let mut planes = vec![vec![7i32], vec![6i32]];
        let out = join_channels(&mut planes, 10, 1).unwrap();
        assert_eq!(out, vec![10, 4], "mid/side 要还原成左右声道");
        // left/side
        let mut planes = vec![vec![10i32], vec![6i32]];
        assert_eq!(join_channels(&mut planes, 8, 1).unwrap(), vec![10, 4]);
        // right/side
        let mut planes = vec![vec![6i32], vec![4i32]];
        assert_eq!(join_channels(&mut planes, 9, 1).unwrap(), vec![10, 4]);
        // 独立双声道
        let mut planes = vec![vec![3i32], vec![9i32]];
        assert_eq!(join_channels(&mut planes, 1, 1).unwrap(), vec![3, 9]);
    }

    #[test]
    fn block_size_and_sample_rate_codes() {
        let mut r = BitReader::new(&[0xFF, 0xFF]);
        assert_eq!(decode_block_size(&mut r, 1).unwrap(), 192);
        assert_eq!(decode_block_size(&mut r, 2).unwrap(), 576);
        assert_eq!(decode_block_size(&mut r, 5).unwrap(), 4608);
        assert_eq!(decode_block_size(&mut r, 8).unwrap(), 256);
        assert_eq!(decode_block_size(&mut r, 15).unwrap(), 32768);
        assert!(decode_block_size(&mut r, 0).is_err(), "码 0 是保留值");
        assert_eq!(decode_sample_rate(&mut r, 9, 0).unwrap(), 44100);
        assert_eq!(decode_sample_rate(&mut r, 0, 96000).unwrap(), 96000);
        assert!(decode_sample_rate(&mut r, 15, 0).is_err(), "码 15 是保留值");
    }
}
