//! NAL 单元的分类，以及**按 id 索引**的参数集注册表。
//!
//! ## 为什么软解要自己拿一份参数集，而不是直接用 [`crate::h264::Config`]
//!
//! `Config` 是给硬解后端用的：它只认 `avcC` 里的**第一组** SPS/PPS，
//! 因为硬解后端把参数集一次性喂给系统解码器就完事。软解不一样 ——
//! 码流里**随时可能插入新的 SPS/PPS**（每个 id 一份，切片头按
//! `pic_parameter_set_id` 去取），而且新 SPS 换了尺寸时**参考帧必须整体作废**。
//! 所以这里按 `id` 存两张表，`put_*` 返回 id 让调用方去比对"是不是变了"。
//!
//! ## 支持口径
//!
//! 与 [`crate::h264`] 一致：8 位 4:2:0 逐行帧、只做滑动窗口参考帧标记。
//! 参数集解析直接复用 [`crate::h264::Sps::parse`] / [`crate::h264::Pps::parse`]
//! （它们是照 ITU-T H.264 §7.3.2 逐字段写的，三个硬解后端也在用同一份），
//! **不在这里另写一份** —— 同一份语法有两份实现，迟早会分叉。

use me_core::{MediaError, Result};

use crate::h264::{self, Pps, Sps};

/// NAL 类型（§7.4.1）—— 软解会遇到的都在这里，其余按"忽略"处理。
pub const NAL_SLICE: u8 = 1;
pub const NAL_SLICE_DPA: u8 = 2;
pub const NAL_SLICE_DPB: u8 = 3;
pub const NAL_SLICE_DPC: u8 = 4;
pub const NAL_SLICE_IDR: u8 = 5;
pub const NAL_SEI: u8 = 6;
pub const NAL_SPS: u8 = 7;
pub const NAL_PPS: u8 = 8;
pub const NAL_AUD: u8 = 9;
pub const NAL_END_SEQUENCE: u8 = 10;
pub const NAL_END_STREAM: u8 = 11;
pub const NAL_FILLER: u8 = 12;
pub const NAL_SPS_EXT: u8 = 13;
/// 辅助切片（SVC/MVC 的扩展切片是 14/20/21）—— 引擎不支持，遇到要**明确报出来**。
pub const NAL_PREFIX: u8 = 14;
pub const NAL_SUBSET_SPS: u8 = 15;
pub const NAL_SLICE_EXT: u8 = 20;

/// NAL 头的 `nal_type`（低 5 位）。
#[inline]
pub fn nal_type(nal: &[u8]) -> u8 {
    nal.first().map(|b| b & 0x1F).unwrap_or(0)
}

/// NAL 头的 `nal_ref_idc`（第 5~6 位）。0 = 非参考。
#[inline]
pub fn nal_ref_idc(nal: &[u8]) -> u8 {
    nal.first().map(|b| (b >> 5) & 0x03).unwrap_or(0)
}

/// 只读出 PPS 的 `(pps_id, sps_id)` 两个字段（§7.3.2.2 开头就是它们的 `ue(v)`）。
///
/// 为什么单独写这个小函数：解析完整的 PPS **需要它指向的 SPS**（表张数依赖
/// `chroma_format_idc`），而"它指向哪个 SPS"正是这里要读的东西 —— 用
/// `Pps::parse` 配一个缺省 SPS 去试探，遇到 FMO 之类会在读到 id **之后**报错，
/// 于是"id 读到了没有"就得靠调用方猜。两步走更干脆：先拿 id，再拿真 SPS 解析。
fn read_pps_ids(nal: &[u8]) -> Result<(u8, u8)> {
    if nal.len() < 2 {
        return Err(MediaError::Bitstream("PPS 太短".into()));
    }
    let rbsp = me_core::bits::remove_emulation_prevention(&nal[1..]);
    let mut r = me_core::bits::BitReader::new(&rbsp);
    let pps_id = u8::try_from(r.read_ue()?)
        .map_err(|_| MediaError::Bitstream("PPS 的 id 越界".into()))?;
    let sps_id = u8::try_from(r.read_ue()?)
        .map_err(|_| MediaError::Bitstream("PPS 的 sps_id 越界".into()))?;
    Ok((pps_id, sps_id))
}

/// 切片 / 参数集 NAL → **"NAL 头字节 + 去竞争字节后的负载"**。
///
/// 引擎里所有位读（切片头解析、CAVLC 熵层、`more_rbsp_data()`）都吃这一条缓冲
/// —— 三处各拼一遍迟早会分叉（比如一处忘了去竞争字节，读出来就完全是另一条流）。
pub fn rbsp_of_nal(nal: &[u8]) -> Vec<u8> {
    let mut raw = Vec::with_capacity(nal.len());
    if let Some(&h) = nal.first() {
        raw.push(h);
    }
    if nal.len() > 1 {
        raw.extend_from_slice(&me_core::bits::remove_emulation_prevention(&nal[1..]));
    }
    raw
}

/// 只读出切片头的 `pic_parameter_set_id`（§7.3.3 的前三个字段）。
///
/// 切片头的**完整**解析需要它所属的 SPS/PPS（字段位宽依赖它们），而"它用哪个
/// PPS"正是这里要读的东西 —— 先读 id、查表、再完整解析，比"配一份缺省参数集
/// 试解一遍"干净得多。
pub fn read_slice_pps_id(nal: &[u8]) -> Result<u8> {
    if nal.len() < 2 {
        return Err(MediaError::Bitstream("切片 NAL 太短".into()));
    }
    let raw = rbsp_of_nal(nal);
    let mut r = me_core::bits::BitReader::new(&raw);
    let _header = r.read_u32(8)?;
    let _first_mb_in_slice = r.read_ue()?;
    let _slice_type = r.read_ue()?;
    u8::try_from(r.read_ue()?).map_err(|_| MediaError::Bitstream("切片头的 pps_id 越界".into()))
}

/// 参数集注册表。
pub struct ParamSets {
    sps: Vec<Option<Sps>>,
    pps: Vec<Option<Pps>>,
}

impl Default for ParamSets {
    fn default() -> Self {
        Self::new()
    }
}

impl ParamSets {
    pub fn new() -> Self {
        Self {
            sps: vec![None; 32],
            pps: vec![None; 256],
        }
    }

    /// 收一个 SPS NAL（含头字节），返回它的 id。
    pub fn put_sps(&mut self, nal: &[u8]) -> Result<u8> {
        let sps = Sps::parse(nal)?;
        sps.check_supported()?;
        let id = sps.id as usize;
        if id >= self.sps.len() {
            return Err(MediaError::Bitstream(format!("SPS id {id} 越界（上限 31）")));
        }
        self.sps[id] = Some(sps);
        Ok(id as u8)
    }

    /// 收一个 PPS NAL，返回它的 id（需要它指向的 SPS 已经在表里）。
    pub fn put_pps(&mut self, nal: &[u8]) -> Result<u8> {
        let (_pps_id, sps_id) = read_pps_ids(nal)?;
        let sps = self
            .sps
            .get(sps_id as usize)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| {
                MediaError::Bitstream(format!(
                    "PPS 指向的 SPS {sps_id} 还没出现过（码流里的 PPS 在 SPS 之前？）"
                ))
            })?;
        let pps = Pps::parse(nal, sps)?;
        pps.check_supported()?;
        if pps.sps_id != sps.id {
            return Err(MediaError::Bitstream(format!(
                "PPS 指向 SPS {}，拿到的是 SPS {}",
                pps.sps_id, sps.id
            )));
        }
        let id = pps.id as usize;
        self.pps[id] = Some(pps);
        Ok(id as u8)
    }

    pub fn sps(&self, id: u8) -> Result<&Sps> {
        self.sps
            .get(id as usize)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| MediaError::Bitstream(format!("切片引用了没出现过的 SPS {id}")))
    }

    pub fn pps(&self, id: u8) -> Result<&Pps> {
        self.pps
            .get(id as usize)
            .and_then(|p| p.as_ref())
            .ok_or_else(|| MediaError::Bitstream(format!("切片引用了没出现过的 PPS {id}")))
    }

    /// 从 `avcC` 里预装第一组参数集（流内可能还带别的，`put_*` 会覆盖同 id 的）。
    pub fn load_from_avcc(&mut self, avcc: &[u8]) -> Result<()> {
        let (sps_list, pps_list) = h264::parse_avcc(avcc)?;
        for nal in &sps_list {
            self.put_sps(nal)?;
        }
        for nal in &pps_list {
            self.put_pps(nal)?;
        }
        Ok(())
    }

    /// 把 `avcC` / TS 里那种**多 NAL 参数集负载**（`extradata` 的 Annex-B 形态）
    /// 也收进来 —— 直接打包成完整 NAL 是不行的（没有长度前缀），
    /// 而已知它是 Annex-B 时按起始码切就行。
    pub fn load_annexb_params(&mut self, data: &[u8]) -> Result<()> {
        for nal in split_annexb(data) {
            match nal_type(nal) {
                NAL_SPS => {
                    self.put_sps(nal)?;
                }
                NAL_PPS => {
                    self.put_pps(nal)?;
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// 按 Annex-B 起始码切 NAL（`00 00 01` 或 `00 00 00 01`）。
///
/// 引擎内部的采样统一是 AVCC，**这个函数只用于 `extradata`**：TS 解出来的
/// extradata 是 Annex-B 参数集（见 DEVELOPMENT.md 硬约定第 5 条），
/// 而 `avcC` 那条路走 [`ParamSets::load_from_avcc`]。两条路都留着，
/// 是因为"来源是哪种形态"由容器如实报告，不能靠猜。
pub fn split_annexb(data: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0usize;
    while i + 2 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::with_capacity(starts.len());
    for (k, &s) in starts.iter().enumerate() {
        let e = starts.get(k + 1).map(|&n| n - 3).unwrap_or(data.len());
        if e > s {
            out.push(&data[s..e]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h264::tests_support::{build_pps, Bits};

    fn sps_nal(profile: u8, width_mbs: u32, height_map_units: u32) -> Vec<u8> {
        let mut w = Bits::default();
        w.f(0x67, 8);
        w.f(profile as u32, 8);
        w.f(0, 8);
        w.f(30, 8);
        w.ue(0);
        if h264::is_high_profile(profile) {
            w.ue(1);
            w.ue(0);
            w.ue(0);
            w.bit(false);
            w.bit(false);
        }
        w.ue(0);
        w.ue(0);
        w.ue(0);
        w.ue(1); // max_num_ref_frames
        w.bit(false);
        w.ue(width_mbs - 1);
        w.ue(height_map_units - 1);
        w.bit(true);
        w.bit(true);
        w.bit(false);
        w.bit(false);
        w.f(0b1000_0000, 8);
        w.finish()
    }

    fn pps_nal() -> Vec<u8> {
        let mut w = Bits::default();
        w.f(0x68, 8);
        build_pps(&mut w);
        w.finish()
    }

    #[test]
    fn classifies_nal_headers() {
        assert_eq!(nal_type(&[0x67]), NAL_SPS);
        assert_eq!(nal_type(&[0x68]), NAL_PPS);
        assert_eq!(nal_type(&[0x65]), NAL_SLICE_IDR);
        assert_eq!(nal_type(&[0x41]), NAL_SLICE);
        assert_eq!(nal_ref_idc(&[0x65]), 3);
        assert_eq!(nal_ref_idc(&[0x01]), 0);
        assert_eq!(nal_type(&[]), 0, "空 NAL 不 panic");
    }

    #[test]
    fn stores_and_reads_parameter_sets_by_id() {
        let avcc = {
            let sps = sps_nal(77, 40, 23);
            let pps = pps_nal();
            let mut v = vec![1u8, 77, 0x00, 30, 0xFF, 0xE1];
            v.extend_from_slice(&(sps.len() as u16).to_be_bytes());
            v.extend_from_slice(&sps);
            v.push(0x01);
            v.extend_from_slice(&(pps.len() as u16).to_be_bytes());
            v.extend_from_slice(&pps);
            v
        };
        let mut ps = ParamSets::new();
        ps.load_from_avcc(&avcc).unwrap();
        assert_eq!(ps.sps(0).unwrap().coded_width(), 640);
        assert_eq!(ps.pps(0).unwrap().id, 0);
        assert!(ps.sps(7).is_err(), "没出现过的 id 要报错而不是给个默认值");
    }

    /// 同一个 id 再出现就是**更新**（码流中途换尺寸是合法行为）。
    #[test]
    fn a_second_sps_with_the_same_id_replaces_it() {
        let small = sps_nal(77, 20, 12);
        let big = sps_nal(77, 40, 23);
        let mut ps = ParamSets::new();
        assert_eq!(ps.put_sps(&small).unwrap(), 0);
        assert_eq!(ps.sps(0).unwrap().coded_width(), 320);
        ps.put_sps(&big).unwrap();
        assert_eq!(ps.sps(0).unwrap().coded_width(), 640);
    }

    #[test]
    fn pps_before_its_sps_is_rejected() {
        let mut ps = ParamSets::new();
        let e = ps.put_pps(&pps_nal()).unwrap_err();
        assert!(format!("{e}").contains("SPS 0"));
    }

    #[test]
    fn splits_annexb_parameter_sets() {
        let sps = sps_nal(77, 10, 6);
        let pps = pps_nal();
        let mut data = vec![0u8, 0, 0, 1];
        data.extend_from_slice(&sps);
        data.extend_from_slice(&[0, 0, 1]);
        data.extend_from_slice(&pps);
        let nals = split_annexb(&data);
        assert_eq!(nals.len(), 2);
        assert_eq!(nal_type(nals[0]), NAL_SPS);
        assert_eq!(nal_type(nals[1]), NAL_PPS);

        let mut ps = ParamSets::new();
        ps.load_annexb_params(&data).unwrap();
        assert_eq!(ps.sps(0).unwrap().coded_height(), 96);
    }
}
