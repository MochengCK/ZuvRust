//! MPEG-TS 封装：把带完整时序的采样流写成一条可播的传输流。
//!
//! ## 必须做对的四件事（任何一件错了都会"能生成但播不了"）
//!
//! 1. **PAT/PMT 要周期性重发**。播放器可能从流中间接入（拖进度条、切台、
//!    直播追上来），它靠 PAT/PMT 才知道"这个节目里有哪几条轨、分别是什么编码"。
//!    只在开头写一次，中间接入的播放器就只能看到一个黑屏。
//! 2. **PCR 要周期性写**。PCR 是 TS 里唯一自带墙钟含义的字段，解码器靠它
//!    建立时间基准、做音视频同步；缺了它不少播放器会直接不播。
//! 3. **AAC 在 TS 里必须带 ADTS 头**。fMP4 里的 AAC 是裸帧，直接搬进 TS
//!    会得到一个"没有任何播放器认得"的流 —— 转换由这里负责。
//! 4. **视频必须是 Annex-B**，并且**每个关键帧前面要重发参数集**
//!    （SPS/PPS）。从中间切入的播放器如果拿不到参数集，解不出画面。
//!
//! ## 包内填充用 adaptation field，不用垃圾字节
//!
//! 载荷不足 184 字节时，规范的做法是用 adaptation field 填充。直接拿 0xFF
//! 把包填满（很多简易实现这么干）会让解复用器把填充当数据 ——
//! 音频尤其明显：一堆 0xFF 会被当成 ADTS 同步字，整条音轨就废了。

use me_core::{ByteWriter, CodecId, MediaError, MediaKind, Packet, Result, Timebase, Track};

use crate::adts::{adts_header, is_adts};
use crate::convert;
use crate::sink::Sink;

/// 裸 TS 包长。
pub const TS_PACKET_LEN: usize = 188;
/// 同步字。
const SYNC: u8 = 0x47;
/// PAT 固定 PID。
const PAT_PID: u16 = 0x0000;
/// 我们用的 PMT PID（不占 0，也不与 ES 冲突）。
const PMT_PID: u16 = 0x1000;
/// 第一个 ES 用的 PID。
const FIRST_ES_PID: u16 = 0x0101;
/// 节目号。
const PROGRAM_NUMBER: u16 = 1;
/// 每隔多少字节重发一次 PAT/PMT（约 100ms 一次，规范上限就是 100ms）。
const SECTION_INTERVAL_BYTES: usize = 8 * 1024;
/// 每隔多少媒体时间写一次 PCR（40ms 是广播里的常见值）。
const PCR_INTERVAL_TICKS: i64 = 90_000 * 40 / 1000;
/// 一个 TS 包的载荷容量。
const PAYLOAD_CAPACITY: usize = TS_PACKET_LEN - 4;

/// 一条轨道的封装状态。
#[derive(Debug, Clone)]
struct EsTrack {
    track: Track,
    pid: u16,
    stream_id: u8,
    /// 参数集（Annex-B 形态），视频每个关键帧前重发。
    params: Vec<u8>,
    /// 连续性计数器。
    cc: u8,
    /// 上一次写 PCR 的时间（90kHz）。
    last_pcr: Option<i64>,
    /// 上一次重发 PAT/PMT 时已经写出的字节数。
    last_section_bytes: usize,
}

/// MPEG-TS 封装器。
pub struct TsMuxer {
    tracks: Vec<EsTrack>,
    /// 产物落点（内存或直写目标）。
    ///
    /// TS 是**纯流式**容器：每个 188 字节的包自包含，PAT/PMT 周期重发而不是
    /// 回头改，所以完全没有"攒齐再写"的理由 —— 之前的 `Vec<u8>` 只是实现方便，
    /// 代价是内存占用与产物大小成正比。
    sink: Sink,
    /// **时间戳基准（微秒）**：第一条采样（不一定是第一轨的）的时间。
    ///
    /// 两个理由必须按微秒算：
    /// 1. 容器里没有"负时间戳"这种东西（TS 的 PTS 是 33 位无符号），而解析层
    ///    如实报告的 dts 可能是负的（MP4 的编辑列表会平移出负值）；
    /// 2. **各轨时基不同**（fMP4 的视频常见 15360，音频是采样率），
    ///    拿一条轨的刻度去减另一条轨的刻度是错的。
    /// 归零放在 muxer：整条流一起平移，采样间距与音视频相对偏移都保得住。
    base_us: Option<i64>,
    /// 是否已经写过 PAT/PMT。
    sections_written: bool,
    /// PAT / PMT 各自的连续性计数器（重发表时也要递增，不能每次从 0 开始）。
    cc_pat: u8,
    cc_pmt: u8,
    /// 已写出的采样数（诊断用）。
    packets_written: u64,
}

impl Default for TsMuxer {
    fn default() -> Self {
        Self::new()
    }
}

impl TsMuxer {
    /// 内存落点（测试与小产物）。
    pub fn new() -> Self {
        Self::with_sink(Sink::mem())
    }

    /// 直写目标：产物**不驻留内存**（大文件合并走这条）。
    pub fn with_writer(w: Box<dyn std::io::Write + Send>) -> Self {
        Self::with_sink(Sink::writer(w))
    }

    fn with_sink(sink: Sink) -> Self {
        Self {
            tracks: Vec::new(),
            sink,
            base_us: None,
            sections_written: false,
            cc_pat: 0,
            cc_pmt: 0,
            packets_written: 0,
        }
    }

    /// 声明轨道（决定 PMT 的内容）。必须在写采样之前调用一次。
    pub fn set_tracks(&mut self, tracks: &[Track]) -> Result<()> {
        if tracks.is_empty() {
            return Err(MediaError::Container("TS 至少需要一条轨道".into()));
        }
        self.tracks.clear();
        for (i, t) in tracks.iter().enumerate() {
            let pid = FIRST_ES_PID + i as u16;
            // stream_type 决定播放器找哪个解码器：写错就是"有数据但解不出来"
            let _stream_type = t.codec.ts_stream_type();
            let stream_id = match t.kind {
                MediaKind::Video => 0xE0,
                MediaKind::Audio => 0xC0,
                MediaKind::Subtitle => 0xBD,
                MediaKind::Other => 0xBD,
            };
            let params = if matches!(t.kind, MediaKind::Video) && !t.extradata.is_empty() {
                convert::extradata_to_annexb(&t.codec, &t.extradata).unwrap_or_default()
            } else {
                Vec::new()
            };
            self.tracks.push(EsTrack {
                track: t.clone(),
                pid,
                stream_id,
                params,
                cc: 0,
                last_pcr: None,
                last_section_bytes: 0,
            });
        }
        // 先把表写出去，这样即使一条采样都没有，产物也是一个合法的 TS
        self.write_sections()?;
        Ok(())
    }

    /// 写一条采样。
    pub fn write_packet(&mut self, p: &Packet) -> Result<()> {
        if self.tracks.is_empty() {
            return Err(MediaError::Container("写采样之前必须先 set_tracks".into()));
        }
        // 起点归零 + **换算到 90kHz**（TS 的时间戳是固定 90kHz 的，而输入轨的
        // 时基可能是 15360 / 采样率……直接当 90kHz 编码会让时间戳错好几倍，
        // 实测一条 15360 时基的视频在 TS 里会快 5.9 倍）
        let tb = self.tracks.iter().find(|t| t.track.id == p.track_id).map(|t| t.track.timebase);
        let Some(tb) = tb else {
            return Err(MediaError::Container(format!("轨道 {} 不在 PMT 里", p.track_id)));
        };
        let base_us = *self.base_us.get_or_insert(tb.ticks_to_us(p.dts));
        let pts = Timebase::MPEG_TS.us_to_ticks(tb.ticks_to_us(p.pts) - base_us);
        let dts = Timebase::MPEG_TS.us_to_ticks(tb.ticks_to_us(p.dts) - base_us);
        let idx = self
            .tracks
            .iter()
            .position(|t| t.track.id == p.track_id)
            .ok_or_else(|| MediaError::Container(format!("轨道 {} 不在 PMT 里", p.track_id)))?;

        // PCR：挂在视频轨上（没有视频就挂第一条音频轨）
        let pcr_owner = self
            .tracks
            .iter()
            .position(|t| matches!(t.track.kind, MediaKind::Video))
            .unwrap_or(0);
        if idx == pcr_owner {
            let need = match self.tracks[idx].last_pcr {
                Some(last) => (dts - last).abs() >= PCR_INTERVAL_TICKS,
                None => true,
            };
            if need {
                self.write_pcr(dts)?;
                self.tracks[idx].last_pcr = Some(dts);
            }
        }

        // PAT/PMT 周期性重发
        if self.sink.bytes_written().saturating_sub(self.tracks[0].last_section_bytes as u64)
            >= SECTION_INTERVAL_BYTES as u64
        {
            self.write_sections()?;
        }

        let es = self.payload_of(idx, p)?;
        let pes = build_pes(self.tracks[idx].stream_id, pts, dts, &es, false);
        let pid = self.tracks[idx].pid;
        let mut cc = self.tracks[idx].cc;
        emit_payload(&mut self.sink, pid, &mut cc, &pes)?;
        self.tracks[idx].cc = cc;
        self.packets_written += 1;
        Ok(())
    }

    /// 收尾：返回整条流（流式模式下为空 —— 字节已经写进目标了）。
    ///
    /// 会在末尾补一份 PAT/PMT，让"从尾部接入"的解析器也能立刻认出轨道。
    pub fn finish(&mut self) -> Result<Vec<u8>> {
        self.write_sections()?;
        self.sink.flush()?;
        Ok(self.sink.take_bytes())
    }

    /// 已经写了多少字节（`mux` 的进度上报用）。
    pub fn bytes_written(&self) -> u64 {
        self.sink.bytes_written()
    }

    /// 已写采样数。
    pub fn packets_written(&self) -> u64 {
        self.packets_written
    }

    /// 一条采样的 ES 载荷（视频补参数集、音频补 ADTS 头）。
    fn payload_of(&mut self, idx: usize, p: &Packet) -> Result<Vec<u8>> {
        let codec = self.tracks[idx].track.codec.clone();
        let kind = self.tracks[idx].track.kind;
        match kind {
            MediaKind::Video => {
                let mut es = Vec::with_capacity(p.data.len() + self.tracks[idx].params.len() + 8);
                // 关键帧前面重发参数集：从中间切入的播放器靠它才能解出画面
                let need_params = p.keyframe
                    && !self.tracks[idx].params.is_empty()
                    && !convert::has_parameter_sets(&codec, &p.data);
                if need_params {
                    es.extend_from_slice(&self.tracks[idx].params);
                }
                es.extend_from_slice(&p.data);
                Ok(es)
            }
            MediaKind::Audio if codec == CodecId::Aac => {
                // 已经是 ADTS 就原样搬（重封装 TS→TS 的场景），裸帧才补头
                if is_adts(&p.data) {
                    return Ok(p.data.clone());
                }
                let asc = &self.tracks[idx].track.extradata;
                let mut es = adts_header(asc, p.data.len()).ok_or_else(|| {
                    MediaError::Bitstream(
                        "AAC 轨缺少 AudioSpecificConfig，补不出 ADTS 头（extradata 为空或不是 ASC）".into(),
                    )
                })?;
                es.extend_from_slice(&p.data);
                Ok(es)
            }
            _ => Ok(p.data.clone()),
        }
    }

    /// 写 PAT + PMT。
    fn write_sections(&mut self) -> Result<()> {
        let pat = build_pat(PMT_PID);
        emit_payload(&mut self.sink, PAT_PID, &mut self.cc_pat, &pat)?;

        let pmt = build_pmt(&self.tracks);
        emit_payload(&mut self.sink, PMT_PID, &mut self.cc_pmt, &pmt)?;

        let written = self.sink.bytes_written() as usize;
        for t in &mut self.tracks {
            t.last_section_bytes = written;
        }
        self.sections_written = true;
        Ok(())
    }

    /// 写一个只带 adaptation field 的 PCR 包。
    ///
    /// 用一个独立的包（`afc = 2`，没有载荷）而不是塞进视频包：
    /// 简单、不会打乱 PES 的字节流，且规范允许。
    fn write_pcr(&mut self, dts_90k: i64) -> Result<()> {
        let pcr_27m = (dts_90k.max(0) as i128 * 300) as u64;
        let mut p = [0xFFu8; TS_PACKET_LEN];
        p[0] = SYNC;
        let pid = self
            .tracks
            .iter()
            .find(|t| matches!(t.track.kind, MediaKind::Video))
            .map(|t| t.pid)
            .unwrap_or(FIRST_ES_PID);
        p[1] = ((pid >> 8) as u8) & 0x1F;
        p[2] = (pid & 0xFF) as u8;
        p[3] = 0x20; // afc = 2（只有 adaptation field），cc = 0
        p[4] = 7; // adaptation field 长度
        p[5] = 0x10; // PCR 标志
        let base = pcr_27m / 300;
        let ext = (pcr_27m % 300) as u16;
        p[6] = (base >> 25) as u8;
        p[7] = (base >> 17) as u8;
        p[8] = (base >> 9) as u8;
        p[9] = (base >> 1) as u8;
        p[10] = (((base & 0x01) as u8) << 7) | 0x7E | ((ext >> 8) as u8 & 0x01);
        p[11] = (ext & 0xFF) as u8;
        self.sink.write_all(&p)
    }
}

/// PAT：`program_number → PMT PID`。
///
/// 注意 section 的**开头两个字节是 `transport_stream_id`** —— 解析器是按
/// "table_id(1) + 长度(2) + tsid(2) + 版本(1) + 段号(1) + 末段号(1)" 之后
/// 才读节目条目的，少写这两个字节会让整张表错位两个字节，症状是
/// **解析器一个节目都认不出来**（也就没有任何轨道）。
fn build_pat(pmt_pid: u16) -> Vec<u8> {
    let mut section = Vec::new();
    section.extend_from_slice(&[0x00, 0x01]); // transport_stream_id
    section.extend_from_slice(&[0xC1, 0x00, 0x00]); // version/current_next, section_number, last
    section.extend_from_slice(&PROGRAM_NUMBER.to_be_bytes());
    section.push(0xE0 | ((pmt_pid >> 8) as u8 & 0x1F));
    section.push((pmt_pid & 0xFF) as u8);
    section_payload(0x00, &section)
}

/// 给 section 补上长度与 CRC 占位，前面加上 `pointer_field`。
///
/// 长度字段必须算对：解析器是按它来切 section 的，算错就会读到别的包里，
/// 表现为"PAT/PMT 偶尔解析失败"。CRC32 我们不解也不校验，但字段必须留着。
fn section_payload(table_id: u8, section: &[u8]) -> Vec<u8> {
    let mut s = Vec::with_capacity(section.len() + 8);
    s.push(table_id);
    let sec_len = section.len() + 4; // section 内容 + CRC32
    s.push(0xB0 | ((sec_len >> 8) as u8 & 0x0F));
    s.push((sec_len & 0xFF) as u8);
    s.extend_from_slice(section);
    s.extend_from_slice(&[0, 0, 0, 0]); // CRC32 占位
    let mut out = vec![0x00u8]; // pointer_field
    out.extend_from_slice(&s);
    out
}

/// PMT：节目里有哪几条基本流、分别是什么编码、语言是什么。
///
/// section 开头是 `program_number(2) + 版本(1) + 段号(1) + 末段号(1)`，
/// 然后才是 PCR_PID —— 少写 program_number 会让解析器把 PCR_PID 读成别的字段。
fn build_pmt(tracks: &[EsTrack]) -> Vec<u8> {
    let mut section = Vec::new();
    section.extend_from_slice(&PROGRAM_NUMBER.to_be_bytes());
    section.extend_from_slice(&[0xC1, 0x00, 0x00]);
    section.push(0xE0 | ((tracks[0].pid >> 8) as u8 & 0x1F)); // PCR_PID（挂第一条轨）
    section.push((tracks[0].pid & 0xFF) as u8);
    section.extend_from_slice(&[0xF0, 0x00]); // program_info_length = 0
    for t in tracks {
        section.push(t.track.codec.ts_stream_type());
        section.push(0xE0 | ((t.pid >> 8) as u8 & 0x1F));
        section.push((t.pid & 0xFF) as u8);
        // 语言描述符：宿主界面上"这条轨是什么语言"就靠它
        let lang = t.track.language.as_deref().unwrap_or("");
        let lang = lang.as_bytes();
        if lang.len() == 3 {
            section.push(0xF0);
            section.push(6); // 描述符长度：tag(1)+len(1)+3 字母+audio_type(1)
            section.push(0x0A);
            section.push(4);
            section.extend_from_slice(lang);
            section.push(0x00);
        } else {
            section.extend_from_slice(&[0xF0, 0x00]);
        }
    }
    section_payload(0x02, &section)
}

/// 组装 PES 包。
///
/// `known_len = false` 时 PES_packet_length 写 0（"不定长"）——
/// 视频一帧常常超过 64KB，而 PES 的长度字段只有 16 位，写不下。
fn build_pes(stream_id: u8, pts: i64, dts: i64, es: &[u8], force_len: bool) -> Vec<u8> {
    let has_dts = dts != pts;
    let header_len = if has_dts { 10 } else { 5 };
    let pes_len: usize = if !force_len && es.len() + 3 + header_len > 0xFFFF {
        0
    } else {
        3 + header_len + es.len()
    };
    let mut w = ByteWriter::with_capacity(es.len() + 32);
    w.bytes(&[0x00, 0x00, 0x01, stream_id]);
    w.u16_be(pes_len as u16);
    w.u8(0x80); // '10' + 无加扰/无优先级/无对齐要求
    w.u8(if has_dts { 0xC0 } else { 0x80 }); // PTS_DTS_flags
    w.u8(header_len as u8);
    w.bytes(&encode_timestamp(pts));
    if has_dts {
        w.bytes(&encode_timestamp(dts));
    }
    w.bytes(es);
    w.into_vec()
}

/// 33 位时间戳编码（5 字节，穿插标记位）。
fn encode_timestamp(v: i64) -> [u8; 5] {
    let v = v.max(0) as u64 & 0x1_FFFF_FFFF;
    [
        0x21 | ((v >> 29) as u8 & 0x0E),
        (v >> 22) as u8,
        0x01 | ((v >> 14) as u8 & 0xFE),
        (v >> 7) as u8,
        0x01 | ((v << 1) as u8 & 0xFE),
    ]
}

/// 把一段载荷切成 TS 包写出去（末包用 adaptation field 填充）。
fn emit_payload(out: &mut Sink, pid: u16, cc: &mut u8, payload: &[u8]) -> Result<()> {
    let mut off = 0usize;
    let mut first = true;
    loop {
        let remaining = payload.len() - off;
        let take = remaining.min(PAYLOAD_CAPACITY);
        let stuffing = PAYLOAD_CAPACITY - take;
        let mut p = [0xFFu8; TS_PACKET_LEN];
        p[0] = SYNC;
        p[1] = if first { 0x40 } else { 0x00 } | (((pid >> 8) as u8) & 0x1F);
        p[2] = (pid & 0xFF) as u8;
        let payload_at;
        if stuffing > 0 {
            p[3] = 0x30 | (*cc & 0x0F); // afc = 3
            p[4] = (stuffing - 1) as u8; // adaptation_field_length
            if stuffing >= 2 {
                p[5] = 0x00; // flags = 0（纯填充）
            }
            payload_at = 4 + stuffing;
        } else {
            p[3] = 0x10 | (*cc & 0x0F); // afc = 1
            payload_at = 4;
        }
        p[payload_at..payload_at + take].copy_from_slice(&payload[off..off + take]);
        out.write_all(&p)?;
        *cc = (*cc + 1) & 0x0F;
        off += take;
        first = false;
        if off >= payload.len() {
            break;
        }
    }
    Ok(())
}

/// 90kHz 刻度 → 微秒（给上层算进度用）。
pub fn ticks_to_us(tb: Timebase, ticks: i64) -> i64 {
    tb.ticks_to_us(ticks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use me_core::{Cursor, ReadAt};
    use std::collections::HashMap;
    use std::sync::Arc;

    fn video_track() -> Track {
        let mut t = Track::new(1, MediaKind::Video, CodecId::H264, Timebase::MPEG_TS);
        t.width = 640;
        t.height = 360;
        t.extradata = {
            let mut v = vec![0, 0, 0, 1];
            v.extend_from_slice(&sps());
            v.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE, 0x3C, 0x80]);
            v
        };
        t
    }

    fn audio_track() -> Track {
        let mut t = Track::new(2, MediaKind::Audio, CodecId::Aac, Timebase::MPEG_TS);
        t.sample_rate = 44_100;
        t.channels = 2;
        t.language = Some("eng".into());
        // AudioSpecificConfig：AAC-LC / 44100 / 立体声
        t.extradata = vec![0x12, 0x10];
        t
    }

    fn sps() -> Vec<u8> {
        fn push(bits: &mut Vec<u8>, v: u64, n: u32) {
            for i in (0..n).rev() {
                bits.push(((v >> i) & 1) as u8);
            }
        }
        fn ue(bits: &mut Vec<u8>, v: u32) {
            let v1 = v as u64 + 1;
            let n = 64 - v1.leading_zeros();
            for _ in 0..n - 1 {
                bits.push(0);
            }
            for i in (0..n).rev() {
                bits.push(((v1 >> i) & 1) as u8);
            }
        }
        let mut bits = Vec::new();
        push(&mut bits, 66, 8);
        push(&mut bits, 0, 8);
        push(&mut bits, 30, 8);
        ue(&mut bits, 0);
        ue(&mut bits, 0);
        ue(&mut bits, 0);
        ue(&mut bits, 0);
        ue(&mut bits, 1);
        push(&mut bits, 0, 1);
        ue(&mut bits, 39); // 40*16 = 640
        ue(&mut bits, 22); // 23*16 = 368 - 8 = 360
        push(&mut bits, 1, 1);
        push(&mut bits, 1, 1);
        push(&mut bits, 1, 1);
        ue(&mut bits, 0);
        ue(&mut bits, 0);
        ue(&mut bits, 0);
        ue(&mut bits, 4);
        push(&mut bits, 0, 1);
        let mut bytes = vec![0x67u8];
        let mut cur = 0u8;
        let mut n = 0;
        for b in &bits {
            cur = (cur << 1) | b;
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

    fn video_packet(pts: i64, keyframe: bool) -> Packet {
        let mut v = vec![0u8, 0, 0, 1];
        if keyframe {
            v.push(0x65); // IDR
        } else {
            v.push(0x41); // 非 IDR
        }
        v.extend_from_slice(&[0x11, 0x22, 0x33, 0x44]);
        let mut p = Packet::new(1, pts, pts, 3000, v);
        p.keyframe = keyframe;
        p
    }

    fn audio_packet(pts: i64) -> Packet {
        // 裸 AAC 帧（不带 ADTS 头）
        let mut p = Packet::new(2, pts, pts, 2089, vec![0x21, 0x10, 0x05, 0x55]);
        p.keyframe = true;
        p
    }

    fn mux_both() -> Vec<u8> {
        let mut m = TsMuxer::new();
        m.set_tracks(&[video_track(), audio_track()]).unwrap();
        m.write_packet(&video_packet(0, true)).unwrap();
        m.write_packet(&audio_packet(0)).unwrap();
        m.write_packet(&video_packet(3000, false)).unwrap();
        m.write_packet(&audio_packet(2089)).unwrap();
        m.write_packet(&video_packet(6000, true)).unwrap();
        m.finish().unwrap()
    }

    #[test]
    fn output_is_188_byte_packets_with_sync_words() {
        let data = mux_both();
        assert_eq!(data.len() % TS_PACKET_LEN, 0, "整条流必须是 188 的整数倍");
        for (i, chunk) in data.chunks(TS_PACKET_LEN).enumerate() {
            assert_eq!(chunk[0], SYNC, "第 {i} 个包丢了同步字");
        }
    }

    #[test]
    fn roundtrip_through_our_own_demuxer() {
        // 最有力的自检：产物能被自己的解析器解回来，且轨道/时间戳/数据都对
        let data = mux_both();
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
        let mut d = me_container::TsDemuxer::new(src, "mem").unwrap();
        let info = d.info();
        assert_eq!(info.tracks.len(), 2, "PAT/PMT 必须能让解析器认出两条轨");
        let v = info.video().expect("要有视频轨");
        assert_eq!(v.codec, CodecId::H264);
        assert_eq!((v.width, v.height), (640, 360));
        let a = info.audio().expect("要有音频轨");
        assert_eq!(a.codec, CodecId::Aac);
        assert_eq!(a.sample_rate, 44_100);
        assert_eq!(a.channels, 2);
        assert_eq!(a.language.as_deref(), Some("eng"));

        let mut video = Vec::new();
        let mut audio = Vec::new();
        while let Some(p) = d.next_packet().unwrap() {
            match p.track_id {
                1 => video.push(p),
                2 => audio.push(p),
                _ => {}
            }
        }
        assert_eq!(video.len(), 3, "三条视频采样");
        assert_eq!(audio.len(), 2, "两条音频采样");
        assert_eq!(video[0].pts, 0);
        assert_eq!(video[1].pts, 3000);
        assert_eq!(video[2].pts, 6000);
        assert!(video[0].keyframe && video[2].keyframe && !video[1].keyframe);
        assert_eq!(audio[0].pts, 0);
        assert_eq!(audio[1].pts, 2089);
    }

    #[test]
    fn every_keyframe_carries_parameter_sets() {
        let data = mux_both();
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
        let mut d = me_container::TsDemuxer::new(src, "mem").unwrap();
        let mut keyframes = 0;
        while let Some(p) = d.next_packet().unwrap() {
            if p.track_id != 1 || !p.keyframe {
                continue;
            }
            keyframes += 1;
            assert!(
                convert::has_parameter_sets(&CodecId::H264, &p.data),
                "关键帧前面必须带 SPS/PPS，否则从中间切入的播放器黑屏"
            );
        }
        assert_eq!(keyframes, 2);
    }

    #[test]
    fn aac_gets_an_adts_header_when_input_is_raw() {
        let data = mux_both();
        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(data));
        let mut d = me_container::TsDemuxer::new(src, "mem").unwrap();
        let mut checked = 0;
        while let Some(p) = d.next_packet().unwrap() {
            if p.track_id != 2 {
                continue;
            }
            checked += 1;
            let f = me_container::ts::parse_adts(&p.data).expect("TS 里的 AAC 必须带 ADTS 头");
            assert_eq!(f.sample_rate, 44_100);
            assert_eq!(f.channels, 2);
            assert_eq!(f.frame_len, p.data.len(), "ADTS 头里的帧长必须与实际一致");
        }
        assert_eq!(checked, 2);
    }

    #[test]
    fn already_adts_input_is_not_double_wrapped() {
        let mut m = TsMuxer::new();
        m.set_tracks(&[audio_track()]).unwrap();
        // 造一个真的 ADTS 帧（7 字节头 + 4 字节载荷）
        let mut frame = adts_header(&[0x12, 0x10], 4).unwrap();
        frame.extend_from_slice(&[0x21, 0x10, 0x05, 0x55]);
        m.write_packet(&Packet::new(2, 0, 0, 2089, frame.clone())).unwrap();
        let out = m.finish().unwrap();

        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(out));
        let mut d = me_container::TsDemuxer::new(src, "mem").unwrap();
        let p = d.next_packet().unwrap().unwrap();
        assert_eq!(p.data, frame, "已经是 ADTS 的输入必须原样搬运，不能套两层头");
    }

    #[test]
    fn pat_pmt_are_repeated_for_mid_stream_joiners() {
        // 写足够多的采样，让 PAT/PMT 的周期重发被触发
        let mut m = TsMuxer::new();
        m.set_tracks(&[video_track(), audio_track()]).unwrap();
        for i in 0..400 {
            m.write_packet(&video_packet(i * 3000, i % 10 == 0)).unwrap();
            m.write_packet(&audio_packet(i * 2089)).unwrap();
        }
        let out = m.finish().unwrap();
        // 数 PAT（pid 0）出现的次数
        let mut pat_count = 0;
        let mut pmt_count = 0;
        for chunk in out.chunks(TS_PACKET_LEN) {
            if chunk[0] != SYNC {
                continue;
            }
            let pid = (((chunk[1] & 0x1F) as u16) << 8) | chunk[2] as u16;
            if pid == PAT_PID {
                pat_count += 1;
            } else if pid == PMT_PID {
                pmt_count += 1;
            }
        }
        assert!(pat_count >= 3, "PAT 必须周期重发（实际 {pat_count} 次）");
        assert_eq!(pat_count, pmt_count, "PAT 与 PMT 必须成对出现");
    }

    #[test]
    fn continuity_counters_advance_per_pid() {
        let data = mux_both();
        let mut cc: HashMap<u16, u8> = HashMap::new();
        for chunk in data.chunks(TS_PACKET_LEN) {
            if chunk[0] != SYNC {
                continue;
            }
            let pid = (((chunk[1] & 0x1F) as u16) << 8) | chunk[2] as u16;
            let afc = (chunk[3] >> 4) & 0x03;
            let this = chunk[3] & 0x0F;
            if afc == 2 {
                continue; // 纯 adaptation field 的包不参与计数
            }
            match cc.get(&pid) {
                None => {
                    cc.insert(pid, this);
                }
                Some(prev) => {
                    let expect = (*prev + 1) & 0x0F;
                    assert_eq!(this, expect, "pid 0x{pid:X} 的连续性计数器不连续");
                    cc.insert(pid, this);
                }
            }
        }
        assert!(cc.len() >= 3, "至少有 PAT/PMT/ES 三类 pid");
    }

    #[test]
    fn pcr_is_written_periodically() {
        let mut m = TsMuxer::new();
        m.set_tracks(&[video_track()]).unwrap();
        for i in 0..100 {
            m.write_packet(&video_packet(i * 3000, i % 10 == 0)).unwrap();
        }
        let out = m.finish().unwrap();
        let mut pcr_count = 0;
        for chunk in out.chunks(TS_PACKET_LEN) {
            if chunk[0] != SYNC {
                continue;
            }
            let afc = (chunk[3] >> 4) & 0x03;
            if afc == 2 && chunk[5] & 0x10 != 0 {
                pcr_count += 1;
            }
        }
        assert!(pcr_count >= 2, "PCR 必须周期性出现（实际 {pcr_count} 次）");
    }

    #[test]
    fn large_frames_span_multiple_packets() {
        let mut m = TsMuxer::new();
        m.set_tracks(&[video_track()]).unwrap();
        // 一帧 20KB：远超一个 TS 包的 184 字节
        let mut big = vec![0u8, 0, 0, 1, 0x65];
        big.extend(std::iter::repeat(0x55).take(20_000));
        let mut p = Packet::new(1, 0, 0, 3000, big.clone());
        p.keyframe = true;
        m.write_packet(&p).unwrap();
        let out = m.finish().unwrap();
        assert!(out.len() > 20_000, "大帧必须被切成多个包");

        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(out));
        let mut d = me_container::TsDemuxer::new(src, "mem").unwrap();
        let got = d.next_packet().unwrap().unwrap();
        // 关键帧前面会补上 SPS/PPS，所以解析回来的数据比原始帧长，但尾巴必须一致
        assert!(got.data.len() > big.len(), "参数集应该被加在关键帧前面");
        assert!(got.data.ends_with(&big[..]), "原始帧字节必须原样保留在尾部");
        assert!(
            convert::has_parameter_sets(&CodecId::H264, &got.data),
            "补上去的必须是参数集"
        );
    }

    #[test]
    fn timestamps_are_converted_to_90khz() {
        // TS 的时间戳固定 90kHz，而输入轨的时基可能是 15360（fMP4 常见）。
        // 直接当 90kHz 编码会让时间戳错好几倍（实测 5.9 倍）。
        let mut t = video_track();
        t.timebase = Timebase::new(15_360); // 每帧 512 刻度 = 30fps
        let mut m = TsMuxer::new();
        m.set_tracks(&[t]).unwrap();
        for i in 0..30 {
            let mut p = video_packet(i * 512, i == 0);
            p.dts = i * 512;
            p.pts = i * 512;
            m.write_packet(&p).unwrap();
        }
        let out = m.finish().unwrap();

        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(out));
        let mut d = me_container::TsDemuxer::new(src, "mem").unwrap();
        let mut last: Option<i64> = None;
        let mut count = 0;
        while let Some(p) = d.next_packet().unwrap() {
            if let Some(prev) = last {
                // 每帧应该是 1/30 秒 = 3000 个 90kHz 刻度
                assert_eq!(p.pts - prev, 3000, "第 {count} 帧的时间戳间隔不对（时基没换算？）");
            }
            last = Some(p.pts);
            count += 1;
        }
        assert_eq!(count, 30);
    }

    #[test]
    fn start_offset_is_normalized_to_zero() {
        // 解析层会如实报告负时间戳（MP4 编辑列表平移），muxer 必须把起点归零
        let mut m = TsMuxer::new();
        m.set_tracks(&[video_track()]).unwrap();
        let mut p = video_packet(0, true);
        p.pts = -3000;
        p.dts = -3000;
        m.write_packet(&p).unwrap();
        m.write_packet(&video_packet(0, false)).unwrap();
        let out = m.finish().unwrap();

        let src: Arc<dyn ReadAt> = Arc::new(Cursor::new(out));
        let mut d = me_container::TsDemuxer::new(src, "mem").unwrap();
        let first = d.next_packet().unwrap().unwrap();
        assert_eq!(first.pts, 0, "负时间戳必须被平移成 0");
        let second = d.next_packet().unwrap().unwrap();
        assert_eq!(second.pts, 3000, "平移不能改变采样间距");
    }

    #[test]
    fn errors_are_explicit() {
        let mut m = TsMuxer::new();
        assert!(m.write_packet(&video_packet(0, true)).is_err(), "没声明轨道就写采样要报错");
        assert!(m.set_tracks(&[]).is_err(), "空轨道表要报错");
        m.set_tracks(&[video_track()]).unwrap();
        assert!(m.write_packet(&audio_packet(0)).is_err(), "不在 PMT 里的轨道要报错");

        // AAC 缺少 AudioSpecificConfig：必须报错，不能产出一条没人能解的音频
        let mut m2 = TsMuxer::new();
        let mut a = audio_track();
        a.extradata.clear();
        m2.set_tracks(&[a]).unwrap();
        let e = m2.write_packet(&audio_packet(0));
        assert!(e.is_err());
        assert!(e.unwrap_err().to_string().contains("AudioSpecificConfig"));
    }

    #[test]
    fn timestamp_encoding_matches_the_demuxer() {
        // 编解码必须互为逆运算（33 位、穿插标记位）
        for v in [0i64, 1, 90_000, 93600, 4_294_967_295, 8_589_934_591] {
            let e = encode_timestamp(v);
            let got = me_container::ts::read_timestamp(&e);
            assert_eq!(got, v & 0x1_FFFF_FFFF, "时间戳 {v} 编解码不一致");
        }
    }
}
