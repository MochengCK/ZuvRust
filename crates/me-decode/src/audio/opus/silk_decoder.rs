//! SILK 解码器的**整合层**（参考实现 `silk/dec_API.c` 的 `silk_Decode`）。
//!
//! 把前几轮的分层实现（熵层 / 参数重建 / 核心合成 / 立体声解混 / 重采样）串成一次
//! "SILK 帧解码"，并负责参考实现里那一堆**声道与采样率的状态机**：
//!
//! ```text
//! 每包第一帧：按 payload_ms 定 nb_subfr / nFramesPerPacket，按 fs_kHz 配表与重采样器
//! 单→双声道：第二声道整体重新初始化（silk_init_decoder）
//! 双→单声道：把声道 0 的重采样器状态抄给声道 1（stereo_to_mono）
//! 上一帧只编了中间声道、这一帧有侧声道：侧声道的预测类状态整体复位
//! 每帧：立体声预测 → 逐声道（中间 / 侧）→ 中间侧解混 → 重采样 → 交织
//! ```
//!
//! ## 为什么这些"簿记"要单独写一层
//!
//! 它们全是**跨帧/跨包的状态转移**，而且条件很细：
//! - `set_fs` 只在**包的第一帧**调用（不是每帧）—— 每帧调会把预测状态反复复位；
//! - 单→双声道的第二声道必须是**全新状态**（不是"接着解"），否则前几帧会有残留；
//! - 双→单声道时重采样器状态要**继承**，不然采样率没变却会听到一次瞬态。
//!
//! 这些语义在对拍里体现为"某一段整体错位"，光看单帧结果看不出原因，
//! 所以这里照参考实现逐条抄，并在注释里写清"为什么"。
//!
//! ## 输出
//!
//! [`SilkDecoder::decode_frame`] 每次产出一条**交织好的**输出（`n_channels_api` 个声道），
//! 采样率是初始化时给的 API 采样率（引擎用 48 kHz）。内部采样率（8/12/16 kHz）由
//! 包头的带宽决定。

use me_core::{MediaError, Result};

use super::celt::{Bandwidth, OpusMode, Toc};
use super::entropy::RangeDecoder;
use super::silk::{
    self, SilkChannel, CODE_CONDITIONALLY, CODE_INDEPENDENTLY, CODE_INDEPENDENTLY_NO_LTP_SCALING,
    MAX_FRAME_LENGTH, MAX_LPC_ORDER, MAX_SUB_FRAME_LENGTH, TYPE_NO_VOICE_ACTIVITY,
};
use super::silk_params::{self, SilkDecoderControl};
use super::silk_plc;
use super::silk_resample::SilkResampler;
use super::silk_synth::{self, SilkStereoState};

/// 参考实现里 `samplesOut1_tmp` 的长度（`MAX_FS_KHZ * MAX_FRAME_LENGTH_MS + 2`）。
const TMP_LEN: usize = 16 * 20 + 2;
/// 一帧输出的最大样点数（320 个内部样点 @16 kHz → 960 @48 kHz）。
const MAX_OUT: usize = 960;

/// SILK 解码器（一个 Ogg/Opus 逻辑流一个实例）。
pub struct SilkDecoder {
    /// 两个内部声道；`None` 表示"还没用到"。单→双声道切换时第二声道会被重建。
    channels: [Option<SilkChannel>; 2],
    /// 上一个包的内部声道数（判"单↔双"的切换）。
    n_channels_internal: usize,
    /// 上一个包的**输出**声道数（判"输出声道数变了"要重置立体声/重采样器）。
    n_channels_api: usize,
    /// 输出声道数（引擎固定 2）。
    api_channels: usize,
    /// API 采样率（引擎固定 48000）。
    api_rate: i32,
    stereo: SilkStereoState,
    /// 每条内部声道的重采样器（内部率 → API 率）。
    resamplers: [Option<SilkResampler>; 2],
    prev_decode_only_middle: bool,
    /// 参考实现的 `samplesOut1_tmp`：解码结果写在 `[2..)`，解混结果在 `[1..)`。
    mid: [i16; TMP_LEN],
    side: [i16; TMP_LEN],
    /// 一帧的激励脉冲（复用，避免每次分配）。
    pulses: [i32; MAX_FRAME_LENGTH],
    /// 一帧的内部采样率输出（复用）。
    xq: [i16; MAX_FRAME_LENGTH],
    /// 重采样输入（`samplesOut1_tmp[n][1..]` 的副本，避免与重采样器状态的借用冲突）。
    resample_in: [i16; MAX_FRAME_LENGTH],
    /// 重采样输出（复用）。
    resample_out: [i16; MAX_OUT],
    /// 交织好的输出。
    out: [i16; MAX_OUT * 2],
    out_len: usize,
}

impl SilkDecoder {
    /// 新建（`silk_InitDecoder` + 解码器初始化）。
    pub fn new(api_rate: i32, api_channels: usize) -> Result<Self> {
        if api_rate != 48000 {
            return Err(MediaError::Unsupported(format!(
                "SILK：目前只支持 48 kHz 输出（收到 {api_rate} Hz）"
            )));
        }
        if !(1..=2).contains(&api_channels) {
            return Err(MediaError::Unsupported(format!(
                "SILK：输出声道数 {api_channels} 不支持"
            )));
        }
        Ok(Self {
            channels: [None, None],
            n_channels_internal: 0,
            n_channels_api: 0,
            api_channels,
            api_rate,
            stereo: SilkStereoState::default(),
            resamplers: [None, None],
            prev_decode_only_middle: false,
            mid: [0; TMP_LEN],
            side: [0; TMP_LEN],
            pulses: [0; MAX_FRAME_LENGTH],
            xq: [0; MAX_FRAME_LENGTH],
            resample_in: [0; MAX_FRAME_LENGTH],
            resample_out: [0; MAX_OUT],
            out: [0; MAX_OUT * 2],
            out_len: 0,
        })
    }

    /// 上一帧产出的交织输出。
    pub fn out(&self) -> &[i16] {
        &self.out[..self.out_len]
    }

    /// **仅供调试**：上一帧喂给重采样器的输入（对应参考实现 `samplesOut1_tmp[n][1..]`）。
    #[doc(hidden)]
    pub fn debug_last_resample_input(&self) -> &[i16] {
        &self.resample_in
    }

    /// 解一帧 SILK（对应参考实现的一次 `silk_Decode` 调用）。
    ///
    /// `first_frame_in_packet` 是参考实现的 `newPacketFlag`：为真时清"本包已解帧数"
    /// 并（在包的第一帧）按包头重配采样率/帧长与重采样器。
    ///
    /// 返回本帧**每声道**的输出样点数；样点内容用 [`Self::out`] 取。
    pub fn decode_frame(
        &mut self,
        toc: Toc,
        dec: &mut RangeDecoder<'_>,
        first_frame_in_packet: bool,
    ) -> Result<usize> {
        if toc.mode == OpusMode::CeltOnly {
            return Err(MediaError::Unsupported(
                "SILK: CELT-only 的包不该进 SILK 解码器".into(),
            ));
        }
        let fs_khz = silk_internal_fs_khz(toc);
        let payload_ms = toc.frame_size / 48; // 48 kHz 下的样本数 → ms
        let c = if toc.stereo { 2 } else { 1 };

        if first_frame_in_packet {
            for slot in self.channels.iter_mut().flatten() {
                slot.start_packet();
            }
        }

        // 单 → 双：第二声道整体重建（参考实现 `silk_init_decoder`）
        let mono_to_stereo = c > self.n_channels_internal;
        if mono_to_stereo {
            self.channels[1] = None;
            self.resamplers[1] = None;
        }
        let stereo_to_mono = c == 1
            && self.n_channels_internal == 2
            && self.channels[0].as_ref().map(|ch| ch.fs_khz) == Some(fs_khz);

        for slot in self.channels.iter_mut().take(c) {
            if slot.is_none() {
                *slot = Some(SilkChannel::new(fs_khz, payload_ms)?);
            }
        }

        // 包的第一帧：按包头重配（参考实现只在 `nFramesDecoded == 0` 时调 `silk_decoder_set_fs`）
        let first_in_packet_for_state = self.channels[0].as_ref().unwrap().n_frames_decoded == 0;
        if first_in_packet_for_state {
            for (n, slot) in self.channels.iter_mut().take(c).enumerate() {
                let ch = slot.as_mut().unwrap();
                let fs_changed = ch.fs_khz != fs_khz;
                ch.set_fs(fs_khz, payload_ms)?;
                // 注意：`SilkChannel::new` 会先把 fs 配好，所以"新建的声道"在这里
                // `fs_changed` 是 false —— 必须另判"重采样器还不存在"
                if fs_changed || self.resamplers[n].is_none() {
                    self.resamplers[n] =
                        Some(SilkResampler::new((fs_khz * 1000) as i32, self.api_rate)?);
                }
            }
        }

        // 输出从"单声道 API / 单声道内部"切到"双声道"时：立体声记忆清零 + 声道 1
        // 的重采样器继承声道 0（参考实现那两条 memcpy）
        if self.api_channels == 2
            && c == 2
            && (self.n_channels_api == 1 || self.n_channels_internal == 1)
        {
            self.stereo.pred_prev_q13 = [0, 0];
            self.stereo.s_side = [0, 0];
            if self.resamplers[0].is_some() {
                self.resamplers[1] = self.resamplers[0].clone();
            }
        }
        self.n_channels_internal = c;
        self.n_channels_api = self.api_channels;
        if stereo_to_mono {
            self.resamplers[1] = self.resamplers[0].clone();
        }

        // ── 包首标志（VAD / LBRR）+ LBRR 跳过，只在包的第一帧 ──
        if first_in_packet_for_state {
            for slot in self.channels.iter_mut().take(c) {
                slot.as_mut().unwrap().decode_packet_flags(dec)?;
            }
            self.skip_lbrr(dec, c)?;
        }

        let frame_index = self.channels[0].as_ref().unwrap().n_frames_decoded;
        let n_frames_per_packet = self.channels[0].as_ref().unwrap().n_frames_per_packet;
        if frame_index >= n_frames_per_packet {
            return Err(MediaError::Bitstream(format!(
                "SILK: 包内帧数超出（{frame_index} >= {n_frames_per_packet}）"
            )));
        }

        // ── 立体声预测（每帧解一次）──
        let mut mid_only = false;
        let mut pred_q13 = [0i32; 2];
        if c == 2 {
            pred_q13 = silk::stereo_decode_pred(dec)?;
            let side_vad = self.channels[1].as_ref().unwrap().vad_flags[frame_index];
            if !side_vad {
                mid_only = silk::stereo_decode_mid_only(dec)?;
            }
        }
        // 上一帧只编了中间声道、这一帧有侧声道 → 侧声道预测状态复位
        if c == 2 && !mid_only && self.prev_decode_only_middle {
            if let Some(s) = self.channels[1].as_mut() {
                s.out_buf = [0; MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH];
                s.s_lpc_q14_buf = [0; MAX_LPC_ORDER];
                s.lag_prev = 100;
                s.last_gain_index = 10;
                s.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
                s.first_frame_after_reset = true;
            }
        }
        let has_side = c == 2 && !mid_only;

        // ── 逐声道解码 ──
        let mut frame_length = 0usize;
        for ch in 0..c {
            if ch == 1 && !has_side {
                continue;
            }
            let ch0_frames = self.channels[0].as_ref().unwrap().n_frames_decoded;
            let idx = ch0_frames as i32 - ch as i32;
            let cond = if idx <= 0 {
                CODE_INDEPENDENTLY
            } else if ch > 0 && self.prev_decode_only_middle {
                CODE_INDEPENDENTLY_NO_LTP_SCALING
            } else {
                CODE_CONDITIONALLY
            };
            let state = self.channels[ch].as_mut().unwrap();
            state.decode_indices(dec, frame_index, cond)?;
            let fl = state.frame_length;
            frame_length = fl;
            state.decode_pulses(dec, &mut self.pulses)?;
            let mut control = SilkDecoderControl::default();
            silk_params::decode_parameters(state, &mut control, cond)?;
            state.first_frame_after_reset = false;
            silk_synth::decode_core(state, &mut control, &self.pulses, &mut self.xq[..fl]);
            // 好帧也要跑一遍 PLC 的"状态更新"（它把这一帧的 LPC/LTP/增益存下来，
            // 供下次丢包外推）—— 参考实现里这一步在 `silk_decode_frame` 里无条件执行
            silk_plc::plc(state, &mut control, &mut self.xq[..fl], false);
            silk_synth::finish_frame(state, &mut control, &mut self.xq[..fl], false);
            // 注意：写进 `samplesOut1_tmp` 的是**衔接与舒适噪声之后**的信号
            let buf = if ch == 0 {
                &mut self.mid
            } else {
                &mut self.side
            };
            buf[2..2 + fl].copy_from_slice(&self.xq[..fl]);
            state.n_frames_decoded += 1;
        }

        // ── 中间/侧 → 左/右（或单声道缓冲）──
        if c == 2 {
            if !has_side {
                self.side[2..2 + frame_length].fill(0);
            }
            silk_synth::stereo_ms_to_lr(
                &mut self.stereo,
                &mut self.mid,
                &mut self.side,
                pred_q13,
                fs_khz,
                frame_length,
            );
        } else {
            silk_synth::stereo_mid_only_copy(&mut self.stereo, &mut self.mid, frame_length);
        }
        self.prev_decode_only_middle = mid_only;

        self.resample_and_interleave(c, fs_khz, frame_length)
    }

    /// 丢一帧（参考实现 `silk_Decode` 的 `lostFlag == FLAG_PACKET_LOST` 分支）。
    ///
    /// 与正常路径的差别只有四处：
    ///
    /// 1. **不读码流**（不解熵、不解脉冲、不解参数）；
    /// 2. `nChannelsInternal` / `internalSampleRate` 沿用**上一个真实包**的值
    ///    （参考实现里 `DecControl` 只在 `data != NULL` 时更新）；
    /// 3. 立体声预测系数沿用 `sStereo.pred_prev_Q13`，`decode_only_middle` 取 0
    ///    （于是 `has_side = true`：两个声道都要藏）；
    /// 4. 帧尾把每条声道的 `LastGainIndex` 强制成 10 —— 去掉丢包时的增益钳位，
    ///    否则"能量正在下降时连续丢包"会出现能量反弹。
    ///
    /// `payload_ms` 是参考实现的 `DecControl.payloadSize_ms`（调用方给
    /// `IMAX(10, 1000 * audiosize / Fs)`），它决定 `nb_subfr` 与"一个包几帧"。
    pub fn decode_lost_frame(&mut self, payload_ms: u32, first_frame_in_packet: bool) -> Result<usize> {
        let c = self.n_channels_internal;
        let fs_khz = self
            .channels[0]
            .as_ref()
            .ok_or_else(|| MediaError::Unsupported("SILK: 还没解过任何包，无法隐藏".into()))?
            .fs_khz;

        if first_frame_in_packet {
            for slot in self.channels.iter_mut().flatten() {
                slot.start_packet();
            }
        }
        // 包的第一帧：按 DecControl 的 payloadSize_ms 重配帧长（采样率不变）
        if self.channels[0].as_ref().unwrap().n_frames_decoded == 0 {
            for slot in self.channels.iter_mut().take(c) {
                slot.as_mut().unwrap().set_fs(fs_khz, payload_ms)?;
            }
        }

        // 立体声预测系数：丢包时不读码流，沿用上一帧的
        let mut pred_q13 = [0i32; 2];
        if c == 2 {
            pred_q13 = self.stereo.pred_prev_q13;
        }
        // `decode_only_middle = 0` → 上一帧只编中间声道时，这一帧要把侧声道预测状态复位
        if c == 2 && self.prev_decode_only_middle {
            if let Some(s) = self.channels[1].as_mut() {
                s.out_buf = [0; MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH];
                s.s_lpc_q14_buf = [0; MAX_LPC_ORDER];
                s.lag_prev = 100;
                s.last_gain_index = 10;
                s.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
                s.first_frame_after_reset = true;
            }
        }

        let mut frame_length = 0usize;
        for ch in 0..c {
            // 丢包时 `has_side = !decode_only_middle = true` → 两个声道都要藏
            let state = self.channels[ch].as_mut().unwrap();
            frame_length = state.frame_length;
            let mut control = SilkDecoderControl::default();
            silk_plc::plc(state, &mut control, &mut self.xq[..frame_length], true);
            silk_synth::finish_frame(state, &mut control, &mut self.xq[..frame_length], true);
            let buf = if ch == 0 {
                &mut self.mid
            } else {
                &mut self.side
            };
            buf[2..2 + frame_length].copy_from_slice(&self.xq[..frame_length]);
            state.n_frames_decoded += 1;
            // 丢包时去掉增益钳位（防止能量"反弹"）
            state.last_gain_index = 10;
        }

        if c == 2 {
            silk_synth::stereo_ms_to_lr(
                &mut self.stereo,
                &mut self.mid,
                &mut self.side,
                pred_q13,
                fs_khz,
                frame_length,
            );
        } else {
            silk_synth::stereo_mid_only_copy(&mut self.stereo, &mut self.mid, frame_length);
        }
        // 注意：丢包时**不更新** `prev_decode_only_middle`（参考实现如此）

        self.resample_and_interleave(c, fs_khz, frame_length)
    }

    /// 重采样到 API 采样率 + 交织（正常路径与丢包路径共用）。
    fn resample_and_interleave(
        &mut self,
        c: usize,
        fs_khz: u32,
        frame_length: usize,
    ) -> Result<usize> {
        let n_out = frame_length * self.api_rate as usize / (fs_khz as usize * 1000);
        let n_resample = self.api_channels.min(c);
        self.out[..n_out * self.api_channels].fill(0);
        for n in 0..n_resample {
            // 重采样器读的是 `samplesOut1_tmp[n][1..]`（含 2 个前导样点缓冲的第 1 个）
            let src = if n == 0 { &self.mid } else { &self.side };
            self.resample_in[..frame_length].copy_from_slice(&src[1..1 + frame_length]);
            let r = self.resamplers[n]
                .as_mut()
                .ok_or_else(|| MediaError::Bitstream("SILK: 重采样器未初始化".into()))?;
            r.process(&mut self.resample_out, &self.resample_in, frame_length);
            for i in 0..n_out {
                self.out[n + i * self.api_channels] = self.resample_out[i];
            }
        }
        // 单声道内部流 → 双声道输出：右声道复制左声道
        if self.api_channels == 2 && c == 1 {
            for i in 0..n_out {
                self.out[1 + 2 * i] = self.out[2 * i];
            }
        }
        self.out_len = n_out * self.api_channels;
        Ok(n_out)
    }

    /// 正常解码时把 LBRR 数据解掉丢弃（参考实现那段 `skip all LBRR data`）。
    fn skip_lbrr(&mut self, dec: &mut RangeDecoder<'_>, c: usize) -> Result<()> {
        let nf = self.channels[0].as_ref().unwrap().n_frames_per_packet;
        for i in 0..nf {
            for n in 0..c {
                if !self.channels[n].as_ref().unwrap().lbrr_flags[i] {
                    continue;
                }
                if c == 2 && n == 0 {
                    let _ = silk::stereo_decode_pred(dec)?;
                    if !self.channels[1].as_ref().unwrap().lbrr_flags[i] {
                        let _ = silk::stereo_decode_mid_only(dec)?;
                    }
                }
                let cond = if i == 0 {
                    CODE_INDEPENDENTLY
                } else if self.channels[n].as_ref().unwrap().lbrr_flags[i - 1] {
                    CODE_CONDITIONALLY
                } else {
                    CODE_INDEPENDENTLY
                };
                let state = self.channels[n].as_mut().unwrap();
                silk::skip_lbrr(dec, state, i, cond, &mut self.pulses)?;
            }
        }
        Ok(())
    }
}

/// 一个包里含有几个 SILK 帧（按包头时长推：10/20 ms → 1 个、40 → 2、60 → 3）。
///
/// **注意与 `Toc::frames` 区分**：`Toc::frames` 是"这个包里塞了几个 Opus 帧"
/// （自定界帧数），而 SILK 的一个 Opus 帧内部还可能切成 2~3 个 20 ms 的 SILK 帧
/// （`silk_Decode` 被调用 2~3 次）。两者混用会让驱动少跑子帧 —— 症状是"第一帧
/// 完全对、之后整体错位"。
pub fn frames_in_packet(payload_ms: u32) -> Result<usize> {
    Ok(match payload_ms {
        10 | 20 => 1,
        40 => 2,
        60 => 3,
        other => {
            return Err(MediaError::Unsupported(format!(
                "SILK: 包时长 {other} ms 不支持（只有 10/20/40/60）"
            )))
        }
    })
}

/// SILK 的内部采样率：SILK-only 按带宽，混合模式固定 16 kHz。
/// 参考实现在 SILK-only 且带宽不是 NB/MB/WB 时会给 16 kHz 并 `silk_assert(0)`
/// —— 那是"不该出现"的兜底，这里同样按 16 kHz 处理（与发布版行为一致）。
pub fn silk_internal_fs_khz(toc: Toc) -> u32 {
    if toc.mode == OpusMode::Hybrid {
        return 16;
    }
    match toc.bandwidth {
        Bandwidth::Narrow => 8,
        Bandwidth::Medium => 12,
        _ => 16,
    }
}
