//! SILK 解码器（参考实现 `silk/` 目录）——本轮落地**熵层**：
//! 包首标志、帧索引、激励脉冲（含立体声预测的熵解码）。
//!
//! ## 为什么 SILK 可以对"逐位相同"
//!
//! 参考实现的 SILK 解码器是**纯整数**实现（`main.h` 无条件包含 `SigProc_FIX.h`；
//! 唯一带 `FIXED_POINT` 分支的 `silk_LPC_inverse_pred_gain_Q24` 只给编码器用），
//! 所以定点构建与浮点构建解出来的样本完全一致 —— SILK 这条链的对拍口径是
//! **逐位相同**，不是 CELT 那种"最大差几个量化步长"。
//!
//! ## 熵层在码流里的位置
//!
//! 一个 Opus 帧（SILK 模式下 10~60 ms）里的 SILK 数据依次是：
//!
//! ```text
//! [包首] 每声道：VAD 标志(每 SILK 帧一位) + LBRR 标志/位图
//! [每 SILK 帧] 立体声预测(仅双声道) → 帧索引 → 激励脉冲 → （后续层）参数重建 → 核心合成
//! ```
//!
//! 本模块做前两段。它们解出来的东西是后面所有层唯一的输入，**错一个符号后面全错**，
//! 所以对拍先卡这一层（参考 dump 的 `SIDX/SGAINS/SLTP/SNLSF/SPULSE/SPRED` 行）。
//!
//! ## 几个必须照抄的语义（都对着参考实现核过）
//!
//! 1. **区间解码器是"每 Opus 帧一个"**：同一帧里的 1~3 个 SILK 帧（以及包首标志）
//!    共用同一个区间解码器，几帧之间不清状态。
//! 2. **条件编码**：包内第 1 帧独立编码（增益分两级解、LTP scaling 要解），
//!    之后的帧条件编码（增益解增量、pitch 可解增量）。参考实现里"哪一帧算独立"
//!    的判据是 `FrameIndex = channel0.nFramesDecoded - n`（侧声道因此比中间声道
//!    晚一帧才转条件编码），照抄即可。
//! 3. **10 ms 帧**（nb_subfr=2）没有 NLSF 插值系数（固定为 4），pitch contour 表
//!    换成 10 ms 版本。
//! 4. **壳编码的 LSB 扩展**：脉冲数解到 `MAX_PULSES+1` 说明后面还有 LSB，
//!    最多 10 次；第 10 次时表指针要 +1（不允许再出现 `MAX_PULSES+1`）。
//! 5. **符号解码**用 `dec_map(x) = 2x-1`（0 → -1、1 → +1），参考实现刻意避开了 if。
//! 6. **LBRR**（低码率冗余，用于 FEC）在正常解码时要**跳过但照常解一遍符号**，
//!    否则区间解码器会漂。本引擎照样实现（本轮素材里没有 LBRR 帧，这条路径
//!    只有代码审查级别的保证，见 `skip_lbrr`）。

use me_core::{MediaError, Result};

use super::entropy::RangeDecoder;
use super::silk_fix::{div32_16, smlabb, smulwb};
use super::silk_tables as t;
use super::silk_tables::NlsfCodebook;

/// 一个 SILK 帧最多 4 个 5 ms 子帧（10 ms 帧是 2 个）。
pub const MAX_NB_SUBFR: usize = 4;
/// 一个包里最多 3 个 SILK 帧（60 ms）。
pub const MAX_FRAMES_PER_PACKET: usize = 3;
/// WB 的 LPC 阶数（NB/MB 是 10）。
pub const MAX_LPC_ORDER: usize = 16;
pub const MIN_LPC_ORDER: usize = 10;
/// LTP 抽头数。
pub const LTP_ORDER: usize = 5;
/// 子帧长度（ms）。
pub const SUB_FRAME_LENGTH_MS: usize = 5;
/// 最长帧（20 ms @ 16 kHz）。
pub const MAX_FRAME_LENGTH: usize = 320;
/// 最长**子**帧（5 ms @ 16 kHz）。
pub const MAX_SUB_FRAME_LENGTH: usize = 80;
/// LTP 记忆长度（ms）——参考实现 `LTP_MEM_LENGTH_MS`。
///
/// 20 ms：16 kHz 下正好是一整帧（320），8 kHz 的 20 ms 帧只有 160，所以"移窗"
/// （`silk_decode_frame` 末尾那段 memmove）是真在动，不是空操作。
pub const LTP_MEM_LENGTH_MS: usize = 20;
/// 壳编码块大小（16 个样本），块内脉冲数上限是 [`MAX_PULSES`]。
const SHELL_CODEC_FRAME_LENGTH: usize = 16;
const LOG2_SHELL_CODEC_FRAME_LENGTH: usize = 4;
const MAX_NB_SHELL_BLOCKS: usize = MAX_FRAME_LENGTH / SHELL_CODEC_FRAME_LENGTH;
/// 每块最多 16 个脉冲；解到 17 表示"后面还有 LSB"。
const MAX_PULSES: i32 = 16;
/// 脉冲"速率等级"数。
const N_RATE_LEVELS: usize = 10;
/// NLSF 残差的量化幅度（±4）；解到边界说明还有扩展位。
const NLSF_QUANT_MAX_AMPLITUDE: i32 = 4;

/// 信号类型（参考实现 `define.h`）。
pub const TYPE_NO_VOICE_ACTIVITY: i32 = 0;
pub const TYPE_UNVOICED: i32 = 1;
pub const TYPE_VOICED: i32 = 2;

/// 条件编码类型（参考实现 `define.h`）。
pub const CODE_INDEPENDENTLY: i32 = 0;
pub const CODE_INDEPENDENTLY_NO_LTP_SCALING: i32 = 1;
pub const CODE_CONDITIONALLY: i32 = 2;

/// `SILK_FIX_CONST(0.5 / STEREO_QUANT_SUB_STEPS, 16)` = `round(0.1 · 2¹⁶)`。
const STEREO_MID_STEP_Q16: i32 = 6554;

/// 采样率相关的表选择（参考实现 `silk_decoder_set_fs`）。
#[derive(Clone, Copy)]
pub struct FsTables {
    /// LPC 阶数（NB/MB 10、WB 16）。
    pub lpc_order: usize,
    /// NLSF 码本（NB/MB 或 WB）。
    pub nlsf_cb: &'static NlsfCodebook,
    /// 20 ms 帧的 pitch contour 表。
    pub pitch_contour: &'static [u8],
    /// 10 ms 帧的 pitch contour 表。
    pub pitch_contour_10ms: &'static [u8],
    /// pitch lag 低位用的均匀 iCDF（8/6/4 元，随采样率变）。
    pub pitch_low_bits: &'static [u8],
}

/// 按内部采样率（kHz）选表。
pub fn fs_tables(fs_khz: u32) -> Result<FsTables> {
    Ok(match fs_khz {
        8 => FsTables {
            lpc_order: MIN_LPC_ORDER,
            nlsf_cb: &t::NLSF_CB_NB_MB,
            pitch_contour: &t::PITCH_CONTOUR_NB_ICDF,
            pitch_contour_10ms: &t::PITCH_CONTOUR_10MS_NB_ICDF,
            pitch_low_bits: &t::UNIFORM4_ICDF,
        },
        12 => FsTables {
            lpc_order: MIN_LPC_ORDER,
            nlsf_cb: &t::NLSF_CB_NB_MB,
            pitch_contour: &t::PITCH_CONTOUR_WB_ICDF,
            pitch_contour_10ms: &t::PITCH_CONTOUR_10MS_ICDF,
            pitch_low_bits: &t::UNIFORM6_ICDF,
        },
        16 => FsTables {
            lpc_order: MAX_LPC_ORDER,
            nlsf_cb: &t::NLSF_CB_WB,
            pitch_contour: &t::PITCH_CONTOUR_WB_ICDF,
            pitch_contour_10ms: &t::PITCH_CONTOUR_10MS_ICDF,
            pitch_low_bits: &t::UNIFORM8_ICDF,
        },
        _ => {
            return Err(MediaError::Unsupported(format!(
                "SILK: 内部采样率 {fs_khz} kHz 不支持（只有 8/12/16）"
            )))
        }
    })
}

/// 一帧解出来的量化索引（参考实现 `SideInfoIndices`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SideInfoIndices {
    pub gains_indices: [i8; MAX_NB_SUBFR],
    pub ltp_index: [i8; MAX_NB_SUBFR],
    /// NLSF 索引：第 0 个是一级码本，后面是每个系数的二级残差（最多 order+1）。
    pub nlsf_indices: [i8; MAX_LPC_ORDER + 1],
    pub lag_index: i16,
    pub contour_index: i8,
    pub signal_type: i8,
    pub quant_offset_type: i8,
    pub nlsf_interp_coef_q2: i8,
    pub per_index: i8,
    pub ltp_scale_index: i8,
    pub seed: i8,
}

impl Default for SideInfoIndices {
    fn default() -> Self {
        Self {
            gains_indices: [0; MAX_NB_SUBFR],
            ltp_index: [0; MAX_NB_SUBFR],
            nlsf_indices: [0; MAX_LPC_ORDER + 1],
            lag_index: 0,
            contour_index: 0,
            signal_type: 0,
            quant_offset_type: 0,
            nlsf_interp_coef_q2: 0,
            per_index: 0,
            ltp_scale_index: 0,
            seed: 0,
        }
    }
}

/// 一个 SILK 声道的解码状态（熵层 + 参数重建 + 核心合成用到的全部跨帧状态）。
pub struct SilkChannel {
    pub fs_khz: u32,
    /// 帧内子帧数（20 ms 帧 4 个、10 ms 帧 2 个）。
    pub nb_subfr: usize,
    /// 帧长（内部采样率下的样本数）。
    pub frame_length: usize,
    /// 子帧长（内部采样率下的样本数）。
    pub subfr_length: usize,
    /// LTP 记忆长度（`LTP_MEM_LENGTH_MS * fs_khz`）。
    pub ltp_mem_length: usize,
    pub tables: FsTables,
    /// 本包每个 SILK 帧的 VAD 标志。
    pub vad_flags: [bool; MAX_FRAMES_PER_PACKET],
    pub lbrr_flag: bool,
    pub lbrr_flags: [bool; MAX_FRAMES_PER_PACKET],
    /// 本包已解到第几帧。
    pub n_frames_decoded: usize,
    pub n_frames_per_packet: usize,
    /// 上一次的信号类型（条件编码用）。
    pub ec_prev_signal_type: i32,
    /// 上一次的 lag 索引（pitch 增量编码用）。
    pub ec_prev_lag_index: i16,
    /// 最近一帧解出的索引。
    pub indices: SideInfoIndices,
    /// 上一帧的增益索引（增益增量编码的基准，参考实现 `LastGainIndex`）。
    pub last_gain_index: i32,
    /// 上一帧的 NLSF（前半帧插值用，参考实现 `prevNLSF_Q15`）。
    pub prev_nlsf_q15: [i16; MAX_LPC_ORDER],
    /// 刚复位过（采样率切换/丢包）→ 这一帧不允许 NLSF 插值。
    pub first_frame_after_reset: bool,
    /// 连续丢包计数（干净码流恒为 0；非 0 时 LPC 要做带宽扩展）。
    pub loss_cnt: i32,
    /// 上一帧的信号类型（合成时判"浊音丢包→清音"的过渡，参考实现 `prevSignalType`）。
    pub prev_signal_type: i32,
    /// 上一帧最后一个子帧的 pitch lag（参考实现 `lagPrev`，丢包路径要用）。
    pub lag_prev: i32,
    /// 上一个子帧的增益 Q16（合成时用于算增益调整因子，参考实现 `prev_gain_Q16`）。
    pub prev_gain_q16: i32,
    /// LTP 历史（`outBuf`）：容量 = `MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH`。
    pub out_buf: [i16; MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH],
    /// LPC 合成滤波器的状态（`sLPC_Q14_buf`）。
    pub s_lpc_q14_buf: [i32; MAX_LPC_ORDER],
    /// 本帧的激励（Q14，`exc_Q14`）——CNG 要取其中一个子帧当缓冲。
    pub exc_q14: [i32; MAX_FRAME_LENGTH],
    /// 丢包隐藏的状态（参考实现 `silk_PLC_struct`）。
    pub plc: super::silk_plc::SilkPlcState,
    /// 舒适噪声的状态（参考实现 `silk_CNG_struct`）。
    pub cng: super::silk_plc::SilkCngState,
}

impl SilkChannel {
    /// 全新的声道状态（参考实现 `silk_init_decoder`）：
    /// 熵状态归零，再按 `fs_khz` / `payload_ms` 配表。
    pub fn new(fs_khz: u32, payload_ms: u32) -> Result<Self> {
        let mut c = Self {
            fs_khz: 0,
            nb_subfr: 0,
            frame_length: 0,
            subfr_length: 0,
            ltp_mem_length: 0,
            // 占位，set_fs 会立刻覆盖（建表不会失败，8 kHz 是合法档）
            tables: fs_tables(8)?,
            vad_flags: [false; MAX_FRAMES_PER_PACKET],
            lbrr_flag: false,
            lbrr_flags: [false; MAX_FRAMES_PER_PACKET],
            n_frames_decoded: 0,
            n_frames_per_packet: 1,
            ec_prev_signal_type: TYPE_NO_VOICE_ACTIVITY,
            ec_prev_lag_index: 0,
            indices: SideInfoIndices::default(),
            last_gain_index: 10,
            prev_nlsf_q15: [0; MAX_LPC_ORDER],
            first_frame_after_reset: true,
            loss_cnt: 0,
            prev_signal_type: TYPE_NO_VOICE_ACTIVITY,
            lag_prev: 100,
            // 参考实现 `silk_init_decoder`：初值是 1.0（不是 0）——合成时要用它做除法
            prev_gain_q16: 65536,
            out_buf: [0; MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH],
            s_lpc_q14_buf: [0; MAX_LPC_ORDER],
            exc_q14: [0; MAX_FRAME_LENGTH],
            // `silk_init_decoder` 里会调 `silk_PLC_Reset` / `silk_CNG_Reset`；
            // 这里交给"`fs_khz` 不匹配就复位"那条路（初值 0 一定不匹配）
            plc: Default::default(),
            cng: Default::default(),
        };
        c.set_fs(fs_khz, payload_ms)?;
        Ok(c)
    }

    /// 按包参数重配（参考实现 `silk_decoder_set_fs` + `silk_Decode` 里设
    /// `nb_subfr` / `nFramesPerPacket` 的那段）。
    ///
    /// **跨包的熵状态（`ec_prev_*`）与 VAD/LBRR 标志不在这里清** ——
    /// 采样率在包之间切换（带宽切换）时必须留着，否则条件编码会错。
    /// `payload_ms` 是这一帧的时长（10/20/40/60 ms 对应 1/1/2/3 个 SILK 帧）。
    pub fn set_fs(&mut self, fs_khz: u32, payload_ms: u32) -> Result<()> {
        let (nb_subfr, n_frames_per_packet) = match payload_ms {
            10 => (2, 1),
            20 => (MAX_NB_SUBFR, 1),
            40 => (MAX_NB_SUBFR, 2),
            60 => (MAX_NB_SUBFR, 3),
            _ => {
                return Err(MediaError::Unsupported(format!(
                    "SILK: 帧时长 {payload_ms} ms 不支持（只有 10/20/40/60）"
                )))
            }
        };
        self.tables = fs_tables(fs_khz)?;
        if self.fs_khz != fs_khz {
            // 参考实现 `silk_decoder_set_fs`：换采样率会把预测类状态整体复位
            // （熵状态 `ec_prev_*` 与 `prev_gain_Q16` 不在此列 —— 前者跨包的条件编码还要用，
            //   后者只在 `silk_init_decoder` 里置 1.0）
            self.last_gain_index = 10;
            self.first_frame_after_reset = true;
            self.ec_prev_lag_index = 0;
            self.lag_prev = 100;
            self.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
            self.out_buf = [0; MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH];
            self.s_lpc_q14_buf = [0; MAX_LPC_ORDER];
        }
        self.fs_khz = fs_khz;
        self.nb_subfr = nb_subfr;
        self.n_frames_per_packet = n_frames_per_packet;
        self.subfr_length = SUB_FRAME_LENGTH_MS * fs_khz as usize;
        self.frame_length = nb_subfr * self.subfr_length;
        self.ltp_mem_length = LTP_MEM_LENGTH_MS * fs_khz as usize;
        Ok(())
    }

    /// 一个包开始时清零"本包已解帧数"（参考实现 `silk_Decode` 里 `newPacketFlag` 那段）。
    ///
    /// 这个计数参与条件编码的判据（`FrameIndex = channel0.nFramesDecoded - ch`），
    /// **必须每包清零**：不清的话跨包会被当成"包内后续帧"而错误地走条件编码。
    pub fn start_packet(&mut self) {
        self.n_frames_decoded = 0;
    }

    /// 一个包的第一帧之前：VAD 与 LBRR 标志（参考实现 `silk_Decode` 开头那段）。
    ///
    /// **必须每个包调一次**（每个声道各一次）：这些位在码流最前面，漏读就全错。
    pub fn decode_packet_flags(&mut self, dec: &mut RangeDecoder<'_>) -> Result<()> {
        for i in 0..self.n_frames_per_packet {
            self.vad_flags[i] = dec.dec_bit_logp(1) == 1;
        }
        self.lbrr_flag = dec.dec_bit_logp(1) == 1;
        self.lbrr_flags = [false; MAX_FRAMES_PER_PACKET];
        if self.lbrr_flag {
            if self.n_frames_per_packet == 1 {
                self.lbrr_flags[0] = true;
            } else {
                // 2 帧与 3 帧各有一张位图 iCDF（参考实现 `silk_LBRR_flags_iCDF_ptr`）
                let table: &[u8] = if self.n_frames_per_packet == 2 {
                    &t::LBRR_FLAGS_2_ICDF
                } else {
                    &t::LBRR_FLAGS_3_ICDF
                };
                let symbol = dec.dec_icdf(table, 8)? + 1;
                for i in 0..self.n_frames_per_packet {
                    self.lbrr_flags[i] = (symbol >> i) & 1 == 1;
                }
            }
        }
        Ok(())
    }

    /// 帧索引（参考实现 `silk_decode_indices`）。
    ///
    /// `frame_index` 是这一帧在包内的序号（VAD 标志就按它取）；
    /// `cond_coding` 由调用方按参考实现那段 `FrameIndex` 规则算好（见模块文档第 2 条）。
    pub fn decode_indices(
        &mut self,
        dec: &mut RangeDecoder<'_>,
        frame_index: usize,
        cond_coding: i32,
    ) -> Result<()> {
        let idx = &mut self.indices;
        // ① 信号类型 + 量化偏移（VAD 帧与 LBRR 帧用另一张表）
        let ix = if self.vad_flags[frame_index] {
            dec.dec_icdf(&t::TYPE_OFFSET_VAD_ICDF, 8)? as i32 + 2
        } else {
            dec.dec_icdf(&t::TYPE_OFFSET_NO_VAD_ICDF, 8)? as i32
        };
        idx.signal_type = (ix >> 1) as i8;
        idx.quant_offset_type = (ix & 1) as i8;

        // ② 增益：条件编码解增量；独立编码分两级（高 3 位走信号类型相关的表）
        if cond_coding == CODE_CONDITIONALLY {
            idx.gains_indices[0] = dec.dec_icdf(&t::DELTA_GAIN_ICDF, 8)? as i8;
        } else {
            let st = idx.signal_type as usize;
            let msb = dec.dec_icdf(&t::GAIN_ICDF[st * 8..st * 8 + 8], 8)? as i8;
            let lsb = dec.dec_icdf(&t::UNIFORM8_ICDF, 8)? as i8;
            idx.gains_indices[0] = (msb << 3) + lsb;
        }
        for i in 1..self.nb_subfr {
            idx.gains_indices[i] = dec.dec_icdf(&t::DELTA_GAIN_ICDF, 8)? as i8;
        }

        // ③ NLSF：一级码本 → 展开（决定每个系数的熵表与预测系数）→ 逐系数的二级残差
        let cb = self.tables.nlsf_cb;
        let nv = cb.n_vectors;
        let half = (idx.signal_type >> 1) as usize;
        let cb1 = dec.dec_icdf(&cb.cb1_icdf[half * nv..half * nv + nv], 8)? as usize;
        idx.nlsf_indices[0] = cb1 as i8;
        let mut ec_ix = [0i16; MAX_LPC_ORDER];
        let mut pred_q8 = [0u8; MAX_LPC_ORDER];
        nlsf_unpack(cb, cb1, &mut ec_ix, &mut pred_q8);
        for (i, &ec) in ec_ix.iter().enumerate().take(cb.order) {
            let base = ec as usize;
            let mut ix =
                dec.dec_icdf(&cb.ec_icdf[base..base + 2 * NLSF_QUANT_MAX_AMPLITUDE as usize + 1], 8)?
                    as i32;
            if ix == 0 {
                ix -= dec.dec_icdf(&t::NLSF_EXT_ICDF, 8)? as i32;
            } else if ix == 2 * NLSF_QUANT_MAX_AMPLITUDE {
                ix += dec.dec_icdf(&t::NLSF_EXT_ICDF, 8)? as i32;
            }
            idx.nlsf_indices[i + 1] = (ix - NLSF_QUANT_MAX_AMPLITUDE) as i8;
        }

        // ④ NLSF 插值系数（只有 20 ms 帧有；10 ms 帧固定 4）
        idx.nlsf_interp_coef_q2 = if self.nb_subfr == MAX_NB_SUBFR {
            dec.dec_icdf(&t::NLSF_INTERP_ICDF, 8)? as i8
        } else {
            4
        };

        // ⑤ 浊音帧：pitch lag（可增量）、contour、LTP 增益与缩放
        if idx.signal_type as i32 == TYPE_VOICED {
            let mut decode_absolute = true;
            let mut lag = idx.lag_index as i32;
            if cond_coding == CODE_CONDITIONALLY && self.ec_prev_signal_type == TYPE_VOICED {
                let delta = dec.dec_icdf(&t::PITCH_DELTA_ICDF, 8)? as i32;
                if delta > 0 {
                    lag = self.ec_prev_lag_index as i32 + (delta - 9);
                    decode_absolute = false;
                }
            }
            if decode_absolute {
                lag = dec.dec_icdf(&t::PITCH_LAG_ICDF, 8)? as i32 * (self.fs_khz as i32 >> 1);
                lag += dec.dec_icdf(self.tables.pitch_low_bits, 8)? as i32;
            }
            idx.lag_index = lag as i16;
            self.ec_prev_lag_index = idx.lag_index;

            let contour = if self.nb_subfr == MAX_NB_SUBFR {
                self.tables.pitch_contour
            } else {
                self.tables.pitch_contour_10ms
            };
            idx.contour_index = dec.dec_icdf(contour, 8)? as i8;

            idx.per_index = dec.dec_icdf(&t::LTP_PER_INDEX_ICDF, 8)? as i8;
            // 三档 VQ 码本的表长是 8/16/32，按 PERIndex 取（下标表是机器生成的）
            let per = idx.per_index as usize;
            let lo = t::LTP_GAIN_ICDF_OFFSETS[per] as usize;
            let hi = if per + 1 < t::LTP_GAIN_ICDF_OFFSETS.len() {
                t::LTP_GAIN_ICDF_OFFSETS[per + 1] as usize
            } else {
                t::LTP_GAIN_ICDF.len()
            };
            for k in 0..self.nb_subfr {
                idx.ltp_index[k] = dec.dec_icdf(&t::LTP_GAIN_ICDF[lo..hi], 8)? as i8;
            }
            idx.ltp_scale_index = if cond_coding == CODE_INDEPENDENTLY {
                dec.dec_icdf(&t::LTP_SCALE_ICDF, 8)? as i8
            } else {
                0
            };
        }
        self.ec_prev_signal_type = idx.signal_type as i32;

        // ⑥ 填充噪声的种子（每帧都解，与是否浊音无关）
        idx.seed = dec.dec_icdf(&t::UNIFORM4_ICDF, 8)? as i8;
        Ok(())
    }

    /// 激励脉冲（参考实现 `silk_decode_pulses`）：速率等级 → 每块脉冲数（含 LSB 扩展）
    /// → 壳解码 → LSB → 符号。
    ///
    /// `pulses` 至少要 `壳块数 × 16` 长：12 kHz 的 10 ms 帧（120 个样本）会多解半个块
    /// （8 个样本的尾巴写在那里但**不参与后续计算**）—— 参考实现把它写进
    /// `MAX_FRAME_LENGTH` 数组的余量里，这里也要求调用者给够。
    pub fn decode_pulses(&self, dec: &mut RangeDecoder<'_>, pulses: &mut [i32]) -> Result<()> {
        // 壳块数；12 kHz 的 10 ms 帧（120 个样本）会多出一个半块
        let mut iter = self.frame_length >> LOG2_SHELL_CODEC_FRAME_LENGTH;
        if iter * SHELL_CODEC_FRAME_LENGTH < self.frame_length {
            iter += 1;
        }
        if pulses.len() < iter * SHELL_CODEC_FRAME_LENGTH {
            return Err(MediaError::Bitstream(format!(
                "SILK: 脉冲缓冲区太短（{} < {}）",
                pulses.len(),
                iter * SHELL_CODEC_FRAME_LENGTH
            )));
        }
        let st = (self.indices.signal_type as usize) >> 1;
        let rate_level = dec.dec_icdf(&t::RATE_LEVELS_ICDF[st * (N_RATE_LEVELS - 1)..], 8)? as usize;

        let mut sum_pulses = [0i32; MAX_NB_SHELL_BLOCKS];
        let mut n_lshifts = [0i32; MAX_NB_SHELL_BLOCKS];
        for i in 0..iter {
            n_lshifts[i] = 0;
            sum_pulses[i] = dec.dec_icdf(&t::PULSES_PER_BLOCK_ICDF[rate_level * 18..], 8)? as i32;
            while sum_pulses[i] == MAX_PULSES + 1 {
                n_lshifts[i] += 1;
                // 已经拿到 10 个 LSB：表指针 +1（不允许再出现 MAX_PULSES+1）
                let off = (N_RATE_LEVELS - 1) * 18 + usize::from(n_lshifts[i] == 10);
                sum_pulses[i] = dec.dec_icdf(&t::PULSES_PER_BLOCK_ICDF[off..], 8)? as i32;
            }
        }

        // 壳解码（每块 16 个脉冲的二叉树）
        for i in 0..iter {
            let block = &mut pulses[i * SHELL_CODEC_FRAME_LENGTH..(i + 1) * SHELL_CODEC_FRAME_LENGTH];
            if sum_pulses[i] > 0 {
                shell_decode(block, dec, sum_pulses[i])?;
            } else {
                block.fill(0);
            }
        }

        // LSB 扩展：每块每个样本再补 nLS 个低位
        for i in 0..iter {
            let n_ls = n_lshifts[i];
            if n_ls > 0 {
                let block = &mut pulses[i * SHELL_CODEC_FRAME_LENGTH..(i + 1) * SHELL_CODEC_FRAME_LENGTH];
                for q in block.iter_mut() {
                    let mut abs_q = *q;
                    for _ in 0..n_ls {
                        abs_q = super::silk_fix::lshift(abs_q, 1) + dec.dec_icdf(&t::LSB_ICDF, 8)? as i32;
                    }
                    *q = abs_q;
                }
                // 标记"这块非空"，符号解码要按它决定用哪张符号表
                sum_pulses[i] |= n_ls << 5;
            }
        }

        decode_signs(dec, pulses, &sum_pulses[..iter], self.indices.signal_type as i32, self.indices.quant_offset_type as i32)?;
        Ok(())
    }
}

/// 参考实现 `silk_NLSF_unpack`：从二级码本的选择字节里展开出每个系数的
/// 熵表下标（`ec_ix`，解码侧不用、只有编码侧要）与预测系数（`pred_Q8`）。
pub fn nlsf_unpack(cb: &NlsfCodebook, cb1_index: usize, ec_ix: &mut [i16], pred_q8: &mut [u8]) {
    let order = cb.order;
    let mut p = cb1_index * order / 2;
    let mut i = 0;
    while i < order {
        let entry = cb.ec_sel[p];
        p += 1;
        let group = (2 * NLSF_QUANT_MAX_AMPLITUDE + 1) as i16;
        ec_ix[i] = (((entry >> 1) & 7) as i16) * group;
        pred_q8[i] = cb.pred_q8[i + (entry & 1) as usize * (order - 1)];
        ec_ix[i + 1] = (((entry >> 5) & 7) as i16) * group;
        pred_q8[i + 1] = cb.pred_q8[i + ((entry >> 4) & 1) as usize * (order - 1) + 1];
        i += 2;
    }
}

/// 参考实现 `silk_shell_decoder`：把"一块 16 个样本、总脉冲数 p"的二叉树逐层劈开。
///
/// 每一层的劈分都是一个 iCDF 符号（`p_child1`），`p_child2 = p - p_child1`；
/// `p == 0` 时整棵子树都是 0、**不读任何位** —— 这是壳编码省位的关键。
fn shell_decode(pulses0: &mut [i32], dec: &mut RangeDecoder<'_>, pulses4: i32) -> Result<()> {
    debug_assert_eq!(pulses0.len(), SHELL_CODEC_FRAME_LENGTH);
    let mut pulses1 = [0i32; 8];
    let mut pulses2 = [0i32; 4];
    let mut pulses3 = [0i32; 2];
    let split = |dec: &mut RangeDecoder<'_>, p: i32, table: &[u8]| -> Result<(i32, i32)> {
        if p > 0 {
            let c1 = dec.dec_icdf(&table[t::SHELL_CODE_TABLE_OFFSETS[p as usize] as usize..], 8)? as i32;
            Ok((c1, p - c1))
        } else {
            Ok((0, 0))
        }
    };
    let (a, b) = split(dec, pulses4, &t::SHELL_CODE_TABLE3)?;
    pulses3[0] = a;
    pulses3[1] = b;
    let (a, b) = split(dec, pulses3[0], &t::SHELL_CODE_TABLE2)?;
    pulses2[0] = a;
    pulses2[1] = b;
    let (a, b) = split(dec, pulses2[0], &t::SHELL_CODE_TABLE1)?;
    pulses1[0] = a;
    pulses1[1] = b;
    let (a, b) = split(dec, pulses1[0], &t::SHELL_CODE_TABLE0)?;
    pulses0[0] = a;
    pulses0[1] = b;
    let (a, b) = split(dec, pulses1[1], &t::SHELL_CODE_TABLE0)?;
    pulses0[2] = a;
    pulses0[3] = b;
    let (a, b) = split(dec, pulses2[1], &t::SHELL_CODE_TABLE1)?;
    pulses1[2] = a;
    pulses1[3] = b;
    let (a, b) = split(dec, pulses1[2], &t::SHELL_CODE_TABLE0)?;
    pulses0[4] = a;
    pulses0[5] = b;
    let (a, b) = split(dec, pulses1[3], &t::SHELL_CODE_TABLE0)?;
    pulses0[6] = a;
    pulses0[7] = b;
    let (a, b) = split(dec, pulses3[1], &t::SHELL_CODE_TABLE2)?;
    pulses2[2] = a;
    pulses2[3] = b;
    let (a, b) = split(dec, pulses2[2], &t::SHELL_CODE_TABLE1)?;
    pulses1[4] = a;
    pulses1[5] = b;
    let (a, b) = split(dec, pulses1[4], &t::SHELL_CODE_TABLE0)?;
    pulses0[8] = a;
    pulses0[9] = b;
    let (a, b) = split(dec, pulses1[5], &t::SHELL_CODE_TABLE0)?;
    pulses0[10] = a;
    pulses0[11] = b;
    let (a, b) = split(dec, pulses2[3], &t::SHELL_CODE_TABLE1)?;
    pulses1[6] = a;
    pulses1[7] = b;
    let (a, b) = split(dec, pulses1[6], &t::SHELL_CODE_TABLE0)?;
    pulses0[12] = a;
    pulses0[13] = b;
    let (a, b) = split(dec, pulses1[7], &t::SHELL_CODE_TABLE0)?;
    pulses0[14] = a;
    pulses0[15] = b;
    Ok(())
}

/// 参考实现 `silk_decode_signs`：给非零脉冲补符号。
///
/// 每块一个 iCDF：它的第 0 项按"块内脉冲数（截到 6）"选 —— 脉冲越多、
/// 正号越可能。符号本身用 `dec_map(x) = 2x - 1` 映射。
fn decode_signs(
    dec: &mut RangeDecoder<'_>,
    pulses: &mut [i32],
    sum_pulses: &[i32],
    signal_type: i32,
    quant_offset_type: i32,
) -> Result<()> {
    let base = 7 * (quant_offset_type + (signal_type << 1)) as usize;
    for (i, &p) in sum_pulses.iter().enumerate() {
        if p > 0 {
            let icdf = [t::SIGN_ICDF[base + super::silk_fix::min(p & 0x1F, 6) as usize], 0];
            let block = &mut pulses[i * SHELL_CODEC_FRAME_LENGTH..(i + 1) * SHELL_CODEC_FRAME_LENGTH];
            for q in block.iter_mut() {
                if *q > 0 {
                    // dec_map: 0 → -1、1 → +1
                    *q *= 2 * dec.dec_icdf(&icdf, 8)? as i32 - 1;
                }
            }
        }
    }
    Ok(())
}

/// 立体声预测系数（参考实现 `silk_stereo_decode_pred`）。
///
/// 注意最后一句 `pred[0] -= pred[1]`：返回的是"中声道预测器"与"边声道预测器"的
/// 组合形式，后面 `silk_stereo_MS_to_LR` 就是按这个口径用的。
pub fn stereo_decode_pred(dec: &mut RangeDecoder<'_>) -> Result<[i32; 2]> {
    let joint = dec.dec_icdf(&t::STEREO_PRED_JOINT_ICDF, 8)? as i32;
    let ix2 = [div32_16(joint, 5), joint - 5 * div32_16(joint, 5)];
    let mut ix = [[0i32; 3]; 2];
    for n in 0..2 {
        ix[n][2] = ix2[n];
        ix[n][0] = dec.dec_icdf(&t::UNIFORM3_ICDF, 8)? as i32;
        ix[n][1] = dec.dec_icdf(&t::UNIFORM5_ICDF, 8)? as i32;
    }
    let mut pred = [0i32; 2];
    for n in 0..2 {
        let hi = ix[n][0] + 3 * ix[n][2];
        let low_q13 = t::STEREO_PRED_QUANT_Q13[hi as usize] as i32;
        let step_q13 = smulwb(
            t::STEREO_PRED_QUANT_Q13[hi as usize + 1] as i32 - low_q13,
            STEREO_MID_STEP_Q16,
        );
        pred[n] = smlabb(low_q13, step_q13, 2 * ix[n][1] + 1);
    }
    pred[0] -= pred[1];
    Ok(pred)
}

/// `mid-only` 标志（参考实现 `silk_stereo_decode_mid_only`）：只有中间声道被编码。
pub fn stereo_decode_mid_only(dec: &mut RangeDecoder<'_>) -> Result<bool> {
    Ok(dec.dec_icdf(&t::STEREO_ONLY_CODE_MID_ICDF, 8)? != 0)
}

/// 跳过一个 LBRR 帧的数据（参考实现里"Regular decoding: skip all LBRR data"那段）。
///
/// LBRR 是给 FEC 用的低码率冗余，正常解码时要**照常解一遍符号再丢掉** ——
/// 不解的话区间解码器会停在错误的位置上，后面全错。
#[allow(clippy::too_many_arguments)]
pub fn skip_lbrr(
    dec: &mut RangeDecoder<'_>,
    channel: &mut SilkChannel,
    frame_index: usize,
    cond_coding: i32,
    pulses: &mut [i32],
) -> Result<()> {
    channel.decode_indices(dec, frame_index, cond_coding)?;
    channel.decode_pulses(dec, pulses)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fs_tables_follow_the_reference_selection() {
        // 8 kHz：NB 码本 + NB contour + 4 元低位表
        let f8 = fs_tables(8).unwrap();
        assert_eq!(f8.lpc_order, MIN_LPC_ORDER);
        assert_eq!(f8.pitch_low_bits.len(), 4);
        assert_eq!(f8.pitch_contour.len(), 11);
        assert_eq!(f8.pitch_contour_10ms.len(), 3);
        // 12 kHz：还是 NB/MB 码本，但 6 元低位表
        let f12 = fs_tables(12).unwrap();
        assert_eq!(f12.lpc_order, MIN_LPC_ORDER);
        assert_eq!(f12.pitch_low_bits.len(), 6);
        assert_eq!(f12.pitch_contour.len(), 34);
        assert_eq!(f12.pitch_contour_10ms.len(), 12);
        // 16 kHz：WB 码本（16 阶）+ 8 元低位表
        let f16 = fs_tables(16).unwrap();
        assert_eq!(f16.lpc_order, MAX_LPC_ORDER);
        assert_eq!(f16.pitch_low_bits.len(), 8);
        assert!(fs_tables(24).is_err(), "24 kHz 不是 SILK 的内部采样率");
    }

    #[test]
    fn channel_lengths_follow_frame_size() {
        let c = SilkChannel::new(16, 20).unwrap();
        assert_eq!(c.nb_subfr, 4);
        assert_eq!(c.frame_length, 320, "20 ms @ 16 kHz");
        assert_eq!(c.n_frames_per_packet, 1);
        let c = SilkChannel::new(12, 10).unwrap();
        assert_eq!(c.nb_subfr, 2);
        assert_eq!(c.frame_length, 120, "10 ms @ 12 kHz（唯一会出现 8 个壳块的组合）");
        let c = SilkChannel::new(8, 60).unwrap();
        assert_eq!(c.n_frames_per_packet, 3);
        assert_eq!(c.frame_length, 160);
        assert!(SilkChannel::new(16, 5).is_err());
    }

    #[test]
    fn nlsf_unpack_matches_the_reference_arithmetic() {
        // 用 NB/MB 码本的第 0 个向量：选择字节的第 0 项决定前两个系数的表与预测
        let cb = &t::NLSF_CB_NB_MB;
        let mut ec_ix = [0i16; MAX_LPC_ORDER];
        let mut pred = [0u8; MAX_LPC_ORDER];
        nlsf_unpack(cb, 0, &mut ec_ix, &mut pred);
        let entry = cb.ec_sel[0];
        assert_eq!(ec_ix[0], (((entry >> 1) & 7) as i16) * 9);
        assert_eq!(ec_ix[1], (((entry >> 5) & 7) as i16) * 9);
        assert!(ec_ix[0] + 8 < cb.ec_icdf.len() as i16, "表下标必须落在 ec_iCDF 里");
        assert_eq!(pred[0], cb.pred_q8[(entry & 1) as usize * (cb.order - 1)]);
        // WB 码本（order 16）也要落界
        let cb = &t::NLSF_CB_WB;
        for v in 0..cb.n_vectors {
            nlsf_unpack(cb, v, &mut ec_ix, &mut pred);
            for &ec in ec_ix.iter().take(cb.order) {
                assert!((ec as usize) + 2 * (NLSF_QUANT_MAX_AMPLITUDE as usize) < cb.ec_icdf.len());
            }
        }
    }

    #[test]
    fn shell_decode_splits_the_pulse_count() {
        // 空载荷：所有 split 的 p 都是 0 → 不读位、全零（壳编码的省位特性）
        let data = [0u8; 8];
        let mut dec = RangeDecoder::new(&data);
        let mut block = [7i32; SHELL_CODEC_FRAME_LENGTH];
        shell_decode(&mut block, &mut dec, 0).unwrap();
        assert!(block.iter().all(|&v| v == 0));
        assert_eq!(dec.tell(), 1, "p==0 时一个位都不该读");
    }
}
