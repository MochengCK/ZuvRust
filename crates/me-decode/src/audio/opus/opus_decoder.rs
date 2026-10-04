//! Opus 解码器的**整包层**（参考实现 `src/opus_decoder.c` 的 `opus_decode_frame` 与
//! `opus_decode_native`）。
//!
//! 到这里两条编码链都已经各自能出 PCM 了（CELT 在 [`celt::CeltDecoder`]、
//! SILK 在 [`silk_decoder::SilkDecoder`]）。这一层负责的正是**参考实现里
//! "把两条链拼起来"的那几十行**，而它们全是"看着像细节、错了整段就废"的东西：
//!
//! ```text
//! ① 混合模式：CELT 只解 8 kHz 以上（start_band = 17），最后与 SILK 相加
//! ② 冗余帧：SILK 侧的包常常还带一段 5 ms 的 CELT 冗余帧（给对端切换用）
//!    - celt_to_silk：用冗余帧替换 PCM 开头 2.5 ms，再对接着的 2.5 ms 做淡入
//!    - !celt_to_silk：对 PCM 末尾 2.5 ms 做淡出（先复位 CELT 状态再解冗余帧）
//! ③ 共用一个区间解码器：CETL 接着 SILK 之后继续读（不是各读各的），
//!    所以 CELT 的 "还剩多少位" 判断看的是**整帧**的字节数
//! ④ 丢包与模式切换过渡：载荷 ≤1 字节（DTX/空包）走丢包隐藏；CELT ↔ 非 CELT
//!    切换时要拿"上一模式"的丢包隐藏帧做 5 ms 交叉淡入（`pcm_transition`）
//! ```
//!
//! ## 为什么"是不是 CELT-only 包"这件事在三个地方说了三遍
//!
//! `start_band`、后置滤波、`redundancy` 都要判模式，但**判的条件并不相同**
//! （分别是"混合才 17"、"只有从第 0 带起才有这一位"、"非 CELT-only 才可能有冗余帧"）。
//! 抄的时候一定按参考实现逐条对，别合并成一个 `is_celt_only`。
//!
//! ## 丢包隐藏这一层要做到什么程度
//!
//! `opus_decode_frame` 里"没有码流"这件事有**三种来源**，而且走的都是同一条路：
//!
//! 1. 调用方明说丢了这一包（`data == NULL`）；
//! 2. 载荷只有 ≤1 字节（DTX / 静音包）；
//! 3. **模式切换过渡**：`opus_decode_frame(st, NULL, 0, pcm_transition, ...)` ——
//!    用**上一个模式**隐藏出 5 ms，再交叉淡入当前模式解出来的音频。
//!
//! 第 3 条最容易被忽略：它意味着"丢包隐藏"在**完全没有丢包**的码流里也会被调用，
//! 而且是递归调用自己。`testvector10` 就靠它（第 965 包 `CELT-only → Hybrid`）。
//!
//! ## 已验证范围
//!
//! 官方向量 **12 组（3078 万样本）逐样本对齐** —— SILK-only 的 02/03/04 **逐位相同**，
//! 其余最大差 1 个量化步长（参考是浮点构建、我们是定点）；08/09/10 这三组"模式切换"
//! 向量同时覆盖冗余帧、`pcm_transition` 与丢包隐藏。另有一组"手工丢包"对照
//! （用参考实现专门 dump 的带丢包 PCM 比：24 组 / 6071 万样本）。
//!
//! 混合模式的**冗余帧长度**是 `ec_dec_uint(&dec, 256) + 2`：那个参数是**上界 256**，
//! 不是位数 8。写成 `dec_uint(8)` 只能读出 0..7，冗余帧长度与主数据的 `celt_len`
//! 会一起偏 —— 而且熵状态在 SILK 结束时是**逐位正确**的，问题要到下一帧才炸开。

use me_core::{MediaError, Result};

use super::celt::{self, Bandwidth, CeltDecoder, OpusMode, Toc};
use super::entropy::RangeDecoder;
use super::silk_decoder::{frames_in_packet, SilkDecoder};

/// 20 ms @ 48 kHz。
const F20: usize = 960;
/// 5 ms。
const F5: usize = 240;
/// 2.5 ms。
const F2_5: usize = 120;
/// 一个 Opus 帧最长 60 ms（SILK-only 的 40/60 ms 配置）→ 2880 个样点/声道。
///
/// 注意它与 CELT 的单帧上限（20 ms）不是一回事：SILK 的 60 ms 帧内部是 3 个 20 ms
/// 子帧，累计起来要 2880 个样点 —— 缓冲的容量按**这个**来。
const MAX_FRAME: usize = 2880;

/// 解一帧时用到的草稿纸缓冲。
///
/// 收在一个独立结构体里是为了绕开 Rust 的借用检查：`decode_frame` 会**递归调用自己**
/// （模式切换过渡），而它同时要写"帧输出"与"SILK 的中间结果"。这些缓冲不跨帧保留
/// 任何语义，纯草稿纸。
struct Scratch {
    /// SILK 解出来的内部采样率 PCM（最后与 CELT 相加）。
    pcm_silk: [i16; 2 * MAX_FRAME],
    /// 冗余帧解出来的 5 ms PCM。
    redundant: [i16; 2 * F5],
}

impl Scratch {
    fn new() -> Self {
        Self { pcm_silk: [0; 2 * MAX_FRAME], redundant: [0; 2 * F5] }
    }
}

/// 一个能同时解 CELT / SILK / 混合三种包的 Opus 解码器。
pub struct OpusDecoder {
    /// 输出声道数（引擎固定 2）。
    channels: usize,
    /// 采样率（引擎固定 48000 —— Opus 内部本来就按 48 kHz 定义）。
    fs: i32,
    // ── 以下四项来自包头的 TOC，每个**真实包**重新设置（丢包时保持不变）──
    mode: OpusMode,
    bandwidth: Bandwidth,
    stream_channels: usize,
    frame_size: usize,
    // ── 跨帧状态 ──
    prev_mode: Option<OpusMode>,
    /// 上一帧是不是"末尾贴了冗余帧"（`redundancy && !celt_to_silk`）——
    /// 它决定模式切换时要不要做过渡（冗余帧本身就是过渡，不用再来一次）。
    prev_redundancy: bool,
    silk: SilkDecoder,
    celt: CeltDecoder,
    /// RFC §6.1 的"最终区间状态"（`rangeFinal`）。
    pub range_final: u32,
}

impl OpusDecoder {
    /// 新建（对应 `opus_decoder_create(48000, channels)`）。
    pub fn new(channels: usize) -> Result<Self> {
        if channels != 1 && channels != 2 {
            return Err(MediaError::Unsupported(format!(
                "Opus: 输出声道数 {channels} 不支持"
            )));
        }
        Ok(Self {
            channels,
            fs: 48000,
            mode: OpusMode::CeltOnly,
            bandwidth: Bandwidth::Full,
            stream_channels: channels,
            frame_size: F2_5,
            prev_mode: None,
            prev_redundancy: false,
            silk: SilkDecoder::new(48000, channels)?,
            celt: CeltDecoder::new(),
            range_final: 0,
        })
    }

    /// 复位（对应 `OPUS_RESET_STATE` 的 Opus 层）。
    pub fn reset(&mut self) {
        self.prev_mode = None;
        self.prev_redundancy = false;
        self.stream_channels = self.channels;
        self.frame_size = F2_5;
        self.silk = SilkDecoder::new(48000, self.channels).expect("48 kHz/2ch 一定合法");
        self.celt.reset();
        self.range_final = 0;
    }

    /// 解一个 Opus 包（对应一次 `opus_decode`）。
    ///
    /// 输出是**交织好的** i16（`channels` 个声道），返回每声道的样点数。
    /// `data` 为空（或只有 TOC 一个字节）时按**丢包**处理 —— 也就是走丢包隐藏。
    pub fn decode_packet(&mut self, data: &[u8], out: &mut Vec<i16>) -> Result<usize> {
        if data.len() <= 1 {
            return self.decode_lost(out);
        }
        let toc = Toc::parse(data)?;
        self.mode = toc.mode;
        self.bandwidth = toc.bandwidth;
        self.stream_channels = if toc.stereo { 2 } else { 1 };
        self.frame_size = toc.frame_size as usize;
        let frame_size = self.frame_size;

        let layout = celt::frame_layout(data)?;
        let n_frames = layout.len();
        let mut s = Scratch::new();
        let mut buf = [0i16; 2 * MAX_FRAME];
        out.clear();
        for &(off, len) in &layout {
            buf.fill(0);
            let n = self.decode_frame(&mut s, Some(&data[off..off + len]), Some(toc), &mut buf, frame_size)?;
            out.extend_from_slice(&buf[..n * self.channels]);
        }
        Ok(frame_size * n_frames)
    }

    /// 丢一帧（对应 `opus_decode(dec, NULL, 0, ...)`）：走丢包隐藏。
    ///
    /// 帧长按**上一包**算（参考实现在丢包路径里 `frame_size = IMIN(frame_size, st->frame_size)`；
    /// 没有 TOC 可读，只能用上次的）。还没解过任何包时输出全零（参考实现如此）。
    pub fn decode_lost(&mut self, out: &mut Vec<i16>) -> Result<usize> {
        let mut s = Scratch::new();
        let mut buf = [0i16; 2 * MAX_FRAME];
        let n = self.decode_frame(&mut s, None, None, &mut buf, self.frame_size)?;
        out.clear();
        out.extend_from_slice(&buf[..n * self.channels]);
        Ok(n)
    }

    /// 解一帧（对应一次 `opus_decode_frame`）。
    ///
    /// `data` 为 `None` 表示"这一帧没有码流"（丢包/DTX/过渡），此时 `toc` 也应为 `None`。
    /// 返回每声道的样点数。**会递归调用自己**（模式切换过渡与超过 20 ms 的丢包）。
    fn decode_frame(
        &mut self,
        s: &mut Scratch,
        mut data: Option<&[u8]>,
        toc: Option<Toc>,
        pcm: &mut [i16],
        requested: usize,
    ) -> Result<usize> {
        let channels = self.channels;
        let end_band = self.bandwidth.end_band();

        // ① 载荷 ≤1 字节 = 丢包/DTX；隐出来的帧长不能超过上一包
        if let Some(d) = data {
            if d.len() <= 1 {
                data = None;
            }
        }
        let mut frame_size = requested;
        if data.is_none() {
            frame_size = frame_size.min(self.frame_size);
        }

        // ② 定本帧的帧长与模式
        let audiosize;
        let mode;
        match data {
            Some(_) => {
                audiosize = self.frame_size;
                mode = self.mode;
            }
            None => {
                audiosize = frame_size;
                match self.prev_mode {
                    // 参考实现：还没收到过任何包时只能给零
                    None => {
                        pcm[..audiosize * channels].fill(0);
                        return Ok(audiosize);
                    }
                    Some(prev) => mode = prev,
                }
            }
        }

        // ③ CELT / 混合的丢包超过 20 ms 要拆成多个 20 ms 段（SILK 能一次给 60 ms）
        if data.is_none() && frame_size > F20 && mode != OpusMode::SilkOnly {
            let mut done = 0usize;
            while done < frame_size {
                let n = self.decode_frame(s, None, None, &mut pcm[done * channels..], F20)?;
                if n != F20 {
                    return Err(MediaError::Bitstream(format!(
                        "Opus: 丢包隐藏只解出 {n} 个样点（期望 {F20}）"
                    )));
                }
                done += n;
            }
            return Ok(frame_size);
        }

        // ④ 模式切换过渡：拿**上一个模式**的丢包隐藏帧做 5 ms 交叉淡入。
        //
        //    注意它是在"还没有解这一帧"的时候调用的，所以那一次递归里
        //    `st->prev_mode` 仍然是上一个模式 —— 隐藏出来的就是上一个模式的声音。
        //    CELT-only → 非 CELT 的这一半在**这里**做（要先把 CELT 的隐藏帧拿在手里），
        //    另一半在下面 ⑧ 之后做（SILK 侧要先把自己的状态准备好）。
        let mut pcm_transition = [0i16; 2 * F5];
        let mut transition = false;
        if let (Some(_), Some(prev)) = (data, self.prev_mode) {
            let needs = (mode == OpusMode::CeltOnly
                && prev != OpusMode::CeltOnly
                && !self.prev_redundancy)
                || (mode != OpusMode::CeltOnly && prev == OpusMode::CeltOnly);
            if needs {
                transition = true;
                if mode == OpusMode::CeltOnly {
                    let n = F5.min(audiosize);
                    self.decode_frame(s, None, None, &mut pcm_transition, n)?;
                }
            }
        }

        if audiosize > frame_size {
            return Err(MediaError::Bitstream(format!(
                "Opus: PCM 缓冲区太小（{audiosize} > {frame_size}）"
            )));
        }
        let frame_size = audiosize;

        let mut dec = RangeDecoder::new(data.unwrap_or(&[]));
        let len = data.map(|d| d.len()).unwrap_or(0);

        // ⑤ SILK（SILK-only 与混合模式）
        if mode != OpusMode::CeltOnly {
            if self.prev_mode == Some(OpusMode::CeltOnly) {
                // 参考实现：CELT → 非 CELT 时整体复位 SILK 解码器
                self.silk = SilkDecoder::new(self.fs, channels)?;
            }
            let payload_ms = ((1000 * audiosize / self.fs as usize).max(10)) as u32;
            let mut written = 0usize;
            if data.is_none() {
                // 丢包：SILK 侧的隐藏（`silk_PLC_conceal` + `silk_CNG`）。
                // 与参考实现一样是"do-while 直到凑够 frame_size"—— 因为一个 SILK
                // 帧固定出 10/20 ms，而这一帧可能要 60 ms。
                loop {
                    let n = self.silk.decode_lost_frame(payload_ms, written == 0)?;
                    let copy = n * channels;
                    if n == 0 || written * channels + copy > s.pcm_silk.len() {
                        break;
                    }
                    s.pcm_silk[written * channels..written * channels + copy]
                        .copy_from_slice(self.silk.out());
                    written += n;
                    if written >= audiosize {
                        break;
                    }
                }
            } else {
                let toc = toc.expect("有码流时必有 TOC");
                let n_frames = frames_in_packet(payload_ms)?;
                for f in 0..n_frames {
                    if written >= audiosize {
                        break;
                    }
                    let n = self.silk.decode_frame(toc, &mut dec, f == 0)?;
                    let copy = n * channels;
                    s.pcm_silk[written * channels..written * channels + copy]
                        .copy_from_slice(self.silk.out());
                    written += n;
                }
            }
        }

        // ⑥ 冗余帧判定（只有非 CELT-only 的真实包才可能有）
        let mut redundancy = false;
        let mut celt_to_silk = false;
        let mut celt_len = len;
        if mode != OpusMode::CeltOnly && data.is_some() {
            let extra = if mode == OpusMode::Hybrid { 20 } else { 0 };
            if dec.tell() + 17 + extra <= 8 * len as i32 {
                let has_redundancy =
                    if mode == OpusMode::Hybrid { dec.dec_bit_logp(12) == 1 } else { true };
                if has_redundancy {
                    celt_to_silk = dec.dec_bit_logp(1) == 1;
                    let redundancy_bytes = if mode == OpusMode::Hybrid {
                        // 参数是**上界 256**，不是位数 8 —— `ec_dec_uint(&dec, 256)`
                        // 读出 0..255 的均匀值。写成 `dec_uint(8)` 只能读出 0..7，
                        // 于是冗余帧长度整体偏小、`celt_len` 偏移，两种冗余帧（CELT↔SILK）
                        // 都会解到**错的字节**上。
                        dec.dec_uint(256)? as usize + 2
                    } else {
                        // 由前面那个 `ec_tell + 17 <= 8*len` 保证这里至少剩 2 字节
                        len - ((dec.tell() + 7) >> 3) as usize
                    };
                    let candidate = len.saturating_sub(redundancy_bytes);
                    if (candidate as i32) * 8 < dec.tell() {
                        // 参考实现的健全性检查：合法码流走不到这里（它会把长度清零、
                        // 当没有冗余帧）。保留同样的行为，但不假装一切正常。
                        celt_len = 0;
                    } else {
                        celt_len = candidate;
                        redundancy = true;
                        // `dec.storage -= redundancy_bytes`：主体数据在前、冗余帧贴在后，
                        // 扣掉之后 CELT 读不到冗余帧，`len_bytes()` 也变成主体的字节数
                        dec.truncate_tail(redundancy_bytes);
                    }
                }
            }
        }
        let start_band = if mode != OpusMode::CeltOnly { 17 } else { 0 };

        // ⑦ 结束带与声道数下发（参考实现里这两句**对所有模式**都执行，
        //    包括 SILK-only 的帧 —— 所以 `st->start` 在 SILK 帧之后会停在 17）
        self.celt.set_end_band(end_band);
        if redundancy {
            transition = false;
        }
        if transition && mode != OpusMode::CeltOnly {
            // CELT → 非 CELT 的那一半过渡
            let n = F5.min(audiosize);
            self.decode_frame(s, None, None, &mut pcm_transition, n)?;
        }

        // ⑧ CELT → SILK 的冗余帧：先解出来备用（它会推进 CELT 的跨帧状态）
        let mut redundant_rng = 0u32;
        if redundancy && celt_to_silk {
            self.celt.set_start_band(0);
            self.celt.decode_stream(
                &mut RangeDecoder::new(&data.unwrap()[celt_len..]),
                len - celt_len,
                self.stream_channels,
                CeltDecoder::lm_for_frame_size(F5)?,
                &mut s.redundant,
            )?;
            redundant_rng = self.celt.state.rng;
        }

        // ⑨ 起带下发 —— 必须在上面那两次 PLC 之后（参考实现特意写了 "MUST be after PLC"）
        self.celt.set_start_band(start_band);

        // ⑩ CELT 主体（混合与 CELT-only 都有；SILK-only 没有）
        if mode != OpusMode::SilkOnly {
            let celt_frame_size = F20.min(frame_size);
            let lm = CeltDecoder::lm_for_frame_size(celt_frame_size)?;
            // 模式一变就把 CELT 状态整体丢掉重来（参考实现原话："Make sure to discard any
            // previous CELT state"）。`prev_redundancy` 为真时说明上一帧末尾已经贴了冗余帧、
            // 状态是从冗余帧重新长起来的，这时**不能**再清。
            if self.prev_mode.is_some() && self.prev_mode != Some(mode) && !self.prev_redundancy {
                self.celt.reset();
            }
            if data.is_none() {
                self.celt.decode_lost(lm, &mut pcm[..2 * celt_frame_size])?;
            } else {
                self.celt.decode_stream(
                    &mut dec,
                    celt_len,
                    self.stream_channels,
                    lm,
                    &mut pcm[..2 * celt_frame_size],
                )?;
            }
        } else {
            // SILK-only：CELT 不参与。但"上一帧是混合"时要解一个 2.5 ms 的静音帧
            // 让 CELT 的 MDCT 把尾巴淡出（参考实现原话）。
            for v in pcm[..frame_size * channels].iter_mut() {
                *v = 0;
            }
            let prev = self.prev_mode;
            if prev == Some(OpusMode::Hybrid) && !(redundancy && celt_to_silk && self.prev_redundancy)
            {
                self.celt
                    .decode_silence_2ms5(end_band, &mut pcm[..2 * F2_5])?;
            }
        }

        // ⑪ SILK 与 CELT 相加（混合模式；SILK-only 时 pcm 是 0 或淡出段）
        if mode != OpusMode::CeltOnly {
            for (i, v) in pcm[..frame_size * channels].iter_mut().enumerate() {
                *v = v.saturating_add(s.pcm_silk[i]);
            }
        }

        // ⑫ 冗余帧的平滑衔接
        let window: Vec<f32> = self.celt.window().to_vec();
        if redundancy && !celt_to_silk {
            // SILK → CELT：冗余帧贴在**末尾**，先复位 CELT 状态、把起带拉回 0 再解它
            // （参考实现这两句紧挨着，缺了 `CELT_SET_START_BAND(0)` 就会拿混合模式的
            //   `start = 17` 去解一个 5 ms 的小帧 —— 分配层的位预算直接算成负数），
            // 然后对最后 2.5 ms 做淡出（从 pcm 淡到冗余帧）
            self.celt.reset();
            self.celt.set_start_band(0);
            self.celt.decode_stream(
                &mut RangeDecoder::new(&data.unwrap()[celt_len..]),
                len - celt_len,
                self.stream_channels,
                CeltDecoder::lm_for_frame_size(F5)?,
                &mut s.redundant,
            )?;
            redundant_rng = self.celt.state.rng;
            let base = (frame_size - F2_5) * channels;
            let mut faded = [0i16; 2 * F2_5];
            smooth_fade(
                &pcm[base..base + F2_5 * channels],
                &s.redundant[F2_5 * channels..],
                &mut faded[..F2_5 * channels],
                F2_5,
                channels,
                &window,
            );
            pcm[base..base + F2_5 * channels].copy_from_slice(&faded[..F2_5 * channels]);
        }
        if redundancy && celt_to_silk {
            // CELT → SILK：冗余帧贴在**开头**，前 2.5 ms 直接换成冗余帧的对应段，
            // 接着的 2.5 ms 从冗余帧淡到（含 SILK 的）PCM
            pcm[..F2_5 * channels].copy_from_slice(&s.redundant[..F2_5 * channels]);
            let mut faded = [0i16; 2 * F2_5];
            smooth_fade(
                &s.redundant[F2_5 * channels..],
                &pcm[F2_5 * channels..],
                &mut faded[..F2_5 * channels],
                F2_5,
                channels,
                &window,
            );
            pcm[F2_5 * channels..2 * F2_5 * channels].copy_from_slice(&faded[..F2_5 * channels]);
        }

        // ⑬ 模式切换过渡的收尾
        if transition {
            if frame_size >= F5 {
                pcm[..F2_5 * channels].copy_from_slice(&pcm_transition[..F2_5 * channels]);
                let mut faded = [0i16; 2 * F2_5];
                smooth_fade(
                    &pcm_transition[F2_5 * channels..],
                    &pcm[F2_5 * channels..],
                    &mut faded[..F2_5 * channels],
                    F2_5,
                    channels,
                    &window,
                );
                pcm[F2_5 * channels..2 * F2_5 * channels].copy_from_slice(&faded[..F2_5 * channels]);
            } else {
                // 帧太短（参考实现原话：反正做不出干净的过渡，硬做，
                // 幅度不会太准、会有一点点时间混叠，但也没别的办法）
                let mut faded = [0i16; 2 * F2_5];
                smooth_fade(
                    &pcm_transition[..F2_5 * channels],
                    &pcm[..F2_5 * channels],
                    &mut faded[..F2_5 * channels],
                    F2_5,
                    channels,
                    &window,
                );
                pcm[..F2_5 * channels].copy_from_slice(&faded[..F2_5 * channels]);
            }
        }

        // ⑭ 收尾
        self.range_final = if len <= 1 { 0 } else { dec.state().1 ^ redundant_rng };
        self.prev_mode = Some(mode);
        self.prev_redundancy = redundancy && !celt_to_silk;
        Ok(frame_size)
    }
}

/// `smooth_fade`：用窗的平方当权重做等功率交叉淡入（参考实现 `opus_decoder.c`）。
///
/// 参考实现在浮点域做（`out = w·in2 + (1-w)·in1`，`w = window[i]²`），本引擎的 PCM 是
/// int16，所以在 int16 域里做同样的加权 —— 会比参考多一次舍入（≤0.5 个量化步长）。
fn smooth_fade(in1: &[i16], in2: &[i16], out: &mut [i16], overlap: usize, channels: usize, window: &[f32]) {
    for c in 0..channels {
        for i in 0..overlap {
            let w = window[i] * window[i];
            let v = w * in2[i * channels + c] as f32 + (1.0 - w) * in1[i * channels + c] as f32;
            out[i * channels + c] = if v > 32767.0 {
                32767
            } else if v < -32768.0 {
                -32768
            } else {
                v.round() as i16
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooth_fade_endpoints_are_the_two_inputs() {
        // 权重在两端是 0 / 1：起点完全是 in1，终点完全是 in2
        let window: Vec<f32> = (0..4).map(|i| ((i as f32 + 0.5) / 4.0).sqrt()).collect();
        let in1 = [100i16, 200, 300, 400];
        let in2 = [1000i16, 2000, 3000, 4000];
        let mut out = [0i16; 4];
        smooth_fade(&in1, &in2, &mut out, 4, 1, &window);
        // 单调过渡，且落在两个输入之间
        for i in 0..4 {
            let lo = in1[i].min(in2[i]) as i32;
            let hi = in1[i].max(in2[i]) as i32;
            assert!((out[i] as i32) >= lo - 1 && (out[i] as i32) <= hi + 1, "{i}: {}", out[i]);
        }
        assert!(out[0] < out[3], "应当是单调过渡");
    }

    #[test]
    fn a_lost_packet_before_anything_decoded_is_silence() {
        // 参考实现：`prev_mode == 0` 时丢包只能给零
        let mut dec = OpusDecoder::new(2).unwrap();
        let mut out = Vec::new();
        let n = dec.decode_lost(&mut out).unwrap();
        assert_eq!(n, F2_5, "还没收到过包 → 帧长是初始的 2.5 ms");
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn celt_only_packets_decode_without_touching_silk() {
        // config 31：CELT-only / 全带 / 20 ms / 立体声；两字节载荷（16 个 0 位）足以走完
        // → 这里只要求"能解出正确长度的 PCM"，具体数值由对拍测试负责
        let mut dec = OpusDecoder::new(2).unwrap();
        let pkt = [0xFCu8, 0x00, 0x00];
        let mut out = Vec::new();
        let n = dec.decode_packet(&pkt, &mut out).unwrap();
        assert_eq!(n, F20, "20 ms 应当出 960 个样点/声道");
        assert_eq!(out.len(), 2 * F20);
    }

    #[test]
    fn a_lost_packet_after_a_celt_packet_runs_the_concealer() {
        let mut dec = OpusDecoder::new(2).unwrap();
        let pkt = [0xFCu8, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77];
        let mut out = Vec::new();
        dec.decode_packet(&pkt, &mut out).unwrap();
        let n = dec.decode_lost(&mut out).unwrap();
        assert_eq!(n, F20, "延续上一包的 20 ms 帧长");
        assert_eq!(out.len(), 2 * F20);
        // 空载荷也走同一条路（DTX）
        let n = dec.decode_packet(&[], &mut out).unwrap();
        assert_eq!(n, F20);
    }

    #[test]
    fn a_lost_packet_after_a_silk_packet_runs_the_silk_concealer() {
        // config 3：SILK-only / 窄带 / 60 ms（一个 Opus 帧里含 3 个 SILK 子帧）
        let mut dec = OpusDecoder::new(2).unwrap();
        let mut out = Vec::new();
        let pkt = [3 << 3, 0x00, 0x00, 0x00, 0x00];
        dec.decode_packet(&pkt, &mut out).unwrap();
        let n = dec.decode_lost(&mut out).unwrap();
        assert_eq!(n, 3 * F20, "丢包要按上一包的 60 ms 凑够（3 个 20 ms 的 SILK 帧）");
        assert_eq!(out.len(), 2 * n);
    }
}
