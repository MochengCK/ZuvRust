//! 手工跑：SILK **参数重建层**（熵层 + `silk_decode_parameters`）与参考实现对拍。
//!
//! SILK 的这条链是纯整数实现，所以对拍口径是**逐位相同**：任何一个符号解错，
//! 后面的 `tell` 就会偏，dump 里的行会立刻对不上（SPULSE 一行是整帧 160~320 个
//! 脉冲值，全对才算过）。
//!
//! 参考 dump 里用到的行（`/tmp/opus_patch_silk.py` 插桩，顺序固定）：
//!
//! ```text
//! SFLAGS <ch> <帧数> <VAD...> <LBRR_flag> <LBRR...>   包首标志（每包每声道一行）
//! SPRED  <pred0_Q13> <pred1_Q13> <mid_only>            立体声预测（每帧）
//! SCH    <帧号> <声道> <condCoding>                     即将解这一帧
//! SIDX   <帧号> <fs_kHz> <nb_subfr> <帧长> <信号类型> <量化偏移> <NLSF插值> <种子>
//!        <lagIndex> <contourIndex> <PERIndex> <LTP_scaleIndex>
//! SGAINS <增益索引…> / SLTP <LTP 索引…> / SNLSF <NLSF 索引…>
//! SPULSE <整帧的激励脉冲>
//! ```
//!
//! 这一层验的是 `Gains_Q16`（对数域增益反量化 + 滞回）、`PredCoef_Q12`（NLSF 二级码本
//! → 稳定化 → NLSF2A，含插值与复位后的禁插值）、`pitchL`（lag 码本）与 LTP 参数。
//! 全是整数运算，口径同样是**逐位相同**。
//!
//! 驱动方式与参考实现一致（`silk_Decode` 的循环）：
//! 每包先解各声道的 VAD/LBRR 标志；然后每个 SILK 帧：立体声预测（双声道时）→
//! 中间声道 → 侧声道（`mid-only` 时跳过）。条件编码类型按参考实现那段
//! `FrameIndex = channel0.nFramesDecoded - n` 的规则算 —— 测试里把它**先算再比**，
//! 所以 dump 的 `SCH` 行同时校验了这条规则本身。
//!
//! 跑法：
//!
//! ```text
//! OPUS_TV_DIR=/tmp/opus_tv_pkts \
//!   cargo test -p me-decode --test opus_silk_indices_ref -- --ignored --nocapture
//! ```

use std::env;

use me_decode::audio::opus::celt::{self, OpusMode, Toc};
use me_decode::audio::opus::entropy::RangeDecoder;
use me_decode::audio::opus::silk;
use me_decode::audio::opus::silk::{SilkChannel, CODE_CONDITIONALLY, CODE_INDEPENDENTLY, LTP_ORDER};
use me_decode::audio::opus::silk_params::{self, SilkDecoderControl};

fn parse_packets(bytes: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 4 <= bytes.len() {
        let n =
            u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]) as usize;
        pos += 4;
        assert!(pos + n <= bytes.len(), "包文件被截断");
        out.push(&bytes[pos..pos + n]);
        pos += n;
    }
    out
}

/// dump 里的一条 SILK 事件（按出现顺序）。
#[derive(Debug, Clone)]
enum Event {
    Flags { ch: usize, n_frames: usize, vad: Vec<bool>, lbrr_flag: bool, lbrr: Vec<bool> },
    Pred { pred: [i32; 2], mid_only: bool },
    Frame {
        frame: usize,
        ch: usize,
        cond: i32,
        sidx: Vec<i64>,
        gains: Vec<i64>,
        ltp: Vec<i64>,
        nlsf: Vec<i64>,
        pulses: Vec<i64>,
        params: Vec<i64>,
        nlsf_v: Vec<i64>,
    },
}

fn ints(s: &str) -> Vec<i64> {
    s.split_whitespace().map(|v| v.parse().unwrap()).collect()
}

/// 正在拼装的一帧：帧号 / 声道 / condCoding / SIDX / 增益 / LTP / NLSF / 脉冲。
type PendingFrame = (usize, usize, i32, Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>);

fn parse_events(dump: &str) -> Vec<Event> {
    let mut out = Vec::new();
    let mut cur: Option<PendingFrame> = None;
    // 一帧的 SIDX..SPULSE 之后是下一条结构行（SCH/SFLAGS/SPRED/PKT/文件尾）——
    // 必须在那时把它结掉，否则每包最后一帧会被推到下一包的开头之后。
    macro_rules! flush {
        () => {
            if let Some(c) = cur.take() {
                out.push(Event::Frame {
                    frame: c.0,
                    ch: c.1,
                    cond: c.2,
                    sidx: c.3,
                    gains: c.4,
                    ltp: c.5,
                    nlsf: c.6,
                    pulses: c.7,
                    params: c.8,
                    nlsf_v: c.9,
                });
            }
        };
    }
    for line in dump.lines() {
        if line.starts_with("PKT ") {
            flush!();
            continue;
        }
        if let Some(rest) = line.strip_prefix("SFLAGS ") {
            flush!();
            let v = ints(rest);
            let ch = v[0] as usize;
            let n_frames = v[1] as usize;
            let vad = v[2..2 + n_frames].iter().map(|&x| x == 1).collect();
            let lbrr_flag = v[2 + n_frames] == 1;
            let lbrr = v[3 + n_frames..3 + 2 * n_frames].iter().map(|&x| x == 1).collect();
            out.push(Event::Flags { ch, n_frames, vad, lbrr_flag, lbrr });
            continue;
        }
        if let Some(rest) = line.strip_prefix("SPRED ") {
            flush!();
            let v = ints(rest);
            out.push(Event::Pred { pred: [v[0] as i32, v[1] as i32], mid_only: v[2] == 1 });
            continue;
        }
        if let Some(rest) = line.strip_prefix("SCH ") {
            flush!();
            let v = ints(rest);
            cur = Some((v[0] as usize, v[1] as usize, v[2] as i32, vec![], vec![], vec![], vec![], vec![], vec![], vec![]));
            continue;
        }
        let Some(c) = cur.as_mut() else { continue };
        if let Some(rest) = line.strip_prefix("SIDX ") {
            c.3 = ints(rest);
        } else if let Some(rest) = line.strip_prefix("SGAINS ") {
            c.4 = ints(rest);
        } else if let Some(rest) = line.strip_prefix("SLTP ") {
            c.5 = ints(rest);
        } else if let Some(rest) = line.strip_prefix("SNLSF ") {
            c.6 = ints(rest);
        } else if let Some(rest) = line.strip_prefix("SPULSE ") {
            c.7 = ints(rest);
        } else if let Some(rest) = line.strip_prefix("SPARAM ") {
            c.8 = ints(rest);
        } else if let Some(rest) = line.strip_prefix("SNLSFV ") {
            c.9 = ints(rest);
        }
    }
    if let Some(c) = cur.take() {
        out.push(Event::Frame {
            frame: c.0, ch: c.1, cond: c.2, sidx: c.3, gains: c.4, ltp: c.5, nlsf: c.6, pulses: c.7,
            params: c.8,
            nlsf_v: c.9,
        });
    }
    out
}

/// SILK 的内部采样率（参考实现 `opus_decoder.c`：SILK-only 按带宽，混合固定 16 kHz）。
fn internal_fs_khz(toc: Toc) -> u32 {
    use me_decode::audio::opus::celt::Bandwidth::*;
    if toc.mode == OpusMode::Hybrid {
        return 16;
    }
    match toc.bandwidth {
        Narrow => 8,
        Medium => 12,
        _ => 16,
    }
}

#[derive(Default)]
struct Stats {
    packets: usize,
    frames: usize,
    pulses: usize,
    max_pulse: i32,
}

fn run_vector(label: &str, pkts: &[u8], dump: &str) -> Stats {
    let packets = parse_packets(pkts);
    let events = parse_events(dump);
    let mut ev = events.iter().peekable();

    // 跨包状态：两个声道 + "上帧是不是只编了中间声道"
    let mut channels: [Option<SilkChannel>; 2] = [None, None];
    let mut n_ch_internal = 0usize;
    let mut prev_mid_only = false;
    let mut st = Stats::default();
    let mut pulses_buf = vec![0i32; 320];

    let next = |ev: &mut std::iter::Peekable<std::slice::Iter<'_, Event>>, what: &str| -> Event {
        ev.next()
            .unwrap_or_else(|| panic!("{label}: 参考 dump 里没有更多事件了（在找 {what}）"))
            .clone()
    };

    for (pkt_idx, pkt) in packets.iter().enumerate() {
        let toc = Toc::parse(pkt).unwrap_or_else(|e| panic!("{label}: 包 {pkt_idx} TOC：{e}"));
        if toc.mode == OpusMode::CeltOnly {
            continue;
        }
        st.packets += 1;
        let c = if toc.stereo { 2 } else { 1 };
        // 声道数在同一流里会变（mono/stereo 交替）：
        // - mono→stereo：参考实现会 `silk_init_decoder` 第二声道（整块重置）
        // - 之后按当前包补齐缺的声道状态
        if c > n_ch_internal {
            channels[1] = None;
        }
        let fs_khz = internal_fs_khz(toc);
        let payload_ms = toc.frame_size / 48; // 48 kHz 下的样本数 → ms
        for slot in channels.iter_mut().take(c) {
            if slot.is_none() {
                *slot = Some(SilkChannel::new(fs_khz, payload_ms).unwrap());
            }
        }
        n_ch_internal = c;
        let payload = celt::frame_payload(pkt, toc, 0).unwrap();
        let mut dec = RangeDecoder::new(payload);

        // ① 包首标志（并把"本包已解帧数"清零，条件编码要用）
        for (ch, slot) in channels.iter_mut().take(c).enumerate() {
            let ch_state = slot.as_mut().unwrap();
            ch_state.set_fs(fs_khz, payload_ms).unwrap();
            ch_state.start_packet();
            ch_state.decode_packet_flags(&mut dec).unwrap_or_else(|e| {
                panic!("{label}: 包 {pkt_idx} 声道 {ch} 标志解码失败：{e}")
            });
            match next(&mut ev, "SFLAGS") {
                Event::Flags { ch: dch, n_frames, vad, lbrr_flag, lbrr } => {
                    assert_eq!(dch, ch, "{label}: 包 {pkt_idx} SFLAGS 声道顺序不对");
                    assert_eq!(n_frames, ch_state.n_frames_per_packet);
                    let got_vad: Vec<bool> = (0..n_frames).map(|i| ch_state.vad_flags[i]).collect();
                    assert_eq!(got_vad, vad, "{label}: 包 {pkt_idx} 声道 {ch} 的 VAD 标志");
                    assert_eq!(lbrr_flag, ch_state.lbrr_flag, "{label}: 包 {pkt_idx} LBRR 标志");
                    let got_lbrr: Vec<bool> = (0..n_frames).map(|i| ch_state.lbrr_flags[i]).collect();
                    assert_eq!(got_lbrr, lbrr, "{label}: 包 {pkt_idx} LBRR 位图");
                }
                other => panic!("{label}: 包 {pkt_idx} 期望 SFLAGS，得到 {other:?}"),
            }
        }

        // ② 逐帧（一个 SILK 帧 = 一次 `silk_Decode` 调用）
        //
        // 注意顺序：立体声预测的**符号**在声道循环之前解，但参考实现的 `SPRED` dump
        // 打在 `silk_stereo_MS_to_LR` 那里 —— 也就是声道循环**之后**（每个 SILK 帧
        // 一次调用、一次 dump），所以比对要放在声道之后。
        for frame in 0..channels[0].as_ref().unwrap().n_frames_per_packet {
            let mut mid_only = false;
            let mut pred = [0i32; 2];
            // 立体声预测（参考实现只在两声道时解；mid-only 标志还要看侧声道的 VAD）
            if c == 2 {
                pred = silk::stereo_decode_pred(&mut dec).unwrap();
                let side_vad = channels[1].as_ref().unwrap().vad_flags[frame];
                mid_only = if !side_vad {
                    silk::stereo_decode_mid_only(&mut dec).unwrap()
                } else {
                    false
                };
            }
            let has_side = c == 2 && !mid_only;
            for ch in 0..c {
                if ch == 1 && !has_side {
                    continue;
                }
                // 条件编码（参考实现：FrameIndex = channel0.nFramesDecoded - ch）
                let ch0_frames = channels[0].as_ref().unwrap().n_frames_decoded;
                let frame_index = ch0_frames as i32 - ch as i32;
                let cond = if frame_index <= 0 {
                    CODE_INDEPENDENTLY
                } else if ch > 0 && prev_mid_only {
                    silk::CODE_INDEPENDENTLY_NO_LTP_SCALING
                } else {
                    CODE_CONDITIONALLY
                };
                let ch_state = channels[ch].as_mut().unwrap();
                ch_state
                    .decode_indices(&mut dec, frame, cond)
                    .unwrap_or_else(|e| panic!("{label}: 包 {pkt_idx} 帧 {frame} 声道 {ch} 索引失败：{e}"));
                let n = ch_state.frame_length;
                // 给的缓冲区比帧长大：12 kHz 的 10 ms 帧会多写半个壳块（尾巴是划水的）
                let pulses = &mut pulses_buf[..];
                ch_state
                    .decode_pulses(&mut dec, pulses)
                    .unwrap_or_else(|e| panic!("{label}: 包 {pkt_idx} 帧 {frame} 声道 {ch} 脉冲失败：{e}"));

                match next(&mut ev, "SCH") {
                    Event::Frame { frame: wf, ch: wch, cond: wcond, sidx, gains, ltp, nlsf, pulses: wpulses, params, nlsf_v } => {
                        assert_eq!((wf, wch), (frame, ch), "{label}: 包 {pkt_idx} 帧/声道顺序不对");
                        assert_eq!(wcond, cond, "{label}: 包 {pkt_idx} 帧 {frame} 声道 {ch} 的 condCoding");
                        let idx = &ch_state.indices;
                        assert_eq!(sidx[0], frame as i64);
                        assert_eq!(sidx[1] as u32, ch_state.fs_khz, "{label}: fs_kHz");
                        assert_eq!(sidx[2] as usize, ch_state.nb_subfr, "{label}: nb_subfr");
                        assert_eq!(sidx[3] as usize, n, "{label}: frame_length");
                        assert_eq!(sidx[4], idx.signal_type as i64, "{label}: signalType");
                        assert_eq!(sidx[5], idx.quant_offset_type as i64, "{label}: quantOffsetType");
                        assert_eq!(sidx[6], idx.nlsf_interp_coef_q2 as i64, "{label}: NLSFInterpCoef");
                        assert_eq!(sidx[7], idx.seed as i64, "{label}: Seed");
                        let ctx = format!("{label}: 包 {pkt_idx} 帧 {frame} 声道 {ch}");
                        assert_eq!(sidx[8], idx.lag_index as i64, "{ctx} lagIndex");
                        assert_eq!(sidx[9], idx.contour_index as i64, "{ctx} contourIndex");
                        assert_eq!(sidx[10], idx.per_index as i64, "{ctx} PERIndex");
                        assert_eq!(sidx[11], idx.ltp_scale_index as i64, "{ctx} LTP_scaleIndex");
                        let order = ch_state.tables.lpc_order;
                        assert_eq!(
                            gains,
                            (0..ch_state.nb_subfr).map(|i| idx.gains_indices[i] as i64).collect::<Vec<_>>(),
                            "{label}: 包 {pkt_idx} 帧 {frame} 声道 {ch} 增益索引"
                        );
                        assert_eq!(
                            ltp,
                            (0..ch_state.nb_subfr).map(|i| idx.ltp_index[i] as i64).collect::<Vec<_>>(),
                            "{label}: LTP 索引"
                        );
                        assert_eq!(
                            nlsf,
                            (0..order + 1).map(|i| idx.nlsf_indices[i] as i64).collect::<Vec<_>>(),
                            "{label}: 包 {pkt_idx} 帧 {frame} 声道 {ch} NLSF 索引"
                        );
                        let got: Vec<i64> = pulses[..n].iter().map(|&v| v as i64).collect();
                        assert_eq!(
                            got, wpulses,
                            "{label}: 包 {pkt_idx} 帧 {frame} 声道 {ch} 的激励脉冲不一致（第一处差在 {}）",
                            got.iter().zip(&wpulses).position(|(a, b)| a != b).unwrap_or(0)
                        );
                        st.max_pulse = st.max_pulse.max(pulses[..n].iter().map(|v| v.abs()).max().unwrap_or(0));
                        st.pulses += n;

                        // NLSF（解码出来的 + 插值出来的）先逐位比 —— 它在 SPARAM 上游
                        let ch_state = channels[ch].as_mut().unwrap();
                        let order = ch_state.tables.lpc_order;
                        let nsf = ch_state.nb_subfr;
                        let nlsf_now =
                            silk_params::nlsf_decode(&ch_state.indices.nlsf_indices, ch_state.tables.nlsf_cb);
                        for i in 0..order {
                            assert_eq!(nlsf_now[i] as i64, nlsf_v[i], "{ctx} NLSF[{i}]");
                        }
                        let interp = nlsf_v[2 * order] as i32;
                        assert_eq!(
                            interp, ch_state.indices.nlsf_interp_coef_q2 as i32,
                            "{ctx} NLSFInterpCoef_Q2（dump 里的取点在解码之后）"
                        );
                        if interp < 4 {
                            for i in 0..order {
                                let want = ch_state.prev_nlsf_q15[i] as i64
                                    + ((interp as i64
                                        * (nlsf_now[i] as i64 - ch_state.prev_nlsf_q15[i] as i64))
                                        >> 2);
                                assert_eq!(want, nlsf_v[order + i], "{ctx} 插值 NLSF0[{i}]");
                            }
                        }

                        // 参数重建 + 与 SPARAM 逐项比对
                        let mut control = SilkDecoderControl::default();
                        silk_params::decode_parameters(ch_state, &mut control, cond).unwrap();
                        assert_eq!(
                            params.len(),
                            nsf + 2 * order + LTP_ORDER * nsf + 1 + nsf,
                            "{ctx} 的 SPARAM 字段数不对"
                        );
                        for k in 0..nsf {
                            assert_eq!(
                                control.gains_q16[k] as i64, params[k],
                                "{ctx} 第 {k} 个子帧的 Gains_Q16"
                            );
                        }
                        for half in 0..2 {
                            for i in 0..order {
                                assert_eq!(
                                    control.pred_coef_q12[half][i] as i64,
                                    params[nsf + half * order + i],
                                    "{ctx} PredCoef_Q12[{half}][{i}]"
                                );
                            }
                        }
                        let ltp_base = nsf + 2 * order;
                        for i in 0..LTP_ORDER * nsf {
                            assert_eq!(
                                control.ltp_coef_q14[i] as i64,
                                params[ltp_base + i],
                                "{ctx} LTPCoef_Q14[{i}]"
                            );
                        }
                        assert_eq!(
                            control.ltp_scale_q14 as i64,
                            params[ltp_base + LTP_ORDER * nsf],
                            "{ctx} LTP_scale_Q14"
                        );
                        let pitch_base = ltp_base + LTP_ORDER * nsf + 1;
                        for k in 0..nsf {
                            assert_eq!(
                                control.pitch_l[k] as i64,
                                params[pitch_base + k],
                                "{ctx} pitchL[{k}]"
                            );
                        }
                        ch_state.first_frame_after_reset = false;
                    }
                    other => panic!("{label}: 包 {pkt_idx} 期望 SCH，得到 {other:?}"),
                }
                channels[ch].as_mut().unwrap().n_frames_decoded += 1;
            }
            // ③ 这一帧的 SPRED（打在参考实现的 MS→LR 处，即声道循环之后）
            if c == 2 {
                match next(&mut ev, "SPRED") {
                    Event::Pred { pred: want, mid_only: wm } => {
                        assert_eq!(pred, want, "{label}: 包 {pkt_idx} 帧 {frame} 立体声预测");
                        assert_eq!(mid_only, wm, "{label}: 包 {pkt_idx} 帧 {frame} mid-only");
                    }
                    other => panic!("{label}: 包 {pkt_idx} 期望 SPRED，得到 {other:?}"),
                }
            }
            prev_mid_only = mid_only;
        }
        st.frames += channels[0].as_ref().unwrap().n_frames_per_packet;
    }
    assert!(ev.next().is_none(), "{label}: 参考 dump 里还有没消费的事件");
    st
}

#[test]
#[ignore = "手工跑：需要 OPUS_TV_DIR（dump 由 /tmp/opus_rebuild_all.sh 重新生成）"]
fn silk_parameters_match_reference_on_official_test_vectors() {
    let dir = env::var("OPUS_TV_DIR").expect("要设 OPUS_TV_DIR");
    // SILK-only 与混合都要：混合包的 SILK 段（16 kHz）走的是同一条熵链
    let wanted = ["testvector02", "testvector03", "testvector04", "testvector05", "testvector06", "testvector12"];
    let (mut packets, mut frames, mut pulses, mut max_pulse, mut vectors) = (0usize, 0usize, 0usize, 0i32, 0usize);
    for name in wanted {
        let pkts_path = format!("{dir}/{name}.pkts.bin");
        if !std::path::Path::new(&pkts_path).exists() {
            println!("{name}: 素材不在，跳过");
            continue;
        }
        let pkts = std::fs::read(&pkts_path).expect("读包");
        let dump = std::fs::read_to_string(format!("{dir}/{name}.dump.txt")).expect("读 dump");
        let st = run_vector(name, &pkts, &dump);
        println!(
            "{name}: 熵层 + 参数一致 {} 包 / {} 帧 / {} 个脉冲（最大幅度 {}）",
            st.packets, st.frames, st.pulses, st.max_pulse
        );
        packets += st.packets;
        frames += st.frames;
        pulses += st.pulses;
        max_pulse = max_pulse.max(st.max_pulse);
        vectors += 1;
    }
    assert!(vectors >= 6, "向量组数太少：{vectors}");
    assert!(packets >= 5000, "核对包数太少：{packets}");
    println!("合计：{vectors} 组、{packets} 包、{frames} 帧、{pulses} 个脉冲 + 每帧增益/LPC/LTP/pitch 逐位一致（最大幅度 {max_pulse}）");
}
