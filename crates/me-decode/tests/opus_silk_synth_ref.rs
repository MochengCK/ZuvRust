//! 手工跑：SILK **核心合成**（`decode_core` + 立体声 MS→LR）与参考实现对拍。
//!
//! SILK 这条链是纯整数实现，所以口径是**逐样本相同**：
//!
//! - `SCORE` 行 = `silk_decode_core` 的输出（内部采样率、未重采样、未解混）；
//! - `SLR0/SLR1` 行 = `silk_stereo_MS_to_LR` 之后两条声道的内部采样率样点。
//!
//! 与参数层的对拍不同，这一层**不做逐帧状态播种**：`out_buf`（LTP 历史）、
//! `sLPC_Q14_buf`、`lag_prev`、`prev_gain_q16`、`prev_signal_type` 全部靠顺序解码
//! 自然传承 —— 只有这样才验得到"状态传递的语义"。任意一处传错，从该帧起 `SCORE`
//! 会整体漂掉（通常表现为 rms 还对、逐样本全错）。
//!
//! 跑法：
//!
//! ```text
//! OPUS_TV_DIR=/tmp/opus_tv_pkts \
//!   cargo test -p me-decode --test opus_silk_synth_ref -- --ignored --nocapture
//! ```

use std::env;

use me_decode::audio::opus::celt::{self, Bandwidth, OpusMode, Toc};
use me_decode::audio::opus::entropy::RangeDecoder;
use me_decode::audio::opus::silk::{
    self, SilkChannel, CODE_CONDITIONALLY, CODE_INDEPENDENTLY, CODE_INDEPENDENTLY_NO_LTP_SCALING,
    MAX_FRAME_LENGTH, MAX_LPC_ORDER, TYPE_NO_VOICE_ACTIVITY,
};
use me_decode::audio::opus::silk_params::{self, SilkDecoderControl};
use me_decode::audio::opus::silk_synth::{self, SilkStereoState};

/// 参考实现里 `samplesOut1_tmp` 的长度（`MAX_FS_KHZ * MAX_FRAME_LENGTH_MS + 2`）。
const TMP_LEN: usize = 16 * 20 + 2;

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

fn ints(s: &str) -> Vec<i64> {
    s.split_whitespace().map(|v| v.parse().unwrap()).collect()
}

/// 解析 `SLR0/SLR1` 行。
///
/// 插桩脚本的历史版本把标签写进了循环体的格式串（`"SLR0%s%d"`），于是**每个值**前面
/// 都带一个 `SLR0`；新版只在行首打一次。两种都兼容，免得依赖 dump 的生成时间。
fn slr_ints(rest: &str, label: &str) -> Vec<i64> {
    ints(&rest.replace(label, " "))
}

#[derive(Debug, Clone)]
enum Event {
    Flags { ch: usize },
    Pred { pred: [i32; 2], mid_only: bool },
    Frame { frame: usize, ch: usize, score: Vec<i64> },
    Slr { left: Vec<i64>, right: Vec<i64> },
}

fn parse_events(dump: &str) -> Vec<Event> {
    let mut out = Vec::new();
    // 正在拼装的帧：帧号 / 声道 / SCORE
    let mut cur: Option<(usize, usize, Vec<i64>)> = None;
    macro_rules! flush {
        () => {
            if let Some((frame, ch, score)) = cur.take() {
                out.push(Event::Frame { frame, ch, score });
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
            out.push(Event::Flags { ch: ints(rest)[0] as usize });
            continue;
        }
        if let Some(rest) = line.strip_prefix("SPRED ") {
            flush!();
            let v = ints(rest);
            out.push(Event::Pred { pred: [v[0] as i32, v[1] as i32], mid_only: v[2] == 1 });
            continue;
        }
        if let Some(rest) = line.strip_prefix("SLR0 ") {
            let left = slr_ints(rest, "SLR0");
            out.push(Event::Slr { left, right: Vec::new() });
            continue;
        }
        if let Some(rest) = line.strip_prefix("SLR1 ") {
            let right = slr_ints(rest, "SLR1");
            match out.last_mut() {
                Some(Event::Slr { right: r, .. }) => *r = right,
                other => panic!("SLR1 之前没有 SLR0：{other:?}"),
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("SCH ") {
            flush!();
            let v = ints(rest);
            cur = Some((v[0] as usize, v[1] as usize, Vec::new()));
            continue;
        }
        let Some((_, _, score)) = cur.as_mut() else { continue };
        if let Some(rest) = line.strip_prefix("SCORE ") {
            *score = ints(rest);
        }
    }
    flush!();
    out
}

/// SILK 的内部采样率（参考实现 `opus_decoder.c`：SILK-only 按带宽，混合固定 16 kHz）。
fn internal_fs_khz(toc: Toc) -> u32 {
    if toc.mode == OpusMode::Hybrid {
        return 16;
    }
    match toc.bandwidth {
        Bandwidth::Narrow => 8,
        Bandwidth::Medium => 12,
        _ => 16,
    }
}

#[derive(Default)]
struct Stats {
    packets: usize,
    frames: usize,
    stereo_frames: usize,
    samples: usize,
    max_abs: i32,
}

fn run_vector(label: &str, pkts: &[u8], dump: &str) -> Stats {
    let packets = parse_packets(pkts);
    let events = parse_events(dump);
    let mut ev = events.iter().peekable();

    let mut channels: [Option<SilkChannel>; 2] = [None, None];
    let mut n_ch_internal = 0usize;
    let mut prev_mid_only = false;
    let mut stereo = SilkStereoState::default();
    let mut st = Stats::default();
    let mut pulses_buf = vec![0i32; MAX_FRAME_LENGTH];
    let mut xq = vec![0i16; MAX_FRAME_LENGTH];
    // 与参考实现的 `samplesOut1_tmp` 完全同构：解码结果写在 [2..)，解混结果在 [1..)
    let mut mid = vec![0i16; TMP_LEN];
    let mut side = vec![0i16; TMP_LEN];

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
        if c > n_ch_internal {
            channels[1] = None;
        }
        let fs_khz = internal_fs_khz(toc);
        let payload_ms = toc.frame_size / 48;
        for slot in channels.iter_mut().take(c) {
            if slot.is_none() {
                *slot = Some(SilkChannel::new(fs_khz, payload_ms).unwrap());
            }
        }
        n_ch_internal = c;
        let payload = celt::frame_payload(pkt, toc, 0).unwrap();
        let mut dec = RangeDecoder::new(payload);

        for (ch, slot) in channels.iter_mut().take(c).enumerate() {
            let ch_state = slot.as_mut().unwrap();
            ch_state.set_fs(fs_khz, payload_ms).unwrap();
            ch_state.start_packet();
            ch_state.decode_packet_flags(&mut dec).unwrap();
            match next(&mut ev, "SFLAGS") {
                Event::Flags { ch: dch } => assert_eq!(dch, ch, "{label}: 包 {pkt_idx} SFLAGS 声道"),
                other => panic!("{label}: 包 {pkt_idx} 期望 SFLAGS，得到 {other:?}"),
            }
        }

        for frame in 0..channels[0].as_ref().unwrap().n_frames_per_packet {
            let mut mid_only = false;
            let mut pred = [0i32; 2];
            if c == 2 {
                pred = silk::stereo_decode_pred(&mut dec).unwrap();
                let side_vad = channels[1].as_ref().unwrap().vad_flags[frame];
                mid_only = if !side_vad {
                    silk::stereo_decode_mid_only(&mut dec).unwrap()
                } else {
                    false
                };
            }
            // 参考实现：上一帧只编了中间声道、这一帧有侧声道 → 侧声道状态整体复位
            if c == 2 && !mid_only && prev_mid_only {
                let s = channels[1].as_mut().unwrap();
                s.out_buf = [0; MAX_FRAME_LENGTH + 2 * 80];
                s.s_lpc_q14_buf = [0; MAX_LPC_ORDER];
                s.lag_prev = 100;
                s.last_gain_index = 10;
                s.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
                s.first_frame_after_reset = true;
            }
            let has_side = c == 2 && !mid_only;
            let frame_length = channels[0].as_ref().unwrap().frame_length;

            for ch in 0..c {
                if ch == 1 && !has_side {
                    continue;
                }
                let ch0_frames = channels[0].as_ref().unwrap().n_frames_decoded;
                let frame_index = ch0_frames as i32 - ch as i32;
                let cond = if frame_index <= 0 {
                    CODE_INDEPENDENTLY
                } else if ch > 0 && prev_mid_only {
                    CODE_INDEPENDENTLY_NO_LTP_SCALING
                } else {
                    CODE_CONDITIONALLY
                };
                let ch_state = channels[ch].as_mut().unwrap();
                ch_state
                    .decode_indices(&mut dec, frame, cond)
                    .unwrap_or_else(|e| panic!("{label}: 包 {pkt_idx} 帧 {frame} 声道 {ch} 索引：{e}"));
                let fl = ch_state.frame_length;
                ch_state
                    .decode_pulses(&mut dec, &mut pulses_buf)
                    .unwrap_or_else(|e| panic!("{label}: 包 {pkt_idx} 帧 {frame} 声道 {ch} 脉冲：{e}"));
                let mut control = SilkDecoderControl::default();
                silk_params::decode_parameters(ch_state, &mut control, cond).unwrap();
                ch_state.first_frame_after_reset = false;

                // ① 核心合成
                silk_synth::decode_core(ch_state, &mut control, &pulses_buf[..], &mut xq[..fl]);

                match next(&mut ev, "SCH") {
                    Event::Frame { frame: wf, ch: wch, score } => {
                        assert_eq!((wf, wch), (frame, ch), "{label}: 包 {pkt_idx} 帧/声道顺序");
                        assert_eq!(score.len(), fl, "{label}: 包 {pkt_idx} 帧 {frame} SCORE 长度");
                        let got: Vec<i64> = xq[..fl].iter().map(|&v| v as i64).collect();
                        if got != score {
                            let at = got.iter().zip(&score).position(|(a, b)| a != b).unwrap_or(0);
                            let d = (got[at] - score[at]).abs();
                            panic!(
                                "{label}: 包 {pkt_idx} 帧 {frame} 声道 {ch} 的 SCORE 不一致：\
                                 第一处差在样点 {at}（{}/{}，差 {}）",
                                got[at], score[at], d
                            );
                        }
                        st.samples += fl;
                        st.max_abs = st.max_abs.max(got.iter().map(|v| v.abs() as i32).max().unwrap_or(0));
                    }
                    other => panic!("{label}: 包 {pkt_idx} 期望 SCH，得到 {other:?}"),
                }

                // ② 帧尾收尾（推进 LTP 历史 + 舒适噪声状态），**再**把信号写进
                //    临时缓冲区 —— 参考实现里 `samplesOut1_tmp` 装的就是"收尾之后"的信号
                silk_synth::finish_frame(ch_state, &control, &mut xq[..fl], false);
                let buf = if ch == 0 { &mut mid } else { &mut side };
                buf[2..2 + fl].copy_from_slice(&xq[..fl]);
                ch_state.n_frames_decoded += 1;
            }

            if c == 2 {
                if !has_side {
                    // 参考实现：跳过的侧声道缓冲区被清零（长度 = 本帧长）
                    side[2..2 + frame_length].fill(0);
                }
                silk_synth::stereo_ms_to_lr(
                    &mut stereo,
                    &mut mid[..],
                    &mut side[..],
                    pred,
                    fs_khz,
                    frame_length,
                );
                match next(&mut ev, "SPRED") {
                    Event::Pred { pred: wp, mid_only: wm } => {
                        assert_eq!(pred, wp, "{label}: 包 {pkt_idx} 帧 {frame} 立体声预测");
                        assert_eq!(mid_only, wm, "{label}: 包 {pkt_idx} 帧 {frame} mid-only");
                    }
                    other => panic!("{label}: 包 {pkt_idx} 期望 SPRED，得到 {other:?}"),
                }
                match next(&mut ev, "SLR") {
                    Event::Slr { left, right } => {
                        let gl: Vec<i64> = mid[2..2 + frame_length].iter().map(|&v| v as i64).collect();
                        let gr: Vec<i64> = side[2..2 + frame_length].iter().map(|&v| v as i64).collect();
                        assert_eq!(gl, left, "{label}: 包 {pkt_idx} 帧 {frame} SLR0（左声道）");
                        assert_eq!(gr, right, "{label}: 包 {pkt_idx} 帧 {frame} SLR1（右声道）");
                    }
                    other => panic!("{label}: 包 {pkt_idx} 期望 SLR，得到 {other:?}"),
                }
                st.stereo_frames += 1;
            } else {
                // 单声道：参考实现只维护 sMid 的两个前导样点
                silk_synth::stereo_mid_only_copy(&mut stereo, &mut mid[..], frame_length);
            }
            prev_mid_only = mid_only;
            st.frames += 1;
        }
    }
    assert!(ev.next().is_none(), "{label}: 参考 dump 里还有没消费的事件");
    st
}

#[test]
#[ignore = "手工跑：需要 OPUS_TV_DIR（dump 由 /tmp/opus_rebuild_all.sh 重新生成）"]
fn silk_synthesis_matches_reference_on_official_test_vectors() {
    let dir = env::var("OPUS_TV_DIR").expect("要设 OPUS_TV_DIR");
    let wanted = [
        "testvector02",
        "testvector03",
        "testvector04",
        "testvector05",
        "testvector06",
        "testvector12",
    ];
    let (mut packets, mut frames, mut stereo_frames, mut samples, mut max_abs, mut vectors) =
        (0usize, 0usize, 0usize, 0usize, 0i32, 0usize);
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
            "{name}: 合成逐样本一致 {} 包 / {} 帧（其中 {} 个立体声帧）/ {} 个样点（最大幅度 {}）",
            st.packets, st.frames, st.stereo_frames, st.samples, st.max_abs
        );
        packets += st.packets;
        frames += st.frames;
        stereo_frames += st.stereo_frames;
        samples += st.samples;
        max_abs = max_abs.max(st.max_abs);
        vectors += 1;
    }
    assert!(vectors >= 6, "向量组数太少：{vectors}");
    assert!(stereo_frames > 0, "一条立体声帧都没验到，参数可能传错了");
    println!(
        "合计：{vectors} 组、{packets} 包、{frames} 帧（{stereo_frames} 个立体声帧）、\
         {samples} 个样点逐样本一致（最大幅度 {max_abs}）"
    );
}
