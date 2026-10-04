//! 手工跑：SILK **整链**（熵层 → 参数 → 合成 → 立体声解混 → 重采样 → 交织）与参考实现
//! 对拍 —— 比的是参考实现 `silk_Decode` 的**最终输出**（API 采样率、已交织）。
//!
//! 这一条与 `opus_silk_synth_ref.rs` 的分工：
//!
//! | 测试 | 对拍对象 | 覆盖到哪一层 |
//! |---|---|---|
//! | `opus_silk_synth_ref.rs` | `SCORE` / `SLR0/SLR1` | 核心合成 + 立体声解混（**内部**采样率）|
//! | 本文件 | `SOUT` | 上面全部 + **重采样器 + 交织顺序** |
//!
//! `SOUT` 的期望值来自参考实现 `silk_Decode` 返回前的 `samplesOut`（见
//! `/tmp/opus_patch_silk.py` 的第 ⑧ 段插桩）。它是纯 SILK-only 流的"应交货物"，
//! 所以拿它可以在**不接 CELT** 的前提下把 SILK 这条链验到底。
//!
//! 跑法：
//!
//! ```text
//! OPUS_TV_DIR=/tmp/opus_tv_pkts \
//!   cargo test -p me-decode --test opus_silk_decode_ref -- --ignored --nocapture
//! ```

use std::env;

use me_decode::audio::opus::celt::{self, OpusMode, Toc};
use me_decode::audio::opus::entropy::RangeDecoder;
use me_decode::audio::opus::silk_decoder::{frames_in_packet, SilkDecoder};

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

/// dump 里的 `SOUT` 行（按出现顺序）。
fn parse_sout(dump: &str) -> Vec<Vec<i64>> {
    dump.lines()
        .filter_map(|l| l.strip_prefix("SOUT "))
        .map(|rest| {
            rest.split_whitespace()
                .map(|v| v.parse::<i64>().expect("SOUT 里出现非整数"))
                .collect()
        })
        .collect()
}

#[test]
#[ignore = "手工跑：需要 OPUS_TV_DIR（dump 由 /tmp/opus_rebuild_all.sh 重新生成）"]
fn silk_full_chain_matches_reference_output() {
    let dir = env::var("OPUS_TV_DIR").expect("要设 OPUS_TV_DIR");
    let wanted = ["testvector02", "testvector03", "testvector04"];
    let (mut packets_total, mut frames_total, mut samples_total, mut vectors) = (0, 0, 0, 0);
    for name in wanted {
        let pkts_path = format!("{dir}/{name}.pkts.bin");
        if !std::path::Path::new(&pkts_path).exists() {
            println!("{name}: 素材不在，跳过");
            continue;
        }
        let pkts = std::fs::read(&pkts_path).expect("读包");
        let dump = std::fs::read_to_string(format!("{dir}/{name}.dump.txt")).expect("读 dump");
        let expected = parse_sout(&dump);
        assert!(!expected.is_empty(), "{name}: dump 里没有 SOUT 行（插桩没重生成？）");
        let packets = parse_packets(&pkts);
        let mut silk = SilkDecoder::new(48000, 2).unwrap();
        let mut idx = 0usize;
        let mut packets_used = 0usize;
        for (pkt_idx, pkt) in packets.iter().enumerate() {
            let toc = Toc::parse(pkt).unwrap_or_else(|e| panic!("{name}: 包 {pkt_idx} TOC：{e}"));
            if toc.mode == OpusMode::CeltOnly {
                continue;
            }
            packets_used += 1;
            let payload = celt::frame_payload(pkt, toc, 0).unwrap();
            let mut dec = RangeDecoder::new(payload);
            // 注意：是 SILK 子帧数（60 ms 包 = 3 个），不是 `toc.frames`（Opus 帧数）
            let n_silk_frames = frames_in_packet(toc.frame_size / 48).unwrap();
            for f in 0..n_silk_frames {
                let n = silk
                    .decode_frame(toc, &mut dec, f == 0)
                    .unwrap_or_else(|e| panic!("{name}: 包 {pkt_idx} 帧 {f}：{e}"));
                let want = expected
                    .get(idx)
                    .unwrap_or_else(|| panic!("{name}: 参考 dump 的 SOUT 不够（第 {idx} 条）"));
                assert_eq!(want.len(), n * 2, "{name}: 包 {pkt_idx} 帧 {f} 的 SOUT 长度");
                let got: Vec<i64> = silk.out().iter().map(|&v| v as i64).collect();
                if got != *want {
                    let at = got.iter().zip(want).position(|(a, b)| a != b).unwrap_or(0);
                    panic!(
                        "{name}: 包 {pkt_idx} 帧 {f} 的输出不一致：第一处差在交织下标 {at}\
                         （{} / {}）",
                        got[at], want[at]
                    );
                }
                idx += 1;
                frames_total += 1;
                samples_total += n;
            }
        }
        assert_eq!(idx, expected.len(), "{name}: 参考 dump 里还有没消费的 SOUT");
        println!(
            "{name}: 整链输出逐样本一致 {packets_used} 包 / {idx} 帧 / {samples_total} 个样点"
        );
        packets_total += packets_used;
        vectors += 1;
    }
    assert!(vectors >= 3, "向量组数太少：{vectors}");
    println!(
        "合计：{vectors} 组、{packets_total} 包、{frames_total} 帧、{samples_total} 个样点\
         （含重采样与交织）逐样本一致"
    );
}
