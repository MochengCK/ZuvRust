//! 丢包隐藏（PLC）对拍：拿参考实现在**同样的丢包图样**下解出来的 PCM 逐样本比。
//!
//! 参考侧由一个独立小工具产出（`/tmp/opus_plc_probe.c`，直接调参考实现的
//! `opus_decode(dec, NULL, 0, ...)`）：
//!
//! ```sh
//! /tmp/opus_plc_probe <pkts.bin> <out.pcm> <丢包下标或区间>
//! # 用法示例：tv01 第 10~20 包全丢
//! /tmp/opus_plc_probe testvector01.pkts.bin out.pcm 10-20
//! ```
//!
//! 清单文件 `plc_cases.txt` 由 `/tmp/opus_plc_gen.sh` 生成，两边读同一份，
//! 所以"生成"和"校验"永远不会漂。目前 24 组：
//!
//! - **CELT-only**（tv01/07/11）：空载荷、单帧丢、连续丢 2/4/6/11 帧；
//!   最大差 1~2 个量化步长（CELT 是浮点，不可能逐位相同）。
//! - **SILK-only**（tv02/03/04）：单帧丢、连丢 2~6 帧；**逐样本完全相同（最大差 0）**
//!   —— SILK 是纯整数实现，所以这里能做到逐位一致，这也是最强的验收形式。
//!
//! 跑法：
//! ```sh
//! sh /tmp/opus_plc_gen.sh          # 生成参考 PCM（约 1 分钟）
//! OPUS_TV_DIR=/tmp/opus_tv_pkts cargo test -p me-decode --test opus_plc_ref -- --ignored --nocapture
//! ```

use std::env;

use me_decode::audio::opus::opus_decoder::OpusDecoder;

fn parse_packets(bytes: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 4 <= bytes.len() {
        let n = u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
            as usize;
        pos += 4;
        if pos + n > bytes.len() {
            break;
        }
        out.push(&bytes[pos..pos + n]);
        pos += n;
    }
    out
}

fn parse_pcm(path: &str) -> Vec<i16> {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("读不了 {path}: {e}"));
    raw.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn parse_idx(path: &str) -> Vec<usize> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("读不了 {path}: {e}"))
        .lines()
        .filter_map(|l| l.split_whitespace().nth(3).and_then(|v| v.parse::<usize>().ok()))
        .collect()
}

/// `10-13,20` → 下标集合。
fn parse_spec(spec: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for tok in spec.split(',') {
        if let Some((a, b)) = tok.split_once('-') {
            let a: usize = a.parse().unwrap();
            let b: usize = b.parse().unwrap();
            out.extend(a..=b);
        } else {
            out.push(tok.parse().unwrap());
        }
    }
    out
}

#[derive(Default)]
struct Stats {
    packets: usize,
    samples: usize,
    max_diff: i32,
    over_two: usize,
    exact: usize,
    first_bad: Option<String>,
}

fn run_case(dir: &str, name: &str, spec: &str) -> Stats {
    let lost = parse_spec(spec);
    let raw = std::fs::read(format!("{dir}/{name}.pkts.bin")).unwrap();
    let packets = parse_packets(&raw);
    let want = parse_pcm(&format!("{dir}/plc_{name}_{spec}.pcm"));
    let want_len = parse_idx(&format!("{dir}/plc_{name}_{spec}.idx"));
    assert_eq!(
        want_len.len(),
        packets.len(),
        "{name} [{spec}]: 参考侧的包数对不上"
    );

    let mut dec = OpusDecoder::new(2).unwrap();
    let mut out = Vec::new();
    let mut st = Stats::default();
    let mut cursor = 0usize; // 每声道

    for (i, pkt) in packets.iter().enumerate() {
        let n = if lost.contains(&i) {
            dec.decode_lost(&mut out).unwrap()
        } else {
            dec.decode_packet(pkt, &mut out).unwrap()
        };
        assert_eq!(n, want_len[i], "{name} [{spec}]: 包 {i} 的样点数对不上");
        let mine = &out[..n * 2];
        let theirs = &want[cursor * 2..(cursor + n) * 2];
        for (j, (a, b)) in mine.iter().zip(theirs).enumerate() {
            let d = (*a as i32 - *b as i32).abs();
            if d > st.max_diff {
                st.max_diff = d;
                if st.first_bad.is_none() && d > 2 {
                    st.first_bad = Some(format!(
                        "包 {i} 第 {} 个样点（声道 {}）：我们 {a}，参考 {b}",
                        j / 2,
                        j % 2
                    ));
                }
            }
            if d > 2 {
                st.over_two += 1;
            }
            if d == 0 {
                st.exact += 1;
            }
            st.samples += 1;
        }
        st.packets += 1;
        cursor += n;
    }
    assert_eq!(cursor * 2, want.len(), "{name} [{spec}]: 总样点数对不上");
    st
}

#[test]
#[ignore = "手工跑：需要 OPUS_TV_DIR（先跑 /tmp/opus_plc_gen.sh）"]
fn packet_loss_concealment_matches_the_reference() {
    let dir = env::var("OPUS_TV_DIR").expect("要设 OPUS_TV_DIR");
    let cases = std::fs::read_to_string(format!("{dir}/plc_cases.txt"))
        .expect("缺少 plc_cases.txt（先跑 /tmp/opus_plc_gen.sh）");

    let (mut total, mut total_samples, mut worst) = (0usize, 0usize, 0i32);
    for line in cases.lines() {
        let mut it = line.split_whitespace();
        let (Some(name), Some(spec)) = (it.next(), it.next()) else {
            continue;
        };
        let st = run_case(&dir, name, spec);
        let exact_pct = 100.0 * st.exact as f64 / st.samples.max(1) as f64;
        println!(
            "{name} [{spec}]: {} 包 / {} 样本；逐样本相同 {exact_pct:.4}%；最大差 {} 步",
            st.packets, st.samples, st.max_diff
        );
        if let Some(bad) = &st.first_bad {
            println!("  首处偏差（>2 步）：{bad}");
        }
        total += 1;
        total_samples += st.samples;
        worst = worst.max(st.max_diff);
        assert!(st.first_bad.is_none(), "{name} [{spec}] 与参考不一致");
    }
    println!("合计 {total} 组 / {total_samples} 样本，最大差 {worst} 个量化步长");
    assert!(total >= 24, "用例太少：{total}");
    assert!(worst <= 2, "最大差 {worst} 个量化步长，超出预期");
}
