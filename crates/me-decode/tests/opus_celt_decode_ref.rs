//! 手工跑：**顺序整链解码**（前端 → 分配 → 形状 → 合成）与参考实现对拍。
//!
//! 前面几层的测试都是**逐帧播种**的：每帧从 dump 里恢复帧前状态，好坏都是那一帧的事。
//! 这一条不一样 —— 用 [`CeltDecoder`] 从第 0 帧顺序解到底，**状态自然传承**。
//! 它能抓到播种式测试在结构上抓不到的东西：
//!
//! - 单声道帧的能量合并（立体声流里混单声道帧时，参考实现会先把两声道能量取 max）；
//! - 静音帧收尾（能量置 -28、历史能量的前推语义）；
//! - LCG 种子的取回（每帧末 `st->rng = dec->rng`，反塌陷/折叠都吃它）；
//! - 合成状态（重叠缓冲 / 后置滤波 / 去加重）的跨帧接力。
//!
//! 两项对拍：
//!
//! 1. **PCM 逐样本**（整条流；合成层那条测试只比了 dump 覆盖的前 400 帧）；
//! 2. **RFC §6.1 的"解码器最终区间状态"**：官方向量的 `.bit` 里每个包带一个
//!    `enc_final_range`（`/tmp/opus_tv_pkts/*.range.txt`），解码器解完该包之后的
//!    区间状态必须与它一致 —— 熵层里任何一处解错，这里都会对不上。
//!
//! 跑法（真实素材；素材是 96 kbps 音乐，整条都是 CELT）：
//!
//! ```text
//! OPUS_PKTS=/tmp/opus_pkts.bin OPUS_REF_PCM=/tmp/opus_dbg.pcm \
//!   cargo test -p me-decode --test opus_celt_decode_ref -- --ignored --nocapture
//! ```
//!
//! 跑法（官方向量）：加 `OPUS_TV_DIR=/tmp/opus_tv_pkts`。只跑**纯 CELT-only** 的
//! 01 / 07 / 11：02/03/04 全是 SILK，05/06/12 有混合包，08/09/10 里混着 SILK 包
//! （参考实现在模式切换处会 `CELT_RESET_STATE`，本引擎还没接 SILK，接不上）。
//! 整条流有点慢，加 `OPUS_DECODE_FRAMES=<n>` 可以只解前 n 帧（快速回归用）。

use std::env;

use me_decode::audio::opus::celt::{self, CeltDecoder, OpusMode, Toc};

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

fn read_pcm(path: &str) -> Vec<i16> {
    std::fs::read(path)
        .expect("读参考 PCM")
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

#[derive(Default)]
struct Stats {
    frames: usize,
    samples: usize,
    max_diff: i32,
    /// 差 ≥2 的样本数
    over_one: usize,
    first_bad: Option<String>,
    /// 区间状态核对过的包数 / 错配的包（RFC §6.1）
    range_checked: usize,
    range_bad: Vec<String>,
}

/// 顺序解一条流，逐帧与参考 PCM 比；给了 `ranges` 就顺带核对每包的最终区间状态。
fn decode_stream(
    label: &str,
    packets: &[&[u8]],
    ref_pcm: &[i16],
    ranges: Option<&[u32]>,
    max_frames: usize,
) -> Stats {
    let mut dec = CeltDecoder::new();
    let mut st = Stats::default();
    let mut cursor = 0usize; // 参考 PCM 里的样本游标（每声道）
    let mut pcm = vec![0i16; 2 * 960];

    for (pkt_idx, pkt) in packets.iter().enumerate() {
        let toc = Toc::parse(pkt).unwrap_or_else(|e| panic!("{label}: 第 {pkt_idx} 包 TOC：{e}"));
        let pkt_samples = toc.frame_size as usize * toc.frames as usize;
        if toc.mode != OpusMode::CeltOnly {
            // SILK / 混合还没接：PCM 位置照样往前走，但不核对
            cursor += pkt_samples;
            continue;
        }
        let layout = celt::frame_layout(pkt).unwrap();
        assert_eq!(layout.len(), toc.frames as usize);
        for (off, len) in layout {
            if st.frames >= max_frames {
                return st;
            }
            let n = toc.frame_size as usize;
            assert!(cursor + n <= ref_pcm.len(), "{label}: 参考 PCM 比流短");
            let out = &mut pcm[..2 * n];
            dec.decode_frame(&pkt[off..off + len], toc, out).unwrap_or_else(|e| {
                panic!("{label}: 第 {pkt_idx} 包第 {} 帧解码失败：{e}", st.frames)
            });
            let base = cursor * 2;
            for (j, &got) in out.iter().enumerate() {
                let want = ref_pcm[base + j] as i32;
                let d = (got as i32 - want).abs();
                st.max_diff = st.max_diff.max(d);
                if d >= 2 {
                    st.over_one += 1;
                    if st.first_bad.is_none() {
                        st.first_bad = Some(format!(
                            "{label} 第 {} 帧（包 {pkt_idx}）声道 {} 第 {} 个样本：我们 {got} vs 参考 {want}（差 {d}）",
                            st.frames,
                            j % 2,
                            j / 2,
                        ));
                    }
                }
            }
            st.samples += out.len();
            st.frames += 1;
            cursor += n;
        }
        if let Some(want_rng) = ranges {
            // 参考实现每帧末尾把 `st->rangeFinal` 覆盖成该帧的区间状态，
            // 所以"包的最终区间"就是最后一帧解完时的区间状态。
            let got = dec.state.rng;
            let want = want_rng[pkt_idx];
            st.range_checked += 1;
            if got != want {
                st.range_bad.push(format!(
                    "{label} 包 {pkt_idx}：区间状态 {got} vs 编码器 {want}（解到第 {} 帧）",
                    st.frames
                ));
            }
        }
    }
    st
}

fn report(label: &str, st: &Stats) {
    if let Some(bad) = &st.first_bad {
        println!("  {bad}");
    }
    let frac = st.over_one as f64 / st.samples.max(1) as f64;
    println!(
        "{label}: 顺序解 {} 帧、{} 样本；最大差 {} 个量化步长；差 ≥2 占比 {:.2e}；\
         区间状态核对 {} 包、错 {}",
        st.frames,
        st.samples,
        st.max_diff,
        frac,
        st.range_checked,
        st.range_bad.len()
    );
    for b in st.range_bad.iter().take(5) {
        println!("  {b}");
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[test]
#[ignore = "手工跑：需要 OPUS_PKTS / OPUS_REF_PCM（见文件顶部说明）"]
fn sequential_decode_matches_reference_on_a_real_opus_file() {
    let pkts = std::fs::read(env::var("OPUS_PKTS").expect("要设 OPUS_PKTS")).expect("读包");
    let pcm_path = env::var("OPUS_REF_PCM").unwrap_or_else(|_| "/tmp/opus_dbg.pcm".into());
    let ref_pcm = read_pcm(&pcm_path);
    let packets = parse_packets(&pkts);
    let max_frames = env_usize("OPUS_DECODE_FRAMES", usize::MAX);

    let st = decode_stream("真实 .opus", &packets, &ref_pcm, None, max_frames);
    report("真实 .opus", &st);
    assert!(st.first_bad.is_none(), "PCM 与参考不一致（见上面第一处）");
    assert!(st.max_diff <= 2, "最大差 {} 个量化步长", st.max_diff);
    assert!(st.frames > 10_000, "核对帧数太少：{}（设 OPUS_DECODE_FRAMES 了？）", st.frames);
}

#[test]
#[ignore = "手工跑：需要 OPUS_TV_DIR（官方向量的 dump / PCM / range 由 /tmp/opus_regen_tv_dumps.sh 生成）"]
fn sequential_decode_matches_reference_on_official_test_vectors() {
    let dir = env::var("OPUS_TV_DIR").expect("要设 OPUS_TV_DIR");
    let max_frames = env_usize("OPUS_DECODE_FRAMES", usize::MAX);
    // 纯 CELT-only 的向量（见文件顶部说明）
    let wanted = ["testvector01", "testvector07", "testvector11"];
    let (mut frames, mut max_diff, mut range_checked, mut vectors) = (0usize, 0i32, 0usize, 0usize);
    for name in wanted {
        let pkts_path = format!("{dir}/{name}.pkts.bin");
        if !std::path::Path::new(&pkts_path).exists() {
            println!("{name}: 素材不在，跳过");
            continue;
        }
        let pkts = std::fs::read(&pkts_path).expect("读包");
        let ref_pcm = read_pcm(&format!("{dir}/{name}.pcm"));
        let ranges: Vec<u32> = std::fs::read_to_string(format!("{dir}/{name}.range.txt"))
            .expect("读 range.txt")
            .lines()
            .map(|l| l.trim().parse().unwrap())
            .collect();
        let packets = parse_packets(&pkts);
        assert_eq!(ranges.len(), packets.len(), "{name}: range.txt 与包数不符");

        let st = decode_stream(name, &packets, &ref_pcm, Some(&ranges), max_frames);
        report(name, &st);
        assert!(st.first_bad.is_none(), "{name}: PCM 与参考不一致（见上面第一处）");
        assert!(st.max_diff <= 2, "{name}: 最大差 {} 个量化步长", st.max_diff);
        assert!(
            st.range_bad.is_empty(),
            "{name}: {} 个包的最终区间状态与编码器不一致（RFC §6.1）",
            st.range_bad.len()
        );
        frames += st.frames;
        range_checked += st.range_checked;
        max_diff = max_diff.max(st.max_diff);
        vectors += 1;
    }
    assert!(vectors >= 3, "纯 CELT-only 的向量不该少于 3 组");
    println!(
        "官方向量合计：{vectors} 组、{frames} 帧；最大差 {max_diff} 个量化步长；\
         RFC §6.1 区间状态核对 {range_checked} 包全对"
    );
}
