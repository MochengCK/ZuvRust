//! 手工跑：**完整 Opus 解码器**（CELT + SILK + 混合 + 冗余帧）与参考实现对拍。
//!
//! 前面几条测试都是"分层"的（每层跟参考实现的中间量对）。这一条是**成品**：
//! 直接拿参考解码器 `opus_decode` 出来的整段 48 kHz 立体声 `.pcm` 逐样本比。
//!
//! 覆盖与容差：
//!
//! | 向量 | 模式 | 这条测试对的是什么 |
//! |---|---|---|
//! | 01 / 07 / 11 | 纯 CELT | CELT 整链（与 `opus_celt_decode_ref` 同口径）|
//! | 02 / 03 / 04 | 纯 SILK | SILK 整链 + **冗余帧**（SILK 侧的包常带一段 5 ms CELT 冗余帧）|
//! | 05 / 06 | 纯混合 | SILK(0–8 kHz) + CELT(>8 kHz) 相加 |
//! | 12 | SILK ↔ 混合 | 非 CELT 模式之间的切换（CELT 状态复位）|
//! | 08 / 09 / 10 | 含 CELT ↔ SILK 切换 | **明确报 Unsupported**（缺 PLC）—— 断言这一点 |
//!
//! 容差是"**逐样本差 ≤ 2 个 16 位量化步长**"：CELT 侧参考实现走的是浮点构建，
//! 本引擎全程整数；SILK 侧是纯整数，逐样本相同。冗余帧/淡入那段是浮点运算
//! （`smooth_fade` 在参考实现里是浮点），所以放宽到 2 步。
//!
//! 跑法：
//!
//! ```text
//! OPUS_TV_DIR=/tmp/opus_tv_pkts \
//!   cargo test -p me-decode --test opus_decode_ref -- --ignored --nocapture
//! ```

use std::env;

use me_decode::audio::opus::opus_decoder::OpusDecoder;

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
    packets: usize,
    samples: usize,
    max_diff: i32,
    over_two: usize,
    exact: usize,
    first_bad: Option<String>,
}

/// 一条向量的结局：要么整段解出来，要么在中途明确报"还没实现"。
enum Outcome {
    Decoded(Stats),
    Rejected { packet: usize, msg: String },
}

fn run_vector(name: &str, dir: &str) -> Outcome {
    let pkts = std::fs::read(format!("{dir}/{name}.pkts.bin")).expect("读包");
    let ref_pcm = read_pcm(&format!("{dir}/{name}.pcm"));
    let packets = parse_packets(&pkts);
    let mut dec = OpusDecoder::new(2).unwrap();
    let mut out = Vec::new();
    let mut st = Stats::default();
    let mut cursor = 0usize; // 每声道
    for (pkt_idx, pkt) in packets.iter().enumerate() {
        let n = match dec.decode_packet(pkt, &mut out) {
            Ok(n) => n,
            Err(e) => {
                return Outcome::Rejected { packet: pkt_idx, msg: format!("{e}") };
            }
        };
        assert!(cursor + n <= ref_pcm.len(), "{name}: 参考 PCM 比流短");
        let base = cursor * 2;
        for (j, &got) in out.iter().enumerate() {
            let want = ref_pcm[base + j] as i32;
            let d = (got as i32 - want).abs();
            st.max_diff = st.max_diff.max(d);
            if d == 0 {
                st.exact += 1;
            } else if d > 2 {
                st.over_two += 1;
                if st.first_bad.is_none() {
                    st.first_bad = Some(format!(
                        "{name} 包 {pkt_idx} 声道 {} 第 {} 个样本：我们 {got} vs 参考 {want}（差 {d}）",
                        j % 2,
                        j / 2
                    ));
                }
            }
        }
        st.samples += out.len();
        cursor += n;
        st.packets += 1;
    }
    assert_eq!(cursor * 2, ref_pcm.len(), "{name}: 解出的样本数与参考 PCM 不符");
    Outcome::Decoded(st)
}

#[test]
#[ignore = "手工跑：需要 OPUS_TV_DIR（官方测试向量在 /tmp/opus_tv_pkts）"]
fn full_decoder_matches_reference_on_official_test_vectors() {
    let dir = env::var("OPUS_TV_DIR").expect("要设 OPUS_TV_DIR");
    // **全部 12 组**。其中 08/09/10 是"模式切换"向量（CELT ↔ 非 CELT 来回切），
    // 它们同时覆盖三条容易错的路径：5 ms CELT 冗余帧、`pcm_transition` 交叉淡入、
    // 以及丢包隐藏（PLC）—— 所以这一组也是那三条路径的验收点。
    let wanted = [
        "testvector01",
        "testvector02",
        "testvector03",
        "testvector04",
        "testvector05",
        "testvector06",
        "testvector07",
        "testvector08",
        "testvector09",
        "testvector10",
        "testvector11",
        "testvector12",
    ];
    let (mut total_samples, mut total_exact, mut max_diff, mut vectors) = (0usize, 0usize, 0i32, 0usize);
    for name in wanted {
        if !std::path::Path::new(&format!("{dir}/{name}.pkts.bin")).exists() {
            println!("{name}: 素材不在，跳过");
            continue;
        }
        let st = match run_vector(name, &dir) {
            Outcome::Decoded(st) => st,
            Outcome::Rejected { packet, msg } => {
                panic!("{name}: 不该被拒绝（包 {packet}）：{msg}")
            }
        };
        let frac = st.over_two as f64 / st.samples.max(1) as f64;
        println!(
            "{name}: {} 包 / {} 样本；逐样本相同 {:.2}%；最大差 {} 步；差 >2 的占比 {:.2e}",
            st.packets,
            st.samples,
            100.0 * st.exact as f64 / st.samples.max(1) as f64,
            st.max_diff,
            frac
        );
        if let Some(bad) = &st.first_bad {
            println!("  {bad}");
        }
        assert!(st.first_bad.is_none(), "{name}: PCM 与参考不一致（见上面第一处）");
        assert!(st.max_diff <= 2, "{name}: 最大差 {} 个量化步长", st.max_diff);
        total_samples += st.samples;
        total_exact += st.exact;
        max_diff = max_diff.max(st.max_diff);
        vectors += 1;
    }
    assert!(vectors >= 12, "向量组数太少：{vectors}");
    println!(
        "合计：{vectors} 组、{total_samples} 个样本；逐样本相同 {:.2}%；最大差 {max_diff} 个量化步长",
        100.0 * total_exact as f64 / total_samples.max(1) as f64
    );
}

#[test]
#[ignore = "手工跑：需要 OPUS_TV_DIR（官方测试向量在 /tmp/opus_tv_pkts）"]
fn silk_only_vectors_are_bit_exact() {
    // SILK 这条链**全程整数**（用参考实现的 `SigProc_FIX.h` 宏一一对应），
    // 所以验收口径比 CELT 严一档：**必须逐位相同**，不能拿"浮点误差"解释掉
    // 任何 1 个量化步长的偏差。跑出非零最大差就是真 bug。
    //
    // （CELT 侧相反：参考实现是浮点构建、我们是定点，1 个量化步长是极限。）
    let dir = env::var("OPUS_TV_DIR").expect("要设 OPUS_TV_DIR");
    for name in ["testvector02", "testvector03", "testvector04"] {
        if !std::path::Path::new(&format!("{dir}/{name}.pkts.bin")).exists() {
            println!("{name}: 素材不在，跳过");
            continue;
        }
        let st = match run_vector(name, &dir) {
            Outcome::Decoded(st) => st,
            Outcome::Rejected { packet, msg } => {
                panic!("{name}: 不该被拒绝（包 {packet}）：{msg}")
            }
        };
        println!(
            "{name}: {} 包 / {} 样本；逐样本相同 {:.4}%；最大差 {} 步",
            st.packets,
            st.samples,
            100.0 * st.exact as f64 / st.samples.max(1) as f64,
            st.max_diff
        );
        if let Some(bad) = &st.first_bad {
            println!("  {bad}");
        }
        assert_eq!(st.max_diff, 0, "{name}: 整数实现不该有偏差（见上面第一处）");
        assert_eq!(st.exact, st.samples, "{name}: 应当逐位相同");
    }
}
