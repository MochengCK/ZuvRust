//! 手工跑：把**合成段**（IMDCT / 重叠相加 / 后置滤波 / 去加重）与参考实现的 PCM 对拍。
//!
//! 前面几层（前端/分配/形状）都是**逐帧播种**、逐帧核对的；合成段不一样 —— 它有三样
//! 跨帧记忆（重叠加窗、后置滤波、去加重），所以这里**顺序跑**：从流的第 0 帧开始，
//! 状态自然传承，只把每帧的输入从 dump 里播种：
//!
//! ```text
//! X0/X1      反塌陷之后的归一化频谱      ← 播种（也要过一遍 denormalise）
//! BAND0/BAND1 log2Amp 之后的频带能量     ← 播种（静音帧在参考实现里会被清 0，这里照做）
//! FRAME      帧参数（transient / silence / 后置滤波 / 起止带）
//! ```
//!
//! 逐帧播种会把状态机的问题盖住（比如重叠缓冲少存了一段、后置滤波的 old/current
//! 推进错了 —— 这些错误在"每帧重置"的跑法下根本不会显形），所以这一层必须顺序跑。
//!
//! 比对口径：**逐样本 PCM（i16）**。这是有损编解码器的浮点路径，不可能逐位一致
//! （参考实现自己的 float/fixed 两个构建也做不到），所以看两个量：
//!
//! - `max`：整段最大差（几个量化步长以内）；
//! - `frac`：差 ≥ 2 LSB 的样本占比（单看最大值容易被一两个落点带偏 —— 一个样本
//!   恰好跨过取整边界就差 1，差 2 以上才说明信号路径有系统性偏差）。
//!
//! dump 里 `X0/X1` 只覆盖**前 400 帧**（`opus_patch_dbg_v2.py` 里的 `X_FRAMES`），
//! 所以对拍帧数就是 400；要扩就把插桩脚本改了重跑 `opus_rebuild_all.sh`。
//!
//! 跑法（真实素材）：
//!
//! ```text
//! OPUS_PKTS=/tmp/opus_pkts.bin OPUS_CELT_DUMP=/tmp/opus_dbg_frames.txt \
//! OPUS_REF_PCM=/tmp/opus_dbg.pcm \
//!   cargo test -p me-decode --test opus_celt_synth_ref -- --ignored --nocapture
//! ```
//!
//! 跑法（官方向量，各自一份 `.pcm`）：加 `OPUS_TV_DIR=/tmp/opus_tv_pkts`。
//! 只跑**纯 CELT-only** 的向量（01 / 07 / 11）：02/03/04 是 SILK-only，05/06/12 有
//! 混合包，08/09/10 里混着 SILK 包 —— 那些 CELT 状态在参考实现里会被 `RESET_STATE`
//! 打断，本引擎还没接 SILK，顺序跑会错位（等第 9 步做完再纳入）。

use std::collections::HashMap;
use std::env;

use me_decode::audio::opus::bands;
use me_decode::audio::opus::celt::{PostFilter, Toc, NB_EBANDS};
use me_decode::audio::opus::mdct::Mdct;
use me_decode::audio::opus::synth::{synth_frame, SynthInput, SynthState};

/// 参考实现 dump 出来的一帧（只留合成段用得上的）。
#[derive(Clone)]
struct Frame {
    pkt: usize,
    /// 这一帧是包里第几帧（PCM 偏移要按它累加）。
    frame_in_pkt: usize,
    /// `X0/X1`：反塌陷之后的归一化频谱。
    x: HashMap<String, Vec<f32>>,
    /// `BAND0/BAND1`：log2Amp 之后的频带能量（静音帧这里还没被清 0）。
    band: HashMap<String, Vec<f32>>,
    silence: bool,
    transient: bool,
    postfilter: PostFilter,
    /// 参考实现里的 `st->start` / `st->end`（CELT-only 是 0 / 带宽结束带）。
    start: usize,
    end: usize,
}

impl Default for Frame {
    fn default() -> Self {
        Self {
            pkt: 0,
            frame_in_pkt: 0,
            x: HashMap::new(),
            band: HashMap::new(),
            silence: false,
            transient: false,
            postfilter: PostFilter { pitch: 0, gain: 0.0, tapset: 0 },
            start: 0,
            end: 21,
        }
    }
}

fn floats(line: &str, prefix: &str) -> Vec<f32> {
    line.strip_prefix(prefix)
        .unwrap_or_else(|| panic!("期望以 {prefix} 开头：{line}"))
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}

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

fn parse_dump(dump: &str) -> Vec<Frame> {
    let mut frames: Vec<Frame> = Vec::new();
    let mut cur = Frame::default();
    let mut started = false;
    let mut cur_pkt = 0usize;
    let mut frame_in_pkt = 0usize;
    for line in dump.lines() {
        if let Some(rest) = line.strip_prefix("PKT ") {
            cur_pkt = rest.split_whitespace().next().unwrap().parse().unwrap();
            frame_in_pkt = 0;
            continue;
        }
        if let Some(rest) = line.strip_prefix("FRAME ") {
            if started {
                frames.push(std::mem::take(&mut cur));
            }
            started = true;
            cur = Frame { pkt: cur_pkt, frame_in_pkt, ..Default::default() };
            let mut kv = HashMap::new();
            for pair in rest.split_whitespace() {
                if let Some((k, v)) = pair.split_once('=') {
                    kv.insert(k, v);
                }
            }
            let num = |k: &str, d: i64| kv.get(k).and_then(|v| v.parse::<i64>().ok()).unwrap_or(d);
            cur.silence = num("silence", 0) == 1;
            cur.transient = num("trans", 0) == 1;
            let qg = num("qg", -1);
            cur.postfilter = PostFilter {
                pitch: num("pitch", 0) as i32,
                gain: 0.09375 * (qg + 1) as f32,
                tapset: num("tapset", 0) as u32,
            };
            cur.start = num("start", 0) as usize;
            cur.end = num("end", 21) as usize;
            frame_in_pkt += 1;
            continue;
        }
        for prefix in ["BAND", "X"] {
            if let Some(c) = line.as_bytes().get(prefix.len()) {
                if line.starts_with(prefix) && c.is_ascii_digit() {
                    let key = format!("{prefix}{}", *c as char);
                    let vals = floats(line, &format!("{key} "));
                    if prefix == "BAND" {
                        cur.band.insert(key, vals);
                    } else {
                        cur.x.insert(key, vals);
                    }
                    break;
                }
            }
        }
    }
    if started {
        frames.push(cur);
    }
    frames
}

#[derive(Default)]
struct Stats {
    frames: usize,
    samples: usize,
    /// 最大差（量化步长）
    max_diff: i32,
    /// 差 ≥ 1 的样本数（取整边界翻面的落点）
    differ: usize,
    /// 差 ≥ 2 的样本数
    over_one: usize,
    /// 第一处超过阈值的样本（报错用）
    first_bad: Option<String>,
    /// 覆盖率：瞬态（短块）/ 静音 / 后置滤波开 的帧数
    transient: usize,
    silence: usize,
    postfilter_on: usize,
}

/// 顺序跑一条流的前 `limit` 帧，逐样本与参考 PCM 比。
///
/// `ref_pcm` 是参考实现解出来的 i16（交错），偏移按"每个包解出多少样本"累加。
fn run_stream(label: &str, packets: &[&[u8]], frames: &[Frame], ref_pcm: &[i16], limit: usize) -> Stats {
    let mdct = Mdct::new_48k();
    let mut state = SynthState::default();

    // 每个包解出多少样本（48 kHz）：帧长 × 帧数。PCM 里 SILK/混合/PLC 的包也占位。
    let mut prefix = vec![0usize; packets.len() + 1];
    for (i, p) in packets.iter().enumerate() {
        let toc = Toc::parse(p).unwrap_or_else(|e| panic!("{label}: 第 {i} 个包的 TOC 解析失败：{e}"));
        prefix[i + 1] = prefix[i] + toc.frame_size as usize * toc.frames as usize;
    }
    assert!(prefix[packets.len()] * 2 <= ref_pcm.len(), "{label}: 参考 PCM 比包序列短");

    let mut st = Stats::default();
    for (idx, f) in frames.iter().enumerate() {
        if idx >= limit || f.x.is_empty() {
            break;
        }
        let pkt = packets[f.pkt];
        let toc = Toc::parse(pkt).unwrap();
        let channels = if toc.stereo { 2 } else { 1 };
        let m = 1usize << toc.lm();
        let n = toc.frame_size as usize;
        assert_eq!(f.end, toc.bandwidth.end_band(), "{label}: 第 {idx} 帧结束带与 TOC 不符");

        // ① 播种：反塌陷之后的频谱 + log2Amp 之后的频带能量
        let mut x: Vec<Vec<f32>> = Vec::new();
        let mut band_e = [[0f32; NB_EBANDS]; 2];
        for (ch, e) in band_e.iter_mut().enumerate().take(channels) {
            let xv = f
                .x
                .get(&format!("X{ch}"))
                .unwrap_or_else(|| panic!("{label}: 第 {idx} 帧缺 X{ch}"));
            assert_eq!(xv.len(), n, "{label}: 第 {idx} 帧 X{ch} 长度不对");
            x.push(xv.clone());
            let bv = f
                .band
                .get(&format!("BAND{ch}"))
                .unwrap_or_else(|| panic!("{label}: 第 {idx} 帧缺 BAND{ch}"));
            e.copy_from_slice(&bv[..NB_EBANDS]);
        }
        // 静音帧：参考实现把 bandE 整张清 0（dump 的 BAND 行在清零**之前**）。
        if f.silence {
            for e in band_e.iter_mut().take(channels) {
                *e = [0.0; NB_EBANDS];
            }
        }

        // ② 反归一化 → 合成
        let freq = bands::denormalise_bands(&x, &band_e, f.end, channels, m);
        let mut pcm = vec![0i16; 2 * n];
        let inp = SynthInput {
            stream_channels: channels,
            out_channels: 2,
            lm: toc.lm(),
            transient: f.transient,
            start: f.start,
            end: f.end,
            postfilter: f.postfilter,
            freq: &freq,
        };
        synth_frame(&mut state, &mdct, &inp, &mut pcm);

        // ③ 与参考 PCM 逐样本比
        let off = (prefix[f.pkt] + f.frame_in_pkt * n) * 2;
        assert!(off + 2 * n <= ref_pcm.len(), "{label}: 第 {idx} 帧 PCM 偏移越界");
        for (j, &got) in pcm.iter().enumerate() {
            let want = ref_pcm[off + j] as i32;
            let d = (got as i32 - want).abs();
            st.max_diff = st.max_diff.max(d);
            if d >= 1 {
                st.differ += 1;
            }
            if d >= 2 {
                st.over_one += 1;
                if st.first_bad.is_none() {
                    st.first_bad = Some(format!(
                        "{label} 第 {idx} 帧（包 {} 第 {} 帧，lm={} trans={} 静音={}）声道 {} \
                         第 {} 个样本：我们 {got} vs 参考 {want}（差 {d}）",
                        f.pkt,
                        f.frame_in_pkt,
                        toc.lm(),
                        f.transient,
                        f.silence,
                        j % 2,
                        j / 2,
                    ));
                }
            }
        }
        st.samples += pcm.len();
        st.frames += 1;
        st.transient += usize::from(f.transient);
        st.silence += usize::from(f.silence);
        st.postfilter_on += usize::from(f.postfilter.gain > 0.0);
    }
    st
}

/// 跑一条流并断言误差在口径内。
fn assert_stream(label: &str, pkts: &[u8], dump: &str, ref_pcm: &[i16], limit: usize) -> Stats {
    let packets = parse_packets(pkts);
    let frames = parse_dump(dump);
    let st = run_stream(label, &packets, &frames, ref_pcm, limit);
    assert!(st.frames > 0, "{label}: 一帧都没跑（dump 里没有 CELT 帧？）");
    if let Some(bad) = &st.first_bad {
        println!("  {bad}");
    }
    let frac = st.over_one as f64 / st.samples as f64;
    let diff_frac = st.differ as f64 / st.samples as f64;
    println!(
        "{label}: 合成 {} 帧、{} 样本（瞬态 {} / 静音 {} / 后置滤波开 {}）；\
         最大差 {} 个量化步长；有差 {:.2e}、差 ≥2 的占比 {:.2e}",
        st.frames,
        st.samples,
        st.transient,
        st.silence,
        st.postfilter_on,
        st.max_diff,
        diff_frac,
        frac
    );
    // 浮点路径的落点差：1 个量化步长是取整边界的正常漂移（当前基线就是 1）。
    // 系统性偏差会先在"差 ≥2 的占比"上露头，两条一起卡。
    assert!(st.max_diff <= 2, "{label}: 最大差 {} 超过容忍（2 个量化步长）", st.max_diff);
    assert!(frac < 1e-5, "{label}: 差 ≥2 的样本占比 {frac:.2e} 太大");
    st
}

#[test]
#[ignore = "手工跑：需要 OPUS_PKTS / OPUS_CELT_DUMP / OPUS_REF_PCM（见文件顶部说明）"]
fn synthesis_matches_reference_pcm_on_a_real_opus_file() {
    let pkts = std::fs::read(env::var("OPUS_PKTS").expect("要设 OPUS_PKTS")).expect("读包");
    let dump = std::fs::read_to_string(env::var("OPUS_CELT_DUMP").expect("要设 OPUS_CELT_DUMP"))
        .expect("读 dump");
    let pcm_path = env::var("OPUS_REF_PCM").unwrap_or_else(|_| "/tmp/opus_dbg.pcm".into());
    let pcm_bytes = std::fs::read(&pcm_path).expect("读参考 PCM");
    let ref_pcm: Vec<i16> = pcm_bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();

    let st = assert_stream("真实 .opus", &pkts, &dump, &ref_pcm, 400);
    assert_eq!(st.frames, 400, "dump 的 X 覆盖前 400 帧，帧数对不上说明 dump 不完整");
}

#[test]
#[ignore = "手工跑：需要 OPUS_TV_DIR（官方向量的 dump 与 PCM 由 /tmp/opus_regen_tv_dumps.sh 生成）"]
fn synthesis_matches_reference_pcm_on_official_test_vectors() {
    let dir = env::var("OPUS_TV_DIR").expect("要设 OPUS_TV_DIR");
    // 只有纯 CELT-only 的向量能顺序跑（见文件顶部说明）
    let wanted = ["testvector01", "testvector07", "testvector11"];
    let (mut frames, mut max_diff, mut vectors) = (0usize, 0i32, 0usize);
    for name in wanted {
        let pkts_path = format!("{dir}/{name}.pkts.bin");
        let dump_path = format!("{dir}/{name}.dump.txt");
        let pcm_path = format!("{dir}/{name}.pcm");
        if !std::path::Path::new(&pkts_path).exists() {
            println!("{name}: 素材不在，跳过");
            continue;
        }
        let pkts = std::fs::read(&pkts_path).expect("读包");
        let dump = std::fs::read_to_string(&dump_path).expect("读 dump");
        let pcm_bytes = std::fs::read(&pcm_path).expect("读参考 PCM");
        let ref_pcm: Vec<i16> = pcm_bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        let st = assert_stream(name, &pkts, &dump, &ref_pcm, 400);
        frames += st.frames;
        max_diff = max_diff.max(st.max_diff);
        vectors += 1;
    }
    assert!(vectors >= 3, "带 CELT 帧的纯 CELT 向量不该少于 3 组");
    assert!(frames >= 1000, "核对帧数太少：{frames}");
    println!("官方向量合计：{vectors} 组、{frames} 帧；最大差 {max_diff} 个量化步长");
}
