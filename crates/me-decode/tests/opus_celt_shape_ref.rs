//! 手工跑：把自研的**频带形状解码**（`quant_all_bands` + 反塌陷）与插桩版参考实现对拍。
//!
//! 这一层是 CELT 里最绕的一段（递归二分 + θ + 折叠 + 反塌陷），所以对拍要做的
//! **不只是**比最终的 `X`：参考实现还会逐带 dump 一行 `QB`，把"这一带用多少位、
//! 从哪里折叠、解完之后 tell 在哪"都记下来。一旦分叉，QB 能直接告诉你从哪一带开始。
//!
//! dump 里与本轮有关的行（顺序固定，其余行忽略）：
//!
//! ```text
//! SEED   <LCG 种子>               ← 参考实现里它就是上一帧结束时的区间状态
//! LOGE0/1, LOGE20/1               ← 反塌陷要用的两帧历史能量
//! QB <带> <N> <b> <lowband> <cm_x> <cm_y> <tell_frac>
//! X0/X1                            ← 归一化频谱（**反塌陷之后**才 dump 的）
//! ```
//!
//! **播种纪律**：`I0/I1`（能量）、`SEED`（LCG）、`LOGE/LOGE2`（历史）三样都从 dump
//! 里取。这样每一帧都是独立核对的 —— 否则只要有一帧错了，后面全都会跟着漂。
//!
//! 跑法：
//!
//! ```text
//! OPUS_PKTS=/tmp/opus_pkts.bin OPUS_CELT_DUMP=/tmp/opus_dbg_frames.txt \
//! OPUS_TV_DIR=/tmp/opus_tv_pkts \
//!   cargo test -p me-decode --test opus_celt_shape_ref -- --ignored --nocapture
//! ```
//!
//! 官方向量的 dump 由 `/tmp/opus_regen_tv_dumps.sh` 生成（插桩脚本改动之后要重跑）。
//! 加 `OPUS_SHAPE_LENIENT=1` 可以把"逐带不一致"收集起来继续跑（定位用），
//! 默认是遇到第一帧就停下并把整张带表打出来。

use std::collections::HashMap;
use std::env;

use me_decode::audio::opus::{bands, celt, finish_frame, rate, RangeDecoder, Toc};

/// 参考实现 dump 出来的一行 `QB`。
#[derive(Default, Clone)]
struct Qb {
    n: usize,
    b: i32,
    lowband: isize,
    x_cm: u32,
    y_cm: u32,
    tell: i32,
}

#[derive(Default, Clone)]
struct Frame {
    pkt: usize,
    frame_in_pkt: u32,
    i: HashMap<String, Vec<f32>>,
    fine: HashMap<String, Vec<f32>>,
    loge: HashMap<String, Vec<f32>>,
    loge2: HashMap<String, Vec<f32>>,
    x: HashMap<String, Vec<f32>>,
    qb: Vec<Qb>,
    seed: u32,
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

/// 按行前缀驱动解析（不假设"每帧一定有 1 还是 2 个声道"）。
fn parse_dump(dump: &str) -> Vec<Frame> {
    let mut frames: Vec<Frame> = Vec::new();
    let mut cur = Frame::default();
    let mut started = false;
    let mut cur_pkt = 0usize;
    let mut frame_in_pkt = 0u32;
    for line in dump.lines() {
        if let Some(rest) = line.strip_prefix("PKT ") {
            cur_pkt = rest.split_whitespace().next().unwrap().parse().unwrap();
            frame_in_pkt = 0;
            continue;
        }
        if line.starts_with("I0 ") {
            if started {
                frames.push(std::mem::take(&mut cur));
            }
            started = true;
            cur = Frame { pkt: cur_pkt, frame_in_pkt, ..Default::default() };
            cur.i.insert("I0".into(), floats(line, "I0 "));
            continue;
        }
        if line.starts_with("I1 ") {
            cur.i.insert("I1".into(), floats(line, "I1 "));
            continue;
        }
        if line.starts_with("FRAME ") {
            frame_in_pkt += 1;
            continue;
        }
        if let Some(rest) = line.strip_prefix("SEED ") {
            cur.seed = rest.trim().parse().unwrap();
            continue;
        }
        if let Some(rest) = line.strip_prefix("QB ") {
            let v: Vec<i64> = rest.split_whitespace().map(|x| x.parse().unwrap()).collect();
            assert_eq!(v.len(), 7, "QB 行应当有 7 个数：{line}");
            cur.qb.push(Qb {
                n: v[1] as usize,
                b: v[2] as i32,
                lowband: v[3] as isize,
                x_cm: v[4] as u32,
                y_cm: v[5] as u32,
                tell: v[6] as i32,
            });
            continue;
        }
        for prefix in ["FINE", "LOGE2", "LOGE", "X"] {
            if line.starts_with(prefix) && line.len() > prefix.len() {
                let c = line.as_bytes()[prefix.len()];
                if c.is_ascii_digit() {
                    let key = format!("{prefix}{}", c as char);
                    let vals = floats(line, &format!("{key} "));
                    match prefix {
                        "FINE" => cur.fine.insert(key, vals),
                        "LOGE" => cur.loge.insert(key, vals),
                        "LOGE2" => cur.loge2.insert(key, vals),
                        _ => cur.x.insert(key, vals),
                    };
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

struct Stats {
    frames: usize,
    skipped: usize,
    max_x_err: f32,
    /// X 里最大的差出现在哪一带（定位用）
    worst_band: usize,
    /// 逐带不一致的帧（宽松模式下收集，最后一起报）
    bad: Vec<String>,
}

/// 第 `j` 个系数落在哪一带（只用于报错信息）。
fn band_of(j: usize, n: usize) -> usize {
    let m = (n / celt::SHORT_MDCT as usize) as i32;
    for i in 0..celt::NB_EBANDS {
        let (lo, hi) = (
            m * bands::band_edge(i) as i32,
            m * bands::band_edge(i + 1) as i32,
        );
        if (j as i32) >= lo && (j as i32) < hi {
            return i;
        }
    }
    celt::NB_EBANDS - 1
}

/// 一帧跑完整条链：前端 → 分配 → 细能量 → 形状 → 反塌陷位 → 帧尾细能量 → 反塌陷。
///
/// 返回 `true` 表示"这一帧真的核对了"。**只有 CELT-only 的帧才会返回 true**：
/// SILK-only 与混合模式还没接（`decode_front` 会明确报 `Unsupported`），
/// 把它们算成"核对过"是自欺欺人。
fn run_frame(label: &str, idx: usize, pkt: &[u8], f: &Frame, out: &mut Stats) -> bool {
    let toc = match Toc::parse(pkt) {
        Ok(t) => t,
        Err(_) => {
            out.skipped += 1;
            return false;
        }
    };
    // SILK-only / 混合模式还没接：跳过（不产噪声，也不算核对过）
    if toc.mode != celt::OpusMode::CeltOnly {
        out.skipped += 1;
        return false;
    }
    // 走到这里说明"参考那边一定解码了这一帧"，dump 里就必须有 FINE/QB 行。
    // 缺了只能是"插桩脚本改过但 dump 没重跑"，必须当场喊出来 —— 否则会静默少核对一大截。
    assert!(
        f.fine.contains_key("FINE0") && !f.qb.is_empty(),
        "{label} 第 {idx} 帧（包 {}）是 CELT-only，但 dump 里没有 FINE/QB 行 —— \
         dump 需要重跑（/tmp/opus_regen_tv_dumps.sh），或者当时有别的进程在写同一个文件",
        f.pkt
    );
    let payload = match celt::frame_payload(pkt, toc, f.frame_in_pkt) {
        Ok(p) => p,
        Err(_) => {
            out.skipped += 1;
            return false;
        }
    };
    let channels = if toc.stereo { 2 } else { 1 };
    let start = 0usize;
    let end = toc.bandwidth.end_band();
    let lm = toc.lm();

    // ① 用参考的帧前状态播种（能量 / 历史能量 / LCG 种子）
    let i0: [f32; celt::NB_EBANDS] = f.i["I0"].clone().try_into().unwrap();
    let i1: [f32; celt::NB_EBANDS] = match f.i.get("I1") {
        Some(v) => v.clone().try_into().unwrap(),
        None => i0,
    };
    let mut st = celt::CeltState { old_ebands: [i0, i1], rng: f.seed, ..Default::default() };
    for ch in 0..2 {
        if let Some(v) = f.loge.get(&format!("LOGE{ch}")) {
            st.old_log_e[ch] = v.clone().try_into().unwrap();
        }
        if let Some(v) = f.loge2.get(&format!("LOGE2{ch}")) {
            st.old_log_e2[ch] = v.clone().try_into().unwrap();
        }
    }

    let mut dec = RangeDecoder::new(payload);
    let front = celt::decode_front(&mut dec, toc.lm(), toc.bandwidth.end_band(), 0, channels, &mut st)
        .unwrap_or_else(|e| panic!("{label} 第 {idx} 帧前端失败：{e}"));
    let alloc = rate::decode_allocation(&mut dec, toc.lm(), channels, front.transient, start, end)
        .unwrap_or_else(|e| panic!("{label} 第 {idx} 帧分配失败：{e}"));
    bands::unquant_fine_energy(&mut dec, &alloc, channels, start, end, &mut st)
        .unwrap_or_else(|e| panic!("{label} 第 {idx} 帧细能量失败：{e}"));

    // ② 细能量（和参考的 FINE 行对照：失败点能一眼看出在形状之前还是之后）
    for ch in 0..channels {
        let want = &f.fine[&format!("FINE{ch}")];
        for band in start..end {
            assert!(
                (st.old_ebands[ch][band] - want[band]).abs() < 1e-3,
                "{label} 第 {idx} 帧 ch{ch} 第 {band} 带细能量：{} vs {}",
                st.old_ebands[ch][band],
                want[band]
            );
        }
    }

    // ③ 形状解码
    let total_bits = (payload.len() as i32) * (8 << rate::BITRES) - alloc.anti_collapse_rsv;
    let inp = bands::ShapeInput {
        pulses: &alloc.pulses,
        tf: &front.tf,
        start,
        end,
        lm,
        transient: front.transient,
        spread: front.spread,
        dual_stereo: alloc.dual_stereo,
        intensity: alloc.intensity,
        coded_bands: alloc.coded_bands,
        total_bits,
        balance: alloc.balance,
        seed: f.seed,
    };
    let mut shape = bands::quant_all_bands(&mut dec, &inp, channels)
        .unwrap_or_else(|e| panic!("{label} 第 {idx} 帧形状失败：{e}"));

    // ④ 逐带核对（参考的 QB 行）。不一致时把整张带表打出来，别只报第一个错。
    assert_eq!(
        shape.trace.len(),
        f.qb.len(),
        "{label} 第 {idx} 帧：QB 行数 {} 与我们的带数 {} 不一致",
        f.qb.len(),
        shape.trace.len()
    );
    let mut bad = Vec::new();
    for (t, w) in shape.trace.iter().zip(f.qb.iter()) {
        if t.n != w.n
            || t.b != w.b
            || t.lowband != w.lowband
            || (t.x_cm & 0xFF) != w.x_cm & 0xFF
            || (t.y_cm & 0xFF) != w.y_cm & 0xFF
            || t.tell != w.tell
        {
            bad.push(format!(
                "  带{:2}: n {:3}/{:3}  b {:6}/{:6}  lowband {:5}/{:5}  cm {:x}/{:x} vs {:x}/{:x}  tell {:6}/{:6}",
                t.band, t.n, w.n, t.b, w.b, t.lowband, w.lowband,
                t.x_cm & 0xFF, t.y_cm & 0xFF, w.x_cm & 0xFF, w.y_cm & 0xFF, t.tell, w.tell
            ));
        }
    }
    if !bad.is_empty() {
        let desc = format!(
            "{label} 第 {idx} 帧（包 {} 第 {} 帧，lm={lm} dual={} intensity={} trans={} spread={}）\
             逐带不一致（我们/参考）：\n{}",
            f.pkt,
            f.frame_in_pkt,
            alloc.dual_stereo,
            alloc.intensity,
            front.transient,
            front.spread,
            bad.join("\n")
        );
        if env::var("OPUS_SHAPE_LENIENT").is_ok() {
            // 宽松模式：记下来继续跑（定位用）。注意后面的解码已经脱同步，看的是"哪里第一处错"
            out.bad.push(desc);
            return true;
        }
        panic!("{desc}");
    }

    // ⑤ 反塌陷开关位 + 帧尾细能量
    let ac_on = if alloc.anti_collapse_rsv > 0 {
        dec.dec_bits(1).unwrap() != 0
    } else {
        false
    };
    let bits_left = payload.len() as i32 * 8 - dec.tell();
    bands::unquant_energy_finalise(&mut dec, &alloc, channels, start, end, bits_left, &mut st)
        .unwrap_or_else(|e| panic!("{label} 第 {idx} 帧帧尾细能量失败：{e}"));
    if ac_on {
        bands::anti_collapse(
            &mut shape.x,
            &shape.collapse_masks,
            lm,
            channels,
            start,
            end,
            &st.old_ebands,
            &st.old_log_e,
            &st.old_log_e2,
            &alloc.pulses,
            shape.seed,
        );
    }

    // ⑥ 归一化频谱（参考的 X 行 —— 只 dump 了每次运行的前若干帧）
    out.frames += 1;
    if f.x.is_empty() {
        finish_frame(&mut st, channels, front.silence);
        return true;
    }
    let n = toc.frame_size as usize;
    for ch in 0..channels {
        let want = &f.x[&format!("X{ch}")];
        for j in 0..n {
            let d = (shape.x[ch][j] - want[j]).abs();
            let band = band_of(j, n);
            if d > out.max_x_err {
                out.max_x_err = d;
                out.worst_band = band;
            }
            // 差得离谱时把附近一小段一起打出来 —— 只报一个系数往往看不出是"整条带错"
            // 还是"差一个因子"
            if d >= 1e-3 {
                let lo = j.saturating_sub(4);
                let hi = (j + 5).min(n);
                panic!(
                    "{label} 第 {idx} 帧（包 {} 第 {} 帧，lm={lm} trans={} spread={}）ch{ch} \
                     第 {j} 个系数（第 {band} 带）：我们 {} vs 参考 {}\n  我们 [{lo},{hi}) = {:?}\n  参考 [{lo},{hi}) = {:?}",
                    f.pkt,
                    f.frame_in_pkt,
                    front.transient,
                    front.spread,
                    shape.x[ch][j],
                    want[j],
                    &shape.x[ch][lo..hi],
                    &want[lo..hi]
                );
            }
        }
    }
    finish_frame(&mut st, channels, front.silence);
    true
}

fn compare(label: &str, pkts: &[u8], dump: &str) -> Stats {
    let packets = parse_packets(pkts);
    let frames = parse_dump(dump);
    // SILK-only 的官方向量里 CELT 解码器根本不会被调用 → dump 里一帧都没有。
    // 那不是失败，是"这组向量还没轮到"（等第 9 步做完 SILK 再说）。
    if frames.is_empty() {
        return Stats { frames: 0, skipped: 0, max_x_err: 0f32, worst_band: 0, bad: Vec::new() };
    }
    // 调试开关：`OPUS_SHAPE_TRACE=<帧号>` 会把那一帧的 `quant_band` 递归轨迹打到
    // stderr（格式与参考实现的 `OPUS_DBG_FRAME` 一致，可以直接 diff）。
    if let Ok(v) = env::var("OPUS_SHAPE_TRACE") {
        use std::sync::atomic::Ordering;
        me_decode::audio::opus::bands::SHAPE_TRACE.store(v.parse().unwrap(), Ordering::Relaxed);
        me_decode::audio::opus::bands::SHAPE_TRACE_N.store(0, Ordering::Relaxed);
    }
    let mut st = Stats { frames: 0, skipped: 0, max_x_err: 0f32, worst_band: 0, bad: Vec::new() };
    let mut celt_only = 0usize;
    for (idx, f) in frames.iter().enumerate() {
        assert!(f.pkt < packets.len(), "{label}: 参考解了更多包");
        if run_frame(label, idx, packets[f.pkt], f, &mut st) {
            celt_only += 1;
        }
    }
    // "该核对的帧"与"真核对过的帧"必须一一对应 —— 这条比绝对帧数更能抓住回归
    assert_eq!(
        celt_only, st.frames,
        "{label}: CELT 帧数 {celt_only} 与实际核对过的帧数 {} 不一致",
        st.frames
    );
    st
}

#[test]
#[ignore = "手工跑：需要 OPUS_PKTS 与 OPUS_CELT_DUMP（见文件顶部说明）"]
fn band_shape_matches_reference_on_a_real_opus_file() {
    let pkts = std::fs::read(env::var("OPUS_PKTS").expect("要设 OPUS_PKTS")).expect("读包");
    let dump = std::fs::read_to_string(env::var("OPUS_CELT_DUMP").expect("要设 OPUS_CELT_DUMP"))
        .expect("读 dump");
    let st = compare("真实 .opus", &pkts, &dump);
    for b in st.bad.iter().take(6) {
        println!("{b}");
    }
    assert!(st.bad.is_empty(), "有 {} 帧逐带不一致（见上面打印）", st.bad.len());
    assert!(st.frames > 10_000, "核对帧数太少：{}", st.frames);
    println!(
        "真实 .opus：逐带 + 归一化频谱一致 {} 帧（跳过 {}），X 最大差 {:.6}（最大出现在第 {} 带）",
        st.frames, st.skipped, st.max_x_err, st.worst_band
    );
}

#[test]
#[ignore = "手工跑：需要 OPUS_TV_DIR（官方向量的 dump 由 /tmp/opus_regen_tv_dumps.sh 生成）"]
fn band_shape_matches_reference_on_official_test_vectors() {
    let dir = env::var("OPUS_TV_DIR").expect("要设 OPUS_TV_DIR");
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .expect("读目录")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.to_string_lossy().ends_with(".pkts.bin"))
        .collect();
    entries.sort();
    let (mut frames, mut skipped, mut vectors) = (0usize, 0usize, 0usize);
    let mut max_err = 0f32;
    for path in entries {
        let name = path
            .file_name()
            .expect("文件名")
            .to_string_lossy()
            .trim_end_matches(".pkts.bin")
            .to_string();
        let dump_path = path.with_file_name(format!("{name}.dump.txt"));
        if !dump_path.exists() {
            continue;
        }
        let pkts = std::fs::read(&path).expect("读包文件");
        let dump = std::fs::read_to_string(&dump_path).expect("读 dump");
        let st = compare(&name, &pkts, &dump);
        assert!(st.bad.is_empty(), "{name}: 有 {} 帧逐带不一致", st.bad.len());
        if st.frames == 0 {
            // SILK-only：CELT 解码器根本没被调用；混合：CELT 那段还没接（`decode_front`
            // 会明确报 Unsupported）。两种都不算失败，但要说清楚是哪一种。
            if st.skipped == 0 {
                println!("{name}: 这一组没有 CELT 帧（纯 SILK-only）");
            } else {
                println!("{name}: 跳过 {} 帧（混合/SILK-only，CELT 那一段还没接）", st.skipped);
            }
            continue;
        }
        println!(
            "{name}: 逐带 + 频谱一致 {} 帧（跳过 {}），X 最大差 {:.6}",
            st.frames, st.skipped, st.max_x_err
        );
        frames += st.frames;
        skipped += st.skipped;
        max_err = max_err.max(st.max_x_err);
        vectors += 1;
    }
    // 12 组向量里带 CELT 帧的是 01 / 07 / 08 / 09 / 10 / 11 六组（合计 18784 帧，
    // 与参考插桩里数出来的一致）；其余是 SILK-only 或混合。
    assert!(vectors >= 6, "带 CELT 帧的向量太少了：{vectors}");
    assert!(frames >= 18_000, "核对帧数太少：{frames}");
    println!(
        "官方向量合计：{vectors} 组、{frames} 帧一致（跳过 {skipped}），X 最大差 {max_err:.6}"
    );
}
