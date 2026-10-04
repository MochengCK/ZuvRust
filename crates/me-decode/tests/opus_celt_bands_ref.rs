//! 手工跑：把自研的**细能量解码**与**归一化还原**与插桩版参考实现对拍（默认 `#[ignore]`）。
//!
//! 这个 dump 每帧的行比较多（顺序固定）：
//!
//! ```text
//! PKT <序号> <载荷长度>
//! I0/I1        帧前能量
//! FRAME        len/silence/trans/intra/pf/pitch/qg/tapset/spread/tell/start/end/nbEBands
//! TF / E0 / E1
//! ALLOC / OFF / PL / FQ / FP
//! FINE0/FINE1  细能量解码之后的能量
//! BAND0/BAND1  log2Amp 之后的频带能量
//! X0/X1        归一化频谱（只 dump 前若干帧）
//! FREQ0/FREQ1  denormalise 之后的频谱（同上）
//! ```
//!
//! 按**行前缀**驱动解析（不依赖"某帧一定有 1 个还是 2 个声道"这种细节）。
//!
//! 跑法：
//!
//! ```text
//! OPUS_PKTS=/tmp/opus_pkts.bin OPUS_CELT_DUMP=/tmp/opus_dbg_frames.txt \
//!   cargo test -p me-decode --test opus_celt_bands_ref -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::env;

use me_decode::audio::opus::{bands, celt, finish_frame, rate, RangeDecoder, Toc};

#[derive(Default, Clone)]
struct Frame {
    pkt: usize,
    frame_in_pkt: u32,
    i: HashMap<String, Vec<f32>>,
    band: HashMap<String, Vec<f32>>,
    x: HashMap<String, Vec<f32>>,
    freq: HashMap<String, Vec<f32>>,
    fine: HashMap<String, Vec<f32>>,
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
        let n = u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
            as usize;
        pos += 4;
        assert!(pos + n <= bytes.len(), "包文件被截断");
        out.push(&bytes[pos..pos + n]);
        pos += n;
    }
    out
}

#[test]
#[ignore = "手工跑：需要 OPUS_PKTS 与 OPUS_CELT_DUMP（见文件顶部说明）"]
fn fine_energy_and_denormalise_match_reference() {
    let pkts = std::fs::read(env::var("OPUS_PKTS").expect("要设 OPUS_PKTS")).expect("读包");
    let dump = std::fs::read_to_string(env::var("OPUS_CELT_DUMP").expect("要设 OPUS_CELT_DUMP"))
        .expect("读 dump");
    let packets = parse_packets(&pkts);

    let mut frames: Vec<Frame> = Vec::new();
    let mut cur = Frame::default();
    let mut started = false;
    let mut cur_pkt = 0usize;
    let mut frame_in_pkt = 0u32;
    let mut lines = dump.lines().peekable();
    while let Some(line) = lines.next() {
        if let Some(rest) = line.strip_prefix("PKT ") {
            let mut it = rest.split_whitespace();
            cur_pkt = it.next().unwrap().parse().unwrap();
            frame_in_pkt = 0;
            continue;
        }
        if let Some(rest) = line.strip_prefix("I0 ") {
            // 新的一帧开始：先把上一帧结算掉
            if started {
                frames.push(std::mem::take(&mut cur));
            }
            started = true;
            cur = Frame { pkt: cur_pkt, frame_in_pkt, ..Default::default() };
            cur.i.insert("I0".into(), floats(&format!("I0 {rest}"), "I0 "));
            continue;
        }
        if let Some(rest) = line.strip_prefix("I1 ") {
            cur.i.insert("I1".into(), floats(&format!("I1 {rest}"), "I1 "));
            continue;
        }
        if line.starts_with("FRAME ") {
            // 帧头本身这里不校对（前面的测试已经校过），只要把它记下来当作边界
            frame_in_pkt += 1;
            continue;
        }
        for prefix in ["FINE", "BAND", "X", "FREQ"] {
            if line.starts_with(prefix) && line.len() > prefix.len() {
                let c = line.as_bytes()[prefix.len()];
                if c.is_ascii_digit() {
                    let key = format!("{prefix}{}", (c as char));
                    let vals = floats(line, &format!("{key} "));
                    match prefix {
                        "FINE" => cur.fine.insert(key, vals),
                        "BAND" => cur.band.insert(key, vals),
                        "X" => cur.x.insert(key, vals),
                        _ => cur.freq.insert(key, vals),
                    };
                    break;
                }
            }
        }
    }
    if started {
        frames.push(cur);
    }
    assert!(frames.len() > 10_000, "帧数太少：{}", frames.len());

    let mut checked_fine = 0usize;
    let mut checked_denorm = 0usize;
    let mut max_fine_err = 0f32;
    let mut max_freq_err = 0f32;
    let mut skipped = 0usize;
    for (idx, f) in frames.iter().enumerate() {
        let pkt = packets[f.pkt];
        let toc = match Toc::parse(pkt) {
            Ok(t) => t,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        if toc.mode != celt::OpusMode::CeltOnly {
            skipped += 1;
            continue;
        }
        let payload = match celt::frame_payload(pkt, toc, f.frame_in_pkt) {
            Ok(p) => p,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        let channels = if toc.stereo { 2 } else { 1 };
        let start = 0usize;
        let end = toc.bandwidth.end_band();
        // 用参考的帧前能量播种，把被测单元隔离出来
        let i0: [f32; celt::NB_EBANDS] = f.i["I0"].clone().try_into().unwrap();
        let i1: [f32; celt::NB_EBANDS] = match f.i.get("I1") {
            Some(v) => v.clone().try_into().unwrap(),
            None => i0,
        };
        let mut st = celt::CeltState { old_ebands: [i0, i1], ..Default::default() };
        let mut dec = RangeDecoder::new(payload);
        let front = celt::decode_front(&mut dec, toc.lm(), toc.bandwidth.end_band(), 0, channels, &mut st)
            .unwrap_or_else(|e| panic!("第 {idx} 帧前端失败：{e}"));
        let alloc = rate::decode_allocation(&mut dec, toc.lm(), channels, front.transient, start, end)
            .unwrap_or_else(|e| panic!("第 {idx} 帧分配失败：{e}"));
        bands::unquant_fine_energy(&mut dec, &alloc, channels, start, end, &mut st)
            .unwrap_or_else(|e| panic!("第 {idx} 帧细能量失败：{e}"));

        // ① 细能量之后的能量
        for ch in 0..channels {
            let want = &f.fine[&format!("FINE{ch}")];
            for band in start..end {
                let d = (st.old_ebands[ch][band] - want[band]).abs();
                max_fine_err = max_fine_err.max(d);
                assert!(
                    d < 1e-3,
                    "第 {idx} 帧 ch{ch} 第 {band} 带细能量之后：{} vs {}",
                    st.old_ebands[ch][band],
                    want[band]
                );
            }
        }
        checked_fine += 1;
        // 帧尾状态更新（静音帧的 -28、单声道回填）
        finish_frame(&mut st, channels, front.silence);

        // ② 归一化还原（只有前若干帧 dump 了 X/FREQ）
        if f.x.is_empty() {
            continue;
        }
        let n = (toc.frame_size as usize / 8) * 8; // M * 120，M = 1<<LM
        let x: Vec<Vec<f32>> = (0..channels)
            .map(|ch| f.x[&format!("X{ch}")].clone())
            .collect();
        let band_e: Vec<[f32; celt::NB_EBANDS]> = (0..channels)
            .map(|ch| {
                if front.silence {
                    // 参考实现：静音帧在 log2Amp 之后会把 bandE 清 0（于是 freq 全 0）
                    return [0f32; celt::NB_EBANDS];
                }
                let v = &f.band[&format!("BAND{ch}")];
                let mut arr = [0f32; celt::NB_EBANDS];
                arr.copy_from_slice(&v[..celt::NB_EBANDS]);
                arr
            })
            .collect();
        let m = toc.frame_size as usize / celt::SHORT_MDCT as usize;
        let freq = bands::denormalise_bands(&x, &band_e, end, channels, m);
        for ch in 0..channels {
            let want = &f.freq[&format!("FREQ{ch}")];
            for j in 0..n {
                let d = (freq[ch][j] - want[j]).abs();
                max_freq_err = max_freq_err.max(d);
                assert!(
                    d < 1e-2,
                    "第 {idx} 帧 ch{ch} 第 {j} 个系数：{} vs {}（band={}）",
                    freq[ch][j],
                    want[j],
                    j / m
                );
            }
        }
        checked_denorm += 1;
    }
    println!(
        "细能量：{checked_fine} 帧一致（最大差 {max_fine_err:.6}）；归一化还原：{checked_denorm} 帧一致（最大差 {max_freq_err:.6}）；跳过 {skipped}"
    );
}
