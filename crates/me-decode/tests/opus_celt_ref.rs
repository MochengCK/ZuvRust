//! 手工跑：把自研 CELT 前端（帧头 + 粗能量 + TF + spread）与**插桩版参考实现**逐帧对拍。
//!
//! 参考实现由 `/tmp/opus_patch_dbg.py` 插桩（`OPUS_DBG_DUMP` 门控），每帧 dump：
//!
//! ```text
//! PKT <包序号> <载荷长度>      由我自己的探针打印（SILK-only 的包不会被 CELT 解码）
//! I0 <21 个带能量>             帧前状态（粗能量解码之前）
//! I1 <21 个带能量>             同上，第二声道
//! FRAME len=… silence=… trans=… intra=… pf=… pitch=… qg=… tapset=… spread=… tell=…
//! TF <21 个调整量>
//! E0 <21 个带能量>             粗能量解码之后
//! E1 <21 个带能量>
//! ```
//!
//! **为什么要用 I0/I1 给每帧播种**：参考实现解完粗能量之后还会用**细能量**
//! （比特分配之后才解）修正 `oldBandE`。本引擎这一层还没做，所以累积状态会漂 ——
//! 拿参考的"帧前状态"播种，就把被测单元干净地隔离出来，每一帧都能独立核对。
//!
//! 跑法：
//!
//! ```text
//! OPUS_PKTS=/tmp/opus_pkts.bin OPUS_CELT_DUMP=/tmp/opus_dbg_frames.txt \
//! OPUS_TV_DIR=/tmp/opus_tv_pkts \
//!   cargo test -p me-decode --test opus_celt_ref -- --ignored --nocapture
//! ```

use std::env;

use me_decode::audio::opus::{celt, rate, RangeDecoder, Toc};

/// 按小端长度前缀切包。
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

fn floats(line: &str, prefix: &str, n: usize) -> Vec<f32> {
    let body = line
        .strip_prefix(prefix)
        .unwrap_or_else(|| panic!("期望以 {prefix} 开头的行：{line}"));
    let v: Vec<f32> = body.split_whitespace().map(|x| x.parse().unwrap()).collect();
    assert_eq!(v.len(), n, "{prefix} 行应当有 {n} 个数：{line}");
    v
}

fn ints(line: &str, prefix: &str, n: usize) -> Vec<i32> {
    let body = line
        .strip_prefix(prefix)
        .unwrap_or_else(|| panic!("期望以 {prefix} 开头的行：{line}"));
    let v: Vec<i32> = body.split_whitespace().map(|x| x.parse().unwrap()).collect();
    assert_eq!(v.len(), n, "{prefix} 行应当有 {n} 个数：{line}");
    v
}

/// 逐帧比较，返回（核对帧数, 跳过的帧数, 能量最大差）。
fn compare(label: &str, pkts: &[u8], dump: &str) -> (usize, usize, f32) {
    let packets = parse_packets(pkts);
    let mut lines = dump.lines().peekable();
    let mut checked = 0usize;
    let mut skipped = 0usize;
    let mut max_err = 0f32;
    let mut cur_pkt = 0usize;
    let mut frame_in_pkt = 0u32;
    while let Some(line) = lines.next() {
        if let Some(rest) = line.strip_prefix("PKT ") {
            let mut it = rest.split_whitespace();
            cur_pkt = it.next().unwrap().parse().unwrap();
            frame_in_pkt = 0;
            continue;
        }
        if !line.starts_with("I0 ") {
            continue;
        }
        let i0: [f32; celt::NB_EBANDS] = floats(line, "I0 ", celt::NB_EBANDS).try_into().unwrap();
        let i1: [f32; celt::NB_EBANDS] =
            floats(&lines.next().expect("I1 行"), "I1 ", celt::NB_EBANDS)
                .try_into()
                .unwrap();
        let frame_line = lines.next().expect("FRAME 行");
        let rest = frame_line.strip_prefix("FRAME ").expect("FRAME 行");
        let field = |name: &str| -> i64 {
            rest.split_whitespace()
                .find_map(|kv| kv.strip_prefix(name).and_then(|v| v.strip_prefix('=')))
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("{label}: dump 行里没有 {name}：{frame_line}"))
        };
        let want_len = field("len") as usize;
        let want_silence = field("silence") == 1;
        let want_trans = field("trans") == 1;
        let want_intra = field("intra") == 1;
        let want_pf = field("pf") == 1;
        let want_pitch = field("pitch") as i32;
        let want_qg = field("qg") as i32;
        let want_tapset = field("tapset") as u32;
        let want_spread = field("spread") as u32;
        let want_tell = field("tell") as i32;
        let want_start = field("start") as usize;
        let want_end = field("end") as usize;
        let want_tf = ints(&lines.next().expect("TF 行"), "TF ", celt::NB_EBANDS);
        let want_e0 = floats(&lines.next().expect("E0 行"), "E0 ", celt::NB_EBANDS);
        let want_e1 = floats(&lines.next().expect("E1 行"), "E1 ", celt::NB_EBANDS);
        // 分配结果（参考实现在 compute_allocation 之后 dump）
        let alloc_line = lines.next().expect("ALLOC 行");
        let alloc_rest = alloc_line.strip_prefix("ALLOC ").expect("ALLOC 行");
        let afield = |name: &str| -> i64 {
            alloc_rest
                .split_whitespace()
                .find_map(|kv| kv.strip_prefix(name).and_then(|v| v.strip_prefix('=')))
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("{label}: ALLOC 行里没有 {name}：{alloc_line}"))
        };
        let want_trim = afield("trim") as i32;
        let want_coded = afield("coded") as usize;
        let want_intensity = afield("intensity") as i32;
        let want_dual = afield("dual") == 1;
        let want_balance = afield("balance") as i32;
        let want_ac = afield("ac") as i32;
        let want_bits = afield("bits") as i32;
        let want_off = ints(&lines.next().expect("OFF 行"), "OFF ", celt::NB_EBANDS);
        let want_pl = ints(&lines.next().expect("PL 行"), "PL ", celt::NB_EBANDS);
        let want_fq = ints(&lines.next().expect("FQ 行"), "FQ ", celt::NB_EBANDS);
        let want_fp = ints(&lines.next().expect("FP 行"), "FP ", celt::NB_EBANDS);

        assert!(cur_pkt < packets.len(), "{label}: 参考解了更多包");
        let pkt = packets[cur_pkt];
        // 跳过：PLC（len 0/1）与还没接的模式（SILK / 混合）
        if want_len == 0 || pkt.is_empty() {
            skipped += 1;
            continue;
        }
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
        let payload = match celt::frame_payload(pkt, toc, frame_in_pkt) {
            Ok(p) => p,
            Err(_) => {
                skipped += 1;
                frame_in_pkt += 1;
                continue;
            }
        };
        frame_in_pkt += 1;
        assert_eq!(
            want_len,
            payload.len(),
            "{label}: 第 {checked} 个核对帧（包 {cur_pkt}）载荷长度与参考不一致"
        );
        let channels = if toc.stereo { 2 } else { 1 };

        // 用参考的帧前状态播种，隔离掉还没实现的层
        let mut state = celt::CeltState { old_ebands: [i0, i1], ..Default::default() };
        let mut dec = RangeDecoder::new(payload);
        let front = celt::decode_front(&mut dec, toc.lm(), toc.bandwidth.end_band(), 0, channels, &mut state)
            .unwrap_or_else(|e| panic!("{label}: 包 {cur_pkt} 解码失败：{e}"));

        assert_eq!(front.silence, want_silence, "{label}: 包 {cur_pkt} silence");
        assert_eq!(front.transient, want_trans, "{label}: 包 {cur_pkt} transient");
        assert_eq!(front.intra, want_intra, "{label}: 包 {cur_pkt} intra");
        assert_eq!(front.postfilter_on, want_pf, "{label}: 包 {cur_pkt} 后置滤波开关");
        assert_eq!(front.spread, want_spread, "{label}: 包 {cur_pkt} spread");
        // 频带范围来自"模式 + 带宽"（NB 13 / MB·WB 17 / SWB 19 / FB 21），逐帧核对
        assert_eq!(front.start, want_start, "{label}: 包 {cur_pkt} start 带");
        assert_eq!(front.end, want_end, "{label}: 包 {cur_pkt} end 带");
        if want_pf {
            assert_eq!(front.postfilter.pitch, want_pitch, "{label}: 包 {cur_pkt} pitch");
            let got_qg = (front.postfilter.gain / 0.09375) as i32 - 1;
            assert_eq!(got_qg, want_qg, "{label}: 包 {cur_pkt} 增益（qg）");
            assert_eq!(front.postfilter.tapset, want_tapset, "{label}: 包 {cur_pkt} tapset");
        }
        for (i, &w) in want_e0.iter().enumerate() {
            let d = (front.coarse_energy[0][i] - w).abs();
            max_err = max_err.max(d);
            assert!(
                d < 1e-3,
                "{label}: 包 {cur_pkt} ch0 第 {i} 带能量：{} vs {w}",
                front.coarse_energy[0][i]
            );
        }
        if channels == 2 {
            for (i, &w) in want_e1.iter().enumerate() {
                let d = (front.coarse_energy[1][i] - w).abs();
                max_err = max_err.max(d);
                assert!(
                    d < 1e-3,
                    "{label}: 包 {cur_pkt} ch1 第 {i} 带能量：{} vs {w}",
                    front.coarse_energy[1][i]
                );
            }
        }
        for (i, &w) in want_tf.iter().enumerate() {
            // 参考实现的 `tf_res` 在 [start,end) 之外是**未初始化内存**（我 dump 到了垃圾），
            // 所以只比这个区间 —— 那才是码流里真解出来的部分。
            if i < front.start || i >= front.end {
                continue;
            }
            assert_eq!(front.tf[i], w, "{label}: 包 {cur_pkt} 第 {i} 带 TF");
        }
        assert_eq!(front.tell, want_tell, "{label}: 包 {cur_pkt} tell（比特分配要用它）");

        // 比特分配：接着同一个区间解码器往下解
        let alloc = rate::decode_allocation(
            &mut dec,
            toc.lm(),
            channels,
            front.transient,
            front.start,
            front.end,
        )
        .unwrap_or_else(|e| panic!("{label}: 包 {cur_pkt} 分配失败：{e}"));
        assert_eq!(alloc.alloc_trim, want_trim, "{label}: 包 {cur_pkt} alloc_trim");
        assert_eq!(alloc.coded_bands, want_coded, "{label}: 包 {cur_pkt} codedBands");
        assert_eq!(alloc.intensity, want_intensity, "{label}: 包 {cur_pkt} intensity");
        assert_eq!(alloc.dual_stereo, want_dual, "{label}: 包 {cur_pkt} dual_stereo");
        assert_eq!(alloc.balance, want_balance, "{label}: 包 {cur_pkt} balance");
        assert_eq!(alloc.anti_collapse_rsv, want_ac, "{label}: 包 {cur_pkt} 反塌陷保留位");
        // 注意：参考实现的这些数组在 `[start,end)` 之外**没有写过**（是我 dump 到的垃圾值），
        // 只比有效区间 —— 与 tf_res 同一个道理。
        assert_eq!(alloc.bits, want_bits, "{label}: 包 {cur_pkt} 分配输入位数");
        for i in front.start..front.end {
            assert_eq!(alloc.offsets[i], want_off[i], "{label}: 包 {cur_pkt} 第 {i} 带 dynalloc");
            assert_eq!(alloc.pulses[i], want_pl[i], "{label}: 包 {cur_pkt} 第 {i} 带 PVQ 位数");
            assert_eq!(alloc.fine_quant[i], want_fq[i], "{label}: 包 {cur_pkt} 第 {i} 带细能量位");
            assert_eq!(
                alloc.fine_priority[i], want_fp[i],
                "{label}: 包 {cur_pkt} 第 {i} 带细能量优先级"
            );
        }
        checked += 1;
    }
    (checked, skipped, max_err)
}

#[test]
#[ignore = "手工跑：需要 OPUS_PKTS 与 OPUS_CELT_DUMP（见文件顶部说明）"]
fn celt_front_matches_reference_on_a_real_opus_file() {
    let pkts = std::fs::read(env::var("OPUS_PKTS").expect("要设 OPUS_PKTS")).expect("读包文件");
    let dump = std::fs::read_to_string(env::var("OPUS_CELT_DUMP").expect("要设 OPUS_CELT_DUMP"))
        .expect("读 dump");
    let (checked, skipped, max_err) = compare("真实 .opus", &pkts, &dump);
    assert!(checked > 10_000, "核对帧数太少：{checked}");
    println!("真实 .opus：逐帧一致 {checked} 帧（跳过 {skipped}），能量最大差 {max_err:.6}");
}

#[test]
#[ignore = "手工跑：需要 OPUS_TV_DIR（官方向量转换后的目录，见文件顶部说明）"]
fn celt_front_matches_reference_on_official_test_vectors() {
    let dir = env::var("OPUS_TV_DIR").expect("要设 OPUS_TV_DIR");
    let mut total = 0usize;
    let mut total_skipped = 0usize;
    let mut max_err = 0f32;
    let mut vectors = 0usize;
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .expect("读目录")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.to_string_lossy().ends_with(".pkts.bin"))
        .collect();
    entries.sort();
    for path in entries {
        // 文件名形如 testvector01.pkts.bin → 基础名 testvector01
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
        let (checked, skipped, err) = compare(&name, &pkts, &dump);
        println!("{name}: 核对 {checked} 帧（跳过 {skipped}），能量最大差 {err:.6}");
        total += checked;
        total_skipped += skipped;
        max_err = max_err.max(err);
        vectors += 1;
    }
    assert!(vectors >= 10, "向量太少了：{vectors}");
    assert!(total > 15_000, "核对帧数太少：{total}");
    println!("官方向量合计：{vectors} 组、逐帧一致 {total} 帧（跳过 {total_skipped}），能量最大差 {max_err:.6}");
}
