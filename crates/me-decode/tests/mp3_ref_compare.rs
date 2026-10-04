//! MP3 解码器的**手工端到端比对**（不带样例文件，默认跑不到）。
//!
//! 为什么做成 `#[ignore]` 的集成测试而不是普通单测：项目约定单测不依赖样例文件与网络，
//! 而"自研解码器解出来对不对"只能拿真实码流与参考解码器的 PCM 比 —— 所以这条链路留成
//! 手工可复现的一步，跑法：
//!
//! ```text
//! MP3_BIT=/tmp/mp3ref/ILL2_layer3.bit \
//! MP3_REF_F32=/tmp/mp3ref/ill2_ref_f32.pcm \
//!   cargo test -p me-decode --test mp3_ref_compare -- --ignored --nocapture
//! ```
//!
//! 参考 PCM 由 `/tmp/mp3ref/mp3refdf`（minimp3，float32 满刻度 ±1.0）产出，
//! 与本解码器一样**不做编码器延迟补偿**，所以可以逐帧对齐。
//!
//! 判据（三条一起看，缺一条都不够）：
//! 1. **逐帧归一化相关** ≈ 1：整段波形是不是同一段音乐；
//! 2. **逐帧幅度比** ≈ 1：`OUTPUT_SCALE` 之类的量纲常数对不对；
//! 3. **逐样本最大绝对差** < 4 个量化步长（本解码器输出 i16，参考是 float32，
//!    理论下限就是 1/32768）—— 这条是硬断言，也是唯一不会被近静音帧搅乱的一条。
//!
//! 近静音帧（参考 rms < −60 dBFS）不参与相关度：那时解码器自己的量化噪声与信号同量级，
//! 相关系数会掉到 0.6 也说明不了任何问题，只看逐样本最大差。

use std::path::PathBuf;

use me_decode::audio::mp3::Mp3Decoder;
use me_decode::audio::AudioDecoder;

/// 读一个 f32 LE 的交织 PCM。
fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("读参考 PCM 失败 {path:?}: {e}"));
    assert!(bytes.len().is_multiple_of(4), "参考 PCM 长度不是 4 的倍数");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// 归一化相关系数（两段长度必须一致）。
fn corr(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        sa += x;
        sb += y;
        saa += x * x;
        sbb += y * y;
        sab += x * y;
    }
    let (ma, mb) = (sa / n as f64, sb / n as f64);
    let cov = sab / n as f64 - ma * mb;
    let (va, vb) = (saa / n as f64 - ma * ma, sbb / n as f64 - mb * mb);
    if va <= 0.0 || vb <= 0.0 {
        return if va <= 0.0 && vb <= 0.0 { 1.0 } else { 0.0 };
    }
    cov / (va.sqrt() * vb.sqrt())
}

#[test]
#[ignore = "手工跑：需要 /tmp/mp3ref 下的码流与参考 PCM（见本文件顶部说明）"]
fn decode_matches_reference_pcm() {
    let bit = PathBuf::from(std::env::var("MP3_BIT").expect("要设 MP3_BIT"));
    let ref_pcm = PathBuf::from(std::env::var("MP3_REF_F32").expect("要设 MP3_REF_F32"));
    let data = std::fs::read(&bit).expect("读码流失败");
    let reference = read_f32(&ref_pcm);

    // 按 1 KiB 一小块喂进去：顺带验证"包里含半帧/多帧"那条路径
    let mut dec = Mp3Decoder::open(&[]).expect("开解码器");
    let mut mine: Vec<f32> = Vec::new();
    let mut frames = 0usize;
    let mut rate = 0u32;
    let mut channels = 0u8;
    for chunk in data.chunks(1024) {
        for pcm in dec.decode(chunk, 0).expect("解码失败") {
            rate = pcm.sample_rate;
            channels = pcm.channels;
            frames += 1;
            mine.extend(pcm.samples.iter().map(|s| *s as f32 / 32768.0));
        }
    }
    for pcm in dec.flush().expect("收尾失败") {
        frames += 1;
        mine.extend(pcm.samples.iter().map(|s| *s as f32 / 32768.0));
    }

    let per_frame = if rate == 0 { 0 } else { (if rate <= 24000 { 576 } else { 1152 }) * channels as usize };
    println!(
        "{}: {} Hz {}ch 解出 {frames} 帧 {} 样本 | 参考 {} 样本（{:.1} 帧）",
        bit.file_name().unwrap().to_string_lossy(),
        rate,
        channels,
        mine.len(),
        reference.len(),
        reference.len() as f64 / per_frame.max(1) as f64
    );
    assert_eq!(dec.stream_info(), (rate, channels, 16), "stream_info 要跟首帧一致");
    assert!(per_frame > 0, "一帧都没解出来");

    // 逐帧对齐比对（两边的帧长一样，直接按帧索引切）
    // 注意：`per_frame` 取的是**最后一帧**的声道数。l3-he_mode 这类流逐帧切换声道模式
    // （单声道/立体声/联合/双声道都有），用固定的 per_frame 切会把帧边界切歪 ——
    // 不过因为是按同一套下标同时切两边，逐样本差仍然是逐样本差，判据不受影响。
    let n_frames = mine.len().min(reference.len()) / per_frame;
    let mut worst = (0usize, 1.0f64, 1.0f64);
    let mut sum_mine = 0.0f64;
    let mut sum_ref = 0.0f64;
    let mut max_diff = 0.0f32;
    let mut good = 0usize;
    let mut quiet = 0usize;
    for f in 0..n_frames {
        let s = f * per_frame;
        let a = &mine[s..s + per_frame];
        let b = &reference[s..s + per_frame];
        let c = corr(a, b);
        let (ra, rb) = (rms(a), rms(b));
        println!("   帧 {f:3}: corr={c:+.6} 我的 rms={ra:.5} 参考 rms={rb:.5}");
        // 逐样本最大差对**所有**帧统计（包括近静音帧）：任何一处解错都会立刻顶破它。
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        sum_mine += ra;
        sum_ref += rb;
        // 近静音的帧（编码器填充/静音段）不给相关度指标：参考 rms 掉到 -60 dBFS 以下时，
        // 本解码器自己的 i16 量化噪声已经与信号同量级，相关系数会掉到 0.6~0.99 却毫无意义
        // （实际差异仍是半个量化步长，见上面那条最大差统计）。这种帧只要求两边都接近静音。
        if rb < 1e-3 {
            quiet += 1;
            if ra < 2e-3 {
                good += 1;
            }
            continue;
        }
        if c > 0.999 {
            good += 1;
        }
        if c < worst.1 {
            worst = (f, c, if rb > 0.0 { ra / rb } else { 0.0 });
        }
    }
    let ratio = if sum_ref > 0.0 { sum_mine / sum_ref } else { 0.0 };
    println!(
        "帧 {n_frames}（静音 {quiet}）| 相关 > 0.999 的帧 {good}/{n_frames} | 最差帧 #{} corr={:.6} 幅度比={:.4}",
        worst.0, worst.1, worst.2
    );
    println!("整体幅度比 = {ratio:.6} | 逐样本最大绝对差 = {max_diff:.3e}");

    assert!(
        good as f64 / n_frames.max(1) as f64 > 0.9,
        "{} 帧里只有 {good} 帧与参考一致（最差第 {} 帧 corr={:.6}）",
        n_frames,
        worst.0,
        worst.1
    );
    assert!(
        (ratio - 1.0).abs() < 0.02,
        "整体幅度比 {ratio:.6} 偏离 1 太多（OUTPUT_SCALE 或某个增益算错了）"
    );
    // 最硬的一条：本解码器输出 i16，参考是 float32，所以逐样本差应当只在**一个量化步长**
    // 量级（1/32768 = 3.05e-5）。留到 4 个步长是给"参考自己也有 ~1e-6 近似误差 +
    // 我的 f32 运算顺序"留余量。任何结构性错误（少做一次混叠抑制、强度区多吞一带、
    // 哈夫曼错一位……）都会直接顶破这条 —— 而且它不像相关度那样会被近静音帧搅乱。
    assert!(
        max_diff < 4.0 / 32768.0,
        "逐样本最大差 {max_diff:.3e} 超过了 4 个 i16 量化步长（{:.3e}）—— 有地方算错了",
        4.0 / 32768.0
    );
}

fn rms(v: &[f32]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    (v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / v.len() as f64).sqrt()
}

/// 把每帧的中间量（反量化后的谱、频率反转后的子带样本）追加写到 `ME_MP3_DUMP` 目录，
/// 用于和参考解码器的同一阶段逐值对拍 —— 定位"从哪一步开始不一样"。
#[test]
#[ignore = "手工跑：需要 MP3_BIT 与 ME_MP3_DUMP（详见本文件顶部）"]
fn dump_intermediate_stages() {
    let bit = PathBuf::from(std::env::var("MP3_BIT").expect("要设 MP3_BIT"));
    assert!(
        std::env::var("ME_MP3_DUMP").is_ok(),
        "要设 ME_MP3_DUMP=<目录>，中间量才会写出去"
    );
    let data = std::fs::read(&bit).expect("读码流失败");
    let mut dec = Mp3Decoder::open(&[]).expect("开解码器");
    let mut frames = 0usize;
    let mut pcm_out: Vec<u8> = Vec::new();
    for chunk in data.chunks(4096) {
        for pcm in dec.decode(chunk, 0).expect("解码失败") {
            frames += 1;
            for s in &pcm.samples {
                pcm_out.extend_from_slice(&(*s as f32 / 32768.0).to_le_bytes());
            }
        }
    }
    if let Ok(dir) = std::env::var("ME_MP3_DUMP") {
        std::fs::write(format!("{dir}/my_pcm.pcm"), &pcm_out).expect("写 PCM 失败");
    }
    println!("{bit:?} 解出 {frames} 帧，中间量已追加写出");
}
