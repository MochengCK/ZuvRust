//! 手工跑：把自研熵层与**参考实现**对拍（默认 `#[ignore]`）。
//!
//! 两条互补的路子：
//!
//! 1. `range_decoder_matches_reference_encoder` —— 用参考实现的**编码器**
//!    （`entenc.c`，通过 `/tmp/opus_ref_encprobe.c` 那个小探针）编码一段混合了
//!    icdf / bit_logp / 均匀整数 / 原始比特的符号序列，再让自研解码器解回来，
//!    **逐符号**比对（含每个符号之后的 `tell_frac`）。这条覆盖的是"真实码流暂时
//!    覆盖不到"的那几条路径（CELT/SILK 还没接，所以真实包里现在只能走到第一个符号）。
//! 2. `silence_flag_and_tell_match_reference_on_real_packets` —— 真实 `.opus` 的
//!    每个音频包：第一个符号是 CELT 的 silence 标志（`{32767,1}/32768` = `logp 15`），
//!    解完它之后的 `tell` / `tell_frac` 必须与参考实现**逐位一致**
//!    （规范 §4.1.6：这两个值必须与编码器相同，比特分配全靠它）。
//!
//! 跑法（素材见两个测试各自的说明）：
//!
//! ```text
//! OPUS_ENC_BIN=/tmp/opus_enc.bin OPUS_ENC_LOG=/tmp/opus_enc_log.txt \
//! OPUS_PKTS=/tmp/opus_pkts.bin OPUS_REF_DUMP=/tmp/opus_ref_dump.txt \
//!   cargo test -p me-decode --test opus_entropy_ref -- --ignored --nocapture
//! ```

use std::env;

use me_decode::audio::opus::RangeDecoder;

/// 参考编码器用的那张表：{56, 50, 90, 60}/256 → icdf = {200, 150, 60, 0}
const REF_ICDF: [u8; 4] = [200, 150, 60, 0];

#[test]
#[ignore = "手工跑：先编译 /tmp/opus_ref_encprobe.c 生成素材（见文件顶部说明）"]
fn range_decoder_matches_reference_encoder() {
    let bin = env::var("OPUS_ENC_BIN").expect("要设 OPUS_ENC_BIN（参考编码器产出的字节）");
    let log = env::var("OPUS_ENC_LOG").expect("要设 OPUS_ENC_LOG（符号日志）");
    let bytes = std::fs::read(&bin).expect("读编码结果");
    let text = std::fs::read_to_string(&log).expect("读符号日志");
    let mut d = RangeDecoder::new(&bytes);
    let mut n = 0usize;
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f[0] {
            "icdf" => {
                let want: u32 = f[1].parse().unwrap();
                let got = d.dec_icdf(&REF_ICDF, 8).expect("解 icdf 符号");
                assert_eq!(got, want, "第 {n} 个符号（icdf）不一致");
            }
            "logp" => {
                let logp: u32 = f[1].parse().unwrap();
                let want: u32 = f[2].parse().unwrap();
                assert_eq!(d.dec_bit_logp(logp), want, "第 {n} 个符号（bit_logp）不一致");
            }
            "uint" => {
                let ft: u32 = f[1].parse().unwrap();
                let want: u32 = f[2].parse().unwrap();
                assert_eq!(d.dec_uint(ft).expect("解均匀整数"), want, "第 {n} 个符号（uint）不一致");
            }
            "bits" => {
                let bits: u32 = f[1].parse().unwrap();
                let want: u32 = f[2].parse().unwrap();
                assert_eq!(d.dec_bits(bits).expect("解原始比特"), want, "第 {n} 个符号（bits）不一致");
            }
            other => panic!("日志里有不认识的行：{other}"),
        }
        // 每解完一个符号，与编码器的 tell_frac 对一次（规范要求必须逐位一致）
        let want_frac: u32 = f[f.len() - 1].parse().unwrap();
        assert_eq!(
            d.tell_frac(),
            want_frac,
            "第 {n} 个符号（{}）之后的 tell_frac 不一致",
            f[0]
        );
        n += 1;
    }
    println!("与参考编码器逐符号一致：{n} 个符号，编码结果 {} 字节", bytes.len());
}

#[test]
#[ignore = "手工跑：需要 OPUS_PKTS 与 OPUS_REF_DUMP（见文件顶部说明）"]
fn silence_flag_and_tell_match_reference_on_real_packets() {
    let pkts_path = env::var("OPUS_PKTS").expect("要设 OPUS_PKTS（长度前缀的包序列）");
    let dump_path = env::var("OPUS_REF_DUMP").expect("要设 OPUS_REF_DUMP（参考实现的熵层 dump）");
    let pkts = std::fs::read(&pkts_path).expect("读包文件");
    let dump = std::fs::read_to_string(&dump_path).expect("读参考 dump");

    let mut pos = 0usize;
    let mut idx = 0usize;
    let mut silences = 0usize;
    let mut max_tell = 0i32;
    for line in dump.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f.len(), 5, "参考 dump 每行应当是 5 列：{line}");
        let len: usize = f[1].parse().unwrap();
        let want_silence: u32 = f[2].parse().unwrap();
        let want_tell: i32 = f[3].parse().unwrap();
        let want_frac: u32 = f[4].parse().unwrap();
        assert!(pos + 4 + len <= pkts.len(), "包文件比 dump 短（第 {idx} 个包）");
        let chunk_len = u32::from_le_bytes([pkts[pos], pkts[pos + 1], pkts[pos + 2], pkts[pos + 3]]) as usize;
        assert_eq!(chunk_len, len, "第 {idx} 个包的长度与 dump 不一致");
        pos += 4;
        let pkt = &pkts[pos..pos + len];
        pos += len;

        let mut d = RangeDecoder::new(pkt);
        // 每个 CELT 帧的第一个符号就是 silence 标志
        let silence = d.dec_bit_logp(15);
        assert_eq!(silence, want_silence, "第 {idx} 个包的 silence 标志不一致");
        assert_eq!(d.tell(), want_tell, "第 {idx} 个包的 tell 不一致");
        assert_eq!(d.tell_frac(), want_frac, "第 {idx} 个包的 tell_frac 不一致");
        silences += silence as usize;
        max_tell = max_tell.max(want_tell);
        idx += 1;
    }
    assert!(idx > 1000, "包数太少（{idx}），素材是不是不对？");
    println!(
        "与参考实现逐包一致：{idx} 个真实包；其中 silence=1 的有 {silences} 个；tell 最大 {max_tell}"
    );
}
