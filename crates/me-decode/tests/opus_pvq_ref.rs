//! 手工跑：把自研的 PVQ 组合数解码与**参考实现**逐用例对拍（默认 `#[ignore]`）。
//!
//! 素材由 `/tmp/opus_ref_pvqprobe.c` 生成（用参考实现的编码器 `encode_pulses` 造码流，
//! 再用**参考自己的** `decode_pulses` 解出期望矢量）—— 这样比的是"两个解码器"，
//! 不牵扯编码器的规范化约定。覆盖面：n ∈ {2,3,4,5,8,16,17,32,60,100,176} × k 各档 × 5 组。
//!
//! 跑法：
//!
//! ```text
//! OPUS_PVQ_CASES=/tmp/opus_pvq_cases.txt \
//!   cargo test -p me-decode --test opus_pvq_ref -- --ignored --nocapture
//! ```

use std::env;

use me_decode::audio::opus::{cwrs, RangeDecoder};

#[test]
#[ignore = "手工跑：需要 OPUS_PVQ_CASES（见文件顶部说明）"]
fn pvq_decoding_matches_the_reference() {
    let path = env::var("OPUS_PVQ_CASES").expect("要设 OPUS_PVQ_CASES");
    let text = std::fs::read_to_string(&path).expect("读用例文件");
    let mut cases = 0usize;
    let mut max_n = 0usize;
    let mut max_k = 0usize;
    for line in text.lines() {
        let mut it = line.split(' ');
        let n: usize = it.next().unwrap().parse().unwrap();
        let k: usize = it.next().unwrap().parse().unwrap();
        let nbytes: usize = it.next().unwrap().parse().unwrap();
        let hex = it.next().unwrap();
        let want: Vec<i32> = it
            .take(n)
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(want.len(), n, "用例里的期望矢量长度不对：{line}");
        let stream: Vec<u8> = (0..nbytes)
            .map(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap())
            .collect();

        // V(n,k) 也要对得上（它决定 dec_uint 的 ft，错了序号就全错）
        let (mut u, count) = cwrs::u_row(n, k);
        let mut dec = RangeDecoder::new(&stream);
        let idx = dec
            .dec_uint(count)
            .unwrap_or_else(|e| panic!("n={n} k={k} 第 {cases} 个用例解序号失败：{e}"));
        let got = cwrs::cwrsi(n, k, idx, &mut u).expect("还原矢量");
        if got != want {
            eprintln!("失败用例：n={n} k={k} V={count} 流长={} 字节", stream.len());
            eprintln!("  hex = {hex}");
            eprintln!("  我解出的序号 = {idx}");
            eprintln!("  我解出 = {got:?}");
            eprintln!("  参考解 = {want:?}");
        }
        assert_eq!(got, want, "n={n} k={k} 的第 {cases} 个用例与参考不一致");
        cases += 1;
        max_n = max_n.max(n);
        max_k = max_k.max(k);
    }
    assert!(cases > 1000, "用例太少：{cases}");
    println!("与参考实现一致：{cases} 个用例（n 最大 {max_n}，k 最大 {max_k}）");
}
