//! 手工跑：拿一份**真实 `.opus`** 逐包核对解封装器（默认 `#[ignore]`）。
//!
//! 为什么要有这一条：Ogg 页结构、跨页组包、granule→pts 这些在真实文件上才会遇到的问题
//! （`OpusTags` 跨页、末尾 end trimming、一个包跨页……）光靠手造页的单测覆盖不全。
//! 单测不依赖外部文件的约定照旧 —— 这条测试默认跳过，手工跑：
//!
//! ```text
//! OGG_FILE=/tmp/ehren-paper_lights-96.opus \
//!   cargo test -p me-container --test ogg_ref_walk -- --ignored --nocapture
//! ```
//!
//! 素材是 opus-codec.org 的 `ehren-paper_lights-96.opus`（231 页 / 11406 个 20 ms 包），
//! 期望值是拿一份**独立的 Python 解析器**（照 RFC 3533 / RFC 7845 另写的一遍）算出来的：
//! 包数 11406、每包 960 样本、总时长 10,949,760 样本、末页 granule 10,949,476
//! （差 284 = 末尾 trim）。

use std::env;
use std::sync::Arc;

use me_container::OggOpusDemuxer;
use me_core::{FileSource, ReadAt};

#[test]
#[ignore = "手工跑：需要 OGG_FILE 指向真实 .opus（见本文件顶部说明）"]
fn real_opus_walks_packet_by_packet() {
    let path = env::var("OGG_FILE").expect("要设 OGG_FILE 指向一份真实 .opus");
    let src: Arc<dyn ReadAt> = Arc::new(FileSource::open(&path).expect("打开素材"));
    let mut d = OggOpusDemuxer::new(src, &path).expect("解析 Ogg Opus");

    let head = d.opus_head();
    println!(
        "OpusHead: {} 声道 / pre-skip {} / 输入采样率 {} / 增益 Q7.8 {} / 映射族 {}",
        head.channels, head.pre_skip, head.input_sample_rate, head.output_gain_q78, head.mapping_family
    );
    println!("时长 {:?} µs", d.duration_us());

    let mut count = 0u64;
    let mut total = 0u64;
    let mut prev: Option<(i64, u32)> = None;
    let mut longest = 0u32;
    while let Some(p) = d.next_packet().expect("取包") {
        if p.data.is_empty() {
            panic!("第 {count} 个包是空的（规范要求空包按坏包处理）");
        }
        if p.duration == 0 {
            panic!("第 {count} 个包时长为 0");
        }
        if let Some((ppts, pdur)) = prev {
            assert_eq!(
                p.pts,
                ppts + pdur as i64,
                "第 {count} 个包的 pts 与上一个不连续（解码器会以为中间有洞）"
            );
        } else if count == 0 {
            println!("第一个包：pts {} 时长 {} 字节 {}", p.pts, p.duration, p.data.len());
        }
        longest = longest.max(p.duration);
        prev = Some((p.pts, p.duration));
        total += p.duration as u64;
        count += 1;
    }
    let (last_pts, last_dur) = prev.expect("至少要有包");
    println!(
        "共 {count} 个包，时长合计 {total} 样本；末包 pts {} 时长 {}；自算末尾位置 {}",
        last_pts,
        last_dur,
        last_pts + last_dur as i64
    );
    println!("最长包 {longest} 样本（Opus 上限 5760 = 120 ms）");
    assert!(longest <= 5760, "包长超过 Opus 规范上限");

    // 针对这份素材的硬断言（换素材就要跟着改，见文件顶部）
    assert_eq!(count, 11406, "包数与独立 Python 解析器算出来的不一致");
    assert_eq!(total, 10_949_760, "样本总数与独立解析器不一致");
    assert_eq!(d.duration_us(), Some(228_106_666), "可播时长（已扣 pre-skip）");

    // seek：落点必须是"granule ≤ 目标"的那一页页首，且随后的包时间戳与从头解一致
    let landed = d.seek(60_000_000).expect("seek 到 60 s");
    let first = d.next_packet().expect("取包").expect("落点后有包");
    println!("seek(60 s) → 落点 {landed} µs，首包 pts {}", first.pts);
    assert!(landed <= 60_000_000, "落点不能晚于请求位置（Ogg 是按 granule 定位的）");
    assert!(
        (first.pts - landed / 1000 * 48).abs() <= 48,
        "落点与首包时间戳对不上：落点 {landed} µs，首包 pts {}",
        first.pts
    );
    let second = d.next_packet().expect("取包").expect("落点后有第二个包");
    assert_eq!(second.pts, first.pts + first.duration as i64, "落点之后要连续");
}
