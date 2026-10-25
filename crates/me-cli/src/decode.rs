//! `decode` 子命令：把输入的视频轨**解成帧**。
//!
//! 这是播放内核的一块（不是给人用的"播放器"）：`play` 之后会把"解码 + 上屏 +
//! 时钟"串起来，而这一条只回答"引擎自己能不能把 H.264 解成画面"。
//! 之所以先把它单独做出来：解码是整条链里最容易悄悄错的一环（时间戳、参数集、
//! 像素格式），单独跑一遍能一眼看出对错。
//!
//! 输出协议（与其它子命令一致：stdout 只出 JSON 行，字节产物单独落盘）：
//!
//! ```text
//! {"ok":true,"type":"decode-meta","data":{codec,width,height,backend}}
//! {"ok":true,"type":"decode-progress","data":{frames,lastPtsUs}}
//! {"ok":true,"result":{frames,width,height,backend,elapsedMs,outDir}}
//! ```
//!
//! `--out=<目录>` 时每一帧落一个 `<目录>/frame-000001.nv12`（紧凑 NV12），
//! 旁边一个 `frames.json` 记每帧的时间戳与大小 —— 校验解码对不对就靠它。

#[cfg(feature = "video")]
use std::io::Write;
use std::time::Instant;

#[cfg(feature = "video")]
use me_core::CodecId;
use me_core::{MediaError, MediaKind, Result};
#[cfg(feature = "video")]
use me_mux::convert;
use me_stream::{OpenOptions, Player, PlayerConfig};

use crate::json;

/// 真实素材的验收（手工跑，见文件末尾的 `real_av1_tests`）。
#[cfg(all(test, feature = "video"))]
mod real_av1_tests {
    use super::*;
    use me_decode::video::context::CdfContext;
    use me_decode::video::entropy::{required_consumed_bits, SymbolDecoder};
    use me_decode::video::intra::CurrFrame;
    use me_decode::video::frame_header::{
        FrameHeader, FrameType, HeaderTail, ObuLayerIds, RefFrameStore, PRIMARY_REF_NONE,
    };
    use me_decode::video::obu::{ObuIter, ObuType};
    use me_decode::video::block::{decode_tile, BlockGrid};
    use me_decode::video::coeff::CoeffContexts;
    use me_decode::video::mv::{
        motion_field_estimation, motion_field_storage, MotionField, MotionFieldStore,
    };
    use me_decode::video::partition::{decode_tile_partitions, MiGrid};
    use me_decode::video::sequence_header::SequenceHeader;
    use me_decode::video::tile::parse_tile_group;

    /// 打开一个输入并返回（播放器, 视频轨 id, 解码器初始化数据）。
    fn open_video(input: &str) -> (Player, u32, Vec<u8>) {
        let client = me_stream::default_client().unwrap();
        let opened = me_stream::open_input(&client, input, &OpenOptions::default()).unwrap();
        let (src, _note) = crate::source::pick_source(opened).unwrap();
        let cfg = PlayerConfig { max_buffer_us: 1_000_000, max_buffer_packets: 1024 };
        let player = Player::open(src, input, cfg).unwrap();
        let (track_id, config) = {
            let info = player.info();
            let (track, config) = video_track_and_config(&info, input).unwrap();
            assert_eq!(track.codec, CodecId::Av1, "这份素材的视频轨应当是 AV1");
            (track.id, config)
        };
        (player, track_id, config)
    }

    fn real_file() -> String {
        std::env::var("AV1_REAL_FILE").expect("请设置 AV1_REAL_FILE")
    }

    fn read_bit_at(payload: &[u8], pos: &mut usize) -> std::result::Result<u32, String> {
        if *pos / 8 >= payload.len() {
            return Err("读到载荷之外".into());
        }
        let b = (payload[*pos / 8] >> (7 - (*pos % 8))) & 1;
        *pos += 1;
        Ok(b as u32)
    }

    fn read_bits_at(
        payload: &[u8],
        pos: &mut usize,
        n: u32,
    ) -> std::result::Result<u32, String> {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | read_bit_at(payload, pos)?;
        }
        Ok(v)
    }

    /// 校验一个 `OBU_FRAME` 的**瓦片组**与 `tile_info()` 自洽（规范 §5.9.3）。
    ///
    /// 这段语法值得单独校：它一边用 `tile_info()` 算出来的
    /// `TileCols` / `TileRows` / `TileSizeBytes`，一边把整段瓦片数据按"每个瓦片
    /// 声明的长度"切开 —— **所有瓦片加起来必须正好用完全部数据**。
    /// 位读错一位（网格、尺寸字段宽度、字节对齐）这条就会崩，所以它是
    /// "tile_info 真的读对了"的直接证据（比"值看起来合理"硬得多）。
    ///
    /// `me-decode` 的位读器是 crate 内部的，这里手写一个够用的小读器：
    /// tile group 头部在字节对齐前最多 `1 + 2 * (cols_log2 + rows_log2)` 位。
    fn check_tile_group(
        payload: &[u8],
        header_bytes: usize,
        ti: &me_decode::video::params::TileInfo,
    ) -> std::result::Result<(), String> {
        let mut bit_pos = header_bytes * 8;
        let num_tiles = ti.num_tiles();
        let (tg_start, tg_end) = if num_tiles > 1 && read_bit_at(payload, &mut bit_pos)? == 1 {
            let bits = ti.cols_log2 + ti.rows_log2;
            let s = read_bits_at(payload, &mut bit_pos, bits)?;
            let e = read_bits_at(payload, &mut bit_pos, bits)?;
            (s, e)
        } else {
            (0, num_tiles - 1)
        };
        if tg_start > tg_end || tg_end >= num_tiles {
            return Err(format!(
                "tg_start={tg_start}、tg_end={tg_end} 越界（共 {num_tiles} 个瓦片）"
            ));
        }
        while bit_pos % 8 != 0 {
            if read_bit_at(payload, &mut bit_pos)? != 0 {
                return Err("tile group 头部的字节对齐位不是 0".into());
            }
        }
        let mut pos = bit_pos / 8;
        if pos >= payload.len() {
            return Err("tile group 头部就把这段数据吃完了".into());
        }
        let mut left = payload.len() - pos;
        let size_bytes = ti.tile_size_bytes as usize;
        for tile in tg_start..=tg_end {
            if tile == tg_end {
                if left == 0 {
                    return Err("末瓦片拿到的长度是 0".into());
                }
                pos += left;
                left = 0;
                break;
            }
            if size_bytes == 0 {
                return Err("有多个瓦片却 LengthSize = 0（tile_size_bytes 读错了？）".into());
            }
            if left < size_bytes + 1 {
                return Err(format!("第 {tile} 个瓦片的尺寸字段越界（只剩 {left} 字节）"));
            }
            let mut size = 0usize;
            for _ in 0..size_bytes {
                size = (size << 8) | payload[pos] as usize;
                pos += 1;
            }
            let tile_size = size + 1;
            if left < size_bytes + tile_size {
                return Err(format!(
                    "第 {tile} 个瓦片声明 {tile_size} 字节，但只剩 {} 字节",
                    left - size_bytes
                ));
            }
            left -= size_bytes + tile_size;
            pos += tile_size;
        }
        if left != 0 || pos != payload.len() {
            return Err(format!(
                "瓦片数据没用完（游标 {pos}、剩 {left}、总长 {}）",
                payload.len()
            ));
        }
        Ok(())
    }

    /// 真实 AV1 素材：把**整条视频轨的每一个采样**都走一遍 OBU/序列头解析。
    ///
    /// ```text
    /// AV1_REAL_FILE=<真实 AV1 文件> \
    ///   cargo test -p me-cli --bin zuvrust -- --ignored --nocapture
    /// ```
    ///
    /// 为什么需要它：合成样本只能证明"位序自洽"（写测试的人与写解析器的人出自
    /// 同一份理解，会一起错 —— 第一版解析器就是这样少读了一位，把 1920×1080
    /// 8bit 4:2:0 解成 "10bit mono、6×1792"）。真实码流才会暴露多字节 leb128、
    /// 瓦片分块、OBU 序列差异这些形态。
    #[test]
    #[ignore = "手工跑：需要 AV1_REAL_FILE 指向一份真实 AV1 文件"]
    fn real_av1_file_walks_every_sample() {
        let path = real_file();
        let (mut player, track_id, config) = open_video(&path);
        let record = me_decode::video::av1::Av1Config::parse(&config).unwrap();
        println!(
            "av1C 记录：profile {} level {} · {}bit{}",
            record.seq_profile,
            record.seq_level_idx,
            record.bit_depth,
            if record.subsampling_x && record.subsampling_y { " 4:2:0" } else { "" }
        );

        let mut samples = 0u64;
        let mut with_frame = 0u64;
        let mut with_seq_header = 0u64;
        let mut first_seq = None;
        while let Some(pkt) = player.next_packet().unwrap() {
            if pkt.track_id != track_id {
                continue;
            }
            samples += 1;
            let info = me_decode::video::av1::inspect_sample(&pkt.data)
                .unwrap_or_else(|e| panic!("第 {samples} 个采样切不出 OBU：{e}"));
            if info.has_bitstream {
                with_frame += 1;
            }
            if let Some(sh) = info.sequence_header {
                with_seq_header += 1;
                if first_seq.is_none() {
                    first_seq = Some(sh);
                }
            }
        }
        println!("采样 {samples}、含帧/瓦片 {with_frame}、带序列头 {with_seq_header}");

        assert!(samples > 100, "样本太少（{samples}），是不是读错了轨");
        assert!(with_frame * 10 >= samples * 9, "至少 90% 的采样该带帧/瓦片数据");
        let sh = first_seq.expect("整条轨里应当至少有一个序列头");
        // 序列头与容器里的 av1C 记录必须一致（不一致说明解析器或容器有一边错了）
        assert_eq!(sh.seq_profile, record.seq_profile, "profile 两边不一致");
        assert_eq!(sh.color.bit_depth, record.bit_depth, "位深两边不一致");
        assert_eq!(sh.color.subsampling_x, record.subsampling_x);
        assert_eq!(sh.color.subsampling_y, record.subsampling_y);
        // 这个属性是这一轨自己的（真实素材是 1080p）
        assert_eq!(sh.max_frame_width, 1920);
        assert_eq!(sh.max_frame_height, 1080);
        println!(
            "序列头：{} · 最大 {}×{}",
            sh.pixel_format_text(),
            sh.max_frame_width,
            sh.max_frame_height
        );
    }

    /// 真实 AV1 素材：**逐帧解析非压缩头**（M2），把硬不变量钉死。
    ///
    /// 这些不变量都不是"能自洽就算过"，而是外部世界的约束：
    /// - 关键帧必须与序列头同时出现（编码器就是这样打包的）；
    /// - 每帧的尺寸都必须是这一轨的尺寸（否则 frame_size 系列读错了）；
    /// - inter 帧的 order_hint 必须**逐帧 +1**（模 2^order_hint_bits）——
    ///   这是"位真的读对了"最有说服力的证据：错一位这里立刻乱掉。
    #[test]
    #[ignore = "手工跑：需要 AV1_REAL_FILE 指向一份真实 AV1 文件"]
    fn real_av1_file_frame_headers_are_consistent() {
        let path = real_file();
        let (mut player, track_id, _config) = open_video(&path);

        let mut seq: Option<SequenceHeader> = None;
        let mut store = RefFrameStore::new();
        let mut frames = 0u64;
        let mut keyframes = 0u64;
        let mut inter_frames = 0u64;
        let mut show_existing = 0u64;
        let mut hidden = 0u64;
        let mut samples_with_seq_header = 0u64;
        // "新解码并被显示"的帧：这些帧的 order_hint 应当严格 +1（模 2^bits）
        let mut displayed_new = 0u64;
        let mut displayed_hint_jumps = 0u32;
        // 编码器在关键帧上重置显示计数（合法，见下面的断言）
        let mut hint_resets = 0u32;
        let mut last_displayed_hint: Option<u8> = None;
        let mut max_header_bytes = 0usize;
        let mut first_frames: Vec<String> = Vec::new();
        let mut jumps: Vec<String> = Vec::new();
        let mut obu_types = std::collections::BTreeMap::new();
        // ── M2.5 / M2.6（非压缩头的参数段）的统计 ──
        let mut tile_grid: Option<(u32, u32, u32)> = None;
        let mut tile_grid_mismatch = 0u32;
        let mut qidx_min = 255u8;
        let mut qidx_max = 0u8;
        let mut lossless_frames = 0u64;
        let mut lr_frames = 0u64;
        let mut grain_frames = 0u64;
        let mut ref_select_frames = 0u64;
        let mut skip_mode_frames = 0u64;
        let mut tx_modes: std::collections::BTreeMap<u8, u64> = Default::default();
        let mut cdef_dampings: std::collections::BTreeMap<u8, u64> = Default::default();
        let mut header_bits_min = usize::MAX;
        let mut header_bits_max = 0usize;
        let mut frame_obus = 0u64;
        let mut frame_obu_with_data = 0u64;
        let mut frame_header_obus = 0u64;
        let mut frame_header_obus_exact = 0u64;
        let mut tile_group_bad = 0u64;
        let mut tile_group_errors: Vec<String> = Vec::new();

        while let Some(pkt) = player.next_packet().unwrap() {
            if pkt.track_id != track_id {
                continue;
            }
            let mut has_seq_header = false;
            let mut iter = ObuIter::new(&pkt.data);
            while let Some(obu) = iter.next_obu().expect("OBU 走不动了") {
                *obu_types.entry(obu.header.obu_type.name()).or_insert(0u32) += 1;
                match obu.header.obu_type {
                    ObuType::SequenceHeader => {
                        has_seq_header = true;
                        seq = Some(SequenceHeader::parse(obu.payload).expect("序列头解析失败"));
                    }
                    ObuType::Frame | ObuType::FrameHeader => {
                        let seq = seq.as_ref().expect("帧之前必须先有序列头");
                        let ids = ObuLayerIds {
                            temporal_id: obu.header.temporal_id,
                            spatial_id: obu.header.spatial_id,
                        };
                        // 两种 OBU 的收尾不同：帧头 OBU 用 `trailing_bits`，
                        // 帧 OBU 用 `byte_alignment`（后面还接瓦片组）
                        let tail = match obu.header.obu_type {
                            ObuType::FrameHeader => HeaderTail::TrailingBits,
                            _ => HeaderTail::ByteAlignment,
                        };
                        let fh = FrameHeader::parse(obu.payload, seq, &mut store, ids, tail)
                            .unwrap_or_else(|e| {
                                panic!(
                                    "第 {} 帧的帧头解析失败：{e}\n  OBU 类型 {:?}、载荷 {} 字节：{}",
                                    frames + 1,
                                    obu.header.obu_type,
                                    obu.payload.len(),
                                    obu.payload
                                        .iter()
                                        .take(32)
                                        .map(|b| format!("{b:02x}"))
                                        .collect::<Vec<_>>()
                                        .join(" ")
                                )
                            });
                        frames += 1;
                        max_header_bytes = max_header_bytes.max(fh.bits_consumed.div_ceil(8));
                        assert!(
                            fh.bits_consumed <= obu.payload.len() * 8,
                            "非压缩头读过了载荷（{} 位 > {} 位）",
                            fh.bits_consumed,
                            obu.payload.len() * 8
                        );
                        // 尺寸：这一轨是 1080p 的定尺寸流（帧间帧的尺寸是从参考帧继承的）
                        assert_eq!(fh.upscaled_width, 1920, "第 {frames} 帧宽度不对");
                        assert_eq!(fh.frame_height, 1080, "第 {frames} 帧高度不对");
                        assert_eq!(fh.render_width, 1920, "第 {frames} 帧渲染宽度不对");
                        assert_eq!(fh.render_height, 1080, "第 {frames} 帧渲染高度不对");
                        match fh.frame_type {
                            FrameType::Key => {
                                keyframes += 1;
                                assert_eq!(fh.refresh_frame_flags, 0xFF, "关键帧该刷新全部参考槽");
                                assert!(fh.is_intra && fh.error_resilient_mode);
                            }
                            FrameType::Inter => {
                                inter_frames += 1;
                                assert!(!fh.is_intra);
                            }
                            other => panic!("这条流的帧类型出现了 {other:?}（不该有）"),
                        }
                        // 显示顺序：`show_frame = 1` 的帧与被 `show_existing_frame` 显示的帧
                        // 合起来才是"观众看到的顺序"，它们的 order_hint 必须严格 0,1,2,…（模 128）
                        if fh.show_existing_frame {
                            show_existing += 1;
                        } else if !fh.show_frame {
                            hidden += 1;
                        } else {
                            displayed_new += 1;
                        }
                        if fh.show_existing_frame || fh.show_frame {
                            if let Some(prev) = last_displayed_hint {
                                let expect = (prev + 1) & 0x7F;
                                if fh.order_hint != expect {
                                    // 允许"编码器自己在关键帧上重置计数"这一种情况
                                    // （真实素材每 10896 帧重置一次：prev=31 → 0）
                                    if fh.frame_type == FrameType::Key && fh.order_hint == 0 {
                                        hint_resets += 1;
                                    } else {
                                        displayed_hint_jumps += 1;
                                        if jumps.len() < 8 {
                                            jumps.push(format!(
                                                "第 {frames} 帧：{} prev={prev} → {}",
                                                fh.frame_type.name(),
                                                fh.order_hint
                                            ));
                                        }
                                    }
                                }
                            }
                            last_displayed_hint = Some(fh.order_hint);
                        }
                        if first_frames.len() < 24 {
                            first_frames.push(format!(
                                "#{frames} {}{} hint={} refresh={:#04x} override={} from_ref={} hdr={}位",
                                fh.frame_type.name(),
                                if fh.show_existing_frame {
                                    " show-existing"
                                } else if fh.show_frame {
                                    " show"
                                } else {
                                    " hidden"
                                },
                                fh.order_hint,
                                fh.refresh_frame_flags,
                                fh.frame_size_override,
                                fh.size_from_ref,
                                fh.bits_consumed
                            ));
                        }
                        // ── 参数段（tile_info → film_grain）的检查 ──
                        // 解析器会在末尾校验 byte_alignment，所以"能解析成功"本身
                        // 就已经是位长正确的证据；下面再叠一层"值必须自洽"。
                        let is_frame_obu = obu.header.obu_type == ObuType::Frame;
                        let p = &fh.params;
                        header_bits_min = header_bits_min.min(fh.bits_consumed);
                        header_bits_max = header_bits_max.max(fh.bits_consumed);
                        if is_frame_obu {
                            frame_obus += 1;
                            if fh.payload_bits_after_header > 0 {
                                frame_obu_with_data += 1;
                            }
                        } else {
                            frame_header_obus += 1;
                            if fh.payload_bits_after_header == 0 {
                                frame_header_obus_exact += 1;
                            }
                        }
                        if !fh.show_existing_frame {
                            // show_existing_frame 的帧没有参数段（规范读到那里就 return 了）
                            let grid = (p.tile_info.cols, p.tile_info.rows, p.tile_info.tile_size_bytes);
                            match tile_grid {
                                None => tile_grid = Some(grid),
                                Some(g) if g != grid => tile_grid_mismatch += 1,
                                _ => {}
                            }
                            qidx_min = qidx_min.min(p.quant.base_q_idx);
                            qidx_max = qidx_max.max(p.quant.base_q_idx);
                            if p.all_lossless {
                                lossless_frames += 1;
                            }
                            if p.lr.uses_lr {
                                lr_frames += 1;
                            }
                            if p.film_grain.apply_grain {
                                grain_frames += 1;
                            }
                            if p.reference_select {
                                ref_select_frames += 1;
                            }
                            if p.skip_mode_present {
                                skip_mode_frames += 1;
                            }
                            *tx_modes.entry(p.tx_mode).or_insert(0) += 1;
                            *cdef_dampings.entry(p.cdef.damping).or_insert(0) += 1;
                            // 瓦片组必须与 tile_info 自洽（show_existing 帧没有瓦片数据）
                            if is_frame_obu {
                                if let Err(msg) =
                                    check_tile_group(obu.payload, fh.bits_consumed / 8, &p.tile_info)
                                {
                                    tile_group_bad += 1;
                                    if tile_group_errors.len() < 5 {
                                        tile_group_errors.push(format!("第 {frames} 帧：{msg}"));
                                    }
                                }
                            }
                        }
                        store.refresh(&fh);
                    }
                    _ => {}
                }
            }
            if has_seq_header {
                samples_with_seq_header += 1;
            }
        }

        println!(
            "帧 {frames}（关键帧 {keyframes}、inter {inter_frames}；新解码并显示 {displayed_new}、\
             隐藏 {hidden}、show_existing {show_existing}）"
        );
        println!("带序列头的采样 {samples_with_seq_header}；非压缩头最长 {max_header_bytes} 字节");
        println!("编码器在关键帧上重置显示计数 {hint_resets} 次");
        println!(
            "显示顺序（show_frame + show_existing 合起来 {displayed} 帧）的 order_hint 不连续处 {displayed_hint_jumps}",
            displayed = displayed_new + show_existing
        );
        println!("OBU 类型分布 {obu_types:?}");
        println!("非压缩头长度 {header_bits_min}..{header_bits_max} 位（含 byte_alignment）");
        println!("瓦片网格（列/行/尺寸字段宽度）{tile_grid:?}，不一致 {tile_grid_mismatch} 处");
        println!(
            "base_q_idx {qidx_min}..{qidx_max}；无损帧 {lossless_frames}；用环路恢复 {lr_frames}；\
             胶片颗粒 {grain_frames}；reference_select {ref_select_frames}；skip_mode {skip_mode_frames}"
        );
        println!("tx_mode 分布 {tx_modes:?}；CDEF 阻尼分布 {cdef_dampings:?}");
        println!(
            "OBU_FRAME {frame_obus}（非压缩头之后有瓦片数据 {frame_obu_with_data}）；\
             OBU_FRAME_HEADER {frame_header_obus}（载荷恰好用完 {frame_header_obus_exact}）"
        );
        println!("瓦片组与 tile_info 不自洽 {tile_group_bad} 处");
        for e in &tile_group_errors {
            println!("  {e}");
        }
        for j in &jumps {
            println!("  不连续：{j}");
        }
        println!("前几帧：");
        for line in &first_frames {
            println!("  {line}");
        }

        assert!(frames > 100, "帧太少（{frames}）");
        assert!(keyframes > 0 && inter_frames > 0, "关键帧与 inter 帧都该有");
        // 关键帧与序列头是同一个采样里的东西（编码器的打包习惯）
        assert_eq!(
            samples_with_seq_header, keyframes,
            "带序列头的采样数应当等于关键帧数（{samples_with_seq_header} vs {keyframes}）"
        );
        // **最强的一条**：观众看到的顺序（显示帧 + show_existing 显示的帧）必须是
        // order_hint 严格 +1（模 128）。这条流是分层 B 帧金字塔（隐藏帧拿未来的 hint，
        // 再靠 show_existing 显示出来），只要位读错一位，这里立刻乱掉。
        assert!(displayed_new > 0 && show_existing > 0, "应当既有新显示帧也有 show_existing");
        // 显示顺序必须严格 +1（模 128），**唯一**允许的例外是编码器自己在关键帧上
        // 把计数重置为 0（真实素材每 10896 帧一次，严格周期）
        assert_eq!(
            displayed_hint_jumps, 0,
            "显示顺序的 order_hint 必须严格 +1（不连续 {displayed_hint_jumps} 处 / {} 帧）",
            displayed_new + show_existing
        );
        assert!(hint_resets > 0, "这条流里有周期性的关键帧重置（{hint_resets} 次）");
        // 隐藏帧是"以后才显示"的帧：它们的 hint 不参与显示顺序，但必须落在合法范围
        assert!(hidden > 0, "这条流应当有隐藏帧（金字塔结构）");

        // ── 参数段（M2.5/M2.6）的不变量 ──
        // 1) 整条流的瓦片网格必须一致：它由分辨率唯一决定，任何一位读错都会变
        assert_eq!(
            tile_grid_mismatch, 0,
            "整条流的瓦片网格必须一致（{tile_grid_mismatch} 处不一致，首个网格 {tile_grid:?}）"
        );
        let grid = tile_grid.expect("至少应当有一帧给出瓦片网格");
        assert!(grid.0 >= 1 && grid.1 >= 1, "瓦片数不合法：{grid:?}");
        // 2) 每个 OBU_FRAME 的瓦片组必须与 tile_info 自洽（长度正好用完）
        assert_eq!(
            tile_group_bad, 0,
            "瓦片组与 tile_info 必须自洽（{tile_group_bad} 处不自洽）"
        );
        assert!(frame_obus > 1000, "OBU_FRAME 太少（{frame_obus}）");
        // 3) OBU_FRAME 的非压缩头之后必须有瓦片数据；
        //    OBU_FRAME_HEADER 的载荷应当到字节对齐为止**恰好**用完
        assert_eq!(
            frame_obu_with_data, frame_obus,
            "OBU_FRAME 的非压缩头之后应当还有瓦片数据（{frame_obu_with_data}/{frame_obus}）"
        );
        assert_eq!(
            frame_header_obus_exact, frame_header_obus,
            "OBU_FRAME_HEADER 的载荷应当恰好用完（{frame_header_obus_exact}/{frame_header_obus}）"
        );
        // 4) 非压缩头不该超过 512 字节（真素材实测远小于此）
        assert!(header_bits_max < 8 * 512, "非压缩头太长：{header_bits_max} 位");
        // 5) 量化步长不能恒为 0（有损编码），tx_mode/CDEF 阻尼都要落在合法集合里
        assert!(qidx_max > 0, "base_q_idx 恒为 0 说明量化参数读错了");
        assert!(
            tx_modes.keys().all(|m| *m <= 2),
            "tx_mode 只可能是 0/1/2：{tx_modes:?}"
        );
        assert!(
            cdef_dampings.keys().all(|d| (3..=6).contains(d)),
            "CDEF 阻尼只可能是 3..=6：{cdef_dampings:?}"
        );
    }

    /// M3 的真素材验收：**瓦片 → 熵解码器**。
    ///
    /// 做两件事：
    /// 1. 每个 `OBU_FRAME` 的瓦片组都用**引擎的正式实现**（`video::tile`）切出来
    ///    （不再是测试里手写的副本），每帧一个瓦片；
    /// 2. 每个瓦片 `init_symbol`（读 15 位建算术状态），检查初始不变量；
    ///    再抽若干个瓦片把整条算术流**读满**，要求：
    ///    - 每读一次符号 `SymbolValue < SymbolRange`（算术解码的核心不变量）；
    ///    - 读满（`SymbolMaxBits <= 0`）之后位置指示器**正好**等于瓦片字节数
    ///      —— 这条是"补读位的账"没算错的硬证据。
    ///
    /// **还不做**：`exit_symbol()`。它检查的是瓦片末尾的 trailing bits，只有按真实
    /// 语法把算术流消耗干净才会落在正确位置上 —— 这个测试是**乱读**符号（没有语法
    /// 含义），位置对不上是正常的。真语法的 `exit_symbol()` 验收见
    /// `real_av1_file_blocks_consume_the_entropy_stream` 与
    /// `bundled_probe_streams_pass_exit_symbol_exactly`。
    #[test]
    #[ignore = "手工跑：需要 AV1_REAL_FILE 指向一份真实 AV1 文件"]
    fn real_av1_file_tiles_feed_the_entropy_decoder() {
        let path = real_file();
        let (mut player, track_id, _config) = open_video(&path);

        let mut seq: Option<SequenceHeader> = None;
        let mut store = RefFrameStore::new();
        let mut frames = 0u64;
        let mut groups = 0u64;
        let mut tiles = 0u64;
        let mut min_tile = usize::MAX;
        let mut max_tile = 0usize;
        let mut drain_attempts = 0u64;
        let mut drain_ok = 0u64;
        let mut failures: Vec<String> = Vec::new();

        while let Some(pkt) = player.next_packet().unwrap() {
            if pkt.track_id != track_id {
                continue;
            }
            let mut iter = ObuIter::new(&pkt.data);
            while let Some(obu) = iter.next_obu().expect("OBU 走不动了") {
                match obu.header.obu_type {
                    ObuType::SequenceHeader => {
                        seq = Some(SequenceHeader::parse(obu.payload).expect("序列头解析失败"));
                    }
                    ObuType::Frame => {
                        let seq = seq.as_ref().expect("帧之前必须先有序列头");
                        let ids = ObuLayerIds {
                            temporal_id: obu.header.temporal_id,
                            spatial_id: obu.header.spatial_id,
                        };
                        let fh = FrameHeader::parse(
                            obu.payload,
                            seq,
                            &mut store,
                            ids,
                            HeaderTail::ByteAlignment,
                        )
                        .unwrap_or_else(|e| panic!("帧头解析失败：{e}"));
                        frames += 1;
                        // 帧头 + byte_alignment 之后就是瓦片组
                        // 诊断旋钮：瓦片起点的字节偏移（默认 0）
                    let shift: i64 = std::env::var("AV1_TILE_SHIFT")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    let off = fh.bits_consumed / 8;
                    let off = (off as i64 + shift).clamp(0, obu.payload.len() as i64) as usize;
                    let body = &obu.payload[off..];
                        let group = parse_tile_group(body, &fh.params.tile_info).unwrap_or_else(
                            |e| panic!("第 {frames} 帧的瓦片组切分失败：{e}"),
                        );
                        groups += 1;
                        for t in &group.tiles {
                            tiles += 1;
                            min_tile = min_tile.min(t.data.len());
                            max_tile = max_tile.max(t.data.len());
                            // init_symbol：读 15 位建立算术状态
                            let d = SymbolDecoder::new(t.data, !fh.disable_cdf_update)
                                .unwrap_or_else(|e| {
                                    panic!("第 {frames} 帧的瓦片 {} 初始化失败：{e}", t.index)
                                });
                            assert!(
                                d.symbol_value() < d.symbol_range(),
                                "第 {frames} 帧瓦片 {} 初始就不满足 SymbolValue < SymbolRange",
                                t.index
                            );
                            // 抽样"读满"（每个瓦片要读上万次，全跑太慢）
                            if drain_attempts < 16 {
                                drain_attempts += 1;
                                let mut d =
                                    SymbolDecoder::new(t.data, !fh.disable_cdf_update).unwrap();
                                let mut reads = 0u64;
                                while d.symbol_max_bits() > 0 {
                                    d.read_bool().expect("算术流读不动了");
                                    assert!(
                                        d.symbol_value() < d.symbol_range(),
                                        "第 {frames} 帧瓦片 {} 第 {reads} 次读符号后不变量被破坏",
                                        t.index
                                    );
                                    reads += 1;
                                }
                                if d.bit_pos() == t.data.len() * 8
                                    && (-14..=0).contains(&d.symbol_max_bits())
                                {
                                    drain_ok += 1;
                                } else if failures.len() < 4 {
                                    failures.push(format!(
                                        "第 {frames} 帧瓦片 {}：读满 {reads} 次后位置 {}（应为 {}）、\
                                         SymbolMaxBits {}",
                                        t.index,
                                        d.bit_pos(),
                                        t.data.len() * 8,
                                        d.symbol_max_bits()
                                    ));
                                }
                            }
                        }
                        store.refresh(&fh);
                    }
                    _ => {}
                }
            }
        }

        println!(
            "OBU_FRAME {frames}、瓦片组 {groups}、瓦片 {tiles}（字节 {min_tile}..{max_tile}）"
        );
        println!("读满校验：抽样 {drain_attempts} 个瓦片、通过 {drain_ok}");
        for f in &failures {
            println!("  {f}");
        }

        assert!(frames > 1000, "帧太少（{frames}）");
        assert_eq!(groups, frames, "每个 OBU_FRAME 都该有一个瓦片组");
        assert_eq!(tiles, groups, "这条流是单瓦片（每帧一个瓦片）");
        // 真实流里有极小的帧（实测最小 15 字节 —— 内容与前一帧几乎一样时压得极狠），
        // 但至少要够 `init_symbol` 读那 15 位
        assert!(min_tile >= 2, "瓦片小到读不出 15 位初始值：{min_tile}");
        assert!(max_tile > 1000, "最大的瓦片太小，不像真的算术流：{max_tile}");
        assert_eq!(
            drain_ok, drain_attempts,
            "抽样的瓦片都必须能\"读满且位置正好到末尾\""
        );
        assert!(drain_attempts >= 16, "抽样数不够：{drain_attempts}");
    }

    /// M4a 的真素材验收：**分区树**（规范 §5.11.4）。
    ///
    /// 这是第一次用熵解码器读**真实语法**；这条测试只走到分区树为止，所以还不能
    /// `exit_symbol()`（算术流只被分区符号消耗了一部分）。能验的是结构不变量：
    ///
    /// 1. **分区树恰好覆盖整个瓦片** —— 每个 4×4 位置被写且只被写一次
    ///    （递归结构、`Partition_Subsize` 表、`hasRows/hasCols` 边界处理，任何一处错
    ///    都会表现为漏块或重叠）；
    /// 2. **大瓦片遍历完分区树后算术流还剩着位** —— 分区符号只占一小部分，
    ///    后面还有块内容。这条能抓住"上下文/CDF 选错导致符号乱、消耗异常"。
    #[test]
    #[ignore = "手工跑：需要 AV1_REAL_FILE 指向一份真实 AV1 文件"]
    fn real_av1_file_partition_trees_cover_every_tile() {
        let path = real_file();
        let (mut player, track_id, _config) = open_video(&path);

        let mut seq: Option<SequenceHeader> = None;
        let mut store = RefFrameStore::new();
        let mut frames = 0u64;
        // 调试用旋钮：只跑前 N 帧（默认全跑）。整轨要建 3 万多次帧级网格，debug 下不快。
        let max_frames: u64 = std::env::var("AV1_MAX_FRAMES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(u64::MAX);
        let mut trees = 0u64;
        let mut uncovered_total = 0u64;
        let mut nodes_total = 0u64;
        let mut leaves_total = 0u64;
        let mut max_depth = 0u32;
        let mut partition_hist = [0u64; 10];
        let mut big_trees = 0u64;
        let mut big_with_bits = 0u64;
        let mut min_bits_left = i64::MAX;
        let mut failures: Vec<String> = Vec::new();
        // 帧级 `MiSizes` 网格跨帧复用（每帧重建 13 万格太浪费；`reset_filled` 清标记，
        // 而且 `partition_ctx` 只认本帧写过的位置，不会被上一帧残留污染）
        let mut grid: Option<MiGrid> = None;

        while let Some(pkt) = player.next_packet().unwrap() {
            if frames >= max_frames {
                break;
            }
            if pkt.track_id != track_id {
                continue;
            }
            let mut iter = ObuIter::new(&pkt.data);
            while let Some(obu) = iter.next_obu().expect("OBU 走不动了") {
                match obu.header.obu_type {
                    ObuType::SequenceHeader => {
                        seq = Some(SequenceHeader::parse(obu.payload).expect("序列头解析失败"));
                    }
                    ObuType::Frame => {
                        if frames >= max_frames {
                            break;
                        }
                        let seq = seq.as_ref().expect("帧之前必须先有序列头");
                        let ids = ObuLayerIds {
                            temporal_id: obu.header.temporal_id,
                            spatial_id: obu.header.spatial_id,
                        };
                        let fh = FrameHeader::parse(
                            obu.payload,
                            seq,
                            &mut store,
                            ids,
                            HeaderTail::ByteAlignment,
                        )
                        .unwrap_or_else(|e| panic!("帧头解析失败：{e}"));
                        frames += 1;
                        // 诊断旋钮：瓦片起点的字节偏移（默认 0）
                    let shift: i64 = std::env::var("AV1_TILE_SHIFT")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    let off = fh.bits_consumed / 8;
                    let off = (off as i64 + shift).clamp(0, obu.payload.len() as i64) as usize;
                    let body = &obu.payload[off..];
                        let group = parse_tile_group(body, &fh.params.tile_info).unwrap_or_else(
                            |e| panic!("第 {frames} 帧的瓦片组切分失败：{e}"),
                        );
                        // 帧级的 MiSizes 网格（tile 之间共享坐标系，但不重叠）
                        let mi_rows =
                            *fh.params.tile_info.mi_row_starts.last().unwrap_or(&0) as usize;
                        let mi_cols =
                            *fh.params.tile_info.mi_col_starts.last().unwrap_or(&0) as usize;
                        let slot = grid.get_or_insert_with(|| MiGrid::new(mi_rows, mi_cols));
                        if slot.rows() != mi_rows || slot.cols() != mi_cols {
                            *slot = MiGrid::new(mi_rows, mi_cols);
                        }
                        slot.reset_filled();
                        for t in &group.tiles {
                            let tree = decode_tile_partitions(
                                t,
                                &fh.params.tile_info,
                                slot,
                                fh.disable_cdf_update,
                            )
                            .unwrap_or_else(|e| {
                                panic!("第 {frames} 帧瓦片 {} 的分区树失败：{e}", t.index)
                            });
                            trees += 1;
                            uncovered_total += tree.uncovered as u64;
                            nodes_total += tree.stats.nodes;
                            leaves_total += tree.stats.leaves;
                            max_depth = max_depth.max(tree.stats.max_depth);
                            for (i, v) in tree.stats.partition_histogram.iter().enumerate() {
                                partition_hist[i] += v;
                            }
                            if tree.uncovered > 0 && failures.len() < 4 {
                                failures.push(format!(
                                    "第 {frames} 帧瓦片 {}：{} 个 4×4 位置没被覆盖",
                                    t.index, tree.uncovered
                                ));
                            }
                            // 大瓦片（>= 8 KiB）遍历完分区树后应当还剩位
                            if t.data.len() >= 8192 {
                                big_trees += 1;
                                min_bits_left = min_bits_left.min(tree.bits_left);
                                if tree.bits_left > 0 {
                                    big_with_bits += 1;
                                } else if failures.len() < 4 {
                                    failures.push(format!(
                                        "第 {frames} 帧瓦片 {}（{} 字节）：分区树读完流里只剩 {} 位",
                                        t.index,
                                        t.data.len(),
                                        tree.bits_left
                                    ));
                                }
                            }
                        }
                        store.refresh(&fh);
                    }
                    _ => {}
                }
            }
        }

        println!("OBU_FRAME {frames}、分区树 {trees} 棵");
        println!(
            "节点 {nodes_total}、叶子（块）{leaves_total}、最大深度 {max_depth}、\
             平均每棵 {:.1} 个节点",
            nodes_total as f64 / trees.max(1) as f64
        );
        println!("分区类型分布（NONE/HORZ/VERT/SPLIT/HORZ_A/HORZ_B/VERT_A/VERT_B/HORZ_4/VERT_4）");
        println!("  {partition_hist:?}");
        println!(
            "大瓦片 {big_trees} 棵、其中遍历完还剩位的 {big_with_bits}、最小剩余 {min_bits_left} 位"
        );
        for f in &failures {
            println!("  {f}");
        }

        assert!(frames > 1000, "帧太少（{frames}）");
        assert!(trees > 100, "分区树太少（{trees}）");
        assert_eq!(uncovered_total, 0, "分区树必须恰好覆盖瓦片（{uncovered_total} 个位置没覆盖）");
        assert!(leaves_total >= nodes_total, "叶子数不该少于节点数");
        assert!(max_depth <= 7, "分区树深度不该超过 7（实得 {max_depth}）");
        assert!(big_trees > 0, "大瓦片太少（{big_trees}）");
        assert_eq!(
            big_with_bits, big_trees,
            "大瓦片遍历完分区树后应当还剩着算术流（实得 {big_with_bits}/{big_trees}，\
             最小剩余 {min_bits_left} 位）"
        );
    }

    /// **M4b 的真素材验收：熵层的最终硬验收。**
    ///
    /// 按真实语法（`decode_partition` + `decode_block` + `coeffs`）把整条瓦片的算术流
    /// 读干净，然后调 `exit_symbol()` —— 它检查的是瓦片末尾的 **trailing bits**：
    /// 凡是符号数、每个符号用的 CDF / 上下文、位记账里任何一处错位，
    /// 读到的位置就不会正好落在编码器写下的那个 `1` + 补零上。
    ///
    /// 这条比任何单测都硬：它是**端到端的位对齐证据**，不需要参考解码器。
    #[test]
    #[ignore = "手工跑：需要 AV1_REAL_FILE 指向一份真实 AV1 文件"]
    fn real_av1_file_blocks_consume_the_entropy_stream() {
        let path = real_file();
        let (mut player, track_id, _config) = open_video(&path);

        let mut seq: Option<SequenceHeader> = None;
        let mut store = RefFrameStore::new();
        let mut mf_store = MotionFieldStore::new();
        let mut frames = 0u64;
        let mut tiles = 0u64;
        let mut exit_ok = 0u64;
        let mut exit_fail = 0u64;
        let mut blocks_total = 0u64;
        let mut symbols_total = 0u64;
        let mut leaves_total = 0u64;
        let mut covered_bad = 0u64;
        let mut overlaps = 0u64;
        // 只留第一帧那一趟的超块位消耗轨迹（诊断"多读"是否均匀）
        let mut sb_trace: Vec<u32> = Vec::new();
        let mut sb_trace_leaves: Vec<u32> = Vec::new();
        let mut sb_trace_size = 0usize;
        let mut txb_size_hist = [[0u64; 2]; 19];
        let mut y_modes: Vec<u64> = Vec::new();
        let mut uv_modes: Vec<u64> = Vec::new();
        let mut blk_tx_sizes: Vec<u64> = Vec::new();
        let mut leaf_hist: Vec<u64> = Vec::new();
        let mut no_chroma_blocks = 0u64;
        let mut leaves_n = 0u64;
        let mut coeff_txbs = 0u64;
        let mut coeff_all_zero = 0u64;
        let mut coeff_eob_sum = 0u64;
        let mut coeff_eob_max = 0u32;
        let mut coeff_golomb = 0u64;
        let mut coeff_literals = 0u64;
        let mut coeff_txtype = 0u64;
        let mut coeff_bits = 0u64;
        let mut block_bits = 0u64;
        let mut mode_bits = 0u64;
        let mut residual_bits = 0u64;
        let mut bits_eob = 0u64;
        let mut bits_base = 0u64;
        let mut bits_sign = 0u64;
        let mut extra_bits = 0u64;
        let mut sign_bits = 0u64;
        let mut tile_bits = 0u64;
        let mut block_bits = 0u64;
        let mut mode_bits = 0u64;
        let mut residual_bits = 0u64;
        let mut bits_eob = 0u64;
        let mut bits_base = 0u64;
        let mut bits_sign = 0u64;
        let mut extra_bits = 0u64;
        let mut sign_bits = 0u64;
        let mut intra_frames = 0u64;
        let mut exit_ok_intra = 0u64;
        let mut exit_ok_inter = 0u64;
        // 精确判据（比 `exit_symbol` 的过/不过敏感）：消耗位必须**恰好**等于
        // 「瓦片里最后一个 `1` 的位置」，见 `entropy::required_consumed_bits`。
        let mut exact_ok = 0u64;
        let mut exact_fail = 0u64;
        let mut failures: Vec<String> = Vec::new();
        let max_frames: u64 = std::env::var("AV1_MAX_FRAMES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(u64::MAX);

        let mut mi_grid: Option<MiGrid> = None;
        let mut block_grid: Option<BlockGrid> = None;
        let mut coeff_ctxs: Option<CoeffContexts> = None;

        while let Some(pkt) = player.next_packet().unwrap() {
            if frames >= max_frames {
                break;
            }
            if pkt.track_id != track_id {
                continue;
            }
            let mut iter = ObuIter::new(&pkt.data);
            while let Some(obu) = iter.next_obu().expect("OBU 走不动了") {
                match obu.header.obu_type {
                    ObuType::SequenceHeader => {
                        seq = Some(SequenceHeader::parse(obu.payload).expect("序列头解析失败"));
                    }
                    ObuType::Frame => {
                        if frames >= max_frames {
                            break;
                        }
                        let seq = seq.as_ref().expect("帧之前必须先有序列头");
                        let ids = ObuLayerIds {
                            temporal_id: obu.header.temporal_id,
                            spatial_id: obu.header.spatial_id,
                        };
                        let fh = FrameHeader::parse(
                            obu.payload,
                            seq,
                            &mut store,
                            ids,
                            HeaderTail::ByteAlignment,
                        )
                        .unwrap_or_else(|e| panic!("帧头解析失败：{e}"));
                        frames += 1;
                        if fh.is_intra {
                            intra_frames += 1;
                        }
                        // `use_ref_frame_mvs` 时先做运动场估计（规范里在非压缩头里做）
                        let field: Option<MotionField> = if fh.use_ref_frame_mvs {
                            let mut f = MotionField::default();
                            motion_field_estimation(&mut f, &fh, seq, &mf_store);
                            Some(f)
                        } else {
                            None
                        };
                        // 诊断旋钮：瓦片起点的字节偏移（默认 0）
                    let shift: i64 = std::env::var("AV1_TILE_SHIFT")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    let off = fh.bits_consumed / 8;
                    let off = (off as i64 + shift).clamp(0, obu.payload.len() as i64) as usize;
                    let body = &obu.payload[off..];
                        let group = parse_tile_group(body, &fh.params.tile_info).unwrap_or_else(
                            |e| panic!("第 {frames} 帧的瓦片组切分失败：{e}"),
                        );
                        let mi_rows =
                            *fh.params.tile_info.mi_row_starts.last().unwrap_or(&0) as usize;
                        let mi_cols =
                            *fh.params.tile_info.mi_col_starts.last().unwrap_or(&0) as usize;
                        let g = mi_grid.get_or_insert_with(|| MiGrid::new(mi_rows, mi_cols));
                        if g.rows() != mi_rows || g.cols() != mi_cols {
                            *g = MiGrid::new(mi_rows, mi_cols);
                        }
                        g.reset_filled();
                        let bg = block_grid.get_or_insert_with(|| BlockGrid::new(mi_rows, mi_cols));
                        if bg.rows() != mi_rows || bg.cols() != mi_cols {
                            *bg = BlockGrid::new(mi_rows, mi_cols);
                        }
                        bg.latch_prev_segment_ids();
                        bg.reset_frame();
                        let cc = coeff_ctxs.get_or_insert_with(|| {
                            CoeffContexts::new(
                                mi_rows,
                                mi_cols,
                                seq.color.subsampling_x,
                                seq.color.subsampling_y,
                                if seq.color.mono_chrome { 1 } else { 3 },
                            )
                        });
                        // 帧级 CDF 上下文（§7.20）：`primary_ref_frame == PRIMARY_REF_NONE`
                        // 用默认表（关键帧/帧内帧），否则从参考槽整份载入 ——
                        // 帧间帧没有这一步连位都读不对。
                        let mut cdf_ctx = if fh.primary_ref_frame == PRIMARY_REF_NONE {
                            CdfContext::defaults(fh.params.quant.base_q_idx as u32)
                        } else {
                            store.load_cdfs(fh.ref_frame_idx[fh.primary_ref_frame as usize])
                        };
                        // 环路恢复的帧级状态（按帧头的 LoopRestorationSize 开数组）
                        let mut lr = me_decode::video::lr::LrState::new(
                            seq,
                            &fh.params.lr,
                            fh.frame_height,
                            fh.upscaled_width,
                        );
                        for t in &group.tiles {
                            tiles += 1;
                            if std::env::var("AV1_TRACE_TXB").is_ok() {
                                tile_bits = t.data.len() as u64 * 8;
                            }
                            let r = decode_tile(
                                t,
                                seq,
                                &fh,
                                &store,
                                field.as_ref(),
                                &mut cdf_ctx,
                                g,
                                bg,
                                cc,
                                &mut lr,
                                None,
                            )
                            .unwrap_or_else(|e| {
                                panic!("第 {frames} 帧瓦片 {} 的块级语法失败：{e}", t.index)
                            });
                            // 精确判据：消耗位必须恰好落在「瓦片里最后一个 1」那一位
                            let required = required_consumed_bits(t.data)
                                .unwrap_or_else(|| panic!("第 {frames} 帧瓦片 {} 里一个 1 都没有", t.index));
                            let consumed = t.data.len() as i64 * 8 - 15 - r.bits_left;
                            if consumed == required {
                                exact_ok += 1;
                            } else {
                                exact_fail += 1;
                                if failures.len() < 5 {
                                    failures.push(format!(
                                        "第 {frames} 帧瓦片 {}：消耗 {consumed} 位 ≠ 应有 {required} 位（差 {}）",
                                        t.index,
                                        consumed - required
                                    ));
                                }
                            }
                            coeff_txbs += r.blocks.coeff.txbs;
                            coeff_all_zero += r.blocks.coeff.all_zero;
                            coeff_eob_sum += r.blocks.coeff.eob_sum;
                            coeff_eob_max = coeff_eob_max.max(r.blocks.coeff.eob_max);
                            coeff_golomb += r.blocks.coeff.golomb;
                            coeff_literals += r.blocks.coeff.literals;
                            coeff_txtype += r.blocks.coeff.tx_type_reads;
                            coeff_bits += r.blocks.coeff.bits;
                            block_bits += r.blocks.bits;
                            mode_bits += r.blocks.mode_bits;
                            residual_bits += r.blocks.coeff.residual_bits;
                            bits_eob += r.blocks.coeff.bits_eob;
                            bits_base += r.blocks.coeff.bits_base;
                            bits_sign += r.blocks.coeff.bits_sign;
                            extra_bits += r.blocks.coeff.extra_bits;
                            sign_bits += r.blocks.coeff.sign_bits;
                            for (i, v) in r.blocks.coeff.txb_size_histogram.iter().enumerate() {
                                txb_size_hist[i][0] += v[0];
                                txb_size_hist[i][1] += v[1];
                            }
                            no_chroma_blocks += r.blocks.no_chroma_blocks;
                            leaves_n += 1;
                            if leaf_hist.is_empty() {
                                leaf_hist = r.partition.leaf_histogram.to_vec();
                            } else {
                                for (i, v) in r.partition.leaf_histogram.iter().enumerate() {
                                    leaf_hist[i] += v;
                                }
                            }
                            if y_modes.is_empty() {
                                y_modes = r.blocks.y_mode_histogram.to_vec();
                                uv_modes = r.blocks.uv_mode_histogram.to_vec();
                                blk_tx_sizes = r.blocks.tx_size_histogram.to_vec();
                            } else {
                                for (i, v) in r.blocks.y_mode_histogram.iter().enumerate() {
                                    y_modes[i] += v;
                                }
                                for (i, v) in r.blocks.uv_mode_histogram.iter().enumerate() {
                                    uv_modes[i] += v;
                                }
                                for (i, v) in r.blocks.tx_size_histogram.iter().enumerate() {
                                    blk_tx_sizes[i] += v;
                                }
                            }
                            blocks_total += r.blocks.blocks;
                            symbols_total += r.blocks.symbols;
                            leaves_total += r.partition.leaves;
                            covered_bad += r.uncovered as u64;
                            overlaps += r.overlaps as u64;
                            if frames == 1 && t.index == 0 {
                                sb_trace = r.sb_bits.clone();
                                sb_trace_leaves = r.sb_leaves.clone();
                                sb_trace_size = t.data.len();
                            }
                            if r.exit_ok {
                                exit_ok += 1;
                                if fh.is_intra {
                                    exit_ok_intra += 1;
                                } else {
                                    exit_ok_inter += 1;
                                }
                            } else {
                                exit_fail += 1;
                                if failures.len() < 5 {
                                    failures.push(format!(
                                        "第 {frames} 帧瓦片 {}：exit_symbol 失败                                         （剩余 {} 位、覆盖缺口 {}、块 {}）",
                                        t.index, r.bits_left, r.uncovered, r.blocks.blocks
                                    ));
                                }
                            }
                        }
                        // 参考帧更新（含运动矢量场存储）
                        let (saved_rf, saved_mvs) =
                            motion_field_storage(bg, &fh, seq, &store);
                        for slot in 0..8 {
                            if (fh.refresh_frame_flags >> slot) & 1 == 1 {
                                mf_store.save(
                                    slot,
                                    mi_rows,
                                    mi_cols,
                                    fh.is_intra,
                                    saved_rf.clone(),
                                    saved_mvs.clone(),
                                    fh.order_hints,
                                );
                            }
                        }
                        if !fh.disable_frame_end_update_cdf {
                            store.save_cdfs(fh.refresh_frame_flags, &cdf_ctx);
                        }
                        store.refresh(&fh);
                    }
                    _ => {}
                }
            }
        }

        println!(
            "OBU_FRAME {frames}（帧内 {intra_frames}）、瓦片 {tiles}、块 {blocks_total}\
             、模式符号 {symbols_total}、分区叶子 {leaves_total}"
        );
        println!("exit_symbol：通过 {exit_ok}（帧内 {exit_ok_intra} / 帧间 {exit_ok_inter}）、\
                  失败 {exit_fail}、覆盖缺口 {covered_bad}、重复写入 {overlaps}");
        if !sb_trace.is_empty() {
            let tile_bits = (sb_trace_size * 8) as i64;
            let total: u64 = sb_trace.iter().map(|&v| v as u64).sum();
            let mean = total as f64 / sb_trace.len() as f64;
            let min = *sb_trace.iter().min().unwrap();
            let max = *sb_trace.iter().max().unwrap();
            println!(
                "第 1 帧瓦片 {sb_trace_size} 字节 = {tile_bits} 位；超块 {} 个，\
                 每超块消耗 均值 {mean:.0} 位（{min}..{max}）；语法合计 {total} 位\
                 （瓦片剩余 {tile_bits} - 已读）",
                sb_trace.len()
            );
            // 累计消耗跨过瓦片容量（真实反转点）的位置
            let cap = 8 * sb_trace_size as i64 - 15;
            let mut acc = 0i64;
            let mut turn = sb_trace.len();
            for (i, v) in sb_trace.iter().enumerate() {
                acc += *v as i64;
                if acc > cap {
                    turn = i;
                    break;
                }
            }
            println!("  前 12 个超块：{:?}", &sb_trace[..sb_trace.len().min(12)]);
            println!("  中段 12 个：{:?}", &sb_trace[40..52.min(sb_trace.len())]);
            println!("  超块消耗反转点（累计超过瓦片容量 {cap} 位）：第 {turn} 个 / 共 {} 个（{:.1}%）", sb_trace.len(), turn as f64 * 100.0 / sb_trace.len() as f64);
            let leaves_head: Vec<u32> = sb_trace_leaves[..sb_trace_leaves.len().min(12)].to_vec();
            println!("  前 12 个超块的块数：{leaves_head:?}");
            let l: u64 = sb_trace_leaves.iter().map(|&v| v as u64).sum();
            let lmin = *sb_trace_leaves.iter().min().unwrap();
            let lmax = *sb_trace_leaves.iter().max().unwrap();
            println!(
                "  每超块块数 均值 {:.2}（{lmin}..{lmax}），合计 {l}",
                l as f64 / sb_trace_leaves.len() as f64
            );
        }
        println!(
            "系数：变换块 {coeff_txbs}、全零 {coeff_all_zero}、eob 合计 {coeff_eob_sum}（最大 {coeff_eob_max}）\
             、golomb {coeff_golomb}、字面量 {coeff_literals}、tx_type {coeff_txtype}"
        );
        println!("位记账：块共 {block_bits} 位、模式 {mode_bits} 位、残差 {residual_bits} 位、系数 {coeff_bits} 位");
        println!(
            "系数分段：eob 段 {bits_eob}、base 段 {bits_base}、符号段 {bits_sign}（eob 附加位 {extra_bits}、符号位 {sign_bits}）"
        );
        {
            const NAMES: [&str; 19] = [
                "4x4", "8x8", "16x16", "32x32", "64x64", "4x8", "8x4", "8x16", "16x8", "16x32",
                "32x16", "32x64", "64x32", "4x16", "16x4", "8x32", "32x8", "16x64", "64x16",
            ];
            let mut line = String::from("变换块尺寸分布（个数/位数/每块位）:");
            for (i, v) in txb_size_hist.iter().enumerate() {
                if v[0] > 0 {
                    line.push_str(&format!(
                        " {}={}/{}/{:.1}",
                        NAMES[i],
                        v[0],
                        v[1],
                        v[1] as f64 / v[0] as f64
                    ));
                }
            }
            println!("{line}");
        }
        {
            const BN: [&str; 22] = [
                "4x4", "4x8", "8x4", "8x8", "8x16", "16x8", "16x16", "16x32", "32x16", "32x32",
                "32x64", "64x32", "64x64", "64x128", "128x64", "128x128", "4x16", "16x4", "8x32",
                "32x8", "16x64", "64x16",
            ];
            let mut line = String::from("叶子块尺寸分布:");
            let mut small = 0u64;
            for (i, v) in leaf_hist.iter().enumerate() {
                if *v > 0 {
                    line.push_str(&format!(" {}={}", BN[i], v));
                }
                let bw4 = 1u64 << (i % 4).min(3);
                let _ = bw4;
                if matches!(i, 0 | 1 | 2 | 16 | 17) {
                    small += *v;
                }
            }
            println!("{line}");
            println!("  其中「可能 HasChroma=0」的窄块（4x4/4x8/8x4/4x16/16x4）= {small}");
        }
        {
            let c = &seq.as_ref().expect("有序列头").color;
            println!(
                "HasChroma=0 的块数: {no_chroma_blocks}；色度抽样 subsampling_x={} subsampling_y={}；mono_chrome={}",
                c.subsampling_x, c.subsampling_y, c.mono_chrome
            );
        }
        println!("块级 y_mode 直方图（DC/V/H/D45/D135/D113/D157/D203/SMOOTH/SMOOTH_V/SMOOTH_H/PAETH/CFL）: {y_modes:?}");
        println!("块级 uv_mode 直方图: {uv_modes:?}");
        println!("块级 tx_size 直方图（4x4,8x8,16x16,32x32,64x64,...）: {blk_tx_sizes:?}");
        for f in &failures {
            println!("  {f}");
        }

        assert!(frames >= 1, "一帧都没解（{frames}）");
        assert_eq!(covered_bad, 0, "分区树不该有覆盖缺口");
        assert_eq!(
            blocks_total, leaves_total,
            "每个分区叶子应当恰好对应一个 `decode_block`"
        );
        assert!(coeff_bits > 0 && mode_bits > 0, "位记账没动起来");
        // 位记账必须自洽：块级 = 模式 + 残差，残差 = 系数
        assert_eq!(
            residual_bits, coeff_bits,
            "残差跨度应当等于系数之和（{residual_bits} vs {coeff_bits}）"
        );
        assert_eq!(
            mode_bits + residual_bits,
            block_bits,
            "块级记账应当等于模式 + 残差"
        );
        // *** 最终硬验收：`exit_symbol()` 的 trailing bits ***
        assert!(
            exit_fail == 0,
            "有 {exit_fail} 个瓦片的 trailing bits 对不上（熵层的最终硬验收：\
             每个瓦片必须恰好消耗到「瓦片里最后一个 1」那一位）"
        );
        // 更敏感的那条：消耗位必须**恰好**等于应有消耗位（不是"差一点也算过"）
        println!(
            "精确判据：{exact_ok} 个瓦片逐位吻合、{exact_fail} 个不吻合\
             （exit_symbol 通过 {exit_ok}：帧内 {exit_ok_intra} / 帧间 {exit_ok_inter}）"
        );
        assert!(
            exact_fail == 0,
            "有 {exact_fail} 个瓦片的消耗位不等于应有消耗位：\n{}",
            failures.join("\n")
        );
    }

    /// 这份流开了哪些工具 —— 决定了 M4b 会不会走到 palette / intrabc 等分支。
    #[test]
    #[ignore = "手工跑：需要 AV1_REAL_FILE 指向一份真实 AV1 文件"]
    fn real_av1_file_feature_census() {
        let path = real_file();
        let (mut player, track_id, _config) = open_video(&path);
        let mut seq: Option<SequenceHeader> = None;
        let mut store = RefFrameStore::new();
        let mut frames = 0u64;
        let max_frames: u64 = std::env::var("AV1_MAX_FRAMES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3000);
        let mut inter = 0u64;
        let mut use_mvs = 0u64;
        let mut screen = 0u64;
        let mut intrabc = 0u64;
        let mut warped = 0u64;
        let mut switchable_motion_mode = 0u64;
        let mut reference_select = 0u64;
        let mut skip_mode = 0u64;
        let mut seg_enabled = 0u64;
        let mut reduced_tx = 0u64;
        let mut multi_tile = 0u64;
        while let Some(pkt) = player.next_packet().unwrap() {
            if frames >= max_frames {
                break;
            }
            if pkt.track_id != track_id {
                continue;
            }
            let mut iter = ObuIter::new(&pkt.data);
            while let Some(obu) = iter.next_obu().expect("OBU 走不动了") {
                match obu.header.obu_type {
                    ObuType::SequenceHeader => {
                        seq = Some(SequenceHeader::parse(obu.payload).expect("序列头"));
                    }
                    ObuType::Frame => {
                        let seq = seq.as_ref().unwrap();
                        let ids = ObuLayerIds {
                            temporal_id: obu.header.temporal_id,
                            spatial_id: obu.header.spatial_id,
                        };
                        let fh = FrameHeader::parse(
                            obu.payload,
                            seq,
                            &mut store,
                            ids,
                            HeaderTail::ByteAlignment,
                        )
                        .unwrap();
                        frames += 1;
                        if !fh.is_intra {
                            inter += 1;
                            use_mvs += fh.use_ref_frame_mvs as u64;
                            screen += fh.allow_screen_content_tools as u64;
                            intrabc += fh.allow_intrabc as u64;
                            warped += fh.params.allow_warped_motion as u64;
                            switchable_motion_mode += fh.is_motion_mode_switchable as u64;
                            reference_select += fh.params.reference_select as u64;
                            skip_mode += fh.params.skip_mode_present as u64;
                            seg_enabled += fh.params.segmentation.enabled as u64;
                            reduced_tx += fh.params.reduced_tx_set as u64;
                            multi_tile += (fh.params.tile_info.num_tiles() > 1) as u64;
                        }
                        store.refresh(&fh);
                    }
                    _ => {}
                }
            }
        }
        let seq = seq.expect("要有序列头");
        println!("帧 {frames}（帧间 {inter}）");
        println!(
            "帧级：use_ref_frame_mvs {use_mvs} / screen_content_tools {screen} / intrabc {intrabc} \
             / warped_motion {warped} / motion_mode_switchable {switchable_motion_mode}"
        );
        println!(
            "帧级：reference_select {reference_select} / skip_mode {skip_mode} / segmentation {seg_enabled} \
             / reduced_tx_set {reduced_tx} / 多瓦片 {multi_tile}"
        );
        println!(
            "序列级：order_hint {} / ref_frame_mvs {} / dual_filter {} / cdef {} / filter_intra {} \
             / interintra {} / masked_compound {} / jnt_comp {} / 128x128 {}",
            seq.enable_order_hint,
            seq.enable_ref_frame_mvs,
            seq.enable_dual_filter,
            seq.enable_cdef,
            seq.enable_filter_intra,
            seq.enable_interintra_compound,
            seq.enable_masked_compound,
            seq.enable_jnt_comp,
            seq.use_128x128_superblock,
        );
        assert!(frames > 10, "帧太少");
    }

    /// **分级定位**：用 Chrome 的 WebCodecs 生成的「复杂度递增」参照流，逐级跑 `exit_symbol`。
    ///
    /// 这些流是**真实编码器**（libaom）写出来的，最小一份只有 31 字节（64×64 单色）——
    /// 是目前唯一能"分级定位"的手段：**哪一级开始失败，那条语法路径就有问题**。
    /// 用 `tools/av1probe/gen.html` + `serve.py` 生成，`AV1_PROBE_DIR` 指向产物目录。
    #[test]
    #[ignore = "手工跑：需要先用 tools/av1probe 生成参照流"]
    fn probe_obu_streams_step_up_complexity() {
        let dir = std::env::var("AV1_PROBE_DIR").expect("设 AV1_PROBE_DIR 指向 probe 产物目录");
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .expect("读不了目录")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "bin").unwrap_or(false))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("put_L"))
                    .unwrap_or(false)
            })
            .collect();
        if files.is_empty() {
            panic!("目录 {dir} 里没有 put_L*.bin（先跑 tools/av1probe）");
        }
        files.sort();
        let mut failed: Vec<String> = Vec::new();
        for path in &files {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let data = std::fs::read(path).expect("读文件");
            match decode_obu_stream_once(&data) {
                Ok(s) => {
                    println!(
                        "{name}: 字节 {}、帧 {}、瓦片 {}、块 {}、叶 {}、缺口 {}、重叠 {}、\
                         真实消耗 {} 位、exit {}（剩 {}）、模式/残差 {} / {} 位",
                        data.len(),
                        s.frames,
                        s.tiles,
                        s.blocks,
                        s.leaves,
                        s.uncovered,
                        s.overlaps,
                        s.consumed,
                        if s.exit_ok { "OK" } else { "FAIL" },
                        s.bits_left,
                        s.mode_bits,
                        s.residual_bits
                    );
                    let mut ph = String::new();
                    for (i, v) in s.partition_hist.iter().enumerate() {
                        if *v > 0 {
                            ph.push_str(&format!(" {}={}", PART_NAMES[i], v));
                        }
                    }
                    println!("    分区:{ph}");
                    let mut lh = String::new();
                    const BN: [&str; 22] = [
                        "4x4", "4x8", "8x4", "8x8", "8x16", "16x8", "16x16", "16x32", "32x16",
                        "32x32", "32x64", "64x32", "64x64", "64x128", "128x64", "128x128", "4x16",
                        "16x4", "8x32", "32x8", "16x64", "64x16",
                    ];
                    for (i, v) in s.leaf_hist.iter().enumerate() {
                        if *v > 0 {
                            lh.push_str(&format!(" {}={}", BN[i], v));
                        }
                    }
                    println!("    叶子:{lh}");
                    println!(
                        "    应有消耗 {} 位 / 实际消耗 {} 位（差 {}）{}",
                        s.required,
                        s.exit_consumed,
                        s.exit_consumed - s.required,
                        if s.exit_ok { "" } else { "  ← exit_symbol 失败" }
                    );
                    if !s.exit_ok {
                        failed.push(format!("{name}（剩 {} 位）", s.bits_left));
                    }
                }
                Err(e) => {
                    println!("{name}: 解码失败 {e}");
                    failed.push(format!("{name}（{e}）"));
                }
            }
        }
        println!(
            "---- 通过 {} / {} ----",
            files.len() - failed.len(),
            files.len()
        );
        for f in &failed {
            println!("  未通过：{f}");
        }
        assert!(failed.is_empty(), "有参照流没过：{failed:?}");
    }

    const PART_NAMES: [&str; 10] = [
        "NONE", "HORZ", "VERT", "SPLIT", "HORZ_A", "HORZ_B", "VERT_A", "VERT_B", "HORZ_4", "VERT_4",
    ];

    /// **`exit_symbol` 的常驻回归测试**（不需要 Chrome、不需要外部素材）。
    ///
    /// 用 `tools/av1probe/samples/` 里那批「真实编码器（libaom）写出来」的小流，
    /// 对每条流断言两件事：
    /// ① 每个瓦片都通过 `exit_symbol`（trailing bits 校验）；
    /// ② **实际消耗的算术位数 == 「应有消耗」**（= 瓦片里最后一个 `1` 的位置，
    ///    由规范 §8.2.6 的 `trailingBitPosition` 化简而来）。
    ///
    /// ② 是关键：它是个**精确整数**，比"过/不过"敏感得多。Golomb 长度码极性写反
    /// 这种 bug 就是靠它抓出来的（当时 14 条流里 11 条不过，差额整齐地是 10~11 位）。
    ///
    /// ⚠️ 判据只能有**一份实现**：它曾在本测试里被就地抄过一份、扫位顺序写反，
    /// 于是 14 条流全被报成"差 1~6 位"、看着像"漏读了符号"（详见
    /// `me_decode::video::entropy::required_consumed_bits` 的说明）。
    #[test]
    fn bundled_probe_streams_pass_exit_symbol_exactly() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/av1probe/samples");
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("读不了参照流目录 {dir}：{e}（它应该随仓库提供）"))
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("put_L") && n.ends_with(".bin"))
                    .unwrap_or(false)
            })
            .collect();
        assert!(!files.is_empty(), "目录 {dir} 里没有 put_L*.bin");
        files.sort();
        let mut bad = Vec::new();
        for path in &files {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let data = std::fs::read(path).expect("读文件");
            let s = decode_obu_stream_once(&data)
                .unwrap_or_else(|e| panic!("{name} 解码失败：{e}"));
            if !s.exit_ok {
                bad.push(format!("{name}: exit_symbol 失败（剩 {} 位）", s.bits_left));
            } else if s.exit_consumed != s.required {
                bad.push(format!(
                    "{name}: 消耗 {} 位 ≠ 应有 {} 位（差 {}）",
                    s.exit_consumed,
                    s.required,
                    s.exit_consumed - s.required
                ));
            }
        }
        println!("参照流 {} 条，全部通过：{files:?}", files.len());
        assert!(bad.is_empty(), "有参照流没对上：\n{}", bad.join("\n"));
    }

    /// **M5 的像素级验收**：拿"源内容已知"的参照流把像素解出来。
    ///
    /// `#808080` 恰好等于帧内 DC 预测在"没有邻居"时的默认值
    /// （`1 << (BitDepth-1)` = 128），而后面每个块的边又都是已经重建好的邻居
    /// —— 邻居也是 128，于是**残差恒为 0**。所以这条流的正确输出是
    /// "每个平面、每个样本都等于 128"。
    ///
    /// 这条验收**不需要参考解码器**（本机没有 ffmpeg/dav1d/aomdec），
    /// 而且它把整条链路一起钉住：瓦片切分 → 分区树 → 块级语法 → 帧内预测 →
    /// 平面写回。预测的边取值、`haveAbove/haveLeft` 的推导、`maxX/maxY` 的钳位、
    /// 色度平面的行列数与色度邻居的奇偶换算……任何一处错，平坦性就会破。
    #[test]
    fn bundled_probe_flat_gray_samples_reconstruct_to_128() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/av1probe/samples");
        let mut checked = 0u64;
        for name in [
            "put_L30_64x64_flat.bin",
            "put_L40_32x32_flat.bin",
            "put_L44_16x16_flat.bin",
        ] {
            let data = std::fs::read(format!("{dir}/{name}")).expect("读参照流");
            let frame = decode_obu_stream_to_pixels(&data)
                .unwrap_or_else(|e| panic!("{name} 的像素重建失败：{e}"));
            for plane in 0..frame.num_planes as u8 {
                let (w, h) = frame.plane_dims(plane);
                let p = frame.plane(plane);
                for y in 0..h {
                    for x in 0..w {
                        let v = p.get(x, y);
                        assert_eq!(
                            v, 128,
                            "{name} 平面 {plane} 的 ({x},{y}) 解出 {v}，而源内容是平坦的 128"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 6000, "样本太少（只验了 {checked} 个采样）");
        println!("M5 像素验收：{checked} 个采样全部等于 128");
    }

    /// **M5 像素验收②**：带**非零残差**的参照流必须解出"完全平坦"的像素。
    ///
    /// 上面那条 `#808080` 的残差恒为 0（源值恰好等于帧内 DC 预测的默认值），
    /// 所以它只钉住了预测与平面写回。这一条换成 16 / 64 / 112 / 144 / 154 / 192 / 240
    /// 这些**必须靠残差重建**的灰度 —— 反量化（§7.12.4）或反变换（§7.13）里
    /// 任何一处算错，块与块之间、乃至块内部就不会再是同一个值。
    ///
    /// 判据刻意**不依赖"源灰度到底是多少"**（那只有生成器知道，`samples/` 里
    /// 只有文件名没有颜色）：
    /// ① 亮度必须 `min == max`（源是纯色 ⇒ 解出来必须处处相等）；
    /// ② 色度必须恒为 128（源是灰度图 ⇒ 色度中性）；
    /// ③ 每个样本还要"非 128"——否则这一条就退化成了上一条。
    ///
    /// 同一灰度的 64/32/16 三个尺寸都在名单里，顺带互相印证。
    #[test]
    fn bundled_probe_flat_samples_reconstruct_flat_with_nonzero_residual() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/av1probe/samples");
        let mut planes_checked = 0u64;
        for name in [
            "put_L31_64x64_flat.bin",
            "put_L36_64x64_flat.bin",
            "put_L33_64x64_flat.bin",
            "put_L34_64x64_flat.bin",
            "put_L46_64x64_flat.bin",
            "put_L35_64x64_flat.bin",
            "put_L32_64x64_flat.bin",
            "put_L41_32x32_flat.bin",
            "put_L45_16x16_flat.bin",
            "put_L42_32x32_flat.bin",
        ] {
            let data = std::fs::read(format!("{dir}/{name}")).expect("读参照流");
            let frame = decode_obu_stream_to_pixels(&data)
                .unwrap_or_else(|e| panic!("{name} 的像素重建失败：{e}"));
            for plane in 0..frame.num_planes as u8 {
                let (w, h) = frame.plane_dims(plane);
                let p = frame.plane(plane);
                let v0 = p.get(0, 0);
                for y in 0..h {
                    for x in 0..w {
                        let v = p.get(x, y);
                        assert_eq!(
                            v, v0,
                            "{name} 平面 {plane} 的 ({x},{y}) 解出 {v}，而 (0,0) 是 {v0} —— 源是纯色，解出来必须完全平坦"
                        );
                    }
                }
                if plane > 0 {
                    assert_eq!(
                        v0, 128,
                        "{name} 平面 {plane} 解出 {v0}，而源是灰度图 ⇒ 色度必须是中性灰 128"
                    );
                } else {
                    assert_ne!(
                        v0, 128,
                        "{name} 挑的就是非 128 的灰度，否则这一条退化成残差恒零的上一条"
                    );
                }
                planes_checked += 1;
            }
        }
        assert!(planes_checked >= 20, "验得太少：{planes_checked} 个平面");
        println!("M5 像素验收（非零残差）：{planes_checked} 个平面全部平坦");
    }

    /// **M5 像素验收③**：`L43` 是四象限图（四个纯色块），判据比"平坦"更硬。
    ///
    /// 四个象限必须**各自完全平坦、且彼此颜色不同**：只要分区树的块边界、
    /// `haveAbove/haveLeft` 的推导、或残差叠加的位置错一格，象限内部就会花掉；
    /// 只要两个象限被解成同一个值，说明"块归属"判错了。
    ///
    /// 同样不依赖具体灰度（颜色只有生成器知道），只钉结构。
    #[test]
    fn bundled_probe_quadrants_reconstruct_four_flat_quadrants() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/av1probe/samples");
        let data =
            std::fs::read(format!("{dir}/put_L43_32x32_quadrants.bin")).expect("读参照流");
        let frame = decode_obu_stream_to_pixels(&data).expect("L43 的像素重建");
        let (w, h) = frame.plane_dims(0);
        assert!(w >= 4 && h >= 4, "四象限图至少要 4×4，现在是 {w}×{h}");
        let (hw, hh) = (w / 2, h / 2);
        let corners = [(0usize, 0usize), (hw, 0), (0, hh), (hw, hh)];
        let mut vals = Vec::new();
        for (cx, cy) in corners {
            let v0 = frame.plane(0).get(cx, cy);
            for y in cy..cy + hh {
                for x in cx..cx + hw {
                    assert_eq!(
                        frame.plane(0).get(x, y),
                        v0,
                        "象限 ({cx},{cy}) 内部不平坦：({x},{y}) 与 (0,0) 不同"
                    );
                }
            }
            vals.push(v0);
        }
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert_ne!(
                    vals[i], vals[j],
                    "第 {i} 个象限与第 {j} 个象限解出了同一个值 {}（源是四个不同颜色）",
                    vals[i]
                );
            }
        }
        println!("M5 像素验收（四象限）：四个象限各自平坦，值 = {vals:?}");
    }

    /// **诊断用**：把 `samples/` 里每条参照流解到像素，打印三个平面的值域与均值。
    ///
    /// 参照流的源内容是生成器画出来的**已知图案**（见 `tools/av1probe/gen.html`），
    /// 灰度 `#rrggbb` 在 BT.601/709 下恰好满足 `Y = r = g = b`、`U = V = 128`
    /// （系数和为 1），所以"解出来的 Y 应当处处等于那个灰度值"是一条
    /// **不依赖参考解码器**的硬验收。有损编码会带来 ±1~2 的偏差，
    /// 所以先跑这条诊断把实际值域摸清楚，再写断言。
    #[test]
    #[ignore = "诊断用：打印参照流的像素统计（用 --nocapture 看）"]
    fn probe_print_sample_histograms() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/av1probe/samples");
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("读不了 {dir}：{e}"))
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.starts_with("put_L") && n.ends_with(".bin"))
            .collect();
        names.sort();
        for name in names {
            let data = std::fs::read(format!("{dir}/{name}")).expect("读参照流");
            let frame = match decode_obu_stream_to_pixels(&data) {
                Ok(f) => f,
                Err(e) => {
                    println!("{name}: 解码失败：{e}");
                    continue;
                }
            };
            let mut line = format!("{name}:");
            for plane in 0..frame.num_planes as u8 {
                let (w, h) = frame.plane_dims(plane);
                let p = frame.plane(plane);
                let (mut min, mut max, mut sum) = (u16::MAX, 0u16, 0u64);
                for y in 0..h {
                    for x in 0..w {
                        let v = p.get(x, y);
                        min = min.min(v);
                        max = max.max(v);
                        sum += u64::from(v);
                    }
                }
                line.push_str(&format!(
                    "  P{plane}({w}×{h}) min={min} max={max} avg={:.1}",
                    sum as f64 / (w * h) as f64
                ));
            }
            println!("{line}");
        }
    }

    /// 把一条参照流解到**像素**（M5）。只处理单帧的 keyframe 参照流。
    fn decode_obu_stream_to_pixels(data: &[u8]) -> Result<CurrFrame> {
        let mut seq: Option<SequenceHeader> = None;
        let mut store = RefFrameStore::new();
        let mut iter = ObuIter::new(data);
        while let Some(obu) = iter.next_obu()? {
            match obu.header.obu_type {
                ObuType::SequenceHeader => seq = Some(SequenceHeader::parse(obu.payload)?),
                ObuType::Frame => {
                    let seq = seq
                        .as_ref()
                        .ok_or_else(|| MediaError::Bitstream("帧之前没有序列头".into()))?;
                    let ids = ObuLayerIds {
                        temporal_id: obu.header.temporal_id,
                        spatial_id: obu.header.spatial_id,
                    };
                    let fh = FrameHeader::parse(
                        obu.payload,
                        seq,
                        &mut store,
                        ids,
                        HeaderTail::ByteAlignment,
                    )?;
                    let off = fh.bits_consumed / 8;
                    let group = parse_tile_group(&obu.payload[off..], &fh.params.tile_info)?;
                    let mi_rows = *fh.params.tile_info.mi_row_starts.last().unwrap_or(&0) as usize;
                    let mi_cols = *fh.params.tile_info.mi_col_starts.last().unwrap_or(&0) as usize;
                    let mut g = MiGrid::new(mi_rows, mi_cols);
                    g.reset_filled();
                    let mut bg = BlockGrid::new(mi_rows, mi_cols);
                    bg.latch_prev_segment_ids();
                    bg.reset_frame();
                    let num_planes = if seq.color.mono_chrome { 1 } else { 3 };
                    let mut cc = CoeffContexts::new(
                        mi_rows,
                        mi_cols,
                        seq.color.subsampling_x,
                        seq.color.subsampling_y,
                        num_planes,
                    );
                    let mut frame = CurrFrame::new(
                        fh.frame_width as usize,
                        fh.frame_height as usize,
                        seq.color.subsampling_x,
                        seq.color.subsampling_y,
                        num_planes as usize,
                        seq.color.bit_depth,
                    );
                    let mut cdf_ctx = CdfContext::defaults(fh.params.quant.base_q_idx as u32);
                    // 环路恢复的帧级状态（按帧头的 LoopRestorationSize 开数组）
                    let mut lr = me_decode::video::lr::LrState::new(
                        seq,
                        &fh.params.lr,
                        fh.frame_height,
                        fh.upscaled_width,
                    );
                    for t in &group.tiles {
                        decode_tile(
                            t,
                            seq,
                            &fh,
                            &store,
                            None,
                            &mut cdf_ctx,
                            &mut g,
                            &mut bg,
                            &mut cc,
                            &mut lr,
                            Some(&mut frame),
                        )?;
                    }
                    return Ok(frame);
                }
                _ => {}
            }
        }
        Err(MediaError::Bitstream("流里没有帧".into()))
    }

    /// 临时诊断：把外部的裸 OBU 文件用**测试路径**解到像素并落盘 Y 平面
    /// （用来二分"驱动接线"还是"共享解码路径"的问题），顺带打印语法层自洽指标。
    #[test]
    #[ignore]
    fn tmp_decode_external_obu_to_pixels() {
        let path = std::env::var("AV1_OBU_FILE").expect("AV1_OBU_FILE");
        let data = std::fs::read(&path).unwrap();
        let mut seq: Option<SequenceHeader> = None;
        let mut store = RefFrameStore::new();
        let mut iter = ObuIter::new(&data);
        while let Some(obu) = iter.next_obu().unwrap() {
            match obu.header.obu_type {
                ObuType::SequenceHeader => seq = Some(SequenceHeader::parse(obu.payload).unwrap()),
                ObuType::Frame => {
                    let seq = seq.as_ref().unwrap();
                    let ids = ObuLayerIds {
                        temporal_id: obu.header.temporal_id,
                        spatial_id: obu.header.spatial_id,
                    };
                    let fh = FrameHeader::parse(
                        obu.payload,
                        seq,
                        &mut store,
                        ids,
                        HeaderTail::ByteAlignment,
                    )
                    .unwrap();
                    println!(
                        "帧头：payload={} 字节、bits_consumed={}（瓦片从第 {} 字节起）、intra={} {}x{} upscaled={} base_q={} tx_mode={} 单瓦片={}",
                        obu.payload.len(),
                        fh.bits_consumed,
                        fh.bits_consumed / 8,
                        fh.is_intra,
                        fh.frame_width,
                        fh.frame_height,
                        fh.upscaled_width,
                        fh.params.quant.base_q_idx,
                        fh.params.tx_mode,
                        fh.params.tile_info.num_tiles() == 1
                    );
                    println!(
                        "  FH 细节：show_frame={} err_resilient={} disable_cdf={} order_hint={} primary_ref={} refresh={:#04x} short_signaling={} ref_idx={:?} use_ref_mvs={} allow_hp_mv={} interp={} mm_switch={} disable_end_cdf={} render={}x{} frame_size_override={} superres={} \
                         lf_level={:?} cdef_bits={} lr={} tx_mode={} delta_q={} seg={} warp={} reduced_tx={} gm_types={:?}",
                        fh.show_frame, fh.error_resilient_mode, fh.disable_cdf_update, fh.order_hint, fh.primary_ref_frame,
                        fh.refresh_frame_flags, fh.frame_refs_short_signaling, fh.ref_frame_idx, fh.use_ref_frame_mvs,
                        fh.allow_high_precision_mv, fh.interpolation_filter, fh.is_motion_mode_switchable, fh.disable_frame_end_update_cdf,
                        fh.render_width, fh.render_height, fh.frame_size_override, fh.use_superres,
                        fh.params.loop_filter.level, fh.params.cdef.bits, fh.params.lr.uses_lr, fh.params.tx_mode,
                        fh.params.delta_q.present, fh.params.segmentation.enabled, fh.params.allow_warped_motion, fh.params.reduced_tx_set,
                        fh.params.global_motion.gm_type
                    );
                    let off = fh.bits_consumed / 8;
                    let group = parse_tile_group(&obu.payload[off..], &fh.params.tile_info).unwrap();
                    let mi_rows = *fh.params.tile_info.mi_row_starts.last().unwrap_or(&0) as usize;
                    let mi_cols = *fh.params.tile_info.mi_col_starts.last().unwrap_or(&0) as usize;
                    let mut g = MiGrid::new(mi_rows, mi_cols);
                    g.reset_filled();
                    let mut bg = BlockGrid::new(mi_rows, mi_cols);
                    bg.latch_prev_segment_ids();
                    bg.reset_frame();
                    let num_planes = if seq.color.mono_chrome { 1 } else { 3 };
                    let mut cc = CoeffContexts::new(
                        mi_rows,
                        mi_cols,
                        seq.color.subsampling_x,
                        seq.color.subsampling_y,
                        num_planes,
                    );
                    let syntax_only = std::env::var("AV1_OBU_SYNTAX_ONLY").is_ok();
                    let mut frame = CurrFrame::new(
                        fh.frame_width as usize,
                        fh.frame_height as usize,
                        seq.color.subsampling_x,
                        seq.color.subsampling_y,
                        num_planes as usize,
                        seq.color.bit_depth,
                    );
                    let mut cdf_ctx = CdfContext::defaults(fh.params.quant.base_q_idx as u32);
                    // 环路恢复的帧级状态（按帧头的 LoopRestorationSize 开数组）
                    let mut lr = me_decode::video::lr::LrState::new(
                        seq,
                        &fh.params.lr,
                        fh.frame_height,
                        fh.upscaled_width,
                    );
                    for t in &group.tiles {
                        let r = decode_tile(
                            t,
                            seq,
                            &fh,
                            &store,
                            None,
                            &mut cdf_ctx,
                            &mut g,
                            &mut bg,
                            &mut cc,
                            &mut lr,
                            if syntax_only { None } else { Some(&mut frame) },
                        )
                        .unwrap();
                        println!(
                            "瓦片 {}：exit_ok={} bits_left={} 未覆盖={} 重叠={} 块={} 叶子={} txb={} all_zero={} eob和={} 残差位={} 模式位={}",
                            t.index,
                            r.exit_ok,
                            r.bits_left,
                            r.uncovered,
                            r.overlaps,
                            r.blocks.blocks,
                            r.partition.leaves,
                            r.blocks.coeff.txbs,
                            r.blocks.coeff.all_zero,
                            r.blocks.coeff.eob_sum,
                            r.blocks.coeff.residual_bits,
                            r.blocks.mode_bits,
                        );
                        println!(
                            "  精确判据：应有消耗 {}，实际消耗 {}",
                            required_consumed_bits(t.data).unwrap_or(-1),
                            t.data.len() as i64 * 8 - 15 - r.bits_left
                        );
                    }
                    let (w, h) = frame.plane_dims(0);
                    let p = frame.plane(0);
                    let mut out = Vec::with_capacity(w * h);
                    for y in 0..h {
                        for x in 0..w {
                            out.push(p.get(x, y) as u8);
                        }
                    }
                    std::fs::write("/tmp/lerxu-av1-verify/harness_frame0.y", &out).unwrap();
                    println!("harness decoded {w}x{h}，Y 平面已落盘");
                    if std::env::var("AV1_OBU_ALL_FRAMES").is_ok() {
                        continue;
                    }
                    return;
                }
                _ => {}
            }
        }
        panic!("流里没有帧");
    }

    #[derive(Default)]
    struct ProbeSummary {
        frames: u64,
        tiles: u64,
        blocks: u64,
        leaves: u64,
        uncovered: u64,
        overlaps: u64,
        consumed: i64,
        bits_left: i64,
        exit_ok: bool,
        mode_bits: u64,
        residual_bits: u64,
        partition_hist: [u64; 10],
        leaf_hist: [u64; 22],
        /// `exit_symbol` 唯一能给出的**绝对判据**：这份瓦片里「最后一个 1」的位置。
        /// 化简规范 §8.2.6 的 `trailingBitPosition` 得到，合法的解码必须恰好消耗到那一位。
        /// 具体的计算收在 `me_decode::video::entropy::required_consumed_bits()` 里
        /// （**只有那一份实现**：这里曾经把扫位顺序写反，取到了「末字节最高位的 1」，
        /// 于是 14 条参照流被误报成"少读 1~6 位"，白查了一轮）。
        required: i64,
        /// 自研解码器实际消耗的算术位（`SymbolMaxBits` 的差分）—— 按 `exit_symbol`
        /// 的口径算，和上面那个内部位记账不是一回事。
        exit_consumed: i64,
    }

    /// 解一个**裸 OBU 流**（WebCodecs 直接吐出来的那种）里的所有帧。
    fn decode_obu_stream_once(data: &[u8]) -> Result<ProbeSummary> {
        let mut seq: Option<SequenceHeader> = None;
        let mut store = RefFrameStore::new();
        let mut out = ProbeSummary::default();
        let mut iter = ObuIter::new(data);
        while let Some(obu) = iter.next_obu()? {
            match obu.header.obu_type {
                ObuType::SequenceHeader => {
                    seq = Some(SequenceHeader::parse(obu.payload)?);
                }
                ObuType::Frame => {
                    let seq = seq
                        .as_ref()
                        .ok_or_else(|| MediaError::Bitstream("帧之前没有序列头".into()))?;
                    let ids = ObuLayerIds {
                        temporal_id: obu.header.temporal_id,
                        spatial_id: obu.header.spatial_id,
                    };
                    let fh = FrameHeader::parse(
                        obu.payload,
                        seq,
                        &mut store,
                        ids,
                        HeaderTail::ByteAlignment,
                    )?;
                    if std::env::var("AV1_TRACE_FH").is_ok() {
                        println!(
                            "  FH: payload={} 字节、bits_consumed={}（%8={}）、\
                             payload_bits_after_header={}、mi={}x{}、sb128={}、tx_mode={}、\
                             base_q={}",
                            obu.payload.len(),
                            fh.bits_consumed,
                            fh.bits_consumed % 8,
                            fh.payload_bits_after_header,
                            fh.mi_rows,
                            fh.mi_cols,
                            seq.use_128x128_superblock,
                            fh.params.tx_mode,
                            fh.params.quant.base_q_idx
                        );
                    }
                    // 诊断旋钮：瓦片起点的字节偏移（默认 0）
                    let shift: i64 = std::env::var("AV1_TILE_SHIFT")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    let off = fh.bits_consumed / 8;
                    let off = (off as i64 + shift).clamp(0, obu.payload.len() as i64) as usize;
                    let body = &obu.payload[off..];
                    if std::env::var("AV1_TRACE_TOOLS").is_ok() {
                        println!(
                            "    工具：seq[order_hint={} ref_mvs={} dual_filter={} cdef={} filter_intra={} interintra={} masked={} jnt={} 128x128={} screen={} mono={} 420=({},{})] \\
                              fh[intra={} allow_intrabc={} cdef_bits={} delta_q={} reduced_tx={} tx_mode={}] mi={}x{}",
                            seq.enable_order_hint,
                            seq.enable_ref_frame_mvs,
                            seq.enable_dual_filter,
                            seq.enable_cdef,
                            seq.enable_filter_intra,
                            seq.enable_interintra_compound,
                            seq.enable_masked_compound,
                            seq.enable_jnt_comp,
                            seq.use_128x128_superblock,
                            fh.allow_screen_content_tools,
                            seq.color.mono_chrome,
                            seq.color.subsampling_x,
                            seq.color.subsampling_y,
                            fh.is_intra,
                            fh.allow_intrabc,
                            fh.params.cdef.bits,
                            fh.params.delta_q.present,
                            fh.params.reduced_tx_set,
                            fh.params.tx_mode,
                            fh.mi_rows,
                            fh.mi_cols
                        );
                    }
                    if std::env::var("AV1_TRACE_FH").is_ok() {
                        println!(
                            "     tile 区 {} 字节，头 4 字节 {:02x?}",
                            body.len(),
                            &body[..body.len().min(4)]
                        );
                    }
                    let group = parse_tile_group(body, &fh.params.tile_info)?;
                    let mi_rows = *fh.params.tile_info.mi_row_starts.last().unwrap_or(&0) as usize;
                    let mi_cols = *fh.params.tile_info.mi_col_starts.last().unwrap_or(&0) as usize;
                    let mut g = MiGrid::new(mi_rows, mi_cols);
                    g.reset_filled();
                    let mut bg = BlockGrid::new(mi_rows, mi_cols);
                    bg.latch_prev_segment_ids();
                    bg.reset_frame();
                    let mut cc = CoeffContexts::new(
                        mi_rows,
                        mi_cols,
                        seq.color.subsampling_x,
                        seq.color.subsampling_y,
                        if seq.color.mono_chrome { 1 } else { 3 },
                    );
                    let field: Option<MotionField> = if fh.use_ref_frame_mvs {
                        let mut f = MotionField::default();
                        motion_field_estimation(&mut f, &fh, seq, &MotionFieldStore::new());
                        Some(f)
                    } else {
                        None
                    };
                    out.frames += 1;
                    let mut cdf_ctx = CdfContext::defaults(fh.params.quant.base_q_idx as u32);
                    // 环路恢复的帧级状态（按帧头的 LoopRestorationSize 开数组）
                    let mut lr = me_decode::video::lr::LrState::new(
                        seq,
                        &fh.params.lr,
                        fh.frame_height,
                        fh.upscaled_width,
                    );
                    for t in &group.tiles {
                        out.tiles += 1;
                        // 「应有消耗」= 瓦片里最后一个 1 的位置（规范 §8.2.6 化简而来）
                        out.required += required_consumed_bits(t.data)
                            .unwrap_or_else(|| panic!("第 {} 个瓦片全是 0 位，不是合法码流", t.index));
                        let r = decode_tile(
                            t,
                            seq,
                            &fh,
                            &store,
                            field.as_ref(),
                            &mut cdf_ctx,
                            &mut g,
                            &mut bg,
                        &mut cc,
                        &mut lr,
                        None,
                    )?;
                    out.exit_consumed += t.data.len() as i64 * 8 - 15 - r.bits_left;
                        out.blocks += r.blocks.blocks;
                        out.leaves += r.partition.leaves;
                        out.uncovered += r.uncovered as u64;
                        out.overlaps += r.overlaps as u64;
                        out.consumed += r.partition_bits + r.blocks.bits as i64;
                        out.mode_bits += r.blocks.mode_bits;
                        out.residual_bits += r.blocks.coeff.residual_bits;
                        out.bits_left = r.bits_left;
                        out.exit_ok |= r.exit_ok;
                        for (i, v) in r.partition.partition_histogram.iter().enumerate() {
                            out.partition_hist[i] += v;
                        }
                        for (i, v) in r.partition.leaf_histogram.iter().enumerate() {
                            out.leaf_hist[i] += v;
                        }
                    }
                    store.refresh(&fh);
                }
                _ => {}
            }
        }
        Ok(out)
    }
}

/// 找出视频轨并把它要的解码器初始化数据备好。
///
/// 容器里存的可能是 `avcC`（MP4）或 Annex-B 的 SPS/PPS（TS），统一成 `avcC` ——
/// 解码层只认这一种。**AV1 例外**：它是 OBU 流，没有 Annex-B 形态，`av1C`
/// 原样送进解码层（见 `me_mux::convert::extradata_to_avcc`）。
#[cfg(feature = "video")]
pub(crate) fn video_track_and_config(
    info: &me_core::MediaInfo,
    name: &str,
) -> Result<(me_core::Track, Vec<u8>)> {
    let track = info
        .tracks
        .iter()
        .find(|t| t.kind == MediaKind::Video)
        .ok_or_else(|| MediaError::Container(format!("{name}: 这条输入里没有视频轨")))?
        .clone();
    if !matches!(track.codec, CodecId::H264 | CodecId::Av1) {
        return Err(MediaError::Unsupported(format!(
            "{name}: 解码目前接了 H.264 与 AV1（这条轨是 {}）",
            track.codec.name()
        )));
    }
    let config = if convert::looks_like_avcc(&track.codec, &track.extradata) {
        track.extradata.clone()
    } else {
        convert::extradata_to_avcc(&track.codec, &track.extradata, 4)?
    };
    Ok((track, config))
}

/// 一次解码的选项。
pub struct DecodeOptions {
    /// **音频模式**：解音频轨（`--out` 这时是 WAV 文件路径，不是目录）。
    pub audio: bool,
    /// 帧落盘目录（None = 只统计，不落盘）。
    pub out_dir: Option<String>,
    /// 最多解多少帧（None = 整条轨）。
    /// 只有视频路径读它；纯音频构建保留字段是为了命令行参数形状不变。
    #[allow(dead_code)]
    pub max_frames: Option<u64>,
    /// 从哪一刻开始（微秒；None = 从头）。
    pub seek_us: Option<i64>,
    pub open: OpenOptions,
}

/// 解出视频帧（并可选落盘）。
pub fn decode_to_frames(input: &str, opts: DecodeOptions) -> Result<u8> {
    if opts.audio {
        #[cfg(feature = "audio")]
        {
            return decode_audio(input, opts);
        }
        #[cfg(not(feature = "audio"))]
        {
            return Err(crate::modules::disabled(
                "音频解析（解码）",
                "audio",
                "decode --audio",
            ));
        }
    }
    #[cfg(feature = "video")]
    {
        decode_video(input, opts)
    }
    #[cfg(not(feature = "video"))]
    {
        let _ = (input, opts);
        Err(crate::modules::disabled("视频解析（解码）", "video", "decode"))
    }
}

/// 视频模式：把视频轨解成 NV12 帧（`--out` 是目录）。
#[cfg(feature = "video")]
fn decode_video(input: &str, opts: DecodeOptions) -> Result<u8> {
    let client = me_stream::default_client()?;
    let opened = me_stream::open_input(&client, input, &opts.open)?;
    let (src, _note) = crate::source::pick_source(opened)?;
    // 解码不需要"边下边播"的缓冲：来一个采样解一个
    let cfg = PlayerConfig { max_buffer_us: 1_000_000, max_buffer_packets: 1024 };
    let mut player = Player::open(src, input, cfg)?;
    let info = player.info();

    // 视频轨 + 解码器初始化数据（形态归一化在这一个函数里，见它的说明）
    let (track, config) = video_track_and_config(&info, input)?;
    let mut decoder = me_decode::open_video_decoder(track.codec.clone(), &config)?;

    if let Some(us) = opts.seek_us {
        player.seek_us(us)?;
    }

    let started = Instant::now();
    let mut out_dir: Option<String> = None;
    let mut index_file: Option<std::fs::File> = None;
    if let Some(dir) = &opts.out_dir {
        std::fs::create_dir_all(dir)?;
        out_dir = Some(dir.clone());
        let f = std::fs::File::create(std::path::Path::new(dir).join("frames.json"))?;
        index_file = Some(f);
    }

    let mut frames: u64 = 0;
    let mut first_pts_us: Option<i64> = None;
    let mut width = 0u32;
    let mut height = 0u32;
    let mut index = String::from("[\n");
    let mut first_entry = true;

    'outer: while let Some(pkt) = player.next_packet()? {
        if pkt.track_id != track.id {
            continue;
        }
        // 引擎里的采样 → 解码层要的形态（H.264/H.265 是 Annex-B → AVCC；
        // AV1 是 OBU 流，原样透传 —— 见 `convert::to_decoder_sample`）
        let sample = convert::to_decoder_sample(&track.codec, &pkt.data, 4);
        if sample.is_empty() {
            continue;
        }
        let pts_us = track.timebase.ticks_to_us(pkt.pts);
        for frame in decoder.decode(&sample, pts_us)? {
            if max_reached(&mut frames, opts.max_frames) {
                break 'outer;
            }
            if first_pts_us.is_none() {
                first_pts_us = Some(frame.pts_us);
            }
            width = frame.width;
            height = frame.height;
            frames += 1;
            if let Some(dir) = &out_dir {
                let path = std::path::Path::new(dir).join(format!("frame-{frames:06}.nv12"));
                std::fs::write(&path, &frame.nv12)?;
                if index_file.is_some() {
                    if !first_entry {
                        index.push_str(",\n");
                    }
                    first_entry = false;
                    index.push_str(&format!(
                        "{{\"n\":{frames},\"file\":\"{}\",\"ptsUs\":{},\"bytes\":{}}}",
                        path.file_name().unwrap_or_default().to_string_lossy(),
                        frame.pts_us,
                        frame.nv12.len()
                    ));
                }
            }
            if frames.is_multiple_of(30) {
                println!(
                    "{}",
                    json::progress_line(
                        "decode-progress",
                        serde_json::json!({ "frames": frames, "lastPtsUs": frame.pts_us })
                    )
                );
            }
        }
    }
    // 收尾：把解码器里排队的帧吐出来（有 B 帧时最后几帧在这里）
    for frame in decoder.flush()? {
        if max_reached(&mut frames, opts.max_frames) {
            break;
        }
        width = frame.width;
        height = frame.height;
        frames += 1;
        if let Some(dir) = &out_dir {
            let path = std::path::Path::new(dir).join(format!("frame-{frames:06}.nv12"));
            std::fs::write(&path, &frame.nv12)?;
            if index_file.is_some() {
                if !first_entry {
                    index.push_str(",\n");
                }
                first_entry = false;
                index.push_str(&format!(
                    "{{\"n\":{frames},\"file\":\"{}\",\"ptsUs\":{},\"bytes\":{}}}",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    frame.pts_us,
                    frame.nv12.len()
                ));
            }
        }
    }
    if let Some(f) = index_file.as_mut() {
        index.push_str("\n]\n");
        f.write_all(index.as_bytes())?;
    }

    let summary = serde_json::json!({
            "frames": frames,
            "width": width,
            "height": height,
            "backend": decoder.backend(),
            "firstPtsUs": first_pts_us.unwrap_or(0),
            "elapsedMs": started.elapsed().as_millis() as u64,
        "outDir": out_dir.unwrap_or_default(),
    });
    println!("{}", json::ok_line(&summary));
    Ok(crate::exit_code::OK)
}

/// 到帧数上限了吗（到就把 `frames` 停在边界，调用方负责跳出）。
#[cfg(feature = "video")]
fn max_reached(frames: &mut u64, max: Option<u64>) -> bool {
    matches!(max, Some(m) if *frames >= m)
}

/// 音频模式：把音频轨解成 PCM 写成 WAV（校验解码器、也给人一个"能听"的产物）。
#[cfg(feature = "audio")]
fn decode_audio(input: &str, opts: DecodeOptions) -> Result<u8> {
    let client = me_stream::default_client()?;
    let opened = me_stream::open_input(&client, input, &opts.open)?;
    let (src, _note) = crate::source::pick_source(opened)?;
    let cfg = PlayerConfig { max_buffer_us: 1_000_000, max_buffer_packets: 1024 };
    let mut player = Player::open(src, input, cfg)?;
    let info = player.info();
    let track = info
        .tracks
        .iter()
        .find(|t| t.kind == MediaKind::Audio)
        .ok_or_else(|| MediaError::Container(format!("{input}: 这条输入里没有音频轨")))?
        .clone();
    let mut decoder =
        me_decode::audio::open_audio_decoder(track.codec.clone(), &track.extradata)?;
    let (mut rate, mut channels, mut bits) = decoder.stream_info();

    // `--seek` 在音频模式过去**被静默忽略**（只有视频路径接了它），校验时容易以为是
    // "定位到这儿解出来就是不一样"。这里与视频路径对齐：先定位再解。
    // 裸流（mp3 / flac）没有索引，落点是"字节比例插值 + 同步到真帧头"（见 DEVELOPMENT.md 七）。
    let seeked_to_us = match opts.seek_us {
        Some(us) => Some(player.seek_us(us)?),
        None => None,
    };

    let started = Instant::now();
    let mut samples: Vec<i32> = Vec::new();
    let mut chunks = 0u64;
    let mut last_pts_us: i64 = 0;
    while let Some(pkt) = player.next_packet()? {
        if pkt.track_id != track.id {
            continue;
        }
        let pts_us = track.timebase.ticks_to_us(pkt.pts);
        for chunk in decoder.decode(&pkt.data, pts_us)? {
            rate = chunk.sample_rate;
            channels = chunk.channels;
            bits = chunk.bits_per_sample;
            samples.extend_from_slice(&chunk.samples);
            chunks += 1;
            last_pts_us = chunk.pts_us;
        }
    }
    for chunk in decoder.flush()? {
        samples.extend_from_slice(&chunk.samples);
        chunks += 1;
    }

    let out = opts.out_dir.clone().unwrap_or_else(|| "out.wav".to_string());
    super::wav::write_wav(
        &out,
        rate,
        channels,
        if bits <= 16 { 16 } else if bits <= 24 { 24 } else { 32 },
        &samples,
    )?;

    println!(
        "{}",
        json::ok_line(&serde_json::json!({
            "codec": track.codec.name(),
            "backend": decoder.backend(),
            "chunks": chunks,
            "samplesPerChannel": samples.len() / channels.max(1) as usize,
            "sampleRate": rate,
            "channels": channels,
            "bitsPerSample": bits,
            "lastPtsUs": last_pts_us,
            // 只有请求过 `--seek` 才有值（否则是 null）：实际落点。裸流是按字节比例
            // 插值算出来的，可能比请求位置晚一帧左右 —— 输出 WAV 的**起点**就是这个时刻。
            "seekedToUs": seeked_to_us,
            "elapsedMs": started.elapsed().as_millis() as u64,
            "out": out,
        }))
    );
    Ok(crate::exit_code::OK)
}
