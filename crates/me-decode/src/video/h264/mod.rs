//! **自研 H.264 软解**（进行中）。
//!
//! ## 为什么必须有它
//!
//! [`crate::lib`] 的后端策略是"硬解优先，自研软解兜底"，而 H.264 是**覆盖面最广**
//! 的那个编码：`ME_DECODE=software`（宿主「设置 → 视频 → 解码方式 → 仅软解」）
//! 这一档以前在 H.264 上是死路 —— 硬解起不来（老机器、无 GPU 的容器、
//! 驱动不认这张卡）就彻底没得播。AV1 那条路已经证明"自研软解"是可行的
//! （[`crate::video`]），H.264 补齐之后，**两个最主流的编码都有兜底**。
//!
//! ## 与硬解共用同一套"共识层"
//!
//! 参数集与切片头的解析**不在这里重写** —— [`crate::h264`] 那份是照 §7.3 逐字段
//! 写的、三个硬解后端（VideoToolbox / VA-API / Media Foundation）已经在用，
//! 而且它已经把"引擎不支持什么"用 [`crate::h264::Sps::check_supported`] 说死了
//! （8 位 4:2:0、逐行帧、滑动窗口参考帧标记）。软解**沿用同一口径**：
//! 支持面与硬解完全一致，用户切到软解不会突然多出或少掉某种素材。
//!
//! ## 分层（照 H.264 的解析顺序，与 AV1 那套"逐层对拍"同一个做法）
//!
//! | 层 | 内容 | 状态 |
//! |---|---|---|
//! | NAL / 访问单元 | AVCC 长度前缀切分、类型分类、参数集按 id 注册（含中途更新）| [`nal`] ✅ |
//! | 参数集 / 切片头 | §7.3.2 / §7.3.3 全字段 + §8.2.1 POC + §8.2.4/8.2.5 参考帧列表 | [`crate::h264`] ✅（硬解共用的那份）|
//! | 像素容器 | 三平面（紧 stride）+ 裁剪 + 紧凑 NV12 输出 | [`frame`] ✅ |
//! | 熵层·CAVLC | §9.2 `coeff_token` / `level` / `run_before` / `total_zeros` | [`cavlc`] ✅（表由 [`cavlc_tables`] 自动抄自规范，脚本带前缀码/Kraft/条数三重自检）|
//! | 熵层·CABAC | §9.3 算术解码器 + §9.3.1 上下文初始化 | 待做 |
//! | 宏块层（**I 切片**）| §7.3.4 `mb_type` / `mb_pred` / `coded_block_pattern` / `mb_qp_delta` / `residual()` + `nC` 推导（§9.2.1 第 4~7 步）+ `I_PCM` | [`mb`] ✅ **真实 IDR 切片上「恰好停在 `rbsp_trailing_bits`」已过**（两枚夹具见 [`mb`] 的单测）|
//! | 宏块层（P/B 切片）| `mb_skip_run` / `sub_mb_pred` / 运动矢量差 | 待做 |
//! | 帧内预测 | §8.3 `I_16x16` 四种 / `I_NxN`（4×4 / 8×8 各九种）/ 色度四种 + 参考样本低通 | [`intra`] ✅（**纯函数**，单测钉住方向模式、DC 的三种回退、8×8 的参考滤波）|
//! | 反变换 | §8.5 反扫描 + 反量化 + 4×4/8×8 整数变换 + `Intra16x16`/色度 DC 的 Hadamard | [`transform`] ✅ |
//! | 重建接线 | 取参考样本（宏块级 + **块级**可用性）→ 预测 → 加残差 → 写回 | [`reconstruct`] ✅ 能出帧（`ME_DECODE=software` 已跑通）|
//! | 去块滤波 | §8.7 遍历顺序、阈值（表 8-16/8-17）、`bS < 4` 与 `bS == 4` 两套滤波器、亮度与色度 | [`deblock`] ✅（**目前 `bS` 只覆盖帧内**，见该模块抬头）|
//! | 帧间预测 | §8.4 运动矢量预测、亚像素插值、加权预测、B 切片 | **待做**（真实素材的主体）|
//! | 输出顺序 | DPB 重排（按 POC）+ 显示时间戳 | 待做（`reorder` 与 `ready` 队列已就位）|
//!
//! ## 现状：I 切片已经**逐样本等于 ffmpeg 的原生解码器**（2026-10-06）
//!
//! `ME_DECODE=software zuvrust decode <片> --count=N --out=…` 在 libx264 编的
//! CAVLC 流上跑通，输出紧凑 NV12。判据是 `tools/h264ref/`：把同一个访问单元
//! 用 **ffmpeg 自带的 h264 解码器**解成 NV12，**逐样本**比。
//!
//! | 覆盖 | 用例 | 结果 |
//! |---|---|---|
//! | 帧内三种模式 | `nodeblock_flat` / `_quad` / `_grad` / `_moving` | **0 差异** |
//! | AC 系数 + 多个 QP | `nodeblock_detail`（crf 18）/ `_qp12` / `_qp30` | **0 差异** |
//! | 色度残差（`cbp_chroma != 0`）| `nodeblock_color` | **0 差异** |
//! | 盘面预测 | `nodeblock_diag` | **0 差异** |
//! | 8×8 变换（`transform_size_8x8_flag`）| `nodeblock_detail_8x8` | **0 差异** |
//! | 高频纹理 | `nodeblock_noise` | **0 差异** |
//! | **§8.7 去块** | `deblock_grad` / `_quad` / `_detail` / `_color` / `_noise` | **0 差异**（对照**含去块**的参考帧）|
//! | CABAC | `cabac_*` | 待做 |
//! | 帧间 | `inter_*` | 待做 |
//!
//! 回归测试 [`mb::tests::a_real_i_slice_reconstructs_bit_exactly_like_the_reference`]
//! 把两枚**真实访问单元 + 参考 NV12** 固化进 `testdata/`，任何机器上 `cargo test` 都能跑，
//! 覆盖"切片头 → CAVLC → 宏块层 → 预测 → 变换 → 色度 → 去块 → NV12"整条链。
//!
//! ### 两条踩过很多次的坑（写在这里，别再犯）
//!
//! 1. **规范 PDF 的文字层在"数字"上不可信**：`2^5` 会被抽成 `25`、表 8-16/8-17 的
//!    矩阵根本抽不出来。凡是从文字层抄的数字，一律用 `pymupdf` 渲染成图**再看一遍**
//!    （`tools/gen_h264_cavlc_tables.py` 那套抽取脚本 + 渲染复核是同一件事的两面）。
//! 2. **`x264-params deblock=0:0` 不是"关掉去块"**，它只是"用 QP 推导的默认强度"。
//!    拿"没去块"的参照只能在**解码侧** `-skip_loop_filter all`（`make_cases.py` 会把
//!    那份落到 `<name>.refnoflt/`）。开发去块那一层时靠它把"预测+变换"和"去块"分开验。
//!
//! ## 下一步（按这个顺序接，别跳）
//!
//! 1. **§9.3 CABAC**：真实素材（x264/各硬件编码器的默认输出）**几乎全是 CABAC**，
//!    不接它等于自研软解只能解 Baseline。这是当前最大的缺口。
//! 2. **§8.4 帧间预测**（P/B 切片）：`mb_skip_run` / `sub_mb_pred` / 运动矢量预测 /
//!    6 抽头亚像素插值 / 加权预测 / B 的 direct 模式 / POC 重排与 DPB。
//! 3. 接上之后 **`deblock::bs_for_edge` 的后两档（`bS = 2 / 1`）必须补齐**
//!    （要看非零系数、运动矢量与参考索引），否则帧间流的去块强度会算小。
//!
//! ## 诊断开关
//!
//! | 开关 | 作用 |
//! |---|---|
//! | `H264_TRACE_FRAME=1` | 每帧一行：`poc / frame_num / 切片数 / MB 数 / 是否有残差` |
//! | `H264_TRACE_MB=1` | 逐宏块：`addr / mb_type / qp / cbp / 预测模式` |

pub mod cavlc;
pub mod cavlc_tables;
pub mod deblock;
pub mod frame;
pub mod intra;
pub mod mb;
pub mod nal;
pub mod reconstruct;
pub mod transform;

use me_core::{MediaError, Result};

use crate::h264::{self, Pps, SliceHeader, Sps};
use crate::VideoDecoder;
use crate::VideoFrame;

use frame::FrameBuf;

/// 自研 H.264 软解器。
pub struct H264Decoder {
    /// 初始化配置（`avcC` 原文 + NAL 长度前缀宽度）——喂进来的采样是 AVCC。
    cfg: h264::Config,
    /// 按 id 索引的参数集（码流中途可以更新）。
    ps: nal::ParamSets,
    /// 已解码待输出的图（按 POC 重排的输出顺序缓冲）。
    ready: Vec<ReadyPic>,
    /// 显示顺序缓冲的容量（`max_num_reorder_frames`）。
    reorder: usize,
    /// 吃过的采样数 / 解出的图数（诊断用）。
    samples: u64,
    pictures: u64,
}

/// 一张已经解完、等着按显示顺序吐出去的图。
struct ReadyPic {
    frame: FrameBuf,
    poc: i32,
    pts_us: i64,
}

impl H264Decoder {
    /// 用 `avcC` 打开（`config` 就是容器给的解码器初始化数据）。
    pub fn open(config: &[u8]) -> Result<Self> {
        let cfg = h264::Config::parse(config)?;
        let mut ps = nal::ParamSets::new();
        ps.load_from_avcc(config)?;
        let reorder = cfg.sps.max_num_reorder_frames() as usize;
        tracing::info!(
            "H.264 自研软解：{}×{}（编码 {}×{}）、重排深度 {}、{}",
            cfg.sps.width(),
            cfg.sps.height(),
            cfg.sps.coded_width(),
            cfg.sps.coded_height(),
            reorder,
            profile_name(cfg.sps.profile_idc)
        );
        Ok(Self {
            cfg,
            ps,
            ready: Vec::new(),
            reorder,
            samples: 0,
            pictures: 0,
        })
    }

    /// 打开时的"进度说明"（日志/诊断用，与 AV1 的 `decode_outlook` 同角色）。
    pub fn outlook(&self) -> String {
        format!(
            "自研 H.264 软解：{}×{} {}、重排深度 {}",
            self.cfg.sps.width(),
            self.cfg.sps.height(),
            profile_name(self.cfg.sps.profile_idc),
            self.reorder
        )
    }

    /// 按显示顺序吐出已经能确定的图（POC 最小的先出）。
    ///
    /// 现在还没被调用：调用点在 [`Self::decode_picture`] 重建完之后（见抬头那张表）。
    #[allow(dead_code, reason = "重建那一层接上之后由 decode_picture 调用")]
    fn take_ready(&mut self) -> Vec<VideoFrame> {
        let mut out = Vec::new();
        while self.ready.len() > self.reorder {
            let Some(idx) = (0..self.ready.len()).min_by_key(|&i| self.ready[i].poc) else {
                break;
            };
            let pic = self.ready.remove(idx);
            out.push(VideoFrame {
                width: pic.frame.disp_w as u32,
                height: pic.frame.disp_h as u32,
                pts_us: pic.pts_us,
                nv12: pic.frame.to_nv12(),
            });
        }
        out
    }
}

impl VideoDecoder for H264Decoder {
    fn decode(&mut self, sample: &[u8], pts_us: i64) -> Result<Vec<VideoFrame>> {
        if sample.is_empty() {
            return Err(MediaError::Bitstream("H.264 采样是空的".into()));
        }
        self.samples += 1;
        let nals = h264::split_avcc_sample(sample, self.cfg.nal_length_size)?;
        let mut slices: Vec<Slice> = Vec::new();
        for nal in nals {
            match nal::nal_type(nal) {
                nal::NAL_SPS => {
                    self.ps.put_sps(nal)?;
                }
                nal::NAL_PPS => {
                    self.ps.put_pps(nal)?;
                }
                nal::NAL_SEI | nal::NAL_AUD | nal::NAL_FILLER => {}
                nal::NAL_END_SEQUENCE | nal::NAL_END_STREAM => {}
                t if h264::is_vcl(t) => slices.push(Slice::parse(nal, &self.ps)?),
                other => {
                    return Err(MediaError::Unsupported(format!(
                        "H.264 的这个 NAL 类型还没接：{other}（SVC/MVC 的扩展切片在 14/20/21）"
                    )))
                }
            }
        }
        if slices.is_empty() {
            // 只有参数集/SEI 的采样是正常的（fMP4 的 avcC 之后常跟着一条），不是错误
            return Ok(Vec::new());
        }
        self.pictures += 1;
        let pics = self.decode_picture(&slices, pts_us)?;
        Ok(pics)
    }

    fn flush(&mut self) -> Result<Vec<VideoFrame>> {
        // 冲刷：把攒着等重排的图按 POC 顺序全部吐出（输入读完了，不会再有更小的 POC）
        let mut out = Vec::new();
        while !self.ready.is_empty() {
            let Some(idx) = (0..self.ready.len()).min_by_key(|&i| self.ready[i].poc) else {
                break;
            };
            let pic = self.ready.remove(idx);
            out.push(VideoFrame {
                width: pic.frame.disp_w as u32,
                height: pic.frame.disp_h as u32,
                pts_us: pic.pts_us,
                nv12: pic.frame.to_nv12(),
            });
        }
        Ok(out)
    }

    fn backend(&self) -> &'static str {
        "self-h264"
    }
}

/// 采样里的一条切片：解析好的头 + **NAL 头 + 去竞争字节后的负载**。
///
/// 熵层（[`cavlc::Cavlc::new`]）与 `more_rbsp_data()` 读的都是后一条缓冲 ——
/// 与切片头解析走 [`nal::rbsp_of_nal`] 同一份，三处拼一条流。
struct Slice {
    header: SliceHeader,
    raw: Vec<u8>,
    sps: Sps,
    pps: Pps,
}

impl Slice {
    fn parse(nal: &[u8], ps: &nal::ParamSets) -> Result<Self> {
        let pps_id = nal::read_slice_pps_id(nal)?;
        let pps = ps.pps(pps_id)?.clone();
        let sps = ps.sps(pps.sps_id)?.clone();
        let header = h264::parse_slice_header(nal, &sps, &pps)?;
        header.check_supported()?;
        // `H264_TRACE_SLICE=1`：把切片头里影响重建与去块的字段打出来。
        // 排查"QP 不对 / 去块该不该跑 / 加权预测开没开"时，这三类是第一个要看的东西。
        if slice_trace() {
            tracing::info!(
                "slice: type={} idr={} first_mb={} frame_num={} poc_lsb={} qp_delta={} \
                 (QPY0={}) deblock_idc={} a_c0_off2={} beta_off2={} cabac_idc={} \
                 ref_l0={} ref_l1={} sdmv={}",
                header.slice_type,
                header.idr,
                header.first_mb_in_slice,
                header.frame_num,
                header.pic_order_cnt_lsb,
                header.slice_qp_delta,
                26 + pps.pic_init_qp_minus26 + header.slice_qp_delta,
                header.disable_deblocking_filter_idc,
                header.slice_alpha_c0_offset_div2,
                header.slice_beta_offset_div2,
                header.cabac_init_idc,
                header.num_ref_idx_l0_active,
                header.num_ref_idx_l1_active,
                header.direct_spatial_mv_pred_flag,
            );
            tracing::info!(
                "  PPS: entropy={} deblock_present={} constrained_intra={} pic_init_qp={} \
                 chroma_off={}/{} t8={} weighted_pred={} weighted_bipred={} \
                 ref_l0_def={} slice_groups={} profile={}",
                pps.entropy_coding_mode_flag,
                pps.deblocking_filter_control_present_flag,
                pps.constrained_intra_pred_flag,
                pps.pic_init_qp_minus26,
                pps.chroma_qp_index_offset,
                pps.second_chroma_qp_index_offset,
                pps.transform_8x8_mode_flag,
                pps.weighted_pred_flag,
                pps.weighted_bipred_idc,
                pps.num_ref_idx_l0_default_active_minus1,
                pps.num_slice_groups_minus1,
                sps.profile_idc,
            );
        }
        Ok(Self {
            header,
            raw: nal::rbsp_of_nal(nal),
            sps,
            pps,
        })
    }
}

/// `H264_TRACE_SLICE=1` 时打切片头摘要（热路径上只读一次环境变量）。
fn slice_trace() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("H264_TRACE_SLICE").is_ok_and(|v| v != "0"))
}

impl H264Decoder {
    /// 解一张图（一个访问单元里的全部切片）。
    fn decode_picture(&mut self, slices: &[Slice], pts_us: i64) -> Result<Vec<VideoFrame>> {
        let first = &slices[0].header;
        let sps = &slices[0].sps;
        // 一张图的所有切片必须同属一帧
        for s in slices {
            if s.header.frame_num != first.frame_num {
                return Err(MediaError::Bitstream(format!(
                    "同一个访问单元里的切片 frame_num 不一致（{} vs {}）",
                    s.header.frame_num, first.frame_num
                )));
            }
        }
        let mut info = mb::PictureInfo::new(
            sps.coded_width() as usize / 16,
            sps.coded_height() as usize / 16,
        );
        for s in slices {
            mb::decode_i_slice(&s.header, &s.sps, &s.pps, &s.raw, &mut info)?;
        }
        // 走到这里说明**语法层全通了**（`decode_i_slice` 末尾的
        // `check_rbsp_trailing` 是这条绝对判据）。接着重建像素。
        let mut frame = FrameBuf::for_sps(sps);
        reconstruct::reconstruct_picture(&mut frame, &mut info, sps, &slices[0].pps)?;
        // ⚠️ 去块滤波（§8.7）还没做 —— 输出的是"去块之前"的构造样本。
        // 与参考解码器对拍时要把编码器的去块关掉（`x264opts deblock=0:0`），
        // 否则差的是"去块还没做"而不是预测/变换错。
        // §8.7：去块滤波在**整张图重建完成之后**做（所以帧内预测读的是去块前样本）
        deblock::deblock_picture(&mut frame, &mut info, &slices[0].pps, first)?;
        Ok(vec![VideoFrame {
            width: frame.disp_w as u32,
            height: frame.disp_h as u32,
            pts_us,
            nv12: frame.to_nv12(),
        }])
    }
}

/// `profile_idc` 的人话名字（日志/错误文案用）。
fn profile_name(profile_idc: u8) -> &'static str {
    match profile_idc {
        66 => "Baseline",
        77 => "Main",
        88 => "Extended",
        100 => "High",
        110 => "High 10",
        122 => "High 4:2:2",
        244 => "High 4:4:4",
        44 => "CAVLC 4:4:4",
        _ => "未知 profile",
    }
}
