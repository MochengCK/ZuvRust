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
//! | 熵层·CABAC | §9.3 算术解码器 + §9.3.1 上下文初始化 + 残差/宏块层 | [`cabac`] / [`cabac_mb`] ⚠️ 见下 |
//! | 宏块层（**I 切片**）| §7.3.4 `mb_type` / `mb_pred` / `coded_block_pattern` / `mb_qp_delta` / `residual()` + `nC` 推导（§9.2.1 第 4~7 步）+ `I_PCM` | [`mb`] ✅ **真实 IDR 切片上「恰好停在 `rbsp_trailing_bits`」已过**（两枚夹具见 [`mb`] 的单测）|
//! | 宏块层（**P 切片**，CAVLC）| `mb_skip_run` / `mb_type` 表 7-13 / `sub_mb_type` 表 7-14 / `ref_idx` / `mvd` | [`mb::decode_p_slice`] ✅ |
//! | 宏块层（B 切片）| `mb_type` 表 7-14、双向预测 | 待做 |
//! | 帧内预测 | §8.3 `I_16x16` 四种 / `I_NxN`（4×4 / 8×8 各九种）/ 色度四种 + 参考样本低通 | [`intra`] ✅（**纯函数**，单测钉住方向模式、DC 的三种回退、8×8 的参考滤波）|
//! | 反变换 | §8.5 反扫描 + 反量化 + 4×4/8×8 整数变换 + `Intra16x16`/色度 DC 的 Hadamard | [`transform`] ✅ |
//! | 重建接线 | 取参考样本（宏块级 + **块级**可用性）→ 预测 → 加残差 → 写回 | [`reconstruct`] ✅ 能出帧（`ME_DECODE=software` 已跑通）|
//! | 去块滤波 | §8.7 遍历顺序、阈值（表 8-16/8-17）、`bS < 4` 与 `bS == 4` 两套滤波器、亮度与色度 | [`deblock`] ✅（**目前 `bS` 只覆盖帧内**，见该模块抬头）|
//! | 帧间预测（**P 切片**）| §8.4.1 运动矢量预测（按 ref 匹配数分支）、§8.4.2 六抽头 + 双线性亚像素插值、加权预测 | [`inter`] ✅ 单参考列表 |
//! | 参考帧管理 | 滑动窗口标记、POC（type 0 / 2）、按 POC 输出重排 | [`H264Decoder::decode_picture`] ✅ |
//! | 帧间预测（B 切片）| 双向预测、`direct` 模式、加权双向预测 | 待做 |
//! | §8.7 去块的帧间 bS | `bS = 2/1` 两档（非零系数、运动矢量/参考索引差异）| 待做（目前帧间边按 0 处理）|
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
//! ## CABAC（2026-11-06）：算术层与 I16 已经对上，4×4 残差还有偏差
//!
//! | 层 | 状态 |
//! |---|---|
//! | §9.3.1 算术解码器（`decode_decision` / `bypass` / `terminate`）| ✅ 与 FFmpeg 的 C 实现**逐 bin 一致**（用它当参照跑同一段码流对拍）|
//! | §9.3.1.1 上下文初始化 | ✅ 表按规范生成；公式是 `((m * SliceQPY) >> 4) + n` |
//! | 宏块层（I 切片：mb_type / 帧内模式 / cbp / qp_delta）| ✅ 与 CAVLC 解码同一内容的语法逐项一致 |
//! | **I16 的 DC 残差** | ✅ **逐样本等于 FFmpeg**（16×16、32×32 平坦帧全对）|
//! | 4×4 亮度残差 | ❌ 显著性图从第 2 个系数起就偏（`cbp` 之后立刻分叉）|
//!
//! 因此 [`cabac_mb`] **默认关闭**（`JISHI_H264_CABAC=1` 才启用）：半成品放出去
//! 用户看到的是花屏，比"明确报不支持"糟得多。默认仍然走"硬解优先、软解只接
//! CAVLC"的老路。
//!
//! ### 已经踩过、值得记下来的四个坑
//!
//! 1. **上下文初值公式**：是 `((m * SliceQPY) >> 4) + n`，**不是**
//!    `((m * (SliceQPY − 26)) >> 8) + n`。后者在 QP=26 附近才碰巧接近，别的 QP 上
//!    整组上下文都错，症状是"前几个 bin 看着对，越往后越错，最后读不到
//!    `end_of_slice_flag`"。
//! 2. **`condTermFlag` 的"未加权"那几处**：`mb_type` / `mb_skip_flag` /
//!    `intra_chroma_pred_mode` / `transform_size_8x8_flag` 的左邻与上邻**各算 1**；
//!    `coded_block_pattern` / `cbf` / `ref_idx` 才是 `A + 2B`。写成 2 倍会让
//!    第二行第一个宏块开始错位。
//! 3. **邻居不可用时的默认值不是 0**：DC 块的 `cbf` 按 `b_intra`（=1）算；
//!    亮度 `coded_block_pattern` 按 `0xF`（四位全 1）算 —— 对应实现里邻居 cbp
//!    记成 `-1` 的那套写法。按 0 算，第一个宏块的上下文就偏 3。
//! 4. **`I_PCM` 之后算术解码器要重启**（对齐 → 原始样本 → `codIRange = 510`），
//!    不是继续用旧状态。
//!
//! 剩下的 4×4 残差问题，下一步的抓手是：用 `H264_CABAC_TRACE=1` 看 `ctx=134`
//! 之后的显著性 bin 序列，与"同一内容 CAVLC 编出来的系数"对照（注意：两种熵编码
//! 的 RD 量化可能选不同的系数，不能直接比数值，只能比"是否在合理位置"）。
//! ## 现状（2026-11-05）：P 切片已经能出画
//!
//! `ME_DECODE=software` 现在能解 **CAVLC 的 I/P 切片**（Baseline / 一部分 Main）。
//! 验收口径与 I 切片一样是**逐样本对比 FFmpeg 的原生解码器**（`--skip_loop_filter
//! all` 跳过参考侧的去块，这样能把"帧间预测/变换"与"去块"分开验）：
//!
//! | 用例 | 结果 |
//! |---|---|
//! | `partitions=none`（只 16×16）+ 运动 | 前两帧 **0 差异**，之后亮度差 ≤1 |
//! | `partitions=all`（含 8×8 子分区）| 前几帧 **0 差异**，个别宏块差几个样本 |
//! | 16×8 / 8×16 分区 | 能解，**个别宏块的 MV 与参考不一致**（见下）|
//! | 色度 | 帧间时**最多差 1–2**（插值取整口径），I 帧 **0 差异** |
//!
//! ### 三条已经踩过、别再犯的坑
//!
//! 1. **运动矢量预测不是"对可用邻居取中位数"**：要先数**参考索引与当前相同的
//!    邻居个数**（0 / 1 / ≥2 三种分支，见 [`inter::predict_mv`]）。写错的表现
//!    不是崩，而是"整块偏 5～20 个像素"。
//! 2. **跳过宏块（`mb_skip_run`）走的是 §8.4.1.1，不是 §8.4.1.3**：左或上邻居
//!    **不可用**就整体归零；左/上邻居是**帧内**则按 0 参与但不归零。两种"没有 MV"
//!    必须分开记（网格里 `-1` = 不可用、`-2` = 帧内），否则跳过的平坦区会被
//!    邻居的 MV 一路带走。
//! 3. **P 切片的循环不能"末宏块之后再读一个 `mb_skip_run`"**：那几位是
//!    `rbsp_trailing_bits`，症状是"多读了 1 位"。
//!
//! ### 已知限制
//!
//! **一张图拆成多个切片时还不通**：x264 在 320×240 以上会按高度自动切 3 个切片，
//! 实测第二个切片（`first_mb_in_slice > 0`）从第 7 个宏块起语法错位
//! （症状是 `QPY` 越界）。单切片素材（本项目全部回归用例）正常。
//! 排查入口：`H264_TRACE_MB=1` 会打出每个切片的 `first_mb`，配合
//! `H264_TRACE_MVP=1` / `H264_DUMP_MV=1` 定位。
//!
//! ## 下一步（按这个顺序接，别跳）
//!
//! 1. **§9.3 CABAC**：真实素材（x264/各硬件编码器的默认输出）**几乎全是 CABAC**，
//!    不接它等于自研软解只能解 Baseline。这是当前最大的缺口。
//! 2. **B 切片**：`mb_type` 表 7-14、双向预测、`direct` 模式、加权双向预测。
//! 3. **去块的帧间 `bS`**（`bs_for_edge` 的后两档）：目前帧间宏块之间的边按 0 处理，
//!    出来的是"没去块"的画面，与正常播放器相比块边缘会偏锐。
//! 4. 16×8 / 8×16 分区上仍有个别宏块的 MV 与参考不一致（`H264_DUMP_MV=1` 可以
//!    和参考解码器的运动矢量逐个对拍），需要继续缩小范围。
//!
//! ## 诊断开关
//!
//! | 开关 | 作用 |
//! |---|---|
//! | `H264_TRACE_FRAME=1` | 每帧一行：`poc / frame_num / 切片数 / MB 数 / 是否有残差` |
//! | `H264_TRACE_MB=1` | 逐宏块：`addr / mb_type / qp / cbp / 预测模式` |

pub mod cabac;
pub mod cabac_mb;
pub mod cabac_tables;
pub mod cavlc;
pub mod cavlc_tables;
pub mod deblock;
pub mod frame;
pub mod inter;
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
    /// 短期参考帧，**最近解码的在前**（滑动窗口维护，MSB 还没做 MMCO）。
    refs: Vec<RefFrame>,
    /// 参考帧上限（`max_num_ref_frames`）。
    max_refs: usize,
    /// POC 推导需要的"上一张参考图的 lsb/msb"（§8.2.1.1）。
    poc: PocState,
}

/// 一张短期参考帧。
struct RefFrame {
    frame: FrameBuf,
    frame_num: u32,
    poc: i32,
}

/// §8.2.1.1（`pic_order_cnt_type == 0`）要的上一次状态。
#[derive(Default, Clone, Copy)]
struct PocState {
    prev_msb: i32,
    prev_lsb: i32,
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
        let max_refs = cfg.sps.max_num_ref_frames.max(1) as usize;
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
            refs: Vec::new(),
            max_refs,
            poc: PocState::default(),
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
        let first = slices[0].header.clone();
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
        if std::env::var("H264_TRACE_MB").is_ok_and(|v| v != "0") {
            for s in slices {
                eprintln!(
                    "切片: first_mb={} type={} qp_delta={} deblock={} offset={} 字节={}",
                    s.header.first_mb_in_slice,
                    s.header.slice_type,
                    s.header.slice_qp_delta,
                    s.header.disable_deblocking_filter_idc,
                    s.header.slice_data_bit_offset,
                    s.raw.len()
                );
            }
        }
        let mut info = mb::PictureInfo::new(
            sps.coded_width() as usize / 16,
            sps.coded_height() as usize / 16,
        );
        info.pic_index = self.pictures;
        for s in slices {
            let cabac = s.pps.entropy_coding_mode_flag;
            if s.header.is_i() {
                if cabac {
                    cabac_mb::decode_i_slice(&s.header, &s.sps, &s.pps, &s.raw, &mut info)?;
                } else {
                    mb::decode_i_slice(&s.header, &s.sps, &s.pps, &s.raw, &mut info)?;
                }
            } else if s.header.is_p() {
                if cabac {
                    cabac_mb::decode_p_slice(&s.header, &s.sps, &s.pps, &s.raw, &mut info)?;
                } else {
                    mb::decode_p_slice(&s.header, &s.sps, &s.pps, &s.raw, &mut info)?;
                }
            } else {
                return Err(MediaError::Unsupported(format!(
                    "B 切片还没接（slice_type = {}）。P 切片已经能解，见 h264/mod.rs 抬头那张表",
                    s.header.slice_type
                )));
            }
        }

        // 诊断：把每个 4×4 块的 MV 打出来（与参考解码器的运动矢量对拍）
        if std::env::var("H264_DUMP_MV").is_ok_and(|v| v != "0") {
            let pic = self.pictures;
            for addr in 0..info.mb.len() {
                let mb = &info.mb[addr];
                if let Some(it) = &mb.inter {
                    for y4 in 0..4 {
                        for x4 in 0..4 {
                            let e = it.mv[y4][x4];
                            eprintln!(
                                "MV {pic} {} {} {} {} {} {}",
                                addr,
                                x4,
                                y4,
                                e.mv.x,
                                e.mv.y,
                                e.ref_idx
                            );
                        }
                    }
                }
            }
        }

        let poc = self.compute_poc(&first, sps);
        let ref_list: Vec<&FrameBuf> = self.refs.iter().map(|r| &r.frame).collect();
        let mut frame = FrameBuf::for_sps(sps);
        reconstruct::reconstruct_picture(&mut frame, &mut info, sps, &slices[0].pps, &ref_list, &first)?;
        // §8.7：去块滤波在整张图重建完成之后做（所以帧内预测读的是去块前样本）
        deblock::deblock_picture(&mut frame, &mut info, &slices[0].pps, &first)?;

        // 参考帧管理：滑动窗口（没有 MMCO 的那一档，参数集层已经拒掉了 MMCO）
        if first.is_reference() {
            if first.idr {
                self.refs.clear();
            }
            self.refs.insert(
                0,
                RefFrame {
                    frame: frame.clone(),
                    frame_num: first.frame_num,
                    poc,
                },
            );
            while self.refs.len() > self.max_refs {
                self.refs.pop();
            }
        }

        // 显示顺序：POC 小的先出。没有 B 帧时 reorder = 0，这里就是"解一张出一张"。
        self.ready.push(ReadyPic {
            frame,
            poc,
            pts_us,
        });
        Ok(self.take_ready())
    }

    /// §8.2.1 的 POC 推导。
    ///
    /// 只实现到了 `pic_order_cnt_type` 0 与 2 —— 这是现实里几乎全部素材的形态
    /// （x264 默认 type 0，关掉 B 帧后常见 type 2）。type 1 需要
    /// `offset_for_ref_frame[]` 的累计，等真碰到再补。
    fn compute_poc(&mut self, h: &SliceHeader, sps: &Sps) -> i32 {
        if h.idr {
            self.poc = PocState::default();
            return 0;
        }
        match sps.pic_order_cnt_type {
            0 => {
                let max_lsb = 1i32 << sps.log2_max_pic_order_cnt_lsb;
                let lsb = h.pic_order_cnt_lsb as i32;
                let (prev_msb, prev_lsb) = (self.poc.prev_msb, self.poc.prev_lsb);
                let mut msb = prev_msb;
                if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
                    msb = prev_msb + max_lsb;
                } else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
                    msb = prev_msb - max_lsb;
                }
                if h.is_reference() {
                    self.poc.prev_msb = msb;
                    self.poc.prev_lsb = lsb;
                }
                msb + lsb
            }
            // 帧编码：`PicOrderCnt = 2 * frame_num`（§8.2.1.2 的化简）
            2 => 2 * h.frame_num as i32,
            other => {
                let _ = other;
                2 * h.frame_num as i32
            }
        }
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
