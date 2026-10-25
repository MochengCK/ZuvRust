//! 自研 AV1 解码器：**帧级驱动**（自研软解的最后一块接线）。
//!
//! 这一层把下面几层拼成一条完整的解码管线，每帧的顺序照规范 §7.18 的
//! `decode_frame_wrapup` 抄：
//!
//! ```text
//! 采样（一个时间单元）＝ 一串 OBU
//!   ├─ 序列头 / 时间分隔符 / 元数据 → 更新参数 / 忽略
//!   ├─ 帧头（OBU_FRAME_HEADER / OBU_FRAME）
//!   │    ├─ show_existing_frame = 1 → 直接把参考槽里的像素当输出（不解码）
//!   │    └─ 否则：建 PendingFrame（网格 / CDF 上下文 / 运动场）
//!   ├─ 瓦片组（OBU_TILE_GROUP，或 OBU_FRAME 里自带的那段）
//!   └─ 瓦片收齐 ⇒ finish_frame( )
//!        ① 逐瓦片 decode_tile（重建像素）
//!        ② decode_frame_wrapup：去块 → CDEF → 上采样 → 环路恢复 → LrFrame
//!        ③ 运动场存储（§7.19）+ 参考帧更新（§7.20：像素 / CDF / 元数据）
//!        ④ show_frame ⇒ 输出一帧（NV12）
//! ```
//!
//! ## 三个契约（与宿主/上层对齐）
//!
//! 1. **`pts_us` 取"触发显示的那个采样"的时间戳**（不是被显示帧的编码时刻）：
//!    AV1 的显示事件 = `show_frame` 或 `show_existing_frame`，而容器里
//!    **一个采样就是一个时间单元**（一个显示时隙）——隐藏帧（`show_frame = 0`，
//!    alt-ref）不显示、不占时隙，稍后由某个 `show_existing_frame` 采样显示出来。
//!    所以"显示事件发生时的采样时间戳"就是这一帧的显示时间（规范 §7.18 的输出
//!    过程 + "每个时间单元恰好一个显示帧"的流约束）。
//! 2. **输出像素是最终像素**：经环路滤波与超分上采样（`LrFrame`），与参考帧
//!    存储的是同一份。
//! 3. **输出一律 8 位 4:2:0 NV12**：10/12 位右移，4:4:4/4:2:2 下采样，mono 填中灰
//!    —— 引擎对外的像素格式只有 NV12（见 `me_decode::VideoFrame`）。
//!
//! ## 关键帧的"参考帧重置"（§5.9.2）
//!
//! `frame_type == KEY_FRAME && show_frame == 1` 时要清空所有参考槽（有效性、像素、
//! CDF 全作废）—— 这是"随机访问点"的语义。放在帧头解析**之后**做：解析期间
//! 还要读参考帧的尺寸/顺序，清零了反而读错。

use super::flags;
use me_core::{CodecId, MediaError, Result};

use super::block::{decode_tile, BlockGrid};
use super::coeff::CoeffContexts;
use super::context::CdfContext;
use super::filters;
use super::frame_header::{
    FrameHeader, FrameType, HeaderTail, ObuLayerIds, RefFrameStore, PRIMARY_REF_NONE,
};
use super::intra::CurrFrame;
use super::lr::LrState;
use super::mv::{motion_field_estimation, motion_field_storage, MotionField, MotionFieldStore};
use super::obu::{ObuIter, ObuType};
use super::partition::MiGrid;
use super::sequence_header::SequenceHeader;
use super::tile::{parse_tile_group, Tile};
use crate::{VideoDecoder, VideoFrame};

/// 一个采样里"读到了什么"的摘要（诊断与错误文案共用）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleSummary {
    pub obu_count: usize,
    /// 有帧头/帧/瓦片组（= 这一帧真的有像素要重建）
    pub has_bitstream: bool,
    /// 采样的第一个序列头（一个采样里通常只在开头出现一次）
    pub sequence_header: Option<SequenceHeader>,
}

/// 解析一个 AV1 采样的结构（不重建像素）。
///
/// 这是"采样里有什么 OBU"的轻量观察面：CLI 的 `probe` 与 VideoToolbox 那一路
/// （`macos.rs`）都用它。
pub fn inspect_sample(data: &[u8]) -> Result<SampleSummary> {
    if data.is_empty() {
        return Err(MediaError::Bitstream("AV1 采样是空的".into()));
    }
    let mut it = ObuIter::new(data);
    let mut obu_count = 0usize;
    let mut has_bitstream = false;
    let mut sequence_header = None;
    while let Some(obu) = it.next_obu()? {
        obu_count += 1;
        if obu.header.obu_type == ObuType::SequenceHeader {
            let sh = SequenceHeader::parse(obu.payload)?;
            if sequence_header.is_none() {
                sequence_header = Some(sh);
            }
        }
        if obu.header.obu_type.carries_bitstream() {
            has_bitstream = true;
        }
    }
    if obu_count == 0 {
        return Err(MediaError::Bitstream("这段数据里没有 OBU".into()));
    }
    Ok(SampleSummary { obu_count, has_bitstream, sequence_header })
}

/// 一个瓦片的**自有副本**（瓦片组可能跨采样到达，所以不能借用采样的字节）。
#[derive(Debug, Clone)]
struct OwnedTile {
    index: u32,
    row: u32,
    col: u32,
    mi_row_start: u32,
    mi_row_end: u32,
    mi_col_start: u32,
    mi_col_end: u32,
    data: Vec<u8>,
}

impl OwnedTile {
    fn from_tile(t: &Tile<'_>) -> Self {
        Self {
            index: t.index,
            row: t.row,
            col: t.col,
            mi_row_start: t.mi_row_start,
            mi_row_end: t.mi_row_end,
            mi_col_start: t.mi_col_start,
            mi_col_end: t.mi_col_end,
            data: t.data.to_vec(),
        }
    }

    fn as_tile(&self) -> Tile<'_> {
        Tile {
            index: self.index,
            row: self.row,
            col: self.col,
            mi_row_start: self.mi_row_start,
            mi_row_end: self.mi_row_end,
            mi_col_start: self.mi_col_start,
            mi_col_end: self.mi_col_end,
            data: &self.data,
        }
    }
}

/// 正在解的一帧（帧头已读、瓦片可能还没收齐）。
struct PendingFrame {
    fh: FrameHeader,
    cdf: CdfContext,
    mi_grid: MiGrid,
    block_grid: BlockGrid,
    coeff_ctxs: CoeffContexts,
    /// 环路恢复的帧级状态（`LrType` / `LrSgrSet` / `LrSgrXqd` / `LrWiener`，§5.11.48）。
    /// 语法阶段填它、滤波阶段（`filters::decode_frame_wrapup`）读它。
    lr: LrState,
    /// `use_ref_frame_mvs` 时的运动场（帧头解析阶段算好）
    motion_field: Option<MotionField>,
    tiles: Vec<OwnedTile>,
    want_tiles: u32,
}

/// 后台滤镜线程的产物。
struct Filtered {
    lr: CurrFrame,
    block_grid: BlockGrid,
    mi_grid: MiGrid,
    fh: FrameHeader,
    seq: SequenceHeader,
    pts_us: i64,
    show_frame: bool,
    refresh_flags: u8,
}

/// 一帧的**后台滤镜任务**（去块 → CDEF → 超分 → LR）。语法 N+1 可与它重叠，
/// join 之后才提交参考像素并产出显示帧。
struct FilterJob {
    handle: std::thread::JoinHandle<Result<Filtered>>,
}

impl FilterJob {
    #[allow(clippy::too_many_arguments)]
    fn start(
        frame: CurrFrame,
        fh: FrameHeader,
        seq: SequenceHeader,
        block_grid: BlockGrid,
        mi_grid: MiGrid,
        lr: LrState,
        pts_us: i64,
        show_frame: bool,
        refresh_flags: u8,
    ) -> Result<Self> {
        let fh2 = fh.clone();
        let seq2 = seq.clone();
        let _ = (&fh2, &seq2);
        let handle = std::thread::Builder::new()
            .name("av1-filter".into())
            .spawn(move || {
                let lr_out = super::filters::decode_frame_wrapup(
                    frame,
                    &fh2,
                    &seq2,
                    &block_grid,
                    &mi_grid,
                    &lr,
                )?;
                Ok(Filtered {
                    lr: lr_out,
                    block_grid,
                    mi_grid,
                    fh: fh2,
                    seq: seq2,
                    pts_us,
                    show_frame,
                    refresh_flags,
                })
            })
            .map_err(|e| MediaError::Io(format!("启动 AV1 滤镜线程失败：{e}")))?;
        Ok(Self { handle })
    }

    fn join(self) -> Result<Filtered> {
        match self.handle.join() {
            Ok(r) => r,
            Err(_) => Err(MediaError::Bitstream("AV1 滤镜线程 panic".into())),
        }
    }
}

/// 自研 AV1 软解（进度见 [`crate::video`] 的分层表）。
pub struct Av1Decoder {
    /// 容器给的 `av1C`（ISO/IEC 14496-15 的 AV1CodecConfigurationRecord）
    config: Vec<u8>,
    /// 已经解析到的序列头（通常来自第一个采样；也可能来自 `av1C` 的 configOBUs）
    seq: Option<SequenceHeader>,
    /// 参考帧（元数据 + 像素 + CDF 上下文，§7.20）
    store: RefFrameStore,
    /// 运动场存储（§7.19，`use_ref_frame_mvs` 要用）
    mf_store: MotionFieldStore,
    /// 正在解、等瓦片组的那一帧
    pending: Option<PendingFrame>,
    /// 后台滤镜中的上一帧（`ME_FRAME_PIPE=1`）
    pending_filter: Option<FilterJob>,
    /// **跨帧复用的网格池**（`BlockGrid` + `MiGrid`）。
    ///
    /// 1080p 这两个网格加起来约 11 MB，每帧 `new` 一次的话光分配（大块走 `mmap`、
    /// 每页首次写入缺页、释放时 `madvise` + `vm_deallocate`）就要 **3.07 ms/帧**。
    /// 尺寸没变就复用同一份、只 `reset_all` 重填（语义与 `new` 完全一致）。
    grid_pool: Option<(BlockGrid, MiGrid)>,
    /// 上一帧的 `new_pending` 耗时（网格 + 系数上下文；`AV1_PHASE_TIMING` 用）。
    /// ⚠️ 它**不在** `tiles`/`filters` 里 —— 那两段只包 `finish_frame`，
    /// 而网格是在**帧头解析**时就建好的。要算"单帧完整预算"必须把它加进来。
    last_setup_ms: f64,
    /// `setup` 的三段拆账（grid / ctx / mf），`AV1_PHASE_TIMING` 打出来。
    last_setup_parts: (f64, f64, f64),
    /// **跨帧复用的运动场缓冲**（1080p 一份 ≈ 2 MB；每帧新开要走 mmap + 512 次缺页）。
    mf_buf: Option<MotionField>,
    /// 当前采样的时间戳（显示事件用它，见模块抬头的契约 1）
    sample_pts_us: i64,
    /// 诊断：已解帧数（含 show_existing）
    frames_parsed: u64,
    /// 诊断：已解**带瓦片**的帧数（不含 show_existing —— 计数口径与 libaom
    /// 插桩的 `MARK tile frame=` 行对齐，见 `block::TRACE_TILE_FRAME_NO`）
    tile_frames: u64,
    /// 诊断：最近一帧的非压缩头
    last_frame: Option<FrameHeader>,
    /// 诊断：最近一帧的瓦片摘要（瓦片数、字节总数）
    last_tiles: Option<(usize, usize)>,
    /// 诊断：最近一帧的块数
    last_blocks: u64,
    /// 诊断：跳过的"没有显示事件"的采样数
    header_only_samples: u64,
}

impl Av1Decoder {
    /// 用 `av1C` 建解码器。
    ///
    /// `av1C` 是**必需**的：它带着 profile/位深/色度采样，是"这一轨能不能解"的
    /// 第一道判据（引擎里 AV1 的 extradata 就是它，不存在 Annex-B 形态）。
    ///
    /// 记录之后的字节是 **configOBUs**，里面通常就有一份序列头（B 站那份素材就是）
    /// —— 拿它当初始序列头很有用：**seek 到流中间时，那个采样里没有序列头**。
    /// 解析失败不算错误（不是所有 `av1C` 都带 configOBUs），如实忽略即可。
    pub fn open(config: &[u8]) -> Result<Self> {
        validate_av1c(config)?;
        let mut sequence_header = None;
        if config.len() > 4 {
            let mut it = ObuIter::new(&config[4..]);
            while let Ok(Some(obu)) = it.next_obu() {
                if obu.header.obu_type == ObuType::SequenceHeader {
                    if let Ok(sh) = SequenceHeader::parse(obu.payload) {
                        sequence_header = Some(sh);
                        break;
                    }
                }
            }
        }
        Ok(Self {
            config: config.to_vec(),
            seq: sequence_header,
            store: RefFrameStore::new(),
            mf_store: MotionFieldStore::new(),
            pending: None,
            pending_filter: None,
            grid_pool: None,
            last_setup_ms: 0.0,
            last_setup_parts: (0.0, 0.0, 0.0),
            mf_buf: None,
            sample_pts_us: 0,
            frames_parsed: 0,
            tile_frames: 0,
            last_frame: None,
            last_tiles: None,
            last_blocks: 0,
            header_only_samples: 0,
        })
    }

    /// 已经读到的序列头（诊断/测试用）。
    pub fn sequence_header(&self) -> Option<&SequenceHeader> {
        self.seq.as_ref()
    }

    /// 最近一帧的非压缩头（诊断/测试用）。
    pub fn last_frame(&self) -> Option<&FrameHeader> {
        self.last_frame.as_ref()
    }

    /// 最近一帧的瓦片摘要：`(瓦片数, 瓦片字节总数)`（诊断/测试用）。
    pub fn last_tiles(&self) -> Option<(usize, usize)> {
        self.last_tiles
    }

    /// 最近一帧分区树切出来的块数（诊断用）。
    pub fn last_blocks(&self) -> u64 {
        self.last_blocks
    }

    /// `av1C` 原样（诊断用；硬解那一路要拿它建格式描述）。
    pub fn config(&self) -> &[u8] {
        &self.config
    }

    /// 关键帧刷新时的整体重置（§5.9.2）：参考帧有效性、像素一起作废。
    /// （CDF 上下文不用清：关键帧的 `primary_ref_frame` 必为 NONE，走默认表。）
    fn reset_reference_state(&mut self) {
        self.store.reset();
        self.store.clear_pixels();
    }

    /// `show_existing_frame`：不解码，直接把参考槽里的像素当这一帧的输出。
    ///
    /// 参考槽里存的是**未加颗粒**的像素，所以输出前要按该槽保存的颗粒参数
    /// （§5.9.30 的 `load_grain_params(frame_to_show_map_idx)`）补上颗粒。
    fn show_existing(&mut self, fh: &FrameHeader) -> Result<Option<VideoFrame>> {
        let slot = fh.frame_to_show_map_idx as usize;
        // `cloned()` 是 **`Arc` 的引用计数加一**，不是帧拷贝 —— 共享像素零成本。
        let mut frame = self
            .store
            .pixel(slot)
            .ok_or_else(|| {
                MediaError::Bitstream(format!(
                    "show_existing_frame 指向参考槽 {slot}，但那个槽还没有像素（流被截断，或 seek 到了没有随机访问点的位置）"
                ))
            })?;
        let fg = &self.store.grain[slot];
        if fg.apply_grain {
            let is_identity = self
                .seq
                .as_ref()
                .is_some_and(|s| s.color.matrix_coefficients == 0);
            // 加颗粒要改像素 —— 只有这一档才分裂出一份独占副本（原来是**无条件**
            // 整帧 `clone()`，1080p 每帧白拷 6.2 MB）。
            let own = std::sync::Arc::make_mut(&mut frame);
            super::grain::apply_film_grain(own, fg, is_identity);
        }
        Ok(Some(nv12_frame(&frame, self.sample_pts_us)?))
    }

    /// 建一帧的待解状态（网格 / CDF 上下文 / 运动场）。
    fn new_pending(&mut self, fh: FrameHeader, seq: &SequenceHeader) -> PendingFrame {
        let mi_rows = fh.mi_rows as usize;
        let mi_cols = fh.mi_cols as usize;
        let _t_setup = std::time::Instant::now();
        // 网格从池里取（尺寸一般一模一样）；`reset_all` 的语义与 `new` 逐格一致。
        let (mut block_grid, mut mi_grid) = self
            .grid_pool
            .take()
            .map(|(b, m)| {
                // 尺寸变了也能用：`reset_all` 会把长度改到 `mi_rows × mi_cols`。
                (b, m)
            })
            .unwrap_or_else(|| (BlockGrid::new(0, 0), MiGrid::new(0, 0)));
        block_grid.reset_all(mi_rows, mi_cols);
        mi_grid.reset_all(mi_rows, mi_cols);
        let _grid_ms = _t_setup.elapsed().as_secs_f64() * 1e3;
        block_grid.latch_prev_segment_ids();
        // ⚠️ **不要**在这里再调 `reset_frame()`（2026-10-05 删掉）。
        // `reset_all` 刚刚把 `cdef_idx` / `lf_tx_sizes` / `above|left_seg_pred_ctx` /
        // `written` **全都重填过一遍**，而 `reset_frame` 填的正是这同一组数组 ⇒
        // 每帧白写 ≈ 1 MB（cdef_idx 是 i32，1080p 就 0.52 MB）。
        // `BlockGrid::reset_frame` 这个方法本身留着（语义清楚、单测会用），只是
        // 生产路径上不再有"连着重置两次"这件事。
        let num_planes = if seq.color.mono_chrome { 1usize } else { 3 };
        let coeff_ctxs = CoeffContexts::new(
            mi_rows,
            mi_cols,
            seq.color.subsampling_x,
            seq.color.subsampling_y,
            num_planes as u8,
        );
        let _ctx_ms = _t_setup.elapsed().as_secs_f64() * 1e3 - _grid_ms;
        // `use_ref_frame_mvs` 时在帧头解析阶段就把运动场估出来（规范 §7.9）。
        // ⚠️ 它也算进 `setup`（2026-10-04）：`sample` 里 `project_frame` 常年占几百个
        // 样本，而它既不在 `tiles`/`filters`、原来也不在 `setup` 里 —— 那会让
        // "单帧预算 = setup + tiles + filters" 少算这一块（`MFMV_STACK_SIZE` 个源帧
        // × 全部 8×8 格的投影）。
        let motion_field = if fh.use_ref_frame_mvs && !fh.is_intra {
            let mut f = self.mf_buf.take().unwrap_or_default();
            motion_field_estimation(&mut f, &fh, seq, &self.mf_store);
            Some(f)
        } else {
            None
        };
        let _mf_ms = _t_setup.elapsed().as_secs_f64() * 1e3 - _grid_ms - _ctx_ms;
        self.last_setup_ms = _grid_ms + _ctx_ms + _mf_ms;
        self.last_setup_parts = (_grid_ms, _ctx_ms, _mf_ms);
        // CDF 上下文：帧内/关键帧用默认表，帧间帧从参考槽整份载入（§7.20）
        let cdf = if fh.primary_ref_frame == PRIMARY_REF_NONE {
            CdfContext::defaults(fh.params.quant.base_q_idx as u32)
        } else {
            self.store.load_cdfs(fh.ref_frame_idx[fh.primary_ref_frame as usize])
        };
        let want_tiles = fh.params.tile_info.num_tiles();
        // 环路恢复的帧级数组要按 `LoopRestorationSize` 与帧尺寸先开好
        let lr = LrState::new(seq, &fh.params.lr, fh.frame_height, fh.upscaled_width);
        PendingFrame {
            fh,
            cdf,
            mi_grid,
            block_grid,
            coeff_ctxs,
            lr,
            motion_field,
            tiles: Vec::new(),
            want_tiles,
        }
    }

    fn pending_complete(&self) -> bool {
        self.pending
            .as_ref()
            .is_some_and(|pf| pf.tiles.len() as u32 >= pf.want_tiles)
    }

    /// `finish_frame`：瓦片收齐后把这一帧解出来，并按 §7.18–§7.20 收尾。
    fn finish_frame(&mut self, mut pf: PendingFrame) -> Result<Option<VideoFrame>> {
        let seq = self.seq.clone().ok_or_else(|| {
            MediaError::Bitstream("解码帧时没有序列头（av1C 与码流都没给）".into())
        })?;
        let num_planes = if seq.color.mono_chrome { 1usize } else { 3 };
        // 瓦片可能分多个瓦片组到达；按序号排好再解（每个瓦片各有独立的算术解码器，
        // 顺序不影响正确性，但排好更便于排查）
        pf.tiles.sort_by_key(|t| t.index);
        self.last_tiles = Some((
            pf.tiles.len(),
            pf.tiles.iter().map(|t| t.data.len()).sum::<usize>(),
        ));
        let mut frame = CurrFrame::new(
            pf.fh.frame_width as usize,
            pf.fh.frame_height as usize,
            seq.color.subsampling_x,
            seq.color.subsampling_y,
            num_planes,
            seq.color.bit_depth,
        );
        if flags::trace_cfl() {
            println!("=== FRAME #{} ===", self.frames_parsed);
        }
        super::block::TRACE_FRAME_NO.store(self.frames_parsed, std::sync::atomic::Ordering::Relaxed);
        super::block::TRACE_TILE_FRAME_NO
            .store(self.tile_frames, std::sync::atomic::Ordering::Relaxed);
        let mut blocks = 0u64;
        let phase_t = flags::phase_timing();
        let t_tiles = std::time::Instant::now();
        // 诊断（`AV1_SYNTAX_ONLY=1`）：只读语法、不重建（见 `flags::syntax_only`）。
        let syntax_only = flags::syntax_only();
        for t in &pf.tiles {
            let tile = t.as_tile();
            let r = decode_tile(
                &tile,
                &seq,
                &pf.fh,
                &self.store,
                pf.motion_field.as_ref(),
                &mut pf.cdf,
                &mut pf.mi_grid,
                &mut pf.block_grid,
                &mut pf.coeff_ctxs,
                &mut pf.lr,
                if syntax_only { None } else { Some(&mut frame) },
            )?;
            blocks += r.blocks.blocks;
        }
        self.last_blocks = blocks;
        self.tile_frames += 1;

        if flags::trace_tx1() && self.frames_parsed < 2 && blocks > 0 {
            let p = frame.plane(0);
            println!(
                "收尾前 plane0：(0,0)={} (4,0)={} (0,4)={} (4,4)={} (100,50)={}",
                p.get(0, 0), p.get(4, 0), p.get(0, 4), p.get(4, 4), p.get(100, 50)
            );
        }
        if let Some((lx, ly, r)) = flags::trace_region() {
            trace_region_line(&frame, lx, ly, r, "pre", &pf.fh);
        }
        let tiles_ms = t_tiles.elapsed().as_secs_f64() * 1e3;
        let t_filters = std::time::Instant::now();
        // ① 后处理（去块 → CDEF → 超分 → 环路恢复）→ LrFrame；这份像素同时是
        //    参考帧与输出帧的来源。
        //    `AV1_SYNTAX_ONLY=1` 时跳过滤波（像素是垃圾），但**参考帧槽仍照常更新** ——
        //    后续帧的语法（`is_scaled` / `num_warp_samples` 等）依赖参考帧已就绪，
        //    早退会让那些判据变样、语法跟着错位（踩过：第二个带瓦片的帧就报"多读 2 位"）。
        let lr = if syntax_only {
            frame
        } else {
            filters::decode_frame_wrapup(
                frame,
                &pf.fh,
                &seq,
                &pf.block_grid,
                &pf.mi_grid,
                &pf.lr,
            )?
        };
        // 参考帧与输出共享**同一份**像素 ⇒ 包成 `Arc`（`store_pixels` 零拷贝；
        // 1080p 一帧 6.2 MB，原来每个被刷新的槽都要拷一份 ⇒ 每帧 6~12 MB 的 memmove）。
        // 真要改像素时（加胶片颗粒）走 `Arc::make_mut` —— 独占时零拷贝，共享时才分裂。
        let mut lr = std::sync::Arc::new(lr);
        if flags::trace_tx1() && self.frames_parsed < 2 {
            let p = lr.plane(0);
            println!(
                "收尾后 plane0：(0,0)={} (4,0)={} (0,4)={} (4,4)={} (100,50)={}",
                p.get(0, 0), p.get(4, 0), p.get(0, 4), p.get(4, 4), p.get(100, 50)
            );
        }
        if let Some((lx, ly, r)) = flags::trace_region() {
            trace_region_line(&lr, lx, ly, r, "post", &pf.fh);
        }

        let filters_ms = t_filters.elapsed().as_secs_f64() * 1e3;
        let t_ref = std::time::Instant::now();
        // ② 参考帧更新（§7.19 运动场存储 + §7.20 像素/CDF/元数据）
        if pf.fh.refresh_frame_flags != 0 {
            // 2026-10-24：**最后一个刷新槽直接 move**（其余才 clone）。1080p 一份
            // `SavedMvs` 是 0.5 MB + 1.0 MB —— 原来每个刷新槽都拷一遍，单槽刷新
            // （这份素材里的常态）等于白拷 1.5 MB/帧。槽里的旧内容在本帧解码结束后
            // 已经死了 ⇒ 顺手拿它当 `motion_field_storage` 的落点，省掉每帧 1.5 MB
            // 的 `vec![]`（新页 + 首次写入缺页）。
            let refresh = pf.fh.refresh_frame_flags;
            let last_slot = (0..8usize).rev().find(|s| (refresh >> s) & 1 == 1);
            let (scratch_rf, scratch_mvs) = match last_slot {
                Some(s) => (
                    std::mem::take(&mut self.mf_store.slots[s].ref_frames),
                    std::mem::take(&mut self.mf_store.slots[s].mvs),
                ),
                None => (Vec::new(), Vec::new()),
            };
            let (mut saved_rf, mut saved_mvs) = super::mv::motion_field_storage_into(
                &pf.block_grid,
                &pf.fh,
                &seq,
                &self.store,
                scratch_rf,
                scratch_mvs,
            );
            for slot in 0..8usize {
                if refresh >> slot & 1 == 1 {
                    let (rf, mvs) = if Some(slot) == last_slot {
                        (std::mem::take(&mut saved_rf), std::mem::take(&mut saved_mvs))
                    } else {
                        (saved_rf.clone(), saved_mvs.clone())
                    };
                    self.mf_store.save(
                        slot,
                        pf.fh.mi_rows as usize,
                        pf.fh.mi_cols as usize,
                        pf.fh.is_intra,
                        rf,
                        mvs,
                        pf.fh.order_hints,
                    );
                }
            }
            self.store
                .store_pixels(pf.fh.refresh_frame_flags, std::sync::Arc::clone(&lr));
            if !pf.fh.disable_frame_end_update_cdf {
                self.store.save_cdfs(pf.fh.refresh_frame_flags, &pf.cdf);
            }
            self.store.refresh(&pf.fh);
        }

        // ③ 输出（§7.18）：显示事件发生在这一帧完成之时。
        //    胶片颗粒只加在**输出**上（参考帧槽里存的是加颗粒之前的 LrFrame），
        //    所以这里必须在 `store_pixels` 之后才动 `lr`。
        let ref_ms = t_ref.elapsed().as_secs_f64() * 1e3;
        self.frames_parsed += 1;
        if super::flags::trace_pool() {
            let (h, m, g, b, n) = super::pool::stats();
            eprintln!(
                "POOL frame={} hits={h} misses={m} gave={g} pooled={:.1}MB bufs={n}",
                self.frames_parsed,
                b as f64 / (1 << 20) as f64
            );
        }
        let out = if pf.fh.show_frame {
            let t_out = std::time::Instant::now();
            let fg = &pf.fh.params.film_grain;
            if fg.apply_grain {
                let is_identity = seq.color.matrix_coefficients == 0;
                super::grain::apply_film_grain(std::sync::Arc::make_mut(&mut lr), fg, is_identity);
            }
            let frame = nv12_frame(&lr, self.sample_pts_us)?;
            if phase_t {
                let (mc_ms, txb_ms, mark_ms) = super::block::take_replay_dbg();
                eprintln!(
                    "PHASE frame={} shown=1 tiles={:.1}ms filters={:.1}ms ref={:.1}ms out={:.1}ms setup={:.1}ms mc={:.1}ms txb={:.1}ms mark={:.1}ms grid={:.1}ms ctx={:.1}ms mf={:.1}ms",
                    self.frames_parsed,
                    tiles_ms,
                    filters_ms,
                    ref_ms,
                    t_out.elapsed().as_secs_f64() * 1e3,
                    self.last_setup_ms,
                    mc_ms,
                    txb_ms,
                    mark_ms,
                    self.last_setup_parts.0,
                    self.last_setup_parts.1,
                    self.last_setup_parts.2,
                );
            }
            Some(frame)
        } else {
            if phase_t {
                let (mc_ms, txb_ms, mark_ms) = super::block::take_replay_dbg();
                eprintln!(
                    "PHASE frame={} shown=0 tiles={:.1}ms filters={:.1}ms ref={:.1}ms setup={:.1}ms mc={:.1}ms txb={:.1}ms mark={:.1}ms grid={:.1}ms ctx={:.1}ms mf={:.1}ms",
                    self.frames_parsed, tiles_ms, filters_ms, ref_ms, self.last_setup_ms, mc_ms, txb_ms, mark_ms,
                    self.last_setup_parts.0, self.last_setup_parts.1, self.last_setup_parts.2,
                );
            }
            None
        };
        // 网格回池（只增不减；尺寸变了 `reset_all` 会自己调长度）。见 `grid_pool`。
        self.grid_pool = Some((pf.block_grid, pf.mi_grid));
        // 运动场缓冲也回池（见 `mf_buf`）。
        if pf.motion_field.is_some() {
            self.mf_buf = pf.motion_field;
        }
        Ok(out)
    }
}

/// 把瓦片组加进待解帧（`OBU_TILE_GROUP` 或 `OBU_FRAME` 自带的瓦片段）。
fn add_tile_group(pf: &mut PendingFrame, body: &[u8]) -> Result<()> {
    let group = parse_tile_group(body, &pf.fh.params.tile_info)?;
    for t in &group.tiles {
        pf.tiles.push(OwnedTile::from_tile(t));
    }
    if pf.tiles.len() as u32 > pf.want_tiles {
        return Err(MediaError::Bitstream(format!(
            "收到的瓦片比 tile_info 声明的多：{} > {}",
            pf.tiles.len(),
            pf.want_tiles
        )));
    }
    Ok(())
}

impl Av1Decoder {
    /// 两帧流水线的帧完成处理：当前帧语法 parse 先跑（与上一帧的后台滤镜重叠），
    /// 然后 join 上一帧、提交像素/输出，再回放当前帧并启动它的滤镜。
    fn finish_frame_pipeline(
        &mut self,
        mut pf: PendingFrame,
        out: &mut Vec<VideoFrame>,
    ) -> Result<()> {
        let pipeline = flags::frame_pipe()
            && flags::row_mt()
            && !flags::syntax_only()
            && pf.want_tiles == 1
            && pf.tiles.len() == 1;
        if !pipeline {
            self.drain_pending_filter(out)?;
            if let Some(vf) = self.finish_frame(pf)? {
                out.push(vf);
            }
            return Ok(());
        }
        let seq = self.seq.clone().ok_or_else(|| {
            MediaError::Bitstream("解码帧时没有序列头（av1C 与码流都没给）".into())
        })?;
        let num_planes = if seq.color.mono_chrome { 1usize } else { 3 };
        let pts_us = self.sample_pts_us;
        let mut frame = CurrFrame::new(
            pf.fh.frame_width as usize,
            pf.fh.frame_height as usize,
            seq.color.subsampling_x,
            seq.color.subsampling_y,
            num_planes,
            seq.color.bit_depth,
        );
        pf.tiles.sort_by_key(|t| t.index);
        self.last_tiles = Some((1, pf.tiles[0].data.len()));
        super::block::TRACE_FRAME_NO
            .store(self.frames_parsed, std::sync::atomic::Ordering::Relaxed);
        super::block::TRACE_TILE_FRAME_NO
            .store(self.tile_frames, std::sync::atomic::Ordering::Relaxed);

        // ① 先只跑语法（与上一帧的后台滤镜重叠）。
        let tile = pf.tiles[0].as_tile();
        let parsed = super::block::parse_tile(
            &tile,
            &seq,
            &pf.fh,
            &self.store,
            pf.motion_field.as_ref(),
            &mut pf.cdf,
            &mut pf.mi_grid,
            &mut pf.block_grid,
            &mut pf.coeff_ctxs,
            &mut pf.lr,
            Some(&mut frame),
        )?;
        drop(tile);
        let mut pending = Some(parsed);
        let mut blocks = 0u64;

        if self.pending_filter.is_none() {
            // 第一帧：没有可重叠的滤镜，直接回放并启动。
            let p = pending.take().expect("刚放进去");
            let r = super::block::replay_tile(p, &mut pf.mi_grid)?;
            blocks += r.blocks.blocks;
            self.start_filter_job(pf, frame, seq, pts_us, blocks)?;
            return Ok(());
        }

        // ② 语法已就绪；join 上一帧的滤镜并提交参考像素/输出。
        let job = self.pending_filter.take().expect("刚判过 Some");
        let filtered = job.join()?;
        let Filtered {
            lr: filtered_lr,
            block_grid,
            mi_grid,
            fh: filtered_fh,
            seq: filtered_seq,
            pts_us: filtered_pts,
            show_frame,
            refresh_flags,
        } = filtered;
        let lr = std::sync::Arc::new(filtered_lr);
        self.store
            .store_pixels(refresh_flags, std::sync::Arc::clone(&lr));
        self.grid_pool = Some((block_grid, mi_grid));
        if show_frame {
            let mut lr = lr;
            let fg = &filtered_fh.params.film_grain;
            if fg.apply_grain {
                let is_identity = filtered_seq.color.matrix_coefficients == 0;
                super::grain::apply_film_grain(
                    std::sync::Arc::make_mut(&mut lr),
                    fg,
                    is_identity,
                );
            }
            out.push(nv12_frame(&lr, filtered_pts)?);
        }

        // ③ 上一帧像素已落槽 ⇒ 刷新快照，回放当前帧，启动它的滤镜。
        pending
            .as_mut()
            .expect("pending 还在")
            .refresh_refs(&self.store);
        let r = super::block::replay_tile(pending.take().expect("刚判过"), &mut pf.mi_grid)?;
        blocks += r.blocks.blocks;
        self.start_filter_job(pf, frame, seq, pts_us, blocks)?;
        Ok(())
    }

    /// 提交一帧的参考元数据/CDF，并把滤镜丢到后台线程。
    fn start_filter_job(
        &mut self,
        mut pf: PendingFrame,
        frame: CurrFrame,
        seq: SequenceHeader,
        pts_us: i64,
        blocks: u64,
    ) -> Result<()> {
        let refresh_flags = pf.fh.refresh_frame_flags;
        let show_frame = pf.fh.show_frame;
        self.last_blocks = blocks;
        self.tile_frames += 1;
        self.frames_parsed += 1;
        if refresh_flags != 0 {
            if let Some((saved_rf, saved_mvs)) =
                Some(motion_field_storage(&pf.block_grid, &pf.fh, &seq, &self.store))
            {
                for slot in 0..8usize {
                    if (refresh_flags >> slot) & 1 == 1 {
                        self.mf_store.save(
                            slot,
                            pf.fh.mi_rows as usize,
                            pf.fh.mi_cols as usize,
                            pf.fh.is_intra,
                            saved_rf.clone(),
                            saved_mvs.clone(),
                            pf.fh.order_hints,
                        );
                    }
                }
            }
            if !pf.fh.disable_frame_end_update_cdf {
                self.store.save_cdfs(refresh_flags, &pf.cdf);
            }
            self.store.refresh(&pf.fh);
        }
        let job = FilterJob::start(
            frame,
            pf.fh.clone(),
            seq,
            pf.block_grid,
            pf.mi_grid,
            pf.lr,
            pts_us,
            show_frame,
            refresh_flags,
        )?;
        self.pending_filter = Some(job);
        if pf.motion_field.is_some() {
            self.mf_buf = pf.motion_field.take();
        }
        Ok(())
    }

    /// join 后台滤镜、提交像素并产出显示帧（顺序路径/收尾用）。
    fn drain_pending_filter(&mut self, out: &mut Vec<VideoFrame>) -> Result<()> {
        let Some(job) = self.pending_filter.take() else {
            return Ok(());
        };
        let filtered = job.join()?;
        let Filtered {
            lr: filtered_lr,
            block_grid,
            mi_grid,
            fh: filtered_fh,
            seq: filtered_seq,
            pts_us: filtered_pts,
            show_frame,
            refresh_flags,
        } = filtered;
        let lr = std::sync::Arc::new(filtered_lr);
        self.store
            .store_pixels(refresh_flags, std::sync::Arc::clone(&lr));
        self.grid_pool = Some((block_grid, mi_grid));
        if show_frame {
            let mut lr = lr;
            let fg = &filtered_fh.params.film_grain;
            if fg.apply_grain {
                let is_identity = filtered_seq.color.matrix_coefficients == 0;
                super::grain::apply_film_grain(
                    std::sync::Arc::make_mut(&mut lr),
                    fg,
                    is_identity,
                );
            }
            out.push(nv12_frame(&lr, filtered_pts)?);
        }
        Ok(())
    }
}

impl VideoDecoder for Av1Decoder {
    fn decode(&mut self, sample: &[u8], pts_us: i64) -> Result<Vec<VideoFrame>> {
        if sample.is_empty() {
            return Err(MediaError::Bitstream("AV1 采样是空的".into()));
        }
        self.sample_pts_us = pts_us;
        let mut out = Vec::new();
        let mut it = ObuIter::new(sample);
        while let Some(obu) = it.next_obu()? {
            match obu.header.obu_type {
                ObuType::SequenceHeader => {
                    let sh = SequenceHeader::parse(obu.payload)?;
                    // 序列头换了（尺寸/位深/色度任一变化）⇒ 旧参考帧整体作废
                    if let Some(old) = &self.seq {
                        if old.color.bit_depth != sh.color.bit_depth
                            || old.color.subsampling_x != sh.color.subsampling_x
                            || old.color.subsampling_y != sh.color.subsampling_y
                            || old.color.mono_chrome != sh.color.mono_chrome
                            || old.max_frame_width != sh.max_frame_width
                            || old.max_frame_height != sh.max_frame_height
                        {
                            self.drain_pending_filter(&mut out)?;
                            self.reset_reference_state();
                        }
                    }
                    self.seq = Some(sh);
                }
                ObuType::TemporalDelimiter
                | ObuType::Metadata
                | ObuType::Padding
                | ObuType::RedundantFrameHeader => {}
                ObuType::Frame | ObuType::FrameHeader => {
                    let seq = self.seq.clone().ok_or_else(|| {
                        MediaError::Bitstream(
                            "这段 AV1 采样里没有序列头，而 av1C 里也没有可用的 configOBUs"
                                .into(),
                        )
                    })?;
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
                    let fh = FrameHeader::parse(obu.payload, &seq, &mut self.store, ids, tail)?;
                    self.last_frame = Some(fh.clone());
                    // 关键帧 + 显示 ⇒ 参考帧状态整体重置（§5.9.2）。放在解析之后：
                    // 解析期间还要用旧参考帧读尺寸/顺序。
                    if fh.frame_type == FrameType::Key && fh.show_frame {
                        self.drain_pending_filter(&mut out)?;
                        self.reset_reference_state();
                    }
                    if fh.show_existing_frame {
                        self.drain_pending_filter(&mut out)?;
                        self.frames_parsed += 1;
                        if let Some(vf) = self.show_existing(&fh)? {
                            out.push(vf);
                        }
                        continue;
                    }
                    if let Some(pf) = &self.pending {
                        if (pf.tiles.len() as u32) < pf.want_tiles {
                            return Err(MediaError::Bitstream(format!(
                                "上一帧（{} 个瓦片）还没收齐瓦片组，就出现了新的帧头",
                                pf.want_tiles
                            )));
                        }
                    }
                    let mut pf = self.new_pending(fh, &seq);
                    if obu.header.obu_type == ObuType::Frame {
                        let body = &obu.payload[pf.fh.bits_consumed / 8..];
                        add_tile_group(&mut pf, body)?;
                    }
                    self.pending = Some(pf);
                    if self.pending_complete() {
                        let pf = self.pending.take().expect("刚放进去");
                        self.finish_frame_pipeline(pf, &mut out)?;
                    }
                }
                ObuType::TileGroup => {
                    if self.pending.is_none() {
                        return Err(MediaError::Bitstream(
                            "瓦片组 OBU 之前没有帧头（流被截断？）".into(),
                        ));
                    }
                    {
                        let pf = self.pending.as_mut().expect("刚判过");
                        add_tile_group(pf, obu.payload)?;
                    }
                    if self.pending_complete() {
                        let pf = self.pending.take().expect("刚判过 complete");
                        self.finish_frame_pipeline(pf, &mut out)?;
                    }
                }
                other => {
                    return Err(MediaError::Unsupported(format!(
                        "AV1 的这个 OBU 类型还没接：{}",
                        other.name()
                    )));
                }
            }
        }
        if out.is_empty() && self.pending.is_none() && self.pending_filter.is_none() {
            self.header_only_samples += 1;
        }
        Ok(out)
    }

    fn flush(&mut self) -> Result<Vec<VideoFrame>> {
        // 两帧流水线会把最后一帧的输出推迟到下一次 decode/flush；这里收尾。
        // 半截的帧（瓦片没收齐）是数据问题，不在这里假装补全。
        let mut out = Vec::new();
        self.drain_pending_filter(&mut out)?;
        Ok(out)
    }

    fn backend(&self) -> &'static str {
        "self-av1"
    }
}

/// `AV1_TRACE_REGION=<lx>,<ly>[,r]` 的一行输出：按亮度坐标打周围 `(2r+1)²` 的 Y 与
/// U/V（色度坐标 = 亮度坐标 >> 1，本插桩只用于 4:2:0 的真素材逐帧定位）。
///
/// `pre` = 环路滤波前、`post` = 环路滤波后（参考槽里存的就是后者）。
fn trace_region_line(frame: &CurrFrame, lx: i32, ly: i32, radius: i32, tag: &str, fh: &FrameHeader) {
    let mut s = format!(
        "FREG {tag} oh={} show={}",
        fh.order_hint,
        u8::from(fh.show_frame)
    );
    for (plane, name, cx, cy) in [(0u8, "y", lx, ly), (1, "u", lx >> 1, ly >> 1), (2, "v", lx >> 1, ly >> 1)] {
        if plane as usize >= frame.num_planes {
            continue;
        }
        let p = frame.plane(plane);
        s.push_str(&format!(" {name}="));
        for r in -radius..=radius {
            for c in -radius..=radius {
                let (yy, xx) = (cy + r, cx + c);
                if yy < 0 || xx < 0 || yy >= p.h as i32 || xx >= p.w as i32 {
                    s.push_str(" .");
                } else {
                    s.push_str(&format!(" {}", p.get(xx as usize, yy as usize)));
                }
            }
        }
    }
    eprintln!("{s}");
}

/// `CurrFrame`（或存进参考槽的 `LrFrame`）→ 引擎对外的 **NV12** 帧。
///
/// 位深与色度格式的归一化就发生在这里（见模块抬头的契约 3）：
/// - 8 位直接搬；10/12 位右移到 8 位；
/// - 4:2:0 的 CbCr 直接交织；4:2:2 横向平均；4:4:4 双双平均；mono 填 128。
fn nv12_frame(frame: &CurrFrame, pts_us: i64) -> Result<VideoFrame> {
    let w = frame.planes[0].w;
    let h = frame.planes[0].h;
    if w % 2 != 0 || h % 2 != 0 {
        return Err(MediaError::Unsupported(format!(
            "AV1 输出要求偶数尺寸（NV12），这一帧是 {w}×{h}"
        )));
    }
    if flags::lr_stats() {
        use std::sync::atomic::Ordering::Relaxed;
        let f = |a: &std::sync::atomic::AtomicU64| a.swap(0, Relaxed) as f64 / 1e6;
        eprintln!(
            "SGR calls={} box0={:.2}ms box1={:.2}ms final={:.2}ms",
            super::lr::SGR_CALLS.swap(0, Relaxed),
            f(&super::lr::SGR_BOX0_NS),
            f(&super::lr::SGR_BOX1_NS),
            f(&super::lr::SGR_FINAL_NS),
        );
    }
    if flags::eob_stats() {
        use std::sync::atomic::Ordering::Relaxed;
        let b = &super::coeff::EOB_BLKS;
        let p = &super::coeff::EOB_PIX;
        eprintln!(
            "EOB blks[1]={} [<=4]={} [<=16]={} [<=64]={} [>64]={} | pix[1]={} [<=4]={} [<=16]={} [<=64]={} [>64]={}",
            b[0].swap(0, Relaxed),
            b[1].swap(0, Relaxed),
            b[2].swap(0, Relaxed),
            b[3].swap(0, Relaxed),
            b[4].swap(0, Relaxed),
            p[0].swap(0, Relaxed),
            p[1].swap(0, Relaxed),
            p[2].swap(0, Relaxed),
            p[3].swap(0, Relaxed),
            p[4].swap(0, Relaxed),
        );
        eprintln!(
            "EOB dc_only hit={} miss_type={}",
            super::coeff::DC_ONLY_HIT.swap(0, Relaxed),
            super::coeff::DC_ONLY_MISS.swap(0, Relaxed),
        );
    }
    if flags::mc_stats() {
        use std::sync::atomic::Ordering::Relaxed;
        let k = &super::inter::MC_PIX;
        eprintln!(
            "MC direct={}/{} cavg={} pix int={} sub={} avg={} dist={} wedge={} other={} obmc={} ii={}",
            super::inter::DIRECT_SUBPEL_BLOCKS.swap(0, Relaxed),
            super::inter::DIRECT_SUBPEL_PIXELS.swap(0, Relaxed),
            super::inter::COMP_AVG_PIX.swap(0, Relaxed),
            k[0].swap(0, Relaxed),
            k[1].swap(0, Relaxed),
            k[2].swap(0, Relaxed),
            k[3].swap(0, Relaxed),
            k[4].swap(0, Relaxed),
            k[5].swap(0, Relaxed),
            k[6].swap(0, Relaxed),
            k[7].swap(0, Relaxed),
        );
    }
    let bit_depth = frame.planes[0].bit_depth;
    let shift = u32::from(bit_depth.saturating_sub(8));
    let n = w * h * 3 / 2;
    // 池化：缓冲**每个字节都会被下面写满**（亮度 w*h + 色度 w*h/2），不必清零。
    let mut nv12 = if flags::no_nv12_pool() {
        vec![0u8; n]
    } else {
        super::pool::take_u8(n).unwrap_or_else(|| vec![0u8; n])
    };
    debug_assert_eq!(nv12.len(), n);
    let cw = w / 2;
    let ch = h / 2;
    let cbase = w * h;
    // 按**行切片**落盘，而不是逐像素 `Plane::get`。
    //
    // `get` 每次都要做一次 `debug_assert` + `y·stride + x` 的乘法，release 下虽然
    // 只剩乘法与边界，但 1920×1080 就是 207 万次 —— 采样里 `nv12_frame` 自己占 7%，
    // 全是这类"按坐标取址"的开销。改成"拿一行连续切片、内层只做移位与窄化"之后
    // 每次输出只有一次乘加，而且编译器能把内层向量化。
    if frame.planes[0].is_u8() {
        // **8bit（u8 存储）**：亮度整行 `copy_from_slice`，色度按抽样直接交错 ——
        // 不再有移位/窄化（`shift == 0`，与 u16 路径逐位等价）。
        let (luma_row, dst) = (frame.planes[0].stride, &mut nv12[..w * h]);
        let src = frame.planes[0].px8();
        for y in 0..h {
            dst[y * w..y * w + w].copy_from_slice(&src[y * luma_row..y * luma_row + w]);
        }
        if frame.num_planes == 1 {
            nv12[cbase..].fill(128);
        } else {
            let (u, v) = (&frame.planes[1], &frame.planes[2]);
            let (sub_x, sub_y) = (frame.sub_x, frame.sub_y);
            let (ustride, vstride) = (u.stride, v.stride);
            let (udata, vdata) = (u.px8(), v.px8());
            let dst = &mut nv12[cbase..];
            for y in 0..ch {
                let d = &mut dst[y * 2 * cw..(y + 1) * 2 * cw];
                if !sub_x && !sub_y {
                    // 4:4:4：2×2 平均（与 u16 路径同一式子：和 >> 2）
                    let r0 = y * 2;
                    let (ur0, ur1) = (&udata[r0 * ustride..], &udata[(r0 + 1) * ustride..]);
                    let (vr0, vr1) = (&vdata[r0 * vstride..], &vdata[(r0 + 1) * vstride..]);
                    for x in 0..cw {
                        let su = ur0[2 * x] as u32
                            + ur0[2 * x + 1] as u32
                            + ur1[2 * x] as u32
                            + ur1[2 * x + 1] as u32;
                        let sv = vr0[2 * x] as u32
                            + vr0[2 * x + 1] as u32
                            + vr1[2 * x] as u32
                            + vr1[2 * x + 1] as u32;
                        d[2 * x] = (su >> 2) as u8;
                        d[2 * x + 1] = (sv >> 2) as u8;
                    }
                } else if sub_x && !sub_y {
                    // 4:2:2：横向平均
                    let ur = &udata[y * ustride..y * ustride + 2 * cw];
                    let vr = &vdata[y * vstride..y * vstride + 2 * cw];
                    for x in 0..cw {
                        d[2 * x] = ((ur[2 * x] as u32 + ur[2 * x + 1] as u32) >> 1) as u8;
                        d[2 * x + 1] = ((vr[2 * x] as u32 + vr[2 * x + 1] as u32) >> 1) as u8;
                    }
                } else {
                    // 4:2:0：交错写入 UV
                    let ur = &udata[y * ustride..y * ustride + cw];
                    let vr = &vdata[y * vstride..y * vstride + cw];
                    for x in 0..cw {
                        d[2 * x] = ur[x];
                        d[2 * x + 1] = vr[x];
                    }
                }
            }
        }
        return Ok(VideoFrame { width: w as u32, height: h as u32, pts_us, nv12 });
    }
    {
        // ⚠️ 2026-10-24 第一版显式 AVX2 是**逐行**调用 target_feature 函数（无法内联）
        // ⇒ 每帧上千次真实调用，测出来 +1.9%。现在整个平面一次调用（行循环在内核里）。
        let (luma_row, dst) = (frame.planes[0].stride, &mut nv12[..w * h]);
        let src = frame.planes[0].px16();
        // ⚠️ 两次显式 AVX2（逐行 / 整平面一次调用）都更慢（+1.9% / +0.4%）——
        // LLVM 的自动向量化已经够好，这一段的瓶颈是**内存带宽**（读 4.2 MB、写 3.1 MB）。
        for y in 0..h {
            let s = &src[y * luma_row..y * luma_row + w];
            let d = &mut dst[y * w..y * w + w];
            for (dv, sv) in d.iter_mut().zip(s.iter()) {
                *dv = (*sv >> shift) as u8;
            }
        }
    }
    if frame.num_planes == 1 {
        for px in nv12[cbase..].iter_mut() {
            *px = 128;
        }
    } else {
        let (u, v) = (&frame.planes[1], &frame.planes[2]);
        let (sub_x, sub_y) = (frame.sub_x, frame.sub_y);
        let (ustride, vstride) = (u.stride, v.stride);
        let (udata, vdata) = (u.px16(), v.px16());
        let dst = &mut nv12[cbase..];
        for y in 0..ch {
            let d = &mut dst[y * 2 * cw..(y + 1) * 2 * cw];
            if !sub_x && !sub_y {
                // 4:4:4：2×2 平均
                let r0 = y * 2;
                let (ur0, ur1) = (&udata[r0 * ustride..], &udata[(r0 + 1) * ustride..]);
                let (vr0, vr1) = (&vdata[r0 * vstride..], &vdata[(r0 + 1) * vstride..]);
                for x in 0..cw {
                    let su = ur0[2 * x] as u32 + ur0[2 * x + 1] as u32
                        + ur1[2 * x] as u32
                        + ur1[2 * x + 1] as u32;
                    let sv = vr0[2 * x] as u32 + vr0[2 * x + 1] as u32
                        + vr1[2 * x] as u32
                        + vr1[2 * x + 1] as u32;
                    d[2 * x] = ((su >> 2) >> shift) as u8;
                    d[2 * x + 1] = ((sv >> 2) >> shift) as u8;
                }
            } else if sub_x && !sub_y {
                // 4:2:2：横向平均
                let ur = &udata[y * ustride..y * ustride + 2 * cw];
                let vr = &vdata[y * vstride..y * vstride + 2 * cw];
                for x in 0..cw {
                    let su = (ur[2 * x] as u32 + ur[2 * x + 1] as u32) >> 1;
                    let sv = (vr[2 * x] as u32 + vr[2 * x + 1] as u32) >> 1;
                    d[2 * x] = (su >> shift) as u8;
                    d[2 * x + 1] = (sv >> shift) as u8;
                }
            } else {
                // 4:2:0：交错写入 UV。每 8 对样本一次 `packusw` + `punpcklbw`，
                // 而不是 16 次单字节写（1080p 每帧约 52 万对）。
                let ur = &udata[y * ustride..y * ustride + cw];
                let vr = &vdata[y * vstride..y * vstride + cw];
                #[cfg(target_arch = "x86_64")]
                {
                    use std::arch::x86_64::*;
                    let mut x = 0usize;
                    while x + 8 <= cw {
                        // SAFETY: 读 `ur/vr[x..x+8]`（在 cw 内）、写 `d[2x..2x+16]`
                        // （在 2*cw 内）；样本移位后 ≤ 255（10/12bit 的 2/4 位）。
                        unsafe {
                            let uu = _mm_loadu_si128(ur.as_ptr().add(x) as *const __m128i);
                            let vv = _mm_loadu_si128(vr.as_ptr().add(x) as *const __m128i);
                            let uu = match shift {
                                0 => uu,
                                2 => _mm_srli_epi16::<2>(uu),
                                _ => _mm_srli_epi16::<4>(uu),
                            };
                            let vv = match shift {
                                0 => vv,
                                2 => _mm_srli_epi16::<2>(vv),
                                _ => _mm_srli_epi16::<4>(vv),
                            };
                            let ub = _mm_packus_epi16(uu, uu);
                            let vb = _mm_packus_epi16(vv, vv);
                            let uv = _mm_unpacklo_epi8(ub, vb);
                            _mm_storeu_si128(d.as_mut_ptr().add(2 * x) as *mut __m128i, uv);
                        }
                        x += 8;
                    }
                    for x in x..cw {
                        d[2 * x] = (ur[x] >> shift) as u8;
                        d[2 * x + 1] = (vr[x] >> shift) as u8;
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                for x in 0..cw {
                    d[2 * x] = (ur[x] >> shift) as u8;
                    d[2 * x + 1] = (vr[x] >> shift) as u8;
                }
            }
        }
    }
    Ok(VideoFrame { width: w as u32, height: h as u32, pts_us, nv12 })
}

/// 校验 `av1C`（AV1CodecConfigurationRecord，ISO/IEC 14496-15 §5.3.4 之后的定义）。
///
/// 字段布局：`[0]` = marker(1) + version(7)（必须是 `0x81`），`[1]` = seq_profile(3) |
/// seq_level_idx_0(5)，`[2]` = seq_tier_0(1) | high_bitdepth(1) | twelve_bit(1) |
/// monochrome(1) | chroma_subsampling_x(1) | chroma_subsampling_y(1) |
/// chroma_sample_position(2)，`[3]` = 保留(3) | initial_presentation_delay 相关(5)。
///
/// 为什么要在这里校验而不是直接交给解码器：`0x81` 这个 marker 是**唯一**能把
/// "av1C" 和"任意一段字节"区分开的东西（AV1 没有 Annex-B 起始码可查）。
pub fn validate_av1c(config: &[u8]) -> Result<()> {
    if config.len() < 4 {
        return Err(MediaError::Bitstream(format!(
            "av1C 太短（{} 字节，至少 4 字节）",
            config.len()
        )));
    }
    if config[0] != 0x81 {
        return Err(MediaError::Bitstream(format!(
            "av1C 的 marker/version 必须是 0x81（读到 0x{:02X}）",
            config[0]
        )));
    }
    Ok(())
}

/// 从 `av1C` 里读"这一轨的 profile / 位深 / 色度采样"（不解析码流，纯读配置盒）。
///
/// 用途：**在解封装之后、解码之前**就能判断"这台机器/这个后端能不能解"，
/// 以及给日志一个"这一轨是什么"的说法（与序列头里的值应当一致；不一致说明
/// 容器的配置盒与码流本身打架，那本身就是个值得报出来的问题）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Av1Config {
    pub seq_profile: u8,
    pub seq_level_idx: u8,
    pub seq_tier: u8,
    pub bit_depth: u8,
    pub mono_chrome: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub chroma_sample_position: u8,
}

impl Av1Config {
    /// 从 `av1C` 解析；字段宽度见 [`validate_av1c`] 的注释。
    pub fn parse(config: &[u8]) -> Result<Self> {
        validate_av1c(config)?;
        let seq_profile = config[1] >> 5;
        let seq_level_idx = config[1] & 0x1F;
        let seq_tier = config[2] >> 7;
        let high_bitdepth = (config[2] >> 6) & 1 == 1;
        let twelve_bit = (config[2] >> 5) & 1 == 1;
        let bit_depth = if twelve_bit {
            12
        } else if high_bitdepth {
            10
        } else {
            8
        };
        Ok(Self {
            seq_profile,
            seq_level_idx,
            seq_tier,
            bit_depth,
            mono_chrome: (config[2] >> 4) & 1 == 1,
            subsampling_x: (config[2] >> 3) & 1 == 1,
            subsampling_y: (config[2] >> 2) & 1 == 1,
            chroma_sample_position: config[2] & 0x03,
        })
    }
}

/// 一个 AV1 轨"能不能在这台机器上播"的判决（日志与错误文案用）。
///
/// 硬解能不能用由平台决定（[`crate::hardware_backend`]），软解的进度由本模块决定
/// —— 把两件事分开说，用户看到的才不是一句含糊的"不支持"。
pub fn decode_outlook(config: &[u8]) -> String {
    match Av1Config::parse(config) {
        Ok(c) => format!(
            "AV1 profile {} level {}{}、{}bit {}",
            c.seq_profile,
            c.seq_level_idx,
            if c.seq_tier == 1 { " (High tier)" } else { "" },
            c.bit_depth,
            if c.mono_chrome {
                "mono"
            } else if c.subsampling_x && c.subsampling_y {
                "4:2:0"
            } else if c.subsampling_x {
                "4:2:2"
            } else {
                "4:4:4"
            }
        ),
        Err(e) => format!("av1C 读不出来：{e}"),
    }
}

/// 这个编码是不是 AV1（一层薄包装，避免调用方到处写 `matches!`）。
pub fn is_av1(codec: &CodecId) -> bool {
    matches!(codec, CodecId::Av1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_av1c(profile: u8, level: u8, high: bool, twelve: bool, mono: bool) -> Vec<u8> {
        let b1 = (profile << 5) | (level & 0x1F);
        let b2 = ((high as u8) << 6) | ((twelve as u8) << 5) | ((mono as u8) << 4) | 0b1100;
        vec![0x81, b1, b2, 0x00]
    }

    /// 造一个"只有分隔符 + 序列头"的样本（没有像素要重建）。
    fn sample_with_sequence_header() -> Vec<u8> {
        let mut w = crate::video::testbits::BitWriter::default();
        crate::video::testbits::write_reduced_still_picture_1080p(&mut w);
        let payload = w.finish().to_vec();
        let mut sample = vec![(2u8 << 3) | 0b0000_0010, 0x00]; // 时间分隔符（长度 0）
        sample.push((1u8 << 3) | 0b0000_0010); // 序列头 OBU
        sample.push(payload.len() as u8);
        sample.extend_from_slice(&payload);
        sample
    }

    /// 造一个**合法且完整**的关键帧 OBU（`OBU_FRAME`，含参数段与 `byte_alignment`）。
    ///
    /// 每一位都按 `seq` 的条件写（`seq_force_screen_content_tools` 是不是 SELECT、
    /// 有没有 frame_id、有没有超分…都决定这一位读不读）。合成序列头与真实 `av1C`
    /// 里的序列头都能喂进来 —— 这也是"序列头 → 帧头"两层之间的一致性检查。
    fn key_frame_obu(seq: &crate::video::sequence_header::SequenceHeader) -> Vec<u8> {
        use crate::video::sequence_header::{SELECT_INTEGER_MV, SELECT_SCREEN_CONTENT_TOOLS};
        use crate::video::testbits::{write_min_params, BitWriter};
        let mut w = BitWriter::default();
        if !seq.reduced_still_picture_header {
            w.bit(false); // show_existing_frame
            w.f(0, 2); // frame_type = KEY_FRAME
            w.bit(true); // show_frame
            // showable_frame / error_resilient_mode 由推导得出，不读位
        }
        w.bit(false); // disable_cdf_update
        // allow_screen_content_tools：只有 seq_force == SELECT 才是帧头里的位
        let sct = if seq.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS {
            w.bit(false);
            false
        } else {
            seq.seq_force_screen_content_tools != 0
        };
        // force_integer_mv：帧内帧最终强制为 1，但这一位只在"允许屏幕内容工具
        // 且 seq_force_integer_mv == SELECT"时才读
        if sct && seq.seq_force_integer_mv == SELECT_INTEGER_MV {
            w.bit(false);
        }
        if seq.frame_id_numbers_present {
            w.f(0, seq.frame_id_length as u32); // current_frame_id
        }
        if !seq.reduced_still_picture_header {
            w.bit(false); // frame_size_override_flag = 0 → 用序列头的最大尺寸
        }
        w.f(0, seq.order_hint_bits as u32); // order_hint
        // primary_ref_frame：帧内 → PRIMARY_REF_NONE，不读位
        if seq.decoder_model_info_present {
            w.bit(false); // buffer_removal_times_present_flag
        }
        // frame_size()：frame_size_override = 0 → 尺寸不读；但超分位要按序列头来
        if seq.enable_superres {
            w.bit(false); // use_superres
        }
        w.bit(false); // render_and_frame_size_different
        // allow_intrabc：允许屏幕内容工具且没有超分时才读
        if sct {
            w.bit(false);
        }
        if !seq.reduced_still_picture_header {
            w.bit(false); // disable_frame_end_update_cdf
        }
        write_min_params(&mut w, seq, true);
        w.byte_align();
        let mut payload = w.finish().to_vec();
        // 单瓦片流：帧头（+ 对齐）之后就是那唯一瓦片的**算术流**。
        // 这里补一段确定性字节当瓦片数据 —— 不是真算术流，解码会在数据层
        // 如实失败（测试就是盯这一点：报"数据不对"，而不是"功能没做"）。
        payload.extend((0..64u32).map(|i| ((i * 37 + 11) % 251) as u8));
        assert!(payload.len() < 128, "单字节 OBU 长度字段装得下：{}", payload.len());
        let mut obu = vec![(6u8 << 3) | 0b0000_0010]; // OBU_FRAME
        obu.push(payload.len() as u8);
        obu.extend_from_slice(&payload);
        obu
    }

    #[test]
    fn validates_av1c_marker() {
        assert!(validate_av1c(&[0x81, 0, 0x0C, 0]).is_ok());
        assert!(validate_av1c(&[0x00, 0, 0, 0]).is_err(), "marker 必须是 0x81");
        assert!(validate_av1c(&[0x81, 0]).is_err(), "太短");
    }

    #[test]
    fn parses_av1c_fields() {
        let c = Av1Config::parse(&make_av1c(0, 8, false, false, false)).unwrap();
        assert_eq!(c.seq_profile, 0);
        assert_eq!(c.seq_level_idx, 8);
        assert_eq!(c.bit_depth, 8);
        assert!(!c.mono_chrome);

        let c = Av1Config::parse(&make_av1c(2, 5, true, false, true)).unwrap();
        assert_eq!(c.seq_profile, 2);
        assert_eq!(c.bit_depth, 10);
        assert!(c.mono_chrome, "mono 位读出来是对的");

        let c = Av1Config::parse(&make_av1c(2, 5, true, true, false)).unwrap();
        assert_eq!(c.bit_depth, 12, "twelve_bit 优先于 high_bitdepth");
    }

    #[test]
    fn inspects_a_header_only_sample() {
        let sample = sample_with_sequence_header();
        let info = inspect_sample(&sample).unwrap();
        assert_eq!(info.obu_count, 2);
        assert!(!info.has_bitstream, "只有分隔符与序列头");
        let sh = info.sequence_header.expect("应当解析出序列头");
        assert_eq!(sh.max_frame_width, 1920);
        assert_eq!(sh.max_frame_height, 1080);
    }

    #[test]
    fn header_only_sample_is_not_an_error() {
        let cfg = make_av1c(0, 8, false, false, false);
        let mut dec = Av1Decoder::open(&cfg).unwrap();
        let frames = dec.decode(&sample_with_sequence_header(), 0).unwrap();
        assert!(frames.is_empty(), "还没有像素可解，不是失败");
        assert!(dec.sequence_header().is_some(), "序列头要留下来（后面每一层都要用）");
    }

    #[test]
    fn a_synthetic_frame_never_reports_unimplemented() {
        // 合成帧的瓦片是"凑数的字节"，不是真算术流。**自研软解现在会真的往下解**
        // （算术解码器碰到随机数据不会报错，只会产出乱码符号），所以这条合成帧
        // 可能"解出帧"也可能"在数据层失败" —— 这条测试盯的不是成败，而是：
        // ① 绝不能出现"功能没实现"那种话（那是接线没做完的信号）；
        // ② 非压缩头本身必须读通（这才是"帧头层"的验收）；
        // ③ 万一解出帧，尺寸与 NV12 缓冲长度必须是对的。
        let cfg = make_av1c(0, 8, false, false, false);
        let mut dec = Av1Decoder::open(&cfg).unwrap();
        dec.decode(&sample_with_sequence_header(), 0).unwrap();
        let seq = dec.sequence_header().expect("序列头已解出").clone();
        let mut frame_sample = sample_with_sequence_header();
        frame_sample.extend_from_slice(&key_frame_obu(&seq));
        match dec.decode(&frame_sample, 0) {
            Ok(frames) => {
                for f in &frames {
                    assert_eq!((f.width, f.height), (1920, 1080), "解出帧也要是对应尺寸");
                    assert_eq!(
                        f.nv12.len(),
                        VideoFrame::expected_len(1920, 1080),
                        "NV12 缓冲长度必须匹配"
                    );
                }
            }
            Err(e) => {
                let text = format!("{e}");
                assert!(
                    !text.contains("还没"),
                    "驱动层那句'还没接线'不该再出现（凑数数据要么解完、要么死在数据层）：{text}"
                );
            }
        }
        // 非压缩头本身必须读通（这是"帧头层"的验收）
        let fh = dec.last_frame().expect("帧头应当解析成功");
        assert_eq!(fh.frame_type.name(), "key");
        assert_eq!(fh.upscaled_width, 1920);
        assert_eq!(fh.params.tile_info.cols, 1, "这条合成头是单瓦片");
        assert_eq!(fh.bits_consumed % 8, 0, "末尾必须字节对齐");
        assert_eq!(dec.last_tiles(), Some((1, 64)), "一个瓦片、64 字节");
    }

    /// 真实素材的 `av1C`（B 站那份 ASMR）：记录 4 字节 + configOBUs 里的序列头 OBU。
    ///
    /// 为什么把它钉进来：**seek 到流中间时，那个采样里没有序列头**，
    /// 只能靠 `av1C` 里的 configOBUs 兜底。顺带这也是"真实序列头"的第二份回归样本。
    const REAL_AV1C_WITH_SEQ: [u8; 20] = [
        0x81, 0x08, 0x0C, 0x00, // record：profile 0 / level 8 / 8bit 4:2:0
        0x0A, 0x0E, // OBU 头（type 1 = 序列头）+ 长度 14
        0x00, 0x00, 0x00, 0x42, 0xab, 0xbf, 0xc3, 0x77, 0xee, 0x66, 0x40, 0x40, 0x40, 0x41,
    ];

    #[test]
    fn takes_the_initial_sequence_header_from_av1c_config_obus() {
        let dec = Av1Decoder::open(&REAL_AV1C_WITH_SEQ).unwrap();
        let sh = dec
            .sequence_header()
            .expect("av1C 的 configOBUs 里带序列头，应当被当成初始序列头");
        assert_eq!(sh.seq_profile, 0);
        assert_eq!(sh.max_frame_width, 1920);
        assert_eq!(sh.max_frame_height, 1080);
        assert_eq!(sh.pixel_format_text(), "8bit 4:2:0");
        // 记录里的 profile/位深必须与码流里的序列头一致
        let record = Av1Config::parse(&REAL_AV1C_WITH_SEQ).unwrap();
        assert_eq!(record.seq_profile, sh.seq_profile);
        assert_eq!(record.bit_depth, sh.color.bit_depth);
    }

    #[test]
    fn a_frame_without_sequence_header_uses_the_one_from_config() {
        // 只有"帧 OBU"的采样（相当于 seek 到流中间）：靠 av1C 里的序列头也能读帧头。
        // 瓦片是凑数数据（可能解出乱码帧、也可能在数据层失败），但**绝不能**报
        // "没有序列头" —— av1C 里那份就该被用上，这是 seek 可用性的底线。
        let mut dec = Av1Decoder::open(&REAL_AV1C_WITH_SEQ).unwrap();
        let seq = dec.sequence_header().expect("av1C 的 configOBUs 里带序列头").clone();
        let sample = key_frame_obu(&seq);
        if let Err(e) = dec.decode(&sample, 0) {
            let text = format!("{e}");
            assert!(
                !text.contains("序列头"),
                "不该报'没有序列头'（av1C 里那份就该被用上）：{text}"
            );
        }
        assert_eq!(dec.last_frame().map(|f| f.frame_type.name()), Some("key"));
        assert_eq!(dec.last_frame().map(|f| f.upscaled_width), Some(1920));
        let fh = dec.last_frame().unwrap();
        assert_eq!(fh.params.tile_info.cols, 1, "这条合成头是单瓦片");
        assert_eq!(fh.bits_consumed % 8, 0, "末尾必须字节对齐");
    }

    #[test]
    fn empty_sample_is_rejected() {
        assert!(inspect_sample(&[]).is_err());
        let cfg = make_av1c(0, 8, false, false, false);
        let mut dec = Av1Decoder::open(&cfg).unwrap();
        assert!(dec.decode(&[], 0).is_err(), "空采样是数据错误，不能当 EOF");
    }

    #[test]
    fn outlook_text_is_readable() {
        let text = decode_outlook(&make_av1c(0, 8, false, false, false));
        assert!(text.contains("profile 0"), "{text}");
        assert!(text.contains("8bit 4:2:0"), "{text}");
        assert!(decode_outlook(&[0, 0, 0, 0]).contains("读不出来"));
    }
}
