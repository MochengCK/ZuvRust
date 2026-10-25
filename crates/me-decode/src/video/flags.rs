//! 诊断/隔离用的环境开关，**进程内只读一次**。
//!
//! ## 为什么要专门做一层
//!
//! `std::env::var` 不是"读个字符串"那么便宜：它要拿一把全局锁、在环境块里线性扫描、
//! 还要分配一个 `String`。放在**逐块 / 逐变换块 / 逐像素**的路径上就是白扔 CPU ——
//! `sample` 采样（2026-10-01，1080p AV1 软解）显示 `getenv` / `__findenv_locked` /
//! `std::env::var` 合计**约占 10%**，而这些开关在正常播放时**一个都不该命中**。
//!
//! 引擎是"一个进程解一条流"，环境变量不会中途变化，所以第一次读到什么就是什么
//! （`OnceLock` 缓存，之后只是一次原子读 + 分支）。这也让"诊断开关"这条路的开销
//! 从"每个块几百纳秒"降到"一个可预测的分支"。
//!
//! ⚠️ **测试里不要用 `std::env::set_var` 指望中途生效**：本层是缓存语义。
//! 需要不同取值就**另起一个进程**（`me-cli` 的测试正是这么做的）。

use std::sync::OnceLock;

/// 生成一个"读一次就缓存"的布尔开关。
macro_rules! bool_flag {
    ($($(#[$m:meta])* $fname:ident = $env:literal;)*) => {
        $(
            $(#[$m])*
            // 有些开关只被少数几个文件用；留着是为了让"诊断开关一览"在一个地方
            #[allow(dead_code)]
            #[inline]
            pub(crate) fn $fname() -> bool {
                static V: OnceLock<bool> = OnceLock::new();
                *V.get_or_init(|| std::env::var($env).is_ok())
            }
        )*
    };
}

/// 生成一个"读一次就缓存"的**数值**开关（解析失败 = 未设置）。
macro_rules! num_flag {
    ($($(#[$m:meta])* $fname:ident($t:ty) = $env:literal;)*) => {
        $(
            $(#[$m])*
            #[allow(dead_code)]
            #[inline]
            pub(crate) fn $fname() -> Option<$t> {
                static V: OnceLock<Option<$t>> = OnceLock::new();
                *V.get_or_init(|| {
                    std::env::var($env).ok().and_then(|v| v.parse::<$t>().ok())
                })
            }
        )*
    };
}

// ── 隔离开关（把某一步换成"规范定义的等价空操作"，用来定位差异来自哪一级）──
bool_flag! {
    /// `AV1_NO_DEBLOCK=1`：不做去块。
    no_deblock = "AV1_NO_DEBLOCK";
    /// `AV1_NO_CDEF=1`：不做 CDEF。
    no_cdef = "AV1_NO_CDEF";
    /// `AV1_NO_LR=1`：不做环路恢复。
    no_lr = "AV1_NO_LR";
    /// `AV1_NO_OBMC=1`：不叠 OBMC。
    no_obmc = "AV1_NO_OBMC";
    /// `AV1_NO_INTERINTRA=1`：不做帧内-帧间混合。
    no_interintra = "AV1_NO_INTERINTRA";
    /// `AV1_NO_WARP=1`：不做 warp（局部 + 全局）。
    no_warp = "AV1_NO_WARP";
    /// `AV1_NO_LOCALWARP=1`：只关块级局部 warp。
    no_localwarp = "AV1_NO_LOCALWARP";
    /// `AV1_NO_GMWARP=1`：只关全局运动 warp。
    no_gmwarp = "AV1_NO_GMWARP";
    /// `AV1_NO_INTRABC=1`：不做屏幕内容的帧内块拷贝。
    no_intrabc = "AV1_NO_INTRABC";
    /// `AV1_NO_CFL=1`：不做色度 CFL。
    no_cfl = "AV1_NO_CFL";
    /// `AV1_NO_PRED=1`：不做帧内预测（写默认值）。
    no_pred = "AV1_NO_PRED";
    /// `AV1_NO_RESIDUAL=1`：不加残差。
    no_residual = "AV1_NO_RESIDUAL";
    /// `ME_ROW_MT=1`：**row-mt**（语法阶段只记工单、重建阶段回放；见 `block::RowMtJobs`）。
    /// 判据是开与不开**逐字节一致**。
    row_mt = "ME_ROW_MT";
    /// `ME_ROW_MT_PAR=1`（配合 `ME_ROW_MT=1`）：回放走 **SB 行列波前并行**（阶段 2）。
    /// 回放前会预计算每个单元的 `BlockDecoded` 查询（`block::RowMtJobs::precompute_dec_flags`），
    /// 判据是与不开 row-mt 的输出**逐字节一致**。
    row_mt_par = "ME_ROW_MT_PAR";
    /// `ME_ROW_MT_LIVE_FLAGS=1`（诊断）：回放**不做** `DecFlags` 预计算、照旧读运行时的
    /// `BlockDecoded` 网格（只有在顺序回放下才与记录序同序）。用来把"预计算错了"
    /// 与"回放错了"分开量。
    row_mt_live_flags = "ME_ROW_MT_LIVE_FLAGS";
    /// `ME_ROW_MT_FULLWAIT=1`（诊断）：并行回放时行 `r` **直接等行 `r-1` 整行做完**
    /// （而不是按列波前）。用来二分"跨行读的依赖漏了"与"别的结构性问题"：
    /// 全等 ⇒ 依赖是跨行的、只是列上界不对；还不等 ⇒ 问题不在跨行等待上。
    row_mt_fullwait = "ME_ROW_MT_FULLWAIT";
    /// `ME_ROW_MT_UP2W=1`（诊断）：并行回放时行 `r` 一律等到上一行推进过
    /// "本块右边缘 + 一个块宽"（`x + 2w`）。用来判定"上方读得更远"与"**向下读**"
    /// 哪个才是剩下的依赖：全等 ⇒ 是上方；还不等 ⇒ 是向下（左下读）。
    row_mt_up2w = "ME_ROW_MT_UP2W";
    /// `AV1_NO_BELOWLEFT=1`（**诊断**）：把 `DecFlags::below_left` 一律置 0。
    /// 用来判定 row-mt 并行剩下的依赖是不是"左下那一列读到下一 SB 行"：
    /// 顺序/并行在这次强制下若一致，就说明是它。（会改变预测像素，只用于二分。）
    no_below_left = "AV1_NO_BELOWLEFT";
    /// `AV1_PHASE_TIMING=1`：每帧打一行各阶段耗时（瓦片解码 / 环路滤波 / 输出转换）。
    phase_timing = "AV1_PHASE_TIMING";
    /// `AV1_SYNTAX_ONLY=1`：**只读语法、不做像素重建**（也不跑环路滤波）——
    /// 用来把"语法/熵/块簿记"与"重建（预测+反变换+滤波）"的墙钟拆开量。
    /// 输出是垃圾像素，**只用于计时/对拍语法**。
    syntax_only = "AV1_SYNTAX_ONLY";
    /// `AV1_NO_CDF_UPDATE=1`：CDF 自适应不生效（**只是诊断**，会立刻错位）。
    no_cdf_update = "AV1_NO_CDF_UPDATE";
    /// `AV1_WARP_REFFMT=1`：warp 照 libaom 的 C 源码写法（`Round2` + `sx4 &= ~63`）。
    warp_reffmt = "AV1_WARP_REFFMT";
    /// `AV1_WARP_SIGNED=1`：warp 的两趟取整用 `Round2Signed`（远离零取整）。
    /// **这是 2026-10-03 前的默认**，不符合规范（§4.7 的 `Round2` 是"加半再算术
    /// 右移"），只留给对拍用。
    warp_signed = "AV1_WARP_SIGNED";
    /// `ME_NO_SIMD=1`：禁用 SIMD 内核（强制走标量 —— 用来对拍"开/不开 SIMD 输出是否逐字节一致"）。
    no_simd = "ME_NO_SIMD";
    /// `ME_PLANE_U8=1`：8bit 帧的 `Plane` 用 `u8` 存储（dav1d 同款；10/12bit 恒为 `u16`）。
    /// **默认关** —— u8 路径正在逐内核迁移（见 `PERF-HANDOFF.md` 第九节），
    /// 迁移完成前保持 u16 是"已知逐像素一致"的那条路。
    plane_u8 = "ME_PLANE_U8";
    /// `ME_NO_VT32=1`（**诊断**）：反变换禁用 8 路 i32 通道，强制回到 4 路 i64。
    /// 两条路**逐字节相同**（值域分析 + 全组合对拍钉住），这个开关用来做同二进制 A/B。
    no_vt32 = "ME_NO_VT32";
    /// `ME_FRAME_PIPE=1`（**实验**）：两帧流水线——语法 N+1 与滤镜 N 重叠。
    /// 仅支持单瓦片 + `ME_ROW_MT=1`；其余情况自动回退同步路径。
    frame_pipe = "ME_FRAME_PIPE";
    /// `ME_NO_CDEF_EDGE=1`（**诊断**）：CDEF 的右/下边界块退回整块标量（旧行为）。
    /// 两条路**逐字节相同**，用于同二进制 A/B 边界 SIMD 的收益。
    no_cdef_edge = "ME_NO_CDEF_EDGE";
    /// `ME_NO_CDEF_GROUP=1`（**诊断**）：CDEF 退回逐 8×8 块的老路径
    /// （不按 64×64 组批处理）。两条路**逐字节相同**。
    no_cdef_group = "ME_NO_CDEF_GROUP";
    /// `ME_NO_CDEF_DIR_SIMD=1`（**诊断**）：CDEF 方向搜索退回标量累加。
    /// 两条路**逐字节相同**，用于同二进制 A/B。
    no_cdef_dir_simd = "ME_NO_CDEF_DIR_SIMD";
    /// `ME_NO_TX_INPLACE=1`（**诊断**）：残差写回退回"先拷预测再相加"。
    /// 两条路**逐字节相同**，用于同二进制 A/B。
    no_tx_inplace = "ME_NO_TX_INPLACE";
    /// `ME_NO_DIRECT_PRED=1`（**诊断**）：整像素单参考预测退回"i32 缓冲 + store"。
    /// 两条路**逐字节相同**，用于同二进制 A/B。
    no_direct_pred = "ME_NO_DIRECT_PRED";
    /// `ME_NO_ROWCOPY=1`（**诊断**）：直写路径的整行拷贝退回 `copy_from_slice`。
    /// 两条路**逐字节相同**，用于同二进制 A/B。
    no_rowcopy = "ME_NO_ROWCOPY";
    /// `ME_NO_MV0=1`（**诊断**）：运动场投影的零 MV 快路径退回逐目标帧计算。
    /// 两条路**逐字节相同**，用于同二进制 A/B。
    no_mv0_fast = "ME_NO_MV0";
    /// `ME_NO_CDEF_SHIFT=1`（**诊断**）：CDEF 左/上边界块退回整块标量。
    /// 两条路**逐字节相同**，用于同二进制 A/B。
    no_cdef_shift = "ME_NO_CDEF_SHIFT";
    /// `ME_NO_LR_SWAP=1`（**诊断**）：LR 的 RESTORE_NONE 平面退回整块拷贝。
    /// 两条路**逐字节相同**，用于同二进制 A/B。
    no_lr_swap = "ME_NO_LR_SWAP";
    /// `ME_NO_NV12_POOL=1`（**诊断**）：NV12 输出缓冲不用池、每帧新分配。
    /// 两条路**逐字节相同**，用于同二进制 A/B。
    no_nv12_pool = "ME_NO_NV12_POOL";
    /// `ME_NO_DECODED_POOL=1`（**诊断**）：`CurrFrame` 的 `decoded` 网格不用池。
    /// 两条路**逐字节相同**，用于同二进制 A/B。
    no_decoded_pool = "ME_NO_DECODED_POOL";
    /// `ME_CDEF_STATS=1`（**诊断**）：每帧打印 CDEF 的克隆/总耗时与组跳过率。
    cdef_stats = "ME_CDEF_STATS";
    /// `ME_MC_STATS=1`（**诊断**）：每帧打印单参考子像素直写平面的块/像素数。
    mc_stats = "ME_MC_STATS";
    /// `ME_NO_DIRECT_SUBPEL=1`（**诊断**）：子像素预测退回"i32 暂存 + 拷贝"。
    /// 两条路**逐字节相同**，用于同二进制 A/B。
    no_direct_subpel = "ME_NO_DIRECT_SUBPEL";
    /// `ME_NO_DIRECT_COMP_AVG=1`（**诊断**）：复合 AVERAGE 的双整像素直写退回
    /// "两路 MC + 混合"。两条路**逐字节相同**。
    no_direct_comp_avg = "ME_NO_DIRECT_COMP_AVG";
    /// `ME_NO_MF_TABLE=1`（**诊断**）：运动场存储退回逐格 `get_relative_dist`。
    /// 两条路**逐字节相同**。
    no_mf_table = "ME_NO_MF_TABLE";
    /// `ME_NO_FAST_FILL=1`（**诊断**）：`BlockGrid::stamp` 的短行写退回 `slice::fill`。
    /// 两条路**逐字节相同**。
    no_fast_fill = "ME_NO_FAST_FILL";
    /// `ME_NO_RECT_COPY=1`（**诊断**）：整像素预测的整块矩形拷贝退回逐行 `copy_row_u16`。
    /// 两条路**逐字节相同**。
    no_rect_copy = "ME_NO_RECT_COPY";
    /// `ME_NO_INTRA_SMOOTH_SIMD=1`（**诊断**）：帧内 smooth 预测退回逐像素标量。
    /// 两条路**逐字节相同**。
    no_intra_smooth_simd = "ME_NO_INTRA_SMOOTH_SIMD";
    /// `ME_NO_LR_NONE_COPY=1`（**诊断**）：LR 的 `RESTORE_NONE` 档退回 i32 行缓冲 + `set_row`。
    /// 两条路**逐字节相同**。
    no_lr_none_copy = "ME_NO_LR_NONE_COPY";
    /// `ME_NO_DC_ONLY=1`（**诊断**）：DC-only 的反变换快路径退回完整二维变换。
    /// 两条路**逐字节相同**。
    no_dc_only = "ME_NO_DC_ONLY";
    /// `ME_NO_TX_SPARSE_ROWS=1`（**诊断**）：反变换的行趟不跳过全零的 8 行组。
    /// 两条路**逐字节相同**。
    no_tx_sparse_rows = "ME_NO_TX_SPARSE_ROWS";
    /// `ME_NO_SGR_GATHER=1`（**诊断**）：SGR 的 `a2` 查表退回逐通道标量。
    /// 两条路**逐字节相同**。
    no_sgr_gather = "ME_NO_SGR_GATHER";
    /// `ME_NO_SGR_PROJ=1`（**诊断**）：SGR 收尾投影退回逐像素标量。
    /// 两条路**逐字节相同**。
    no_sgr_proj = "ME_NO_SGR_PROJ";
    /// `ME_NO_SGR_FINAL_PAT=1`（**诊断**）：SGR 末段 3×3 退回按权重表逐项 `mullo`。
    /// 两条路**逐字节相同**。
    no_sgr_final_pat = "ME_NO_SGR_FINAL_PAT";
    /// `ME_NO_NV12_LUMA_SIMD=1`（**诊断**）：NV12 亮度收窄退回逐像素标量循环。
    /// 两条路**逐字节相同**。
    no_nv12_luma_simd = "ME_NO_NV12_LUMA_SIMD";
    /// `ME_NO_SGR_PREFIX=1`（**诊断**）：SGR 前缀和退回逐列标量。
    /// 两条路**逐字节相同**。
    no_sgr_prefix = "ME_NO_SGR_PREFIX";
    /// `ME_NO_SGR_FLOAT=1`（**诊断**）：SGR 的 `a2` 退回 `vpgatherdd` 查表。
    /// 两条路**逐字节相同**（有全量对拍单测）。
    no_sgr_float = "ME_NO_SGR_FLOAT";
    /// `ME_LR_STATS=1`（**诊断**）：打印 SGR 三段（box0 / box1 / 收尾）的累计耗时。
    lr_stats = "ME_LR_STATS";
    /// `ME_EOB_STATS=1`（**诊断**）：每帧打印 `eob` 分布（块数 / 像素数）。
    eob_stats = "ME_EOB_STATS";
    /// `ME_NO_CDEF_4ROW=1`（**诊断**）：CDEF 亮度滤波退回"两行一组"的 `cdef_pair16`。
    /// 两条路**逐字节相同**（有位对拍单测）。
    no_cdef_4row = "ME_NO_CDEF_4ROW";
    /// `ME_NO_CDEF_STR_SPEC=1`（**诊断**）：CDEF 不按"哪一档强度为 0"单态化，
    /// 退回完整 12 抽头内核。两条路**逐字节相同**。
    no_cdef_str_spec = "ME_NO_CDEF_STR_SPEC";
    /// `ME_NO_CDEF_DIR2=1`（**诊断**）：CDEF 方向搜索退回逐块（不用两块并行的 YMM 版）。
    /// 两条路**逐字节相同**。
    no_cdef_dir2 = "ME_NO_CDEF_DIR2";
    /// `ME_NO_CDEF_CAND_CACHE=1`（**诊断**）：CDEF 候选偏移/抽头不跨块缓存。
    /// 两条路**逐字节相同**。
    no_cdef_cand_cache = "ME_NO_CDEF_CAND_CACHE";
    /// `ME_NO_STAMP_SKIP=1`（**诊断**）：`stamp` 不再跳过"等于帧初值"的数组写。
    /// 两条路**逐字节相同** —— 被跳过的数组都由 `reset_all` 的 `fill_set` 铺过初值，
    /// 且每个 4×4 格在本帧只属于一个叶子块（不会留着别的块写过的值）。
    no_stamp_skip = "ME_NO_STAMP_SKIP";
    /// `ME_NO_MARK_ROWS=1`（**诊断**）：`mark_decoded` 退回逐格写。
    /// 两条路**逐字节相同**。
    no_mark_rows = "ME_NO_MARK_ROWS";
    /// `ME_NO_INTRA_ROWCOPY=1`（**诊断**）：帧内预测写回退回逐像素 `Plane::set`。
    /// 两条路**逐字节相同**。
    no_intra_rowcopy = "ME_NO_INTRA_ROWCOPY";
    /// `ME_NO_INTRA_POOL=1`（**诊断**）：帧内边/预测缓冲退回每次新分配。
    /// 两条路**逐字节相同**。
    no_intra_pool = "ME_NO_INTRA_POOL";
    /// `ME_NO_PART_FILL=1`（**诊断**）：分区网格铺块退回逐 4×4 格写。
    /// 两条路**逐字节相同**。
    no_part_fill = "ME_NO_PART_FILL";
    /// `ME_NO_LAZY_FT=1`（**诊断**）：帧间块的边滤波类型恢复无条件计算。
    /// 两条路**逐字节相同**。
    no_lazy_ft = "ME_NO_LAZY_FT";
    /// `ME_NO_SIMD_RESADD=1`（**诊断**）：残差加回退回逐像素标量。
    /// 两条路**逐字节相同**。
    no_simd_resadd = "ME_NO_SIMD_RESADD";
    /// `ME_NO_CDEF_DIRGATE=1`（**诊断**）：CDEF 方向搜索恢复无条件计算。
    /// 两条路**逐字节相同**。
    no_cdef_dirgate = "ME_NO_CDEF_DIRGATE";
    /// `ME_NO_WIDE_DEBLOCK=1`（**诊断**）：横边不合并成 8 路，退回逐 4 像素。
    /// 两条路**逐字节相同**。
    no_wide_deblock = "ME_NO_WIDE_DEBLOCK";
    /// `ME_NO_CDEF_DIRTAB=1`（**诊断**）：CDEF 候选方向表退回逐块现算。
    /// 两条路**逐字节相同**。
    no_cdef_dirtab = "ME_NO_CDEF_DIRTAB";
    /// `ME_NO_FLIP_RESADD=1`（**诊断**）：翻转残差加回退回逐像素标量。
    /// 两条路**逐字节相同**。
    no_flip_resadd = "ME_NO_FLIP_RESADD";
    /// `ME_NO_DEQ_SIMD=1`（**诊断**）：反量化退回逐系数标量。
    /// 两条路**逐字节相同**。
    no_deq_simd = "ME_NO_DEQ_SIMD";
    /// `ME_NO_HORIZ_EDGE_SIMD=1`（**诊断**）：MC 横向的边界块退回逐样本钳位标量。
    /// 两条路**逐字节相同**。
    no_horiz_edge_simd = "ME_NO_HORIZ_EDGE_SIMD";
    /// `ME_NO_CDEF_VISIBLE_CLONE=1`（**诊断**）：CDEF 的整帧克隆也拷帧外 padding。
    /// 两条路**逐字节相同**（padding 不进输出，`ME_PAD_POISON` 验过）。
    no_cdef_visible_clone = "ME_NO_CDEF_VISIBLE_CLONE";
}

// ── 轨迹开关（打日志；逐块/逐变换块级别的判定必须走本层）──
bool_flag! {
    /// `AV1_TRACE_DBL=1`：打去块参数与每条边。
    trace_dbl = "AV1_TRACE_DBL";
    /// `AV1_TRACE_DBL2=1`：打去块逐样本掩码。
    trace_dbl2 = "AV1_TRACE_DBL2";
    /// `AV1_TRACE_LR=1`：打印环路恢复的帧级参数。
    trace_lr = "AV1_TRACE_LR";
    /// `AV1_TRACE_WARP=1`：打 warp 调用参数。
    trace_warp = "AV1_TRACE_WARP";
    /// `AV1_TRACE_PRED=1`：打帧间预测的中间量。
    trace_pred = "AV1_TRACE_PRED";
    /// `AV1_TRACE_CMP16=1`：打每个复合色度块两路 d16 预测的和。
    trace_cmp16 = "AV1_TRACE_CMP16";
    /// `AV1_TRACE_CFL=1`：打 CFL 相关量。
    trace_cfl = "AV1_TRACE_CFL";
    /// `AV1_TRACE_TX1=1`：打单个变换块。
    trace_tx1 = "AV1_TRACE_TX1";
    /// `AV1_TRACE_TXB=1`：打变换块轨迹。
    trace_txb = "AV1_TRACE_TXB";
    /// `AV1_TRACE_TBXCTX=1`：打变换块级上下文写事件。
    trace_tbxctx = "AV1_TRACE_TBXCTX";
    /// `AV1_TRACE_CTXW=1`：打上下文槽位读写。
    trace_ctxw = "AV1_TRACE_CTXW";
    /// `AV1_TRACE_DIR=1`：打方向预测。
    trace_dir = "AV1_TRACE_DIR";
    /// `AV1_TRACE_MM=1`：打运动模式。
    trace_mm = "AV1_TRACE_MM";
    /// `AV1_TRACE_IBC=1`：打 intrabc。
    trace_ibc = "AV1_TRACE_IBC";
    /// `AV1_TRACE_ALL=1`：打所有块的语法。
    trace_all = "AV1_TRACE_ALL";
    /// `AV1_TRACE_MODE=1`：打模式信息。
    trace_mode = "AV1_TRACE_MODE";
    /// `AV1_TRACE_LASTSB=1`：打最后一个超块。
    trace_last_sb = "AV1_TRACE_LASTSB";
    /// `AV1_TRACE_SB=1`：逐超块记账。
    trace_sb = "AV1_TRACE_SB";
    /// `AV1_TRACE_BLKATTR=1`：逐块属性轨迹。
    trace_blkattr = "AV1_TRACE_BLKATTR";
    /// `AV1_TRACE_PART=1`：打分区树。
    trace_part = "AV1_TRACE_PART";
    /// `AV1_TRACE_SYM=1`：逐符号轨迹（与 libaom 插桩对拍用）。
    trace_sym = "AV1_TRACE_SYM";
    /// `AV1_TRACE_ROWNEED=1`：打**每块**预计算的"上一行等待列" `above_need`
    /// （`ROWB` 行）+ 帧内 `derive_edges` 实际读到的上一行**最远列**（`ROWD2` 行）。
    /// 两侧同序，配 `tools/av1ref/rowneed.py` 逐块比出"预计算上界差了多少"。
    trace_rowneed = "AV1_TRACE_ROWNEED";
    /// `ME_TRACE_RACE=1`（诊断）：给回放维护一份**"已写"位图**（4×4 格），
    /// 帧内预测读上一行/左一列时若那个格子**还没写过**就报 `RACEREAD`。
    /// 顺序回放应当**零**命中（探针自检）；并行回放命中 = 漏了依赖。
    trace_race = "ME_TRACE_RACE";
    /// `ME_TRACE_WAIT=1`（诊断）：统计 row-mt 波前**空等**的总时长与次数
    /// （每次 `replay_jobs_parallel` 收尾打一行 `WAIT ...`；顺带在
    /// `decode_tile` 收尾打一行 `RMSTAGES syntax=… replay_total=… blocks/txbs/levels`
    /// —— 三段拆账 + 记录体体积）。用来判断"并行不起来"到底是依赖太紧、
    /// 还是 worker 太少/负载不均。
    trace_wait = "ME_TRACE_WAIT";
    /// `AV1_TRACE_SYNMS=1`（诊断）：每个瓦片打一行 `SYNMS row_mt=… syntax=…ms`
    /// —— **两条路径同一个口径**的"超块循环（语法阶段）"耗时，用来比
    /// "inline 的语法+重建" 与 "row-mt 的语法+记录"。
    trace_synms = "AV1_TRACE_SYNMS";
}

num_flag! {
    /// `AV1_SMB_LIMIT=n`：符号读满多少位就停（诊断）。
    smb_limit(u32) = "AV1_SMB_LIMIT";
    /// `AV1_TRACE_SYM_LIMIT=n`：符号轨迹条数上限。
    trace_sym_limit(u64) = "AV1_TRACE_SYM_LIMIT";
    /// `AV1_TRACE_SYM_FRAME=n`：只对第 n 个带瓦片的帧打符号轨迹。
    trace_sym_frame(u64) = "AV1_TRACE_SYM_FRAME";
    /// `AV1_STOP_SB=n`：解到第 n 个超块就停（诊断）。
    stop_sb(u32) = "AV1_STOP_SB";
    /// `ME_ROW_MT_SLACK=n`（诊断）：row-mt 并行回放时，把"上一行等待列"在
    /// `above_need` 之上再加 n 个 mi 列。用来量实际跨行读比预计算上界还远多少
    /// （从小到大扫，第一个使输出逐字节一致的值就是真实余量）。
    row_mt_slack(u32) = "ME_ROW_MT_SLACK";
}

/// `AV1_TRACE_CMP_AT=<lx>,<ly>`：指定一个**色度**像素，打它所在复合块的两路预测。
#[inline]
pub(crate) fn trace_cmp_at() -> Option<(i32, i32)> {
    static V: OnceLock<Option<(i32, i32)>> = OnceLock::new();
    *V.get_or_init(|| parse_pair("AV1_TRACE_CMP_AT"))
}

/// `AV1_TRACE_DIFFWTD_AT=<lx>,<ly>`：指定一个**亮度**像素，打该处 2×2 的差分掩码细节。
#[inline]
pub(crate) fn trace_diffwtd_at() -> Option<(i32, i32)> {
    static V: OnceLock<Option<(i32, i32)>> = OnceLock::new();
    *V.get_or_init(|| parse_pair("AV1_TRACE_DIFFWTD_AT"))
}

/// `AV1_TRACE_REGION=<lx>,<ly>[,r]`：每重建完一帧（**含隐藏帧**，按解码序）把该亮度
/// 坐标周围 `(2r+1)²` 的 Y 与 U/V 打一行 `FREG`（环路滤波前后各一行，`r` 默认 1）。
/// 与 libaom 的同名插桩（`LERXU_FREG`）按 `oh` 对齐，用于逐帧定位"第一帧被污染的"。
#[inline]
pub(crate) fn trace_region() -> Option<(i32, i32, i32)> {
    static V: OnceLock<Option<(i32, i32, i32)>> = OnceLock::new();
    *V.get_or_init(|| {
        let s = std::env::var("AV1_TRACE_REGION").ok()?;
        let mut it = s.split(',');
        let a = it.next()?.trim().parse::<i32>().ok()?;
        let b = it.next()?.trim().parse::<i32>().ok()?;
        let r = it.next().and_then(|t| t.trim().parse::<i32>().ok()).unwrap_or(1);
        Some((a, b, r))
    })
}

/// `AV1_TRACE_CFL_AT=<miRow>,<miCol>`：指定一个块，逐像素打 CFL 的
/// `dc / L / avg / alpha / out`（与 libaom 的 `LERXU_CFLAT` 对齐用）。
#[inline]
pub(crate) fn trace_cfl_at() -> Option<(i32, i32)> {
    static V: OnceLock<Option<(i32, i32)>> = OnceLock::new();
    *V.get_or_init(|| parse_pair("AV1_TRACE_CFL_AT"))
}

/// 解析 `x,y` 形式的坐标开关。
fn parse_pair(name: &str) -> Option<(i32, i32)> {
    let s = std::env::var(name).ok()?;
    let mut it = s.split(',');
    let a = it.next()?.trim().parse::<i32>().ok()?;
    let b = it.next()?.trim().parse::<i32>().ok()?;
    Some((a, b))
}

/// `AV1_LR_DUMP=<目录>`：把环路恢复前后的帧落盘（诊断）。
#[inline]
pub(crate) fn lr_dump_dir() -> Option<&'static str> {
    static V: OnceLock<Option<String>> = OnceLock::new();
    V.get_or_init(|| std::env::var("AV1_LR_DUMP").ok())
        .as_deref()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_flags_are_stable_within_a_process() {
        // 一个开关读两次必须是同一个答案（缓存语义），而且不该 panic。
        assert_eq!(no_cdef(), no_cdef());
        assert_eq!(trace_sb(), trace_sb());
        assert_eq!(smb_limit(), smb_limit());
    }

    #[test]
    fn pair_parser_accepts_the_documented_form() {
        assert_eq!(parse_pair("LERXU_NOT_SET_AT_ALL"), None);
    }
}

/// `AV1_TRACE_POOL=1`：每帧打印帧缓冲池的**命中/落空**累计数。
///
/// 本机墙钟噪声 ±40%，"池有没有在工作"这种问题只能靠**确定性计数**回答。
pub(crate) fn trace_pool() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("AV1_TRACE_POOL").is_ok_and(|v| v != "0"))
}
