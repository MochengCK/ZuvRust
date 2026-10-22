//! AV1 **去块滤波**（规范 §7.14）—— 自研软解的第一个环路滤波器。
//!
//! 它是**解码环的一部分**（不是后处理）：滤波后的样本会进参考帧，所以写错一处，
//! 后续帧的预测就会跟着错。验收口径因此很硬：受控流跑出来的每一帧都要与参考
//! 解码器（对拍脚本，见 `tools/`）逐样本一致。
//!
//! ## 结构（照规范抄）
//!
//! ```text
//! loop_filter( )                      §7.14.1  三平面 × 两遍（先竖边后横边）
//!   └─ loop_filter_edge( plane, pass, row, col ) §7.14.2
//!        ├─ onScreen / isBlockEdge / isTxEdge / applyFilter
//!        ├─ filter size                §7.14.3  min(txW/H, 16 或 8)
//!        ├─ adaptive filter strength   §7.14.4/§7.14.5  lvl/limit/blimit/thresh
//!        └─ sample filtering           §7.14.6
//!             ├─ filter mask           §7.14.6.2  hevMask/filterMask/flatMask/flatMask2
//!             ├─ narrow filter         §7.14.6.3  4 抽头（hev 时只动各 1 个）
//!             └─ wide filter           §7.14.6.4  8/16 抽头低通
//! ```
//!
//! ## 三个容易错的地方
//!
//! 1. **`x`/`y` 用"未调整"的 `col`/`row` 算**，而 `MiSizes`/`LoopfilterTxSizes`/
//!    强度选择用**调整后**（`row | subY`、`col | subX`）的坐标 —— 规范里
//!    `x = col * MI_SIZE` 在 `col |= subX` **之前**。色度平面因此天然跳过
//!    "平面内部不存在的边"（x = 0 那条会被 `onScreen` 拒掉）。
//! 2. **`lvl == 0` 时要拿"另一侧"的块再算一遍强度**（§7.14.2 中间那段）——
//!    这是"边界两侧强度不一致时以强的那侧为准"的实现方式。
//! 3. **窄滤波器的 `>> 3` 是算术右移**（`filter1`/`filter2` 对负数要向下取整），
//!    用 `div_euclid` / `floor_div` 那类"数学右移"实现才与规范一致；
//!    Rust 的 `>>` 对 `i32` 恰好就是算术右移 ✓。
//!
//! 未实现的分支一律 `Err`（例如越界取样），绝不静默跳过：静默会让"滤波结果错了"
//! 变成"下一帧预测神秘漂移"，排查代价高得多。
//!
//! ## 并行（2026-10-01 深夜）：按**行带 / 列带**切
//!
//! 去块是**就地**滤波，看着像"只能串行"，但按 `(dx, dy)` 一推就清楚了：
//! `loop_filter_edge` 的最后一步是 `for i in 0..MI_SIZE { sample_filter(…, xp + dy·i, yp + dx·i, …) }`，
//! 而 `sample_filter` 的读窗口只沿 `(dx, dy)` 方向铺开。于是
//!
//! * **pass 0**（`dx = 1, dy = 0`，滤竖边）：一个 `(row, col)` 只碰像素行 `[yp, yp+4)`
//!   ⇒ 不同的 `row` 完全独立，**按行切**；
//! * **pass 1**（`dx = 0, dy = 1`，滤横边）：一个 `(row, col)` 只碰像素列 `[xp, xp+4)`
//!   ⇒ **按列切**。
//!
//! ⚠️ 反过来"pass 1 按行切"是**错**的：同一个 `col` 上相邻的 `row` 会互相读，
//! 而规范要求按 `row` 升序读到**已经滤过**的结果。
//! 细节与不变量写在 [`run_pass`] / [`PlaneView`] 上。
//!
//! 滤波（`filters::decode_frame_wrapup`）里对 `frame` 是 `&mut`，所以并行时用
//! [`PlaneView`]（裸指针 + 带的不变量）而不是把签名全改成"行切片"。
//!
//! ## SIMD（2026-10-02）：一条边的 4 条平行线一次滤完
//!
//! 与并行正交的另一刀：`loop_filter_edge` 收尾那次 4 样本循环换成一次
//! [`super::simd::deblock_edge4`]（AVX2 + SSE4.1，4 个 i32 通道 = 4 条线）。
//! 掩码（hev/filter/flat/flat2）与窄/宽滤波器全是**逐通道**运算，
//! 逐线分支（`hev` 保不保 `q1/p1`、选窄还是宽、`flat2` 走不走 16 抽头）用
//! `blendv` 融合；**竖边**的 4 条线在内存里跨 stride —— 读入与写出各做一次
//! 4×8 / 4×4 转置，横边则是每档 `k` 连续 4 个 u16 直接加载。
//!
//! 与别处 SIMD 同一个约定：**只吃内点**（窗口完全落在帧内），边界回落标量；
//! 标量路径仍是对拍基准，两条路**逐字节一致**
//! （[`tests::simd_edge_matches_scalar_bit_for_bit`] 有结构化 + 随机两段扫描）。
//! 实测（1080p 真素材，单线程采样）：去块占忙时 **6.4% → 4.2%**；
//! 整帧单线程 12247 → 11359 ms（**1.06x**）。

use super::flags;
use me_core::{MediaError, Result};

use super::block::{block_height, block_width, BlockGrid, TX_HEIGHT, TX_WIDTH};
use super::coeff::{plane_residual_size, MI_SIZE};
use super::frame_header::FrameHeader;
use super::intra::CurrFrame;
use super::mv::{GLOBAL_GLOBALMV, GLOBALMV, INTRA_FRAME, NEARESTMV};
use super::params::{MAX_LOOP_FILTER, MAX_SEGMENTS, SEG_LVL_ALT_LF_Y_V};
use super::partition::MiGrid;
use super::sequence_header::SequenceHeader;

/// `loop_filter( )`（§7.14.1）。
pub fn loop_filter(
    frame: &mut CurrFrame,
    fh: &FrameHeader,
    seq: &SequenceHeader,
    grid: &BlockGrid,
    mi_grid: &MiGrid,
) -> Result<()> {
    let lf = &fh.params.loop_filter;
    if flags::trace_dbl() {
        println!(
            "DBL 参数：level={:?} sharpness={} delta_enabled={} ref_deltas={:?} mode_deltas={:?} delta_lf_present={} multi={} res={} seg_enabled={}",
            lf.level, lf.sharpness, lf.delta_enabled, lf.ref_deltas, lf.mode_deltas,
            fh.params.delta_lf.present, fh.params.delta_lf.multi, fh.params.delta_lf.res,
            fh.params.segmentation.enabled
        );
    }
    for plane in 0..frame.num_planes as u8 {
        // 色度平面只在有强度时过滤（亮度恒做两遍）
        if plane == 0 || lf.level[1 + plane as usize] != 0 {
            for pass in 0..2u8 {
                run_pass(frame, plane, pass, fh, seq, grid, mi_grid)?;
            }
        }
    }
    Ok(())
}

/// 一遍滤波（`pass` = 0 滤竖边 / 1 滤横边），按**行带**或**列带**切成多线程跑。
///
/// ## 为什么能切（这条推理只对去块成立，搬到别处要先自己验一遍）
///
/// `loop_filter_edge` 的最后一步是
/// `for i in 0..MI_SIZE { sample_filter(…, xp + dy*i, yp + dx*i, …) }`，
/// 而 `sample_filter` 的读窗口只沿 `(dx, dy)` 方向铺开（`read_window` 取的是
/// `(x + dx·k, y + dy·k)`）。于是：
///
/// * **pass 0**（`dx = 1, dy = 0`，滤竖边）：一个 `(row, col)` 只碰像素行
///   `[yp, yp+4)`，读也不出这个行区间 ⇒ 不同的 `row` **完全独立**，按行切；
/// * **pass 1**（`dx = 0, dy = 1`，滤横边）：一个 `(row, col)` 只碰像素列
///   `[xp, xp+4)` ⇒ 按列切。
///
/// 带**内部**的顺序仍然是规范顺序（同一行里 `col` 升序 / 同一列里 `row` 升序），
/// 而带之间互不相交 ⇒ 合并起来与串行**逐位相同**（不是"差不多"，是 `md5` 一样）。
///
/// ⚠️ 反过来"pass 1 按行切"是**错**的：同一个 `col` 上相邻的 `row` 会互相读
/// （读窗口沿 y 铺开 −3..+2），而规范要求按 `row` 升序读到**已经滤过**的结果。
fn run_pass(
    frame: &mut CurrFrame,
    plane: u8,
    pass: u8,
    fh: &FrameHeader,
    seq: &SequenceHeader,
    grid: &BlockGrid,
    mi_grid: &MiGrid,
) -> Result<()> {
    let sub_x = if plane == 0 { 0 } else { i32::from(seq.color.subsampling_x) };
    let sub_y = if plane == 0 { 0 } else { i32::from(seq.color.subsampling_y) };
    let row_step = 1i32 << sub_y;
    let col_step = 1i32 << sub_x;
    let mi_rows = fh.mi_rows as i32;
    let mi_cols = fh.mi_cols as i32;

    // 切分轴：pass 0 沿 `row`、pass 1 沿 `col`
    let outer_len = if pass == 0 { mi_rows } else { mi_cols };
    let outer_step = if pass == 0 { row_step } else { col_step };
    let outer: Vec<i32> = (0..outer_len).step_by(outer_step.max(1) as usize).collect();

    // 每个 job：`outer[start..end]` + 对应的像素带 `[band_lo, band_hi)`
    let workers = super::parallel::worker_count().min(outer.len().max(1)).max(1);
    let per = outer.len().div_ceil(workers).max(1);
    let sub = if pass == 0 { sub_y } else { sub_x };
    let mut jobs: Vec<(usize, usize, i32, i32)> = Vec::new();
    let mut a = 0usize;
    while a < outer.len() {
        let b = (a + per).min(outer.len());
        // 该 job 覆盖的像素带：第一个外层下标的左上像素，到最后一个下标的右下像素
        let lo = (outer[a] * MI_SIZE) >> sub;
        let hi = ((outer[b - 1] * MI_SIZE) >> sub) + MI_SIZE;
        jobs.push((a, b, lo, hi));
        a = b;
    }

    let p = &mut frame.planes[plane as usize];
    let (ptr, stride, pw, ph, bd) = (
        if p.is_u8() {
            PlanePtr::U8(p.px8_mut().as_mut_ptr())
        } else {
            PlanePtr::U16(p.px16_mut().as_mut_ptr())
        },
        p.stride,
        p.w as i32,
        p.h as i32,
        p.bit_depth,
    );
    let mk = |band: Band| PlaneView {
        ptr,
        stride,
        plane,
        w: pw,
        h: ph,
        bit_depth: bd,
        band,
    };

    if jobs.len() <= 1 {
        // 串行：一个视图覆盖整平面（`ME_THREADS=1` 走这条，用来对拍）
        let view = mk(Band::Whole);
        for &o in &outer {
            run_line(&view, pass, o, mi_rows, mi_cols, fh, seq, grid, mi_grid)?;
        }
        return Ok(());
    }

    let views: Vec<PlaneView> = jobs
        .iter()
        .map(|&(_, _, lo, hi)| {
            mk(if pass == 0 {
                Band::Rows(lo, hi)
            } else {
                Band::Cols(lo, hi)
            })
        })
        .collect();

    let mut first_err: Option<MediaError> = None;
    std::thread::scope(|scope| {
        let handles: Vec<_> = views
            .iter()
            .zip(jobs.iter())
            .map(|(view, &(a, b, _, _))| {
                let outer = &outer;
                scope.spawn(move || -> Result<()> {
                    for &o in &outer[a..b] {
                        run_line(view, pass, o, mi_rows, mi_cols, fh, seq, grid, mi_grid)?;
                    }
                    Ok(())
                })
            })
            .collect();
        for h in handles {
            match h.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
                Err(_) => {
                    if first_err.is_none() {
                        first_err = Some(MediaError::Bitstream("去块并行的某个线程 panic 了".into()));
                    }
                }
            }
        }
    });
    if let Some(e) = first_err {
        return Err(e);
    }
    Ok(())
}

/// 一个 job 在自己那条"外层下标"上扫完一整条线。
///
/// pass 0 的 `o` 是 `row`（内层扫 `col`），pass 1 的 `o` 是 `col`（内层扫 `row`）。
/// 两条都保持"带内按规范顺序"，见 [`run_pass`] 的推理。
#[allow(clippy::too_many_arguments)]
fn run_line(
    view: &PlaneView,
    pass: u8,
    o: i32,
    mi_rows: i32,
    mi_cols: i32,
    fh: &FrameHeader,
    seq: &SequenceHeader,
    grid: &BlockGrid,
    mi_grid: &MiGrid,
) -> Result<()> {
    // 本平面的色度抽样（亮度为 0）—— 用来在平面格点与 mi 坐标之间换算。
    let (sub_x, sub_y) = if view.plane == 0 {
        (0i32, 0i32)
    } else {
        (
            i32::from(seq.color.subsampling_x),
            i32::from(seq.color.subsampling_y),
        )
    };
    // **按变换块跳着走**（2026-10-04）。`isTxEdge` 只可能在"这一格是自己所在变换块的
    // 第一格"处成立，而 `lf_tx_sizes` 每格存的正是它所属变换块的大小 ⇒ 从块首按
    // `TX_WIDTH/HEIGHT` 跳，永远恰好落在下一个块首上。原来按 `step` 逐格走，**92%
    // 的格子**要进 `lf_gate` 才被丢掉（1080p 一帧约 39 万次）；现在只走块首。
    //
    // 跳不到"该滤的边"不会发生：`[s, s + W(s))` 整段属于同一个变换块
    // （`coeff.rs` 就是按块的 span 铺 `lf_tx_sizes` 的），段内没有别的块首。
    // 万一跳到的新位置不是块首，`loop_filter_edge` 里的 `lf_gate` 也会原样丢掉它 ——
    // 与旧路径的**实际滤波集合逐边相同**（旧的"先 gate 再进"只是把这次调用省了）。
    let cols = grid.cols();
    let table = &grid.lf_tx_sizes[view.plane as usize];
    if pass == 0 {
        let row = ((o | sub_y) >> sub_y) as usize;
        let mut col = 0i32;
        while col < mi_cols {
            let tx = table[row * cols + ((col | sub_x) >> sub_x) as usize];
            loop_filter_edge(view, pass, o, col, tx, fh, seq, grid, mi_grid)?;
            col += ((TX_WIDTH[tx as usize] as i32) >> 2) << sub_x;
        }
    } else {
        let col = ((o | sub_x) >> sub_x) as usize;
        let mut row = 0i32;
        while row < mi_rows {
            let tx = table[(((row | sub_y) >> sub_y) as usize) * cols + col];
            loop_filter_edge(view, pass, row, o, tx, fh, seq, grid, mi_grid)?;
            row += ((TX_HEIGHT[tx as usize] as i32) >> 2) << sub_y;
        }
    }
    Ok(())
}

/// 去块用的**平面视图**：一个平面缓冲 + 这段 worker 允许碰的带。
///
/// `get` 仍然照旧做帧内边界检查（越界是**真的错了**，要报错）；`set` 在 release 下
/// 不做检查（规范保证落点都在帧内），两条都额外用 `debug_assert` 钉住"不越带"。
#[derive(Clone, Copy)]
enum PlanePtr {
    U8(*mut u8),
    U16(*mut u16),
}

struct PlaneView {
    ptr: PlanePtr,
    stride: usize,
    plane: u8,
    w: i32,
    h: i32,
    bit_depth: u8,
    band: Band,
}

/// 允许访问的像素带（闭开）。
#[derive(Clone, Copy)]
enum Band {
    /// 整平面（串行路径与单测）
    Whole,
    /// pass 0：只管这些**行**
    Rows(i32, i32),
    /// pass 1：只管这些**列**
    Cols(i32, i32),
}

// SAFETY: 每个 worker 只碰自己那条带里的像素，带之间互不相交（见 `run_pass` 的不变量）。
unsafe impl Send for PlaneView {}
unsafe impl Sync for PlaneView {}

impl PlaneView {
    #[inline(always)]
    fn in_band(&self, x: i32, y: i32) -> bool {
        match self.band {
            Band::Whole => true,
            Band::Rows(a, b) => y >= a && y < b,
            Band::Cols(a, b) => x >= a && x < b,
        }
    }

    #[inline(always)]
    fn is_u8(&self) -> bool {
        matches!(self.ptr, PlanePtr::U8(_))
    }

    #[inline(always)]
    fn as_u16_ptr(&self) -> *mut u16 {
        match self.ptr {
            PlanePtr::U16(p) => p,
            PlanePtr::U8(_) => unreachable!("u8 平面不能走 deblock 的 u16 SIMD 内核"),
        }
    }

    /// 读一个样本；越界返回错误（规范保证不会越界，越界说明前面读错了）。
    #[inline(always)]
    fn get(&self, x: i32, y: i32) -> Result<i32> {
        if x < 0 || y < 0 || x >= self.w || y >= self.h {
            return Err(MediaError::Bitstream(format!(
                "去块滤波取样越界：平面 {} 的 ({x},{y})（平面是 {}×{}）",
                self.plane, self.w, self.h
            )));
        }
        Ok(self.get_in_frame(x, y))
    }

    /// 读一个**已经确认落在帧内**的样本。调用方负责边界（见 [`read_window`]）。
    #[inline(always)]
    fn get_in_frame(&self, x: i32, y: i32) -> i32 {
        debug_assert!(
            x >= 0 && y >= 0 && x < self.w && y < self.h,
            "读越出帧内：({x},{y})"
        );
        debug_assert!(self.in_band(x, y), "去块读到别人的带里了：({x},{y})");
        // SAFETY: 见 `get`；此处由调用方保证。
        let i = y as usize * self.stride + x as usize;
        unsafe {
            match self.ptr {
                PlanePtr::U8(p) => i32::from(*p.add(i)),
                PlanePtr::U16(p) => i32::from(*p.add(i)),
            }
        }
    }

    #[inline(always)]
    fn set(&self, x: i32, y: i32, v: i32) {
        debug_assert!(
            x >= 0 && y >= 0 && x < self.w && y < self.h,
            "去块写址越出帧内：({x},{y})"
        );
        debug_assert!(self.in_band(x, y), "去块写到别人的带里了：({x},{y})");
        let max = (1i32 << self.bit_depth) - 1;
        // SAFETY: 同 `get`；规范保证去块的落点都在帧内。
        let i = y as usize * self.stride + x as usize;
        unsafe {
            match self.ptr {
                PlanePtr::U8(p) => *p.add(i) = v.clamp(0, max) as u8,
                PlanePtr::U16(p) => *p.add(i) = v.clamp(0, max) as u16,
            }
        }
    }
}

/// **是不是 2 的幂的整数倍**（`v % step == 0`，`step` 必为 2 的幂）。
///
/// ## 为什么单独写这一条（2026-10-02）
///
/// AV1 里"这是不是块边 / 变换块边"的判断都是这个形状，而 `Block_Width` 与
/// `Tx_Width` **全是 2 的幂**（4/8/16/32/64）。写成按位与之后语义与 `%` 逐位相同
/// （`v & (2^k - 1) == 0 ⟺ v % 2^k == 0`；`v` 由 `col * MI_SIZE` 得来，必非负），
/// 且不依赖"编译器能否从运行时表推出幂次"。
///
/// ⚠️ **实测坦白**：在这台 i5-1030NG7 上换过去**没有可测收益**（交错 min：110.8 →
/// 110.2 ms/帧，噪声内）。说明当前 rustc 已经能把这个 `%` 优化成乘法+移位序列，
/// 或者两次 `idiv` 并不是这条边的瓶颈。留着它的理由是**语义明确**（"对齐"这个
/// 意图直说）与**跨平台**（32 位 `idiv` 在别的目标上确实是 20~40 周期）。
/// 记在这里，免得后来者以为这里藏着性能。
#[inline(always)]
fn aligned(v: i32, step: i32) -> bool {
    // ⚠️ 这里用 `debug_assert!` 而不是 `assert!`：它是**逐边调用**的（1080p 一帧
    // 约 39 万次），release 下留一个 panic 分支会让 `#[inline(always)]` 进来的
    // 这段带上冷路径、反而比两个 `idiv` 还贵（2026-10-02 实测慢 3%）。
    //
    // 那 `step == 0` 了怎么办？不可能：`step` 只来自 `TX_WIDTH/HEIGHT`（全是
    // 4/8/…/64）与 `block_width/height`，而 `plane_residual_size` 的下界是
    // `BLOCK_4X4`（4）。真出现 0 说明 `MiSize` 网格坏了 —— 那种情况在
    // `loop_filter_edge` 里已经先撞上 `BLOCK_INVALID` 的检查报错了（就在本函数
    // 调用前几行），所以这里不需要再兜一层。
    debug_assert!(
        step > 0 && (step as u32).is_power_of_two(),
        "块边/变换块边的步长必须是 2 的幂（读到 {step}）"
    );
    v & (step - 1) == 0
}

/// `loop_filter_edge( plane, pass, row, col )`（§7.14.2）。
#[allow(clippy::too_many_arguments)]
/// [`loop_filter_edge`] 的**入口判定**：帧内 + 非首行列 + `isTxEdge`。
///
/// ## 为什么单独抽出来（2026-10-04）
///
/// 1080p 一帧约 **33 万次**调用 `loop_filter_edge`，而其中 **92% 在这里就退出**
/// （`AV1_TRACE_DBL` 实测：一帧只有 26 968 条边真的滤波）。这种"几乎必退"的分布下，
/// 一次函数调用的**序言 + 8 个参数搬运**成了纯开销 —— `sample` 采样里
/// `loop_filter_edge` 自己（678）竟比它的 SIMD 内核 `deblock_edge4`（380）还贵。
///
/// ⇒ 抽成 `#[inline(always)]`，让 [`run_line`] 的热循环直接把它内联展开，
/// 只有判定为真（8%）才真的进 `loop_filter_edge`。
///
/// ⚠️ **本函数是这条判定的唯一来源**：`loop_filter_edge` 自己也调它，
/// 所以不存在"两处逻辑走样"的风险。
/// `tx_sz` 由调用方**已经读好**传进来（2026-10-05）：`run_line` 按变换块跳着走，
/// 每次都已经从 `lf_tx_sizes` 取过这一格的变换尺寸；这里再取一次是**同一格同一张
/// 数组的重复读**（每次一条缓存访问）。`loop_filter_edge` 后面还要用同一个值，
/// 于是原先一次边要读三次 `lf_tx_sizes`，现在只剩一次。
#[inline(always)]
fn lf_gate(
    pass: u8,
    row_in: i32,
    col_in: i32,
    fh: &FrameHeader,
    sub_x: i32,
    sub_y: i32,
    tx_sz: u8,
) -> bool {
    // `x` / `y` 用的是**未调整**的 row/col（规范 §7.14.2 的顺序）
    let x = col_in * MI_SIZE;
    let y = row_in * MI_SIZE;
    // onScreen
    if x >= fh.frame_width as i32 || y >= fh.frame_height as i32 {
        return false;
    }
    if (pass == 0 && x == 0) || (pass == 1 && y == 0) {
        return false;
    }
    // `isTxEdge` 排在 MiSize / planeSz 之前（理由见 `loop_filter_edge` 里的长注释：
    // 92% 的边在这一步就退出，不该先付 `mi_grid.get` + `plane_residual_size`）。
    if pass == 0 {
        aligned(x >> sub_x, TX_WIDTH[tx_sz as usize] as i32)
    } else {
        aligned(y >> sub_y, TX_HEIGHT[tx_sz as usize] as i32)
    }
}

#[allow(clippy::too_many_arguments)]
fn loop_filter_edge(
    view: &PlaneView,
    pass: u8,
    row_in: i32,
    col_in: i32,
    tx_sz: u8,
    fh: &FrameHeader,
    seq: &SequenceHeader,
    grid: &BlockGrid,
    mi_grid: &MiGrid,
) -> Result<()> {
    let plane = view.plane;
    let sub_x = if plane == 0 { 0 } else { u8::from(seq.color.subsampling_x) };
    let sub_y = if plane == 0 { 0 } else { u8::from(seq.color.subsampling_y) };
    let (dx, dy) = if pass == 0 { (1i32, 0i32) } else { (0i32, 1i32) };

    // ── 入口判定抽到 [`lf_gate`]（2026-10-04）─────────────────────────────
    //
    // 原先这段（onScreen + 非首行列 + `isTxEdge`）直接写在本函数里。它是**唯一**
    // 的退出闸门，而 92% 的调用在这里就返回 —— 于是"进来再出去"的调用开销
    // （序言 + 8 个参数）比那 8% 的真滤波还贵。
    //
    // 现在由 [`run_line`] 的**热循环先内联判一次**；这里再判一次是为了让本函数
    // "单独调用也自洽"（判定已在 `lf_gate` 内联，重复的成本可忽略）。
    if !lf_gate(pass, row_in, col_in, fh, i32::from(sub_x), i32::from(sub_y), tx_sz) {
        return Ok(());
    }

    let x = col_in * MI_SIZE;
    let y = row_in * MI_SIZE;
    let row = row_in | sub_y as i32;
    let col = col_in | sub_x as i32;
    let xp = x >> sub_x;
    let yp = y >> sub_y;
    let prev_row = row - (dy << sub_y);
    let prev_col = col - (dx << sub_x);

    let cols = grid.cols() as i32;
    let plane_idx = plane as usize;

    let mi_size = mi_grid.get(row, col);
    if mi_size == super::partition::BLOCK_INVALID {
        return Err(MediaError::Bitstream(format!(
            "去块滤波：({row},{col}) 处的 MiSize 还没写过（分区树缺口）"
        )));
    }
    let plane_size =
        plane_residual_size(mi_size, plane, seq.color.subsampling_x, seq.color.subsampling_y);
    let gi = grid.idx(row, col);
    let skip = grid.skips[gi] != 0;
    let is_intra = grid.ref_frames[gi][0] <= INTRA_FRAME;
    let prev_tx_sz = grid.lf_tx_sizes[plane_idx]
        [((prev_row >> sub_y) as usize) * cols as usize + ((prev_col >> sub_x) as usize)];

    // isBlockEdge（§7.14.2）；`isTxEdge` 已在上面判过（为真才走到这里）
    let is_block_edge = if pass == 0 {
        aligned(xp, block_width(plane_size))
    } else {
        aligned(yp, block_height(plane_size))
    };
    let apply_filter = is_block_edge || !skip || is_intra;
    if !apply_filter {
        return Ok(());
    }

    // 滤波尺寸（§7.14.3）
    let base_size = if pass == 0 {
        TX_WIDTH[prev_tx_sz as usize].min(TX_WIDTH[tx_sz as usize]) as i32
    } else {
        TX_HEIGHT[prev_tx_sz as usize].min(TX_HEIGHT[tx_sz as usize]) as i32
    };
    let filter_size = if plane == 0 { 16.min(base_size) } else { 8.min(base_size) };

    // 强度（§7.14.4/§7.14.5）：本侧为 0 时改用另一侧
    let (mut lvl, mut limit, mut blimit, mut thresh) =
        filter_strength(grid, fh, row, col, plane, pass);
    if lvl == 0 {
        let (l2, li2, bl2, t2) = filter_strength(grid, fh, prev_row, prev_col, plane, pass);
        lvl = l2;
        limit = li2;
        blimit = bl2;
        thresh = t2;
    }
    if lvl <= 0 {
        return Ok(());
    }

    if flags::trace_dbl() && plane == 0 {
        println!(
            "DBL plane={plane} pass={pass} edge=({row},{col}) xp={xp} yp={yp} lvl={lvl} limit={limit} fs={filter_size} tx={tx_sz} prev_tx={prev_tx_sz} block_edge={is_block_edge} tx_edge=1 skip={skip} intra={is_intra}"
        );
    }

    // ── SIMD 快路径（2026-10-02）：一条边的 4 条平行线一次滤完 ──────────────
    //
    // 只吃**内点**：窗口（含竖边转置加载的整段 8/16 列）完全落在帧内才走，
    // 边界一律回落标量 —— 标量路径已被逐样本对拍验过，两条路合起来仍逐位一致。
    // `trace_dbl2` 的逐样本打印只存在于标量路径，开启时绕开 SIMD 保住诊断。
    let need = window_need(plane, filter_size);
    let load_pad = if need >= 7 {
        8
    } else if need >= 3 {
        4
    } else {
        2
    };
    let simd_in_bounds = if pass == 0 {
        xp - load_pad >= 0 && xp + need - 1 < view.w && yp + MI_SIZE < view.h
    } else {
        xp >= 0 && xp + MI_SIZE < view.w && yp - need >= 0 && yp + need - 1 < view.h
    };
    if simd_in_bounds && super::simd::avx2_available() && view.is_u8() && !flags::trace_dbl2() {
        deblock_edge4_u8_scratch(
            view, xp, yp, dx, dy, plane, filter_size, need, limit, blimit, thresh,
        )?;
        return Ok(());
    }
    if simd_in_bounds && super::simd::avx2_available() && !view.is_u8() && !flags::trace_dbl2() {
        // SAFETY: 已判 AVX2；上面的 `simd_in_bounds` 就是 `deblock_edge4` 的安全前提
        // （读写全部在帧内、不越出所在行；转置加载多读的列也在这个界内）。
        unsafe {
            super::simd::deblock_edge4(
                view.as_u16_ptr(),
                view.stride as isize,
                view.bit_depth,
                xp,
                yp,
                dx,
                dy,
                plane,
                filter_size,
                need,
                limit,
                blimit,
                thresh,
            );
        }
        return Ok(());
    }
    for i in 0..MI_SIZE {
        // ⚠️ 这一行就是 `run_pass` 里"按行带/列带切"的**唯一依据**：
        // `sample_filter` 的读窗口只沿 `(dx, dy)` 铺开，所以 pass 0 只碰行、pass 1 只碰列。
        sample_filter(view, xp + dy * i, yp + dx * i, limit, blimit, thresh, dx, dy, filter_size)?;
    }
    Ok(())
}

/// u8 平面的去块边：把窗口扩宽到 u16 临时缓冲，跑现有 `deblock_edge4`，
/// 再把结果收窄写回。只给 `simd_in_bounds` 的边用（窗口全在帧内）。
#[allow(clippy::too_many_arguments)]
fn deblock_edge4_u8_scratch(
    view: &PlaneView,
    xp: i32,
    yp: i32,
    dx: i32,
    dy: i32,
    plane: u8,
    filter_size: i32,
    need: i32,
    limit: i32,
    blimit: i32,
    thresh: i32,
) -> Result<()> {
    let load_pad = if need >= 7 {
        8
    } else if need >= 3 {
        4
    } else {
        2
    };
    // 竖边：窗口是 4 行 × [xp-load_pad, xp-load_pad+w)；横边：2·need 行 × 4 列。
    let (w, h, x0, y0) = if dy == 1 {
        (4usize, (2 * need) as usize, 0i32, need)
    } else {
        let w = if need >= 7 { 16 } else if need >= 3 { 8 } else { 4 };
        (w, 4usize, load_pad, 0i32)
    };
    let mut buf = [0u16; 16 * 16];
    let src_x0 = if dy == 1 { xp } else { xp - load_pad };
    let src_y0 = if dy == 1 { yp - need } else { yp };
    for r in 0..h {
        for c in 0..w {
            let sx = src_x0 + c as i32;
            let sy = src_y0 + r as i32;
            buf[r * w + c] = if sx >= 0 && sy >= 0 && sx < view.w && sy < view.h {
                view.get(sx, sy)? as u16
            } else {
                0
            };
        }
    }
    // SAFETY: 已判 AVX2；窗口按 `simd_in_bounds` 全在帧内，临时缓冲尺寸覆盖内核读写。
    unsafe {
        super::simd::deblock_edge4(
            buf.as_mut_ptr(),
            w as isize,
            view.bit_depth,
            x0,
            y0,
            dx,
            dy,
            plane,
            filter_size,
            need,
            limit,
            blimit,
            thresh,
        );
    }
    for r in 0..h {
        for c in 0..w {
            let sx = src_x0 + c as i32;
            let sy = src_y0 + r as i32;
            if sx >= 0 && sy >= 0 && sx < view.w && sy < view.h {
                view.set(sx, sy, buf[r * w + c] as i32);
            }
        }
    }
    Ok(())
}

/// 自适应滤波强度（§7.14.4 + §7.14.5）：返回 `(lvl, limit, blimit, thresh)`。
fn filter_strength(
    grid: &BlockGrid,
    fh: &FrameHeader,
    row: i32,
    col: i32,
    plane: u8,
    pass: u8,
) -> (i32, i32, i32, i32) {
    let lf = &fh.params.loop_filter;
    let i = if plane == 0 { pass as usize } else { plane as usize + 1 };
    let feature = SEG_LVL_ALT_LF_Y_V + i;

    // **常量快路径**：没有 delta_lf、没有 ref/mode delta、分段也没有 LF 特征时，
    // `lvl` 只跟 `(plane, pass)` 有关 ⇒ 连 `grid` 的四处随机读都不用做。
    // 这是很常见的一档（分段关闭或分段不调 LF 的流），而 `filter_strength`
    // 每条边都会调一次、lvl==0 时还要为另一侧再调一次。
    let seg_lf = fh.params.segmentation.enabled
        && (0..MAX_SEGMENTS)
            .any(|s| fh.params.segmentation.feature_active(s, feature));
    if !lf.delta_enabled && !fh.params.delta_lf.present && !seg_lf {
        return lf_derive(lf, lf.level[i] as i32);
    }

    let gi = grid.idx(row, col);
    let seg = grid.seg_ids[gi] as usize;
    let ref_frame = grid.ref_frames[gi][0];
    let mode = grid.y_modes[gi];
    let mode_type = if mode >= NEARESTMV && mode != GLOBALMV && mode != GLOBAL_GLOBALMV {
        1usize
    } else {
        0usize
    };
    let dl = &fh.params.delta_lf;
    let delta_lf = if !dl.multi {
        grid.delta_lfs[gi][0]
    } else {
        grid.delta_lfs[gi][if plane == 0 { pass as usize } else { plane as usize + 1 }]
    };

    // §7.14.5：baseFilterLevel → 分割特征 → ref/mode delta
    let mut lvl = (delta_lf + lf.level[i] as i32).clamp(0, MAX_LOOP_FILTER);
    let segp = &fh.params.segmentation;
    if segp.feature_active(seg, feature) {
        lvl = (segp.feature_data[seg][feature] + lvl).clamp(0, MAX_LOOP_FILTER);
    }
    if lf.delta_enabled {
        let n_shift = lvl >> 5;
        if ref_frame as i32 == INTRA_FRAME as i32 {
            lvl += lf.ref_deltas[INTRA_FRAME as usize] << n_shift;
        } else if (ref_frame as usize) < lf.ref_deltas.len() {
            lvl += (lf.ref_deltas[ref_frame as usize] << n_shift)
                + (lf.mode_deltas[mode_type] << n_shift);
        }
        lvl = lvl.clamp(0, MAX_LOOP_FILTER);
    }

    lf_derive(lf, lvl)
}

/// `limit / blimit / thresh` 由 `sharpness` 与 `lvl` 推出（§7.14.4/§7.14.5）。
#[inline]
fn lf_derive(lf: &super::params::LoopFilterParams, lvl: i32) -> (i32, i32, i32, i32) {
    let sharp = lf.sharpness as i32;
    let shift = if sharp > 4 {
        2
    } else if sharp > 0 {
        1
    } else {
        0
    };
    let limit = if sharp > 0 {
        (lvl >> shift).clamp(1, 9 - sharp)
    } else {
        1.max(lvl >> shift)
    };
    (lvl, limit, 2 * (lvl + 2) + limit, lvl >> 4)
}

/// 一条边需要读多宽的窗口（半径上界，见 [`read_window`] 的 `need`）：
/// 16 抽头 → 7、8 抽头亮度 → 4、色度 → 3、4 抽头 → 2。
///
/// 标量取样与 SIMD 快路径的内点判据**共用这一份**（两处各写一份 = 定时炸弹）。
#[inline(always)]
fn window_need(plane: u8, filter_size: i32) -> i32 {
    if filter_size == 4 {
        2
    } else if plane != 0 {
        3
    } else if filter_size == 8 {
        4
    } else {
        7
    }
}

/// 一个样本位置上的边（`sample_filter`，§7.14.6）：先算掩码，再选窄/宽滤波器。
#[allow(clippy::too_many_arguments)]
fn sample_filter(
    view: &PlaneView,
    x: i32,
    y: i32,
    limit: i32,
    blimit: i32,
    thresh: i32,
    dx: i32,
    dy: i32,
    filter_size: i32,
) -> Result<()> {
    let plane = view.plane;
    let bit_depth = view.bit_depth as i32;
    let bd_shift = bit_depth - 8;

    // 取样本：`p_k` 在边前第 k+1 个，`q_k` 在边后第 k 个（§7.14.6.2）
    // 取样位置的统一口径（规范 §7.14.6.2）：
    //   `q_k = CurrFrame[y + k*dy][x + k*dx]`（k ≥ 0），
    //   `p_k = CurrFrame[y - (k+1)*dy][x - (k+1)*dx]` ——
    //   用 `k` 表示时 p 就是**负方向**：`p_{m}` 在 `x + dx*(-m-1)`，
    //   等价于"`k` 取负值时位置也是 `x + dx*k`"。所以 `at(k) = (x + k*dx, y + k*dy)`。
    //   （这里曾写成 `x - dx*(k+1)`：k=-1 取到的是 q0 本身，于是掩码永远拿"同一个
    //   样本"比、滤波只在自己身上绕圈 —— 症状是"平坦区没效果、台阶处效果错"。）
    //
    // 采样**只读一次**：原本每个掩码分支各回平面读一遍（16 抽头那条路一次
    // `sample_filter` 要读上百个样本），现在一次读进 14 项的窗口，掩码与滤波器
    // 都从窗口取 —— 这是去块里最大的一处浪费（`wide_filter` 曾占忙时 13%）。
    let need = window_need(plane, filter_size);
    let w = read_window(view, x, y, dx, dy, need)?;
    let p1 = w[wi(-2)];
    let p0 = w[wi(-1)];
    let q0 = w[wi(0)];
    let q1 = w[wi(1)];

    // hevMask（§7.14.6.2）
    let thresh_bd = thresh << bd_shift;
    let hev_mask = ((p1 - p0).abs() > thresh_bd) as u8 | (((q1 - q0).abs() > thresh_bd) as u8);

    // filterLen
    let filter_len = if filter_size == 4 {
        4
    } else if plane != 0 {
        6
    } else if filter_size == 8 {
        8
    } else {
        16
    };

    // filterMask
    let limit_bd = limit << bd_shift;
    let blimit_bd = blimit << bd_shift;
    let mut mask = 0u8;
    mask |= ((p1 - p0).abs() > limit_bd) as u8;
    mask |= ((q1 - q0).abs() > limit_bd) as u8;
    mask |= ((p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 > blimit_bd) as u8;
    if filter_len >= 6 {
        let p2 = w[wi(-3)];
        let q2 = w[wi(2)];
        mask |= ((p2 - p1).abs() > limit_bd) as u8;
        mask |= ((q2 - q1).abs() > limit_bd) as u8;
    }
    if filter_len >= 8 {
        let p3 = w[wi(-4)];
        let q3 = w[wi(3)];
        mask |= ((p3 - w[wi(-3)]).abs() > limit_bd) as u8;
        mask |= ((q3 - w[wi(2)]).abs() > limit_bd) as u8;
    }
    let filter_mask = mask == 0;
    if flags::trace_dbl2() && x < 12 && y < 4 {
        println!(
            "  SF x={x} y={y} fs={filter_size} fl={filter_len} p1={p1} p0={p0} q0={q0} q1={q1} mask={mask} hev={hev_mask} limit={limit} blimit={blimit}"
        );
    }
    if !filter_mask {
        return Ok(());
    }

    // flatMask（filterSize >= 8 才需要）
    let mut flat_mask = true;
    if filter_size >= 8 {
        let threshold_bd = 1i32 << bd_shift;
        let mut m = 0u8;
        m |= ((p1 - p0).abs() > threshold_bd) as u8;
        m |= ((q1 - q0).abs() > threshold_bd) as u8;
        let p2 = w[wi(-3)];
        let q2 = w[wi(2)];
        m |= ((p2 - p0).abs() > threshold_bd) as u8;
        m |= ((q2 - q0).abs() > threshold_bd) as u8;
        if filter_len >= 8 {
            let p3 = w[wi(-4)];
            let q3 = w[wi(3)];
            m |= ((p3 - p0).abs() > threshold_bd) as u8;
            m |= ((q3 - q0).abs() > threshold_bd) as u8;
        }
        flat_mask = m == 0;
    }

    // flatMask2（filterSize >= 16 才需要）
    let mut flat_mask2 = true;
    if filter_size >= 16 {
        let threshold_bd = 1i32 << bd_shift;
        let mut m = 0u8;
        let p6 = w[wi(-7)];
        let p5 = w[wi(-6)];
        let p4 = w[wi(-5)];
        let q4 = w[wi(4)];
        let q5 = w[wi(5)];
        let q6 = w[wi(6)];
        m |= ((p6 - p0).abs() > threshold_bd) as u8;
        m |= ((q6 - q0).abs() > threshold_bd) as u8;
        m |= ((p5 - p0).abs() > threshold_bd) as u8;
        m |= ((q5 - q0).abs() > threshold_bd) as u8;
        m |= ((p4 - p0).abs() > threshold_bd) as u8;
        m |= ((q4 - q0).abs() > threshold_bd) as u8;
        flat_mask2 = m == 0;
    }

    if filter_size == 4 || !flat_mask {
        narrow_filter(view, x, y, dx, dy, hev_mask != 0, bit_depth, &w)?;
    } else if filter_size == 8 || !flat_mask2 {
        wide_filter(view, x, y, dx, dy, 3, &w)?;
    } else {
        wide_filter(view, x, y, dx, dy, 4, &w)?;
    }
    Ok(())
}

/// 一条边周围的样本窗口：`w[wi(k)] = CurrFrame[y + k·dy][x + k·dx]`。
///
/// `k ∈ [-7, 6]`：16 抽头的宽滤波器在 `log2Size = 4` 时 `p` 能取到
/// `[-(n+1), n] = [-7, 6]`（§7.14.6.4）。
type Win = [i32; 14];

/// 宽滤波器最多产出多少个样本（`log2Size = 4` ⇒ `2n = 12`）。
const MAX_WIDE_OUT: usize = 12;

/// 窗口下标：`k = 0` 落在正中间（第 7 项）。
#[inline(always)]
const fn wi(k: i32) -> usize {
    (7 + k) as usize
}

/// 规范取抽头时的**下标钳位**（§7.14.6.4 的 `p = Clip3(-(n+1), n, i+j)`）
/// 之后再换成窗口下标 —— 抽头只会落在窗口内，所以先钳位就不需要额外边界判断。
#[inline(always)]
fn wcl<const N: i32>(k: i32) -> usize {
    wi(k.clamp(-(N + 1), N))
}

/// 把一条边周围的样本**一次**读进窗口。
///
/// `need` = 需要的半径上界：16 抽头 → 7、8 抽头亮度 → 4、色度 → 3、4 抽头 → 2。
/// **只读真正会用到的范围**：多读会撞上 `read_px` 的越界报错（规范只保证用到的那部分在帧内）。
///
/// ## 为什么先做一次范围检查再逐点读
///
/// 旧写法是每个样本调一次 [`PlaneView::get`]，于是同一条边要构造 4~14 个
/// `Result` 并各自比较 4 个坐标 —— 16 抽头那条路（亮度 tx 边 16×16 的常客）
/// 一次就是 14 次。这里先把**两个端点**算出来：窗口是沿 `(dx, dy)` 的一条**直线**
/// 上的连续点，所以端点的 min/max 就是整条窗口的包围盒，一次检查 ≡ 逐样本检查。
///
/// ⚠️ `dx`/`dy` 是单位向量（`(1,0)` 或 `(0,1)`），所以包围盒退化成一条线段；
/// 若将来出现斜向的取样，这里的 min/max 仍然成立，不必改。
fn read_window(view: &PlaneView, x: i32, y: i32, dx: i32, dy: i32, need: i32) -> Result<Win> {
    let ax = x - dx * need;
    let ay = y - dy * need;
    let bx = x + dx * (need - 1);
    let by = y + dy * (need - 1);
    let (lo_x, hi_x) = if ax <= bx { (ax, bx) } else { (bx, ax) };
    let (lo_y, hi_y) = if ay <= by { (ay, by) } else { (by, ay) };
    if lo_x < 0 || lo_y < 0 || hi_x >= view.w || hi_y >= view.h {
        // 报错要与逐点检查时一致：给出**第一个**越界的样本坐标。
        for k in -need..need {
            let (sx, sy) = (x + dx * k, y + dy * k);
            if sx < 0 || sy < 0 || sx >= view.w || sy >= view.h {
                return Err(MediaError::Bitstream(format!(
                    "去块滤波取样越界：平面 {} 的 ({sx},{sy})（平面是 {}×{}）",
                    view.plane, view.w, view.h
                )));
            }
        }
        // 包围盒说不越界但逐点说没越界 → 不可能（包围盒是逐点的超集）
        unreachable!("read_window 的包围盒检查与逐点检查必须一致");
    }
    let mut w = [0i32; 14];
    for k in -need..need {
        w[wi(k)] = view.get_in_frame(x + dx * k, y + dy * k);
    }
    Ok(w)
}

/// 窄滤波器（§7.14.6.3）：4 抽头；`hevMask` 时只改边界两侧各一个样本。
#[allow(clippy::too_many_arguments)]
fn narrow_filter(
    view: &PlaneView,
    x: i32,
    y: i32,
    dx: i32,
    dy: i32,
    hev: bool,
    bit_depth: i32,
    w: &Win,
) -> Result<()> {
    let off = 0x80i32 << (bit_depth - 8);
    let q0 = w[wi(0)];
    let q1 = w[wi(1)];
    let p0 = w[wi(-1)];
    let p1 = w[wi(-2)];
    let ps1 = p1 - off;
    let ps0 = p0 - off;
    let qs0 = q0 - off;
    let qs1 = q1 - off;
    let mut filter = if hev { filter4_clamp(ps1 - qs1, bit_depth) } else { 0 };
    filter = filter4_clamp(filter + 3 * (qs0 - ps0), bit_depth);
    let filter1 = filter4_clamp(filter + 4, bit_depth) >> 3;
    let filter2 = filter4_clamp(filter + 3, bit_depth) >> 3;
    let oq0 = filter4_clamp(qs0 - filter1, bit_depth) + off;
    let op0 = filter4_clamp(ps0 + filter2, bit_depth) + off;
    view.set(x, y, oq0);
    view.set(x - dx, y - dy, op0);
    if !hev {
        let filter = (filter1 + 1) >> 1; // Round2(filter1, 1)
        let oq1 = filter4_clamp(qs1 - filter, bit_depth) + off;
        let op1 = filter4_clamp(ps1 + filter, bit_depth) + off;
        view.set(x + dx, y + dy, oq1);
        view.set(x - dx * 2, y - dy * 2, op1);
    }
    Ok(())
}

/// 宽滤波器（§7.14.6.4）：`log2Size` = 3（8 抽头）或 4（16 抽头）。
///
/// 抽头全部从窗口里取（不再回平面读），滤波结果先用定长数组收着再写回
/// —— 因为 `f[i]` 之间会互相覆盖（规范也是先算完再写）。
///
/// ## 为什么拆成三个常量泛型的特化（2026-10-02）
///
/// 采样里这个函数单独占 **20.5%**（去块里的第一大头）。原因不在算术量，而在
/// 旧写法的**形状**：
///
/// * 每算一个 `f[i]` 都跑一遍 `j = -n..=n`，每个 `j` 要 `clamp` 一次下标、
///   再判一次 `j.abs() <= n2` —— 而 `n`（6/3/2）与 `n2`（0/1）**只由
///   `log2Size` 与平面决定**，也就是全帧只取 3 组值；
/// * 于是编译期什么都不知道，`t += w[wi(p)] * tap` 里的两个连续数组下标各带一次
///   边界检查与乘法。
///
/// 拆成 `WIDE_8_LUMA`（n=3, n2=0）/ `WIDE_8_CHROMA`（n=2, n2=1）/
/// `WIDE_16`（n=6, n2=1）三个 `const N` 之后，`n` 与 `n2` 变成编译期常量，
/// `tap` 的选择被折叠成常量、`j.abs()` 不再是分支、`p` 的钳位也只剩 `min`/`max`
/// 两条指令。**算术顺序原样保留**（`t` 的累加仍然从 `j = -n` 到 `n` 单调递增），
/// 所以结果与旧实现逐位相同 —— 有单测 [`tests::wide_filter_matches_the_running_sum_reference`] 钉住。
#[allow(clippy::too_many_arguments)]
fn wide_filter(
    view: &PlaneView,
    x: i32,
    y: i32,
    dx: i32,
    dy: i32,
    log2_size: i32,
    w: &Win,
) -> Result<()> {
    let plane = view.plane;
    // 规范里 `n` 与 `n2` 的取值（`log2Size == 3` 且亮度时 `n2 = 0`）
    if log2_size == 4 {
        wide_filter_const::<6, 1>(view, x, y, dx, dy, 4, w)
    } else if plane == 0 {
        wide_filter_const::<3, 0>(view, x, y, dx, dy, 3, w)
    } else {
        wide_filter_const::<2, 1>(view, x, y, dx, dy, 3, w)
    }
}

/// [`wide_filter`] 的实际内核：`N` = `n`、`N2` = `n2`、`L2` = `log2Size`。
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn wide_filter_const<const N: i32, const N2: i32>(
    view: &PlaneView,
    x: i32,
    y: i32,
    dx: i32,
    dy: i32,
    log2_size: i32,
    w: &Win,
) -> Result<()> {
    // `f[i] = Round2(Σ_{j=-N..N} tap(j)·w[clip(i+j)], log2Size)`
    //
    // 内层按**编译期已知的 N/N2** 展开：`tap` 的选择被折叠成常量（`|j| <= N2`
    // 在 N2 = 0 时恒假、在 N2 = 1 时只对 j ∈ {-1,0,1} 为真），`j.abs()` 不再是
    // 运行时分支，`p` 的钳位只剩 `min`/`max` 两条指令。
    //
    // ⚠️ 累加顺序**必须**保持 `j` 从 `-N` 递增到 `N`：整数加法虽然可结合，
    // 但 i32 溢出的行为（wrap）在重排后可能不同 —— 这里的值域远不会溢出，
    // 保持同序是为了让"与旧实现逐位相同"这件事不依赖值域论证。旧实现留在
    // 单测里当参照（`wide_filter_reference`）。
    let mut f = [0i32; MAX_WIDE_OUT];
    for (i, slot) in f.iter_mut().enumerate().take((2 * N) as usize) {
        let i = i as i32 - N;
        let mut t = 0i32;
        let mut j = -N;
        while j <= N {
            let tap = if j.abs() <= N2 { 2 } else { 1 };
            t += w[wcl::<N>(i + j)] * tap;
            j += 1;
        }
        *slot = (t + (1 << (log2_size - 1))) >> log2_size; // Round2(t, log2Size)
    }
    for (i, v) in f.iter().enumerate().take((2 * N) as usize) {
        let i = i as i32 - N;
        view.set(x + i * dx, y + i * dy, *v);
    }
    Ok(())
}

/// `filter4_clamp( value )`（§7.14.6.3）。
#[inline(always)]
fn filter4_clamp(v: i32, bit_depth: i32) -> i32 {
    v.clamp(-(1 << (bit_depth - 1)), (1 << (bit_depth - 1)) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **旧实现**的 `wide_filter`，原样保留作为参照。
    ///
    /// 2026-10-02：`wide_filter` 重写成常量泛型的直写法（去掉运行时 `clamp` 与
    /// `tap` 分支）。这类"重写热路径"的改动最容易在某个角上偏一格（比如 `p` 的
    /// 钳位范围写错），而且**整帧 md5 未必抓得到**（落在边界外的组合可能本素材
    /// 不出现）。所以把老代码留在这里，逐样本对拍。
    fn wide_filter_reference(
        view: &PlaneView,
        x: i32,
        y: i32,
        dx: i32,
        dy: i32,
        log2_size: i32,
        w: &Win,
    ) -> Result<()> {
        let plane = view.plane;
        let n: i32 = if log2_size == 4 {
            6
        } else if plane == 0 {
            3
        } else {
            2
        };
        let n2: i32 = if log2_size == 3 && plane == 0 { 0 } else { 1 };

        let mut f = [0i32; 12];
        let mut i = -n;
        while i < n {
            let mut t = 0i32;
            let mut j = -n;
            while j <= n {
                let p = (i + j).clamp(-(n + 1), n);
                let tap = if j.abs() <= n2 { 2 } else { 1 };
                t += w[wi(p)] * tap;
                j += 1;
            }
            f[(i + n) as usize] = (t + (1 << (log2_size - 1))) >> log2_size;
            i += 1;
        }
        let mut i = -n;
        while i < n {
            view.set(x + i * dx, y + i * dy, f[(i + n) as usize]);
            i += 1;
        }
        Ok(())
    }

    /// 造一个能同时跑到"钳位"与"加倍的抽头"的窗口：值取非单调序列，
    /// 免得对称索引的笔误被抵消掉。
    fn probe_win() -> Win {
        let mut w = [0i32; 14];
        for (k, slot) in w.iter_mut().enumerate() {
            *slot = ((k as i32 * 37) % 61) + 3;
        }
        w
    }

    /// 三档形状（16 抽头 / 8 抽头亮度 / 8 抽头色度）都要与旧实现逐位一致。
    #[test]
    fn wide_filter_matches_the_running_sum_reference() {
        use crate::video::intra::{CurrFrame, Plane};
        let w = probe_win();
        // 三档：(log2_size, plane)
        for (log2_size, plane) in [(4i32, 0u8), (3, 0), (3, 1)] {
            let mut frame = CurrFrame::new(32, 32, true, true, 3, 8);
            frame.planes[0] = Plane::new(32, 32, 8);
            let mut a = frame.planes.clone();
            let mut b = frame.planes.clone();
            let va = PlaneView {
                ptr: PlanePtr::U16(a[0].px16_mut().as_mut_ptr()),
                stride: a[0].stride,
                plane,
                w: 32,
                h: 32,
                bit_depth: 8,
                band: Band::Whole,
            };
            let vb = PlaneView {
                ptr: PlanePtr::U16(b[0].px16_mut().as_mut_ptr()),
                stride: b[0].stride,
                plane,
                w: 32,
                h: 32,
                bit_depth: 8,
                band: Band::Whole,
            };
            // 竖向与横向各跑一次（`set` 的地址推导不同）
            for (dx, dy) in [(1i32, 0i32), (0, 1)] {
                wide_filter(&va, 16, 16, dx, dy, log2_size, &w).unwrap();
                wide_filter_reference(&vb, 16, 16, dx, dy, log2_size, &w).unwrap();
            }
            assert_eq!(
                a[0].px16(), b[0].px16(),
                "wide_filter 与旧实现不一致：log2_size={log2_size} plane={plane}"
            );
        }
    }

    /// 窗口读取：越界的报错坐标要与逐点检查给出**同一个**样本。
    #[test]
    fn read_window_reports_the_first_out_of_range_sample() {
        use crate::video::intra::{CurrFrame, Plane};
        let mut frame = CurrFrame::new(16, 16, false, false, 1, 8);
        frame.planes[0] = Plane::new(16, 16, 8);
        let view = PlaneView {
            ptr: PlanePtr::U16(frame.planes[0].px16_mut().as_mut_ptr()),
            stride: frame.planes[0].stride,
            plane: 0,
            w: 16,
            h: 16,
            bit_depth: 8,
            band: Band::Whole,
        };
        // 横向：x=1、need=4 ⇒ 窗口 [-3, 4)，第一个越界的是 x=-3
        let e = read_window(&view, 1, 8, 1, 0, 4).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("(-3,8)"), "报错应当指向第一个越界样本，实际：{msg}");
        // 竖向：y=1、need=7 ⇒ 窗口从 k=-7 起，第一个越界的是 y = 1-7 = -6
        let e = read_window(&view, 8, 1, 0, 1, 7).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("(8,-6)"), "报错应当指向第一个越界样本，实际：{msg}");
        // 完全在帧内：不该报错，且窗口值就是平面里的值
        let w = read_window(&view, 8, 8, 1, 0, 2).unwrap();
        for k in -2..2 {
            assert_eq!(w[wi(k)], view.get_in_frame(8 + k, 8));
        }
    }

    /// `aligned` 必须与 `%` 在**整个可能的步长集合**上等价 —— 它替掉了
    /// `loop_filter_edge` 里两处取模（`Block_Width` / `Tx_Width` 都是 2 的幂）。
    #[test]
    fn aligned_matches_the_modulo_it_replaces() {
        for step in [4i32, 8, 16, 32, 64] {
            for v in -0..2000 {
                assert_eq!(
                    aligned(v, step),
                    v % step == 0,
                    "aligned({v}, {step}) 与 % 不一致"
                );
            }
        }
    }

    /// `aligned` 的入参 `step` 只可能是 2 的幂；非法值在 debug 档要**明确报错**
    /// 而不是静默返回 false（静默 = 把"算出非法 MiSize"变成"这条边不滤"）。
    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "2 的幂"))]
    fn aligned_rejects_a_non_power_of_two_step() {
        let _ = aligned(4, 0);
    }

    /// `delta_lfs` 网格的维度与 `FRAME_LF_COUNT` 一致（去块索引会用到）。
    #[test]
    fn delta_lf_grid_has_four_slots() {
        let grid = BlockGrid::new(4, 4);
        assert_eq!(grid.delta_lfs.len(), 16);
        assert_eq!(grid.delta_lfs[0].len(), super::super::params::FRAME_LF_COUNT);
    }

    /// 平坦区域（全同值）：`filterMask` 必为假（差值全 0 ≤ blimit），滤波器不动像素。
    #[test]
    fn flat_region_is_left_untouched() {
        use crate::video::intra::Plane;
        let mut plane = Plane::new(16, 16, 8);
        for y in 0..16 {
            for x in 0..16 {
                plane.set(x, y, 100);
            }
        }
        let mut frame = CurrFrame::new(16, 16, false, false, 1, 8);
        frame.planes[0] = plane;
        let view = PlaneView {
            ptr: PlanePtr::U16(frame.planes[0].px16_mut().as_mut_ptr()),
            stride: frame.planes[0].stride,
            plane: 0,
            w: 16,
            h: 16,
            bit_depth: 8,
            band: Band::Whole,
        };
        sample_filter(&view, 8, 8, 8, 16, 4, 1, 0, 4).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(frame.planes[0].get(x, y), 100, "平坦区不该被改动");
            }
        }
    }

    /// 给对拍测试造四种形态的 64×64 样本：噪声 / 大块台阶 / 细台阶（近值）/
    /// 近平坦带尖刺 —— 目的是把 `filterMask` / `flatMask` / `flatMask2` / `hevMask`
    /// 的各类组合都逼出来（纯噪声只会全部"不平"，纯平坦又全部"不滤"）。
    #[cfg(target_arch = "x86_64")]
    fn fill_pattern(p: &mut crate::video::intra::Plane, shape: u32, seed: u32, bit_depth: u8) {
        let max = (1i32 << bit_depth) - 1;
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(shape.wrapping_mul(7919));
        let mut next = move || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as i32
        };
        for y in 0..64usize {
            for x in 0..64usize {
                let v = match shape {
                    0 => next() % (max + 1),
                    1 => ((x as i32 / 16 + y as i32 / 16) % 3) * (max / 2) + next() % 4,
                    2 => {
                        let base =
                            if (x as i32 / 4 + y as i32 / 4) % 2 == 0 { max / 3 } else { max / 3 + 2 };
                        base + next() % 5
                    }
                    _ => {
                        if (x * 7 + y * 13) % 37 == 0 {
                            (next() % 2) * max
                        } else {
                            max / 2
                        }
                    }
                };
                p.set(x, y, v.clamp(0, max) as u16);
            }
        }
    }

    /// 跑一条边的对拍：标量 4 次 `sample_filter` vs SIMD 一次 `deblock_edge4`。
    #[cfg(target_arch = "x86_64")]
    #[allow(clippy::too_many_arguments)]
    fn check_one_edge(
        plane: u8,
        filter_size: i32,
        bd: u8,
        shape: u32,
        seed: u32,
        dx: i32,
        dy: i32,
        xp: i32,
        yp: i32,
        limit: i32,
        blimit: i32,
        thresh: i32,
        changed: &mut usize,
    ) {
        use crate::video::intra::Plane;
        let need = window_need(plane, filter_size);
        let load_pad = if need >= 7 { 8 } else if need >= 3 { 4 } else { 2 };
        let ok = if dx == 1 {
            xp - load_pad >= 0 && xp + need - 1 < 64 && yp + MI_SIZE <= 64
        } else {
            xp + MI_SIZE <= 64 && yp - need >= 0 && yp + need - 1 < 64
        };
        assert!(ok, "测试位置不是内点：({xp},{yp}) dx={dx} need={need}");

        let mut a = Plane::new(64, 64, bd);
        let mut b = Plane::new(64, 64, bd);
        fill_pattern(&mut a, shape, seed, bd);
        b.px16_mut().copy_from_slice(a.px16());
        let orig = a.px16().to_vec();

        let va = PlaneView {
            ptr: PlanePtr::U16(a.px16_mut().as_mut_ptr()),
            stride: a.stride,
            plane,
            w: 64,
            h: 64,
            bit_depth: bd,
            band: Band::Whole,
        };
        for i in 0..MI_SIZE {
            sample_filter(
                &va,
                xp + dy * i,
                yp + dx * i,
                limit,
                blimit,
                thresh,
                dx,
                dy,
                filter_size,
            )
            .unwrap();
        }
        // SAFETY: 已判 AVX2；位置是内点（上面的 assert），满足 `deblock_edge4` 的范围要求。
        unsafe {
            super::super::simd::deblock_edge4(
                b.px16_mut().as_mut_ptr(),
                b.stride as isize,
                bd,
                xp,
                yp,
                dx,
                dy,
                plane,
                filter_size,
                need,
                limit,
                blimit,
                thresh,
            );
        }
        if a.px16() != b.px16() {
            let mut diffs = String::new();
            let mut shown = 0;
            for (i, (x, y)) in a.px16().iter().zip(b.px16().iter()).enumerate() {
                if x != y && shown < 8 {
                    diffs.push_str(&format!(
                        "  ({},{}) 标量 {x} / SIMD {y}\n",
                        i % a.stride,
                        i / a.stride
                    ));
                    shown += 1;
                }
            }
            panic!(
                "SIMD 与标量不一致：plane={plane} fs={filter_size} bd={bd} shape={shape} \
                 seed={seed} dx={dx} dy={dy} ({xp},{yp}) limit={limit} blimit={blimit} \
                 thresh={thresh}\n{diffs}"
            );
        }
        if a.px16() != orig {
            *changed += 1;
        }
    }

    /// **SIMD 内核必须与标量逐字节一致**（`simd::deblock_edge4` vs 4 次 `sample_filter`）。
    ///
    /// 两段扫描：
    /// 1. **结构化**：7 组 (平面, 滤波尺寸, 位深) × 4 种形态 × 多个种子 × 两组强度 ×
    ///    竖边/横边两个位置；
    /// 2. **随机**：位置、强度、内容全随机 —— 专门撞"某一档掩码 / 某一支滤波器"的组合。
    ///
    /// 判据都是**整个平面缓冲逐字节相等**（含 padding 区）。
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_edge_matches_scalar_bit_for_bit() {
        if !super::super::simd::avx2_available() {
            println!("这台机器没有 AVX2，跳过");
            return;
        }
        let cases: [(u8, i32, u8); 7] = [
            (0, 4, 8),
            (0, 8, 8),
            (0, 16, 8),
            (1, 4, 8),
            (1, 8, 8),
            (0, 16, 12),
            (2, 8, 10),
        ];
        let mut changed = 0usize;
        for (plane, filter_size, bd) in cases {
            let max = (1i32 << bd) - 1;
            for shape in 0..4u32 {
                for seed in 0..4u32 {
                    for &(dx, dy, xp, yp) in &[(1i32, 0i32, 16i32, 16i32), (0, 1, 24, 32)] {
                        for &(limit, blimit, thresh) in
                            &[(2i32, 8i32, 0i32), (max / 8, max / 2, max / 16)]
                        {
                            check_one_edge(
                                plane, filter_size, bd, shape, seed, dx, dy, xp, yp, limit,
                                blimit, thresh, &mut changed,
                            );
                        }
                    }
                }
            }
        }
        // 随机扫：位置、强度、内容都随机
        let mut rng = 0x0bad_f00du32;
        let mut next = move || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            rng >> 8
        };
        for (plane, filter_size, bd) in cases {
            let need = window_need(plane, filter_size);
            for shape in 0..48u32 {
                let vertical = next() % 2 == 0;
                let (dx, dy) = if vertical { (1i32, 0i32) } else { (0, 1) };
                let (xp, yp) = if vertical {
                    (8 + (next() % 48) as i32, (next() % 57) as i32)
                } else {
                    ((next() % 60) as i32, need + (next() % (64 - 2 * need + 1) as u32) as i32)
                };
                let limit = (next() % 64) as i32;
                let blimit = (next() % 200) as i32;
                let thresh = (next() % 64) as i32;
                check_one_edge(
                    plane, filter_size, bd, shape, next(), dx, dy, xp, yp, limit, blimit, thresh,
                    &mut changed,
                );
            }
        }
        // 防"全是空转"：至少有一部分组合真的滤了像素
        assert!(changed > 100, "对拍样本大多没被滤（changed={changed}），测试形同虚设");
    }
}
