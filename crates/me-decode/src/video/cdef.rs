//! AV1 **CDEF**（约束方向增强滤波，规范 §7.15）。
//!
//! CDEF 是**环路滤波的第一级**（在去块之后、环路恢复之前，见 [`super::filters`]）：
//! 它按 8×8 块检测"边缘方向"，然后只沿**垂直于边缘**的方向做一次受限的
//! 5 抽头滤波 —— 目的是去掉压缩产生的"振铃"（ringing），同时不糊掉边缘。
//!
//! ```text
//! CDEF( )                    §7.15    逐 8×8 块（参数按 64×64 一组）
//!   ├─ cdef_block( r, c )    §7.15.1  拷贝 → skip 判据 → 亮度/色度各滤一次
//!   │    └─ cdef_direction   §7.15.2  用**亮度**样本算 8 个方向的代价，取最大
//!   └─ CDEF filter process   §7.15.3  主抽头 2 个 + 次抽头 2×(两个方向)，共 12 个样本
//! ```
//!
//! ## 四个容易写错的地方
//!
//! 1. **`cdef_idx` 按 64×64 分组**：`baseR = r & ~15`、`baseC = c & ~15`，
//!    然后**查表**取这一组的参数。写成"按 8×8 取"会在跨 64×64 时取到别的组。
//! 2. **`priStr` 有两处修正**：先 `<< coeffShift`（高位深），再按 `var` 缩放
//!    `(priStr * (4 + varStr) + 8) >> 4`。`var == 0`（块内没有方向性）时
//!    **`priStr` 直接归零** —— 此时只剩次抽头在工作。
//! 3. **可用性判据是"帧级"的**（`is_inside_filter_region` 用 `MiRows`/`MiCols`，
//!    不是瓦片范围）：候选样本只要落在帧的 4×4 网格内就算可用，取的是
//!    `CurrFrame` 的样本。越出这个范围的样本**完全不参与**（不加进 sum、
//!    也不参与 min/max）—— 少写一个 `CdefAvailable` 判断会让边缘整体偏。
//! 4. **色度用的是 `Cdef_Uv_Dir` 重映射后的方向**，而且 `damping` 比亮度**小 1**。
//!
//! ## 取址与帧边界
//!
//! 规范里 `cdef_get_at` 的候选样本可以落在**帧外**（`MiRows`/`MiCols` 是向上取整到
//! 8 的值），而 `CurrFrame` 在规范里是带 border 的缓冲。我们用 [`super::intra::Plane`]
//! 的 frame border 承接这些取址（见 `intra::PLANE_PAD`），所以坐标只要落在
//! "帧的 4×4 网格内"就直接取 —— 帧外样本的值不参与输出。

use super::flags;
use me_core::{MediaError, Result};

use super::block::{BlockGrid, BLOCK_64X64, BLOCK_8X8, NUM_4X4_BLOCKS_WIDE};
use super::frame_header::FrameHeader;
use super::intra::CurrFrame;
use super::sequence_header::SequenceHeader;

/// `MI_SIZE_LOG2`：一个 4×4 块的像素边长是 `1 << 2`。
const MI_SIZE_LOG2: u32 = 2;
/// CDEF 的滤波单位是 8×8 块 ⇒ 在 4×4 MI 坐标里步长是 2。
const CDEF_STEP_MI: i32 = 2;

/// `Cdef_Pri_Taps[2][2]`（规范 §7.15.3）。
const CDEF_PRI_TAPS: [[i32; 2]; 2] = [[4, 2], [3, 3]];
/// `Cdef_Sec_Taps[2][2]`（规范 §7.15.3）。
const CDEF_SEC_TAPS: [[i32; 2]; 2] = [[2, 1], [2, 1]];
/// `Cdef_Directions[8][2][2]`（规范 §7.15.3）。
const CDEF_DIRECTIONS: [[[i32; 2]; 2]; 8] = [
    [[-1, 1], [-2, 2]],
    [[0, 1], [-1, 2]],
    [[0, 1], [0, 2]],
    [[0, 1], [1, 2]],
    [[1, 1], [2, 2]],
    [[1, 0], [2, 1]],
    [[1, 0], [2, 0]],
    [[1, 0], [2, -1]],
];
/// `Cdef_Uv_Dir[subsampling_x][subsampling_y][8]`（规范 §7.15.1）。
const CDEF_UV_DIR: [[[usize; 8]; 2]; 2] = [
    [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 2, 2, 3, 4, 6, 0]],
    [[7, 0, 2, 4, 5, 6, 6, 6], [0, 1, 2, 3, 4, 5, 6, 7]],
];
/// `Div_Table[9]`（规范 §7.15.2）。
const DIV_TABLE: [i32; 9] = [0, 840, 420, 280, 210, 168, 140, 120, 105];

/// 组内块级统计（`ME_CDEF_STATS=1`）：`[访问, 全skip, 零强度, 真滤波]`。
/// 用来回答"取消整帧克隆后，补拷的量到底有多大"（2026-10-24）。
static CDEF_BLK: [std::sync::atomic::AtomicUsize; 4] = [
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
];

/// `CdefFrame = CDEF(CurrFrame)`（§7.15）。
///
/// 输入是**去块之后**的 `CurrFrame`；输出 `CdefFrame` 会继续走上采样与环路恢复。
pub fn apply(
    frame: &CurrFrame,
    fh: &FrameHeader,
    seq: &SequenceHeader,
    grid: &BlockGrid,
) -> Result<Option<CurrFrame>> {
    // `CodedLossless` 或 `enable_cdef = 0` ⇒ 规范定义的输出就是整份拷贝
    // 诊断隔离用：`AV1_NO_CDEF=1` 也走这条"不滤波"的路（与 libaom 的 `LERXU_NO_CDEF`
    // 成对使用，用来一次定位"差异是哪一级滤波器引入的"）。
    // ⚠️ 返回 `None`（**不是** `frame.clone()`）：规范里 `CdefFrame` 这一档
    // 逐像素等于 `CurrFrame`，调用方可以**零拷贝**复用同一份 —— 1080p 一帧
    // 6.2 MB 的 memmove 就是这么白花的（2026-10-04）。
    if !seq.enable_cdef || fh.params.coded_lossless || flags::no_cdef() {
        return Ok(None);
    }

    // 规范里 CdefFrame 就是"逐块拷贝 CurrFrame 之后就滤波"的产物：
    // 先整份拷贝（等价于规范里每块开头那两段拷贝），滤波时只改被选中的块。
    //
    // ⚠️ **2026-10-06 试过"只拷 padding + 对不滤波的块补块拷贝"来省掉这次 memcpy
    // —— 实测 +14.2%（0/28 轮更快），已回退。** 原因是 CDEF 的 scan 是**逐 8×8
    // 块**走的，而 `skip` 块（四个 4×4 全 skip）占比很高 ⇒ 每帧要付几十万次
    // "8 样本/行"的小拷贝（平均 16~128 字节，调用开销秒杀大块 memcpy）。
    // 也就是说：**在这台机器上，"6 MB 一次 memcpy"比"只拷不滤波的那部分"还便宜**
    // —— 除非把拷贝按行聚成长串，而那需要重排 `apply` 的扫描。详见
    // `PERF-HANDOFF.md` 二十四。
    //
    // ⚠️ **2026-10-24 又试了一次"按 64×64 组补拷"**（跳过的组整组拷、其余不拷）：
    // 仍然不对 —— `cdef_group_fast` 内部还有两条**块级**的"不滤波"出口
    // （四个 4×4 全 skip；参数组强弱都为 0），它们同样依赖整份克隆留下的原像素
    // （`ME_CDEF_FULL_CLONE` 的对拍在第 2 帧就抓到了）。要支持按组补拷就得回到
    // 块级拷贝，也就是上面那次失败的形状 ⇒ 不再尝试。
    let mi_rows = grid.rows() as i32;
    let mi_cols = grid.cols() as i32;
    let step4 = NUM_4X4_BLOCKS_WIDE[BLOCK_8X8] as i32;
    let cdef_size4 = NUM_4X4_BLOCKS_WIDE[BLOCK_64X64] as i32;
    let mask = !(cdef_size4 - 1);

    // 整帧一个被滤波的组都没有 ⇒ 规范输出就是 CurrFrame（**零拷贝**）。
    // 组级表只有 `ceil(mi_cols/16) × ceil(mi_rows/16)` 项（1080p 约 510 项），
    // 这次扫描比 6.9 MB 的克隆便宜得多；静止/低强度段落里这种帧确实存在。
    let mut any_filtered = false;
    'groups: for r in (0..mi_rows).step_by(cdef_size4 as usize) {
        for c in (0..mi_cols).step_by(cdef_size4 as usize) {
            let slot = ((r & mask) as usize) * mi_cols as usize + ((c & mask) as usize);
            if grid.cdef_idx.get(slot).copied().unwrap_or(-1) >= 0 {
                any_filtered = true;
                break 'groups;
            }
        }
    }
    if !any_filtered {
        return Ok(None);
    }

    // ⚠️ **2026-10-24 又试了一次"不克隆 + 按组/块补拷"**（目标缓冲 `CurrFrame::new`
    // 现开，被跳过的组按 64×64 行拷、恒等块按 8×8 拷）：**+1.3%（4/10 轮）**。
    // 克隆不只是"留原像素"：它把目标页**预热**了，滤镜随后写同一批缓存行才不会
    // 缺页/写分配；省下的 6.25 MB 拷贝被这笔账吃回去还倒贴。⇒ 保持整帧克隆。
    let t_clone = std::time::Instant::now();
    let mut cdef = if flags::no_cdef_visible_clone() {
        frame.clone()
    } else {
        frame.clone_visible()
    };
    let clone_ms = t_clone.elapsed().as_secs_f64() * 1e3;
    let t_all = std::time::Instant::now();

    // `ME_CDEF_STATS=1`：组跳过率（诊断用，默认不扫）。
    let (mut stat_groups, mut stat_skipped) = (0usize, 0usize);
    if flags::cdef_stats() {
        for r in (0..mi_rows).step_by(cdef_size4 as usize) {
            for c in (0..mi_cols).step_by(cdef_size4 as usize) {
                let slot = ((r & mask) as usize) * mi_cols as usize
                    + ((c & mask) as usize);
                stat_groups += 1;
                if grid.cdef_idx.get(slot).copied().unwrap_or(-1) < 0 {
                    stat_skipped += 1;
                }
            }
        }
    }

    // 每个 64×64 组只读 `CurrFrame`、只写 `CdefFrame` —— 两个缓冲不同，
    // 所以**按超块行切分**就是安全的并行：第 k 段的写区域与第 k+1 段不重叠。
    let sb_rows: Vec<i32> = (0..mi_rows).step_by(step4 as usize).collect();
    let workers = super::parallel::worker_count().min(sb_rows.len().max(1)).max(1);
    let per = sb_rows.len().div_ceil(workers).max(1);
    let sub_y = i32::from(seq.color.subsampling_y);
    // ranges[k][p] = 第 k 个 worker 在第 p 个平面的**结束像素行**
    let mut ranges: Vec<Vec<usize>> = Vec::new();
    let mut chunks: Vec<&[i32]> = Vec::new();
    let mut a = 0usize;
    while a < sb_rows.len() {
        let b = (a + per).min(sb_rows.len());
        let last = sb_rows[b - 1] + step4; // 该段最后一行的超块**整行**上界
        let mut v = Vec::with_capacity(cdef.planes.len());
        for (p, plane) in cdef.planes.iter().enumerate() {
            let sy = if p == 0 { 0 } else { sub_y };
            let end = ((last << MI_SIZE_LOG2) >> sy).max(0) as usize;
            let _ = plane;
            v.push(end);
        }
        ranges.push(v);
        chunks.push(&sb_rows[a..b]);
        a = b;
    }
    let sinks = super::parallel::split_frame_rows(&mut cdef.planes, &ranges);

    // 组级快路径的帧常量：**内点 64×64 组**（整组 + 2 样本边距都在帧内）的所有
    // 8×8 块都不可能碰到越界候选 ⇒ 不必逐块做边界判定、不必补算边界条带。
    // 1080p 上 510 个组里约 464 个走快路径。
    let cdef_groups = 1usize << fh.params.cdef.bits;
    let step = step4 as usize;

    let pool: Vec<_> = chunks
        .into_iter()
        .enumerate()
        .map(|(k, rows)| {
            let sink = &sinks[k];
            move || -> Result<()> {
                if flags::no_cdef_group() {
                    for &r in rows {
                        for c in (0..mi_cols).step_by(step) {
                            let slot = ((r & mask) as usize) * mi_cols as usize
                                + ((c & mask) as usize);
                            let idx = grid.cdef_idx.get(slot).copied().unwrap_or(-1);
                            cdef_block(frame, sink, fh, seq, grid, r, c, idx)?;
                        }
                    }
                    return Ok(());
                }
                let chunk_end = rows.last().map(|r| r + step4).unwrap_or(0);
                let mut i = 0usize;
                while i < rows.len() {
                    let r = rows[i];
                    // 组行完整，或这是**帧底那半行**（本 worker 一直管到帧底）⇒ 都能
                    // 按组批处理；后者只跑组内还在帧里的那几块。
                    let row_group = (r & mask) == r
                        && r < mi_rows
                        && (r + cdef_size4 <= mi_rows || chunk_end >= mi_rows)
                        && (r + cdef_size4 <= chunk_end || chunk_end >= mi_rows);
                    if row_group {
                        // 这一行是一个完整组行的**第一行**：一次处理组内 8 个块行
                        // （下面的 `while` 把 `i` 直接推过它们，绝不能再逐行滤一遍）。
                        let mut c = 0i32;
                        while c + cdef_size4 <= mi_cols {
                            let slot = r as usize * mi_cols as usize + c as usize;
                            let idx = grid.cdef_idx.get(slot).copied().unwrap_or(-1);
                            // `cdef_idx` 是**组级**参数：`-1` 表示整组不滤 —— 一次跳过 64 块。
                            if idx < 0 {
                                c += cdef_size4;
                                continue;
                            }
                            if super::simd::avx2_available() {
                                cdef_group_fast(
                                    frame, sink, seq, grid, fh, r, c, idx as usize,
                                    cdef_groups,
                                )?;
                            } else {
                                for br in (r..r + cdef_size4).step_by(step) {
                                    for bc in (c..c + cdef_size4).step_by(step) {
                                        cdef_block(frame, sink, fh, seq, grid, br, bc, idx)?;
                                    }
                                }
                            }
                            c += cdef_size4;
                        }
                        // 帧宽不是 64 的整数倍时，右边缘还有一列**不完整的组**：
                        // 只能逐块做（`mi_cols` 是 4×4 数，这里按 8×8 块步进）。
                        while c < mi_cols {
                            let slot = r as usize * mi_cols as usize
                                + (c & mask) as usize;
                            let idx = grid.cdef_idx.get(slot).copied().unwrap_or(-1);
                            for br in (r..r + cdef_size4).step_by(step) {
                                cdef_block(frame, sink, fh, seq, grid, br, c, idx)?;
                            }
                            c += step4;
                        }
                        let next = r + cdef_size4;
                        while i < rows.len() && rows[i] < next {
                            i += 1;
                        }
                        continue;
                    }
                    // 非组对齐的行（只会出现在 worker 分段的开头/帧底）⇒ 逐块。
                    for c in (0..mi_cols).step_by(step) {
                        let slot = ((r & mask) as usize) * mi_cols as usize
                            + ((c & mask) as usize);
                        let idx = grid.cdef_idx.get(slot).copied().unwrap_or(-1);
                        cdef_block(frame, sink, fh, seq, grid, r, c, idx)?;
                    }
                    i += 1;
                }
                Ok(())
            }
        })
        .collect();

    if pool.len() <= 1 {
        for f in pool {
            f()?;
        }
    } else {
        let mut first_err: Option<MediaError> = None;
        std::thread::scope(|scope| {
            let handles: Vec<_> = pool.into_iter().map(|f| scope.spawn(f)).collect();
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
                            first_err = Some(MediaError::Bitstream(
                                "CDEF 并行的某个线程 panic 了".into(),
                            ));
                        }
                    }
                }
            }
        });
        if let Some(e) = first_err {
            return Err(e);
        }
    }
    if flags::cdef_stats() {
        eprintln!(
            "CDEF clone={clone_ms:.2}ms total={:.2}ms groups={stat_groups} skipped={stat_skipped} blk={}/{}/{}/{}",
            t_all.elapsed().as_secs_f64() * 1e3,
            CDEF_BLK[0].swap(0, std::sync::atomic::Ordering::Relaxed),
            CDEF_BLK[1].swap(0, std::sync::atomic::Ordering::Relaxed),
            CDEF_BLK[2].swap(0, std::sync::atomic::Ordering::Relaxed),
            CDEF_BLK[3].swap(0, std::sync::atomic::Ordering::Relaxed),
        );
    }
    Ok(Some(cdef))
}

/// `cdef_block( r, c, idx )`（规范 §7.15.1）。
#[allow(clippy::too_many_arguments)]
fn cdef_block(
    curr: &CurrFrame,
    sinks: &[super::parallel::RowsSink],
    fh: &FrameHeader,
    seq: &SequenceHeader,
    grid: &BlockGrid,
    r: i32,
    c: i32,
    idx: i32,
) -> Result<()> {
    // `idx == -1` ⇒ 不滤波（拷贝已经在 `apply` 里做过了）
    if idx < 0 {
        return Ok(());
    }
    let params = &fh.params.cdef;
    let groups = 1usize << params.bits;
    let idx = idx as usize;
    if idx >= groups {
        return Err(MediaError::Bitstream(format!(
            "AV1 CDEF 的参数组号越界：{idx}（这一帧只声明了 {groups} 组）"
        )));
    }

    let coeff_shift = u32::from(seq.color.bit_depth) - 8;
    let mi_cols = grid.cols() as i32;
    let mi_rows = grid.rows() as i32;

    // `skip = Skips[r][c] && Skips[r+1][c] && Skips[r][c+1] && Skips[r+1][c+1]`
    // 四个 4×4 全 skip 才跳过（`Skips` 是 4×4 单位）
    let skips_at = |rr: i32, cc: i32| -> bool {
        if rr < 0 || cc < 0 || rr >= mi_rows || cc >= mi_cols {
            return false;
        }
        grid.skips
            .get(rr as usize * mi_cols as usize + cc as usize)
            .is_some_and(|v| *v != 0)
    };
    let skip = skips_at(r, c)
        && skips_at(r + 1, c)
        && skips_at(r, c + 1)
        && skips_at(r + 1, c + 1);
    if skip {
        return Ok(());
    }

    let uv_pri_early = i32::from(params.uv_pri_strength[idx]) << coeff_shift;
    // §7.15.2：方向与方差都从**亮度**样本里量。但只有"亮度主抽头非零"或
    // "色度主抽头非零"时才真的用得到（`dir` 在 pri==0 时恒取 0、`var` 只喂
    // `pri_str`）—— 其余情况跳过这次 8 方向搜索。
    let (y_dir, var) = if flags::no_cdef_dirgate()
        || i32::from(params.y_pri_strength[idx]) != 0
        || uv_pri_early != 0
    {
        cdef_direction(curr, r, c)
    } else {
        (0, 0)
    };

    let mut pri_str = i32::from(params.y_pri_strength[idx]) << coeff_shift;
    let sec_str = i32::from(params.y_sec_strength[idx]) << coeff_shift;
    let dir = if pri_str == 0 { 0usize } else { y_dir };
    let var_shifted = var >> 6;
    let var_str = if var_shifted != 0 {
        floor_log2(var_shifted).min(12)
    } else {
        0
    };
    pri_str = if var != 0 { (pri_str * (4 + var_str) + 8) >> 4 } else { 0 };
    let damping = i32::from(params.damping) + coeff_shift as i32;

    cdef_filter(
        curr, sinks, seq, grid, 0, r, c, pri_str, sec_str, damping, dir,
    );

    if curr.num_planes == 1 {
        return Ok(());
    }

    // 色度：方向要经 `Cdef_Uv_Dir` 重映射，`damping` 比亮度小 1
    let uv_pri = i32::from(params.uv_pri_strength[idx]) << coeff_shift;
    let uv_sec = i32::from(params.uv_sec_strength[idx]) << coeff_shift;
    let sub_x = usize::from(seq.color.subsampling_x);
    let sub_y = usize::from(seq.color.subsampling_y);
    // 色度**不做** `var` 修正（规范 §7.15.1 只给亮度算了 `(priStr * (4 + varStr) + 8) >> 4`）
    let uv_dir = if uv_pri == 0 {
        0usize
    } else {
        CDEF_UV_DIR[sub_x][sub_y][y_dir]
    };
    let uv_damping = i32::from(params.damping) + coeff_shift as i32 - 1;
    cdef_filter(
        curr, sinks, seq, grid, 1, r, c, uv_pri, uv_sec, uv_damping, uv_dir,
    );
    cdef_filter(
        curr, sinks, seq, grid, 2, r, c, uv_pri, uv_sec, uv_damping, uv_dir,
    );
    Ok(())
}

/// 一个 **内点 64×64 组**的 CDEF（`r0`/`c0` 是组的 MI 原点，都是 16 的倍数）。
///
/// 与逐块路径的差别**只在簿记**：帧常量、组级强度、平面基址/步长/写指针各取一次；
/// 8×8 块不再做 `interior`/`coded_w`/`x_border` 这一套判定，也不补算边界条带。
/// 像素运算与 [`cdef_filter`] 完全同式（同一个 [`cdef_direction`]、同一对标量边界
/// 内核），所以输出逐字节相同 —— 「全内点」是这里省掉那些检查的**前提**，
/// 由调用方用 66 样本的边距判过。
#[allow(clippy::too_many_arguments)]
fn cdef_group_fast(
    curr: &CurrFrame,
    sinks: &[super::parallel::RowsSink],
    seq: &SequenceHeader,
    grid: &BlockGrid,
    fh: &FrameHeader,
    r0: i32,
    c0: i32,
    idx: usize,
    groups: usize,
) -> Result<()> {
    if idx >= groups {
        return Err(MediaError::Bitstream(format!(
            "AV1 CDEF 的参数组号越界：{idx}（这一帧只声明了 {groups} 组）"
        )));
    }
    let params = &fh.params.cdef;
    let coeff_shift = u32::from(seq.color.bit_depth) - 8;
    let damping = i32::from(params.damping) + coeff_shift as i32;
    let y_pri_base = i32::from(params.y_pri_strength[idx]) << coeff_shift;
    let y_sec = i32::from(params.y_sec_strength[idx]) << coeff_shift;
    let uv_pri = i32::from(params.uv_pri_strength[idx]) << coeff_shift;
    let uv_sec = i32::from(params.uv_sec_strength[idx]) << coeff_shift;
    let sub_x = u32::from(seq.color.subsampling_x);
    let sub_y = u32::from(seq.color.subsampling_y);
    let mi_cols = grid.cols() as i32;
    let mi_rows = grid.rows() as i32;
    let skips = &grid.skips;

    let luma = &curr.planes[0];
    let data_y = luma.px16();
    let stride_y = luma.stride as isize;
    let (dst_y, dst_stride_y, _, _) = sinks[0].direct();

    let chroma = curr.num_planes > 1;
    let (data_u, data_v, dst_u, dst_v, stride_uv) = if chroma {
        let p1 = &curr.planes[1];
        let p2 = &curr.planes[2];
        let (du, ds, _, _) = sinks[1].direct();
        let (dv, ds2, _, _) = sinks[2].direct();
        debug_assert_eq!(ds, ds2, "两个色度平面的步长应当一致");
        (Some(p1.px16()), Some(p2.px16()), du, dv, p1.stride as isize)
    } else {
        (None, None, std::ptr::null_mut(), std::ptr::null_mut(), 0)
    };
    let uv_dir_table =
        &CDEF_UV_DIR[usize::from(seq.color.subsampling_x)][usize::from(seq.color.subsampling_y)];
    let uv_damping = damping - 1;
    let cw = 8usize >> sub_x;
    let ch = 8usize >> sub_y;
    let need_uv = chroma && (uv_pri != 0 || uv_sec != 0);

    // 候选偏移/抽头按 (方向, 主抽头档) **跨块缓存**（2026-10-24）：组内相邻块的
    // 方向经常相同（平坦/同纹理区域），而 `cdef_candidates` 每次要写 12 个
    // `isize` + 12 个 `i32`。缓存命中时整段省掉，且两个数组留在寄存器/栈热处。
    let cache_cands = !flags::no_cdef_cand_cache();
    let mut y_offs = [0isize; super::simd::CDEF_CANDS];
    let mut y_taps = [0i32; super::simd::CDEF_CANDS];
    let mut y_key = usize::MAX;
    let mut uv_offs = [0isize; super::simd::CDEF_CANDS];
    let mut uv_taps = [0i32; super::simd::CDEF_CANDS];
    let mut uv_key = usize::MAX;

    // 帧底那一组可能不满 8 个块行；组内其它方向一定完整（调用方已判）。
    let block_rows = (((mi_rows - r0) as usize).div_ceil(CDEF_STEP_MI as usize)).min(8);
    // 方向搜索**提前按块行做**：相邻两块的 8×8 塞进一个 YMM（每个 128 位半区一块，
    // `vpshufb` 本来就按半区独立），指令数减半。是否算方向是**组级常量**
    // （强度来自参数组），所以整行要么全算、要么全不跳。
    let need_dir = flags::no_cdef_dirgate() || y_pri_base != 0 || uv_pri != 0;
    let dir2_ok = !flags::no_cdef_dir2()
        && !crate::video::flags::no_cdef_dir_simd()
        && super::simd::avx2_available()
        && !luma.is_u8();
    let mut dir_row = [(0usize, 0i32); 8];
    for br in 0..block_rows {
        let r = r0 + br as i32 * CDEF_STEP_MI;
        if need_dir {
            let mut elig = [false; 8];
            for bc in 0..8usize {
                let c = c0 + bc as i32 * CDEF_STEP_MI;
                let sc = (r as usize) * mi_cols as usize + c as usize;
                let mi_cols_u = mi_cols as usize;
                let skip = skips[sc] != 0
                    && skips[sc + 1] != 0
                    && skips[sc + mi_cols_u] != 0
                    && skips[sc + mi_cols_u + 1] != 0;
                let bx0 = c << MI_SIZE_LOG2;
                let by0 = r << MI_SIZE_LOG2;
                let interior = bx0 >= 2
                    && by0 >= 2
                    && (bx0 + 9) >> MI_SIZE_LOG2 < mi_cols
                    && (by0 + 9) >> MI_SIZE_LOG2 < mi_rows;
                elig[bc] = !skip && interior;
            }
            let sh = u32::from(seq.color.bit_depth) - 8;
            let mut bc = 0usize;
            while bc < 8 {
                if !elig[bc] {
                    bc += 1;
                    continue;
                }
                if dir2_ok && bc + 1 < 8 && elig[bc + 1] {
                    let ca = c0 + bc as i32 * CDEF_STEP_MI;
                    let cb = c0 + (bc + 1) as i32 * CDEF_STEP_MI;
                    // SAFETY: 已判 AVX2 且非 u8；两块都是"内点"（上面刚判过），
                    // 8×8 样本连边距都在平面内。
                    let (d0, d1) = unsafe {
                        super::simd::cdef_direction_avx2_x2(
                            data_y,
                            stride_y as usize,
                            (ca << MI_SIZE_LOG2) as usize,
                            (r << MI_SIZE_LOG2) as usize,
                            (cb << MI_SIZE_LOG2) as usize,
                            (r << MI_SIZE_LOG2) as usize,
                            sh,
                        )
                    };
                    dir_row[bc] = d0;
                    dir_row[bc + 1] = d1;
                    bc += 2;
                } else {
                    let c = c0 + bc as i32 * CDEF_STEP_MI;
                    dir_row[bc] = cdef_direction(curr, r, c);
                    bc += 1;
                }
            }
        }
        for bc in 0..8 {
            let c = c0 + bc * CDEF_STEP_MI;
            let sc = (r as usize) * mi_cols as usize + c as usize;
            // 四个 4×4 全 skip 才跳过（`Skips` 以 4×4 为单位；块内两个方向各 2 个）
            let mi_cols_u = mi_cols as usize;
            let skip = skips[sc] != 0
                && skips[sc + 1] != 0
                && skips[sc + mi_cols_u] != 0
                && skips[sc + mi_cols_u + 1] != 0;
            if skip {
                if flags::cdef_stats() {
                    use std::sync::atomic::Ordering::Relaxed;
                    CDEF_BLK[0].fetch_add(1, Relaxed);
                    CDEF_BLK[1].fetch_add(1, Relaxed);
                }
                continue;
            }
            // 组里绝大多数块是内点；**真正贴帧边的只有最后一两块**。
            // 判据与 [`cdef_filter`] 的 `interior` 同式（块是 8 对齐的，
            // 亮度内点蕴含色度内点：`(x+9)>>2 == ((x>>1)+5)<<1>>2`）。
            let bx0 = c << MI_SIZE_LOG2;
            let by0 = r << MI_SIZE_LOG2;
            let interior = bx0 >= 2
                && by0 >= 2
                && (bx0 + 9) >> MI_SIZE_LOG2 < mi_cols
                && (by0 + 9) >> MI_SIZE_LOG2 < mi_rows;
            if !interior {
                cdef_block(curr, sinks, fh, seq, grid, r, c, idx as i32)?;
                continue;
            }
            // 两个平面的强度都为 0 ⇒ 滤波是恒等变换（`acc = 0`、`Clip3(min,max,cur) = cur`），
            // 连方向都不必算。参数组里这种情况很常见（尤其是静止段落）。
            let need_y = y_pri_base != 0 || y_sec != 0;
            if !need_y && !need_uv {
                if flags::cdef_stats() {
                    use std::sync::atomic::Ordering::Relaxed;
                    CDEF_BLK[0].fetch_add(1, Relaxed);
                    CDEF_BLK[2].fetch_add(1, Relaxed);
                }
                continue;
            }
            if flags::cdef_stats() {
                use std::sync::atomic::Ordering::Relaxed;
                CDEF_BLK[0].fetch_add(1, Relaxed);
                CDEF_BLK[3].fetch_add(1, Relaxed);
            }
            let (y_dir, var) = dir_row[bc as usize];

            if need_y {
                let dir = if y_pri_base == 0 { 0 } else { y_dir };
                let var_shifted = var >> 6;
                let var_str = if var_shifted != 0 {
                    floor_log2(var_shifted).min(12)
                } else {
                    0
                };
                let pri_str = if var != 0 {
                    (y_pri_base * (4 + var_str) + 8) >> 4
                } else {
                    0
                };
                if pri_str != 0 || y_sec != 0 {
                    let tap_sel = ((pri_str >> coeff_shift) & 1) as usize;
                    let key = dir + 8 * tap_sel;
                    if !cache_cands || key != y_key {
                        cdef_candidates(
                            dir,
                            stride_y,
                            &CDEF_PRI_TAPS[tap_sel],
                            &CDEF_SEC_TAPS[tap_sel],
                            &mut y_offs,
                            &mut y_taps,
                        );
                        y_key = key;
                    }
                    let pri_adj = if pri_str == 0 {
                        0
                    } else {
                        (damping - floor_log2(pri_str)).max(0)
                    } as u32;
                    let sec_adj = if y_sec == 0 {
                        0
                    } else {
                        (damping - floor_log2(y_sec)).max(0)
                    } as u32;
                    // SAFETY: 调用方保证整组 + 2 样本边距都在可见/已重建区域内，
                    // 且这块的 8 行都属于本 worker 的 `RowsSink`。
                    unsafe {
                        if flags::no_cdef_4row() {
                            // 两行一组的版本：状态少一半（不溢出寄存器）。
                            super::simd::cdef_pair16(
                                data_y,
                                stride_y,
                                (c << MI_SIZE_LOG2) as isize,
                                (r << MI_SIZE_LOG2) as isize,
                                8,
                                8,
                                &y_offs,
                                &y_taps,
                                pri_str,
                                pri_adj,
                                y_sec,
                                sec_adj,
                                dst_y,
                                dst_stride_y,
                            );
                        } else {
                        super::simd::cdef_pair16x4(
                            data_y,
                            stride_y,
                            (c << MI_SIZE_LOG2) as isize,
                            (r << MI_SIZE_LOG2) as isize,
                            8,
                            8,
                            &y_offs,
                            &y_taps,
                            pri_str,
                            pri_adj,
                            y_sec,
                            sec_adj,
                            dst_y,
                            dst_stride_y,
                        );
                        }
                    }
                }
            }

            if need_uv {
                let uv_dir = if uv_pri == 0 {
                    0usize
                } else {
                    uv_dir_table[y_dir]
                };
                let uv_tap = ((uv_pri >> coeff_shift) & 1) as usize;
                let uv_key_now = uv_dir + 8 * uv_tap;
                if !cache_cands || uv_key_now != uv_key {
                    cdef_candidates(
                        uv_dir,
                        stride_uv,
                        &CDEF_PRI_TAPS[uv_tap],
                        &CDEF_SEC_TAPS[uv_tap],
                        &mut uv_offs,
                        &mut uv_taps,
                    );
                    uv_key = uv_key_now;
                }
                let pri_adj = if uv_pri == 0 {
                    0
                } else {
                    (uv_damping - floor_log2(uv_pri)).max(0)
                } as u32;
                let sec_adj = if uv_sec == 0 {
                    0
                } else {
                    (uv_damping - floor_log2(uv_sec)).max(0)
                } as u32;
                let x0 = ((c << MI_SIZE_LOG2) >> sub_x) as isize;
                let y0 = ((r << MI_SIZE_LOG2) >> sub_y) as isize;
                let du = data_u.unwrap();
                let dv = data_v.unwrap();
                // SAFETY: 同亮度路径；色度坐标是亮度坐标右移得到，边距只会更宽。
                unsafe {
                    if cw == 4 && ch == 4 {
                        super::simd::cdef_quad8_uv(
                            du, dv, stride_uv, x0, y0, &uv_offs, &uv_taps, uv_pri, pri_adj,
                            uv_sec, sec_adj, dst_u, dst_v, stride_uv,
                        );
                    } else {
                        super::simd::cdef_pair16x4(
                            du, stride_uv, x0, y0, ch, cw, &uv_offs, &uv_taps, uv_pri, pri_adj,
                            uv_sec, sec_adj, dst_u, stride_uv,
                        );
                        super::simd::cdef_pair16x4(
                            dv, stride_uv, x0, y0, ch, cw, &uv_offs, &uv_taps, uv_pri, pri_adj,
                            uv_sec, sec_adj, dst_v, stride_uv,
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

/// 把 12 个候选的**线性偏移**（`dy·stride + dx`）与抽头排成内核要的顺序：
/// 前 4 个是主抽头、后 8 个是次抽头（两个次方向各 2 个抽头 × 2 个符号）。
#[inline]
#[inline(always)]
fn cdef_candidates(
    dir: usize,
    stride: isize,
    pri_taps: &[i32; 2],
    sec_taps: &[i32; 2],
    offs: &mut [isize; super::simd::CDEF_CANDS],
    taps: &mut [i32; super::simd::CDEF_CANDS],
) {
    let (dxs, dys) = cdef_dirs(dir);
    for c in 0..super::simd::CDEF_CANDS {
        offs[c] = dys[c] as isize * stride + dxs[c] as isize;
        taps[c] = if c < 4 {
            pri_taps[c / 2]
        } else {
            sec_taps[(c - 4) / 4]
        };
    }
}

/// 12 个候选的 `(dx, dy)`（相对当前像素），顺序与 [`cdef_candidates`] 一致。
///
/// 2026-10-24：改成**编译期表**。原来每个块都要跑一遍"2×2 主抽头 + 4 组次抽头、
/// 带符号与方向旋转"的小循环（30 多条指令 + 分支）——1080p 一帧几万次。
const CDEF_DIRS_TAB: [[[i8; 2]; super::simd::CDEF_CANDS]; 8] = {
    let mut tab = [[[0i8; 2]; super::simd::CDEF_CANDS]; 8];
    let mut dir = 0usize;
    while dir < 8 {
        let mut n = 0usize;
        let mut k = 0usize;
        while k < 2 {
            let [dy, dx] = CDEF_DIRECTIONS[dir][k];
            // 先负后正（与旧循环的 `for sign in [-1, 1]` 一致）
            tab[dir][n] = [-dy as i8, -dx as i8];
            tab[dir][n + 1] = [dy as i8, dx as i8];
            n += 2;
            k += 1;
        }
        let mut k = 0usize;
        while k < 2 {
            let mut o = 0usize;
            while o < 2 {
                let d = (dir + if o == 0 { 6 } else { 2 }) & 7; // dir_off = -2 / +2
                let [dy, dx] = CDEF_DIRECTIONS[d][k];
                tab[dir][n] = [-dy as i8, -dx as i8];
                tab[dir][n + 1] = [dy as i8, dx as i8];
                n += 2;
                o += 1;
            }
            k += 1;
        }
        assert!(n == super::simd::CDEF_CANDS);
        dir += 1;
    }
    tab
};

#[inline]
fn cdef_dirs(dir: usize) -> ([i8; super::simd::CDEF_CANDS], [i8; super::simd::CDEF_CANDS]) {
    if flags::no_cdef_dirtab() {
        let mut dxs = [0i8; super::simd::CDEF_CANDS];
        let mut dys = [0i8; super::simd::CDEF_CANDS];
        let mut n = 0usize;
        for k in 0..2usize {
            for sign in [-1i32, 1] {
                let [dy, dx] = CDEF_DIRECTIONS[dir][k];
                dys[n] = (sign * dy) as i8;
                dxs[n] = (sign * dx) as i8;
                n += 1;
            }
        }
        for k in 0..2usize {
            for dir_off in [-2i32, 2] {
                let d = ((dir as i32 + dir_off) & 7) as usize;
                let [dy, dx] = CDEF_DIRECTIONS[d][k];
                for sign in [-1i32, 1] {
                    dys[n] = (sign * dy) as i8;
                    dxs[n] = (sign * dx) as i8;
                    n += 1;
                }
            }
        }
        return (dxs, dys);
    }
    let row = &CDEF_DIRS_TAB[dir];
    let mut dxs = [0i8; super::simd::CDEF_CANDS];
    let mut dys = [0i8; super::simd::CDEF_CANDS];
    for c in 0..super::simd::CDEF_CANDS {
        dys[c] = row[c][0];
        dxs[c] = row[c][1];
    }
    (dxs, dys)
}

/// `CDEF direction process`（规范 §7.15.2）：返回 `(yDir, var)`。
#[inline(always)]
fn cdef_direction(curr: &CurrFrame, r: i32, c: i32) -> (usize, i32) {
    let plane = &curr.planes[0];
    let bit_depth = plane.bit_depth;
    let x0 = (c << MI_SIZE_LOG2) as usize;
    let y0 = (r << MI_SIZE_LOG2) as usize;
    let sh = u32::from(bit_depth - 8);
    // AVX2 版把 partial 全程留在寄存器里（见 `simd::cdef_direction_avx2` 的推导）。
    // CDEF 只吃 u16 平面（u8 存储的 CDEF 已回退）⇒ u8 时走标量。
    #[cfg(target_arch = "x86_64")]
    if !flags::no_cdef_dir_simd() && super::simd::avx2_available() && !plane.is_u8() {
        // SAFETY: 已判 AVX2；8×8 块都在平面内（`PLANE_PAD` 覆盖右/下边距）。
        return unsafe {
            super::simd::cdef_direction_avx2(plane.px16(), plane.stride, x0, y0, sh)
        };
    }
    let mut partial = [[0i32; 16]; 8];
    dir_accum_scalar(&mut partial, plane, x0, y0, sh);
    dir_cost(&partial)
}

/// 由 8 个方向的 partial 求 `(yDir, var)`（规范 §7.15.2 后半）。
fn dir_cost(partial: &[[i32; 16]; 8]) -> (usize, i32) {
    let mut cost = [0i32; 8];
    for i in 0..8 {
        cost[2] += partial[2][i] * partial[2][i];
        cost[6] += partial[6][i] * partial[6][i];
    }
    cost[2] *= DIV_TABLE[8];
    cost[6] *= DIV_TABLE[8];
    for i in 0..7 {
        cost[0] += (partial[0][i] * partial[0][i] + partial[0][14 - i] * partial[0][14 - i])
            * DIV_TABLE[i + 1];
        cost[4] += (partial[4][i] * partial[4][i] + partial[4][14 - i] * partial[4][14 - i])
            * DIV_TABLE[i + 1];
    }
    cost[0] += partial[0][7] * partial[0][7] * DIV_TABLE[8];
    cost[4] += partial[4][7] * partial[4][7] * DIV_TABLE[8];
    for i in (1..8).step_by(2) {
        for j in 0..=4 {
            cost[i] += partial[i][3 + j] * partial[i][3 + j];
        }
        cost[i] *= DIV_TABLE[8];
        for j in 0..3 {
            cost[i] += (partial[i][j] * partial[i][j] + partial[i][10 - j] * partial[i][10 - j])
                * DIV_TABLE[2 * j + 2];
        }
    }

    let mut best_cost = 0i32;
    let mut y_dir = 0usize;
    for i in 0..8 {
        if cost[i] > best_cost {
            best_cost = cost[i];
            y_dir = i;
        }
    }
    (y_dir, (best_cost - cost[(y_dir + 4) & 7]) >> 10)
}

/// `CDEF direction process` 的**累加段**（§7.15.2 前半）—— 标量参照实现。
///
/// 标量式是那段 `partial[0][i+j] += x[j] … partial[7][i/2+j] += x[j]` 的 8×8 双重
/// 循环（**512 次带散列下标的加**，1080p 每帧 3.24 万个 8×8 块都要做一遍）。
/// 关键观察：**每个方向都只是"把整行 `xs` 加到某个随 `i` 变化的偏移上"** ⇒
/// 一条 8 宽向量加就够了。
///
/// ## 逐方向推导（`xs[j] = 样本 - 128`，当前行 `i`）
///
/// | 方向 | 标量式 | 向量式 |
/// |---|---|---|
/// | 0 | `partial[0][i+j] += xs[j]` | `p0[i..i+8] += xs` |
/// | 1 | `partial[1][i+j/2] += xs[j]` | `p1[i..i+8] += y` |
/// | 2 | `partial[2][i] += xs[j]` | `p2[i] += Σ xs` |
/// | 3 | `partial[3][3+i-j/2] += xs[j]` | `p3[i..i+8] += rev(y)` |
/// | 4 | `partial[4][7+i-j] += xs[j]` | `p4[i..i+8] += rev(xs)` |
/// | 5 | `partial[5][3-i/2+j] += xs[j]` | `p5[(3-i/2)..+8] += xs` |
/// | 6 | `partial[6][j] += xs[j]` | `p6 += xs`（寄存器里跨行累加） |
/// | 7 | `partial[7][i/2+j] += xs[j]` | `p7[(i/2)..+8] += xs` |
///
/// `y[k] = xs[2k] + xs[2k+1]`（`k < 4`，高 4 通道是 **0**）。高位那 4 个 0 加进去
/// **不改动**任何格点 —— 它们是**别的行**的累加器，加 0 是恒等。`rev` 把 8 个通道
/// 反序（方向 4 要的正是 `x7…x0`）。
///
/// 整数加可结合，且每一格点上的加法次序与标量式**逐条相同** ⇒ `partial` 逐位相同
/// （`ME_NO_SIMD` 开/关的输出逐字节一致就是这条的验收）。
fn dir_accum_scalar(
    partial: &mut [[i32; 16]; 8],
    plane: &super::intra::Plane,
    x0: usize,
    y0: usize,
    sh: u32,
) {
    let stride = plane.stride;
    for i in 0..8usize {
        let base = (y0 + i) * stride + x0;
        let mut xs = [0i32; 8];
        if plane.is_u8() {
            let d = plane.px8();
            for (j, slot) in xs.iter_mut().enumerate() {
                *slot = (d[base + j] >> sh) as i32 - 128;
            }
        } else {
            let d = plane.px16();
            for (j, slot) in xs.iter_mut().enumerate() {
                *slot = (d[base + j] >> sh) as i32 - 128;
            }
        }
        for (j, &x) in xs.iter().enumerate() {
            partial[0][i + j] += x;
            partial[1][i + j / 2] += x;
            partial[2][i] += x;
            partial[3][3 + i - j / 2] += x;
            partial[4][7 + i - j] += x;
            partial[5][3 - i / 2 + j] += x;
            partial[6][j] += x;
            partial[7][i / 2 + j] += x;
        }
    }
}

/// `CDEF filter process`（规范 §7.15.3）。
///
/// **只读 `CurrFrame`、只写 `CdefFrame`** —— 这是规范里"滤波结果不回流到输入"
/// 的硬要求（否则同一次滤波里后面的样本会用到已经改过的邻居）。
#[allow(clippy::too_many_arguments)]
fn cdef_filter(
    curr: &CurrFrame,
    sinks: &[super::parallel::RowsSink],
    seq: &SequenceHeader,
    grid: &BlockGrid,
    plane: u8,
    r: i32,
    c: i32,
    pri_str: i32,
    sec_str: i32,
    damping: i32,
    dir: usize,
) {
    let sub_x = if plane > 0 { u32::from(seq.color.subsampling_x) } else { 0 };
    let sub_y = if plane > 0 { u32::from(seq.color.subsampling_y) } else { 0 };
    let x0 = (c << MI_SIZE_LOG2) >> sub_x;
    let y0 = (r << MI_SIZE_LOG2) >> sub_y;
    let w = 8usize >> sub_x;
    let h = 8usize >> sub_y;
    let coeff_shift = u32::from(seq.color.bit_depth) - 8;
    let mi_rows = grid.rows() as i32;
    let mi_cols = grid.cols() as i32;
    let tap_sel = ((pri_str >> coeff_shift) & 1) as usize;

    // `constrain()` 里的 `damping_adj = Max(0, damping - FloorLog2(threshold))` **只跟阈值有关**
    // ⇒ 每个块算一次。原实现把它放在 `constrain` 里，于是**每个候选样本**都要重算一遍
    // （一个像素 20 个候选 ⇒ 20 次 `FloorLog2`），CDEF 曾因此占到忙时 33%。
    let pri_adj = if pri_str == 0 { 0 } else { (damping - floor_log2(pri_str)).max(0) };
    let sec_adj = if sec_str == 0 { 0 } else { (damping - floor_log2(sec_str)).max(0) };
    let pri_taps = &CDEF_PRI_TAPS[tap_sel];
    let sec_taps = &CDEF_SEC_TAPS[tap_sel];

    let plane_ref = &curr.planes[plane as usize];
    let data = plane_ref.px16();
    let stride = plane_ref.stride;
    let sink = &sinks[plane as usize];

    // **候选表每块只算一次**：`(dx, dy)` 只跟方向有关，线性偏移再带上平面 stride。
    // 顺序约定：前 4 个主抽头、后 8 个次抽头（与 SIMD 内核一致）。
    let (dxs, dys) = cdef_dirs(dir);
    let mut offs = [0isize; super::simd::CDEF_CANDS];
    let mut taps = [0i32; super::simd::CDEF_CANDS];
    let mut thr = [0i32; super::simd::CDEF_CANDS];
    let mut adj = [0u32; super::simd::CDEF_CANDS];
    for c in 0..super::simd::CDEF_CANDS {
        offs[c] = dys[c] as isize * stride as isize + dxs[c] as isize;
        taps[c] = if c < 4 {
            pri_taps[c / 2]
        } else {
            sec_taps[(c - 4) / 4]
        };
        let (t, a) = if c < 4 {
            (pri_str, pri_adj)
        } else {
            (sec_str, sec_adj)
        };
        thr[c] = t;
        adj[c] = a as u32;
    }

    // **内点判定**：20 个候选最多外扩 2 个样本，把整块的极值折回 MI 网格判一次 ——
    // 内点就不必逐候选做那 4 个越界比较（外圈超块才走带检查的一般路径）。
    let y_lo_mi = ((y0 - 2).max(0) << sub_y) >> MI_SIZE_LOG2;
    let y_hi_mi = ((y0 + h as i32 + 1) << sub_y) >> MI_SIZE_LOG2;
    let x_lo_mi = ((x0 - 2).max(0) << sub_x) >> MI_SIZE_LOG2;
    let x_hi_mi = ((x0 + w as i32 + 1) << sub_x) >> MI_SIZE_LOG2;
    let interior =
        y0 >= 2 && x0 >= 2 && y_lo_mi >= 0 && x_lo_mi >= 0 && y_hi_mi < mi_rows && x_hi_mi < mi_cols;

    // **候选有效性 = 列掩码 & 行掩码**。规范只要求候选落在帧的 4×4 网格内；
    // 对每列/每行把 12 个候选的有效性算成 12 位掩码（每块一次），标量循环里
    // 不用再做坐标换算、也不用走 `Plane::get`（原来每个候选都要 4 次比较 +
    // 两次移位 + 一次 match）。内点块两个掩码全 1，同一段代码走无分支快路径。
    let x_lim = ((mi_cols << MI_SIZE_LOG2) >> sub_x) as i32;
    let y_lim = ((mi_rows << MI_SIZE_LOG2) >> sub_y) as i32;
    let all_bits = (1u16 << super::simd::CDEF_CANDS) - 1;
    let mut cmask = [all_bits; 8];
    let mut rmask = [all_bits; 8];
    if !interior {
        for (j, m) in cmask.iter_mut().enumerate().take(w) {
            let x = x0 + j as i32;
            let mut bits = 0u16;
            for (c, &dx) in dxs.iter().enumerate() {
                let xx = x + dx as i32;
                if xx >= 0 && xx < x_lim {
                    bits |= 1 << c;
                }
            }
            *m = bits;
        }
        for (i, m) in rmask.iter_mut().enumerate().take(h) {
            let y = y0 + i as i32;
            let mut bits = 0u16;
            for (c, &dy) in dys.iter().enumerate() {
                let yy = y + dy as i32;
                if yy >= 0 && yy < y_lim {
                    bits |= 1 << c;
                }
            }
            *m = bits;
        }
    }

    // **SIMD**：两行 × 8 列一次算完，i16 通道（见 `simd::cdef_pair16`）。
    //
    // 2026-10-22：**右/下边界块也走 SIMD**。内核读的是平面缓冲，而平面在右侧/底部
    // 有 72 样本的 padding，越界候选的读取是**可寻址**的；它们只会污染最右/最下
    // 1~2 行/列的**输出**，这些像素随后用标量重算（越界候选不参与）。
    // 左/上边界仍走标量：平面没有负向 padding。
    // 左/上边界没有负向 padding：把内核起点**右/下移 2 个样本**再跑 SIMD
    // （`x_eff >= 2` ⇒ 候选 `x_eff-2 >= 0` 可寻址），多写的 2 列/行由标量补算。
    // 只有帧边缘的第一块会有位移，写出去的行列仍在同一 64×64 组内（不会被别的
    // worker 撞到），而且这些像素随后都会被自己的块覆盖或按掩码重算。
    let shift = if super::flags::no_cdef_shift() { 0 } else { 2 };
    let x_eff = if x0 < shift { x0 + shift } else { x0 };
    let y_eff = if y0 < shift { y0 + shift } else { y0 };
    let cap_rows = (data.len() / stride) as i32;
    let simd = super::simd::avx2_available()
        && (interior || !super::flags::no_cdef_edge())
        && x_eff + 9 < stride as i32
        && y_eff + h as i32 + 1 < cap_rows
        && (x_eff >= 2 && y_eff >= 2);

    // 候选最大外扩 2 个样本 ⇒ 从这一列/行起输出可能受越界候选影响。
    let coded_w = (mi_cols << MI_SIZE_LOG2) >> sub_x;
    let coded_h = (mi_rows << MI_SIZE_LOG2) >> sub_y;
    let x_border = (coded_w - 2 - x0).clamp(0, w as i32) as usize;
    let y_border = (coded_h - 2 - y0).clamp(0, h as i32) as usize;

    // 标量逐像素：`masked = false` 只在"候选全在帧内"时用（整块标量回退）。
    let filter_pixels = |xl: usize, xh: usize, yl: usize, yh: usize, masked: bool| {
        for i in yl..yh {
            let y = y0 as usize + i;
            let row = y * stride;
            let rm = if masked { rmask[i] } else { all_bits };
            for j in xl..xh {
                let x = x0 as usize + j;
                let off = row + x;
                let mask = if masked { cmask[j] & rm } else { all_bits };
                let cur = i32::from(data[off]);
                let mut max = cur;
                let mut min = cur;
                let mut sum = 0i32;
                for c in 0..super::simd::CDEF_CANDS {
                    if mask & (1 << c) != 0 {
                        // SAFETY(逻辑)：掩码已经保证这个候选落在帧的 4×4 网格内，
                        // 平面缓冲（stride = w + 72）覆盖这些地址。
                        let p = i32::from(data[(off as isize + offs[c]) as usize]);
                        cdef_acc(
                            p, cur, thr[c], adj[c] as i32, taps[c], &mut sum, &mut max, &mut min,
                        );
                    }
                }
                let adjust = if sum < 0 { 1 } else { 0 };
                let v = clip3(min, max, cur + ((8 + sum - adjust) >> 4));
                sink.set(x, y, v);
            }
        }
    };

    if simd {
        // **就地写目标平面**：内核按 `(y0+r)*dst_stride + x0` 直接写 `CdefFrame`，
        // 省掉"内核→scratch→memcpy 进平面"的那次往返（1080p 一帧 78 万行 × 16 字节）。
        let (dptr, dstride, dy0, dy1) = sink.direct();
        debug_assert!(
            y_eff as usize >= dy0 && (y_eff as usize + h) <= dy1,
            "CDEF 的 SIMD 路径要写出自己那段行了"
        );
        // SAFETY: `simd` 已保证 AVX2 可用、`pri_adj`/`sec_adj` 远小于 16，
        // 12 个候选 × 8 通道的取址全部落在**源**平面缓冲内（上面的窗口判据）；
        // `dptr`/`dstride` 来自本 worker 的 `RowsSink`，写入范围由上面的断言守住。
        //
        // ⚠️ **4×4 的块走 `cdef_quad8`**（2026-10-05 第三十五轮）：`cdef_pair16` 一次
        // 算 2 行 × **8** 列，而 4:2:0 的色度块只有 **4** 列 ⇒ 一半的通道白算。
        // `cdef_quad8` 一次算 **4 行 × 4 列**（4 个 qword），16 个通道全部用满，
        // 迭代次数从 `h/2` 降到 `h/4`。取址要求**更弱**（每行只读 4 个 u16）。
        unsafe {
            if w == 4 && h == 4 {
                // SAFETY: 同上；`w == h == 4` 正是本分支的前提。
                super::simd::cdef_quad8(
                    data,
                    stride as isize,
                    x_eff as isize,
                    y_eff as isize,
                    &offs,
                    &taps,
                    pri_str,
                    pri_adj as u32,
                    sec_str,
                    sec_adj as u32,
                    dptr,
                    dstride,
                );
            } else {
                super::simd::cdef_pair16(
                    data,
                    stride as isize,
                    x_eff as isize,
                    y_eff as isize,
                    h,
                    w,
                    &offs,
                    &taps,
                    pri_str,
                    pri_adj as u32,
                    sec_str,
                    sec_adj as u32,
                    dptr,
                    dstride,
                );
            }
        }
        // 位移后内核写出的矩形是 `[x_eff, x_eff+w) × [y_eff, y_eff+h)`：落在原块
        // **外面**的那 2 列/行要还原成 `CurrFrame` 的值 —— 相邻块可能是 skip
        // （它自己不滤波、也就不会覆盖回来）。还原后行为与"没位移"逐字节相同。
        if x_eff > x0 || y_eff > y0 {
            for yy in y_eff..y_eff + h as i32 {
                for xx in x_eff..x_eff + w as i32 {
                    if xx < x0 || xx >= x0 + w as i32 || yy < y0 || yy >= y0 + h as i32 {
                        let v = i32::from(data[yy as usize * stride + xx as usize]);
                        sink.set(xx as usize, yy as usize, v);
                    }
                }
            }
        }
        // 边界条带用标量补算（右条带 + 下条带，二者不重叠）。
        if x_border < w {
            filter_pixels(x_border, w, 0, h, true);
        }
        if y_border < h {
            filter_pixels(0, x_border, y_border, h, true);
        }
        // 位移出来的左/上条带（与上面重叠也没关系：同一块从 `CurrFrame` 读、写同样的值）。
        if x_eff > x0 {
            filter_pixels(0, (x_eff - x0) as usize, 0, h, true);
        }
        if y_eff > y0 {
            filter_pixels(0, w, 0, (y_eff - y0) as usize, true);
        }
        return;
    }

    filter_pixels(0, w, 0, h, !interior);
}

/// 累加一个候选样本（已把 `constrain` 的两个参数算好）。
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn cdef_acc(
    p: i32,
    cur: i32,
    threshold: i32,
    damping_adj: i32,
    tap: i32,
    sum: &mut i32,
    max: &mut i32,
    min: &mut i32,
) {
    *sum += tap * constrain_adj(p - cur, threshold, damping_adj);
    if p > *max {
        *max = p;
    }
    if p < *min {
        *min = p;
    }
}

/// `constrain( diff, threshold, damping )` 的**参数已提好**版本（§7.15.3）。
#[inline(always)]
fn constrain_adj(diff: i32, threshold: i32, damping_adj: i32) -> i32 {
    if threshold == 0 {
        return 0;
    }
    let mag = diff.abs();
    let sign = if diff < 0 { -1 } else { 1 };
    sign * clip3(0, mag, threshold - (mag >> damping_adj))
}

/// `FloorLog2( x )`：`x >= 1`。
#[inline]
fn floor_log2(x: i32) -> i32 {
    debug_assert!(x >= 1);
    31 - x.leading_zeros() as i32
}

/// `Clip3( low, high, v )`。
#[inline]
fn clip3(low: i32, high: i32, v: i32) -> i32 {
    v.max(low).min(high)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_taps_stay_inside_the_two_sample_neighbourhood() {
        // CDEF 的次抽头刻意取到 ±2，但绝不该更远 —— 取到 ±3 会读到别的块
        for (dir, entry) in CDEF_DIRECTIONS.iter().enumerate() {
            for (k, tap) in entry.iter().enumerate() {
                for v in tap {
                    assert!(
                        (-2..=2).contains(v),
                        "方向 {dir} 的第 {k} 个抽头越出 ±2：{v}"
                    );
                }
            }
        }
        // 四个轴对齐方向（0/2/4/6）的次抽头是主抽头的 2 倍；1/3/5/7 是插值方向，
        // 不成倍数（表里直接给的是近似值）
        for d in [0usize, 2, 4, 6] {
            for t in 0..2 {
                assert_eq!(
                    CDEF_DIRECTIONS[d][1][t],
                    2 * CDEF_DIRECTIONS[d][0][t],
                    "方向 {d} 的次抽头应当是主抽头的 2 倍"
                );
            }
        }
    }

    /// 与 `constrain` 等价（把 `damping_adj` 现算出来）—— 测试用原来的口径描述不变量。
    fn constrain(diff: i32, threshold: i32, damping: i32) -> i32 {
        if threshold == 0 {
            return 0;
        }
        let adj = (damping - floor_log2(threshold)).max(0);
        constrain_adj(diff, threshold, adj)
    }

    #[test]
    fn constrain_is_antisymmetric_and_bounded() {
        for &threshold in &[4, 10, 63, 255] {
            for &damping in &[0, 3, 6] {
                for diff in -300i32..=300 {
                    let c = constrain(diff, threshold, damping);
                    assert_eq!(
                        constrain(-diff, threshold, damping),
                        -c,
                        "constrain 必须关于 diff 反对称（diff={diff}）"
                    );
                    assert!(
                        c.abs() <= diff.abs().min(threshold),
                        "constrain 的结果不能超过阈值、也不能把差值的符号放大：diff={diff} 得到 {c}"
                    );
                }
            }
        }
    }

    #[test]
    fn constrain_with_zero_threshold_contributes_nothing() {
        // `threshold == 0` 是"这一档不滤波"的表达，必须恒为 0（否则 CDEF 会偷偷改像素）
        for diff in [-100, -1, 0, 1, 100] {
            assert_eq!(constrain(diff, 0, 3), 0);
        }
    }

    #[test]
    fn floor_log2_matches_integer_log() {
        for x in 1..=1024i32 {
            // 最大的 i 使 2^i <= x（只到 30，避免 1 << 31 溢出成负数）
            let want = (0..31).rev().find(|i| (1i32 << i) <= x).unwrap();
            assert_eq!(floor_log2(x), want, "floor_log2({x})");
        }
    }

    #[test]
    fn div_table_is_decreasing() {
        // `Div_Table[i] ≈ 840 / i`：代价函数靠它把不同长度的边归一化
        for i in 1..9 {
            assert_eq!(DIV_TABLE[i], 840 / i as i32, "Div_Table[{i}]");
        }
        assert_eq!(DIV_TABLE[0], 0, "伪代码里 Div_Table[0] 从不用到，但约定为 0");
    }

    /// [`dir_accum_scalar`] 必须与"照规范逐条抄"的标量式**逐格相同**。
    ///
    /// 这是 2026-10-05 那次向量化的护栏：8 个方向全被重新表达成"把整行加到某个随
    /// `i` 变化的偏移上"，**写错一个偏移只会让 `y_dir` 整体偏**（画面出现方向性拖影），
    /// 从最终像素差很难定位。所以这里独立写第二遍标量式对拍 `partial`。
    /// **方向搜索的 AVX2 版必须与标量版给出同一个 `(yDir, var)`**。
    ///
    /// AVX2 版把 partial 留在寄存器里、代价用 `pmaddwd` 按**对称位置**成对求平方和，
    /// 分组次序与标量不同；整数无溢出时结果必然逐位相同 —— 这条就是那个前提的护栏。
    /// 覆盖 8/10/12bit × 随机/斜条纹/棋盘 × 5 个块位置。
    #[test]
    fn cdef_direction_avx2_matches_scalar_bit_for_bit() {
        #[cfg(not(target_arch = "x86_64"))]
        {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if !crate::video::simd::avx2_available() {
                return;
            }
            const W: usize = 40;
            const H: usize = 40;
            for &(bit_depth, sh) in &[(8u8, 0u32), (10, 2), (12, 4)] {
                for pattern in 0..3 {
                    let mut plane = crate::video::intra::Plane::new(W, H, bit_depth);
                    if plane.is_u8() {
                        return;
                    }
                    let stride = plane.stride;
                    let max = (1i32 << bit_depth) - 1;
                    {
                        let d = plane.px16_mut();
                        let mut seed = 0x1234_5678u32;
                        for y in 0..H {
                            for x in 0..W {
                                let v = match pattern {
                                    0 => {
                                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                                        ((seed >> 16) % (max as u32 + 1)) as i32
                                    }
                                    1 => ((x * 7 + y * 13) % 97) as i32 * max / 97,
                                    _ => {
                                        if (x / 4 + y / 4) % 2 == 0 {
                                            max
                                        } else {
                                            0
                                        }
                                    }
                                };
                                d[y * stride + x] = v as u16;
                            }
                        }
                    }
                    for &(x0, y0) in &[(0usize, 0usize), (8, 8), (16, 24), (24, 16), (31, 32)] {
                        let mut partial = [[0i32; 16]; 8];
                        super::dir_accum_scalar(&mut partial, &plane, x0, y0, sh);
                        let want = super::dir_cost(&partial);
                        // SAFETY: 已判 AVX2；块完全落在 40×40（stride 40+72）内。
                        let got = unsafe {
                            crate::video::simd::cdef_direction_avx2(plane.px16(), stride, x0, y0, sh)
                        };
                        assert_eq!(
                            got, want,
                            "pattern={pattern} bd={bit_depth} at ({x0},{y0})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cdef_direction_avx2_x2_matches_scalar_bit_for_bit() {
        #[cfg(target_arch = "x86_64")]
        {
            if !crate::video::simd::avx2_available() {
                return;
            }
            const W: usize = 48;
            const H: usize = 40;
            for &(bit_depth, sh) in &[(8u8, 0u32), (10, 2), (12, 4)] {
                for pattern in 0..3 {
                    let mut plane = crate::video::intra::Plane::new(W, H, bit_depth);
                    if plane.is_u8() {
                        return;
                    }
                    let stride = plane.stride;
                    let max = (1i32 << bit_depth) - 1;
                    {
                        let d = plane.px16_mut();
                        let mut seed = 0x9e37_79b9u32;
                        for y in 0..H {
                            for x in 0..W {
                                let v = match pattern {
                                    0 => {
                                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                                        ((seed >> 16) % (max as u32 + 1)) as i32
                                    }
                                    1 => ((x * 5 + y * 11) % 89) as i32 * max / 89,
                                    _ => {
                                        if (x / 4 + y / 4) % 2 == 0 { max } else { 0 }
                                    }
                                };
                                d[y * stride + x] = v as u16;
                            }
                        }
                    }
                    for &(xa, ya, xb, yb) in &[
                        (0usize, 0usize, 8usize, 0usize),
                        (8, 8, 16, 8),
                        (16, 24, 24, 16),
                        (24, 0, 31, 0),
                        (0, 32, 8, 32),
                        (32, 8, 40, 8),
                    ] {
                        let mut pa = [[0i32; 16]; 8];
                        super::dir_accum_scalar(&mut pa, &plane, xa, ya, sh);
                        let wa = super::dir_cost(&pa);
                        let mut pb = [[0i32; 16]; 8];
                        super::dir_accum_scalar(&mut pb, &plane, xb, yb, sh);
                        let wb = super::dir_cost(&pb);
                        // SAFETY: 已判 AVX2；两块 8×8 都在 48×40（带 PLANE_PAD）内。
                        let got = unsafe {
                            crate::video::simd::cdef_direction_avx2_x2(
                                plane.px16(),
                                stride,
                                xa,
                                ya,
                                xb,
                                yb,
                                sh,
                            )
                        };
                        assert_eq!(got, (wa, wb), "pattern={pattern} bd={bit_depth} a=({xa},{ya}) b=({xb},{yb})");
                    }
                }
            }
        }
    }

    #[test]
    fn cdef_dir_accum_matches_the_scalar_reference() {
        const W: usize = 40;
        const H: usize = 40;
        for &(bit_depth, sh) in &[(8u8, 0u32), (10, 2), (12, 4)] {
            let mut plane = crate::video::intra::Plane::new(W, H, bit_depth);
            let stride = plane.stride;
            let max = (1i32 << bit_depth) - 1;
            // 造出**有方向性**的内容（斜条纹 + 台阶），保证 8 个方向的代价互不相同
            if plane.is_u8() {
                let d = plane.px8_mut();
                for y in 0..H {
                    for x in 0..W {
                        d[y * stride + x] = (((x * 7 + y * 3) % 97) as i32 * max / 97) as u8;
                    }
                }
            } else {
                let d = plane.px16_mut();
                for y in 0..H {
                    for x in 0..W {
                        d[y * stride + x] = (((x * 7 + y * 3) % 97) as i32 * max / 97) as u16;
                    }
                }
            }
            // 独立取样本（与 `dir_accum_scalar` 走的不是同一段代码）
            let sample = |x: usize, y: usize| -> i32 {
                if plane.is_u8() {
                    plane.px8()[y * stride + x] as i32
                } else {
                    plane.px16()[y * stride + x] as i32
                }
            };
            for &(x0, y0) in &[(0usize, 0usize), (8, 8), (16, 24), (24, 16), (31, 32)] {
                let mut got = [[0i32; 16]; 8];
                super::dir_accum_scalar(&mut got, &plane, x0, y0, sh);
                let mut want = [[0i32; 16]; 8];
                for i in 0..8usize {
                    let mut xs = [0i32; 8];
                    for (j, slot) in xs.iter_mut().enumerate() {
                        *slot = (sample(x0 + j, y0 + i) >> sh) - 128;
                    }
                    for (j, &x) in xs.iter().enumerate() {
                        want[0][i + j] += x;
                        want[1][i + j / 2] += x;
                        want[2][i] += x;
                        want[3][3 + i - j / 2] += x;
                        want[4][7 + i - j] += x;
                        want[5][3 - i / 2 + j] += x;
                        want[6][j] += x;
                        want[7][i / 2 + j] += x;
                    }
                }
                for d in 0..8 {
                    assert_eq!(
                        got[d], want[d],
                        "方向 {d} 在 ({x0},{y0}) bit_depth={bit_depth} 对不上"
                    );
                }
            }
        }
    }
}
