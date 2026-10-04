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

    // 规范里 CdefFrame 就是"逐块拷贝 CurrFrame 之后再就地滤波"的产物：
    // 先整份拷贝（等价于规范里每块开头那两段拷贝），滤波时只改被选中的块。
    let mut cdef = frame.clone();

    let mi_rows = grid.rows() as i32;
    let mi_cols = grid.cols() as i32;
    let step4 = NUM_4X4_BLOCKS_WIDE[BLOCK_8X8] as i32;
    let cdef_size4 = NUM_4X4_BLOCKS_WIDE[BLOCK_64X64] as i32;
    let mask = !(cdef_size4 - 1);

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

    let pool: Vec<_> = chunks
        .into_iter()
        .enumerate()
        .map(|(k, rows)| {
            let sink = &sinks[k];
            move || -> Result<()> {
                // SIMD 内核的落盘暂存：**每个 worker 一份、跨块复用**。
                // 原来它是 `cdef_filter` 里的一行 `let mut out = [0i16; 64]` —— 那是
                // **每次调用**一次 128 字节的清零，而 1080p 一帧有三个平面 ×
                // (1920/8)×(1080/8) = 32400 个 8×8 块 ⇒ **~12 MB/帧的白写**。
                // 内核只会产出 `h × w` 个样本，而且每个都会被读一遍 ⇒ 不需要初值。
                let mut scratch = vec![0i16; super::simd::LANES * 8];
                for &r in rows {
                    for c in (0..mi_cols).step_by(step4 as usize) {
                        let base_r = r & mask;
                        let base_c = c & mask;
                        let slot = base_r as usize * mi_cols as usize + base_c as usize;
                        let idx = grid.cdef_idx.get(slot).copied().unwrap_or(-1);
                        cdef_block(frame, sink, fh, seq, grid, r, c, idx, &mut scratch)?;
                    }
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
    scratch: &mut [i16],
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

    // §7.15.2：方向与方差都从**亮度**样本里量
    let (y_dir, var) = cdef_direction(curr, r, c);

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
        curr, sinks, seq, grid, 0, r, c, pri_str, sec_str, damping, dir, scratch,
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
        curr, sinks, seq, grid, 1, r, c, uv_pri, uv_sec, uv_damping, uv_dir, scratch,
    );
    cdef_filter(
        curr, sinks, seq, grid, 2, r, c, uv_pri, uv_sec, uv_damping, uv_dir, scratch,
    );
    Ok(())
}

/// `CDEF direction process`（规范 §7.15.2）：返回 `(yDir, var)`。
fn cdef_direction(curr: &CurrFrame, r: i32, c: i32) -> (usize, i32) {
    let plane = &curr.planes[0];
    let bit_depth = plane.bit_depth;
    let mut partial = [[0i32; 15]; 8];

    let x0 = (c << MI_SIZE_LOG2) as usize;
    let y0 = (r << MI_SIZE_LOG2) as usize;
    let sh = u32::from(bit_depth - 8);
    let stride = plane.stride;
    // **逐行切片**读（2026-10-04）：原来每个样本一次 `Plane::get` —— 那是**每样本**
    // 一次 debug_assert + 一次 `y·stride + x` 乘法。这里一行取一次切片，
    // 8 个样本一次读全；partial 的累加顺序变了但全是 i32 整数加、逐位不变。
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
    let var = (best_cost - cost[(y_dir + 4) & 7]) >> 10;
    (y_dir, var)
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
    scratch: &mut [i16],
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

    // **内点判定**：20 个候选最多外扩 2 个样本，把整块的极值折回 MI 网格判一次 ——
    // 内点就不必逐候选做那 4 个越界比较（外圈超块才走带检查的一般路径）。
    let y_lo_mi = ((y0 - 2).max(0) << sub_y) >> MI_SIZE_LOG2;
    let y_hi_mi = ((y0 + h as i32 + 1) << sub_y) >> MI_SIZE_LOG2;
    let x_lo_mi = ((x0 - 2).max(0) << sub_x) >> MI_SIZE_LOG2;
    let x_hi_mi = ((x0 + w as i32 + 1) << sub_x) >> MI_SIZE_LOG2;
    let interior =
        y0 >= 2 && x0 >= 2 && y_lo_mi >= 0 && x_lo_mi >= 0 && y_hi_mi < mi_rows && x_hi_mi < mi_cols;

    // **SIMD**：两行 × 8 列一次算完，i16 通道（见 `simd::cdef_pair16`）。
    // 除了 `interior`（保证 12 个候选在**帧的 4×4 网格内** ⇒ 值是正确的），
    // 还要额外保证**取址**留在平面缓冲内：8 通道读最远到 `x0 + 9`（`dx ≤ 2` 再加 7 列），
    // 行最远到 `y0 + h + 1`。色度（`w = 4`）只用前 4 个通道，另外 4 个读到的是右边的样本、
    // 直接丢弃 —— 所以这里只判"读得到"，不判"用得上"。
    let cap_rows = (data.len() / stride) as i32;
    let simd = interior
        && super::simd::avx2_available()
        && x0 >= 2
        && y0 >= 2
        && x0 + 9 < stride as i32
        && y0 + h as i32 + 1 < cap_rows;
    if simd {
        // 12 个候选的偏移（相对块左上角）与抽头：**只跟方向/强度有关**，
        // 与行无关 ⇒ 每块算一次。约定前 4 个是主抽头、后 8 个是次抽头。
        let mut offs = [0isize; super::simd::CDEF_CANDS];
        let mut taps = [0i32; super::simd::CDEF_CANDS];
        let mut n = 0usize;
        for k in 0..2usize {
            for sign in [-1i32, 1] {
                let [dy, dx] = CDEF_DIRECTIONS[dir][k];
                offs[n] = (sign * dy) as isize * stride as isize + (sign * dx) as isize;
                taps[n] = pri_taps[k];
                n += 1;
            }
        }
        for k in 0..2usize {
            for dir_off in [-2i32, 2] {
                let d = ((dir as i32 + dir_off) & 7) as usize;
                let [dy, dx] = CDEF_DIRECTIONS[d][k];
                for sign in [-1i32, 1] {
                    offs[n] = (sign * dy) as isize * stride as isize + (sign * dx) as isize;
                    taps[n] = sec_taps[k];
                    n += 1;
                }
            }
        }
        let out = &mut scratch[..super::simd::LANES * 8]; // h ≤ 8，每行 8 列
        // SAFETY: `simd` 已保证 AVX2 可用、`pri_adj`/`sec_adj` 远小于 16，
        // 且 12 个候选 × 8 通道的取址全部落在平面缓冲内（上面的窗口判据）。
        unsafe {
            super::simd::cdef_pair16(
                data,
                stride as isize,
                x0 as isize,
                y0 as isize,
                h,
                w,
                &offs,
                &taps,
                pri_str,
                pri_adj as u32,
                sec_str,
                sec_adj as u32,
                out,
            );
        }
        for i in 0..h {
            let y = y0 as usize + i;
            // 内核输出已过 `Clip3(mn, mx, ·)`（`mn`/`mx` 都是合法样本）⇒ 值必在
            // `[0, 2^bd)` 内，落盘不需要再逐样本钳位（`set_row_i16` 里有断言）。
            sink.set_row_i16(x0 as usize, y, &out[i * w..i * w + w]);
        }
        return;
    }

    for i in 0..h {
        let y = y0 as usize + i;
        for j in 0..w {
            let x = x0 as usize + j;
            let cur = i32::from(data[y * stride + x]);
            let mut max = cur;
            let mut min = cur;
            let mut sum = 0i32;
            for k in 0..2 {
                for sign in [-1i32, 1] {
                    if interior {
                        let [dy, dx] = CDEF_DIRECTIONS[dir][k];
                        let py = (y as i32 + sign * dy) as usize;
                        let px = (x as i32 + sign * dx) as usize;
                        // SAFETY(逻辑)：interior 已保证 20 个候选都在平面缓冲内
                        cdef_acc(
                            i32::from(data[py * stride + px]),
                            cur, pri_str, pri_adj, pri_taps[k], &mut sum, &mut max, &mut min,
                        );
                        for dir_off in [-2i32, 2] {
                            let d = ((dir as i32 + dir_off) & 7) as usize;
                            let [dy, dx] = CDEF_DIRECTIONS[d][k];
                            let py = (y as i32 + sign * dy) as usize;
                            let px = (x as i32 + sign * dx) as usize;
                            cdef_acc(
                                i32::from(data[py * stride + px]),
                                cur, sec_str, sec_adj, sec_taps[k], &mut sum, &mut max, &mut min,
                            );
                        }
                    } else {
                        if let Some(p) = cdef_get_at(
                            curr, plane, x0, y0, i as i32, j as i32, dir, k, sign, sub_x, sub_y,
                            mi_rows, mi_cols,
                        ) {
                            cdef_acc(p, cur, pri_str, pri_adj, pri_taps[k], &mut sum, &mut max, &mut min);
                        }
                        for dir_off in [-2i32, 2] {
                            let d = ((dir as i32 + dir_off) & 7) as usize;
                            if let Some(s2) = cdef_get_at(
                                curr, plane, x0, y0, i as i32, j as i32, d, k, sign, sub_x,
                                sub_y, mi_rows, mi_cols,
                            ) {
                                cdef_acc(s2, cur, sec_str, sec_adj, sec_taps[k], &mut sum, &mut max, &mut min);
                            }
                        }
                    }
                }
            }
            let adjust = if sum < 0 { 1 } else { 0 };
            let v = clip3(min, max, cur + ((8 + sum - adjust) >> 4));
            sink.set(x, y, v);
        }
    }
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

/// `cdef_get_at( ... )`（规范 §7.15.3）：取候选样本，不在滤波区域内返回 `None`。
///
/// 候选位置可以落在帧的 4×4 网格**之外**（那就不参与），也可以落在"网格内但像素
/// 尺寸外"（`MiRows`/`MiCols` 向上取整到 8 造成的），后者照常取 —— 由
/// `Plane` 的 frame border 承接地址。
#[allow(clippy::too_many_arguments)]
fn cdef_get_at(
    curr: &CurrFrame,
    plane: u8,
    x0: i32,
    y0: i32,
    i: i32,
    j: i32,
    dir: usize,
    k: usize,
    sign: i32,
    sub_x: u32,
    sub_y: u32,
    mi_rows: i32,
    mi_cols: i32,
) -> Option<i32> {
    let y = y0 + i + sign * CDEF_DIRECTIONS[dir][k][0];
    let x = x0 + j + sign * CDEF_DIRECTIONS[dir][k][1];
    let candidate_r = (y << sub_y) >> MI_SIZE_LOG2;
    let candidate_c = (x << sub_x) >> MI_SIZE_LOG2;
    if candidate_r < 0 || candidate_c < 0 || candidate_r >= mi_rows || candidate_c >= mi_cols {
        return None;
    }
    Some(i32::from(
        curr.planes[plane as usize].get(x as usize, y as usize),
    ))
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
}
