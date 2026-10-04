//! AV1 的**分区树**（规范 §5.11.4 `decode_partition`）—— 自研软解 M4a。
//!
//! 这是引擎里第一段"**用熵解码器读真实语法**"的代码：瓦片数据里第一个有语义的
//! 符号就是分区类型。它把这一个瓦片切成一个个块（`decode_block` 的输入），
//! 同时把每个 4×4 位置的块尺寸记进 `MiSizes` 网格 —— 后面所有语法（模式、系数、
//! 环路滤波的块级判断）都靠这张网格。
//!
//! ```text
//! decode_partition( r, c, bSize ) {
//!   if ( r >= MiRows || c >= MiCols ) return
//!   AvailU = is_inside( r - 1, c );  AvailL = is_inside( r, c - 1 )
//!   num4x4 = Num_4x4_Blocks_Wide[ bSize ]; halfBlock4x4 = num4x4 >> 1
//!   hasRows = ( r + halfBlock4x4 ) < MiRows
//!   hasCols = ( c + halfBlock4x4 ) < MiCols
//!   if ( bSize < BLOCK_8X8 )            partition = PARTITION_NONE          // 不读符号
//!   else if ( hasRows && hasCols )      partition = S()                     // 完整 10 符号
//!   else if ( hasCols )                 split_or_horz = S()  → SPLIT / HORZ  // 派生 CDF（3 项）
//!   else if ( hasRows )                 split_or_vert = S()  → SPLIT / VERT
//!   else                                partition = PARTITION_SPLIT
//!   …
//! }
//! ```
//!
//! ## 三处容易写错的地方（都对照着规范原文抄）
//!
//! 1. **分区的 CDF 上下文用的是 `MiSizes` 网格里邻居的宽/高 log2**，不是"邻居是不是
//!    更小"（`above = AvailU && (Mi_Width_Log2[ MiSizes[r-1][c] ] < bsl)`）。
//!    所以 `MiSizes` 必须**每个 4×4 位置都填**（规范里 `decode_block` 是用
//!    `bw4 × bh4` 双重循环填的，不是只写起点）。
//! 2. `slit_or_horz` / `split_or_vert` 的 CDF 是**现算的派生表**：把
//!    `partitionCdf` 里那些"不该出现"的符号的概率加起来当符号 0 的概率。
//!    BLOCK_128X128 时没有 `PARTITION_VERT_4`（`Default_Partition_W128_Cdf` 只有 8 个符号）。
//! 3. `hasRows` / `hasCols` 用的是**帧**的 `MiRows` / `MiCols`，而 `is_inside` 用的是
//    **瓦片**的 `MiRowStart/End`、`MiColStart/End` —— 两者不是一回事。
//!
//! ## 本轮的验收口径
//!
//! `decode_block` 还没做（那是 M4b），所以**不调 `exit_symbol()`**：算术流只被分区
//! 符号消耗了一部分，末尾位置对不上是正常的。可验证的是**结构不变量**：
//! 分区树必须**恰好覆盖整个瓦片**（每个 4×4 位置被写且只被写一次），
//! 且遍历完树后算术流**没有被读空**（后面还有块内容）。

use me_core::{MediaError, Result};

use super::entropy::SymbolDecoder;
use super::params::TileInfo;
use super::tile::Tile;

// ── 块尺寸（规范 §6.4.1 的常量表）─────────────────────────────────────────

/// `BLOCK_SIZES`
pub const BLOCK_SIZES: usize = 22;
pub const BLOCK_4X4: usize = 0;
pub const BLOCK_4X8: usize = 1;
pub const BLOCK_8X4: usize = 2;
pub const BLOCK_8X8: usize = 3;
pub const BLOCK_8X16: usize = 4;
pub const BLOCK_16X8: usize = 5;
pub const BLOCK_16X16: usize = 6;
pub const BLOCK_16X32: usize = 7;
pub const BLOCK_32X16: usize = 8;
pub const BLOCK_32X32: usize = 9;
pub const BLOCK_32X64: usize = 10;
pub const BLOCK_64X32: usize = 11;
pub const BLOCK_64X64: usize = 12;
pub const BLOCK_64X128: usize = 13;
pub const BLOCK_128X64: usize = 14;
pub const BLOCK_128X128: usize = 15;
pub const BLOCK_4X16: usize = 16;
pub const BLOCK_16X4: usize = 17;
pub const BLOCK_8X32: usize = 18;
pub const BLOCK_32X8: usize = 19;
pub const BLOCK_16X64: usize = 20;
pub const BLOCK_64X16: usize = 21;
/// `BLOCK_INVALID`（表里的"不适用"哨兵）
pub const BLOCK_INVALID: u8 = 22;

// ── 分区类型（规范 §6.4.2）───────────────────────────────────────────────

pub const PARTITION_NONE: usize = 0;
pub const PARTITION_HORZ: usize = 1;
pub const PARTITION_VERT: usize = 2;
pub const PARTITION_SPLIT: usize = 3;
pub const PARTITION_HORZ_A: usize = 4;
pub const PARTITION_HORZ_B: usize = 5;
pub const PARTITION_VERT_A: usize = 6;
pub const PARTITION_VERT_B: usize = 7;
pub const PARTITION_HORZ_4: usize = 8;
pub const PARTITION_VERT_4: usize = 9;
/// `PARTITION_TYPES`
pub const PARTITION_TYPES: usize = 10;
/// `PARTITION_CONTEXTS`
pub const PARTITION_CONTEXTS: usize = 4;

/// `Mi_Width_Log2[ BLOCK_SIZES ]`（宽度的 log2，单位 4 样本）
pub const MI_WIDTH_LOG2: [u8; BLOCK_SIZES] =
    [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 0, 2, 1, 3, 2, 4];
/// `Mi_Height_Log2[ BLOCK_SIZES ]`
pub const MI_HEIGHT_LOG2: [u8; BLOCK_SIZES] =
    [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 2, 0, 3, 1, 4, 2];
/// `Num_4x4_Blocks_Wide[ BLOCK_SIZES ]`
pub const NUM_4X4_BLOCKS_WIDE: [u32; BLOCK_SIZES] =
    [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16];
/// `Num_4x4_Blocks_High[ BLOCK_SIZES ]`
pub const NUM_4X4_BLOCKS_HIGH: [u32; BLOCK_SIZES] =
    [1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4];

/// `Partition_Subsize[ partition ][ bSize ]`（规范 §9.x 的表；`22` 表示 `BLOCK_INVALID`）
pub const PARTITION_SUBSIZE: [[u8; BLOCK_SIZES]; PARTITION_TYPES] = [
    // PARTITION_NONE：还是自己（对方形块）
    [0, 22, 22, 3, 22, 22, 6, 22, 22, 9, 22, 22, 12, 22, 22, 15, 22, 22, 22, 22, 22, 22],
    // PARTITION_HORZ：高度减半
    [22, 22, 22, 2, 22, 22, 5, 22, 22, 8, 22, 22, 11, 22, 22, 14, 22, 22, 22, 22, 22, 22],
    // PARTITION_VERT：宽度减半
    [22, 22, 22, 1, 22, 22, 4, 22, 22, 7, 22, 22, 10, 22, 22, 13, 22, 22, 22, 22, 22, 22],
    // PARTITION_SPLIT：宽高各减半
    [22, 22, 22, 0, 22, 22, 3, 22, 22, 6, 22, 22, 9, 22, 22, 12, 22, 22, 22, 22, 22, 22],
    // PARTITION_HORZ_A：最大子块是下半（与 HORZ 同形）
    [22, 22, 22, 2, 22, 22, 5, 22, 22, 8, 22, 22, 11, 22, 22, 14, 22, 22, 22, 22, 22, 22],
    // PARTITION_HORZ_B：同上
    [22, 22, 22, 2, 22, 22, 5, 22, 22, 8, 22, 22, 11, 22, 22, 14, 22, 22, 22, 22, 22, 22],
    // PARTITION_VERT_A：最大子块是右半（与 VERT 同形）
    [22, 22, 22, 1, 22, 22, 4, 22, 22, 7, 22, 22, 10, 22, 22, 13, 22, 22, 22, 22, 22, 22],
    // PARTITION_VERT_B：同上
    [22, 22, 22, 1, 22, 22, 4, 22, 22, 7, 22, 22, 10, 22, 22, 13, 22, 22, 22, 22, 22, 22],
    // PARTITION_HORZ_4：高度 1/4
    [22, 22, 22, 22, 22, 22, 17, 22, 22, 19, 22, 22, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22],
    // PARTITION_VERT_4：宽度 1/4
    [22, 22, 22, 22, 22, 22, 16, 22, 22, 18, 22, 22, 20, 22, 22, 22, 22, 22, 22, 22, 22, 22],
];

// ── 分区 CDF ────────────────────────────────────────────────────────────
//
// 表本身在 `cdf_tables`（**从规范原文自动生成**，见 `tools/gen_cdf_tables.py`）。
// 这里只放"每瓦片一份副本、就地自适应"的状态。

/// 一个瓦片的**分区 CDF 副本**（规范里 `init_symbol` 时把 `Default_*` 拷成 `Tile*`）。
///
/// 每个瓦片一份、就地自适应 —— 这正是 AV1 的"瓦片独立解码"：瓦片之间不共享 CDF。
#[derive(Debug, Clone)]
pub struct PartitionCdfs {
    pub w8: [[u16; 5]; PARTITION_CONTEXTS],
    pub w16: [[u16; 11]; PARTITION_CONTEXTS],
    pub w32: [[u16; 11]; PARTITION_CONTEXTS],
    pub w64: [[u16; 11]; PARTITION_CONTEXTS],
    pub w128: [[u16; 9]; PARTITION_CONTEXTS],
}

impl Default for PartitionCdfs {
    fn default() -> Self {
        use crate::video::cdf_tables as T;
        Self {
            w8: T::DEFAULT_PARTITION_W8_CDF,
            w16: T::DEFAULT_PARTITION_W16_CDF,
            w32: T::DEFAULT_PARTITION_W32_CDF,
            w64: T::DEFAULT_PARTITION_W64_CDF,
            w128: T::DEFAULT_PARTITION_W128_CDF,
        }
    }
}

impl PartitionCdfs {
    /// 按 `bsl`（= `Mi_Width_Log2[bSize]`）取那份 CDF（`bsl >= 2` 时是 10 个符号）。
    fn symbols_4(&self, bsl: u8, ctx: usize) -> &[u16] {
        match bsl {
            1 => &self.w8[ctx],
            2 => &self.w16[ctx],
            3 => &self.w32[ctx],
            4 => &self.w64[ctx],
            _ => &self.w128[ctx], // bsl = 5
        }
    }

    fn symbols_4_mut(&mut self, bsl: u8, ctx: usize) -> &mut [u16] {
        match bsl {
            1 => &mut self.w8[ctx],
            2 => &mut self.w16[ctx],
            3 => &mut self.w32[ctx],
            4 => &mut self.w64[ctx],
            _ => &mut self.w128[ctx],
        }
    }
}

// ── MiSizes 网格 ────────────────────────────────────────────────────────

/// 帧级的 `MiSizes` 网格（每个 4×4 位置一个块尺寸）。
///
/// 分区树的上下文要读邻居的块尺寸，所以这张网格必须**每个位置都填**
/// （规范里 `decode_block` 用 `bw4 × bh4` 的双重循环写）。
/// 另外带一份 `filled` 掩码：分区树"恰好覆盖瓦片"这条不变量就靠它验。
#[derive(Debug, Clone)]
pub struct MiGrid {
    rows: usize,
    cols: usize,
    sizes: Vec<u8>,
    filled: Vec<bool>,
    /// **重复写入**的位置数（诊断）。`uncovered == 0` 只说明"没有缺口"，
    /// 说明不了"没有重叠" —— 分区树解错了也可能恰好铺满。这张计数就是补这一刀。
    overlaps: usize,
}

impl MiGrid {
    /// 空壳（配 [`MiGrid::reset_all`] 用）。
    pub fn empty() -> Self {
        Self { rows: 0, cols: 0, sizes: Vec::new(), filled: Vec::new(), overlaps: 0 }
    }

    /// 建一张 `rows × cols` 的网格，初值 `BLOCK_INVALID`（未解码）。
    pub fn new(rows: usize, cols: usize) -> Self {
        let mut g = Self::empty();
        g.reset_all(rows, cols);
        g
    }

    /// 按 `rows × cols` 重填（跨帧复用同一张网格时用；语义与 `new` 一致）。
    pub fn reset_all(&mut self, rows: usize, cols: usize) {
        self.rows = rows;
        self.cols = cols;
        self.sizes.clear();
        self.sizes.resize(rows * cols, BLOCK_INVALID);
        self.filled.clear();
        self.filled.resize(rows * cols, false);
        self.overlaps = 0;
    }

    /// 重复写入的位置数（诊断）。
    pub fn overlaps(&self) -> usize {
        self.overlaps
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// 读某个位置的块尺寸（**只该在 `is_inside` 为真时读**）。
    pub fn get(&self, r: i32, c: i32) -> u8 {
        if r < 0 || c < 0 || r as usize >= self.rows || c as usize >= self.cols {
            return BLOCK_INVALID;
        }
        self.sizes[r as usize * self.cols + c as usize]
    }

    /// 某个位置是否已被分区树写过。
    pub fn is_filled(&self, r: usize, c: usize) -> bool {
        self.filled[r * self.cols + c]
    }

    /// 清掉"已写"标记（**不清尺寸**）：跨帧复用同一张网格时用。
    ///
    /// 尺寸不必清 —— 本帧的树会覆盖每一个位置；而 `neighbour_size` 只在
    /// `filled` 为真时才返回尺寸，所以残留的旧尺寸不会被读到。
    pub fn reset_filled(&mut self) {
        self.filled.fill(false);
        self.overlaps = 0;
    }

    /// 读邻居尺寸，**只有本帧已写过**才返回（防跨帧残留污染）。
    fn neighbour_size(&self, r: i32, c: i32) -> Option<u8> {
        if r < 0 || c < 0 || r as usize >= self.rows || c as usize >= self.cols {
            return None;
        }
        let i = r as usize * self.cols + c as usize;
        if self.filled[i] {
            Some(self.sizes[i])
        } else {
            None
        }
    }

    /// 把一个块铺到网格上（= 规范里 `decode_block` 开头的那个双重循环）。
    ///
    /// 超出帧边界的部分跳过（帧最右/最下的块会越界）。
    fn fill(&mut self, r: i32, c: i32, bsize: u8) {
        if bsize as usize >= BLOCK_SIZES {
            return;
        }
        let w = NUM_4X4_BLOCKS_WIDE[bsize as usize] as i32;
        let h = NUM_4X4_BLOCKS_HIGH[bsize as usize] as i32;
        for y in 0..h {
            for x in 0..w {
                let (rr, cc) = (r + y, c + x);
                if rr < 0 || cc < 0 || rr as usize >= self.rows || cc as usize >= self.cols {
                    continue;
                }
                let i = rr as usize * self.cols + cc as usize;
                if self.filled[i] {
                    self.overlaps += 1;
                }
                self.sizes[i] = bsize;
                self.filled[i] = true;
            }
        }
    }
}

// ── 分区树 ──────────────────────────────────────────────────────────────

/// 一棵分区树的统计（诊断 + 验收）。
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PartitionStats {
    /// `decode_partition` 调用次数（树的节点数）
    pub nodes: u64,
    /// 叶子块数（= 规范里 `decode_block` 的调用次数）
    pub leaves: u64,
    pub max_depth: u32,
    /// 每种分区类型读到了多少次
    pub partition_histogram: [u64; PARTITION_TYPES],
    /// 叶子块的尺寸分布
    pub leaf_histogram: [u64; BLOCK_SIZES],
    /// 读过的算术符号数
    pub symbols: u64,
}

/// 分区树的解码上下文（一个瓦片一份）。
pub struct PartitionCtx<'a, 'b> {
    pub dec: &'a mut SymbolDecoder<'b>,
    pub cdfs: &'a mut PartitionCdfs,
    pub grid: &'a mut MiGrid,
    pub mi_rows: i32,
    pub mi_cols: i32,
    pub mi_row_start: i32,
    pub mi_row_end: i32,
    pub mi_col_start: i32,
    pub mi_col_end: i32,
    pub stats: PartitionStats,
    /// 叶子块的回调（`decode_block`）。`None` = 只走分区树（M4a 的验收口径）。
    pub leaf_hook:
        Option<&'a mut dyn FnMut(&mut SymbolDecoder<'b>, i32, i32, u8, &MiGrid) -> Result<()>>,
}

impl PartitionCtx<'_, '_> {
    /// `is_inside( r, c )`：用**瓦片**的范围（不是帧的）。
    fn is_inside(&self, r: i32, c: i32) -> bool {
        r >= self.mi_row_start && r < self.mi_row_end && c >= self.mi_col_start && c < self.mi_col_end
    }

    /// 分区符号的上下文（规范 §8.3.2 的 `partition:` 条目）。
    ///
    /// `AvailU`/`AvailL` 为真时那个邻居一定已经解码过（raster 顺序），
    /// 但这里仍对"未解码（`BLOCK_INVALID`）"做了兜底 —— 少一次数组越界 panic。
    fn partition_ctx(&self, bsize: usize, r: i32, c: i32) -> usize {
        let bsl = MI_WIDTH_LOG2[bsize];
        let smaller = |nb: Option<u8>, log2: &[u8; BLOCK_SIZES]| -> bool {
            match nb {
                Some(nb) => (nb as usize) < BLOCK_SIZES && log2[nb as usize] < bsl,
                None => false,
            }
        };
        let above = self.is_inside(r - 1, c) && smaller(self.grid.neighbour_size(r - 1, c), &MI_WIDTH_LOG2);
        let left = self.is_inside(r, c - 1) && smaller(self.grid.neighbour_size(r, c - 1), &MI_HEIGHT_LOG2);
        (left as usize) * 2 + above as usize
    }

    fn read_partition_symbol(&mut self, bsize: usize, r: i32, c: i32) -> Result<usize> {
        let bsl = MI_WIDTH_LOG2[bsize];
        let pctx = self.partition_ctx(bsize, r, c);
        // `cdfs` 与 `dec` 是不同的字段，可以同时可变借用
        let sym = {
            let cdf = self.cdfs.symbols_4_mut(bsl, pctx);
            self.dec.label = "partition";
            self.dec.read_symbol(cdf)?
        };
        if Self::trace_part() && self.stats.symbols < 80 {
            println!(
                "   part#{:<3} r={:<3} c={:<3} bs={:<2} bsl={} ctx={} -> {}",
                self.stats.symbols, r, c, bsize, bsl, pctx, sym
            );
        }
        self.stats.symbols += 1;
        Ok(sym as usize)
    }

    /// `split_or_horz` / `split_or_vert`（规范 §8.3.2）：用 `partitionCdf` 现算一份 3 项的派生 CDF。
    ///
    /// 规范：把那些"在这个方向上不允许出现"的符号的概率加起来当符号 0 的概率。
    /// **两者的合并组是镜像的、不能共用**：
    /// - `split_or_horz`（底边越界 ⇒ 只剩 SPLIT / HORZ）：VERT、VERT_B、VERT_4 并入；
    /// - `split_or_vert`（右边越界 ⇒ 只剩 SPLIT / VERT）：HORZ、HORZ_B、HORZ_4 并入。
    ///
    /// `vertical` = 走 `split_or_vert` 那一支。另外 `bsl` 为 1（8×8）时不会走到这里，
    /// BLOCK_128X128 时表里没有 `PARTITION_*_4`。
    fn read_split_or(&mut self, bsize: usize, r: i32, c: i32, vertical: bool) -> Result<bool> {
        let bsl = MI_WIDTH_LOG2[bsize];
        let pctx = self.partition_ctx(bsize, r, c);
        let mut cdf = [0u16; 3];
        {
            let pcdf = self.cdfs.symbols_4(bsl, pctx);
            let p = |i: usize| pcdf[i] as i32;
            let d = |i: usize| p(i) - p(i - 1);
            let (a, b, c_, d_, e, quarter) = if vertical {
                (
                    d(PARTITION_HORZ),
                    d(PARTITION_SPLIT),
                    d(PARTITION_HORZ_A),
                    d(PARTITION_HORZ_B),
                    d(PARTITION_VERT_A),
                    PARTITION_HORZ_4,
                )
            } else {
                (
                    d(PARTITION_VERT),
                    d(PARTITION_SPLIT),
                    d(PARTITION_HORZ_A),
                    d(PARTITION_VERT_A),
                    d(PARTITION_VERT_B),
                    PARTITION_VERT_4,
                )
            };
            let mut psum = a + b + c_ + d_ + e;
            if bsize != BLOCK_128X128 {
                psum += d(quarter);
            }
            cdf[0] = ((1 << 15) - psum) as u16;
            cdf[1] = 1 << 15;
            cdf[2] = 0;
        }
        let sym = self.dec.read_symbol(&mut cdf)?;
        if Self::trace_part() && self.stats.symbols < 80 {
            println!(
                "   part#{:<3} r={:<3} c={:<3} bs={:<2} bsl={} ctx={} -> {} (split_or_{})",
                self.stats.symbols,
                r,
                c,
                bsize,
                bsl,
                pctx,
                sym,
                if vertical { "vert" } else { "horz" }
            );
        }
        self.stats.symbols += 1;
        Ok(sym == 1)
    }

    /// trace 开关（`AV1_TRACE_PART`）：只读一次环境变量。
    fn trace_part() -> bool {
        static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *T.get_or_init(|| std::env::var("AV1_TRACE_PART").is_ok())
    }

    /// 叶子块：铺 `MiSizes`、计数，然后（`leaf_hook` 存在时）走 `decode_block`。
    fn leaf(&mut self, r: i32, c: i32, bsize: u8) -> Result<()> {
        if Self::trace_part() && self.stats.leaves < 40 {
            println!(
                "      leaf#{:<3} r={:<3} c={:<3} bs={:<2}",
                self.stats.leaves, r, c, bsize
            );
        }
        self.grid.fill(r, c, bsize);
        self.stats.leaves += 1;
        self.stats.leaf_histogram[bsize as usize] += 1;
        if let Some(hook) = self.leaf_hook.as_mut() {
            hook(&mut *self.dec, r, c, bsize, &*self.grid)?;
        }
        Ok(())
    }

    pub fn decode_partition(&mut self, r: i32, c: i32, bsize: usize, depth: u32) -> Result<()> {
        if r >= self.mi_rows || c >= self.mi_cols {
            return Ok(());
        }
        self.stats.nodes += 1;
        self.stats.max_depth = self.stats.max_depth.max(depth);

        let num4x4 = NUM_4X4_BLOCKS_WIDE[bsize] as i32;
        let half = num4x4 >> 1;
        let quarter = half >> 1;
        let has_rows = (r + half) < self.mi_rows;
        let has_cols = (c + half) < self.mi_cols;

        let partition = if bsize < BLOCK_8X8 {
            PARTITION_NONE
        } else if has_rows && has_cols {
            self.read_partition_symbol(bsize, r, c)?
        } else if has_cols {
            if self.read_split_or(bsize, r, c, false)? {
                PARTITION_SPLIT
            } else {
                PARTITION_HORZ
            }
        } else if has_rows {
            if self.read_split_or(bsize, r, c, true)? {
                PARTITION_SPLIT
            } else {
                PARTITION_VERT
            }
        } else {
            PARTITION_SPLIT
        };
        if partition >= PARTITION_TYPES {
            return Err(MediaError::Bitstream(format!(
                "分区类型越界：{partition}（bsl = {}）",
                MI_WIDTH_LOG2[bsize]
            )));
        }
        self.stats.partition_histogram[partition] += 1;

        let sub_size = PARTITION_SUBSIZE[partition][bsize];
        if sub_size as usize >= BLOCK_SIZES {
            return Err(MediaError::Bitstream(format!(
                "分区 {partition} 不适用于块尺寸 {bsize}（Partition_Subsize 给的是 INVALID）"
            )));
        }
        // `split_size` 只有 SPLIT 分支用得到（BLOCK_4X4 的 SPLIT 是 INVALID，
        // 但那种尺寸根本不会走到 SPLIT）
        let split_size = PARTITION_SUBSIZE[PARTITION_SPLIT][bsize];

        match partition {
            PARTITION_NONE => self.leaf(r, c, sub_size)?,
            PARTITION_HORZ => {
                self.leaf(r, c, sub_size)?;
                if has_rows {
                    self.leaf(r + half, c, sub_size)?;
                }
            }
            PARTITION_VERT => {
                self.leaf(r, c, sub_size)?;
                if has_cols {
                    self.leaf(r, c + half, sub_size)?;
                }
            }
            PARTITION_SPLIT => {
                if split_size as usize >= BLOCK_SIZES {
                    return Err(MediaError::Bitstream(format!(
                        "块尺寸 {bsize} 不支持 SPLIT 分区（Partition_Subsize 给的是 INVALID）"
                    )));
                }
                self.decode_partition(r, c, split_size as usize, depth + 1)?;
                self.decode_partition(r, c + half, split_size as usize, depth + 1)?;
                self.decode_partition(r + half, c, split_size as usize, depth + 1)?;
                self.decode_partition(r + half, c + half, split_size as usize, depth + 1)?;
            }
            PARTITION_HORZ_A => {
                self.leaf(r, c, split_size)?;
                self.leaf(r, c + half, split_size)?;
                self.leaf(r + half, c, sub_size)?;
            }
            PARTITION_HORZ_B => {
                self.leaf(r, c, sub_size)?;
                self.leaf(r + half, c, split_size)?;
                self.leaf(r + half, c + half, split_size)?;
            }
            PARTITION_VERT_A => {
                self.leaf(r, c, split_size)?;
                self.leaf(r + half, c, split_size)?;
                self.leaf(r, c + half, sub_size)?;
            }
            PARTITION_VERT_B => {
                self.leaf(r, c, sub_size)?;
                self.leaf(r, c + half, split_size)?;
                self.leaf(r + half, c + half, split_size)?;
            }
            PARTITION_HORZ_4 => {
                // 规范 §5.11.4 写的是 `r + quarterBlock4x4 * k`（k = 0..3），
                // 这里把 k=0/1 写开，省得 clippy 报"恒为 0 / 无效果"
                self.leaf(r, c, sub_size)?;
                self.leaf(r + quarter, c, sub_size)?;
                self.leaf(r + quarter * 2, c, sub_size)?;
                if r + quarter * 3 < self.mi_rows {
                    self.leaf(r + quarter * 3, c, sub_size)?;
                }
            }
            _ => {
                // PARTITION_VERT_4
                self.leaf(r, c, sub_size)?;
                self.leaf(r, c + quarter, sub_size)?;
                self.leaf(r, c + quarter * 2, sub_size)?;
                if c + quarter * 3 < self.mi_cols {
                    self.leaf(r, c + quarter * 3, sub_size)?;
                }
            }
        }
        Ok(())
    }
}

/// 一趟分区树解码的结果。
pub struct PartitionTree {
    pub stats: PartitionStats,
    /// 遍历完树后算术流还剩多少真实位（**应当仍为正**：后面还有块内容）
    pub bits_left: i64,
    /// 覆盖检查发现的问题（空 = 每个 4×4 位置都被恰好覆盖）
    pub uncovered: usize,
}

/// 把一个瓦片的**分区树**走完（规范 §5.11.1 `decode_tile` 的外循环 + §5.11.4）。
///
/// `grid` 是**帧级**的 `MiSizes`（调用方持有，因为 tile 之间共享这张网格的坐标系）。
pub fn decode_tile_partitions(
    tile: &Tile<'_>,
    ti: &TileInfo,
    grid: &mut MiGrid,
    disable_cdf_update: bool,
) -> Result<PartitionTree> {
    let mi_rows = *ti.mi_row_starts.last().unwrap_or(&0) as i32;
    let mi_cols = *ti.mi_col_starts.last().unwrap_or(&0) as i32;
    let mut dec = SymbolDecoder::new(tile.data, !disable_cdf_update)?;
    let mut cdfs = PartitionCdfs::default();
    let sb_size = if ti.sb_shift == 5 { BLOCK_128X128 } else { BLOCK_64X64 };
    let sb_size4 = NUM_4X4_BLOCKS_WIDE[sb_size] as i32;

    let mut ctx = PartitionCtx {
        dec: &mut dec,
        cdfs: &mut cdfs,
        grid,
        mi_rows,
        mi_cols,
        mi_row_start: tile.mi_row_start as i32,
        mi_row_end: tile.mi_row_end as i32,
        mi_col_start: tile.mi_col_start as i32,
        mi_col_end: tile.mi_col_end as i32,
        stats: PartitionStats::default(),
        leaf_hook: None,
    };

    let mut r = tile.mi_row_start as i32;
    while r < tile.mi_row_end as i32 {
        let mut c = tile.mi_col_start as i32;
        while c < tile.mi_col_end as i32 {
            ctx.decode_partition(r, c, sb_size, 0)?;
            c += sb_size4;
        }
        r += sb_size4;
    }

    // 覆盖检查：瓦片范围内每个 4×4 位置都被写过
    let mut uncovered = 0usize;
    for rr in tile.mi_row_start..tile.mi_row_end {
        for cc in tile.mi_col_start..tile.mi_col_end {
            if !ctx.grid.is_filled(rr as usize, cc as usize) {
                uncovered += 1;
            }
        }
    }
    let bits_left = ctx.dec.symbol_max_bits();
    Ok(PartitionTree { stats: ctx.stats, bits_left, uncovered })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::cdf_tables::{
        DEFAULT_PARTITION_W128_CDF, DEFAULT_PARTITION_W16_CDF, DEFAULT_PARTITION_W32_CDF,
        DEFAULT_PARTITION_W64_CDF, DEFAULT_PARTITION_W8_CDF,
    };

    #[test]
    fn tables_match_the_spec() {
        // 方形块在表里的下标是 0/3/6/9/12/15
        for (i, b) in [BLOCK_4X4, BLOCK_8X8, BLOCK_16X16, BLOCK_32X32, BLOCK_64X64, BLOCK_128X128]
            .into_iter()
            .enumerate()
        {
            assert_eq!(MI_WIDTH_LOG2[b] as usize, i, "方形块的宽度 log2");
            assert_eq!(MI_HEIGHT_LOG2[b] as usize, i, "方形块的高度 log2");
            assert_eq!(NUM_4X4_BLOCKS_WIDE[b], 1 << i);
            assert_eq!(NUM_4X4_BLOCKS_HIGH[b], 1 << i);
        }
        // SPLIT 对方形块给出小一号的方形；NONE 给回自己
        for b in [BLOCK_8X8, BLOCK_16X16, BLOCK_32X32, BLOCK_64X64, BLOCK_128X128] {
            assert_eq!(
                PARTITION_SUBSIZE[PARTITION_NONE][b] as usize,
                b,
                "NONE 应当是原尺寸"
            );
            assert_eq!(
                MI_WIDTH_LOG2[PARTITION_SUBSIZE[PARTITION_SPLIT][b] as usize],
                MI_WIDTH_LOG2[b] - 1,
                "SPLIT 应当宽高各减半"
            );
        }
        assert_eq!(PARTITION_SUBSIZE[PARTITION_SPLIT][BLOCK_4X4], BLOCK_INVALID);
        assert_eq!(PARTITION_SUBSIZE[PARTITION_HORZ_4][BLOCK_16X16] as usize, BLOCK_16X4);
        assert_eq!(PARTITION_SUBSIZE[PARTITION_VERT_4][BLOCK_16X16] as usize, BLOCK_4X16);
        // 矩形块上不该有有效的分区
        for b in [BLOCK_4X8, BLOCK_16X4, BLOCK_64X16] {
            for p in 0..PARTITION_TYPES {
                assert_eq!(
                    PARTITION_SUBSIZE[p][b], BLOCK_INVALID,
                    "矩形块 {b} 不该有 partition {p}"
                );
            }
        }
    }

    #[test]
    fn cdf_tables_are_monotonic_and_normalized() {
        // 递增、末项 1<<15、后面跟计数器 —— 这份表的方向写反就会"符号全落最后一档"
        for row in DEFAULT_PARTITION_W8_CDF.iter() {
            check_cdf(row);
        }
        for row in DEFAULT_PARTITION_W16_CDF.iter() {
            check_cdf(row);
        }
        for row in DEFAULT_PARTITION_W32_CDF.iter() {
            check_cdf(row);
        }
        for row in DEFAULT_PARTITION_W64_CDF.iter() {
            check_cdf(row);
        }
        for row in DEFAULT_PARTITION_W128_CDF.iter() {
            check_cdf(row);
        }
        // 128×128 不能 HORZ_4/VERT_4 → 只有 8 个符号
        assert_eq!(DEFAULT_PARTITION_W128_CDF[0].len(), 9);
    }

    fn check_cdf(row: &[u16]) {
        let n = row.len() - 1;
        for i in 1..n {
            assert!(row[i] > row[i - 1], "CDF 必须严格递增：{row:?}");
        }
        assert_eq!(row[n - 1], 1 << 15, "末项必须是 1<<15：{row:?}");
        assert_eq!(row[n], 0, "计数器初值是 0：{row:?}");
    }

    #[test]
    fn mi_grid_fills_every_4x4_of_a_block() {
        let mut g = MiGrid::new(16, 16);
        g.fill(2, 3, BLOCK_16X16 as u8);
        assert_eq!(g.get(2, 3), BLOCK_16X16 as u8);
        assert_eq!(g.get(2 + 3, 3 + 3), BLOCK_16X16 as u8, "右下角也要被填");
        assert!(!g.is_filled(2 + 4, 3), "块外不该被写");
        assert!(!g.is_filled(1, 3));
    }

    #[test]
    fn mi_grid_clips_at_the_frame_edge() {
        let mut g = MiGrid::new(5, 5);
        // 8×8 的块放在右下角：只该写到 5×5 之内
        g.fill(3, 3, BLOCK_8X8 as u8);
        assert_eq!(g.get(4, 4), BLOCK_8X8 as u8);
        assert!(!g.is_filled(0, 0));
        // 不该 panic
        g.fill(4, 4, BLOCK_64X64 as u8);
    }

    #[test]
    fn partition_ctx_uses_neighbour_block_shape() {
        // 整个 32×32 都是瓦片范围；在 (8,8) 要解一个 16×16（bsl = 2）。
        // 上面 (7,8) 是 8×8（bsl = 1 < 2）→ above = 1；
        // 左边 (8,7) 是 4×4（bsl = 0 < 2）→ left = 1 ⇒ ctx = 3
        let mut g = MiGrid::new(32, 32);
        g.fill(6, 8, BLOCK_8X8 as u8); // 覆盖 (6..8, 8..10) → (7,8) 是 8×8
        g.fill(8, 7, BLOCK_4X4 as u8); // 只覆盖 (8,7)
        let data = [0x5Au8; 64];
        let mut dec = SymbolDecoder::new(&data, true).unwrap();
        let mut cdfs = PartitionCdfs::default();
        {
            let ctx = PartitionCtx {
                dec: &mut dec,
                cdfs: &mut cdfs,
                grid: &mut g,
                mi_rows: 32,
                mi_cols: 32,
                mi_row_start: 0,
                mi_row_end: 32,
                mi_col_start: 0,
                mi_col_end: 32,
                stats: PartitionStats::default(),
                leaf_hook: None,
            };
            assert_eq!(ctx.partition_ctx(BLOCK_16X16, 8, 8), 3, "above + left 都更小");
        }
        // 邻居铺成和当前块同尺寸（16×16）→ 两个方向都不算"更小" ⇒ ctx = 0
        g.fill(4, 8, BLOCK_16X16 as u8); // 覆盖 (4..8, 8..12) → (7,8) 是 16×16
        g.fill(8, 4, BLOCK_16X16 as u8); // 覆盖 (8..12, 4..8) → (8,7) 是 16×16
        {
            let ctx = PartitionCtx {
                dec: &mut dec,
                cdfs: &mut cdfs,
                grid: &mut g,
                mi_rows: 32,
                mi_cols: 32,
                mi_row_start: 0,
                mi_row_end: 32,
                mi_col_start: 0,
                mi_col_end: 32,
                stats: PartitionStats::default(),
                leaf_hook: None,
            };
            assert_eq!(ctx.partition_ctx(BLOCK_16X16, 8, 8), 0, "邻居不必当前块小");
        }
    }
}
