//! 帧内**数据并行**的小工具（自研软解里唯一需要线程的地方）。
//!
//! ## 为什么只有滤波器能并行
//!
//! 熵解码是**一条算术流**（每个符号的解码依赖前一个符号的自适应状态），
//! 所以"按块重建"这件事天生是串行的；帧间预测又依赖已经重建好的参考帧。
//! 真正天然无依赖的是三级环路滤波之后的那几段：
//!
//! | 阶段 | 读 | 写 | 能不能并行 |
//! |---|---|---|---|
//! | 去块 | 就地 | 就地 | ✗ 相邻块会互相读（规范顺序要求） |
//! | CDEF | `CurrFrame`（去块后） | `CdefFrame`（另一份缓冲） | ✓ 每个 64×64 组独立 |
//! | 环路恢复 | `CdefFrame` / `CurrFrame` | **第三份**输出缓冲 | ✓ 每个 4×4 块独立 |
//! | 上采样 | 输入一份 | 另一份 | ✓ 行独立 |
//!
//! 也就是说：**只要读写不是同一份缓冲，就能按行切**。这里的做法就是
//! "把输出缓冲按行切成 N 段、每段一个线程"，而不是去动解码主循环。
//!
//! ## 为什么用 `unsafe` 的共享写句柄
//!
//! 输出缓冲是一整块 `Vec<u16>`（带 `stride`），而写入位置由"这个块在平面里的
//! 坐标"决定 —— 想用安全的 `split_at_mut` 表达，就得把几乎所有滤波函数的签名
//! 从 `&mut CurrFrame` 改成"行切片"，那是一次大改。这里改用显式的
//! [`RowsSink`]（裸指针 + 行区间）并**只允许写自己那段行**：
//! 每个 worker 拿到的 `[y0, y1)` 互不相交，写入的块也必然落在自己那段里
//! （调用方按 MI 行切分，块的像素行是这个范围的子集）。不变量写在 `RowsSink` 上。

/// 线程数：默认取机器并行度，可用 `ME_THREADS=n` 覆盖（1 = 退回串行，便于对拍）。
pub(crate) fn worker_count() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        if let Some(v) = std::env::var("ME_THREADS").ok().and_then(|v| v.parse::<usize>().ok()) {
            return v.max(1);
        }
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    })
}

/// **只写自己那段行**的输出缓冲句柄。
///
/// 由调用方保证：不同 worker 的 `[y0, y1)` 不重叠，且所有写入的 `y` 落在
/// `[y0, y1)` 之内，写出的 `x` 落在 `[0, w)` 之内。
pub(crate) struct RowsSink {
    ptr: *mut u16,
    stride: usize,
    cap_rows: usize,
    y0: usize,
    y1: usize,
    bit_depth: u8,
}

// SAFETY: 见类型文档 —— 每个 worker 只碰自己那段行的 `[0, stride)` 列，区域互不相交。
unsafe impl Send for RowsSink {}
unsafe impl Sync for RowsSink {}

impl RowsSink {
    /// 从**整块平面缓冲**的基址建一个"只管 `[y0, y1)` 行"的写句柄。
    ///
    /// 注意 `set()` 用的是**绝对行号**，所以句柄必须持平面缓冲的**基址**
    /// （而不是某段的切片起点）—— 这条踩过坑：指向切片起点时，偏移会被重复
    /// 计算一次（写第 544 行实际落在第 1088 行），多线程下直接段错误。
    ///
    /// # Safety
    /// 调用方必须保证：
    /// 1. `base .. base + len` 是一块有效的可写缓冲，`len == stride * cap_rows`；
    /// 2. 同一块缓冲上**任意两个 `RowsSink` 的 `[y0, y1)` 互不重叠**，
    ///    且所有写入的 `y` 都落在自己的 `[y0, y1)` 内。
    pub(crate) unsafe fn from_base(
        base: *mut u16,
        len: usize,
        stride: usize,
        y0: usize,
        y1: usize,
        bit_depth: u8,
    ) -> Self {
        let cap_rows = len / stride.max(1);
        debug_assert!(y1 <= cap_rows && y0 <= y1, "RowsSink 的行区间不合法");
        Self {
            ptr: base,
            stride,
            cap_rows,
            y0,
            y1,
            bit_depth,
        }
    }

    /// 写一个样本。`x`/`y` 是**平面坐标**（不是相对本段的）。
    #[inline(always)]
    pub(crate) fn set(&self, x: usize, y: usize, v: i32) {
        debug_assert!(y >= self.y0 && y < self.y1, "RowsSink 写到别人那段行了");
        debug_assert!(y < self.cap_rows && x < self.stride, "RowsSink 写到平面缓冲外了");
        let max = (1i32 << self.bit_depth) - 1;
        // SAFETY: 见 `from_base` 的不变量。
        unsafe {
            *self.ptr.add(y * self.stride + x) = v.clamp(0, max) as u16;
        }
    }

    /// 一次写**连续一段**样本（SIMD 路径用：整块滤波后按行落盘）。
    ///
    /// 与 [`Self::set`] 同一套不变量，只是把"逐样本函数调用"换成一次起址 + 内层循环
    /// —— CDEF 一行 8 个像素逐次调 `set` 的话，边界比较与钳位会各做 8 遍。
    #[inline]
    pub(crate) fn set_row(&self, x: usize, y: usize, vals: &[i32]) {
        debug_assert!(y >= self.y0 && y < self.y1, "RowsSink 写到别人那段行了");
        debug_assert!(y < self.cap_rows && x + vals.len() <= self.stride, "RowsSink 写出行容量外了");
        let max = (1i32 << self.bit_depth) - 1;
        // SAFETY: 见 `from_base` 的不变量。
        unsafe {
            let base = self.ptr.add(y * self.stride + x);
            for (i, v) in vals.iter().enumerate() {
                *base.add(i) = (*v).clamp(0, max) as u16;
            }
        }
    }
    /// 同 [`Self::set_row`]，但样本是 **i16**（[`super::simd::cdef_pair16`] 的 16 通道
    /// 内核产出：两行 × 8 列一次算完）。
    ///
    /// ⚠️ **不做逐样本钳位** —— 内核已经过 `Clip3(mn, mx, ·)`，而 `mn`/`mx` 都是合法
    /// 样本 ⇒ 值必在 `[0, 2^bd)` 内（debug 构建里有断言兜着）。整行一次 `memcpy`，
    /// 而不是"取/比较/截断/存"四步的循环：CDEF 每帧要写 ~78 万行 × 8 个样本。
    #[inline]
    pub(crate) fn set_row_i16(&self, x: usize, y: usize, vals: &[i16]) {
        debug_assert!(y >= self.y0 && y < self.y1, "RowsSink 写到别人那段行了");
        debug_assert!(y < self.cap_rows && x + vals.len() <= self.stride, "RowsSink 写出行容量外了");
        debug_assert!(
            vals.iter().all(|v| *v >= 0 && i32::from(*v) < (1i32 << self.bit_depth)),
            "set_row_i16 拿到越界样本（调用方应当已钳位）"
        );
        // SAFETY: 见 `from_base` 的不变量；`i16` 与 `u16` 是同一块内存的两种读法
        //（值是 0..2^bd，最高位恒为 0）。
        unsafe {
            let base = self.ptr.add(y * self.stride + x) as *mut i16;
            std::ptr::copy_nonoverlapping(vals.as_ptr(), base, vals.len());
        }
    }
}

/// 按"每个 worker 的行区间"把一帧的**所有平面**切成互不重叠的写句柄。
///
/// `ranges[k][p]` = 第 k 个 worker 在第 p 个平面里的**结束行**（闭开区间的上界；
/// 下界由"上一个 worker 的结束行"接力得到，所以调用方只要保证单调递增即可）。
/// 返回 `sinks[worker][plane]`。
pub(crate) fn split_frame_rows(
    planes: &mut [super::intra::Plane],
    ranges: &[Vec<usize>],
) -> Vec<Vec<RowsSink>> {
    let workers = ranges.len();
    let mut out: Vec<Vec<RowsSink>> =
        (0..workers).map(|_| Vec::with_capacity(planes.len())).collect();
    for (p, plane) in planes.iter_mut().enumerate() {
        let stride = plane.stride;
        let cap_rows = plane.sample_count() / stride; // 含底部的 border padding
        let bd = plane.bit_depth;
        let base = plane.px16_mut().as_mut_ptr();
        let len = plane.sample_count();
        let mut cursor = 0usize;
        for (k, r) in ranges.iter().enumerate() {
            let end = r[p].min(cap_rows).max(cursor);
            // SAFETY: 各 worker 的行区间由 `cursor` 接力得到，两两不重叠；
            // `base`/`len` 是**整块**平面缓冲（`set` 用绝对行号，必须给基址）。
            out[k].push(unsafe { RowsSink::from_base(base, len, stride, cursor, end, bd) });
            cursor = end;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::intra::Plane;

    /// `split_frame_rows` 给的写句柄必须"各管自己那段行、合起来覆盖整帧"。
    #[test]
    fn split_frame_rows_covers_every_row_exactly_once() {
        let mut planes = vec![Plane::new(8, 4, 8), Plane::new(4, 2, 8)];
        // 两个 worker：luma 行 [0,3) / [3,4)；色度行 [0,1) / [1,2)
        let ranges = vec![vec![3usize, 1usize], vec![4usize, 2usize]];
        let sinks = split_frame_rows(&mut planes, &ranges);
        assert_eq!(sinks.len(), 2);
        assert_eq!(sinks[0].len(), 2);
        // 各写一个样本：写进各自那段行的第 0 行
        sinks[0][0].set(0, 0, 10);
        sinks[1][0].set(0, 3, 20);
        sinks[0][1].set(0, 0, 30);
        sinks[1][1].set(0, 1, 40);
        let luma = &planes[0];
        assert_eq!(luma.px16()[0], 10);
        assert_eq!(luma.px16()[3 * luma.stride], 20);
        let chroma = &planes[1];
        assert_eq!(chroma.px16()[0], 30);
        assert_eq!(chroma.px16()[1 * chroma.stride], 40);
    }

    /// 写值仍然照位深钳位（并行只是换了写路径，语义不能变）。
    #[test]
    fn rows_sink_clamps_to_bit_depth() {
        let mut planes = vec![Plane::new(4, 2, 8)];
        let ranges = vec![vec![2usize]];
        let sinks = split_frame_rows(&mut planes, &ranges);
        sinks[0][0].set(1, 0, 300);
        sinks[0][0].set(2, 1, -5);
        assert_eq!(planes[0].px16()[1], 255);
        assert_eq!(planes[0].px16()[planes[0].stride + 2], 0);
    }

    /// 线程数：`ME_THREADS` 只是覆盖，取值必须至少 1。
    #[test]
    fn worker_count_is_at_least_one() {
        assert!(worker_count() >= 1);
    }
}
