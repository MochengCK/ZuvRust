//! 帧缓冲的**复用池**（线程局部）。
//!
//! ## 为什么需要
//!
//! 一帧 1080p 的三个平面加起来 ≈ 6 MB。软解每帧要开好几份帧缓冲
//! （`CdefFrame`、环路恢复的输出……），如果每次都 `vec![…]` 新开一块：
//!
//! 1. 大分配直接落到 `mmap`（glibc/macOS 的 malloc 都这样），**每一页首次写入
//!    都要缺页中断 + 内核清零** —— 6 MB 就是 1500 页；
//! 2. 用完 `munmap` 还回去，下一次再 mmap 再来一遍。
//!
//! `sample` 采样里这一块的账很清楚：`_platform_bzero`（新建时那次填充）
//! + `madvise`（还内存）+ `_kernelrpc_mach_vm_deallocate_trap` 三项加起来 ~7%，
//! 而且**内存带宽被缺页打断**，让 memmove 看起来也变慢了。
//!
//! ## 做法
//!
//! 就一件事：**按长度回收 `Vec<u16>`**，`Plane` 的 `new`/`clone` 优先从池里拿，
//! `Drop` 时还回去（见 `intra.rs` 里 `Plane` 的三个 impl）。池是**线程局部**的，
//! 没有锁；解码主线程拿/还，滤波的 worker 只按裸指针写已存在的缓冲、不分配。
//!
//! ⚠️ 两条约定：
//! * `take` 回来的缓冲里是**上一帧的残留**（不是零、也不是中灰）——调用方必须自己
//!   写满（`Plane::new` 填中灰、`Plane::clone` 整份 `copy_from_slice`）。
//! * 池有**上限**（[`MAX_BYTES`]）：再多就该让分配器回收，否则等于自己造了个泄漏。
//!   分辨率/尺寸对不上的缓冲会自然被挤出去。

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

/// 命中/落空计数（`AV1_TRACE_POOL=1` 时打印）。
///
/// **为什么用计数而不是计时**：本机墙钟噪声能到 ±40%，"命中率"是**确定性**的量，
/// 能回答"池到底有没有在工作"这个计时回答不了的问题（2026-10-04）。
pub(crate) static HITS: AtomicU64 = AtomicU64::new(0);
pub(crate) static MISSES: AtomicU64 = AtomicU64::new(0);
pub(crate) static GAVE: AtomicU64 = AtomicU64::new(0);

/// `(命中, 落空, 归还, 池内字节, 池内块数)`。
pub(crate) fn stats() -> (u64, u64, u64, usize, usize) {
    let (mut b, mut n) = (0usize, 0usize);
    POOL.with(|p| {
        let p = p.borrow();
        b = p.bytes;
        n = p.bufs.len();
    });
    (
        HITS.load(Ordering::Relaxed),
        MISSES.load(Ordering::Relaxed),
        GAVE.load(Ordering::Relaxed),
        b,
        n,
    )
}

/// 池的记账：块列表 + 总字节。上限按**字节**而不是块数 —— 1080p 的 luma 平面
/// ≈ 4.5 MB、chroma ≈ 1.2 MB；只按块数封顶的话，能存几块取决于分辨率：
/// 小尺寸占死名额、大尺寸一块就爆。
#[derive(Default)]
struct Pool {
    bufs: Vec<Vec<u16>>,
    bytes: usize,
}

impl Pool {
    fn take(&mut self, len: usize) -> Option<Vec<u16>> {
        let i = self.bufs.iter().position(|v| v.len() == len)?;
        let v = self.bufs.swap_remove(i);
        // ⚠️ 必须把字节数也减掉：不然"取出去再还回来"会**重复计数**，
        // 池的字节账只增不减 ⇒ 跑一阵子之后 `give` 全部被拒，池等于失效。
        self.bytes -= v.len() * std::mem::size_of::<u16>();
        Some(v)
    }

    /// 放不下的直接丢掉（让分配器回收），否则等于自己造了个泄漏。
    fn give(&mut self, v: Vec<u16>) {
        let n = v.len() * std::mem::size_of::<u16>();
        if n == 0 {
            return; // 空的（比如 `mem::take` 之后的壳），没什么可回收的
        }
        if self.bytes + n <= max_bytes() {
            self.bytes += n;
            self.bufs.push(v);
        }
    }
}

/// [`Pool`] 的 8bit 版（契约逐条相同，只是元素类型不同）。
#[derive(Default)]
struct Pool8 {
    bufs: Vec<Vec<u8>>,
    bytes: usize,
}

impl Pool8 {
    fn take(&mut self, len: usize) -> Option<Vec<u8>> {
        let i = self.bufs.iter().position(|v| v.len() == len)?;
        let v = self.bufs.swap_remove(i);
        self.bytes -= v.len();
        Some(v)
    }

    fn give(&mut self, v: Vec<u8>) {
        if v.is_empty() {
            return;
        }
        if self.bytes + v.len() <= max_bytes() {
            self.bytes += v.len();
            self.bufs.push(v);
        }
    }
}

thread_local! {
    /// 每块 `Vec<u16>` 的**长度**就是它当初服务过的平面缓冲长度。
    static POOL: RefCell<Pool> = const { RefCell::new(Pool { bufs: Vec::new(), bytes: 0 }) };
    /// 8bit 平面的池（`Vec<u8>`）：与 `POOL` 同一套约定，只是元素宽度不同。
    static POOL8: RefCell<Pool8> = const { RefCell::new(Pool8 { bufs: Vec::new(), bytes: 0 }) };
}

/// 池里最多留多少**字节**（u16 池与 u8 池各自算）。
///
/// 2026-10-04 之前是「最多 8 块」。**8 块根本不够**：稳态下活着的缓冲有
/// 「参考帧槽（最多 8 帧 × 3 平面）」+「本帧的解码帧 / CDEF 帧 / LR 输出」，
/// 1080p 下超过 20 块 ⇒ 池每帧都在 miss，`vec![0u16; n]` 重新走 mmap，
/// **每个平面第一页都要缺页中断**。改成按字节封顶（96 MB）后同一分辨率下能存下
/// 全部工作缓冲。
const MAX_BYTES: usize = 96 << 20;

/// 池上限（字节）。`ME_POOL_MAX_BYTES` 可临时改它 —— 用来在**确定性**的条件下
/// 对比不同上限的命中率（本机墙钟噪声太大，计时回答不了"池够不够"这个问题）。
pub(crate) fn max_bytes() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ME_POOL_MAX_BYTES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(MAX_BYTES)
    })
}

/// 取一块长度**恰好** `len` 的缓冲；池里没有就返回 `None`（调用方自己新开）。
///
/// 拿到的缓冲内容是**残留数据**，调用方负责写满 —— 这正是这个池的价值所在：
/// 省掉的不是那次写，而是"新开一块"带来的缺页与内核清零。
pub(crate) fn take(len: usize) -> Option<Vec<u16>> {
    if len == 0 {
        return None;
    }
    POOL.with(|p| {
        let r = p.borrow_mut().take(len);
        if r.is_some() {
            HITS.fetch_add(1, Ordering::Relaxed);
        } else {
            MISSES.fetch_add(1, Ordering::Relaxed);
        }
        r
    })
}

/// 归还一块缓冲。
///
/// 长度对不上的留在池里也不会被 `take` 选中（`take` 要求长度**恰好**相等），
/// 最终被 [`MAX_BYTES`] 挤掉 —— 不需要在这里做精确的尺寸分桶。
pub(crate) fn give(v: Vec<u16>) {
    if v.capacity() == 0 {
        return; // 空的（比如 `mem::take` 之后的壳），没什么可回收的
    }
    POOL.with(|p| {
        GAVE.fetch_add(1, Ordering::Relaxed);
        p.borrow_mut().give(v)
    });
}

/// [`take`] 的 u8 版（8bit 平面的缓冲；契约完全相同）。
pub(crate) fn take_u8(len: usize) -> Option<Vec<u8>> {
    if len == 0 {
        return None;
    }
    POOL8.with(|p| p.borrow_mut().take(len))
}

/// [`give`] 的 u8 版。
pub(crate) fn give_u8(v: Vec<u8>) {
    if v.capacity() == 0 {
        return;
    }
    POOL8.with(|p| p.borrow_mut().give(v));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 还回去的缓冲必须能被同长度的 `take` 拿回来（复用池的基本契约）。
    #[test]
    fn give_then_take_round_trips_same_length() {
        let v = vec![7u16; 1000];
        give(v);
        let got = take(1000).expect("刚还回去的缓冲应该能取回来");
        assert_eq!(got.len(), 1000);
        assert_eq!(got[999], 7, "复用池不做清零，残留数据原样保留");
        // 换一个长度就取不到了（避免"拿错尺寸"这类隐性 bug）
        assert!(take(999).is_none());
    }

    /// 反复「取 → 还」**不能**把字节账耗光（`bytes` 只增不减的老 bug 就是这么漏的：
    /// 跑一阵子之后 `give` 全被拒，池静默失效 —— 从外面看只是"慢慢变慢"）。
    #[test]
    fn pool_accounting_survives_repeated_take_and_give() {
        let n = 1 << 20; // 2 MB（u16）
        for i in 0..200 {
            let v = take(n).unwrap_or_else(|| vec![0u16; n]);
            assert!(take(n).is_none(), "池里不该有第二块同长度的缓冲（第 {i} 轮）");
            give(v);
        }
        assert!(take(n).is_some(), "200 轮之后池应当还能交出同一块缓冲");
    }

    /// 池有上限：一直还也不该无限增长（上限按**字节**，不是块数）。
    #[test]
    fn pool_is_bounded() {
        for _ in 0..64 {
            give(vec![1u16; 8 << 20]);
        }
        let mut got = 0usize;
        while take(8 << 20).is_some() {
            got += 1;
            assert!(got < 64, "上限没生效");
        }
        assert!(got <= max_bytes() / (8 << 20), "留的字节数超过上限");
    }
}
