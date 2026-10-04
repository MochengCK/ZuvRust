//! 片存储：把 BT 的"片"映射成上层容器解析器要的"可随机访问的字节流"。
//!
//! 这是**边下边播真正落地的地方**：容器解析器拿到的 [`me_core::ReadAt`] 就是
//! [`PieceStore`]。它和普通内存源的区别只有一条：
//!
//! > **读不到的区间不是错误，是"现在还没有"。** [`PieceStore::read_at`] 会老老实实
//! > 返回"从 offset 开始连续能给你多少字节"（可能是 0），而不是报错。
//! > 播放器据此决定"停一下等缓冲"，BT 层据此知道"播放头卡在这里，赶紧下这一段"。
//!
//! 三个设计决定，理由都写在各自的位置：
//!
//! 1. **每个片一个独立分配的槽位，而不是一整块 `total` 大小的缓冲。**
//!    见 [`PieceStore`] 的字段注释 —— 一整块缓冲意味着 40GB 的种子要 40GB 内存，
//!    而且"淘汰"根本无从下手（`Vec` 不会把中间的字节还给操作系统）。
//! 2. **内部可变性用 `&mut self`，不加内部锁。** 理由见 [`PieceStore`] 的说明。
//! 3. **[`PieceStore::contiguous_prefix`] 是"单调水位线"，不是"当前可用区间"。**
//!    回收（[`PieceStore::evict_before`]）**不会**让它倒退 —— 理由见该方法的注释。

use me_core::error::{MediaError, Result};
use me_core::source::ReadAt;

/// 片在逻辑字节流里的布局（纯计算，无状态）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PieceLayout {
    pub piece_length: u64,
    pub total: u64,
    pub num_pieces: u32,
}

impl PieceLayout {
    /// `piece_length` 为 0 时会被钳成 1：宁可给出一个"每片 1 字节"的退化布局，
    /// 也不要在这里 `panic` 或者让后面每个除法都成为除零陷阱。
    pub fn new(piece_length: u64, total: u64) -> Self {
        let piece_length = piece_length.max(1);
        // 手写向上取整而不是 div_ceil：语义一样，但对最低支持的编译器版本不敏感
        let n = total / piece_length + u64::from(total % piece_length != 0);
        let num_pieces = n.min(u32::MAX as u64) as u32;
        Self {
            piece_length,
            total,
            num_pieces,
        }
    }

    pub fn piece_offset(&self, index: u32) -> u64 {
        index as u64 * self.piece_length
    }

    /// 第 `index` 片的长度。**最后一片通常短于 `piece_length`**。
    pub fn piece_size(&self, index: u32) -> u32 {
        if index >= self.num_pieces {
            return 0;
        }
        let start = self.piece_offset(index);
        let remain = self.total.saturating_sub(start);
        remain.min(self.piece_length).min(u32::MAX as u64) as u32
    }

    /// 某个字节偏移属于哪一片。
    ///
    /// 越界（`offset >= total`）时返回**最后一片**而不是报错：调用方（读路径）
    /// 是在"读到末尾了"的边界上调用它的，报错会逼着每个调用点都写一遍范围判断。
    pub fn piece_index_of(&self, offset: u64) -> u32 {
        if self.num_pieces == 0 {
            return 0;
        }
        let idx = offset / self.piece_length;
        idx.min(self.num_pieces as u64 - 1) as u32
    }

    /// `[start, start+len)` 覆盖到的所有片号（含首尾，升序）。
    pub fn pieces_for_range(&self, start: u64, len: u64) -> Vec<u32> {
        if len == 0 || self.num_pieces == 0 {
            return Vec::new();
        }
        let first = self.piece_index_of(start);
        let last = self.piece_index_of(start.saturating_add(len - 1));
        (first..=last).collect()
    }

    pub fn last_piece(&self) -> u32 {
        self.num_pieces.saturating_sub(1)
    }
}

/// 片的位图（BT 线协议形态：第 0 片是第 0 字节的最高位）。
///
/// 两个坑都处理在这一层：
/// - **末字节的补齐位**：`num_pieces` 不是 8 的倍数时，最后一个字节里有几位
///   "不存在"。对面（或我们自己）把这几位置 1 的话，`count()` 会多算，
///   表现为"声称有 20 片，实际只有 19 片"，最后一片永远下不到。
///   所以 [`Bitfield::from_bytes`] 会**主动把补齐位清 0**。
/// - `count` **增量维护**：每次 `count()` 都去数一遍位图在 20 万片的种子上是
///   每帧几十 KB 的扫描，没必要。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bitfield {
    bits: Vec<u8>,
    num_pieces: u32,
    count: u32,
}

impl Bitfield {
    pub fn new(num_pieces: u32) -> Self {
        Self {
            bits: vec![0u8; Self::byte_len(num_pieces)],
            num_pieces,
            count: 0,
        }
    }

    fn byte_len(num_pieces: u32) -> usize {
        num_pieces.div_ceil(8) as usize
    }

    pub fn num_pieces(&self) -> u32 {
        self.num_pieces
    }

    pub fn get(&self, index: u32) -> bool {
        if index >= self.num_pieces {
            return false;
        }
        let byte = (index / 8) as usize;
        let mask = 0x80u8 >> (index % 8);
        self.bits[byte] & mask != 0
    }

    pub fn set(&mut self, index: u32, v: bool) {
        if index >= self.num_pieces {
            return;
        }
        let byte = (index / 8) as usize;
        let mask = 0x80u8 >> (index % 8);
        let was = self.bits[byte] & mask != 0;
        if was == v {
            return;
        }
        if v {
            self.bits[byte] |= mask;
            self.count += 1;
        } else {
            self.bits[byte] &= !mask;
            self.count -= 1;
        }
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    /// BT 线协议形态：高位在前，末字节的补齐位恒为 0。
    pub fn to_bytes(&self) -> Vec<u8> {
        self.bits.clone()
    }

    /// 从线协议字节还原。
    ///
    /// `num_pieces` 必须是**真实的**片总数；`b` 可以比需要的长（多余字节丢弃），
    /// 但短了就是错误 —— 短了说明对面发的位图与它的片数不自洽。
    pub fn from_bytes(b: &[u8], num_pieces: u32) -> Result<Self> {
        let need = Self::byte_len(num_pieces);
        if b.len() < need {
            return Err(MediaError::protocol(format!(
                "位图太短：{num_pieces} 片需要 {need} 字节，实际 {}",
                b.len()
            )));
        }
        let mut bits = b[..need].to_vec();
        // 清掉补齐位：不这么做的话，`Bitfield(0xff)` + 4 片会被算成 8 片
        if !num_pieces.is_multiple_of(8) {
            if let Some(last) = bits.last_mut() {
                let used = (num_pieces % 8) as u8;
                *last &= 0xffu8 << (8 - used);
            }
        }
        let count = bits.iter().map(|b| b.count_ones()).sum();
        Ok(Self {
            bits,
            num_pieces,
            count,
        })
    }

    pub fn is_complete(&self) -> bool {
        self.count == self.num_pieces
    }
}

/// 内存片存储。**实现 [`ReadAt`]，可以直接喂给容器解析器。**
///
/// ## 为什么是"每片一个 `Option<Vec<u8>>`"而不是一整块 `Vec<u8>`
///
/// 一整块按 `total` 分配的缓冲有两个致命问题：
///
/// - **内存**：一个 40GB 的种子会直接要 40GB 内存。BT 边下边播的常见场景是
///   "用户点开一个 4K 大片，只看前 10 分钟"，为它预留全片内存是不可接受的。
/// - **淘汰**：`Vec` 无法把中间的字节还给操作系统。那么
///   [`PieceStore::evict_before`] 就退化成一个"只改标志位、不省内存"的假动作，
///   而这个函数的全部意义就是"内存有上限"。所以槽位必须能真正被释放，
///   也就是必须**分开分配**（`Option<Vec<u8>>`：`take()` 掉就真的还回去了）。
///
/// 每个槽位的长度恒等于 `piece_size(index)`（最后一片就是它真实的短长度），
/// 这样 `read_at` 里的切片不用再判断边界。
///
/// ## 为什么内部可变性用 `&mut self` 而不是内部 `RwLock`
///
/// 写路径（`write_piece` / `evict_before` / `mark_have_unchecked`）天然是"独占改状态"，
/// 读路径（`read_at`）是 `&self`。两种做法都可行：
///
/// - 内部 `RwLock`：`PieceStore` 自己就能跨线程共享，但每次 `read_at`
///   （播放器**每帧**都在调）都要过一次锁，而且"加锁"和"版图变更"两套并发规则
///   会同时存在，读代码的人要同时想两件事。
/// - `&mut self`（**本实现选的**）：编译器保证"有可变借用时没有任何读者"。
///   外层把 `PieceStore` 包成 `Arc<RwLock<PieceStore>>`，写的时候拿写锁、
///   读的时候拿读锁 —— 锁语义只有一层，而且读路径在锁内是纯内存拷贝，没有嵌套锁。
///
/// 代价是 `PieceStore` 本身不提供内部并发；这是**刻意**的：
/// 本层不做线程调度（那是外层 `me-stream` 的事），这里只保证"能在 `&self` 下读、
/// 能在 `&mut self` 下改"。`Send + Sync` 由字段类型自动满足（有测试守着）。
pub struct PieceStore {
    layout: PieceLayout,
    piece_hashes: Vec<[u8; 20]>,
    have: Bitfield,
    /// 见类型文档：每片独立分配，`None` = 这片当前不在内存里。
    slots: Vec<Option<Vec<u8>>>,
    /// 已就绪的**连续前缀**长度（单调水位线，见 [`PieceStore::evict_before`]）。
    contiguous: u64,
    /// 当前在内存里的字节数（`have` 为真且槽位存在的片之和）。增量维护，别每次重算。
    bytes_have: u64,
}

impl PieceStore {
    /// `hashes.len()` 与 `layout.num_pieces` 不一致时**不报错**（签名没有 `Result`），
    /// 而是补齐/截断并记一条 warning：缺的那些片因为哈希是 `[0;20]`，
    /// 校验必然失败，所以是"安全地永远下不到"，不会引入坏数据。
    pub fn new(layout: PieceLayout, hashes: Vec<[u8; 20]>) -> Self {
        let n = layout.num_pieces as usize;
        let mut piece_hashes = hashes;
        if piece_hashes.len() != n {
            tracing::warn!(
                expect = n,
                got = piece_hashes.len(),
                "片哈希数量与片数不一致，缺失的位置用零哈希填充（这些片将永远校验失败）"
            );
            piece_hashes.resize(n, [0u8; 20]);
        }
        Self {
            layout,
            piece_hashes,
            have: Bitfield::new(layout.num_pieces),
            slots: (0..n).map(|_| None).collect(),
            contiguous: 0,
            bytes_have: 0,
        }
    }

    pub fn layout(&self) -> &PieceLayout {
        &self.layout
    }

    pub fn have(&self, index: u32) -> bool {
        self.have.get(index)
    }

    /// 写入一整片。
    ///
    /// **必须校验 SHA-1，不通过就直接丢掉**。不校验的话，一个恶意/故障 peer
    /// 随便塞一段字节进来，最终表现是"播放到某个位置突然花屏，
    /// 而且用户重播一次还是花屏"—— 因为坏数据已经落盘了。
    /// 宁可多花一次 SHA-1（几十 MB/s 的量级，完全不是瓶颈），也不要放坏数据进去。
    pub fn write_piece(&mut self, index: u32, data: &[u8]) -> Result<()> {
        if index >= self.layout.num_pieces {
            return Err(MediaError::protocol(format!(
                "片号 {index} 越界（共 {} 片）",
                self.layout.num_pieces
            )));
        }
        if self.have.get(index) {
            // 已经有的片再收一遍：直接接受（可能是并发重复请求的迟到应答），
            // 不重复写、也不报错。
            return Ok(());
        }
        let expect = self.layout.piece_size(index) as usize;
        if data.len() != expect {
            return Err(MediaError::protocol(format!(
                "第 {index} 片长度不对：期望 {expect}，实得 {}",
                data.len()
            )));
        }
        let got = crate::sha1(data);
        if got != self.piece_hashes[index as usize] {
            return Err(MediaError::protocol(format!(
                "第 {index} 片 SHA-1 校验失败（数据已丢弃）"
            )));
        }
        let size = expect as u64;
        self.slots[index as usize] = Some(data.to_vec());
        self.have.set(index, true);
        self.bytes_have += size;
        self.extend_contiguous();
        Ok(())
    }

    /// 标记一片为"已拥有"（调用方保证它已经校验过）。
    ///
    /// **注意**：这条路径只能记帐，没有真数据可放，所以槽位会被填零。
    /// 它存在的场景是"片已经在磁盘上/别处，内存副本由外层另外提供"；
    /// 如果你的字节就在手上，**请用 [`PieceStore::write_piece`]**，
    /// 否则读出来的是零字节（`read_at` 会"成功"返回全 0，比报错更难查）。
    pub fn mark_have_unchecked(&mut self, index: u32) {
        if index >= self.layout.num_pieces || self.have.get(index) {
            return;
        }
        let size = self.layout.piece_size(index) as usize;
        self.slots[index as usize] = Some(vec![0u8; size]);
        self.have.set(index, true);
        self.bytes_have += size as u64;
        self.extend_contiguous();
    }

    pub fn bitfield(&self) -> &Bitfield {
        &self.have
    }

    /// **已就绪的连续前缀**：从第 0 字节起一直连续到哪。
    ///
    /// 边下边播的"能播到第几字节"就是它：容器解析器只有拿到连续字节才能解析，
    /// 中间有空洞时读请求会短读。
    ///
    /// 它是一条**单调水位线**：只会前进，[`PieceStore::evict_before`] 不会让它倒退。
    /// 原因是这个值会被"已经播过的播放器"用来判断"我还要不要重读"，
    /// 水位线倒退会让播放器以为自己读到的地方又没数据了 → 花屏；
    /// 而"水位线以内某片被回收了"是**可以容忍**的（那部分已经播过去了），
    /// 真要被回读时 `read_at` 会短读，外层可以按 [`PieceStore::missing`] 重新排程。
    pub fn contiguous_prefix(&self) -> u64 {
        self.contiguous
    }

    /// 当前在内存里、确实可读的所有区间，`(start, end)` 半开区间，升序。
    pub fn ready_ranges(&self) -> Vec<(u64, u64)> {
        let mut out = Vec::new();
        let mut i = 0u32;
        while i < self.layout.num_pieces {
            if self.have.get(i) && self.slots[i as usize].is_some() {
                let start = self.layout.piece_offset(i);
                let mut end = start + self.layout.piece_size(i) as u64;
                let mut j = i + 1;
                while j < self.layout.num_pieces
                    && self.have.get(j)
                    && self.slots[j as usize].is_some()
                {
                    end += self.layout.piece_size(j) as u64;
                    j += 1;
                }
                out.push((start, end));
                i = j;
            } else {
                i += 1;
            }
        }
        out
    }

    /// 当前缺哪些片（升序）。**包含被 [`PieceStore::evict_before`] 回收掉的片** ——
    /// 回收之后它们确实又"没数据"了，外层据此重新调度。
    pub fn missing(&self) -> Vec<u32> {
        (0..self.layout.num_pieces)
            .filter(|&i| !self.have.get(i))
            .collect()
    }

    /// 当前在内存里的字节数。
    pub fn bytes_have(&self) -> u64 {
        self.bytes_have
    }

    /// `[start, start+len)` 覆盖的片里，第一个还没拿到的片号。
    pub fn first_missing_piece_in(&self, start: u64, len: u64) -> Option<u32> {
        self.layout
            .pieces_for_range(start, len)
            .into_iter()
            .find(|&i| !self.have.get(i))
    }

    /// 回收播放头之前已经播完的片，返回释放的片数。
    ///
    /// ## 保守策略（宁可少淘汰，绝不让前缀倒退）
    ///
    /// 只回收**完整落在 `limit = min(offset, contiguous_prefix)` 之前**的片，即
    /// `piece_offset(i) + piece_size(i) <= limit`。这条规则同时挡住两种情况：
    ///
    /// - `offset` 还没走到（播放头在片中间）→ 不动它，因为播放器下一秒可能还要读这一片；
    /// - 与连续前缀的末梢相交 → 不动它，因为那是"刚下好、马上要给播放器"的部分。
    ///
    /// 反过来，"前缀之后还没下到的片"本来就没数据可回收，
    /// "安静很久的老片"这里的判断依据只有播放头 —— 所以能少回收就少回收，
    /// 内存压力交给外层的策略决定（它可以调更激进的 `offset`）。
    ///
    /// **前缀不倒退**：`contiguous` 字段只增不减，回收不碰它。
    /// 回收只清 `have` 位与槽位，于是这些片重新出现在 [`PieceStore::missing`] 里，
    /// 外层如果发现播放器要回读，可以重新把它们下回来。
    pub fn evict_before(&mut self, offset: u64) -> usize {
        let limit = offset.min(self.contiguous);
        let mut freed = 0usize;
        for i in 0..self.layout.num_pieces {
            if !self.have.get(i) {
                continue;
            }
            let end = self.layout.piece_offset(i) + self.layout.piece_size(i) as u64;
            if end > limit {
                // 片号递增 ⇒ 后面的片只会更靠后，直接收工
                break;
            }
            if self.slots[i as usize].take().is_some() {
                self.have.set(i, false);
                self.bytes_have -= self.layout.piece_size(i) as u64;
                freed += 1;
            }
        }
        freed
    }

    /// 把连续前缀往前推进（只在写入之后调用，**永远只增**）。
    fn extend_contiguous(&mut self) {
        if self.contiguous >= self.layout.total {
            self.contiguous = self.layout.total;
            return;
        }
        let mut idx = (self.contiguous / self.layout.piece_length) as u32;
        while idx < self.layout.num_pieces && self.have.get(idx) {
            self.contiguous += self.layout.piece_size(idx) as u64;
            idx += 1;
        }
        if self.contiguous > self.layout.total {
            self.contiguous = self.layout.total;
        }
    }

    /// 读路径的核心：从 `offset` 起**连续可读**的字节数（到第一个空洞为止）。
    fn read_contiguous(&self, offset: u64, out: &mut [u8]) -> usize {
        if out.is_empty() || offset >= self.layout.total {
            return 0;
        }
        let want = out.len().min((self.layout.total - offset) as usize);
        let mut done = 0usize;
        let mut at = offset;
        while done < want {
            let idx = self.layout.piece_index_of(at);
            let Some(slot) = self.slots.get(idx as usize).and_then(|s| s.as_deref()) else {
                break; // 空洞：就此打住，返回已读到的部分
            };
            if !self.have.get(idx) {
                break; // 理论上不会发生（清位与清槽位是同一个动作），当作空洞处理
            }
            let in_piece = (at - self.layout.piece_offset(idx)) as usize;
            if in_piece >= slot.len() {
                break;
            }
            let n = (slot.len() - in_piece).min(want - done);
            out[done..done + n].copy_from_slice(&slot[in_piece..in_piece + n]);
            done += n;
            at += n as u64;
        }
        done
    }
}

impl ReadAt for PieceStore {
    /// 长度永远是已知的：`.torrent` 里写着总长（这也是 BT 源比 HLS 源好的一点）。
    fn len(&self) -> Option<u64> {
        Some(self.layout.total)
    }

    /// **读还没下到的区间返回 `Ok(已连续读到的字节数)`（可能是 0），不是错误。**
    ///
    /// 这条语义是边下边播的关键：上层想做的是"缓冲不够就等一会儿再读"，
    /// 而不是"把源判成坏的"。真到末尾（`offset >= total`）也是 `Ok(0)` ——
    /// 和"还没下到"在返回值上是一样的，但两者的区别上层用 `len()` 就能分清。
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        Ok(self.read_contiguous(offset, buf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout_and_hashes(data: &[u8], piece_length: u64) -> (PieceLayout, Vec<[u8; 20]>) {
        let layout = PieceLayout::new(piece_length, data.len() as u64);
        let hashes = (0..layout.num_pieces)
            .map(|i| {
                let off = layout.piece_offset(i) as usize;
                let size = layout.piece_size(i) as usize;
                crate::sha1(&data[off..off + size])
            })
            .collect();
        (layout, hashes)
    }

    fn filled(store: &mut PieceStore, data: &[u8]) {
        for i in 0..store.layout().num_pieces {
            let off = store.layout().piece_offset(i) as usize;
            let size = store.layout().piece_size(i) as usize;
            store.write_piece(i, &data[off..off + size]).unwrap();
        }
    }

    #[test]
    fn layout_basics_with_short_last_piece() {
        let l = PieceLayout::new(4, 10);
        assert_eq!(l.num_pieces, 3);
        assert_eq!(l.piece_offset(0), 0);
        assert_eq!(l.piece_offset(2), 8);
        assert_eq!(l.piece_size(0), 4);
        assert_eq!(l.piece_size(2), 2, "最后一片比片长短");
        assert_eq!(l.piece_size(3), 0);
        assert_eq!(l.last_piece(), 2);
        assert_eq!(l.piece_index_of(0), 0);
        assert_eq!(l.piece_index_of(3), 0);
        assert_eq!(l.piece_index_of(4), 1);
        assert_eq!(l.piece_index_of(9), 2);
        assert_eq!(l.piece_index_of(1000), 2, "越界钳到最后一片，不 panic");
        // 区间覆盖
        assert_eq!(l.pieces_for_range(3, 2), vec![0, 1], "跨片边界要算两片");
        assert_eq!(l.pieces_for_range(0, 4), vec![0]);
        assert_eq!(l.pieces_for_range(0, 5), vec![0, 1]);
        assert_eq!(l.pieces_for_range(9, 1), vec![2]);
        assert!(l.pieces_for_range(0, 0).is_empty());

        // 空种子 / 片长为 0 的退化输入
        let empty = PieceLayout::new(4, 0);
        assert_eq!(empty.num_pieces, 0);
        assert_eq!(empty.last_piece(), 0);
        assert!(empty.pieces_for_range(0, 10).is_empty());
        let degenerate = PieceLayout::new(0, 3);
        assert_eq!(degenerate.piece_length, 1);
        assert_eq!(degenerate.num_pieces, 3);
    }

    #[test]
    fn bitfield_spare_bits_are_not_have() {
        // 4 片，末字节高 4 位是补齐位：对面发 0xff 进来也不能算成 8 片
        let bf = Bitfield::from_bytes(&[0xff], 4).unwrap();
        assert_eq!(bf.num_pieces(), 4);
        assert_eq!(bf.count(), 4);
        assert!(bf.is_complete());
        assert!(bf.get(0) && bf.get(3));
        assert!(!bf.get(4) && !bf.get(8), "补齐位必须被清掉");

        // 5 片 + 0xf8 = 前 5 位为 1
        let bf = Bitfield::from_bytes(&[0xf8], 5).unwrap();
        assert_eq!(bf.count(), 5);
        // 5 片 + 0xff 也要得 5
        let bf = Bitfield::from_bytes(&[0xff], 5).unwrap();
        assert_eq!(bf.count(), 5);
        assert_eq!(bf.to_bytes(), vec![0xf8], "to_bytes 也要抹掉补齐位");
    }

    #[test]
    fn bitfield_set_get_count_and_roundtrip() {
        let mut bf = Bitfield::new(10);
        assert_eq!(bf.count(), 0);
        assert!(!bf.is_complete());
        bf.set(0, true);
        bf.set(7, true);
        bf.set(9, true);
        assert_eq!(bf.count(), 3);
        assert_eq!(bf.get(0), true);
        assert_eq!(bf.get(7), true);
        assert_eq!(bf.get(9), true);
        assert_eq!(bf.get(1), false);
        assert_eq!(bf.get(100), false, "越界读是 false 而不是 panic");
        // 重复置位不重复计数
        bf.set(0, true);
        assert_eq!(bf.count(), 3);
        bf.set(0, false);
        assert_eq!(bf.count(), 2);

        let raw = bf.to_bytes();
        assert_eq!(raw.len(), 2);
        // 第 0 片上面已经被清掉了，所以首字节只剩第 7 片那一位
        assert_eq!(raw[0], 0b0000_0001);
        assert_eq!(raw[1], 0b0100_0000);
        assert_eq!(Bitfield::from_bytes(&raw, 10).unwrap(), bf);
        // 多余的尾字节会被丢掉
        assert_eq!(Bitfield::from_bytes(&[raw[0], raw[1], 0xff], 10).unwrap(), bf);
        // 太短则报错
        let e = Bitfield::from_bytes(&[0x00], 10).unwrap_err();
        assert!(matches!(e, MediaError::Protocol(_)), "{e:?}");
        // 越界 set 静默忽略
        bf.set(99, true);
        assert_eq!(bf.count(), 2);
    }

    #[test]
    fn write_piece_verifies_hash_and_rejects_bad_data() {
        let data = b"abcdefghij";
        let (layout, hashes) = layout_and_hashes(data, 4);
        let mut store = PieceStore::new(layout, hashes);
        assert_eq!(store.bytes_have(), 0);
        assert_eq!(store.contiguous_prefix(), 0);

        // 坏数据：内容和哈希对不上，必须丢弃
        let mut bad = b"abcd".to_vec();
        bad[0] = b'X';
        let e = store.write_piece(0, &bad).unwrap_err();
        assert!(matches!(e, MediaError::Protocol(_)), "{e:?}");
        assert!(!store.have(0), "校验失败绝不能写入");
        assert_eq!(store.bytes_have(), 0);
        assert_eq!(store.contiguous_prefix(), 0);

        // 长度不对
        assert!(store.write_piece(0, b"abc").is_err());
        // 片号越界
        assert!(store.write_piece(99, b"abcd").is_err());

        // 正确数据
        store.write_piece(0, b"abcd").unwrap();
        assert!(store.have(0));
        assert_eq!(store.bytes_have(), 4);
        assert_eq!(store.contiguous_prefix(), 4);

        // 乱序写入：前缀只在"接上了"之后才前进
        store.write_piece(2, b"ij").unwrap();
        assert_eq!(store.contiguous_prefix(), 4, "空着的第 1 片挡着，前缀不能跳过去");
        store.write_piece(1, b"efgh").unwrap();
        assert_eq!(store.contiguous_prefix(), 10, "补齐之后前缀一次走到底");
        assert!(store.bitfield().is_complete());
        assert!(store.missing().is_empty());

        // 重复写入同一片：幂等
        store.write_piece(1, b"efgh").unwrap();
        assert_eq!(store.bytes_have(), 10);
    }

    #[test]
    fn read_at_short_read_is_not_an_error() {
        let data = b"abcdefghij";
        let (layout, hashes) = layout_and_hashes(data, 4);
        let mut store = PieceStore::new(layout, hashes);
        // 只有第 0 片
        store.write_piece(0, b"abcd").unwrap();

        let mut buf = [0u8; 10];
        // 从 0 读 10：只给得出第 0 片的 4 字节 —— Ok(4)，不是 Err
        assert_eq!(store.read_at(0, &mut buf).unwrap(), 4);
        assert_eq!(&buf[..4], b"abcd");
        // 从第 1 片开始读：一片都没有 → Ok(0)
        assert_eq!(store.read_at(4, &mut buf).unwrap(), 0);
        // 中间的空洞也一样：给了 0、1 两片后从 2 开始读
        store.write_piece(1, b"efgh").unwrap();
        assert_eq!(store.read_at(0, &mut buf).unwrap(), 8);
        assert_eq!(&buf[..8], b"abcdefgh");
        assert_eq!(store.read_at(8, &mut buf).unwrap(), 0, "最后一片还没下到");

        // 真到末尾是 Ok(0)
        store.write_piece(2, b"ij").unwrap();
        assert_eq!(store.read_at(10, &mut buf).unwrap(), 0);
        assert_eq!(store.read_at(11, &mut buf).unwrap(), 0);
        assert_eq!(store.read_at(0, &mut buf).unwrap(), 10);
        // 末尾短读：要 10 字节只有 2 字节可给
        assert_eq!(store.read_at(8, &mut buf).unwrap(), 2);
        assert_eq!(&buf[..2], b"ij");
        // 空 buf
        assert_eq!(store.read_at(0, &mut []).unwrap(), 0);

        // ReadAt 的默认方法也能用
        assert_eq!(store.len(), Some(10));
        assert!(!store.is_empty());
        assert_eq!(store.read_exact_at(0, 10).unwrap(), data.to_vec());
        assert!(store.read_exact_at(0, 11).unwrap_err().is_need_more());
        assert_eq!(store.read_range(2, 100).unwrap().len(), 8);
    }

    #[test]
    fn ready_ranges_report_holes() {
        let data = b"abcdefghijklmnopqrstuvwxyz"; // 26 字节
        let (layout, hashes) = layout_and_hashes(data, 8); // 8/8/8/2
        assert_eq!(layout.num_pieces, 4);
        let mut store = PieceStore::new(layout, hashes);
        assert!(store.ready_ranges().is_empty());
        assert_eq!(store.missing(), vec![0, 1, 2, 3]);

        store.write_piece(0, b"abcdefgh").unwrap();
        store.write_piece(2, b"qrstuvwx").unwrap();
        assert_eq!(
            store.ready_ranges(),
            vec![(0, 8), (16, 24)],
            "两条区间，中间是空洞"
        );
        assert_eq!(store.missing(), vec![1, 3]);
        assert_eq!(store.first_missing_piece_in(0, 26), Some(1));
        assert_eq!(store.first_missing_piece_in(16, 8), None, "只覆盖到已有的第 2 片");
        assert_eq!(store.first_missing_piece_in(16, 10), Some(3), "跨到第 3 片才有空洞");
        assert_eq!(store.first_missing_piece_in(0, 8), None);
        assert_eq!(store.first_missing_piece_in(0, 0), None);
        assert_eq!(store.bytes_have(), 16);

        // 反序补齐，区间合并
        store.write_piece(3, b"yz").unwrap();
        store.write_piece(1, b"ijklmnop").unwrap();
        assert_eq!(store.ready_ranges(), vec![(0, 26)]);
        assert_eq!(store.contiguous_prefix(), 26);
    }

    #[test]
    fn evict_never_regresses_prefix_and_never_frees_unplayed() {
        let data = b"abcdefghij";
        let (layout, hashes) = layout_and_hashes(data, 4); // 4/4/2
        let mut store = PieceStore::new(layout, hashes);
        filled(&mut store, data);
        assert_eq!(store.contiguous_prefix(), 10);
        assert_eq!(store.bytes_have(), 10);

        // 播放头还在 0：一个字都不该回收
        assert_eq!(store.evict_before(0), 0);
        assert_eq!(store.bytes_have(), 10);
        // 播放头在第 0 片中间：第 0 片是"完整落在 4 之前"，可以回收
        assert_eq!(store.evict_before(4), 1);
        assert_eq!(store.bytes_have(), 6);
        assert_eq!(store.contiguous_prefix(), 10, "前缀绝不能倒退");
        // 前缀没退，但被回收的区域真的读不到了（短读，不是错误）
        let mut buf = [0u8; 10];
        assert_eq!(store.read_at(0, &mut buf).unwrap(), 0);
        assert_eq!(store.read_at(4, &mut buf).unwrap(), 6);
        assert_eq!(store.ready_ranges(), vec![(4, 10)]);
        // 回收掉的片重新变成"缺"
        assert_eq!(store.missing(), vec![0]);

        // 播放头走到末尾：剩下的全回收，前缀仍然不动
        assert_eq!(store.evict_before(10), 2);
        assert_eq!(store.bytes_have(), 0);
        assert!(store.ready_ranges().is_empty());
        assert_eq!(store.contiguous_prefix(), 10);
        assert_eq!(store.missing(), vec![0, 1, 2]);
        // 再回收一次是幂等的
        assert_eq!(store.evict_before(10), 0);

        // 重新下回来：前缀不会因为"已经过了"就停在旧值上
        filled(&mut store, data);
        assert_eq!(store.contiguous_prefix(), 10);
        assert_eq!(store.bytes_have(), 10);
    }

    #[test]
    fn evict_is_limited_by_contiguous_prefix() {
        let data = b"abcdefghij";
        let (layout, hashes) = layout_and_hashes(data, 4);
        let mut store = PieceStore::new(layout, hashes);
        // 只有前两片（前缀 = 8），但播放头已经跳到 10
        store.write_piece(0, b"abcd").unwrap();
        store.write_piece(1, b"efgh").unwrap();
        assert_eq!(store.contiguous_prefix(), 8);
        assert_eq!(
            store.evict_before(10),
            2,
            "上限是 min(播放头, 连续前缀)：只回收前缀之内的片"
        );
        assert_eq!(store.contiguous_prefix(), 8, "前缀不倒退");
        assert_eq!(store.bytes_have(), 0);

        // 还没下到的片本来也没东西可回收
        let (layout, hashes) = layout_and_hashes(data, 4);
        let mut store = PieceStore::new(layout, hashes);
        assert_eq!(store.evict_before(10), 0);
    }

    #[test]
    fn mark_have_unchecked_keeps_bookkeeping_consistent() {
        let data = b"abcdefghij";
        let (layout, hashes) = layout_and_hashes(data, 4);
        let mut store = PieceStore::new(layout, hashes);
        store.mark_have_unchecked(1);
        assert!(store.have(1));
        assert_eq!(store.bytes_have(), 4);
        assert_eq!(store.contiguous_prefix(), 0, "第 0 片还没有，前缀不动");
        assert_eq!(store.ready_ranges(), vec![(4, 8)]);
        // 越界与重复都是无操作
        store.mark_have_unchecked(1);
        store.mark_have_unchecked(99);
        assert_eq!(store.bytes_have(), 4);
    }

    #[test]
    fn hash_count_mismatch_is_safe() {
        let data = b"abcdefghij";
        let (layout, _) = layout_and_hashes(data, 4);
        // 少给一个哈希：不 panic、不缺片数，缺的那片永远校验不过
        let mut store = PieceStore::new(layout, vec![[9u8; 20]; 2]);
        assert_eq!(store.layout().num_pieces, 3);
        assert!(store.write_piece(2, b"ij").is_err(), "零哈希必然校验失败");
        // 多给哈希也不炸
        let mut store = PieceStore::new(layout, vec![[1u8; 20]; 9]);
        assert!(store.write_piece(0, b"abcd").is_err());
    }

    #[test]
    fn piece_store_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PieceStore>();
        assert_send_sync::<Bitfield>();
        assert_send_sync::<PieceLayout>();
    }

    #[test]
    fn works_behind_a_lock_like_the_outer_layer_does() {
        // 外层会把它包成 Arc<RwLock<..>>，这里模拟一次"读锁读、写锁写"的用法，
        // 确认 `read_at(&self)` 在锁内可用（这正是选 `&mut self` 方案的验收点）。
        use std::sync::{Arc, RwLock};
        let data = b"abcdefghij";
        let (layout, hashes) = layout_and_hashes(data, 4);
        let store = Arc::new(RwLock::new(PieceStore::new(layout, hashes)));
        {
            let mut w = store.write().unwrap();
            w.write_piece(0, b"abcd").unwrap();
        }
        let handle = {
            let store = Arc::clone(&store);
            std::thread::spawn(move || {
                let r = store.read().unwrap();
                let mut buf = [0u8; 4];
                r.read_at(0, &mut buf).unwrap()
            })
        };
        assert_eq!(handle.join().unwrap(), 4);
    }
}
