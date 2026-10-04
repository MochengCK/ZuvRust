//! 片选择：**边下边播的心脏**。
//!
//! 普通 BT 客户端的默认策略是 rarest-first（最稀有的片先下）——它的目标是
//! "尽快让整个种子在 swarm 里存活"，完全不关心本地播放。边下边播要的是
//! **播放头附近立刻有连续数据**，否则播放器每读几十 KB 就短读一次，
//! 表现就是"刚打开就转圈、播两秒停一秒"。
//!
//! 所以这里的优先级**完全由播放头的位置推导**，而不是由全局稀有度推导：
//!
//! | 档位 | 区间（相对播放头 `P`） | 排序 | 为什么 |
//! |---|---|---|---|
//! | [`Prio::Stream`] | `[P, P+stream_window)` | **按片号升序** | 这一档必须形成连续前缀，播放器才能一路读下去 |
//! | [`Prio::Next`] | `[P+stream_window, P+next_window)` | 按片号升序 | 提前把"再往前一段"铺好，抗抖动 |
//! | [`Prio::Soon`] | `[P-back_window, P)` | 按片号升序 | 给 seek 回退用；回退是最常见的用户操作 |
//! | [`Prio::Backfill`] | 其余 | **rarest-first** | 这里才轮到"让种子健康"的目标，且不影响播放 |
//!
//! 两个刻意的决定：
//!
//! 1. **Stream 档内不做 rarest-first。** 直觉上"缺得最厉害的片先下"更好，
//!    但在播放头窗口里它是错的：稀有度排序会把窗口里的片按"谁被请求得少"
//!    打散，于是下载顺序在窗口内来回跳，连续前缀迟迟不成立 ——
//!    而"连续前缀"正是播放器能不能解出下一个容器头的前提。
//! 2. **优先级是 `pick` 时现算的，不维护"排好序的待下队列"。**
//!    播放头会随 seek 剧烈跳变，维护队列意味着每次 seek 都要重建整个队列，
//!    重建期间的数据结构还会处于中间态。现算的代价只是一次 O(片数) 扫描，
//!    换来的是"任何时刻查到的优先级都是当前播放头的真值"
//!    （[`PiecePicker::set_playhead`] 之后 [`PiecePicker::prio_of`] 立刻变档）。
//!
//! 本层**不碰网络**：`pick` 只回答"该下哪几片"，谁去下、下多久是上层的事。

use std::collections::HashMap;

use crate::storage::{Bitfield, PieceLayout};

/// 片优先级档位。`Stream` 是唯一"必须马上有"的一档。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prio {
    /// 播放头往前 `stream_window_bytes`：缺了它播放就会卡住。
    Stream,
    /// 再往前一档：提前铺路，避免播放头推进时来不及。
    Next,
    /// 播放头往回 `back_window_bytes`：给 seek 回退留的缓冲。
    Soon,
    /// 其余：这里才用 rarest-first。
    Backfill,
}

/// 调度参数。默认值是按"1080p/4K 视频 + 家用带宽"取的。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PickerConfig {
    /// 播放头往前的"必须马上有"区间。
    ///
    /// 32 MiB 的依据：容器解析器（MP4 的 moov/碎片、TS 的 PMT）在 seek 之后
    /// 需要重新读一段索引，加上播放器自身的解码缓冲，量级在几十 MB；
    /// 再大就变成"为了保险而浪费带宽"（这一段是要抢在最前面的）。
    pub stream_window_bytes: u64,
    /// 再往前一档。
    ///
    /// 128 MiB ≈ 十几秒到几十秒的码流，够扛住一次短暂的带宽抖动。
    pub next_window_bytes: u64,
    /// 播放头往回的一档，供 seek 回退。
    ///
    /// 比往前的一档小得多是**故意**的：回退是"用户主动跳回去"，
    /// 这时他会看到一次可接受的缓冲，而把回退缓冲做大等于常年浪费带宽。
    pub back_window_bytes: u64,
    /// 同一片最多同时/累计发出多少个请求。
    ///
    /// 3 是"够用来重试、又不至于把带宽浪费在重复数据上"的经验值：
    /// 一个片请求出去卡住了（peer 不回、连接断了），还可以换两个 peer 再试。
    pub max_requests_per_piece: u8,
    /// 请求超时（秒），同时充当失败退避的时长。
    ///
    /// 超过它就由 [`PiecePicker::expire_requests`] 回收；一片失败次数到达
    /// `max_requests_per_piece` 后也只在这么长时间内被冷落（见 [`PiecePicker::mark_failed`]）。
    pub deadline_secs: u64,
    /// peer 被冷处理的时长（秒）。
    pub snub_secs: u64,
}

impl Default for PickerConfig {
    fn default() -> Self {
        Self {
            stream_window_bytes: 32 * 1024 * 1024,
            next_window_bytes: 128 * 1024 * 1024,
            back_window_bytes: 8 * 1024 * 1024,
            max_requests_per_piece: 3,
            deadline_secs: 30,
            snub_secs: 60,
        }
    }
}

/// 调度统计。给上层做 UI / 日志用（"还有多少卡在播放头附近"）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PickerStats {
    /// 还缺、且属于 [`Prio::Stream`] 的片数 —— **这是"卡不卡"的唯一指标**。
    pub stream_pending: u32,
    /// 还缺、且属于 [`Prio::Backfill`] 的片数。
    pub backfill_pending: u32,
    /// 当前在飞（已请求、未回）的请求数。
    pub in_flight: u32,
    /// 已拥有的片数。
    pub completed: u32,
    /// 播放头所在的片号。
    pub playhead_piece: u32,
}

/// 每片的状态。刻意不放进 `HashMap<u32, _>`：片号是稠密的 0..N，
/// 用 `Vec` 下标访问比哈希快一个量级，而 `pick` 每次都要扫全表。
#[derive(Debug, Clone, Copy, Default)]
struct PieceState {
    have: bool,
    /// 在飞请求数。**同时被当作 Backfill 的 rarity 代理** ——
    /// 在飞越多说明越多人正在下它，也就越"不稀有"，见 [`PiecePicker::pick`]。
    in_flight: u32,
    /// 累计失败次数（超时/断连）。
    failed: u32,
    /// 最近一次失败的时间（秒），用于退避判定。
    failed_at: u64,
}

/// 一条在飞请求。保留 `peer` 是为了能查出"是哪个 peer 卡住了"，
/// 而 `at` 是超时回收的唯一依据。
///
/// `needed` 记的是"发出这条请求时，这一片是否落在某个窗口里"（即当时不是 Backfill）。
/// 它是 [`PiecePicker::cancelled_in_flight`] 的全部依据，见该方法的说明。
#[derive(Debug, Clone, Copy)]
struct InFlight {
    index: u32,
    peer: u64,
    at: u64,
    needed: bool,
}

/// 片选择器。
#[derive(Debug)]
pub struct PiecePicker {
    layout: PieceLayout,
    cfg: PickerConfig,
    playhead: u64,
    pieces: Vec<PieceState>,
    in_flight: Vec<InFlight>,
    /// peer → 冷处理截止时刻。用 `HashMap` 而不是 `BTreeMap`：
    /// 这里只做点查，不迭代，哈希更合适。
    snubbed: HashMap<u64, u64>,
}

impl PiecePicker {
    pub fn new(layout: PieceLayout, cfg: PickerConfig) -> Self {
        let mut cfg = cfg;
        if cfg.next_window_bytes < cfg.stream_window_bytes {
            // 这不是崩溃（Next 档会退化成空集，那一段直接掉进 Backfill），
            // 但几乎肯定不是调用方想要的，所以钳一下并明确告诉它。
            tracing::warn!(
                stream_window = cfg.stream_window_bytes,
                next_window = cfg.next_window_bytes,
                "next_window_bytes 小于 stream_window_bytes，已钳到与 stream 窗口相等"
            );
            cfg.next_window_bytes = cfg.stream_window_bytes;
        }
        if cfg.max_requests_per_piece == 0 {
            // 0 会让每片永远不可选（`in_flight >= 0` 恒成立），调度器直接死掉。
            // 这种"配置写错就静默不干活"的坑比报错难查得多，所以在这里兜住。
            tracing::warn!("max_requests_per_piece 为 0，已钳到 1");
            cfg.max_requests_per_piece = 1;
        }
        Self {
            pieces: vec![PieceState::default(); layout.num_pieces as usize],
            layout,
            cfg,
            playhead: 0,
            in_flight: Vec::new(),
            snubbed: HashMap::new(),
        }
    }

    pub fn layout(&self) -> &PieceLayout {
        &self.layout
    }

    pub fn config(&self) -> &PickerConfig {
        &self.cfg
    }

    /// **播放头动了**（正常播放推进 / seek）：重排优先级。
    ///
    /// 越界（`offset > total`）钳到末尾而不是报错：调用方是拿"播放器报的进度"
    /// 直接喂进来的，片尾多出几个字节很正常。
    pub fn set_playhead(&mut self, offset: u64) {
        self.playhead = offset.min(self.layout.total);
        // 这里**不**动任何状态：见模块开头的第 2 条 —— 优先级现算，
        // 所以这一行之后 `prio_of` 立刻返回新档位，没有任何"过渡态"。
    }

    pub fn playhead(&self) -> u64 {
        self.playhead
    }

    /// 从 [`crate::storage::PieceStore`] 同步"已经有了"。
    pub fn mark_have(&mut self, index: u32) {
        let Some(p) = self.pieces.get_mut(index as usize) else {
            return;
        };
        if p.have {
            return;
        }
        p.have = true;
        // 清掉这一片的在飞请求：不清的话 `in_flight` 会永远不为 0，
        // 让 Backfill 的 rarity 估计失真（一片早下完了却还被算成"很多人抢"），
        // 而且 `expire_requests` 会为一片已完成的数据反复记账。
        p.in_flight = 0;
        self.in_flight.retain(|r| r.index != index);
    }

    /// 挑下一批要下的片（最多 `want` 个，**同一批里不重复**）。
    ///
    /// 只做筛选与排序，不改任何状态 —— "请求已经发出去了"由调用方在真正发出后
    /// 用 [`PiecePicker::mark_requested`] 告知，因为 `pick` 并不知道这一批请求
    /// 最后有没有发出去（可能连接刚好断了）。
    pub fn pick(&mut self, peer_has: &Bitfield, want: usize, now_secs: u64) -> Vec<u32> {
        if want == 0 {
            return Vec::new();
        }
        let mut stream = Vec::new();
        let mut next = Vec::new();
        let mut soon = Vec::new();
        // Backfill 需要 rarity 排序，所以先攒成 (在飞数, 片号) 再排。
        let mut backfill: Vec<(u32, u32)> = Vec::new();
        for i in 0..self.layout.num_pieces {
            if !self.selectable(i, peer_has, now_secs) {
                continue;
            }
            match self.prio_of(i) {
                // 片号是递增遍历的，所以这三个 Vec 天然升序，不需要再排。
                Prio::Stream => stream.push(i),
                Prio::Next => next.push(i),
                Prio::Soon => soon.push(i),
                Prio::Backfill => backfill.push((self.pieces[i as usize].in_flight, i)),
            }
        }
        // (在飞数, 片号) 元组排序 = 在飞少的优先，同档按片号升序 ——
        // 正好就是"rarest-first + 稳定兜底"。在飞数近似 rarity 的好处是
        // **不需要全局 peer 统计**：一个片越多人正在下，说明它越不稀有。
        backfill.sort_unstable();

        let mut out: Vec<u32> = Vec::with_capacity(want.min(self.layout.num_pieces as usize));
        for group in [stream, next, soon] {
            for i in group {
                if out.len() == want {
                    return out;
                }
                out.push(i);
            }
        }
        for (_, i) in backfill {
            if out.len() == want {
                break;
            }
            out.push(i);
        }
        out
    }

    /// 记录"已经向某个 peer 请求过这片"。
    pub fn mark_requested(&mut self, index: u32, peer: u64, now_secs: u64) {
        let Some(p) = self.pieces.get_mut(index as usize) else {
            return;
        };
        if p.have {
            return;
        }
        p.in_flight += 1;
        self.in_flight.push(InFlight {
            index,
            peer,
            at: now_secs,
            // 记下"发这条请求时它是不是当务之急"：只有当时需要的片，
            // 在 playhead 搬走之后才谈得上"甩到窗口外"（见 cancelled_in_flight）。
            needed: self.prio_of(index) != Prio::Backfill,
        });
    }

    /// 因为 seek 而**已经不再需要**的在飞请求，返回应当向对应 peer 发 `Cancel` 的片号
    /// （升序、去重）。
    ///
    /// ## 语义（为什么是"当时需要、现在不需要"）
    ///
    /// 判据是：**发这条请求时**这一片落在某个窗口里（`needed == true`），
    /// 而**现在**它已经掉出所有窗口、回到 [`Prio::Backfill`]。
    ///
    /// - 只有 seek 才会造成这种"当时急需、如今无关"的落差。播放头往前走时，
    ///   老的 Stream 片会依次降为 Soon/Backfill，那些**已经不需要马上要**的数据
    ///   继续占着 peer 的发送窗口，就是纯粹的带宽浪费 —— 而我们正急着要新位置的片。
    ///   发一条 Cancel 让 peer 立刻腾出带宽，这是"seek 之后要快"的关键一步。
    /// - 反过来，**发出时就是 Backfill 的请求不在此列**：它本来就是后台补片，
    ///   没有"被 seek 甩掉"这回事，贸然取消只会让种子永远补不齐。
    ///
    /// ## 调用方要做什么
    ///
    /// 这个方法只**回答**该取消哪些片（`&self`，不改状态）。调用方对每个片号向
    /// 对应 peer 发 `Cancel` 后，那条在飞请求会走两条路之一被清掉：
    /// peer 遵守 Cancel 就不再回数据，由 [`PiecePicker::expire_requests`] 超时回收；
    /// peer 不理会照发数据，则 `mark_have` 时一并清掉。两者都收敛，
    /// 所以这里不需要（也不应该）自己改状态。
    pub fn cancelled_in_flight(&self) -> Vec<u32> {
        let mut out: Vec<u32> = self
            .in_flight
            .iter()
            .filter(|r| r.needed && self.prio_of(r.index) == Prio::Backfill)
            .map(|r| r.index)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// 超时/断连：计数加一；超过 `max_requests_per_piece` 就进退避。
    pub fn mark_failed(&mut self, index: u32, now_secs: u64) {
        let Some(p) = self.pieces.get_mut(index as usize) else {
            return;
        };
        p.failed = p.failed.saturating_add(1);
        p.failed_at = now_secs;
        // 顺手回收一条在飞记录：失败意味着"这条请求不会有应答了"。
        if let Some(pos) = self.in_flight.iter().position(|r| r.index == index) {
            self.in_flight.remove(pos);
            p.in_flight = p.in_flight.saturating_sub(1);
        }
    }

    /// 某个 peer 连续给不出数据 → 冷处理。
    pub fn mark_snubbed(&mut self, peer: u64, now_secs: u64) {
        self.snubbed
            .insert(peer, now_secs.saturating_add(self.cfg.snub_secs));
    }

    pub fn is_snubbed(&self, peer: u64, now_secs: u64) -> bool {
        self.snubbed
            .get(&peer)
            .is_some_and(|&until| now_secs < until)
    }

    /// 回收超时的在飞请求（把片放回可挑状态），返回回收的请求数。
    ///
    /// **刻意不计入 `failed`**：超时回收的语义是"再给一次机会"，
    /// 而退避是 [`PiecePicker::mark_failed`] 的显式语义（上层知道那是一次真失败）。
    /// 两者混在一起的话，一个只是慢的 peer 会把片推入退避，反而更难下到。
    pub fn expire_requests(&mut self, now_secs: u64) -> usize {
        let timeout = self.cfg.deadline_secs;
        let mut expired: Vec<InFlight> = Vec::new();
        self.in_flight.retain(|r| {
            if now_secs.saturating_sub(r.at) >= timeout {
                expired.push(*r);
                false
            } else {
                true
            }
        });
        for r in &expired {
            tracing::debug!(
                piece = r.index,
                peer = r.peer,
                "在飞请求超时，片已放回可挑状态"
            );
            if let Some(p) = self.pieces.get_mut(r.index as usize) {
                p.in_flight = p.in_flight.saturating_sub(1);
            }
        }
        // 顺手清掉过期的冷处理记录：否则长时间运行时 peer 不断进出，
        // 这张表会只增不减（每个条目才 16 字节，但没理由让它无限长）。
        self.snubbed.retain(|_, until| now_secs < *until);
        expired.len()
    }

    /// 某个片当前属于哪一档。**纯位置函数**：不看 `have`、不看在飞数。
    pub fn priority_of(&self, index: u32) -> Prio {
        if index >= self.layout.num_pieces {
            return Prio::Backfill;
        }
        let start = self.layout.piece_offset(index);
        let end = start + self.layout.piece_size(index) as u64;
        let ph = self.playhead;

        let stream_start = ph;
        let stream_end = ph.saturating_add(self.cfg.stream_window_bytes);
        if overlaps(start, end, stream_start, stream_end) {
            return Prio::Stream;
        }
        // 顺序很重要：Stream → Next → Soon。一片同时落在两个窗口里时
        // 取更"紧急"的那档（比如 seek 回退后，回退缓冲与前进窗口可能重叠）。
        let next_end = ph.saturating_add(self.cfg.next_window_bytes);
        if overlaps(start, end, stream_end, next_end) {
            return Prio::Next;
        }
        let soon_start = ph.saturating_sub(self.cfg.back_window_bytes);
        if overlaps(start, end, soon_start, ph) {
            return Prio::Soon;
        }
        Prio::Backfill
    }

    /// [`PiecePicker::priority_of`] 的短名。内部调用与测试沿用旧名，
    /// 对外契约以 `priority_of` 为准（两者是同一个函数）。
    pub fn prio_of(&self, index: u32) -> Prio {
        self.priority_of(index)
    }

    pub fn stats(&self) -> PickerStats {
        let mut stream_pending = 0u32;
        let mut backfill_pending = 0u32;
        let mut completed = 0u32;
        // 这里现数一遍而不是增量维护 `completed`：`stats` 是给 UI/日志看的，
        // 调用频率极低，而增量维护意味着"每个改动 have 的地方都要记得 +1"——
        // 少一处就是永久的统计偏差，不划算。
        for i in 0..self.layout.num_pieces {
            let p = &self.pieces[i as usize];
            if p.have {
                completed += 1;
                continue;
            }
            match self.prio_of(i) {
                Prio::Stream => stream_pending += 1,
                Prio::Backfill => backfill_pending += 1,
                Prio::Next | Prio::Soon => {}
            }
        }
        PickerStats {
            stream_pending,
            backfill_pending,
            in_flight: self.in_flight.len() as u32,
            completed,
            playhead_piece: self.layout.piece_index_of(self.playhead),
        }
    }

    /// 这个片现在能不能选给某个 peer。
    ///
    /// **注意**：冷处理（[`PiecePicker::is_snubbed`]）不在这里判断 ——
    /// `pick` 没有 peer 参数，冷处理是调用方在"把 `pick` 的结果交给哪个 peer"
    /// 这一步自己过滤的（见方法文档：本层只回答"该下哪几片"）。
    fn selectable(&self, index: u32, peer_has: &Bitfield, now_secs: u64) -> bool {
        let p = &self.pieces[index as usize];
        !p.have
            // peer 没有的片要了也白要（Bitfield 越界读恒为 false，所以
            // 对面片数与本地不一致时也不会误选）。
            && peer_has.get(index)
            && p.in_flight < self.cfg.max_requests_per_piece as u32
            && !self.in_backoff(index, now_secs)
    }

    /// 是不是处在"失败太多次"的退避期里。
    ///
    /// 退避是**有期限**的：`deadline_secs` 之后自动恢复。
    /// 永久退避会让一片坏数据把整个种子卡死（永远下不到它），
    /// 而 BT 里"这个 peer 给不出"和"这片谁都给不出"是完全不同的两件事。
    fn in_backoff(&self, index: u32, now_secs: u64) -> bool {
        let p = &self.pieces[index as usize];
        p.failed >= self.cfg.max_requests_per_piece as u32
            && now_secs < p.failed_at.saturating_add(self.cfg.deadline_secs)
    }
}

/// 半开区间 `[a_start, a_end)` 与 `[b_start, b_end)` 是否相交。
fn overlaps(a_start: u64, a_end: u64, b_start: u64, b_end: u64) -> bool {
    a_start < b_end && b_start < a_end
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试用布局：片长 4 字节、共 10 片（40 字节）。
    /// 窗口参数也用字节数，于是"第几片落在哪一档"可以直接心算。
    fn layout() -> PieceLayout {
        PieceLayout::new(4, 40)
    }

    fn cfg() -> PickerConfig {
        PickerConfig {
            stream_window_bytes: 8,   // 2 片
            next_window_bytes: 16,    // 再 2 片
            back_window_bytes: 4,     // 往回 1 片
            max_requests_per_piece: 3,
            deadline_secs: 30,
            snub_secs: 60,
        }
    }

    fn picker() -> PiecePicker {
        PiecePicker::new(layout(), cfg())
    }

    /// 全部 10 片都有的 peer 位图。
    fn all() -> Bitfield {
        let mut b = Bitfield::new(10);
        for i in 0..10 {
            b.set(i, true);
        }
        b
    }

    /// 只有指定片号的位图。
    fn only(idx: &[u32]) -> Bitfield {
        let mut b = Bitfield::new(10);
        for &i in idx {
            b.set(i, true);
        }
        b
    }

    #[test]
    fn default_config_is_sane() {
        let c = PickerConfig::default();
        assert_eq!(c.stream_window_bytes, 32 * 1024 * 1024);
        assert_eq!(c.next_window_bytes, 128 * 1024 * 1024);
        assert_eq!(c.back_window_bytes, 8 * 1024 * 1024);
        assert_eq!(c.max_requests_per_piece, 3);
        assert_eq!(c.deadline_secs, 30);
        assert_eq!(c.snub_secs, 60);
        // 窗口必须一层套一层，否则档位之间会出现空洞
        assert!(c.stream_window_bytes < c.next_window_bytes);
    }

    #[test]
    fn bad_config_is_clamped_not_silently_dead() {
        let mut c = cfg();
        c.next_window_bytes = 2; // 比 stream 窗口还小
        c.max_requests_per_piece = 0; // 会让调度器一片都挑不出来
        let mut p = PiecePicker::new(layout(), c);
        assert_eq!(p.config().next_window_bytes, p.config().stream_window_bytes);
        assert_eq!(p.config().max_requests_per_piece, 1);
        // 钳过之后必须还能干活（max_requests_per_piece=0 会让 selectable 恒假）
        assert_eq!(p.pick(&all(), 2, 0), vec![0, 1]);
    }

    #[test]
    fn stream_priority_is_index_ascending_not_rarest_first() {
        let mut p = picker();
        p.set_playhead(0);
        // 播放头 0：Stream = 片 0,1（[0,8)）；Next = 片 2,3（[8,16)）
        assert_eq!(p.prio_of(0), Prio::Stream);
        assert_eq!(p.prio_of(1), Prio::Stream);
        assert_eq!(p.prio_of(2), Prio::Next);
        assert_eq!(p.prio_of(3), Prio::Next);
        assert_eq!(p.prio_of(4), Prio::Backfill);

        let got = p.pick(&all(), 10, 0);
        assert_eq!(
            got,
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "窗口内按片号升序（连续前缀），窗口外才轮到 rarest-first"
        );
    }

    #[test]
    fn stream_window_makes_a_contiguous_prefix() {
        // 这一条是"为什么 Stream 档不能 rarest-first"的行为断言：
        // 播放头在片 0，从 0 开始一片一片拿，拿到的顺序必须是 0,1,2,...
        let mut p = picker();
        p.set_playhead(0);
        let mut order = Vec::new();
        for _ in 0..10 {
            let got = p.pick(&all(), 1, 0);
            assert_eq!(got.len(), 1);
            order.push(got[0]);
            p.mark_have(got[0]);
        }
        assert_eq!(order, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn backfill_is_rarest_first() {
        // 把三个窗口都关掉，只剩 Backfill 档，单独验它的排序规则
        let mut c = cfg();
        c.stream_window_bytes = 0;
        c.next_window_bytes = 0;
        c.back_window_bytes = 0;
        let mut p = PiecePicker::new(layout(), c);
        p.set_playhead(0);
        for i in 0..10 {
            assert_eq!(p.prio_of(i), Prio::Backfill);
        }

        // 片 5 被两个人抢、片 3 被一个人抢 → 它们"更不稀有"，应当排到最后
        p.mark_requested(5, 100, 0);
        p.mark_requested(5, 101, 0);
        p.mark_requested(3, 102, 0);
        let got = p.pick(&all(), 10, 0);
        assert_eq!(
            got,
            vec![0, 1, 2, 4, 6, 7, 8, 9, 3, 5],
            "在飞少的先下（rarest-first 的近似），同在飞数按片号升序"
        );
    }

    #[test]
    fn seek_promotes_backfill_to_stream_immediately() {
        let mut p = picker();
        p.set_playhead(0);
        // 片 5 覆盖 [20,24)：播放头在 0 时它离得太远
        assert_eq!(p.prio_of(5), Prio::Backfill);

        // 用户 seek 到 20 字节（第 5 片的开头）
        p.set_playhead(20);
        assert_eq!(
            p.prio_of(5),
            Prio::Stream,
            "set_playhead 之后原来 Backfill 的片必须立刻升档，不能等下个 tick"
        );
        assert_eq!(p.prio_of(6), Prio::Stream);
        assert_eq!(p.prio_of(7), Prio::Next);
        assert_eq!(p.prio_of(8), Prio::Next);
        assert_eq!(p.prio_of(4), Prio::Soon, "往回 4 字节的那片是回退缓冲");
        assert_eq!(p.prio_of(3), Prio::Backfill, "再往回就不管了");
        assert_eq!(p.playhead(), 20);
    }

    #[test]
    fn pick_after_seek_starts_at_the_new_playhead() {
        // **关键回归**：seek 之后 pick 的第一个片必须落在新窗口里，
        // 而不是继续补"老位置"（这正是边下边播最容易做错的地方：
        // 把优先级算对了却把待下队列留在原地）
        let mut p = picker();
        p.set_playhead(0);
        // 先把播放头附近下掉几片，制造"老位置还有没下完的片"的现场
        p.mark_have(0);
        p.mark_have(1);
        // 把 Next 窗口里的片也占满（每片达到上限），否则 pick 会先挑它们
        for _ in 0..3 {
            p.mark_requested(2, 1, 0);
            p.mark_requested(3, 1, 0);
        }
        // 老窗口里的片都还在飞/已有，此时 pick 会往 Backfill 走
        assert_eq!(p.pick(&all(), 1, 0), vec![4]);

        // 现在 seek 到 24 字节（第 6 片）
        p.set_playhead(24);
        let got = p.pick(&all(), 3, 0);
        assert_eq!(got, vec![6, 7, 8], "第一个片必须是新播放头所在的片");
        assert_eq!(
            p.prio_of(got[0]),
            Prio::Stream,
            "pick 出来的第一个片必须属于 Stream 档"
        );
        // 而且它确实落在 [playhead, playhead+stream_window) 里
        assert!(p.layout().piece_offset(got[0]) >= 24);
        assert!(p.layout().piece_offset(got[0]) < 24 + p.config().stream_window_bytes);
    }

    #[test]
    fn pick_skips_have_and_pieces_the_peer_lacks() {
        let mut p = picker();
        p.set_playhead(0);
        p.mark_have(0);
        // peer 没有片 1、也没有片 5
        let got = p.pick(&only(&[0, 2, 3, 4, 6, 7, 8, 9]), 10, 0);
        assert!(!got.contains(&0), "已拥有的片不能再下");
        assert!(!got.contains(&1), "peer 没有的片不能选");
        assert!(!got.contains(&5));
        assert_eq!(got, vec![2, 3, 4, 6, 7, 8, 9]);

        // peer 位图完全空（对面片数为 0 也不该 panic）
        assert!(p.pick(&Bitfield::new(0), 10, 0).is_empty());
    }

    #[test]
    fn one_batch_has_no_duplicates_and_respects_want() {
        let mut p = picker();
        p.set_playhead(0);
        let got = p.pick(&all(), 3, 0);
        assert_eq!(got, vec![0, 1, 2]);
        let got = p.pick(&all(), 100, 0);
        assert_eq!(got.len(), 10, "want 超过可用片数时只给可用的");
        let mut sorted = got.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), got.len(), "同一批里不能有重复片");

        // 8 片已有 → 只剩 2 片可挑
        for i in 0..8 {
            p.mark_have(i);
        }
        assert_eq!(p.pick(&all(), 100, 0), vec![8, 9]);
        assert!(p.pick(&all(), 0, 0).is_empty(), "want=0 直接返回空");
    }

    #[test]
    fn requested_pieces_are_retried_up_to_the_limit_then_skipped() {
        let mut p = picker();
        p.set_playhead(0);
        assert_eq!(p.pick(&all(), 1, 0), vec![0]);

        // 同一片向两个不同 peer 各请求一次：还在上限内，可以再向第三个 peer 要
        p.mark_requested(0, 1, 0);
        p.mark_requested(0, 2, 0);
        assert_eq!(p.stats().in_flight, 2);
        assert_eq!(p.pick(&all(), 1, 0), vec![0], "没到上限，允许换 peer 重试");

        // 第三次之后就到 max_requests_per_piece 了
        p.mark_requested(0, 3, 0);
        assert_eq!(p.pick(&all(), 1, 0), vec![1], "到上限就不该再选它");
    }

    #[test]
    fn expire_requests_puts_pieces_back_into_the_pool() {
        let mut p = picker();
        p.set_playhead(0);
        p.mark_requested(0, 1, 100);
        p.mark_requested(0, 2, 100);
        p.mark_requested(0, 3, 100);
        assert_eq!(p.stats().in_flight, 3);
        // 到上限了，挑不到它
        assert_eq!(p.pick(&all(), 1, 100), vec![1]);

        // 29 秒后还没超时
        assert_eq!(p.expire_requests(129), 0);
        assert_eq!(p.stats().in_flight, 3);
        // 满 30 秒：三条都回收
        assert_eq!(p.expire_requests(130), 3);
        assert_eq!(p.stats().in_flight, 0);
        assert_eq!(p.pick(&all(), 1, 130), vec![0], "回收后这片重新可挑");
        // 幂等
        assert_eq!(p.expire_requests(200), 0);
    }

    #[test]
    fn failures_lead_to_a_time_bounded_backoff() {
        let mut p = picker();
        p.set_playhead(0);
        // 片 0 失败 3 次（= max_requests_per_piece）→ 进退避
        for _ in 0..3 {
            p.mark_failed(0, 0);
        }
        assert_eq!(p.pick(&all(), 1, 5), vec![1], "退避期内不能再选它");
        // 退避是有期限的：deadline_secs 之后自动恢复
        assert_eq!(p.pick(&all(), 1, 30), vec![0], "退避到期后重新可挑");

        // 只失败 2 次不算超限
        let mut p = picker();
        p.set_playhead(0);
        p.mark_failed(0, 0);
        p.mark_failed(0, 0);
        assert_eq!(p.pick(&all(), 1, 5), vec![0]);
    }

    #[test]
    fn mark_failed_recycles_one_in_flight_request() {
        let mut p = picker();
        p.set_playhead(0);
        p.mark_requested(0, 1, 0);
        assert_eq!(p.stats().in_flight, 1);
        p.mark_failed(0, 1);
        assert_eq!(p.stats().in_flight, 0, "失败意味着那条在飞请求不会有应答了");
        // 没有在飞记录时再失败也不该下溢
        p.mark_failed(0, 2);
        assert_eq!(p.stats().in_flight, 0);
        assert_eq!(p.stats().completed, 0);
    }

    #[test]
    fn mark_have_is_idempotent_and_clears_in_flight() {
        let mut p = picker();
        p.set_playhead(0);
        p.mark_requested(0, 1, 0);
        p.mark_have(0);
        assert_eq!(p.stats().completed, 1);
        assert_eq!(p.stats().in_flight, 0);
        assert_eq!(p.stats().stream_pending, 1);
        // 重复 mark_have 不会把 completed 加两次
        p.mark_have(0);
        assert_eq!(p.stats().completed, 1);
        // 越界片号静默忽略
        p.mark_have(999);
        assert_eq!(p.stats().completed, 1);
        // 已拥有的片也不会再被 pick 出来
        assert!(!p.pick(&all(), 10, 0).contains(&0));
    }

    #[test]
    fn snub_is_time_bounded_and_per_peer() {
        let mut p = picker();
        assert!(!p.is_snubbed(7, 0), "没被冷处理过的 peer 不是 snubbed");
        p.mark_snubbed(7, 100);
        assert!(p.is_snubbed(7, 100));
        assert!(p.is_snubbed(7, 159), "60 秒之内都算冷处理");
        assert!(!p.is_snubbed(7, 160), "到点自动解冻");
        assert!(!p.is_snubbed(8, 100), "冷处理是按 peer 隔离的");
        // 冷处理不影响别的 peer
        p.mark_snubbed(7, 200);
        assert!(p.is_snubbed(7, 250));
    }

    #[test]
    fn stats_report_the_stream_pressure() {
        let mut p = picker();
        p.set_playhead(0);
        let s = p.stats();
        assert_eq!(s.stream_pending, 2, "片 0,1 在 Stream 窗口里");
        assert_eq!(s.backfill_pending, 6, "片 4..9 是 Backfill（片 2,3 是 Next）");
        assert_eq!(s.in_flight, 0);
        assert_eq!(s.completed, 0);
        assert_eq!(s.playhead_piece, 0);

        p.mark_have(0);
        p.mark_requested(1, 42, 0);
        let s = p.stats();
        assert_eq!(s.completed, 1);
        assert_eq!(s.stream_pending, 1);
        assert_eq!(s.in_flight, 1);

        // seek 到 20 字节（第 5 片）：窗口整体搬家
        p.set_playhead(20);
        let s = p.stats();
        assert_eq!(s.playhead_piece, 5);
        assert_eq!(s.stream_pending, 2, "片 5,6");
        assert_eq!(s.backfill_pending, 4, "片 1,2,3,9");
        assert_eq!(s.completed, 1, "have 的片不随 seek 变化");
        assert_eq!(s.in_flight, 1);
    }

    #[test]
    fn playhead_is_clamped_to_the_end() {
        let mut p = picker();
        p.set_playhead(u64::MAX);
        assert_eq!(p.playhead(), 40, "越界钳到总长，不 panic");
        assert_eq!(p.stats().playhead_piece, 9, "落在最后一片上");
        // 末尾时"往回"的那一档还能给出一片回退缓冲
        assert_eq!(p.prio_of(9), Prio::Soon);
    }

    #[test]
    fn degenerate_layouts_do_not_panic() {
        // 空种子
        let mut p = PiecePicker::new(PieceLayout::new(4, 0), cfg());
        assert_eq!(p.stats().completed, 0);
        assert!(p.pick(&all(), 10, 0).is_empty());
        assert_eq!(p.prio_of(0), Prio::Backfill);
        // 片长为 0 会被 PieceLayout 钳成 1（这里只要求不炸）
        let p2 = PiecePicker::new(PieceLayout::new(0, 5), cfg());
        assert_eq!(p2.layout().num_pieces, 5);
    }

    #[test]
    fn last_short_piece_is_covered_by_the_windows() {
        // 片长 4、总长 10 → 3 片（4/4/2）。播放头落在最后那片中间时，
        // 最后一片必须仍然算 Stream（它的长度短于 piece_length，边界最容易算错）
        let mut p = PiecePicker::new(PieceLayout::new(4, 10), cfg());
        p.set_playhead(8);
        assert_eq!(p.prio_of(2), Prio::Stream);
        assert_eq!(p.prio_of(1), Prio::Soon, "[4,8) 是回退缓冲");
        assert_eq!(p.prio_of(0), Prio::Backfill);
        assert_eq!(p.pick(&all(), 1, 0), vec![2]);
    }

    #[test]
    fn seek_cancels_in_flight_requests_that_left_the_windows() {
        let mut p = picker();
        p.set_playhead(0);
        // 播放头 0 时：片 0 是 Stream、片 2 是 Next（都算"当时需要"），片 6 是 Backfill
        p.mark_requested(0, 1, 0);
        p.mark_requested(2, 1, 0);
        p.mark_requested(6, 1, 0);
        assert!(p.cancelled_in_flight().is_empty(), "还没 seek，没有该取消的");

        // seek 到 24（第 6 片）：片 0 与片 2 都掉出所有窗口
        p.set_playhead(24);
        assert_eq!(p.prio_of(0), Prio::Backfill);
        assert_eq!(p.prio_of(2), Prio::Backfill);
        assert_eq!(p.prio_of(6), Prio::Stream);
        assert_eq!(
            p.cancelled_in_flight(),
            vec![0, 2],
            "只有「当时急需、如今掉出窗口」的请求才该取消；片 6 当时就是后台片，不在此列"
        );
    }

    #[test]
    fn cancelled_in_flight_keeps_requests_still_inside_a_window() {
        let mut p = picker();
        p.set_playhead(0);
        p.mark_requested(0, 1, 0); // Stream
        // seek 到 4（第 1 片）：片 0 从 Stream 降为 Soon，仍在回退窗口里 → 不该取消
        p.set_playhead(4);
        assert_eq!(p.prio_of(0), Prio::Soon);
        assert!(p.cancelled_in_flight().is_empty());

        // 再跳到末尾：片 0 彻底掉出窗口，这时才取消
        p.set_playhead(40);
        assert_eq!(p.prio_of(0), Prio::Backfill);
        assert_eq!(p.cancelled_in_flight(), vec![0]);
    }

    #[test]
    fn cancelled_in_flight_ignores_background_requests() {
        let mut p = picker();
        p.set_playhead(0);
        // 片 7 在播放头 0 时就是 Backfill：后台补片没有"被 seek 甩掉"这回事
        p.mark_requested(7, 1, 0);
        p.set_playhead(20);
        assert!(
            p.cancelled_in_flight().is_empty(),
            "发出时就不紧急的请求不该被取消，否则种子永远补不齐"
        );
    }

    #[test]
    fn cancelled_in_flight_is_deduped_sorted_and_cleared_by_have() {
        let mut p = picker();
        p.set_playhead(0);
        // 同一片向两个 peer 发过请求：取消集合里只应出现一次
        p.mark_requested(1, 1, 0);
        p.mark_requested(1, 2, 0);
        p.mark_requested(0, 3, 0);
        p.set_playhead(40); // 两片都掉出窗口
        assert_eq!(p.cancelled_in_flight(), vec![0, 1]);
        // 片到手之后在飞记录被清掉，自然也不再出现在取消集合里
        p.mark_have(1);
        assert_eq!(p.cancelled_in_flight(), vec![0]);
    }
}
