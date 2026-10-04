//! 多轨交织：把若干条轨道的采样按时间**交错**成一条流。
//!
//! ## 为什么必须交织
//!
//! 直接"先写完视频轨再写音频轨"在 MP4/TS 里是**不可播**的：播放器读文件是
//! 顺序的，它得在读到音频数据之前先读到它要播的那一段 —— 把两条轨各写一半，
//! 播放器要么卡住要么只播一条。所以容器要求数据在文件里**按时间大致交错**。
//!
//! ## 为什么按 DTS 而不是 PTS 排
//!
//! 有 B 帧时，`pts` 在**解码顺序**里会往回跳（显示顺序 ≠ 解码顺序）。
//! 拿 `pts` 做跨轨比较会得出"这一帧比上一帧早"的荒唐结论，进而把同一条轨道的
//! 采样顺序打乱 —— 解码器拿到乱序的参考帧就是花屏。
//! `dts` 在解码顺序里单调递增，所以：
//!
//! - **轨间**比较用 `dts`（换算到微秒，因为各轨时基不同）；
//! - **轨内**严格保持输入顺序，一个采样都不重排。
//!
//! ## 排序必须确定
//!
//! `dts` 相同时（音频与视频的第一条采样都在 0 是常态）按轨道号定序。
//! 不定序的话输出顺序会随 `HashMap` 的迭代顺序变化 —— 那是最难复现的一类 bug。

use std::collections::{HashMap, VecDeque};

use me_core::{MediaError, Packet, Result, Track};

/// 一条轨道的时基（用来把刻度换算成微秒做跨轨比较）。
#[derive(Debug, Clone, Copy)]
struct TrackTimebase {
    id: u32,
    ticks_per_second: u32,
}

impl TrackTimebase {
    fn to_us(self, ticks: i64) -> i64 {
        if self.ticks_per_second == 0 {
            return 0;
        }
        ((ticks as i128) * 1_000_000 / self.ticks_per_second as i128) as i64
    }
}

/// 多轨交织器。
///
/// 用法：`push` 各轨的采样（每轨按自己的顺序），`pop` 取交织后的流；
/// 全部推完之后调 `flush`（或 `drain_all`）取剩下的。
pub struct Interleaver {
    tracks: Vec<TrackTimebase>,
    /// 每条轨道手上"还没轮到输出"的采样队列。
    ///
    /// 为什么是队列而不是"只留一条"：调用方通常是从解封装器**连续**推入的，
    /// 要求它"推一条就得取走"会把调用点写得很别扭；而且某条轨道暂时没有数据时，
    /// 另一条轨道的采样会把它卡住。队列版本把这件事交给这里：
    /// 能确定顺序就吐，确定不了就先攒着（攒的量由 `window_us` 兜底）。
    pending: HashMap<u32, VecDeque<Packet>>,
    /// 已经到头的轨道。
    finished: Vec<u32>,
    /// 交织好的输出队列。
    out: VecDeque<Packet>,
    /// 每条轨道已推入多少条（诊断用）。
    pushed: HashMap<u32, u64>,
    /// 窗口（微秒）：等待更早采样的最大跨度。
    ///
    /// 为什么需要窗口：只有当**每条未到头的轨道手上都有采样**时，才能确定
    /// "手上最小的 dts 就是全局最小"。某条轨道还没送来采样时，无从知道它
    /// 下一条会不会更早 —— 盲目输出就会把顺序搞反。但也不能无限等
    /// （那条轨道可能整段都没有数据），所以手上采样的跨度超过窗口就照现有信息输出。
    window_us: i64,
}

impl Interleaver {
    /// 默认窗口：2 秒。真实文件的轨道数据通常几十毫秒内就会交错出现，
    /// 2 秒足够宽（不会误判），也足够窄（不会为了等一条轨道把内存撑爆）。
    pub const DEFAULT_WINDOW_US: i64 = 2_000_000;

    pub fn new(tracks: &[Track]) -> Self {
        Self::with_window_us(tracks, Self::DEFAULT_WINDOW_US)
    }

    pub fn with_window_us(tracks: &[Track], window_us: i64) -> Self {
        Self {
            tracks: tracks
                .iter()
                .map(|t| TrackTimebase { id: t.id, ticks_per_second: t.timebase.ticks_per_second })
                .collect(),
            pending: HashMap::new(),
            finished: Vec::new(),
            out: VecDeque::new(),
            pushed: HashMap::new(),
            window_us: window_us.max(0),
        }
    }

    fn timebase_of(&self, id: u32) -> Option<TrackTimebase> {
        self.tracks.iter().find(|t| t.id == id).copied()
    }

    /// 推入一条采样（同一轨道必须按顺序推，本函数不做轨内重排）。
    pub fn push(&mut self, p: Packet) -> Result<()> {
        if self.timebase_of(p.track_id).is_none() {
            return Err(MediaError::Container(format!(
                "交织器不认识轨道 {}（它不在 set_tracks 给的列表里）",
                p.track_id
            )));
        }
        let id = p.track_id;
        self.pending.entry(id).or_default().push_back(p);
        *self.pushed.entry(id).or_insert(0) += 1;
        self.drain();
        Ok(())
    }

    /// 标记某条轨道到头了（之后不再等它）。
    pub fn finish_track(&mut self, track_id: u32) {
        if !self.finished.contains(&track_id) {
            self.finished.push(track_id);
        }
        self.drain();
    }

    /// 所有轨道都到头。
    pub fn finish_all(&mut self) {
        let ids: Vec<u32> = self.tracks.iter().map(|t| t.id).collect();
        for id in ids {
            self.finish_track(id);
        }
    }

    /// 取一条交织后的采样。
    pub fn pop(&mut self) -> Option<Packet> {
        self.drain();
        self.out.pop_front()
    }

    /// 把剩下的（含还在等窗口的）全部按 dts 顺序吐出来。
    ///
    /// 调用方在"输入已经推完"之后必须调它，否则最后每条轨道都会剩一些没输出。
    pub fn flush(&mut self) {
        self.finish_all();
        self.drain();
    }

    /// 一次取出全部结果。
    pub fn drain_all(&mut self) -> Vec<Packet> {
        self.flush();
        self.out.drain(..).collect()
    }

    /// 轨道数。
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// 还在缓冲里等着的采样总数（诊断用；合并时靠它盯内存）。
    pub fn buffered(&self) -> usize {
        self.pending.values().map(|q| q.len()).sum()
    }

    /// 内部：把可以确定顺序的采样搬进输出队列。
    fn drain(&mut self) {
        loop {
            if self.pending.values().all(|q| q.is_empty()) {
                return;
            }
            // 还没到头的轨道是否都已经有采样了？没有就得等（除非跨度超过窗口）
            let waiting = self
                .tracks
                .iter()
                .map(|t| t.id)
                .any(|id| !self.finished.contains(&id) && self.pending.get(&id).map(|q| q.is_empty()).unwrap_or(true));
            if waiting {
                let (min_us, max_us) = self.pending_span();
                if max_us - min_us <= self.window_us {
                    return;
                }
                // 跨度太大：再等下去只会让内存越堆越多，照现有信息输出
            }

            // 取 (dts_us, 轨道号) 最小的那条队首
            let mut best: Option<(i64, u32)> = None;
            for (id, q) in &self.pending {
                let Some(front) = q.front() else { continue };
                let Some(tb) = self.timebase_of(*id) else { continue };
                let key = (tb.to_us(front.dts), *id);
                if best.map(|b| key < b).unwrap_or(true) {
                    best = Some(key);
                }
            }
            let Some((_, best_id)) = best else { return };
            if let Some(q) = self.pending.get_mut(&best_id) {
                if let Some(p) = q.pop_front() {
                    self.out.push_back(p);
                }
            }
        }
    }

    fn pending_span(&self) -> (i64, i64) {
        let mut min = i64::MAX;
        let mut max = i64::MIN;
        for (id, q) in &self.pending {
            let Some(front) = q.front() else { continue };
            let Some(back) = q.back() else { continue };
            let Some(tb) = self.timebase_of(*id) else { continue };
            min = min.min(tb.to_us(front.dts));
            max = max.max(tb.to_us(back.dts));
        }
        if min == i64::MAX {
            (0, 0)
        } else {
            (min, max)
        }
    }
}

/// 便捷函数：把"每轨的采样序列"一次交织好（测试与 CLI 都用得上）。
pub fn interleave_all(tracks: &[Track], packets: &[Packet]) -> Result<Vec<Packet>> {
    let mut it = Interleaver::new(tracks);
    // 按轨道分组、保持各轨内部顺序
    let mut by_track: HashMap<u32, Vec<&Packet>> = HashMap::new();
    for p in packets {
        by_track.entry(p.track_id).or_default().push(p);
    }
    // 轮流给每条轨道各推一条，模拟"交织存放"的输入顺序
    let ids: Vec<u32> = tracks.iter().map(|t| t.id).collect();
    let mut cursor: HashMap<u32, usize> = ids.iter().map(|id| (*id, 0)).collect();
    loop {
        let mut any = false;
        for id in &ids {
            let idx = *cursor.get(id).unwrap_or(&0);
            if let Some(l) = by_track.get(id) {
                if idx < l.len() {
                    it.push(l[idx].clone())?;
                    cursor.insert(*id, idx + 1);
                    any = true;
                }
            }
        }
        if !any {
            break;
        }
    }
    Ok(it.drain_all())
}

#[cfg(test)]
mod tests {
    use super::*;
    use me_core::{CodecId, MediaKind, Timebase};

    fn track(id: u32, kind: MediaKind, tps: u32) -> Track {
        Track::new(id, kind, CodecId::H264, Timebase::new(tps))
    }

    fn pkt(track_id: u32, dts: i64, pts: i64, size: usize) -> Packet {
        Packet::new(track_id, pts, dts, 0, vec![0u8; size])
    }

    #[test]
    fn interleaves_two_tracks_with_different_timebases() {
        // 视频 30000 时基、音频 44100 时基：直接比刻度是错的，必须换算成微秒
        let tracks = vec![track(1, MediaKind::Video, 30_000), track(2, MediaKind::Audio, 44_100)];
        let mut it = Interleaver::new(&tracks);
        // 视频每帧 1000 刻度 = 33.33ms；音频每帧 1024 刻度 = 23.22ms
        for i in 0..5 {
            it.push(pkt(1, i * 1000, i * 1000, 10)).unwrap();
        }
        for i in 0..7 {
            it.push(pkt(2, i * 1024, i * 1024, 4)).unwrap();
        }
        let out = it.drain_all();
        assert_eq!(out.len(), 12);
        // 换算成微秒后必须非递减
        let mut last = i64::MIN;
        for p in &out {
            let tps = if p.track_id == 1 { 30_000 } else { 44_100 };
            let us = p.dts * 1_000_000 / tps;
            assert!(us >= last, "输出必须按时间非递减：{us} < {last}");
            last = us;
        }
        // 第一条必须是视频（dts 都是 0，轨道号小的先出）——排序必须确定
        assert_eq!(out[0].track_id, 1);
    }

    #[test]
    fn same_timestamp_is_ordered_by_track_id_deterministically() {
        let tracks = vec![track(7, MediaKind::Audio, 1000), track(3, MediaKind::Video, 1000)];
        for _ in 0..8 {
            let mut it = Interleaver::new(&tracks);
            it.push(pkt(7, 0, 0, 1)).unwrap();
            it.push(pkt(3, 0, 0, 1)).unwrap();
            let out = it.drain_all();
            assert_eq!(out[0].track_id, 3, "dts 相同时按轨道号定序（不许随 HashMap 顺序变）");
        }
    }

    #[test]
    fn per_track_order_is_never_reordered() {
        // 视频有 B 帧：pts 会往回跳，但 dts 单调 —— 输出必须保持**推入顺序**
        let tracks = vec![track(1, MediaKind::Video, 30_000), track(2, MediaKind::Audio, 44_100)];
        let video_pts = [0i64, 3000, 1000, 2000, 6000, 4000, 5000];
        let mut it = Interleaver::new(&tracks);
        for (i, pts) in video_pts.iter().enumerate() {
            it.push(pkt(1, i as i64 * 1000, *pts, 1)).unwrap();
            it.push(pkt(2, i as i64 * 1024, i as i64 * 1024, 1)).unwrap();
        }
        let out = it.drain_all();
        let got: Vec<i64> = out.iter().filter(|p| p.track_id == 1).map(|p| p.pts).collect();
        assert_eq!(got, video_pts, "同轨内的顺序（含 pts 回跳）必须原样保留");
    }

    #[test]
    fn finished_track_does_not_block_the_others() {
        let tracks = vec![track(1, MediaKind::Video, 1000), track(2, MediaKind::Audio, 1000)];
        let mut it = Interleaver::new(&tracks);
        it.push(pkt(1, 0, 0, 1)).unwrap();
        it.finish_track(1); // 视频只有一条，到头了
        for i in 0..5 {
            it.push(pkt(2, i * 100, i * 100, 1)).unwrap();
        }
        let out = it.drain_all();
        assert_eq!(out.len(), 6, "一条轨道到头不该让别的轨道卡住");
        assert_eq!(out[0].track_id, 1);
    }

    #[test]
    fn window_prevents_unbounded_waiting() {
        // 只推视频、不推音频、也不声明音频到头：不能无限等下去
        let tracks = vec![track(1, MediaKind::Video, 1000), track(2, MediaKind::Audio, 1000)];
        let mut it = Interleaver::with_window_us(&tracks, 1000);
        // 视频时间戳跨了 500ms（远超 1ms 的窗口）
        for i in 0..10 {
            it.push(pkt(1, i * 100_000, i * 100_000, 1)).unwrap();
        }
        let mut out = Vec::new();
        while let Some(p) = it.pop() {
            out.push(p);
        }
        assert!(!out.is_empty(), "超过窗口后必须按现有信息输出，不能一直等");
    }

    #[test]
    fn buffering_stays_bounded_when_one_track_is_silent() {
        // 视频推 100 条、音频一条都不推：缓冲不能无上限地涨
        let tracks = vec![track(1, MediaKind::Video, 1000), track(2, MediaKind::Audio, 1000)];
        let mut it = Interleaver::with_window_us(&tracks, 50_000);
        for i in 0..100 {
            it.push(pkt(1, i * 10_000, i * 10_000, 1)).unwrap();
        }
        assert!(it.buffered() < 40, "超过窗口的必须已经输出，实际缓冲 {}", it.buffered());
    }

    #[test]
    fn unknown_track_is_an_error_not_a_panic() {
        let tracks = vec![track(1, MediaKind::Video, 1000)];
        let mut it = Interleaver::new(&tracks);
        let e = it.push(pkt(9, 0, 0, 1));
        assert!(e.is_err());
        assert!(e.unwrap_err().to_string().contains("轨道 9"));
    }

    #[test]
    fn interleave_all_groups_by_track() {
        let tracks = vec![track(1, MediaKind::Video, 1000), track(2, MediaKind::Audio, 1000)];
        let packets = vec![
            pkt(1, 0, 0, 1),
            pkt(2, 0, 0, 1),
            pkt(1, 100, 100, 1),
            pkt(2, 50, 50, 1),
            pkt(1, 200, 200, 1),
        ];
        let out = interleave_all(&tracks, &packets).unwrap();
        assert_eq!(out.len(), 5);
        let times: Vec<i64> = out.iter().map(|p| p.dts).collect();
        assert_eq!(times, vec![0, 0, 50, 100, 200], "按时间交织且同刻先出轨道号小的");
    }

    #[test]
    fn no_tracks_is_harmless() {
        let mut it = Interleaver::new(&[]);
        assert!(it.pop().is_none());
        it.flush();
        assert!(it.drain_all().is_empty());
    }
}
