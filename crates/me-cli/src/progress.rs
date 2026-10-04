//! 流式进度上报（NDJSON 一行一条）。
//!
//! 为什么要有节流：下载/合并会在一秒内产生几千个事件，逐个吐给宿主会让
//! 渲染进程忙于解析 JSON、界面反而卡住。这里按"时间间隔"节流，
//! 并且**最后一定补一条 100%**，否则进度条会永远停在 99%。

use std::time::{Duration, Instant};

use crate::json;

/// 两次进度行之间的最小间隔。
const MIN_INTERVAL: Duration = Duration::from_millis(200);

pub struct Progress {
    enabled: bool,
    last: Instant,
    emitted: u64,
    kind: &'static str,
}

impl Progress {
    pub fn new(enabled: bool, kind: &'static str) -> Self {
        Self {
            enabled,
            // 让第一次上报立刻发生（否则小任务全程没有进度）
            last: Instant::now() - MIN_INTERVAL,
            emitted: 0,
            kind,
        }
    }

    /// 该不该现在上报（节流）。
    pub fn due(&self) -> bool {
        self.enabled && self.last.elapsed() >= MIN_INTERVAL
    }

    pub fn maybe_report(&mut self, data: &serde_json::Value) {
        if self.due() {
            self.report(data);
        }
    }

    pub fn report(&mut self, data: &serde_json::Value) {
        if !self.enabled {
            return;
        }
        self.last = Instant::now();
        self.emitted += 1;
        println!("{}", json::progress_line(self.kind, data.clone()));
    }

    /// 收尾：补一条最终进度（100%）。
    pub fn finish(&mut self, data: &serde_json::Value) {
        self.report(data);
    }

    /// 已经上报过多少条（只给测试用：验证节流行为）。
    #[cfg(test)]
    pub fn emitted(&self) -> u64 {
        self.emitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_report_is_immediate() {
        let p = Progress::new(true, "mux");
        assert!(p.due(), "第一次必须立刻上报，否则小任务全程没有进度");
    }

    #[test]
    fn disabled_never_reports() {
        let mut p = Progress::new(false, "mux");
        assert!(!p.due());
        p.report(&serde_json::json!({"percent": 50}));
        assert_eq!(p.emitted(), 0);
    }

    #[test]
    fn throttles_back_to_back_reports() {
        let mut p = Progress::new(true, "mux");
        p.report(&serde_json::json!({"percent": 1}));
        assert!(!p.due(), "紧接着不该再报（宿主会被刷屏）");
        assert_eq!(p.emitted(), 1);
    }
}
