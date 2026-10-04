//! 时间基与时间换算。
//!
//! 全引擎**只用整数**表达时间。理由见 crate 文档第 1 条。

use serde::{Deserialize, Serialize};

/// 有理数时间基（`num / den` 秒）。
///
/// 为什么不用"每秒刻度数"一个整数：MPEG-TS 的 PCR/PTS 是 90kHz，
/// fMP4 常见 1000 或 90000，而**音频采样率**（44100）本身就是一个时基 ——
/// 用 `Rational` 表示能统一处理，也不会在 44100 → 90000 这类换算上引入误差。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rational {
    pub num: u32,
    pub den: u32,
}

impl Rational {
    pub const fn new(num: u32, den: u32) -> Self {
        Self { num, den }
    }

    /// 每秒的刻度数（时基的"倒数"）。
    pub fn ticks_per_second(&self) -> u64 {
        if self.num == 0 {
            return 0;
        }
        self.den as u64 / self.num as u64
    }

    /// 一个刻度是多少纳秒（用于精确换算，避免浮点）。
    pub fn nanos_per_tick(&self) -> u64 {
        if self.den == 0 {
            return 0;
        }
        // num/den 秒 = num * 1e9 / den 纳秒
        (self.num as u64) * 1_000_000_000 / (self.den as u64)
    }
}

/// 一条轨道的时基（每秒刻度数 + 便于换算的辅助）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Timebase {
    /// 每秒的刻度数（MPEG-TS = 90000，fMP4 常见 1000/90000，音频 = 采样率）。
    pub ticks_per_second: u32,
}

impl Timebase {
    pub const fn new(ticks_per_second: u32) -> Self {
        Self { ticks_per_second }
    }

    /// MPEG-TS 的时基（90kHz，规范固定）。
    pub const MPEG_TS: Timebase = Timebase::new(90_000);

    /// 刻度 → 微秒（**先乘后除**，把精度损失压到最小）。
    pub fn ticks_to_us(&self, ticks: i64) -> i64 {
        if self.ticks_per_second == 0 {
            return 0;
        }
        // ticks * 1e6 / tps：用 i128 中间量防溢出（90kHz 下 i64 也够，
        // 但 fMP4 里见过 1e9 的时基，乘 1e6 就爆了）
        let v = (ticks as i128) * 1_000_000i128;
        (v / self.ticks_per_second as i128) as i64
    }

    /// 微秒 → 刻度（四舍五入，保证"换算过去再换回来"不会丢一格）。
    pub fn us_to_ticks(&self, us: i64) -> i64 {
        let v = (us as i128) * self.ticks_per_second as i128;
        let half = 500_000i128; // 1e6 / 2
        let r = if v >= 0 { (v + half) / 1_000_000 } else { (v - half) / 1_000_000 };
        r as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ts_timebase_roundtrip() {
        let tb = Timebase::MPEG_TS;
        for us in [0i64, 1, 1000, 33_366, 1_000_000, 7_200_000_000] {
            let ticks = tb.us_to_ticks(us);
            let back = tb.ticks_to_us(ticks);
            // 90kHz 的一格是 11.11us，误差必须在一格之内
            assert!((back - us).abs() <= 12, "us={us} ticks={ticks} back={back}");
        }
    }

    #[test]
    fn audio_timebase_is_exact_at_frame_boundaries() {
        // 44100Hz 下一个 1024 采样的 AAC 帧 = 23219.95us；
        // 按刻度算必须精确到采样
        let tb = Timebase::new(44_100);
        assert_eq!(tb.us_to_ticks(1_000_000), 44_100);
        assert_eq!(tb.ticks_to_us(44_100), 1_000_000);
        assert_eq!(tb.us_to_ticks(tb.ticks_to_us(1024)), 1024);
    }

    #[test]
    fn huge_timescale_does_not_overflow() {
        // fMP4 里出现过 1e9 时基：ticks * 1e6 会溢出 i64，这里必须走 i128
        let tb = Timebase::new(1_000_000_000);
        assert_eq!(tb.ticks_to_us(2_000_000_000), 2_000_000);
    }

    #[test]
    fn zero_timebase_is_inert_not_panicking() {
        let tb = Timebase::new(0);
        assert_eq!(tb.ticks_to_us(123), 0);
    }

    #[test]
    fn rational_nanos() {
        assert_eq!(Rational::new(1, 90_000).nanos_per_tick(), 11_111);
        assert_eq!(Rational::new(1, 1000).nanos_per_tick(), 1_000_000);
        assert_eq!(Rational::new(0, 1000).nanos_per_tick(), 0);
    }
}
