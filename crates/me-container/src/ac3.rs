//! AC-3 的配置盒（`dac3`）解析。
//!
//! ## 为什么需要它
//!
//! fMP4 里的 AC-3 采样条目是 `ac-3`，后面必须跟一个 `dac3` 配置盒
//! （ETSI TS 102 366 Annex F）。而 **MPEG-TS 与裸流里没有这个盒子** ——
//! 参数（采样率码、码率码、声道配置…）全都写在每个 syncframe 的头里。
//!
//! 所以把 TS / 裸 AC-3 合并进 fMP4 时，只能从**码流的第一帧**把这些字段读出来，
//! 再拼成 `dac3`。读侧（`me-container/src/mp4.rs`）本来就在解析 `dac3`，
//! 这里补的是对称的那一半。
//!
//! ## 字段布局（`dac3`，3 字节 = 24 位，大端位序）
//!
//! ```text
//! fscod          2  采样率码（0=48k 1=44.1k 2=32k）
//! bsid           5  比特流标识（8 = AC-3 的常规值）
//! bsmod          3  比特流模式
//! acmod          3  声道配置
//! lfeon          1  LFE 开关
//! bit_rate_code  5  码率码（= syncframe 的 frmsizecod >> 1）
//! reserved       5  保留（0）
//! ```

/// 位读取器（大端位序）。
///
/// 只在本模块用：AC-3 的头部字段是**跨字节的位字段**，按位读比手写掩码清楚得多。
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read(&mut self, n: u32) -> Option<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            let byte = *self.data.get(self.pos / 8)?;
            let bit = (byte >> (7 - (self.pos % 8))) & 1;
            v = (v << 1) | u32::from(bit);
            self.pos += 1;
        }
        Some(v)
    }

    /// 读过但不关心取值的字段（`cmixlev` / `surmixlev` 这类）。
    fn skip(&mut self, n: u32) -> Option<()> {
        self.read(n).map(|_| ())
    }
}

/// 从 AC-3 的 syncframe 里读出来的配置盒字段。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dac3 {
    /// 采样率码：0=48 kHz、1=44.1 kHz、2=32 kHz。
    pub fscod: u8,
    /// 比特流标识：AC-3 是 8（E-AC-3 是 16 及以上）。
    pub bsid: u8,
    /// 比特流模式。
    pub bsmod: u8,
    /// 声道配置（0=1+1、1=单声道、2=立体声、3~7=多声道）。
    pub acmod: u8,
    /// LFE 开关。
    pub lfeon: u8,
    /// 码率码（0~18），换算表在规范里。
    pub bit_rate_code: u8,
}

impl Dac3 {
    /// 拼成 `dac3` 的 24 位载荷。
    pub fn to_payload(self) -> [u8; 3] {
        let brc = self.bit_rate_code & 0x1F;
        let bsmod = self.bsmod & 0x07;
        [
            ((self.fscod & 0x03) << 6) | ((self.bsid & 0x1F) << 1) | (bsmod >> 2),
            ((bsmod & 0x03) << 6)
                | ((self.acmod & 0x07) << 3)
                | ((self.lfeon & 0x01) << 2)
                | (brc >> 3),
            (brc & 0x07) << 5,
        ]
    }
}

/// AC-3 syncframe 的同步字。
const AC3_SYNC: u16 = 0x0B77;

/// 从一帧 AC-3 码流里读出 `dac3` 需要的字段。
///
/// 传进来的应当是**一整个 syncframe**（至少它的头部，前 8 字节就够）。
/// 不是 AC-3（同步字不对 / 太短）时返回 `None` —— 调用方据此报"这不是 AC-3"，
/// 而不是写一个字段全 0 的配置盒出去。
pub fn parse_dac3(frame: &[u8]) -> Option<Dac3> {
    if frame.len() < 8 {
        return None;
    }
    if u16::from_be_bytes([frame[0], frame[1]]) != AC3_SYNC {
        return None;
    }
    // syncword(16) + crc1(16) 之后才是字段
    let mut r = BitReader::new(&frame[4..]);
    let fscod = r.read(2)?;
    let frmsizecod = r.read(6)?;
    let bsid = r.read(5)?;
    let bsmod = r.read(3)?;
    let acmod = r.read(3)?;
    // 下面三个字段是**条件出现**的，跳错一位后面全错位
    if (acmod & 0x1) != 0 && acmod != 0x1 {
        r.skip(2)?; // cmixlev
    }
    if (acmod & 0x4) != 0 {
        r.skip(2)?; // surmixlev
    }
    if acmod == 0x2 {
        r.skip(2)?; // dsurmod
    }
    let lfeon = r.read(1)?;
    Some(Dac3 {
        fscod: fscod as u8,
        bsid: bsid as u8,
        bsmod: bsmod as u8,
        acmod: acmod as u8,
        lfeon: lfeon as u8,
        bit_rate_code: (frmsizecod >> 1) as u8,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 造一个 AC-3 syncframe 头。`acmod` 决定后面有几个条件字段。
    fn ac3_frame(fscod: u8, frmsizecod: u8, bsid: u8, bsmod: u8, acmod: u8, lfeon: u8) -> Vec<u8> {
        let mut bits: Vec<u8> = Vec::new();
        let mut push = |v: u32, n: u32| {
            for i in (0..n).rev() {
                bits.push(((v >> i) & 1) as u8);
            }
        };
        push(fscod as u32, 2);
        push(frmsizecod as u32, 6);
        push(bsid as u32, 5);
        push(bsmod as u32, 3);
        push(acmod as u32, 3);
        // 条件字段一律填 0（解析器只需跳过它们，取值不参与 dac3）
        if (acmod & 0x1) != 0 && acmod != 0x1 {
            push(0, 2);
        }
        if (acmod & 0x4) != 0 {
            push(0, 2);
        }
        if acmod == 0x2 {
            push(0, 2);
        }
        push(lfeon as u32, 1);

        let mut out = vec![0x0B, 0x77, 0x00, 0x00]; // syncword + crc1
        let mut byte = 0u8;
        let mut n = 0;
        for b in bits {
            byte = (byte << 1) | b;
            n += 1;
            if n == 8 {
                out.push(byte);
                byte = 0;
                n = 0;
            }
        }
        if n > 0 {
            out.push(byte << (8 - n));
        }
        out.resize(out.len().max(8), 0);
        out
    }

    #[test]
    fn parses_a_stereo_frame() {
        // 立体声（acmod=2）：有一条 dsurmod 条件字段，跳错就全错位
        let f = ac3_frame(0, 20, 8, 0, 2, 0);
        let d = parse_dac3(&f).expect("应当认出来");
        assert_eq!(d.fscod, 0, "48 kHz");
        assert_eq!(d.bsid, 8);
        assert_eq!(d.acmod, 2);
        assert_eq!(d.lfeon, 0);
        assert_eq!(d.bit_rate_code, 10, "frmsizecod 20 >> 1");
    }

    #[test]
    fn parses_a_51_frame_with_lfe() {
        // 5.1（acmod=7）：cmixlev + surmixlev 两条都要跳，且带 LFE
        let f = ac3_frame(0, 38, 8, 0, 7, 1);
        let d = parse_dac3(&f).expect("应当认出来");
        assert_eq!(d.acmod, 7);
        assert_eq!(d.lfeon, 1, "LFE 开关跳错位就会读到 0");
        assert_eq!(d.bit_rate_code, 19);
    }

    #[test]
    fn parses_a_mono_frame_without_extra_fields() {
        // acmod=1（单声道）：cmixlev 不出现。跳多了就会把 lfeon 读错
        let f = ac3_frame(1, 8, 8, 0, 1, 0);
        let d = parse_dac3(&f).expect("应当认出来");
        assert_eq!(d.fscod, 1, "44.1 kHz");
        assert_eq!(d.acmod, 1);
        assert_eq!(d.bit_rate_code, 4);
    }

    #[test]
    fn payload_roundtrips_through_the_bit_layout() {
        let d = Dac3 { fscod: 2, bsid: 8, bsmod: 5, acmod: 7, lfeon: 1, bit_rate_code: 19 };
        let p = d.to_payload();
        // 逐位核对：fscod(2) bsid(5) bsmod(3) acmod(3) lfeon(1) bit_rate_code(5) reserved(5)
        let v = ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | p[2] as u32;
        assert_eq!((v >> 22) & 0x03, 2, "fscod");
        assert_eq!((v >> 17) & 0x1F, 8, "bsid");
        assert_eq!((v >> 14) & 0x07, 5, "bsmod");
        assert_eq!((v >> 11) & 0x07, 7, "acmod");
        assert_eq!((v >> 10) & 0x01, 1, "lfeon");
        assert_eq!((v >> 5) & 0x1F, 19, "bit_rate_code");
        assert_eq!(v & 0x1F, 0, "reserved 必须是 0");
    }

    #[test]
    fn rejects_non_ac3_and_short_input() {
        assert!(parse_dac3(&[]).is_none());
        assert!(parse_dac3(&[0x0B, 0x77]).is_none(), "太短");
        assert!(parse_dac3(&[0xFF, 0xFB, 0x90, 0x00, 0, 0, 0, 0]).is_none(), "那是 MP3 的同步字");
        assert!(
            parse_dac3(&[0x0B, 0x77, 0, 0, 0, 0, 0, 0, 0, 0]).is_some(),
            "同步字对了就该认（字段可以是 0）"
        );
    }
}
