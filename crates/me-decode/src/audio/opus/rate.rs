//! CELT 的**比特分配**（参考实现 `rate.c` 的 `compute_allocation` / `interp_bits2pulses`）。
//!
//! 这一步决定"每个频带能拿到多少位"，是后面细能量与 PVQ 形状的输入。它**不是纯算术**：
//! 解出来的位要通过区间解码器与编码器**逐位对齐**（跳过哪些带、强度立体声从哪带起、
//! 双重立体声开不开，都有专门的符号），所以只要分配算错一点，后面的符号全都会读偏。
//!
//! 表（`LOG_N` / `CACHE_CAPS_50` / `E_BANDS` / `BAND_ALLOCATION` / `LOG2_FRAC_TABLE`）
//! **由脚本从参考实现生成**（`/tmp/opus_gen_rate_tables.py`）—— 手抄一定会错。
//!
//! ## 顺序（照参考实现的解码顺序，别改）
//!
//! ```text
//! dynalloc 提升（每带一串 logp 符号）→ alloc_trim → compute_allocation
//!   （后者内部还会解"跳过带"符号、强度立体声、双重立体声）
//! ```

use me_core::Result;

use super::celt::NB_EBANDS;
use super::entropy::RangeDecoder;
use super::rate_tables::*;

/// 位分配的精度：一切"位"都以 1/8 位为单位（参考实现的 `BITRES`）。
pub const BITRES: u32 = 3;

/// 二分查找分配曲线的步数。
const ALLOC_STEPS: u32 = 6;

/// 细能量最多几个比特。
const MAX_FINE_BITS: i32 = 8;

/// 细能量的"公平份额"偏移。
const FINE_OFFSET: i32 = 21;

/// 分配曲线的倾斜表（参考实现的 `trim_icdf`）。
const TRIM_ICDF: [u8; 11] = [126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0];

/// "位" → 1/8 位（参考实现里到处是这种换算）。
const fn bits8(bits: i32) -> i32 {
    bits << BITRES
}

/// 每带的分配上限（参考实现 `init_caps`）。
pub fn init_caps(lm: u32, channels: usize) -> [i32; NB_EBANDS] {
    let mut cap = [0i32; NB_EBANDS];
    let c = channels as i32;
    for (i, slot) in cap.iter_mut().enumerate() {
        let n = ((E_BANDS[i + 1] - E_BANDS[i]) as i32) << lm;
        let idx = NB_EBANDS * (2 * lm as usize + channels - 1) + i;
        *slot = ((CACHE_CAPS_50[idx] as i32 + 64) * c * n) >> 2;
    }
    cap
}

/// 一帧的比特分配结果。
#[derive(Debug, Clone, PartialEq)]
pub struct Allocation {
    /// 每个频带分给 PVQ 形状的位数（1/8 位）。
    pub pulses: [i32; NB_EBANDS],
    /// 每个频带的细能量比特数（**位**，不是 1/8 位 —— 参考实现如此）。
    pub fine_quant: [i32; NB_EBANDS],
    pub fine_priority: [i32; NB_EBANDS],
    /// dynalloc 给每个带加的位（1/8 位）。
    pub offsets: [i32; NB_EBANDS],
    pub cap: [i32; NB_EBANDS],
    pub alloc_trim: i32,
    /// 实际编码到第几带（其余被跳过）。
    pub coded_bands: usize,
    /// 强度立体声从第几带起（0 = 不用）。
    pub intensity: i32,
    pub dual_stereo: bool,
    /// 留给 PVQ 再平衡的余量（1/8 位）。
    pub balance: i32,
    /// 反塌陷保留位。
    pub anti_collapse_rsv: i32,
    /// 交给 PVQ 的位数（1/8 位）。
    pub bits: i32,
}

/// 解出这一帧的比特分配（从 spread 之后接着读）。
///
/// `transient` 决定是否给"反塌陷"留一位；`channels` 是编码声道数。
pub fn decode_allocation(
    dec: &mut RangeDecoder<'_>,
    lm: u32,
    channels: usize,
    transient: bool,
    start: usize,
    end: usize,
) -> Result<Allocation> {
    let c = channels as i32;
    let cap = init_caps(lm, channels);

    // ① dynalloc：每个带一串"还要不要再加一点位"的 logp 符号
    let mut offsets = [0i32; NB_EBANDS];
    let mut total_bits = bits8((dec.len_bytes() * 8) as i32);
    let mut tell = dec.tell_frac() as i32;
    let mut dynalloc_logp = 6u32;
    for i in start..end {
        let width = c * ((E_BANDS[i + 1] - E_BANDS[i]) as i32) << lm;
        // quanta：一次提升给 1/8 位的量，夹在 1/8~1 位/样本之间
        let quanta = bits8(width).min(bits8(6).max(width));
        let mut dynalloc_loop_logp = dynalloc_logp;
        let mut boost = 0i32;
        while tell + bits8(dynalloc_loop_logp as i32) < total_bits && boost < cap[i] {
            let flag = dec.dec_bit_logp(dynalloc_loop_logp);
            tell = dec.tell_frac() as i32;
            if flag == 0 {
                break;
            }
            boost += quanta;
            total_bits -= quanta;
            dynalloc_loop_logp = 1;
        }
        offsets[i] = boost;
        if boost > 0 {
            dynalloc_logp = (dynalloc_logp - 1).max(2);
        }
    }

    // ② alloc_trim：分配曲线的整体倾斜（还有余量才解）
    let alloc_trim = if tell + bits8(6) <= total_bits {
        dec.dec_icdf(&TRIM_ICDF, 7)? as i32
    } else {
        5
    };

    // ③ 交给分配器的位数：整帧减去已经用掉的，再留 1 位给终止
    let mut bits = bits8((dec.len_bytes() * 8) as i32) - dec.tell_frac() as i32 - 1;
    let anti_collapse_rsv = if transient && lm >= 2 && bits >= ((lm as i32 + 2) << BITRES) {
        1 << BITRES
    } else {
        0
    };
    bits -= anti_collapse_rsv;

    // ④ 分配本身（内部还会解"跳过带"、强度立体声、双重立体声）
    let mut intensity = 0i32;
    let mut dual_stereo = 0i32;
    let mut balance = 0i32;
    let mut pulses = [0i32; NB_EBANDS];
    let mut fine_quant = [0i32; NB_EBANDS];
    let mut fine_priority = [0i32; NB_EBANDS];
    let coded_bands = compute_allocation(
        start, end, &offsets, &cap, alloc_trim, &mut intensity, &mut dual_stereo, bits,
        &mut balance, &mut pulses, &mut fine_quant, &mut fine_priority, channels, lm, dec,
    )?;

    Ok(Allocation {
        pulses,
        fine_quant,
        fine_priority,
        offsets,
        cap,
        alloc_trim,
        coded_bands,
        intensity,
        dual_stereo: dual_stereo != 0,
        balance,
        anti_collapse_rsv,
        bits,
    })
}

/// 参考实现 `compute_allocation`（解码路径）。
#[allow(clippy::too_many_arguments)]
fn compute_allocation(
    start: usize,
    end: usize,
    offsets: &[i32; NB_EBANDS],
    cap: &[i32; NB_EBANDS],
    alloc_trim: i32,
    intensity: &mut i32,
    dual_stereo: &mut i32,
    total_in: i32,
    balance: &mut i32,
    pulses: &mut [i32; NB_EBANDS],
    fine_quant: &mut [i32; NB_EBANDS],
    fine_priority: &mut [i32; NB_EBANDS],
    channels: usize,
    lm: u32,
    dec: &mut RangeDecoder<'_>,
) -> Result<usize> {
    let c = channels as i32;
    let mut total = total_in.max(0);
    let mut skip_start = start;
    // 留一位用于"手动跳过带"的终止符号
    let mut skip_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
    total -= skip_rsv;
    // 立体声要留强度/双重立体声的位
    let mut intensity_rsv = 0i32;
    let mut dual_stereo_rsv = 0i32;
    if c == 2 {
        intensity_rsv = LOG2_FRAC_TABLE[end - start] as i32;
        if intensity_rsv > total {
            intensity_rsv = 0;
        } else {
            total -= intensity_rsv;
            dual_stereo_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
            total -= dual_stereo_rsv;
        }
    }

    // 每带的阈值与倾斜
    let mut thresh = [0i32; NB_EBANDS];
    let mut trim_offset = [0i32; NB_EBANDS];
    for j in start..end {
        let width = (E_BANDS[j + 1] - E_BANDS[j]) as i32;
        thresh[j] = bits8(c).max(bits8(3 * width << lm) >> 4);
        trim_offset[j] = c * width * (alloc_trim - 5 - lm as i32) * (end - j - 1) as i32
            * (1 << (lm + BITRES))
            >> 6;
        // 单系数频带少给点（它靠"每个系数一个粗值"已经够好了）
        if width << lm == 1 {
            trim_offset[j] -= bits8(c);
        }
    }

    // 在分配曲线之间二分
    let nb_alloc = 11usize;
    let mut lo = 1usize;
    let mut hi = nb_alloc - 1;
    loop {
        let mut done = 0;
        let mut psum = 0i32;
        let mid = (lo + hi) >> 1;
        for j in (start..end).rev() {
            let n = (E_BANDS[j + 1] - E_BANDS[j]) as i32;
            let mut bitsj = c * n * (BAND_ALLOCATION[mid * NB_EBANDS + j] as i32) << lm >> 2;
            if bitsj > 0 {
                bitsj = (bitsj + trim_offset[j]).max(0);
            }
            bitsj += offsets[j];
            if bitsj >= thresh[j] || done != 0 {
                done = 1;
                psum += bitsj.min(cap[j]);
            } else if bitsj >= bits8(c) {
                psum += bits8(c);
            }
        }
        if psum > total {
            if mid == 0 {
                break;
            }
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
        if lo > hi {
            break;
        }
    }
    hi = lo;
    lo -= 1;

    let mut bits1 = [0i32; NB_EBANDS];
    let mut bits2 = [0i32; NB_EBANDS];
    for j in start..end {
        let n = (E_BANDS[j + 1] - E_BANDS[j]) as i32;
        let mut b1 = c * n * (BAND_ALLOCATION[lo * NB_EBANDS + j] as i32) << lm >> 2;
        let mut b2 = if hi >= nb_alloc {
            cap[j]
        } else {
            c * n * (BAND_ALLOCATION[hi * NB_EBANDS + j] as i32) << lm >> 2
        };
        if b1 > 0 {
            b1 = (b1 + trim_offset[j]).max(0);
        }
        if b2 > 0 {
            b2 = (b2 + trim_offset[j]).max(0);
        }
        if lo > 0 {
            b1 += offsets[j];
        }
        b2 += offsets[j];
        if offsets[j] > 0 {
            skip_start = j;
        }
        bits2[j] = (b2 - b1).max(0);
        bits1[j] = b1;
    }

    let coded_bands = interp_bits2pulses(
        start,
        end,
        skip_start,
        &bits1,
        &bits2,
        &thresh,
        cap,
        total,
        balance,
        &mut skip_rsv,
        intensity,
        &mut intensity_rsv,
        dual_stereo,
        &mut dual_stereo_rsv,
        pulses,
        fine_quant,
        fine_priority,
        channels,
        lm,
        dec,
    )?;
    Ok(coded_bands)
}

/// 参考实现 `interp_bits2pulses`（解码路径）。
#[allow(clippy::too_many_arguments)]
fn interp_bits2pulses(
    start: usize,
    end: usize,
    skip_start: usize,
    bits1: &[i32; NB_EBANDS],
    bits2: &[i32; NB_EBANDS],
    thresh: &[i32; NB_EBANDS],
    cap: &[i32; NB_EBANDS],
    total_in: i32,
    balance_out: &mut i32,
    skip_rsv: &mut i32,
    intensity: &mut i32,
    intensity_rsv: &mut i32,
    dual_stereo: &mut i32,
    dual_stereo_rsv: &mut i32,
    bits: &mut [i32; NB_EBANDS],
    ebits: &mut [i32; NB_EBANDS],
    fine_priority: &mut [i32; NB_EBANDS],
    channels: usize,
    lm: u32,
    dec: &mut RangeDecoder<'_>,
) -> Result<usize> {
    let c = channels as i32;
    let stereo_level = if channels > 1 { 1 } else { 0 };
    let mut total = total_in;
    let alloc_floor = bits8(c);
    let log_m = (lm as i32) << BITRES;

    // 二分插值参数
    let mut lo = 0i32;
    let mut hi = 1i32 << ALLOC_STEPS;
    for _ in 0..ALLOC_STEPS {
        let mid = (lo + hi) >> 1;
        let mut psum = 0i32;
        let mut done = 0;
        for j in (start..end).rev() {
            let tmp = bits1[j] + ((mid * bits2[j]) >> ALLOC_STEPS);
            if tmp >= thresh[j] || done != 0 {
                done = 1;
                psum += tmp.min(cap[j]);
            } else if tmp >= alloc_floor {
                psum += alloc_floor;
            }
        }
        if psum > total {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let mut psum = 0i32;
    let mut done = 0;
    for j in (start..end).rev() {
        let mut tmp = bits1[j] + ((lo * bits2[j]) >> ALLOC_STEPS);
        if tmp < thresh[j] && done == 0 {
            tmp = if tmp >= alloc_floor { alloc_floor } else { 0 };
        } else {
            done = 1;
        }
        tmp = tmp.min(cap[j]);
        bits[j] = tmp;
        psum += tmp;
    }

    // 从后往前决定跳过哪些带（解一个"还要不要继续跳"的符号）
    let mut coded_bands = end;
    loop {
        let j = coded_bands - 1;
        if j <= skip_start {
            total += *skip_rsv;
            break;
        }
        let mut left = total - psum;
        let percoeff = left / ((E_BANDS[coded_bands] - E_BANDS[start]) as i32);
        left -= (E_BANDS[coded_bands] - E_BANDS[start]) as i32 * percoeff;
        let rem = (left - (E_BANDS[j] - E_BANDS[start]) as i32).max(0);
        let band_width = (E_BANDS[coded_bands] - E_BANDS[j]) as i32;
        let mut band_bits = bits[j] + percoeff * band_width + rem;
        if band_bits >= thresh[j].max(alloc_floor + (1 << BITRES)) {
            if dec.dec_bit_logp(1) != 0 {
                break; // 这个带也要跳过
            }
            psum += 1 << BITRES;
            band_bits -= 1 << BITRES;
        }
        // 把这个带原来的分配收回
        psum -= bits[j] + *intensity_rsv;
        if *intensity_rsv > 0 {
            *intensity_rsv = LOG2_FRAC_TABLE[j - start] as i32;
        }
        psum += *intensity_rsv;
        if band_bits >= alloc_floor {
            psum += alloc_floor;
            bits[j] = alloc_floor;
        } else {
            bits[j] = 0;
        }
        coded_bands -= 1;
    }

    // 强度与双重立体声
    if *intensity_rsv > 0 {
        *intensity = start as i32 + dec.dec_uint((coded_bands + 1 - start) as u32)? as i32;
    } else {
        *intensity = 0;
    }
    if *intensity <= start as i32 {
        total += *dual_stereo_rsv;
        *dual_stereo_rsv = 0;
    }
    if *dual_stereo_rsv > 0 {
        *dual_stereo = dec.dec_bit_logp(1) as i32;
    } else {
        *dual_stereo = 0;
    }

    // 剩下的位按系数个数平分给已编码的带
    let mut left = total - psum;
    let percoeff = left / ((E_BANDS[coded_bands] - E_BANDS[start]) as i32);
    left -= (E_BANDS[coded_bands] - E_BANDS[start]) as i32 * percoeff;
    for j in start..coded_bands {
        bits[j] += percoeff * (E_BANDS[j + 1] - E_BANDS[j]) as i32;
    }
    for j in start..coded_bands {
        let tmp = left.min((E_BANDS[j + 1] - E_BANDS[j]) as i32);
        bits[j] += tmp;
        left -= tmp;
    }

    // 细分能量与 PVQ 的位数；这里还要做一次"再平衡"
    let mut balance = 0i32;
    let mut j = start;
    while j < coded_bands {
        let n0 = (E_BANDS[j + 1] - E_BANDS[j]) as i32;
        let n = n0 << lm;
        bits[j] += balance;
        let excess;
        if n > 1 {
            excess = (bits[j] - cap[j]).max(0);
            bits[j] -= excess;
            // 立体声的自由度多一倍（强度立体声以上还要多一个）
            let den = c * n
                + i32::from(c == 2 && n > 2 && *dual_stereo == 0 && (j as i32) < *intensity);
            let nc_log_n = den * (LOG_N[j] as i32 + log_m);
            let mut offset = (nc_log_n >> 1) - den * FINE_OFFSET;
            if n == 2 {
                offset += (den << BITRES) >> 2;
            }
            if bits[j] + offset < (den * 2) << BITRES {
                offset += nc_log_n >> 2;
            } else if bits[j] + offset < (den * 3) << BITRES {
                offset += nc_log_n >> 3;
            }
            ebits[j] = (bits[j] + offset + (den << (BITRES - 1))) / (den << BITRES);
            ebits[j] = ebits[j].max(0);
            if c * ebits[j] > (bits[j] >> BITRES) {
                ebits[j] = bits[j] >> stereo_level >> BITRES;
            }
            ebits[j] = ebits[j].min(MAX_FINE_BITS);
            fine_priority[j] = i32::from(ebits[j] * (den << BITRES) >= bits[j] + offset);
            bits[j] -= c * ebits[j] << BITRES;
        } else {
            // 单系数带：位全给细能量（留一个符号位）
            excess = (bits[j] - bits8(c)).max(0);
            bits[j] -= excess;
            ebits[j] = 0;
            fine_priority[j] = 1;
        }
        if excess > 0 {
            let extra_fine = (excess >> (stereo_level + BITRES)).min(MAX_FINE_BITS - ebits[j]);
            ebits[j] += extra_fine;
            let extra_bits = extra_fine * c << BITRES;
            fine_priority[j] = i32::from(extra_bits >= excess - balance);
            balance = excess - extra_bits;
        } else {
            balance = excess;
        }
        j += 1;
    }
    *balance_out = balance;

    // 被跳过的带：位全给细能量
    while j < end {
        ebits[j] = bits[j] >> stereo_level >> BITRES;
        bits[j] = 0;
        fine_priority[j] = i32::from(ebits[j] < 1);
        j += 1;
    }

    Ok(coded_bands)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_edges_and_tables_are_well_formed() {
        // 频带边界：21 个带 → 22 个边界，单调递增，最后一个是 100（2.5 ms 单位）
        assert_eq!(E_BANDS.len(), NB_EBANDS + 1);
        assert!(E_BANDS.windows(2).all(|w| w[0] < w[1]), "频带边界必须严格递增");
        assert_eq!(E_BANDS[0], 0);
        assert_eq!(E_BANDS[NB_EBANDS], 100);
        // 表长度与参考实现一致（脚本生成并回读校验，这里再钉一次长度）
        assert_eq!(LOG_N.len(), NB_EBANDS);
        assert_eq!(CACHE_CAPS_50.len(), NB_EBANDS * 8, "LM 0..3 × 声道 1..2 = 8 组");
        assert_eq!(BAND_ALLOCATION.len(), 11 * NB_EBANDS, "11 条分配曲线");
        assert_eq!(LOG2_FRAC_TABLE.len(), 24, "带数最多 21，加 3 个余量");
        // 0 码率那条曲线应当全是 0
        assert!(BAND_ALLOCATION[..NB_EBANDS].iter().all(|&v| v == 0));
        // log2 分数表单调不减
        assert!(LOG2_FRAC_TABLE.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn caps_are_positive_and_follow_the_formula() {
        for lm in 0..4u32 {
            for channels in 1..=2usize {
                let cap = init_caps(lm, channels);
                let c = channels as i32;
                for i in 0..NB_EBANDS {
                    let n = ((E_BANDS[i + 1] - E_BANDS[i]) as i32) << lm;
                    let idx = NB_EBANDS * (2 * lm as usize + channels - 1) + i;
                    let want = ((CACHE_CAPS_50[idx] as i32 + 64) * c * n) >> 2;
                    assert_eq!(cap[i], want, "LM={lm} C={channels} 第 {i} 带");
                    assert!(cap[i] > 0, "上限必须为正");
                }
            }
        }
    }

    #[test]
    fn trim_table_is_the_documented_one() {
        // 参考实现的 trim_icdf：非增、以 0 收尾（icdf 表的硬要求）
        assert_eq!(TRIM_ICDF.len(), 11);
        assert_eq!(TRIM_ICDF[10], 0, "icdf 表必须以 0 收尾");
        assert!(TRIM_ICDF.windows(2).all(|w| w[0] >= w[1]), "icdf 应当单调不增");
    }

    #[test]
    fn allocation_without_bits_is_all_zero() {
        // 位数耗尽（静音帧就是这样）时不该给任何带分位：这里直接对着一串 0 字节跑一遍
        let data = [0u8; 32];
        let mut dec = RangeDecoder::new(&data);
        dec.force_tell(32 * 8); // 假装整帧都读过了
        let a = decode_allocation(&mut dec, 3 /* lm=3 → 20 ms */, 1, false, 0, 21).unwrap();
        assert!(a.pulses.iter().all(|&v| v == 0), "没有位可分：pulses 应全为 0");
        assert_eq!(a.alloc_trim, 5, "没余量时 alloc_trim 取缺省 5");
    }
}
