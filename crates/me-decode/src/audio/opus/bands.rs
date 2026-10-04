//! CELT 的**频带级**操作（参考实现 `bands.c` / `quant_bands.c` 里不属于纯熵解码的部分）。
//!
//! 落地三块：
//!
//! 1. **细能量**（`unquant_fine_energy` / `unquant_energy_finalise`）：粗能量只到 6 dB
//!    一档，细能量再补 `fine_quant[i]` 位（浮点公式 `(q2+0.5)·2^(14-q)/16384 - 0.5`）；
//!    帧尾再把没用完的零头一位一符号地补上。它直接写 `old_ebands` —— **跨帧状态到
//!    这一步才定型**（这也是上一轮对拍时"必须用参考的帧前状态播种"的原因）。
//! 2. **频带形状**（[`quant_all_bands`]）：把每个带分到的位数换成"单位范数的形状"。
//! 3. **归一化还原**（[`denormalise_bands`]）：把形状按每带能量放大成频谱，交给 IMDCT。
//!    频带之外的系数一律置 0（带宽窄的帧就是这样省位的）。
//!
//! ## 形状解码为什么长这样
//!
//! 每个带要做的是"用 L1 范数为 K 的整数矢量近似 N 个 MDCT 系数"，但直接这么干在低频
//! （N 大）精度不够、在高频（N 小）又浪费位数。参考实现把它们拼成一棵**递归二分树**：
//!
//! ```text
//! quant_band(N) ──┬─ 位数超过缓存能表达的上限（N>2）→ 劈成两半，各自递归
//!                 ├─ 立体声 → 劈成 mid/side，用 θ 描述两者能量比
//!                 └─ 否则   → 直接 PVQ（alg_unquant）
//! ```
//!
//! 每个内部节点都要**解一个 θ**（两半的能量怎么分）；θ==0 / θ==16384 是"全给一半"
//! 的退化情形，另一半靠**折叠**从低带借谱、不占位。θ 走位精确的整数路径
//! （`bitexact_cos` / `bitexact_log2tan`），因为它会影响后面的比特分配本身。
//!
//! 两件"不做就会听到噪声"的事：
//!
//! - **折叠**（fold）：没分到脉冲的高带不静音，而是从**低带**搬一段谱过来
//!   （`lowband`），再叠一个 ±1/256 的微噪声；`lowband_offset` 决定从哪搬。
//! - **反塌陷**（[`anti_collapse`]）：瞬态帧里某个短块可能被量化成全 0，听感是"咔"；
//!   这里按"当前能量 − 近两帧能量"填噪声。
//!
//! ## 对拍纪律（这一层尤其要守）
//!
//! - 每个带开头的 `tell_frac` 参与 `b` 的计算，**必须逐位对齐**。
//! - 折叠与填噪吃 LCG 种子，种子是**跨帧状态**（参考实现每帧末把它设成解码器最终的
//!   区间状态）—— 对拍时用参考 dump 的 `SEED` 播种。
//! - 参考实现 dump 的 `pulses[]` / `tf_res[]` 在 `[start,end)` 之外是**未初始化内存**，
//!   比对只比这个区间。
//!
//! ## 递归轨迹（调这一层时最有用的工具）
//!
//! 带级对拍（`tests/opus_celt_shape_ref.rs` 里的 `QB` 行）只能告诉你"哪一带开始错"，
//! 而这一层是个递归树，真正要定位得看**每一次 `quant_band` 调用**。所以
//! `ShapeDec::trace` 会在设了 `OPUS_SHAPE_TRACE=<帧号>` 时，对指定的那一帧打印：
//!
//! ```text
//! QIN  <层> <带> <N> <b> <B> <LM> <tell_frac> <val> <rng>
//! QD   <层> <带> <stereo> <qn> <itheta> <qalloc> <b> <N> <LM>   θ 解完之后
//! QS   <层> <带> <mbits> <sbits> <delta> <itheta>               两半的位数定完之后
//! QOUT <层> <带> <cm> <tell_frac>                               这一次调用返回时
//! ```
//!
//! 参考实现的 `bands.c` 有完全对应的插桩（`/tmp/opus_patch_dbg.py`），两边 `diff`
//! 一下就能把范围缩到"哪一次调用、哪个符号"。下面这些坑都是这么抓出来的。
//!
//! ## 已经踩过的坑（都是对拍/轨迹抓出来的）
//!
//! - **`effective_lowband` 用的是 `eBands[lowband_offset]`，不是当前带的边界。**
//!   写成当前带的话，从第 2 带起每次都会自己往前挪一格，折叠源整条错位。
//! - **`SHR32(mid,1)` 在浮点构建下是空操作** —— `mid2` 就是 `mid` 本身。
//!   按 Q15→Q14 的注释去除以 2，会给 X 多出一个 2 倍增益，还会让 `El/Er`
//!   恰好避开"能量太小就塌陷"的保护（症状：只有前几带对，后面全是 2 倍）。
//! - **三角分布 θ 的 `isqrt32(8*(ft-fm-1)+1)` 里那个 `+1` 不能少**：
//!   少了它 `isqrt` 的取整边界整体偏一格，θ 差一档 → `mbits/sbits` 差十几位 →
//!   只有少数帧的最后几带对不上（很难从频谱上看出来，必须靠轨迹）。
//! - **单声道"劈成两半"是在 `x` 内部劈**（前一半当 mid、后一半当 side），
//!   和立体声分裂用 `x`/`y` 两个缓冲区是两回事，别混着写。

use me_core::Result;

use super::bands_tables::{CACHE_BITS, CACHE_INDEX};
use super::celt::{CeltState, NB_EBANDS, SHORT_MDCT};
use super::entropy::RangeDecoder;
use super::rate::{Allocation, BITRES};
use super::mathops::{celt_exp2, celt_rsqrt, celt_sqrt};
use super::rate_tables::{E_BANDS, LOG_N};
use super::vq::{alg_unquant, lcg_rand, renormalise_vector};

/// `MAX_FINE_BITS`：细能量一位最多补到 8 位。
const MAX_FINE_BITS: i32 = 8;

/// `bits2pulses` 里的二分步数（参考实现的 `LOG_MAX_PSEUDO`）。
const LOG_MAX_PSEUDO: usize = 6;

/// θ 量化的两个偏移（参考实现 `rate.h`）。N==2 的立体声走"两相"路径，偏移更大。
const QTHETA_OFFSET: i32 = 4;
const QTHETA_OFFSET_TWOPHASE: i32 = 16;

/// 扩散决策里 `SPREAD_AGGRESSIVE` 的取值（前端解出来的 spread 就是这个数域）。
const SPREAD_AGGRESSIVE: u32 = 3;

/// 参考实现 `exp2_table8`：`compute_qn` 用它把 qb 变成 θ 的量化档数。
const EXP2_TABLE8: [i32; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];

/// 参考实现 `ordery_table`：哈达玛序（`B0<<recombine` 个时隙的重排顺序）。
const ORDERY_TABLE: [usize; 30] = [
    1, 0, //
    3, 0, 2, 1, //
    7, 0, 4, 3, 6, 1, 5, 2, //
    15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5,
];

/// 参考实现 `bit_interleave_table`（重组时把折叠掩码按 bit 交错）。
const BIT_INTERLEAVE_TABLE: [u8; 16] = [0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3];

/// 参考实现 `bit_deinterleave_table`（上面那步的逆操作）。
const BIT_DEINTERLEAVE_TABLE: [u8; 16] = [
    0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F, 0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3, 0xFC, 0xFF,
];

/// 标准 48 kHz 模式的 `effEBands`（= `nbEBands`）。
///
/// 参考实现里 `i >= effEBands` 的带会把输出重定向到一块**公用 scratch**（那些带超出
/// 短 MDCT 的长度）。只有 Opus Custom 模式才会出现这种情况，本引擎不支持 —— 所以这个
/// 常量的意义是"把那条死路用断言钉死"，而不是留一段没被验证过的代码。
pub const EFF_EBANDS: usize = 21;

// ── 表查询 ──

/// 频带边界（单位是 2.5 ms 的 MDCT 系数个数，长度 `NB_EBANDS+1`）。
///
/// 只暴露给对拍测试把"第 j 个系数"换算成带号，好让报错信息能直接指到带上。
pub fn band_edge(i: usize) -> usize {
    E_BANDS[i] as usize
}

/// 参考实现 `get_pulses`：把"脉冲数编码" q 映射成实际脉冲数 K。
///
/// q 是"每 8 个一组的对数刻度"：q<8 时 K=q；再往上每加一组就翻倍、尾数加 8。
/// 小的 K 精确、大的 K 只用几个符号就能表示 —— 这是 CELT 用一个字节表达 K 的办法。
#[inline]
pub fn get_pulses(i: i32) -> usize {
    if i < 8 {
        i as usize
    } else {
        ((8 + (i & 7)) << ((i >> 3) - 1)) as usize
    }
}

/// 参考实现 `bits2pulses`：在缓存里二分，"这么多位最多能编几个脉冲"。
///
/// `lm` 是**当前**的 LM；参考实现内部先 `LM++` 再查表（多出来的那一行给"再劈一次"
/// 的递归用）。返回的是脉冲数**编码** q，不是脉冲数本身（要过 [`get_pulses`]）。
pub fn bits2pulses(band: usize, lm: i32, bits: i32) -> i32 {
    let row = (lm + 1) as usize;
    let base = CACHE_INDEX[row * NB_EBANDS + band] as usize;
    let cache = &CACHE_BITS[base..];
    let mut lo = 0usize;
    let mut hi = cache[0] as usize;
    let bits = bits - 1;
    for _ in 0..LOG_MAX_PSEUDO {
        let mid = (lo + hi + 1) >> 1;
        if cache[mid] as i32 >= bits {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let lo_bits = if lo == 0 { -1 } else { cache[lo] as i32 };
    if bits - lo_bits <= cache[hi] as i32 - bits {
        lo as i32
    } else {
        hi as i32
    }
}

/// 参考实现 `pulses2bits`：K=q 时实际要花多少位（缓存里存的是"减一"，这里加回来）。
pub fn pulses2bits(band: usize, lm: i32, pulses: i32) -> i32 {
    if pulses == 0 {
        return 0;
    }
    let row = (lm + 1) as usize;
    let base = CACHE_INDEX[row * NB_EBANDS + band] as usize;
    CACHE_BITS[base + pulses as usize] as i32 + 1
}

// ── θ 的位精确整数工具（θ 会影响比特分配，不能换成浮点近似）──

/// `EC_ILOG`：位宽（0 时给 0）。
fn ilog32(x: u32) -> u32 {
    32 - x.leading_zeros()
}

/// 参考实现 `FRAC_MUL16`：Q15 定点乘法。
///
/// **注意它在浮点构建下仍然是整数运算**（`(16384 + (int16)a·(int16)b) >> 15`），
/// 而且参数会被**截成 16 位**。这不是笔误：`bitexact_cos` / `bitexact_log2tan` 的
/// 位精确性就靠它，所以这里必须逐位照抄，不能"改成浮点乘法"。
fn frac_mul16(a: i32, b: i32) -> i32 {
    (16384 + (a as i16 as i32) * (b as i16 as i32)) >> 15
}

/// 参考实现 `bitexact_cos`：把 θ（0..=16384）变成 Q15 的 `cos(π/2·θ/16384)`。
///
/// 它是个多项式近似，**故意不追求数学精确** —— 只要平台间一致就行。别换成 `.cos()`。
///
/// **返回值的 16 位回绕要照抄**：参考实现的返回类型是 `opus_int16`，而 θ<64 时
/// `x2` 会被算成 32767、`1+x2` 溢出成 -32768（来源里那句 `celt_assert(x2<=32766)`
/// 就是在声明"这种输入不该出现"）。有效码流里 θ 只会是 `k·16384/qn`（qn ≤ 256），
/// 所以 θ ≥ 64 恒成立；这里保留回绕只是为了"万一走到也和参考一模一样"。
fn bitexact_cos(x: i32) -> i32 {
    let x2 = (4096 + x * x) >> 13;
    let x2 = (32767 - x2) + frac_mul16(x2, -7651 + frac_mul16(x2, 8277 + frac_mul16(-626, x2)));
    (1 + x2) as i16 as i32
}

/// 参考实现 `bitexact_log2tan`：`log2(isin/icos)` 的定点近似。
fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    let lc = ilog32(icos as u32);
    let ls = ilog32(isin as u32);
    let icos = icos << (15 - lc);
    let isin = isin << (15 - ls);
    (ls as i32 - lc as i32) * (1 << 11) + frac_mul16(isin, frac_mul16(isin, -2597) + 7932)
        - frac_mul16(icos, frac_mul16(icos, -2597) + 7932)
}

/// 参考实现 `isqrt32`：整数平方根（θ 的三角分布解码要用它反解）。
fn isqrt32(mut val: u32) -> u32 {
    if val == 0 {
        return 0;
    }
    let mut g = 0u32;
    let mut bshift = (ilog32(val) as i32 - 1) >> 1;
    let mut b = 1u32 << bshift;
    loop {
        let t = ((g << 1) + b) << bshift;
        if t <= val {
            g += b;
            val -= t;
        }
        b >>= 1;
        bshift -= 1;
        if bshift < 0 {
            break;
        }
    }
    g
}

/// 参考实现 `compute_qn`：θ 要量化成多少个档。
///
/// 上界 `b-pulse_cap-(4<<BITRES)` 保证"θ 取到最大也还剩至少一个脉冲给 side"——
/// 否则 side 一个脉冲都没有、又不会被折叠，就会整块塌掉。
fn compute_qn(n: usize, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    let mut n2 = 2 * n as i32 - 1;
    if stereo && n == 2 {
        n2 -= 1;
    }
    let mut qb = (b - pulse_cap - (4 << BITRES)).min((b + n2 * offset) / n2);
    qb = qb.min(8 << BITRES);
    if qb < (1 << BITRES >> 1) {
        1
    } else {
        let qn = EXP2_TABLE8[(qb & 0x7) as usize] >> (14 - (qb >> BITRES));
        ((qn + 1) >> 1) << 1
    }
}

// ── 时频重排 ──

/// 参考实现 `haar1`：一步 Haar 小波（时频分辨率的一对偶变换）。
///
/// 解码侧只用它处理 **lowband**：折叠过去的谱必须和当前带的时间分辨对上，
/// 所以要跟着编码侧一起"把 lowband 变回去"。
fn haar1(x: &mut [f32], n0: usize, stride: usize) {
    let n0 = n0 >> 1;
    // 参考实现的 `QCONST16(.70710678f,15)`。clippy 会建议换成
    // `f32::consts::FRAC_1_SQRT_2`（位模式下是同一个值），但这里的规矩是
    // **字面量照抄参考实现**，所以显式放行这条 lint。
    #[allow(clippy::approx_constant)]
    const K: f32 = 0.707_106_78;
    for i in 0..stride {
        for j in 0..n0 {
            let a = K * x[stride * 2 * j + i];
            let b = K * x[stride * (2 * j + 1) + i];
            x[stride * 2 * j + i] = a + b;
            x[stride * (2 * j + 1) + i] = a - b;
        }
    }
}

/// 参考实现 `deinterleave_hadamard`：把频序的样点重排成时序（短块交错）。
fn deinterleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    debug_assert!(stride > 0);
    let n = n0 * stride;
    let mut tmp = vec![0f32; n];
    if hadamard {
        let ordery = &ORDERY_TABLE[stride - 2..];
        for i in 0..stride {
            for j in 0..n0 {
                tmp[ordery[i] * n0 + j] = x[j * stride + i];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[i * n0 + j] = x[j * stride + i];
            }
        }
    }
    x[..n].copy_from_slice(&tmp);
}

/// 参考实现 `interleave_hadamard`：上面那步的逆操作（解完形状之后撤掉）。
fn interleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    debug_assert!(stride > 0);
    let n = n0 * stride;
    let mut tmp = vec![0f32; n];
    if hadamard {
        let ordery = &ORDERY_TABLE[stride - 2..];
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[ordery[i] * n0 + j];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[i * n0 + j];
            }
        }
    }
    x[..n].copy_from_slice(&tmp);
}

/// 立体声合并（参考实现 `stereo_merge`）：把解出来的 mid/side 变回左右声道。
///
/// `|X±Y|² = |X|² + |Y|² ± 2·Σxy`，所以分别取两个范数的倒数当增益就能让两声道
/// 都回到单位能量。若某一侧能量小到 6e-4 以下（几乎纯 mid 或纯 side），参考实现
/// 直接让 `Y=X` —— 这时候再按范数算增益会把数值放大到离谱。
fn stereo_merge(x: &mut [f32], y: &mut [f32], mid: f32, n: usize) {
    let mut xp = 0f32;
    let mut side = 0f32;
    for j in 0..n {
        xp += x[j] * y[j];
        side += y[j] * y[j];
    }
    let xp = mid * xp;
    // 参考实现写的是 `mid2 = SHR32(mid, 1)`（把 Q15 的 mid 降到 Q14 去和 X/Y 对齐），
    // 但**浮点构建下 `SHR32` 是空操作** —— 所以这里就是 mid 本身，千万别除以 2：
    // 除了会在 X 上多出一个 2 倍增益，还会把 `El/Er` 算成 0.25 那段、
    // 正好绕过下面那个"能量太小就塌陷"的保护。
    let mid2 = mid;
    let el = mid2 * mid2 + side - 2.0 * xp;
    let er = mid2 * mid2 + side + 2.0 * xp;
    if er < 6e-4 || el < 6e-4 {
        y[..n].copy_from_slice(&x[..n]);
        return;
    }
    let lgain = celt_rsqrt(el);
    let rgain = celt_rsqrt(er);
    for j in 0..n {
        let l = mid * x[j];
        let r = y[j];
        x[j] = lgain * (l - r);
        y[j] = rgain * (l + r);
    }
}

// ── 能量：细能量、帧尾零头、归一化还原 ──

/// 细能量解码：把每个带的 `fine_quant[i]` 位读出来，补到频带能量上。
///
/// 必须在比特分配之后调用（`fine_quant` 来自分配结果），且**顺序不能变**：
/// 它直接决定后面所有帧的能量状态。
pub fn unquant_fine_energy(
    dec: &mut RangeDecoder<'_>,
    alloc: &Allocation,
    channels: usize,
    start: usize,
    end: usize,
    state: &mut CeltState,
) -> Result<()> {
    for i in start..end {
        let q = alloc.fine_quant[i];
        if q <= 0 {
            continue;
        }
        for ch in 0..channels {
            let q2 = dec.dec_bits(q as u32)?;
            // 浮点构建下的公式（参考实现非 FIXED_POINT 分支）
            let offset =
                (q2 as f32 + 0.5) * ((1u32 << (14 - q as u32)) as f32) * (1.0 / 16384.0) - 0.5;
            state.old_ebands[ch][i] += offset;
        }
    }
    Ok(())
}

/// **帧尾的细能量补充**（参考实现 `unquant_energy_finalise`）。
///
/// 比特分配只能给每个带定一个"整数位"的细能量，剩下的零头（`bits_left`）在这里
/// 一位一符号地用掉：按 `fine_priority` 分两轮（优先的先拿），每带每声道一位，
/// 解出来的 0/1 变成 ±半个细能量档。
///
/// `bits_left` 是 `len*8 - ec_tell(dec)` —— **必须在形状解码之后算**（形状把大头花掉了）。
pub fn unquant_energy_finalise(
    dec: &mut RangeDecoder<'_>,
    alloc: &Allocation,
    channels: usize,
    start: usize,
    end: usize,
    bits_left: i32,
    state: &mut CeltState,
) -> Result<()> {
    let mut bits_left = bits_left;
    for prio in 0..2i32 {
        for i in start..end {
            if bits_left < channels as i32 {
                break;
            }
            if alloc.fine_quant[i] >= MAX_FINE_BITS || alloc.fine_priority[i] != prio {
                continue;
            }
            for ch in 0..channels {
                let q2 = dec.dec_bits(1)? as f32;
                let offset = (q2 - 0.5)
                    * ((1u32 << (14 - alloc.fine_quant[i] as u32 - 1)) as f32)
                    * (1.0 / 16384.0);
                state.old_ebands[ch][i] += offset;
                bits_left -= 1;
            }
        }
    }
    Ok(())
}

// ── 能量：对数 → 线性（`log2Amp`）──

/// 频带能量的"均值中心"（参考实现 `quant_bands.c` 的 `eMeans`，浮点构建那一份）。
///
/// 定点构建里是 `103, 100, 92, ...` 这样的整数（`SHL16(x,6)` 后当 Q15 用），
/// 浮点构建里就是它们的 1/16：`103/16 = 6.4375`。每个数都是 1/16 的整数倍，
/// f32 里精确可表示 —— 单测里拿整数表交叉核对过。
const E_MEANS: [f32; NB_EBANDS] = [
    6.4375, 6.25, 5.75, 5.3125, 5.0625, //
    4.8125, 4.5, 4.375, 4.875, 4.6875, //
    4.5625, 4.4375, 4.875, 4.625, 4.3125, //
    4.5, 4.375, 4.625, 4.75, 4.4375, //
    3.75,
];

/// 参考实现 `log2Amp`：把对数能量换成线性幅度（[`denormalise_bands`] 的输入）。
///
/// 浮点构建下 `SHL16`/`PSHR32` 都是空操作，所以公式就是 `2^(old + eMeans[i])`；
/// `[start, end)` 之外一律清 0 —— bandE 会被缩带（带宽切换）的帧用到，旧值不能留。
pub fn log2_amp(
    band_e: &mut [[f32; NB_EBANDS]; 2],
    old_ebands: &[[f32; NB_EBANDS]; 2],
    start: usize,
    end: usize,
    channels: usize,
) {
    for (ch, e) in band_e.iter_mut().enumerate().take(channels) {
        e[..start].fill(0.0);
        for i in start..end {
            e[i] = celt_exp2(old_ebands[ch][i] + E_MEANS[i]);
        }
        e[end..].fill(0.0);
    }
}

/// 归一化还原：把形状 `X` 按频带能量放大成频谱。
///
/// `x[ch]` 是归一化形状（长度 `m * SHORT_MDCT`），`band_e[ch][band]` 是 `log2Amp`
/// 之后的线性能量。返回每声道的频谱（与 `x` 同长，频带外为 0）。
pub fn denormalise_bands(
    x: &[Vec<f32>],
    band_e: &[[f32; NB_EBANDS]],
    end: usize,
    channels: usize,
    m: usize,
) -> Vec<Vec<f32>> {
    let n = m * SHORT_MDCT as usize;
    let mut out = Vec::with_capacity(channels);
    for ch in 0..channels {
        let mut freq = vec![0f32; n];
        let src = &x[ch];
        for band in 0..end {
            let g = band_e[ch][band];
            let from = m * E_BANDS[band] as usize;
            let to = m * E_BANDS[band + 1] as usize;
            for j in from..to {
                freq[j] = src[j] * g;
            }
        }
        // 频带之外（含被跳过的带）一律 0
        out.push(freq);
    }
    out
}

// ── 反塌陷 ──

/// 反塌陷（参考实现 `anti_collapse`）：给被量化成 0 的短块**填按能量估计的噪声**。
///
/// 瞬态帧里每个短块单独走 PVQ，某个块可能一个脉冲都没分到（"塌陷"）。空着不填的话
/// 相邻块之间会突然断一下，听感是"咔"。填的幅度由**当前能量与近两帧能量的差**估计：
/// 差得越多说明这是个瞬态，噪声可以大一点；`thresh` 是上限（分到的脉冲越少、
/// `depth` 越小、上限越大）。
///
/// `collapse_masks[i*C+c]` 的第 k 位表示"第 k 个短块还有非零系数"，为 0 才需要填。
/// 种子（LCG）在参考实现里是**传值**的，改动不外传 —— 这里保持一致。
#[allow(clippy::too_many_arguments)]
pub fn anti_collapse(
    x: &mut [Vec<f32>],
    collapse_masks: &[u8],
    lm: u32,
    channels: usize,
    start: usize,
    end: usize,
    log_e: &[[f32; NB_EBANDS]],
    prev1: &[[f32; NB_EBANDS]],
    prev2: &[[f32; NB_EBANDS]],
    pulses: &[i32; NB_EBANDS],
    seed: u32,
) {
    let mut seed = seed;
    for i in start..end {
        let n0 = (E_BANDS[i + 1] - E_BANDS[i]) as usize;
        // depth 是"每样本多少 1/8 位"
        let depth = (1 + pulses[i]) / ((n0 << lm) as i32);
        let thresh = 0.5 * celt_exp2(-0.125 * depth as f32);
        let sqrt_1 = celt_rsqrt((n0 << lm) as f32);

        for c in 0..channels {
            let mut p1 = prev1[c][i];
            let mut p2 = prev2[c][i];
            if channels == 1 {
                // 单声道帧：两个声道的历史取较大值（编码时是同一个信号）
                p1 = p1.max(prev1[1][i]);
                p2 = p2.max(prev2[1][i]);
            }
            let ediff = (log_e[c][i] - p1.min(p2)).max(0.0);
            // 短块的短 MDCT 能量换算系数不同（LM==3 时乘 √2）
            let mut r = 2.0 * celt_exp2(-ediff);
            if lm == 3 {
                // 参考实现的 `1.41421356f`（√2 的近似），同样照抄字面量
                #[allow(clippy::approx_constant)]
                const SQRT2: f32 = 1.414_213_6;
                r *= SQRT2;
            }
            r = thresh.min(r);
            r *= sqrt_1;

            let base = (E_BANDS[i] as usize) << lm;
            let cm = collapse_masks[i * channels + c];
            let mut renormalize = false;
            for k in 0..(1usize << lm) {
                if cm & (1 << k) == 0 {
                    for j in 0..n0 {
                        seed = lcg_rand(seed);
                        let v = if seed & 0x8000 != 0 { r } else { -r };
                        x[c][base + (j << lm) + k] = v;
                    }
                    renormalize = true;
                }
            }
            if renormalize {
                renormalise_vector(&mut x[c][base..], n0 << lm, 1.0);
            }
        }
    }
}

// ── 形状解码本体 ──

/// 形状解码的输入（一帧的）。
pub struct ShapeInput<'a> {
    /// 每个带分给 PVQ 的位数（1/8 位）。
    pub pulses: &'a [i32; NB_EBANDS],
    /// 每个带的时频调整量。
    pub tf: &'a [i32; NB_EBANDS],
    pub start: usize,
    pub end: usize,
    pub lm: u32,
    /// 是否是瞬态帧（决定短块数 B）。
    pub transient: bool,
    pub spread: u32,
    pub dual_stereo: bool,
    pub intensity: i32,
    pub coded_bands: usize,
    /// `len*(8<<BITRES) - anti_collapse_rsv`（1/8 位）。
    pub total_bits: i32,
    /// 比特分配留下的余量（1/8 位）。
    pub balance: i32,
    /// 折叠/填噪的 LCG 种子（参考实现里是上一帧结束时的区间状态）。
    pub seed: u32,
}

/// 每个频带解完之后留下的记录（对拍与调试用）。
///
/// 参考实现的插桩就是按这几个量 dump 的：一旦形状与参考分叉，这一行能直接指出
/// "是从哪一带、哪个 `tell_frac` 开始的"，比只看最终的 X 好查太多。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandTrace {
    /// 带号。
    pub band: usize,
    /// 这一带实际用的 N（`M*(eBands[i+1]-eBands[i])`）。
    pub n: usize,
    /// 这一带拿到的位数（1/8 位）。
    pub b: i32,
    /// 折叠来源在 `norm` 里的偏移；-1 表示这一带不折叠（用 LCG）。
    pub lowband: isize,
    /// 折叠掩码（左/右声道）。
    pub x_cm: u32,
    pub y_cm: u32,
    /// 解完这一带之后的 `tell_frac`。
    pub tell: i32,
}

/// 形状解码的输出。
pub struct ShapeOutput {
    /// 每声道的归一化频谱（长度 `M*SHORT_MDCT`）。
    pub x: Vec<Vec<f32>>,
    /// 折叠掩码 `[band*C + ch]`（[`anti_collapse`] 要用）。
    pub collapse_masks: Vec<u8>,
    /// 每一带的解码记录（见 [`BandTrace`]）。
    pub trace: Vec<BandTrace>,
    /// 这一帧把种子推进到哪（参考实现把它丢掉，这里返回只是方便对拍）。
    pub seed: u32,
}

/// 形状解码器：递归时用 `&mut self` 往下传（只持有熵解码器与滚动种子）。
struct ShapeDec<'a, 'b> {
    dec: &'a mut RangeDecoder<'b>,
    seed: u32,
    trace: bool,
}

/// 对拍调试用：只跟踪第 n 次 `quant_all_bands` 调用的完整递归轨迹（-1 = 关闭）。
/// 与参考实现的 `OPUS_DBG_FRAME` 同一个用途：`QIN` 进 `QOUT` 出，格式逐字对齐。
pub static SHAPE_TRACE: std::sync::atomic::AtomicIsize = std::sync::atomic::AtomicIsize::new(-1);
/// 已经调用过多少次 `quant_all_bands`（测试在开跑之前清零）。
pub static SHAPE_TRACE_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// N==1 的特例：解一个符号位（位数不够就当正）。
fn n1_sign(dec: &mut RangeDecoder<'_>, b: &mut i32, remaining_bits: &mut i32) -> Result<f32> {
    let mut sign = 0u32;
    if *remaining_bits >= 1 << BITRES {
        sign = dec.dec_bits(1)?;
        *remaining_bits -= 1 << BITRES;
        *b -= 1 << BITRES;
    }
    // NORM_SCALING 在浮点构建下是 1.0
    Ok(if sign != 0 { -1.0 } else { 1.0 })
}

impl ShapeDec<'_, '_> {
    /// 参考实现 `quant_band`（解码路径）。
    ///
    /// 参数多是因为它要处理四件事：单/立体声、分裂/不分裂、递归层级、以及折叠用的
    /// `lowband` / `lowband_out`。**逐行对着参考实现读**是最省事的办法。
    ///
    /// 几处容易看错的地方：
    ///
    /// - `blk`（参考实现的 `B`）在"劈成两半"之后会变成 `(B+1)>>1`；
    /// - `lm` 在每次"劈成两半"时减一，**可以减到 -1**（`CACHE_INDEX` 因此多出一行）；
    /// - `n0` 是**没被劈之前**的 N，函数末尾写 `lowband_out` 用的是它；
    /// - `fill` 是折叠掩码的"种子"，一路上被掩、被移位。
    #[allow(clippy::too_many_arguments)]
    fn quant_band(
        &mut self,
        i: usize,
        x: &mut [f32],
        mut y: Option<&mut [f32]>,
        n_in: usize,
        b_in: i32,
        spread: u32,
        blk_in: usize,
        intensity: i32,
        tf_change_in: i32,
        lowband_in: Option<&[f32]>,
        remaining_bits: &mut i32,
        lm_in: i32,
        mut lowband_out: Option<&mut [f32]>,
        level: i32,
        gain: f32,
        fill_in: u32,
    ) -> Result<u32> {
        if self.trace {
            eprintln!(
                "QIN {} {} {} {} {} {} {} {} {}",
                level, i, n_in, b_in, blk_in, lm_in,
                self.dec.tell_frac(), self.dec.state().0, self.dec.state().1
            );
        }
        let stereo = y.is_some();
        let n0 = n_in;
        let mut n = n_in;
        let mut b = b_in;
        let mut blk = blk_in;
        let mut lm = lm_in;
        let mut tf_change = tf_change_in;
        let mut fill = fill_in;
        let mut n_b = n_in / blk_in;
        let b0_init = blk_in;
        // 长块：只有"本来就只有一个块"的帧才走哈达玛序
        let long_blocks = b0_init == 1;
        let mut recombine = 0usize;
        let mut time_divide = 0usize;
        let mut cm: u32 = 0;
        // 分裂时算出来、收尾还要用
        let mut mid = 0f32;
        let mut inv = false;

        // ── 特例：N==1。一个系数只有符号可编，位数不够就不写 ──
        if n == 1 {
            x[0] = n1_sign(self.dec, &mut b, remaining_bits)?;
            if let Some(yv) = y.as_deref_mut() {
                yv[0] = n1_sign(self.dec, &mut b, remaining_bits)?;
            }
            if let Some(out) = lowband_out.as_deref_mut() {
                // SHR16(X[0],4) 在浮点构建下是空操作
                out[0] = x[0];
            }
            if self.trace {
                eprintln!("QOUT {} {} {} {}", level, i, 1u32, self.dec.tell_frac());
            }
            return Ok(1);
        }

        // lowband 会被就地改写（haar1 / 时间重排），所以**要改就一定先拷一份**。
        // 参考实现用一块公用的 `lowband_scratch`；这里用局部 Vec，语义相同：
        // 下面"需要拷贝"的判断与"需要改写"的条件完全重合，所以只要 `copied` 为真，
        // 改写就发生在副本上，绝不会动到调用方（以及 `norm` 那块折叠源）。
        let mut low_owned: Vec<f32> = Vec::new();
        let mut copied = false;

        // ── 时频分辨率重组（只在单声道的最外层做）──
        if !stereo && level == 0 {
            if tf_change > 0 {
                recombine = tf_change as usize;
            }
            copied = lowband_in.is_some()
                && (recombine != 0 || (n_b % 2 == 0 && tf_change < 0) || b0_init > 1);
            if copied {
                low_owned = lowband_in.unwrap()[..n].to_vec();
            }
            for k in 0..recombine {
                if copied {
                    haar1(&mut low_owned[..n], n >> k, 1 << k);
                }
                fill = BIT_INTERLEAVE_TABLE[(fill & 0xF) as usize] as u32
                    | (BIT_INTERLEAVE_TABLE[(fill >> 4) as usize] as u32) << 2;
            }
            blk >>= recombine;
            n_b <<= recombine;

            // tf_change 为负时反过来：把时间分辨率拆细、频带并粗
            while n_b % 2 == 0 && tf_change < 0 {
                if copied {
                    haar1(&mut low_owned[..n], n_b, blk);
                }
                fill |= fill << blk;
                blk <<= 1;
                n_b >>= 1;
                time_divide += 1;
                tf_change += 1;
            }
        }
        // 重组之后的 B0 / N_B0（非最外层或立体声时就是原值）
        let b0 = blk;
        let n_b0 = n_b;
        if !stereo && level == 0 && b0 > 1 {
            let stride = b0 << recombine;
            let half = n_b >> recombine;
            if copied {
                deinterleave_hadamard(&mut low_owned[..n], half, stride, long_blocks);
            }
        }
        // 到这里 lowband 再也不会被改写，直接借用（副本或调用方的切片）
        let low_src: Option<&[f32]> = if copied { Some(&low_owned[..n]) } else { lowband_in };

        // ── 要不要"劈成两半"：位数超过缓存能表达的上限就劈 ──
        let cache_base = CACHE_INDEX[((lm + 1) as usize) * NB_EBANDS + i] as usize;
        let cache_max_bits = CACHE_BITS[cache_base + CACHE_BITS[cache_base] as usize] as i32;
        let mut split = stereo;
        // 单声道时"劈成两半"是**在 x 内部**劈（前一半当 mid、后一半当 side），
        // 而立体声分裂用的是 x/y 两个缓冲区 —— 下面统一成 (xa, xb) 两种来源。
        let mut mono_split = false;
        if !stereo && lm != -1 && b > cache_max_bits + 12 && n > 2 {
            n >>= 1;
            split = true;
            mono_split = true;
            lm -= 1;
            if blk == 1 {
                fill = (fill & 1) | (fill << 1);
            }
            blk = (blk + 1) >> 1;
        }
        // 劈完之后"另一半"的 lowband 偏移就是当前的 N
        let next_lowband_off = n;

        if split {
            // 分裂之后的两半：立体声是 (X, Y)，单声道是 x 的前后半段。
            // 注意 `xa/xb` 借用了 `x`（单声道）或 `x`/`y`（立体声），所以它们
            // 只活在这个块里 —— 块外的收尾逻辑重新借 `x`/`y`。
            let (xa, xb): (&mut [f32], &mut [f32]) = if mono_split {
                debug_assert_eq!(x.len(), 2 * n, "单声道分裂时 x 应当正好是两半");
                let (l, r) = x.split_at_mut(n);
                (l, r)
            } else {
                (x, y.as_deref_mut().expect("立体声分裂必须有 Y"))
            };

            let pulse_cap = LOG_N[i] as i32 + lm * (1 << BITRES);
            let offset = (pulse_cap >> 1)
                - if stereo && n == 2 { QTHETA_OFFSET_TWOPHASE } else { QTHETA_OFFSET };
            let mut qn = compute_qn(n, b, offset, pulse_cap, stereo);
            // 强度立体声以上的带：形状两声道共用，θ 就没意义了（qn 强制为 1）
            if stereo && (i as i32) >= intensity {
                qn = 1;
            }

            let tell = self.dec.tell_frac() as i32;
            let mut itheta = 0i32;
            if qn != 1 {
                if stereo && n > 2 {
                    // 立体声的 θ 用"台阶分布"：x0 以下每档概率 3，往上一律 1
                    let p0 = 3i32;
                    let x0 = qn / 2;
                    let ft = p0 * (x0 + 1) + x0;
                    let fs = self.dec.decode(ft as u32) as i32;
                    let xx = if fs < (x0 + 1) * p0 {
                        fs / p0
                    } else {
                        x0 + 1 + (fs - (x0 + 1) * p0)
                    };
                    let (fl, fh) = if xx <= x0 {
                        (p0 * xx, p0 * (xx + 1))
                    } else {
                        ((xx - 1 - x0) + (x0 + 1) * p0, (xx - x0) + (x0 + 1) * p0)
                    };
                    self.dec.update(fl as u32, fh as u32, ft as u32);
                    itheta = xx;
                } else if b0 > 1 || stereo {
                    // 有短块（或立体声）：均匀分布
                    itheta = self.dec.dec_uint((qn + 1) as u32)? as i32;
                } else {
                    // 其余情况（单声道、长块）：三角分布
                    let ft = ((qn >> 1) + 1) * ((qn >> 1) + 1);
                    let fm = self.dec.decode(ft as u32) as i32;
                    let (ith, fs, fl);
                    if fm < ((qn >> 1) * ((qn >> 1) + 1) >> 1) {
                        ith = ((isqrt32((8 * fm + 1) as u32) - 1) >> 1) as i32;
                        fs = ith + 1;
                        fl = ith * (ith + 1) >> 1;
                    } else {
                        // 注意这个 `+1`：它是"开方再取整"的**进位修正**，
                        // 少写它会让 isqrt 的边界整体偏一格（θ 差一档 → 位数分配跟着偏）
                        ith = (2 * (qn + 1) - isqrt32((8 * (ft - fm - 1) + 1) as u32) as i32) >> 1;
                        fs = qn + 1 - ith;
                        fl = ft - ((qn + 1 - ith) * (qn + 2 - ith) >> 1);
                    }
                    self.dec.update(fl as u32, (fl + fs) as u32, ft as u32);
                    itheta = ith;
                }
                itheta = itheta * 16384 / qn;
            } else if stereo {
                // θ 只有一档：编码器改成传一个"反相"标志
                inv = if b > 2 << BITRES && *remaining_bits > 2 << BITRES {
                    self.dec.dec_bit_logp(2) != 0
                } else {
                    false
                };
                itheta = 0;
            }
            let qalloc = self.dec.tell_frac() as i32 - tell;
            b -= qalloc;
            if self.trace {
                eprintln!(
                    "QD {} {} {} {} {} {} {} {} {}",
                    level, i, stereo as u8, qn, itheta, qalloc, b, n, lm
                );
            }

            let orig_fill = fill;
            let iside;
            let mut delta;
            if itheta == 0 {
                // 全给 mid：低 B 位的折叠掩码清掉（side 不占位、也不折叠）
                fill &= (1u32 << blk) - 1;
                iside = 0;
                delta = -16384;
            } else if itheta == 16384 {
                // 全给 side
                fill &= ((1u32 << blk) - 1) << blk;
                iside = 32767;
                delta = 16384;
            } else {
                let imid = bitexact_cos(itheta);
                iside = bitexact_cos(16384 - itheta);
                delta = frac_mul16(((n - 1) << 7) as i32, bitexact_log2tan(iside, imid));
            }
            let imid = if itheta == 0 {
                32767
            } else if itheta == 16384 {
                0
            } else {
                bitexact_cos(itheta)
            };
            // mid/side 的增益（Q15 → 浮点）。参考实现在浮点构建下就是 /32768
            mid = (1.0 / 32768.0) * imid as f32;
            let side = (1.0 / 32768.0) * iside as f32;

            if n == 2 && stereo {
                // N==2 的立体声是特例：side 只用一个符号位（mid/side 在这里正交）
                let mut sbits = 0i32;
                if itheta != 0 && itheta != 16384 {
                    sbits = 1 << BITRES;
                }
                let mbits = b - sbits;
                let swap = itheta > 8192;
                *remaining_bits -= qalloc + sbits;
                let mut sign = 0u32;
                if sbits != 0 {
                    sign = self.dec.dec_bits(1)?;
                }
                let sign = 1 - 2 * sign as i32;

                if swap {
                    // x2 = Y（先把 Y 解开，再由它反解 X）
                    cm = self.quant_band(
                        i,
                        xb,
                        None,
                        n,
                        mbits,
                        spread,
                        blk,
                        intensity,
                        tf_change,
                        low_src,
                        remaining_bits,
                        lm,
                        // 注意：参考实现这里传的是 `lowband_out`，**不是 NULL**。
                        // N==2 的立体声带本来就窄，正是靠这次回写把 `norm` 填上，
                        // 后面的带才能折叠到它 —— 传 NULL 的话 fold 源恒为 0，
                        // 后面每一条带都退化成"纯噪声"（听起来像白噪，很难查）。
                        lowband_out.as_deref_mut(),
                        level,
                        gain,
                        orig_fill,
                    )?;
                    
                    xa[0] = -sign as f32 * xb[1];
                    xa[1] = sign as f32 * xb[0];
                } else {
                    // x2 = X（先把 X 解开，再由它反解 Y）
                    cm = self.quant_band(
                        i,
                        xa,
                        None,
                        n,
                        mbits,
                        spread,
                        blk,
                        intensity,
                        tf_change,
                        low_src,
                        remaining_bits,
                        lm,
                        // 注意：参考实现这里传的是 `lowband_out`，**不是 NULL**。
                        // N==2 的立体声带本来就窄，正是靠这次回写把 `norm` 填上，
                        // 后面的带才能折叠到它 —— 传 NULL 的话 fold 源恒为 0，
                        // 后面每一条带都退化成"纯噪声"（听起来像白噪，很难查）。
                        lowband_out.as_deref_mut(),
                        level,
                        gain,
                        orig_fill,
                    )?;
                    
                    xb[0] = -sign as f32 * xa[1];
                    xb[1] = sign as f32 * xa[0];
                }
                // mid/side 混回左右
                
                xa[0] = mid * xa[0];
                xa[1] = mid * xa[1];
                xb[0] = side * xb[0];
                xb[1] = side * xb[1];
                let t0 = xa[0];
                xa[0] = t0 - xb[0];
                xb[0] = t0 + xb[0];
                let t1 = xa[1];
                xa[1] = t1 - xb[1];
                xb[1] = t1 + xb[1];
            } else {
                // ── 常规分裂 ──
                if b0 > 1 && !stereo && (itheta & 0x3fff) != 0 {
                    if itheta > 8192 {
                        // 预回声掩蔽的粗略近似
                        delta -= delta >> (4 - lm);
                    } else {
                        // 后掩蔽：每 10 ms 约 1.5 dB
                        delta = 0.min(delta + (n << BITRES >> (5 - lm)) as i32);
                    }
                }
                let mut mbits = 0.max(b.min((b - delta) / 2));
                let mut sbits = b - mbits;
                if self.trace {
                    eprintln!("QS {} {} {} {} {} {}", level, i, mbits, sbits, delta, itheta);
                }
                *remaining_bits -= qalloc;

                let next_level = if stereo { 0 } else { level + 1 };
                let mut rebalance = *remaining_bits;
                let blk_shift = ((b0 >> 1) as u32) & if stereo { 0 } else { u32::MAX };
                // side 只在单声道分裂时才接着折叠
                let need_side_lb = !stereo;

                if mbits >= sbits {
                    let mid_gain = if stereo { 1.0 } else { gain * mid };
                    cm = self.quant_band(
                        i,
                        xa,
                        None,
                        n,
                        mbits,
                        spread,
                        blk,
                        intensity,
                        tf_change,
                        low_src,
                        remaining_bits,
                        lm,
                        if stereo { lowband_out.as_deref_mut() } else { None },
                        next_level,
                        mid_gain,
                        fill,
                    )?;
                    rebalance = mbits - (rebalance - *remaining_bits);
                    if rebalance > 3 << BITRES && itheta != 0 {
                        sbits += rebalance - (3 << BITRES);
                    }
                    let side_lb = if need_side_lb {
                        low_src.map(|lb| &lb[next_lowband_off..])
                    } else {
                        None
                    };
                    let cm2 = self.quant_band(
                        i,
                        xb,
                        None,
                        n,
                        sbits,
                        spread,
                        blk,
                        intensity,
                        tf_change,
                        side_lb,
                        remaining_bits,
                        lm,
                        None,
                        next_level,
                        gain * side,
                        fill >> blk,
                    )?;
                    cm |= cm2 << blk_shift;
                } else {
                    let side_lb = if need_side_lb {
                        low_src.map(|lb| &lb[next_lowband_off..])
                    } else {
                        None
                    };
                    cm = self.quant_band(
                        i,
                        xb,
                        None,
                        n,
                        sbits,
                        spread,
                        blk,
                        intensity,
                        tf_change,
                        side_lb,
                        remaining_bits,
                        lm,
                        None,
                        next_level,
                        gain * side,
                        fill >> blk,
                    )? << blk_shift;
                    rebalance = sbits - (rebalance - *remaining_bits);
                    if rebalance > 3 << BITRES && itheta != 16384 {
                        mbits += rebalance - (3 << BITRES);
                    }
                    let mid_gain = if stereo { 1.0 } else { gain * mid };
                    let cm2 = self.quant_band(
                        i,
                        xa,
                        None,
                        n,
                        mbits,
                        spread,
                        blk,
                        intensity,
                        tf_change,
                        low_src,
                        remaining_bits,
                        lm,
                        if stereo { lowband_out.as_deref_mut() } else { None },
                        next_level,
                        mid_gain,
                        fill,
                    )?;
                    cm |= cm2;
                }
            }
        } else {
            // ── 不分裂：直接 PVQ（或者没分到脉冲 → 清 0 / 填噪 / 折叠）──
            let mut q = bits2pulses(i, lm, b);
            let mut curr_bits = pulses2bits(i, lm, q);
            *remaining_bits -= curr_bits;
            // 万一超支就往下退脉冲数（参考实现保证最终不为负）
            while *remaining_bits < 0 && q > 0 {
                *remaining_bits += curr_bits;
                q -= 1;
                curr_bits = pulses2bits(i, lm, q);
                *remaining_bits -= curr_bits;
            }
            if q != 0 {
                let k = get_pulses(q);
                cm = alg_unquant(x, n, k, spread, blk, self.dec, gain)?;
            } else {
                let cm_mask = (1u32 << blk) - 1;
                fill &= cm_mask;
                if fill == 0 {
                    // 连折叠都不做：整条带清零
                    for v in x[..n].iter_mut() {
                        *v = 0.0;
                    }
                } else {
                    if low_src.is_none() {
                        // 没有低带可折：纯噪声（幅度是整数，参考实现就是 `(int32)seed>>20`）
                        for v in x[..n].iter_mut() {
                            self.seed = lcg_rand(self.seed);
                            *v = ((self.seed as i32) >> 20) as f32;
                        }
                        cm = cm_mask;
                    } else {
                        // 折叠：搬低带的谱，叠一个约 −48 dB 的微噪声（避免完全重复）
                        let lb = low_src.unwrap();
                        for j in 0..n {
                            self.seed = lcg_rand(self.seed);
                            let tmp = if self.seed & 0x8000 != 0 {
                                1.0 / 256.0
                            } else {
                                -1.0 / 256.0
                            };
                            x[j] = lb[j] + tmp;
                        }
                        cm = fill;
                    }
                    renormalise_vector(x, n, gain);
                }
            }
        }

        // ── 收尾：立体声合并；单声道最外层则是把时间重排撤回、并写 lowband_out ──
        if stereo {
            let yv = y.as_deref_mut().expect("立体声");
            if n != 2 {
                stereo_merge(x, yv, mid, n);
            }
            if inv {
                for v in yv[..n].iter_mut() {
                    *v = -*v;
                }
            }
        } else if level == 0 {
            let mut n_b = n_b0;
            let mut bb = b0;
            if b0 > 1 {
                let stride = b0 << recombine;
                let half = n_b >> recombine;
                interleave_hadamard(x, half, stride, long_blocks);
            }
            for _ in 0..time_divide {
                bb >>= 1;
                n_b <<= 1;
                cm |= cm >> bb;
                haar1(x, n_b, bb);
            }
            for k in 0..recombine {
                // cm 在这里必然 < 16：recombine>0 时 B0 = B>>recombine 已经缩小过一半以上
                debug_assert!((cm as usize) < BIT_DEINTERLEAVE_TABLE.len());
                cm = BIT_DEINTERLEAVE_TABLE[(cm as usize) & 0xF] as u32;
                haar1(x, n0 >> k, 1usize << k);
            }
            bb <<= recombine;
            if let Some(out) = lowband_out.as_deref_mut() {
                // 浮点构建下 `SHL32(EXTEND32(N0),22)` 是空操作，所以 n = √N0
                let g = celt_sqrt(n0 as f32);
                for (j, v) in x[..n0].iter().enumerate() {
                    out[j] = g * v;
                }
            }
            cm &= (1u32 << bb) - 1;
        }
        if self.trace {
            eprintln!("QOUT {} {} {} {}", level, i, cm, self.dec.tell_frac());
        }
        Ok(cm)
    }
}

/// 解出整帧所有频带的形状（参考实现 `quant_all_bands`，解码路径）。
///
/// 它是"带级"的调度器，负责三件事：
///
/// 1. **给每个带算它有多少位可用**：`pulses[i]`（分配结果）+ 前面带没花完的余量，
///    再夹到 `remaining_bits+1` 以内（绝不能透支到帧外）。
/// 2. **维护折叠来源**（`lowband_offset` / `effective_lowband`）：只要"本带之前的
///    一定范围还有 1 bit/sample 的深度"，就允许从这里折叠。
/// 3. **维护一块 `norm` 缓冲**：形状解完之后会把"归一化后的低带"写回去，供后面的带
///    折叠使用（注意它不是 `X` 本身 —— `X` 是最后的输出，不能拿来当折叠源）。
///
/// 立体声有两种模式：`dual_stereo` 时两声道**各自独立**解形状（相当于两个单声道）；
/// 否则两声道在 `quant_band` 里做 mid/side 联合编码。
pub fn quant_all_bands(
    dec: &mut RangeDecoder<'_>,
    inp: &ShapeInput<'_>,
    channels: usize,
) -> Result<ShapeOutput> {
    debug_assert!(channels == 1 || channels == 2);
    debug_assert!(inp.end <= EFF_EBANDS, "本引擎不支持 effEBands 之外的频带");

    let m = 1usize << inp.lm;
    let size = m * SHORT_MDCT as usize;
    let blk = if inp.transient { m } else { 1 };
    let c = channels;
    let start = inp.start;
    let end = inp.end;

    let n_call = SHAPE_TRACE_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let trace = SHAPE_TRACE.load(std::sync::atomic::Ordering::Relaxed) == n_call as isize;
    let mut sh = ShapeDec { dec, seed: inp.seed, trace };
    let mut x = vec![vec![0f32; size]; c];
    // `norm[c]` 存"供折叠用的低带"（参考实现里是 `_norm` 那块 scratch）
    let norm_len = m * E_BANDS[NB_EBANDS] as usize;
    let mut norm = vec![0f32; norm_len];
    let mut norm2 = vec![0f32; norm_len];
    // lowband 的副本：参考实现直接让 lowband 指着 norm（读在写之前），
    // 这里拷一小份出来，既避开别名、语义又完全一致（见 `quant_band` 的说明）
    let max_band = m * E_BANDS
        .windows(2)
        .map(|w| (w[1] - w[0]) as usize)
        .max()
        .unwrap();
    let mut lb_a = vec![0f32; max_band];
    let mut lb_b = vec![0f32; max_band];

    let mut collapse_masks = vec![0u8; NB_EBANDS * c];
    let mut trace = Vec::with_capacity(end - start);
    let mut balance = inp.balance;
    let mut lowband_offset = 0usize;
    let mut update_lowband = true;
    let mut dual_stereo = inp.dual_stereo;

    for i in start..end {
        let off = m * E_BANDS[i] as usize;
        let n_band = m * (E_BANDS[i + 1] - E_BANDS[i]) as usize;
        let tell = sh.dec.tell_frac() as i32;

        // ① 这一带能拿多少位
        if i != start {
            balance -= tell;
        }
        let mut remaining_bits = inp.total_bits - tell - 1;
        let b = if (i as i32) < inp.coded_bands as i32 {
            let div = 3.min(inp.coded_bands as i32 - i as i32);
            (remaining_bits + 1)
                .min(inp.pulses[i] + balance / div)
                .min(16383)
                .max(0)
        } else {
            0
        };

        // ② 折叠来源能不能往前推：本带**之前**至少还有一整条带的宽度可用
        let band_left = m * E_BANDS[i] as usize;
        if band_left >= n_band + m * E_BANDS[start] as usize
            && (update_lowband || lowband_offset == 0)
        {
            lowband_offset = i;
        }

        let tf_change = inp.tf[i];
        // 折叠是否允许：AGGRESSIVE 扩散下只有多短块或 tf_change<0 时才折
        let allow_fold = lowband_offset != 0
            && (inp.spread != SPREAD_AGGRESSIVE || blk > 1 || tf_change < 0);
        let mut eff_lowband: isize = -1;
        let (mut x_cm, mut y_cm);
        if allow_fold {
            // 注意用的是 `lowband_offset` 那一带的边界，**不是当前带的**：
            // "能从这里折叠"是由"最近一条够深的带"决定的。
            // 差可以是负的（低位那几带比当前带窄得多），所以要有符号地算，
            // 再夹到"至少从第 start 带起"。
            let fold_src = (m * E_BANDS[lowband_offset] as usize) as isize;
            let base = (m * E_BANDS[start] as usize) as isize;
            eff_lowband = base.max(fold_src - n_band as isize);
            // 折叠源要覆盖 [eff_lowband, eff_lowband+N)，把这些带的掩码并起来
            let mut fold_start = lowband_offset;
            loop {
                fold_start -= 1;
                if m * E_BANDS[fold_start] as usize <= eff_lowband as usize {
                    break;
                }
            }
            let mut fold_end = lowband_offset - 1;
            loop {
                fold_end += 1;
                if m * E_BANDS[fold_end] as usize >= eff_lowband as usize + n_band {
                    break;
                }
            }
            x_cm = 0;
            y_cm = 0;
            for fi in fold_start..fold_end {
                x_cm |= collapse_masks[fi * c] as u32;
                y_cm |= collapse_masks[fi * c + c - 1] as u32;
            }
        } else {
            // 走 LCG 折叠时几乎每个块都非零，掩码按"全都有"算
            x_cm = (1u32 << blk) - 1;
            y_cm = x_cm;
        }

        // ③ dual stereo 到强度立体声起点就结束（那一带开始两声道共用形状）
        if dual_stereo && i as i32 == inp.intensity {
            dual_stereo = false;
            for j in (m * E_BANDS[start] as usize)..band_left {
                norm[j] = 0.5 * (norm[j] + norm2[j]);
            }
        }

        if allow_fold {
            let e = eff_lowband as usize;
            lb_a[..n_band].copy_from_slice(&norm[e..e + n_band]);
            lb_b[..n_band].copy_from_slice(&norm2[e..e + n_band]);
        }

        let (xc0, xc1) = x.split_at_mut(1);
        let band_out0 = &mut norm[off..off + n_band];
        if dual_stereo {
            let lb1 = if allow_fold { Some(&lb_a[..n_band]) } else { None };
            x_cm = sh.quant_band(
                i,
                &mut xc0[0][off..off + n_band],
                None,
                n_band,
                b / 2,
                inp.spread,
                blk,
                inp.intensity,
                tf_change,
                lb1,
                &mut remaining_bits,
                inp.lm as i32,
                Some(band_out0),
                0,
                1.0,
                x_cm,
            )?;
            let lb2 = if allow_fold { Some(&lb_b[..n_band]) } else { None };
            y_cm = sh.quant_band(
                i,
                &mut xc1[0][off..off + n_band],
                None,
                n_band,
                b / 2,
                inp.spread,
                blk,
                inp.intensity,
                tf_change,
                lb2,
                &mut remaining_bits,
                inp.lm as i32,
                Some(&mut norm2[off..off + n_band]),
                0,
                1.0,
                y_cm,
            )?;
        } else {
            let lb1 = if allow_fold { Some(&lb_a[..n_band]) } else { None };
            let band_y = if c == 2 {
                Some(&mut xc1[0][off..off + n_band])
            } else {
                None
            };
            x_cm = sh.quant_band(
                i,
                &mut xc0[0][off..off + n_band],
                band_y,
                n_band,
                b,
                inp.spread,
                blk,
                inp.intensity,
                tf_change,
                lb1,
                &mut remaining_bits,
                inp.lm as i32,
                Some(band_out0),
                0,
                1.0,
                x_cm | y_cm,
            )?;
            y_cm = x_cm;
        }

        collapse_masks[i * c] = x_cm as u8;
        collapse_masks[i * c + c - 1] = y_cm as u8;
        balance += inp.pulses[i] + tell;
        trace.push(BandTrace {
            band: i,
            n: n_band,
            b,
            lowband: eff_lowband,
            x_cm,
            y_cm,
            tell: sh.dec.tell_frac() as i32,
        });
        // 只有"本带还有 1 bit/sample 的深度"时才允许把它当折叠源
        update_lowband = b > (n_band << BITRES) as i32;
    }

    Ok(ShapeOutput { x, collapse_masks, trace, seed: sh.seed })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::opus::rate;

    #[test]
    fn e_means_matches_the_fixed_point_table() {
        // 参考实现的定点表（quant_bands.c）除以 16 —— 两处独立写下的数字必须对得上
        const FIXED: [i32; NB_EBANDS] = [
            103, 100, 92, 85, 81, 77, 72, 70, 78, 75, 73, 71, 78, 74, 69, 72, 70, 74, 76, 71, 60,
        ];
        for (i, (&f, &q)) in E_MEANS.iter().zip(FIXED.iter()).enumerate() {
            assert_eq!(f, q as f32 / 16.0, "第 {i} 带");
        }
    }

    #[test]
    fn log2_amp_is_two_to_the_energy_plus_mean() {
        let mut band_e = [[123.0f32; NB_EBANDS]; 2];
        let mut old = [[0.0f32; NB_EBANDS]; 2];
        old[0][3] = 1.0; // 2^(1 + eMeans[3]) = 2^6.3125
        old[1][3] = -3.5;
        log2_amp(&mut band_e, &old, 2, 5, 2);
        assert!((band_e[0][3] - celt_exp2(1.0 + E_MEANS[3])).abs() < 1e-6);
        assert!((band_e[1][3] - celt_exp2(-3.5 + E_MEANS[3])).abs() < 1e-6);
        // 区间外必须清 0（老值不能留）
        assert_eq!(band_e[0][0], 0.0);
        assert_eq!(band_e[0][1], 0.0);
        assert_eq!(band_e[0][5], 0.0);
        assert_eq!(band_e[1][20], 0.0);
        // 区间内但没设过的带：2^(0 + eMeans)
        assert!((band_e[0][4] - celt_exp2(E_MEANS[4])).abs() < 1e-6);
    }

    #[test]
    fn denormalise_scales_each_band_by_its_energy() {
        let m = 8usize;
        let n = m * SHORT_MDCT as usize;
        let mut x = vec![vec![1.0f32; n]];
        x[0][0] = 0.5;
        let mut band_e = [[0f32; NB_EBANDS]; 2];
        band_e[0][0] = 4.0;
        band_e[0][1] = 2.0;
        let freq = denormalise_bands(&x, &band_e, 2, 1, m);
        // 第 0 带：[0, m*E_BANDS[1])，能量 4 → x*4
        assert_eq!(freq[0][0], 2.0);
        assert_eq!(freq[0][1], 4.0);
        // 第 1 带的起点
        let b1 = m * E_BANDS[1] as usize;
        assert_eq!(freq[0][b1], 2.0);
        // 第 2 带起（end=2）应当是 0
        let b2 = m * E_BANDS[2] as usize;
        assert_eq!(freq[0][b2], 0.0);
        assert_eq!(freq[0][n - 1], 0.0, "频带之外一律 0");
    }

    #[test]
    fn get_pulses_follows_the_reference_scale() {
        // q<8 时 K=q
        for q in 0..8 {
            assert_eq!(get_pulses(q), q as usize);
        }
        // q>=8：(8 + q&7) << ((q>>3)-1)
        assert_eq!(get_pulses(8), 8);
        assert_eq!(get_pulses(9), 9);
        assert_eq!(get_pulses(15), 15);
        assert_eq!(get_pulses(16), 16);
        assert_eq!(get_pulses(17), 18);
        assert_eq!(get_pulses(20), 24);
        assert_eq!(get_pulses(23), 30);
        // 单调不减 —— 否则 bits2pulses 的二分就不成立了
        let mut last = 0;
        for q in 0..40 {
            let k = get_pulses(q);
            assert!(k >= last, "q={q} 时 K 反而变小了");
            last = k;
        }
    }

    #[test]
    fn pulses_cache_is_sorted_which_the_binary_search_relies_on() {
        // CACHE_BITS 每段是 `[K_max, bits(1), bits(2), …]`：第 0 项是二分上界，
        // 第 1 项起才是"K=k 需要几位减一"，所以只要求**从第 1 项开始**非减。
        for band in 0..NB_EBANDS {
            for lm in 0..4i32 {
                let base = CACHE_INDEX[(lm + 1) as usize * NB_EBANDS + band] as usize;
                let k_max = CACHE_BITS[base] as usize;
                assert!(k_max >= 1, "band={band} lm={lm} 的段首不是 K_max");
                let seg = &CACHE_BITS[base + 1..base + k_max + 1];
                assert!(
                    seg.windows(2).all(|w| w[0] <= w[1]),
                    "band={band} lm={lm} 的位数表不是非减的"
                );
            }
        }
    }

    #[test]
    fn bits2pulses_round_trips_through_pulses2bits() {
        // `pulses2bits(bits2pulses(bits))` 必须还是同一个位数 —— 缓存里可能有好几档 K
        // 的位数相同，所以只要求"落到等价的 K 上"。
        // 注意 q 只能取到该段的 K_max（超出去就会读到下一段，参考实现也是这个约定）。
        for band in 0..NB_EBANDS {
            for lm in 0..4i32 {
                let base = CACHE_INDEX[(lm + 1) as usize * NB_EBANDS + band] as usize;
                let k_max = CACHE_BITS[base] as i32;
                for q in 1..=k_max {
                    let bits = pulses2bits(band, lm, q);
                    if bits <= 1 {
                        continue;
                    }
                    let back = bits2pulses(band, lm, bits);
                    assert_eq!(
                        pulses2bits(band, lm, back),
                        bits,
                        "band={band} lm={lm} q={q}（{bits} 位）回推成了 {back}"
                    );
                }
            }
        }
    }

    #[test]
    fn bits2pulses_is_monotone_in_bits() {
        // 位数越多，能编的脉冲数不能变少
        for band in [0usize, 8, 16, 20] {
            let mut last = 0i32;
            for bits in 1..4000 {
                let q = bits2pulses(band, 3, bits);
                assert!(q >= last, "band={band} bits={bits} 时脉冲数回退了");
                last = q;
            }
        }
    }

    #[test]
    fn bitexact_cos_spans_the_expected_range() {
        // θ ≥ 64（有效码流里 θ = k·16384/qn 且 qn ≤ 256，所以必然是 64 的倍数起步）
        // 时，取值落在 (0, 32767] 且单调递减
        let mut last = i32::MAX;
        for theta in 64..16384 {
            let c = bitexact_cos(theta);
            assert!((1..=32767).contains(&c), "θ={theta} 得到 {c}");
            assert!(c <= last, "θ={theta} 时不再单调递减");
            last = c;
        }
        assert_eq!(bitexact_cos(64), 32767);
        // θ=16384 会让 `tmp` 越过 32767（参考实现那句 `celt_assert(tmp<=32767)`
        // 就是在声明它不该出现），所以测试只覆盖到 16383
        // θ<64 是参考实现断言排除的输入；真走到那里会因 16 位回绕变成 -32768，
        // 这里照样复现（见 `bitexact_cos` 的注释）
        assert_eq!(bitexact_cos(1), -32768);
    }

    #[test]
    fn isqrt32_matches_floating_sqrt() {
        for v in [1u32, 2, 3, 4, 5, 9, 99, 100, 16641, 1 << 20, (1 << 30) + 7] {
            let want = (v as f64).sqrt().floor() as u32;
            assert_eq!(isqrt32(v), want, "isqrt32({v})");
        }
    }

    #[test]
    fn haar_and_the_two_reorders_are_invertible() {
        // haar1 是正交变换：做两次应当（在 2 的幂长度下）回到原值
        let orig: Vec<f32> = (0..16).map(|i| (i as f32) * 1.25 - 8.0).collect();
        let mut x = orig.clone();
        haar1(&mut x, 16, 1);
        haar1(&mut x, 16, 1);
        for (i, (a, b)) in x.iter().zip(orig.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "第 {i} 个没还原：{a} vs {b}");
        }

        // interleave ∘ deinterleave = 恒等（两种 hadamard 都要）
        for hadamard in [false, true] {
            for (n0, stride) in [(4usize, 2usize), (2, 4), (1, 8), (8, 2)] {
                let src: Vec<f32> = (0..(n0 * stride)).map(|i| i as f32 + 1.0).collect();
                let mut x = src.clone();
                deinterleave_hadamard(&mut x, n0, stride, hadamard);
                interleave_hadamard(&mut x, n0, stride, hadamard);
                for (i, (a, b)) in x.iter().zip(src.iter()).enumerate() {
                    assert_eq!(a, b, "n0={n0} stride={stride} hadamard={hadamard} 第 {i} 个");
                }
            }
        }
    }

    #[test]
    fn stereo_merge_recovers_two_unit_channels() {
        // 造一对**正交**的单位范数 mid/side：正交时
        //   El = Er = mid² + 1（浮点构建下 `SHR32(mid,1)` 是空操作！），
        // 增益 = 1/√El，左右输出就是 (mid·X ∓ Y)/√El。这里按这个式子独立算一遍再比。
        let n = 16usize;
        let mut x: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let mut y: Vec<f32> = (0..n).map(|i| if i % 2 == 1 { 1.0 } else { 0.0 }).collect();
        renormalise_vector(&mut x, n, 1.0);
        renormalise_vector(&mut y, n, 1.0);
        let xy: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        assert!(xy.abs() < 1e-6, "构造出来的 mid/side 应当正交");

        let mid: f32 = 32767.0 / 32768.0;
        let want_l: Vec<f32> = (0..n).map(|j| mid * x[j] - y[j]).collect();
        let want_r: Vec<f32> = (0..n).map(|j| mid * x[j] + y[j]).collect();
        let gain = 1.0 / (mid * mid + 1.0).sqrt();
        stereo_merge(&mut x, &mut y, mid, n);
        for j in 0..n {
            assert!((x[j] - gain * want_l[j]).abs() < 1e-5, "左声道第 {j} 个");
            assert!((y[j] - gain * want_r[j]).abs() < 1e-5, "右声道第 {j} 个");
        }
        // 正交输入下两声道能量相同
        let el: f32 = x.iter().map(|a| a * a).sum();
        let er: f32 = y.iter().map(|a| a * a).sum();
        assert!((el - er).abs() < 1e-5, "对称输入下两声道能量应当相同：{el} vs {er}");
    }

    #[test]
    fn stereo_merge_collapses_when_one_side_is_empty() {
        // side 全 0 → Er/El 会踩到下限，参考实现直接让 Y=X
        let mut x = vec![0.5f32, -0.5, 0.5, -0.5];
        let mut y = vec![0.0f32; 4];
        stereo_merge(&mut x, &mut y, 1.0, 4);
        assert_eq!(x, y, "退化情形下两声道应当相同");
    }

    #[test]
    fn energy_finalise_spends_exactly_the_remaining_bits() {
        // 给足够多的零头位：每个"还没到 8 位"的带都要被花掉 channels 位
        let mut alloc = rate::Allocation {
            pulses: [0; NB_EBANDS],
            fine_quant: [0; NB_EBANDS],
            fine_priority: [0; NB_EBANDS],
            offsets: [0; NB_EBANDS],
            cap: [0; NB_EBANDS],
            alloc_trim: 5,
            coded_bands: 0,
            intensity: 0,
            dual_stereo: false,
            balance: 0,
            anti_collapse_rsv: 0,
            bits: 0,
        };
        alloc.fine_quant[0] = 3;
        alloc.fine_quant[1] = MAX_FINE_BITS; // 已经满位，跳过
        alloc.fine_priority[0] = 0;
        let data = [0u8; 32];
        let mut dec = RangeDecoder::new(&data);
        let mut st = CeltState::default();
        unquant_energy_finalise(&mut dec, &alloc, 1, 0, 2, 100, &mut st).unwrap();
        // 第 0 带补了 ±半档：±0.5·2^(14-3-1)/16384 = ±0.5·1024/16384 = ±0.03125
        assert!(
            (st.old_ebands[0][0].abs() - 0.03125).abs() < 1e-6,
            "第 0 带应当补上 ±0.03125，实际 {}",
            st.old_ebands[0][0]
        );
        assert_eq!(st.old_ebands[0][1], 0.0, "满 8 位的带不该再补");
    }

    #[test]
    fn energy_finalise_stops_when_bits_run_out() {
        let alloc = rate::Allocation {
            pulses: [0; NB_EBANDS],
            fine_quant: [1; NB_EBANDS],
            fine_priority: [0; NB_EBANDS],
            offsets: [0; NB_EBANDS],
            cap: [0; NB_EBANDS],
            alloc_trim: 5,
            coded_bands: 0,
            intensity: 0,
            dual_stereo: false,
            balance: 0,
            anti_collapse_rsv: 0,
            bits: 0,
        };
        let data = [0u8; 32];
        let mut dec = RangeDecoder::new(&data);
        let mut st = CeltState::default();
        // 只剩 2 位、两声道 → 只能补第 0 带那一次
        unquant_energy_finalise(&mut dec, &alloc, 2, 0, 21, 2, &mut st).unwrap();
        assert_ne!(st.old_ebands[0][0], 0.0);
        for i in 1..NB_EBANDS {
            assert_eq!(st.old_ebands[0][i], 0.0, "位用完了，第 {i} 带不该再补");
        }
    }

    #[test]
    fn anti_collapse_fills_only_the_collapsed_blocks() {
        // 一条带：LM=2（4 个短块）、N0=4、掩码只有第 0 块有系数
        let m = 4usize;
        let size = m * SHORT_MDCT as usize;
        let mut x = vec![vec![0f32; size]; 1];
        let mut masks = vec![0u8; NB_EBANDS];
        masks[17] = 0b1110; // 第 0 个短块塌陷
        let log_e = [[-10f32; NB_EBANDS]; 2];
        let prev1 = [[-20f32; NB_EBANDS]; 2];
        let prev2 = [[-20f32; NB_EBANDS]; 2];
        let mut pulses = [0i32; NB_EBANDS];
        pulses[17] = 8;
        anti_collapse(
            &mut x,
            &masks,
            2,
            1,
            17,
            18,
            &log_e,
            &prev1,
            &prev2,
            &pulses,
            0x1234,
        );
        let base = E_BANDS[17] as usize * m;
        // 第 0 个短块（k=0）被填了噪声、第 1 个短块（k=1）应当还是 0
        let filled = (0..4).any(|j| x[0][base + (j << 2)] != 0.0);
        assert!(filled, "塌陷的短块应当被填上噪声");
        for j in 0..4 {
            assert_eq!(x[0][base + (j << 2) + 1], 0.0, "没塌陷的短块不该被动");
        }
    }

    #[test]
    fn quant_all_bands_on_all_zero_data_produces_a_finite_shape() {
        // 位数全部用完（静音帧那样）时不该 panic，也不该产出 NaN/Inf
        let data = [0u8; 64];
        let mut dec = RangeDecoder::new(&data);
        dec.force_tell(64 * 8);
        let pulses = [0i32; NB_EBANDS];
        let tf = [0i32; NB_EBANDS];
        let inp = ShapeInput {
            pulses: &pulses,
            tf: &tf,
            start: 0,
            end: 21,
            lm: 3,
            transient: false,
            spread: 2,
            dual_stereo: false,
            intensity: 0,
            coded_bands: 1,
            total_bits: 64 * 8 * 8,
            balance: 0,
            seed: 0,
        };
        let out = quant_all_bands(&mut dec, &inp, 1).unwrap();
        assert_eq!(out.x.len(), 1);
        assert_eq!(out.x[0].len(), 8 * SHORT_MDCT as usize);
        for v in &out.x[0] {
            assert!(v.is_finite(), "形状里出现了非有限值：{v}");
        }
        assert_eq!(out.collapse_masks.len(), NB_EBANDS);
    }

    #[test]
    fn quant_all_bands_silences_bands_above_the_bandwidth() {
        // 只编到第 3 带：第 3 带之后必须是 0（宽度窄的帧就是这样省位的）
        let data = [0u8; 64];
        let mut dec = RangeDecoder::new(&data);
        dec.force_tell(64 * 8);
        let pulses = [0i32; NB_EBANDS];
        let tf = [0i32; NB_EBANDS];
        let inp = ShapeInput {
            pulses: &pulses,
            tf: &tf,
            start: 0,
            end: 3,
            lm: 0,
            transient: false,
            spread: 2,
            dual_stereo: false,
            intensity: 0,
            coded_bands: 0,
            total_bits: 64 * 8 * 8,
            balance: 0,
            seed: 0,
        };
        let out = quant_all_bands(&mut dec, &inp, 1).unwrap();
        let from = E_BANDS[3] as usize;
        assert!(out.x[0][from..].iter().all(|&v| v == 0.0), "带宽之外应当是 0");
    }
}
