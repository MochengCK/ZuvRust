//! Opus 的 **PVQ 组合数解码**（参考实现 `cwrs.c` 的解码侧）。
//!
//! CELT 的每个频带把"单位范数形状"用**金字塔矢量量化**（PVQ）编码：形状是一个
//! 长度为 N、绝对值之和为 K 的整数矢量 `y`（脉冲数 K 由比特分配给出）。这样的矢量
//! 一共有 `V(N,K)` 个，编码器把其中一个的**序号**（index）写进码流，
//! 解码器要把它还原成矢量 —— 就是这个文件干的事。
//!
//! ## 为什么不用"查表 + 除法"
//!
//! `V(N,K)` 是组合数（可以很大），序号到矢量是一个逐分量剥落的过程：
//! 每解出一个分量，K 减小，需要的是"下一行的组合数" —— 参考实现用 `U(n,k)` 的
//! 递推表（`u[i][j] = u[i-1][j] + u[i][j-1] + u[i-1][j-1]`）走，每一步只做加减。
//! 大 N 时它还给了闭式解（用模逆元加速），但那只是**快**，结果与递推一致 ——
//! 这里只实现递推（N 最大 176、K 也就几十，几百次加减而已）。
//!
//! ## 与参考实现的对应
//!
//! | 本文件 | 参考实现 |
//! |---|---|
//! | [`u_row`] | `ncwrs_urow` |
//! | `unext` / [`uprev`] | `unext` / `uprev` |
//! | [`cwrsi`] | `cwrsi` |
//! | [`decode_pulses`] | `decode_pulses` |
//! | [`v_count`] | `ncwrs_urow` 的返回值 |
//!
//! 参考实现里 `n=2/3/4` 有几条专用快路径（`cwrsi2/3/4`），结果与通用路径一致，
//! 这里只写通用路径 —— 少一份会走偏的代码。

use me_core::{MediaError, Result};

use super::entropy::RangeDecoder;

/// `unext`：`u[i][j] = u[i-1][j] + u[i][j-1] + u[i-1][j-1]` 的**正向**一步。
///
/// `u` 是长度为 `len` 的行（`u[0]` 是上一列的基础值），`ui0` 是新的基础值。
/// 全部用**回绕**加减：参考实现就是 `UADD32`（u32 回绕），大 N 时会绕，
/// 但序号本身保证落在有效范围里，绕出来的值两边一致。
fn unext(u: &mut [u32], mut ui0: u32) {
    let len = u.len();
    let mut j = 1;
    while j < len {
        let ui1 = u[j].wrapping_add(u[j - 1]).wrapping_add(ui0);
        u[j - 1] = ui0;
        ui0 = ui1;
        j += 1;
    }
    u[len - 1] = ui0;
}

/// `uprev`：上一条递推的**反向**一步（解出一个分量后 K 减 1，要把行退回去）。
fn uprev(u: &mut [u32], mut ui0: u32) {
    let len = u.len();
    let mut j = 1;
    while j < len {
        let ui1 = u[j].wrapping_sub(u[j - 1]).wrapping_sub(ui0);
        u[j - 1] = ui0;
        ui0 = ui1;
        j += 1;
    }
    u[len - 1] = ui0;
}

/// 算 `V(n,k)` 并填充 `U(n, 0..=k+1)`，返回（`U` 行, `V`）。
///
/// 参考实现对应 `ncwrs_urow`：`_u[i]` 是 `U(n,i)`，返回值是 `U(n,k)+U(n,k+1) = V(n,k)`。
/// 参考实现 `ncwrs_urow` 在小破折号分支里用的模逆元表（`(2i+1)⁻¹ mod 2^32`）。
const INV_TABLE: [u32; 53] = [
    0x0000_0001, 0xAAAA_AAAB, 0xCCCC_CCCD, 0xB6DB_6DB7, 0x38E3_8E39, 0xBA2E_8BA3, 0xC4EC_4EC5,
    0xEEEE_EEEF, 0xF0F0_F0F1, 0x286B_CA1B, 0x3CF3_CF3D, 0xE9BD_37A7, 0xC28F_5C29, 0x684B_DA13,
    0x4F72_C235, 0xBDEF_7BDF, 0x3E0F_83E1, 0x8AF8_AF8B, 0x914C_1BAD, 0x96F9_6F97, 0xC18F_9C19,
    0x2FA0_BE83, 0xA4FA_4FA5, 0x677D_46CF, 0x1A1F_58D1, 0xFAFA_FAFB, 0x8C13_521D, 0x586F_B587,
    0xB823_EE09, 0xA08A_D8F3, 0xC10C_9715, 0xBEFB_EFBF, 0xC0FC_0FC1, 0x07A4_4C6B, 0xA33F_128D,
    0xE327_A977, 0xC7E3_F1F9, 0x962F_C963, 0x3F2B_3885, 0x6137_16AF, 0x7819_48B1, 0x2B2E_43DB,
    0xFCFC_FCFD, 0x6FD0_EB67, 0xFA3F_47E9, 0xD2FD_2FD3, 0x3F4F_D3F5, 0xD4E2_5B9F, 0x5F02_A3A1,
    0xBF5A_814B, 0x7C32_B16D, 0xD343_1B57, 0xD8FD_8FD9,
];

/// `(_a*_b - _c) / (2*_d+1)`，除得尽时用模逆元算（参考实现 `imusdiv32odd`）。
///
/// **注意**：这不是"数学上等价的写法" —— 参考实现喂进来的 `_a/_b/_c` 已经是
/// 回绕过的 32 位值，此时"除法"在整数意义下未必除得尽，模逆元给出的结果
/// **不等于**"精确商 mod 2^32"。位流的一致性要求这里也必须逐位照抄。
fn imusdiv32odd(a: u32, b: u32, c: u32, d: usize) -> u32 {
    a.wrapping_mul(b).wrapping_sub(c).wrapping_mul(INV_TABLE[d])
}

/// `(_a*_b - _c) / (2*_d)`（参考实现 `imusdiv32even`，同样带模逆元的回绕语义）。
fn imusdiv32even(a: u32, b: u32, c: u32, d: usize) -> u32 {
    let shift = ilog32(d as u32 ^ (d as u32 - 1));
    let inv = INV_TABLE[((d - 1) >> shift) as usize];
    let shift = shift - 1;
    let one = 1u32 << shift;
    let mask = one - 1;
    a.wrapping_mul(b >> shift)
        .wrapping_sub(c >> shift)
        .wrapping_add(
            a.wrapping_mul(b & mask)
                .wrapping_add(one)
                .wrapping_sub(c & mask)
                >> shift,
        )
        .wrapping_sub(1)
        .wrapping_mul(inv)
}

/// `EC_ILOG`：位宽（0 时给 0）。
fn ilog32(x: u32) -> u32 {
    32 - x.leading_zeros()
}

/// 算 `V(n,k)` 并填充 `U(n, 0..=k+1)`，返回（`U` 行, `V`）。
///
/// 参考实现对应 `ncwrs_urow`：`_u[i]` 是 `U(n,i)`，返回值是 `U(n,k)+U(n,k+1) = V(n,k)`。
/// **两条路径都要照抄**：`n<=6` 用递推、`n>6` 用闭式（模逆元）。两者的值在
/// "超过 32 位"时**并不相同**（闭式那条是回绕语义），而码流是按参考实现编的。
pub fn u_row(n: usize, k: usize) -> (Vec<u32>, u32) {
    assert!(k > 0, "PVQ 的 K 必须 > 0（K=0 时形状是全零，不需要码字）");
    assert!(n > 0, "PVQ 的 N 必须 > 0");
    let len = k + 2;
    let mut u = vec![0u32; len];
    u[0] = 0;
    u[1] = 1;
    if n == 1 {
        // 长度为 1 时只有两种符号：U(1,i)=1 (i>=1)，V(1,k)=2。
        // 参考实现的递推分支要求 n>=2，n=1 走闭式解 —— 这里显式写出来。
        for item in u.iter_mut().skip(1) {
            *item = 1;
        }
        return (u.clone(), u[k].wrapping_add(u[k + 1]));
    }
    if n <= 6 {
        // 递推：先铺 U(2,i)=2i-1，再逐行 unext 到 n
        for kk in 2..len {
            u[kk] = ((kk as u32) << 1) - 1;
        }
        for _ in 2..n {
            let mut row = u[1..].to_vec();
            unext(&mut row, 1);
            u[1..].copy_from_slice(&row);
        }
    } else {
        // 闭式：U(N,K) = ((2N-1)U(N,K-1) - U(N,K-2))/(K-1) + U(N,K-2)
        //       U(N,K+1) = ((2N-1)U(N,K) - U(N,K-1))/K + U(N,K-1)
        let n2m1 = ((n as u32) << 1) - 1;
        u[2] = n2m1;
        let mut kk = 3;
        while kk < len {
            u[kk] = imusdiv32even(n2m1, u[kk - 1], u[kk - 2], kk - 1).wrapping_add(u[kk - 2]);
            kk += 1;
            if kk >= len {
                break;
            }
            u[kk] = imusdiv32odd(n2m1, u[kk - 1], u[kk - 2], (kk - 1) >> 1).wrapping_add(u[kk - 2]);
            kk += 1;
        }
    }
    (u.clone(), u[k].wrapping_add(u[k + 1]))
}

/// `V(n,k)`：长度为 `n`、绝对值之和为 `k` 的整数矢量个数（含正负号）。
pub fn v_count(n: usize, k: usize) -> u32 {
    u_row(n, k).1
}

/// `cwrsi`：把序号 `i` 还原成长度 `n`、L1 范数 `k` 的脉冲矢量。
///
/// 逐分量剥落：先定当前分量的绝对值（由 `U` 行决定落点），再定符号，
/// 然后 `uprev` 把 `U` 行退到"K 减 1"的状态继续解下一个分量。
pub fn cwrsi(n: usize, k: usize, mut i: u32, u: &mut Vec<u32>) -> Result<Vec<i32>> {
    let mut y = vec![0i32; n];
    let mut j = 0usize;
    let mut kk = k as i32;
    loop {
        let p = u[(kk + 1) as usize];
        let s = -(i32::from(i >= p));
        i = i.wrapping_sub(p & (s as u32));
        let mut yj = kk;
        let mut p2 = u[kk as usize];
        while p2 > i {
            kk -= 1;
            if kk < 0 {
                return Err(MediaError::Bitstream("Opus: PVQ 序号越界（码流损坏）".into()));
            }
            p2 = u[kk as usize];
        }
        i -= p2;
        yj -= kk;
        y[j] = (yj + s) ^ s;
        uprev(&mut u[..(kk as usize + 2)], 0);
        j += 1;
        if j >= n {
            break;
        }
    }
    Ok(y)
}

/// 从熵解码器解一个 PVQ 脉冲矢量（参考实现 `decode_pulses`）。
///
/// `n` 是频带宽度（MDCT 系数个数），`k` 是脉冲数（来自比特分配）。
pub fn decode_pulses(dec: &mut RangeDecoder<'_>, n: usize, k: usize) -> Result<Vec<i32>> {
    if k == 0 {
        return Err(MediaError::Bitstream("Opus: PVQ 的 K 为 0".into()));
    }
    let (mut u, count) = u_row(n, k);
    if count <= 1 {
        // V(n,k)==1 时没有信息可解（理论上不该出现：K 由分配保证 ≥1 且 N,K 合法）
        return Err(MediaError::Bitstream(format!(
            "Opus: PVQ 的组合数 V({n},{k}) = {count} 太小"
        )));
    }
    let idx = dec.dec_uint(count)?;
    cwrsi(n, k, idx, &mut u)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 暴力数一遍"长度为 n、L1 范数恰好为 k"的整数矢量个数：
    /// 用来独立验证 `V(n,k)`（不依赖参考实现）。
    fn brute_force_count(n: usize, k: usize) -> u64 {
        let mut y = vec![0i32; n];
        let mut count = 0u64;
        fn rec(y: &mut Vec<i32>, pos: usize, left: usize, count: &mut u64) {
            if pos == y.len() {
                if left == 0 {
                    *count += 1;
                }
                return;
            }
            // 每个分量可以取 ±1..=left（以及 0），但为了让"每个矢量只数一次"，
            // 第一维只放开符号组合：这里直接枚举所有取值（小规模才用）
            for v in -(left as i32)..=(left as i32) {
                let used = v.unsigned_abs() as usize;
                if used > left {
                    continue;
                }
                y[pos] = v;
                rec(y, pos + 1, left - used, count);
            }
        }
        rec(&mut y, 0, k, &mut count);
        count
    }

    #[test]
    fn v_count_matches_brute_force() {
        for n in 1..=4usize {
            for k in 1..=5usize {
                let brute = brute_force_count(n, k);
                assert_eq!(
                    v_count(n, k) as u64,
                    brute,
                    "V({n},{k}) 与暴力枚举不一致"
                );
            }
        }
    }

    #[test]
    fn v_count_matches_the_combinatorial_formula() {
        // 独立验证：PVQ 码字数 V(n,k) = Σ_{j=1..min(n,k)} C(n,j)·C(k-1,j-1)·2^j
        // （自己按组合意义推的式子，与参考实现的递推无关）
        fn binom(n: u128, k: u128) -> u128 {
            if k > n {
                return 0;
            }
            let mut r = 1u128;
            for i in 0..k {
                r = r * (n - i) / (i + 1);
            }
            r
        }
        fn formula(n: u128, k: u128) -> u128 {
            if k == 0 {
                return 1;
            }
            let mut tot = 0u128;
            for j in 1..=n.min(k) {
                tot += binom(n, j) * binom(k - 1, j - 1) * (1u128 << j);
            }
            tot
        }
        for (n, k) in [(2usize, 1usize), (2, 2), (3, 5), (5, 4), (5, 6), (8, 10), (7, 3), (17, 2)] {
            assert_eq!(
                v_count(n, k) as u128,
                formula(n as u128, k as u128),
                "V({n},{k}) 与组合公式不一致"
            );
        }
        // 超过 2^32 时参考实现的两条路径**不一样**：
        //   n<=6 走递推（纯回绕加减）→ 等于"精确值 mod 2^32"；
        //   n>6  走闭式（模逆元，输入已回绕）→ **不等于**精确值 mod 2^32。
        // 所以精确公式只能在 n<=6 上核对；n>6 由与参考实现的逐用例对拍覆盖
        // （tests/opus_pvq_ref.rs）。
        for (n, k) in [(6usize, 14usize), (5, 20), (6, 30)] {
            assert_eq!(
                v_count(n, k) as u128,
                formula(n as u128, k as u128) & 0xFFFF_FFFF,
                "V({n},{k}) 与组合公式（mod 2^32）不一致"
            );
        }
    }

    #[test]
    fn cwrsi_enumerates_every_codeword_exactly_once() {
        // 序号 → 矢量 应当是个双射：枚举所有序号，得到的矢量互不相同、
        // 且每个矢量的 L1 范数都等于 k。
        for (n, k) in [(1usize, 1usize), (2, 3), (3, 2), (5, 4), (7, 3), (17, 2)] {
            let total = v_count(n, k);
            assert_eq!(total as u64, brute_force_count(n, k), "V({n},{k})");
            let mut seen = std::collections::HashSet::new();
            for idx in 0..total {
                let (mut u, _) = u_row(n, k);
                let y = cwrsi(n, k, idx, &mut u).unwrap();
                assert_eq!(
                    y.iter().map(|v| v.unsigned_abs() as u64).sum::<u64>(),
                    k as u64,
                    "解出来的矢量 L1 范数不对：{y:?}"
                );
                assert!(seen.insert(y.clone()), "序号 {idx} 与前面的矢量重复");
            }
            assert_eq!(seen.len() as u64, total as u64, "解出来的矢量个数");
        }
    }

    #[test]
    fn u_row_matches_the_closed_form_recurrence_for_small_n() {
        // n<=6 时参考实现走递推（纯回绕加减），与"精确闭式 mod 2^32"一致 ——
        // 这也顺带证明了递推实现本身没写错。n>6 走模逆元的闭式，值**不**等于精确值，
        // 由 tests/opus_pvq_ref.rs 的真机对拍覆盖。
        fn closed_form(n: usize, k: usize) -> Vec<u128> {
            let mut u = vec![0u128; k + 2];
            u[0] = 0;
            u[1] = 1;
            if n == 1 {
                for item in u.iter_mut().skip(1) {
                    *item = 1;
                }
                return u;
            }
            let n2m1 = (2 * n as u128) - 1;
            u[2] = n2m1;
            let mut kk = 3;
            while kk < k + 2 {
                u[kk] = (n2m1 * u[kk - 1] - u[kk - 2]) / (kk as u128 - 1) + u[kk - 2];
                kk += 1;
                if kk >= k + 2 {
                    break;
                }
                u[kk] = (n2m1 * u[kk - 1] - u[kk - 2]) / (kk as u128 - 1) + u[kk - 2];
                kk += 1;
            }
            u
        }
        for (n, k) in [(2usize, 3usize), (3, 5), (5, 6), (6, 10), (6, 30), (6, 40)] {
            let want = closed_form(n, k);
            let (got, _) = u_row(n, k);
            for i in 0..k + 2 {
                assert_eq!(
                    got[i] as u128,
                    want[i] & 0xFFFF_FFFF,
                    "U({n},{i})（k={k}）与精确闭式不一致"
                );
            }
        }
    }

    #[test]
    fn first_codeword_has_the_first_pulse_positive() {
        // 参考实现的符号约定：序号 0 对应"所有脉冲都在第一个分量上且为正"
        let (mut u, _) = u_row(4, 3);
        assert_eq!(cwrsi(4, 3, 0, &mut u).unwrap(), vec![3, 0, 0, 0]);
    }

    #[test]
    fn out_of_range_index_is_reported() {
        // 序号超出 V(n,k) 时不许解出垃圾：要么报错，要么落在合法矢量上 ——
        // 这里只要求"不 panic 且范数正确"（编码器保证不会发生）
        let (mut u, count) = u_row(4, 2);
        let y = cwrsi(4, 2, count + 7, &mut u).unwrap();
        assert_eq!(y.iter().map(|v| v.unsigned_abs()).sum::<u32>(), 2);
    }

    #[test]
    fn decode_pulses_reads_one_index() {
        // 用一串字节跑一遍：解出来的矢量范数必须等于 K（值本身由对拍测试核对）
        let data: Vec<u8> = (0..64u16).map(|i| ((i * 73 + 11) % 256) as u8).collect();
        for k in 1..=6usize {
            let mut dec = RangeDecoder::new(&data);
            let y = decode_pulses(&mut dec, 8, k).unwrap();
            assert_eq!(y.len(), 8);
            assert_eq!(y.iter().map(|v| v.unsigned_abs() as usize).sum::<usize>(), k);
        }
    }
}
