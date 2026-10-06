//! **帧内预测**（ITU-T H.264 §8.3）：`Intra_16x16` 四种、`Intra_4x4` / `Intra_8x8`
//! 各九种、色度四种。
//!
//! 这一层是**纯函数**：给它参考样本和模式，吐预测块。取样本（含可用性推导）
//! 在 [`super::reconstruct`] —— 那里才知道"哪块已经解出来了"。
//!
//! ## 二维数组的布局
//!
//! 一律 `idx = y * n + x`（**行优先、y 在前**），和 [`super::transform`] 的
//! `i * n + j`（`i` = 行）是同一个约定。
//!
//! ## 参考样本的记号
//!
//! 规范用 `p[x, y]`，其中 `x = -1, y = -1..n-1` 是**左边一列**、
//! `x = 0..2n-1, y = -1` 是**上面一行**、`p[-1, -1]` 是左上角。这里用
//! [`Refs4`] / [`Refs8`] 把它摆平：
//!
//! - `top[x]` = `p[x, -1]`（`x = 0..(2n-1)`，`Intra_4x4` 用 `0..7`）
//! - `left[y]` = `p[-1, y]`（`y = 0..n-1`）
//! - `top_left` = `p[-1, -1]`
//!
//! ## 三条容易漏的规则
//!
//! 1. **右上角那 4 个（8×8 是 8 个）样本要"拖过去"**（§8.3.1.2 / §8.3.2.2）：
//!    当 `p[4..7, -1]` 不可用而 `p[3, -1]` 可用时，用 `p[3, -1]` 顶替它们
//!    **并标记为可用**。不写这条，`Diagonal_Down_Left` 之类的模式在帧右边界会花。
//! 2. **`Intra_4x4` 的 `p[4..7, -1]`，当 `luma4x4BlkIdx` 是 3 或 11 时一律不可用**
//!    （那是"右上角"宏块，不是直接邻居）。
//! 3. **`Intra_8x8` 的参考样本要先过一遍低通**（§8.3.2.2.1，式 8-78 ~ 8-88）——
//!    斜率模式在 8×8 上不滤波会有明显的块状锯齿。

/// `Intra_4x4` 的 13 个参考样本（`p[x,-1]` 取前 4 个就够，但方向模式要 0..7）。
#[derive(Debug, Clone, Copy)]
pub struct Refs4 {
    pub top: [i32; 8],
    pub left: [i32; 4],
    pub top_left: i32,
    pub top_avail: [bool; 8],
    pub left_avail: [bool; 4],
    pub top_left_avail: bool,
}

/// `Intra_8x8` 的参考样本（`p[x,-1]` 到 15，`p[-1,y]` 到 7）。
#[derive(Debug, Clone, Copy)]
pub struct Refs8 {
    pub top: [i32; 16],
    pub left: [i32; 8],
    pub top_left: i32,
    pub top_avail: [bool; 16],
    pub left_avail: [bool; 8],
    pub top_left_avail: bool,
}

impl Refs4 {
    #[inline]
    fn t(&self, x: i32) -> i32 {
        if x < 0 {
            self.top_left
        } else {
            self.top[x as usize]
        }
    }

    #[inline]
    fn l(&self, y: i32) -> i32 {
        if y < 0 {
            self.top_left
        } else {
            self.left[y as usize]
        }
    }
}

impl Refs8 {
    #[inline]
    fn t(&self, x: i32) -> i32 {
        if x < 0 {
            self.top_left
        } else {
            self.top[x as usize]
        }
    }

    #[inline]
    fn l(&self, y: i32) -> i32 {
        if y < 0 {
            self.top_left
        } else {
            self.left[y as usize]
        }
    }
}

/// `Intra_4x4` / `Intra_8x8` 的九种模式（表 8-2 / 8-3）。
pub const N4X4_MODES: usize = 9;

/// `Intra_4x4` 的预测（§8.3.1.2.1 ~ §8.3.1.2.9）。
///
/// 返回值是 4×4 的预测块（`idx = y * 4 + x`）。
pub fn predict_4x4(mode: u8, r: &Refs4, bit_depth: u8) -> [i32; 16] {
    let mut p = [0i32; 16];
    let set = |x: usize, y: usize, v: i32, p: &mut [i32; 16]| p[y * 4 + x] = v;
    match mode {
        0 => {
            for y in 0..4 {
                for x in 0..4 {
                    set(x, y, r.t(x as i32), &mut p);
                }
            }
        }
        1 => {
            for y in 0..4 {
                for x in 0..4 {
                    set(x, y, r.l(y as i32), &mut p);
                }
            }
        }
        2 => {
            let v = dc_value(
                r.top_avail[..4].iter().all(|&a| a),
                r.left_avail.iter().all(|&a| a),
                || r.top[..4].iter().sum::<i32>(),
                || r.left.iter().sum::<i32>(),
                4,
                bit_depth,
            );
            p = [v; 16];
        }
        3 => {
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let v = if x == 3 && y == 3 {
                        (r.t(6) + 3 * r.t(7) + 2) >> 2
                    } else {
                        (r.t(x + y) + 2 * r.t(x + y + 1) + r.t(x + y + 2) + 2) >> 2
                    };
                    set(x as usize, y as usize, v, &mut p);
                }
            }
        }
        4 => {
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let v = if x > y {
                        (r.t(x - y - 2) + 2 * r.t(x - y - 1) + r.t(x - y) + 2) >> 2
                    } else if x < y {
                        (r.l(y - x - 2) + 2 * r.l(y - x - 1) + r.l(y - x) + 2) >> 2
                    } else {
                        (r.t(0) + 2 * r.top_left + r.l(0) + 2) >> 2
                    };
                    set(x as usize, y as usize, v, &mut p);
                }
            }
        }
        5 => {
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let z = 2 * x - y;
                    let v = if z >= 0 && z % 2 == 0 {
                        (r.t(x - (y >> 1) - 1) + r.t(x - (y >> 1)) + 1) >> 1
                    } else if z == 1 || z == 3 || z == 5 {
                        (r.t(x - (y >> 1) - 2) + 2 * r.t(x - (y >> 1) - 1) + r.t(x - (y >> 1)) + 2)
                            >> 2
                    } else if z == -1 {
                        (r.l(0) + 2 * r.top_left + r.t(0) + 2) >> 2
                    } else {
                        (r.l(y - 1) + 2 * r.l(y - 2) + r.l(y - 3) + 2) >> 2
                    };
                    set(x as usize, y as usize, v, &mut p);
                }
            }
        }
        6 => {
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let z = 2 * y - x;
                    let v = if z >= 0 && z % 2 == 0 {
                        (r.l(y - (x >> 1) - 1) + r.l(y - (x >> 1)) + 1) >> 1
                    } else if z == 1 || z == 3 || z == 5 {
                        (r.l(y - (x >> 1) - 2) + 2 * r.l(y - (x >> 1) - 1) + r.l(y - (x >> 1)) + 2)
                            >> 2
                    } else if z == -1 {
                        (r.l(0) + 2 * r.top_left + r.t(0) + 2) >> 2
                    } else {
                        (r.t(x - 1) + 2 * r.t(x - 2) + r.t(x - 3) + 2) >> 2
                    };
                    set(x as usize, y as usize, v, &mut p);
                }
            }
        }
        7 => {
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let half = x + (y >> 1);
                    let v = if y % 2 == 0 {
                        (r.t(half) + r.t(half + 1) + 1) >> 1
                    } else {
                        (r.t(half) + 2 * r.t(half + 1) + r.t(half + 2) + 2) >> 2
                    };
                    set(x as usize, y as usize, v, &mut p);
                }
            }
        }
        8 => {
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let z = x + 2 * y;
                    let q = y + (x >> 1);
                    let v = if z >= 0 && z % 2 == 0 && z <= 4 {
                        (r.l(q) + r.l(q + 1) + 1) >> 1
                    } else if z == 1 || z == 3 {
                        (r.l(q) + 2 * r.l(q + 1) + r.l(q + 2) + 2) >> 2
                    } else if z == 5 {
                        (r.l(2) + 3 * r.l(3) + 2) >> 2
                    } else {
                        r.l(3)
                    };
                    set(x as usize, y as usize, v, &mut p);
                }
            }
        }
        _ => {
            // 到不了：mb 层的 `rem_intra4x4_pred_mode` 是 3 位，最大 7 ⇒ 模式最大 8
            p = [1 << (bit_depth - 1); 16];
        }
    }
    p
}

/// `Intra_8x8` 的预测（§8.3.2.2.2 ~ §8.3.2.2.10）；`r` 必须是**已滤波**的参考样本。
pub fn predict_8x8(mode: u8, r: &Refs8, bit_depth: u8) -> [i32; 64] {
    let mut p = [0i32; 64];
    let set = |x: usize, y: usize, v: i32, p: &mut [i32; 64]| p[y * 8 + x] = v;
    match mode {
        0 => {
            for y in 0..8 {
                for x in 0..8 {
                    set(x, y, r.t(x as i32), &mut p);
                }
            }
        }
        1 => {
            for y in 0..8 {
                for x in 0..8 {
                    set(x, y, r.l(y as i32), &mut p);
                }
            }
        }
        2 => {
            let v = dc_value(
                r.top_avail[..8].iter().all(|&a| a),
                r.left_avail.iter().all(|&a| a),
                || r.top[..8].iter().sum::<i32>(),
                || r.left.iter().sum::<i32>(),
                8,
                bit_depth,
            );
            p = [v; 64];
        }
        3 => {
            for y in 0..8i32 {
                for x in 0..8i32 {
                    let v = if x == 7 && y == 7 {
                        (r.t(14) + 3 * r.t(15) + 2) >> 2
                    } else {
                        (r.t(x + y) + 2 * r.t(x + y + 1) + r.t(x + y + 2) + 2) >> 2
                    };
                    set(x as usize, y as usize, v, &mut p);
                }
            }
        }
        4 => {
            for y in 0..8i32 {
                for x in 0..8i32 {
                    let v = if x > y {
                        (r.t(x - y - 2) + 2 * r.t(x - y - 1) + r.t(x - y) + 2) >> 2
                    } else if x < y {
                        (r.l(y - x - 2) + 2 * r.l(y - x - 1) + r.l(y - x) + 2) >> 2
                    } else {
                        (r.t(0) + 2 * r.top_left + r.l(0) + 2) >> 2
                    };
                    set(x as usize, y as usize, v, &mut p);
                }
            }
        }
        5 => {
            for y in 0..8i32 {
                for x in 0..8i32 {
                    let z = 2 * x - y;
                    let v = if z >= 0 && z % 2 == 0 {
                        (r.t(x - (y >> 1) - 1) + r.t(x - (y >> 1)) + 1) >> 1
                    } else if z > 0 {
                        (r.t(x - (y >> 1) - 2) + 2 * r.t(x - (y >> 1) - 1) + r.t(x - (y >> 1)) + 2)
                            >> 2
                    } else if z == -1 {
                        (r.l(0) + 2 * r.top_left + r.t(0) + 2) >> 2
                    } else {
                        // ⚠️ 8×8 这一支与 4×4 **不一样**：用的是 `y - 2x - k`，不是 `y - k`
                        (r.l(y - 2 * x - 1) + 2 * r.l(y - 2 * x - 2) + r.l(y - 2 * x - 3) + 2) >> 2
                    };
                    set(x as usize, y as usize, v, &mut p);
                }
            }
        }
        6 => {
            for y in 0..8i32 {
                for x in 0..8i32 {
                    let z = 2 * y - x;
                    let v = if z >= 0 && z % 2 == 0 {
                        (r.l(y - (x >> 1) - 1) + r.l(y - (x >> 1)) + 1) >> 1
                    } else if z > 0 {
                        (r.l(y - (x >> 1) - 2) + 2 * r.l(y - (x >> 1) - 1) + r.l(y - (x >> 1)) + 2)
                            >> 2
                    } else if z == -1 {
                        (r.l(0) + 2 * r.top_left + r.t(0) + 2) >> 2
                    } else {
                        (r.t(x - 2 * y - 1) + 2 * r.t(x - 2 * y - 2) + r.t(x - 2 * y - 3) + 2) >> 2
                    };
                    set(x as usize, y as usize, v, &mut p);
                }
            }
        }
        7 => {
            for y in 0..8i32 {
                for x in 0..8i32 {
                    let half = x + (y >> 1);
                    let v = if y % 2 == 0 {
                        (r.t(half) + r.t(half + 1) + 1) >> 1
                    } else {
                        (r.t(half) + 2 * r.t(half + 1) + r.t(half + 2) + 2) >> 2
                    };
                    set(x as usize, y as usize, v, &mut p);
                }
            }
        }
        8 => {
            for y in 0..8i32 {
                for x in 0..8i32 {
                    let z = x + 2 * y;
                    let q = y + (x >> 1);
                    let v = if z <= 12 && z % 2 == 0 {
                        (r.l(q) + r.l(q + 1) + 1) >> 1
                    } else if (1..=11).contains(&z) && z % 2 == 1 {
                        (r.l(q) + 2 * r.l(q + 1) + r.l(q + 2) + 2) >> 2
                    } else if z == 13 {
                        (r.l(6) + 3 * r.l(7) + 2) >> 2
                    } else {
                        r.l(7)
                    };
                    set(x as usize, y as usize, v, &mut p);
                }
            }
        }
        _ => {
            p = [1 << (bit_depth - 1); 64];
        }
    }
    p
}

/// `Intra_8x8` 的**参考样本低通**（§8.3.2.2.1，式 8-78 ~ 8-88）。
///
/// 只对"整行/整列都可用"的那些做；`p′[-1,-1]` 单独按式 8-82 ~ 8-84 处理。
pub fn filter_refs8(r: &Refs8) -> Refs8 {
    let mut f = *r;
    // 上一行：p'[x,-1]，x = 0..15
    if r.top_avail[..16].iter().all(|&a| a) {
        f.top[0] = if r.top_left_avail {
            (r.top_left + 2 * r.top[0] + r.top[1] + 2) >> 2
        } else {
            (3 * r.top[0] + r.top[1] + 2) >> 2
        };
        for x in 1..15 {
            f.top[x] = (r.top[x - 1] + 2 * r.top[x] + r.top[x + 1] + 2) >> 2;
        }
        f.top[15] = (r.top[14] + 3 * r.top[15] + 2) >> 2;
    }
    // 左一列：p'[-1,y]，y = 0..7
    if r.left_avail.iter().all(|&a| a) {
        f.left[0] = if r.top_left_avail {
            (r.top_left + 2 * r.left[0] + r.left[1] + 2) >> 2
        } else {
            (3 * r.left[0] + r.left[1] + 2) >> 2
        };
        for y in 1..7 {
            f.left[y] = (r.left[y - 1] + 2 * r.left[y] + r.left[y + 1] + 2) >> 2;
        }
        f.left[7] = (r.left[6] + 3 * r.left[7] + 2) >> 2;
    }
    // 左上角
    if r.top_left_avail {
        f.top_left = if !r.top_avail[0] || !r.left_avail[0] {
            if r.top_avail[0] {
                (3 * r.top_left + r.top[0] + 2) >> 2
            } else if r.left_avail[0] {
                (3 * r.top_left + r.left[0] + 2) >> 2
            } else {
                r.top_left
            }
        } else {
            (r.top[0] + 2 * r.top_left + r.left[0] + 2) >> 2
        };
    }
    f
}

/// DC 模式的三种分支（§8.3.1.2.3 / §8.3.2.2.4 / 色度 §8.3.4.1 共用同一种结构）。
fn dc_value(
    top_ok: bool,
    left_ok: bool,
    top_sum: impl Fn() -> i32,
    left_sum: impl Fn() -> i32,
    n: i32,
    bit_depth: u8,
) -> i32 {
    match (top_ok, left_ok) {
        (true, true) => (top_sum() + left_sum() + n) / (2 * n),
        (false, true) => (left_sum() + (n / 2)) / n,
        (true, false) => (top_sum() + (n / 2)) / n,
        (false, false) => 1 << (bit_depth - 1),
    }
}

/// `Intra_16x16` 的预测（§8.3.3.1 ~ §8.3.3.4），输出 16×16（`idx = y * 16 + x`）。
///
/// `top` / `left` 各 16 个样本，`top_left` = `p[-1,-1]`。
pub fn predict_16x16(
    mode: u8,
    top: &[i32; 16],
    left: &[i32; 16],
    top_left: i32,
    top_ok: bool,
    left_ok: bool,
    bit_depth: u8,
) -> [i32; 256] {
    let mut p = [0i32; 256];
    match mode {
        0 => {
            for y in 0..16 {
                for x in 0..16 {
                    p[y * 16 + x] = top[x];
                }
            }
        }
        1 => {
            for y in 0..16 {
                for x in 0..16 {
                    p[y * 16 + x] = left[y];
                }
            }
        }
        2 => {
            let v = dc_value(
                top_ok,
                left_ok,
                || top.iter().sum::<i32>(),
                || left.iter().sum::<i32>(),
                16,
                bit_depth,
            );
            p = [v; 256];
        }
        3 => {
            // 式 8-122 ~ 8-127
            // ⚠️ `x' = 7` 时 `p[6 - 7]` 是 `p[-1, -1]`（左上角），不是越界
            let t = |i: i32| if i < 0 { top_left } else { top[i as usize] };
            let l = |i: i32| if i < 0 { top_left } else { left[i as usize] };
            let mut h = 0i32;
            for x in 0..8i32 {
                h += (x + 1) * (t(8 + x) - t(6 - x));
            }
            let mut v = 0i32;
            for y in 0..8i32 {
                v += (y + 1) * (l(8 + y) - l(6 - y));
            }
            let a = 16 * (left[15] + top[15]);
            let b = (5 * h + 32) >> 6;
            let c = (5 * v + 32) >> 6;
            let _ = top_left;
            for y in 0..16i32 {
                for x in 0..16i32 {
                    p[(y * 16 + x) as usize] =
                        (a + b * (x - 7) + c * (y - 7) + 16) >> 5;
                }
            }
        }
        _ => p = [1 << (bit_depth - 1); 256],
    }
    p
}

/// 色度预测（§8.3.4.1 ~ §8.3.4.4），输出 `8×8`（4:2:0 的 `MbWidthC`×`MbHeightC`）。
///
/// `top` / `left` 各 8 个，`top_ok` / `left_ok` 是**整行/整列**的可用性
/// （DC 模式还要按 4×4 子块细分，见 [`crate::video::h264::reconstruct`]）。
pub fn predict_chroma(
    mode: u8,
    top: &[i32; 8],
    left: &[i32; 8],
    top_avail: [bool; 8],
    left_avail: [bool; 8],
    bit_depth: u8,
) -> [i32; 64] {
    let mut p = [0i32; 64];
    match mode {
        1 => {
            for y in 0..8 {
                for x in 0..8 {
                    p[y * 8 + x] = left[y];
                }
            }
        }
        2 => {
            for y in 0..8 {
                for x in 0..8 {
                    p[y * 8 + x] = top[x];
                }
            }
        }
        3 => {
            // 式 8-144 ~ 8-149：4:2:0 时 xCF = 0、yCF = 0
            let mut h = 0i32;
            for x in 0..4i32 {
                h += (x + 1) * (top[(4 + x) as usize] - top[(2 - x) as usize]);
            }
            let mut v = 0i32;
            for y in 0..4i32 {
                v += (y + 1) * (left[(4 + y) as usize] - left[(2 - y) as usize]);
            }
            let a = 16 * (left[7] + top[7]);
            let b = (34 * h + 32) >> 6;
            let c = (34 * v + 32) >> 6;
            for y in 0..8i32 {
                for x in 0..8i32 {
                    p[(y * 8 + x) as usize] = (a + b * (x - 3) + c * (y - 3) + 16) >> 5;
                }
            }
        }
        _ => {
            // DC（§8.3.4.1）：**可用性按 4×4 子块各算各的**，不是整块一个值。
            // 而且规范对"上/左都可用"、"xO>0 且 yO==0"、"xO==0 且 yO>0" 三种位置
            // 的**优先级**写得不一样（后者先看上、前者先看左），照着写。
            for by in 0..2usize {
                for bx in 0..2usize {
                    let x0 = bx * 4;
                    let y0 = by * 4;
                    let t = top_avail[x0..x0 + 4].iter().all(|&a| a);
                    let l = left_avail[y0..y0 + 4].iter().all(|&a| a);
                    let sum_t: i32 = top[x0..x0 + 4].iter().sum();
                    let sum_l: i32 = left[y0..y0 + 4].iter().sum();
                    let v = if (bx == 0 && by == 0) || (bx > 0 && by > 0) {
                        match (t, l) {
                            (true, true) => (sum_t + sum_l + 4) >> 3,
                            (false, true) => (sum_l + 2) >> 2,
                            (true, false) => (sum_t + 2) >> 2,
                            (false, false) => 1 << (bit_depth - 1),
                        }
                    } else if bx > 0 {
                        // 上一行、右边的子块：先看上
                        if t {
                            (sum_t + 2) >> 2
                        } else if l {
                            (sum_l + 2) >> 2
                        } else {
                            1 << (bit_depth - 1)
                        }
                    } else {
                        // 左一列、下边的子块：先看左
                        if l {
                            (sum_l + 2) >> 2
                        } else if t {
                            (sum_t + 2) >> 2
                        } else {
                            1 << (bit_depth - 1)
                        }
                    };
                    for y in y0..y0 + 4 {
                        for x in x0..x0 + 4 {
                            p[y * 8 + x] = v;
                        }
                    }
                }
            }
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refs4(top: [i32; 8], left: [i32; 4], tl: i32) -> Refs4 {
        Refs4 {
            top,
            left,
            top_left: tl,
            top_avail: [true; 8],
            left_avail: [true; 4],
            top_left_avail: true,
        }
    }

    /// 模式 0/1 就是**照抄**一行/一列 —— 手算得出来，是最基本的一条。
    #[test]
    fn vertical_and_horizontal_are_plain_copies() {
        let r = refs4([10, 20, 30, 40, 50, 60, 70, 80], [11, 22, 33, 44], 5);
        let v = predict_4x4(0, &r, 8);
        for y in 0..4 {
            assert_eq!(&v[y * 4..y * 4 + 4], &[10, 20, 30, 40], "第 {y} 行");
        }
        let h = predict_4x4(1, &r, 8);
        for y in 0..4 {
            assert_eq!(h[y * 4..y * 4 + 4], [11 + y as i32 * 11; 4], "第 {y} 行");
        }
    }

    /// DC：四边都可用时是**八个样本的和 + 4 再 >> 3**（不是 16 个样本）。
    #[test]
    fn dc_uses_eight_samples_with_rounding() {
        let r = refs4([10, 20, 30, 40, 0, 0, 0, 0], [11, 22, 33, 44], 5);
        let d = predict_4x4(2, &r, 8);
        let expect = (10 + 20 + 30 + 40 + 11 + 22 + 33 + 44 + 4) >> 3;
        assert!(d.iter().all(|&v| v == expect), "{d:?}");
    }

    /// 只有左边可用 → `(左四 + 2) >> 2`；只有上边可用 → `(上四 + 2) >> 2`；
    /// 都不可用 → `1 << (BitDepth-1)`。
    #[test]
    fn dc_falls_back_the_way_the_spec_says() {
        let mut r = refs4([10, 20, 30, 40, 0, 0, 0, 0], [11, 22, 33, 44], 5);
        r.top_avail = [false; 8];
        let d = predict_4x4(2, &r, 8);
        assert!(d.iter().all(|&v| v == (11 + 22 + 33 + 44 + 2) >> 2));

        let mut r = refs4([10, 20, 30, 40, 0, 0, 0, 0], [11, 22, 33, 44], 5);
        r.left_avail = [false; 4];
        let d = predict_4x4(2, &r, 8);
        assert!(d.iter().all(|&v| v == (10 + 20 + 30 + 40 + 2) >> 2));

        let mut r = refs4([10, 20, 30, 40, 0, 0, 0, 0], [11, 22, 33, 44], 5);
        r.left_avail = [false; 4];
        r.top_avail = [false; 8];
        let d = predict_4x4(2, &r, 8);
        assert!(d.iter().all(|&v| v == 128));
    }

    /// 模式 4（Diagonal_Down_Right）的对角线上是 `(t0 + 2*tl + l0 + 2) >> 2`
    /// —— 这一条把"左上角参与"钉住。
    #[test]
    fn diagonal_down_right_uses_the_corner() {
        let r = refs4([10, 20, 30, 40, 0, 0, 0, 0], [11, 22, 33, 44], 5);
        let d = predict_4x4(4, &r, 8);
        let expect = (10 + 2 * 5 + 11 + 2) >> 2;
        assert_eq!(d[0], expect, "(0,0)");
        assert_eq!(d[5], expect, "(1,1)");
        assert_eq!(d[10], expect, "(2,2)");
        assert_eq!(d[15], expect, "(3,3)");
    }

    /// 16×16 的 DC 是**十六个**样本的和 + 16 再 >> 5（比 4×4 多一倍样本）。
    #[test]
    fn intra16x16_dc_sums_sixteen() {
        let top = [16i32; 16];
        let left = [32i32; 16];
        let p = predict_16x16(2, &top, &left, 0, true, true, 8);
        let expect = (16 * 16 + 16 * 32 + 16) >> 5;
        assert!(p.iter().all(|&v| v == expect), "{expect}");
    }

    /// 16×16 的 Plane：`a/b/c` 全为 0 时（平场）输出必须还是那个平场值。
    #[test]
    fn intra16x16_plane_of_a_flat_field_stays_flat() {
        let top = [100i32; 16];
        let left = [100i32; 16];
        let p = predict_16x16(3, &top, &left, 100, true, true, 8);
        // a = 16*(100+100) = 3200，b = c = 0 ⇒ (3200 + 16) >> 5 = 100
        assert!(p.iter().all(|&v| v == 100), "{:?}", &p[..8]);
    }

    /// 8×8 的参考样本滤波：平场必须**保持不变**（这是低通的基本性质，
    /// 系数和是 1）。
    #[test]
    fn refs8_filter_keeps_a_flat_field_flat() {
        let r = Refs8 {
            top: [77; 16],
            left: [77; 8],
            top_left: 77,
            top_avail: [true; 16],
            left_avail: [true; 8],
            top_left_avail: true,
        };
        let f = filter_refs8(&r);
        assert!(f.top.iter().all(|&v| v == 77));
        assert!(f.left.iter().all(|&v| v == 77));
        assert_eq!(f.top_left, 77);
    }

    /// 色度 V/H 就是照抄；色度 DC 在四边都可用时是 4×4 子块的 `(上四+左四+4)>>3`。
    #[test]
    fn chroma_modes_copy_and_dc() {
        let top = [10i32, 20, 30, 40, 50, 60, 70, 80];
        let left = [1i32, 2, 3, 4, 5, 6, 7, 8];
        let v = predict_chroma(2, &top, &left, [true; 8], [true; 8], 8);
        assert_eq!(&v[..8], &[10, 20, 30, 40, 50, 60, 70, 80]);
        let h = predict_chroma(1, &top, &left, [true; 8], [true; 8], 8);
        assert!(h[..8].iter().all(|&x| x == 1));
        let dc = predict_chroma(0, &top, &left, [true; 8], [true; 8], 8);
        // ⚠️ 三种位置的**优先级不一样**（§8.3.4.1）：
        //   子块 (4,0)（`bx>0, by==0`）**先看上**：上可用就只用上
        assert_eq!(dc[4], (50 + 60 + 70 + 80 + 2) >> 2);
        //   子块 (0,4)（`bx==0, by>0`）**先看左**：左可用就只用左
        assert_eq!(dc[4 * 8], (5 + 6 + 7 + 8 + 2) >> 2);
        //   子块 (4,4)（两边都 >0）才是"上下都算"的全平均
        let all = predict_chroma(0, &top, &left, [true; 8], [true; 8], 8);
        assert_eq!(all[4 * 8 + 4], (50 + 60 + 70 + 80 + 5 + 6 + 7 + 8 + 4) >> 3);
    }
}
