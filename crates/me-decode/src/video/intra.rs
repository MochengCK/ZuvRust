//! AV1 **帧内预测**（规范 §7.11.2）—— 自研软解 M5 的前半。
//!
//! 这一层是"第一次真的碰像素"：前面所有层读的都是语法（模式、系数），
//! 这里要按规范把**已经重建好的邻居样本**（上一行、左一列，以及
//! above-right / below-left 的延伸段）拼成 `AboveRow` / `LeftCol`，
//! 再按帧内模式算出一整块预测值。
//!
//! ## 七个预测器（规范 §7.11.2 的六个过程 + 一个前置分支）
//!
//! | 模式 | 过程 | 特点 |
//! |---|---|---|
//! | `DC_PRED` | §7.11.2.5 | 上下/左右边的平均值，`(w+h)>>1` 的四舍五入 |
//! | `PAETH_PRED` | §7.11.2.2 | 三选一（左/上/左上），按距离最小 |
//! | `SMOOTH_PRED` / `_V_` / `_H_` | §7.11.2.6 | 查 `Sm_Weights_Tx_*` 做线性插值 |
//! | `V/H/D45/D67/D135/D113/D157/D203` | §7.11.2.4 | 方向预测，`Dr_Intra_Derivative` 定斜率 |
//! | `filter_intra` | §7.11.2.3 | 递归 4×2 块滤波，查 `Intra_Filter_Taps` |
//!
//! ## 三处最容易写错的地方（都是"不报错、只是像素偏一点"）
//!
//! 1. **没有邻居时的填充值是**`AboveRow[i] = (1 << (BitDepth-1)) - 1` 而
//!    `LeftCol[i] = (1 << (BitDepth-1)) + 1` —— **不对称**，一个减一一个加一。
//!    顺手写成同一个值，纯 DC 块会整体偏 1。
//! 2. **边滤波与上采样只在 `enable_intra_edge_filter` 打开时做**，而且
//!    `pAngle` 正好等于 90 / 180 时要**跳过**（规范：`pAngle is not equal to 90
//!    and pAngle is not equal to 180`）。方向预测的斜率是拿滤波后的边算的，
//!    漏了或多了都会让整块偏。
//! 3. **`AboveRow[-1]`（左上角那个样本）的取值分四种情形**，而且
//!    `LeftCol[-1]` 直接等于它 —— 不是各自算一份。
//!
//! ## 验收口径
//!
//! - 单测钉每个预测器的**解析解**：例如 DC 在"全平坦的边"上必须给出那个值、
//!   无邻居必须给 `1 << (BitDepth-1)`、`V_PRED` 必须逐列复制 `AboveRow`、
//!   `H_PRED` 必须逐行复制 `LeftCol`、`pAngle == 90/180` 的退化情形。
//! - 端到端钉在 `tools/av1probe/samples/` 上：那批流的**源内容是已知的**
//!   （`#808080` 是全 128、`#101010` 是全 16），所以"解出来的像素配不配得上
//!   生成器画的东西"是一条不依赖参考解码器的真验收。

use super::flags;
use me_core::{MediaError, Result};

use super::block::{DC_PRED, PAETH_PRED, SMOOTH_H_PRED, SMOOTH_PRED, SMOOTH_V_PRED, V_PRED};

/// `INTRA_EDGE_TAPS`：边滤波的抽头数。
pub const INTRA_EDGE_TAPS: usize = 5;
/// `INTRA_EDGE_KERNELS`：三档强度的核。
pub const INTRA_EDGE_KERNEL: [[i32; INTRA_EDGE_TAPS]; 3] =
    [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
/// `INTRA_FILTER_SCALE_BITS`：filter_intra 的缩放位移。
pub const INTRA_FILTER_SCALE_BITS: u32 = 4;
/// `ANGLE_STEP`：角度增量每档多少度。
pub const ANGLE_STEP: i32 = 3;

/// `Mode_To_Angle[INTRA_MODES]`（规范 §9.3）。
pub const MODE_TO_ANGLE: [i32; 13] = [0, 90, 180, 45, 135, 113, 157, 203, 67, 0, 0, 0, 0];

/// `Dr_Intra_Derivative[90]`（规范 §9.3）：方向预测的斜率分子（6 位小数）。
pub const DR_INTRA_DERIVATIVE: [i32; 90] = [
    0, 0, 0, 1023, 0, 0, 547, 0, 0, 372, 0, 0, 0, 0, 273, 0, 0, 215, 0, 0, 178, 0, 0, 151, 0, 0,
    132, 0, 0, 116, 0, 0, 102, 0, 0, 0, 90, 0, 0, 80, 0, 0, 71, 0, 0, 64, 0, 0, 57, 0, 0, 51, 0,
    0, 45, 0, 0, 0, 40, 0, 0, 35, 0, 0, 31, 0, 0, 27, 0, 0, 23, 0, 0, 19, 0, 0, 15, 0, 0, 0, 0,
    11, 0, 0, 7, 0, 0, 3, 0, 0,
];

/// `Sm_Weights_Tx_4x4`
pub const SM_WEIGHTS_TX_4X4: [i32; 4] = [255, 149, 85, 64];
/// `Sm_Weights_Tx_8x8`
pub const SM_WEIGHTS_TX_8X8: [i32; 8] = [255, 197, 146, 105, 73, 50, 37, 32];
/// `Sm_Weights_Tx_16x16`
pub const SM_WEIGHTS_TX_16X16: [i32; 16] = [
    255, 225, 196, 170, 145, 123, 102, 84, 68, 54, 43, 33, 26, 20, 17, 16,
];
/// `Sm_Weights_Tx_32x32`
pub const SM_WEIGHTS_TX_32X32: [i32; 32] = [
    255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111, 101, 92, 83, 74, 66, 59, 52, 45,
    39, 34, 29, 25, 21, 17, 14, 12, 10, 9, 8, 8,
];
/// `Sm_Weights_Tx_64x64`
pub const SM_WEIGHTS_TX_64X64: [i32; 64] = [
    255, 248, 240, 233, 225, 218, 210, 203, 196, 189, 182, 176, 169, 163, 156, 150, 144, 138, 133,
    127, 121, 116, 111, 106, 101, 96, 91, 86, 82, 77, 73, 69, 65, 61, 57, 54, 50, 47, 44, 41, 38,
    35, 32, 29, 27, 25, 22, 20, 18, 16, 15, 13, 12, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 4,
];

/// `Intra_Filter_Taps[INTRA_FILTER_MODES][8][7]`（规范 §9.3）。
pub const INTRA_FILTER_TAPS: [[[i32; 7]; 8]; 5] = [
    [
        [-6, 10, 0, 0, 0, 12, 0],
        [-5, 2, 10, 0, 0, 9, 0],
        [-3, 1, 1, 10, 0, 7, 0],
        [-3, 1, 1, 2, 10, 5, 0],
        [-4, 6, 0, 0, 0, 2, 12],
        [-3, 2, 6, 0, 0, 2, 9],
        [-3, 2, 2, 6, 0, 2, 7],
        [-3, 1, 2, 2, 6, 3, 5],
    ],
    [
        [-10, 16, 0, 0, 0, 10, 0],
        [-6, 0, 16, 0, 0, 6, 0],
        [-4, 0, 0, 16, 0, 4, 0],
        [-2, 0, 0, 0, 16, 2, 0],
        [-10, 16, 0, 0, 0, 0, 10],
        [-6, 0, 16, 0, 0, 0, 6],
        [-4, 0, 0, 16, 0, 0, 4],
        [-2, 0, 0, 0, 16, 0, 2],
    ],
    [
        [-8, 8, 0, 0, 0, 16, 0],
        [-8, 0, 8, 0, 0, 16, 0],
        [-8, 0, 0, 8, 0, 16, 0],
        [-8, 0, 0, 0, 8, 16, 0],
        [-4, 4, 0, 0, 0, 0, 16],
        [-4, 0, 4, 0, 0, 0, 16],
        [-4, 0, 0, 4, 0, 0, 16],
        [-4, 0, 0, 0, 4, 0, 16],
    ],
    [
        [-2, 8, 0, 0, 0, 10, 0],
        [-1, 3, 8, 0, 0, 6, 0],
        [-1, 2, 3, 8, 0, 4, 0],
        [0, 1, 2, 3, 8, 2, 0],
        [-1, 4, 0, 0, 0, 3, 10],
        [-1, 3, 4, 0, 0, 4, 6],
        [-1, 2, 3, 4, 0, 4, 4],
        [-1, 2, 2, 3, 4, 3, 3],
    ],
    [
        [-12, 14, 0, 0, 0, 14, 0],
        [-10, 0, 14, 0, 0, 12, 0],
        [-9, 0, 0, 14, 0, 11, 0],
        [-8, 0, 0, 0, 14, 10, 0],
        [-10, 12, 0, 0, 0, 0, 14],
        [-9, 1, 12, 0, 0, 0, 12],
        [-8, 0, 0, 12, 0, 1, 11],
        [-7, 0, 0, 1, 12, 1, 9],
    ],
];

/// `Round2( x, n )`（规范 §3）。
#[inline]
pub fn round2(x: i32, n: u32) -> i32 {
    if n == 0 {
        return x;
    }
    (x + (1 << (n - 1))) >> n
}

/// `Round2Signed( x, n )`：负数**向远离零的方向**取整（半个单位进位），
/// 即 `x >= 0 ? Round2(x,n) : -Round2(-x,n)`（与 libaom 的
/// `ROUND_POWER_OF_TWO_SIGNED` 一致）。
#[inline]
pub fn round2_signed(x: i32, n: u32) -> i32 {
    if x < 0 {
        -round2(-x, n)
    } else {
        round2(x, n)
    }
}

/// `Clip1( x )`。
#[inline]
pub fn clip1(x: i32, bit_depth: u8) -> u16 {
    ((1i32 << bit_depth) - 1).min(x.max(0)) as u16
}

/// 这个模式是不是方向模式（规范 `is_directional_mode`：`V_PRED..D67_PRED`）。
#[inline]
pub fn is_directional_mode(mode: u8) -> bool {
    (V_PRED..=super::block::D67_PRED).contains(&mode)
}

/// `is_smooth( row, col, plane )` 的模式部分：是不是三种 smooth 之一。
#[inline]
pub fn is_smooth_mode(mode: u8) -> bool {
    matches!(mode, SMOOTH_PRED | SMOOTH_V_PRED | SMOOTH_H_PRED)
}

/// `get_filter_type( plane )`（§7.11.2.8）的**模式部分**。
///
/// 调用方负责按规范把邻居坐标换算好（色度有 `MiRow/MiCol` 的奇偶调整），
/// 并且"邻居是帧间块"要传 `None`（规范的 `is_smooth` 在
/// `RefFrames[row][col][0] > INTRA_FRAME` 时直接 `return 0`）。
/// 本函数只做"有没有邻居 + 邻居模式是不是 smooth"的判定。
pub fn get_filter_type(above: Option<u8>, left: Option<u8>) -> u8 {
    let above_smooth = above.map(is_smooth_mode).unwrap_or(false);
    let left_smooth = left.map(is_smooth_mode).unwrap_or(false);
    u8::from(above_smooth || left_smooth)
}

/// 像素平面右侧/底部预留的边界量（**frame border**）。
///
/// AV1 里"块伸出帧"是常态而不是异常：`hasRows = (r + half) < MiRows`，而
/// `MiRows`/`MiCols` 只把帧尺寸向上取整到 8，于是最后一个块可以探出帧外
/// （1080 高时 64×64 块覆盖到像素 1088）。规范与参考解码器（libaom 的
/// `AOM_BORDER_IN_PIXELS`、dav1d 的 aligned 分配）都靠**帧缓冲的 border**
/// 吸收这些读写 —— 规范原文里 `lumaX = Min(lumaX, MaxLumaW - (1 << subX))`
/// 这类钳位只保证"落在块内"，并不保证落在帧内。
///
/// 所以这里必须给平面留出 border：**帧外样本不进输出、不进参考帧取样**（MC 一
/// 律钳在帧内），但**会被 CFL 读**（§7.11.5 的 `MaxLumaW/H` 是块自身的范围、
/// 不钳帧）—— 所以帧外的像素**我们必须真的重建**（与 dav1d / libaom 一致），
/// 不能只留个"地址合法"的空位。
///
/// 72 的依据：`MiRows = 2 * ((H + 7) >> 3)`、`MiCols` 同理；分区搜索在边缘只
/// 要求 `hasRows = (r + halfBlock4x4) < MiRows`，所以最大块（128×128）的底能到
/// `4 * MiRows + 60 ≤ H + 67` —— 旧值 64 切得不够（`H % 8 == 1` 时差 3 行）。
const PLANE_PAD: usize = 72;

/// 一个像素平面（`CurrFrame[plane]` 的对应物）。
///
/// `stride = w + PLANE_PAD`、行数 `= h + PLANE_PAD` —— 见 [`PLANE_PAD`]。
/// **`stride != w`**，凡是按行寻址的地方一律用 `stride`，不要用 `w`。
///
/// 缓冲**从 [`super::pool`] 来、回池里去**（所以 `Clone`/`Drop` 是手写的，
/// 不是 derive）：一帧 1080p 三个平面 ≈ 6 MB，每帧新开一块就是每帧一次
/// mmap + 1500 次缺页 + 内核清零 —— 采样里那 ~7% 就花在这上面。
/// 像素缓冲：**8bit 走 `u8`**（与 dav1d 同款 —— 全链路内存流量减半、8 位内核
/// 的 SIMD 通道数翻倍），10/12bit 仍是 `u16`。
///
/// 为什么不是"一律 u16"：1080p 三个平面 6.2 MB，一帧里被参考读、被滤波读写、
/// 被拷贝，全链路都是 2 字节/样本；而 8bit 的样本只值 1 字节。dav1d 的 8bit
/// 内核（`uint8_t` + `pmaddubsw`/`pavgb` 一路）正是靠这个把吞吐翻倍。
///
/// 两种存储**恒有其一**：`new` 按 `bit_depth` 选，`px8()`/`px16()` 取用
/// （取错会当场 panic，不会静默读到别的宽度）。
#[derive(Debug)]
pub enum PxBuf {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

#[derive(Debug)]
pub struct Plane {
    /// ⚠️ **私有**：所有访问一律走 `px8()`/`px16()`/`get()`/`set()` ——
    /// 8bit 与 10/12bit 的存储宽度不同，编译器要能替我们把每个站点都找出来。
    data: PxBuf,
    pub stride: usize,
    pub w: usize,
    pub h: usize,
    pub bit_depth: u8,
}

impl Plane {
    pub fn new(w: usize, h: usize, bit_depth: u8) -> Self {
        Self::new_impl(w, h, bit_depth, false)
    }

    /// 强制 u16 存储（`ME_PLANE_U8` 的桥接/诊断路径用）。
    pub fn new_u16(w: usize, h: usize, bit_depth: u8) -> Self {
        Self::new_impl(w, h, bit_depth, true)
    }

    fn new_impl(w: usize, h: usize, bit_depth: u8, force_u16: bool) -> Self {
        let mid = 1u16 << (bit_depth - 1);
        let stride = w + PLANE_PAD;
        let rows = h + PLANE_PAD;
        let n = stride * rows;
        if bit_depth <= 8 && flags::plane_u8() && !force_u16 {
            // 8bit：u8 缓冲（池按 (宽度, 长度) 复用，见 `pool`）。
            let mut data = match super::pool::take_u8(n) {
                Some(v) => v,
                None => vec![0u8; n],
            };
            Self::fill_padding_u8(&mut data, stride, w, h, rows, mid as u8);
            Self { data: PxBuf::U8(data), stride, w, h, bit_depth }
        } else {
            // 10/12bit：保持 u16。
            let mut data = match super::pool::take(n) {
                Some(v) => v,
                None => vec![0u16; n],
            };
            Self::fill_padding(&mut data, stride, w, h, rows, mid);
            Self { data: PxBuf::U16(data), stride, w, h, bit_depth }
        }
    }

    /// 8bit 平面的样本缓冲（`bit_depth > 8` 时 panic）。
    #[inline]
    pub fn px8(&self) -> &[u8] {
        match &self.data {
            PxBuf::U8(v) => v,
            PxBuf::U16(_) => panic!("px8()：这个平面是 {}bit（u16 存储）", self.bit_depth),
        }
    }

    #[inline]
    pub fn px8_mut(&mut self) -> &mut [u8] {
        match &mut self.data {
            PxBuf::U8(v) => v,
            PxBuf::U16(_) => panic!("px8_mut()：这个平面是 {}bit（u16 存储）", self.bit_depth),
        }
    }

    /// 10/12bit 平面的样本缓冲（8bit 时 panic）。
    #[inline]
    pub fn px16(&self) -> &[u16] {
        match &self.data {
            PxBuf::U16(v) => v,
            PxBuf::U8(_) => panic!("px16()：这个平面是 {}bit（u8 存储）", self.bit_depth),
        }
    }

    #[inline]
    pub fn px16_mut(&mut self) -> &mut [u16] {
        match &mut self.data {
            PxBuf::U16(v) => v,
            PxBuf::U8(_) => panic!("px16_mut()：这个平面是 {}bit（u8 存储）", self.bit_depth),
        }
    }

    /// 是不是 u8 存储（8bit 帧恒为真）。
    #[inline]
    pub fn is_u8(&self) -> bool {
        matches!(self.data, PxBuf::U8(_))
    }

    /// 缓冲里的样本总数（含 padding；8bit 时按**样本**数，不是字节数）。
    #[inline]
    pub fn sample_count(&self) -> usize {
        match &self.data {
            PxBuf::U8(v) => v.len(),
            PxBuf::U16(v) => v.len(),
        }
    }

    /// 只把**边界（padding）**填成 `mid`，可见区 `w × h` **不碰**。
    ///
    /// ## 为什么可以只填边界
    ///
    /// 两个事实合起来就够：
    /// 1. **可见区每个样本都会被重建写到** —— 分区树按 4×4 铺满 `MiRows × MiCols`
    ///    （它就是帧尺寸向上取整到 8），每个叶子的预测/残差都会写自己那块，
    ///    而所有**读**邻居的地方读的都是已经解出来的样本（帧内预测的可用性、
    ///    CFL 的 `MaxLuma`、MC 的参考帧取样各自有钳位）。
    /// 2. **padding 必须真的是 `mid`** —— 帧外样本不进输出、不进参考取样，
    ///    但会被 CFL 读（`MaxLumaW/H` 是块自身的范围、只钳到块内），也会被
    ///    帧内预测当"不存在的邻居"读。原来整块填 `mid` 就是为了这个。
    ///
    /// 所以"可见区填 `mid`"这一半是**纯白写**：1080p 三个平面 ≈ 6 MB，一帧要写
    /// 二十多 MB（解码帧 + CDEF 帧 + 环路恢复输出 + 上采样……），而边界只有
    /// `72 × stride + (h) × 72` 个样本，**少一个数量级**。
    ///
    /// ⚠️ 这条论证成立的前提是"可见区确实被写满"。改分区/重建流程时若出现
    /// "某些样本可能不被任何块覆盖"，这里就要退回去整块填。
    fn fill_padding(data: &mut [u16], stride: usize, w: usize, h: usize, rows: usize, mid: u16) {
        // 右侧：可见的 `h` 行，每行 `[w, stride)`。
        for y in 0..h {
            data[y * stride + w..y * stride + stride].fill(mid);
        }
        // 底部：`[h, rows)` 整行（含右下角那一块）。
        data[h * stride..rows * stride].fill(mid);
    }

    /// [`Self::fill_padding`] 的 u8 版（同一套论证）。
    fn fill_padding_u8(data: &mut [u8], stride: usize, w: usize, h: usize, rows: usize, mid: u8) {
        for y in 0..h {
            data[y * stride + w..y * stride + stride].fill(mid);
        }
        data[h * stride..rows * stride].fill(mid);
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> u16 {
        // border 之内都合法（帧外的样本值无意义，但必须可寻址）；越出 border
        // 说明是**真的**算错了地址，而不是"块伸出帧"，所以照旧 panic。
        let i = y * self.stride + x;
        match &self.data {
            PxBuf::U8(v) => {
                debug_assert!(
                    i < v.len(),
                    "平面取址越出 frame border：(x={x}, y={y})，平面 {}×{}",
                    self.w,
                    self.h
                );
                u16::from(v[i])
            }
            PxBuf::U16(v) => {
                debug_assert!(
                    i < v.len(),
                    "平面取址越出 frame border：(x={x}, y={y})，平面 {}×{}",
                    self.w,
                    self.h
                );
                v[i]
            }
        }
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: u16) {
        let i = y * self.stride + x;
        match &mut self.data {
            PxBuf::U8(d) => {
                debug_assert!(
                    i < d.len(),
                    "平面写址越出 frame border：(x={x}, y={y})，平面 {}×{}",
                    self.w,
                    self.h
                );
                // 8bit 的样本必然 ≤ 255（调用方都过 `clip1`）；截断只是保底。
                d[i] = v.min(255) as u8;
            }
            PxBuf::U16(d) => {
                debug_assert!(
                    i < d.len(),
                    "平面写址越出 frame border：(x={x}, y={y})，平面 {}×{}",
                    self.w,
                    self.h
                );
                d[i] = v;
            }
        }
    }
}

// `Plane` 的缓冲一律从 `super::pool` 来、回池里去（不 derive `Clone` 的理由就在这）。
impl Clone for Plane {
    fn clone(&self) -> Self {
        let data = match &self.data {
            // 复用：直接 `copy_from_slice` 覆盖，**没有分配、也没有多余的填充写**
            // （新开一块要先把整块 memset 成 mid，再把源数据拷进去 —— 白写一遍）。
            PxBuf::U8(src) => PxBuf::U8(match super::pool::take_u8(src.len()) {
                Some(mut v) => {
                    v.copy_from_slice(src);
                    v
                }
                None => src.clone(),
            }),
            PxBuf::U16(src) => PxBuf::U16(match super::pool::take(src.len()) {
                Some(mut v) => {
                    v.copy_from_slice(src);
                    v
                }
                None => src.clone(),
            }),
        };
        Self {
            data,
            stride: self.stride,
            w: self.w,
            h: self.h,
            bit_depth: self.bit_depth,
        }
    }
}

impl Drop for Plane {
    fn drop(&mut self) {
        match std::mem::replace(&mut self.data, PxBuf::U8(Vec::new())) {
            PxBuf::U8(v) => super::pool::give_u8(v),
            PxBuf::U16(v) => super::pool::give(v),
        }
    }
}

/// `predict_intra` 的输入（规范里那一串变量 + 几个来自帧级状态的开关）。
#[derive(Debug, Clone)]
pub struct IntraBlock {
    pub plane: u8,
    /// 变换块左上角在**本平面**坐标里的位置
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
    pub have_left: bool,
    pub have_above: bool,
    pub have_above_right: bool,
    pub have_below_left: bool,
    /// `YMode` / `UVMode`（CFL 时调用方要传 `DC_PRED`，见规范 `residual()`）
    pub mode: u8,
    pub angle_delta: i8,
    pub use_filter_intra: bool,
    pub filter_intra_mode: u8,
    pub enable_intra_edge_filter: bool,
    /// `get_filter_type(plane)` 的结果（§7.11.2.8）
    pub filter_type: u8,
    pub max_x: usize,
    pub max_y: usize,
    pub bit_depth: u8,
}

/// 边数组的下标偏移：规范的 `AboveRow[-2]` / `LeftCol[-2]` 存在 `[0]`。
const EDGE_OFF: usize = 2;

/// `predict_intra( plane, x, y, haveLeft, haveAbove, haveAboveRight, haveBelowLeft, mode, log2W, log2H )`
/// （规范 §7.11.2.1）。
pub fn predict_intra(plane: &mut Plane, b: &IntraBlock) -> Result<()> {
    let (w, h) = (b.w, b.h);
    if w == 0 || h == 0 || !w.is_power_of_two() || !h.is_power_of_two() {
        return Err(MediaError::Bitstream(format!(
            "AV1 帧内预测的块尺寸非法：{w}×{h}（必须是 2 的幂）"
        )));
    }
    // 块**允许**伸出帧的右/下边界 —— 这是规范的正常形态，不是错误：
    // `hasRows = (r + half) < MiRows`，而 MiRows/MiCols 只把帧尺寸向上取整到 8，
    // 于是最后一行的 64×64 这类大块仍会探出帧外（1080 高时覆盖到像素 1088）。
    // 参考解码器（libaom / dav1d）**整块照算、照写**（帧缓冲带 border 承接），
    // 我们同样写进 `Plane` 的 `PLANE_PAD`：帧外样本不进输出、也不进参考帧取样
    // （MC 一律钳在帧内），但 **CFL 会读它**（§7.11.5 的 `MaxLumaW/H` 是块自身
    // 的范围、不钳帧）—— 2026-10-03 证明"裁掉等价"是错的，见 `coeff.rs` 的
    // `update_max_luma` 与下面写回处的注释。
    // 这里只挡"块起点本身就在帧外"——那才是解析错了。
    if b.x >= plane.w || b.y >= plane.h {
        return Err(MediaError::Bitstream(format!(
            "AV1 帧内预测的块起点落在平面之外：({}, {})，而平面是 {}×{}",
            b.x, b.y, plane.w, plane.h
        )));
    }
    if b.bit_depth != plane.bit_depth {
        return Err(MediaError::Bitstream("AV1 帧内预测的位深不一致".into()));
    }
    // `haveLeft` / `haveAbove` 为真就必须真的存在那个邻居（规范里它们由
    // `AvailL` / `AvailU` 推出来，帧边界上必然为假）。这里显式挡一下，
    // 免得 `x - 1` 在坐标 0 上下溢 panic。
    if b.have_left && b.x == 0 {
        return Err(MediaError::Bitstream(
            "AV1 帧内预测：haveLeft 为真但块已经在最左列（x = 0）".into(),
        ));
    }
    if b.have_above && b.y == 0 {
        return Err(MediaError::Bitstream(
            "AV1 帧内预测：haveAbove 为真但块已经在最上行（y = 0）".into(),
        ));
    }

    // 边数组：`AboveRow[i]` 存 `above[i + EDGE_OFF]`。上采样会把索引推到
    // `2*numPx-2`（numPx ≤ w+h），所以按 2*(w+h) 开。
    let span = 2 * (w + h) + 8;
    let mut above = vec![0i32; span + EDGE_OFF];
    let mut left = vec![0i32; span + EDGE_OFF];
    derive_edges(plane, None, b, &mut above, &mut left);

    let mut pred = vec![0u16; w * h];

    if b.plane == 0 && b.use_filter_intra {
        // §7.11.2.3 递归帧内预测（filter intra）
        recursive_intra(&above, &left, b.filter_intra_mode, w, h, b.bit_depth, &mut pred)?;
    } else if is_directional_mode(b.mode) {
        directional(&mut above, &mut left, b, w, h, &mut pred);
    } else if is_smooth_mode(b.mode) {
        smooth(&above, &left, b.mode, w, h, &mut pred);
    } else if b.mode == DC_PRED {
        dc(&above, &left, b.have_left, b.have_above, w, h, b.bit_depth, &mut pred);
    } else if b.mode == PAETH_PRED {
        basic(&above, &left, w, h, &mut pred);
    } else {
        return Err(MediaError::Bitstream(format!(
            "AV1 帧内预测不认识模式 {}（0..12 之外；色度 CFL=13 要在调用方换成 DC_PRED）",
            b.mode
        )));
    }

    // **整块**写回（含伸出帧右/下边界的部分，落在 `Plane` 的 `PLANE_PAD` 里）。
    // 帧外的写**不能裁**：色度走 CFL 时（`predict_cfl`）要按 `MaxLumaW/H`（= 块
    // 自身的范围，规范原文不钳帧）把帧外的**重建亮度**累加进 `lumaAvg` —— 参考
    // 解码器（dav1d / libaom）都真的重建了那块像素。裁掉之后帧底色度块的
    // `lumaAvg` 会偏、把整个色度块（含可见行）拉偏 ±1（2026-10-03 定位）。
    for y in b.y..b.y + h {
        for x in b.x..b.x + w {
            plane.set(x, y, pred[(y - b.y) * w + (x - b.x)]);
        }
    }
    Ok(())
}

/// `AV1_TRACE_ROWDEP` 的记账：读到"还没重建标记"的格子。
///
/// 顺序解码里这**完全合法**（那块还没解码 ⇒ 拿到初值），并行回放时同一格却可能
/// 已被别的 worker 写过 ⇒ 读到的值随调度变 ⇒ 竞态。用来把"跨行依赖到底在哪"标出来。
fn rowdep_log(kind: &str, b: &IntraBlock, px: usize, py: usize) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 40 {
        eprintln!(
            "ROWDEP#{n} {kind} plane={} blk=({},{}) {}x{} read=({px},{py}) cell=({},{}) \
             have_l/a={}/{} ar={} bl={}",
            b.plane,
            b.x,
            b.y,
            b.w,
            b.h,
            py >> 2,
            px >> 2,
            b.have_left,
            b.have_above,
            b.have_above_right,
            b.have_below_left
        );
    }
}

/// `ME_TRACE_RACE` 的记账：读到**还没重建**的格子（真正的竞态）。
fn raceread_log(kind: &str, b: &IntraBlock, px: usize, py: usize) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 40 || n % 200000 == 0 {
        eprintln!(
            "RACEREAD#{n} {kind} plane={} blk=({},{}) {}x{} read=({px},{py}) cell=({},{}) ar={}",
            b.plane,
            b.x,
            b.y,
            b.w,
            b.h,
            py >> 2,
            px >> 2,
            b.have_above_right
        );
    }
}

/// §7.11.2.1 开头：把 `AboveRow` / `LeftCol`（含 `[-1]` 与 `[-2]`）拼出来。
///
/// `frame` 除了取平面，还被 `AV1_TRACE_ROWDEP` 用来查"这个像素所在的 4×4 格子
/// 到底重建成没有"：**顺序解码下读到未标记的格子 = 拿到初值**（合法且确定），
/// 而并行回放时同一格可能已被别的 worker 写过 ⇒ 那就是竞态。
/// trace 开关（`AV1_TRACE_ROWDEP`）：逐行记录参考行实际读到的最远列。
///
/// ⚠️ **必须缓存**（2026-10-04）：调用点在 [`derive_edges`] 里，是**逐帧内块**的路径。
fn trace_rowdep() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("AV1_TRACE_ROWDEP").is_ok())
}

/// trace 开关（`AV1_TRACE_UVR=<x>,<y>`）：只对指定色度起点的块打 UVR 预测。
///
/// ⚠️ **必须缓存**：调用点在色度帧内预测的**逐块**路径上（原先每块要读 1~3 次环境变量）。
fn trace_uvr() -> Option<&'static str> {
    static T: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    T.get_or_init(|| std::env::var("AV1_TRACE_UVR").ok()).as_deref()
}

fn derive_edges(
    plane: &Plane,
    dep: Option<&CurrFrame>,
    b: &IntraBlock,
    above: &mut [i32],
    left: &mut [i32],
) {
    let (w, h) = (b.w, b.h);
    let mid = 1i32 << (b.bit_depth - 1);
    let n = w + h;
    let dep_on = dep.is_some() && trace_rowdep();
    // `AV1_TRACE_ROWNEED=1`：记下"上一行实际读到的最远列"（本平面 4×4 单位），
    // 与 `block::precompute_dec_flags` 预计算的等待上界逐块对照。
    let trace_need = flags::trace_rowneed();
    let trace_race = flags::trace_race();
    let mut above_hi: i64 = -1;

    for i in 0..n {
        above[i + EDGE_OFF] = if !b.have_above && b.have_left {
            plane.get(b.x - 1, b.y) as i32
        } else if !b.have_above {
            mid - 1
        } else {
            let above_limit = b
                .max_x
                .min(b.x + if b.have_above_right { 2 * w } else { w } - 1);
            let (px, py) = (above_limit.min(b.x + i), b.y - 1);
            if trace_need && (px >> 2) as i64 > above_hi {
                above_hi = (px >> 2) as i64;
            }
            // 诊断（`ME_TRACE_RACE=1`）：上一行的这个 4×4 格**还没重建**就读了它。
            // 顺序回放里这**不该发生**（规范保证 `haveAbove` 为真时上面那行已重建）；
            // 并行回放里命中 = 跨行依赖漏了。
            if trace_race
                && !super::block::racemap::is_written(
                    (py >> 2) as i64,
                    (px >> 2) as i64,
                    b.plane > 0,
                )
            {
                raceread_log("above", b, px, py);
            }
            if dep_on && !dep.unwrap().decoded(b.plane, (py >> 2) as i64, (px >> 2) as i64) {
                rowdep_log("above", b, px, py);
            }
            plane.get(px, py) as i32
        };
    }
    if trace_need && b.have_above {
        eprintln!(
            "ROWD2 i={} p={} r4={} c4={} w4={} ar={} hi={above_hi}",
            super::block::CUR_BLOCK_IDX.load(std::sync::atomic::Ordering::Relaxed),
            b.plane,
            b.y >> 2,
            b.x >> 2,
            w >> 2,
            u8::from(b.have_above_right)
        );
    }
    for i in 0..n {
        left[i + EDGE_OFF] = if !b.have_left && b.have_above {
            plane.get(b.x, b.y - 1) as i32
        } else if !b.have_left {
            mid + 1
        } else {
            let left_limit = b
                .max_y
                .min(b.y + if b.have_below_left { 2 * h } else { h } - 1);
            let (px, py) = (b.x - 1, left_limit.min(b.y + i));
            if dep_on && !dep.unwrap().decoded(b.plane, (py >> 2) as i64, (px >> 2) as i64) {
                rowdep_log("left", b, px, py);
            }
            plane.get(px, py) as i32
        };
    }

    // AboveRow[-1]：左上角那个样本，分四种情形
    let corner = match (b.have_above, b.have_left) {
        (true, true) => plane.get(b.x - 1, b.y - 1) as i32,
        (true, false) => plane.get(b.x, b.y - 1) as i32,
        (false, true) => plane.get(b.x - 1, b.y) as i32,
        (false, false) => mid,
    };
    above[EDGE_OFF - 1] = corner;
    // LeftCol[-1] 直接等于 AboveRow[-1]
    left[EDGE_OFF - 1] = corner;
}

/// 取 `AboveRow[i]`
#[inline]
fn a(above: &[i32], i: i64) -> i32 {
    above[(i + EDGE_OFF as i64) as usize]
}

/// 取 `LeftCol[i]`
#[inline]
fn l(left: &[i32], i: i64) -> i32 {
    left[(i + EDGE_OFF as i64) as usize]
}

/// §7.11.2.2 基本帧内预测（`PAETH_PRED`）。
fn basic(above: &[i32], left: &[i32], w: usize, h: usize, pred: &mut [u16]) {
    for i in 0..h {
        for j in 0..w {
            let base = a(above, j as i64) + l(left, i as i64) - a(above, -1);
            let p_left = (base - l(left, i as i64)).abs();
            let p_top = (base - a(above, j as i64)).abs();
            let p_topleft = (base - a(above, -1)).abs();
            pred[i * w + j] = if p_left <= p_top && p_left <= p_topleft {
                l(left, i as i64) as u16
            } else if p_top <= p_topleft {
                a(above, j as i64) as u16
            } else {
                a(above, -1) as u16
            };
        }
    }
}

/// §7.11.2.5 `DC_PRED`。
#[allow(clippy::too_many_arguments)]
fn dc(
    above: &[i32],
    left: &[i32],
    have_left: bool,
    have_above: bool,
    w: usize,
    h: usize,
    bit_depth: u8,
    pred: &mut [u16],
) {
    let value = match (have_left, have_above) {
        (true, true) => {
            let mut sum: i32 = 0;
            for k in 0..h {
                sum += l(left, k as i64);
            }
            for k in 0..w {
                sum += a(above, k as i64);
            }
            sum += ((w + h) >> 1) as i32;
            sum / (w + h) as i32
        }
        (true, false) => {
            let mut sum: i32 = 0;
            for k in 0..h {
                sum += l(left, k as i64);
            }
            clip1((sum + ((h >> 1) as i32)) >> (h.trailing_zeros()), bit_depth) as i32
        }
        (false, true) => {
            let mut sum: i32 = 0;
            for k in 0..w {
                sum += a(above, k as i64);
            }
            clip1((sum + ((w >> 1) as i32)) >> (w.trailing_zeros()), bit_depth) as i32
        }
        (false, false) => 1i32 << (bit_depth - 1),
    };
    let v = clip1(value, bit_depth);
    pred.fill(v);
}

/// §7.11.2.6 `SMOOTH_*_PRED`。
fn smooth(above: &[i32], left: &[i32], mode: u8, w: usize, h: usize, pred: &mut [u16]) {
    let wx = sm_weights(w);
    let wy = sm_weights(h);
    for i in 0..h {
        for j in 0..w {
            let v = match mode {
                SMOOTH_PRED => {
                    wy[i] * a(above, j as i64)
                        + (256 - wy[i]) * l(left, h as i64 - 1)
                        + wx[j] * l(left, i as i64)
                        + (256 - wx[j]) * a(above, w as i64 - 1)
                }
                SMOOTH_V_PRED => wy[i] * a(above, j as i64) + (256 - wy[i]) * l(left, h as i64 - 1),
                _ => wx[j] * l(left, i as i64) + (256 - wx[j]) * a(above, w as i64 - 1),
            };
            let shift = if mode == SMOOTH_PRED { 9 } else { 8 };
            pred[i * w + j] = ((v + (1 << (shift - 1))) >> shift).clamp(0, 65535) as u16;
        }
    }
}

fn sm_weights(n: usize) -> &'static [i32] {
    match n {
        4 => &SM_WEIGHTS_TX_4X4,
        8 => &SM_WEIGHTS_TX_8X8,
        16 => &SM_WEIGHTS_TX_16X16,
        32 => &SM_WEIGHTS_TX_32X32,
        64 => &SM_WEIGHTS_TX_64X64,
        // 规范只定义了 2..6 的 log2；调用方保证只在合法尺寸上调。
        _ => &SM_WEIGHTS_TX_64X64,
    }
}

/// §7.11.2.4 方向帧内预测（含边滤波与上采样）。
fn directional(above: &mut [i32], left: &mut [i32], b: &IntraBlock, w: usize, h: usize, pred: &mut [u16]) {
    // `angle_delta` 由调用方按平面给（亮度 = AngleDeltaY、色度 = AngleDeltaUV）
    let p_angle = MODE_TO_ANGLE[b.mode as usize] + b.angle_delta as i32 * ANGLE_STEP;

    let mut upsample_above = 0usize;
    let mut upsample_left = 0usize;

    if b.enable_intra_edge_filter {
        if p_angle != 90 && p_angle != 180 {
            if (90..180).contains(&p_angle) && w + h >= 24 {
                let corner = round2(l(left, 0) * 5 + a(above, -1) * 6 + a(above, 0) * 5, 4);
                above[EDGE_OFF - 1] = corner;
                left[EDGE_OFF - 1] = corner;
            }
            let filter_type = b.filter_type as i32;
            if b.have_above {
                let strength = edge_filter_strength(w, h, filter_type, p_angle - 90);
                let num_px = (w.min(b.max_x + 1 - b.x)) + if p_angle < 90 { h } else { 0 } + 1;
                edge_filter(above, num_px, strength);
            }
            if b.have_left {
                let strength = edge_filter_strength(w, h, filter_type, p_angle - 180);
                let num_px = (h.min(b.max_y + 1 - b.y)) + if p_angle > 180 { w } else { 0 } + 1;
                edge_filter(left, num_px, strength);
            }
        }
        upsample_above = usize::from(edge_upsample(w, h, b.filter_type as i32, p_angle - 90));
        if upsample_above == 1 {
            let num_px = w + if p_angle < 90 { h } else { 0 };
            edge_upsample_apply(above, num_px, b.bit_depth);
        }
        upsample_left = usize::from(edge_upsample(w, h, b.filter_type as i32, p_angle - 180));
        if upsample_left == 1 {
            let num_px = h + if p_angle > 180 { w } else { 0 };
            edge_upsample_apply(left, num_px, b.bit_depth);
        }
    }

    let dx = if p_angle < 90 {
        DR_INTRA_DERIVATIVE[p_angle as usize]
    } else if p_angle > 90 && p_angle < 180 {
        // 规范：`pAngle > 90 && pAngle < 180` 时才有 dx。
        // **不能写成 `pAngle < 180`** —— 那样 pAngle == 90 会去取 `Dr[90]`（表长 90，越界）。
        DR_INTRA_DERIVATIVE[(180 - p_angle) as usize]
    } else {
        0
    };
    let dy = if (90..180).contains(&p_angle) {
        DR_INTRA_DERIVATIVE[(p_angle - 90) as usize]
    } else if p_angle > 180 {
        DR_INTRA_DERIVATIVE[(270 - p_angle) as usize]
    } else {
        0
    };

    if p_angle < 90 {
        let max_base_x = ((w + h - 1) << upsample_above) as i64;
        for i in 0..h {
            for j in 0..w {
                let idx = (i as i64 + 1) * dx as i64;
                let base = (idx >> (6 - upsample_above)) + ((j << upsample_above) as i64);
                if base < max_base_x {
                    let shift = ((idx << upsample_above) >> 1) & 0x1F;
                    pred[i * w + j] = round2(
                        a(above, base) * (32 - shift as i32) + a(above, base + 1) * shift as i32,
                        5,
                    ) as u16;
                } else {
                    pred[i * w + j] = a(above, max_base_x) as u16;
                }
            }
        }
    } else if (90..180).contains(&p_angle) {
        for i in 0..h {
            for j in 0..w {
                let idx = ((j as i64) << 6) - (i as i64 + 1) * dx as i64;
                let base = idx >> (6 - upsample_above);
                if base >= -((1i64) << upsample_above) {
                    let shift = ((idx << upsample_above) >> 1) & 0x1F;
                    pred[i * w + j] = round2(
                        a(above, base) * (32 - shift as i32) + a(above, base + 1) * shift as i32,
                        5,
                    ) as u16;
                } else {
                    let idx2 = ((i as i64) << 6) - (j as i64 + 1) * dy as i64;
                    let base2 = idx2 >> (6 - upsample_left);
                    let shift = ((idx2 << upsample_left) >> 1) & 0x1F;
                    pred[i * w + j] = round2(
                        l(left, base2) * (32 - shift as i32) + l(left, base2 + 1) * shift as i32,
                        5,
                    ) as u16;
                }
            }
        }
    } else if p_angle > 180 {
        let dbg = flags::trace_dir();
        if dbg {
            println!(
                "DIR plane={} xy=({},{}) mode={} ang={} pAngle={} dy={} up_l={} up_a={} LeftCol[0..8]={:?} AboveRow[0..4]={:?}",
                b.plane, b.x, b.y, b.mode, b.angle_delta, p_angle, dy, upsample_left, upsample_above,
                (0..8).map(|k| l(left, k)).collect::<Vec<_>>(),
                (0..4).map(|k| a(above, k)).collect::<Vec<_>>()
            );
        }
        for i in 0..h {
            for j in 0..w {
                let idx = (j as i64 + 1) * dy as i64;
                let base = (idx >> (6 - upsample_left)) + ((i << upsample_left) as i64);
                let shift = ((idx << upsample_left) >> 1) & 0x1F;
                pred[i * w + j] = round2(
                    l(left, base) * (32 - shift as i32) + l(left, base + 1) * shift as i32,
                    5,
                ) as u16;
            }
        }
    } else if p_angle == 90 {
        for i in 0..h {
            for j in 0..w {
                pred[i * w + j] = a(above, j as i64) as u16;
            }
        }
    } else {
        // p_angle == 180
        for i in 0..h {
            for j in 0..w {
                pred[i * w + j] = l(left, i as i64) as u16;
            }
        }
    }
}

/// §7.11.2.9 边滤波强度。
///
/// **这里刻意照抄规范的表格结构**（`blkWh <= 12` 与 `blkWh <= 16` 两支的结论一样，
/// 但规范就是写了两行）—— 合并成一支语义等价，却会让"对着原文复核"变难，
/// 所以留着并显式关掉这条 lint。
#[allow(clippy::if_same_then_else)]
fn edge_filter_strength(w: usize, h: usize, filter_type: i32, delta: i32) -> i32 {
    let d = delta.abs();
    let blk_wh = w + h;
    let mut strength = 0;
    if filter_type == 0 {
        if blk_wh <= 8 {
            if d >= 56 {
                strength = 1;
            }
        } else if blk_wh <= 12 {
            if d >= 40 {
                strength = 1;
            }
        } else if blk_wh <= 16 {
            if d >= 40 {
                strength = 1;
            }
        } else if blk_wh <= 24 {
            if d >= 8 {
                strength = 1;
            }
            if d >= 16 {
                strength = 2;
            }
            if d >= 32 {
                strength = 3;
            }
        } else if blk_wh <= 32 {
            strength = 1;
            if d >= 4 {
                strength = 2;
            }
            if d >= 32 {
                strength = 3;
            }
        } else {
            strength = 3;
        }
    } else if blk_wh <= 8 {
        if d >= 40 {
            strength = 1;
        }
        if d >= 64 {
            strength = 2;
        }
    } else if blk_wh <= 16 {
        if d >= 20 {
            strength = 1;
        }
        if d >= 48 {
            strength = 2;
        }
    } else if blk_wh <= 24 {
        if d >= 4 {
            strength = 3;
        }
    } else {
        strength = 3;
    }
    strength
}

/// §7.11.2.10 边是否要上采样。
fn edge_upsample(w: usize, h: usize, filter_type: i32, delta: i32) -> bool {
    let d = delta.abs();
    let blk_wh = w + h;
    if d == 0 || d >= 40 {
        false
    } else if filter_type == 0 {
        blk_wh <= 16
    } else {
        blk_wh <= 8
    }
}

/// §7.11.2.11 边上采样（就地改写 `buf`）。
fn edge_upsample_apply(buf: &mut [i32], num_px: usize, bit_depth: u8) {
    if num_px == 0 {
        return;
    }
    // dup[0] = buf[-1]; dup[i+2] = buf[i] for i = -1..numPx-1; dup[numPx+2] = buf[numPx-1]
    let mut dup = vec![0i32; num_px + 3];
    dup[0] = a(buf, -1);
    for i in -1..num_px as i64 {
        dup[(i + 2) as usize] = a(buf, i);
    }
    dup[num_px + 2] = a(buf, num_px as i64 - 1);

    set_a(buf, -2, dup[0]);
    for i in 0..num_px {
        let s = -dup[i] + 9 * dup[i + 1] + 9 * dup[i + 2] - dup[i + 3];
        set_a(buf, 2 * i as i64 - 1, clip1(round2(s, 4), bit_depth) as i32);
        set_a(buf, 2 * i as i64, dup[i + 2]);
    }
}

/// §7.11.2.12 边滤波。规范分 `AboveRow` / `LeftCol` 两个入口，但两边的算式完全一样
/// （`edge[i] = buf[i-1]`，结果写回 `buf[i-1]`），所以这里只留一个入口 ——
/// 传哪个数组就滤哪个。
fn edge_filter(buf: &mut [i32], sz: usize, strength: i32) {
    if strength == 0 || sz == 0 {
        return;
    }
    // `edge` 必须在**改写之前**一次取好（规范是"先派生 edge 再原地滤"）
    let edge: Vec<i32> = (0..sz).map(|i| a(buf, i as i64 - 1)).collect();
    let kernel = &INTRA_EDGE_KERNEL[(strength - 1) as usize];
    for i in 1..sz {
        let mut s = 0i32;
        for (j, tap) in kernel.iter().enumerate() {
            let k = (i as i64 - 2 + j as i64).clamp(0, sz as i64 - 1) as usize;
            s += tap * edge[k];
        }
        set_a(buf, i as i64 - 1, (s + 8) >> 4);
    }
}

/// 写 `AboveRow[i]` / `LeftCol[i]`（`i` 可以是 -1/-2）。
#[inline]
fn set_a(buf: &mut [i32], i: i64, v: i32) {
    buf[(i + EDGE_OFF as i64) as usize] = v;
}

/// §7.11.2.3 递归帧内预测（`filter_intra`）。
fn recursive_intra(
    above: &[i32],
    left: &[i32],
    filter_intra_mode: u8,
    w: usize,
    h: usize,
    bit_depth: u8,
    pred: &mut [u16],
) -> Result<()> {
    let mode = filter_intra_mode as usize;
    if mode >= INTRA_FILTER_TAPS.len() {
        return Err(MediaError::Bitstream(format!(
            "AV1 filter_intra 模式越界：{filter_intra_mode}"
        )));
    }
    let w4 = w >> 2;
    let h2 = h >> 1;
    for i2 in 0..h2 {
        for j4 in 0..w4 {
            let mut p = [0i32; 7];
            for (i, slot) in p.iter_mut().enumerate() {
                *slot = if i < 5 {
                    if i2 == 0 {
                        a(above, ((j4 << 2) + i) as i64 - 1)
                    } else if j4 == 0 && i == 0 {
                        l(left, ((i2 << 1) as i64) - 1)
                    } else {
                        pred[((i2 << 1) - 1) * w + (j4 << 2) + i - 1] as i32
                    }
                } else if j4 == 0 {
                    l(left, ((i2 << 1) + i - 5) as i64)
                } else {
                    pred[((i2 << 1) + i - 5) * w + (j4 << 2) - 1] as i32
                };
            }
            for i1 in 0..2 {
                for j1 in 0..4 {
                    let mut pr = 0i32;
                    for (i, &pv) in p.iter().enumerate() {
                        pr += INTRA_FILTER_TAPS[mode][(i1 << 2) + j1][i] * pv;
                    }
                    pred[((i2 << 1) + i1) * w + (j4 << 2) + j1] =
                        clip1(round2_signed(pr, INTRA_FILTER_SCALE_BITS), bit_depth);
                }
            }
        }
    }
    Ok(())
}

/// 一帧的像素平面 + 每个平面的「已重建」标记（规范 §5.11.1 的 `BlockDecoded`）。
///
/// ## `BlockDecoded` 的语义（照 §21945 那段的解释）
///
/// 规范里它是**超块局部**的数组，外加一圈 4×4 的边框：边框上"上边一行 / 左边一列"
/// 若落在瓦片内就初始化为 1（那些块确实已经重建了），其余为 0。
/// 它只有两个用途：算 `haveAboveRight` 与 `haveBelowLeft`。
///
/// 这里用**帧级全局**的网格实现，语义完全等价而且更好懂：
/// 一个 4×4 位置为 1 ⟺ 它**已经被本瓦片重建过**。
/// 「上边一行/左边一列」在栅格序下天然就是已重建的，「右上方还在下一个超块里」
/// 则天然是 0 —— 与规范那圈边框的初始化结果一致。
#[derive(Debug, Clone)]
pub struct CurrFrame {
    pub planes: Vec<Plane>,
    /// 每个平面的 4×4 已重建标记（按**平面**分辨率）
    decoded: Vec<Vec<u8>>,
    decoded_cols: Vec<usize>,
    pub sub_x: bool,
    pub sub_y: bool,
    pub num_planes: usize,
}

impl CurrFrame {
    pub fn new(
        width: usize,
        height: usize,
        sub_x: bool,
        sub_y: bool,
        num_planes: usize,
        bit_depth: u8,
    ) -> Self {
        Self::new_impl(width, height, sub_x, sub_y, num_planes, bit_depth, false)
    }

    /// 强制 u16 平面（`ME_PLANE_U8` 的 LR 桥接用）。
    pub fn new_u16(
        width: usize,
        height: usize,
        sub_x: bool,
        sub_y: bool,
        num_planes: usize,
        bit_depth: u8,
    ) -> Self {
        Self::new_impl(width, height, sub_x, sub_y, num_planes, bit_depth, true)
    }

    fn new_impl(
        width: usize,
        height: usize,
        sub_x: bool,
        sub_y: bool,
        num_planes: usize,
        bit_depth: u8,
        force_u16: bool,
    ) -> Self {
        let mut planes = Vec::with_capacity(num_planes);
        let mut decoded = Vec::with_capacity(num_planes);
        let mut decoded_cols = Vec::with_capacity(num_planes);
        for plane in 0..num_planes {
            let (pw, ph) = if plane == 0 {
                (width, height)
            } else {
                (
                    (width + usize::from(sub_x)) >> u32::from(sub_x),
                    (height + usize::from(sub_y)) >> u32::from(sub_y),
                )
            };
            planes.push(if force_u16 { Plane::new_u16(pw, ph, bit_depth) } else { Plane::new(pw, ph, bit_depth) });
            let cols = pw.div_ceil(4);
            decoded_cols.push(cols);
            decoded.push(vec![0u8; cols * ph.div_ceil(4)]);
        }
        Self {
            planes,
            decoded,
            decoded_cols,
            sub_x,
            sub_y,
            num_planes,
        }
    }

    /// 整帧扩宽成 u16 的副本（`ME_PLANE_U8` 的桥接/参考帧存储用）。
    /// 连 padding 一起复制，逐样本等价。
    pub fn to_u16_copy(&self) -> CurrFrame {
        let mut out = CurrFrame::new_u16(
            self.planes[0].w,
            self.planes[0].h,
            self.sub_x,
            self.sub_y,
            self.num_planes,
            self.planes[0].bit_depth,
        );
        for p in 0..self.num_planes {
            let s = &self.planes[p];
            let d = &mut out.planes[p];
            if s.is_u8() {
                for (dv, sv) in d.px16_mut().iter_mut().zip(s.px8().iter()) {
                    *dv = u16::from(*sv);
                }
            } else {
                d.px16_mut().copy_from_slice(s.px16());
            }
        }
        out
    }

    /// 整帧收窄成 u8 的副本（仅在 `ME_PLANE_U8` 打开时得到 u8 平面）。
    pub fn to_u8_copy(&self) -> CurrFrame {
        let mut out = CurrFrame::new(
            self.planes[0].w,
            self.planes[0].h,
            self.sub_x,
            self.sub_y,
            self.num_planes,
            self.planes[0].bit_depth,
        );
        for p in 0..self.num_planes {
            let s = &self.planes[p];
            let d = &mut out.planes[p];
            if d.is_u8() {
                for (dv, sv) in d.px8_mut().iter_mut().zip(s.px16().iter()) {
                    *dv = *sv as u8;
                }
            } else {
                d.px16_mut().copy_from_slice(s.px16());
            }
        }
        out
    }

    #[inline]
    pub fn plane(&self, plane: u8) -> &Plane {
        &self.planes[plane as usize]
    }

    #[inline]
    pub fn plane_mut(&mut self, plane: u8) -> &mut Plane {
        &mut self.planes[plane as usize]
    }

    /// 某个平面的采样尺寸。
    pub fn plane_dims(&self, plane: u8) -> (usize, usize) {
        if plane == 0 {
            let p = &self.planes[0];
            (p.w, p.h)
        } else {
            let p = &self.planes[plane as usize];
            (p.w, p.h)
        }
    }

    /// `BlockDecoded[ plane ][ r ][ c ]`：越界（帧外）当作 0。
    #[inline]
    pub fn decoded(&self, plane: u8, r: i64, c: i64) -> bool {
        let p = plane as usize;
        if r < 0 || c < 0 {
            return false;
        }
        let cols = self.decoded_cols[p];
        let (r, c) = (r as usize, c as usize);
        if r * cols + c >= self.decoded[p].len() || c >= cols {
            return false;
        }
        self.decoded[p][r * cols + c] != 0
    }

    /// `BlockDecoded` 网格的只读视图 `(格子, 每行格数)`。
    ///
    /// row-mt 的 `DecFlags` 预计算要用它：预计算时网格里已有的标记都是
    /// **本批工单之前**（更早的 flush / 调色板块）写下的 ⇒ 对批内所有单元可见。
    pub fn decoded_grid(&self, plane: u8) -> (&[u8], usize) {
        (&self.decoded[plane as usize], self.decoded_cols[plane as usize])
    }

    /// 把一块 4×4 区域标成"已重建"（规范 `transform_block` 结尾那两重循环）。
    pub fn mark_decoded(&mut self, plane: u8, r: i64, c: i64, w4: i32, h4: i32) {
        let p = plane as usize;
        let cols = self.decoded_cols[p];
        for i in 0..h4 {
            for j in 0..w4 {
                let (rr, cc) = (r + i as i64, c + j as i64);
                if rr < 0 || cc < 0 {
                    continue;
                }
                let (rr, cc) = (rr as usize, cc as usize);
                if cc >= cols || rr * cols + cc >= self.decoded[p].len() {
                    continue;
                }
                self.decoded[p][rr * cols + cc] = 1;
            }
        }
    }
}

/// 一个变换块的帧内预测 —— 把 §5.11.35 `transform_block` 里那串**可用性推导**
/// 与 §7.11.2 的预测连起来。
///
/// 规范原文（`predict_intra` 的调用点）：
///
/// ```text
/// haveLeft  = ( plane == 0 ? AvailL : AvailLChroma ) || x > 0
/// haveAbove = ( plane == 0 ? AvailU : AvailUChroma ) || y > 0
/// haveAboveRight = BlockDecoded[plane][ (subBlockMiRow >> subY) - 1 ]
///                                      [ (subBlockMiCol >> subX) + stepX ]
/// haveBelowLeft  = BlockDecoded[plane][ (subBlockMiRow >> subY) + stepY ]
///                                      [ (subBlockMiCol >> subX) - 1 ]
/// ```
///
/// `step_x` / `step_y` 是**本变换块**在平面 4×4 单位下的宽高（`Tx_Width[txSz] >> 2`），
/// `sub_block_mi_row/col` 是超块内的行/列号（`row & sbMask`，见规范
/// `transform_block` 开头）。
/// `have_above_right` / `have_below_left`（§7.11.2 里对 `BlockDecoded` 的两个查询）。
///
/// ⚠️ 这两个标志的本质是「那个 4×4 格子所属的单元，在本瓦片的**解码顺序**里是否更早」，
/// 而解码顺序是**划分树顺序**（`VERT_4`/`HORZ_4` 下会跨行、列优先）⇒ row-mt 的
/// 行并行回放**必须**在回放前把它们按顺序算好（`block::RowMtJobs::precompute_dec_flags`），
/// 不能依赖运行时"谁先标记"（那会随并行而变）。
#[derive(Debug, Clone, Copy, Default)]
pub struct DecFlags {
    pub above_right: bool,
    pub below_left: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn predict_intra_txb(
    frame: &mut CurrFrame,
    plane: u8,
    start_x: usize,
    start_y: usize,
    tx_sz: u8,
    sub_block_mi_row: i32,
    sub_block_mi_col: i32,
    step_x: i32,
    step_y: i32,
    x4_in_block: i32,
    y4_in_block: i32,
    avail_l: bool,
    avail_u: bool,
    mode: u8,
    angle_delta: i8,
    use_filter_intra: bool,
    filter_intra_mode: u8,
    filter_type: u8,
    enable_intra_edge_filter: bool,
    // row-mt：预先算好的 `BlockDecoded` 两个查询（`None` = 照旧读运行时网格）
    flags: Option<DecFlags>,
) -> Result<()> {
    let sub_x = i32::from(plane > 0 && frame.sub_x);
    let sub_y = i32::from(plane > 0 && frame.sub_y);
    let w = super::block::TX_WIDTH[tx_sz as usize] as usize;
    let h = super::block::TX_HEIGHT[tx_sz as usize] as usize;
    let (max_x, max_y) = frame.plane_dims(plane);

    // 规范用的是**超块内局部坐标**（BlockDecoded 是超块局部数组），而这里的网格是
    // **帧级全局**的（见 `CurrFrame::decoded` 的说明）——两者只在第一个超块里巧合相等。
    // 必须换算成全局的平面 4×4 坐标，否则跨超块时标记会串台：`haveBelowLeft` 之类
    // 判成真就会去读**还没解码**的像素（初值 128），预测出来的边带着脏数据。
    let row = (start_y << sub_y) >> 2; // 全局亮度 4×4 行（MI_SIZE_LOG2 = 2）
    let col = (start_x << sub_x) >> 2; // 全局亮度 4×4 列
    let _ = (sub_block_mi_row, sub_block_mi_col);
    let (have_above_right, have_below_left) = match flags {
        Some(f) => (f.above_right, f.below_left),
        None => (
            frame.decoded(
                plane,
                (row >> sub_y) as i64 - 1,
                (col >> sub_x) as i64 + step_x as i64,
            ),
            frame.decoded(
                plane,
                (row >> sub_y) as i64 + step_y as i64,
                (col >> sub_x) as i64 - 1,
            ),
        ),
    };

    // ⚠️ 这一段**只为 `AV1_TRACE_UVR` 那条诊断打印服务**（`details_*` 的唯一去处
    // 就是下面的 `println!`）。以前是无条件算的：每个帧内变换块 2 个 `vec!`
    // （`ab`/`lf`，堆分配 + 清零）+ 2 个 `to_vec()` + 一整遍 `derive_edges` ——
    // 采样里光 `predict_intra_txb` 名下的 bzero 就有 ~31 个样本（2026-10-04）。
    let uvr = trace_uvr();
    let (details_above, details_left) = if uvr.is_some() {
        let (mw, mh) = (w, h);
        let span = 2 * (mw + mh) + 8;
        let mut ab = vec![0i32; span + 2];
        let mut lf = vec![0i32; span + 2];
        let tmp_ib = IntraBlock {
            plane,
            x: start_x,
            y: start_y,
            w,
            h,
            have_left: avail_l || x4_in_block > 0,
            have_above: avail_u || y4_in_block > 0,
            have_above_right,
            have_below_left,
            mode,
            angle_delta,
            use_filter_intra,
            filter_intra_mode,
            enable_intra_edge_filter,
            filter_type,
            max_x: max_x.saturating_sub(1),
            max_y: max_y.saturating_sub(1),
            bit_depth: frame.plane(plane).bit_depth,
        };
        derive_edges(frame.plane(plane), Some(frame), &tmp_ib, &mut ab, &mut lf);
        (ab[0..8].to_vec(), lf[0..8].to_vec())
    } else {
        (Vec::new(), Vec::new())
    };
    let ib = IntraBlock {
        plane,
        x: start_x,
        y: start_y,
        w,
        h,
        have_left: avail_l || x4_in_block > 0,
        have_above: avail_u || y4_in_block > 0,
        have_above_right,
        have_below_left,
        mode,
        angle_delta,
        use_filter_intra,
        filter_intra_mode,
        enable_intra_edge_filter,
        filter_type,
        max_x: max_x.saturating_sub(1),
        max_y: max_y.saturating_sub(1),
        bit_depth: frame.plane(plane).bit_depth,
    };
    if flags::no_pred() {
        return Ok(());
    }
    if let Some(spec) = uvr {
        if plane > 0 {
            let parts: Vec<i32> = spec.split(',').filter_map(|v| v.trim().parse().ok()).collect();
            if parts.len() == 2 && start_x as i32 == parts[0] && start_y as i32 == parts[1] {
                println!(
                    "UVR plane={plane} start=({start_x},{start_y}) tx={tx_sz} mode={mode} ang={angle_delta} \
                     have_l/a={avail_l}/{avail_u} (block x4/y4={x4_in_block},{y4_in_block}) filt={filter_type} \
                     above[0..6]={:?} left[0..6]={:?}",
                    &details_above, &details_left
                );
            }
        }
    }
    let dst = &mut frame.planes[plane as usize];
    predict_intra(dst, &ib)?;
    if let Some(spec) = uvr.filter(|_| plane > 0) {
        let parts: Vec<i32> = spec.split(',').filter_map(|v| v.trim().parse().ok()).collect();
        if parts.len() == 2 && start_x as i32 == parts[0] && start_y as i32 == parts[1] {
            let p = frame.plane(plane);
            let mut rows = Vec::new();
            for i in 0..4 { rows.push((0..4).map(|j| p.get(start_x as usize + j, start_y as usize + i)).collect::<Vec<_>>()); }
            println!("UVR 预测结果: {rows:?}");
        }
    }
    Ok(())
}

/// `predict_palette( plane, startX, startY, x, y, txSz )`（规范 §7.11.4 的
/// "Palette prediction process"）。
///
/// 与别的预测都不一样：它**不看邻居像素**，只拿 `palette_tokens` 解出来的索引图
/// 去查颜色表 —— `CurrFrame[plane][startY+i][startX+j] = palette[map[...]]`。
///
/// 参数口径：
/// - `x` / `y` 是这个变换块在**它所属预测块内**的 4×4 偏移（与 `predict_intra_txb`
///   同一套口径），所以索引图的下标要乘 4 才是样本位置；
/// - `map_stride` 是索引图的行宽，以**该平面的样本**为单位（亮度用块宽，色度用
///   右移后的块宽）—— 推理块比变换块大时，这一层不能省。
#[allow(clippy::too_many_arguments)]
pub fn predict_palette(
    frame: &mut CurrFrame,
    plane: u8,
    start_x: usize,
    start_y: usize,
    tx_sz: u8,
    x: i32,
    y: i32,
    palette: &[u16],
    map: &[u8],
    map_stride: usize,
) -> Result<()> {
    use super::block::{TX_HEIGHT, TX_WIDTH};
    let w = TX_WIDTH[tx_sz as usize] as usize;
    let h = TX_HEIGHT[tx_sz as usize] as usize;
    for i in 0..h {
        for j in 0..w {
            let mi = (y * 4 + i as i32) as usize;
            let mj = (x * 4 + j as i32) as usize;
            let idx = map[mi * map_stride + mj] as usize;
            let v = palette
                .get(idx)
                .copied()
                .ok_or_else(|| MediaError::Bitstream("AV1 palette：索引超出颜色表".into()))?;
            frame.plane_mut(plane).set(start_x + j, start_y + i, v);
        }
    }
    Ok(())
}

/// `predict_chroma_from_luma( plane, startX, startY, txSz )`（规范 §7.11.5）。
///
/// CFL 的模型：色度 = DC 预测 + α × (重建亮度 − 亮度均值)。所以调用方**必须先**
/// 做一次 `DC_PRED` 的帧内预测（规范 `transform_block` 里就是把 mode 换成
/// `DC_PRED` 再调 `predict_intra`，然后才调本函数）。
///
/// 三个细节：
/// - `L` 是 3 位小数的**下采样**亮度（4:2:0 取 2×2 平均再 ×2），
///   `lumaAvg` 是它在整个变换块上的平均（`Round2(…, log2W + log2H)`）；
/// - `lumaX/lumaY` 要按 `MaxLumaW/MaxLumaH` 钳位 —— 那是"当前块已重建的
///   亮度范围"，块在帧边缘被裁时防止读到还没重建的样本；
/// - 最后一步是 `Clip1(dc + Round2Signed(α × (L − lumaAvg), 6))`，**不改
///   luma**。
#[allow(clippy::too_many_arguments)]
pub fn predict_cfl(
    frame: &mut CurrFrame,
    plane: u8,
    start_x: usize,
    start_y: usize,
    tx_sz: u8,
    alpha: i32,
    max_luma_w: i32,
    max_luma_h: i32,
) -> Result<()> {
    use super::block::{TX_HEIGHT, TX_HEIGHT_LOG2, TX_WIDTH, TX_WIDTH_LOG2};
    let sub_x = i32::from(frame.sub_x);
    let sub_y = i32::from(frame.sub_y);
    let w = TX_WIDTH[tx_sz as usize] as usize;
    let h = TX_HEIGHT[tx_sz as usize] as usize;
    let bit_depth = frame.planes[plane as usize].bit_depth;

    let luma = &frame.planes[0];
    let mut l = vec![0i32; w * h];
    let mut luma_avg: i64 = 0;
    for i in 0..h {
        let mut luma_y = ((start_y + i) as i32) << sub_y;
        luma_y = luma_y.min(max_luma_h - (1 << sub_y));
        if luma_y < 0 {
            return Err(MediaError::Bitstream("CFL：亮度行钳位到了负数（MaxLumaH 没设对）".into()));
        }
        for j in 0..w {
            let mut luma_x = ((start_x + j) as i32) << sub_x;
            luma_x = luma_x.min(max_luma_w - (1 << sub_x));
            if luma_x < 0 {
                return Err(MediaError::Bitstream(
                    "CFL：亮度列钳位到了负数（MaxLumaW 没设对）".into(),
                ));
            }
            let mut t = 0i32;
            for dy in 0..=sub_y {
                for dx in 0..=sub_x {
                    t += luma.get((luma_x + dx) as usize, (luma_y + dy) as usize) as i32;
                }
            }
            let v = t << (3 - sub_x - sub_y);
            l[i * w + j] = v;
            luma_avg += i64::from(v);
        }
    }
    let n = TX_WIDTH_LOG2[tx_sz as usize] as i32 + TX_HEIGHT_LOG2[tx_sz as usize] as i32;
    let luma_avg = super::mv::round2_i64(luma_avg, n) as i32;

    if flags::trace_cfl() {
        println!(
            "CFL f={} plane={plane} start=({start_x},{start_y}) tx={tx_sz} alpha={alpha} lumaAvg={luma_avg} MaxLumaW/H=({max_luma_w},{max_luma_h}) L00={} L01={} L10={}",
            super::block::TRACE_FRAME_NO.load(std::sync::atomic::Ordering::Relaxed),
            l[0], l[1.min(w * h - 1)], l[w.min(w * h - 1)]
        );
    }
    let dst = &mut frame.planes[plane as usize];
    let skip = flags::no_cfl();
    let dbg_at = flags::trace_cfl_at();
    for i in 0..h {
        for j in 0..w {
            let dc = dst.get(start_x + j, start_y + i) as i32;
            if skip {
                continue;
            }
            let scaled = round2_signed(alpha * (l[i * w + j] - luma_avg), 6);
            let out = clip1(dc + scaled, bit_depth);
            if let Some((mr, mc)) = dbg_at {
                if ((start_y << sub_y) as i32) / 4 == mr && ((start_x << sub_x) as i32) / 4 == mc {
                    println!(
                        "CFLD me plane={plane} i={i} j={j} dc={dc} L={} avg={luma_avg} alpha={alpha} out={out}",
                        l[i * w + j]
                    );
                }
            }
            dst.set(start_x + j, start_y + i, out);
        }
    }
    Ok(())
}

#[cfg(test)]
mod currframe_tests {
    use super::*;

    #[test]
    fn frame_planes_follow_subsampling() {
        let f = CurrFrame::new(32, 16, true, true, 3, 8);
        assert_eq!(f.plane_dims(0), (32, 16));
        assert_eq!(f.plane_dims(1), (16, 8));
        assert_eq!(f.plane_dims(2), (16, 8));
        let f2 = CurrFrame::new(32, 16, false, false, 1, 10);
        assert_eq!(f2.plane_dims(0), (32, 16));
        assert_eq!(f2.num_planes, 1);
        assert_eq!(f2.plane(0).bit_depth, 10);
    }

    #[test]
    fn decoded_flags_start_empty_and_can_be_marked() {
        let mut f = CurrFrame::new(64, 64, true, true, 3, 8);
        assert!(!f.decoded(0, 0, 0));
        f.mark_decoded(0, 0, 0, 2, 2);
        assert!(f.decoded(0, 0, 0));
        assert!(f.decoded(0, 1, 1));
        assert!(!f.decoded(0, 0, 2), "只标了 2×2 个 4×4");
        assert!(!f.decoded(0, -1, 0), "帧外一律 0");
        assert!(!f.decoded(0, 0, 999), "越界一律 0");
        assert!(!f.decoded(1, 0, 0), "平面之间互不影响");
    }

    #[test]
    fn txb_prediction_writes_a_flat_block_when_there_are_no_neighbours() {
        // 帧左上角的 4×4，四个可用性都是假 ⇒ DC 取 1<<(bd-1) = 128
        let mut f = CurrFrame::new(32, 32, true, true, 3, 8);
        predict_intra_txb(
            &mut f, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, false, false, DC_PRED, 0, false, 0, 0, false,
            None,
        )
        .unwrap();
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(f.plane(0).get(j, i), 128);
            }
        }
    }

    #[test]
    fn txb_prediction_sees_the_block_above_it() {
        let mut f = CurrFrame::new(32, 32, true, true, 3, 8);
        // 先把上一行 4×4 标成"已重建"并填上 200（下面的 txb 从 y=4 开始 ⇒ 上一行是 y=3）
        f.mark_decoded(0, 0, 0, 1, 1);
        for j in 0..4 {
            f.plane_mut(0).set(j, 3, 200);
        }
        // 下面这个 txb：AvailU = true（上方 4×4 已重建）⇒ V_PRED 复制上边
        predict_intra_txb(
            &mut f, 0, 0, 4, 0, 1, 0, 1, 1, 0, 0, false, true, V_PRED, 0, false, 0, 0, false,
            None,
        )
        .unwrap();
        for i in 4..8 {
            for j in 0..4 {
                assert_eq!(f.plane(0).get(j, i), 200, "V_PRED 应当复制上一行");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::block::{
        D135_PRED, D157_PRED, D203_PRED, D45_PRED, D67_PRED, H_PRED, SMOOTH_H_PRED, SMOOTH_V_PRED,
    };
    use super::*;

    /// 一个块：左上角在 (1, 1)，上边与左边都由测试数据铺好。
    fn blk(mode: u8, w: usize, h: usize) -> IntraBlock {
        IntraBlock {
            plane: 0,
            x: 1,
            y: 1,
            w,
            h,
            have_left: true,
            have_above: true,
            have_above_right: true,
            have_below_left: true,
            mode,
            angle_delta: 0,
            use_filter_intra: false,
            filter_intra_mode: 0,
            enable_intra_edge_filter: false,
            filter_type: 0,
            max_x: 4096,
            max_y: 4096,
            bit_depth: 8,
        }
    }

    /// 铺一个"边齐备"的平面：第 0 行 = `AboveRow[0..]`、第 0 列 = `LeftCol[0..]`、
    /// (0,0) = `AboveRow[-1]`。块从 (1,1) 开始，所以 `AboveRow[j] == above[j]`、
    /// `LeftCol[i] == left[i]` 直接成立。
    ///
    /// `have_above_right`/`have_below_left` 为真时规范会读到 `2*w`/`2*h` 那么远，
    /// 所以 `above` 至少要 `2*w` 个、`left` 至少要 `2*h` 个。
    fn plane_with_edges(w: usize, h: usize, above: &[u16], left: &[u16]) -> Plane {
        plane_with_corner(w, h, above, left, above[0])
    }

    /// 同上，但单独指定 `AboveRow[-1]`（左上角那个样本）—— 它按规范来自
    /// `CurrFrame[y-1][x-1]`，与 `AboveRow[0]` 不是同一个源。
    fn plane_with_corner(w: usize, h: usize, above: &[u16], left: &[u16], corner: u16) -> Plane {
        let mut p = Plane::new(1 + 2 * w, 1 + 2 * h, 8);
        for (j, v) in above.iter().enumerate().take(2 * w) {
            p.set(1 + j, 0, *v);
        }
        for (i, v) in left.iter().enumerate().take(2 * h) {
            p.set(0, 1 + i, *v);
        }
        p.set(0, 0, corner);
        p
    }

    fn ramp(n: usize, base: u16, step: u16) -> Vec<u16> {
        (0..n).map(|i| base + i as u16 * step).collect()
    }

    #[test]
    fn dc_without_any_neighbour_is_midpoint() {
        let mut p = Plane::new(8, 8, 8);
        let mut b = blk(DC_PRED, 4, 4);
        b.x = 0;
        b.y = 0;
        b.have_left = false;
        b.have_above = false;
        predict_intra(&mut p, &b).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(p.get(j, i), 128, "无邻居的 DC 预测必须是 1<<(BitDepth-1)");
            }
        }
    }

    #[test]
    fn missing_neighbour_fills_are_asymmetric_as_the_spec_says() {
        // §7.11.2.1：没有上边时 `AboveRow[i] = (1 << (BitDepth-1)) - 1`、
        // 没有左边时 `LeftCol[i] = (1 << (BitDepth-1)) + 1` —— **一个减一一个加一**。
        // 顺手写成对称的同一个值，只有"两边都缺"的块会整体偏 1 —— V/H 会把填充值
        // 原样复制出去，所以拿它们一眼就能看出来。
        let mut p = Plane::new(8, 8, 8);
        let mut b = blk(V_PRED, 4, 4);
        b.x = 0;
        b.y = 0;
        b.have_left = false;
        b.have_above = false;
        predict_intra(&mut p, &b).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(p.get(j, i), 127, "缺上边时 AboveRow 的填充值是 (1<<(bd-1))-1");
            }
        }

        let mut q = Plane::new(8, 8, 8);
        let mut b2 = blk(H_PRED, 4, 4);
        b2.x = 0;
        b2.y = 0;
        b2.have_left = false;
        b2.have_above = false;
        predict_intra(&mut q, &b2).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(q.get(j, i), 129, "缺左边时 LeftCol 的填充值是 (1<<(bd-1))+1");
            }
        }
    }

    #[test]
    fn a_missing_side_replicates_the_available_one() {
        // 只有左边、没有上边（块贴着帧顶）：AboveRow[i] = CurrFrame[y][x-1]
        let mut p = Plane::new(8, 8, 8);
        p.set(0, 0, 200);
        for i in 1..4 {
            p.set(0, i, 10);
        }
        let mut b = blk(V_PRED, 4, 4);
        b.x = 1;
        b.y = 0;
        b.have_above = false;
        b.have_left = true;
        predict_intra(&mut p, &b).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(p.get(1 + j, i), 200, "缺上边时 AboveRow 复制 CurrFrame[y][x-1]");
            }
        }

        // 同一份数据看 H_PRED 与 DC，验证 LeftCol 的取值与 leftAvg 的位移
        let mut p2 = Plane::new(8, 8, 8);
        p2.set(0, 0, 200);
        for i in 1..4 {
            p2.set(0, i, 10);
        }
        let mut b2 = blk(H_PRED, 4, 4);
        b2.x = 1;
        b2.y = 0;
        b2.have_above = false;
        b2.have_left = true;
        predict_intra(&mut p2, &b2).unwrap();
        assert_eq!(p2.get(1, 0), 200, "LeftCol[0]");
        for i in 1..4 {
            assert_eq!(p2.get(1, i), 10, "LeftCol[{i}]");
        }

        let mut p3 = Plane::new(8, 8, 8);
        p3.set(0, 0, 200);
        for i in 1..4 {
            p3.set(0, i, 10);
        }
        let mut b3 = blk(DC_PRED, 4, 4);
        b3.x = 1;
        b3.y = 0;
        b3.have_above = false;
        b3.have_left = true;
        predict_intra(&mut p3, &b3).unwrap();
        // leftAvg = Clip1((200 + 10*3 + (4>>1)) >> 2) = (230 + 2) >> 2 = 58
        for i in 0..4 {
            assert_eq!(p3.get(1, i), 58);
        }
    }

    #[test]
    fn claiming_a_neighbour_that_cannot_exist_is_rejected() {
        // `haveLeft = 1` 却在 x = 0：规范里不可能出现，必须报错而不是 `x - 1` 下溢 panic
        let mut p = Plane::new(8, 8, 8);
        let mut b = blk(DC_PRED, 4, 4);
        b.x = 0;
        b.have_left = true;
        assert!(predict_intra(&mut p, &b).is_err());
        let mut b2 = blk(DC_PRED, 4, 4);
        b2.y = 0;
        b2.have_above = true;
        assert!(predict_intra(&mut p, &b2).is_err());
    }

    #[test]
    fn dc_averages_the_available_edges() {
        // 上边全 10、左边全 20 ⇒ sum = 4*20 + 4*10 = 120，再加 (w+h)>>1 = 4 ⇒ 124/8 = 15
        let above = [10u16; 12];
        let left = [20u16; 12];
        let mut p = plane_with_edges(4, 4, &above, &left);
        let b = blk(DC_PRED, 4, 4);
        predict_intra(&mut p, &b).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(p.get(1 + j, 1 + i), 15);
            }
        }
    }

    #[test]
    fn dc_above_only_and_left_only_use_their_own_shift() {
        let above = [9u16; 12];
        let left = [9u16; 12];
        // 只看上边：aboveAvg = Clip1((4*9 + 2) >> 2) = 9
        let mut p = plane_with_edges(4, 4, &above, &left);
        let mut b = blk(DC_PRED, 4, 4);
        b.have_left = false;
        predict_intra(&mut p, &b).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(p.get(1 + j, 1 + i), 9);
            }
        }
        // 只看左边：leftAvg = Clip1((4*9 + 2) >> 2) = 9
        let mut p2 = plane_with_edges(4, 4, &above, &left);
        let mut b2 = blk(DC_PRED, 4, 4);
        b2.have_above = false;
        predict_intra(&mut p2, &b2).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(p2.get(1 + j, 1 + i), 9);
            }
        }
    }

    #[test]
    fn v_and_h_pred_copy_the_edges() {
        let above = ramp(12, 10, 1);
        let left = ramp(12, 90, 1);
        let mut p = plane_with_edges(4, 4, &above, &left);
        predict_intra(&mut p, &blk(V_PRED, 4, 4)).unwrap();
        for (j, expected) in above.iter().enumerate().take(4) {
            for i in 0..4 {
                assert_eq!(p.get(1 + j, 1 + i), *expected, "V_PRED 每行都复制 AboveRow");
            }
        }
        let mut p2 = plane_with_edges(4, 4, &above, &left);
        predict_intra(&mut p2, &blk(H_PRED, 4, 4)).unwrap();
        for (i, expected) in left.iter().enumerate().take(4) {
            for j in 0..4 {
                assert_eq!(p2.get(1 + j, 1 + i), *expected, "H_PRED 每列都复制 LeftCol");
            }
        }
    }

    #[test]
    fn paeth_picks_the_closest_of_three() {
        // 边全平坦 ⇒ base == 三者，第一个分支（LeftCol）胜出
        let above = [100u16; 12];
        let left = [100u16; 12];
        let mut p = plane_with_edges(4, 4, &above, &left);
        predict_intra(&mut p, &blk(PAETH_PRED, 4, 4)).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(p.get(1 + j, 1 + i), 100);
            }
        }

        // 手算一格：左=0、上=200、左上=100 ⇒ base = 200+0-100 = 100
        //   |base - left| = 100、|base - above| = 100、|base - topleft| = 0
        //   ⇒ pLeft <= pTop 成立但 pLeft <= pTopLeft 不成立；pTop <= pTopLeft 也不成立
        //   ⇒ 取 AboveRow[-1] = 100
        let above2 = [200u16; 12];
        let left2 = [0u16; 12];
        let mut p2 = plane_with_corner(4, 4, &above2, &left2, 100);
        predict_intra(&mut p2, &blk(PAETH_PRED, 4, 4)).unwrap();
        assert_eq!(p2.get(1, 1), 100, "PAETH 应当选左上角");
        assert_eq!(p2.get(1, 2), 100, "整块的三个候选值固定，所以每格都取左上角");
    }

    #[test]
    fn smooth_pred_on_flat_edges_is_flat() {
        for mode in [SMOOTH_PRED, SMOOTH_V_PRED, SMOOTH_H_PRED] {
            let above = [77u16; 12];
            let left = [77u16; 12];
            let mut p = plane_with_edges(4, 4, &above, &left);
            predict_intra(&mut p, &blk(mode, 4, 4)).unwrap();
            for i in 0..4 {
                for j in 0..4 {
                    assert_eq!(p.get(1 + j, 1 + i), 77, "模式 {mode} 在平坦边上应当保持平坦");
                }
            }
        }
    }

    #[test]
    fn smooth_v_and_h_copy_their_far_edge() {
        // SMOOTH_V 的权重全在 AboveRow：smWeights[i]=255（i=0）时
        // pred[0][j] = (255*above[j] + 1*left[h-1]) / 256
        let above = [200u16; 12];
        let left = [40u16; 12];
        let mut p = plane_with_edges(4, 4, &above, &left);
        predict_intra(&mut p, &blk(SMOOTH_V_PRED, 4, 4)).unwrap();
        // (255*200 + 1*40) = 51040，Round2(51040, 8) = (51040+128)>>8 = 199.9 → 199
        assert_eq!(p.get(1, 1), 199);
        // 最后一行权重最小（Sm_Weights_Tx_4x4[3] = 64）：
        // (64*200 + 192*40) = 12800+7680 = 20480，+128 >> 8 = 80.5 → 80
        assert_eq!(p.get(1, 4), 80);
    }

    #[test]
    fn directional_90_and_180_degenerate_to_edge_copies() {
        let above = ramp(12, 10, 10);
        let left = ramp(12, 90, 1);
        let mut p = plane_with_edges(4, 4, &above, &left);
        predict_intra(&mut p, &blk(V_PRED, 4, 4)).unwrap();
        for (j, expected) in above.iter().enumerate().take(4) {
            assert_eq!(p.get(1 + j, 1), *expected, "pAngle=90 ⇒ 复制 AboveRow");
        }
        let mut p2 = plane_with_edges(4, 4, &above, &left);
        predict_intra(&mut p2, &blk(H_PRED, 4, 4)).unwrap();
        for (i, expected) in left.iter().enumerate().take(4) {
            assert_eq!(p2.get(1, 1 + i), *expected, "pAngle=180 ⇒ 复制 LeftCol");
        }
    }

    #[test]
    fn directional_d45_slope_is_the_spec_derivative() {
        // D45_PRED ⇒ pAngle = 45 ⇒ dx = Dr_Intra_Derivative[45] = 64
        // 第一行 (i=0)：idx = 1*64 = 64；base = 64>>6 = 1 + (j<<0) = j+1；shift = (64>>1)&31 = 0
        // ⇒ pred[0][j] = AboveRow[j+1]
        let above = ramp(12, 0, 7);
        let left = ramp(12, 100, 1);
        let mut p = plane_with_edges(4, 4, &above, &left);
        predict_intra(&mut p, &blk(D45_PRED, 4, 4)).unwrap();
        for j in 0..4 {
            assert_eq!(p.get(1 + j, 1), above[j + 1], "D45 第一行应当采样 AboveRow[j+1]");
        }
    }

    #[test]
    fn all_directional_modes_stay_within_the_edge_range() {
        // 边是单调递增的 ⇒ 滤波核权重非负求和为 16、上采样带 Clip1、
        // 方向插值是凸组合 ⇒ 结果不该跑出边的取值范围。
        let above = ramp(12, 40, 3);
        let left = ramp(12, 60, 2);
        let lo = *above.iter().min().unwrap().min(left.iter().min().unwrap());
        let hi = *above.iter().max().unwrap().max(left.iter().max().unwrap());
        for mode in [D45_PRED, D135_PRED, D67_PRED, D157_PRED, D203_PRED, V_PRED, H_PRED] {
            for delta in -3i8..=3 {
                for edge_filter in [false, true] {
                    let mut p = plane_with_edges(4, 4, &above, &left);
                    let mut b = blk(mode, 4, 4);
                    b.angle_delta = delta;
                    b.enable_intra_edge_filter = edge_filter;
                    predict_intra(&mut p, &b).unwrap();
                    for i in 0..4 {
                        for j in 0..4 {
                            let v = p.get(1 + j, 1 + i);
                            assert!(
                                v >= lo && v <= hi,
                                "模式 {mode}（delta {delta}、边滤波 {edge_filter}）在 ({i},{j}) \
                                 给出 {v}，超出边的范围 {lo}..={hi}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn edge_filter_strength_and_upsample_follow_the_tables() {
        // §7.11.2.9 的表格格点（`blkWh = w + h` 决定走哪一支）
        assert_eq!(edge_filter_strength(4, 4, 0, 0), 0);
        assert_eq!(edge_filter_strength(4, 4, 0, 56), 1);
        assert_eq!(edge_filter_strength(4, 4, 0, 55), 0);
        assert_eq!(edge_filter_strength(4, 6, 0, 40), 1, "blkWh=10 走 <=12 那一支");
        assert_eq!(edge_filter_strength(8, 8, 0, 8), 0, "blkWh=16 走 <=16 那一支，要 d>=40");
        assert_eq!(edge_filter_strength(8, 8, 0, 40), 1);
        assert_eq!(edge_filter_strength(12, 12, 0, 8), 1, "blkWh=24 走 <=24 那一支");
        assert_eq!(edge_filter_strength(12, 12, 0, 16), 2);
        assert_eq!(edge_filter_strength(12, 12, 0, 32), 3);
        assert_eq!(edge_filter_strength(16, 16, 0, 0), 1, "blkWh=32 直接给 1");
        assert_eq!(edge_filter_strength(16, 16, 0, 4), 2);
        assert_eq!(edge_filter_strength(16, 16, 0, 32), 3);
        assert_eq!(edge_filter_strength(32, 32, 0, 0), 3, "blkWh>32 恒为 3");
        assert_eq!(edge_filter_strength(4, 4, 1, 40), 1);
        assert_eq!(edge_filter_strength(4, 4, 1, 64), 2);
        assert_eq!(edge_filter_strength(8, 8, 1, 20), 1, "filterType=1 的 blkWh<=16 支");
        assert_eq!(edge_filter_strength(8, 8, 1, 48), 2);
        assert_eq!(edge_filter_strength(12, 12, 1, 4), 3);
        assert_eq!(edge_filter_strength(32, 32, 1, 0), 3);
        // §7.11.2.10
        assert!(!edge_upsample(4, 4, 0, 0), "d=0 不上采样");
        assert!(edge_upsample(4, 4, 0, 20));
        assert!(!edge_upsample(4, 4, 0, 40), "d>=40 不上采样");
        assert!(edge_upsample(8, 8, 0, 20), "filterType=0 时 blkWh<=16 就上采样");
        assert!(!edge_upsample(12, 12, 0, 20), "blkWh=24 超过 16，不上采样");
        assert!(edge_upsample(4, 4, 1, 20), "filterType=1 时 blkWh<=8 才上采样");
        assert!(!edge_upsample(8, 8, 1, 20), "filterType=1 且 blkWh=16 超过 8，不上采样");
    }

    #[test]
    fn edge_upsample_keeps_the_originals_on_even_positions() {
        let mut buf = vec![0i32; 40 + EDGE_OFF];
        for i in -1..8i64 {
            set_a(&mut buf, i, 10 + i as i32 * 10);
        }
        let orig: Vec<i32> = (-1..8i64).map(|i| a(&buf, i)).collect();
        edge_upsample_apply(&mut buf, 8, 8);
        assert_eq!(a(&buf, -2), orig[0], "buf[-2] = dup[0] = buf[-1]");
        for i in 0..8 {
            assert_eq!(a(&buf, 2 * i as i64), orig[i + 1], "偶数位是原值");
        }
        // 奇数位是插值：对斜坡数据应当单调
        for i in 0..7 {
            assert!(
                a(&buf, 2 * i as i64 + 1) <= a(&buf, 2 * i as i64 + 2),
                "插值结果应当保持单调"
            );
        }
    }

    #[test]
    fn edge_filter_smooths_a_step_and_keeps_flat_data_flat() {
        // 平坦数据滤波后不变
        let mut flat = vec![0i32; 40 + EDGE_OFF];
        for i in -2..12i64 {
            set_a(&mut flat, i, 123);
        }
        edge_filter(&mut flat, 8, 3);
        for i in 0..8i64 {
            assert_eq!(a(&flat, i), 123);
        }
        // 阶跃数据滤波后跳变被摊开（不再只有一个跳点）
        let mut step = vec![0i32; 40 + EDGE_OFF];
        for i in -1..12i64 {
            set_a(&mut step, i, if i < 3 { 0 } else { 255 });
        }
        let before: Vec<i32> = (0..8i64).map(|i| a(&step, i)).collect();
        edge_filter(&mut step, 8, 3);
        let after: Vec<i32> = (0..8i64).map(|i| a(&step, i)).collect();
        assert_ne!(before, after, "阶跃处应当被滤动");
        assert_eq!(after[0], before[0], "第 0 位不参与滤波（规范从 i=1 开始）");
    }

    #[test]
    fn recursive_intra_runs_for_every_filter_mode() {
        let above = [50u16; 12];
        let left = [50u16; 12];
        for mode in 0..5u8 {
            let mut p = plane_with_edges(4, 4, &above, &left);
            let mut b = blk(DC_PRED, 4, 4);
            b.use_filter_intra = true;
            b.filter_intra_mode = mode;
            predict_intra(&mut p, &b).unwrap();
            // 每种 filter_intra 的抽头每行和都是 16 ⇒ 平坦边上保持平坦
            for i in 0..4 {
                for j in 0..4 {
                    assert_eq!(p.get(1 + j, 1 + i), 50, "filter_intra 模式 {mode} 应当保持平坦");
                }
            }
        }
    }

    #[test]
    fn filter_intra_taps_each_row_sums_to_sixteen() {
        for (m, taps) in INTRA_FILTER_TAPS.iter().enumerate() {
            for (r, row) in taps.iter().enumerate() {
                let sum: i32 = row.iter().sum();
                assert_eq!(sum, 16, "filter_intra 模式 {m} 第 {r} 行抽头和应当是 16（缩放位移 4）");
            }
        }
    }

    #[test]
    fn oversized_block_is_clipped_not_rejected() {
        // **块伸出帧的右/下边界是规范允许的常态**（`hasRows = (r + half) < MiRows`
        // 决定了最后一行/列的块会探出去），所以不报错：预测整块照算，
        // 只把落在帧内的样本写回。见 `predict_intra` 开头的说明。
        let mut p = Plane::new(4, 4, 8);
        predict_intra(&mut p, &blk(DC_PRED, 8, 8)).expect("块伸出帧应当被裁剪，而不是报错");

        // 真正的错是"块**起点**已经在帧外"——那说明分区/坐标算错了
        let mut q = Plane::new(4, 4, 8);
        let mut oob = blk(DC_PRED, 4, 4);
        oob.x = 4;
        assert!(predict_intra(&mut q, &oob).is_err(), "块起点在帧外必须报错");

        let mut r = Plane::new(64, 64, 10);
        assert!(predict_intra(&mut r, &blk(DC_PRED, 4, 4)).is_err(), "位深不一致必须报错");
        let mut s = Plane::new(64, 64, 8);
        assert!(predict_intra(&mut s, &blk(13, 4, 4)).is_err(), "CFL(13) 不能在 predict_intra 里直接用");
    }

    #[test]
    fn round2_and_clip1_match_the_spec() {
        assert_eq!(round2(0, 0), 0);
        assert_eq!(round2(3, 1), 2, "(3 + 1) >> 1");
        assert_eq!(round2(1, 1), 1, "(1 + 1) >> 1");
        assert_eq!(round2(24, 4), 2, "(24 + 8) >> 4");
        assert_eq!(round2_signed(-3, 1), -2, "负数远离零取整");
        assert_eq!(round2_signed(3, 1), 2);
        assert_eq!(round2_signed(0, 4), 0);
        assert_eq!(clip1(-5, 8), 0);
        assert_eq!(clip1(300, 8), 255);
        assert_eq!(clip1(300, 10), 300);
        assert_eq!(clip1(2000, 10), 1023);
    }

    #[test]
    fn tables_have_the_spec_shapes() {
        assert_eq!(DR_INTRA_DERIVATIVE.len(), 90);
        assert_eq!(DR_INTRA_DERIVATIVE[3], 1023, "Dr[3] 必须是满精度档");
        assert_eq!(DR_INTRA_DERIVATIVE[6], 547);
        assert_eq!(DR_INTRA_DERIVATIVE[9], 372);
        assert_eq!(DR_INTRA_DERIVATIVE[45], 64, "45° 那档是 64（1.0 的定点）");
        assert_eq!(DR_INTRA_DERIVATIVE[87], 3, "最陡档");
        assert_eq!(DR_INTRA_DERIVATIVE[88], 0);
        assert_eq!(MODE_TO_ANGLE, [0, 90, 180, 45, 135, 113, 157, 203, 67, 0, 0, 0, 0]);
        for m in [SMOOTH_PRED, SMOOTH_V_PRED, SMOOTH_H_PRED] {
            assert!(!is_directional_mode(m));
        }
        for m in [V_PRED, H_PRED, D45_PRED, D135_PRED, D67_PRED, D157_PRED, D203_PRED] {
            assert!(is_directional_mode(m), "模式 {m} 应当是方向模式");
        }
        assert!(!is_directional_mode(DC_PRED));
        assert!(!is_directional_mode(PAETH_PRED));
    }

    #[test]
    fn get_filter_type_only_looks_at_smooth_neighbours() {
        assert_eq!(get_filter_type(None, None), 0);
        assert_eq!(get_filter_type(Some(DC_PRED), Some(V_PRED)), 0);
        assert_eq!(get_filter_type(Some(SMOOTH_PRED), None), 1);
        assert_eq!(get_filter_type(None, Some(SMOOTH_V_PRED)), 1);
        assert_eq!(get_filter_type(Some(SMOOTH_H_PRED), Some(PAETH_PRED)), 1);
    }

    /// `lumaAvg` 必须把**伸出帧底**的那部分重建亮度也累加进去：§7.11.5 的
    /// `MaxLumaW/MaxLumaH` 是**块自身**的范围（规范原文 `startX + stepX * 4`，
    /// **不钳帧**），帧外的样本参考解码器也真的重建了。
    ///
    /// 2026-10-03 踩过：`update_max_luma` 曾把 MaxLuma 钳到帧内 ⇒ 帧底色度块的
    /// 平均亮度偏小 ⇒ **整个**色度块（连可见行一起）被拉偏 ±1，真素材上表现为
    /// 孤立的 ±1 单点、还会沿参考链跨帧传染。这条护栏钉的就是"读帧外亮度"：
    /// 可见亮度全 0、帧外（`PLANE_PAD` 里）全 255 —— 若哪天真去钳帧，最后两行
    /// 的输出会从 116 掉回 84。
    #[test]
    fn cfl_average_reads_luma_beyond_the_visible_frame() {
        let mut f = CurrFrame::new(16, 16, true, true, 3, 8);
        for y in 0..16 {
            for x in 0..16 {
                f.plane_mut(0).set(x, y, 0);
            }
        }
        for y in 16..20 {
            for x in 0..16 {
                f.plane_mut(0).set(x, y, 255);
            }
        }
        for y in 0..10 {
            for x in 0..8 {
                f.plane_mut(1).set(x, y, 100); // 占位：CFL 之前这里是 DC 预测值
            }
        }
        // 色度 4×4 变换块放在帧底 (0,6)：i=2,3 的亮度行是 16..19（可见帧只有 16 行）。
        // 亮度变换块按 8×16（行 8..23）算 ⇒ MaxLumaH = 24 > 16。
        predict_cfl(&mut f, 1, 0, 6, 0, 1, 8, 24).unwrap();
        // L：i=0,1 全 0；i=2,3 全 2040（4 样本 ×255，再左移 1）
        // ⇒ lumaAvg = Round2(8×0 + 8×2040, 4) = 1020
        // ⇒ scaled = Round2Signed(±1020, 6) = ±16 ⇒ 100∓16 / 100±16
        for i in 0..4usize {
            for j in 0..4 {
                let got = f.plane(1).get(j, 6 + i) as i32;
                let want = if i < 2 { 84 } else { 116 };
                assert_eq!(got, want, "CFL 输出 (j={j}, i={i}) 没读到帧外亮度");
            }
        }
        // 反证（护栏自证）：把 `MaxLumaH` 误钳到帧（16）时后两行读回 0 ⇒ 四条行
        // 全变回 DC=100。真钳了帧的实现在上面那条断言就会红。
        for y in 0..10 {
            for x in 0..8 {
                f.plane_mut(1).set(x, y, 100);
            }
        }
        predict_cfl(&mut f, 1, 0, 6, 0, 1, 8, 16).unwrap();
        for i in 0..4usize {
            for j in 0..4 {
                assert_eq!(f.plane(1).get(j, 6 + i), 100, "钳帧的反证样例不对");
            }
        }
    }
}
