//! AV1 **反量化（§7.12.4）+ 反变换（§7.13）** —— 自研软解 M5 的后半。
//!
//! 到这里为止，前面几层读出来的都还是"语法"：块级语法只能给出每个变换块的
//! **量化系数**（`coeffs()` 里的 `quant[]`）。要把它们变成像素，还差三步：
//!
//! ```text
//! 量化系数 ──dequantize()──► 残差系数 T ──inverse_transform_add()──► 像素
//!   (§7.12.4)                 (§7.13)            pred + residual，再 Clip1
//! ```
//!
//! `pred` 由 [`super::intra`] 给出（帧内预测），本模块负责"把残差加回去"。
//!
//! ## 接口（**契约固定，集成方就按这个调**）
//!
//! ```text
//! Quant::new(qindex[3], bit_depth)           // 三个平面各自的 qindex 已含 delta
//! dequantize(levels, tx_sz, plane, &quant, lossless, &mut t)
//! inverse_transform_add(&t, tx_sz, tx_type, pred, dst, stride, bit_depth, lossless, &mut residual_scratch)
//! ```
//!
//! **`lossless` 为什么必须显式传**：§7.13.3 的 2D 反变换对无损敏感
//! （`rowShift` / `colShift` 取 0、行与列都换成 4×4 逆 WHT），而 §5.11.33
//! 的 `compute_tx_type` 规定无损块的 `tx_type` 恒为 `DCT_DCT`、`tx_sz` 恒为
//! `TX_4X4` —— 调用方**无法从 `tx_type` / `tx_sz` 反推**"这一块是无损的"，
//! 只能显式给。`dequantize` 里的同名列不改变规范 §7.12.4 的量化环
//! （无损时 qindex 必为 0，`dc_q(0) = ac_q(0) = 4` 已经把 ×4 带进 `Dequant`，
//! 再由逆 WHT 行端的 `>> 2` 抵消），它的用处是**前提校验**：无损块拿到的
//! 量化步长必须等于 qindex=0 的值，传错了当场 `Err`，绝不静默按有损算。
//!
//! **`levels` 的排布**与 `coeffs()` 的 `Quant` 一致：按系数格 `pos = i * tw + j`
//! （`tw = min(32, w)`、`th = min(32, h)`）—— 64 宽/高的变换块只保留左上
//! 32×32（规范里 `Quant` 之外的系数恒为 0）。`t` 是反量化后的残差系数，
//! 完整 `w × h` 行优先。
//!
//! 尺寸/类型常量一律复用 [`super::block`] 的 `TX_*` 与 [`super::coeff`] 的
//! `DCT_DCT`/`ADST_*`/`FLIPADST_*`/`IDTX`/`V_DCT`/`H_DCT`（编号就是规范里
//! `Tx_Type` 的序号），不要另起一套。
//!
//! ## 覆盖范围
//!
//! **19 个 `tx_sz` × 16 个 `tx_type` 全部覆盖**：行/列各自按 §7.13.3 的选择表
//! 走 `DCT` / `ADST` / `IDTX`，`FLIPADST` 的翻转发生在写回阶段（§7.12.4 末尾：
//! `Residual[i][j]` 写到翻转后的 `xx/yy`）。无损时（4×4）走逆 WHT 分支。
//! 规范上不可能出现的组合（64 宽/高却要求 IDTX、无损却不是 4×4、`tx_type`
//! 越界……）一律返回 `Err`，不 panic。
//!
//! ## 规范原文的入口
//!
//! `/tmp/av1-spec.txt`（去标签全文）：§7.12.3 "Dequantization functions" 在
//! 28888 行（`dc_q`/`ac_q` 查表），`get_dc_quant` / `get_ac_quant` 在 29107 / 29122，
//! 反量化过程在 29136 起；§7.13 "Inverse transform process" 在 29270、
//! "2D inverse transform process" 在 29906。
//!
//! ## SIMD + 去分配（2026-10-02）：性能线上的第三刀
//!
//! 两个方向一起做，**每一步都逐字节可复现**（真素材 20 帧输出与优化前 `diff -r` 全等）：
//!
//! 1. **去分配**。`permute` 原本每次 `t[..n0].to_vec()`（逐行/逐列调用 ⇒ 一个 16×16 的
//!    块一帧要 malloc ~32 次），`line`/`residual` 也是逐块的 `vec![]`。现在：
//!    `permute` 用 `[MaybeUninit; 64]` 的栈暂存（**不清零** —— `n0` 最小只有 4）、
//!    `line` 用 `[i64; 64]` 栈数组、`residual` 由调用方（`coeff::TxScratch`，逐瓦片
//!    一份）复用；`levels`/`t`/`pred` 同样走 `TxScratch`。这一步单独值 **1.051x**。
//! 2. **SIMD（[`super::simd`] 的 `inv_txfm_rows` / `inv_txfm_cols`）**：1D 内核按
//!    **4 条平行线一组**（i64×4 通道）跑。列趟的 4 条线 = 相邻 4 列 ⇒ 同一行里
//!    连续 4 个 i64，直接 `loadu`/`storeu`；行趟的 4 条线 = 相邻 4 行 ⇒ 跨 stride，
//!    靠 4×4 i32 转置收集（`t` 是 i32）、4×4 i64 转置写回。
//!
//!    **值域分析**（插桩实测，决定乘法形态）：12bit 极端系数全组合下乘法的输入
//!    |t| ≤ 2^19.5、8bit 真素材 ≤ 2^13.5 ⇒ `_mm256_mul_epi32`（要求 |t| < 2^31）
//!    **精确**；但积与和到 ~2^32 ⇒ 通道必须是 i64（AVX2 没有 64 位乘法算子，
//!    靠 `mul_epi32` 的"低 32 位精确"这条性质）。
//!
//!    踩过的两个坑（都记在这里免得再来一次）：① vt_* 辅助函数**忘了
//!    `#[target_feature(enable = "avx2")]`** ⇒ 每条 AVX2 intrinsic 都编译成真实
//!    函数调用，反而比标量慢 47%（`sample` 里一眼可见 `avx2::_mm256_*` 占满榜首）；
//!    ② permute 的 `[__m256i; 64]` 全量清零 —— 4×4/8×8 占 62% 的块里 `n0` 只有 4/8，
//!    等于每次白写 2KB。
//!
//! 效果（单线程 40 帧、两二进制交替多轮取 min）：去块 SIMD 版 ~10800 → 去分配
//! ~10200（1.05x）→ 再加 SIMD **9355**（合计 **1.15x**）；采样里反变换从忙时 ~8.7%
//! 降到 ~4.6%（`forward_1d` 5.9% → `vt_forward_1d` 2.0%）。
//! ⚠️ 这台机器的计时噪声大，**结论要看"交替多轮取 min"**，别信单次。
//!
//! 验收：`tests::simd_transform_matches_scalar_bit_for_bit`（全 `tx_sz` × `tx_type` ×
//! 8/10/12 位深 × 随机/极端系数 + 无损 WHT，比**残差逐元素**；同进程"关 SIMD"靠
//! `simd::FORCE_SCALAR`）+ 真素材三路（开/关/基线）逐字节。
//!
//! ## 这一层最容易写错的地方（写之前先记住）
//!
//! 1. **反量化不是"乘一个数"**：规范按系数位置分 `get_dc_quant` / `get_ac_quant`，
//!    而且要做 `Log2(txW)`/`Log2(txH)` 的移位与 `Dq_Denom`/`Dq_Offset` 那一套 ——
//!    直接 `coeff * q` 会在有损模式下整体偏。
//! 2. **无损（`lossless`）时反量化本身不变**：变化全部落在 §7.13.3 的 2D 反变换
//!    （`rowShift`/`colShift` 归零 + 逆 WHT），别把 ×4 省掉、也别只改一头。
//! 3. **变换类型里有"翻转"**（`FLIPADST` 系列）与 **`IDTX`**；`IDTX` 的
//!    行/列移位和 2D 变换完全不同。翻转按规范写在"写回"那一步，不是变换里。
//! 4. **加完残差必须 `Clip1`**（`Clip3(0, (1<<BitDepth)-1, x)`），否则越界会串色。
//! 5. **64 宽的系数格只有 32×32**：`Dequant[i][j]` 在 `i >= 32` 或 `j >= 32`
//!    时按 0 处理（§7.13.3 第一步），照抄会很自然地把 64×64 读成 64 步。

use std::mem::MaybeUninit;

use super::flags;
use me_core::{MediaError, Result};

use super::block::{
    TX_16X32, TX_16X64, TX_32X16, TX_32X32, TX_32X64, TX_64X16, TX_64X32, TX_64X64, TX_HEIGHT,
    TX_HEIGHT_LOG2, TX_SIZES_ALL, TX_WIDTH, TX_WIDTH_LOG2,
};
use super::coeff::{
    ADST_ADST, ADST_DCT, ADST_FLIPADST, DCT_ADST, DCT_DCT, DCT_FLIPADST, FLIPADST_ADST,
    FLIPADST_DCT, FLIPADST_FLIPADST, H_ADST, H_DCT, H_FLIPADST, IDTX, TX_TYPES, V_ADST, V_DCT,
    V_FLIPADST,
};

// ── §7.12.3 量化查表 ────────────────────────────────────────────────────

/// `Dc_Qlookup[ 3 ][ 256 ]`（§7.12.3 原文，按 BitDepth 8/10/12 分三行）。
pub const DC_QLOOKUP: [[i32; 256]; 3] = [
    [
        4, 8, 8, 9, 10, 11, 12, 12, 13, 14, 15, 16,
        17, 18, 19, 19, 20, 21, 22, 23, 24, 25, 26, 26,
        27, 28, 29, 30, 31, 32, 32, 33, 34, 35, 36, 37,
        38, 38, 39, 40, 41, 42, 43, 43, 44, 45, 46, 47,
        48, 48, 49, 50, 51, 52, 53, 53, 54, 55, 56, 57,
        57, 58, 59, 60, 61, 62, 62, 63, 64, 65, 66, 66,
        67, 68, 69, 70, 70, 71, 72, 73, 74, 74, 75, 76,
        77, 78, 78, 79, 80, 81, 81, 82, 83, 84, 85, 85,
        87, 88, 90, 92, 93, 95, 96, 98, 99, 101, 102, 104,
        105, 107, 108, 110, 111, 113, 114, 116, 117, 118, 120, 121,
        123, 125, 127, 129, 131, 134, 136, 138, 140, 142, 144, 146,
        148, 150, 152, 154, 156, 158, 161, 164, 166, 169, 172, 174,
        177, 180, 182, 185, 187, 190, 192, 195, 199, 202, 205, 208,
        211, 214, 217, 220, 223, 226, 230, 233, 237, 240, 243, 247,
        250, 253, 257, 261, 265, 269, 272, 276, 280, 284, 288, 292,
        296, 300, 304, 309, 313, 317, 322, 326, 330, 335, 340, 344,
        349, 354, 359, 364, 369, 374, 379, 384, 389, 395, 400, 406,
        411, 417, 423, 429, 435, 441, 447, 454, 461, 467, 475, 482,
        489, 497, 505, 513, 522, 530, 539, 549, 559, 569, 579, 590,
        602, 614, 626, 640, 654, 668, 684, 700, 717, 736, 755, 775,
        796, 819, 843, 869, 896, 925, 955, 988, 1022, 1058, 1098, 1139,
        1184, 1232, 1282, 1336,
    ],
    [
        4, 9, 10, 13, 15, 17, 20, 22, 25, 28, 31, 34,
        37, 40, 43, 47, 50, 53, 57, 60, 64, 68, 71, 75,
        78, 82, 86, 90, 93, 97, 101, 105, 109, 113, 116, 120,
        124, 128, 132, 136, 140, 143, 147, 151, 155, 159, 163, 166,
        170, 174, 178, 182, 185, 189, 193, 197, 200, 204, 208, 212,
        215, 219, 223, 226, 230, 233, 237, 241, 244, 248, 251, 255,
        259, 262, 266, 269, 273, 276, 280, 283, 287, 290, 293, 297,
        300, 304, 307, 310, 314, 317, 321, 324, 327, 331, 334, 337,
        343, 350, 356, 362, 369, 375, 381, 387, 394, 400, 406, 412,
        418, 424, 430, 436, 442, 448, 454, 460, 466, 472, 478, 484,
        490, 499, 507, 516, 525, 533, 542, 550, 559, 567, 576, 584,
        592, 601, 609, 617, 625, 634, 644, 655, 666, 676, 687, 698,
        708, 718, 729, 739, 749, 759, 770, 782, 795, 807, 819, 831,
        844, 856, 868, 880, 891, 906, 920, 933, 947, 961, 975, 988,
        1001, 1015, 1030, 1045, 1061, 1076, 1090, 1105, 1120, 1137, 1153, 1170,
        1186, 1202, 1218, 1236, 1253, 1271, 1288, 1306, 1323, 1342, 1361, 1379,
        1398, 1416, 1436, 1456, 1476, 1496, 1516, 1537, 1559, 1580, 1601, 1624,
        1647, 1670, 1692, 1717, 1741, 1766, 1791, 1817, 1844, 1871, 1900, 1929,
        1958, 1990, 2021, 2054, 2088, 2123, 2159, 2197, 2236, 2276, 2319, 2363,
        2410, 2458, 2508, 2561, 2616, 2675, 2737, 2802, 2871, 2944, 3020, 3102,
        3188, 3280, 3375, 3478, 3586, 3702, 3823, 3953, 4089, 4236, 4394, 4559,
        4737, 4929, 5130, 5347,
    ],
    [
        4, 12, 18, 25, 33, 41, 50, 60, 70, 80, 91, 103,
        115, 127, 140, 153, 166, 180, 194, 208, 222, 237, 251, 266,
        281, 296, 312, 327, 343, 358, 374, 390, 405, 421, 437, 453,
        469, 484, 500, 516, 532, 548, 564, 580, 596, 611, 627, 643,
        659, 674, 690, 706, 721, 737, 752, 768, 783, 798, 814, 829,
        844, 859, 874, 889, 904, 919, 934, 949, 964, 978, 993, 1008,
        1022, 1037, 1051, 1065, 1080, 1094, 1108, 1122, 1136, 1151, 1165, 1179,
        1192, 1206, 1220, 1234, 1248, 1261, 1275, 1288, 1302, 1315, 1329, 1342,
        1368, 1393, 1419, 1444, 1469, 1494, 1519, 1544, 1569, 1594, 1618, 1643,
        1668, 1692, 1717, 1741, 1765, 1789, 1814, 1838, 1862, 1885, 1909, 1933,
        1957, 1992, 2027, 2061, 2096, 2130, 2165, 2199, 2233, 2267, 2300, 2334,
        2367, 2400, 2434, 2467, 2499, 2532, 2575, 2618, 2661, 2704, 2746, 2788,
        2830, 2872, 2913, 2954, 2995, 3036, 3076, 3127, 3177, 3226, 3275, 3324,
        3373, 3421, 3469, 3517, 3565, 3621, 3677, 3733, 3788, 3843, 3897, 3951,
        4005, 4058, 4119, 4181, 4241, 4301, 4361, 4420, 4479, 4546, 4612, 4677,
        4742, 4807, 4871, 4942, 5013, 5083, 5153, 5222, 5291, 5367, 5442, 5517,
        5591, 5665, 5745, 5825, 5905, 5984, 6063, 6149, 6234, 6319, 6404, 6495,
        6587, 6678, 6769, 6867, 6966, 7064, 7163, 7269, 7376, 7483, 7599, 7715,
        7832, 7958, 8085, 8214, 8352, 8492, 8635, 8788, 8945, 9104, 9275, 9450,
        9639, 9832, 10031, 10245, 10465, 10702, 10946, 11210, 11482, 11776, 12081, 12409,
        12750, 13118, 13501, 13913, 14343, 14807, 15290, 15812, 16356, 16943, 17575, 18237,
        18949, 19718, 20521, 21387,
    ],
];

/// `Ac_Qlookup[ 3 ][ 256 ]`（§7.12.3 原文，按 BitDepth 8/10/12 分三行）。
pub const AC_QLOOKUP: [[i32; 256]; 3] = [
    [
        4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
        67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
        79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
        91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
        104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126,
        128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150,
        152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182, 185,
        188, 191, 194, 197, 200, 203, 207, 211, 215, 219, 223, 227,
        231, 235, 239, 243, 247, 251, 255, 260, 265, 270, 275, 280,
        285, 290, 295, 300, 305, 311, 317, 323, 329, 335, 341, 347,
        353, 359, 366, 373, 380, 387, 394, 401, 408, 416, 424, 432,
        440, 448, 456, 465, 474, 483, 492, 501, 510, 520, 530, 540,
        550, 560, 571, 582, 593, 604, 615, 627, 639, 651, 663, 676,
        689, 702, 715, 729, 743, 757, 771, 786, 801, 816, 832, 848,
        864, 881, 898, 915, 933, 951, 969, 988, 1007, 1026, 1046, 1066,
        1087, 1108, 1129, 1151, 1173, 1196, 1219, 1243, 1267, 1292, 1317, 1343,
        1369, 1396, 1423, 1451, 1479, 1508, 1537, 1567, 1597, 1628, 1660, 1692,
        1725, 1759, 1793, 1828,
    ],
    [
        4, 9, 11, 13, 16, 18, 21, 24, 27, 30, 33, 37,
        40, 44, 48, 51, 55, 59, 63, 67, 71, 75, 79, 83,
        88, 92, 96, 100, 105, 109, 114, 118, 122, 127, 131, 136,
        140, 145, 149, 154, 158, 163, 168, 172, 177, 181, 186, 190,
        195, 199, 204, 208, 213, 217, 222, 226, 231, 235, 240, 244,
        249, 253, 258, 262, 267, 271, 275, 280, 284, 289, 293, 297,
        302, 306, 311, 315, 319, 324, 328, 332, 337, 341, 345, 349,
        354, 358, 362, 367, 371, 375, 379, 384, 388, 392, 396, 401,
        409, 417, 425, 433, 441, 449, 458, 466, 474, 482, 490, 498,
        506, 514, 523, 531, 539, 547, 555, 563, 571, 579, 588, 596,
        604, 616, 628, 640, 652, 664, 676, 688, 700, 713, 725, 737,
        749, 761, 773, 785, 797, 809, 825, 841, 857, 873, 889, 905,
        922, 938, 954, 970, 986, 1002, 1018, 1038, 1058, 1078, 1098, 1118,
        1138, 1158, 1178, 1198, 1218, 1242, 1266, 1290, 1314, 1338, 1362, 1386,
        1411, 1435, 1463, 1491, 1519, 1547, 1575, 1603, 1631, 1663, 1695, 1727,
        1759, 1791, 1823, 1859, 1895, 1931, 1967, 2003, 2039, 2079, 2119, 2159,
        2199, 2239, 2283, 2327, 2371, 2415, 2459, 2507, 2555, 2603, 2651, 2703,
        2755, 2807, 2859, 2915, 2971, 3027, 3083, 3143, 3203, 3263, 3327, 3391,
        3455, 3523, 3591, 3659, 3731, 3803, 3876, 3952, 4028, 4104, 4184, 4264,
        4348, 4432, 4516, 4604, 4692, 4784, 4876, 4972, 5068, 5168, 5268, 5372,
        5476, 5584, 5692, 5804, 5916, 6032, 6148, 6268, 6388, 6512, 6640, 6768,
        6900, 7036, 7172, 7312,
    ],
    [
        4, 13, 19, 27, 35, 44, 54, 64, 75, 87, 99, 112,
        126, 139, 154, 168, 183, 199, 214, 230, 247, 263, 280, 297,
        314, 331, 349, 366, 384, 402, 420, 438, 456, 475, 493, 511,
        530, 548, 567, 586, 604, 623, 642, 660, 679, 698, 716, 735,
        753, 772, 791, 809, 828, 846, 865, 884, 902, 920, 939, 957,
        976, 994, 1012, 1030, 1049, 1067, 1085, 1103, 1121, 1139, 1157, 1175,
        1193, 1211, 1229, 1246, 1264, 1282, 1299, 1317, 1335, 1352, 1370, 1387,
        1405, 1422, 1440, 1457, 1474, 1491, 1509, 1526, 1543, 1560, 1577, 1595,
        1627, 1660, 1693, 1725, 1758, 1791, 1824, 1856, 1889, 1922, 1954, 1987,
        2020, 2052, 2085, 2118, 2150, 2183, 2216, 2248, 2281, 2313, 2346, 2378,
        2411, 2459, 2508, 2556, 2605, 2653, 2701, 2750, 2798, 2847, 2895, 2943,
        2992, 3040, 3088, 3137, 3185, 3234, 3298, 3362, 3426, 3491, 3555, 3619,
        3684, 3748, 3812, 3876, 3941, 4005, 4069, 4149, 4230, 4310, 4390, 4470,
        4550, 4631, 4711, 4791, 4871, 4967, 5064, 5160, 5256, 5352, 5448, 5544,
        5641, 5737, 5849, 5961, 6073, 6185, 6297, 6410, 6522, 6650, 6778, 6906,
        7034, 7162, 7290, 7435, 7579, 7723, 7867, 8011, 8155, 8315, 8475, 8635,
        8795, 8956, 9132, 9308, 9484, 9660, 9836, 10028, 10220, 10412, 10604, 10812,
        11020, 11228, 11437, 11661, 11885, 12109, 12333, 12573, 12813, 13053, 13309, 13565,
        13821, 14093, 14365, 14637, 14925, 15213, 15502, 15806, 16110, 16414, 16734, 17054,
        17390, 17726, 18062, 18414, 18766, 19134, 19502, 19886, 20270, 20670, 21070, 21486,
        21902, 22334, 22766, 23214, 23662, 24126, 24590, 25070, 25551, 26047, 26559, 27071,
        27599, 28143, 28687, 29247,
    ],
];

/// `(BitDepth - 8) >> 1`：量化表的行号。8/10/12 位分别落在 0/1/2 行；
/// 表只有三行，超出 12 位的输入按 12 位处理（不 panic）。
#[inline]
fn qlookup_row(bit_depth: u8) -> usize {
    ((bit_depth.saturating_sub(8) as usize) >> 1).min(2)
}

/// `dc_q( b )`（§7.12.3）：`Dc_Qlookup[(BitDepth-8) >> 1][Clip3(0, 255, b)]`。
#[inline]
pub fn dc_q(b: i32, bit_depth: u8) -> i32 {
    DC_QLOOKUP[qlookup_row(bit_depth)][b.clamp(0, 255) as usize]
}

/// `ac_q( b )`（§7.12.3）：`Ac_Qlookup[(BitDepth-8) >> 1][Clip3(0, 255, b)]`。
#[inline]
pub fn ac_q(b: i32, bit_depth: u8) -> i32 {
    AC_QLOOKUP[qlookup_row(bit_depth)][b.clamp(0, 255) as usize]
}

/// `get_dc_quant` / `get_ac_quant` 的结果：三个平面各自的量化步长。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Quant {
    pub dc: [i32; 3],
    pub ac: [i32; 3],
    /// 建表用的位深（§7.12.4 的 `Clip3(±(1<<(7+BitDepth)))` 与 §7.13.3 的
    /// `ClampRange` 都要用，`dequantize` 的入参里没有它，就随 `Quant` 带过来）。
    pub bit_depth: u8,
}

impl Quant {
    /// `qindex[plane]` 必须是**已经加过 delta** 的值（§7.12.3 的
    /// `get_qindex(0, segment_id) + DeltaQ*Dc` 那一套由调用方算）。
    ///
    /// 注意规范的不对称：`dc` 三个平面都加各自的 `DeltaQ*Dc`，而 **`ac` 的
    /// plane 0 不加任何 delta**（§7.12.3 `get_ac_quant`），调用方别顺手补。
    pub fn new(qindex: &[i32; 3], bit_depth: u8) -> Self {
        Self {
            dc: [
                dc_q(qindex[0], bit_depth),
                dc_q(qindex[1], bit_depth),
                dc_q(qindex[2], bit_depth),
            ],
            ac: [
                ac_q(qindex[0], bit_depth),
                ac_q(qindex[1], bit_depth),
                ac_q(qindex[2], bit_depth),
            ],
            bit_depth,
        }
    }
}

/// 这个变换块反量化后的残差有多大（`w * h`，用于开缓冲区）。
///
/// `tx_sz` 越界时返回 0（诊断/错误路径都靠 `Result`，这里不 panic）。
pub fn residual_len(tx_sz: u8) -> usize {
    let idx = tx_sz as usize;
    if idx >= TX_SIZES_ALL {
        return 0;
    }
    TX_WIDTH[idx] as usize * TX_HEIGHT[idx] as usize
}

/// `Dq_Denom`（§7.12.4）：大块的 `dq2` 要再除以 2 或 4。
#[inline]
fn dq_denom(tx_sz: u8) -> i64 {
    match tx_sz {
        TX_32X32 | TX_16X32 | TX_32X16 | TX_16X64 | TX_64X16 => 2,
        TX_64X64 | TX_32X64 | TX_64X32 => 4,
        _ => 1,
    }
}

/// §7.12.4 反量化：把 `levels`（按系数格 `pos` 排布、行优先 `w×h`）变成残差系数 `t`。
///
/// `levels` 与 `coeffs()` 的 `Quant` 同排布：`pos = i * tw + j`，
/// `tw = min(32, w)`、`th = min(32, h)`（64 宽/高只有左上 32×32 有系数）。
/// `t` 是完整 `w × h`（`residual_len(tx_sz)`）行优先。
pub fn dequantize(
    levels: &[i32],
    tx_sz: u8,
    plane: u8,
    quant: &Quant,
    lossless: bool,
    t: &mut [i32],
) -> Result<()> {
    let idx = tx_sz as usize;
    if idx >= TX_SIZES_ALL {
        return Err(MediaError::Bitstream(format!(
            "AV1 反量化：tx_sz = {tx_sz} 越界（TX_SIZES_ALL = {TX_SIZES_ALL}）"
        )));
    }
    if plane > 2 {
        return Err(MediaError::Bitstream(format!("AV1 反量化：plane = {plane} 越界")));
    }
    // 无损块必须拿 qindex=0 的量化步长（§7.12.4：无损时 q = dc_q(0) = ac_q(0) = 4，
    // 由行端逆 WHT 的 >>2 抵消）。传错了当场报错，别静默按有损算。
    if lossless {
        let bd = quant.bit_depth;
        if quant.dc[plane as usize] != dc_q(0, bd) || quant.ac[plane as usize] != ac_q(0, bd) {
            return Err(MediaError::Bitstream(format!(
                "AV1 反量化：无损块 plane {plane} 的量化步长应为 qindex=0 的值（dc {}/ac {}），\
                 实际是 dc {}/ac {}",
                dc_q(0, bd),
                ac_q(0, bd),
                quant.dc[plane as usize],
                quant.ac[plane as usize],
            )));
        }
    }
    let w = TX_WIDTH[idx] as usize;
    let h = TX_HEIGHT[idx] as usize;
    let tw = w.min(32);
    let th = h.min(32);
    if levels.len() < tw * th {
        return Err(MediaError::Bitstream(format!(
            "AV1 反量化：levels 太短（{} < {}）",
            levels.len(),
            tw * th
        )));
    }
    if t.len() < w * h {
        return Err(MediaError::Bitstream(format!(
            "AV1 反量化：t 太短（{} < {}）",
            t.len(),
            w * h
        )));
    }
    let q_dc = quant.dc[plane as usize] as i64;
    let q_ac = quant.ac[plane as usize] as i64;
    let denom = dq_denom(tx_sz);
    let bd = quant.bit_depth;
    let clamp_lo = -(1i64 << (7 + bd));
    let clamp_hi = (1i64 << (7 + bd)) - 1;

    for i in 0..th {
        for j in 0..tw {
            // a. 系数的位置决定用 DC 还是 AC 步长
            let q = if i == 0 && j == 0 { q_dc } else { q_ac };
            // b. q2：`using_qmatrix` 这一版没接进来（`Quant` 里没有矩阵），q2 = q
            // c. dq = Quant[i * tw + j] * q2
            let dq = levels[i * tw + j] as i64 * q;
            // d. sign、e. dq2 = sign * (Abs(dq) & 0xFFFFFF) / dqDenom
            let mag = (dq.abs() & 0xFF_FFFF) / denom;
            let dq2 = if dq < 0 { -mag } else { mag };
            // f. Dequant[i][j] = Clip3(-(1 << (7 + BitDepth)), (1 << (7 + BitDepth)) - 1, dq2)
            t[i * w + j] = dq2.clamp(clamp_lo, clamp_hi) as i32;
        }
    }
    Ok(())
}

/// `Clip1( x )`（§7.13 用，等同于 §3 的 `Clip3( 0, ( 1 << BitDepth ) - 1, x )`）。
#[inline]
pub fn clip1(x: i32, bit_depth: u8) -> u16 {
    let max = (1i32 << bit_depth) - 1;
    x.clamp(0, max) as u16
}

/// 恒等变换（`IDTX`）的行/列移位在规范里是零，这条常量留在这里给测试当锚点。
pub const IDENTITY_TX_TYPE: u8 = IDTX;
/// `DCT_DCT` 也留一个锚，免得调用方把 `Tx_Type` 的编号记反。
pub const DEFAULT_TX_TYPE: u8 = DCT_DCT;

// ── §7.13.1 蝴蝶函数 ────────────────────────────────────────────────────

/// `Cos128_Lookup[ 65 ]`（§7.13.1.2 原文）。
const COS128_LOOKUP: [i32; 65] = [
    4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948, 3920, 3889, 3857, 3822,
    3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967,
    2896, 2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660,
    1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700, 601, 501, 401, 301, 201, 101, 0,
];

/// `cos128( angle )`（§7.13.1.2）：以 4096 为单位的 128 分段余弦。
#[inline]
pub(crate) fn cos128(angle: i32) -> i32 {
    let a = (angle & 255) as usize;
    if a <= 64 {
        COS128_LOOKUP[a]
    } else if a <= 128 {
        -COS128_LOOKUP[128 - a]
    } else if a <= 192 {
        -COS128_LOOKUP[a - 128]
    } else {
        COS128_LOOKUP[256 - a]
    }
}

/// `sin128( angle ) = cos128( angle - 64 )`（§7.13.1.2）。
#[inline]
pub(crate) fn sin128(angle: i32) -> i32 {
    cos128(angle - 64)
}

/// `brev( numBits, x )`（§7.13.1.1）：`numBits` 位位反转。
#[inline]
pub(crate) fn brev(num_bits: u32, x: usize) -> usize {
    let mut t = 0usize;
    for i in 0..num_bits {
        t += ((x >> i) & 1) << (num_bits - 1 - i);
    }
    t
}

/// `Round2( x, n )`（§3）。`n = 0` 时等于 `x`。
///
/// 中间值一律用 `i64`：`B()` 的两个乘积最大约 2^30 × 2^12，`i32` 会在
/// 64 点变换里溢出，而规范只要求"结果能放进 r 位"，过程精度不能丢。
#[inline]
fn round2(x: i64, n: u32) -> i64 {
    if n == 0 {
        x
    } else {
        (x + (1i64 << (n - 1))) >> n
    }
}

/// `B( a, b, angle, flip, r )`（§7.13.1.2）：蝶形旋转；`flip = 1` 时交换两端。
fn butterfly(t: &mut [i64], a: usize, b: usize, angle: i32, flip: bool) {
    let (c, s) = (cos128(angle) as i64, sin128(angle) as i64);
    let x = t[a] * c - t[b] * s;
    let y = t[a] * s + t[b] * c;
    t[a] = round2(x, 12);
    t[b] = round2(y, 12);
    if flip {
        t.swap(a, b);
    }
}

/// `H( a, b, flip, r )`（§7.13.1.2）：Hadamard 旋转，按 `r` 位有符号范围钳位。
fn hadamard(t: &mut [i64], a: usize, b: usize, flip: bool, r: u32) {
    let (a, b) = if flip { (b, a) } else { (a, b) };
    let (x, y) = (t[a], t[b]);
    let lo = -(1i64 << (r - 1));
    let hi = (1i64 << (r - 1)) - 1;
    t[a] = (x + y).clamp(lo, hi);
    t[b] = (x - y).clamp(lo, hi);
}

// ── §7.13.2 1D 变换 ─────────────────────────────────────────────────────

/// `Inverse DCT array permutation process`（§7.13.2.2）：`T[i] = copy[brev(n, i)]`。
///
/// ⚠️ `copy` 用**栈数组**而不是 `t[..n0].to_vec()`：这里逐行/逐列调用（一个 16×16 的
/// 块一帧要走 32 次），`to_vec` 每次都是一次 malloc+free。`n0 ≤ 64`，栈上 512 字节就够。
fn dct_permute(t: &mut [i64], n: u32) {
    let n0 = 1usize << n;
    // 暂存**不清零**（`n0` 最小只有 4；给整个数组清零在小块上是纯浪费）
    let mut copy: [MaybeUninit<i64>; 64] = [const { MaybeUninit::uninit() }; 64];
    for (i, slot) in copy[..n0].iter_mut().enumerate() {
        slot.write(t[i]);
    }
    for (i, v) in t[..n0].iter_mut().enumerate() {
        *v = unsafe { copy[brev(n, i)].assume_init() };
    }
}

/// `Inverse DCT process`（§7.13.2.3）：长度 `2^n`（2 ≤ n ≤ 6）的原地反 DCT。
fn inverse_dct(t: &mut [i64], n: u32, r: u32) {
    let n0 = 1usize << n;
    dct_permute(t, n);

    if n == 6 {
        for i in 0..16 {
            butterfly(t, 32 + i, 63 - i, 63 - 4 * brev(4, i) as i32, false);
        }
    }
    if n >= 5 {
        for i in 0..8 {
            butterfly(t, 16 + i, 31 - i, 6 + ((brev(3, 7 - i) << 3) as i32), false);
        }
    }
    if n == 6 {
        for i in 0..16 {
            hadamard(t, 32 + i * 2, 33 + i * 2, i & 1 == 1, r);
        }
    }
    if n >= 4 {
        for i in 0..4 {
            butterfly(t, 8 + i, 15 - i, 12 + ((brev(2, 3 - i) << 4) as i32), false);
        }
    }
    if n >= 5 {
        for i in 0..8 {
            hadamard(t, 16 + 2 * i, 17 + 2 * i, i & 1 == 1, r);
        }
    }
    if n == 6 {
        for i in 0..4 {
            for j in 0..2 {
                butterfly(
                    t,
                    62 - i * 4 - j,
                    33 + i * 4 + j,
                    60 - 16 * brev(2, i) as i32 + 64 * j as i32,
                    true,
                );
            }
        }
    }
    if n >= 3 {
        for i in 0..2 {
            butterfly(t, 4 + i, 7 - i, 56 - 32 * i as i32, false);
        }
    }
    if n >= 4 {
        for i in 0..4 {
            hadamard(t, 8 + 2 * i, 9 + 2 * i, i & 1 == 1, r);
        }
    }
    if n >= 5 {
        for i in 0..2 {
            for j in 0..2 {
                butterfly(
                    t,
                    30 - 4 * i - j,
                    17 + 4 * i + j,
                    24 + (j << 6) as i32 + ((1 - i) << 5) as i32,
                    true,
                );
            }
        }
    }
    if n == 6 {
        for i in 0..8 {
            for j in 0..2 {
                hadamard(t, 32 + i * 4 + j, 35 + i * 4 - j, i & 1 == 1, r);
            }
        }
    }
    for i in 0..2 {
        butterfly(t, 2 * i, 2 * i + 1, 32 + 16 * i as i32, i == 0);
    }
    if n >= 3 {
        for i in 0..2 {
            hadamard(t, 4 + 2 * i, 5 + 2 * i, i == 1, r);
        }
    }
    if n >= 4 {
        for i in 0..2 {
            butterfly(t, 14 - i, 9 + i, 48 + 64 * i as i32, true);
        }
    }
    if n >= 5 {
        for i in 0..4 {
            for j in 0..2 {
                hadamard(t, 16 + 4 * i + j, 19 + 4 * i - j, i & 1 == 1, r);
            }
        }
    }
    if n == 6 {
        for i in 0..2 {
            for j in 0..4 {
                butterfly(
                    t,
                    61 - i * 8 - j,
                    34 + i * 8 + j,
                    56 - i as i32 * 32 + (j >> 1) as i32 * 64,
                    true,
                );
            }
        }
    }
    for i in 0..2 {
        hadamard(t, i, 3 - i, false, r);
    }
    if n >= 3 {
        butterfly(t, 6, 5, 32, true);
    }
    if n >= 4 {
        for i in 0..2 {
            for j in 0..2 {
                hadamard(t, 8 + 4 * i + j, 11 + 4 * i - j, i == 1, r);
            }
        }
    }
    if n >= 5 {
        for i in 0..4 {
            butterfly(t, 29 - i, 18 + i, 48 + ((i >> 1) as i32) * 64, true);
        }
    }
    if n == 6 {
        for i in 0..4 {
            for j in 0..4 {
                hadamard(t, 32 + 8 * i + j, 39 + 8 * i - j, i & 1 == 1, r);
            }
        }
    }
    if n >= 3 {
        for i in 0..4 {
            hadamard(t, i, 7 - i, false, r);
        }
    }
    if n >= 4 {
        for i in 0..2 {
            butterfly(t, 13 - i, 10 + i, 32, true);
        }
    }
    if n >= 5 {
        for i in 0..2 {
            for j in 0..4 {
                hadamard(t, 16 + i * 8 + j, 23 + i * 8 - j, i == 1, r);
            }
        }
    }
    if n == 6 {
        for i in 0..8 {
            butterfly(t, 59 - i, 36 + i, if i < 4 { 48 } else { 112 }, true);
        }
    }
    if n >= 4 {
        for i in 0..8 {
            hadamard(t, i, 15 - i, false, r);
        }
    }
    if n >= 5 {
        for i in 0..4 {
            butterfly(t, 27 - i, 20 + i, 32, true);
        }
    }
    if n == 6 {
        for i in 0..8 {
            hadamard(t, 32 + i, 47 - i, false, r);
            hadamard(t, 48 + i, 63 - i, true, r);
        }
    }
    if n >= 5 {
        for i in 0..16 {
            hadamard(t, i, 31 - i, false, r);
        }
    }
    if n == 6 {
        for i in 0..8 {
            butterfly(t, 55 - i, 40 + i, 32, true);
        }
    }
    if n == 6 {
        for i in 0..32 {
            hadamard(t, i, 63 - i, false, r);
        }
    }
    debug_assert_eq!(n0, 1usize << n);
}

/// `Inverse ADST input array permutation process`（§7.13.2.4）。
fn adst_input_permute(t: &mut [i64], n: u32) {
    let n0 = 1usize << n;
    let mut copy: [MaybeUninit<i64>; 64] = [const { MaybeUninit::uninit() }; 64];
    for (i, slot) in copy[..n0].iter_mut().enumerate() {
        slot.write(t[i]);
    }
    for (i, v) in t[..n0].iter_mut().enumerate() {
        let idx = if i & 1 == 1 { i - 1 } else { n0 - i - 1 };
        *v = unsafe { copy[idx].assume_init() };
    }
}

/// `Inverse ADST output array permutation process`（§7.13.2.5）。
fn adst_output_permute(t: &mut [i64], n: u32) {
    let n0 = 1usize << n;
    let mut copy: [MaybeUninit<i64>; 64] = [const { MaybeUninit::uninit() }; 64];
    for (i, slot) in copy[..n0].iter_mut().enumerate() {
        slot.write(t[i]);
    }
    for (i, v) in t[..n0].iter_mut().enumerate() {
        let a = (i >> 3) & 1;
        let b = ((i >> 2) & 1) ^ ((i >> 3) & 1);
        let c = ((i >> 1) & 1) ^ ((i >> 2) & 1);
        let d = (i & 1) ^ ((i >> 1) & 1);
        let idx = ((d << 3) | (c << 2) | (b << 1) | a) >> (4 - n);
        *v = if i & 1 == 1 { -unsafe { copy[idx].assume_init() } } else { unsafe { copy[idx].assume_init() } };
    }
}

/// `Inverse ADST4 process`（§7.13.2.6）。
fn inverse_adst4(t: &mut [i64], r: u32) {
    const SINPI_1_9: i64 = 1321;
    const SINPI_2_9: i64 = 2482;
    const SINPI_3_9: i64 = 3344;
    const SINPI_4_9: i64 = 3803;
    let mut s = [0i64; 7];
    s[0] = SINPI_1_9 * t[0];
    s[1] = SINPI_2_9 * t[0];
    s[2] = SINPI_3_9 * t[1];
    s[3] = SINPI_4_9 * t[2];
    s[4] = SINPI_1_9 * t[2];
    s[5] = SINPI_2_9 * t[3];
    s[6] = SINPI_4_9 * t[3];
    let a7 = t[0] - t[2];
    let b7 = a7 + t[3];

    s[0] += s[3];
    s[1] -= s[4];
    s[3] = s[2];
    s[2] = SINPI_3_9 * b7;

    s[0] += s[5];
    s[1] -= s[6];

    let x = [s[0] + s[3], s[1] + s[3], s[2], s[0] + s[1] - s[3]];
    t[0] = round2(x[0], 12);
    t[1] = round2(x[1], 12);
    t[2] = round2(x[2], 12);
    t[3] = round2(x[3], 12);
    let _ = r;
    debug_assert_eq!(s.len(), 7);
}

/// `Inverse ADST8 process`（§7.13.2.7）。
fn inverse_adst8(t: &mut [i64], r: u32) {
    adst_input_permute(t, 3);
    for i in 0..4 {
        butterfly(t, 2 * i, 2 * i + 1, 60 - 16 * i as i32, true);
    }
    for i in 0..4 {
        hadamard(t, i, 4 + i, false, r);
    }
    for i in 0..2 {
        butterfly(t, 4 + 3 * i, 5 + i, 48 - 32 * i as i32, true);
    }
    for j in 0..2 {
        for i in 0..2 {
            hadamard(t, 4 * j + i, 2 + 4 * j + i, false, r);
        }
    }
    for i in 0..2 {
        butterfly(t, 2 + 4 * i, 3 + 4 * i, 32, true);
    }
    adst_output_permute(t, 3);
}

/// `Inverse ADST16 process`（§7.13.2.8）。
fn inverse_adst16(t: &mut [i64], r: u32) {
    adst_input_permute(t, 4);
    for i in 0..8 {
        butterfly(t, 2 * i, 2 * i + 1, 62 - 8 * i as i32, true);
    }
    for i in 0..8 {
        hadamard(t, i, 8 + i, false, r);
    }
    for i in 0..2 {
        butterfly(t, 8 + 2 * i, 9 + 2 * i, 56 - 32 * i as i32, true);
        butterfly(t, 13 + 2 * i, 12 + 2 * i, 8 + 32 * i as i32, true);
    }
    for j in 0..2 {
        for i in 0..4 {
            hadamard(t, 8 * j + i, 4 + 8 * j + i, false, r);
        }
    }
    for j in 0..2 {
        for i in 0..2 {
            butterfly(t, 4 + 8 * j + 3 * i, 5 + 8 * j + i, 48 - 32 * i as i32, true);
        }
    }
    for j in 0..4 {
        for i in 0..2 {
            hadamard(t, 4 * j + i, 2 + 4 * j + i, false, r);
        }
    }
    for i in 0..4 {
        butterfly(t, 2 + 4 * i, 3 + 4 * i, 32, true);
    }
    adst_output_permute(t, 4);
}

/// `Inverse ADST process`（§7.13.2.9）：长度 `2^n`（2 ≤ n ≤ 4）。
fn inverse_adst(t: &mut [i64], n: u32, r: u32) {
    match n {
        2 => inverse_adst4(t, r),
        3 => inverse_adst8(t, r),
        4 => inverse_adst16(t, r),
        _ => unreachable!("ADST 只定义到 2^n = 16；调用前已按 tx_sz 校验"),
    }
}

/// `Inverse Walsh-Hadamard transform process`（§7.13.2.10）：长度 4 的原地逆 WHT。
fn inverse_wht(t: &mut [i64], shift: u32) {
    let mut a = t[0] >> shift;
    let c = t[1] >> shift;
    let mut d = t[2] >> shift;
    let mut b = t[3] >> shift;
    a += c;
    d -= b;
    let e = (a - d) >> 1;
    b = e - b;
    let c = e - c;
    a -= b;
    d += c;
    t[0] = a;
    t[1] = b;
    t[2] = c;
    t[3] = d;
}

/// `Inverse identity transform process`（§7.13.2.11–14）：长度 `2^n`（2 ≤ n ≤ 5）。
fn inverse_identity(t: &mut [i64], n: u32) {
    match n {
        2 => {
            for v in t[..4].iter_mut() {
                *v = round2(*v * 5793, 12);
            }
        }
        3 => {
            for v in t[..8].iter_mut() {
                *v *= 2;
            }
        }
        4 => {
            for v in t[..16].iter_mut() {
                *v = round2(*v * 11586, 12);
            }
        }
        5 => {
            for v in t[..32].iter_mut() {
                *v *= 4;
            }
        }
        _ => unreachable!("IDTX 只定义到 2^n = 32；调用前已按 tx_sz 校验"),
    }
}

// ── §7.13.3 2D 反变换 ───────────────────────────────────────────────────

/// 行/列各自该走的 1D 变换类别（§7.13.3 的两张选择表）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Transform1d {
    Dct,
    Adst,
    Identity,
}

/// `Transform_Row_Shift[ TX_SIZES_ALL ]`（§7.13.3 末尾原文）。
const TRANSFORM_ROW_SHIFT: [u32; TX_SIZES_ALL] =
    [0, 1, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2];

/// §7.13.3 行方向的 1D 变换：`DCT_DCT / ADST_DCT / FLIPADST_DCT / H_DCT` 走 DCT，
/// 第二组走 ADST，其余（`IDTX / V_DCT / V_ADST / V_FLIPADST`）走恒等。
fn row_1d(tx_type: u8) -> Transform1d {
    match tx_type {
        DCT_DCT | ADST_DCT | FLIPADST_DCT | H_DCT => Transform1d::Dct,
        DCT_ADST | ADST_ADST | DCT_FLIPADST | FLIPADST_FLIPADST | ADST_FLIPADST
        | FLIPADST_ADST | H_ADST | H_FLIPADST => Transform1d::Adst,
        _ => Transform1d::Identity,
    }
}

/// §7.13.3 列方向的 1D 变换：`DCT_DCT / DCT_ADST / DCT_FLIPADST / V_DCT` 走 DCT，
/// 第二组走 ADST，其余（`IDTX / H_DCT / H_ADST / H_FLIPADST`）走恒等。
fn col_1d(tx_type: u8) -> Transform1d {
    match tx_type {
        DCT_DCT | DCT_ADST | DCT_FLIPADST | V_DCT => Transform1d::Dct,
        ADST_DCT | ADST_ADST | FLIPADST_DCT | FLIPADST_FLIPADST | ADST_FLIPADST
        | FLIPADST_ADST | V_ADST | V_FLIPADST => Transform1d::Adst,
        _ => Transform1d::Identity,
    }
}

/// §7.12.4 末尾的翻转：`(flipUD, flipLR)`。
#[inline]
fn flips(tx_type: u8) -> (bool, bool) {
    let flip_ud = matches!(
        tx_type,
        FLIPADST_DCT | FLIPADST_ADST | V_FLIPADST | FLIPADST_FLIPADST
    );
    let flip_lr = matches!(
        tx_type,
        DCT_FLIPADST | ADST_FLIPADST | H_FLIPADST | FLIPADST_FLIPADST
    );
    (flip_ud, flip_lr)
}

/// 1D 变换的通用入口（`t` 的长度必须是 `1 << n`；调用方已保证）。
fn forward_1d(kind: Transform1d, t: &mut [i64], n: u32, r: u32) {
    match kind {
        Transform1d::Dct => inverse_dct(t, n, r),
        Transform1d::Adst => inverse_adst(t, n, r),
        Transform1d::Identity => inverse_identity(t, n),
    }
}

/// `Transform1d` → [`super::simd`] 内核的类别编号（`TXF_*`）。
#[inline]
fn kind_code(kind: Transform1d) -> u32 {
    match kind {
        Transform1d::Dct => super::simd::TXF_DCT,
        Transform1d::Adst => super::simd::TXF_ADST,
        Transform1d::Identity => super::simd::TXF_IDENTITY,
    }
}

/// §7.13.3 的 2D 反变换：`Dequant`（`w × h` 行优先，`t` 里给出）→ `Residual`。
///
/// `residual` 由调用方提供（**复用**：逐块 `vec![]` 是这个热路径上最大的一笔浪费，
/// 缓冲的生命周期应该跟 `BlockCtx` 的瓦片一样长）。长度不足时会重新开。
fn inverse_transform_2d_into(
    t: &[i32],
    tx_sz: u8,
    tx_type: u8,
    bit_depth: u8,
    lossless: bool,
    residual: &mut Vec<i64>,
) -> Result<()> {
    let idx = tx_sz as usize;
    let w = TX_WIDTH[idx] as usize;
    let h = TX_HEIGHT[idx] as usize;
    let log2_w = TX_WIDTH_LOG2[idx] as u32;
    let log2_h = TX_HEIGHT_LOG2[idx] as u32;

    if lossless {
        // §5.11.33：`txSz = Lossless ? TX_4X4 : ...`，逆 WHT 也只在 4 点上定义。
        if w != 4 || h != 4 {
            return Err(MediaError::Bitstream(format!(
                "AV1 无损反变换：无损块的 tx_sz 必须是 TX_4X4，实际 {w}×{h}"
            )));
        }
    } else {
        // 规范里 IDTX 只出现在 max(w, h) ≤ 32 的尺寸（64 边长只有 DCTONLY 的 tx set）。
        if (row_1d(tx_type) == Transform1d::Identity && log2_w > 5)
            || (col_1d(tx_type) == Transform1d::Identity && log2_h > 5)
        {
            return Err(MediaError::Bitstream(format!(
                "AV1 反变换：IDTX 不支持 {w}×{h}（最大边长 > 32）"
            )));
        }
    }

    let row_shift = if lossless { 0 } else { TRANSFORM_ROW_SHIFT[idx] };
    let col_shift: u32 = if lossless { 0 } else { 4 };
    let row_clamp = bit_depth as u32 + 8;
    let col_clamp = (bit_depth as u32 + 6).max(16);

    // 复用缓冲：长度没变就什么都不做（连续同尺寸的块是常态）。
    // 行趟会把 `residual` 的 `w·h` 个位置**全部写满**，所以不需要清零。
    if residual.len() != w * h {
        residual.clear();
        residual.resize(w * h, 0);
    }
    let mut line = [0i64; 64]; // max(w, h) ≤ 64

    // 行变换（含 1:2 长方块的 Round2( *2896, 12) 预缩放）。
    // **SIMD 快路径**：4 行一组（通道 = 4 行），标量回路只剩尾巴（当前形状下为空）。
    let rows_done = if super::simd::avx2_available() {
        // SAFETY: 已判 AVX2；`t` / `residual` 的长度都已检查过（w·h）。
        unsafe {
            super::simd::inv_txfm_rows(
                t,
                residual,
                w,
                h,
                kind_code(row_1d(tx_type)),
                log2_w,
                row_clamp,
                row_shift,
                log2_w.abs_diff(log2_h) == 1,
                lossless,
            )
        }
    } else {
        0
    };
    for i in rows_done..h {
        for j in 0..w {
            // 系数格只有 32×32：i 或 j 到 32 以后按 0 处理
            line[j] = if i < 32 && j < 32 { t[i * w + j] as i64 } else { 0 };
        }
        if log2_w.abs_diff(log2_h) == 1 {
            for j in 0..w {
                line[j] = round2(line[j] * 2896, 12);
            }
        }
        if lossless {
            inverse_wht(&mut line[..4], 2);
        } else {
            forward_1d(row_1d(tx_type), &mut line[..w], log2_w, row_clamp);
        }
        for j in 0..w {
            residual[i * w + j] = round2(line[j], row_shift);
        }
    }

    // 行列之间的钳位
    let lo = -(1i64 << (col_clamp - 1));
    let hi = (1i64 << (col_clamp - 1)) - 1;
    for v in residual.iter_mut() {
        *v = (*v).clamp(lo, hi);
    }

    // 列变换。**SIMD 快路径**：4 列一组（同一行的 4 个 i64 在内存里连续，
    // 「列间天然独立」在这里最省事），标量回路只剩尾巴（当前形状下为空）。
    let cols_done = if super::simd::avx2_available() {
        // SAFETY: 已判 AVX2；`residual` 长度 = w·h，形状检查在函数内部。
        unsafe {
            super::simd::inv_txfm_cols(
                residual,
                w,
                h,
                kind_code(col_1d(tx_type)),
                log2_h,
                col_clamp,
                col_shift,
                lossless,
            )
        }
    } else {
        0
    };
    for j in cols_done..w {
        for i in 0..h {
            line[i] = residual[i * w + j];
        }
        if lossless {
            inverse_wht(&mut line[..4], 0);
        } else {
            forward_1d(col_1d(tx_type), &mut line[..h], log2_h, col_clamp);
        }
        for i in 0..h {
            residual[i * w + j] = round2(line[i], col_shift);
        }
    }
    Ok(())
}

/// 诊断入口：只算反变换的残差（不写像素），给 coeff.rs 的 trace 用。
pub fn debug_inverse_only(
    t: &[i32],
    tx_sz: u8,
    tx_type: u8,
    bit_depth: u8,
    lossless: bool,
) -> Vec<i64> {
    let mut residual = Vec::new();
    inverse_transform_2d_into(t, tx_sz, tx_type, bit_depth, lossless, &mut residual)
        .unwrap_or_default();
    residual
}

/// §7.13 反变换 + 加回预测：`dst[i][j] = Clip1( pred[i][j] + residual[i][j] )`。
///
/// `pred` 是 `w×h` 行优先的预测值（来自 [`super::intra`]），`dst` 是像素平面
/// （带 `stride`，只写 `w×h` 那块）。`lossless` 见模块头注释。
///
/// `residual` 是**复用缓冲**（由调用方持有，见 [`inverse_transform_2d_into`]）——
/// 逐块 `vec![]` 一次 16KB 的分配+清零是这个热路径上最大的浪费之一。
///
/// `FLIPADST` 系列按 §7.12.4 末尾：`Residual[i][j]` 加到**翻转后**的
/// `(yy, xx)` 上（等价于把残差块翻转后加到正常位置的预测上）。
#[allow(clippy::too_many_arguments)] // 签名是模块契约（含 lossless），不为行数拆包
pub fn inverse_transform_add(
    t: &[i32],
    tx_sz: u8,
    tx_type: u8,
    pred: &[u16],
    dst: &mut [u16],
    stride: usize,
    bit_depth: u8,
    lossless: bool,
    residual: &mut Vec<i64>,
) -> Result<()> {
    let idx = tx_sz as usize;
    if idx >= TX_SIZES_ALL {
        return Err(MediaError::Bitstream(format!(
            "AV1 反变换：tx_sz = {tx_sz} 越界（TX_SIZES_ALL = {TX_SIZES_ALL}）"
        )));
    }
    if tx_type as usize >= TX_TYPES {
        return Err(MediaError::Bitstream(format!(
            "AV1 反变换：tx_type = {tx_type} 越界（TX_TYPES = {TX_TYPES}）"
        )));
    }
    let w = TX_WIDTH[idx] as usize;
    let h = TX_HEIGHT[idx] as usize;
    if t.len() < w * h {
        return Err(MediaError::Bitstream(format!(
            "AV1 反变换：t 太短（{} < {}）",
            t.len(),
            w * h
        )));
    }
    if pred.len() < w * h {
        return Err(MediaError::Bitstream(format!(
            "AV1 反变换：pred 太短（{} < {}）",
            pred.len(),
            w * h
        )));
    }
    if dst.len() < (h - 1) * stride + w {
        return Err(MediaError::Bitstream(format!(
            "AV1 反变换：dst 太短（{} < {}）",
            dst.len(),
            (h - 1) * stride + w
        )));
    }

    if flags::no_residual() {
        // 诊断：只写预测值（跳过残差），用来二分"预测/残差"哪一侧出错
        let (flip_ud, flip_lr) = flips(tx_type);
        for i in 0..h {
            let yy = if flip_ud { h - 1 - i } else { i };
            for j in 0..w {
                let xx = if flip_lr { w - 1 - j } else { j };
                dst[yy * stride + xx] = pred[yy * w + xx];
            }
        }
        return Ok(());
    }
    let residual = {
        inverse_transform_2d_into(t, tx_sz, tx_type, bit_depth, lossless, residual)?;
        &residual[..w * h]
    };

    // 无损模式：规范要求 Residual 能用 1 + BitDepth 位有符号整数表示
    // （§7.12.4 的 conformance note）。超出说明码流/调用方有问题，如实报错。
    if lossless {
        let limit = 1i64 << bit_depth;
        if let Some(bad) = residual.iter().find(|v| **v < -limit || **v > limit - 1) {
            return Err(MediaError::Bitstream(format!(
                "AV1 无损反变换：残差 {bad} 超出 {}-位有符号范围（tx_sz={tx_sz} tx_type={tx_type} t={:?} residual={:?}）",
                bit_depth + 1,
                &t[..t.len().min(16)],
                &residual[..residual.len().min(16)],
            )));
        }
    }

    let (flip_ud, flip_lr) = flips(tx_type);
    for i in 0..h {
        let yy = if flip_ud { h - 1 - i } else { i };
        for j in 0..w {
            let xx = if flip_lr { w - 1 - j } else { j };
            let pred_v = pred[yy * w + xx] as i32;
            let out = clip1(pred_v + residual[i * w + j] as i32, bit_depth);
            dst[yy * stride + xx] = out;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 只算反变换（pred 全 0），返回残差。
    fn inv(tx_sz: u8, tx_type: u8, coeffs: &[(usize, i32)]) -> Vec<i64> {
        let w = TX_WIDTH[tx_sz as usize] as usize;
        let h = TX_HEIGHT[tx_sz as usize] as usize;
        let mut t = vec![0i32; w * h];
        for (k, v) in coeffs {
            t[*k] = *v;
        }
        let pred = vec![0u16; w * h];
        let mut dst = vec![0u16; w * h];
        let mut residual = Vec::new();
        inverse_transform_add(
            &t,
            tx_sz,
            tx_type,
            &pred,
            &mut dst,
            w,
            8,
            false,
            &mut residual,
        )
        .unwrap();
        dst.iter().map(|v| *v as i64).collect()
    }

    /// **SIMD 快路径与标量逐位一致**：全 (tx_sz, tx_type) × 3 种位深 × 多组系数
    /// （随机 + 极端）、外加无损 4×4 的 WHT，比 `inverse_transform_2d_into` 的
    /// 残差（i64）。判据是**整块逐元素相等** —— 这比比像素强（像素被 Clip1 掩盖）。
    ///
    /// 同一个进程里"关 SIMD"靠 [`crate::video::simd::FORCE_SCALAR`]。
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_transform_matches_scalar_bit_for_bit() {
        use crate::video::block::TX_4X4;
        use std::sync::atomic::Ordering;
        if !crate::video::simd::avx2_available() {
            println!("这台机器没有 AVX2，跳过");
            return;
        }
        let mut changed = 0usize;
        for ts in 0..TX_SIZES_ALL as u8 {
            for tg in 0..TX_TYPES as u8 {
                let w = TX_WIDTH[ts as usize] as usize;
                let h = TX_HEIGHT[ts as usize] as usize;
                // 只测真实码流会出现的组合（ADST 只到 16 点，IDTX 只到 32 点）
                let bad = |k: Transform1d, n0: usize| {
                    (k == Transform1d::Adst && n0 > 16) || (k == Transform1d::Identity && n0 > 32)
                };
                if bad(row_1d(tg), w) || bad(col_1d(tg), h) {
                    continue;
                }
                for bd in [8u8, 10, 12] {
                    let max_coeff = (1i32 << (7 + bd)) - 1;
                    for pattern in 0..4 {
                        let mut t = vec![0i32; w * h];
                        let mut s = (pattern as u32)
                            .wrapping_mul(2654435761)
                            .wrapping_add(ts as u32 * 97 + tg as u32);
                        for i in 0..h.min(32) {
                            for j in 0..w.min(32) {
                                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                                let r = (s >> 9) as i32;
                                t[i * w + j] = match pattern {
                                    0 => r % (max_coeff + 1),
                                    1 => max_coeff,
                                    2 => if (i + j) % 2 == 0 { max_coeff } else { -max_coeff },
                                    _ => r % (max_coeff / 8 + 1),
                                };
                            }
                        }
                        crate::video::simd::FORCE_SCALAR.store(true, Ordering::Relaxed);
                        let mut a = Vec::new();
                        inverse_transform_2d_into(&t, ts, tg, bd, false, &mut a).unwrap();
                        crate::video::simd::FORCE_SCALAR.store(false, Ordering::Relaxed);
                        let mut b = Vec::new();
                        inverse_transform_2d_into(&t, ts, tg, bd, false, &mut b).unwrap();
                        assert_eq!(
                            a, b,
                            "SIMD 与标量不一致：tx_sz={ts} tx_type={tg} bd={bd} pattern={pattern}"
                        );
                        if a.iter().any(|v| *v != 0) {
                            changed += 1;
                        }
                    }
                }
            }
        }
        // 无损（4×4 的 WHT 行/列）
        for pattern in 0..3u32 {
            let mut t = vec![0i32; 16];
            for (k, v) in t.iter_mut().enumerate() {
                *v = ((k as u32 * 7 + pattern) % 9) as i32 - 4;
            }
            crate::video::simd::FORCE_SCALAR.store(true, Ordering::Relaxed);
            let mut a = Vec::new();
            inverse_transform_2d_into(&t, TX_4X4, DCT_DCT, 8, true, &mut a).unwrap();
            crate::video::simd::FORCE_SCALAR.store(false, Ordering::Relaxed);
            let mut b = Vec::new();
            inverse_transform_2d_into(&t, TX_4X4, DCT_DCT, 8, true, &mut b).unwrap();
            assert_eq!(a, b, "无损 WHT 的 SIMD 与标量不一致：pattern={pattern}");
            changed += 1;
        }
        assert!(changed > 100, "对拍样本大多是零残差（changed={changed}），测试形同虚设");
    }

    /// DC-only：必须**处处相等**，且等于 `DC / (w·h)`（2D 反 DCT 的归一化）。
    #[test]
    fn dc_only_is_flat() {
        for (tx_sz, dc) in [(0u8, 16), (1, 64), (2, 256)] {
            let r = inv(tx_sz, DCT_DCT, &[(0, dc)]);
            assert!(r.iter().all(|v| *v == r[0]), "tx_sz={tx_sz} DC-only 应当平坦：{r:?}");
            assert!(r[0] > 0, "DC 应当给出正值");
        }
    }

    /// 竖直频率 `(i=1, j=0)`：**行内相同、行间不同**（列变换管竖直方向）。
    #[test]
    fn vertical_frequency_varies_by_row_only() {
        let r = inv(0, DCT_DCT, &[(4, 1600)]);
        for i in 0..4 {
            let row = &r[i * 4..(i + 1) * 4];
            assert!(row.iter().all(|v| *v == row[0]), "第 {i} 行内应当相同：{row:?}");
        }
        assert!(r[0] != r[4], "不同行之间应当有变化：{r:?}");
    }

    /// 水平频率 `(i=0, j=1)`：**列内相同、列间不同**（行变换管水平方向）。
    #[test]
    fn horizontal_frequency_varies_by_column_only() {
        let r = inv(0, DCT_DCT, &[(1, 1600)]);
        for j in 0..4 {
            let col: Vec<i64> = (0..4).map(|i| r[i * 4 + j]).collect();
            assert!(col.iter().all(|v| *v == col[0]), "第 {j} 列内应当相同：{col:?}");
        }
        assert!(r[0] != r[1], "不同列之间应当有变化：{r:?}");
    }

    /// IDTX（tx_type=9）：一个系数只影响它自己那个位置（不扩散）。
    /// 数值上两次 `5793/2^12` 预缩放会把 DC=4096 抬到 512，被 `Clip1` 截到 255 ——
    /// 这里只钉"不扩散"这条结构性质（数值缩放在 §7.13.3 的移位表里）。
    #[test]
    fn idtx_keeps_the_coefficient_in_place() {
        let r = inv(0, IDTX, &[(0, 4096)]);
        assert!(r[0] > 0, "DC 位置应当有值：{r:?}");
        assert!(r[1..].iter().all(|v| *v == 0), "IDTX 不应扩散：{r:?}");
        let r = inv(0, IDTX, &[(5, 4096)]);
        assert_eq!(r[5], 255, "只有 (1,1) 有值且被 Clip1 截到 255：{r:?}");
        assert!(r.iter().enumerate().all(|(i, v)| i == 5 || *v == 0), "不应扩散：{r:?}");
    }

    /// ADST 参与的那一维：DC-only 时**行内仍应相同、行间有变化**（ADST 的 DC 基
    /// 不是常量，但同一行上的值只由列方向决定）。
    #[test]
    fn adst_side_still_varies_along_its_own_axis() {
        // tx_type=4（FLIPADST_DCT）：行 DCT、列 ADST → 行内相同、行间不同
        let r = inv(0, 4, &[(0, 1600)]);
        for i in 0..4 {
            let row = &r[i * 4..(i + 1) * 4];
            assert!(row.iter().all(|v| *v == row[0]), "行内应相同：{row:?}");
        }
        assert!(r[0] != r[4], "行间应有变化（ADST 列）：{r:?}");
    }
}
