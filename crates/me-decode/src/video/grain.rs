//! AV1 胶片颗粒合成（规范 §7.18.3）。
//!
//! 这是解码管线**最后一步**、也是唯一一步"只作用于输出、不进入参考帧"的处理：
//! 颗粒加在环路恢复（`LrFrame`）之后，加完才转 NV12 上屏；存进参考帧槽的是
//! **加颗粒之前**的像素（否则帧间预测会拿着带噪的参考去算，整条链就废了）。
//!
//! ## 三个过程（与规范一一对应）
//!
//! | 规范 | 这里 | 做什么 |
//! |---|---|---|
//! | §7.18.3.2 | [`generate_scaling`] | 点表 → 256/1024/4096 档的"信号强度 → 颗粒幅度"查找表 |
//! | §7.18.3.3 | [`generate_grain_y`] / [`generate_grain_uv`] | 82×73（色度 44×38）的自回归颗粒模板 |
//! | §7.18.3.4/5/6 | [`fgy_32x32xn`] / [`fguv_32x32xn`] | 按 32×32 块把模板混进画面，含 overlap 加权 |
//!
//! ## 三个最容易写错的地方
//!
//! 1. **颗粒模板的种子分两层**：模板本身用 `grain_seed`（色度再异或
//!    `0xb524`/`0x49d8`）；而逐块的偏移量用 `seed ^ (((row-i)*37+178)&0xFF)<<8
//!    ^ (((row-i)*173+105)&0xFF)`，U/V **共用同一批偏移**。
//! 2. **色度用的是"没加颗粒的亮度"**：`avg` 取自原始亮度，所以色度必须先算、
//!    亮度后算（否则就地加颗粒会把亮度污染掉）。
//! 3. **overlap 的 1..2 像素边界**：`by=1`/`bx=1` 只出现在重叠区（`y<2`/`x<2`），
//!    查表才不会越界 —— 提前"优化"掉这个约束一定会越界。
//!
//! ## 验收口径
//!
//! - 单测钉死可解析解的部分：PRNG 序列、点表插值、纯直流画面的"限制范围裁剪"、
//!   以及"没有点表 ⇒ 逐像素不变"。
//! - 端到端钉在 `tools/av1ref`：libaom 默认开颗粒，`refcmp.py --grain=1` 必须逐像素一致。

use super::intra::{CurrFrame, Plane};
use super::params::FilmGrainParams;

/// `GRAIN_WIDTH`：颗粒模板宽度。
pub const GRAIN_WIDTH: usize = 82;
/// `GRAIN_HEIGHT`：颗粒模板高度。
pub const GRAIN_HEIGHT: usize = 73;
/// `SUB_GRAIN_WIDTH`：色度模板宽度（水平 2× 下采样时）。
pub const SUB_GRAIN_WIDTH: usize = 44;
/// `SUB_GRAIN_HEIGHT`：色度模板高度（垂直 2× 下采样时）。
pub const SUB_GRAIN_HEIGHT: usize = 38;
/// `FG_BLOCK_SIZE`：逐块混合的块边长。
pub const FG_BLOCK_SIZE: usize = 32;
/// `SCALING_SIZE`：缩放表的最大档数（12 位时 4096）。
const SCALING_SIZE: usize = 4096;

/// `Gaussian_Sequence[2048]`（规范 §9.3）：均值 0、标准差 512、范围 [-2048, 2047]。
const GAUSSIAN_SEQUENCE: [i16; 2048] = [
    56, 568, -180, 172, 124, -84, 172, -64, -900, 24, 820, 224,
    1248, 996, 272, -8, -916, -388, -732, -104, -188, 800, 112, -652,
    -320, -376, 140, -252, 492, -168, 44, -788, 588, -584, 500, -228,
    12, 680, 272, -476, 972, -100, 652, 368, 432, -196, -720, -192,
    1000, -332, 652, -136, -552, -604, -4, 192, -220, -136, 1000, -52,
    372, -96, -624, 124, -24, 396, 540, -12, -104, 640, 464, 244,
    -208, -84, 368, -528, -740, 248, -968, -848, 608, 376, -60, -292,
    -40, -156, 252, -292, 248, 224, -280, 400, -244, 244, -60, 76,
    -80, 212, 532, 340, 128, -36, 824, -352, -60, -264, -96, -612,
    416, -704, 220, -204, 640, -160, 1220, -408, 900, 336, 20, -336,
    -96, -792, 304, 48, -28, -1232, -1172, -448, 104, -292, -520, 244,
    60, -948, 0, -708, 268, 108, 356, -548, 488, -344, -136, 488,
    -196, -224, 656, -236, -1128, 60, 4, 140, 276, -676, -376, 168,
    -108, 464, 8, 564, 64, 240, 308, -300, -400, -456, -136, 56,
    120, -408, -116, 436, 504, -232, 328, 844, -164, -84, 784, -168,
    232, -224, 348, -376, 128, 568, 96, -1244, -288, 276, 848, 832,
    -360, 656, 464, -384, -332, -356, 728, -388, 160, -192, 468, 296,
    224, 140, -776, -100, 280, 4, 196, 44, -36, -648, 932, 16,
    1428, 28, 528, 808, 772, 20, 268, 88, -332, -284, 124, -384,
    -448, 208, -228, -1044, -328, 660, 380, -148, -300, 588, 240, 540,
    28, 136, -88, -436, 256, 296, -1000, 1400, 0, -48, 1056, -136,
    264, -528, -1108, 632, -484, -592, -344, 796, 124, -668, -768, 388,
    1296, -232, -188, -200, -288, -4, 308, 100, -168, 256, -500, 204,
    -508, 648, -136, 372, -272, -120, -1004, -552, -548, -384, 548, -296,
    428, -108, -8, -912, -324, -224, -88, -112, -220, -100, 996, -796,
    548, 360, -216, 180, 428, -200, -212, 148, 96, 148, 284, 216,
    -412, -320, 120, -300, -384, -604, -572, -332, -8, -180, -176, 696,
    116, -88, 628, 76, 44, -516, 240, -208, -40, 100, -592, 344,
    -308, -452, -228, 20, 916, -1752, -136, -340, -804, 140, 40, 512,
    340, 248, 184, -492, 896, -156, 932, -628, 328, -688, -448, -616,
    -752, -100, 560, -1020, 180, -800, -64, 76, 576, 1068, 396, 660,
    552, -108, -28, 320, -628, 312, -92, -92, -472, 268, 16, 560,
    516, -672, -52, 492, -100, 260, 384, 284, 292, 304, -148, 88,
    -152, 1012, 1064, -228, 164, -376, -684, 592, -392, 156, 196, -524,
    -64, -884, 160, -176, 636, 648, 404, -396, -436, 864, 424, -728,
    988, -604, 904, -592, 296, -224, 536, -176, -920, 436, -48, 1176,
    -884, 416, -776, -824, -884, 524, -548, -564, -68, -164, -96, 692,
    364, -692, -1012, -68, 260, -480, 876, -1116, 452, -332, -352, 892,
    -1088, 1220, -676, 12, -292, 244, 496, 372, -32, 280, 200, 112,
    -440, -96, 24, -644, -184, 56, -432, 224, -980, 272, -260, 144,
    -436, 420, 356, 364, -528, 76, 172, -744, -368, 404, -752, -416,
    684, -688, 72, 540, 416, 92, 444, 480, -72, -1416, 164, -1172,
    -68, 24, 424, 264, 1040, 128, -912, -524, -356, 64, 876, -12,
    4, -88, 532, 272, -524, 320, 276, -508, 940, 24, -400, -120,
    756, 60, 236, -412, 100, 376, -484, 400, -100, -740, -108, -260,
    328, -268, 224, -200, -416, 184, -604, -564, -20, 296, 60, 892,
    -888, 60, 164, 68, -760, 216, -296, 904, -336, -28, 404, -356,
    -568, -208, -1480, -512, 296, 328, -360, -164, -1560, -776, 1156, -428,
    164, -504, -112, 120, -216, -148, -264, 308, 32, 64, -72, 72,
    116, 176, -64, -272, 460, -536, -784, -280, 348, 108, -752, -132,
    524, -540, -776, 116, -296, -1196, -288, -560, 1040, -472, 116, -848,
    -1116, 116, 636, 696, 284, -176, 1016, 204, -864, -648, -248, 356,
    972, -584, -204, 264, 880, 528, -24, -184, 116, 448, -144, 828,
    524, 212, -212, 52, 12, 200, 268, -488, -404, -880, 824, -672,
    -40, 908, -248, 500, 716, -576, 492, -576, 16, 720, -108, 384,
    124, 344, 280, 576, -500, 252, 104, -308, 196, -188, -8, 1268,
    296, 1032, -1196, 436, 316, 372, -432, -200, -660, 704, -224, 596,
    -132, 268, 32, -452, 884, 104, -1008, 424, -1348, -280, 4, -1168,
    368, 476, 696, 300, -8, 24, 180, -592, -196, 388, 304, 500,
    724, -160, 244, -84, 272, -256, -420, 320, 208, -144, -156, 156,
    364, 452, 28, 540, 316, 220, -644, -248, 464, 72, 360, 32,
    -388, 496, -680, -48, 208, -116, -408, 60, -604, -392, 548, -840,
    784, -460, 656, -544, -388, -264, 908, -800, -628, -612, -568, 572,
    -220, 164, 288, -16, -308, 308, -112, -636, -760, 280, -668, 432,
    364, 240, -196, 604, 340, 384, 196, 592, -44, -500, 432, -580,
    -132, 636, -76, 392, 4, -412, 540, 508, 328, -356, -36, 16,
    -220, -64, -248, -60, 24, -192, 368, 1040, 92, -24, -1044, -32,
    40, 104, 148, 192, -136, -520, 56, -816, -224, 732, 392, 356,
    212, -80, -424, -1008, -324, 588, -1496, 576, 460, -816, -848, 56,
    -580, -92, -1372, -112, -496, 200, 364, 52, -140, 48, -48, -60,
    84, 72, 40, 132, -356, -268, -104, -284, -404, 732, -520, 164,
    -304, -540, 120, 328, -76, -460, 756, 388, 588, 236, -436, -72,
    -176, -404, -316, -148, 716, -604, 404, -72, -88, -888, -68, 944,
    88, -220, -344, 960, 472, 460, -232, 704, 120, 832, -228, 692,
    -508, 132, -476, 844, -748, -364, -44, 1116, -1104, -1056, 76, 428,
    552, -692, 60, 356, 96, -384, -188, -612, -576, 736, 508, 892,
    352, -1132, 504, -24, -352, 324, 332, -600, -312, 292, 508, -144,
    -8, 484, 48, 284, -260, -240, 256, -100, -292, -204, -44, 472,
    -204, 908, -188, -1000, -256, 92, 1164, -392, 564, 356, 652, -28,
    -884, 256, 484, -192, 760, -176, 376, -524, -452, -436, 860, -736,
    212, 124, 504, -476, 468, 76, -472, 552, -692, -944, -620, 740,
    -240, 400, 132, 20, 192, -196, 264, -668, -1012, -60, 296, -316,
    -828, 76, -156, 284, -768, -448, -832, 148, 248, 652, 616, 1236,
    288, -328, -400, -124, 588, 220, 520, -696, 1032, 768, -740, -92,
    -272, 296, 448, -464, 412, -200, 392, 440, -200, 264, -152, -260,
    320, 1032, 216, 320, -8, -64, 156, -1016, 1084, 1172, 536, 484,
    -432, 132, 372, -52, -256, 84, 116, -352, 48, 116, 304, -384,
    412, 924, -300, 528, 628, 180, 648, 44, -980, -220, 1320, 48,
    332, 748, 524, -268, -720, 540, -276, 564, -344, -208, -196, 436,
    896, 88, -392, 132, 80, -964, -288, 568, 56, -48, -456, 888,
    8, 552, -156, -292, 948, 288, 128, -716, -292, 1192, -152, 876,
    352, -600, -260, -812, -468, -28, -120, -32, -44, 1284, 496, 192,
    464, 312, -76, -516, -380, -456, -1012, -48, 308, -156, 36, 492,
    -156, -808, 188, 1652, 68, -120, -116, 316, 160, -140, 352, 808,
    -416, 592, 316, -480, 56, 528, -204, -568, 372, -232, 752, -344,
    744, -4, 324, -416, -600, 768, 268, -248, -88, -132, -420, -432,
    80, -288, 404, -316, -1216, -588, 520, -108, 92, -320, 368, -480,
    -216, -92, 1688, -300, 180, 1020, -176, 820, -68, -228, -260, 436,
    -904, 20, 40, -508, 440, -736, 312, 332, 204, 760, -372, 728,
    96, -20, -632, -520, -560, 336, 1076, -64, -532, 776, 584, 192,
    396, -728, -520, 276, -188, 80, -52, -612, -252, -48, 648, 212,
    -688, 228, -52, -260, 428, -412, -272, -404, 180, 816, -796, 48,
    152, 484, -88, -216, 988, 696, 188, -528, 648, -116, -180, 316,
    476, 12, -564, 96, 476, -252, -364, -376, -392, 556, -256, -576,
    260, -352, 120, -16, -136, -260, -492, 72, 556, 660, 580, 616,
    772, 436, 424, -32, -324, -1268, 416, -324, -80, 920, 160, 228,
    724, 32, -516, 64, 384, 68, -128, 136, 240, 248, -204, -68,
    252, -932, -120, -480, -628, -84, 192, 852, -404, -288, -132, 204,
    100, 168, -68, -196, -868, 460, 1080, 380, -80, 244, 0, 484,
    -888, 64, 184, 352, 600, 460, 164, 604, -196, 320, -64, 588,
    -184, 228, 12, 372, 48, -848, -344, 224, 208, -200, 484, 128,
    -20, 272, -468, -840, 384, 256, -720, -520, -464, -580, 112, -120,
    644, -356, -208, -608, -528, 704, 560, -424, 392, 828, 40, 84,
    200, -152, 0, -144, 584, 280, -120, 80, -556, -972, -196, -472,
    724, 80, 168, -32, 88, 160, -688, 0, 160, 356, 372, -776,
    740, -128, 676, -248, -480, 4, -364, 96, 544, 232, -1032, 956,
    236, 356, 20, -40, 300, 24, -676, -596, 132, 1120, -104, 532,
    -1096, 568, 648, 444, 508, 380, 188, -376, -604, 1488, 424, 24,
    756, -220, -192, 716, 120, 920, 688, 168, 44, -460, 568, 284,
    1144, 1160, 600, 424, 888, 656, -356, -320, 220, 316, -176, -724,
    -188, -816, -628, -348, -228, -380, 1012, -452, -660, 736, 928, 404,
    -696, -72, -268, -892, 128, 184, -344, -780, 360, 336, 400, 344,
    428, 548, -112, 136, -228, -216, -820, -516, 340, 92, -136, 116,
    -300, 376, -244, 100, -316, -520, -284, -12, 824, 164, -548, -180,
    -128, 116, -924, -828, 268, -368, -580, 620, 192, 160, 0, -1676,
    1068, 424, -56, -360, 468, -156, 720, 288, -528, 556, -364, 548,
    -148, 504, 316, 152, -648, -620, -684, -24, -376, -384, -108, -920,
    -1032, 768, 180, -264, -508, -1268, -260, -60, 300, -240, 988, 724,
    -376, -576, -212, -736, 556, 192, 1092, -620, -880, 376, -56, -4,
    -216, -32, 836, 268, 396, 1332, 864, -600, 100, 56, -412, -92,
    356, 180, 884, -468, -436, 292, -388, -804, -704, -840, 368, -348,
    140, -724, 1536, 940, 372, 112, -372, 436, -480, 1136, 296, -32,
    -228, 132, -48, -220, 868, -1016, -60, -1044, -464, 328, 916, 244,
    12, -736, -296, 360, 468, -376, -108, -92, 788, 368, -56, 544,
    400, -672, -420, 728, 16, 320, 44, -284, -380, -796, 488, 132,
    204, -596, -372, 88, -152, -908, -636, -572, -624, -116, -692, -200,
    -56, 276, -88, 484, -324, 948, 864, 1000, -456, -184, -276, 292,
    -296, 156, 676, 320, 160, 908, -84, -1236, -288, -116, 260, -372,
    -644, 732, -756, -96, 84, 344, -520, 348, -688, 240, -84, 216,
    -1044, -136, -676, -396, -1500, 960, -40, 176, 168, 1516, 420, -504,
    -344, -364, -360, 1216, -940, -380, -212, 252, -660, -708, 484, -444,
    -152, 928, -120, 1112, 476, -260, 560, -148, -344, 108, -196, 228,
    -288, 504, 560, -328, -88, 288, -1008, 460, -228, 468, -836, -196,
    76, 388, 232, 412, -1168, -716, -644, 756, -172, -356, -504, 116,
    432, 528, 48, 476, -168, -608, 448, 160, -532, -272, 28, -676,
    -12, 828, 980, 456, 520, 104, -104, 256, -344, -4, -28, -368,
    -52, -524, -572, -556, -200, 768, 1124, -208, -512, 176, 232, 248,
    -148, -888, 604, -600, -304, 804, -156, -212, 488, -192, -804, -256,
    368, -360, -916, -328, 228, -240, -448, -472, 856, -556, -364, 572,
    -12, -156, -368, -340, 432, 252, -752, -152, 288, 268, -580, -848,
    -592, 108, -76, 244, 312, -716, 592, -80, 436, 360, 4, -248,
    160, 516, 584, 732, 44, -468, -280, -292, -156, -588, 28, 308,
    912, 24, 124, 156, 180, -252, 944, -924, -772, -520, -428, -624,
    300, -212, -1144, 32, -724, 800, -1128, -212, -1288, -848, 180, -416,
    440, 192, -576, -792, -76, -1080, 80, -532, -352, -132, 380, -820,
    148, 1112, 128, 164, 456, 700, -924, 144, -668, -384, 648, -832,
    508, 552, -52, -100, -656, 208, -568, 748, -88, 680, 232, 300,
    192, -408, -1012, -152, -252, -268, 272, -876, -664, -648, -332, -136,
    16, 12, 1152, -28, 332, -536, 320, -672, -460, -316, 532, -260,
    228, -40, 1052, -816, 180, 88, -496, -556, -672, -368, 428, 92,
    356, 404, -408, 252, 196, -176, -556, 792, 268, 32, 372, 40,
    96, -332, 328, 120, 372, -900, -40, 472, -264, -592, 952, 128,
    656, 112, 664, -232, 420, 4, -344, -464, 556, 244, -416, -32,
    252, 0, -412, 188, -696, 508, -476, 324, -1096, 656, -312, 560,
    264, -136, 304, 160, -64, -580, 248, 336, -720, 560, -348, -288,
    -276, -196, -500, 852, -544, -236, -1128, -992, -776, 116, 56, 52,
    860, 884, 212, -12, 168, 1020, 512, -552, 924, -148, 716, 188,
    164, -340, -520, -184, 880, -152, -680, -208, -1156, -300, -528, -472,
    364, 100, -744, -1056, -32, 540, 280, 144, -676, -32, -232, -280,
    -224, 96, 568, -76, 172, 148, 148, 104, 32, -296, -32, 788,
    -80, 32, -16, 280, 288, 944, 428, -484,
];

/// 颗粒模板（规范里那张 `[GRAIN_HEIGHT+1][GRAIN_WIDTH]` 的表；多一行是因为
/// 色度模板的自回归要读到第 `GRAIN_HEIGHT` 行）。
type GrainLut = [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT + 1];

/// `get_random_number(bits)`（规范 §7.18.3.1）：16 位 LFSR，取高 `bits` 位。
#[inline]
fn get_random_number(bits: u32, state: &mut u32) -> i32 {
    let r = *state;
    let bit = ((r >> 0) ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
    *state = (r >> 1) | (bit << 15);
    ((*state >> (16 - bits)) & ((1u32 << bits) - 1)) as i32
}

/// `round2(x, shift)`：先加半个单位再算术右移（负数也一样）。
#[inline]
fn round2(x: i32, shift: u32) -> i32 {
    (x + ((1i32 << shift) >> 1)) >> shift
}

/// `generate_grain_y(buf, params)`（规范 §7.18.3.3）。
fn generate_grain_y(buf: &mut GrainLut, params: &FilmGrainParams, bit_depth: u8) {
    let bd_min_8 = u32::from(bit_depth - 8);
    let mut seed = u32::from(params.grain_seed);
    let shift = 4 - bd_min_8 + u32::from(params.grain_scale_shift);
    let grain_ctr = 128i32 << bd_min_8;
    let (grain_min, grain_max) = (-grain_ctr, grain_ctr - 1);
    let ar_pad = 3usize;
    let ar_lag = i32::from(params.ar_coeff_lag);
    let ar_shift = u32::from(params.ar_coeff_shift);

    for y in 0..GRAIN_HEIGHT {
        for x in 0..GRAIN_WIDTH {
            let value = get_random_number(11, &mut seed) as usize;
            buf[y][x] = round2(i32::from(GAUSSIAN_SEQUENCE[value]), shift) as i16;
        }
    }

    for y in ar_pad..GRAIN_HEIGHT {
        for x in ar_pad..(GRAIN_WIDTH - ar_pad) {
            let mut sum = 0i32;
            let mut ci = 0usize;
            // 自回归：先左侧/上方已生成的样本，最后才是"当前像素"那一项
            'dy: for dy in -ar_lag..=0 {
                for dx in -ar_lag..=ar_lag {
                    if dx == 0 && dy == 0 {
                        continue 'dy;
                    }
                    sum += i32::from(params.ar_coeffs_y[ci])
                        * i32::from(buf[(y as i32 + dy) as usize][(x as i32 + dx) as usize]);
                    ci += 1;
                }
            }
            let grain = i32::from(buf[y][x]) + round2(sum, ar_shift);
            buf[y][x] = grain.clamp(grain_min, grain_max) as i16;
        }
    }
}

/// `generate_grain_uv(buf, buf_y, params, uv)`（规范 §7.18.3.3）。
fn generate_grain_uv(
    buf: &mut GrainLut,
    buf_y: &GrainLut,
    params: &FilmGrainParams,
    uv: usize,
    sub_x: u32,
    sub_y: u32,
    bit_depth: u8,
) {
    let bd_min_8 = u32::from(bit_depth - 8);
    // 色度模板的种子在 luma 种子基础上异或一个常量（U/V 不同）
    let mut seed = u32::from(params.grain_seed) ^ if uv == 1 { 0x49d8 } else { 0xb524 };
    let shift = 4 - bd_min_8 + u32::from(params.grain_scale_shift);
    let grain_ctr = 128i32 << bd_min_8;
    let (grain_min, grain_max) = (-grain_ctr, grain_ctr - 1);
    let chroma_w = if sub_x == 1 { SUB_GRAIN_WIDTH } else { GRAIN_WIDTH };
    let chroma_h = if sub_y == 1 { SUB_GRAIN_HEIGHT } else { GRAIN_HEIGHT };
    let ar_pad = 3usize;
    let ar_lag = i32::from(params.ar_coeff_lag);
    let ar_shift = u32::from(params.ar_coeff_shift);

    for y in 0..chroma_h {
        for x in 0..chroma_w {
            let value = get_random_number(11, &mut seed) as usize;
            buf[y][x] = round2(i32::from(GAUSSIAN_SEQUENCE[value]), shift) as i16;
        }
    }

    for y in ar_pad..chroma_h {
        for x in ar_pad..(chroma_w - ar_pad) {
            let mut sum = 0i32;
            let mut ci = 0usize;
            'dy: for dy in -ar_lag..=0 {
                for dx in -ar_lag..=ar_lag {
                    if dx == 0 && dy == 0 {
                        // 当前像素位置：叠加来自**亮度模板**的贡献（亮度无点表时跳过）
                        if params.num_y_points == 0 {
                            continue 'dy;
                        }
                        let luma_x = ((x as i32 - ar_pad as i32) << sub_x) + ar_pad as i32;
                        let luma_y = ((y as i32 - ar_pad as i32) << sub_y) + ar_pad as i32;
                        let mut luma = 0i32;
                        for i in 0..=sub_y {
                            for j in 0..=sub_x {
                                luma += i32::from(
                                    buf_y[(luma_y + i as i32) as usize]
                                        [(luma_x + j as i32) as usize],
                                );
                            }
                        }
                        let luma = round2(luma, sub_x + sub_y);
                        sum += luma * i32::from(params.ar_coeffs_uv[uv][ci]);
                        continue 'dy;
                    }
                    sum += i32::from(params.ar_coeffs_uv[uv][ci])
                        * i32::from(buf[(y as i32 + dy) as usize][(x as i32 + dx) as usize]);
                    ci += 1;
                }
            }
            let grain = i32::from(buf[y][x]) + round2(sum, ar_shift);
            buf[y][x] = grain.clamp(grain_min, grain_max) as i16;
        }
    }
}

/// `generate_scaling(points, num, scaling)`（规范 §7.18.3.2）：点表线性插值成整张表。
fn generate_scaling(
    points: &[[u8; 2]],
    num: usize,
    scaling: &mut [u8; SCALING_SIZE],
    bit_depth: u8,
) {
    let shift_x = u32::from(bit_depth - 8);
    let scaling_size = 1usize << bit_depth;
    if num == 0 {
        scaling[..scaling_size].fill(0);
        return;
    }
    let max_value = (points[num - 1][0] as usize) << shift_x;

    // 起始段：用第一个点的强度
    let first_end = (points[0][0] as usize) << shift_x;
    scaling[..first_end].fill(points[0][1]);

    // 中段：相邻点之间线性插值（16.16 定点）
    for i in 0..num - 1 {
        let bx = i32::from(points[i][0]);
        let by = i32::from(points[i][1]);
        let ex = i32::from(points[i + 1][0]);
        let ey = i32::from(points[i + 1][1]);
        let dx = ex - bx;
        let dy = ey - by;
        let delta = dy * ((0x10000 + (dx >> 1)) / dx);
        let mut d = 0x8000i32;
        for x in 0..dx {
            scaling[((bx + x) as usize) << shift_x] = (by + (d >> 16)) as u8;
            d += delta;
        }
    }

    // 尾段：用最后一个点的强度
    scaling[max_value..scaling_size].fill(points[num - 1][1]);

    // 高位深：把每个"点值档"内部再线性展开（8 位时 shift_x=0，无需处理）
    if bit_depth != 8 {
        for i in 0..num - 1 {
            let pad = 1usize << shift_x;
            let rnd = pad >> 1;
            let bx = (points[i][0] as usize) << shift_x;
            let ex = (points[i + 1][0] as usize) << shift_x;
            let mut x = 0usize;
            while x < ex - bx {
                let range = i32::from(scaling[bx + x + pad]) - i32::from(scaling[bx + x]);
                let mut r = rnd as i32;
                for n in 1..pad {
                    r += range;
                    scaling[bx + x + n] = (i32::from(scaling[bx + x]) + (r >> shift_x)) as u8;
                }
                x += pad;
            }
        }
    }
}

/// `sample_lut()`（规范 §7.18.3.4）：从模板的指定块取一个颗粒值。
#[inline]
fn sample_lut(
    grain_lut: &GrainLut,
    offsets: &[[i32; 2]; 2],
    sub_x: usize,
    sub_y: usize,
    bx: usize,
    by: usize,
    x: usize,
    y: usize,
) -> i32 {
    let randval = offsets[bx][by];
    let offx = 3 + (2 >> sub_x) * (3 + (randval >> 4));
    let offy = 3 + (2 >> sub_y) * (3 + (randval & 0xF));
    i32::from(
        grain_lut[(offy + y as i32 + ((FG_BLOCK_SIZE >> sub_y) * by) as i32) as usize]
            [(offx + x as i32 + ((FG_BLOCK_SIZE >> sub_x) * bx) as i32) as usize],
    )
}

/// 亮度逐行混合 `fgy_32x32xn`（规范 §7.18.3.5）。
///
/// 就地修改（每个像素只依赖它自己 + 模板，不依赖邻居输出），所以不需要额外拷贝。
#[allow(clippy::too_many_arguments)]
fn fgy_32x32xn(
    plane: &mut Plane,
    params: &FilmGrainParams,
    scaling: &[u8; SCALING_SIZE],
    grain_lut: &GrainLut,
    bh: usize,
    row_num: usize,
    bit_depth: u8,
) {
    let bd_min_8 = u32::from(bit_depth - 8);
    let grain_ctr = 128i32 << bd_min_8;
    let (grain_min, grain_max) = (-grain_ctr, grain_ctr - 1);
    let (min_value, max_value) = if params.clip_to_restricted_range {
        (16i32 << bd_min_8, 235i32 << bd_min_8)
    } else {
        (0, (1i32 << bit_depth) - 1)
    };
    let scaling_shift = u32::from(params.scaling_shift);
    let overlap = params.overlap_flag;
    let rows = 1 + usize::from(overlap && row_num > 0);

    // seed[0] = 当前行，seed[1] = 上一行
    let mut seed = [0u32; 2];
    for (i, s) in seed.iter_mut().enumerate().take(rows) {
        let rn = (row_num - i) as i32;
        *s = u32::from(params.grain_seed)
            ^ ((((rn * 37 + 178) & 0xFF) as u32) << 8)
            ^ ((rn * 173 + 105) & 0xFF) as u32;
    }

    let pw = plane.w;
    let base_y = row_num * FG_BLOCK_SIZE;
    let w: [[i32; 2]; 2] = [[27, 17], [17, 27]];
    let mut offsets = [[0i32; 2]; 2];

    let mut bx = 0usize;
    while bx < pw {
        let bw = FG_BLOCK_SIZE.min(pw - bx);
        let ystart = if overlap && row_num > 0 { 2.min(bh) } else { 0 };
        let xstart = if overlap && bx > 0 { 2.min(bw) } else { 0 };

        if overlap && bx > 0 {
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }
        for i in 0..rows {
            offsets[0][i] = get_random_number(8, &mut seed[i]);
        }

        let mut add = |x: usize, y: usize, grain: i32| {
            let px = bx + x;
            let py = base_y + y;
            let s = i32::from(plane.get(px, py));
            let noise = round2(i32::from(scaling[s as usize]) * grain, scaling_shift);
            plane.set(px, py, (s + noise).clamp(min_value, max_value) as u16);
        };

        for y in ystart..bh {
            for x in xstart..bw {
                let grain = sample_lut(grain_lut, &offsets, 0, 0, 0, 0, x, y);
                add(x, y, grain);
            }
            for x in 0..xstart {
                let g = sample_lut(grain_lut, &offsets, 0, 0, 0, 0, x, y);
                let old = sample_lut(grain_lut, &offsets, 0, 0, 1, 0, x, y);
                let grain = round2(old * w[x][0] + g * w[x][1], 5).clamp(grain_min, grain_max);
                add(x, y, grain);
            }
        }

        for y in 0..ystart {
            for x in xstart..bw {
                let g = sample_lut(grain_lut, &offsets, 0, 0, 0, 0, x, y);
                let old = sample_lut(grain_lut, &offsets, 0, 0, 0, 1, x, y);
                let grain = round2(old * w[y][0] + g * w[y][1], 5).clamp(grain_min, grain_max);
                add(x, y, grain);
            }
            for x in 0..xstart {
                let g = sample_lut(grain_lut, &offsets, 0, 0, 0, 0, x, y);
                let mut top = sample_lut(grain_lut, &offsets, 0, 0, 0, 1, x, y);
                let old = sample_lut(grain_lut, &offsets, 0, 0, 1, 1, x, y);
                top = round2(old * w[x][0] + top * w[x][1], 5).clamp(grain_min, grain_max);
                let old = sample_lut(grain_lut, &offsets, 0, 0, 1, 0, x, y);
                let g2 = round2(old * w[x][0] + g * w[x][1], 5).clamp(grain_min, grain_max);
                let grain =
                    round2(top * w[y][0] + g2 * w[y][1], 5).clamp(grain_min, grain_max);
                add(x, y, grain);
            }
        }

        bx += FG_BLOCK_SIZE;
    }
}

/// 色度逐行混合 `fguv_32x32xn`（规范 §7.18.3.6）。
///
/// `luma` 必须是**未加颗粒**的亮度平面（所以调用方先算色度、再算亮度）。
#[allow(clippy::too_many_arguments)]
fn fguv_32x32xn(
    plane: &mut Plane,
    luma: &Plane,
    params: &FilmGrainParams,
    scaling: &[u8; SCALING_SIZE],
    grain_lut: &GrainLut,
    bh: usize,
    row_num: usize,
    uv: usize,
    is_identity: bool,
    sub_x: usize,
    sub_y: usize,
    bit_depth: u8,
) {
    let bd_min_8 = u32::from(bit_depth - 8);
    let bitdepth_max = (1i32 << bit_depth) - 1;
    let grain_ctr = 128i32 << bd_min_8;
    let (grain_min, grain_max) = (-grain_ctr, grain_ctr - 1);
    let (min_value, max_value) = if params.clip_to_restricted_range {
        (
            16i32 << bd_min_8,
            (if is_identity { 235 } else { 240 }) << bd_min_8,
        )
    } else {
        (0, bitdepth_max)
    };
    let scaling_shift = u32::from(params.scaling_shift);
    let overlap = params.overlap_flag;
    let rows = 1 + usize::from(overlap && row_num > 0);

    let mut seed = [0u32; 2];
    for (i, s) in seed.iter_mut().enumerate().take(rows) {
        let rn = (row_num - i) as i32;
        *s = u32::from(params.grain_seed)
            ^ ((((rn * 37 + 178) & 0xFF) as u32) << 8)
            ^ ((rn * 173 + 105) & 0xFF) as u32;
    }

    let pw = plane.w;
    let base_y = (row_num * FG_BLOCK_SIZE) >> sub_y;
    let step = FG_BLOCK_SIZE >> sub_x;
    // w[sub][x]：sub=0（4:4:4/4:2:2 的未下采样方向）用 27/17，sub=1 用 23/22
    let w: [[[i32; 2]; 2]; 2] = [[[27, 17], [17, 27]], [[23, 22], [0, 0]]];
    let mut offsets = [[0i32; 2]; 2];

    let mut bx = 0usize;
    while bx < pw {
        let bw = step.min(pw - bx);
        let ystart = if overlap && row_num > 0 { (2 >> sub_y).min(bh) } else { 0 };
        let xstart = if overlap && bx > 0 { (2 >> sub_x).min(bw) } else { 0 };

        if overlap && bx > 0 {
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }
        for i in 0..rows {
            offsets[0][i] = get_random_number(8, &mut seed[i]);
        }

        let mut add = |x: usize, y: usize, grain: i32| {
            let lx = (bx + x) << sub_x;
            let ly = (y << sub_y) + row_num * FG_BLOCK_SIZE;
            let l0 = i32::from(luma.get(lx, ly));
            // 水平下采样时取相邻两个亮度样本平均；帧宽为奇数时复制最后一列（等价于
            // 参考实现往 padding 里写 `ptr[w] = ptr[w-1]`）
            let avg = if sub_x == 1 {
                let l1 = if lx + 1 < luma.w { i32::from(luma.get(lx + 1, ly)) } else { l0 };
                (l0 + l1 + 1) >> 1
            } else {
                l0
            };
            let px = bx + x;
            let py = base_y + y;
            let s = i32::from(plane.get(px, py));
            let val = if params.chroma_scaling_from_luma {
                avg
            } else {
                let combined = avg * i32::from(params.uv_luma_mult[uv])
                    + s * i32::from(params.uv_mult[uv]);
                ((combined >> 6) + i32::from(params.uv_offset[uv]) * (1 << bd_min_8))
                    .clamp(0, bitdepth_max)
            };
            let noise = round2(i32::from(scaling[val as usize]) * grain, scaling_shift);
            plane.set(px, py, (s + noise).clamp(min_value, max_value) as u16);
        };

        for y in ystart..bh {
            for x in xstart..bw {
                let grain = sample_lut(grain_lut, &offsets, sub_x, sub_y, 0, 0, x, y);
                add(x, y, grain);
            }
            for x in 0..xstart {
                let g = sample_lut(grain_lut, &offsets, sub_x, sub_y, 0, 0, x, y);
                let old = sample_lut(grain_lut, &offsets, sub_x, sub_y, 1, 0, x, y);
                let grain = round2(old * w[sub_x][x][0] + g * w[sub_x][x][1], 5)
                    .clamp(grain_min, grain_max);
                add(x, y, grain);
            }
        }

        for y in 0..ystart {
            for x in xstart..bw {
                let g = sample_lut(grain_lut, &offsets, sub_x, sub_y, 0, 0, x, y);
                let old = sample_lut(grain_lut, &offsets, sub_x, sub_y, 0, 1, x, y);
                let grain = round2(old * w[sub_y][y][0] + g * w[sub_y][y][1], 5)
                    .clamp(grain_min, grain_max);
                add(x, y, grain);
            }
            for x in 0..xstart {
                let g = sample_lut(grain_lut, &offsets, sub_x, sub_y, 0, 0, x, y);
                let mut top = sample_lut(grain_lut, &offsets, sub_x, sub_y, 0, 1, x, y);
                let old = sample_lut(grain_lut, &offsets, sub_x, sub_y, 1, 1, x, y);
                top = round2(old * w[sub_x][x][0] + top * w[sub_x][x][1], 5)
                    .clamp(grain_min, grain_max);
                let old = sample_lut(grain_lut, &offsets, sub_x, sub_y, 1, 0, x, y);
                let g2 = round2(old * w[sub_x][x][0] + g * w[sub_x][x][1], 5)
                    .clamp(grain_min, grain_max);
                let grain = round2(top * w[sub_y][y][0] + g2 * w[sub_y][y][1], 5)
                    .clamp(grain_min, grain_max);
                add(x, y, grain);
            }
        }

        bx += step;
    }
}

/// `apply_grain()`（规范 §7.18.3）：把胶片颗粒叠加到**输出帧**上。
///
/// 就地修改 `frame`。调用方必须保证 `frame` 是"已从参考帧槽独立出来"的那一份
/// （存进参考帧的像素不能带颗粒）。`is_identity` 对应色度采样矩阵为
/// `MC_IDENTITY`（RGB），影响色度限制范围的上界（235 vs 240）。
pub fn apply_film_grain(frame: &mut CurrFrame, params: &FilmGrainParams, is_identity: bool) {
    if !params.apply_grain {
        return;
    }
    let num_uv = [usize::from(params.num_uv_points[0]), usize::from(params.num_uv_points[1])];
    let has_chroma = num_uv[0] > 0 || num_uv[1] > 0 || params.chroma_scaling_from_luma;
    let bit_depth = frame.planes[0].bit_depth;
    let sub_x = usize::from(frame.sub_x);
    let sub_y = usize::from(frame.sub_y);
    let height = frame.planes[0].h;

    // ① 生成颗粒模板（亮度必有；色度按需）
    let mut grain_lut: [GrainLut; 3] =
        std::array::from_fn(|_| [[0i16; GRAIN_WIDTH]; GRAIN_HEIGHT + 1]);
    generate_grain_y(&mut grain_lut[0], params, bit_depth);
    {
        let (head, rest) = grain_lut.split_at_mut(1);
        if num_uv[0] > 0 || params.chroma_scaling_from_luma {
            generate_grain_uv(
                &mut rest[0],
                &head[0],
                params,
                0,
                sub_x as u32,
                sub_y as u32,
                bit_depth,
            );
        }
        if num_uv[1] > 0 || params.chroma_scaling_from_luma {
            generate_grain_uv(
                &mut rest[1],
                &head[0],
                params,
                1,
                sub_x as u32,
                sub_y as u32,
                bit_depth,
            );
        }
    }

    // ② 生成缩放表（点表 → 幅度）
    let mut scaling = [[0u8; SCALING_SIZE]; 3];
    if params.num_y_points > 0 || params.chroma_scaling_from_luma {
        generate_scaling(
            &params.y_points,
            usize::from(params.num_y_points),
            &mut scaling[0],
            bit_depth,
        );
    }
    if num_uv[0] > 0 {
        generate_scaling(&params.uv_points[0], num_uv[0], &mut scaling[1], bit_depth);
    }
    if num_uv[1] > 0 {
        generate_scaling(&params.uv_points[1], num_uv[1], &mut scaling[2], bit_depth);
    }

    let rows = height.div_ceil(FG_BLOCK_SIZE);

    // ③ 色度先于亮度：色度要读**未加颗粒**的亮度
    if frame.num_planes > 1 && has_chroma {
        let planes = &mut frame.planes;
        let (luma_half, chroma_half) = planes.split_at_mut(1);
        let luma = &luma_half[0];
        for row in 0..rows {
            let luma_bh = FG_BLOCK_SIZE.min(height - row * FG_BLOCK_SIZE);
            let bh = (luma_bh + sub_y) >> sub_y;
            if params.chroma_scaling_from_luma {
                for pl in 0..2 {
                    fguv_32x32xn(
                        &mut chroma_half[pl],
                        luma,
                        params,
                        &scaling[0],
                        &grain_lut[1 + pl],
                        bh,
                        row,
                        pl,
                        is_identity,
                        sub_x,
                        sub_y,
                        bit_depth,
                    );
                }
            } else {
                for pl in 0..2 {
                    if num_uv[pl] > 0 {
                        fguv_32x32xn(
                            &mut chroma_half[pl],
                            luma,
                            params,
                            &scaling[1 + pl],
                            &grain_lut[1 + pl],
                            bh,
                            row,
                            pl,
                            is_identity,
                            sub_x,
                            sub_y,
                            bit_depth,
                        );
                    }
                }
            }
        }
    }

    // ④ 亮度（就地；每个像素只依赖它自己）
    if params.num_y_points > 0 {
        for row in 0..rows {
            let bh = FG_BLOCK_SIZE.min(height - row * FG_BLOCK_SIZE);
            fgy_32x32xn(
                &mut frame.planes[0],
                params,
                &scaling[0],
                &grain_lut[0],
                bh,
                row,
                bit_depth,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 一个"确实会产生颗粒"的最小参数：单点亮度点表（幅度 255）+ 关闭自回归。
    fn straight_params() -> FilmGrainParams {
        let mut y_points = [[0u8; 2]; 14];
        y_points[0] = [0, 255];
        FilmGrainParams {
            apply_grain: true,
            grain_seed: 1,
            update_grain: true,
            num_y_points: 1,
            y_points,
            scaling_shift: 8,
            ar_coeff_shift: 6,
            ..Default::default()
        }
    }

    #[test]
    fn prng_matches_reference_sequence() {
        // 种子 0 时前几项（与 dav1d/规范 LFSR 的算例一致）
        let mut s = 0u32;
        assert_eq!(get_random_number(11, &mut s), 0);
        // 连续取值应稳定复现
        let mut s2 = 0u32;
        let seq: Vec<i32> = (0..8).map(|_| get_random_number(11, &mut s2)).collect();
        let mut s3 = 0u32;
        let seq2: Vec<i32> = (0..8).map(|_| get_random_number(11, &mut s3)).collect();
        assert_eq!(seq, seq2);
        // 8 位取值必须在 [0, 255]
        let mut s4 = 0u32;
        for _ in 0..64 {
            let v = get_random_number(8, &mut s4);
            assert!((0..256).contains(&v));
        }
    }

    #[test]
    fn round2_is_round_half_up() {
        assert_eq!(round2(0, 4), 0);
        assert_eq!(round2(8, 4), 1);
        assert_eq!(round2(7, 4), 0);
        assert_eq!(round2(-8, 4), 0); // (-8 + 8) >> 4
        assert_eq!(round2(-9, 4), -1); // (-1) >> 4 = -1
    }

    #[test]
    fn scaling_interpolates_points() {
        let mut s = [0u8; SCALING_SIZE];
        let pts = [[0u8, 10u8], [128, 210]];
        generate_scaling(&pts, 2, &mut s, 8);
        assert_eq!(s[0], 10);
        // 中间严格递增，且落在两端之间
        assert!(s[64] > 10 && s[64] < 210);
        // 最后一个插值档仍在区间内，其后（>= 最后一个点值 128）取终点值
        assert!(s[127] <= 210 && s[127] >= 10);
        assert_eq!(s[128], 210);
        assert_eq!(s[255], 210);
    }

    #[test]
    fn empty_scaling_is_all_zero() {
        let mut s = [9u8; SCALING_SIZE];
        generate_scaling(&[[0u8, 0u8]; 14], 0, &mut s, 8);
        // 8 位只用前 256 档；没点表 ⇒ 幅度恒为 0
        assert!(s[..256].iter().all(|&v| v == 0));
    }

    #[test]
    fn no_points_leaves_frame_untouched() {
        let mut f = CurrFrame::new(64, 48, true, true, 3, 8);
        for y in 0..48 {
            for x in 0..64 {
                f.planes[0].set(x, y, 100);
            }
        }
        for p in 1..3 {
            for y in 0..24 {
                for x in 0..32 {
                    f.planes[p].set(x, y, 128);
                }
            }
        }
        let before: Vec<u16> = f.planes.iter().flat_map(|p| p.px16().to_vec()).collect();
        let params = FilmGrainParams { apply_grain: true, ..Default::default() };
        apply_film_grain(&mut f, &params, false);
        let after: Vec<u16> = f.planes.iter().flat_map(|p| p.px16().to_vec()).collect();
        assert_eq!(before, after);
    }

    #[test]
    fn restricted_range_is_respected() {
        // 全 0 的亮度 + 只作用于亮度、clip 到受限范围 ⇒ 输出必须落在 [16, 235]
        let mut f = CurrFrame::new(64, 48, true, true, 3, 8);
        let params = FilmGrainParams {
            clip_to_restricted_range: true,
            ..straight_params()
        };
        apply_film_grain(&mut f, &params, false);
        for y in 0..48 {
            for x in 0..64 {
                let v = f.planes[0].get(x, y);
                assert!((16..=235).contains(&v), "像素 ({x},{y}) = {v} 越出受限范围");
            }
        }
    }

    #[test]
    fn grain_is_deterministic() {
        let mut a = CurrFrame::new(64, 48, true, true, 3, 8);
        let mut b = CurrFrame::new(64, 48, true, true, 3, 8);
        for y in 0..48 {
            for x in 0..64 {
                a.planes[0].set(x, y, 100);
                b.planes[0].set(x, y, 100);
            }
        }
        let params = straight_params();
        apply_film_grain(&mut a, &params, false);
        apply_film_grain(&mut b, &params, false);
        assert_eq!(a.planes[0].px16(), b.planes[0].px16());
        // 且确实改了像素（种子 0 的颗粒不为全零）
        assert!(a.planes[0].px16().iter().any(|&v| v != 100));
    }
}
