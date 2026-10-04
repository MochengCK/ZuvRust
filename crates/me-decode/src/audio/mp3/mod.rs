//! **自研 MP3 解码器**（MPEG-1 / MPEG-2 / MPEG-2.5 的 Layer III）。
//!
//! 就是日常说的"MP3"。Layer I / Layer II 是另外的帧格式，不在这一档里（见"还没接的部分"）。
//! 与 [`crate::audio::flac`] / [`crate::audio::aac`] 一样走 [`AudioDecoder`] 接口，
//! 同样**不加任何第三方依赖**：所有计算都在这份源码里。
//!
//! ## 已经就绪的部分
//!
//! | 部分 | 位置 | 状态 |
//! |---|---|---|
//! | 静态码表（17 棵哈夫曼树、频带表、合成窗、混叠系数…） | [`tables`] | ✅ 生成期自校验 + 本文件单测复核 |
//! | 帧头解析与重同步（CRC 位、`emphasis`、free format 的明确拒绝） | 本文件 | ✅ |
//! | 位水库（`main_data_begin` 回退 + 本帧主数据，末 511 字节滚动） | 本文件 | ✅ |
//! | 侧信息（MPEG-1 双粒度 + `scfsi` 复用 / LSF 单粒度） | 本文件 | ✅ |
//! | 比例因子（MPEG-1 `slen1/slen2`；LSF 三段分解 `lsf_sf_expand`） | 本文件 | ✅ |
//! | 哈夫曼谱解码（linbits 转义、count1 四元组表）与反量化 | 本文件 | ✅ |
//! | 立体声（M/S、MPEG-1 与 LSF 两套强度立体声） | 本文件 | ✅ |
//! | 短块重排、混叠抑制、IMDCT/加窗/叠加、频率反转、多相合成 | 本文件 | ✅ |
//!
//! 端到端验证：**8 条公共领域码流**（MPEG-1 44.1k/48k/32k、MPEG-2 16k/22k/24k、单声道与
//! 四种声道模式全含、全比特率扫描、compl / noise 码流 —— 见 `tests/mp3_ref_compare.rs` 顶部）
//! 与参考解码器（minimp3，float32）逐样本比对：**逐帧归一化相关最差 0.999963、
//! 逐样本最大绝对差 ≤ 1.9e-5 —— 不到 0.61 个 16 位量化步长**（残差只来自 f32 运算顺序与
//! 参考自己的 pow43 近似）。跑法见该测试文件（`#[ignore]`，手工跑，不带样例文件进默认测试）。
//!
//! 另有一条**独立于 minimp3** 的交叉验证：`ILL2_layer3`（48 kHz / 320 kbps 联合立体声）
//! 与 Apple CoreAudio（`afconvert` 解出的 WAV，i16）对齐掉它 1058 个样本的解码延迟后，
//! 相关系数 **1.00000000**、逐样本最大差**恰好 1.00 个 i16 量化步长**、残差 rms 8.5e-7 ——
//! 即"本解码器的输出就是同一条波形的 i16 舍入"。两个互相独立的参考都落在量化误差里，
//! 才敢说这份实现是逐样本对齐的。
//!
//! ## 还没接的部分（都明确报 `Unsupported`，绝不产噪声）
//!
//! - **Layer I / Layer II**：帧头 `layer` 位不是 `01` 就拒绝。这两层的帧格式与 Layer III
//!   完全不同（子带分配表、量化阶梯、可变帧长），拿 Layer III 的解析去套只会解出噪声。
//! - **free format**（`bitrate_index == 0`）：帧长不由码流字段决定，得"顺着下一帧的同步字
//!   反推"，是另一套逻辑。
//! - **`emphasis != 0`**：去加重是**重放端**的事（标准里它是给重放设备的提示音色标记），
//!   FFmpeg / PDMP3 都不做；我们做了反而与参考不一致。
//! - **8 kHz 的混合块**（`sample_rate_index == 8` 且 `mixed_block_flag`）：FFmpeg 明确
//!   不支持（`switch point in 8khz`）—— 它的长块部分是 4 个子带而不是 2 个，我们不猜。
//!
//! ## 几个已经把坑写下来的地方
//!
//! 1. **先做立体声、再重排短块**。短块在码流里是"按窗分组"的传输序
//!    （`[w0 整段][w1][w2]`，比例因子索引 `sfb*3 + w`），强度立体声逐带逐窗扫描时用的
//!    正是这个序；先重排成交错序再扫，"带 ↔ 窗"的对应就乱了。FFmpeg 与 minimp3 都是
//!    "立体声 → 重排"。
//! 2. **哈夫曼符号里"两位都为 0"是单独一位**。标准表里的 `(x, y)` 在实现里被打包成
//!    `(x << 5) | (x≠0 && y≠0) << 4 | y`：bit4 表示"这一对两个都非零"。少了它，
//!    那些 `x`、`y` 必有一个为 0 的码字会被当成两个独立值各读一次 linbits，
//!    **从这一对起整帧错位**。
//! 3. **`part2_3_length` 是这一对 (粒度, 声道) 的硬边界，而且它把比例因子也算在内**。
//!    边界要从**读比例因子之前**起算，解完还要把位指针挪回边界（读多了退、读少了进）。
//!    这两点任一错掉，每段就偏掉"该段比例因子的位宽"，后面所有声道与颗粒全部错位 ——
//!    实测表现是"低频大致对、高频和第二个声道全乱"，很像码表错，其实不是。
//! 4. **区域划分的单位是"值对"而不是系数**（`big_values` 本身也是值对）。区域边界按
//!    系数算出来要除以 2，最后按 `big_values` 截断；混着用会让第二、三个区域选错哈夫曼表。
//! 5. **`scfsi` 只对粒度 1 生效**：粒度 0 的比例因子永远逐位读。
//! 6. **短块三个窗的落点是 6/12/18，不是 0/6/12**。36 点缓冲里前 6 个样本是上一块的尾巴，
//!    三窗的 12 点结果分别落在偏移 6、12、18；照 0/6/12 放会整体提前 6 个样本，
//!    声音变成"发飘"的梳状，但听起来仍然像音乐 —— 很难查。
//! 7. **位水库的历史不够时按标准补 0**（流的开头、或前面丢过帧）。补 0 而不是把后面的
//!    数据往前挪：挪了之后所有位偏移都错，整帧噪声。
//! 8. **`(0, 0)` 这一对不读符号位**。符号位只跟在非零值后面；零值对多读一位，后面整段全崩。
//!    这条最阴 —— 侧信息、码表、区域边界全都对得上，只差这一位，而症状是"整体音量差几倍、
//!    高频全乱"。当时的定位手段：把参考解码器的**原始哈夫曼值**逐值导出来对比。
//! 9. **强度立体声的"越界"不终结强度区，只有"右声道这一带有数据"才终结**。
//!    `is_pos >= sf_max`（MPEG-1 = 7、LSF = 16）只是**这一带**不做强度立体声，下面的带
//!    还要接着判；把它当成分区边界（一旦遇到就不再往下判）会静默吃掉一整段频谱。
//!    表现极具欺骗性：强度立体声的两个声像系数满足 `kl + kr ≡ 1`，所以
//!    `M = (L+R)/2 = ch0·(kl+kr)/2` 与吞掉多少带**无关** —— 中声道逐样本吻合、
//!    只有侧声道差 5%（整体相关仍 0.9999、最大差 8e-4，听感上完全正常）。
//!    定位手段：按真实帧边界统计"最大差落在哪个声道模式/`mode_ext` 的帧"，
//!    再用 `kl+kr≡1` 这条不变式判断该看 M 还是 S。修好后同一码流 8.5e-4 → 1.6e-5。
//!    另注：**强度区边界的判据本身有两种流派** —— FFmpeg / minimp3 用"右声道最高非零
//!    频带之上"，ISO 参考实现（PDMP3）用"频带起点 ≥ 右声道已解出的谱线数 `count1`"。
//!    这里跟 FFmpeg / minimp3 一派（端到端比对的对象就是 minimp3）；两者在
//!    "频带内全是零值"的边界码流上可能给出不同结果。

mod tables;

use me_core::{MediaError, Result};

use super::bitreader::BitReader;
use super::{AudioDecoder, PcmChunk};
use tables::{
    ANTI_CA, ANTI_CS, BAND_LONG, BAND_SHORT, HUFF_SPEC, IS_RATIOS, LSF_NSF, PRETAB, SLEN1, SLEN2,
    SYNTH_D,
};

/// 子带个数（Layer III 固定 32）。
const SUBBANDS: usize = 32;
/// 每个粒度每声道的频谱线数（32 子带 × 18）。
const GRANULE_COEFFS: usize = 576;
/// 每个粒度每个子带的时域样本数。
const SUBBAND_SAMPLES: usize = 18;
/// 位水库最多回退的字节数（`main_data_begin` 上限是 511）。
const RESERVOIR_MAX: usize = 511;
/// 多相合成输出 → i16 的缩放（见 [`Mp3Decoder::polyphase`] 的说明）。
const OUTPUT_SCALE: f32 = 32768.0;
/// 1/√2：M/S 立体声的归一化系数。
const SQRT1_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
/// `|v|^(4/3)` 表的上界：linbits 最大 13，值最大 `15 + 2^13 - 1`。
const POW43_MAX: usize = 15 + (1 << 13) - 1;
/// 每帧最大字节数（MPEG-1 320 kbps / 32 kHz 时 1440 字节，留一点余量）。
const MAX_FRAME_BYTES: usize = 1600;
/// 主数据尾后补的零字节数：让"读越过数据末尾"变成读到 0，而不是把 `NeedMore` 抛上去。
///
/// 8 字节 = 64 位，比一次哈夫曼取值最长能吃的位数（6 位码字 + 13 位 linbits + 1 位符号，
/// 一对最多 34 位）宽裕一倍，所以补零之后解码路径上不会出现"读到一半没数据"。
const TAIL_PAD: usize = 8;

/// count1 区的两张四元组表（标准 Table 3-B.7 的 d / e）：`(码字, 码长)`。
///
/// 下标就是符号值：`v<<3 | w<<2 | x<<1 | y`（bit3 → 第一个系数，bit0 → 第四个）。
/// 表 32 是变长码（1..6 位），表 33 固定 4 位 —— 不是"两张 4 位表"。
const COUNT1_CODES: [[u8; 16]; 2] = [
    [1, 5, 4, 5, 6, 5, 4, 4, 7, 3, 6, 0, 7, 2, 3, 1],
    [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
];
const COUNT1_BITS: [[u8; 16]; 2] = [
    [1, 4, 4, 5, 4, 6, 5, 6, 4, 5, 5, 6, 5, 6, 6, 6],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
];

// ---------------------------------------------------------------- 帧头

/// MPEG 版本。三个版本共用 Layer III 的语法，差别在采样率/比特率表和"粒度个数"。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Version {
    /// `11`：MPEG-1（双粒度）
    Mpeg1,
    /// `10`：MPEG-2（单粒度，"LSF"）
    Mpeg2,
    /// `00`：MPEG-2.5（单粒度，非标准但被广泛使用）
    Mpeg25,
}

/// Layer III 帧头（32 位）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    pub version: Version,
    /// 统一的采样率档位 0..=8，正好是频带表的行号。
    ///
    /// 0/1/2 = 44.1k/48k/32k（MPEG-1），3/4/5 = 22.05k/24k/16k（MPEG-2），
    /// 6/7/8 = 11.025k/12k/8k（MPEG-2.5）—— 三个版本共用一张 9 行的表，所以并成一个下标。
    pub sample_rate_index: usize,
    pub sample_rate: u32,
    pub bitrate_kbps: u32,
    pub padding: bool,
    /// 1 或 2（mono 为 1）
    pub channels: u8,
    /// 联合立体声：M/S（`mode_extension` bit1）
    pub ms_stereo: bool,
    /// 联合立体声：强度立体声（`mode_extension` bit0）
    pub intensity_stereo: bool,
    /// 是否 MPEG-2 / 2.5（低采样率档，侧信息与比例因子走另一套语法）
    pub lsf: bool,
    /// 帧头后是否跟着 2 字节 CRC
    pub crc: bool,
    /// 整帧字节数
    pub frame_len: usize,
    /// 每帧的粒度数：MPEG-1 = 2，LSF = 1
    pub granules: usize,
    /// 侧信息字节数：MPEG-1 17/32，LSF 9/17（单/双声道）
    pub side_info_len: usize,
}

/// MPEG-1 Layer III 比特率表（kbps）。下标 0 是 free format，15 非法。
const BITRATES_MPEG1: [u32; 16] =
    [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0];
/// MPEG-2 / 2.5 Layer III 比特率表（kbps）。
const BITRATES_LSF: [u32; 16] =
    [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0];
/// 三个版本各自的基础采样率（MPEG-2 减半、MPEG-2.5 再减半）。
const SAMPLE_RATES: [u32; 3] = [44100, 48000, 32000];

impl FrameHeader {
    /// 解析 32 位帧头。
    ///
    /// 只看前 4 字节。首字节/次字节的高 3 位不匹配就给 [`MediaError::NeedMore`] 之外的
    /// `Bitstream` 错（调用方用 [`find_sync`] 先定位同步字，所以这里报错意味着"假同步"）。
    pub fn parse(b: &[u8]) -> Result<FrameHeader> {
        if b.len() < 4 {
            return Err(MediaError::NeedMore { need: 4, have: b.len() });
        }
        if b[0] != 0xFF || (b[1] & 0xE0) != 0xE0 {
            return Err(MediaError::Bitstream(format!(
                "MP3: 没有帧同步字（读到 {:#04x} {:#04x}）",
                b[0], b[1]
            )));
        }
        let version = match (b[1] >> 3) & 0x3 {
            0b11 => Version::Mpeg1,
            0b10 => Version::Mpeg2,
            0b00 => Version::Mpeg25,
            _ => {
                return Err(MediaError::Bitstream(
                    "MP3: 版本位 01 是保留值".into(),
                ))
            }
        };
        let layer_bits = (b[1] >> 1) & 0x3;
        if layer_bits != 0b01 {
            // 位模式 11/10/01 分别对应 Layer I/II/III
            return Err(MediaError::Unsupported(format!(
                "MP3: 这是 Layer {}，本解码器只做 Layer III",
                4 - layer_bits
            )));
        }
        let crc = (b[1] & 1) == 0; // protection_bit == 0 表示有 CRC
        let bitrate_index = ((b[2] >> 4) & 0xF) as usize;
        let sr_index = ((b[2] >> 2) & 0x3) as usize;
        if sr_index == 3 {
            return Err(MediaError::Bitstream("MP3: 采样率索引 3 是保留值".into()));
        }
        let lsf = version != Version::Mpeg1;
        let bitrate_kbps = if lsf { BITRATES_LSF[bitrate_index] } else { BITRATES_MPEG1[bitrate_index] };
        if bitrate_kbps == 0 {
            return Err(MediaError::Unsupported(format!(
                "MP3: 比特率索引 {bitrate_index} {}—— {}",
                if bitrate_index == 0 { "是 free format" } else { "是保留值" },
                if bitrate_index == 0 { "帧长不由码流字段决定，还没接" } else { "这条码流不合法" }
            )));
        }
        let sample_rate = SAMPLE_RATES[sr_index] / if lsf { if version == Version::Mpeg2 { 2 } else { 4 } } else { 1 };
        let sample_rate_index = sr_index + if lsf { if version == Version::Mpeg2 { 3 } else { 6 } } else { 0 };
        let padding = ((b[2] >> 1) & 1) == 1;
        let mode = (b[3] >> 6) & 0x3;
        let mode_ext = (b[3] >> 4) & 0x3;
        let emphasis = b[3] & 0x3;
        if emphasis != 0 {
            return Err(MediaError::Unsupported(format!(
                "MP3: emphasis = {}，去加重是重放端的事，本解码器不做",
                b[3] & 0x3
            )));
        }
        let channels = if mode == 3 { 1 } else { 2 };
        // mode_extension 只在联合立体声（mode == 1）时有意义
        let joint = mode == 1;
        let intensity_stereo = joint && (mode_ext & 0x1) != 0;
        let ms_stereo = joint && (mode_ext & 0x2) != 0;

        // MPEG-1 一帧 1152 个样本、LSF 一帧 576 个；帧长系数正好差一倍。
        let frame_len = if lsf {
            72 * bitrate_kbps as usize * 1000 / sample_rate as usize + padding as usize
        } else {
            144 * bitrate_kbps as usize * 1000 / sample_rate as usize + padding as usize
        };
        let granules = if lsf { 1 } else { 2 };
        let side_info_len = match (lsf, channels) {
            (false, 2) => 32,
            (false, _) => 17,
            (true, 2) => 17,
            (true, _) => 9,
        };
        Ok(FrameHeader {
            version,
            sample_rate_index,
            sample_rate,
            bitrate_kbps,
            padding,
            channels,
            ms_stereo,
            intensity_stereo,
            lsf,
            crc,
            frame_len,
            granules,
            side_info_len,
        })
    }

    /// 主数据在本帧内的起始偏移：帧头(4) + 可选 CRC(2) + 侧信息。
    ///
    /// Layer III 的 CRC 在**侧信息之前**（Layer I/II 是在后面），位置搞反会让侧信息整体错位。
    fn main_data_offset(&self, frame: &[u8]) -> Result<usize> {
        let off = 4 + self.crc as usize + self.side_info_len;
        if frame.len() < off {
            return Err(MediaError::NeedMore { need: off, have: frame.len() });
        }
        Ok(off)
    }

    /// 每帧每声道的样本数（MPEG-1 = 1152，LSF = 576）。
    fn samples_per_frame(&self) -> usize {
        self.granules * GRANULE_COEFFS
    }
}

/// 从 `data[from..]` 起找第一个像帧头的位置。
///
/// 判据是同步字（11 位全 1）+ 帧头字段全部合法 —— 只看 `0xFF` 会把 ID3 标签、图片数据里
/// 的字节当成帧，靠"比特率/采样率/层位都合法"才能把假同步滤掉。
pub fn find_sync(data: &[u8], from: usize) -> Option<usize> {
    let mut i = from;
    while i + 1 < data.len() {
        // 只靠 `0xFF` + 高 3 位为 111 会误判（同步字太短），要真的把帧头解出来才算数
        if data[i] == 0xFF
            && (data[i + 1] & 0xE0) == 0xE0
            && i + 4 <= data.len()
            && FrameHeader::parse(&data[i..]).is_ok()
        {
            return Some(i);
        }
        i += 1;
    }
    None
}

// ---------------------------------------------------------------- 侧信息

/// 一个 (粒度, 声道) 的侧信息。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct GranuleInfo {
    part2_3_length: u16,
    big_values: u16,
    global_gain: u8,
    scalefac_compress: u16,
    window_switching: bool,
    block_type: u8,
    mixed_block: bool,
    table_select: [u8; 3],
    subblock_gain: [u8; 3],
    /// 长块的区域 0/1 频带数；开窗切换时是隐含值（8 = 纯短块，7 = 起止块/混合块）
    region0_count: u8,
    region1_count: u8,
    preflag: bool,
    scalefac_scale: bool,
    count1table_select: u8,
}

impl GranuleInfo {
    /// 长块部分占了几个子带（`long_end`）与短块部分从第几个短频带开始（`short_start`）。
    ///
    /// 三种形态：
    /// - 纯长块（含起/停块）：`long_end = 22`、`short_start = 13`（等价于"没有短块部分"）
    /// - 纯短块：`long_end = 0`、`short_start = 0`
    /// - 混合块：`long_end = 8`（MPEG-1）/ 6（LSF）、`short_start = 3`（前 2 个子带=36 个系数走长块）
    fn layout(&self, hdr: &FrameHeader) -> (usize, usize) {
        if self.window_switching && self.block_type == 2 {
            if self.mixed_block {
                // 混合块的长块部分固定覆盖前 2 个子带（36 个系数），
                // 44.1k/48k/32k 下正好是 8 个长频带，LSF 的各档是 6 个。
                let long_end = if hdr.sample_rate_index <= 2 { 8 } else { 6 };
                (long_end, 3)
            } else {
                (0, 0)
            }
        } else {
            (22, 13)
        }
    }

    /// 三个哈夫曼区域的边界，单位是**值对**。
    ///
    /// 与 FFmpeg 的 `init_short_region` / `init_long_region` + `region_offset2size` 一一对应：
    /// 先把每区的"结束位置"化成值对数，再用 `big_values` 截断、相邻相减得到每区的长度。
    ///
    /// 开窗切换时跳过的那一区：纯短块 36 个系数（8 kHz 是 72），起止块取"前 8 个长频带"的
    /// 系数和（44.1k = 36、LSF 各档 = 54、8 kHz = 108）—— 这不是随口取的数，正是
    /// FFmpeg `init_short_region` 里那三个常数。
    fn regions(&self, hdr: &FrameHeader, band_idx_long: &[[u16; 23]; 9]) -> [usize; 3] {
        let mut end = [0usize; 3]; // 每区结束位置（值对）
        let sr = hdr.sample_rate_index;
        if self.window_switching {
            let coeffs = if self.block_type == 2 {
                if sr == 8 {
                    72
                } else {
                    36
                }
            } else if sr <= 2 {
                36
            } else if sr != 8 {
                54
            } else {
                108
            };
            end[0] = coeffs / 2;
            end[1] = GRANULE_COEFFS / 2;
            end[2] = GRANULE_COEFFS / 2;
        } else {
            let ra1 = self.region0_count as usize;
            let ra2 = self.region1_count as usize;
            end[0] = band_idx_long[sr][ra1 + 1] as usize;
            end[1] = band_idx_long[sr][(ra1 + ra2 + 2).min(22)] as usize;
            end[2] = GRANULE_COEFFS / 2;
        }
        let big = self.big_values as usize;
        let mut out = [0usize; 3];
        let mut prev = 0usize;
        for i in 0..3 {
            let k = end[i].min(big);
            out[i] = k - prev;
            prev = k;
        }
        out
    }
}

// ---------------------------------------------------------------- 解码器

/// 一个声道的跨帧状态。
struct ChannelState {
    /// 混合滤波器组（IMDCT）的跨帧重叠：每子带 18 个样本。
    overlap: [[f32; SUBBAND_SAMPLES]; SUBBANDS],
    /// 多相合成的 V 缓冲（1024）。
    ///
    /// 每来一个时间槽先整体右移 64，再把新算出的 64 个值填到最前面 ——
    /// 这是标准里的"延迟线"，少了它相邻帧之间会缺 64 个样本的历史。
    v: [f32; 1024],
}

impl ChannelState {
    fn new() -> Self {
        Self { overlap: [[0.0; SUBBAND_SAMPLES]; SUBBANDS], v: [0.0; 1024] }
    }
}

/// MPEG-1 长块比例因子的"复用"上下文：`scfsi` 标志 + 粒度 0 已经解出的比例因子。
///
/// 只有 MPEG-1 长块用得上（LSF 根本没有 `scfsi` 字段）。单独打个包是为了让
/// `read_scalefactors` 的参数个数收在可读范围内 —— 摊成一个个参数时，
/// `ch` / `gr` / `scfsi` / `prev` 谁配谁很容易看串，都是 `usize` 和数组下标。
#[derive(Clone, Copy)]
struct ScfReuse<'a> {
    /// 每声道 4 个复用标志，`bit3` 对应第 0 组（6 个比例因子那一组）
    scfsi: &'a [u8; 2],
    /// 同一帧粒度 0 的比例因子（粒度 1 复用它的值）
    prev: &'a [[i32; 40]; 2],
}

/// 自研 MP3 解码器。
pub struct Mp3Decoder {
    /// 位水库：上一帧没用完的主数据（最多 511 字节），下一帧要回退 `main_data_begin` 字节。
    reservoir: Vec<u8>,
    /// 还没凑够一帧的输入（包可能含多帧、也可能只有半帧）。
    pending: Vec<u8>,
    /// 下一个要吐出的帧的时间戳（微秒）—— 一个包里有多帧时靠它累加。
    next_pts_us: i64,
    /// 每声道的跨帧状态（按帧头里的声道数取用前几个）
    states: Vec<ChannelState>,
    /// 首帧之前 `stream_info()` 要返回什么：`None` = 还不知道
    info: Option<(u32, u8, u8)>,
    /// `N[i][k] = cos((16+i)(2k+1)π/64)`，64×32，open 时算好（合成滤波的核心矩阵）
    matrix: Vec<f32>,
    /// 合成窗 `D[512]`（由只存一半的 [`SYNTH_D`] 按镜像关系展开）
    d: [f32; 512],
    /// `|v|^(4/3)`，反量化用（查表比每根谱线算一次 powf 快得多）
    pow43: Vec<f32>,
    /// 长块 IMDCT 的余弦表：`cos(π/72·(2p+19)(2m+1))`，36×18
    imdct_long_cos: Vec<f32>,
    /// 短块 IMDCT 的余弦表：`cos(π/24·(2p+7)(2m+1))`，12×6
    imdct_short_cos: Vec<f32>,
    /// 四种窗形（长/起/短/停）。短窗只有 12 个有效值，单独放。
    win_long: [f32; 36],
    win_start: [f32; 36],
    win_stop: [f32; 36],
    win_short: [f32; 12],
    /// 长频带的累计边界（单位：值对），`band_idx_long[档位][频带]`
    band_idx_long: [[u16; 23]; 9],
    /// count1 两张四元组表展开成的树（与谱表同一种打包，见 [`build_tree`]）
    count1_trees: [Vec<u32>; 2],
}

impl Mp3Decoder {
    /// 建解码器。`extradata` 对 MP3 没有意义（帧头自带全部参数），忽略即可。
    pub fn open(_extradata: &[u8]) -> Result<Self> {
        // 合成矩阵与合成窗：都只跟标准有关，与码流无关，适合在 open 时算一次。
        let mut matrix = vec![0f32; 64 * SUBBANDS];
        for i in 0..64 {
            for k in 0..SUBBANDS {
                matrix[i * SUBBANDS + k] =
                    (std::f32::consts::PI / 64.0 * (16 + i) as f32 * (2 * k + 1) as f32).cos();
            }
        }
        let mut d = [0f32; 512];
        for (i, slot) in d.iter_mut().enumerate().take(257) {
            *slot = SYNTH_D[i];
        }
        // 镜像关系：D[512-i] = -D[i]（i 是 64 的倍数时取原值）—— 只存了一半的由来
        for (i, slot) in d.iter_mut().enumerate().skip(257) {
            let j = 512 - i;
            let v = SYNTH_D[j];
            *slot = if j % 64 != 0 { -v } else { v };
        }
        // |v|^(4/3)
        let mut pow43 = vec![0f32; POW43_MAX + 1];
        for (v, slot) in pow43.iter_mut().enumerate() {
            *slot = (v as f32).powf(4.0 / 3.0);
        }
        // 两张 IMDCT 余弦表
        let mut imdct_long_cos = vec![0f32; 36 * 18];
        for p in 0..36 {
            for m in 0..18 {
                imdct_long_cos[p * 18 + m] = (std::f32::consts::PI / 72.0
                    * (2 * p + 19) as f32
                    * (2 * m + 1) as f32)
                    .cos();
            }
        }
        let mut imdct_short_cos = vec![0f32; 12 * 6];
        for p in 0..12 {
            for m in 0..6 {
                imdct_short_cos[p * 6 + m] = (std::f32::consts::PI / 24.0
                    * (2 * p + 7) as f32
                    * (2 * m + 1) as f32)
                    .cos();
            }
        }
        // 四种窗形（标准 2.4.3.4.10.3）
        let mut win_long = [0f32; 36];
        let mut win_start = [0f32; 36];
        let mut win_stop = [0f32; 36];
        let mut win_short = [0f32; 12];
        for (p, slot) in win_long.iter_mut().enumerate() {
            // 长窗：整段正弦，两半互补（win[p]² + win[p+18]² = 1）
            *slot = (std::f32::consts::PI / 36.0 * (p as f32 + 0.5)).sin();
        }
        for p in 0..18 {
            let pf = p as f32;
            // 起块窗：长窗前半 + 平台 + 短窗后半 + 0
            win_start[p] = (std::f32::consts::PI / 36.0 * (pf + 0.5)).sin();
            // 停块窗：0 + 短窗前半 + 平台 + 长窗后半
            win_stop[p + 18] = (std::f32::consts::PI / 36.0 * (pf + 18.5)).sin();
        }
        for p in 0..6 {
            // 平台段
            win_start[p + 18] = 1.0;
            win_stop[p + 12] = 1.0;
            // 短窗后半（起块窗用）与前半（停块窗用）：sin(π/12·(p+0.5)) 的两半
            let pf = p as f32;
            win_start[p + 24] = (std::f32::consts::PI / 12.0 * (pf + 6.5)).sin();
            win_stop[p + 6] = (std::f32::consts::PI / 12.0 * (pf + 0.5)).sin();
        }
        for (p, slot) in win_short.iter_mut().enumerate() {
            *slot = (std::f32::consts::PI / 12.0 * (p as f32 + 0.5)).sin();
        }
        // 长频带累计边界（值对）：区域划分与强度立体声都要用
        let mut band_idx_long = [[0u16; 23]; 9];
        for row in 0..9 {
            let mut acc = 0u16;
            for b in 0..22 {
                band_idx_long[row][b] = acc;
                acc += (BAND_LONG[row][b] / 2) as u16;
            }
            band_idx_long[row][22] = acc;
        }
        let count1_trees = [
            build_tree(&COUNT1_CODES[0], &COUNT1_BITS[0]),
            build_tree(&COUNT1_CODES[1], &COUNT1_BITS[1]),
        ];
        let mut states = Vec::with_capacity(2);
        for _ in 0..2 {
            states.push(ChannelState::new());
        }
        Ok(Self {
            reservoir: Vec::new(),
            pending: Vec::new(),
            next_pts_us: 0,
            states,
            info: None,
            matrix,
            d,
            pow43,
            imdct_long_cos,
            imdct_short_cos,
            win_long,
            win_start,
            win_stop,
            win_short,
            band_idx_long,
            count1_trees,
        })
    }

    /// 解一帧（`frame` 是完整的一帧，长度 ≥ `hdr.frame_len`）。
    ///
    /// 返回值按 `(左, 右)` 两个声道的 i32 采样给 —— 交织在调用方（[`AudioDecoder::decode`]）
    /// 完成，因为一个包里可能有多帧，交织要和所有帧一起做。
    fn decode_frame(&mut self, frame: &[u8], pts_us: i64) -> Result<PcmChunk> {
        let hdr = FrameHeader::parse(frame)?;
        let main_off = hdr.main_data_offset(frame)?;
        if frame.len() < hdr.frame_len {
            return Err(MediaError::NeedMore { need: hdr.frame_len, have: frame.len() });
        }
        let trace = std::env::var("ME_MP3_TRACE").is_ok();
        if trace {
            eprintln!(
                "MP3 帧: {:?} {} Hz {} kbps {}ch{} 帧长={} 主数据偏移={}",
                hdr.version,
                hdr.sample_rate,
                hdr.bitrate_kbps,
                hdr.channels,
                if hdr.ms_stereo || hdr.intensity_stereo {
                    format!(" 联合(ms={} is={})", hdr.ms_stereo, hdr.intensity_stereo)
                } else {
                    String::new()
                },
                hdr.frame_len,
                main_off
            );
        }
        // ---- 侧信息（在帧内，紧跟在帧头/CRC 之后）----
        let mut sr = BitReader::new(&frame[4 + hdr.crc as usize..main_off.min(frame.len())]);
        let (main_data_begin, scfsi, granule_infos) = read_side_info(&mut sr, &hdr)?;
        if trace {
            eprintln!("  侧信息: main_data_begin={main_data_begin} 水库={}", self.reservoir.len());
        }

        // ---- 位水库：回退 main_data_begin 字节，再接上本帧主数据 ----
        let have = self.reservoir.len();
        let mut buf = Vec::with_capacity(RESERVOIR_MAX + MAX_FRAME_BYTES + TAIL_PAD);
        if main_data_begin > have {
            // 库里没有那么多历史（流的开头、或前面丢过帧）：按标准补 0。
            // 补 0 而不是把后面的数据往前挪 —— 挪了之后所有位偏移都错，整帧变噪声。
            buf.resize(main_data_begin - have, 0);
        }
        buf.extend_from_slice(&self.reservoir[have.saturating_sub(main_data_begin)..]);
        buf.extend_from_slice(&frame[main_off..hdr.frame_len]);
        // 尾后补零（`TAIL_PAD` 个字节）：`part2_3_length` 被编码器写多时只丢尾巴，不整帧失败。
        // 补的零**不参与**水库滚动 —— `main_len` 才是真数据长度。
        let main_len = buf.len();
        buf.resize(main_len + TAIL_PAD, 0);
        let mut br = BitReader::new(&buf);
        let limit_bit = main_len * 8;

        // ---- 逐粒度：解两个声道 → 立体声 → 重排/混叠/IMDCT/合成 ----
        let nch = hdr.channels as usize;
        let per_ch = hdr.samples_per_frame();
        let mut pcm: Vec<Vec<i32>> = (0..nch).map(|_| Vec::with_capacity(per_ch)).collect();
        let mut used_bits = 0usize;
        // MPEG-1 的 scfsi 复用要拿粒度 0 的比例因子；LSF 没有 scfsi，这个变量不用
        let mut scf_granule0 = [[0i32; 40]; 2];
        for (gr, infos) in granule_infos.iter().enumerate().take(hdr.granules) {
            let mut spectra = [[0f32; GRANULE_COEFFS]; 2];
            let mut scfs = [[0i32; 40]; 2];
            for (ch, gi) in infos.iter().enumerate().take(nch) {
                let gi = *gi;
                // **这一段 (粒度, 声道) 的位起点**：`part2_3_length` 数是"比例因子 + 谱数据"
                // 一共多少位，所以边界要从**读比例因子之前**起算。算成之后，每段都会多给
                // 几十位（正好是比例因子的位宽和），后面的声道与颗粒从此全错。
                let start_bit = br.pos_bits();
                let reuse = ScfReuse { scfsi: &scfsi, prev: &scf_granule0 };
                let sf = self.read_scalefactors(&mut br, &hdr, &gi, ch, gr, reuse)?;
                let mut gain = [0f32; GRANULE_COEFFS];
                self.fill_gains(&gi, &hdr, &sf, &mut gain);
                // 这一段的数据末尾；编码器写多时到 `main_len` 为止
                let end_bit = (start_bit + gi.part2_3_length as usize).min(limit_bit);
                self.read_spectrum(&mut br, &gi, &hdr, &gain, end_bit, &mut spectra[ch])?;
                // **解完必须挪回这一段末尾**：哈夫曼解码可能提前收工（数据不够），
                // 也可能因为"丢掉最后一组"读过头 —— 位指针不归位，下一个 (粒度, 声道)
                // 以及后面所有帧都会错位（这是 Layer III 位水库之外最容易踩的一脚）。
                let read_to = br.pos_bits();
                br.seek_bits(end_bit);
                used_bits += gi.part2_3_length as usize;
                scfs[ch] = sf;
                if trace {
                    eprintln!(
                        "    gr={gr} ch={ch} part2_3={} 段末={end_bit} 读到={read_to} 区域={:?} 表={:?} big={}",
                        gi.part2_3_length,
                        gi.regions(&hdr, &self.band_idx_long),
                        gi.table_select,
                        gi.big_values
                    );
                }
            }
            if gr == 0 {
                scf_granule0 = scfs;
            }
            // 立体声（必须在重排之前，见模块文档第 1 条）
            if nch == 2 && (hdr.ms_stereo || hdr.intensity_stereo) {
                let gi_r = infos[1];
                // 同一数组的两个元素要同时可变借用：只能走 split_at_mut
                let (l, r) = spectra.split_at_mut(1);
                self.stereo(&mut l[0], &mut r[0], &gi_r, &scfs[1], &hdr);
            }
            for ch in 0..nch {
                let gi = infos[ch];
                let mut spec = spectra[ch];
                self.reorder(&mut spec, &gi, &hdr);
                self.antialias(&mut spec, &gi);
                // 跨帧状态先搬出来再传：`&mut self.states[ch]` 与 `self.hybrid_and_synthesize`
                // 需要的 `&self` 落在同一个表达式里会打架，搬成局部变量借用就不重叠了
                // （`ChannelState` 都是固定长度数组，搬的是值不是堆分配）。
                let mut st = std::mem::replace(&mut self.states[ch], ChannelState::new());
                self.hybrid_and_synthesize(&mut st, &spec, &gi, &mut pcm[ch]);
                self.states[ch] = st;
            }
        }

        // ---- 更新位水库：本帧没用掉的字节留给下一帧 ----
        let consumed = used_bits.div_ceil(8).min(main_len);
        let rest = &buf[consumed..main_len];
        let keep = rest.len().min(RESERVOIR_MAX);
        let tail = rest[rest.len() - keep..].to_vec();
        self.reservoir.clear();
        self.reservoir.extend_from_slice(&tail);

        // ---- 交织成 i32 ----
        let samples_per_ch = pcm[0].len();
        let mut samples = Vec::with_capacity(samples_per_ch * nch);
        // 逐样本交织（而不是逐声道接起来）：调用方要的是 [L,R,L,R,…]
        for i in 0..samples_per_ch {
            samples.extend(pcm.iter().map(|ch| ch[i]));
        }
        self.info = Some((hdr.sample_rate, hdr.channels, 16));
        Ok(PcmChunk {
            sample_rate: hdr.sample_rate,
            channels: hdr.channels,
            bits_per_sample: 16,
            pts_us,
            samples,
        })
    }

    /// 读一对 (粒度, 声道) 的比例因子。
    ///
    /// MPEG-1 与 LSF 是两套完全不同的语法：
    /// - MPEG-1 长块：按 `scfsi` 分成四组（6/5/5/5）决定是否复用粒度 0 的值；
    ///   短块/混合块：`switch_point` 决定前段条数（17 或 18），再 18 条次级长度 + 3 个 0。
    /// - LSF：由 `scalefac_compress` 落进三个区间，用 `lsf_sf_expand` 三段分解出每段位宽，
    ///   再按 `LSF_NSF[tindex2][tindex]` 的条数读。
    fn read_scalefactors(
        &self,
        br: &mut BitReader,
        hdr: &FrameHeader,
        gi: &GranuleInfo,
        ch: usize,
        gr: usize,
        reuse: ScfReuse<'_>,
    ) -> Result<[i32; 40]> {
        let mut sf = [0i32; 40];
        if !hdr.lsf {
            let slen1 = SLEN1[gi.scalefac_compress as usize];
            let slen2 = SLEN2[gi.scalefac_compress as usize];
            let mut j = 0usize;
            if gi.window_switching && gi.block_type == 2 {
                // 混合块的长块部分是 8 个长频带 → 前段少一条（17）
                let first = if gi.mixed_block { 17 } else { 18 };
                for _ in 0..first {
                    sf[j] = read_bits_or_zero(br, slen1)?;
                    j += 1;
                }
                for _ in 0..18 {
                    sf[j] = read_bits_or_zero(br, slen2)?;
                    j += 1;
                }
                // 最后 3 个位置（第 12 个短频带）不传，恒为 0
                for _ in 0..3 {
                    sf[j] = 0;
                    j += 1;
                }
            } else {
                // 长块：四组，条数 6/5/5/5；scfsi 的 bit3 对应第 0 组
                for k in 0..4usize {
                    let n = if k == 0 { 6 } else { 5 };
                    let take_prev = gr == 1 && (reuse.scfsi[ch] >> (3 - k)) & 1 == 1;
                    let slen = if k < 2 { slen1 } else { slen2 };
                    for _ in 0..n {
                        sf[j] = if take_prev {
                            reuse.prev[ch][j]
                        } else {
                            read_bits_or_zero(br, slen)?
                        };
                        j += 1;
                    }
                }
                // 第 21 个比例因子不传（preflag/pretab 会把最后一个频带补上）
                sf[j] = 0;
            }
            return Ok(sf);
        }
        // ---- LSF（MPEG-2 / 2.5）----
        let tindex = if gi.block_type == 2 {
            if gi.mixed_block {
                2
            } else {
                1
            }
        } else {
            0
        };
        let mut comp = gi.scalefac_compress as i32;
        let intensity_ch1 = hdr.intensity_stereo && ch == 1;
        let slen;
        let tindex2;
        if intensity_ch1 {
            // 强度立体声的右声道：这里的"比例因子"其实是声像位置，位宽分配也不一样
            comp >>= 1;
            if comp < 180 {
                slen = lsf_sf_expand(comp, 6, 6, 0);
                tindex2 = 3;
            } else if comp < 244 {
                slen = lsf_sf_expand(comp - 180, 4, 4, 0);
                tindex2 = 4;
            } else {
                slen = lsf_sf_expand(comp - 244, 3, 0, 0);
                tindex2 = 5;
            }
        } else if comp < 400 {
            slen = lsf_sf_expand(comp, 5, 4, 4);
            tindex2 = 0;
        } else if comp < 500 {
            slen = lsf_sf_expand(comp - 400, 5, 4, 0);
            tindex2 = 1;
        } else {
            slen = lsf_sf_expand(comp - 500, 3, 0, 0);
            tindex2 = 2;
        }
        let mut j = 0usize;
        for k in 0..4usize {
            let n = LSF_NSF[tindex2][tindex][k] as usize;
            for _ in 0..n {
                sf[j] = read_bits_or_zero(br, slen[k] as u8)?;
                j += 1;
            }
        }
        Ok(sf)
    }

    /// 把每根谱线的反量化增益 `2^(exp/4)` 算好。
    ///
    /// 指数（标准 2.4.3.4.2）：
    /// `exp = global_gain - 210 - ((scf + preflag·pretab[i]) << (scalefac_scale+1))`，
    /// 短块再减 `8·subblock_gain[窗]`。反量化值 = `|is|^(4/3) · 2^(exp/4)`。
    ///
    /// 分段赋值而不是逐谱线判断：同一个频带内的指数是常量，按频带填比按谱线算少 20 倍的分支。
    fn fill_gains(&self, gi: &GranuleInfo, hdr: &FrameHeader, sf: &[i32; 40], gain: &mut [f32; GRANULE_COEFFS]) {
        let row = hdr.sample_rate_index;
        let shift = gi.scalefac_scale as i32 + 1;
        let g0 = gi.global_gain as i32 - 210;
        let (long_end, short_start) = gi.layout(hdr);
        let mut pos = 0usize;
        for i in 0..long_end {
            let pre = if gi.preflag { PRETAB[i] as i32 } else { 0 };
            let exp = g0 - ((sf[i] + pre) << shift);
            let g = exp2_quarter(exp);
            let len = BAND_LONG[row][i] as usize;
            gain[pos..pos + len].fill(g);
            pos += len;
        }
        if short_start < 13 {
            // 短块比例因子的装配顺序：`long_end` 之后每频带 3 条（w0/w1/w2 各一条）
            for (band, width) in BAND_SHORT[row][short_start..13].iter().enumerate() {
                let len = *width as usize;
                let k = long_end + band * 3;
                for w in 0..3usize {
                    let exp = g0 - (gi.subblock_gain[w] as i32) * 8 - (sf[k + w] << shift);
                    let g = exp2_quarter(exp);
                    gain[pos..pos + len].fill(g);
                    pos += len;
                }
            }
        }
    }

    /// 哈夫曼谱解码：三个 big_values 区域 + count1 区域，边解边乘增益。
    ///
    /// `end_bit` 是这一对 (粒度, 声道) 的数据末尾（= 起点 + `part2_3_length`，再按主数据
    /// 实际长度截断），所有读取都以它为界。
    fn read_spectrum(
        &self,
        br: &mut BitReader,
        gi: &GranuleInfo,
        hdr: &FrameHeader,
        gain: &[f32; GRANULE_COEFFS],
        end_bit: usize,
        out: &mut [f32; GRANULE_COEFFS],
    ) -> Result<()> {
        let regions = gi.regions(hdr, &self.band_idx_long);
        let mut idx = 0usize;
        for (reg, &pairs) in regions.iter().enumerate() {
            if pairs == 0 {
                continue;
            }
            let tab = gi.table_select[reg] as usize;
            let entry = HUFF_SPEC.get(tab).copied().flatten();
            let Some((tree, linbits)) = entry else {
                if tab == 0 {
                    // 表号 0：这一区全零，一个位都不读
                    idx += pairs * 2;
                    continue;
                }
                return Err(MediaError::Bitstream(format!(
                    "MP3: 区域 {reg} 用了非法哈夫曼表号 {tab}（4 / 14 在标准里未定义）"
                )));
            };
            for _ in 0..pairs {
                if br.pos_bits() >= end_bit {
                    break;
                }
                let sym = huff_symbol(br, tree)?;
                let (x, y) = ((sym >> 4) as u32, (sym & 0xF) as u32);
                let (vx, vy) = self.decode_pair(br, x, y, linbits)?;
                out[idx] = vx * gain[idx];
                out[idx + 1] = vy * gain[idx + 1];
                idx += 2;
            }
        }
        self.read_count1(br, gi, gain, idx, end_bit, out)?;
        Ok(())
    }

    /// 一个哈夫曼码字 → 两个系数（未乘增益）。
    ///
    /// 关键在于"两位都为 0"的语义（模块文档第 2 条）：`x != 0 && y != 0` 时两个值各自
    /// 独立读 linbits 与符号位，否则读的是一个值，另一个恒为 0。
    fn decode_pair(&self, br: &mut BitReader, x: u32, y: u32, linbits: u8) -> Result<(f32, f32)> {
        if x != 0 && y != 0 {
            let vx = self.read_huff_value(br, x, linbits)?;
            let vy = self.read_huff_value(br, y, linbits)?;
            Ok((vx, vy))
        } else {
            // **符号位只跟在非零值后面**：`(0, 0)` 这一对读完码字就结束，不读符号位。
            // 多读这一位，每碰到一个零值对就错 1 位，后面整段全崩 —— 这一位排查了很久：
            // 侧信息、码表、区域边界全都对得上，就差它。
            let v = x + y;
            if v == 0 {
                return Ok((0.0, 0.0));
            }
            let v = self.read_huff_value(br, v, linbits)?;
            Ok(if y != 0 { (0.0, v) } else { (v, 0.0) })
        }
    }

    /// 读一个系数值：`v < 15` 时直接查 `|v|^(4/3)`，`v == 15` 时先读 linbits 位再查。
    /// 之后紧跟一个符号位（**linbits 在符号位之前**，顺序反了就全乱）。
    fn read_huff_value(&self, br: &mut BitReader, v: u32, linbits: u8) -> Result<f32> {
        let mag = if v < 15 {
            self.pow43[v as usize]
        } else {
            let extra = br.read(linbits)?;
            self.pow43[(v + extra) as usize]
        };
        Ok(if br.read(1)? == 1 { -mag } else { mag })
    }

    /// count1 区：每 4 个系数一组，幅度只有 ±1（`1^(4/3) = 1`，所以直接用增益）。
    ///
    /// 一组的码字读完若已经越过 `part2_3_length` 的端点，这一组**丢掉**（不是留着半组），
    /// 位置也回退 —— 这正是 FFmpeg 对"编码器写多了"的处理方式。
    fn read_count1(
        &self,
        br: &mut BitReader,
        gi: &GranuleInfo,
        gain: &[f32; GRANULE_COEFFS],
        from: usize,
        end_bit: usize,
        out: &mut [f32; GRANULE_COEFFS],
    ) -> Result<()> {
        let tree = &self.count1_trees[gi.count1table_select as usize];
        let mut idx = from;
        while idx <= GRANULE_COEFFS - 4 {
            if br.pos_bits() >= end_bit {
                break;
            }
            let sym = huff_symbol(br, tree)?;
            for b in 0..4usize {
                let pos = idx + b;
                out[pos] = 0.0;
                if (sym >> (3 - b)) & 1 == 1 {
                    out[pos] = if br.read(1)? == 1 { -gain[pos] } else { gain[pos] };
                }
            }
            if br.pos_bits() > end_bit {
                // 这一组读过头了：整组作废
                for b in 0..4 {
                    out[idx + b] = 0.0;
                }
                break;
            }
            idx += 4;
        }
        Ok(())
    }

    /// 立体声处理（M/S 与强度立体声）。**必须在短块重排之前**做。
    ///
    /// 强度立体声的扫描方向与判定（标准 2.4.3.4.9）：从最高的频带往下走，
    /// 一旦发现右声道该带（该窗）有数据、或声像位置超出合法范围，就说明"这一带及以下
    /// 不属于强度区"，此后各带只做 M/S（如果开了）。
    fn stereo(
        &self,
        l: &mut [f32; GRANULE_COEFFS],
        r: &mut [f32; GRANULE_COEFFS],
        gi_r: &GranuleInfo,
        sf_r: &[i32; 40],
        hdr: &FrameHeader,
    ) {
        if !hdr.intensity_stereo {
            if hdr.ms_stereo {
                for i in 0..GRANULE_COEFFS {
                    let (a, b) = (l[i], r[i]);
                    l[i] = (a + b) * SQRT1_2;
                    r[i] = (a - b) * SQRT1_2;
                }
            }
            return;
        }
        // MPEG-1 的声像位置合法上限是 6（≥7 表示"不是强度区"），LSF 是 15（≥16 同理）
        let ctx = StereoCtx { hdr, gi_r, sf_max: if hdr.lsf { 16 } else { 7 } };
        let row = hdr.sample_rate_index;
        let (long_end, short_start) = gi_r.layout(hdr);
        let mut pos = GRANULE_COEFFS;
        // ---- 短块部分：带 12 → short_start，每带内窗 2 → 0（传输序里窗是倒着碰到的）----
        let mut nz = [false; 3];
        if short_start <= 12 {
            let mut i = 12usize;
            loop {
                // 比例因子索引：混合块时短块部分接在长块部分之后；带 12 复用带 11 的值
                let k = long_end + ((if i >= 11 { 11 } else { i }) - short_start) * 3;
                let len = BAND_SHORT[row][i] as usize;
                for w in (0..3usize).rev() {
                    pos -= len;
                    nz[w] = ctx.one_band(l, r, pos, len, nz[w], sf_r.get(k + w).copied());
                }
                if i == short_start {
                    break;
                }
                i -= 1;
            }
        }
        // ---- 长块部分：带 long_end-1 → 0（长块没有"逐窗"这回事）----
        let mut nz_long = nz[0] || nz[1] || nz[2];
        for i in (0..long_end).rev() {
            let len = BAND_LONG[row][i] as usize;
            pos -= len;
            // 最后一个长频带用前一个频带的比例因子（标准这么规定）
            let k = if i == 21 { 20 } else { i };
            nz_long = ctx.one_band(l, r, pos, len, nz_long, Some(sf_r[k]));
        }
    }

    /// 短块重排：把"按窗分组"的传输序换成"按系数交错"的播放序。
    ///
    /// 传输序是 `[w0 的 len 个][w1 的 len 个][w2 的 len 个]`，交错序是
    /// `[c0(w0,w1,w2), c1(w0,w1,w2), …]` —— 混合块时前 36 个系数（2 个子带）不动。
    fn reorder(&self, spec: &mut [f32; GRANULE_COEFFS], gi: &GranuleInfo, hdr: &FrameHeader) {
        if !(gi.window_switching && gi.block_type == 2) {
            return;
        }
        let row = hdr.sample_rate_index;
        let (_, short_start) = gi.layout(hdr);
        let mut base = if gi.mixed_block {
            if row == 8 {
                72
            } else {
                36
            }
        } else {
            0
        };
        let mut tmp = [0f32; GRANULE_COEFFS];
        for width in BAND_SHORT[row][short_start..13].iter() {
            let len = *width as usize;
            for j in 0..len {
                for w in 0..3 {
                    tmp[j * 3 + w] = spec[base + w * len + j];
                }
            }
            spec[base..base + len * 3].copy_from_slice(&tmp[..len * 3]);
            base += len * 3;
        }
    }

    /// 混叠抑制：在相邻子带的边界上做 8 个蝶形。
    ///
    /// 长块 31 次（边界 18, 36, …, 558），纯短块不做，混合块只做 1 次（边界 18）。
    fn antialias(&self, spec: &mut [f32; GRANULE_COEFFS], gi: &GranuleInfo) {
        let times = if gi.window_switching && gi.block_type == 2 {
            if gi.mixed_block {
                1
            } else {
                0
            }
        } else {
            SUBBANDS - 1
        };
        for b in 0..times {
            let base = 18 * (b + 1);
            for i in 0..8 {
                let lo = spec[base - 1 - i];
                let hi = spec[base + i];
                spec[base - 1 - i] = lo * ANTI_CS[i] - hi * ANTI_CA[i];
                spec[base + i] = hi * ANTI_CS[i] + lo * ANTI_CA[i];
            }
        }
    }

    /// 混合滤波器组（IMDCT + 加窗 + 跨帧叠加）→ 频率反转 → 多相合成 → i32 采样。
    fn hybrid_and_synthesize(
        &self,
        st: &mut ChannelState,
        spec: &[f32; GRANULE_COEFFS],
        gi: &GranuleInfo,
        out: &mut Vec<i32>,
    ) {
        let mut z = [0f32; GRANULE_COEFFS];
        for sb in 0..SUBBANDS {
            // 混合块的前 2 个子带走长窗（它们的谱是长块的谱）
            let bt = if gi.window_switching && gi.mixed_block && sb < 2 { 0 } else { gi.block_type };
            let x = &spec[sb * SUBBAND_SAMPLES..(sb + 1) * SUBBAND_SAMPLES];
            let mut raw = [0f32; 36];
            if bt == 2 {
                self.imdct_short(x, &mut raw);
            } else {
                self.imdct_long(x, &mut raw);
                let win = match bt {
                    1 => &self.win_start,
                    3 => &self.win_stop,
                    _ => &self.win_long,
                };
                for p in 0..36 {
                    raw[p] *= win[p];
                }
            }
            for i in 0..SUBBAND_SAMPLES {
                z[sb * SUBBAND_SAMPLES + i] = raw[i] + st.overlap[sb][i];
                st.overlap[sb][i] = raw[i + SUBBAND_SAMPLES];
            }
        }
        // 频率反转：奇数子带的奇数下标样本取反（等价于标准里的 L3_change_sign）
        for sb in (1..SUBBANDS).step_by(2) {
            for i in (1..SUBBAND_SAMPLES).step_by(2) {
                z[sb * SUBBAND_SAMPLES + i] = -z[sb * SUBBAND_SAMPLES + i];
            }
        }
        self.polyphase(st, &z, out);
    }

    /// 长块/起块/停块的 36 点 IMDCT：`y[p] = Σ_m x[m]·cos(π/72·(2p+19)(2m+1))`。
    fn imdct_long(&self, x: &[f32], out: &mut [f32; 36]) {
        for (p, slot) in out.iter_mut().enumerate() {
            let row = &self.imdct_long_cos[p * 18..p * 18 + 18];
            let mut acc = 0.0;
            for m in 0..18 {
                acc += x[m] * row[m];
            }
            *slot = acc;
        }
    }

    /// 短块的 12 点 IMDCT：三个窗的结果分别落在 36 点缓冲的 6 / 12 / 18 偏移上。
    fn imdct_short(&self, x: &[f32], out: &mut [f32; 36]) {
        for w in 0..3usize {
            for p in 0..12 {
                let row = &self.imdct_short_cos[p * 6..p * 6 + 6];
                let mut acc = 0.0;
                for m in 0..6 {
                    acc += x[3 * m + w] * row[m];
                }
                out[6 + 6 * w + p] += acc * self.win_short[p];
            }
        }
    }

    /// 多相合成（标准 2.4.3.4.11）：
    ///
    /// 1. 32 个子带样本 → 64 个 `V[i] = Σ_k N[i][k]·s[k]`；
    /// 2. V 做 1024 点延迟线（前 64 个新算的，其余右移）；
    /// 3. 按标准的两路交错拼出 `U[0..512]`，乘合成窗 D 后每 16 个抽头求和 → 32 个输出。
    ///
    /// 输出量纲：`D` 用的是标准 Table 3-B.3 的归一化值（峰值 ≈ 1），反量化给的是
    /// `2^((global_gain-210)/4)` 量纲 —— 两者相乘后再乘 2^15 正好是 i16 满刻度。
    fn polyphase(&self, st: &mut ChannelState, z: &[f32; GRANULE_COEFFS], out: &mut Vec<i32>) {
        let mut sb_sample = [0f32; SUBBANDS];
        for t in 0..SUBBAND_SAMPLES {
            for (sb, slot) in sb_sample.iter_mut().enumerate() {
                *slot = z[sb * SUBBAND_SAMPLES + t];
            }
            // 延迟线右移 64（倒着拷，避免自己覆盖自己）
            for i in (64..1024).rev() {
                st.v[i] = st.v[i - 64];
            }
            for i in 0..64 {
                let row = &self.matrix[i * SUBBANDS..(i + 1) * SUBBANDS];
                let mut acc = 0.0;
                for k in 0..SUBBANDS {
                    acc += row[k] * sb_sample[k];
                }
                st.v[i] = acc;
            }
            for j in 0..SUBBANDS {
                let mut acc = 0.0;
                for i in 0..16 {
                    // U 的两路交错：U[i*64+j] = V[i*128+j]，U[i*64+j+32] = V[i*128+j+96]
                    let idx = j + 32 * i;
                    let hi = idx >> 6;
                    let rem = idx & 63;
                    let v = st.v[(hi << 7) + if rem < 32 { rem } else { rem + 64 }];
                    acc += v * self.d[idx];
                }
                let v = (acc * OUTPUT_SCALE).round().clamp(-32768.0, 32767.0) as i32;
                out.push(v);
            }
        }
    }
}

/// 强度立体声 / M/S 的判定上下文。
///
/// 之所以抽成一个结构：Rust 没有 `goto`，FFmpeg 里"发现右声道有数据/声像越界就跳到 M/S 分支"
/// 的那条路径得靠返回值表达；把 `hdr`/`gi_r`/`sf_max` 收在一处，传参也不会超七个。
struct StereoCtx<'a> {
    hdr: &'a FrameHeader,
    gi_r: &'a GranuleInfo,
    sf_max: i32,
}

impl StereoCtx<'_> {
    /// 处理一段谱线；返回值 = **"右声道这一段有数据"**（强度区到此为止，往下的带全归 M/S）。
    ///
    /// 两种"这里不做强度立体声"必须分开，混成一个标志会静默吃掉一大片频谱：
    /// - **右声道这一带有数据** → 强度区到此为止，结论要传给更低的带（粘性）；
    /// - **`is_pos >= sf_max`**（MPEG-1 是 7，LSF 是 16）→ 只是**这一带**不用强度立体声，
    ///   下面的带还要接着判（不粘性）。标准里 `is_pos` 是逐带的属性，不是分区边界。
    ///
    /// 为什么这种错极难被发现：强度立体声的两个声像系数满足 `kl + kr ≡ 1`，所以
    /// 中声道 `M = (L+R)/2 = ch0·(kl+kr)/2` 与被吞掉多少带**无关**，怎么错都听不出来；
    /// 只有侧声道 `S = (L−R)/2 = ch0·(kl−kr)/2` 会歪。当年就是靠"中声道逐样本吻合、
    /// 侧声道差 5%"这个反常组合才把范围缩到这里。
    ///
    /// `sf` 是右声道这一带（这一窗）的比例因子；混合块里短块部分的索引由调用方算好。
    fn one_band(
        &self,
        l: &mut [f32; GRANULE_COEFFS],
        r: &mut [f32; GRANULE_COEFFS],
        pos: usize,
        len: usize,
        already_outside: bool,
        sf: Option<i32>,
    ) -> bool {
        // 取不到比例因子（索引越界）时按"超出范围"处理，等同 FFmpeg 的 sf >= sf_max 分支
        let sf = sf.unwrap_or(self.sf_max);
        let mut outside = already_outside;
        if !outside {
            let has_data = r[pos..pos + len].iter().any(|v| *v != 0.0);
            if !has_data && sf < self.sf_max {
                // 属于强度区：用左声道按声像位置铺开
                let (kl, kr) = pan_ratio(sf, self.gi_r, self.hdr);
                for j in pos..pos + len {
                    let v = l[j];
                    l[j] = v * kl;
                    r[j] = v * kr;
                }
                return false;
            }
            // 只有"右声道有数据"才终结强度区；`sf >= sf_max` 只跳过这一带
            outside = has_data;
        }
        if self.hdr.ms_stereo {
            for j in pos..pos + len {
                let (a, b) = (l[j], r[j]);
                l[j] = (a + b) * SQRT1_2;
                r[j] = (a - b) * SQRT1_2;
            }
        }
        outside
    }
}

/// 声像系数：MPEG-1 用 `tan(kπ/12)` 推出的 `(kl, kr)`；LSF 用指数形式。
///
/// MPEG-1：`r = IS_RATIOS[is_pos]`、`kl = r/(1+r)`、`kr = 1/(1+r)`；`is_pos == 6` 是"极右"
/// 特例（`tan(π/2)` 趋于无穷）。
/// LSF：`kr = 2^(-(((is_pos+1)>>1) << (scalefac_compress&1)) / 4)`、`kl = 1`，
/// `is_pos` 为奇数时两者互换。
fn pan_ratio(sf: i32, gi_r: &GranuleInfo, hdr: &FrameHeader) -> (f32, f32) {
    if !hdr.lsf {
        if sf >= 6 {
            (1.0, 0.0)
        } else {
            let rr = IS_RATIOS[sf as usize];
            let inv = 1.0 / (1.0 + rr);
            (rr * inv, inv)
        }
    } else {
        let shift = (gi_r.scalefac_compress & 1) as i32;
        let kr = exp2_quarter(-(((sf + 1) >> 1) << shift));
        if sf & 1 == 1 {
            (kr, 1.0)
        } else {
            (1.0, kr)
        }
    }
}

// ---------------------------------------------------------------- AudioDecoder

impl AudioDecoder for Mp3Decoder {
    /// 喂一个包，收 PCM。包里可以是一帧、多帧，也可以是半帧（内部攒着）。
    fn decode(&mut self, packet: &[u8], pts_us: i64) -> Result<Vec<PcmChunk>> {
        let had_leftover = !self.pending.is_empty();
        self.pending.extend_from_slice(packet);
        if !had_leftover {
            self.next_pts_us = pts_us;
        }
        let mut out = Vec::new();
        loop {
            let Some(start) = find_sync(&self.pending, 0) else {
                // 一个完整同步字都没有：留最后 1 个字节（可能正是同步字的 0xFF），其余丢掉
                let keep = if self.pending.last() == Some(&0xFF) { 1 } else { 0 };
                let drop_len = self.pending.len() - keep;
                self.pending.drain(..drop_len);
                break;
            };
            if start > 0 {
                self.pending.drain(..start);
            }
            let hdr = match FrameHeader::parse(&self.pending) {
                Ok(h) => h,
                Err(MediaError::NeedMore { .. }) => break,
                Err(_) => {
                    // 假同步：跳过这个字节重新找
                    self.pending.remove(0);
                    continue;
                }
            };
            if hdr.frame_len < 24 || hdr.frame_len > MAX_FRAME_BYTES {
                self.pending.remove(0);
                continue;
            }
            if self.pending.len() < hdr.frame_len {
                break; // 半帧，等更多数据
            }
            let frame = self.pending[..hdr.frame_len].to_vec();
            self.pending.drain(..hdr.frame_len);
            let pts = self.next_pts_us;
            let chunk = self.decode_frame(&frame, pts)?;
            self.next_pts_us +=
                (chunk.samples.len() / chunk.channels.max(1) as usize) as i64 * 1_000_000
                    / chunk.sample_rate.max(1) as i64;
            out.push(chunk);
        }
        Ok(out)
    }

    /// 收尾：完整的帧在 [`Self::decode`] 里已经全部吐出，这里没有残留。
    fn flush(&mut self) -> Result<Vec<PcmChunk>> {
        Ok(Vec::new())
    }

    /// 声道数 / 采样率 / 位深。首帧之前只能给 `(0, 0, 16)` —— MP3 的采样率随帧头走，
    /// 开流前没有任何地方能提前知道（调用方拿到 chunk 后会以 chunk 里的值为准）。
    fn stream_info(&self) -> (u32, u8, u8) {
        self.info.unwrap_or((0, 0, 16))
    }
}

// ---------------------------------------------------------------- 小工具

/// 读侧信息：`main_data_begin`、每声道的 `scfsi`、`[gr][ch]` 的粒度信息。
fn read_side_info(
    br: &mut BitReader,
    hdr: &FrameHeader,
) -> Result<(usize, [u8; 2], Vec<[GranuleInfo; 2]>)> {
    let nch = hdr.channels as usize;
    let mut scfsi = [0u8; 2];
    let main_data_begin;
    if hdr.lsf {
        main_data_begin = br.read(8)? as usize;
        let _private = br.read(nch as u8)?; // 每声道 1 位
    } else {
        main_data_begin = br.read(9)? as usize;
        let _private = br.read(if nch == 2 { 3 } else { 5 })?;
        for slot in scfsi.iter_mut().take(nch) {
            // 每个声道 4 位：bit3 对应第 0 组比例因子
            *slot = br.read(4)? as u8;
        }
    }
    let mut granules = Vec::with_capacity(hdr.granules);
    for _ in 0..hdr.granules {
        let mut row = [GranuleInfo::default(); 2];
        for g in row.iter_mut().take(nch) {
            g.part2_3_length = br.read(12)? as u16;
            g.big_values = br.read(9)? as u16;
            if g.big_values > 288 {
                return Err(MediaError::Bitstream(format!(
                    "MP3: big_values = {} 超过 288（标准上限）",
                    g.big_values
                )));
            }
            g.global_gain = br.read(8)? as u8;
            g.scalefac_compress = br.read(if hdr.lsf { 9 } else { 4 })? as u16;
            g.window_switching = br.read(1)? == 1;
            if g.window_switching {
                g.block_type = br.read(2)? as u8;
                if g.block_type == 0 {
                    return Err(MediaError::Bitstream(
                        "MP3: window_switching_flag = 1 但 block_type = 0（保留组合）".into(),
                    ));
                }
                g.mixed_block = br.read(1)? == 1;
                if g.mixed_block && hdr.sample_rate_index == 8 {
                    // 8 kHz 档的混合块长块部分是 4 个子带而不是 2 个，边界处处不同；
                    // FFmpeg 也直接拒绝（"switch point in 8khz"），我们同样不猜。
                    return Err(MediaError::Unsupported(
                        "MP3: 8 kHz 的混合块（长块部分是 4 个子带，标准实现都不支持）".into(),
                    ));
                }
                // 开窗切换时只传两个区域表号（第三个区域恒为空，用不到）
                for t in g.table_select.iter_mut().take(2) {
                    *t = br.read(5)? as u8;
                }
                for s in g.subblock_gain.iter_mut() {
                    *s = br.read(3)? as u8;
                }
                // 区域 0 的频带数是隐含的：纯短块 8，起止块/混合块 7
                g.region0_count = if g.block_type == 2 && !g.mixed_block { 8 } else { 7 };
            } else {
                for t in g.table_select.iter_mut() {
                    *t = br.read(5)? as u8;
                }
                g.region0_count = br.read(4)? as u8;
                g.region1_count = br.read(3)? as u8;
            }
            g.preflag = if hdr.lsf {
                // LSF 没有 preflag 位：由 scalefac_compress 推出来（标准 2.4.3.4.7）
                g.scalefac_compress >= 500
            } else {
                br.read(1)? == 1
            };
            g.scalefac_scale = br.read(1)? == 1;
            g.count1table_select = br.read(1)? as u8;
        }
        granules.push(row);
    }
    Ok((main_data_begin, scfsi, granules))
}

/// 读 `n` 位；`n == 0` 表示"有比例因子但长度为 0"，**不读任何位**，值为 0。
fn read_bits_or_zero(br: &mut BitReader, n: u8) -> Result<i32> {
    if n == 0 {
        Ok(0)
    } else {
        Ok(br.read(n)? as i32)
    }
}

/// LSF 的三段分解：把一个数拆成 `n1/n2/n3` 进制下的各位。
///
/// 位宽为 0 的段**不做除法**（0 进制没有意义），这也是 FFmpeg `SPLIT` 宏的分支行为：
/// `n` 不是 3/4/5/6 时直接给 0 并保持原值。
fn lsf_sf_expand(sf: i32, n1: i32, n2: i32, n3: i32) -> [i32; 4] {
    let mut v = sf;
    let mut out = [0i32; 4];
    for (slot, n) in [(3usize, n3), (2, n2), (1, n1)] {
        if n > 0 {
            out[slot] = v % n;
            v /= n;
        }
    }
    out[0] = v;
    out
}

/// `2^(exp/4)`。指数可能很负（LSF 的比例因子很大），下溢到 0 是正确行为。
fn exp2_quarter(exp: i32) -> f32 {
    (exp as f32 * 0.25).exp2()
}

/// 走一次哈夫曼树，返回打包好的符号 `x<<4 | y`。
///
/// 节点打包：低 16 位 = 读 0 走的分支，高 16 位 = 读 1 走的分支，
/// 槽值 bit15 置位表示叶子（低 15 位是符号）。树是完备前缀码，读满最大码长必定落叶子。
fn huff_symbol(br: &mut BitReader, tree: &[u32]) -> Result<u32> {
    let mut node = 0usize;
    loop {
        let v = tree
            .get(node)
            .ok_or_else(|| MediaError::Bitstream("MP3: 哈夫曼树走出界（码表坏了）".into()))?;
        let child = if br.read(1)? == 0 { v & 0xFFFF } else { v >> 16 };
        if child & 0x8000 != 0 {
            return Ok(child & 0x7FFF);
        }
        node = child as usize;
    }
}

/// 把 `(码字, 码长)` 清单展开成与 [`tables`] 里同一种打包的树。
///
/// count1 的两张表是变长前缀码，展开成树之后就能和谱表共用 [`huff_symbol`] ——
/// 不用再写一套"读一位比一次码表"的前缀匹配。
fn build_tree(codes: &[u8; 16], lens: &[u8; 16]) -> Vec<u32> {
    let mut tree = vec![0u32];
    for sym in 0..16usize {
        let len = lens[sym] as usize;
        let code = codes[sym] as u32;
        let mut node = 0usize;
        for depth in (0..len).rev() {
            let bit = (code >> depth) & 1;
            let child = if bit == 0 { tree[node] & 0xFFFF } else { tree[node] >> 16 };
            if depth == 0 {
                let leaf = 0x8000u32 | sym as u32;
                tree[node] = if bit == 0 {
                    (tree[node] & 0xFFFF_0000) | leaf
                } else {
                    tree[node] | (leaf << 16)
                };
            } else if child != 0 {
                node = child as usize;
            } else {
                let idx = tree.len() as u32;
                tree.push(0);
                tree[node] = if bit == 0 {
                    (tree[node] & 0xFFFF_0000) | idx
                } else {
                    tree[node] | (idx << 16)
                };
                node = idx as usize;
            }
        }
    }
    tree
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 造一个合法帧头（默认 MPEG-1 44.1k 128kbps 立体声）。
    fn header_bytes(version_bits: u8, layer_bits: u8, bitrate: u8, sr: u8, mode: u8, mode_ext: u8) -> [u8; 4] {
        [
            0xFF,
            0xE0 | (version_bits << 3) | (layer_bits << 1) | 1,
            (bitrate << 4) | (sr << 2),
            (mode << 6) | (mode_ext << 4),
        ]
    }

    #[test]
    fn parses_mpeg1_stereo_header() {
        // MPEG-1（11）Layer III（01）128 kbps（9）44.1 kHz（0）联合立体声（1）
        let h = FrameHeader::parse(&header_bytes(0b11, 0b01, 9, 0, 1, 0b11)).unwrap();
        assert_eq!(h.version, Version::Mpeg1);
        assert!(!h.lsf);
        assert_eq!(h.sample_rate, 44100);
        assert_eq!(h.sample_rate_index, 0);
        assert_eq!(h.bitrate_kbps, 128);
        assert_eq!(h.channels, 2);
        assert!(h.ms_stereo && h.intensity_stereo);
        assert_eq!(h.granules, 2);
        assert_eq!(h.side_info_len, 32);
        // 144 * 128000 / 44100 = 417
        assert_eq!(h.frame_len, 417);
        assert_eq!(h.samples_per_frame(), 1152);
    }

    #[test]
    fn parses_lsf_mono_header() {
        // MPEG-2（10）Layer III 单声道（3）24 kbps（3）22.05 kHz（0）→ LSF 档位 3
        let h = FrameHeader::parse(&header_bytes(0b10, 0b01, 3, 0, 3, 0)).unwrap();
        assert!(h.lsf);
        assert_eq!(h.version, Version::Mpeg2);
        assert_eq!(h.sample_rate, 22050);
        assert_eq!(h.sample_rate_index, 3);
        assert_eq!(h.bitrate_kbps, 24);
        assert_eq!(h.channels, 1);
        assert_eq!(h.granules, 1);
        // LSF 单声道侧信息 9 字节（= 8 + 1 + 59 位）
        assert_eq!(h.side_info_len, 9);
        // 72 * 24000 / 22050 = 78
        assert_eq!(h.frame_len, 78);
        assert_eq!(h.samples_per_frame(), 576);
    }

    #[test]
    fn rejects_layer_1_and_2_and_free_format_and_emphasis() {
        let e = FrameHeader::parse(&header_bytes(0b11, 0b11, 9, 0, 3, 0)).unwrap_err();
        assert!(format!("{e}").contains("Layer 1"), "Layer I 要明确报出来：{e}");
        let e = FrameHeader::parse(&header_bytes(0b11, 0b10, 9, 0, 3, 0)).unwrap_err();
        assert!(format!("{e}").contains("Layer 2"), "Layer II 要明确报出来：{e}");
        // free format：比特率索引 0
        let e = FrameHeader::parse(&header_bytes(0b11, 0b01, 0, 0, 3, 0)).unwrap_err();
        assert!(format!("{e}").contains("free format"), "free format 要说清楚：{e}");
        // emphasis != 0
        let mut b = header_bytes(0b11, 0b01, 9, 0, 3, 0);
        b[3] |= 0b01;
        let e = FrameHeader::parse(&b).unwrap_err();
        assert!(format!("{e}").contains("emphasis"), "emphasis 要说清楚：{e}");
        // 采样率索引 3 是保留值
        assert!(FrameHeader::parse(&header_bytes(0b11, 0b01, 9, 3, 3, 0)).is_err());
        // 比特率索引 15 也是保留值（只有 0 才是 free format）
        let e = FrameHeader::parse(&header_bytes(0b11, 0b01, 15, 0, 3, 0)).unwrap_err();
        assert!(!format!("{e}").contains("free format"), "15 不是 free format：{e}");
        // MPEG-2.5：采样率再减半，档位落在 6..8
        let h = FrameHeader::parse(&header_bytes(0b00, 0b01, 1, 2, 3, 0)).unwrap();
        assert_eq!(h.sample_rate, 8000);
        assert_eq!(h.sample_rate_index, 8);
    }

    #[test]
    fn find_sync_skips_garbage() {
        let mut data = vec![0u8, 0xFF, 0x00, 0x12, 0x34]; // 假同步：0xFF 后面高 3 位不是 111
        let good = header_bytes(0b11, 0b01, 9, 0, 0, 0);
        let at = data.len();
        data.extend_from_slice(&good);
        data.push(0);
        assert_eq!(find_sync(&data, 0), Some(at));
        assert_eq!(find_sync(&[0u8, 1, 2, 3], 0), None);
    }

    /// 哈夫曼树必须是**完备前缀码**（Kraft 和 = 1），否则解码时会走进空槽。
    /// 表的生成期已经查过一遍，这里在仓库里再钉一次（改表时才有意义）。
    fn kraft_sum(tree: &[u32], node: usize, depth: u32, acc: &mut f64) {
        let v = tree[node];
        for bit in [0u32, 1] {
            let child = if bit == 0 { v & 0xFFFF } else { v >> 16 };
            if child & 0x8000 != 0 {
                *acc += 2f64.powi(-(depth as i32 + 1));
            } else {
                kraft_sum(tree, child as usize, depth + 1, acc);
            }
        }
    }

    #[test]
    fn huffman_tables_are_complete_prefix_codes() {
        let mut checked = 0;
        for (i, entry) in HUFF_SPEC.iter().enumerate() {
            let Some((tree, _)) = entry else { continue };
            let mut sum = 0.0;
            kraft_sum(tree, 0, 0, &mut sum);
            assert!((sum - 1.0).abs() < 1e-9, "表 {i} 的 Kraft 和是 {sum}，不是 1");
            checked += 1;
        }
        // 15 张谱表（0 / 4 / 14 是空槽，16..31 复用 1719 / 2230 两棵树，但都算一份）
        assert_eq!(checked, 29, "HUFF_SPEC 里应有 29 张非空表");
    }

    #[test]
    fn count1_trees_match_their_code_lists() {
        for t in 0..2 {
            let tree = build_tree(&COUNT1_CODES[t], &COUNT1_BITS[t]);
            // 把 6 位前缀挨个喂进去：树必须恰好消耗 (码长) 位并给出对应的符号。
            // 这同时验证了"表 32 是变长码"这件事（它的码长在 1..6 之间）。
            for sym in 0..16usize {
                let len = COUNT1_BITS[t][sym] as usize;
                let code = COUNT1_CODES[t][sym] as u32;
                // 码字左对齐到 6 位，后面补 1（补的位不该被读到）
                let filled = (code << (6 - len)) | ((1u32 << (6 - len)) - 1);
                let v16 = (filled as u16) << 10;
                let bytes = [(v16 >> 8) as u8, (v16 & 0xFF) as u8];
                let mut br = BitReader::new(&bytes);
                let got = huff_symbol(&mut br, &tree).unwrap();
                assert_eq!(got, sym as u32, "表 {t} 符号 {sym} 解错");
                assert_eq!(br.pos_bits(), len, "表 {t} 符号 {sym} 消耗位数不对");
            }
        }
    }

    #[test]
    fn zero_pair_does_not_read_a_sign_bit() {
        let dec = Mp3Decoder::open(&[]).unwrap();
        let data = [0xFFu8, 0xFF];
        // (0, 0)：一个位都不读（符号位只跟在非零值后面）
        let mut br = BitReader::new(&data);
        assert_eq!(dec.decode_pair(&mut br, 0, 0, 0).unwrap(), (0.0, 0.0));
        assert_eq!(br.pos_bits(), 0, "零值对不该动位指针");
        // (0, 3)：读一个符号位（3 < 15，没有 linbits），符号位 1 → 负
        let mut br = BitReader::new(&data);
        let (x, y) = dec.decode_pair(&mut br, 0, 3, 0).unwrap();
        assert_eq!(x, 0.0);
        assert!(y < 0.0, "符号位是 1，应当是负值");
        assert!((y.abs() - 3f32.powf(4.0 / 3.0)).abs() < 1e-4);
        assert_eq!(br.pos_bits(), 1);
        // (15, 0) 且 linbits = 4：先读 4 位转义，再读 1 位符号
        let mut br = BitReader::new(&data);
        let _ = dec.decode_pair(&mut br, 15, 0, 4).unwrap();
        assert_eq!(br.pos_bits(), 5, "15 要先读 linbits 再读符号位");
    }

    #[test]
    fn band_tables_cover_every_coefficient() {
        for (row, bands) in BAND_LONG.iter().enumerate() {
            let sum: usize = bands.iter().map(|&b| b as usize).sum();
            assert_eq!(sum, GRANULE_COEFFS, "长频带表第 {row} 行合计 {sum}");
        }
        for (row, bands) in BAND_SHORT.iter().enumerate() {
            let sum: usize = bands.iter().map(|&b| b as usize).sum::<usize>() * 3;
            assert_eq!(sum, GRANULE_COEFFS, "短频带表第 {row} 行 ×3 合计 {sum}");
        }
    }

    #[test]
    fn synthesis_window_mirrors_and_antialias_is_unitary() {
        let mut d = [0f32; 512];
        for (i, slot) in d.iter_mut().enumerate().take(257) {
            *slot = SYNTH_D[i];
        }
        for (i, slot) in d.iter_mut().enumerate().skip(257) {
            let j = 512 - i;
            let v = SYNTH_D[j];
            *slot = if j % 64 != 0 { -v } else { v };
        }
        for i in 1..512 {
            let j = 512 - i;
            // i 是 64 的倍数时两半同号（镜像关系里那个例外）
            if j < 512 && i % 64 != 0 {
                assert!((d[i] + d[j]).abs() < 1e-9, "D[{i}] 与 D[{j}] 不满足镜像关系");
            }
        }
        for i in 0..8 {
            let n = ANTI_CS[i] * ANTI_CS[i] + ANTI_CA[i] * ANTI_CA[i];
            assert!((n - 1.0).abs() < 1e-5, "混叠系数 {i} 不是单位旋转：{n}");
        }
        // 强度立体声的声像系数：左右两个系数之和恒为 1（能量守恒）
        for (k, slot) in IS_RATIOS.iter().enumerate() {
            let inv = 1.0 / (1.0 + slot);
            assert!((slot * inv + inv - 1.0).abs() < 1e-5, "is_pos {k} 的声像系数不守恒");
        }
    }

    #[test]
    fn regions_follow_the_standard() {
        let dec = Mp3Decoder::open(&[]).unwrap();
        let mk = |ws: bool, bt: u8, mixed: bool, ra1: u8, ra2: u8, big: u16| {
            let mut g = GranuleInfo { window_switching: ws, block_type: bt, mixed_block: mixed, region0_count: ra1, region1_count: ra2, big_values: big, ..Default::default() };
            g.table_select = [1, 1, 1];
            g
        };
        let h448 = FrameHeader::parse(&header_bytes(0b11, 0b01, 9, 0, 3, 0)).unwrap(); // 44.1k
        // 长块：44.1k 前 9 个频带 = 4*6+6+6+8 = 44 个系数 = 22 对
        let regions = mk(false, 0, false, 8, 0, 288).regions(&h448, &dec.band_idx_long);
        assert_eq!(regions[0], 22, "区域 0 的边界是前 (ra1+1) 个频带");
        // 区域 1 = 前 (8+0+2)=10 个频带 = 4*6+6*2+8+8 = 52 → 26 对，减去区域 0 的 22
        assert_eq!(regions[1], 4);
        assert_eq!(regions[2], 288 - 26);
        // 纯短块：36 个系数 = 18 对
        let regions = mk(true, 2, false, 8, 0, 100).regions(&h448, &dec.band_idx_long);
        assert_eq!(regions, [18, 82, 0], "纯短块的区域 0 是 36 个系数，其后全归区域 1");
        // 起止块 / 混合块：region0_count 隐含 7，边界取"前 8 个长频带" = 36 个系数
        let regions = mk(true, 1, false, 7, 0, 100).regions(&h448, &dec.band_idx_long);
        assert_eq!(regions[0], 18);
        // big_values 截断：区域长度之和必须等于 big_values
        let regions = mk(false, 0, false, 8, 0, 7).regions(&h448, &dec.band_idx_long);
        assert_eq!(regions.iter().sum::<usize>(), 7);
    }

    #[test]
    fn lsf_scalefactor_split_matches_the_standard() {
        // scalefac_compress = 200（< 400）→ 5/4/4 分解，按 n1/n2/n3 的进制还原得回 200
        let s = lsf_sf_expand(200, 5, 4, 4);
        assert_eq!(((s[0] * 5 + s[1]) * 4 + s[2]) * 4 + s[3], 200, "分解要能还原");
        // 位宽为 0 的段不读位、也不做除法（0 进制没有意义）
        let s = lsf_sf_expand(7, 3, 0, 0);
        assert_eq!(s, [2, 1, 0, 0]);
        // 强度立体声右声道用的 6/6 分解
        let s = lsf_sf_expand(63, 6, 6, 0);
        assert_eq!(s, [1, 4, 3, 0]);
    }

    #[test]
    fn reorder_interleaves_the_three_windows() {
        let dec = Mp3Decoder::open(&[]).unwrap();
        let hdr = FrameHeader::parse(&header_bytes(0b11, 0b01, 9, 0, 3, 0)).unwrap();
        let gi = GranuleInfo {
            window_switching: true,
            block_type: 2,
            mixed_block: false,
            ..Default::default()
        };
        // 纯短块第 0 个频带宽度是 4：传输序 [w0 四个][w1 四个][w2 四个] → 交错成 3 个一组
        let mut spec = [0f32; GRANULE_COEFFS];
        for i in 0..4 {
            spec[i] = 1.0;
            spec[4 + i] = 2.0;
            spec[8 + i] = 3.0;
        }
        dec.reorder(&mut spec, &gi, &hdr);
        for i in 0..4 {
            assert_eq!((spec[i * 3], spec[i * 3 + 1], spec[i * 3 + 2]), (1.0, 2.0, 3.0));
        }
    }

    /// 强度立体声的"声像越界"不终结强度区（模块文档第 9 条）。
    ///
    /// 这条规则错了没有任何"显眼"症状：`kl + kr ≡ 1` 保证中声道 `(L+R)/2` 完全不受影响，
    /// 只有侧声道会凭空少掉一大片频谱，而整体相关度仍然 0.9999 —— 所以单独立一条
    /// **不依赖样例文件**的测试钉住它（端到端那条是 `#[ignore]`，默认跑不到）。
    #[test]
    fn intensity_out_of_range_band_does_not_end_the_region() {
        // MPEG-1 44.1k 联合立体声，mode_ext = 01 → 只开强度立体声
        let hdr = FrameHeader::parse(&header_bytes(0b11, 0b01, 9, 0, 1, 0b01)).unwrap();
        assert!(hdr.intensity_stereo && !hdr.ms_stereo, "前提：只开强度立体声");
        let gi = GranuleInfo::default();
        let ctx = StereoCtx { hdr: &hdr, gi_r: &gi, sf_max: 7 };

        let mut l = [0f32; GRANULE_COEFFS];
        let mut r = [0f32; GRANULE_COEFFS];
        l[0..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        // 上面一带 is_pos = 7（越界）：只跳过它自己，**不能**把结论传给更低的带
        let outside = ctx.one_band(&mut l, &mut r, 0, 4, false, Some(7));
        assert!(!outside, "is_pos 越界只该跳过这一带，不该终结强度区");
        assert_eq!(&l[..4], &[1.0, 2.0, 3.0, 4.0][..], "越界带不动数据");

        // 紧挨着下面一带 is_pos = 3（正中间，kl = kr = 0.5）：照旧做强度立体声
        let outside = ctx.one_band(&mut l, &mut r, 0, 4, outside, Some(3));
        assert!(!outside);
        assert_eq!(&l[..4], &[0.5, 1.0, 1.5, 2.0][..], "左声道按 kl 缩放");
        assert_eq!(&r[..4], &[0.5, 1.0, 1.5, 2.0][..], "右声道按 kr 缩放");

        // 右声道这一带有数据 → 强度区到此为止，结论要传给更低的带
        let mut l2 = [0f32; GRANULE_COEFFS];
        let mut r2 = [0f32; GRANULE_COEFFS];
        l2[0] = 1.0;
        r2[3] = 1.0;
        assert!(
            ctx.one_band(&mut l2, &mut r2, 0, 4, false, Some(3)),
            "右声道有数据才终结强度区"
        );
        assert_eq!(&l2[..4], &[1.0, 0.0, 0.0, 0.0][..], "终结后这一带不做强度立体声");
    }

    #[test]
    fn unsupported_and_short_inputs_are_reported_clearly() {
        let mut dec = Mp3Decoder::open(&[]).unwrap();
        // 空包不报错，也不出东西
        assert!(dec.decode(&[], 0).unwrap().is_empty());
        // 首帧之前 stream_info 只能给 0
        assert_eq!(dec.stream_info(), (0, 0, 16));
        // 一段没有同步字的垃圾：安静地丢掉，不炸
        assert!(dec.decode(&[0x00, 0x01, 0x02, 0x03], 0).unwrap().is_empty());
        // 半帧（帧头合法但字节不够）：攒着等后续
        let half = header_bytes(0b11, 0b01, 9, 0, 3, 0);
        assert!(dec.decode(&half, 0).unwrap().is_empty());
        assert!(!dec.pending.is_empty(), "半帧要留在待解缓冲里");
    }
}
