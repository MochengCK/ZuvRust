//! **宏块层**（§7.3.4 切片数据 / §7.3.5 宏块层）：切片数据怎么走、每个宏块的语法
//! 长什么样，以及 §9.2.1 第 4~7 步那个 `nC` 推导。
//!
//! ## 这一层在干什么
//!
//! 到 [`super::cavlc`] 为止，我们只会解"一个 4×4 块的系数"。这一层把它接成
//! **宏块**：`mb_type`（表 7-11）→ 帧内预测模式 → `coded_block_pattern`（表 9-4）
//! → `mb_qp_delta` → `residual()`（亮度 DC/AC、8×8 变换、色度 DC/AC 的分支）。
//!
//! ## 两个容易被忽略但一定会踩的点
//!
//! 1. **`nC` 用的是"左邻/上邻 4×4 块的 TotalCoeff"，不是当前块自己的**
//!    （§9.2.1 第 4~7 步）。NOTE 1 说得很明确：`Intra_16x16` 与色度块那边
//!    **不算 DC** —— 因为 DC 是单独解出来的。所以这张计数网格里，
//!    `Intra_16x16` 宏块存的是 **AC 的个数**，而 `I_NxN` / 帧间宏块存的是全量。
//!    存错的症状是"从某个宏块起整片花"。
//! 2. **可用性就是"在不在画面里"**。渐进帧、无 FMO、无 MBAFF 时，
//!    `mbAddrA` 可用 ⟺ 块不在最左列、`mbAddrB` 可用 ⟺ 块不在最上行 ——
//!    不需要那一整套 `6.4.11.x` 邻居推导。这是引擎的支持口径（参数集那层已经
//!    把 FMO/场编码拒掉了）换来的简化。
//!
//! ## 验收
//!
//! [`super::cavlc::Cavlc::check_rbsp_trailing`]：一条切片的所有宏块解完之后，
//! 读位必须**正好**停在 `rbsp_trailing_bits` 上。不需要参考解码器，
//! 而且多读/少读一位都对不上。

use me_core::{MediaError, Result};

use super::cavlc::Cavlc;
use super::cavlc_tables::{CODED_BLOCK_PATTERN_INTRA, CODED_BLOCK_PATTERN_INTER};
use crate::h264::{Pps, SliceHeader, Sps};

/// `mb_type == 0`（I 切片）：`I_NxN`（帧内 4×4 或 8×8）。
pub const I_NXN: u32 = 0;
/// `mb_type == 25`：`I_PCM`（原始样本直接写进画面）。
pub const I_PCM: u32 = 25;

/// 4:2:0 下一个宏块的色度样本数（§6.4.3 的 `MbWidthC` / `MbHeightC`）。
///
/// 引擎的支持口径已经写死成 8 位 4:2:0（[`crate::h264::Sps::check_supported`]），
/// 所以这里是常量；哪天要接 4:2:2，这里必须跟着 `SubWidthC` / `SubHeightC` 走。
const MB_W_C: usize = 8;
const MB_H_C: usize = 8;

/// 宏块用哪种帧内预测（表 7-11 的 `MbPartPredMode` 与 `transform_size_8x8_flag`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraPred {
    /// `I_NxN` 且 `transform_size_8x8_flag == 0`：16 个 4×4 块
    NxN4x4,
    /// `I_NxN` 且 `transform_size_8x8_flag == 1`：4 个 8×8 块
    NxN8x8,
    /// `I_16x16`：整个宏块一个预测模式
    I16x16,
    /// `I_PCM`
    Pcm,
}

/// `I_16x16` 的 `mb_type` 拆出来的三段（表 7-11 的结构就是"1 + 模式×4 + 色度cbp×3 + 亮度cbp×2"）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct I16x16Parts {
    pub pred_mode: u8,
    pub cbp_chroma: u32,
    pub cbp_luma: u32,
}

/// `1 <= mb_type <= 24` 时把三段拆出来；`I_NxN` / `I_PCM` 返回 `None`。
pub fn i16x16_parts(mb_type: u32) -> Option<I16x16Parts> {
    let t = mb_type.checked_sub(1)?;
    if t >= 24 {
        return None;
    }
    Some(I16x16Parts {
        pred_mode: (t % 4) as u8,
        cbp_chroma: (t / 4) % 3,
        cbp_luma: if t >= 12 { 15 } else { 0 },
    })
}

/// 一个宏块解出来的语法信息（预测与去块滤波都要用）。
#[derive(Debug, Clone)]
pub struct MbInfo {
    pub mb_type: u32,
    pub intra_pred: IntraPred,
    /// `Intra_16x16` 的预测模式（0=垂直 1=水平 2=DC 3=平面），其余无意义
    pub intra16x16_pred_mode: u8,
    /// 色度的预测模式（0..3 对应规范里的四种，4 = `Intra_8x8` 的 DC 之类，见 §8.3.4）
    pub intra_chroma_pred_mode: u8,
    /// `prev_intra4x4_pred_mode_flag` / `rem_intra4x4_pred_mode`（**最终模式**要
    /// 等重建那一层按 §8.3.1.1 从邻居推，所以这里存的是原始语法）
    pub intra4x4_prev: [bool; 16],
    pub intra4x4_rem: [u8; 16],
    pub intra8x8_prev: [bool; 4],
    pub intra8x8_rem: [u8; 4],
    pub cbp_luma: u32,
    pub cbp_chroma: u32,
    pub transform_8x8: bool,
    /// 这个宏块的 `QPY`（= 26 + `pic_init_qp_minus26` + `slice_qp_delta` + 累计 `mb_qp_delta`）
    pub qp_y: i32,
    /// `I_PCM`：去块滤波时它的 `QPY` 要**当 0 用**（表 8-16 那一节的口径）
    pub is_pcm: bool,
}

impl Default for MbInfo {
    fn default() -> Self {
        Self {
            mb_type: u32::MAX,
            intra_pred: IntraPred::I16x16,
            intra16x16_pred_mode: 0,
            intra_chroma_pred_mode: 0,
            intra4x4_prev: [false; 16],
            intra4x4_rem: [0; 16],
            intra8x8_prev: [false; 4],
            intra8x8_rem: [0; 4],
            cbp_luma: 0,
            cbp_chroma: 0,
            transform_8x8: false,
            qp_y: 26,
            is_pcm: false,
        }
    }
}

impl MbInfo {
    /// 这个宏块是不是帧内（去块滤波的 bS 推导要看）。
    ///
    /// 现在 I 切片里必然全是帧内；帧间那一层接上后要按 `intra_pred` 判。
    #[allow(dead_code, reason = "去块滤波那一层才用得到")]
    pub fn is_intra(&self) -> bool {
        matches!(
            self.intra_pred,
            IntraPred::NxN4x4 | IntraPred::NxN8x8 | IntraPred::I16x16 | IntraPred::Pcm
        )
    }
}

/// 一个宏块的系数（**按扫描序**，重建那一层直接拿去反扫描/反量化/反变换）。
#[derive(Debug, Clone)]
pub struct MbCoeffs {
    /// `Intra_16x16` 的 16 个亮度 DC（4×4 网格）
    pub luma_dc: [i32; 16],
    /// 16 个 4×4 亮度块；`Intra_16x16` 时存的是 **AC**（第 0 位恒为 0）
    pub luma4: [[i32; 16]; 16],
    /// `transform_size_8x8_flag == 1` 时的 4 个 8×8 亮度块（其余情况全零）
    pub luma8: [[i32; 64]; 4],
    pub cb_dc: [i32; 4],
    pub cr_dc: [i32; 4],
    pub cb4: [[i32; 16]; 4],
    pub cr4: [[i32; 16]; 4],
    /// `I_PCM` 的原始样本：256 亮度 + 64 Cb + 64 Cr（4:2:0），其余宏块是空
    pub pcm: Vec<u8>,
}

impl Default for MbCoeffs {
    fn default() -> Self {
        Self {
            luma_dc: [0; 16],
            luma4: [[0; 16]; 16],
            luma8: [[0; 64]; 4],
            cb_dc: [0; 4],
            cr_dc: [0; 4],
            cb4: [[0; 16]; 4],
            cr4: [[0; 16]; 4],
            pcm: Vec::new(),
        }
    }
}

/// **`nC` 推导用的非零计数网格**（§9.2.1 第 4~7 步）。
///
/// 存的是"相邻 4×4 块该报给 `nC` 的那个数"：`I_PCM` 报 16；`Intra_16x16`
/// 与色度块报 **AC** 的个数（注释见模块抬头）；其余报全量 `TotalCoeff`。
/// `CodedBlockPattern` 对应位为 0 的块自然就是 0，不用特判。
pub struct TcMap {
    /// 亮度：`mb_cols*4` × `mb_rows*4`
    luma: Vec<u8>,
    /// Cb / Cr：各 `mb_cols*2` × `mb_rows*2`
    cb: Vec<u8>,
    cr: Vec<u8>,
    /// 亮度 4×4 网格的**列数**（每行 `cols4` 个）—— 行数由 `luma.len()/cols4` 得出
    cols4: usize,
}

impl TcMap {
    pub fn new(mb_cols: usize, mb_rows: usize) -> Self {
        let (cols4, rows4) = (mb_cols * 4, mb_rows * 4);
        Self {
            luma: vec![0; cols4 * rows4],
            cb: vec![0; (cols4 / 2) * (rows4 / 2)],
            cr: vec![0; (cols4 / 2) * (rows4 / 2)],
            cols4,
        }
    }

    #[inline]
    fn luma_set(&mut self, x4: usize, y4: usize, v: u8) {
        self.luma[y4 * self.cols4 + x4] = v;
    }

    /// 亮度 4×4 块 `(x4, y4)` 的 `nC`：左邻与上邻都**只在画面内**才算可用。
    pub fn luma_nc(&self, x4: usize, y4: usize) -> i32 {
        let a = (x4 > 0).then(|| self.luma[y4 * self.cols4 + x4 - 1] as i32);
        let b = (y4 > 0).then(|| self.luma[(y4 - 1) * self.cols4 + x4] as i32);
        combine(a, b)
    }

    #[inline]
    fn chroma_set(v: &mut [u8], cols: usize, x4: usize, y4: usize, n: u8) {
        v[y4 * cols + x4] = n;
    }

    fn chroma_nc(v: &[u8], cols: usize, x4: usize, y4: usize) -> i32 {
        let a = (x4 > 0).then(|| v[y4 * cols + x4 - 1] as i32);
        let b = (y4 > 0).then(|| v[(y4 - 1) * cols + x4] as i32);
        combine(a, b)
    }

    pub fn cb_nc(&self, x4: usize, y4: usize) -> i32 {
        Self::chroma_nc(&self.cb, self.cols4 / 2, x4, y4)
    }

    pub fn cr_nc(&self, x4: usize, y4: usize) -> i32 {
        Self::chroma_nc(&self.cr, self.cols4 / 2, x4, y4)
    }
}

/// §9.2.1 第 7 步。
fn combine(a: Option<i32>, b: Option<i32>) -> i32 {
    match (a, b) {
        (Some(a), Some(b)) => (a + b + 1) >> 1,
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => 0,
    }
}

/// 一张图的宏块级产物。
pub struct PictureInfo {
    pub mb_cols: usize,
    pub mb_rows: usize,
    pub mb: Vec<MbInfo>,
    pub coeffs: Vec<MbCoeffs>,
    pub tc: TcMap,
}

impl PictureInfo {
    pub fn new(mb_cols: usize, mb_rows: usize) -> Self {
        Self {
            mb_cols,
            mb_rows,
            mb: vec![MbInfo::default(); mb_cols * mb_rows],
            coeffs: vec![MbCoeffs::default(); mb_cols * mb_rows],
            tc: TcMap::new(mb_cols, mb_rows),
        }
    }

    #[inline]
    fn mb_at(&mut self, addr: usize) -> (&mut MbInfo, &mut MbCoeffs) {
        (&mut self.mb[addr], &mut self.coeffs[addr])
    }
}

/// 解一条 **I 切片**的数据部分，把结果写进 `info`。
///
/// `raw` 是"NAL 头 + 去竞争字节后的负载"，与切片头解析用的是同一条缓冲。
pub fn decode_i_slice(
    header: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    raw: &[u8],
    info: &mut PictureInfo,
) -> Result<()> {
    if !header.is_i() {
        return Err(MediaError::Unsupported(format!(
            "宏块层现在只接了 I 切片（这条是 slice_type = {}），P/B 见 h264/mod.rs 抬头那张表",
            header.slice_type
        )));
    }
    if pps.entropy_coding_mode_flag {
        return Err(MediaError::Unsupported(
            "这条流用的是 CABAC（entropy_coding_mode_flag = 1），自研软解还没接 —— 见 h264/mod.rs 抬头那张表"
                .into(),
        ));
    }
    let mut c = Cavlc::new(raw, header.slice_data_bit_offset)?;
    let total_mb = info.mb_cols * info.mb_rows;
    let mut addr = header.first_mb_in_slice as usize;
    if addr >= total_mb {
        return Err(MediaError::Bitstream(format!(
            "first_mb_in_slice = {} 超过了这张图的宏块数 {total_mb}",
            header.first_mb_in_slice
        )));
    }
    // §7.3.4：`QPY` 在切片内**累加**（`mb_qp_delta` 是相对上一个宏块的增量）
    let mut qp_y = 26 + pps.pic_init_qp_minus26 + header.slice_qp_delta;
    loop {
        qp_y = macroblock_layer(&mut c, sps, pps, info, addr, qp_y)?;
        if !c.more_rbsp_data() {
            break;
        }
        addr += 1;
        if addr >= total_mb {
            return Err(MediaError::Bitstream(format!(
                "宏块还没解完就到了图末尾（已解 {addr} 个，图里只有 {total_mb} 个）—— 多半是语法错位"
            )));
        }
    }
    c.check_rbsp_trailing()?;
    Ok(())
}

/// §7.3.5 `macroblock_layer()`；返回这个宏块之后的 `QPY`。
fn macroblock_layer(
    c: &mut Cavlc<'_>,
    sps: &Sps,
    pps: &Pps,
    info: &mut PictureInfo,
    addr: usize,
    mut qp_y: i32,
) -> Result<i32> {
    let mb_type = c.ue()?;
    let (mbx, mby) = (addr % info.mb_cols, addr / info.mb_cols);
    let (x4_0, y4_0) = (mbx * 4, mby * 4);

    if mb_type > I_PCM {
        return Err(MediaError::Unsupported(format!(
            "I 切片里出现了 mb_type = {mb_type}（表 7-11 只到 25 = I_PCM）"
        )));
    }

    if mb_type == I_PCM {
        // I_PCM：对齐之后**原样** 8 位一读写进画面，没有残差、没有 QP 增量。
        // 去块滤波时它的 QPY 要当 0 用（表 8-16 那一节），所以 `qp_y` 不动、
        // 只用 `is_pcm` 打标。
        c.align_zero_bits()?;
        // `pcm_sample_luma`（256）+ `pcm_sample_chroma`（Cb 64 + Cr 64，4:2:0）
        let mut pcm = Vec::with_capacity(256 + 2 * MB_W_C * MB_H_C);
        for _ in 0..256 + 2 * MB_W_C * MB_H_C {
            pcm.push(c.byte()?);
        }
        let (mb, coeffs) = info.mb_at(addr);
        *mb = MbInfo {
            mb_type,
            intra_pred: IntraPred::Pcm,
            is_pcm: true,
            qp_y,
            ..MbInfo::default()
        };
        *coeffs = MbCoeffs { pcm, ..MbCoeffs::default() };
        // 相邻块看到的"这个块的 TotalCoeff"是 16（§9.2.1 第 6 步）
        for dy in 0..4 {
            for dx in 0..4 {
                info.tc.luma_set(x4_0 + dx, y4_0 + dy, 16);
            }
        }
        let ccols = info.tc.cols4 / 2;
        // ⚠️ 色度网格是按 **4×4 块**数的：一个宏块占 2×2 个色度 4×4 块，
        // 所以起点是 `mbx * 2` —— 写成 `mbx * MB_W_C`（色度**样本**数）会越界。
        for dy in 0..2 {
            for dx in 0..2 {
                TcMap::chroma_set(&mut info.tc.cb, ccols, mbx * 2 + dx, mby * 2 + dy, 16);
                TcMap::chroma_set(&mut info.tc.cr, ccols, mbx * 2 + dx, mby * 2 + dy, 16);
            }
        }
        return Ok(qp_y);
    }

    // ── 预测模式 ────────────────────────────────────────────────────────────
    let mut transform_8x8 = false;
    if pps.transform_8x8_mode_flag && mb_type == I_NXN {
        transform_8x8 = c.u(1)? == 1;
    }
    let parts = i16x16_parts(mb_type);
    let intra_pred = match parts {
        Some(_) => IntraPred::I16x16,
        None if transform_8x8 => IntraPred::NxN8x8,
        None => IntraPred::NxN4x4,
    };

    let mut mb = MbInfo {
        mb_type,
        intra_pred,
        transform_8x8,
        ..MbInfo::default()
    };
    if let Some(p) = parts {
        mb.intra16x16_pred_mode = p.pred_mode;
        mb.cbp_chroma = p.cbp_chroma;
        mb.cbp_luma = p.cbp_luma;
    }

    // mb_pred()：帧内那半（§7.3.5.1）
    match intra_pred {
        IntraPred::NxN4x4 => {
            for i in 0..16 {
                mb.intra4x4_prev[i] = c.u(1)? == 1;
                if !mb.intra4x4_prev[i] {
                    mb.intra4x4_rem[i] = c.u(3)? as u8;
                }
            }
        }
        IntraPred::NxN8x8 => {
            for i in 0..4 {
                mb.intra8x8_prev[i] = c.u(1)? == 1;
                if !mb.intra8x8_prev[i] {
                    mb.intra8x8_rem[i] = c.u(3)? as u8;
                }
            }
        }
        _ => {}
    }
    if sps.chroma_format_idc != 1 {
        return Err(MediaError::Unsupported(
            "色度格式不是 4:2:0，自研软解还没接".into(),
        ));
    }
    // §7.3.5.1：`intra_chroma_pred_mode` 在**三种帧内模式都要读**
    // （包括 `Intra_16x16` —— 漏读会让整个宏块从这一位起错位）
    if intra_pred != IntraPred::Pcm {
        mb.intra_chroma_pred_mode = c.ue()? as u8;
        if mb.intra_chroma_pred_mode > 4 {
            return Err(MediaError::Bitstream(format!(
                "intra_chroma_pred_mode = {} 越界（0..4）",
                mb.intra_chroma_pred_mode
            )));
        }
    }

    // coded_block_pattern：I 切片一律走表 9-4(a) 的"帧内"那一列
    if intra_pred != IntraPred::I16x16 {
        let code_num = c.ue()? as usize;
        if code_num >= 48 {
            return Err(MediaError::Bitstream(format!(
                "coded_block_pattern 的 codeNum = {code_num} 越界（0..47）"
            )));
        }
        // ⚠️ 拆法见式 (7-38)：**低位是亮度、高位是色度**
        // （`Luma = cbp % 16`、`Chroma = cbp / 16`）—— 反过来会让每个宏块的残差
        // 布点整体错位，画面糊成一片但语法照样走得下去。
        let cbp = CODED_BLOCK_PATTERN_INTRA[code_num];
        let _ = CODED_BLOCK_PATTERN_INTER; // 帧间那一列等 P/B 接上再用
        mb.cbp_luma = (cbp % 16) as u32;
        mb.cbp_chroma = (cbp / 16) as u32;
    }

    // mb_qp_delta 与 residual()
    if mb.cbp_luma > 0 || mb.cbp_chroma > 0 || intra_pred == IntraPred::I16x16 {
        qp_y += c.se()?;
        if !(0..=51).contains(&qp_y) {
            return Err(MediaError::Bitstream(format!(
                "这个宏块的 QPY = {qp_y} 越界（0..51）—— 多半是语法错位"
            )));
        }
        mb.qp_y = qp_y;
        residual(c, sps, pps, info, addr, &mb)?;
    } else {
        mb.qp_y = qp_y;
    }

    // `H264_TRACE_MB=1`：逐宏块的语法摘要（排查"哪一类宏块开始不对"最快的一条）
    if trace_mb() {
        let nz = |b: &[i32; 16]| b.iter().filter(|&&v| v != 0).count();
        let c = &info.coeffs[addr];
        let nz_ac: usize = (0..16).map(|i| nz(&c.luma4[i])).sum();
        tracing::info!(
            "MB {addr}: type={mb_type} pred={:?} mode16={} chroma={} cbp=({:x},{}) t8={} qp={qp_y} \
             dc_nz={} ac_nz={nz_ac}",
            mb.intra_pred,
            mb.intra16x16_pred_mode,
            mb.intra_chroma_pred_mode,
            mb.cbp_luma,
            mb.cbp_chroma,
            mb.transform_8x8,
            nz(&c.luma_dc),
        );
    }
    let (mb_slot, _) = info.mb_at(addr);
    *mb_slot = mb;
    Ok(qp_y)
}

/// §7.3.5.3 `residual(0, 15)`。
fn residual(
    c: &mut Cavlc<'_>,
    _sps: &Sps,
    _pps: &Pps,
    info: &mut PictureInfo,
    addr: usize,
    mb: &MbInfo,
) -> Result<()> {
    let mbx = addr % info.mb_cols;
    let mby = addr / info.mb_cols;
    let (x4_0, y4_0) = (mbx * 4, mby * 4);
    let mut coeffs = MbCoeffs::default();

    // ── 亮度 ────────────────────────────────────────────────────────────────
    //
    // ⚠️ 两个**很容易搞反**的地方，都在 §7.3.5.3.1 的这几行里：
    //
    // 1. **`CodedBlockPatternLuma` 是 4 位、一位管一个 8×8 组**
    //    （条件写的是 `& (1 << i8x8)`，不是 `1 << (i8x8*4+i4x4)`）。
    //    按 16 位去测会让"只该有 4 个块有残差"的宏块多解或少解 12 个块。
    // 2. **CAVLC 下即使 `transform_size_8x8_flag == 1`，系数也是按 4 个 4×4 读的**
    //    （条件 `!transform_size_8x8_flag || !entropy_coding_mode_flag` 恒真），
    //    读完再按 `level8x8[i8x8][4*i + i4x4] = level4x4[...][i]` 反交错进 8×8 数组。
    //    只有 CABAC 才走 `maxNumCoeff = 64` 那一支 —— 那一支连 `total_zeros`
    //    的表都只到 `tzVlcIndex = 15`，所以按 64 去读必然撞墙。
    let is_i16 = mb.intra_pred == IntraPred::I16x16;
    if is_i16 {
        // `Intra16x16DCLevel`：nC 按 `luma4x4BlkIdx = 0` 的邻居算（NOTE 2）
        let n_c = info.tc.luma_nc(x4_0, y4_0);
        let (blk, _) = c.residual_block(16, n_c)?;
        coeffs.luma_dc = blk;
    }
    for i8x8 in 0..4usize {
        let has_group = mb.cbp_luma & (1 << i8x8) != 0;
        for i4x4 in 0..4usize {
            let idx = i8x8 * 4 + i4x4;
            // ⚠️ luma4x4BlkIdx 是**象限序**（图 6-10），不是行主序；
            // `blk4_xy` 给的是**像素**偏移，而 `x4_0/y4_0` 是 4×4 块单位 —— 要除 4
            let (dx, dy) = super::reconstruct::blk4_xy(idx);
            let bx = x4_0 + dx / 4;
            let by = y4_0 + dy / 4;
            if !has_group {
                info.tc.luma_set(bx, by, 0);
                continue;
            }
            let n_c = info.tc.luma_nc(bx, by);
            if is_i16 {
                // AC：15 个系数，`maxNumCoeff = 15`；DC 在单独那块里
                let (blk, _) = c.residual_block(15, n_c)?;
                coeffs.luma4[idx] = blk;
                info.tc.luma_set(bx, by, total_of(&blk, 15) as u8);
            } else {
                let (blk, _) = c.residual_block(16, n_c)?;
                coeffs.luma4[idx] = blk;
                info.tc.luma_set(bx, by, total_of(&blk, 16) as u8);
                if mb.transform_8x8 {
                    // §7.3.5.3.1 的原文（`residual_luma`）就是这么一行：
                    //
                    // ```text
                    // if( !entropy_coding_mode_flag && transform_size_8x8_flag )
                    //     for( i = 0; i < 16; i++ )
                    //         level8x8[ i8x8 ][ 4 * i + i4x4 ] = level4x4[ i8x8 * 4 + i4x4 ][ i ];
                    // ```
                    //
                    // 也就是**按 `4 * i + i4x4` 直接交错**，不是"把子块摆回空间位置"。
                    // ⚠️ 这条我一度按"zigzag 槽位"重写过（看着更"对"），
                    // 结果 8×8 用例整帧偏 —— 规范怎么写的就怎么抄。
                    for (i, &v) in blk.iter().enumerate() {
                        coeffs.luma8[i8x8][4 * i + i4x4] = v;
                    }
                }
            }
        }
    }

    // ── 色度（4:2:0） ───────────────────────────────────────────────────────
    let ccols = info.tc.cols4 / 2;
    // 色度网格按 4×4 块数：一个宏块 = 2×2 个色度 4×4 块
    let (cx0, cy0) = (mbx * 2, mby * 2);
    // 表 7-15 的口径：`cbp_chroma != 0` ⇒ DC 一定解（值 2 是"DC 可能有"），
    // `& 2` ⇒ AC 解。所以 DC 的条件是 `!= 0`、AC 的条件是 `& 2`。
    let chroma_dc = mb.cbp_chroma != 0;
    let chroma_ac = mb.cbp_chroma & 2 != 0;
    for i_cbcr in 0..2 {
        // ⚠️ 位序是 §7.3.5.3 定的，**不能按分量交替读**：
        //
        // ```text
        // for( iCbCr = 0; iCbCr < 2; iCbCr++ )      ← 两个分量的 DC 先读完
        //     if( (CodedBlockPatternChroma & 3) && startIdx == 0 )
        //         residual_block( ChromaDCLevel[iCbCr], 0, 4*NumC8x8−1, 4*NumC8x8 )
        // for( iCbCr = 0; iCbCr < 2; iCbCr++ )      ← 再读 AC
        //     for( i8x8 ... ) for( i4x4 ... )
        //         if( CodedBlockPatternChroma & 2 )
        //             residual_block( ChromaACLevel[iCbCr][i8x8*4+i4x4], ..., 15 )
        // ```
        //
        // 也就是 **Cb DC → Cr DC → Cb AC×4 → Cr AC×4**。写成
        // 「Cb DC → Cb AC×4 → Cr DC → Cr AC×4」在 `cbp_chroma == 1`（只有 DC）时
        // 恰好一样，所以纯 DC 的用例全过；一旦有 AC（`& 2`）整条流就从这里串开，
        // 症状是 CAVLC 报"run 越界"之类的**看起来像表抄错**的错。
        let dc_nc = -1; // 4:2:0 的色度 DC 固定 nC = −1（§9.2.1）
        let mut dc = [0i32; 4];
        if chroma_dc {
            let (blk, _) = c.residual_block(4, dc_nc)?;
            dc.copy_from_slice(&blk[..4]);
        }
        if i_cbcr == 0 {
            coeffs.cb_dc = dc;
        } else {
            coeffs.cr_dc = dc;
        }
    }

    // 现在才轮到色度 AC —— 同样先 Cb 四块、再 Cr 四块
    for i_cbcr in 0..2 {
        // ⚠️ `NumC8x8 = 4 / (SubWidthC * SubHeightC)`（§7.3.5.3）—— **4:2:0 时是 1**，
        // 也就是色度 AC 这里只循环一圈 `i4x4 = 0..3`（一个 8×8 色度区域正好是四块
        // 4×4）。写成 `i8x8 = 0..4` 会把坐标乘多一倍，直接越界。
        for i4x4 in 0..4usize {
            let bx = cx0 + (i4x4 % 2);
            let by = cy0 + (i4x4 / 2);
            if !chroma_ac {
                TcMap::chroma_set(
                    if i_cbcr == 0 { &mut info.tc.cb } else { &mut info.tc.cr },
                    ccols,
                    bx,
                    by,
                    0,
                );
                continue;
            }
            let n_c = if i_cbcr == 0 {
                info.tc.cb_nc(bx, by)
            } else {
                info.tc.cr_nc(bx, by)
            };
            let (blk, _) = c.residual_block(15, n_c)?;
            if i_cbcr == 0 {
                coeffs.cb4[i4x4] = blk;
                TcMap::chroma_set(&mut info.tc.cb, ccols, bx, by, total_of(&blk, 15) as u8);
            } else {
                coeffs.cr4[i4x4] = blk;
                TcMap::chroma_set(&mut info.tc.cr, ccols, bx, by, total_of(&blk, 15) as u8);
            }
        }
    }

    info.coeffs[addr] = coeffs;
    Ok(())
}

/// `H264_TRACE_MB=1` 时逐宏块打摘要（热路径上每次都读环境变量太贵，缓存一次）。
fn trace_mb() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("H264_TRACE_MB").is_ok_and(|v| v != "0"))
}

/// 一个块里非零系数的个数（写回 `nC` 网格用）。
fn total_of(blk: &[i32; 16], len: usize) -> u32 {
    blk[..len].iter().filter(|&&v| v != 0).count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h264 as ps_parser;

    /// 拆开夹具：`u32 be avcC 长度 | avcC | u32 be 采样长度 | 采样`。
    ///
    /// 夹具是 [`tools/h264ref/dump_fixture.py`](../../../../tools/h264ref/dump_fixture.py)
    /// 从 libx264 编出来的 mp4 里抠的**真实访问单元** —— 不是手造位流。
    /// 手造位流只能证明"位序自洽"（写测试的和写解析器的是同一份理解，会一起错），
    /// 真实码流才能证伪"规范理解"。
    fn split_fixture(bytes: &[u8]) -> (&[u8], &[u8]) {
        let l = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let avcc = &bytes[4..4 + l];
        let s0 = 4 + l;
        let s = u32::from_be_bytes(bytes[s0..s0 + 4].try_into().unwrap()) as usize;
        (avcc, &bytes[s0 + 4..s0 + 4 + s])
    }

    /// 把夹具里的 I 切片解到语法层；成功 = **一位不差地读完了整条切片**。
    fn decode_fixture(bytes: &[u8]) -> Result<PictureInfo> {
        let (avcc, sample) = split_fixture(bytes);
        let cfg = ps_parser::Config::parse(avcc)?;
        let mut ps = crate::video::h264::nal::ParamSets::new();
        ps.load_from_avcc(avcc)?;
        let nals = ps_parser::split_avcc_sample(sample, cfg.nal_length_size)?;
        let mut out = None;
        for nal in nals {
            if !ps_parser::is_vcl(crate::video::h264::nal::nal_type(nal)) {
                continue;
            }
            let pps_id = crate::video::h264::nal::read_slice_pps_id(nal)?;
            let pps = ps.pps(pps_id)?.clone();
            let sps = ps.sps(pps.sps_id)?.clone();
            let header = ps_parser::parse_slice_header(nal, &sps, &pps)?;
            header.check_supported()?;
            let raw = crate::video::h264::nal::rbsp_of_nal(nal);
            let mut info =
                PictureInfo::new(sps.coded_width() as usize / 16, sps.coded_height() as usize / 16);
            decode_i_slice(&header, &sps, &pps, &raw, &mut info)?;
            out = Some(info);
        }
        out.ok_or_else(|| MediaError::Bitstream("夹具里没有 VCL 切片".into()))
    }

    /// **这一层最硬的判据**：libx264 用 Baseline（CAVLC）写出来的 IDR 切片，
    /// 从 `slice_header` 到最后一个宏块，读位必须**正好**停在 `rbsp_trailing_bits`。
    ///
    /// 多读一位（语法多解了东西）或少读一位（漏了分支）都对不上，而且
    /// **不需要参考解码器**。`coeff_token` 选错列、`nC` 推导差一格、
    /// `coded_block_pattern` 拆反（式 7-38 的 `% 16` / `/ 16`）、
    /// `NumC8x8` 记成 4 —— 这些都能被它抓出来。
    #[test]
    fn a_real_cavlc_i_slice_consumes_the_bitstream_exactly() {
        let info = decode_fixture(include_bytes!("testdata/i_slice_cavlc.bin"))
            .expect("平场 96×64 的 IDR 切片应当能一位不差地解完");
        assert_eq!(info.mb.len(), 24, "6×4 个宏块");
        assert!(
            info.mb.iter().all(|m| m.mb_type != u32::MAX),
            "24 个宏块都得解到（有一个没解到就说明循环提前停了）"
        );
    }

    /// 同一件事，换一张内容更"难"的流：四个各自平坦的象限 ⇒
    /// 帧内预测方向、`coded_block_pattern` 的取值分布都不一样。
    #[test]
    fn a_real_cavlc_i_slice_with_quadrants_also_consumes_exactly() {
        let info = decode_fixture(include_bytes!("testdata/i_slice_cavlc_quadrants.bin"))
            .expect("四象限 96×64 的 IDR 切片应当能一位不差地解完");
        assert_eq!(info.mb.len(), 24);
        assert!(info.mb.iter().all(|m| m.mb_type != u32::MAX));
    }

    /// **整帧逐样本 == 参考解码器** —— 这一层最外的判据。
    ///
    /// 夹具里同时固化了**同一个访问单元**用 FFmpeg 原生 h264 解码器解出来的
    /// 裸 NV12（见 `tools/h264ref/dump_fixture.py --ref-out`），所以这条测试
    /// 比的是**外部实现**，不是"拿自己的实现跟自己的理解对"。
    ///
    /// 它盖住的范围：切片头 → CAVLC → 宏块层 → 帧内预测（16×16 / 4×4 / 8×8、
    /// 色度四种、参考样本可用性）→ 反扫描 / 反量化 / 反变换（含 `Intra_16x16`
    /// 的 DC Hadamard 与色度 DC 2×2）→ 加残差 → NV12 打包。
    ///
    /// 覆盖范围包含 **§8.7 去块滤波**（参考帧是正常解码的结果）。
    /// ⚠️ `x264-params deblock=0:0` 并不是"关掉去块"，只是"用 QP 推导的默认强度"，
    /// 所以这两个夹具的参考帧里是带去块的。
    fn reconstruct_fixture(bytes: &[u8]) -> Result<Vec<u8>> {
        let (avcc, sample) = split_fixture(bytes);
        let cfg = ps_parser::Config::parse(avcc)?;
        let mut ps = crate::video::h264::nal::ParamSets::new();
        ps.load_from_avcc(avcc)?;
        for nal in ps_parser::split_avcc_sample(sample, cfg.nal_length_size)? {
            if !ps_parser::is_vcl(crate::video::h264::nal::nal_type(nal)) {
                continue;
            }
            let pps_id = crate::video::h264::nal::read_slice_pps_id(nal)?;
            let pps = ps.pps(pps_id)?.clone();
            let sps = ps.sps(pps.sps_id)?.clone();
            let header = ps_parser::parse_slice_header(nal, &sps, &pps)?;
            header.check_supported()?;
            let raw = crate::video::h264::nal::rbsp_of_nal(nal);
            let mut info =
                PictureInfo::new(sps.coded_width() as usize / 16, sps.coded_height() as usize / 16);
            decode_i_slice(&header, &sps, &pps, &raw, &mut info)?;
            let mut frame = crate::video::h264::frame::FrameBuf::for_sps(&sps);
            crate::video::h264::reconstruct::reconstruct_picture(&mut frame, &mut info, &sps, &pps)?;
            // §8.7：去块在整图重建之后做，参考帧也是**含去块**的
            crate::video::h264::deblock::deblock_picture(&mut frame, &mut info, &pps, &header)?;
            return Ok(frame.to_nv12());
        }
        Err(MediaError::Bitstream("夹具里没有 VCL 切片".into()))
    }

    #[test]
    fn a_real_i_slice_reconstructs_bit_exactly_like_the_reference() {
        for (name, fixture, reference) in [
            (
                "平场 96×64",
                include_bytes!("testdata/i_slice_cavlc.bin") as &[u8],
                include_bytes!("testdata/i_slice_cavlc.nv12") as &[u8],
            ),
            (
                "四象限 96×64",
                include_bytes!("testdata/i_slice_cavlc_quadrants.bin") as &[u8],
                include_bytes!("testdata/i_slice_cavlc_quadrants.nv12") as &[u8],
            ),
        ] {
            let got = reconstruct_fixture(fixture)
                .unwrap_or_else(|e| panic!("{name}：重建失败 {e}"));
            assert_eq!(got.len(), reference.len(), "{name}：NV12 长度");
            if let Some(i) = got.iter().zip(reference).position(|(a, b)| a != b) {
                let (w, h) = (96usize, 64usize);
                let plane = if i < w * h { "Y" } else { "CbCr" };
                panic!(
                    "{name}：{plane} 平面第 {i} 个样本不一致（我们 {} / 参考 {}）\n\
                     共 {} 处不一致",
                    got[i],
                    reference[i],
                    got.iter().zip(reference).filter(|(a, b)| a != b).count()
                );
            }
        }
    }

    /// 表 7-11 的结构：`1 + 模式×4 + 色度 cbp×3 + 亮度 cbp×2`。
    #[test]
    fn i16x16_mb_type_splits_into_three_parts() {
        let p = i16x16_parts(1).unwrap();
        assert_eq!((p.pred_mode, p.cbp_chroma, p.cbp_luma), (0, 0, 0), "I_16x16_0_0_0");
        let p = i16x16_parts(4).unwrap();
        assert_eq!((p.pred_mode, p.cbp_chroma, p.cbp_luma), (3, 0, 0), "I_16x16_3_0_0");
        let p = i16x16_parts(5).unwrap();
        assert_eq!((p.pred_mode, p.cbp_chroma, p.cbp_luma), (0, 1, 0), "I_16x16_0_1_0");
        let p = i16x16_parts(13).unwrap();
        assert_eq!((p.pred_mode, p.cbp_chroma, p.cbp_luma), (0, 0, 15), "I_16x16_0_0_1");
        let p = i16x16_parts(24).unwrap();
        assert_eq!((p.pred_mode, p.cbp_chroma, p.cbp_luma), (3, 2, 15), "I_16x16_3_2_1");
        assert!(i16x16_parts(0).is_none(), "I_NxN 不是 I_16x16");
        assert!(i16x16_parts(25).is_none(), "I_PCM 不是 I_16x16");
    }

    /// §9.2.1 第 7 步的三种合并方式。
    #[test]
    fn nc_combines_neighbours_like_the_spec() {
        assert_eq!(combine(Some(4), Some(6)), 5, "(4+6+1)>>1");
        assert_eq!(combine(Some(4), None), 4);
        assert_eq!(combine(None, Some(6)), 6);
        assert_eq!(combine(None, None), 0);
    }

    /// 可用性就是"在不在画面里"：最左列没有左邻、最上行没有上邻。
    #[test]
    fn tc_map_treats_out_of_picture_as_unavailable() {
        let mut m = TcMap::new(2, 2);
        m.luma_set(0, 0, 7);
        m.luma_set(1, 0, 5);
        m.luma_set(0, 1, 3);
        // (1,1) 的左邻是 (0,1)=3、上邻是 (1,0)=5
        assert_eq!(m.luma_nc(1, 1), (3 + 5 + 1) >> 1);
        // (0,0) 两边都在画面外
        assert_eq!(m.luma_nc(0, 0), 0);
        // (1,0) 只有左邻
        assert_eq!(m.luma_nc(1, 0), 7);
        // (0,1) 只有上邻
        assert_eq!(m.luma_nc(0, 1), 7, "上邻是 (0,0) = 7");
    }
}
