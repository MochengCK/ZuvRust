//! **手写 SIMD 内核**（`std::arch`，运行时探测；不引任何第三方库）。
//!
//! ## 为什么必须自己写
//!
//! 同机同素材实测：dav1d 单线程 11.4 ms/帧（88 fps），我们优化前 ~1017 ms/帧。
//! dav1d 的省力点就两样 —— **SIMD 内核**与**线程架构**。这一层补的是第一样：
//! 按 dav1d `src/x86/*.asm` / libaom `av1/common/*_sse2.c` 的**做法**重写，
//! 但不抄它们的代码（我们只用标准 intrinsic，且每一条都与标量路径**逐位等价**）。
//!
//! ## 约定
//!
//! 1. **运行时探测**：`avx2_available()`（进程内判一次；`ME_NO_SIMD=1` 可强制走标量，
//!    用来做"开了 SIMD 与没开 SIMD 输出是否逐字节一致"的对拍）。
//! 2. **只吃内点**：SIMD 版本要求窗口完整落在帧内（左右各留够抽头）；
//!    边界块与尾部一律**回落到标量路径** —— 标量路径已经被逐样本对拍验过，
//!    这样"快路径 + 慢路径"合起来仍然是逐位一致的。
//! 3. **算术顺序变了但结果不变**：整数加法可结合、可交换，**但值域要逐条自己算** ——
//!    这一条曾经被写成"系数绝对值和 ≤ 128"，那是**错的**（见下），
//!    结果引入过一次像素级回归。
//! 4. **非 x86_64 平台**（Apple Silicon 的 arm64 构建）：本模块整体退化为"没有 SIMD"，
//!    调用方自动走标量 —— 所以这份代码在任何平台都能编译、行为一致（只是慢些）。
//!
//! ## ⚠️ 一次像素级回归的教训：**位宽假设必须按"前缀和"算，不能拍 `Σ|c|`**
//!
//! `SUBPEL_FILTERS` 每行的**代数和**恒为 128（这是"滤波器保直流"的要求），
//! 但**绝对值和**不是：96 个相位里 48 个 `Σ|c| > 128`，最坏
//! `[-4,12,-24,80,80,-24,12,-4]` 是 **240**。`mc_horiz16_8bpc` 当初按
//! "`128·255 = 32640 < 32767` ⇒ 装得进 i16"来设计，**前提就错了**：
//! 240·255 = 61200 会回绕，而且判据是**累加过程中的每个前缀和**（同样能到 240），
//! 不是最后那一项。症状是"整像素 MV 全对、只有分数 MV 偶尔几个像素错到 255"——
//! 因为相位 0（δ）只有一个非零抽头，天然不溢出。
//!
//! 两条护栏（都不是"应该没事"，而是可执行的断言）：
//! * [`x86::mc_horiz16_8bpc`] 改用**带偏置的 i16 累加**（[`HORIZ16_BIAS`]），
//!   96 相位 × 8 前缀全值域枚举证明落在 i16 内，指令数不变；
//! * [`tests::horiz16_bias_bounds_hold_for_every_phase`] 把 `Σ|c|` / 前缀和 /
//!   单项乘积的界全部钉死，改表或改偏置立刻红。
//!
//! 更普适的一条：**自测的输入要能覆盖被考验的那条路径**。旧版
//! `horiz16_matches_scalar_bit_for_bit` 只用常量行（全 255 / 斜坡），
//! 而常量行上 `Σ c·s = s·Σc = 128s ≤ 32640` **永远不溢出** —— 恰好是最盲的盲区。

// `flags::no_simd()` 只在 x86 分支里用到（AVX2 探测），非 x86 上它是未使用导入
// —— 这个警告正是"这段代码只在某个架构下编译过"留下的痕迹，别拿 `allow` 盖掉。
#[cfg(target_arch = "x86_64")]
use super::flags;

/// 测试专用：置位后 [`avx2_available`] 恒为 false —— 用来在同进程里对拍
/// 「SIMD 与标量输出逐字节一致」（生产构建里这段不存在）。
#[cfg(test)]
pub(crate) static FORCE_SCALAR: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// 这台机器能不能用 AVX2 内核（进程内判一次）。
#[inline]
pub(crate) fn avx2_available() -> bool {
    #[cfg(test)]
    if FORCE_SCALAR.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static V: OnceLock<bool> = OnceLock::new();
        *V.get_or_init(|| {
            !flags::no_simd() && std::arch::is_x86_feature_detected!("avx2")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// SIMD 一次处理的输出个数（AVX2 = 8 个 i32 通道）。
pub(crate) const LANES: usize = 8;

/// [`x86::mc_horiz16_8bpc`] 的 i16 累加器偏置：`-128 · 128 = -16384`。
///
/// **这个常数不是随手取的，它是那个内核敢用 16 位通道的全部理由。**
///
/// 原本的假设是"8bit 样本 ≤ 255 且 `Σ|coef| = 128` ⇒ 最坏 `128·255 = 32640`
/// 装得进 i16"。前半句对，**后半句错**：`SUBPEL_FILTERS` 的 96 个真实相位里
/// 有 48 个 `Σ|c| > 128`，最坏 `[-4,12,-24,80,80,-24,12,-4]` 是 **240**
/// （`Σc = 128` 只约束"整行代数和"，管不住逐项绝对值和，也管不住**前缀和**——
/// 前缀和同样能到 240）。所以直接 i16 累加会**回绕**。
///
/// 垫上这个偏置后（全程算 `BIAS + Σ c·s`，展宽到 i32 再 `+16384` 抵消），
/// 对全部 96 相位 × 全部 8 种前缀长度取样本极值，累加值恒落在
/// `[-30664, 30536] ⊂ i16`；`mullo_epi16` 的单项极值 `±128·255 = ±32640` 也在界内。
/// 偏置把值域从 `[0, 32640]` 平移到 `[-16384, 16256]` 附近，正好叫正负两侧的
/// 最坏前缀都躲开回绕。
///
/// 边界由 [`tests::horiz16_bias_bounds_hold_for_every_phase`] 逐步钉住 ——
/// 改滤波器表或改这个常数，它立刻红。
pub(crate) const HORIZ16_BIAS: i16 = -16384;

/// CDEF 一个像素的候选样本数：主抽头 `2`（`k`）× `2`（符号）= 4，
/// 次抽头 `2`（方向 ±2）× `2`（`k`）× `2`（符号）= 8，合计 12（§7.15.3）。
/// 约定前 4 个是主抽头、后 8 个是次抽头（两组的阈值与阻尼不同）。
pub(crate) const CDEF_CANDS: usize = 12;


/// `Round2( x, n )` 的**标量**版（i32），给混合内核的尾巴与非 x86 平台用。
///
/// 与 `inter::round2_i32` 逐位一致（"加半再算术右移"）；`n == 0` 时原样返回。
#[inline]
pub(crate) fn rnd_scalar(x: i32, n: u32) -> i32 {
    if n == 0 {
        x
    } else {
        (x + (1 << (n - 1))) >> n
    }
}

/// 反变换的 1D 类别（与 `transform::Transform1d` 一一对应；无损的 WHT 单独走
/// `lossless` 开关，不占这里的编号）。定义在模块级，非 x86 平台也能引用。
pub(crate) const TXF_DCT: u32 = 0;
pub(crate) const TXF_ADST: u32 = 1;
pub(crate) const TXF_IDENTITY: u32 = 2;

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;
    use std::mem::MaybeUninit;
    use super::super::transform::brev;
    use super::{rnd_scalar, TXF_ADST, TXF_DCT};

    /// `Round2(x, n)`：与 `inter::round2_i32` **逐位等价**（都是"加半再算术右移"）。
    ///
    /// 移位量是编译期常量（`_mm256_srai_epi32` 要立即数），所以写成 match；
    /// 取值只可能来自 `rounding_vars()`：横向 3/5、纵向 7/9/11。
    #[inline(always)]
    pub(super) unsafe fn rnd(acc: __m256i, n: u32) -> __m256i {
        match n {
            0 => acc,
            3 => _mm256_srai_epi32::<3>(_mm256_add_epi32(acc, _mm256_set1_epi32(4))),
            5 => _mm256_srai_epi32::<5>(_mm256_add_epi32(acc, _mm256_set1_epi32(16))),
            7 => _mm256_srai_epi32::<7>(_mm256_add_epi32(acc, _mm256_set1_epi32(64))),
            9 => _mm256_srai_epi32::<9>(_mm256_add_epi32(acc, _mm256_set1_epi32(256))),
            11 => _mm256_srai_epi32::<11>(_mm256_add_epi32(acc, _mm256_set1_epi32(1024))),
            // 不在 `rounding_vars` 的值域里：交由调用方走标量（返回原值，绝不算错）
            _ => acc,
        }
    }

    /// 横向 8 抽头，一次算 **8 个输出**（i32 累加，源样本 u16）。
    ///
    /// 每个输出 `c`：`Σ_t coef[t] · row[base + c + t - 3]`。
    ///
    /// # Safety
    /// - 必须已确认 AVX2 可用（调用方查 [`super::avx2_available`]）。
    /// - `row` 必须覆盖 `[base - 3, base + LANES + 4)`；`out.len() >= 8`。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn mc_horiz8(
        row: &[u16],
        base: isize,
        coef: &[i16; 8],
        round: u32,
        out: &mut [i16],
    ) {
        let p = row.as_ptr();
        let mut acc = _mm256_setzero_si256();
        for (t, c) in coef.iter().enumerate() {
            let s16 = _mm_loadu_si128(p.offset(base + t as isize - 3) as *const __m128i);
            let s32 = _mm256_cvtepu16_epi32(s16);
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s32, _mm256_set1_epi32(*c as i32)));
        }
        // 取整后收窄回 i16 再存（理由同 `mc_horiz16_8bpc`）。最坏边界是 **12bit**：
        // `Σ|c|·4095 = 982800`、`>> inter_round0(5)` ⇒ **30712 < 32767**，仍装得下；
        // 8bit 只有 ±4.5e3。这样全位深共用一套 i16 中间量，纵向的缓存占用减半。
        let v = rnd(acc, round);
        let q = _mm_packs_epi32(
            _mm256_castsi256_si128(v),
            _mm256_extracti128_si256::<1>(v),
        );
        _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, q);
    }

    /// 横向 8 抽头、**一次 16 个输出**（16 位通道；**只用于 8bit**）。
    ///
    /// ## ⚠️ 这里曾经是像素级正确性回归的根因（2026-10-03）
    ///
    /// 最初的注释写着"8bit 下样本 ≤ 255 且 `Σ|coef| = 128` ⇒ 整条累加都能装进
    /// i16（最坏 `128 · 255 = 32640 < 32767`）"。**这个推理是错的**：
    ///
    /// * `Σ|c| = 128` 只是**行代数和**（`Σc`）为 128 —— 它约束的是"滤波器不改变
    ///   直流"，跟**绝对值和**是两回事。96 个真实相位里 **48 个 `Σ|c| > 128`**，
    ///   最坏 `[-4,12,-24,80,80,-24,12,-4]` 的 `Σ|c| = 240`。
    /// * 用 i16 累加不仅要看 `Σ|c|`，还要看**累加过程中的每一个前缀和**；
    ///   逐项绝对值求和的**前缀**同样能到 240 ⇒ `240 · 255 = 61200 > 32767`，**回绕**。
    /// * 症状因此极其"挑食"：相位 0（δ，只有一个非零抽头）永不溢出 ⇒
    ///   **整像素 MV 全对**；只有**分数 MV** 撞上高对比邻域（样本接近 0/255 混排）
    ///   才在块内零星几个像素上错，且 `max|Δ|` 能到 255。它躲过了当时那份
    ///   只用常量行（全 255 / 斜坡）的自测 —— 常量行上 `Σ c·s = s·Σc = 128s`
    ///   永远不溢出，**恰好是误差最大的盲区**。
    ///
    /// 修法：i16 通道改成**带偏置累加**（[`super::HORIZ16_BIAS`]，全程算
    /// `BIAS + Σ c·s`，展宽到 i32 后 `+16384` 抵消回来）。每个抽头仍是一次
    /// `mullo_epi16` + `add_epi16`、仍一次覆盖 16 个输出，**指令数一条不多**，
    /// 而值域被平移到 `[-30664, 30536]`（96 相位 × 8 前缀全值域枚举钉住）⇒
    /// 与 i32 版/标量**逐位一致**。
    ///
    /// # Safety
    /// 必须已确认 AVX2 可用；`row` 要覆盖 `[base-3, base+16+4)`、`out.len() >= 16`。
    ///
    /// ⚠️ **`round` 必须 ≥ 2**（调用方传的是 `inter_round0`，8bit 3 / 12bit 5）：
    /// 输出收窄到 i16 是靠"取整之后值域只有 ±4.5e3（12bit ±30712）"这条论证，
    /// 而 `round = 0`（不做取整）时 `Σ|c|·255` 能到 36720，超出 i16。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn mc_horiz16_8bpc(
        row: &[u16],
        base: isize,
        coef: &[i16; 8],
        round: u32,
        out: &mut [i16],
    ) {
        let p = row.as_ptr();
        // 从偏置起步：整条链都在 i16 内跑，且正负两侧都不回绕（见 HORIZ16_BIAS）。
        let mut acc = _mm256_set1_epi16(super::HORIZ16_BIAS);
        for (t, c) in coef.iter().enumerate() {
            let s = _mm256_loadu_si256(p.offset(base + t as isize - 3) as *const __m256i);
            acc = _mm256_add_epi16(acc, _mm256_mullo_epi16(s, _mm256_set1_epi16(*c)));
        }
        // 展宽到 i32 后把偏置抵消掉 ⇒ 还原成真正的 Σ c·s（值域 [0, 32640]）。
        let bias = _mm256_set1_epi32(-(super::HORIZ16_BIAS as i32));
        let lo = _mm256_add_epi32(
            _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc)),
            bias,
        );
        let hi = _mm256_add_epi32(
            _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(acc)),
            bias,
        );
        // 取整之后**收窄回 i16** 再存（2026-10-04）。8bit 的 `rnd(Σc·s, inter_round0)`
        // 值域只有 `±4.5e3`（`Σ|c|·255 = 61200`，`>>3` 后 7650，实际系数下更小），
        // i16 绰绰有余 —— 而这一收窄让**纵向要读的中间量缓存占用直接减半**
        // （128 宽的块 × (h+7) 行：i32 是 69 KB、超出 L1；i16 只有 34.5 KB）。
        // dav1d 的 `prep_8tap_c` 用的就是 `int16_t mid[128*135]`。
        let packed = _mm256_packs_epi32(rnd(lo, round), rnd(hi, round));
        // `packs_epi32` 按 128 位通道交错（`[lo0..3, hi0..3, lo4..7, hi4..7]`）
        // ⇒ 用 64 位块重排 `0,2,1,3` 还原成 `[lo0..7, hi0..7]`。
        let v = _mm256_permute4x64_epi64::<0xD8>(packed);
        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, v);
    }

    /// **横向趟的整块版**：把 `rows` 里给出的每一行都做完 8 抽头横向滤波，
    /// 一次调用覆盖整块（而不是"每行每 16 列一次调用"）。
    ///
    /// ## 为什么要成批
    ///
    /// `block_inter_prediction` 的实测（`bench_mc_block`）：64×64 的块 **1.97 ns/像素**，
    /// 16×16 的 **3.7 ns/像素** —— 而两个内核自己加起来只要 **1.1 ns/像素**。
    /// 差出来的那部分是**逐行/逐批的脚手架**：每次都要重算行下标（`i64` 钳位）、
    /// 重切 `&[u16]`/`&mut [i16]`（边界检查）、重新进 `#[target_feature]` 函数
    /// （**不能内联**，一次真实调用）。一行 16 列就是 4~5 次这样的开销。
    ///
    /// 这里把"行循环"整个搬进内核：一行只做一次指针推进，内层 16 列一批照旧。
    /// 语义与逐行版**逐位相同**（同一个 `mc_horiz16_8bpc` / `mc_horiz8` / 标量式）。
    ///
    /// `rows[r]` 是**已经钳好的**源行号（调用方负责 `clip(..., last_y)`）；
    /// `interior_x` 为真表示整块在水平方向留够了抽头（`base-3 >= 0` 且
    /// `base + wu + 3 <= last_x`），可以整批走 SIMD；否则整批走标量。
    ///
    /// `Difference weight mask process`（§7.11.3.12）的 **8 路内核**。
    ///
    /// 标量式（`inter.rs::build_diffwtd_mask`）：
    /// `m = Clip3(0, 64, 38 + Round2(|p0 - p1|, shift) / 16)`，
    /// 再按 `mask_type` 取补（`64 - m`）。
    ///
    /// 化简（两个量都是**非负整数**，右移就是向下取整除法，除法位数可以并起来）：
    /// `Round2(x, s) / 16 == ((x + (1 << (s-1))) >> s) >> 4 == (x + (1 << (s-1))) >> (s + 4)`
    /// ⇒ 一次 `add` + **一次**算术右移，省掉中间量。（`>>` 在非负值上与 `/` 一致，
    /// 单测 `diffwtd_mask_shift_fusion_matches_two_step` 把这条钉住。）
    ///
    /// # Safety
    /// 必须已确认 AVX2 可用；`p0`/`p1`/`out` 至少 `n` 项；`shift >= 1`。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn diffwtd_mask8(
        p0: &[i32],
        p1: &[i32],
        n: usize,
        shift: u32,
        mask_type: u8,
        out: &mut [i32],
    ) {
        let half = _mm256_set1_epi32(1i32 << (shift - 1));
        let total = _mm256_set1_epi32((shift + 4) as i32);
        let v38 = _mm256_set1_epi32(38);
        let v64 = _mm256_set1_epi32(64);
        let zero = _mm256_setzero_si256();
        let a0 = p0.as_ptr();
        let a1 = p1.as_ptr();
        let o = out.as_mut_ptr();
        let mut i = 0usize;
        while i + 8 <= n {
            let d = _mm256_abs_epi32(_mm256_sub_epi32(
                _mm256_loadu_si256(a0.add(i) as *const __m256i),
                _mm256_loadu_si256(a1.add(i) as *const __m256i),
            ));
            let m = _mm256_add_epi32(v38, _mm256_srav_epi32(_mm256_add_epi32(d, half), total));
            let m = _mm256_min_epi32(_mm256_max_epi32(m, zero), v64);
            let m = if mask_type != 0 { _mm256_sub_epi32(v64, m) } else { m };
            _mm256_storeu_si256(o.add(i) as *mut __m256i, m);
            i += 8;
        }
        while i < n {
            let d = (*a0.add(i) - *a1.add(i)).abs();
            let m = (38 + ((d + (1 << (shift - 1))) >> (shift + 4))).clamp(0, 64);
            *o.add(i) = if mask_type != 0 { 64 - m } else { m };
            i += 1;
        }
    }

    /// # Safety
    /// 必须已确认 AVX2 可用；`data` 覆盖所有 `rows[r] * stride + [0, stride)`；
    /// `out` 至少 `rows.len() * wu` 项。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn mc_horiz_rows_8bpc(
        data: &[u16],
        stride: usize,
        rows: &[usize],
        base: isize,
        wu: usize,
        coef: &[i16; 8],
        round: u32,
        interior_x: bool,
        last_x: i64,
        out: &mut [i16],
    ) {
        let dp = data.as_ptr();
        let op = out.as_mut_ptr();
        let cv: [__m256i; 8] = std::array::from_fn(|t| _mm256_set1_epi16(coef[t]));
        let bias = _mm256_set1_epi32(-(super::HORIZ16_BIAS as i32));
        for (r, &sy) in rows.iter().enumerate() {
            let row = dp.add(sy * stride);
            let out_row = op.add(r * wu);
            let mut c = 0usize;
            if interior_x {
                while c + 16 <= wu {
                    let p = row.offset(base + c as isize - 3);
                    let mut acc = _mm256_set1_epi16(super::HORIZ16_BIAS);
                    for (t, cvv) in cv.iter().enumerate() {
                        let s = _mm256_loadu_si256(p.add(t) as *const __m256i);
                        acc = _mm256_add_epi16(acc, _mm256_mullo_epi16(s, *cvv));
                    }
                    let lo = _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc)),
                        bias,
                    );
                    let hi = _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(acc)),
                        bias,
                    );
                    let packed = _mm256_packs_epi32(rnd(lo, round), rnd(hi, round));
                    _mm256_storeu_si256(
                        out_row.add(c) as *mut __m256i,
                        _mm256_permute4x64_epi64::<0xD8>(packed),
                    );
                    c += 16;
                }
            }
            while c < wu {
                let bc = base + c as isize;
                let mut s = 0i32;
                for (t, k) in coef.iter().enumerate() {
                    let sx = (bc + t as isize - 3).clamp(0, last_x as isize) as usize;
                    s += *k as i32 * i32::from(*row.add(sx));
                }
                *out_row.add(c) = rnd_scalar(s, round) as i16;
                c += 1;
            }
        }
    }

    /// **纵向趟的整块版**：`h` 行 × `wu` 列一次做完（`pmaddwd` 两两配对，
    /// 见 [`mc_vert8`]）。语义与逐行版逐位相同。
    ///
    /// # Safety
    /// 必须已确认 AVX2 可用；`inter` 至少 `(h + 7) * stride` 项（`stride == wu`）、
    /// `out` 至少 `h * wu` 项。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn mc_vert_rows_8bpc(
        inter: &[i16],
        wu: usize,
        h: usize,
        coef: &[i16; 8],
        round: u32,
        out: *mut i32,
    ) {
        let ip = inter.as_ptr();
        let c64 = _mm256_set1_epi32(64);
        let _ = c64;
        let pairs: [__m256i; 4] = std::array::from_fn(|j| {
            _mm256_set1_epi32(((coef[2 * j + 1] as u16 as i32) << 16) | (coef[2 * j] as u16 as i32))
        });
        for r in 0..h {
            let base = ip.add(r * wu);
            let out_row = out.add(r * wu);
            let mut c = 0usize;
            while c + 16 <= wu {
                let mut lo_acc = _mm256_setzero_si256();
                let mut hi_acc = _mm256_setzero_si256();
                for (j, pair) in pairs.iter().enumerate() {
                    let t = 2 * j;
                    let a = _mm256_loadu_si256(base.add(t * wu + c) as *const __m256i);
                    let b = _mm256_loadu_si256(base.add((t + 1) * wu + c) as *const __m256i);
                    lo_acc = _mm256_add_epi32(
                        lo_acc,
                        _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), *pair),
                    );
                    hi_acc = _mm256_add_epi32(
                        hi_acc,
                        _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), *pair),
                    );
                }
                let lo = rnd(lo_acc, round);
                let hi = rnd(hi_acc, round);
                _mm256_storeu_si256(
                    out_row.add(c) as *mut __m256i,
                    _mm256_permute2x128_si256::<0x20>(lo, hi),
                );
                _mm256_storeu_si256(
                    out_row.add(c + 8) as *mut __m256i,
                    _mm256_permute2x128_si256::<0x31>(lo, hi),
                );
                c += 16;
            }
            while c < wu {
                let mut s = 0i32;
                for (t, k) in coef.iter().enumerate() {
                    s += *k as i32 * i32::from(*ip.add((r + t) * wu + c));
                }
                *out_row.add(c) = rnd_scalar(s, round);
                c += 1;
            }
        }
    }

    /// **整数相位（δ 滤波器）**的横向+纵向合成：一次算 **8 个输出**。
    ///
    /// `pred = Round2(128 · Round2(128 · src, r0), r1)`（见 `inter` 里快路径 1 的说明：
    /// 相位 0 的 8 抽头是 δ，但两趟取整量随位深与复合与否变，不能直接取源样本）。
    ///
    /// # Safety
    /// 同 [`mc_horiz8`]（但只需要覆盖 `[base, base + LANES)`）。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn mc_delta8(row: &[u16], base: isize, r0: u32, r1: u32, out: &mut [i32]) {
        let s16 = _mm_loadu_si128(row.as_ptr().offset(base) as *const __m128i);
        let s32 = _mm256_cvtepu16_epi32(s16);
        let h = rnd(
            _mm256_slli_epi32::<7>(s32),
            r0,
        );
        let v = rnd(_mm256_slli_epi32::<7>(h), r1);
        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, v);
    }

    /// **整数相位的化简形式**：`src << post_round`，一次算 **8 个输出**。
    ///
    /// ## 为什么可以化简（2026-10-02）
    ///
    /// 原始式子是两趟 `Round2`：`Round2(128·src, r0)` 再 `Round2(128··, r1)`。
    /// 因为 `(x + 2^(n-1)) >> n` 里 `x` 恰好是 `2^(7+n') ` 的倍数，两趟的取整量
    /// **正好互相抵消**，结果恒等于
    ///
    /// ```text
    /// src << (2·FILTER_BITS − (r0 + r1))        // = src << post_round
    /// ```
    ///
    /// 对全部四种组合（8/12 位 × 复合/非复合）在**全值域**上验证过：8 位 2×256 个
    /// 输入、12 位 2×4096 个输入，无例外（`post_round` 依次是 0 / 4 / 0 / 2）。
    /// 这个化简不是"看着像"，它把每 8 个通道的两轮 `slli`+`add`+`srai` 变成
    /// **一条变长左移**——而软解里整像素 MV 占绝大多数，这条路径是常客。
    ///
    /// `post_round` 只在 0…4 之间，用 `_mm256_sllv_epi32` 的立即数版本即可。
    ///
    /// # Safety
    /// `sgr_filter` **末段**的 3×3 加权（§7.17.3 的收尾），一次 8 列。
    ///
    /// 规范写法是"每个输出 (i,j) 用 `A`/`B` 表里的 3×3 邻域加权求和"：
    ///
    /// ```text
    /// out[j] = Round2( (Σ_{3×3} w·A)·u + (Σ_{3×3} w·B), shift )
    /// ```
    ///
    /// `u` 取 `UpscaledCdefFrame` 的同一位置。这是 `box_filter` 里**最大的一段**
    /// （一档 64×64 要 9·4096 = 36864 次乘加 ×2 个累加器），而它对 8 个相邻列
    /// **完全独立**（`A`/`B` 同行相邻、`u` 同行相邻）⇒ 天然 8 路。
    ///
    /// # Safety
    /// - 必须已确认 AVX2 可用；
    /// - `base + dyi·aw + dxi + 7 < min(a_buf.len(), b_buf.len())`（调用方保证
    ///   `base` 落在表的行内、且这一批 8 列都在 `w` 之内）；
    /// - `u` 起址往后 8 个 `u16` 可读；`out.len() >= 8`。
    ///
    /// ⚠️ **结果不做 `clip1`** —— 规范在 `box_filter` 这一步只写 `F`，钳位发生在
    /// `sgr_filter` 最后的投影混合里。这里多钳一次会让整条 SGR 路径的像素全错
    /// （第一版就是被"逐字节对拍"抓出来的）。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn sgr_final_8(
        a_buf: &[i32],
        b_buf: &[i32],
        aw: usize,
        base: usize,
        wts: &[[i32; 3]; 3],
        u: *const u16,
        shift: u32,
        out: &mut [i32],
    ) {
        let mut va = _mm256_setzero_si256();
        let mut vb = _mm256_setzero_si256();
        for (dyi, wrow) in wts.iter().enumerate() {
            for (dxi, &cw) in wrow.iter().enumerate() {
                // 权重表里有一半是 0（`pass == 0` 的偶数行、`pass == 1` 的两侧）
                // ⇒ 直接跳过，能省掉近一半的 load+mul+add。
                if cw == 0 {
                    continue;
                }
                let c = _mm256_set1_epi32(cw);
                let idx = base + dyi * aw + dxi;
                let ra = _mm256_loadu_si256(a_buf.as_ptr().add(idx) as *const __m256i);
                let rb = _mm256_loadu_si256(b_buf.as_ptr().add(idx) as *const __m256i);
                va = _mm256_add_epi32(va, _mm256_mullo_epi32(ra, c));
                vb = _mm256_add_epi32(vb, _mm256_mullo_epi32(rb, c));
            }
        }
        let uv = _mm256_cvtepu16_epi32(_mm_loadu_si128(u as *const __m128i));
        let v = _mm256_add_epi32(_mm256_mullo_epi32(va, uv), vb);
        // `round2(v, shift)`：算术右移，与 `lr.rs` 的 `round2` 同一语义。
        // 用**变量**移位（`vpsravd`）——`shift` 是运行期的（由 `pass` 与行的奇偶
        // 决定 8 或 9），而 `_mm256_srai_epi32` 的立即数必须是编译期常量。
        let v = _mm256_srav_epi32(
            _mm256_add_epi32(v, _mm256_set1_epi32(1 << (shift - 1))),
            _mm256_set1_epi32(shift as i32),
        );
        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, v);
    }

    /// `box_filter` **竖直滑动和**的列更新（一次 8 列、i32 通道）：
    /// `cs[k] += add[k] − sub[k]`、`cs2[k] += add[k]² − sub[k]²`。
    ///
    /// `sub_row` 传 `null` 表示"**只加不减**" —— 那一趟是 `i = -1` 的**初始累加**
    /// （行窗口 `[y-1-r, y-1+r]`），调用方先把 `cs`/`cs2` 清零再逐行调它。
    ///
    /// 为什么值得单独做：整列滑动和是 `box_filter` 里最后一块**逐列标量**的热点
    /// （每趟 `h × cols` 次、每次两个**间接**取样 + 4 条 i32 算术）。这里两个取样
    /// 各是一次 8×u16 的连续读（调用方保证列连续：`xs[k] == col0 + k`），后面全是
    /// 逐位相同的 i32 整数运算。
    ///
    /// # Safety
    /// - 必须已确认 AVX2 可用；
    /// - `add_row`（以及非空的 `sub_row`）至少可读 `cols` 个 `u16`；
    /// - `cs`/`cs2` 至少可读写 `cols` 个 `i32`。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn sgr_cols_update8(
        add_row: *const u16,
        sub_row: *const u16,
        cs: *mut i32,
        cs2: *mut i32,
        cols: usize,
    ) {
        let has_sub = !sub_row.is_null();
        let mut k = 0usize;
        while k + 8 <= cols {
            let a = _mm256_cvtepu16_epi32(_mm_loadu_si128(add_row.add(k) as *const __m128i));
            let ca = _mm256_loadu_si256(cs.add(k) as *const __m256i);
            let c2a = _mm256_loadu_si256(cs2.add(k) as *const __m256i);
            let (da, d2a) = if has_sub {
                let s = _mm256_cvtepu16_epi32(_mm_loadu_si128(sub_row.add(k) as *const __m128i));
                (
                    _mm256_sub_epi32(a, s),
                    _mm256_sub_epi32(_mm256_mullo_epi32(a, a), _mm256_mullo_epi32(s, s)),
                )
            } else {
                (a, _mm256_mullo_epi32(a, a))
            };
            _mm256_storeu_si256(cs.add(k) as *mut __m256i, _mm256_add_epi32(ca, da));
            _mm256_storeu_si256(cs2.add(k) as *mut __m256i, _mm256_add_epi32(c2a, d2a));
            k += 8;
        }
        while k < cols {
            let a = *add_row.add(k) as i32;
            if has_sub {
                let s = *sub_row.add(k) as i32;
                *cs.add(k) += a - s;
                *cs2.add(k) += a * a - s * s;
            } else {
                *cs.add(k) += a;
                *cs2.add(k) += a * a;
            }
            k += 1;
        }
    }

    /// `Round2(x, sh)` 的 8 通道版（`sh == 0` 时是恒等）。
    ///
    /// 用**变量**移位（`vpsravd`）：`sh` 是运行期的（8bit 时 A 的移 0、D 的移 0），
    /// 而 `_mm256_srai_epi32` 的立即数必须是编译期常量。
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn r2v(x: __m256i, sh: u32) -> __m256i {
        if sh == 0 {
            x
        } else {
            _mm256_srav_epi32(
                _mm256_add_epi32(x, _mm256_set1_epi32(1 << (sh - 1))),
                _mm256_set1_epi32(sh as i32),
            )
        }
    }

    /// `box_filter` 里 **A/B 表**那段"每输出一次"的算术（§7.17.3），一次 8 列。
    ///
    /// 规范对每个输出 (i,j) 要算：
    ///
    /// ```text
    /// A = Round2(Σ c², 2(bd-8))   D = Round2(Σ c, bd-8)
    /// p = max(0, A·n - D²)        z = Round2(p·s_factor, SGRPROJ_MTABLE_BITS)
    /// a2 = X_BY_XPLUS1[z] 的四个分支   b2 = (256 - a2)·D·one_over_n
    /// ```
    ///
    /// 其中 `Σ c²` / `Σ c` 原本是**水平滑动**（串行增量），调用方已换成
    /// **前缀和相减**（`pre_s2[hi]-pre_s2[lo]`）⇒ 8 个相邻列完全独立，这里就能 8 路。
    ///
    /// ⚠️ `a2` 的查表/除法仍是**逐通道标量**（`z < 0` 那条分支是真实路径：
    /// `p·s_factor` 会在 `i32` 里溢出，见 `lr.rs` 的注释），但它是 8 次标量查表，
    /// 而不是 8 段独立的向量。
    ///
    /// # Safety
    /// - 必须已确认 AVX2 可用；
    /// - `pre_s`/`pre_s2` 至少覆盖 `[lo, hi + 8)`；`a_out`/`b_out` 长度 ≥ 8。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn sgr_ab_8(
        pre_s: &[i32],
        pre_s2: &[i32],
        lo: usize,
        hi: usize,
        n: i32,
        s_factor: i32,
        one_over_n: i32,
        sh_a: u32,
        sh_d: u32,
        mtab_bits: u32,
        recip_bits: u32,
        table: &[i32; 256],
        a_out: &mut [i32],
        b_out: &mut [i32],
    ) {
        let sa = _mm256_sub_epi32(
            _mm256_loadu_si256(pre_s2.as_ptr().add(hi) as *const __m256i),
            _mm256_loadu_si256(pre_s2.as_ptr().add(lo) as *const __m256i),
        );
        let sb = _mm256_sub_epi32(
            _mm256_loadu_si256(pre_s.as_ptr().add(hi) as *const __m256i),
            _mm256_loadu_si256(pre_s.as_ptr().add(lo) as *const __m256i),
        );
        let a = r2v(sa, sh_a);
        let d = r2v(sb, sh_d);
        let p = _mm256_max_epi32(
            _mm256_sub_epi32(
                _mm256_mullo_epi32(a, _mm256_set1_epi32(n)),
                _mm256_mullo_epi32(d, d),
            ),
            _mm256_setzero_si256(),
        );
        let z = r2v(
            _mm256_mullo_epi32(p, _mm256_set1_epi32(s_factor)),
            mtab_bits,
        );
        let mut zs = [0i32; 8];
        _mm256_storeu_si256(zs.as_mut_ptr() as *mut __m256i, z);
        let mut a2s = [0i32; 8];
        for k in 0..8 {
            let zz = zs[k];
            a2s[k] = if zz >= 255 {
                256
            } else if zz > 0 {
                table[zz as usize]
            } else if zz == 0 {
                1
            } else {
                // `SGRPROJ_SGR_BITS == 8`（与 `lr.rs` 的那个常量同源）
                ((zz << 8) + (zz / 2)) / (zz + 1)
            };
        }
        let a2v = _mm256_loadu_si256(a2s.as_ptr() as *const __m256i);
        // `b2 = (256 - a2) · D · one_over_n`。⚠️ 中间乘积会**在 i32 里回绕**
        // （256 · 6375 · 4096 ≈ 6.7e9 > 2³¹）—— 与标量的 `*` 同样按低 32 位算，
        // `mullo_epi32` 恰好同语义；**这不是 bug，别"顺手"换 i64**。
        let b2 = _mm256_mullo_epi32(
            _mm256_mullo_epi32(
                _mm256_sub_epi32(_mm256_set1_epi32(1 << 8), a2v),
                sb,
            ),
            _mm256_set1_epi32(one_over_n),
        );
        _mm256_storeu_si256(a_out.as_mut_ptr() as *mut __m256i, a2v);
        _mm256_storeu_si256(b_out.as_mut_ptr() as *mut __m256i, r2v(b2, recip_bits));
    }

    /// 同 [`mc_delta8`]。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn mc_shift8(row: &[u16], base: isize, post_round: u32, out: &mut [i32]) {
        let s16 = _mm_loadu_si128(row.as_ptr().offset(base) as *const __m128i);
        let s32 = _mm256_cvtepu16_epi32(s16);
        let v = match post_round {
            0 => s32,
            1 => _mm256_slli_epi32::<1>(s32),
            2 => _mm256_slli_epi32::<2>(s32),
            3 => _mm256_slli_epi32::<3>(s32),
            4 => _mm256_slli_epi32::<4>(s32),
            // 不在化简值域里：**不猜**，原样返回（调用方另有一般路径）
            _ => s32,
        };
        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, v);
    }

    /// **CDEF**（§7.15.3）一次滤完**两行**（16 个 **i16** 通道 = 2 行 × 8 列）。
    ///
    /// 一个像素要取 12 个候选（主抽头 2×2 + 次抽头 2 方向×2×2），每个候选都要
    /// `constrain(候选 - 当前)` 再乘抽头累加，还要同时更新 max/min。
    /// 标量版每个像素做 12 次这套运算；这里**两行 × 8 列一次做完**（低 128 位 = 行 `r`、
    /// 高 128 位 = 行 `r+1`）—— 固定一"对行"时，"候选相对当前像素的位移"对两行是
    /// **同一个常数**，所以每个候选只要两次 128 位载入 + 一次拼接。
    ///
    /// ## 为什么可以走 i16（原版是 i32、一行 8 个）
    ///
    /// 全部中间量的界（12 位最坏情形，8 位只会更小）：
    ///
    /// | 量 | 界 |
    /// |---|---|
    /// | 样本 `cur`/`cand` | ≤ 4095 |
    /// | `diff` / `mag` | ±4095 |
    /// | 阈值 `thr`（`pri/sec_strength << coeff_shift`）| ≤ 240 |
    /// | `constrain` 结果 `val` | ≤ 240 |
    /// | `acc = Σ_{12} tap·val`（`tap ≤ 4`）| ≤ 11520 |
    /// | `cur + ((8 + acc - adjust) >> 4)` | ⊆ [-719, 4815] |
    ///
    /// 全部远在 i16 内 ⇒ 通道数从 8 翻到 16，且**不再需要零扩展**（原版每个候选
    /// 一次 `cvtepu16_epi32`）。最后 `Clip3(mn, mx, ·)` 把结果收回 `[0, 2^bd)`。
    ///
    /// `offs[c]` 是第 c 个候选相对**块左上角像素**的线性偏移（`dy·stride + dx`），
    /// 约定 **`0..4` 是主抽头、`4..12` 是次抽头**（两组的阈值/阻尼不同）。
    /// 色度（`wu = 4`）时每行只用前 4 个通道，第 4..8 个读到的是右边的样本、被丢弃
    /// —— 所以只判"读得到"，不判"用得上"。
    ///
    /// # Safety
    /// - 必须已确认 AVX2 可用。
    /// - 对每个 `r ∈ [0, h)`、每个 `c ∈ [0, 12)`，
    ///   `data` 必须覆盖 `((y0+r)·stride + x0 + offs[c]) .. + 8 个 u16`。
    /// - `out.len() >= h · wu`，`wu ∈ {4, 8}`。
    /// - `adj` 必须 < 16（`_mm256_sra_epi16` 的语义要求；调用方给的
    ///   `damping_adj ≤ damping ≤ 11`，且 `mag < 2^16` ⇒ 16 位以上的移位两边同为 0）。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn cdef_pair16(
        data: &[u16],
        stride: isize,
        x0: isize,
        y0: isize,
        h: usize,
        wu: usize,
        offs: &[isize; super::CDEF_CANDS],
        taps: &[i32; super::CDEF_CANDS],
        pri_thr: i32,
        pri_adj: u32,
        sec_thr: i32,
        sec_adj: u32,
        out: &mut [i16],
    ) {
        debug_assert!(wu == 4 || wu == 8);
        let base = data.as_ptr();
        let zero = _mm256_setzero_si256();
        let one = _mm256_set1_epi16(1);
        let eight = _mm256_set1_epi16(8);
        let pri_t = _mm256_set1_epi16(pri_thr as i16);
        let sec_t = _mm256_set1_epi16(sec_thr as i16);
        let pri_s = _mm_cvtsi32_si128(pri_adj as i32);
        let sec_s = _mm_cvtsi32_si128(sec_adj as i32);
        let tap_v: [__m256i; super::CDEF_CANDS] =
            std::array::from_fn(|c| _mm256_set1_epi16(taps[c] as i16));
        let op = out.as_mut_ptr();
        let mut r = 0usize;
        while r < h {
            // `h` 为奇数时最后一行只算一次（高半读同一行）、只写一次。
            let r1 = if r + 1 < h { r + 1 } else { r };
            let off0 = (y0 + r as isize) * stride + x0;
            let off1 = (y0 + r1 as isize) * stride + x0;
            // SAFETY: 见函数文档的不变量（调用方保证整块 8 通道的取址都在缓冲内）
            let cur = _mm256_set_m128i(
                _mm_loadu_si128(base.offset(off1) as *const __m128i),
                _mm_loadu_si128(base.offset(off0) as *const __m128i),
            );
            let mut acc = zero;
            let mut mx = cur;
            let mut mn = cur;
            for c in 0..super::CDEF_CANDS {
                let (thr, sh) = if c < 4 { (pri_t, pri_s) } else { (sec_t, sec_s) };
                let cand = _mm256_set_m128i(
                    _mm_loadu_si128(base.offset(off1 + offs[c]) as *const __m128i),
                    _mm_loadu_si128(base.offset(off0 + offs[c]) as *const __m128i),
                );
                // `constrain( cand - cur, thr, adj )`（§7.15.3）：
                //   sign(diff) · Clip3(0, |diff|, thr - (|diff| >> adj))
                // `sign_epi16(t, diff)` 正好给"取符号、`diff == 0` 归零"，
                // 与标量里 `sign = if diff < 0 { -1 } else { 1 }` 逐位一致
                // （`diff == 0` 时 `mag == 0` ⇒ `t == 0`，谁乘都是 0）。
                let diff = _mm256_sub_epi16(cand, cur);
                let mag = _mm256_abs_epi16(diff);
                let t = _mm256_sub_epi16(thr, _mm256_sra_epi16(mag, sh));
                let t = _mm256_max_epi16(_mm256_min_epi16(t, mag), zero);
                let val = _mm256_sign_epi16(t, diff);
                acc = _mm256_add_epi16(acc, _mm256_mullo_epi16(val, tap_v[c]));
                mx = _mm256_max_epi16(mx, cand);
                mn = _mm256_min_epi16(mn, cand);
            }
            // `adjust = sum < 0 ? 1 : 0`；`v = cur + ((8 + sum - adjust) >> 4)`
            let adjust = _mm256_and_si256(_mm256_cmpgt_epi16(zero, acc), one);
            let v = _mm256_add_epi16(
                cur,
                _mm256_srai_epi16::<4>(_mm256_add_epi16(_mm256_sub_epi16(eight, adjust), acc)),
            );
            // `Clip3(min, max, v)`：min/max 都来自合法样本 ⇒ 结果自然在 [0, 2^bd) 内
            let v = _mm256_min_epi16(_mm256_max_epi16(v, mn), mx);
            let lo = _mm256_castsi256_si128(v);
            let hi = _mm256_extracti128_si256::<1>(v);
            let dst = op.add(r * wu);
            if wu == 8 {
                _mm_storeu_si128(dst as *mut __m128i, lo);
                if r + 1 < h {
                    _mm_storeu_si128(dst.add(8) as *mut __m128i, hi);
                }
            } else {
                _mm_storel_epi64(dst as *mut __m128i, lo);
                if r + 1 < h {
                    _mm_storel_epi64(dst.add(4) as *mut __m128i, hi);
                }
            }
            r += 2;
        }
    }

    /// 纵向 8 抽头，一次算 **16 个输出**（i16 中间量，`pmaddwd` 两两配对）。
    ///
    /// 每个输出 `j`：`Σ_t coef[t] · inter[(base_row + t)·stride + col + j]`。
    ///
    /// ## 为什么用 `madd` 而不是"展宽到 i32 再乘"
    ///
    /// 展宽版每个抽头要 `cvtepi16_epi32` ×2 + `mullo_epi32` ×2（≈6 条指令/抽头），
    /// `pmaddwd` 一次把**两个**抽头配成对（`c_t·x_t + c_{t+1}·x_{t+1}`）累进 i32，
    /// 于是每对抽头 2 条 `unpack` + 2 条 `madd` + 2 条 `add`。微基准
    /// （`bench_mc_kernels`）里纵向 **1.09 → 0.55 ns/像素**。
    ///
    /// 值域：两个乘积各自 ≤ `|c|·32767`，配对求和仍远在 i32 内（与展宽版的累加值域
    /// 一致 ⇒ 结果**逐位相同**；`ME_NO_SIMD` 对拍单测钉住这一点）。
    ///
    /// ⚠️ `col` 不能漏！单测里列偏移一直是 0，漏了也照样过 —— 但真素材上
    /// 第 8 列之后的每一批都会读错位置（`mc_vert8` 的第一版就这样错了，
    /// 靠"开/关 SIMD 输出逐字节一致"才抓出来）。
    ///
    /// # Safety
    /// - 必须已确认 AVX2 可用。
    /// - `inter` 必须覆盖 `[base_row·stride + col, (base_row+8)·stride + col + 16)`；
    ///   `out.len() >= 16`。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn mc_vert8(
        inter: &[i16],
        base_row: usize,
        col: usize,
        stride: usize,
        coef: &[i16; 8],
        round: u32,
        out: &mut [i32],
    ) {
        let base = inter.as_ptr().add(base_row * stride + col);
        // `unpacklo/unpackhi` 是**按 128 位道**交错的：lo 收下"每道前 8 个 i16"、
        // hi 收下"每道后 8 个"。所以累加完 `lo_acc` 装的是输出 `{0..4, 8..12}`、
        // `hi_acc` 装 `{4..8, 12..16}`，最后按道拼回去（见下面的 `permute2x128`）。
        let mut lo_acc = _mm256_setzero_si256();
        let mut hi_acc = _mm256_setzero_si256();
        let mut t = 0usize;
        while t < 8 {
            let a = _mm256_loadu_si256(base.add(t * stride) as *const __m256i);
            let b = _mm256_loadu_si256(base.add((t + 1) * stride) as *const __m256i);
            let pair =
                _mm256_set1_epi32(((coef[t + 1] as u16 as i32) << 16) | (coef[t] as u16 as i32));
            lo_acc =
                _mm256_add_epi32(lo_acc, _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), pair));
            hi_acc =
                _mm256_add_epi32(hi_acc, _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), pair));
            t += 2;
        }
        let lo = rnd(lo_acc, round);
        let hi = rnd(hi_acc, round);
        _mm256_storeu_si256(
            out.as_mut_ptr() as *mut __m256i,
            _mm256_permute2x128_si256::<0x20>(lo, hi),
        );
        _mm256_storeu_si256(
            out.as_mut_ptr().add(8) as *mut __m256i,
            _mm256_permute2x128_si256::<0x31>(lo, hi),
        );
    }

    // ── 去块（§7.14.6）：一条边的 4 条平行线一次滤完 ─────────────────────
    //
    // 这里用的是 **128 位（4 个 i32 通道）** 而不是别处的 256 位：一次
    // `loop_filter_edge` 只处理 `MI_SIZE = 4` 条线，通道数正好 4；把两条边
    // 拼进 256 位看着更满，但相邻边的强度/滤波尺寸未必一样、窗口在内存里
    // 也不连续 —— 反而要拆一半来做分支。
    //
    // 布局：`V[k]` 的第 i 个通道 = 第 i 条线上、垂直方向偏移 k 的样本
    // （`k ∈ [-8, 7]`，`k = 0` 就是边后第一个样本 `q0`）。
    //
    // * **横边**（`dx = 0, dy = 1`）：线 = 列，`V[k][i] = plane[yp+k][xp+i]`
    //   ⇒ 每个 k 就是一行里连续 4 个 u16，一次 `loadl` + 零扩展；
    // * **竖边**（`dx = 1, dy = 0`）：线 = 行，`V[k][i] = plane[yp+i][xp+k]`
    //   ⇒ 内存里跨 stride，**先做 4×8 / 4×4 转置**（本文件 `dbl_load4x8` /
    //   `dbl_load4x4`），写回时再转置回去（`dbl_store_v4`）。
    //
    // 掩码（hev/filter/flat/flat2）与窄/宽滤波器都照 `deblock.rs` 的标量式
    // 逐条照抄，只是换成通道级；「哪条线选哪个滤波器、`hev` 的线哪两列不动」
    // 全是逐通道掩码 + `blendv`。没写出的列不 store，写出的列要么是滤波结果、
    // 要么是原值 —— 与标量**逐位一致**（`deblock.rs` 的测试逐字节对拍）。

    /// `V[k]` 的下标换算（`k ∈ [-8, 7]`）。
    #[inline(always)]
    const fn vi(k: i32) -> usize {
        (k + 8) as usize
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dbl_adiff(a: __m128i, b: __m128i) -> __m128i {
        _mm_abs_epi32(_mm_sub_epi32(a, b))
    }

    /// `filter4_clamp` 的通道版（§7.14.6.3）：夹到 `[-(1<<(bd-1)), (1<<(bd-1))-1]`。
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dbl_clamp(v: __m128i, bit_depth: i32) -> __m128i {
        let lo = _mm_set1_epi32(-(1 << (bit_depth - 1)));
        let hi = _mm_set1_epi32((1 << (bit_depth - 1)) - 1);
        _mm_min_epi32(_mm_max_epi32(v, lo), hi)
    }

    /// 4 行 × 8 个 u16（列 `x0..x0+8`）→ 4 个「双列」向量，每向量 = 2 列 × 4 行。
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dbl_load4x8(base: *const u16, stride: isize, x0: isize) -> [__m128i; 4] {
        let r0 = _mm_loadu_si128(base.offset(x0) as *const __m128i);
        let r1 = _mm_loadu_si128(base.offset(x0 + stride) as *const __m128i);
        let r2 = _mm_loadu_si128(base.offset(x0 + 2 * stride) as *const __m128i);
        let r3 = _mm_loadu_si128(base.offset(x0 + 3 * stride) as *const __m128i);
        let a0 = _mm_unpacklo_epi16(r0, r1);
        let a1 = _mm_unpackhi_epi16(r0, r1);
        let a2 = _mm_unpacklo_epi16(r2, r3);
        let a3 = _mm_unpackhi_epi16(r2, r3);
        [
            _mm_unpacklo_epi32(a0, a2), // 列 (0, 1)
            _mm_unpackhi_epi32(a0, a2), // 列 (2, 3)
            _mm_unpacklo_epi32(a1, a3), // 列 (4, 5)
            _mm_unpackhi_epi32(a1, a3), // 列 (6, 7)
        ]
    }

    /// 4 行 × 4 个 u16（列 `x0..x0+4`）→ 2 个「双列」向量。
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dbl_load4x4(base: *const u16, stride: isize, x0: isize) -> [__m128i; 2] {
        let r0 = _mm_loadl_epi64(base.offset(x0) as *const __m128i);
        let r1 = _mm_loadl_epi64(base.offset(x0 + stride) as *const __m128i);
        let r2 = _mm_loadl_epi64(base.offset(x0 + 2 * stride) as *const __m128i);
        let r3 = _mm_loadl_epi64(base.offset(x0 + 3 * stride) as *const __m128i);
        let a0 = _mm_unpacklo_epi16(r0, r1);
        let a1 = _mm_unpacklo_epi16(r2, r3);
        [_mm_unpacklo_epi32(a0, a1), _mm_unpackhi_epi32(a0, a1)]
    }

    /// 把「双列」向量（2 列 × 4 行 u16）零扩展进 `v` 的两个槽位。
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dbl_put_pair(v: &mut [__m128i; 16], t: usize, b: __m128i) {
        v[t] = _mm_cvtepu16_epi32(b);
        v[t + 1] = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(b));
    }

    /// 窄滤波器（§7.14.6.3）的通道版：返回 `k = -2..2` 四列的结果。
    ///
    /// `hev` 的线只写 `q0`/`p0`（`q1`/`p1` 保持原值）—— 靠最后两处 `blendv` 实现。
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dbl_narrow(v: &[__m128i; 16], hev: __m128i, bit_depth: i32) -> [__m128i; 4] {
        let off = _mm_set1_epi32(0x80 << (bit_depth - 8));
        let (q0, q1, p0, p1) = (v[vi(0)], v[vi(1)], v[vi(-1)], v[vi(-2)]);
        let ps1 = _mm_sub_epi32(p1, off);
        let ps0 = _mm_sub_epi32(p0, off);
        let qs0 = _mm_sub_epi32(q0, off);
        let qs1 = _mm_sub_epi32(q1, off);
        // `filter = hev ? clamp(p1̂ - q1̂) : 0`，再 `clamp(filter + 3·(q0̂ - p0̂))`
        let f0 = _mm_and_si128(hev, dbl_clamp(_mm_sub_epi32(ps1, qs1), bit_depth));
        let d3 = {
            let d = _mm_sub_epi32(qs0, ps0);
            _mm_add_epi32(d, _mm_slli_epi32::<1>(d))
        };
        let filter = dbl_clamp(_mm_add_epi32(f0, d3), bit_depth);
        let filter1 = _mm_srai_epi32::<3>(dbl_clamp(
            _mm_add_epi32(filter, _mm_set1_epi32(4)),
            bit_depth,
        ));
        let filter2 = _mm_srai_epi32::<3>(dbl_clamp(
            _mm_add_epi32(filter, _mm_set1_epi32(3)),
            bit_depth,
        ));
        let oq0 = _mm_add_epi32(dbl_clamp(_mm_sub_epi32(qs0, filter1), bit_depth), off);
        let op0 = _mm_add_epi32(dbl_clamp(_mm_add_epi32(ps0, filter2), bit_depth), off);
        // `filter = Round2(filter1, 1)`，只对非 hev 的线写 q1/p1
        let fh = _mm_srai_epi32::<1>(_mm_add_epi32(filter1, _mm_set1_epi32(1)));
        let oq1 = _mm_add_epi32(dbl_clamp(_mm_sub_epi32(qs1, fh), bit_depth), off);
        let op1 = _mm_add_epi32(dbl_clamp(_mm_add_epi32(ps1, fh), bit_depth), off);
        // `hev` 的线保留原值（`blendv`：mask 置位取**第二个**操作数）
        [
            _mm_blendv_epi8(op1, p1, hev),
            op0,
            oq0,
            _mm_blendv_epi8(oq1, q1, hev),
        ]
    }

    /// 宽滤波器（§7.14.6.4）的通道版：`N` = `n`、`N2` = `n2`、`L2` = `log2Size`。
    ///
    /// 返回 `f[i]`（`i = -N..N` 按序），对应列 `xp + i`。累加顺序与标量一致
    /// （`j` 从 `-N` 递增），只是换成通道级。
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dbl_wide<const N: i32, const N2: i32, const L2: i32>(
        v: &[__m128i; 16],
    ) -> [__m128i; 12] {
        let mut f = [_mm_setzero_si128(); 12];
        let mut i = -N;
        while i < N {
            let mut t = _mm_setzero_si128();
            let mut j = -N;
            while j <= N {
                let p = (i + j).clamp(-(N + 1), N);
                let s = v[vi(p)];
                t = _mm_add_epi32(
                    t,
                    if j.abs() <= N2 {
                        _mm_slli_epi32::<1>(s)
                    } else {
                        s
                    },
                );
                j += 1;
            }
            f[(i + N) as usize] =
                _mm_srai_epi32::<L2>(_mm_add_epi32(t, _mm_set1_epi32(1 << (L2 - 1))));
            i += 1;
        }
        f
    }

    /// 写回（横边）：`f[i]`（列 `xp−n+i`）→ 行 `yp−n+i` 的连续 4 个 u16。
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dbl_store_h4(
        data: *mut u16,
        stride: isize,
        x: isize,
        y: isize,
        f: &[__m128i],
        n: usize,
    ) {
        let yn = y - n as isize;
        let mut i = 0usize;
        while i < 2 * n {
            let p = _mm_packus_epi32(f[i], f[i + 1]); // [行 i 的 4 个, 行 i+1 的 4 个]
            let row = data.offset((yn + i as isize) * stride + x);
            _mm_storel_epi64(row as *mut __m128i, p);
            _mm_storel_epi64(row.offset(stride) as *mut __m128i, _mm_srli_si128::<8>(p));
            i += 2;
        }
    }

    /// 写回（竖边）：`f[i]`（列 `xp−n+i`）→ 4 行；4×4 转置把「列通道」变回「行连续」。
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dbl_store_v4(
        data: *mut u16,
        stride: isize,
        x: isize,
        y: isize,
        f: &[__m128i],
        n: usize,
    ) {
        let cols = 2 * n;
        let base = x - n as isize;
        let mut j = 0usize;
        while j + 4 <= cols {
            // 4 列一组：两两打包后按行交错（每行 4 个 u16 = 8 字节）
            let b0 = _mm_packus_epi32(f[j], f[j + 1]);
            let b1 = _mm_packus_epi32(f[j + 2], f[j + 3]);
            let t0 = _mm_unpacklo_epi16(b0, _mm_srli_si128::<8>(b0));
            let t1 = _mm_unpacklo_epi16(b1, _mm_srli_si128::<8>(b1));
            let u0 = _mm_unpacklo_epi32(t0, t1); // 行 0、行 1
            let u1 = _mm_unpackhi_epi32(t0, t1); // 行 2、行 3
            let row = data.offset(y * stride + base + j as isize);
            _mm_storel_epi64(row as *mut __m128i, u0);
            _mm_storel_epi64(row.offset(stride) as *mut __m128i, _mm_srli_si128::<8>(u0));
            _mm_storel_epi64(row.offset(2 * stride) as *mut __m128i, u1);
            _mm_storel_epi64(
                row.offset(3 * stride) as *mut __m128i,
                _mm_srli_si128::<8>(u1),
            );
            j += 4;
        }
        if j < cols {
            // 余 2 列（只有 8 抽头亮度 n = 3 会走到）：每行 4 字节
            let b = _mm_packus_epi32(f[j], f[j + 1]);
            let t = _mm_unpacklo_epi16(b, _mm_srli_si128::<8>(b)); // 32 位单元 = 一行
            let row = data.offset(y * stride + base + j as isize);
            (row as *mut u32).write_unaligned(_mm_cvtsi128_si32(t) as u32);
            (row.offset(stride) as *mut u32).write_unaligned(_mm_extract_epi32::<1>(t) as u32);
            (row.offset(2 * stride) as *mut u32).write_unaligned(_mm_extract_epi32::<2>(t) as u32);
            (row.offset(3 * stride) as *mut u32).write_unaligned(_mm_extract_epi32::<3>(t) as u32);
        }
    }

    /// **去块的一条边**（§7.14.6）：4 条平行线一起滤。
    ///
    /// `need` = `deblock::window_need(...)`（需要的半径上界：16 抽头 7、8 抽头亮度 4、
    /// 色度 3、4 抽头 2）—— 由调用方算好，避免两处各写一份。
    ///
    /// # Safety
    /// - 必须已确认 AVX2 可用。
    /// - 下列读写必须全部落在**帧内且不越出所在行**（调用方按 `deblock.rs` 的
    ///   内点判据保证；`data` 覆盖到 `(yp+3)·stride + xp + need`）：
    ///   * 竖边（`dx = 1`）：`xp ≥ load_pad`（`need ≥ 7` 时 8，`need ∈ {3,4}` 时 4，
    ///     否则 2）、`xp + need - 1 < w`、`yp + 3 < h`；
    ///   * 横边（`dy = 1`）：`xp ≥ 0`、`xp + 3 < w`、`yp ≥ need`、`yp + need - 1 < h`。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn deblock_edge4(
        data: *mut u16,
        stride: isize,
        bit_depth: u8,
        xp: i32,
        yp: i32,
        dx: i32,
        dy: i32,
        plane: u8,
        filter_size: i32,
        need: i32,
        limit: i32,
        blimit: i32,
        thresh: i32,
    ) {
        let (x, y) = (xp as isize, yp as isize);
        let bd_shift = i32::from(bit_depth) - 8;
        let bit_depth = i32::from(bit_depth);

        // ── 取窗口 ────────────────────────────────────────────────────
        let mut v = [_mm_setzero_si128(); 16];
        if dy == 1 {
            for k in -need..need {
                let row = data.offset((y + k as isize) * stride + x);
                v[vi(k)] = _mm_cvtepu16_epi32(_mm_loadl_epi64(row as *const __m128i));
            }
        } else {
            // 竖边：4 行 × 窗口 → 转置（`rows` 是这 4 行的首地址）
            let rows = data.offset(y * stride) as *const u16;
            if need >= 7 {
                // 列 [-8, 8)：两段 4×8 覆盖 16 列（只用 [-7, 6]）
                let a = dbl_load4x8(rows, stride, x - 8);
                let b = dbl_load4x8(rows, stride, x);
                let mut t = 0usize;
                for p in a {
                    dbl_put_pair(&mut v, t, p);
                    t += 2;
                }
                for p in b {
                    dbl_put_pair(&mut v, t, p);
                    t += 2;
                }
            } else if need == 2 {
                // 列 [-2, 2)
                let b = dbl_load4x4(rows, stride, x - 2);
                dbl_put_pair(&mut v, 6, b[0]);
                dbl_put_pair(&mut v, 8, b[1]);
            } else {
                // 列 [-4, 4)（色度 8 抽头也用这条：只用 [-3, 2]）
                let b = dbl_load4x8(rows, stride, x - 4);
                let mut t = 4usize;
                for p in b {
                    dbl_put_pair(&mut v, t, p);
                    t += 2;
                }
            }
        }

        // ── 掩码（§7.14.6.2），全部通道级 ─────────────────────────────
        let zero = _mm_setzero_si128();
        let (p1, p0, q0, q1) = (v[vi(-2)], v[vi(-1)], v[vi(0)], v[vi(1)]);
        let thresh_v = _mm_set1_epi32(thresh << bd_shift);
        let hev = _mm_or_si128(
            _mm_cmpgt_epi32(dbl_adiff(p1, p0), thresh_v),
            _mm_cmpgt_epi32(dbl_adiff(q1, q0), thresh_v),
        );
        let limit_v = _mm_set1_epi32(limit << bd_shift);
        let blimit_v = _mm_set1_epi32(blimit << bd_shift);
        let mut m = _mm_or_si128(
            _mm_cmpgt_epi32(dbl_adiff(p1, p0), limit_v),
            _mm_cmpgt_epi32(dbl_adiff(q1, q0), limit_v),
        );
        let t = _mm_add_epi32(
            _mm_slli_epi32::<1>(dbl_adiff(p0, q0)),
            _mm_srai_epi32::<1>(dbl_adiff(p1, q1)),
        );
        m = _mm_or_si128(m, _mm_cmpgt_epi32(t, blimit_v));
        if filter_size != 4 {
            let (p2, q2) = (v[vi(-3)], v[vi(2)]);
            m = _mm_or_si128(m, _mm_cmpgt_epi32(dbl_adiff(p2, p1), limit_v));
            m = _mm_or_si128(m, _mm_cmpgt_epi32(dbl_adiff(q2, q1), limit_v));
            if plane == 0 {
                let (p3, q3) = (v[vi(-4)], v[vi(3)]);
                m = _mm_or_si128(m, _mm_cmpgt_epi32(dbl_adiff(p3, p2), limit_v));
                m = _mm_or_si128(m, _mm_cmpgt_epi32(dbl_adiff(q3, q2), limit_v));
            }
        }
        let pass_mask = _mm_cmpeq_epi32(m, zero); // 通道全 1 = 这条线要滤
        if _mm_movemask_epi8(pass_mask) == 0 {
            return;
        }
        // flatMask / flatMask2（§7.14.6.2）
        let mut flat = zero;
        let mut flat2 = zero;
        if filter_size >= 8 {
            let th = _mm_set1_epi32(1 << bd_shift);
            let (p2, q2) = (v[vi(-3)], v[vi(2)]);
            flat = _mm_or_si128(
                _mm_cmpgt_epi32(dbl_adiff(p1, p0), th),
                _mm_cmpgt_epi32(dbl_adiff(q1, q0), th),
            );
            flat = _mm_or_si128(flat, _mm_cmpgt_epi32(dbl_adiff(p2, p0), th));
            flat = _mm_or_si128(flat, _mm_cmpgt_epi32(dbl_adiff(q2, q0), th));
            if plane == 0 {
                let (p3, q3) = (v[vi(-4)], v[vi(3)]);
                flat = _mm_or_si128(flat, _mm_cmpgt_epi32(dbl_adiff(p3, p0), th));
                flat = _mm_or_si128(flat, _mm_cmpgt_epi32(dbl_adiff(q3, q0), th));
            }
        }
        if filter_size >= 16 {
            let th = _mm_set1_epi32(1 << bd_shift);
            for k in [-7i32, -6, -5] {
                flat2 = _mm_or_si128(flat2, _mm_cmpgt_epi32(dbl_adiff(v[vi(k)], p0), th));
            }
            for k in [4i32, 5, 6] {
                flat2 = _mm_or_si128(flat2, _mm_cmpgt_epi32(dbl_adiff(v[vi(k)], q0), th));
            }
        }
        let flat_mask = _mm_cmpeq_epi32(flat, zero);
        let flat2_mask = _mm_cmpeq_epi32(flat2, zero);

        // 每条线选哪个滤波器（标量同款：fs==4 或不平 ⇒ 窄；fs>=16 且两个 flat 才 16 抽头）
        let wide_lane = if filter_size >= 8 { flat_mask } else { zero };
        let wide16_lane = if filter_size >= 16 {
            _mm_and_si128(flat_mask, flat2_mask)
        } else {
            zero
        };
        let wide8_lane = _mm_andnot_si128(wide16_lane, wide_lane);
        let narrow_lane = _mm_andnot_si128(wide_lane, pass_mask);
        let w8_lane = _mm_and_si128(wide8_lane, pass_mask);
        let w16_lane = _mm_and_si128(wide16_lane, pass_mask);

        // 写出跨度（列数 = 2n）：fs==4 只有窄滤波器（4 列）；fs>=8 取该档宽滤波器的
        // 跨度（色度 n=2、亮度 8 抽头 n=3、16 抽头 n=6）—— 窄线只写中间 4 列，
        // 外面几列经 blend 保持原值。
        let n = if filter_size == 4 {
            2usize
        } else if filter_size >= 16 {
            6
        } else if plane == 0 {
            3
        } else {
            2
        };

        let need_narrow = _mm_movemask_epi8(narrow_lane) != 0;
        let need_w8 = _mm_movemask_epi8(w8_lane) != 0;
        let need_w16 = _mm_movemask_epi8(w16_lane) != 0;
        let nv = if need_narrow {
            dbl_narrow(&v, hev, bit_depth)
        } else {
            [zero; 4]
        };
        // 宽滤波器：`N` = `n`（亮度 8 抽头 3 / 色度 2 / 16 抽头 6），与标量同款
        let nw8 = if plane == 0 { 3i32 } else { 2 };
        let w8 = if need_w8 {
            if plane == 0 {
                dbl_wide::<3, 0, 3>(&v)
            } else {
                dbl_wide::<2, 1, 3>(&v)
            }
        } else {
            [zero; 12]
        };
        let w16 = if need_w16 {
            dbl_wide::<6, 1, 4>(&v)
        } else {
            [zero; 12]
        };

        // ── 逐列融合：原值打底，再按"这类线非空"叠窄/宽 8/宽 16 ────────────
        //
        // 三类线的集合互斥（一条线只选一个滤波器）；掩码为空的融合**整段跳过**
        // ——它本来就恒等，省下的是逐列的 `blendv`。
        let mut out = [zero; 12];
        for (i, k) in (-(n as i32)..n as i32).enumerate() {
            out[i] = v[vi(k)];
        }
        if need_narrow {
            for k in -2i32..2 {
                let i = (k + n as i32) as usize;
                out[i] = _mm_blendv_epi8(out[i], nv[(k + 2) as usize], narrow_lane);
            }
        }
        if need_w8 {
            for k in -nw8..nw8 {
                let i = (k + n as i32) as usize;
                out[i] = _mm_blendv_epi8(out[i], w8[(k + nw8) as usize], w8_lane);
            }
        }
        if need_w16 {
            for k in -6i32..6 {
                let i = (k + n as i32) as usize;
                out[i] = _mm_blendv_epi8(out[i], w16[(k + 6) as usize], w16_lane);
            }
        }

        if dx == 1 {
            dbl_store_v4(data, stride, x, y, &out, n);
        } else {
            dbl_store_h4(data, stride, x, y, &out, n);
        }
    }

    // ── 反变换（§7.13.2）：4 条线一起跑的 1D 内核 ─────────────────────────
    //
    // 布局：`V[i]` 的 4 个 **i64 通道 = 4 条平行线**（相邻 4 行或相邻 4 列）的第 i 个
    // 元素。每条线走的是与 `transform.rs` 标量版**同一套**索引表 / 取整 / 钳位，
    // 只是换成通道级 —— 所以两条路的输出**逐位相同**（有全组合对拍单测钉住）。
    //
    // **值域分析（2026-10-02 插桩实测）**：乘法的输入 |t| ≤ 2^19.5（12bit 极端系数
    // 全组合扫一遍；8bit 真素材 ≤ 2^13.5）⇒ `_mm256_mul_epi32`（要求 |t| < 2^31）
    // **精确**；但积与和要到 ~2^32 ⇒ 通道必须是 **i64**（不能退回 i32 通道）。
    //
    // 两个驱动各自负责"4 条线"的收集/写回形态：
    // * [`inv_txfm_cols`]：线 = 相邻 4 **列** ⇒ 同一行的 4 个 i64 在内存里连续，
    //   直接 `loadu`/`storeu`（这是"列间天然独立"最省事的一半）；
    // * [`inv_txfm_rows`]：线 = 相邻 4 **行** ⇒ 跨 stride，靠 4×4 i32 转置收集、
    //   4×4 i64 转置写回（见 `vt_transpose4`）。

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_add(a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi64(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_sub(a: __m256i, b: __m256i) -> __m256i {
        _mm256_sub_epi64(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_neg(a: __m256i) -> __m256i {
        _mm256_sub_epi64(_mm256_setzero_si256(), a)
    }

    /// `v * c`（`c` 广播成 i32 系数）。要求 |v| < 2^31 —— 见上面的值域分析。
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_mul(v: __m256i, c: i32) -> __m256i {
        _mm256_mul_epi32(v, _mm256_set1_epi64x(c as i64))
    }

    /// 4 个 i64 通道的**算术右移**（`n ∈ 1..=63`）：AVX2 没有 `srai_epi64`，手工拼 ——
    /// 每个 64 位通道的符号位在高 32 位半字里，先复制到整个通道，再做逻辑右移 + 或。
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_srai(v: __m256i, n: u32) -> __m256i {
        debug_assert!((1..=63).contains(&n));
        let sign = _mm256_shuffle_epi32(_mm256_srai_epi32(v, 31), 0b11_11_01_01);
        let hi = _mm256_sllv_epi64(sign, _mm256_set1_epi64x((64 - n) as i64));
        _mm256_or_si256(_mm256_srlv_epi64(v, _mm256_set1_epi64x(n as i64)), hi)
    }

    /// `Round2( v, n )`（§3）；`n = 0` 时等于 `v`。
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_round2(v: __m256i, n: u32) -> __m256i {
        if n == 0 {
            return v;
        }
        vt_srai(vt_add(v, _mm256_set1_epi64x(1i64 << (n - 1))), n)
    }

    /// `Clip3( lo, hi, v )`（64 位有符号比较；`cmpgt_epi64` + `blendv`）。
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_clamp(v: __m256i, lo: i64, hi: i64) -> __m256i {
        let vl = _mm256_set1_epi64x(lo);
        let vh = _mm256_set1_epi64x(hi);
        let a = _mm256_blendv_epi8(v, vl, _mm256_cmpgt_epi64(vl, v));
        _mm256_blendv_epi8(a, vh, _mm256_cmpgt_epi64(v, vh))
    }

    /// `B( a, b, angle, flip, r )`（§7.13.1.2）的通道版。
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_butterfly(v: &mut [__m256i], a: usize, b: usize, angle: i32, flip: bool) {
        let (c, s) = (
            super::super::transform::cos128(angle),
            super::super::transform::sin128(angle),
        );
        let x = _mm256_sub_epi64(vt_mul(v[a], c), vt_mul(v[b], s));
        let y = _mm256_add_epi64(vt_mul(v[a], s), vt_mul(v[b], c));
        v[a] = vt_round2(x, 12);
        v[b] = vt_round2(y, 12);
        if flip {
            v.swap(a, b);
        }
    }

    /// `H( a, b, flip, r )`（§7.13.1.2）的通道版。
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_hadamard(v: &mut [__m256i], a: usize, b: usize, flip: bool, r: u32) {
        let (a, b) = if flip { (b, a) } else { (a, b) };
        let (x, y) = (v[a], v[b]);
        let lo = -(1i64 << (r - 1));
        let hi = (1i64 << (r - 1)) - 1;
        v[a] = vt_clamp(vt_add(x, y), lo, hi);
        v[b] = vt_clamp(vt_sub(x, y), lo, hi);
    }

    /// `Inverse DCT array permutation process`（§7.13.2.2）的通道版。
    #[target_feature(enable = "avx2")]
    unsafe fn vt_dct_permute(v: &mut [__m256i], n: u32) {
        let n0 = 1usize << n;
        // 暂存**不清零**：只写 `[..n0]`、只读 `[..n0]`（`n0` 最小只有 4，
        // 给整个 2KB 数组清零在小块上是纯浪费）。
        let mut copy: [MaybeUninit<__m256i>; 64] = [const { MaybeUninit::uninit() }; 64];
        for (i, slot) in copy[..n0].iter_mut().enumerate() {
            slot.write(v[i]);
        }
        for (i, slot) in v[..n0].iter_mut().enumerate() {
            *slot = copy[super::super::transform::brev(n, i)].assume_init();
        }
    }

    /// `Inverse ADST4 process`（§7.13.2.6）的通道版。
    #[target_feature(enable = "avx2")]
    unsafe fn vt_inverse_adst4(v: &mut [__m256i], r: u32) {
        const SINPI_1_9: i32 = 1321;
        const SINPI_2_9: i32 = 2482;
        const SINPI_3_9: i32 = 3344;
        const SINPI_4_9: i32 = 3803;
        let mut s = [_mm256_setzero_si256(); 7];
        s[0] = vt_mul(v[0], SINPI_1_9);
        s[1] = vt_mul(v[0], SINPI_2_9);
        s[2] = vt_mul(v[1], SINPI_3_9);
        s[3] = vt_mul(v[2], SINPI_4_9);
        s[4] = vt_mul(v[2], SINPI_1_9);
        s[5] = vt_mul(v[3], SINPI_2_9);
        s[6] = vt_mul(v[3], SINPI_4_9);
        let a7 = vt_sub(v[0], v[2]);
        let b7 = vt_add(a7, v[3]);
        s[0] = vt_add(s[0], s[3]);
        s[1] = vt_sub(s[1], s[4]);
        s[3] = s[2];
        s[2] = vt_mul(b7, SINPI_3_9);
        s[0] = vt_add(s[0], s[5]);
        s[1] = vt_sub(s[1], s[6]);
        let x = [
            vt_add(s[0], s[3]),
            vt_add(s[1], s[3]),
            s[2],
            vt_sub(vt_add(s[0], s[1]), s[3]),
        ];
        for (k, slot) in v[..4].iter_mut().enumerate() {
            *slot = vt_round2(x[k], 12);
        }
        let _ = r;
    }


// ===== 自动转录（从 transform.rs 的标量版逐条换名）=====
#[target_feature(enable = "avx2")]
unsafe fn vt_inverse_dct(v: &mut [__m256i], n: u32, r: u32) {
    let n0 = 1usize << n;
    vt_dct_permute(v, n);

    if n == 6 {
        for i in 0..16 {
            vt_butterfly(v, 32 + i, 63 - i, 63 - 4 * brev(4, i) as i32, false);
        }
    }
    if n >= 5 {
        for i in 0..8 {
            vt_butterfly(v, 16 + i, 31 - i, 6 + ((brev(3, 7 - i) << 3) as i32), false);
        }
    }
    if n == 6 {
        for i in 0..16 {
            vt_hadamard(v, 32 + i * 2, 33 + i * 2, i & 1 == 1, r);
        }
    }
    if n >= 4 {
        for i in 0..4 {
            vt_butterfly(v, 8 + i, 15 - i, 12 + ((brev(2, 3 - i) << 4) as i32), false);
        }
    }
    if n >= 5 {
        for i in 0..8 {
            vt_hadamard(v, 16 + 2 * i, 17 + 2 * i, i & 1 == 1, r);
        }
    }
    if n == 6 {
        for i in 0..4 {
            for j in 0..2 {
                vt_butterfly(
                    v,
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
            vt_butterfly(v, 4 + i, 7 - i, 56 - 32 * i as i32, false);
        }
    }
    if n >= 4 {
        for i in 0..4 {
            vt_hadamard(v, 8 + 2 * i, 9 + 2 * i, i & 1 == 1, r);
        }
    }
    if n >= 5 {
        for i in 0..2 {
            for j in 0..2 {
                vt_butterfly(
                    v,
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
                vt_hadamard(v, 32 + i * 4 + j, 35 + i * 4 - j, i & 1 == 1, r);
            }
        }
    }
    for i in 0..2 {
        vt_butterfly(v, 2 * i, 2 * i + 1, 32 + 16 * i as i32, i == 0);
    }
    if n >= 3 {
        for i in 0..2 {
            vt_hadamard(v, 4 + 2 * i, 5 + 2 * i, i == 1, r);
        }
    }
    if n >= 4 {
        for i in 0..2 {
            vt_butterfly(v, 14 - i, 9 + i, 48 + 64 * i as i32, true);
        }
    }
    if n >= 5 {
        for i in 0..4 {
            for j in 0..2 {
                vt_hadamard(v, 16 + 4 * i + j, 19 + 4 * i - j, i & 1 == 1, r);
            }
        }
    }
    if n == 6 {
        for i in 0..2 {
            for j in 0..4 {
                vt_butterfly(
                    v,
                    61 - i * 8 - j,
                    34 + i * 8 + j,
                    56 - i as i32 * 32 + (j >> 1) as i32 * 64,
                    true,
                );
            }
        }
    }
    for i in 0..2 {
        vt_hadamard(v, i, 3 - i, false, r);
    }
    if n >= 3 {
        vt_butterfly(v, 6, 5, 32, true);
    }
    if n >= 4 {
        for i in 0..2 {
            for j in 0..2 {
                vt_hadamard(v, 8 + 4 * i + j, 11 + 4 * i - j, i == 1, r);
            }
        }
    }
    if n >= 5 {
        for i in 0..4 {
            vt_butterfly(v, 29 - i, 18 + i, 48 + ((i >> 1) as i32) * 64, true);
        }
    }
    if n == 6 {
        for i in 0..4 {
            for j in 0..4 {
                vt_hadamard(v, 32 + 8 * i + j, 39 + 8 * i - j, i & 1 == 1, r);
            }
        }
    }
    if n >= 3 {
        for i in 0..4 {
            vt_hadamard(v, i, 7 - i, false, r);
        }
    }
    if n >= 4 {
        for i in 0..2 {
            vt_butterfly(v, 13 - i, 10 + i, 32, true);
        }
    }
    if n >= 5 {
        for i in 0..2 {
            for j in 0..4 {
                vt_hadamard(v, 16 + i * 8 + j, 23 + i * 8 - j, i == 1, r);
            }
        }
    }
    if n == 6 {
        for i in 0..8 {
            vt_butterfly(v, 59 - i, 36 + i, if i < 4 { 48 } else { 112 }, true);
        }
    }
    if n >= 4 {
        for i in 0..8 {
            vt_hadamard(v, i, 15 - i, false, r);
        }
    }
    if n >= 5 {
        for i in 0..4 {
            vt_butterfly(v, 27 - i, 20 + i, 32, true);
        }
    }
    if n == 6 {
        for i in 0..8 {
            vt_hadamard(v, 32 + i, 47 - i, false, r);
            vt_hadamard(v, 48 + i, 63 - i, true, r);
        }
    }
    if n >= 5 {
        for i in 0..16 {
            vt_hadamard(v, i, 31 - i, false, r);
        }
    }
    if n == 6 {
        for i in 0..8 {
            vt_butterfly(v, 55 - i, 40 + i, 32, true);
        }
    }
    if n == 6 {
        for i in 0..32 {
            vt_hadamard(v, i, 63 - i, false, r);
        }
    }
    debug_assert_eq!(n0, 1usize << n);
}

#[target_feature(enable = "avx2")]
unsafe fn vt_inverse_adst8(v: &mut [__m256i], r: u32) {
    vt_adst_input_permute(v, 3);
    for i in 0..4 {
        vt_butterfly(v, 2 * i, 2 * i + 1, 60 - 16 * i as i32, true);
    }
    for i in 0..4 {
        vt_hadamard(v, i, 4 + i, false, r);
    }
    for i in 0..2 {
        vt_butterfly(v, 4 + 3 * i, 5 + i, 48 - 32 * i as i32, true);
    }
    for j in 0..2 {
        for i in 0..2 {
            vt_hadamard(v, 4 * j + i, 2 + 4 * j + i, false, r);
        }
    }
    for i in 0..2 {
        vt_butterfly(v, 2 + 4 * i, 3 + 4 * i, 32, true);
    }
    vt_adst_output_permute(v, 3);
}

#[target_feature(enable = "avx2")]
unsafe fn vt_inverse_adst16(v: &mut [__m256i], r: u32) {
    vt_adst_input_permute(v, 4);
    for i in 0..8 {
        vt_butterfly(v, 2 * i, 2 * i + 1, 62 - 8 * i as i32, true);
    }
    for i in 0..8 {
        vt_hadamard(v, i, 8 + i, false, r);
    }
    for i in 0..2 {
        vt_butterfly(v, 8 + 2 * i, 9 + 2 * i, 56 - 32 * i as i32, true);
        vt_butterfly(v, 13 + 2 * i, 12 + 2 * i, 8 + 32 * i as i32, true);
    }
    for j in 0..2 {
        for i in 0..4 {
            vt_hadamard(v, 8 * j + i, 4 + 8 * j + i, false, r);
        }
    }
    for j in 0..2 {
        for i in 0..2 {
            vt_butterfly(v, 4 + 8 * j + 3 * i, 5 + 8 * j + i, 48 - 32 * i as i32, true);
        }
    }
    for j in 0..4 {
        for i in 0..2 {
            vt_hadamard(v, 4 * j + i, 2 + 4 * j + i, false, r);
        }
    }
    for i in 0..4 {
        vt_butterfly(v, 2 + 4 * i, 3 + 4 * i, 32, true);
    }
    vt_adst_output_permute(v, 4);
}

#[target_feature(enable = "avx2")]
unsafe fn vt_adst_input_permute(v: &mut [__m256i], n: u32) {
    let n0 = 1usize << n;
    let mut copy: [MaybeUninit<__m256i>; 64] = [const { MaybeUninit::uninit() }; 64];
    for (i, slot) in copy[..n0].iter_mut().enumerate() {
        slot.write(v[i]);
    }
    for (i, slot) in v[..n0].iter_mut().enumerate() {
        let idx = if i & 1 == 1 { i - 1 } else { n0 - i - 1 };
        *slot = copy[idx].assume_init();
    }
}

#[target_feature(enable = "avx2")]
unsafe fn vt_adst_output_permute(v: &mut [__m256i], n: u32) {
    let n0 = 1usize << n;
    let mut copy: [MaybeUninit<__m256i>; 64] = [const { MaybeUninit::uninit() }; 64];
    for (i, slot) in copy[..n0].iter_mut().enumerate() {
        slot.write(v[i]);
    }
    for (i, slot) in v[..n0].iter_mut().enumerate() {
        let a = (i >> 3) & 1;
        let b = ((i >> 2) & 1) ^ ((i >> 3) & 1);
        let c = ((i >> 1) & 1) ^ ((i >> 2) & 1);
        let d = (i & 1) ^ ((i >> 1) & 1);
        let idx = ((d << 3) | (c << 2) | (b << 1) | a) >> (4 - n);
        *slot = if i & 1 == 1 { vt_neg(copy[idx].assume_init()) } else { copy[idx].assume_init() };
    }
}


    /// `Inverse ADST process`（§7.13.2.9）的通道版。
    #[target_feature(enable = "avx2")]
    unsafe fn vt_inverse_adst(v: &mut [__m256i], n: u32, r: u32) {
        match n {
            2 => vt_inverse_adst4(v, r),
            3 => vt_inverse_adst8(v, r),
            4 => vt_inverse_adst16(v, r),
            _ => unreachable!("ADST 只定义到 2^n = 16；调用前已按 tx_sz 校验"),
        }
    }

    /// `Inverse Walsh-Hadamard transform process`（§7.13.2.10）的通道版。
    /// `shift ∈ {0, 2}`（行端 2、列端 0），都按**算术**右移。
    #[target_feature(enable = "avx2")]
    unsafe fn vt_inverse_wht(v: &mut [__m256i], shift: u32) {
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn shr(v: __m256i, shift: u32) -> __m256i {
            if shift == 0 {
                v
            } else {
                vt_srai(v, shift)
            }
        }
        let mut a = shr(v[0], shift);
        let c = shr(v[1], shift);
        let mut d = shr(v[2], shift);
        let mut b = shr(v[3], shift);
        a = vt_add(a, c);
        d = vt_sub(d, b);
        let e = shr(vt_sub(a, d), 1);
        b = vt_sub(e, b);
        let c = vt_sub(e, c);
        a = vt_sub(a, b);
        d = vt_add(d, c);
        v[0] = a;
        v[1] = b;
        v[2] = c;
        v[3] = d;
    }

    /// `Inverse identity transform process`（§7.13.2.11–14）的通道版。
    #[target_feature(enable = "avx2")]
    unsafe fn vt_inverse_identity(v: &mut [__m256i], n: u32) {
        match n {
            2 => {
                for slot in v[..4].iter_mut() {
                    *slot = vt_round2(vt_mul(*slot, 5793), 12);
                }
            }
            3 => {
                for slot in v[..8].iter_mut() {
                    *slot = vt_add(*slot, *slot);
                }
            }
            4 => {
                for slot in v[..16].iter_mut() {
                    *slot = vt_round2(vt_mul(*slot, 11586), 12);
                }
            }
            5 => {
                for slot in v[..32].iter_mut() {
                    *slot = _mm256_slli_epi64::<2>(*slot);
                }
            }
            _ => unreachable!("IDTX 只定义到 2^n = 32；调用前已按 tx_sz 校验"),
        }
    }

    /// 1D 变换的通道版入口（与标量的 `forward_1d` 同一张分派表）。
    #[target_feature(enable = "avx2")]
    unsafe fn vt_forward_1d(kind: u32, v: &mut [__m256i], n: u32, r: u32) {
        match kind {
            TXF_DCT => vt_inverse_dct(v, n, r),
            TXF_ADST => vt_inverse_adst(v, n, r),
            _ => vt_inverse_identity(v, n),
        }
    }

    /// 4×4 i64 转置（AVX2：`unpack` + `permute2x128`，8 条指令）。
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_transpose4(v: [__m256i; 4]) -> [__m256i; 4] {
        let t0 = _mm256_unpacklo_epi64(v[0], v[1]);
        let t1 = _mm256_unpackhi_epi64(v[0], v[1]);
        let t2 = _mm256_unpacklo_epi64(v[2], v[3]);
        let t3 = _mm256_unpackhi_epi64(v[2], v[3]);
        [
            _mm256_permute2x128_si256(t0, t2, 0x20),
            _mm256_permute2x128_si256(t1, t3, 0x20),
            _mm256_permute2x128_si256(t0, t2, 0x31),
            _mm256_permute2x128_si256(t1, t3, 0x31),
        ]
    }

    /// `Round2( x, n )` 的**全档位**版（`rnd` 只覆盖插值的几档）。
    ///
    /// 复合混合的位移量是 `6 + InterPostRound` ∈ {6,8,10}、平均是 `1 + InterPostRound`
    /// ∈ {1,3,5}、距离加权是 `4 + InterPostRound` ∈ {4,6,8}，都不在 `rnd` 的档位里，
    /// 所以这里把 0..=12 补全（仍然是立即数，编译期展开成一条 `srai`）。
    #[inline(always)]
    unsafe fn rnd_any(acc: __m256i, n: u32) -> __m256i {
        let half = |k: u32| _mm256_set1_epi32(1i32 << (k - 1));
        match n {
            0 => acc,
            1 => _mm256_srai_epi32::<1>(_mm256_add_epi32(acc, half(1))),
            2 => _mm256_srai_epi32::<2>(_mm256_add_epi32(acc, half(2))),
            3 => _mm256_srai_epi32::<3>(_mm256_add_epi32(acc, half(3))),
            4 => _mm256_srai_epi32::<4>(_mm256_add_epi32(acc, half(4))),
            5 => _mm256_srai_epi32::<5>(_mm256_add_epi32(acc, half(5))),
            6 => _mm256_srai_epi32::<6>(_mm256_add_epi32(acc, half(6))),
            7 => _mm256_srai_epi32::<7>(_mm256_add_epi32(acc, half(7))),
            8 => _mm256_srai_epi32::<8>(_mm256_add_epi32(acc, half(8))),
            9 => _mm256_srai_epi32::<9>(_mm256_add_epi32(acc, half(9))),
            10 => _mm256_srai_epi32::<10>(_mm256_add_epi32(acc, half(10))),
            11 => _mm256_srai_epi32::<11>(_mm256_add_epi32(acc, half(11))),
            // 不在值域里：**不猜**（原值返回，调用方另有标量路径）
            _ => acc,
        }
    }

    /// 把 8 个 i32 收窄成 8 个 u16 并写进 `dst`（`Clip1` 的上界由 `max` 给）。
    #[inline(always)]
    unsafe fn pack_store8(dst: *mut u16, v: __m256i, max: i32) {
        let c = _mm256_min_epi32(_mm256_max_epi32(v, _mm256_setzero_si256()), _mm256_set1_epi32(max));
        let packed = _mm256_packus_epi32(c, c);
        let ordered = _mm256_permute4x64_epi64::<0b0000_1000>(packed);
        _mm_storeu_si128(dst as *mut __m128i, _mm256_castsi256_si128(ordered));
    }

    /// **掩码混合**一行（§7.11.3.14）：`Clip1( Round2( m·p0 + (64-m)·p1, shift ) )`，
    /// 一次 8 个像素。`m` 逐像素给（楔形是现算、DIFFWTD 是查表、OBMC 是掩码表）。
    ///
    /// # Safety
    /// 必须已确认 AVX2 可用；`p0`/`p1`/`m` 至少 `n` 项、`dst` 处可写 `n` 个 u16。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn blend_masked8(
        dst: *mut u16,
        p0: *const i32,
        p1: *const i32,
        m: *const i32,
        n: usize,
        shift: u32,
        max: i32,
    ) {
        let c64 = _mm256_set1_epi32(64);
        let mut j = 0usize;
        while j + super::LANES <= n {
            let v0 = _mm256_loadu_si256(p0.add(j) as *const __m256i);
            let v1 = _mm256_loadu_si256(p1.add(j) as *const __m256i);
            let vm = _mm256_loadu_si256(m.add(j) as *const __m256i);
            let s = _mm256_add_epi32(
                _mm256_mullo_epi32(vm, v0),
                _mm256_mullo_epi32(_mm256_sub_epi32(c64, vm), v1),
            );
            pack_store8(dst.add(j), rnd_any(s, shift), max);
            j += super::LANES;
        }
        while j < n {
            let vm = *m.add(j);
            let v = rnd_scalar(vm * *p0.add(j) + (64 - vm) * *p1.add(j), shift).clamp(0, max);
            *dst.add(j) = v as u16;
            j += 1;
        }
    }

    /// **掩码混合**一行，`p0` 取自 u16 平面（就地读原像素）、`p1` 是 i32 预测值：
    /// `Clip1( Round2( m·p0 + (64-m)·Round2(p1, p1_shift) 钳到 p1_max, shift ) )`。
    ///
    /// 两条用它的路径形状一样、只是 `p1` 的预处理不同：
    /// * **OBMC**（§7.11.3.10）：`p0` = 当前重建值、`p1` = 邻居预测（先钳到 `max`）；
    /// * **interintra**（§7.11.3.13/14）：`p1` = 帧间那半先 `Round2(·, InterPostRound)`
    ///   再钳到 `bit_max`，而 `p0` 是已经写在平面里的帧内那半。
    ///
    /// # Safety
    /// 必须已确认 AVX2 可用；`p0`/`dst` 至少 `n` 项（可以指向同一处——逐 8 个像素
    /// 先读后写）、`p1`/`m` 至少 `n` 项。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn blend_masked_u16in8(
        dst: *mut u16,
        p0: *const u16,
        p1: *const i32,
        m: *const i32,
        n: usize,
        shift: u32,
        max: i32,
        p1_shift: u32,
        p1_max: i32,
    ) {
        let c64 = _mm256_set1_epi32(64);
        let pmax = _mm256_set1_epi32(p1_max);
        let zero = _mm256_setzero_si256();
        let mut j = 0usize;
        while j + super::LANES <= n {
            let v0 = _mm256_cvtepu16_epi32(_mm_loadu_si128(p0.add(j) as *const __m128i));
            let v1 = rnd_any(_mm256_loadu_si256(p1.add(j) as *const __m256i), p1_shift);
            let v1 = _mm256_min_epi32(_mm256_max_epi32(v1, zero), pmax);
            let vm = _mm256_loadu_si256(m.add(j) as *const __m256i);
            let s = _mm256_add_epi32(
                _mm256_mullo_epi32(vm, v0),
                _mm256_mullo_epi32(_mm256_sub_epi32(c64, vm), v1),
            );
            pack_store8(dst.add(j), rnd_any(s, shift), max);
            j += super::LANES;
        }
        while j < n {
            let vm = *m.add(j);
            let v0 = i32::from(*p0.add(j));
            let v1 = rnd_scalar(*p1.add(j), p1_shift).clamp(0, p1_max);
            let v = rnd_scalar(vm * v0 + (64 - vm) * v1, shift).clamp(0, max);
            *dst.add(j) = v as u16;
            j += 1;
        }
    }

    /// **平均混合**一行（`COMPOUND_AVERAGE`）：`Clip1( Round2( p0 + p1, shift ) )`。
    ///
    /// # Safety
    /// 同 [`blend_masked8`]（不需要 `m`）。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn blend_avg8(
        dst: *mut u16,
        p0: *const i32,
        p1: *const i32,
        n: usize,
        shift: u32,
        max: i32,
    ) {
        let mut j = 0usize;
        while j + super::LANES <= n {
            let v0 = _mm256_loadu_si256(p0.add(j) as *const __m256i);
            let v1 = _mm256_loadu_si256(p1.add(j) as *const __m256i);
            pack_store8(dst.add(j), rnd_any(_mm256_add_epi32(v0, v1), shift), max);
            j += super::LANES;
        }
        while j < n {
            let v = rnd_scalar(*p0.add(j) + *p1.add(j), shift).clamp(0, max);
            *dst.add(j) = v as u16;
            j += 1;
        }
    }

    /// **距离加权混合**一行（`COMPOUND_DISTANCE`）：
    /// `Clip1( Round2( fwd·p0 + bck·p1, shift ) )`（两个权重之和恒为 16）。
    ///
    /// # Safety
    /// 同 [`blend_masked8`]。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn blend_dist8(
        dst: *mut u16,
        p0: *const i32,
        p1: *const i32,
        fwd: i32,
        bck: i32,
        n: usize,
        shift: u32,
        max: i32,
    ) {
        let wf = _mm256_set1_epi32(fwd);
        let wb = _mm256_set1_epi32(bck);
        let mut j = 0usize;
        while j + super::LANES <= n {
            let v0 = _mm256_loadu_si256(p0.add(j) as *const __m256i);
            let v1 = _mm256_loadu_si256(p1.add(j) as *const __m256i);
            let s = _mm256_add_epi32(_mm256_mullo_epi32(wf, v0), _mm256_mullo_epi32(wb, v1));
            pack_store8(dst.add(j), rnd_any(s, shift), max);
            j += super::LANES;
        }
        while j < n {
            let v = rnd_scalar(fwd * *p0.add(j) + bck * *p1.add(j), shift).clamp(0, max);
            *dst.add(j) = v as u16;
            j += 1;
        }
    }

    /// 把一行预测值（i32）**钳位后落到平面行**（i32 → u16），一次 8 个样本。
    ///
    /// 帧间预测各条写回路径原来是逐像素 `Plane::set`（每像素一次下标计算 +
    /// 一次边界检查）。这里：`min`（钳上界）→ `packus`（负数收成 0、同时收窄成
    /// u16）→ 顺序修正（`packus` 是按 128 位道交错的）→ 一次 16 字节写。
    ///
    /// # Safety
    /// 必须已确认 AVX2 可用；`dst` 处可写 `n` 个 u16、`src` 至少 `n` 项。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn store_row_clamped(dst: *mut u16, src: &[i32], n: usize, max: i32) {
        let vmax = _mm256_set1_epi32(max);
        let mut j = 0usize;
        while j + 8 <= n {
            let v = _mm256_min_epi32(
                _mm256_loadu_si256(src.as_ptr().add(j) as *const __m256i),
                vmax,
            );
            let packed = _mm256_packus_epi32(v, v);
            let ordered = _mm256_permute4x64_epi64::<0b0000_1000>(packed);
            _mm_storeu_si128(dst.add(j) as *mut __m128i, _mm256_castsi256_si128(ordered));
            j += 8;
        }
        while j < n {
            *dst.add(j) = (*src.get_unchecked(j)).clamp(0, max) as u16;
            j += 1;
        }
    }

    /// 反变换的**行趟**：4 行一组。返回处理掉的**行数**（4 的倍数；0 = 全走标量）。
    ///
    /// `t` 是 `w×h` 行优先的系数（i32；系数格 32×32 以外按 0），`data` 是 `w×h`
    /// 行优先的残差（i64）。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn inv_txfm_rows(
        t: &[i32],
        data: &mut [i64],
        w: usize,
        h: usize,
        kind: u32,
        log2_w: u32,
        r: u32,
        row_shift: u32,
        rect_prescale: bool,
        lossless: bool,
    ) -> usize {
        if !w.is_multiple_of(4) || !h.is_multiple_of(4) || w > 64 || h > 64 {
            return 0;
        }
        let mut v = [_mm256_setzero_si256(); 64];
        let mut i = 0usize;
        while i + 4 <= h {
            // 收集：通道 k = 行 (i+k) 的元素 j（行/列 ≥ 32 按 0，与标量同）
            let mut j = 0usize;
            while j + 4 <= w {
                if j >= 32 {
                    v[j] = _mm256_setzero_si256();
                    v[j + 1] = _mm256_setzero_si256();
                    v[j + 2] = _mm256_setzero_si256();
                    v[j + 3] = _mm256_setzero_si256();
                } else {
                    // 4 行 × 4 列 → 4×4 i32 转置 → 符号扩展成 i64 通道
                    let mut rows = [_mm_setzero_si128(); 4];
                    for (k, row) in rows.iter_mut().enumerate() {
                        if i + k < 32 {
                            *row = _mm_loadu_si128(
                                t.as_ptr().add((i + k) * w + j) as *const __m128i
                            );
                        }
                    }
                    let t0 = _mm_unpacklo_epi32(rows[0], rows[1]);
                    let t1 = _mm_unpackhi_epi32(rows[0], rows[1]);
                    let t2 = _mm_unpacklo_epi32(rows[2], rows[3]);
                    let t3 = _mm_unpackhi_epi32(rows[2], rows[3]);
                    v[j] = _mm256_cvtepi32_epi64(_mm_unpacklo_epi64(t0, t2));
                    v[j + 1] = _mm256_cvtepi32_epi64(_mm_unpackhi_epi64(t0, t2));
                    v[j + 2] = _mm256_cvtepi32_epi64(_mm_unpacklo_epi64(t1, t3));
                    v[j + 3] = _mm256_cvtepi32_epi64(_mm_unpackhi_epi64(t1, t3));
                }
                j += 4;
            }
            // 1:2 长方块的预缩放（§7.13.3：`Round2( *2896, 12 )`）
            if rect_prescale {
                for slot in v[..w].iter_mut() {
                    *slot = vt_round2(vt_mul(*slot, 2896), 12);
                }
            }
            if lossless {
                vt_inverse_wht(&mut v[..4], 2);
            } else {
                vt_forward_1d(kind, &mut v[..w], log2_w, r);
            }
            // 写回：4 列一组做 4×4 i64 转置 → 每行的 4 个元素连续落盘
            let mut j = 0usize;
            while j + 4 <= w {
                let rows = vt_transpose4([v[j], v[j + 1], v[j + 2], v[j + 3]]);
                for (k, row) in rows.iter().enumerate() {
                    let o = vt_round2(*row, row_shift);
                    _mm256_storeu_si256(
                        data.as_mut_ptr().add((i + k) * w + j) as *mut __m256i,
                        o,
                    );
                }
                j += 4;
            }
            i += 4;
        }
        i
    }

    /// 反变换的**列趟**：4 列一组（`data` 里同一行的 4 个 i64 连续）。
    /// 返回处理掉的**列数**（4 的倍数；0 = 全走标量）。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn inv_txfm_cols(
        data: &mut [i64],
        w: usize,
        h: usize,
        kind: u32,
        log2_h: u32,
        r: u32,
        col_shift: u32,
        lossless: bool,
    ) -> usize {
        if !w.is_multiple_of(4) || !h.is_multiple_of(4) || w > 64 || h > 64 {
            return 0;
        }
        let mut v = [_mm256_setzero_si256(); 64];
        let mut j = 0usize;
        while j + 4 <= w {
            for (i, slot) in v[..h].iter_mut().enumerate() {
                *slot = _mm256_loadu_si256(data.as_ptr().add(i * w + j) as *const __m256i);
            }
            if lossless {
                vt_inverse_wht(&mut v[..4], 0);
            } else {
                vt_forward_1d(kind, &mut v[..h], log2_h, r);
            }
            for (i, slot) in v[..h].iter().enumerate() {
                let o = vt_round2(*slot, col_shift);
                _mm256_storeu_si256(data.as_mut_ptr().add(i * w + j) as *mut __m256i, o);
            }
            j += 4;
        }
        j
    }

    /// **Wiener 横向趟**：整档（`nrows = h + 6` 行 × `w` 列）一次做完，中间量写 i32。
    ///
    /// 只在**列窗口不触边**（`x - 3 >= 0 && x + w + 3 <= plane_end_x`）时调用 ——
    /// 此时 `lr.rs` 的列解析表是等差的（`xs[c] = x - 3 + c`），8 列一组连续读即可；
    /// 行解析表 `rows_ptr/offs` 照旧传进来（行可能跨 stripe / 跨帧，表里已解析好）。
    ///
    /// 与标量**逐位一致**：`vpmulld`/`vpaddd` 就是 i32 回绕乘加（i32 加法在回绕意义下
    /// 可交换结合，标量的累加顺序不影响结果），`rnd` 同 `round2`，
    /// `vpminsd`/`vpmaxsd` 同 `clamp`。
    ///
    /// 尾组（`w` 不是 8 的倍数）**重叠重算**最后一组：滤波是纯函数，重叠写出的值相同，
    /// 省掉标量尾巴。
    ///
    /// # Safety
    /// 必须已确认 AVX2 可用；`rows_ptr.len() >= nrows`、`offs.len() >= nrows`；
    /// `interm.len() >= nrows * w`；`w >= 8`；各行基址 + 偏移 + `col0 + w + 6`
    /// 落在各自的平面缓冲内（调用方已按解析表保证）。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn wiener_horiz_rows(
        rows_ptr: &[*const u16],
        offs: &[usize],
        nrows: usize,
        col0: isize,
        w: usize,
        coef: &[i32; 7],
        round: u32,
        lo: i32,
        hi: i32,
        interm: &mut [i32],
    ) {
        let cv: [__m256i; 7] = std::array::from_fn(|t| _mm256_set1_epi32(coef[t]));
        let lo_v = _mm256_set1_epi32(lo);
        let hi_v = _mm256_set1_epi32(hi);
        let ip = interm.as_mut_ptr();
        for r in 0..nrows {
            let row = (*rows_ptr.get_unchecked(r)).add(*offs.get_unchecked(r));
            let out_row = ip.add(r * w);
            let mut c = 0usize;
            while c < w {
                let group = if c + 8 <= w { c } else { w - 8 };
                let p = row.offset(col0 + group as isize);
                let mut acc = _mm256_setzero_si256();
                for (t, cvv) in cv.iter().enumerate() {
                    let s = _mm256_cvtepu16_epi32(_mm_loadu_si128(p.add(t) as *const __m128i));
                    acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s, *cvv));
                }
                let v = rnd(acc, round);
                let v = _mm256_min_epi32(_mm256_max_epi32(v, lo_v), hi_v);
                _mm256_storeu_si256(out_row.add(group) as *mut __m256i, v);
                c = group + 8;
            }
        }
    }

    /// **Wiener 纵向趟**的一行：
    /// `out[c] = Clip1( Round2( Σ_t coef[t] · interm[(r + t) * w + c], round ) )`
    /// （`Clip1` 到 `[0, max]`；标量侧随后还会经 `set_row` 再钳一次，值相同）。
    ///
    /// # Safety
    /// 必须已确认 AVX2 可用；`interm.len() >= (r + 7) * w`、`out.len() >= w`、`w >= 8`。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn wiener_vert_row(
        interm: &[i32],
        w: usize,
        r: usize,
        coef: &[i32; 7],
        round: u32,
        max: i32,
        out: &mut [i32],
    ) {
        let cv: [__m256i; 7] = std::array::from_fn(|t| _mm256_set1_epi32(coef[t]));
        let zero = _mm256_setzero_si256();
        let max_v = _mm256_set1_epi32(max);
        let ip = interm.as_ptr();
        let op = out.as_mut_ptr();
        let mut c = 0usize;
        while c < w {
            let group = if c + 8 <= w { c } else { w - 8 };
            let mut acc = _mm256_setzero_si256();
            for (t, cvv) in cv.iter().enumerate() {
                let s = _mm256_loadu_si256(ip.add((r + t) * w + group) as *const __m256i);
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(s, *cvv));
            }
            let v = rnd(acc, round);
            let v = _mm256_min_epi32(_mm256_max_epi32(v, zero), max_v);
            _mm256_storeu_si256(op.add(group) as *mut __m256i, v);
            c = group + 8;
        }
    }

}

#[cfg(target_arch = "x86_64")]
pub(crate) use x86::{
    diffwtd_mask8,
    cdef_pair16, deblock_edge4, inv_txfm_cols, inv_txfm_rows, mc_delta8, mc_horiz16_8bpc,
    blend_avg8, blend_dist8, blend_masked8, blend_masked_u16in8, mc_horiz8, mc_horiz_rows_8bpc,
    mc_shift8, mc_vert8, mc_vert_rows_8bpc, sgr_ab_8, sgr_cols_update8, sgr_final_8,
    store_row_clamped, wiener_horiz_rows, wiener_vert_row,
};

/// 非 x86_64：没有 AVX2 内核，`avx2_available()` 恒为 false ⇒ 这两个入口
/// 永远走不到；留着是为了"这份代码在任何平台都能编译"（返回 0 = 全走标量）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn inv_txfm_rows(
    _t: &[i32],
    _data: &mut [i64],
    _w: usize,
    _h: usize,
    _kind: u32,
    _log2_w: u32,
    _r: u32,
    _row_shift: u32,
    _rect_prescale: bool,
    _lossless: bool,
) -> usize {
    0
}

/// 非 x86_64：见 [`inv_txfm_rows`]（`mc_horiz16_8bpc` 恒不调用，见 `avx2_available`）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn mc_horiz16_8bpc(
    _row: &[u16],
    _base: isize,
    _coef: &[i16; 8],
    _round: u32,
    _out: &mut [i32],
) {
}
#[cfg(not(target_arch = "x86_64"))]
pub(crate) unsafe fn store_row_clamped(_dst: *mut u16, _src: &[i32], _n: usize, _max: i32) {}

/// 非 x86_64 的 CDEF 桩（`avx2_available()` 恒假 ⇒ 调用方走标量）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cdef_pair16(
    _data: &[u16],
    _stride: isize,
    _x0: isize,
    _y0: isize,
    _h: usize,
    _wu: usize,
    _offs: &[isize; CDEF_CANDS],
    _taps: &[i32; CDEF_CANDS],
    _pri_thr: i32,
    _pri_adj: u32,
    _sec_thr: i32,
    _sec_adj: u32,
    _out: &mut [i16],
) {
}

/// 非 x86_64 的 Wiener 桩（`avx2_available()` 恒假 ⇒ 调用方走标量）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn wiener_horiz_rows(
    _rows_ptr: &[*const u16],
    _offs: &[usize],
    _nrows: usize,
    _col0: isize,
    _w: usize,
    _coef: &[i32; 7],
    _round: u32,
    _lo: i32,
    _hi: i32,
    _interm: &mut [i32],
) {
}

/// 非 x86_64 的 Wiener 桩，见 [`wiener_horiz_rows`]。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn wiener_vert_row(
    _interm: &[i32],
    _w: usize,
    _r: usize,
    _coef: &[i32; 7],
    _round: u32,
    _max: i32,
    _out: &mut [i32],
) {
}

/// 非 x86_64 的混合桩（`avx2_available()` 恒假 ⇒ 调用方走标量）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn blend_masked8(
    _dst: *mut u16,
    _p0: *const i32,
    _p1: *const i32,
    _m: *const i32,
    _n: usize,
    _shift: u32,
    _max: i32,
) {
}

/// 非 x86_64 的混合桩，见 [`blend_masked_u16in8`]。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn blend_masked_u16in8(
    _dst: *mut u16,
    _p0: *const u16,
    _p1: *const i32,
    _m: *const i32,
    _n: usize,
    _shift: u32,
    _max: i32,
    _p1_shift: u32,
    _p1_max: i32,
) {
}

/// 非 x86_64 的整块横向桩（`avx2_available()` 恒假 ⇒ 走标量逐行）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn mc_horiz_rows_8bpc(
    _data: &[u16],
    _stride: usize,
    _rows: &[usize],
    _base: isize,
    _wu: usize,
    _coef: &[i16; 8],
    _round: u32,
    _interior_x: bool,
    _last_x: i64,
    _out: &mut [i16],
) {
}

/// 非 x86_64 的整块纵向桩，见 [`mc_horiz_rows_8bpc`]。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn mc_vert_rows_8bpc(
    _inter: &[i16],
    _wu: usize,
    _h: usize,
    _coef: &[i16; 8],
    _round: u32,
    _out: *mut i32,
) {
}

/// 非 x86_64 的混合桩，见 [`blend_masked8`]。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn blend_avg8(
    _dst: *mut u16,
    _p0: *const i32,
    _p1: *const i32,
    _n: usize,
    _shift: u32,
    _max: i32,
) {
}

/// 非 x86_64 的混合桩，见 [`blend_masked8`]。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn blend_dist8(
    _dst: *mut u16,
    _p0: *const i32,
    _p1: *const i32,
    _fwd: i32,
    _bck: i32,
    _n: usize,
    _shift: u32,
    _max: i32,
) {
}

/// 非 x86_64 的列趟桩，见 [`inv_txfm_rows`]。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn inv_txfm_cols(
    _data: &mut [i64],
    _w: usize,
    _h: usize,
    _kind: u32,
    _log2_h: u32,
    _r: u32,
    _col_shift: u32,
    _lossless: bool,
) -> usize {
    0
}

// ── 下面这组桩：deblock / SGR(自引导恢复) / 遮蔽混合 / 8 通道取整内核 ───────────
//
// **它们与上面那批是同一份契约，只是长期漏掉了**（Apple Silicon、Linux arm64、
// Android arm64 因此整体编译不过 —— 只有 x86_64 构建过，所以一直没暴露）。
// 每一处调用点都由 `avx2_available()`（或由它推导出的 `simd` / `simd_cols` /
// `avx2` 布尔）门住，而 `avx2_available()` 在非 x86_64 上**恒为 false**（编译期
// 常量），所以这些函数在 arm64 上永不执行，调用方一律落到标量路径。
//
// ⚠️ 新增桩时必须逐个确认调用点确实带那道门：桩是**空实现**，一旦被真的调用，
// 结果是"画面悄悄错一点点"，比编译不过危险得多。

/// 非 x86_64 的去块桩（`avx2_available()` 恒假 ⇒ `deblock` 走标量 `filter_edge`）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn deblock_edge4(
    _data: *mut u16,
    _stride: isize,
    _bit_depth: u8,
    _xp: i32,
    _yp: i32,
    _dx: i32,
    _dy: i32,
    _plane: u8,
    _filter_size: i32,
    _need: i32,
    _limit: i32,
    _blimit: i32,
    _thresh: i32,
) {
}

/// 非 x86_64 的 SGR 列累加桩（`avx2_available()` 恒假 ⇒ 逐列标量累加）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn sgr_cols_update8(
    _add_row: *const u16,
    _sub_row: *const u16,
    _cs: *mut i32,
    _cs2: *mut i32,
    _cols: usize,
) {
}

/// 非 x86_64 的 SGR 系数桩（`avx2_available()` 恒假 ⇒ 标量逐列算 a/b）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn sgr_ab_8(
    _pre_s: &[i32],
    _pre_s2: &[i32],
    _lo: usize,
    _hi: usize,
    _n: i32,
    _s_factor: i32,
    _one_over_n: i32,
    _sh_a: u32,
    _sh_d: u32,
    _mtab_bits: u32,
    _recip_bits: u32,
    _table: &[i32; 256],
    _a_out: &mut [i32],
    _b_out: &mut [i32],
) {
}

/// 非 x86_64 的 SGR 收尾桩（`avx2_available()` 恒假 ⇒ 标量逐样本合成）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn sgr_final_8(
    _a_buf: &[i32],
    _b_buf: &[i32],
    _aw: usize,
    _base: usize,
    _wts: &[[i32; 3]; 3],
    _u: *const u16,
    _shift: u32,
    _out: &mut [i32],
) {
}

/// 非 x86_64 的差异加权遮蔽桩（`avx2_available()` 恒假 ⇒
/// `inter::build_diffwtd_mask` 走标量两步式）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn diffwtd_mask8(
    _p0: &[i32],
    _p1: &[i32],
    _n: usize,
    _shift: u32,
    _mask_type: u8,
    _out: &mut [i32],
) {
}

/// 非 x86_64 的取整/移位桩（`avx2_available()` 恒假 ⇒ 标量 `round2` + 左移）。
#[cfg(not(target_arch = "x86_64"))]
pub(crate) unsafe fn mc_shift8(_row: &[u16], _base: isize, _post_round: u32, _out: &mut [i32]) {}

/// 非 x86_64 的横向 8 通道桩（`avx2_available()` 恒假 ⇒ 标量抽头累加）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn mc_horiz8(
    _row: &[u16],
    _base: isize,
    _coef: &[i16; 8],
    _round: u32,
    _out: &mut [i16],
) {
}

#[cfg(test)]
mod tests {
    use super::*;
    /// `diffwtd_mask8` vs **标量两步式**（`inter::build_diffwtd_mask` 的写法）。
    ///
    /// 内核把 `Round2(|Δ|, s) / 16` 并成 `(|Δ| + half) >> (s + 4)` —— 这条代数
    /// 只对**非负**整数成立（右移 = 向下取整除法，两层可以并）。这里把 8/10/12bit
    /// 的 `shift`（4/6/8）与实际出现的 `|Δ|`（预测值是 `pixel << InterPostRound`，
    /// 所以上限是 `((1<<bd)-1) << 4`）全扫一遍，含撞 64 上限与 `mask_type = 1`。
    ///
    /// 只跑在 x86_64：这条用例测的是 **AVX2 内核**与标量两步式的逐位一致，
    /// 非 x86_64 平台上没有这个内核（`avx2_available()` 恒假，调用方直接走标量），
    /// 标量路径本身由解码用例 + `ME_NO_SIMD` 对拍覆盖。
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn diffwtd_mask8_matches_the_two_step_formula() {
        if !avx2_available() {
            return;
        }
        for bd in [8u32, 10, 12] {
            for ipr in [4u32, 7] {
                let shift = bd.saturating_sub(8) + ipr;
                let maxd = ((1i32 << bd) - 1) << ipr;
                let mut dels = vec![
                    0i32, 1, 2, 7, 8, 15, 16, 17, 31, 32, 63, 64, 127, 128, 255, 256, 511, 1023,
                    2047, 4095, 4096, 8191, 16383, 32767, 65535,
                ];
                dels.push(maxd);
                dels.push(maxd - 1);
                let n = dels.len();
                let p0: Vec<i32> = dels.clone();
                let p1: Vec<i32> = vec![0; n];
                // 再加上"两边都非零"的组合，避免只测了 p1 == 0。
                let mut p0b = Vec::new();
                let mut p1b = Vec::new();
                for &d in dels.iter() {
                    let hi = maxd.min(i32::MAX - d);
                    p0b.push(hi);
                    p1b.push(hi - d);
                }
                for (a, b) in [(&p0, &p1), (&p0b, &p1b)] {
                    for mt in [0u8, 1] {
                        let mut got = vec![0i32; n];
                        // SAFETY: 已判 AVX2；三个缓冲都恰好 n 项；shift >= 1。
                        unsafe { diffwtd_mask8(a, b, n, shift, mt, &mut got) };
                        for k in 0..n {
                            let d = (a[k] - b[k]).abs();
                            // 标量两步式（与改动前的实现逐字一致）
                            let step1 = (d + (1 << (shift - 1))) >> shift;
                            let m = (38 + step1 / 16).clamp(0, 64);
                            let want = if mt != 0 { 64 - m } else { m };
                            assert_eq!(
                                got[k], want,
                                "bd={bd} ipr={ipr} mt={mt} k={k} |d|={d}"
                            );
                        }
                    }
                }
            }
        }
    }

    use std::hint::black_box;

    /// **微基准**（`cargo test --release -p me-decode --lib -- --ignored --nocapture`）：
    /// MC 三个内核的**每像素耗时**。
    ///
    /// 为什么要它：`PHASE` 只告诉我们"MC 一共 20 ms/帧"，那是**内核 + 脚手架**的总和。
    /// 两边各占多少、该改哪一边，只有把内核单独拎出来量才知道（本机负载会飘，
    /// 所以只看**同一进程内**的相对量：ns/像素）。
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore]
    fn bench_mc_kernels() {
        if !avx2_available() {
            eprintln!("这台机器没有 AVX2，跳过");
            return;
        }
        let coef: [i16; 8] = [-4, 12, -24, 80, 80, -24, 12, -4];
        let stride = 140usize;
        let rows = 140usize;
        let row: Vec<u16> = (0..stride * rows)
            .map(|i| ((i * 37 + 11) % 256) as u16)
            .collect();
        let iters = 200_000usize;

        // 横向：一次 16 个输出
        let mut out16 = [0i16; 16];
        let mut sink = 0i64;
        let t = std::time::Instant::now();
        for i in 0..iters {
            let base = std::hint::black_box((i % 64) as isize);
            // SAFETY: 已判 AVX2；base ∈ [0,63] ⇒ 读 [base-3, base+20) ⊂ 140
            unsafe { mc_horiz16_8bpc(&row, base, &coef, 3, &mut out16) };
            sink += out16[black_box(0)] as i64;
        }
        let ns = t.elapsed().as_secs_f64() * 1e9 / (iters as f64);
        eprintln!("mc_horiz16_8bpc   {:.2} ns/调用 = {:.3} ns/像素 (sink={sink})", ns, ns / 16.0);

        // 纵向：一次 16 个输出（i16 中间量）
        let inter: Vec<i16> = (0..stride * rows).map(|i| (i % 300) as i16).collect();
        let mut out32 = [0i32; 16];
        let mut sink2 = 0i64;
        let t = std::time::Instant::now();
        for i in 0..iters {
            let r = std::hint::black_box(i % 64);
            // SAFETY: 已判 AVX2；r ∈ [0,63] ⇒ 读 [(r..r+8) 行] ⊂ 140 行
            unsafe { mc_vert8(&inter, r, 0, stride, &coef, 11, &mut out32) };
            sink2 += out32[black_box(3)] as i64;
        }
        let ns = t.elapsed().as_secs_f64() * 1e9 / (iters as f64);
        eprintln!("mc_vert8          {:.2} ns/调用 = {:.3} ns/像素 (sink={sink2})", ns, ns / 16.0);

        // 整像素（δ 相位）：一次 8 个输出
        let mut outd = [0i32; 8];
        let t = std::time::Instant::now();
        let mut sink3 = 0i64;
        for i in 0..iters {
            let base = std::hint::black_box((i % 64) as isize);
            // SAFETY: 已判 AVX2；base ∈ [0,63] ⇒ 读 [base, base+8) ⊂ 140
            unsafe { mc_shift8(&row, base, 2, &mut outd) };
            sink3 += outd[black_box(5)] as i64;
        }
        let ns = t.elapsed().as_secs_f64() * 1e9 / (iters as f64);
        eprintln!("mc_shift8         {:.2} ns/调用 = {:.3} ns/像素 (sink={sink3})", ns, ns / 8.0);

        // 落盘（钳位 + 收窄 + 一次 16 字节写）：一次 8 个像素
        let src: Vec<i32> = (0..8).map(|i| i * 17).collect();
        let mut dstrow = [0u16; 8];
        let t = std::time::Instant::now();
        for _ in 0..iters {
            // SAFETY: 已判 AVX2；dstrow 有 8 项、src 有 8 项
            unsafe { store_row_clamped(dstrow.as_mut_ptr(), &src, 8, 255) };
        }
        let ns = t.elapsed().as_secs_f64() * 1e9 / (iters as f64);
        eprintln!("store_row_clamped {:.2} ns/调用 = {:.3} ns/像素", ns, ns / 8.0);
    }

    /// 有 AVX2 的机器上，SIMD 与标量必须**逐位一致**（这就是"能开就开"的底气）。
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn horiz8_matches_scalar_bit_for_bit() {
        if !avx2_available() {
            return;
        }
        let coef: [i16; 8] = [0, 2, -14, 84, 66, -12, 2, 0];
        let row: Vec<u16> = (0..64u16).map(|i| (i * 37) % 4096).collect();
        for round in [3u32, 5] {
            for base in [3usize, 4, 9] {
                let mut got = [0i16; 8];
                // SAFETY: 已判 AVX2；base 在 [3, 9] ⇒ 覆盖 [base-3, base+12) ⊂ 64
                unsafe { mc_horiz8(&row, base as isize, &coef, round, &mut got) };
                for (c, g) in got.iter().enumerate() {
                    let mut s = 0i32;
                    for (t, k) in coef.iter().enumerate() {
                        s += *k as i32 * row[base + c + t - 3] as i32;
                    }
                    let want = if round == 0 { s } else { (s + (1 << (round - 1))) >> round };
                    assert_eq!(i32::from(*g), want, "c={c} round={round} base={base}");
                }
            }
        }
    }

    /// **化简形式必须与两趟取整逐位一致** —— 在全部四种 `(位深, 复合)` 组合上
    /// 跑**全值域**（8 位 256 个、12 位 4096 个输入），一个都不能差。
    ///
    /// 这条测试是"把两趟 `Round2` 换成一次左移"这个改动的全部依据。
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn shift8_equals_the_two_round_passes_over_the_whole_domain() {
        if !avx2_available() {
            return;
        }
        // 与 `inter::rounding_vars` 同一张表（这里照抄一份，好让测试独立于它的实现）
        let cases: [(u8, bool, u32, u32, u32); 4] = [
            (8, false, 3, 11, 0),
            (8, true, 3, 7, 4),
            (12, false, 5, 9, 0),
            (12, true, 5, 7, 2),
        ];
        for (bit_depth, _compound, r0, r1, post) in cases {
            let domain = 1u16 << bit_depth;
            let width = 64usize;
            let row: Vec<u16> = (0..width).map(|i| (i as u16) % domain).collect();
            for base in [0usize, 8, 16] {
                let mut got = [0i32; 8];
                // SAFETY: 已判 AVX2；base+8 <= width
                unsafe { mc_shift8(&row, base as isize, post, &mut got) };
                for (i, g) in got.iter().enumerate() {
                    let src = row[base + i] as i32;
                    let h1 = (128 * src + (1 << (r0 - 1))) >> r0;
                    let want = (128 * h1 + (1 << (r1 - 1))) >> r1;
                    assert_eq!(
                        *g, want,
                        "位深 {bit_depth} 的 src={src}：化简值 {} ≠ 两趟取整 {want}",
                        *g
                    );
                }
            }
            // 全值域扫一遍（只查数学，不用 SIMD，保证 4096 档也覆盖完整）
            for src in 0..domain as i32 {
                let h1 = (128 * src + (1 << (r0 - 1))) >> r0;
                let want = (128 * h1 + (1 << (r1 - 1))) >> r1;
                assert_eq!(src << post, want, "位深 {bit_depth} 的 src={src} 化简不成立");
            }
        }
    }

    /// 16 位通道的横向内核（`mc_horiz16_8bpc`）必须与标量**逐位一致**。
    ///
    /// ## 这份测试在 2026-10-03 之前是**残缺的**（这就是它放走 i16 溢出的原因）
    ///
    /// 旧版用 `max_sample ∈ {0, 255}` 造行 —— 那只会产出**常量行**。而常量行上
    /// `Σ c·s = s·Σc = 128·s ≤ 32640`，**永远不溢出**，正好绕开被考验的那条路径。
    /// 于是"最坏累加 32640 < 32767"这句话在测试里**恒真**，却在真素材上假。
    ///
    /// 现在改成三类**必然暴露溢出**的输入：
    /// 1. **全部 96 个真实相位**（不是手挑 4 个）—— 溢出只发生在 `Σ|c| > 128`
    ///    的那 48 个相位上，抽样漏一个就白测；
    /// 2. **0/255 混排行**（`(i·37) & 0x80 ? 255 : 0`）—— 让"正系数撞 255、
    ///    负系数撞 0"这种最坏前缀组合真的出现；
    /// 3. **随机行** × 每个相位多组 —— 兜住前两类没想到的组合。
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn horiz16_matches_scalar_bit_for_bit() {
        use crate::video::inter::SUBPEL_FILTERS;
        if !avx2_available() {
            return;
        }
        let width = 64usize;
        let mut phases: Vec<[i16; 8]> = Vec::new();
        for g in SUBPEL_FILTERS.iter() {
            for f in g.iter() {
                phases.push(*f);
            }
        }
        assert_eq!(phases.len(), 96, "滤波器表应是 6 组 × 16 相位");

        // 行样本：常量 / 斜坡 / 0-255 混排 / 随机，逐相位都跑一遍
        let mut rng: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 33) as u16
        };
        let rows_all255: Vec<u16> = vec![255u16; width];
        let rows_flat: Vec<u16> = (0..width as u16).map(|i| i * 4).collect();
        let rows_alt: Vec<u16> = (0..width)
            .map(|i| if (i * 37) & 0x80 != 0 { 255 } else { 0 })
            .collect();
        let rows_rnd: Vec<u16> = (0..width).map(|_| next() & 0xFF).collect();
        let row_sets: [(&str, &Vec<u16>); 4] = [
            ("全 255", &rows_all255),
            ("斜坡", &rows_flat),
            ("0/255 混排", &rows_alt),
            ("随机", &rows_rnd),
        ];

        for coef in phases {
            for (tag, row) in row_sets.iter() {
                // ⚠️ `round` 只测真实取值（`inter_round0` = 3/5）：内核的输出收窄到
                // i16 是**以 `round >= 3` 为前提**的（`round = 0` 时 `Σc·s` 可达
                // `144·255 = 36720`，装不下 —— 没有调用方会那么传）。
                for round in [3u32, 5] {
                    for base in [3usize, 4, 20, width - 20] {
                        let mut got = [0i16; 16];
                        // SAFETY: 已判 AVX2；base ∈ [3, 44] ⇒ 覆盖 [base-3, base+20) ⊆ 64
                        unsafe { mc_horiz16_8bpc(row, base as isize, &coef, round, &mut got) };
                        for (c, g) in got.iter().enumerate() {
                            let mut s = 0i32;
                            for (t, k) in coef.iter().enumerate() {
                                s += *k as i32 * row[base + c + t - 3] as i32;
                            }
                            let want =
                                if round == 0 { s } else { (s + (1 << (round - 1))) >> round };
                            assert_eq!(
                                i32::from(*g), want,
                                "行={tag} c={c} round={round} base={base} coef={coef:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// **把 16 位通道的界一步步钉死**（不是"约等于"）：`SUBPEL_FILTERS` 的
    /// 每个相位、每个累加前缀，取样本极值，都必须落在 i16 内。
    ///
    /// 这条测试是 2026-10-03 那个像素级回归的正向护栏：
    /// * 改滤波器系数 → 若某段前缀和超界，这里立刻红；
    /// * 改 [`HORIZ16_BIAS`] → 这里立刻红。
    ///
    /// 同时把"**前缀和**才是判据、`Σ|c|` 不是"这件事写成可执行的断言 ——
    /// 顺带记录当时的事实：96 个相位里 48 个 `Σ|c| > 128`、最坏 240。
    #[test]
    fn horiz16_bias_bounds_hold_for_every_phase() {
        use crate::video::inter::SUBPEL_FILTERS;
        const LIM: i32 = 32767;
        const MIN: i32 = -32768;
        let bias = super::HORIZ16_BIAS as i32;
        let mut over_128 = 0usize;
        let mut worst_abs = 0i32;
        for group in SUBPEL_FILTERS.iter() {
            for coef in group.iter() {
                let sum: i32 = coef.iter().map(|c| *c as i32).sum();
                assert_eq!(sum, 128, "相位 {coef:?} 的行代数和应为 128");
                let abs_sum: i32 = coef.iter().map(|c| (*c as i32).abs()).sum();
                if abs_sum > 128 {
                    over_128 += 1;
                }
                worst_abs = worst_abs.max(abs_sum);
                // 模拟 i16 累加：累加前缀，每步都用能把它顶到最偏的样本（0 或 255）
                for k in 1..=8 {
                    let pre = &coef[..k];
                    let up: i32 =
                        bias + 255 * pre.iter().filter(|c| **c > 0).map(|c| *c as i32).sum::<i32>();
                    let dn: i32 =
                        bias + 255 * pre.iter().filter(|c| **c < 0).map(|c| *c as i32).sum::<i32>();
                    assert!(
                        up <= LIM && dn >= MIN,
                        "相位 {coef:?} 前 {k} 项的 i16 累加越界：{dn}..{up}（偏置 {bias}）"
                    );
                }
                // 单项乘积（mullo_epi16）也不能回绕
                for c in coef.iter() {
                    assert!((*c as i32 * 255).abs() <= LIM, "单项 {c}·255 超出 i16");
                }
            }
        }
        assert_eq!(over_128, 48, "Σ|c| > 128 的相位个数（2026-10-03 实测 48/96）");
        assert_eq!(worst_abs, 240, "最坏 Σ|c|（2026-10-03 实测 240）");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn delta8_matches_scalar_bit_for_bit() {
        if !avx2_available() {
            return;
        }
        let row: Vec<u16> = (0..32u16).map(|i| (i * 101) % 4096).collect();
        for (r0, r1) in [(3u32, 11u32), (3, 7), (5, 9), (5, 7)] {
            let mut got = [0i32; 8];
            // SAFETY: 已判 AVX2；只读 8 个样本
            unsafe { mc_delta8(&row, 4, r0, r1, &mut got) };
            for (i, g) in got.iter().enumerate() {
                let src = row[4 + i] as i32;
                let h = (128 * src + (1 << (r0 - 1))) >> r0;
                let want = (128 * h + (1 << (r1 - 1))) >> r1;
                assert_eq!(*g, want, "i={i} r0={r0} r1={r1}");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vert8_matches_scalar_bit_for_bit() {
        if !avx2_available() {
            return;
        }
        let stride = 32usize;
        let rows = 12usize;
        // 中间量现在是 **i16**（与真实 MC 同一量级：±3000）
        let inter: Vec<i16> = (0..stride * rows)
            .map(|i| (((i as i32) * 61) % 6000 - 3000) as i16)
            .collect();
        let coef: [i16; 8] = [0, 2, -14, 84, 66, -12, 2, 0];
        for round in [7u32, 9, 11] {
            // `mc_vert8` 一次吃 16 个（i16 一个寄存器装 16 个）⇒ col + 16 ≤ stride
            for col in [0usize, 8, 16] {
            let mut got = [0i32; 16];
            // SAFETY: 已判 AVX2；base_row=2、8 行、stride=32、列偏移 col（+16 ≤ 32）
            unsafe { mc_vert8(&inter, 2, col, stride, &coef, round, &mut got) };
            for (c, g) in got.iter().enumerate() {
                let mut s = 0i32;
                for (t, k) in coef.iter().enumerate() {
                    s += *k as i32 * i32::from(inter[(2 + t) * stride + col + c]);
                }
                let want = (s + (1 << (round - 1))) >> round;
                assert_eq!(*g, want, "c={c} round={round} col={col}");
            }
            }
        }
    }

    /// **改这一刀之前的 CDEF 内核**（i32 通道、一次一行 8 个像素）—— 只作基准对照。
    ///
    /// 2026-10-04 第二十七轮把生产路径换成 [`cdef_pair16`]（i16 通道、两行一起），
    /// 本函数留着回答"到底快了多少"：本机墙钟的噪声到 ±40%，只有**同一进程内交替**
    /// 的微基准能给出可信的相对值。
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn cdef_block8_ref(
        data: &[u16],
        stride: isize,
        x0: isize,
        y0: isize,
        h: usize,
        offs: &[isize; CDEF_CANDS],
        taps: &[i32; CDEF_CANDS],
        pri_thr: i32,
        pri_adj: u32,
        sec_thr: i32,
        sec_adj: u32,
        out: &mut [i32],
    ) {
        use std::arch::x86_64::*;
        let zero = _mm256_setzero_si256();
        let one = _mm256_set1_epi32(1);
        let eight = _mm256_set1_epi32(8);
        let base = data.as_ptr();
        for r in 0..h {
            let row = (y0 + r as isize) * stride + x0;
            let cur = _mm256_cvtepu16_epi32(_mm_loadu_si128(base.offset(row) as *const __m128i));
            let mut acc = zero;
            let mut mx = cur;
            let mut mn = cur;
            for c in 0..CDEF_CANDS {
                let (thr, adj) = if c < 4 { (pri_thr, pri_adj) } else { (sec_thr, sec_adj) };
                let cand = _mm256_cvtepu16_epi32(_mm_loadu_si128(
                    base.offset(row + offs[c]) as *const __m128i
                ));
                let diff = _mm256_sub_epi32(cand, cur);
                let mag = _mm256_abs_epi32(diff);
                let t = _mm256_sub_epi32(
                    _mm256_set1_epi32(thr),
                    _mm256_srav_epi32(mag, _mm256_set1_epi32(adj as i32)),
                );
                let t = _mm256_max_epi32(_mm256_min_epi32(t, mag), zero);
                let val = _mm256_sign_epi32(t, diff);
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(val, _mm256_set1_epi32(taps[c])));
                mx = _mm256_max_epi32(mx, cand);
                mn = _mm256_min_epi32(mn, cand);
            }
            let adjust = _mm256_and_si256(_mm256_cmpgt_epi32(zero, acc), one);
            let v = _mm256_add_epi32(
                cur,
                _mm256_srai_epi32::<4>(_mm256_add_epi32(_mm256_sub_epi32(eight, adjust), acc)),
            );
            let v = _mm256_min_epi32(_mm256_max_epi32(v, mn), mx);
            _mm256_storeu_si256(out.as_mut_ptr().add(r * LANES) as *mut __m256i, v);
        }
    }

    /// CDEF 内核**微基准**：`cdef_pair16`（i16、两行）vs `cdef_block8_ref`（i32、一行）。
    ///
    /// ```sh
    /// cargo test --release -p me-decode --lib bench_cdef -- --ignored --nocapture
    /// ```
    ///
    /// 三轮交替、每轮取两次的最小值（机器负载的干扰只能靠"交替 + 取 min"压下去）。
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "微基准，手工跑（见函数文档）"]
    fn bench_cdef_kernels() {
        if !avx2_available() {
            eprintln!("这台机器没有 AVX2，跳过");
            return;
        }
        let stride = 40usize;
        let rows = 40usize;
        let data: Vec<u16> = (0..stride * rows)
            .map(|i| ((i * 37 + 11) % 256) as u16)
            .collect();
        let dir = 3usize;
        let (x0, y0) = (8isize, 8isize);
        let mut offs = [0isize; CDEF_CANDS];
        let mut taps = [0i32; CDEF_CANDS];
        let mut n = 0usize;
        const DIRS: [[[i32; 2]; 2]; 8] = [
            [[-1, 1], [-2, 2]],
            [[0, 1], [-1, 2]],
            [[0, 1], [0, 2]],
            [[0, 1], [1, 2]],
            [[1, 1], [2, 2]],
            [[1, 0], [2, 1]],
            [[1, 0], [2, 0]],
            [[1, 0], [2, -1]],
        ];
        for k in 0..2usize {
            for sign in [-1i32, 1] {
                let [dy, dx] = DIRS[dir][k];
                offs[n] = (sign * dy) as isize * stride as isize + (sign * dx) as isize;
                taps[n] = [4, 2][k];
                n += 1;
            }
        }
        for k in 0..2usize {
            for dir_off in [-2i32, 2] {
                let d = ((dir as i32 + dir_off) & 7) as usize;
                let [dy, dx] = DIRS[d][k];
                for sign in [-1i32, 1] {
                    offs[n] = (sign * dy) as isize * stride as isize + (sign * dx) as isize;
                    taps[n] = [2, 1][k];
                    n += 1;
                }
            }
        }
        let iters = 400_000usize;
        let (pri_thr, sec_thr, adj) = (32i32, 8i32, 1u32);
        let mut out_i32 = [0i32; 8 * LANES];
        let mut out_i16 = [0i16; 8 * 8];
        let mut best_new = f64::MAX;
        let mut best_old = f64::MAX;
        for _ in 0..3 {
            // 两个内核交替跑，机器负载对两者的影响才可比。
            let t = std::time::Instant::now();
            for _ in 0..iters {
                std::hint::black_box(unsafe {
                    cdef_pair16(
                        std::hint::black_box(&data),
                        stride as isize,
                        x0,
                        y0,
                        8,
                        8,
                        &offs,
                        &taps,
                        pri_thr,
                        adj,
                        sec_thr,
                        adj,
                        &mut out_i16,
                    )
                });
            }
            let ns = t.elapsed().as_secs_f64() * 1e9 / iters as f64;
            best_new = best_new.min(ns);

            let t = std::time::Instant::now();
            for _ in 0..iters {
                std::hint::black_box(unsafe {
                    cdef_block8_ref(
                        std::hint::black_box(&data),
                        stride as isize,
                        x0,
                        y0,
                        8,
                        &offs,
                        &taps,
                        pri_thr,
                        adj,
                        sec_thr,
                        adj,
                        &mut out_i32,
                    )
                });
            }
            let ns = t.elapsed().as_secs_f64() * 1e9 / iters as f64;
            best_old = best_old.min(ns);
        }
        eprintln!(
            "CDEF 8×8 块（亮度）：新 i16 两行 {best_new:7.1} ns/块、旧 i32 一行 {best_old:7.1} ns/块 \
             ⇒ {:.2}x",
            best_old / best_new
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn cdef_pair16_matches_scalar_bit_for_bit() {
        if !avx2_available() {
            return;
        }
        // 一块 40×32 的合成样本（stride 留足 → 8 通道读取不会越界）
        let stride = 40usize;
        let rows = 32usize;
        let mut data: Vec<u16> = (0..stride * rows)
            .map(|i| (((i * 37) % 251) as u16) + 3)
            .collect();
        // 造出台阶与振铃，才检验得到 constrain 的分支
        for y in 0..rows {
            for x in 0..stride {
                if (x / 8 + y / 8) % 2 == 0 {
                    data[y * stride + x] = (data[y * stride + x] / 2).max(1);
                }
            }
        }
        let dir = 3usize;
        let (x0, y0) = (8isize, 8isize);
        // 与 cdef.rs 里的表保持一致
        const DIRS: [[[i32; 2]; 2]; 8] = [
            [[-1, 1], [-2, 2]],
            [[0, 1], [-1, 2]],
            [[0, 1], [0, 2]],
            [[0, 1], [1, 2]],
            [[1, 1], [2, 2]],
            [[1, 0], [2, 1]],
            [[1, 0], [2, 0]],
            [[1, 0], [2, -1]],
        ];
        const PRI_TAPS: [[i32; 2]; 2] = [[4, 2], [3, 3]];
        const SEC_TAPS: [[i32; 2]; 2] = [[2, 1], [2, 1]];
        // `h`/`wu` 三种形态都覆盖：亮度 8×8、色度 4×4、以及**奇数行**（最后一行单独算）。
        for (h, wu) in [(8usize, 8usize), (4, 4), (5, 8)] {
            for (pri_str, sec_str, pri_adj, sec_adj, tap_sel) in [
                (32i32, 8i32, 0u32, 1u32, 0usize),
                (7, 3, 2, 3, 1),
                (0, 5, 0, 0, 0),
                // 12 位的最坏量级：阈值 240、`constrain` 结果 240、12 项 tap 累加 11520
                (240, 240, 0, 0, 0),
                (240, 240, 0, 0, 1),
            ] {
                let pri_taps = PRI_TAPS[tap_sel];
                let sec_taps = SEC_TAPS[tap_sel];
                let mut offs = [0isize; CDEF_CANDS];
                let mut taps = [0i32; CDEF_CANDS];
                let mut n = 0usize;
                for k in 0..2usize {
                    for sign in [-1i32, 1] {
                        let [dy, dx] = DIRS[dir][k];
                        offs[n] = (sign * dy) as isize * stride as isize + (sign * dx) as isize;
                        taps[n] = pri_taps[k];
                        n += 1;
                    }
                }
                for k in 0..2usize {
                    for dir_off in [-2i32, 2] {
                        let d = ((dir as i32 + dir_off) & 7) as usize;
                        let [dy, dx] = DIRS[d][k];
                        for sign in [-1i32, 1] {
                            offs[n] = (sign * dy) as isize * stride as isize + (sign * dx) as isize;
                            taps[n] = sec_taps[k];
                            n += 1;
                        }
                    }
                }
                let mut got = vec![0i16; h * wu];
                // SAFETY: 已判 AVX2；(x0,y0)=(8,8)、h≤8、候选 ±2 ⇒ 最远读到 (x0+9, y0+9)，
                // 都在 40×32 内；`out.len() == h·wu`。
                unsafe {
                    cdef_pair16(
                        &data, stride as isize, x0, y0, h, wu, &offs, &taps, pri_str, pri_adj,
                        sec_str, sec_adj, &mut got,
                    )
                };
                // 标量参照：就是 cdef.rs 里那段逐像素的写法
                for r in 0..h {
                    for c in 0..wu {
                        let idx = (y0 as usize + r) * stride + x0 as usize + c;
                        let cur = data[idx] as i32;
                        let (mut mx, mut mn, mut sum) = (cur, cur, 0i32);
                        for ci in 0..CDEF_CANDS {
                            let (thr, adj) =
                                if ci < 4 { (pri_str, pri_adj) } else { (sec_str, sec_adj) };
                            let p = data[idx.wrapping_add_signed(offs[ci])] as i32;
                            let diff = p - cur;
                            let mag = diff.abs();
                            let sign = if diff < 0 { -1 } else { 1 };
                            let t = (thr - (mag >> adj)).clamp(0, mag);
                            sum += taps[ci] * sign * t;
                            mx = mx.max(p);
                            mn = mn.min(p);
                        }
                        let adjust = if sum < 0 { 1 } else { 0 };
                        let want = (cur + ((8 + sum - adjust) >> 4)).clamp(mn, mx);
                        assert_eq!(
                            got[r * wu + c] as i32,
                            want,
                            "r={r} c={c} h={h} wu={wu} pri={pri_str} sec={sec_str}"
                        );
                    }
                }
            }
        }
    }

    /// `sgr_cols_update8`：与逐列标量**逐位相同**（含 `sub_row = null` 的只加不减档、
    /// 以及 `cols` 不是 8 的倍数的尾巴）。数据取边界值 + 伪随机（`i*37+11`）。
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn sgr_cols_update_matches_scalar_bit_for_bit() {
        if !avx2_available() {
            return;
        }
        for &cols in &[1usize, 7, 8, 9, 15, 16, 17, 70] {
            let mk = |seed: usize| -> Vec<u16> {
                (0..cols)
                    .map(|i| match (i + seed) % 5 {
                        0 => 0u16,
                        1 => 255,
                        2 => 1,
                        3 => 254,
                        _ => (((i * 37 + 11 + seed * 101) % 256) as u16),
                    })
                    .collect()
            };
            let add_row = mk(0);
            let sub_row = mk(3);
            for &sub_null in &[false, true] {
                // 初值选成负数/大数都有（验证 i32 通道的加减不依赖符号）
                let init: Vec<i32> = (0..cols)
                    .map(|i| ((i as i32 * 977 + 13) % 20011) - 10000)
                    .collect();
                let mut cs = init.clone();
                let mut cs2 = init.clone();
                // SAFETY: 已判 AVX2；四个切片都 >= cols 项。
                unsafe {
                    sgr_cols_update8(
                        add_row.as_ptr(),
                        if sub_null {
                            std::ptr::null()
                        } else {
                            sub_row.as_ptr()
                        },
                        cs.as_mut_ptr(),
                        cs2.as_mut_ptr(),
                        cols,
                    )
                };
                for k in 0..cols {
                    let a = add_row[k] as i32;
                    let (dw, dw2) = if sub_null {
                        (a, a * a)
                    } else {
                        let s = sub_row[k] as i32;
                        (a - s, a * a - s * s)
                    };
                    assert_eq!(cs[k], init[k] + dw, "cols={cols} sub_null={sub_null} k={k} cs");
                    assert_eq!(cs2[k], init[k] + dw2, "cols={cols} sub_null={sub_null} k={k} cs2");
                }
            }
        }
    }

    /// Wiener 两个内核：与同式的标量循环**逐位相同**。
    ///
    /// 覆盖：8bit/12bit 的取整量（3/5 与 9/11）、`w` 不是 8 的倍数的**重叠尾组**、
    /// clamp 的上下边界（数据取满 12bit 值域 + 中间量取满 clamp 外）。
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn wiener_kernels_match_scalar_bit_for_bit() {
        if !avx2_available() {
            return;
        }
        // 伪随机（xorshift）：12bit 值域，且掺 0/255/4095 的极端行。
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 19) as i32
        };
        let coef: [i32; 7] = [-12, 19, -34, 182, -34, 19, -12]; // filter[3] = 128-2*(-27)
        for &(round_h, round_v, lo, hi, max) in
            &[(3u32, 11u32, -128i32, 383i32, 255i32), (5, 9, -512, 1535, 4095)]
        {
            for &w in &[8usize, 17, 64] {
                for &nrows in &[7usize, 70] {
                    // 行留够：最后一组的 8 宽读最高触到行内下标 `col0 + w + 5`。
                    let stride = w + 16;
                    let col0 = 3usize;
                    let src: Vec<u16> = (0..nrows * stride)
                        .map(|i| match i % 13 {
                            0 => 0,
                            1 => max as u16,
                            2 => 255,
                            _ => (next() % (max + 1)) as u16,
                        })
                        .collect();
                    let offs: Vec<usize> = (0..nrows).map(|k| k * stride).collect();
                    // `rows_ptr[k]` 是**帧基址**、`offs[k]` 才是行偏移（内核自己相加，
                    // 与 `lr.rs` 的解析表同语义）。
                    let ptrs: Vec<*const u16> = (0..nrows).map(|_| src.as_ptr()).collect();
                    let mut got = vec![0i32; nrows * w];
                    let mut want = vec![0i32; nrows * w];
                    // SAFETY: 已判 AVX2；每个源行都有 `stride = w + 8` 项 ⇒
                    // `col0 + w - 8 + 6 + 8 <= stride` 成立。
                    unsafe {
                        wiener_horiz_rows(
                            &ptrs, &offs, nrows, col0 as isize, w, &coef, round_h, lo, hi,
                            &mut got,
                        );
                    }
                    for r in 0..nrows {
                        for c in 0..w {
                            let mut acc = 0i32;
                            for t in 0..7 {
                                acc += coef[t] * src[r * stride + col0 + c + t] as i32;
                            }
                            want[r * w + c] = rnd_scalar(acc, round_h).clamp(lo, hi);
                        }
                    }
                    assert_eq!(got, want, "horiz w={w} nrows={nrows} round={round_h}");

                    // 纵向：中间量取到 clamp 外（±30000）以覆盖上下界。
                    let interm: Vec<i32> = (0..(nrows + 6) * w)
                        .map(|i| ((next() % 60001) - 30000).abs() * if i % 11 == 0 { -1 } else { 1 })
                        .collect();
                    let h = nrows - 6;
                    for r in 0..h {
                        let mut got = vec![0i32; w];
                        // SAFETY: 已判 AVX2；`interm` 有 `(h + 6) * w` 项，读到 `(r + 6) * w + w`。
                        unsafe {
                            wiener_vert_row(&interm, w, r, &coef, round_v, max, &mut got);
                        }
                        for c in 0..w {
                            let mut acc = 0i32;
                            for t in 0..7 {
                                acc += coef[t] * interm[(r + t) * w + c];
                            }
                            let want = rnd_scalar(acc, round_v).clamp(0, max);
                            assert_eq!(got[c], want, "vert r={r} c={c} w={w} round={round_v}");
                        }
                    }
                }
            }
        }
    }
}
