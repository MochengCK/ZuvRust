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

/// 横向内核**收尾**用的两个常数（[`x86::mc_horiz16_8bpc`] 等，2026-10-05 第三十三轮）。
///
/// ## 收尾原本在做什么
///
/// 带偏置累加之后 `acc = Σ c·s − 16384`（全是 i16 通道），要还原成
/// `Round2(Σ c·s, n) = (Σ + 2^(n−1)) >> n`。旧写法是
/// "`cvtepi16_epi32` ×2 → `add_epi32` 抵消偏置 ×2 → `rnd`（`add` + `srai`）×2 →
/// `packs_epi32` → `permute4x64`"，**11 条指令**，而且 16 个输出要拆成两个
/// 8 通道的 i32 向量再拼回去。
///
/// ## 现在只有 2 条：`pmulhrsw` + `paddw`
///
/// `PMULHRSW(a, b) = sat16((a·b + 2¹⁴) >> 15)`。取 `b = 2^(15−n)`（[`horiz_round_mul`]）：
///
/// ```text
///   (acc·2^(15−n) + 2¹⁴) >> 15  ==  (acc + 2^(n−1)) >> n              （n ≤ 14 时整除）
///                               ==  ((Σ − 2¹⁴) + 2^(n−1)) >> n
///                               ==  (Σ + 2^(n−1)) >> n − 2^(14−n)     （2¹⁴ 被 2^n 整除）
/// ```
///
/// ⇒ 再加上 [`horiz_round_add`] 就是 `Round2(Σ c·s, n)`。
///
/// ⚠️ 是**加**不是减：`acc` 已经比真正的 `Σ` 小了 `2¹⁴`，所以 `pmulhrsw` 的结果
/// 比目标**小** `2^(14−n)`。写反过一次（`psubw`），`horiz16_matches_scalar_bit_for_bit`
/// 当场变红 —— 那条单测就是为这种事存在的。
///
/// 全程留在 i16：`pmulhrsw` 的饱和分支**永远不触发**（`Σ` 的值域使取整结果落在
/// `[−1785, 5865]` 附近，再加一个 ≤ 8192 的常数也远在 i16 内）。
/// 省下的是 9 条指令 / 16 个输出 —— 横向内核整条链从 ~36 条降到 ~20 条。
pub(crate) const fn horiz_round_mul(round: u32) -> i16 {
    (1i32 << (15 - round)) as i16
}

/// 见 [`horiz_round_mul`]：`2^(14−round)` = `16384 >> round`（`round ≤ 14` 时整除）。
pub(crate) const fn horiz_round_add(round: u32) -> i16 {
    (16384i32 >> round) as i16
}

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

    /// [`rnd`] 的 **128 位版**（同样的档位、同样的"未知档位原样返回"语义）。
    ///
    /// 给纵向 MC 的 **8 宽 / 4 宽**补丁用（2026-10-05 第三十七轮）：色度块窄到
    /// `wu = 8`（33.7%）甚至 `4`（6.6%）时，16 宽的主循环一次都进不去。
    #[inline]
    pub(super) unsafe fn rnd128(acc: __m128i, n: u32) -> __m128i {
        match n {
            0 => acc,
            3 => _mm_srai_epi32::<3>(_mm_add_epi32(acc, _mm_set1_epi32(4))),
            5 => _mm_srai_epi32::<5>(_mm_add_epi32(acc, _mm_set1_epi32(16))),
            7 => _mm_srai_epi32::<7>(_mm_add_epi32(acc, _mm_set1_epi32(64))),
            9 => _mm_srai_epi32::<9>(_mm_add_epi32(acc, _mm_set1_epi32(256))),
            11 => _mm_srai_epi32::<11>(_mm_add_epi32(acc, _mm_set1_epi32(1024))),
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
    /// `BIAS + Σ c·s`）。每个抽头仍是一次 `mullo_epi16` + `add_epi16`、仍一次
    /// 覆盖 16 个输出，值域被平移到 `[-30664, 30536]`（96 相位 × 8 前缀全值域
    /// 枚举钉住）⇒ 与 i32 版/标量**逐位一致**。
    ///
    /// 2026-10-05（第三十三轮）把**收尾**也从 11 条压到 2 条（`pmulhrsw` + `paddw`，
    /// 推导见 [`super::horiz_round_mul`]）—— 内核整条链 ≈ 20 条 / 16 个输出。
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
        // ── 收尾（2026-10-05 第三十三轮）：`pmulhrsw` + `paddw`，2 条指令 ──────
        //
        // 原来这里要 `cvtepi16_epi32` ×2 → `add_epi32` ×2（抵消偏置）→ `rnd` ×2
        // （各 `add` + `srai`）→ `packs_epi32` → `permute4x64`，共 11 条。
        // 推导见 [`super::horiz_round_mul`]：`PMULHRSW(acc, 2^(15−n))` 恰好等于
        // `(acc + 2^(n−1)) >> n`，而 `acc = Σ − 2¹⁴` ⇒ 再加上 `2^(14−n)` 就还原成
        // `Round2(Σ c·s, n)`。全程 i16、无饱和、**与旧实现逐位相同**
        // （由 `horiz16_matches_scalar_bit_for_bit` 对 96 个相位逐值对拍）。
        debug_assert!(
            (1..=14).contains(&round),
            "pmulhrsw 收尾要求 round ∈ 1..=14（读到 {round}）"
        );
        let v = _mm256_add_epi16(
            _mm256_mulhrs_epi16(acc, _mm256_set1_epi16(super::horiz_round_mul(round))),
            _mm256_set1_epi16(super::horiz_round_add(round)),
        );
        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, v);
    }

    /// **横向 8 抽头、u8 源**（16 位通道，**只用于 8bit**）—— 用来量"u8 存储"这条路的
    /// 真实收益，**结论：不是收益、是代价**。
    ///
    /// ## 为什么写它（2026-10-05 把这条算清了）
    ///
    /// `AV1-AUDIT.md` 里"8bit 走 u8 存储（~5~10%）"的前置一直写着"MC 横向内核换
    /// `pmaddubsw`"。真正算一遍之后：
    ///
    /// * `pmaddubsw(u8 像素, i8 系数)` 的每一对是 `sat(i16, s·c_a + s'·c_b)`，
    ///   而 AV1 的 `SUBPEL_FILTERS` 在 **384 个「(相位, 抽头对)」里有 18 个**
    ///   `|c_a| + |c_b| > 32767/255 = 128.5`（最坏 **138**：
    ///   `[-2,8,-18,120,26,-10,6,-2]` 的第 2 对，`138·255 = 35190`）⇒ **会饱和 ⇒
    ///   直接用是错的**。要绕开只能拆抽头 / 加偏置，多出来的指令把"一次算两个抽头"
    ///   的便宜吃回去。
    /// * 8bit 的非 δ 相位 `max|c| = 126`（装得进 i8）—— 唯一超过的是 **δ 相位的 128**，
    ///   而 δ（整像素）本来就走 `mc_shift_row`。⇒ **卡住的不是系数范围，是配对和**。
    /// * 退一步只把"载入"换成 u8 + `vpmovzxbw`（不做 `pmaddubsw`，就是本函数）：
    ///   数据 L1 常驻（一行 8 抽头才 256 B），瓶颈是**载入条数**不是字节数 ——
    ///   u8 版每条只搬 16 B，条数**一模一样**（8 条），却多出 8 条 `vpmovzxbw`
    ///   ⇒ 纯亏。`bench_mc_kernels` 就是拿它来把这条量出来（同一台机器、同一段时间）。
    ///
    /// # Safety
    /// 必须已确认 AVX2 可用；`row` 覆盖 `[base-3, base+16+4)`、`out.len() >= 16`；
    /// `round >= 2`（同 [`Self::mc_horiz16_8bpc`]）。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn mc_horiz16_u8(
        row: &[u8],
        base: isize,
        coef: &[i16; 8],
        round: u32,
        out: &mut [i16],
    ) {
        let p = row.as_ptr();
        let mut acc = _mm256_set1_epi16(super::HORIZ16_BIAS);
        for (t, c) in coef.iter().enumerate() {
            // 一次 16 字节载入 + `vpmovzxbw`（LLVM 会把载入折进这条指令）
            let s8 = _mm_loadu_si128(p.offset(base + t as isize - 3) as *const __m128i);
            let s = _mm256_cvtepu8_epi16(s8);
            acc = _mm256_add_epi16(acc, _mm256_mullo_epi16(s, _mm256_set1_epi16(*c)));
        }
        // 收尾同 [`Self::mc_horiz16_8bpc`]（`pmulhrsw` + `paddw`）。
        let v = _mm256_add_epi16(
            _mm256_mulhrs_epi16(acc, _mm256_set1_epi16(super::horiz_round_mul(round))),
            _mm256_set1_epi16(super::horiz_round_add(round)),
        );
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

    /// 横向 8 抽头的 **8 输出版**（128 位通道）—— 见 [`mc_horiz_rows_8bpc`] 里
    /// 「为什么需要它」那段。
    ///
    /// 语义与 [`mc_horiz16_8bpc`] 的**前 8 个通道**逐位相同（同一条带偏置的
    /// i16 累加链、同一套 `pmulhrsw` 收尾），只是通道数减半、源样本用
    /// `_mm_loadu_si128` 读 16 字节。
    ///
    /// # Safety
    /// - 必须已确认 AVX2 可用。
    /// - `row` 覆盖 `[base - 3, base + 11)` 这 15 个 u16（`row` 必须是**这一行的首
    ///   地址**，`base` 是输出 `c = 0` 对应的源下标）。
    /// - `out` 至少 8 项。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn mc_horiz8_8bpc(row: *const u16, base: isize, coef: &[i16; 8], round: u32, out: *mut i16) {
        let p = row.offset(base - 3);
        let mut acc = _mm_set1_epi16(super::HORIZ16_BIAS);
        for (t, c) in coef.iter().enumerate() {
            // SAFETY: 见函数文档；`t ∈ [0, 8)` ⇒ 最远读到 `base - 3 + 7 + 7 = base + 11`。
            let s = _mm_loadu_si128(p.add(t) as *const __m128i);
            acc = _mm_add_epi16(acc, _mm_mullo_epi16(s, _mm_set1_epi16(*c)));
        }
        // 收尾同 [`mc_horiz16_8bpc`]（`pmulhrsw` + `paddw`）。
        let v = _mm_add_epi16(
            _mm_mulhrs_epi16(acc, _mm_set1_epi16(super::horiz_round_mul(round))),
            _mm_set1_epi16(super::horiz_round_add(round)),
        );
        _mm_storeu_si128(out as *mut __m128i, v);
    }

    /// # Safety
    /// 必须已确认 AVX2 可用；`data` 覆盖所有 `rows[r] * stride + [0, stride)`；
    /// `out` 至少 `rows.len() * wu` 项。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn mc_horiz_rows_8bpc(
        data: &[u16],
        stride: usize,
        y0: i64,
        last_y: i64,
        h: usize,
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
        // 收尾常数（`pmulhrsw` + `paddw`，见 [`super::horiz_round_mul`]）——整块共用，
        // 不必每个 16 列批次重算。
        debug_assert!(
            (1..=14).contains(&round),
            "pmulhrsw 收尾要求 round ∈ 1..=14（读到 {round}）"
        );
        let mul = _mm256_set1_epi16(super::horiz_round_mul(round));
        let add = _mm256_set1_epi16(super::horiz_round_add(round));
        // 边界块的钳位窗口（[`MAX_SB_SIZE`] + 6）；只有 `!interior_x` 会用到。
        let mut scratch = [0u16; 136];
        for r in 0..h {
            // 行下标**现算**，不收 `&[usize]`：调用方为它开一个
            // `[0usize; MAX_SB_SIZE + 16]` 的数组就等于每次调用先把 1.1 KB 清零
            // （1080p 一帧约 7.6 万次调用 ≈ 88 MB 白写）。见 `mc_shift_rows` 的同款说明。
            let sy = (y0 + r as i64 - 3).clamp(0, last_y) as usize;
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
                    _mm256_storeu_si256(
                        out_row.add(c) as *mut __m256i,
                        _mm256_add_epi16(_mm256_mulhrs_epi16(acc, mul), add),
                    );
                    c += 16;
                }
                // ── 8 宽补一刀（2026-10-05）────────────────────────────────
                //
                // 原来这条循环的门槛是 `c + 16 <= wu`，于是**所有 `wu ∈ [8, 16)` 的块
                // 整段走标量**。4:2:0 下 `BLOCK_16X16` 的色度平面正好 8 宽、
                // `BLOCK_8X8`/`BLOCK_4X4` 的色度是 4 宽 —— 真素材实测：**64% 的块
                // 色度宽 < 16**（33.7% 是 8、6.6% 是 4），也就是色度那一路的横向
                // 滤波原本全是标量。
                //
                // 8 个输出从 `base + c - 3` 起读 8 抽头 × 8 元素 ⇒ 最远
                // `base + c + 11`；而 `c + 8 ≤ wu` ⇒ 最远 `base + wu + 3`，
                // **正是 `interior_x` 保证的那个界**（不多读一个字节）。
                while c + 8 <= wu {
                    // SAFETY: 已判 AVX2；上面那条不等式就是 `mc_horiz8_8bpc` 的前提。
                    mc_horiz8_8bpc(row, base + c as isize, coef, round, out_row.add(c));
                    c += 8;
                }
                // 不钳位的标量尾：`interior_x` 已经保证 `[base-3, base+wu+3]` 全在帧内，
                // 逐样本 `clamp` 是白做（那是给**边界块**用的）。
                while c < wu {
                    let bc = base + c as isize - 3;
                    let mut s = 0i32;
                    for (t, k) in coef.iter().enumerate() {
                        s += *k as i32 * i32::from(*row.offset(bc + t as isize));
                    }
                    *out_row.add(c) = rnd_scalar(s, round) as i16;
                    c += 1;
                }
                continue;
            }
            // ── 边界块（`!interior_x`）：把窗口**钳位复制到栈上**，再走同一条 SIMD ──
            //
            // 原来整行逐样本标量（每像素 8 次"钳位 + 乘加"）。行宽 ≤ 128 ⇒ 窗口
            // ≤ 134 个样本；先把 `[base-3, base+wu+3)` 钳位拷进 scratch，后面的
            // 16 宽 / 8 宽 / 标量尾都从 scratch 读，**语义与逐样本钳位逐位相同**。
            // （真素材上边界块约占 MC 横向时间的 3%。）
            if crate::video::flags::no_horiz_edge_simd() {
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
                continue;
            }
            let n = wu + 6;
            debug_assert!(n <= 136, "边界块的钳位窗口超出栈缓冲");
            for i in 0..n {
                let sx = (base + i as isize - 3).clamp(0, last_x as isize) as usize;
                scratch[i] = *row.add(sx);
            }
            let sc = scratch.as_ptr();
            while c + 16 <= wu {
                let p = sc.add(c);
                let mut acc = _mm256_set1_epi16(super::HORIZ16_BIAS);
                for (t, cvv) in cv.iter().enumerate() {
                    let s = _mm256_loadu_si256(p.add(t) as *const __m256i);
                    acc = _mm256_add_epi16(acc, _mm256_mullo_epi16(s, *cvv));
                }
                _mm256_storeu_si256(
                    out_row.add(c) as *mut __m256i,
                    _mm256_add_epi16(_mm256_mulhrs_epi16(acc, mul), add),
                );
                c += 16;
            }
            while c + 8 <= wu {
                mc_horiz8_8bpc(sc, (c + 3) as isize, coef, round, out_row.add(c));
                c += 8;
            }
            while c < wu {
                let mut s = 0i32;
                for (t, k) in coef.iter().enumerate() {
                    s += *k as i32 * i32::from(*sc.add(c + t));
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
            // ── 8 宽 / 4 宽补刀（2026-10-05 第三十七轮）────────────────────
            //
            // 主循环的门槛是 `c + 16 <= wu` ⇒ **`wu < 16` 的块整段走标量**。而 4:2:0 的
            // 色度块宽分布与亮度完全不同：`BLOCK_16X16` 的色度正好 8 宽（33.7%）、
            // `BLOCK_8X8`/`BLOCK_4X4` 是 4 宽（6.6%）—— **四成的色度块纵向整段是标量**，
            // 而纵向标量每个像素要 8 次 i32 乘加。
            //
            // 与横向那条 `mc_horiz8_8bpc` 同一个道理（那里补过、纵向漏了）。128 位通道
            // 每轮 8 个输出：4 对抽头 × (2 载入 + 2 `unpack` + 2 `madd` + 2 `add`)
            // + 2 取整 + 2 存 = 36 条 / 8 输出（比 16 宽版的 38 条 / 16 输出贵一倍，
            // 但比标量的 ~16 条/像素便宜 3.5 倍）。
            //
            // `wu` 恒是 4 的倍数（块宽 4/8/16/…）⇒ 补完 8 宽再补 4 宽之后
            // **标量尾永远不会进入**（留着它只为"万一"）。
            let pairs128: [__m128i; 4] = std::array::from_fn(|j| {
                _mm_set1_epi32(
                    ((coef[2 * j + 1] as u16 as i32) << 16) | (coef[2 * j] as u16 as i32),
                )
            });
            while c + 8 <= wu {
                let mut lo_acc = _mm_setzero_si128();
                let mut hi_acc = _mm_setzero_si128();
                for (j, pair) in pairs128.iter().enumerate() {
                    let t = 2 * j;
                    let a = _mm_loadu_si128(base.add(t * wu + c) as *const __m128i);
                    let b = _mm_loadu_si128(base.add((t + 1) * wu + c) as *const __m128i);
                    lo_acc =
                        _mm_add_epi32(lo_acc, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), *pair));
                    hi_acc =
                        _mm_add_epi32(hi_acc, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), *pair));
                }
                _mm_storeu_si128(out_row.add(c) as *mut __m128i, rnd128(lo_acc, round));
                _mm_storeu_si128(out_row.add(c + 4) as *mut __m128i, rnd128(hi_acc, round));
                c += 8;
            }
            while c + 4 <= wu {
                let mut acc = _mm_setzero_si128();
                for (j, pair) in pairs128.iter().enumerate() {
                    let t = 2 * j;
                    let a = _mm_loadl_epi64(base.add(t * wu + c) as *const __m128i);
                    let b = _mm_loadl_epi64(base.add((t + 1) * wu + c) as *const __m128i);
                    acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), *pair));
                }
                _mm_storeu_si128(out_row.add(c) as *mut __m128i, rnd128(acc, round));
                c += 4;
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

    /// **纵向 8 抽头 + 直接写 u16 平面**（单参考、8bit 的直写路径）。
    ///
    /// 与 [`Self::mc_vert_rows_8bpc`] 的算术**逐位相同**（同一串 `pmaddwd` 累加、
    /// 同一取整），只是把"i32 中间缓冲"换成"取整 → 钳到 `[0, max]` → `packus` →
    /// 一次 32 字节写进目标平面"。省掉的是：
    ///
    /// * 整块 `w·h·4` 字节的暂存写（1080p 一帧几 MB），
    /// * 以及之后"再读一遍 i32、逐行钳位、转 u16、写平面"的整趟拷贝。
    ///
    /// 钳位在这里做（`packus` 只饱和到 0..65535，8bit 还要收到 `max`）。
    ///
    /// # Safety
    /// - 必须已确认 AVX2 可用。
    /// - `inter` 至少 `(h+7) * wu` 项；`dst` 起址之后每一行 `[0, wu)` 可写
    ///   （`dst_stride` 是平面步长，`dst` 已含块的 `(x, y)` 偏移）。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn mc_vert_rows_u16(
        inter: &[i16],
        wu: usize,
        h: usize,
        coef: &[i16; 8],
        round: u32,
        max: i32,
        dst: *mut u16,
        dst_stride: isize,
    ) {
        let ip = inter.as_ptr();
        let vmax = _mm256_set1_epi32(max);
        let zero = _mm256_setzero_si256();
        let pairs: [__m256i; 4] = std::array::from_fn(|j| {
            _mm256_set1_epi32(((coef[2 * j + 1] as u16 as i32) << 16) | (coef[2 * j] as u16 as i32))
        });
        let clamp = |v: __m256i| _mm256_min_epi32(_mm256_max_epi32(v, zero), vmax);
        for r in 0..h {
            let base = ip.add(r * wu);
            let out_row = dst.offset(r as isize * dst_stride);
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
                // 累加器的通道顺序是 `{0..4, 8..12}` / `{4..8, 12..16}`，
                // 先按 128 位道拼回 0..16，再 `packus` + `permute4x64(0xD8)` 排好序。
                let lo = clamp(rnd(lo_acc, round));
                let hi = clamp(rnd(hi_acc, round));
                let v0 = _mm256_permute2x128_si256::<0x20>(lo, hi);
                let v1 = _mm256_permute2x128_si256::<0x31>(lo, hi);
                let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(v0, v1));
                _mm256_storeu_si256(out_row.add(c) as *mut __m256i, packed);
                c += 16;
            }
            let pairs128: [__m128i; 4] = std::array::from_fn(|j| {
                _mm_set1_epi32(
                    ((coef[2 * j + 1] as u16 as i32) << 16) | (coef[2 * j] as u16 as i32),
                )
            });
            let vmax128 = _mm_set1_epi32(max);
            let zero128 = _mm_setzero_si128();
            let clamp128 =
                |v: __m128i| _mm_min_epi32(_mm_max_epi32(v, zero128), vmax128);
            while c + 8 <= wu {
                let mut lo_acc = _mm_setzero_si128();
                let mut hi_acc = _mm_setzero_si128();
                for (j, pair) in pairs128.iter().enumerate() {
                    let t = 2 * j;
                    let a = _mm_loadu_si128(base.add(t * wu + c) as *const __m128i);
                    let b = _mm_loadu_si128(base.add((t + 1) * wu + c) as *const __m128i);
                    lo_acc =
                        _mm_add_epi32(lo_acc, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), *pair));
                    hi_acc =
                        _mm_add_epi32(hi_acc, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), *pair));
                }
                let lo = clamp128(rnd128(lo_acc, round));
                let hi = clamp128(rnd128(hi_acc, round));
                _mm_storeu_si128(
                    out_row.add(c) as *mut __m128i,
                    _mm_packus_epi32(lo, hi),
                );
                c += 8;
            }
            while c + 4 <= wu {
                let mut acc = _mm_setzero_si128();
                for (j, pair) in pairs128.iter().enumerate() {
                    let t = 2 * j;
                    let a = _mm_loadl_epi64(base.add(t * wu + c) as *const __m128i);
                    let b = _mm_loadl_epi64(base.add((t + 1) * wu + c) as *const __m128i);
                    acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), *pair));
                }
                let v = clamp128(rnd128(acc, round));
                _mm_storel_epi64(
                    out_row.add(c) as *mut __m128i,
                    _mm_packus_epi32(v, v),
                );
                c += 4;
            }
            while c < wu {
                let mut s = 0i32;
                for (t, k) in coef.iter().enumerate() {
                    s += *k as i32 * i32::from(*ip.add((r + t) * wu + c));
                }
                *out_row.add(c) = rnd_scalar(s, round).clamp(0, max) as u16;
                c += 1;
            }
        }
    }

    /// **纵向两路 + AVERAGE 混合融合**（复合 AVERAGE、8bit、不缩放）。
    ///
    /// 输入是两路各自的横向中间量（i16），输出直接是目标平面的 u16：
    /// `Clip1(Round2(p0 + p1, 1 + InterPostRound))`，其中 `p = Round2(Σ fy·mid, InterRound1)`。
    /// 与"两路各跑一遍 `mc_vert_rows_8bpc` 写 i32、再 `blend_avg8`"逐位相同，
    /// 但**两路中间结果都不落内存**（省掉 2×(4B 写 + 4B 读) 每像素）。
    ///
    /// # Safety
    /// - 已确认 AVX2 可用；`h + 7 <= a.len()/wu`、`h + 7 <= b.len()/wu`；
    /// - `dst` 的 `h` 行（`dst_stride` 步长）各 `[0, wu)` 可写；
    /// - `round ∈ 1..=14`、`blend_shift ∈ 1..=31`。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn mc_vert2_avg_u16(
        a: &[i16],
        b: &[i16],
        wu: usize,
        h: usize,
        fya: &[i16; 8],
        fyb: &[i16; 8],
        round: u32,
        blend_shift: u32,
        max: i32,
        dst: *mut u16,
        dst_stride: isize,
    ) {
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let vmax = _mm256_set1_epi32(max);
        let zero = _mm256_setzero_si256();
        let bias = _mm256_set1_epi32(1i32 << (blend_shift - 1));
        let pair = |c: &[i16; 8], j: usize| -> __m256i {
            _mm256_set1_epi32(((c[2 * j + 1] as u16 as i32) << 16) | (c[2 * j] as u16 as i32))
        };
        let pa: [__m256i; 4] = std::array::from_fn(|j| pair(fya, j));
        let pb: [__m256i; 4] = std::array::from_fn(|j| pair(fyb, j));
        let pair128 = |c: &[i16; 8], j: usize| -> __m128i {
            _mm_set1_epi32(((c[2 * j + 1] as u16 as i32) << 16) | (c[2 * j] as u16 as i32))
        };
        let pa128: [__m128i; 4] = std::array::from_fn(|j| pair128(fya, j));
        let pb128: [__m128i; 4] = std::array::from_fn(|j| pair128(fyb, j));
        let vmax128 = _mm_set1_epi32(max);
        let zero128 = _mm_setzero_si128();
        let bias128 = _mm_set1_epi32(1i32 << (blend_shift - 1));
        let mix = |x: __m256i, y: __m256i| -> __m256i {
            let s = _mm256_add_epi32(_mm256_add_epi32(rnd(x, round), rnd(y, round)), bias);
            let sh = _mm256_set1_epi32(blend_shift as i32);
            let v = _mm256_srav_epi32(s, sh);
            _mm256_min_epi32(_mm256_max_epi32(v, zero), vmax)
        };
        let mix128 = |x: __m128i, y: __m128i| -> __m128i {
            let s = _mm_add_epi32(_mm_add_epi32(rnd128(x, round), rnd128(y, round)), bias128);
            let v = _mm_sra_epi32(s, _mm_cvtsi32_si128(blend_shift as i32));
            _mm_min_epi32(_mm_max_epi32(v, zero128), vmax128)
        };
        for r in 0..h {
            let ba = ap.add(r * wu);
            let bb = bp.add(r * wu);
            let out_row = dst.offset(r as isize * dst_stride);
            let mut c = 0usize;
            while c + 16 <= wu {
                let mut alo = _mm256_setzero_si256();
                let mut ahi = _mm256_setzero_si256();
                let mut blo = _mm256_setzero_si256();
                let mut bhi = _mm256_setzero_si256();
                for (j, (qa, qb)) in pa.iter().zip(pb.iter()).enumerate() {
                    let t = 2 * j;
                    let a0 = _mm256_loadu_si256(ba.add(t * wu + c) as *const __m256i);
                    let a1 = _mm256_loadu_si256(ba.add((t + 1) * wu + c) as *const __m256i);
                    alo = _mm256_add_epi32(alo, _mm256_madd_epi16(_mm256_unpacklo_epi16(a0, a1), *qa));
                    ahi = _mm256_add_epi32(ahi, _mm256_madd_epi16(_mm256_unpackhi_epi16(a0, a1), *qa));
                    let b0 = _mm256_loadu_si256(bb.add(t * wu + c) as *const __m256i);
                    let b1 = _mm256_loadu_si256(bb.add((t + 1) * wu + c) as *const __m256i);
                    blo = _mm256_add_epi32(blo, _mm256_madd_epi16(_mm256_unpacklo_epi16(b0, b1), *qb));
                    bhi = _mm256_add_epi32(bhi, _mm256_madd_epi16(_mm256_unpackhi_epi16(b0, b1), *qb));
                }
                let lo = mix(alo, blo);
                let hi = mix(ahi, bhi);
                let v0 = _mm256_permute2x128_si256::<0x20>(lo, hi);
                let v1 = _mm256_permute2x128_si256::<0x31>(lo, hi);
                let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(v0, v1));
                _mm256_storeu_si256(out_row.add(c) as *mut __m256i, packed);
                c += 16;
            }
            while c + 8 <= wu {
                let mut alo = _mm_setzero_si128();
                let mut ahi = _mm_setzero_si128();
                let mut blo = _mm_setzero_si128();
                let mut bhi = _mm_setzero_si128();
                for (j, (qa, qb)) in pa128.iter().zip(pb128.iter()).enumerate() {
                    let t = 2 * j;
                    let a0 = _mm_loadu_si128(ba.add(t * wu + c) as *const __m128i);
                    let a1 = _mm_loadu_si128(ba.add((t + 1) * wu + c) as *const __m128i);
                    alo = _mm_add_epi32(alo, _mm_madd_epi16(_mm_unpacklo_epi16(a0, a1), *qa));
                    ahi = _mm_add_epi32(ahi, _mm_madd_epi16(_mm_unpackhi_epi16(a0, a1), *qa));
                    let b0 = _mm_loadu_si128(bb.add(t * wu + c) as *const __m128i);
                    let b1 = _mm_loadu_si128(bb.add((t + 1) * wu + c) as *const __m128i);
                    blo = _mm_add_epi32(blo, _mm_madd_epi16(_mm_unpacklo_epi16(b0, b1), *qb));
                    bhi = _mm_add_epi32(bhi, _mm_madd_epi16(_mm_unpackhi_epi16(b0, b1), *qb));
                }
                let lo = mix128(alo, blo);
                let hi = mix128(ahi, bhi);
                _mm_storeu_si128(out_row.add(c) as *mut __m128i, _mm_packus_epi32(lo, hi));
                c += 8;
            }
            while c + 4 <= wu {
                let mut aa = _mm_setzero_si128();
                let mut ab = _mm_setzero_si128();
                for (j, (qa, qb)) in pa128.iter().zip(pb128.iter()).enumerate() {
                    let t = 2 * j;
                    let a0 = _mm_loadl_epi64(ba.add(t * wu + c) as *const __m128i);
                    let a1 = _mm_loadl_epi64(ba.add((t + 1) * wu + c) as *const __m128i);
                    aa = _mm_add_epi32(aa, _mm_madd_epi16(_mm_unpacklo_epi16(a0, a1), *qa));
                    let b0 = _mm_loadl_epi64(bb.add(t * wu + c) as *const __m128i);
                    let b1 = _mm_loadl_epi64(bb.add((t + 1) * wu + c) as *const __m128i);
                    ab = _mm_add_epi32(ab, _mm_madd_epi16(_mm_unpacklo_epi16(b0, b1), *qb));
                }
                let v = mix128(aa, ab);
                _mm_storel_epi64(out_row.add(c) as *mut __m128i, _mm_packus_epi32(v, v));
                c += 4;
            }
            while c < wu {
                let (mut sa, mut sb) = (0i32, 0i32);
                for t in 0..8usize {
                    sa += fya[t] as i32 * i32::from(*ap.add((r + t) * wu + c));
                    sb += fyb[t] as i32 * i32::from(*bp.add((r + t) * wu + c));
                }
                let v = (rnd_scalar(sa, round) + rnd_scalar(sb, round) + (1 << (blend_shift - 1)))
                    >> blend_shift;
                *out_row.add(c) = v.clamp(0, max) as u16;
                c += 1;
            }
        }
    }

    /// **残差加回（带水平翻转）**：`dst[j] = clip1(dst[j] + res[n-1-j])`。
    ///
    /// 逆变换带 `flip_lr` 的块（部分 ADST 变体）原来走逐像素标量；这里把
    /// 残差行**倒序**后再走同一条 16 宽流水线（`permutevar8x32` 两次 + 交换两半）。
    ///
    /// # Safety
    /// 已确认 AVX2；`dst` 可读写 `n` 个 u16、`res` 可读 `n` 个 i32。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn add_residual_u16_rev(
        dst: *mut u16,
        res: *const i32,
        n: usize,
        max: i32,
    ) {
        let zero = _mm256_setzero_si256();
        let vmax = _mm256_set1_epi32(max);
        let rev = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        let mut j = 0usize;
        while j + 16 <= n {
            // 目标 `[j, j+16)` 对应残差 `[n-j-16, n-j)`（倒序）
            let base = n - j - 16;
            let a = _mm256_permutevar8x32_epi32(
                _mm256_loadu_si256(res.add(base) as *const __m256i),
                rev,
            );
            let b = _mm256_permutevar8x32_epi32(
                _mm256_loadu_si256(res.add(base + 8) as *const __m256i),
                rev,
            );
            // 倒序后：前 8 个 = b 的倒序、后 8 个 = a 的倒序
            let d0 = _mm256_cvtepu16_epi32(_mm_loadu_si128(dst.add(j) as *const __m128i));
            let d1 = _mm256_cvtepu16_epi32(_mm_loadu_si128(dst.add(j + 8) as *const __m128i));
            let s0 = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(d0, b), zero), vmax);
            let s1 = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(d1, a), zero), vmax);
            let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(s0, s1));
            _mm256_storeu_si256(dst.add(j) as *mut __m256i, packed);
            j += 16;
        }
        while j < n {
            let v = (i32::from(*dst.add(j)) + *res.add(n - 1 - j)).clamp(0, max);
            *dst.add(j) = v as u16;
            j += 1;
        }
    }

    /// **反量化（8 个系数一批）**：`dq = sign · (|lv·q| & 0xFFFFFF) >> dqDenom`，
    /// 再 `Clip3(-(1<<(7+bd)), …, ·)`。
    ///
    /// ## 为什么不用 i64
    ///
    /// 规范式是 `lv·q`（i64）再取绝对值、掩 24 位。但：
    /// * `m = (lv·q) mod 2^24` 用 `mullo_epi32` + `& 0xFFFFFF` 就能拿到（低 24 位与
    ///   64 位乘积一致）；
    /// * `dq < 0 ⇔ lv < 0`（`q` 恒正）⇒ 符号只看 `lv`；
    /// * `|dq| mod 2^24 = lv >= 0 ? m : (-m) mod 2^24`。
    ///
    /// 于是整条链路都在 **8 路 i32** 里（原实现逐系数标量）。
    ///
    /// `n` 是系数个数（`tw·th`）；`q_dc` 用在第 0 个，其余用 `q_ac`。
    ///
    /// # Safety
    /// 已确认 AVX2；`levels` 可读 `n` 个 i32、`t` 可写 `n` 个 i32。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn dequantize_8(
        levels: *const i32,
        t: *mut i32,
        n: usize,
        q_dc: i32,
        q_ac: i32,
        denom_shift: u32,
        lo: i32,
        hi: i32,
    ) {
        let zero = _mm256_setzero_si256();
        let mask24 = _mm256_set1_epi32(0x00FF_FFFF);
        let vlo = _mm256_set1_epi32(lo);
        let vhi = _mm256_set1_epi32(hi);
        let q_first = _mm256_setr_epi32(q_dc, q_ac, q_ac, q_ac, q_ac, q_ac, q_ac, q_ac);
        let q_rest = _mm256_set1_epi32(q_ac);
        let mut i = 0usize;
        while i + 8 <= n {
            let lv = _mm256_loadu_si256(levels.add(i) as *const __m256i);
            if _mm256_testz_si256(lv, lv) != 0 {
                _mm256_storeu_si256(t.add(i) as *mut __m256i, zero);
                i += 8;
                continue;
            }
            let q = if i == 0 { q_first } else { q_rest };
            let m = _mm256_and_si256(_mm256_mullo_epi32(lv, q), mask24);
            let neg = _mm256_cmpgt_epi32(zero, lv);
            let negm = _mm256_and_si256(_mm256_sub_epi32(zero, m), mask24);
            let mag = _mm256_blendv_epi8(m, negm, neg);
            let mag = _mm256_srlv_epi32(mag, _mm256_set1_epi32(denom_shift as i32));
            let v = _mm256_sub_epi32(_mm256_xor_si256(mag, neg), neg);
            let v = _mm256_min_epi32(_mm256_max_epi32(v, vlo), vhi);
            _mm256_storeu_si256(t.add(i) as *mut __m256i, v);
            i += 8;
        }
        while i < n {
            let lv = *levels.add(i);
            let q = if i == 0 { q_dc } else { q_ac };
            let dq = lv as i64 * q as i64;
            let mag = ((dq.abs() & 0xFF_FFFF) >> denom_shift) as i32;
            let v = if dq < 0 { -mag } else { mag };
            *t.add(i) = v.clamp(lo, hi);
            i += 1;
        }
    }

    /// **残差加回平面（一行）**：`dst[j] = clip1(dst[j] + res[j])`，16 个样本一批。
    ///
    /// 逆变换的收尾原来是**逐像素标量**（每像素 widen + add + clamp + 窄化），
    /// 1080p 一帧几百万像素。这里 16 个一批：`cvtepu16_epi32` ×2 + 加 + 钳位 +
    /// `packus` + 一次 32 字节写回。
    ///
    /// # Safety
    /// - 已确认 AVX2；`dst` 可读写 `n` 个 u16，`res` 可读 `n` 个 i32；
    /// - `max = (1 << bit_depth) - 1`。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn add_residual_u16(dst: *mut u16, res: *const i32, n: usize, max: i32) {
        let zero = _mm256_setzero_si256();
        let vmax = _mm256_set1_epi32(max);
        let mut j = 0usize;
        while j + 16 <= n {
            let d0 = _mm256_cvtepu16_epi32(_mm_loadu_si128(dst.add(j) as *const __m128i));
            let d1 = _mm256_cvtepu16_epi32(_mm_loadu_si128(dst.add(j + 8) as *const __m128i));
            let r0 = _mm256_loadu_si256(res.add(j) as *const __m256i);
            let r1 = _mm256_loadu_si256(res.add(j + 8) as *const __m256i);
            let s0 = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(d0, r0), zero), vmax);
            let s1 = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(d1, r1), zero), vmax);
            let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(s0, s1));
            _mm256_storeu_si256(dst.add(j) as *mut __m256i, packed);
            j += 16;
        }
        while j + 8 <= n {
            let d0 = _mm256_cvtepu16_epi32(_mm_loadu_si128(dst.add(j) as *const __m128i));
            let r0 = _mm256_loadu_si256(res.add(j) as *const __m256i);
            let s0 = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(d0, r0), zero), vmax);
            let lo = _mm256_castsi256_si128(s0);
            let hi = _mm256_extracti128_si256::<1>(s0);
            _mm_storeu_si128(dst.add(j) as *mut __m128i, _mm_packus_epi32(lo, hi));
            j += 8;
        }
        while j < n {
            let v = (i32::from(*dst.add(j)) + *res.add(j)).clamp(0, max);
            *dst.add(j) = v as u16;
            j += 1;
        }
    }

    /// **复合 AVERAGE + 双整像素**：两路参考按行取四舍五入平均，直接写目标平面。
    ///
    /// 规范式是 `Clip1(Round2(p0 + p1, 1 + InterPostRound))`，而整像素位置的两路
    /// 预测值都是 `16·src`（`post_round = 14 - (3 + 7) = 4`）⇒
    /// `(16s0 + 16s1 + 16) >> 5 == (s0 + s1 + 1) >> 1`，正好是 `pavgw`。
    ///
    /// # Safety
    /// - 必须已确认 AVX2 可用；
    /// - `a`/`b` 的 `[ay + r, ay + r + h) × [ax, ax + w)` 与
    ///   `[by + r, by + r + h) × [bx, bx + w)` 都在各自平面内（调用方已判内点）；
    /// - `dst` 的同一矩形可写（帧平面含 `PLANE_PAD` 的填充区）。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn comp_avg_rows(
        a: *const u16,
        a_stride: usize,
        ax: usize,
        ay: usize,
        b: *const u16,
        b_stride: usize,
        bx: usize,
        by: usize,
        dst: *mut u16,
        dst_stride: usize,
        dx: usize,
        dy: usize,
        w: usize,
        h: usize,
    ) {
        for r in 0..h {
            let pa = a.add((ay + r) * a_stride + ax);
            let pb = b.add((by + r) * b_stride + bx);
            let pd = dst.add((dy + r) * dst_stride + dx);
            let mut c = 0usize;
            while c + 16 <= w {
                let va = _mm256_loadu_si256(pa.add(c) as *const __m256i);
                let vb = _mm256_loadu_si256(pb.add(c) as *const __m256i);
                _mm256_storeu_si256(pd.add(c) as *mut __m256i, _mm256_avg_epu16(va, vb));
                c += 16;
            }
            while c + 8 <= w {
                let va = _mm_loadu_si128(pa.add(c) as *const __m128i);
                let vb = _mm_loadu_si128(pb.add(c) as *const __m128i);
                _mm_storeu_si128(pd.add(c) as *mut __m128i, _mm_avg_epu16(va, vb));
                c += 8;
            }
            while c < w {
                let s = *pa.add(c) as u32 + *pb.add(c) as u32;
                *pd.add(c) = ((s + 1) >> 1) as u16;
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

    /// [`sgr_final_8`] 的**按权重档单态化**版本（2026-10-25）。
    ///
    /// `box_filter` 的 3×3 权重只可能是三种固定图案（见 `lr.rs` 里 `wt` 的构造）：
    /// - `PAT = 0`（`pass == 0`、`i` 偶）：缓冲行 0 与行 2 各 `[5, 6, 5]`；
    /// - `PAT = 1`（`pass == 0`、`i` 奇）：只有缓冲行 1 是 `[5, 6, 5]`；
    /// - `PAT = 2`（`pass == 1`）：九格全 4，仅四角为 3。
    ///
    /// 三种都能化成"先求和再缩放"：`5Σ+m`、`4Σ−角`。省掉每个缓冲 3~9 次
    /// `vpmulld`（本机 2 uop）。等价性要求**模 2³² 环运算**：`b_buf` 的
    /// `p*s_factor` 会溢出 i32 ⇒ 项可以是任意 i32，逐项乘与先和后缩放在
    /// 模 2³² 下才逐位一致（`5Σ+m` 里 `6m = 5m+m` 同理）。`SHIFT` 是行档常量
    /// （`pass == 0` 奇数行为 8，其余 9）⇒ 用立即数移位，省一次广播。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn sgr_final_8_pat<const PAT: u8, const SHIFT: i32>(
        a_buf: &[i32],
        b_buf: &[i32],
        aw: usize,
        base: usize,
        u: *const u16,
        out: &mut [i32],
    ) {
        #[inline(always)]
        unsafe fn ld(buf: &[i32], idx: usize) -> __m256i {
            _mm256_loadu_si256(buf.as_ptr().add(idx) as *const __m256i)
        }
        #[inline(always)]
        unsafe fn wsum<const PAT: u8>(buf: &[i32], base: usize, aw: usize) -> __m256i {
            match PAT {
                // `5(r0[c0]+r0[c1]+r0[c2]) + r0[c1]`
                1 => {
                    let r0 = ld(buf, base + aw);
                    let r1 = ld(buf, base + aw + 1);
                    let r2 = ld(buf, base + aw + 2);
                    let s = _mm256_add_epi32(_mm256_add_epi32(r0, r1), r2);
                    _mm256_add_epi32(_mm256_add_epi32(s, _mm256_slli_epi32::<2>(s)), r1)
                }
                // `5(行0 六格 + 行2 六格) + (两行中列)`
                0 => {
                    let x0 = ld(buf, base);
                    let x1 = ld(buf, base + 1);
                    let x2 = ld(buf, base + 2);
                    let y0 = ld(buf, base + 2 * aw);
                    let y1 = ld(buf, base + 2 * aw + 1);
                    let y2 = ld(buf, base + 2 * aw + 2);
                    let s = _mm256_add_epi32(
                        _mm256_add_epi32(_mm256_add_epi32(x0, x1), _mm256_add_epi32(x2, y0)),
                        _mm256_add_epi32(y1, y2),
                    );
                    let m = _mm256_add_epi32(x1, y1);
                    _mm256_add_epi32(_mm256_add_epi32(s, _mm256_slli_epi32::<2>(s)), m)
                }
                // `4·Σ9 − 四角`
                _ => {
                    let r00 = ld(buf, base);
                    let r01 = ld(buf, base + 1);
                    let r02 = ld(buf, base + 2);
                    let r10 = ld(buf, base + aw);
                    let r11 = ld(buf, base + aw + 1);
                    let r12 = ld(buf, base + aw + 2);
                    let r20 = ld(buf, base + 2 * aw);
                    let r21 = ld(buf, base + 2 * aw + 1);
                    let r22 = ld(buf, base + 2 * aw + 2);
                    let s = _mm256_add_epi32(
                        _mm256_add_epi32(
                            _mm256_add_epi32(_mm256_add_epi32(r00, r01), _mm256_add_epi32(r02, r10)),
                            _mm256_add_epi32(_mm256_add_epi32(r11, r12), _mm256_add_epi32(r20, r21)),
                        ),
                        r22,
                    );
                    let c = _mm256_add_epi32(
                        _mm256_add_epi32(r00, r02),
                        _mm256_add_epi32(r20, r22),
                    );
                    _mm256_sub_epi32(_mm256_slli_epi32::<2>(s), c)
                }
            }
        }
        let va = wsum::<PAT>(a_buf, base, aw);
        let vb = wsum::<PAT>(b_buf, base, aw);
        let uv = _mm256_cvtepu16_epi32(_mm_loadu_si128(u as *const __m128i));
        let v = _mm256_add_epi32(_mm256_mullo_epi32(va, uv), vb);
        let v = _mm256_srai_epi32::<SHIFT>(_mm256_add_epi32(
            v,
            _mm256_set1_epi32(1 << (SHIFT - 1)),
        ));
        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, v);
    }

    /// `box_filter` 的**前缀和**（两条 i32 数组一起做、一次 8 列）：
    /// `pre[k+1] = pre[k] + cs[k]`（`pre[0] = 0`），填到 `pre[1..=n]`。
    ///
    /// 原来两趟各 `n` 次**串行**标量加（`n = w + 2 + 2r`，1080p 一个 64 宽分块
    /// 每行 ~70 次 × 2 条链）。改成 8 路前缀扫描：lane 内 `slli+add` 两次、
    /// 再把低 128 位的末元素广播到高 128 位 —— 整数加法**可结合**，结果逐位相同。
    ///
    /// # Safety
    /// - 必须已确认 AVX2 可用；
    /// - `cs`/`cs2` 至少 `n` 项，`pre`/`pre2` 至少 `n + 1` 项。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn sgr_prefix8(
        cs: &[i32],
        cs2: &[i32],
        pre: &mut [i32],
        pre2: &mut [i32],
        n: usize,
    ) {
        #[inline(always)]
        unsafe fn scan8(v: __m256i) -> __m256i {
            let v = _mm256_add_epi32(v, _mm256_slli_si256::<4>(v));
            let v = _mm256_add_epi32(v, _mm256_slli_si256::<8>(v));
            // 把低 128 位的第 3 个元素搬到高 128 位的 4 个通道（低 128 位清零）：
            // `permute2x128::<0x08>` = [zero, v.low]，再在 128 位内广播第 3 个元素。
            let carry = _mm256_shuffle_epi32::<0xFF>(_mm256_permute2x128_si256::<0x08>(v, v));
            _mm256_add_epi32(v, carry)
        }
        pre[0] = 0;
        pre2[0] = 0;
        let mut s1 = _mm256_setzero_si256();
        let mut s2 = _mm256_setzero_si256();
        let mut k = 0usize;
        while k + 8 <= n {
            let v1 = _mm256_add_epi32(
                scan8(_mm256_loadu_si256(cs.as_ptr().add(k) as *const __m256i)),
                s1,
            );
            let v2 = _mm256_add_epi32(
                scan8(_mm256_loadu_si256(cs2.as_ptr().add(k) as *const __m256i)),
                s2,
            );
            _mm256_storeu_si256(pre.as_mut_ptr().add(k + 1) as *mut __m256i, v1);
            _mm256_storeu_si256(pre2.as_mut_ptr().add(k + 1) as *mut __m256i, v2);
            s1 = _mm256_permutevar8x32_epi32(v1, _mm256_set1_epi32(7));
            s2 = _mm256_permutevar8x32_epi32(v2, _mm256_set1_epi32(7));
            k += 8;
        }
        let mut a = _mm256_extract_epi32::<0>(s1);
        let mut b = _mm256_extract_epi32::<0>(s2);
        while k < n {
            a += *cs.get_unchecked(k);
            b += *cs2.get_unchecked(k);
            *pre.get_unchecked_mut(k + 1) = a;
            *pre2.get_unchecked_mut(k + 1) = b;
            k += 1;
        }
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

    /// `a2 = ((z << 8) + z/2) / (z + 1)` 的**浮点等价式**（2026-10-25）。
    ///
    /// 对 `z ∈ [1, 254]`，这个整数式与 `round(256·z/(z+1))`（就近取偶）**逐值相同**：
    /// `256z/(z+1) = 256 − 256/(z+1)`，值恰好落在 `x.5` 上当且仅当
    /// `512/(z+1)` 是奇数，而区间内 `z+1 ∈ [2,255]` 不含这样的值（512 的奇商只有
    /// `z+1 = 512`）。`z = 0` 取 1、`z >= 255` 取 256，与常量表一致。
    /// 单测 `sgr_a2_vec_matches_integer_formula` 对 `0..=255` 全量对拍。
    ///
    /// 为什么值得：`vpgatherdd` 在本机约 12~20 uop、20 拍延迟，而这里是
    /// 2 次 `cvtdq2ps` + 1 次 `vdivps`（吞吐 5 拍）+ 1 次 `cvtps2dq` + max/blend。
    /// ⚠️ `z < 0`（`p·s_factor` 的 i32 溢出档）**不走这里**：调用方仍用标量兜底，
    /// 本函数只保证非负输入与表相同（负输入被夹到 0 ⇒ 返回 1，调用方随后覆盖）。
    ///
    /// # Safety
    /// 必须已确认 AVX2 可用。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn sgr_a2_vec(z: __m256i) -> __m256i {
        let zero = _mm256_setzero_si256();
        let zc = _mm256_min_epi32(_mm256_max_epi32(z, zero), _mm256_set1_epi32(255));
        let zf = _mm256_cvtepi32_ps(zc);
        let q = _mm256_div_ps(
            _mm256_mul_ps(_mm256_set1_ps(256.0), zf),
            _mm256_add_ps(zf, _mm256_set1_ps(1.0)),
        );
        let a2 = _mm256_max_epi32(_mm256_cvtps_epi32(q), _mm256_set1_epi32(1));
        let hi = _mm256_cmpgt_epi32(zc, _mm256_set1_epi32(254));
        _mm256_blendv_epi8(a2, _mm256_set1_epi32(256), hi)
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
        // 2026-10-25：**gather 查表**。原来逐通道 8 次"比较 + 查表 + 分支"，
        // 而 `z` 的分布很散、分支预测不上。改成一次 `vpgatherdd`：
        // 先把 `z` 夹到 `[0,255]`（表末项已改成 256 ⇒ `z >= 255` 也正确），
        // 负值（真实素材里会因 i32 溢出出现）再用标量兜底。
        let idx = _mm256_min_epi32(_mm256_max_epi32(z, _mm256_setzero_si256()), _mm256_set1_epi32(255));
        let mut a2v = if !crate::video::flags::no_sgr_gather()
            && !crate::video::flags::no_sgr_float()
        {
            // 2026-10-25：**浮点等价式**取代 `vpgatherdd`（见 [`sgr_a2_vec`]）。
            sgr_a2_vec(z)
        } else if crate::video::flags::no_sgr_gather() {
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
                    ((zz << 8) + (zz / 2)) / (zz + 1)
                };
            }
            _mm256_loadu_si256(a2s.as_ptr() as *const __m256i)
        } else {
            _mm256_i32gather_epi32::<4>(table.as_ptr(), idx)
        };
        let neg = _mm256_cmpgt_epi32(_mm256_setzero_si256(), z);
        if _mm256_movemask_epi8(neg) != 0 {
            let mut zs = [0i32; 8];
            _mm256_storeu_si256(zs.as_mut_ptr() as *mut __m256i, z);
            let mut a2s = [0i32; 8];
            _mm256_storeu_si256(a2s.as_mut_ptr() as *mut __m256i, a2v);
            for k in 0..8 {
                let zz = zs[k];
                if zz < 0 {
                    // `SGRPROJ_SGR_BITS == 8`（与 `lr.rs` 的那个常量同源）
                    a2s[k] = ((zz << 8) + (zz / 2)) / (zz + 1);
                }
            }
            a2v = _mm256_loadu_si256(a2s.as_ptr() as *const __m256i);
        }
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

    /// [`mc_shift8`] 的**整行版**（2026-10-05）。
    ///
    /// 原来调用方是"每 8 个像素调一次 `mc_shift8`"—— 而 `#[target_feature]` 函数
    /// **不能被没有该 feature 的调用方内联**，于是整像素 MV 那条路上每一行要付
    /// `1920/8 = 240` 次真实函数调用（含序言/尾声/参数搬运）。把行循环搬进来之后
    /// 内层只剩 load / widen / shift / store。
    ///
    /// 语义与逐段调用 `mc_shift8` **逐位相同**（同一段代码、同一个 `post_round` 分支）。
    ///
    /// # Safety
    /// 必须已确认 AVX2 可用；`row` 覆盖 `[base, base + n)`；`out` 至少 `n` 项。
    /// 返回**实际用 SIMD 处理掉的个数**（`n` 向下取到 8 的倍数），余数留给调用方标量。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn mc_shift_row(
        row: &[u16],
        base: isize,
        post_round: u32,
        n: usize,
        out: &mut [i32],
    ) -> usize {
        let p = row.as_ptr();
        let o = out.as_mut_ptr();
        let mut c = 0usize;
        while c + 8 <= n {
            // SAFETY: 见函数文档；`c + 8 <= n` ⇒ 读 `[base+c, base+c+8) ⊂ [base, base+n)`。
            let s16 = _mm_loadu_si128(p.offset(base + c as isize) as *const __m128i);
            let s32 = _mm256_cvtepu16_epi32(s16);
            let v = match post_round {
                0 => s32,
                1 => _mm256_slli_epi32::<1>(s32),
                2 => _mm256_slli_epi32::<2>(s32),
                3 => _mm256_slli_epi32::<3>(s32),
                4 => _mm256_slli_epi32::<4>(s32),
                _ => s32,
            };
            _mm256_storeu_si256(o.add(c) as *mut __m256i, v);
            c += 8;
        }
        // 4 宽尾段（色度/窄块）：4 个 u16 → 4 个 i32，一条 128 位载入 + 零扩展 + 存储。
        if c + 4 <= n {
            let s16 = _mm_loadl_epi64(p.offset(base + c as isize) as *const __m128i);
            let s32 = _mm_cvtepu16_epi32(s16);
            let v = match post_round {
                0 => s32,
                1 => _mm_slli_epi32::<1>(s32),
                2 => _mm_slli_epi32::<2>(s32),
                3 => _mm_slli_epi32::<3>(s32),
                4 => _mm_slli_epi32::<4>(s32),
                _ => s32,
            };
            _mm_storeu_si128(o.add(c) as *mut __m128i, v);
            c += 4;
        }
        c
    }

    /// **整数相位（δ）快路径的整块版**：把 `rows` 里每一行都做完 `s << post_round`。
    ///
    /// ## 为什么成批（2026-10-05 第三十七轮）
    ///
    /// 原来调用方在 `for r in 0..h` 里**逐行**调 [`Self::mc_shift_row`] —— 而它是
    /// `#[target_feature(enable = "avx2")]`、**不能被无 feature 的调用方内联**
    /// ⇒ 1080p 一帧约 **20 万次真实函数调用**（每次都要付序言/epilogue）。
    /// 与第二十九轮 `mc_horiz16_8bpc → mc_horiz_rows_8bpc` 同一个原因、同一个改法。
    ///
    /// 语义与逐行版**逐位相同**（同一串指令、同一个 `post_round` 分派）。
    /// 返回三段的实际处理列数（各行相同，因为 `n` 是同一个）——调用方用它接着走标量尾。
    ///
    /// # Safety
    /// - 必须已确认 AVX2 可用。
    /// - `data` 覆盖所有 `rows[r] * stride + [0, stride)`；
    ///   `out` 至少 `rows.len() * n` 项。
    ///
    /// ⚠️ **行下标在核心里现算**（`clip(y0 + r, lastY)`），刻意**不**收一个
    /// `&[usize]`：调用方为它开一个 `[0usize; MAX_SB_SIZE]` 的数组就等于每次调用
    /// **先把 1 KB 清零**（1080p 一帧 20 万次调用 ≈ 200 MB 白写）—— 第一版就是这么写的，
    /// 实测反而**慢了 1.2%**，改成现算才回来。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn mc_shift_rows(
        data: &[u16],
        stride: usize,
        y0: i64,
        last_y: i64,
        h: usize,
        base: isize,
        post_round: u32,
        n: usize,
        out: *mut i32,
    ) -> usize {
        let dp = data.as_ptr();
        let mut c = 0usize;
        for r in 0..h {
            let sy = (y0 + r as i64).clamp(0, last_y) as usize;
            let row = dp.add(sy * stride);
            let o = out.add(r * n);
            let mut k = 0usize;
            while k + 8 <= n {
                // SAFETY: 见函数文档；`k + 8 <= n` ⇒ 读 `[base+k, base+k+8) ⊂ [base, base+n)`。
                let s16 = _mm_loadu_si128(row.offset(base + k as isize) as *const __m128i);
                let s32 = _mm256_cvtepu16_epi32(s16);
                let v = match post_round {
                    0 => s32,
                    1 => _mm256_slli_epi32::<1>(s32),
                    2 => _mm256_slli_epi32::<2>(s32),
                    3 => _mm256_slli_epi32::<3>(s32),
                    4 => _mm256_slli_epi32::<4>(s32),
                    _ => s32,
                };
                _mm256_storeu_si256(o.add(k) as *mut __m256i, v);
                k += 8;
            }
            // 4 宽尾段（色度/窄块）：与上面的 8 宽同一套移位，只是 128 位。
            if k + 4 <= n {
                let s16 = _mm_loadl_epi64(row.offset(base + k as isize) as *const __m128i);
                let s32 = _mm_cvtepu16_epi32(s16);
                let v = match post_round {
                    0 => s32,
                    1 => _mm_slli_epi32::<1>(s32),
                    2 => _mm_slli_epi32::<2>(s32),
                    3 => _mm_slli_epi32::<3>(s32),
                    4 => _mm_slli_epi32::<4>(s32),
                    _ => s32,
                };
                _mm_storeu_si128(o.add(k) as *mut __m128i, v);
                k += 4;
            }
            c = k;
        }
        c
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
    /// - `dst` 覆盖 `((y0+h-1)·dst_stride + x0) .. + wu 个 u16`（**就地写目标平面**；
    ///   `dst_stride` 通常是平面 stride，`x0` 是平面内的绝对列）。
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
        dst: *mut u16,
        dst_stride: isize,
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
        // 抽头归并后只剩 4 个不同值（主 c=0/2、次 c=4/8）。
        let pri_taps = [
            _mm256_set1_epi16(taps[0] as i16),
            _mm256_set1_epi16(taps[2] as i16),
        ];
        let sec_taps = [
            _mm256_set1_epi16(taps[4] as i16),
            _mm256_set1_epi16(taps[8] as i16),
        ];
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
            // 同抽头的候选先相加再乘（模 2^16 环等价），`constrain` 用无符号饱和减
            // —— 与 `cdef_pair16x4` 同一套化简，两版逐位一致（有单测钉住）。
            for grp in 0..2usize {
                let mut g = zero;
                for k in 0..2usize {
                    let c = grp * 2 + k;
                    let cand = _mm256_set_m128i(
                        _mm_loadu_si128(base.offset(off1 + offs[c]) as *const __m128i),
                        _mm_loadu_si128(base.offset(off0 + offs[c]) as *const __m128i),
                    );
                    let diff = _mm256_sub_epi16(cand, cur);
                    let mag = _mm256_abs_epi16(diff);
                    let t = _mm256_subs_epu16(pri_t, _mm256_sra_epi16(mag, pri_s));
                    let t = _mm256_min_epu16(t, mag);
                    g = _mm256_add_epi16(g, _mm256_sign_epi16(t, diff));
                    mx = _mm256_max_epi16(mx, cand);
                    mn = _mm256_min_epi16(mn, cand);
                }
                acc = _mm256_add_epi16(acc, _mm256_mullo_epi16(g, pri_taps[grp]));
            }
            for grp in 0..2usize {
                let mut g = zero;
                for k in 0..4usize {
                    let c = 4 + grp * 4 + k;
                    let cand = _mm256_set_m128i(
                        _mm_loadu_si128(base.offset(off1 + offs[c]) as *const __m128i),
                        _mm_loadu_si128(base.offset(off0 + offs[c]) as *const __m128i),
                    );
                    let diff = _mm256_sub_epi16(cand, cur);
                    let mag = _mm256_abs_epi16(diff);
                    let t = _mm256_subs_epu16(sec_t, _mm256_sra_epi16(mag, sec_s));
                    let t = _mm256_min_epu16(t, mag);
                    g = _mm256_add_epi16(g, _mm256_sign_epi16(t, diff));
                    mx = _mm256_max_epi16(mx, cand);
                    mn = _mm256_min_epi16(mn, cand);
                }
                acc = _mm256_add_epi16(acc, _mm256_mullo_epi16(g, sec_taps[grp]));
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
            let dst0 = dst.offset((y0 + r as isize) * dst_stride + x0);
            if wu == 8 {
                _mm_storeu_si128(dst0 as *mut __m128i, lo);
                if r + 1 < h {
                    _mm_storeu_si128(
                        dst.offset((y0 + r1 as isize) * dst_stride + x0) as *mut __m128i,
                        _mm256_extracti128_si256::<1>(v),
                    );
                }
            } else {
                _mm_storel_epi64(dst0 as *mut __m128i, lo);
                if r + 1 < h {
                    _mm_storel_epi64(
                        dst.offset((y0 + r1 as isize) * dst_stride + x0) as *mut __m128i,
                        _mm256_extracti128_si256::<1>(v),
                    );
                }
            }
            r += 2;
        }
    }

    /// [`cdef_pair16`] 的**四行一组**版本：一次算 4 行 × 8 列（两个 YMM）。
    ///
    /// 两行一组时每个候选的 `diff → abs → sra → sub → min → max → sign → mul → add`
    /// 是一条 ~9 级的依赖链；四行版让**两条链同时在飞**（每轮 32 个通道、循环减半），
    /// 用 ILP 盖住延迟。像素运算与 [`cdef_pair16`] 逐条相同 ⇒ 输出逐位一致
    /// （`cdef_pair16x4_matches_pair16_bit_for_bit` 钉住）。
    ///
    /// # Safety
    /// - 必须已确认 AVX2 可用；`h` 必须是 **4 的倍数**。
    /// - 对每个 `r ∈ [0, h)`、每个 `c ∈ [0, 12)`，`data` 必须覆盖
    ///   `((y0+r)·stride + x0 + offs[c]) .. + 8 个 u16`。
    /// - `dst` 覆盖 `((y0+h-1)·dst_stride + x0) .. + wu 个 u16`。
    /// - `adj` 必须 < 16（同 [`cdef_pair16`]）。
    /// [`cdef_pair16x4_impl`] 的**按"哪一档强度为 0"分派**（2026-10-25）。
    ///
    /// 抽头档的约束量是 `t = subs_epu16(thr, mag >> adj)`：`thr == 0` 时
    /// **恒为 0** ⇒ 这一档的 4（或 8）个候选只影响最后的 `min/max` 钳位，
    /// 整条 `sub → abs → sra → subs → min → sign → add` 链可以整段消掉。
    /// 本素材 **78% 的亮度块 `pri_str == 0`**（局部方差为 0 ⇒ `pri_str` 被清零），
    /// 而只要次档不为 0 整块仍要滤波 ⇒ 这些块白白跑了 4 个主抽头的约束链。
    ///
    /// 调用方保证两档不会同时为 0（那整块就不滤波），这里仍保守走完整版。
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn cdef_pair16x4(
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
        dst: *mut u16,
        dst_stride: isize,
    ) {
        macro_rules! call {
            ($($t:tt)*) => {
                cdef_pair16x4_impl::<$($t)*>(
                    data, stride, x0, y0, h, wu, offs, taps, pri_thr, pri_adj, sec_thr, sec_adj,
                    dst, dst_stride,
                )
            };
        }
        if crate::video::flags::no_cdef_str_spec() || (pri_thr != 0) == (sec_thr != 0) {
            call!(true, true)
        } else if pri_thr == 0 {
            call!(false, true)
        } else {
            call!(true, false)
        }
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn cdef_pair16x4_impl<const PRI: bool, const SEC: bool>(
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
        dst: *mut u16,
        dst_stride: isize,
    ) {
        debug_assert!(wu == 4 || wu == 8);
        debug_assert!(h % 4 == 0 && h >= 4);
        let base = data.as_ptr();
        let zero = _mm256_setzero_si256();
        let one = _mm256_set1_epi16(1);
        let eight = _mm256_set1_epi16(8);
        let pri_t = _mm256_set1_epi16(pri_thr as i16);
        let sec_t = _mm256_set1_epi16(sec_thr as i16);
        let pri_s = _mm_cvtsi32_si128(pri_adj as i32);
        let sec_s = _mm_cvtsi32_si128(sec_adj as i32);
        // 抽头归并后只剩 4 个不同值（主 c=0/2、次 c=4/8）⇒ 只广播这 4 个，
        // 不再建 12 项 YMM 数组（那会把 12×32 字节压到栈上）。
        let pri_taps = [
            _mm256_set1_epi16(taps[0] as i16),
            _mm256_set1_epi16(taps[2] as i16),
        ];
        let sec_taps = [
            _mm256_set1_epi16(taps[4] as i16),
            _mm256_set1_epi16(taps[8] as i16),
        ];
        let mut r = 0usize;
        while r < h {
            let off0 = (y0 + r as isize) * stride + x0;
            let off1 = off0 + stride;
            let off2 = off1 + stride;
            let off3 = off2 + stride;
            // SAFETY: 见函数文档的不变量。
            let cur_a = _mm256_set_m128i(
                _mm_loadu_si128(base.offset(off1) as *const __m128i),
                _mm_loadu_si128(base.offset(off0) as *const __m128i),
            );
            let cur_b = _mm256_set_m128i(
                _mm_loadu_si128(base.offset(off3) as *const __m128i),
                _mm_loadu_si128(base.offset(off2) as *const __m128i),
            );
            let mut acc_a = zero;
            let mut acc_b = zero;
            let mut mx_a = cur_a;
            let mut mx_b = cur_b;
            let mut mn_a = cur_a;
            let mut mn_b = cur_b;
            // ── **同抽头的候选先相加、再乘一次**（2026-10-24）────────────────────
            // `cdef_candidates` 的抽头是成组的：主抽头 c=0/1 同 tap、c=2/3 同 tap；
            // 次抽头 c=4..7 同 tap、c=8..11 同 tap。整数加乘在**模 2^16** 下构成环
            // ⇒ `v0·t + v1·t ≡ (v0+v1)·t`，逐位相同（内核本来就用 `_mm256_add_epi16`
            // 的回绕，见 `cdef_pair16` 的说明）。
            // 12 次乘 → 4 次；每组多 1 次相加。
            if PRI {
                for grp in 0..2usize {
                    let mut ga = zero;
                    let mut gb = zero;
                    for k in 0..2usize {
                        let c = grp * 2 + k;
                        let o = offs[c];
                        let cand_a = _mm256_set_m128i(
                            _mm_loadu_si128(base.offset(off1 + o) as *const __m128i),
                            _mm_loadu_si128(base.offset(off0 + o) as *const __m128i),
                        );
                        let cand_b = _mm256_set_m128i(
                            _mm_loadu_si128(base.offset(off3 + o) as *const __m128i),
                            _mm_loadu_si128(base.offset(off2 + o) as *const __m128i),
                        );
                        let diff_a = _mm256_sub_epi16(cand_a, cur_a);
                        let mag_a = _mm256_abs_epi16(diff_a);
                        let t_a = _mm256_subs_epu16(pri_t, _mm256_sra_epi16(mag_a, pri_s));
                        let t_a = _mm256_min_epu16(t_a, mag_a);
                        ga = _mm256_add_epi16(ga, _mm256_sign_epi16(t_a, diff_a));
                        mx_a = _mm256_max_epi16(mx_a, cand_a);
                        mn_a = _mm256_min_epi16(mn_a, cand_a);

                        let diff_b = _mm256_sub_epi16(cand_b, cur_b);
                        let mag_b = _mm256_abs_epi16(diff_b);
                        let t_b = _mm256_subs_epu16(pri_t, _mm256_sra_epi16(mag_b, pri_s));
                        let t_b = _mm256_min_epu16(t_b, mag_b);
                        gb = _mm256_add_epi16(gb, _mm256_sign_epi16(t_b, diff_b));
                        mx_b = _mm256_max_epi16(mx_b, cand_b);
                        mn_b = _mm256_min_epi16(mn_b, cand_b);
                    }
                    acc_a = _mm256_add_epi16(acc_a, _mm256_mullo_epi16(ga, pri_taps[grp]));
                    acc_b = _mm256_add_epi16(acc_b, _mm256_mullo_epi16(gb, pri_taps[grp]));
                }
            } else {
                // 主档强度为 0 ⇒ 约束量恒 0，只留钳位要用的 min/max。
                for c in 0..4usize {
                    let o = offs[c];
                    let cand_a = _mm256_set_m128i(
                        _mm_loadu_si128(base.offset(off1 + o) as *const __m128i),
                        _mm_loadu_si128(base.offset(off0 + o) as *const __m128i),
                    );
                    let cand_b = _mm256_set_m128i(
                        _mm_loadu_si128(base.offset(off3 + o) as *const __m128i),
                        _mm_loadu_si128(base.offset(off2 + o) as *const __m128i),
                    );
                    mx_a = _mm256_max_epi16(mx_a, cand_a);
                    mn_a = _mm256_min_epi16(mn_a, cand_a);
                    mx_b = _mm256_max_epi16(mx_b, cand_b);
                    mn_b = _mm256_min_epi16(mn_b, cand_b);
                }
            }
            if SEC {
                for grp in 0..2usize {
                    let mut ga = zero;
                    let mut gb = zero;
                    for k in 0..4usize {
                        let c = 4 + grp * 4 + k;
                        let o = offs[c];
                        let cand_a = _mm256_set_m128i(
                            _mm_loadu_si128(base.offset(off1 + o) as *const __m128i),
                            _mm_loadu_si128(base.offset(off0 + o) as *const __m128i),
                        );
                        let cand_b = _mm256_set_m128i(
                            _mm_loadu_si128(base.offset(off3 + o) as *const __m128i),
                            _mm_loadu_si128(base.offset(off2 + o) as *const __m128i),
                        );
                        let diff_a = _mm256_sub_epi16(cand_a, cur_a);
                        let mag_a = _mm256_abs_epi16(diff_a);
                        let t_a = _mm256_subs_epu16(sec_t, _mm256_sra_epi16(mag_a, sec_s));
                        let t_a = _mm256_min_epu16(t_a, mag_a);
                        ga = _mm256_add_epi16(ga, _mm256_sign_epi16(t_a, diff_a));
                        mx_a = _mm256_max_epi16(mx_a, cand_a);
                        mn_a = _mm256_min_epi16(mn_a, cand_a);

                        let diff_b = _mm256_sub_epi16(cand_b, cur_b);
                        let mag_b = _mm256_abs_epi16(diff_b);
                        let t_b = _mm256_subs_epu16(sec_t, _mm256_sra_epi16(mag_b, sec_s));
                        let t_b = _mm256_min_epu16(t_b, mag_b);
                        gb = _mm256_add_epi16(gb, _mm256_sign_epi16(t_b, diff_b));
                        mx_b = _mm256_max_epi16(mx_b, cand_b);
                        mn_b = _mm256_min_epi16(mn_b, cand_b);
                    }
                    acc_a = _mm256_add_epi16(acc_a, _mm256_mullo_epi16(ga, sec_taps[grp]));
                    acc_b = _mm256_add_epi16(acc_b, _mm256_mullo_epi16(gb, sec_taps[grp]));
                }
            } else {
                for c in 4..12usize {
                    let o = offs[c];
                    let cand_a = _mm256_set_m128i(
                        _mm_loadu_si128(base.offset(off1 + o) as *const __m128i),
                        _mm_loadu_si128(base.offset(off0 + o) as *const __m128i),
                    );
                    let cand_b = _mm256_set_m128i(
                        _mm_loadu_si128(base.offset(off3 + o) as *const __m128i),
                        _mm_loadu_si128(base.offset(off2 + o) as *const __m128i),
                    );
                    mx_a = _mm256_max_epi16(mx_a, cand_a);
                    mn_a = _mm256_min_epi16(mn_a, cand_a);
                    mx_b = _mm256_max_epi16(mx_b, cand_b);
                    mn_b = _mm256_min_epi16(mn_b, cand_b);
                }
            }
            let adj_a = _mm256_and_si256(_mm256_cmpgt_epi16(zero, acc_a), one);
            let va = _mm256_add_epi16(
                cur_a,
                _mm256_srai_epi16::<4>(_mm256_add_epi16(
                    _mm256_sub_epi16(eight, adj_a),
                    acc_a,
                )),
            );
            let va = _mm256_min_epi16(_mm256_max_epi16(va, mn_a), mx_a);
            let adj_b = _mm256_and_si256(_mm256_cmpgt_epi16(zero, acc_b), one);
            let vb = _mm256_add_epi16(
                cur_b,
                _mm256_srai_epi16::<4>(_mm256_add_epi16(
                    _mm256_sub_epi16(eight, adj_b),
                    acc_b,
                )),
            );
            let vb = _mm256_min_epi16(_mm256_max_epi16(vb, mn_b), mx_b);
            let mut out = [
                _mm256_castsi256_si128(va),
                _mm256_extracti128_si256::<1>(va),
                _mm256_castsi256_si128(vb),
                _mm256_extracti128_si256::<1>(vb),
            ];
            for (i, v) in out.iter_mut().enumerate() {
                let d = dst.offset((y0 + (r + i) as isize) * dst_stride + x0);
                if wu == 8 {
                    _mm_storeu_si128(d as *mut __m128i, *v);
                } else {
                    _mm_storel_epi64(d as *mut __m128i, *v);
                }
            }
            r += 4;
        }
    }

    /// CDEF **方向搜索**（§7.15.2）的 AVX2 版 —— partial 全程在寄存器里。
    ///
    /// ## 为什么换（2026-10-23）
    ///
    /// 旧实现把 8 个方向的 partial 放在 512 B 栈数组里，每行对 6 个方向做
    /// "读 32 B → 加 → 写 32 B"；相邻两行写的窗口**重叠 7 项** ⇒
    /// store-to-load forwarding 每次都失败（约 10 周期），一行 6 次。
    /// 实测方向搜索 ≈ 4 ms/帧，比亮度滤波内核还贵。
    ///
    /// ## 做法
    ///
    /// * 每个方向只开两个 XMM 累加器：`lo` = 位置 0..7、`hi` = 位置 8..15。
    ///   一行 8 个 i16 用两条 `pshufb`（常量掩码，按 `base` 查表）搬到
    ///   `[base, base+8)`，分别加进 lo/hi —— **纯寄存器**。
    /// * 代价用 `pmaddwd` 求**对称位置**的平方和（i16→i32），再 `vpmulld`
    ///   乘 Div 表、横加。对称位置权重相同（`Div[|k-7|+1]`），与标量的
    ///   `(p[k]² + p[14-k]²)·Div[k+1]` **逐位相同**。
    /// * 所有中间量是 i16：`|partial| ≤ 8·128 = 1024`；平方和扩到 i32。
    ///
    /// 方向 0/4 的 partial 是对角线（`p0[i+j] += s[i][j]`）、1/3/5/7 是
    /// "两步一折"（`j/2`/`i/2`），逐方向的 `(值向量, base)` 见下面的循环。
    ///
    /// # Safety
    /// 必须已确认 AVX2；`data` 覆盖 `(y0..y0+8)×(x0..x0+8)`。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn cdef_direction_avx2(
        data: &[u16],
        stride: usize,
        x0: usize,
        y0: usize,
        sh: u32,
    ) -> (usize, i32) {
        use std::arch::x86_64::*;

        // 一行 8 个 i16 的"搬到 [base, base+8)"掩码（`pshufb`，0x80 = 清零）。
        const fn shift_lo(base: usize) -> [u8; 16] {
            let mut m = [0x80u8; 16];
            let mut l = base;
            while l < 8 {
                m[2 * l] = ((l - base) * 2) as u8;
                m[2 * l + 1] = ((l - base) * 2 + 1) as u8;
                l += 1;
            }
            m
        }
        const fn shift_hi(base: usize) -> [u8; 16] {
            let mut m = [0x80u8; 16];
            let mut l = 0;
            while l < base {
                m[2 * l] = ((8 - base + l) * 2) as u8;
                m[2 * l + 1] = ((8 - base + l) * 2 + 1) as u8;
                l += 1;
            }
            m
        }
        const SHIFT_LO: [[u8; 16]; 8] = [
            shift_lo(0), shift_lo(1), shift_lo(2), shift_lo(3),
            shift_lo(4), shift_lo(5), shift_lo(6), shift_lo(7),
        ];
        const SHIFT_HI: [[u8; 16]; 8] = [
            shift_hi(0), shift_hi(1), shift_hi(2), shift_hi(3),
            shift_hi(4), shift_hi(5), shift_hi(6), shift_hi(7),
        ];
        // 反转 8 个 i16（方向 4 的镜像行）。
        const REV8: [u8; 16] = [14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1];
        // 反转 4 个 i16 并把高 4 格清零（方向 3 的 ry）。
        const REV4Z: [u8; 16] = [6, 7, 4, 5, 2, 3, 0, 1, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80];
        // `hi` 的重排：偶方向取 (p8..p14) 的对称项、奇方向取 (p8,p9,p10)。
        const RH_EVEN: [u8; 16] = [12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 0x80, 0x80];
        const RH_ODD: [u8; 16] = [4, 5, 2, 3, 0, 1, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80];

        #[inline(always)]
        unsafe fn add_row(
            v: __m128i,
            base: usize,
            lo: &mut __m128i,
            hi: &mut __m128i,
            shift_lo: &[[u8; 16]; 8],
            shift_hi: &[[u8; 16]; 8],
        ) {
            let ml = _mm_loadu_si128(shift_lo[base].as_ptr() as *const __m128i);
            let mh = _mm_loadu_si128(shift_hi[base].as_ptr() as *const __m128i);
            *lo = _mm_add_epi16(*lo, _mm_shuffle_epi8(v, ml));
            *hi = _mm_add_epi16(*hi, _mm_shuffle_epi8(v, mh));
        }

        // `Σ (lo[k]² + rh[k]²)·w[k]`（8 个 lane 一次 pmaddwd）。
        #[inline(always)]
        unsafe fn pair_cost(lo: __m128i, rh: __m128i, w: __m256i) -> i32 {
            let il = _mm_unpacklo_epi16(lo, rh);
            let ih = _mm_unpackhi_epi16(lo, rh);
            let v = _mm256_set_m128i(ih, il);
            let sq = _mm256_madd_epi16(v, v);
            let m = _mm256_mullo_epi32(sq, w);
            let hi = _mm256_extracti128_si256::<1>(m);
            let sum = _mm_add_epi32(_mm256_castsi256_si128(m), hi);
            let sum = _mm_hadd_epi32(sum, sum);
            let sum = _mm_hadd_epi32(sum, sum);
            _mm_cvtsi128_si32(sum)
        }

        let c128 = _mm_set1_epi16(128);
        let zero = _mm_setzero_si128();
        // 8 行 i16（`s = (sample >> sh) - 128`）。放栈上有两个好处：
        // 每个方向按需重载（L1 命中）、寄存器压力不随方向数增长。
        let mut rows = [[0i16; 8]; 8];
        for i in 0..8usize {
            let p = data.as_ptr().add((y0 + i) * stride + x0);
            let v = _mm_loadu_si128(p as *const __m128i);
            let v = match sh {
                0 => v,
                2 => _mm_srli_epi16::<2>(v),
                _ => _mm_srli_epi16::<4>(v),
            };
            let v = _mm_sub_epi16(v, c128);
            _mm_storeu_si128(rows[i].as_mut_ptr() as *mut __m128i, v);
        }
        let row = |i: usize| -> __m128i {
            _mm_loadu_si128(rows[i].as_ptr() as *const __m128i)
        };

        // ── 六个"折线"方向：各累加 2 个 XMM ────────────────────────────
        let (mut lo0, mut hi0) = (zero, zero);
        let (mut lo4, mut hi4) = (zero, zero);
        let (mut lo1, mut hi1) = (zero, zero);
        let (mut lo3, mut hi3) = (zero, zero);
        let (mut lo5, mut hi5) = (zero, zero);
        let (mut lo7, mut hi7) = (zero, zero);
        let rev8 = _mm_loadu_si128(REV8.as_ptr() as *const __m128i);
        let rev4z = _mm_loadu_si128(REV4Z.as_ptr() as *const __m128i);
        for i in 0..8usize {
            let r = row(i);
            // 0：`p0[i+j] += s[i][j]`
            add_row(r, i, &mut lo0, &mut hi0, &SHIFT_LO, &SHIFT_HI);
            // 4：镜像行 `p4[i+j] += s[i][7-j]`
            add_row(_mm_shuffle_epi8(r, rev8), i, &mut lo4, &mut hi4, &SHIFT_LO, &SHIFT_HI);
            // 1 / 3：`y[t] = s[2t]+s[2t+1]`（`phaddw`）
            let y = _mm_hadd_epi16(r, zero);
            add_row(y, i, &mut lo1, &mut hi1, &SHIFT_LO, &SHIFT_HI);
            add_row(_mm_shuffle_epi8(y, rev4z), i, &mut lo3, &mut hi3, &SHIFT_LO, &SHIFT_HI);
            // 5：`p5[3 - i/2 + j] += s[i][j]`
            add_row(r, 3 - i / 2, &mut lo5, &mut hi5, &SHIFT_LO, &SHIFT_HI);
            // 7：`p7[i/2 + j] += s[i][j]`
            add_row(r, i / 2, &mut lo7, &mut hi7, &SHIFT_LO, &SHIFT_HI);
        }
        // 6：列和（8 个 i16 lane）。
        let mut cols = zero;
        for i in 0..8usize {
            cols = _mm_add_epi16(cols, row(i));
        }
        // 2：行和（标量平方和 × Div[8]）。
        let ones = _mm_set1_epi16(1);
        let mut cost2 = 0i32;
        for i in 0..8usize {
            let s = _mm_madd_epi16(row(i), ones);
            let s = _mm_hadd_epi32(s, s);
            let s = _mm_hadd_epi32(s, s);
            let t = _mm_cvtsi128_si32(s);
            cost2 += t * t * 105;
        }

        let rh_even = _mm_loadu_si128(RH_EVEN.as_ptr() as *const __m128i);
        let rh_odd = _mm_loadu_si128(RH_ODD.as_ptr() as *const __m128i);
        let w_even = _mm256_setr_epi32(840, 420, 280, 210, 168, 140, 120, 105);
        let w_odd = _mm256_setr_epi32(420, 210, 140, 105, 105, 105, 105, 105);

        let cost0 = pair_cost(lo0, _mm_shuffle_epi8(hi0, rh_even), w_even);
        let cost4 = pair_cost(lo4, _mm_shuffle_epi8(hi4, rh_even), w_even);
        let cost1 = pair_cost(lo1, _mm_shuffle_epi8(hi1, rh_odd), w_odd);
        let cost3 = pair_cost(lo3, _mm_shuffle_epi8(hi3, rh_odd), w_odd);
        let cost5 = pair_cost(lo5, _mm_shuffle_epi8(hi5, rh_odd), w_odd);
        let cost7 = pair_cost(lo7, _mm_shuffle_epi8(hi7, rh_odd), w_odd);
        let sq = _mm_madd_epi16(cols, cols);
        let m = _mm_mullo_epi32(sq, _mm_set1_epi32(105));
        let m = _mm_hadd_epi32(m, m);
        let m = _mm_hadd_epi32(m, m);
        let cost6 = _mm_cvtsi128_si32(m);

        let costs = [cost0, cost1, cost2, cost3, cost4, cost5, cost6, cost7];
        let mut best_cost = 0i32;
        let mut y_dir = 0usize;
        for (i, &c) in costs.iter().enumerate() {
            if c > best_cost {
                best_cost = c;
                y_dir = i;
            }
        }
        (y_dir, (best_cost - costs[(y_dir + 4) & 7]) >> 10)
    }

    /// [`cdef_direction_avx2`] 的**两块版**：两个相邻的 8×8 块各占一个 128 位半区。
    ///
    /// 方向搜索是"逐行 `pshufb` + 加"；`vpshufb` 本来就**按 128 位半区独立**工作
    /// ⇒ 两个块的同一行用同一套掩码一次算完，**指令数减半**（每半区一条通道）。
    /// 掩码、`rev8`/`rev4z`、`rh_*` 都是 16 字节，广播到两个半区即可。
    ///
    /// # Safety
    /// 已确认 AVX2；两个块的 8×8 样本都在 `data` 内（`stride` 覆盖）。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn cdef_direction_avx2_x2(
        data: &[u16],
        stride: usize,
        x0a: usize,
        y0a: usize,
        x0b: usize,
        y0b: usize,
        sh: u32,
    ) -> ((usize, i32), (usize, i32)) {
        use std::arch::x86_64::*;

        const fn shift_lo(base: usize) -> [u8; 16] {
            let mut m = [0x80u8; 16];
            let mut l = base;
            while l < 8 {
                m[2 * l] = ((l - base) * 2) as u8;
                m[2 * l + 1] = ((l - base) * 2 + 1) as u8;
                l += 1;
            }
            m
        }
        const fn shift_hi(base: usize) -> [u8; 16] {
            let mut m = [0x80u8; 16];
            let mut l = 0;
            while l < base {
                m[2 * l] = ((8 - base + l) * 2) as u8;
                m[2 * l + 1] = ((8 - base + l) * 2 + 1) as u8;
                l += 1;
            }
            m
        }
        const SHIFT_LO: [[u8; 16]; 8] = [
            shift_lo(0), shift_lo(1), shift_lo(2), shift_lo(3),
            shift_lo(4), shift_lo(5), shift_lo(6), shift_lo(7),
        ];
        const SHIFT_HI: [[u8; 16]; 8] = [
            shift_hi(0), shift_hi(1), shift_hi(2), shift_hi(3),
            shift_hi(4), shift_hi(5), shift_hi(6), shift_hi(7),
        ];
        const REV8: [u8; 16] = [14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1];
        const REV4Z: [u8; 16] = [6, 7, 4, 5, 2, 3, 0, 1, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80];
        const RH_EVEN: [u8; 16] = [12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 0x80, 0x80];
        const RH_ODD: [u8; 16] = [4, 5, 2, 3, 0, 1, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80];

        // 掩码广播到两个 128 位半区（`vpshufb` 按半区独立）
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn bcast(m: &[u8; 16]) -> __m256i {
            _mm256_broadcastsi128_si256(_mm_loadu_si128(m.as_ptr() as *const __m128i))
        }
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn add_row(
            v: __m256i,
            base: usize,
            lo: &mut __m256i,
            hi: &mut __m256i,
            clo: &[[u8; 16]; 8],
            chi: &[[u8; 16]; 8],
        ) {
            *lo = _mm256_add_epi16(*lo, _mm256_shuffle_epi8(v, bcast(&clo[base])));
            *hi = _mm256_add_epi16(*hi, _mm256_shuffle_epi8(v, bcast(&chi[base])));
        }
        /// 每半区各自的 `Σ (lo[k]² + rh[k]²)·w[k]`
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn pair_cost2(lo: __m256i, rh: __m256i, wlo: __m256i, whi: __m256i) -> [i32; 2] {
            let il = _mm256_unpacklo_epi16(lo, rh);
            let ih = _mm256_unpackhi_epi16(lo, rh);
            let sl = _mm256_madd_epi16(il, il);
            let sh2 = _mm256_madd_epi16(ih, ih);
            let m = _mm256_add_epi32(
                _mm256_mullo_epi32(sl, wlo),
                _mm256_mullo_epi32(sh2, whi),
            );
            let m = _mm256_hadd_epi32(m, m);
            let m = _mm256_hadd_epi32(m, m);
            [
                _mm_cvtsi128_si32(_mm256_castsi256_si128(m)),
                _mm_cvtsi128_si32(_mm256_extracti128_si256::<1>(m)),
            ]
        }

        let c128 = _mm256_set1_epi16(128);
        let zero = _mm256_setzero_si256();
        let mut rows = [[0i16; 16]; 8];
        for i in 0..8usize {
            let pa = data.as_ptr().add((y0a + i) * stride + x0a);
            let pb = data.as_ptr().add((y0b + i) * stride + x0b);
            let v = _mm256_set_m128i(
                _mm_loadu_si128(pb as *const __m128i),
                _mm_loadu_si128(pa as *const __m128i),
            );
            let v = match sh {
                0 => v,
                2 => _mm256_srli_epi16::<2>(v),
                _ => _mm256_srli_epi16::<4>(v),
            };
            let v = _mm256_sub_epi16(v, c128);
            _mm256_storeu_si256(rows[i].as_mut_ptr() as *mut __m256i, v);
        }
        let row = |i: usize| -> __m256i {
            _mm256_loadu_si256(rows[i].as_ptr() as *const __m256i)
        };

        let (mut lo0, mut hi0) = (zero, zero);
        let (mut lo4, mut hi4) = (zero, zero);
        let (mut lo1, mut hi1) = (zero, zero);
        let (mut lo3, mut hi3) = (zero, zero);
        let (mut lo5, mut hi5) = (zero, zero);
        let (mut lo7, mut hi7) = (zero, zero);
        let rev8 = bcast(&REV8);
        let rev4z = bcast(&REV4Z);
        for i in 0..8usize {
            let r = row(i);
            add_row(r, i, &mut lo0, &mut hi0, &SHIFT_LO, &SHIFT_HI);
            add_row(
                _mm256_shuffle_epi8(r, rev8),
                i,
                &mut lo4,
                &mut hi4,
                &SHIFT_LO,
                &SHIFT_HI,
            );
            let y = _mm256_hadd_epi16(r, zero);
            add_row(y, i, &mut lo1, &mut hi1, &SHIFT_LO, &SHIFT_HI);
            add_row(
                _mm256_shuffle_epi8(y, rev4z),
                i,
                &mut lo3,
                &mut hi3,
                &SHIFT_LO,
                &SHIFT_HI,
            );
            add_row(r, 3 - i / 2, &mut lo5, &mut hi5, &SHIFT_LO, &SHIFT_HI);
            add_row(r, i / 2, &mut lo7, &mut hi7, &SHIFT_LO, &SHIFT_HI);
        }
        let mut cols = zero;
        for i in 0..8usize {
            cols = _mm256_add_epi16(cols, row(i));
        }
        let ones = _mm256_set1_epi16(1);
        let mut cost2 = [0i32; 2];
        for i in 0..8usize {
            let s = _mm256_madd_epi16(row(i), ones);
            let s = _mm256_hadd_epi32(s, s);
            let s = _mm256_hadd_epi32(s, s);
            let a = _mm_cvtsi128_si32(_mm256_castsi256_si128(s));
            let b = _mm_cvtsi128_si32(_mm256_extracti128_si256::<1>(s));
            cost2[0] += a * a * 105;
            cost2[1] += b * b * 105;
        }

        let rh_even = bcast(&RH_EVEN);
        let rh_odd = bcast(&RH_ODD);
        let w_lo = _mm256_setr_epi32(840, 420, 280, 210, 840, 420, 280, 210);
        let w_hi = _mm256_setr_epi32(168, 140, 120, 105, 168, 140, 120, 105);
        let w_lo_odd = _mm256_setr_epi32(420, 210, 140, 105, 420, 210, 140, 105);
        let w_hi_odd = _mm256_setr_epi32(105, 105, 105, 105, 105, 105, 105, 105);

        let c0 = pair_cost2(lo0, _mm256_shuffle_epi8(hi0, rh_even), w_lo, w_hi);
        let c4 = pair_cost2(lo4, _mm256_shuffle_epi8(hi4, rh_even), w_lo, w_hi);
        let c1 = pair_cost2(lo1, _mm256_shuffle_epi8(hi1, rh_odd), w_lo_odd, w_hi_odd);
        let c3 = pair_cost2(lo3, _mm256_shuffle_epi8(hi3, rh_odd), w_lo_odd, w_hi_odd);
        let c5 = pair_cost2(lo5, _mm256_shuffle_epi8(hi5, rh_odd), w_lo_odd, w_hi_odd);
        let c7 = pair_cost2(lo7, _mm256_shuffle_epi8(hi7, rh_odd), w_lo_odd, w_hi_odd);
        let sq = _mm256_madd_epi16(cols, cols);
        let m = _mm256_mullo_epi32(sq, _mm256_set1_epi32(105));
        let m = _mm256_hadd_epi32(m, m);
        let m = _mm256_hadd_epi32(m, m);
        let c6 = [
            _mm_cvtsi128_si32(_mm256_castsi256_si128(m)),
            _mm_cvtsi128_si32(_mm256_extracti128_si256::<1>(m)),
        ];

        let mut out = [(0usize, 0i32); 2];
        for half in 0..2 {
            let costs = [
                c0[half], c1[half], cost2[half], c3[half], c4[half], c5[half], c6[half], c7[half],
            ];
            let mut best_cost = 0i32;
            let mut y_dir = 0usize;
            for (i, &c) in costs.iter().enumerate() {
                if c > best_cost {
                    best_cost = c;
                    y_dir = i;
                }
            }
            out[half] = (y_dir, (best_cost - costs[(y_dir + 4) & 7]) >> 10);
        }
        (out[0], out[1])
    }

    /// **CDEF**（§7.15.3）一次滤完**四行 × 四列**（16 个 i16 通道，**全部用满**）。
    ///
    /// ## 为什么色度要单独一条（2026-10-05 第三十五轮）
    ///
    /// [`cdef_pair16`] 一次算 **2 行 × 8 列**。亮度块正好 8×8 ⇒ 16 个通道
    /// **全部用满**；而 4:2:0 的**色度块只有 4×4**（`w = h = 4`）⇒ 那 16 个通道里
    /// **只有 8 个有用**（另外 8 个读到的是右边的样本、算完直接丢弃）——
    /// **一半的算力白花**，而色度恰好占 CDEF 内核时间的一半
    /// （每平面 `h/2 = 2` 次迭代 × 2 个平面 = 与亮度同样的 4 次）。
    ///
    /// 本函数把 **4 行 × 4 列**塞进同一个 256 位寄存器（4 个 qword，每个 qword 是
    /// 一行的 4 个 u16）⇒ 16 个通道**全部有用**，迭代次数从 `h/2` 降到 `h/4`。
    /// 每个候选的取址从"两次 128 位载入 + 拼接"变成"四次 64 位载入 + 拼接"，
    /// **指令条数不变、每次迭代多产出一倍**。
    ///
    /// 逐条与 [`cdef_pair16`] 相同（同一个 `constrain` 展开、同一个 `Clip3`），
    /// 判据是 `cdef_quad8_matches_cdef_pair16` 对 8 个方向 × 随机数据逐像素对拍。
    ///
    /// # Safety
    /// - 必须已确认 AVX2 可用。
    /// - `h` 必须是 **4**（本函数不算尾行）。
    /// - 对每个 `c ∈ [0, 12)`、`j ∈ [0, 4)`，
    ///   `data` 必须覆盖 `((y0+j)·stride + x0 + offs[c]) .. + 4 个 u16`
    ///   （`cdef_pair16` 要求同样的地址上可读 **8** 个 ⇒ 本要求**更弱**）。
    /// - `dst` 覆盖 `((y0+3)·dst_stride + x0) .. + 4 个 u16`。
    /// - `adj` 必须 < 16（同 `cdef_pair16`）。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn cdef_quad8_uv(
        du: &[u16],
        dv: &[u16],
        stride: isize,
        x0: isize,
        y0: isize,
        offs: &[isize; super::CDEF_CANDS],
        taps: &[i32; super::CDEF_CANDS],
        pri_thr: i32,
        pri_adj: u32,
        sec_thr: i32,
        sec_adj: u32,
        dst_u: *mut u16,
        dst_v: *mut u16,
        dst_stride: isize,
    ) {
        debug_assert!(super::CDEF_CANDS == 12, "本内核按 12 个候选写死");
        // 2026-10-25：**U/V 合并成一次调用**。原来每帧 ~6.5 万次 `cdef_quad8`，
        // 每次都要付一遍函数序言 + 8 条广播；两个平面的参数完全相同，合成一次
        // 就只付一遍（体在下面循环里跑两趟）。
        for (data, dst) in [(du, dst_u), (dv, dst_v)] {
        let base = data.as_ptr();
        let zero = _mm256_setzero_si256();
        let one = _mm256_set1_epi16(1);
        let eight = _mm256_set1_epi16(8);
        let pri_t = _mm256_set1_epi16(pri_thr as i16);
        let sec_t = _mm256_set1_epi16(sec_thr as i16);
        let pri_s = _mm_cvtsi32_si128(pri_adj as i32);
        let sec_s = _mm_cvtsi32_si128(sec_adj as i32);
        let pri_taps = [
            _mm256_set1_epi16(taps[0] as i16),
            _mm256_set1_epi16(taps[2] as i16),
        ];
        let sec_taps = [
            _mm256_set1_epi16(taps[4] as i16),
            _mm256_set1_epi16(taps[8] as i16),
        ];
        // 把 4 行（每行 4 个 u16）拼成一个 256 位向量：qword `j` = 行 `j`。
        // `lin` 是**相对块左上角**的线性偏移（候选位移就在这个尺度上），
        // 所以基准地址要带上 `y0·stride` —— 漏掉它就读成第 0..3 行了
        // （第一版就这样错的，被 `cdef_quad8_matches_cdef_pair16_bit_for_bit` 抓住）。
        let gather = |lin: isize| -> __m256i {
            let o = x0 + y0 * stride + lin;
            _mm256_set_epi64x(
                _mm_cvtsi128_si64(_mm_loadl_epi64(
                    base.offset(o + 3 * stride) as *const __m128i
                )),
                _mm_cvtsi128_si64(_mm_loadl_epi64(
                    base.offset(o + 2 * stride) as *const __m128i
                )),
                _mm_cvtsi128_si64(_mm_loadl_epi64(
                    base.offset(o + stride) as *const __m128i
                )),
                _mm_cvtsi128_si64(_mm_loadl_epi64(base.offset(o) as *const __m128i)),
            )
        };
        let cur = gather(0);
        let mut acc = zero;
        let mut mx = cur;
        let mut mn = cur;
        // 同抽头的候选先相加、再乘一次（与 `cdef_pair16x4` 同一套推导：模 2^16 环上
        // `v0·t + v1·t ≡ (v0+v1)·t`，逐位相同）。12 次乘 → 4 次。
        for grp in 0..2usize {
            let mut g = zero;
            for k in 0..2usize {
                let c = grp * 2 + k;
                let cand = gather(offs[c]);
                let diff = _mm256_sub_epi16(cand, cur);
                let mag = _mm256_abs_epi16(diff);
                let t = _mm256_subs_epu16(pri_t, _mm256_sra_epi16(mag, pri_s));
                let t = _mm256_min_epu16(t, mag);
                g = _mm256_add_epi16(g, _mm256_sign_epi16(t, diff));
                mx = _mm256_max_epi16(mx, cand);
                mn = _mm256_min_epi16(mn, cand);
            }
            acc = _mm256_add_epi16(acc, _mm256_mullo_epi16(g, pri_taps[grp]));
        }
        for grp in 0..2usize {
            let mut g = zero;
            for k in 0..4usize {
                let c = 4 + grp * 4 + k;
                let cand = gather(offs[c]);
                let diff = _mm256_sub_epi16(cand, cur);
                let mag = _mm256_abs_epi16(diff);
                let t = _mm256_subs_epu16(sec_t, _mm256_sra_epi16(mag, sec_s));
                let t = _mm256_min_epu16(t, mag);
                g = _mm256_add_epi16(g, _mm256_sign_epi16(t, diff));
                mx = _mm256_max_epi16(mx, cand);
                mn = _mm256_min_epi16(mn, cand);
            }
            acc = _mm256_add_epi16(acc, _mm256_mullo_epi16(g, sec_taps[grp]));
        }
        let adjust = _mm256_and_si256(_mm256_cmpgt_epi16(zero, acc), one);
        let v = _mm256_add_epi16(
            cur,
            _mm256_srai_epi16::<4>(_mm256_add_epi16(_mm256_sub_epi16(eight, adjust), acc)),
        );
        let v = _mm256_min_epi16(_mm256_max_epi16(v, mn), mx);
        // 写回：4 行 × 4 个 u16，每个 qword 一行。
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256::<1>(v);
        let q = [lo, _mm_unpackhi_epi64(lo, lo), hi, _mm_unpackhi_epi64(hi, hi)];
        for (j, qq) in q.iter().enumerate() {
            _mm_storel_epi64(
                dst.offset((y0 + j as isize) * dst_stride + x0) as *mut __m128i,
                *qq,
            );
        }
        }
    }

    /// 单平面版（自测/兼容用）：把同一份数据跑两遍（第二遍写回同一 dst，结果不变）。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn cdef_quad8(
        data: &[u16],
        stride: isize,
        x0: isize,
        y0: isize,
        offs: &[isize; super::CDEF_CANDS],
        taps: &[i32; super::CDEF_CANDS],
        pri_thr: i32,
        pri_adj: u32,
        sec_thr: i32,
        sec_adj: u32,
        dst: *mut u16,
        dst_stride: isize,
    ) {
        cdef_quad8_uv(
            data, data, stride, x0, y0, offs, taps, pri_thr, pri_adj, sec_thr, sec_adj, dst, dst,
            dst_stride,
        )
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

    /// `t'[k]`：把 `k` 钳到 `[-(N+1), N]` 再取 `V[k]`（与旧的 `(i+j).clamp(…)` 逐位相同）。
    ///
    /// 全部调用点的 `k` 都是**编译期常量**（`N`/`N2` 是 const 泛型、外层循环会展开）
    /// ⇒ 钳位与下标都折掉，只剩一次寄存器选择。
    #[inline(always)]
    fn dbl_at<const N: i32>(v: &[__m128i; 16], k: i32) -> __m128i {
        v[vi(k.clamp(-(N + 1), N))]
    }

    /// 宽滤波器（§7.14.6.4）的通道版：`N` = `n`、`N2` = `n2`、`L2` = `log2Size`。
    ///
    /// 返回 `f[i]`（`i = -N..N` 按序），对应列 `xp + i`。**与旧写法逐位相同**，
    /// 但每档从 `2N+1` 次向量加降到 **4 次**（见下面的精确递推）。
    ///
    /// ## 旧写法：`2N+1` 项直接求和，输出之间重复算了 `2N-3` 项
    ///
    /// ```text
    ///   A(i) = Σ_{j=-N..N} w(j) · t'[i+j],   w(j) = (|j| ≤ N2) ? 2 : 1
    ///   f[i] = Round2( A(i), L2 )
    /// ```
    ///
    /// 16 抽头（`N = 6`）时每个输出 13 项 ⇒ 12 个输出 **156 次 `paddd`**——
    /// 这是去块内核里最贵的一段。
    ///
    /// ## 精确递推：**4 条指令**产出下一个输出
    ///
    /// 把 `A(i+1) - A(i)` 按 `k = j+1` 重排：
    ///
    /// ```text
    ///   A(i+1) = Σ_{k=-N+1..N+1} w(k-1)·t'[i+k]
    ///   A(i+1) - A(i) = w(N)·t'[i+N+1] - w(-N)·t'[i-N]
    ///                   + Σ_{k=-N+1..N} ( w(k-1) - w(k) )·t'[i+k]
    /// ```
    ///
    /// * `w` **偶对称**（`w(j) = w(-j)`，因为判据是 `|j|`）⇒ 两个边界项合成
    ///   `t'[i+N+1] - t'[i-N]`（`w(N) = w(-N) = 1`，只要 `N2 < N`——三个实例都满足）；
    /// * `w(k-1) - w(k)` 只在 `|·| ≤ N2` 这个**区间指示函数**的两端非零：
    ///   `k = -N2` 处由 2→1（**−1**）、`k = N2+1` 处由 1→2（**+1**）。
    ///
    /// ⇒ **`A(i+1) = A(i) + t'[i+N+1] - t'[i-N] + t'[i+N2+1] - t'[i-N2]`**
    ///
    /// **整数加法可结合、可交换 ⇒ 递推结果与直接求和逐位相同**（不是"近似"）。
    /// 前 8 抽头（`N = 3, N2 = 0`）与色度 8 抽头（`N = 2, N2 = 1`）用同一个式子。
    ///
    /// 代价对比（`N = 6`）：旧 13×12 = 156 次加 + 6 次 `slli`；
    /// 新 13（首项）+ 11×4 = 57 次加 + 6 次 `slli`。
    ///
    /// 判据：[`tests::dbl_wide_recurrence_matches_the_direct_sum`] 对三个
    /// `(N, N2, L2)` 实例 × 逐样本极值/随机数据对**旧的直接求和**逐位对拍。
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(crate) unsafe fn dbl_wide<const N: i32, const N2: i32, const L2: i32>(
        v: &[__m128i; 16],
    ) -> [__m128i; 12] {
        let half = _mm_set1_epi32(1 << (L2 - 1));
        // 首项 `A(-N)`：`2N+1` 项直接求和（每项权重 1 或 2）
        let mut a = _mm_setzero_si128();
        let mut j = -N;
        while j <= N {
            let s = dbl_at::<N>(v, j - N);
            a = _mm_add_epi32(a, if j.abs() <= N2 { _mm_slli_epi32::<1>(s) } else { s });
            j += 1;
        }
        let mut f = [_mm_setzero_si128(); 12];
        let mut i = -N;
        f[0] = _mm_srai_epi32::<L2>(_mm_add_epi32(a, half));
        while i < N - 1 {
            a = _mm_add_epi32(a, dbl_at::<N>(v, i + N + 1));
            a = _mm_sub_epi32(a, dbl_at::<N>(v, i - N));
            a = _mm_add_epi32(a, dbl_at::<N>(v, i + N2 + 1));
            a = _mm_sub_epi32(a, dbl_at::<N>(v, i - N2));
            i += 1;
            f[(i + N) as usize] = _mm_srai_epi32::<L2>(_mm_add_epi32(a, half));
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

    /// 8 路版辅助：`|a-b|`（i32 通道）。
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dbl8_adiff(a: __m256i, b: __m256i) -> __m256i {
        _mm256_abs_epi32(_mm256_sub_epi32(a, b))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dbl8_clamp(v: __m256i, bit_depth: i32) -> __m256i {
        let lo = _mm256_set1_epi32(-(1 << (bit_depth - 1)));
        let hi = _mm256_set1_epi32((1 << (bit_depth - 1)) - 1);
        _mm256_min_epi32(_mm256_max_epi32(v, lo), hi)
    }

    /// 窄滤波器（8 路）：与 [`super::dbl_narrow`] 逐式相同，只是通道数翻倍。
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dbl8_narrow(v: &[__m256i; 16], hev: __m256i, bit_depth: i32) -> [__m256i; 4] {
        let off = _mm256_set1_epi32(0x80 << (bit_depth - 8));
        let (q0, q1, p0, p1) = (v[vi(0)], v[vi(1)], v[vi(-1)], v[vi(-2)]);
        let ps1 = _mm256_sub_epi32(p1, off);
        let ps0 = _mm256_sub_epi32(p0, off);
        let qs0 = _mm256_sub_epi32(q0, off);
        let qs1 = _mm256_sub_epi32(q1, off);
        let f0 = _mm256_and_si256(hev, dbl8_clamp(_mm256_sub_epi32(ps1, qs1), bit_depth));
        let d3 = {
            let d = _mm256_sub_epi32(qs0, ps0);
            _mm256_add_epi32(d, _mm256_slli_epi32::<1>(d))
        };
        let filter = dbl8_clamp(_mm256_add_epi32(f0, d3), bit_depth);
        let filter1 = _mm256_srai_epi32::<3>(dbl8_clamp(
            _mm256_add_epi32(filter, _mm256_set1_epi32(4)),
            bit_depth,
        ));
        let filter2 = _mm256_srai_epi32::<3>(dbl8_clamp(
            _mm256_add_epi32(filter, _mm256_set1_epi32(3)),
            bit_depth,
        ));
        let oq0 = _mm256_add_epi32(dbl8_clamp(_mm256_sub_epi32(qs0, filter1), bit_depth), off);
        let op0 = _mm256_add_epi32(dbl8_clamp(_mm256_add_epi32(ps0, filter2), bit_depth), off);
        let fh = _mm256_srai_epi32::<1>(_mm256_add_epi32(filter1, _mm256_set1_epi32(1)));
        let oq1 = _mm256_add_epi32(dbl8_clamp(_mm256_sub_epi32(qs1, fh), bit_depth), off);
        let op1 = _mm256_add_epi32(dbl8_clamp(_mm256_add_epi32(ps1, fh), bit_depth), off);
        [
            _mm256_blendv_epi8(op1, p1, hev),
            op0,
            oq0,
            _mm256_blendv_epi8(oq1, q1, hev),
        ]
    }

    #[inline(always)]
    fn dbl8_at<const N: i32>(v: &[__m256i; 16], k: i32) -> __m256i {
        v[vi(k.clamp(-(N + 1), N))]
    }

    /// 宽滤波器（8 路）：与 [`super::dbl_wide`] 同一递推、同一判据。
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dbl8_wide<const N: i32, const N2: i32, const L2: i32>(
        v: &[__m256i; 16],
    ) -> [__m256i; 12] {
        let half = _mm256_set1_epi32(1 << (L2 - 1));
        let mut a = _mm256_setzero_si256();
        let mut j = -N;
        while j <= N {
            let s = dbl8_at::<N>(v, j - N);
            a = _mm256_add_epi32(a, if j.abs() <= N2 { _mm256_slli_epi32::<1>(s) } else { s });
            j += 1;
        }
        let mut f = [_mm256_setzero_si256(); 12];
        let mut i = -N;
        f[0] = _mm256_srai_epi32::<L2>(_mm256_add_epi32(a, half));
        while i < N - 1 {
            a = _mm256_add_epi32(a, dbl8_at::<N>(v, i + N + 1));
            a = _mm256_sub_epi32(a, dbl8_at::<N>(v, i - N));
            a = _mm256_add_epi32(a, dbl8_at::<N>(v, i + N2 + 1));
            a = _mm256_sub_epi32(a, dbl8_at::<N>(v, i - N2));
            i += 1;
            f[(i + N) as usize] = _mm256_srai_epi32::<L2>(_mm256_add_epi32(a, half));
        }
        f
    }

    /// 8 路去块的**掩码 + 滤波器 + 融合**（横/竖两条边共用）：
    /// 入参是已经按"通道 = 沿边方向的 8 个像素"排好的 16 个窗口向量，
    /// 返回 `(out, n)`：`out[i]` = 列 `xp-n+i`（横边则是行 `yp-n+i`）。
    #[inline]
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn dbl8_core(
        v: &[__m256i; 16],
        plane: u8,
        filter_size: i32,
        bd_shift: i32,
        bit_depth: i32,
        limit: i32,
        blimit: i32,
        thresh: i32,
    ) -> Option<([__m256i; 12], usize)> {
        let zero = _mm256_setzero_si256();
        let (p1, p0, q0, q1) = (v[vi(-2)], v[vi(-1)], v[vi(0)], v[vi(1)]);
        let thresh_v = _mm256_set1_epi32(thresh << bd_shift);
        let hev = _mm256_or_si256(
            _mm256_cmpgt_epi32(dbl8_adiff(p1, p0), thresh_v),
            _mm256_cmpgt_epi32(dbl8_adiff(q1, q0), thresh_v),
        );
        let limit_v = _mm256_set1_epi32(limit << bd_shift);
        let blimit_v = _mm256_set1_epi32(blimit << bd_shift);
        let mut m = _mm256_or_si256(
            _mm256_cmpgt_epi32(dbl8_adiff(p1, p0), limit_v),
            _mm256_cmpgt_epi32(dbl8_adiff(q1, q0), limit_v),
        );
        let t = _mm256_add_epi32(
            _mm256_slli_epi32::<1>(dbl8_adiff(p0, q0)),
            _mm256_srai_epi32::<1>(dbl8_adiff(p1, q1)),
        );
        m = _mm256_or_si256(m, _mm256_cmpgt_epi32(t, blimit_v));
        if filter_size != 4 {
            let (p2, q2) = (v[vi(-3)], v[vi(2)]);
            m = _mm256_or_si256(m, _mm256_cmpgt_epi32(dbl8_adiff(p2, p1), limit_v));
            m = _mm256_or_si256(m, _mm256_cmpgt_epi32(dbl8_adiff(q2, q1), limit_v));
            if plane == 0 {
                let (p3, q3) = (v[vi(-4)], v[vi(3)]);
                m = _mm256_or_si256(m, _mm256_cmpgt_epi32(dbl8_adiff(p3, p2), limit_v));
                m = _mm256_or_si256(m, _mm256_cmpgt_epi32(dbl8_adiff(q3, q2), limit_v));
            }
        }
        let pass_mask = _mm256_cmpeq_epi32(m, zero);
        if _mm256_movemask_epi8(pass_mask) == 0 {
            return None;
        }
        let mut flat = zero;
        let mut flat2 = zero;
        if filter_size >= 8 {
            let th = _mm256_set1_epi32(1 << bd_shift);
            let (p2, q2) = (v[vi(-3)], v[vi(2)]);
            flat = _mm256_or_si256(
                _mm256_cmpgt_epi32(dbl8_adiff(p1, p0), th),
                _mm256_cmpgt_epi32(dbl8_adiff(q1, q0), th),
            );
            flat = _mm256_or_si256(flat, _mm256_cmpgt_epi32(dbl8_adiff(p2, p0), th));
            flat = _mm256_or_si256(flat, _mm256_cmpgt_epi32(dbl8_adiff(q2, q0), th));
            if plane == 0 {
                let (p3, q3) = (v[vi(-4)], v[vi(3)]);
                flat = _mm256_or_si256(flat, _mm256_cmpgt_epi32(dbl8_adiff(p3, p0), th));
                flat = _mm256_or_si256(flat, _mm256_cmpgt_epi32(dbl8_adiff(q3, q0), th));
            }
        }
        if filter_size >= 16 {
            let th = _mm256_set1_epi32(1 << bd_shift);
            for k in [-7i32, -6, -5] {
                flat2 = _mm256_or_si256(flat2, _mm256_cmpgt_epi32(dbl8_adiff(v[vi(k)], p0), th));
            }
            for k in [4i32, 5, 6] {
                flat2 = _mm256_or_si256(flat2, _mm256_cmpgt_epi32(dbl8_adiff(v[vi(k)], q0), th));
            }
        }
        let flat_mask = _mm256_cmpeq_epi32(flat, zero);
        let flat2_mask = _mm256_cmpeq_epi32(flat2, zero);
        let wide_lane = if filter_size >= 8 { flat_mask } else { zero };
        let wide16_lane = if filter_size >= 16 {
            _mm256_and_si256(flat_mask, flat2_mask)
        } else {
            zero
        };
        let wide8_lane = _mm256_andnot_si256(wide16_lane, wide_lane);
        let narrow_lane = _mm256_andnot_si256(wide_lane, pass_mask);
        let w8_lane = _mm256_and_si256(wide8_lane, pass_mask);
        let w16_lane = _mm256_and_si256(wide16_lane, pass_mask);
        let n = if filter_size == 4 {
            2usize
        } else if filter_size >= 16 {
            6
        } else if plane == 0 {
            3
        } else {
            2
        };
        let nv = if _mm256_movemask_epi8(narrow_lane) != 0 {
            dbl8_narrow(&v, hev, bit_depth)
        } else {
            [zero; 4]
        };
        let nw8 = if plane == 0 { 3i32 } else { 2 };
        let w8 = if _mm256_movemask_epi8(w8_lane) != 0 {
            if plane == 0 {
                dbl8_wide::<3, 0, 3>(&v)
            } else {
                dbl8_wide::<2, 1, 3>(&v)
            }
        } else {
            [zero; 12]
        };
        let w16 = if _mm256_movemask_epi8(w16_lane) != 0 {
            dbl8_wide::<6, 1, 4>(&v)
        } else {
            [zero; 12]
        };
        let mut out = [zero; 12];
        for (i, k) in (-(n as i32)..n as i32).enumerate() {
            out[i] = v[vi(k)];
        }
        if _mm256_movemask_epi8(narrow_lane) != 0 {
            for k in -2i32..2 {
                let i = (k + n as i32) as usize;
                out[i] = _mm256_blendv_epi8(out[i], nv[(k + 2) as usize], narrow_lane);
            }
        }
        if _mm256_movemask_epi8(w8_lane) != 0 {
            for k in -nw8..nw8 {
                let i = (k + n as i32) as usize;
                out[i] = _mm256_blendv_epi8(out[i], w8[(k + nw8) as usize], w8_lane);
            }
        }
        if _mm256_movemask_epi8(w16_lane) != 0 {
            for k in -6i32..6 {
                let i = (k + n as i32) as usize;
                out[i] = _mm256_blendv_epi8(out[i], w16[(k + 6) as usize], w16_lane);
            }
        }
        Some((out, n))
    }

    /// **横边**的 8 路去块内核：一条横边一次滤 **8 列**（相邻两个 4 像素段）。
    ///
    /// 与两次 [`deblock_edge4`]（`xp`、`xp+4`）逐字节相同：横边按**列**独立，
    /// 两段之间没有读写依赖。竖边要 8×8 转置，暂不走这条。
    ///
    /// # Safety
    /// - 已确认 AVX2；窗口 `[yp-need, yp+need)` × `[xp, xp+8)` 全在帧内
    ///   （调用方保证 `xp+8 <= w`、`yp >= need`、`yp+need-1 < h`）。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn deblock_edge8_h(
        data: *mut u16,
        stride: isize,
        bit_depth: u8,
        xp: i32,
        yp: i32,
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

        let mut v = [_mm256_setzero_si256(); 16];
        for k in -need..need {
            let row = data.offset((y + k as isize) * stride + x);
            v[vi(k)] = _mm256_cvtepu16_epi32(_mm_loadu_si128(row as *const __m128i));
        }

        let Some((out, n)) = dbl8_core(
            &v, plane, filter_size, bd_shift, bit_depth, limit, blimit, thresh,
        ) else {
            return;
        };
        // 写回（横边）：`out[i]` = 行 `y-n+i` 的 8 个 u16。两行一次打包。
        let yn = y - n as isize;
        let mut i = 0usize;
        while i < 2 * n {
            let p = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(out[i], out[i + 1]));
            let row = data.offset((yn + i as isize) * stride + x);
            _mm_storeu_si128(row as *mut __m128i, _mm256_castsi256_si128(p));
            _mm_storeu_si128(
                row.offset(stride) as *mut __m128i,
                _mm256_extracti128_si256::<1>(p),
            );
            i += 2;
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
    // **精确**；12bit 的积与和要到 ~2^32 ⇒ 通道必须是 **i64**。
    // **8/10bit 的界只到 2^28 / 2^30 ⇒ 走 `const I32 = true` 的 8 路 i32 版**
    // （第四十五刀；内核逻辑与 i64 版是同一份，只有元素宽度不同）。
    //
    // 两个驱动各自负责"4 条线"的收集/写回形态：
    // * [`inv_txfm_cols`]：线 = 相邻 4 **列** ⇒ 同一行的 4 个 i64 在内存里连续，
    //   直接 `loadu`/`storeu`（这是"列间天然独立"最省事的一半）；
    // * [`inv_txfm_rows`]：线 = 相邻 4 **行** ⇒ 跨 stride，靠 4×4 i32 转置收集、
    //   4×4 i64 转置写回（见 `vt_transpose4`）。

    // `I32 = true`：8 个 i32 通道（8bit/10bit）；`false`：4 个 i64 通道（12bit 与
    // 小尺寸块的兜底）。两条路的**算法逐条相同**，只有元素宽度不同 —— 值域见
    // `transform.rs` 的「值域分析」：8bit 的中间量 ≤ 2^28、10bit ≤ 2^30 ⇒ i32 装得下，
    // 12bit 到 2^32 ⇒ 必须 i64。
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_add<const I32: bool>(a: __m256i, b: __m256i) -> __m256i {
        if I32 {
            _mm256_add_epi32(a, b)
        } else {
            _mm256_add_epi64(a, b)
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_sub<const I32: bool>(a: __m256i, b: __m256i) -> __m256i {
        if I32 {
            _mm256_sub_epi32(a, b)
        } else {
            _mm256_sub_epi64(a, b)
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_neg<const I32: bool>(a: __m256i) -> __m256i {
        if I32 {
            _mm256_sub_epi32(_mm256_setzero_si256(), a)
        } else {
            _mm256_sub_epi64(_mm256_setzero_si256(), a)
        }
    }

    /// `v * c`（`c` 广播）。i64 通道走 `mul_epi32`（要求 |v| < 2^31，见值域分析）；
    /// i32 通道走 `mullo_epi32`（低 32 位；乘积在 i32 内不溢出，同样由值域分析保证）。
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_mul<const I32: bool>(v: __m256i, c: i32) -> __m256i {
        if I32 {
            _mm256_mullo_epi32(v, _mm256_set1_epi32(c))
        } else {
            _mm256_mul_epi32(v, _mm256_set1_epi64x(c as i64))
        }
    }

    /// **算术右移**：i32 通道直接 `srav`；i64 没有 `srai_epi64`，手工拼 ——
    /// 每个 64 位通道的符号位在高 32 位半字里，先复制到整个通道，再做逻辑右移 + 或。
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_srai<const I32: bool>(v: __m256i, n: u32) -> __m256i {
        debug_assert!(if I32 { (1..=31).contains(&n) } else { (1..=63).contains(&n) });
        if I32 {
            _mm256_srav_epi32(v, _mm256_set1_epi32(n as i32))
        } else {
            let sign = _mm256_shuffle_epi32(_mm256_srai_epi32(v, 31), 0b11_11_01_01);
            let hi = _mm256_sllv_epi64(sign, _mm256_set1_epi64x((64 - n) as i64));
            _mm256_or_si256(_mm256_srlv_epi64(v, _mm256_set1_epi64x(n as i64)), hi)
        }
    }

    /// `Round2( v, n )`（§3）；`n = 0` 时等于 `v`。
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_round2<const I32: bool>(v: __m256i, n: u32) -> __m256i {
        if n == 0 {
            return v;
        }
        let half = if I32 {
            _mm256_set1_epi32(1i32 << (n - 1))
        } else {
            _mm256_set1_epi64x(1i64 << (n - 1))
        };
        vt_srai::<I32>(vt_add::<I32>(v, half), n)
    }

    /// `v >> N`（**算术**右移，`N` 是编译期常量）—— **3 条**指令，比 [`vt_srai`] 少 2 条。
    ///
    /// ## 怎么来的（2026-10-05）
    ///
    /// AVX2 没有 `srai_epi64`。[`vt_srai`] 的写法是"把符号位复制到整个通道、
    /// 逻辑右移、再或回去"：`shuffle_epi32` + `sllv_epi64` + `srlv_epi64` + `or`
    /// = **5 条**（其中两条还是**变量**移位）。
    ///
    /// 换个角度：**一个 i64 就是两个 i32**，而 32 位有 `srai_epi32`。
    /// 结果的高 32 位恰好是 `hi >> N` 的**算术**值（`srai_epi32` 一条搞定），
    /// 低 32 位恰好是"逻辑 64 位右移"的低半（`srli_epi64` 一条搞定）⇒
    /// 两者按 32 位半字 `blend` 一下就行。
    ///
    /// ⚠️ **第一版写错过**：想当然写成 `(v << (64-N)) >> (64-N)` —— 那其实是
    /// **逻辑**右移（左移把符号位丢掉了），负数全错。是 `ME_THREADS=1` 的
    /// **逐字节对拍**在第一次构建后就抓出来的；现在这条由
    /// `vt_srai_const_matches_arithmetic_shift` 对着 **Rust 自己的 `>>`** 逐值对拍。
    ///
    /// ⚠️ `N` 必须 ∈ `1..=31`（`srai_epi32` 的立即数上限；生产里最大是 12）。
    /// `SHL`（= `64 - N`）只用来断言互补，**由调用方传** —— stable Rust 还不允许
    /// 在 const 泛型里做算术。
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_srai_c<const I32: bool, const N: i32, const SHL: i32>(v: __m256i) -> __m256i {
        debug_assert_eq!(N + SHL, 64, "vt_srai_c 的两个常量必须互补");
        debug_assert!((1..=31).contains(&N), "vt_srai_c 的移位量必须在 1..=31");
        if I32 {
            _mm256_srai_epi32::<N>(v)
        } else {
            // 一个 i64 = 两个 i32：`srli_epi64` 给出**正确**的低 32 位，
            // `srai_epi32` 给出**正确**的高 32 位（低 32 位无意义），按半字拼起来。
            _mm256_blend_epi32::<0xAA>(_mm256_srli_epi64::<N>(v), _mm256_srai_epi32::<N>(v))
        }
    }

    /// [`vt_round2`] 的**常量移位**版：`(v + (1 << (N-1))) >> N`。语义逐位相同。
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_round2_c<const I32: bool, const N: i32, const SHL: i32>(v: __m256i) -> __m256i {
        let half = if I32 {
            _mm256_set1_epi32(1i32 << (N - 1))
        } else {
            _mm256_set1_epi64x(1i64 << (N - 1))
        };
        vt_srai_c::<I32, N, SHL>(vt_add::<I32>(v, half))
    }

    /// 单测入口：把 [`vt_srai_c`] 的效果摊成 4 个 `i64`（`mod tests` 在 `x86` 外面，
    /// 拿不到这个私有函数，也不想为它放宽可见性）。
    #[cfg(test)]
    pub(crate) unsafe fn vt_srai_c_probe<const N: i32, const SHL: i32>(v: [i64; 4]) -> [i64; 4] {
        let x = _mm256_loadu_si256(v.as_ptr() as *const __m256i);
        let mut o = [0i64; 4];
        _mm256_storeu_si256(o.as_mut_ptr() as *mut __m256i, vt_srai_c::<false, N, SHL>(x));
        o
    }

    /// `Clip3( lo, hi, v )`：i32 通道用 `max/min`；i64 用 `cmpgt_epi64` + `blendv`。
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_clamp<const I32: bool>(v: __m256i, lo: i64, hi: i64) -> __m256i {
        if I32 {
            debug_assert!(lo >= i32::MIN as i64 && hi <= i32::MAX as i64);
            let vl = _mm256_set1_epi32(lo as i32);
            let vh = _mm256_set1_epi32(hi as i32);
            _mm256_min_epi32(_mm256_max_epi32(v, vl), vh)
        } else {
            let vl = _mm256_set1_epi64x(lo);
            let vh = _mm256_set1_epi64x(hi);
            let a = _mm256_blendv_epi8(v, vl, _mm256_cmpgt_epi64(vl, v));
            _mm256_blendv_epi8(a, vh, _mm256_cmpgt_epi64(v, vh))
        }
    }

    /// `B( a, b, angle, flip, r )`（§7.13.1.2）的通道版。
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_butterfly<const I32: bool>(v: &mut [__m256i], a: usize, b: usize, angle: i32, flip: bool) {
        let (c, s) = (
            super::super::transform::cos128(angle),
            super::super::transform::sin128(angle),
        );
        let x = vt_sub::<I32>(vt_mul::<I32>(v[a], c), vt_mul::<I32>(v[b], s));
        let y = vt_add::<I32>(vt_mul::<I32>(v[a], s), vt_mul::<I32>(v[b], c));
        v[a] = vt_round2_c::<I32, 12, 52>(x);
        v[b] = vt_round2_c::<I32, 12, 52>(y);
        if flip {
            v.swap(a, b);
        }
    }

    /// `H( a, b, flip, r )`（§7.13.1.2）的通道版。
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_hadamard<const I32: bool>(v: &mut [__m256i], a: usize, b: usize, flip: bool, r: u32) {
        let (a, b) = if flip { (b, a) } else { (a, b) };
        let (x, y) = (v[a], v[b]);
        let lo = -(1i64 << (r - 1));
        let hi = (1i64 << (r - 1)) - 1;
        v[a] = vt_clamp::<I32>(vt_add::<I32>(x, y), lo, hi);
        v[b] = vt_clamp::<I32>(vt_sub::<I32>(x, y), lo, hi);
    }

    /// `Inverse DCT array permutation process`（§7.13.2.2）的通道版。
    #[target_feature(enable = "avx2")]
    unsafe fn vt_dct_permute(v: &mut [__m256i], n: u32) {
        let n0 = 1usize << n;
        // 暂存**不清零**：只写 `[..n0]`、只读 `[..n0]`（`n0` 最小只有 4，
        // 给整个 2KB 数组清零在小块上是纯浪费）。
        //
        // ⚠️ 2026-10-24 试过"原地交换"（位反转是对合 ⇒ `i < brev(i)` 时 swap）：
        // **+3.5%（3/10 轮）** —— 逐元素的分支 + 变址 `swap` 输给下面这两段
        // 直写、无分支、可展开的拷贝。已回退，别再试。
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
    unsafe fn vt_inverse_adst4<const I32: bool>(v: &mut [__m256i], r: u32) {
        const SINPI_1_9: i32 = 1321;
        const SINPI_2_9: i32 = 2482;
        const SINPI_3_9: i32 = 3344;
        const SINPI_4_9: i32 = 3803;
        let mut s = [_mm256_setzero_si256(); 7];
        s[0] = vt_mul::<I32>(v[0], SINPI_1_9);
        s[1] = vt_mul::<I32>(v[0], SINPI_2_9);
        s[2] = vt_mul::<I32>(v[1], SINPI_3_9);
        s[3] = vt_mul::<I32>(v[2], SINPI_4_9);
        s[4] = vt_mul::<I32>(v[2], SINPI_1_9);
        s[5] = vt_mul::<I32>(v[3], SINPI_2_9);
        s[6] = vt_mul::<I32>(v[3], SINPI_4_9);
        let a7 = vt_sub::<I32>(v[0], v[2]);
        let b7 = vt_add::<I32>(a7, v[3]);
        s[0] = vt_add::<I32>(s[0], s[3]);
        s[1] = vt_sub::<I32>(s[1], s[4]);
        s[3] = s[2];
        s[2] = vt_mul::<I32>(b7, SINPI_3_9);
        s[0] = vt_add::<I32>(s[0], s[5]);
        s[1] = vt_sub::<I32>(s[1], s[6]);
        let x = [
            vt_add::<I32>(s[0], s[3]),
            vt_add::<I32>(s[1], s[3]),
            s[2],
            vt_sub::<I32>(vt_add::<I32>(s[0], s[1]), s[3]),
        ];
        for (k, slot) in v[..4].iter_mut().enumerate() {
            *slot = vt_round2_c::<I32, 12, 52>(x[k]);
        }
        let _ = r;
    }


// ===== 自动转录（从 transform.rs 的标量版逐条换名）=====
#[target_feature(enable = "avx2")]
unsafe fn vt_inverse_dct<const I32: bool>(v: &mut [__m256i], n: u32, r: u32) {
    let n0 = 1usize << n;
    vt_dct_permute(v, n);

    if n == 6 {
        for i in 0..16 {
            vt_butterfly::<I32>(v, 32 + i, 63 - i, 63 - 4 * brev(4, i) as i32, false);
        }
    }
    if n >= 5 {
        for i in 0..8 {
            vt_butterfly::<I32>(v, 16 + i, 31 - i, 6 + ((brev(3, 7 - i) << 3) as i32), false);
        }
    }
    if n == 6 {
        for i in 0..16 {
            vt_hadamard::<I32>(v, 32 + i * 2, 33 + i * 2, i & 1 == 1, r);
        }
    }
    if n >= 4 {
        for i in 0..4 {
            vt_butterfly::<I32>(v, 8 + i, 15 - i, 12 + ((brev(2, 3 - i) << 4) as i32), false);
        }
    }
    if n >= 5 {
        for i in 0..8 {
            vt_hadamard::<I32>(v, 16 + 2 * i, 17 + 2 * i, i & 1 == 1, r);
        }
    }
    if n == 6 {
        for i in 0..4 {
            for j in 0..2 {
                vt_butterfly::<I32>(
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
            vt_butterfly::<I32>(v, 4 + i, 7 - i, 56 - 32 * i as i32, false);
        }
    }
    if n >= 4 {
        for i in 0..4 {
            vt_hadamard::<I32>(v, 8 + 2 * i, 9 + 2 * i, i & 1 == 1, r);
        }
    }
    if n >= 5 {
        for i in 0..2 {
            for j in 0..2 {
                vt_butterfly::<I32>(
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
                vt_hadamard::<I32>(v, 32 + i * 4 + j, 35 + i * 4 - j, i & 1 == 1, r);
            }
        }
    }
    for i in 0..2 {
        vt_butterfly::<I32>(v, 2 * i, 2 * i + 1, 32 + 16 * i as i32, i == 0);
    }
    if n >= 3 {
        for i in 0..2 {
            vt_hadamard::<I32>(v, 4 + 2 * i, 5 + 2 * i, i == 1, r);
        }
    }
    if n >= 4 {
        for i in 0..2 {
            vt_butterfly::<I32>(v, 14 - i, 9 + i, 48 + 64 * i as i32, true);
        }
    }
    if n >= 5 {
        for i in 0..4 {
            for j in 0..2 {
                vt_hadamard::<I32>(v, 16 + 4 * i + j, 19 + 4 * i - j, i & 1 == 1, r);
            }
        }
    }
    if n == 6 {
        for i in 0..2 {
            for j in 0..4 {
                vt_butterfly::<I32>(
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
        vt_hadamard::<I32>(v, i, 3 - i, false, r);
    }
    if n >= 3 {
        vt_butterfly::<I32>(v, 6, 5, 32, true);
    }
    if n >= 4 {
        for i in 0..2 {
            for j in 0..2 {
                vt_hadamard::<I32>(v, 8 + 4 * i + j, 11 + 4 * i - j, i == 1, r);
            }
        }
    }
    if n >= 5 {
        for i in 0..4 {
            vt_butterfly::<I32>(v, 29 - i, 18 + i, 48 + ((i >> 1) as i32) * 64, true);
        }
    }
    if n == 6 {
        for i in 0..4 {
            for j in 0..4 {
                vt_hadamard::<I32>(v, 32 + 8 * i + j, 39 + 8 * i - j, i & 1 == 1, r);
            }
        }
    }
    if n >= 3 {
        for i in 0..4 {
            vt_hadamard::<I32>(v, i, 7 - i, false, r);
        }
    }
    if n >= 4 {
        for i in 0..2 {
            vt_butterfly::<I32>(v, 13 - i, 10 + i, 32, true);
        }
    }
    if n >= 5 {
        for i in 0..2 {
            for j in 0..4 {
                vt_hadamard::<I32>(v, 16 + i * 8 + j, 23 + i * 8 - j, i == 1, r);
            }
        }
    }
    if n == 6 {
        for i in 0..8 {
            vt_butterfly::<I32>(v, 59 - i, 36 + i, if i < 4 { 48 } else { 112 }, true);
        }
    }
    if n >= 4 {
        for i in 0..8 {
            vt_hadamard::<I32>(v, i, 15 - i, false, r);
        }
    }
    if n >= 5 {
        for i in 0..4 {
            vt_butterfly::<I32>(v, 27 - i, 20 + i, 32, true);
        }
    }
    if n == 6 {
        for i in 0..8 {
            vt_hadamard::<I32>(v, 32 + i, 47 - i, false, r);
            vt_hadamard::<I32>(v, 48 + i, 63 - i, true, r);
        }
    }
    if n >= 5 {
        for i in 0..16 {
            vt_hadamard::<I32>(v, i, 31 - i, false, r);
        }
    }
    if n == 6 {
        for i in 0..8 {
            vt_butterfly::<I32>(v, 55 - i, 40 + i, 32, true);
        }
    }
    if n == 6 {
        for i in 0..32 {
            vt_hadamard::<I32>(v, i, 63 - i, false, r);
        }
    }
    debug_assert_eq!(n0, 1usize << n);
}


#[target_feature(enable = "avx2")]
unsafe fn vt_inverse_adst8<const I32: bool>(v: &mut [__m256i], r: u32) {
    vt_adst_input_permute(v, 3);
    for i in 0..4 {
        vt_butterfly::<I32>(v, 2 * i, 2 * i + 1, 60 - 16 * i as i32, true);
    }
    for i in 0..4 {
        vt_hadamard::<I32>(v, i, 4 + i, false, r);
    }
    for i in 0..2 {
        vt_butterfly::<I32>(v, 4 + 3 * i, 5 + i, 48 - 32 * i as i32, true);
    }
    for j in 0..2 {
        for i in 0..2 {
            vt_hadamard::<I32>(v, 4 * j + i, 2 + 4 * j + i, false, r);
        }
    }
    for i in 0..2 {
        vt_butterfly::<I32>(v, 2 + 4 * i, 3 + 4 * i, 32, true);
    }
    vt_adst_output_permute::<I32>(v, 3);
}

#[target_feature(enable = "avx2")]
unsafe fn vt_inverse_adst16<const I32: bool>(v: &mut [__m256i], r: u32) {
    vt_adst_input_permute(v, 4);
    for i in 0..8 {
        vt_butterfly::<I32>(v, 2 * i, 2 * i + 1, 62 - 8 * i as i32, true);
    }
    for i in 0..8 {
        vt_hadamard::<I32>(v, i, 8 + i, false, r);
    }
    for i in 0..2 {
        vt_butterfly::<I32>(v, 8 + 2 * i, 9 + 2 * i, 56 - 32 * i as i32, true);
        vt_butterfly::<I32>(v, 13 + 2 * i, 12 + 2 * i, 8 + 32 * i as i32, true);
    }
    for j in 0..2 {
        for i in 0..4 {
            vt_hadamard::<I32>(v, 8 * j + i, 4 + 8 * j + i, false, r);
        }
    }
    for j in 0..2 {
        for i in 0..2 {
            vt_butterfly::<I32>(v, 4 + 8 * j + 3 * i, 5 + 8 * j + i, 48 - 32 * i as i32, true);
        }
    }
    for j in 0..4 {
        for i in 0..2 {
            vt_hadamard::<I32>(v, 4 * j + i, 2 + 4 * j + i, false, r);
        }
    }
    for i in 0..4 {
        vt_butterfly::<I32>(v, 2 + 4 * i, 3 + 4 * i, 32, true);
    }
    vt_adst_output_permute::<I32>(v, 4);
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
unsafe fn vt_adst_output_permute<const I32: bool>(v: &mut [__m256i], n: u32) {
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
        *slot = if i & 1 == 1 { vt_neg::<I32>(copy[idx].assume_init()) } else { copy[idx].assume_init() };
    }
}


    /// `Inverse ADST process`（§7.13.2.9）的通道版。
    #[target_feature(enable = "avx2")]
    unsafe fn vt_inverse_adst<const I32: bool>(v: &mut [__m256i], n: u32, r: u32) {
        match n {
            2 => vt_inverse_adst4::<I32>(v, r),
            3 => vt_inverse_adst8::<I32>(v, r),
            4 => vt_inverse_adst16::<I32>(v, r),
            _ => unreachable!("ADST 只定义到 2^n = 16；调用前已按 tx_sz 校验"),
        }
    }

    /// `Inverse Walsh-Hadamard transform process`（§7.13.2.10）的通道版。
    /// `shift ∈ {0, 2}`（行端 2、列端 0），都按**算术**右移。
    #[target_feature(enable = "avx2")]
    unsafe fn vt_inverse_wht<const I32: bool>(v: &mut [__m256i], shift: u32) {
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn shr<const I32: bool>(v: __m256i, shift: u32) -> __m256i {
            if shift == 0 {
                v
            } else {
                vt_srai::<I32>(v, shift)
            }
        }
        let mut a = shr::<I32>(v[0], shift);
        let c = shr::<I32>(v[1], shift);
        let mut d = shr::<I32>(v[2], shift);
        let mut b = shr::<I32>(v[3], shift);
        a = vt_add::<I32>(a, c);
        d = vt_sub::<I32>(d, b);
        let e = shr::<I32>(vt_sub::<I32>(a, d), 1);
        b = vt_sub::<I32>(e, b);
        let c = vt_sub::<I32>(e, c);
        a = vt_sub::<I32>(a, b);
        d = vt_add::<I32>(d, c);
        v[0] = a;
        v[1] = b;
        v[2] = c;
        v[3] = d;
    }

    /// `Inverse identity transform process`（§7.13.2.11–14）的通道版。
    #[target_feature(enable = "avx2")]
    unsafe fn vt_inverse_identity<const I32: bool>(v: &mut [__m256i], n: u32) {
        match n {
            2 => {
                for slot in v[..4].iter_mut() {
                    *slot = vt_round2_c::<I32, 12, 52>(vt_mul::<I32>(*slot, 5793));
                }
            }
            3 => {
                for slot in v[..8].iter_mut() {
                    *slot = vt_add::<I32>(*slot, *slot);
                }
            }
            4 => {
                for slot in v[..16].iter_mut() {
                    *slot = vt_round2_c::<I32, 12, 52>(vt_mul::<I32>(*slot, 11586));
                }
            }
            5 => {
                for slot in v[..32].iter_mut() {
                    *slot = if I32 { _mm256_slli_epi32::<2>(*slot) } else { _mm256_slli_epi64::<2>(*slot) };
                }
            }
            _ => unreachable!("IDTX 只定义到 2^n = 32；调用前已按 tx_sz 校验"),
        }
    }

    /// 1D 变换的通道版入口（与标量的 `forward_1d` 同一张分派表）。
    ///
    /// ⚠️ 2026-10-24 试过给这里加 `#[inline]`（想让"三层 match 的分派器"消失）：
    /// **+4.9%（2/10 轮）** —— 三个变换体被展开进每个调用点，代码膨胀反而伤
    /// I-cache。**保持不内联。**
    #[target_feature(enable = "avx2")]
    unsafe fn vt_forward_1d<const I32: bool>(kind: u32, v: &mut [__m256i], n: u32, r: u32) {
        match kind {
            TXF_DCT => vt_inverse_dct::<I32>(v, n, r),
            TXF_ADST => vt_inverse_adst::<I32>(v, n, r),
            _ => vt_inverse_identity::<I32>(v, n),
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

    /// 8×8 i32 转置（转置是自逆的，收集与写回共用这一份）。
    ///
    /// 输入 `v[k]` = 第 k 行的 8 个 i32；输出 `o[j]` = 第 j 列（lane k = 第 k 行）。
    /// 标准的三级 `unpack` + `permute2x128` 网络，共 24 条指令。
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vt_transpose8(v: [__m256i; 8]) -> [__m256i; 8] {
        let t0 = _mm256_unpacklo_epi32(v[0], v[1]);
        let t1 = _mm256_unpackhi_epi32(v[0], v[1]);
        let t2 = _mm256_unpacklo_epi32(v[2], v[3]);
        let t3 = _mm256_unpackhi_epi32(v[2], v[3]);
        let t4 = _mm256_unpacklo_epi32(v[4], v[5]);
        let t5 = _mm256_unpackhi_epi32(v[4], v[5]);
        let t6 = _mm256_unpacklo_epi32(v[6], v[7]);
        let t7 = _mm256_unpackhi_epi32(v[6], v[7]);
        let u0 = _mm256_unpacklo_epi64(t0, t2);
        let u1 = _mm256_unpackhi_epi64(t0, t2);
        let u2 = _mm256_unpacklo_epi64(t1, t3);
        let u3 = _mm256_unpackhi_epi64(t1, t3);
        let u4 = _mm256_unpacklo_epi64(t4, t6);
        let u5 = _mm256_unpackhi_epi64(t4, t6);
        let u6 = _mm256_unpacklo_epi64(t5, t7);
        let u7 = _mm256_unpackhi_epi64(t5, t7);
        [
            _mm256_permute2x128_si256(u0, u4, 0x20),
            _mm256_permute2x128_si256(u1, u5, 0x20),
            _mm256_permute2x128_si256(u2, u6, 0x20),
            _mm256_permute2x128_si256(u3, u7, 0x20),
            _mm256_permute2x128_si256(u0, u4, 0x31),
            _mm256_permute2x128_si256(u1, u5, 0x31),
            _mm256_permute2x128_si256(u2, u6, 0x31),
            _mm256_permute2x128_si256(u3, u7, 0x31),
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
    /// 8bit 平面的 **u16 → u8 收窄**（一次 32 像素、按行扫完整平面）。
    ///
    /// `packus_epi16` 在 256 位上是**按 128 位 lane** 打包的，直接存出去的顺序是
    /// `[a0..a7, b0..b7, a8..a15, b8..b15]`，要再 `permute4x64::<0xD8>` 还原成
    /// `[a0..a15, b0..b15]`。`shift` 是 10/12bit 的右移（8bit 为 0）。
    ///
    /// # Safety
    /// 必须已确认 AVX2 可用；`src` 可读 `h` 行 × `w` 项（行距 `src_stride`），
    /// `dst` 可写 `h` 行 × `w` 字节（行距 `dst_stride`）。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn narrow_u16_u8_rows(
        src: *const u16,
        src_stride: usize,
        dst: *mut u8,
        dst_stride: usize,
        w: usize,
        h: usize,
        shift: u32,
    ) {
        for y in 0..h {
            let s = src.add(y * src_stride);
            let d = dst.add(y * dst_stride);
            let mut x = 0usize;
            while x + 32 <= w {
                let a = _mm256_loadu_si256(s.add(x) as *const __m256i);
                let b = _mm256_loadu_si256(s.add(x + 16) as *const __m256i);
                let (a, b) = match shift {
                    0 => (a, b),
                    2 => (_mm256_srli_epi16::<2>(a), _mm256_srli_epi16::<2>(b)),
                    _ => (_mm256_srli_epi16::<4>(a), _mm256_srli_epi16::<4>(b)),
                };
                let p = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(a, b));
                _mm256_storeu_si256(d.add(x) as *mut __m256i, p);
                x += 32;
            }
            while x + 8 <= w {
                let a = _mm_loadu_si128(s.add(x) as *const __m128i);
                let a = match shift {
                    0 => a,
                    2 => _mm_srli_epi16::<2>(a),
                    _ => _mm_srli_epi16::<4>(a),
                };
                _mm_storel_epi64(d.add(x) as *mut __m128i, _mm_packus_epi16(a, a));
                x += 8;
            }
            while x < w {
                *d.add(x) = (*s.add(x) >> shift) as u8;
                x += 1;
            }
        }
    }

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
                    *slot = vt_round2_c::<false, 12, 52>(vt_mul::<false>(*slot, 2896));
                }
            }
            if lossless {
                vt_inverse_wht::<false>(&mut v[..4], 2);
            } else {
                vt_forward_1d::<false>(kind, &mut v[..w], log2_w, r);
            }
            // 写回：4 列一组做 4×4 i64 转置 → 每行的 4 个元素连续落盘
            let mut j = 0usize;
            while j + 4 <= w {
                let rows = vt_transpose4([v[j], v[j + 1], v[j + 2], v[j + 3]]);
                for (k, row) in rows.iter().enumerate() {
                    // `row_shift ∈ {0, 1, 2}`（`TRANSFORM_ROW_SHIFT`）⇒ 全都能常量展开
                    let o = match row_shift {
                        0 => *row,
                        1 => vt_round2_c::<false, 1, 63>(*row),
                        2 => vt_round2_c::<false, 2, 62>(*row),
                        other => vt_round2::<false>(*row, other),
                    };
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
                vt_inverse_wht::<false>(&mut v[..4], 0);
            } else {
                vt_forward_1d::<false>(kind, &mut v[..h], log2_h, r);
            }
            for (i, slot) in v[..h].iter().enumerate() {
                // `col_shift ∈ {0, 4}`（`transform.rs`：无损 0、其余 4）⇒ 常量展开
                let o = match col_shift {
                    0 => *slot,
                    4 => vt_round2_c::<false, 4, 60>(*slot),
                    other => vt_round2::<false>(*slot, other),
                };
                _mm256_storeu_si256(data.as_mut_ptr().add(i * w + j) as *mut __m256i, o);
            }
            j += 4;
        }
        j
    }

    /// 反变换的**行趟（8bit/10bit）**：8 行一组、8 个 i32 通道。
    ///
    /// 与 4 路 i64 版**逐位相同**（值域见 `transform.rs` 抬头：8bit ≤ 2^28、
    /// 10bit ≤ 2^30）。只接 `w`、`h` 都是 8 的倍数的形状；4 宽/4 高的块仍走 i64 版。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn inv_txfm_rows_i32(
        t: &[i32],
        data: &mut [i32],
        w: usize,
        h: usize,
        kind: u32,
        log2_w: u32,
        r: u32,
        row_shift: u32,
        rect_prescale: bool,
        lossless: bool,
    ) -> usize {
        if !w.is_multiple_of(8) || !h.is_multiple_of(8) || w > 64 || h > 64 {
            return 0;
        }
        let mut v = [_mm256_setzero_si256(); 64];
        let mut i = 0usize;
        while i + 8 <= h {
            // 收集：8 行 × 8 列 → 转置成"通道 = 行"
            let mut j = 0usize;
            while j + 8 <= w {
                if j >= 32 {
                    v[j..j + 8].fill(_mm256_setzero_si256());
                } else {
                    let mut rows = [_mm256_setzero_si256(); 8];
                    for (k, row) in rows.iter_mut().enumerate() {
                        if i + k < 32 {
                            *row = _mm256_loadu_si256(
                                t.as_ptr().add((i + k) * w + j) as *const __m256i
                            );
                        }
                    }
                    let cols = vt_transpose8(rows);
                    v[j..j + 8].copy_from_slice(&cols);
                }
                j += 8;
            }
            if rect_prescale {
                for slot in v[..w].iter_mut() {
                    *slot = vt_round2_c::<true, 12, 52>(vt_mul::<true>(*slot, 2896));
                }
            }
            if lossless {
                vt_inverse_wht::<true>(&mut v[..4], 2);
            } else {
                vt_forward_1d::<true>(kind, &mut v[..w], log2_w, r);
            }
            // 写回：转置回来 → 每行的 8 个元素连续落盘
            let mut j = 0usize;
            while j + 8 <= w {
                let rows = vt_transpose8([
                    v[j],
                    v[j + 1],
                    v[j + 2],
                    v[j + 3],
                    v[j + 4],
                    v[j + 5],
                    v[j + 6],
                    v[j + 7],
                ]);
                for (k, row) in rows.iter().enumerate() {
                    let o = match row_shift {
                        0 => *row,
                        1 => vt_round2_c::<true, 1, 63>(*row),
                        2 => vt_round2_c::<true, 2, 62>(*row),
                        other => vt_round2::<true>(*row, other),
                    };
                    _mm256_storeu_si256(
                        data.as_mut_ptr().add((i + k) * w + j) as *mut __m256i,
                        o,
                    );
                }
                j += 8;
            }
            i += 8;
        }
        i
    }

    /// 反变换的**列趟（8bit/10bit）**：8 列一组（同一行的 8 个 i32 连续）。
    /// 返回处理掉的**列数**（8 的倍数；0 = 全走标量）。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn inv_txfm_cols_i32(
        data: &mut [i32],
        w: usize,
        h: usize,
        kind: u32,
        log2_h: u32,
        r: u32,
        col_shift: u32,
        lossless: bool,
    ) -> usize {
        if !w.is_multiple_of(8) || !h.is_multiple_of(8) || w > 64 || h > 64 {
            return 0;
        }
        let mut v = [_mm256_setzero_si256(); 64];
        let mut j = 0usize;
        while j + 8 <= w {
            for (i, slot) in v[..h].iter_mut().enumerate() {
                *slot = _mm256_loadu_si256(data.as_ptr().add(i * w + j) as *const __m256i);
            }
            if lossless {
                vt_inverse_wht::<true>(&mut v[..4], 0);
            } else {
                vt_forward_1d::<true>(kind, &mut v[..h], log2_h, r);
            }
            for (i, slot) in v[..h].iter().enumerate() {
                let o = match col_shift {
                    0 => *slot,
                    4 => vt_round2_c::<true, 4, 60>(*slot),
                    other => vt_round2::<true>(*slot, other),
                };
                _mm256_storeu_si256(data.as_mut_ptr().add(i * w + j) as *mut __m256i, o);
            }
            j += 8;
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

    /// `SMOOTH_PRED` / `SMOOTH_V_PRED` / `SMOOTH_H_PRED` 的**一行**（8 像素）。
    ///
    /// 规范式（`smooth` 的标量版）每像素 4 次乘加；这里把 `j` 方向铺成 8 个 i32 通道：
    /// `v = wy·A[j] + (256−wy)·Llast + wx[j]·L[i] + (256−wx[j])·Alast`
    /// （V/H 两档各去掉一项）。值域：8/12bit 下 `A,L ≤ 4095`、权重 ≤ 256 ⇒ 和 ≤ 2^22，
    /// i32 装得下；与标量式**逐位相同**（乘加顺序不影响整数结果，取整只做一次）。
    ///
    /// `mode`：0 = 双线性、1 = 竖直、2 = 水平（调用方按预测模式传）。
    ///
    /// # Safety
    /// 已确认 AVX2；`above` 可读 8 个 i32、`wx` 可读 8 个 i32、`out` 可写 8 个 u16。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn intra_smooth_row(
        above: *const i32,
        wx: *const i32,
        wy: i32,
        l_i: i32,
        a_last: i32,
        l_last: i32,
        mode: u32,
        shift: u32,
        out: *mut u16,
    ) {
        use std::arch::x86_64::*;
        // `above` 是 i32 边数组（样本值 0..2^bd-1），直接 8 通道装载
        let av = _mm256_loadu_si256(above as *const __m256i);
        let wvv = _mm256_loadu_si256(wx as *const __m256i);
        let r#const = _mm256_set1_epi32(256);
        let wyv = _mm256_set1_epi32(wy);
        let li = _mm256_set1_epi32(l_i);
        let alast = _mm256_set1_epi32(a_last);
        let llast = _mm256_set1_epi32(l_last);
        let v = match mode {
            1 => _mm256_add_epi32(
                _mm256_mullo_epi32(wyv, av),
                _mm256_mullo_epi32(_mm256_sub_epi32(r#const, wyv), llast),
            ),
            2 => _mm256_add_epi32(
                _mm256_mullo_epi32(wvv, li),
                _mm256_mullo_epi32(_mm256_sub_epi32(r#const, wvv), alast),
            ),
            _ => {
                let t1 = _mm256_mullo_epi32(wyv, av);
                let t2 = _mm256_mullo_epi32(_mm256_sub_epi32(r#const, wyv), llast);
                let t3 = _mm256_mullo_epi32(wvv, li);
                let t4 = _mm256_mullo_epi32(_mm256_sub_epi32(r#const, wvv), alast);
                _mm256_add_epi32(_mm256_add_epi32(t1, t2), _mm256_add_epi32(t3, t4))
            }
        };
        // `(v + (1 << (shift-1))) >> shift`（v >= 0 ⇒ 算术/逻辑右移同值）
        let rb = _mm256_set1_epi32(1 << (shift - 1));
        let cnt = _mm_cvtsi32_si128(shift as i32);
        let r = _mm256_srl_epi32(_mm256_add_epi32(v, rb), cnt);
        // 收窄到 u16：低 8 个 i32 → 8 个 u16
        let p = _mm256_packus_epi32(r, r);
        let q = _mm_unpacklo_epi64(
            _mm256_castsi256_si128(p),
            _mm256_extracti128_si256::<1>(p),
        );
        _mm_storeu_si128(out as *mut __m128i, q);
    }

    /// `w×h` 的 u16 矩形拷贝（目标/源可以不同 stride），**整块一次调用**。
    ///
    /// ⚠️ 逐行调用 libc `memcpy` 时，每行 128 字节也要付一次真实调用；而
    /// `#[target_feature]` 又不能内联进普通函数 ⇒ 行循环必须放在核心里。
    ///
    /// # Safety
    /// 已确认 AVX2；`src` 可读 `h` 行的 `sstride` 个 u16（至少 `w` 个），
    /// `dst` 可写 `h` 行的 `dstride` 个 u16（至少 `w` 个）。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn copy_rect_u16(
        dst: *mut u16,
        dstride: usize,
        src: *const u16,
        sstride: usize,
        w: usize,
        h: usize,
    ) {
        use std::arch::x86_64::*;
        for i in 0..h {
            let s = src.add(i * sstride);
            let d = dst.add(i * dstride);
            let mut c = 0usize;
            while c + 32 <= w {
                let a = _mm256_loadu_si256(s.add(c) as *const __m256i);
                let b = _mm256_loadu_si256(s.add(c + 16) as *const __m256i);
                _mm256_storeu_si256(d.add(c) as *mut __m256i, a);
                _mm256_storeu_si256(d.add(c + 16) as *mut __m256i, b);
                c += 32;
            }
            while c + 16 <= w {
                let a = _mm256_loadu_si256(s.add(c) as *const __m256i);
                _mm256_storeu_si256(d.add(c) as *mut __m256i, a);
                c += 16;
            }
            while c + 8 <= w {
                let a = _mm_loadu_si128(s.add(c) as *const __m128i);
                _mm_storeu_si128(d.add(c) as *mut __m128i, a);
                c += 8;
            }
            while c < w {
                *d.add(c) = *s.add(c);
                c += 1;
            }
        }
    }

    /// 往 u16 平面加**一个常量残差**：`dst = Clip1(dst + c)`（DC-only 快路径用）。
    ///
    /// DC-only 的 2-D 反变换是恒定平面（有单测钉住），所以整块只要一次
    /// "逐像素加常量 + 钳位"，不必跑蝶形。
    ///
    /// # Safety
    /// 已确认 AVX2；`dst` 可写 `h` 行、每行 `stride` 个 u16（至少 `w` 个）。
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn add_const_plane(
        dst: *mut u16,
        stride: usize,
        w: usize,
        h: usize,
        c: i32,
        max: i32,
    ) {
        use std::arch::x86_64::*;
        // ⚠️ 8bit 的 i16 路径要求 `p + c` 不越出 i16：`p <= 255` ⇒ 只要求 `c <= 32512`
        // （第一版漏了这条：`c = 32767` 时 `255 + c` 回绕成负数，整帧哈希就变了）。
        if max <= 255 && c <= 32512 {
            // 8bit：**16 通道 i16 + `packus`**（`p <= 255`、`c` 在界内 ⇒ 和不会溢出 i16，
            // `packus` 的饱和正好就是 `Clip1`）。
            // ⚠️ 平面是 **u16 存储**：16 个样本要写回 32 字节（不是"打包成 u8"）。
            // 钳位用 i16 的 min/max（`p + c` 已在 i16 范围内，255 < 32767）。
            let cv = _mm256_set1_epi16(c as i16);
            let zero16 = _mm256_setzero_si256();
            let max16 = _mm256_set1_epi16(max as i16);
            for y in 0..h {
                let row = dst.add(y * stride);
                let mut x = 0usize;
                while x + 16 <= w {
                    let p = _mm256_loadu_si256(row.add(x) as *const __m256i);
                    let v = _mm256_min_epi16(
                        _mm256_max_epi16(_mm256_add_epi16(p, cv), zero16),
                        max16,
                    );
                    _mm256_storeu_si256(row.add(x) as *mut __m256i, v);
                    x += 16;
                }
                while x < w {
                    *row.add(x) = ((*row.add(x) as i32 + c).clamp(0, max)) as u16;
                    x += 1;
                }
            }
            return;
        }
        let cv = _mm256_set1_epi32(c);
        let zero = _mm256_setzero_si256();
        let maxv = _mm256_set1_epi32(max);
        for y in 0..h {
            let row = dst.add(y * stride);
            let mut x = 0usize;
            while x + 8 <= w {
                let p = _mm256_cvtepu16_epi32(_mm_loadu_si128(row.add(x) as *const __m128i));
                let v = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(p, cv), zero), maxv);
                let q = _mm256_packus_epi32(v, v);
                let q = _mm_unpacklo_epi64(
                    _mm256_castsi256_si128(q),
                    _mm256_extracti128_si256::<1>(q),
                );
                _mm_storeu_si128(row.add(x) as *mut __m128i, q);
                x += 8;
            }
            while x < w {
                *row.add(x) = ((*row.add(x) as i32 + c).clamp(0, max)) as u16;
                x += 1;
            }
        }
    }


    /// **SGR 收尾投影的一行**（8 像素）：`out = Clip1(Round2(w1·u + w0·f0 + w2·f1, shift))`。
    ///
    /// `u` 是 `CdefFrame` 的样本（左移 `SGRPROJ_RST_BITS = 4` 后参与），`f0`/`f1` 是两档
    /// box filter 的结果（i32）。`F0`/`F1` = 该档 `r == 0`（此时没有 f，用 `u` 代替）——
    /// 编译期常量，4 个两种布尔组合各一份实例，循环里没有分支。
    ///
    /// # Safety
    /// 已确认 AVX2；`u` 可读 `n` 个 u16、`f0`/`f1` 在需要时可读 `n` 个 i32、
    /// `out` 可写 `n` 个 i16（值域 `[0, max]`，与 u16 位型相同）。
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn sgr_proj_row8<const F0: bool, const F1: bool>(
        u: *const u16,
        f0: *const i32,
        f1: *const i32,
        out: *mut i16,
        n: usize,
        w0: i32,
        w1: i32,
        w2: i32,
        shift: u32,
        max: i32,
    ) {
        use std::arch::x86_64::*;
        let vw0 = _mm256_set1_epi32(w0);
        let vw1 = _mm256_set1_epi32(w1);
        let vw2 = _mm256_set1_epi32(w2);
        let rnd = _mm256_set1_epi32(1 << (shift - 1));
        let cnt = _mm_cvtsi32_si128(shift as i32);
        let zero = _mm256_setzero_si256();
        let vmax = _mm256_set1_epi32(max);
        let mut j = 0usize;
        while j + 8 <= n {
            let uv = _mm256_slli_epi32::<4>(
                _mm256_cvtepu16_epi32(_mm_loadu_si128(u.add(j) as *const __m128i)),
            );
            let a = if F0 {
                _mm256_loadu_si256(f0.add(j) as *const __m256i)
            } else {
                uv
            };
            let b = if F1 {
                _mm256_loadu_si256(f1.add(j) as *const __m256i)
            } else {
                uv
            };
            let v = _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(vw1, uv),
                    _mm256_mullo_epi32(vw0, a),
                ),
                _mm256_mullo_epi32(vw2, b),
            );
            // `round2(x, n) = (x + (1<<(n-1))) >> n`（**算术**右移，与标量一致）
            let s = _mm256_sra_epi32(_mm256_add_epi32(v, rnd), cnt);
            let s = _mm256_min_epi32(_mm256_max_epi32(s, zero), vmax);
            let q = _mm256_packus_epi32(s, s);
            let q = _mm_unpacklo_epi64(
                _mm256_castsi256_si128(q),
                _mm256_extracti128_si256::<1>(q),
            );
            _mm_storeu_si128(out.add(j) as *mut __m128i, q);
            j += 8;
        }
        while j < n {
            let uv = i32::from(*u.add(j)) << 4;
            let a = if F0 { *f0.add(j) } else { uv };
            let b = if F1 { *f1.add(j) } else { uv };
            let v = w1 * uv + w0 * a + w2 * b;
            *out.add(j) = ((v + (1 << (shift - 1))) >> shift).clamp(0, max) as i16;
            j += 1;
        }
    }


}

// 这份重导出是**给单测/微基准留的入口**：`inter.rs` 的 `block_inter_prediction`
// 现在都走"整块版"内核，于是 `mc_horiz16_8bpc` / `mc_delta8` / `mc_shift8` / `mc_vert8`
// 在非 test 构建里没有生产调用点。去掉重导出会把它们变成 `never used` 的 dead_code
// 警告（噪声更大），所以这里留着并显式放行。
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub(crate) use x86::{
    diffwtd_mask8,
    cdef_direction_avx2, cdef_pair16, cdef_pair16x4, cdef_quad8, cdef_quad8_uv,
    deblock_edge4, inv_txfm_cols, inv_txfm_cols_i32, inv_txfm_rows,
    inv_txfm_rows_i32,
    blend_avg8, blend_dist8, blend_masked8, blend_masked_u16in8, mc_delta8, mc_horiz8,
    mc_horiz16_8bpc, mc_horiz16_u8, mc_horiz_rows_8bpc, mc_shift8, mc_shift_row,
    mc_shift_rows, mc_vert8,
    mc_vert_rows_8bpc, mc_vert_rows_u16, comp_avg_rows, mc_vert2_avg_u16, add_residual_u16,
    deblock_edge8_h, add_residual_u16_rev, dequantize_8,
    sgr_ab_8, sgr_a2_vec, sgr_cols_update8, sgr_final_8, sgr_final_8_pat, sgr_prefix8,
    store_row_clamped,
    wiener_horiz_rows,
    wiener_vert_row,
    intra_smooth_row, cdef_direction_avx2_x2, copy_rect_u16, add_const_plane, sgr_proj_row8,
    narrow_u16_u8_rows,
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
///
/// ⚠️ 这份桩的最后一个是 `&mut [i16]`，**与 x86 实现一致**。曾经在这里写成
/// `&mut [i32]`（从 `mc_horiz8` 那类返回 `usize` 的内核抄来的形状），后果是：
/// **Intel 机器上一切正常**（这份桩根本不参与编译），只有 arm64 的 CI 在编译
/// 测试目标时才会报 `mismatched types`。桩的签名是"另一份实现的镜像"，
/// **抄错一次就是一次只在别的架构上出现的编译失败** —— 见
/// `tools/check_arch_stubs.py`（它会把这个漂移在 Intel 上就拦住）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn mc_horiz16_8bpc(
    _row: &[u16],
    _base: isize,
    _coef: &[i16; 8],
    _round: u32,
    _out: &mut [i16],
) {
}
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn mc_horiz16_u8(
    _row: &[u8],
    _base: isize,
    _coef: &[i16; 8],
    _round: u32,
    _out: &mut [i16],
) {
}
#[cfg(not(target_arch = "x86_64"))]
pub(crate) unsafe fn store_row_clamped(_dst: *mut u16, _src: &[i32], _n: usize, _max: i32) {}

/// 非 x86_64 的整数相位整块桩（`avx2_available()` 恒假 ⇒ 调用方走标量）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn mc_shift_rows(
    _data: &[u16],
    _stride: usize,
    _y0: i64,
    _last_y: i64,
    _h: usize,
    _base: isize,
    _post_round: u32,
    _n: usize,
    _out: *mut i32,
) -> usize {
    0
}

/// 非 x86_64 的方向搜索桩（`avx2_available()` 恒假 ⇒ 调用方走标量）。
#[cfg(not(target_arch = "x86_64"))]
pub(crate) unsafe fn cdef_direction_avx2(
    _data: &[u16],
    _stride: usize,
    _x0: usize,
    _y0: usize,
    _sh: u32,
) -> (usize, i32) {
    (0, 0)
}

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
    _dst: *mut u16,
    _dst_stride: isize,
) {
}

/// 非 x86_64 的 CDEF 4×4 桩（见 [`cdef_pair16`] 的同名桩）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cdef_quad8(
    _data: &[u16],
    _stride: isize,
    _x0: isize,
    _y0: isize,
    _offs: &[isize; CDEF_CANDS],
    _taps: &[i32; CDEF_CANDS],
    _pri_thr: i32,
    _pri_adj: u32,
    _sec_thr: i32,
    _sec_adj: u32,
    _dst: *mut u16,
    _dst_stride: isize,
) {
}

/// 非 x86_64 的 CDEF 四行桩（见 [`cdef_pair16`] 的同名桩）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cdef_pair16x4(
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
    _dst: *mut u16,
    _dst_stride: isize,
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
    _y0: i64,
    _last_y: i64,
    _h: usize,
    _base: isize,
    _wu: usize,
    _coef: &[i16; 8],
    _round: u32,
    _interior_x: bool,
    _last_x: i64,
    _out: &mut [i16],
) {
}


/// 非 x86_64：没有 AVX2 内核（`avx2_available()` 恒假 ⇒ 调用方不会走到这里）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn comp_avg_rows(
    _a: *const u16,
    _a_stride: usize,
    _ax: usize,
    _ay: usize,
    _b: *const u16,
    _b_stride: usize,
    _bx: usize,
    _by: usize,
    _dst: *mut u16,
    _dst_stride: usize,
    _dx: usize,
    _dy: usize,
    _w: usize,
    _h: usize,
) {
}


/// 非 x86_64：没有 AVX2 内核（`avx2_available()` 恒假 ⇒ 调用方不会走到这里）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn mc_vert2_avg_u16(
    _a: &[i16],
    _b: &[i16],
    _wu: usize,
    _h: usize,
    _fya: &[i16; 8],
    _fyb: &[i16; 8],
    _round: u32,
    _blend_shift: u32,
    _max: i32,
    _dst: *mut u16,
    _dst_stride: isize,
) {
}



/// 非 x86_64：没有 AVX2 内核（`avx2_available()` 恒假 ⇒ 调用方不会走到这里）。
#[cfg(not(target_arch = "x86_64"))]
pub(crate) unsafe fn add_residual_u16(_dst: *mut u16, _res: *const i32, _n: usize, _max: i32) {}


/// 非 x86_64：8 路横边内核桩。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn deblock_edge8_h(
    _data: *mut u16,
    _stride: isize,
    _bit_depth: u8,
    _xp: i32,
    _yp: i32,
    _plane: u8,
    _filter_size: i32,
    _need: i32,
    _limit: i32,
    _blimit: i32,
    _thresh: i32,
) {
}





/// 非 x86_64：翻转残差桩。
#[cfg(not(target_arch = "x86_64"))]
pub(crate) unsafe fn add_residual_u16_rev(_dst: *mut u16, _res: *const i32, _n: usize, _max: i32) {}


/// 非 x86_64：反量化桩。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dequantize_8(
    _levels: *const i32,
    _t: *mut i32,
    _n: usize,
    _q_dc: i32,
    _q_ac: i32,
    _denom_shift: u32,
    _lo: i32,
    _hi: i32,
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


/// 非 x86_64：没有 AVX2 内核（`avx2_available()` 恒假 ⇒ 调用方不会走到这里）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn mc_vert_rows_u16(
    _inter: &[i16],
    _wu: usize,
    _h: usize,
    _coef: &[i16; 8],
    _round: u32,
    _max: i32,
    _dst: *mut u16,
    _dst_stride: isize,
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

/// 非 x86_64：见 [`inv_txfm_rows_i32`]（`avx2_available()` 恒假 ⇒ 永不执行）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn inv_txfm_rows_i32(
    _t: &[i32],
    _data: &mut [i32],
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

/// 非 x86_64：见 [`inv_txfm_cols_i32`]（`avx2_available()` 恒假 ⇒ 永不执行）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn inv_txfm_cols_i32(
    _data: &mut [i32],
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

/// 非 x86_64 的 SGR 收尾投影桩（`avx2_available()` 恒假 ⇒ 走标量）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn sgr_proj_row8<const F0: bool, const F1: bool>(
    _u: *const u16,
    _f0: *const i32,
    _f1: *const i32,
    _out: *mut i16,
    _n: usize,
    _w0: i32,
    _w1: i32,
    _w2: i32,
    _shift: u32,
    _max: i32,
) {
}

/// 非 x86_64 的常量残差桩（`avx2_available()` 恒假 ⇒ 调用方走完整反变换）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn add_const_plane(
    _dst: *mut u16,
    _stride: usize,
    _w: usize,
    _h: usize,
    _c: i32,
    _max: i32,
) {
}

/// 非 x86_64 的矩形拷贝桩（`avx2_available()` 恒假 ⇒ 调用方走逐行）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn copy_rect_u16(
    _dst: *mut u16,
    _dstride: usize,
    _src: *const u16,
    _sstride: usize,
    _w: usize,
    _h: usize,
) {
}

/// 非 x86_64 的两块方向搜索桩（`avx2_available()` 恒假 ⇒ 走标量）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cdef_direction_avx2_x2(
    _data: &[u16],
    _stride: usize,
    _x0a: usize,
    _y0a: usize,
    _x0b: usize,
    _y0b: usize,
    _sh: u32,
) -> ((usize, i32), (usize, i32)) {
    ((0, 0), (0, 0))
}

/// 非 x86_64 的帧内 smooth 桩（`avx2_available()` 恒假 ⇒ 走标量）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn intra_smooth_row(
    _above: *const i32,
    _wx: *const i32,
    _wy: i32,
    _l_i: i32,
    _a_last: i32,
    _l_last: i32,
    _mode: u32,
    _shift: u32,
    _out: *mut u16,
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

/// 非 x86_64 的 [`narrow_u16_u8_rows`] 桩（`avx2_available()` 恒假 ⇒ 走标量收窄）。
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn narrow_u16_u8_rows(
    _src: *const u16,
    _src_stride: usize,
    _dst: *mut u8,
    _dst_stride: usize,
    _w: usize,
    _h: usize,
    _shift: u32,
) {
}

/// 非 x86_64 的 [`sgr_prefix8`] 桩（`avx2_available()` 恒假 ⇒ 走标量前缀和）。
#[cfg(not(target_arch = "x86_64"))]
pub(crate) unsafe fn sgr_prefix8(
    _cs: &[i32],
    _cs2: &[i32],
    _pre: &mut [i32],
    _pre2: &mut [i32],
    _n: usize,
) {
}

/// 非 x86_64 的 [`sgr_final_8_pat`] 桩（`avx2_available()` 恒假 ⇒ 走标量 3×3）。
#[cfg(not(target_arch = "x86_64"))]
pub(crate) unsafe fn sgr_final_8_pat<const PAT: u8, const SHIFT: i32>(
    _a_buf: &[i32],
    _b_buf: &[i32],
    _aw: usize,
    _base: usize,
    _u: *const u16,
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

/// 非 x86_64 的 [`mc_shift_row`] 桩：返回 0（一个也没用 SIMD 处理）⇒ 全走标量。
#[cfg(not(target_arch = "x86_64"))]
pub(crate) unsafe fn mc_shift_row(
    _row: &[u16],
    _base: isize,
    _post_round: u32,
    _n: usize,
    _out: &mut [i32],
) -> usize {
    0
}

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

    /// [`sgr_a2_vec`] 的浮点等价式必须与**整数原式**
    /// `((z << 8) + z/2) / (z + 1)` 逐值相同（`z = 0` 取 1、`z >= 255` 取 256）。
    ///
    /// 覆盖 `0..=255` 全表 + 越界正数（`>= 255` 一律 256）。
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn sgr_a2_vec_matches_integer_formula() {
        use std::arch::x86_64::*;
        if !avx2_available() {
            return;
        }
        let mut zs: Vec<i32> = (0..256).collect();
        zs.extend_from_slice(&[255, 256, 1000, 65_535, i32::MAX]);
        while zs.len() % 8 != 0 {
            zs.push(0);
        }
        let mut got = vec![0i32; zs.len()];
        for chunk in 0..zs.len() / 8 {
            // SAFETY: 已判 AVX2；每次读写 8 个 i32，下标在界内。
            unsafe {
                let v = _mm256_loadu_si256(zs.as_ptr().add(chunk * 8) as *const __m256i);
                _mm256_storeu_si256(
                    got.as_mut_ptr().add(chunk * 8) as *mut __m256i,
                    sgr_a2_vec(v),
                );
            }
        }
        for (i, &z) in zs.iter().enumerate() {
            let want = if z >= 255 {
                256
            } else if z > 0 {
                ((z << 8) + (z / 2)) / (z + 1)
            } else if z == 0 {
                1
            } else {
                1 // 负值调用方另行覆盖；这里只锁"夹到 0 ⇒ 1"
            };
            assert_eq!(got[i], want, "z={z}");
        }
    }

    /// [`sgr_prefix8`] 必须与**朴素前缀和**逐项相同（含负值、跨 8 段进位、尾巴）。
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn sgr_prefix8_matches_scalar() {
        if !avx2_available() {
            return;
        }
        for n in [0usize, 1, 7, 8, 9, 16, 17, 63, 70, 72] {
            let cs: Vec<i32> = (0..n)
                .map(|i| ((i * 37) as i32 % 251) - 100 + (i as i32 % 7) * 13)
                .collect();
            let cs2: Vec<i32> = (0..n)
                .map(|i| 1000 - ((i * 91) as i32 % 977) + (i as i32 % 5) * 31)
                .collect();
            let mut got = vec![0i32; n + 1];
            let mut got2 = vec![0i32; n + 1];
            // SAFETY: 已判 AVX2；两个源各 `n` 项、两个目标各 `n+1` 项。
            unsafe {
                sgr_prefix8(&cs, &cs2, &mut got, &mut got2, n);
            }
            let mut want = vec![0i32; n + 1];
            let mut want2 = vec![0i32; n + 1];
            for k in 0..n {
                want[k + 1] = want[k] + cs[k];
                want2[k + 1] = want2[k] + cs2[k];
            }
            assert_eq!(got, want, "n={n}");
            assert_eq!(got2, want2, "n={n}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    /// `mc_horiz16_u8`（u8 源 + `vpmovzxbw`）必须与 u16 版**逐位一致**。
    ///
    /// 这是"8bit 走 u8 存储"那条路的安全性论证：两者做的是**同一串 16 位算术**，
    /// 只是载入宽度不同 ⇒ 只要源样本相同，输出必须逐位相同。有了这条，
    /// 后面真去迁移存储时可以放心用 `ME_PLANE_U8` 开/关做逐字节对拍。
    #[cfg(target_arch = "x86_64")]
    /// `vt_srai_c`（**3 条**）必须与**算术右移**逐值相同。
    ///
    /// 参照系直接用 Rust 自己的 `i64 >> n`（那就是算术右移），比"跟另一份 SIMD 实现
    /// 对拍"更强：不会两个都错还互相印证。
    ///
    /// ## 为什么这条单测是**必需**的
    ///
    /// 第一版把 `(v << (64-N)) >> (64-N)` 当成算术右移 —— 那其实是**逻辑**右移，
    /// 负数全错。`ME_THREADS=1` 的逐字节对拍在第一次构建后就把它抓出来了；
    /// 这条单测是"再快也不会漏"的那一道闸。
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vt_srai_const_matches_arithmetic_shift() {
        if !avx2_available() {
            return;
        }
        // 覆盖值域：0、±1、±(2^k)、i64::MIN/MAX 附近的若干组合
        let mut vals: Vec<i64> = vec![
            0,
            1,
            -1,
            2,
            -2,
            i64::MIN,
            i64::MAX,
            i64::MIN + 1,
            i64::MAX - 1,
            i64::MIN / 2,
            i64::MAX / 2,
            -(1i64 << 62),
            (1i64 << 62) - 1,
            123456789012345,
            -987654321098765,
        ];
        for k in 0..63 {
            vals.push(1i64 << k);
            vals.push(-(1i64 << k));
            vals.push((1i64 << k) - 1);
            vals.push(-(1i64 << k) + 1);
        }
        for n in 1..=31u32 {
            for chunk in vals.chunks(4) {
                let mut input = [0i64; 4];
                for (k, slot) in input.iter_mut().enumerate() {
                    *slot = chunk.get(k).copied().unwrap_or(0);
                }
                // 参照系 = **Rust 自己的 `>>`**（就是算术右移）—— 比"跟另一份 SIMD
                // 实现对拍"更强：两份实现同时错还会互相印证。
                let want: [i64; 4] = std::array::from_fn(|k| input[k] >> n);
                // SAFETY: 已判 AVX2；生产只用到 1/2/4/12，这里再多验几档，
                // 其余档位用标量右移顶一下让循环完整。
                let got = unsafe {
                    match n {
                        1 => x86::vt_srai_c_probe::<1, 63>(input),
                        2 => x86::vt_srai_c_probe::<2, 62>(input),
                        3 => x86::vt_srai_c_probe::<3, 61>(input),
                        4 => x86::vt_srai_c_probe::<4, 60>(input),
                        5 => x86::vt_srai_c_probe::<5, 59>(input),
                        8 => x86::vt_srai_c_probe::<8, 56>(input),
                        12 => x86::vt_srai_c_probe::<12, 52>(input),
                        16 => x86::vt_srai_c_probe::<16, 48>(input),
                        31 => x86::vt_srai_c_probe::<31, 33>(input),
                        _ => std::array::from_fn(|k| input[k] >> n),
                    }
                };
                assert_eq!(got, want, "n={n} input={input:?}");
            }
        }
    }

    #[test]
    fn mc_horiz16_u8_matches_the_u16_version_bit_for_bit() {
        if !avx2_available() {
            return;
        }
        let stride = 140usize;
        let rows = 8usize;
        // 造一段带高低对比的样本（纯常量会掩盖取整/饱和类错误）
        let row16: Vec<u16> = (0..stride * rows)
            .map(|i| (((i * 37 + 11) % 256) as u16).min(255))
            .collect();
        let row8: Vec<u8> = row16.iter().map(|v| *v as u8).collect();
        let filters: [[i16; 8]; 4] = [
            [0, 0, 0, 128, 0, 0, 0, 0],
            [-4, 12, -24, 80, 80, -24, 12, -4],
            [-2, 8, -18, 120, 26, -10, 6, -2],
            [1, -5, 20, 112, 12, -4, 1, 0],
        ];
        for (f, coef) in filters.iter().enumerate() {
            for round in 3u32..=5 {
                for base in 3isize..=96 {
                    let mut a = [0i16; 16];
                    let mut b = [0i16; 16];
                    // SAFETY: 已判 AVX2；base ≤ 96 ⇒ 读 [base-3, base+20) ⊂ 8 行 × 140
                    unsafe {
                        mc_horiz16_8bpc(&row16, base, coef, round, &mut a);
                        mc_horiz16_u8(&row8, base, coef, round, &mut b);
                    }
                    assert_eq!(a, b, "filter={f} round={round} base={base}");
                }
            }
        }
    }

    /// 纵向 MC 的整块内核必须与**标量式**逐位相同，**每一档宽度都要覆盖**。
    ///
    /// ## 这条单测在防什么（2026-10-05 第三十七轮）
    ///
    /// `mc_vert_rows_8bpc` 原来是"16 宽主循环 + 标量尾"。而 4:2:0 的色度块有
    /// **四成宽 < 16**（8 宽 33.7%、4 宽 6.6%）⇒ 那些块的纵向整段走标量。
    /// 补了 8 宽 / 4 宽两条补丁之后，**`wu ∈ {4, 8, 12, 20, 24, 40}` 这些"非 16 的倍数"
    /// 或"不足 16"的宽度**才是被考验的路径 —— 只测 `wu = 16/32/64` 会一条都碰不到
    /// （横向那条 `mc_horiz8_8bpc` 当年就是靠"宽度分布"才发现的）。
    ///
    /// 参照系是**标量式**（`Σ_t coef[t]·inter[(r+t)·wu + c]` 再 `Round2`），
    /// 与 `inter.rs::mc_vertical_pass` 的标量分支逐字一致。
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn mc_vert_rows_matches_scalar_for_every_width() {
        if !avx2_available() {
            return;
        }
        const FILTERS: [[i16; 8]; 4] = [
            [0, 0, 0, 128, 0, 0, 0, 0],
            [-4, 12, -24, 80, 80, -24, 12, -4],
            [-2, 8, -18, 120, 26, -10, 6, -2],
            [1, -5, 20, 112, 12, -4, 1, 0],
        ];
        // `inter` 的每一行留 8 个 i16 的余量（内核按 `wu` 走行，不会越读）
        for &wu in &[4usize, 8, 12, 16, 20, 24, 28, 40, 44, 64] {
            let rows = 8 + 7;
            let mut rng: u64 = 0x9E37_79B9_7F4A_7C15 ^ (wu as u64);
            let mut next = move || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng >> 33
            };
            let inter: Vec<i16> = (0..rows * wu)
                .map(|i| match i % 5 {
                    0 => 0i16,
                    1 => -1785,
                    2 => 5865,
                    3 => 4095,
                    _ => ((next() % 12001) as i32 - 6000) as i16,
                })
                .collect();
            for (f, coef) in FILTERS.iter().enumerate() {
                for &round in &[0u32, 7, 9, 11] {
                    for &h in &[1usize, 3, 4, 8] {
                        let mut got = vec![0i32; wu * h];
                        // SAFETY: 已判 AVX2；`inter` 覆盖 (h+7) × wu 项（恰好 `rows × wu`）
                        unsafe {
                            mc_vert_rows_8bpc(
                                &inter,
                                wu,
                                h,
                                coef,
                                round,
                                got.as_mut_ptr(),
                            )
                        };
                        for r in 0..h {
                            for c in 0..wu {
                                let mut s = 0i32;
                                for (t, k) in coef.iter().enumerate() {
                                    s += *k as i32 * i32::from(inter[(r + t) * wu + c]);
                                }
                                let want = if round == 0 {
                                    s
                                } else {
                                    (s + (1 << (round - 1))) >> round
                                };
                                assert_eq!(
                                    got[r * wu + c], want,
                                    "wu={wu} f={f} round={round} h={h} r={r} c={c}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

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
            // ⚠️ `base ≥ 3`（旧版从 0 起，`base-3` 是负下标 —— 这条基准本身是 UB）
            let base = std::hint::black_box((i % 64) as isize + 3);
            // SAFETY: 已判 AVX2；base ∈ [3, 66] ⇒ 读 [base-3, base+20) ⊂ 140
            unsafe { mc_horiz16_8bpc(&row, base, &coef, 3, &mut out16) };
            sink += out16[black_box(0)] as i64;
        }
        let ns = t.elapsed().as_secs_f64() * 1e9 / (iters as f64);
        eprintln!("mc_horiz16_8bpc   {:.2} ns/调用 = {:.3} ns/像素 (sink={sink})", ns, ns / 16.0);

        // 横向：**u8 源**的同一条内核（`vpmovzxbw` 加宽，不做 `pmaddubsw`）。
        // 用来量"8bit 走 u8 存储"对 MC 横向的真实价值 —— 见内核文档的推导。
        let row8: Vec<u8> = row.iter().map(|v| *v as u8).collect();
        let mut out16u8 = [0i16; 16];
        let mut sink_u8 = 0i64;
        let t = std::time::Instant::now();
        for i in 0..iters {
            let base = std::hint::black_box((i % 64) as isize + 3);
            // SAFETY: 已判 AVX2；base ∈ [3, 66] ⇒ 读 [base-3, base+20) ⊂ 140
            unsafe { mc_horiz16_u8(&row8, base, &coef, 3, &mut out16u8) };
            sink_u8 += out16u8[black_box(0)] as i64;
        }
        let ns = t.elapsed().as_secs_f64() * 1e9 / (iters as f64);
        eprintln!(
            "mc_horiz16_u8     {:.2} ns/调用 = {:.3} ns/像素 (sink={sink_u8})",
            ns,
            ns / 16.0
        );

        // 横向：一次 **8** 个输出（2026-10-05 新增的 8 宽内核）。
        // 和上面那条比 `ns/像素` 才有意义 —— 两者做的是同一件事的不同宽度。
        let mut out8 = [0i16; 8];
        let mut sink8 = 0i64;
        let t = std::time::Instant::now();
        for i in 0..iters {
            // ⚠️ `base ≥ 3`：内核从 `base - 3` 起读，base = 0 会越界（旧版这条基准有这个问题）
            let base = std::hint::black_box((i % 64) as isize + 3);
            // SAFETY: 已判 AVX2；base ∈ [3, 66] ⇒ 读 [base-3, base+12) ⊂ 140
            unsafe { x86::mc_horiz8_8bpc(row.as_ptr(), base, &coef, 3, out8.as_mut_ptr()) };
            sink8 += out8[black_box(0)] as i64;
        }
        let ns = t.elapsed().as_secs_f64() * 1e9 / (iters as f64);
        eprintln!("mc_horiz8_8bpc    {:.2} ns/调用 = {:.3} ns/像素 (sink={sink8})", ns, ns / 8.0);

        // 标量参照：同一件事的 8 输出标量写法（= `mc_horiz_rows_8bpc` 的尾循环，
        // 也是 `wu ∈ [8,16)` 的块在改动**之前**走的唯一一条路）。
        let mut sink_sc = 0i64;
        let t = std::time::Instant::now();
        for i in 0..iters {
            // ⚠️ `base ≥ 3`：标量尾循环里 `bc - 3` 会读到 `base-3`，base=0 时越界
            let base = std::hint::black_box((i % 64) as isize + 3);
            let mut o = [0i16; 8];
            for (c, slot) in o.iter_mut().enumerate() {
                let bc = base + c as isize - 3;
                let mut s = 0i32;
                for (t, k) in coef.iter().enumerate() {
                    s += *k as i32 * i32::from(row[(bc + t as isize) as usize]);
                }
                *slot = rnd_scalar(s, 3) as i16;
            }
            sink_sc += o[black_box(0)] as i64;
        }
        let ns = t.elapsed().as_secs_f64() * 1e9 / (iters as f64);
        eprintln!("横向标量(8 输出)  {:.2} ns/调用 = {:.3} ns/像素 (sink={sink_sc})", ns, ns / 8.0);

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
        // 目标缓冲与源同 stride（内核按 `(y0+r)*dst_stride + x0` 写，`x0 = 8`）
        let mut out_i16 = vec![0u16; stride * rows];
        let mut best_new = f64::MAX;
        let mut best_x4 = f64::MAX;
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
                        out_i16.as_mut_ptr(),
                        stride as isize,
                    )
                });
            }
            let ns = t.elapsed().as_secs_f64() * 1e9 / iters as f64;
            best_new = best_new.min(ns);

            let t = std::time::Instant::now();
            for _ in 0..iters {
                std::hint::black_box(unsafe {
                    cdef_pair16x4(
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
                        out_i16.as_mut_ptr(),
                        stride as isize,
                    )
                });
            }
            let ns = t.elapsed().as_secs_f64() * 1e9 / iters as f64;
            best_x4 = best_x4.min(ns);

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
            "CDEF 8×8 块（亮度）：i16 两行 {best_new:7.1} ns/块、i16 四行 {best_x4:7.1} ns/块、\
             旧 i32 一行 {best_old:7.1} ns/块 ⇒ 四行/两行 {:.2}x",
            best_new / best_x4
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
                // ⚠️ 目标缓冲必须与**源同 stride、同 x0/y0 约定** —— 内核按
                // `(y0+r)*dst_stride + x0` 算地址，`x0 = 8` 意味着最小宽度要 ≥ 17。
                // 一开始写成 `h*wu` + `dst_stride = wu` 就越界了（单测当场 SIGSEGV）。
                let mut dst_buf = vec![0u16; stride * rows];
                // SAFETY: 已判 AVX2；(x0,y0)=(8,8)、h≤8、候选 ±2 ⇒ 最远读到 (x0+9, y0+9)，
                // 都在 40×32 内；目标缓冲同尺寸。
                unsafe {
                    cdef_pair16(
                        &data, stride as isize, x0, y0, h, wu, &offs, &taps, pri_str, pri_adj,
                        sec_str, sec_adj, dst_buf.as_mut_ptr(), stride as isize,
                    )
                };
                let at = |r: usize, c: usize| dst_buf[(y0 as usize + r) * stride + x0 as usize + c] as i16;
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
                            at(r, c) as i32,
                            want,
                            "r={r} c={c} h={h} wu={wu} pri={pri_str} sec={sec_str}"
                        );
                    }
                }
            }
        }
    }

    /// **融合的"纵向两路 + AVERAGE"必须与"两路纵向 + 平均"逐位一致**
    /// （`mc_vert2_avg_u16` vs `mc_vert_rows_8bpc` ×2 + 标量混合）。
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn mc_vert2_avg_u16_matches_vert_then_average() {
        if !avx2_available() {
            return;
        }
        let fya: [i16; 8] = [1, -3, 7, 23, 23, 7, -3, 1];
        let fyb: [i16; 8] = [-2, 5, 11, 18, 18, 11, 5, -2];
        let round = 7u32;
        let blend_shift = 5u32;
        for (wu, h) in [(4usize, 4usize), (5, 3), (8, 6), (16, 8), (24, 12), (36, 9)] {
            let rows = h + 7;
            let a: Vec<i16> = (0..rows * wu)
                .map(|i| (((i * 37) % 8001) as i16) - 4000)
                .collect();
            let b: Vec<i16> = (0..rows * wu)
                .map(|i| (((i * 91 + 17) % 8001) as i16) - 4000)
                .collect();
            let mut pa = vec![0i32; wu * h];
            let mut pb = vec![0i32; wu * h];
            // SAFETY: 已判 AVX2；两路中间量都有 `(h+7)·wu` 项。
            unsafe {
                mc_vert_rows_8bpc(&a, wu, h, &fya, round, pa.as_mut_ptr());
                mc_vert_rows_8bpc(&b, wu, h, &fyb, round, pb.as_mut_ptr());
            }
            let want: Vec<u16> = (0..wu * h)
                .map(|i| {
                    let s = pa[i] + pb[i] + (1i32 << (blend_shift - 1));
                    (s >> blend_shift).clamp(0, 255) as u16
                })
                .collect();
            let mut got = vec![0u16; wu * h];
            // SAFETY: 同上；目标按 `wu` 步长连续存放。
            unsafe {
                mc_vert2_avg_u16(
                    &a, &b, wu, h, &fya, &fyb, round, blend_shift, 255,
                    got.as_mut_ptr(), wu as isize,
                );
            }
            assert_eq!(got, want, "wu={wu} h={h}");
        }
    }

    /// [`cdef_pair16x4`]（四行一组）必须与 [`cdef_pair16`]（两行一组）**逐位相同**。
    ///
    /// 两条路的像素式完全相同、只是向量打包方式不同；这里对 8 个方向、
    /// 两种宽度、五组强度（含"某一档强度为 0"的单态化分支）逐字节对拍两块目标缓冲。
    ///
    /// ⚠️ 2026-10-25：这条测试的 `#[cfg]`/`#[test]` 之前**粘到了下一个函数**上
    /// （`#[test]` 后面隔着文档注释又跟了一个 `#[test]`），于是它一直没被跑过 ——
    /// 属性必须紧贴被标注的 `fn`。
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn cdef_pair16x4_matches_pair16_bit_for_bit() {
        if !avx2_available() {
            return;
        }
        let stride = 40usize;
        let rows = 32usize;
        let mut data: Vec<u16> = (0..stride * rows)
            .map(|i| (((i * 53) % 251) as u16) + 3)
            .collect();
        for y in 0..rows {
            for x in 0..stride {
                if (x / 8 + y / 8) % 2 == 0 {
                    data[y * stride + x] = (data[y * stride + x] / 2).max(1);
                }
            }
        }
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
        let (x0, y0) = (8isize, 8isize);
        for dir in 0..8usize {
            for (h, wu) in [(8usize, 8usize), (4, 4), (8, 4)] {
                for (pri_str, sec_str, pri_adj, sec_adj, tap_sel) in [
                    (32i32, 8i32, 0u32, 1u32, 0usize),
                    (7, 3, 2, 3, 1),
                    (240, 240, 0, 0, 0),
                    // 单档为 0：走 `cdef_pair16x4_impl` 的 min/max-only 单态化，
                    // 约束量恒 0 ⇒ 必须与两行版的完整式子逐位相同。
                    (0, 8, 0, 1, 0),
                    (32, 0, 0, 0, 0),
                ] {
                    let mut offs = [0isize; CDEF_CANDS];
                    let mut taps = [0i32; CDEF_CANDS];
                    let mut n = 0usize;
                    for k in 0..2usize {
                        for sign in [-1i32, 1] {
                            let [dy, dx] = DIRS[dir][k];
                            offs[n] =
                                (sign * dy) as isize * stride as isize + (sign * dx) as isize;
                            taps[n] = PRI_TAPS[tap_sel][k];
                            n += 1;
                        }
                    }
                    for k in 0..2usize {
                        for dir_off in [-2i32, 2] {
                            let d = ((dir as i32 + dir_off) & 7) as usize;
                            let [dy, dx] = DIRS[d][k];
                            for sign in [-1i32, 1] {
                                offs[n] = (sign * dy) as isize * stride as isize
                                    + (sign * dx) as isize;
                                taps[n] = SEC_TAPS[tap_sel][k];
                                n += 1;
                            }
                        }
                    }
                    let mut a = vec![0u16; stride * rows];
                    let mut b = vec![0u16; stride * rows];
                    // SAFETY: 已判 AVX2；(x0,y0)=(8,8)、h≤8、候选 ±2 ⇒ 读取在 40×32 内。
                    unsafe {
                        cdef_pair16(
                            &data, stride as isize, x0, y0, h, wu, &offs, &taps, pri_str,
                            pri_adj, sec_str, sec_adj, a.as_mut_ptr(), stride as isize,
                        );
                        cdef_pair16x4(
                            &data, stride as isize, x0, y0, h, wu, &offs, &taps, pri_str,
                            pri_adj, sec_str, sec_adj, b.as_mut_ptr(), stride as isize,
                        );
                    }
                    assert_eq!(
                        a, b,
                        "dir={dir} h={h} wu={wu} pri={pri_str} sec={sec_str}"
                    );
                }
            }
        }
    }

    /// `cdef_quad8`（4 行 × 4 列）必须与 `cdef_pair16`（2 行 × 8 列、只取前 4 列）
    /// **逐位相同**。
    ///
    /// ## 为什么要这条（而不是直接对着拍标量）
    ///
    /// `cdef_pair16` 已经有 [`cdef_pair16_matches_scalar_bit_for_bit`] 钉着标量。
    /// 本函数是**同一个数学式子的第二份实现**（只是"哪些通道装哪一行"不同），
    /// 所以对着那份已被标量钉住的实现对拍，比再写一遍标量更强也更短 ——
    /// 它同时挡住"新内核的 qword 拼装顺序写反"这类**只在 4×4 上出现**的错误。
    ///
    /// 覆盖：8 个方向 × 3 组阈值/阻尼 × 两类数据（阶梯+振铃 / 伪随机）。
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn cdef_quad8_matches_cdef_pair16_bit_for_bit() {
        if !avx2_available() {
            return;
        }
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
        let stride = 40usize;
        let rows = 32usize;
        let (x0, y0) = (8isize, 8isize);
        // 两类数据：① 阶梯 + 振铃（`constrain` 的各条分支都会走到）
        //          ② 伪随机（含 0/4095 极值，验算 i16 通道不溢出）
        let mut stepped: Vec<u16> = (0..stride * rows)
            .map(|i| (((i * 37) % 251) as u16) + 3)
            .collect();
        for y in 0..rows {
            for x in 0..stride {
                if (x / 8 + y / 8) % 2 == 0 {
                    stepped[y * stride + x] = (stepped[y * stride + x] / 2).max(1);
                }
            }
        }
        let noisy: Vec<u16> = (0..stride * rows)
            .map(|i| if i % 97 == 0 { 4095 } else { ((i * 2654) % 4096) as u16 })
            .collect();
        for (tag, data) in [("阶梯", &stepped), ("伪随机", &noisy)] {
            for dir in 0..8usize {
                for (pri_str, sec_str, pri_adj, sec_adj, tap_sel) in [
                    (32i32, 8i32, 0u32, 1u32, 0usize),
                    (7, 3, 2, 3, 1),
                    (0, 5, 0, 0, 0),
                    (240, 240, 0, 1, 1),
                ] {
                    let pri_taps = PRI_TAPS[tap_sel];
                    let sec_taps = SEC_TAPS[tap_sel];
                    let mut offs = [0isize; CDEF_CANDS];
                    let mut taps = [0i32; CDEF_CANDS];
                    let mut n = 0usize;
                    for k in 0..2usize {
                        for sign in [-1i32, 1] {
                            let [dy, dx] = DIRS[dir][k];
                            offs[n] =
                                (sign * dy) as isize * stride as isize + (sign * dx) as isize;
                            taps[n] = pri_taps[k];
                            n += 1;
                        }
                    }
                    for k in 0..2usize {
                        for dir_off in [-2i32, 2] {
                            let d = ((dir as i32 + dir_off) & 7) as usize;
                            let [dy, dx] = DIRS[d][k];
                            for sign in [-1i32, 1] {
                                offs[n] = (sign * dy) as isize * stride as isize
                                    + (sign * dx) as isize;
                                taps[n] = sec_taps[k];
                                n += 1;
                            }
                        }
                    }
                    let mut a = vec![0u16; stride * rows];
                    let mut b = vec![0u16; stride * rows];
                    // SAFETY: 已判 AVX2；(x0,y0)=(8,8)、h=4、候选 ±2 ⇒ 最远读到
                    // (x0+5, y0+6)，都在 40×32 内；两个目标缓冲同尺寸同约定。
                    unsafe {
                        cdef_pair16(
                            data, stride as isize, x0, y0, 4, 4, &offs, &taps, pri_str, pri_adj,
                            sec_str, sec_adj, a.as_mut_ptr(), stride as isize,
                        );
                        cdef_quad8(
                            data, stride as isize, x0, y0, &offs, &taps, pri_str, pri_adj,
                            sec_str, sec_adj, b.as_mut_ptr(), stride as isize,
                        );
                    }
                    for r in 0..4usize {
                        for c in 0..4usize {
                            let i = (y0 as usize + r) * stride + x0 as usize + c;
                            assert_eq!(
                                a[i], b[i],
                                "{tag} dir={dir} pri={pri_str} sec={sec_str} r={r} c={c}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// `dbl_wide` 的**精确递推**必须与**直接求和**逐位相同。
    ///
    /// ## 这条单测在防什么
    ///
    /// 递推 `A(i+1) = A(i) + t'[i+N+1] − t'[i−N] + t'[i+N2+1] − t'[i−N2]` 的推导
    /// 依赖两件事，任何一件写错都会**从某个 i 开始整段偏掉**（而不是只错一格）：
    ///
    /// 1. `w(j) = (|j| ≤ N2) ? 2 : 1` **对 `j` 偶对称** ⇒ 两个边界项同权；
    /// 2. `w(k−1) − w(k)` 只在 `|·| ≤ N2` 这个区间指示函数的**两端**非零，
    ///    即 `k = −N2`（−1）与 `k = N2+1`（+1）。
    ///
    /// `N2 = 0`（亮度的 8 抽头宽滤波器）是边界情形：那两个点退化成 `k = 0` 与 `k = 1`
    /// —— 所以三个实例都跑，别只测 `N2 ≥ 1` 的。
    ///
    /// 参照系是**直接求和**（旧写法），不是"另一个递推"。数据覆盖 0 / 极值混排 /
    /// 等差 / 伪随机，且**四条通道各给不同值**（通道间串了也能看出来）。
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dbl_wide_recurrence_matches_the_direct_sum() {
        use std::arch::x86_64::{_mm_setr_epi32, _mm_setzero_si128, _mm_storeu_si128};
        if !avx2_available() {
            return;
        }
        let mut rng: u64 = 0x243F_6A88_85A3_08D3;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 33) as i32
        };
        // 三个真实实例：(N, N2, L2) —— 亮度 8 抽头 / 色度 8 抽头 / 16 抽头
        for (n, n2, l2) in [(3i32, 0i32, 3u32), (2, 1, 3), (6, 1, 4)] {
            for case in 0..6 {
                // 14 个槽（索引 −7..6），每槽 4 条通道
                let mut slots = [[0i32; 4]; 14];
                for (si, slot) in slots.iter_mut().enumerate() {
                    for (lane, cell) in slot.iter_mut().enumerate() {
                        *cell = match case {
                            0 => 0,
                            1 => {
                                if (si + lane) % 2 == 0 {
                                    4095
                                } else {
                                    -4095
                                }
                            }
                            2 => (si as i32) * 977 - 6000 + lane as i32,
                            3 => if si % 3 == 0 { 4095 } else { 0 },
                            _ => (next() % 8191) - 4095,
                        };
                    }
                }
                let mut v: [std::arch::x86_64::__m128i; 16] =
                    std::array::from_fn(|_| unsafe { _mm_setzero_si128() });
                for k in -(n + 1)..=n {
                    let s = &slots[(k + 7) as usize];
                    v[(k + 8) as usize] = unsafe { _mm_setr_epi32(s[0], s[1], s[2], s[3]) };
                }
                let got = unsafe {
                    match (n, n2, l2) {
                        (3, 0, 3) => x86::dbl_wide::<3, 0, 3>(&v),
                        (2, 1, 3) => x86::dbl_wide::<2, 1, 3>(&v),
                        _ => x86::dbl_wide::<6, 1, 4>(&v),
                    }
                };
                // 参照系：直接求和（旧写法），逐通道算
                for i in -n..n {
                    for lane in 0..4usize {
                        let mut s = 0i32;
                        for j in -n..=n {
                            let p = (i + j).clamp(-(n + 1), n);
                            let t = slots[(p + 7) as usize][lane];
                            s += if j.abs() <= n2 { 2 * t } else { t };
                        }
                        let want = (s + (1 << (l2 - 1))) >> l2;
                        // ⚠️ `_mm_extract_epi32` 在 Rust 里是 **const 泛型**（`::<IMM8>`），
                        // 不能拿运行时下标当参数 —— 存到数组再按下标取。
                        let mut lanes = [0i32; 4];
                        unsafe {
                            _mm_storeu_si128(
                                lanes.as_mut_ptr() as *mut std::arch::x86_64::__m128i,
                                got[(i + n) as usize],
                            )
                        };
                        let got_lane = lanes[lane];
                        assert_eq!(
                            got_lane, want,
                            "N={n} N2={n2} L2={l2} case={case} i={i} lane={lane}"
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
