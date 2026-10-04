//! **自研 Opus 解码器**——三条编码路径（CELT-only / SILK-only / 混合）连同冗余帧、
//! 模式切换过渡与两侧的丢包隐藏都已落地并逐层对拍，**已经接进引擎**
//! （`decode --audio` 能把 `.opus` 写成 WAV）。
//!
//! ## 为什么从熵层开始
//!
//! Opus 的码流是**完全熵编码**的：SILK 与 CELT 的每一段参数都通过同一个区间解码器
//! （RFC 6716 §4.1）取出，连"这一帧是不是静音"这种 1 位标志也是。它是两条编码路径
//! （SILK / CELT）共同的地基，而且规范把它定义得**位精确**（所有运算都是整数、
//! 不允许任何近似），所以它能单独拿出来验证 —— 见下面的"怎么验证"。
//!
//! ## 分层与进度（照 RFC 6716 的章节）
//!
//! | 层 | 内容 | 状态 |
//! |---|---|---|
//! | 熵层 | 区间解码器、原始比特、`ec_tell`/`ec_tell_frac` | ✅ 与参考实现逐位对拍 |
//! | CELT 前端 | TOC、拆帧、帧头标志、粗能量、TF 变化、扩散 | ✅ 逐帧对拍（11406 + 18784 帧）|
//! | CELT 分配 | dynalloc 提升、alloc_trim、`compute_allocation` | ✅ 逐帧对拍（同上）|
//! | CELT 形状 | PVQ 组合数解码（`cwrs`）| ✅ 与参考编码器 1480 用例对拍 |
//! | CELT 形状 | 细能量解码、归一化还原（`denormalise_bands`）| ✅ 逐帧对拍 |
//! | CELT 形状 | 频带形状（`quant_band` 递归分裂、θ 立体声、折叠）、反塌陷 | ✅ 逐带 + 逐样本对拍（11406 + 18784 帧）|
//! | CELT 合成 | IMDCT（混合基 FFT）、重叠相加、后置滤波、去加重 | ✅ PCM 逐样本对拍 |
//! | CELT 整链 | 顺序解码（状态自然传承）+ RFC §6.1 最终区间状态 | ✅ 真实素材全 11406 帧逐样本一致；向量 01/07/11 的 6886 个包区间状态全对 |
//! | SILK 熵层 | 包首标志、立体声预测、帧索引（NLSF/pitch/LTP/增益/种子）、激励脉冲（壳编码） | ✅ 逐位对拍（向量 02/03/04/05/06/12：8693 包 / 265 万脉冲）|
//! | SILK 参数 | NLSF→LPC、增益反量化、pitch lag、LTP 参数 | ✅ 逐位对拍（同上 8693 包 / 9769 帧）|
//! | SILK 合成 | 激励重建、LTP + LPC 合成、立体声 MS→LR | ✅ 逐样本对拍（9769 帧 / 265 万样点）|
//! | SILK 重采样 | 内部率 → API 率（2 倍上采样 + FIR 分数插值）| ✅ 与参考 `silk_resampler` 逐样点一致 |
//! | SILK 整链 | 声道/采样率状态机 + 重采样 + 交织（`silk_Decode`）| ✅ 逐样本对拍（`SOUT`）|
//! | 整包 | 混合模式相加（CELT 侧 `start_band = 17`）、冗余帧与 `smooth_fade`、模式切换复位 | ✅ 官方向量 11/12 逐样本对拍（2771 万样本、最大差 1 个量化步长）|
//! | CELT PLC | `celt_decode_lost`：pitch 预测（pitch 搜索 + LPC 白化/合成 + 衰减 + TDAC）与噪声填充（`backgroundLogE` + LCG）| ✅ 13 组"手工丢包"逐样本对拍（3493 万样本、最大差 2 步）|
//! | 整包 PLC | 空载荷/DTX、超过 20 ms 的丢包拆段、模式切换过渡（`pcm_transition` 交叉淡入）| ✅ 官方向量 08/09/10 全部**逐样本对齐**（最大差 1 步）|
//! | 接入 | `open_audio_decoder`（`CodecId::Opus`）→ `decode --audio` 可写 WAV | ✅ 真实 `.opus` 与参考解码逐样本差 ≤1 步 |
//! | SILK PLC | `silk_PLC_conceal`（LTP/LPC 外推 + 随机激励 + 逐子帧衰减 + pitch 漂移）与 `silk_CNG`（舒适噪声）| ✅ **逐样本完全相同**（tv02/03/04 的 11 组丢包图样、最大差 0）|
//! | 容器 | `.opus`（Ogg）| ✅ 已接（`me-container/src/ogg.rs`）|
//!
//! **三条编码路径（SILK-only / 混合 / CELT-only）现在都能解成 48 kHz 立体声 PCM**，
//! 并已经接进 [`crate::audio::open_audio_decoder`]：`decode --audio` 能把 `.opus`
//! 写成 WAV、`stream --audio=pcm` 也能喂 PCM。**两侧的丢包隐藏都做完了**：
//! 空载荷（DTX）、连续丢包、SILK ↔ CELT 模式切换时的 5 ms 过渡都能解；
//! SILK 侧的隐藏与舒适噪声是**逐位精确**的（整数实现），CELT 侧在 1~2 个量化步长内。
//!
//! ## 官方向量验收（12 组、3078 万样本）
//!
//! `tests/opus_decode_ref.rs`（`OPUS_TV_DIR` 指向 `*.pkts.bin`/`*.pcm`）：
//!
//! - 混合模式的冗余帧长度是 `ec_dec_uint(&dec, 256) + 2` —— **参数是上界 256，
//!   不是位数 8**。写成 8 只能读出 0..7，冗余帧长度与主数据的 `celt_len` 会一起偏，
//!   两种冗余帧（CELT→SILK / SILK→CELT）都会解到错的字节上。症状很有迷惑性：
//!   熵状态在 SILK 结束时**逐位相同**（`val`/`rng` 都对），但从这里开始分叉，
//!   而且只在"混合模式 + 有冗余帧"的包上发作（12 组向量里只有 10 会撞上）。
//! - CELT 侧（浮点参考 vs 我们的定点）**最大差 1 个量化步长**；
//!   SILK-only 的 02/03/04 是**逐位相同**（`silk_only_vectors_are_bit_exact`）。
//!
//! ## 怎么验证（这条对本项目特别重要）
//!
//! 1. **与规范等价的专用方法互相印证**：`dec_bit_logp` / `dec_icdf` 在规范里是
//!    "与 `decode()` + `update()` 数学等价"的快捷路径，单测里就按朴素路径再算一遍互对。
//! 2. **拿参考实现对拍真实码流**：`ec_tell` / `ec_tell_frac` 必须**逐位一致**（规范
//!    §4.1.6 明确要求"必须与编码器返回完全相同的值"），而真实 Opus 包里第一个符号
//!    就是 CELT 的 silence 标志，所以"初始化 + 一个符号 + 规范化 + tell"这条链能在
//!    11406 个真实包上逐个核对 —— 见 `tests/opus_entropy_ref.rs`（手工跑）。
//! 3. **官方测试向量**（RFC 6716 §6.1）：完整解码器对拍要用它，素材已备好。
//!
//! ## 分层的对拍入口（都在 tests/ 下，默认 `#[ignore]`）
//!
//! | 测试 | 对拍的内容 |
//! |---|---|
//! | `opus_entropy_ref.rs` | 熵层：真实包逐包的 silence 位 + `tell` + `tell_frac` |
//! | `opus_celt_ref.rs` | 前端 + 比特分配：帧头标志、粗能量、TF、spread、分配结果 |
//! | `opus_celt_bands_ref.rs` | 细能量 + 归一化还原 |
//! | `opus_celt_shape_ref.rs` | 频带形状 + 反塌陷：逐带 `QB` 轨迹 + 逐样本归一化频谱 |
//! | `opus_celt_synth_ref.rs` | 合成：合成状态顺序传承 → **逐样本 PCM**（i16） |
//! | `opus_celt_decode_ref.rs` | 整链顺序解码：全量 PCM + **RFC §6.1 最终区间状态** |
//! | `opus_silk_indices_ref.rs` | SILK 熵层：包首标志、帧索引、激励脉冲（逐位） |
//! | `opus_silk_params_ref.rs` | SILK 参数重建：增益 / LPC / LTP / pitch（逐位） |
//! | `opus_silk_synth_ref.rs` | SILK 核心合成 + 立体声解混：`SCORE` / `SLR`（逐样本） |
//! | `opus_silk_decode_ref.rs` | SILK 整链：`SOUT`（含重采样与交织，逐样本） |
//! | `opus_decode_ref.rs` | **整包**：11+1 组官方向量的整段 PCM（逐样本）+ SILK 逐位 |
//! | `opus_plc_ref.rs` | 丢包隐藏：同一套"丢包图样"下的整段 PCM（逐样本） |
//! | `opus_decode_ref.rs` | **整包**（CELT + SILK + 混合 + 冗余帧）：参考解码器的整段 `.pcm` |
//! | `opus_pvq_ref.rs` | PVQ 组合数解码（用参考**编码器**造的用例） |
//!
//! 参考实现侧的插桩脚本是 **`/tmp/opus_patch_dbg_v2.py`**（CELT，旧版
//! `opus_patch_dbg.py` 里有调试残留，别用）与 **`/tmp/opus_patch_silk.py`**（SILK）。
//! 改插桩之后跑 **`/tmp/opus_rebuild_all.sh`** 一条龙：重插桩（两个脚本都跑）→ 编译
//! → 重生成真实素材与官方测试向量的**全部** dump —— dump 与插桩版本不一致是对拍
//! 误报的头号来源。SILK 码表由 `/tmp/opus_dump_silk_tables.c` +
//! `/tmp/opus_gen_silk_tables.py` 生成（同样带回读校验）。

pub mod audio_decoder;
pub mod bands;
pub mod bands_tables;
pub mod celt;
pub mod celt_plc;
pub mod cwrs;
pub mod entropy;
pub mod mathops;
pub mod mdct;
pub mod opus_decoder;
pub mod rate;
pub mod rate_tables;
pub mod silk;
pub mod silk_decoder;
pub mod silk_fix;
pub mod silk_params;
pub mod silk_plc;
pub mod silk_resample;
pub mod silk_synth;
pub mod silk_tables;
pub mod synth;
pub mod vq;

pub use bands::{
    anti_collapse, denormalise_bands, quant_all_bands, unquant_energy_finalise,
    unquant_fine_energy, ShapeInput, ShapeOutput,
};
pub use celt::{decode_front, finish_frame, update_history, Bandwidth, CeltFront, CeltState, OpusMode, Toc};
pub use rate::{decode_allocation, Allocation};
pub use entropy::RangeDecoder;
