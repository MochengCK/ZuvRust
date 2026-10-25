# 性能线交接（2026-10-03 下午，用户："播放 AV1 还一卡一卡"）

> 目标：把 1080p AV1 软解从"能用"推到"能实时播"（33 ms/帧）。  
> **最新（2026-10-23，第五十一刀）**：1080p 单线程约 **60 ms/帧**（CPU 时间口径；
> 第四十八刀折算 ~63 ms，第四十九/五十刀各 −0.7%，第五十一刀 −2.7% 后折算）。
> 相比 10-06 的 76.2 ms 累计约 **−21%**，目标 33 ms 还差约 1.8×。
>
> 现在（2026-10-03 晚，36 帧 8 线程，同窗口交替取最优）：**inline 201 ms/帧、  
> row-mt 并行 186 ms/帧（~5.4 fps）**。⚠️ 本机噪声大，绝对值与下午那轮不可直接比，  
> **只信同一窗口内的交替 A/B**。  
> 已落三刀（都逐字节可复现）：① LR 按 64 像素一档 ② row-mt 等待改短睡  
> ③ **row-mt 语法阶段不再重复预测**（第三节，已修，1.33x）——row-mt 现在**第一次快过 inline**。

## 一、本轮落地的（已验证，可直接往上盖）

| 改动                                                                                 | 效果（同窗口交替 A/B）                                    | 验证                                                     |
| ---------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------ |
| **环路恢复按 64 像素一档**（`lr.rs::LR_CHUNK`，原先按 4×4 块调用、`box_filter` 的 A/B 重复算 4~6 倍）      | 滤波阶段 8 线程 **46→32 ms**；整帧 **1.10x**（196.6→178.8） | 真素材 20 帧对 dav1d 逐像素、13 条受控流、384 单测、`ME_NO_SIMD` 开/关逐字节 |
| **row-mt 等待 `yield_now`→20µs 短睡**（`block.rs::wait_until`，4 物理核/8 逻辑核上 yield 仍在烧前端） | ≈1.03x                                           | row-mt 顺序/并行回放逐字节一致                                    |

- 文档已同步：`crates/me-decode/src/video/mod.rs` 抬头的"第十六轮"、`DEVELOPMENT.md` 性能段（⑬）。
- 已回填 `extra/darwin/x64/engine/zuvrust`，备份 `/tmp/me-extra-prev-20261003-perf-round16`。

## 二、当前水位（拆账，单线程；本机噪声大，只信同一时间窗内的交替对比）

```
纯语法（AV1_SYNTAX_ONLY=1）   16~26 ms
重建（= 瓦片 − 纯语法）       ~45~50 ms   ← MC 为主，完全没并行
滤波                          ~72~100 ms  ← LR 已从 ~54 降到 ~40，去块 ~10、CDEF ~7
8 线程：瓦片（串行）~93~104 + 滤波 ~32~35 + 参考/输出 ~8  → 130~190 ms/帧
```

忙时占比（`sample`，单线程）：**MC（`compute_prediction` + 内核 + warp）≈23%、LR ≈12%、  
去块 ≈10%、CDEF ≈6.7%、反变换 ≈4%、纯语法 ≈13%**。

## 三、下一刀（下个会话从这里接）：row-mt 把整帧预测**跑了两遍** → ✅ **已修（2026-10-03 晚）**

> **结论：已落地。** 修法就是下面"建议修法"的第一种（`filter(|_| !recording)`）。  
> 实测：`RMSTAGES` 的 `syntax=` **58~~61ms → 16~~38ms**；row-mt 并行 **9613 → 6715 ms**  
> （36 帧 8 线程，同窗口交替 A/B 取最优，**1.33x**），并**第一次快过 inline**（6715 vs 7255）。  
> 默认路径不受影响（`syntax_phase` 为假 ⇒ `recording` 恒假，旧/新二进制默认输出**逐字节相同**）；  
> 调色板 flush 不受影响（`inline_block` 那一支 `recording` 为假，照旧就地预测）。  
> 验证：真素材 20 帧 vs dav1d **0 差**、13 条受控流、intrabc 3 条 + interintra 1 条（默认与  
> row-mt 都逐像素一致且互相逐字节一致）、384 单测、`ME_NO_SIMD` 开/关逐字节。  
> **下一步的瓶颈已经变成"回放"**（`replay_total` 53~184ms）——见第四节的第 1 条。

**结论：row-mt 的"语法阶段"里 `compute_prediction` 仍在执行，结果被丢弃、回放时再算一遍。**  
这是本轮最大的发现，也是 row-mt 一直比 inline 慢 ~28 ms/帧 的根因。

### 证据链

1. `block.rs` `decode_tile`：`BlockCtx::new(..., frame)` **先**用真 frame 建上下文（5061 行），  
   `ctx.row_mt.syntax_phase = true` **后**设（5065-5068 行）⇒ 语法阶段 `self.frame` 是 `Some`。
2. `block.rs` 帧间分支（**2187 行**）：
   ```rust
   if recording { job_idx = Some(self.row_mt.start_block(...)); }
   if let Some(frame) = self.frame.as_deref_mut() {
       compute_prediction(frame, ...)?;      // ← 不判 recording/语法阶段
   }
   ```
   而 `ReplayScratch`/`replay_one` 在回放里**又调一次** `compute_prediction` ⇒ 一遍白跑。
3. 给回放加桩（`ME_RM_NO_REPLAY`，**已删**）后 `sample` 里  
   `compute_prediction`/`block_inter_prediction`/`mc_vert8`/`mc_horiz16`/`block_warp`  
   依然屠榜 —— 它们只能在语法阶段跑。
4. 同口径计时（`AV1_TRACE_SYNMS=1`，只看超块循环）：  
   **inline "语法+重建" 65~~67 ms ≈ row-mt "语法+记录" 58~~61 ms**，而纯语法只要 16~26 ms  
   —— 差出来的 ~40 ms 正好是那遍被丢弃的 MC 预测。
5. 把整块记录关掉时反而**变快**（`reconstructing = frame.is_some() && !recording` 变真、走回内联重建）  
   —— 反证 `frame` 在语法阶段确实是 `Some`。

### 建议修法

把 2187 行那处调用改成"**只在就地重建时**跑"（例如 `if self.frame.is_some() && !recording`  
或干脆在进入语法阶段前把 `frame` 从 `BlockCtx` 里摘掉、回放前装回）。⚠️ 两个注意点：

- **调色板的 flush 路径要保住**：语法阶段遇到 palette 块会 `replay_jobs()` 把已记录工单先回放、  
  再就地重建本块（`block.rs` 2109-2115 行）——它依赖"被回放的块已经把像素写出来"，  
  摘 frame 的改法要保证这条链路仍然成立。
- **必须验逐字节**：`ME_ROW_MT=1`（含 `ME_ROW_MT_PAR=1`）与默认路径解同一素材同帧数，  
  输出逐字节相同；再跑真素材 20 帧对 dav1d 逐像素。

### 预期

语法阶段回到 ~~25~~35 ms。瓦片 = 语法 ~30 + 回放（单线程 85 ms / 4 线程 ~~35 ms）  
⇒ \*\*8 线程整帧 90110 ms（9~~11 fps）\*\*，比现在再快 ~1.4x。做到这一步，row-mt 才第一次  
真正比 inline 快（现在它是慢的：瓦片 110.6 vs 89.3）。

## 三·五、第四刀（已落地 2026-10-03 深夜）：**记录体瘦身** → 1.37~1.49x

> 接第三节。修掉"整帧预测跑两遍"之后，瓶颈确实转到了"记录"：**row-mt 的语法阶段  
> 恒比纯语法慢 ~1.8x**（同帧同窗口对比，`--count=1`，4 线程）。
>
> | 帧       | 纯语法 `tiles` | row-mt `syntax` | 比值   |
> | ------- | ----------- | --------------- | ---- |
> | 1 (KEY) | 47.2 ms     | 73.3 ms         | 1.55 |
> | 2       | 83.5 ms     | 175.1 ms        | 2.10 |
> | 3       | 26.4 ms     | 54.9 ms         | 2.08 |
> | 4       | 18.6 ms     | 53.5 ms         | 2.88 |
>
> ⚠️ 第三节最后那句"下一步瓶颈是**回放**"**被这一刀修正了**：回放（`replay_total`）本身  
> 没有可省的结构性浪费，真正的病根在**记录**。

### 根因①（大头）：`coeffs` 里的 `quant: [i32; 1024]` **栈数组**

`coeff.rs::coeffs` 原本写着 `let mut quant = [0i32; MAX_TXB_COEFFS];`（4 KB 栈数组）：

- `levels_out = None`（纯语法）时，它只喂 `cul_level` 累加 ⇒ **LLVM 能把整个数组优化掉**；
- `levels_out = Some`（row-mt 记录、inline 重建）时，末尾要  
  `out[..n].copy_from_slice(&quant[..n])` ⇒ 数组**必须真物化**。

物化实测 **~2 µs/次**（1024 个 i32 全清，且 LLVM 生成的是逐元素 store、**不是** `memset`）。  
一帧 3 万个变换块 ⇒ **单个大帧 60~90 ms**。

**修法**：把 `quant` 挪进 `TxScratch` 做**复用缓冲**（新增 `TxScratch::quant`，  
`coeffs` 多收一个 `quant: &mut [i32]` 参数），每次只 `fill(0)` **`min(w*h, MAX_TXB_COEFFS)` 个**  
——真素材上平均不到 100，而不是固定 1024。

> 改完后"`out=Some` 让 coeffs 贵 4.5x"的现象**消失**（同帧 coeffs：纯语法 75.1 ms vs  
> row-mt 69.9 ms）。coeffs 现在就是熵解码本身的成本，没有可省的结构性浪费了。

### 根因②（次之）：`push_txb` 按 `w*h` **整段拷**

`levels` 的有效长度其实只有 `min(w*h, MAX_TXB_COEFFS)`：64 尺寸的变换只把系数放在  
**左上 32×32**（`dequantize` 也只读 `min(w,32)×min(h,32)`）。原来按 `w*h` 拷，  
对 64×64 白拷 3/4。**修法**：写入侧与读取侧统一截到 `min(w*h, MAX_TXB_COEFFS)`。

**`scratch_len` 本身不是问题**（实测全帧 3 ms）。三段计时（`dbg_sc` / `dbg_co` / `dbg_pu`）  
是这次临时加的诊断，定位完**已删除**，交付树干净。

### 实测（36 帧 8 线程 row-mt 并行，同窗口交替取最优）

|           | 耗时                                |
| --------- | --------------------------------- |
| 改前（上一轮基线） | 6134 ms                           |
| **改后**    | **4106 ms**                       |
|           | **≈1.5x**（两轮独立测量：1.369x / 1.493x） |

`RMSTAGES` 的 `syntax=`（7 帧样本）合计 **421 → 376 ms**；`levels` 同步下降（首帧 12.5 MB → 10.1 MB）。

### 验收（缺一不可，全过）

真素材 20 帧 vs dav1d：**默认 / 标量（`ME_NO_SIMD`）/ row-mt 顺序 / row-mt 并行 四条路径全部 0 差**，  
且四者两两**逐字节相同** 20/20；13 条受控流全绿；`cargo test --release -p me-decode --lib` **384 passed**。

### 副产物（重要，别再重复试）：**本机 8 线程不如 4 线程**

`ME_THREADS` 扫描（row-mt 并行，36 帧）：1→5973、2→4678、**4→4301**、8→4678 ms。  
本机是 **4 物理核 / 8 逻辑核**：8 线程走超线程，而并行回放的**有效并行度本身就只够 ~8 路**  
（反对角线 `diag = 2*band + col` 分层；1080p 只有 9 带 × 15 超块列 = 135 个作业，  
且"同带左邻列"也算依赖 ⇒ 一条带内是一条链）。worker 利用率：**4 线程 75.6%、8 线程 46.3%**  
（×2 相对物理核 ⇒ ~92%，即**物理核已经跑满**）。

⇒ 想再快只能**减少关键路径 / 串行部分**（帧级流水，或去掉"同带左邻列"这个保守依赖），  
**不是加线程**。本机做 A/B 时 `ME_THREADS=4` 比 8 更稳。

## 三·六、第五刀（已落地 2026-10-04 凌晨）：**"同带左邻列"依赖改为条件性**

> 接三·五节。4 线程的 worker 利用率只有 **74.9%**，并行回放仍有空等。

**根因**：作业图里那条"**同带左邻列**"边（`deps[2]`）本来只为**帧内预测的左邻读取**  
（`intra::derive_edges` 的 `LeftCol`）而画，但切作业时是**无条件**加上的 ⇒  
**一条带内的 15 个列作业串成一条链**，有效并行度被压到"反对角线上 ~8 个"。

**而纯帧间块不读当前帧像素**：预测来自参考帧；`warp` 的样本只从**邻居的 MV** 估计  
（`mv::MotionField::find_warp_samples` 的文档明说"从不读当前块"）；`intrabc` 是唯一例外，  
但它在切作业时就把整帧退回了顺序回放。

**修法**：判据 `reads_cur_frame(b) = !is_inter || (interintra && !no_interintra())`，  
**整列纯帧间时就不画那条边**（`block.rs` 的 `deps[2]` 赋值处 + 新增 `reads_cur_frame`）。

**实测**（36 帧，同窗口交替 A/B）：

| 线程 | 第四刀后    | 第五刀后        | 增量              |
| -- | ------- | ----------- | --------------- |
| 4  | 4947 ms | 4971 ms     | **0.995x（无收益）** |
| 8  | 6010 ms | **5535 ms** | **1.085x**      |

**4 线程总时间没变，但内部变快了**：worker 利用率 **74.9% → 83.8%**、  
回放作用域 **28.3 → 23.7 ms**。4 线程下瓶颈已转移到别处（串行 `syntax` ≈15ms、  
并行 `filters` ≈31ms），收益被掩盖 —— **这一刀是为更多物理核准备的**。

**正确性**：真素材 **60 帧**三路径（默认 / row-mt 顺序 / row-mt 并行）两两**逐字节相同**，  
默认与 row-mt 并行各自 vs dav1d **0 差**。  
⚠️ `ME_TRACE_RACE=1` 探针 0 命中，但它**只覆盖帧内预测的读取路径**  
（`racemap::is_written` 只在 `intra.rs` 被调用），**对帧间块是盲的** —— 本次证据是逐字节。

### 顺带厘清：那"缺口"是 IO，不是解码

4 线程整轮 5083ms / 40 解码帧 = **127 ms/帧**，而 `PHASE` 求和只有 107.9 ms/帧  
（Σtiles 2430 + Σfilters 1607 + ref 190 + out 88）。缺口 **19.2 ms/帧**  
大概率是**落盘 IO**（36 帧 nv12 ≈ 112 MB）。  
⇒ **benchmark 测纯解码时别把 `--out` 的写盘算进去**；真实解码成本看 `PHASE`：  
稳态 **tiles 47.7 + filters 31.0 ≈ 83.9 ms/帧**；尖峰（I/ARF）帧 tiles 218~326ms，  
3 个尖峰帧就占了 Σtiles 的 **32%**。

## 三·七、第六刀（已落地 2026-10-04 凌晨）：record 阶段**直写 arena** → 语法 -11%

> 接三·六节。第五刀只让 4 线程的**回放**变快（利用率 74.9%→83.8%），总时间没动 ——  
> 因为瓶颈在别处。重测指向**仍然完全串行的语法阶段**。

**事实**：row-mt 的语法阶段仍比纯语法贵 **1.67x**（同口径、4 线程稳态）：  
`AV1_TRACE_SYNMS` 的 `syntax=` **20.9 ms/帧** vs `AV1_SYNTAX_ONLY` 的 `tiles` **12.6 ms/帧**  
⇒ **记录税 8.4 ms/帧（占整帧 10%）**。

**根因**：record 阶段有**两次**拷贝 —— `coeffs` 先把系数写进 `TxScratch::levels`，  
`push_txb` 再整段 `extend_from_slice` 进 arena。

**修法**：新增 `RowMtJobs::reserve_levels(n)`（`resize` 预留并清零、返回下标）与  
`push_txb_meta(job)`，让 `coeffs` **直接写 arena** 的那段切片 —— 两次并一次。借用冲突用  
`let mut rec = self.record.take()` / 末尾 `self.record = rec` 放回绕开（NLL 允许：`out` 在  
`coeffs` 之后即亡，`rec` 随后可移动）。

**实测**：`syntax` **20.9 → 18.6 ms/帧（-11%）**；记录税 8.4 → 6.0 ms/帧。

**⚠️ 整帧收益测不出来（本机噪声太大）**：同一二进制 5 轮 `elapsedMs` 波动 ±25%。  
36 帧交替 A/B 给出 4T **1.056x** / 8T **0.972x**，改用中位数则是 4T **1.15x** / 8T **1.02x**  
—— **连符号都不稳定**。用 100 帧的**进程内** `PHASE` 求和对比：tiles 4134→3481（-15.8%）、  
filters 2714→2378（-12.4%）；而 `filters` 与本次改动**毫无关系**却"快了"12.4%，  
说明那是机器漂移 —— **扣掉漂移后 tiles 的真实收益约 3%**，与 `syntax` 的 -11% 量级自洽。  
⇒ **以后量 <5% 的差异一律用进程内指标（`PHASE` / `SYNMS`），别信 `elapsedMs`。**

**正确性**：真素材 **60 帧**四路径（默认 / 标量 / row-mt 顺序 / row-mt 并行）两两**逐字节相同**、  
各自 vs dav1d **0 差**；13 条受控流；384 单测。

### 本轮论证后**放弃**的方向（别再试）：LR 竖直方向按档

`lr.rs` 的水平已按 `LR_CHUNK = 64` 一档，竖直还是 `MI_SIZE = 4` 一行一调，看着像对称的缺口。  
但 `box_filter` 的主体是末段的 `h×w×9` 三位加权（**本来就与 h 成正比、无重复**），竖直档只摊薄  
"初始竖直窗口累加"（`cols×(2r+1)`）与 `restore_block` 的 per-call 开销 —— 按采样比例估算  
合计 **<1%**。代价却是：`SourceSample` 的 `stripe` 要从块级常量改成逐行判定（64 行档必跨 stripe），  
且 `SGR_AB_MAX`/`SGR_F_MAX`/`WIENER_MAX_CELLS` 要从 ~6KB 放到 ~84KB 栈。**不做。**

## 三·十、第九刀（已落地 2026-10-04）：**MC 中间量改 i16 + 一处白扫** → 整帧 −38%

> 接三·九。这一刀前面先做了一次**决定性的对照测量**，结论比优化本身更重要。

### ⚠️ 先校准目标：**硬件完全够，是我们的实现慢 10 倍**

| 解码器 | 本机速度 |
|---|---|
| **dav1d（libdav1d）单线程** | **8.6 ms/帧 = 116 fps** |
| FFmpeg 原生 av1 单线程 | 10.5 ms/帧 = 95 fps |
| **我们（4 线程 row-mt）** | **58.6 ms/帧 = 17 fps** |

⇒ 目标 33 ms/帧 **只需达到 dav1d 的 1/4 效率**，不是"硬件不够"。**任何"做不到"的判断都要先看这张表。**

### 分段定位（单线程，44 帧）

```
纯语法（AV1_SYNTAX_ONLY）  24.6 ms      ← 熵解码本身不慢
重建 = 完整 − 纯语法       137.8 ms     ← 全部差距都在这里
  ├── MC（帧间预测）        70.2   ← 75%
  ├── 残差（反量化+反变换）  21.8
  └── 簿记                  1.2
```

（这三段现在由 `AV1_PHASE_TIMING` 直接打印：`mc=` / `txb=` / `mark=`，见 `replay_one`。）

### 两处修改

1. **`some_use_intra` 的扫描限定在 `MiSize < BLOCK_8X8`**（`compute_prediction`）。
   规范 §7.11.3.1 里 `someUseIntra` **只可能出现在"子 8×8"那一支**（`MiSize < BLOCK_8X8`）；
   ≥ 8×8 的块整块走普通路径，块内不存在"某几个 4×4 是帧内"。
   原实现**无条件扫整块** —— 128×128 要扫 **32×32 = 1024 个 4×4 单元 × 3 个平面**
   （每次一次 `grid.idx` 乘加 + 一次随机读），而真素材里 128×128 有 602 次/8 帧。
   **单线程 tiles 109.8 → 97.9 ms（−10.8%）**。
2. **MC 的横向中间量 i32 → i16**（`PredScratch::mc: Vec<i16>`，`mc_horiz16_8bpc` /
   `mc_horiz8` 输出 i16，`mc_vert8` 读 i16 且**一次 16 个输出**）。
   依据：`round2(Σc·s, inter_round0)` 的值域 **8bit ±4.5e3、12bit 最坏 30712**，都装得下
   i16；而 128 宽的块那一份从 **69 KB 降到 34.5 KB**。dav1d 的 `prep_8tap_c` 用的正是
   `int16_t mid[128*135]`。
   ⚠️ **前提是 `round >= 2`**（调用方只传 `inter_round0` = 3/5）——`round = 0` 时
   `Σ|c|·255 = 36720` 会溢 i16，这条约束已写进内核文档与单测。

### 实测（同窗口交替，100 帧，取中位）

| 配置 | tiles | filters | 合计 |
|---|---|---|---|
| **4T row-mt 并行** | **40.6** | 18.0 | **58.6 ms = 17.1 fps** |
| 4T inline | 63.8 | 15.8 | 79.6 |
| 8T inline | 66.2 | 15.3 | 81.5 |

（三·九节末尾是 94 ms/帧 ⇒ **本轮 −38%**。`row-mt` 仍是**正**优化。）

### ⚠️ 测量方法的一课

用 `stream` 命令做"10 秒窗口数产物"的吞吐测量时，**结果会被"预热"彻底污染**：
同一配置连测 6 次得到 8.1 → 15.1 → 20.9 → 25.9 → 30.6 → 34.4 fps（单调爬升）。
基于它得出的"row-mt 是负优化"是**完全错误**的结论（同窗口交替一测就翻过来了）。
⇒ **吞吐类测量必须"同窗口交替 + 多轮取最优"**，单次窗口数字一律不可信。

## 四、后续两刀（在第三刀之后做）

1. ~~**回放瘦身 / 记录体瘦身**~~ → ✅ **已做（见三·五节）**。  
   病根不是 levels 的**字节数**（拷 12.5 MB 只要 ~1 ms），而是 **`coeffs` 里那个 4 KB `quant`  
   栈数组**在 `levels_out=Some` 下必须物化（~~2 µs × 3 万块 = 几十 ms/帧）。  
   改法：`quant` 挪进 `TxScratch` 复用 + 只 `fill(0)` 有效范围；`levels` 顺带截到  
   `min(w*h, MAX_TXB_COEFFS)`。\*\*实测 1.37~~1.49x。\**
2. **帧级流水**：第 N+1 帧的语法与第 N 帧的滤波重叠（两者都不依赖对方像素）。  
   参考 dav1d `src/thread_task.c` 的任务图；更彻底的是照 dav1d 的 row-mt —— **不物化记录**、  
   逐 SB 行流水（语法线程 + 重建 worker 共享同一份 in-flight 数据）。
3. 内核 SIMD（可选，收益递减但确定）：MC 纵向 16 位通道（`mc_vert8`）、`warp::block_warp`


（全标量）、LR 竖直方向也按档。

## 五、量法与工具（照抄即可）

```sh
cd ZuvRust
pgrep -fl zuvrust          # ⚠️ 先清残留：一个漏杀的 stream 进程能让整轮对比高 30%（踩过）
pkill -f "zuvrust decode"  # 必要时

# 拆账
ME_THREADS=1 AV1_PHASE_TIMING=1 ./target/release/zuvrust decode /tmp/a.mp4 --count=44 --out=/tmp/perf_t --json
ME_THREADS=1 AV1_SYNTAX_ONLY=1 AV1_PHASE_TIMING=1 ...   # 纯语法
ME_ROW_MT=1 ME_TRACE_WAIT=1 ...                          # 每帧 RMSTAGES（语法/回放/记录体体积）+ WAIT 行
AV1_TRACE_SYNMS=1 ...                                    # 两条路径同口径的"超块循环"耗时

# 剖面（本机没有 perf；用 sample）
(ME_THREADS=1 ./target/release/zuvrust decode ... &); sleep 3
PID=$(pgrep -f "out=/tmp/perf_xxx" | head -1); sample $PID 10 -file /tmp/prof.txt
grep -A20 "Sort by top of stack" /tmp/prof.txt

# 交替 A/B（绝对毫秒不可比，只信同窗口的交替+取 min）
for r in 1 2 3; do for b in /tmp/me_base ./target/release/zuvrust; do
  env ME_THREADS=8 $b decode /tmp/a.mp4 --count=36 --out=/tmp/perf_t --json | tail -1
done; done

# 端到端（stream 真实吞吐，写文件再离线数帧；别用慢速 Python 实时读管道，会被自己限速）
./target/release/zuvrust stream /tmp/a.mp4 --json --video=frames --audio=pcm --no-grow > /tmp/s.bin
```

```

**验收清单（每次改动缺一不可）**：真素材 20 帧 vs dav1d 逐像素（`tools/av1ref/refcmp.py`）、
13 条受控流（`tools/av1ref/case_regress.py`）、`cargo test --release -p me-decode --lib`、
`ME_NO_SIMD=1` 开/关逐字节、row-mt 顺序/并行逐字节、最后 `stream` 端到端 fps。

## 六、本轮留下的诊断（正式保留，见 `flags.rs` 文档）

- `ME_TRACE_WAIT=1`：`WAIT ...`（波前空等）+ **`RMSTAGES syntax=… replay_total=… blocks/txbs/levels`**（row-mt 三段拆账 + 记录体体积）。
- `AV1_TRACE_SYNMS=1`：`SYNMS row_mt=… syntax=…ms`（两条路径同口径的语法阶段耗时）。

三个临时桩（`ME_RM_NO_RECORD` / `ME_RM_NO_RECORD2` / `ME_RM_NO_REPLAY`）**已全部删除**，
交付树是干净的（在它们存在时也已确认：无这些环境变量时行为不变，20 帧逐像素一致）。
```

---

# 七、2026-10-04：**单线程目标** + 三个消掉的整帧拷贝 + 测量口径两个坑

## 目标重设（用户要求）

对照物（dav1d / FFmpeg 原生 av1）都是**单线程**跑的 ⇒ 目标改为**单线程达 30fps（33 ms/帧）**。

| 解码器 | 单线程实测 |
|---|---|
| dav1d（libdav1d）| **13.2 ms/帧** |
| 我们 | 本轮开始时 ≈ 76 ms/帧（同窗口相对值 **≈5.8x**）|

⚠️ 上一轮记的"dav1d 8.6 ms"是**多线程**数 —— PyAV 的 `threads` 选项**没作用到解码器**。
要显式设 `codec_context.thread_count = 1` 且 `thread_type = "NONE"` 才是 13.2 ms。

## ⚠️ 坑 1：`PHASE` 的窗口**不包含** `new_pending`

`tiles`/`filters` 只包 `finish_frame`，而网格（`BlockGrid`/`MiGrid`）是在**帧头解析**阶段
就建好的。单帧总预算 = **`setup + tiles + filters + ref + out`**。
本轮把 `setup=` 加进了 `PHASE`（`Av1Decoder::last_setup_ms`）。
**只盯 `tiles+filters` 会把这 3 ms 算漏** —— 我因此误判过"网格复用没收益"。

## ⚠️ 坑 2：本机负载会飘到 load ≈ 10（8 线程机）

同一份二进制在不同时间窗测出 **71 / 96 / 113 / 197 ms**。所以：

* **绝对 fps 一律不可信**（本机现在跑任何基准都带 ~1.5~2x 放大）；
* 只有**同窗口交替 A/B + 多轮取中位**能比较 —— 本轮的结论全是这么得来的；
* 报数必须同时给**相对值**。

## 本轮落地的（每一处都"改动前后逐字节一致"）

| 改动 | 实测 |
|---|---|
| 4 处**热路径裸 `std::env::var`** 改缓存（`AV1_TRACE_CTXW` ×2 在**逐变换块**；`AV1_TRACE_ROWDEP`/`AV1_TRACE_UVR` 在**逐帧内块**）| 单线程 −3% 量级；采样里 `__findenv_locked` 消失 |
| **`BlockGrid`/`MiGrid` 跨帧复用**（新增 `reset_all`，语义与 `new()` 逐格一致）| `setup` **3.07 → 1.02 ms/帧** |
| **参考帧改 `Arc<CurrFrame>`**（原来**每个被刷新的槽**各拷 6.2 MB）| `ref` 4.1 → 1.2 ms/帧 |
| **`cdef::apply` 空操作时返回 `None`**（原来 `frame.clone()` 整帧 6.2 MB）| `ref`/`out` 一起降 |

同窗口 A/B（120 帧单线程、三轮中位）：**197.7 → 179.4 ms = 1.10x**；
A 侧 `setup` 未计入（老二进制不报该字段）⇒ 真实约 **+12%**。

## 仍然差多少（诚实）

机器较闲时的单线程数约 **68~76 ms/帧**，离 33 ms 还差 **2.0~2.3x**；而 dav1d 单线程 13.2 ms
⇒ 我们要落进它的 **2.5x** 以内才算达标。按采样剩余大头是**铺开的**：
MC 28%、内存/分配 12%、LR 10%、去块 10%、CDEF 8%、变换 6%、语法/熵 11%。

**结构性两件（下一轮该动的）**：

1. **`Plane` 是 `Vec<u16>`**：8bit 也按 2 字节/样本走 ⇒ 全链路 2x 内存流量、SIMD 通道数减半
   （dav1d 8bit 走 `uint8_t` + 8bit 内核）。这是"慢 dav1d 好几倍"里最系统性的一条。
2. **`BlockGrid` 每帧仍要写 ≈11 MB 初值**（`palette_colors` 4.1 MB、`delta_lfs` 2 MB、
   `cdef_idx` 0.5 MB……）。想再省必须逐数组论证"本帧必被写满"（`MiGrid` 已有先例：
   `reset_filled` 只清标记，尺寸靠"本帧必被覆盖"）—— 那要逐个读路径过一遍，别拍脑袋。

## ⚠️ 消融语义的一处变化（下次做归因实验必读）

`AV1_NO_CDEF=1` 现在返回 `None`，`lr::apply` 就**借用 `up_curr` 当 `CdefFrame`** ——
规范语义等价（`CdefFrame ≡ CurrFrame`），但"两个缓冲"变成了"同一个缓冲"。
已验：`NO_CDEF` 与 `NO_LR+NO_CDEF` 两条消融下**改前改后逐字节相同**。

---

# 八、2026-10-04（接着推，第二段会话）：两刀内核 + **同窗口对照读到 6.4x**

## 本轮落地（都逐字节可复现，每一处都做了"同二进制内 A/B"）

| 改动 | 位置 | 实测（100 帧、同窗口交替取 min/中位） |
|---|---|---|
| **`box_filter` 竖直滑动和的列 SIMD** | `simd::sgr_cols_update8`（新内核 + 单测）+ `lr.rs` 的 `interior_cols` 分支 | min 108.04 → **105.35**（−2.5%）、中位 109.38 → 107.22（−2.0%） |
| **`warp::block_warp` 内点快路径** | `warp.rs`：取样窗口整段在平面内时省掉**每抽头一次 `clamp` + 带断言的 `get`**，行指针提到 `i2` 循环外 | 107.83/107.70 → **103.69/101.97**（−4~5.5%） |

- 验证：真素材 20 帧对 dav1d **逐像素一致**；13 条受控流无退化；`cargo test --release -p me-decode --lib`
  **385 passed**（新增 `sgr_cols_update_matches_scalar_bit_for_bit`）；`ME_NO_SIMD` 开/关逐字节；
  row-mt 顺序 / 默认 / 并行**三路逐字节相同**。
- 已回填 extra，备份 `/tmp/me-extra-prev-20261004-perf-round17`。

## 同窗口对照（这是唯一可信的绝对刻度）

| 解码器 | 同一时间窗 |
|---|---|
| **dav1d 单线程**（PyAV，`thread_count=1` + `thread_type="NONE"`）| **18.2 / 19.0 ms/帧** |
| 我们（单线程，`PHASE` 和 = tiles+filters+ref）| **119.6 / 117.0 ms/帧** |

⇒ **6.4x**（机器当时 load≈3.6）。目标 33 ms ⇒ 还要 **3.5x**；dav1d 的 2.5x 线是 ≈46 ms。

## 剩余清单（按采样占比，单线程；`sample` 的 top）

| 项 | 占比 | 还能动什么 |
|---|---|---|
| MC（内核 13% + 编排 5.5% + **warp 4%→已削**）| ~19% | 内核已 16 位/向量化；剩下的在编排与 dav1d 级内核 |
| **LR**（SGR 内核 6% + `box_filter` 标量 4% + 其他）| ~12% | 竖直和已 SIMD（本轮）；后面只能整体按单元/SIMD |
| **去块**（`edge4` 5.1% + `loop_filter_edge` 2.9% + `run_line` 2.4% + `filter_strength` 1.1%）| ~11.6% | `run_line`/`filter_strength` 是**逐边**标量门（92% 直接跳过）——可批量化/并入内核 |
| **CDEF**（`block8` 4.2% + `apply` 3.3% + `cdef_filter` 2.2%）| ~9.7% | 剩下的是**方向搜索**（逐块 8 方向 × 20 候选）——dav1d 有 `cdef_dir_avx2` |
| 变换 5% / 语法与熵 ~8% / `project_frame` 2.7% / 内存(memmove+bzero) 4.3% / 尾巴 ~24% | | 摊得很平，单项 <1.5% |

## ⚠️ 三条口径（别再踩）

1. **`--out` 的落盘 IO 不算解码**：`elapsedMs` 120 ms 里 ~14 ms 是写 40 个 nv12（`write`/`stat` 在采样里可见）。
   **解码成本看 `PHASE` 求和**（本轮 117~120 就是它）。
2. **dav1d 的对照必须显式设 `thread_count=1` + `thread_type="NONE"`**（PyAV 的 `threads` 选项不生效）。
3. **<5% 的改动一律"同二进制内 A/B"**（临时 env 门，测完即删）；两进制交替也行，但绝对毫秒一律不可比。

## 下一段会话的第一刀（结构性，唯一能到 2~3x 的方向）

**`Plane` 的 8bit 走 `u8`**（另一会话在"七"节已经论证过）：全链路 2x 内存流量、SIMD 通道数翻倍。
落地方式（不要半途而废）：

1. `Plane` 增加 u8 存储（8bit 时用 `Vec<u8>`，10/12bit 仍 `Vec<u16>`），`get`/`set` 双态；
2. **一次性**把读取该平面的所有热点换成 u8 内核（MC 三个、deblock、CDEF、SGR、intra、warp、nv12），
   其余（标量、grain、upscale）走 `get/set` 转换 —— 中间不能留"半 u8 半 u16"的混合态；
3. 验收照旧：20 帧逐像素 + 13 受控流 + 385 单测 + `ME_NO_SIMD` + row-mt 三路逐字节。

预估 **1.3~1.5x**；再叠上表里的去块/CDEF 标量簇（~5%）与 MC 编排（~3%），才有机会摸到 46 ms（dav1d 2.5x）。

---

# 九、2026-10-04：u8 平面重构**开工**（访问器层已落地；内核迁移待续）

## 已落地（默认路径零变化，已验证）

1. **`Plane` 双态存储**（`intra.rs`）：
   - `enum PxBuf { U8(Vec<u8>), U16(Vec<u16>) }`，`data` 字段改成**私有**（编译器/`grep` 都能把站点找全）；
   - 访问器：`px8()` / `px8_mut()` / `px16()` / `px16_mut()`（取错宽度**当场 panic**，不静默）、
     `is_u8()`、`sample_count()`；`get`/`set`/`Clone`/`Drop` 全部双态；
   - `pool.rs` 增加 u8 池（`take_u8`/`give_u8`，同一套"按长度复用 + 上限"约定）；
   - `Plane::new` 在 `bit_depth <= 8 && ME_PLANE_U8=1` 时才用 u8 —— **默认关**（见 `flags::plane_u8`）。
2. **32 个生产站点 + 20 个测试站点**已从 `X.data` 改成显式 `X.px16()`/`px16_mut()`：
   行为零变化（u16 路径原样），但**每个需要双态化的地方现在都能 `grep px16(` 一眼列出来**。

**验收（u8 关 = 今天的行为）**：真素材 20 帧 vs dav1d 逐像素一致；`cargo test --release -p me-decode --lib`
**385 passed**；与上一版二进制 8 帧输出**逐字节相同**。已回填 extra。

## 为什么没在本段一次做完（诚实核算）

按内核逐个核算 u8 的收益（单线程 ~105 ms 的盘子里）：

| 受益面 | 占比 | 期望 | 说明 |
|---|---|---|---|
| 去块 / CDEF / LR 的**滤波器内核** | ~13% | **1.5~2x** | u8 通道数翻倍（`pminub`/`pavgb` 一路）；这是大头 |
| memmove / bzero / 整帧拷贝 | ~4.3% | ~2x | 字节数减半 |
| MC 内核 | ~13% | ~1.15x | 它**是 mullo 累加型、ALU 密集**：u16 的 16 通道已近上限，u8 主要省加载；真收益要配 `pmaddubsw` 型自研内核（饱和坑见下）|
| intra / transform 标量、warp 内点 | ~6% | ~1.1x | |
| **合计** | | **~8~12%** | **不是 1.3~1.5x**（与"七"节的乐观估计不同，这里按内核结构重算过）|

⇒ 它是**必要的战略性一步**（后续 8-bit SIMD 的前置），但单靠它到不了 33 ms。
**同时**还有一块同量级、低风险的：去块标量簇（`run_line` 2.4% + `loop_filter_edge` 2.9% + `filter_strength` 1.1%）
+ CDEF 方向搜索（3.3%）+ MC 编排（5.5%）+ `project_frame` 2.7% ≈ **17% 可拿**。

## 下一段会话的精确清单（照做即可）

1. **内核双态化（推荐 trait 泛型，别手抄两份）**：在 `simd.rs` 加
   ```rust
   trait PxKind: Copy { unsafe fn load16_i16(p: *const Self) -> __m256i;
                        unsafe fn load8_i32(p: *const Self) -> __m256i;
                        unsafe fn store16_clamped(p: *mut Self, v: __m256i, max: i32);
                        unsafe fn store8_clamped(p: *mut Self, v: __m256i, max: i32); }
   ```
   （u8 用 `_mm_loadu_si128`/`_mm256_cvtepu8_epi16`、u16 用 `_mm256_loadu_si256`；存储侧 u8 走
   `_mm256_packus_epi16` + 16 字节写），然后把下列**内核逐个泛型化**（每个只改 load/store/widen 十来行）：
   `mc_horiz_rows_8bpc`、`mc_horiz16_8bpc`/`mc_horiz8`、`mc_shift8`、`mc_delta8`（读侧）；
   `store_row_clamped`、`blend_avg8`、`blend_dist8`、`blend_masked8`、`blend_masked_u16in8`（写侧）；
   `deblock_edge4`（`PlaneView` 也要双态）、`cdef_block8`、`sgr_final_8`、`sgr_cols_update8`（读写侧）。
2. **32 个站点**（`grep -n "px16(" crates/me-decode/src/video/*.rs`）逐个改成"按平面存储 dispatch"：
   热站点（inter 的 7 条写回 + 5 处内核调用、deblock 的 `PlaneView`、cdef 的 `plane_ref`、lr 的
   `rows_ptr` 两张表 + `RESTORE_NONE` 拷贝、warp 内点）走泛型内核；冷站点（`upscale`、`grain`、测试）
   直接走 `get/set` 循环（慢一点点，正确性优先）。
3. `coeff.rs::inverse_transform_add(pred: &[u16], dst: &mut [u16])` 也要泛型化（或按存储分两支）。
4. **验收**（u8 打开）：`ME_PLANE_U8=1` 下 20 帧逐像素 + 13 受控流 + 385 单测 + `ME_NO_SIMD` 开/关 +
   row-mt 顺序/默认/并行逐字节；再"同二进制内 A/B"（`ME_PLANE_U8` 就是现成的门）量收益。
5. 收益达标（≥8%）再翻默认；否则保留开关、如实记账。

⚠️ **`pmaddubsw` 的饱和坑**（想给 MC 写 dav1d 式内核时必读）：它把**相邻两抽头的加权和**饱和到 i16，
而 AV1 的半像素相位（如 `[-4,12,-24,80,80,-24,12,-4]`）相邻两抽头之和可达 **160**，`255×160 > 32767`
会饱和出错。dav1d 用"样本先减 128（有符号化）+ 结果补 `128·Σk = 16384`"绕开 —— 正好就是我们
`HORIZ16_BIAS` 那套；要用 i8 系数就得先把这条恒等式推平。

---

# 十、2026-10-04（续）：**u8 镜像实验**（负结果，已全部回退）+ 当前状态

## 假设与做法

微基准给出一个刺眼的对照：`mc_horiz16_8bpc` 在**缓存内**只要 **1.10 ns/像素**，
而在真素材的 1080p 参考平面上等效 **3~7x** ⇒ 猜"MC 的一半成本是访存（带宽/足迹）"。
于是做了个**不动存储层**的验证版：给参考帧建一份 **u8 镜像**（`Plane::mirror`，
每样本 1 字节），只把 MC 读侧的三个内核换成 u8 版
（`mc_horiz_rows_u8` / `mc_shift8_u8` / `mc_horiz8_u8`：每个抽头
`_mm_loadu_si128` + `cvtepu8_epi16`，下游的偏置累加/取整一字不改）——
按设计两条路**必须逐字节相同**（`ME_NO_U8MIRROR=1` 就是 A/B 门）。

## 结果：**慢 ~1.7%**（三轮一致），按纪律**全部回退**

| 轮次 | 不建镜像（u16） | 建 u8 镜像 |
|---|---|---|
| 1 | 99.16 ms | 101.63 ms |
| 2 | 102.40 ms | 104.67 ms |
| 3 | 99.74 ms | 100.00 ms |

**结论（这条要记住）**：MC 不是**带宽**受限，而是**散乱访问的延迟/TLB**受限——
"每样本减半字节"没兑现，反而每个抽头多了一条**零扩展指令**（u8 load 16B + widen
vs u16 一次 32B load），净亏。镜像本身的构建也要 ~1 ms/帧。

⇒ **全存储 u8 翻转时，MC 内核不要改 u8**（收益为负，已实测）；u8 剩下的理由只有
**滤波器内核的通道数**（去块/CDEF/变换：u8 通道翻倍，`pminub`/`pavgb` 一路）
与**拷贝减半**，量级从早期的"1.3~1.5x"下调到 **~5~7%**。

## 回退后的状态（全绿）

`Plane` 只保留 [`PxBuf`] 双态**访问器层**（`px8/px16/...`、`data` 私有、u8 池、
`ME_PLANE_U8` 默认关）。真素材 20 帧 vs dav1d 逐像素一致；385 单测；
13 条受控流无退化；**与建镜像时的输出逐字节相同**（证明回退干净）。

## 这一段之后还剩什么（都是小刀，按性价比排）

| 项 | 现在占 | 做法 |
|---|---|---|
| 去块 `run_line` 的逐边扫描（~78 万次/帧）| 2.6% | 逐边 gate 已很瘦；试过"位图预扫"推导后**不划算**（建位图的成本 ≈ 现扫描）⇒ 要动就得**在块解码时顺手登记边表** |
| CDEF `cdef_direction` 的 8×8 逐样本 `get()` | ~3% 里的一半 | 改整行切片读（每样本省一次断言+乘法），注意保持 i32 逐位 |
| `project_frame` 的 `field.set` 边界检查 + 2.7% | 3.2% | 位置有效性每格算一次、set 的检查外提 |
| `mv_scale` 每次两个除法 | ~0.5% | 每帧按 (ref, plane) 预计算 scale（不缩放时就是 1<<14）|
| `BlockGrid` 每帧 ~11 MB 初值 | 1~2% | 逐数组论证"本帧必被写满"（`MiGrid::reset_filled` 是先例）|

## 十·五、这段还落了一刀：CDEF 方向搜索改整行读

`cdef_direction` 的 8×8 取样循环原来**每样本一次 `Plane::get`**（一次 debug_assert
+ 一次 `y·stride + x` 乘法），现在**逐行取切片、8 个样本一次读全**（`partial` 的
累加顺序变了，但全是 i32 整数加、逐位不变）。

- 实测（100 帧、两进制交替）：90.33/92.55/90.01 → **89.69/88.90/90.13**（min −1.2%）；
- 验收：20 帧逐像素 + 与改前**逐字节相同** + 385 单测 + 13 受控流 + `ME_NO_SIMD` 开/关
  + row-mt 顺序/并行**三路逐字节相同**。

**本段（第八~十节）合计**：LR 列 SIMD（+2.5%）、warp 内点快路径（+4~5%）、
CDEF 方向整行读（+0.7~1.2%），**都按"同二进制/两进制交替 A/B"量的**；
u8 镜像（负结果，已回退）。当前单线程 `PHASE` 和 ≈ **89~105 ms/帧**（机器负载相关），
同窗口 dav1d 单线程 18~19 ms ⇒ **还差 ~5x**。

## 十·六、这段又试了两处（一处留、一处回退）

| 试的 | 结果 | 处置 |
|---|---|---|
| `mv_scale` 的**同尺寸快路径**（无超分时 `x_scale = y_scale = 1<<10`，代数上恒等；省每次调用两个 i64 除法）| 测不出（<1%，噪声内）但**严格更少操作**、10 行 | ✅ **保留**（逐字节一致） |
| `BlockGrid` 的块语义 stamp 改**整行 `fill`**（把 1024 次逐格闭包换成 32 次整行填）| **测不出**（118.95 → 118.93，三轮噪声内） | ❌ **回退** —— 说明 `stamp` 的 1.9% 不是闭包开销，而是**那 ~11 MB/帧的数组写入本身**（要省就得逐数组论证"本帧必被写满"，见下的清单） |

验收（两处都在场/回退后各跑过）：真素材 20 帧**逐像素一致**、385 单测、13 条受控流、
row-mt 顺序/并行**逐字节相同**、与相关中间版本的输出**逐字节相同**。已回填 extra。

**剩余小刀（按性价比）**：① `BlockGrid` 逐数组论证"必被写满"免初值（1~2%）；
② 去块逐边扫描改成"块解码时登记边表"（~2%，要动 block.rs 的簿记）；
③ `project_frame` 的 5 次投影（~1%）。三者之外，再往下就是 **dav1d 级的内核重写**
（8 位通道滤波内核 = u8 全存储翻转 ~5~7%，或把 MC 的 mullo 链换成 `pmaddubsw` 型
——注意 `[-4,12,-24,80,80,-24,12,-4]` 的相邻两抽头和 = 160 会撞 `pmaddubsw` 的 i16 饱和，
dav1d 用"样本 −128 + 补 16384"绕，正是我们的 `HORIZ16_BIAS` 那套）。

## 十·七、小刀①落地：`BlockGrid` 逐数组免初值（2026-10-04）

### 做法：逐数组论证"本帧必被写满"

`decode_block` 的两个收尾 `stamp` 是**无条件**的（第一段在 `residual()` 之前、
第二段在结尾；见那两处注释），而块铺满整张 `mi_rows × mi_cols` 网格 ⇒ 任一跨块读
（上/左邻居、OBMC 候选、调色板缓存、复合上下文……）落到的格子，都已被**本帧更早**
的 `decode_block` 盖过。于是这 9 个数组的初值是纯白写，只留 `ensure_len`（长度变了
才分配 + 填一次）：

* 第一段收尾：`y_modes`、`ref_frames`；
* 第二段收尾：`skips`、`skip_modes`、`is_inters`、`inter_tx_sizes`、`delta_lfs`、
  `palette_sizes`、`palette_colors`。

合计省 ≈ **6.1 MB/帧**的写（`palette_colors` 32 B/格是大头）。

**保留初值的**（读它们的地方不受"本格必被写"保护）：
* `uv_modes` —— 子 8×8 的色度归"右下角那个 4×4"，其余格可能一直没被写，而
  `get_uv_mode` 会读；
* `mvs`/`comp_*`/`wedge_*`/`mask_types`/`interp_filters`/`interintra_modes` ——
  只在 `is_inter` 时写；读它们的地方都有 `ref_frames` 门（`obmc_blend`：候选必须
  `ref_frames[gi][0] > INTRA_FRAME`），但"门与写同源"这条依赖链太隐晦，留着（~2 MB）；
* `tx_types`/`lf_tx_sizes`/`cdef_idx` —— 按变换块/超块写，且**整帧解完之后**被
  帧级滤波与 CDEF 读。

### ⚠️ 画外音：顺带查实的一处**语义**（不是性能）

`seg_ids`/`prev_seg_ids` 也保留了初值，因为 `latch_prev_segment_ids` 排在
`reset_all` **之后**、而 `reset_all` 会把 `seg_ids` 清 0 ⇒ 现在 `prev_seg_ids`
**恒为全 0**（等价 libaom 的 `last_frame_seg_map = NULL` 分支，`decodemv.c`
`get_predicted_segment_id`）。dav1d 取的是**主参考槽存下来的 segmap**
（`decode.c`：`c->refs[refidx[pri_ref]].segmap`，另有"尺寸必须相等"检查），
libaom 取 `cm->prev_frame->seg_map`。⇒ `temporal_update`（或 `!update_map`）的流上
我们**可能**与参考实现分歧；真素材 60 帧逐像素一致只说明**现用素材没撞上**。
**要修得先做"按参考槽存 segmap"**（`RefFrameStore` 每槽带一份 map + 尺寸检查 +
`!update_map` 时整份拷贝），别在性能刀里顺手改。

### 实测（同窗口交替，5 轮，44 帧，单线程 `PHASE` 求和）

| 轮 | 旧 `setup` | 新 `setup` |
|---|---|---|
| 1 | 87.9 | 56.4 |
| 2 | 88.5 | 63.9 |
| 3 | 90.6 | 53.2 |
| 4 | 92.0 | 60.3 |
| 5 | 85.1 | 50.1 |

`setup` **−0.65~0.75 ms/帧**（旧 1.9 → 新 1.2；5 轮区间不重叠）；`tiles+filters`
两进制 **min 持平**（5835.8 vs 5829.9）⇒ 整帧净收益 ~0.5%，不吹。比预估的"1~2%"低，
原因大概是 6 MB 的写本来就被 store buffer 吃掉大半。

⚠️ **别对 `a.mp4` 跑无边界的 `stream`**：它是 20 分钟 / ≈3.6 万帧，落盘要 36 GB
（这一轮踩了，幸好及时杀掉）。

### 验收（全绿）

真素材 20 帧与改前**逐字节相同**（21 个 nv12 全 `cmp`）、与 dav1d 逐像素一致；
13 条受控流 + 调色板 3 条 + intrabc 3 条逐条 0 差；385 单测；`ME_NO_SIMD` 开/关、
row-mt 默认/顺序/并行**三路逐字节相同**。回填 `extra/darwin/x64/engine/zuvrust`
（md5 `2ab938de5d13c456c11e13a3d84614b4`，备份 `/tmp/me-extra-prev-20261004-round20`）。

**剩余小刀**：② 去块逐边扫描改"块解码时登记边表"（~2%）；③ `project_frame` 5 次投影（~1%）。

## 十·八、Wiener 两趟 SIMD（2026-10-04）—— filters −9.4%

### 先量天花板，再动手

`wiener_filter` 是**纯标量**的两趟 7 抽头 FIR（横向 `(h+6)·w·7` 次乘加、纵向 `h·w·7`），
而真素材 LR 类型全是 **wiener(1)/switchable(3)**（`AV1_TRACE_LR`：`types=[1,1,1]` /
`[3,3,3]`，只在 q 更高时才 `uses=false`）⇒ 它每帧都在跑。它是被内联进
`lr::apply::{{closure}}` 的（采样里没有独立符号），所以先用**临时桩**
（`AV1_NO_WIENER=1` 跳过算术只写行）在 `sample` 里量出：closure **401 → 181 样本**，
即 **wiener ≈ 3.2%** 的引擎时间（桩会让后续帧解出垃圾、`tiles` 也会漂 —— 只能看
同一个符号的占比，不能看整帧求和）。

### 改动（`simd.rs` 两个内核 + `lr.rs` 派发）

* `wiener_horiz_rows`：整档 `(h+6)×w`，8 列一组 `cvtepu16_epi32 + vpmulld + vpaddd`，
  `rnd` 取整、`min/max` 钳位后写 i32 中间量。**只在列窗口不触边时走**
  （`x-3 >= 0 && x+w+3 <= plane_end_x`，此时列解析表等差，可连续读）；
  触边档走原标量。尾组（`w` 非 8 倍）**重叠重算**，省掉标量尾巴。
* `wiener_vert_row`：每输出行 7 个 i32 行段 × `vpmulld/vpaddd`，`rnd` 后钳到
  `[0, max]` 写回 `row[..w]`（`set_row` 照旧）。
* 与标量**逐位一致**的论证：`vpmulld`/`vpaddd` 就是 i32 回绕乘加（i32 加法在回绕
  意义下可交换结合 ⇒ 标量的累加顺序无关紧要）、`rnd` 与 `round2` 同式、
  `vpminsd/vpmaxsd` 与 `clamp` 同式。

### 实测（同二进制内 `AV1_WIENER_SCALAR` 门，3 轮交替，44 帧单线程 `PHASE`）

| 轮 | SIMD `filters` | 标量 `filters` |
|---|---|---|
| 1 | 1856.5 | 2103.2 |
| 2 | **1843.0** | 2149.6 |
| 3 | 1975.7 | 2043.0 |

min：2043.0 → 1843.0 = **−200 ms / 47 帧 = −4.3 ms/帧**（filters −9.4%，
整帧约 −4.7%，三轮区间不重叠）。采样复测：`lr::apply::{{closure}}` **401 → 221 样本**。

### 验收（全绿）

新内核单测 `wiener_kernels_match_scalar_bit_for_bit`（8/17/64 列 × 尾组 × 8/12bit
取整 × clamp 上下界，逐位比对）；真素材 20 帧**与改前逐字节相同**、与 dav1d 逐像素
一致；`ME_NO_SIMD` 开/关、row-mt 默认/顺序/并行三路逐字节；13+3+3 条受控流；
386 单测。回填 `extra/darwin/x64/engine/zuvrust`（md5 `10e000e2…`，
备份 `/tmp/me-extra-prev-20261004-round21`）。

**小坑（写测试时踩的）**：`rows_ptr[k]` 是**帧基址**、`offs[k]` 才是行偏移
（内核自己相加）—— 测试里把指针指到 `src[o..]` 又加 `offs` 就双重偏移了。

## 十·九、去块扫描改"按变换块跳"（2026-10-04）—— filters 再 −2.5 ms/帧

`run_line` 原来沿一条线**逐格**（step = `1<<sub_x/y`，1080p 一帧约 **39 万次**）走，
每格进一次内联的 `lf_gate`，**92% 在这里被丢掉**（只有约 3 万条边真滤波）。
采样里 `run_line`(217) + `loop_filter_edge`(205) ≈ **5.6%** 的引擎时间。

**改动**：`isTxEdge` 只可能在"这一格是自己所在变换块的第一格"处成立，而
`lf_tx_sizes` 每格存的正是它所属变换块的大小（`coeff.rs` 按块的 span 铺）⇒
**从块首按 `TX_WIDTH/HEIGHT` 跳**，永远恰好落在下一个块首上，一帧只看几万个位置。
一致性论证：`[s, s+W(s))` 整段属于同一个变换块、段内没有别的块首；万一跳到非块首，
`loop_filter_edge` 里的 `lf_gate` 也会原样丢掉 —— **实际滤波集合逐边不变**。

### 实测（同窗口交替 6 轮，44 帧，单线程 `PHASE` 的 `filters` 求和）

| 轮 | 跳扫前（w22 的上一版） | 跳扫后 |
|---|---|---|
| 1 | 1894.2 | 1907.3 |
| 2 | 1988.4 | 1812.3 |
| 3 | 1889.5 | 1780.9 |
| 4 | 1880.9 | 1762.6 |
| 5 | 1920.7 | 1804.5 |
| 6 | 1908.5 | 1736.7 |

中位 **1901.5 → 1784.4 = −117 ms / 47 帧 ≈ −2.5 ms/帧**（filters −6%，整帧 ~2.4%；
6 轮里 5 轮有利，r1 劣属噪声）。验收：20 帧**逐字节相同** + dav1d 逐像素一致 +
13+3+3 受控流 + 386 单测 + `ME_NO_SIMD` 开/关 + row-mt 三路逐字节相同。
回填 `extra/darwin/x64/engine/zuvrust`（md5 `4ada343b…`，
备份 `/tmp/me-extra-prev-20261004-round22`）。

## 十·十、`project_frame` 提表 + **`setup` 口径补齐**（2026-10-04）

### 先补齐口径：运动场估计原来**不在任何 PHASE 里**

采样里 `mv::project_frame` 常年占 300 个左右的样本，而它既不在 `tiles`/`filters`
（那两段只包 `finish_frame`）、也**不在 `setup`**（`last_setup_ms` 在
`motion_field_estimation` **之前**就结算了）⇒ "单帧预算 = setup + tiles + filters"
一直漏掉这块。现在把它算进 `setup`。补齐后实测（60 帧）：
**`setup` 平均 4.18 ms/帧、稳态 2~6 ms、首帧 15 ms** —— 也就是运动场估计 ≈ **3 ms/帧**
（占单帧 ~3.5%）；这一段以前是隐形的。

### 改动（`mv.rs::project_frame`，纯提表/提前判，逐位不变）

* `ref_offset` / `pos_valid` **只跟 `src_ref` 有关**（格子里的取值只有 8 种）⇒
  两张 8 项小表在循环外算一次（原来每格重算 `get_relative_dist`：几个分支 + 移位）；
* `field.set` 的边界检查+目标帧平面步长提到循环外：**越界就整格跳过** ——
  原来 7 个目标帧各判一次、且越界时 7 次 `mv_projection_clipped` 全是白算；
* `denom_clip = ref_offset.max(1)`（`pos_valid` 已保证 `|ref_offset| ≤ 31`，与
  原来的 `.min(MAX).max(1)` 等价）；`MotionField::set` 随之删除（没了调用方）。

### 实测

`sample` 同一符号：`project_frame` **318 → 243 样本（−24%）≈ −0.6 ms/帧**；
墙钟 A/B（44 帧、**不带 `--out`** 以避开落盘 IO、5 轮交替）：中位
**5.12 → 4.97 s ≈ −0.8%**（噪声 ±1%，两进制 min 持平）。验收：20 帧**逐字节相同**、
dav1d 逐像素一致、13+3+3 受控流、386 单测、`ME_NO_SIMD`/row-mt 三路逐字节。
回填 `extra/darwin/x64/engine/zuvrust`（md5 `8d3de2e0…`，
备份 `/tmp/me-extra-prev-20261004-round23`）。

⚠️ 注意：`setup` 的**读数口径变了**（含运动场估计），其后与本文档前面的
"setup 1.9 → 1.2"（不含）**不能直接比**。

## 十·十一、`predict_intra_txb` 里的**诊断白算**（2026-10-04）—— ~2%

`intra::predict_intra_txb` 里有一段**只为 `AV1_TRACE_UVR` 打印服务**的代码：
2 个 `vec![0i32; span+2]` + 一遍 `derive_edges` + 2 个 `to_vec()` 存
`details_above/left` —— 而 `details_*` 的**唯一**去处就是那条 `println!`。
它是**无条件**算的：每个帧内变换块 4 次堆分配。改成 `if uvr.is_some()` 才做
（`uvr` 就是原来那两次 `trace_uvr()` 调用合一的缓存值）。

### 实测与一条**方法教训**

单独 A/B（4 轮、轮内换序、44 帧、不带 `--out`）：**4/4 轮更快**
（min 4.87 → 4.76 s、中位 4.93 → 4.84 s ≈ **−1.8~2.3%**）。
⚠️ **之前的 A/B 都让旧二进制先跑**，而"每轮第二个进程"疑似系统性偏快
（dbjump 那两轮对同一对二进制给出 −3% 却也 −0%，顺序偏差没被控制）——
**以后每轮都要换序**（本轮已改）。

三刀合计（去块跳扫 + `project_frame` 提表 + 本刀）的 6 轮换序 A/B：
中位 **5.02 → 4.86 s（−3.2%）**、min 4.98 → 4.74（−4.8%）。

验收：20 帧**逐字节相同** + dav1d 逐像素一致 + 13+3+3 受控流 + 386 单测 +
`ME_NO_SIMD`/row-mt 三路逐字节。回填 `extra/darwin/x64/engine/zuvrust`
（md5 `3aa46171…`，备份 `/tmp/me-extra-prev-20261004-round24`）。

**下一刀候选（按采样）**：`cdef::apply` 闭包 ~3%（每块 offs/taps 重算）、
`lr::apply` 闭包 ~2.7%（box_filter/sgr 的标量段）、`bzero` ~2%（散在逐块分配：
`block_inter_prediction`/`lr::apply`/`box_filter`/`transform_block` 各 10~20 样本）、
`compute_prediction` ~4.7%（MC 编排）。

## 十·十二、`COMPOUND_DIFFWTD` 掩码：向量化 + 跨块复用（2026-10-04）—— tiles −6%

### 先量频率（决定值不值得动）

用 `AV1_TRACE_BLKATTR=1` 数出来（12 帧样本）：帧间块 13343 里 **DIFFWTD 2269 个（17%）**，
面积合计 7.93M 像素 ⇒ **每帧 189 个块、66 万像素**。于是那段代码每帧要：

* `vec![0i32; w*h]` × 189 ⇒ **2.58 MB 的分配 + 清零**；
* 逐像素标量算掩码（`|Δ|` → `Round2` → `/16` → `clamp` → 取补）≈ 66 万像素
  —— 它就是 `compute_prediction` 自占 4.7% 的主体。

### 改动

1. **缓冲复用**：`PredScratch` 新增 `mask: Vec<i32>`（**跨块复用**；只在长度需要增长时
   `resize`，之后 `truncate` 到 `n` —— `truncate` **不写内存**）。
   `compute_prediction` 里那个 `Option<Vec<i32>>` 随之删掉、`predict_inter` 少一个参数。
   **平面循环的顺序**保证"plane 0 先生成、色度再读"。
2. **向量化**（`simd::diffwtd_mask8`）：把两步右移并成一步 ——
   `Round2(x, s) / 16 == (x + (1 << (s-1))) >> (s + 4)`
   （**非负**整数下两层向下取整可以并）⇒ 每 8 像素 `sub / abs / add / srav / min / max`。
   标量尾巴照旧。新单测 `diffwtd_mask8_matches_the_two_step_formula` 把
   8/10/12bit × 两个 `ipr` × 实际值域（含撞 64 上限、`mask_type = 0/1`）**逐点**与两步式比对。

### 实测（同窗口交替 **6 轮、每轮换序**，44 帧单线程 `PHASE`）

| | 6 轮 tiles | 中位 |
|---|---|---|
| 改前 | 44 / 49 / 46 / 51 / 47 / 47 | **47.2** |
| 改后 | 42 / 45 / 44 / 44 / 47 / 44 | **44.4（−6.0%）** |

`filters` 25.9 → 25.9（不动）⇒ 整帧 76.9 → **74.1（−3.6%）**；6 轮里改后每轮 ≤ 改前（5 胜 1 平）。

### 验收（全绿）

真素材 40 帧与改前**逐字节相同**、与 dav1d 逐像素一致；**`ME_NO_SIMD` 开/关逐字节相同**
（新内核与标量路径逐位一致）；row-mt 默认/顺序/并行三路逐字节；13 条受控流；**387 单测**。
回填 `extra/darwin/x64/engine/zuvrust`（备份 `/tmp/me-extra-prev-20261004-round25`）。

### 同段会话的另外两条结论（都是口径）

1. **多线程只买到 ~1.15x**（同窗口交替 2 轮）：1T inline 51.6/55.9、4T 49.0/51.4、
   8T 50.5/51.7、**8T row-mt 并行 44.6/47.9** ⇒ 最好也只有 ~22 fps，
   **"流畅播放"光靠线程也不够**。⚠️ 当刻本机 `load ≈ 44`，并行度被吃掉了；
   换空闲窗口重测会有不同的绝对数，但相对关系（row-mt 并行最优）应当保持。
2. **`cdef_direction` 每 64×64 只调一次**（`cdef_block` 里那一次），**不是**每 8×8 ⇒
   方向搜索不是 CDEF 的成本；成本在 `cdef_filter` 的逐像素抽头逻辑 + `cdef_block8` 的 u16 数据。
   一句话：**filters 上我们是"半标量 + u16"，dav1d 是"全 SIMD + u8"。**

### 这一刀的可复制套路（值得再用）

**"逐像素标量 + 每块 `vec![]`"是这套引擎里最值钱的两种坏味道**，找它们的方法：

1. 用采样找"自占时间高但看不出在算什么"的符号（这次是 `compute_prediction`）；
2. 拿 `AV1_TRACE_BLKATTR` **数出这类块的真实频率与面积**（不要凭感觉"应该不多"）；
3. 数出来的量 × 每像素成本，就能判断值不值得动（这次 66 万像素/帧 → 值得）；
4. 化简算术（并移位）＋ 复用缓冲（`truncate` 不写内存），并**用一条单测把化简钉死**。

---

# 十一、2026-10-04（第三段会话）：第二十七刀 —— CDEF 内核 **i16 通道 + 两行打包**、帧缓冲池按字节封顶

## 先把目标与水位重新量准（本机 load ≈ 5~6 的窗口）

| | 单线程 |
|---|---|
| **dav1d**（PyAV，`thread_count=1` + `thread_type="NONE"`）| **11.6 ms/帧** |
| 我们（`tiles 28.5 + filters 14.4 + setup 4.4 + out 1.3`）| **≈ 48.7 ms/帧**（好窗口 38.6）|

⇒ **4.2x**；目标 33 ms ⇒ 还差 **1.48x**。采样里 CDEF 是 filters 最大的一块：
`cdef_block8` **5.2%** + `cdef::apply` 闭包 3.5% + `cdef_filter` 3.3% ≈ **12%**。

## ⚠️ 本轮最该记住的一条：**这台机器上 ±5% 以下的改动不可测**

同一对二进制、8 轮换序交替，`PHASE` 总和能从 **44.8 漂到 90.6 ms（±40%）**；
另一段窗口里旧版 min/median 是 49.07/51.05，另一段是 52.51/83.48。**墙钟只能信
"两版都稳定"的窗口。** 两种可靠替代（本轮都用了）：

1. **同进程交替的内核微基准**（`#[ignore]` 的 `bench_*`，`--nocapture` 手工跑）——
   机器负载对两者的影响在同一次调用里几乎抵消。
   ```sh
   cargo test --release -p me-decode --lib bench_cdef -- --ignored --nocapture
   ```
2. **确定性计数** —— 命中率 / 访存次数 / 分配次数这类**不随负载变**的量。
   本轮新增 `AV1_TRACE_POOL=1`（池的命中/落空/池内字节）与 `ME_POOL_MAX_BYTES`（上限旋钮）。

## 刀 1：CDEF 内核 i32 → **i16**，一次一行 → **一次两行**

`simd::cdef_block8`（8 个 i32 通道）→ `simd::cdef_pair16`（**16 个 i16 通道** =
2 行 × 8 列，低 128 位 = 行 `r`、高 128 位 = 行 `r+1`）。两个变化叠起来：
通道数翻倍；每个候选从"128 位载入 + `cvtepu16_epi32` 零扩展"变成
"两次 128 位载入 + 一次 `_mm256_set_m128i`"。

**i16 装得下的界**（12 位最坏，8 位更小）：

| 量 | 界 |
|---|---|
| 样本 / `diff` / `mag` | ≤ 4095 |
| 阈值 `pri/sec_strength << coeff_shift` | ≤ 240 |
| `constrain` 结果 | ≤ 240 |
| `acc = Σ_{12} tap·val`（`tap ≤ 4`）| ≤ 11520 |
| `cur + ((8 + acc - adjust) >> 4)` | ⊆ [-719, 4815] |

**顺带砍掉同一段热路径上的两处白活**：

* `cdef_filter` 里 `let mut out = [0i16; 64]` 是**每次调用**一次 128 字节清零；
  1080p 一帧 3 平面 × 32400 块 ⇒ **~12 MB/帧的白写**。改成**每个 CDEF worker 一份、
  跨块复用**的 `scratch`（`apply` → `cdef_block` → `cdef_filter` 串下去）。
* `RowsSink::set_row_i16` 从"逐样本取值/比较/截断/存"改成**一次 `memcpy`** ——
  内核输出已过 `Clip3(mn, mx, ·)`，`mn`/`mx` 都是合法样本 ⇒ 必在 `[0, 2^bd)`。

**实测（内核级微基准，同进程交替 3 轮取 min）**：8×8 亮度块
**448.6 → 199.9 ns ⇒ 2.24x**。

## 刀 2：帧缓冲池「最多 8 块」→「按字节封顶 96 MB」

1080p 的 luma 平面 ≈ 4.5 MB、chroma ≈ 1.2 MB。**8 块只装得下 3~5 块工作缓冲**，
而稳态活着的缓冲超过 20 块（参考帧槽最多 8 帧 × 3 平面 + 本帧的解码帧 / CDEF 帧 /
LR 输出）⇒ 池每帧都在 miss，`vec![0u16; n]` 重走 mmap，**每个平面第一页都要缺页中断**。

**用确定性计数量的**（`AV1_TRACE_POOL=1`，40 帧真素材）：

| 池上限 | 命中率 | 稳态落空 |
|---|---|---|
| 1 MB | 0.0% | 7.7 次/帧 |
| 8 MB（≈ 旧「8 块」）| 54.2% | 3.5 次/帧 |
| 16 MB | 86.3% | ~1 次/帧 |
| **96 MB（新默认）** | **89.2%** | **0 次/帧**（33 次落空全在头 12 帧的预热期）|

⚠️ 顺带修掉**我自己引入的一个记账 bug**：`take` 时没把 `bytes` 减掉 ⇒
"取出去再还回来"重复计数，跑一阵子后 `give` 全被拒、池**静默失效**（从外面看只是
"慢慢变慢"，极难发现）。新单测 `pool_accounting_survives_repeated_take_and_give`
钉住（200 轮取还之后池仍能交出同一块缓冲）。

## 整帧结果（一个**噪声很小**的窗口，8 轮换序交替，120 帧单线程）

| | min | median |
|---|---|---|
| 改前（`md5 f45494c4`）| 49.07 | 51.05 |
| **改后** | **47.28** | **49.79** |

⇒ **−2.5%（中位）/ −3.6%（min）**。同窗 dav1d 单线程 **11.60 ms** ⇒ 仍 **4.3x**。

## 验收（全绿）

真素材 20 帧与改前**逐字节相同**（21 个 nv12 全 `cmp`）、与 dav1d **逐像素一致**；
`ME_NO_SIMD` 开/关逐字节；row-mt 默认/顺序/并行三路逐字节；13 条受控流全 0 差；
**workspace 882 项 0 failed**（`me-decode` lib 387 项 + 3 个 `#[ignore]` 微基准）。
回填 `extra/darwin/x64/engine/zuvrust`（md5 `dd665bdd…`，备份
`/tmp/me-extra-prev-20261004-round27`）。

## 一处订正 + 下一步

* **订正**：第二十六轮写的"`cdef_direction` 每 64×64 只调一次"**不对** ——
  `cdef_block` 是按 **8×8** 调的（`step4 = NUM_4X4_BLOCKS_WIDE[BLOCK_8X8] = 2`），
  方向搜索确实**每 8×8 块一次**（1080p ≈ 3.24 万次/帧）。原意是"三个平面共用一次"。
* **下一步**（按采样）：
  * `deblock_edge4` **5.4%** —— i32 通道、一次只滤 4 条线，与本轮 CDEF 同类。
    ⚠️ 但要改 8/16 线**得同时让 `run_line` 一次走 8 行**（`sample_filter` 那句
    "逐 4 个样本一行"是 `run_pass` 切行带的唯一依据），属**跨模块**改动。
  * `vt_forward_1d` 4.7% —— 反变换走 **i64 通道**（只有 4 路并行），但值域论证说
    i32 **装不下**，**不能照搬本轮做法**。
  * `mc_horiz_rows_8bpc` 8.5%（已 16 通道，剩下是 `packs` 收尾）；
    `lr::apply` 闭包 2.9% + `box_filter` 1.6%；`mv::project_frame` 3.2%。
  * 结构性的一条仍是"8bit 走 u8 存储"（`ME_PLANE_U8` 开关还在，默认关）。

---

# 十二、2026-10-05：迁移到 ZuvRust 后的第一轮 —— 深挖 `setup`，砍掉 3.7%

## 水位重新量准

| | 单线程 |
|---|---|
| dav1d（PyAV，`thread_count=1` + `thread_type="NONE"`）| **15~17 ms/帧** |
| ZuvRust（`tiles+filters+ref+out+setup`）| **≈ 50 ms/帧** |

⇒ 约 **3x**；目标 33 ms ⇒ 还差 **1.5x**。
拆账（稳定窗口）：`tiles 39` + `filters 17` + `setup 3.7` + `out 0.5`。

`setup` 原来是黑箱。本轮给 `PHASE` 加了 **`grid=` / `ctx=` / `mf=`** 三个字段，
一次就看清楚了：`grid 0.7` + `ctx 0.1` + **`mf`（运动场估计）2.9 ms**。

## 关键发现：`motion_field_estimation` 是"隐形的大头"

`mf` 原来是 **4.14 ms/帧**（占整帧 ~6%），比任何单个内核都显眼。两处根因：

1. **`project_frame` 的内层 7 个目标帧，每格每目标帧两次 i64 乘。**
   `mv_projection_clipped(mv, num, den)` 展开是 `round2_signed(mv · num · Div_Mult[den], 14)`，
   而 `num` 只跟 `(src, dst)` 有关、`den` 只跟 `src` 有关 ⇒ 提前算成
   `k = num · Div_Mult[den]`（8×8 小表，每帧每源帧一次），每格每目标帧降到**一次**乘。
   整数乘可结合 ⇒ 与原来**逐位相同**。
2. **`MotionField::new` 每帧 `vec![哨兵; 7·h8·w8]` ≈ 2 MB**，走 mmap + 写入时 ~512 次缺页。
   改成**跨帧复用同一块缓冲**（新增 `MotionField::reset`），只把它重置成哨兵
   （`fill` 仍要跑 —— 没被投影到的格子必须回哨兵，否则会读到上一帧的值）。

⇒ `mf` **4.14 → 2.91 ms（−30%）**。

新单测 `project_k_matches_the_reference_projection` 把参考实现 `mv_projection` 留成
`#[cfg(test)]`，对 **全部 `den`(1..=31) × 全部 `num`(−31..=31) × 11 组覆盖值域的 `mv`**
逐值对拍 —— 这是"两次乘并成一次"的安全性论证钉死的那一条。

## 另外两处

* **CDEF 的 SIMD 内核就地写目标平面**：原来是"内核写 scratch → 再 `set_row_i16` memcpy 进
  `CdefFrame`"，1080p 一帧 3 平面 × 32400 块 × 8 行 = **78 万次 16 字节的往返**。
  现在内核直接按 `(y0+r)·dst_stride + x0` 写（`RowsSink::direct()` 给出基址/步长/行区间）。
* **`palette_colors` 只在真是调色板块时写**：这一格 32 B，1080p 一帧 13 万格 ⇒ 原来无条件写
  就是 **4.2 MB/帧**的白写，而真素材里调色板块几乎为零。安全性：**所有**读它的地方都在
  `get_palette_cache` 里，且各自被 `palette_sizes[邻居][plane] > 0` 挡着
  （`palette_sizes` 仍然无条件写，所以那道门是准的）。

## 实测（8 轮换序交替，120 帧，单线程）

| | min | median | 8 轮分布 |
|---|---|---|---|
| 改前 | 48.50 | 52.05 | 48.5 / 50.8 / 50.9 / 51.9 / 52.2 / 52.3 / 52.6 / 52.8 |
| **改后** | 49.20 | **50.10** | 49.2 / 49.8 / 49.8 / 50.1 / 50.1 / 50.2 / 50.2 / 50.2 |

⇒ **−3.7%（中位）**。这一轮窗口比较稳（B 侧 8 轮全落在 49.2~50.2），所以这个数可信。

**试过但没采纳**：`-C target-cpu=native` 只值 **0~3.5%**（在噪声内），而且会把产物绑死在
构建机的指令集上 —— 与"CI 出五份产物发给未知 CPU 的用户"冲突。要做就走**运行时派发**。

## 验收（全绿）

真素材 20 帧与改前**逐字节相同**（21 个 nv12 全 `cmp`）、与 dav1d **逐像素一致**；
`ME_NO_SIMD` 开/关逐字节；row-mt 顺序 / 并行逐字节；**7 条受控流**（`make_cases.py` 重新生成）
全 0 差；**workspace 897 passed / 0 failed**；`cargo check --release --target
aarch64-apple-darwin -p me-decode` 通过。
回填 `extra/darwin/x64/engine/zuvrust`（md5 `d2c4bdb1…`，旧版备份 `/tmp/zv-prev-20261005-round28`）。

## ⚠️ 一个被单测当场抓住的坑

CDEF 内核换成"直接写目标平面"之后，**测试的目标缓冲必须与源同 stride、同 `x0/y0` 约定**
（内核按 `(y0+r)·dst_stride + x0` 算地址）。第一版测试写成 `vec![0u16; h*wu]` +
`dst_stride = wu`，而 `x0 = 8` ⇒ 越界，**单测直接 SIGSEGV**。
**教训**：内核一旦改成"裸指针 + 绝对行号"，调用方/测试的缓冲都得按同一个约定造。

## 下一步

**全面检查报告已写进 `AV1-AUDIT.md`**（正确性硬判据 / 未实现路径 / 安全性 / 三平台 /
测试缺口 / 按收益-风险排序的优化清单）。要点：

1. 去块逐边门控改"每格预计算 level LUT"（~1%，低风险）
2. `cdef_direction` 向量化（~1.7%，中风险，必须逐位对拍）
3. `mc_shift8` 扩 16 宽 + `mc_horiz_rows` 的 epilogue 合并偏置与取整（~1%，低风险）
4. 热路径边界检查消除（未知，可能 2~5%）
5. **8bit 走 u8 存储**（~5~10%，**前置是 MC 横向内核换 `pmaddubsw`**；2026-10-04 的"u8 镜像"
   实验是 **+1.7% 慢**，因为只换了载入没换算法）
6. 去块内核 i16/多线（~3%，高风险：要同时改 `run_line`/`run_pass` 的切行带依据，
   而且 8 线会跨两个 mi 行 ⇒ level 不同）


---

# 十三、2026-10-05（第二段会话）：第二十九刀 —— MC 8 宽内核 / `mc_shift_row` / CDEF 方向搜索向量化

## ⚠️ 本轮最重要的方法论：**墙钟在这台机器上已经彻底不可用了**

同一对二进制做 10~24 轮换序交替，`PHASE` 总和的 **min 与 median 能差 20%**
（实测 A 侧 `min 50.6` 而 `median 61.6`，同一份二进制两次跑 50.6 与 61.6 ms/帧）。
load 在 3~18 之间跳，**换序 + 多轮都救不回来**。

替代口径：**子进程的 CPU 时间**（`getrusage(RUSAGE_CHILDREN)` 的 `user + sys`）。
自检（A=B 同一二进制、4 轮）显示轮内噪声仍有 ~4%，但 **12 轮取配对中位**之后
能给出可信的结论（12 轮里 11 轮同向）。脚本：`/tmp/cputime_ab.py`。

> 先前的三种手段依然有效，按用途分：
> | 想知道 | 用什么 |
> |---|---|
> | 内核快了多少 | 同进程 `#[ignore]` 微基准（`bench_*`，`--nocapture`）|
> | 某个机制有没有在生效 | 确定性计数（`AV1_TRACE_POOL` 命中率、`SYNMS`、`WAIT`）|
> | 整帧快了多少 | **CPU 时间交替 A/B，取配对中位**（墙钟已不可用）|

## 一、先量清楚两件事（都推翻了旧假设）

### 1. 块宽分布：亮度和色度**完全不同**

`AV1_TRACE_INTER=all` 统计 25488 个帧间块（**注意 `bs=` 打的是数值不是名字**）：

| 平面 | 宽度分布 | `≥16` 占比 |
|---|---|---|
| 亮度 | 128:5.7% / 64:20.8% / 32:33.1% / 16:33.7% / 8:6.2% / 4:0.4% | **93.4%** |
| 色度（4:2:0）| 64:5.7% / 32:20.8% / 16:33.1% / **8:33.7%** / **4:6.6%** | **59.7%** |

⇒ **40% 的块，色度横向 MC 整段走标量**（16 宽的 SIMD 门槛是 `c + 16 <= wu`）。
这就是本轮第一刀。

### 2. row-mt 的并行**没有任何问题**，天花板在"语法阶段是串行的"

`AV1_TRACE_SYNMS=1` + `ME_TRACE_WAIT=1`（8T、`ME_ROW_MT=1 ME_ROW_MT_PAR=1`）：

| 量 | 稳态中位 |
|---|---|
| `syntax`（超块循环：熵 + 块簿记 + 记工单）| **13.8 ms/帧** |
| `scope`（并行回放墙钟）| 17.9 ms/帧 |
| ⇒ 语法占瓦片 | **44%** |
| worker 空等占比 | **43.3%** |

9 带 × 15 列 = 135 个作业/帧。**8 个 worker 有 43% 时间在等**，而 `wait_total`
累计 ≈ `scope × 8 × 0.43` ⇒ 回放本身**已经接近这台机器能给的并行度**
（4 物理核 + 超线程 ≈ 5.2 个有效核；把回放压到理论极限也只有 1.8x）。

⇒ **多线程的上限 = 1 / (0.44 + 0.56/5.2) ≈ 1.9x**，而不是"同步写坏了"。
再往上只有两条路：**把语法阶段本身做快**（它是单线程的硬底），或者**帧级并行**
（跨帧，属于架构改动）。实测扩展比：8T 1.10x、8T row-mt 1.12x、8T row-mt 并行 1.23x。

## 二、刀 1：MC 横向的 **8 宽内核**（`mc_horiz8_8bpc`）

`mc_horiz_rows_8bpc` 的内层门槛原来只有 `c + 16 <= wu`；补一条 `c + 8 <= wu`
的 128 位通道版本（`_mm_loadu_si128` × 8 抽头 + `_mm_mullo_epi16`）。
**取址边界与 16 宽那条不同**：8 个输出从 `base + c - 3` 起、最远读到 `base + c + 11`，
而 `c + 8 ≤ wu` ⇒ 最远 `base + wu + 3`，**正是 `interior_x` 保证的那个界**
（不多读一个字节，所以不需要放宽任何前提）。

顺带把标量尾循环拆成两支：`interior_x` 为真时**逐样本 `clamp` 是白做**
（界已由 `interior_x` 保证），去掉之后窄块（`wu < 8`）的标量也快了一截。

**内核微基准（同进程，140×140 缓冲）**：

| 内核 | ns/调用 | ns/像素 |
|---|---|---|
| `mc_horiz16_8bpc`（16 输出，256 位）| 13.52 | **0.845** |
| `mc_horiz8_8bpc`（8 输出，128 位）| 10.41 | **1.301** |
| 横向标量（8 输出 = 改动前 `wu ∈ [8,16)` 走的唯一路径）| 23.06 | **2.882** |

⇒ 对那 40% 的块，**标量 → SIMD = 2.22x**。
（8 宽按像素仍比 16 宽差 1.65x —— 同样的 8 次载入只换 8 个输出。
想让它也到 0.845 就得"算 16 存 8"，但那要读到 `base + 19`，**超出 `interior_x`
给的界**，得再加一个更严的前提，收益只有 ~0.5%，本轮没做。）

## 三、刀 2：`mc_shift_row` —— 把行循环搬进内核

整像素 MV 那条路（`block_inter_prediction` 快路径 1）原来是
`while c + 8 <= wu { mc_shift8(row, x0+c, ...) }` —— **每 8 个像素一次真实函数调用**
（1920 宽的一行 = 240 次；`#[target_feature]` 函数不能被没有该 feature 的调用方内联，
这正是 `mc_horiz_rows_8bpc` 当初存在的理由，`mc_shift8` 漏了）。
新增 `mc_shift_row`（返回 SIMD 处理掉的个数，余数留给标量），语义逐位相同。

**采样占比：`mc_shift8` 3.5% → `mc_shift_row` 2.3%。**

## 四、刀 3：CDEF 方向搜索向量化（`cdef_dir_accum`）

`cdef_direction` 的累加段是 `partial[0][i+j] += x[j] … partial[7][i/2+j] += x[j]`
的 8×8 双重循环 = **512 次带散列下标的加**，1080p 每帧 3.24 万个 8×8 块都要做。

关键观察：**每个方向都只是"把整行 `xs` 加到某个随 `i` 变化的偏移上"** ⇒
一条 8 宽向量加就够。`partial` 每行**补到 16 项**（64 字节 = 一条缓存行），
这样 `p[d][i..i+8]`（`i ≤ 7`）永远落在本行内：

| 方向 | 标量式 | 向量式 |
|---|---|---|
| 0 | `p[0][i+j] += xs[j]` | `p0[i..i+8] += xs` |
| 1 | `p[1][i+j/2] += xs[j]` | `p1[i..i+8] += y`（`y[k]=xs[2k]+xs[2k+1]`，高 4 通道 0）|
| 2 | `p[2][i] += xs[j]` | `p2[i] += Σ xs` |
| 3 | `p[3][3+i-j/2] += xs[j]` | `p3[i..i+8] += rev(y)` |
| 4 | `p[4][7+i-j] += xs[j]` | `p4[i..i+8] += rev(xs)` |
| 5 | `p[5][3-i/2+j] += xs[j]` | `p5[(3-i/2)..+8] += xs` |
| 6 | `p[6][j] += xs[j]` | `p6 += xs`（寄存器里跨行累加）|
| 7 | `p[7][i/2+j] += xs[j]` | `p7[(i/2)..+8] += xs` |

`y` 由 `hadd(xs, 0)` 一次得到（`[x01,x23,0,0,x45,x67,0,0]`），一次
`permutevar8x32` 取 `[0,1,4,5]` 就是 `y`、取 `[5,4,1,0]` 就是 `rev(y)`。
高位那 4 个 **0 加进去不改动别的行**的累加器（加 0 是恒等）。

**护栏**：新单测 `cdef_dir_accum_matches_the_scalar_reference` 独立再写一遍标量式，
对 3 种位深 × 5 组坐标逐格对拍 `partial`。写错一个偏移只会让 `y_dir` 整体偏
（画面出现方向性拖影），**从最终像素差几乎定位不到**，所以这条单测是必需的。

## 五、刀 4：去掉每帧重复的网格重置

`av1.rs` 里 `reset_all()` 之后紧跟一个 `reset_frame()`，而后者填的
（`cdef_idx` / `lf_tx_sizes` / `above|left_seg_pred_ctx` / `written`）
**正是 `reset_all` 刚刚填过的那一组** ⇒ 每帧白写 ≈ **1 MB**
（`cdef_idx` 是 i32，1080p 就 0.52 MB）。删掉那次调用（方法本身留着）。

## 六、顺手修掉的两处**基准**缺陷

`bench_mc_kernels` 里 `base` 从 **0** 开始取（`i % 64`），而横向内核从
`base - 3` 起读 —— **`base = 0` 时是负下标，这条基准一直是 UB**（16 宽那条也一样）。
新加的标量对照一开始也踩了同一个坑（越界 panic）。三处都改成 `base ≥ 3`。

## 七、实测（12 轮交替，**CPU 时间口径**，120 帧/轮，单线程）

| | 中位 | min | max |
|---|---|---|---|
| 旧（`md5 d2c4bdb1`）| 13.86 s | 12.67 | 14.30 |
| **新** | **13.53 s** | 12.67 | 13.73 |

**配对中位 B/A = 0.9557（−4.4%）**；中位之比 0.9756（−2.4%）；
**12 轮里 11 轮 B 更快**。旧版的中位在轮间从 12.67 爬到 14.30（机器负载漂移），
新版稳在 12.67~13.73 —— 所以**配对（同轮相比）比"中位之比"更可信**。

## 八、验收（全绿）

* 真素材 20 帧与改前**逐字节相同**；
* `ME_NO_SIMD` 开/关**逐字节相同**（覆盖 CDEF 方向搜索的标量/向量两条路）；
* row-mt 并行（4T）**逐字节相同**；
* 与 dav1d **逐像素一致**（`--grain=0`，三平面）；
* **7 条受控流**（libaom）全 0 差、无退化；
* **workspace 898 passed / 0 failed**（`me-decode` lib 390 项，含新增的
  `cdef_dir_accum_matches_the_scalar_reference`）；
* `cargo check --release --target aarch64-apple-darwin -p me-decode` **0 error**；
  `cargo clippy -p me-decode --all-targets` **0 error**。

## 九、⚠️ 发现但未解决的问题

* **`ME_PLANE_U8=1` 现在会 panic**（`intra.rs:298 px16_mut()：这个平面是 8bit（u8 存储）`）
  —— 去块 `run_pass` 与 MC 都还在要 `px16()`。**u8 迁移远未完成**，别把它当可用开关。
  它仍是**系统性的一刀**（1080p 的平面占用与访存各减半），但前置工作量大：
  去块 / CDEF / LR / MC / `add_residual` 全部要按存储宽度分派。
* **`BlockGrid` 每帧的初值填写**：`reset_all` 里还有 ~3.6 MB/帧 的 `fill_set`
  （`mvs` 1.04 MB、`cdef_idx` 0.52 MB、`interp_filters` 0.26 MB …）。
  余下那些"本帧必被写满"的证据链不完整（`uv_modes` 有 4×4 的色度归位问题等），
  没敢动。
* **`nv12_frame` 的 `vec![0u8; w·h·3/2]`**：1080p 每帧 3.1 MB 的 `bzero`
  （采样里 `bzero` 2.1~2.3%）。缓冲区整块都会被写满，理论上可以不初始化，
  但要引入 `set_len` 级别的 `unsafe`，本轮没做。


---

# 十四、2026-10-05（第三段会话）：第三十刀 —— 语法阶段专项（−22%）

## 定位方法：**只对语法阶段采样**

上一轮知道"`syntax` 13.8 ms/帧 = 瓦片 44%"，但采样里语法相关符号只占 ~13% ——
对不上。这一轮直接用 `AV1_SYNTAX_ONLY=1` 跑（**跳过全部像素重建**）再 `sample`，
于是采样的分母**就是语法阶段本身**，一眼看出前几名：

| 符号 | 占语法阶段 |
|---|---|
| `decode_block` | 15.9% |
| `read_symbol_inner` | 11.4% |
| **`mv::project_frame`** | **13.0%** |
| `BlockGrid::stamp` | 6.1% |
| `nv12_frame` / `bzero` | 5.3% / 3.4% |
| MV 栈（`find_mv_stack`+`add_ref_mv_candidate`+`add_tpl_ref_mv`）| 7.9% |

> **这是定位"某个阶段内部热点"的标准姿势**：拿一个把其余阶段关掉的开关
> （`AV1_SYNTAX_ONLY`）跑，再采样。

## 刀 1：`MotionField` 布局 `[ref][y][x]` → `[y][x][ref]`（`mf` **−25%**）

规范写的是 `MotionFieldMvs[ref][y8][x8]`，但**访问模式**是"投影一格就把 7 个目标
（`LAST_FRAME..=ALTREF_FRAME`）一次性写满"。`[ref]` 在前 ⇒ 那 7 次写落在**相隔
`h8·w8·8` 字节**（1080p 约 259 KB）的 7 个不同平面上，**每次都是一条独立缓存行**。

改成 `[y8][x8][ref]` 之后那 7 次是**一段连续的 56 字节**（`base` 对齐到 7 个 `Mv`，
互不重叠），缓存行触碰从 7 条降到 1~2 条；读取方只有 `MotionField::get` 一处，
索引一次就够。内存顺带从 8 槽降到 7 槽（2.07 → 1.81 MB）。

⚠️ `INTRA_FRAME` 那一槽**不存在**（规范从不写它）—— 老布局里它恒为哨兵，
新布局在 `get` 里显式返回哨兵，**可观测行为相同**（`ME_NO_SIMD`/逐字节验收钉住）。

**实测（`mf` 段，10 轮交替）**：**3.075 → 2.300 ms/帧（−25%）**，10 轮全部同向、极紧。

## 刀 2：`BlockGrid::stamp` 改**按行给切片**

原来是 `f(grid, i)` **逐格**调用，闭包里 8~14 个 `g.field[i] = v` —— 每个都要
**重新加载 `Vec` 的数据指针 + 一次越界判断**，而编译器**无法**把这些指针提升到
`f(self, i)` 之外（它不知道闭包会不会改别的字段）。1080p 一帧 13 万个 4×4 格。

改成 `stamp` 把每个 4×4 **行**裁好之后交给闭包（`f(grid, lo, hi)`），闭包写
`g.y_modes[lo..hi].fill(v)`：越界判断每行一次、数据指针每行一次。
**语法阶段占比 6.1% → 3.0%**（`memset` 从此出现在采样里，那正是 `fill` 被向量化）。

## 刀 3：`read_bits_at` —— **逐位循环 → 64 位大端窗口**

这是本轮最值钱的一刀。原来是：

```rust
for i in 0..n { let p = pos + i;
    if p / 8 >= data.len() { return Err(...) }
    let bit = (data[p / 8] >> (7 - (p % 8))) & 1; ... }
```

**逐位一次除法 + 取模 + 越界判断 + 装载**。而归一化**几乎每个符号都要读 1~15 位**
（`num_bits = bits.min(avail)`）⇒ 每个符号几十条指令全花在拆位上。

改成一次取 **8 字节大端窗口**（末尾不足 8 字节时左对齐补零），
`off + n ≤ 7 + 32 = 39 ≤ 64` ⇒ 一次移位 + 一次掩码。越界口径与逐位版**逐位一致**
（`have` = 第一个越界的下标）。新单测 `read_bits_at_matches_the_bitwise_version`
对**每个位位置 × 每个位长（0..=32）**与逐位参照逐值对拍
（⚠️ 它一开始写到 `n = 33` 撞上了函数契约的 `debug_assert!(n <= 32)` ——
**`cargo test` 会开调试断言**，所以契约得在测试里也守住）。

## 刀 4：CDF 自适应 —— 一个带分支的循环 → **两段单调无分支循环**

```rust
let mut tmp = 0;
for i in 0..n-1 { if i == symbol { tmp = CDF_PROB_TOP; } ... if tmp < v { v - ((v-tmp)>>rate) } else { v + ((tmp-v)>>rate) } }
```

`tmp` 只在 `i == symbol` 那一步变，所以拆成
`i < symbol` 全是 `v - (v>>rate)`、`i >= symbol` 全是 `v + ((TOP - v)>>rate)`。
**循环体内不再有分支**，两段都是单调映射。`symbol == n-1` 时第二段为空区间，
与原来"循环上界不含 `n-1`"**逐项一致**（新单测
`cdf_adapt_split_matches_the_single_loop_version` 对 `n ∈ {2,3,4,5,8,11,13,14,16}`
× 每个 `symbol` × `rate ∈ 3..=8` 对拍）。

## 实测（全部交替 A/B，取配对中位）

| 量 | 旧 | 新 | 比 |
|---|---|---|---|
| **纯语法 `tiles`**（`AV1_SYNTAX_ONLY`，8 轮）| 10.725 ms | **8.375 ms** | **−22%**（8/8 轮同向）|
| **`setup`**（6 轮）| 3.80 ms | **3.00 ms** | **−21%**（6/6 轮同向）|
| 完整解码 `tiles`（10 轮）| 36.83 ms | 36.25 ms | 配对中位 −3.0% |
| **总账：CPU 时间 / 120 帧**（12 轮）| 9.64 s | **9.44 s** | **−2.1%**（10/12 轮同向）|

> ⚠️ **别拿"语法 −22%"去乘占比**：语法阶段约占整帧 29%，但这一批只砍掉其中一部分，
> 而且 `setup` 里的 `mf` 与 `tiles` 里的语法是分开的。总账 −2.1% 才是可信的数
> —— **段级百分比只用来排序，最终必须看总账**。

## 验收（全绿）

20 帧与改前**逐字节相同**、`ME_NO_SIMD` 开/关**逐字节相同**、row-mt 并行**逐字节相同**；
与 dav1d **逐像素一致**（三平面 `--grain=0`）；**7 条受控流 0 差无退化**；
**workspace 全过**（`me-decode` lib 392 项，含两条新增对拍单测）；
aarch64 `cargo check` **0 error**；clippy **0 error**。

## ⚠️ 一处踩坑

**`cargo test` 会开调试断言**（即使 `--release`），所以 `debug_assert!` 写在热路径的
契约里时，**单测也必须守这个契约** —— 我的 `read_bits_at` 对拍测试一开始跑到
`n = 33` 就把自己打挂了（生产调用点的 `n` 只有 1~15）。

## 下一批（按当前采样排序）

| 目标 | 占整帧 | 说明 |
|---|---|---|
| `mc_horiz_rows_8bpc` | 8.6% | u16 下已接近极限（8 次载入 / 16 输出）；要再快得走 u8 + `pmaddubsw` |
| `deblock_edge4` + `loop_filter_edge` + `filter_strength` | 11.7% | 内核 i32/4 通道；加宽要同时改 `run_line`/`run_pass` |
| `vt_forward_1d` + `dequantize` | 8.1% | 反变换走 i64；需要重新论证 i16/i32 |
| LR（`lr::apply`+`sgr_ab_8`+`sgr_final_8`+`box_filter`）| 8.7% | |
| `bzero` + `memmove` | 4.8% | `transform_block` 里每块的 `quant.fill(0)`（`coeffs` 用，平均不到 100 个 i32）+ `nv12` 每帧 3.1 MB |
| MV 栈 | ~5% | `find_mv_stack`/`add_ref_mv_candidate`/`add_tpl_ref_mv` |
| `project_frame` | 2.5% | 还剩 8 次 `project_k`/格，可试 i64 SIMD（4 路）|


---

# 十五、2026-10-05（第四段会话）：第三十一刀 —— "别做白工"批 + **把 u8 路线算清楚**

## A. 一批"去掉白做的工作"（每项 <0.5%，合起来 ~1%）

| 刀 | 内容 |
|---|---|
| `deblock` | `run_line` 已经按变换块跳着走、每次都从 `lf_tx_sizes` 取过 `tx_sz`；`lf_gate` 与 `loop_filter_edge` 又各取一遍 ⇒ **一条边读 3 次同一张数组**。改成 `run_line` 只读一次、往下传，`lf_gate` 不再碰数组。 |
| `transform` | `inverse_transform_2d_into` 的复用缓冲原来是 `if residual.len() != w*h { clear(); resize(w*h, 0) }` —— **每换一次块尺寸就白写一遍**（最大 32 KB 的 `i64`）。真素材块尺寸 64→32→16→8 反复跳。改成**只增不减**（`resize` 只在长不够时发生），并把"行趟写满 `[0, w·h)`"这条不变量用 `let res = &mut residual[..w*h]` 显式化。 |
| `dequantize` | 原来每个系数一条**硬件除法** `(Abs(dq) & 0xFFFFFF) / dqDenom`（~20-40 周期；32×32 块 = 1024 次）。而 `dq_denom` **只返回 1/2/4**，分子恒非负 ⇒ **除法就是右移**。另外 `levels` 只有 `< eob` 非零 ⇒ **零系数直接落 0**（省掉乘/移位/钳位）。 |

**实测（12 轮交替，CPU 时间 / 120 帧）**：配对中位 **B/A = 0.9944（−0.56%）**、
中位之比 0.9859（−1.4%）、min 0.9776；12 轮里 7 轮更快 —— **已到本机噪声边缘**。

## B. ⭐ 把 "8bit 走 u8 存储" 这条**算清楚**（本段会话最值钱的产出）

审计清单第 5 条一直写着"8bit 走 u8 存储 ~5~10%，前置是 MC 横向内核换 `pmaddubsw`"。
这一轮把这条**推导 + 实测**都做了，结论是**它站不住**：

### B1. `pmaddubsw` 在 AV1 的 8 抽头滤波器上**会饱和**

`pmaddubsw(u8 像素, i8 系数)` 的每一对输出是 `sat(i16, s·c_a + s'·c_b)`。
把 `SUBPEL_FILTERS` 全表（96 相位 = **384 个抽头对**）扫一遍：

| 量 | 值 |
|---|---|
| 非 δ 相位 `max|c|` | **126**（装得进 i8 ✓）|
| δ 相位的 `|c|` | 128（但 δ = 整像素，本来走 `mc_shift_row`）|
| **会饱和的抽头对数** | **18 / 384** |
| 最坏配对 Σ|c| | **138**（`[-2,8,-18,120,26,-10,6,-2]` 的第 2 对：`138 · 255 = 35190 > 32767`）|

⇒ **卡住的不是系数范围，是配对和**。要绕开只能把大系数对拆开（多花 maddubs）或加偏置，
多出来的指令把"一次算两个抽头"的便宜吃回去。

### B2. 退一步只换"载入宽度"（u8 + `vpmovzxbw`）—— **实测更慢**

新内核 `x86::mc_horiz16_u8`（与 `mc_horiz16_8bpc` **同一串 16 位算术**，只是从 u8 载入并加宽），
新增并发微基准 `bench_mc_kernels` 里横比、新增单测 `mc_horiz16_u8_matches_the_u16_version_bit_for_bit`
（4 组滤波 × `round` 3..=5 × 94 个 `base` 逐位对拍）保证它**是对的**：

| 内核 | ns/像素（同进程，取两次较稳的窗口）|
|---|---|
| `mc_horiz16_8bpc`（u16 源）| **0.767 / 0.812** |
| `mc_horiz16_u8`（u8 源 + 加宽）| **0.854 / 0.874** |

⇒ **u8 源慢 ~11%**。原因：数据**L1 常驻**（一行 8 抽头才 256 B），瓶颈是**载入条数**
（都是 8 条），不是字节数；u8 版白白多出 8 条 `vpmovzxbw`。

### B3. 这条推翻了什么

* `AV1-AUDIT.md` 第 5 条的**前置**（"换 `pmaddubsw`"）**不成立** ⇒ "u8 存储 ~5~10%"这个
  收益数字**没有依据**，必须重新推。
* 2026-10-04 那次"u8 镜像 +1.7%"的实验**不是"只换载入没换算法"那么简单** ——
  算法侧那条路（`pmaddubsw`）**本身就不通**。
* u8 存储**可能**值钱的地方要另找：CDEF / LR / 去块的**平面读字节数减半**
  （那是 cache/带宽口径，不是算术口径）。**要重新量，别再引用旧数字。**

## 验收（全绿）

20 帧与改前**逐字节相同**、`ME_NO_SIMD` 开/关**逐字节相同**、row-mt 并行**逐字节相同**；
与 dav1d **逐像素一致**（三平面）；**7 条受控流 0 差无退化**；
`me-decode` lib **393 passed / 0 failed**（新增 `mc_horiz16_u8_...` 与
`dequant_uses_a_shift_not_a_division` 两条对拍）；**workspace 901 passed / 0 failed**；
aarch64 `cargo check` **0 error**；clippy **0 error**。


---

# 十六、2026-10-05（第五段会话）：第三十二刀 —— 反变换的**基础算子**（−3.0%）

## 为什么挑 `vt_forward_1d`

重采样（load ≈ 5.2 的好窗口）：`mc_horiz_rows_8bpc` 9.0% / `deblock_edge4` 6.6% /
**`vt_forward_1d` 6.5%** —— 反变换已经是第 3 大，而且是**一个模块内部**的事（不改跨模块契约）。

先看它原来的样子（`simd.rs` 的 `x86` 模块）：

* 通道是 **i64**（4 路并行）—— 这是**必须的**：文件里 2026-10-02 的插桩实测写着
  "乘法输入 `|t| ≤ 2^19.5`，但**积与和要到 ~2^32**" ⇒ `i32` 通道装不下。
  所以"i64→i32 换 8 路"这条路**不通**，只能在**每条指令**上省。
* AVX2 **没有 `srai_epi64`**。原 `vt_srai` 用"复制符号位 → 逻辑右移 → 或回去"，
  **5 条**指令（`shuffle_epi32` + `sllv` + `srlv` + `or`）。
* 而 `vt_round2(x, 12)` 在**每个蝴蝶**里要调 **2 次**（32 点 ≈ 15 个蝴蝶、64 点 ≈ 30 个）
  ⇒ 一次 1D 变换光这一项就是 ~150-300 条指令。
* `cos128`/`sin128` 是**四象限分支**的查表，每个蝴蝶 2 次。
* `brev` 是**逐位拼的循环**，而 `vt_dct_permute` 对**每个元素**都要调一次
  （32 点 = 32 次、64 点 = 64 次）。

## 三刀

### 刀 1：i64 算术右移 **5 条 → 3 条**（`vt_srai_c`）

**一个 i64 就是两个 i32**，而 32 位**有** `srai_epi32`：

| 指令 | 作用 |
|---|---|
| `srli_epi64::<N>` | 低 32 位**已是**答案；高 32 位 = `hi` 的**逻辑**右移 |
| `srai_epi32::<N>` | 高 32 位**已是**答案（低 32 位无意义） |
| `blend_epi32::<0xAA>` | 低半取前者、高半取后者 |

配套 `vt_round2_c::<N, SHL>`（`SHL = 64 - N`，因为 stable Rust 不允许在 const 泛型里做算术）。
把所有**常量移位**的落点都换掉：蝴蝶 ×2（`12`）、ADST4（`12`）、identity 预缩放（`12`）、
1:2 长方块预缩放（`12`）、行写回（`row_shift ∈ {0,1,2}`）、列写回（`col_shift ∈ {0,4}`）。

### 刀 2：`cos128` / `sin128` 的分支 → **256 项直查表**

原式按 `a = angle & 255` 落象限决定"取哪个下标 + 要不要取负"，每次 2~3 个分支。
`COS128_FULL` 在**编译期**按同一式子求值，函数体只剩 `掩码 + 查表`。
`sin128(angle) = COS128_FULL[(angle - 64) & 255]`（负角的补码低 8 位天然落在 `0..256`）。

### 刀 3：`brev` 的逐位循环 → **64 项表**

`brev(numBits, x) = BREV6[x & 63] >> (6 - numBits)` —— `reverse6(x)` 的**高** `numBits` 位
恰好是"低 `numBits` 位反转后的值"。`vt_dct_permute` 每个元素一次调用，这条省得最直接。

## ⚠️ 我写错了第一版，被**逐字节对拍**当场抓住

刀 1 的第一版想当然写成 `(v << (64-N)) >> (64-N)`，注释里还写着"只要 2 条".
**那是逻辑右移**：左移把符号位丢出 64 位之外，右移补的自然是 0 ⇒ **负数全错**。
`ME_THREADS=1` 的逐字节对拍在**第一次构建后就把它抓出来了**（`diff -r` 满屏）。

修完还补了一条更强的单测 `vt_srai_const_matches_arithmetic_shift`：
参照系直接用 **Rust 自己的 `i64 >> n`**（那就是算术右移），
对 `N ∈ 1..=31` × 含 `i64::MIN/MAX/-1/0/±2^k` 的输入逐值对拍 ——
比"跟另一份 SIMD 实现对拍"强，**不会两个都错还互相印证**。

**教训（值得写进流程）**：`-1 >> 1` 这类"负数样本"必须出现在自测里。
如果当时只用正数/常量样本，这条会一路错到线上。

## 实测

**采样占比**：`vt_forward_1d` **6.5% → 4.5%（−31%）**。

**整帧（14 轮交替，CPU 时间 / 120 帧，load ≈ 11~12）**：

| | 中位 | min | max |
|---|---|---|---|
| 旧（`md5 4030c6ba`）| 10.16 s | 9.12 | 11.34 |
| **新**（`md5 3e10dd78`）| **9.86 s** | **8.90** | 11.07 |

⇒ **配对中位 B/A = 0.9698（−3.0%）**，中位之比 0.9703（−3.0%），
**14 轮里 13 轮更快**（唯一那轮 1.0289）。这是近几轮**最干净**的一次结果。

> 同一批改动里 `vt_srai_c` 单独那版（`155d7c71`）在 10 轮窗口里**测不出**（配对中位
> +0.9%、窗口从 9.4 漂到 16.2 s）—— **窗口不稳时单体改动一律不可测**，合批 + 长轮次
> 才出得来数。这又一次印证了"段级/单刀百分比只用来排序"。

## 验收（全绿）

20 帧与改前**逐字节相同**、`ME_NO_SIMD` 开/关**逐字节相同**、row-mt 并行**逐字节相同**；
与 dav1d **逐像素一致**（三平面）；**7 条受控流 0 差无退化**；
`me-decode` lib **396 passed / 0 failed**（新增
`vt_srai_const_matches_arithmetic_shift` / `cos128_full_table_matches_the_quadrant_version` /
`brev_table_matches_the_bitwise_version`）；**workspace 904 passed / 0 failed**；aarch64 `cargo check` **0 error**；
clippy **0 error**。

## 下一步（按新采样排序）

| 目标 | 占整帧 | 说明 |
|---|---|---|
| `mc_horiz_rows_8bpc` | **8.2%** | L1 常驻、载入条数已是最少 ⇒ 只能改算法（u8 那条**已证不通**）|
| `deblock_edge4` | **7.7%** | i32 / 4 通道；加宽到 8 线要**同时**改 `run_line`/`run_pass` 的切行带依据 |
| `vt_forward_1d` | 4.5% | 这一轮已经削了 31%，剩下的主要是 `vt_clamp`（6 条）与蝴蝶的乘法 |
| `cdef_pair16` | 4.0% | 已经 16 通道 i16 |
| `loop_filter_edge` + `filter_strength` | 6.0% | 去块门控；可试"每格预计算 strength LUT" |
| `project_frame` | 2.6% | 还剩 7 次 `project_k`/格，可试 i64 SIMD（4 路）|
| `block_warp` | 2.3% | i64 标量、每 8×8 块；`r2` 闭包可单态化 |

---

## 十七、第三十三/三十四轮（2026-10-05 第六段）：立基线 + 两刀

### 0. 先说结论：我们在哪

CPU 时间口径（`ME_THREADS=1`、1080p B 站真素材、`decode --count=100` 不落盘）：

| | 数字 |
|---|---|
| 本机软解 | **62.9 ms/帧**（`PHASE` 合计 59.3） |
| dav1d 单线程（同口径，`thread_count=1` + `thread_type=NONE`） | **17.3 ms/帧** |
| 比值 | **5.28x**（要 30fps 需 33 ms ⇒ 还要 1.9x） |

`PHASE` 拆分：**`tiles` 38.3 / `filters` 16.8 / `setup` 3.2 / `ref` 1.0** ms。

### 1. ⭐ 新增工具：`tools/av1ref/cpuab.py`（本项目之前缺的那块）

墙钟在本机 ±20%（`min` 与 `median` 能差 20%），`elapsedMs`/`PHASE` 只在
"两版都稳定"的窗口里可比。**子进程 `getrusage` 的 CPU 时间**与"同时有多少别的
进程抢核"无关，代价是它把并行折成总工作量 ⇒ **只在同一线程配置下比较**。
脚本做的是**同窗口交替**（默认奇偶轮换顺序）+ **配对比值的中位数**
（比"中位之比"稳，它抵消窗口漂移）+ 报"更快轮数/总轮数"。

```sh
python3 tools/av1ref/cpuab.py base=/tmp/me_base new=./target/release/zuvrust \
    --count=100 --rounds=24            # 默认 env 只有 ME_THREADS=1
```

### 2. 三条否定性结论（省掉了后面的白工）

1. **`-C target-cpu=native` 只快 1.7%**（10 轮配对中位 `0.9834`）⇒ 这台机器上
   "没全局开 AVX2" **不是**瓶颈；全局 `+avx2` 附带的"`#[target_feature]` 函数
   可被内联"的好处，**不值得用可移植性换**。
2. **u8 存储路线作废**（`pmaddubsw` 饱和 + u8 载入慢 11%，见第十六节）。
3. **帧缓冲中灰填充早就是"只填 padding"**（`intra.rs::Plane::new`）⇒ `sample` 里
   那 3% `bzero` **不是**它。

### 3. ⭐ 隔离开关量出来的真实阶段成本（比 `sample` 的函数占比可靠）

`cpuab.py` 拿"关掉某一级"的包装脚本与基线交替比：

| 关掉 | 配对中位 | 同向轮数 |
|---|---|---|
| `AV1_NO_DEBLOCK=1` | **−10.3%** | 8/8 |
| `AV1_NO_CDEF=1` | **−10.5%** | 8/8 |
| `AV1_NO_LR=1` | **−6.8%** | 7/8 |

⇒ **三个环路滤波合计 ≈ 27.6%**，与 `sample` 的函数占比（12.8 / 10.1 / 9.3%）
互相印证。**这是本项目第一次拿到"按阶段"的硬数字。**

### 4. 刀 1：横向 MC 收尾 **11 条 → 2 条**

`mc_horiz16_8bpc` / `mc_horiz8_8bpc` / `mc_horiz16_u8` 与 `mc_horiz_rows_8bpc`
的 16 宽内联段末尾都是：

```text
cvtepi16_epi32 ×2 → add_epi32 ×2（抵消偏置）→ rnd ×2（各 add+srai）
→ packs_epi32 → permute4x64            = 11 条
```

带偏置累加的 `acc = Σ c·s − 2¹⁴` **全程在 i16 里**，而

```text
PMULHRSW(a, 2^(15−n)) = sat16((a·2^(15−n) + 2¹⁴) >> 15) = (a + 2^(n−1)) >> n
```

⇒ `mulhrs(acc, 1<<(15-n)) + (16384>>n)` **两条指令**就是 `Round2(Σ c·s, n)`。
结果值域 `[−1785+2^(14−n), 5865+2^(14−n)]` 远在 i16 内 ⇒ **饱和分支永不触发**
（`horiz16_matches_scalar_bit_for_bit` 逐值对拍钉住）。整条链 ≈ 36 → ≈ 20 条 / 16 输出。

⚠️ **写反过一次**：第一版按"减去 `2^(14−n)`"写（`psubw`），因为 `acc` 比真正的
`Σ` **小** `2¹⁴`，所以 `pmulhrsw` 的结果比目标**小** —— 是**加**不是减。
`horiz16_matches_scalar_bit_for_bit`（96 相位 × 4 类行 × `round ∈ {3,5}`
× 4 个 base）当场变红。推导写在 `simd::horiz_round_mul` 的文档里。

### 5. 刀 2：`quant` 的清零只清补集

`sample` 的**调用树**（不是 top-of-stack 表）指出 `bzero` 的第二大来源：
`ResidualCtx::transform_block` = **整帧 2.1%**，即每 个变换块一次
`quant[..min(w·h,1024)].fill(0)`（真素材平均不到 100 格、30k+ 个块/帧）。

* `txb_skip == 1`（`all_zero`）的块在 `coeffs` 里就 `return Ok(0)`，
  `eob == 0` ⇒ `add_residual` 不被调用 ⇒ **`quant` 一个字节都没被读**；
* 其余块只需"没写到的位置是 0"，已写到的位置马上被幅度循环覆写。

⇒ 把清零挪进 `coeffs`、放在**解析出 eob 之后**，只清 `[0,n) − scan[0..eob)`。
`DEFAULT_SCAN_*` 是**不单调的对角扫描**（`0,1,4,8,5,2,…`）⇒ 补集不能顺序扫出，
用一张 16 项位图标 `scan[0..eob]`，再清没打标的位置、`mx` 之后整段一次 `fill`；
`mx + 1 > n/2`（稠密块）退回整块清。**最终状态与"先整块 `fill(0)`"逐位相同。**

### 6. 实测

| 口径 | 窗口 | 配对中位 | 同向 |
|---|---|---|---|
| 刀 1 单独 | 12 轮 / 100 帧 | **−1.4%** | 10/12 |
| 两刀合计 | 12 轮（load 18~23，**作废**） | −1.3% | 7/12 |
| 两刀合计 | **24 轮 / 100 帧（干净窗口）** | **−1.5%** | **18/24** |

⇒ 刀 2 落在本机噪声里。它**理论上必然更少**（严格少写一遍），保留的依据是
"结构性的少写 + 逐字节相同"，不是那 −1.5%。

### 7. 验收（两刀全绿）

24 帧与改前**逐字节相同**、**7 条受控流 0 差**、真素材 60 帧三平面与 dav1d
**逐像素一致**、`ME_NO_SIMD` 开/关**逐字节相同**、row-mt 串行/并行**逐字节相同**、
`me-decode` lib 396 passed / 0 failed、**workspace 904 passed / 0 failed**、
clippy **0 error**、aarch64-apple-darwin `cargo check` **0 error**。

### 8. 下一步（按"隔离开关 + 采样"合排）

| 目标 | 占整帧 | 判断 |
|---|---|---|
| `deblock_edge4` | **10.3%** | 4 线 / 4 个 i32 通道（**SSE 宽**）。加宽到 8 线要**同时**改 `run_line`/`run_pass` 的切行带依据（**两条相邻的 `outer` 才是 8 行**），且两条边的 `filter_size`/`need` 必须一致才能合批（否则宽滤波器的 const 泛型分不开）⇒ 跨模块、风险最高、收益也最高 |
| `mc_horiz_rows_8bpc` | 8.1% → ≈7% | 已 20 条/16 输出。剩下的瓶颈是**载入条数**（8 条 32 B、多数跨缓存行）⇒ 只能"少载入 + `vpalignr` 移位" |
| `cdef` 整级 | **10.5%** | `cdef_pair16` 已 16 通道 i16；`cdef_filter` + `dir_accum` 还占 ~4% |
| `vt_forward_1d` | 5.0% | **i64 通道**（只有 4 路并行）且值域论证说 i32 装不下 ⇒ 只能继续逐条省 |
| `Plane::clone`（CDEF 前那一次整帧拷贝） | 1.3% | dav1d 也拷，暂看不可免 |
| `lr` 整级 | **6.8%** | `sgr_ab_8` + `sgr_final_8` + `box_filter`，已 SIMD |

---

## 十八、第三十五轮（2026-10-05 第七段）：CDEF 的 4×4 块 —— 通道"用一半"→"全用满"

### 观察

CDEF 内核对每个像素取 **12 个候选**（主抽头 2×2 + 次抽头 2 方向×2×2），
代价与"处理了多少个通道"成正比。`simd::cdef_pair16` 一次算
**2 行 × 8 列**（16 个 i16 通道，低/高 128 位各一行）。

| 块 | 尺寸 | 16 个通道里有用几个 |
|---|---|---|
| 亮度 | 8×8 | **16（全部用满）** |
| **4:2:0 色度** | **4×4** | **8（一半白花）** |

那 8 个被丢弃的通道读到的是块**右边**的样本，算完直接扔。
而色度恰好占 CDEF 内核时间的一半：每平面 `h/2 = 2` 次迭代 × 2 个平面 = 4 次，
与亮度的 `8/2 = 4` 次**相同**。

### 改法

新增 `simd::cdef_quad8`：**4 行 × 4 列**塞进同一个 256 位寄存器
（4 个 qword，每个 qword 是一行的 4 个 u16，用 `_mm256_set_epi64x` 拼）。
16 个通道**全部有用**，迭代次数从 `h/2` 降到 `h/4`。

* 每个候选的取址：`_mm_loadl_epi64` ×4 + `set_epi64x`（原来是 `_mm_loadu_si128` ×2
  + `set_m128i`）—— **指令条数不变，每次迭代多产出一倍**；
* 取址要求**更弱**（每行只读 4 个 u16，`cdef_pair16` 要求同样的地址上可读 8 个）
  ⇒ `cdef.rs` 的窗口判据（`x0 + 9 < stride` 等）**一行都不用改**，
  改动只在 `cdef_filter` 里加一个 `if w == 4 && h == 4` 的分派；
* 亮度**没有**对应的问题：8×8 时 16 通道已满，
  而"4 行 × 8 列"要用 2 组寄存器，指令数与现在两次迭代相同（**没有收益**）。

⚠️ **第一版漏了 `y0·stride`**：`gather` 里写成 `x0 + lin`，读成了第 0..3 行
而不是 `y0..y0+3`。被新写的对拍单测当场抓住。

### 单测

`cdef_quad8_matches_cdef_pair16_bit_for_bit`：**8 个方向 × 4 组阈值/阻尼 ×
两类数据（阶梯+振铃 / 含 0 与 4095 极值的伪随机）**，对 `cdef_pair16` 逐像素对拍。

> **为什么不对着标量再写一遍**：`cdef_pair16` 已经有
> `cdef_pair16_matches_scalar_bit_for_bit`（对 `cdef.rs` 里的逐像素标量式）钉着，
> 对它对拍**同样强、而且更短**，顺带挡住"qword 拼装顺序写反"这类
> **只在 4×4 上出现**的错误（新写的那个 bug 就是这类）。

### 实测 —— 测不出来

| 轮次 | 配对中位 | 同向 | 备注 |
|---|---|---|---|
| 20 轮 / 100 帧 | −0.9% | 12/20 | load 10~29 |
| **32 轮 / 100 帧** | **−0.2%** | 19/32 | load 10~29，本轮最佳窗口 |

⇒ **保留的依据是"严格更少的工作量 + 逐字节相同"，不是那个数字。**
这正是本项目一贯的判据（见"单边/本体对照"与"段级百分比只用来排序"两条）：
**结构上必然更少、且外部参考对拍全绿**的改动，即使幅度落在本机噪声里也该留；
反之只有"实测数字好看"而说不清为什么快的，不留。

### 验收

24 帧与改前**逐字节相同**、**7 条受控流 0 差**、真素材 60 帧三平面与 dav1d
**逐像素一致**、`ME_NO_SIMD` 开/关**逐字节相同**、row-mt 串/并行**逐字节相同**、
`me-decode` lib 的 8 项 CDEF 相关测试**全过**（新增 1 项）、
**workspace 905 passed / 0 failed**、clippy **0 error**、
aarch64-apple-darwin `cargo check` **0 error**。

### 下一步（修正上一轮那张表）

| 目标 | 占整帧 | 判断 |
|---|---|---|
| `deblock_edge4` | **10.3%** | 4 线 / 4 个 i32 通道（**SSE 宽**）。⭐ **上一轮引用的"旧注释"里那条否决理由是错的**：`simd.rs` 说"两条边拼 256 位"的两个理由是①"强度/滤波尺寸未必一样"②"窗口在内存里不连续"。实际上②**不成立**（pass 0 时相邻的 `o` 就是相邻 4 行，窗口连续）；①**可以解决**——`limit/blimit/thresh` 三个常数用 `_mm256_inserti128` 拼成"低半 A / 高半 B"只要 2 条指令，不是"要拆一半来做分支"。**真正唯一的阻塞是 `filter_size`/`need` 不同 ⇒ 宽滤波器的 const 泛型分不开**，那就回落两次 4 宽调用。剩下的活是 `run_line`/`run_pass`：`o` 的 col 走法两条边**不同**（变换块尺寸不同 ⇒ 步进不同），所以要"主边走完 + 记录被配对的 col + 从另一条边的走法里跳过"，配对条件必须**两边算出同一个结果**（是确定性的，但要保证只算一次）|
| `dbl_wide`（去块宽滤波器）| 含在 10.3% 里 | 12 档 × 13 抽头 = **156 次向量加**。抽头权重对 `j` **偶对称** ⇒ 精确递推 `A[i+1] = A[i] + t[i+7] − t[i−6] + t[i+2] − t[i−1]`（**4 条**而不是 13 条），**整数加法逐位可结合 ⇒ 结果逐位相同**。边界钳位在递推里要分两段（`i ≤ −1` 减常量 `t[-7]`、加 `t[i+7]`；`i ≥ 0` 加常量 `t[6]`、减 `t[i-6]`）——都是**编译期常量分支**，不增加运行时分支 |
| `mc_horiz_rows_8bpc` | ≈7% | 已 20 条/16 输出。剩下的是**载入条数**（8 条 32 B、多数跨缓存行）⇒ 只能"少载入 + `vpalignr` 移位" |
| `vt_forward_1d` | 5.0% | **i64 通道**（4 路）且值域论证说 i32 装不下 |
| `sgr_ab_8` / `sgr_final_8` | 5.0% | LR 整级 6.8% |
| `Plane::clone`（CDEF 前那一次整帧拷贝）| 1.3% | dav1d 也拷，暂看不可免 |

---

## 十九、第三十六轮（2026-10-05 第八段）：去块宽滤波器的**精确递推** —— **−1.3%**

### 观察

去块整级占整帧 **10.3%**（三个环路滤波里最大）。其中 `dbl_wide`（§7.14.6.4）是
**每个输出直接求和**：

```text
  A(i) = Σ_{j=-N..N} w(j) · t'[i+j],   w(j) = (|j| ≤ N2) ? 2 : 1
  f[i] = Round2( A(i), L2 )
```

16 抽头（`N = 6`、`L2 = 4`）时**每个输出 13 项**、12 个输出
⇒ **156 次 `paddd`**（再加 12×(`slli` + `add` + `srai`)）。

### 递推

把 `A(i+1) − A(i)` 按 `k = j+1` 重排：

```text
  A(i+1) = Σ_{k=-N+1..N+1} w(k-1)·t'[i+k]
  A(i+1) − A(i) = w(N)·t'[i+N+1] − w(−N)·t'[i−N]
                  + Σ_{k=-N+1..N} ( w(k−1) − w(k) )·t'[i+k]
```

两个关键事实：

1. **`w` 偶对称**（判据是 `|j|`，不是 `j`）⇒ `w(N) = w(−N) = 1`（三个实例都
   `N2 < N`），两个边界项同权 ⇒ 合成 `t'[i+N+1] − t'[i−N]`；
2. **`w(k−1) − w(k)` 只在 `|·| ≤ N2` 这个区间指示函数的两端非零**：
   `k = −N2` 处 2→1（**−1**）、`k = N2+1` 处 1→2（**+1**）。

⇒ **`A(i+1) = A(i) + t'[i+N+1] − t'[i−N] + t'[i+N2+1] − t'[i−N2]`** —— **4 条指令**。

**整数加法可结合、可交换 ⇒ 与直接求和逐位相同**（不是"近似"）。
代价：`N = 6` 时 13（首项）+ 11×4 = **57 次加**（156 → 57）。

三个实例共用一个式子：

| 实例 | `N` | `N2` | `L2` | 用在 |
|---|---|---|---|---|
| 亮度 8 抽头 | 3 | **0** | 3 | `dbl_wide::<3,0,3>` |
| 色度 8 抽头 | 2 | 1 | 3 | `dbl_wide::<2,1,3>` |
| 16 抽头 | 6 | 1 | 4 | `dbl_wide::<6,1,4>` |

⚠️ **`N2 = 0` 是边界情形**：两个非零点退化成 `k = 0` 与 `k = 1`
（`Σ_{k}` 里 `−N2 = 0`、`N2+1 = 1`）。**只测 `N2 ≥ 1` 会漏掉它** —— 三个都要跑。

### 单测

`dbl_wide_recurrence_matches_the_direct_sum`：**3 个实例 × 6 类数据
（全 0 / ±4095 极值混排 / 等差 / 稀疏 / 伪随机，四条通道各给不同值）
× 逐输出 × 逐通道**，对**旧的直接求和**逐位对拍。

> 参照系必须是**那份旧写法本身**（而不是"另一个递推"）：递推写错会
> **从某个 `i` 起整段偏掉**，只有直接求和能同时钉住"哪一项的符号反了"。

### 实测

| 轮次 | 配对中位 | 同向 | 备注 |
|---|---|---|---|
| **32 轮 / 100 帧** | **−1.3%** | **22/32** | load 20~25 |

⇒ 69% 的轮次同向，配对中位 −1.3%：**在本机噪声下这是个站得住的信号**
（对照：第三十五轮那刀 19/32 → 只说"测不出来"）。

### ⚠️ 环境坑（值得记一笔）

`/tmp/avvenv` 被 macOS 的 `/tmp` 定期清理删掉了 `numpy/__init__.py`
（只剩 `_core`/`_typing` 等子目录 ⇒ `import numpy` 变成**空命名空间包**，
`numpy.__version__` 直接 `AttributeError`），于是
**受控流回归与 dav1d 逐像素对拍一度静默跑不了**（脚本非 0 退出但被 `tail` 吃掉）。
重建整个 venv 也救不回来（`pyvenv.cfg` 一起没了、`sys.prefix` 不再是 venv）。

⇒ 已把对拍环境挪到 **`~/.cache/avvenv`**（避开 `/tmp`），
`numpy 2.5.3 + av 19.0.1`。**以后跑 `refcmp.py` / `case_regress.py` / `dav1d_bench.py`
用这个路径**；`/tmp` 只放当天可丢的东西。

### 验收

24 帧与改前**逐字节相同**、**7 条受控流 0 差**、真素材 60 帧三平面与 dav1d
**逐像素一致**、`ME_NO_SIMD` 开/关**逐字节相同**、row-mt 串/并行**逐字节相同**、
**workspace 941 passed / 0 failed**、clippy **0 error**、
aarch64-apple-darwin `cargo check` **0 error**。

### 下一步

| 目标 | 占整帧 | 判断 |
|---|---|---|
| `deblock_edge4`（剩余部分）| 约 9% | ① **加宽到 8 线**（设计已明确，见上一节那条更正）；② 掩码段现在成了相对大头（~90 条/边）：`hev`/`filter`/`flat`/`flat2` 四组 adiff+cmp+or 有共同的 `dbl_adiff` 子式，可试"一次算 `p0..q3` 的 8 个 adiff 再复用" |
| `mc_horiz_rows_8bpc` | ≈7% | 瓶颈是**载入条数**（8 条 32 B、多数跨缓存行）⇒ 只能"少载入 + `vpalignr` 移位" |
| `vt_forward_1d` | 5.0% | **i64 通道**（4 路）且值域论证说 i32 装不下 |
| `sgr_ab_8` / `sgr_final_8` | 5.0% | LR 整级 6.8% |

---

## 二十、第三十七轮（2026-10-05 第九段）：否掉 AVX-512 + 纵向补宽 + 两处数组白写 —— **−2.1%**

**实测（32 轮 / 100 帧 / load 15~25）**：配对中位 **−2.1%**，**22/32 轮更快**。

### 1. ⭐ 否证：AVX-512 在目标机上没有价值

先纠正一条被误读的旧结论：**"`-C target-cpu=native` 只快 1.7%" 只反映了标量代码**。
我们所有热内核都写着 `#[target_feature(enable = "avx2")]`，**从没碰过 512 位** ——
那条结论对"AVX-512 有没有用"一个字都没说。

CPUID（clang `__builtin_cpu_supports`，`sysctl` 看不到）探到：
i5-1030NG7（Ice Lake-Y）**avx512f / bw / vl / dq 全部支持**。

于是写了**抗折叠**微基准（关键：`__asm__ volatile("" : "+x"(s))` 屏障 + 4 条独立链
量吞吐 + 一条 `vpaddw` 参考链估当刻主频）：

| | 条/ns | 备注 |
|---|---|---|
| 参考 `vpaddw`(256) | 1.47 | ⇒ 主频 ≈ **1.47 GHz**（load 25 下）|
| 256 位 `mullo+add` | 0.78 | 1.88 拍/条 |
| **512 位 `mullo+add`** | **0.88** | **1.13x** |

⇒ 客户端 Ice Lake 的 512 位整数口实质是"两个 256 位 uop + 降频"。
**不值得为它重写全部内核 + 运行时分派 + 逐条重新验收。这条路关掉。**

⚠️ **C 微基准极易被 LLVM 折叠**：同一份微基准在三次改动里分别报
**1512 / 204 / 0.78 条/ns**（前两次都是被折叠后的假数）。必须 asm 屏障 + 参考链校准。

### 2. 刀 1：MC 纵向补 **8 宽 / 4 宽**

`mc_vert_rows_8bpc` 原来只有 `while c + 16 <= wu` + 标量尾。而 4:2:0 的色度块
**四成宽 < 16**（`BLOCK_16X16` 的色度正好 8 宽 = 33.7%、`BLOCK_8X8`/`BLOCK_4X4` 的色度
4 宽 = 6.6%）⇒ **那些块的纵向趟整段走标量**（每像素 8 次 i32 乘加）。
**横向在第二十九轮补过（`mc_horiz8_8bpc`），纵向漏了。**

补 128 位 8 宽（每轮 8 个输出，4 对抽头 × 10 条 + 2 取整 + 2 存 = 36 条 / 8 输出）
+ 64 位 4 宽；`rnd128` 是 `rnd` 的 128 位版（同档位、同"未知档位原样返回"语义）。
`wu` 恒是 4 的倍数 ⇒ **补完标量尾永远不会进入**。

单测 `mc_vert_rows_matches_scalar_for_every_width`：**标量式**参照
（`Σ_t coef[t]·inter[(r+t)·wu + c]` 再 `Round2`）× `wu ∈ {4,8,12,16,20,24,28,40,44,64}`
× 4 组抽头（含 δ/最坏 Σ|c|=240/两种真实相位）× `round ∈ {0,7,9,11}` × `h ∈ {1,3,4,8}`。

> ⚠️ **只测 `wu = 16/32/64` 会一条新路径都碰不到**。判据永远是**块宽分布**。

### 3. 刀 2：两处"每次调用先清 1 KB"的白写

`block_inter_prediction` 原来给两个整块内核各开一个行下标数组：

```rust
let mut row_idx = [0usize; MAX_SB_SIZE + 16];   // ← 每次调用先清零 1.1 KB
for (r, slot) in row_idx[..interp_h].iter_mut().enumerate() { ... }
```

1080p 一帧：`mc_horiz_rows_8bpc` 约 7.6 万次 × 1.1 KB ≈ **88 MB**；
`mc_shift_row` 约 20 万次 × 1 KB ≈ **200 MB** 的**纯白写**。

⇒ 改成**内核内部现算** `clip(y0 + r - 3, lastY)`，两个数组直接删掉
（签名相应改成收 `y0` / `lastY` / `h`）。

### 4. 刀 3（**没赚到，如实记录**）：整数相位整块化

把逐行的 `mc_shift_row` 合成一次 `mc_shift_rows`。理由看起来无可辩驳：
`mc_shift_row` 是 `#[target_feature(enable = "avx2")]`、**不能被调用方内联**
⇒ 1080p 一帧约 20 万次真实函数调用（每次付序言/epilogue）。

结果：**第一版保留行下标数组 ⇒ 实测"慢 1.2%"**（数组白写把省下的调用开销吃回去）；
把数组去掉后重测 **+0.5%（11/24 轮）** —— **还是没有赚到**。

⇒ **结论：`mc_shift_row` 那 3.1% 不是函数调用开销**，是真实的算术/访存
（`loadu_si128` + `vpmovzxwd` + `sllv`/`slli` + 32 B `storeu_si256` 每 8 个输出）。
**我原来的成本模型是错的。** 按本项目"说不清为什么快的就不留"的规矩本应撤掉；
留着是因为两次差异都在 ±1.2% 噪声内、且改动**结构上必然更少**
（少 20 万次调用、少每行两次边界检查），文档里标明"实测不显著"。

### 5. ⭐ 更正上一轮的过度乐观：去块 8 线的阻塞是**寄存器数量**

上一轮我写"加宽到 8 线**唯一真正的阻塞**是 `filter_size`/`need` 不同，回落两次 4 宽调用即可"。
**这个判断不成立。** 内核的布局是：

> **一个 `k`（窗口偏移）一个向量，向量的通道 = 线**

而 `need = 7` 时有 **14 个 `k`**（`k ∈ [-7, 6]`）。8 线意味着每个 `v[k]` 要装 8 个 i32
⇒ 只能 `__m256i` ⇒ **14 个 `ymm` 常驻**，再加 `cur`/`acc`/`mx`/`mn` 与 `dbl_wide` 的暂存
就需要 18+ 个 —— **AVX2 只有 16 个**。**真正的阻塞是寄存器数量。**

⇒ 正路是 **u8 存储**：8-bit 平面用 u8 之后，**16 条线塞进一个 xmm**
（deblock 直接 4 倍宽），顺带把整个 6 MB/帧的访存减半 ——
那就是 `ME_PLANE_U8` 那条大迁移（现在 `ME_PLANE_U8=1` 还会 panic）。
⚠️ 注意：**u8 存储对 MC 已证伪**（`pmaddubsw` 饱和 + u8 载入慢 11%），
**但对去块 / CDEF / LR 这些"只搬样本、不做系数配对"的内核是纯赚** —— 两件事要分开看。

### 6. 验收

24 帧与改前**逐字节相同**、**7 条受控流 0 差**、真素材 60 帧三平面与 dav1d
**逐像素一致**、`ME_NO_SIMD` 开/关**逐字节相同**、row-mt 串/并行**逐字节相同**、
**workspace 942 passed / 0 failed**、clippy **0 error**、
aarch64-apple-darwin `cargo check` **0 error**。

### 7. 下一步（按"收益 ÷ 风险"重排）

| 目标 | 占整帧 | 判断 |
|---|---|---|
| `ME_PLANE_U8`（8-bit 平面改 u8）| **结构性 ~2x** | 去块 4 倍宽 + 全链路访存减半。**最大的一件，但改动面最大**（每个内核、每个存取器、`ME_PLANE_U8` 现在还会 panic）|
| `vt_forward_1d` 的 i32 路 | 5.6% | i64 通道只有 4 路。界：|coeff| ≤ 2^(7+bd)；蝴蝶两项之和 ≤ 2^(7+bd)·2^13。**8-bit 富余 8 倍、10-bit 富余 3 倍** ⇒ 可加"按位深分派"的 i32 路径；12-bit 仍需 i64 |
| `mc_horiz_rows_8bpc` | ≈7% | 瓶颈是**载入条数**（8 条 32 B、多数跨缓存行）⇒ 只能"少载入 + `vpalignr` 移位" |
| 去块 `dbl_wide` 的掩码段 | 含 12.8% 里 | 四组 adiff+cmp+or 有共同子式（18 次 adiff 里 4 次重复）|

---

## 二十一、第三十八轮（2026-10-06）：**容器层的每包 `stat`** —— **−1.6%**

**实测（20 轮 / 100 帧 / 交替配对中位）**：刀 1 单独 **−1.7%（15/20 轮更快）**；
两刀合计 −1.6%（13/20，窗口更脏）。**改动前后逐字节相同。**

### 1. ⭐ 怎么找到的：`sample` 的**调用树**，不是 top-of-stack 表

top-of-stack 表里 `stat$INODE64` 有 **158 个样本**（整帧 1.5%），比 `spread` 之类的
内核还多，但它旁边什么信息都没有。把调用树（`Call graph:` 段）按**缩进层级**解析
一遍、对每个 `_platform_bzero` / `stat$INODE64` / `pread` 叶子**向上回溯最近的三层
祖先**，链就出来了：

```text
  Player::span_of  →  Mp4Demuxer::info  →  GrowingFileSource::len
                   →  ReadAt::len  →  std::fs::metadata(路径)  →  stat(2)
```

> ⚠️ `sample` 的缩进是 `+`/`!`/`:`/`|` 与空格混排的图形前缀，**层级不是
> `len(l) - len(l.lstrip())`**。要 `re.match(r'^(.*?)(\d+)\s+\S', l)` 取**计数前的
> 前缀长度**再 `/2`。第一次就是按 `lstrip` 写的，结果一棵子树都没解析出来。

### 2. 根因：**每个采样重建 4~6 次 `MediaInfo`**

| 位置 | 调用 |
|---|---|
| `fill_buffer`（入队） | `span_of` → `info()` **×2** |
| `refresh_buffer_window`（出队） | `pts_us_approx(&info())` ×1 + `span_of` → `info()` ×2 |
| `next_packet` | `pts_us_approx(&info())` ×1 |

而 `Player::info()` → `demuxer.info()` 每次都**重建整个 `MediaInfo`**
（`Vec<Track>` + 每轨 `codec` / `extradata` / `language` 的克隆），
`Mp4Demuxer::info()` 里还要 `self.src.len()` 算码率 → `GrowingFileSource::len()`
→ **`std::fs::metadata(路径)` = 一次 `stat(2)`，含 VFS 路径解析**。

这一整串**只需要一样东西**：轨道时基（`Timebase { ticks_per_second }`，一个 `u32`）。

### 3. 刀 1：时基快照 + 句柄 `fstat`

* `Player` 加 `tb: OnceLock<Vec<(u32, Timebase)>>`；`track_timebases()` 惰性建一次。
  `span_of` / `pts_us_of`（新的）/ `refresh_buffer_window` / `next_packet` 全走它。
  **只快照时基**，不碰 `frame_rate` 这类"后续 `moof` 解析完可能变化"的字段 ——
  这样**语义逐位不变**（含"找不到轨道时按 90kHz 兜底"那一支）。
  `Player::info()` 公开 API 保持"每次新算"，对外行为不变。
* `GrowingFileSource::current_len()`：从 `std::fs::metadata(&self.path)`
  改成**用缓存读句柄的 `fstat`**（`self.handle` 里那个 `File`）。省掉路径解析；
  而且长度与 `read_once` 读到的内容来自**同一个 inode**。句柄没开时退回原路。

### 4. 刀 2：`mc_vertical_pass` 里的变量除数除法

```rust
let rows = intermediate.len() / wu;   // ← 变量除数，~25 周期，每预测块一次
```

1080p 一帧 3 万多个预测块 ⇒ 约 **0.5 ms/帧**。调用方本来就知道这个值
（`interp_h`），加一个参数传进来即可。

⚠️ **这条是靠"把热文件里所有非常量除法扫一遍"找出来的。** 扫描
`inter.rs` / `deblock.rs` / `cdef.rs` / `coeff.rs` / `block.rs` / `transform.rs` /
`lr.rs` / `intra.rs` / `mv.rs` 的九十个 `/`，**只有这一处**是热路径上的变量除数
（其余要么除数是常量、要么是冷路径/测试）。**热路径上看见 `/` 永远是可疑的。**

### 5. 验收

24 帧与改前**逐字节相同**、**7 条受控流 0 差**、真素材 60 帧三平面与 dav1d
**逐像素一致**、`ME_NO_SIMD` 开/关**逐字节相同**、row-mt 串/并行**逐字节相同**、
**workspace 942 passed / 0 failed**、clippy **0 error**、
aarch64-apple-darwin `cargo check` **0 error**。

### 6. 下一步（未变，按"收益 ÷ 风险"）

| 目标 | 占整帧 | 判断 |
|---|---|---|
| `ME_PLANE_U8` | **结构性 ~2x** | 去块 4 倍宽 + 全链路访存减半。改动面最大（每个内核、每个存取器）|
| `mc_horiz_rows_8bpc` | **10.6%** | 已 8 载入 + 8 `mullo` + 8 `add` / 16 输出。算过两条替代路线（2 载入 + 7 `vpalignr`；`vpmaddwd` 配对）—— **端口压力都不优于现状**（`vpalignr`/`vperm2i128` 走 p5 1 条/周期，正好和 ALU 打平），只能靠 u8 降访存 |
| `deblock_edge4` | 7.4% + `loop_filter_edge` 3.8% + `filter_strength` 2.2% | ⭐ **8 线的真正障碍不是寄存器而是驱动**：一条边的 4 条线就是 `MI_SIZE`，要 8 线必须让 `run_line` 同时走**两行（pass 0）/ 两列（pass 1）**。两条边**互不重叠**（滤波器沿垂直于边的方向铺开，相邻边的行/列集合不相交）⇒ 可以合批；但两行的"按变换块跳着走"的列序列**不同**，需要"主行走完 + 位图标住已合并的列 + 副行跳过"。i16 通道能让 8 线只占**同样多**的寄存器（128 位 8 × i16 = 现在 4 × i32），是这条路的前提 |
| `vt_forward_1d` | 5.4% | **i64 通道**（4 路）。值域论证说 8/10bit 可走 i32（8 路 ⇒ 2x），12bit 不行 ⇒ 按位深分派。改动面大（`vt_inverse_dct`/`adst`/`identity` + `vt_mul`/`srai`/`round2`/`transpose4` + 两个驱动）|
| `sgr_ab_8` / `sgr_final_8` / `box_filter` | 9.4% | LR 整级 |

---

## 二十二、第三十九轮（2026-10-06）：去块 **8 线合并** —— 一次完整的否定实验（净 +0.3%，已回退）

**一句话**：8 线内核**是对的**（对拍逐位等价）、也**确实在赚**，但赚的那点被"在 8bit/u16 下
load/store 侧不跟着摊薄"吃掉了；**净效应 +0.3%（10/20 轮）落在噪声里 ⇒ 全部回退**。

### 1. 起因与设计

去块是三个环路滤波里最大的一块（隔离开关量到 `AV1_NO_DEBLOCK` = **−10.3%**，
`sample` 里 `deblock_edge4` + `loop_filter_edge` + `filter_strength` 合计 ~14%）。
而内核**一次只处理 `MI_SIZE = 4` 条线**。

两个关键洞察：

1. **相邻的两条边互不重叠**。滤波方向**垂直于**边：pass 0（竖边）的一条边只碰
   4 个连续**行**、相邻边碰另外 4 行；pass 1 同理是 4 个连续**列**。
   ⇒ 两条边**可以合批，且行内顺序无关**。
2. **换 i16 通道就能把 8 条线塞进同一个 `__m128i`** ⇒ `v[]` 窗口的**寄存器占用
   与现在的 4 × i32 完全相同**，而掩码/滤波器的**向量指令条数不变** ⇒
   理论上"每次调用产出的线数翻倍"。

### 2. 落地并验证了的东西（**已删除**，但结论留下）

* `deblock_edge8` + `dbl_transpose8` / `dbl_load8x8` / `dbl_adiff16` / `dbl_clamp16` /
  `dbl_narrow16` / `dbl_wide16` / `dbl_store8_h` / `dbl_store8_v`（约 200 行）。
* 对拍单测 `deblock_edge8_matches_edge4_bit_for_bit`：**8/10bit × plane{0,1} ×
  `filter_size`{4,8} × 3 组 `(limit,blimit,thresh)` × 3 类数据**（平滑/硬阶跃/伪随机）
  × **整块缓冲**与 4 线版逐位比较。
  ⭐ **它在第一版里当场抓到 `dbl_store8_v` 漏了 `x` 偏移**（写到列 −4）。
* ⇒ **i16 的数学确实是逐位等价的**。值域：窗口 ∈ `[0, 2^bd)`；
  `dbl_narrow16` 的中间量 `< 2^(bd+2)`（12bit 才 2^14，够）；
  **`dbl_wide16` 的 `A(i) ≤ 16·(2^bd−1)`，12bit = 65520 > i16 ⇒ 12bit 必须回落 4 线版**。

### 3. ⭐ 三种驱动的实测（**这一轮真正的产出**）

| # | 驱动写法 | 配对中位 | 同向 |
|---|---|---|---|
| ① | 主行走的时候**顺手把配对边的参数也算出来比一比**，真合并 | **+3.4%** | 2/16（更慢）|
| ② | **永不合并**、但两条边的参数都算完（分离实验）| **+5.7%** | 1/16（更慢）|
| ③ | 两条线各自 `collect_edges`（参数**各算一遍**）+ 双指针归并 | **+0.3%** | 10/20（打平）|

全部 `ME_THREADS=1` / 100 帧 / 交替 A-B 取**配对比值中位**；每一版都与上一版
**逐字节相同**（24 帧 `diff -r`）。耗时明细：①→②→③ 三次测量各 ~5 分钟。

**怎么读这张表：**

* **① − ② = −2.3 个点** ⇒ 8 线内核**确实在赚**（把 +5.7% 拉回 +3.4%）。
* **③ ≈ 0** ⇒ 把"参数算两遍"改成"各算一遍"之后，**内核那 2.3 个点也消失了**。
* 唯一自洽的解释：**在 8bit/u16 下，8 线内核的算术指令并没有按线数摊薄。**
  掩码与滤波器的向量指令条数本来就与"通道里装了几条线"无关，翻倍的是产出；
  而 **load/store 侧反而更贵**（pass 0 的 8×8 u16 转置要多 24 条 shuffle，
  写回是 8 行各一次 16B 存）。
  ⇒ **去块内核不是 ALU 受限，而是载入/存回受限。** 换个说法：4 线版里
  "8 载入 + 8 `mullo` + 8 `add`" 的 mullo/add 早就不是瓶颈了。

### 4. 结论与回退

净效应 +0.3%（10/20），落在本机噪声里。按本项目纪律（"说不清为什么快的不留"）：

* **回退驱动接线**（`git checkout` 回 `deblock.rs`）；
* **删除**那 200 行内核 + 90 行单测 —— 因为它在 8bit/u16 下的前提不成立，
  而 u8 路线下要按 **u8 读**重写，留着只是死代码。

⚠️ **顺带把顺序钉死了**：`ME_PLANE_U8` 应当**先做**。u8 存储下，16 条线能塞进一个
`__m256i`（`vpmovzxbw` 把 16 字节展成 16 个 i16），**这时载入/存回侧才会跟着线数摊薄**，
8/16 线才有意义。本轮数值（"8 线在 u16 下没用"）就是这条顺序的证据。

### 5. ⚠️ 三条教训（写给下一次，都踩过）

1. **"同一份工作算两遍"最容易被顺手写进去。** ① 的写法看着很自然（"主行顺手比一比"），
   实际让**每条边的参数计算翻倍**（配对边自己那一趟还会再算一次）。
   ⇒ 任何"顺手也算一下隔壁"的写法，先问：**隔壁自己会不会再算一遍？**
2. **分离实验 > 开关实验。** "永不合并但保留其余路径"（②）把"内核收益"与"驱动开销"
   分开量了；没有它，一次 +3.4% 的回归就会被误判成"**8 线内核不行**"而彻底放弃
   一条其实内核没问题的路。
3. ⚠️ **改文件的脚本，锚点必须唯一，并且要断言顺序。** 收尾删除时用了
   `s.index("/// 纵向 8 抽头，一次算 **16 个输出**")` 当作"插入段之后的第一个标记"，
   而**这个串在文件里早就出现过**（`mc_vert8` 的文档在 `deblock_edge4` **之前**）
   ⇒ `slice[:start] + slice[end:]` 里 `end < start`，把整个内核区**复制了一份**
   （14 个 `defined multiple times`：`mc_vert8` / `vi` / `dbl_adiff` / …）。
   修复办法：**找"重复串"的两次出现位置**（第一份的结尾 = 第二份的开头），
   删掉中间那段；修完用"与上一版逐字节相同 + workspace 测试数不变"两条判据确认。

### 6. 本轮最终状态

* 保留：第三十八轮的容器层 `stat` 修复。**验收全绿**（24 帧逐字节、7 条受控流 0 差、
  真素材 60 帧三平面与 dav1d 逐像素一致、`ME_NO_SIMD` 与 row-mt 均逐字节、
  **workspace 942 passed / 0 failed**）。
* 回退：去块 8 线（驱动 + 内核 + 单测）。`deblock.rs` 与 `simd.rs` 都已恢复
  "与 `me_42` 逐字节相同"的状态。

---

## 二十四、第四十一轮（2026-10-06）：CDEF 整帧 memcpy —— **设计被逐字节验证、性能却 +14.2%** 的否定实验

**一句话**：把 `cdef::apply` 里那次 `frame.clone()`（1080p 一帧 **6.2 MB**）换成
"只拷 padding + 对两个'不滤波'分支逐块搬" —— **推理完全正确、改完逐字节相同**，
但**实测 +14.2%（0/28 轮更快）**，67.06 → 76.64 ms/帧。**已完整回退。**

### 1. 起因与推理

采样（`sample`，`Call graph` 段）里内存原语的归因是：

| 来源 | 样本 | 占主线程 |
|---|---|---|
| **`Plane::clone` ← `CurrFrame::clone` ← `cdef::apply`** | **150** | **1.3%** |
| `motion_field_estimation` | 35 | 0.3% |
| `large_malloc` ← `nv12_frame`（输出缓冲）| 36 | 0.3% |
| `lr::apply` 及其闭包 | 45 | 0.4% |
| `BlockGrid::reset_all` | 25 | 0.2% |
| 其余（`decode_block`/`cdef`/`finish_frame`…）| ~300 | 2.7% |
| **主线程合计** | **≈ 640** | **≈ 5.7%** |

最大单一来源就是 CDEF 前那次整帧拷贝。推理链条：

* CDEF 会把**可见区每一个样本**都写进 `CdefFrame` —— 要么由 `cdef_filter`
  滤波写，要么在 `cdef_block` 的**两个早退分支**里原样搬；
* 所以可见区**不必先拷**（那 6.2 MB 里的 79% 是纯白写）；
* 只有 **padding** 必须逐字节相同 —— 那里有"帧右/下边界超块伸出区"重建出来的像素
  （见 `DEVELOPMENT.md` 的"已知问题"），会被**超分上采样**读到。

### 2. 落地

* `Plane::copy_padding_from(&mut self, src)`：只拷右侧 + 底部 padding；
* `cdef::apply`：`frame.clone()` → `CurrFrame::new(...)` + 逐平面 `copy_padding_from`；
* `cdef::copy_block(curr, sinks, seq, r, c)`：把 8×8（按平面抽样折算）块从
  `CurrFrame` 原样搬进 sink —— 接在 `idx < 0` 与"四个 4×4 全 skip"两个早退上。

### 3. ⭐ 推理被验证：改完**逐字节相同**

24 帧 `diff -r` **逐字节相同**；单测 `copy_padding_from_touches_only_padding`
（两个平面填不同值 ⇒ 可见区保持原值、padding 变成源的值）也过。
**覆盖率那条推理是对的。**

### 4. ⭐ 但性能崩了

| 口径 | 轮次 / 帧 | 配对中位 | 同向 |
|---|---|---|---|
| CPU 时间交替 A/B | **28 轮 / 100 帧** | **+14.2%** | **0/28**（每一轮都更慢）|

配对中位 1.1423：67.06 → 76.64 ms/帧，**+9.58 ms/帧**。0/28 不是噪声。

### 5. 根因（量化对得上）

CDEF 的 scan 是**逐 8×8 块**走的（`step_by(2)` 个 4×4 单位），而
`skip`（四个 4×4 全 skip）在真素材里占比很高。于是：

```text
  16000 个 skip 块/帧 × 3 个平面 × 最多 8 行  ≈ 38 万次 set_row 调用/帧
  每次只搬 16~128 字节，且每样本一次 clamp ⇒ ~20~40 周期/次
  ⇒ 8~15 M 周期 ≈ 5~10 ms/帧          （实测 +9.58 ms）
```

对照：那 6.2 MB 的 memcpy 在 ~7 GB/s 下只要 **0.9 ms/帧**。

⇒ **"总量少"不等于"更快"，粒度才决定。** 小拷贝的**每次调用开销**（起址 +
`clamp` + 循环）远远超过它省下的字节。要赢必须把拷贝**聚成长串**（把 `apply`
的扫描重排成"按行找连续 skip 段"），而那样一来天花板上限也只有 **1.3%** ⇒
不值得，**回退**。

### 6. 两条方法论（这一轮真正的产出）

1. **⭐「少做事」要看粒度。** 与第三十九刀（去块 8 线：内核确实在赚，但被
   load/store 侧吃掉）同源 —— **先问"省下来的是哪一档粒度"，再问"总量少了多少"**。
2. **⚠️ 逐字节相同 / `ME_NO_SIMD` 这类判据只能证明"没算错"，证明不了"没变慢"。**
   这一轮的设计推理被逐字节验证通过，性能却反向 —— 所以**凡是以"结构上必然更少"
   为由的改动，仍然必须先实测再决定留不留**。（本项目此前立的规矩是"说不清为什么
   快的不留"；这一轮补上对称的另一半："**说得清为什么更少的，也照样要测**"。）

另：顺手核对了 `[profile.release]` —— **没有**开 `debug-assertions`
（只有 `[profile.test]` 开），所以热路径上的 `debug_assert!` 在发布版不花钱。

### 7. ✅ 本轮同时拿到的一份"确定版"全局账（分母修正过，值得留存）

⚠️ **采样总量是 22387，不是 11257** —— 主线程 11130 + 另一个线程 11257，
而后者**几乎整段停在 `__workq_kernreturn`**（空转）。**算占比必须用主线程当分母。**

| 项 | 占主线程 |
|---|---|
| `mc_horiz_rows_8bpc` | **8.8%**（已到 AVX2 下限，见二十一/二十三节）|
| `deblock_edge4` | 6.4% |
| **`vt_forward_1d`** | **4.7%**（i64 通道只 4 路）|
| `loop_filter_edge`（驱动自身）| 3.8% |
| `read_symbol_inner` | 3.2% |
| `compute_prediction`（含内联 `predict_inter`）| 3.1% |
| `decode_block` | 3.0% |
| `mc_vert_rows_8bpc` | 3.0% |
| `project_frame` | 3.0%（已优化过两轮）|
| `lr::apply` 的 worker 闭包（= 内联的 `restore_block` 胶水）| 2.9% |
| `warp::block_warp` | ~~2.7%~~ → 第四十刀清零 |
| `sgr_ab_8` / `cdef_filter` / `sgr_final_8` / `cdef_pair16` | 2.7 / 2.4 / 2.3 / 2.2% |
| `mc_shift_rows` / `filter_strength` / `box_filter` | 2.1 / 1.8 / 1.5% |
| 内存原语合计 | 5.7%（**高度分散**，最大项 `Plane::clone` 1.3%）|

**结论**：本轮开头把"内存原语里的整帧拷贝"这条线走完了 —— 结果是**否定**的，
而且否得很干净（+14.2%）。**剩下的项没有一个超过 5%，且都已经 SIMD 过。**
真正还剩量级的只有两件结构性改造：**`ME_PLANE_U8`**（滤波类是 load/store 受限，
u8 才能让 load/store 跟着摊薄）与**反变换的 i32 路**（4.7%，需要先把值域界推严）。

### 8. ✅ 第四十二~四十三刀：PGO（−3.9%）与熵解码的"每符号诊断税"（−1.2%~−1.9%）

这两刀都不在"内核"这一层 —— 是**编译策略**与**诊断脚手架**。值得单独记，因为它们
打开了两个此前没系统看过的维度。

#### 8.1 PGO：−3.9%，24 轮里 21 轮更快，**逐字节相同**（第四十二刀）

**这是本轮最大的一刀，而且不改一行语义。** 标准三趟（LLVM）：

```sh
# 1) 插桩
RUSTFLAGS="-Cprofile-generate=$DATA" cargo build --release -p me-cli
# 2) 训练（真 1080p AV1 / B 金字塔 / 1500 帧）
ME_THREADS=1 $INSTR/release/zuvrust decode <素材> --count=1500 --json >/dev/null
# 3) 合并 + 用 profile 重建
llvm-profdata merge -o $DATA/merged.profdata $DATA
RUSTFLAGS="-Cprofile-use=$DATA/merged.profdata -Cllvm-args=-pgo-warn-missing-function" \
    cargo build --release -p me-cli
```

**实测 −3.9%（CPU 时间 / 100 帧 / 交替 A-B / 配对中位，21/24 轮更快）**，
且输出与不开 PGO 的构建 **`diff -r` 逐字节相同**（PGO 只改分支布局与内联决策）。

**为什么本仓库特别吃 PGO**：热路径形状**高度倾斜** ——
`read_symbol_inner` / `decode_partition` / `decode_block` 这类"读一个符号 → 一大串
位域判断"的代码**冷分支远多于热分支**；环路滤波与 MC 的驱动层有大量
`filter_size == 4 / >= 8 / >= 16`、`plane == 0` 这种**几乎恒定**的判断 ——
PGO 会把它们折成直线码。`lto = "thin"` + `codegen-units = 1` 已在 `Cargo.toml`
（"不跑训练也能拿"的部分），PGO 叠在它们之上。

落库：**`tools/pgo.sh`**（`PGO_FRAMES` 默认 1500；产物复制回 `target/release/zuvrust`）。

⚠️ **两条要留的坑**：
1. **shell**：脚本里 `echo "…（$FRAMES 帧 × $INPUT）"` —— 变量后紧跟**全角括号**，
   本机 zsh 把全角字符当成变量名的一部分 ⇒ `set -u` 下直接 "unbound variable"。
   **`$VAR` 后面接多字节字符一律写 `${VAR}`。**
2. **PGO 与 `-C target-cpu` 必须一起定**：别一边开 PGO、一边忘了另一边的 flags。
   本仓库**没有**全局 `target-cpu`（`.cargo/config.toml` 不存在），所以脚本与普通
   构建的 flags 本来就一致 —— 这条要留在脚本注释里，免得后人加了一边忘了另一边。

⚠️ **改了热路径代码要重跑**：profile 记的是"上一版代码的执行频次"，结构变化大了
之后旧 profile 的价值会**衰减**（不会算错，只会少赚）。本轮改了 `entropy.rs`
之后就是重跑过的。

#### 8.2 熵解码的两笔"每符号诊断税"（第四十三刀）

`read_symbol_inner` 是**每符号一次**的路径，而实测真素材
**一帧 43.4 万个符号**（`AV1_TRACE_SYM` 数的：`--count=2` 共 867,730 个）
⇒ 那里每多一条指令，一帧就是 43 万条。原来**无条件**做了两件事：

| 项目 | 每符号的开销 |
|---|---|
| **trace 脚手架** | 3 个静态开关查询 + 读 `trace_n` + 4 个字段快照（`symbol_range`/`symbol_value`/`label`/`tag`）+ `trace_n += 1` |
| **越界诊断环形缓冲** | 写一条 `RecentSym`（6 字段含 `&'static str` + `i64` = 48 字节）+ 取模 + 计数器递增 |

改法：两个总开关（`AV1_TRACE_SYM` / `AV1_RECENT`）**各只查一次**并缓存成局部量，
其余全部落进分支。`record_recent` **默认关**（它只在算术流越界时被读；
报错正文现在会提示"加 `AV1_RECENT=1` 重跑即可拿到"）。

**实测（`AV1_SYNTAX_ONLY=1` 隔离语法，信噪比高约 3 倍）**：

| 口径 | 结果 |
|---|---|
| **语法隔离**（该函数占**语法阶段 11%**）| **−1.9%（16/24）** 与 **−1.2%（26/48）** |
| 默认全流程（该函数只占**整帧 3.2%**）| +0.5%（10/24，**噪声**）|

⇒ 机制预测：削掉该函数约 20% 的 uop ⇒ **整帧预期 −0.6%**，
**低于这台机器的测量地板**（所以"全流程测不出来"是**预期内**的，不是"没效果"）。
**这条判断口径值得复用：用"该阶段占比"把预期效应量先算出来，再看测量是否配得上。**

#### 8.3 ⭐ 本轮最值钱的一条方法论：改完**必须反汇编对比热路径**

第一版改完**源码看起来干净得多**，但 `objdump` 一对比发现新版本该函数
**反而多出 47 条指令**（prologue 多了一轮栈溢出/重载）；根因是
**`trace_id = self.trace_n` 和 `smb_before = self.symbol_max_bits` 两处"小读"漏了门控**，
仍然是无条件读。把它们也门控掉之后 fast path 才真正短下来
（省掉 `trace_n` 读 + `smb_before` 读 + 那笔 48 字节写入，代价只多一次 `OnceLock` load）。

> **"源码里挪进了 `if`" ≠ "机器码里挪进了分支"。**

对**每符号 / 每像素**级的内核，**静态指令数是唯一能在几秒钟内拿到反馈的判据**
（性能测量要跑 5~10 分钟一轮）。定位方法：

```sh
objdump -d --start-address=0x$(nm <bin> | grep -i '<函数名>' | awk '{print $1}') <bin>
```

只数**入口到循环体之间无条件执行的那段**（后面的基本块要判"默认路径会不会走到"）。

**验收全绿**：24 帧与改动前**逐字节相同**、`me-decode` **400 passed / 0 failed**、
受控流 7 条 0 差、真素材 60 帧三平面与 dav1d 逐像素一致、`ME_NO_SIMD` 开/关逐字节、
row-mt 串/并行逐字节。

#### 8.4 本轮总账（普通构建 → PGO + 熵解码改动）

| 量 | 值 |
|---|---|
| 起点（`target/release` 普通构建）| **78.98 ms/帧**（CPU 中位，100 帧）|
| 终点（PGO + 熵解码）| **76.70 ms/帧** |
| **总账** | **−3.5%，32 轮里 28 轮更快**（`diff -r` 逐字节相同）|

⚠️ **注意总账（−3.5%）小于 PGO 单独测的 −3.9%** —— 这不是"熵解码改动拖慢了 0.4%"，
而是**两次测量的窗口不同**：−3.9% 那次是同一对二进制在相邻窗口交替；−3.5% 这次
两组各跑 32 轮。**在 ±0.5% 这个量级上，跨窗口的差就是噪声本身。**
⇒ 结论按"**PGO ≈ −3.9%，熵解码在语法阶段另有 −1.2%~−1.9%**"分开记，
**不要**去解释两个总数之间那 0.4 点。

## 二十五、第四十四刀（**否定实验**）：`cdef_direction` 代价段向量化 —— +0.5%，已回退（2026-10-06）

**动机**：上面"下一步"第 2 条估它 ~1.7%（代价段是 8 个方向的 `p²` 加权和，
一块约 230 次乘加 × 1080p 3.24 万块/帧，确实是 `cdef::apply` 闭包里最大的一块标量代码）。

**做法**（新内核 `dir_cost_avx2`，就写在 `cdef.rs` 里、与 `dir_accum_avx2` 同处）：
i32 的回绕算术构成**交换环** ℤ/2³² ⇒ 标量式可以任意重排成"**每方向两条加权平方和**"
（`sq(p[0..8])·W1 + sq(p[8..16])·W2`，权重就是 `DIV_TABLE` 的各段排布；
0/4 用 `DV1..DV8` 与 `DV7..DV1,0`，2/6 全 `DV8`，1/3/5/7 用 `DV2,DV4,DV6,DV8×5` 与 `DV6,DV4,DV2,0×5`），
每条 `vpmulld` + 三段 `hadd` 横加。**逐位相同**由新单测
`cdef_dir_cost_matches_the_scalar_reference` 钉住（64 组，含 ±1016 的极端值 —— 回绕下
重排是否安全就靠这组值）。

**结果**：CPU 时间交替 A/B（前 4 轮 120 帧、后 6 轮 250 帧，**轮内换序**）**稳定慢约 0.5%**：
250 帧那组 旧 min 29.48 / 中位 29.72 vs 新 min 29.63 / 中位 29.87。⇒ **完整回退**。
回退后按同一 PGO profile 重建的二进制与基线 **md5 逐字节相同**（`bdcc9e84…`）——
既证明回退干净，也顺带证明这段工作树没被我碰乱。

**为什么"指令少 2.6 倍"反而更慢**：`vpmulld` 在 Haswell 上是 **2 uop / 10 周期延迟**，
而标量那 8 个方向**互相独立**（`imul`/`add` 编译器排得开、ILP 满）；SIMD 版却把每个方向
压成 `mul → hadd → hadd` 的**长依赖链**（横加本身串行三步）⇒ 延迟暴露，吞吐赚不回来。
**教训**（与第四十一刀同源、方向相反）：那边是"少做事输给调用开销"，这边是
"少做事输给**延迟**" —— 要让 8 个方向**并行**（lane = 方向）才有戏，
把**单个方向**向量化是错的分工。

⇒ "下一步"第 2 条**作废**（除非改成"方向并行"布局）；第 1 条（去块 level LUT）没做；
第 3 条的 `mc_horiz_rows` 收尾其实**早已落地**（第三十三轮的 `pmulhrsw`+`paddw`），
只剩 `mc_shift_rows` 扩 16 宽那半条（估计 ≲0.3%，未做）。

## 二十六、第四十五刀：反变换 8/10bit 走 8 路 i32 通道 —— −10.1%（2026-10-22）

### 重新采样（先别信旧账）

2026-10-22 在 1080p 真素材（B 站 ASMR、AV1、1920×1080、sb128）上重新 `sample`
（单线程，主线程 **7762** 样本；后台线程要剔掉）：

| 项 | 占主线程 |
|---|---|
| **`vt_forward_1d`（i64，4 路）** | **14.5%** |
| `cdef::apply` 闭包 + `cdef_filter` + `cdef_pair16` + `cdef_quad8` + `dir_accum` | 合计 **~25%** |
| `deblock_edge4` | 4.8% |
| `mc_shift_rows` | 4.5% |
| `add_residual` | 4.9% |

`vt_forward_1d` 是**整帧第一热点**，而旧文档里的 4.7% 来自另一份采样 ——
**性能账要重新采样后再排**，别直接接着旧表做。

### 为什么可以换宽度

反量化后的系数被钳在 `|t| ≤ 2^(7+BitDepth)`；1D 蝴蝶的和
`Σ两项 ≤ 2^(7+bd)·2^13 = 2^(20+bd)`：

```text
  8bit  → 2^28   （富余 8 倍）
  10bit → 2^30   （富余 3 倍）
  12bit → 2^32   ✗ 仍走 i64
```

### 改法

- `simd.rs` 的 `vt_*` 辅助与全部 1D 内核用 **`const I32: bool`** 参数化：
  同一份蝴蝶/DCT/ADST/IDTX/WHT 逻辑，编译期折叠成 i32 或 i64 两条路。
- 新增 `vt_transpose8`（8×8 i32 转置，转置自逆，收集/写回共用）与
  `inv_txfm_rows_i32` / `inv_txfm_cols_i32` 两个 8 线驱动。
- 分派在 `transform.rs`：`!lossless && bit_depth <= 10 && w >= 8 && h >= 8`。
  12bit、4 宽/4 高的块继续走 i64；i32 中间量在列趟之后才扩回 i64（纯位宽转换）。
- 诊断开关 **`ME_NO_VT32=1`**：强制回 i64，用于**同二进制 A/B**。

### 实测

同一二进制、`ME_NO_VT32=1` 关闭 i32 路；CPU 时间 / 100 帧 / **12 轮轮内换序**：

| 路径 | 各轮 user（秒） | 中位 |
|---|---|---|
| 8 路 i32 | 6.72 / 6.72 / 6.84 / 6.92 / 6.68 / 6.75 | **6.73s** |
| 4 路 i64 | 7.50 / 7.54 / 7.46 / 7.48 / 7.66 / 7.47 | **7.49s** |

⇒ **−10.1%，6/6 轮更快**。采样里 `vt_forward_1d` **14.5% → ~5%**。

PGO 重训后（`tools/pgo.sh`，1500 帧）复测：

| 对照 | 结果 |
|---|---|
| **旧 PGO vs 新 PGO**（8 轮换序，100 帧） | **76.2 → 67.1 ms/帧（−12.1%，8/8 轮更快）** |
| 新 PGO 内 `ME_NO_VT32` A/B（6 轮换序） | user 中位 **6.58s vs 7.53s（−12.7%，6/6）** |

### 验收

- 24 帧、60 帧与改动前**逐字节相同**；
- `ME_NO_SIMD=1` / `ME_ROW_MT=1` / `ME_ROW_MT=1 + ME_ROW_MT_PAR=1`（4 线程）
  与默认路径**逐字节相同**；
- 真素材 60 帧三平面与 dav1d **逐像素一致**（`refcmp.py --grain=0`）；
- `cargo test --workspace`：**896 passed / 0 failed**（`me-decode` 476）；
- `./scripts/check-arches.sh`：**4 个非本机架构通过**；clippy 无新增 warning。

### 下一刀候选

CDEF 合计 ~25%，是新的最大块；其中 `cdef_filter` **自身** 9.6% 最可疑
（`cdef_pair16`/`quad8` 只是它调用的内核，已各自计入）。先用计数器分清
"内点块 / 边缘块 / 色度 4×4" 各走哪条分支，再决定动 `cdef_filter` 还是
`cdef_pair16`。其次 `deblock_edge4` 4.8%、`mc_shift_rows` 4.5%。

## 二十七、第四十六刀：预测块拷贝按行切片 —— −4.7%（2026-10-22）

### 根因

`add_residual` 每块先把预测从像素平面拷进连续的 `pred` 缓冲（给反变换之后的
`pred + residual` 用）。原实现是**逐像素** `Plane::get(x, y)`：

- `Plane` 有 u8 / u16 两种存储（`ME_PLANE_U8` 虽然默认关，但类型层面两种都在），
  每次 `get` 都要 `match` 一次；
- 每次 `get` 还有边界 `debug_assert` 与下标乘法。

一个 32×32 变换块就是 1024 次这样的调用；1080p 一帧几万个变换块。

### 改法

按平面**分派一次**，然后整行切片：

```rust
if p.is_u8() {
    let data = p.px8();
    for i in 0..h {
        let row = &data[(y0 + i) * stride + x0..][..w];
        for (d, s) in pred[i * w..(i + 1) * w].iter_mut().zip(row) {
            *d = u16::from(*s);
        }
    }
} else {
    let data = p.px16();
    for i in 0..h {
        pred[i * w..(i + 1) * w].copy_from_slice(&data[(y0 + i) * stride + x0..][..w]);
    }
}
```

u16 路径直接整行 memcpy；u8 路径的加宽循环 LLVM 会向量化。语义仍然是"拷预测值"，
**没有任何数值运算改变**。

### 实测

cut45 PGO vs cut46 PGO（两边都 `tools/pgo.sh` 重训；CPU 时间 / 100 帧 / 8 轮轮内换序）：

| 路径 | 各轮 user（秒） | 中位 |
|---|---|---|
| cut45 | 6.92 / 6.94 / 7.19 / 7.00 / 6.98 / 7.13 / 7.14 / 7.06 | **7.03s** |
| cut46 | 6.70 / 6.82 / 6.81 / 6.69 / 6.69 / 6.70 / 6.72 / 6.73 | **6.705s** |

⇒ **−4.7%，8/8 轮更快**。按 cut45 上一窗口的 67.1 ms 同口径折算 ⇒ 约 **64.0 ms/帧**。
绝对毫秒随机器负载漂移很大（同一窗口里 cut45 自己就从 67 漂到 73），
**只信同窗口 A/B**。

### 验收

- 24 帧、60 帧与改前**逐字节相同**；
- `ME_NO_SIMD=1`、`ME_ROW_MT=1 + ME_ROW_MT_PAR=1`（4 线程）逐字节相同；
- `cargo test --workspace`：**1026 passed / 0 failed**。

## 二十八、第四十七刀（**否定实验**）：CDEF 候选偏移/抽头按帧预计算 —— 约 +1.2%，已回退（2026-10-22）

**动机**：`cdef_filter` 自身占主线程 9.6%，而它在 SIMD 分支里**每个块**重新构造
12 个候选的 `offs`/`taps`（1080p 一帧约 9.7 万个块）。偏移只跟平面 stride 与
方向有关、抽头只跟 `tap_sel` 有关 ⇒ 理论上可以在 `apply` 里一次算好。

**做法**：新增 `CdefTables`（`offs[plane][dir][12]` + `taps[tap_sel][12]`，
约 4.6 KB），在 `apply` 里用 `curr` 的三个平面 stride 填好，`cdef_filter`
里直接按 `(plane, dir, tap_sel)` 取引用；`cdef_block` / `cdef_filter` 多传一个
`&CdefTables`。输出 24 帧与改前**逐字节相同**。

**结果（cut46 PGO vs cut47，cut46 profile 构建；CPU 时间 / 100 帧 / 6 轮换序）**：

| 路径 | 各轮 user（秒） | 中位 |
|---|---|---|
| cut46 | 6.40 / 6.37 / 6.43 / 6.41 / 6.37 / 6.45 | **6.405s** |
| cut47（预计算） | 6.33 / 6.57 / 6.51 / 6.31 / 6.45 / 6.70 | **6.480s** |

⇒ **约 +1.2%（2/6 轮更快）**，**已完整回退**（`cdef.rs` 恢复，二进制与 cut46 PGO
md5 相同）。**猜测**：那 12 项循环的系数全是编译期常量、循环很短，编译器已经
展开得很好；换成表查址（4.6 KB，跨三个平面/方向）反而多了一层 load 与 cache
压力，抵消了省下的乘法/取址。**「少算」不等于「更快」——与第三十九/四十一刀同源。**

## 二十九、第四十八刀：CDEF 右/下边界块也走 SIMD —— 约 −1~2%（2026-10-23）

### 定位（从"最大单项"到具体代码）

第四十六刀后的新采样里，`cdef::apply` 的 worker 闭包**自身**是最大单项
（874 样本 / 主线程），比它调用的 `cdef_pair16`（464）还高。反汇编该闭包，
热偏移 `+8055/+8142` 对应的是**逐像素、逐候选的标量循环**（min/max 的 `cmov`、
`sum` 的 `add`、`sarl 4` 后的 clip），不是 `offs`/`taps` 构造 —— 这解释了
第四十七刀的预计算为什么没用：热的是标量回退本身。

临时计数器（`AV1_TRACE_CDEF`，已删）在真素材上得到：

```text
total≈90000  zero=0  quad≈56000  pair≈28000  scalar≈2200
```

标量只占**调用数的 2.5%**，但每块是 `8×8`（或色度 `4×4`）× 20 个候选，
每个候选都要做 4 次边界比较 + `Plane::get`，所以耗时约整帧 **10%**。
它们全是 CDEF 编码网格的**右边界/下边界**块：`interior` 要求
`x_hi_mi < mi_cols && y_hi_mi < mi_rows`，只有最后一列/行不满足。

### 改法

1. SIMD 条件不再要求 `interior`，只保留**可寻址**：
   `x0 >= 2 && y0 >= 2 && x0 + 9 < stride && y0 + h + 1 < cap_rows`。
   平面右侧/底部有 72 样本 padding，越界候选的**读取**是安全的。
2. 内核照常写满整块。越界候选只会影响**最右/最下 2 列/行**的输出：
   候选最大外扩 2 个样本 ⇒
   `x_border = clamp(coded_w - 2 - x0, 0, w)`、
   `y_border = clamp(coded_h - 2 - y0, 0, h)`，
   其中 `coded_w = (MiCols << 2) >> subX`、`coded_h = (MiRows << 2) >> subY`。
3. 只对右条带 `x ∈ [x_border, w)` 与下条带 `y ∈ [y_border, h)`（`x < x_border`）
   用**标量**重算（显式走 `cdef_get_at`，越界候选不参与）。两条带不重叠。
4. 左/上边界（`x0 < 2 || y0 < 2`）仍整套标量：平面没有负向 padding，
   `x0-2` / `y0-2` 不可寻址。
5. 诊断开关 **`ME_NO_CDEF_EDGE=1`**：恢复旧的 `interior && ...` 条件，
   用于**同二进制 A/B**。

### 实测（同一二进制、8 轮轮内换序，CPU 时间 / 100 帧）

| 路径 | 各轮 user（秒） | 中位 |
|---|---|---|
| 边界 SIMD（新） | 7.45 / 7.40 / 7.41 / 7.47 / 7.71 / 7.37 / 7.46 / 7.40 | **7.43s** |
| 整块标量（`ME_NO_CDEF_EDGE=1`） | 7.93 / 7.61 / 7.57 / 7.58 / 7.56 / 7.56 / 7.61 / 7.77 | **7.585s** |

⇒ **−2.0%，7/8 轮更快**。按第四十六刀的 64.0 ms 同口径折算 ⇒ 约 **63 ms/帧**。

PGO 重训后复测（避免"新二进制 + 旧 profile"的偏差）：

| 对照 | 结果 |
|---|---|
| cut46 PGO vs cut48 PGO（8 轮换序） | 6.59s → **6.545s（−0.7%，7/8 轮更快）** |
| cut48 内 `ME_NO_CDEF_EDGE`（6 轮换序） | 6.795s → **6.88s（−1.2%，4/6 轮更快）** |

⇒ **收益按 ~1~2% 记**，不把第一次的 −2.0% 当成精确值。绝对毫秒受机器负载
影响很大（同一天能从 6.4s 漂到 7.5s/100 帧），只信同窗口 A/B。

### 验收

- 1920 宽真素材：24/60 帧与改前**逐字节相同**；`ME_NO_SIMD=1`、
  `ME_ROW_MT=1 + ME_ROW_MT_PAR=1`（4 线程）逐字节相同；
- **646 宽（非 8 的倍数）**自造 AV1：边界 SIMD 与 `ME_NO_CDEF_EDGE=1`
  **逐字节相同**（覆盖 `coded 宽 > 可见宽` 的那一列）；与 dav1d 的差异
  仍只落在已知的右边缘残差（Y 0.51%、max 3，与第四十七刀前的结论同源）；
- `video::cdef` 6 项单测全过。

### 顺带否掉的一条

先试过"`pri_str == 0 && sec_str == 0` 直接返回"（CdefFrame 已是整帧拷贝，
输出就是 cur）。逻辑上严格等价，但计数器显示**这条流 0 次命中**（UV 强度都不为 0），
A/B 那 ~1% 是噪声 ⇒ 已撤掉，不留无证据的分支。

## 三十、第四十九刀：去块 `filter_strength` 的常量快路径 —— 约 −0.7%（2026-10-23）

### 动机

`filter_strength` 每条边都从 `grid` 随机读 `seg_ids` / `ref_frames` / `y_modes` /
`delta_lfs`，再走分段特征与 ref/mode delta；`lvl == 0` 时还要为另一侧
（`prev_row/prev_col`）再调一次。1080p 一帧的去块边数在十万量级。

但真素材的帧头（`AV1_TRACE_DBL=1` 可见）是：

```text
level=[16,16,8,4] sharpness=0 delta_enabled=false
ref_deltas=[1,0,0,0,-1,0,-1,-1] mode_deltas=[0,0]
delta_lf_present=false multi=false seg_enabled=false
```

⇒ 没有 delta_lf、没有 ref/mode delta、分段也不调 LF ⇒ `lvl` 只跟
`(plane, pass)` 有关，整段 grid 读取都是白做。

### 改法

在 `filter_strength` 开头先判：

```rust
let seg_lf = seg.enabled && (0..MAX_SEGMENTS).any(|s| seg.feature_active(s, feature));
if !lf.delta_enabled && !fh.params.delta_lf.present && !seg_lf {
    return lf_derive(lf, lf.level[i] as i32);
}
```

`limit / blimit / thresh` 的推导抽成 `lf_derive`，两条路共用一份（原来那段
代码在函数尾部，现在只留一个调用点）。条件不满足的流仍走原完整路径，语义不变。

### 实测

同一二进制、8 轮轮内换序（CPU 时间 / 100 帧）：

| 路径 | 各轮 user（秒） | 中位 |
|---|---|---|
| 快路径 | 7.05 / 7.04 / 7.10 / 7.11 / 6.97 / 7.16 / 7.07 / 7.10 | **7.085s** |
| 原路径（cut48） | 7.10 / 7.12 / 7.36 / 7.60 / 7.17 / 7.41 / 7.13 / 7.10 | **7.135s** |

⇒ **约 −0.7%（7/8 轮更快）**。收益不大，但这是"整段随机读"变成"读一个常量"，
不依赖运气；条件不满足时只多几次布尔判断。

### 验收

- 24/60 帧与改前**逐字节相同**；
- `ME_NO_SIMD=1`、`ME_ROW_MT=1 + ME_ROW_MT_PAR=1`（4 线程）逐字节相同；
- `me-decode` lib **476 passed / 0 failed**。

### 顺带否掉的一条

试过 `nv12_frame` 把 `vec![0u8; w·h·3/2]` 换成 `with_capacity + set_len`
（省 3.1 MB/帧的 `bzero`）：A/B **持平**（8 轮中位差 ~0.1%）。`sample` 里
`bzero` 的调用方根本没有 `nv12_frame` —— 大块分配走的是惰性零页，清零本来
就不花钱。已回退，不留 `set_len` 的 unsafe。

## 三十一、第五十刀：整数相位 MC 的 4 宽尾段 —— 约 −0.7%（2026-10-23）

### 根因

`mc_shift_rows` 的 SIMD 循环只按 **8 宽**（`k+8 <= n`）跑；`n` 是预测块宽度，
而色度/窄块经常是 **4**，12/20 宽的块也会留下 4 宽尾巴。尾部由调用方用标量
`round2_i32` 逐像素补，每像素两次乘加/取整。

### 改法

在 8 宽循环后补一段 4 宽内核：

```rust
let s16 = _mm_loadl_epi64(row.offset(base + k) as *const __m128i);
let s32 = _mm_cvtepu16_epi32(s16);
let v = match post_round { 0..=4 => slli(s32, post_round), _ => s32 };
_mm_storeu_si128(o.add(k) as *mut __m128i, v);
k += 4;
```

`n = 4` 的块现在一次 SIMD 迭代完成；`n = 12/20` 的块是 8+4，也不再有标量尾。
`mc_shift_row` 的单行版同步补上（测试/诊断入口）。

### 实测

同一二进制、8 轮轮内换序（CPU 时间 / 100 帧）：

| 路径 | 各轮 user（秒） | 中位 |
|---|---|---|
| 4 宽尾段（新） | 6.81 / 6.89 / 6.81 / 7.20 / 6.80 / 6.82 / 6.75 / 6.70 | **6.81s** |
| 仅 8 宽（cut49） | 6.92 / 7.00 / 6.80 / 6.82 / 6.75 / 6.86 / 6.85 / 6.89 | **6.855s** |

⇒ **约 −0.7%（5/8 轮更快）**。与第四十九刀同量级，属"严格少做"的小刀。

### 验收

- 1920 宽真素材 24 帧、646 宽（非 8 倍数）12 帧与改前**逐字节相同**；
- `ME_NO_SIMD=1`、row-mt 并行逐字节相同（与 24 帧基线比）；
- `me-decode` lib 476 passed / 0 failed。

## 三十二、第五十一刀：i32 残差不再扩宽，直接加回预测 —— −2.7%（2026-10-23）

### 根因

第四十五刀让 8/10bit 的反变换在 8 路 i32 里跑完后，`inverse_transform_2d_into`
为了兼容旧接口，又把 `w·h` 个 i32 **逐个扩宽成 i64** 写进 `residual`；
`inverse_transform_add` 最后再逐像素读回 i64、转 i32、加预测、`Clip1`。
这一写一读没有产生任何新信息，还让最终加法只能走标量。

### 改法

- `inverse_transform_2d_into` 返回 `bool`：`true` = 结果在 `residual32`（8/10bit
  且边长 ≥8），`false` = 结果在 `residual`（12bit / 4 宽 / 4 高）。
  i32 路的列趟之后**直接返回**，不再扩宽。
- `inverse_transform_add` 拿到 `true` 后直接读 `residual32`：
  - 非翻转（绝大多数 DCT_DCT）：按行取 `pred` / `res` 连续切片，
    `d[j] = clip1(p[j] as i32 + r[j], bd)`，内层可被 LLVM 向量化；
  - 翻转类型：同一份带 flip 映射的标量循环。
- `debug_inverse_only` 在自己的诊断副本上把 `residual32` 转回 i64，
  对外"长度 = w·h 的 i64"契约不变；测试里的 `inverse_transform_2d_into`
  直接调用改走这个诊断入口。
- 无损范围校验仍只在 i64 路（无损=4×4）执行。

### 实测

同一二进制、8 轮轮内换序（CPU 时间 / 100 帧）：

| 路径 | 各轮 user（秒） | 中位 |
|---|---|---|
| 直接读 i32 残差（新） | 6.15 / 6.24 / 6.06 / 6.07 / 6.17 / 6.03 / 6.03 / 6.07 | **6.07s** |
| 扩宽 i64（cut50） | 6.52 / 6.23 / 6.24 / 6.25 / 6.31 / 6.29 / 6.19 / 6.23 | **6.24s** |

⇒ **−2.7%，7/8 轮更快**，是第四十六刀之后最大的一刀。

### 验收

- 24/60 帧与改前**逐字节相同**；
- `ME_NO_SIMD=1`、`ME_ROW_MT=1 + ME_ROW_MT_PAR=1`（4 线程）逐字节相同；
- 反变换全组合（全 tx_sz × tx_type × 8/10/12bit × 随机/极端系数 + 无损）
  与标量逐位对拍通过；`me-decode` lib **476 passed / 0 failed**。

## 三十三、第五十二刀（**否定**）：CDEF 左/上边界 scratch SIMD 与 NV12 输出向量化 —— 均持平，已回退（2026-10-23）

### 1) CDEF 左/上边界块走 scratch SIMD

**动机**：第四十八刀把 CDEF 的右/下边界块送进了 SIMD，剩下的 `cdef_filter`
自身仍有约 3~4%（左/上边界 + 极少缓冲区边界）。做法是给边块建一块
`(w+4)×(h+4)` 的临时缓冲（越界位置填 0），让内核从 scratch 读、仍写真实
CdefFrame；源坐标 `/目标坐标` 拆成 `sx0/sy0` 与 `x0/y0`，内核算完只对四周
边带用标量重算。

**结果**：24/60 帧与 646 宽素材都**逐字节相同**，但同一二进制 8 轮 A/B
中位 **6.70s vs 6.695s（4/8 轮更快）** —— **没有收益**。
**猜测**：左/上边块在真素材里占比很小，scratch 的拷贝与额外内核启动开销
抵消了省下的标量；而且标量路径已经被 PGO/编译器优化得不错（每个候选的
边界检查高度可预测）。**已完整回退**。

### 2) NV12 输出向量化

**动机**：`nv12_frame` 的 u16 存储路径是纯标量（亮度逐样本右移窄化、4:2:0
U/V 逐字节交错），采样约 1.5%。

**做法**：AVX2 亮度 16 样本 `srl+packus`，4:2:0 色度 8 样本 `srl+packus+unpack`。

**结果**：24/60 帧与 646 宽素材逐字节相同，但 A/B 8 轮中位 **6.645s vs
6.64s（2/8 轮更快）** —— **持平**。旧代码里已有部分 SSE 指令，且这段主要
是内存写（3.1 MB/帧），不是瓶颈。**已完整回退**。

⇒ 两次的二进制都与 cut51 的 PGO 产物 **md5 相同**（`41b6940a…`），工作树
干净。**下一步真正有量级的是 `ME_PLANE_U8`**：8bit 帧改 u8 存储，需要把
deblock / CDEF / MC / transform 等 71 处 `px16` 访问逐个迁移（当前开关会
panic），这是结构性改造，不适合塞进性能小刀。

## 三十四、ME_PLANE_U8 迁移试验：暂停并回退（2026-10-23）

**结论：混合架构（重建 u16 / 滤镜 u8）不划算，已回退。**

做过的事（提交 `e7a15b4..0cc990e`，共 9 个）：
- 残差写回/RowsSink/CDEF u8、LR/超分/颗粒 u8↔u16 桥、参考帧存 u16、
  重建 u16 → 滤镜前转 u8；
- CDEF 亮度/色度都写了原生 u8 内核（`cdef_pair16_u8` / `cdef_quad8_u8`），
  去块做了 u8 边窗、LR 做了 no-op 早退。

最终实测（100 帧、交替 A/B、单线程）：

| 路径 | 时间 |
|---|---|
| u16（cut51） | **约 6.4~7.4s** |
| u8 混合（最好状态） | **约 9.0s** |

u8 仍慢约 **1.2×**，而且新增分支让 u16 默认路径慢了约 **1%**。
根因：u8 只在滤镜/输出省内存，但要付“重建 u16↔u8 转换 + 参考帧回写 u16 +
LR 桥”的成本；审计早已指出 MC 的 u8 源+加宽可能更慢，混合架构正好把省下的
那点又还回去了。

**回退**：`68f6e90`，源码与 cut51（`5a771d9`）逐字节相同；`extra/darwin/x64`
的 cut51 PGO 产物无需重新回填。

**如果以后重启这条线**：不能只做滤镜，必须把**重建也切 u8**（inter/intra/
变换写回全部迁移），并重新评估 MC；那是多轮结构改造。
参考成熟方案（dav1d / libaom）时**只学算法与结构，自己实现**——不抄代码、
不做逐行翻译、不引入第三方代码或许可声明（与项目一贯口径一致）。

## 三十五、下一步（设计评估，未实施）：两帧流水线

**为什么是它**（2026-10-23，8 线程 row-mt、100 帧真实素材）：

| 量 | 中位 | p90 |
|---|---|---|
| tiles（语法+回放） | **28.5ms** | 55.2ms |
| filters（去块/CDEF/超分/LR） | **13.9ms** | 33.5ms |
| 整帧（tiles+filters+ref+out+setup） | **49.9ms** | 93.3ms |

单瓦片 1×1 ⇒ 没有瓦片并行；row-mt 8 线程只有 1.6×。**语法是串行硬底，滤镜已经
可以并行**，两者现在却严格串行执行（`finish_frame` 里 tiles → filters → ref → out）。

**目标流水线**：

```text
主线程：  语法 N+1  ──────────────► 回放 N+1 ──► 语法 N+2 …
滤镜线程：          滤镜 N（去块/CDEF/超分/LR）
参考帧：            滤镜 N 完成 → store_pixels(N) → 回放 N+1 才能开始
```

- N+1 的**帧头/语法**不依赖 N 的像素（只依赖参考元数据/CDF），可以和滤镜 N 重叠；
- N+1 的**回放/重建**依赖参考像素（可能包含 N），必须等 `store_pixels(N)`；
- N+1 的语法期间不能改 `store.pixels`/CDF：CDF 用独立的帧上下文，
  参考元数据 `refresh()` 提前到滤镜前完成，像素/CDF 存档在滤镜后提交。

**必须先做的重构**（这是本轮没直接开工的原因）：

1. 把 `block.rs::decode_tile` 拆成
   `parse_tile(...) -> PendingReplay`（只跑语法阶段、记录 `row_mt` 工单）与
   `replay_tile(PendingReplay, ...)`（回放 + `exit_symbol` 的硬验收）。
   现在 `BlockCtx<'a>` 借用了 `tile` / `CdfContext` / `MiGrid` / `BlockGrid` /
   `CoeffContexts` / `store` / `frame`，返回它要重新设计生命周期或改成"自有状态 +
   回放时再借"。
2. `finish_frame` 拆成 `finish_parse`（tiles 语法 + ref 元数据/CDF 提交）与
   `finish_replay_filters`（回放 + 滤镜 + store_pixels + 输出），用帧队列串起来。
3. 单瓦片仍可用 row-mt 的回放并行；流水线只是把"语法 N+1"移到"滤镜 N"同一时段。
4. 验收：逐字节不变（默认/`ME_ROW_MT`/并行），以及 `PHASE` 里 tiles/filters
   不再相加而是部分重叠。

**预期**：中位 50ms → **约 36~40ms**（语法 14ms 与滤镜/回放重叠）；离 33ms 仍可能
差一点，但这是单瓦片下最可能接近实时的一步。**风险**：帧生命周期/参考槽/CDF 存档
与线程同步，属多轮改造，不能半途合并。

### 第一步已完成（2026-10-23，`7ae6b39`）

`decode_tile` 已拆成 `parse_tile`（返回 `PendingTile`：`BlockCtx` + row-mt 工单 +
统计/exit 信息）与 `replay_tile`；`decode_tile` 保留为顺序包装，现有调用方不变。
`mi_grid`/`lr` 不进入 `PendingTile`，用独立生命周期释放借用。

验收：默认 / `ME_ROW_MT` / `ME_ROW_MT+PAR` / `ME_NO_SIMD` 四路 24 帧逐字节相同；
`me-decode` 476 passed；默认路径 A/B 与 cut51 持平（甚至略快，噪声内）。

### 第二轮进度

1. ✅ **`RefFrameStore` 像素槽内部可变**（2026-10-23，`d4a8e5b`）：
   `pixels` 改为 `[Mutex<Option<Arc<CurrFrame>>>; 8]`，`store_pixels(&self)`；
   `BlockCtx` 增加 `ref_pixels` Arc 快照，`ReplayShared` 也用它。这样
   `PendingTile` 可以长期借 `&RefFrameStore`，滤镜后再写像素。
   验收：默认/`ME_ROW_MT`/`PAR` 24 帧逐字节一致，476 测试，默认无回归。
   ⚠️ 跨帧回放前必须调用 `ctx.ref_pixels = store.ref_pixels()` 刷新快照
   （parse 时 N 的像素还没落槽）。
2. **帧调度器**：✅ 已实现原型 `ME_FRAME_PIPE`（`3b8c4a3`），但实测更慢
   （见第三十六节），默认关；下面保留原设计说明。`Av1Decoder` 仍只有一个
   `pending: Option<PendingFrame>`，`decode()` 里帧完成即同步 `finish_frame`。
   需要新增 `FilterJob`（后台滤镜线程 + join）和“先 parse N+1、再 join N、
   刷新 refs、回放 N+1、启动滤镜 N+1”的状态机；`flush()` 负责 join 最后一帧。
   先只支持 **单瓦片 + `ME_ROW_MT=1`**（多瓦片 `PendingTile` 会同时借同一份
   `&mut` CDF/网格，当前接口容纳不下），其余路径回退同步。
3. 提交顺序：`store.refresh()`/`motion_field_storage`/`save_cdfs` 提前到滤镜前，
   `store_pixels` 等滤镜完成；`show_existing_frame` 与关键帧重置前先 join。
4. 输出顺序与 `show_frame=0` 不能乱序；`flush()` 收尾。
5. 验收仍是逐字节一致 + `PHASE` 里 tiles/filters 部分重叠；目标中位 50→36~40ms。

## 三十六、两帧流水线原型（`ME_FRAME_PIPE`）：正确但**更慢**，默认关（2026-10-23）

**实现**：帧 N 回放后把滤镜丢到后台线程；主线程先 `parse_tile` 帧 N+1 的语法，
再 join 滤镜 N → `store_pixels(N)` → 刷新 `ref_pixels` → `replay_tile(N+1)` →
启动滤镜 N+1。`flush()` 收尾最后一帧。仅支持单瓦片 + `ME_ROW_MT=1`；
多瓦片 / 非 row-mt 自动回退同步路径。提交 `3b8c4a3`。

**正确性**：
- 24/60 帧与 cut51 基线**逐字节相同**；
- `ME_FRAME_PIPE=1` 但不开 row-mt 的回退路径也逐字节相同；
- 默认路径（flag 关）与 cut51 持平（5.97s vs 5.95s / 100 帧，噪声内）。

**性能（100 帧真实素材，交替 A/B）**：

| 配置 | 无流水线 | 有流水线 |
|---|---|---|
| 1 线程 row-mt | 7.0s | 7.3s |
| 4 线程 row-mt+par | **5.15s** | 5.9s |
| 8 线程 row-mt+par | **5.38s** | 7 线程 + 流水线 6.0s |

⇒ **所有配置都更慢**。原因：滤镜阶段已经用满可用核心（CDEF/去块/LR 内部
就带 worker 池），主线程再跑语法 N+1 只是抢核；单瓦片下没有额外空闲并行度。
**默认不启用**，`ME_FRAME_PIPE` 仅作为实验开关保留。

**保留的地基**（行为不变的独立提交）：`parse_tile`/`replay_tile` 拆分
（`7ae6b39`）、`RefFrameStore.pixels` 内部可变 + `BlockCtx.ref_pixels`
快照（`d4a8e5b`）。以后若在多瓦片/多帧独立流上重启帧级调度，可直接复用。

## 三十七、第五十三刀：CDEF 按 64×64 组批处理 + 四行内核 —— −4.8%（2026-10-23）

### 起点

先用**隔离开关**量真实占比（CPU 时间 / 100 帧 / 6 轮轮内换序）：

| 开关 | 配对中位 | 同向 |
|---|---|---|
| `AV1_NO_CDEF=1` | **−23.9%**（61.3 → 46.2 ms）| 6/6 |
| `AV1_NO_DEBLOCK=1` | −9.7%（61.3 → 55.5 ms）| 6/6 |

⇒ CDEF 是当前最大的单项（≈ 15 ms/帧）。函数级采样里
`cdef_pair16` 5.0%、`cdef_quad8` 3.7%、`dir_accum_avx2` 4.8%、
`cdef_filter` 4.3%，而 `apply` 的驱动闭包本身也有 5% 量级 ——
**驱动和内核几乎一样贵**（每块 3 次滤波调用、每次都要重算
`interior`/`coded_w`/`x_border`/候选偏移表）。

### 改法

1. **`apply` 按 64×64 组走**：组级只取一次 `cdef_idx` 与强度；
   `cdef_idx < 0` 时整组（64 块）直接跳过 —— 它本来就是**组级**参数。
2. **组内 8×8 块**：内点（块 + 2 样本边距都在帧内）直接进
   `cdef_group_fast`，不做边界判定、不补算边界条带；真正的帧边块
   （每帧约 1%）回落原来的 `cdef_block`。
3. **帧底那半行也按组跑**：`r + 64 > mi_rows` 的组不再整行退化成逐块，
   只跑组内仍在帧里的块行。
4. **零强度 = 恒等**：两个平面强度都为 0 时 `acc = 0`、`Clip3(min,max,cur) = cur`，
   连方向都不算（本素材里不常见，但语义上是白拿的）。
5. 新内核 **`cdef_pair16x4`**：一次四行（两个 YMM、两条独立依赖链），
   微基准 `bench_cdef_kernels` 里两行 286 ns → 四行 263 ns（**1.09x**）。
   像素式与 `cdef_pair16` 逐条相同，单测 `cdef_pair16x4_matches_pair16_bit_for_bit`
   覆盖 8 方向 × {8×8, 4×4, 8×4} × 3 组强度逐字节对拍。
6. 临时同二进制开关 **`ME_NO_CDEF_GROUP=1`** 退回逐块老路（两条路逐字节相同），
   用来做 A/B；保留为诊断开关。

### ⚠️ 踩坑：逐字节一致 ≠ 没有白干

第一版组循环**没有把行迭代器推过组内那 8 个块行** ⇒ 每个块被组路径滤一遍、
又被逐行路径再滤一遍（最多 8 次）。**24 帧输出仍然逐字节相同** ——
CDEF 读 `CurrFrame`、写克隆，重复滤波是幂等的。只有计时暴露了 **+15.3%（0/10 轮）**。
教训：`md5` 只能证明"没算错"，证明不了"没多算"；批处理类改动必须**单独核工作量**
（这里数一下每个块被访问的次数即可）。

### 实测

同二进制 A/B（100 帧、10 轮轮内换序、配对中位）：

| 路径 | CPU 中位 | ms/帧 |
|---|---|---|
| 逐 8×8 块老路（`ME_NO_CDEF_GROUP=1`）| 6.850 s | 68.50 |
| 组批处理 + 四行内核 | 6.488 s | 64.88 |

⇒ **−4.8%，8/8 轮更快**。关掉 CDEF 仍能省 23.9% ⇒ 下一刀仍在 CDEF 内部。

### 验收（全绿）

- 24 帧、60 帧与 cut51 **逐字节相同**；
- `ME_NO_SIMD=1`、`ME_ROW_MT=1 + ME_ROW_MT_PAR=1`（4 线程）**逐字节相同**；
- 真素材 60 帧三平面与 dav1d **逐像素一致**（`refcmp.py --grain=0`）；
- `cargo test --workspace --release`：**0 failed**（含新增的 x4 对拍）；
- **PGO 重训**（`tools/pgo.sh`，1500 帧）：旧 PGO vs 新 PGO（100 帧、6 轮轮内换序）
  **61.80 → 59.34 ms/帧（−4.2%，6/6 轮）**；同一窗口 10 轮那次是 −5.2%。

### 下一步（CDEF 内部新排序）

| 项 | 估计 | 说明 |
|---|---|---|
| `dir_accum` + `cdef_direction` | ≈ 4 ms | 512 B 栈上 partial、**每行 6 次重叠 RMW**（store-forwarding 冲突）；要改成寄存器累加 + i16 partial + 权重向量 |
| `cdef_quad8`（色度 4×4）| ≈ 2.3 ms | 每候选 4 次 8B 载入 + 3 次 insert；可试 U/V 合批或相邻块同方向合并 |
| `cdef_pair16x4`（亮度）| ≈ 3 ms | 四行版已是两条链；再往上要 8 行（寄存器不够） |
| 边界块 | ≈ 1 ms | 已从 5% 的块降到 ~1% |

## 三十八、第五十四刀：CDEF 边界块"候选掩码" + 方向搜索寄存器化 —— 约 −4%（2026-10-23）

### 1. 边界块的标量路径（`cdef_filter` 采样 **4.6% → 2.1%**）

上一刀把内点块都改成组批处理之后，采样里 `cdef_filter` 仍有 4.6% —— 它只剩下
**帧边块**（左右各一列、上下各一行，约 2.3% 的块；每块 3 个平面）。慢的原因是
每个像素的 12 个候选都走 `cdef_get_at`：坐标换算 + 两次移位 + 4 次越界比较 +
`Plane::get` 的 `u8/u16` 分派，**每个候选约 10 条指令，纯粹是取址开销**。

改法（语义不变，只是把"每像素算一遍"改成"每块算一遍"）：

* 12 个候选的 `(dx, dy)` 与线性偏移 `dy·stride + dx` 每块算一次；
* **候选有效性 = 列掩码 & 行掩码**：规范只要求候选落在帧的 4×4 网格内
  （`0 ≤ (x<<sub_x)>>2 < MiCols`，等价于 `0 ≤ x < (MiCols<<2)>>sub_x`），
  所以对每一列/每一行可以把 12 个候选压成 12 位掩码（每块各算 8 次）；
* 内点块两个掩码全 1，走无分支的标量快路径；SIMD 的右/下边界条带复用同一段代码。

这条路径与 `cdef_group_fast` 的 `interior` 判据逐条等价；24/60 帧逐字节、
`ME_NO_SIMD`、row-mt 并行、dav1d 60 帧逐像素全部一致。

### 2. 方向搜索（`dir_accum`+`cdef_direction` 采样 **6.6% → 2.8%**）

旧实现把 8 个方向的 partial 放在 512 B 栈数组里，每行对 6 个方向做
"读 32 B → 加 → 写 32 B"；**相邻两行写的窗口重叠 7 项**，store-to-load
forwarding 每次失败（~10 周期），一行 6 次。1080p 有 3.24 万个 8×8 块。

新实现（`simd::cdef_direction_avx2`）：

* 每个方向只开 2 个 XMM 累加器（`lo` = 位置 0..7、`hi` = 8..15）；
  一行的 8 个 i16 用两条 `pshufb`（按 `base` 查常量掩码）搬到 `[base, base+8)`
  再相加 —— **纯寄存器、没有内存 RMW**；
* 代价用 `pmaddwd` 求**对称位置**的平方和（i16→i32），`vpmulld` 乘 Div 表、
  横加。对称位置权重相同（`Div[|k-7|+1]`）⇒ 与标量逐位相同；
* 方向 1/3 的 `y = s[2t]+s[2t+1]` 用 `phaddw`，4 用 `pshufb` 反序。
* 全部中间量 i16（`|partial| ≤ 8·128`）；新单测
  `cdef_direction_avx2_matches_scalar_bit_for_bit` 覆盖 8/10/12bit ×
  随机/斜条纹/棋盘 × 5 个块位置，与标量路径的 `(yDir, var)` **逐位对拍**。

**实测**：

| 对照 | 结果 |
|---|---|
| 同二进制：方向 SIMD vs 纯标量（`ME_NO_CDEF_DIR_SIMD=1`） | **−10.7%（10/10 轮）** |
| 采样：方向搜索占忙时 | 6.6% → **2.8%** |
| 采样：`cdef_filter` 占忙时 | 4.6% → **2.1%** |

> ⚠️ 同二进制那条是"新 SIMD vs **纯标量**"（旧的内存 RMW 版已删），
> 所以 −10.7% 是"相对最慢的标量"而不是"相对上一版"；相对上一版的净收益
> 以采样与最终 PGO A/B 为准。

**最终 PGO A/B**（旧 PGO = 只有组批处理那版；100 帧、10 轮轮内换序）：

| 路径 | CPU 中位 | ms/帧 |
|---|---|---|
| 上一版 PGO（组批处理） | 6.007 s | 60.07 |
| 本刀 PGO（边界掩码 + 方向寄存器化） | **5.661 s** | **56.61** |

⇒ **−5.7%，10/10 轮更快**。单线程水位从 60.1 ms/帧 降到 **56.6 ms/帧**。

### 3. 否定项：整数 MC 的边界块

想让右/下边界块也走 `mc_shift_rows`（内核本来就逐行钳位，只需把跨过 `last_x`
的列补算成标量）。实测：非 PGO 版相对旧 PGO **+4.0%（0/8 轮）**，恰好等于 PGO
本身的差值 ⇒ **这条素材上边界块占比太低，没有收益，已回退**。

⚠️ 过程中踩到一个值得记的坑：**2 宽色度块（4×4 亮度块的 4:2:0 色度）内核
一列都不写、返回 0**；补算起点必须取「内核实际写出的列数」与「右边界起点」
的 **min**，漏掉前者会让这些块保留旧预测值（60 帧对拍在第 34 帧的 U 平面抓到
204 个 ±1 的差异）。

### 验收

24/60 帧与 cut51 逐字节相同；`ME_NO_SIMD=1`、`ME_ROW_MT=1+PAR=1` 逐字节；
dav1d 60 帧三平面逐像素一致；`cargo test --workspace --release`：
**1028 passed / 0 failed**（含两条新增对拍）。

## 三十九、第五十五刀：残差写回就地化 + NV12 UV 交错 SIMD（2026-10-23）

### 1. 残差写回不再先拷预测（`coeff.rs` / `transform.rs`）

`add_residual` 原来先把预测值从帧平面整块拷进一个 `pred` 缓冲，再让
`inverse_transform_add` 读 `pred` + 残差写回平面。预测本来就在平面里 ⇒
新增 [`transform::inverse_transform_add_in_place`]：加法循环**直接读平面**
（翻转时读写的也是同一个 `(yy, xx)`，块内不串），1080p 一帧几万个变换块
省掉一次"平面→pred"的读+写与一次回读。

同二进制 A/B（`ME_NO_TX_INPLACE=1` 退回旧路，100 帧、10 轮轮内换序）：
**−0.9%（7/10 轮）**。诊断（`AV1_TRACE_TX1` 要打印 pred）与 u8 存储仍走旧路径。

### 2. NV12 的 4:2:0 UV 交错向量化（`av1.rs`）

原来每对 U/V 样本两次单字节写（1080p 每帧约 52 万对）。现在每 8 对一次
`packusw` + `punpcklbw`：移位（10/12bit）后值 ≤ 255，饱和收窄与标量逐位相同。
24 帧输出与改前**逐字节相同**。

### 验收

24 帧逐字节；`ME_NO_TX_INPLACE=1` 旧路逐字节；row-mt 并行逐字节；
60 帧与 dav1d 三平面逐像素一致。本刀与第五十六刀一起做的 PGO 终测见下。

## 四十、第五十六刀：整像素单参考预测**直写帧平面** —— −1.2%（2026-10-23）

### 推导（为什么可以省掉 i32 缓冲）

非复合帧间的两趟取整量满足 `InterRound0 + InterRound1 = 14`（8/10bit
`3+11`、12bit `5+9`）⇒ `post_round = 2·FILTER_BITS − (…) = 0`；整像素相位下
`mc_shift_rows` 输出的 i32 预测值**就是源样本本身**，而 `store_row_clamped`
只做 `clamp(0, max)`（平面里的样本本来就在值域内）。于是这条路径可以完全绕开
"MC → i32 pred 缓冲 → 逐行 store"：

```
predict_inter（单参考、整像素、无 interintra、无 OBMC）
  → store_integer_pred：按行把参考样本拷贝进帧平面
```

行/列钳位与标量路径逐条相同（`clip(y0+i, last_y)` / `clip(x0+c, last_x)`）；
内点整行 `copy_from_slice`，边界逐像素钳位。省掉的是每像素 **4 B 写 + 4 B 读**
的 i32 缓冲，以及 `store_row_clamped` 的逐行调用。

### 实测

同二进制 A/B（`ME_NO_DIRECT_PRED=1` 退回旧路，100 帧、10 轮轮内换序）：
**−1.2%（8/10 轮）**。采样里 `mc_shift_rows` 这一步此前占 7.4%。

### 验收

24/60 帧与改前逐字节相同；dav1d 60 帧三平面逐像素一致；`ME_NO_DIRECT_PRED`
两条路输出相同。分数相位没有走这条捷径（需要给纵向 MC 内核加 u16 输出），
留作下一步。

### 两刀的 PGO 终测（第五十五 + 五十六刀一起）

| 路径 | CPU 中位 | ms/帧 |
|---|---|---|
| 上一版 PGO（只有 CDEF 两刀） | 5.880 s | 58.80 |
| 本版 PGO（残差就地 + NV12 交错 + 直写平面） | **5.758 s** | **57.58** |

⇒ **−2.2%，10/10 轮更快**。与两条同二进制 A/B（−0.9% + −1.2%）自洽；
NV12 那条没有单独量到可分辨的收益，但它是纯"更少指令"的等价改写，保留。
全量测试 **1028 passed / 0 failed**。

## 四十一、第五十七刀：直写路径的整行拷贝内联 —— −0.9%（2026-10-23）

第五十六刀的直写平面用 `copy_from_slice` 拷每一行。行长是块宽（4~64 个 u16），
而 `copy_from_slice` 对**运行时长度**编译成 `memmove` 调用 —— 每帧几十万行、
每行 8~128 字节，函数调用与分派开销比拷贝本身还大（采样里 `_platform_memmove`
一度升到 7%）。

改成按宽度内联：2/4/8/16/32 个 u16 分别用一次 `u32`/`u64`/`u128`/两段/四段
未对齐读写，其余宽度（64/128）仍走 `copy_from_slice`（256B 已摊薄）。
同二进制 A/B（`ME_NO_ROWCOPY=1`）：**−0.9%（7/10 轮）**。

## 四十二、第五十八刀：零 MV 的运动场投影快路径 —— −1.4%（2026-10-23）

`project_frame` 每格要为 7 个目标帧各做一次 `project_k`（2 次 i64 乘 + 取整 +
钳位）。但 `project_k(0, k) = 0` 对**任何** `k` 都成立 ⇒ 零 MV 的格子 7 个投影
全是 `(0,0)`、落点也相同。静止内容里这种格子很多。

快路径：`mv == (0,0)` 时只算一次 `block_position`，然后把这格 7 个槽位直接写成
`[0, 0]`，省掉 16 次 i64 乘法与 8 次取整/钳位。同二进制 A/B（`ME_NO_MV0=1`）：
**−1.4%（8/10 轮）**。24/60 帧输出逐字节、dav1d 60 帧逐像素一致。

### 五刀合计的 PGO 终测（五十五 ~ 五十八刀）

| 路径 | CPU 中位 | ms/帧 |
|---|---|---|
| 上一版 PGO（只有 CDEF 两刀） | 5.640 s | 56.40 |
| 本版 PGO（残差就地 + NV12 + 直写 + 行拷贝 + 零MV） | **5.390 s** | **53.90** |

⇒ **−5.2%，9/10 轮更快**。全量测试 **1028 passed / 0 failed**；
24/60 帧逐字节、`ME_NO_SIMD`/row-mt 逐字节、dav1d 60 帧逐像素一致。

## 四十三、第五十九刀：CDEF 左/上边界块的"位移 SIMD" —— −0.6%（2026-10-23）

帧的左/上边界没有负向 padding，`cdef_filter` 原来对这类块（约 2.3% 的块）**整块
走标量**（每像素 12 个候选逐一判定）。现在把内核起点右/下移 2 个样本再跑 SIMD
（`x_eff ≥ 2` ⇒ 候选 `x_eff-2 ≥ 0` 可寻址），多写的 2 列/行由已有的**掩码标量**
补算：

* 位移后内核写出的矩形是 `[x_eff, x_eff+w) × [y_eff, y_eff+h)`；落在原块**外面**
  的 2 列/行先**从 `CurrFrame` 还原** —— 相邻块可能是 skip（它自己不滤波、
  也就不会覆盖回来）。还原后与"没位移"逐字节相同。
* 左/上条带再用 `filter_pixels(..., masked=true)` 重算（与右/下条带重叠也无妨：
  同一块从 `CurrFrame` 读、写同样的值）。

同二进制 A/B（`ME_NO_CDEF_SHIFT=1`）：**−0.6%（6/10 轮）**；PGO 版对上一版 PGO：
**52.79 → 52.17 ms/帧（−1.0%，8/10 轮）**。
⚠️ 第一版忘了"还原块外写入"，60 帧对拍在第 16 帧抓到差异（skip 邻块的像素被
邻块的滤波结果覆盖）；补上还原后 24/60 帧逐字节、dav1d 60 帧逐像素一致。
全量测试 **1028 passed / 0 failed**。

## 四十四、本轮两个否定实验：OBMC 直写（PGO 下无收益）+ CDEF pri0 专化（2026-10-23）

### 1. 整像素直写路径覆盖 OBMC —— 非 PGO 版 −2.2%，PGO 下 **−0.1% ⇒ 已回退**

`obmc_blend` 读的正是平面里的当前预测，所以直写平面与旧的"i32 缓冲 → store"
语义等价，理论上去掉 `motion_mode == OBMC` 的排除是安全的、能多覆盖一批块。
实测出现了**两个不同的答案**：

| 构建 | 同二进制 A/B（`ME_NO_DIRECT_PRED=1`） |
|---|---|
| 非 PGO | **−2.2%（9/10 轮）** |
| PGO（发货配置） | **−0.1%（5/10 轮）** |

⇒ **以 PGO 为准：没有收益，一行条件已回退**（连同这一轮的 PGO 产物一起丢弃，
交付仍是上一版）。教训：<5% 的结论必须在**发货构建（PGO）**上复核 —— 非 PGO
的 inlining/layout 会放大或伪造差异。

### 2. CDEF `pri_str == 0` 的专门化内核 —— −0.0% ⇒ 已回退

本素材 78% 的亮度块 `pri_str == 0`（主抽头阈值 0 ⇒ 约束恒 0、只剩 min/max）。
用 const-generic 写了 `cdef_pair16x4_impl<const PRI0: bool>` 跳过 4 个主候选的
约束式：同二进制 A/B **−0.0%（5/10 轮）**，没有可分辨收益 ⇒ 已回退。
与去块 8 线的结论一致：**这些内核是载入/存回受限，不是约束算术受限**。

### 当前交付

仍是 `4dc5ba89…`（52.17 ms/帧，CDEF 边界位移那一版）。源码已回到该版状态。

### 3. 去块"帧级常量强度"提升 —— −0.2%（6/10 轮）⇒ 已回退

`filter_strength` 每条边都重算（含"分段是否有 LF 特征"的 8 段扫描）。把这套判定
提到帧级（每 (plane,pass) 一次），并让 lvl==0 的整遍 pass 直接跳过、把 `prev_tx_sz`
的读取挪到确认要滤之后。同二进制 A/B（`ME_NO_DBL_FIXED=1`）：**−0.2%（6/10 轮）**，
落在噪声里 ⇒ 已回退（签名改动不值得为 0.1~0.2% 保留）。

## 四十五、第六十一刀：LR 的 `RESTORE_NONE` 平面零拷贝 —— 计数干净，计时受机器波动干扰（2026-10-23）

`lr::apply` 对 `frame_restoration_type[plane] == RESTORE_NONE` 的平面原来整平面
`copy_from_slice`（1080p 色度两平面约 **2.5 MB/帧**，而本素材绝大多数帧的色度都是
`RESTORE_NONE`）。两份源本来就是**按值**传进来的（`up_curr: CurrFrame`、
`up_cdef: Option<CurrFrame>`），所以可以直接把源平面**换**进输出帧：

```rust
let src_frame = up_cdef.as_mut().unwrap_or(&mut up_curr);
std::mem::swap(&mut out.planes[plane], &mut src_frame.planes[plane]);
```

换出来的新平面跟着源一起回池；`RESTORE_NONE` 的平面不会被任何滤波路径再读，
所以逐像素不变。同二进制开关 `ME_NO_LR_SWAP=1` 退回整块拷贝。

**验收**：24/60 帧逐字节、dav1d 60 帧逐像素一致；`cargo test --workspace --release`
**1028 passed / 0 failed**。

⚠️ **计时一度不可用**：第一次交替 A/B 的 CPU 时间从 2.8s 飘到 6.4s
（机器负载/频率波动），配对中位 1.0091（4/10 轮）。PGO 重建后换窗口复测：
**59.23 → 58.89 ms/帧（−0.2%，5/6 轮）** —— 幅度小但与"只减少一次 2.5 MB 拷贝"
的预期一致，保留。

## 四十六、两个否定实验：CDEF 色度 U/V 合并（PGO 下无收益）+ 4 宽变换 4 路 i32（证据否掉）（2026-10-23）

### 1. CDEF 色度 U/V 合并内核 —— 非 PGO −1.3%，PGO **−0.1% ⇒ 已回退**

U/V 的 4×4 块方向/抽头/阈值完全相同，写了 `cdef_quad8_uv` 一次滤完两个平面
（省一次内核序言、12 次抽头广播与一次调用），单测与两次 `cdef_quad8` 逐字节相同。

| 构建 | 同二进制 A/B（`ME_NO_CDEF_UV=1`） |
|---|---|
| 非 PGO | **−1.3%（5/8 轮）** |
| PGO（发货配置） | **−0.1%（3/6 轮）** |

⇒ 与第四十四节的 OBMC 直写**完全同一个模式**：非 PGO 上"省一次调用/序言"看起来
有 1 个点，PGO 一内联就消失了。**结论：这类"省调用开销"的改动在 PGO 下不值得做**，
以后直接跳过。已回退。

### 2. 4 宽变换走 4 路 i32 —— 用计数否掉，不写

给 `inverse_transform_2d_into` 加计数器实测：`w < 8 || h < 8` 的变换块占
**14~21% 的调用数**，但只占 **0.2~0.4% 的像素**（它们都是 4×4/4×8/8×4）。
`vt_forward_1d` 的成本按像素算 ⇒ 把窄块从 i64 换到 4 路 i32 最多值 0.1%，
不值得写一套 128 位内核。证据留在这里，免得以后重做。

### 当前交付

仍是 `be26bc11…`（LR 零拷贝那一版）。
## 四十七、否定实验：NV12 亮度收窄显式 AVX2（PGO 下持平）+ 行级自采样（2026-10-23）

### 1. 行级自采样：没有藏起来的大头

带 `-C debuginfo=1` 重编后 `sample` 的调用图会带 `file.rs:line`；按"没有更深子节点的
行"聚合出**自采样**，得到的热点排序是：`vt_forward_1d` 调用行 202、LR `restore_block`
92、`mc_shift_rows` 79、`compute_prediction` 60、残差加法 45、NV12 亮度 45、
`grid.stamp` 37 …… —— 全部是已知的内核/驱动，**没有"藏起来的大头"**。
这条方法值得留着（`RUSTFLAGS="-C debuginfo=1" cargo build --release` + `sample`）。

### 2. NV12 亮度收窄 —— PGO 下持平，已回退

按上面 45 个自采样点的提示，把 u16→u8 亮度行写成显式 AVX2（32 像素：两条载入 +
`packusw` + lane 置换 + 一条存）。结果：

| 构建 | 对上一版 PGO 的配对中位 |
|---|---|
| 第一版（**忘了 `#[target_feature]`**） | **+4.6%（1/6）** |
| 修好后（有 `#[target_feature]`） | **+4.0%（1/6）** ≈ PGO 差 |

第一版的教训又踩了一次：intrinsics 写在普通函数里 ⇒ 每条都变成真实函数调用
（`otool` 里能看到 26 处 `callq ...avx2...`）。修好后虽然指令正确了，但这一行
本来就被 LLVM 自动向量化得差不多，**PGO 下没有可分辨收益** ⇒ 已回退。

### 当前交付

仍是 `be26bc11…`（LR 零拷贝那一版，约 52 ms/帧）。
## 四十八、下一步（结构性）：u8 原生重建的分期计划

五轮微优化（pri0、OBMC 直写、去块强度提升、U/V 合并、NV12 收窄）在 **PGO** 下
全部落在 ±0.2% 内，且行级自采样确认没有隐藏热点 —— u16 路线的增量空间已经用尽。
要把 52 ms 往 33 ms 推，只能走 **u8 原生**（dav1d 同款）。之前的混合方案失败在
"重建 u16 + 滤镜 u8 + 来回转换"；这次必须**连重建一起切**，默认路径保持不变，
全部在 `ME_PLANE_U8` 后面分期做：

1. **重建写 u8**：`inverse_transform_add` 的 8bit 分支直接写 `u8`（clamp1 到
   0..255），`intra` 的预测落点改 `u8`；10/12bit 仍走 u16。
   关口：`ME_PLANE_U8=1` 与默认路径**逐字节相同**、dav1d 逐像素、全量测试。
2. **帧间**：MC 读 `u8` 并加宽（⚠️ 第十六节测过"u8 源 + 加宽"比 u16 慢 11%，
   要在这个阶段复测；若确实更慢，MC 可以保持 u16 读，其余照切）。
3. **滤镜**：去块（u8 窗口，16 线/YMM）、CDEF（u8 通道）、LR（u8 读写）、
   NV12（8bit 直接整行拷贝）。
4. **结算**：两路同二进制 A/B；只有在 PGO 上 ≥5% 才切默认、删旧路径，
   否则保留开关、如实记录——**不为了"切了 u8"而切**。

参考实现只学算法/结构（dav1d / libaom），代码全部自己写、不抄不译。

### 3. `stamp` 跳过"等于帧初值"的写入 —— **会破坏熵解码**，立即回退

`BlockGrid::stamp` 每块写 13 个数组。想省掉"值等于 `reset_all` 初值"的那几条
（单参考帧间块能省 7~8 条）。但 `comp_group_idxs` / `compound_idxs` 等**不是纯
簿记**：它们会被后续块当作 CDF 上下文读（`block.rs` 的 `comp_group_idx` /
`compound_idx` 语法元素），而单参考块写进去的值与帧初值不同。
实测第 4 帧就在 `coeff_base` 上多读 6 位、瓦片越界 ⇒ **已回退**。
教训：**BlockGrid 的字段是语法状态，不是渲染元数据**，"值等于初值"不代表可省。

## 四十九、第六十四刀：NV12 输出缓冲池化 —— −2.1%（7/8 轮，2026-10-23）

`nv12_frame` 每帧 `vec![0u8; w*h*3/2]`（1080p = **3.1 MB**）：分配、清零，用完在
`VideoFrame` 析构时释放。而这条路径**每个字节都会被写满**（亮度 `w*h` + 色度
`w*h/2`，所有 4:2:0/4:2:2/4:4:4/单色分支都覆盖），清零是纯白写。

改法：
* `VideoFrame` 实现 `Drop`：把 `nv12` `mem::take` 出来还给已有的 u8 缓冲池
  （`video::pool::give_u8`，按长度复用，上限 96 MB）；
* `nv12_frame` 先 `take_u8(n)`，未命中再新分配。

同二进制 A/B（`ME_NO_NV12_POOL=1` 每帧新分配）：**−2.1%（7/8 轮）**，55.46 → 54.46 ms/帧；
PGO 版对上一版 PGO：**63.78 → 62.33 ms/帧（−1.6%，6/8 轮）**。

⚠️ 注意这笔收益比"3 MB 带宽 ≈ 0.3 ms"大得多（实测 ~1.2 ms）：省掉的是
**每帧一次 3 MB 的 mmap + 写入时缺页 + 释放**，不是单纯的内存带宽。
又一次印证：**估算不能代替实测**。

验收：24/60 帧逐字节、dav1d 60 帧逐像素一致；`cargo test --workspace --release`
**1028 passed / 0 failed**。

## 五十、第六十五刀：`CurrFrame::decoded` 网格池化 —— 同二进制 −1.0%（6/8，2026-10-23）

`CurrFrame::new` 和 `CurrFrame::clone` 每个平面都开一块 4×4 "已重建"标记网格
（1080p：亮度 270×480 + 两个色度 135×240 ≈ 0.19 MB/帧），而每帧要建三份
（重建帧 / CDEF 克隆 / LR 输出）⇒ 约 **0.6 MB/帧**的分配 + 清零/拷贝。

改法沿用上一刀的池：`new` 从 `video::pool` 取同长度缓冲后 `fill(0)`（标记必须
清零），手写 `Clone` 取池后 `copy_from_slice`，`Drop` 把三块网格还回池。
同二进制 A/B（`ME_NO_DECODED_POOL=1`）：**−1.0%（6/8 轮）**，57.66 → 56.98 ms/帧；
PGO 版对上一版：**54.89 → 54.29 ms/帧（−0.4%，5/8 轮）**。
24/60 帧逐字节、dav1d 60 帧逐像素一致；全量测试 **1028 passed / 0 failed**。

## 五十一、否定实验：跳过帧外 padding 填充（0xAA 毒化验证了语义，但性能更差）（2026-10-23）

每帧 `Plane::new` 会填三份约 0.9 MB 的帧外 padding（中灰）。代码注释声称"帧外样本
不进输出、不进参考帧取样"。先用 `ME_PAD_POISON=1` 把 padding 填成 **0xAA** 验证：
24/60 帧与基线**逐字节相同**、dav1d 60 帧逐像素一致 ⇒ 语义上确实可以任意填。

再用 `ME_NO_PAD_FILL=1` 完全不填（池里是什么就是什么）：
**同二进制 A/B +0.9%（3/8 轮）——更慢**。

原因：填充本身把这些"右列/底行"写热了；CDEF 在边界块上本来就会读到这些缓存行，
不填则要先付缺页/冷读。**"少写一点"不等于更快**，这条和第四十九节的 NV12 池化
（省 mmap/缺页才赚）正好是同一枚硬币的两面。默认保持填充；两个诊断开关保留。

## 五十二、两个否定实验：瓦片 scratch 池化、运动场"参考方向表"（2026-10-23）

### 1. `PredScratch`/`TxScratch` 池化 —— −0.5%（4/8 轮），噪声内 ⇒ 回退

瓦片级复用缓冲每帧从空增长（约 200 KB）。用线程局部池跨帧复用容量、
`replay_tile` 收尾显式回收（在 `BlockCtx` 上加 `Drop` 会和它持有的
`&mut CurrFrame`/`&SequenceHeader` 借用冲突，所以用显式 `recycle()`）。
同二进制 A/B **−0.5%（4/8）**，与"只有 MB 级 mmap/缺页才划算"的判断一致 ⇒ 回退。

### 2. `motion_field_storage` 的"参考方向可用"表 —— **+1.9%（2/8）** ⇒ 回退

每帧 13 万 mi 格 × 2 列表各调一次 `get_relative_dist`，看起来适合提成 8 项表。
实测反而更慢：原判定的分支高度可预测、`get_relative_dist` 也很便宜，
换成每格一次 `usable[r]` 查表（多一次索引与依赖）反而破坏了流水线。
**又一次"想当然的更少工作 ≠ 更快"。**

### 当前交付

`b162e7d1…`（NV12 池 + decoded 网格池那一版），单线程约 54 ms/帧。

## 五十三、否定实验：`BlockGrid.mvs` 跳过逐帧清零 —— 100 帧通过，但**违反文档约定，已回退**（2026-10-23）

`reset_all` 每帧把 `mvs`（2 MB）整片填零。尝试改成 `ensure_len`（只首帧/尺寸变化
时清零）：同二进制 **−1.2%（8/8）**、100 帧逐字节 + dav1d 逐像素 + 1028 测试全过。

但复核源码文档时发现 `BlockGrid` 顶部**明确把 `mvs` 列进"必须保留初值"**：

> `mvs` / `comp_*` / `wedge_*` / `mask_types` / `interp_filters` /
> `interintra_modes` —— 只在 `is_inter` 时写；读它们的地方虽然都有 `ref_frames` 门，
> 但"门与写同源"这条依赖链**太隐晦，留着初值**。

这条约定是作者刻意做的保守决策（同类尝试在第五十四节第 ~70 帧就翻车）。
100 帧的单流通过不足以推翻它 ⇒ **已回退**，交付回到 `b162e7d1…`。
教训：**代码里写明"必须保留"的初值，不要因为单流测试通过就省**。

## 五十四、否定实验：`BlockGrid` 的 compound/wedge/interp/interintra 整片初值 —— 第 ~70 帧才暴露，已回退（2026-10-23）

`reset_all` 里还有 8 个 `fill_set`（`comp_group_idxs`/`compound_idxs`/`compound_types`/
`wedge_*`/`mask_types`/`interp_filters`/`interintra_modes`），合计约 **1.1 MB/帧**。
按"读者都先查 ref_frames / 块类型"的分析改成 `ensure_len`，**24/60 帧全过**；
但 100 帧的 A/B 在第 **~70 帧**报出
`COMPOUND_DIFFWTD 掩码尺寸对不上：1024 != 4×4` —— 说明仍有**未枚举的读者**
会读未写格子（`dw_mask` 的长度是上一块留下的）。已整体回退。

⚠️ **协议更新**：以后所有"跳写/复用"类改动，验证窗口用 **≥100 帧**（60 帧不够）。
`mvs` 那条之所以能省，是因为 inter 块写在前、intra 格无读者（且 100 帧复核通过）。

## 五十五、收尾审计（2026-10-23）：目标未达成，交付状态与证据

### 结论（如实）

**单线程 33 ms/帧的目标没有达到。** 当前发货产物（PGO `b162e7d1…`）在本机
sustained **约 54 ms/帧**（约 18.5 fps，随温度/负载浮动），差距 **~1.63x**。
u16 + 逐块/逐 4×4 的架构已经把增量优化的空间用尽：近十项微优化只有三项过了
PGO 同向关口，其余全部在 ±0.2% 或为负。

### 今天被采纳的刀（按 PGO / 同二进制证据）

| 刀 | 结果 |
|---|---|
| CDEF 64×64 组批处理 + 四行内核 | 同二进制 −4.8%（8/10）|
| CDEF 边界候选掩码 + 方向搜索寄存器化 | PGO 60.07 → 56.61（−5.7%，10/10）|
| 残差写回就地化 | 同二进制 −0.9%（7/10）|
| NV12 UV 交错 SIMD | 输出逐字节相同（未单独量到收益，保留）|
| 整像素单参考预测直写平面 | 同二进制 −1.2%（8/10）|
| 直写行拷贝内联 | 同二进制 −0.9%（7/10）|
| 零 MV 运动场投影快路径 | 同二进制 −1.4%（8/10）|
| LR `RESTORE_NONE` 零拷贝 | PGO −0.2%（5/6）|
| NV12 输出缓冲池化 | PGO −1.6%（6/8）|
| `CurrFrame::decoded` 网格池化 | PGO −0.4%（5/8）|

### 被否掉的（都留在文档里，避免重做）

CDEF pri0 专化、OBMC 直写（PGO 假阳性）、去块强度提升、CDEF U/V 合并、NV12 亮度
显式 AVX2、4 宽变换 4 路 i32、padding 跳过填充、瓦片 scratch 池化、运动场参考方向表、
`BlockGrid` 的 compound/wedge 等整片初值（**第 ~70 帧才暴露**，见五十四节）、
`BlockGrid.mvs` 跳过清零（100 帧通过，但违反"必须保留初值"的文档约定，见五十三节）。

### 复现与验收（本次审计实际跑过的）

* 素材：`/tmp/a.mp4`（1920×1080，见项目记忆里的恢复命令）；
* `ME_THREADS=1 … decode /tmp/a.mp4 --count=100 --out=…`：**100 帧全部落盘、无报错**；
* 24 帧与基线**逐字节相同**；60 帧 vs dav1d `refcmp.py --grain=0`：**逐像素一致**；
* `ME_NO_SIMD=1`、`ME_ROW_MT=1 + ME_ROW_MT_PAR=1`（4 线程）与默认路径**逐字节相同**；
* `cargo test --workspace --release`：**1028 passed / 0 failed**；
* 发货产物 `extra/darwin/x64/engine/zuvrust` md5 `b162e7d10412696fcc76abff807ed18a`。

### 要继续往 33 ms 走的唯一路线

见第四十八节：**u8 原生重建**（重建写 u8 → 帧间 → 滤镜链 → 结算），全在
`ME_PLANE_U8` 之后分期做；每期以"≥100 帧逐字节 + dav1d 逐像素 + PGO A/B ≥5%"
为关口，达不到就保留开关、不切默认。这是多轮结构改造，不是单刀能完成的。

## 五十六、否定实验：CDEF 按 64×64 组补拷（省整帧克隆）—— 第 2 帧就暴露，已回退（2026-10-24）

`cdef::apply` 每帧先 `frame.clone()`（1080p 约 7 MB memmove + 一次池分配），再只写
被滤波的组。真素材里 `cdef_idx == -1` 的组约占 **一半**（`ME_CDEF_STATS=1` 实测
32400 个 8×8 块里 1600~26600 个跳过，逐帧波动大），看起来"不滤波的组不必拷"是一笔
白捡的钱。

改法：先扫一遍组级表；全跳过就 `return Ok(None)`（零拷贝），否则给 `CdefFrame`
开一份**不填内容**的骨架（`Plane::empty` + `CurrFrame::empty_like`），跳过的组按
**整组 64×64** 补拷（不是 2026-10-06 失败的 8×8 块级拷贝）。

结果：第 2 帧就与 `ME_CDEF_FULL_CLONE=1` 不一致（109 个组、每组约 3500/4096 像素）。
原因不在拷贝粒度，而在 CDEF **内部还有两条块级"不滤波"出口**、同样依赖整份克隆
留下的原像素：

1. `cdef_group_fast` 的"四个 4×4 全 skip ⇒ 跳过"；
2. 参数组强弱都为 0（`!need_y && !need_uv`）⇒ 恒等变换，也直接 `continue`。

要让按组补拷成立，就得回到**块级**补拷 —— 正是 2026-10-06 那次 +14.2% 的形状。
**结论：这条路走到头了**，整帧克隆保留；`ME_CDEF_STATS` 诊断开关保留（可看跳过率）。

## 五十七、第六十六/六十七刀：复合 AVERAGE 直写（整像素 + 融合纵向平均）—— 同二进制 −2.9%（8/8，2026-10-24）

### 采样先改方向：这份素材 97% 的帧间像素是**复合**

`ME_MC_STATS=1` 按类别累计亮度预测像素（1080p 单帧）：

| 类别 | 像素/帧 |
|---|---|
| 复合 AVERAGE | **2.3M ~ 6.0M** |
| 复合 DIFFWTD（`other`）| 0.65M ~ 0.95M |
| 复合 DISTANCE | 0.28M ~ 0.33M |
| 单参考整像素 | 0.12M ~ 0.62M |
| 单参考子像素 | 3.7 万 ~ 4.0 万 |
| OBMC / interintra / warp | < 5 万 |

所以之前那些"单参考直写"的刀（整像素、子像素）在这份素材上只值 1% 出头；
**复合 AVERAGE** 才是主战场。

### 第一刀：双整像素 ⇒ 规范式退化成 `pavgw`

`COMPOUND_AVERAGE` 的混合是 `Clip1(Round2(p0 + p1, 1 + InterPostRound))`，
而整像素位置的两路预测值都是 `16·src`（`post_round = 14 − (3 + 7) = 4`）⇒

```text
Round2(16·s0 + 16·s1, 5) = (s0 + s1 + 1) >> 1        // 正是 `_mm256_avg_epu16`
```

于是两路 MC 与混合全不做：直接从两张参考平面按行 `avg` 到当前平面
（`simd::comp_avg_rows`）。两路参考窗口都判内点才走，边缘块回落通用路径。
同二进制 A/B（`ME_NO_DIRECT_COMP_AVG=1`）：**−1.1%（6/8 轮）**，覆盖约 2.9M 像素/帧。

### 第二刀：亚像素 ⇒ 两路横向 + **融合纵向/平均**

剩下的 AVERAGE 块是亚像素，成本是"两路各写一块 i32 暂存 → 再读两路做平均"
（约 16~18 字节/像素的往返）。新内核 `simd::mc_vert2_avg_u16` 把**纵向 8 抽头和
平均合成一次**：两路横向中间量（i16，本来就要）算完后，纵向在寄存器里混合、
钳位、`packus` 成 u16 直接写平面 —— 两路 d16 预测值**一次都不落内存**。

* 两路横向仍用现成的 `mc_horiz_rows_8bpc`（含行钳位、`interior_x` 判据）；
* 纵向内核的两路 `pmaddwd` 链 + `avg` 在同一个 16 通道循环里；
* 只开 8bit / 不缩放 / 非 warp / 非 trace；其余回落通用路径。

### 证据

* 同二进制 A/B（新 vs `ME_NO_DIRECT_COMP_AVG=1`，60 帧 × 8 轮交替、配对中位）：
  **0.9713（−2.9%），8/8 轮更快**；
* **100 帧逐字节相同**（`/tmp/v100a` vs `/tmp/v100b`，md5 `6e7c64bc…`）；
* 60 帧 vs dav1d `refcmp.py --grain=0`：**逐像素一致**；
* `cargo test --workspace --release`：**1029 passed / 0 failed**（36 个 suite，含新加的
  `mc_vert2_avg_u16_matches_vert_then_average`：融合内核 vs 两路纵向+标量平均
  逐位对拍，覆盖 wu = 4/5/8/16/24/36）；
* 发货产物 `extra/darwin/x64/engine/zuvrust` md5 `1dcdd3afb1286e8ee81f9444486b21e7`
  （PGO，构建在 `target/pgo/release/zuvrust`；`cargo test/build` 会覆盖
  `target/release/zuvrust`，回填别拷错）。

### 留给下一轮的（按像素量排序）

1. **DIFFWTD**（0.75M 像素/帧）：掩码由两路预测差生成 ⇒ 现在要落两路 i32 +
   两遍（掩码、混合）。把"两路纵向 + 掩码 + 混合"也融合成一次，预计再省
   ~0.8 ms（掩码的色度折算规则是难点）；
2. **DISTANCE**（0.3M 像素/帧）：同一套融合内核换成加权（`fwd/bck`、shift=8），
   量小；
3. 复合路径的 i16 暂存（剩下的通用块）—— 与上面的融合重叠，收益递减。

### PGO 复测：代码收益在"跨构建"比较里被噪声淹没（如实记录）

同一天晚些时候用 PGO 重建（`tools/pgo.sh /tmp/a.mp4`，1500 帧训练）再对拍：

| 对比 | 结果 |
|---|---|
| 新 PGO vs 它的非 PGO 构建（同源代码）| −4.1%（8/8）⇒ PGO 本身正常 |
| 新 PGO vs 发货的上一版 PGO（`b162e7d1…`）| **−0.5%（5/10）** —— 噪声内 |
| 新 PGO 的**回退路径** vs 上一版 PGO | +2.8%（0/8） |

也就是说：**代码层面确实少做了工作**（同二进制 A/B −2.9%、8/8），但两个
**独立 PGO 构建**之间的布局/内联差异能把 3% 量级的差异淹没掉；上一版 PGO 的
profile 恰好对这份素材更友好。用户时间与周期数（100 帧单跑）略偏新构建
（user 5.10 vs 5.24 s、cycles 75.0M vs 76.1M/帧），而**退休指令数几乎持平**
（198.8M vs 198.0M/帧，+0.4%）—— 说明融合内核省下的是内存往返，不是指令条数。

⚠️ **协议补充**：以后 ≥2% 的改动要出 PGO 版时，最好**同一份 profile** 分别构建
新旧两版（或至少把"回退路径"也打进同一个二进制用 env 切换），否则跨构建的
PGO 差异（±2~3%）会把结论搅浑。这次的 env 开关（`ME_NO_DIRECT_COMP_AVG`）
已经能在一个二进制里对拍代码收益，但 PGO 版的该对比只跑过 60 帧 × 8 轮的非 PGO
构建；下一轮应当给 PGO 版留一份可对拍的构建。

结论：**新构建照发**（代码严格更少工作、user/cycles 略优），但对外口径仍是
"单线程约 54~56 ms/帧"，没有把这次改动吹成"又快了 3%"。

## 五十八、否定实验：DIFFWTD 的"两路纵向 + 掩码 + 混合"融合（2026-10-24）

按第五十七节的清单试了复合 DIFFWTD（约 0.75M 亮度像素/帧）的融合：新增
`mc_vert2_diffwtd_u16`，在纵向内核里一次算出两路 d16、由 `|p0−p1|` 生成掩码
（亮度写出供色度复用、色度读折算好的掩码）、再做 `Round2(m·p0 + (64−m)·p1, 10)`
混合并直接写平面。理论上省掉"两路各写一块 i32 + 掩码单独一趟 + 混合再读两路"
（约 26 B/像素的往返）。

实测（同二进制 `ME_NO_DIRECT_COMP_DIFF=1` 对拍，60 帧 × 8 轮、配对中位）：

| 版本 | 结果 |
|---|---|
| 融合（亮度 + 色度）| **+4.7%（0/8）** |
| 只融合亮度 | **+1.3%（1/8）** |

**更慢**，已整体回退。原因（与本轮 AVERAGE 融合的差别）：AVERAGE 的规范式能退化成
`pavgw`（一条指令、16 位通道），而 DIFFWTD 的掩码 + 加权混合是 **i32** 运算
（`mullo_epi32` ×2、可变移位、逐 8 通道），把原本由两个**专用窄内核**
（`diffwtd_mask8` 8×i32、`blend_masked8` 8×i32，全部命中 L1 热缓冲）分担的工作
塞进一个"两路 madd + 掩码 + 混合"的大内核，寄存器压力与指令数都上去了；
中间缓冲本来就是 L1 热的 ⇒ "少写内存"换不来时间。又一次印证：
**热缓冲上的分步内核比融合大内核快**（与 2026-10-06 的 CDEF 块级拷贝失败同源）。

配对注意：这套对拍用的是**非 PGO** 构建 + env 开关，方向明确（0/8、1/8）才敢下结论；
若差异在 ±2% 内，按第五十七节的协议应改用同一 profile 的双构建复测。

## 五十九、两处小刀：运动场存储的"过去参考"表 + `BlockGrid::stamp` 短行写（2026-10-24）

### 1. `motion_field_storage` 的 `usable` 表 —— −0.9%（6/8）

`motion_field_storage` 每格每列表都调一次 `get_relative_dist(order_hint[ref_idx], order_hint)`
（1080p 一帧 6.5 万次），而它的结果**只跟参考帧类型**（LAST..ALTREF 7 种）有关。
做成 8 项 `usable[r]` 表，格子里只剩一次查表（`ref_idx` 的间接也一并省掉）。
同二进制 A/B（`ME_NO_MF_TABLE=1`）：**−0.9%（6/8）**。

⚠️ 与第四十八节 `project_frame` 里那次"提表"失败的区别：那次提表把**每格都要算的
裁剪/有效性判断**换成查表，破坏了流水线；这次换掉的是**每格两次 i64-ish 的距离计算**
（`get_relative_dist` 内含分支与移位），表只有 8 字节、必然在 L1。

### 2. `BlockGrid::stamp` 的短行写 —— −0.4%（5/8，噪声边缘）

用 `RUSTFLAGS="-C debuginfo=1"` 构建后 `sample`，调用图直接把 `bzero` 的来源指到
`block.rs:843`（`stamp` 的闭包行）。`stamp` 每块每行要写 7~13 个数组，而 4×4/8×8
块的行宽只有 1~4 格 ⇒ 每行十几次 `slice::fill`（= `memset` 调用）。
新增 `fill_span`（长度 ≤ 16 时展开成写循环）替换掉了热闭包里的全部 `.fill`。

同二进制 A/B（`ME_NO_FAST_FILL=1`）：**−0.4%（5/8）** —— 比采样暗示的 1.4% 小得多，
说明 PGO 下 LLVM 已经把大部分短 `fill` 内联成了存储，libc 调用的只剩少数档位。
保留（语义完全一致、方向为正），但**不把它算成一笔大收益**。

### 本轮的方法论收获

1. **debuginfo + `sample` 的调用图能直接给到 Rust 源码行**（`block.rs:843`）——以后
   找"谁在 memset/memmove"不用再猜；
2. 小刀在 PGO 下普遍只剩 0.3~0.9%：`stamp` 短写、运动场表都验证了这一点；
   要靠这个量级堆到 33 ms 需要几十刀，**结构性改造（u8 原生 / 宽内核）才是主线**。

### PGO 复测与交付（第六十九刀之后）

* 用 `tools/pgo.sh /tmp/a.mp4`（1500 帧）重建 PGO：**`ff76d12e…`**；
* **同一份 PGO 二进制**里用 `ME_NO_MF_TABLE=1 ME_NO_FAST_FILL=1` 对拍两处小刀：
  **−0.5%（5/8，100 帧 × 8 轮）** —— 与两块非 PGO 的
  −0.9%、−0.4% 方向一致，量级都在噪声边缘；保留（语义不变、方向为正）；
* **100 帧逐字节相同**（默认 vs 两个开关都开，md5 `6e7c64bc…`）；
* 发货产物回填 `extra/darwin/x64/engine/zuvrust` md5
  **`ff76d12eded2827d3dcd2158742c4da4`**；构建在 `target/pgo/release/zuvrust`。

### 顺带的否定实验：64/128 宽行拷贝改内联 —— +1.0%，已回退

`copy_row_u16` 只内联到 32 宽，64/128 宽落到 `copy_from_slice`（memcpy）。按 16 字节
块内联展开后同二进制 A/B **+1.0%（3/8）** —— 128/256 字节这一档 libc memcpy 更快
（ERMS/宽载入），已回退并留注释。又一次"手写循环不总是比 libc 快"。

## 六十、帧内预测的三次堆分配 + 逐像素写回 + 分区网格铺块（2026-10-24）

采样（debuginfo 调用图，按叶子行聚合）把 `predict_intra` 的两类浪费摆出来：

1. **每个帧内块 3 次 `vec![0; …]`**（`above`/`left` 各约 2 KB、`pred` 最大 32 KB）；
2. 写回是**逐像素** `plane.set(x, y, v)`（每像素一次 `y*stride+x` 乘法 + 枚举分派）。

改法：

* [intra.rs] 把三个缓冲换成本地线程复用（`thread_local!` + `RefCell`，**每块只借一次**，
  不在像素循环里）；保留"先清到原范围"的语义（未写到的位置仍读 0）。新增
  `ME_NO_INTRA_POOL` 用于对拍。
* 写回改成**按行 `copy_from_slice`**（u16 平面；u8 存储仍走逐像素，因为 `Plane::set`
  里还有 `min(255)`）。新增 `ME_NO_INTRA_ROWCOPY`。
* [partition.rs] `PartitionGrid::fill` 也是逐 4×4 格的"两写一判"，改成**按行**：
  越界判断每行一次、`fill_span` 一次铺满一行，重叠计数用一次线性扫描
  （顺序无关，只累加）。新增 `ME_NO_PART_FILL`。

同二进制 A/B（80 帧 × 8 轮、配对中位）：

| 改动 | 结果 |
|---|---|
| 帧内写回按行 | **−0.9%（6/8）** |
| 帧内缓冲复用 | **−1.0%（5/8）** |
| 分区网格按行铺块 | **−0.9%（5/8）** |

合计约 **−2.8%**（各项分开测的，合并后会有重叠）。24 帧逐字节相同。

⚠️ 这三处都是"每块/每格一次"的簿记，单项 1% 上下 —— 与第五十七节的大刀（复合预测）
不是一个量级；但它们是**纯 CPU 指令数**的削减（不是内存往返），在 PGO 上也应当站得住
（见下节的同二进制 PGO 复测）。

### PGO 复测（帧内/分区三处小刀）

用 `tools/pgo.sh /tmp/a.mp4` 重建（**`fed27be1…`**），在**同一份 PGO 二进制**里用
`ME_NO_INTRA_ROWCOPY=1 ME_NO_INTRA_POOL=1 ME_NO_PART_FILL=1` 对拍：

* **−4.8%（8/8，100 帧 × 8 轮）**：52.16 → **49.93 ms/帧**（这一窗口机器偏快，
  只用于配对比较）；非 PGO 三项分开测是 −0.9%/−1.0%/−0.9%，PGO 下合并收益更大
  （说明这几处"每块一次"的簿记在 PGO 的内联布局里被放大）；
* **100 帧逐字节相同**（默认 vs 三个开关全开，md5 `6e7c64bc…`）；
* 与上一版 PGO（`ff76d12e…`）跨构建对拍：**−1.2%（7/10）** —— 跨构建噪声仍在，
  以同二进制数为准；
* 发货产物回填 `extra/darwin/x64/engine/zuvrust` md5
  **`fed27be161081e1090d085c684194b5e`**；
* `cargo test --workspace --release`：见下（1029 项）。

## 六十一、帧间块的边滤波类型按需计算 + SIMD 残差加回 + CDEF 方向门（2026-10-24）

三处"每块/每样本"的簿记：

1. **`filterType` 按需**：`decode_block` 对**每个帧间块**都算两遍 `filter_type_for`
   （各 6 次邻居网格查询），而它只被 **interintra** 使用。改成 `interintra` 时才算。
   同二进制（`ME_NO_LAZY_FT=1`）：非 PGO **−0.8%（6/8）**；PGO 同二进制复测
   **+0.3%（3/8）** —— 落在噪声里，**保留**（语义不变、少做工作；如实记录未获
   PGO 证据）。
2. **残差加回 SIMD**：逆变换收尾原来是逐像素 `clip1(dst + res)`；改成一次 16 个
   （`cvtepu16_epi32` ×2 + 加 + 钳位 + `packus`）。同二进制（`ME_NO_SIMD_RESADD=1`）：
   **−0.6%（6/8）**。
3. **CDEF 方向搜索按需**：`cdef_direction`（8 方向代价）只在"亮度主抽头 ≠ 0 或
   色度主抽头 ≠ 0"时才用得到（`dir` 在 pri==0 时恒取 0、`var` 只喂 `pri_str`）。
   两处调用点都加了门。同二进制（`ME_NO_CDEF_DIRGATE=1`）：**−1.1%（5/8）**。

24 帧逐字节相同。三处都随下一次 PGO 重建一起验证（同二进制、100 帧 × 8 轮）。

### PGO 复测（六十一节三项合并）

`tools/pgo.sh /tmp/a.mp4` 重建（**`a5a30ea3…`**），同一份 PGO 二进制里
`ME_NO_LAZY_FT=1 ME_NO_SIMD_RESADD=1 ME_NO_CDEF_DIRGATE=1` 对拍：

* **−4.5%（8/8，100 帧 × 8 轮）**：59.89 → **56.63 ms/帧**（该窗口机器偏慢，
  只用于配对）；
* **100 帧逐字节相同**（默认 vs 三个开关全开，md5 `6e7c64bc…`）；
* 发货产物回填 `extra/darwin/x64/engine/zuvrust` md5
  **`a5a30ea356bb00bf7bfbbd8cf8bf8fe9`**；
* `cargo test --workspace --release`：见下。

### 本轮累计（2026-10-24 全天）

从 `b162e7d1…`（~54–56 ms）到 `a5a30ea3…`：复合 AVERAGE 直写/融合纵向
（同二进制 −2.9%）、运动场表 + stamp 短行、帧内缓冲复用/按行写回、分区网格按行、
边滤波类型按需、残差加行 SIMD、CDEF 方向门 —— 每个改动都有 env 开关可对拍；
PGO 同二进制合计约 **−8~9%**（分两批测：−4.8% + −4.5%，有重叠），
跨构建对拍只有 −1~2%（PGO 布局噪声），**如实以同二进制为准**。

剩余大头（按当前采样）：`deblock_edge4` 6.4% + `run_line` 2.6%（4 路 i32、
每 4 像素一调用 —— 要 8 路化必须重排按 4 像素一行的遍历，3–5 轮）、
`vt_forward_1d` 5.2%（4×4/8×8 走 16 位双行打包才能再翻倍）、
CDEF 两个内核 ~7%（候选 gather 已无处可省，见分析）、`decode_block` 8.8%（熵解码
簿记，dav1d 也要花这一档）。要到 33 ms 仍需 u8 原生重建或这些内核的结构性重写。

## 六十二、去块横边 8 路内核（相邻两条 4 像素线合并）—— 同二进制 −2.7%（6/8，2026-10-24）

去块是仅剩的"4 路 i32"内核：一次只滤 **4 像素**（`deblock_edge4` 的 4 个通道），
而 `run_pass` 的内层是按 4 像素一条线走的。**横边（pass 1）沿列方向两段互不依赖**
（每段的读窗口跨边、只碰自己那 4 列）⇒ 可以把相邻两条线合并成一次 8 路调用。

实现：

1. `loop_filter_edge` 拆成 `edge_params`（只算参数，返回 `EdgeJob`）+ `apply_edge`
   （4 路 SIMD 或标量落笔）—— 合并路径要**先比较两段的参数**（滤波尺寸、阈值、
   内点判定都必须一致）再决定走不走 8 路；
2. 新增 `simd::deblock_edge8_h`：`deblock_edge4` 的 256 位版（窗口加载、掩码、
   窄/宽 8 抽头/宽 16 抽头、写回全部照搬；横边的写回是"两行一次打包"）。
   竖边要 8×8 转置，暂不走；
3. `run_outer` 在 `pass == 1` 时成对处理外层下标；两条线的**变换块序列**先整条比对，
   对不齐就整条退回逐段；参数不一致的段也只退回那一段。

验收：

* 单测 `deblock_edge8_h_matches_two_edge4_calls`：7 组（fs 4/8/16 × 亮度/色度 ×
  8/10/12 bit）在随机 + 平坦/台阶图案上，与"两次 4 路调用"**逐字节相同**；
* **24/100 帧逐字节相同**（默认 vs `ME_NO_WIDE_DEBLOCK=1`）；
* 同二进制 A/B（80 帧 × 8 轮、配对中位）：**−2.7%（6/8）**；
* PGO 复测见下节。

### PGO 复测（去块横边 8 路）

`tools/pgo.sh` 重建（**`dcc47604…`**），同一份 PGO 二进制里 `ME_NO_WIDE_DEBLOCK=1` 对拍：

* **−1.4%（7/8，100 帧 × 8 轮）**：56.05 → 55.26 ms/帧（非 PGO 上是 −2.7%）；
* **100 帧逐字节相同**（md5 `6e7c64bc…`）；
* 单测 `deblock_edge8_h_matches_two_edge4_calls` 通过；
* 发货产物回填 `extra/darwin/x64/engine/zuvrust` md5
  **`dcc476043edb0769d37647f2f27d5d24`**。

竖边（pass 0）也能同样合并，但写回要 8×8 转置（2n = 4/6/12 三种宽度），
是下一步；预计再 −1～2%。

## 六十三、符号解码器：去掉每符号的固定开销（2026-10-24）

`read_symbol_inner` 每帧要跑 **43 万次**（项目里早先数过），所以"每符号一次"的
任何开销都会被放大：

1. **CDF 的两条不变量检查**（`cdf.len() >= 2`、`cdf[n-1] == 1<<15`）折成
   `debug_assert!` —— 它们检查的是**我们自己的表**（默认表 / 参考槽载入），
   不是码流数据；真正防越界的 `symbol >= n` 仍在 release 里；
2. **三个诊断开关**（`trace_sym` / `record_recent` / `smb_limit()`）原来每符号各做一次
   `OnceLock` acquire load，改成 `SymbolDecoder` 构造时取一次、存字段。

同二进制 A/B：**−1.0%（5/8）**（第一项）→ 两项合并后 **−1.5%（6/8）**。
24/100 帧逐字节相同。

⚠️ 顺带否掉一个：把 `read_bits_at` 的 `data.len()*8` 缓存到字段（"每符号一次乘法"）
实测 **+0.4%（2/8）** —— 多一个函数参数反而破坏了热路径的内联，已回退。
又一次"想当然的更少工作 ≠ 更快"。

### 同 profile 的 PGO A/B（六十一节三项之外的符号解码器）

为绕开"跨构建 PGO 噪声"（第五十七节），这轮改用**同一份 `merged.profdata`**
分别构建新旧两版：

* 新：`target/pgo/release/zuvrust` = **`aa4dabc0…`**（含符号解码器改动 + 横边 8 路去块）；
* 旧：把熵解码那三处改回原样、用**同一个 profile** 构建（`36ca90b6…`）；
* A/B（100 帧 × 8 轮、配对中位）：旧 46.45 ms/帧 → 新 46.50，**−0.5%（4/8）** ——
  方向为正但很小（非 PGO 上是 −1.5%）。**同一窗口里绝对数 46.5 ms/帧**，是
  本机目前见过的最好窗口（此前 49.9）。
* 24 帧逐字节相同（新旧两个 PGO 二进制）；
* 发货产物回填 `extra/darwin/x64/engine/zuvrust` md5
  **`aa4dabc07ac44fcd0e82d0c047a4913f`**。

⚠️ 这轮还验证了"同 profile 双构建"这条路可行：`RUSTFLAGS="-Cprofile-use=…merged.profdata"`
+ 独立 `CARGO_TARGET_DIR`，比跨 PGO 构建干净得多（第五十七节那条协议现在有了实操）。

## 六十四、CDEF 克隆只拷可见行（padding 不再白拷）—— 同二进制 −1.3%（6/8，2026-10-24）

`cdef::apply` 每帧 `frame.clone()` 整份拷贝（1080p 约 6.9 MB），其中底部
`PLANE_PAD = 72` 行 padding 占亮度 **6.25%**、色度 **11.8%** 的拷贝量。
而 padding **从不进输出**（`ME_PAD_POISON` 把 padding 填 0xAA 后输出逐字节相同）。

新增 `Plane::clone_visible` / `CurrFrame::clone_visible`：池里取同长度缓冲后只
`copy_from_slice` 可见的 `stride * h` 行，其余留池里的旧内容；CDEF 用它替换
`frame.clone()`（`ME_NO_CDEF_VISIBLE_CLONE=1` 可退回整份克隆）。

* 同二进制 A/B（80 帧 × 8 轮、配对中位）：**−1.3%（6/8）**，51.74 → 51.14 ms/帧；
* **24/100 帧逐字节相同**；PGO 复测见下。

### PGO 复测（可见行克隆）

* 新 PGO **`32275877…`**；同一二进制 `ME_NO_CDEF_VISIBLE_CLONE=1` 对拍：
  **−1.1%（6/8，100 帧 × 8 轮）**，47.30 → **46.52 ms/帧**（冷机窗口）；
* **100 帧逐字节相同**；
* 发货产物回填 `extra/darwin/x64/engine/zuvrust` md5
  **`3227587700e65cb31d0bca0745934a0a`**。

### 本轮否定实验（两条）

1. **横边 + 竖边都合并成 8 路**：竖边写回要 8×8 转置，实测把横边单独拿到的 −2.7%
   全部吐回去（合并后 +0.2%）⇒ 只保留横边。
2. **NV12 输出显式 AVX2**（亮度收窄 + UV 交错）：+0.1%（4/8，噪声内）——
   与 2026-10-05 那次的结论一致：标量循环已被 LLVM 向量化、这一档是内存受限。

## 六十五、CDEF：整帧无滤波组 ⇒ 零拷贝返回（2026-10-24）

`cdef::apply` 进门先克隆整帧。但 `cdef_idx` 是**组级**表：若 510 个组全是 `-1`，
规范输出就逐像素等于 `CurrFrame` ⇒ 可以直接 `Ok(None)`（上一层零拷贝复用）。
现在在克隆**之前**扫一遍组级表（约 510 次读，代价可忽略），全跳过就返回 `None`。

静止段落/低强度段确实会出现这种帧；本素材上收益很小（绝大多数帧都有滤波组），
属于"零成本买一个边界情况"。

### 同轮否定实验：相位 0 的横向 δ 快路径

`phase_x == 0` 时横向趟是 δ（`out = src << (7-round)`），省 8 次乘加。实现后
同二进制 A/B 只有 **−0.4%（4/8）** —— 相位 0 的块太少，已在噪声内，**回退**。
（调试途中发现中间量的行布局是"第 r 行 = 源行 `y0+r-3`"，写错会整体错 3 行；
单测 `mc_shift_rows_i16_matches_the_delta_formula` 一跑就抓到 —— 又一次"先写对拍单测"的价值。）

### PGO 交付（CDEF all-skip 之后）

* 新 PGO **`a594856e…`**；24 帧逐字节（`c6a4273…`）；
* 发货产物回填 `extra/darwin/x64/engine/zuvrust` md5
  **`a594856e9c4a72c1848d56a64203e917`**；
* 测试与 100 帧见下。

## 六十六、CDEF 候选方向表：逐块现算 → 编译期常量表 —— 同二进制 −1.0%（7/8，2026-10-24）

`cdef_candidates` 每个块都要算 12 个候选的 `(dx, dy)`：主抽头 2×2 + 次抽头
4 组（方向 ±2 旋转、各 ±1），带符号、带循环分支 —— 1080p 一帧几万次。
改成 `const CDEF_DIRS_TAB: [[[i8;2];12];8]`（编译期展开，`assert!` 钉住长度），
运行时只剩 12 次表读 + 12 次 `dy*stride+dx`。

* 同二进制 A/B（80 帧 × 8 轮）：**−1.0%（7/8）**；
* PGO 同二进制（100 帧 × 8 轮）：**−0.7%（5/8）**，49.55 → 49.26 ms/帧；
* 24/100 帧逐字节相同；新 PGO **`262cfeed…`**，发货产物已回填。

## 六十七、翻转残差加回的 SIMD 化 —— 同二进制 −1.7%（6/8，2026-10-24）

逆变换收尾的 `flip_ud` / `flip_lr` 分支（部分 ADST 变体）一直是**逐像素标量**：
`dst[yy][xx] = clip1(dst[yy][xx] + res[i][j])`，`xx = w-1-j`。

改法：

* `flip_ud`（只上下翻）：行号映射 `yy = h-1-i`，其余照旧走 [`add_residual_u16`]；
* `flip_lr`（左右翻）：新增 `add_residual_u16_rev` —— 把残差行**倒序**后再走同一条
  16 宽流水线（`_mm256_permutevar8x32_epi32` 两次 + 交换两半），目标行仍是正序；
* 两条都翻：行映射 + 倒序。

同二进制 A/B（`ME_NO_FLIP_RESADD=1`，80 帧 × 8 轮、配对中位）：**−1.7%（6/8）**，
51.92 → 51.72 ms/帧；24/100 帧逐字节相同。PGO 复测见下。

### PGO 复测（翻转残差加回）

* 新 PGO **`826e29d7…`**；同二进制 `ME_NO_FLIP_RESADD=1` 对拍：
  **−0.9%（6/8，100 帧 × 8 轮）**，51.10 → 50.75 ms/帧；
* **100 帧逐字节相同**；发货产物回填 `extra/darwin/x64/engine/zuvrust` md5
  **`826e29d717c6ddea3851b52120891f54`**。

## 六十八、反量化改 8 路 SIMD（i32 内做完 64 位式）—— 同二进制 −1.9%、PGO −3.3%（2026-10-24）

`dequantize` 一直逐系数标量：`dq = lv·q`（规范是 i64）→ `|dq| & 0xFFFFFF` → `>> dqDenom`
→ 再 `Clip3`。看起来必须 64 位，其实**不需要**：

* `(lv·q) mod 2^24` 用 `mullo_epi32` + `& 0xFFFFFF` 就能拿（低 24 位与 64 位乘积一致）；
* `dq < 0 ⇔ lv < 0`（量化步长恒正）⇒ 符号只看 `lv`；
* `|dq| mod 2^24 = lv >= 0 ? m : (-m) mod 2^24`。

于是新增 `simd::dequantize_8`：8 个系数一批（全零批直接写 0 跳过），
只在 `w == tw && h == th`（边长 ≤ 32）时启用，64×64 的非常规布局仍走标量。

* 单测 `dequantize_8_matches_the_scalar_formula`：6 组参数（含负系数、DC/AC 不同步长、
  三种 dqDenom、8/10/12bit、极端 `q = 2^19`）与标量式逐位相同；
* 同二进制 A/B：**−1.9%（7/8）**；PGO 同二进制：**−3.3%（7/8）**，
  **49.61 → 47.56 ms/帧**；
* 24/100 帧逐字节相同；新 PGO **`fe3e7106…`**，发货产物已回填。

## 六十九、MC 横向的边界块：钳位窗口拷到栈上 + 同一条 SIMD（2026-10-24）

`mc_horiz_rows_8bpc` 里 `interior_x == false` 的块（贴帧左右边）整行走**逐样本标量**：
每像素 8 次"钳位 + 乘加"（debuginfo 采样里这一行占 **2.9%**）。

改法：边界块先把 `[base-3, base+wu+3)` 这段窗口**钳位复制到栈缓冲**（≤134 个样本），
后面的 16 宽 / 8 宽 / 标量尾全部从 scratch 读 —— 与逐样本钳位**逐位相同**。

同二进制 A/B（`ME_NO_HORIZ_EDGE_SIMD=1`）：**−0.5%（6/8）** —— 比采样暗示的小，
说明这份素材里真正贴边的块不多；24/100 帧逐字节相同。保留（方向为正、边界有界）。

⚠️ 顺带否掉一个：**运动场投影 SIMD（8 路 i32，7 个目标帧一次算）**实测 **+0.1%（3/8）**
—— 每格的开销主要在"读参考帧的运动场 + block_position"，投影本身不是瓶颈，已回退。

### PGO 复测（边界块横向 SIMD）

同一份 PGO（**`b06de9c8…`**）里 `ME_NO_HORIZ_EDGE_SIMD=1` 对拍：**−0.0%（4/8）**
—— 非 PGO 的 −0.5% 在 PGO 下归零。改动语义等价、有界，**保留**（如实记录无 PGO 收益）；
100 帧逐字节相同；发货产物回填 `extra/darwin/x64/engine/zuvrust` md5
**`b06de9c8ef2376e7f0bf293d8ba798a4`**。

### 否定实验：`lf_tx_sizes` 铺块改按行 —— +0.7%（4/8），已回退

`transform_block` 结尾把 `step_y × step_x` 个 4×4 格写成 `lf_tx_sizes`（逐格带
`if rr < rows && cc < cols`）。改成按行 + `fill_span` 后同二进制 A/B **+0.7%（4/8）**
—— LLVM 对"逐格 store + 可预测边界"的展开比"fill_span 的小循环 + 长度分支"更快。
已回退（又一次"手写循环不总是更快"，与 64/128 宽行拷贝那次同源）。

## 七十、`INTRA_SCRATCH` 懒初始化 → `const` 初始化 —— 同二进制 −0.5%（5/8），PGO `a7202fa7…`（2026-10-24）

debuginfo 采样里 `intra.rs:506` 独占 **0.9%**，而那一行只是 `INTRA_SCRATCH.with(|s| ...)`。
原因不是借用，是 **TLS 懒初始化**：`thread_local!` 用默认的 `RefCell::new(...)` 时每线程第一次访问
会走 `register_dtor`/惰性初始化桩（`__tls_get_addr` + 初始化检查），热路径上每次 `with` 都要过一次分支。

改成 `const { RefCell::new(IntraScratch { above: Vec::new(), left: Vec::new(), pred: Vec::new() }) }`
（`Vec::new()` 是 const 的），初始化前移到 TLS 映像里，`with` 只剩取地址。

* 同二进制 A/B：**−0.5%（5/8）**；
* 24/100 帧逐字节相同；PGO **`a7202fa7a150725aadb0e4ac95a2b88f`**；
* 发货产物回填 `extra/darwin/x64/engine/zuvrust`（md5 已核对）。

⚠️ 顺带复习：这刀的价值不在 0.5%，而在确认「`thread_local!` 懒初始化在热循环里是要花钱的」——
以后新增 TLS 缓冲一律写 `const {}` 初始化。

## 七十一、2026-10-24 会战：簿记四刀（同 PGO −4.7%，10/10 轮）

这一轮先用 `AV1_PHASE_TIMING` 把帧拆成 `tiles / filters / out / setup`，再用
debuginfo 采样对着**行号**排了一遍。两个此前被低估的事实：

1. **滤镜链不是 10%，是 ~25%**（CDEF ~11%、去块 ~8%、LR ~6%，PGO 后的同窗口口径）；
2. **每块/每边的"算参数"比"干活"还贵**：`deblock::edge_params` 3.6%、
   `mark_decoded` 1.3%、`stamp` 的零填充 3%+、CDEF 候选表 1%+。

四刀（都留了 env 回退开关、都逐字节对拍）：

| 刀 | 做法 | 非 PGO A/B | 同 profile PGO |
|---|---|---|---|
| `mark_decoded` | 逐格"两次越界判断 + 写"→ 先夹行/列区间再按行 `fill(1)` | −1.9%（6/8，与下一刀合并测） | −2.2%（7/8，前三刀合并） |
| `stamp` 跳默认值 | 对 `reset_all` 用 **`fill_set`** 铺过初值的数组（comp/wedge/mask/interp/mvs/interintra/uv/seg），值等于帧初值时跳过整行写。`ensure_len` 那批（y_modes/ref_frames/skips/…）**绝不能跳** | 同上 | 同上 |
| `edge_params` / `apply_edge` | `#[inline(always)]` ⇒ `run_line` 循环里 `sub_x/sub_y/cols/plane_idx` 与 `lf_tx_sizes` 基址被提出循环 | −2.6%（8/8） | 同上 |
| CDEF 候选表 | `(方向, 主抽头档)` 跨块缓存 12 个 `offs`/`taps`（组内方向常相同），`cdef_candidates`/`cdef_direction` 加内联 | −1.9%（5/8） | 同上 |
| LR `RESTORE_NONE` | 逐样本读进 i32 行缓冲 + `set_row` → **直接按行 `copy_nonoverlapping` u16** | — | −2.2%（7/8，与本行下一刀合并） |
| 运动场存槽 | 最后一个刷新槽 `mem::take` 代替 `clone`（1080p 一份 1.5 MB） | — | 同上 |

* 同 PGO 汇总：**52.21 → 49.88 ms/帧，−4.7%（10/10 轮更快）**；
* 24/100 帧逐字节、全量测试 **1031 passed / 0 failed**；
* 新 PGO 产物 md5 **`299227b9d82c90406f197556f7cddb88`**，已回填。

⚠️ 顺带否掉：**把 DCT 的位反转置换折进转置装载**（省 `vt_dct_permute` 的整块
拷贝）实测 **+0.7%（3/8）** —— 散射写的寻址开销吃掉了拷贝，已回退。
又一次"少搬内存 ≠ 更快"。

下一步（按新采样的自时间）：`vt_forward_1d` 5.2%（i32 8 通道蝶形）、
`cdef_pair16x4/quad8` 6%（12 抽头 i16）、`edge_params` 剩下的部分、
`memmove+bzero` 仍占 ~9%（CDEF 整帧克隆是大头，要改成"未过滤块逐块拷"才能省）。

### 否定实验二则（同窗口、逐字节相同才看时间）

1. **NV12 亮度收窄改显式 AVX2**（`packus` + 4×64 位重排，32 像素一组）：
   **+1.9%（2/8 轮）**。标量式 `(*sv >> shift) as u8` 虽是"截断"，LLVM 的自动向量化
   已经把它变成 pack 类指令；手写的逐行调用反而更贵。**已回退**（这是第二次踩到
   "NV12 显式 SIMD"，与第七批的 +0.1% 同源）。
2. **DCT 位反转置换折进转置装载**（散射写 `v[brev(i)]`、省 `vt_dct_permute` 的
   整块拷贝）：**+0.7%（3/8 轮）**。散射写的寻址把拷贝省下的又吃回去了。
   **已回退**。

### 补记：运动场落槽复用（同 PGO −0.4%，5/10）

`motion_field_storage` 每帧 `vec![NONE; n] + vec![[0,0]; n]` = 1.5 MB；而**最后一个
刷新槽**的旧内容在本帧解码结束后已经死了 ⇒ 用 `motion_field_storage_into` 把槽里的
两个 `Vec` 取出来直接当落点（顺带按行切片、去掉逐格乘加与 4 次越界判断）。
同 PGO **−0.4%（5/10）**，在噪声边上，保留（无副作用、省内存抖动）。
新产物 md5 **`150594f75a352f486dad05fb2b857820`**，已回填。

## 七十二、2026-10-24 第二轮：帧内预测的 SIMD 与去块内联（同 PGO −1.1%）

三刀（都逐字节、都留 env 回退）：

1. **`intra::smooth` 的 8 路 SIMD**（`simd::intra_smooth_row`）：规范式每像素 4 次
   乘加 + 4 次带 `EDGE_OFF` 的取址，原来**全标量**。把 `j` 方向铺成 8 个 i32 通道
   （`wy·A[j] + (256−wy)·Llast + wx[j]·L[i] + (256−wx[j])·Alast`，V/H 各去一项），
   取整只在最后做一次 ⇒ 与标量式逐位相同。
   非 PGO **−2.3~3.0%（8/8）**、**同 PGO −1.1%**（PGO 已经把标量回路编得不错）。
2. **`predict_intra_buf` 不再清零 `pred_buf`**：所有预测器都写满 `w×h`
   （`dc` 是 `fill`、`smooth`/`basic`/`directional` 双循环、`recursive_intra` 按 4×2
   铺满，且只读同块内更早写过的邻居）⇒ 清零是纯白写（32×32 每次 4 KB）。
3. **`loop_filter_edge` 加 `#[inline(always)]`**：与 `edge_params`/`apply_edge`
   的内联配套，让 `run_line` 的循环不变量（`sub_x/sub_y/cols`、`lf_tx_sizes` 基址）
   继续被提出循环。非 PGO −0.5%（6/8）。

* 同 PGO 对上一版发货产物：**49.56 → 48.55 ms/帧，−1.1%（6/10）**；
* 24/100 帧逐字节、全量测试 **1031 passed / 0 failed**；
* 新产物 md5 **`a0573bbb18094da55ba245c8d814f6c0`**，已回填。

⚠️ 又一次确认"**先用采样判定'是不是标量'再动手**"：`smooth` 只占 1.6% 采样，
但换成 SIMD 后整帧掉 2~3% —— 标量循环在采样里被摊薄，实际是"每像素十几次操作"。

### 否定实验三则

* CDEF "不克隆 + 按组/块补拷"：**+1.3%（4/10）** —— 整帧克隆其实在**预热目标页**，
  省下的 6.25 MB 拷贝被滤镜写入的缺页/写分配吃回去还倒贴。已回退。
* （前一轮）NV12 亮度显式 AVX2 +1.9%、DCT 位反转装载散射 +0.7%。

### 第二轮的两个否定（都没进产物）

* **运动场投影去重**（源参考类型指向同一槽 + 同 order_hint + 同 dst_sign 时跳过）：
  非 PGO −0.9%（9/10），**同 PGO +0.3%（4/10，噪声）** ⇒ 回退（保持源码 == 发货产物）。
* **逆序投影 + 逐格"最后写者"掩码**（规格上后投影覆盖先投影）：**+0.4%（4/10）** ⇒ 回退。
  ⚠️ 中途踩到一个真坑：同一源内部也有多对一投影（多个源格投到同一目标格），
  第一版用"写过的位图"把**源内**的重复写也跳了，**第 11 帧码流就失步**
  （`coeff_base_eob` 多读 3 位）。正确判据是"最后写它的源序号 > 本源的序号"。
  虽然修好后逐字节一致，但速度没赚 —— 说明这份素材里跨源重复投影不多。

## 七十三、2026-10-24 第三轮：CDEF 抽头归并 + 角度预测行快路径（同 profile −0.9%）

1. **CDEF 同抽头候选先相加再乘**（`cdef_pair16x4` / `cdef_quad8`）：
   `cdef_candidates` 的抽头是成组的（主 c=0/1、c=2/3；次 c=4..7、c=8..11 各同 tap）。
   整数加乘在**模 2^16** 下构成环 ⇒ `v0·t + v1·t ≡ (v0+v1)·t`，与逐候选乘加**逐位相同**；
   每行 12 次 `vpmullw` → 4 次。单测 `cdef_pair16x4_matches_pair16_bit_for_bit` /
   `cdef_quad8_matches_cdef_pair16_bit_for_bit` + 100 帧逐字节都过。
   **同 profile PGO −0.9%（7/10）**。
2. **`directional` 的 90°/180° 行快路径**：V/H 模式每行分别等于 `AboveRow[0..w]`
   与常数，原来逐像素取址 + 窄化；现在一次转 u16 / 一次 `fill` 后按行拷。

* 全量测试 **1031 passed / 0 failed**；新产物 md5 **`24c5e3d2104dd53f62806c1d3e3672c6`**。
* ⚠️ 跨构建（重新训练 PGO）对上一版发货产物是 **±0.0%（5/10）** —— 又一次撞到
  ±2~3% 的跨构建噪声地板；**代码收益以同 profile A/B 为准**。

### 阶段实测（当前构建，100 帧中位数）

`tiles 23.2ms` + `filters 17.4ms`（去块 4.8 / **CDEF 12.2** / LR 0~5）+ `out 0.6` +
`setup 1.7` + `mf 0.9` ≈ 46.5ms。**CDEF 是单笔最大（约 25%）**，且它的内核已经
是 16 通道 i16 —— 再往下只能减运算（本轮）或换 u8 通道（大改）。

## 七十四、2026-10-24 第四轮：读取位窗口 + CDF 循环（同 profile：语法阶段 −2.9%、整帧 −0.5%）

`AV1_SYNTAX_ONLY=1` 把语法阶段隔出来是 **11.3 ms/帧**（整帧 ~45 ms 的 25%），
信噪比比整帧高约 3 倍 —— 这一轮就只在这个阶段量。

1. **`read_bits_at` 的 8 字节窗口改一次非对齐读**：原来 `copy_from_slice(&data[..8])`
   走 memcpy，而它**每次重归一化**都调用（一帧 43 万符号，绝大多数要进）。
   改成 `u64::from_be(read_unaligned(ptr))`（x86 上就是 `movq` + `bswap`）。
2. **CDF 符号循环去掉可证明冗余的边界检查**：开头已经验过 `cdf[n-1] == 1<<15`
   ⇒ 在 `symbol == n-1` 处 `f == 0`、`cur == 0`，而 `symbol_value >= 0` 恒成立
   ⇒ 循环**必然在 ≤ n-1 处 break**。于是 `cdf[symbol]` 用 `get_unchecked`，
   那条 `symbol >= n` 的错误返回折成 `debug_assert`（它的触发条件已被开头校验排除）。

* 同 profile PGO：**语法阶段 11.25 → 10.83 ms（−2.9%，9/10）**、**整帧 −0.5%（6/10）**；
* 100 帧逐字节、全量测试 1031 / 0；
* 新产物 md5 **`d6758ec0be4e41eef335444289d6cd0a`**（跨构建对上一版 ±0.0%，又撞噪声地板）。

⚠️ 本机窗口在 44.8~50.8 ms 之间漂（无风扇），**只看配对中位，不看绝对毫秒**。

## 七十五、2026-10-24 第五轮：CDEF 方向搜索两块并行（同二进制 −1.1%、发货 PGO −2.0%）

方向搜索是逐行的 `pshufb` + 加法；`vpshufb` 本来就**按 128 位半区独立**工作 ⇒
把**相邻两个 8×8 块**各放一个半区，用同一套掩码一次算完，指令数减半。

* 新内核 `simd::cdef_direction_avx2_x2(data, stride, x0a,y0a, x0b,y0b, sh)`：
  掩码/`rev8`/`rev4z`/`rh_*` 广播到两个半区；`pair_cost` 改成每半区各出一份
  （`hadd` 按半区独立）；`cost2`/`cost6` 的横向和也按半区提取。
* `cdef_group_fast` 里**按块行先算整行方向**（`dir_row[8]`）：是否算方向是**组级
  常量**，同一行里相邻且都"不 skip + 内点"的块成对走 x2，落单的走原来的 x1。
* 单测 **`cdef_direction_avx2_x2_matches_scalar_bit_for_bit`**（3 种位深 × 3 种图案 ×
  6 对坐标，与标量 `dir_accum_scalar`+`dir_cost` 逐位比对）；100 帧逐字节。

* 同二进制 A/B（`ME_NO_CDEF_DIR2=1`）：**−1.1%（11/12）**（另一窗口 −0.3%，6/10）；
* 重训 PGO 对上一版发货产物：**49.46 → 48.52 ms/帧，−2.0%（10/10）**；
* 全量测试 **1032 passed / 0 failed**；产物 md5 **`37a533f3c08e327bd2d8b91d93cd5086`**。

经验：**"每行一次 pshufb" 这类按行的内核，天然适合两块并行的半区分工**；
下一步同理可考虑 CDEF 的滤波器（`cdef_pair16x4` 已经是 2×YMM 状态，寄存器不够
再翻倍，但 chroma 的 `cdef_quad8` 可以按"两个相邻色度块"合并 gather）。

### 补记：短行拷贝 / stamp 快路径 / CDEF 抽头广播（同 profile 合计 ±0.0%）

* 帧内预测写回与 LR `RESTORE_NONE` 的行拷贝改走 `copy_row_u16`（≤32 样本内联定长
  拷贝，避开 `copy_from_slice` 的 memmove 调用）——单独测 **−0.7%（6/10）**；
* `BlockGrid::stamp` 增加"整块在网格内"的无夹取快路径（绝大多数块）；
* CDEF 两个内核的抽头广播按归并后的 4 个值构建（原来是 12 项 YMM 数组，会压栈）。
* 三项合计**同 profile 对上一版 ±0.0%（5/10）**，最终重训 PGO 再测 **−0.5%（5/10）**
  —— 都在噪声边上；因为都是**纯减少工作且逐字节等价**，保留并回填。
* 全量测试 **1032 / 0**；新产物 md5 **`55f9ee79de97329ec6f9f91bb289e82a`**。

## 七十六、2026-10-24 第六轮：CDEF `constrain` 的无符号饱和化简（同 profile −1.1%、发货 −1.5%）

`constrain` 的三条指令

```text
t = thr - (mag >> adj)          // 可能为负
t = min(t, mag); t = max(t, 0)
```

在 `mag ≥ 0`、`thr ≥ 0`（且都 < 2^15，本来就用 i16 承载）时**等价于**

```text
t = subs_epu16(thr, mag >> adj);   // 无符号饱和减：负数自然变 0
t = min_epu16(t, mag);             // 两边都非负 ⇒ 无符号 min 同值
```

⇒ 每个候选少一条指令（`vpmaxsw` 消失）。按 12 个候选 × 2 个半区算，CDEF 滤波内核
少约 8% 的 ALU 操作。

* 位对拍单测（`cdef_pair16x4_matches_pair16_bit_for_bit` /
  `cdef_quad8_matches_cdef_pair16_bit_for_bit`）全过、100 帧逐字节；
* 同 profile PGO **−1.1%（8/10）**；重训 PGO 对上一版发货产物 **−1.5%（6/10）**；
* 新产物 md5 **`9fcda4134bc21705541f462d10e43711`**。

经验：**"夹到非负再取 min"这类三段式，先看能不能用无符号饱和运算压成一段**；
CDEF 的 constrain 正是这种形状。

## 七十七、2026-10-24 第七轮：整像素预测的整块矩形拷贝（同 profile −0.9%、发货 −2.5%）

用"调用图里最近的带行号祖先"把 `memmove`（7.2%）精确归因，最大一笔是
**`inter::store_integer_pred` 的 `copy_row_u16`（inter.rs:596，135 个采样 ≈ 3.1%）**：
64 宽的整像素预测行（128 字节）每次都退化成一次 libc `memcpy` 调用。

改法：新增 `simd::copy_rect_u16(dst, dstride, src, sstride, w, h)` —— **整个块一次
调用**，行循环在 `#[target_feature(enable="avx2")]` 内核里面（`target_feature` 不能
内联进普通函数，逐行调用就等于每行一次真实调用）。32/16/8/标量四档兜底。
前提：y 方向不需要逐行钳位（否则源行会重复）、横向窗口完全在源平面内。

* 100 帧逐字节；同二进制 A/B（`ME_NO_RECT_COPY=1`）**−0.9%（7/10）**；
* 重训 PGO 对上一版发货产物 **50.62 → 49.40 ms/帧，−2.5%（8/10）**；
* 全量测试 **1032 passed / 0 failed**；产物 md5 **`27a811d7a3e43f08bb77485dabcdfe2a`**。

⚠️ 另一笔 `memmove` 大头是 `Plane::clone_visible`（intra.rs:315，113 采样 = 2.6%）——
就是 CDEF 的整帧克隆，已知"去掉更慢"（预热目标页），不动。

### 否定实验：`sgr_filter` 的四个中间缓冲挪进线程局部 —— +4.6%（1/10），已回退

`a`/`b`/`f0`/`f1` 是四个栈上定长数组（≈68 KB），看起来每次调用都在序言里整片清零，
而 `sgr_filter` 每帧要调上千次（LR 只在 23/104 帧启用，但启用时**单帧 LR 就是 21 ms**）。
改成 TLS + 只清 `a`/`b` 的上下边界行后：**+4.6%（1/10 轮）**。

原因：`box_filter` 内联之后 LLVM 能证明 `f0`/`f1` 被写满，**把整片初始化删掉了**；
而 TLS 的数组隔着 `RefCell` 借用，别名信息变差、死存储消除也失效 ⇒ 反而更慢。
**结论：热路径上的"复用缓冲"要看 LLVM 能否自己消除初始化，别想当然。** 已回退。

### LR 的实测口径（2026-10-24）

`FILTER` 分段：`lf` 均值 5.0 ms、`cdef` 均值 11.1 ms、**`lr` 均值 4.7 ms**
（但 LR 只在 23/104 帧启用，启用时单帧 **21.3 ms**）。SGR 的算术量
（`box_filter` 每像素约 10 次加减 × 2 档 + `sgr_final` 每 8 像素 36 条 + `sgr_ab` 的
逐通道查表）已经接近算法下限，剩下的只能靠换更窄的数据通道。

### 否定实验：DCT 位反转置换改原地交换 —— +3.5%（3/10），已回退

位反转是**对合**（`brev(brev(i)) == i`），于是置换只由 2-循环与不动点组成，
看起来可以"原地 `i < brev(i)` 时 `swap`"，把 4·n0 次内存操作降到 2·n0。
实测 **+3.5%（3/10 轮）**：逐元素的分支 + 变址 `swap` 输给了原来两段
**直写、无分支、可展开**的拷贝。已回退。

### 当前热点（非 PGO debuginfo 叶子采样，`--count=300` 无落盘）

`vt_forward_1d` 6.7%、`cdef_pair16x4` 5.4%、`comp_avg_rows` 5.2%、`nv12_frame` 4.8%、
`deblock_edge4` 4.6%、`run_outer` 4.4%、`cdef_quad8` 4.2%、`bzero` 3.9%、
`copy_rect_u16` 3.9%、`sgr_final_8` 3.8%、`project_frame` 3.8%、`edge8_h` 3.4%、
`sgr_ab_8` 3.2%、`mc_horiz` 2.8%、`decode_block` 2.7%、`lr·closure` 2.7%、
`read_symbol` 2.7%、`cdef_direction_x2` 2.5%。

CDEF 滤波内核的实测吞吐只有 **~0.4 op/cycle**（按指令数估应为 ~3）—— 说明它卡在
**寄存器溢出 / 载入延迟**上，而不是算术量；再往下要么重排状态（4 行 × 2 半区已占
10 个 YMM + 4 个抽头），要么换 asm 级写法。

### 否定实验：`X_BY_XPLUS1` 表末项 + `min(255)` 消分支 —— +3.8%（0/10），已回退

把表 `t[255]` 改成 256、调用方写 `table[(z as usize).min(255)]`，理论上每 8 通道少
两个分支。实测 **+3.8%（0/10 轮）**：`.min()` 带来的依赖链比"几乎总是预测对的分支"
更贵（`z` 的分布高度集中，分支预测器几乎不miss）。已回退。

### 小结（本轮四个否定）

| 改动 | 结果 |
|---|---|
| `sgr_filter` 的 68 KB 栈缓冲挪 TLS | **+4.6%（1/10）** |
| DCT 位反转置换改原地 `swap` | **+3.5%（3/10）** |
| `X_BY_XPLUS1` + `min(255)` 消分支 | **+3.8%（0/10）** |
| CDEF 亮度滤波退回"两行一组" | +1.5% 于我（即四行版更快，保留四行版） |

共同教训：**这些位置的"看似更少的工作"都输给了现状** —— LLVM 的死存储消除/分支预测
比手写"优化"更聪明。凡是"减少分支/消除初始化/减少访存"的想法，先做同 profile A/B。

### 否定实验（2026-10-24 晚，全部同 profile A/B、全部逐字节）

| 改动 | 结果 | 原因 |
|---|---|---|
| `comp_avg_rows`/`copy_rect_u16` 加软件预取下一行 | **+3.4%（1/10）** | 硬件预取器对大跨步已够好，预取指令纯加开销 |
| 去块窗口装载改指针自增（去掉逐行乘法） | **+3.8%（1/10）** | LLVM 早已做强度削减，手写版破坏了它的展开 |
| 全局 `-C target-cpu=x86-64-v3`（AVX2 基线） | **+1.6%（3/10）** | AVX2 自动向量化/编码在这个内核组合上不如 SSE2 基线 + PGO |
| `vt_forward_1d` 分派器加 `#[inline]` | **+4.9%（2/10）** | 三个变换体展开进每个调用点，代码膨胀伤 I-cache |

**结论**：经过两个多小时的连续 A/B，当前内核布局（PGO + 显式 AVX2 + 标量基线）
已经是一个相当好的局部最优；**靠"减少分支/减少乘法/减少初始化/加提示"这一类
微调已经找不到正收益了**。要再往 33 ms 走，必须换实现层级（见下）。

### 通往 33 ms 的剩余路线（按"收益/风险"排序，均未开始）

1. **u8 原生重建**（`PERF-HANDOFF` 第四十八节的分期计划）：8bit 帧全链路（重建/MC/
   去块/CDEF/LR/NV12）改 `u8` 存储。按内存受限内核（`comp_avg_rows` 5.2%、
   `nv12` 4.8%、`copy_rect` 3.9%、CDEF 克隆 2.6%、`mc_horiz` 2.8%）合计约 19%
   估算，减半访存可换 **6~9%**。⚠️ 2026-10-23 的"重建 u16 + 滤镜 u8"混合方案慢 1.2×，
   必须**连重建一起切**。
2. **反变换 16 位分档**：现在 8bit 也走 i32 8 通道（`vt_forward_1d` 6.7%）。
   dav1d 用 16 位通道 + `pmaddwd` 把每趟的通道数翻倍；需要按 dav1d 的缩放方式
   重写蝶形并重新做值域证明（工作量大、要逐位对拍）。
3. **CDEF 内核重排**：实测吞吐只有 ~0.4 op/cycle（寄存器溢出）。可尝试把 4 行 ×
   2 半区的状态拆成两次 2 行遍历（已测：当前四行版反而快 1.5%），或直接写 asm。
4. **熵解码**：`read_symbol_inner` 2.7%，`coeffs` 1.9% —— 与 dav1d 的 msac 差距在
   2 倍量级，属"每符号几十条指令"的稳定开销。

### 否定实验：`read_bool` 专用快路径 —— 语法阶段 +3.4%（1/10），已回退

思路：`read_bool` 的 CDF 是常量（`EQUIPROBABLE_CDF`），通用路径每次都要过四个诊断
开关、CDF 合法性校验、两轮符号循环、`update` 分支；专用版直接算两档、共用
`renormalize()`（从 `read_symbol_inner` 抽出的**唯一**实现）。
⚠️ 第一版把比较方向写反了（`value >= cur` 落在**低**区间 ⇒ 符号 0），第 1 帧就错；
修正后逐字节一致，但**语法阶段 +3.4%（1/10 轮）更慢** —— PGO 下通用路径已经够快，
专用版多出的 `trace_sym || record_recent` 判断与 `#[track_caller]` 参数反而更贵。
已回退。

## 七十八、2026-10-25 长会话收口：从 52.2 → 44.8 ms（逐版配对 A/B），33 ms 仍未达

### 这一轮真正落地的（每一步都有同 profile / 同二进制配对 A/B）

| 产物 | 改动 | 配对 A/B |
|---|---|---|
| `299227b9` | 簿记四刀（`mark_decoded` 按行、`stamp` 跳默认值、去块 `edge_params` 内联、CDEF 候选表缓存） | **−4.7%（10/10）** |
| `150594f7` | 运动场落槽复用 | −0.4% |
| `a0573bbb` | `intra::smooth` 8 路 SIMD、`pred_buf` 不再清零、`loop_filter_edge` 内联 | −1.1% |
| `24c5e3d2` | CDEF 抽头归并（12 乘 → 4 乘）+ `directional` 90°/180° 行快路径 | −0.9%（同 profile） |
| `d6758ec0` | 熵解码：`read_bits_at` 一次非对齐读 + `bswap`、CDF 循环去冗余边界检查（语法阶段 −2.9%） | −0.5% |
| `37a533f3` | **CDEF 方向搜索两块并行**（每 128 位半区一个 8×8） | **−2.0%（10/10）** |
| `9fcda413` | CDEF `constrain` 用 `subs_epu16 + min_epu16` | −1.5% |
| `27a811d7` | **整像素预测的整块矩形拷贝**（`copy_rect_u16`） | **−2.5%（8/10）** |

阶段实测（中位）：`tiles 23.2` + `filters 17.4`（去块 4.8 / CDEF **9.2**（原 12.2）/ LR 0–21）
+ `setup+mf+ref 3.3` + `out 0.6` ≈ 46.5 ms；**快窗口整帧 44.8–45.4 ms**。

### 这一轮被证伪的（全部已回退，全部逐字节等价）

CDEF 不克隆+补拷、NV12 显式 AVX2（逐行/整平面两版）、DCT 位反转折叠/原地交换、
运动场投影去重/逆序掩码、`sgr_filter` TLS 复用、SGR 查表消分支、软件预取、
去块指针自增、`target-cpu=v3`、变换分派器内联、`read_bool` 快路径、
**fat LTO**、DIFFWTD 掩码切片路径 —— **13 类、20 余次实验，成功率为 0**。

### 结论

微调层面（减分支/减乘法/去初始化/预取/换基线/换 LTO/加内联）**已经没有任何正收益**；
PGO + LLVM 把这套内核排到了强局部最优。**33 ms 需要换实现层级**：
u8 原生重建（预计 6~9%）、反变换 16 位分档、CDEF asm 级重排、更快的 msac —— 都是
"按 dav1d 的量级重写内核"的活，不是单轮回填能做到的。

## 七十九、2026-10-25 按第三方总结推进：EOB 稀疏反变换（同二进制 −4.0%，14/14）

### 先做的两件"量"的事

1. **AVX-512 在本机不可用**：`sysctl` 的 `leaf7_features` 报告 AVX512F/BW/VL/VNNI，
   但 Rust 的 `is_x86_feature_detected!` 对**全部 AVX-512 特性返回 false** ——
   说明系统没给 AVX-512 打开 XCR0 状态，真用会非法指令。**优先级 2 在本机作废**
   （保留 AVX2 路线）。
2. **EOB 分布**（`ME_EOB_STATS=1`，新加的诊断）：
   ```
   blks: eob==1 1295 | <=4 310 | <=16 410 | <=64 134 | >64 3
   pix : eob==1 1.62M | <=4 0.68M | <=16 1.33M | <=64 0.49M | >64 9k
   ```
   ⇒ **DC-only 占 60% 的变换块、39% 的变换像素**，而且都是 32×32/64×64 的大块。
   `EOB dc_only hit=254 miss_type=7`（97% 的 DC-only 块是 DCT_DCT）。

### 落地：DC-only 反变换快路径（`ME_NO_DC_ONLY` 可关）

* 单测 **`dc_only_dct_dct_is_constant_plane`**：15 种 tx 尺寸 × 5 个 DC 值，
  DCT_DCT 的 DC-only 反变换输出是**恒定平面**（逐元素相等）；
* 单测 **`dc_only_constant_matches_the_full_transform`**：快路径算出的常量与
  **完整二维变换**逐元素相同（含 1:2 长方块的预缩放、中间钳位、行/列 shift）；
* `transform::dc_only_constant()` 只跑两趟一维标量变换（行趟取第 0 个、列趟取第 0 个）；
* `simd::add_const_plane()` 整块加常量：**8bit 走 16 通道 i16 + min/max**（u16 存储，
  写回 32 字节），10/12bit 走 8 通道 i32。⚠️ 第一版把"打包成 u8"写进去了（平面是
  u16 存储）⇒ 哈希立刻变；第二版又漏了 `p + c` 的 i16 溢出守卫（`c = 32767` 时回绕）
  —— 两次都被 100 帧逐字节抓回。

**测量**：同二进制 A/B（`ME_NO_DC_ONLY=1` vs 默认）**−4.0%（14/14 轮，该窗口 74→71 ms）**；
跨构建 PGO **±0.0%**（重训布局噪声把它盖住了 —— 仍以同二进制为准）。
全量测试 **1056 passed / 0 failed**、100 帧逐字节；产物 md5 **`c46c3ff3d32ab6e9951a6567e738303e`**。

### 为什么没有继续做"EOB≤4/16 包围盒"

推导后确认：二维变换的**列趟省不掉**（行趟输出会把非零扩散到所有列），
只能省"零系数行"的行趟；而标量行趟比 8 行并行的 SIMD 行趟贵 ~8 倍 ⇒
只有 `max_r ≤ 3` 才划算，收益远小于 DC-only。**不做**。

## 八十、2026-10-25 第二轮（按第三方总结）：SGR 收尾向量化 + `a2` 查表 gather

1. **SGR 收尾投影向量化**（`simd::sgr_proj_row8<F0,F1>`）：`w1·u + w0·f0 + w2·f1` →
   `Round2` → `Clip1` → 写 u16，一行 8 像素一次算完，`r0/r1 是否为 0` 用 const 泛型
   消掉分支；直接落 `set_row_i16`（原来先写 i32 行再整行转 u16）。
2. **`sgr_ab_8` 的 `a2` 查表改 AVX2 gather**：`X_BY_XPLUS1` 的末项改成 256，
   于是 `min(max(z,0),255)` 一次 `vpgatherdd` 就够；负值（i32 溢出档）仍标量兜底。
   `ME_NO_SGR_GATHER` / `ME_NO_SGR_PROJ` 可回退，两条路逐字节相同。

**测量**：同二进制 A/B（两个开关同时开）**−0.2%（6/12）** —— 收益小是因为
SGR 只占**平均帧**约 5%（它集中在 LR 帧上）。逐字节一致、全量 **1056 / 0**。
产物 md5 **`7e6d8fe774cba461cb9df446efa3d637`**。

### 这一轮量到的 LR 真相（`ME_LR_STATS=1` + 逐帧 PHASE）

* SGR 内部：**box_filter 两档 45% + 50%，收尾投影只占 5%**（所以投影优化只值 0.2%）；
* 逐帧分布（100 帧，PHASE 合计）：
  ```
  全部      p50 39.1  p90 84.0  p99 151.5  max 158.4
  非 LR 帧  p50 26.5  p90 40.0            ← 已经 < 33 ms
  LR 帧     p50 54.6  p90 101.0           ← 平均被这批拖上去
  ```
  **LR 帧（22%）是尾部的唯一来源**；SGR 的算术量集中在 box_filter 的
  "竖直滑动 + 前缀和 + 逐 8 列 a2 计算"，这是下一步唯一值得动的点。

## 八十一、2026-10-25 第三轮：SGR 末段 3×3 按权重档单态化（同二进制 −2.4%，15/20）

### 观察

上一轮量到的 SGR 内部构成（`sgr_final_8` ≈ 46%、`sgr_ab_8` ≈ 36%、前缀和 11%、竖直 7%）
里，`sgr_final_8` 是"每 8 列 × 两个缓冲 × 6~9 次 `vpmulld`"——而**权重只有三种固定图案**：

| 图案 | 条件 | 3×3 权重 |
|---|---|---|
| `PAT=0` | `pass == 0`、`i` 偶 | 行 0 与行 2 各 `[5,6,5]`，行 1 全 0 |
| `PAT=1` | `pass == 0`、`i` 奇 | 只有行 1 是 `[5,6,5]` |
| `PAT=2` | `pass == 1` | 九格全 4，四角 3 |

### 改法（`simd::sgr_final_8_pat<PAT, SHIFT>`）

三种图案都能化成"**先求和再缩放**"，把逐项乘法全消掉：

* `PAT=1`：`5Σ + m`（`m` = 中列）—— 3 次 load、4 add + 1 shift；
* `PAT=0`：`5Σ₆ + m₀ + m₂` —— 6 次 load、9 add + 1 shift；
* `PAT=2`：`4Σ₉ − Σ角` —— 9 次 load、13 add/sub + 1 shift。

**等价性靠模 2³² 环运算**：`b_buf` 的 `p*s_factor` 会溢出 i32（既有语义，见
`simd.rs` 的 `sgr_ab_8` 注释），项可以是任意 i32；`5Σ + m` 对应 `5x + 6m = 5(x+m) + m`、
`4Σ − 角` 对应 `4x − 3c`，在模 2³² 下逐位相同。`SHIFT`（8 / 9）也做成常量
⇒ 用立即数 `vpsrad`，省掉 `vpsravd` 的广播。`ME_NO_SGR_FINAL_PAT=1` 退回按权重表
逐项乘的通用版，**两条路逐字节相同**（24 帧对拍）。

### 测量

* **同二进制 A/B：−2.4%（配对中位 0.9757，15/20 轮）**；
* `ME_LR_STATS=1`（100 帧累计）：`box0 131.2 → 94.2 ms`、`box1 161.9 → 111.6 ms`（−29%/−31%）；
  折算每 LR 帧省 ~9 ms、整帧平均 ~0.9 ms；
* ⚠️ **第一遍只跑 12 轮，得到 +0.1%（5/12）** —— 同一对二进制再跑 20 轮就是 −2.4%（15/20）。
  **12 轮不够**：窗口噪声能吃掉 2% 级真收益，判定"无效"前先加轮数。
* 100 帧逐字节、全量 **1056 passed / 0 failed**；
* 跨构建 PGO `759488d0…` vs 上一版 `7e6d8fe7…`：−1.1%（7/10，跨构建噪声内，方向一致）。

产物 md5 **`759488d0f55a5b3e53981769834e85ef`**（已回填 `extra/darwin/x64/engine/zuvrust`）。

### 下一步（证据更新）

SGR 内部占比在本次之后约为 **`sgr_ab_8` ~50% ≫ 前缀和 ~15% ≫ 竖直 ~10% ≫ 收尾 ~30%**；
`box_filter` 仍是 LR 帧的唯一大头。其余候选按第三方清单：缓冲式 msac（语法阶段）、
CDEF 寄存器级重排（CDEF 仍 ~9 ms/帧）。老 LR 帧 p50 54.6 的构成里 SGR ~29 ms，本次 −9 ms。

## 八十二、2026-10-25 第四轮：CDEF 强度单态化 + SGR 前缀和向量化（同二进制 −1.6%，15/20）

### 先采样再选刀

用 `-Cdebuginfo=1` + 同一份 PGO profile 另建一个二进制跑 `sample`：叶子热点已经**非常平**
（最大 2~5%）：`comp_avg_rows` ~5%、去块（`run_outer`+`edge4`+`edge8_h`）~10%、
CDEF（`pair16x4`+`quad8`+方向）~12%、MC 一族 ~6%、`read_symbol_inner` ~1%、SGR ~5%。
⇒ 按排名挑"再快一点的 SIMD"已经没有油水；这一轮改挑**整段可省的恒等式**。

### 刀一：CDEF 按"哪一档强度为 0"单态化（`ME_NO_CDEF_STR_SPEC` 可关）

* `cdef.rs` 里 `pri_str` 在**局部方差为 0** 时被清零（本素材 78% 的亮度 8×8 块），
  而只要次档非 0，整块仍要滤波；
* 约束量 `t = subs_epu16(thr, mag >> adj)` 在 `thr == 0` 时**恒为 0** ⇒ 该档的 4（或 8）个
  候选**只影响最后的 min/max 钳位**，`sub → abs → sra → subs → min → sign → add`
  整条链可以整段删掉；
* 实现：`cdef_pair16x4_impl<const PRI: bool, const SEC: bool>` + 同名分派器
  （按 `pri_thr`/`sec_thr` 是否为 0 选三个实例之一，两档同为 0 时调用方本来就不滤波）；
* ⚠️ **顺手抓到一个"测试从未运行"的坑**：`cdef_pair16x4_matches_pair16_bit_for_bit` 的
  `#[cfg]`/`#[test]` 与 `fn` 之间夹了一段文档注释（后面还跟着另一个 `#[test]`），
  属性于是粘到了下一个函数 `mc_vert2_avg_u16_matches_vert_then_average` 上 ——
  这条四行内核的对拍**一直没被跑过**。已把属性贴回原函数，并补了 `pri=0` / `sec=0`
  两组强度（都对着两行版逐位对拍）。
* 实测 **−0.6%（12/20）**：比预期小 —— 这个内核是**载入受限**（min/max 仍要载入全部 12 个候选），
  删掉的只是 ALU。**"少算"不等于"少等"。**

### 刀二：SGR 前缀和向量化（`ME_NO_SGR_PREFIX` 可关）

* `box_filter` 每行两条 `pre[k+1] = pre[k] + cs[k]`（`cols = w + 2 + 2r ≈ 70`）都是**串行链**；
  换成 8 路前缀扫描 `simd::sgr_prefix8`：lane 内两次 `slli+add`，再把低 128 位的末元素
  广播到高 128 位（`permute2x128::<0x08>` + `shuffle_epi32::<0xFF>`），两条数组一起做、
  尾巴标量收。整数加法可结合 ⇒ 逐位相同（单测 `sgr_prefix8_matches_scalar`，
  覆盖 0/1/7/8/9/16/17/63/70/72 列与负值；
  ⚠️ 第一版把 `permute2x128` 的立即数写反（`0x80`），进位加到了低半区 —— 单测当场抓住）；
* 两个 `[i32; 74]` 的行缓冲**提到行循环外**（原来每行重新零初始化 ~600 B，而 `[1..=cols]`
  每行都会被写满、`[0]` 恒为 0）；
* `ME_LR_STATS=1` 实测（10 个 LR 帧累计，两轮交替）：
  `box0 106.8/103.8 → 83.7/86.6`、`box1 122.5/123.8 → 96.7/99.7` ⇒ **box_filter −21%**，
  每 LR 帧约省 **5 ms**；整帧平均只摊到 ~0.5 ms（本窗口 LR 帧只占 10%）。

### 合计与验收

* 同二进制 A/B（两个开关同时关 vs 默认）：**−1.6%（配对中位 0.9842，15/20）**；
* 100 帧逐字节、全量 **1057 passed / 0 failed**；
* 产物 md5 **`92bb57160d99adf5e58b47594532f396`**（已回填 `extra/darwin/x64/engine/zuvrust`）。

### 这一轮的判断

叶子排名已经不能指导选点（全部 ≤5%）；现在赚的是"**某一档参数/状态让整段计算变成恒等式**"
的结构化剪枝 —— CDEF 的强度档、SGR 的权重档（上一轮）都属于这一类，而且都能写成
const 泛型在编译期消掉。SGR 里还剩 `sgr_ab_8`（a2/b2，gather + i32 回绕语义）
与竖直滑动；Wiener 的 `pmaddwd` 化是清单里下一项。

## 八十三、2026-10-25 第五轮：SGR 的 `a2` 用浮点等价式取代 gather（box_filter −14%）

### 观察

`box_filter` 每 8 列要算一次
`a2 = ((z << 8) + z/2) / (z + 1)`（`z ∈ [1,254]`），原实现查 256 项常量表
（`X_BY_XPLUS1`），`vpgatherdd` 在本机约 **12~20 uop、20 拍延迟**；而 `z` 的分布很散，
分支预测帮不上忙。它是 SGR 第二轮之后 `box_filter` 里最大的一块。

### 代数等价式

`256z/(z+1) = 256 − 256/(z+1)`；它恰好落在 `x.5` 上当且仅当 `512/(z+1)` 是奇数，
而 `z + 1 ∈ [2,255]` 内没有这样的值（512 的奇商只有 `z+1 = 512`）⇒
**`((z<<8) + z/2)/(z+1)` 与 `round(256z/(z+1))`（就近取偶）逐值相同**。
Python 全量核对 1..254 无差异；Rust 单测 `sgr_a2_vec_matches_integer_formula`
对 `0..=255` + `>=255`/`i32::MAX` 全量对拍。

### 实现（`simd::sgr_a2_vec`，`ME_NO_SGR_FLOAT` 可回退到 gather）

* `clamp(z,0,255)` → `cvtepi32_ps` → `256·z/(z+1)`（一次 `vdivps`）→ `cvtps2dq`
  → `max(·,1)`（`z = 0` 给 1）→ `z ≥ 255` 用 blend 盖成 256 —— 与常量表逐值相同；
* ⚠️ **负 `z` 仍走原来的标量兜底**（`p·s_factor` 在 i32 里溢出那一档，逐位保持）；
* 单测里 `0..=255` 与越界正数都覆盖，负值的语义由调用方保证（先夹 0 再覆盖）。

### 测量

* `ME_LR_STATS=1`（10 个 LR 帧累计 ×2 轮，交替）：
  `box0 91.4/91.8 → 76.9/81.1`、`box1 106.5/106.1 → 89.1/92.6` ⇒ **box_filter −14%**，
  每 LR 帧再省 ~2.9 ms；
* 同二进制 20 轮配对 A/B：**−1.1%（0.9893，13/20）**；
* 100 帧逐字节、全量 **1058** 单测通过；产物 md5 **`803f5c3ed36c13e16068d91633d3672c`**。

### 教训

`vpgatherdd` 不该用在"小整数除法"这种**值域有限**的场景：能证明取值集合/表达式
等价时，代数等价式（这里是浮点除 + 就近取偶）比查表快得多。同类的还有上一轮的
3×3 权重档（`5Σ+m`/`4Σ−角`）与前缀和（进位广播）——**"有限值域 + 环等价"是本阶段的主线**。

### 附：CDEF 内核微基准（为下一刀定位）

`ME_CDEF_STATS=1` 拆开 CDEF 的构成（满载窗口）：`clone ≈ 1.1 ms/帧`、`total ≈ 13~23 ms/帧`
（冷帧更高），内核 + 方向搜索占 **92%**；`bench_cdef_kernels`（`#[ignore]` 微基准）
给出 **212 ns / 8×8 块**（i16 四行 208.6、两行 212.4、旧 i32 一行 464.9）⇒
`31,654 块/帧 × 210 ns ≈ 6.6 ms/帧`，与相位计时一致。
反汇编看 `<true,true>` 实例：循环体 ~266 条向量指令、17 处栈溢出 —— 指令数估算
（~280 uop/32 px）对应理论 ~2.5 cycle/px，实测 ~6~8 cycle/px ⇒ **IPC ≈ 1，仍是延迟/溢出受限**。
下一刀应在 `bench_cdef_kernels` 上迭代（不用 PGO，1 秒一轮），而不是盲改。

## 八十四、2026-10-25 频率校准：本机在满载窗口只有 **~1.23 GHz**（解释所有绝对毫秒）

用一条**单周期依赖链**（内联汇编 `add $1`，不会被编译器折叠）标定当前 CPU 频率：

```
0.807 / 0.819 ns per 1-cycle-chain ⇒ 1.24 / 1.22 GHz   （load average 4.2~14）
```

i5-1030NG7 的标称基频 1.1 GHz、单核睿频 3.5 GHz。也就是说**这段时间的所有绝对毫秒数
都是在 1.2 GHz 档测出来的**；同一份代码在 3.5 GHz 时理论上只有这里的 **~1/2.9**。

这解释了全部"自相矛盾"的现象：

* 历史"快窗口 44.8 ms" ≈ 当时的频率更高（约 1.9~2.4 GHz）；现在的 48~50 ms 不是回归；
* `bench_cdef_kernels` 的 208~213 ns/块，在 1.23 GHz 下是 **~256 拍/64 像素 = 4.0 拍/像素**，
  与指令数估算（~3.1 拍/像素）只差 30% ⇒ **CDEF 内核没有 4 倍优化空间**，
  之前用"2.5 GHz 假设"算出的 8 拍/像素是错的；
* 33 ms 目标因此必须**按频率口径看**：按 1.23 GHz 折算，33 ms 对应
  3.5 GHz 下的 ~11.6 ms —— 现在的代码在睿频下约 **17~18 ms/帧**。
  ⇒ **"没达到 33 ms" 在当前是"机器被 IDE/系统占满 + 无风扇降频"的复合结果**，
  单纯靠代码优化的余量并没有账面看起来那么大。

**纪律更新**：以后报绝对毫秒必须同时报标定频率（`/tmp/freq3` 那 6 行程序即可），
否则跨窗口的数字没有可比性；判断改动只看**同窗口配对 A/B**（这条不变）。

### 这一轮否定的实验（已回退）

CDEF 四行内核"少占寄存器"（抽头不提前广播，省 4 个 YMM；`adjust` 用 `srli 15` 代替
`and(…, one)`，再省 1 个）：微基准 **208.3 vs 212.8 ns/块**（噪声内，0/2 轮更快），
反汇编里溢出也没减少。⇒ 寄存器压力不是瓶颈，**已回退**。

## 八十五、2026-10-25 第六轮：反变换跳过全零的 8 行组（同二进制 −3.2%，18/20）

### 观察

`inverse_transform_2d_into` 的 i32 行趟是按 **8 行一组**推进的；而真素材里 **60% 的变换块
`eob == 1`**（这批已被 DC-only 常量平面快路径接走），**剩下的小 EOB 块大多是 32×32/64×64**
⇒ 系数只落在最上面一两行，底下的 8 行组**整组都是零**。行趟输入全零 ⇒ 输出也全零。

换口径说：EOB 分布（第八十节）里 `eob ∈ (1,4]` 的 310 块 + `≤16` 的 410 块，像素数
0.68M + 1.33M —— 其中大量是 32×32/64×64 的大块，**它们的行趟有 3/4 ~ 7/8 是纯零**。

### 改法（`ME_NO_TX_SPARSE_ROWS` 可关）

* 进 i32 路径前，从**最后一行往上**扫 `t`（反量化后的系数格），找最后一个非零行；
  `h_eff = ceil((last+1)/8)·8`（内核要求 8 的倍数，且至少 8）。扫到就停 ⇒
  开销与非零区高度成正比，稀疏块几乎免费。
* 只让行趟跑 `h_eff` 行；**被跳过的行组必须清零**（列趟仍要读它们）：`res32[h_eff·w..].fill(0)`。
* `h ≤ 8` 或 `h_eff == h` 时行为与旧路径**完全一致**；扫描只看前 32 行（规范：系数行 ≥32 恒零）。
* 逐字节正确性：全零行组的行趟输出恒为 0（所有正变换的零输入都是零输出），
  清零与"真的跑一遍"逐位相同 —— 100 帧逐字节已验证。

### 测量

* 同二进制 20 轮配对 A/B：**−3.2%（配对中位 0.9680，18/20 轮）** —— 本轮最大的一笔；
* 100 帧逐字节、全量 **1058** 单测通过；产物 md5 **`38ea0b2e125498121d974a57b7979eb4`**。

### 判断

这条能成立的关键是"**行趟按 8 行组推进**"这个实现事实：它让"跳过零行"从
"退回标量 8 倍慢"（第八十节的结论）变成"整组跳过 + 一次 memset"。⇒ 优化被推翻时，
要记得**推翻的是当时的实现形状**，换成组粒度就能重新成立。

## 八十六、2026-10-25 第七轮：NV12 亮度收窄用 `packus + permute4x64`（同二进制 −1.0%，12/20）

### 背景（推翻旧结论的第三次）

上一版注释写着"两次显式 AVX2（逐行 / 整平面一次调用）都更慢（+1.9% / +0.4%）"。
但带 debuginfo 的采样里，**`av1.rs:1182`（亮度收窄循环）仍是前 10 的叶子行**
（约 2.8%），反汇编看它只有 **SSE 128 位**（`psrlw`/`packuswb`，一次 8 像素）。

### 改法（`simd::narrow_u16_u8_rows`，`ME_NO_NV12_LUMA_SIMD` 可关）

* 一次 32 像素：两条 256 位载入 →（10/12bit 时 `srli`）→ `packus_epi16` →
  **`permute4x64::<0xD8>`** 修正 128 位 lane 的打包顺序 → 一条 32 字节存；
* 尾巴 8 像素走 SSE、再不足走标量；整个平面**一次调用**（行循环在内核里）。
* 与旧的两版差别：旧版是 u16→i32→pack（多一倍通道和一条打包），这里是
  **直接在 16 位域打包**，每 32 像素只要 6 条向量指令。

### 测量

* 同二进制 20 轮配对 A/B：**−1.0%（0.9900，12/20 轮）** —— 比预期小（瓶颈在内存带宽：
  读 4.2 MB、写 3.1 MB），但方向为正、逐字节一致；
* 100 帧逐字节、全量 **1058** 单测通过；产物 md5 **`9c680b56eb92f1e1affa9f224c26b30e`**。

### 记录

这是第三次"显式 AVX2 vs 自动向量化"的对照：**前两次负、这次微正**，差别在实现形状
（16 位域打包 vs 32 位域打包）。⇒ 旧的否定结论要连着"当时那版的内核形状"一起记，
不能当成"这个方向永远不行"。
