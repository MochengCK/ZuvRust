# tools/av1ref —— 参考解码器对拍

自研 AV1 软解（`crates/me-decode/src/video/`）的**唯一硬判据**：把我们的输出与
**dav1d** 的输出逐像素比。`tools/av1probe/` 那批手造参照流能钉住语法解析，
但回答不了"重建是否正确"；真实素材又太复杂。这里补上中间那一层。

## 两条腿

| 角色 | 工具 | 怎么来的 |
|---|---|---|
| **参考解码器**（oracle） | dav1d（经 PyAV 的 FFmpeg） | `pip install av` —— wheel 内置换 libdav1d 的 FFmpeg，**不用 brew/ffmpeg** |
| **受控编码器**（造流） | aomenc（libaom） | 源码 `https://storage.googleapis.com/aom-releases/libaom-<ver>.tar.gz`，本地 cmake 编译 |
| （备选编码器） | libsvtav1（经 PyAV） | `pip install av` 就有；**但它只支持 64 超块、且不开 loop restoration** |

**为什么两个编码器都要**：它们的能力互补 —— 这才是能逐项隔离变量的前提。

| 特性 | libaom | SVT-AV1 | 项目里谁缺 |
|---|---|---|---|
| 128×128 超块 | ✅（默认） | ❌ 只支持 64 | —— |
| loop restoration | ✅ 默认开 | ❌ 不支持 | —— |
| **film grain synthesis** | ✅ 默认开（`--denoise-noise-level` / `--film-grain-test` / `--tune-content=film`） | ❌ 不开 | ✅ 我们已实现（§7.18，2026-10-01） |
| **超分上采样**（§7.16） | ✅（`--superres-mode` / `--superres-denominator`） | ❌ | ✅ 我们已实现（2026-10-01） |
| **帧内-帧间混合预测**（`IsInterIntra`，§7.11.2） | ✅ 默认开，但 `--cpu-used>=3` 会**静默关掉**（见下） | ❌ | ✅ 我们已实现（2026-10-01） |
| **屏幕内容·调色板**（palette，§5.11.18/19） | ✅ 但只在 `--tune-content=screen` 下开 | ❌ | ✅ 我们已实现（2026-10-01） |
| **屏幕内容·帧内块拷贝**（`use_intrabc`，§5.11.5） | ✅ 同上，且 `--cpu-used>=3` 会**静默关掉**（见下） | ❌ | ✅ 我们已实现（2026-10-01） |
| 多帧 inter / 复合 / warp | ✅ | ✅ | —— |

## 编译 aomenc（一次性）

```sh
curl -sSL -o /tmp/libaom.tar.gz https://storage.googleapis.com/aom-releases/libaom-3.9.0.tar.gz
tar xzf /tmp/libaom.tar.gz -C /tmp
cd /tmp/libaom-3.9.0 && mkdir build && cd build
cmake .. -G "Unix Makefiles" -DAOM_TARGET_CPU=generic \
  -DCONFIG_AV1_ENCODER=1 -DENABLE_EXAMPLES=1 -DENABLE_TOOLS=1 \
  -DENABLE_DOCS=0 -DENABLE_TESTS=0
make -j8 aomenc              # 产物 build/aomenc
```

**两个坑**：

1. **`AOM_TARGET_CPU=generic` 是必须的**：本机 nasm 是 3.x，而 aom 3.9 只认 2.x 的
   multipass 语法，配 CMake 会直接报 `Unsupported nasm: multipass optimization not
   supported.`。关掉汇编即可（generic 构建慢几倍，但功能完整）。
2. **`--cpu-used` 不能太高**：`--cpu-used=8` 时 libaom **会跳过 loop restoration**
   （`--enable-restoration=1` 静默失效，`lr1`/`lr0` 两份码流会**字节完全相同** ——
   这就是判据）。要验 LR 必须 `--cpu-used<=4`。

## 用法

```sh
# 造受控流（可逐项开关工具）+ 用 dav1d 出参考
python3 make_cases.py --out=/tmp/refs

# 胶片颗粒那批用例（--film-grain-test=N，另见 make_cases.py 里的 grain_cases）
python3 make_cases.py --out=/tmp/grefs --grain

# 超分上采样那批用例（§7.16：--superres-mode / --superres-denominator）
python3 make_cases.py --out=/tmp/srefs --superres

# 帧内-帧间混合预测那批用例（§7.11.2 的 IsInterIntra）
python3 make_cases.py --out=/tmp/iirefs --interintra

# 屏幕内容·调色板那批用例（§5.11.18 / §5.11.19 / §7.11.4，须 --tune-content=screen）
python3 make_cases.py --out=/tmp/prefs --palette

# 屏幕内容·帧内块拷贝那批用例（§5.11.5 的 use_intrabc，须 --cpu-used=2）
python3 make_cases.py --out=/tmp/ibcrefs --intrabc

# 我们的引擎解码
zuvrust decode /tmp/refs/case.mp4 --count 4 --out=/tmp/out_case --json

# 逐像素对拍（Y/U/V 三个平面各报一行）
python3 refcmp.py /tmp/refs/case.mp4 --dir=/tmp/out_case --n=4
```

**`--grain` 开关的含义（2026-10-01 起反过来了）**：颗粒合成我们已经做了，所以
**默认就是 `--grain=1`，比的是含颗粒的最终像素**。`--grain=0`（给 dav1d 加
`filmgrain=0`）现在只剩一个用途：**反证颗粒确实生效** ——

```sh
refcmp.py aom.mp4 --dir=/tmp/out            # 默认 grain=1：应逐像素一致
refcmp.py aom.mp4 --dir=/tmp/out --grain=0  # 摘掉颗粒：应出现大面积差异
```

如果 `--grain=0` 下也一致，说明这条流**根本没有颗粒**，拿它验颗粒是白验
（libaom 只有 `--denoise-noise-level` / `--film-grain-test` / `--tune-content=film`
三种方式才会真的写颗粒参数）。

## 实测结论（2026-10-01）

| 用例 | 结果 |
|---|---|
| libaom 640×360 帧内（128 超块、LR、CDEF） | **逐像素一致**（`--grain=0`） |
| libaom 1920×1080 帧内 | **逐像素一致**（`--grain=0`） |
| SVT-AV1 512×512 六帧（I + P，含复合 / OBMC / warp） | **逐像素一致**（`--grain=0`） |
| SVT-AV1 1920×1080 六帧 | **逐像素一致**（`--grain=0`） |
| libaom `--film-grain-test=1..16`（640×360 × 4 帧，Y/U/V） | **逐像素一致**（`--grain=1`） |
| libaom `--denoise-noise-level=20`（640×360，含 10 位 `--bit-depth=10`） | **逐像素一致**（`--grain=1`） |
| libaom alt-ref 流（40 帧，其中 13 个 `show_existing_frame`） | **逐像素一致**（`--grain=1`） |
| libaom 超分 `--superres-mode=1`：648→432 / 648→576 / 1920→1280 / 640→320（`coded 宽是 8 的倍数`） | **逐像素一致**（`--grain=1`） |
| libaom 超分 640→427（`coded 宽不是 8 的倍数`） | Y 差 ~0.7%（max 3）、U/V 一致 —— 见下方"帧右边界"坑 |
| libaom `--grain=0` 对照 | 差异 88~94%、max 24 —— 反证颗粒被摘掉 |
| libaom **interintra** 1280×720 × 6 帧（`--cpu-used=2`） | **逐像素一致**（`--grain=1`） |
| 同上，`AV1_NO_INTERINTRA=1` 摘掉 interintra | 帧 1~5 亮度差 0.08% → 1.47%（max 100），色度一致 —— 反证这条流真的用了 interintra |
| libaom **palette** 屏幕内容流（640×360 × 1 / 1280×720 × 1 与 × 6 帧，`--tune-content=screen`） | **逐像素一致**（`--grain=1`，Y/U/V 三平面） |
| libaom **intrabc** 屏幕内容流（640×360 × 1 / 1280×720 × 1 与 × 4 帧，`--cpu-used=2`） | **逐像素一致**（`--grain=1`，Y/U/V 三平面） |
| 同上，`AV1_NO_INTRABC=1` 摘掉块拷贝 | 色度差 **81.28%**（640）/ **81.52%**（720）—— 反证这两条流真的用了 intrabc |

⇒ **解码器本体（CDEF / LR / 分区 / 变换 / inter 预测 / 复合 / warp / interintra / 颗粒合成 / 屏幕内容流的 palette 与 intrabc）全对。**
16 组 `film_grain_test_vectors` 一起覆盖了：`ar_coeff_lag` 0/2/3（自回归滤波）、
`num_y_points` 1~14、`num_uv_points` 0~9、`chroma_scaling_from_luma=1`、
`overlap_flag` 0/1、`clip_to_restricted_range` 0/1、`grain_scale_shift` 0/1/2、
`cb,cr_mult/luma_mult/offset` ≠ 默认值，以及 **`update_grain=0`**
（第 2 帧起 `ref_idx` 0→1 从参考槽继承参数，见 t05/t11）。

## 造颗粒流的两个坑（都踩过）

1. **`--tune-content=film` 会吞掉 `--film-grain-test`**。`av1/encoder/encoder.c`
   在打包前有这么一段：`if (content == AOM_CONTENT_FILM) set_grain_syn_params(cm);`
   —— 它把 `cm->film_grain_params` **整个覆盖**成 `ypts=1 / uvpts=(1,1) /
   ar_coeff_lag=0` 的最小形态。所以只要带了这个开关，16 组测试向量就全变成同一份
   参数（表面现象：16 条流的参数、甚至码流长度都一模一样）。
2. **`update_grain` 只写在 inter 帧里**（规范 §5.9.30：`if (frame_type ==
   INTER_FRAME) update_grain`，关键帧一律按 1 处理）。所以拿
   `--kf-max-dist=1`（全关键帧）造流，永远验不到 `load_grain_params(ref_idx)`
   那条路 —— 必须让第 2 帧起是 inter 帧，并且 `--film-grain-test` 选到
   `update_parameters = 0` 的那几组（5 / 9 / 11）。

## 造超分流的两个坑（都踩过）

1. **关键帧也要压**：`--superres-kf-denominator` 只管关键帧 —— 只给
   `--superres-denominator` 的话关键帧可能不超分。而 `--kf-max-dist=1`（全关键帧）
   造流时，**唯一能单帧对拍的就是关键帧**，所以两个开关必须一起给。
2. **宽度要选 `coded 宽（FrameWidth）是 8 的倍数`的尺寸**，否则上采样本身的表现会被
   **帧右边界超块伸出区**的已知问题盖住。`FrameWidth = UpscaledWidth * 8 / denom`，
   例如 `640 + denom 12 → 427`（不是 8 的倍数）就会中招。所以 `superres_cases` 特意用
   648 / 1920：`648→432`、`648→576`、`1920→1280` 都是 8 的倍数，只有这几条才是在
   单纯验上采样。反过来，`640_sr_fix12` / `640_sr_nolr` 这两条是**故意留下来当反例的**
   —— 它们编出来就是为了证明"残差来自帧右边界、与超分无关"（隔离实验见 README 正文
   的"已知问题：帧右边界超块伸出区"）。

## 造 interintra 流的三个坑（都踩过）

1. **`--cpu-used >= 3` 会让 libaom 静默关掉 interintra**。`av1/encoder/speed_features.c`
   在 `set_good_speed_feature_framesize_dependent` / `..._framesize_independent` 里各有一段
   `enable_interintra_compound &= (disable_interintra_wedge_var_thresh != UINT_MAX)`；
   实测 1280×720 下 cpu-used 0/1/2 写 1、**3/4/6 写 0**。写成 0 的流里一个 interintra 块
   都不可能有 —— 拿它验 interintra 就是白验（这轮先用 cpu-used 4/5/6 试了几十条，
   怎么调内容都不命中，最后是**把序列头那一位读出来**才定位到的）。
   本条流必须 `--cpu-used=2`。
2. **`min(w, h) >= 480`**：同一个函数里低于 480p 的分支直接把阈值设成 `UINT_MAX`。
   640×360 无论 cpu-used 取多少都是关的。
3. **高度取 4 的倍数**：`MiRows = 2 * ((FrameHeight + 7) >> 3)` 是向上取整的，540 会多出
   4 像素的"帧底超块伸出区"（与超分那节说的帧右边界是同一类已知问题），会把结果盖住。
   720 / 1080 都整除，选它们。

判据用 **`AV1_NO_INTERINTRA`**（与 `AV1_NO_OBMC` 同一套路）：置上之后 interintra 块
**退化成纯帧间预测**（不炸、不报错），于是"这条流到底用没用 interintra"变成一次
**差异对拍**：正着跑应当逐像素一致，摘掉之后应当大面积开始差。只有"摘掉也一致"
才说明这条流没用到 interintra —— 光看正着跑一致是分不清"没用"与"用对了"的。
这条流实测摘掉后帧 1~5 亮度差 0.08% → 1.47%，且**楔形与 smooth 两条路都走到了**
（块级计数：楔形 21 次、smooth 15 次调用，按每块 3 个平面算约 7 + 5 个块）。

## 造屏幕内容流的坑（palette / intrabc，都踩过）

**编码侧**：

1. **`--tune-content=screen` 是两个工具的总开关**。libaom 只在
   `AOM_CONTENT_SCREEN` 下把 `allow_screen_content_tools` 置 1
   （`av1/encoder/encoder.c` 的 `av1_set_screen_content_options`）；置 0 的序列头里
   `palette_mode_info()` / `use_intrabc` 根本不会被调用 —— 正着跑一致也证明不了任何事。
2. **验 palette 要 `--enable-intrabc=0`**（`palette_cases` 就是这么造的）：屏幕内容下
   libaom 默认把 intrabc 也开上，"重复且多色"的块会被它抢走，混在一起分不清残差该记在谁头上。
   `intrabc_cases` 反过来**两个都开**：少色块归 palette，重复多色块归 intrabc，同场竞争。
3. **`--cpu-used >= 3` 又静默关掉 intrabc**（与 interintra 同一类坑）：
   `speed_features.c` 里 `if (speed >= 3) sf->mv_sf.use_intrabc = 0;`，
   `rd_pick_intrabc_mode_sb` 一进门就返回 `INT64_MAX`。本轮的 640/720 三条用例一开始
   就是这么白验的（正着跑一致、`AV1_NO_INTRABC=1` **也**一致 —— 后者才是判据）。
4. **只有帧内帧才可能开 `allow_intrabc`**（规范写在 `if (frame_is_intra)` 里）⇒
   intrabc 用例必须**全关键帧**（`--kf-max-dist=1`），否则除第 0 帧外走不到这条语法。
5. **素材**：palette 要"全局 ≤8 色"（`screen_content`，色块边界落在 8 的倍数上，
   色度也要有颜色）；intrabc 要"周期 ≥256px 的重复内容，且重复块本身多色"
   （`intrabc_content`，下半屏确定性噪声）—— 纯色会被方向预测白拿、少色会被 palette 抢走。

**解码侧**（本轮 intrabc 对拍暴露的两个错，都是"看着对、其实错一格"）：

- **帧间块（含 intrabc 拷贝块）也要在 `transform_block` 结尾标 `BlockDecoded`**。
  少标之后相邻块的 `haveAboveRight` / `haveBelowLeft` 推导出错，640 用例差 5.16%。
- **intrabc 的色度尺寸按 `plane_residual_size` 的 4×4 粒度算**（`num4x4W*4`），
  **不是**"亮度尺寸 >> sub"：intrabc 块在 `RefFrames` 网格里被记成 `INTRA_FRAME`
  ⇒ 走 `compute_prediction` 的 `someUseIntra` 那一支；且 `HasChroma=0` 的块
  （4:2:0 下 4×4 / 8×4 / 4×8 落在 8×8 单元"上/左半"）**不做色度**，整个单元的色度
  由右下角那个 4×4 块一起预测。按 `>> sub` 只会写左上 2×2，其余像素停在 128，
  还会被后续 intrabc 拷贝沿参考链放大（720 用例的 7 处色度洞就是这么来的）。

## 判据的陷阱（踩过的）

- **两份"不同参数"的码流可能字节相同** ⇒ 参数没生效。用 `md5` 比码流，
  别信 `--help` 里写了就以为生效（`--sb-size` 在 SVT-AV1 上就是空转）。
- **`--enable-restoration` / `--enable-*` 在 aomenc 上要为每份流单独确认生效**
  （见上面 cpu-used 的坑）。
- **容器/解码器选项**：要关 dav1d 的 grain，必须用 `av.CodecContext.create("libdav1d","r")`
  再设 `options` —— 容器自带的上下文不让你设。
- **别用 `av.codec.Codec("libdav1d","r")`**：某些脚本里会抛 `UnknownCodecError`；
  用 `av.CodecContext.create` 或容器的默认解码器。
- **headless Chrome 不能当 oracle**：媒体管线不推进（`<video>` 的 `seeked` 永不触发），
  `--virtual-time-budget` 也救不了。
- **对比要用同一 dtype**：`uint8` 与 `int32` 算出的 md5 当然不同，别据此下结论。

## 剩下的像素差怎么查（2026-10-01 建的工具）

`refcmp.py` 给出"差多少"，但回答不了"该查哪一层"。补两个：

| 脚本 | 作用 |
|---|---|
| `diffmap.py` | **差异定性**：差值直方图 / 孤立点比例 / 与 4·8·16·32·64 网格的关系（对比整图基线）/ 集中在哪些 128×128 超块 / `--probe` 把头几个差异点的邻域摊开打 |
| `motion_cases.py` | **补"运动"这一档用例**：`--kind=shift` 亚像素平移、`--kind=mosaic` 每块运动各异 |

### 真素材（B 站 ASMR 1080p）那 0.2% 的定性结论

```
diffmap.py /tmp/a.mp4 --dir=<decode --out> --frames=1,10,42 --probe
```

- 帧 0（**纯帧内**）**逐像素一致** ⇒ 帧内预测 / 反变换 / 去块 / CDEF / LR 都对；
- 差异 **100% 孤立单点**（4 邻域内没有第二个差异点）、**96% 只差 ±1**、max 上界 23；
- 差异**只落在帧间块**里（帧 1：136/136），且集中在**内容边缘**（局部梯度 6–60 的点
  占 59%，整图只占 20%）；
- 把块模式统计出来：该帧 2497 个帧间块里 **OBMC 0 块、warp 0 块、interintra 0 块**
  （74% 是复合）⇒ 这三个模式在真素材这里**根本没被用到**，不可能是它们的锅。

⇒ **"孤立单点 + 只差 ±1 + 只在边缘"排除"MV 差一点"**（那会沿边缘成条），
指向**像素运算里的舍入平局**。已排除：胶片颗粒（dav1d 开/关 grain 逐像素相同 ⇒
这份素材没有颗粒）、CFL（`AV1_NO_CFL` 完全无效）、dual-filter（序列头里就是 false）。

### 覆盖盲区（重要）

`make_cases.py` 的用例表其实**没覆盖"运动"**：libaom 那批全是 `--kf-max-dist=1`
（全关键帧）、SVT 那条的内容**不带帧号**（静止）、唯一会动的 `moving_object()`
是 24px/8px 的**整数**位移。⇒ **分数精度 MV 从来没验过**。
`motion_cases.py` 把这一档补齐后，`shift` / `mosaic` 两条在当前解码器上**逐像素一致**
（滤波器开、关都一致）⇒ 子像素运动补偿与三个环路滤波器在合成内容上都精确。

⚠️ `--altref` 在 8 帧短片里**没生效**（码流与不开时**字节相同**）——
"排 B 金字塔"这一档目前**还是空的**，要用更长的片子、并且**比 md5 确认开关真生效**。

## 逐块属性对拍 + 滤波器分级隔离（2026-10-01 下半场）

`diffmap.py` 只说"差在哪儿"，要问"**是不是某个块属性/某一级滤波器搞的**"，用这两个：

| 脚本 | 作用 | 配套开关 |
|---|---|---|
| `blkattr.py` | 逐块属性对拍（位置/尺寸/帧内帧间/参考帧/两个 MV/插值滤波/变换尺寸/运动模式/复合类型）| 我们 `AV1_TRACE_BLKATTR=1`；libaom `LERXU_BLKATTR=1` |
| `filter_bisect.py` | 把**去块 / CDEF / 环路恢复**逐个关掉，与 libaom **同配置**比 | 我们 `AV1_NO_DEBLOCK/NO_CDEF/NO_LR`；libaom `LERXU_NO_DEBLOCK/NO_CDEF/NO_LR` |

**为什么需要 `LERXU_NO_*`**：libaom 自带的 `pbi->skip_loop_filter` **只盖 CDEF**，
不够做分级隔离，所以在 `av1/decoder/decodeframe.c` 里自己加了三个。

**判读要点**：

- 属性对拍里 **libaom 的 `mbmi` 是复用的** —— 帧内块/单参考块上的 `motion_mode` /
  `interp_filters` / `interinter_comp.type` 是**上一块残留值**，只在 `inter=1`
  （复合类型再加 `ref1>0`）的块上比才作数，否则会看到一堆假差异。
- **`ct`（复合类型）两边枚举序不同**：我们 `{WEDGE:0, DIFFWTD:1, AVERAGE:2, INTRA:3,
  DISTANCE:4}`，libaom `{AVERAGE:0, DISTANCE:1, WEDGE:2, DIFFWTD:3, INTRA:4}` ——
  逐例换算完全吻合，**纯编码差异**。判复合路径对不对要看受控流的逐像素一致。

**2026-10-01 在 B 站 ASMR 上的结论**：

```
blkattr.py：7518 个帧间块 —— 位置/尺寸/帧间标记/skip/参考帧/两个 MV/插值滤波/
            变换尺寸/运动模式/comp_group_idx **100% 一致**
filter_bisect.py：all_on / no_deblock / no_cdef / no_lr / **all_off 全都对不上**
            （帧 1 差 157~193 个像素，max ≤ 4）
```

⇒ **不是滤波器**（全关仍差）、**不是块属性**（全对）⇒ 差异只能在**预测/残差的像素运算**里。
再按块归类：**135/136 个差异像素落在 `skip` 块**（没有残差、输出 = 预测本身）⇒
**锁定帧间预测的像素路径**。下一步是"逐解码序帧（含隐藏的 alt-ref）比对重建结果，
找到**第一个被污染的帧**"，见 `video/mod.rs` 的"剩下的像素差"一节。

## 外科式中和实验：把差异钉到"某一个工具"上（2026-10-01 建）

`refcmp.py` 说"差多少"，`diffmap.py` 说"长什么样"，但回答不了"**是哪一层的锅**"。
`probe_neutralize.py` 补上这一步：

```sh
python probe_neutralize.py 8 noobmc nowarp nolmwarp      # 或直接不带用例名跑全部
```

做法是**两边同时关掉同一个工具**（libaom 侧用新加的 `LERXU_NO_*`，都打在**预测阶段**、
不动熵解码），看整帧差异是否归零：

    原样        帧1 183
    关 OBMC     帧1   0     ← 差异全在这个工具上
    关 warp     亮度 0 / 色度仍在

判读要点：**只要某档归零，差异就全在那个工具上**；归零不了但骤降，说明还有别的来源。
这个办法**不依赖帧号映射**，是这类"语法逐符号一致、像素差 ±1"问题最快的入口。

⚠️ 格式陷阱：`aomdec --i420 --rawvideo` 是**平面式 I420**，我们的引擎写 **NV12**。
整块缓冲直接比对必然全错（曾据此得出"帧 0 差 999000 样本"的假结论），必须逐平面拆。

## 归因必须按显示序（`attr_by_frame.py`）

块轨迹（`AV1_TRACE_BLKATTR`）是**解码序**，落盘帧是**显示序**，B 金字塔下 `--count=3`
就有 **8 段**块轨迹（隐藏的 alt-ref 也带瓦片）。早期按同下标对，得出的
"差异全在 skip 块"是**假的**，已作废。

正确对法：块轨迹每行带 `oh=<order_hint>` / `sf=<show_frame>`，
**显示帧 j ↔ order_hint == j 的那一段**；`show_existing_frame` 的显示帧不在轨迹里
（没有瓦片数据），但它的内容就是更早解码的那个隐藏帧、order_hint 正好等于显示序号。

```sh
python attr_by_frame.py --frames 8
```

## 真素材（B 站 ASMR 1080p）在 2026-10-01 下半场的结果

修掉两处 OBMC bug 后（见 `me-decode/src/video/mod.rs` 抬头）：

| 阶段 | 帧1 | 帧2 | 帧3 | 帧7 | 备注 |
|---|---|---|---|---|---|
| 上半场 | 183 | 584 | 911 | — | 逐帧放大 |
| 现在 | **0** | 3~16 | 5~42 | ~42 | `max` 恒为 1，**不再放大** |
| 关 OBMC | 0 | 3 | 6 | — | 与"现在"几乎相同 ⇒ OBMC 已无残差 |

剩下的两块（**2026-10-03 全部修完，真素材 60 帧逐像素一致** —— 结论与证据见
`ROWMT-REGRESS-HANDOFF.md` 的"收尾"一节，根因都不是猜的那两个）：

- **亮度**（曾是块级 LOCALWARP 的锅）：真根因是 warp 的取整用了 `Round2Signed`，
  规范 §4.7 的 `Round2` 与 dav1d 的 `FILTER_WARP_RND` 都是"加半再**算术右移**"
  （只在恰好半格时不同，真素材就撞上了）。曾经的"照 libaom C 源码改反而更差"
  是**把 `Round2` 与 libaom 那步 `sx4 &= ~63` 混在一起比**了；后者才是真变差的原因。
- **色度**（曾以为在复合块的 d16 预测里）：真根因是 **CFL 的 `lumaAvg`** ——
  §7.11.5 的 `MaxLumaW/H` 是**块自身的范围、不钳帧**，帧底色度块要把**帧外的重建亮度**
  也累加进去；我们当年把 MaxLuma 钳到帧内、又裁掉了帧外像素的写回 ⇒ 平均亮度偏 4，
  整个色度块被拉偏 ±1 并沿参考链传染。

## 定位方法：解码序逐帧对拍（`AV1_TRACE_REGION` ↔ `LERXU_FREG`，2026-10-03 建）

`probe_neutralize.py` 只能把差异钉到"某个工具"，再往下就要知道"**第一帧被污染的**"。
这对探针就是干这个的：给一个亮度坐标，**每重建完一帧**（`show_frame=0` 的隐藏
alt-ref 也在内、按解码序）把该点周围 `(2r+1)²` 的 Y/U/V 打一行 `FREG`，环路滤波前后
各一行；两边日志按 `oh` 对齐、逐值 diff，**第一处不一致的行就是根帧**，再用
`AV1_TRACE_BLKATTR` 找到覆盖该像素的块。

```sh
# 我们（1 = 半径，即 3×3；中心是亮度坐标）
ME_THREADS=1 AV1_TRACE_REGION=1240,804,16 zuvrust decode in.mp4 --count=14 --out=/tmp/r >/dev/null 2>/tmp/me.log
# 参考（libaom 侧的同名插桩，见 ROWMT-REGRESS-HANDOFF.md）
LERXU_FREG=1240,804,16 aomdec --threads=1 --row-mt=0 --i420 --rawvideo --limit=14 -o /tmp/ref.yuv in.ivf 2>/tmp/aom.log
```

**"两帧同序、逐值对齐"是关键**：不要按显示帧号对（B 金字塔下显示序 ≠ 解码序），
`oh` 才是两边的公共坐标。

再往下一层（块内部量）用 `AV1_TRACE_CFL_AT=mr,mc` ↔ libaom `LERXU_CFLAT=mr,mc`：
逐像素打 CFL 的 `dc / L / avg / alpha / out`。2026-10-03 就是靠它一眼看出
`avg` 差 4（Q3）、进而锁定 MaxLuma。

## row-mt 并行回放的三个探针（2026-10-02 建）

`ME_ROW_MT_PAR=1` 把重建阶段按 **(SB 行 × SB 列) 作业**并行。它的正确性只有一条判据：
**与顺序回放（`ME_ROW_MT=1`）逐字节一致**。这三件工具是当时把竞态和并行度问题挖出来用的：

| 开关 / 脚本 | 查什么 | 合格判据 |
|---|---|---|
| `ME_TRACE_RACE=1` | 回放时维护一份 4×4 的**"已写"位图**，帧内预测读上一行/左一列时若那格还没写完就报 `RACEREAD` | **顺序回放零命中**（自检）；并行回放也零命中 = 没有读到未重建像素 |
| `AV1_TRACE_ROWNEED=1` + `rowneed.py` | 预计算的跨行上界 `above_need`（`ROWB` 行）vs 帧内边推导**实际读到**的最远列（`ROWD2` 行，带块序号） | `rowneed.py` 报 **`上界不足的块 0 个`** |
| `ME_TRACE_WAIT=1` | 波前**空等**的总时长/次数 + 每次回放的墙钟（`WAIT ... scope=`） | 不是对错判据，是用来量"并行不起来是等谁" |

```sh
ME=.../zuvrust
ME_ROW_MT=1 AV1_TRACE_ROWNEED=1 $ME decode in.mp4 --count=4 --out=/tmp/rn --json >/dev/null 2>/tmp/rn.log
python rowneed.py /tmp/rn.log                 # 期望：上界不足的块 0 个
ME_ROW_MT=1 ME_ROW_MT_PAR=1 ME_THREADS=4 ME_TRACE_WAIT=1 $ME decode in.mp4 --count=8 --out=/tmp/w --json >/dev/null 2>/tmp/w.log
grep '^WAIT' /tmp/w.log                       # jobs=135（9 带 × 15 列）
```

### 三个必须记住的坑

1. **探针盖章必须等残差也加完**。`replay_txbs` 原先在 `reconstruct_block`（预测）之后就盖
   "已写"，残差还没加 ⇒ 读者看到"已写但缺残差"的像素，探针**假阴性**
   （当时并行输出 92% 像素差、探针却 0 命中）。
2. **`rowneed.py` 必须逐帧分批**。日志里会解出多帧（含隐藏的 alt-ref），每帧的块序号都从 0
   开始，混在一起比会跨帧撞车（踩过：2358 个"上界不足"全是错配）。
3. **语法阶段的调色板 flush 会污染 `ROWD2`**：它也会调 `predict_intra`，而那时代码里的
   `CUR_BLOCK_IDX` 还是上一帧回放留下的值。`rowneed.py` 用"位置一致性校验"把这类行剔除
   （本机真素材 8 帧里有 1929 条）。
