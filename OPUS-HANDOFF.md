# Opus 解码器完成 · 交接说明（2026-09-29 凌晨 · 第 4 轮）

> 本文档说的是**解码器**这一侧（权威进度表：`crates/me-decode/src/audio/opus/mod.rs` 顶部）。
> **容器与封装**这一侧的交接（fMP4 的 `dOps`/`vpcC` 写法、Opus/VP9 写进 mp4）见
> `MUX-HANDOFF.md`。

> **状态：Opus 解码器已经没有已知缺口。**
> 本轮修掉了最后一处残余偏差（`testvector10` 模式切换处的 5 ms CELT 冗余帧），
> **12 组官方向量（3078 万样本）全部逐样本对齐**，端到端 `decode --audio` 与参考
> 逐样本最大差 1 个量化步长。
> 权威进度表在 `crates/me-decode/src/audio/opus/mod.rs` 顶部 —— 接着做之前先看它。
> 本文档取代 `HANDOFF-2026-09-29-silk-plc.md`（那一轮做的是 SILK 侧 PLC）。

---

## 一、根因：一个参数写错，症状会跑到 20 帧之外

**`opus_decoder.rs` 里混合模式的冗余帧长度读错了参数：**

```rust
// ✗ 错：`dec_uint(8)` 的 8 是「位数」吗？不是 —— `dec_uint(ft)` 的参数是**上界**，
//    ft=8 只能读出 0..7，于是 redundancy_bytes = 2..9
dec.dec_uint(8)? as usize + 2

// ✓ 对（参考实现：`ec_dec_uint(&dec, 256)+2`）
dec.dec_uint(256)? as usize + 2
```

`redundancy_bytes` 同时决定两件事，所以错了会**双杀**：

1. **冗余帧从哪开始** —— `celt_len = len - redundancy_bytes`，而冗余帧数据在
   `data[celt_len..]`。长度错 → 解到**错的字节**上；
2. **主体数据多长** —— `dec.truncate_tail(redundancy_bytes)` 之后
   `len_bytes()` 变成主体长度，CELT 的 "还剩多少位" 判断跟着错。

只影响 **`mode == Hybrid` 且真的有冗余帧**的包（SILK-only 走另一条公式
`len-((tell+7)>>3)`，那条一直是对的）。12 组官方向量里只有 **10** 会撞上 ——
所以 tv05/06/12 看起来"已经对齐"。

### 症状（记住这个形状，下次一眼认出来）

- 差异**只出现在每个冗余帧那一帧**，之后**逐帧衰减**（2645 → 33 → 6 → 3 → 消失；
  另一处 4402 → 4074 → …）；两声道同向、幅度略有不同、末尾变成一层
  Nyquist 级别的高频振铃（最大差 17013）。
- 一旦看到"**某个点之后整体错位、再慢慢衰减回去**"，先怀疑**状态**（`decode_mem` /
  `preemph_mem` / 冗余帧长度这类"跨帧记账"），而不是当帧的算术。

---

## 二、定位过程（这套二分法很好用，值得复用）

1. **逐帧差异图**（临时测试：逐包 → 逐子帧算 `max_diff`）→ 差异只在
   `testvector10` 的 8 处 `celt_to_silk` + 6 处 `silk_to_celt` 冗余帧上，
   且随后衰减 → 定位到"冗余帧解码"。
2. **看状态**：`OpusCustomDecoder` 结构体只在 `celt.c` 里可见（`opus_decoder.c`
   读不到字段），所以把桩插在 **`celt_decode_with_ec` 的入口/出口**：
   `RDIN/RDOUT` 打 `N/LM/C/CC/start/end/len`、`postfilter_period(_old)/gain(_old)/tapset(_old)`、
   `preemph_memD`、`rng`、`loss_count`、`decode_mem` 的尾部与 overlap 区。
   自研侧加一个 `debug_state(tag)` 打同样的行。
   → **状态逐位一致**（rng=158958592、pf 341/0.375/1、preemph 2885.94458、decode_mem 尾部全对）
   → 排除"incoming state 错"。
3. **看熵层位置**：给参考的 `silk_Decode` 打 `SD1..SD7`（入口 / VAD+LBRR 标志 /
   LBRR 长度 / LBRR 跳过 / 立体声预测 / 每声道解码 / 出口），每行带
   `ec_tell` + `rng` + `val`；自研侧同样打。
   → **每一步的 `val`/`rng` 都逐位相同**（`SD6 ch=1` 两边都是 `tell=1793 val=592 rng=16109626`）
   → 排除"SILK 位流消耗错"。
4. **看冗余帧判定点**：参考打印 `RDCHK tell / RDRD tell+rng+val+redun+c2s+rbytes+len`。
   → 参考 `rbytes=180`、我们 `rbytes=7` ⇒ `dec_uint` 读出的值不同
   ⇒ **一眼看到 `dec_uint(8)` 应该是 `dec_uint(256)`**。

改动只有一行，tv10 的 **4606 帧立刻全部逐样本一致**。

> 教训：**"前面全对、从某处开始整体错" 这类症状，先花 3 分钟把两侧的状态与
> 中间量打成同样的行逐行 diff**，比盯着输出猜快得多。本轮从"排除状态"到
> "看到 `dec_uint` 参数错"一共 4 步。

---

## 三、怎么验证（这套办法很值钱，别换）

```sh
cd /Users/zzzz/Documents/Lerxu/ZuvRust
export PATH="$HOME/.cargo/bin:$PATH"

cargo test -p me-decode --lib                      # 单测（不需要素材）

# 整包对拍：12 组官方向量的整段 PCM（**严格**：最大差 ≤2 步，SILK 要求逐位相同）
OPUS_TV_DIR=/tmp/opus_tv_pkts cargo test -p me-decode --test opus_decode_ref -- --ignored --nocapture

# 丢包隐藏：给参考实现喂同一套丢包图样（24 组 / 6071 万样本）
sh /tmp/opus_plc_gen.sh
OPUS_TV_DIR=/tmp/opus_tv_pkts cargo test -p me-decode --test opus_plc_ref -- --ignored --nocapture

# 端到端
./target/release/zuvrust decode /tmp/ehren-paper_lights-96.opus --audio --out=/tmp/out.wav
```

当前基线（2026-09-29）：

| 项目 | 结果 |
|---|---|
| `opus_decode_ref` 12 组整包 | **3078 万样本，99.99% 逐样本相同，最大差 1 步**（tv02/03/04 逐位相同）|
| `opus_plc_ref` 24 组丢包 | CELT 侧最大差 2 步；**SILK 侧最大差 0**（6071 万样本）|
| `cargo test --workspace` | **660 项全过** |
| 端到端 WAV vs 参考 PCM | 最大差 1 步；长度差正好一个 `pre_skip=356`；0.0223% 样本不同 |

测试口径：`silk_only_vectors_are_bit_exact`（SILK 纯整数 → **必须最大差 0**）
与 `full_decoder_matches_reference_on_official_test_vectors`（12 组，≤2 步）。
**别把 SILK 的 1 步偏差当浮点误差放过去** —— 那一定是真 bug。

---

## 四、调试基建（都在 `/tmp`，重建很便宜）

| 东西 | 用途 |
|---|---|
| `/tmp/opus-rfc6716` | 参考实现主构建（跑 dump 用，**别弄脏**）|
| `/tmp/opus-dbg` | + SILK/CELT 分层 dump 插桩（`/tmp/opus_patch_silk.py` 等）|
| `/tmp/opus-plcdbg` | + PLC 插桩（`PLCB/PLCE/PLCF/PLCG/PLCT`、`PLCP/PLCL/PLCD/PLDI/PLDS`）|
| `/tmp/opus-reddbg` | + 本轮的状态/位流插桩（`RDIN/RDOUT/RDCHK/RDRD/RDMRK/SD1..SD7`）|
| `/tmp/opus_rebuild_all.sh` | 重建主构建 + 重生成全部 dump（约 7 分钟）|
| `/tmp/opus_plc_probe.c` + `/tmp/opus_plc_gen.sh` | 丢包图样驱动（清单两边共用，永不漂）|

自研侧一律用 `RDDBG=1` / `PLCDBG=1` 环境变量开同样的行，**逐行 diff**。

**纪律**：改任一插桩脚本都必须整条重跑（dump 与插桩版本不一致是对拍误报的头号来源）。

---

## 五、还差什么（Opus 之外）

Opus 这条线**没有已知缺口了**。工程上还能做的（都不是"修 bug"）：

- **fMP4 里写 Opus / VP9**（`dOps` / `vpcC` 配置盒）：现在 `remux` 到 fMP4 会明确
  报不支持（见 DEVELOPMENT.md「已知限制」）。解码器做完之后，这项才算真正可用。
- `me-decode` 的 clippy 还有约 60 条风格 warning（`operator precedence might not be
  obvious` 之类，集中在 `bands.rs` / `celt.rs` / `rate.rs`），纯清理。
- 其它编解码器（Vorbis 等）仍明确报 `Unsupported`。
