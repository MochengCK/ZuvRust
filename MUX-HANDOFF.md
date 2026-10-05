# fMP4 的 Opus / VP9 封装 · 交接说明（2026-09-29 · 第 5 轮）

> **状态：fMP4 现在支持 H.264 / H.265 / AV1 / VP9 + AAC / FLAC / Opus。**
> 解码器侧的权威进度表仍在 `crates/me-decode/src/audio/opus/mod.rs` 顶部（`OPUS-HANDOFF.md`
> 是解码器的交接）；本文档说的是**容器与封装**这一侧。
> DEVELOPMENT.md 的「各层状态」与「七、已知限制」已经同步更新。

---

## 一、本轮改了什么

| 文件 | 改动 |
|---|---|
| `crates/me-container/src/opus_config.rs` | **新增**：`OpusHead`（Ogg，小端）↔ `dOps`（ISOBMFF，大端）的**唯一**转换处 + 7 项单测 |
| `crates/me-container/src/lib.rs` | 导出 `opus_config`（`OpusConfig` / `dops_to_opus_head` / `opus_head_to_dops`） |
| `crates/me-container/src/ogg.rs` | `parse_opus_head` 改为复用 `opus_config`（**行为与错误文案不变**，消除重复解析） |
| `crates/me-container/src/mp4.rs` | `stsd` 的 `Opus` 分支：把 `dOps` **就地规范化成 `OpusHead`**；声道数以 `dOps` 为准；采样率兜底 48 kHz |
| `crates/me-mux/src/mp4.rs` | `write_audio_entry` 支持 Opus（`Opus` + `dOps`，时基必须 48 kHz）；`write_visual_entry` 支持 VP9（`vp09` + `vpcC`，含长度与初始化数据校验）；`convert_sample` 对 VP9 原样搬；`set_tracks` 放行 VP9；模块文档补第 6/7 条 |
| `crates/me-cli/src/mux.rs` | `check_codec_support` 的 fMP4 白名单加 Opus / VP9（**不加的话 CLI 会在开工前就拒掉**，muxer 里再支持也没用） |
| `crates/me-mux/tests/vp9_from_webm_manual.rs` | **新增**（默认 `#[ignore]`）：把真实 WebM 的 VP9 帧写进 fMP4，交给外部播放器检查 |

### 两个设计决定（别再改回去）

1. **Opus 的 extradata 全引擎统一是 `OpusHead` 形态**，`dOps` 在容器层读入时就地规范化。
   理由：视频的 extradata 有"形态"之分是因为它同时决定**采样数据**的排列
   （要不要长度前缀），而 Opus 的 extradata **只**是解码器初始化数据，采样字节
   在两种容器里完全相同 —— 让解码器认识两种形态没有收益，只会多一条出错路径。
   （对比：`me-mux::convert` 里视频仍坚持"解析层如实报告来源形态、转换归 muxer"。）
2. **Opus 轨的时基不是 48000 就直接报错**，不做换算。fMP4 里 Opus 的采样
   `duration` 就是 48 kHz 刻度，拿别的时基去解释会让时间轴整体漂。

---

## 二、怎么验证的（照抄即可）

```sh
cd /Users/zzzz/Documents/Lerxu/ZuvRust
export PATH="$HOME/.cargo/bin:$PATH"

# 1) 单元测试（不需要任何素材）
cargo test --workspace                       # 672 项通过

# 2) 端到端：真实 .opus → fMP4 → 读回解码，与直接解 .opus 逐字节比对
./target/release/zuvrust remux /tmp/ehren.mp4 /tmp/ehren-paper_lights-96.opus --json
./target/release/zuvrust decode /tmp/ehren.mp4 --audio --out=/tmp/from_mp4.wav --json
./target/release/zuvrust decode /tmp/ehren-paper_lights-96.opus --audio --out=/tmp/from_ogg.wav --json
cmp /tmp/from_mp4.wav /tmp/from_ogg.wav      # 无输出 = 逐字节相同

# 3) 外部裁判（本引擎不依赖它们）：macOS 的 CoreAudio / AVFoundation
afinfo /tmp/ehren.mp4                        # 2 ch, 48000 Hz, opus, 11406 packets
swift /tmp/avcheck.swift /tmp/ehren.mp4      # duration=228.120s substype=opus

# 4) VP9：引擎里没有 VP9 的输入源，用真实 WebM 的帧喂进 muxer
curl -o /tmp/vp9.webm \
  https://test-videos.co.uk/vids/bigbuckbunny/webm/vp9/360/Big_Buck_Bunny_360_10s_1MB.webm
WEBM_FILE=/tmp/vp9.webm VP9_OUT=/tmp/vp9.mp4 \
  cargo test -p me-mux --test vp9_from_webm_manual -- --ignored --nocapture
swift /tmp/avcheck.swift /tmp/vp9.mp4        # duration=10.001s substype=vp09 rate=29.997
```

**真解码**用内置浏览器（本机 Chromium 自带 VP9 解码器；AVFoundation 在本机
**没有** VP9 解码器，只能解析容器，`avdecode` 会报 "Cannot Decode / -12906"，
那是环境限制不是文件问题）。把产物放到一个本地 HTTP 服务下再用浏览器打开
（内置浏览器拦 `file://`）：

```sh
cd /tmp && python3 -m http.server 8912 --bind 127.0.0.1 &
# 然后浏览器打开 http://127.0.0.1:8912/media_av.html?f=vp9.mp4&at=5
```

本轮实测结果：

| 检查 | 结果 |
|---|---|
| `.opus` → fMP4 → 读回解码 vs 直接解 `.opus` | 两个 WAV **逐字节相同**（10,949,404 样点/声道，43797660 字节） |
| `afinfo` / AVFoundation 看 Opus 产物 | 认出 opus 2ch/48kHz、228.120 s（与源一致） |
| 浏览器播 Opus 产物（WebAudio 取能量） | 播到 t=5.08s，峰值 RMS **0.1276**（真有声音，不是静音） |
| 浏览器播 VP9 产物（解一帧画到 canvas） | 播到 t=5.09s，画布 640×360、均值 83.41 / 方差 1420（真有画面，不是黑屏） |
| 浏览器 seek | 索引读完（时长收敛到 11.145 / 10.001）后 `seek(8)` 精确落到 **8.000** |

---

## 三、两个必须记住的发现

### 1. Matroska 的 VP9 `CodecPrivate` 就是 `vpcC` 载荷 —— 但常见文件**没有**它

- 有 `CodecPrivate` 时：它和 fMP4 里 `vpcC` 的**载荷**是同一串字节（不含 box 头），
  所以将来 WebM → fMP4 这条链在配置上是**零转换**的。
- 但 libvpx / ffmpeg 产出的 WebM **大多不写 `CodecPrivate`**（本轮下载的 Big Buck Bunny
  就是 0 字节）。这时只能从**第一帧的帧头**读 profile 与位深（`frame_marker(2)=0b10`
  → `profile_low/high_bit` → 关键帧的 color_config 里才有位深），其余按 VP9 默认值填。
  手工测试里的 `vp9_profile_and_depth()` + `vpcc()` 就是这个兜底实现，可以直接搬。
- **推论**：现在"VP9 轨缺 `vpcC` 就报错"这条纪律对 **fMP4 → fMP4** 是对的
  （配置盒本该原样带着），但**将来接 WebM 输入时必须先补这个推导**，否则真实文件
  会全被挡在门外。

### 2. 分片 fMP4 不写 `sidx` / `mehd` / `mfra` ⇒ 时长与 seek 是"渐进已知"的

实测（**与本轮改动无关，H.264/AAC 的老产物一样**）：

- Chromium 刚打开时只知道第一个分片的时长（就是 `DEFAULT_FRAGMENT_US` = 2 秒），
  `durationchange` 随读取一路上涨到真值；此期间 seek 到"还没读到的位置"会被夹回开头。
  索引建好后 seek 精确（11.145 s 的产物 `seek(8) → 8.000`）。
- AVFoundation 一次扫全文件，直接给准确时长（228.120 / 10.001 秒）。
- 对"边下边播"没有影响（本来就是顺序读）；影响的是"打开就跳到末尾"这类用法。

根治办法是给每个分片补 `sidx`（DASH 的标准做法；分片头写、不影响流式直写），
或收尾时补 `mfra`/`tfra`（只对文件产物有意义，管道产物补不了）。

---

## 四、下一步候选（按价值排序）

1. **分片补 `sidx`**：时长与 seek 立刻变准，宿主那边"打开进度条就跳到真长度"的体验直接受益，
   而且改动集中在 `me-mux/src/mp4.rs` 的分片写出处（每个 `moof` 前插一个 `sidx`）。
2. **WebM（Matroska）解封装**：现在 VP9 / Opus 都**没有真实输入源**（Opus 只有 `.opus` 的 Ogg 容器）。
   接 WebM 时记得：EBML 的"长度未知"判据是**值位全 1**（不是某个字节掩码 —— 这个坑
   本轮又在手工测试里踩了一次，8 字节长度的 `Segment` 会被误判）；VP9 的配置要按上面第 1 条推导。
3. **TS 侧的诚实性**：`CodecId::ts_stream_type()` 对 AV1/VP9/FLAC/Opus 都给 `0x06`
   （private stream），播放器基本不认。CLI 层 `check_codec_support` 已经把这些组合拒掉了，
   但 `me-mux` 的 `TsMuxer` 自己没检查 —— 直接用它的人会静默产出"有数据但解不出来"的 TS。
4. `me-decode` / `me-mux` 还有约 60 条 clippy 风格告警（纯清理，无功能影响）。
