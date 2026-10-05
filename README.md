# ZuvRust

**完全自研的媒体引擎** —— 一个可执行文件，零第三方媒体库依赖。

它不链接 FFmpeg，也不链接任何别的媒体库：容器解析、容器层合并、AV1 软解、
音频软解全部自己实现。调用方（播放器、下载器、任何程序）通过
**命令行 + NDJSON** 与它通信，**不链接它的任何代码**。

> 想了解内部结构、硬约定、性能数据与改造记录，见 **[DEVELOPMENT.md](DEVELOPMENT.md)**。

---

## 能做什么

| 子命令 | 说明 |
| --- | --- |
| `probe <输入>` | 探测容器与轨道（编码、分辨率、时长、码率…） |
| `mux <输出> <输入>...` | **合并**：把多个输入接成一个文件（角色由探测决定，详见下） |
| `remux <输出> <输入>` | 换容器重新封装，**不重编码** |
| `download <地址>` | 下载 HTTP 单文件 / HLS / DASH（含 AES-128 加密分片） |
| `play <输入>` | 播放数据源，按采样喂给调用方 |
| `stream <输入>` | 流式播放：一行元信息 + 分片 fMP4 字节（喂给 MSE 用） |
| `decode <输入>` | 解码：视频轨出帧，音频轨出 WAV |
| `version` | 打印版本与本机支持的能力 |

`mux` 的语义值得单独说：**角色由容器探测决定，与文件名无关**。含视频轨的输入进
"画面序列"、含音频轨的进"声音序列"，第 j 段 = 画面序列[j] + 声音序列[j]，段内
多轨交织、段末把时间轴推到各轨的最大值再进下一段。所以**调用方要保证"同角色输入的
传入顺序 = 时间顺序"**。HLS 分片、下载器切出来的分段，都是这么拼成完整文件的。

它**不做编码/转码** —— 只做封装与解码。

---

## 支持范围

**容器**：MPEG-TS、ISO BMFF（mp4 / mov / m4s）、Ogg、FLAC、ADTS、MPEG 音频

**协议**：HTTP(S)、HLS、DASH

**编码标识**（容器里认得出，用于写 `stsd` / `PMT`）：
H.264、H.265、AV1、VP9、VP8、AAC、MP3、Opus、FLAC、AC-3、E-AC-3

**解码**（"认得出"不等于"解得开"，这里才是能解的）：

| 编码 | 解码方式 |
| --- | --- |
| AV1 | **自研软解**（完整实现，逐像素与参考解码器对拍过）；Apple Silicon（M3 及以后）优先走 VideoToolbox **硬解** |
| H.264 | macOS 走 VideoToolbox **硬解** |
| Opus / AAC / FLAC / MP3 | **自研软解** |

⚠️ **AV1 软解目前还不是实时性能**：1080p 单线程约 50 ms/帧（约 20 fps），
目标是 33 ms。在 Apple Silicon 上它会自动走硬解，所以那道坎主要影响 Intel 机器。
具体数据与后续计划见 [DEVELOPMENT.md](DEVELOPMENT.md) 的性能段。

**BT / 磁力不在本引擎内**：磁力由下载引擎（XferRust）负责，ZuvRust 只负责
播放 / 合并它写出来的文件（见下面的"边下边播"）。

---

## 用法

```sh
# 探测
zuvrust probe movie.mkv --json

# 把 HLS 下载下来的分片拼成一个 mp4
zuvrust mux out.mp4 seg000.ts seg001.ts seg002.ts --json

# 下载（HLS 会自动选码率；可用 --max-height 限高）
zuvrust download "https://example.com/index.m3u8" --out movie.mp4 --progress

# 换容器（不重编码）
zuvrust remux out.mp4 in.ts --json

# 解成帧（落盘）或解成 WAV
zuvrust decode movie.mp4 --count 30 --out frames/
zuvrust decode music.flac --audio --out music.wav
```

**界面上的进度**来自 `--progress`：它会额外吐进度行，调用方按行解析即可。

---

## 调用方契约（重要）

宿主**只依赖三样东西**：退出码、stdout 的字节、stderr 的字节。

**输出**按 `--json` 分两种形态：

- `--json`：**恰好一行** JSON —— 成功 `{"ok":true,"result":{…}}`，
  进度 `{"ok":true,"type":"…","data":{…}}`，错误走 stderr
  `{"ok":false,"error":{"kind":"…","message":"…","retryable":…}}`。
  `kind` 是机器可读分类（`io` / `container` / `bitstream` / `protocol` /
  `unsupported` / `need-more` / `cancelled`），`retryable` 表示重试有没有意义。
- 不带 `--json`：**缩进过的多行** JSON（给人看的），没有 `ok` / `result` 包装。

**退出码**是对外契约的一部分 —— 调用方靠它决定"要不要重试"：

| 码 | 含义 |
| --- | --- |
| 0 | 成功 |
| 2 | 用法错误（命令拼错了，重试无用） |
| 3 | 输入不可用（打不开 / 容器认不出 / 码流非法，重试无用） |
| 4 | 临时失败（网络超时、数据还没到，**可以重试**） |
| 5 | 引擎内部错误 |
| 6 | 被取消 |

这条契约有测试守着：`crates/me-cli/tests/cli_contract.rs` 会**真的起一个进程**，
逐条验证退出码、两种输出形态、stderr 的编码与上色、带空格与中文的路径等等。

---

## 边下边播

下载和播放是两个引擎，它们之间只有一条协作接口：**可用性图**。

下载器把"哪些字节已经真的下到了"写成一个 JSON 文件（片位图），通过
`--avail=<文件>` 交给 ZuvRust。于是"还没下到的区间"是**等**而不是**损坏** ——
没有这张图的话，空洞读出来是零字节，容器层会把它当成坏数据直接终止播放。

播放位置也会反向上报（`playhead`），下载器据此优先下播放位置附近的数据。

```sh
zuvrust stream movie.mp4 --avail=/path/to/avail.json --wait-timeout=30000
```

`--wait-timeout` 是单次等待的上限（默认 30000 ms）：网络已经断了时，调用方可以
调小它以便更快失败、提示用户。

---

## 平台与产物

每个平台的产物是一个 tar 包，里面是 `zuvrust`（Windows 为 `zuvrust.exe`）、
`LICENSE` 与 `README.md`：

| 平台 | 产物 |
| --- | --- |
| macOS Intel | `zuvrust-darwin-x86_64.tar.gz` |
| macOS Apple Silicon | `zuvrust-darwin-arm64.tar.gz` |
| Linux x86_64 | `zuvrust-linux-x86_64.tar.gz` |
| Linux arm64 | `zuvrust-linux-arm64.tar.gz`（静态链接） |
| Windows x86_64 | `zuvrust-windows-x86_64.tar.gz` |

产物由 CI 在每个平台上**原生构建 + 跑完测试**之后才打包，并且打包后会再下载
**产物本身**验一遍（架构、包内容、契约）。也就是说，从 Release 拿到的包是
"跑过测试且被独立验证过"的那一个。

---

## 构建与自测

```sh
cargo build --release --bin zuvrust      # 产物在 target/release/zuvrust

cargo test --workspace                   # 全量测试（单元 + 集成 + 契约）
./scripts/check-arches.sh                # 用非本机架构编译一遍（见下）
./scripts/verify-artifact.sh <包> zuvrust "<期望架构>"   # 验收打包产物
```

**`check-arches.sh` 不是可选项**：引擎里大量 SIMD 实现按 `target_arch` 分叉，
非 x86 的那一侧是**桩** —— 桩写错了在本机（Intel）永远浮现不出来，只有别的架构
编译时才报错。这个脚本就是用"另一个架构 + `--all-targets`"把这类错误提前到本地。

---

## 许可

**GPL-3.0 或更高版本**（GNU General Public License v3.0 or later）。
完整许可证文本见 [LICENSE](LICENSE)。

---

## 深入

- 分层架构、必须守住的硬约定、通信层契约、性能数据与逐刀记录 →
  **[DEVELOPMENT.md](DEVELOPMENT.md)**
- AV1 软解的全面检查报告（正确性 / 功能完整性 / 三平台 / 优化清单）→
  [AV1-AUDIT.md](AV1-AUDIT.md)
- 参考实现对拍工具 → [`tools/av1ref/`](tools/av1ref/)
