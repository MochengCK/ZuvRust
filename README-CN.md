# ZuvRust

[English](README.md) | 简体中文

**面向应用集成的自研媒体引擎。** 一个可执行文件、一套稳定的 JSON-RPC API，
不依赖任何第三方媒体库。

ZuvRust 自己实现容器解析、转封装、合并、AV1 软解、H.264 硬解与音频解码；
不链接 FFmpeg、GStreamer 或其它媒体引擎。应用通过进程边界与它通信：
启动 `zuvrust serve --stdio`，发送 JSON-RPC 请求，接收进度通知与二进制媒体帧。

- **稳定的集成接口**：JSON-RPC 2.0、请求 ID、取消、进度与二进制流。见 [API-CN.md](API-CN.md)。
- **官方 JavaScript 客户端**：[sdk/js](sdk/js)，零依赖，Node 18+。
- **崩溃隔离**：每个任务跑在独立子进程里，坏文件不会拖垮应用进程。
- **按模块构建**：只编自己需要的 `audio` / `video` / `mux`，也可任意组合。
- **Apache-2.0**：商业与开源产品都可以宽松使用。

## 能做什么

| 能力 | 说明 |
| --- | --- |
| 探测 | 容器、轨道、编码、分辨率、时长、码率 |
| 合并 | 音视频合并、分片串接，输出 MP4 / MKV / WebM / TS / FLAC / ADTS |
| 转封装 | 换容器，不重编码 |
| 下载 | HTTP(S)、HLS（含 AES-128）、DASH |
| 视频解码 | AV1 自研软解；H.264 走系统硬解（VideoToolbox / Media Foundation / VA-API） |
| 音频解码 | AAC-LC、MP3、FLAC、Opus（全部自研） |
| 流式播放 | 输出分片 fMP4 给 MSE，或解码后的 NV12 / PCM 包流给自绘播放器 |
| 增长文件 | 读取另一个进程仍在写入的文件，支持边下边播 |

ZuvRust **不做编码/转码**。

## 集成方式

### 1. 启动服务

```sh
zuvrust serve --stdio
```

stdin/stdout 上的每条消息都是带长度前缀的帧。忽略 5 字节帧头后，一个请求如下：

```json
{"jsonrpc":"2.0","id":1,"method":"probe","params":{"input":"movie.mkv"}}
```

响应包含容器与轨道信息。长任务会发 `progress` 通知；流式任务会发二进制帧，
并以 `stream/end` 通知结束。完整协议、方法参数、`MEV1` 包格式与错误码见
**[API-CN.md](API-CN.md)**。

### 2. 使用官方 JS SDK

```js
import { ZuvRust } from 'zuvrust-client'

const engine = await ZuvRust.start({ bin: '/opt/zuvrust/zuvrust' })

const info = await engine.probe('movie.mkv')

await engine.mux('out.mp4', ['video.mp4', 'audio.m4a'], {
  fragmentMs: 2000,
  onProgress: (data, type) => console.log(type, data.percent)
})

for await (const event of engine.stream('movie.mp4', { mode: 'frames', audioPcm: true })) {
  if (event.type === 'video') render(event.data, event.width, event.height)
  if (event.type === 'audio') play(event.data, event.sampleRate, event.channels)
  if (event.type === 'end' && !event.ok) console.error(event.error)
}

await engine.close()
```

完整客户端 API 见 [sdk/js/README.md](sdk/js/README.md)。

### 3. 或继续用一次性 CLI

```sh
zuvrust probe movie.mkv --json
zuvrust mux out.mp4 video.mp4 audio.m4a --json --progress
zuvrust remux out.mkv in.ts --json
zuvrust download "https://example.com/index.m3u8" --out movie.mp4 --progress
zuvrust decode movie.mp4 --count 30 --out frames/
zuvrust decode music.flac --audio --out music.wav
```

CLI 契约（退出码、NDJSON 形状、stderr 编码）见 [DEVELOPMENT.md](DEVELOPMENT.md)，
并由真实进程级测试覆盖。

## 支持范围

**容器**：MPEG-TS、ISO BMFF（mp4 / mov / m4s）、Matroska / WebM、Ogg、FLAC、
ADTS、MPEG 音频。

**合并输出**（按扩展名选封装器；编码组合不被目标容器支持时明确报错，
绝不产出坏文件）：

| 扩展名 | 容器 | 装得下的编码 |
| --- | --- | --- |
| `.mp4` `.m4s` `.mov` `.m4a` | 分片 fMP4 | H.264 / H.265 / AV1 / VP9 + AAC / MP3 / FLAC / Opus / AC-3 |
| `.mkv` `.mka` | Matroska | 上面全部 |
| `.webm` | WebM 子集 | VP8 / VP9 / AV1 + Opus |
| `.ts` `.m2ts` | MPEG-TS | H.264 / H.265 + AAC / MP3 / AC-3 |
| `.flac` | 裸 FLAC | 仅 FLAC |
| `.aac` | 裸 ADTS | 仅 AAC |

**协议**：HTTP(S)、HLS、DASH。

**解码**

| 编码 | 实现 |
| --- | --- |
| AV1 | 自研软解；Apple Silicon M3 及以后优先走 VideoToolbox 硬解 |
| H.264 | 三平台系统硬解：macOS VideoToolbox、Windows Media Foundation + D3D11、Linux VA-API |
| AAC-LC / MP3 / FLAC / Opus | 自研软解 |

AV1 软解正确性已逐像素对拍，但老机器上还不是实时性能：1080p 单线程约
50–55 ms/帧（约 18–20 fps，随频率与散热浮动）。Apple Silicon 会自动走硬解。
具体数据与优化记录见 [DEVELOPMENT.md](DEVELOPMENT.md)。

**BT / 磁力不在本引擎内。** ZuvRust 只负责播放或合并下载引擎产出的文件。
两者的协作接口是[可用性图](API-CN.md#stream)：下载器写出哪些字节已经到盘，
ZuvRust 对缺失区间**等待**而不是当成损坏，同时反向上报播放头。

## 按模块构建

引擎拆成三个可独立编译的模块：

| feature | 包含 | 典型应用 |
| --- | --- | --- |
| `audio` | AAC / MP3 / FLAC / Opus 解码器，`decode --audio` | 音频播放器、转换工具 |
| `video` | AV1 软解 + H.264 硬解，`decode`、`stream --video=frames` | 视频播放器 |
| `mux` | probe / mux / remux / fMP4 流转封装 | 下载器、合并工具 |

任意组合都合法。`probe`、`download`、`play`、`serve` 永远保留。
`zuvrust version --json` 的 `modules` 字段会报告实际编进来的模块；
调用未编译的命令会返回退出码 2 和明确提示。

```sh
./scripts/build-modules.sh --release --modules=mux
./scripts/build-modules.sh --release --modules=audio,video
./scripts/build-modules.sh --release --modules=audio,video,mux
```

默认 Release 产物是给 Lerxu 桌面端的 `mux` 构建（桌面端播放界面暂时停用）。
需要完整引擎时从源码构建，或手动运行 CI 工作流并填 `modules=audio,video,mux`。

## 从源码构建

要求 Rust stable。Linux ARM64 交叉构建额外需要 `cargo-zigbuild`。

```sh
cargo build --release                 # 三个模块全开
./scripts/build-modules.sh --release  # 带参数校验的构建包装
cargo test --workspace
./scripts/check-arches.sh             # 交叉架构编译门禁
```

产物在 `target/release/zuvrust`（Windows 为 `zuvrust.exe`）。

## 平台与产物

| 平台 | 产物 |
| --- | --- |
| macOS Intel | `zuvrust-darwin-x86_64.tar.gz` |
| macOS Apple Silicon | `zuvrust-darwin-arm64.tar.gz` |
| Linux x86_64 | `zuvrust-linux-x86_64.tar.gz` |
| Linux arm64 | `zuvrust-linux-arm64.tar.gz`（静态） |
| Windows x86_64 | `zuvrust-windows-x86_64.tar.gz` |

CI 在每个平台原生构建并跑完测试，再下载**产物本身**做架构、模块、启动与契约
验收，全部通过才允许发布。压缩包内含二进制、`LICENSE`、`NOTICE`、中英文 README、
协议文档与 `sdk/js` 客户端源码。

## 设计要点

- 容器解析与合并是纯字节操作，不碰解码。
- 解码是独立一层，可在构建期裁掉。
- 时间统一用整数时基，只在跨轨比较时换算成微秒。
- 产物要么正确，要么明确报错；绝不静默写坏文件。
- JSON-RPC 服务是子进程监督者：既保留进程隔离，又只暴露一个集成端点。

更多内部约定、性能数据与改造记录见 [DEVELOPMENT.md](DEVELOPMENT.md)；

## 许可

Apache License 2.0。见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

商业与开源应用都可以按 Apache-2.0 的条款使用。引擎不包含任何第三方媒体库代码。
