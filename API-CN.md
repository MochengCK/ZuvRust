# ZuvRust API

[English](API.md) | 简体中文

本文是 `zuvrust` 媒体引擎的集成契约，覆盖常驻 JSON-RPC 服务
（`zuvrust serve --stdio`）、流式二进制帧和统一错误模型。

一次性 CLI（`zuvrust probe ...`、`zuvrust mux ...`）继续保留。服务模式让应用
保持一个监督进程、并发发请求、接收进度并取消任务，而不用自己拼命令行。

## 设计

`serve` 是**监督者**，不是进程内共享解码器。每个任务跑在 `zuvrust` 子进程里：

- 坏输入把容器解析打崩时，应用收到的是 JSON-RPC 错误，而不是一起崩溃；
- 任务可并发，互不共享解码状态；
- 取消任务会终止子进程（Unix 先发 `SIGTERM`，让它清理半截产物）；
- 协议进程本身不含媒体代码，任何模块组合都可用。

代价是每个任务多一次进程启动（毫秒级）。probe/mux/remux/download 可以忽略；
每条流各占一个子进程。

## 传输

服务通过 stdin/stdout 说 JSON-RPC 2.0，使用长度前缀分帧。**不开放网络端口。**
所有整数小端。

```text
frame   := u32 payload_len
           u8  type
           u8[payload_len] payload

type 0  := JSON-RPC 消息（UTF-8 JSON）
type 1  := 二进制流块：
           u32 stream_id
           u32 kind
           u8[payload_len - 8] data
```

限制：JSON 帧 8 MiB，二进制帧 64 MiB。客户端请求只能使用 type 0；
响应与通知使用 type 0，流数据使用 type 1。

第一个请求应为 `initialize`，响应会告诉客户端协议版本、引擎能力与分帧常量。

## 方法

| 方法 | 用途 |
| --- | --- |
| `initialize` | 握手：协议版本、引擎版本、模块、能力 |
| `version` | 与 `zuvrust version --json` 相同的对象 |
| `ping` | 健康检查 |
| `probe` | 容器与轨道信息 |
| `mux` | 合并/串接输入为一个文件 |
| `remux` | 换容器，不重编码 |
| `download` | 下载 HTTP/HLS/DASH |
| `decode` | 视频解码成帧，或音频解码成 WAV |
| `stream` | 输出 fMP4 字节流或解码帧流 |
| `cancel` | 取消请求或流 |
| `shutdown` | 停止服务 |

带 `id` 的请求一定收到恰好一个响应；没有 `id` 的通知不应答。不支持批量请求。

### `initialize`

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{
  "protocolVersion":1,
  "client":{"name":"my-app","version":"1.0.0"}
}}
```

结果：

```json
{
  "protocolVersion": 1,
  "engine": { "与 version --json 相同": "..." },
  "framing": {
    "header": "u32le payload_len + u8 type",
    "typeJson": 0,
    "typeBinary": 1,
    "maxJsonBytes": 8388608,
    "maxBinaryBytes": 67108864,
    "binaryPayload": "u32le streamId + u32le kind + bytes"
  },
  "binaryKinds": {"fmp4": 0, "frames": 1},
  "methods": ["initialize", "version", "ping", "probe", "mux", "remux",
              "download", "decode", "stream", "cancel", "shutdown"],
  "notifications": ["progress", "stream/end"],
  "isolation": "one child process per job"
}
```

`engine.modules` 是这个二进制实际编译的模块。纯 `mux` 构建没有 `audio` / `video`；
对它调用 `decode` 会得到错误码 `1002`、`kind: "unsupported"`。

### `probe`

```json
{"jsonrpc":"2.0","id":2,"method":"probe","params":{
  "input":"/media/movie.mkv",
  "grow":false,
  "size":123456789,
  "maxHeight":1080,
  "maxBandwidth":8000000
}}
```

除 `input` 外都可选。`grow` 用于仍在写入的下载文件。结果与 CLI
`probe --json` 的 `result` 相同：`container`、`durationUs`、`tracks[]`、
`bitrate`、`source`。

### `mux`

```json
{"jsonrpc":"2.0","id":3,"method":"mux","params":{
  "output":"/media/out.mp4",
  "inputs":["/media/video.mp4","/media/audio.m4a"],
  "fragmentMs":2000,
  "tracks":"both"
}}
```

`inputs` 必须是非空数组。同角色输入按给定顺序串接。`fragmentMs` 控制分片时长；
`tracks` 取 `both`（默认）、`audio` 或 `video`。进度以 `progress` 通知发送，
最终结果与 CLI `mux --json` 相同。

### `remux`

```json
{"jsonrpc":"2.0","id":4,"method":"remux","params":{
  "output":"/media/out.mkv","input":"/media/in.ts",
  "fragmentMs":2000,"tracks":"both"
}}
```

`output`、`input` 必填；`fragmentMs` 与 `tracks` 可选，含义同 `mux`。
进度通知的 type 是 `mux`。

### `download`

```json
{"jsonrpc":"2.0","id":5,"method":"download","params":{
  "input":"https://example.com/index.m3u8",
  "output":"/media/movie.mp4",
  "maxHeight":1080
}}
```

`output` 可选；不传时引擎按 URL 推导文件名。DASH 输入会每条轨生成一个文件，
并在结果里列出。进度通知的 type 是 `download`。

### `decode`

```json
{"jsonrpc":"2.0","id":6,"method":"decode","params":{
  "input":"/media/movie.mp4",
  "count":30,
  "out":"/media/frames",
  "seek":12.5
}}
```

视频模式把 `frame-000001.nv12` 与 `frames.json` 写进 `out`。
音频模式（`"audio": true`、`"out": "/media/audio.wav"`）写 WAV。
`noGrow` 关闭增长文件读取。`decodeMode` 取 `auto`（默认）、`hardware`
（仅硬解）或 `software`（不碰硬件）；`threads` 设置解码线程数。
进度通知的 type 是 `decode-progress`。

### `stream`

```json
{"jsonrpc":"2.0","id":7,"method":"stream","params":{
  "input":"/media/movie.mp4",
  "mode":"fmp4",
  "seek":0,
  "fragmentMs":500,
  "avail":"/tmp/avail.json",
  "waitTimeoutMs":30000
}}
```

`mode` 为 `fmp4`（默认）或 `frames`：

- `fmp4`：引擎转封装为分片 MP4。二进制块的 `kind = 0`。结果里的 `meta.mime`
  与 `meta.tracks` 描述该喂给播放器什么。
- `frames`：引擎把视频解成 NV12，可选把音频解成 PCM。二进制块 `kind = 1`，
  内容为下面的 `MEV1` 包。加 `"audioPcm": true` 才包含音频。

可选字段：`mode`、`seek`（秒）、`fragmentMs`、`audioPcm`、`avail`（可用性图路径）、
`waitTimeoutMs`、`grow`、`size`、`maxHeight`、`maxBandwidth`、`decodeMode`
（`auto|hardware|software`）、`threads`（解码线程数）。`decodeMode`/`threads`
只作用于该任务的子进程，不影响共享服务进程。

流就绪后先返回响应：

```json
{"jsonrpc":"2.0","id":7,"result":{
  "streamId":1,
  "meta":{"mime":"video/mp4; codecs=\"avc1.64001F\"", "tracks":["..."]}
}}
```

**响应一定先于该流的任何二进制帧与 `progress` 通知**；客户端在响应里登记
`streamId` 即可，不会和数据竞态。

之后 type-1 帧的 `stream_id = 1`。流以 `stream/end` 通知结束：

```json
{"jsonrpc":"2.0","method":"stream/end","params":{
  "streamId":1,"ok":true
}}
```

失败时：

```json
{"jsonrpc":"2.0","method":"stream/end","params":{
  "streamId":1,"ok":false,
  "error":{"code":1003,"message":"...","data":{"kind":"io","retryable":false,"exitCode":3}}
}}
```

播放头等进度通过 `progress` 通知发送，`requestId` 与 `stream` 请求一致。

#### `MEV1` 帧包

`mode = "frames"` 时，拼接后的二进制载荷由 `MEV1` 包组成：

```text
magic      4 字节   "MEV1"
kind       u32      0 = 视频 NV12，1 = 音频 S16LE
a          u32      视频：宽  | 音频：采样率
b          u32      视频：高  | 音频：声道数
pts_us     i64      显示时间戳（微秒）
len        u32      载荷长度
payload    len 字节
```

视频载荷是紧凑 NV12（`w*h` 亮度 + `w*(h/2)` 交织色度）；音频是交织 S16LE。
一个包可能跨多个二进制帧，客户端必须缓冲（官方 SDK 已处理）。如果只想原样
转发包流，给 SDK 传 `rawFrames: true`，它会直接输出 `{type:"frames", data}`
块而不解析。

### `cancel`

```json
{"jsonrpc":"2.0","id":8,"method":"cancel","params":{"requestId":7}}
```

或 `{"streamId":1}`。结果：`{"cancelled":true|false}`。取消会终止子进程；
Unix 先发 `SIGTERM`，让引擎删除写了一半的合并产物。原请求会以错误码 `1006`
（cancelled）结束，或对应流以取消错误结束。

### `shutdown`

返回 `{}` 并取消运行中的任务后退出。客户端也可以直接关闭 stdin，
EOF 视同 shutdown。

## 通知

| 方法 | 参数 |
| --- | --- |
| `progress` | `{requestId, type, data}`，`type` 是任务名（`mux`、`download`、`decode-progress`、`playhead` 等） |
| `log` | `{requestId, line}`，子进程 stderr 的非 JSON 日志行 |
| `stream/end` | `{streamId, ok, error?}` |

## 错误

JSON-RPC 传输错误使用标准码：

| 码 | 含义 |
| --- | --- |
| -32700 | 解析错误 |
| -32600 | 非法请求（含批量请求） |
| -32601 | 方法不存在 |
| -32602 | 参数非法 |

引擎错误使用：

| 码 | CLI 退出码 | `data.kind` |
| --- | --- | --- |
| 1002 | 2 | `unsupported`（用法错误或模块未编译） |
| 1003 | 3 | `io` / `container` / `bitstream`（输入不可用，不要重试） |
| 1004 | 4 | `protocol` / `need-more`（临时失败，可以重试） |
| 1005 | 5 | 引擎内部错误 |
| 1006 | 6 | `cancelled` |

`data` 还带 `retryable` 与 `exitCode`。引擎给中文错误文案；
机器判据是 `kind` 与 `retryable`。

## 并发

可以同时有多个请求。每个请求有自己的子进程；二进制帧用 `stream_id` 复用同一条
stdout。服务串行化 JSON 写入，帧不会交错。客户端可以同时开多条流。

## 安全

服务只用 stdin/stdout，不监听网络端口；访问控制由持有管道的进程决定。
所有媒体都按不可信输入处理；不在进程内跑任务正是为了崩溃隔离。

## JavaScript 客户端

官方零依赖客户端在 [`sdk/js`](sdk/js)：

```js
import { ZuvRust } from 'zuvrust-client'

const engine = await ZuvRust.start({ bin: '/opt/zuvrust/zuvrust' })

await engine.mux('out.mp4', ['video.mp4', 'audio.m4a'], {
  onProgress: (data, type) => console.log(type, data.percent)
})

for await (const event of engine.stream('movie.mp4', { mode: 'frames', audioPcm: true })) {
  if (event.type === 'video') render(event.data, event.width, event.height)
  if (event.type === 'audio') play(event.data, event.sampleRate, event.channels)
}

await engine.close()
```

完整客户端接口见 [`sdk/js/README.md`](sdk/js/README.md)。
