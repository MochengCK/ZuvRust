# ZuvRust API

English | [简体中文](API-CN.md)

This document is the integration contract for the `zuvrust` media engine. It covers
the long-lived JSON-RPC service (`zuvrust serve --stdio`), the binary framing used
for media streams, and the error model shared by every method.

The one-shot CLI (`zuvrust probe ...`, `zuvrust mux ...`) remains supported. The
service exists so an application can keep one supervisor process, send concurrent
requests, receive progress, and cancel work without rebuilding command lines.

## Design

`serve` is a **supervisor**, not a shared in-process decoder. Every job runs in a
child `zuvrust` process:

- a malformed input that crashes a container parser produces a JSON-RPC error, not
  an application crash;
- jobs can run concurrently without sharing decoder state;
- cancelling a job terminates its child (graceful `SIGTERM` first on Unix);
- the protocol process itself contains no media code and is available in every
  build.

The cost is one process start per job (milliseconds). For probe/mux/remux/download
this is negligible; long-lived streaming is one process per stream.

## Transport

The service speaks JSON-RPC 2.0 over stdin/stdout using length-prefixed frames.
It does **not** open a network port.

All integers are little-endian.

```text
frame   := u32 payload_len
           u8  type
           u8[payload_len] payload

type 0  := JSON-RPC message (UTF-8 JSON)
type 1  := binary stream chunk:
           u32 stream_id
           u32 kind
           u8[payload_len - 8] data
```

Limits: JSON frames 8 MiB, binary frames 64 MiB. Requests must use type 0.
Responses and notifications use type 0; stream data uses type 1.

The first request should be `initialize`. The response describes the engine,
compiled modules and framing constants.

## Methods

| Method | Purpose |
| --- | --- |
| `initialize` | Handshake: protocol version, engine version, modules, capabilities. |
| `version` | Same capability object as `zuvrust version --json`. |
| `ping` | Health check. |
| `probe` | Container and track information. |
| `mux` | Merge/concatenate inputs into one file. |
| `remux` | Change container without re-encoding. |
| `download` | Download HTTP/HLS/DASH. |
| `decode` | Decode video to frames or audio to WAV. |
| `stream` | Stream fMP4 bytes or decoded frames. |
| `cancel` | Cancel a request or stream. |
| `shutdown` | Stop the service. |

Methods with a request `id` always receive exactly one response. Notifications
(messages without `id`) are not acknowledged. Batch requests are not supported.

### `initialize`

Request:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{
  "protocolVersion":1,
  "client":{"name":"my-app","version":"1.0.0"}
}}
```

Result:

```json
{
  "protocolVersion": 1,
  "engine": { "...": "same object as version --json" },
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

`engine.modules` is authoritative for what this binary can do. A `mux`-only build
does not include `audio` or `video`; calling `decode` on it fails with code `1002`
and `kind: "unsupported"`.

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

All fields except `input` are optional. `grow` is for files that are still being
written (downloads). The result is the same object as the CLI `probe --json`
result: `container`, `durationUs`, `tracks[]`, `bitrate`, `source`.

### `mux`

```json
{"jsonrpc":"2.0","id":3,"method":"mux","params":{
  "output":"/media/out.mp4",
  "inputs":["/media/video.mp4","/media/audio.m4a"],
  "fragmentMs":2000,
  "tracks":"both"
}}
```

`inputs` must be a non-empty array. Same-role inputs are concatenated in the given
order. `fragmentMs` controls fragment duration. `tracks` is `both` (default),
`audio` or `video`. Progress is emitted as `progress` notifications; the final
result is the same object as the CLI `mux --json` result.

### `remux`

```json
{"jsonrpc":"2.0","id":4,"method":"remux","params":{
  "output":"/media/out.mkv","input":"/media/in.ts",
  "fragmentMs":2000,"tracks":"both"
}}
```

`output`, `input` are required. `fragmentMs` and `tracks` are optional and have
the same meaning as in `mux`. Progress notifications use type `mux`.

### `download`

```json
{"jsonrpc":"2.0","id":5,"method":"download","params":{
  "input":"https://example.com/index.m3u8",
  "output":"/media/movie.mp4",
  "maxHeight":1080
}}
```

`output` is optional; when omitted the engine derives a name from the URL.
DASH inputs produce one file per track and report them in the result.
Progress notifications use type `download`.

### `decode`

```json
{"jsonrpc":"2.0","id":6,"method":"decode","params":{
  "input":"/media/movie.mp4",
  "count":30,
  "out":"/media/frames",
  "seek":12.5
}}
```

Video mode writes `frame-000001.nv12` files plus `frames.json` into `out`.
Audio mode (`"audio": true`, `"out": "/media/audio.wav"`) writes a WAV.
`noGrow` disables growing-file reads. `decodeMode` is `auto` (default),
`hardware` (hardware only) or `software` (never touch hardware); `threads`
sets the decoder thread count. Progress notifications use type
`decode-progress`.

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

`mode` is `fmp4` (default) or `frames`:

- `fmp4`: the engine remuxes to fragmented MP4. Binary chunks have `kind = 0`.
  The result `meta.mime` and `meta.tracks` describe what to feed to a player.
- `frames`: the engine decodes video to NV12 and optionally audio to PCM.
  Binary chunks have `kind = 1` and contain `MEV1` packets (below). Use
  `"audioPcm": true` to include audio.

Optional fields: `mode`, `seek` (seconds), `fragmentMs`, `audioPcm`, `avail`
(availability bitmap path), `waitTimeoutMs`, `grow`, `size`, `maxHeight`,
`maxBandwidth`, `decodeMode` (`auto|hardware|software`), `threads` (decoder
thread count). `decodeMode`/`threads` are applied to the job child process; the
shared service process itself is unaffected.

The response is sent as soon as the stream is ready:

```json
{"jsonrpc":"2.0","id":7,"result":{
  "streamId":1,
  "meta":{"mime":"video/mp4; codecs=\"avc1.64001F\"", "tracks":["..."]}
}}
```

After that, type-1 frames carry `stream_id = 1`. The stream ends with a
`stream/end` notification:

```json
{"jsonrpc":"2.0","method":"stream/end","params":{
  "streamId":1,"ok":true
}}
```

On failure:

```json
{"jsonrpc":"2.0","method":"stream/end","params":{
  "streamId":1,"ok":false,
  "error":{"code":1003,"message":"...","data":{"kind":"io","retryable":false,"exitCode":3}}
}}
```

Playhead updates and other progress arrive as `progress` notifications whose
`requestId` matches the `stream` request.

#### `MEV1` frame packet

When `mode = "frames"`, the concatenated binary payload contains `MEV1` packets:

```text
magic      4 bytes  "MEV1"
kind       u32      0 = video NV12, 1 = audio S16LE
a          u32      video: width   | audio: sample rate
b          u32      video: height  | audio: channels
pts_us     i64      presentation timestamp in microseconds
len        u32      payload length
payload    len bytes
```

Video payload is compact NV12 (`w*h` luma + `w*(h/2)` interleaved chroma).
Audio payload is interleaved S16LE. A chunk may split a packet; clients must
buffer across binary frames (the SDK does this). If you only need to forward the
packet stream unchanged, pass `rawFrames: true` to the SDK and it will emit
`{type:"frames", data}` chunks without parsing.

### `cancel`

```json
{"jsonrpc":"2.0","id":8,"method":"cancel","params":{"requestId":7}}
```

or `{"streamId":1}`. Result: `{"cancelled":true|false}`. Cancelling a job
terminates its child; on Unix the engine receives `SIGTERM` first so it can remove
half-written mux output. The original request resolves with code `1006`
(cancelled) or its stream ends with a cancelled error.

### `shutdown`

Returns `{}` and exits after cancelling running jobs. Clients may also close stdin;
EOF is treated as shutdown.

## Notifications

| Method | Params |
| --- | --- |
| `progress` | `{requestId, type, data}` — `type` is the task name (`mux`, `download`, `decode-progress`, `playhead`, ...) |
| `log` | `{requestId, line}` — non-JSON stderr lines from the job |
| `stream/end` | `{streamId, ok, error?}` |

## Errors

JSON-RPC transport errors use standard codes:

| Code | Meaning |
| --- | --- |
| -32700 | Parse error |
| -32600 | Invalid request (including batch requests) |
| -32601 | Method not found |
| -32602 | Invalid params |

Engine errors use:

| Code | CLI exit | `data.kind` |
| --- | --- | --- |
| 1002 | 2 | `unsupported` (usage or module not compiled) |
| 1003 | 3 | `io` / `container` / `bitstream` (bad input, do not retry) |
| 1004 | 4 | `protocol` / `need-more` (transient, retry is meaningful) |
| 1005 | 5 | internal engine error |
| 1006 | 6 | `cancelled` |

`data` also contains `retryable` and `exitCode`. Error messages come from the
engine and are currently Chinese; `kind` and `retryable` are the stable machine
contract.

## Concurrency

Multiple requests may be in flight at once. Each request has its own child
process; binary frames are multiplexed by `stream_id`. JSON writes are serialized
by the service, so frames never interleave. A client may have multiple streams
open at once.

## Security

The service only uses stdin/stdout and never listens on a network port. Access
control is whatever owns the process pipes. The engine treats all media as
untrusted input; crash isolation is the reason jobs are not run in-process.

## JavaScript client

The official zero-dependency client lives in [`sdk/js`](sdk/js):

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

See [`sdk/js/README.md`](sdk/js/README.md) for the full client surface.
