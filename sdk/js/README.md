# zuvrust-client

Official Node.js client for [ZuvRust](https://github.com/MochengCK/ZuvRust).
Zero dependencies, Node 18+.

It starts `zuvrust serve --stdio` and speaks the framed JSON-RPC protocol documented
in [API.md](https://github.com/MochengCK/ZuvRust/blob/main/API.md). Each job still
runs in its own child process, so a bad input cannot take down your application.

## Install

From this repository:

```sh
npm install /path/to/ZuvRust/sdk/js
```

The engine binary itself is distributed separately. Pass its path with `bin`:

```js
import { ZuvRust } from 'zuvrust-client'

const engine = await ZuvRust.start({ bin: '/opt/zuvrust/zuvrust' })
```

## Examples

```js
const info = await engine.probe('movie.mkv')
console.log(info.tracks)

await engine.mux('out.mp4', ['video.mp4', 'audio.m4a'], {
  fragmentMs: 2000,
  onProgress: (data, type) => console.log(type, data.percent)
})

for await (const event of engine.stream('movie.mp4', { mode: 'frames', audioPcm: true })) {
  if (event.type === 'video') draw(event.width, event.height, event.data)
  if (event.type === 'audio') play(event.sampleRate, event.channels, event.data)
  if (event.type === 'end' && !event.ok) console.error(event.error)
}

await engine.close()
```

## API

| Method | Description |
| --- | --- |
| `ZuvRust.start(options)` | Spawn the engine and complete `initialize`. |
| `version()` / `ping()` | Capability probe / health check. |
| `probe(input, options)` | Container and track information. |
| `mux(output, inputs, options)` | Merge / concatenate inputs. |
| `remux(output, input, options)` | Change container without re-encoding. |
| `download(input, options)` | HTTP / HLS / DASH download. |
| `decode(input, options)` | Decode video frames or audio to WAV. |
| `stream(input, options)` | Async iterable of fMP4 bytes or decoded frames. |
| `cancel(requestId)` | Cancel a running request. |
| `close()` | Shut down the engine process. |

All request methods accept `onProgress(data, type)` and an `AbortSignal`.
Errors are `ZuvRustError` with `code`, `kind`, `retryable` and `exitCode`.
Features that were not compiled into the engine (for example `decode` in a
`mux`-only build) fail with code `1002` and `kind: "unsupported"`.

See [API.md](https://github.com/MochengCK/ZuvRust/blob/main/API.md) for the wire
protocol and method parameters.

## License

Apache-2.0.
