# ZuvRust

English | [简体中文](README-CN.md)

**A self-contained media engine for applications.** One executable, a stable
JSON-RPC API, and zero third-party media libraries.

ZuvRust implements container parsing, remuxing, muxing, AV1 software decoding,
H.264 hardware decoding and audio decoding itself. Applications talk to it over a
process boundary: start `zuvrust serve --stdio`, send JSON-RPC requests, receive
progress notifications and binary media frames.

- **Stable integration surface** — JSON-RPC 2.0, request IDs, cancellation,
  progress and binary streaming. See [API.md](API.md).
- **Official JavaScript client** — zero-dependency Node.js SDK in [sdk/js](sdk/js).
- **Crash isolation** — every job runs in its own child process; a malformed file
  cannot take down your application.
- **Modular builds** — compile only what your application needs: `audio`,
  `video`, `mux`, or any combination.
- **Apache-2.0** — permissive licensing for commercial and open-source products.

## What it does

| Capability | Details |
| --- | --- |
| Probe | Container, tracks, codecs, resolution, duration, bitrate |
| Mux | Merge audio/video inputs, concatenate segments, choose MP4 / MKV / WebM / TS / FLAC / ADTS |
| Remux | Change container without re-encoding |
| Download | HTTP(S), HLS (including AES-128), DASH |
| Decode video | AV1 software decoder; H.264 via VideoToolbox / Media Foundation / VA-API |
| Decode audio | AAC-LC, MP3, FLAC, Opus (self-implemented) |
| Stream | Fragmented MP4 for MSE, or decoded NV12 / PCM packets for custom players |
| Growing files | Play a file while another process is still downloading it |

ZuvRust does **not** encode or transcode.

## Integration

### 1. Start the service

```sh
zuvrust serve --stdio
```

All messages on stdin/stdout are length-prefixed frames. A request looks like
this (after the 5-byte frame header):

```json
{"jsonrpc":"2.0","id":1,"method":"probe","params":{"input":"movie.mkv"}}
```

The response contains container and track information. Long jobs emit
`progress` notifications; streams emit binary frames plus a `stream/end`
notification. The complete protocol, method parameters, binary `MEV1` packet
layout and error codes are in **[API.md](API.md)**.

### 2. Use the JavaScript SDK

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

See [sdk/js/README.md](sdk/js/README.md) for the full client API.

### 3. Or use the one-shot CLI

```sh
zuvrust probe movie.mkv --json
zuvrust mux out.mp4 video.mp4 audio.m4a --json --progress
zuvrust remux out.mkv in.ts --json
zuvrust download "https://example.com/index.m3u8" --out movie.mp4 --progress
zuvrust decode movie.mp4 --count 30 --out frames/
zuvrust decode music.flac --audio --out music.wav
```

The CLI contract (exit codes, NDJSON shapes, stderr encoding) is documented in
[DEVELOPMENT.md](DEVELOPMENT.md) and covered by process-level tests.

## Supported formats

**Containers:** MPEG-TS, ISO BMFF (mp4 / mov / m4s), Matroska / WebM,
Ogg, FLAC, ADTS, MPEG audio.

**Mux output** (chosen by the output extension; unsupported codec combinations
fail explicitly instead of writing a broken file):

| Extension | Container | Codecs |
| --- | --- | --- |
| `.mp4` `.m4s` `.mov` `.m4a` | fragmented MP4 | H.264 / H.265 / AV1 / VP9 + AAC / MP3 / FLAC / Opus / AC-3 |
| `.mkv` `.mka` | Matroska | all of the above |
| `.webm` | WebM subset | VP8 / VP9 / AV1 + Opus |
| `.ts` `.m2ts` | MPEG-TS | H.264 / H.265 + AAC / MP3 / AC-3 |
| `.flac` | raw FLAC | FLAC only |
| `.aac` | raw ADTS | AAC only |

**Protocols:** HTTP(S), HLS, DASH.

**Decoders**

| Codec | Implementation |
| --- | --- |
| AV1 | Self-implemented software decoder; VideoToolbox hardware decode on Apple Silicon M3+ |
| H.264 | System hardware decode on macOS (VideoToolbox), Windows (Media Foundation + D3D11), Linux (VA-API) |
| AAC-LC / MP3 / FLAC / Opus | Self-implemented software decoders |

AV1 software decoding is correct but not yet real-time on older machines:
roughly 50–55 ms per 1080p frame on a single Intel core (about 18–20 fps,
depending on frequency and thermals). Apple Silicon uses hardware decoding.
Details and the optimization log are in [DEVELOPMENT.md](DEVELOPMENT.md).

**BT / magnet links are out of scope.** ZuvRust only plays or muxes files produced
by a download engine. The handoff is [the availability map](API.md#stream): the
downloader writes which byte ranges exist; ZuvRust waits for missing ranges
instead of treating them as corrupt data, and reports its playhead back.

## Module builds

The engine is split into three independently compiled modules:

| Feature | Includes | Typical application |
| --- | --- | --- |
| `audio` | AAC / MP3 / FLAC / Opus decoders, `decode --audio` | audio players, converters |
| `video` | AV1 software decode + H.264 hardware decode, `decode`, `stream --video=frames` | video players |
| `mux` | probe / mux / remux / fMP4 stream remuxing | downloaders, merging tools |

Any combination is valid. `probe`, `download`, `play` and `serve` are always
included. `zuvrust version --json` reports the compiled modules in `modules`;
calling a command whose module was not compiled returns exit code 2 with a clear
message.

```sh
./scripts/build-modules.sh --release --modules=mux
./scripts/build-modules.sh --release --modules=audio,video
./scripts/build-modules.sh --release --modules=audio,video,mux
```

The default release artifacts are built with `mux` only for the Lerxu desktop
application (its playback UI is temporarily disabled). For the full engine, build
from source or run the CI workflow manually with `modules=audio,video,mux`.

## Build from source

Requirements: Rust stable. Linux ARM64 cross builds additionally use
`cargo-zigbuild`.

```sh
cargo build --release                 # all modules
./scripts/build-modules.sh --release  # validated module build wrapper
cargo test --workspace
./scripts/check-arches.sh             # cross-architecture compile gate
```

The binary is `target/release/zuvrust` (`zuvrust.exe` on Windows).

## Platforms

| Platform | Artifact |
| --- | --- |
| macOS Intel | `zuvrust-darwin-x86_64.tar.gz` |
| macOS Apple Silicon | `zuvrust-darwin-arm64.tar.gz` |
| Linux x86_64 | `zuvrust-linux-x86_64.tar.gz` |
| Linux arm64 | `zuvrust-linux-arm64.tar.gz` (static) |
| Windows x86_64 | `zuvrust-windows-x86_64.tar.gz` |

Each artifact is built and tested natively in CI, then downloaded and verified as
a binary (architecture, module list, startup and contract tests) before release.
Archives include the binary, `LICENSE`, `NOTICE`, English and Chinese READMEs,
the protocol reference, and the `sdk/js` client source.

## Design notes

- Container parsing and muxing are byte-level operations; they never decode.
- Decoding is a separate layer, optional at build time.
- Timestamps use integer timebases and are converted to microseconds only when
  comparing across tracks.
- Output is either correct or an explicit error; the engine never silently writes
  a broken file.
- The JSON-RPC service supervises child jobs, preserving process isolation while
  providing one integration endpoint.

More internal invariants, performance data and engineering notes:
[DEVELOPMENT.md](DEVELOPMENT.md).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

You can use ZuvRust in commercial and open-source applications under the terms of
Apache-2.0. The engine contains no third-party media library code.
