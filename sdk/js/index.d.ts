export interface ZuvRustErrorData {
  kind?: string
  retryable?: boolean
  exitCode?: number | null
}

export class ZuvRustError extends Error {
  code: number
  data: ZuvRustErrorData | null
  kind: string
  retryable: boolean
  exitCode: number | null
  constructor(message: string, options?: { code?: number; data?: ZuvRustErrorData | null })
}

export interface VersionInfo {
  name: string
  version: string
  modules: string[]
  containers: string[]
  protocols: string[]
  features: string[]
  streaming: {
    mode: string
    downloadEngine: string
    playheadHint: boolean
  }
}

export interface InitializeResult {
  protocolVersion: number
  engine: VersionInfo
  framing: Record<string, unknown>
  binaryKinds: { fmp4: number; frames: number }
  methods: string[]
  notifications: string[]
  isolation: string
}

export interface ClientOptions {
  /** Executable name or path. Default: `zuvrust` (PATH). */
  bin?: string
  /** Extra global args placed before `serve --stdio`. */
  args?: string[]
  env?: Record<string, string | undefined>
  /** `pipe` (default, emitted as `stderr` events) or `inherit`. */
  stderr?: 'pipe' | 'inherit'
  windowsHide?: boolean
  /** Initialize timeout. Default: 10000 ms. */
  timeoutMs?: number
}

export interface RequestOptions {
  onProgress?: (data: Record<string, unknown>, type: string) => void
  signal?: AbortSignal
}

export interface ProbeOptions extends RequestOptions {
  grow?: boolean
  size?: number
  maxHeight?: number
  maxBandwidth?: number
}

export interface MuxOptions extends RequestOptions {
  fragmentMs?: number
  tracks?: 'both' | 'audio' | 'video'
}

export interface RemuxOptions extends MuxOptions {}

export interface DownloadOptions extends RequestOptions {
  output?: string
  maxHeight?: number
  maxBandwidth?: number
}

export interface DecodeOptions extends RequestOptions {
  audio?: boolean
  out?: string
  count?: number
  seek?: number
  noGrow?: boolean
  /** Decoder preference: auto (default), hardware-only, software-only. */
  decodeMode?: 'auto' | 'hardware' | 'software'
  /** Decoder thread count; omitted/0 = engine default. */
  threads?: number
}

export interface StreamOptions extends RequestOptions {
  /** `fmp4` (default) or decoded frame stream (`MEV1` packets). */
  mode?: 'fmp4' | 'frames'
  seek?: number
  fragmentMs?: number
  audioPcm?: boolean
  /** Forward raw MEV1 chunks instead of parsing them into video/audio events. */
  rawFrames?: boolean
  avail?: string
  waitTimeoutMs?: number
  grow?: boolean
  size?: number
  maxHeight?: number
  maxBandwidth?: number
  /** Decoder preference: auto (default), hardware-only, software-only. */
  decodeMode?: 'auto' | 'hardware' | 'software'
  /** Decoder thread count; omitted/0 = engine default. */
  threads?: number
}

export interface MetaEvent {
  type: 'meta'
  meta: Record<string, unknown>
}

export interface ProgressEvent {
  type: 'progress'
  name: string
  data: Record<string, unknown>
}

export interface Fmp4Event {
  type: 'fmp4'
  data: Buffer
}

export interface RawFramesEvent {
  type: 'frames'
  data: Buffer
}

export interface VideoEvent {
  type: 'video'
  width: number
  height: number
  ptsUs: number
  data: Buffer
}

export interface AudioEvent {
  type: 'audio'
  sampleRate: number
  channels: number
  ptsUs: number
  data: Buffer
}

export interface EndEvent {
  type: 'end'
  ok: boolean
  error?: ZuvRustError
}

export type StreamEvent =
  | MetaEvent
  | ProgressEvent
  | Fmp4Event
  | RawFramesEvent
  | VideoEvent
  | AudioEvent
  | EndEvent

export class StreamHandle implements AsyncIterable<StreamEvent> {
  readonly meta: Record<string, unknown> | null
  readonly streamId: number | null
  readonly requestId: number | null
  next(): Promise<IteratorResult<StreamEvent>>
  return(): Promise<IteratorResult<StreamEvent>>
  cancel(): Promise<{ cancelled: boolean }>
  [Symbol.asyncIterator](): AsyncIterableIterator<StreamEvent>
}

export class ZuvRust {
  readonly bin: string
  readonly args: string[]
  engine: VersionInfo | null

  constructor(options?: ClientOptions)
  static start(options?: ClientOptions): Promise<ZuvRust>

  initialize(timeoutMs?: number): Promise<InitializeResult>
  version(): Promise<VersionInfo>
  ping(): Promise<Record<string, never>>
  probe(input: string, options?: ProbeOptions): Promise<Record<string, unknown>>
  mux(output: string, inputs: string[], options?: MuxOptions): Promise<Record<string, unknown>>
  remux(output: string, input: string, options?: RemuxOptions): Promise<Record<string, unknown>>
  download(input: string, options?: DownloadOptions): Promise<Record<string, unknown>>
  decode(input: string, options?: DecodeOptions): Promise<Record<string, unknown>>
  stream(input: string, options?: StreamOptions): StreamHandle
  cancel(requestId: number): Promise<{ cancelled: boolean }>
  close(): Promise<void>

  on(event: 'progress', listener: (params: Record<string, unknown>) => void): this
  on(event: 'log', listener: (line: string) => void): this
  on(event: 'stderr', listener: (line: string) => void): this
  on(event: 'notification', listener: (message: Record<string, unknown>) => void): this
  on(event: 'close', listener: (info: { code: number | null; signal: string | null }) => void): this
}

export default ZuvRust
