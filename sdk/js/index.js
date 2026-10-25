/**
 * ZuvRust 官方 Node.js 客户端。
 *
 * 它只做一件事：把 `zuvrust serve --stdio` 的分帧 JSON-RPC 协议封成
 * Promise / AsyncIterable。零依赖，Node >= 18。
 *
 * 传输帧（小端）：`u32 payload_len + u8 type + payload`。
 * type=0 是 JSON-RPC；type=1 是二进制流（`u32 streamId + u32 kind + bytes`）。
 * 完整协议见仓库根目录的 API.md。
 */

import { spawn } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const { version: CLIENT_VERSION } = require('./package.json')

const TYPE_JSON = 0
const TYPE_BINARY = 1
const MAX_JSON_BYTES = 8 << 20
const MAX_BINARY_BYTES = 64 << 20
const KIND_FMP4 = 0
const KIND_FRAMES = 1
const FRAME_HEADER_BYTES = 28
const FRAME_MAGIC = Buffer.from('MEV1')

const defaultSleep = ms => new Promise(resolve => setTimeout(resolve, ms))

function abortError () {
  const err = new Error('The operation was aborted')
  err.name = 'AbortError'
  return err
}

/** 引擎错误：code 是 JSON-RPC 码，kind/retryable/exitCode 来自引擎。 */
export class ZuvRustError extends Error {
  constructor (message, { code = 0, data = null } = {}) {
    super(message || 'ZuvRust error')
    this.name = 'ZuvRustError'
    this.code = code
    this.data = data
    this.kind = data && data.kind ? data.kind : ''
    this.retryable = !!(data && data.retryable)
    this.exitCode = data && Number.isInteger(data.exitCode) ? data.exitCode : null
  }
}

function rpcError (error) {
  return new ZuvRustError(error && error.message, {
    code: error && Number.isInteger(error.code) ? error.code : 0,
    data: error && error.data ? error.data : null
  })
}

/**
 * 一个正在推送二进制数据的流。
 *
 * 用 `for await` 消费：
 *   - `{ type:'meta', meta }`
 *   - `{ type:'progress', name, data }`
 *   - fMP4：`{ type:'fmp4', data: Buffer }`
 *   - frames：`{ type:'video'|'audio', ptsUs, data, ... }`
 *   - `{ type:'end', ok, error? }`
 */
export class StreamHandle {
  constructor (client, { rawFrames = false } = {}) {
    this.client = client
    this.rawFrames = rawFrames
    this.requestId = null
    this.streamId = null
    this.meta = null
    this._queue = []
    this._waiters = []
    this._ended = false
    this._error = null
    this._frameBuf = Buffer.alloc(0)
  }

  _started (result) {
    this.streamId = result.streamId
    this.meta = result.meta
    this._pushEvent({ type: 'meta', meta: result.meta })
  }

  _progress (params) {
    this._pushEvent({ type: 'progress', name: params.type, data: params.data })
  }

  _push (kind, data) {
    if (kind === KIND_FMP4) {
      this._pushEvent({ type: 'fmp4', data: Buffer.from(data) })
      return
    }
    if (kind !== KIND_FRAMES) return
    if (this.rawFrames) {
      this._pushEvent({ type: 'frames', data: Buffer.from(data) })
      return
    }
    this._frameBuf = Buffer.concat([this._frameBuf, data])
    while (this._frameBuf.length >= FRAME_HEADER_BYTES) {
      if (!this._frameBuf.subarray(0, 4).equals(FRAME_MAGIC)) {
        this._fail(new ZuvRustError('invalid MEV1 frame from engine'))
        return
      }
      const packetKind = this._frameBuf.readUInt32LE(4)
      const a = this._frameBuf.readUInt32LE(8)
      const b = this._frameBuf.readUInt32LE(12)
      const ptsUs = Number(this._frameBuf.readBigInt64LE(16))
      const len = this._frameBuf.readUInt32LE(24)
      if (this._frameBuf.length < FRAME_HEADER_BYTES + len) return
      const payload = Buffer.from(
        this._frameBuf.subarray(FRAME_HEADER_BYTES, FRAME_HEADER_BYTES + len)
      )
      this._frameBuf = this._frameBuf.subarray(FRAME_HEADER_BYTES + len)
      if (packetKind === 0) {
        this._pushEvent({ type: 'video', width: a, height: b, ptsUs, data: payload })
      } else {
        this._pushEvent({ type: 'audio', sampleRate: a, channels: b, ptsUs, data: payload })
      }
    }
  }

  _endedByServer (params) {
    const error = params && params.ok === false && params.error
      ? rpcError(params.error)
      : null
    if (error) this._error = error
    this._ended = true
    this._pushEvent({
      type: 'end',
      ok: !!(params && params.ok),
      error: error || undefined
    })
    this.client._streams.delete(this.streamId)
    this.client._streamRequests.delete(this.requestId)
  }

  _fail (error) {
    this._error = error
    this._ended = true
    this._queue.length = 0
    if (this.streamId != null) this.client._streams.delete(this.streamId)
    this.client._streamRequests.delete(this.requestId)
    this._drain()
  }

  _pushEvent (event) {
    this._queue.push(event)
    this._drain()
  }

  _drain () {
    while (this._waiters.length > 0 && this._queue.length > 0) {
      this._waiters.shift().resolve({ value: this._queue.shift(), done: false })
    }
    if (this._ended && this._queue.length === 0) {
      while (this._waiters.length > 0) {
        const waiter = this._waiters.shift()
        if (this._error) waiter.reject(this._error)
        else waiter.resolve({ value: undefined, done: true })
      }
    }
  }

  next () {
    if (this._queue.length > 0) {
      return Promise.resolve({ value: this._queue.shift(), done: false })
    }
    if (this._ended) {
      return this._error
        ? Promise.reject(this._error)
        : Promise.resolve({ value: undefined, done: true })
    }
    return new Promise((resolve, reject) => {
      this._waiters.push({ resolve, reject })
    })
  }

  return () {
    return this.cancel().then(() => ({ value: undefined, done: true }))
  }

  [Symbol.asyncIterator] () {
    return this
  }

  /** 取消这条流；返回 `{cancelled:boolean}`。 */
  cancel () {
    if (this.requestId == null || this._ended) return Promise.resolve({ cancelled: false })
    return this.client.cancel(this.requestId)
  }
}

/**
 * 常驻引擎客户端。
 *
 * 用 `await ZuvRust.start({ bin })` 启动；它会拉起 `zuvrust serve --stdio`
 * 并完成 initialize。一个客户端可以并发跑多个请求；每个任务在引擎侧仍是
 * 独立子进程（崩溃隔离）。
 */
export class ZuvRust extends EventEmitter {
  constructor ({
    bin = 'zuvrust',
    args = [],
    env = process.env,
    stderr = 'pipe',
    windowsHide = true
  } = {}) {
    super()
    this.bin = bin
    this.args = args
    this._closed = false
    this._closing = false
    this._exited = false
    this._buffer = Buffer.alloc(0)
    this._nextId = 1
    this._pending = new Map()
    this._streams = new Map()
    this._streamRequests = new Map()
    this._stderrTail = ''
    this.engine = null

    this._child = spawn(bin, [...args, 'serve', '--stdio'], {
      stdio: ['pipe', 'pipe', stderr === 'inherit' ? 'inherit' : 'pipe'],
      env,
      windowsHide
    })
    this._child.stdout.on('data', chunk => this._onData(chunk))
    this._child.stdin.on('error', error => {
      this._failAll(new ZuvRustError(`engine stdin: ${error.message}`))
    })
    this._child.on('error', error => {
      this._failAll(new ZuvRustError(`engine process: ${error.message}`))
    })
    if (stderr !== 'inherit') {
      this._child.stderr.on('data', chunk => this._onStderr(chunk))
    }
    this._exitPromise = new Promise(resolve => {
      this._child.once('exit', (code, signal) => {
        this._exited = true
        this._onExit(code, signal)
        resolve()
      })
    })
    this._ready = new Promise((resolve, reject) => {
      this._child.once('spawn', resolve)
      this._child.once('error', reject)
    })
    this._ready.catch(() => {})
  }

  /** 启动并完成 initialize。 */
  static async start (options = {}) {
    const client = new ZuvRust(options)
    await client._ready
    await client.initialize(options.timeoutMs)
    return client
  }

  async initialize (timeoutMs = 10000) {
    const result = await this._withTimeout(
      this._request('initialize', {
        protocolVersion: 1,
        client: { name: 'zuvrust-client', version: CLIENT_VERSION }
      }),
      timeoutMs,
      'initialize'
    )
    if (result.protocolVersion !== 1) {
      throw new ZuvRustError(`unsupported protocol version ${result.protocolVersion}`)
    }
    this.engine = result.engine
    return result
  }

  version () {
    return this._request('version', {})
  }

  ping () {
    return this._request('ping', {})
  }

  probe (input, options = {}) {
    const { onProgress, signal, ...params } = options
    return this._request('probe', { input, ...params }, { onProgress, signal })
  }

  mux (output, inputs, options = {}) {
    const { onProgress, signal, ...params } = options
    return this._request('mux', { output, inputs, ...params }, { onProgress, signal })
  }

  remux (output, input, options = {}) {
    const { onProgress, signal, ...params } = options
    return this._request('remux', { output, input, ...params }, { onProgress, signal })
  }

  download (input, options = {}) {
    const { onProgress, signal, ...params } = options
    return this._request('download', { input, ...params }, { onProgress, signal })
  }

  decode (input, options = {}) {
    const { onProgress, signal, ...params } = options
    return this._request('decode', { input, ...params }, { onProgress, signal })
  }

  /** 开一条流；返回可 `for await` 的 StreamHandle。 */
  stream (input, options = {}) {
    const { onProgress, signal, rawFrames = false, ...params } = options
    const handle = new StreamHandle(this, { rawFrames })
    const promise = this._request(
      'stream',
      { input, ...params },
      { onProgress, signal, stream: handle }
    )
    promise.catch(error => handle._fail(error))
    return handle
  }

  cancel (requestId) {
    return this._request('cancel', { requestId })
  }

  async close () {
    if (this._closed || this._closing) return
    this._closing = true
    try {
      await this._withTimeout(this._request('shutdown', {}), 1000, 'shutdown')
    } catch (_) {}
    try { this._child.stdin.end() } catch (_) {}
    await Promise.race([this._exitPromise, defaultSleep(1000)])
    if (!this._exited) {
      try { this._child.kill() } catch (_) {}
      await Promise.race([this._exitPromise, defaultSleep(1000)])
    }
    this._closed = true
  }

  _withTimeout (promise, ms, label) {
    if (!ms || ms <= 0) return promise
    return Promise.race([
      promise,
      defaultSleep(ms).then(() => {
        throw new ZuvRustError(`${label} timed out after ${ms}ms`)
      })
    ])
  }

  _request (method, params = {}, { onProgress = null, signal = null, stream = null } = {}) {
    if (this._closed) {
      return Promise.reject(new ZuvRustError('engine client is closed'))
    }
    const id = this._nextId++
    if (stream) stream.requestId = id
    return new Promise((resolve, reject) => {
      const entry = {
        id,
        method,
        onProgress,
        stream,
        settled: false,
        resolve: value => settle(resolve, value),
        reject: error => settle(reject, error)
      }
      let cleanup = null
      const settle = (fn, value) => {
        if (entry.settled) return
        entry.settled = true
        this._pending.delete(id)
        if (cleanup) cleanup()
        fn(value)
      }
      this._pending.set(id, entry)
      try {
        this._writeJson({ jsonrpc: '2.0', id, method, params })
      } catch (error) {
        settle(reject, error)
        return
      }
      if (signal) {
        if (signal.aborted) {
          this.cancel(id).catch(() => {})
          settle(reject, abortError())
          if (stream) stream._fail(abortError())
          return
        }
        const onAbort = () => {
          this.cancel(id).catch(() => {})
          settle(reject, abortError())
          if (stream) stream._fail(abortError())
        }
        signal.addEventListener('abort', onAbort, { once: true })
        cleanup = () => signal.removeEventListener('abort', onAbort)
      }
    })
  }

  _writeJson (value) {
    const payload = Buffer.from(JSON.stringify(value), 'utf8')
    if (payload.length > MAX_JSON_BYTES) {
      throw new ZuvRustError('JSON frame too large')
    }
    const head = Buffer.allocUnsafe(5)
    head.writeUInt32LE(payload.length, 0)
    head.writeUInt8(TYPE_JSON, 4)
    this._child.stdin.write(Buffer.concat([head, payload]))
  }

  _onData (chunk) {
    this._buffer = Buffer.concat([this._buffer, chunk])
    while (this._buffer.length >= 5) {
      const length = this._buffer.readUInt32LE(0)
      const type = this._buffer.readUInt8(4)
      const max = type === TYPE_JSON ? MAX_JSON_BYTES : MAX_BINARY_BYTES
      if (length > max) {
        this._failAll(new ZuvRustError(`frame length ${length} exceeds limit ${max}`))
        return
      }
      if (this._buffer.length < 5 + length) return
      const payload = this._buffer.subarray(5, 5 + length)
      this._buffer = this._buffer.subarray(5 + length)
      if (type === TYPE_JSON) {
        let message
        try {
          message = JSON.parse(payload.toString('utf8'))
        } catch (error) {
          this._failAll(new ZuvRustError(`invalid JSON from engine: ${error.message}`))
          return
        }
        this._onMessage(message)
      } else if (type === TYPE_BINARY) {
        if (payload.length < 8) continue
        const streamId = payload.readUInt32LE(0)
        const kind = payload.readUInt32LE(4)
        const data = payload.subarray(8)
        const handle = this._streams.get(streamId)
        if (handle) handle._push(kind, data)
      }
    }
  }

  _onMessage (message) {
    if (message && typeof message.method === 'string') {
      const params = message.params || {}
      if (message.method === 'progress') {
        const stream = this._streamRequests.get(params.requestId)
        if (stream) stream._progress(params)
        const entry = this._pending.get(params.requestId)
        if (entry && entry.onProgress) entry.onProgress(params.data, params.type)
        this.emit('progress', params)
        return
      }
      if (message.method === 'stream/end') {
        const handle = this._streams.get(params.streamId)
        if (handle) handle._endedByServer(params)
        return
      }
      if (message.method === 'log') {
        const line = params.line || ''
        if (line) this.emit('log', line)
        return
      }
      this.emit('notification', message)
      return
    }
    if (!message || !Object.prototype.hasOwnProperty.call(message, 'id')) return
    const entry = this._pending.get(message.id)
    if (!entry) return
    if (message.error) {
      entry.reject(rpcError(message.error))
      return
    }
    if (entry.stream) {
      this._streams.set(message.result.streamId, entry.stream)
      this._streamRequests.set(message.id, entry.stream)
      entry.stream._started(message.result)
    }
    entry.resolve(message.result)
  }

  _onStderr (chunk) {
    const text = chunk.toString('utf8')
    this._stderrTail = (this._stderrTail + text).slice(-8192)
    for (const line of text.split('\n')) {
      if (line.trim()) this.emit('stderr', line)
    }
  }

  _onExit (code, signal) {
    const error = new ZuvRustError(
      `engine exited${code == null ? '' : ` with code ${code}`}${signal ? ` (${signal})` : ''}`,
      { code: 1005, data: { kind: 'internal', retryable: false, exitCode: code } }
    )
    this._failAll(error)
    this.emit('close', { code, signal })
  }

  _failAll (error) {
    for (const entry of this._pending.values()) entry.reject(error)
    this._pending.clear()
    for (const handle of this._streams.values()) handle._fail(error)
    this._streams.clear()
    this._streamRequests.clear()
  }
}

export default ZuvRust
