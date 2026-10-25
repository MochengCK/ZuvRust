#!/usr/bin/env node
/**
 * SDK 自测：不需要样例媒体，覆盖启动/握手/能力、错误映射、流错误、取消与关闭。
 * 真正的 mux / fMP4 数据路径由 `crates/me-cli/tests/serve_contract.rs` 覆盖。
 *
 * 用法：
 *   node sdk/js/test/run.mjs [--bin <zuvrust>]
 *   ZUVRUST_BIN=/path/to/zuvrust npm test
 */

import { existsSync } from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { ZuvRust, ZuvRustError, StreamHandle } from '../index.js'

const require = createRequire(import.meta.url)
const sdkPkg = require('../package.json')

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(__dirname, '../../..')

const argBin = process.argv.includes('--bin')
  ? process.argv[process.argv.indexOf('--bin') + 1]
  : ''
const bin = argBin || process.env.ZUVRUST_BIN || path.join(repoRoot, 'target/debug/zuvrust')

let passed = 0
const failures = []

function ok (condition, name, detail = '') {
  if (condition) {
    passed++
    console.log(`  ✓ ${name}`)
  } else {
    failures.push(`${name}${detail ? ` — ${detail}` : ''}`)
    console.log(`  ✗ ${name}${detail ? ` — ${detail}` : ''}`)
  }
}

function eq (actual, expected, name) {
  ok(actual === expected, name, `expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`)
}

function mev1 (kind, a, b, ptsUs, data) {
  const head = Buffer.alloc(28)
  head.write('MEV1', 0, 4, 'ascii')
  head.writeUInt32LE(kind, 4)
  head.writeUInt32LE(a, 8)
  head.writeUInt32LE(b, 12)
  head.writeBigInt64LE(ptsUs, 16)
  head.writeUInt32LE(data.length, 24)
  return Buffer.concat([head, data])
}

async function expectError (promise, name, check) {
  try {
    await promise
    ok(false, name, 'expected an error')
  } catch (error) {
    ok(error instanceof ZuvRustError || error.name === 'AbortError', name, `${error && error.name}`)
    if (check) check(error)
  }
}

async function main () {
  // 帧解析的纯逻辑：不需要引擎，直接喂 MEV1 包。
  {
    const video = mev1(0, 640, 360, 123456n, Buffer.from([1, 2, 3, 4]))
    const audio = mev1(1, 48000, 2, 123456n, Buffer.from([5, 6]))
    const handle = new StreamHandle({ _streams: new Map(), _streamRequests: new Map() })
    // 故意拆成两段，验证跨 chunk 的粘包处理。
    handle._push(1, Buffer.concat([video, audio]).subarray(0, 31))
    handle._push(1, Buffer.concat([video, audio]).subarray(31))
    handle._endedByServer({ ok: true })
    const events = []
    for await (const event of handle) events.push(event)
    ok(events.some(e => e.type === 'video' && e.width === 640 && e.height === 360), 'SDK 解出视频帧')
    ok(events.some(e => e.type === 'audio' && e.sampleRate === 48000 && e.channels === 2), 'SDK 解出音频帧')
    eq(events.filter(e => e.type === 'end').length, 1, 'SDK 收到 end 事件')

    const fmp4 = new StreamHandle({ _streams: new Map(), _streamRequests: new Map() })
    fmp4._push(0, Buffer.from('fmp4-bytes'))
    fmp4._endedByServer({ ok: true })
    const fmp4Events = []
    for await (const event of fmp4) fmp4Events.push(event)
    eq(fmp4Events[0].type, 'fmp4', 'SDK 透传 fMP4 字节')

    const raw = new StreamHandle({ _streams: new Map(), _streamRequests: new Map() }, { rawFrames: true })
    raw._push(1, Buffer.concat([video, audio]))
    raw._endedByServer({ ok: true })
    const rawEvents = []
    for await (const event of raw) rawEvents.push(event)
    eq(rawEvents[0].type, 'frames', 'rawFrames 模式不透传解析后的帧')
    eq(rawEvents[0].data.length, video.length + audio.length, 'rawFrames 原样转发 MEV1 字节')
  }

  if (!existsSync(bin)) {
    console.error(`找不到引擎：${bin}\n先构建：cargo build -p me-cli --no-default-features --features mux`)
    process.exit(1)
  }

  console.log(`engine: ${bin}`)
  const engine = await ZuvRust.start({ bin })
  try {
    const init = await engine.initialize()
    eq(init.protocolVersion, 1, 'initialize 协议版本')
    ok(Array.isArray(init.engine.modules) && init.engine.modules.length > 0, 'initialize 带 modules')
    ok(init.engine.modules.includes('mux'), 'mux 构建至少含 mux 模块', JSON.stringify(init.engine.modules))

    const version = await engine.version()
    eq(version.name, 'ZuvRust', 'version 名称')
    eq(version.version, sdkPkg.version, 'SDK 与引擎版本一致（版本入口统一）')
    ok(Array.isArray(version.features), 'version 带 features')

    const pong = await engine.ping()
    ok(pong && typeof pong === 'object', 'ping 返回对象')

    await expectError(
      engine.probe('/definitely/not/here.mp4'),
      'probe 缺文件报错',
      error => {
        eq(error.code, 1003, 'probe 错误码 1003')
        eq(error.kind, 'io', 'probe 错误 kind=io')
        eq(error.retryable, false, 'probe 缺文件不可重试')
      }
    )

    const cancel = await engine.cancel(999999)
    eq(cancel.cancelled, false, '取消不存在的请求返回 false')

    // 未编 audio 时 decode --audio 必须是 1002 / unsupported；
    // 编了 audio 时缺文件是 1003（io / container / bitstream 都算输入不可用）。
    await expectError(
      engine.decode('/definitely/not/here.mp4', { audio: true }),
      'decode 错误映射',
      error => {
        ok(error.code === 1002 || error.code === 1003, 'decode 错误码 1002/1003', String(error.code))
        ok(
          ['unsupported', 'io', 'container', 'bitstream'].includes(error.kind),
          'decode kind',
          error.kind
        )
      }
    )

    const stream = engine.stream('/definitely/not/here.mp4', { mode: 'fmp4' })
    await expectError(
      (async () => {
        for await (const _ of stream) {
          // 不应该有数据
        }
      })(),
      'stream 缺文件在迭代时抛错',
      error => {
        eq(error.code, 1003, 'stream 错误码 1003')
      }
    )

    const controller = new AbortController()
    controller.abort()
    await expectError(
      engine.probe('/definitely/not/here.mp4', { signal: controller.signal }),
      '已中止的 AbortSignal 立即失败',
      error => eq(error.name, 'AbortError', '错误名 AbortError')
    )
  } finally {
    await engine.close()
  }

  eq(engine.engine !== null, true, 'close 后仍保留能力信息')
  console.log(`\n通过 ${passed} 项${failures.length ? `，失败 ${failures.length} 项` : '，全部通过'}`)
  if (failures.length) {
    for (const failure of failures) console.error(`  - ${failure}`)
    process.exit(1)
  }
}

main().catch(error => {
  console.error(error)
  process.exit(1)
})
