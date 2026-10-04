# av1probe —— 用 headless Chrome 造「参照 AV1 流」

本项目没有 ffmpeg / dav1d / aomdec，但**有 Chrome**。Chrome 的 WebCodecs `VideoEncoder`
在 macOS 上走 libaom 软件编码 —— 于是可以自己造一批**真实编码器写出来的** AV1 流，
用来验证自研解码器（`crates/me-decode`）的语法实现。

## 用法

```sh
cd ZuvRust/tools/av1probe
python3 serve.py /tmp/av1out 8799 &            # 收结果的本地服务（页面也从这里发）
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
rm -rf /tmp/pX; ( "$CHROME" --headless=new --disable-gpu --no-first-run \
    --user-data-dir=/tmp/pX --dump-dom "http://127.0.0.1:8799/gen.html" & ); sleep 20
cd ../.. && AV1_PROBE_DIR=/tmp/av1out ~/.cargo/bin/cargo test -p me-cli \
    --bin zuvrust probe_obu_streams_step_up_complexity -- --ignored --nocapture
```

想要更多样本就改 `gen.html` 里的 `levels`（`[级别号, 宽, 高, 图案]`），图案实现在 `draw()`。

## 已跑出来的结论（2026-09-30）

- `#808080`（Y=128 恰好等于帧内 DC 预测值 128 ⇒ **残差恰好全零**）在 16/32/64 三个尺寸上
  **`exit_symbol` 全部通过** ⇒ 分区树、`mode_info`、`HasChroma`、`all_zero=1` 那条路径
  **被真实编码器的输出证明是正确的**。
- **残差非零的流现在也全部通过了**：14 条参照流每一瓦片都 ① `exit_symbol` 通过、
  ② 消耗的算术位**恰好**等于「瓦片里最后一个 `1` 的位置」。
  > 早先这里写的是"任何残差非零的流全部失败（少读 2 位 ~ 少读 75002 位不等）"——
  > **那是判据自己的 bug**：`required` 在 `me-cli` 里就地扫位时把顺序写反了，
  > 取到的是「末字节最高位那侧的 `1`」，于是每条流被整齐地误报成"少读 1~6 位"。
  > 判据现在只有一份实现（Rust 侧 `me_decode::video::entropy::required_consumed_bits`，
  > 本目录的 `required_bits.py` 是它的独立 Python 复算），改正后一条都不差。
- `samples/` 里留着这批流（每个 < 10 KB），省得每次重跑 Chrome。常驻回归测试是
  `me-cli` 的 `bundled_probe_streams_pass_exit_symbol_exactly`（不需要 Chrome）。

`AV1_TRACE_FH` / `AV1_TRACE_PART` / `AV1_TRACE_MODE` / `AV1_TRACE_TXB` / `AV1_TRACE_SYM`
五个环境变量能把解码过程逐层打印出来，用法见
`~/.workbuddy/skills/lerxu-av1-softdecode/SKILL.md` 的 §4c。
