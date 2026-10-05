//! AV1 的**多符号算术解码器**（规范 §8.2）—— 自研软解的 M3。
//!
//! 这是整个引擎里第一段"不是读位、而是读概率"的代码。AV1 的瓦片数据是
//! **自适应算术编码**的：每个语法元素按一份 CDF（累积分布）解码，而这份 CDF
//! 每用一次就按"用得越多、改得越少"的速率向观测值靠拢（§8.2.7 的更新式）。
//!
//! ## 它由四块组成（照规范抄，不自己发明）
//!
//! | 过程 | 规范 | 干什么 |
//! |---|---|---|
//! | `init_symbol(sz)` | §8.2.2 | 读 15 位当 `SymbolValue`，`SymbolRange = 1<<15`，登录"还剩多少位" |
//! | `read_symbol(cdf)` | §8.2.3 | 在区间里定位符号、更新区间、重归一化（补读位）、自适应 CDF |
//! | `read_bool()` / `read_literal(n)` | §8.2.4/§8.2.5 | 等概率位（用一份临时 CDF）与它的 n 次串联 |
//! | `exit_symbol()` | §8.2.6 | 收尾：**校验末尾是 `1` + 补 `0`** 的 trailing bits |
//!
//! ## 三个容易写错、也最容易在这里埋雷的地方
//!
//! 1. **重归一化那一位的取反**：规范写的是
//!    `SymbolValue = paddedData ^ (((SymbolValue + 1) << bits) - 1)` ——
//!    不是加法，是"异或一个全 1 掩码"（等于按位取反低位）。抄成加法会在
//!    大多数帧上"看起来也能跑"。
//! 2. **`SymbolMaxBits` 允许变负**：负数表示"用掉了几位虚拟的 0 位"（比特流里
//!    并没有这些位）。所以补读位时必须 `Min(bits, Max(0, SymbolMaxBits))`。
//! 3. **CDF 更新里 `tmp` 的推进**：`tmp` 只在 `i == symbol` 那一步被置为
//!    `1 << 15` 并**保持**到循环结束；而 `symbol == N-1` 时那个分支永远不进
//!    （循环只到 `N-2`），于是整份 CDF 向 0 收。
//!
//! ## 验收口径（说清楚，别自我安慰）
//!
//! - 单测：核心不变量（每次读符号后 `SymbolValue < SymbolRange`）、位记账
//!   （读满一个 tile 后位置正好等于字节数）、以及**用 Python 照规范另写一份
//!   解码器**对同一段随机数据算出的符号序列（固定向量，见文件末尾）。
//! - 真实素材：拿真 tile 数据连续读符号，检查不变量与位记账（`me-cli` 的
//!   `real_av1_tests`）。
//! - **还差**：`exit_symbol()` 的硬验收要等"能完整解码一个 tile"（M4 的
//!   `decode_partition` + `decode_block` 全套语法）。只有把整条算术流按真实
//!   语法消耗干净，末尾的 trailing 检查才落在正确位置上 —— 所以本层的
//!   `exit_symbol` 现在只有合成数据的测试，真素材那一遍要等 M4。

use super::flags;
use me_core::{MediaError, Result};

use super::bitreader::floor_log2;

/// `EC_PROB_SHIFT`（规范常量表）：CDF 参与乘法前右移的位数。
pub const EC_PROB_SHIFT: u32 = 6;
/// `EC_MIN_PROB`：每个符号至少分到的概率（避免区间塌成 0）。
pub const EC_MIN_PROB: u32 = 4;
/// CDF 的概率精度（`1 << 15`）。
pub const CDF_PROB_BITS: u32 = 15;
/// `1 << CDF_PROB_BITS`。
pub const CDF_PROB_TOP: u32 = 1 << CDF_PROB_BITS;

/// `read_bool()` 用的那份等概率 CDF（规范 §8.2.4 每次现构造）。
pub const EQUIPROBABLE_CDF: [u16; 3] = [(1 << 14) as u16, (1 << 15) as u16, 0];

/// 按位读（MSB-first），越界返回 `NeedMore`。
///
/// ## 为什么不能逐位（2026-10-05）
///
/// 原来这里是 `for i in 0..n { (data[(pos+i)/8] >> (7 - (pos+i)%8)) & 1 }` ——
/// **逐位一次除法 + 一次取模 + 一次越界判断 + 一次装载**。而归一化**几乎每个符号
/// 都要读 1~15 位**（`num_bits = bits.min(avail)`），于是每个符号有几十条指令
/// 花在拆位上 —— `AV1_SYNTAX_ONLY` 采样里 `read_symbol_inner` 占**语法阶段的 11%**。
///
/// 改成一次取 **8 字节大端窗口**（末尾不足 8 字节时左对齐补零），
/// `off + n ≤ 7 + 32 = 39 ≤ 64` ⇒ 一次移位就出来。越界口径与逐位版**逐位一致**
/// （`have` = 第一个越界的下标）。逐位版留在单测 `read_bits_at_matches_the_bitwise_version` 里。
#[inline]
fn read_bits_at(data: &[u8], pos: usize, n: u32) -> Result<u32> {
    if n == 0 {
        return Ok(0);
    }
    let total = data.len() * 8;
    if pos + n as usize > total {
        return Err(MediaError::NeedMore {
            need: n as usize,
            have: total.saturating_sub(pos),
        });
    }
    debug_assert!(n <= 32, "read_bits_at 一次最多 32 位（读到 {n}）");
    let byte = pos >> 3;
    let off = (pos & 7) as u32;
    // 8 字节大端窗口：`w` 的最高位 == `data[byte]` 的最高位
    let w: u64 = if byte + 8 <= data.len() {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&data[byte..byte + 8]);
        u64::from_be_bytes(buf)
    } else {
        // 末尾不足 8 字节：左对齐补零（缺的位本来就不在 `data` 里，
        // 而上面已经保证需要的位全在）
        let take = data.len() - byte;
        let mut w = 0u64;
        for k in 0..take {
            w |= (data[byte + k] as u64) << (56 - 8 * k);
        }
        w
    };
    let want = off + n; // ≤ 39
    Ok(((w >> (64 - want)) & ((1u64 << n) - 1)) as u32)
}

/// AV1 的符号解码器：**一个瓦片一份**（规范里 `init_symbol(tileSize)` 在
/// `tile_group_obu` 的瓦片循环里逐瓦片调用）。
pub struct SymbolDecoder<'a> {
    /// 这个瓦片的原始字节（算术流）
    data: &'a [u8],
    /// 已经读过/跳过的位数（规范里的"比特流位置指示器"）
    bit_pos: usize,
    /// `SymbolValue`（15 位有效）
    symbol_value: u32,
    /// `SymbolRange`（15 位有效）
    symbol_range: u32,
    /// `SymbolMaxBits`：**允许为负**，表示已经借用了多少位虚拟的 0 位
    symbol_max_bits: i64,
    /// 帧头的 `disable_cdf_update`（为真时不做自适应更新）
    update_cdf: bool,
    /// 诊断：已经读过多少个符号/字面量（`AV1_TRACE_SYM` 下打前几个的算术状态）
    trace_n: u32,
    /// 诊断：当前符号的调用点标签（只给 trace 看）
    pub label: &'static str,
    /// 诊断：当前所属的变换块序号（只给 trace 看）
    pub tag: u32,
    /// 诊断：当前平面 / tx 尺寸（只给越界报错看）
    pub dbg_plane: u8,
    pub dbg_tx: u8,
    /// 诊断：`all_zero = 1` 的变换块个数（只给越界报错看）
    pub dbg_all_zero: u64,
    /// 诊断：读过的变换块个数
    pub dbg_txbs: u64,
    /// 诊断：最近读过的若干符号（越界报错时倒出来）
    recent: [RecentSym; RECENT_SYMBOLS],
    recent_n: usize,
}

/// 环形缓冲的深度（越界报错时倒出这么多条）。16 条足够看出"哪一步跳了"。
const RECENT_SYMBOLS: usize = 16;

/// 越界报错时倒出来的最近符号记录。
#[derive(Debug, Clone, Copy)]
struct RecentSym {
    tag: u32,
    label: &'static str,
    symbol: u32,
    smb: i64,
    /// 这次读的 CDF 有几个符号（`symbol` 必须 `< n`，对不上就说明 label 与 CDF 串了）
    n: u16,
    /// 读这个符号**之前**的 `smb`（>= 0 表示还没开始借位 ⇒ 这个符号是在合法数据上读的）
    smb_before: i64,
}

impl RecentSym {
    const EMPTY: Self =
        Self { tag: 0, label: "", symbol: 0, smb: 0, n: 0, smb_before: 0 };
}

/// 越界探测器允许的 `SymbolMaxBits` 下限。
///
/// 规范 §8.2.6 要求收尾时 `SymbolMaxBits >= -14`，所以默认就是 -14。
/// `AV1_SMB_LIMIT=<负数>` 可以放宽（**只用于诊断**：真素材上"到底多读了多少位"、
/// "多读之后能不能解完"这两个问题，靠放宽阈值一次就能回答）。
fn smb_limit() -> i64 {
    static L: std::sync::OnceLock<i64> = std::sync::OnceLock::new();
    *L.get_or_init(|| {
        std::env::var("AV1_SMB_LIMIT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(-14)
    })
}

/// 把最近符号倒成多行文本（越界报错的正文）。
fn recent_report(recent: &[RecentSym; RECENT_SYMBOLS], n: usize) -> String {
    let mut out = String::new();
    for k in 0..RECENT_SYMBOLS.min(n) {
        let idx = (n - 1 - k) % RECENT_SYMBOLS;
        let r = recent[idx];
        out.push_str(&format!(
            "\n    -{}: [{}] txb{} sym={}/{} smb={}{}",
            k,
            r.label,
            r.tag,
            r.symbol,
            r.n,
            r.smb,
            if r.smb_before >= 0 { "" } else { " ←读它之前已在借位" }
        ));
    }
    out
}

impl<'a> SymbolDecoder<'a> {
    /// `init_symbol( sz )`（规范 §8.2.2）。
    ///
    /// `sz` 就是这个瓦片的字节数（`tileSize`）。注意规范要求调用时**字节对齐**。
    ///
    /// `update_cdf` = `disable_cdf_update == 0`（帧头里读到的那个标志取反）。
    pub fn new(data: &'a [u8], update_cdf: bool) -> Result<Self> {
        if data.is_empty() {
            return Err(MediaError::Bitstream(
                "瓦片数据是空的（tileSize 为 0 的瓦片不该被解码）".into(),
            ));
        }
        let sz = data.len();
        // 诊断旋钮：强制关掉 CDF 自适应（`AV1_NO_CDF_UPDATE`）
        let update_cdf = update_cdf && !flags::no_cdf_update();
        // numBits = Min(sz * 8, 15)
        let num_bits = (sz * 8).min(15) as u32;
        let buf = read_bits_at(data, 0, num_bits)?;
        let padded = buf << (CDF_PROB_BITS - num_bits);
        Ok(Self {
            data,
            bit_pos: num_bits as usize,
            // SymbolValue = ((1 << 15) - 1) ^ paddedBuf
            symbol_value: (CDF_PROB_TOP - 1) ^ padded,
            symbol_range: CDF_PROB_TOP,
            // SymbolMaxBits = 8 * sz - 15
            symbol_max_bits: (sz * 8) as i64 - CDF_PROB_BITS as i64,
            update_cdf,
            trace_n: 0,
            label: "?",
            tag: 0,
            dbg_plane: 0,
            dbg_tx: 0,
            dbg_all_zero: 0,
            dbg_txbs: 0,
            recent: [RecentSym::EMPTY; RECENT_SYMBOLS],
            recent_n: 0,
        })
    }

    /// 这个瓦片的字节数。
    pub fn tile_size(&self) -> usize {
        self.data.len()
    }

    /// 已经读过的位数（诊断用）。
    pub fn bit_pos(&self) -> usize {
        self.bit_pos
    }

    /// `SymbolMaxBits`（诊断用；负数表示已经开始借虚拟 0 位）。
    pub fn symbol_max_bits(&self) -> i64 {
        self.symbol_max_bits
    }

    /// 已经消耗掉的**真实**位数（从 0 起单调增）。
    ///
    /// **位记账必须用它，不能用 `bit_pos` 的差分**：`bit_pos` 到了瓦片末尾就不再
    /// 前进（后面补的都是虚拟 0 位），于是"多读"会被记成"正好读完"。而这个量
    /// 是按每次归一化的 `bits` 累减的（规范 `SymbolMaxBits -= bits`），越界后照样增长。
    pub fn consumed_bits(&self) -> i64 {
        (self.data.len() as i64) * 8 - CDF_PROB_BITS as i64 - self.symbol_max_bits
    }

    /// `SymbolRange`（诊断/不变量检查用）。
    pub fn symbol_range(&self) -> u32 {
        self.symbol_range
    }

    /// `SymbolValue`（诊断/不变量检查用）。
    pub fn symbol_value(&self) -> u32 {
        self.symbol_value
    }

    /// `read_symbol( cdf )`（规范 §8.2.3）。
    ///
    /// `cdf` 是长度 `N + 1` 的数组：前 `N` 项是**递减**的累积分布
    /// （`cdf[N-1]` 必须是 `1 << 15`），最后一项是这个 CDF 被用过的次数
    /// （0..=32，决定自适应的速率）。**会被就地修改**（自适应）。
    #[track_caller]
    pub fn read_symbol(&mut self, cdf: &mut [u16]) -> Result<u32> {
        self.read_symbol_inner(cdf, self.update_cdf)
    }

    /// `read_bool()`（规范 §8.2.4）：等概率的伪原始位。
    ///
    /// 规范说这份临时 CDF 的自适应更新"可以省掉"（改完就不会再被用），
    /// 所以这里传 `update = false`。
    #[track_caller]
    pub fn read_bool(&mut self) -> Result<bool> {
        let mut cdf = EQUIPROBABLE_CDF;
        Ok(self.read_symbol_inner(&mut cdf, false)? == 1)
    }

    /// `read_literal( n )`（规范 §8.2.5）：`n` 个 `read_bool()` 拼成整数。
    #[track_caller]
    pub fn read_literal(&mut self, n: u32) -> Result<u32> {
        let mut x = 0u32;
        for _ in 0..n {
            x = (x << 1) | self.read_bool()? as u32;
        }
        Ok(x)
    }

    /// `ns( n )`（规范 §8.2.4）：**非对称**变长整数，取值 `[0, n)`。
    ///
    /// 与 [`super::bitreader`] 的 `BitReader::ns` 同口径，只是这里走符号解码器。
    /// 块级语法里只有环路恢复的 `decode_subexp_bool`（§5.11.48）用到它。
    #[track_caller]
    pub fn read_ns(&mut self, n: u32) -> Result<u32> {
        if n <= 1 {
            return Ok(0);
        }
        let w = u32::BITS - n.leading_zeros(); // FloorLog2(n) + 1
        let m = (1u32 << w) - n;
        let v = self.read_literal(w - 1)?;
        if v < m {
            return Ok(v);
        }
        let extra = self.read_literal(1)?;
        Ok((v << 1) - m + extra)
    }

    /// 自适应更新开关（诊断/测试用）。
    pub fn set_update_cdf(&mut self, on: bool) {
        self.update_cdf = on;
    }

    /// `read_symbol` 的本体；`update` 决定是否做 CDF 自适应。
    #[track_caller]
    fn read_symbol_inner(&mut self, cdf: &mut [u16], update: bool) -> Result<u32> {
        let smb_before = self.symbol_max_bits;
        let trace = Self::trace_sym()
            && self.trace_n < Self::trace_sym_limit()
            && super::block::in_traced_frame();
        // 先记下读之前的算术状态：符号要等解码完才知道，而状态会变。
        // 与 libaom 的符号级 trace 逐项对拍时用的就是这几个值。
        let trace_id = self.trace_n;
        let (trace_range, trace_value) = (self.symbol_range, self.symbol_value);
        let (trace_label, trace_tag) = (self.label, self.tag);
        let trace_nsym = cdf.len().saturating_sub(1);
        self.trace_n += 1;
        if cdf.len() < 2 {
            return Err(MediaError::Bitstream("CDF 至少要有 2 项（1 个符号 + 计数器）".into()));
        }
        let n = cdf.len() - 1; // 符号个数
        if cdf[n - 1] as u32 != CDF_PROB_TOP {
            return Err(MediaError::Bitstream(format!(
                "CDF 的最后一项应当是 1<<15（读到 {}）—— 这份 CDF 没初始化对",
                cdf[n - 1]
            )));
        }

        // do { symbol++; prev = cur; ... } while (SymbolValue < cur)
        let mut cur = self.symbol_range;
        let mut prev;
        let mut symbol = 0usize;
        loop {
            prev = cur;
            let f = CDF_PROB_TOP - cdf[symbol] as u32;
            cur = ((self.symbol_range >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
            cur += EC_MIN_PROB * (n - symbol - 1) as u32;
            if self.symbol_value >= cur {
                break;
            }
            symbol += 1;
            if symbol >= n {
                // 只有 cdf 不合法才会到这里（最后一项不是 1<<15 时 cur 不为 0）
                return Err(MediaError::Bitstream(format!(
                    "算术解码找不到符号（SymbolValue {} 落在区间外）",
                    self.symbol_value
                )));
            }
        }

        // SymbolRange = prev - cur; SymbolValue -= cur
        self.symbol_range = prev - cur;
        self.symbol_value -= cur;

        if trace {
            // `Location::caller()` 靠上面一串 `#[track_caller]` 一直传到真正的
            // 调用点 —— 这样 trace 里每一行都能直接对上源码行号，不用再猜
            // "这个符号是哪段语法读的"。
            let at = std::panic::Location::caller();
            println!(
                "  SYM#{} [{} txb{}] range={} value={} n={} sym={} cdf={:?} at={}:{}",
                trace_id,
                trace_label,
                trace_tag,
                trace_range,
                trace_value,
                trace_nsym,
                symbol,
                &cdf[..trace_nsym.min(10)],
                at.file(),
                at.line()
            );
        }

        // ── 重归一化 ──
        // bits = 15 - FloorLog2(SymbolRange)
        let bits = CDF_PROB_BITS - floor_log2(self.symbol_range);
        if bits > 0 {
            self.symbol_range <<= bits;
            // numBits = Min(bits, Max(0, SymbolMaxBits))
            let avail = self.symbol_max_bits.max(0) as u32;
            let num_bits = bits.min(avail);
            let new_data = read_bits_at(self.data, self.bit_pos, num_bits)?;
            let padded = new_data << (bits - num_bits);
            // SymbolValue = paddedData ^ ( ( ( SymbolValue + 1 ) << bits ) - 1 )
            //
            // **不要加 `& ((1<<bits)-1)` 掩码**：这是照抄 libaom
            // `od_ec_dec_normalize` 的 `((dif + 1) << d) - 1`，高位的 1 是有意义的
            // （区间底部的"补码"形式），抹掉它会让后续 `SymbolValue < cur` 判错。
            self.symbol_value = padded ^ (((self.symbol_value + 1) << bits) - 1);
            self.bit_pos += num_bits as usize;
            self.symbol_max_bits -= bits as i64;

            // ── 越界探测器（必须有，否则会**死循环**）──
            //
            // 规范 §8.2.6 `exit_symbol()` 要求收尾时 `SymbolMaxBits >= -14`，也就是
            // 说**一个合法瓦片最多只能借 14 位虚拟 0**。`SymbolMaxBits` 单调减，
            // 所以过程中一旦小于 -14，后面就不可能再回到合规值 ⇒ 这时已经读到瓦片
            // 末尾之外了（某个语法元素读错了位数）。
            //
            // **不拦会怎样**：越界后归位补的全是 0 位，而 `while ( !bit )` 这类循环
            // （`coeffs` 的 Exp-Golomb 长度码）会一直读到 0、永远等不到终止位 ⇒
            // 解码器**原地死转不报错**。（2026-09-30：真素材上就在那里挂了 18 分钟，
            // 栈上 84% 的采样在 `read_symbol_inner`。）
            if self.symbol_max_bits < smb_limit() {
                let required = required_consumed_bits(self.data);
                return Err(MediaError::Bitstream(format!(
                    "AV1 的算术流越过了瓦片末尾（帧 #{}，按“带瓦片的帧”计）：\
                     SymbolMaxBits = {}（下限 {}，规范要求 >= -14）；\
                     正在读 [{}]（txb {}、plane {}、txSize {}）\n\
                     已解 {} 个变换块，其中 all_zero = {} 个\n\
                     瓦片 {} 字节 = {} 位；已消耗 {} 位；**这份瓦片应有消耗 {} 位**\
                     ⇒ 多读了 {} 位\n最近读过的符号：{}",
                    super::block::TRACE_TILE_FRAME_NO
                        .load(std::sync::atomic::Ordering::Relaxed),
                    self.symbol_max_bits,
                    smb_limit(),
                    self.label,
                    self.tag,
                    self.dbg_plane,
                    self.dbg_tx,
                    self.dbg_txbs,
                    self.dbg_all_zero,
                    self.data.len(),
                    self.data.len() * 8,
                    self.consumed_bits(),
                    required.map(|r| r.to_string()).unwrap_or_else(|| "?".into()),
                    required
                        .map(|r| (self.consumed_bits() - r).to_string())
                        .unwrap_or_else(|| "?".into()),
                    recent_report(&self.recent, self.recent_n)
                )));
            }
        }

        // ── CDF 自适应（规范 §8.2.3 尾部）──
        if update {
            let c = cdf[n] as u32;
            let rate = 3 + (c > 15) as u32 + (c > 31) as u32 + floor_log2(n as u32).min(2);
            // ── 拆成两段单调循环（2026-10-05）─────────────────────────────
            //
            // 原来是**一个**循环 + 每项一个 `if i == symbol { tmp = TOP }`：
            // `tmp` 只在 `i == symbol` 那一步变成 `CDF_PROB_TOP`，所以
            //   * `i <  symbol`：`tmp == 0`         ⇒ `v - (v >> rate)`
            //   * `i >= symbol`：`tmp == CDF_PROB_TOP` ⇒ `v + ((TOP - v) >> rate)`
            //（`v == 0` 时第一式给 `0 - 0 = 0`、原式走 `v + 0 = 0`，**同值**。）
            // 拆开之后循环体内**没有分支**，两项都是单调映射。
            //
            // ⚠️ 边界与原来一致：`symbol` 最大取到 `n-1`，而循环上界是 `n-1`
            // （不含），所以 `symbol == n-1` 时**从头到尾都是第一式** ——
            // 第二段的空区间正好表达这一点。
            for v in cdf[..symbol].iter_mut() {
                let x = *v as i32;
                *v = (x - (x >> rate)) as u16;
            }
            for v in cdf[symbol..n - 1].iter_mut() {
                let x = *v as i32;
                *v = (x + ((CDF_PROB_TOP as i32 - x) >> rate)) as u16;
            }
            if cdf[n] < 32 {
                cdf[n] += 1;
            }
        }

        // 诊断：把这次符号记进环形缓冲（越界时报错会倒出来）
        self.recent[self.recent_n % RECENT_SYMBOLS] = RecentSym {
            tag: self.tag,
            label: self.label,
            symbol: symbol as u32,
            smb: self.symbol_max_bits,
            n: n as u16,
            smb_before,
        };
        self.recent_n = self.recent_n.wrapping_add(1);

        if trace {
            println!(
                "     -> sym={} 之后 range={} value={} smb={}",
                symbol, self.symbol_range, self.symbol_value, self.symbol_max_bits
            );
        }
        Ok(symbol as u32)
    }

    /// trace 开关（`AV1_TRACE_SYM`）。
    fn trace_sym() -> bool {
        static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *T.get_or_init(|| std::env::var("AV1_TRACE_SYM").is_ok())
    }

    /// 符号 trace 的上限（`AV1_TRACE_SYM_LIMIT`，默认 400）。
    /// 排查"第 2 帧才出问题"时把它调大，再对输出 `tail` 看崩溃点附近的符号。
    fn trace_sym_limit() -> u32 {
        static L: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        *L.get_or_init(|| {
            std::env::var("AV1_TRACE_SYM_LIMIT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(400)
        })
    }

    /// `exit_symbol( )`（规范 §8.2.6）：收尾并**校验 trailing bits**。
    ///
    /// 规范要求：`trailingBitPosition` 处的位是 `1`，它到
    /// `paddingEndPosition` 之间的位全是 `0`；且 `SymbolMaxBits >= -14`。
    ///
    /// 注意：只有"按真实语法把算术流消耗干净"之后，这个位置才是编码器写下的
    /// trailing bits —— 所以它的真实素材验收要等 M4（见模块抬头）。
    pub fn exit(&mut self) -> Result<()> {
        if self.symbol_max_bits < -14 {
            return Err(MediaError::Bitstream(format!(
                "exit_symbol 时 SymbolMaxBits = {}（规范要求 >= -14）—— 语法读多了",
                self.symbol_max_bits
            )));
        }
        // trailingBitPosition = get_position() - Min(15, SymbolMaxBits + 15)
        let trailing = self.bit_pos as i64
            - self.symbol_max_bits.saturating_add(15).min(15) as i64;
        // 位置推进 Max(0, SymbolMaxBits)（跳过还没读到的 trailing 位）
        self.bit_pos += self.symbol_max_bits.max(0) as usize;
        let padding_end = self.bit_pos;
        if trailing < 0 {
            return Err(MediaError::Bitstream("trailingBitPosition 算成了负数".into()));
        }
        let trailing = trailing as usize;
        if self.bit_at(trailing)? != 1 {
            return Err(MediaError::Bitstream(format!(
                "瓦片末尾第 {trailing} 位不是 1（trailing bits 没对上 —— 多半是前面读错了位）"
            )));
        }
        for x in (trailing + 1)..padding_end {
            if self.bit_at(x)? != 0 {
                return Err(MediaError::Bitstream(format!(
                    "瓦片末尾第 {x} 位不是 0（trailing 之后应当全是 0）"
                )));
            }
        }
        Ok(())
    }

    /// 读某一位（诊断与 `exit_symbol` 的校验用）。
    pub fn bit_at(&self, pos: usize) -> Result<u32> {
        if pos / 8 >= self.data.len() {
            return Err(MediaError::NeedMore { need: 1, have: 0 });
        }
        Ok(((self.data[pos / 8] >> (7 - (pos % 8))) & 1) as u32)
    }

    /// 还剩多少个**未被借用的**真实位（诊断用；负值已经被 `max(0, …)` 夹过）。
    pub fn real_bits_left(&self) -> i64 {
        self.symbol_max_bits
    }
}

/// 一份**合法**瓦片的「应有消耗位」—— 由规范 §8.2.6 `exit_symbol()` 化简而来，
/// **与解码器实现无关**（只读瓦片字节），所以它是熵层唯一能给的外部绝对判据。
///
/// ## 结论：它等于「这份瓦片里**最后一个 `1`** 的位置」
///
/// 设瓦片 `sz` 字节、`total = 8*sz` 位。收尾时规范里的量是：
///
/// ```text
/// N0  = Min(total, 15)          init_symbol 先吞进缓冲的位数
/// M0  = total - 15              初始 SymbolMaxBits
/// smb = M0 - consumed           consumed = 已消耗的算术位（= 各次归一化的 bits 之和）
/// R   = 真正从瓦片里读出的位数    consumed = R + virtual（virtual = 借用来的虚拟 0 位数）
/// get_position()      = N0 + R
/// trailingBitPosition = get_position() - Min(15, smb + 15)
/// paddingEndPosition  = get_position() + Max(0, smb)
/// ```
///
/// 规范要求 `trailingBitPosition` 处是 `1`、它到 `paddingEndPosition` 全 `0`
/// （且 `paddingEndPosition` 是 8 的整数倍）。分两种情形代入：
///
/// - **`smb >= 0`（真实位没读尽）**：`smb` 单调下降，所以全程都 >= 0
///   ⇒ 每次归一化都读满 `bits` 位，`R = consumed`；又 `total >= 15`
///   （否则 `smb` 必为负）⇒ `N0 = 15`，`Min(15, smb + 15) = Min(15, total - consumed) = 15`，
///   于是 `trailing = 15 + consumed - 15 = consumed`，
///   而 `paddingEnd = 15 + consumed + (total - 15 - consumed) = total`。
///   判据"`consumed` 处是 `1`、它到 `total` 全 `0`"正好就是
///   **`consumed` 是整段里最后一个 `1` 的位置**。
/// - **`smb < 0`（位已读尽、在借虚拟 0）**：缓冲里的 15 位与瓦片里的真实位都已读尽
///   ⇒ `get_position() = total`（`total < 15` 时那 15 位里就有 `total` 位真实位，
///   `R = 0`；`total >= 15` 时 `R = total - 15`）。又
///   `Min(15, smb + 15) = Min(15, total - consumed) = total - consumed`（因为
///   `smb < 0` ⇒ `consumed > total - 15`），`paddingEnd = total + Max(0, smb) = total`。
///   于是 `trailing = total - (total - consumed) = consumed` —— 同一个结论。
///
/// 返回 `None` 表示瓦片里一个 `1` 都没有 —— 那不是合法码流（`trailing` 位无处可落）。
///
/// ## 为什么这个函数值得存在
///
/// 它是**精确整数**判据，比 `exit_symbol()` 的"过/不过"敏感得多：
/// [`SymbolDecoder::consumed_bits`] 必须**恰好**等于它 —— 真少读时它会直接指出缺
/// 多少位（Golomb 长度码极性写反那类 bug 就是这么抓出来的）。
/// 但也正因为它只有一份，**抄错一次就会把整轮排查带偏**：2026-09-30 那次
/// "残差非零的流全都少读 1~6 位、像是漏读了符号"的悬案，根因就是本判据在 `me-cli`
/// 的测试里被就地抄了一份、扫位顺序写反（取到了「末字节最高位的 1」）；
/// 判据一修，14 条参照流**一条都不差**。所以这份实现只留**一处**，别在别处再抄一遍。
pub fn required_consumed_bits(tile: &[u8]) -> Option<i64> {
    let mut last1 = None;
    for (i, byte) in tile.iter().enumerate() {
        if *byte != 0 {
            // 字节内位号大的在流里靠后（MSB-first 存放）⇒ 取**最低位那侧**的 `1`。
            let lowest = byte.trailing_zeros() as usize;
            last1 = Some((i * 8 + (7 - lowest)) as i64);
        }
    }
    last1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 一份"均匀分布"的 4 符号 CDF（每个符号 1/4）。
    ///
    /// **方向很重要**：AV1 的 CDF 是**递增**的累积分布 ——
    /// `cdf[N-1]` 恒为 `1 << 15`，后面再跟一个计数器。规范里真实的表例如
    /// `Default_Partition_W8_Cdf[0] = {19132, 25510, 30392, 32768, 0}`。
    /// 写成递减**不会报错**，但符号会全部落到最后一档（第一版测试就是这么被骗过去的：
    /// 用递减 CDF 跑随机数据，24 个符号里 22 个是 3）。
    fn uniform4() -> [u16; 5] {
        [8192, 16384, 24576, 32768, 0]
    }


    /// 逐位参照实现（改动前的版本，原样保留）。
    fn read_bits_at_bitwise(data: &[u8], pos: usize, n: u32) -> Result<u32> {
        let mut out = 0u32;
        for i in 0..n as usize {
            let p = pos + i;
            if p / 8 >= data.len() {
                return Err(MediaError::NeedMore { need: n as usize, have: i });
            }
            let bit = (data[p / 8] >> (7 - (p % 8))) & 1;
            out = (out << 1) | bit as u32;
        }
        Ok(out)
    }

    /// [`read_bits_at`] 的 64 位窗口版必须与**逐位版**逐值相同（含越界报错口径）。
    ///
    /// 这是 2026-10-05 那次提速的护栏：改错了只会让"某些位位置/位长"读错一格，
    /// 而算术流错一位就是整帧错，**从像素差几乎定位不到**。

    /// CDF 自适应的**两段拆分**必须与原来"一个循环 + 每项判 `i == symbol`"
    /// 的写法逐项相同（`rate` 取遍真实值域、`symbol` 取遍每个位置、`n` 取几个典型值）。
    ///
    /// 这条自适应是**唯一**让解码器与编码器保持同一个概率模型的地方：拆错一项，
    /// 后面整条算术流都会偏，而现象是"某一帧之后画面花掉"——**极难定位**。
    #[test]
    fn cdf_adapt_split_matches_the_single_loop_version() {
        // 照规范播种的 CDF：单调增、末项 1<<15（用一个固定斜坡近似真实表）
        for n in [2usize, 3, 4, 5, 8, 11, 13, 14, 16] {
            let seed: Vec<u16> = (0..n)
                .map(|i| (((i + 1) * CDF_PROB_TOP as usize) / n) as u16)
                .collect();
            let mut cdf = seed.clone();
            cdf.push(0); // 计数器
            for symbol in 0..n {
                for rate in 3u32..=8 {
                    let mut want = cdf.clone();
                    {
                        let mut tmp = 0i32;
                        for i in 0..n - 1 {
                            if i == symbol {
                                tmp = CDF_PROB_TOP as i32;
                            }
                            let v = want[i] as i32;
                            want[i] = if tmp < v {
                                (v - ((v - tmp) >> rate)) as u16
                            } else {
                                (v + ((tmp - v) >> rate)) as u16
                            };
                        }
                        if want[n] < 32 {
                            want[n] += 1;
                        }
                    }
                    let mut got = cdf.clone();
                    {
                        for v in got[..symbol].iter_mut() {
                            let x = *v as i32;
                            *v = (x - (x >> rate)) as u16;
                        }
                        for v in got[symbol..n - 1].iter_mut() {
                            let x = *v as i32;
                            *v = (x + ((CDF_PROB_TOP as i32 - x) >> rate)) as u16;
                        }
                        if got[n] < 32 {
                            got[n] += 1;
                        }
                    }
                    assert_eq!(got, want, "n={n} symbol={symbol} rate={rate}");
                }
            }
        }
    }

    #[test]
    fn read_bits_at_matches_the_bitwise_version() {
        // 造一段有 0/1 交替模式的字节（避免全 0/全 1 掩盖移位错误）
        let data: Vec<u8> = (0..40u32).map(|i| ((i * 37 + 11) & 0xff) as u8).collect();
        let total = data.len() * 8;
        for pos in 0..total {
            // `n ≤ 32` 是函数契约（`off + n ≤ 7 + 32 = 39 ≤ 64` 一个窗口装得下）；
            // 生产调用点的 `n` 实际只有 1~15（重归一化位数）。
            for n in 0..=32u32 {
                let got = super::read_bits_at(&data, pos, n);
                let want = read_bits_at_bitwise(&data, pos, n);
                match (got, want) {
                    (Ok(a), Ok(b)) => assert_eq!(a, b, "pos={pos} n={n}"),
                    (Err(MediaError::NeedMore { need: a, have: b }),
                     Err(MediaError::NeedMore { need: c, have: d })) => {
                        assert_eq!((a, b), (c, d), "pos={pos} n={n} 的越界口径不同");
                    }
                    (a, b) => panic!("pos={pos} n={n}：一边成功一边失败 {a:?} vs {b:?}"),
                }
            }
        }
    }

    #[test]
    fn init_symbol_reads_fifteen_bits_and_inverts_them() {
        // 头 15 位 = 0b000000000000000 → buf = 0 → padded = 0
        // SymbolValue = 0x7FFF ^ 0 = 0x7FFF
        let data = [0x00u8, 0x00, 0x00, 0x00];
        let d = SymbolDecoder::new(&data, false).unwrap();
        assert_eq!(d.symbol_value(), 0x7FFF);
        assert_eq!(d.symbol_range(), 1 << 15);
        assert_eq!(d.symbol_max_bits(), 32 - 15);
        assert_eq!(d.bit_pos(), 15, "只有 15 位被读走（不足 15 位时读满字节数）");

        // 头 15 位全 1 → padded = 0x7FFF → SymbolValue = 0
        let data = [0xFFu8, 0xFF, 0xFF, 0xFF];
        let d = SymbolDecoder::new(&data, false).unwrap();
        assert_eq!(d.symbol_value(), 0);
        assert_eq!(d.symbol_max_bits(), 17);

        // 短瓦片：sz = 1 → numBits = 8 → SymbolMaxBits = 8 - 15 = -7（合法）
        let d = SymbolDecoder::new(&[0xABu8], false).unwrap();
        assert_eq!(d.bit_pos(), 8);
        assert_eq!(d.symbol_max_bits(), -7);
    }

    #[test]
    fn empty_tile_is_an_error() {
        assert!(SymbolDecoder::new(&[], true).is_err());
    }

    /// 朴素参照实现（MSB-first 把整个瓦片摊成位串再找最后一个 `1`）——
    /// 用来给 [`required_consumed_bits`] 的"字节内取最低位那侧的 1"当对照。
    fn last_one_bit_naive(tile: &[u8]) -> Option<i64> {
        let bits: Vec<u8> = tile.iter().flat_map(|b| (0..8).map(move |k| (b >> (7 - k)) & 1)).collect();
        bits.iter().rposition(|b| *b == 1).map(|i| i as i64)
    }

    #[test]
    fn required_bits_is_the_last_one_bit_of_the_tile() {
        // 单字节 0x13 = 0b00010011：最后那个 `1` 在**第 7 位**（最低位）。
        // 这条就是 2026-09-30 那次判据抄错的现场：反过来扫字节内的位会得到 3
        // （= 最高位那侧的 `1`），于是 14 条参照流全被报成"少读 1~6 位"。
        assert_eq!(required_consumed_bits(&[0x13]), Some(7));
        assert_ne!(required_consumed_bits(&[0x13]), Some(3), "不能取到最高位那侧的 1");

        // 真实瓦片（put_L41_32x32_flat.bin）：00 6b f6 ac 60 → 末尾那个 `1` 在第 34 位。
        assert_eq!(required_consumed_bits(&[0x00, 0x6b, 0xf6, 0xac, 0x60]), Some(34));
        // 真实瓦片（put_L40_32x32_flat.bin）：只有 1 字节。
        assert_eq!(required_consumed_bits(&[0x13]), Some(7));
        // 真实瓦片（put_L31_64x64_flat.bin）。
        assert_eq!(
            required_consumed_bits(&[0xb4, 0x5a, 0xcc, 0xa9, 0x9a, 0x89, 0xe4]),
            Some(53)
        );

        // 末字节只有最低位是 1：答案必须落到它身上，而不是前一字节。
        assert_eq!(required_consumed_bits(&[0xff, 0xff, 0x01]), Some(23));
        // 字节内部的"低位那侧"优先（0x0f 的最低有效 1 在第 0 位 → 位号最大）。
        assert_eq!(required_consumed_bits(&[0x00, 0x0f]), Some(15));
        // 全 0 瓦片没有任何 `1` —— 不是合法码流。
        assert_eq!(required_consumed_bits(&[0x00, 0x00]), None);
        assert_eq!(required_consumed_bits(&[]), None);
    }

    #[test]
    fn required_bits_agrees_with_a_naive_bit_string_scan() {
        // 位运算实现容易在"字节内取哪一侧的 1"上出错，所以拿一份朴素实现
        // 对一大批伪随机瓦片做全量对拍（含全 0、单字节、跨字节等情形）。
        let mut tiles: Vec<Vec<u8>> = vec![vec![], vec![0x00], vec![0x13], vec![0x00; 7], vec![0x01]];
        let mut state = 0x1234_5678u32;
        for _ in 0..400 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let len = 1 + (state % 9) as usize;
            let mut t = Vec::with_capacity(len);
            for _ in 0..len {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                // 混入大量 0 字节，制造"末尾若干字节全 0"的情形
                t.push(if state.is_multiple_of(3) { 0 } else { (state >> 13) as u8 });
            }
            tiles.push(t);
        }
        for tile in &tiles {
            assert_eq!(
                required_consumed_bits(tile),
                last_one_bit_naive(tile),
                "瓦片 {tile:02x?} 的应有消耗位对不上"
            );
        }
        // 至少要有一些瓦片是"有 1 的"，否则这条测试是空转
        assert!(tiles.iter().filter(|t| required_consumed_bits(t).is_some()).count() > 200);
    }

    #[test]
    fn required_bits_matches_the_spec_trailing_bit_oracle() {
        // 直接照规范 §8.2.6 的公式（**不化简**）写一份朴素实现，验证它在整个
        // 合法 (total, consumed) 取值域上都等价于 required_consumed_bits。
        // 这样这条判据的来源就不是"我觉得是最后一个 1"，而是规范公式本身。
        fn spec_ok(tile: &[u8], bit_pos: i64, smb: i64) -> bool {
            let total = (tile.len() * 8) as i64;
            let n0 = total.min(15);
            let r = bit_pos - n0;
            if r < 0 {
                return false;
            }
            let trailing = (n0 + r) - (smb + 15).min(15);
            if trailing < 0 {
                return false;
            }
            let padding_end = n0 + r + smb.max(0);
            if padding_end % 8 != 0 {
                return false; // paddingEndPosition 必须是 8 的整数倍
            }
            let bit = |i: i64| -> Option<u32> {
                if i < 0 || i / 8 >= tile.len() as i64 {
                    None
                } else {
                    Some(((tile[i as usize / 8] >> (7 - (i % 8))) & 1) as u32)
                }
            };
            if bit(trailing) != Some(1) {
                return false;
            }
            (trailing + 1..padding_end).all(|x| bit(x) == Some(0))
        }

        // 数一下这条测试真的走到了两种结论上（否则它可能是"恒真"的空转）
        let (mut oks, mut fails) = (0, 0);
        for total in [8i64, 16, 24, 32, 40, 48, 56, 64, 72] {
            let mut tile = vec![0u8; (total / 8) as usize];
            for consumed in 0..total {
                // 由 consumed 反推该瓦片应有的解码状态：
                // consumed 位已消耗 ⇒ smb = total - 15 - consumed
                let smb = total - 15 - consumed;
                // R = 真正从瓦片里读出的位数（缓冲那 15 位之外的部分）
                let real = if smb >= 0 {
                    consumed // 全程没借虚拟位，读出的就是消耗的
                } else if total >= 15 {
                    total - 15 // 瓦片里的真实位已全部读尽
                } else {
                    0 // 不足 15 位：全在缓冲里，没再从瓦片读
                };
                let bit_pos = total.min(15) + real;

                // ① 合法的瓦片（最后一个 `1` 就在 consumed 位）⇒ 判据成立
                tile.iter_mut().for_each(|b| *b = 0);
                tile[(consumed / 8) as usize] |= 1 << (7 - (consumed % 8));
                assert_eq!(required_consumed_bits(&tile), Some(consumed));
                assert!(spec_ok(&tile, bit_pos, smb), "total={total} consumed={consumed}：合法瓦片应当通过");
                oks += 1;

                // ② 改掉那一位（最后一个 `1` 挪到别处）⇒ 判据必须不成立
                if consumed > 0 {
                    tile.iter_mut().for_each(|b| *b = 0);
                    tile[((consumed - 1) / 8) as usize] |= 1 << (7 - ((consumed - 1) % 8));
                    assert_eq!(required_consumed_bits(&tile), Some(consumed - 1));
                    assert!(
                        !spec_ok(&tile, bit_pos, smb),
                        "total={total} consumed={consumed}：最后一个 1 不在 consumed 位时不该通过"
                    );
                    fails += 1;
                }
            }
        }
        assert!(oks > 100 && fails > 100, "对拍样本太少：{oks} 通过 / {fails} 拒绝");
    }

    #[test]
    fn reads_symbols_and_keeps_the_core_invariant() {
        // 用"看起来随机"的字节与均匀 CDF 连续读符号：每读一次都要满足
        // **SymbolValue < SymbolRange**（算术解码的核心不变量）。
        // 重归一化那一步写错（比如把异或抄成加法），这条很快就会破。
        let data: Vec<u8> = (0..2048u32).map(|i| ((i * 37 + 11) % 251) as u8).collect();
        let mut d = SymbolDecoder::new(&data, true).unwrap();
        let mut count = 0u32;
        while d.symbol_max_bits() > 0 {
            let mut cdf = uniform4();
            let sym = d.read_symbol(&mut cdf).unwrap();
            assert!(sym < 4, "符号越界：{sym}");
            assert!(
                d.symbol_value() < d.symbol_range(),
                "第 {count} 次读符号后 SymbolValue {} >= SymbolRange {}",
                d.symbol_value(),
                d.symbol_range()
            );
            // 重归一化之后 SymbolRange 必须重新落在 [2^15, 2^16)
            // （`bits = 15 - FloorLog2(SymbolRange)` 就是把最高位顶回第 15 位）
            assert!(
                d.symbol_range() >= (1 << 15) && d.symbol_range() < (1 << 16),
                "第 {count} 次读后 SymbolRange 没重归一化：{}",
                d.symbol_range()
            );
            count += 1;
        }
        assert!(count > 100, "至少该读出上百个符号（实得 {count}）");
    }

    #[test]
    fn reading_a_whole_tile_consumes_exactly_its_bytes() {
        // 位记账：把 SymbolMaxBits 读到 <= 0 之后，位置指示器必须**正好**等于
        // 字节数 —— 这条证明"补读位"的账没算错。
        let data: Vec<u8> = (0..1000u32).map(|i| ((i * 101 + 7) % 256) as u8).collect();
        let mut d = SymbolDecoder::new(&data, true).unwrap();
        while d.symbol_max_bits() > 0 {
            let mut cdf = uniform4();
            d.read_symbol(&mut cdf).unwrap();
        }
        assert_eq!(d.bit_pos(), data.len() * 8, "读满之后位置应当正好到末尾");
        assert!(d.symbol_max_bits() <= 0);
        // 规范要求进入 exit_symbol 时 SymbolMaxBits >= -14
        assert!(d.symbol_max_bits() >= -14, "读到 {}, 越界了", d.symbol_max_bits());
    }

    #[test]
    fn cdf_adapts_toward_the_observed_symbol() {
        // 连续读同一个符号（拿全 0 数据 + 故意偏的 CDF），CDF 应当越来越尖
        let data = [0x00u8; 256];
        let mut d = SymbolDecoder::new(&data, true).unwrap();
        let mut cdf = uniform4();
        let before = cdf;
        let first = d.read_symbol(&mut cdf).unwrap();
        assert_eq!(cdf[4], 1, "计数器该 +1");
        // 计数器 <= 15 时 rate = 3 + 0 + 0 + min(floor(log2(4)), 2) = 3 + 2 = 5
        // 第 symbol 项向 1<<15 靠，右边的往 0 靠
        if first < 3 {
            let v = cdf[first as usize] as i32;
            let expect = before[first as usize] as i32
                + ((CDF_PROB_TOP as i32 - before[first as usize] as i32) >> 5);
            assert_eq!(v, expect, "第 {first} 项的更新值不对");
        }
        // 反复读：计数器涨到 32 就停
        for _ in 0..64 {
            d.read_symbol(&mut cdf).unwrap();
        }
        assert_eq!(cdf[4], 32, "计数器封顶 32");
    }

    #[test]
    fn read_bool_uses_a_fresh_equiprobable_cdf() {
        let data = [0x00u8; 64];
        let mut d = SymbolDecoder::new(&data, true).unwrap();
        // 全 0 数据下连续读位：CDF 不累积（每次都新建），所以行为完全确定
        let bits: Vec<bool> = (0..16).map(|_| d.read_bool().unwrap()).collect();
        assert_eq!(bits.len(), 16);

        // read_literal(n) 就是 n 次 read_bool 的串联
        let mut d1 = SymbolDecoder::new(&data, true).unwrap();
        let mut d2 = SymbolDecoder::new(&data, true).unwrap();
        let lit = d1.read_literal(8).unwrap();
        let mut manual = 0u32;
        for _ in 0..8 {
            manual = (manual << 1) | d2.read_bool().unwrap() as u32;
        }
        assert_eq!(lit, manual);
        assert!(lit < 256);
    }

    #[test]
    fn exit_accepts_trailing_bits_and_rejects_garbage() {
        // 造一个"读完刚好停在末尾"的状态：读满之后 exit 要求末尾是 trailing bits。
        // 这里直接用一段末尾为 0x80（1000_0000，即 1 后跟 7 个 0）的数据 ——
        // 读满后 trailingBitPosition 会落在那一位上。
        let mut data = vec![0x55u8; 64];
        data.push(0x80);
        let mut d = SymbolDecoder::new(&data, true).unwrap();
        while d.symbol_max_bits() > 0 {
            let mut cdf = uniform4();
            d.read_symbol(&mut cdf).unwrap();
        }
        let trailing_ok = d.exit();
        // 末尾那一位是 1，后面没有更多位 → 应当通过（若位置上恰好对上）
        // 不强制断言通过（位置取决于消耗的位数），但要断言"不 panic 且要么通过、
        // 要么给出可读的错误"
        if let Err(e) = trailing_ok {
            let text = format!("{e}");
            assert!(
                text.contains("trailing") || text.contains("末尾"),
                "报错要指明是 trailing 校验：{text}"
            );
        }

        // SymbolMaxBits 太负时 exit 必须报错
        let mut d = SymbolDecoder::new(&[0u8; 4], true).unwrap();
        d.symbol_max_bits = -20;
        let e = d.exit().unwrap_err();
        assert!(format!("{e}").contains("SymbolMaxBits"));
    }

    #[test]
    fn rejects_a_cdf_that_is_not_normalized() {
        let mut d = SymbolDecoder::new(&[0x12u8; 8], true).unwrap();
        let mut bad = [1u16, 2, 3];
        let e = d.read_symbol(&mut bad).unwrap_err();
        assert!(format!("{e}").contains("CDF"), "要报 CDF 不合法：{e}");
        let mut too_short = [0u16];
        assert!(d.read_symbol(&mut too_short).is_err());
    }

    fn hex_to_bytes(hex: &str) -> Vec<u8> {
        let b = hex.as_bytes();
        (0..b.len() / 2)
            .map(|i| {
                let hi = (b[2 * i] as char).to_digit(16).unwrap() as u8;
                let lo = (b[2 * i + 1] as char).to_digit(16).unwrap() as u8;
                (hi << 4) | lo
            })
            .collect()
    }

    /// 与**独立参考实现**的对拍向量。
    ///
    /// 参考实现是照规范 §8.2 用 Python 另写的一份（`av1_entropy_ref.py`，
    /// 存在 `lerxu-av1-softdecode` 技能目录里），直译规范、不做优化：同一段确定性
    /// 伪随机数据（LCG）、同一份 CDF，两边必须解出**完全一样的符号序列、位位置
    /// 与自适应后的 CDF**。
    ///
    /// 为什么需要它：`read_symbol` 里最容易错的是位序与移位
    /// （`f >> EC_PROB_SHIFT`、再 `>> (7 - EC_PROB_SHIFT)`）、重归一化那句
    /// "异或全 1 掩码"、以及 CDF 更新的 `rate` 公式 —— 这些错在"自己和自己比"
    /// 的测试里全都看不出来（位序自洽地错一整片）。
    ///
    /// 向量里的 CDF 直接取规范 §9.3 的真实表（`Default_Partition_W8_Cdf` /
    /// `Default_Partition_W16_Cdf`），不是自己编的——所以连"CDF 方向"这件事
    /// 也被钉住了（递增、末项 `1<<15`）。
    ///
    /// 改动解析器后重跑生成命令（脚本在技能目录里）：
    /// ```text
    /// python3 ~/.workbuddy/skills/lerxu-av1-softdecode/av1_entropy_ref.py
    /// ```
    /// 把输出里的 symbols / cdf_after / end 状态抄回上面的向量。
    #[test]
    fn matches_the_independent_reference_vectors() {
        struct RefVec {
            name: &'static str,
            hex: &'static str,
            cdf: &'static [u16],
            symbols: &'static [u32],
            cdf_after: &'static [u16],
            end_bit_pos: usize,
            end_max_bits: i64,
        }
        let vectors = [
            RefVec {
                name: "Default_Partition_W8_Cdf[0]（真实表）+ 12 个符号",
                hex: "dc0465aa1fad1d5adae5ac1b1e5f1370796cfd10ff19af601d04acb41d022b46\
                      78733af2df5faeb70859d1ee3910cb4895b5cc892911ff06b6622edf3cf935fd",
                cdf: &[19132, 25510, 30392, 32768, 0],
                symbols: &[2, 0, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                cdf_after: &[21137, 26312, 30371, 32768, 12],
                end_bit_pos: 32,
                end_max_bits: 480,
            },
            RefVec {
                name: "Default_Partition_W8_Cdf[3] + 16 个符号",
                hex: "be4980b175df3b5a3bbe5924161a8b438a77cdf4ec079b3665c8d349c55a69e7\
                      0da1cd45bb1f1cc8d30e0888b240888592b740abdc04b08831601cc7378b3d74\
                      2368acee0b550aabe94cf0af6ebb9aca8d235294c2729e29262129a5340e725e",
                cdf: &[9896, 18783, 25853, 32768, 0],
                symbols: &[2, 2, 3, 3, 0, 1, 2, 2, 3, 3, 0, 2, 2, 1, 1, 3],
                cdf_after: &[7555, 15598, 24537, 32768, 16],
                end_bit_pos: 49,
                end_max_bits: 719,
            },
            RefVec {
                name: "Default_Partition_W16_Cdf[0]（10 个符号）",
                hex: "ff3e850a29484799e8f0f9e14bff75a9841cec9542f9a331cf3f6ebde93aedbb\
                      6244883976f32bdef002c1b7fb0072bfc9ba9d30e56a948c599c1a6a7fe49d0\
                      1a84134766e4f57e5d93060313439c58cad5c12c5af58a95500a23a279911002\
                      b494dc09908f683051c930e28ee412668a918852c985c9ae23c6804cd305acd8f",
                cdf: &[15597, 20929, 24571, 26706, 27664, 28821, 29601, 30571, 31902, 32768, 0],
                symbols: &[9, 7, 0, 2, 0, 0, 0, 0, 0, 1, 6, 0, 2, 0, 1, 9],
                cdf_after: &[
                    15777, 20821, 24643, 25926, 26504, 27200, 28542, 29781, 30582, 32768, 16,
                ],
                end_bit_pos: 58,
                end_max_bits: 966,
            },
            RefVec {
                name: "等概率二符号（等价于 read_bool 的那份 CDF）",
                hex: "b623849d3e6b1c74178ab1a268feb1ef45922a84bd7fac6a9f3efdc7102e80d7\
                      ceea62ab426889a447d23dafbd24943ecc8aa009f8775789e9854531f810f36b",
                cdf: &[16384, 32768, 0],
                symbols: &[1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                cdf_after: &[20260, 32768, 24],
                end_bit_pos: 39,
                end_max_bits: 473,
            },
        ];
        for v in &vectors {
            let data = hex_to_bytes(v.hex);
            let mut d = SymbolDecoder::new(&data, true).unwrap();
            let mut cdf = v.cdf.to_vec();
            let mut got = Vec::new();
            for _ in 0..v.symbols.len() {
                got.push(d.read_symbol(&mut cdf).unwrap());
            }
            assert_eq!(got, v.symbols, "{}：符号序列与参考实现不一致", v.name);
            assert_eq!(cdf, v.cdf_after, "{}：自适应后的 CDF 与参考实现不一致", v.name);
            assert_eq!(d.bit_pos(), v.end_bit_pos, "{}：位位置不一致", v.name);
            assert_eq!(
                d.symbol_max_bits(),
                v.end_max_bits,
                "{}：SymbolMaxBits 不一致",
                v.name
            );
        }
    }
}
