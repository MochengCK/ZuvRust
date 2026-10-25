//! 帧级后处理的**编排**（规范 §7.14–§7.17）：去块 → CDEF → 上采样 → 环路恢复。
//!
//! 这是 `decode_frame_wrapup`（§7.18）里"滤波那一段"的直译，顺序不能换：
//!
//! ```text
//! CurrFrame ──deblock──► CurrFrame ──CDEF──► CdefFrame
//!                                               │
//!               ┌───────────── 上采样（§7.16，超分）─────────────┐
//!               ▼                                                ▼
//!       UpscaledCurrFrame                              UpscaledCdefFrame
//!               └──────────► 环路恢复（§7.17）──────────► LrFrame
//! ```
//!
//! `LrFrame` 就是**对外输出**、也是**存进参考帧**的那份像素（§7.20 的
//! `FrameStore[i] = LrFrame`）—— 所以这条链上任何一步写错，都会污染后续帧。

use super::flags;
use me_core::Result;

use super::block::BlockGrid;
use super::frame_header::FrameHeader;
use super::intra::CurrFrame;
use super::partition::MiGrid;
use super::sequence_header::SequenceHeader;

/// `decode_frame_wrapup` 的滤波段：返回 `LrFrame`。
///
/// 顺序与条件照规范 §7.18（"如果 `loop_filter_level[0] != 0` 或 `[1] != 0`
/// 才跑去块"这种门也一并抄下来 —— 它决定"零强度时去块是不是被跳过"）。
pub fn decode_frame_wrapup(
    mut frame: CurrFrame,
    fh: &FrameHeader,
    seq: &SequenceHeader,
    grid: &BlockGrid,
    mi_grid: &MiGrid,
    lr: &super::lr::LrState,
) -> Result<CurrFrame> {
    // 诊断（`AV1_LR_DUMP=<目录>`）：逐阶段落盘（去块前/后），
    // 用来与 libaom 的同阶段比对 —— 差异"由哪一层引入"一次就能定位。
    let dump_dir = flags::lr_dump_dir();
    let dump_n = super::block::TRACE_FRAME_NO.load(std::sync::atomic::Ordering::Relaxed);
    if let Some(d) = &dump_dir {
        super::lr::lr_dump(&frame, d, "predb", dump_n);
    }
    let phase_t = flags::phase_timing();
    let t_lf = std::time::Instant::now();
    // §7.14 去块（原地修改 CurrFrame）
    let lf = &fh.params.loop_filter;
    if (lf.level[0] != 0 || lf.level[1] != 0) && !flags::no_deblock() {
        super::deblock::loop_filter(&mut frame, fh, seq, grid, mi_grid)?;
    }
    if let Some(d) = &dump_dir {
        super::lr::lr_dump(&frame, d, "deb", dump_n);
    }
    let lf_ms = t_lf.elapsed().as_secs_f64() * 1e3;
    let t_cdef = std::time::Instant::now();
    // §7.15 CDEF（不改 CurrFrame；产出 CdefFrame）
    let cdef = super::cdef::apply(&frame, fh, seq, grid)?;
    let cdef_ms = t_cdef.elapsed().as_secs_f64() * 1e3;
    // §7.16 上采样（超分）。**没开超分时它按规范就是恒等变换**
    // （`UpscaledWidth == FrameWidth`）—— 那就把两份帧**直接搬过去**，
    // 省掉两次整帧拷贝（1080p 一帧 3 个平面 ≈ 6 MB，两份就是 12 MB，
    // 在 `sample` 里曾经是 memmove 的最大来源）。
    let (up_curr, up_cdef) = if fh.use_superres {
        let up_curr = super::upscale::apply(&frame, fh, seq)?;
        // 没做 CDEF 时上采样 `CurrFrame`（内容与 `CdefFrame` 相同，只是要新缓冲）。
        let up_cdef = match &cdef {
            Some(c) => super::upscale::apply(c, fh, seq)?,
            None => super::upscale::apply(&frame, fh, seq)?,
        };
        (up_curr, Some(up_cdef))
    } else {
        (frame, cdef)
    };
    // §7.17 环路恢复（参数来自块级语法读出的 `LrType` / `LrWiener` / `LrSgrXqd`）。
    // 按值传进去：不用超分时这里就是**零拷贝移动**，`lr::apply` 内部
    // "不滤波"那一档直接把 `CdefFrame` 当输出（见那边的注释）。
    let t_lr = std::time::Instant::now();
    let out = super::lr::apply(up_curr, up_cdef, fh, seq, lr)?;
    if phase_t {
        eprintln!(
            "FILTER frame={} lf={:.1}ms cdef={:.1}ms lr={:.1}ms",
            dump_n,
            lf_ms,
            cdef_ms,
            t_lr.elapsed().as_secs_f64() * 1e3
        );
    }
    Ok(out)
}
