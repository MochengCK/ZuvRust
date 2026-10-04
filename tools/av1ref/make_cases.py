#!/usr/bin/env python3
"""造 AV1 受控参照流（内容已知、工具可逐项开关），并用 dav1d 出参考。

    python3 make_cases.py --out=/tmp/refs [--aomenc=/tmp/libaom-3.9.0/build_g/aomenc]

产出 `<out>/<名字>.mp4` 与 `<out>/<名字>_ref.npy`（dav1d 解出的 yuv420p，形状 (帧, h*3/2, w)）。

**内容要"自然"**：大范围平滑渐变 + 多尺度柔和块噪声。硬边条纹/小块棋盘会被 libaom
判定为屏幕内容 → 触发 **palette / intrabc**，那会挡住你想验的东西。所以默认用例
显式关掉它们（`--enable-palette=0 --enable-intrabc=0 --tune-content=film`）。

屏幕内容流这两样现在**各自有专属用例与专属素材**（都已在 2026-10-01 对拍通过）：
`--palette`（`screen_content`，全局 ≤8 色）与 `--intrabc`（`intrabc_content`，
周期 256px 的重复内容）；坑见对应的用例注释。
"""
import argparse
import os
import subprocess
import sys
import numpy as np
import av


def _layer(scale, lo, hi, rng, h, w):
    s = rng.integers(lo, hi, size=(h // scale + 1, w // scale + 1)).astype(np.float64)
    return np.kron(s, np.ones((scale, scale)))[:h, :w]


def content(yy, xx, seed=0):
    """自然感内容：多尺度柔和块噪声 + 径向渐变"""
    h, w = yy.shape
    rng = np.random.default_rng(seed)
    v = (
        _layer(64, 60, 200, rng, h, w) * 0.5
        + _layer(16, 60, 200, rng, h, w) * 0.3
        + _layer(4, 80, 180, rng, h, w) * 0.2
    )
    cy, cx = h * 0.5, w * 0.5
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / max(h, w)
    return np.clip(v * (1 - 0.4 * r) + 30, 0, 255).astype(np.uint8)


def moving_object(yy, xx, i):
    """**interintra**（§7.11.2 的 `IsInterIntra`）的引导内容。

    为什么非它不可：interintra 是"一块里帧内、帧间各管一半"的预测，光有平移或光有
    遮挡都触发不了它。这里造的是**物体尾部露出新背景**：

    - 背景**水平方向平滑**（帧内的 H/V 预测接得上）；
    - 物体是一组**高对比竖条纹**（帧间能跟住，但一旦跟错就整片错）；
    - 物体每帧右移 24px、下移 8px —— 都**不是 4 的倍数**，免得边缘正好压在块边界上
      （压上了，编码器用普通的矩形分割就够了，用不着楔形掩码）。

    于是物体尾部那条带子里：左半是新露出的背景（帧内好、帧间坏），右半是物体
    （帧间好、帧内坏）—— 正好是楔形 / smooth 两种 interintra 掩码的形状。
    """
    h, w = yy.shape
    bg = content(yy, xx, 0).astype(np.int32)
    ox = int(w * 0.10) + i * 24
    oy = int(h * 0.18) + i * 8
    oh, ow = int(h * 0.55), int(w * 0.42)
    obj = np.where(((xx // 6) % 2) == 0, 40, 215).astype(np.float64) + 0.35 * content(yy, xx, 7)
    out = bg.copy()
    x0, x1 = max(0, ox), min(w, ox + ow)
    y0, y1 = max(0, oy), min(h, oy + oh)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = obj[y0:y1, x0:x1]
    return np.clip(out, 0, 255).astype(np.uint8)


# UI 截图的 8 组颜色（Y / U / V 各一份）。色差都拉得够开，免得编码器把它们量化到一起。
_UI_Y = np.array([28.0, 76.0, 124.0, 172.0, 220.0, 252.0, 52.0, 196.0])
_UI_U = np.array([96.0, 128.0, 160.0, 176.0, 112.0, 84.0, 144.0, 68.0])
_UI_V = np.array([128.0, 96.0, 160.0, 64.0, 176.0, 112.0, 48.0, 192.0])
# 面板：整块纯色矩形，四条边都落在 8 的倍数上。
_UI_PANELS = ((16, 24, 96, 216, 4), (104, 32, 184, 224, 1),
              (192, 16, 264, 208, 5), (272, 40, 344, 232, 2))
# 8 色竖条带的左边界（x 右边这一整片都是 8px 宽的 8 色竖条）。**必须避开所有面板**，
# 否则面板会把它盖掉，凑不出"一块 8 色"。
_UI_CHECK_X = 320


def _ui_index(yy, xx, i):
    """UI 截图的"色号图"：每个位置属于哪一种颜色（逐坐标可求值，亮度/色度共用）。

    三段拼起来，为的是把 `palette_size_*` 的**各个档位**都逼出来：

    - 32×32 的大格子（左侧）⇒ 大块里只有 2~4 色（最常见的档位 2/3）；
    - 8px 宽的 8 色竖条带（`x >= _UI_CHECK_X`）⇒ 越宽的块里颜色越多：8 宽的块 1 色、
      32 宽 4 色、64 宽 8 色。这是把 `palette_size_*` 的**后半段档位**逼出来的关键，
      因为 8px 正好等于最小块宽，编码器没法靠"切小"把颜色数降到 2；
    - 纯色面板 ⇒ 大量单色块。
    """
    xs = xx + i * 12
    # 底：32×32 的格子按 (2*gy + gx) % 8 上色 —— 64×64 的块跨 2×2 格 ⇒ 最多 4 色
    k = ((yy // 32) * 2 + (xs // 32)) % _UI_Y.size
    # 右侧带：8px 宽的 8 色竖条，且**每 64 行整体错开 3 格**。错开这一下很关键 ——
    # 纯竖条会被方向预测几乎零残差地预测掉（实测码流小十倍、palette 块也没了），
    # 加上竖向的相位跳变才逼得编码器去用 palette。
    k = np.where(xs >= _UI_CHECK_X, (xs // 8 + (yy // 64) * 3) % _UI_Y.size, k)
    for py0, px0, py1, px1, kk in _UI_PANELS:
        k = np.where((yy >= py0) & (yy < py1) & (xs >= px0) & (xs < px1), kk, k)
    return k


def screen_content(yy, xx, i=0):
    """**屏幕内容**（UI 截图感）的引导内容 —— palette（§5.11.18 / §5.11.19 / §7.11.4）
    的专属素材。返回 `(Y, U, V)` 三个平面。

    为什么非它不可：palette 的启用判据是"块内的**颜色数** <= 8"，libaom 是拿着
    颜色直方图去和 `DC_PRED` 等常规帧内模式比 RD 才决定用不用的。自然纹理（哪怕
    是"大色块 + 一点点噪声"）几乎永远超过 8 色 —— 拿它验 palette，一个 palette 块
    都不会出现，正着跑也对得上，那是**白验**（分不清"没用"与"用对了"）。

    所以这里刻意造"UI 截图"：**全局只用 8 种纯色**，色块边界落在 8 的倍数上。

    **色度也要有颜色**（而不是恒 128）：`palette_tokens` 里色度是**另一段**语法 ——
    索引图按 `subsampling_x/y` 右移，V 分量还有"增量 + 符号位 + 环形回绕"这条与 Y/U
    都不同的编码。色度恒 128 的流里这些分支一次都走不到，等于没验。

    每帧整体右移 12 像素（**不是 8 的倍数**）：多帧用例里左边会露出新内容，逼出
    真正的 intra 块，从而让 palette 也出现在 inter 帧的 `inter_frame_mode_info` 里
    （那条路径与关键帧的 `intra_frame_mode_info` 是分开的两个调用点）。
    """
    k = _ui_index(yy, xx, i)
    # 文字状短线：2 像素高的纯色横条，**只改亮度不改色度**（白字黑底在屏幕上本来
    # 就没有色度边缘），顺便制造大量"2 色块"。只画在棋盘带以外，免得把那里凑不满
    # 8 色。
    band = ((yy % 16) < 2) & (((xx + i * 12) // 8) % 5 != 0) & ((xx + i * 12) < _UI_CHECK_X)
    y = _UI_Y[np.where(band, 0, k)]
    kc = _ui_index(yy[::2, ::2], xx[::2, ::2], i)
    return (y.astype(np.uint8),
            _UI_U[kc].astype(np.uint8),
            _UI_V[kc].astype(np.uint8))


def intrabc_content(yy, xx, i=0):
    """**周期性屏幕内容** —— intrabc（§5.11.11 的 `use_intrabc`，预测见 §7.11.3.1
    的 `refIdx = -1`）的专属素材。返回 `(Y, U, V)`。

    为什么非它不可：intrabc 的本质是"这一块和**前面某个已经重建好的位置**长得
    一模一样，与其花残差不如直接拷"。所以内容必须**周期重复**，而且周期要够大：

    - 下方 `INTRABC_DELAY_SB64 = 4` 个 64×64 块之内不许当源，且源必须落在当前块
      的**栅格序之前**。`is_mv_valid` 另有一条"同超块行内不能离得太近"的约束
      （`gradient` 那几行）。周期取 **256 像素**（= 4 个 64 块行）刚好够。
    - 重复的那一块还得是**帧内预测和 palette 都啃不动**的。纯色块会被 DC/方向预测
      白拿（一轮实测：纯竖条的码流小十倍、intrabc 一次都没出现），少色块又会被
      palette 抢走。所以下半屏用**确定性噪声**（每 256×256 重复一次，块内颜色数
      远超 8）—— 只有"从上面 256 像素处拷"这一条路。

    上半屏仍保留 8 色平铺：让 palette 与 intrabc **同场竞争**，别把 palette 那条
    路验没了。整体再按帧右移 12 像素（不是 8 的倍数），顺带覆盖不同相位。
    """
    xs = xx + i * 12
    ym, xm = yy % 256, xs % 256
    # 半透明的"确定性噪声"：只用整数运算，保证同一 (ym, xm) 每次都一样
    n = ((ym.astype(np.int64) * 73856093) ^ (xm.astype(np.int64) * 19349663)) % 200
    k = ((ym // 32) * 2 + (xm // 32)) % _UI_Y.size
    y = np.where(ym >= 128, 28 + n, _UI_Y[k])
    cym, cxm = ym[::2, ::2], xm[::2, ::2]
    cn = ((cym.astype(np.int64) * 73856093) ^ (cxm.astype(np.int64) * 19349663)) % 200
    ck = ((cym // 32) * 2 + (cxm // 32)) % _UI_Y.size
    loud = cym >= 128
    return (
        y.astype(np.uint8),
        np.where(loud, 64 + cn // 2, _UI_U[ck]).astype(np.uint8),
        np.where(loud, 192 - cn // 2, _UI_V[ck]).astype(np.uint8),
    )


def write_y4m(path, w, h, frames, chroma=None):
    """写 4:2:0 的 y4m。`frames` 是逐帧的亮度平面；`chroma` 给了就逐帧用 `(u, v)`，
    否则用恒 128（早期用例都靠这个默认值）。"""
    if chroma is None:
        chroma = [(np.full((h // 2, w // 2), 128, np.uint8),) * 2] * len(frames)
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F30:1 Ip A1:1 C420jpeg\n".encode())
        for y, (u, v) in zip(frames, chroma):
            f.write(b"FRAME\n")
            f.write(y.tobytes())
            f.write(u.astype(np.uint8).tobytes())
            f.write(v.astype(np.uint8).tobytes())


def ivf_to_mp4(ivf, mp4):
    inp = av.open(ivf)
    vs = inp.streams.video[0]
    out = av.open(mp4, "w")
    s = out.add_stream("av1", rate=30)
    s.width, s.height, s.pix_fmt = vs.width, vs.height, "yuv420p"
    s.codec_context.extradata = vs.codec_context.extradata
    for pkt in inp.demux(vs):
        if pkt.size == 0 or pkt.pts is None:
            continue
        pkt.stream = s
        out.mux(pkt)
    out.close()
    inp.close()


def ref_frames(mp4, grain=1):
    c = av.open(mp4)
    vs = c.streams.video[0]
    refs = []
    if grain == 0:
        ctx = av.CodecContext.create("libdav1d", "r")
        ctx.options = {"filmgrain": "0"}
        ctx.extradata = vs.codec_context.extradata
        ctx.open()
        for pkt in c.demux(vs):
            for fr in ctx.decode(pkt):
                refs.append(fr.to_ndarray(format="yuv420p").copy())
    else:
        for fr in c.decode(vs):
            refs.append(fr.to_ndarray(format="yuv420p").copy())
    c.close()
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/refs")
    ap.add_argument("--aomenc", default="/tmp/libaom-3.9.0/build_g/aomenc")
    ap.add_argument("--svt", action="store_true", help="只用 PyAV 的 libsvtav1（不用 aomenc）")
    ap.add_argument("--grain", action="store_true",
                    help="改出胶片颗粒用例（--film-grain-test=N，见 grain_cases）")
    ap.add_argument("--superres", action="store_true",
                    help="改出超分上采样用例（§7.16：--superres-mode / --superres-denominator）")
    ap.add_argument("--interintra", action="store_true",
                    help="改出帧内-帧间混合预测用例（§7.11.2 的 IsInterIntra）")
    ap.add_argument("--palette", action="store_true",
                    help="改出调色板用例（§5.11.18 / §5.11.19 / §7.11.4，屏幕内容）")
    ap.add_argument("--intrabc", action="store_true",
                    help="改出帧内块拷贝用例（§5.11.5 的 use_intrabc，屏幕内容）")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 用例表：(名字, 宽, 高, 帧数, 额外 aomenc 开关)
    # 覆盖"从简到繁"：小尺寸 → 1080p；单帧 → 多帧 inter；超块 64/128；LR 开/关
    cases = [
        ("aom_640_f1", 640, 360, 1, []),
        ("aom_640_f6", 640, 360, 6, ["--kf-max-dist=32"]),
        ("aom_1080_f1", 1920, 1080, 1, []),
        ("aom_1080_f4", 1920, 1080, 4, ["--kf-max-dist=32"]),
        ("aom_1080_lr0", 1920, 1080, 1, ["--enable-restoration=0"]),
        ("aom_1080_sb64", 1920, 1080, 1, ["--sb-size=64"]),
        ("aom_1080_nocdef", 1920, 1080, 1, ["--enable-cdef=0"]),
    ]

    # 胶片颗粒用例（§7.18）。两条必须知道的规矩：
    #
    # 1. **不能带 `--tune-content=film`**。那个开关会让 libaom 在打包前调用
    #    `set_grain_syn_params()`（`av1/encoder/encoder.c`）**覆盖**掉
    #    `cm->film_grain_params`，`--film-grain-test` 就被无声吞掉了 —— 只会
    #    拿到 ypts=1 / uvpts=(1,1) / ar_coeff_lag=0 这条最小形态。
    # 2. **`--kf-max-dist=32`**：`update_grain` 只出现在 inter 帧的码流里，
    #    全关键帧（默认 `--kf-max-dist=1`）永远走不到 `load_grain_params`。
    #
    # `--film-grain-test=N` 取的是 libaom 自带的 16 组 `film_grain_test_vectors`
    # （`av1/encoder/grain_test_vectors.h`），逐组覆盖不同形态：
    #   t01 = lag2/14点表/clip1/overlap0/cb,cr_mult≠128   t02 = lag3
    #   t05 = **update_grain=0**（第 2 帧起 ref_idx 0→1）  t06 = 无 UV 点表
    #   t15 = **chroma_scaling_from_luma=1**              t16 = grain_scale_shift=2
    grain_cases = [
        ("aom_640_g_t01", 640, 360, 4, ["--film-grain-test=1"]),
        ("aom_640_g_t02", 640, 360, 4, ["--film-grain-test=2"]),
        ("aom_640_g_t05", 640, 360, 4, ["--film-grain-test=5"]),
        ("aom_640_g_t06", 640, 360, 4, ["--film-grain-test=6"]),
        ("aom_640_g_t15", 640, 360, 4, ["--film-grain-test=15"]),
        ("aom_640_g_t16", 640, 360, 4, ["--film-grain-test=16"]),
        # 内建降噪颗粒（lag=0、ar_coeff_lag=0）—— 与上面互补
        ("aom_640_g_dn", 640, 360, 4, ["--denoise-noise-level=20"]),
        # alt-ref → `show_existing_frame`：颗粒必须按**参考槽里存的参数**补
        ("aom_640_g_arf", 640, 360, 40, ["--denoise-noise-level=20", "--lag-in-frames=25",
                                         "--auto-alt-ref=1", "--enable-fwd-kf=1",
                                         "--kf-min-dist=16", "--kf-max-dist=64"]),
    ]
    # 超分上采样用例（§7.16）。要点：
    #
    # 1. `--superres-mode=1` 是**固定**模式（每帧都用同一个分母），另两种
    #    (`2` random / `3` qthresh) 会让分母逐帧变，适合当"多分母"的补充覆盖；
    #    分母越大压得越狠（`FrameWidth = UpscaledWidth * 8 / denom`），
    #    范围 9~16。
    # 2. **关键帧也要压**：`--superres-kf-denominator` 只管关键帧，不给的话
    #    关键帧可能不压 —— 而关键帧恰恰是唯一能单帧对拍的那一帧，所以必须一起给。
    # 3. 横向上采样和环路恢复在管线里相邻（去块 → CDEF → 上采样 → 环路恢复），
    #    两组开关交叉覆盖一下，免得"上采样对但和 LR 拼错了"被漏掉。
    #
    # **宽度取 648 / 1920，不用 640** —— 这条很关键：`FrameWidth` 若不是 8 的倍数，
    # 超块会伸到帧右边以外（`MiCols*4 > FrameWidth`），而**帧右边界那一小块像素的
    # 处理是另一个已知问题**（见下面 `aom_640_sr_fix12`），会把上采样本身的表现
    # 盖住。所以要"钉死上采样"就得先选一个 `FrameWidth` 是 8 的倍数的尺寸：
    #   648 + denom 12 → 432（8×54）  648 + denom 9 → 576（8×72）
    #   1920 + denom 12 → 1280（8×160）
    # 这几个的 `stepX` 都不是整数（不是整倍数缩放），正好把取整路径也走上。
    superres_cases = [
        ("aom_648_sr_fix12", 648, 360, 1,
         ["--superres-mode=1", "--superres-denominator=12", "--superres-kf-denominator=12"]),
        ("aom_648_sr_fix9", 648, 360, 1,
         ["--superres-mode=1", "--superres-denominator=9", "--superres-kf-denominator=9"]),
        ("aom_1080_sr_fix12", 1920, 1080, 1,
         ["--superres-mode=1", "--superres-denominator=12", "--superres-kf-denominator=12"]),
        ("aom_640_sr_fix16", 640, 360, 1,
         ["--superres-mode=1", "--superres-denominator=16", "--superres-kf-denominator=16"]),
        ("aom_640_sr_fix12", 640, 360, 1,
         ["--superres-mode=1", "--superres-denominator=12", "--superres-kf-denominator=12"]),
        ("aom_640_sr_nolr", 640, 360, 1,
         ["--superres-mode=1", "--superres-denominator=12", "--superres-kf-denominator=12",
          "--enable-restoration=0"]),
    ]
    # interintra 用例（§7.11.2）。三个必须知道的坑：
    #
    # 1. **`--cpu-used` 必须 <= 2**。libaom 会在 `set_good_speed_feature_framesize_dependent`
    #    里做 `enable_interintra_compound &= (disable_interintra_wedge_var_thresh != UINT_MAX)`
    #    —— 实测 1280x720 下 cpu-used 0/1/2 写成 1，3/4/6 写成 **0**。写成了 0 的流里
    #    一个 interintra 块都不可能有，怎么调内容都验不出来（这轮就白查了一轮）。
    # 2. **`min(w, h) >= 480`**：同一个函数里，低于 480p 的分支直接把阈值设成
    #    `UINT_MAX` —— 640×360 无论 cpu-used 取多少都是关的。
    # 3. **高度要取 4 的倍数**：`MiRows = 2 * ((FrameHeight + 7) >> 3)` 是**向上取整**的，
    #    720 恰好整除（180×4），540 却会多出 4 像素 —— 帧底那一行超块伸出区是另一个
    #    已知问题（见 README 的"帧右边界超块伸出区"），会把 interintra 的表现盖住。
    #
    # 帧数取 6 是实测出来的：物体走 10 帧（240px）之后编码器就不再选 interintra 了。
    interintra_cases = [
        ("aom_720_ii_f6", 1280, 720, 6,
         ["--cpu-used=2", "--cq-level=45", "--kf-max-dist=32"], moving_object),
    ]
    # palette 用例（§5.11.18 / §5.11.19 / §7.11.4）。三个必须知道的点：
    #
    # 1. **必须 `--tune-content=screen`**：libaom 只在 `AOM_CONTENT_SCREEN` 下把
    #    `allow_screen_content_tools` 置 1（`av1/encoder/encoder.c` 的
    #    `av1_set_screen_content_options`）；置 0 的序列头里 `palette_mode_info()`
    #    根本不会被调用，正着跑一致也证明不了任何事。
    # 2. **素材必须是"少色"的**：palette 的启用判据是块内颜色数 <= 8。自然纹理
    #    （哪怕是"大色块 + 噪声"）几乎永远超 8 色。所以用上面的 `screen_content()`。
    # 3. **`--enable-intrabc=0`**：屏幕内容下 libaom 默认把 intrabc 也开上。intrabc
    #    是下一步单独验的东西，混在一起就分不清残差该记在谁头上。
    #
    # 单帧用例验 `intra_frame_mode_info` 那条调用点；六帧用例（第 2 帧起是 inter 帧）
    # 顺带验 `inter_frame_mode_info` 那条 —— 左边露出的新内容要靠 intra 块补上。
    palette_cases = [
        ("aom_640_pal_f1", 640, 360, 1, [], screen_content),
        ("aom_720_pal_f1", 1280, 720, 1, [], screen_content),
        ("aom_720_pal_f6", 1280, 720, 6, ["--kf-max-dist=32"], screen_content),
    ]
    # intrabc 用例（§5.11.5 的 `use_intrabc`）。三个必须知道的点：
    #
    # 1. **只有帧内帧才可能开 `allow_intrabc`**。规范在 `frame_header_obu` 里把它
    #    写在 `if ( frame_is_intra )` 里面，libaom 的 `av1_set_screen_content_options`
    #    也只在无参考帧时置 1。所以这批用例必须**全关键帧**（`--kf-max-dist=1`）——
    #    否则除第 0 帧外根本走不到这条语法。
    # 2. 帧头那一层跟着 intrabc 一起被关掉的东西（CDEF / 环路恢复 / 去块 level）
    #    我们的 `params.rs` 已经处理，这里不用额外开关。
    # 3. 素材要**周期性重复且颜色多**（见 `intrabc_content`）：纯色/少色会被
    #    方向帧内预测或 palette 抢走，白验。
    # 4. **`--cpu-used` 必须 <= 2**，又一个"静默关掉"的坑：`speed_features.c` 的
    #    `set_good_speed_features_framesize_independent` 里 `if (speed >= 3)
    #    sf->mv_sf.use_intrabc = 0;`，而 `rd_pick_intrabc_mode_sb` 一进门就查
    #    `!cpi->sf.mv_sf.use_intrabc` → 直接返回 INT64_MAX。cpu-used >= 3 的流里
    #    一个 intrabc 块都不可能有（本轮的 640/720 三条用例一开始就是这么白验的：
    #    正着跑逐像素一致、`AV1_NO_INTRABC=1` 也逐像素一致 —— 后者才是判据）。
    intrabc_cases = [
        ("aom_640_ibc_f1", 640, 360, 1, ["--cpu-used=2"], intrabc_content),
        ("aom_720_ibc_f1", 1280, 720, 1, ["--cpu-used=2"], intrabc_content),
        ("aom_720_ibc_f4", 1280, 720, 4, ["--cpu-used=2"], intrabc_content),
    ]

    if args.grain:
        cases = grain_cases
    elif args.superres:
        cases = superres_cases
    elif args.interintra:
        cases = interintra_cases
    elif args.palette:
        cases = palette_cases
    elif args.intrabc:
        cases = intrabc_cases

    for case in cases:
        name, w, h, n, extra = case[:5]
        cfn = case[5] if len(case) > 5 else content
        y4m, ivf = f"/tmp/{name}.y4m", f"/tmp/{name}.ivf"
        mp4 = os.path.join(args.out, f"{name}.mp4")
        yy, xx = np.mgrid[0:h, 0:w]
        # 内容函数可以只给亮度（返回数组），也可以连色度一起给（返回 `(Y, U, V)`）——
        # 后者是 palette 用例必需的，理由见 `screen_content()` 的说明。
        made = [cfn(yy, xx, i) for i in range(n)]
        if isinstance(made[0], tuple):
            luma = [m[0] for m in made]
            chroma = [(m[1], m[2]) for m in made]
        else:
            luma, chroma = made, None
        write_y4m(y4m, w, h, luma, chroma)

        if args.svt:
            c = av.open(mp4, "w")
            s = c.add_stream("libsvtav1", rate=30)
            s.width, s.height, s.pix_fmt = w, h, "yuv420p"
            s.options = {"preset": "4", "crf": "40", "svtav1-params": "keyint=32"}
            u = np.full((h // 2, w // 2), 128, np.uint8)
            for i in range(n):
                f = av.VideoFrame(w, h, "yuv420p")
                for k, (pl, pw, ph) in enumerate(((content(yy, xx, i), w, h), (u, w // 2, h // 2), (u, w // 2, h // 2))):
                    dst = np.frombuffer(f.planes[k], dtype=np.uint8).reshape(ph, -1)
                    dst[:, :pw] = pl
                for pkt in s.encode(f):
                    c.mux(pkt)
            for pkt in s.encode(None):
                c.mux(pkt)
            c.close()
        else:
            cmd = [
                args.aomenc, "--codec=av1", "--passes=1", f"--limit={n}", "--good",
                "--cpu-used=4", "--end-usage=q", "--cq-level=40", "--sb-size=128",
                "--enable-palette=0", "--enable-intrabc=0",
            ]
            if args.palette:
                # 屏幕内容工具要显式打开：只有 `--tune-content=screen` 会让 libaom 把
                # `allow_screen_content_tools` 置 1，palette 才可能出现。
                # intrabc 仍然关着（见 palette_cases 的第 3 条）。
                cmd += ["--tune-content=screen", "--enable-palette=1"]
            elif args.intrabc:
                # 两条屏幕内容工具**都开着**：palette 在同场竞争（少色块归它），
                # intrabc 抢"重复且多色"的块。只开 intrabc 反而会让少色块落到
                # 别的模式上，浪费覆盖。
                cmd += ["--tune-content=screen", "--enable-palette=1", "--enable-intrabc=1"]
            elif not args.grain:
                # `--tune-content=film` 在非颗粒用例里用来躲开 palette/intrabc 的误判；
                # 颗粒用例绝不能带它（见上面对 grain_cases 的说明）。
                cmd.append("--tune-content=film")
            cmd += ["--kf-max-dist=1", "--kf-min-dist=1", "--threads=4"] + extra + ["-o", ivf, y4m]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"{name}: aomenc 失败\n  {(r.stderr or r.stdout)[-400:]}")
                continue
            ivf_to_mp4(ivf, mp4)

        for f in (y4m, ivf):
            if os.path.exists(f):
                os.remove(f)
        refs = ref_frames(mp4)
        np.save(os.path.join(args.out, f"{name}_ref.npy"), np.stack(refs))
        print(f"{name:<16} {w}x{h} {n} 帧  码流 {os.path.getsize(mp4):>8}B  参考 {len(refs)} 帧")

    print(f"\n完成 → {args.out}")
    print("下一步：")
    if args.grain:
        print(f"  zuvrust decode {args.out}/aom_640_g_t01.mp4 --count 4 --out=/tmp/out_g1 --json")
        print(f"  python3 refcmp.py {args.out}/aom_640_g_t01.mp4 --dir=/tmp/out_g1 --n=4   # grain=1（默认）")
    elif args.superres:
        print(f"  zuvrust decode {args.out}/aom_640_sr_fix12.mp4 --count 1 --out=/tmp/out_sr --json")
        print(f"  python3 refcmp.py {args.out}/aom_640_sr_fix12.mp4 --dir=/tmp/out_sr --n=1")
    elif args.interintra:
        print(f"  zuvrust decode {args.out}/aom_720_ii_f6.mp4 --count 6 --out=/tmp/out_ii --json")
        print(f"  python3 refcmp.py {args.out}/aom_720_ii_f6.mp4 --dir=/tmp/out_ii --n=6")
        print(f"  # 隔离实验（证明这条流确实用了 interintra）：摘掉之后应当开始出现差异")
        print(f"  AV1_NO_INTERINTRA=1 zuvrust decode {args.out}/aom_720_ii_f6.mp4 --count 6 --out=/tmp/out_ii_off --json")
        print(f"  python3 refcmp.py {args.out}/aom_720_ii_f6.mp4 --dir=/tmp/out_ii_off --n=6")
    elif args.palette:
        print(f"  zuvrust decode {args.out}/aom_640_pal_f1.mp4 --count 1 --out=/tmp/out_pal --json")
        print(f"  python3 refcmp.py {args.out}/aom_640_pal_f1.mp4 --dir=/tmp/out_pal --n=1")
        print(f"  zuvrust decode {args.out}/aom_720_pal_f6.mp4 --count 6 --out=/tmp/out_pal6 --json")
        print(f"  python3 refcmp.py {args.out}/aom_720_pal_f6.mp4 --dir=/tmp/out_pal6 --n=6")
    elif args.intrabc:
        print(f"  zuvrust decode {args.out}/aom_720_ibc_f4.mp4 --count 4 --out=/tmp/out_ibc --json")
        print(f"  python3 refcmp.py {args.out}/aom_720_ibc_f4.mp4 --dir=/tmp/out_ibc --n=4")
        print(f"  # 隔离实验（证明这条流确实用了 intrabc）：摘掉拷贝之后应当出现大面积差异")
        print(f"  AV1_NO_INTRABC=1 zuvrust decode {args.out}/aom_720_ibc_f4.mp4 --count 4 --out=/tmp/out_ibc_off --json")
        print(f"  python3 refcmp.py {args.out}/aom_720_ibc_f4.mp4 --dir=/tmp/out_ibc_off --n=4")
    else:
        print(f"  zuvrust decode {args.out}/aom_1080_f1.mp4 --count 1 --out=/tmp/out1 --json")
        print(f"  python3 refcmp.py {args.out}/aom_1080_f1.mp4 --dir=/tmp/out1 --grain=0")


if __name__ == "__main__":
    sys.exit(main())
