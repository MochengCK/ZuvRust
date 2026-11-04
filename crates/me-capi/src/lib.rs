//! me-capi —— 嵌入式播放会话层。
//!
//! 把 `me-stream` 的解封装与 `me-decode` 的解码包成"一个会话对象"，
//! 通过 C ABI 交给桌面播放器**进程内**调用：
//!
//! ```text
//!   宿主（Qt/QML）
//!        │  zuvsrc_open / play / seek / select_track
//!        ▼
//!   ┌──────────────── me-capi ────────────────┐
//!   │ 视频管线线程：Player → H.264/AV1 解码 → │
//!   │               NV12 帧队列（有界 6 帧）  │
//!   │ 音频管线线程：Player → AAC/MP3/… 解码 → │
//!   │               S16 立体声环形缓冲（有界）│
//!   └────────────────────────────────────────┘
//!        │  zuvsrc_acquire_video / read_audio
//!        ▼
//!   宿主的 Metal 渲染与 CoreAudio 输出（时钟在宿主侧）
//! ```
//!
//! ## 为什么是"双管线"而不是"一条解封装线程 + 两个解码线程"
//!
//! 解封装器是 `&mut` 的，单线程解复用意味着 AV1 软解一帧 50 ms 时音频跟着卡；
//! 而本地文件的两次打开成本只是多解析一次文件头（页缓存共享，没有额外 IO）。
//! 换来的是**音频永远不被视频解码拖住** —— 这是"播放流畅"里最要紧的一条。
//!
//! ## 线程与内存边界
//!
//! - 队列都有上限：视频 6 帧（1080p 约 18 MB），音频 1.5 秒。解到上限就停，
//!   所以内存占用与片长无关，只与"缓冲窗口"有关。
//! - 暂停时两个线程都阻塞在条件变量上：不解码、不读盘，CPU 归零。
//! - 寻址是非阻塞的：递增 `seq`，两个线程各自醒来、丢弃队列、重开解码器。
//!   宿主用 [`zuvsrc_seek_done`] 判断是否落稳。
//! - 每个 FFI 入口都做了参数校验；解码线程整体包在 `catch_unwind` 里，
//!   坏码流最多让线程报错，不会把宿主进程带走。

mod subtitle;

use std::collections::VecDeque;
use std::ffi::{c_char, c_int, CStr};
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicI64, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use me_core::{CodecId, MediaKind, ReadAt, Timebase};
use me_decode::audio::AudioDecoder;
use me_decode::{VideoDecoder, VideoFrame};
use me_mux::convert;
use me_stream::{OpenOptions, OpenedSource, Player, PlayerConfig};

// ─────────────────────────────────────────────────────────────
// 常量
// ─────────────────────────────────────────────────────────────

/// 视频队列上限（帧）。
///
/// 深度要能吸收**解码器自身的输出延迟**：VideoToolbox 异步解码 + B 帧重排，
/// 输出比输入滞后约 0.3 s。队列太浅（6 帧 = 200 ms）时，解码线程刚填满就阻塞，
/// 输出永远落后音频时钟 300 ms，于是每一帧都被判"太旧"丢掉 —— 实测画面 4 fps。
/// 16 帧 ≈ 530 ms，正好覆盖这段延迟；1080p 约 48 MB，可以接受。
fn video_queue_frames(codec: Option<&CodecId>) -> usize {
    use std::sync::OnceLock;
    static CAP: OnceLock<Option<usize>> = OnceLock::new();
    let forced = *CAP.get_or_init(|| {
        std::env::var("ZUV_QUEUE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v >= 1 && *v <= 4096)
    });
    if let Some(v) = forced {
        return v;
    }
    // 自研软解（AV1）没有硬件那 0.3 s 的输出延迟，队列放浅一点：
    // 队列越深，解码器要同时持有的帧越多，缓冲池周转不过来反而更慢。
    match codec {
        Some(CodecId::Av1) => 8,
        _ => 16,
    }
}
/// 音频缓冲上限（微秒）。1.5 s 足够吸收一次 AV1 解码抖动，又不会拖慢 seek。
const AUDIO_BUFFER_US: i64 = 1_500_000;
/// 解封装缓冲：一个采样进来就处理，不做"边下边播"的大缓冲。
const PLAYER_MAX_BUFFER_US: i64 = 500_000;
const PLAYER_MAX_BUFFER_PACKETS: usize = 256;

const STATE_IDLE: u8 = 0;
const STATE_LOADED: u8 = 1;
const STATE_PLAYING: u8 = 2;
const STATE_PAUSED: u8 = 3;
const STATE_ERROR: u8 = 7;

/// 字幕解析状态
const SUB_NONE: u8 = 0;
const SUB_LOADING: u8 = 1;
const SUB_READY: u8 = 2;
const SUB_ERROR: u8 = 3;

pub const ZUV_OK: c_int = 0;
pub const ZUV_ERR: c_int = -1;
pub const ZUV_ERR_ARG: c_int = -2;
pub const ZUV_ERR_STATE: c_int = -3;

const TRACK_NONE: i32 = -1;

// ─────────────────────────────────────────────────────────────
// C ABI 数据结构
// ─────────────────────────────────────────────────────────────

/// 轨道信息（`codec` / `language` / `title` 是定长内联字符串，宿主拷走即可）。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ZuvTrackInfo {
    pub track_id: i32,
    /// 0 = 视频，1 = 音频，2 = 字幕
    pub kind: i32,
    pub codec: [u8; 16],
    pub language: [u8; 16],
    pub title: [u8; 96],
    pub width: u32,
    pub height: u32,
    pub frame_rate: f64,
    pub sample_rate: u32,
    pub channels: u32,
    pub bit_rate: i64,
}

impl Default for ZuvTrackInfo {
    fn default() -> Self {
        Self {
            track_id: TRACK_NONE,
            kind: 0,
            codec: [0; 16],
            language: [0; 16],
            title: [0; 96],
            width: 0,
            height: 0,
            frame_rate: 0.0,
            sample_rate: 0,
            channels: 0,
            bit_rate: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ZuvMediaInfo {
    pub container: [u8; 32],
    pub duration_us: i64,
    pub bit_rate: i64,
    pub track_count: i32,
    pub video_index: i32,
    pub audio_index: i32,
    pub tracks: [ZuvTrackInfo; 16],
}

impl Default for ZuvMediaInfo {
    fn default() -> Self {
        Self {
            container: [0; 32],
            duration_us: 0,
            bit_rate: 0,
            track_count: 0,
            video_index: TRACK_NONE,
            audio_index: TRACK_NONE,
            tracks: [ZuvTrackInfo::default(); 16],
        }
    }
}

/// 一帧 NV12（`data` 指向会话内部所有，`zuvsrc_release_video` 之前一直有效）。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ZuvFrame {
    pub data: *const u8,
    pub bytes: u32,
    pub width: u32,
    pub height: u32,
    pub pts_us: i64,
}

// ─────────────────────────────────────────────────────────────
// 内部状态
// ─────────────────────────────────────────────────────────────

/// 一条轨道的解码参数（打开时算好，线程里直接取用）。
struct TrackDef {
    id: u32,
    codec: CodecId,
    /// 视频：avcC / av1C；音频：AudioSpecificConfig 等
    config: Vec<u8>,
    timebase: Timebase,
}

struct Ctrl {
    playing: bool,
    closed: bool,
    /// 每次 seek / 切轨递增；两条管线看到变化就重来
    seq: u64,
    target_us: i64,
    v_track: i32,
    a_track: i32,
    /// 视频管线"跳转对齐"的代次与目标位置（只影响视频，音频继续播）
    ///
    /// 为什么需要它：解码跟不上时，画面会落后音频越来越远。丢帧只会让画面
    /// 定格 —— 解码器仍在原地一帧帧解，永远追不回来。必须真的把**视频**这条
    /// 管线跳到当前位置，音频不受影响。
    v_resync: u64,
    v_resync_target: i64,
    /// 这次 seek 是"预览"用途：落点在哪就从哪出帧，不必追到目标时间。
    /// 悬停缩略图要的是"立刻给一张画面"，追帧反而会超时。
    preview: bool,
}

struct Vq {
    frames: VecDeque<VideoFrame>,
    /// 宿主正在使用的那一帧（指针由它持有，释放前不能动）
    leased: Option<VideoFrame>,
    seq: u64,
    closed: bool,
}

struct AudioSeg {
    samples: Vec<i16>,
    /// 已消费的采样数（交织下标）
    pos: usize,
    rate: u32,
    channels: u32,
}

struct Aq {
    segs: VecDeque<AudioSeg>,
    buffered_us: i64,
    seq: u64,
    closed: bool,
    rate: u32,
    channels: u32,
}

/// 字幕：整条轨解析完的 cue 表（文本字幕总量很小，一次读完最省事）
struct Subs {
    state: u8,
    track: i32,
    /// 代次：换轨/重载时 +1，旧线程回来发现代次变了就丢弃结果
    generation: u64,
    cues: Vec<subtitle::Cue>,
    err: String,
}

struct Shared {
    ctrl: Mutex<Ctrl>,
    cv: Condvar,

    vq: Mutex<Vq>,
    cv_v: Condvar,
    aq: Mutex<Aq>,
    cv_a: Condvar,

    state: AtomicU8,
    /// 最后一条错误（CString：`zuvsrc_last_error` 直接把指针交出去）
    err: Mutex<std::ffi::CString>,

    has_video: AtomicBool,
    has_audio: AtomicBool,
    v_eof: AtomicBool,
    a_eof: AtomicBool,
    v_seek_done: AtomicU64,
    a_seek_done: AtomicU64,
    /// 视频管线寻址的**实际落点**（关键帧位置）与对应的 seq；
    /// 音频管线等它落定后再按同一位置寻址，音画才不会一个在关键帧、一个在目标点
    seek_landed: Mutex<(u64, i64)>,
    /// 本会话的视频队列深度（按编码选：硬解 16、软解 8）
    vq_cap: AtomicU64,
    v_decodes: AtomicU64,
    a_decodes: AtomicU64,
    /// 解码调用累计耗时（纳秒）与调用次数：用来判断"解码慢"还是"被上游限速"
    v_decode_ns: AtomicU64,
    subs: Mutex<Subs>,
    last_pts: AtomicI64,
}

impl Shared {
    fn new() -> Self {
        Self {
            ctrl: Mutex::new(Ctrl {
                playing: false,
                closed: false,
                seq: 0,
                target_us: 0,
                v_track: TRACK_NONE,
                a_track: TRACK_NONE,
                v_resync: 0,
                v_resync_target: 0,
                preview: false,
            }),
            cv: Condvar::new(),
            vq: Mutex::new(Vq {
                frames: VecDeque::new(),
                leased: None,
                seq: 0,
                closed: false,
            }),
            cv_v: Condvar::new(),
            aq: Mutex::new(Aq {
                segs: VecDeque::new(),
                buffered_us: 0,
                seq: 0,
                closed: false,
                rate: 0,
                channels: 0,
            }),
            cv_a: Condvar::new(),
            state: AtomicU8::new(STATE_IDLE),
            err: Mutex::new(std::ffi::CString::new("").unwrap()),
            has_video: AtomicBool::new(false),
            has_audio: AtomicBool::new(false),
            v_eof: AtomicBool::new(false),
            a_eof: AtomicBool::new(false),
            v_seek_done: AtomicU64::new(0),
            a_seek_done: AtomicU64::new(0),
            seek_landed: Mutex::new((0, -1)),
            vq_cap: AtomicU64::new(16),
            v_decodes: AtomicU64::new(0),
            a_decodes: AtomicU64::new(0),
            v_decode_ns: AtomicU64::new(0),
            subs: Mutex::new(Subs {
                state: SUB_NONE,
                track: -1,
                generation: 0,
                cues: Vec::new(),
                err: String::new(),
            }),
            last_pts: AtomicI64::new(0),
        }
    }

    fn fail(&self, msg: impl Into<String>) {
        let msg = msg.into();
        let mut e = self.err.lock().unwrap_or_else(|p| p.into_inner());
        if e.as_bytes().is_empty() {
            *e = std::ffi::CString::new(msg.replace('\0', " ")).unwrap_or_default();
        }
        drop(e);
        self.state.store(STATE_ERROR, Ordering::SeqCst);
        // 唤醒可能阻塞的调用方
        let mut c = self.ctrl.lock().unwrap_or_else(|p| p.into_inner());
        c.playing = false;
        drop(c);
        self.cv.notify_all();
        self.cv_v.notify_all();
        self.cv_a.notify_all();
    }
}

/// 会话句柄（`MediaEngine` 那层的一个成员）。
pub struct ZuvSource {
    sh: Arc<Shared>,
    /// 容器标签（打开时读一次，之后不变）
    tags: Mutex<me_core::MediaTags>,
    /// 视频显示旋转（0/90/180/270），打开时读一次
    rotation: AtomicU16,
    threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
    defs: Mutex<Arc<Vec<TrackDef>>>,
    duration_us: AtomicI64,
    path: Mutex<String>,
}

// ─────────────────────────────────────────────────────────────
// 小工具
// ─────────────────────────────────────────────────────────────

/// 把当前线程提到用户交互级 QoS（macOS）。
///
/// 播放管线线程是从宿主的主线程 spawn 出来的，默认会落到 utility 档：系统据此
/// 压低频、合并唤醒，硬解一帧从 12 ms 变成 50+ ms。播放是"必须按时序推进"的
/// 任务，这一档必须自己声明，不能指望继承。
#[cfg(target_os = "macos")]
fn promote_thread_qos() {
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    unsafe {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

#[cfg(not(target_os = "macos"))]
fn promote_thread_qos() {}

fn fill_str<const N: usize>(dst: &mut [u8; N], src: &str) {
    let bytes = src.as_bytes();
    let n = bytes.len().min(N - 1);
    dst[..n].copy_from_slice(&bytes[..n]);
    for b in dst[n..].iter_mut() {
        *b = 0;
    }
}

fn video_config(codec: &CodecId, extradata: &[u8]) -> Result<Vec<u8>, String> {
    if convert::looks_like_avcc(codec, extradata) {
        Ok(extradata.to_vec())
    } else {
        convert::extradata_to_avcc(codec, extradata, 4).map_err(|e| format!("{e}"))
    }
}

fn pick_source(opened: OpenedSource) -> Option<Arc<dyn ReadAt>> {
    match opened {
        OpenedSource::Single(s) => Some(s),
        OpenedSource::Hls(h) => Some(h.source),
        OpenedSource::DashTracks(list) => {
            let pick = list
                .iter()
                .position(|t| t.content_type == "video")
                .unwrap_or(0);
            let mut list = list;
            Some(list.swap_remove(pick).source)
        }
    }
}

fn open_player(path: &str) -> Result<Player, String> {
    // 本地文件走 FileSource：`me_stream::open_input` 会先建一个 HTTP 客户端
    // （读系统代理配置，实测 0.7 s），本地播放完全用不到 —— 省掉这段就是
    // "点开就播"和"点开等一下"的区别。
    let src: Arc<dyn ReadAt> = if path.contains("://") {
        let client = me_stream::default_client().map_err(|e| format!("{e}"))?;
        let opened = me_stream::open_input(&client, path, &OpenOptions::default())
            .map_err(|e| format!("{e}"))?;
        pick_source(opened).ok_or_else(|| "这条输入没有可播放的字节流".to_string())?
    } else {
        Arc::new(me_core::FileSource::open(path).map_err(|e| format!("{e}"))?)
    };
    Player::open(
        src,
        path,
        PlayerConfig {
            max_buffer_us: PLAYER_MAX_BUFFER_US,
            max_buffer_packets: PLAYER_MAX_BUFFER_PACKETS,
        },
    )
    .map_err(|e| format!("{e}"))
}

/// i32 采样按原始位深缩到 i16。
#[inline]
fn to_i16(v: i32, bits: u8) -> i16 {
    if bits > 16 {
        (v >> (bits - 16)).clamp(-32768, 32767) as i16
    } else if bits < 16 {
        (v << (16 - bits)).clamp(-32768, 32767) as i16
    } else {
        v.clamp(-32768, 32767) as i16
    }
}

/// 任意声道 → 立体声（保持能量，不做响度归一）。
fn downmix_stereo(samples: &[i32], channels: u32, bits: u8, out: &mut Vec<i16>) {
    let ch = channels.max(1) as usize;
    out.clear();
    if samples.is_empty() {
        return;
    }
    let frames = samples.len() / ch;
    out.reserve(frames * 2);
    match ch {
        1 => {
            for &s in samples.iter().take(frames) {
                let v = to_i16(s, bits);
                out.push(v);
                out.push(v);
            }
        }
        2 => {
            for f in 0..frames {
                out.push(to_i16(samples[f * 2], bits));
                out.push(to_i16(samples[f * 2 + 1], bits));
            }
        }
        6 => {
            // 5.1：FL FR FC LFE BL BR
            for f in 0..frames {
                let b = f * 6;
                let l = samples[b] as f32 + 0.707 * samples[b + 2] as f32
                    + 0.707 * samples[b + 4] as f32;
                let r = samples[b + 1] as f32 + 0.707 * samples[b + 2] as f32
                    + 0.707 * samples[b + 5] as f32;
                out.push(to_i16(l as i32, bits));
                out.push(to_i16(r as i32, bits));
            }
        }
        _ => {
            for f in 0..frames {
                let b = f * ch;
                let mut acc = 0i64;
                for c in 0..ch {
                    acc += samples[b + c] as i64;
                }
                let v = to_i16((acc / ch as i64) as i32, bits);
                out.push(v);
                out.push(v);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
// 视频管线
// ─────────────────────────────────────────────────────────────

fn video_loop(path: String, defs: Arc<Vec<TrackDef>>, sh: Arc<Shared>) {
    let mut player: Option<Player> = None;
    let mut dec: Option<(u32, Box<dyn VideoDecoder>)> = None;
    let mut seen_seq = 0u64;
    let mut seen_v_resync = 0u64;
    // seek 落点早于目标时：解码到这里之前只当参考帧，不出队
    let mut display_from: Option<i64> = None;
    let mut eof = false;
    // 寻址 / 起播后先解出至少一帧：暂停状态下拖动进度条也要看到画面，
    // 解到一帧就回到"等播放"的挂起状态，不会偷偷跑完整个片。
    let mut need_prime = true;

    loop {
        let (playing, closed, seq, target, track, v_resync, v_resync_target, preview_seek) = {
            let c = sh.ctrl.lock().unwrap_or_else(|p| p.into_inner());
            (
                c.playing,
                c.closed,
                c.seq,
                c.target_us,
                c.v_track,
                c.v_resync,
                c.v_resync_target,
                c.preview,
            )
        };
        if closed {
            return;
        }

        // ── 视频对齐：只把视频管线跳到当前播放位置（音频不动）──
        if v_resync != seen_v_resync {
            seen_v_resync = v_resync;
            {
                let mut q = sh.vq.lock().unwrap_or_else(|p| p.into_inner());
                q.frames.clear();
                q.leased = None;
            }
            sh.cv_v.notify_all();
            // 解码器要重建：跳过去的第一个包是关键帧，旧的参考帧已经没用了
            dec = None;
            eof = false;
            need_prime = true;
            sh.v_eof.store(false, Ordering::SeqCst);
            if player.is_none() {
                match open_player(&path) {
                    Ok(p) => player = Some(p),
                    Err(e) => {
                        sh.fail(format!("视频管线打不开输入：{e}"));
                        return;
                    }
                }
            }
            if let Some(p) = player.as_mut() {
                // 直接跳到目标位置。落点是最接近的关键帧（可能略早或略晚），
                // 宿主会**强制显示**跳过来的第一帧 —— 不能设 display_from：
                // 那要求把落点到目标之间的帧全解一遍，解码跟不上时反而永远追不上
                // （实测每 2 秒重触发一次、画面一直冻住）。
                if let Err(e) = p.seek_us(v_resync_target) {
                    sh.fail(format!("视频对齐失败：{e}"));
                    return;
                }
            }
            display_from = None;
            continue;
        }

        // ── 寻址 / 切轨：清队列、丢掉解码器（重新建时才拿新的 extradata）──
        if seq != seen_seq {
            {
                let mut q = sh.vq.lock().unwrap_or_else(|p| p.into_inner());
                q.frames.clear();
                q.seq = seq;
            }
            sh.cv_v.notify_all();
            dec = None;
            seen_seq = seq;
            eof = false;
            need_prime = true;
            sh.v_eof.store(false, Ordering::SeqCst);

            if track < 0 {
                sh.v_seek_done.store(seq, Ordering::SeqCst);
                // 没有视频轨：挂到下一次控制变化
                if wait_seq_change(&sh, seq) {
                    continue;
                }
                return;
            }

            if player.is_none() {
                match open_player(&path) {
                    Ok(p) => player = Some(p),
                    Err(e) => {
                        sh.fail(format!("视频管线打不开输入：{e}"));
                        return;
                    }
                }
            }
            let mut landed = target;
            if let Some(p) = player.as_mut() {
                match p.seek_us(target) {
                    Ok(l) => landed = l,
                    Err(e) => {
                        sh.fail(format!("视频寻址失败：{e}"));
                        return;
                    }
                }
            }
            // 落点早于目标（关键帧在目标之前）时，中间这些帧要解码但**不出队**：
            // 它们是后续帧的参考，缺了会花屏；但显示出来就是"拖到 60s 却从 55s 播"。
            // 预览用的 seek 不追帧：追帧要把落点到目标之间的帧全解一遍，
            // 悬停时会超时拿不到图。播放用的 seek 才需要精确到目标。
            display_from = if !preview_seek && landed < target { Some(target) } else { None };
            *sh.seek_landed.lock().unwrap_or_else(|e| e.into_inner()) = (seq, landed);
            sh.v_seek_done.store(seq, Ordering::SeqCst);
            continue;
        }

        if eof {
            // 播完之后等 seek 或关闭，不空转
            need_prime = false;
            if wait_seq_change(&sh, seq) {
                continue;
            }
            return;
        }

        // 不在播放、且已经没有"要出一帧预览"的任务：挂起（暂停时 CPU 归零）
        if !playing && !need_prime {
            if wait_ctrl(&sh, seq) {
                continue;
            }
            return;
        }

        // 队列满：阻塞在条件变量上，让解码与显示同速
        {
            let mut q = sh.vq.lock().unwrap_or_else(|p| p.into_inner());
            while q.frames.len() >= sh.vq_cap.load(Ordering::Relaxed) as usize
                && q.seq == seq
                && !q.closed
            {
                let (g, _) = sh
                    .cv_v
                    .wait_timeout(q, Duration::from_millis(50))
                    .unwrap_or_else(|p| p.into_inner());
                q = g;
            }
            if q.closed || q.seq != seq {
                continue;
            }
        }

        let Some(def) = defs.iter().find(|d| d.id as i32 == track) else {
            sh.v_seek_done.store(seq, Ordering::SeqCst);
            if wait_seq_change(&sh, seq) {
                continue;
            }
            return;
        };

        if player.is_none() {
            match open_player(&path) {
                Ok(p) => player = Some(p),
                Err(e) => {
                    sh.fail(format!("视频管线打不开输入：{e}"));
                    return;
                }
            }
        }
        if dec.is_none() {
            match me_decode::open_video_decoder(def.codec.clone(), &def.config) {
                Ok(d) => {
                    eprintln!("[zuvsrc] 视频解码后端: {} ({})", d.backend(), def.codec.name());
                    dec = Some((def.id, d))
                }
                Err(e) => {
                    sh.fail(format!("视频解码器起不来（{}）：{e}", def.codec.name()));
                    return;
                }
            }
            // 建好解码器后回到循环顶部重新取一次控制状态，避免拿旧 seq 用新解码器
            continue;
        }

        let pkt = match player.as_mut().unwrap().next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => {
                // 解码器里可能还压着 B 帧，先冲干净
                if let Some((_, d)) = dec.as_mut() {
                    match d.flush() {
                        Ok(frames) => {
                            for f in frames {
                                sh.v_decodes.fetch_add(1, Ordering::Relaxed);
                                push_frame(&sh, f, seq);
                            }
                        }
                        Err(e) => {
                            sh.fail(format!("视频冲刷失败：{e}"));
                            return;
                        }
                    }
                }
                eof = true;
                need_prime = false;
                sh.v_eof.store(true, Ordering::SeqCst);
                continue;
            }
            Err(e) => {
                sh.fail(format!("视频解封装失败：{e}"));
                return;
            }
        };
        if pkt.track_id != def.id {
            continue;
        }
        let sample = convert::to_decoder_sample(&def.codec, &pkt.data, 4);
        if sample.is_empty() {
            continue;
        }
        let pts_us = def.timebase.ticks_to_us(pkt.pts);
        let t_dec = Instant::now();
        let decoded = match dec.as_mut().unwrap().1.decode(&sample, pts_us) {
            Ok(f) => f,
            Err(e) => {
                sh.fail(format!("视频解码失败：{e}"));
                return;
            }
        };
        sh.v_decode_ns
            .fetch_add(t_dec.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if !decoded.is_empty() {
            need_prime = false;
        }
        for f in decoded {
            sh.v_decodes.fetch_add(1, Ordering::Relaxed);
            sh.last_pts.store(f.pts_us, Ordering::Relaxed);
            if let Some(from) = display_from {
                // 2ms 余量：容得下时间戳取整，别把正好落在目标上的帧丢掉
                if f.pts_us + 2_000 < from {
                    continue;
                }
                display_from = None;
            }
            if !push_frame(&sh, f, seq) {
                break;
            }
        }
    }
}

fn push_frame(sh: &Arc<Shared>, f: VideoFrame, seq: u64) -> bool {
    let mut q = sh.vq.lock().unwrap_or_else(|p| p.into_inner());
    let start = Instant::now();
    while q.frames.len() >= sh.vq_cap.load(Ordering::Relaxed) as usize {
        if q.closed || q.seq != seq {
            return false;
        }
        if start.elapsed() > Duration::from_secs(2) {
            // 宿主长时间不取帧（比如最小化）：丢掉这帧，别让内存滚雪球
            return true;
        }
        let (g, _) = sh
            .cv_v
            .wait_timeout(q, Duration::from_millis(20))
            .unwrap_or_else(|p| p.into_inner());
        q = g;
    }
    if q.closed || q.seq != seq {
        return false;
    }
    // **按 pts 有序插入**：VideoToolbox 在 B 帧场景下会先吐出后面帧的
    // 回调（实测出现 1.167 s 在 1.133 s 之前），队列一旦乱序，宿主按队首
    // pts 判"到点没有"就会持续误判 —— 表现是整片 70% 的帧被当"太旧"丢掉。
    // 队列只有十几帧，线性插进去的成本可以忽略。
    let pos = q.frames.iter().position(|x| x.pts_us > f.pts_us);
    match pos {
        Some(i) => q.frames.insert(i, f),
        None => q.frames.push_back(f),
    }
    sh.cv_v.notify_all();
    true
}

/// 等一次控制变化（seek / 播放 / 关闭）。返回 true 表示"继续循环"，false 表示退出线程。
fn wait_ctrl(sh: &Arc<Shared>, seq: u64) -> bool {
    let mut c = sh.ctrl.lock().unwrap_or_else(|p| p.into_inner());
    while !c.closed && c.seq == seq && !c.playing {
        let (g, _) = sh
            .cv
            .wait_timeout(c, Duration::from_millis(200))
            .unwrap_or_else(|p| p.into_inner());
        c = g;
    }
    if c.closed {
        return false;
    }
    true
}

/// 只等"寻址 / 切轨 / 关闭"这一类变化（不看播放状态）。
///
/// 播到末尾、或者这条轨根本不存在时用它挂起线程 —— 用 [`wait_ctrl`] 会在
/// "正在播放"时立刻返回，变成 100% CPU 的空转。
fn wait_seq_change(sh: &Arc<Shared>, seq: u64) -> bool {
    let mut c = sh.ctrl.lock().unwrap_or_else(|p| p.into_inner());
    while !c.closed && c.seq == seq {
        let (g, _) = sh
            .cv
            .wait_timeout(c, Duration::from_millis(200))
            .unwrap_or_else(|p| p.into_inner());
        c = g;
    }
    !c.closed
}

// ─────────────────────────────────────────────────────────────
// 音频管线
// ─────────────────────────────────────────────────────────────

fn audio_loop(path: String, defs: Arc<Vec<TrackDef>>, sh: Arc<Shared>) {
    let mut player: Option<Player> = None;
    let mut dec: Option<(u32, Box<dyn AudioDecoder>)> = None;
    let mut seen_seq = 0u64;
    // seek 之后要跳过的音频（目标之前的部分）
    let mut audio_skip_until: Option<i64> = None;
    let mut eof = false;

    loop {
        let (playing, closed, seq, target, track) = {
            let c = sh.ctrl.lock().unwrap_or_else(|p| p.into_inner());
            (c.playing, c.closed, c.seq, c.target_us, c.a_track)
        };
        if closed {
            return;
        }

        if seq != seen_seq {
            {
                let mut q = sh.aq.lock().unwrap_or_else(|p| p.into_inner());
                q.segs.clear();
                q.buffered_us = 0;
                q.seq = seq;
            }
            sh.cv_a.notify_all();
            dec = None;
            seen_seq = seq;
            eof = false;
            sh.a_eof.store(false, Ordering::SeqCst);

            if track < 0 {
                sh.a_seek_done.store(seq, Ordering::SeqCst);
                if wait_seq_change(&sh, seq) {
                    continue;
                }
                return;
            }

            if player.is_none() {
                match open_player(&path) {
                    Ok(p) => player = Some(p),
                    Err(e) => {
                        sh.fail(format!("音频管线打不开输入：{e}"));
                        return;
                    }
                }
            }
            // 有视频时等视频先落定，按同一个位置寻址（否则音频在目标点、
            // 视频在关键帧，一开播就差半个 GOP）
            let mut audio_target = target;
            if sh.has_video.load(Ordering::SeqCst) {
                let wait_until = Instant::now() + Duration::from_millis(500);
                loop {
                    let landed = *sh.seek_landed.lock().unwrap_or_else(|e| e.into_inner());
                    if landed.0 == seq {
                        audio_target = landed.1.max(0);
                        break;
                    }
                    if Instant::now() >= wait_until {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
            }
            let mut landed_audio = audio_target;
            if let Some(p) = player.as_mut() {
                match p.seek_us(audio_target) {
                    Ok(l) => landed_audio = l,
                    Err(e) => {
                        sh.fail(format!("音频寻址失败：{e}"));
                        return;
                    }
                }
            }
            // 和目标差太多就从目标开始出声（差一点无所谓，硬切会有爆音）
            audio_skip_until = if target - landed_audio > 50_000 { Some(target) } else { None };
            sh.a_seek_done.store(seq, Ordering::SeqCst);
            continue;
        }

        if eof {
            if wait_seq_change(&sh, seq) {
                continue;
            }
            return;
        }

        if !playing {
            if wait_ctrl(&sh, seq) {
                continue;
            }
            return;
        }

        // 缓冲够多了就停：内存上限与片长无关
        {
            let mut q = sh.aq.lock().unwrap_or_else(|p| p.into_inner());
            while q.buffered_us >= AUDIO_BUFFER_US && q.seq == seq && !q.closed {
                let (g, _) = sh
                    .cv_a
                    .wait_timeout(q, Duration::from_millis(50))
                    .unwrap_or_else(|p| p.into_inner());
                q = g;
            }
            if q.closed || q.seq != seq {
                continue;
            }
        }

        let Some(def) = defs.iter().find(|d| d.id as i32 == track) else {
            sh.a_seek_done.store(seq, Ordering::SeqCst);
            if wait_seq_change(&sh, seq) {
                continue;
            }
            return;
        };

        if player.is_none() {
            match open_player(&path) {
                Ok(p) => player = Some(p),
                Err(e) => {
                    sh.fail(format!("音频管线打不开输入：{e}"));
                    return;
                }
            }
        }
        if dec.is_none() {
            match me_decode::audio::open_audio_decoder(def.codec.clone(), &def.config) {
                Ok(d) => dec = Some((def.id, d)),
                Err(e) => {
                    sh.fail(format!("音频解码器起不来（{}）：{e}", def.codec.name()));
                    return;
                }
            }
            continue;
        }

        let pkt = match player.as_mut().unwrap().next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => {
                if let Some((_, d)) = dec.as_mut() {
                    match d.flush() {
                        Ok(chunks) => {
                            for c in chunks {
                                push_audio(&sh, &c, seq);
                            }
                        }
                        Err(e) => {
                            sh.fail(format!("音频冲刷失败：{e}"));
                            return;
                        }
                    }
                }
                eof = true;
                sh.a_eof.store(true, Ordering::SeqCst);
                continue;
            }
            Err(e) => {
                sh.fail(format!("音频解封装失败：{e}"));
                return;
            }
        };
        if pkt.track_id != def.id {
            continue;
        }
        let pts_us = def.timebase.ticks_to_us(pkt.pts);
        // 视频落点早于目标时，音频也跟着跳过这段（否则会比画面早半个 GOP 出声）
        if let Some(until) = audio_skip_until {
            let dur = def.timebase.ticks_to_us(pkt.duration as i64);
            if pts_us + dur <= until {
                continue;
            }
            audio_skip_until = None;
        }
        let chunks = match dec.as_mut().unwrap().1.decode(&pkt.data, pts_us) {
            Ok(c) => c,
            Err(e) => {
                sh.fail(format!("音频解码失败：{e}"));
                return;
            }
        };
        for c in chunks {
            sh.a_decodes.fetch_add(1, Ordering::Relaxed);
            if !push_audio(&sh, &c, seq) {
                break;
            }
        }
    }
}

fn push_audio(sh: &Arc<Shared>, chunk: &me_decode::audio::PcmChunk, seq: u64) -> bool {
    let mut stereo: Vec<i16> = Vec::new();
    downmix_stereo(
        &chunk.samples,
        chunk.channels as u32,
        chunk.bits_per_sample,
        &mut stereo,
    );
    if stereo.is_empty() {
        return true;
    }
    let rate = chunk.sample_rate.max(1);
    let channels = 2u32;
    let dur_us = (stereo.len() as i64 / channels as i64) * 1_000_000 / rate as i64;

    let mut q = sh.aq.lock().unwrap_or_else(|p| p.into_inner());
    let start = Instant::now();
    while q.buffered_us + dur_us > AUDIO_BUFFER_US && q.seq == seq && !q.closed {
        if start.elapsed() > Duration::from_secs(2) {
            return true; // 宿主不来取：丢块，别堆内存
        }
        let (g, _) = sh
            .cv_a
            .wait_timeout(q, Duration::from_millis(20))
            .unwrap_or_else(|p| p.into_inner());
        q = g;
    }
    if q.closed || q.seq != seq {
        return false;
    }
    q.buffered_us += dur_us;
    q.rate = rate;
    q.channels = channels;
    q.segs.push_back(AudioSeg {
        samples: stereo,
        pos: 0,
        rate,
        channels,
    });
    sh.cv_a.notify_all();
    true
}

// ─────────────────────────────────────────────────────────────
// 生命周期
// ─────────────────────────────────────────────────────────────

/// 创建一个空会话。
#[no_mangle]
pub extern "C" fn zuvsrc_create(out: *mut *mut ZuvSource) -> c_int {
    if out.is_null() {
        return ZUV_ERR_ARG;
    }
    let src = Box::new(ZuvSource {
        sh: Arc::new(Shared::new()),
        tags: Mutex::new(me_core::MediaTags::default()),
        rotation: AtomicU16::new(0),
        threads: Mutex::new(Vec::new()),
        defs: Mutex::new(Arc::new(Vec::new())),
        duration_us: AtomicI64::new(0),
        path: Mutex::new(String::new()),
    });
    unsafe { *out = Box::into_raw(src) };
    ZUV_OK
}

unsafe fn as_source<'a>(p: *mut ZuvSource) -> Option<&'a ZuvSource> {
    p.as_ref()
}

/// 打开一个输入并启动两条管线。`info` 可以为 NULL。
#[no_mangle]
pub unsafe extern "C" fn zuvsrc_open(
    p: *mut ZuvSource,
    path: *const c_char,
    info: *mut ZuvMediaInfo,
) -> c_int {
    let Some(src) = as_source(p) else {
        return ZUV_ERR_ARG;
    };
    if path.is_null() {
        return ZUV_ERR_ARG;
    }
    let Ok(path) = CStr::from_ptr(path).to_str() else {
        return ZUV_ERR_ARG;
    };
    let path = path.to_string();

    // 先关掉上一个
    close_inner(src);

    let t0 = Instant::now();
    let player = match open_player(&path) {
        Ok(p) => p,
        Err(e) => {
            src.sh.fail(format!("打开失败：{e}"));
            return ZUV_ERR;
        }
    };
    let t_demux = t0.elapsed();
    let media = player.info();
    let tags = player.tags();
    let rotation = player.video_rotation();
    let t_info = t0.elapsed();
    drop(player);
    if std::env::var_os("ZUV_TRACE").is_some() {
        eprintln!(
            "[zuvsrc] {path}: demux={:?} info={:?} total={:?}",
            t_demux,
            t_info - t_demux,
            t_info
        );
    }

    *src.tags.lock().unwrap_or_else(|e| e.into_inner()) = tags;
    src.rotation.store(rotation, Ordering::Relaxed);

    let mut defs: Vec<TrackDef> = Vec::new();
    let mut out_info = ZuvMediaInfo::default();
    fill_str(&mut out_info.container, &media.container);
    out_info.duration_us = media.duration_us.unwrap_or(0);
    out_info.bit_rate = media.bitrate.unwrap_or(0) as i64;

    let mut v_track = TRACK_NONE;
    let mut a_track = TRACK_NONE;
    let mut n = 0usize;

    for t in &media.tracks {
        if n >= 16 {
            break;
        }
        let (kind, config) = match t.kind {
            MediaKind::Video => match video_config(&t.codec, &t.extradata) {
                Ok(c) => (0, Some(c)),
                Err(e) => {
                    // 认得出是视频轨但没有解码器：报出来，别静默成"没有画面"
                    src.sh.fail(format!("视频轨解码参数不可用（{}）：{e}", t.codec.name()));
                    return ZUV_ERR;
                }
            },
            MediaKind::Audio => match t.codec {
                CodecId::Aac | CodecId::Mp3 | CodecId::Flac | CodecId::Opus => {
                    (1, Some(t.extradata.clone()))
                }
                // 引擎没有这个音频编码的解码器（AC-3 / E-AC-3）：照样放画面，
                // 只是没声音 —— 比整个文件打不开好
                _ => (1, None),
            },
            _ => (2, None),
        };

        if let Some(cfg) = config {
            if kind == 0 && v_track == TRACK_NONE {
                v_track = t.id as i32;
            }
            if kind == 1 && a_track == TRACK_NONE {
                a_track = t.id as i32;
            }
            defs.push(TrackDef {
                id: t.id,
                codec: t.codec.clone(),
                config: cfg,
                timebase: t.timebase,
            });
        }

        let mut ti = ZuvTrackInfo {
            track_id: t.id as i32,
            kind,
            width: t.width,
            height: t.height,
            sample_rate: t.sample_rate,
            channels: t.channels as u32,
            ..Default::default()
        };
        fill_str(&mut ti.codec, &t.codec.name());
        if let Some(l) = &t.language {
            fill_str(&mut ti.language, l);
        }
        if let Some(l) = &t.title {
            fill_str(&mut ti.title, l);
        }
        if let Some((num, den)) = t.frame_rate {
            if den > 0 {
                ti.frame_rate = num as f64 / den as f64;
            }
        }
        out_info.tracks[n] = ti;
        n += 1;
    }
    out_info.track_count = n as i32;
    out_info.video_index = media
        .tracks
        .iter()
        .position(|t| t.kind == MediaKind::Video)
        .map(|i| i as i32)
        .unwrap_or(TRACK_NONE);
    out_info.audio_index = media
        .tracks
        .iter()
        .position(|t| t.kind == MediaKind::Audio)
        .map(|i| i as i32)
        .unwrap_or(TRACK_NONE);

    if !info.is_null() {
        *info = out_info;
    }

    *src.path.lock().unwrap_or_else(|e| e.into_inner()) = path.clone();
    src.duration_us.store(out_info.duration_us, Ordering::SeqCst);

    let video_codec = defs
        .iter()
        .find(|d| d.id as i32 == v_track)
        .map(|d| d.codec.clone())
        .or_else(|| defs.first().map(|d| d.codec.clone()));
    src.sh
        .vq_cap
        .store(video_queue_frames(video_codec.as_ref()) as u64, Ordering::Relaxed);

    let has_video = v_track != TRACK_NONE;
    let has_audio = a_track != TRACK_NONE;
    src.sh.has_video.store(has_video, Ordering::SeqCst);
    src.sh.has_audio.store(has_audio, Ordering::SeqCst);

    {
        let mut c = src.sh.ctrl.lock().unwrap_or_else(|e| e.into_inner());
        c.playing = false;
        c.closed = false;
        c.seq = 1;
        c.target_us = 0;
        c.v_track = v_track;
        c.a_track = a_track;
    }
    {
        let mut q = src.sh.vq.lock().unwrap_or_else(|e| e.into_inner());
        q.seq = 1;
        q.closed = false;
    }
    {
        let mut q = src.sh.aq.lock().unwrap_or_else(|e| e.into_inner());
        q.seq = 1;
        q.closed = false;
        q.rate = media
            .tracks
            .iter()
            .find(|t| t.kind == MediaKind::Audio)
            .map(|t| t.sample_rate)
            .unwrap_or(0);
        q.channels = 2;
    }
    src.sh.v_seek_done.store(1, Ordering::SeqCst);
    src.sh.a_seek_done.store(1, Ordering::SeqCst);
    src.sh.v_eof.store(false, Ordering::SeqCst);
    src.sh.a_eof.store(false, Ordering::SeqCst);
    src.sh.state.store(STATE_LOADED, Ordering::SeqCst);
    {
        let mut e = src.sh.err.lock().unwrap_or_else(|e| e.into_inner());
        *e = std::ffi::CString::new("").unwrap();
    }

    let mut handles = src.threads.lock().unwrap_or_else(|e| e.into_inner());
    // 先把上一轮的线程收回来（close_inner 已置 closed）
    for h in handles.drain(..) {
        let _ = h.join();
    }

    let defs = Arc::new(defs);
    *src.defs.lock().unwrap_or_else(|e| e.into_inner()) = defs.clone();

    if has_video {
        let sh = src.sh.clone();
        let p = path.clone();
        let d = defs.clone();
        if let Ok(h) = std::thread::Builder::new()
            .name("zuv-video".into())
            .spawn(move || {
                promote_thread_qos();
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    video_loop(p, d, sh.clone())
                }));
                if r.is_err() {
                    sh.fail("视频管线内部异常（坏码流？）");
                }
            })
        {
            handles.push(h);
        }
    }
    if has_audio {
        let sh = src.sh.clone();
        let p = path.clone();
        let d = defs.clone();
        if let Ok(h) = std::thread::Builder::new()
            .name("zuv-audio".into())
            .spawn(move || {
                promote_thread_qos();
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    audio_loop(p, d, sh.clone())
                }));
                if r.is_err() {
                    sh.fail("音频管线内部异常（坏码流？）");
                }
            })
        {
            handles.push(h);
        }
    }

    ZUV_OK
}

fn close_inner(src: &ZuvSource) {
    {
        let mut c = src.sh.ctrl.lock().unwrap_or_else(|e| e.into_inner());
        c.closed = true;
        c.playing = false;
        c.seq = c.seq.wrapping_add(1);
    }
    {
        let mut q = src.sh.vq.lock().unwrap_or_else(|e| e.into_inner());
        q.closed = true;
        q.frames.clear();
        q.leased = None;
    }
    {
        let mut q = src.sh.aq.lock().unwrap_or_else(|e| e.into_inner());
        q.closed = true;
        q.segs.clear();
        q.buffered_us = 0;
    }
    src.sh.cv.notify_all();
    src.sh.cv_v.notify_all();
    src.sh.cv_a.notify_all();

    let mut handles = src.threads.lock().unwrap_or_else(|e| e.into_inner());
    for h in handles.drain(..) {
        let _ = h.join();
    }
    src.sh.state.store(STATE_IDLE, Ordering::SeqCst);
}

#[no_mangle]
pub unsafe extern "C" fn zuvsrc_close(p: *mut ZuvSource) {
    if let Some(src) = as_source(p) {
        close_inner(src);
    }
}

#[no_mangle]
pub unsafe extern "C" fn zuvsrc_destroy(p: *mut ZuvSource) {
    if p.is_null() {
        return;
    }
    {
        let src = &*p;
        close_inner(src);
    }
    drop(Box::from_raw(p));
}

// ─────────────────────────────────────────────────────────────
// 播放控制
// ─────────────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn zuvsrc_play(p: *mut ZuvSource) -> c_int {
    let Some(src) = as_source(p) else {
        return ZUV_ERR_ARG;
    };
    {
        let mut c = src.sh.ctrl.lock().unwrap_or_else(|e| e.into_inner());
        if c.closed {
            return ZUV_ERR_STATE;
        }
        c.playing = true;
    }
    src.sh.cv.notify_all();
    src.sh.state.store(STATE_PLAYING, Ordering::SeqCst);
    ZUV_OK
}

#[no_mangle]
pub unsafe extern "C" fn zuvsrc_pause(p: *mut ZuvSource) -> c_int {
    let Some(src) = as_source(p) else {
        return ZUV_ERR_ARG;
    };
    {
        let mut c = src.sh.ctrl.lock().unwrap_or_else(|e| e.into_inner());
        c.playing = false;
    }
    src.sh.cv.notify_all();
    src.sh.state.store(STATE_PAUSED, Ordering::SeqCst);
    ZUV_OK
}

/// 预览用的寻址：落点在哪就从哪出帧，**不追到目标时间**。
///
/// 悬停缩略图每移动一次就请求一次，追帧要把半个 GOP 解一遍（实测 1-3 秒），
/// 交互上等不起；预览允许误差在半个关键帧间隔内。
#[no_mangle]
pub unsafe extern "C" fn zuvsrc_seek_preview(p: *mut ZuvSource, pts_us: i64) -> c_int {
    seek_impl(p, pts_us, true)
}

/// 非阻塞寻址：立即返回，管线各自落位。用 [`zuvsrc_seek_done`] 判断是否落稳。
#[no_mangle]
pub unsafe extern "C" fn zuvsrc_seek(p: *mut ZuvSource, pts_us: i64) -> c_int {
    seek_impl(p, pts_us, false)
}

unsafe fn seek_impl(p: *mut ZuvSource, pts_us: i64, preview: bool) -> c_int {
    let Some(src) = as_source(p) else {
        return ZUV_ERR_ARG;
    };
    let seq = {
        let mut c = src.sh.ctrl.lock().unwrap_or_else(|e| e.into_inner());
        if c.closed {
            return ZUV_ERR_STATE;
        }
        c.seq = c.seq.wrapping_add(1);
        c.target_us = pts_us.max(0);
        // 预览标志只对这一次 seek 生效
        c.preview = preview;
        c.seq
    };
    {
        let mut q = src.sh.vq.lock().unwrap_or_else(|e| e.into_inner());
        q.seq = seq;
        q.frames.clear();
    }
    {
        let mut q = src.sh.aq.lock().unwrap_or_else(|e| e.into_inner());
        q.seq = seq;
        q.segs.clear();
        q.buffered_us = 0;
    }
    src.sh.v_seek_done.store(seq.wrapping_sub(1), Ordering::SeqCst);
    src.sh.a_seek_done.store(seq.wrapping_sub(1), Ordering::SeqCst);
    src.sh.v_eof.store(false, Ordering::SeqCst);
    src.sh.a_eof.store(false, Ordering::SeqCst);
    src.sh.cv.notify_all();
    src.sh.cv_v.notify_all();
    src.sh.cv_a.notify_all();
    ZUV_OK
}

/// 本次寻址视频管线的实际落点（关键帧位置）；没有视频轨或尚未落定时返回 -1。
#[no_mangle]
pub unsafe extern "C" fn zuvsrc_seek_landed_us(p: *mut ZuvSource) -> i64 {
    let Some(src) = as_source(p) else {
        return -1;
    };
    let seq = {
        let c = src.sh.ctrl.lock().unwrap_or_else(|e| e.into_inner());
        c.seq
    };
    if !src.sh.has_video.load(Ordering::SeqCst) {
        let c = src.sh.ctrl.lock().unwrap_or_else(|e| e.into_inner());
        return c.target_us;
    }
    let landed = *src.sh.seek_landed.lock().unwrap_or_else(|e| e.into_inner());
    if landed.0 == seq {
        landed.1
    } else {
        -1
    }
}

#[no_mangle]
pub unsafe extern "C" fn zuvsrc_seek_done(p: *mut ZuvSource) -> c_int {
    let Some(src) = as_source(p) else {
        return 1;
    };
    let seq = {
        let c = src.sh.ctrl.lock().unwrap_or_else(|e| e.into_inner());
        c.seq
    };
    let v = src.sh.v_seek_done.load(Ordering::SeqCst);
    let a = src.sh.a_seek_done.load(Ordering::SeqCst);
    ((v >= seq) && (a >= seq)) as c_int
}

/// 切轨（`kind` 0=视频 1=音频），并从 `resume_us` 继续。
#[no_mangle]
pub unsafe extern "C" fn zuvsrc_select_track(
    p: *mut ZuvSource,
    kind: c_int,
    track_id: c_int,
    resume_us: i64,
) -> c_int {
    let Some(src) = as_source(p) else {
        return ZUV_ERR_ARG;
    };
    let seq = {
        let mut c = src.sh.ctrl.lock().unwrap_or_else(|e| e.into_inner());
        if c.closed {
            return ZUV_ERR_STATE;
        }
        if kind == 0 {
            c.v_track = track_id;
        } else if kind == 1 {
            c.a_track = track_id;
        } else {
            return ZUV_ERR_ARG;
        }
        c.seq = c.seq.wrapping_add(1);
        c.target_us = resume_us.max(0);
        c.seq
    };
    {
        let mut q = src.sh.vq.lock().unwrap_or_else(|e| e.into_inner());
        q.seq = seq;
        q.frames.clear();
    }
    {
        let mut q = src.sh.aq.lock().unwrap_or_else(|e| e.into_inner());
        q.seq = seq;
        q.segs.clear();
        q.buffered_us = 0;
    }
    src.sh.v_eof.store(false, Ordering::SeqCst);
    src.sh.a_eof.store(false, Ordering::SeqCst);
    src.sh.cv.notify_all();
    src.sh.cv_v.notify_all();
    src.sh.cv_a.notify_all();
    ZUV_OK
}

#[no_mangle]
pub unsafe extern "C" fn zuvsrc_state(p: *mut ZuvSource) -> c_int {
    match as_source(p) {
        Some(s) => s.sh.state.load(Ordering::SeqCst) as c_int,
        None => STATE_IDLE as c_int,
    }
}

#[no_mangle]
pub unsafe extern "C" fn zuvsrc_duration_us(p: *mut ZuvSource) -> i64 {
    match as_source(p) {
        Some(s) => s.duration_us.load(Ordering::SeqCst),
        None => 0,
    }
}

// ─────────────────────────────────────────────────────────────
// 字幕
// ─────────────────────────────────────────────────────────────

/// 解析一条字幕轨（异步）。`track_id` 来自 [`zuvsrc_open`] 填的 track 表，
/// 传 -1 表示关闭字幕。
///
/// 为什么整条轨一次读完：文本字幕（SRT/ASS）一部片子通常几十 KB 到几 MB，
/// 一次扫完换来"按时间查询是一次二分"，比边播边取简单得多，也不会在
/// 播放线程上产生额外抖动。
/// 把**视频管线**跳到 `pts_us`（音频继续播）。
///
/// 用于"解码跟不上、画面越拖越远"时的一次性对齐：丢帧解决不了问题，
/// 因为解码器仍在原地解；只有真的 seek 才能把积压丢掉。
#[no_mangle]
pub unsafe extern "C" fn zuvsrc_resync_video(p: *mut ZuvSource, pts_us: i64) -> c_int {
    let Some(src) = as_source(p) else {
        return ZUV_ERR_ARG;
    };
    {
        let mut c = src.sh.ctrl.lock().unwrap_or_else(|e| e.into_inner());
        if c.closed {
            return ZUV_ERR_STATE;
        }
        c.v_resync += 1;
        c.v_resync_target = pts_us.max(0);
    }
    src.sh.cv.notify_all();
    src.sh.cv_v.notify_all();
    ZUV_OK
}

/// 视频的显示旋转（0 / 90 / 180 / 270）。
///
/// 手机竖屏视频把画面横着编码，靠 `tkhd` 矩阵声明该转多少度；宿主按这个值
/// 旋转画面并交换宽高，否则竖屏片子会横着显示。
#[no_mangle]
pub unsafe extern "C" fn zuvsrc_video_rotation(p: *mut ZuvSource) -> c_int {
    let Some(src) = as_source(p) else {
        return 0;
    };
    src.rotation.load(Ordering::Relaxed) as c_int
}

/// 读容器标签（标题 / 艺术家 / 专辑），写进三个 UTF-8 缓冲。
///
/// 缓冲容量为 0 或指针为空时跳过那一项；返回 1 表示至少有一项非空。
#[no_mangle]
pub unsafe extern "C" fn zuvsrc_metadata(
    p: *mut ZuvSource,
    title: *mut c_char,
    title_cap: c_int,
    artist: *mut c_char,
    artist_cap: c_int,
    album: *mut c_char,
    album_cap: c_int,
) -> c_int {
    let Some(src) = as_source(p) else {
        return 0;
    };
    let tags = src.tags.lock().unwrap_or_else(|e| e.into_inner()).clone();

    let mut any = false;
    {
        let mut put = |value: &Option<String>, buf: *mut c_char, cap: c_int| {
            if buf.is_null() || cap <= 0 {
                return;
            }
            let bytes = value.as_deref().unwrap_or("").as_bytes();
            let n = bytes.len().min(cap as usize - 1);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, n);
            *buf.add(n) = 0;
            if n > 0 {
                any = true;
            }
        };
        put(&tags.title, title, title_cap);
        put(&tags.artist, artist, artist_cap);
        put(&tags.album, album, album_cap);
    }
    any as c_int
}

#[no_mangle]
pub unsafe extern "C" fn zuvsrc_load_subtitles(p: *mut ZuvSource, track_id: i32) -> c_int {
    let Some(src) = as_source(p) else {
        return ZUV_ERR_ARG;
    };
    let generation = {
        let mut s = src.sh.subs.lock().unwrap_or_else(|e| e.into_inner());
        s.generation += 1;
        s.cues.clear();
        s.err.clear();
        if track_id < 0 {
            s.state = SUB_NONE;
            s.track = -1;
            return ZUV_OK;
        }
        s.state = SUB_LOADING;
        s.track = track_id;
        s.generation
    };

    let path = src.path.lock().map(|g| g.clone()).unwrap_or_default();
    if path.is_empty() {
        let mut s = src.sh.subs.lock().unwrap_or_else(|e| e.into_inner());
        s.state = SUB_ERROR;
        s.err = "会话还没有打开文件".into();
        return ZUV_ERR_STATE;
    }

    let sh = src.sh.clone();
    std::thread::spawn(move || load_subtitle_track(path, track_id, generation, sh));
    ZUV_OK
}

fn load_subtitle_track(path: String, track_id: i32, generation: u64, sh: Arc<Shared>) {
    promote_thread_qos();
    let result = (|| -> Result<Vec<subtitle::Cue>, String> {
        let mut player = open_player(&path)?;
        let info = player.info();
        let track = info
            .tracks
            .iter()
            .find(|t| t.id as i32 == track_id)
            .ok_or_else(|| format!("没有 id={track_id} 的字幕轨"))?;
        let tb = track.timebase;
        let codec = track.codec.clone();
        let mut cues: Vec<subtitle::Cue> = Vec::new();
        while let Some(pkt) = player.next_packet().map_err(|e| format!("{e}"))? {
            if pkt.track_id as i32 != track_id {
                continue;
            }
            let Some(text) = subtitle::packet_text(&codec, &pkt.data) else {
                continue;
            };
            let start = tb.ticks_to_us(pkt.pts);
            let mut dur = tb.ticks_to_us(pkt.duration as i64);
            if dur <= 0 {
                // 没有 BlockDuration 时按 2 秒兜底，免得整条字幕永不消失
                dur = 2_000_000;
            }
            cues.push(subtitle::Cue { start_us: start, end_us: start + dur, text });
        }
        cues.sort_by_key(|c| c.start_us);
        Ok(cues)
    })();

    let mut s = sh.subs.lock().unwrap_or_else(|e| e.into_inner());
    if s.generation != generation {
        return; // 已经换轨或关闭，丢弃这次结果
    }
    match result {
        Ok(cues) => {
            s.cues = cues;
            s.state = SUB_READY;
        }
        Err(e) => {
            s.err = e;
            s.state = SUB_ERROR;
        }
    }
}

/// 直接加载一个外部字幕文件（`.srt`）。宿主在视频旁边找到同名字幕时用它。
#[no_mangle]
pub unsafe extern "C" fn zuvsrc_load_subtitle_file(
    p: *mut ZuvSource,
    path: *const c_char,
) -> c_int {
    let Some(src) = as_source(p) else {
        return ZUV_ERR_ARG;
    };
    if path.is_null() {
        return ZUV_ERR_ARG;
    }
    let Ok(path) = CStr::from_ptr(path).to_str() else {
        return ZUV_ERR_ARG;
    };
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            let mut s = src.sh.subs.lock().unwrap_or_else(|e| e.into_inner());
            s.state = SUB_ERROR;
            s.err = format!("读不了字幕文件：{e}");
            return ZUV_ERR;
        }
    };
    let text = String::from_utf8_lossy(&data);
    let cues = subtitle::parse_srt(&text);
    let mut s = src.sh.subs.lock().unwrap_or_else(|e| e.into_inner());
    s.generation += 1;
    s.track = -2; // -2 = 外部字幕文件
    s.err.clear();
    s.cues = cues;
    s.state = SUB_READY;
    ZUV_OK
}

/// 直接加载一段字幕**文本**（UTF-8，宿主已经做好编码转换）。
///
/// 为什么让宿主读文件：字幕文件的编码五花八门（GBK / Big5 / UTF-16），
/// 引擎里只认 UTF-8，而宿主那边有系统的编码转换能力，判断起来更省事。
#[no_mangle]
pub unsafe extern "C" fn zuvsrc_load_subtitle_text(
    p: *mut ZuvSource,
    text: *const c_char,
    len: i32,
) -> c_int {
    let Some(src) = as_source(p) else {
        return ZUV_ERR_ARG;
    };
    if text.is_null() || len <= 0 {
        return ZUV_ERR_ARG;
    }
    let bytes = std::slice::from_raw_parts(text as *const u8, len as usize);
    let body = String::from_utf8_lossy(bytes);
    let cues = subtitle::parse_text(&body);
    if cues.is_empty() {
        let mut st = src.sh.subs.lock().unwrap_or_else(|e| e.into_inner());
        st.state = SUB_ERROR;
        st.err = "这段文本里没有解析出字幕（格式不认识？）".into();
        return ZUV_ERR;
    }
    let mut st = src.sh.subs.lock().unwrap_or_else(|e| e.into_inner());
    st.generation += 1;
    st.track = -2; // -2 = 外部字幕文件
    st.err.clear();
    st.cues = cues;
    st.state = SUB_READY;
    ZUV_OK
}

/// 字幕状态：0 未启用、1 解析中、2 就绪、3 失败。
#[no_mangle]
pub unsafe extern "C" fn zuvsrc_subtitle_state(p: *mut ZuvSource) -> c_int {
    let Some(src) = as_source(p) else {
        return SUB_NONE as c_int;
    };
    let s = src.sh.subs.lock().unwrap_or_else(|e| e.into_inner());
    s.state as c_int
}

/// 解析失败时的原因（成功或还没开始时返回空串）。
#[no_mangle]
pub unsafe extern "C" fn zuvsrc_subtitle_error(p: *mut ZuvSource) -> *const c_char {
    let Some(src) = as_source(p) else {
        return std::ptr::null();
    };
    let s = src.sh.subs.lock().unwrap_or_else(|e| e.into_inner());
    s.err.as_ptr() as *const c_char
}

/// 取 `pts_us` 时刻要显示的字幕文本，写进 `buf`（UTF-8，不带结尾 0 之外的多余字符）。
///
/// 返回写入的字节数；0 表示这一刻没有字幕（或还没解析完）。
#[no_mangle]
pub unsafe extern "C" fn zuvsrc_subtitle_text(
    p: *mut ZuvSource,
    pts_us: i64,
    buf: *mut c_char,
    cap: c_int,
) -> c_int {
    let Some(src) = as_source(p) else {
        return 0;
    };
    if buf.is_null() || cap <= 1 {
        return 0;
    }
    let s = src.sh.subs.lock().unwrap_or_else(|e| e.into_inner());
    if s.state != SUB_READY || s.cues.is_empty() {
        return 0;
    }
    // 找到 start <= pts 的最后一条
    let idx = match s.cues.binary_search_by(|c| c.start_us.cmp(&pts_us)) {
        Ok(i) => i,
        Err(0) => return 0,
        Err(i) => i - 1,
    };
    let cue = &s.cues[idx];
    if pts_us >= cue.end_us {
        return 0;
    }
    let bytes = cue.text.as_bytes();
    let n = bytes.len().min(cap as usize - 1);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, n);
    *buf.add(n) = 0;
    n as c_int
}

#[no_mangle]
pub unsafe extern "C" fn zuvsrc_last_error(p: *mut ZuvSource) -> *const c_char {
    match as_source(p) {
        Some(s) => {
            let e = s.sh.err.lock().unwrap_or_else(|p| p.into_inner());
            // CString 的缓冲挂在会话上，指针在下一次 fail 之前一直有效
            e.as_ptr()
        }
        None => std::ptr::null(),
    }
}

// ─────────────────────────────────────────────────────────────
// 出帧 / 收 PCM
// ─────────────────────────────────────────────────────────────

/// 取一帧 NV12。返回 1 = 有帧，0 = 超时，<0 = 错误。
///
/// 成功时 `out.data` 指向会话内部缓冲，直到 [`zuvsrc_release_video`] 前一直有效。
#[no_mangle]
pub unsafe extern "C" fn zuvsrc_acquire_video(
    p: *mut ZuvSource,
    out: *mut ZuvFrame,
    timeout_ms: c_int,
) -> c_int {
    let Some(src) = as_source(p) else {
        return ZUV_ERR_ARG;
    };
    if out.is_null() {
        return ZUV_ERR_ARG;
    }
    let deadline = Instant::now() + Duration::from_millis(timeout_ms.max(0) as u64);
    let mut q = src.sh.vq.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if q.leased.is_none() {
            if let Some(f) = q.frames.pop_front() {
                q.leased = Some(f);
                src.sh.cv_v.notify_all();
                break;
            }
        }
        if Instant::now() >= deadline || q.closed {
            return 0;
        }
        let (g, _) = src
            .sh
            .cv_v
            .wait_timeout(q, Duration::from_millis(10))
            .unwrap_or_else(|p| p.into_inner());
        q = g;
    }
    let f = q.leased.as_ref().unwrap();
    *out = ZuvFrame {
        data: f.nv12.as_ptr(),
        bytes: f.nv12.len() as u32,
        width: f.width,
        height: f.height,
        pts_us: f.pts_us,
    };
    1
}

#[no_mangle]
pub unsafe extern "C" fn zuvsrc_release_video(p: *mut ZuvSource) {
    if let Some(src) = as_source(p) {
        let mut q = src.sh.vq.lock().unwrap_or_else(|e| e.into_inner());
        q.leased = None;
        src.sh.cv_v.notify_all();
    }
}

/// 本机是否支持该编码的**平台硬解**。`codec` 取 `"h264"` / `"h265"` / `"av1"`。
///
/// 宿主用它决定"硬件解码"开关是否点得动。不要按编码名猜：AV1 在 Intel Mac
/// 上根本没有硬解，猜出来就是一个说谎的开关。
#[no_mangle]
pub unsafe extern "C" fn zuvsrc_hwaccel_supported(codec: *const c_char) -> c_int {
    if codec.is_null() {
        return 0;
    }
    let Ok(name) = CStr::from_ptr(codec).to_str() else {
        return 0;
    };
    let id = match name {
        "h264" => CodecId::H264,
        "h265" | "hevc" => CodecId::H265,
        "av1" => CodecId::Av1,
        _ => return 0,
    };
    me_decode::hardware_decode_supported(id) as c_int
}

/// 丢掉当前租借的那一帧（宿主决定"追最新帧"时用）。
#[no_mangle]
pub unsafe extern "C" fn zuvsrc_drop_leased_video(p: *mut ZuvSource) {
    if let Some(src) = as_source(p) {
        let mut q = src.sh.vq.lock().unwrap_or_else(|e| e.into_inner());
        q.leased = None;
        src.sh.cv_v.notify_all();
    }
}

/// 队首帧的显示时间（不取走）。1 = 有，0 = 队列空。
///
/// 宿主用它做"该不该现在上屏"的判断：先看 pts，再决定取不取 —— 这样丢帧
/// 逻辑不需要同时持有两帧。
#[no_mangle]
pub unsafe extern "C" fn zuvsrc_video_next_pts(p: *mut ZuvSource, pts_us: *mut i64) -> c_int {
    let Some(src) = as_source(p) else {
        return ZUV_ERR_ARG;
    };
    if pts_us.is_null() {
        return ZUV_ERR_ARG;
    }
    let q = src.sh.vq.lock().unwrap_or_else(|e| e.into_inner());
    match q.frames.front() {
        Some(f) => {
            *pts_us = f.pts_us;
            1
        }
        None => 0,
    }
}

/// 读立体声 S16 PCM。返回写入的**帧数**（每帧 2 个采样），0 表示暂时没有。
#[no_mangle]
pub unsafe extern "C" fn zuvsrc_read_audio(
    p: *mut ZuvSource,
    dst: *mut i16,
    max_frames: c_int,
    rate: *mut c_int,
    channels: *mut c_int,
) -> c_int {
    let Some(src) = as_source(p) else {
        return ZUV_ERR_ARG;
    };
    if dst.is_null() || max_frames <= 0 {
        return ZUV_ERR_ARG;
    }
    let want = max_frames as usize;
    let mut written = 0usize;

    let mut q = src.sh.aq.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(r) = rate.as_mut() {
        *r = q.rate as c_int;
    }
    if let Some(c) = channels.as_mut() {
        *c = q.channels as c_int;
    }
    while written < want {
        let Some(seg) = q.segs.front_mut() else {
            break;
        };
        let ch = seg.channels.max(1) as usize;
        let total_frames = seg.samples.len() / ch;
        let consumed_frames = seg.pos / ch;
        let remain = total_frames.saturating_sub(consumed_frames);
        if remain == 0 {
            let seg_frames = total_frames;
            let dur_us = seg_frames as i64 * 1_000_000 / seg.rate.max(1) as i64;
            q.buffered_us = (q.buffered_us - dur_us).max(0);
            q.segs.pop_front();
            continue;
        }
        let take = remain.min(want - written);
        let from = seg.pos;
        let to = from + take * ch;
        let dst_slice = std::slice::from_raw_parts_mut(dst.add(written * ch), take * ch);
        dst_slice.copy_from_slice(&seg.samples[from..to]);
        seg.pos = to;
        written += take;
        if seg.pos >= seg.samples.len() {
            let dur_us = total_frames as i64 * 1_000_000 / seg.rate.max(1) as i64;
            q.buffered_us = (q.buffered_us - dur_us).max(0);
            q.segs.pop_front();
        }
    }
    src.sh.cv_a.notify_all();
    written as c_int
}

#[no_mangle]
pub unsafe extern "C" fn zuvsrc_audio_buffered_ms(p: *mut ZuvSource) -> i64 {
    match as_source(p) {
        Some(s) => {
            let q = s.sh.aq.lock().unwrap_or_else(|e| e.into_inner());
            q.buffered_us / 1000
        }
        None => 0,
    }
}

/// 两条管线都读到末尾（宿主据此判定播放结束）。
#[no_mangle]
pub unsafe extern "C" fn zuvsrc_at_end(p: *mut ZuvSource) -> c_int {
    match as_source(p) {
        Some(s) => {
            let v = !s.sh.has_video.load(Ordering::SeqCst) || s.sh.v_eof.load(Ordering::SeqCst);
            let a = !s.sh.has_audio.load(Ordering::SeqCst) || s.sh.a_eof.load(Ordering::SeqCst);
            (v && a) as c_int
        }
        None => 1,
    }
}

/// 诊断计数：`[视频解码帧数, 音频解码块数, 视频队列深度, 音频缓冲毫秒]`。
#[no_mangle]
pub unsafe extern "C" fn zuvsrc_stats(p: *mut ZuvSource, out: *mut u64) -> c_int {
    let Some(src) = as_source(p) else {
        return ZUV_ERR_ARG;
    };
    if out.is_null() {
        return ZUV_ERR_ARG;
    }
    let vq = src.sh.vq.lock().unwrap_or_else(|e| e.into_inner()).frames.len() as u64;
    let aq = src.sh.aq.lock().unwrap_or_else(|e| e.into_inner()).buffered_us.max(0) as u64 / 1000;
    unsafe {
        *out.add(0) = src.sh.v_decodes.load(Ordering::Relaxed);
        *out.add(1) = src.sh.a_decodes.load(Ordering::Relaxed);
        *out.add(2) = vq;
        *out.add(3) = aq;
        *out.add(4) = src.sh.v_decode_ns.load(Ordering::Relaxed);
        *out.add(5) = 0;
    }
    ZUV_OK
}
