//! 字节源接入：把"文件 / 还在增长的下载文件 / HTTP / HLS / DASH"统一成 [`ReadAt`]。
//!
//! ## 与下载引擎的分工（重要边界）
//!
//! **BT 下载由下载引擎（XferRust）负责，媒体引擎只负责播放。**
//! 所以这里**没有** BT 会话（不连 peer、不 announce tracker、不跑 DHT），
//! 只有 [`GrowingFileSource`]：读一个**另一个进程正在写入、还没写完的文件**，
//! 请求的区间还没落盘就阻塞等待。宿主把播放头位置转达给下载引擎，
//! 由下载引擎决定"先下哪一片" —— 两个引擎之间只有"一个文件 + 一条播放头提示"，
//! 没有共享内存、没有链接、没有私有协议，各自可以独立升级。

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use me_core::{MediaError, ReadAt, Result};

use me_net::http::{HttpClient, HttpConfig};
use me_net::{dash, hls};

// ---------------------------------------------------------------------------
// 取消标志
// ---------------------------------------------------------------------------

/// 取消标志：宿主按 Ctrl-C / 点"停止"时置位，阻塞中的读立刻返回。
#[derive(Clone, Default)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        // 信号是**进程级**的，所以任何标志都要看它：宿主按 Ctrl-C 时，
        // 阻塞中的读必须立刻返回，而不是等自己的 30 秒超时
        self.0.load(Ordering::SeqCst) || SIGNAL_FLAG.load(Ordering::SeqCst)
    }

    /// 被取消就返回 `Cancelled`（退出码 6）。
    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(MediaError::Cancelled)
        } else {
            Ok(())
        }
    }

    /// 安装 SIGINT / SIGTERM 处理器（宿主 Ctrl-C 时让引擎干净退出）。
    ///
    /// 直接用 `signal(2)` 而不是引入第三方 crate：本工程的硬约束是不加依赖，
    /// 而信号处理器里只需要"置一个原子位"这一个动作 —— 那正是信号安全的。
    pub fn install() -> Self {
        unsafe extern "C" {
            fn signal(sig: i32, handler: unsafe extern "C" fn(i32)) -> usize;
        }
        unsafe extern "C" fn on_signal(_sig: i32) {
            // 信号处理器里不能分配内存、不能加锁：只写一个原子量
            SIGNAL_FLAG.store(true, Ordering::SeqCst);
        }
        const SIGINT: i32 = 2;
        const SIGTERM: i32 = 15;
        unsafe {
            signal(SIGINT, on_signal);
            signal(SIGTERM, on_signal);
        }
        Self::new()
    }
}

/// 信号处理器专用的全局标志（异步信号安全：只读写原子量）。
static SIGNAL_FLAG: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// 增长文件
// ---------------------------------------------------------------------------

/// [`GrowingFileSource`] 的行为参数。
#[derive(Debug, Clone)]
pub struct GrowingOptions {
    /// 轮询间隔：等数据时多久看一次文件长度。
    pub poll_interval: Duration,
    /// 一次等待的上限；**只要文件还在长大就不算超时**（每次长度变化就重置计时）。
    pub wait_timeout: Duration,
    /// "文件不再增长"的判定时长：请求的位置已在末尾之后、且文件这么久没变大
    /// 就认为**到末尾了**（返回 `Ok(0)`）。
    ///
    /// 为什么需要它：读一个**已经下完**的文件（最常见的播放场景）时，
    /// 请求越过末尾之后必须立刻给出"读完了"，否则每次播完都要白等一整个
    /// `wait_timeout`，用户看到的是"播完了还在转圈"。1.5 秒与宿主侧
    /// Provider 的判据一致，也足够长（避免把"下载恰好停顿了一下"误判成读完）。
    pub idle_timeout: Duration,
    /// **可用性图**文件路径（宿主写的"哪些字节已经下到了"，见 [`crate::avail`]）。
    ///
    /// `None` = 不知道（本地文件、HTTP 下载……），照老样子读：没下到的位置读出来
    /// 是零，由上层去判断。`Some(path)` 且图是新鲜的时，请求到"还没下到"的区间
    /// 会**等**（与等文件长大同一套超时语义），而不是把零字节当数据交出去 ——
    /// 那是 BT 边下边播里"播到空洞就报码流损坏"的根因。
    pub availability: Option<PathBuf>,
}

impl Default for GrowingOptions {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_millis(50),
            wait_timeout: Duration::from_secs(30),
            // 实测下来的取舍：太短了会把"下载卡顿一下"当成读完（音频会提前静音），
            // 太长了读完一个文件还是要等很久。1.5 秒是两边都站得住的点。
            idle_timeout: Duration::from_millis(1500),
            availability: None,
        }
    }
}

/// "另一个进程正在写"的文件源（边下边播）。
///
/// 语义：
/// - `read_at` 请求的区间还没落盘 → **阻塞等待**，直到有数据、超时或被取消；
/// - 超时返回 [`MediaError::NeedMore`]（这是"暂时没有"，宿主可以重试），
///   **不是** `Ok(0)`（那是"真的到末尾了"，会让上层以为文件就这么长）；
/// - `len()` 返回**最终长度**（如果下载引擎告诉了的话）而不是当前已下载的长度 ——
///   否则容器层会以为文件只有这么长，MP4 的 `moov` 在末尾时直接读不到。
pub struct GrowingFileSource {
    path: PathBuf,
    opts: GrowingOptions,
    /// 下载引擎给出的最终大小（未知为 None）。
    expected_len: AtomicU64,
    /// 是否已知最终大小（`AtomicU64` 没有 Option，用这个布尔区分 0 与"未知"）。
    has_expected: AtomicBool,
    /// 播放头位置（字节）。宿主读它、转达给下载引擎。
    playhead: AtomicU64,
    /// **缓存的读句柄**。
    ///
    /// 为什么必须缓存：一部两小时的片子有二十多万个采样，而每个采样都要读一次 ——
    /// 每次都 `File::open` 的话，光是 open/close 的系统调用开销就能到几百秒
    /// （实测 1 MB 的短片就已经因此慢了 2.2 秒）。句柄一直拿着还有个好处：
    /// 文件在长大时同一个 fd 上继续 pread 能读到新写入的数据（我们本来就要这样）。
    handle: Mutex<Option<std::fs::File>>,
    /// "已经读到过末尾"的结论缓存（值 = 当时的文件长度）。
    ///
    /// 判定一次 EOF 要等一个 `idle_timeout`（默认 1.5 秒）。不缓存的话，
    /// 容器层每次往末尾探一下都要重新等一遍 —— 实测一次播放会多等好几轮。
    /// 缓存之后只在**文件又长大了**时失效（下载重新开始的情形）。
    settled_len: AtomicU64,
    /// 已经判定过 EOF（配合 `settled_len`）。
    settled: AtomicBool,
    cancel: CancelFlag,
    /// 诊断：一共等了多久、超时过几次。
    waited_ms: AtomicU64,
    timeouts: AtomicU64,
    /// 因为"这段还没下到"而等过的次数（诊断：边下边播卡不卡就看它）。
    avail_waits: AtomicU64,
}

impl GrowingFileSource {
    pub fn new(path: impl AsRef<Path>, opts: GrowingOptions) -> Self {
        Self::with_cancel(path, opts, CancelFlag::new())
    }

    /// 用宿主/CLI 的取消标志构造（Ctrl-C 能立刻打断等待）。
    pub fn with_cancel(path: impl AsRef<Path>, opts: GrowingOptions, cancel: CancelFlag) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            opts,
            expected_len: AtomicU64::new(0),
            has_expected: AtomicBool::new(false),
            playhead: AtomicU64::new(0),
            handle: Mutex::new(None),
            settled_len: AtomicU64::new(0),
            settled: AtomicBool::new(false),
            cancel,
            waited_ms: AtomicU64::new(0),
            timeouts: AtomicU64::new(0),
            avail_waits: AtomicU64::new(0),
        }
    }

    /// 告诉它文件的**最终**大小（下载引擎从种子元信息里知道）。
    pub fn set_expected_len(&self, len: Option<u64>) {
        match len {
            Some(l) => {
                self.expected_len.store(l, Ordering::SeqCst);
                self.has_expected.store(true, Ordering::SeqCst);
            }
            None => self.has_expected.store(false, Ordering::SeqCst),
        }
    }

    /// 取消标志（宿主/CLI 可以拿它去打断等待）。
    pub fn cancel_flag(&self) -> CancelFlag {
        self.cancel.clone()
    }

    pub fn playhead(&self) -> u64 {
        self.playhead.load(Ordering::SeqCst)
    }

    pub fn set_playhead(&self, offset: u64) {
        self.playhead.store(offset, Ordering::SeqCst);
    }

    /// 当前磁盘上已经有多少字节（诊断用）。
    pub fn downloaded_bytes(&self) -> u64 {
        std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0)
    }

    /// 等待过多久（毫秒）与超时次数（诊断用）。
    pub fn wait_stats(&self) -> (u64, u64) {
        (self.waited_ms.load(Ordering::SeqCst), self.timeouts.load(Ordering::SeqCst))
    }

    /// 因为"这段还没下到"等过几次（诊断：边下边播卡不卡就看它）。
    pub fn availability_waits(&self) -> u64 {
        self.avail_waits.load(Ordering::SeqCst)
    }

    /// 现在磁盘上的长度（文件不存在算 0）。
    fn current_len(&self) -> u64 {
        std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0)
    }

    /// 读一次（用缓存句柄，见 [`read_once_impl`]）。
    fn read_once(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        read_once_impl(&self.path, &self.handle, offset, buf)
    }
}

impl ReadAt for GrowingFileSource {
    fn len(&self) -> Option<u64> {
        // 最终长度优先：容器层要用它判断"文件有多长"，
        // 拿"已下载的长度"会让 moov 在末尾的文件永远读不到
        if self.has_expected.load(Ordering::SeqCst) {
            return Some(self.expected_len.load(Ordering::SeqCst));
        }
        let cur = self.current_len();
        if cur == 0 {
            // 文件还没出现：长度**未知**而不是 0（0 会让上层以为"空文件"直接放弃）
            None
        } else {
            Some(cur)
        }
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        self.playhead.store(offset, Ordering::SeqCst);
        // 最终大小已知，且请求的位置在它之后 → **真的到末尾了**。
        //
        // 这个判断不是优化，是正确性：文件读完之后不去等，否则每次播完一个
        // 下好的文件都要白等一整个 `wait_timeout`（默认 30 秒），用户看到的是
        // "播完了界面还在转圈"。
        if self.has_expected.load(Ordering::SeqCst)
            && offset >= self.expected_len.load(Ordering::SeqCst)
        {
            return Ok(0);
        }
        // 已经判定过"读到末尾"：只要文件没再长大就直接回答，不再重新等一轮。
        // （判定一次要一个 `idle_timeout`，而容器层会反复往末尾探。）
        if self.settled.load(Ordering::SeqCst) {
            let settled_len = self.settled_len.load(Ordering::SeqCst);
            if offset >= settled_len && self.current_len() <= settled_len {
                return Ok(0);
            }
            // 文件又长大了（下载重新开始）：之前的结论作废
            self.settled.store(false, Ordering::SeqCst);
        }
        // ── 空洞等待 ────────────────────────────────────────────────
        //
        // 宿主给了可用性图、且图是新鲜的时：请求的区间**还没下到**就在这里等，
        // 不要读出一片零字节交上去（上层会把它当真实数据 → H.264 找不到起始码
        // → "码流损坏" → 播放终止）。等待期间反复重读图（宿主每秒刷一次），
        // 一到位就继续；超时语义与"等文件长大"一致。
        if let Some(map_path) = self.opts.availability.clone() {
            if let Some(avail) = crate::avail::AvailFile::load(&map_path) {
                if avail.is_fresh() && !avail.avail.covers(offset, offset + buf.len() as u64) {
                    self.avail_waits.fetch_add(1, Ordering::SeqCst);
                    tracing::info!(
                        offset,
                        len = buf.len(),
                        "这段还没下到，等下载引擎把它补上（可用性图）"
                    );
                    let wait_started = Instant::now();
                    loop {
                        self.cancel.check()?;
                        if wait_started.elapsed() >= self.opts.wait_timeout {
                            self.timeouts.fetch_add(1, Ordering::SeqCst);
                            self.waited_ms.fetch_add(
                                wait_started.elapsed().as_millis() as u64,
                                Ordering::SeqCst,
                            );
                            return Err(MediaError::NeedMore { need: buf.len(), have: 0 });
                        }
                        std::thread::sleep(self.opts.poll_interval);
                        match crate::avail::AvailFile::load(&map_path) {
                            // 图变旧了（宿主不再刷新）：不再拦着 —— 退回老行为，
                            // 否则播放会被一张过期的图永久卡住。
                            Some(a) if !a.is_fresh() => break,
                            Some(a) if a.avail.covers(offset, offset + buf.len() as u64) => break,
                            // 图坏了/还是没到：继续等
                            _ => continue,
                        }
                    }
                    self.waited_ms
                        .fetch_add(wait_started.elapsed().as_millis() as u64, Ordering::SeqCst);
                }
            }
        }

        // ── 快路径：本地完整文件 ──────────────────────────────────────
        //
        // 没有可用性图（**本地文件、不在边下边播**）时先直接读一次。`read_once`
        // 只是一次 `pread(2)`，命中就读到了；而下面那段"等文件长大"的逻辑每一步
        // 都要 `stat(2)` 才敢决定要不要读 —— 采样里 `stat$INODE64` 单独占 **5.7%**
        // （1080p 一帧要从容器取几十次采样，每次都白付一次 stat）。
        //
        // ⚠️ 为什么必须"没有可用性图"才走这条路：下载器可能**预分配**出一个
        // 稀疏文件，读到还没下到的空洞时 `pread` 会成功返回一片**零字节**（不是
        // 短读）。上层拿到零字节会当成真实数据 → H.264 找不到起始码 → 报"码流
        // 损坏"。有可用性图时那个检查必须挡在读取**之前**，所以这里绕开它。
        //
        // ⚠️ `settled` 之后也不走：那说明文件已经被判定读完了，快路径会一次次
        // 去 pread 一个已知已经没有数据的位置。
        if self.opts.availability.is_none() && !self.settled.load(Ordering::SeqCst) {
            if let Ok(n) = self.read_once(offset, buf) {
                if n > 0 {
                    return Ok(n);
                }
            }
        }

        let started = Instant::now();
        let mut last_len = self.current_len();
        let mut last_change = Instant::now();
        loop {
            self.cancel.check()?;
            let cur = self.current_len();
            if cur != last_len {
                last_len = cur;
                // 文件还在长大 → 重置超时计时：下载慢不该被误判成失败
                last_change = Instant::now();
            }
            if offset < cur {
                let n = self.read_once(offset, buf)?;
                if n > 0 {
                    self.waited_ms.fetch_add(started.elapsed().as_millis() as u64, Ordering::SeqCst);
                    return Ok(n);
                }
            }
            // 请求的位置已经在文件末尾之后，且文件**不再增长** → 到末尾了。
            //
            // 判据与宿主侧的 Provider 一致（"连续 stat 大小不变超过一段时间"）：
            // 下载完成、被暂停、失败，三种情况在这里都该给出"读完了"这个答案，
            // 而不是让播放器一直等一个永远不会来的字节。
            // 只有"文件在长 / 刚长过"时才继续等 —— 那才是真正的"边下边播"。
            if offset >= cur && last_change.elapsed() >= self.opts.idle_timeout {
                self.settled_len.store(cur, Ordering::SeqCst);
                self.settled.store(true, Ordering::SeqCst);
                return Ok(0);
            }
            if last_change.elapsed() >= self.opts.wait_timeout {
                self.timeouts.fetch_add(1, Ordering::SeqCst);
                self.waited_ms.fetch_add(started.elapsed().as_millis() as u64, Ordering::SeqCst);
                return Err(MediaError::NeedMore { need: buf.len(), have: 0 });
            }
            std::thread::sleep(self.opts.poll_interval);
        }
    }
}

/// 用缓存句柄读一次。
///
/// 句柄读不到数据时**重开一次**再试：文件被"写临时文件 + rename"的方式替换过时，
/// 旧 fd 指向的是已经被删除的 inode（读出来永远是旧内容或 0）。
/// 这条兜底让"下载引擎用 rename 落盘"的实现也能正常播。
fn read_once_impl(path: &Path, handle: &Mutex<Option<std::fs::File>>, offset: u64, buf: &mut [u8]) -> Result<usize> {
    let mut guard = handle.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = std::fs::File::open(path).ok();
    }
    if let Some(f) = guard.as_ref() {
        if let Ok(n) = pread_at(f, offset, buf) {
            if n > 0 {
                return Ok(n);
            }
        }
    }
    *guard = std::fs::File::open(path).ok();
    match guard.as_ref() {
        Some(f) => Ok(pread_at(f, offset, buf)?),
        None => Ok(0),
    }
}

/// 平台无关的 `pread`（不改变文件偏移，可在多线程共用同一个句柄）。
fn pread_at(file: &std::fs::File, offset: u64, buf: &mut [u8]) -> std::io::Result<usize> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        file.seek_read(buf, offset)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, offset, buf);
        Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "当前平台缺少 pread 实现"))
    }
}

// ---------------------------------------------------------------------------
// 输入分类与统一入口
// ---------------------------------------------------------------------------

/// 输入的种类（按 URL/扩展名判断）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    File,
    Http,
    Hls,
    Dash,
    Magnet,
}

impl InputKind {
    pub fn name(self) -> &'static str {
        match self {
            InputKind::File => "file",
            InputKind::Http => "http",
            InputKind::Hls => "hls",
            InputKind::Dash => "dash",
            InputKind::Magnet => "magnet",
        }
    }
}

/// 按协议/扩展名判断输入种类。
pub fn classify(input: &str) -> InputKind {
    let lower = input.to_ascii_lowercase();
    if lower.starts_with("magnet:") {
        return InputKind::Magnet;
    }
    if lower.starts_with("http://") || lower.starts_with("https://") {
        let path = lower.split(['?', '#']).next().unwrap_or("");
        if path.ends_with(".m3u8") || path.ends_with(".m3u") {
            return InputKind::Hls;
        }
        if path.ends_with(".mpd") {
            return InputKind::Dash;
        }
        return InputKind::Http;
    }
    InputKind::File
}

/// 打开输入时的选项。
///
/// 手写 `Debug` 而不是 derive：`CancelFlag` 里是 `Arc<AtomicBool>`，
/// derive 出来的 Debug 会打出一串无意义的指针地址，日志里没用还占地方。
#[derive(Clone)]
pub struct OpenOptions {
    /// 边下边播：本地文件用 [`GrowingFileSource`]（另一个进程可能还在写它）。
    pub grow: bool,
    /// 下载引擎给出的最终大小（边下边播时用得上）。
    pub expected_len: Option<u64>,
    /// 选流上限。
    pub max_height: Option<u32>,
    pub max_bandwidth: Option<u64>,
    pub grow_opts: GrowingOptions,
    pub cancel: CancelFlag,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            grow: false,
            expected_len: None,
            max_height: None,
            max_bandwidth: None,
            grow_opts: GrowingOptions::default(),
            cancel: CancelFlag::new(),
        }
    }
}

impl std::fmt::Debug for OpenOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenOptions")
            .field("grow", &self.grow)
            .field("expected_len", &self.expected_len)
            .field("max_height", &self.max_height)
            .field("max_bandwidth", &self.max_bandwidth)
            .field("grow_opts", &self.grow_opts)
            .field("cancelled", &self.cancel.is_cancelled())
            .finish()
    }
}

/// HLS 点播/直播源。
///
/// **直播是"快照"语义**：拿到清单那一刻的分片列表被拼成一条流，
/// 不做时移、不做清单轮询刷新（宿主若要追直播，重新 `open_hls` 即可）。
/// 这样刻意保持"一次打开 = 一条不可变的字节流"，上层的缓冲/寻址逻辑才简单。
pub struct HlsSource {
    pub source: Arc<dyn ReadAt>,
    pub duration_us: Option<i64>,
    pub is_live: bool,
    pub segment_count: usize,
    pub variant_bandwidth: Option<u64>,
    pub variant_resolution: Option<(u32, u32)>,
    /// 直播快照对应的媒体序号（宿主用它判断"这次打开的是哪一段"）。
    pub media_sequence: u64,
}

/// DASH 的一条轨道（视频与音频是两条独立的字节流）。
pub struct DashTrack {
    pub source: Arc<dyn ReadAt>,
    pub content_type: String,
    pub codec: Option<String>,
    pub mime: Option<String>,
    pub representation_id: String,
    pub duration_us: Option<i64>,
    pub segment_count: usize,
}

/// 打开结果。
pub enum OpenedSource {
    /// 单条字节流（本地文件 / 直链 HTTP）。
    Single(Arc<dyn ReadAt>),
    /// HLS（已经把分片拼成一条流）。
    Hls(HlsSource),
    /// DASH：按轨开，视频与音频各一条。
    DashTracks(Vec<DashTrack>),
}

impl std::fmt::Debug for OpenedSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenedSource::Single(s) => write!(f, "OpenedSource::Single(len={:?})", s.len()),
            OpenedSource::Hls(h) => write!(
                f,
                "OpenedSource::Hls(segments={}, live={}, duration={:?})",
                h.segment_count, h.is_live, h.duration_us
            ),
            OpenedSource::DashTracks(t) => write!(f, "OpenedSource::DashTracks({} 条轨)", t.len()),
        }
    }
}

/// 打开一个输入。
pub fn open_input(client: &HttpClient, input: &str, opts: &OpenOptions) -> Result<OpenedSource> {
    match classify(input) {
        InputKind::Magnet => Err(MediaError::Unsupported(
            "磁力下载由下载引擎（XferRust）负责，媒体引擎只播放已下载/正在下载的文件；\
             请传下载引擎写出的文件路径"
                .into(),
        )),
        InputKind::File => {
            if opts.grow {
                let g = GrowingFileSource::with_cancel(input, opts.grow_opts.clone(), opts.cancel.clone());
                g.set_expected_len(opts.expected_len);
                Ok(OpenedSource::Single(Arc::new(g)))
            } else {
                Ok(OpenedSource::Single(Arc::new(me_core::FileSource::open(input)?)))
            }
        }
        InputKind::Http => {
            // 直链：先问长度，能 Range 就惰性读
            let src = client.get_source(input)?;
            Ok(OpenedSource::Single(src))
        }
        InputKind::Hls => Ok(OpenedSource::Hls(open_hls(client, input, opts)?)),
        InputKind::Dash => Ok(OpenedSource::DashTracks(open_dash_tracks(client, input, opts)?)),
    }
}

/// 打开 HLS：主清单 → 选流 → 媒体清单 → 把分片拼成一条字节流。
pub fn open_hls(client: &HttpClient, url: &str, opts: &OpenOptions) -> Result<HlsSource> {
    let text = fetch_text(client, url)?;
    let base = url.to_string();
    let playlist = hls::parse(&text, &base)?;
    let (media_url, variant) = match playlist {
        hls::Playlist::Media(_) => (url.to_string(), None),
        hls::Playlist::Master(m) => {
            let v = hls::pick_variant(&m, opts.max_height, opts.max_bandwidth).ok_or_else(|| {
                MediaError::Protocol(format!("主清单里没有可用的变体：{url}"))
            })?;
            (v.uri.clone(), Some(v.clone()))
        }
    };
    let media_text = fetch_text(client, &media_url)?;
    let media = match hls::parse(&media_text, &media_url)? {
        hls::Playlist::Media(m) => m,
        hls::Playlist::Master(_) => {
            return Err(MediaError::Protocol(format!(
                "按主清单取到的还是主清单（层级没到底）：{media_url}"
            )))
        }
    };

    let mut chain = me_core::ChainedSource::new();
    // init 段（fMP4 的 HLS 有它）
    if let Some(map) = &media.init_map {
        let seg = build_segment_source(client, &map.uri, map.byte_range.as_ref(), None)?;
        let len = seg.len().unwrap_or(0);
        chain.push(0, len, seg);
    }
    for seg in &media.segments {
        let src = build_segment_source(client, &seg.uri, seg.byte_range.as_ref(), seg.key.as_ref())?;
        let len = src.len().unwrap_or(0);
        chain.push(chain.len().unwrap_or(0), len, src);
    }
    let duration_us = if media.end_list {
        Some(media.segments.iter().map(|s| s.duration_us).sum())
    } else {
        None
    };
    Ok(HlsSource {
        source: Arc::new(chain),
        duration_us,
        is_live: media.is_live,
        segment_count: media.segments.len(),
        variant_bandwidth: variant.as_ref().map(|v| v.bandwidth),
        variant_resolution: variant.as_ref().and_then(|v| v.resolution),
        media_sequence: media.media_sequence,
    })
}

/// 打开 DASH：选一条视频轨与一条音频轨（各自一条字节流）。
pub fn open_dash_tracks(client: &HttpClient, url: &str, opts: &OpenOptions) -> Result<Vec<DashTrack>> {
    let text = fetch_text(client, url)?;
    let mpd = dash::parse_mpd(&text, url)?;
    let period = mpd
        .first_period()
        .ok_or_else(|| MediaError::Protocol(format!("MPD 里没有 Period：{url}")))?;
    let mut out = Vec::new();
    for content_type in ["video", "audio"] {
        let Some(rep) = dash::pick_representation(
            &period.adaptations,
            content_type,
            opts.max_height,
            opts.max_bandwidth,
        ) else {
            continue;
        };
        let template = rep.segment_template.as_ref().or(period
            .adaptations
            .iter()
            .find(|a| a.representations.iter().any(|r| r.id == rep.id))
            .and_then(|a| a.segment_template.as_ref()));
        let Some(template) = template else {
            return Err(MediaError::Unsupported(
                "这条 MPD 用的是 SegmentList/单文件，暂不支持（只有 SegmentTemplate 能展开）".into(),
            ));
        };
        let timescale = template.timescale.max(1);
        let period_ticks = mpd.period_duration_ticks(period, timescale);
        let mut chain = me_core::ChainedSource::new();
        // 模板展开出来的是**相对 URL**，必须按 MPD 的地址解析成绝对地址 ——
        // 直接把相对地址丢给 HTTP 客户端会连请求都发不出去（解析失败被吞掉），
        // 表现为"拼出来的流是 0 字节"
        if let Some(init) = template.initialization_url(&rep.id, rep.bandwidth) {
            let init = me_net::url::resolve(url, &init);
            let src = client.get_source(&init)?;
            let len = src.len().unwrap_or(0);
            chain.push(0, len, src);
        }
        let segs = template.media_urls(&rep.id, rep.bandwidth, period_ticks);
        for (u, _start, _dur) in &segs {
            let abs = me_net::url::resolve(url, u);
            let src = client.get_source(&abs)?;
            let len = src.len().unwrap_or(0);
            chain.push(chain.len().unwrap_or(0), len, src);
        }
        out.push(DashTrack {
            source: Arc::new(chain),
            content_type: content_type.to_string(),
            codec: rep.codecs.clone(),
            mime: rep.mime.clone(),
            representation_id: rep.id.clone(),
            duration_us: mpd.duration_us,
            segment_count: segs.len(),
        });
    }
    if out.is_empty() {
        return Err(MediaError::Protocol(format!("MPD 里没有可用的视频/音频轨：{url}")));
    }
    Ok(out)
}

/// 一个 HLS 分片的字节源。
///
/// 三种情况分别处理，取舍都是刻意的：
/// - **明文**：用 [`HttpSource`] 惰性读（Range 请求），seek 到大后面不会白拉一堆；
/// - **带字节范围**：裁成逻辑区间（[`RangeSource`]）；
/// - **加密（AES-128）**：**整段拉取、解密后驻内存**。为什么不能惰性：
///   CBC 是按 16 字节块链式解密的，取中间一段必须先有它前面所有块的密文；
///   要惰性就得"按块对齐地拉 + 从上一个关键块解起"，复杂且容易解错 ——
///   而 HLS 分片通常只有几百 KB，整段放内存是划算的取舍。
fn build_segment_source(
    client: &HttpClient,
    uri: &str,
    range: Option<&hls::ByteRange>,
    key: Option<&hls::KeyInfo>,
) -> Result<Arc<dyn ReadAt>> {
    if let Some(k) = key {
        if k.method != hls::KeyMethod::None {
            let raw = match range {
                Some(r) => {
                    let resp = client.get_range(uri, r.start, r.end)?;
                    if !resp.partial {
                        // 服务器无视 Range 回了全量：自己按偏移切，别把整段当分片
                        let s = r.start as usize;
                        let e = match r.end {
                            Some(e) => (e as usize + 1).min(resp.data.len()),
                            None => resp.data.len(),
                        };
                        resp.data.get(s..e).map(|d| d.to_vec()).ok_or_else(|| {
                            MediaError::Protocol(format!("字节范围 {r:?} 超出实际响应长度：{uri}"))
                        })?
                    } else {
                        resp.data
                    }
                }
                None => client.get(uri)?,
            };
            let plain = hls::decrypt(k, client, raw)?;
            return Ok(Arc::new(me_core::Cursor::new(plain)));
        }
    }
    match range {
        Some(r) => {
            let base = client.get_source(uri)?;
            let len = match r.end {
                Some(e) => e.saturating_sub(r.start) + 1,
                None => base.len().unwrap_or(0).saturating_sub(r.start),
            };
            Ok(Arc::new(RangeSource::new(base, r.start, len)))
        }
        None => client.get_source(uri),
    }
}

fn fetch_text(client: &HttpClient, url: &str) -> Result<String> {
    let bytes = client.get(url)?;
    String::from_utf8(bytes).map_err(|_| MediaError::Protocol(format!("清单不是 UTF-8 文本：{url}")))
}

/// 把一个字节源裁成 `[base, base+len)` 的逻辑区间（HLS 的 `#EXT-X-BYTERANGE` 用它）。
pub struct RangeSource {
    inner: Arc<dyn ReadAt>,
    base: u64,
    len: u64,
}

impl RangeSource {
    pub fn new(inner: Arc<dyn ReadAt>, base: u64, len: u64) -> Self {
        Self { inner, base, len }
    }
}

impl ReadAt for RangeSource {
    fn len(&self) -> Option<u64> {
        Some(self.len)
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        if offset >= self.len {
            return Ok(0);
        }
        let want = buf.len().min((self.len - offset) as usize);
        self.inner.read_at(self.base + offset, &mut buf[..want])
    }
}

/// 默认的 HTTP 客户端（宿主没特殊要求时用它）。
pub fn default_client() -> Result<HttpClient> {
    HttpClient::new(HttpConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use me_core::Cursor;
    use std::io::{Seek, Write};
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn classify_covers_all_input_kinds() {
        assert_eq!(classify("magnet:?xt=urn:btih:abc"), InputKind::Magnet);
        assert_eq!(classify("https://a.com/b.m3u8"), InputKind::Hls);
        assert_eq!(classify("https://a.com/b.m3u8?token=1"), InputKind::Hls);
        assert_eq!(classify("https://a.com/b.mpd"), InputKind::Dash);
        assert_eq!(classify("https://a.com/b.ts"), InputKind::Http);
        assert_eq!(classify("/tmp/a.mp4"), InputKind::File);
        assert_eq!(classify("a.m3u8"), InputKind::File, "本地清单走文件路径（不猜协议）");
    }

    #[test]
    fn magnet_is_rejected_with_a_clear_reason() {
        let client = default_client().unwrap();
        let e = open_input(&client, "magnet:?xt=urn:btih:abc", &OpenOptions::default()).unwrap_err();
        assert!(matches!(e, MediaError::Unsupported(_)));
        let msg = e.to_string();
        assert!(msg.contains("下载引擎"), "文案必须点明该由谁下载：{msg}");
    }

    #[test]
    fn range_source_clips_reads() {
        let inner: Arc<dyn ReadAt> = Arc::new(Cursor::new((0u8..100).collect()));
        let r = RangeSource::new(inner, 10, 20);
        assert_eq!(r.len(), Some(20));
        let mut buf = [0u8; 8];
        assert_eq!(r.read_at(0, &mut buf).unwrap(), 8);
        assert_eq!(buf, [10, 11, 12, 13, 14, 15, 16, 17]);
        // 越界读必须返回 0（不是错误、也不是把范围外的数据给出去）
        assert_eq!(r.read_at(20, &mut buf).unwrap(), 0);
        assert_eq!(r.read_at(15, &mut buf).unwrap(), 5);
    }

    fn temp_path(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("me-stream-{tag}-{}-{}.bin", std::process::id(), rand_suffix()));
        p
    }

    fn rand_suffix() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos() as u64
    }

    #[test]
    fn growing_source_reads_data_that_arrives_later() {
        let path = temp_path("grow");
        let writer_path = path.clone();
        // 后台线程：先写 10 字节，隔一会儿再写 10 字节
        let t = std::thread::spawn(move || {
            let mut f = std::fs::File::create(&writer_path).unwrap();
            f.write_all(&[1u8; 10]).unwrap();
            f.flush().unwrap();
            std::thread::sleep(Duration::from_millis(150));
            f.write_all(&[2u8; 10]).unwrap();
            f.flush().unwrap();
        });

        let src = GrowingFileSource::new(&path, GrowingOptions {
            poll_interval: Duration::from_millis(10),
            wait_timeout: Duration::from_secs(5),
            ..Default::default()
        });
        // 先读已有的 10 字节
        let mut buf = [0u8; 10];
        assert_eq!(src.read_at(0, &mut buf).unwrap(), 10);
        assert_eq!(buf, [1u8; 10]);
        // 再读还没写的部分：必须**阻塞等到**它出现
        let mut buf2 = [0u8; 10];
        let n = src.read_at(10, &mut buf2).unwrap();
        assert_eq!(n, 10, "应该等到后写入的数据");
        assert_eq!(buf2, [2u8; 10]);
        assert_eq!(src.playhead(), 10, "读的位置要能被上报给下载引擎");
        t.join().unwrap();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn growing_source_times_out_with_need_more_not_zero() {
        let path = temp_path("timeout");
        std::fs::write(&path, [0u8; 4]).unwrap();
        let src = GrowingFileSource::new(&path, GrowingOptions {
            poll_interval: Duration::from_millis(5),
            wait_timeout: Duration::from_millis(60),
            ..Default::default()
        });
        let mut buf = [0u8; 16];
        let e = src.read_at(4, &mut buf).unwrap_err();
        // 关键：必须是 NeedMore（"暂时没有"，可重试），不能是 Ok(0)（"到末尾了"）
        assert!(matches!(e, MediaError::NeedMore { .. }), "实际：{e}");
        assert_eq!(src.wait_stats().1, 1, "超时次数要能被诊断");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn growing_source_reports_eof_once_the_file_stops_growing() {
        // **最常见的播放场景**：文件已经下完。读到末尾之后必须立刻给出"读完了"，
        // 而不是等到 wait_timeout（默认 30 秒）—— 否则每次播完都要白等半分钟，
        // 用户看到"播完了界面还在转圈"。
        let path = temp_path("eof");
        std::fs::write(&path, [0u8; 100]).unwrap();
        let src = GrowingFileSource::new(&path, GrowingOptions {
            poll_interval: Duration::from_millis(10),
            wait_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_millis(120),
            ..Default::default()
        });
        let started = std::time::Instant::now();
        let mut buf = [0u8; 16];
        // 已知最终大小：越界读应当**立刻**返回 0
        src.set_expected_len(Some(100));
        assert_eq!(src.read_at(100, &mut buf).unwrap(), 0);
        assert!(started.elapsed() < Duration::from_millis(50), "已知最终大小时不该等待");

        // 不知道最终大小时：等 idle_timeout 后也应当给出 0（下载停了/已下完）
        let src2 = GrowingFileSource::new(&path, GrowingOptions {
            poll_interval: Duration::from_millis(10),
            wait_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_millis(120),
            ..Default::default()
        });
        let started2 = std::time::Instant::now();
        assert_eq!(src2.read_at(100, &mut buf).unwrap(), 0);
        let waited = started2.elapsed();
        assert!(
            waited >= Duration::from_millis(100) && waited < Duration::from_secs(2),
            "未知大小时应当按 idle_timeout 判定读完（实际等了 {waited:?}）"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn growing_source_still_waits_while_the_file_is_growing() {
        // 与上一条相对：文件**正在长**的时候必须老实等，不能因为"当前没有"就判读完
        let path = temp_path("still-grow");
        std::fs::write(&path, [0u8; 10]).unwrap();
        let writer_path = path.clone();
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(80));
            let mut f = std::fs::OpenOptions::new().append(true).open(&writer_path).unwrap();
            f.write_all(&[7u8; 30]).unwrap();
            f.flush().unwrap();
            std::thread::sleep(Duration::from_millis(80));
            let mut f = std::fs::OpenOptions::new().append(true).open(&writer_path).unwrap();
            f.write_all(&[8u8; 30]).unwrap();
            f.flush().unwrap();
        });
        let src = GrowingFileSource::new(&path, GrowingOptions {
            poll_interval: Duration::from_millis(10),
            wait_timeout: Duration::from_secs(5),
            idle_timeout: Duration::from_millis(1500),
            ..Default::default()
        });
        let mut buf = [0u8; 30];
        let n = src.read_at(10, &mut buf).unwrap();
        assert_eq!(n, 30, "应当等到后写入的数据");
        assert_eq!(buf, [7u8; 30]);
        t.join().unwrap();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn growing_source_len_prefers_expected_length() {
        let path = temp_path("len");
        std::fs::write(&path, [0u8; 100]).unwrap();
        let src = GrowingFileSource::new(&path, GrowingOptions::default());
        assert_eq!(src.len(), Some(100), "没有最终长度时用当前长度");
        src.set_expected_len(Some(1_000_000));
        assert_eq!(src.len(), Some(1_000_000), "有最终长度时优先用它（moov 在末尾的文件靠它）");
        src.set_expected_len(None);
        assert_eq!(src.len(), Some(100));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn growing_source_len_is_unknown_before_the_file_appears() {
        let path = temp_path("missing");
        let src = GrowingFileSource::new(&path, GrowingOptions::default());
        assert_eq!(src.len(), None, "文件还不存在时是'未知'而不是 0（0 会让上层直接放弃）");
    }

    #[test]
    fn growing_source_cancel_breaks_the_wait() {
        let path = temp_path("cancel");
        std::fs::write(&path, [0u8; 4]).unwrap();
        let src = Arc::new(GrowingFileSource::new(&path, GrowingOptions {
            poll_interval: Duration::from_millis(5),
            wait_timeout: Duration::from_secs(30),
            ..Default::default()
        }));
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let s = src.clone();
        let t = std::thread::spawn(move || {
            let mut buf = [0u8; 16];
            let r = s.read_at(100, &mut buf);
            c.store(1, Ordering::SeqCst);
            r
        });
        // 等它进入等待，再取消
        std::thread::sleep(Duration::from_millis(80));
        assert_eq!(counter.load(Ordering::SeqCst), 0, "这时应该还在等");
        src.cancel_flag().cancel();
        let r = t.join().unwrap();
        assert!(matches!(r, Err(MediaError::Cancelled)), "取消必须立刻结束阻塞");
        let _ = std::fs::remove_file(&path);
    }

    /// 空洞等待：宿主给的可用性图说"这段还没下到"时**等**它下下来，
    /// 而不是把零字节当数据交出去（那会被码流层判成损坏，播放直接终止）。
    #[test]
    fn growing_source_waits_for_the_availability_map() {
        let path = temp_path("avail-wait");
        // 4 片、每片 1 KiB：中间那片此刻还是零（下载引擎先扩文件后填数据）
        let mut data = vec![7u8; 4096];
        data[2048..3072].fill(0); // 片 2 还是零（下载引擎先扩文件后填数据）
        std::fs::write(&path, &data).unwrap();
        let map = temp_path("avail-wait-map.json");
        // 位图 1101_0000 → 片 0/1/3 已有、片 2（文件内 [1024,2048)）没有
        std::fs::write(
            &map,
            r#"{"fileOffset":0,"pieceLength":1024,"total":4096,"bitfield":"d0"}"#,
        )
        .unwrap();
        let src = GrowingFileSource::new(
            &path,
            GrowingOptions {
                poll_interval: Duration::from_millis(10),
                wait_timeout: Duration::from_secs(5),
                availability: Some(map.clone()),
                ..Default::default()
            },
        );
        // 300ms 后"下载完成"：真数据落到盘上 + 图更新
        let writer_path = path.clone();
        let map_path = map.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            let mut f = std::fs::OpenOptions::new().write(true).open(&writer_path).unwrap();
            f.seek(std::io::SeekFrom::Start(2048)).unwrap();
            f.write_all(&[9u8; 1024]).unwrap();
            f.flush().unwrap();
            std::fs::write(
                &map_path,
                r#"{"fileOffset":0,"pieceLength":1024,"total":4096,"bitfield":"f0"}"#,
            )
            .unwrap();
        });

        let mut buf = [0u8; 16];
        let n = src.read_at(2048, &mut buf).unwrap();
        assert_eq!(n, 16);
        assert_eq!(buf, [9u8; 16], "等到的必须是补上来的真实数据，不是零");
        assert!(src.availability_waits() >= 1, "要记下「为空洞等过」这件事（诊断要用）");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&map);
    }

    /// 图里说"没有"但一直没有补上：超时返回 NeedMore（可重试），
    /// **不能**返回零字节让上层当数据用。
    #[test]
    fn availability_wait_times_out_as_need_more() {
        let path = temp_path("avail-timeout");
        std::fs::write(&path, vec![0u8; 4096]).unwrap();
        let map = temp_path("avail-timeout-map.json");
        std::fs::write(
            &map,
            r#"{"fileOffset":0,"pieceLength":1024,"total":4096,"bitfield":"00"}"#,
        )
        .unwrap();
        let src = GrowingFileSource::new(
            &path,
            GrowingOptions {
                poll_interval: Duration::from_millis(10),
                wait_timeout: Duration::from_millis(200),
                availability: Some(map.clone()),
                ..Default::default()
            },
        );
        let mut buf = [0u8; 16];
        let e = src.read_at(0, &mut buf).unwrap_err();
        assert!(matches!(e, MediaError::NeedMore { .. }), "实际：{e}");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&map);
    }

    /// 没有可用性图（普通本地文件、HTTP 下载）时行为与从前一致：照读。
    #[test]
    fn no_availability_map_keeps_old_behaviour() {
        let path = temp_path("no-avail");
        std::fs::write(&path, vec![3u8; 64]).unwrap();
        let src = GrowingFileSource::new(
            &path,
            GrowingOptions {
                poll_interval: Duration::from_millis(10),
                wait_timeout: Duration::from_secs(1),
                ..Default::default()
            },
        );
        let mut buf = [0u8; 8];
        assert_eq!(src.read_at(0, &mut buf).unwrap(), 8);
        assert_eq!(buf, [3u8; 8]);
        assert_eq!(src.availability_waits(), 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn growing_source_does_not_burn_cpu_while_waiting() {
        // 只验证"轮询间隔被遵守"：60ms 的等待里不应该有几百次 stat
        let path = temp_path("cpu");
        std::fs::write(&path, [0u8; 4]).unwrap();
        let src = GrowingFileSource::new(&path, GrowingOptions {
            poll_interval: Duration::from_millis(20),
            wait_timeout: Duration::from_millis(70),
            ..Default::default()
        });
        let mut buf = [0u8; 8];
        let _ = src.read_at(4, &mut buf);
        let (waited, _) = src.wait_stats();
        assert!(waited >= 50, "确实等了一会儿（实际 {waited}ms）");
        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // HLS / DASH：起一个真 HTTP 服务器，走完整链路
    // -----------------------------------------------------------------------

    use crate::testutil::{Route, TestServer};

    fn read_all(src: &Arc<dyn ReadAt>) -> Vec<u8> {
        let mut out = Vec::new();
        let mut off = 0u64;
        let mut buf = [0u8; 4096];
        loop {
            let n = src.read_at(off, &mut buf).unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&buf[..n]);
            off += n as u64;
        }
        out
    }

    #[test]
    fn hls_vod_concatenates_segments_in_order() {
        let s1 = vec![0xAAu8; 100];
        let s2 = vec![0xBBu8; 200];
        let s3 = vec![0xCCu8; 50];
        let playlist = "#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXT-X-PLAYLIST-TYPE:VOD\n\
                        #EXTINF:2.00000,\nseg1.ts\n#EXTINF:2.00000,\nseg2.ts\n#EXTINF:1.00000,\nseg3.ts\n#EXT-X-ENDLIST\n";
        let server = TestServer::serve(vec![
            ("/index.m3u8", Route::text(playlist)),
            ("/seg1.ts", Route::new(s1.clone())),
            ("/seg2.ts", Route::new(s2.clone())),
            ("/seg3.ts", Route::new(s3.clone())),
        ]);
        let client = default_client().unwrap();
        let hls = open_hls(&client, &server.url("/index.m3u8"), &OpenOptions::default()).unwrap();
        assert!(!hls.is_live, "有 ENDLIST 就是点播");
        assert_eq!(hls.segment_count, 3);
        assert_eq!(hls.duration_us, Some(5_000_000), "时长按分片时长相加（整数微秒）");

        let mut want = Vec::new();
        want.extend_from_slice(&s1);
        want.extend_from_slice(&s2);
        want.extend_from_slice(&s3);
        assert_eq!(read_all(&hls.source), want, "三个分片必须按顺序逐字节拼起来");
    }

    #[test]
    fn hls_master_playlist_picks_a_variant() {
        let master = "#EXTM3U\n\
                      #EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=640x360\nlow/index.m3u8\n\
                      #EXT-X-STREAM-INF:BANDWIDTH=3000000,RESOLUTION=1920x1080\nhigh/index.m3u8\n";
        let media = "#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXTINF:2.0,\nv.ts\n#EXT-X-ENDLIST\n";
        let server = TestServer::serve(vec![
            ("/master.m3u8", Route::text(master)),
            ("/low/index.m3u8", Route::text(media)),
            ("/high/index.m3u8", Route::text(media)),
            ("/high/v.ts", Route::new(vec![1u8; 10])),
            ("/low/v.ts", Route::new(vec![2u8; 10])),
        ]);
        let client = default_client().unwrap();
        // 上限 720p：应该选中 640x360 那条（3000000 那条超了）
        let opts = OpenOptions { max_height: Some(720), ..Default::default() };
        let hls = open_hls(&client, &server.url("/master.m3u8"), &opts).unwrap();
        assert_eq!(hls.variant_resolution, Some((640, 360)));
        assert_eq!(read_all(&hls.source), vec![2u8; 10], "取的是被选中（低码率）变体的分片");
        assert!(
            server.request_count("high/") == 0,
            "没选中的变体不该被请求（省带宽）：{:?}",
            server.requests()
        );
    }

    #[test]
    fn hls_byte_range_segments_are_clipped() {
        // 一个文件里放三段数据，用 BYTERANGE 各取一段
        let mut blob = vec![0u8; 30];
        for (i, b) in blob.iter_mut().enumerate() {
            *b = i as u8;
        }
        let playlist = "#EXTM3U\n#EXT-X-TARGETDURATION:2\n\
                        #EXTINF:1.0,\n#EXT-X-BYTERANGE:10@0\nall.bin\n\
                        #EXTINF:1.0,\n#EXT-X-BYTERANGE:10@10\nall.bin\n\
                        #EXTINF:1.0,\n#EXT-X-BYTERANGE:10@20\nall.bin\n\
                        #EXT-X-ENDLIST\n";
        let server = TestServer::serve(vec![
            ("/p.m3u8", Route::text(playlist)),
            ("/all.bin", Route::new(blob.clone())),
        ]);
        let client = default_client().unwrap();
        let hls = open_hls(&client, &server.url("/p.m3u8"), &OpenOptions::default()).unwrap();
        let got = read_all(&hls.source);
        assert_eq!(got, blob, "按字节范围切出来的三段拼起来必须等于原文件");
    }

    #[test]
    fn hls_encrypted_segment_is_decrypted() {
        // 密文由 OpenSSL（node 的 crypto）对 "ZuvRust-AES128!!"（16 字节）加密得到，
        // 向量在这里钉死：引擎自己解出来必须完全一致。
        // 注意 PKCS7 在"输入正好是块大小"时会**再加一整个块**，所以明文 16 字节 → 密文 32 字节。
        // ⚠️ 明文与密文是一对，改名时**两边必须一起重新生成** —— 只改断言里的明文会让
        // 长度对不上（曾经就因此挂过：改名把 16 字节的明文变成 12 字节，密文还是旧的 32 字节）。
        let key = "000102030405060708090a0b0c0d0e0f";
        let iv = "101112131415161718191a1b1c1d1e1f";
        let ct = "e8cc4920a4734cddc5c68f25628bdb6acc1f8e31a593f189c923ab53d89973ef";
        let ct_bytes = hex_to_bytes(ct);
        assert_eq!(ct_bytes.len(), 32, "向量本身要自洽：16 字节明文 → 32 字节密文");
        let playlist = format!(
            "#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXT-X-KEY:METHOD=AES-128,URI=\"key.bin\",IV=0x{iv}\n\
             #EXTINF:2.0,\nenc.ts\n#EXT-X-ENDLIST\n"
        );
        let server = TestServer::serve(vec![
            ("/p.m3u8", Route::text(playlist)),
            ("/key.bin", Route::new(hex_to_bytes(key))),
            ("/enc.ts", Route::new(ct_bytes)),
        ]);
        let client = default_client().unwrap();
        let hls = open_hls(&client, &server.url("/p.m3u8"), &OpenOptions::default()).unwrap();
        let got = read_all(&hls.source);
        assert_eq!(got, b"ZuvRust-AES128!!", "加密分片必须被解成明文（去掉 PKCS7 填充）");
    }

    fn hex_to_bytes(s: &str) -> Vec<u8> {
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn dash_track_concatenates_init_and_segments() {
        let mpd = r#"<?xml version="1.0"?>
<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" type="static" mediaPresentationDuration="PT6S">
  <Period>
    <AdaptationSet contentType="video" mimeType="video/mp4" codecs="avc1.64001f">
      <SegmentTemplate timescale="1000" duration="2000" startNumber="1"
                       initialization="init-$RepresentationID$.mp4"
                       media="seg-$RepresentationID$-$Number$.m4s"/>
      <Representation id="v0" bandwidth="1000000" width="640" height="360"/>
    </AdaptationSet>
  </Period>
</MPD>"#;
        let init = vec![0x11u8; 20];
        let seg1 = vec![0x22u8; 30];
        let seg2 = vec![0x33u8; 30];
        let seg3 = vec![0x44u8; 30];
        let server = TestServer::serve(vec![
            ("/s.mpd", Route::text(mpd)),
            ("/init-v0.mp4", Route::new(init.clone())),
            ("/seg-v0-1.m4s", Route::new(seg1.clone())),
            ("/seg-v0-2.m4s", Route::new(seg2.clone())),
            ("/seg-v0-3.m4s", Route::new(seg3.clone())),
        ]);
        let client = default_client().unwrap();
        let tracks = match open_dash_tracks(&client, &server.url("/s.mpd"), &OpenOptions::default()) {
            Ok(t) => t,
            Err(e) => panic!("打开 MPD 失败：{e}；请求记录：{:?}", server.requests()),
        };
        assert_eq!(tracks.len(), 1, "这份 MPD 只有视频轨");
        assert_eq!(tracks[0].content_type, "video");
        assert_eq!(tracks[0].representation_id, "v0");
        assert_eq!(
            tracks[0].segment_count, 3,
            "MPD 里 6 秒 / 每段 2 秒 = 3 段；请求记录：{:?}",
            server.requests()
        );
        assert!(tracks[0].source.len().unwrap_or(0) > 0, "拼出来的流不该是 0 字节；请求记录：{:?}", server.requests());
        let mut want = Vec::new();
        want.extend_from_slice(&init);
        want.extend_from_slice(&seg1);
        want.extend_from_slice(&seg2);
        want.extend_from_slice(&seg3);
        assert_eq!(read_all(&tracks[0].source), want, "init + 三个媒体段必须按顺序拼起来");
    }

    #[test]
    fn hls_byte_range_works_when_the_server_ignores_range() {
        // 有些 CDN 无视 Range 直接回全量。这时**绝不能**把整段当成"你要的那一段"
        // —— 那会让拼出来的流整体错位。引擎要自己按偏移切。
        let blob: Vec<u8> = (0u8..30).collect();
        let playlist = "#EXTM3U\n#EXT-X-TARGETDURATION:2\n\
                        #EXTINF:1.0,\n#EXT-X-BYTERANGE:10@10\nall.bin\n#EXT-X-ENDLIST\n";
        let server = TestServer::serve(vec![
            ("/p.m3u8", Route::text(playlist)),
            ("/all.bin", Route::new(blob.clone()).no_range()),
        ]);
        let client = default_client().unwrap();
        let hls = open_hls(&client, &server.url("/p.m3u8"), &OpenOptions::default()).unwrap();
        let got = read_all(&hls.source);
        assert_eq!(got, blob[10..20].to_vec(), "服务器无视 Range 时引擎要自己按偏移切");
    }

    #[test]
    fn hls_live_playlist_has_no_duration() {        let playlist = "#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXTINF:2.0,\na.ts\n#EXTINF:2.0,\nb.ts\n";
        let server = TestServer::serve(vec![
            ("/live.m3u8", Route::text(playlist)),
            ("/a.ts", Route::new(vec![1u8; 4])),
            ("/b.ts", Route::new(vec![2u8; 4])),
        ]);
        let client = default_client().unwrap();
        let hls = open_hls(&client, &server.url("/live.m3u8"), &OpenOptions::default()).unwrap();
        assert!(hls.is_live, "没有 ENDLIST 就是直播");
        assert_eq!(hls.duration_us, None, "直播不该报时长（报了宿主会画一个假进度条）");
    }
}
