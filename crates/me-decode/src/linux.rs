//! Linux 硬解：**VA-API**（Video Acceleration API）。
//!
//! 为什么用它：它是 Linux 上显卡厂商（Intel/AMD/NVIDIA）共用的**系统级**解码接口
//! —— 就像 macOS 的 VideoToolbox、Windows 的 Media Foundation 一样，属于操作系统
//! 本来就有的能力，不是 FFmpeg/GStreamer 那类第三方媒体引擎。
//!
//! ## 这条路的特殊之处：**它只做一半**
//!
//! VA-API 的解码入口是 **VLD（变长解码）**：驱动从"熵解码"往后接手，**前面全得
//! 调用方自己做** —— SPS/PPS/slice 头解析、参考帧列表构造（H.264 8.2.4）、
//! 输出顺序缓冲（按 POC 重排）、按 `frame_cropping` 裁出显示尺寸。所以本文件
//! 一半代码在把 [`crate::h264`] 的解析结果翻译成 VA-API 的结构体，另一半在管
//! "面池"。
//!
//! ## 面（surface）的所有权规则
//!
//! 驱动不拥有面，调用方拥有。三条规则必须同时满足，否则症状是**静默的错帧**：
//!
//! 1. 一张面在"被当作参考帧"期间不能复用（否则驱动手里的参考帧内容已被覆盖）；
//! 2. 一张面在"还没按显示顺序吐出去"期间不能复用（否则画面顺序错）；
//! 3. 复用一个面之前必须同步过（驱动可能还在往里写）。
//!
//! 所以面池大小 = 参考帧上限 + 重排窗口 + 余量，不够用时**先把显示队头吐出去
//! 再复用**（宁可早出一帧，也不覆盖还没读完的面）。
//!
//! ## 不支持的构造
//!
//! 解析层（[`crate::h264`]）会把隔行/10 位/FMO/MMCO 之类明确报成 `Unsupported`；
//! 本文件只补那些"只有到了驱动这一层才看得见"的：尺寸中途变化、面池不够、
//! 驱动给回来的图像格式不是 NV12 —— 一样明确报错，不静默出黑帧。

use std::ffi::{c_char, c_int, c_void, CStr};
use std::os::fd::AsRawFd;
use std::sync::OnceLock;

use me_core::{CodecId, MediaError, Result};

use crate::h264::{
    self, Config, Dpb, PocState, RefPic, SliceHeader, NAL_AUD, NAL_PPS, NAL_SEI, NAL_SPS,
};
use crate::VideoFrame;

// ── 类型与常量（逐字对齐 libva 的 va/va.h） ─────────────────────────────────
//
// 手写声明而不是引 crate：用到的符号就这些，而"零第三方依赖"是引擎底线。
// 结构体布局按 va.h 原文抄（`uint32_t va_reserved[VA_PADDING_LOW]` = 4 个 u32、
// `VA_PADDING_MEDIUM` = 8 个），里面的位域 union（seq_fields / pic_fields）
// 用 u32 + 手工移位表示 —— 布局等价，但能写测试，也不用碰 Rust 没有的位域。

type VADisplay = *mut c_void;
type VAStatus = c_int;
type VAConfigID = u32;
type VAContextID = u32;
type VASurfaceID = u32;
type VABufferID = u32;
type VAImageID = u32;
type VAProfile = c_int;
type VAEntrypoint = c_int;
type VABufferType = c_int;
type VASurfaceStatus = c_int;

const VA_STATUS_SUCCESS: VAStatus = 0;

const VA_PROFILE_H264_BASELINE: VAProfile = 5;
const VA_PROFILE_H264_MAIN: VAProfile = 6;
const VA_PROFILE_H264_HIGH: VAProfile = 7;
const VA_PROFILE_H264_CONSTRAINED_BASELINE: VAProfile = 13;
const VA_ENTRYPOINT_VLD: VAEntrypoint = 1;

const VA_RT_FORMAT_YUV420: u32 = 0x0000_0001;

const VA_PICTURE_PARAMETER_BUFFER_TYPE: VABufferType = 0;
const VA_IQ_MATRIX_BUFFER_TYPE: VABufferType = 1;
const VA_SLICE_PARAMETER_BUFFER_TYPE: VABufferType = 4;
const VA_SLICE_DATA_BUFFER_TYPE: VABufferType = 5;
const VA_SLICE_DATA_FLAG_ALL: u32 = 0;

const VA_PROGRESSIVE: c_int = 0x1;
const VA_FOURCC_NV12: u32 = 0x3231_564E;
const VA_INVALID_SURFACE: VASurfaceID = 0xFFFF_FFFF;

const VA_PICTURE_H264_INVALID: u32 = 0x0000_0001;
const VA_PICTURE_H264_SHORT_TERM_REFERENCE: u32 = 0x0000_0008;

const VASURFACE_READY: VASurfaceStatus = 4;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct VaPictureH264 {
    picture_id: VASurfaceID,
    frame_idx: u32,
    flags: u32,
    top_field_order_cnt: i32,
    bottom_field_order_cnt: i32,
    reserved: [u32; 4], // va_reserved[VA_PADDING_LOW]
}

impl VaPictureH264 {
    fn invalid() -> Self {
        Self {
            picture_id: VA_INVALID_SURFACE,
            flags: VA_PICTURE_H264_INVALID,
            ..Self::default()
        }
    }

    fn short_term(surface: VASurfaceID, frame_num: u32, poc: i32) -> Self {
        Self {
            picture_id: surface,
            frame_idx: frame_num,
            flags: VA_PICTURE_H264_SHORT_TERM_REFERENCE,
            top_field_order_cnt: poc,
            bottom_field_order_cnt: poc,
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct VaConfigAttrib {
    attrib_type: u32,
    value: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct VaPictureParameterBufferH264 {
    curr_pic: VaPictureH264,
    reference_frames: [VaPictureH264; 16],
    picture_width_in_mbs_minus1: u16,
    picture_height_in_mbs_minus1: u16,
    bit_depth_luma_minus8: u8,
    bit_depth_chroma_minus8: u8,
    num_ref_frames: u8,
    /// 位域 union（`seq_fields`）—— 拼装见 [`seq_fields`]
    seq_fields: u32,
    num_slice_groups_minus1: u8,
    slice_group_map_type: u8,
    slice_group_change_rate_minus1: u16,
    pic_init_qp_minus26: i8,
    pic_init_qs_minus26: i8,
    chroma_qp_index_offset: i8,
    second_chroma_qp_index_offset: i8,
    /// 位域 union（`pic_fields`）—— 拼装见 [`pic_fields`]
    pic_fields: u32,
    frame_num: u16,
    reserved: [u32; 8], // va_reserved[VA_PADDING_MEDIUM]
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct VaIqMatrixBufferH264 {
    scaling_list_4x4: [[u8; 16]; 6],
    scaling_list_8x8: [[u8; 64]; 2],
    reserved: [u32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct VaSliceParameterBufferH264 {
    slice_data_size: u32,
    slice_data_offset: u32,
    slice_data_flag: u32,
    slice_data_bit_offset: u16,
    first_mb_in_slice: u16,
    slice_type: u8,
    direct_spatial_mv_pred_flag: u8,
    num_ref_idx_l0_active_minus1: u8,
    num_ref_idx_l1_active_minus1: u8,
    cabac_init_idc: u8,
    slice_qp_delta: i8,
    disable_deblocking_filter_idc: u8,
    slice_alpha_c0_offset_div2: i8,
    slice_beta_offset_div2: i8,
    ref_pic_list0: [VaPictureH264; 32],
    ref_pic_list1: [VaPictureH264; 32],
    luma_log2_weight_denom: u8,
    chroma_log2_weight_denom: u8,
    luma_weight_l0_flag: u8,
    luma_weight_l0: [i16; 32],
    luma_offset_l0: [i16; 32],
    chroma_weight_l0_flag: u8,
    chroma_weight_l0: [[i16; 2]; 32],
    chroma_offset_l0: [[i16; 2]; 32],
    luma_weight_l1_flag: u8,
    luma_weight_l1: [i16; 32],
    luma_offset_l1: [i16; 32],
    chroma_weight_l1_flag: u8,
    chroma_weight_l1: [[i16; 2]; 32],
    chroma_offset_l1: [[i16; 2]; 32],
    reserved: [u32; 4],
}

impl Default for VaSliceParameterBufferH264 {
    fn default() -> Self {
        // 全 0 起步，但**参考帧列表要填成"无效项"**：0 号面是真实存在的面，
        // 留 0 会让驱动以为"参考帧是 0 号面"，那是静默错帧。
        let mut s: Self = unsafe { std::mem::zeroed() };
        s.ref_pic_list0 = [VaPictureH264::invalid(); 32];
        s.ref_pic_list1 = [VaPictureH264::invalid(); 32];
        s
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct VaImageFormat {
    fourcc: u32,
    byte_order: u32,
    bits_per_pixel: u32,
    depth: u32,
    red_mask: u32,
    green_mask: u32,
    blue_mask: u32,
    alpha_mask: u32,
    reserved: [u32; 4], // va_reserved[VA_PADDING_LOW]
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct VaImage {
    image_id: VAImageID,
    format: VaImageFormat,
    buf: VABufferID,
    width: u16,
    height: u16,
    data_size: u32,
    num_planes: u32,
    pitches: [u32; 3],
    offsets: [u32; 3],
    num_palette_entries: i32,
    entry_bytes: i32,
    component_order: [i8; 4],
    reserved: [u32; 4], // va_reserved[VA_PADDING_LOW]
}

// ── 动态加载 libva ───────────────────────────────────────────────────────────

#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

/// `RTLD_NOW`：立刻解析全部符号，缺什么在加载时就知道（别等解码中途崩）。
const RTLD_NOW: c_int = 2;

/// libva 的函数表。
///
/// 为什么动态加载而不是直接链接：**引擎要能在没装 libva 的机器上照常跑**
/// （容器、CI、无桌面的服务器都是这样）—— 链接期依赖会让整个二进制起不来，
/// 而我们只需要"探测不到就明确报没有硬解"。
struct LibVa {
    _handle: *mut c_void,
    initialize: unsafe extern "C" fn(VADisplay, *mut c_int, *mut c_int) -> VAStatus,
    terminate: unsafe extern "C" fn(VADisplay) -> VAStatus,
    query_vendor_string: unsafe extern "C" fn(VADisplay) -> *const c_char,
    error_str: unsafe extern "C" fn(VAStatus) -> *const c_char,
    query_config_entrypoints:
        unsafe extern "C" fn(VADisplay, VAProfile, *mut VAEntrypoint, *mut c_int) -> VAStatus,
    create_config: unsafe extern "C" fn(
        VADisplay,
        VAProfile,
        VAEntrypoint,
        *const VaConfigAttrib,
        c_int,
        *mut VAConfigID,
    ) -> VAStatus,
    destroy_config: unsafe extern "C" fn(VADisplay, VAConfigID) -> VAStatus,
    create_surfaces: unsafe extern "C" fn(
        VADisplay,
        u32,
        u32,
        u32,
        *mut VASurfaceID,
        u32,
        *mut c_void,
        u32,
    ) -> VAStatus,
    destroy_surfaces: unsafe extern "C" fn(VADisplay, *mut VASurfaceID, c_int) -> VAStatus,
    create_context: unsafe extern "C" fn(
        VADisplay,
        VAConfigID,
        c_int,
        c_int,
        c_int,
        *mut VASurfaceID,
        c_int,
        *mut VAContextID,
    ) -> VAStatus,
    destroy_context: unsafe extern "C" fn(VADisplay, VAContextID) -> VAStatus,
    create_buffer: unsafe extern "C" fn(
        VADisplay,
        VAContextID,
        VABufferType,
        u32,
        u32,
        *const c_void,
        *mut VABufferID,
    ) -> VAStatus,
    destroy_buffer: unsafe extern "C" fn(VADisplay, VABufferID) -> VAStatus,
    map_buffer: unsafe extern "C" fn(VADisplay, VABufferID, *mut *mut c_void) -> VAStatus,
    unmap_buffer: unsafe extern "C" fn(VADisplay, VABufferID) -> VAStatus,
    begin_picture: unsafe extern "C" fn(VADisplay, VAContextID, VASurfaceID) -> VAStatus,
    render_picture:
        unsafe extern "C" fn(VADisplay, VAContextID, *const VABufferID, c_int) -> VAStatus,
    end_picture: unsafe extern "C" fn(VADisplay, VAContextID) -> VAStatus,
    sync_surface: unsafe extern "C" fn(VADisplay, VASurfaceID) -> VAStatus,
    query_surface_status:
        unsafe extern "C" fn(VADisplay, VASurfaceID, *mut VASurfaceStatus) -> VAStatus,
    derive_image: unsafe extern "C" fn(VADisplay, VASurfaceID, *mut VaImage) -> VAStatus,
    create_image:
        unsafe extern "C" fn(VADisplay, *const VaImageFormat, c_int, c_int, *mut VaImage) -> VAStatus,
    get_image:
        unsafe extern "C" fn(VADisplay, VASurfaceID, c_int, c_int, u32, u32, VAImageID) -> VAStatus,
    destroy_image: unsafe extern "C" fn(VADisplay, VAImageID) -> VAStatus,
}

/// `vaGetDisplayDRM`（在 `libva-drm.so.2` 里，和主库分开加载）。
type GetDisplayDrm = unsafe extern "C" fn(c_int) -> VADisplay;

// 安全性：`LibVa` 里全是函数指针 + 一个不透明句柄，跨线程只读共享是安全的。
unsafe impl Send for LibVa {}
unsafe impl Sync for LibVa {}

impl LibVa {
    fn load() -> Option<LibVa> {
        unsafe {
            // libva.so.2 是运行时的正式名字；libva.so 只存在于开发包（构建机）
            let handle = [b"libva.so.2\0".as_slice(), b"libva.so\0".as_slice()]
                .iter()
                .map(|name| dlopen(name.as_ptr() as *const c_char, RTLD_NOW))
                .find(|h| !h.is_null())?;
            Some(LibVa {
                _handle: handle,
                initialize: sym(handle, b"vaInitialize\0")?,
                terminate: sym(handle, b"vaTerminate\0")?,
                query_vendor_string: sym(handle, b"vaQueryVendorString\0")?,
                error_str: sym(handle, b"vaErrorStr\0")?,
                query_config_entrypoints: sym(handle, b"vaQueryConfigEntrypoints\0")?,
                create_config: sym(handle, b"vaCreateConfig\0")?,
                destroy_config: sym(handle, b"vaDestroyConfig\0")?,
                create_surfaces: sym(handle, b"vaCreateSurfaces\0")?,
                destroy_surfaces: sym(handle, b"vaDestroySurfaces\0")?,
                create_context: sym(handle, b"vaCreateContext\0")?,
                destroy_context: sym(handle, b"vaDestroyContext\0")?,
                create_buffer: sym(handle, b"vaCreateBuffer\0")?,
                destroy_buffer: sym(handle, b"vaDestroyBuffer\0")?,
                map_buffer: sym(handle, b"vaMapBuffer\0")?,
                unmap_buffer: sym(handle, b"vaUnmapBuffer\0")?,
                begin_picture: sym(handle, b"vaBeginPicture\0")?,
                render_picture: sym(handle, b"vaRenderPicture\0")?,
                end_picture: sym(handle, b"vaEndPicture\0")?,
                sync_surface: sym(handle, b"vaSyncSurface\0")?,
                query_surface_status: sym(handle, b"vaQuerySurfaceStatus\0")?,
                derive_image: sym(handle, b"vaDeriveImage\0")?,
                create_image: sym(handle, b"vaCreateImage\0")?,
                get_image: sym(handle, b"vaGetImage\0")?,
                destroy_image: sym(handle, b"vaDestroyImage\0")?,
            })
        }
    }

    fn msg(&self, status: VAStatus) -> String {
        if status == VA_STATUS_SUCCESS {
            return "成功".into();
        }
        unsafe {
            let p = (self.error_str)(status);
            if p.is_null() {
                format!("VAStatus {status}")
            } else {
                format!("{}（VAStatus {status:#x}）", CStr::from_ptr(p).to_string_lossy())
            }
        }
    }
}

/// 从已加载的库里取一个符号（`name` 是带 `\0` 的字节串）。
unsafe fn sym<T: Copy>(handle: *mut c_void, name: &[u8]) -> Option<T> {
    let p = dlsym(handle, name.as_ptr() as *const c_char);
    if p.is_null() {
        return None;
    }
    Some(std::mem::transmute_copy(&p))
}

/// `libva.so.2` + `libva-drm.so.2` 的加载结果（进程内只做一次）。
struct VaLibs {
    va: LibVa,
    get_display_drm: GetDisplayDrm,
}

fn libs() -> Option<&'static VaLibs> {
    static LIBS: OnceLock<Option<VaLibs>> = OnceLock::new();
    LIBS.get_or_init(|| unsafe {
        let va = LibVa::load()?;
        // DRM 那条库缺了就直接判"没有硬解"：我们走的就是 DRM render 节点这条路
        //（不依赖 X11/Wayland，SSH 与无桌面环境一样能用）。
        let drm = dlopen(b"libva-drm.so.2\0".as_ptr() as *const c_char, RTLD_NOW);
        if drm.is_null() {
            return None;
        }
        let get_display_drm: GetDisplayDrm = sym(drm, b"vaGetDisplayDRM\0")?;
        Some(VaLibs { va, get_display_drm })
    })
    .as_ref()
}

/// 打开 DRM render 节点（`/dev/dri/renderD128`）并初始化一个显示连接。
///
/// 返回的**文件句柄必须一直活着**：libva 拿的是这个 fd，提前关掉会让后续所有
/// 调用落到一个失效的 fd 上（这是"明明初始化成功却什么都失败"的经典原因）。
fn open_display(libs: &'static VaLibs) -> Result<(std::fs::File, VADisplay)> {
    let path = "/dev/dri/renderD128";
    let file = std::fs::File::open(path).map_err(|e| {
        MediaError::Unsupported(format!(
            "打不开 {path}（{e}）：这台机器没有可用的 DRM 渲染节点"
        ))
    })?;
    let display = unsafe { (libs.get_display_drm)(file.as_raw_fd()) };
    if display.is_null() {
        return Err(MediaError::Unsupported(
            "vaGetDisplayDRM 返回空指针（DRM 节点在，但 libva 不认）".into(),
        ));
    }
    let mut major = 0;
    let mut minor = 0;
    let status = unsafe { (libs.va.initialize)(display, &mut major, &mut minor) };
    if status != VA_STATUS_SUCCESS {
        return Err(MediaError::Unsupported(format!(
            "vaInitialize 失败：{}",
            libs.va.msg(status)
        )));
    }
    Ok((file, display))
}

/// 探测（进程内缓存：探测要开设备，不能每次问都来一遍）。
/// 返回驱动名 —— `None` 就是"这台机器没有可用的 VA-API 硬解"。
fn probe() -> Option<&'static str> {
    static PROBE: OnceLock<Option<String>> = OnceLock::new();
    PROBE
        .get_or_init(|| {
            let libs = libs()?;
            let (file, display) = open_display(libs).ok()?;
            let vendor = unsafe {
                let p = (libs.va.query_vendor_string)(display);
                if p.is_null() {
                    String::from("未知驱动")
                } else {
                    CStr::from_ptr(p).to_string_lossy().trim().to_string()
                }
            };
            // 三个 profile 里**任意一个**能列出 VLD 入口，就说明能硬解 H.264
            let mut available = false;
            for profile in [
                VA_PROFILE_H264_CONSTRAINED_BASELINE,
                VA_PROFILE_H264_MAIN,
                VA_PROFILE_H264_HIGH,
            ] {
                let mut entrypoints: [VAEntrypoint; 8] = [0; 8];
                let mut count: c_int = 0;
                let status = unsafe {
                    (libs.va.query_config_entrypoints)(
                        display,
                        profile,
                        entrypoints.as_mut_ptr(),
                        &mut count,
                    )
                };
                if status == VA_STATUS_SUCCESS
                    && count > 0
                    && entrypoints[..(count.min(8) as usize)].contains(&VA_ENTRYPOINT_VLD)
                {
                    available = true;
                    break;
                }
            }
            unsafe { (libs.va.terminate)(display) };
            drop(file);
            if !available {
                tracing::info!("VA-API 起来了（{vendor}）但没有 H.264 VLD 入口");
                return None;
            }
            tracing::info!("VA-API 可用：{vendor}");
            Some(vendor)
        })
        .as_deref()
}

/// 这台机器上有没有 VA-API 硬解（这个后端只接了 H.264）。
pub fn hardware_decode_available(codec: &CodecId) -> bool {
    match codec {
        CodecId::H264 => probe().is_some(),
        _ => false,
    }
}

// ── 解码器 ───────────────────────────────────────────────────────────────────

/// 一张面的使用状态（三条所有权规则见文件头）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotUse {
    /// 空闲（上一次用完前已经同步过）
    Free,
    /// 正在被这一张图使用
    Decoding,
    /// 当参考帧（滑动窗口还没淘汰它）
    Reference,
    /// 已解码、等按显示顺序吐出去
    PendingDisplay,
    /// 两个身份同时成立
    ReferenceAndPending,
}

impl SlotUse {
    fn is_reference(self) -> bool {
        matches!(self, SlotUse::Reference | SlotUse::ReferenceAndPending)
    }

    fn is_pending(self) -> bool {
        matches!(self, SlotUse::PendingDisplay | SlotUse::ReferenceAndPending)
    }

    fn free(self) -> bool {
        self == SlotUse::Free
    }
}

/// 一张图（一个采样）的解码计划：所有 slice 的原样 NAL + 第一片的头。
///
/// 送驱动的切片数据缓冲就是**带 NAL 头的原始字节**（防竞争字节原样保留），
/// 从哪儿开始算切片数据由 `slice_data_bit_offset` 告诉驱动（va.h 的注释写得很死）。
struct PicturePlan {
    header: SliceHeader,
    slices: Vec<Vec<u8>>,
    pts_us: i64,
}

/// 等显示顺序的一帧（面还没读出来 —— 读早了会白拷一份）。
struct PendingFrame {
    slot: usize,
    poc: i32,
    pts_us: i64,
}

/// 重排窗口的封顶值。
///
/// VUI 没声明 `max_num_reorder_frames` 时，规范隐含值是 `MaxDpbFrames`（可能到 16），
/// 照 16 缓冲等于"再来 16 帧才吐第一帧"，4K 素材光显示队列就 200 MB。现实里重排
/// 深度就是编码器的 B 帧层级（x264/ffmpeg 默认 3、Apple 软件编码器实测是 1），
/// 封顶 8 既够用又不吃内存；真遇到更深的流只会让个别帧早出一拍，而且
/// [`VaDecoder::bump_one`] 会**明确告警**，不静默。
const MAX_REORDER_WINDOW: u32 = 8;

pub struct VaDecoder {
    libs: &'static VaLibs,
    /// DRM 的 fd 得一直活着（见 [`open_display`]）
    _drm_file: std::fs::File,
    display: VADisplay,
    config: VAConfigID,
    context: VAContextID,
    profile: VAProfile,
    coded_width: u32,
    coded_height: u32,
    display_width: u32,
    display_height: u32,
    /// 裁剪偏移，单位是 **CropUnit**（4:2:0 时 1 个单位 = 2 个亮度样本）
    crop_left_units: u32,
    crop_top_units: u32,
    sps: h264::Sps,
    pps: h264::Pps,
    nal_length_size: usize,
    poc: PocState,
    dpb: Dpb,
    pending: Vec<PendingFrame>,
    reorder_window: usize,
    slots: Vec<VASurfaceID>,
    slot_use: Vec<SlotUse>,
    max_emitted_poc: i32,
    /// `vaDeriveImage` 失败过就不再每次重试（退回 create+get 那条路）
    derive_unavailable: bool,
    trace: bool,
}

impl VaDecoder {
    /// 用 `avcC` 开一个 VA-API 硬解解码器。
    pub fn open(avcc: &[u8]) -> Result<Self> {
        let libs = libs().ok_or_else(|| {
            MediaError::Unsupported(
                "这台机器没有 VA-API 硬解（找不到 libva.so.2 / libva-drm.so.2）".into(),
            )
        })?;
        let cfg = Config::parse(avcc)?;
        let (file, display) = open_display(libs)?;

        let mut dec = Self {
            libs,
            _drm_file: file,
            display,
            config: 0,
            context: 0,
            profile: 0,
            coded_width: cfg.sps.coded_width(),
            coded_height: cfg.sps.coded_height(),
            display_width: cfg.sps.width(),
            display_height: cfg.sps.height(),
            crop_left_units: cfg.sps.crop_left,
            crop_top_units: cfg.sps.crop_top,
            reorder_window: cfg.sps.max_num_reorder_frames().min(MAX_REORDER_WINDOW) as usize,
            sps: cfg.sps.clone(),
            pps: cfg.pps.clone(),
            nal_length_size: cfg.nal_length_size,
            poc: PocState::new(),
            dpb: Dpb::default(),
            pending: Vec::new(),
            slots: Vec::new(),
            slot_use: Vec::new(),
            max_emitted_poc: i32::MIN,
            derive_unavailable: false,
            trace: std::env::var_os("ME_VA_TRACE").is_some(),
        };
        if let Err(e) = dec.init_va() {
            dec.teardown();
            return Err(e);
        }
        Ok(dec)
    }

    fn init_va(&mut self) -> Result<()> {
        let va = &self.libs.va;
        let mut last_err = String::new();
        for profile in profile_candidates(&self.sps) {
            let status = unsafe {
                (va.create_config)(
                    self.display,
                    profile,
                    VA_ENTRYPOINT_VLD,
                    std::ptr::null(),
                    0,
                    &mut self.config,
                )
            };
            if status == VA_STATUS_SUCCESS {
                self.profile = profile;
                break;
            }
            last_err = va.msg(status);
        }
        if self.config == 0 {
            return Err(MediaError::Unsupported(format!(
                "驱动建不了 H.264 解码配置（SPS profile {}）：{last_err}",
                self.sps.profile_idc
            )));
        }

        // 面池：参考帧上限 + 重排窗口 + 余量（余量给"正在解的那张"和"正在读的那张"）
        let pool = (self.sps.max_num_ref_frames as usize + self.reorder_window + 3).clamp(6, 20);
        let mut surfaces = vec![0u32; pool];
        let status = unsafe {
            (va.create_surfaces)(
                self.display,
                VA_RT_FORMAT_YUV420,
                self.coded_width,
                self.coded_height,
                surfaces.as_mut_ptr(),
                pool as u32,
                std::ptr::null_mut(),
                0,
            )
        };
        if status != VA_STATUS_SUCCESS {
            return Err(MediaError::Unsupported(format!(
                "建 {} 张 {}x{} 解码面失败：{}",
                pool,
                self.coded_width,
                self.coded_height,
                va.msg(status)
            )));
        }
        self.slots = surfaces;
        self.slot_use = vec![SlotUse::Free; pool];

        let status = unsafe {
            (va.create_context)(
                self.display,
                self.config,
                self.coded_width as c_int,
                self.coded_height as c_int,
                VA_PROGRESSIVE,
                self.slots.as_mut_ptr(),
                self.slots.len() as c_int,
                &mut self.context,
            )
        };
        if status != VA_STATUS_SUCCESS {
            return Err(MediaError::Unsupported(format!(
                "建解码上下文失败：{}",
                va.msg(status)
            )));
        }
        if self.trace {
            tracing::info!(
                "VA-API 解码器：{}x{}（显示 {}x{}）面池 {} 张 / profile {} / 重排窗口 {}",
                self.coded_width,
                self.coded_height,
                self.display_width,
                self.display_height,
                self.slots.len(),
                self.profile,
                self.reorder_window
            );
        }
        Ok(())
    }

    fn teardown(&mut self) {
        let va = &self.libs.va;
        unsafe {
            if self.context != 0 {
                (va.destroy_context)(self.display, self.context);
                self.context = 0;
            }
            if !self.slots.is_empty() {
                (va.destroy_surfaces)(
                    self.display,
                    self.slots.as_mut_ptr(),
                    self.slots.len() as c_int,
                );
                self.slots.clear();
                self.slot_use.clear();
            }
            if self.config != 0 {
                (va.destroy_config)(self.display, self.config);
                self.config = 0;
            }
            (va.terminate)(self.display);
        }
    }

    fn set_sps(&mut self, sps: h264::Sps) -> Result<()> {
        if sps.coded_width() != self.coded_width || sps.coded_height() != self.coded_height {
            return Err(MediaError::Unsupported(format!(
                "码流中途换了分辨率（{}x{} → {}x{}），这一版不支持",
                self.coded_width,
                self.coded_height,
                sps.coded_width(),
                sps.coded_height()
            )));
        }
        self.sps = sps;
        Ok(())
    }

    fn set_pps(&mut self, pps: h264::Pps) -> Result<()> {
        self.pps = pps;
        Ok(())
    }

    /// 取一张空面：没有空的就先把显示队头吐出去（吐出来的帧进 `out`）。
    fn take_surface(&mut self, out: &mut Vec<VideoFrame>) -> Result<usize> {
        if let Some(idx) = self.slot_use.iter().position(|s| s.free()) {
            self.slot_use[idx] = SlotUse::Decoding;
            return Ok(idx);
        }
        // 全都占用：按显示顺序排出最早的一帧，腾出它的面。
        // （面池 ≥ 参考上限 + 窗口 + 3，正常到不了这里；到了说明上游把
        //   参考帧记多了，宁可先出一帧也不覆盖还没读完的面。）
        if self.pending.is_empty() {
            return Err(MediaError::Bitstream(
                "面池全被参考帧占住且没有待显示帧可挤：参考帧记账出错".into(),
            ));
        }
        self.bump_one(out)?;
        let idx = self
            .slot_use
            .iter()
            .position(|s| s.free())
            .ok_or_else(|| MediaError::Bitstream("腾不出空面（不该发生）".into()))?;
        self.slot_use[idx] = SlotUse::Decoding;
        Ok(idx)
    }

    fn decode_picture(&mut self, plan: PicturePlan, out: &mut Vec<VideoFrame>) -> Result<()> {
        let first = plan.header.clone();
        if self.pps.id != first.pps_id {
            return Err(MediaError::Bitstream(format!(
                "切片引用了没见过的 PPS（id {}）",
                first.pps_id
            )));
        }
        // IDR：参考帧全清；显示队列也先排空 —— POC 会从 0 重来，两代之间没法比大小
        if first.idr {
            self.dpb.clear();
            self.flush_pending(out)?;
        }
        let poc = self.poc.compute(&self.sps, &first)?;
        let slot = self.take_surface(out)?;
        let surface = self.slots[slot];

        let mut buffers: Vec<VABufferID> = Vec::new();
        let pic_param = self.picture_params(surface, &first, poc);
        // ⚠️ 这里的失败路径必须把面还回池子：面池是**记账**出来的，
        // 漏还一张就是"越跑越少"的慢性泄漏（最后报"面池全被占住"）
        match self.create_buffer(VA_PICTURE_PARAMETER_BUFFER_TYPE, as_bytes(&pic_param)) {
            Ok(b) => buffers.push(b),
            Err(e) => {
                self.slot_use[slot] = SlotUse::Free;
                return Err(e);
            }
        }
        if self.pps.has_scaling_matrix(&self.sps) {
            let (l4, l8) = self.pps.effective_scaling(&self.sps);
            let iq = VaIqMatrixBufferH264 {
                scaling_list_4x4: l4,
                scaling_list_8x8: l8,
                reserved: [0; 4],
            };
            match self.create_buffer(VA_IQ_MATRIX_BUFFER_TYPE, as_bytes(&iq)) {
                Ok(b) => buffers.push(b),
                Err(e) => {
                    self.destroy_buffers(&buffers);
                    self.slot_use[slot] = SlotUse::Free;
                    return Err(e);
                }
            }
        }

        let va = &self.libs.va;
        let status = unsafe { (va.begin_picture)(self.display, self.context, surface) };
        if status != VA_STATUS_SUCCESS {
            self.destroy_buffers(&buffers);
            self.slot_use[slot] = SlotUse::Free;
            return Err(MediaError::Bitstream(format!(
                "vaBeginPicture 失败：{}",
                va.msg(status)
            )));
        }
        // 画面级参数（画面参数 + 量化矩阵）一次送进去
        let status = unsafe {
            (va.render_picture)(
                self.display,
                self.context,
                buffers.as_ptr(),
                buffers.len() as c_int,
            )
        };
        if status != VA_STATUS_SUCCESS {
            unsafe { (va.end_picture)(self.display, self.context) };
            self.destroy_buffers(&buffers);
            self.slot_use[slot] = SlotUse::Free;
            return Err(MediaError::Bitstream(format!(
                "送画面参数失败：{}",
                va.msg(status)
            )));
        }

        // 逐 slice：切片参数 + 切片数据（两个缓冲一起送）
        let mut slice_err: Option<String> = None;
        for (i, nal) in plan.slices.iter().enumerate() {
            let sh = if i == 0 {
                first.clone()
            } else {
                match h264::parse_slice_header(nal, &self.sps, &self.pps) {
                    Ok(sh) => sh,
                    Err(e) => {
                        slice_err = Some(format!("第 {} 个切片头解析失败：{e}", i + 1));
                        break;
                    }
                }
            };
            let slice_param = match self.slice_params(&sh, nal.len() as u32) {
                Ok(p) => p,
                Err(e) => {
                    slice_err = Some(format!("{e}"));
                    break;
                }
            };
            let param_buf = match self.create_buffer(VA_SLICE_PARAMETER_BUFFER_TYPE, as_bytes(&slice_param)) {
                Ok(b) => b,
                Err(e) => {
                    slice_err = Some(format!("{e}"));
                    break;
                }
            };
            let data_buf = match self.create_buffer(VA_SLICE_DATA_BUFFER_TYPE, nal) {
                Ok(b) => b,
                Err(e) => {
                    self.destroy_buffers(&[param_buf]);
                    slice_err = Some(format!("{e}"));
                    break;
                }
            };
            let pair = [param_buf, data_buf];
            let status = unsafe { (va.render_picture)(self.display, self.context, pair.as_ptr(), 2) };
            self.destroy_buffers(&pair);
            if status != VA_STATUS_SUCCESS {
                slice_err = Some(format!("送第 {} 个切片失败：{}", i + 1, va.msg(status)));
                break;
            }
        }

        let status = unsafe { (va.end_picture)(self.display, self.context) };
        self.destroy_buffers(&buffers);
        if let Some(e) = slice_err {
            self.slot_use[slot] = SlotUse::Free;
            return Err(MediaError::Bitstream(e));
        }
        if status != VA_STATUS_SUCCESS {
            self.slot_use[slot] = SlotUse::Free;
            return Err(MediaError::Bitstream(format!(
                "vaEndPicture 失败：{}",
                va.msg(status)
            )));
        }

        // ── 参考帧标记（滑动窗口）──
        let max_frame_num = 1u32 << self.sps.log2_max_frame_num;
        if first.is_reference() {
            let removed = self.dpb.mark_sliding_window(
                self.sps.max_num_ref_frames,
                first.frame_num,
                max_frame_num,
            );
            for r in removed {
                let idx = r.slot as usize;
                if matches!(self.slot_use.get(idx), Some(u) if u.is_reference()) {
                    self.slot_use[idx] = if self.slot_use[idx].is_pending() {
                        SlotUse::PendingDisplay
                    } else {
                        SlotUse::Free
                    };
                }
            }
            self.dpb.push(RefPic {
                slot: slot as u32,
                frame_num: first.frame_num,
                poc,
            });
            self.slot_use[slot] = SlotUse::ReferenceAndPending;
        } else {
            self.slot_use[slot] = SlotUse::PendingDisplay;
        }
        self.pending.push(PendingFrame {
            slot,
            poc,
            pts_us: plan.pts_us,
        });

        // 显示顺序：攒过一个重排窗口就把最小的 POC 吐出去
        while self.pending.len() > self.reorder_window {
            self.bump_one(out)?;
        }
        Ok(())
    }

    /// 按显示顺序（POC 最小）吐出一帧，并把它的面读出来。
    fn bump_one(&mut self, out: &mut Vec<VideoFrame>) -> Result<()> {
        let Some(pos) = (0..self.pending.len()).min_by_key(|&i| self.pending[i].poc) else {
            return Err(MediaError::Bitstream("显示队列是空的，没有可吐的帧".into()));
        };
        let frame = self.pending.remove(pos);
        if frame.poc < self.max_emitted_poc {
            // 不静默：这说明流的实际重排深度超过了我们愿意缓冲的量。前端按 pts 选帧，
            // 会让它顶多卡一拍，但日志里必须留痕。
            tracing::warn!(
                "显示顺序出现回退（POC {} 晚于 {} 才吐出）：重排窗口 {} 不够，画面顺序可能有一处错位",
                frame.poc,
                self.max_emitted_poc,
                self.reorder_window
            );
        }
        self.max_emitted_poc = self.max_emitted_poc.max(frame.poc);
        let idx = frame.slot;
        let nv12 = self.read_surface(self.slots[idx])?;
        self.slot_use[idx] = if self.slot_use[idx].is_reference() {
            SlotUse::Reference
        } else {
            SlotUse::Free
        };
        out.push(VideoFrame {
            width: self.display_width,
            height: self.display_height,
            pts_us: frame.pts_us,
            nv12,
        });
        Ok(())
    }

    fn flush_pending(&mut self, out: &mut Vec<VideoFrame>) -> Result<()> {
        while !self.pending.is_empty() {
            self.bump_one(out)?;
        }
        Ok(())
    }

    fn create_buffer(&self, kind: VABufferType, data: &[u8]) -> Result<VABufferID> {
        let va = &self.libs.va;
        let mut id: VABufferID = 0;
        let status = unsafe {
            (va.create_buffer)(
                self.display,
                self.context,
                kind,
                data.len() as u32,
                1,
                data.as_ptr() as *const c_void,
                &mut id,
            )
        };
        if status != VA_STATUS_SUCCESS {
            return Err(MediaError::Io(format!(
                "建 VA 缓冲失败（类型 {kind}）：{}",
                va.msg(status)
            )));
        }
        Ok(id)
    }

    fn destroy_buffers(&self, ids: &[VABufferID]) {
        let va = &self.libs.va;
        for id in ids {
            unsafe {
                (va.destroy_buffer)(self.display, *id);
            }
        }
    }

    /// 画面参数缓冲（`VAPictureParameterBufferH264`）。
    fn picture_params(
        &self,
        surface: VASurfaceID,
        sh: &SliceHeader,
        poc: i32,
    ) -> VaPictureParameterBufferH264 {
        let sps = &self.sps;
        let pps = &self.pps;
        let mut refs = [VaPictureH264::invalid(); 16];
        for (i, r) in self.dpb.refs().iter().take(16).enumerate() {
            refs[i] = VaPictureH264::short_term(self.slots[r.slot as usize], r.frame_num, r.poc);
        }
        VaPictureParameterBufferH264 {
            curr_pic: VaPictureH264 {
                picture_id: surface,
                frame_idx: sh.frame_num,
                flags: if sh.is_reference() {
                    VA_PICTURE_H264_SHORT_TERM_REFERENCE
                } else {
                    0
                },
                top_field_order_cnt: poc,
                bottom_field_order_cnt: poc,
                reserved: [0; 4],
            },
            reference_frames: refs,
            picture_width_in_mbs_minus1: (sps.pic_width_in_mbs - 1) as u16,
            picture_height_in_mbs_minus1: (sps.pic_height_in_map_units - 1) as u16,
            bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
            bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
            num_ref_frames: sps.max_num_ref_frames.min(16) as u8,
            seq_fields: seq_fields(sps),
            num_slice_groups_minus1: 0,
            slice_group_map_type: 0,
            slice_group_change_rate_minus1: 0,
            pic_init_qp_minus26: pps.pic_init_qp_minus26 as i8,
            pic_init_qs_minus26: pps.pic_init_qs_minus26 as i8,
            chroma_qp_index_offset: pps.chroma_qp_index_offset as i8,
            second_chroma_qp_index_offset: pps.second_chroma_qp_index_offset as i8,
            pic_fields: pic_fields(pps, sh),
            frame_num: sh.frame_num as u16,
            reserved: [0; 8],
        }
    }

    /// 切片参数缓冲（`VASliceParameterBufferH264`）。
    fn slice_params(&self, sh: &SliceHeader, nal_len: u32) -> Result<VaSliceParameterBufferH264> {
        if sh.slice_data_bit_offset > u16::MAX as usize {
            return Err(MediaError::Bitstream(format!(
                "切片头太长（{} 位），超出 VA-API 的表示范围",
                sh.slice_data_bit_offset
            )));
        }
        let max_frame_num = 1u32 << self.sps.log2_max_frame_num;
        let mut p = VaSliceParameterBufferH264 {
            slice_data_size: nal_len,
            slice_data_offset: 0,
            slice_data_flag: VA_SLICE_DATA_FLAG_ALL,
            slice_data_bit_offset: sh.slice_data_bit_offset as u16,
            first_mb_in_slice: sh.first_mb_in_slice as u16,
            slice_type: sh.kind(), // VA 要 0..4：P/B/I/SP/SI
            direct_spatial_mv_pred_flag: sh.direct_spatial_mv_pred_flag as u8,
            num_ref_idx_l0_active_minus1: sh.num_ref_idx_l0_active.saturating_sub(1) as u8,
            num_ref_idx_l1_active_minus1: sh.num_ref_idx_l1_active.saturating_sub(1) as u8,
            cabac_init_idc: sh.cabac_init_idc as u8,
            slice_qp_delta: sh.slice_qp_delta as i8,
            disable_deblocking_filter_idc: sh.disable_deblocking_filter_idc as u8,
            slice_alpha_c0_offset_div2: sh.slice_alpha_c0_offset_div2 as i8,
            slice_beta_offset_div2: sh.slice_beta_offset_div2 as i8,
            ..VaSliceParameterBufferH264::default()
        };

        if !sh.is_i() {
            let list0 = self.dpb.ref_list(
                false,
                sh.num_ref_idx_l0_active,
                &sh.l0_mod,
                sh.frame_num,
                max_frame_num,
            )?;
            fill_ref_list(&mut p.ref_pic_list0, &list0, &self.slots);
            if sh.is_b() {
                let list1 = self.dpb.ref_list(
                    true,
                    sh.num_ref_idx_l1_active,
                    &sh.l1_mod,
                    sh.frame_num,
                    max_frame_num,
                )?;
                fill_ref_list(&mut p.ref_pic_list1, &list1, &self.slots);
            }
        }

        // 权重：显式给了就用给的，没给就填"缺省权重"（1 << denom、偏移 0）——
        // flag 为 0 时驱动本该忽略这些值，但我们照样填满，免得驱动读到未初始化内存。
        let weight = sh.pred_weight.clone().unwrap_or_default();
        let denom_luma = weight.luma_log2_weight_denom;
        let denom_chroma = weight.chroma_log2_weight_denom;
        p.luma_log2_weight_denom = denom_luma as u8;
        p.chroma_log2_weight_denom = denom_chroma as u8;
        let (l0_count, l1_count) = if sh.is_i() {
            (0usize, 0usize)
        } else if sh.is_b() {
            (
                sh.num_ref_idx_l0_active as usize,
                sh.num_ref_idx_l1_active as usize,
            )
        } else {
            (sh.num_ref_idx_l0_active as usize, 0usize)
        };
        let explicit = sh.pred_weight.is_some();
        if explicit {
            // VA-API 的语义：flag = 1 表示"这个列表用了显式权重表"
            p.luma_weight_l0_flag = (l0_count > 0) as u8;
            p.chroma_weight_l0_flag = (l0_count > 0) as u8;
            p.luma_weight_l1_flag = (l1_count > 0) as u8;
            p.chroma_weight_l1_flag = (l1_count > 0) as u8;
        }
        fill_weights(
            &weight.l0,
            l0_count,
            denom_luma,
            denom_chroma,
            (&mut p.luma_weight_l0, &mut p.luma_offset_l0),
            (&mut p.chroma_weight_l0, &mut p.chroma_offset_l0),
        );
        fill_weights(
            &weight.l1,
            l1_count,
            denom_luma,
            denom_chroma,
            (&mut p.luma_weight_l1, &mut p.luma_offset_l1),
            (&mut p.chroma_weight_l1, &mut p.chroma_offset_l1),
        );
        Ok(p)
    }

    /// 把一张面读成**紧凑 NV12**（并裁掉 `frame_cropping` 的边）。
    fn read_surface(&mut self, surface: VASurfaceID) -> Result<Vec<u8>> {
        let va = &self.libs.va;
        let status = unsafe { (va.sync_surface)(self.display, surface) };
        if status != VA_STATUS_SUCCESS {
            return Err(MediaError::Bitstream(format!(
                "等解码结果失败：{}",
                va.msg(status)
            )));
        }
        let mut surface_status: VASurfaceStatus = 0;
        let st = unsafe { (va.query_surface_status)(self.display, surface, &mut surface_status) };
        if st == VA_STATUS_SUCCESS && surface_status != VASURFACE_READY {
            return Err(MediaError::Bitstream(format!(
                "这张面还没准备好（状态 {surface_status}）"
            )));
        }

        // 优先"派生"（零拷贝：直接映射驱动那边的 surface 内存）；
        // 驱动不支持就退回 create+get（驱动替我们做一次格式转换，多一次拷贝）。
        let mut image = VaImage::default();
        let derive_status = unsafe { (va.derive_image)(self.display, surface, &mut image) };
        let derived_nv12 =
            derive_status == VA_STATUS_SUCCESS && image.format.fourcc == VA_FOURCC_NV12;
        if derive_status == VA_STATUS_SUCCESS && !derived_nv12 {
            // 派生出来了但不是 NV12：**这张图同样要销毁**（派生图不是"借用"），
            // 不销毁就是每帧漏一个 image
            unsafe { (va.destroy_image)(self.display, image.image_id) };
            image = VaImage::default();
        }
        if !derived_nv12 {
            if !self.derive_unavailable {
                self.derive_unavailable = true;
                tracing::info!(
                    "vaDeriveImage 不可用（{}），改走 vaCreateImage + vaGetImage",
                    va.msg(derive_status)
                );
            }
            image = VaImage::default();
            let format = VaImageFormat {
                fourcc: VA_FOURCC_NV12,
                byte_order: 1, // VA_LSB_FIRST
                bits_per_pixel: 12,
                ..VaImageFormat::default()
            };
            let st = unsafe {
                (va.create_image)(
                    self.display,
                    &format,
                    self.coded_width as c_int,
                    self.coded_height as c_int,
                    &mut image,
                )
            };
            if st != VA_STATUS_SUCCESS {
                return Err(MediaError::Bitstream(format!(
                    "建临时图像失败：{}",
                    va.msg(st)
                )));
            }
            let st = unsafe {
                (va.get_image)(
                    self.display,
                    surface,
                    0,
                    0,
                    self.coded_width,
                    self.coded_height,
                    image.image_id,
                )
            };
            if st != VA_STATUS_SUCCESS {
                unsafe { (va.destroy_image)(self.display, image.image_id) };
                return Err(MediaError::Bitstream(format!(
                    "把画面拷进临时图像失败：{}",
                    va.msg(st)
                )));
            }
        }

        let result = self.copy_image(&image);
        unsafe { (va.destroy_image)(self.display, image.image_id) };
        result
    }

    /// 从映射的图像里拷出紧凑 NV12（顺带裁剪）。
    ///
    /// `vaDeriveImage` 给的 `width/height` 是**面的编码尺寸**，而 VA-API 这一层
    /// 没有裁剪概念 —— 裁剪是 H.264 的语法，得我们自己按 SPS 的 `frame_cropping`
    /// 去掉边（见 [`h264::Sps::width`]）。
    fn copy_image(&self, image: &VaImage) -> Result<Vec<u8>> {
        let va = &self.libs.va;
        let fmt = image.format.fourcc;
        if fmt != VA_FOURCC_NV12 {
            return Err(MediaError::Unsupported(format!(
                "驱动给的图像不是 NV12（fourcc {fmt:#x}），引擎对外只有 NV12"
            )));
        }
        if image.num_planes < 2 {
            return Err(MediaError::Bitstream("NV12 图像应当有两个平面".into()));
        }
        if (image.width as u32) < self.coded_width || (image.height as u32) < self.coded_height {
            return Err(MediaError::Bitstream(format!(
                "图像比面还小（{}x{} < {}x{}）",
                image.width, image.height, self.coded_width, self.coded_height
            )));
        }
        let mut base: *mut c_void = std::ptr::null_mut();
        let status = unsafe { (va.map_buffer)(self.display, image.buf, &mut base) };
        if status != VA_STATUS_SUCCESS || base.is_null() {
            return Err(MediaError::Io(format!("映射图像失败：{}", va.msg(status))));
        }
        let w = self.display_width as usize;
        let h = self.display_height as usize;
        let mut out = vec![0u8; VideoFrame::expected_len(self.display_width, self.display_height)];
        // 裁剪偏移：单位是 CropUnit（4:2:0 时亮度的 1 个单位 = 2 个样本，色度 = 1 个样本）
        let y_x = (self.crop_left_units * 2) as usize;
        let y_y = (self.crop_top_units * 2) as usize;
        let c_x = self.crop_left_units as usize;
        let c_y = self.crop_top_units as usize;
        let y_len = w * h;
        unsafe {
            let y_src = (base as *const u8).add(image.offsets[0] as usize);
            let y_pitch = image.pitches[0] as usize;
            for row in 0..h {
                let src = y_src.add((y_y + row) * y_pitch + y_x);
                std::ptr::copy_nonoverlapping(src, out.as_mut_ptr().add(row * w), w);
            }
            let c_src = (base as *const u8).add(image.offsets[1] as usize);
            let c_pitch = image.pitches[1] as usize;
            for row in 0..h / 2 {
                let src = c_src.add((c_y + row) * c_pitch + c_x);
                // NV12 的色度平面是 CbCr 交织，一行的字节数 = 亮度宽度
                std::ptr::copy_nonoverlapping(src, out.as_mut_ptr().add(y_len + row * w), w);
            }
            (va.unmap_buffer)(self.display, image.buf);
        }
        Ok(out)
    }
}

/// 把 `&T` 当字节切片用（`#[repr(C)]` 的纯数据 POD，没有指针成员）。
fn as_bytes<T: Copy>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) }
}

/// 按流的 profile 排一个候选顺序（用第一个能建起来的）。
///
/// 为什么要有候选：驱动暴露的 profile 粒度不一样 —— 有的只有 `ConstrainedBaseline`，
/// 有的把 Baseline 并进 Main。逐个试比"猜一个"稳。
fn profile_candidates(sps: &h264::Sps) -> Vec<VAProfile> {
    match sps.profile_idc {
        66 => vec![
            VA_PROFILE_H264_CONSTRAINED_BASELINE,
            VA_PROFILE_H264_BASELINE,
            VA_PROFILE_H264_MAIN,
        ],
        77 => vec![VA_PROFILE_H264_MAIN, VA_PROFILE_H264_HIGH],
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 => {
            vec![VA_PROFILE_H264_HIGH, VA_PROFILE_H264_MAIN]
        }
        _ => vec![
            VA_PROFILE_H264_HIGH,
            VA_PROFILE_H264_MAIN,
            VA_PROFILE_H264_BASELINE,
        ],
    }
}

/// `seq_fields` 的位域拼装（位序照 va.h 的 struct 定义，低位在前）。
fn seq_fields(sps: &h264::Sps) -> u32 {
    let mut v = 0u32;
    v |= (sps.chroma_format_idc as u32) & 0x3; // bit 0-1
    v |= (sps.separate_colour_plane_flag as u32) << 2;
    v |= (sps.gaps_in_frame_num_value_allowed_flag as u32) << 3;
    v |= (sps.frame_mbs_only_flag as u32) << 4;
    v |= (sps.mb_adaptive_frame_field_flag as u32) << 5;
    v |= (sps.direct_8x8_inference_flag as u32) << 6;
    // MinLumaBiPredSize8x8：高 profile 的最小明暗块是 8x8（va.h 指向 A.3.3.2）
    v |= (h264::is_high_profile(sps.profile_idc) as u32) << 7;
    v |= ((sps.log2_max_frame_num.saturating_sub(4)) as u32 & 0xF) << 8;
    v |= (sps.pic_order_cnt_type as u32 & 0x3) << 12;
    v |= ((sps.log2_max_pic_order_cnt_lsb.saturating_sub(4)) as u32 & 0xF) << 14;
    v |= (sps.delta_pic_order_always_zero_flag as u32) << 18;
    v
}

/// `pic_fields` 的位域拼装。
fn pic_fields(pps: &h264::Pps, sh: &SliceHeader) -> u32 {
    let mut v = 0u32;
    v |= pps.entropy_coding_mode_flag as u32;
    v |= (pps.weighted_pred_flag as u32) << 1;
    v |= (pps.weighted_bipred_idc as u32 & 0x3) << 2;
    v |= (pps.transform_8x8_mode_flag as u32) << 4;
    v |= (sh.field_pic_flag as u32) << 5;
    v |= (pps.constrained_intra_pred_flag as u32) << 6;
    v |= (pps.bottom_field_pic_order_in_frame_present_flag as u32) << 7;
    v |= (pps.deblocking_filter_control_present_flag as u32) << 8;
    v |= (pps.redundant_pic_cnt_present_flag as u32) << 9;
    v |= (sh.is_reference() as u32) << 10;
    v
}

/// 参考帧列表 → VA 的 `VAPictureH264` 数组（用不到的位置填"无效项"）。
fn fill_ref_list(dst: &mut [VaPictureH264; 32], list: &[RefPic], slots: &[VASurfaceID]) {
    for i in 0..32 {
        dst[i] = match list.get(i) {
            Some(r) => VaPictureH264::short_term(slots[r.slot as usize], r.frame_num, r.poc),
            None => VaPictureH264::invalid(),
        };
    }
}

/// 权重表 → VA 的四个数组。
fn fill_weights(
    entries: &[h264::WeightEntry],
    count: usize,
    denom_luma: u32,
    denom_chroma: u32,
    luma: (&mut [i16; 32], &mut [i16; 32]),
    chroma: (&mut [[i16; 2]; 32], &mut [[i16; 2]; 32]),
) {
    let (luma_w, luma_o) = luma;
    let (chroma_w, chroma_o) = chroma;
    for i in 0..count.min(32) {
        let entry = entries.get(i).copied().unwrap_or_default();
        match entry.luma {
            Some((w, o)) => {
                luma_w[i] = w as i16;
                luma_o[i] = o as i16;
            }
            None => {
                luma_w[i] = 1i16 << denom_luma;
                luma_o[i] = 0;
            }
        }
        match entry.chroma {
            Some(pair) => {
                chroma_w[i] = [pair[0].0 as i16, pair[1].0 as i16];
                chroma_o[i] = [pair[0].1 as i16, pair[1].1 as i16];
            }
            None => {
                chroma_w[i] = [1i16 << denom_chroma; 2];
                chroma_o[i] = [0, 0];
            }
        }
    }
}

impl crate::VideoDecoder for VaDecoder {
    fn decode(&mut self, sample_avcc: &[u8], pts_us: i64) -> Result<Vec<VideoFrame>> {
        if sample_avcc.is_empty() {
            return Ok(Vec::new());
        }
        let nals = h264::split_avcc_sample(sample_avcc, self.nal_length_size)?;
        let mut out = Vec::new();
        let mut plan: Option<PicturePlan> = None;
        for nal in nals {
            match nal[0] & 0x1F {
                NAL_SPS => {
                    let sps = h264::Sps::parse(nal)?;
                    sps.check_supported()?;
                    self.set_sps(sps)?;
                }
                NAL_PPS => {
                    let pps = h264::Pps::parse(nal, &self.sps)?;
                    pps.check_supported()?;
                    self.set_pps(pps)?;
                }
                NAL_SEI | NAL_AUD => {}
                t if h264::is_vcl(t) => {
                    if plan.is_none() {
                        let sh = h264::parse_slice_header(nal, &self.sps, &self.pps)?;
                        sh.check_supported()?;
                        plan = Some(PicturePlan {
                            header: sh,
                            slices: Vec::new(),
                            pts_us,
                        });
                    }
                    if let Some(p) = plan.as_mut() {
                        p.slices.push(nal.to_vec());
                    }
                }
                _ => {}
            }
        }
        if let Some(plan) = plan {
            self.decode_picture(plan, &mut out)?;
        }
        Ok(out)
    }

    fn flush(&mut self) -> Result<Vec<VideoFrame>> {
        let mut out = Vec::new();
        self.flush_pending(&mut out)?;
        Ok(out)
    }

    fn backend(&self) -> &'static str {
        "vaapi"
    }
}

// 安全性：VA-API 的显示连接与上下文**不能被并发使用**，但我们的约定是"一个解码器
// 只在一个线程上用"（`decode` 顺序调用），跨线程只是把所有权搬过去（比如未来把
// 解码放到专用线程）。所以 `Send` 是安全的。
unsafe impl Send for VaDecoder {}

impl Drop for VaDecoder {
    fn drop(&mut self) {
        if self.display.is_null() {
            return;
        }
        self.teardown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VideoDecoder;

    /// 探测在**没有 /dev/dri 的机器**（CI 容器）上必须给出确定的"没有"，
    /// 而不是崩或者挂住。
    #[test]
    fn probe_is_safe_without_a_gpu() {
        let available = hardware_decode_available(&CodecId::H264);
        println!(
            "VA-API 探测：h264={available} driver={:?}",
            probe()
        );
        // 这个后端只接 H.264
        assert!(!hardware_decode_available(&CodecId::Av1));
        assert!(!hardware_decode_available(&CodecId::Vp9));
        assert!(!hardware_decode_available(&CodecId::H265));
    }

    /// 探测不到时，开解码器要给一句**能读懂的话**（而不是崩在空指针上）。
    #[test]
    fn open_without_hardware_reports_a_readable_reason() {
        let e = match VaDecoder::open(&crate::h264::tests_support::mini_avcc()) {
            Ok(d) => {
                // 真机（有 VA-API）上应当能开起来
                assert_eq!(d.backend(), "vaapi");
                assert!(hardware_decode_available(&CodecId::H264));
                return;
            }
            Err(e) => e,
        };
        let text = format!("{e}");
        assert!(
            text.contains("VA-API") || text.contains("libva") || text.contains("DRM"),
            "失败文案要写清缺的是哪一样：{text}"
        );
        assert!(!hardware_decode_available(&CodecId::H264));
    }

    /// 位域拼装照 va.h 的位序（这几条错了驱动会解出错色/错位，而不会报错）。
    #[test]
    fn seq_and_pic_fields_bit_order() {
        let sps = h264::Sps {
            profile_idc: 100,
            chroma_format_idc: 1,
            frame_mbs_only_flag: true,
            direct_8x8_inference_flag: true,
            log2_max_frame_num: 8,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb: 8,
            ..h264::Sps::default()
        };
        let v = seq_fields(&sps);
        assert_eq!(v & 0x3, 1, "chroma_format_idc 占最低 2 位");
        assert_eq!((v >> 4) & 1, 1, "frame_mbs_only_flag 在第 4 位");
        assert_eq!((v >> 6) & 1, 1, "direct_8x8_inference_flag 在第 6 位");
        assert_eq!((v >> 7) & 1, 1, "MinLumaBiPredSize8x8 在第 7 位（高 profile）");
        assert_eq!((v >> 8) & 0xF, 4, "log2_max_frame_num_minus4 在 8-11 位");
        assert_eq!((v >> 14) & 0xF, 4, "log2_max_pic_order_cnt_lsb_minus4 在 14-17 位");

        let pps = h264::Pps {
            entropy_coding_mode_flag: true,
            weighted_pred_flag: true,
            weighted_bipred_idc: 2,
            transform_8x8_mode_flag: true,
            deblocking_filter_control_present_flag: true,
            redundant_pic_cnt_present_flag: true,
            ..h264::Pps::default()
        };
        let sh = SliceHeader {
            nal_ref_idc: 1,
            ..crate::h264::tests_support::minimal_slice()
        };
        let p = pic_fields(&pps, &sh);
        assert_eq!(p & 1, 1, "entropy_coding_mode_flag 是最低位");
        assert_eq!((p >> 1) & 1, 1, "weighted_pred_flag 第 1 位");
        assert_eq!((p >> 2) & 0x3, 2, "weighted_bipred_idc 占 2-3 位");
        assert_eq!((p >> 4) & 1, 1, "transform_8x8_mode_flag 第 4 位");
        assert_eq!((p >> 8) & 1, 1, "deblocking_filter_control_present_flag 第 8 位");
        assert_eq!((p >> 10) & 1, 1, "reference_pic_flag 第 10 位");
    }
}
