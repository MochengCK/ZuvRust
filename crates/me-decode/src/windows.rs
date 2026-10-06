//! Windows 硬解：Media Foundation 的**硬件 H.264 MFT** + D3D11。
//!
//! 为什么用它：Media Foundation 是 Windows 自带的媒体框架，硬件解码器以
//! **MFT（Media Foundation Transform）** 的形式挂进去 —— 和 macOS 的
//! VideoToolbox 同属"操作系统本来就有的能力"，不是第三方媒体引擎。
//!
//! ## 这条路的分工与 Linux 那条正好相反
//!
//! MFT 是**整解码器**：切片头、参考帧、DPB、输出顺序全由它管，我们只做三件事：
//!
//! 1. 把采样从 AVCC 换成 Annex-B（MF 的解码器吃带起始码的字节流）并送进去；
//! 2. 用 D3D11 让解码器直接往 GPU 纹理上解（**零系统内存拷贝**那条路），
//!    再拷到一张 staging 纹理读回来；
//! 3. 把输出帧的句柄收好、错了就说清楚。
//!
//! 所以本文件里没有 POC/DPB 那套东西 —— 那是 [`crate::h264`] 给 VA-API 用的。
//! 这里用 `h264` 只做两件事：从 `avcC` 里取参数集，以及**提前拒绝**不支持的
//! 构造（10 位/隔行/FMO…），免得用户只看到一句看不懂的 HRESULT。
//!
//! ## 异步 MFT（硬件解码器**都是**异步的）
//!
//! 硬件 MFT 会在自己的线程上干活，靠事件（`METransformNeedInput` /
//! `METransformHaveOutput`）通知"该喂了 / 有输出了"。规矩是：
//!
//! - 先用 `MF_TRANSFORM_ASYNC_UNLOCK` 解锁（不然连媒体类型都不让你设）；
//! - `ProcessInput` 可能返回 `MF_E_NOTACCEPTING`（先把输出抽干再喂）；
//! - `ProcessOutput` 返回 `MF_E_TRANSFORM_STREAM_CHANGE` 时要**重新协商输出类型**
//!   （解码器看到真实分辨率后才定得下来，这一步不做就会卡死在第一个采样上）。
//!
//! ## 手写 FFI
//!
//! 用的符号就这些，而"零第三方依赖"是引擎底线（连 `windows` crate 都不引）。
//! vtable 槽位是**绝对平铺索引**（把基类摊平后的编号），与 `windows` crate 的
//! `*_Vtbl` 字段序逐字核对过（含 `IMFAttributes` 里因 cfg 而重复声明的成员，
//! 只数 `pub` 的那份）。

use std::ffi::c_void;
use std::sync::OnceLock;

use me_core::{CodecId, MediaError, Result};

use crate::h264::{self, Config};
use crate::VideoFrame;

// ── COM / MF 的最小底座 ─────────────────────────────────────────────────────

/// `GUID`（与 C 的 `GUID` 同布局）。
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Guid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

macro_rules! guid {
    ($d1:expr, $d2:expr, $d3:expr, $d4:expr) => {
        Guid {
            data1: $d1,
            data2: $d2,
            data3: $d3,
            data4: $d4,
        }
    };
}

// 媒体类型与格式（值取自 Windows SDK 的 mfapi.h / mfidl.h；与 `windows` crate
// 里 `GUID::from_u128` 的常量逐个核对过）
const MF_MEDIA_TYPE_VIDEO: Guid = guid!(0x73646976, 0x0000, 0x0010, [0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71]);
const MF_VIDEO_FORMAT_H264: Guid = guid!(0x34363248, 0x0000, 0x0010, [0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71]);
const MF_VIDEO_FORMAT_NV12: Guid = guid!(0x3231564e, 0x0000, 0x0010, [0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71]);
const MFT_CATEGORY_VIDEO_DECODER: Guid = guid!(0xd6c02d4b, 0x6833, 0x45b4, [0x97, 0x1a, 0x05, 0xa4, 0xb0, 0x4b, 0xab, 0x91]);

// 媒体类型属性键
const MF_MT_MAJOR_TYPE: Guid = guid!(0x48eba18e, 0xf8c9, 0x4687, [0xbf, 0x11, 0x0a, 0x74, 0xc9, 0xf9, 0x6a, 0x8f]);
const MF_MT_SUBTYPE: Guid = guid!(0xf7e34c9a, 0x42e8, 0x4714, [0xb7, 0x4b, 0xcb, 0x29, 0xd7, 0x2c, 0x35, 0xe5]);
const MF_MT_FRAME_SIZE: Guid = guid!(0x1652c33d, 0xd6b2, 0x4012, [0xb8, 0x34, 0x72, 0x03, 0x08, 0x49, 0xa3, 0x7d]);
const MF_MT_INTERLACE_MODE: Guid = guid!(0xe2724bb8, 0xe676, 0x4806, [0xb4, 0xb2, 0xa8, 0xd6, 0xef, 0xb4, 0x4c, 0xcd]);
const MF_MT_MPEG_SEQUENCE_HEADER: Guid = guid!(0x3c036de7, 0x3ad0, 0x4c9e, [0x92, 0x16, 0xee, 0x6d, 0x6a, 0xc2, 0x1c, 0xb3]);
const MF_MT_DEFAULT_STRIDE: Guid = guid!(0x644b4e48, 0x1e02, 0x4516, [0xb0, 0xeb, 0xc0, 0x1c, 0xa9, 0xd4, 0x9a, 0xc6]);
const MF_TRANSFORM_ASYNC: Guid = guid!(0xf81a699a, 0x649a, 0x497d, [0x8c, 0x73, 0x29, 0xf8, 0xfe, 0xd6, 0xad, 0x7a]);
const MF_TRANSFORM_ASYNC_UNLOCK: Guid = guid!(0xe5666d6b, 0x3422, 0x4eb6, [0xa4, 0x21, 0xda, 0x7d, 0xb1, 0xf8, 0xe2, 0x07]);

// 接口 IID（QI 用；值同样与 SDK 头文件核对过）
const IID_IMF_TRANSFORM: Guid = guid!(0xbf94c121, 0x5b05, 0x4e6f, [0x80, 0x00, 0xba, 0x59, 0x89, 0x61, 0x41, 0x4d]);
const IID_IMF_MEDIA_EVENT_GENERATOR: Guid = guid!(0x2cd0bd52, 0xbcd5, 0x4b89, [0xb6, 0x2c, 0xea, 0xdc, 0x0c, 0x03, 0x1e, 0x7d]);
const IID_IMF_DXGI_BUFFER: Guid = guid!(0xe7174cfa, 0x1c9e, 0x48b1, [0x88, 0x66, 0x62, 0x62, 0x26, 0xbf, 0xc2, 0x58]);
const IID_ID3D11_TEXTURE2D: Guid = guid!(0x6f15aaf2, 0xd208, 0x4e89, [0x9a, 0xb4, 0x48, 0x95, 0x35, 0xd3, 0x4f, 0x9c]);
const IID_ID3D10_MULTITHREAD: Guid = guid!(0x9b7e4e00, 0x342c, 0x4106, [0xa1, 0x9f, 0x4f, 0x27, 0x04, 0xf6, 0x89, 0xf0]);

// HRESULT 常量（写成 u32 比较，避免和 i32 的符号位纠缠）
const S_OK: i32 = 0;
const MF_E_TRANSFORM_NEED_MORE_INPUT: u32 = 0xC00D_6D72;
const MF_E_TRANSFORM_STREAM_CHANGE: u32 = 0xC00D_6D61;
const MF_E_NO_MORE_TYPES: u32 = 0xC00D_36B9;
const MF_E_NOTACCEPTING: u32 = 0xC00D_36B5;
const MF_E_TRANSFORM_ASYNC_LOCKED: u32 = 0xC00D_6D77;

// 消息与枚举
const MFT_MESSAGE_COMMAND_FLUSH: u32 = 0;
const MFT_MESSAGE_COMMAND_DRAIN: u32 = 1;
const MFT_MESSAGE_SET_D3D_MANAGER: u32 = 2;
const MFT_MESSAGE_NOTIFY_BEGIN_STREAMING: u32 = 0x1000_0000;
const MFT_MESSAGE_NOTIFY_END_STREAMING: u32 = 0x1000_0001;
const MFT_ENUM_FLAG_HARDWARE: u32 = 0x4;
const MFT_ENUM_FLAG_SORTANDFILTER: u32 = 0x40;
const MFT_OUTPUT_STREAM_PROVIDES_SAMPLES: u32 = 0x0000_0100;
const MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES: u32 = 0x0000_0200;
const ME_TRANSFORM_NEED_INPUT: u32 = 601;
const ME_TRANSFORM_HAVE_OUTPUT: u32 = 602;
const MF_EVENT_FLAG_NO_WAIT: u32 = 0x1;
const MFSTARTUP_FULL: u32 = 0;
const MF_VERSION: u32 = 0x0002_0070;
const MF_VIDEO_INTERLACE_PROGRESSIVE: u32 = 2;

// D3D11
const D3D_DRIVER_TYPE_HARDWARE: u32 = 1;
const D3D11_SDK_VERSION: u32 = 7;
const D3D11_CREATE_DEVICE_VIDEO_SUPPORT: u32 = 0x800;
const D3D11_USAGE_STAGING: u32 = 3;
const D3D11_CPU_ACCESS_READ: u32 = 0x20000;
const D3D11_MAP_READ: u32 = 1;
const DXGI_FORMAT_NV12: u32 = 103;

#[link(name = "mfplat")]
extern "system" {
    fn MFStartup(version: u32, flags: u32) -> i32;
    fn MFTEnumEx(
        category: Guid,
        flags: u32,
        input_type: *const MftRegisterTypeInfo,
        output_type: *const MftRegisterTypeInfo,
        activates: *mut *mut *mut c_void,
        count: *mut u32,
    ) -> i32;
    fn MFCreateMediaType(media_type: *mut *mut c_void) -> i32;
    fn MFCreateSample(sample: *mut *mut c_void) -> i32;
    fn MFCreateMemoryBuffer(max_length: u32, buffer: *mut *mut c_void) -> i32;
    fn MFCreateDXGIDeviceManager(reset_token: *mut u32, manager: *mut *mut c_void) -> i32;
}

#[link(name = "ole32")]
extern "system" {
    fn CoTaskMemFree(ptr: *mut c_void);
    fn CoInitializeEx(reserved: *mut c_void, coinit: u32) -> i32;
}

#[link(name = "d3d11")]
extern "system" {
    fn D3D11CreateDevice(
        adapter: *mut c_void,
        driver_type: u32,
        software: *mut c_void,
        flags: u32,
        feature_levels: *const u32,
        feature_level_count: u32,
        sdk_version: u32,
        device: *mut *mut c_void,
        feature_level: *mut u32,
        immediate_context: *mut *mut c_void,
    ) -> i32;
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MftRegisterTypeInfo {
    major_type: Guid,
    subtype: Guid,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct MftOutputStreamInfo {
    flags: u32,
    size: u32,
    alignment: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MftOutputDataBuffer {
    stream_id: u32,
    sample: *mut c_void,
    buffer: *mut c_void,
    status: u32,
}

impl Default for MftOutputDataBuffer {
    fn default() -> Self {
        Self {
            stream_id: 0,
            sample: std::ptr::null_mut(),
            buffer: std::ptr::null_mut(),
            status: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct DxgiSampleDesc {
    count: u32,
    quality: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct D3d11Texture2DDesc {
    width: u32,
    height: u32,
    mip_levels: u32,
    array_size: u32,
    format: u32,
    sample_desc: DxgiSampleDesc,
    usage: u32,
    bind_flags: u32,
    cpu_access_flags: u32,
    misc_flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct D3d11MappedSubresource {
    data: *mut c_void,
    row_pitch: u32,
    depth_pitch: u32,
}

// ── COM 调用小工具 ──────────────────────────────────────────────────────────
//
// 手写 COM 最容易错的是"槽位号"，所以约定：**槽位是绝对平铺索引**（含 IUnknown
// 的 3 个），每个调用点都把索引写成常量，并在注释里带上成员名。

/// 取对象 vtable 里第 `slot` 个函数的指针。
unsafe fn slot<F: Copy>(obj: *mut c_void, index: usize) -> F {
    let vtable = *(obj as *const *const usize);
    let entry = *vtable.add(index);
    std::mem::transmute_copy(&entry)
}

/// `IUnknown::Release`（槽位 2）。
unsafe fn release(obj: *mut c_void) {
    if obj.is_null() {
        return;
    }
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = slot(obj, 2);
    f(obj);
}

/// `IUnknown::QueryInterface`（槽位 0）：成功返回新引用（调用方负责 release）。
unsafe fn query_interface(obj: *mut c_void, iid: &Guid) -> Option<*mut c_void> {
    if obj.is_null() {
        return None;
    }
    let f: unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32 =
        slot(obj, 0);
    let mut out: *mut c_void = std::ptr::null_mut();
    if f(obj, iid, &mut out) == S_OK && !out.is_null() {
        Some(out)
    } else {
        None
    }
}

/// `IMFAttributes::SetUINT32`（槽位 21）/ `SetUINT64`（22）/ `SetGUID`（24）/
/// `SetBlob`（26）/ `GetUINT32`（7）。
unsafe fn attr_set_u32(attrs: *mut c_void, key: &Guid, value: u32) -> i32 {
    let f: unsafe extern "system" fn(*mut c_void, *const Guid, u32) -> i32 = slot(attrs, 21);
    f(attrs, key, value)
}

unsafe fn attr_set_u64(attrs: *mut c_void, key: &Guid, value: u64) -> i32 {
    let f: unsafe extern "system" fn(*mut c_void, *const Guid, u64) -> i32 = slot(attrs, 22);
    f(attrs, key, value)
}

unsafe fn attr_set_guid(attrs: *mut c_void, key: &Guid, value: &Guid) -> i32 {
    let f: unsafe extern "system" fn(*mut c_void, *const Guid, *const Guid) -> i32 = slot(attrs, 24);
    f(attrs, key, value)
}

unsafe fn attr_set_blob(attrs: *mut c_void, key: &Guid, data: &[u8]) -> i32 {
    let f: unsafe extern "system" fn(*mut c_void, *const Guid, *const u8, u32) -> i32 =
        slot(attrs, 26);
    f(attrs, key, data.as_ptr(), data.len() as u32)
}

unsafe fn attr_get_u32(attrs: *mut c_void, key: &Guid) -> Option<u32> {
    let f: unsafe extern "system" fn(*mut c_void, *const Guid, *mut u32) -> i32 = slot(attrs, 7);
    let mut value = 0u32;
    if f(attrs, key, &mut value) == S_OK {
        Some(value)
    } else {
        None
    }
}

/// 把宽高打包成 MF 的 `MF_MT_FRAME_SIZE`（高 32 位是宽）。
fn pack_frame_size(width: u32, height: u32) -> u64 {
    ((width as u64) << 32) | height as u64
}

/// 每个用到的线程都要初始化过 COM 才碰 MF 对象（MTA 就够，硬件 MFT 都在 MTA 上）。
fn ensure_com() {
    use std::cell::Cell;
    thread_local! {
        static DONE: Cell<bool> = const { Cell::new(false) };
    }
    DONE.with(|done| {
        if done.get() {
            return;
        }
        unsafe {
            // RPC_E_CHANGED_MODE（已经以别的套间模式初始化过）也当成功：能用就行
            CoInitializeEx(std::ptr::null_mut(), 0 /* COINIT_MULTITHREADED */);
        }
        done.set(true);
    });
}

/// MF 平台初始化（进程内一次）。初始化失败就说明这台机器没有 Media Foundation
/// ——那是"没有硬解"最干脆的答案。
fn ensure_mf() -> bool {
    static STARTED: OnceLock<bool> = OnceLock::new();
    *STARTED.get_or_init(|| {
        ensure_com();
        unsafe { MFStartup(MF_VERSION, MFSTARTUP_FULL) == S_OK }
    })
}

/// 枚举**硬件** H.264 解码器。返回可用的 activate 数量（0 = 没有硬解）。
fn enumerate_hardware_decoders() -> Vec<*mut c_void> {
    let input = MftRegisterTypeInfo {
        major_type: MF_MEDIA_TYPE_VIDEO,
        subtype: MF_VIDEO_FORMAT_H264,
    };
    let output = MftRegisterTypeInfo {
        major_type: MF_MEDIA_TYPE_VIDEO,
        subtype: MF_VIDEO_FORMAT_NV12,
    };
    let mut activates: *mut *mut c_void = std::ptr::null_mut();
    let mut count = 0u32;
    let hr = unsafe {
        MFTEnumEx(
            MFT_CATEGORY_VIDEO_DECODER,
            MFT_ENUM_FLAG_HARDWARE | MFT_ENUM_FLAG_SORTANDFILTER,
            &input,
            &output,
            &mut activates,
            &mut count,
        )
    };
    if hr != S_OK || activates.is_null() || count == 0 {
        if !activates.is_null() {
            unsafe { CoTaskMemFree(activates as *mut c_void) };
        }
        return Vec::new();
    }
    let mut out = Vec::with_capacity(count as usize);
    for i in 0..count as usize {
        // 数组是 CoTaskMemAlloc 出来的；元素本身是 COM 引用，取走后要自己释放
        let item = unsafe { *activates.add(i) };
        if !item.is_null() {
            out.push(item);
        }
    }
    unsafe { CoTaskMemFree(activates as *mut c_void) };
    out
}

fn probe() -> bool {
    static PROBE: OnceLock<bool> = OnceLock::new();
    *PROBE.get_or_init(|| {
        if !ensure_mf() {
            tracing::info!("Media Foundation 起不来（MFStartup 失败）");
            return false;
        }
        let found = enumerate_hardware_decoders();
        let available = !found.is_empty();
        for a in &found {
            unsafe { release(*a) };
        }
        if available {
            tracing::info!("Media Foundation 硬件 H.264 解码器：{} 个", found.len());
        } else {
            tracing::info!("Media Foundation 里没有硬件 H.264 解码器");
        }
        available
    })
}

/// 这台机器上有没有 MF 硬件硬解（这个后端只接了 H.264）。
pub fn hardware_decode_available(codec: &CodecId) -> bool {
    match codec {
        CodecId::H264 => probe(),
        _ => false,
    }
}

// ── 解码器 ───────────────────────────────────────────────────────────────────

/// D3D11 那套东西（建不起来就是 `None`，解码退回系统内存输出）。
struct D3d {
    device: *mut c_void,
    context: *mut c_void,
    manager: *mut c_void,
    staging: *mut c_void,
    staging_size: (u32, u32),
}

impl D3d {
    fn create() -> Option<D3d> {
        unsafe {
            let levels = [0xb000u32, 0xa100, 0xa000, 0x9300, 0x9100];
            let mut device: *mut c_void = std::ptr::null_mut();
            let mut context: *mut c_void = std::ptr::null_mut();
            let hr = D3D11CreateDevice(
                std::ptr::null_mut(),
                D3D_DRIVER_TYPE_HARDWARE,
                std::ptr::null_mut(),
                D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
                levels.as_ptr(),
                levels.len() as u32,
                D3D11_SDK_VERSION,
                &mut device,
                std::ptr::null_mut(),
                &mut context,
            );
            if hr != S_OK || device.is_null() || context.is_null() {
                tracing::info!("D3D11 建不起来（{hr:#x}），解码退回系统内存输出");
                return None;
            }
            // 硬件 MFT 会在自己的线程上用这个设备 ⇒ 必须开多线程保护，
            // 否则就是"两个线程同时用立即上下文"的未定义行为。
            if let Some(mt) = query_interface(device, &IID_ID3D10_MULTITHREAD) {
                let f: unsafe extern "system" fn(*mut c_void, i32) -> i32 = slot(mt, 5);
                f(mt, 1); // BOOL TRUE
                release(mt);
            }
            let mut manager: *mut c_void = std::ptr::null_mut();
            let mut token = 0u32;
            let hr = MFCreateDXGIDeviceManager(&mut token, &mut manager);
            if hr != S_OK || manager.is_null() {
                tracing::info!("MFCreateDXGIDeviceManager 失败（{hr:#x}），退回系统内存输出");
                release(device);
                release(context);
                return None;
            }
            // ResetDevice 的参数是 **reset token**（不是设备管理器自己的 token，
            // 顺序也不能反）—— va.h 那边的注释提醒过，这里是 MF 的老坑。
            let f: unsafe extern "system" fn(*mut c_void, *mut c_void, u32) -> i32 =
                slot(manager, 7);
            let hr = f(manager, device, token);
            if hr != S_OK {
                tracing::info!("DXGI 设备管理器不认这个 D3D11 设备（{hr:#x}），退回系统内存输出");
                release(manager);
                release(device);
                release(context);
                return None;
            }
            Some(D3d {
                device,
                context,
                manager,
                staging: std::ptr::null_mut(),
                staging_size: (0, 0),
            })
        }
    }

    /// 把解码器给的纹理拷到 staging 纹理上再读回来（GPU 内存 → 系统内存的**唯一**通路）。
    unsafe fn copy_texture(&mut self, texture: *mut c_void) -> Result<(u32, u32, Vec<u8>, usize)> {
        let get_desc: unsafe extern "system" fn(*mut c_void, *mut D3d11Texture2DDesc) =
            slot(texture, 10);
        let mut desc = D3d11Texture2DDesc::default();
        get_desc(texture, &mut desc);
        if desc.format != DXGI_FORMAT_NV12 {
            return Err(MediaError::Unsupported(format!(
                "解码器给的纹理不是 NV12（DXGI_FORMAT {}）",
                desc.format
            )));
        }
        if self.staging.is_null() || self.staging_size != (desc.width, desc.height) {
            if !self.staging.is_null() {
                release(self.staging);
                self.staging = std::ptr::null_mut();
            }
            let staging_desc = D3d11Texture2DDesc {
                width: desc.width,
                height: desc.height,
                mip_levels: 1,
                array_size: 1,
                format: DXGI_FORMAT_NV12,
                sample_desc: DxgiSampleDesc::default(),
                usage: D3D11_USAGE_STAGING,
                bind_flags: 0,
                cpu_access_flags: D3D11_CPU_ACCESS_READ,
                misc_flags: 0,
            };
            let create: unsafe extern "system" fn(
                *mut c_void,
                *const D3d11Texture2DDesc,
                *const c_void,
                *mut *mut c_void,
            ) -> i32 = slot(self.device, 5);
            let mut staging: *mut c_void = std::ptr::null_mut();
            let hr = create(
                self.device,
                &staging_desc,
                std::ptr::null(),
                &mut staging,
            );
            if hr != S_OK || staging.is_null() {
                return Err(MediaError::Io(format!("建 staging 纹理失败（{hr:#x}）")));
            }
            self.staging = staging;
            self.staging_size = (desc.width, desc.height);
        }
        let copy: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) =
            slot(self.context, 47);
        copy(self.context, self.staging, texture);
        let map: unsafe extern "system" fn(
            *mut c_void,
            *mut c_void,
            u32,
            u32,
            u32,
            *mut D3d11MappedSubresource,
        ) -> i32 = slot(self.context, 14);
        let mut mapped = D3d11MappedSubresource::default();
        let hr = map(
            self.context,
            self.staging,
            0,
            D3D11_MAP_READ,
            0,
            &mut mapped,
        );
        if hr != S_OK || mapped.data.is_null() {
            return Err(MediaError::Io(format!("映射 staging 纹理失败（{hr:#x}）")));
        }
        let width = desc.width as usize;
        let height = desc.height as usize;
        let row_pitch = mapped.row_pitch as usize;
        let depth_pitch = mapped.depth_pitch as usize;
        let mut out = vec![0u8; VideoFrame::expected_len(desc.width, desc.height)];
        let src = mapped.data as *const u8;
        for row in 0..height {
            std::ptr::copy_nonoverlapping(src.add(row * row_pitch), out.as_mut_ptr().add(row * width), width);
        }
        // NV12 的两个平面在 D3D11 的映射里是"两个深度切片"：色度平面从 DepthPitch 开始
        let chroma = src.add(depth_pitch.max(row_pitch * height));
        for row in 0..height / 2 {
            std::ptr::copy_nonoverlapping(
                chroma.add(row * row_pitch),
                out.as_mut_ptr().add(width * height + row * width),
                width,
            );
        }
        let unmap: unsafe extern "system" fn(*mut c_void, *mut c_void, u32) = slot(self.context, 15);
        unmap(self.context, self.staging, 0);
        Ok((desc.width, desc.height, out, row_pitch))
    }
}

impl Drop for D3d {
    fn drop(&mut self) {
        unsafe {
            if !self.staging.is_null() {
                release(self.staging);
            }
            // 顺序：先放管理器（它持着设备的引用），再放上下文与设备。
            // 不在这里给管理器 ResetDevice(null)：那会把设备从管理器上摘下来，
            // 而此刻解码器已经释放过，摘不摘都一样，少一个 ABI 假设少一个坑。
            release(self.manager);
            release(self.context);
            release(self.device);
        }
    }
}

pub struct MfDecoder {
    transform: *mut c_void,
    events: *mut c_void,
    d3d: Option<D3d>,
    /// 输出类型给的显示尺寸（解码器自己按裁剪算好的）
    width: u32,
    height: u32,
    /// 输出缓冲的步幅（系统内存输出时用来去 stride）
    stride: Option<u32>,
    provides_samples: bool,
    output_buffer_size: u32,
    /// 下一个输出帧的兜底 pts（解码器不给时间戳时用）
    fallback_pts_us: i64,
    /// 已经冲过尾（`flush` 之间不重复冲）
    flushing: bool,
    config: Config,
}

impl MfDecoder {
    pub fn open(avcc: &[u8]) -> Result<Self> {
        if !ensure_mf() {
            return Err(MediaError::Unsupported(
                "这台机器起不了 Media Foundation（MFStartup 失败）".into(),
            ));
        }
        // 参数集在这里解析：不支持的构造（10 位/隔行/FMO…）在这就说清楚，
        // 别等 MFT 回一个看不懂的 HRESULT
        let config = Config::parse(avcc)?;
        let activates = enumerate_hardware_decoders();
        if activates.is_empty() {
            return Err(MediaError::Unsupported(
                "Media Foundation 里没有硬件 H.264 解码器（这台机器只有软件解码器）".into(),
            ));
        }

        let mut last_error = String::new();
        for activate in activates {
            match unsafe { Self::open_with(activate, config.clone()) } {
                Ok(dec) => {
                    unsafe { release(activate) };
                    return Ok(dec);
                }
                Err(e) => {
                    last_error = format!("{e}");
                    unsafe { release(activate) };
                }
            }
        }
        Err(MediaError::Unsupported(format!(
            "硬件 H.264 解码器都起不来：{last_error}"
        )))
    }

    unsafe fn open_with(activate: *mut c_void, config: Config) -> Result<Self> {
        // IMFActivate::ActivateObject（槽位 33）——要 IMFTransform 的 IID
        let activate_obj: unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32 =
            slot(activate, 33);
        let mut transform: *mut c_void = std::ptr::null_mut();
        let hr = activate_obj(activate, &IID_IMF_TRANSFORM, &mut transform);
        if hr != S_OK || transform.is_null() {
            return Err(MediaError::Unsupported(format!(
                "ActivateObject 失败（{hr:#x}）"
            )));
        }

        // 异步 MFT 必须先解锁，否则 SetInputType 会返回 MF_E_TRANSFORM_ASYNC_LOCKED
        let mut attrs: *mut c_void = std::ptr::null_mut();
        let get_attrs: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32 =
            slot(transform, 8);
        if get_attrs(transform, &mut attrs) == S_OK && !attrs.is_null() {
            let is_async = attr_get_u32(attrs, &MF_TRANSFORM_ASYNC).unwrap_or(0) != 0;
            if is_async {
                attr_set_u32(attrs, &MF_TRANSFORM_ASYNC_UNLOCK, 1);
            }
            release(attrs);
        }

        let mut d3d = D3d::create();

        // D3D 设备管理器必须**在设媒体类型之前**给过去：硬件 MFT 要靠它决定输出
        // 纹理怎么分配（设完类型再给，多数驱动会拒绝或悄悄退回系统内存拷贝）。
        if let Some(dev) = d3d.as_ref() {
            let message: unsafe extern "system" fn(*mut c_void, u32, usize) -> i32 =
                slot(transform, 23);
            let hr = message(transform, MFT_MESSAGE_SET_D3D_MANAGER, dev.manager as usize);
            if hr != S_OK {
                tracing::info!("解码器不接受 D3D 设备管理器（{hr:#x}），改用系统内存输出");
                d3d = None;
            }
        }

        // ── 输入类型：H.264（Annex-B 采样）+ avcC 当序列头 + 逐行 ──
        let mut input_type: *mut c_void = std::ptr::null_mut();
        if MFCreateMediaType(&mut input_type) != S_OK || input_type.is_null() {
            release(transform);
            return Err(MediaError::Io("MFCreateMediaType 失败".into()));
        }
        let mut hr = attr_set_guid(input_type, &MF_MT_MAJOR_TYPE, &MF_MEDIA_TYPE_VIDEO);
        if hr == S_OK {
            hr = attr_set_guid(input_type, &MF_MT_SUBTYPE, &MF_VIDEO_FORMAT_H264);
        }
        if hr == S_OK {
            hr = attr_set_u64(
                input_type,
                &MF_MT_FRAME_SIZE,
                pack_frame_size(config.sps.width(), config.sps.height()),
            );
        }
        if hr == S_OK {
            hr = attr_set_u32(input_type, &MF_MT_INTERLACE_MODE, MF_VIDEO_INTERLACE_PROGRESSIVE);
        }
        if hr == S_OK {
            // 参数集也挂在媒体类型上（MF 的两条取参数集通路：这里 + 码流里）
            hr = attr_set_blob(input_type, &MF_MT_MPEG_SEQUENCE_HEADER, &config.avcc);
        }
        let set_input: unsafe extern "system" fn(*mut c_void, *mut c_void, u32) -> i32 =
            slot(transform, 15);
        if hr == S_OK {
            hr = set_input(transform, input_type, 0);
        }
        release(input_type);
        if hr != S_OK {
            let hint = if hr as u32 == MF_E_TRANSFORM_ASYNC_LOCKED {
                "（硬件 MFT 是异步的，必须先解锁 MF_TRANSFORM_ASYNC_UNLOCK）"
            } else {
                ""
            };
            release(transform);
            return Err(MediaError::Unsupported(format!(
                "解码器不接受这个 H.264 输入类型（{hr:#x}）{hint}"
            )));
        }

        let mut dec = Self {
            transform,
            events: std::ptr::null_mut(),
            d3d,
            width: config.sps.width(),
            height: config.sps.height(),
            stride: None,
            provides_samples: true,
            output_buffer_size: 0,
            fallback_pts_us: 0,
            flushing: false,
            config,
        };
        dec.events = query_interface(transform, &IID_IMF_MEDIA_EVENT_GENERATOR).unwrap_or(std::ptr::null_mut());
        dec.negotiate_output()?;

        {
            let message: unsafe extern "system" fn(*mut c_void, u32, usize) -> i32 =
                slot(transform, 23);
            let _ = message(transform, MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0);
        }
        let info: unsafe extern "system" fn(*mut c_void, u32, *mut MftOutputStreamInfo) -> i32 =
            slot(transform, 7);
        let mut stream_info = MftOutputStreamInfo::default();
        if info(transform, 0, &mut stream_info) == S_OK {
            dec.provides_samples = stream_info.flags
                & (MFT_OUTPUT_STREAM_PROVIDES_SAMPLES | MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES)
                != 0;
            dec.output_buffer_size = stream_info.size;
        }
        Ok(dec)
    }

    /// 协商输出类型（第一次 + 每次 `MF_E_TRANSFORM_STREAM_CHANGE` 之后）。
    ///
    /// 为什么要"每次"：解码器在没看到真实码流前给不出确定的类型，第一个采样
    /// 之后它会用 `MF_E_TRANSFORM_STREAM_CHANGE` 要求重设 —— 不重设就卡死。
    fn negotiate_output(&mut self) -> Result<()> {
        unsafe {
            let available: unsafe extern "system" fn(*mut c_void, u32, u32, *mut *mut c_void) -> i32 =
                slot(self.transform, 14);
            let set_output: unsafe extern "system" fn(*mut c_void, *mut c_void, u32) -> i32 =
                slot(self.transform, 16);
            let mut index = 0u32;
            loop {
                let mut media_type: *mut c_void = std::ptr::null_mut();
                let hr = available(self.transform, 0, index, &mut media_type);
                if hr as u32 == MF_E_NO_MORE_TYPES {
                    break;
                }
                if hr != S_OK || media_type.is_null() {
                    break;
                }
                let subtype = {
                    let get: unsafe extern "system" fn(*mut c_void, *const Guid, *mut Guid) -> i32 =
                        slot(media_type, 10); // IMFAttributes::GetGUID
                    let mut sub = Guid::default();
                    if get(media_type, &MF_MT_SUBTYPE, &mut sub) == S_OK {
                        Some(sub)
                    } else {
                        None
                    }
                };
                if subtype == Some(MF_VIDEO_FORMAT_NV12) && set_output(self.transform, media_type, 0) == S_OK
                {
                    // 输出尺寸以解码器说的为准（它按裁剪算好了显示尺寸）
                    let size = {
                        let get: unsafe extern "system" fn(*mut c_void, *const Guid, *mut u64) -> i32 =
                            slot(media_type, 8); // IMFAttributes::GetUINT64
                        let mut v = 0u64;
                        if get(media_type, &MF_MT_FRAME_SIZE, &mut v) == S_OK {
                            ((v >> 32) as u32, (v & 0xFFFF_FFFF) as u32)
                        } else {
                            (self.width, self.height)
                        }
                    };
                    self.width = size.0;
                    self.height = size.1;
                    self.stride = attr_get_u32(media_type, &MF_MT_DEFAULT_STRIDE)
                        .map(|s| s & 0x7FFF_FFFF);
                    release(media_type);
                    return Ok(());
                }
                release(media_type);
                index += 1;
            }
            Err(MediaError::Unsupported(
                "解码器没有 NV12 输出类型（引擎对外只有 NV12）".into(),
            ))
        }
    }

    /// 把输出抽干（`ProcessOutput` 循环到"要更多输入"为止）。
    fn drain_outputs(&mut self, out: &mut Vec<VideoFrame>) -> Result<()> {
        unsafe {
            let process_output: unsafe extern "system" fn(
                *mut c_void,
                u32,
                u32,
                *mut MftOutputDataBuffer,
                *mut u32,
            ) -> i32 = slot(self.transform, 25);
            loop {
                let mut data = MftOutputDataBuffer::default();
                let mut our_sample: *mut c_void = std::ptr::null_mut();
                let mut our_buffer: *mut c_void = std::ptr::null_mut();
                if !self.provides_samples {
                    // 解码器不提供样本：我们自己造一个（系统内存缓冲，尺寸用它给的）
                    if MFCreateSample(&mut our_sample) != S_OK || our_sample.is_null() {
                        return Err(MediaError::Io("MFCreateSample 失败".into()));
                    }
                    let size = self.output_buffer_size.max(self.width * self.height * 3 / 2);
                    if MFCreateMemoryBuffer(size, &mut our_buffer) != S_OK || our_buffer.is_null() {
                        release(our_sample);
                        return Err(MediaError::Io("MFCreateMemoryBuffer 失败".into()));
                    }
                    let add: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32 =
                        slot(our_sample, 42);
                    add(our_sample, our_buffer);
                    data.sample = our_sample;
                }
                let mut status = 0u32;
                let hr = process_output(self.transform, 0, 1, &mut data, &mut status);
                if hr == S_OK {
                    // 有些实现把画面放在 pBuffer 上（而不是 pSample）—— 两条都认
                    let produced = if !data.sample.is_null() {
                        Some(self.sample_to_frame(data.sample))
                    } else if !data.buffer.is_null() {
                        Some(self.copy_system_buffer(data.buffer, self.fallback_pts_us + 1))
                    } else {
                        None
                    };
                    if !data.sample.is_null() {
                        release(data.sample);
                    }
                    if !data.buffer.is_null() {
                        release(data.buffer);
                    }
                    if !our_sample.is_null() {
                        release(our_sample);
                    }
                    if !our_buffer.is_null() {
                        release(our_buffer);
                    }
                    match produced {
                        Some(Ok(frame)) => out.push(frame),
                        Some(Err(e)) => return Err(e),
                        None => {}
                    }
                    continue;
                }
                if !our_sample.is_null() {
                    release(our_sample);
                }
                if !our_buffer.is_null() {
                    release(our_buffer);
                }
                match hr as u32 {
                    MF_E_TRANSFORM_NEED_MORE_INPUT => return Ok(()),
                    MF_E_TRANSFORM_STREAM_CHANGE => {
                        // 码流要求重设输出类型（分辨率/格式变了）
                        self.negotiate_output()?;
                        continue;
                    }
                    other => {
                        return Err(MediaError::Bitstream(format!(
                            "ProcessOutput 失败（{other:#x}）"
                        )))
                    }
                }
            }
        }
    }

    /// 抽事件队列（**非阻塞**：`MF_EVENT_FLAG_NO_WAIT`）。
    ///
    /// 硬件 MFT 靠事件说"该喂了/有输出了"；不看事件在某些驱动上会一直
    /// `MF_E_TRANSFORM_NEED_MORE_INPUT`（明明有输入却不干活）。
    fn pump_events(&mut self, out: &mut Vec<VideoFrame>) -> Result<()> {
        if self.events.is_null() {
            return Ok(());
        }
        loop {
            let mut event: *mut c_void = std::ptr::null_mut();
            let get_event: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> i32 =
                unsafe { slot(self.events, 3) };
            let hr = unsafe { get_event(self.events, MF_EVENT_FLAG_NO_WAIT, &mut event) };
            if hr != S_OK || event.is_null() {
                return Ok(());
            }
            let mut kind = 0u32;
            let get_type: unsafe extern "system" fn(*mut c_void, *mut u32) -> i32 =
                unsafe { slot(event, 33) };
            let mut event_status: i32 = 0;
            let get_status: unsafe extern "system" fn(*mut c_void, *mut i32) -> i32 =
                unsafe { slot(event, 35) };
            unsafe {
                get_type(event, &mut kind);
                get_status(event, &mut event_status);
            }
            unsafe { release(event) };
            if event_status != S_OK {
                return Err(MediaError::Bitstream(format!(
                    "解码器上报事件失败（事件 {kind}，状态 {event_status:#x}）"
                )));
            }
            match kind {
                ME_TRANSFORM_HAVE_OUTPUT => self.drain_outputs(out)?,
                ME_TRANSFORM_NEED_INPUT => {}
                _ => {}
            }
        }
    }

    /// 一个输出样本 → 紧凑 NV12 帧。
    fn sample_to_frame(&mut self, sample: *mut c_void) -> Result<VideoFrame> {
        unsafe {
            let mut time: i64 = 0;
            let get_time: unsafe extern "system" fn(*mut c_void, *mut i64) -> i32 =
                slot(sample, 35);
            let pts_us = if get_time(sample, &mut time) == S_OK {
                time / 10 // MF 的时间戳单位是 100ns
            } else {
                self.fallback_pts_us + 1
            };
            self.fallback_pts_us = pts_us;

            let mut buffer: *mut c_void = std::ptr::null_mut();
            let get_buffer: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> i32 =
                slot(sample, 40);
            let hr = get_buffer(sample, 0, &mut buffer);
            if hr != S_OK || buffer.is_null() {
                return Err(MediaError::Io(format!("取输出缓冲失败（{hr:#x}）")));
            }

            // 缓冲是不是 GPU 纹理：是就走 staging 拷贝那条路，不是就直接 Lock
            let dxgi = query_interface(buffer, &IID_IMF_DXGI_BUFFER);
            let frame = if let Some(dxgi) = dxgi {
                if self.d3d.is_none() {
                    release(dxgi);
                    release(buffer);
                    return Err(MediaError::Unsupported(
                        "解码器给的是 GPU（DXGI）缓冲，但这台机器没建起 D3D11 —— 解不出来".into(),
                    ));
                }
                let get_resource: unsafe extern "system" fn(
                    *mut c_void,
                    *const Guid,
                    *mut *mut c_void,
                ) -> i32 = slot(dxgi, 3);
                let mut texture: *mut c_void = std::ptr::null_mut();
                let hr = get_resource(dxgi, &IID_ID3D11_TEXTURE2D, &mut texture);
                release(dxgi);
                if hr != S_OK || texture.is_null() {
                    release(buffer);
                    return Err(MediaError::Io(format!(
                        "从 DXGI 缓冲取纹理失败（{hr:#x}）"
                    )));
                }
                let copied = self
                    .d3d
                    .as_mut()
                    .expect("上面已经确认 d3d 存在")
                    .copy_texture(texture);
                release(texture);
                copied.map(|(w, h, nv12, _)| VideoFrame {
                    width: w,
                    height: h,
                    pts_us,
                    nv12,
                })
            } else {
                self.copy_system_buffer(buffer, pts_us)
            };
            release(buffer);
            frame
        }
    }

    /// 系统内存缓冲 → 紧凑 NV12（按输出类型的 stride 去 stride）。
    unsafe fn copy_system_buffer(&self, buffer: *mut c_void, pts_us: i64) -> Result<VideoFrame> {
        let lock: unsafe extern "system" fn(
            *mut c_void,
            *mut *mut u8,
            *mut u32,
            *mut u32,
        ) -> i32 = slot(buffer, 3);
        let unlock: unsafe extern "system" fn(*mut c_void) -> i32 = slot(buffer, 4);
        let mut data: *mut u8 = std::ptr::null_mut();
        let mut max_len = 0u32;
        let mut cur_len = 0u32;
        let hr = lock(buffer, &mut data, &mut max_len, &mut cur_len);
        if hr != S_OK || data.is_null() {
            return Err(MediaError::Io(format!("锁定输出缓冲失败（{hr:#x}）")));
        }
        let width = self.width as usize;
        let height = self.height as usize;
        let expected = VideoFrame::expected_len(self.width, self.height);
        // stride：优先用输出类型里声明的；没有就按缓冲长度反推（NV12 总长 = stride*h*3/2）
        let stride = self
            .stride
            .map(|s| s as usize)
            .filter(|s| *s >= width && s * height * 3 / 2 <= cur_len as usize)
            .unwrap_or_else(|| {
                if cur_len as usize == expected {
                    width
                } else {
                    (cur_len as usize) * 2 / (3 * height)
                }
            });
        if stride < width || stride * height * 3 / 2 > max_len as usize {
            unlock(buffer);
            return Err(MediaError::Bitstream(format!(
                "输出缓冲的形状不对（stride {stride}、{width}x{height}、{} 字节）",
                cur_len
            )));
        }
        let mut out = vec![0u8; expected];
        for row in 0..height {
            std::ptr::copy_nonoverlapping(data.add(row * stride), out.as_mut_ptr().add(row * width), width);
        }
        let chroma = data.add(stride * height);
        for row in 0..height / 2 {
            std::ptr::copy_nonoverlapping(
                chroma.add(row * stride),
                out.as_mut_ptr().add(width * height + row * width),
                width,
            );
        }
        unlock(buffer);
        Ok(VideoFrame {
            width: self.width,
            height: self.height,
            pts_us,
            nv12: out,
        })
    }

    /// 送一个采样进去（不满足时先抽输出再重试）。
    fn feed(&mut self, sample: *mut c_void, out: &mut Vec<VideoFrame>) -> Result<()> {
        let process_input: unsafe extern "system" fn(*mut c_void, u32, *mut c_void, u32) -> i32 =
            unsafe { slot(self.transform, 24) };
        for _ in 0..8 {
            let hr = unsafe { process_input(self.transform, 0, sample, 0) };
            if hr == S_OK {
                self.pump_events(out)?;
                self.drain_outputs(out)?;
                return Ok(());
            }
            if hr as u32 == MF_E_NOTACCEPTING {
                // 解码器还握着上一个采样：把它吐出来再喂
                self.pump_events(out)?;
                self.drain_outputs(out)?;
                continue;
            }
            return Err(MediaError::Bitstream(format!(
                "ProcessInput 失败（{hr:#x}）"
            )));
        }
        Err(MediaError::Bitstream(
            "解码器一直不接收输入（连续 MF_E_NOTACCEPTING）".into(),
        ))
    }
}

impl crate::VideoDecoder for MfDecoder {
    fn decode(&mut self, sample_avcc: &[u8], pts_us: i64) -> Result<Vec<VideoFrame>> {
        if sample_avcc.is_empty() {
            return Ok(Vec::new());
        }
        ensure_com();
        // MF 的解码器吃 Annex-B（带起始码），引擎内部统一是 AVCC
        let annexb = h264::avcc_to_annexb(sample_avcc, self.config.nal_length_size)?;
        let mut out = Vec::new();
        unsafe {
            let mut sample: *mut c_void = std::ptr::null_mut();
            if MFCreateSample(&mut sample) != S_OK || sample.is_null() {
                return Err(MediaError::Io("MFCreateSample 失败".into()));
            }
            let mut buffer: *mut c_void = std::ptr::null_mut();
            if MFCreateMemoryBuffer(annexb.len() as u32, &mut buffer) != S_OK || buffer.is_null() {
                release(sample);
                return Err(MediaError::Io("MFCreateMemoryBuffer 失败".into()));
            }
            // 采样字节拷进 MF 的内存缓冲
            let lock: unsafe extern "system" fn(*mut c_void, *mut *mut u8, *mut u32, *mut u32) -> i32 =
                slot(buffer, 3);
            let unlock: unsafe extern "system" fn(*mut c_void) -> i32 = slot(buffer, 4);
            let mut data: *mut u8 = std::ptr::null_mut();
            let mut max_len = 0u32;
            let mut cur_len = 0u32;
            let hr = lock(buffer, &mut data, &mut max_len, &mut cur_len);
            if hr != S_OK || data.is_null() {
                release(buffer);
                release(sample);
                return Err(MediaError::Io(format!("锁定输入缓冲失败（{hr:#x}）")));
            }
            std::ptr::copy_nonoverlapping(annexb.as_ptr(), data, annexb.len());
            unlock(buffer);
            let set_len: unsafe extern "system" fn(*mut c_void, u32) -> i32 = slot(buffer, 6);
            set_len(buffer, annexb.len() as u32);
            let add: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32 = slot(sample, 42);
            add(sample, buffer);
            release(buffer);
            let set_time: unsafe extern "system" fn(*mut c_void, i64) -> i32 = slot(sample, 36);
            set_time(sample, pts_us * 10); // µs → 100ns
            let set_duration: unsafe extern "system" fn(*mut c_void, i64) -> i32 = slot(sample, 38);
            set_duration(sample, 0);

            let result = self.feed(sample, &mut out);
            release(sample);
            result?;
        }
        self.flushing = false;
        Ok(out)
    }

    fn flush(&mut self) -> Result<Vec<VideoFrame>> {
        let mut out = Vec::new();
        if self.flushing {
            return Ok(out);
        }
        self.flushing = true;
        unsafe {
            let message: unsafe extern "system" fn(*mut c_void, u32, usize) -> i32 =
                slot(self.transform, 23);
            // DRAIN：让解码器把排队的帧吐出来（不支持就算了，别当错误）
            let _ = message(self.transform, MFT_MESSAGE_COMMAND_DRAIN, 0);
        }
        // DRAIN 是**异步**的（硬件 MFT 在自己的线程上收尾），所以要给它一点时间：
        // 每 2ms 抽一次事件与输出，连续两轮没有新帧就认为冲完了。最坏多等 20ms，
        // 而这是整条流收尾时的一次性代价。
        let mut last = 0usize;
        let mut idle = 0;
        for _ in 0..10 {
            std::thread::sleep(std::time::Duration::from_millis(2));
            if self.pump_events(&mut out).is_err() || self.drain_outputs(&mut out).is_err() {
                break;
            }
            if out.len() == last {
                idle += 1;
                if idle >= 2 {
                    break;
                }
            } else {
                idle = 0;
                last = out.len();
            }
        }
        unsafe {
            let message: unsafe extern "system" fn(*mut c_void, u32, usize) -> i32 =
                slot(self.transform, 23);
            let _ = message(self.transform, MFT_MESSAGE_COMMAND_FLUSH, 0);
            let _ = message(self.transform, MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0);
        }
        Ok(out)
    }

    fn backend(&self) -> &'static str {
        "mediafoundation"
    }
}

// 安全性：COM 对象的线程亲和性由我们保证 —— 一个解码器只在一个线程上用
//（`decode` 顺序调用），跨线程只是把所有权搬过去。硬件 MFT 本身是自由线程的，
// 但"同时从两个线程调同一个 transform"不是我们的用法。
unsafe impl Send for MfDecoder {}

impl Drop for MfDecoder {
    fn drop(&mut self) {
        unsafe {
            if !self.events.is_null() {
                release(self.events);
            }
            if !self.transform.is_null() {
                let message: unsafe extern "system" fn(*mut c_void, u32, usize) -> i32 =
                    slot(self.transform, 23);
                let _ = message(self.transform, MFT_MESSAGE_NOTIFY_END_STREAMING, 0);
                release(self.transform);
            }
            // d3d 的 Drop 自己收拾
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VideoDecoder;

    /// 打包宽高的顺序（低 32 位是高、高 32 位是宽 —— 弄反了就是"1680x1080 的怪尺寸"）。
    #[test]
    fn frame_size_packing_matches_mf_layout() {
        let packed = pack_frame_size(1920, 1080);
        assert_eq!(packed >> 32, 1920);
        assert_eq!(packed & 0xFFFF_FFFF, 1080);
    }

    /// GUID 值必须与 SDK 一致 —— 写错一个数字的症状是"什么都找不到"。
    #[test]
    fn guid_values_match_the_sdk() {
        assert_eq!(MF_VIDEO_FORMAT_H264.data1, 0x3436_3248); // 'H264'
        assert_eq!(MF_MEDIA_TYPE_VIDEO.data1, 0x7364_6976); // 'vids'
        assert_eq!(MF_VIDEO_FORMAT_NV12.data1, 0x3231_564e); // 'NV12'
        assert_eq!(IID_ID3D11_TEXTURE2D.data4[7], 0x9c);
    }

    /// 探测在没有 GPU / 没有硬件解码器的机器（CI runner）上要给出确定的"没有"。
    #[test]
    fn probe_is_safe_on_a_machine_without_hardware() {
        let available = hardware_decode_available(&CodecId::H264);
        println!("Media Foundation 探测：h264={available}");
        assert!(!hardware_decode_available(&CodecId::Vp9));
        assert!(!hardware_decode_available(&CodecId::Av1));
    }

    /// 探测不到时，开解码器必须是**一句能读懂的话**。
    #[test]
    fn open_without_hardware_reports_a_readable_reason() {
        let e = match MfDecoder::open(&crate::h264::tests_support::mini_avcc()) {
            Ok(d) => {
                assert_eq!(d.backend(), "mediafoundation");
                assert!(hardware_decode_available(&CodecId::H264));
                return;
            }
            Err(e) => e,
        };
        let text = format!("{e}");
        assert!(
            text.contains("Media Foundation") || text.contains("H.264"),
            "失败文案要写清缺的是哪一样：{text}"
        );
    }
}
