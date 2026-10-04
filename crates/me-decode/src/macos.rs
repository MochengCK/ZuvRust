//! macOS 硬解：**VideoToolbox**（H.264 与 AV1）。
//!
//! 为什么用它：它是操作系统自带的解码能力（就像系统的音频栈一样），不是
//! FFmpeg/GStreamer 那类第三方媒体引擎；硬解在功耗与 4K 播放上也是唯一实用的选择。
//!
//! ## AV1 的现实（实测，别指望它在 Intel 机上能用）
//!
//! Apple 只从 **M3 及以后**提供 AV1 解码器；Intel 机的 VideoToolbox 连软件兜底
//! 都没有 —— 实测（2026-09-29，MacBookAir9,1 / i5-1030NG7）：
//!
//! ```text
//! VTIsHardwareDecodeSupported('avc1') = true
//! VTIsHardwareDecodeSupported('hvc1') = true
//! VTIsHardwareDecodeSupported('av01') = false
//! VTDecompressionSessionCreate(av01 1920x1080) = -12906   // kVTCouldNotFindVideoDecoderErr
//! ```
//!
//! 所以 AV1 这一档是"硬解优先、**自研软解兜底**"里最需要软解的那个编码：
//! 硬解起不来时 [`crate::open_video_decoder`] 会落到 [`crate::video::av1`]。
//!
//! ## AV1 的会话是**懒建**的
//!
//! `CMVideoFormatDescriptionCreate` 建视频格式描述时**必须给尺寸**，而 AV1 的
//! 尺寸只写在码流的序列头里（`av1C` 里没有）。而播放路径的调用顺序是
//! "先 `open_video_decoder`，再逐个采样喂进来" —— 开解码器时尺寸还不知道。
//! 所以 AV1 这一路把"建格式描述 + 建会话"推迟到**第一个采样进来时**：那时
//! 序列头已经在手里了（见 `video::av1::inspect_sample`）。H.264 不受影响
//! （`avcC` 里的 SPS 自带尺寸，仍然在 `open` 时一次建好）。
//!
//! FFI 是**手写声明**的，没有引第三方 crate：用到的符号就这些，而"零媒体依赖"
//! 是这个引擎的底线（多一个 crate 就多一份 ABI/供应链风险）。
//!
//! 线程模型：VideoToolbox 在**自己的线程**上回调（`decode` 期间与
//! `WaitForAsynchronousFrames` 之前都可能发生），所以共享状态用 `Mutex`；
//! 回调里只做"拷帧 + 记录"，不做任何可能阻塞的事。

use std::ffi::{c_void, CString};
use std::sync::Mutex;

use me_core::{CodecId, MediaError, Result};

use crate::{video::av1, VideoFrame};

// ── 类型别名（CoreFoundation 的引用都是不透明指针） ──────────────────────────

type OSStatus = i32;
type CFTypeRef = *const c_void;
type CFAllocatorRef = *const c_void;
type CFNumberRef = *const c_void;
type CFDictionaryRef = *const c_void;
type CFStringRef = *const c_void;
type CFIndex = isize;
type CMBlockBufferRef = *mut c_void;
type CMFormatDescriptionRef = *mut c_void;
type CMSampleBufferRef = *mut c_void;
type CVImageBufferRef = *mut c_void;
type CVPixelBufferRef = *mut c_void;
type VTDecompressionSessionRef = *mut c_void;
type VTDecodeInfoFlags = u32;
type VTDecodeFrameFlags = u32;

/// `kCVPixelBufferLock_ReadOnly`
const PIXEL_BUFFER_LOCK_READ_ONLY: u64 = 1;

/// `kCMBlockBufferAssureMemoryNowFlag`（让 Create 立刻分配内存块）
const BLOCK_BUFFER_ASSURE_MEMORY_NOW: u32 = 1 << 1;

/// `VTDecodeFrameFlags`：只冲刷、不产出帧（VideoToolbox 的"flush"惯用法）。
const DECODE_FLAG_DO_NOT_OUTPUT_FRAME: VTDecodeFrameFlags = 1 << 3;

/// `kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange`（'420v'，即 NV12）。
const PIXEL_FORMAT_420V: u32 = 0x3432_3076;

/// `CMVideoCodecType`：`'avc1'` / `'hvc1'` / `'av01'`（都是四字符码）。
const CODEC_TYPE_H264: u32 = 0x6176_6331;
const CODEC_TYPE_HEVC: u32 = 0x6876_6331;
const CODEC_TYPE_AV1: u32 = 0x6176_3031;

/// `kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms` = `'sdat'`。
///
/// AV1 的 `av1C` 没有专门的 `CMVideoFormatDescriptionCreateFrom…` 入口，
/// 只能走"通用格式描述 + 采样描述盒"这条路：扩展字典里放一个
/// `{"av1C": <CFData>}`（键名就是盒子名，所以那个字符串必须是 `"av1C"`）。
const CF_NUMBER_SINT32: i32 = 3;

/// `kCFStringEncodingUTF8`（建 `"av1C"` 这个键要用）。
const CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct CmTime {
    value: i64,
    timescale: i32,
    flags: u32,
    epoch: i64,
}

/// `kCMTimeFlags_Valid`
const CM_TIME_VALID: u32 = 1;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct CmSampleTimingInfo {
    duration: CmTime,
    presentation_time_stamp: CmTime,
    decode_time_stamp: CmTime,
}

type VtOutputCallback = extern "C" fn(
    refcon: *mut c_void,
    source_frame_refcon: *mut c_void,
    status: OSStatus,
    info_flags: VTDecodeInfoFlags,
    image_buffer: CVImageBufferRef,
    pts: CmTime,
    duration: CmTime,
);

#[repr(C)]
struct VtOutputCallbackRecord {
    callback: Option<VtOutputCallback>,
    refcon: *mut c_void,
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFRelease(cf: CFTypeRef);
    fn CFDictionaryCreate(
        allocator: CFAllocatorRef,
        keys: *const CFTypeRef,
        values: *const CFTypeRef,
        num_values: CFIndex,
        key_callbacks: *const c_void,
        value_callbacks: *const c_void,
    ) -> CFDictionaryRef;
    fn CFDictionaryCreateMutable(
        allocator: CFAllocatorRef,
        capacity: CFIndex,
        key_callbacks: *const c_void,
        value_callbacks: *const c_void,
    ) -> CFDictionaryRef;
    fn CFDictionarySetValue(dict: CFDictionaryRef, key: CFTypeRef, value: CFTypeRef);
    fn CFDataCreate(allocator: CFAllocatorRef, bytes: *const u8, length: CFIndex) -> CFTypeRef;
    fn CFStringCreateWithCString(allocator: CFAllocatorRef, c_str: *const u8, encoding: u32) -> CFStringRef;
    fn CFNumberCreate(allocator: CFAllocatorRef, kind: i32, value_ptr: *const c_void) -> CFNumberRef;
    static kCFTypeDictionaryKeyCallBacks: c_void;
    static kCFTypeDictionaryValueCallBacks: c_void;
}

#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    fn CVPixelBufferLockBaseAddress(buffer: CVPixelBufferRef, flags: u64) -> OSStatus;
    fn CVPixelBufferUnlockBaseAddress(buffer: CVPixelBufferRef, flags: u64) -> OSStatus;
    fn CVPixelBufferGetWidth(buffer: CVPixelBufferRef) -> usize;
    fn CVPixelBufferGetHeight(buffer: CVPixelBufferRef) -> usize;
    fn CVPixelBufferGetPlaneCount(buffer: CVPixelBufferRef) -> usize;
    fn CVPixelBufferGetBaseAddressOfPlane(buffer: CVPixelBufferRef, plane: usize) -> *mut u8;
    fn CVPixelBufferGetBytesPerRowOfPlane(buffer: CVPixelBufferRef, plane: usize) -> usize;
    fn CVPixelBufferGetPixelFormatType(buffer: CVPixelBufferRef) -> u32;
    static kCVPixelBufferPixelFormatTypeKey: CFStringRef;
    static kCVPixelBufferIOSurfacePropertiesKey: CFStringRef;
}

#[link(name = "CoreMedia", kind = "framework")]
extern "C" {
    fn CMVideoFormatDescriptionCreateFromH264ParameterSets(
        allocator: CFAllocatorRef,
        parameter_set_count: usize,
        parameter_sets: *const *const u8,
        parameter_set_sizes: *const usize,
        nal_unit_header_length: i32,
        format_description_out: *mut CMFormatDescriptionRef,
    ) -> OSStatus;
    /// 通用格式描述：换编码（AV1）时要靠它，因为 CoreMedia 没有
    /// "`CMVideoFormatDescriptionCreateFromAV1…`" 这样的入口。
    fn CMVideoFormatDescriptionCreate(
        allocator: CFAllocatorRef,
        codec_type: u32,
        width: i32,
        height: i32,
        extensions: CFDictionaryRef,
        format_description_out: *mut CMFormatDescriptionRef,
    ) -> OSStatus;
    fn CMBlockBufferCreateWithMemoryBlock(
        allocator: CFAllocatorRef,
        memory_block: *mut c_void,
        block_length: usize,
        block_allocator: CFAllocatorRef,
        custom_block_source: *const c_void,
        offset_to_data: usize,
        data_length: usize,
        flags: u32,
        block_buffer_out: *mut CMBlockBufferRef,
    ) -> OSStatus;
    fn CMBlockBufferReplaceDataBytes(
        source_bytes: *const c_void,
        destination: CMBlockBufferRef,
        offset: usize,
        length: usize,
    ) -> OSStatus;
    fn CMSampleBufferCreateReady(
        allocator: CFAllocatorRef,
        data_buffer: CMBlockBufferRef,
        format_description: CMFormatDescriptionRef,
        sample_count: isize,
        timing_entry_count: isize,
        timing_array: *const CmSampleTimingInfo,
        size_entry_count: isize,
        size_array: *const usize,
        sample_buffer_out: *mut CMSampleBufferRef,
    ) -> OSStatus;
    /// 采样描述盒的键：`{"av1C": <数据>}` 就是这么挂上去的。
    static kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms: CFStringRef;
}

#[link(name = "VideoToolbox", kind = "framework")]
extern "C" {
    /// 这个编码在这台机器上有没有**硬解**。
    ///
    /// 返回 `Boolean`（`unsigned char`）。注意它只说"有没有硬件解码器"，
    /// 不说"有没有能用的解码器"：AV1 在 Intel 机上两者都是否，而 HEVC 在
    /// 某些老机器上是"没有硬解但有软解"。
    fn VTIsHardwareDecodeSupported(codec_type: u32) -> u8;
    fn VTDecompressionSessionCreate(
        allocator: CFAllocatorRef,
        format_description: CMFormatDescriptionRef,
        decoder_specification: CFDictionaryRef,
        destination_image_buffer_attributes: CFDictionaryRef,
        output_callback: *const VtOutputCallbackRecord,
        session_out: *mut VTDecompressionSessionRef,
    ) -> OSStatus;
    fn VTDecompressionSessionDecodeFrame(
        session: VTDecompressionSessionRef,
        sample_buffer: CMSampleBufferRef,
        decode_flags: VTDecodeFrameFlags,
        frame_refcon: *mut c_void,
        info_flags_out: *mut VTDecodeInfoFlags,
    ) -> OSStatus;
    fn VTDecompressionSessionWaitForAsynchronousFrames(
        session: VTDecompressionSessionRef,
    ) -> OSStatus;
    fn VTDecompressionSessionInvalidate(session: VTDecompressionSessionRef);
}

/// 回调与解码线程共享的状态。
struct Shared {
    frames: Mutex<Vec<VideoFrame>>,
    error: Mutex<Option<String>>,
}

/// 硬解后端支持的两个编码。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VtCodec {
    H264,
    Av1,
}

impl VtCodec {
    fn name(self) -> &'static str {
        match self {
            VtCodec::H264 => "h264",
            VtCodec::Av1 => "av1",
        }
    }
}

/// VideoToolbox 上的解码器（H.264 与 AV1 共用会话、回调、拷帧这一整套机制）。
pub struct VtDecoder {
    codec: VtCodec,
    /// `avcC` / `av1C`（AV1 懒建会话时要靠它建格式描述）
    config: Vec<u8>,
    /// 格式描述。H.264 在 `open` 时建好；AV1 等到第一个采样（拿到尺寸）才建。
    format: CMFormatDescriptionRef,
    /// 解码会话。同上：AV1 懒建。
    session: VTDecompressionSessionRef,
    /// 回调的 refcon：`Box::into_raw` 放飞，`Drop` 时收回（VT 只持裸指针）。
    shared: *mut Shared,
}

impl VtDecoder {
    /// 用 `avcC`（H.264 的解码器初始化数据）建会话。
    pub fn open_h264(avcc: &[u8]) -> Result<Self> {
        let (sps_list, pps_list) = parse_avcc(avcc)?;
        // VT 要的是"参数集数组 + 长度数组"。SPS/PPS 各取第一条即可（多 SPS 是
        // 码流切换用的，播放路径上第一条就是当前序列的）。
        let mut sets: Vec<&[u8]> = Vec::new();
        sets.extend(sps_list.iter().map(|v| v.as_slice()));
        sets.extend(pps_list.iter().map(|v| v.as_slice()));
        if sets.is_empty() {
            return Err(MediaError::Bitstream("avcC 里没有 SPS/PPS".into()));
        }
        let pointers: Vec<*const u8> = sets.iter().map(|s| s.as_ptr()).collect();
        let sizes: Vec<usize> = sets.iter().map(|s| s.len()).collect();

        let mut format: CMFormatDescriptionRef = std::ptr::null_mut();
        let status = unsafe {
            CMVideoFormatDescriptionCreateFromH264ParameterSets(
                std::ptr::null(),
                sets.len(),
                pointers.as_ptr(),
                sizes.as_ptr(),
                4, // AVCC：4 字节长度前缀（引擎里统一按 4 字节喂）
                &mut format,
            )
        };
        if status != 0 || format.is_null() {
            return Err(MediaError::Bitstream(format!(
                "VideoToolbox 不接受这组 SPS/PPS（OSStatus {status}）"
            )));
        }

        let shared = Box::into_raw(Box::new(Shared {
            frames: Mutex::new(Vec::new()),
            error: Mutex::new(None),
        }));
        match unsafe { create_session(format, shared, VtCodec::H264) } {
            Ok(session) => Ok(Self {
                codec: VtCodec::H264,
                config: avcc.to_vec(),
                format,
                session,
                shared,
            }),
            Err(e) => {
                unsafe {
                    let _ = Box::from_raw(shared);
                    CFRelease(format);
                }
                Err(e)
            }
        }
    }

    /// 用 `av1C` 建 AV1 解码器。
    ///
    /// **不在这里建会话**：`CMVideoFormatDescriptionCreate` 必须给尺寸，而 AV1 的
    /// 尺寸只在码流的序列头里（`av1C` 里没有）。会话推迟到第一个采样进来时建 ——
    /// 见 [`Self::ensure_av1_session`]。
    pub fn open_av1(av1c: &[u8]) -> Result<Self> {
        av1::validate_av1c(av1c)?;
        let shared = Box::into_raw(Box::new(Shared {
            frames: Mutex::new(Vec::new()),
            error: Mutex::new(None),
        }));
        Ok(Self {
            codec: VtCodec::Av1,
            config: av1c.to_vec(),
            format: std::ptr::null_mut(),
            session: std::ptr::null_mut(),
            shared,
        })
    }

    /// AV1：拿序列头给的尺寸把格式描述与解码会话建起来（只在第一次调用时真正做事）。
    fn ensure_av1_session(&mut self, width: u32, height: u32, bit_depth: u8) -> Result<()> {
        if !self.session.is_null() {
            return Ok(());
        }
        if bit_depth != 8 {
            // 不静默降位：10/12 位的采样按 8 位解释就是一团错色，宁可如实说不支持
            return Err(MediaError::Unsupported(format!(
                "AV1 是 {bit_depth} 位深，而引擎对外的画面格式是 8 位 NV12（高位深要等重建层做转换）"
            )));
        }
        if width == 0 || height == 0 {
            return Err(MediaError::Bitstream("AV1 序列头给的尺寸是 0".into()));
        }
        let format = unsafe { av1_format_description(&self.config, width, height)? };
        match unsafe { create_session(format, self.shared, VtCodec::Av1) } {
            Ok(session) => {
                self.format = format;
                self.session = session;
                Ok(())
            }
            Err(e) => {
                unsafe { CFRelease(format) };
                Err(e)
            }
        }
    }

    /// 取出回调收集到的帧。
    fn take_frames(&self) -> Vec<VideoFrame> {
        let shared = unsafe { &*self.shared };
        let mut guard = match shared.frames.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        std::mem::take(&mut *guard)
    }

    fn take_error(&self) -> Option<String> {
        let shared = unsafe { &*self.shared };
        let mut guard = match shared.error.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.take()
    }
}

/// 建解码会话：目标画面固定 **NV12 + 允许 IOSurface**（让解码器与 GPU 共享内存）。
unsafe fn create_session(
    format: CMFormatDescriptionRef,
    shared: *mut Shared,
    codec: VtCodec,
) -> Result<VTDecompressionSessionRef> {
    let attrs = {
        let fmt_num = CFNumberCreate(
            std::ptr::null(),
            CF_NUMBER_SINT32,
            &(PIXEL_FORMAT_420V as i32) as *const i32 as *const c_void,
        );
        let empty = CFDictionaryCreate(
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            &kCFTypeDictionaryKeyCallBacks as *const c_void,
            &kCFTypeDictionaryValueCallBacks as *const c_void,
        );
        let keys = [kCVPixelBufferPixelFormatTypeKey, kCVPixelBufferIOSurfacePropertiesKey];
        let values = [fmt_num as CFTypeRef, empty as CFTypeRef];
        let dict = CFDictionaryCreate(
            std::ptr::null(),
            keys.as_ptr(),
            values.as_ptr(),
            2,
            &kCFTypeDictionaryKeyCallBacks as *const c_void,
            &kCFTypeDictionaryValueCallBacks as *const c_void,
        );
        CFRelease(fmt_num);
        CFRelease(empty);
        dict
    };
    if attrs.is_null() {
        return Err(MediaError::Io("建解码目标属性失败".into()));
    }

    let record = VtOutputCallbackRecord {
        callback: Some(output_callback),
        refcon: shared as *mut c_void,
    };
    let mut session: VTDecompressionSessionRef = std::ptr::null_mut();
    let status = VTDecompressionSessionCreate(
        std::ptr::null(),
        format,
        std::ptr::null(),
        attrs,
        &record,
        &mut session,
    );
    CFRelease(attrs);
    if status != 0 || session.is_null() {
        let hint = if codec == VtCodec::Av1 && !hardware_decode_available(&CodecId::Av1) {
            "（这台机器没有 AV1 解码器：Apple 只从 M3 起提供，Intel 机连软件兜底都没有 —— 实测 kVTCouldNotFindVideoDecoderErr = -12906）"
        } else {
            ""
        };
        return Err(MediaError::Unsupported(format!(
            "VideoToolbox 起不了这个 {} 流的解码会话（OSStatus {status}）{hint}",
            codec.name()
        )));
    }
    Ok(session)
}

/// 建 AV1 的格式描述：通用入口 + `{"av1C": <配置盒>}` 采样描述。
///
/// 为什么不能用 `…FromH264ParameterSets` 那类专用入口：CoreMedia 没有 AV1 的
/// 专用构造函数，规范的接法就是"通用入口 + 挂 `av1C`"。
unsafe fn av1_format_description(
    av1c: &[u8],
    width: u32,
    height: u32,
) -> Result<CMFormatDescriptionRef> {
    av1::validate_av1c(av1c)?;
    let data = CFDataCreate(std::ptr::null(), av1c.as_ptr(), av1c.len() as CFIndex);
    if data.is_null() {
        return Err(MediaError::Io("建 av1C 的 CFData 失败".into()));
    }
    // 采样描述盒：键名就是盒子名，所以这里的 "av1C" 必须与容器里那个盒子同名
    let key = CFStringCreateWithCString(std::ptr::null(), b"av1C\0".as_ptr(), CF_STRING_ENCODING_UTF8);
    let atoms = CFDictionaryCreateMutable(
        std::ptr::null(),
        1,
        &kCFTypeDictionaryKeyCallBacks as *const c_void,
        &kCFTypeDictionaryValueCallBacks as *const c_void,
    );
    CFDictionarySetValue(atoms, key as CFTypeRef, data);
    let ext = CFDictionaryCreateMutable(
        std::ptr::null(),
        1,
        &kCFTypeDictionaryKeyCallBacks as *const c_void,
        &kCFTypeDictionaryValueCallBacks as *const c_void,
    );
    CFDictionarySetValue(
        ext,
        kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms as CFTypeRef,
        atoms as CFTypeRef,
    );

    let mut format: CMFormatDescriptionRef = std::ptr::null_mut();
    let status = CMVideoFormatDescriptionCreate(
        std::ptr::null(),
        CODEC_TYPE_AV1,
        width as i32,
        height as i32,
        ext,
        &mut format,
    );
    CFRelease(key);
    CFRelease(data);
    CFRelease(atoms);
    CFRelease(ext);
    if status != 0 || format.is_null() {
        return Err(MediaError::Bitstream(format!(
            "VideoToolbox 不接受这个 av1C（OSStatus {status}）"
        )));
    }
    Ok(format)
}

/// 这台机器上某个编码有没有**平台硬解**（诊断/日志用）。
pub fn hardware_decode_available(codec: &CodecId) -> bool {
    let codec_type = match codec {
        CodecId::H264 => CODEC_TYPE_H264,
        CodecId::H265 => CODEC_TYPE_HEVC,
        CodecId::Av1 => CODEC_TYPE_AV1,
        // 其余编码在 VideoToolbox 这边没有硬解通道（VP9/VP8：Apple 只对 VP9 提供
        // 受限支持，且不在这条通路上）
        _ => return false,
    };
    unsafe { VTIsHardwareDecodeSupported(codec_type) != 0 }
}

impl crate::VideoDecoder for VtDecoder {
    fn decode(&mut self, sample_avcc: &[u8], pts_us: i64) -> Result<Vec<VideoFrame>> {
        if sample_avcc.is_empty() {
            return Ok(Vec::new());
        }
        if self.codec == VtCodec::Av1 {
            // AV1 这一步是**必需的**（不只是诊断）：尺寸只在序列头里，而会话要靠
            // 尺寸才能建起来。顺便也让"只有序列头、还没有帧"的采样不白跑一趟。
            let info = av1::inspect_sample(sample_avcc)?;
            if self.session.is_null() {
                let sh = info.sequence_header.as_ref().ok_or_else(|| {
                    MediaError::Bitstream(
                        "这段 AV1 采样里没有序列头，而 av1C 不带尺寸 —— 建不了 VideoToolbox 会话"
                            .into(),
                    )
                })?;
                self.ensure_av1_session(
                    sh.max_frame_width,
                    sh.max_frame_height,
                    sh.color.bit_depth,
                )?;
            }
            if !info.has_bitstream {
                // 只有时间分隔符/序列头/元数据：确实还没有要解的东西
                return Ok(Vec::new());
            }
        }
        let block = unsafe {
            let mut block: CMBlockBufferRef = std::ptr::null_mut();
            let status = CMBlockBufferCreateWithMemoryBlock(
                std::ptr::null(),
                std::ptr::null_mut(),
                sample_avcc.len(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                sample_avcc.len(),
                BLOCK_BUFFER_ASSURE_MEMORY_NOW,
                &mut block,
            );
            if status != 0 || block.is_null() {
                return Err(MediaError::Io(format!("建 CMBlockBuffer 失败（{status}）")));
            }
            let status = CMBlockBufferReplaceDataBytes(
                sample_avcc.as_ptr() as *const c_void,
                block,
                0,
                sample_avcc.len(),
            );
            if status != 0 {
                CFRelease(block);
                return Err(MediaError::Io(format!("拷采样进 CMBlockBuffer 失败（{status}）")));
            }
            block
        };

        let timing = CmSampleTimingInfo {
            duration: CmTime { value: 0, timescale: 1_000_000, flags: 0, epoch: 0 },
            presentation_time_stamp: CmTime {
                value: pts_us,
                timescale: 1_000_000,
                flags: CM_TIME_VALID,
                epoch: 0,
            },
            decode_time_stamp: CmTime {
                value: pts_us,
                timescale: 1_000_000,
                flags: CM_TIME_VALID,
                epoch: 0,
            },
        };
        let size = sample_avcc.len();
        let sample = unsafe {
            let mut sample: CMSampleBufferRef = std::ptr::null_mut();
            let status = CMSampleBufferCreateReady(
                std::ptr::null(),
                block,
                self.format,
                1,
                1,
                &timing,
                1,
                &size,
                &mut sample,
            );
            CFRelease(block); // 采样缓冲自己持有一份
            if status != 0 || sample.is_null() {
                return Err(MediaError::Io(format!("建 CMSampleBuffer 失败（{status}）")));
            }
            sample
        };

        let status = unsafe {
            VTDecompressionSessionDecodeFrame(
                self.session,
                sample,
                DECODE_FLAGS_NONE,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        unsafe { CFRelease(sample) };
        if status != 0 {
            return Err(MediaError::Bitstream(format!(
                "VideoToolbox 拒绝了这个采样（OSStatus {status}）"
            )));
        }
        unsafe { VTDecompressionSessionWaitForAsynchronousFrames(self.session) };
        if let Some(e) = self.take_error() {
            return Err(MediaError::Bitstream(e));
        }
        Ok(self.take_frames())
    }

    fn flush(&mut self) -> Result<Vec<VideoFrame>> {
        if self.session.is_null() {
            // AV1 还没建会话（一个采样都没喂过）：没有排队帧可吐
            return Ok(Vec::new());
        }
        // 空采样 + DoNotOutputFrame = VideoToolbox 的"把排队帧吐出来"惯用法。
        unsafe {
            VTDecompressionSessionDecodeFrame(
                self.session,
                std::ptr::null_mut(),
                DECODE_FLAG_DO_NOT_OUTPUT_FRAME,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            VTDecompressionSessionWaitForAsynchronousFrames(self.session);
        }
        Ok(self.take_frames())
    }

    fn backend(&self) -> &'static str {
        "videotoolbox"
    }
}

/// 普通解码（不要求额外行为）。
const DECODE_FLAGS_NONE: VTDecodeFrameFlags = 0;

// 安全性：VideoToolbox 的会话对象本身没有线程亲和性要求，但**不能被并发使用**。
// 我们的约定是"一个解码器只在一个线程上用"（`decode` 顺序调用），跨线程只是
// "把所有权搬过去"（比如未来把解码放到专用线程），所以 `Send` 是安全的。
// 回调在 VT 自己的线程上跑，它只碰 `Mutex` 保护的状态，不碰会话。
unsafe impl Send for VtDecoder {}

impl Drop for VtDecoder {
    fn drop(&mut self) {
        unsafe {
            // AV1 的会话可能是懒建的 —— 没建过就是空指针，`CFRelease` 不接受空
            if !self.session.is_null() {
                VTDecompressionSessionWaitForAsynchronousFrames(self.session);
                VTDecompressionSessionInvalidate(self.session);
                CFRelease(self.session);
            }
            if !self.format.is_null() {
                CFRelease(self.format);
            }
            // 收回回调的 refcon（VT 已经不再回调了：会话已 Invalidate + Release）
            let _ = Box::from_raw(self.shared);
        }
    }
}

/// VideoToolbox 的输出回调：把 `CVPixelBuffer` 拷成紧凑 NV12。
///
/// **只做拷贝**：回调运行在解码器自己的线程上，任何耗时/阻塞操作都会拖慢整条管线。
extern "C" fn output_callback(
    refcon: *mut c_void,
    _source: *mut c_void,
    status: OSStatus,
    _flags: VTDecodeInfoFlags,
    image: CVImageBufferRef,
    pts: CmTime,
    _duration: CmTime,
) {
    if refcon.is_null() {
        return;
    }
    let shared = unsafe { &*(refcon as *const Shared) };
    let record = |msg: String| {
        if let Ok(mut g) = shared.error.lock() {
            *g = Some(msg);
        }
    };
    if status != 0 || image.is_null() {
        record(format!("解码失败（OSStatus {status}）"));
        return;
    }
    match copy_nv12(image, pts) {
        Ok(frame) => {
            if let Ok(mut g) = shared.frames.lock() {
                g.push(frame);
            }
        }
        Err(e) => record(format!("{e}")),
    }
}

/// `CVPixelBuffer` → 紧凑 NV12。
fn copy_nv12(image: CVImageBufferRef, pts: CmTime) -> Result<VideoFrame> {
    unsafe {
        let pix = image as CVPixelBufferRef;
        let format = CVPixelBufferGetPixelFormatType(pix);
        if format != PIXEL_FORMAT_420V {
            return Err(MediaError::Unsupported(format!(
                "解码器给的不是 NV12（pixel format {format:#x}）"
            )));
        }
        if CVPixelBufferGetPlaneCount(pix) < 2 {
            return Err(MediaError::Bitstream("NV12 应该有两个平面".into()));
        }
        if CVPixelBufferLockBaseAddress(pix, PIXEL_BUFFER_LOCK_READ_ONLY) != 0 {
            return Err(MediaError::Io("锁定解码画面失败".into()));
        }
        let width = CVPixelBufferGetWidth(pix) as u32;
        let height = CVPixelBufferGetHeight(pix) as u32;
        let mut out = vec![0u8; VideoFrame::expected_len(width, height)];
        let (y_len, c_len) = ((width as usize) * (height as usize), (width as usize) * (height as usize) / 2);
        // Y 平面：逐行拷（源有行距，目标是紧凑）
        let y_src = CVPixelBufferGetBaseAddressOfPlane(pix, 0);
        let y_stride = CVPixelBufferGetBytesPerRowOfPlane(pix, 0);
        for row in 0..height as usize {
            let src = y_src.add(row * y_stride);
            let dst = out.as_mut_ptr().add(row * width as usize);
            std::ptr::copy_nonoverlapping(src, dst, width as usize);
        }
        // CbCr 交织平面：高度是画面的一半
        let c_src = CVPixelBufferGetBaseAddressOfPlane(pix, 1);
        let c_stride = CVPixelBufferGetBytesPerRowOfPlane(pix, 1);
        let c_rows = (height as usize) / 2;
        for row in 0..c_rows {
            let src = c_src.add(row * c_stride);
            let dst = out.as_mut_ptr().add(y_len + row * width as usize);
            std::ptr::copy_nonoverlapping(src, dst, width as usize);
        }
        CVPixelBufferUnlockBaseAddress(pix, PIXEL_BUFFER_LOCK_READ_ONLY);
        debug_assert_eq!(out.len(), y_len + c_len);
        Ok(VideoFrame { width, height, pts_us: pts.value * 1_000_000 / pts.timescale.max(1) as i64, nv12: out })
    }
}

/// `avcC` 里拆出来的 SPS / PPS 列表：每一项是一段 NAL 负载。
///
/// 起个别名不只是为了压 clippy 的类型复杂度告警 —— `(Vec<Vec<u8>>, Vec<Vec<u8>>)`
/// 光看签名分不清哪个是 SPS、哪个是 PPS，起了名字调用点才好读。
type ParameterSets = (Vec<Vec<u8>>, Vec<Vec<u8>>);

/// 从 `avcC` 里拆出 SPS / PPS 列表。
///
/// avcC 结构（ISO 14496-15 §5.2.4.1.1）：
/// `[0]=1` `[1..4]=profile/compat/level` `[4]=0xFC|lengthSizeMinusOne`
/// `[5]=0xE0|numSPS` 之后是 `u16 长度 + 内容`，再 `numPPS` + 同样的序列。
fn parse_avcc(avcc: &[u8]) -> Result<ParameterSets> {
    if avcc.len() < 7 || avcc[0] != 1 {
        return Err(MediaError::Bitstream("avcC 头不合法".into()));
    }
    let mut pos = 5usize;
    let num_sps = (avcc[pos] & 0x1F) as usize;
    pos += 1;
    let mut sps = Vec::new();
    for _ in 0..num_sps {
        let len = read_u16(avcc, pos)?;
        pos += 2;
        if pos + len > avcc.len() {
            return Err(MediaError::Bitstream("avcC 里的 SPS 越界".into()));
        }
        sps.push(avcc[pos..pos + len].to_vec());
        pos += len;
    }
    if pos >= avcc.len() {
        return Err(MediaError::Bitstream("avcC 里没有 PPS".into()));
    }
    let num_pps = avcc[pos] as usize;
    pos += 1;
    let mut pps = Vec::new();
    for _ in 0..num_pps {
        let len = read_u16(avcc, pos)?;
        pos += 2;
        if pos + len > avcc.len() {
            return Err(MediaError::Bitstream("avcC 里的 PPS 越界".into()));
        }
        pps.push(avcc[pos..pos + len].to_vec());
        pos += len;
    }
    Ok((sps, pps))
}

fn read_u16(buf: &[u8], pos: usize) -> Result<usize> {
    if pos + 2 > buf.len() {
        return Err(MediaError::Bitstream("avcC 读长度越界".into()));
    }
    Ok(u16::from_be_bytes([buf[pos], buf[pos + 1]]) as usize)
}

/// 让 `CString` 在本文件被用到的编译单元里不报未使用（诊断信息里保留能力）。
#[allow(dead_code)]
fn _unused(_s: CString) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VideoDecoder;

    #[test]
    fn parses_sps_and_pps_from_avcc() {
        // 手工造一个 avcC：[1, profile, compat, level, 0xFF, 0xE1, len=3, SPS..., 0x01, len=2, PPS...]
        let mut avcc = vec![1u8, 0x64, 0x00, 0x28, 0xFF, 0xE1];
        avcc.extend_from_slice(&3u16.to_be_bytes());
        avcc.extend_from_slice(&[0x67, 0x64, 0x00]);
        avcc.push(0x01);
        avcc.extend_from_slice(&2u16.to_be_bytes());
        avcc.extend_from_slice(&[0x68, 0xEE]);
        let (sps, pps) = parse_avcc(&avcc).unwrap();
        assert_eq!(sps, vec![vec![0x67, 0x64, 0x00]]);
        assert_eq!(pps, vec![vec![0x68, 0xEE]]);
    }

    #[test]
    fn rejects_garbage_avcc() {
        assert!(parse_avcc(&[]).is_err());
        assert!(parse_avcc(&[0u8; 8]).is_err(), "version 不是 1 要拒绝");
        // 长度越界
        let bad = vec![1u8, 0x64, 0x00, 0x28, 0xFF, 0xE1, 0xFF, 0xFF, 0x67];
        assert!(parse_avcc(&bad).is_err());
    }

    #[test]
    fn h264_session_opens_with_real_parameter_sets() {
        // 一组真实的 SPS/PPS（640×360，来自 CLI 契约测试里用的那组）。
        // 这条测试的价值：`VtDecoder` 把 H.264 与 AV1 的会话/回调机制合并之后，
        // **桌面端播放的主路径（H.264 硬解）必须仍然建得起会话** —— 建格式描述与
        // 建会话都不需要真码流，所以这里不喂帧也能验。
        let sps: [u8; 27] = [
            0x67, 0x42, 0x00, 0x1E, 0x95, 0xA8, 0x28, 0x0F, 0x00, 0x44, 0xFC, 0xB8, 0x08, 0x80,
            0x00, 0x03, 0x00, 0x08, 0x00, 0x00, 0x03, 0x01, 0xE0, 0x78, 0xB1, 0x6C, 0x90,
        ];
        let pps: [u8; 4] = [0x68, 0xCE, 0x3C, 0x80];
        let mut avcc = vec![1u8, 0x42, 0x00, 0x1E, 0xFF, 0xE1];
        avcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(&sps);
        avcc.push(0x01);
        avcc.extend_from_slice(&(pps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(&pps);

        let dec = VtDecoder::open_h264(&avcc).expect("H.264 会话应当能建起来（这台机器有硬解）");
        assert_eq!(dec.backend(), "videotoolbox");
        assert_eq!(dec.config, avcc, "配置要留在手里（诊断/重建会话要用）");
    }

    #[test]
    fn rejects_garbage_av1c() {
        assert!(VtDecoder::open_av1(&[]).is_err(), "空的 av1C 要拒绝");
        assert!(VtDecoder::open_av1(&[0, 0, 0, 0]).is_err(), "marker 不是 0x81 要拒绝");
    }

    /// AV1：要么真建起会话（M3 及以后），要么给一句**能读懂的原因**。
    ///
    /// 这条测试在两种机器上都成立：Intel 机上它会走 Err 分支并把
    /// `kVTCouldNotFindVideoDecoderErr` 写出来；M3 机上直接成功。
    #[test]
    fn av1_open_ends_in_a_session_or_a_readable_reason() {
        let av1c = [0x81u8, 0x00, 0x0C, 0x00]; // profile 0 / level 0 / 8bit 4:2:0
        let mut dec = match VtDecoder::open_av1(&av1c) {
            Ok(d) => d,
            Err(e) => panic!("open_av1 只校验 av1C，不该在这里失败：{e}"),
        };
        // 会话是懒建的：先喂一段"只有序列头"的 AV1 采样（1920×1080）
        let mut w = crate::video::testbits::BitWriter::default();
        crate::video::testbits::write_reduced_still_picture_1080p(&mut w);
        let payload = w.finish().to_vec();
        let mut sample = vec![(2u8 << 3) | 0b0000_0010, 0x00];
        sample.push((1u8 << 3) | 0b0000_0010);
        sample.push(payload.len() as u8);
        sample.extend_from_slice(&payload);

        match dec.decode(&sample, 0) {
            Ok(frames) => {
                assert!(frames.is_empty(), "只有序列头，没有帧可出");
                assert!(
                    hardware_decode_available(&CodecId::Av1),
                    "能建起会话就说明这台机器有 AV1 解码器（应当与硬解探测一致）"
                );
            }
            Err(e) => {
                let text = format!("{e}");
                assert!(
                    text.contains("av1") || text.contains("AV1"),
                    "失败文案要写明是哪个编码：{text}"
                );
                if !hardware_decode_available(&CodecId::Av1) {
                    assert!(
                        text.contains("这台机器没有 AV1 解码器") || text.contains("OSStatus"),
                        "没有硬解时要说清原因与系统返回码：{text}"
                    );
                }
            }
        }
    }

    #[test]
    fn hardware_probe_names_the_platform() {
        // 只钉"FFI 调得通、对已知编码给出确定答案"（真值随机器变，不做断言）
        let h264 = hardware_decode_available(&CodecId::H264);
        let av1 = hardware_decode_available(&CodecId::Av1);
        // 打印出来便于人工核对（`cargo test -- --nocapture`）
        println!("硬解探测：h264={h264} av1={av1}");
        // 没有硬解的编码一律 false（VP9 不在 VideoToolbox 这条通路上）
        assert!(!hardware_decode_available(&CodecId::Vp9));
        assert!(!hardware_decode_available(&CodecId::Aac));
    }
}
