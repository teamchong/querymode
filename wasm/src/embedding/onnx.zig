//! ONNX Runtime C API bindings for Zig
//!
//! This module provides low-level bindings to the ONNX Runtime C API.
//! See: https://onnxruntime.ai/docs/api/c/

const std = @import("std");

// =============================================================================
// Opaque Types
// =============================================================================

pub const OrtEnv = opaque {};
pub const OrtSession = opaque {};
pub const OrtSessionOptions = opaque {};
pub const OrtRunOptions = opaque {};
pub const OrtValue = opaque {};
pub const OrtMemoryInfo = opaque {};
pub const OrtAllocator = opaque {};
pub const OrtStatus = opaque {};
pub const OrtTypeInfo = opaque {};
pub const OrtTensorTypeAndShapeInfo = opaque {};

// =============================================================================
// Enums
// =============================================================================

pub const OrtLoggingLevel = enum(c_int) {
    verbose = 0,
    info = 1,
    warning = 2,
    @"error" = 3,
    fatal = 4,
};

pub const ONNXTensorElementDataType = enum(c_int) {
    undefined = 0,
    float = 1,
    uint8 = 2,
    int8 = 3,
    uint16 = 4,
    int16 = 5,
    int32 = 6,
    int64 = 7,
    string = 8,
    bool = 9,
    float16 = 10,
    double = 11,
    uint32 = 12,
    uint64 = 13,
    complex64 = 14,
    complex128 = 15,
    bfloat16 = 16,
};

pub const OrtAllocatorType = enum(c_int) {
    invalid = -1,
    device = 0,
    arena = 1,
};

pub const OrtMemType = enum(c_int) {
    cpu_input = -2,
    cpu_output = -1,
    default = 0,
};

// =============================================================================
// API Base and Version
// =============================================================================

pub const ORT_API_VERSION: c_uint = 18; // ONNX Runtime 1.16+

pub const OrtApiBase = extern struct {
    GetApi: *const fn (version: c_uint) callconv(.c) ?*const OrtApi,
    GetVersionString: *const fn () callconv(.c) [*:0]const u8,
};

// =============================================================================
// Main API Structure
// =============================================================================

pub const OrtApi = extern struct {
    // Environment
    CreateEnv: *const fn (
        log_severity_level: OrtLoggingLevel,
        logid: [*:0]const u8,
        out: *?*OrtEnv,
    ) callconv(.c) ?*OrtStatus,

    // Session Options
    CreateSessionOptions: *const fn (out: *?*OrtSessionOptions) callconv(.c) ?*OrtStatus,
    SetIntraOpNumThreads: *const fn (options: *OrtSessionOptions, intra_op_num_threads: c_int) callconv(.c) ?*OrtStatus,
    SetInterOpNumThreads: *const fn (options: *OrtSessionOptions, inter_op_num_threads: c_int) callconv(.c) ?*OrtStatus,
    SetSessionGraphOptimizationLevel: *const fn (options: *OrtSessionOptions, level: c_int) callconv(.c) ?*OrtStatus,

    // Session
    CreateSession: *const fn (
        env: *OrtEnv,
        model_path: [*:0]const u8,
        options: *const OrtSessionOptions,
        out: *?*OrtSession,
    ) callconv(.c) ?*OrtStatus,

    // Run
    Run: *const fn (
        session: *OrtSession,
        run_options: ?*const OrtRunOptions,
        input_names: [*]const [*:0]const u8,
        inputs: [*]const ?*const OrtValue,
        input_len: usize,
        output_names: [*]const [*:0]const u8,
        output_names_len: usize,
        outputs: [*]?*OrtValue,
    ) callconv(.c) ?*OrtStatus,

    // Tensor creation
    CreateTensorWithDataAsOrtValue: *const fn (
        info: *const OrtMemoryInfo,
        p_data: *anyopaque,
        p_data_len: usize,
        shape: [*]const i64,
        shape_len: usize,
        type_: ONNXTensorElementDataType,
        out: *?*OrtValue,
    ) callconv(.c) ?*OrtStatus,

    // Memory info
    CreateCpuMemoryInfo: *const fn (
        type_: OrtAllocatorType,
        mem_type: OrtMemType,
        out: *?*OrtMemoryInfo,
    ) callconv(.c) ?*OrtStatus,

    // Get tensor data
    GetTensorMutableData: *const fn (
        value: *OrtValue,
        out: *?*anyopaque,
    ) callconv(.c) ?*OrtStatus,

    // Type info
    GetTensorTypeAndShape: *const fn (
        value: *const OrtValue,
        out: *?*OrtTensorTypeAndShapeInfo,
    ) callconv(.c) ?*OrtStatus,

    GetDimensionsCount: *const fn (
        info: *const OrtTensorTypeAndShapeInfo,
        out: *usize,
    ) callconv(.c) ?*OrtStatus,

    GetDimensions: *const fn (
        info: *const OrtTensorTypeAndShapeInfo,
        dim_values: [*]i64,
        dim_values_length: usize,
    ) callconv(.c) ?*OrtStatus,

    GetTensorElementType: *const fn (
        info: *const OrtTensorTypeAndShapeInfo,
        out: *ONNXTensorElementDataType,
    ) callconv(.c) ?*OrtStatus,

    // Session info
    SessionGetInputCount: *const fn (
        session: *const OrtSession,
        out: *usize,
    ) callconv(.c) ?*OrtStatus,

    SessionGetOutputCount: *const fn (
        session: *const OrtSession,
        out: *usize,
    ) callconv(.c) ?*OrtStatus,

    // Allocator
    GetAllocatorWithDefaultOptions: *const fn (
        out: *?*OrtAllocator,
    ) callconv(.c) ?*OrtStatus,

    // Names
    SessionGetInputName: *const fn (
        session: *const OrtSession,
        index: usize,
        allocator: *OrtAllocator,
        out: *?[*:0]u8,
    ) callconv(.c) ?*OrtStatus,

    SessionGetOutputName: *const fn (
        session: *const OrtSession,
        index: usize,
        allocator: *OrtAllocator,
        out: *?[*:0]u8,
    ) callconv(.c) ?*OrtStatus,

    // Release functions
    ReleaseEnv: *const fn (env: *OrtEnv) callconv(.c) void,
    ReleaseSession: *const fn (session: *OrtSession) callconv(.c) void,
    ReleaseSessionOptions: *const fn (options: *OrtSessionOptions) callconv(.c) void,
    ReleaseValue: *const fn (value: *OrtValue) callconv(.c) void,
    ReleaseMemoryInfo: *const fn (info: *OrtMemoryInfo) callconv(.c) void,
    ReleaseTensorTypeAndShapeInfo: *const fn (info: *OrtTensorTypeAndShapeInfo) callconv(.c) void,
    ReleaseStatus: *const fn (status: *OrtStatus) callconv(.c) void,

    // Status
    GetErrorCode: *const fn (status: *const OrtStatus) callconv(.c) c_int,
    GetErrorMessage: *const fn (status: *const OrtStatus) callconv(.c) [*:0]const u8,

    // Allocator free
    AllocatorFree: *const fn (allocator: *OrtAllocator, p: *anyopaque) callconv(.c) void,
};

// =============================================================================
// C API Entry Point
// =============================================================================

// ONNX runtime is optional - linked when built with -Donnx=/path
// When ONNX is not linked, OrtGetApiBase returns null

// Check if we have ONNX runtime linked via build options
const build_options = @import("build_options");
const has_onnx = if (@hasDecl(build_options, "use_onnx")) build_options.use_onnx else false;

// Extern declaration for ONNX runtime C API entry point
extern "c" fn OrtGetApiBase_extern() callconv(.c) ?*const OrtApiBase;

/// Get the ONNX Runtime API base
/// Returns null if ONNX runtime is not linked
pub fn OrtGetApiBase() ?*const OrtApiBase {
    if (has_onnx) {
        return OrtGetApiBase_extern();
    } else {
        return null;
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Get the ONNX Runtime API for the specified version
pub fn getApi() ?*const OrtApi {
    const base = OrtGetApiBase() orelse return null;
    return base.GetApi(ORT_API_VERSION);
}

/// Get ONNX Runtime version string
pub fn getVersionString() []const u8 {
    const base = OrtGetApiBase() orelse return "not available";
    const version_ptr = base.GetVersionString();
    return std.mem.sliceTo(version_ptr, 0);
}

/// Check if ONNX runtime is available
pub fn isAvailable() bool {
    return OrtGetApiBase() != null and getApi() != null;
}

// =============================================================================
// Error Handling
// =============================================================================

pub const OrtError = error{
    CreateEnvFailed,
    CreateSessionFailed,
    CreateSessionOptionsFailed,
    CreateTensorFailed,
    CreateMemoryInfoFailed,
    RunFailed,
    GetTensorDataFailed,
    GetTensorShapeFailed,
    InvalidApi,
    SessionInfoFailed,
    AllocatorFailed,
};

/// Convert OrtStatus to Zig error
pub fn checkStatus(api: *const OrtApi, status: ?*OrtStatus) OrtError!void {
    if (status) |s| {
        const code = api.GetErrorCode(s);
        const msg = api.GetErrorMessage(s);
        std.log.err("ONNX Runtime error {d}: {s}", .{ code, msg });
        api.ReleaseStatus(s);
        return OrtError.RunFailed;
    }
}
