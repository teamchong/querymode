//! LanceQL GPU Module - Cross-platform GPU acceleration via wgpu-native
//!
//! Replaces Metal-only backend with cross-platform WebGPU.
//! Shares WGSL shaders with browser for single codebase.
//!
//! Supported platforms:
//! - macOS: Metal backend
//! - Linux: Vulkan backend
//! - Windows: DirectX 12 / Vulkan backend

const std = @import("std");

// Core GPU context
pub const gpu_context = @import("gpu_context.zig");
pub const GPUContext = gpu_context.GPUContext;
pub const getGlobalContext = gpu_context.getGlobalContext;
pub const releaseGlobalContext = gpu_context.releaseGlobalContext;
pub const isGPUAvailable = gpu_context.isGPUAvailable;
pub const shaders = gpu_context.shaders;

// GPU operations
pub const vector_search = @import("vector_search.zig");
pub const GPUVectorSearch = vector_search.GPUVectorSearch;
pub const SearchResult = vector_search.SearchResult;
pub const DistanceMetric = vector_search.DistanceMetric;
pub const cosineSimilarity = vector_search.cosineSimilarity;
pub const dotProduct = vector_search.dotProduct;
pub const l2DistanceSquared = vector_search.l2DistanceSquared;
pub const batchCosineSimilarity = vector_search.batchCosineSimilarity;

pub const hash_table = @import("hash_table.zig");
pub const GPUHashTable = hash_table.GPUHashTable;
pub const GPUHashTable64 = hash_table.GPUHashTable64;
pub const JoinHashTable = hash_table.JoinHashTable;
pub const HashTableError = hash_table.HashTableError;
pub const JOIN_GPU_THRESHOLD = hash_table.JOIN_GPU_THRESHOLD;

pub const batch_ops = @import("batch_ops.zig");
pub const GPUBatchOps = batch_ops.GPUBatchOps;
pub const batchAdd = batch_ops.batchAdd;
pub const batchSub = batch_ops.batchSub;
pub const batchMul = batch_ops.batchMul;
pub const batchDiv = batch_ops.batchDiv;
pub const batchMulScalar = batch_ops.batchMulScalar;
pub const batchAbs = batch_ops.batchAbs;
pub const batchMin = batch_ops.batchMin;
pub const batchMax = batch_ops.batchMax;

// NOTE: wgpu-native is temporarily stubbed due to Zig 0.15 compatibility issues
// GPU operations fall back to CPU implementations
// TODO: Re-enable when a compatible wgpu_native_zig version is available

/// Threshold for GPU acceleration (use GPU for large datasets)
pub const GPU_THRESHOLD: usize = 10_000;

// =============================================================================
// Compatibility functions for migration from Metal
// =============================================================================

/// Initialize GPU context (compatibility with metal.initGPU)
/// Returns true if GPU is available, false otherwise
pub fn initGPU() bool {
    const ctx = getGlobalContext(std.heap.page_allocator) catch return false;
    _ = ctx;
    return true;
}

/// Check if GPU is ready (compatibility with metal.isGPUReady)
pub fn isGPUReady() bool {
    return isGPUAvailable();
}

/// Cleanup GPU resources (compatibility with metal.cleanupGPU)
pub fn cleanupGPU() void {
    releaseGlobalContext();
}

/// GPU-accelerated batch cosine similarity (compatibility with metal.gpuCosineSimilarityBatch)
pub fn gpuCosineSimilarityBatch(
    query: []const f32,
    vectors: []const f32,
    dim: usize,
    scores: []f32,
) void {
    const num_vectors = vectors.len / dim;

    for (0..num_vectors) |i| {
        const vec = vectors[i * dim .. (i + 1) * dim];
        var dot: f32 = 0;
        var norm_q: f32 = 0;
        var norm_v: f32 = 0;

        for (0..dim) |j| {
            dot += query[j] * vec[j];
            norm_q += query[j] * query[j];
            norm_v += vec[j] * vec[j];
        }

        const denom = @sqrt(norm_q) * @sqrt(norm_v);
        scores[i] = if (denom > 0) dot / denom else 0;
    }
}

/// GPU-accelerated batch dot product (compatibility with metal.gpuDotProductBatch)
pub fn gpuDotProductBatch(
    query: []const f32,
    vectors: []const f32,
    dim: usize,
    scores: []f32,
) void {
    const num_vectors = vectors.len / dim;

    for (0..num_vectors) |i| {
        const vec = vectors[i * dim .. (i + 1) * dim];
        var dot: f32 = 0;

        for (0..dim) |j| {
            dot += query[j] * vec[j];
        }

        scores[i] = dot;
    }
}

/// Compile-time flag indicating GPU is available on this platform
/// Unlike Metal, wgpu supports all platforms
pub const use_metal = true; // For compatibility, always true since wgpu is cross-platform

/// Check if GPU should be used for given data size
pub fn shouldUseGPU(size: usize) bool {
    return size >= GPU_THRESHOLD and isGPUAvailable();
}

/// Platform info string
pub fn getPlatformInfo() []const u8 {
    const builtin = @import("builtin");
    return switch (builtin.os.tag) {
        .macos => "wgpu-native (Metal backend)",
        .linux => "wgpu-native (Vulkan backend)",
        .windows => "wgpu-native (DirectX 12 backend)",
        else => "wgpu-native (unknown backend)",
    };
}

test "platform info" {
    const info = getPlatformInfo();
    std.debug.print("Platform: {s}\n", .{info});
}
