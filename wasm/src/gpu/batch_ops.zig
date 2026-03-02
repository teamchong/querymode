//! GPU Batch Operations
//!
//! Provides GPU-accelerated element-wise and reduction operations:
//! - Element-wise: add, sub, mul, div, min, max, abs
//! - Reductions: sum, min, max, count
//!
//! Uses shared WGSL shaders from packages/shared/gpu/shaders/reduce.wgsl

const std = @import("std");
const wgpu = @import("wgpu");
const gpu_context = @import("gpu_context.zig");
const GPUContext = gpu_context.GPUContext;

const Allocator = std.mem.Allocator;

/// GPU threshold - minimum size to use GPU
const GPU_THRESHOLD: usize = 10_000;

/// Batch operations errors
pub const BatchOpsError = error{
    OutOfMemory,
    GPUError,
    PipelineNotReady,
    BufferCreationFailed,
    BindGroupCreationFailed,
    EncoderCreationFailed,
    ComputePassFailed,
    CommandBufferFailed,
    MapFailed,
    StagingBufferFailed,
    GetMappedRangeFailed,
};

/// GPU-accelerated batch operations
pub const GPUBatchOps = struct {
    allocator: Allocator,
    ctx: ?*GPUContext = null,
    reduce_sum_pipeline: ?*wgpu.ComputePipeline = null,
    reduce_sum_final_pipeline: ?*wgpu.ComputePipeline = null,
    reduce_min_pipeline: ?*wgpu.ComputePipeline = null,
    reduce_min_final_pipeline: ?*wgpu.ComputePipeline = null,
    reduce_max_pipeline: ?*wgpu.ComputePipeline = null,
    reduce_max_final_pipeline: ?*wgpu.ComputePipeline = null,

    const Self = @This();

    pub fn init(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = .{
            .allocator = allocator,
        };

        // Try to initialize GPU
        self.ensureGPU() catch {};
        return self;
    }

    fn ensureGPU(self: *Self) !void {
        if (self.ctx != null) return;

        self.ctx = try gpu_context.getGlobalContext(self.allocator);
        const ctx = self.ctx.?;

        // Load reduce shader
        const reduce_shader = try ctx.loadShader(
            "reduce",
            gpu_context.shaders.reduce,
        );

        // Create pipelines
        self.reduce_sum_pipeline = ctx.createPipeline("reduce_sum", reduce_shader, "reduce_sum") catch null;
        self.reduce_sum_final_pipeline = ctx.createPipeline("reduce_sum_final", reduce_shader, "reduce_sum_final") catch null;
        self.reduce_min_pipeline = ctx.createPipeline("reduce_min", reduce_shader, "reduce_min") catch null;
        self.reduce_min_final_pipeline = ctx.createPipeline("reduce_min_final", reduce_shader, "reduce_min_final") catch null;
        self.reduce_max_pipeline = ctx.createPipeline("reduce_max", reduce_shader, "reduce_max") catch null;
        self.reduce_max_final_pipeline = ctx.createPipeline("reduce_max_final", reduce_shader, "reduce_max_final") catch null;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self);
    }

    // =========================================================================
    // Element-wise operations (CPU implementations - GPU shader not available)
    // =========================================================================

    /// Element-wise addition: out[i] = a[i] + b[i]
    pub fn add(self: *Self, a: []const f32, b: []const f32, out: []f32) void {
        _ = self;
        std.debug.assert(a.len == b.len);
        std.debug.assert(a.len == out.len);

        for (a, b, out) |av, bv, *ov| {
            ov.* = av + bv;
        }
    }

    /// Element-wise subtraction: out[i] = a[i] - b[i]
    pub fn sub(self: *Self, a: []const f32, b: []const f32, out: []f32) void {
        _ = self;
        std.debug.assert(a.len == b.len);
        std.debug.assert(a.len == out.len);

        for (a, b, out) |av, bv, *ov| {
            ov.* = av - bv;
        }
    }

    /// Element-wise multiplication: out[i] = a[i] * b[i]
    pub fn mul(self: *Self, a: []const f32, b: []const f32, out: []f32) void {
        _ = self;
        std.debug.assert(a.len == b.len);
        std.debug.assert(a.len == out.len);

        for (a, b, out) |av, bv, *ov| {
            ov.* = av * bv;
        }
    }

    /// Element-wise division: out[i] = a[i] / b[i]
    pub fn div(self: *Self, a: []const f32, b: []const f32, out: []f32) void {
        _ = self;
        std.debug.assert(a.len == b.len);
        std.debug.assert(a.len == out.len);

        for (a, b, out) |av, bv, *ov| {
            ov.* = if (bv != 0) av / bv else 0;
        }
    }

    /// Element-wise minimum: out[i] = min(a[i], b[i])
    pub fn minArrays(self: *Self, a: []const f32, b: []const f32, out: []f32) void {
        _ = self;
        std.debug.assert(a.len == b.len);
        std.debug.assert(a.len == out.len);

        for (a, b, out) |av, bv, *ov| {
            ov.* = @min(av, bv);
        }
    }

    /// Element-wise maximum: out[i] = max(a[i], b[i])
    pub fn maxArrays(self: *Self, a: []const f32, b: []const f32, out: []f32) void {
        _ = self;
        std.debug.assert(a.len == b.len);
        std.debug.assert(a.len == out.len);

        for (a, b, out) |av, bv, *ov| {
            ov.* = @max(av, bv);
        }
    }

    /// Element-wise absolute value: out[i] = |a[i]|
    pub fn abs(self: *Self, a: []const f32, out: []f32) void {
        _ = self;
        std.debug.assert(a.len == out.len);

        for (a, out) |av, *ov| {
            ov.* = @abs(av);
        }
    }

    /// Scalar multiplication: out[i] = a[i] * scalar
    pub fn mulScalar(self: *Self, a: []const f32, scalar: f32, out: []f32) void {
        _ = self;
        std.debug.assert(a.len == out.len);

        for (a, out) |av, *ov| {
            ov.* = av * scalar;
        }
    }

    /// Fused multiply-add: out[i] = a[i] * b[i] * scalar
    pub fn mulArraysScalar(self: *Self, a: []const f32, b: []const f32, scalar: f32, out: []f32) void {
        _ = self;
        std.debug.assert(a.len == b.len);
        std.debug.assert(a.len == out.len);

        for (a, b, out) |av, bv, *ov| {
            ov.* = av * bv * scalar;
        }
    }

    // =========================================================================
    // Reduction operations
    // =========================================================================

    /// Sum reduction
    pub fn sum(self: *Self, data: []const f32) f32 {
        if (data.len >= GPU_THRESHOLD and self.reduce_sum_pipeline != null) {
            if (self.gpuReduceSum(data)) |result| {
                return result;
            }
        }
        return cpuSum(data);
    }

    /// Minimum reduction
    pub fn reduceMin(self: *Self, data: []const f32) f32 {
        if (data.len >= GPU_THRESHOLD and self.reduce_min_pipeline != null) {
            if (self.gpuReduceMin(data)) |result| {
                return result;
            }
        }
        return cpuMin(data);
    }

    /// Maximum reduction
    pub fn reduceMax(self: *Self, data: []const f32) f32 {
        if (data.len >= GPU_THRESHOLD and self.reduce_max_pipeline != null) {
            if (self.gpuReduceMax(data)) |result| {
                return result;
            }
        }
        return cpuMax(data);
    }

    /// Count non-zero elements
    pub fn count(self: *Self, mask: []const f32) f32 {
        // Use sum for count (mask should be 0 or 1)
        return self.sum(mask);
    }

    // =========================================================================
    // GPU implementations
    // =========================================================================

    fn gpuReduceSum(self: *Self, data: []const f32) ?f32 {
        const ctx = self.ctx orelse return null;
        const pipeline = self.reduce_sum_pipeline orelse return null;
        const final_pipeline = self.reduce_sum_final_pipeline orelse return null;

        return self.gpuReduce(ctx, data, pipeline, final_pipeline);
    }

    fn gpuReduceMin(self: *Self, data: []const f32) ?f32 {
        const ctx = self.ctx orelse return null;
        const pipeline = self.reduce_min_pipeline orelse return null;
        const final_pipeline = self.reduce_min_final_pipeline orelse return null;

        return self.gpuReduce(ctx, data, pipeline, final_pipeline);
    }

    fn gpuReduceMax(self: *Self, data: []const f32) ?f32 {
        const ctx = self.ctx orelse return null;
        const pipeline = self.reduce_max_pipeline orelse return null;
        const final_pipeline = self.reduce_max_final_pipeline orelse return null;

        return self.gpuReduce(ctx, data, pipeline, final_pipeline);
    }

    fn gpuReduce(
        self: *Self,
        ctx: *GPUContext,
        data: []const f32,
        pipeline: *wgpu.ComputePipeline,
        final_pipeline: *wgpu.ComputePipeline,
    ) ?f32 {
        const n = data.len;
        const workgroup_size: u32 = 256;
        const num_workgroups: u32 = @intCast((n + workgroup_size - 1) / workgroup_size);

        // Params struct
        const ReduceParams = extern struct {
            size: u32,
            workgroups: u32,
        };

        // First pass params
        const params1 = ReduceParams{
            .size = @intCast(n),
            .workgroups = num_workgroups,
        };

        const params_buffer = ctx.createBufferWithData(
            ReduceParams,
            &[_]ReduceParams{params1},
            .{ .uniform = true, .copy_dst = true },
        ) catch return null;
        defer params_buffer.release();

        const input_buffer = ctx.createBufferWithData(
            f32,
            data,
            .{ .storage = true, .copy_dst = true },
        ) catch return null;
        defer input_buffer.release();

        // Partial results buffer (one per workgroup)
        const partial_buffer = ctx.createBuffer(
            @sizeOf(f32) * num_workgroups,
            .{ .storage = true, .copy_dst = true, .copy_src = true },
        ) catch return null;
        defer partial_buffer.release();

        // Output buffer for final result
        const output_buffer = ctx.createBuffer(
            @sizeOf(f32),
            .{ .storage = true, .copy_src = true },
        ) catch return null;
        defer output_buffer.release();

        // First pass: reduce within workgroups
        {
            const bind_group_layout = pipeline.getBindGroupLayout(0);
            defer bind_group_layout.release();

            const bind_group = ctx.device.createBindGroup(&.{
                .layout = bind_group_layout,
                .entry_count = 3,
                .entries = &[_]wgpu.BindGroupEntry{
                    .{ .binding = 0, .buffer = params_buffer, .size = @sizeOf(ReduceParams) },
                    .{ .binding = 1, .buffer = input_buffer, .size = @sizeOf(f32) * n },
                    .{ .binding = 2, .buffer = partial_buffer, .size = @sizeOf(f32) * num_workgroups },
                },
            }) orelse return null;
            defer bind_group.release();

            const encoder = ctx.device.createCommandEncoder(null) orelse return null;
            defer encoder.release();

            const pass = encoder.beginComputePass(null) orelse return null;
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bind_group, null);
            pass.dispatchWorkgroups(num_workgroups, 1, 1);
            pass.end();

            const cmd = encoder.finish(null) orelse return null;
            defer cmd.release();

            ctx.queue.submit(&[_]*wgpu.CommandBuffer{cmd});
        }

        // Final pass: reduce partial results
        if (num_workgroups > 1) {
            const params2 = ReduceParams{
                .size = num_workgroups,
                .workgroups = 1,
            };

            // Update params buffer
            ctx.queue.writeBuffer(params_buffer, 0, ReduceParams, &[_]ReduceParams{params2});

            const bind_group_layout = final_pipeline.getBindGroupLayout(0);
            defer bind_group_layout.release();

            const bind_group = ctx.device.createBindGroup(&.{
                .layout = bind_group_layout,
                .entry_count = 3,
                .entries = &[_]wgpu.BindGroupEntry{
                    .{ .binding = 0, .buffer = params_buffer, .size = @sizeOf(ReduceParams) },
                    .{ .binding = 1, .buffer = partial_buffer, .size = @sizeOf(f32) * num_workgroups },
                    .{ .binding = 2, .buffer = output_buffer, .size = @sizeOf(f32) },
                },
            }) orelse return null;
            defer bind_group.release();

            const encoder = ctx.device.createCommandEncoder(null) orelse return null;
            defer encoder.release();

            const pass = encoder.beginComputePass(null) orelse return null;
            pass.setPipeline(final_pipeline);
            pass.setBindGroup(0, bind_group, null);
            pass.dispatchWorkgroups(1, 1, 1);
            pass.end();

            const cmd = encoder.finish(null) orelse return null;
            defer cmd.release();

            ctx.queue.submit(&[_]*wgpu.CommandBuffer{cmd});
        }

        // Read back result
        const result_buffer = if (num_workgroups > 1) output_buffer else partial_buffer;
        var result: [1]f32 = undefined;
        self.readBuffer(result_buffer, &result) catch return null;

        return result[0];
    }

    fn readBuffer(self: *Self, buffer: *wgpu.Buffer, out: []f32) !void {
        const ctx = self.ctx orelse return error.GPUError;
        const size = @sizeOf(f32) * out.len;

        const staging = ctx.device.createBuffer(&.{
            .size = size,
            .usage = .{ .map_read = true, .copy_dst = true },
            .mapped_at_creation = .false,
        }) orelse return error.StagingBufferFailed;
        defer staging.release();

        const encoder = ctx.device.createCommandEncoder(null) orelse return error.EncoderCreationFailed;
        defer encoder.release();

        encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
        const cmd = encoder.finish(null) orelse return error.CommandBufferFailed;
        defer cmd.release();

        ctx.queue.submit(&[_]*wgpu.CommandBuffer{cmd});

        var status: wgpu.MapAsyncStatus = .unknown;
        staging.mapAsync(.{ .read = true }, 0, size, &status, struct {
            fn callback(s: *wgpu.MapAsyncStatus, new_status: wgpu.MapAsyncStatus) void {
                s.* = new_status;
            }
        }.callback);

        while (status == .unknown) {
            _ = ctx.device.poll(false, null);
        }

        if (status != .success) return error.MapFailed;

        const mapped = staging.getConstMappedRange(f32, 0, out.len) orelse return error.GetMappedRangeFailed;
        @memcpy(out, mapped);
        staging.unmap();
    }
};

// =============================================================================
// CPU fallback implementations (standalone functions)
// =============================================================================

fn cpuSum(data: []const f32) f32 {
    var result: f32 = 0;
    for (data) |v| {
        result += v;
    }
    return result;
}

fn cpuMin(data: []const f32) f32 {
    if (data.len == 0) return std.math.floatMax(f32);
    var result = data[0];
    for (data[1..]) |v| {
        result = @min(result, v);
    }
    return result;
}

fn cpuMax(data: []const f32) f32 {
    if (data.len == 0) return -std.math.floatMax(f32);
    var result = data[0];
    for (data[1..]) |v| {
        result = @max(result, v);
    }
    return result;
}

// =============================================================================
// Convenience functions (standalone, no GPU context needed)
// =============================================================================

/// Batch element-wise addition
pub fn batchAdd(a: []const f32, b: []const f32, out: []f32) void {
    std.debug.assert(a.len == b.len);
    std.debug.assert(a.len == out.len);
    for (a, b, out) |av, bv, *ov| {
        ov.* = av + bv;
    }
}

/// Batch element-wise subtraction
pub fn batchSub(a: []const f32, b: []const f32, out: []f32) void {
    std.debug.assert(a.len == b.len);
    std.debug.assert(a.len == out.len);
    for (a, b, out) |av, bv, *ov| {
        ov.* = av - bv;
    }
}

/// Batch element-wise multiplication
pub fn batchMul(a: []const f32, b: []const f32, out: []f32) void {
    std.debug.assert(a.len == b.len);
    std.debug.assert(a.len == out.len);
    for (a, b, out) |av, bv, *ov| {
        ov.* = av * bv;
    }
}

/// Batch element-wise division
pub fn batchDiv(a: []const f32, b: []const f32, out: []f32) void {
    std.debug.assert(a.len == b.len);
    std.debug.assert(a.len == out.len);
    for (a, b, out) |av, bv, *ov| {
        ov.* = if (bv != 0) av / bv else 0;
    }
}

/// Batch scalar multiplication
pub fn batchMulScalar(a: []const f32, scalar: f32, out: []f32) void {
    std.debug.assert(a.len == out.len);
    for (a, out) |av, *ov| {
        ov.* = av * scalar;
    }
}

/// Batch absolute value
pub fn batchAbs(a: []const f32, out: []f32) void {
    std.debug.assert(a.len == out.len);
    for (a, out) |av, *ov| {
        ov.* = @abs(av);
    }
}

/// Batch minimum
pub fn batchMin(a: []const f32, b: []const f32, out: []f32) void {
    std.debug.assert(a.len == b.len);
    std.debug.assert(a.len == out.len);
    for (a, b, out) |av, bv, *ov| {
        ov.* = @min(av, bv);
    }
}

/// Batch maximum
pub fn batchMax(a: []const f32, b: []const f32, out: []f32) void {
    std.debug.assert(a.len == b.len);
    std.debug.assert(a.len == out.len);
    for (a, b, out) |av, bv, *ov| {
        ov.* = @max(av, bv);
    }
}

// =============================================================================
// Tests
// =============================================================================

test "batch add" {
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    var out: [4]f32 = undefined;

    batchAdd(&a, &b, &out);

    try std.testing.expectApproxEqAbs(@as(f32, 6), out[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 8), out[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 10), out[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 12), out[3], 0.001);
}

test "batch mul scalar" {
    const a = [_]f32{ 1, 2, 3, 4 };
    var out: [4]f32 = undefined;

    batchMulScalar(&a, 2.5, &out);

    try std.testing.expectApproxEqAbs(@as(f32, 2.5), out[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), out[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 7.5), out[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), out[3], 0.001);
}

test "cpu sum" {
    const data = [_]f32{ 1, 2, 3, 4, 5 };
    const result = cpuSum(&data);
    try std.testing.expectApproxEqAbs(@as(f32, 15), result, 0.001);
}

test "cpu min" {
    const data = [_]f32{ 5, 2, 8, 1, 9 };
    const result = cpuMin(&data);
    try std.testing.expectApproxEqAbs(@as(f32, 1), result, 0.001);
}

test "cpu max" {
    const data = [_]f32{ 5, 2, 8, 1, 9 };
    const result = cpuMax(&data);
    try std.testing.expectApproxEqAbs(@as(f32, 9), result, 0.001);
}

test "GPUBatchOps element-wise" {
    const allocator = std.testing.allocator;
    const ops = try GPUBatchOps.init(allocator);
    defer ops.deinit();

    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    var out: [4]f32 = undefined;

    ops.add(&a, &b, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 6), out[0], 0.001);

    ops.mul(&a, &b, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 5), out[0], 0.001);
}

test "GPUBatchOps reductions" {
    const allocator = std.testing.allocator;
    const ops = try GPUBatchOps.init(allocator);
    defer ops.deinit();

    const data = [_]f32{ 1, 2, 3, 4, 5 };

    try std.testing.expectApproxEqAbs(@as(f32, 15), ops.sum(&data), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1), ops.reduceMin(&data), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5), ops.reduceMax(&data), 0.001);
}
