//! GPU Context - wgpu-native based cross-platform GPU acceleration
//!
//! Provides unified GPU compute via wgpu-native, replacing Metal-only backend.
//! Uses same WGSL shaders as browser for code sharing.

const std = @import("std");
const wgpu = @import("wgpu");

const Allocator = std.mem.Allocator;

/// GPU compute context - manages device, queue, and pipelines
pub const GPUContext = struct {
    allocator: Allocator,
    instance: *wgpu.Instance,
    adapter: *wgpu.Adapter,
    device: *wgpu.Device,
    queue: *wgpu.Queue,
    pipelines: std.StringHashMap(*wgpu.ComputePipeline),
    shader_modules: std.StringHashMap(*wgpu.ShaderModule),

    /// Initialize GPU context
    pub fn init(allocator: Allocator) !*GPUContext {
        const instance = wgpu.Instance.create(null) orelse {
            return error.InstanceCreationFailed;
        };

        const adapter_response = instance.requestAdapterSync(null, 200_000_000);
        // Check the adapter
        const adapter = adapter_response.adapter orelse return error.AdapterRequestFailed;

        const device_response = adapter.requestDeviceSync(instance, null, 200_000_000);
        // Check the device
        const device = device_response.device orelse return error.DeviceRequestFailed;

        const queue = device.getQueue() orelse return error.QueueCreationFailed;

        const ctx = try allocator.create(GPUContext);
        ctx.* = .{
            .allocator = allocator,
            .instance = instance,
            .adapter = adapter,
            .device = device,
            .queue = queue,
            .pipelines = std.StringHashMap(*wgpu.ComputePipeline).init(allocator),
            .shader_modules = std.StringHashMap(*wgpu.ShaderModule).init(allocator),
        };

        return ctx;
    }

    /// Load and compile a WGSL shader
    pub fn loadShader(self: *GPUContext, name: []const u8, source: []const u8) !*wgpu.ShaderModule {
        if (self.shader_modules.get(name)) |existing| {
            return existing;
        }

        const module = self.device.createShaderModule(&.{
            .next_in_chain = null,
            .label = wgpu.StringView.fromSlice(name),
            .source = .{ .wgsl = .{ .code = wgpu.StringView.fromSlice(source) } },
        }) orelse return error.ShaderCompilationFailed;

        try self.shader_modules.put(name, module);
        return module;
    }

    /// Create a compute pipeline
    pub fn createPipeline(
        self: *GPUContext,
        name: []const u8,
        shader: *wgpu.ShaderModule,
        entry_point: []const u8,
    ) !*wgpu.ComputePipeline {
        if (self.pipelines.get(name)) |existing| {
            return existing;
        }

        const pipeline = self.device.createComputePipeline(&.{
            .next_in_chain = null,
            .label = wgpu.StringView.fromSlice(name),
            .layout = null, // Auto layout
            .compute = .{
                .next_in_chain = null,
                .module = shader,
                .entry_point = wgpu.StringView.fromSlice(entry_point),
                .constants = null,
                .constant_count = 0,
            },
        }) orelse return error.PipelineCreationFailed;

        try self.pipelines.put(name, pipeline);
        return pipeline;
    }

    /// Create a GPU buffer
    pub fn createBuffer(self: *GPUContext, size: u64, usage: wgpu.BufferUsage) !*wgpu.Buffer {
        return self.device.createBuffer(&.{
            .next_in_chain = null,
            .label = null,
            .size = size,
            .usage = usage,
            .mapped_at_creation = .false,
        }) orelse return error.BufferCreationFailed;
    }

    /// Create a GPU buffer with initial data
    pub fn createBufferWithData(
        self: *GPUContext,
        comptime T: type,
        data: []const T,
        usage: wgpu.BufferUsage,
    ) !*wgpu.Buffer {
        const size = @sizeOf(T) * data.len;
        const buffer = self.device.createBuffer(&.{
            .next_in_chain = null,
            .label = null,
            .size = size,
            .usage = usage,
            .mapped_at_creation = .true,
        }) orelse return error.BufferCreationFailed;

        // Copy data to mapped buffer
        const mapped = buffer.getMappedRange(T, 0, data.len) orelse {
            buffer.release();
            return error.BufferMappingFailed;
        };
        @memcpy(mapped, data);
        buffer.unmap();

        return buffer;
    }

    /// Write data to a buffer
    pub fn writeBuffer(self: *GPUContext, buffer: *wgpu.Buffer, comptime T: type, data: []const T) void {
        self.queue.writeBuffer(buffer, 0, T, data);
    }

    /// Submit compute work
    pub fn submit(self: *GPUContext, commands: []const *wgpu.CommandBuffer) void {
        self.queue.submit(commands);
    }

    /// Wait for all work to complete
    pub fn waitForCompletion(self: *GPUContext) void {
        // Submit empty to flush and wait
        self.queue.submit(&.{});
        // wgpu-native handles synchronization
    }

    /// Release all resources
    pub fn deinit(self: *GPUContext) void {
        var pipeline_iter = self.pipelines.valueIterator();
        while (pipeline_iter.next()) |pipeline| {
            pipeline.*.release();
        }
        self.pipelines.deinit();

        var shader_iter = self.shader_modules.valueIterator();
        while (shader_iter.next()) |shader| {
            shader.*.release();
        }
        self.shader_modules.deinit();

        self.queue.release();
        self.device.release();
        self.adapter.release();
        self.instance.release();

        self.allocator.destroy(self);
    }

    /// Check if GPU is available
    pub fn isAvailable() bool {
        const instance = wgpu.Instance.create(null) orelse return false;
        defer instance.release();

        const response = instance.requestAdapterSync(null, 100_000_000);
        return response.status == .success and response.adapter != null;
    }

    /// Get device info
    pub fn getDeviceInfo(self: *GPUContext) struct { vendor: []const u8, name: []const u8 } {
        var info: wgpu.AdapterInfo = undefined;
        self.adapter.getInfo(&info);
        return .{
            .vendor = info.vendor.toSlice() orelse "unknown",
            .name = info.device.toSlice() orelse "unknown",
        };
    }
};

/// Global GPU context singleton
var global_context: ?*GPUContext = null;
var global_mutex: std.Thread.Mutex = .{};

/// Get or create global GPU context
pub fn getGlobalContext(allocator: Allocator) !*GPUContext {
    global_mutex.lock();
    defer global_mutex.unlock();

    if (global_context) |ctx| {
        return ctx;
    }

    global_context = try GPUContext.init(allocator);
    return global_context.?;
}

/// Release global context
pub fn releaseGlobalContext() void {
    global_mutex.lock();
    defer global_mutex.unlock();

    if (global_context) |ctx| {
        ctx.deinit();
        global_context = null;
    }
}

/// Check if GPU is available (without initializing)
pub fn isGPUAvailable() bool {
    return GPUContext.isAvailable();
}

// Embedded WGSL shaders from shared directory
pub const shaders = struct {
    pub const join = @embedFile("../../packages/shared/gpu/shaders/join.wgsl");
    pub const group_by = @embedFile("../../packages/shared/gpu/shaders/group_by.wgsl");
    pub const sort = @embedFile("../../packages/shared/gpu/shaders/sort.wgsl");
    pub const reduce = @embedFile("../../packages/shared/gpu/shaders/reduce.wgsl");
    pub const topk_select = @embedFile("../../packages/shared/gpu/shaders/topk_select.wgsl");
    pub const vector_distance = @embedFile("../../packages/shared/gpu/shaders/vector_distance.wgsl");
};

test "GPU availability check" {
    const available = isGPUAvailable();
    std.debug.print("GPU available: {}\n", .{available});
}

test "GPU context creation" {
    const allocator = std.testing.allocator;

    const ctx = GPUContext.init(allocator) catch |err| {
        std.debug.print("GPU init failed (expected on CI): {}\n", .{err});
        return;
    };
    defer ctx.deinit();

    const info = ctx.getDeviceInfo();
    std.debug.print("GPU: {s} ({s})\n", .{ info.name, info.vendor });
}
