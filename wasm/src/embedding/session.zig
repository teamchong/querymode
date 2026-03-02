//! ONNX Runtime Session Wrapper
//!
//! Provides a high-level interface for loading and running ONNX models.

const std = @import("std");
const Allocator = std.mem.Allocator;
const onnx = @import("onnx.zig");

pub const Session = struct {
    allocator: Allocator,
    api: *const onnx.OrtApi,
    env: *onnx.OrtEnv,
    session: *onnx.OrtSession,
    memory_info: *onnx.OrtMemoryInfo,
    ort_allocator: *onnx.OrtAllocator,

    input_names: []const [*:0]const u8,
    output_names: []const [*:0]const u8,
    input_count: usize,
    output_count: usize,

    const Self = @This();

    /// Initialize an ONNX session with the given model file
    pub fn init(allocator: Allocator, model_path: []const u8) !Self {
        // Get the ONNX API
        const api = onnx.getApi() orelse return onnx.OrtError.InvalidApi;

        // Create environment
        var env: ?*onnx.OrtEnv = null;
        try onnx.checkStatus(api, api.CreateEnv(.warning, "lanceql", &env));
        errdefer if (env) |e| api.ReleaseEnv(e);

        // Create session options
        var session_opts: ?*onnx.OrtSessionOptions = null;
        try onnx.checkStatus(api, api.CreateSessionOptions(&session_opts));
        defer if (session_opts) |opts| api.ReleaseSessionOptions(opts);

        // Set optimization level to all optimizations
        try onnx.checkStatus(api, api.SetSessionGraphOptimizationLevel(session_opts.?, 99));

        // Set thread count (use available cores)
        try onnx.checkStatus(api, api.SetIntraOpNumThreads(session_opts.?, 0));

        // Create null-terminated path
        const path_z = try allocator.dupeZ(u8, model_path);
        defer allocator.free(path_z);

        // Create session
        var session: ?*onnx.OrtSession = null;
        try onnx.checkStatus(api, api.CreateSession(env.?, path_z, session_opts.?, &session));
        errdefer if (session) |s| api.ReleaseSession(s);

        // Create CPU memory info
        var memory_info: ?*onnx.OrtMemoryInfo = null;
        try onnx.checkStatus(api, api.CreateCpuMemoryInfo(.arena, .default, &memory_info));
        errdefer if (memory_info) |m| api.ReleaseMemoryInfo(m);

        // Get allocator
        var ort_allocator: ?*onnx.OrtAllocator = null;
        try onnx.checkStatus(api, api.GetAllocatorWithDefaultOptions(&ort_allocator));

        // Get input/output counts
        var input_count: usize = 0;
        var output_count: usize = 0;
        try onnx.checkStatus(api, api.SessionGetInputCount(session.?, &input_count));
        try onnx.checkStatus(api, api.SessionGetOutputCount(session.?, &output_count));

        // Get input names
        const input_names = try allocator.alloc([*:0]const u8, input_count);
        errdefer allocator.free(input_names);
        for (0..input_count) |i| {
            var name: ?[*:0]u8 = null;
            try onnx.checkStatus(api, api.SessionGetInputName(session.?, i, ort_allocator.?, &name));
            input_names[i] = name.?;
        }

        // Get output names
        const output_names = try allocator.alloc([*:0]const u8, output_count);
        errdefer allocator.free(output_names);
        for (0..output_count) |i| {
            var name: ?[*:0]u8 = null;
            try onnx.checkStatus(api, api.SessionGetOutputName(session.?, i, ort_allocator.?, &name));
            output_names[i] = name.?;
        }

        return Self{
            .allocator = allocator,
            .api = api,
            .env = env.?,
            .session = session.?,
            .memory_info = memory_info.?,
            .ort_allocator = ort_allocator.?,
            .input_names = input_names,
            .output_names = output_names,
            .input_count = input_count,
            .output_count = output_count,
        };
    }

    /// Release all ONNX resources
    pub fn deinit(self: *Self) void {
        // Free input/output names
        for (self.input_names) |name| {
            self.api.AllocatorFree(self.ort_allocator, @ptrCast(@constCast(name)));
        }
        self.allocator.free(self.input_names);

        for (self.output_names) |name| {
            self.api.AllocatorFree(self.ort_allocator, @ptrCast(@constCast(name)));
        }
        self.allocator.free(self.output_names);

        // Release ONNX objects
        self.api.ReleaseMemoryInfo(self.memory_info);
        self.api.ReleaseSession(self.session);
        self.api.ReleaseEnv(self.env);
    }

    /// Create a tensor from float data
    pub fn createFloatTensor(self: *Self, data: []f32, shape: []const i64) !*onnx.OrtValue {
        var tensor: ?*onnx.OrtValue = null;
        try onnx.checkStatus(self.api, self.api.CreateTensorWithDataAsOrtValue(
            self.memory_info,
            @ptrCast(data.ptr),
            data.len * @sizeOf(f32),
            shape.ptr,
            shape.len,
            .float,
            &tensor,
        ));
        return tensor.?;
    }

    /// Create a tensor from i64 data
    pub fn createInt64Tensor(self: *Self, data: []i64, shape: []const i64) !*onnx.OrtValue {
        var tensor: ?*onnx.OrtValue = null;
        try onnx.checkStatus(self.api, self.api.CreateTensorWithDataAsOrtValue(
            self.memory_info,
            @ptrCast(data.ptr),
            data.len * @sizeOf(i64),
            shape.ptr,
            shape.len,
            .int64,
            &tensor,
        ));
        return tensor.?;
    }

    /// Run inference with the given inputs
    /// Returns output tensors that must be released by the caller
    pub fn run(self: *Self, inputs: []const ?*const onnx.OrtValue) ![]?*onnx.OrtValue {
        if (inputs.len != self.input_count) {
            return onnx.OrtError.RunFailed;
        }

        // Allocate output array
        const outputs = try self.allocator.alloc(?*onnx.OrtValue, self.output_count);
        @memset(outputs, null);
        errdefer {
            for (outputs) |out| {
                if (out) |o| self.api.ReleaseValue(o);
            }
            self.allocator.free(outputs);
        }

        // Run inference
        try onnx.checkStatus(self.api, self.api.Run(
            self.session,
            null,
            self.input_names.ptr,
            inputs.ptr,
            inputs.len,
            self.output_names.ptr,
            self.output_count,
            outputs.ptr,
        ));

        return outputs;
    }

    /// Get float data from output tensor
    pub fn getFloatData(self: *Self, tensor: *onnx.OrtValue) ![]f32 {
        // Get shape info
        var shape_info: ?*onnx.OrtTensorTypeAndShapeInfo = null;
        try onnx.checkStatus(self.api, self.api.GetTensorTypeAndShape(tensor, &shape_info));
        defer if (shape_info) |s| self.api.ReleaseTensorTypeAndShapeInfo(s);

        // Get dimensions
        var dim_count: usize = 0;
        try onnx.checkStatus(self.api, self.api.GetDimensionsCount(shape_info.?, &dim_count));

        const dims = try self.allocator.alloc(i64, dim_count);
        defer self.allocator.free(dims);
        try onnx.checkStatus(self.api, self.api.GetDimensions(shape_info.?, dims.ptr, dim_count));

        // Calculate total elements
        var total_elements: usize = 1;
        for (dims) |d| {
            total_elements *= @intCast(d);
        }

        // Get data pointer
        var data_ptr: ?*anyopaque = null;
        try onnx.checkStatus(self.api, self.api.GetTensorMutableData(tensor, &data_ptr));

        // Return slice - note: this borrows from the tensor, caller must not release tensor before using
        const float_ptr: [*]f32 = @ptrCast(@alignCast(data_ptr.?));
        return float_ptr[0..total_elements];
    }

    /// Release a tensor
    pub fn releaseTensor(self: *Self, tensor: *onnx.OrtValue) void {
        self.api.ReleaseValue(tensor);
    }

    /// Release output tensors
    pub fn releaseOutputs(self: *Self, outputs: []?*onnx.OrtValue) void {
        for (outputs) |out| {
            if (out) |o| self.api.ReleaseValue(o);
        }
        self.allocator.free(outputs);
    }

    /// Get input names for debugging
    pub fn getInputNames(self: *Self) []const [*:0]const u8 {
        return self.input_names;
    }

    /// Get output names for debugging
    pub fn getOutputNames(self: *Self) []const [*:0]const u8 {
        return self.output_names;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "session api availability" {
    // This test only checks if the API structure is valid
    // Actual model loading requires ONNX runtime to be installed
    if (onnx.getApi()) |api| {
        _ = api;
        std.debug.print("ONNX Runtime API available\n", .{});
    } else {
        std.debug.print("ONNX Runtime not available\n", .{});
    }
}
