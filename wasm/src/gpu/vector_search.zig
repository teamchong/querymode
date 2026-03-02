//! GPU-Accelerated Vector Search
//!
//! Provides GPU-based distance computation and top-K selection for vector search.
//! Uses shared WGSL shaders from packages/shared/gpu/shaders/.

const std = @import("std");
const wgpu = @import("wgpu");
const gpu_context = @import("gpu_context.zig");
const GPUContext = gpu_context.GPUContext;

const Allocator = std.mem.Allocator;

/// Distance metrics
pub const DistanceMetric = enum(u32) {
    cosine = 0,
    l2 = 1,
    dot_product = 2,
};

/// Vector search result
pub const SearchResult = struct {
    indices: []u32,
    scores: []f32,
    allocator: Allocator,

    pub fn deinit(self: *SearchResult) void {
        self.allocator.free(self.indices);
        self.allocator.free(self.scores);
    }
};

/// GPU-accelerated vector search
pub const GPUVectorSearch = struct {
    allocator: Allocator,
    ctx: *GPUContext,
    distance_pipeline: ?*wgpu.ComputePipeline = null,
    topk_pipeline: ?*wgpu.ComputePipeline = null,

    pub fn init(allocator: Allocator) !*GPUVectorSearch {
        const ctx = try gpu_context.getGlobalContext(allocator);

        const self = try allocator.create(GPUVectorSearch);
        self.* = .{
            .allocator = allocator,
            .ctx = ctx,
        };

        try self.compilePipelines();
        return self;
    }

    fn compilePipelines(self: *GPUVectorSearch) !void {
        // Load vector distance shader
        const distance_shader = try self.ctx.loadShader(
            "vector_distance",
            gpu_context.shaders.vector_distance,
        );
        self.distance_pipeline = try self.ctx.createPipeline(
            "compute_distances",
            distance_shader,
            "compute_distances",
        );

        // Load top-K shader
        const topk_shader = try self.ctx.loadShader(
            "topk_select",
            gpu_context.shaders.topk_select,
        );
        self.topk_pipeline = try self.ctx.createPipeline(
            "local_topk",
            topk_shader,
            "local_topk",
        );
    }

    /// Compute distances between query and vectors (GPU)
    pub fn computeDistances(
        self: *GPUVectorSearch,
        query: []const f32,
        vectors: []const f32,
        dim: usize,
        metric: DistanceMetric,
    ) ![]f32 {
        const num_vectors = vectors.len / dim;

        // For small datasets, use CPU
        if (num_vectors < 1000) {
            return self.cpuComputeDistances(query, vectors, dim, metric);
        }

        // GPU path
        const distances = try self.allocator.alloc(f32, num_vectors);
        errdefer self.allocator.free(distances);

        try self.gpuComputeDistances(query, vectors, dim, metric, distances);
        return distances;
    }

    fn gpuComputeDistances(
        self: *GPUVectorSearch,
        query: []const f32,
        vectors: []const f32,
        dim: usize,
        metric: DistanceMetric,
        out_distances: []f32,
    ) !void {
        const num_vectors = vectors.len / dim;

        // Create uniform buffer for params
        const Params = extern struct {
            dim: u32,
            num_vectors: u32,
            num_queries: u32,
            metric: u32,
        };
        const params = Params{
            .dim = @intCast(dim),
            .num_vectors = @intCast(num_vectors),
            .num_queries = 1,
            .metric = @intFromEnum(metric),
        };

        const params_buffer = try self.ctx.createBufferWithData(
            Params,
            &[_]Params{params},
            .{ .uniform = true, .copy_dst = true },
        );
        defer params_buffer.release();

        const query_buffer = try self.ctx.createBufferWithData(
            f32,
            query,
            .{ .storage = true, .copy_dst = true },
        );
        defer query_buffer.release();

        const vectors_buffer = try self.ctx.createBufferWithData(
            f32,
            vectors,
            .{ .storage = true, .copy_dst = true },
        );
        defer vectors_buffer.release();

        const distances_buffer = try self.ctx.createBuffer(
            @sizeOf(f32) * num_vectors,
            .{ .storage = true, .copy_src = true },
        );
        defer distances_buffer.release();

        // Create bind group
        const pipeline = self.distance_pipeline orelse return error.PipelineNotReady;
        const bind_group_layout = pipeline.getBindGroupLayout(0);
        defer bind_group_layout.release();

        const bind_group = self.ctx.device.createBindGroup(&.{
            .layout = bind_group_layout,
            .entry_count = 4,
            .entries = &[_]wgpu.BindGroupEntry{
                .{ .binding = 0, .buffer = params_buffer, .size = @sizeOf(Params) },
                .{ .binding = 1, .buffer = query_buffer, .size = @sizeOf(f32) * query.len },
                .{ .binding = 2, .buffer = vectors_buffer, .size = @sizeOf(f32) * vectors.len },
                .{ .binding = 3, .buffer = distances_buffer, .size = @sizeOf(f32) * num_vectors },
            },
        }) orelse return error.BindGroupCreationFailed;
        defer bind_group.release();

        // Dispatch compute
        const encoder = self.ctx.device.createCommandEncoder(null) orelse return error.EncoderCreationFailed;
        defer encoder.release();

        const pass = encoder.beginComputePass(null) orelse return error.ComputePassFailed;
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind_group, null);
        pass.dispatchWorkgroups(@intCast((num_vectors + 255) / 256), 1, 1);
        pass.end();

        const command_buffer = encoder.finish(null) orelse return error.CommandBufferFailed;
        defer command_buffer.release();

        self.ctx.queue.submit(&[_]*wgpu.CommandBuffer{command_buffer});

        // Read back results
        try self.readBuffer(f32, distances_buffer, out_distances);
    }

    fn readBuffer(self: *GPUVectorSearch, comptime T: type, buffer: *wgpu.Buffer, out: []T) !void {
        const size = @sizeOf(T) * out.len;

        // Create staging buffer for readback
        const staging = self.ctx.device.createBuffer(&.{
            .size = size,
            .usage = .{ .map_read = true, .copy_dst = true },
            .mapped_at_creation = .false,
        }) orelse return error.StagingBufferFailed;
        defer staging.release();

        // Copy to staging
        const encoder = self.ctx.device.createCommandEncoder(null) orelse return error.EncoderCreationFailed;
        defer encoder.release();

        encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
        const cmd = encoder.finish(null) orelse return error.CommandBufferFailed;
        defer cmd.release();

        self.ctx.queue.submit(&[_]*wgpu.CommandBuffer{cmd});

        // Map and read
        var status: wgpu.MapAsyncStatus = .unknown;
        staging.mapAsync(.{ .read = true }, 0, size, &status, struct {
            fn callback(s: *wgpu.MapAsyncStatus, new_status: wgpu.MapAsyncStatus) void {
                s.* = new_status;
            }
        }.callback);

        // Poll until mapped
        while (status == .unknown) {
            _ = self.ctx.device.poll(false, null);
        }

        if (status != .success) return error.MapFailed;

        const mapped = staging.getConstMappedRange(T, 0, out.len) orelse return error.GetMappedRangeFailed;
        @memcpy(out, mapped);
        staging.unmap();
    }

    /// CPU fallback for distance computation
    fn cpuComputeDistances(
        self: *GPUVectorSearch,
        query: []const f32,
        vectors: []const f32,
        dim: usize,
        metric: DistanceMetric,
    ) ![]f32 {
        const num_vectors = vectors.len / dim;
        const distances = try self.allocator.alloc(f32, num_vectors);

        for (0..num_vectors) |i| {
            const vec = vectors[i * dim .. (i + 1) * dim];
            distances[i] = switch (metric) {
                .cosine, .dot_product => blk: {
                    var dot: f32 = 0;
                    for (0..dim) |j| {
                        dot += query[j] * vec[j];
                    }
                    break :blk dot;
                },
                .l2 => blk: {
                    var sum: f32 = 0;
                    for (0..dim) |j| {
                        const d = query[j] - vec[j];
                        sum += d * d;
                    }
                    break :blk @sqrt(sum);
                },
            };
        }

        return distances;
    }

    /// Find top-K from scores
    pub fn topK(
        self: *GPUVectorSearch,
        scores: []const f32,
        k: usize,
        descending: bool,
    ) !SearchResult {
        const n = scores.len;
        const actual_k = @min(k, n);

        // CPU implementation for now (GPU top-K requires more complex shader setup)
        return self.cpuTopK(scores, actual_k, descending);
    }

    fn cpuTopK(
        self: *GPUVectorSearch,
        scores: []const f32,
        k: usize,
        descending: bool,
    ) !SearchResult {
        const n = scores.len;

        // Create index-score pairs
        const Pair = struct { index: u32, score: f32 };
        const pairs = try self.allocator.alloc(Pair, n);
        defer self.allocator.free(pairs);

        for (0..n) |i| {
            pairs[i] = .{ .index = @intCast(i), .score = scores[i] };
        }

        // Sort
        if (descending) {
            std.mem.sort(Pair, pairs, {}, struct {
                fn cmp(_: void, a: Pair, b: Pair) bool {
                    return a.score > b.score;
                }
            }.cmp);
        } else {
            std.mem.sort(Pair, pairs, {}, struct {
                fn cmp(_: void, a: Pair, b: Pair) bool {
                    return a.score < b.score;
                }
            }.cmp);
        }

        // Extract top K
        const indices = try self.allocator.alloc(u32, k);
        const result_scores = try self.allocator.alloc(f32, k);

        for (0..k) |i| {
            indices[i] = pairs[i].index;
            result_scores[i] = pairs[i].score;
        }

        return .{
            .indices = indices,
            .scores = result_scores,
            .allocator = self.allocator,
        };
    }

    /// Full vector search: compute distances + top-K
    pub fn search(
        self: *GPUVectorSearch,
        query: []const f32,
        vectors: []const f32,
        dim: usize,
        k: usize,
        metric: DistanceMetric,
    ) !SearchResult {
        const distances = try self.computeDistances(query, vectors, dim, metric);
        defer self.allocator.free(distances);

        const descending = metric == .cosine or metric == .dot_product;
        return self.topK(distances, k, descending);
    }

    pub fn deinit(self: *GPUVectorSearch) void {
        self.allocator.destroy(self);
    }
};

/// Batch cosine similarity (compatibility function for existing code)
pub fn batchCosineSimilarity(
    allocator: Allocator,
    query: []const f32,
    vectors: []const f32,
    dim: usize,
) ![]f32 {
    // Use CPU for simplicity in compatibility layer
    const num_vectors = vectors.len / dim;
    const scores = try allocator.alloc(f32, num_vectors);

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

    return scores;
}

/// Dot product (single vector pair)
pub fn dotProduct(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var sum: f32 = 0;
    for (0..a.len) |i| {
        sum += a[i] * b[i];
    }
    return sum;
}

/// Cosine similarity (single vector pair)
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var dot: f32 = 0;
    var norm_a: f32 = 0;
    var norm_b: f32 = 0;

    for (0..a.len) |i| {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    const denom = @sqrt(norm_a) * @sqrt(norm_b);
    return if (denom > 0) dot / denom else 0;
}

/// L2 distance squared
pub fn l2DistanceSquared(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var sum: f32 = 0;
    for (0..a.len) |i| {
        const d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

test "cosine similarity" {
    const a = [_]f32{ 1, 0, 0 };
    const b = [_]f32{ 1, 0, 0 };
    const c = [_]f32{ 0, 1, 0 };

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cosineSimilarity(&a, &b), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cosineSimilarity(&a, &c), 0.001);
}

test "dot product" {
    const a = [_]f32{ 1, 2, 3 };
    const b = [_]f32{ 4, 5, 6 };

    try std.testing.expectApproxEqAbs(@as(f32, 32.0), dotProduct(&a, &b), 0.001);
}
