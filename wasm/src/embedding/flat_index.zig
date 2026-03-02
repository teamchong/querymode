//! Flat Vector Index
//!
//! Simple exhaustive search index for exact k-NN queries.
//! Computes distances to all vectors and returns top-k.
//!
//! Use cases:
//! - Small datasets (< 100k vectors)
//! - Exact search required
//! - Baseline for accuracy comparison

const std = @import("std");

/// Distance metrics supported by the index
pub const DistanceMetric = enum {
    /// L2 (Euclidean) distance - lower is more similar
    l2,
    /// Cosine similarity - higher is more similar (internally uses 1 - cosine)
    cosine,
    /// Inner product - higher is more similar (internally uses -dot)
    inner_product,
};

/// Search result: index and distance
pub const SearchResult = struct {
    index: usize,
    distance: f32,
};

/// Flat vector index for exact k-NN search
pub fn FlatIndex(comptime DIM: usize) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        vectors: std.ArrayListUnmanaged([DIM]f32),
        metric: DistanceMetric,

        /// Initialize an empty index
        pub fn init(allocator: std.mem.Allocator, metric: DistanceMetric) Self {
            return Self{
                .allocator = allocator,
                .vectors = std.ArrayListUnmanaged([DIM]f32){},
                .metric = metric,
            };
        }

        /// Initialize from existing vectors
        pub fn initWithVectors(allocator: std.mem.Allocator, vectors: []const [DIM]f32, metric: DistanceMetric) !Self {
            var self = Self.init(allocator, metric);
            try self.vectors.appendSlice(allocator, vectors);
            return self;
        }

        /// Deinitialize and free memory
        pub fn deinit(self: *Self) void {
            self.vectors.deinit(self.allocator);
        }

        /// Add a single vector to the index
        pub fn add(self: *Self, vector: [DIM]f32) !void {
            try self.vectors.append(self.allocator, vector);
        }

        /// Add multiple vectors to the index
        pub fn addBatch(self: *Self, vectors: []const [DIM]f32) !void {
            try self.vectors.appendSlice(self.allocator, vectors);
        }

        /// Number of vectors in the index
        pub fn count(self: *const Self) usize {
            return self.vectors.items.len;
        }

        /// Compute L2 squared distance between two vectors
        fn l2DistanceSquared(a: [DIM]f32, b: [DIM]f32) f32 {
            var sum: f32 = 0.0;
            for (a, b) |va, vb| {
                const diff = va - vb;
                sum += diff * diff;
            }
            return sum;
        }

        /// Compute cosine distance (1 - cosine similarity)
        fn cosineDistance(a: [DIM]f32, b: [DIM]f32) f32 {
            var dot: f32 = 0.0;
            var norm_a: f32 = 0.0;
            var norm_b: f32 = 0.0;

            for (a, b) |va, vb| {
                dot += va * vb;
                norm_a += va * va;
                norm_b += vb * vb;
            }

            const denom = @sqrt(norm_a) * @sqrt(norm_b);
            if (denom == 0.0) return 1.0; // Maximum distance if zero vector

            const cosine_sim = dot / denom;
            return 1.0 - cosine_sim;
        }

        /// Compute negative inner product (for max inner product search)
        fn negInnerProduct(a: [DIM]f32, b: [DIM]f32) f32 {
            var dot: f32 = 0.0;
            for (a, b) |va, vb| {
                dot += va * vb;
            }
            return -dot;
        }

        /// Compute distance based on configured metric
        fn computeDistance(self: *const Self, a: [DIM]f32, b: [DIM]f32) f32 {
            return switch (self.metric) {
                .l2 => l2DistanceSquared(a, b),
                .cosine => cosineDistance(a, b),
                .inner_product => negInnerProduct(a, b),
            };
        }

        /// Search for k nearest neighbors
        /// Returns results sorted by distance (ascending)
        pub fn search(self: *const Self, query: [DIM]f32, k: usize, results_buf: []SearchResult) ![]SearchResult {
            if (self.vectors.items.len == 0) return results_buf[0..0];

            const actual_k = @min(k, self.vectors.items.len);
            if (results_buf.len < actual_k) return error.BufferTooSmall;

            // For small k, use a simple insertion-based approach
            // For larger k, a heap would be more efficient
            var result_count: usize = 0;

            for (self.vectors.items, 0..) |vec, idx| {
                const dist = self.computeDistance(query, vec);

                // Find insertion position
                var insert_pos: usize = result_count;
                while (insert_pos > 0 and dist < results_buf[insert_pos - 1].distance) {
                    insert_pos -= 1;
                }

                // Insert if within top-k
                if (insert_pos < actual_k) {
                    // Shift elements to make room
                    if (result_count < actual_k) {
                        var i = result_count;
                        while (i > insert_pos) : (i -= 1) {
                            results_buf[i] = results_buf[i - 1];
                        }
                        result_count += 1;
                    } else {
                        var i = actual_k - 1;
                        while (i > insert_pos) : (i -= 1) {
                            results_buf[i] = results_buf[i - 1];
                        }
                    }

                    results_buf[insert_pos] = SearchResult{
                        .index = idx,
                        .distance = dist,
                    };
                }
            }

            return results_buf[0..result_count];
        }

        /// Serialize index to bytes for storage
        pub fn serialize(self: *const Self, allocator: std.mem.Allocator) ![]u8 {
            // Format: [metric:u8][count:u64][vectors...]
            const header_size = 1 + 8;
            const vector_bytes = DIM * @sizeOf(f32);
            const total_size = header_size + self.vectors.items.len * vector_bytes;

            var data = try allocator.alloc(u8, total_size);

            // Write metric
            data[0] = @intFromEnum(self.metric);

            // Write count
            std.mem.writeInt(u64, data[1..9], self.vectors.items.len, .little);

            // Write vectors
            var offset: usize = header_size;
            for (self.vectors.items) |vec| {
                const bytes: *const [vector_bytes]u8 = @ptrCast(&vec);
                @memcpy(data[offset..][0..vector_bytes], bytes);
                offset += vector_bytes;
            }

            return data;
        }

        /// Deserialize index from bytes
        pub fn deserialize(allocator: std.mem.Allocator, data: []const u8) !Self {
            if (data.len < 9) return error.InvalidData;

            const metric: DistanceMetric = @enumFromInt(data[0]);
            const vec_count = std.mem.readInt(u64, data[1..9], .little);

            const vector_bytes = DIM * @sizeOf(f32);
            const expected_size = 9 + vec_count * vector_bytes;
            if (data.len < expected_size) return error.InvalidData;

            var self = Self.init(allocator, metric);
            try self.vectors.ensureTotalCapacity(allocator, vec_count);

            var offset: usize = 9;
            for (0..vec_count) |_| {
                const bytes = data[offset..][0..vector_bytes];
                const vec: *const [DIM]f32 = @ptrCast(@alignCast(bytes.ptr));
                try self.vectors.append(allocator, vec.*);
                offset += vector_bytes;
            }

            return self;
        }
    };
}

// =============================================================================
// Tests
// =============================================================================

test "flat index basic search" {
    const allocator = std.testing.allocator;

    var index = FlatIndex(4).init(allocator, .l2);
    defer index.deinit();

    // Add some vectors
    try index.add(.{ 1.0, 0.0, 0.0, 0.0 });
    try index.add(.{ 0.0, 1.0, 0.0, 0.0 });
    try index.add(.{ 0.0, 0.0, 1.0, 0.0 });
    try index.add(.{ 0.5, 0.5, 0.0, 0.0 });

    try std.testing.expectEqual(@as(usize, 4), index.count());

    // Search for nearest to [1, 0, 0, 0]
    var results: [3]SearchResult = undefined;
    const found = try index.search(.{ 1.0, 0.0, 0.0, 0.0 }, 3, &results);

    try std.testing.expectEqual(@as(usize, 3), found.len);
    try std.testing.expectEqual(@as(usize, 0), found[0].index); // Exact match
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), found[0].distance, 0.001);
}

test "flat index cosine distance" {
    const allocator = std.testing.allocator;

    var index = FlatIndex(3).init(allocator, .cosine);
    defer index.deinit();

    // Add normalized vectors
    try index.add(.{ 1.0, 0.0, 0.0 });
    try index.add(.{ 0.0, 1.0, 0.0 });
    try index.add(.{ 0.707, 0.707, 0.0 }); // 45 degrees

    var results: [3]SearchResult = undefined;
    const found = try index.search(.{ 1.0, 0.0, 0.0 }, 3, &results);

    try std.testing.expectEqual(@as(usize, 3), found.len);
    try std.testing.expectEqual(@as(usize, 0), found[0].index); // Exact match
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), found[0].distance, 0.001);

    // 45-degree vector should be second closest
    try std.testing.expectEqual(@as(usize, 2), found[1].index);
}

test "flat index serialize deserialize" {
    const allocator = std.testing.allocator;

    var index = FlatIndex(4).init(allocator, .l2);
    defer index.deinit();

    try index.add(.{ 1.0, 2.0, 3.0, 4.0 });
    try index.add(.{ 5.0, 6.0, 7.0, 8.0 });

    // Serialize
    const data = try index.serialize(allocator);
    defer allocator.free(data);

    // Deserialize
    var loaded = try FlatIndex(4).deserialize(allocator, data);
    defer loaded.deinit();

    try std.testing.expectEqual(@as(usize, 2), loaded.count());
    try std.testing.expectEqual(DistanceMetric.l2, loaded.metric);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), loaded.vectors.items[0][0], 0.001);
}

test "flat index empty search" {
    const allocator = std.testing.allocator;

    var index = FlatIndex(4).init(allocator, .l2);
    defer index.deinit();

    var results: [3]SearchResult = undefined;
    const found = try index.search(.{ 1.0, 0.0, 0.0, 0.0 }, 3, &results);

    try std.testing.expectEqual(@as(usize, 0), found.len);
}
