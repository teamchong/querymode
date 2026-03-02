//! IVF-PQ Vector Index
//!
//! Inverted File with Product Quantization for approximate k-NN.
//! Provides fast search with controllable accuracy/speed tradeoff.
//!
//! Architecture:
//! 1. IVF: K-means clustering partitions vectors into cells
//! 2. PQ: Each vector is compressed using product quantization
//!
//! Use cases:
//! - Large datasets (100k-100M vectors)
//! - Memory-constrained environments
//! - Approximate search acceptable

const std = @import("std");

/// Search result: index and distance
pub const SearchResult = struct {
    index: usize,
    distance: f32,
};

/// IVF-PQ index configuration
pub const IvfPqConfig = struct {
    /// Number of IVF partitions (centroids)
    n_partitions: u32 = 256,
    /// Number of PQ sub-vectors (must evenly divide dimension)
    n_subvectors: u32 = 48,
    /// Number of centroids per sub-quantizer (usually 256 for 8-bit codes)
    n_codes: u32 = 256,
    /// Number of partitions to probe during search
    n_probe: u32 = 10,
};

/// IVF-PQ index for approximate k-NN search
pub fn IvfPqIndex(comptime DIM: usize) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        config: IvfPqConfig,

        // IVF components
        ivf_centroids: [][DIM]f32, // [n_partitions][DIM]
        inverted_lists: []std.ArrayListUnmanaged(u32), // Vector IDs per partition

        // PQ components
        pq_centroids: [][][]f32, // [n_subvectors][n_codes][subvec_dim]
        pq_codes: [][]u8, // [n_vectors][n_subvectors] - compressed codes

        // Original vectors for re-ranking (optional, can be null)
        vectors: ?[][DIM]f32,

        // Computed values
        subvec_dim: usize,
        n_vectors: usize,
        is_trained: bool,

        /// Initialize an empty index
        pub fn init(allocator: std.mem.Allocator, config: IvfPqConfig) !Self {
            if (DIM % config.n_subvectors != 0) {
                return error.InvalidConfig;
            }

            const subvec_dim = DIM / config.n_subvectors;

            // Allocate IVF centroids
            const ivf_centroids = try allocator.alloc([DIM]f32, config.n_partitions);
            @memset(ivf_centroids, [_]f32{0.0} ** DIM);

            // Allocate inverted lists
            const inverted_lists = try allocator.alloc(std.ArrayListUnmanaged(u32), config.n_partitions);
            for (inverted_lists) |*list| {
                list.* = std.ArrayListUnmanaged(u32){};
            }

            // Allocate PQ centroids: [n_subvectors][n_codes][subvec_dim]
            const pq_centroids = try allocator.alloc([][]f32, config.n_subvectors);
            for (pq_centroids) |*subvec_centroids| {
                subvec_centroids.* = try allocator.alloc([]f32, config.n_codes);
                for (subvec_centroids.*) |*centroid| {
                    centroid.* = try allocator.alloc(f32, subvec_dim);
                    @memset(centroid.*, 0.0);
                }
            }

            return Self{
                .allocator = allocator,
                .config = config,
                .ivf_centroids = ivf_centroids,
                .inverted_lists = inverted_lists,
                .pq_centroids = pq_centroids,
                .pq_codes = &[_][]u8{},
                .vectors = null,
                .subvec_dim = subvec_dim,
                .n_vectors = 0,
                .is_trained = false,
            };
        }

        /// Deinitialize and free memory
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.ivf_centroids);

            for (self.inverted_lists) |*list| {
                list.deinit(self.allocator);
            }
            self.allocator.free(self.inverted_lists);

            for (self.pq_centroids) |subvec_centroids| {
                for (subvec_centroids) |centroid| {
                    self.allocator.free(centroid);
                }
                self.allocator.free(subvec_centroids);
            }
            self.allocator.free(self.pq_centroids);

            for (self.pq_codes) |codes| {
                self.allocator.free(codes);
            }
            if (self.pq_codes.len > 0) {
                self.allocator.free(self.pq_codes);
            }

            if (self.vectors) |vecs| {
                self.allocator.free(vecs);
            }
        }

        /// Number of vectors in the index
        pub fn count(self: *const Self) usize {
            return self.n_vectors;
        }

        /// Compute L2 squared distance between two vectors
        fn l2Distance(a: []const f32, b: []const f32) f32 {
            var sum: f32 = 0.0;
            for (a, b) |va, vb| {
                const diff = va - vb;
                sum += diff * diff;
            }
            return sum;
        }

        /// Compute L2 distance for fixed-size arrays
        fn l2DistanceFixed(a: [DIM]f32, b: [DIM]f32) f32 {
            var sum: f32 = 0.0;
            for (a, b) |va, vb| {
                const diff = va - vb;
                sum += diff * diff;
            }
            return sum;
        }

        /// Find nearest centroid
        fn findNearestCentroid(self: *const Self, vec: [DIM]f32) usize {
            var min_dist: f32 = std.math.floatMax(f32);
            var min_idx: usize = 0;

            for (self.ivf_centroids, 0..) |centroid, i| {
                const dist = l2DistanceFixed(vec, centroid);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_idx = i;
                }
            }

            return min_idx;
        }

        /// Train IVF centroids using k-means (simplified)
        fn trainIvf(self: *Self, vectors: [][DIM]f32) void {
            if (vectors.len == 0) return;

            const n_clusters = @min(self.config.n_partitions, vectors.len);

            // Initialize centroids with first n_clusters vectors (simple initialization)
            for (0..n_clusters) |i| {
                @memcpy(&self.ivf_centroids[i], &vectors[i % vectors.len]);
            }

            // Run k-means iterations
            const max_iters: usize = 10;
            var assignments = self.allocator.alloc(usize, vectors.len) catch return;
            defer self.allocator.free(assignments);

            for (0..max_iters) |_| {
                // Assign vectors to nearest centroid
                for (vectors, 0..) |vec, i| {
                    assignments[i] = self.findNearestCentroid(vec);
                }

                // Update centroids
                var counts = self.allocator.alloc(usize, n_clusters) catch return;
                defer self.allocator.free(counts);
                @memset(counts, 0);

                for (self.ivf_centroids[0..n_clusters]) |*c| {
                    @memset(c, 0.0);
                }

                for (vectors, 0..) |vec, i| {
                    const cluster = assignments[i];
                    if (cluster < n_clusters) {
                        for (0..DIM) |d| {
                            self.ivf_centroids[cluster][d] += vec[d];
                        }
                        counts[cluster] += 1;
                    }
                }

                for (0..n_clusters) |c| {
                    if (counts[c] > 0) {
                        const count_f: f32 = @floatFromInt(counts[c]);
                        for (0..DIM) |d| {
                            self.ivf_centroids[c][d] /= count_f;
                        }
                    }
                }
            }
        }

        /// Train PQ codebooks
        fn trainPq(self: *Self, vectors: [][DIM]f32) void {
            if (vectors.len == 0) return;

            const subvec_dim = self.subvec_dim;
            const n_codes = @min(self.config.n_codes, vectors.len);

            // Train each sub-quantizer independently
            for (0..self.config.n_subvectors) |sv| {
                const start = sv * subvec_dim;

                // Initialize codebook with first n_codes sub-vectors
                for (0..n_codes) |c| {
                    for (0..subvec_dim) |d| {
                        self.pq_centroids[sv][c][d] = vectors[c % vectors.len][start + d];
                    }
                }

                // Run k-means on sub-vectors
                const max_iters: usize = 5;
                var assignments = self.allocator.alloc(usize, vectors.len) catch return;
                defer self.allocator.free(assignments);

                for (0..max_iters) |_| {
                    // Assign sub-vectors to nearest code
                    for (vectors, 0..) |vec, i| {
                        var min_dist: f32 = std.math.floatMax(f32);
                        var min_code: usize = 0;

                        for (0..n_codes) |c| {
                            var dist: f32 = 0.0;
                            for (0..subvec_dim) |d| {
                                const diff = vec[start + d] - self.pq_centroids[sv][c][d];
                                dist += diff * diff;
                            }
                            if (dist < min_dist) {
                                min_dist = dist;
                                min_code = c;
                            }
                        }
                        assignments[i] = min_code;
                    }

                    // Update codebook
                    var counts = self.allocator.alloc(usize, n_codes) catch return;
                    defer self.allocator.free(counts);
                    @memset(counts, 0);

                    for (0..n_codes) |c| {
                        @memset(self.pq_centroids[sv][c], 0.0);
                    }

                    for (vectors, 0..) |vec, i| {
                        const code = assignments[i];
                        if (code < n_codes) {
                            for (0..subvec_dim) |d| {
                                self.pq_centroids[sv][code][d] += vec[start + d];
                            }
                            counts[code] += 1;
                        }
                    }

                    for (0..n_codes) |c| {
                        if (counts[c] > 0) {
                            const count_f: f32 = @floatFromInt(counts[c]);
                            for (0..subvec_dim) |d| {
                                self.pq_centroids[sv][c][d] /= count_f;
                            }
                        }
                    }
                }
            }
        }

        /// Encode a vector using PQ
        fn encodeVector(self: *const Self, vec: [DIM]f32) ![]u8 {
            var codes = try self.allocator.alloc(u8, self.config.n_subvectors);

            for (0..self.config.n_subvectors) |sv| {
                const start = sv * self.subvec_dim;
                var min_dist: f32 = std.math.floatMax(f32);
                var min_code: u8 = 0;

                for (0..self.config.n_codes) |c| {
                    var dist: f32 = 0.0;
                    for (0..self.subvec_dim) |d| {
                        const diff = vec[start + d] - self.pq_centroids[sv][c][d];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_code = @intCast(c);
                    }
                }
                codes[sv] = min_code;
            }

            return codes;
        }

        /// Train the index on a set of vectors
        pub fn train(self: *Self, vectors: [][DIM]f32) !void {
            if (vectors.len == 0) return error.EmptyTrainingSet;

            // Train IVF centroids
            self.trainIvf(vectors);

            // Train PQ codebooks
            self.trainPq(vectors);

            self.is_trained = true;
        }

        /// Add vectors to the index (must be trained first)
        pub fn add(self: *Self, vectors: [][DIM]f32) !void {
            if (!self.is_trained) return error.NotTrained;
            if (vectors.len == 0) return;

            // Allocate PQ codes for new vectors
            var new_codes = try self.allocator.alloc([]u8, vectors.len);
            errdefer {
                for (new_codes) |codes| {
                    self.allocator.free(codes);
                }
                self.allocator.free(new_codes);
            }

            // Encode and assign each vector
            for (vectors, 0..) |vec, i| {
                // Find partition
                const partition = self.findNearestCentroid(vec);

                // Encode with PQ
                new_codes[i] = try self.encodeVector(vec);

                // Add to inverted list
                const vec_id: u32 = @intCast(self.n_vectors + i);
                try self.inverted_lists[partition].append(self.allocator, vec_id);
            }

            // Store PQ codes
            if (self.pq_codes.len == 0) {
                self.pq_codes = new_codes;
            } else {
                // Extend existing codes
                const old_codes = self.pq_codes;
                self.pq_codes = try self.allocator.alloc([]u8, old_codes.len + new_codes.len);
                @memcpy(self.pq_codes[0..old_codes.len], old_codes);
                @memcpy(self.pq_codes[old_codes.len..], new_codes);
                self.allocator.free(old_codes);
                self.allocator.free(new_codes);
            }

            self.n_vectors += vectors.len;
        }

        /// Compute asymmetric distance using precomputed distance table
        fn asymmetricDistance(self: *const Self, query: [DIM]f32, codes: []const u8) f32 {
            var dist: f32 = 0.0;

            for (0..self.config.n_subvectors) |sv| {
                const start = sv * self.subvec_dim;
                const code = codes[sv];

                for (0..self.subvec_dim) |d| {
                    const diff = query[start + d] - self.pq_centroids[sv][code][d];
                    dist += diff * diff;
                }
            }

            return dist;
        }

        /// Search for k nearest neighbors
        pub fn search(self: *const Self, query: [DIM]f32, k: usize, results_buf: []SearchResult) ![]SearchResult {
            if (!self.is_trained) return error.NotTrained;
            if (self.n_vectors == 0) return results_buf[0..0];

            const actual_k = @min(k, self.n_vectors);
            if (results_buf.len < actual_k) return error.BufferTooSmall;

            // Find nearest partitions
            var partition_dists: [256]struct { idx: usize, dist: f32 } = undefined;
            const n_partitions = @min(self.config.n_partitions, 256);

            for (0..n_partitions) |i| {
                partition_dists[i] = .{
                    .idx = i,
                    .dist = l2DistanceFixed(query, self.ivf_centroids[i]),
                };
            }

            // Sort partitions by distance
            std.mem.sort(
                @TypeOf(partition_dists[0]),
                partition_dists[0..n_partitions],
                {},
                struct {
                    fn lessThan(_: void, a: @TypeOf(partition_dists[0]), b: @TypeOf(partition_dists[0])) bool {
                        return a.dist < b.dist;
                    }
                }.lessThan,
            );

            // Search in top n_probe partitions
            var result_count: usize = 0;
            const n_probe = @min(self.config.n_probe, n_partitions);

            for (partition_dists[0..n_probe]) |pd| {
                const partition = pd.idx;
                const list = self.inverted_lists[partition];

                for (list.items) |vec_id| {
                    const codes = self.pq_codes[vec_id];
                    const dist = self.asymmetricDistance(query, codes);

                    // Insertion sort into results
                    var insert_pos: usize = result_count;
                    while (insert_pos > 0 and dist < results_buf[insert_pos - 1].distance) {
                        insert_pos -= 1;
                    }

                    if (insert_pos < actual_k) {
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
                            .index = vec_id,
                            .distance = dist,
                        };
                    }
                }
            }

            return results_buf[0..result_count];
        }
    };
}

// =============================================================================
// Tests
// =============================================================================

test "ivf pq index basic" {
    const allocator = std.testing.allocator;

    // Use 8-dim vectors with 2 subvectors of 4 dims each
    var index = try IvfPqIndex(8).init(allocator, .{
        .n_partitions = 4,
        .n_subvectors = 2,
        .n_codes = 4,
        .n_probe = 2,
    });
    defer index.deinit();

    // Create training data
    var vectors: [10][8]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (&vectors) |*vec| {
        for (vec) |*v| {
            v.* = random.float(f32);
        }
    }

    // Train
    try index.train(&vectors);
    try std.testing.expect(index.is_trained);

    // Add vectors
    try index.add(&vectors);
    try std.testing.expectEqual(@as(usize, 10), index.count());

    // Search
    var results: [5]SearchResult = undefined;
    const found = try index.search(vectors[0], 5, &results);

    try std.testing.expect(found.len > 0);
    try std.testing.expect(found.len <= 5);
}
