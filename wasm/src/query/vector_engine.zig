//! DuckDB-Style Vectorized Query Engine
//!
//! Key architectural decisions from DuckDB:
//! - VECTOR_SIZE = 2048 tuples per batch
//! - Selection vectors for lazy filtering
//! - Linear probing hash tables
//! - Columnar DataChunk format
//! - Zero-copy column access

const std = @import("std");

// ============================================================================
// Constants (matching DuckDB)
// ============================================================================

/// Standard vector size - all operators process this many rows at once
/// DuckDB uses 2048 as it fits in L3 cache on modern CPUs
pub const VECTOR_SIZE: usize = 2048;

/// Hash table load factor threshold for resize
/// Higher = better cache locality but more collisions
pub const HASH_LOAD_FACTOR: f32 = 0.85;

/// Null sentinel values
pub const NULL_INT64: i64 = std.math.minInt(i64);
pub const NULL_FLOAT64: f64 = std.math.nan(f64);

// ============================================================================
// SIMD Types (WASM SIMD128 compatible)
// ============================================================================

pub const Vec2i64 = @Vector(2, i64);
pub const Vec2u64 = @Vector(2, u64);
pub const Vec4f32 = @Vector(4, f32);
pub const Vec2f64 = @Vector(2, f64);
pub const Vec4i32 = @Vector(4, i32);
pub const Vec8i16 = @Vector(8, i16);
pub const Vec16i8 = @Vector(16, i8);

// ============================================================================
// Selection Vector - Lazy filtering without materialization
// ============================================================================

/// Selection vector marks which rows are valid without copying data
/// This is KEY to DuckDB's performance - filters don't materialize results
pub const SelectionVector = struct {
    /// Indices of selected rows (null = all rows selected)
    indices: ?[]u32,
    /// Number of selected rows
    count: usize,
    /// Backing storage (owned)
    storage: ?[]u32 = null,

    const Self = @This();

    /// Create selection vector selecting all rows
    pub fn all(row_count: usize) Self {
        return .{ .indices = null, .count = row_count };
    }

    /// Create from explicit indices
    pub fn fromIndices(indices: []u32) Self {
        return .{ .indices = indices, .count = indices.len };
    }

    /// Allocate storage for filtered selection
    pub fn alloc(allocator: std.mem.Allocator, capacity: usize) !Self {
        const storage = try allocator.alloc(u32, capacity);
        return .{ .indices = storage, .count = 0, .storage = storage };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        if (self.storage) |s| allocator.free(s);
        self.storage = null;
        self.indices = null;
    }

    /// Get row index at position (handles both selected and all-rows cases)
    pub inline fn get(self: *const Self, pos: usize) u32 {
        return if (self.indices) |idx| idx[pos] else @intCast(pos);
    }

    /// Check if selecting all rows (no filtering)
    pub inline fn isFlat(self: *const Self) bool {
        return self.indices == null;
    }
};

// ============================================================================
// Vector - Single column of data with validity mask
// ============================================================================

/// Column data types
pub const VectorType = enum(u8) {
    int64,
    int32,
    float64,
    float32,
    bool,
    string,
};

/// Single column vector with optional validity mask
pub const Vector = struct {
    /// Data type
    vtype: VectorType,
    /// Raw data pointer (not owned, points to columnar storage)
    data: [*]u8,
    /// Validity mask (null = all valid). Bit i = 1 means row i is valid
    validity: ?[]u64 = null,
    /// Number of elements
    count: usize,

    const Self = @This();

    /// Create vector from typed slice (zero-copy)
    pub fn fromSlice(comptime T: type, slice: []const T) Self {
        const vtype: VectorType = switch (T) {
            i64 => .int64,
            i32 => .int32,
            f64 => .float64,
            f32 => .float32,
            bool => .bool,
            else => @compileError("Unsupported type"),
        };
        return .{
            .vtype = vtype,
            .data = @constCast(@ptrCast(slice.ptr)),
            .count = slice.len,
        };
    }

    /// Get typed data pointer
    pub inline fn getData(self: *const Self, comptime T: type) [*]const T {
        return @ptrCast(@alignCast(self.data));
    }

    /// Get mutable typed data pointer
    pub inline fn getDataMut(self: *Self, comptime T: type) [*]T {
        return @ptrCast(@alignCast(self.data));
    }

    /// Check if row is valid (not null)
    pub inline fn isValid(self: *const Self, idx: usize) bool {
        if (self.validity) |v| {
            const word = idx / 64;
            const bit = @as(u6, @intCast(idx % 64));
            return (v[word] & (@as(u64, 1) << bit)) != 0;
        }
        return true;
    }
};

// ============================================================================
// DataChunk - Collection of vectors (like a row group)
// ============================================================================

/// DataChunk holds multiple columns for batch processing
/// All vectors have the same row count
pub const DataChunk = struct {
    vectors: []Vector,
    count: usize, // Actual row count (≤ VECTOR_SIZE)
    capacity: usize, // Max capacity (typically VECTOR_SIZE)

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, num_columns: usize) !Self {
        const vectors = try allocator.alloc(Vector, num_columns);
        return .{
            .vectors = vectors,
            .count = 0,
            .capacity = VECTOR_SIZE,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.vectors);
    }

    pub fn setCount(self: *Self, count: usize) void {
        self.count = @min(count, self.capacity);
    }
};

// ============================================================================
// Linear Probing Hash Table (DuckDB style)
// ============================================================================

/// Hash table entry with cached key for cache-efficient probe
const HashEntry = struct {
    /// Upper bits of hash for fast comparison
    hash_bits: u16,
    /// 0 = empty, 1 = occupied, 2 = deleted
    state: u8,
    _pad: u8 = 0,
    /// Row index in source data
    row_idx: u32,
    /// Cached key - avoids memory lookup during probe
    key: i64,
};

/// Result from parallel probe operation
pub const ProbeResult = struct {
    left_indices: []usize,
    right_indices: []usize,
};

/// Linear probing hash table - much faster than chaining for cache locality
pub const LinearHashTable = struct {
    entries: []HashEntry,
    capacity: usize,
    count: usize,
    mask: u64,

    const Self = @This();
    const EMPTY: u8 = 0;
    const OCCUPIED: u8 = 1;

    pub fn init(allocator: std.mem.Allocator, expected_count: usize) !Self {
        // Size to next power of 2, with load factor headroom
        var capacity: usize = 16;
        const target = @as(usize, @intFromFloat(@as(f32, @floatFromInt(expected_count)) / HASH_LOAD_FACTOR));
        while (capacity < target) capacity *= 2;

        const entries = try allocator.alloc(HashEntry, capacity);
        @memset(entries, HashEntry{ .hash_bits = 0, .state = EMPTY, ._pad = 0, .row_idx = 0, .key = 0 });

        return .{
            .entries = entries,
            .capacity = capacity,
            .count = 0,
            .mask = capacity - 1,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.entries);
    }

    /// MurmurHash3 finalizer - excellent distribution
    pub inline fn hash64(key: i64) u64 {
        var h: u64 = @bitCast(key);
        h ^= h >> 33;
        h *%= 0xff51afd7ed558ccd;
        h ^= h >> 33;
        h *%= 0xc4ceb9fe1a85ec53;
        h ^= h >> 33;
        return h;
    }

    /// SIMD hash 2 keys at once
    pub inline fn hash64x2(keys: Vec2i64) Vec2u64 {
        var h: Vec2u64 = @bitCast(keys);
        h ^= h >> @splat(33);
        h *%= @splat(0xff51afd7ed558ccd);
        h ^= h >> @splat(33);
        h *%= @splat(0xc4ceb9fe1a85ec53);
        h ^= h >> @splat(33);
        return h;
    }

    /// Batch hash function - compute hashes for entire batch using SIMD
    pub fn hash64Batch(keys: []const i64, hashes: []u64) void {
        std.debug.assert(hashes.len >= keys.len);
        var i: usize = 0;

        // SIMD path: hash 2 at a time
        while (i + 2 <= keys.len) : (i += 2) {
            const k: Vec2i64 = .{ keys[i], keys[i + 1] };
            const h = hash64x2(k);
            hashes[i] = h[0];
            hashes[i + 1] = h[1];
        }

        // Scalar remainder
        while (i < keys.len) : (i += 1) {
            hashes[i] = hash64(keys[i]);
        }
    }

    /// Prefetch hash table entries for a batch of hashes
    /// Call this BEFORE doing lookups to warm the cache
    pub fn prefetchSlots(self: *const Self, hashes: []const u64) void {
        for (hashes) |h| {
            const pos = @as(usize, @intCast(h & self.mask));
            @prefetch(@as([*]const u8, @ptrCast(&self.entries[pos])), .{
                .rw = .read,
                .locality = 3,
                .cache = .data,
            });
        }
    }

    /// Insert key with linear probing
    pub fn insert(self: *Self, key: i64, row_idx: u32) void {
        const h = hash64(key);
        const hash_bits: u16 = @truncate(h >> 48);
        var pos = @as(usize, @intCast(h & self.mask));

        // Linear probe until empty slot
        while (self.entries[pos].state == OCCUPIED) {
            pos = (pos + 1) & @as(usize, @intCast(self.mask));
        }

        self.entries[pos] = .{
            .hash_bits = hash_bits,
            .state = OCCUPIED,
            ._pad = 0,
            .row_idx = row_idx,
            .key = key,
        };
        self.count += 1;
    }

    /// Build from i64 column - VECTORIZED with cached keys
    pub fn buildFromColumn(self: *Self, data: []const i64) void {
        var i: usize = 0;

        // SIMD path: hash 2 at a time
        while (i + 2 <= data.len) : (i += 2) {
            const key0 = data[i];
            const key1 = data[i + 1];
            const keys: Vec2i64 = .{ key0, key1 };
            const hashes = hash64x2(keys);

            // Insert first
            if (key0 != NULL_INT64) {
                const h0 = hashes[0];
                const hash_bits0: u16 = @truncate(h0 >> 48);
                var pos0 = @as(usize, @intCast(h0 & self.mask));
                while (self.entries[pos0].state == OCCUPIED) {
                    pos0 = (pos0 + 1) & @as(usize, @intCast(self.mask));
                }
                self.entries[pos0] = .{ .hash_bits = hash_bits0, .state = OCCUPIED, ._pad = 0, .row_idx = @intCast(i), .key = key0 };
                self.count += 1;
            }

            // Insert second
            if (key1 != NULL_INT64) {
                const h1 = hashes[1];
                const hash_bits1: u16 = @truncate(h1 >> 48);
                var pos1 = @as(usize, @intCast(h1 & self.mask));
                while (self.entries[pos1].state == OCCUPIED) {
                    pos1 = (pos1 + 1) & @as(usize, @intCast(self.mask));
                }
                self.entries[pos1] = .{ .hash_bits = hash_bits1, .state = OCCUPIED, ._pad = 0, .row_idx = @intCast(i + 1), .key = key1 };
                self.count += 1;
            }
        }

        // Scalar remainder
        while (i < data.len) : (i += 1) {
            if (data[i] != NULL_INT64) {
                self.insert(data[i], @intCast(i));
            }
        }
    }

    /// Probe for key, returns row index or null
    pub fn probe(self: *const Self, key: i64) ?u32 {
        const h = hash64(key);
        const hash_bits: u16 = @truncate(h >> 48);
        var pos = @as(usize, @intCast(h & self.mask));

        // Linear probe until empty or found
        while (self.entries[pos].state != EMPTY) {
            if (self.entries[pos].state == OCCUPIED and
                self.entries[pos].hash_bits == hash_bits)
            {
                return self.entries[pos].row_idx;
            }
            pos = (pos + 1) & @as(usize, @intCast(self.mask));
        }
        return null;
    }

    /// Probe and return all matches (for non-unique keys)
    /// Uses cached keys in hash entries - no external memory lookup needed
    pub fn probeAll(self: *const Self, key: i64, keys_data: []const i64, out: []u32) usize {
        _ = keys_data; // Now unused - we use cached keys
        const h = hash64(key);
        const hash_bits: u16 = @truncate(h >> 48);
        var pos = @as(usize, @intCast(h & self.mask));
        var out_count: usize = 0;

        while (self.entries[pos].state != EMPTY and out_count < out.len) {
            if (self.entries[pos].state == OCCUPIED and
                self.entries[pos].hash_bits == hash_bits and
                self.entries[pos].key == key)
            {
                out[out_count] = self.entries[pos].row_idx;
                out_count += 1;
            }
            pos = (pos + 1) & @as(usize, @intCast(self.mask));
        }
        return out_count;
    }

    /// Batch probe with SIMD hashing + prefetching for JOIN optimization
    /// Uses cached keys in hash entries - no external memory lookup needed
    /// Returns total number of matches written to output buffers
    pub fn probeBatchCached(
        self: *const Self,
        left_keys: []const i64,
        left_out: []usize,
        right_out: []usize,
        left_offset: usize,
    ) usize {
        const batch_size = @min(left_keys.len, VECTOR_SIZE);
        if (batch_size == 0) return 0;

        // Static buffers for batch processing
        var hashes: [VECTOR_SIZE]u64 = undefined;

        // Step 1: SIMD batch hash computation
        hash64Batch(left_keys[0..batch_size], hashes[0..batch_size]);

        // Step 2: Prefetch hash table slots
        self.prefetchSlots(hashes[0..batch_size]);

        // Step 3: Probe with cache-hot hash table (using cached keys)
        var out_count: usize = 0;
        const max_out = @min(left_out.len, right_out.len);

        for (left_keys[0..batch_size], hashes[0..batch_size], 0..) |key, h, i| {
            if (key == NULL_INT64) continue;

            const hash_bits: u16 = @truncate(h >> 48);
            var pos = @as(usize, @intCast(h & self.mask));

            // Find all matches using cached key (no external memory access)
            while (self.entries[pos].state != EMPTY and out_count < max_out) {
                if (self.entries[pos].state == OCCUPIED and
                    self.entries[pos].hash_bits == hash_bits and
                    self.entries[pos].key == key)
                {
                    left_out[out_count] = left_offset + i;
                    right_out[out_count] = self.entries[pos].row_idx;
                    out_count += 1;
                }
                pos = (pos + 1) & @as(usize, @intCast(self.mask));
            }
        }

        return out_count;
    }

    /// Legacy batch probe (for compatibility)
    pub fn probeBatch(
        self: *const Self,
        left_keys: []const i64,
        right_keys: []const i64,
        left_out: []usize,
        right_out: []usize,
        left_offset: usize,
    ) usize {
        _ = right_keys; // Now unused - we use cached keys
        return self.probeBatchCached(left_keys, left_out, right_out, left_offset);
    }

    /// Parallel probe all left keys against the hash table
    /// Spawns multiple threads to probe different chunks of left_keys
    /// Returns lists of (left_idx, right_idx) matches
    pub fn probeAllParallel(
        self: *const Self,
        allocator: std.mem.Allocator,
        left_keys: []const i64,
    ) !ProbeResult {
        const num_threads = @min(std.Thread.getCpuCount() catch 4, 8);
        if (left_keys.len < 10000 or num_threads <= 1) {
            // Fall back to single-threaded for small inputs
            return self.probeAllSingleThread(allocator, left_keys);
        }

        const chunk_size = (left_keys.len + num_threads - 1) / num_threads;

        // Thread-local result buffers
        const ThreadResult = struct {
            left_indices: std.ArrayListUnmanaged(usize),
            right_indices: std.ArrayListUnmanaged(usize),
        };

        var thread_results = try allocator.alloc(ThreadResult, num_threads);
        defer allocator.free(thread_results);
        for (thread_results) |*r| {
            r.left_indices = .{};
            r.right_indices = .{};
        }

        var threads = try allocator.alloc(std.Thread, num_threads);
        defer allocator.free(threads);

        // Worker function
        const Worker = struct {
            fn run(
                ht: *const Self,
                keys: []const i64,
                offset: usize,
                alloc: std.mem.Allocator,
                result: *ThreadResult,
            ) void {
                // Pre-allocate estimate (assume 1:1 match ratio)
                result.left_indices.ensureTotalCapacity(alloc, keys.len) catch return;
                result.right_indices.ensureTotalCapacity(alloc, keys.len) catch return;

                var hashes: [VECTOR_SIZE]u64 = undefined;
                var left_batch: [VECTOR_SIZE * 4]usize = undefined;
                var right_batch: [VECTOR_SIZE * 4]usize = undefined;

                var i: usize = 0;
                while (i < keys.len) {
                    const batch_end = @min(i + VECTOR_SIZE, keys.len);
                    const batch_keys = keys[i..batch_end];
                    const batch_size = batch_keys.len;

                    // Hash + prefetch
                    hash64Batch(batch_keys, hashes[0..batch_size]);
                    ht.prefetchSlots(hashes[0..batch_size]);

                    // Probe each key
                    for (batch_keys, hashes[0..batch_size], 0..) |key, h, j| {
                        if (key == NULL_INT64) continue;

                        const hash_bits: u16 = @truncate(h >> 48);
                        var pos = @as(usize, @intCast(h & ht.mask));
                        var matches: usize = 0;

                        while (ht.entries[pos].state != EMPTY and matches < left_batch.len) {
                            if (ht.entries[pos].state == OCCUPIED and
                                ht.entries[pos].hash_bits == hash_bits and
                                ht.entries[pos].key == key)
                            {
                                left_batch[matches] = offset + i + j;
                                right_batch[matches] = ht.entries[pos].row_idx;
                                matches += 1;
                            }
                            pos = (pos + 1) & @as(usize, @intCast(ht.mask));
                        }

                        // Append matches
                        for (0..matches) |m| {
                            result.left_indices.append(alloc, left_batch[m]) catch return;
                            result.right_indices.append(alloc, right_batch[m]) catch return;
                        }
                    }

                    i = batch_end;
                }
            }
        };

        // Spawn threads
        var spawned: usize = 0;
        errdefer {
            for (threads[0..spawned]) |t| t.join();
            for (thread_results) |*r| {
                r.left_indices.deinit(allocator);
                r.right_indices.deinit(allocator);
            }
        }

        for (0..num_threads) |t| {
            const start = t * chunk_size;
            if (start >= left_keys.len) break;
            const end = @min(start + chunk_size, left_keys.len);
            threads[t] = std.Thread.spawn(.{}, Worker.run, .{
                self,
                left_keys[start..end],
                start,
                allocator,
                &thread_results[t],
            }) catch break;
            spawned += 1;
        }

        // Wait for all threads
        for (threads[0..spawned]) |t| t.join();

        // Merge results
        var total_matches: usize = 0;
        for (thread_results[0..spawned]) |r| {
            total_matches += r.left_indices.items.len;
        }

        var left_out = try allocator.alloc(usize, total_matches);
        errdefer allocator.free(left_out);
        var right_out = try allocator.alloc(usize, total_matches);

        var pos: usize = 0;
        for (thread_results[0..spawned]) |*r| {
            const len = r.left_indices.items.len;
            @memcpy(left_out[pos .. pos + len], r.left_indices.items);
            @memcpy(right_out[pos .. pos + len], r.right_indices.items);
            pos += len;
            r.left_indices.deinit(allocator);
            r.right_indices.deinit(allocator);
        }

        return .{ .left_indices = left_out, .right_indices = right_out };
    }

    /// Single-threaded probe all (fallback for small inputs)
    fn probeAllSingleThread(
        self: *const Self,
        allocator: std.mem.Allocator,
        left_keys: []const i64,
    ) !ProbeResult {
        var left_out = std.ArrayListUnmanaged(usize){};
        var right_out = std.ArrayListUnmanaged(usize){};
        errdefer {
            left_out.deinit(allocator);
            right_out.deinit(allocator);
        }

        try left_out.ensureTotalCapacity(allocator, left_keys.len);
        try right_out.ensureTotalCapacity(allocator, left_keys.len);

        var left_batch: [VECTOR_SIZE * 4]usize = undefined;
        var right_batch: [VECTOR_SIZE * 4]usize = undefined;

        var i: usize = 0;
        while (i < left_keys.len) {
            const batch_end = @min(i + VECTOR_SIZE, left_keys.len);
            const batch_keys = left_keys[i..batch_end];

            const matches = self.probeBatchCached(batch_keys, &left_batch, &right_batch, i);
            try left_out.ensureUnusedCapacity(allocator, matches);
            try right_out.ensureUnusedCapacity(allocator, matches);

            for (0..matches) |m| {
                left_out.appendAssumeCapacity(left_batch[m]);
                right_out.appendAssumeCapacity(right_batch[m]);
            }

            i = batch_end;
        }

        return ProbeResult{
            .left_indices = try left_out.toOwnedSlice(allocator),
            .right_indices = try right_out.toOwnedSlice(allocator),
        };
    }
};

// ============================================================================
// Vectorized Aggregation
// ============================================================================

/// Aggregation state for SUM/COUNT/AVG/MIN/MAX
pub const AggState = struct {
    sum: f64 = 0,
    count: u64 = 0,
    min: f64 = std.math.inf(f64),
    max: f64 = -std.math.inf(f64),

    const Self = @This();

    /// SIMD update from f64 column
    pub fn updateColumnF64(self: *Self, data: []const f64, sel: *const SelectionVector) void {
        if (sel.isFlat()) {
            // Fast path: process entire column with SIMD
            self.updateContiguousF64(data);
        } else {
            // Selection vector path: gather values
            const indices = sel.indices.?;
            for (indices[0..sel.count]) |idx| {
                const v = data[idx];
                self.sum += v;
                self.count += 1;
                if (v < self.min) self.min = v;
                if (v > self.max) self.max = v;
            }
        }
    }

    /// SIMD update from contiguous f64 array
    fn updateContiguousF64(self: *Self, data: []const f64) void {
        var sum_vec: Vec2f64 = @splat(0);
        var min_vec: Vec2f64 = @splat(std.math.inf(f64));
        var max_vec: Vec2f64 = @splat(-std.math.inf(f64));
        var i: usize = 0;

        // SIMD path: 2 elements at a time
        while (i + 2 <= data.len) : (i += 2) {
            const v: Vec2f64 = .{ data[i], data[i + 1] };
            sum_vec += v;
            min_vec = @min(min_vec, v);
            max_vec = @max(max_vec, v);
        }

        self.sum += @reduce(.Add, sum_vec);
        const batch_min = @reduce(.Min, min_vec);
        const batch_max = @reduce(.Max, max_vec);
        if (batch_min < self.min) self.min = batch_min;
        if (batch_max > self.max) self.max = batch_max;
        self.count += i;

        // Scalar remainder
        while (i < data.len) : (i += 1) {
            const v = data[i];
            self.sum += v;
            if (v < self.min) self.min = v;
            if (v > self.max) self.max = v;
        }
        self.count += data.len - (data.len / 2 * 2);
    }

    /// SIMD update from i64 column
    pub fn updateColumnI64(self: *Self, data: []const i64, sel: *const SelectionVector) void {
        if (sel.isFlat()) {
            var sum: i64 = 0;
            var min: i64 = std.math.maxInt(i64);
            var max: i64 = std.math.minInt(i64);
            var i: usize = 0;

            // SIMD path
            while (i + 2 <= data.len) : (i += 2) {
                const v: Vec2i64 = .{ data[i], data[i + 1] };
                sum += @reduce(.Add, v);
                const batch_min = @reduce(.Min, v);
                const batch_max = @reduce(.Max, v);
                if (batch_min < min) min = batch_min;
                if (batch_max > max) max = batch_max;
            }

            // Remainder
            while (i < data.len) : (i += 1) {
                sum += data[i];
                if (data[i] < min) min = data[i];
                if (data[i] > max) max = data[i];
            }

            self.sum += @floatFromInt(sum);
            self.count += data.len;
            if (@as(f64, @floatFromInt(min)) < self.min) self.min = @floatFromInt(min);
            if (@as(f64, @floatFromInt(max)) > self.max) self.max = @floatFromInt(max);
        } else {
            const indices = sel.indices.?;
            for (indices[0..sel.count]) |idx| {
                const v = data[idx];
                self.sum += @floatFromInt(v);
                self.count += 1;
                const vf: f64 = @floatFromInt(v);
                if (vf < self.min) self.min = vf;
                if (vf > self.max) self.max = vf;
            }
        }
    }

    pub fn getAvg(self: *const Self) f64 {
        return if (self.count > 0) self.sum / @as(f64, @floatFromInt(self.count)) else 0;
    }

    /// Update with single value (for row-by-row processing)
    pub inline fn update(self: *Self, val: f64) void {
        self.sum += val;
        self.count += 1;
        if (val < self.min) self.min = val;
        if (val > self.max) self.max = val;
    }

    /// Finalize aggregate result based on function type
    /// Compatible with WASM aggregates.AggFunc enum values
    pub fn finalize(self: *const Self, func_ordinal: u8) f64 {
        // Map ordinal to result: 0=sum, 1=count, 2=avg, 3=min, 4=max
        return switch (func_ordinal) {
            0 => self.sum, // sum
            1 => @floatFromInt(self.count), // count
            2 => self.getAvg(), // avg
            3 => if (self.min == std.math.inf(f64)) 0 else self.min, // min
            4 => if (self.max == -std.math.inf(f64)) 0 else self.max, // max
            else => self.sum,
        };
    }
};

// ============================================================================
// Hash Join Executor
// ============================================================================

/// Execute hash join between two columns
/// Returns (left_indices, right_indices) pairs
pub fn executeHashJoin(
    allocator: std.mem.Allocator,
    left_keys: []const i64,
    right_keys: []const i64,
    max_results: usize,
) !struct { left: []u32, right: []u32 } {
    // Build phase: hash the right (smaller) table
    var ht = try LinearHashTable.init(allocator, right_keys.len);
    defer ht.deinit(allocator);
    ht.buildFromColumn(right_keys);

    // Probe phase: scan left table
    var left_out = try allocator.alloc(u32, max_results);
    errdefer allocator.free(left_out);
    var right_out = try allocator.alloc(u32, max_results);
    errdefer allocator.free(right_out);

    var out_count: usize = 0;
    var match_buf: [64]u32 = undefined;

    for (left_keys, 0..) |key, left_idx| {
        if (key == NULL_INT64) continue;

        // Find all matches in right table
        const matches = ht.probeAll(key, right_keys, &match_buf);
        for (match_buf[0..matches]) |right_idx| {
            if (out_count >= max_results) break;
            left_out[out_count] = @intCast(left_idx);
            right_out[out_count] = right_idx;
            out_count += 1;
        }
    }

    return .{
        .left = left_out[0..out_count],
        .right = right_out[0..out_count],
    };
}

// ============================================================================
// Vectorized Filter Operations
// ============================================================================

/// Filter comparison operators
pub const FilterOp = enum {
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
};

/// Apply filter to i64 column, output selection vector
/// Returns number of matching rows
pub fn filterI64(
    data: []const i64,
    op: FilterOp,
    value: i64,
    input_sel: *const SelectionVector,
    output_sel: *SelectionVector,
) usize {
    const out_indices = output_sel.storage.?;
    var out_count: usize = 0;

    if (input_sel.isFlat()) {
        // Fast path: scan entire column
        var i: usize = 0;

        // SIMD comparison for eq/ne (most common)
        if (op == .eq or op == .ne) {
            const target: Vec2i64 = @splat(value);
            while (i + 2 <= data.len and out_count + 2 <= out_indices.len) : (i += 2) {
                const v: Vec2i64 = .{ data[i], data[i + 1] };
                const cmp = if (op == .eq) v == target else v != target;
                if (cmp[0]) {
                    out_indices[out_count] = @intCast(i);
                    out_count += 1;
                }
                if (cmp[1]) {
                    out_indices[out_count] = @intCast(i + 1);
                    out_count += 1;
                }
            }
        }

        // Scalar remainder / other ops
        while (i < data.len and out_count < out_indices.len) : (i += 1) {
            const match = switch (op) {
                .eq => data[i] == value,
                .ne => data[i] != value,
                .lt => data[i] < value,
                .le => data[i] <= value,
                .gt => data[i] > value,
                .ge => data[i] >= value,
            };
            if (match) {
                out_indices[out_count] = @intCast(i);
                out_count += 1;
            }
        }
    } else {
        // Selection vector path
        const in_indices = input_sel.indices.?;
        for (in_indices[0..input_sel.count]) |idx| {
            if (out_count >= out_indices.len) break;
            const match = switch (op) {
                .eq => data[idx] == value,
                .ne => data[idx] != value,
                .lt => data[idx] < value,
                .le => data[idx] <= value,
                .gt => data[idx] > value,
                .ge => data[idx] >= value,
            };
            if (match) {
                out_indices[out_count] = idx;
                out_count += 1;
            }
        }
    }

    output_sel.count = out_count;
    output_sel.indices = out_indices[0..out_count];
    return out_count;
}

/// Apply filter to f64 column
pub fn filterF64(
    data: []const f64,
    op: FilterOp,
    value: f64,
    input_sel: *const SelectionVector,
    output_sel: *SelectionVector,
) usize {
    const out_indices = output_sel.storage.?;
    var out_count: usize = 0;

    if (input_sel.isFlat()) {
        var i: usize = 0;

        // SIMD for common comparisons
        if (op == .gt or op == .lt) {
            const target: Vec2f64 = @splat(value);
            while (i + 2 <= data.len and out_count + 2 <= out_indices.len) : (i += 2) {
                const v: Vec2f64 = .{ data[i], data[i + 1] };
                const cmp = if (op == .gt) v > target else v < target;
                if (cmp[0]) {
                    out_indices[out_count] = @intCast(i);
                    out_count += 1;
                }
                if (cmp[1]) {
                    out_indices[out_count] = @intCast(i + 1);
                    out_count += 1;
                }
            }
        }

        // Scalar remainder
        while (i < data.len and out_count < out_indices.len) : (i += 1) {
            const match = switch (op) {
                .eq => data[i] == value,
                .ne => data[i] != value,
                .lt => data[i] < value,
                .le => data[i] <= value,
                .gt => data[i] > value,
                .ge => data[i] >= value,
            };
            if (match) {
                out_indices[out_count] = @intCast(i);
                out_count += 1;
            }
        }
    } else {
        const in_indices = input_sel.indices.?;
        for (in_indices[0..input_sel.count]) |idx| {
            if (out_count >= out_indices.len) break;
            const match = switch (op) {
                .eq => data[idx] == value,
                .ne => data[idx] != value,
                .lt => data[idx] < value,
                .le => data[idx] <= value,
                .gt => data[idx] > value,
                .ge => data[idx] >= value,
            };
            if (match) {
                out_indices[out_count] = idx;
                out_count += 1;
            }
        }
    }

    output_sel.count = out_count;
    output_sel.indices = out_indices[0..out_count];
    return out_count;
}

// ============================================================================
// Hash-Based GROUP BY Aggregation
// ============================================================================

/// Group aggregation entry
pub const GroupAggEntry = struct {
    /// First row index for this group (for retrieving group key)
    first_row: u32,
    /// Cached key value for batch processing (avoids array lookup)
    key: i64,
    /// Aggregation state
    agg: AggState,
};

/// Hash-based GROUP BY with aggregation
/// Groups by i64 key column and aggregates f64 value column
pub const HashGroupBy = struct {
    /// Hash table mapping group key hash -> group index
    ht: LinearHashTable,
    /// Group aggregation states
    groups: std.ArrayListUnmanaged(GroupAggEntry),
    /// Allocator
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, expected_groups: usize) !Self {
        return .{
            .ht = try LinearHashTable.init(allocator, expected_groups),
            .groups = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.ht.deinit(self.allocator);
        self.groups.deinit(self.allocator);
    }

    /// Build groups from i64 key column (optimized with batch hashing + prefetch)
    pub fn buildGroups(self: *Self, keys: []const i64) !void {
        var offset: usize = 0;

        // Process in VECTOR_SIZE batches for cache efficiency
        while (offset < keys.len) {
            const batch_end = @min(offset + VECTOR_SIZE, keys.len);
            const batch_keys = keys[offset..batch_end];
            const batch_size = batch_keys.len;

            // Step 1: SIMD batch hash computation
            LinearHashTable.hash64Batch(batch_keys, batch_hashes[0..batch_size]);

            // Step 2: Prefetch all hash table slots
            self.ht.prefetchSlots(batch_hashes[0..batch_size]);

            // Step 3: Find or create groups
            for (batch_keys, batch_hashes[0..batch_size], 0..) |key, h, i| {
                if (key == NULL_INT64) continue;

                const hash_bits: u16 = @truncate(h >> 48);
                var pos = @as(usize, @intCast(h & self.ht.mask));

                // Find existing group or create new one
                var found = false;
                while (self.ht.entries[pos].state == LinearHashTable.OCCUPIED) {
                    if (self.ht.entries[pos].hash_bits == hash_bits) {
                        if (self.groups.items[self.ht.entries[pos].row_idx].key == key) {
                            found = true;
                            break;
                        }
                    }
                    pos = (pos + 1) & @as(usize, @intCast(self.ht.mask));
                }

                if (!found) {
                    // Create new group with cached key
                    const group_idx: u32 = @intCast(self.groups.items.len);
                    try self.groups.append(self.allocator, .{
                        .first_row = @intCast(offset + i),
                        .key = key,
                        .agg = .{},
                    });

                    // Insert into hash table
                    self.ht.entries[pos] = .{
                        .hash_bits = hash_bits,
                        .row_idx = group_idx,
                        .state = LinearHashTable.OCCUPIED,
                        ._pad = 0,
                        .key = key,
                    };
                    self.ht.count += 1;
                }
            }

            offset = batch_end;
        }
    }

    /// Aggregate f64 values into groups (optimized with batch hashing + prefetch)
    pub fn aggregateF64(self: *Self, keys: []const i64, values: []const f64) void {
        var offset: usize = 0;

        // Process in VECTOR_SIZE batches for cache efficiency
        while (offset < keys.len) {
            const batch_end = @min(offset + VECTOR_SIZE, keys.len);
            const batch_keys = keys[offset..batch_end];
            const batch_vals = values[offset..batch_end];
            const batch_size = batch_keys.len;

            // Step 1: SIMD batch hash computation
            LinearHashTable.hash64Batch(batch_keys, batch_hashes[0..batch_size]);

            // Step 2: Prefetch all hash table slots
            self.ht.prefetchSlots(batch_hashes[0..batch_size]);

            // Step 3: Find groups and store indices
            for (batch_keys, batch_hashes[0..batch_size], 0..) |key, h, i| {
                if (key == NULL_INT64) {
                    batch_group_indices[i] = -1;
                    continue;
                }

                const hash_bits: u16 = @truncate(h >> 48);
                var pos = @as(usize, @intCast(h & self.ht.mask));

                var found_idx: i32 = -1;
                while (self.ht.entries[pos].state == LinearHashTable.OCCUPIED) {
                    if (self.ht.entries[pos].hash_bits == hash_bits) {
                        const group_idx = self.ht.entries[pos].row_idx;
                        if (self.groups.items[group_idx].key == key) {
                            found_idx = @intCast(group_idx);
                            break;
                        }
                    }
                    pos = (pos + 1) & @as(usize, @intCast(self.ht.mask));
                }
                batch_group_indices[i] = found_idx;
            }

            // Step 4: Batch aggregate updates
            for (batch_group_indices[0..batch_size], batch_vals) |gidx, val| {
                if (gidx < 0) continue;
                const agg = &self.groups.items[@intCast(gidx)].agg;
                agg.sum += val;
                agg.count += 1;
                if (val < agg.min) agg.min = val;
                if (val > agg.max) agg.max = val;
            }

            offset = batch_end;
        }
    }

    /// Aggregate i64 values into groups (optimized with batch hashing + prefetch)
    pub fn aggregateI64(self: *Self, keys: []const i64, values: []const i64) void {
        var offset: usize = 0;

        // Process in VECTOR_SIZE batches for cache efficiency
        while (offset < keys.len) {
            const batch_end = @min(offset + VECTOR_SIZE, keys.len);
            const batch_keys = keys[offset..batch_end];
            const batch_vals = values[offset..batch_end];
            const batch_size = batch_keys.len;

            // Step 1: SIMD batch hash computation
            LinearHashTable.hash64Batch(batch_keys, batch_hashes[0..batch_size]);

            // Step 2: Prefetch all hash table slots
            self.ht.prefetchSlots(batch_hashes[0..batch_size]);

            // Step 3: Find groups and store indices
            for (batch_keys, batch_hashes[0..batch_size], 0..) |key, h, i| {
                if (key == NULL_INT64) {
                    batch_group_indices[i] = -1;
                    continue;
                }

                const hash_bits: u16 = @truncate(h >> 48);
                var pos = @as(usize, @intCast(h & self.ht.mask));

                var found_idx: i32 = -1;
                while (self.ht.entries[pos].state == LinearHashTable.OCCUPIED) {
                    if (self.ht.entries[pos].hash_bits == hash_bits) {
                        const group_idx = self.ht.entries[pos].row_idx;
                        if (self.groups.items[group_idx].key == key) {
                            found_idx = @intCast(group_idx);
                            break;
                        }
                    }
                    pos = (pos + 1) & @as(usize, @intCast(self.ht.mask));
                }
                batch_group_indices[i] = found_idx;
            }

            // Step 4: Batch aggregate updates
            for (batch_group_indices[0..batch_size], batch_vals) |gidx, val| {
                if (gidx < 0) continue;
                const agg = &self.groups.items[@intCast(gidx)].agg;
                const vf: f64 = @floatFromInt(val);
                agg.sum += vf;
                agg.count += 1;
                if (vf < agg.min) agg.min = vf;
                if (vf > agg.max) agg.max = vf;
            }

            offset = batch_end;
        }
    }

    /// Get number of groups
    pub fn groupCount(self: *const Self) usize {
        return self.groups.items.len;
    }

    /// Get group key at index (uses cached key)
    pub fn getGroupKey(self: *const Self, group_idx: usize, keys: []const i64) i64 {
        _ = keys; // No longer needed - using cached key
        return self.groups.items[group_idx].key;
    }

    /// Get group aggregation state
    pub fn getGroupAgg(self: *const Self, group_idx: usize) *const AggState {
        return &self.groups.items[group_idx].agg;
    }

    /// Get first row index for a group (for representative value lookup)
    pub fn getGroupFirstRow(self: *const Self, group_idx: usize) u32 {
        return self.groups.items[group_idx].first_row;
    }

    // Static batch buffers for batch processing (used by buildGroups/aggregateF64/aggregateI64)
    var batch_hashes: [VECTOR_SIZE]u64 = undefined;
    var batch_group_indices: [VECTOR_SIZE]i32 = undefined;
};

// ============================================================================
// Vectorized Projection (scatter/gather)
// ============================================================================

/// Gather i64 values using selection vector
pub fn gatherI64(data: []const i64, sel: *const SelectionVector, out: []i64) usize {
    const count = @min(sel.count, out.len);
    if (sel.isFlat()) {
        @memcpy(out[0..count], data[0..count]);
    } else {
        const indices = sel.indices.?;
        for (indices[0..count], 0..) |idx, i| {
            out[i] = data[idx];
        }
    }
    return count;
}

/// Gather f64 values using selection vector
pub fn gatherF64(data: []const f64, sel: *const SelectionVector, out: []f64) usize {
    const count = @min(sel.count, out.len);
    if (sel.isFlat()) {
        @memcpy(out[0..count], data[0..count]);
    } else {
        const indices = sel.indices.?;
        for (indices[0..count], 0..) |idx, i| {
            out[i] = data[idx];
        }
    }
    return count;
}

// ============================================================================
// Simple Aggregate (no GROUP BY)
// ============================================================================

/// Compute SUM of f64 column with selection vector
pub fn sumF64(data: []const f64, sel: *const SelectionVector) f64 {
    var state = AggState{};
    state.updateColumnF64(data, sel);
    return state.sum;
}

/// Compute SUM of i64 column with selection vector
pub fn sumI64(data: []const i64, sel: *const SelectionVector) i64 {
    if (sel.isFlat()) {
        var sum: i64 = 0;
        var i: usize = 0;

        // SIMD path
        while (i + 2 <= data.len) : (i += 2) {
            const v: Vec2i64 = .{ data[i], data[i + 1] };
            sum += @reduce(.Add, v);
        }
        while (i < data.len) : (i += 1) {
            sum += data[i];
        }
        return sum;
    } else {
        var sum: i64 = 0;
        const indices = sel.indices.?;
        for (indices[0..sel.count]) |idx| {
            sum += data[idx];
        }
        return sum;
    }
}

/// Count rows with selection vector
pub fn countRows(sel: *const SelectionVector) usize {
    return sel.count;
}

/// Compute MIN of f64 column
pub fn minF64(data: []const f64, sel: *const SelectionVector) f64 {
    var state = AggState{};
    state.updateColumnF64(data, sel);
    return state.min;
}

/// Compute MAX of f64 column
pub fn maxF64(data: []const f64, sel: *const SelectionVector) f64 {
    var state = AggState{};
    state.updateColumnF64(data, sel);
    return state.max;
}

/// Compute AVG of f64 column
pub fn avgF64(data: []const f64, sel: *const SelectionVector) f64 {
    var state = AggState{};
    state.updateColumnF64(data, sel);
    return state.getAvg();
}

// ============================================================================
// Streaming Batch Executor - Single code path for WASM and Native
// ============================================================================

/// Column data types for streaming
pub const ColumnType = enum(u8) {
    int64,
    int32,
    float64,
    float32,
    string,
    bool_,
};

/// A batch of column data - the unit of streaming execution
pub const Batch = struct {
    /// Column data pointers (not owned)
    columns: []ColumnSlice,
    /// Selection vector for this batch
    sel: SelectionVector,
    /// Number of columns
    num_columns: usize,
    /// Offset in source data (for tracking position)
    offset: usize,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, num_columns: usize, capacity: usize) !Self {
        const columns = try allocator.alloc(ColumnSlice, num_columns);
        const sel = try SelectionVector.alloc(allocator, capacity);
        return .{
            .columns = columns,
            .sel = sel,
            .num_columns = num_columns,
            .offset = 0,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.columns);
        self.sel.deinit(allocator);
    }

    pub fn reset(self: *Self, row_count: usize, offset: usize) void {
        self.sel.count = row_count;
        self.sel.indices = null; // Flat selection (all rows)
        self.offset = offset;
    }
};

/// Typed slice for column data
pub const ColumnSlice = union(ColumnType) {
    int64: []const i64,
    int32: []const i32,
    float64: []const f64,
    float32: []const f32,
    string: []const []const u8,
    bool_: []const bool,
};

/// Column reader interface - implemented by both WASM and native
pub const ColumnReader = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Read next batch of rows for a column, returns slice of data
        readBatchI64: *const fn (ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const i64,
        readBatchF64: *const fn (ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const f64,
        readBatchI32: *const fn (ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const i32,
        readBatchF32: *const fn (ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const f32,
        readBatchString: *const fn (ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const []const u8,
        /// Get total row count
        getRowCount: *const fn (ptr: *anyopaque) usize,
        /// Get column type
        getColumnType: *const fn (ptr: *anyopaque, col_idx: usize) ColumnType,
    };

    pub fn readBatchI64(self: *const ColumnReader, col_idx: usize, offset: usize, count: usize) ?[]const i64 {
        return self.vtable.readBatchI64(self.ptr, col_idx, offset, count);
    }

    pub fn readBatchF64(self: *const ColumnReader, col_idx: usize, offset: usize, count: usize) ?[]const f64 {
        return self.vtable.readBatchF64(self.ptr, col_idx, offset, count);
    }

    pub fn readBatchI32(self: *const ColumnReader, col_idx: usize, offset: usize, count: usize) ?[]const i32 {
        return self.vtable.readBatchI32(self.ptr, col_idx, offset, count);
    }

    pub fn readBatchF32(self: *const ColumnReader, col_idx: usize, offset: usize, count: usize) ?[]const f32 {
        return self.vtable.readBatchF32(self.ptr, col_idx, offset, count);
    }

    pub fn readBatchString(self: *const ColumnReader, col_idx: usize, offset: usize, count: usize) ?[]const []const u8 {
        return self.vtable.readBatchString(self.ptr, col_idx, offset, count);
    }

    pub fn getRowCount(self: *const ColumnReader) usize {
        return self.vtable.getRowCount(self.ptr);
    }

    pub fn getColumnType(self: *const ColumnReader, col_idx: usize) ColumnType {
        return self.vtable.getColumnType(self.ptr, col_idx);
    }
};

/// Filter condition for streaming filter
pub const FilterCondition = struct {
    col_idx: usize,
    op: FilterOp,
    value: FilterValue,
};

pub const FilterValue = union(enum) {
    int64: i64,
    float64: f64,
    int32: i32,
    float32: f32,
};

/// Aggregate function type
pub const AggFunc = enum {
    count,
    sum,
    avg,
    min,
    max,
};

/// Aggregate specification
pub const AggSpec = struct {
    func: AggFunc,
    col_idx: usize,
    output_name: []const u8,
};

/// Streaming aggregator - processes batches and accumulates results
pub const StreamingAggregator = struct {
    /// Aggregation states (one per aggregate)
    states: []AggState,
    /// Aggregate specifications
    specs: []const AggSpec,
    /// Allocator
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, specs: []const AggSpec) !Self {
        const states = try allocator.alloc(AggState, specs.len);
        for (states) |*s| {
            s.* = AggState{};
        }
        return .{
            .states = states,
            .specs = specs,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.states);
    }

    /// Process a batch of data
    pub fn processBatch(self: *Self, batch: *const Batch) void {
        for (self.specs, 0..) |spec, i| {
            const col = batch.columns[spec.col_idx];
            switch (col) {
                .int64 => |data| self.states[i].updateColumnI64(data, &batch.sel),
                .float64 => |data| self.states[i].updateColumnF64(data, &batch.sel),
                .int32 => |data| {
                    // Convert to i64 for aggregation
                    if (batch.sel.isFlat()) {
                        for (data) |v| {
                            self.states[i].sum += @floatFromInt(v);
                            self.states[i].count += 1;
                            const vf: f64 = @floatFromInt(v);
                            if (vf < self.states[i].min) self.states[i].min = vf;
                            if (vf > self.states[i].max) self.states[i].max = vf;
                        }
                    } else {
                        const indices = batch.sel.indices.?;
                        for (indices[0..batch.sel.count]) |idx| {
                            const v = data[idx];
                            self.states[i].sum += @floatFromInt(v);
                            self.states[i].count += 1;
                            const vf: f64 = @floatFromInt(v);
                            if (vf < self.states[i].min) self.states[i].min = vf;
                            if (vf > self.states[i].max) self.states[i].max = vf;
                        }
                    }
                },
                .float32 => |data| {
                    if (batch.sel.isFlat()) {
                        for (data) |v| {
                            self.states[i].sum += v;
                            self.states[i].count += 1;
                            if (v < self.states[i].min) self.states[i].min = v;
                            if (v > self.states[i].max) self.states[i].max = v;
                        }
                    } else {
                        const indices = batch.sel.indices.?;
                        for (indices[0..batch.sel.count]) |idx| {
                            const v = data[idx];
                            self.states[i].sum += v;
                            self.states[i].count += 1;
                            if (v < self.states[i].min) self.states[i].min = v;
                            if (v > self.states[i].max) self.states[i].max = v;
                        }
                    }
                },
                else => {}, // String/bool not aggregatable
            }
        }
    }

    /// Get final result for an aggregate
    pub fn getResult(self: *const Self, idx: usize) f64 {
        const state = &self.states[idx];
        return switch (self.specs[idx].func) {
            .count => @floatFromInt(state.count),
            .sum => state.sum,
            .avg => state.getAvg(),
            .min => state.min,
            .max => state.max,
        };
    }

    /// Get count for an aggregate
    pub fn getCount(self: *const Self, idx: usize) u64 {
        return self.states[idx].count;
    }
};

/// Streaming GROUP BY aggregator
pub const StreamingGroupBy = struct {
    /// Hash table for group lookup
    ht: LinearHashTable,
    /// Group aggregation states
    groups: std.ArrayListUnmanaged(GroupState),
    /// Aggregate specifications
    specs: []const AggSpec,
    /// Group key column index
    key_col_idx: usize,
    /// Key data storage (for looking up keys later)
    keys: std.ArrayListUnmanaged(i64),
    /// Allocator
    allocator: std.mem.Allocator,

    const Self = @This();

    const GroupState = struct {
        first_key: i64,
        aggs: []AggState,
    };

    pub fn init(allocator: std.mem.Allocator, key_col_idx: usize, specs: []const AggSpec, expected_groups: usize) !Self {
        return .{
            .ht = try LinearHashTable.init(allocator, expected_groups),
            .groups = .{},
            .specs = specs,
            .key_col_idx = key_col_idx,
            .keys = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.ht.deinit(self.allocator);
        for (self.groups.items) |*g| {
            self.allocator.free(g.aggs);
        }
        self.groups.deinit(self.allocator);
        self.keys.deinit(self.allocator);
    }

    /// Process a batch - group keys and aggregate values
    pub fn processBatch(self: *Self, batch: *const Batch) !void {
        const key_col = batch.columns[self.key_col_idx];
        const keys = switch (key_col) {
            .int64 => |d| d,
            else => return, // Only int64 keys supported for now
        };

        // Process each row in the batch
        if (batch.sel.isFlat()) {
            for (keys, 0..) |key, row_idx| {
                try self.processRow(key, batch, row_idx);
            }
        } else {
            const indices = batch.sel.indices.?;
            for (indices[0..batch.sel.count]) |idx| {
                try self.processRow(keys[idx], batch, idx);
            }
        }
    }

    fn processRow(self: *Self, key: i64, batch: *const Batch, row_idx: usize) !void {
        if (key == NULL_INT64) return;

        // Find or create group
        const h = LinearHashTable.hash64(key);
        const hash_bits: u16 = @truncate(h >> 48);
        var pos = @as(usize, @intCast(h & self.ht.mask));

        var group_idx: ?usize = null;
        while (self.ht.entries[pos].state == LinearHashTable.OCCUPIED) {
            if (self.ht.entries[pos].hash_bits == hash_bits) {
                const gidx = self.ht.entries[pos].row_idx;
                if (self.groups.items[gidx].first_key == key) {
                    group_idx = gidx;
                    break;
                }
            }
            pos = (pos + 1) & @as(usize, @intCast(self.ht.mask));
        }

        if (group_idx == null) {
            // Create new group
            const gidx: u32 = @intCast(self.groups.items.len);
            const aggs = try self.allocator.alloc(AggState, self.specs.len);
            for (aggs) |*a| a.* = AggState{};

            try self.groups.append(self.allocator, .{
                .first_key = key,
                .aggs = aggs,
            });
            try self.keys.append(self.allocator, key);

            self.ht.entries[pos] = .{
                .hash_bits = hash_bits,
                .row_idx = gidx,
                .state = LinearHashTable.OCCUPIED,
                ._pad = 0,
                .key = key,
            };
            self.ht.count += 1;
            group_idx = gidx;
        }

        // Update aggregates for this group
        const group = &self.groups.items[group_idx.?];
        for (self.specs, 0..) |spec, i| {
            const col = batch.columns[spec.col_idx];
            switch (col) {
                .int64 => |data| {
                    const v = data[row_idx];
                    const vf: f64 = @floatFromInt(v);
                    group.aggs[i].sum += vf;
                    group.aggs[i].count += 1;
                    if (vf < group.aggs[i].min) group.aggs[i].min = vf;
                    if (vf > group.aggs[i].max) group.aggs[i].max = vf;
                },
                .float64 => |data| {
                    const v = data[row_idx];
                    group.aggs[i].sum += v;
                    group.aggs[i].count += 1;
                    if (v < group.aggs[i].min) group.aggs[i].min = v;
                    if (v > group.aggs[i].max) group.aggs[i].max = v;
                },
                else => {},
            }
        }
    }

    pub fn groupCount(self: *const Self) usize {
        return self.groups.items.len;
    }

    pub fn getGroupKey(self: *const Self, group_idx: usize) i64 {
        return self.keys.items[group_idx];
    }

    pub fn getGroupAgg(self: *const Self, group_idx: usize, agg_idx: usize, func: AggFunc) f64 {
        const state = &self.groups.items[group_idx].aggs[agg_idx];
        return switch (func) {
            .count => @floatFromInt(state.count),
            .sum => state.sum,
            .avg => state.getAvg(),
            .min => state.min,
            .max => state.max,
        };
    }
};

/// Streaming executor - processes queries in batches
/// This is the SINGLE CODE PATH for both WASM and Native
pub const StreamingExecutor = struct {
    allocator: std.mem.Allocator,
    batch: Batch,
    filter_sel: SelectionVector,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, num_columns: usize) !Self {
        return .{
            .allocator = allocator,
            .batch = try Batch.init(allocator, num_columns, VECTOR_SIZE),
            .filter_sel = try SelectionVector.alloc(allocator, VECTOR_SIZE),
        };
    }

    pub fn deinit(self: *Self) void {
        self.batch.deinit(self.allocator);
        self.filter_sel.deinit(self.allocator);
    }

    /// Execute a simple aggregate query (no GROUP BY)
    /// Returns: array of aggregate results
    pub fn executeAggregate(
        self: *Self,
        reader: *const ColumnReader,
        filter: ?FilterCondition,
        agg_specs: []const AggSpec,
    ) ![]f64 {
        var agg = try StreamingAggregator.init(self.allocator, agg_specs);
        defer agg.deinit();

        const total_rows = reader.getRowCount();
        var offset: usize = 0;

        // Process in batches
        while (offset < total_rows) {
            const batch_size = @min(VECTOR_SIZE, total_rows - offset);

            // Load columns for this batch
            for (agg_specs, 0..) |spec, i| {
                const col_type = reader.getColumnType(spec.col_idx);
                self.batch.columns[i] = switch (col_type) {
                    .int64 => .{ .int64 = reader.readBatchI64(spec.col_idx, offset, batch_size) orelse &[_]i64{} },
                    .float64 => .{ .float64 = reader.readBatchF64(spec.col_idx, offset, batch_size) orelse &[_]f64{} },
                    .int32 => .{ .int32 = reader.readBatchI32(spec.col_idx, offset, batch_size) orelse &[_]i32{} },
                    .float32 => .{ .float32 = reader.readBatchF32(spec.col_idx, offset, batch_size) orelse &[_]f32{} },
                    else => .{ .int64 = &[_]i64{} },
                };
            }

            // Reset batch selection to all rows
            self.batch.reset(batch_size, offset);

            // Apply filter if present
            if (filter) |f| {
                const col = self.batch.columns[f.col_idx];
                _ = switch (col) {
                    .int64 => |data| filterI64(data, f.op, f.value.int64, &self.batch.sel, &self.filter_sel),
                    .float64 => |data| filterF64(data, f.op, f.value.float64, &self.batch.sel, &self.filter_sel),
                    else => self.batch.sel.count,
                };
                self.batch.sel = self.filter_sel;
            }

            // Process batch
            agg.processBatch(&self.batch);

            offset += batch_size;
        }

        // Return results
        const results = try self.allocator.alloc(f64, agg_specs.len);
        for (0..agg_specs.len) |i| {
            results[i] = agg.getResult(i);
        }
        return results;
    }

    /// Execute a GROUP BY query
    /// Returns: (keys, agg_results) where agg_results[group][agg_idx]
    pub fn executeGroupBy(
        self: *Self,
        reader: *const ColumnReader,
        key_col_idx: usize,
        filter: ?FilterCondition,
        agg_specs: []const AggSpec,
    ) !struct { keys: []i64, results: [][]f64 } {
        var gb = try StreamingGroupBy.init(self.allocator, key_col_idx, agg_specs, 1024);
        defer gb.deinit();

        const total_rows = reader.getRowCount();
        var offset: usize = 0;

        // Determine columns needed
        const num_cols = agg_specs.len + 1; // +1 for key column

        // Process in batches
        while (offset < total_rows) {
            const batch_size = @min(VECTOR_SIZE, total_rows - offset);

            // Load key column
            const key_type = reader.getColumnType(key_col_idx);
            self.batch.columns[0] = switch (key_type) {
                .int64 => .{ .int64 = reader.readBatchI64(key_col_idx, offset, batch_size) orelse &[_]i64{} },
                else => .{ .int64 = &[_]i64{} },
            };

            // Load aggregate columns
            for (agg_specs, 1..) |spec, i| {
                const col_type = reader.getColumnType(spec.col_idx);
                self.batch.columns[i] = switch (col_type) {
                    .int64 => .{ .int64 = reader.readBatchI64(spec.col_idx, offset, batch_size) orelse &[_]i64{} },
                    .float64 => .{ .float64 = reader.readBatchF64(spec.col_idx, offset, batch_size) orelse &[_]f64{} },
                    else => .{ .int64 = &[_]i64{} },
                };
            }
            _ = num_cols;

            // Reset batch
            self.batch.reset(batch_size, offset);

            // Apply filter if present
            if (filter) |f| {
                // Find column index in batch
                const col = if (f.col_idx == key_col_idx)
                    self.batch.columns[0]
                else blk: {
                    for (agg_specs, 1..) |spec, i| {
                        if (spec.col_idx == f.col_idx) break :blk self.batch.columns[i];
                    }
                    break :blk self.batch.columns[0];
                };
                _ = switch (col) {
                    .int64 => |data| filterI64(data, f.op, f.value.int64, &self.batch.sel, &self.filter_sel),
                    .float64 => |data| filterF64(data, f.op, f.value.float64, &self.batch.sel, &self.filter_sel),
                    else => self.batch.sel.count,
                };
                self.batch.sel = self.filter_sel;
            }

            // Adjust column indices for batch layout (key is at 0, aggs start at 1)
            var adjusted_batch = self.batch;
            adjusted_batch.columns = self.batch.columns;

            // Create adjusted specs with batch-local indices
            const adjusted_specs = try self.allocator.alloc(AggSpec, agg_specs.len);
            defer self.allocator.free(adjusted_specs);
            for (agg_specs, 0..) |spec, i| {
                adjusted_specs[i] = .{
                    .func = spec.func,
                    .col_idx = i + 1, // Offset by 1 since key is at 0
                    .output_name = spec.output_name,
                };
            }

            // Process with adjusted group by
            var adjusted_gb = StreamingGroupBy{
                .ht = gb.ht,
                .groups = gb.groups,
                .specs = adjusted_specs,
                .key_col_idx = 0, // Key is at index 0 in batch
                .keys = gb.keys,
                .allocator = self.allocator,
            };
            try adjusted_gb.processBatch(&adjusted_batch);
            gb.ht = adjusted_gb.ht;
            gb.groups = adjusted_gb.groups;
            gb.keys = adjusted_gb.keys;

            offset += batch_size;
        }

        // Build result arrays
        const num_groups = gb.groupCount();
        const keys = try self.allocator.alloc(i64, num_groups);
        const results = try self.allocator.alloc([]f64, num_groups);

        for (0..num_groups) |g| {
            keys[g] = gb.getGroupKey(g);
            results[g] = try self.allocator.alloc(f64, agg_specs.len);
            for (0..agg_specs.len) |a| {
                results[g][a] = gb.getGroupAgg(g, a, agg_specs[a].func);
            }
        }

        return .{ .keys = keys, .results = results };
    }

    /// Execute COUNT(*) with optional filter - optimized path
    pub fn executeCount(
        self: *Self,
        reader: *const ColumnReader,
        filter: ?FilterCondition,
        filter_col_idx: usize,
    ) !u64 {
        var total_count: u64 = 0;
        const total_rows = reader.getRowCount();
        var offset: usize = 0;

        while (offset < total_rows) {
            const batch_size = @min(VECTOR_SIZE, total_rows - offset);

            if (filter) |f| {
                // Load filter column
                const col_type = reader.getColumnType(filter_col_idx);
                const col_data: ColumnSlice = switch (col_type) {
                    .int64 => .{ .int64 = reader.readBatchI64(filter_col_idx, offset, batch_size) orelse &[_]i64{} },
                    .float64 => .{ .float64 = reader.readBatchF64(filter_col_idx, offset, batch_size) orelse &[_]f64{} },
                    else => .{ .int64 = &[_]i64{} },
                };

                self.batch.columns[0] = col_data;
                self.batch.reset(batch_size, offset);

                // Apply filter
                const count = switch (col_data) {
                    .int64 => |data| filterI64(data, f.op, f.value.int64, &self.batch.sel, &self.filter_sel),
                    .float64 => |data| filterF64(data, f.op, f.value.float64, &self.batch.sel, &self.filter_sel),
                    else => batch_size,
                };
                total_count += count;
            } else {
                total_count += batch_size;
            }

            offset += batch_size;
        }

        return total_count;
    }
};

// ============================================================================
// Streaming Hash Join - Memory-efficient JOIN for large datasets
// ============================================================================

/// Streaming Hash Join - builds hash table incrementally, streams probe side
/// Key insight: Build side must fit in hash table, but probe side can be arbitrarily large
pub const StreamingHashJoin = struct {
    allocator: std.mem.Allocator,
    ht: LinearHashTable,
    build_keys: std.ArrayList(i64),
    build_complete: bool,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, estimated_build_rows: usize) !Self {
        return .{
            .allocator = allocator,
            .ht = try LinearHashTable.init(allocator, estimated_build_rows),
            .build_keys = std.ArrayList(i64).init(allocator),
            .build_complete = false,
        };
    }

    pub fn deinit(self: *Self) void {
        self.ht.deinit(self.allocator);
        self.build_keys.deinit();
    }

    /// Add a batch of keys from the build side
    /// Call this repeatedly to stream the build side
    pub fn addBuildBatch(self: *Self, keys: []const i64) !void {
        const start_idx = self.build_keys.items.len;
        try self.build_keys.appendSlice(keys);

        // Insert into hash table
        for (keys, start_idx..) |key, idx| {
            if (key != NULL_INT64) {
                self.ht.insert(key, @intCast(idx));
            }
        }
    }

    /// Mark build phase as complete
    pub fn finishBuild(self: *Self) void {
        self.build_complete = true;
    }

    /// Probe with a batch of keys from the probe side
    /// Returns (probe_indices, build_indices) for matching rows
    pub fn probeBatch(
        self: *Self,
        probe_keys: []const i64,
        probe_offset: usize,
        max_results: usize,
    ) !struct { probe: []u32, build: []u32 } {
        if (!self.build_complete) return error.BuildNotComplete;

        var probe_out = try self.allocator.alloc(u32, max_results);
        errdefer self.allocator.free(probe_out);
        var build_out = try self.allocator.alloc(u32, max_results);
        errdefer self.allocator.free(build_out);

        var out_count: usize = 0;
        var match_buf: [64]u32 = undefined;

        for (probe_keys, 0..) |key, local_idx| {
            if (key == NULL_INT64) continue;
            if (out_count >= max_results) break;

            // Find all matches in build side
            const matches = self.ht.probeAll(key, self.build_keys.items, &match_buf);
            for (match_buf[0..matches]) |build_idx| {
                if (out_count >= max_results) break;
                probe_out[out_count] = @intCast(probe_offset + local_idx);
                build_out[out_count] = build_idx;
                out_count += 1;
            }
        }

        return .{
            .probe = probe_out[0..out_count],
            .build = build_out[0..out_count],
        };
    }

    /// Execute streaming hash join between two column readers
    /// Returns all (left_idx, right_idx) matching pairs
    pub fn executeStreaming(
        allocator: std.mem.Allocator,
        build_reader: *const ColumnReader,
        build_col_idx: usize,
        probe_reader: *const ColumnReader,
        probe_col_idx: usize,
        max_results: usize,
    ) !struct { left: []u32, right: []u32 } {
        const build_rows = build_reader.getRowCount();

        // Initialize hash join
        var join = try Self.init(allocator, build_rows);
        defer join.deinit();

        // Build phase: stream build side
        var build_offset: usize = 0;
        while (build_offset < build_rows) {
            const batch_size = @min(VECTOR_SIZE, build_rows - build_offset);
            const keys = build_reader.readBatchI64(build_col_idx, build_offset, batch_size) orelse break;
            try join.addBuildBatch(keys);
            build_offset += batch_size;
        }
        join.finishBuild();

        // Probe phase: stream probe side and collect results
        var all_probe = std.ArrayList(u32).init(allocator);
        errdefer all_probe.deinit();
        var all_build = std.ArrayList(u32).init(allocator);
        errdefer all_build.deinit();

        const probe_rows = probe_reader.getRowCount();
        var probe_offset: usize = 0;

        while (probe_offset < probe_rows and all_probe.items.len < max_results) {
            const batch_size = @min(VECTOR_SIZE, probe_rows - probe_offset);
            const keys = probe_reader.readBatchI64(probe_col_idx, probe_offset, batch_size) orelse break;

            const remaining = max_results - all_probe.items.len;
            const result = try join.probeBatch(keys, probe_offset, remaining);
            defer {
                allocator.free(result.probe);
                allocator.free(result.build);
            }

            try all_probe.appendSlice(result.probe);
            try all_build.appendSlice(result.build);

            probe_offset += batch_size;
        }

        return .{
            .left = try all_probe.toOwnedSlice(),
            .right = try all_build.toOwnedSlice(),
        };
    }
};

// ============================================================================
// Streaming Window Functions - Memory-efficient window computations
// ============================================================================

/// Window function types
pub const WindowFunc = enum {
    row_number,
    rank,
    dense_rank,
    lag,
    lead,
    first_value,
    last_value,
    sum,
    avg,
    min,
    max,
    count,
};

/// Window function specification
pub const WindowSpec = struct {
    func: WindowFunc,
    value_col_idx: usize, // Column for value functions (SUM, AVG, etc.)
    partition_col_idx: ?usize, // PARTITION BY column (null = no partitioning)
    order_col_idx: ?usize, // ORDER BY column (null = no ordering)
    offset: i32 = 1, // For LAG/LEAD
    default_value: f64 = 0, // For LAG/LEAD when out of bounds
};

/// Streaming Window Executor - computes window functions over partitions
/// Accumulates data by partition, sorts, then computes window values
pub const StreamingWindowExecutor = struct {
    allocator: std.mem.Allocator,

    // Partition buffers: key -> (indices, values)
    partitions: std.AutoHashMap(i64, PartitionData),

    const Self = @This();

    const PartitionData = struct {
        indices: std.ArrayList(u32),
        values: std.ArrayList(f64),
        order_keys: std.ArrayList(i64),
    };

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .partitions = std.AutoHashMap(i64, PartitionData).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.partitions.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.indices.deinit();
            entry.value_ptr.values.deinit();
            entry.value_ptr.order_keys.deinit();
        }
        self.partitions.deinit();
    }

    /// Add a batch of rows to partitions
    pub fn addBatch(
        self: *Self,
        partition_keys: ?[]const i64,
        order_keys: ?[]const i64,
        values: []const f64,
        base_offset: usize,
    ) !void {
        for (values, 0..) |val, i| {
            const part_key: i64 = if (partition_keys) |pk| pk[i] else 0;
            const ord_key: i64 = if (order_keys) |ok| ok[i] else @intCast(base_offset + i);

            const gop = try self.partitions.getOrPut(part_key);
            if (!gop.found_existing) {
                gop.value_ptr.* = .{
                    .indices = std.ArrayList(u32).init(self.allocator),
                    .values = std.ArrayList(f64).init(self.allocator),
                    .order_keys = std.ArrayList(i64).init(self.allocator),
                };
            }

            try gop.value_ptr.indices.append(@intCast(base_offset + i));
            try gop.value_ptr.values.append(val);
            try gop.value_ptr.order_keys.append(ord_key);
        }
    }

    /// Compute window function and return results
    /// Returns (original_indices, window_values)
    pub fn compute(self: *Self, spec: WindowSpec) !struct { indices: []u32, values: []f64 } {
        var all_indices = std.ArrayList(u32).init(self.allocator);
        errdefer all_indices.deinit();
        var all_values = std.ArrayList(f64).init(self.allocator);
        errdefer all_values.deinit();

        var iter = self.partitions.iterator();
        while (iter.next()) |entry| {
            const part = entry.value_ptr;
            const n = part.indices.items.len;
            if (n == 0) continue;

            // Sort by order key within partition
            var sorted_idx = try self.allocator.alloc(usize, n);
            defer self.allocator.free(sorted_idx);
            for (0..n) |i| sorted_idx[i] = i;

            std.mem.sort(usize, sorted_idx, part.order_keys.items, struct {
                fn lessThan(keys: []const i64, a: usize, b: usize) bool {
                    return keys[a] < keys[b];
                }
            }.lessThan);

            // Compute window values in sorted order
            for (sorted_idx, 0..) |orig_idx, rank| {
                const win_val: f64 = switch (spec.func) {
                    .row_number => @floatFromInt(rank + 1),
                    .rank => blk: {
                        // Same key = same rank
                        if (rank == 0) break :blk 1;
                        const prev_key = part.order_keys.items[sorted_idx[rank - 1]];
                        const curr_key = part.order_keys.items[orig_idx];
                        if (curr_key == prev_key) {
                            // Find first occurrence
                            var r: usize = rank;
                            while (r > 0 and part.order_keys.items[sorted_idx[r - 1]] == curr_key) r -= 1;
                            break :blk @floatFromInt(r + 1);
                        }
                        break :blk @floatFromInt(rank + 1);
                    },
                    .dense_rank => blk: {
                        if (rank == 0) break :blk 1;
                        var dense: usize = 1;
                        var prev_key = part.order_keys.items[sorted_idx[0]];
                        for (1..rank + 1) |r| {
                            const curr_key = part.order_keys.items[sorted_idx[r]];
                            if (curr_key != prev_key) {
                                dense += 1;
                                prev_key = curr_key;
                            }
                        }
                        break :blk @floatFromInt(dense);
                    },
                    .lag => blk: {
                        const off: usize = @intCast(@max(0, spec.offset));
                        if (rank < off) break :blk spec.default_value;
                        break :blk part.values.items[sorted_idx[rank - off]];
                    },
                    .lead => blk: {
                        const off: usize = @intCast(@max(0, spec.offset));
                        if (rank + off >= n) break :blk spec.default_value;
                        break :blk part.values.items[sorted_idx[rank + off]];
                    },
                    .first_value => part.values.items[sorted_idx[0]],
                    .last_value => part.values.items[sorted_idx[n - 1]],
                    .sum => blk: {
                        var sum: f64 = 0;
                        for (part.values.items) |v| sum += v;
                        break :blk sum;
                    },
                    .avg => blk: {
                        var sum: f64 = 0;
                        for (part.values.items) |v| sum += v;
                        break :blk sum / @as(f64, @floatFromInt(n));
                    },
                    .min => blk: {
                        var m: f64 = part.values.items[0];
                        for (part.values.items[1..]) |v| if (v < m) {
                            m = v;
                        };
                        break :blk m;
                    },
                    .max => blk: {
                        var m: f64 = part.values.items[0];
                        for (part.values.items[1..]) |v| if (v > m) {
                            m = v;
                        };
                        break :blk m;
                    },
                    .count => @floatFromInt(n),
                };

                try all_indices.append(part.indices.items[orig_idx]);
                try all_values.append(win_val);
            }
        }

        return .{
            .indices = try all_indices.toOwnedSlice(),
            .values = try all_values.toOwnedSlice(),
        };
    }

    /// Execute streaming window function on a column reader
    pub fn executeStreaming(
        allocator: std.mem.Allocator,
        reader: *const ColumnReader,
        spec: WindowSpec,
    ) !struct { indices: []u32, values: []f64 } {
        var executor = Self.init(allocator);
        defer executor.deinit();

        const total_rows = reader.getRowCount();
        var offset: usize = 0;

        while (offset < total_rows) {
            const batch_size = @min(VECTOR_SIZE, total_rows - offset);

            // Read partition keys (if partitioning)
            var part_keys: ?[]const i64 = null;
            if (spec.partition_col_idx) |col| {
                part_keys = reader.readBatchI64(col, offset, batch_size);
            }

            // Read order keys (if ordering)
            var ord_keys: ?[]const i64 = null;
            if (spec.order_col_idx) |col| {
                ord_keys = reader.readBatchI64(col, offset, batch_size);
            }

            // Read values
            const values = reader.readBatchF64(spec.value_col_idx, offset, batch_size) orelse break;

            try executor.addBatch(part_keys, ord_keys, values, offset);
            offset += batch_size;
        }

        return executor.compute(spec);
    }
};

// ============================================================================
// String Hash Group By - GROUP BY with string keys
// ============================================================================

/// Hash function for strings (FNV-1a)
pub fn hashString(s: []const u8) u64 {
    var hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for (s) |byte| {
        hash ^= byte;
        hash *%= 0x100000001b3; // FNV prime
    }
    return hash;
}

/// String Hash Group By - groups by string keys with aggregate accumulators
pub const StringHashGroupBy = struct {
    allocator: std.mem.Allocator,
    groups: std.StringHashMap(GroupState),
    agg_count: usize,

    const Self = @This();

    const GroupState = struct {
        counts: []u64,
        sums: []f64,
        mins: []f64,
        maxs: []f64,
    };

    pub fn init(allocator: std.mem.Allocator, num_aggs: usize) Self {
        return .{
            .allocator = allocator,
            .groups = std.StringHashMap(GroupState).init(allocator),
            .agg_count = num_aggs,
        };
    }

    pub fn deinit(self: *Self) void {
        var iter = self.groups.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.counts);
            self.allocator.free(entry.value_ptr.sums);
            self.allocator.free(entry.value_ptr.mins);
            self.allocator.free(entry.value_ptr.maxs);
            // Free the duplicated key
            self.allocator.free(entry.key_ptr.*);
        }
        self.groups.deinit();
    }

    /// Add a batch of (key, values) pairs
    pub fn addBatch(
        self: *Self,
        keys: []const []const u8,
        values: []const []const f64, // values[agg_idx][row_idx]
    ) !void {
        for (keys, 0..) |key, row| {
            const gop = try self.groups.getOrPut(key);
            if (!gop.found_existing) {
                // Duplicate the key string
                const key_copy = try self.allocator.dupe(u8, key);
                gop.key_ptr.* = key_copy;

                // Initialize group state
                gop.value_ptr.* = .{
                    .counts = try self.allocator.alloc(u64, self.agg_count),
                    .sums = try self.allocator.alloc(f64, self.agg_count),
                    .mins = try self.allocator.alloc(f64, self.agg_count),
                    .maxs = try self.allocator.alloc(f64, self.agg_count),
                };
                for (0..self.agg_count) |a| {
                    gop.value_ptr.counts[a] = 0;
                    gop.value_ptr.sums[a] = 0;
                    gop.value_ptr.mins[a] = std.math.inf(f64);
                    gop.value_ptr.maxs[a] = -std.math.inf(f64);
                }
            }

            // Update accumulators
            for (0..self.agg_count) |a| {
                const val = values[a][row];
                if (!std.math.isNan(val) and val != NULL_FLOAT64) {
                    gop.value_ptr.counts[a] += 1;
                    gop.value_ptr.sums[a] += val;
                    if (val < gop.value_ptr.mins[a]) gop.value_ptr.mins[a] = val;
                    if (val > gop.value_ptr.maxs[a]) gop.value_ptr.maxs[a] = val;
                }
            }
        }
    }

    /// Get final results
    /// Returns (keys, results[group_idx][agg_idx])
    pub fn getResults(
        self: *Self,
        agg_funcs: []const AggFunc,
    ) !struct { keys: [][]const u8, results: [][]f64 } {
        const num_groups = self.groups.count();
        const keys = try self.allocator.alloc([]const u8, num_groups);
        errdefer self.allocator.free(keys);

        const results = try self.allocator.alloc([]f64, num_groups);
        errdefer self.allocator.free(results);

        var iter = self.groups.iterator();
        var idx: usize = 0;
        while (iter.next()) |entry| {
            keys[idx] = entry.key_ptr.*;

            results[idx] = try self.allocator.alloc(f64, self.agg_count);
            for (0..self.agg_count) |a| {
                const state = entry.value_ptr;
                results[idx][a] = switch (agg_funcs[a]) {
                    .count => @floatFromInt(state.counts[a]),
                    .sum => state.sums[a],
                    .avg => if (state.counts[a] > 0)
                        state.sums[a] / @as(f64, @floatFromInt(state.counts[a]))
                    else
                        0,
                    .min => if (state.mins[a] == std.math.inf(f64)) 0 else state.mins[a],
                    .max => if (state.maxs[a] == -std.math.inf(f64)) 0 else state.maxs[a],
                };
            }
            idx += 1;
        }

        return .{ .keys = keys, .results = results };
    }
};

// ============================================================================
// ArrayColumnReader - Wraps pre-loaded column arrays
// ============================================================================

/// Column data holder for ArrayColumnReader
pub const ColumnData = struct {
    col_type: ColumnType,
    int64_data: ?[]const i64 = null,
    int32_data: ?[]const i32 = null,
    float64_data: ?[]const f64 = null,
    float32_data: ?[]const f32 = null,
    string_data: ?[]const []const u8 = null,
};

/// ArrayColumnReader - wraps pre-loaded column arrays for streaming execution
/// Use this when columns are already loaded (e.g., from Lance/Parquet files)
pub const ArrayColumnReader = struct {
    columns: []const ColumnData,
    row_count: usize,

    const Self = @This();

    pub fn init(columns: []const ColumnData, row_count: usize) Self {
        return .{
            .columns = columns,
            .row_count = row_count,
        };
    }

    /// Get a ColumnReader interface for use with StreamingExecutor
    pub fn reader(self: *Self) ColumnReader {
        return .{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    const vtable = ColumnReader.VTable{
        .readBatchI64 = readBatchI64Fn,
        .readBatchF64 = readBatchF64Fn,
        .readBatchI32 = readBatchI32Fn,
        .readBatchF32 = readBatchF32Fn,
        .readBatchString = readBatchStringFn,
        .getRowCount = getRowCountFn,
        .getColumnType = getColumnTypeFn,
    };

    fn readBatchI64Fn(ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const i64 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        if (col_idx >= self.columns.len) return null;
        const data = self.columns[col_idx].int64_data orelse return null;
        if (offset >= data.len) return null;
        const end = @min(offset + count, data.len);
        return data[offset..end];
    }

    fn readBatchF64Fn(ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const f64 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        if (col_idx >= self.columns.len) return null;
        const data = self.columns[col_idx].float64_data orelse return null;
        if (offset >= data.len) return null;
        const end = @min(offset + count, data.len);
        return data[offset..end];
    }

    fn readBatchI32Fn(ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const i32 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        if (col_idx >= self.columns.len) return null;
        const data = self.columns[col_idx].int32_data orelse return null;
        if (offset >= data.len) return null;
        const end = @min(offset + count, data.len);
        return data[offset..end];
    }

    fn readBatchF32Fn(ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const f32 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        if (col_idx >= self.columns.len) return null;
        const data = self.columns[col_idx].float32_data orelse return null;
        if (offset >= data.len) return null;
        const end = @min(offset + count, data.len);
        return data[offset..end];
    }

    fn readBatchStringFn(ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const []const u8 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        if (col_idx >= self.columns.len) return null;
        const data = self.columns[col_idx].string_data orelse return null;
        if (offset >= data.len) return null;
        const end = @min(offset + count, data.len);
        return data[offset..end];
    }

    fn getRowCountFn(ptr: *anyopaque) usize {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.row_count;
    }

    fn getColumnTypeFn(ptr: *anyopaque, col_idx: usize) ColumnType {
        const self: *Self = @ptrCast(@alignCast(ptr));
        if (col_idx >= self.columns.len) return .int64;
        return self.columns[col_idx].col_type;
    }
};

// ============================================================================
// Convenience functions for executing queries
// ============================================================================

/// Execute COUNT(*) WHERE filter on pre-loaded column
pub fn executeCountWhere(
    allocator: std.mem.Allocator,
    filter_col: []const f64,
    op: FilterOp,
    value: f64,
) !u64 {
    const columns = [_]ColumnData{
        .{ .col_type = .float64, .float64_data = filter_col },
    };

    var col_reader = ArrayColumnReader.init(&columns, filter_col.len);
    var reader = col_reader.reader();

    var exec = try StreamingExecutor.init(allocator, 1);
    defer exec.deinit();

    return exec.executeCount(&reader, .{
        .col_idx = 0,
        .op = op,
        .value = .{ .float64 = value },
    }, 0);
}

/// Execute SUM(col) on pre-loaded column
pub fn executeSumF64(
    allocator: std.mem.Allocator,
    col: []const f64,
) !f64 {
    const columns = [_]ColumnData{
        .{ .col_type = .float64, .float64_data = col },
    };

    var col_reader = ArrayColumnReader.init(&columns, col.len);
    var reader = col_reader.reader();

    var exec = try StreamingExecutor.init(allocator, 1);
    defer exec.deinit();

    const specs = [_]AggSpec{
        .{ .func = .sum, .col_idx = 0, .output_name = "sum" },
    };

    const results = try exec.executeAggregate(&reader, null, &specs);
    defer allocator.free(results);

    return results[0];
}

/// Execute GROUP BY with SUM on pre-loaded columns
pub fn executeGroupBySumF64(
    allocator: std.mem.Allocator,
    key_col: []const i64,
    value_col: []const f64,
) !struct { keys: []i64, sums: []f64 } {
    const columns = [_]ColumnData{
        .{ .col_type = .int64, .int64_data = key_col },
        .{ .col_type = .float64, .float64_data = value_col },
    };

    var col_reader = ArrayColumnReader.init(&columns, key_col.len);
    var reader = col_reader.reader();

    var exec = try StreamingExecutor.init(allocator, 2);
    defer exec.deinit();

    const specs = [_]AggSpec{
        .{ .func = .sum, .col_idx = 1, .output_name = "sum" },
    };

    const result = try exec.executeGroupBy(&reader, 0, null, &specs);

    // Extract sums from results
    const sums = try allocator.alloc(f64, result.keys.len);
    for (result.results, 0..) |r, i| {
        sums[i] = r[0];
        allocator.free(r);
    }
    allocator.free(result.results);

    return .{ .keys = result.keys, .sums = sums };
}

// ============================================================================
// Tests
// ============================================================================

test "LinearHashTable basic" {
    const allocator = std.testing.allocator;
    var ht = try LinearHashTable.init(allocator, 100);
    defer ht.deinit(allocator);

    ht.insert(42, 0);
    ht.insert(100, 1);
    ht.insert(42, 2); // Duplicate key

    try std.testing.expectEqual(@as(?u32, 0), ht.probe(42)); // First match
    try std.testing.expectEqual(@as(?u32, 1), ht.probe(100));
    try std.testing.expectEqual(@as(?u32, null), ht.probe(999));
}

test "LinearHashTable buildFromColumn" {
    const allocator = std.testing.allocator;
    const data = [_]i64{ 1, 2, 3, 4, 5, NULL_INT64, 7, 8 };

    var ht = try LinearHashTable.init(allocator, data.len);
    defer ht.deinit(allocator);
    ht.buildFromColumn(&data);

    try std.testing.expectEqual(@as(usize, 7), ht.count); // 7 non-null values
    try std.testing.expectEqual(@as(?u32, 0), ht.probe(1));
    try std.testing.expectEqual(@as(?u32, 4), ht.probe(5));
    try std.testing.expectEqual(@as(?u32, null), ht.probe(6)); // Was NULL
}

test "AggState SIMD" {
    var state = AggState{};
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const sel = SelectionVector.all(data.len);

    state.updateColumnF64(&data, &sel);

    try std.testing.expectEqual(@as(f64, 36.0), state.sum);
    try std.testing.expectEqual(@as(u64, 8), state.count);
    try std.testing.expectEqual(@as(f64, 1.0), state.min);
    try std.testing.expectEqual(@as(f64, 8.0), state.max);
}

test "filterI64 basic" {
    const allocator = std.testing.allocator;
    const data = [_]i64{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const input_sel = SelectionVector.all(data.len);

    var output_sel = try SelectionVector.alloc(allocator, data.len);
    defer output_sel.deinit(allocator);

    const count = filterI64(&data, .gt, 5, &input_sel, &output_sel);

    try std.testing.expectEqual(@as(usize, 3), count); // 6, 7, 8
    try std.testing.expectEqual(@as(u32, 5), output_sel.get(0)); // index of 6
    try std.testing.expectEqual(@as(u32, 6), output_sel.get(1)); // index of 7
    try std.testing.expectEqual(@as(u32, 7), output_sel.get(2)); // index of 8
}

test "filterF64 basic" {
    const allocator = std.testing.allocator;
    const data = [_]f64{ 10.0, 50.0, 100.0, 150.0, 200.0 };
    const input_sel = SelectionVector.all(data.len);

    var output_sel = try SelectionVector.alloc(allocator, data.len);
    defer output_sel.deinit(allocator);

    const count = filterF64(&data, .gt, 100.0, &input_sel, &output_sel);

    try std.testing.expectEqual(@as(usize, 2), count); // 150, 200
}

test "HashGroupBy basic" {
    const allocator = std.testing.allocator;
    const keys = [_]i64{ 1, 2, 1, 2, 1, 3 };
    const values = [_]f64{ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0 };

    var gb = try HashGroupBy.init(allocator, 10);
    defer gb.deinit();

    try gb.buildGroups(&keys);
    gb.aggregateF64(&keys, &values);

    try std.testing.expectEqual(@as(usize, 3), gb.groupCount()); // 3 distinct keys

    // Find group for key=1 and check sum
    for (0..gb.groupCount()) |i| {
        const key = gb.getGroupKey(i, &keys);
        const agg = gb.getGroupAgg(i);
        if (key == 1) {
            try std.testing.expectEqual(@as(f64, 90.0), agg.sum); // 10+30+50
            try std.testing.expectEqual(@as(u64, 3), agg.count);
        }
    }
}

test "sumI64 with selection" {
    const allocator = std.testing.allocator;
    const data = [_]i64{ 1, 2, 3, 4, 5, 6, 7, 8 };

    // Test flat selection
    const sel_all = SelectionVector.all(data.len);
    try std.testing.expectEqual(@as(i64, 36), sumI64(&data, &sel_all));

    // Test with selection vector
    var sel = try SelectionVector.alloc(allocator, 3);
    defer sel.deinit(allocator);
    sel.storage.?[0] = 0; // index 0 -> value 1
    sel.storage.?[1] = 2; // index 2 -> value 3
    sel.storage.?[2] = 4; // index 4 -> value 5
    sel.count = 3;
    sel.indices = sel.storage.?[0..3];

    try std.testing.expectEqual(@as(i64, 9), sumI64(&data, &sel)); // 1+3+5
}
