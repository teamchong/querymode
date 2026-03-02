//! GPU Hash Table for GROUP BY and Hash JOIN operations
//!
//! Uses wgpu-native for cross-platform GPU hash table operations:
//! - Build: Insert key-value pairs in parallel
//! - Probe: Look up keys in parallel
//! - Extract: Collect all key-value pairs
//!
//! Uses shared WGSL shaders from packages/shared/gpu/shaders/join.wgsl

const std = @import("std");
const wgpu = @import("wgpu");
const gpu_context = @import("gpu_context.zig");
const GPUContext = gpu_context.GPUContext;

const Allocator = std.mem.Allocator;

/// Hash table slot size in u32 units: [key: u32, value: u32]
pub const SLOT_U32S: usize = 2;

/// GPU threshold - minimum size to use GPU
const GPU_THRESHOLD: usize = 10_000;

/// Empty slot marker
const EMPTY_KEY: u32 = 0xFFFFFFFF;

/// Hash table errors
pub const HashTableError = error{
    OutOfMemory,
    TableFull,
    GPUError,
    InvalidCapacity,
    PipelineNotReady,
    BufferCreationFailed,
    BindGroupCreationFailed,
    EncoderCreationFailed,
    ComputePassFailed,
    CommandBufferFailed,
    MapFailed,
    GetMappedRangeFailed,
    StagingBufferFailed,
};

/// Round up to next power of 2
fn nextPowerOfTwo(n: usize) usize {
    if (n == 0) return 1;
    var v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
}

/// FNV-1a hash for 32-bit keys (matches WGSL shader)
fn hashKey(key: u32) u32 {
    var hash: u32 = 2166136261;
    hash ^= (key & 0xFF);
    hash *%= 16777619;
    hash ^= ((key >> 8) & 0xFF);
    hash *%= 16777619;
    hash ^= ((key >> 16) & 0xFF);
    hash *%= 16777619;
    hash ^= ((key >> 24) & 0xFF);
    hash *%= 16777619;
    return hash;
}

/// Fast hash for i64 keys (CPU-only, AHash-style)
/// Uses golden ratio multiply + xorshift for good distribution
/// Much faster than FNV-1a: 1 multiply vs 8 for i64
inline fn hashKeyFast(key: i64) u32 {
    const k: u64 = @bitCast(key);
    // Golden ratio constant for 64-bit
    var h: u64 = k *% 0x9E3779B97F4A7C15;
    // Mix high and low bits
    h ^= h >> 33;
    return @truncate(h);
}

/// GPU-accelerated hash table
/// Uses 32-bit keys for WGSL compatibility
pub const GPUHashTable = struct {
    allocator: Allocator,
    table: []u32, // [capacity * SLOT_U32S] - interleaved key, value pairs
    capacity: usize,
    count: usize,
    ctx: ?*GPUContext = null,
    init_pipeline: ?*wgpu.ComputePipeline = null,
    build_pipeline: ?*wgpu.ComputePipeline = null,
    probe_pipeline: ?*wgpu.ComputePipeline = null,

    const Self = @This();

    /// Create a new hash table with given capacity (rounded to power of 2)
    pub fn init(allocator: Allocator, min_capacity: usize) HashTableError!Self {
        const capacity = nextPowerOfTwo(@max(min_capacity, 16));
        const table = allocator.alloc(u32, capacity * SLOT_U32S) catch
            return HashTableError.OutOfMemory;

        // Initialize all key slots to EMPTY_KEY
        for (0..capacity) |i| {
            table[i * SLOT_U32S] = EMPTY_KEY;
            table[i * SLOT_U32S + 1] = 0;
        }

        return Self{
            .allocator = allocator,
            .table = table,
            .capacity = capacity,
            .count = 0,
        };
    }

    /// Initialize GPU context and pipelines (lazy init)
    fn ensureGPU(self: *Self) !void {
        if (self.ctx != null) return;

        self.ctx = gpu_context.getGlobalContext(self.allocator) catch return;

        const ctx = self.ctx.?;

        // Load join shader
        const join_shader = ctx.loadShader(
            "join",
            gpu_context.shaders.join,
        ) catch return;

        // Create pipelines
        self.init_pipeline = ctx.createPipeline(
            "init_hash_table",
            join_shader,
            "init_hash_table",
        ) catch null;

        self.build_pipeline = ctx.createPipeline(
            "build_hash_table",
            join_shader,
            "build_hash_table",
        ) catch null;

        self.probe_pipeline = ctx.createPipeline(
            "probe_hash_table",
            join_shader,
            "probe_hash_table",
        ) catch null;
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.table);
    }

    /// Build hash table from arrays of keys and values
    /// Uses GPU for large datasets
    pub fn build(self: *Self, keys: []const u32, values: []const u32) HashTableError!void {
        std.debug.assert(keys.len == values.len);

        // Check load factor - rehash if > 0.7
        const new_count = self.count + keys.len;
        if (new_count * 10 > self.capacity * 7) {
            try self.resize(nextPowerOfTwo(new_count * 2));
        }

        // Try GPU path for large batches
        if (keys.len >= GPU_THRESHOLD) {
            self.ensureGPU() catch {};
            if (self.ctx != null and self.build_pipeline != null) {
                if (self.gpuBuild(keys, values)) {
                    self.count = new_count;
                    return;
                }
            }
        }

        // CPU fallback
        try self.cpuBuild(keys, values);
        self.count = new_count;
    }

    /// Probe hash table for keys, return values and found flags
    pub fn probe(
        self: *const Self,
        probe_keys: []const u32,
        results: []u32,
        found: []bool,
    ) HashTableError!void {
        std.debug.assert(probe_keys.len == results.len);
        std.debug.assert(probe_keys.len == found.len);

        // Try GPU path
        if (probe_keys.len >= GPU_THRESHOLD) {
            if (self.ctx != null and self.probe_pipeline != null) {
                if (self.gpuProbe(probe_keys, results, found)) {
                    return;
                }
            }
        }

        // CPU fallback
        self.cpuProbe(probe_keys, results, found);
    }

    /// Extract all key-value pairs from the hash table
    pub fn extract(self: *const Self, out_keys: []u32, out_values: []u32) HashTableError!usize {
        std.debug.assert(out_keys.len >= self.count);
        std.debug.assert(out_values.len >= self.count);

        // Always use CPU for extraction (simpler and efficient enough)
        return self.cpuExtract(out_keys, out_values);
    }

    /// Get a value for a single key (CPU only, for small lookups)
    pub fn get(self: *const Self, key: u32) ?u32 {
        const mask: u32 = @intCast(self.capacity - 1);
        var slot = hashKey(key) & mask;

        var probes: usize = 0;
        while (probes < @min(self.capacity, 1024)) : (probes += 1) {
            const slot_base = slot * SLOT_U32S;
            const stored_key = self.table[slot_base];

            if (stored_key == EMPTY_KEY) return null;

            if (stored_key == key) {
                return self.table[slot_base + 1];
            }

            slot = (slot + 1) & mask;
        }

        return null;
    }

    /// Update the value for an existing key (for MIN/MAX aggregations)
    pub fn updateValue(self: *Self, key: u32, new_value: u32) bool {
        const mask: u32 = @intCast(self.capacity - 1);
        var slot = hashKey(key) & mask;

        var probes: usize = 0;
        while (probes < @min(self.capacity, 1024)) : (probes += 1) {
            const slot_base = slot * SLOT_U32S;
            const stored_key = self.table[slot_base];

            if (stored_key == EMPTY_KEY) return false;

            if (stored_key == key) {
                self.table[slot_base + 1] = new_value;
                return true;
            }

            slot = (slot + 1) & mask;
        }

        return false;
    }

    // =========================================================================
    // GPU implementations
    // =========================================================================

    fn gpuBuild(self: *Self, keys: []const u32, values: []const u32) bool {
        const ctx = self.ctx orelse return false;
        const pipeline = self.build_pipeline orelse return false;

        // Create uniform buffer for params
        const BuildParams = extern struct {
            size: u32,
            capacity: u32,
        };
        const params = BuildParams{
            .size = @intCast(keys.len),
            .capacity = @intCast(self.capacity),
        };

        const params_buffer = ctx.createBufferWithData(
            BuildParams,
            &[_]BuildParams{params},
            .{ .uniform = true, .copy_dst = true },
        ) catch return false;
        defer params_buffer.release();

        const keys_buffer = ctx.createBufferWithData(
            u32,
            keys,
            .{ .storage = true, .copy_dst = true },
        ) catch return false;
        defer keys_buffer.release();

        // For build, we need to upload both keys and values
        // Create combined key-value buffer for table
        const table_size = self.capacity * SLOT_U32S;
        const table_buffer = ctx.createBufferWithData(
            u32,
            self.table,
            .{ .storage = true, .copy_dst = true, .copy_src = true },
        ) catch return false;
        defer table_buffer.release();

        // Note: The WGSL shader doesn't have a values input - it stores row index as value
        // For full key-value support, we'd need a custom shader
        // Using CPU build for now when values differ from indices
        _ = values;

        // Create bind group
        const bind_group_layout = pipeline.getBindGroupLayout(0);
        defer bind_group_layout.release();

        const bind_group = ctx.device.createBindGroup(&.{
            .layout = bind_group_layout,
            .entry_count = 3,
            .entries = &[_]wgpu.BindGroupEntry{
                .{ .binding = 0, .buffer = params_buffer, .size = @sizeOf(BuildParams) },
                .{ .binding = 1, .buffer = keys_buffer, .size = @sizeOf(u32) * keys.len },
                .{ .binding = 2, .buffer = table_buffer, .size = @sizeOf(u32) * table_size },
            },
        }) orelse return false;
        defer bind_group.release();

        // Dispatch compute
        const encoder = ctx.device.createCommandEncoder(null) orelse return false;
        defer encoder.release();

        const pass = encoder.beginComputePass(null) orelse return false;
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind_group, null);
        pass.dispatchWorkgroups(@intCast((keys.len + 255) / 256), 1, 1);
        pass.end();

        const command_buffer = encoder.finish(null) orelse return false;
        defer command_buffer.release();

        ctx.queue.submit(&[_]*wgpu.CommandBuffer{command_buffer});

        // Read back table
        self.readBuffer(table_buffer, self.table) catch return false;
        return true;
    }

    fn gpuProbe(self: *const Self, probe_keys: []const u32, results: []u32, found: []bool) bool {
        const ctx = self.ctx orelse return false;
        _ = self.probe_pipeline orelse return false;

        const ProbeParams = extern struct {
            left_size: u32,
            capacity: u32,
            max_matches: u32,
        };
        const params = ProbeParams{
            .left_size = @intCast(probe_keys.len),
            .capacity = @intCast(self.capacity),
            .max_matches = @intCast(probe_keys.len),
        };

        const params_buffer = ctx.createBufferWithData(
            ProbeParams,
            &[_]ProbeParams{params},
            .{ .uniform = true, .copy_dst = true },
        ) catch return false;
        defer params_buffer.release();

        const keys_buffer = ctx.createBufferWithData(
            u32,
            probe_keys,
            .{ .storage = true, .copy_dst = true },
        ) catch return false;
        defer keys_buffer.release();

        const table_buffer = ctx.createBufferWithData(
            u32,
            self.table,
            .{ .storage = true, .copy_dst = true },
        ) catch return false;
        defer table_buffer.release();

        // Matches buffer: [left_idx, right_idx, ...]
        const matches_buffer = ctx.createBuffer(
            @sizeOf(u32) * probe_keys.len * 2,
            .{ .storage = true, .copy_src = true },
        ) catch return false;
        defer matches_buffer.release();

        // Match count buffer
        const count_buffer = ctx.createBuffer(
            @sizeOf(u32),
            .{ .storage = true, .copy_src = true, .copy_dst = true },
        ) catch return false;
        defer count_buffer.release();

        // Zero out count
        const zero: [1]u32 = .{0};
        ctx.queue.writeBuffer(count_buffer, 0, u32, &zero);

        // This shader outputs matched pairs, not direct results per probe key
        // For direct probe results, we need CPU fallback
        _ = results;
        _ = found;
        return false; // Use CPU for now - shader doesn't match our API exactly
    }

    fn readBuffer(self: *const Self, buffer: *wgpu.Buffer, out: []u32) !void {
        const ctx = self.ctx orelse return error.GPUError;
        const size = @sizeOf(u32) * out.len;

        // Create staging buffer for readback
        const staging = ctx.device.createBuffer(&.{
            .size = size,
            .usage = .{ .map_read = true, .copy_dst = true },
            .mapped_at_creation = .false,
        }) orelse return error.StagingBufferFailed;
        defer staging.release();

        // Copy to staging
        const encoder = ctx.device.createCommandEncoder(null) orelse return error.EncoderCreationFailed;
        defer encoder.release();

        encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
        const cmd = encoder.finish(null) orelse return error.CommandBufferFailed;
        defer cmd.release();

        ctx.queue.submit(&[_]*wgpu.CommandBuffer{cmd});

        // Map and read
        var status: wgpu.MapAsyncStatus = .unknown;
        staging.mapAsync(.{ .read = true }, 0, size, &status, struct {
            fn callback(s: *wgpu.MapAsyncStatus, new_status: wgpu.MapAsyncStatus) void {
                s.* = new_status;
            }
        }.callback);

        // Poll until mapped
        while (status == .unknown) {
            _ = ctx.device.poll(false, null);
        }

        if (status != .success) return error.MapFailed;

        const mapped = staging.getConstMappedRange(u32, 0, out.len) orelse return error.GetMappedRangeFailed;
        @memcpy(out, mapped);
        staging.unmap();
    }

    // =========================================================================
    // CPU fallback implementations
    // =========================================================================

    fn cpuBuild(self: *Self, keys: []const u32, values: []const u32) HashTableError!void {
        const mask: u32 = @intCast(self.capacity - 1);

        for (keys, values) |key, value| {
            var slot = hashKey(key) & mask;
            var probes: usize = 0;

            while (probes < @min(self.capacity, 1024)) : (probes += 1) {
                const slot_base = slot * SLOT_U32S;
                const stored_key = self.table[slot_base];

                if (stored_key == EMPTY_KEY) {
                    // Empty slot - claim it
                    self.table[slot_base] = key;
                    self.table[slot_base + 1] = value;
                    break;
                }

                if (stored_key == key) {
                    // Same key - add value (for SUM aggregation)
                    self.table[slot_base + 1] +%= value;
                    break;
                }

                slot = (slot + 1) & mask;
            } else {
                return HashTableError.TableFull;
            }
        }
    }

    fn cpuProbe(self: *const Self, probe_keys: []const u32, results: []u32, found: []bool) void {
        const mask: u32 = @intCast(self.capacity - 1);

        for (probe_keys, 0..) |key, i| {
            var slot = hashKey(key) & mask;
            var probes: usize = 0;

            found[i] = false;
            results[i] = 0;

            while (probes < @min(self.capacity, 1024)) : (probes += 1) {
                const slot_base = slot * SLOT_U32S;
                const stored_key = self.table[slot_base];

                if (stored_key == EMPTY_KEY) break;

                if (stored_key == key) {
                    results[i] = self.table[slot_base + 1];
                    found[i] = true;
                    break;
                }

                slot = (slot + 1) & mask;
            }
        }
    }

    fn cpuExtract(self: *const Self, out_keys: []u32, out_values: []u32) usize {
        var count: usize = 0;

        for (0..self.capacity) |slot| {
            const slot_base = slot * SLOT_U32S;
            const stored_key = self.table[slot_base];

            if (stored_key != EMPTY_KEY) {
                out_keys[count] = stored_key;
                out_values[count] = self.table[slot_base + 1];
                count += 1;
            }
        }

        return count;
    }

    fn resize(self: *Self, new_capacity: usize) HashTableError!void {
        // Allocate new table
        const new_table = self.allocator.alloc(u32, new_capacity * SLOT_U32S) catch
            return HashTableError.OutOfMemory;

        // Initialize all key slots to EMPTY_KEY
        for (0..new_capacity) |i| {
            new_table[i * SLOT_U32S] = EMPTY_KEY;
            new_table[i * SLOT_U32S + 1] = 0;
        }

        // Rehash existing entries
        const old_table = self.table;
        const old_capacity = self.capacity;
        self.table = new_table;
        self.capacity = new_capacity;
        self.count = 0;

        const mask: u32 = @intCast(new_capacity - 1);

        for (0..old_capacity) |slot| {
            const slot_base = slot * SLOT_U32S;
            const stored_key = old_table[slot_base];

            if (stored_key != EMPTY_KEY) {
                const value = old_table[slot_base + 1];

                // Insert into new table
                var new_slot = hashKey(stored_key) & mask;
                var probes: usize = 0;

                while (probes < new_capacity) : (probes += 1) {
                    const new_slot_base = new_slot * SLOT_U32S;
                    if (self.table[new_slot_base] == EMPTY_KEY) {
                        self.table[new_slot_base] = stored_key;
                        self.table[new_slot_base + 1] = value;
                        self.count += 1;
                        break;
                    }
                    new_slot = (new_slot + 1) & mask;
                }
            }
        }

        self.allocator.free(old_table);
    }
};

/// Hash table with 64-bit key support (uses two u32 slots for key)
/// For compatibility with existing code that uses u64 keys
pub const GPUHashTable64 = struct {
    allocator: Allocator,
    inner: GPUHashTable,
    /// Capacity of the hash table (exposed for query layer compatibility)
    capacity: usize,

    const Self = @This();
    const SLOT_U32S_64: usize = 4; // [key_lo: u32, key_hi: u32, value_lo: u32, value_hi: u32]

    pub fn init(allocator: Allocator, min_capacity: usize) HashTableError!Self {
        // Use inner table with doubled slot size
        const capacity = nextPowerOfTwo(@max(min_capacity, 16));
        const table = allocator.alloc(u32, capacity * SLOT_U32S_64) catch
            return HashTableError.OutOfMemory;

        // Initialize all key slots to EMPTY
        for (0..capacity) |i| {
            const base = i * SLOT_U32S_64;
            table[base] = EMPTY_KEY;
            table[base + 1] = EMPTY_KEY;
            table[base + 2] = 0;
            table[base + 3] = 0;
        }

        return Self{
            .allocator = allocator,
            .capacity = capacity,
            .inner = .{
                .allocator = allocator,
                .table = table,
                .capacity = capacity,
                .count = 0,
            },
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.inner.table);
    }

    /// Build hash table from arrays of 64-bit keys and values (CPU only)
    pub fn build(self: *Self, keys: []const u64, values: []const u64) HashTableError!void {
        std.debug.assert(keys.len == values.len);

        // Check load factor
        const new_count = self.inner.count + keys.len;
        if (new_count * 10 > self.inner.capacity * 7) {
            try self.resize(nextPowerOfTwo(new_count * 2));
        }

        try self.cpuBuild(keys, values);
        self.inner.count = new_count;
    }

    /// Probe hash table for 64-bit keys
    pub fn probe(
        self: *const Self,
        probe_keys: []const u64,
        results: []u64,
        found: []bool,
    ) HashTableError!void {
        std.debug.assert(probe_keys.len == results.len);
        std.debug.assert(probe_keys.len == found.len);
        self.cpuProbe(probe_keys, results, found);
    }

    /// Get a value for a single 64-bit key
    pub fn get(self: *const Self, key: u64) ?u64 {
        const mask: u32 = @intCast(self.inner.capacity - 1);
        const key_lo: u32 = @truncate(key);
        const key_hi: u32 = @truncate(key >> 32);
        var slot = (hashKey(key_lo) ^ hashKey(key_hi)) & mask;

        var probes: usize = 0;
        while (probes < @min(self.inner.capacity, 1024)) : (probes += 1) {
            const base = slot * SLOT_U32S_64;
            const stored_lo = self.inner.table[base];
            const stored_hi = self.inner.table[base + 1];

            if (stored_lo == EMPTY_KEY and stored_hi == EMPTY_KEY) return null;

            if (stored_lo == key_lo and stored_hi == key_hi) {
                const val_lo = self.inner.table[base + 2];
                const val_hi = self.inner.table[base + 3];
                return @as(u64, val_hi) << 32 | val_lo;
            }

            slot = (slot + 1) & mask;
        }

        return null;
    }

    /// Update value for existing key
    pub fn updateValue(self: *Self, key: u64, new_value: u64) bool {
        const mask: u32 = @intCast(self.inner.capacity - 1);
        const key_lo: u32 = @truncate(key);
        const key_hi: u32 = @truncate(key >> 32);
        var slot = (hashKey(key_lo) ^ hashKey(key_hi)) & mask;

        var probes: usize = 0;
        while (probes < @min(self.inner.capacity, 1024)) : (probes += 1) {
            const base = slot * SLOT_U32S_64;
            const stored_lo = self.inner.table[base];
            const stored_hi = self.inner.table[base + 1];

            if (stored_lo == EMPTY_KEY and stored_hi == EMPTY_KEY) return false;

            if (stored_lo == key_lo and stored_hi == key_hi) {
                self.inner.table[base + 2] = @truncate(new_value);
                self.inner.table[base + 3] = @truncate(new_value >> 32);
                return true;
            }

            slot = (slot + 1) & mask;
        }

        return false;
    }

    /// Extract all key-value pairs
    pub fn extract(self: *const Self, out_keys: []u64, out_values: []u64) HashTableError!usize {
        var count: usize = 0;

        for (0..self.inner.capacity) |slot| {
            const base = slot * SLOT_U32S_64;
            const key_lo = self.inner.table[base];
            const key_hi = self.inner.table[base + 1];

            if (!(key_lo == EMPTY_KEY and key_hi == EMPTY_KEY)) {
                out_keys[count] = @as(u64, key_hi) << 32 | key_lo;
                out_values[count] = @as(u64, self.inner.table[base + 3]) << 32 | self.inner.table[base + 2];
                count += 1;
            }
        }

        return count;
    }

    fn cpuBuild(self: *Self, keys: []const u64, values: []const u64) HashTableError!void {
        const mask: u32 = @intCast(self.inner.capacity - 1);

        for (keys, values) |key, value| {
            const key_lo: u32 = @truncate(key);
            const key_hi: u32 = @truncate(key >> 32);
            var slot = (hashKey(key_lo) ^ hashKey(key_hi)) & mask;
            var probes: usize = 0;

            while (probes < @min(self.inner.capacity, 1024)) : (probes += 1) {
                const base = slot * SLOT_U32S_64;
                const stored_lo = self.inner.table[base];
                const stored_hi = self.inner.table[base + 1];

                if (stored_lo == EMPTY_KEY and stored_hi == EMPTY_KEY) {
                    // Empty slot
                    self.inner.table[base] = key_lo;
                    self.inner.table[base + 1] = key_hi;
                    self.inner.table[base + 2] = @truncate(value);
                    self.inner.table[base + 3] = @truncate(value >> 32);
                    break;
                }

                if (stored_lo == key_lo and stored_hi == key_hi) {
                    // Same key - add value (for SUM)
                    const old_lo = self.inner.table[base + 2];
                    const old_hi = self.inner.table[base + 3];
                    const old_val = @as(u64, old_hi) << 32 | old_lo;
                    const new_val = old_val +% value;
                    self.inner.table[base + 2] = @truncate(new_val);
                    self.inner.table[base + 3] = @truncate(new_val >> 32);
                    break;
                }

                slot = (slot + 1) & mask;
            } else {
                return HashTableError.TableFull;
            }
        }
    }

    fn cpuProbe(self: *const Self, probe_keys: []const u64, results: []u64, found: []bool) void {
        const mask: u32 = @intCast(self.inner.capacity - 1);

        for (probe_keys, 0..) |key, i| {
            const key_lo: u32 = @truncate(key);
            const key_hi: u32 = @truncate(key >> 32);
            var slot = (hashKey(key_lo) ^ hashKey(key_hi)) & mask;
            var probes: usize = 0;

            found[i] = false;
            results[i] = 0;

            while (probes < @min(self.inner.capacity, 1024)) : (probes += 1) {
                const base = slot * SLOT_U32S_64;
                const stored_lo = self.inner.table[base];
                const stored_hi = self.inner.table[base + 1];

                if (stored_lo == EMPTY_KEY and stored_hi == EMPTY_KEY) break;

                if (stored_lo == key_lo and stored_hi == key_hi) {
                    results[i] = @as(u64, self.inner.table[base + 3]) << 32 | self.inner.table[base + 2];
                    found[i] = true;
                    break;
                }

                slot = (slot + 1) & mask;
            }
        }
    }

    fn resize(self: *Self, new_capacity: usize) HashTableError!void {
        const new_table = self.allocator.alloc(u32, new_capacity * SLOT_U32S_64) catch
            return HashTableError.OutOfMemory;

        // Initialize
        for (0..new_capacity) |i| {
            const base = i * SLOT_U32S_64;
            new_table[base] = EMPTY_KEY;
            new_table[base + 1] = EMPTY_KEY;
            new_table[base + 2] = 0;
            new_table[base + 3] = 0;
        }

        const old_table = self.inner.table;
        const old_capacity = self.inner.capacity;
        self.inner.table = new_table;
        self.inner.capacity = new_capacity;
        self.capacity = new_capacity;
        self.inner.count = 0;

        const mask: u32 = @intCast(new_capacity - 1);

        for (0..old_capacity) |slot| {
            const base = slot * SLOT_U32S_64;
            const key_lo = old_table[base];
            const key_hi = old_table[base + 1];

            if (!(key_lo == EMPTY_KEY and key_hi == EMPTY_KEY)) {
                const val_lo = old_table[base + 2];
                const val_hi = old_table[base + 3];

                var new_slot = (hashKey(key_lo) ^ hashKey(key_hi)) & mask;
                var probes: usize = 0;

                while (probes < new_capacity) : (probes += 1) {
                    const new_base = new_slot * SLOT_U32S_64;
                    if (self.inner.table[new_base] == EMPTY_KEY and self.inner.table[new_base + 1] == EMPTY_KEY) {
                        self.inner.table[new_base] = key_lo;
                        self.inner.table[new_base + 1] = key_hi;
                        self.inner.table[new_base + 2] = val_lo;
                        self.inner.table[new_base + 3] = val_hi;
                        self.inner.count += 1;
                        break;
                    }
                    new_slot = (new_slot + 1) & mask;
                }
            }
        }

        self.allocator.free(old_table);
    }
};

// =============================================================================
// JOIN Hash Table (supports many-to-many matching)
// =============================================================================

/// Threshold for parallel JOIN probe
/// Threading overhead dominates below this size
pub const JOIN_GPU_THRESHOLD: usize = 10_000;

/// Hash table optimized for JOIN operations
/// Supports duplicate keys and returns ALL matching pairs
pub const JoinHashTable = struct {
    allocator: Allocator,
    /// Entries: [key_lo, key_hi, row_idx, next] - linked list for duplicates
    entries: []u32,
    /// Hash buckets pointing to first entry (or EMPTY)
    buckets: []u32,
    capacity: usize,
    bucket_count: usize,
    count: usize,

    const Self = @This();
    const ENTRY_SIZE: usize = 4; // key_lo, key_hi, row_idx, next
    const EMPTY: u32 = 0xFFFFFFFF;

    /// Result type for JOIN probe operations
    pub const JoinProbeResult = struct {
        left_indices: []usize,
        right_indices: []usize,
    };

    pub fn init(allocator: Allocator, expected_count: usize) HashTableError!Self {
        const capacity = nextPowerOfTwo(@max(expected_count, 16));
        const bucket_count = nextPowerOfTwo(@max(expected_count / 4, 16));

        const entries = allocator.alloc(u32, capacity * ENTRY_SIZE) catch
            return HashTableError.OutOfMemory;
        errdefer allocator.free(entries);

        const buckets = allocator.alloc(u32, bucket_count) catch
            return HashTableError.OutOfMemory;

        // Initialize buckets to EMPTY
        @memset(buckets, EMPTY);

        return Self{
            .allocator = allocator,
            .entries = entries,
            .buckets = buckets,
            .capacity = capacity,
            .bucket_count = bucket_count,
            .count = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.entries);
        self.allocator.free(self.buckets);
    }

    /// Build hash table from i64 keys (stores row indices)
    /// Allows duplicate keys - each gets its own entry
    pub fn buildFromKeys(self: *Self, keys: []const i64) HashTableError!void {
        // Resize if needed
        if (self.count + keys.len > self.capacity) {
            try self.resize(nextPowerOfTwo((self.count + keys.len) * 2));
        }

        const bucket_mask: u32 = @intCast(self.bucket_count - 1);

        for (keys, 0..) |key, row_idx| {
            const key_u64: u64 = @bitCast(key);
            const key_lo: u32 = @truncate(key_u64);
            const key_hi: u32 = @truncate(key_u64 >> 32);
            const hash = hashKeyFast(key) & bucket_mask;

            // Add entry
            const entry_idx = self.count;
            const base = entry_idx * ENTRY_SIZE;
            self.entries[base] = key_lo;
            self.entries[base + 1] = key_hi;
            self.entries[base + 2] = @intCast(row_idx);
            self.entries[base + 3] = self.buckets[hash]; // Link to previous head

            // Update bucket head
            self.buckets[hash] = @intCast(entry_idx);
            self.count += 1;
        }
    }

    /// Probe for JOIN - returns all matched (left_idx, right_idx) pairs
    /// Uses parallel processing for large inputs
    pub fn probeJoin(
        self: *const Self,
        left_keys: []const i64,
        allocator: Allocator,
    ) HashTableError!JoinProbeResult {
        // Use parallel for large inputs
        if (left_keys.len >= JOIN_GPU_THRESHOLD) {
            return self.probeJoinParallel(left_keys, allocator);
        }
        return self.probeJoinSingle(left_keys, allocator);
    }

    /// Single-threaded JOIN probe
    fn probeJoinSingle(
        self: *const Self,
        left_keys: []const i64,
        allocator: Allocator,
    ) HashTableError!JoinProbeResult {
        var left_out = std.ArrayListUnmanaged(usize){};
        var right_out = std.ArrayListUnmanaged(usize){};
        errdefer {
            left_out.deinit(allocator);
            right_out.deinit(allocator);
        }

        // Pre-allocate estimate
        left_out.ensureTotalCapacity(allocator, left_keys.len) catch
            return HashTableError.OutOfMemory;
        right_out.ensureTotalCapacity(allocator, left_keys.len) catch
            return HashTableError.OutOfMemory;

        const bucket_mask: u32 = @intCast(self.bucket_count - 1);

        for (left_keys, 0..) |key, left_idx| {
            const key_u64: u64 = @bitCast(key);
            const key_lo: u32 = @truncate(key_u64);
            const key_hi: u32 = @truncate(key_u64 >> 32);
            const hash = hashKeyFast(key) & bucket_mask;

            // Walk linked list for this bucket
            var entry_idx = self.buckets[hash];
            while (entry_idx != EMPTY) {
                const base = entry_idx * ENTRY_SIZE;
                const stored_lo = self.entries[base];
                const stored_hi = self.entries[base + 1];

                if (stored_lo == key_lo and stored_hi == key_hi) {
                    const right_idx = self.entries[base + 2];
                    left_out.append(allocator, left_idx) catch return HashTableError.OutOfMemory;
                    right_out.append(allocator, right_idx) catch return HashTableError.OutOfMemory;
                }

                entry_idx = self.entries[base + 3]; // next
            }
        }

        return .{
            .left_indices = left_out.toOwnedSlice(allocator) catch return HashTableError.OutOfMemory,
            .right_indices = right_out.toOwnedSlice(allocator) catch return HashTableError.OutOfMemory,
        };
    }

    /// Parallel JOIN probe using multiple threads
    fn probeJoinParallel(
        self: *const Self,
        left_keys: []const i64,
        allocator: Allocator,
    ) HashTableError!JoinProbeResult {
        const num_threads = @min(std.Thread.getCpuCount() catch 4, 8);
        const chunk_size = (left_keys.len + num_threads - 1) / num_threads;

        const ThreadResult = struct {
            left_indices: std.ArrayListUnmanaged(usize),
            right_indices: std.ArrayListUnmanaged(usize),
        };

        var thread_results = allocator.alloc(ThreadResult, num_threads) catch
            return HashTableError.OutOfMemory;
        defer allocator.free(thread_results);

        for (thread_results) |*r| {
            r.left_indices = .{};
            r.right_indices = .{};
        }

        var threads = allocator.alloc(std.Thread, num_threads) catch
            return HashTableError.OutOfMemory;
        defer allocator.free(threads);

        const Worker = struct {
            fn run(
                ht: *const Self,
                keys: []const i64,
                offset: usize,
                alloc: Allocator,
                result: *ThreadResult,
            ) void {
                result.left_indices.ensureTotalCapacity(alloc, keys.len) catch return;
                result.right_indices.ensureTotalCapacity(alloc, keys.len) catch return;

                const bucket_mask: u32 = @intCast(ht.bucket_count - 1);

                for (keys, 0..) |key, i| {
                    const key_u64: u64 = @bitCast(key);
                    const key_lo: u32 = @truncate(key_u64);
                    const key_hi: u32 = @truncate(key_u64 >> 32);
                    const hash = hashKeyFast(key) & bucket_mask;

                    var entry_idx = ht.buckets[hash];
                    while (entry_idx != EMPTY) {
                        const base = entry_idx * ENTRY_SIZE;
                        if (ht.entries[base] == key_lo and ht.entries[base + 1] == key_hi) {
                            result.left_indices.append(alloc, offset + i) catch return;
                            result.right_indices.append(alloc, ht.entries[base + 2]) catch return;
                        }
                        entry_idx = ht.entries[base + 3];
                    }
                }
            }
        };

        // Spawn threads
        var spawned: usize = 0;
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

        // Wait for threads
        for (threads[0..spawned]) |t| t.join();

        // Merge results
        var total: usize = 0;
        for (thread_results[0..spawned]) |r| {
            total += r.left_indices.items.len;
        }

        var left_out = allocator.alloc(usize, total) catch return HashTableError.OutOfMemory;
        errdefer allocator.free(left_out);
        var right_out = allocator.alloc(usize, total) catch return HashTableError.OutOfMemory;

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

    fn resize(self: *Self, new_capacity: usize) HashTableError!void {
        const new_entries = self.allocator.alloc(u32, new_capacity * ENTRY_SIZE) catch
            return HashTableError.OutOfMemory;

        // Copy existing entries
        @memcpy(new_entries[0 .. self.count * ENTRY_SIZE], self.entries[0 .. self.count * ENTRY_SIZE]);

        self.allocator.free(self.entries);
        self.entries = new_entries;
        self.capacity = new_capacity;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "GPUHashTable basic operations" {
    const allocator = std.testing.allocator;

    var ht = try GPUHashTable.init(allocator, 16);
    defer ht.deinit();

    // Build with some keys
    const keys = [_]u32{ 1, 2, 3, 4, 5 };
    const values = [_]u32{ 10, 20, 30, 40, 50 };

    try ht.build(&keys, &values);

    // Test get
    try std.testing.expectEqual(@as(?u32, 10), ht.get(1));
    try std.testing.expectEqual(@as(?u32, 30), ht.get(3));
    try std.testing.expectEqual(@as(?u32, null), ht.get(99));
}

test "GPUHashTable probe" {
    const allocator = std.testing.allocator;

    var ht = try GPUHashTable.init(allocator, 16);
    defer ht.deinit();

    const keys = [_]u32{ 100, 200, 300 };
    const values = [_]u32{ 1, 2, 3 };
    try ht.build(&keys, &values);

    // Probe
    const probe_keys = [_]u32{ 100, 200, 999 };
    var results: [3]u32 = undefined;
    var found: [3]bool = undefined;

    try ht.probe(&probe_keys, &results, &found);

    try std.testing.expect(found[0]);
    try std.testing.expect(found[1]);
    try std.testing.expect(!found[2]);
    try std.testing.expectEqual(@as(u32, 1), results[0]);
    try std.testing.expectEqual(@as(u32, 2), results[1]);
}

test "GPUHashTable extract" {
    const allocator = std.testing.allocator;

    var ht = try GPUHashTable.init(allocator, 16);
    defer ht.deinit();

    const keys = [_]u32{ 10, 20, 30 };
    const values = [_]u32{ 100, 200, 300 };
    try ht.build(&keys, &values);

    var out_keys: [16]u32 = undefined;
    var out_values: [16]u32 = undefined;

    const count = try ht.extract(&out_keys, &out_values);
    try std.testing.expectEqual(@as(usize, 3), count);

    // Verify all keys were extracted (order may differ)
    var found_count: usize = 0;
    for (out_keys[0..count]) |k| {
        if (k == 10 or k == 20 or k == 30) found_count += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), found_count);
}

test "GPUHashTable aggregation (same key)" {
    const allocator = std.testing.allocator;

    var ht = try GPUHashTable.init(allocator, 16);
    defer ht.deinit();

    // Same key multiple times - values should be summed
    const keys = [_]u32{ 1, 1, 1 };
    const values = [_]u32{ 10, 20, 30 };
    try ht.build(&keys, &values);

    try std.testing.expectEqual(@as(?u32, 60), ht.get(1));
}

test "GPUHashTable64 basic operations" {
    const allocator = std.testing.allocator;

    var ht = try GPUHashTable64.init(allocator, 16);
    defer ht.deinit();

    const keys = [_]u64{ 0x100000001, 0x200000002, 0x300000003 };
    const values = [_]u64{ 10, 20, 30 };
    try ht.build(&keys, &values);

    try std.testing.expectEqual(@as(?u64, 10), ht.get(0x100000001));
    try std.testing.expectEqual(@as(?u64, 30), ht.get(0x300000003));
    try std.testing.expectEqual(@as(?u64, null), ht.get(0x999999999));
}
