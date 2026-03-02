//! GPU-accelerated GROUP BY operations
//!
//! Uses wgpu-native GPU hash tables for cross-platform parallel aggregation.
//! Falls back to CPU implementation for small datasets.
//!
//! Supports:
//! - GROUP BY single int64 column
//! - SUM, COUNT, MIN, MAX aggregations
//!
//! Usage:
//! ```zig
//! var group_by = try GPUGroupBy.init(allocator, .sum);
//! defer group_by.deinit();
//!
//! try group_by.process(group_keys, values);
//!
//! const results = try group_by.getResults();
//! defer allocator.free(results.keys);
//! defer allocator.free(results.aggregates);
//! ```

const std = @import("std");
const gpu = @import("lanceql.gpu");

/// Aggregation type for GROUP BY
pub const AggType = enum {
    sum,
    count,
    min,
    max,
};

/// Result of GROUP BY operation
pub const GroupByResult = struct {
    keys: []u64,
    aggregates: []u64,
    count: usize,
};

/// GPU-accelerated GROUP BY for int64 keys and values
pub const GPUGroupBy = struct {
    allocator: std.mem.Allocator,
    hash_table: gpu.GPUHashTable64,
    agg_type: AggType,

    const Self = @This();

    /// Initialize GROUP BY with specified aggregation type
    pub fn init(allocator: std.mem.Allocator, agg_type: AggType) gpu.HashTableError!Self {
        return Self{
            .allocator = allocator,
            .hash_table = try gpu.GPUHashTable64.init(allocator, 1024),
            .agg_type = agg_type,
        };
    }

    /// Initialize with specific capacity hint
    pub fn initWithCapacity(allocator: std.mem.Allocator, agg_type: AggType, capacity: usize) gpu.HashTableError!Self {
        return Self{
            .allocator = allocator,
            .hash_table = try gpu.GPUHashTable64.init(allocator, capacity),
            .agg_type = agg_type,
        };
    }

    pub fn deinit(self: *Self) void {
        self.hash_table.deinit();
    }

    /// Process a batch of group keys and values
    /// For SUM/COUNT: values are accumulated
    /// For MIN/MAX: CPU post-processing is needed
    pub fn process(self: *Self, group_keys: []const u64, values: []const u64) gpu.HashTableError!void {
        std.debug.assert(group_keys.len == values.len);

        switch (self.agg_type) {
            .sum, .count => {
                // Direct GPU hash table aggregation (values are summed)
                try self.hash_table.build(group_keys, values);
            },
            .min, .max => {
                // MIN/MAX require special handling
                // For now, use CPU path with hash table for grouping
                try self.processMinMax(group_keys, values);
            },
        }
    }

    /// Process batch for MIN/MAX aggregations
    fn processMinMax(self: *Self, group_keys: []const u64, values: []const u64) gpu.HashTableError!void {
        for (group_keys, values) |key, value| {
            if (self.hash_table.get(key)) |existing| {
                // Update if new value is better
                const update = switch (self.agg_type) {
                    .min => value < existing,
                    .max => value > existing,
                    else => false,
                };
                if (update) {
                    // Update the existing value with the new min/max
                    _ = self.hash_table.updateValue(key, value);
                }
            } else {
                // First occurrence - insert
                const keys = [_]u64{key};
                const vals = [_]u64{value};
                try self.hash_table.build(&keys, &vals);
            }
        }
    }

    /// Get aggregation results
    pub fn getResults(self: *Self) gpu.HashTableError!GroupByResult {
        const capacity = self.hash_table.capacity;

        const keys = self.allocator.alloc(u64, capacity) catch
            return gpu.HashTableError.OutOfMemory;
        errdefer self.allocator.free(keys);

        const aggregates = self.allocator.alloc(u64, capacity) catch {
            self.allocator.free(keys);
            return gpu.HashTableError.OutOfMemory;
        };
        errdefer self.allocator.free(aggregates);

        const count = try self.hash_table.extract(keys, aggregates);

        return GroupByResult{
            .keys = keys,
            .aggregates = aggregates,
            .count = count,
        };
    }

    /// Get result for a specific group key
    pub fn getGroup(self: *const Self, key: u64) ?u64 {
        return self.hash_table.get(key);
    }
};

/// High-level GROUP BY interface for f64 values
/// Converts f64 to u64 bits for GPU processing
pub const GPUGroupByF64 = struct {
    allocator: std.mem.Allocator,
    inner: GPUGroupBy,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, agg_type: AggType) gpu.HashTableError!Self {
        return Self{
            .allocator = allocator,
            .inner = try GPUGroupBy.init(allocator, agg_type),
        };
    }

    pub fn deinit(self: *Self) void {
        self.inner.deinit();
    }

    /// Process batch with f64 values
    pub fn process(self: *Self, group_keys: []const u64, values: []const f64) gpu.HashTableError!void {
        // For SUM, we need to handle floating point differently
        // Convert to fixed-point for GPU processing
        const scale: f64 = 1000000.0; // 6 decimal places

        const scaled_values = self.allocator.alloc(u64, values.len) catch
            return gpu.HashTableError.OutOfMemory;
        defer self.allocator.free(scaled_values);

        for (values, 0..) |v, i| {
            // Convert f64 to scaled integer
            const scaled = @as(i64, @intFromFloat(v * scale));
            scaled_values[i] = @bitCast(scaled);
        }

        try self.inner.process(group_keys, scaled_values);
    }

    /// Get results with f64 aggregates
    pub fn getResults(self: *Self) gpu.HashTableError!struct {
        keys: []u64,
        aggregates: []f64,
        count: usize,
    } {
        const result = try self.inner.getResults();
        defer self.allocator.free(result.aggregates);

        const scale: f64 = 1000000.0;

        const f64_aggregates = self.allocator.alloc(f64, result.count) catch {
            self.allocator.free(result.keys);
            return gpu.HashTableError.OutOfMemory;
        };

        for (result.aggregates[0..result.count], 0..) |agg, i| {
            const scaled: i64 = @bitCast(agg);
            f64_aggregates[i] = @as(f64, @floatFromInt(scaled)) / scale;
        }

        return .{
            .keys = result.keys,
            .aggregates = f64_aggregates,
            .count = result.count,
        };
    }
};

// =============================================================================
// Tests
// =============================================================================

test "GPUGroupBy SUM basic" {
    const allocator = std.testing.allocator;

    var group_by = try GPUGroupBy.init(allocator, .sum);
    defer group_by.deinit();

    // Group keys: [1, 2, 1, 2, 1] -> groups 1 and 2
    // Values:     [10, 20, 30, 40, 50]
    // Expected: group 1 = 10+30+50=90, group 2 = 20+40=60
    const keys = [_]u64{ 1, 2, 1, 2, 1 };
    const values = [_]u64{ 10, 20, 30, 40, 50 };

    try group_by.process(&keys, &values);

    // Check individual groups
    try std.testing.expectEqual(@as(?u64, 90), group_by.getGroup(1));
    try std.testing.expectEqual(@as(?u64, 60), group_by.getGroup(2));
}

test "GPUGroupBy COUNT" {
    const allocator = std.testing.allocator;

    var group_by = try GPUGroupBy.init(allocator, .count);
    defer group_by.deinit();

    // Count occurrences: group 1 appears 3 times, group 2 appears 2 times
    const keys = [_]u64{ 1, 2, 1, 2, 1 };
    const values = [_]u64{ 1, 1, 1, 1, 1 }; // Each row counts as 1

    try group_by.process(&keys, &values);

    try std.testing.expectEqual(@as(?u64, 3), group_by.getGroup(1));
    try std.testing.expectEqual(@as(?u64, 2), group_by.getGroup(2));
}

test "GPUGroupBy getResults" {
    const allocator = std.testing.allocator;

    var group_by = try GPUGroupBy.init(allocator, .sum);
    defer group_by.deinit();

    const keys = [_]u64{ 100, 200, 100 };
    const values = [_]u64{ 5, 10, 15 };

    try group_by.process(&keys, &values);

    const results = try group_by.getResults();
    defer allocator.free(results.keys);
    defer allocator.free(results.aggregates);

    try std.testing.expectEqual(@as(usize, 2), results.count);

    // Find results (order may vary)
    var sum_100: ?u64 = null;
    var sum_200: ?u64 = null;

    for (results.keys[0..results.count], results.aggregates[0..results.count]) |k, v| {
        if (k == 100) sum_100 = v;
        if (k == 200) sum_200 = v;
    }

    try std.testing.expectEqual(@as(?u64, 20), sum_100); // 5 + 15
    try std.testing.expectEqual(@as(?u64, 10), sum_200);
}

test "GPUGroupBy large batch" {
    const allocator = std.testing.allocator;

    // Generate 1000 rows with 10 distinct groups (small to avoid GPU threshold)
    const num_rows: usize = 1000;
    const num_groups: usize = 10;
    const expected_per_group: usize = num_rows / num_groups; // 100

    // Hash table needs ~4x capacity for good load factor with open addressing
    var group_by = try GPUGroupBy.initWithCapacity(allocator, .sum, num_groups * 4);
    defer group_by.deinit();

    var keys = try allocator.alloc(u64, num_rows);
    defer allocator.free(keys);
    var values = try allocator.alloc(u64, num_rows);
    defer allocator.free(values);

    for (0..num_rows) |i| {
        keys[i] = @intCast(i % num_groups);
        values[i] = 1; // Count
    }

    try group_by.process(keys, values);

    // Each group should have count = expected_per_group
    for (0..num_groups) |g| {
        const count = group_by.getGroup(@intCast(g));
        try std.testing.expectEqual(@as(?u64, expected_per_group), count);
    }
}

test "GPUGroupByF64 basic" {
    const allocator = std.testing.allocator;

    var group_by = try GPUGroupByF64.init(allocator, .sum);
    defer group_by.deinit();

    const keys = [_]u64{ 1, 2, 1 };
    const values = [_]f64{ 1.5, 2.5, 3.5 };

    try group_by.process(&keys, &values);

    const results = try group_by.getResults();
    defer allocator.free(results.keys);
    defer allocator.free(results.aggregates);

    try std.testing.expectEqual(@as(usize, 2), results.count);

    // Find results
    for (results.keys[0..results.count], results.aggregates[0..results.count]) |k, v| {
        if (k == 1) {
            try std.testing.expectApproxEqAbs(@as(f64, 5.0), v, 0.001); // 1.5 + 3.5
        }
        if (k == 2) {
            try std.testing.expectApproxEqAbs(@as(f64, 2.5), v, 0.001);
        }
    }
}

test "GPUGroupBy MIN basic" {
    const allocator = std.testing.allocator;

    var group_by = try GPUGroupBy.init(allocator, .min);
    defer group_by.deinit();

    // Group 1: values 50, 10, 30 -> min = 10
    // Group 2: values 20, 40 -> min = 20
    const keys = [_]u64{ 1, 2, 1, 2, 1 };
    const values = [_]u64{ 50, 20, 10, 40, 30 };

    try group_by.process(&keys, &values);

    try std.testing.expectEqual(@as(?u64, 10), group_by.getGroup(1));
    try std.testing.expectEqual(@as(?u64, 20), group_by.getGroup(2));
}

test "GPUGroupBy MAX basic" {
    const allocator = std.testing.allocator;

    var group_by = try GPUGroupBy.init(allocator, .max);
    defer group_by.deinit();

    // Group 1: values 10, 50, 30 -> max = 50
    // Group 2: values 40, 20 -> max = 40
    const keys = [_]u64{ 1, 2, 1, 2, 1 };
    const values = [_]u64{ 10, 40, 50, 20, 30 };

    try group_by.process(&keys, &values);

    try std.testing.expectEqual(@as(?u64, 50), group_by.getGroup(1));
    try std.testing.expectEqual(@as(?u64, 40), group_by.getGroup(2));
}

test "GPUGroupBy MIN multiple batches" {
    const allocator = std.testing.allocator;

    var group_by = try GPUGroupBy.init(allocator, .min);
    defer group_by.deinit();

    // First batch
    const keys1 = [_]u64{ 1, 2 };
    const values1 = [_]u64{ 100, 200 };
    try group_by.process(&keys1, &values1);

    // Second batch with smaller values
    const keys2 = [_]u64{ 1, 2 };
    const values2 = [_]u64{ 50, 150 };
    try group_by.process(&keys2, &values2);

    // Third batch - group 1 gets even smaller
    const keys3 = [_]u64{ 1 };
    const values3 = [_]u64{ 25 };
    try group_by.process(&keys3, &values3);

    try std.testing.expectEqual(@as(?u64, 25), group_by.getGroup(1));
    try std.testing.expectEqual(@as(?u64, 150), group_by.getGroup(2));
}

test "GPUGroupBy MAX getResults" {
    const allocator = std.testing.allocator;

    var group_by = try GPUGroupBy.init(allocator, .max);
    defer group_by.deinit();

    const keys = [_]u64{ 100, 200, 100, 200, 100 };
    const values = [_]u64{ 5, 10, 15, 8, 12 };

    try group_by.process(&keys, &values);

    const results = try group_by.getResults();
    defer allocator.free(results.keys);
    defer allocator.free(results.aggregates);

    try std.testing.expectEqual(@as(usize, 2), results.count);

    // Find results (order may vary)
    var max_100: ?u64 = null;
    var max_200: ?u64 = null;

    for (results.keys[0..results.count], results.aggregates[0..results.count]) |k, v| {
        if (k == 100) max_100 = v;
        if (k == 200) max_200 = v;
    }

    try std.testing.expectEqual(@as(?u64, 15), max_100); // max(5, 15, 12)
    try std.testing.expectEqual(@as(?u64, 10), max_200); // max(10, 8)
}
