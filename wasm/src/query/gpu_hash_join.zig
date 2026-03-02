//! GPU-accelerated Hash JOIN operations
//!
//! Uses wgpu-native GPU hash tables for cross-platform parallel join.
//! Falls back to CPU implementation for small datasets.
//!
//! Supports:
//! - INNER JOIN on int64 keys
//! - LEFT OUTER JOIN
//! - Build-probe pattern for efficient joining
//!
//! Usage:
//! ```zig
//! var hash_join = try GPUHashJoin.init(allocator);
//! defer hash_join.deinit();
//!
//! // Build phase: index the smaller table
//! try hash_join.build(build_keys, build_row_ids);
//!
//! // Probe phase: find matches from the larger table
//! const results = try hash_join.innerJoin(probe_keys, probe_row_ids);
//! defer allocator.free(results.build_indices);
//! defer allocator.free(results.probe_indices);
//! ```

const std = @import("std");
const gpu = @import("lanceql.gpu");

/// Threshold for using optimized hash table (uses parallel probe for large data)
pub const JOIN_THRESHOLD: usize = gpu.JOIN_GPU_THRESHOLD;

/// Join result containing matching row indices from both tables
pub const JoinResult = struct {
    /// Row indices from the build table (right side)
    build_indices: []usize,
    /// Row indices from the probe table (left side)
    probe_indices: []usize,
    /// Number of matching pairs
    count: usize,
};

/// Left outer join result with null indicators
pub const LeftJoinResult = struct {
    /// Row indices from the build table (right side), 0 for no match
    build_indices: []usize,
    /// Row indices from the probe table (left side)
    probe_indices: []usize,
    /// true if this probe row had a match
    matched: []bool,
    /// Number of result rows (same as probe table rows for left join)
    count: usize,
};

/// Hash JOIN that supports duplicate keys (many-to-many matching)
/// Uses JoinHashTable with linked lists for duplicate handling
/// Automatically uses parallel probe for large datasets
pub const ManyToManyHashJoin = struct {
    allocator: std.mem.Allocator,
    hash_table: gpu.JoinHashTable,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, expected_build_size: usize) gpu.HashTableError!Self {
        return Self{
            .allocator = allocator,
            .hash_table = try gpu.JoinHashTable.init(allocator, expected_build_size),
        };
    }

    pub fn deinit(self: *Self) void {
        self.hash_table.deinit();
    }

    /// Build phase: Create hash table from build keys (right table)
    /// Supports duplicate keys - each gets its own entry
    pub fn build(self: *Self, keys: []const i64) gpu.HashTableError!void {
        try self.hash_table.buildFromKeys(keys);
    }

    /// Inner join: Return ALL matched (left_idx, right_idx) pairs
    /// Uses parallel probe for large inputs (>= JOIN_THRESHOLD)
    pub fn innerJoin(self: *const Self, probe_keys: []const i64) gpu.HashTableError!JoinResult {
        const result = try self.hash_table.probeJoin(probe_keys, self.allocator);
        return JoinResult{
            .build_indices = result.right_indices,
            .probe_indices = result.left_indices,
            .count = result.left_indices.len,
        };
    }
};

/// Convenience function: Perform hash join with automatic threshold-based optimization
/// Uses ManyToManyHashJoin for any size (handles duplicates correctly)
pub fn hashJoinI64(
    allocator: std.mem.Allocator,
    build_keys: []const i64,
    probe_keys: []const i64,
) gpu.HashTableError!JoinResult {
    var join = try ManyToManyHashJoin.init(allocator, build_keys.len);
    defer join.deinit();

    try join.build(build_keys);
    return join.innerJoin(probe_keys);
}

// =============================================================================
// Tests
// =============================================================================

test "ManyToManyHashJoin inner join basic" {
    const allocator = std.testing.allocator;

    var hash_join = try ManyToManyHashJoin.init(allocator, 10);
    defer hash_join.deinit();

    // Build table: keys 10, 20, 30
    const build_keys = [_]i64{ 10, 20, 30 };
    try hash_join.build(&build_keys);

    // Probe table: keys 20, 30, 40
    const probe_keys = [_]i64{ 20, 30, 40 };
    const result = try hash_join.innerJoin(&probe_keys);
    defer allocator.free(result.build_indices);
    defer allocator.free(result.probe_indices);

    // Should match: (20 -> 1), (30 -> 2)
    try std.testing.expectEqual(@as(usize, 2), result.count);
}

test "ManyToManyHashJoin duplicate keys" {
    const allocator = std.testing.allocator;

    var hash_join = try ManyToManyHashJoin.init(allocator, 10);
    defer hash_join.deinit();

    // Build table with duplicates: key 10 appears twice
    const build_keys = [_]i64{ 10, 10, 20 };
    try hash_join.build(&build_keys);

    // Probe with key 10 - should return TWO matches
    const probe_keys = [_]i64{10};
    const result = try hash_join.innerJoin(&probe_keys);
    defer allocator.free(result.build_indices);
    defer allocator.free(result.probe_indices);

    // Both build rows with key 10 should match
    try std.testing.expectEqual(@as(usize, 2), result.count);
}

test "hashJoinI64 convenience function" {
    const allocator = std.testing.allocator;

    const build_keys = [_]i64{ 1, 2, 3, 4, 5 };
    const probe_keys = [_]i64{ 3, 4, 5, 6, 7 };

    const result = try hashJoinI64(allocator, &build_keys, &probe_keys);
    defer allocator.free(result.build_indices);
    defer allocator.free(result.probe_indices);

    // Should match: 3, 4, 5
    try std.testing.expectEqual(@as(usize, 3), result.count);
}
