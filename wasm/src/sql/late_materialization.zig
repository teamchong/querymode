//! Late Materialization for Memory-Efficient Query Execution
//!
//! This module provides late materialization capabilities for SQL queries,
//! enabling efficient querying of large datasets within memory-constrained
//! environments (e.g., 128MB Cloudflare Workers).
//!
//! ## How It Works
//!
//! Traditional query execution loads all columns into memory:
//! ```
//! 1. Load filter columns (full)
//! 2. Load other columns (full)
//! 3. Filter rows
//! 4. Return all data
//! ```
//!
//! Late materialization:
//! ```
//! 1. Load only filter columns
//! 2. Evaluate WHERE, get row_ids
//! 3. Stream: for each batch of row_ids:
//!    a. Fetch only needed bytes via batched Range requests
//!    b. Output batch
//!    c. Free batch memory
//! ```
//!
//! ## Memory Budget (128MB Worker)
//!
//! | Component           | Memory |
//! |---------------------|--------|
//! | WASM heap overhead  | ~5MB   |
//! | row_ids (1M rows)   | 4MB    |
//! | Page index cache    | ~1MB   |
//! | Single batch (1000) | ~1MB   |
//! | Output buffer       | 8KB    |
//! | **Peak usage**      | ~12MB  |
//!
//! ## Usage
//!
//! ```zig
//! var executor = LateMaterializationExecutor.init(allocator, lazy_table);
//! defer executor.deinit();
//!
//! // Execute filter phase - returns row indices only
//! const row_ids = try executor.executeFilterPhase(filter_col_idx, predicate);
//! defer allocator.free(row_ids);
//!
//! // Create streaming result
//! var streaming = StreamingResult.init(row_ids, columns, batch_size, allocator, false);
//! defer streaming.deinit();
//!
//! // Stream batches
//! while (streaming.nextBatchIndices()) |batch_indices| {
//!     // Materialize this batch
//!     const values = try lazy_table.readInt64ColumnAtIndices(col_idx, batch_indices);
//!     defer allocator.free(values);
//!     // Output values...
//! }
//! ```

const std = @import("std");
const table_mod = @import("lanceql.table");
const result_types = @import("result_types.zig");

const LazyTable = table_mod.LazyTable;
const Table = table_mod.Table;
const StreamingResult = result_types.StreamingResult;
const StreamingBatch = result_types.StreamingBatch;
const StreamingStats = result_types.StreamingStats;
const ColumnSpec = result_types.ColumnSpec;
const LanceColumnType = result_types.LanceColumnType;
const Result = result_types.Result;

/// Comparison operators for filter predicates
pub const CompareOp = enum {
    eq, // ==
    ne, // !=
    lt, // <
    le, // <=
    gt, // >
    ge, // >=
};

/// Filter predicate for a column
pub const Predicate = union(enum) {
    /// Compare column value to a constant
    compare_i64: struct {
        op: CompareOp,
        value: i64,
    },
    compare_f64: struct {
        op: CompareOp,
        value: f64,
    },
    /// Check if value is in a set
    in_i64: []const i64,
    /// Range filter
    between_i64: struct {
        min: i64,
        max: i64,
    },
    between_f64: struct {
        min: f64,
        max: f64,
    },
    /// Always true (SELECT * without WHERE)
    all: void,
};

/// Late Materialization Executor
///
/// Provides memory-efficient query execution by:
/// 1. Evaluating filters on minimal data
/// 2. Returning row indices
/// 3. Materializing only requested rows in batches
pub const LateMaterializationExecutor = struct {
    allocator: std.mem.Allocator,
    lazy_table: *LazyTable,
    stats: StreamingStats,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, lazy_table: *LazyTable) Self {
        return Self{
            .allocator = allocator,
            .lazy_table = lazy_table,
            .stats = StreamingStats.init(),
        };
    }

    /// Execute the filter phase: evaluate predicate and return matching row indices.
    ///
    /// This loads only the filter column, evaluates the predicate, and returns
    /// the indices of matching rows. This is Phase 1 of late materialization.
    ///
    /// Memory usage: O(filter_column_size + result_row_ids)
    pub fn executeFilterPhase(
        self: *Self,
        filter_col_idx: u32,
        predicate: Predicate,
    ) ![]u32 {
        // Special case: no filter (SELECT * without WHERE)
        if (predicate == .all) {
            const row_count = try self.lazy_table.rowCount(filter_col_idx);
            const row_ids = try self.allocator.alloc(u32, row_count);
            for (row_ids, 0..) |*id, i| {
                id.* = @intCast(i);
            }
            self.stats.total_rows = row_count;
            return row_ids;
        }

        // Load filter column
        switch (predicate) {
            .compare_i64 => |cmp| {
                const col_data = try self.lazy_table.readInt64Column(filter_col_idx);
                defer self.allocator.free(col_data);

                return self.filterInt64Column(col_data, cmp.op, cmp.value);
            },
            .compare_f64 => |cmp| {
                const col_data = try self.lazy_table.readFloat64Column(filter_col_idx);
                defer self.allocator.free(col_data);

                return self.filterFloat64Column(col_data, cmp.op, cmp.value);
            },
            .in_i64 => |values| {
                const col_data = try self.lazy_table.readInt64Column(filter_col_idx);
                defer self.allocator.free(col_data);

                return self.filterInt64InSet(col_data, values);
            },
            .between_i64 => |range| {
                const col_data = try self.lazy_table.readInt64Column(filter_col_idx);
                defer self.allocator.free(col_data);

                return self.filterInt64Between(col_data, range.min, range.max);
            },
            .between_f64 => |range| {
                const col_data = try self.lazy_table.readFloat64Column(filter_col_idx);
                defer self.allocator.free(col_data);

                return self.filterFloat64Between(col_data, range.min, range.max);
            },
            .all => unreachable, // Handled above
        }
    }

    /// Filter int64 column by comparison
    fn filterInt64Column(self: *Self, data: []const i64, op: CompareOp, value: i64) ![]u32 {
        var matches = std.ArrayList(u32).init(self.allocator);
        errdefer matches.deinit();

        for (data, 0..) |v, i| {
            const match = switch (op) {
                .eq => v == value,
                .ne => v != value,
                .lt => v < value,
                .le => v <= value,
                .gt => v > value,
                .ge => v >= value,
            };
            if (match) {
                try matches.append(@intCast(i));
            }
        }

        self.stats.total_rows = matches.items.len;
        return matches.toOwnedSlice();
    }

    /// Filter float64 column by comparison
    fn filterFloat64Column(self: *Self, data: []const f64, op: CompareOp, value: f64) ![]u32 {
        var matches = std.ArrayList(u32).init(self.allocator);
        errdefer matches.deinit();

        for (data, 0..) |v, i| {
            const match = switch (op) {
                .eq => v == value,
                .ne => v != value,
                .lt => v < value,
                .le => v <= value,
                .gt => v > value,
                .ge => v >= value,
            };
            if (match) {
                try matches.append(@intCast(i));
            }
        }

        self.stats.total_rows = matches.items.len;
        return matches.toOwnedSlice();
    }

    /// Filter int64 column by set membership
    fn filterInt64InSet(self: *Self, data: []const i64, values: []const i64) ![]u32 {
        // Build hash set for O(1) lookup
        var set = std.AutoHashMap(i64, void).init(self.allocator);
        defer set.deinit();

        for (values) |v| {
            try set.put(v, {});
        }

        var matches = std.ArrayList(u32).init(self.allocator);
        errdefer matches.deinit();

        for (data, 0..) |v, i| {
            if (set.contains(v)) {
                try matches.append(@intCast(i));
            }
        }

        self.stats.total_rows = matches.items.len;
        return matches.toOwnedSlice();
    }

    /// Filter int64 column by range
    fn filterInt64Between(self: *Self, data: []const i64, min: i64, max: i64) ![]u32 {
        var matches = std.ArrayList(u32).init(self.allocator);
        errdefer matches.deinit();

        for (data, 0..) |v, i| {
            if (v >= min and v <= max) {
                try matches.append(@intCast(i));
            }
        }

        self.stats.total_rows = matches.items.len;
        return matches.toOwnedSlice();
    }

    /// Filter float64 column by range
    fn filterFloat64Between(self: *Self, data: []const f64, min: f64, max: f64) ![]u32 {
        var matches = std.ArrayList(u32).init(self.allocator);
        errdefer matches.deinit();

        for (data, 0..) |v, i| {
            if (v >= min and v <= max) {
                try matches.append(@intCast(i));
            }
        }

        self.stats.total_rows = matches.items.len;
        return matches.toOwnedSlice();
    }

    /// Materialize a batch of rows for a specific column.
    ///
    /// This is Phase 2 of late materialization - fetch only the bytes needed
    /// for the requested rows using batched Range requests.
    pub fn materializeBatchInt64(
        self: *Self,
        col_idx: u32,
        row_indices: []const u32,
    ) ![]i64 {
        const result = try self.lazy_table.readInt64ColumnAtIndices(col_idx, row_indices);
        self.stats.batches_materialized += 1;
        return result;
    }

    pub fn materializeBatchFloat64(
        self: *Self,
        col_idx: u32,
        row_indices: []const u32,
    ) ![]f64 {
        const result = try self.lazy_table.readFloat64ColumnAtIndices(col_idx, row_indices);
        self.stats.batches_materialized += 1;
        return result;
    }

    pub fn materializeBatchInt32(
        self: *Self,
        col_idx: u32,
        row_indices: []const u32,
    ) ![]i32 {
        const result = try self.lazy_table.readInt32ColumnAtIndices(col_idx, row_indices);
        self.stats.batches_materialized += 1;
        return result;
    }

    pub fn materializeBatchFloat32(
        self: *Self,
        col_idx: u32,
        row_indices: []const u32,
    ) ![]f32 {
        const result = try self.lazy_table.readFloat32ColumnAtIndices(col_idx, row_indices);
        self.stats.batches_materialized += 1;
        return result;
    }

    /// Create a StreamingResult for iteration
    pub fn createStreamingResult(
        self: *Self,
        row_ids: []const u32,
        select_columns: []ColumnSpec,
        batch_size: usize,
    ) StreamingResult {
        return StreamingResult.init(
            row_ids,
            select_columns,
            batch_size,
            self.allocator,
            false, // Caller owns row_ids
        );
    }

    /// Get execution statistics
    pub fn getStats(self: Self) StreamingStats {
        return self.stats;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "LateMaterializationExecutor: predicate types" {
    // Verify predicate union compiles
    const pred1 = Predicate{ .compare_i64 = .{ .op = .gt, .value = 100 } };
    const pred2 = Predicate{ .compare_f64 = .{ .op = .le, .value = 3.14 } };
    const pred3 = Predicate{ .all = {} };

    _ = pred1;
    _ = pred2;
    _ = pred3;
}

test "StreamingResult: basic iteration" {
    const allocator = std.testing.allocator;

    var row_ids = [_]u32{ 0, 5, 10, 15, 20, 25 };
    const columns = try allocator.alloc(ColumnSpec, 0);

    var streaming = StreamingResult.init(&row_ids, columns, 2, allocator, false);
    defer streaming.deinit();

    try std.testing.expectEqual(@as(usize, 6), streaming.totalRows());
    try std.testing.expect(streaming.hasMore());

    // First batch: [0, 5]
    const batch1 = streaming.nextBatchIndices().?;
    try std.testing.expectEqual(@as(usize, 2), batch1.len);
    try std.testing.expectEqual(@as(u32, 0), batch1[0]);
    try std.testing.expectEqual(@as(u32, 5), batch1[1]);

    // Second batch: [10, 15]
    const batch2 = streaming.nextBatchIndices().?;
    try std.testing.expectEqual(@as(usize, 2), batch2.len);

    // Third batch: [20, 25]
    const batch3 = streaming.nextBatchIndices().?;
    try std.testing.expectEqual(@as(usize, 2), batch3.len);

    // No more batches
    try std.testing.expect(!streaming.hasMore());
    try std.testing.expectEqual(@as(?[]const u32, null), streaming.nextBatchIndices());
}

test "StreamingResult: progress tracking" {
    const allocator = std.testing.allocator;

    var row_ids = [_]u32{ 0, 1, 2, 3 };
    const columns = try allocator.alloc(ColumnSpec, 0);

    var streaming = StreamingResult.init(&row_ids, columns, 2, allocator, false);
    defer streaming.deinit();

    try std.testing.expectEqual(@as(f64, 0.0), streaming.progress());

    _ = streaming.nextBatchIndices();
    try std.testing.expectEqual(@as(f64, 0.5), streaming.progress());

    _ = streaming.nextBatchIndices();
    try std.testing.expectEqual(@as(f64, 1.0), streaming.progress());
}

test "StreamingResult: memory calculation" {
    const allocator = std.testing.allocator;

    var row_ids = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }; // 10 rows
    const columns = try allocator.alloc(ColumnSpec, 0);

    var streaming = StreamingResult.init(&row_ids, columns, 3, allocator, false);
    defer streaming.deinit();

    // 10 rows * 4 bytes = 40 bytes
    try std.testing.expectEqual(@as(usize, 40), streaming.rowIdsMemoryBytes());
}
