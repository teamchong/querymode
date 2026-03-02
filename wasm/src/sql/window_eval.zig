//! Window Function Evaluation - Standalone functions for window function computation
//!
//! This module contains pure functions for evaluating SQL window functions.
//! All functions receive explicit parameters (composition pattern).
//!
//! IMPORTANT: Columns must be pre-loaded into column_cache before calling these functions.
//! The caller (executor) is responsible for preloading PARTITION BY and ORDER BY columns.

const std = @import("std");
const ast = @import("ast");
const Expr = ast.Expr;
const result_types = @import("result_types.zig");
const Result = result_types.Result;
const CachedColumn = result_types.CachedColumn;
pub const window_functions = @import("window_functions.zig");

const WindowFunctionType = window_functions.WindowFunctionType;

/// Window evaluation context - contains state needed for window computation
pub const WindowContext = struct {
    allocator: std.mem.Allocator,
    column_cache: *std.StringHashMap(CachedColumn),
};

/// Evaluate a window function for a set of indices
/// Returns the result column data (int64 array)
pub fn evaluateWindowFunction(
    ctx: WindowContext,
    call: anytype,
    window_spec: *const ast.WindowSpec,
    indices: []const u32,
) ![]i64 {
    const func_type = window_functions.parseWindowFunctionType(call.name) orelse return error.UnsupportedWindowFunction;

    // Build partition groups
    var partitions = try buildWindowPartitions(ctx, window_spec, indices);
    defer {
        var iter = partitions.valueIterator();
        while (iter.next()) |list| {
            list.deinit(ctx.allocator);
        }
        partitions.deinit();
    }

    // Allocate result array
    const results = try ctx.allocator.alloc(i64, indices.len);
    errdefer ctx.allocator.free(results);

    var partition_iter = partitions.iterator();
    while (partition_iter.next()) |entry| {
        var partition_indices = entry.value_ptr.*;

        // Sort partition by ORDER BY if specified
        if (window_spec.order_by) |order_by| {
            sortWindowPartition(ctx, &partition_indices, order_by);
        }

        // Compute window function for this partition
        switch (func_type) {
            .row_number => {
                // ROW_NUMBER: sequential number within partition
                for (partition_indices.items, 0..) |original_idx, rank| {
                    const result_idx = findIndexPosition(indices, original_idx);
                    if (result_idx) |idx| {
                        results[idx] = @intCast(rank + 1);
                    }
                }
            },
            .rank => {
                // RANK: same rank for ties, skip ranks after ties
                computeRank(ctx, results, partition_indices.items, indices, window_spec.order_by, false);
            },
            .dense_rank => {
                // DENSE_RANK: same rank for ties, no gaps
                computeRank(ctx, results, partition_indices.items, indices, window_spec.order_by, true);
            },
            .lag => {
                // LAG: value from N rows before
                const offset: usize = if (call.args.len > 1) blk: {
                    const arg = call.args[1];
                    if (arg == .value and arg.value == .integer) {
                        break :blk @intCast(arg.value.integer);
                    }
                    break :blk 1;
                } else 1;

                const default_val: i64 = if (call.args.len > 2) blk: {
                    const arg = call.args[2];
                    if (arg == .value and arg.value == .integer) {
                        break :blk arg.value.integer;
                    }
                    break :blk 0;
                } else 0;

                computeLagLead(ctx, results, partition_indices.items, indices, call.args, offset, default_val, true);
            },
            .lead => {
                // LEAD: value from N rows after
                const offset: usize = if (call.args.len > 1) blk: {
                    const arg = call.args[1];
                    if (arg == .value and arg.value == .integer) {
                        break :blk @intCast(arg.value.integer);
                    }
                    break :blk 1;
                } else 1;

                const default_val: i64 = if (call.args.len > 2) blk: {
                    const arg = call.args[2];
                    if (arg == .value and arg.value == .integer) {
                        break :blk arg.value.integer;
                    }
                    break :blk 0;
                } else 0;

                computeLagLead(ctx, results, partition_indices.items, indices, call.args, offset, default_val, false);
            },
        }
    }

    return results;
}

/// Build partition groups based on PARTITION BY columns
/// IMPORTANT: Caller must have preloaded partition columns into column_cache
fn buildWindowPartitions(
    ctx: WindowContext,
    window_spec: *const ast.WindowSpec,
    indices: []const u32,
) !std.StringHashMap(std.ArrayListUnmanaged(u32)) {
    var partitions = std.StringHashMap(std.ArrayListUnmanaged(u32)).init(ctx.allocator);
    errdefer {
        var iter = partitions.valueIterator();
        while (iter.next()) |list| {
            list.deinit(ctx.allocator);
        }
        partitions.deinit();
    }

    if (window_spec.partition_by) |partition_cols| {
        // Group rows by partition key
        for (indices) |row_idx| {
            const key = try buildPartitionKey(ctx, partition_cols, row_idx);
            const result = try partitions.getOrPut(key);
            if (!result.found_existing) {
                result.value_ptr.* = .{};
            }
            try result.value_ptr.append(ctx.allocator, row_idx);
        }
    } else {
        // No PARTITION BY - all rows in one partition
        var single_partition: std.ArrayListUnmanaged(u32) = .{};
        for (indices) |row_idx| {
            try single_partition.append(ctx.allocator, row_idx);
        }
        try partitions.put("", single_partition);
    }

    return partitions;
}

/// Build partition key from row values
fn buildPartitionKey(ctx: WindowContext, partition_cols: [][]const u8, row_idx: u32) ![]const u8 {
    var key_parts = std.ArrayList(u8){};
    errdefer key_parts.deinit(ctx.allocator);

    for (partition_cols, 0..) |col_name, i| {
        if (i > 0) try key_parts.append(ctx.allocator, '|');

        const col = ctx.column_cache.get(col_name) orelse return error.ColumnNotFound;
        const value_str = try columnValueToString(ctx.allocator, col, row_idx);
        try key_parts.appendSlice(ctx.allocator, value_str);
    }

    return key_parts.toOwnedSlice(ctx.allocator);
}

/// Convert column value at index to string for key building
pub fn columnValueToString(allocator: std.mem.Allocator, col: CachedColumn, row_idx: u32) ![]const u8 {
    return switch (col) {
        .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| blk: {
            const val = data[row_idx];
            const buf = try allocator.alloc(u8, 32);
            const written = std.fmt.bufPrint(buf, "{d}", .{val}) catch "";
            break :blk written;
        },
        .int32, .date32 => |data| blk: {
            const val = data[row_idx];
            const buf = try allocator.alloc(u8, 16);
            const written = std.fmt.bufPrint(buf, "{d}", .{val}) catch "";
            break :blk written;
        },
        .float64 => |data| blk: {
            const val = data[row_idx];
            const buf = try allocator.alloc(u8, 32);
            const written = std.fmt.bufPrint(buf, "{d}", .{val}) catch "";
            break :blk written;
        },
        .float32 => |data| blk: {
            const val = data[row_idx];
            const buf = try allocator.alloc(u8, 32);
            const written = std.fmt.bufPrint(buf, "{d}", .{val}) catch "";
            break :blk written;
        },
        .bool_ => |data| if (data[row_idx]) "true" else "false",
        .string => |data| data[row_idx],
    };
}

/// Sort partition indices by ORDER BY columns
/// IMPORTANT: Caller must have preloaded ORDER BY columns into column_cache
fn sortWindowPartition(
    ctx: WindowContext,
    partition: *std.ArrayListUnmanaged(u32),
    order_by: []const ast.OrderBy,
) void {
    if (order_by.len == 0) return;

    const first_ob = order_by[0];
    const col_name = first_ob.column;

    const cached_col = ctx.column_cache.get(col_name) orelse return;
    const ascending = first_ob.direction == .asc;

    // Sort partition indices based on column values
    const WindowSortCtx = struct {
        col: CachedColumn,
        asc: bool,

        fn lessThan(c: @This(), a: u32, b: u32) bool {
            const result = switch (c.col) {
                .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| data[a] < data[b],
                .int32, .date32 => |data| data[a] < data[b],
                .float64 => |data| data[a] < data[b],
                .float32 => |data| data[a] < data[b],
                .string => |data| std.mem.lessThan(u8, data[a], data[b]),
                .bool_ => |data| @intFromBool(data[a]) < @intFromBool(data[b]),
            };
            return if (c.asc) result else !result;
        }
    };

    const sort_ctx = WindowSortCtx{ .col = cached_col, .asc = ascending };
    std.mem.sort(u32, partition.items, sort_ctx, WindowSortCtx.lessThan);
}

/// Find position of original row index in indices array
pub fn findIndexPosition(indices: []const u32, original_idx: u32) ?usize {
    for (indices, 0..) |idx, pos| {
        if (idx == original_idx) return pos;
    }
    return null;
}

/// Compute RANK or DENSE_RANK for partition
fn computeRank(
    ctx: WindowContext,
    results: []i64,
    partition: []const u32,
    indices: []const u32,
    order_by: ?[]const ast.OrderBy,
    dense: bool,
) void {
    if (partition.len == 0) return;

    var current_rank: i64 = 1;
    var prev_value: ?i64 = null;
    var rows_at_rank: i64 = 0;

    // Get ORDER BY column for comparison
    const order_col: ?CachedColumn = if (order_by) |ob| blk: {
        if (ob.len == 0) break :blk null;
        const col_name = ob[0].column;
        break :blk ctx.column_cache.get(col_name);
    } else null;

    for (partition, 0..) |original_idx, i| {
        const result_idx = findIndexPosition(indices, original_idx) orelse continue;

        if (order_col) |col| {
            const current_value: i64 = switch (col) {
                .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| data[original_idx],
                .int32, .date32 => |data| data[original_idx],
                .float64 => |data| @intFromFloat(data[original_idx]),
                .float32 => |data| @intFromFloat(data[original_idx]),
                .string => |data| @intCast(std.hash.Wyhash.hash(0, data[original_idx])),
                .bool_ => |data| if (data[original_idx]) 1 else 0,
            };

            if (prev_value) |pv| {
                if (current_value != pv) {
                    // Value changed - update rank
                    if (dense) {
                        current_rank += 1;
                    } else {
                        current_rank += rows_at_rank;
                    }
                    rows_at_rank = 1;
                } else {
                    rows_at_rank += 1;
                }
            } else {
                rows_at_rank = 1;
            }
            prev_value = current_value;
        } else {
            // No ORDER BY - all get rank 1
            _ = i;
        }

        results[result_idx] = current_rank;
    }
}

/// Compute LAG or LEAD for partition
fn computeLagLead(
    ctx: WindowContext,
    results: []i64,
    partition: []const u32,
    indices: []const u32,
    args: []const Expr,
    offset: usize,
    default_val: i64,
    is_lag: bool,
) void {
    if (args.len == 0) return;

    // Get the column to look up
    const col_name = switch (args[0]) {
        .column => |col| col.name,
        else => return,
    };

    const cached_col = ctx.column_cache.get(col_name) orelse return;

    for (partition, 0..) |original_idx, i| {
        const result_idx = findIndexPosition(indices, original_idx) orelse continue;

        // Calculate source index
        const source_partition_idx: ?usize = if (is_lag) blk: {
            if (i < offset) break :blk null;
            break :blk i - offset;
        } else blk: {
            if (i + offset >= partition.len) break :blk null;
            break :blk i + offset;
        };

        if (source_partition_idx) |src_idx| {
            const src_original_idx = partition[src_idx];
            results[result_idx] = switch (cached_col) {
                .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| data[src_original_idx],
                .int32, .date32 => |data| data[src_original_idx],
                .float64 => |data| @intFromFloat(data[src_original_idx]),
                .float32 => |data| @intFromFloat(data[src_original_idx]),
                .string => 0, // LAG/LEAD on strings returns 0 for now
                .bool_ => |data| if (data[src_original_idx]) 1 else 0,
            };
        } else {
            results[result_idx] = default_val;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "window: findIndexPosition" {
    const indices = [_]u32{ 5, 10, 15, 20 };

    try std.testing.expectEqual(@as(?usize, 0), findIndexPosition(&indices, 5));
    try std.testing.expectEqual(@as(?usize, 2), findIndexPosition(&indices, 15));
    try std.testing.expectEqual(@as(?usize, null), findIndexPosition(&indices, 7));
}
