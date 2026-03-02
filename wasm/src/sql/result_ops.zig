//! Result Operations - Standalone functions for DISTINCT, ORDER BY, LIMIT/OFFSET
//!
//! These operations receive explicit allocator parameters instead of using Executor state,
//! enabling them to be tested and used independently.

const std = @import("std");
const ast = @import("ast");
const result_types = @import("result_types.zig");
const Result = result_types.Result;

// ============================================================================
// DISTINCT Implementation
// ============================================================================

/// Apply DISTINCT - remove duplicate rows from result columns
pub fn applyDistinct(
    allocator: std.mem.Allocator,
    columns: []Result.Column,
) !struct { columns: []Result.Column, row_count: usize } {
    if (columns.len == 0) {
        return .{ .columns = columns, .row_count = 0 };
    }

    const total_rows = columns[0].data.len();
    if (total_rows == 0) {
        return .{ .columns = columns, .row_count = 0 };
    }

    // Track unique row keys using StringHashMap
    var seen = std.StringHashMap(void).init(allocator);
    defer {
        var key_iter = seen.keyIterator();
        while (key_iter.next()) |key| allocator.free(key.*);
        seen.deinit();
    }

    // Track which row indices to keep
    var keep_indices = std.ArrayList(usize){};
    defer keep_indices.deinit(allocator);

    // Build row keys and identify unique rows
    for (0..total_rows) |row_idx| {
        const row_key = try buildRowKey(allocator, columns, row_idx);

        if (!seen.contains(row_key)) {
            try seen.put(row_key, {});
            try keep_indices.append(allocator, row_idx);
        } else {
            allocator.free(row_key);
        }
    }

    // If all rows are unique, return original columns
    if (keep_indices.items.len == total_rows) {
        return .{ .columns = columns, .row_count = total_rows };
    }

    // Build new columns with only unique rows
    const unique_count = keep_indices.items.len;
    const new_columns = try allocator.alloc(Result.Column, columns.len);
    errdefer allocator.free(new_columns);

    for (columns, 0..) |col, col_idx| {
        new_columns[col_idx] = try filterColumnByIndices(allocator, col, keep_indices.items);
    }

    // Free original column data
    for (columns) |col| col.data.free(allocator);
    allocator.free(columns);

    return .{ .columns = new_columns, .row_count = unique_count };
}

/// Build a unique key string for a row across all columns (for DISTINCT/set ops)
pub fn buildRowKey(allocator: std.mem.Allocator, columns: []const Result.Column, row_idx: usize) ![]u8 {
    var key = std.ArrayList(u8){};
    errdefer key.deinit(allocator);

    for (columns) |col| {
        try key.append(allocator, '|');
        switch (col.data) {
            .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |vals| {
                var buf: [64]u8 = undefined;
                const str = std.fmt.bufPrint(&buf, "{d}", .{vals[row_idx]}) catch |err| return err;
                try key.appendSlice(allocator, str);
            },
            .int32, .date32 => |vals| {
                var buf: [32]u8 = undefined;
                const str = std.fmt.bufPrint(&buf, "{d}", .{vals[row_idx]}) catch |err| return err;
                try key.appendSlice(allocator, str);
            },
            .float64 => |vals| {
                var buf: [64]u8 = undefined;
                const str = std.fmt.bufPrint(&buf, "{d}", .{vals[row_idx]}) catch |err| return err;
                try key.appendSlice(allocator, str);
            },
            .float32 => |vals| {
                var buf: [32]u8 = undefined;
                const str = std.fmt.bufPrint(&buf, "{d}", .{vals[row_idx]}) catch |err| return err;
                try key.appendSlice(allocator, str);
            },
            .bool_ => |vals| {
                try key.appendSlice(allocator, if (vals[row_idx]) "true" else "false");
            },
            .string => |vals| {
                try key.appendSlice(allocator, vals[row_idx]);
            },
        }
    }

    return key.toOwnedSlice(allocator);
}

/// Filter a column to keep only specified row indices
pub fn filterColumnByIndices(allocator: std.mem.Allocator, col: Result.Column, indices: []const usize) !Result.Column {
    const count = indices.len;

    return Result.Column{
        .name = col.name,
        .data = switch (col.data) {
            .string => |vals| blk: {
                const new_vals = try allocator.alloc([]const u8, count);
                for (indices, 0..) |idx, i| new_vals[i] = try allocator.dupe(u8, vals[idx]);
                break :blk Result.ColumnData{ .string = new_vals };
            },
            inline else => |vals, tag| blk: {
                const new_vals = try allocator.alloc(@TypeOf(vals[0]), count);
                for (indices, 0..) |idx, i| new_vals[i] = vals[idx];
                break :blk @unionInit(Result.ColumnData, @tagName(tag), new_vals);
            },
        },
    };
}

// ============================================================================
// ORDER BY Implementation
// ============================================================================

/// Context for sort comparison
pub const SortContext = struct {
    column: *const Result.Column,
    direction: ast.OrderDirection,
};

/// Apply ORDER BY - sort result columns by specified columns
pub fn applyOrderBy(
    allocator: std.mem.Allocator,
    columns: []Result.Column,
    order_by: []const ast.OrderBy,
) !void {
    if (columns.len == 0) return;

    const row_count = columns[0].data.len();
    if (row_count == 0) return;

    // Create array of indices [0, 1, 2, ..., n-1]
    const indices = try allocator.alloc(usize, row_count);
    defer allocator.free(indices);

    for (indices, 0..) |*idx, i| idx.* = i;

    // Sort indices based on order_by columns
    for (order_by) |order| {
        const sort_col_idx = findColumnIndex(columns, order.column) orelse continue;
        const sort_col = &columns[sort_col_idx];

        const context = SortContext{
            .column = sort_col,
            .direction = order.direction,
        };

        std.mem.sort(usize, indices, context, sortCompare);
    }

    // Reorder all columns based on sorted indices
    for (columns) |*col| {
        try reorderColumn(allocator, col, indices);
    }
}

/// Comparison function for sorting (static, no state needed)
pub fn sortCompare(context: SortContext, a_idx: usize, b_idx: usize) bool {
    const ascending = context.direction == .asc;

    const cmp = switch (context.column.data) {
        .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| blk: {
            const a, const b = .{ data[a_idx], data[b_idx] };
            if (a < b) break :blk std.math.Order.lt;
            if (a > b) break :blk std.math.Order.gt;
            break :blk std.math.Order.eq;
        },
        .int32, .date32 => |data| blk: {
            const a, const b = .{ data[a_idx], data[b_idx] };
            if (a < b) break :blk std.math.Order.lt;
            if (a > b) break :blk std.math.Order.gt;
            break :blk std.math.Order.eq;
        },
        .float64 => |data| blk: {
            const a, const b = .{ data[a_idx], data[b_idx] };
            if (a < b) break :blk std.math.Order.lt;
            if (a > b) break :blk std.math.Order.gt;
            break :blk std.math.Order.eq;
        },
        .float32 => |data| blk: {
            const a, const b = .{ data[a_idx], data[b_idx] };
            if (a < b) break :blk std.math.Order.lt;
            if (a > b) break :blk std.math.Order.gt;
            break :blk std.math.Order.eq;
        },
        .bool_ => |data| blk: {
            const a: u8 = if (data[a_idx]) 1 else 0;
            const b: u8 = if (data[b_idx]) 1 else 0;
            if (a < b) break :blk std.math.Order.lt;
            if (a > b) break :blk std.math.Order.gt;
            break :blk std.math.Order.eq;
        },
        .string => |data| std.mem.order(u8, data[a_idx], data[b_idx]),
    };

    return if (ascending) cmp == .lt else cmp == .gt;
}

/// Find column index by name
pub fn findColumnIndex(columns: []const Result.Column, name: []const u8) ?usize {
    for (columns, 0..) |col, i| {
        if (std.mem.eql(u8, col.name, name)) return i;
    }
    return null;
}

/// Reorder column data based on index array
fn reorderColumn(allocator: std.mem.Allocator, col: *Result.Column, indices: []const usize) !void {
    switch (col.data) {
        .string => |data| {
            const reordered = try allocator.alloc([]const u8, data.len);
            for (indices, 0..) |idx, i| reordered[i] = try allocator.dupe(u8, data[idx]);
            for (data) |str| allocator.free(str);
            allocator.free(data);
            col.data = Result.ColumnData{ .string = reordered };
        },
        inline else => |data, tag| {
            const reordered = try allocator.alloc(@TypeOf(data[0]), data.len);
            for (indices, 0..) |idx, i| reordered[i] = data[idx];
            allocator.free(data);
            col.data = @unionInit(Result.ColumnData, @tagName(tag), reordered);
        },
    }
}

// ============================================================================
// LIMIT/OFFSET Implementation
// ============================================================================

/// Apply LIMIT and OFFSET to result columns (modifies in place)
/// Returns the new row count
pub fn applyLimitOffset(
    allocator: std.mem.Allocator,
    columns: []Result.Column,
    limit: ?u32,
    offset: ?u32,
) usize {
    if (columns.len == 0) return 0;

    const row_count = columns[0].data.len();
    const start = offset orelse 0;

    if (start >= row_count) {
        // Free all data and return 0
        for (columns) |*col| {
            col.data.free(allocator);
            col.data = emptyColumnData(col.data);
        }
        return 0;
    }

    const end = if (limit) |l| @min(start + l, row_count) else row_count;

    // Slice each column
    for (columns) |*col| {
        sliceColumn(allocator, col, start, end) catch {};
    }

    return end - start;
}

/// Slice column data to [start, end) range
fn sliceColumn(allocator: std.mem.Allocator, col: *Result.Column, start: usize, end: usize) !void {
    const new_len = end - start;

    switch (col.data) {
        .string => |data| {
            const sliced = try allocator.alloc([]const u8, new_len);
            for (data[start..end], 0..) |str, i| sliced[i] = try allocator.dupe(u8, str);
            for (data) |str| allocator.free(str);
            allocator.free(data);
            col.data = Result.ColumnData{ .string = sliced };
        },
        inline else => |data, tag| {
            const sliced = try allocator.alloc(@TypeOf(data[0]), new_len);
            @memcpy(sliced, data[start..end]);
            allocator.free(data);
            col.data = @unionInit(Result.ColumnData, @tagName(tag), sliced);
        },
    }
}

/// Create empty column data of the same type
fn emptyColumnData(data: Result.ColumnData) Result.ColumnData {
    return switch (data) {
        .int64 => .{ .int64 = &[_]i64{} },
        .timestamp_s => .{ .timestamp_s = &[_]i64{} },
        .timestamp_ms => .{ .timestamp_ms = &[_]i64{} },
        .timestamp_us => .{ .timestamp_us = &[_]i64{} },
        .timestamp_ns => .{ .timestamp_ns = &[_]i64{} },
        .date64 => .{ .date64 = &[_]i64{} },
        .int32 => .{ .int32 = &[_]i32{} },
        .date32 => .{ .date32 = &[_]i32{} },
        .float64 => .{ .float64 = &[_]f64{} },
        .float32 => .{ .float32 = &[_]f32{} },
        .bool_ => .{ .bool_ = &[_]bool{} },
        .string => .{ .string = &[_][]const u8{} },
    };
}

// ============================================================================
// Column Data Operations
// ============================================================================

/// Concatenate two column data arrays
pub fn concatenateColumnData(
    allocator: std.mem.Allocator,
    left_data: Result.ColumnData,
    right_data: Result.ColumnData,
    left_len: usize,
    right_len: usize,
) !Result.ColumnData {
    const total_len = left_len + right_len;

    return switch (left_data) {
        .string => |left| blk: {
            const right = right_data.string;
            const new_data = try allocator.alloc([]const u8, total_len);
            errdefer allocator.free(new_data);
            for (0..left_len) |i| new_data[i] = try allocator.dupe(u8, left[i]);
            for (0..right_len) |i| new_data[left_len + i] = try allocator.dupe(u8, right[i]);
            break :blk Result.ColumnData{ .string = new_data };
        },
        inline else => |left, tag| blk: {
            const right = @field(right_data, @tagName(tag));
            const new_data = try allocator.alloc(@TypeOf(left[0]), total_len);
            @memcpy(new_data[0..left_len], left[0..left_len]);
            @memcpy(new_data[left_len..], right[0..right_len]);
            break :blk @unionInit(Result.ColumnData, @tagName(tag), new_data);
        },
    };
}

/// Project specific rows from column data (filter by indices)
pub fn projectColumnData(
    allocator: std.mem.Allocator,
    data: Result.ColumnData,
    indices: []const usize,
) !Result.ColumnData {
    const len = indices.len;

    return switch (data) {
        .string => |d| blk: {
            const new_data = try allocator.alloc([]const u8, len);
            errdefer allocator.free(new_data);
            for (indices, 0..) |src_idx, dst_idx| {
                new_data[dst_idx] = try allocator.dupe(u8, d[src_idx]);
            }
            break :blk Result.ColumnData{ .string = new_data };
        },
        inline else => |d, tag| blk: {
            const new_data = try allocator.alloc(@TypeOf(d[0]), len);
            for (indices, 0..) |src_idx, dst_idx| new_data[dst_idx] = d[src_idx];
            break :blk @unionInit(Result.ColumnData, @tagName(tag), new_data);
        },
    };
}

/// Project specific rows from a result set to create a new result
pub fn projectRows(
    allocator: std.mem.Allocator,
    columns: []const Result.Column,
    indices: []const usize,
) !Result {
    const new_row_count = indices.len;
    const col_count = columns.len;

    var new_columns = try allocator.alloc(Result.Column, col_count);
    errdefer allocator.free(new_columns);

    for (0..col_count) |col_idx| {
        const col = columns[col_idx];
        new_columns[col_idx] = Result.Column{
            .name = col.name,
            .data = try projectColumnData(allocator, col.data, indices),
        };
    }

    return Result{
        .columns = new_columns,
        .row_count = new_row_count,
        .allocator = allocator,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "result_ops: applyLimitOffset basic" {
    const allocator = std.testing.allocator;

    // Create test column data
    const data = try allocator.alloc(i64, 5);
    for (data, 0..) |*v, i| v.* = @intCast(i + 1); // [1, 2, 3, 4, 5]

    var columns = try allocator.alloc(Result.Column, 1);
    columns[0] = .{ .name = "id", .data = .{ .int64 = data } };
    defer {
        columns[0].data.free(allocator);
        allocator.free(columns);
    }

    // Apply LIMIT 3 OFFSET 1 -> [2, 3, 4]
    const new_count = applyLimitOffset(allocator, columns, 3, 1);
    try std.testing.expectEqual(@as(usize, 3), new_count);
    try std.testing.expectEqual(@as(i64, 2), columns[0].data.int64[0]);
    try std.testing.expectEqual(@as(i64, 4), columns[0].data.int64[2]);
}

test "result_ops: findColumnIndex" {
    const cols = [_]Result.Column{
        .{ .name = "a", .data = .{ .int64 = &[_]i64{} } },
        .{ .name = "b", .data = .{ .int64 = &[_]i64{} } },
        .{ .name = "c", .data = .{ .int64 = &[_]i64{} } },
    };

    try std.testing.expectEqual(@as(?usize, 0), findColumnIndex(&cols, "a"));
    try std.testing.expectEqual(@as(?usize, 1), findColumnIndex(&cols, "b"));
    try std.testing.expectEqual(@as(?usize, 2), findColumnIndex(&cols, "c"));
    try std.testing.expectEqual(@as(?usize, null), findColumnIndex(&cols, "d"));
}
