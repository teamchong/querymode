//! Set Operations - UNION, INTERSECT, EXCEPT implementations
//!
//! This module contains pure functions for SQL set operations.
//! All functions receive explicit parameters (composition pattern).

const std = @import("std");
const result_types = @import("result_types.zig");
const Result = result_types.Result;
pub const result_ops = @import("result_ops.zig");

/// UNION ALL: Concatenate both result sets (keeping duplicates)
pub fn executeUnionAll(allocator: std.mem.Allocator, left: Result, right: Result) !Result {
    const col_count = left.columns.len;
    const total_rows = left.row_count + right.row_count;

    var new_columns = try allocator.alloc(Result.Column, col_count);
    errdefer allocator.free(new_columns);

    for (0..col_count) |i| {
        const left_col = left.columns[i];
        const right_col = right.columns[i];

        new_columns[i] = Result.Column{
            .name = left_col.name,
            .data = try result_ops.concatenateColumnData(allocator, left_col.data, right_col.data, left.row_count, right.row_count),
        };
    }

    return Result{
        .columns = new_columns,
        .row_count = total_rows,
        .allocator = allocator,
    };
}

/// UNION: Concatenate and remove duplicates
pub fn executeUnionDistinct(allocator: std.mem.Allocator, left: Result, right: Result) !Result {
    // First do UNION ALL, then apply DISTINCT
    var union_all = try executeUnionAll(allocator, left, right);
    errdefer union_all.deinit();

    const distinct_result = try result_ops.applyDistinct(allocator, union_all.columns);
    union_all.columns = distinct_result.columns;
    union_all.row_count = distinct_result.row_count;

    return union_all;
}

/// INTERSECT: Keep only rows that appear in both result sets
pub fn executeIntersect(allocator: std.mem.Allocator, left: Result, right: Result) !Result {
    // Build a hash set of rows from right result
    var right_rows = std.StringHashMap(void).init(allocator);
    defer right_rows.deinit();

    for (0..right.row_count) |i| {
        const key = try result_ops.buildRowKey(allocator, right.columns, i);
        defer allocator.free(key);
        try right_rows.put(try allocator.dupe(u8, key), {});
    }
    defer {
        var iter = right_rows.keyIterator();
        while (iter.next()) |key| {
            allocator.free(key.*);
        }
    }

    // Find matching rows in left result
    var matching_indices = std.ArrayList(usize){};
    defer matching_indices.deinit(allocator);

    var seen = std.StringHashMap(void).init(allocator);
    defer {
        var iter = seen.keyIterator();
        while (iter.next()) |key| {
            allocator.free(key.*);
        }
        seen.deinit();
    }

    for (0..left.row_count) |i| {
        const key = try result_ops.buildRowKey(allocator, left.columns, i);
        defer allocator.free(key);

        // Only include if in right AND not already included (for distinct)
        if (right_rows.contains(key) and !seen.contains(key)) {
            try matching_indices.append(allocator, i);
            try seen.put(try allocator.dupe(u8, key), {});
        }
    }

    return try result_ops.projectRows(allocator, left.columns, matching_indices.items);
}

/// EXCEPT: Keep rows from left that don't appear in right
pub fn executeExcept(allocator: std.mem.Allocator, left: Result, right: Result) !Result {
    // Build a hash set of rows from right result
    var right_rows = std.StringHashMap(void).init(allocator);
    defer right_rows.deinit();

    for (0..right.row_count) |i| {
        const key = try result_ops.buildRowKey(allocator, right.columns, i);
        defer allocator.free(key);
        try right_rows.put(try allocator.dupe(u8, key), {});
    }
    defer {
        var iter = right_rows.keyIterator();
        while (iter.next()) |key| {
            allocator.free(key.*);
        }
    }

    // Find rows in left that are NOT in right
    var non_matching_indices = std.ArrayList(usize){};
    defer non_matching_indices.deinit(allocator);

    var seen = std.StringHashMap(void).init(allocator);
    defer {
        var iter = seen.keyIterator();
        while (iter.next()) |key| {
            allocator.free(key.*);
        }
        seen.deinit();
    }

    for (0..left.row_count) |i| {
        const key = try result_ops.buildRowKey(allocator, left.columns, i);
        defer allocator.free(key);

        // Include if NOT in right AND not already included (for distinct)
        if (!right_rows.contains(key) and !seen.contains(key)) {
            try non_matching_indices.append(allocator, i);
            try seen.put(try allocator.dupe(u8, key), {});
        }
    }

    return try result_ops.projectRows(allocator, left.columns, non_matching_indices.items);
}

// ============================================================================
// Tests
// ============================================================================

test "set_ops: executeUnionAll" {
    const allocator = std.testing.allocator;

    // Create left result (2 rows)
    const left_data = try allocator.alloc(i64, 2);
    defer allocator.free(left_data);
    left_data[0] = 1;
    left_data[1] = 2;

    const left_columns = try allocator.alloc(Result.Column, 1);
    defer allocator.free(left_columns);
    left_columns[0] = Result.Column{ .name = "id", .data = .{ .int64 = left_data } };

    const left = Result{
        .columns = left_columns,
        .row_count = 2,
        .allocator = allocator,
        .owns_data = false, // Don't free in deinit since we manage manually
    };

    // Create right result (2 rows)
    const right_data = try allocator.alloc(i64, 2);
    defer allocator.free(right_data);
    right_data[0] = 3;
    right_data[1] = 4;

    const right_columns = try allocator.alloc(Result.Column, 1);
    defer allocator.free(right_columns);
    right_columns[0] = Result.Column{ .name = "id", .data = .{ .int64 = right_data } };

    const right = Result{
        .columns = right_columns,
        .row_count = 2,
        .allocator = allocator,
        .owns_data = false,
    };

    // Execute UNION ALL
    var result = try executeUnionAll(allocator, left, right);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 4), result.row_count);
    try std.testing.expectEqual(@as(usize, 1), result.columns.len);

    const data = result.columns[0].data.int64;
    try std.testing.expectEqual(@as(i64, 1), data[0]);
    try std.testing.expectEqual(@as(i64, 2), data[1]);
    try std.testing.expectEqual(@as(i64, 3), data[2]);
    try std.testing.expectEqual(@as(i64, 4), data[3]);
}
