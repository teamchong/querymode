//! Shared utilities for table implementations
//!
//! Common helper functions used across Arrow, Avro, ORC, XLSX, and other table types.

const std = @import("std");

/// Generic type conversion for column data arrays.
/// Converts an array of values from one numeric type to another.
pub fn convertArray(
    allocator: std.mem.Allocator,
    comptime From: type,
    comptime To: type,
    values: []const From,
) std.mem.Allocator.Error![]To {
    var result = try allocator.alloc(To, values.len);
    for (values, 0..) |v, i| {
        result[i] = if (To == bool)
            (v != 0)
        else if (@typeInfo(To) == .float)
            @floatCast(v)
        else
            @intCast(v);
    }
    return result;
}

/// Find column index by name in a list of column names.
pub fn findColumnIndex(column_names: []const []const u8, name: []const u8) ?usize {
    for (column_names, 0..) |col_name, i| {
        if (std.mem.eql(u8, col_name, name)) return i;
    }
    return null;
}

// =============================================================================
// Tests
// =============================================================================

test "convertArray int64 to int32" {
    const allocator = std.testing.allocator;
    const input = [_]i64{ 1, 2, 3, 4, 5 };
    const result = try convertArray(allocator, i64, i32, &input);
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 5), result.len);
    try std.testing.expectEqual(@as(i32, 1), result[0]);
    try std.testing.expectEqual(@as(i32, 5), result[4]);
}

test "convertArray int64 to bool" {
    const allocator = std.testing.allocator;
    const input = [_]i64{ 0, 1, 0, 1, 2 };
    const result = try convertArray(allocator, i64, bool, &input);
    defer allocator.free(result);

    try std.testing.expectEqual(false, result[0]);
    try std.testing.expectEqual(true, result[1]);
    try std.testing.expectEqual(false, result[2]);
    try std.testing.expectEqual(true, result[3]);
    try std.testing.expectEqual(true, result[4]);
}

test "convertArray f64 to f32" {
    const allocator = std.testing.allocator;
    const input = [_]f64{ 1.5, 2.5, 3.5 };
    const result = try convertArray(allocator, f64, f32, &input);
    defer allocator.free(result);

    try std.testing.expectEqual(@as(f32, 1.5), result[0]);
    try std.testing.expectEqual(@as(f32, 2.5), result[1]);
}

test "findColumnIndex found" {
    const names = [_][]const u8{ "id", "name", "value" };
    try std.testing.expectEqual(@as(?usize, 0), findColumnIndex(&names, "id"));
    try std.testing.expectEqual(@as(?usize, 1), findColumnIndex(&names, "name"));
    try std.testing.expectEqual(@as(?usize, 2), findColumnIndex(&names, "value"));
}

test "findColumnIndex not found" {
    const names = [_][]const u8{ "id", "name", "value" };
    try std.testing.expectEqual(@as(?usize, null), findColumnIndex(&names, "missing"));
}
