//! Window Functions - Types and utilities for SQL window operations
//!
//! Contains WindowFunctionType enum and window function detection.

const std = @import("std");
const ast = @import("ast");
const Expr = ast.Expr;

/// Window function types
pub const WindowFunctionType = enum {
    row_number,
    rank,
    dense_rank,
    lag,
    lead,
};

/// Check if expression is a window function (has OVER clause)
pub fn isWindowFunction(expr: *const Expr) bool {
    return switch (expr.*) {
        .call => |call| call.window != null,
        else => false,
    };
}

/// Check if SELECT list contains any window functions
pub fn hasWindowFunctions(select_list: []const ast.SelectItem) bool {
    for (select_list) |item| {
        if (isWindowFunction(&item.expr)) {
            return true;
        }
    }
    return false;
}

/// Parse window function name to WindowFunctionType
pub fn parseWindowFunctionType(name: []const u8) ?WindowFunctionType {
    if (name.len < 3 or name.len > 16) return null;

    var upper_buf: [16]u8 = undefined;
    const len = @min(name.len, upper_buf.len);
    const upper_name = std.ascii.upperString(upper_buf[0..len], name[0..len]);

    if (std.mem.eql(u8, upper_name, "ROW_NUMBER")) return .row_number;
    if (std.mem.eql(u8, upper_name, "RANK")) return .rank;
    if (std.mem.eql(u8, upper_name, "DENSE_RANK")) return .dense_rank;
    if (std.mem.eql(u8, upper_name, "LAG")) return .lag;
    if (std.mem.eql(u8, upper_name, "LEAD")) return .lead;

    return null;
}

/// Check if function name is a window function
pub fn isWindowFunctionName(name: []const u8) bool {
    return parseWindowFunctionType(name) != null;
}

// ============================================================================
// Tests
// ============================================================================

test "window: parseWindowFunctionType" {
    try std.testing.expectEqual(WindowFunctionType.row_number, parseWindowFunctionType("ROW_NUMBER").?);
    try std.testing.expectEqual(WindowFunctionType.rank, parseWindowFunctionType("RANK").?);
    try std.testing.expectEqual(WindowFunctionType.dense_rank, parseWindowFunctionType("DENSE_RANK").?);
    try std.testing.expectEqual(WindowFunctionType.lag, parseWindowFunctionType("LAG").?);
    try std.testing.expectEqual(WindowFunctionType.lead, parseWindowFunctionType("LEAD").?);

    // Case insensitive
    try std.testing.expectEqual(WindowFunctionType.row_number, parseWindowFunctionType("row_number").?);
    try std.testing.expectEqual(WindowFunctionType.rank, parseWindowFunctionType("Rank").?);

    // Non-window functions return null
    try std.testing.expectEqual(@as(?WindowFunctionType, null), parseWindowFunctionType("SUM"));
    try std.testing.expectEqual(@as(?WindowFunctionType, null), parseWindowFunctionType("COUNT"));
    try std.testing.expectEqual(@as(?WindowFunctionType, null), parseWindowFunctionType("AVG"));
    try std.testing.expectEqual(@as(?WindowFunctionType, null), parseWindowFunctionType("AB")); // Too short
}

test "window: isWindowFunctionName" {
    // Positive cases
    try std.testing.expect(isWindowFunctionName("ROW_NUMBER"));
    try std.testing.expect(isWindowFunctionName("row_number"));
    try std.testing.expect(isWindowFunctionName("RANK"));
    try std.testing.expect(isWindowFunctionName("DENSE_RANK"));
    try std.testing.expect(isWindowFunctionName("LAG"));
    try std.testing.expect(isWindowFunctionName("LEAD"));

    // Negative cases - aggregate functions
    try std.testing.expect(!isWindowFunctionName("SUM"));
    try std.testing.expect(!isWindowFunctionName("COUNT"));
    try std.testing.expect(!isWindowFunctionName("AVG"));
    try std.testing.expect(!isWindowFunctionName("MIN"));
    try std.testing.expect(!isWindowFunctionName("MAX"));

    // Negative cases - scalar functions
    try std.testing.expect(!isWindowFunctionName("UPPER"));
    try std.testing.expect(!isWindowFunctionName("LOWER"));
}
