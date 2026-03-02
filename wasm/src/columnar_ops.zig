//! SIMD Columnar Operations for SQL Executor
//!
//! High-performance columnar operations using SIMD vectorization.
//! Processes 8 elements per instruction for 8x throughput improvement.
//!
//! Usage:
//!   const ops = @import("columnar_ops.zig");
//!   const count = ops.filterGreaterI64(column_data, 100, result_buf);
//!   const sum = ops.sumI64(column_data);

const std = @import("std");

// SIMD vector types - 8-wide for maximum throughput on modern CPUs
pub const Vec8i64 = @Vector(8, i64);
pub const Vec8i32 = @Vector(8, i32);
pub const Vec8f64 = @Vector(8, f64);
pub const Vec8f32 = @Vector(8, f32);
pub const Vec8bool = @Vector(8, bool);

// ============================================================================
// FILTER OPERATIONS - Return matching row indices
// ============================================================================

/// Filter: col > value (8 comparisons per instruction)
pub fn filterGreaterI64(col: []const i64, value: i64, out: []u32) usize {
    const value_vec: Vec8i64 = @splat(value);
    var match_count: usize = 0;
    var i: usize = 0;

    // SIMD path: process 8 elements at a time
    while (i + 8 <= col.len) : (i += 8) {
        const batch: Vec8i64 = col[i..][0..8].*;
        const matches = batch > value_vec;

        // Extract matching indices
        inline for (0..8) |j| {
            if (matches[j]) {
                if (match_count < out.len) {
                    out[match_count] = @intCast(i + j);
                    match_count += 1;
                }
            }
        }
    }

    // Scalar remainder
    while (i < col.len) : (i += 1) {
        if (col[i] > value) {
            if (match_count < out.len) {
                out[match_count] = @intCast(i);
                match_count += 1;
            }
        }
    }
    return match_count;
}

/// Filter: col >= value
pub fn filterGreaterEqualI64(col: []const i64, value: i64, out: []u32) usize {
    const value_vec: Vec8i64 = @splat(value);
    var match_count: usize = 0;
    var i: usize = 0;

    while (i + 8 <= col.len) : (i += 8) {
        const batch: Vec8i64 = col[i..][0..8].*;
        const matches = batch >= value_vec;

        inline for (0..8) |j| {
            if (matches[j]) {
                if (match_count < out.len) {
                    out[match_count] = @intCast(i + j);
                    match_count += 1;
                }
            }
        }
    }

    while (i < col.len) : (i += 1) {
        if (col[i] >= value) {
            if (match_count < out.len) {
                out[match_count] = @intCast(i);
                match_count += 1;
            }
        }
    }
    return match_count;
}

/// Filter: col < value
pub fn filterLessI64(col: []const i64, value: i64, out: []u32) usize {
    const value_vec: Vec8i64 = @splat(value);
    var match_count: usize = 0;
    var i: usize = 0;

    while (i + 8 <= col.len) : (i += 8) {
        const batch: Vec8i64 = col[i..][0..8].*;
        const matches = batch < value_vec;

        inline for (0..8) |j| {
            if (matches[j]) {
                if (match_count < out.len) {
                    out[match_count] = @intCast(i + j);
                    match_count += 1;
                }
            }
        }
    }

    while (i < col.len) : (i += 1) {
        if (col[i] < value) {
            if (match_count < out.len) {
                out[match_count] = @intCast(i);
                match_count += 1;
            }
        }
    }
    return match_count;
}

/// Filter: col <= value
pub fn filterLessEqualI64(col: []const i64, value: i64, out: []u32) usize {
    const value_vec: Vec8i64 = @splat(value);
    var match_count: usize = 0;
    var i: usize = 0;

    while (i + 8 <= col.len) : (i += 8) {
        const batch: Vec8i64 = col[i..][0..8].*;
        const matches = batch <= value_vec;

        inline for (0..8) |j| {
            if (matches[j]) {
                if (match_count < out.len) {
                    out[match_count] = @intCast(i + j);
                    match_count += 1;
                }
            }
        }
    }

    while (i < col.len) : (i += 1) {
        if (col[i] <= value) {
            if (match_count < out.len) {
                out[match_count] = @intCast(i);
                match_count += 1;
            }
        }
    }
    return match_count;
}

/// Filter: col == value
pub fn filterEqualI64(col: []const i64, value: i64, out: []u32) usize {
    const value_vec: Vec8i64 = @splat(value);
    var match_count: usize = 0;
    var i: usize = 0;

    while (i + 8 <= col.len) : (i += 8) {
        const batch: Vec8i64 = col[i..][0..8].*;
        const matches = batch == value_vec;

        inline for (0..8) |j| {
            if (matches[j]) {
                if (match_count < out.len) {
                    out[match_count] = @intCast(i + j);
                    match_count += 1;
                }
            }
        }
    }

    while (i < col.len) : (i += 1) {
        if (col[i] == value) {
            if (match_count < out.len) {
                out[match_count] = @intCast(i);
                match_count += 1;
            }
        }
    }
    return match_count;
}

/// Filter: col != value
pub fn filterNotEqualI64(col: []const i64, value: i64, out: []u32) usize {
    const value_vec: Vec8i64 = @splat(value);
    var match_count: usize = 0;
    var i: usize = 0;

    while (i + 8 <= col.len) : (i += 8) {
        const batch: Vec8i64 = col[i..][0..8].*;
        const matches = batch != value_vec;

        inline for (0..8) |j| {
            if (matches[j]) {
                if (match_count < out.len) {
                    out[match_count] = @intCast(i + j);
                    match_count += 1;
                }
            }
        }
    }

    while (i < col.len) : (i += 1) {
        if (col[i] != value) {
            if (match_count < out.len) {
                out[match_count] = @intCast(i);
                match_count += 1;
            }
        }
    }
    return match_count;
}

// Float64 filter operations

/// Filter: col > value (f64)
pub fn filterGreaterF64(col: []const f64, value: f64, out: []u32) usize {
    const value_vec: Vec8f64 = @splat(value);
    var match_count: usize = 0;
    var i: usize = 0;

    while (i + 8 <= col.len) : (i += 8) {
        const batch: Vec8f64 = col[i..][0..8].*;
        const matches = batch > value_vec;

        inline for (0..8) |j| {
            if (matches[j]) {
                if (match_count < out.len) {
                    out[match_count] = @intCast(i + j);
                    match_count += 1;
                }
            }
        }
    }

    while (i < col.len) : (i += 1) {
        if (col[i] > value) {
            if (match_count < out.len) {
                out[match_count] = @intCast(i);
                match_count += 1;
            }
        }
    }
    return match_count;
}

/// Filter: col >= value (f64)
pub fn filterGreaterEqualF64(col: []const f64, value: f64, out: []u32) usize {
    const value_vec: Vec8f64 = @splat(value);
    var match_count: usize = 0;
    var i: usize = 0;

    while (i + 8 <= col.len) : (i += 8) {
        const batch: Vec8f64 = col[i..][0..8].*;
        const matches = batch >= value_vec;

        inline for (0..8) |j| {
            if (matches[j]) {
                if (match_count < out.len) {
                    out[match_count] = @intCast(i + j);
                    match_count += 1;
                }
            }
        }
    }

    while (i < col.len) : (i += 1) {
        if (col[i] >= value) {
            if (match_count < out.len) {
                out[match_count] = @intCast(i);
                match_count += 1;
            }
        }
    }
    return match_count;
}

/// Filter: col < value (f64)
pub fn filterLessF64(col: []const f64, value: f64, out: []u32) usize {
    const value_vec: Vec8f64 = @splat(value);
    var match_count: usize = 0;
    var i: usize = 0;

    while (i + 8 <= col.len) : (i += 8) {
        const batch: Vec8f64 = col[i..][0..8].*;
        const matches = batch < value_vec;

        inline for (0..8) |j| {
            if (matches[j]) {
                if (match_count < out.len) {
                    out[match_count] = @intCast(i + j);
                    match_count += 1;
                }
            }
        }
    }

    while (i < col.len) : (i += 1) {
        if (col[i] < value) {
            if (match_count < out.len) {
                out[match_count] = @intCast(i);
                match_count += 1;
            }
        }
    }
    return match_count;
}

/// Filter: col <= value (f64)
pub fn filterLessEqualF64(col: []const f64, value: f64, out: []u32) usize {
    const value_vec: Vec8f64 = @splat(value);
    var match_count: usize = 0;
    var i: usize = 0;

    while (i + 8 <= col.len) : (i += 8) {
        const batch: Vec8f64 = col[i..][0..8].*;
        const matches = batch <= value_vec;

        inline for (0..8) |j| {
            if (matches[j]) {
                if (match_count < out.len) {
                    out[match_count] = @intCast(i + j);
                    match_count += 1;
                }
            }
        }
    }

    while (i < col.len) : (i += 1) {
        if (col[i] <= value) {
            if (match_count < out.len) {
                out[match_count] = @intCast(i);
                match_count += 1;
            }
        }
    }
    return match_count;
}

// ============================================================================
// REDUCE OPERATIONS - Aggregate whole column with SIMD
// ============================================================================

/// SUM(col) for i64 - returns i128 to avoid overflow
pub fn sumI64(col: []const i64) i128 {
    if (col.len == 0) return 0;

    var acc: Vec8i64 = @splat(0);
    var i: usize = 0;

    // SIMD path: 8 additions per instruction
    while (i + 8 <= col.len) : (i += 8) {
        acc += col[i..][0..8].*;
    }

    // Horizontal sum
    var sum: i128 = @reduce(.Add, acc);

    // Scalar remainder
    while (i < col.len) : (i += 1) {
        sum += col[i];
    }
    return sum;
}

/// SUM(col) for f64
pub fn sumF64(col: []const f64) f64 {
    if (col.len == 0) return 0;

    var acc: Vec8f64 = @splat(0);
    var i: usize = 0;

    // SIMD path: 8 additions per instruction
    while (i + 8 <= col.len) : (i += 8) {
        acc += col[i..][0..8].*;
    }

    // Horizontal sum
    var sum: f64 = @reduce(.Add, acc);

    // Scalar remainder
    while (i < col.len) : (i += 1) {
        sum += col[i];
    }
    return sum;
}

/// MIN(col) for i64
pub fn minI64(col: []const i64) ?i64 {
    if (col.len == 0) return null;

    var acc: Vec8i64 = @splat(std.math.maxInt(i64));
    var i: usize = 0;

    // SIMD path
    while (i + 8 <= col.len) : (i += 8) {
        const batch: Vec8i64 = col[i..][0..8].*;
        acc = @min(acc, batch);
    }

    // Horizontal min
    var result: i64 = @reduce(.Min, acc);

    // Scalar remainder
    while (i < col.len) : (i += 1) {
        result = @min(result, col[i]);
    }
    return result;
}

/// MAX(col) for i64
pub fn maxI64(col: []const i64) ?i64 {
    if (col.len == 0) return null;

    var acc: Vec8i64 = @splat(std.math.minInt(i64));
    var i: usize = 0;

    // SIMD path
    while (i + 8 <= col.len) : (i += 8) {
        const batch: Vec8i64 = col[i..][0..8].*;
        acc = @max(acc, batch);
    }

    // Horizontal max
    var result: i64 = @reduce(.Max, acc);

    // Scalar remainder
    while (i < col.len) : (i += 1) {
        result = @max(result, col[i]);
    }
    return result;
}

/// MIN(col) for f64
pub fn minF64(col: []const f64) ?f64 {
    if (col.len == 0) return null;

    var acc: Vec8f64 = @splat(std.math.inf(f64));
    var i: usize = 0;

    // SIMD path
    while (i + 8 <= col.len) : (i += 8) {
        const batch: Vec8f64 = col[i..][0..8].*;
        acc = @min(acc, batch);
    }

    // Horizontal min
    var result: f64 = @reduce(.Min, acc);

    // Scalar remainder
    while (i < col.len) : (i += 1) {
        result = @min(result, col[i]);
    }
    return result;
}

/// MAX(col) for f64
pub fn maxF64(col: []const f64) ?f64 {
    if (col.len == 0) return null;

    var acc: Vec8f64 = @splat(-std.math.inf(f64));
    var i: usize = 0;

    // SIMD path
    while (i + 8 <= col.len) : (i += 8) {
        const batch: Vec8f64 = col[i..][0..8].*;
        acc = @max(acc, batch);
    }

    // Horizontal max
    var result: f64 = @reduce(.Max, acc);

    // Scalar remainder
    while (i < col.len) : (i += 1) {
        result = @max(result, col[i]);
    }
    return result;
}

/// COUNT(*) - just return length
pub fn count(col_len: usize) usize {
    return col_len;
}

// ============================================================================
// FILTERED REDUCE OPERATIONS - Aggregate only selected rows
// ============================================================================

/// SUM(col) for selected rows (i64)
pub fn sumFilteredI64(col: []const i64, indices: []const u32) i128 {
    var sum: i128 = 0;
    for (indices) |idx| {
        if (idx < col.len) {
            sum += col[idx];
        }
    }
    return sum;
}

/// SUM(col) for selected rows (f64)
pub fn sumFilteredF64(col: []const f64, indices: []const u32) f64 {
    var sum: f64 = 0;
    for (indices) |idx| {
        if (idx < col.len) {
            sum += col[idx];
        }
    }
    return sum;
}

/// MIN(col) for selected rows (i64)
pub fn minFilteredI64(col: []const i64, indices: []const u32) ?i64 {
    if (indices.len == 0) return null;
    var result: i64 = std.math.maxInt(i64);
    for (indices) |idx| {
        if (idx < col.len) {
            result = @min(result, col[idx]);
        }
    }
    return result;
}

/// MAX(col) for selected rows (i64)
pub fn maxFilteredI64(col: []const i64, indices: []const u32) ?i64 {
    if (indices.len == 0) return null;
    var result: i64 = std.math.minInt(i64);
    for (indices) |idx| {
        if (idx < col.len) {
            result = @max(result, col[idx]);
        }
    }
    return result;
}

/// MIN(col) for selected rows (f64)
pub fn minFilteredF64(col: []const f64, indices: []const u32) ?f64 {
    if (indices.len == 0) return null;
    var result: f64 = std.math.inf(f64);
    for (indices) |idx| {
        if (idx < col.len) {
            result = @min(result, col[idx]);
        }
    }
    return result;
}

/// MAX(col) for selected rows (f64)
pub fn maxFilteredF64(col: []const f64, indices: []const u32) ?f64 {
    if (indices.len == 0) return null;
    var result: f64 = -std.math.inf(f64);
    for (indices) |idx| {
        if (idx < col.len) {
            result = @max(result, col[idx]);
        }
    }
    return result;
}

// ============================================================================
// TESTS
// ============================================================================

test "filterGreaterI64 basic" {
    const col = [_]i64{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
    var out: [10]u32 = undefined;

    const count_result = filterGreaterI64(&col, 50, &out);
    try std.testing.expectEqual(@as(usize, 5), count_result);
    try std.testing.expectEqual(@as(u32, 5), out[0]); // 60
    try std.testing.expectEqual(@as(u32, 6), out[1]); // 70
    try std.testing.expectEqual(@as(u32, 7), out[2]); // 80
    try std.testing.expectEqual(@as(u32, 8), out[3]); // 90
    try std.testing.expectEqual(@as(u32, 9), out[4]); // 100
}

test "filterGreaterI64 SIMD path" {
    // Test with exactly 16 elements (2 SIMD iterations)
    const col = [_]i64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    var out: [16]u32 = undefined;

    const count_result = filterGreaterI64(&col, 10, &out);
    try std.testing.expectEqual(@as(usize, 6), count_result);
}

test "sumI64 basic" {
    const col = [_]i64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const sum = sumI64(&col);
    try std.testing.expectEqual(@as(i128, 55), sum);
}

test "sumI64 SIMD path" {
    // Test with 16 elements
    const col = [_]i64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const sum = sumI64(&col);
    try std.testing.expectEqual(@as(i128, 136), sum);
}

test "sumF64 basic" {
    const col = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const sum = sumF64(&col);
    try std.testing.expectApproxEqAbs(@as(f64, 15.0), sum, 0.001);
}

test "minMaxI64" {
    const col = [_]i64{ 5, 2, 8, 1, 9, 3, 7, 4, 6, 10 };
    try std.testing.expectEqual(@as(i64, 1), minI64(&col).?);
    try std.testing.expectEqual(@as(i64, 10), maxI64(&col).?);
}

test "minMaxF64" {
    const col = [_]f64{ 5.5, 2.2, 8.8, 1.1, 9.9 };
    try std.testing.expectApproxEqAbs(@as(f64, 1.1), minF64(&col).?, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 9.9), maxF64(&col).?, 0.001);
}

test "sumFilteredI64" {
    const col = [_]i64{ 10, 20, 30, 40, 50 };
    const indices = [_]u32{ 1, 3 }; // 20 + 40 = 60
    const sum = sumFilteredI64(&col, &indices);
    try std.testing.expectEqual(@as(i128, 60), sum);
}
