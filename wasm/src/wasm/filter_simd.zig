//! SIMD-Accelerated Column Filtering for WASM
//!
//! Per-op filter functions using WASM SIMD 128-bit vectors.
//! Eliminates per-element switch dispatch in the hot loop.
//! Pattern from vectorjson's comptime-unrolled SIMD comparisons.

const std = @import("std");

// WASM SIMD 128-bit = 2 x i64 or 2 x f64
const Vec2i64 = @Vector(2, i64);
const Vec2f64 = @Vector(2, f64);

// ============================================================================
// i64 Filters
// ============================================================================

fn filterI64Generic(
    data: []const u8,
    start: usize,
    count: usize,
    comptime op: enum { eq, ne, lt, le, gt, ge },
    value: i64,
    out: [*]u32,
    max: usize,
) usize {
    const val_vec: Vec2i64 = @splat(value);
    var out_count: usize = 0;
    var i: usize = 0;

    // SIMD loop: 2 i64s per iteration
    while (i + 2 <= count and out_count + 2 <= max) {
        const offset = start + i * 8;
        if (offset + 16 > data.len) break;
        const a = std.mem.readInt(i64, data[offset..][0..8], .little);
        const b = std.mem.readInt(i64, data[offset + 8 ..][0..8], .little);
        const vec: Vec2i64 = .{ a, b };
        const mask = switch (op) {
            .eq => vec == val_vec,
            .ne => vec != val_vec,
            .lt => vec < val_vec,
            .le => vec <= val_vec,
            .gt => vec > val_vec,
            .ge => vec >= val_vec,
        };
        if (mask[0]) {
            out[out_count] = @intCast(i);
            out_count += 1;
        }
        if (mask[1]) {
            out[out_count] = @intCast(i + 1);
            out_count += 1;
        }
        i += 2;
    }

    // Scalar tail
    while (i < count and out_count < max) : (i += 1) {
        const offset = start + i * 8;
        if (offset + 8 > data.len) break;
        const col_val = std.mem.readInt(i64, data[offset..][0..8], .little);
        const matches = switch (op) {
            .eq => col_val == value,
            .ne => col_val != value,
            .lt => col_val < value,
            .le => col_val <= value,
            .gt => col_val > value,
            .ge => col_val >= value,
        };
        if (matches) {
            out[out_count] = @intCast(i);
            out_count += 1;
        }
    }
    return out_count;
}

pub fn filterI64Eq(data: []const u8, start: usize, count: usize, value: i64, out: [*]u32, max: usize) usize {
    return filterI64Generic(data, start, count, .eq, value, out, max);
}
pub fn filterI64Ne(data: []const u8, start: usize, count: usize, value: i64, out: [*]u32, max: usize) usize {
    return filterI64Generic(data, start, count, .ne, value, out, max);
}
pub fn filterI64Lt(data: []const u8, start: usize, count: usize, value: i64, out: [*]u32, max: usize) usize {
    return filterI64Generic(data, start, count, .lt, value, out, max);
}
pub fn filterI64Le(data: []const u8, start: usize, count: usize, value: i64, out: [*]u32, max: usize) usize {
    return filterI64Generic(data, start, count, .le, value, out, max);
}
pub fn filterI64Gt(data: []const u8, start: usize, count: usize, value: i64, out: [*]u32, max: usize) usize {
    return filterI64Generic(data, start, count, .gt, value, out, max);
}
pub fn filterI64Ge(data: []const u8, start: usize, count: usize, value: i64, out: [*]u32, max: usize) usize {
    return filterI64Generic(data, start, count, .ge, value, out, max);
}

// ============================================================================
// f64 Filters
// ============================================================================

fn filterF64Generic(
    data: []const u8,
    start: usize,
    count: usize,
    comptime op: enum { eq, ne, lt, le, gt, ge },
    value: f64,
    out: [*]u32,
    max: usize,
) usize {
    const val_vec: Vec2f64 = @splat(value);
    var out_count: usize = 0;
    var i: usize = 0;

    // SIMD loop: 2 f64s per iteration
    while (i + 2 <= count and out_count + 2 <= max) {
        const offset = start + i * 8;
        if (offset + 16 > data.len) break;
        const a: f64 = @bitCast(std.mem.readInt(u64, data[offset..][0..8], .little));
        const b: f64 = @bitCast(std.mem.readInt(u64, data[offset + 8 ..][0..8], .little));
        const vec: Vec2f64 = .{ a, b };
        const mask = switch (op) {
            .eq => vec == val_vec,
            .ne => vec != val_vec,
            .lt => vec < val_vec,
            .le => vec <= val_vec,
            .gt => vec > val_vec,
            .ge => vec >= val_vec,
        };
        if (mask[0]) {
            out[out_count] = @intCast(i);
            out_count += 1;
        }
        if (mask[1]) {
            out[out_count] = @intCast(i + 1);
            out_count += 1;
        }
        i += 2;
    }

    // Scalar tail
    while (i < count and out_count < max) : (i += 1) {
        const offset = start + i * 8;
        if (offset + 8 > data.len) break;
        const col_val: f64 = @bitCast(std.mem.readInt(u64, data[offset..][0..8], .little));
        const matches = switch (op) {
            .eq => col_val == value,
            .ne => col_val != value,
            .lt => col_val < value,
            .le => col_val <= value,
            .gt => col_val > value,
            .ge => col_val >= value,
        };
        if (matches) {
            out[out_count] = @intCast(i);
            out_count += 1;
        }
    }
    return out_count;
}

pub fn filterF64Eq(data: []const u8, start: usize, count: usize, value: f64, out: [*]u32, max: usize) usize {
    return filterF64Generic(data, start, count, .eq, value, out, max);
}
pub fn filterF64Ne(data: []const u8, start: usize, count: usize, value: f64, out: [*]u32, max: usize) usize {
    return filterF64Generic(data, start, count, .ne, value, out, max);
}
pub fn filterF64Lt(data: []const u8, start: usize, count: usize, value: f64, out: [*]u32, max: usize) usize {
    return filterF64Generic(data, start, count, .lt, value, out, max);
}
pub fn filterF64Le(data: []const u8, start: usize, count: usize, value: f64, out: [*]u32, max: usize) usize {
    return filterF64Generic(data, start, count, .le, value, out, max);
}
pub fn filterF64Gt(data: []const u8, start: usize, count: usize, value: f64, out: [*]u32, max: usize) usize {
    return filterF64Generic(data, start, count, .gt, value, out, max);
}
pub fn filterF64Ge(data: []const u8, start: usize, count: usize, value: f64, out: [*]u32, max: usize) usize {
    return filterF64Generic(data, start, count, .ge, value, out, max);
}

// ============================================================================
// Tests
// ============================================================================

test "filter_simd: i64 less than" {
    // 4 values: 10, 20, 30, 40
    var data: [32]u8 = undefined;
    std.mem.writeInt(i64, data[0..8], 10, .little);
    std.mem.writeInt(i64, data[8..16], 20, .little);
    std.mem.writeInt(i64, data[16..24], 30, .little);
    std.mem.writeInt(i64, data[24..32], 40, .little);

    var out: [4]u32 = undefined;
    const count = filterI64Lt(&data, 0, 4, 25, &out, 4);
    try std.testing.expectEqual(@as(usize, 2), count);
    try std.testing.expectEqual(@as(u32, 0), out[0]);
    try std.testing.expectEqual(@as(u32, 1), out[1]);
}

test "filter_simd: i64 equal" {
    var data: [32]u8 = undefined;
    std.mem.writeInt(i64, data[0..8], 10, .little);
    std.mem.writeInt(i64, data[8..16], 20, .little);
    std.mem.writeInt(i64, data[16..24], 20, .little);
    std.mem.writeInt(i64, data[24..32], 40, .little);

    var out: [4]u32 = undefined;
    const count = filterI64Eq(&data, 0, 4, 20, &out, 4);
    try std.testing.expectEqual(@as(usize, 2), count);
    try std.testing.expectEqual(@as(u32, 1), out[0]);
    try std.testing.expectEqual(@as(u32, 2), out[1]);
}

test "filter_simd: f64 greater than" {
    var data: [32]u8 = undefined;
    std.mem.writeInt(u64, data[0..8], @bitCast(@as(f64, 1.5)), .little);
    std.mem.writeInt(u64, data[8..16], @bitCast(@as(f64, 2.5)), .little);
    std.mem.writeInt(u64, data[16..24], @bitCast(@as(f64, 3.5)), .little);
    std.mem.writeInt(u64, data[24..32], @bitCast(@as(f64, 0.5)), .little);

    var out: [4]u32 = undefined;
    const count = filterF64Gt(&data, 0, 4, 2.0, &out, 4);
    try std.testing.expectEqual(@as(usize, 2), count);
    try std.testing.expectEqual(@as(u32, 1), out[0]);
    try std.testing.expectEqual(@as(u32, 2), out[1]);
}

test "filter_simd: odd count (scalar tail)" {
    // 3 values: 10, 20, 30
    var data: [24]u8 = undefined;
    std.mem.writeInt(i64, data[0..8], 10, .little);
    std.mem.writeInt(i64, data[8..16], 20, .little);
    std.mem.writeInt(i64, data[16..24], 30, .little);

    var out: [3]u32 = undefined;
    const count = filterI64Ge(&data, 0, 3, 20, &out, 3);
    try std.testing.expectEqual(@as(usize, 2), count);
    try std.testing.expectEqual(@as(u32, 1), out[0]);
    try std.testing.expectEqual(@as(u32, 2), out[1]);
}
