//! GGUF Utilities for WASM
//!
//! Shared utilities for parsing GGUF model files and transformer operations.
//! Used by both CLIP and MiniLM text encoders.

const std = @import("std");
const format = @import("format.zig");

const readU64LE = format.readU64LE;
const readU32LE = format.readU32LE;

// ============================================================================
// GGUF Type Constants
// ============================================================================

pub const GGUF_TYPE_STRING: u32 = 8;
pub const GGUF_TYPE_ARRAY: u32 = 9;
pub const GGUF_TYPE_UINT32: u32 = 4;
pub const GGUF_TYPE_FLOAT32: u32 = 6;
pub const GGUF_TYPE_BOOL: u32 = 7;
pub const GGUF_TYPE_UINT64: u32 = 10;

pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;

// ============================================================================
// GGUF Parsing Helpers
// ============================================================================

/// Read a GGUF string (length-prefixed)
pub fn ggufReadString(data: []const u8, pos: *usize) []const u8 {
    if (pos.* + 8 > data.len) return "";
    const len: usize = @intCast(readU64LE(data, pos.*));
    pos.* += 8;
    if (pos.* + len > data.len) return "";
    const str = data[pos.* .. pos.* + len];
    pos.* += len;
    return str;
}

/// Skip a GGUF value (for metadata we don't need)
pub fn ggufSkipValue(data: []const u8, pos: *usize, vtype: u32) void {
    switch (vtype) {
        GGUF_TYPE_STRING => {
            _ = ggufReadString(data, pos);
        },
        GGUF_TYPE_UINT32, GGUF_TYPE_FLOAT32 => {
            pos.* += 4;
        },
        GGUF_TYPE_BOOL => {
            pos.* += 1;
        },
        GGUF_TYPE_UINT64 => {
            pos.* += 8;
        },
        GGUF_TYPE_ARRAY => {
            if (pos.* + 12 > data.len) return;
            const atype = readU32LE(data, pos.*);
            pos.* += 4;
            const alen: usize = @intCast(readU64LE(data, pos.*));
            pos.* += 8;
            for (0..alen) |_| {
                ggufSkipValue(data, pos, atype);
            }
        },
        else => {},
    }
}

/// String equality comparison
pub fn strEql(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |ca, cb| {
        if (ca != cb) return false;
    }
    return true;
}

// ============================================================================
// Float16 Conversion
// ============================================================================

/// Convert IEEE 754 half-precision (FP16) to single-precision (FP32)
pub fn f16ToF32(h: u16) f32 {
    const sign: u32 = @as(u32, h >> 15) << 31;
    const exp: u32 = (h >> 10) & 0x1F;
    const mant: u32 = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            return @bitCast(sign);
        }
        // Subnormal
        var e: u32 = 0;
        var m = mant;
        while ((m & 0x400) == 0) {
            m <<= 1;
            e += 1;
        }
        const new_exp = (127 - 15 - e) << 23;
        const new_mant = (m & 0x3FF) << 13;
        return @bitCast(sign | new_exp | new_mant);
    } else if (exp == 31) {
        // Inf/NaN
        return @bitCast(sign | 0x7F800000 | (mant << 13));
    }

    const new_exp = (exp + 127 - 15) << 23;
    const new_mant = mant << 13;
    return @bitCast(sign | new_exp | new_mant);
}

// ============================================================================
// Transformer Math Functions
// ============================================================================

/// Error function (erf) approximation using Abramowitz and Stegun formula 7.1.26
/// Maximum error: 1.5e-7
pub fn erf(x: f32) f32 {
    // Constants for the approximation
    const a1: f32 = 0.254829592;
    const a2: f32 = -0.284496736;
    const a3: f32 = 1.421413741;
    const a4: f32 = -1.453152027;
    const a5: f32 = 1.061405429;
    const p: f32 = 0.3275911;

    // Save the sign of x
    const sign: f32 = if (x < 0) -1.0 else 1.0;
    const abs_x = @abs(x);

    // A&S formula 7.1.26
    const t = 1.0 / (1.0 + p * abs_x);
    const t2 = t * t;
    const t3 = t2 * t;
    const t4 = t3 * t;
    const t5 = t4 * t;

    const y = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * @exp(-abs_x * abs_x);

    return sign * y;
}

/// Standard GELU activation (exact, using erf)
/// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
pub fn gelu(x: f32) f32 {
    const sqrt2_inv: f32 = 0.7071067811865476; // 1 / sqrt(2)
    return x * 0.5 * (1.0 + erf(x * sqrt2_inv));
}

/// Softmax over rows (each row of length seq_len)
pub fn softmax(data: []f32, seq_len: usize) void {
    const num_rows = data.len / seq_len;
    for (0..num_rows) |i| {
        const row = data[i * seq_len .. (i + 1) * seq_len];

        // Find max for numerical stability
        var max_val: f32 = row[0];
        for (row[1..]) |v| {
            if (v > max_val) max_val = v;
        }

        // Exp and sum
        var sum: f32 = 0;
        for (row) |*v| {
            v.* = @exp(v.* - max_val);
            sum += v.*;
        }

        // Normalize
        for (row) |*v| {
            v.* /= sum;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "gguf_utils: f16ToF32 zero" {
    const result = f16ToF32(0);
    try std.testing.expectEqual(@as(f32, 0.0), result);
}

test "gguf_utils: f16ToF32 one" {
    // FP16 representation of 1.0: sign=0, exp=15, mant=0 -> 0x3C00
    const result = f16ToF32(0x3C00);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result, 0.0001);
}

test "gguf_utils: f16ToF32 negative" {
    // FP16 representation of -1.0: sign=1, exp=15, mant=0 -> 0xBC00
    const result = f16ToF32(0xBC00);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), result, 0.0001);
}

test "gguf_utils: strEql match" {
    try std.testing.expect(strEql("hello", "hello"));
}

test "gguf_utils: strEql no match" {
    try std.testing.expect(!strEql("hello", "world"));
}

test "gguf_utils: strEql different length" {
    try std.testing.expect(!strEql("hello", "hi"));
}

test "gguf_utils: erf zero" {
    const result = erf(0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result, 0.0001);
}

test "gguf_utils: erf positive" {
    // erf(1) ≈ 0.8427
    const result = erf(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8427), result, 0.001);
}

test "gguf_utils: erf negative" {
    // erf(-1) ≈ -0.8427
    const result = erf(-1.0);
    try std.testing.expectApproxEqAbs(@as(f32, -0.8427), result, 0.001);
}

test "gguf_utils: gelu zero" {
    const result = gelu(0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result, 0.0001);
}

test "gguf_utils: gelu positive" {
    // GELU(1) ≈ 0.8413
    const result = gelu(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8413), result, 0.001);
}

test "gguf_utils: gelu negative" {
    // GELU(-1) ≈ -0.1587
    const result = gelu(-1.0);
    try std.testing.expectApproxEqAbs(@as(f32, -0.1587), result, 0.001);
}

test "gguf_utils: softmax" {
    var data = [_]f32{ 1.0, 2.0, 3.0 };
    softmax(&data, 3);

    // After softmax, should sum to 1
    var sum: f32 = 0;
    for (data) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.0001);

    // Largest input should have largest probability
    try std.testing.expect(data[2] > data[1]);
    try std.testing.expect(data[1] > data[0]);
}
