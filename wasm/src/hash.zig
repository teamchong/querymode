//! Efficient hash functions for GROUP BY and JOIN operations.
//!
//! Uses FNV-1a hashing optimized for composite keys (multiple columns).
//! Avoids string allocation overhead by hashing raw values directly.

const std = @import("std");

/// FNV-1a constants
pub const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
pub const FNV_PRIME: u64 = 0x100000001b3;

/// Hash a string using FNV-1a
pub fn stringHash(str: []const u8) u64 {
    var hash: u64 = FNV_OFFSET_BASIS;
    for (str) |byte| {
        hash ^= byte;
        hash *%= FNV_PRIME;
    }
    return hash;
}

/// Hash a 64-bit value with good bit mixing (Murmur3 finalizer)
pub fn hashU64(val: u64) u64 {
    var h = val;
    h ^= h >> 33;
    h *%= 0xff51afd7ed558ccd;
    h ^= h >> 33;
    h *%= 0xc4ceb9fe1a85ec53;
    h ^= h >> 33;
    return h;
}

/// Hash a 32-bit value
pub fn hashU32(val: u32) u64 {
    return hashU64(@as(u64, val));
}

/// Hash an i64 by reinterpreting as u64
pub fn hashI64(val: i64) u64 {
    return hashU64(@bitCast(val));
}

/// Hash an i32 by reinterpreting
pub fn hashI32(val: i32) u64 {
    return hashU64(@as(u64, @bitCast(@as(i64, val))));
}

/// Hash a f64 by reinterpreting as u64
pub fn hashF64(val: f64) u64 {
    return hashU64(@bitCast(val));
}

/// Hash a f32 by promoting to f64
pub fn hashF32(val: f32) u64 {
    return hashU64(@as(u64, @bitCast(@as(f64, val))));
}

/// Hash a bool
pub fn hashBool(val: bool) u64 {
    return if (val) @as(u64, 1) else @as(u64, 0);
}

/// Combine two hashes (for composite keys)
pub fn combineHash(h1: u64, h2: u64) u64 {
    var hash = h1;
    hash ^= h2 +% 0x9e3779b97f4a7c15 +% (hash << 6) +% (hash >> 2);
    return hash;
}

// Tests
test "stringHash basic" {
    const h1 = stringHash("hello");
    const h2 = stringHash("hello");
    const h3 = stringHash("world");

    try std.testing.expectEqual(h1, h2);
    try std.testing.expect(h1 != h3);
}

test "hashI64 same values equal" {
    const h1 = hashI64(42);
    const h2 = hashI64(42);
    const h3 = hashI64(43);

    try std.testing.expectEqual(h1, h2);
    try std.testing.expect(h1 != h3);
}

test "combineHash order matters" {
    const h1 = combineHash(hashI64(1), hashI64(2));
    const h2 = combineHash(hashI64(2), hashI64(1));

    try std.testing.expect(h1 != h2);
}

test "hashF64 handles special values" {
    const h_zero = hashF64(0.0);
    const h_one = hashF64(1.0);
    const h_nan = hashF64(std.math.nan(f64));

    try std.testing.expect(h_zero != h_one);
    // NaN should still produce a hash
    _ = h_nan;
}
