//! Lance File Format Utilities
//!
//! Binary reading helpers and footer parsing for Lance files.

const std = @import("std");

// ============================================================================
// Constants
// ============================================================================

pub const FOOTER_SIZE: usize = 40;
pub const LANCE_MAGIC = "LANC";

// ============================================================================
// Binary Reading Helpers
// ============================================================================

pub fn readU64LE(data: []const u8, offset: usize) u64 {
    if (offset + 8 > data.len) return 0;
    return std.mem.readInt(u64, data[offset..][0..8], .little);
}

pub fn readU32LE(data: []const u8, offset: usize) u32 {
    if (offset + 4 > data.len) return 0;
    return std.mem.readInt(u32, data[offset..][0..4], .little);
}

pub fn readU16LE(data: []const u8, offset: usize) u16 {
    if (offset + 2 > data.len) return 0;
    return std.mem.readInt(u16, data[offset..][0..2], .little);
}

pub fn readI64LE(data: []const u8, offset: usize) i64 {
    if (offset + 8 > data.len) return 0;
    return std.mem.readInt(i64, data[offset..][0..8], .little);
}

pub fn readI32LE(data: []const u8, offset: usize) i32 {
    if (offset + 4 > data.len) return 0;
    return std.mem.readInt(i32, data[offset..][0..4], .little);
}

pub fn readI16LE(data: []const u8, offset: usize) i16 {
    if (offset + 2 > data.len) return 0;
    return std.mem.readInt(i16, data[offset..][0..2], .little);
}

pub fn readI8(data: []const u8, offset: usize) i8 {
    if (offset >= data.len) return 0;
    return @bitCast(data[offset]);
}

pub fn readU8(data: []const u8, offset: usize) u8 {
    if (offset >= data.len) return 0;
    return data[offset];
}

pub fn readF64LE(data: []const u8, offset: usize) f64 {
    const bits = readU64LE(data, offset);
    return @bitCast(bits);
}

pub fn readF32LE(data: []const u8, offset: usize) f32 {
    const bits = readU32LE(data, offset);
    return @bitCast(bits);
}

/// Read varint (up to 64 bits)
pub fn readVarint(data: []const u8, offset: *usize) u64 {
    var result: u64 = 0;
    var shift: u6 = 0;

    while (offset.* < data.len) {
        const byte = data[offset.*];
        offset.* += 1;

        result |= @as(u64, byte & 0x7F) << shift;
        if (byte & 0x80 == 0) break;

        shift += 7;
        if (shift >= 64) break;
    }

    return result;
}

// ============================================================================
// Footer Parsing
// ============================================================================

/// Check if data contains a valid Lance file
pub fn isValidLanceFileSlice(data: []const u8) bool {
    if (data.len < FOOTER_SIZE) return false;

    // Check magic at end
    const magic_offset = data.len - 4;
    return data[magic_offset] == 'L' and
        data[magic_offset + 1] == 'A' and
        data[magic_offset + 2] == 'N' and
        data[magic_offset + 3] == 'C';
}

/// Parse footer and get column count
pub fn parseFooterGetColumnsSlice(data: []const u8) u32 {
    if (!isValidLanceFileSlice(data)) return 0;
    const footer_start = data.len - FOOTER_SIZE;
    return readU32LE(data, footer_start + 28);
}

/// Parse footer and get major version
pub fn parseFooterGetMajorVersionSlice(data: []const u8) u16 {
    if (!isValidLanceFileSlice(data)) return 0;
    const footer_start = data.len - FOOTER_SIZE;
    return readU16LE(data, footer_start + 32);
}

/// Parse footer and get minor version
pub fn parseFooterGetMinorVersionSlice(data: []const u8) u16 {
    if (!isValidLanceFileSlice(data)) return 0;
    const footer_start = data.len - FOOTER_SIZE;
    return readU16LE(data, footer_start + 34);
}

/// Get column metadata start offset
pub fn getColumnMetaStartSlice(data: []const u8) u64 {
    if (!isValidLanceFileSlice(data)) return 0;
    const footer_start = data.len - FOOTER_SIZE;
    return readU64LE(data, footer_start + 0);
}

/// Get column metadata offsets start
pub fn getColumnMetaOffsetsStartSlice(data: []const u8) u64 {
    if (!isValidLanceFileSlice(data)) return 0;
    const footer_start = data.len - FOOTER_SIZE;
    return readU64LE(data, footer_start + 8);
}

// ============================================================================
// WASM Exports (raw pointer versions for JavaScript)
// ============================================================================

pub export fn isValidLanceFile(data: [*]const u8, len: usize) u32 {
    if (len < FOOTER_SIZE) return 0;
    return if (isValidLanceFileSlice(data[0..len])) 1 else 0;
}

pub export fn parseFooterGetColumns(data: [*]const u8, len: usize) u32 {
    if (len < FOOTER_SIZE) return 0;
    return parseFooterGetColumnsSlice(data[0..len]);
}

pub export fn parseFooterGetMajorVersion(data: [*]const u8, len: usize) u16 {
    if (len < FOOTER_SIZE) return 0;
    return parseFooterGetMajorVersionSlice(data[0..len]);
}

pub export fn parseFooterGetMinorVersion(data: [*]const u8, len: usize) u16 {
    if (len < FOOTER_SIZE) return 0;
    return parseFooterGetMinorVersionSlice(data[0..len]);
}

pub export fn getColumnMetaStart(data: [*]const u8, len: usize) u64 {
    if (len < FOOTER_SIZE) return 0;
    return getColumnMetaStartSlice(data[0..len]);
}

pub export fn getColumnMetaOffsetsStart(data: [*]const u8, len: usize) u64 {
    if (len < FOOTER_SIZE) return 0;
    return getColumnMetaOffsetsStartSlice(data[0..len]);
}

pub export fn getVersion() u32 {
    return 0x000100; // v0.1.0
}

// ============================================================================
// Tests
// ============================================================================

test "format: readU64LE" {
    const data = [_]u8{ 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08 };
    const result = readU64LE(&data, 0);
    try std.testing.expectEqual(@as(u64, 0x0807060504030201), result);
}

test "format: readU32LE" {
    const data = [_]u8{ 0x01, 0x02, 0x03, 0x04 };
    const result = readU32LE(&data, 0);
    try std.testing.expectEqual(@as(u32, 0x04030201), result);
}

test "format: readVarint" {
    // Single byte: 42
    const data1 = [_]u8{42};
    var offset1: usize = 0;
    try std.testing.expectEqual(@as(u64, 42), readVarint(&data1, &offset1));

    // Multi-byte: 300 = 0xAC 0x02
    const data2 = [_]u8{ 0xAC, 0x02 };
    var offset2: usize = 0;
    try std.testing.expectEqual(@as(u64, 300), readVarint(&data2, &offset2));
}

test "format: isValidLanceFile" {
    // Too short
    const short = [_]u8{ 'L', 'A', 'N', 'C' };
    try std.testing.expect(!isValidLanceFileSlice(&short));

    // Valid (40 bytes with LANC at end)
    var valid: [40]u8 = undefined;
    @memset(&valid, 0);
    valid[36] = 'L';
    valid[37] = 'A';
    valid[38] = 'N';
    valid[39] = 'C';
    try std.testing.expect(isValidLanceFileSlice(&valid));

    // Invalid magic
    var invalid: [40]u8 = undefined;
    @memset(&invalid, 0);
    invalid[36] = 'X';
    invalid[37] = 'X';
    invalid[38] = 'X';
    invalid[39] = 'X';
    try std.testing.expect(!isValidLanceFileSlice(&invalid));
}
