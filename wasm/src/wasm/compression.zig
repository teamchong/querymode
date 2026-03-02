//! Zstd Decompression for WASM
//!
//! Provides zstd decompression utilities for compressed Lance data.

const std = @import("std");

// ============================================================================
// Zstd Decompression
// ============================================================================

/// Decompresses zstd-compressed data in place or to a destination buffer.
/// compressed_ptr: pointer to compressed data
/// compressed_len: length of compressed data
/// decompressed_ptr: pointer to output buffer (must be pre-allocated with enough space)
/// decompressed_capacity: capacity of output buffer
/// Returns: decompressed size on success, 0 on error
pub export fn zstd_decompress(
    compressed_ptr: [*]const u8,
    compressed_len: usize,
    decompressed_ptr: [*]u8,
    decompressed_capacity: usize,
) usize {
    const compressed = compressed_ptr[0..compressed_len];

    // Use std.io.Reader.fixed for the new Zig 0.15 Reader API
    var reader = std.io.Reader.fixed(compressed);

    // Use fixed Writer for output buffer
    var writer = std.io.Writer.fixed(decompressed_ptr[0..decompressed_capacity]);

    // Initialize zstd decompressor (empty buffer = direct streaming mode)
    var zstd_stream = std.compress.zstd.Decompress.init(&reader, &.{}, .{});

    // Stream all decompressed data to writer
    const bytes_written = zstd_stream.reader.streamRemaining(&writer) catch {
        return 0;
    };

    return bytes_written;
}

/// Returns the decompressed size from zstd frame header (if available).
/// This allows JS to know how much memory to allocate before decompression.
/// Returns 0 if size is unknown or error.
pub export fn zstd_get_decompressed_size(compressed_ptr: [*]const u8, compressed_len: usize) usize {
    if (compressed_len < 18) return 0; // Minimum frame header size

    const compressed = compressed_ptr[0..compressed_len];

    // Check magic number (0xFD2FB528 little endian)
    if (compressed[0] != 0x28 or compressed[1] != 0xB5 or
        compressed[2] != 0x2F or compressed[3] != 0xFD)
    {
        return 0;
    }

    // Frame header descriptor
    const fhd = compressed[4];
    const fcs_flag = (fhd >> 6) & 0x03; // Frame content size flag
    const single_segment = (fhd >> 5) & 0x01;
    const dict_id_flag = fhd & 0x03;

    // Calculate header size
    var offset: usize = 5;

    // Window descriptor (if not single segment)
    if (single_segment == 0) {
        offset += 1;
    }

    // Dictionary ID
    const dict_id_sizes = [_]usize{ 0, 1, 2, 4 };
    offset += dict_id_sizes[dict_id_flag];

    // Frame content size
    if (fcs_flag == 0 and single_segment == 1) {
        // 1 byte
        if (offset >= compressed_len) return 0;
        return compressed[offset];
    } else if (fcs_flag == 1) {
        // 2 bytes
        if (offset + 2 > compressed_len) return 0;
        return @as(usize, std.mem.readInt(u16, compressed[offset..][0..2], .little)) + 256;
    } else if (fcs_flag == 2) {
        // 4 bytes
        if (offset + 4 > compressed_len) return 0;
        return std.mem.readInt(u32, compressed[offset..][0..4], .little);
    } else if (fcs_flag == 3) {
        // 8 bytes
        if (offset + 8 > compressed_len) return 0;
        const size = std.mem.readInt(u64, compressed[offset..][0..8], .little);
        if (size > std.math.maxInt(usize)) return 0;
        return @intCast(size);
    }

    return 0;
}

// ============================================================================
// Tests
// ============================================================================

test "compression: zstd_get_decompressed_size too short" {
    const data = [_]u8{ 0x28, 0xB5, 0x2F, 0xFD };
    const result = zstd_get_decompressed_size(&data, data.len);
    try std.testing.expectEqual(@as(usize, 0), result);
}

test "compression: zstd_get_decompressed_size invalid magic" {
    var data: [20]u8 = undefined;
    @memset(&data, 0);
    data[0] = 0x00; // Invalid magic
    const result = zstd_get_decompressed_size(&data, data.len);
    try std.testing.expectEqual(@as(usize, 0), result);
}

test "compression: zstd_decompress empty input" {
    const compressed = [_]u8{};
    var output: [100]u8 = undefined;
    const result = zstd_decompress(&compressed, 0, &output, output.len);
    // Empty input should return 0 (error or no data)
    try std.testing.expectEqual(@as(usize, 0), result);
}
