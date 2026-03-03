//! Compression codecs for WASM
//!
//! Provides decompression for all Parquet compression formats.

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
// Gzip Decompression
// ============================================================================

/// Decompresses gzip-compressed data.
/// Returns: decompressed size on success, 0 on error
pub export fn gzip_decompress(
    compressed_ptr: [*]const u8,
    compressed_len: usize,
    decompressed_ptr: [*]u8,
    decompressed_capacity: usize,
) usize {
    const compressed = compressed_ptr[0..compressed_len];
    var reader = std.io.Reader.fixed(compressed);
    var writer = std.io.Writer.fixed(decompressed_ptr[0..decompressed_capacity]);

    var gzip_stream = std.compress.gzip.decompressor(&reader);
    const bytes_written = gzip_stream.reader.streamRemaining(&writer) catch {
        return 0;
    };
    return bytes_written;
}

// ============================================================================
// LZ4 Block Decompression (Parquet uses raw LZ4 block format)
// ============================================================================

/// Decompresses LZ4 block format data (not LZ4 frame format).
/// Parquet uses raw LZ4 block compression (hadoop codec).
/// Returns: decompressed size on success, 0 on error
pub export fn lz4_block_decompress(
    compressed_ptr: [*]const u8,
    compressed_len: usize,
    decompressed_ptr: [*]u8,
    decompressed_capacity: usize,
) usize {
    const compressed = compressed_ptr[0..compressed_len];
    const output = decompressed_ptr[0..decompressed_capacity];

    var pos: usize = 0;
    var out_pos: usize = 0;

    while (pos < compressed.len) {
        if (pos >= compressed.len) break;
        const token = compressed[pos];
        pos += 1;

        // Literal length
        var lit_len: usize = (token >> 4) & 0x0F;
        if (lit_len == 15) {
            while (pos < compressed.len) {
                const extra = compressed[pos];
                pos += 1;
                lit_len += extra;
                if (extra != 255) break;
            }
        }

        // Copy literals
        if (pos + lit_len > compressed.len) return 0;
        if (out_pos + lit_len > decompressed_capacity) return 0;
        @memcpy(output[out_pos..][0..lit_len], compressed[pos..][0..lit_len]);
        pos += lit_len;
        out_pos += lit_len;

        // Check if we're at the end (last sequence has no match)
        if (pos >= compressed.len) break;

        // Match offset (2 bytes, little-endian)
        if (pos + 2 > compressed.len) return 0;
        const offset: usize = @as(usize, compressed[pos]) | (@as(usize, compressed[pos + 1]) << 8);
        pos += 2;
        if (offset == 0) return 0; // invalid offset

        // Match length
        var match_len: usize = (token & 0x0F) + 4; // minimum match = 4
        if ((token & 0x0F) == 15) {
            while (pos < compressed.len) {
                const extra = compressed[pos];
                pos += 1;
                match_len += extra;
                if (extra != 255) break;
            }
        }

        // Copy match (may overlap — byte-by-byte for overlapping copies)
        if (out_pos < offset) return 0; // offset beyond output
        if (out_pos + match_len > decompressed_capacity) return 0;
        const match_start = out_pos - offset;
        for (0..match_len) |i| {
            output[out_pos + i] = output[match_start + i];
        }
        out_pos += match_len;
    }

    return out_pos;
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

test "compression: gzip_decompress empty input" {
    const compressed = [_]u8{};
    var output: [100]u8 = undefined;
    const result = gzip_decompress(&compressed, 0, &output, output.len);
    try std.testing.expectEqual(@as(usize, 0), result);
}

test "compression: lz4_block_decompress empty input" {
    const compressed = [_]u8{};
    var output: [100]u8 = undefined;
    const result = lz4_block_decompress(&compressed, 0, &output, output.len);
    try std.testing.expectEqual(@as(usize, 0), result);
}

test "compression: lz4_block_decompress literal only" {
    // LZ4 block: token=0x50 → 5 literals, no match (last sequence)
    const compressed = [_]u8{ 0x50, 'h', 'e', 'l', 'l', 'o' };
    var output: [100]u8 = undefined;
    const result = lz4_block_decompress(&compressed, compressed.len, &output, output.len);
    try std.testing.expectEqual(@as(usize, 5), result);
    try std.testing.expectEqualSlices(u8, "hello", output[0..5]);
}
