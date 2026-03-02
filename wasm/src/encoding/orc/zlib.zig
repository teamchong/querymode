//! ORC Compression Block Handler
//!
//! ORC uses a block-based compression format with 3-byte headers.
//! Each block: [3-byte header][compressed or uncompressed data]
//!
//! Header format (little-endian 24-bit):
//! - Bit 0: is_original (1 = uncompressed, 0 = compressed)
//! - Bits 1-23: (original_size - 1) for uncompressed, or compressed_size for compressed
//!
//! Supported codecs: none, snappy, zlib (deflate)

const std = @import("std");
const Allocator = std.mem.Allocator;
const flate = std.compress.flate;

/// ORC compression codecs
pub const CompressionKind = enum {
    none,
    zlib,
    snappy,
    lzo,
    lz4,
    zstd,
    unknown,
};

/// Errors during decompression
pub const DecompressError = error{
    InvalidBlockHeader,
    DecompressionFailed,
    OutputOverflow,
    UnsupportedCompression,
    OutOfMemory,
};

/// Parse ORC 3-byte compression block header
pub const BlockHeader = struct {
    is_original: bool,
    length: u32,

    pub fn parse(data: []const u8) ?BlockHeader {
        if (data.len < 3) return null;

        // Read 3 bytes as little-endian 24-bit value
        const header: u32 = @as(u32, data[0]) |
            (@as(u32, data[1]) << 8) |
            (@as(u32, data[2]) << 16);

        const is_original = (header & 1) == 1;
        const length = (header >> 1);

        return .{
            .is_original = is_original,
            .length = length,
        };
    }
};

/// Decompress a single zlib/deflate block
pub fn decompressZlib(compressed: []const u8, estimated_size: usize, allocator: Allocator) DecompressError![]u8 {
    _ = estimated_size; // Not needed with allocRemaining

    // Create input reader from compressed data using std.Io.Reader.fixed()
    var input_reader = std.Io.Reader.fixed(compressed);

    // Create window buffer for decompression
    var window_buf: [flate.max_window_len]u8 = undefined;

    // Initialize decompressor for raw deflate (no zlib/gzip headers)
    var decomp = flate.Decompress.init(&input_reader, .raw, &window_buf);

    // Read all decompressed data using allocRemaining
    // Use a reasonable limit (10MB should be enough for ORC blocks)
    const max_decompressed_size: usize = 10 * 1024 * 1024;
    const result = decomp.reader.allocRemaining(allocator, std.Io.Limit.limited(max_decompressed_size)) catch |err| {
        return switch (err) {
            error.OutOfMemory => DecompressError.OutOfMemory,
            error.StreamTooLong => DecompressError.OutputOverflow,
            else => DecompressError.DecompressionFailed,
        };
    };

    // Check for decompression errors
    if (decomp.err) |_| {
        allocator.free(result);
        return DecompressError.DecompressionFailed;
    }

    return result;
}

/// Decompress ORC stream with block format
/// Handles multiple compression blocks, each with 3-byte headers
pub fn decompressOrcStream(
    data: []const u8,
    codec: CompressionKind,
    allocator: Allocator,
) DecompressError![]u8 {
    if (codec == .none) {
        // No compression, just copy
        const output = allocator.dupe(u8, data) catch return DecompressError.OutOfMemory;
        return output;
    }

    var output = std.ArrayListUnmanaged(u8){};
    errdefer output.deinit(allocator);

    var pos: usize = 0;

    while (pos < data.len) {
        // Parse block header
        const header = BlockHeader.parse(data[pos..]) orelse {
            return DecompressError.InvalidBlockHeader;
        };
        pos += 3;

        if (header.is_original) {
            // Uncompressed block - length IS the actual uncompressed size
            const chunk_size = header.length;
            if (pos + chunk_size > data.len) {
                return DecompressError.InvalidBlockHeader;
            }
            output.appendSlice(allocator, data[pos..][0..chunk_size]) catch {
                return DecompressError.OutOfMemory;
            };
            pos += chunk_size;
        } else {
            // Compressed block - length is compressed size
            const compressed_size = header.length;
            if (pos + compressed_size > data.len) {
                return DecompressError.InvalidBlockHeader;
            }

            const compressed_data = data[pos..][0..compressed_size];

            switch (codec) {
                .zlib => {
                    // For zlib, we need to estimate output size
                    // Typically deflate has ~2-10x compression ratio
                    // Use a reasonable estimate and grow if needed
                    const estimated_size = compressed_size * 10;
                    const decompressed = try decompressZlib(compressed_data, estimated_size, allocator);
                    defer allocator.free(decompressed);
                    output.appendSlice(allocator, decompressed) catch {
                        return DecompressError.OutOfMemory;
                    };
                },
                .snappy => {
                    // Snappy requires external decompressor - caller should provide decompressed data
                    // Return error here; snappy handling is done in orc.zig
                    return DecompressError.UnsupportedCompression;
                },
                .lzo, .lz4, .zstd => {
                    return DecompressError.UnsupportedCompression;
                },
                .none, .unknown => {
                    return DecompressError.UnsupportedCompression;
                },
            }

            pos += compressed_size;
        }
    }

    return output.toOwnedSlice(allocator) catch return DecompressError.OutOfMemory;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "block header parsing" {
    // Test uncompressed block: is_original=1, length=99 -> 0x000063 << 1 | 1 = 0x0000C7
    // Little-endian: C7 00 00
    const uncompressed = [_]u8{ 0xC7, 0x00, 0x00 };
    const hdr1 = BlockHeader.parse(&uncompressed).?;
    try testing.expect(hdr1.is_original);
    try testing.expectEqual(@as(u32, 99), hdr1.length);

    // Test compressed block: is_original=0, length=50 -> 0x000032 << 1 | 0 = 0x000064
    // Little-endian: 64 00 00
    const compressed = [_]u8{ 0x64, 0x00, 0x00 };
    const hdr2 = BlockHeader.parse(&compressed).?;
    try testing.expect(!hdr2.is_original);
    try testing.expectEqual(@as(u32, 50), hdr2.length);
}

test "decompress uncompressed stream" {
    const allocator = testing.allocator;

    // Create an "uncompressed" ORC block: header + data
    // is_original=1, length=4 (meaning 5 bytes of data)
    // Header: (4 << 1) | 1 = 9 = 0x09
    const data = [_]u8{ 0x09, 0x00, 0x00, 'h', 'e', 'l', 'l', 'o' };

    const result = try decompressOrcStream(&data, .none, allocator);
    defer allocator.free(result);

    // With codec=none, it just copies the data (no block parsing)
    try testing.expectEqualSlices(u8, &data, result);
}
