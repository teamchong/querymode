//! PLAIN encoding decoder for Parquet.
//!
//! PLAIN is the simplest encoding where values are stored back-to-back.
//! See: https://parquet.apache.org/docs/file-format/data-pages/encodings/

const std = @import("std");
const meta = @import("lanceql.format").parquet_metadata;
const Type = meta.Type;

pub const PlainError = error{
    UnexpectedEndOfData,
    InvalidType,
    OutOfMemory,
};

/// PLAIN decoder
pub const PlainDecoder = struct {
    data: []const u8,
    pos: usize,

    const Self = @This();

    pub fn init(data: []const u8) Self {
        return .{
            .data = data,
            .pos = 0,
        };
    }

    /// Remaining bytes
    pub fn remaining(self: Self) usize {
        return self.data.len - self.pos;
    }

    // ========================================================================
    // Boolean decoding
    // ========================================================================

    /// Read booleans (bit-packed, 1 bit per value)
    pub fn readBooleans(self: *Self, count: usize, allocator: std.mem.Allocator) PlainError![]bool {
        const bytes_needed = (count + 7) / 8;
        if (self.pos + bytes_needed > self.data.len) {
            return PlainError.UnexpectedEndOfData;
        }

        const values = try allocator.alloc(bool, count);
        errdefer allocator.free(values);

        // Process 8 values at a time from each byte (avoids div/mod per value)
        const full_bytes = count / 8;
        var idx: usize = 0;

        for (0..full_bytes) |byte_i| {
            const byte = self.data[self.pos + byte_i];
            // Unroll 8 bits
            values[idx] = (byte & 0x01) != 0;
            values[idx + 1] = (byte & 0x02) != 0;
            values[idx + 2] = (byte & 0x04) != 0;
            values[idx + 3] = (byte & 0x08) != 0;
            values[idx + 4] = (byte & 0x10) != 0;
            values[idx + 5] = (byte & 0x20) != 0;
            values[idx + 6] = (byte & 0x40) != 0;
            values[idx + 7] = (byte & 0x80) != 0;
            idx += 8;
        }

        // Handle remaining bits
        if (idx < count) {
            const byte = self.data[self.pos + full_bytes];
            var bit: u3 = 0;
            while (idx < count) : ({
                idx += 1;
                bit += 1;
            }) {
                values[idx] = ((byte >> bit) & 1) != 0;
            }
        }

        self.pos += bytes_needed;
        return values;
    }

    // ========================================================================
    // Generic numeric decoding helper
    // ========================================================================

    /// Generic helper to read fixed-size numeric values (little-endian, direct memcpy)
    fn readNumericValues(self: *Self, comptime T: type, count: usize, allocator: std.mem.Allocator) PlainError![]T {
        const bytes_needed = count * @sizeOf(T);
        if (self.pos + bytes_needed > self.data.len) {
            return PlainError.UnexpectedEndOfData;
        }

        const values = try allocator.alloc(T, count);
        errdefer allocator.free(values);

        const src = self.data[self.pos..][0..bytes_needed];
        @memcpy(std.mem.sliceAsBytes(values), src);

        self.pos += bytes_needed;
        return values;
    }

    // ========================================================================
    // Integer decoding
    // ========================================================================

    /// Read INT32 values (4 bytes each, little-endian)
    pub fn readInt32(self: *Self, count: usize, allocator: std.mem.Allocator) PlainError![]i32 {
        return self.readNumericValues(i32, count, allocator);
    }

    /// Read INT64 values (8 bytes each, little-endian)
    pub fn readInt64(self: *Self, count: usize, allocator: std.mem.Allocator) PlainError![]i64 {
        return self.readNumericValues(i64, count, allocator);
    }

    /// Read INT96 values (12 bytes each, deprecated timestamp format)
    pub fn readInt96(self: *Self, count: usize, allocator: std.mem.Allocator) PlainError![][12]u8 {
        const bytes_needed = count * 12;
        if (self.pos + bytes_needed > self.data.len) {
            return PlainError.UnexpectedEndOfData;
        }

        const values = try allocator.alloc([12]u8, count);
        errdefer allocator.free(values);

        for (0..count) |i| {
            const offset = self.pos + i * 12;
            @memcpy(&values[i], self.data[offset..][0..12]);
        }

        self.pos += bytes_needed;
        return values;
    }

    // ========================================================================
    // Floating point decoding
    // ========================================================================

    /// Read FLOAT values (4 bytes each, IEEE 754)
    pub fn readFloat(self: *Self, count: usize, allocator: std.mem.Allocator) PlainError![]f32 {
        return self.readNumericValues(f32, count, allocator);
    }

    /// Read DOUBLE values (8 bytes each, IEEE 754)
    pub fn readDouble(self: *Self, count: usize, allocator: std.mem.Allocator) PlainError![]f64 {
        return self.readNumericValues(f64, count, allocator);
    }

    // ========================================================================
    // Binary/String decoding
    // ========================================================================

    /// Read BYTE_ARRAY values (length-prefixed)
    /// Returns slices into the original data buffer
    pub fn readByteArray(self: *Self, count: usize, allocator: std.mem.Allocator) PlainError![][]const u8 {
        const values = try allocator.alloc([]const u8, count);
        errdefer allocator.free(values);

        for (0..count) |i| {
            if (self.pos + 4 > self.data.len) {
                return PlainError.UnexpectedEndOfData;
            }

            const len = std.mem.readInt(u32, self.data[self.pos..][0..4], .little);
            self.pos += 4;

            const len_usize: usize = @intCast(len);
            if (self.pos + len_usize > self.data.len) {
                return PlainError.UnexpectedEndOfData;
            }

            values[i] = self.data[self.pos..][0..len_usize];
            self.pos += len_usize;
        }

        return values;
    }

    /// Read FIXED_LEN_BYTE_ARRAY values
    pub fn readFixedLenByteArray(self: *Self, count: usize, type_length: usize, allocator: std.mem.Allocator) PlainError![][]const u8 {
        const bytes_needed = count * type_length;
        if (self.pos + bytes_needed > self.data.len) {
            return PlainError.UnexpectedEndOfData;
        }

        const values = try allocator.alloc([]const u8, count);
        errdefer allocator.free(values);

        for (0..count) |i| {
            const offset = self.pos + i * type_length;
            values[i] = self.data[offset..][0..type_length];
        }

        self.pos += bytes_needed;
        return values;
    }

    // ========================================================================
    // Raw data access
    // ========================================================================

    /// Read raw bytes
    pub fn readBytes(self: *Self, count: usize) PlainError![]const u8 {
        if (self.pos + count > self.data.len) {
            return PlainError.UnexpectedEndOfData;
        }

        const result = self.data[self.pos..][0..count];
        self.pos += count;
        return result;
    }

    /// Skip bytes
    pub fn skip(self: *Self, count: usize) PlainError!void {
        if (self.pos + count > self.data.len) {
            return PlainError.UnexpectedEndOfData;
        }
        self.pos += count;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "read int32" {
    // Values: [1, 2, 256, -1]
    const data = [_]u8{
        0x01, 0x00, 0x00, 0x00, // 1
        0x02, 0x00, 0x00, 0x00, // 2
        0x00, 0x01, 0x00, 0x00, // 256
        0xFF, 0xFF, 0xFF, 0xFF, // -1
    };

    var decoder = PlainDecoder.init(&data);
    const values = try decoder.readInt32(4, std.testing.allocator);
    defer std.testing.allocator.free(values);

    try std.testing.expectEqual(@as(i32, 1), values[0]);
    try std.testing.expectEqual(@as(i32, 2), values[1]);
    try std.testing.expectEqual(@as(i32, 256), values[2]);
    try std.testing.expectEqual(@as(i32, -1), values[3]);
}

test "read int64" {
    const data = [_]u8{
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 1
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // -1
    };

    var decoder = PlainDecoder.init(&data);
    const values = try decoder.readInt64(2, std.testing.allocator);
    defer std.testing.allocator.free(values);

    try std.testing.expectEqual(@as(i64, 1), values[0]);
    try std.testing.expectEqual(@as(i64, -1), values[1]);
}

test "read double" {
    var data: [16]u8 = undefined;
    const val1: f64 = 3.14159;
    const val2: f64 = -2.71828;
    std.mem.writeInt(u64, data[0..8], @bitCast(val1), .little);
    std.mem.writeInt(u64, data[8..16], @bitCast(val2), .little);

    var decoder = PlainDecoder.init(&data);
    const values = try decoder.readDouble(2, std.testing.allocator);
    defer std.testing.allocator.free(values);

    try std.testing.expectApproxEqAbs(val1, values[0], 0.00001);
    try std.testing.expectApproxEqAbs(val2, values[1], 0.00001);
}

test "read byte_array" {
    // Two strings: "hello" and "world"
    const data = [_]u8{
        0x05, 0x00, 0x00, 0x00, // length 5
        'h',  'e',  'l',  'l',  'o', // "hello"
        0x05, 0x00, 0x00, 0x00, // length 5
        'w',  'o',  'r',  'l',  'd', // "world"
    };

    var decoder = PlainDecoder.init(&data);
    const values = try decoder.readByteArray(2, std.testing.allocator);
    defer std.testing.allocator.free(values);

    try std.testing.expectEqualStrings("hello", values[0]);
    try std.testing.expectEqualStrings("world", values[1]);
}

test "read booleans" {
    // 8 booleans packed into 1 byte: true, false, true, true, false, false, true, false
    // = 0b01001101 = 0x4D
    const data = [_]u8{0x4D};

    var decoder = PlainDecoder.init(&data);
    const values = try decoder.readBooleans(8, std.testing.allocator);
    defer std.testing.allocator.free(values);

    try std.testing.expect(values[0] == true);
    try std.testing.expect(values[1] == false);
    try std.testing.expect(values[2] == true);
    try std.testing.expect(values[3] == true);
    try std.testing.expect(values[4] == false);
    try std.testing.expect(values[5] == false);
    try std.testing.expect(values[6] == true);
    try std.testing.expect(values[7] == false);
}

test "read float" {
    var data: [8]u8 = undefined;
    const val1: f32 = 1.5;
    const val2: f32 = -2.25;
    std.mem.writeInt(u32, data[0..4], @bitCast(val1), .little);
    std.mem.writeInt(u32, data[4..8], @bitCast(val2), .little);

    var decoder = PlainDecoder.init(&data);
    const values = try decoder.readFloat(2, std.testing.allocator);
    defer std.testing.allocator.free(values);

    try std.testing.expectApproxEqAbs(val1, values[0], 0.0001);
    try std.testing.expectApproxEqAbs(val2, values[1], 0.0001);
}

test "read int96" {
    // Two INT96 values (12 bytes each) - deprecated timestamp format
    const data = [_]u8{
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, // First
        0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, // Second
    };

    var decoder = PlainDecoder.init(&data);
    const values = try decoder.readInt96(2, std.testing.allocator);
    defer std.testing.allocator.free(values);

    try std.testing.expectEqualSlices(u8, data[0..12], &values[0]);
    try std.testing.expectEqualSlices(u8, data[12..24], &values[1]);
}

test "read fixed_len_byte_array" {
    // Three 4-byte fixed-length values
    const data = [_]u8{
        'A', 'B', 'C', 'D',
        'E', 'F', 'G', 'H',
        'I', 'J', 'K', 'L',
    };

    var decoder = PlainDecoder.init(&data);
    const values = try decoder.readFixedLenByteArray(3, 4, std.testing.allocator);
    defer std.testing.allocator.free(values);

    try std.testing.expectEqualStrings("ABCD", values[0]);
    try std.testing.expectEqualStrings("EFGH", values[1]);
    try std.testing.expectEqualStrings("IJKL", values[2]);
}

test "read bytes and skip" {
    const data = [_]u8{ 'H', 'E', 'L', 'L', 'O', 'W', 'O', 'R', 'L', 'D' };

    var decoder = PlainDecoder.init(&data);
    try std.testing.expectEqual(@as(usize, 10), decoder.remaining());

    const first = try decoder.readBytes(5);
    try std.testing.expectEqualStrings("HELLO", first);
    try std.testing.expectEqual(@as(usize, 5), decoder.remaining());

    try decoder.skip(2);
    try std.testing.expectEqual(@as(usize, 3), decoder.remaining());

    const last = try decoder.readBytes(3);
    try std.testing.expectEqualStrings("RLD", last);
    try std.testing.expectEqual(@as(usize, 0), decoder.remaining());
}

test "unexpected end of data error" {
    const data = [_]u8{ 0x01, 0x02 }; // Only 2 bytes

    var decoder = PlainDecoder.init(&data);

    // Try to read 4 bytes as int32 - should fail
    const result = decoder.readInt32(1, std.testing.allocator);
    try std.testing.expectError(PlainError.UnexpectedEndOfData, result);
}

test "read booleans partial byte" {
    // 5 booleans: true, true, false, true, false = 0b01011 = 0x0B
    const data = [_]u8{0x0B};

    var decoder = PlainDecoder.init(&data);
    const values = try decoder.readBooleans(5, std.testing.allocator);
    defer std.testing.allocator.free(values);

    try std.testing.expect(values[0] == true);
    try std.testing.expect(values[1] == true);
    try std.testing.expect(values[2] == false);
    try std.testing.expect(values[3] == true);
    try std.testing.expect(values[4] == false);
}
