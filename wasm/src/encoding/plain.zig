//! Plain (uncompressed) encoding decoder.
//!
//! Plain encoding stores values directly as fixed-width binary data.
//! This is the simplest encoding and serves as the baseline for MVP.

const std = @import("std");

/// Errors that can occur during decoding
pub const DecodeError = error{
    /// Buffer is not aligned correctly for the data type
    MisalignedBuffer,
    /// Buffer size doesn't match expected row count
    InvalidBufferSize,
    /// Requested index is out of bounds
    IndexOutOfBounds,
};

/// Decoder for plain-encoded column data.
pub const PlainDecoder = struct {
    data: []const u8,

    const Self = @This();

    /// Create a decoder for the given buffer.
    pub fn init(data: []const u8) Self {
        return Self{ .data = data };
    }

    // ========================================================================
    // Generic helpers
    // ========================================================================

    /// Get the count of values for a given type
    fn valueCount(self: Self, comptime T: type) usize {
        return self.data.len / @sizeOf(T);
    }

    /// Read a single integer value at the given index
    fn readIntValue(self: Self, comptime T: type, index: usize) DecodeError!T {
        const size = @sizeOf(T);
        const offset = index * size;
        if (offset + size > self.data.len) {
            return DecodeError.IndexOutOfBounds;
        }
        return std.mem.readInt(T, self.data[offset..][0..size], .little);
    }

    /// Read a single float value at the given index (via bitcast from unsigned int)
    fn readFloatValue(self: Self, comptime T: type, comptime U: type, index: usize) DecodeError!T {
        const bits = try self.readIntValue(U, index);
        return @bitCast(bits);
    }

    /// Read all values into an allocated slice using a read function
    fn readAllValues(self: Self, comptime T: type, allocator: std.mem.Allocator, readFn: fn (Self, usize) DecodeError!T) ![]T {
        const count = self.valueCount(T);
        const result = try allocator.alloc(T, count);
        errdefer allocator.free(result);

        for (0..count) |i| {
            result[i] = readFn(self, i) catch return error.IndexOutOfBounds;
        }
        return result;
    }

    // ========================================================================
    // Int64 decoding
    // ========================================================================

    pub fn int64Count(self: Self) usize {
        return self.valueCount(i64);
    }

    pub fn readInt64(self: Self, index: usize) DecodeError!i64 {
        return self.readIntValue(i64, index);
    }

    pub fn readAllInt64(self: Self, allocator: std.mem.Allocator) ![]i64 {
        return self.readAllValues(i64, allocator, readInt64);
    }

    // ========================================================================
    // UInt64 decoding
    // ========================================================================

    pub fn uint64Count(self: Self) usize {
        return self.valueCount(u64);
    }

    pub fn readUint64(self: Self, index: usize) DecodeError!u64 {
        return self.readIntValue(u64, index);
    }

    // ========================================================================
    // Float64 decoding
    // ========================================================================

    pub fn float64Count(self: Self) usize {
        return self.valueCount(f64);
    }

    pub fn readFloat64(self: Self, index: usize) DecodeError!f64 {
        return self.readFloatValue(f64, u64, index);
    }

    pub fn readAllFloat64(self: Self, allocator: std.mem.Allocator) ![]f64 {
        return self.readAllValues(f64, allocator, readFloat64);
    }

    // ========================================================================
    // Int32 decoding
    // ========================================================================

    pub fn int32Count(self: Self) usize {
        return self.valueCount(i32);
    }

    pub fn readInt32(self: Self, index: usize) DecodeError!i32 {
        return self.readIntValue(i32, index);
    }

    pub fn readAllInt32(self: Self, allocator: std.mem.Allocator) ![]i32 {
        return self.readAllValues(i32, allocator, readInt32);
    }

    // ========================================================================
    // Float32 decoding
    // ========================================================================

    pub fn float32Count(self: Self) usize {
        return self.valueCount(f32);
    }

    pub fn readFloat32(self: Self, index: usize) DecodeError!f32 {
        return self.readFloatValue(f32, u32, index);
    }

    pub fn readAllFloat32(self: Self, allocator: std.mem.Allocator) ![]f32 {
        return self.readAllValues(f32, allocator, readFloat32);
    }

    // ========================================================================
    // Boolean decoding (packed bits)
    // ========================================================================

    /// Get the number of boolean values in the buffer (8 per byte).
    pub fn boolCount(self: Self) usize {
        return self.data.len * 8;
    }

    /// Read a single boolean value at the given index.
    pub fn readBool(self: Self, index: usize) DecodeError!bool {
        const byte_index = index / 8;
        const bit_index: u3 = @intCast(index % 8);

        if (byte_index >= self.data.len) {
            return DecodeError.IndexOutOfBounds;
        }

        return (self.data[byte_index] >> bit_index) & 1 == 1;
    }

    /// Read all boolean values into an allocated slice.
    /// Note: row_count must be provided since booleans are packed (8 per byte).
    pub fn readAllBool(self: Self, allocator: std.mem.Allocator, row_count: usize) ![]bool {
        const result = try allocator.alloc(bool, row_count);
        errdefer allocator.free(result);

        for (0..row_count) |i| {
            result[i] = self.readBool(i) catch return error.IndexOutOfBounds;
        }
        return result;
    }

    // ========================================================================
    // String decoding (variable-length)
    // ========================================================================

    /// Read all strings from Lance string column buffers.
    /// Lance stores strings with:
    ///   - offsets buffer: int32 or int64 array marking END positions
    ///   - data buffer: concatenated UTF-8 bytes
    /// This is a static function that doesn't need a PlainDecoder instance.
    pub fn readAllStrings(
        offsets_buffer: []const u8,
        data_buffer: []const u8,
        allocator: std.mem.Allocator,
    ) ![][]const u8 {
        if (offsets_buffer.len == 0) return &[_][]const u8{};

        // Determine offset size (4 or 8 bytes) based on buffer length
        // Row count = offsets_buffer.len / offset_size
        const offset_size: usize = if (offsets_buffer.len % 8 == 0 and offsets_buffer.len / 8 > 0)
            8
        else if (offsets_buffer.len % 4 == 0)
            4
        else
            return error.InvalidBufferSize;

        const row_count = offsets_buffer.len / offset_size;
        var strings = try allocator.alloc([]const u8, row_count);
        errdefer allocator.free(strings);

        var str_start: usize = 0;
        for (0..row_count) |i| {
            // Each offset marks the END position of that string
            const str_end: usize = if (offset_size == 4)
                std.mem.readInt(u32, offsets_buffer[i * 4 ..][0..4], .little)
            else
                @intCast(std.mem.readInt(u64, offsets_buffer[i * 8 ..][0..8], .little));

            // Extract string slice from data buffer
            if (str_end > data_buffer.len or str_start > str_end) {
                return error.IndexOutOfBounds;
            }

            strings[i] = data_buffer[str_start..str_end];
            str_start = str_end;
        }

        return strings;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "read int64" {
    var data: [24]u8 = undefined;
    std.mem.writeInt(i64, data[0..8], 100, .little);
    std.mem.writeInt(i64, data[8..16], -200, .little);
    std.mem.writeInt(i64, data[16..24], 300, .little);

    const decoder = PlainDecoder.init(&data);

    try std.testing.expectEqual(@as(usize, 3), decoder.int64Count());
    try std.testing.expectEqual(@as(i64, 100), try decoder.readInt64(0));
    try std.testing.expectEqual(@as(i64, -200), try decoder.readInt64(1));
    try std.testing.expectEqual(@as(i64, 300), try decoder.readInt64(2));
}

test "read int64 out of bounds" {
    var data: [8]u8 = undefined;
    std.mem.writeInt(i64, &data, 42, .little);

    const decoder = PlainDecoder.init(&data);
    const result = decoder.readInt64(1);
    try std.testing.expectError(DecodeError.IndexOutOfBounds, result);
}

test "read all int64" {
    const allocator = std.testing.allocator;

    var data: [16]u8 = undefined;
    std.mem.writeInt(i64, data[0..8], 10, .little);
    std.mem.writeInt(i64, data[8..16], 20, .little);

    const decoder = PlainDecoder.init(&data);
    const values = try decoder.readAllInt64(allocator);
    defer allocator.free(values);

    try std.testing.expectEqual(@as(usize, 2), values.len);
    try std.testing.expectEqual(@as(i64, 10), values[0]);
    try std.testing.expectEqual(@as(i64, 20), values[1]);
}

test "read float64" {
    var data: [16]u8 = undefined;
    const f1: f64 = 3.14159;
    const f2: f64 = -2.71828;
    std.mem.writeInt(u64, data[0..8], @bitCast(f1), .little);
    std.mem.writeInt(u64, data[8..16], @bitCast(f2), .little);

    const decoder = PlainDecoder.init(&data);

    try std.testing.expectEqual(@as(usize, 2), decoder.float64Count());
    try std.testing.expectApproxEqRel(f1, try decoder.readFloat64(0), 1e-10);
    try std.testing.expectApproxEqRel(f2, try decoder.readFloat64(1), 1e-10);
}

test "read int32" {
    var data: [8]u8 = undefined;
    std.mem.writeInt(i32, data[0..4], 1000, .little);
    std.mem.writeInt(i32, data[4..8], -2000, .little);

    const decoder = PlainDecoder.init(&data);

    try std.testing.expectEqual(@as(usize, 2), decoder.int32Count());
    try std.testing.expectEqual(@as(i32, 1000), try decoder.readInt32(0));
    try std.testing.expectEqual(@as(i32, -2000), try decoder.readInt32(1));
}

test "read bool" {
    // 0b10110001 = bits: 1,0,0,0,1,1,0,1 (LSB first)
    const data = [_]u8{0b10110001};
    const decoder = PlainDecoder.init(&data);

    try std.testing.expectEqual(@as(usize, 8), decoder.boolCount());
    try std.testing.expectEqual(true, try decoder.readBool(0));
    try std.testing.expectEqual(false, try decoder.readBool(1));
    try std.testing.expectEqual(false, try decoder.readBool(2));
    try std.testing.expectEqual(false, try decoder.readBool(3));
    try std.testing.expectEqual(true, try decoder.readBool(4));
    try std.testing.expectEqual(true, try decoder.readBool(5));
    try std.testing.expectEqual(false, try decoder.readBool(6));
    try std.testing.expectEqual(true, try decoder.readBool(7));
}
