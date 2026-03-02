//! ORC Run-Length Encoding (RLE) Decoder
//!
//! ORC uses a different RLE format than Parquet. There are two versions:
//!
//! RLE V1 (older ORC files):
//! - Control byte determines run type
//! - Negative control: literal run of |control| values
//! - Non-negative control: repeat value (control + 3) times
//!
//! RLE V2 (modern ORC - WriterVersion >= HIVE_8732):
//! - Header byte high 2 bits determine encoding:
//!   0b00: SHORT_REPEAT - small repeated runs
//!   0b01: DIRECT - raw values with bit-packing
//!   0b10: PATCHED_BASE - base value with patches
//!   0b11: DELTA - delta encoding
//!
//! Reference: https://orc.apache.org/specification/ORCv1/

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const RleError = error{
    InvalidHeader,
    UnexpectedEof,
    InvalidEncoding,
    OutOfMemory,
};

/// RLE version
pub const RleVersion = enum {
    v1,
    v2,
};

/// ORC RLE V2 encoding types
pub const RleV2Encoding = enum(u2) {
    short_repeat = 0,
    direct = 1,
    patched_base = 2,
    delta = 3,
};

/// ORC RLE Decoder
pub const OrcRleDecoder = struct {
    data: []const u8,
    pos: usize,
    version: RleVersion,
    is_signed: bool,

    const Self = @This();

    pub fn init(data: []const u8, version: RleVersion, is_signed: bool) Self {
        return .{
            .data = data,
            .pos = 0,
            .version = version,
            .is_signed = is_signed,
        };
    }

    /// Decode signed integers (for DATA streams)
    pub fn decodeIntegers(self: *Self, count: usize, allocator: Allocator) RleError![]i64 {
        var values = allocator.alloc(i64, count) catch return RleError.OutOfMemory;
        errdefer allocator.free(values);

        var idx: usize = 0;
        while (idx < count) {
            if (self.pos >= self.data.len) {
                return RleError.UnexpectedEof;
            }

            switch (self.version) {
                .v1 => idx += try self.decodeV1Run(values[idx..], count - idx),
                .v2 => idx += try self.decodeV2Run(values[idx..], count - idx),
            }
        }

        return values;
    }

    /// Decode unsigned integers (for LENGTH streams)
    pub fn decodeUnsigned(self: *Self, count: usize, allocator: Allocator) RleError![]u64 {
        const signed = try self.decodeIntegers(count, allocator);
        defer allocator.free(signed);

        // Allocate new array for unsigned values to avoid pointer aliasing issues
        const unsigned = allocator.alloc(u64, count) catch return RleError.OutOfMemory;
        for (signed, 0..) |val, i| {
            // Reinterpret each value as unsigned
            unsigned[i] = @bitCast(val);
        }
        return unsigned;
    }

    /// Decode booleans (for PRESENT streams)
    /// Booleans are packed 8 per byte, RLE-encoded
    pub fn decodeBooleans(self: *Self, count: usize, allocator: Allocator) RleError![]bool {
        var values = allocator.alloc(bool, count) catch return RleError.OutOfMemory;
        errdefer allocator.free(values);

        var idx: usize = 0;
        while (idx < count) {
            if (self.pos >= self.data.len) {
                // Remaining values are false (common for non-null columns)
                for (values[idx..]) |*v| {
                    v.* = false;
                }
                break;
            }

            // RLE for booleans: control byte + byte values
            // Each byte contains 8 boolean values
            const control: i8 = @bitCast(self.data[self.pos]);
            self.pos += 1;

            if (control < 0) {
                // Literal run: |control| bytes follow
                const literal_count: usize = @intCast(-@as(i32, control));
                for (0..literal_count) |_| {
                    if (self.pos >= self.data.len or idx >= count) break;
                    const byte = self.data[self.pos];
                    self.pos += 1;

                    // Unpack 8 booleans from byte
                    for (0..8) |bit| {
                        if (idx >= count) break;
                        values[idx] = (byte >> @intCast(7 - bit)) & 1 == 1;
                        idx += 1;
                    }
                }
            } else {
                // Repeated run: repeat next byte (control + 3) times
                const repeat_count: usize = @intCast(@as(i32, control) + 3);
                if (self.pos >= self.data.len) return RleError.UnexpectedEof;
                const byte = self.data[self.pos];
                self.pos += 1;

                for (0..repeat_count) |_| {
                    for (0..8) |bit| {
                        if (idx >= count) break;
                        values[idx] = (byte >> @intCast(7 - bit)) & 1 == 1;
                        idx += 1;
                    }
                }
            }
        }

        return values;
    }

    // ========================================================================
    // RLE V1 Decoder
    // ========================================================================

    fn decodeV1Run(self: *Self, output: []i64, max_count: usize) RleError!usize {
        if (self.pos >= self.data.len) return RleError.UnexpectedEof;

        const control: i8 = @bitCast(self.data[self.pos]);
        self.pos += 1;

        if (control < 0) {
            // Literal run: |control| values follow
            const literal_count = @min(@as(usize, @intCast(-@as(i32, control))), max_count);

            for (0..literal_count) |i| {
                output[i] = try self.readSignedVInt();
            }
            return literal_count;
        } else {
            // Repeated run: repeat value (control + 3) times
            const repeat_count = @min(@as(usize, @intCast(@as(i32, control) + 3)), max_count);
            const value = try self.readSignedVInt();

            for (0..repeat_count) |i| {
                output[i] = value;
            }
            return repeat_count;
        }
    }

    // ========================================================================
    // RLE V2 Decoder
    // ========================================================================

    fn decodeV2Run(self: *Self, output: []i64, max_count: usize) RleError!usize {
        if (self.pos >= self.data.len) return RleError.UnexpectedEof;

        const header = self.data[self.pos];
        const encoding: RleV2Encoding = @enumFromInt((header >> 6) & 0x03);

        return switch (encoding) {
            .short_repeat => self.decodeShortRepeat(output, max_count, header),
            .direct => self.decodeDirect(output, max_count, header),
            .patched_base => self.decodePatchedBase(output, max_count, header),
            .delta => self.decodeDelta(output, max_count, header),
        };
    }

    /// SHORT_REPEAT encoding: small runs of repeated values
    /// Header: 0b00WWWnnn where W=width-1, n=count-3
    fn decodeShortRepeat(self: *Self, output: []i64, max_count: usize, header: u8) RleError!usize {
        self.pos += 1; // consume header

        const width = ((header >> 3) & 0x07) + 1; // 1-8 bytes
        const count = @min(@as(usize, (header & 0x07) + 3), max_count); // 3-10 values

        if (self.pos + width > self.data.len) return RleError.UnexpectedEof;

        // Read value (big-endian)
        var value: i64 = 0;
        for (0..width) |_| {
            value = (value << 8) | @as(i64, self.data[self.pos]);
            self.pos += 1;
        }

        // Sign-extend if needed
        if (self.is_signed) {
            value = zigzagDecode(@bitCast(value));
        }

        for (0..count) |i| {
            output[i] = value;
        }

        return count;
    }

    /// DIRECT encoding: raw values with bit-packing
    /// Header: 0b01WWWWWB + LEN where W=width code (5 bits), B=high bit of count
    fn decodeDirect(self: *Self, output: []i64, max_count: usize, header: u8) RleError!usize {
        self.pos += 1; // consume header

        // Width code is in bits 5-1 (5 bits)
        const width_code = (header >> 1) & 0x1F;
        const bit_width = decodeBitWidth(width_code);

        // Length is 9 bits: high bit from header bit 0, low 8 bits from next byte
        if (self.pos >= self.data.len) return RleError.UnexpectedEof;
        const length_byte = self.data[self.pos];
        self.pos += 1;
        const high_bit: usize = header & 1;
        const count = @min((high_bit << 8) | @as(usize, length_byte) + 1, max_count);

        // Read values with bit-packing
        var bit_pos: usize = 0;
        for (0..count) |i| {
            var value: u64 = 0;
            var bits_read: usize = 0;

            while (bits_read < bit_width) {
                const byte_idx = self.pos + (bit_pos / 8);
                if (byte_idx >= self.data.len) return RleError.UnexpectedEof;

                const bit_offset: usize = bit_pos % 8;
                const bits_in_byte = @min(8 - bit_offset, bit_width - bits_read);

                const byte = self.data[byte_idx];
                const shift_amt: u3 = @intCast(8 - bit_offset - bits_in_byte);
                const mask = (@as(u8, 1) << @as(u3, @intCast(bits_in_byte))) - 1;
                const extracted = (byte >> shift_amt) & mask;

                value = (value << @as(u6, @intCast(bits_in_byte))) | @as(u64, extracted);
                bits_read += bits_in_byte;
                bit_pos += bits_in_byte;
            }

            if (self.is_signed) {
                output[i] = zigzagDecode(value);
            } else {
                output[i] = @bitCast(value);
            }
        }

        // Advance to next byte boundary
        self.pos += (bit_pos + 7) / 8;

        return count;
    }

    /// PATCHED_BASE encoding: base value + patches
    /// Complex encoding for data with outliers - not commonly used
    fn decodePatchedBase(self: *Self, output: []i64, max_count: usize, header: u8) RleError!usize {
        // PATCHED_BASE format:
        // Header: 0b10WWWWW (W = width code for base values)
        // Followed by: base width, patch width, patch gap width, patch list length
        // Then: base values, patch list

        // For now, skip this block and return error since it's rarely used
        // and complex to implement correctly
        _ = self;
        _ = output;
        _ = max_count;
        _ = header;
        return RleError.InvalidEncoding;
    }

    /// DELTA encoding: base + deltas
    /// Header: 0b11WWWWWB + LEN where W=width code (5 bits), B=high bit of count
    fn decodeDelta(self: *Self, output: []i64, max_count: usize, header: u8) RleError!usize {
        self.pos += 1; // consume header

        // Width code is in bits 5-1 (5 bits)
        const width_code = (header >> 1) & 0x1F;
        // DELTA encoding: width_code=0 means constant delta (bit_width=0)
        // This is different from DIRECT where decodeBitWidth(0)=1
        const bit_width = if (width_code == 0) 0 else decodeBitWidth(width_code);

        // Length is 9 bits: high bit from header bit 0, low 8 bits from next byte
        if (self.pos >= self.data.len) return RleError.UnexpectedEof;
        const length_byte = self.data[self.pos];
        self.pos += 1;
        const high_bit: usize = header & 1;
        const count = @min((high_bit << 8) | @as(usize, length_byte) + 1, max_count);

        if (count == 0) return 0;

        // Read base value (signed varint)
        const base = try self.readSignedVInt();
        output[0] = base;

        if (count == 1) return 1;

        // Read delta base (signed varint)
        const delta_base = try self.readSignedVInt();

        // For constant delta, all deltas are delta_base
        if (bit_width == 0) {
            var current = base;
            for (1..count) |i| {
                current += delta_base;
                output[i] = current;
            }
            return count;
        }

        // Read packed deltas
        var current = base + delta_base;
        output[1] = current;

        var bit_pos: usize = 0;
        for (2..count) |i| {
            var delta: u64 = 0;
            var bits_read: usize = 0;

            while (bits_read < bit_width) {
                const byte_idx = self.pos + (bit_pos / 8);
                if (byte_idx >= self.data.len) return RleError.UnexpectedEof;

                const bit_offset: usize = bit_pos % 8;
                const bits_in_byte = @min(8 - bit_offset, bit_width - bits_read);

                const byte = self.data[byte_idx];
                const shift_amt: u3 = @intCast(8 - bit_offset - bits_in_byte);
                const mask = (@as(u8, 1) << @as(u3, @intCast(bits_in_byte))) - 1;
                const extracted = (byte >> shift_amt) & mask;

                delta = (delta << @as(u6, @intCast(bits_in_byte))) | @as(u64, extracted);
                bits_read += bits_in_byte;
                bit_pos += bits_in_byte;
            }

            current += zigzagDecode(delta);
            output[i] = current;
        }

        self.pos += (bit_pos + 7) / 8;
        return count;
    }

    // ========================================================================
    // Helper functions
    // ========================================================================

    /// Read signed varint
    fn readSignedVInt(self: *Self) RleError!i64 {
        const unsigned = try self.readVInt();
        return zigzagDecode(unsigned);
    }

    /// Read unsigned varint
    fn readVInt(self: *Self) RleError!u64 {
        var result: u64 = 0;
        var shift: u6 = 0;

        while (self.pos < self.data.len) {
            const byte = self.data[self.pos];
            self.pos += 1;

            result |= @as(u64, byte & 0x7F) << shift;

            if (byte & 0x80 == 0) {
                return result;
            }

            shift +|= 7;
            if (shift > 63) return RleError.InvalidEncoding;
        }

        return RleError.UnexpectedEof;
    }
};

/// Zigzag decode: (n >> 1) ^ -(n & 1)
fn zigzagDecode(n: u64) i64 {
    const signed: i64 = @bitCast(n);
    return (signed >> 1) ^ (-(signed & 1));
}

/// Map RLE V2 width code to actual bit width
fn decodeBitWidth(code: u8) usize {
    return switch (code) {
        0 => 1,
        1 => 2,
        2 => 3,
        3 => 4,
        4 => 5,
        5 => 6,
        6 => 7,
        7 => 8,
        8 => 9,
        9 => 10,
        10 => 11,
        11 => 12,
        12 => 13,
        13 => 14,
        14 => 15,
        15 => 16,
        16 => 17,
        17 => 18,
        18 => 19,
        19 => 20,
        20 => 21,
        21 => 22,
        22 => 23,
        23 => 24,
        24 => 26,
        25 => 28,
        26 => 30,
        27 => 32,
        28 => 40,
        29 => 48,
        30 => 56,
        31 => 64,
        else => 0,
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "zigzag decode" {
    try testing.expectEqual(@as(i64, 0), zigzagDecode(0));
    try testing.expectEqual(@as(i64, -1), zigzagDecode(1));
    try testing.expectEqual(@as(i64, 1), zigzagDecode(2));
    try testing.expectEqual(@as(i64, -2), zigzagDecode(3));
    try testing.expectEqual(@as(i64, 2), zigzagDecode(4));
}

test "bit width decode" {
    try testing.expectEqual(@as(usize, 1), decodeBitWidth(0));
    try testing.expectEqual(@as(usize, 8), decodeBitWidth(7));
    try testing.expectEqual(@as(usize, 16), decodeBitWidth(15));
    try testing.expectEqual(@as(usize, 64), decodeBitWidth(31));
}
