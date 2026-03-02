//! Protobuf wire format encoder.
//!
//! Implements the protobuf binary wire format for encoding messages.
//! This is the inverse of decoder.zig - used for writing Lance manifests and metadata.
//!
//! Wire types:
//! - 0 (Varint): int32, int64, uint32, uint64, sint32, sint64, bool, enum
//! - 1 (64-bit): fixed64, sfixed64, double
//! - 2 (Length-delimited): string, bytes, embedded messages, packed repeated fields
//! - 5 (32-bit): fixed32, sfixed32, float

const std = @import("std");

/// Protobuf wire types (same as decoder)
pub const WireType = enum(u3) {
    varint = 0,
    fixed64 = 1,
    length_delimited = 2,
    start_group = 3, // deprecated
    end_group = 4, // deprecated
    fixed32 = 5,
};

/// Errors that can occur during protobuf encoding
pub const EncodeError = error{
    OutOfMemory,
    BufferTooSmall,
};

/// Protobuf wire format encoder.
///
/// Builds protobuf-encoded messages by appending to an internal buffer.
/// Supports both managed (ArrayList) and unmanaged (ArrayListUnmanaged) modes.
pub const ProtoEncoder = struct {
    buffer: std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,

    const Self = @This();

    /// Create a new encoder with the given allocator.
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .buffer = std.ArrayListUnmanaged(u8){},
            .allocator = allocator,
        };
    }

    /// Create a new encoder with pre-allocated capacity.
    pub fn initCapacity(allocator: std.mem.Allocator, capacity: usize) !Self {
        var buffer = std.ArrayListUnmanaged(u8){};
        try buffer.ensureTotalCapacity(allocator, capacity);
        return Self{
            .buffer = buffer,
            .allocator = allocator,
        };
    }

    /// Free the encoder's buffer.
    pub fn deinit(self: *Self) void {
        self.buffer.deinit(self.allocator);
    }

    /// Clear the buffer while retaining capacity for reuse.
    pub fn reset(self: *Self) void {
        self.buffer.clearRetainingCapacity();
    }

    /// Get the encoded bytes (does not transfer ownership).
    pub fn getBytes(self: Self) []const u8 {
        return self.buffer.items;
    }

    /// Get owned copy of encoded bytes (caller must free).
    pub fn toOwnedSlice(self: *Self) ![]u8 {
        return self.buffer.toOwnedSlice(self.allocator);
    }

    /// Current size of encoded data.
    pub fn len(self: Self) usize {
        return self.buffer.items.len;
    }

    // ========================================================================
    // Low-level encoding primitives
    // ========================================================================

    /// Write a varint-encoded unsigned integer.
    ///
    /// Varints use 7 bits per byte with MSB indicating continuation.
    pub fn writeVarint(self: *Self, value: u64) !void {
        var v = value;
        while (v >= 0x80) {
            try self.buffer.append(self.allocator, @as(u8, @truncate(v)) | 0x80);
            v >>= 7;
        }
        try self.buffer.append(self.allocator, @truncate(v));
    }

    /// Write a signed varint using ZigZag encoding.
    ///
    /// ZigZag maps signed integers to unsigned: 0->0, -1->1, 1->2, -2->3, etc.
    pub fn writeSignedVarint(self: *Self, value: i64) !void {
        // ZigZag encode: (n << 1) ^ (n >> 63)
        const unsigned: u64 = @bitCast((value << 1) ^ (value >> 63));
        try self.writeVarint(unsigned);
    }

    /// Write a field tag (field number + wire type).
    pub fn writeTag(self: *Self, field_number: u32, wire_type: WireType) !void {
        const tag = (@as(u64, field_number) << 3) | @intFromEnum(wire_type);
        try self.writeVarint(tag);
    }

    /// Write raw bytes without length prefix.
    pub fn writeRawBytes(self: *Self, data: []const u8) !void {
        try self.buffer.appendSlice(self.allocator, data);
    }

    /// Write a fixed 64-bit value (little-endian).
    pub fn writeFixed64Raw(self: *Self, value: u64) !void {
        var bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &bytes, value, .little);
        try self.buffer.appendSlice(self.allocator, &bytes);
    }

    /// Write a fixed 32-bit value (little-endian).
    pub fn writeFixed32Raw(self: *Self, value: u32) !void {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, value, .little);
        try self.buffer.appendSlice(self.allocator, &bytes);
    }

    // ========================================================================
    // Field-level encoding (tag + value)
    // ========================================================================

    /// Write a varint field (wire type 0).
    pub fn writeVarintField(self: *Self, field_number: u32, value: u64) !void {
        try self.writeTag(field_number, .varint);
        try self.writeVarint(value);
    }

    /// Write a signed varint field using ZigZag encoding.
    pub fn writeSignedVarintField(self: *Self, field_number: u32, value: i64) !void {
        try self.writeTag(field_number, .varint);
        try self.writeSignedVarint(value);
    }

    /// Write a bool field (wire type 0, value 0 or 1).
    pub fn writeBoolField(self: *Self, field_number: u32, value: bool) !void {
        try self.writeVarintField(field_number, if (value) 1 else 0);
    }

    /// Write a fixed64 field (wire type 1).
    pub fn writeFixed64Field(self: *Self, field_number: u32, value: u64) !void {
        try self.writeTag(field_number, .fixed64);
        try self.writeFixed64Raw(value);
    }

    /// Write a double field (wire type 1).
    pub fn writeDoubleField(self: *Self, field_number: u32, value: f64) !void {
        try self.writeFixed64Field(field_number, @bitCast(value));
    }

    /// Write a fixed32 field (wire type 5).
    pub fn writeFixed32Field(self: *Self, field_number: u32, value: u32) !void {
        try self.writeTag(field_number, .fixed32);
        try self.writeFixed32Raw(value);
    }

    /// Write a float field (wire type 5).
    pub fn writeFloatField(self: *Self, field_number: u32, value: f32) !void {
        try self.writeFixed32Field(field_number, @bitCast(value));
    }

    /// Write a length-delimited field (wire type 2).
    /// Used for strings, bytes, and embedded messages.
    pub fn writeBytesField(self: *Self, field_number: u32, data: []const u8) !void {
        try self.writeTag(field_number, .length_delimited);
        try self.writeVarint(data.len);
        try self.buffer.appendSlice(self.allocator, data);
    }

    /// Write a string field (same as bytes, wire type 2).
    pub fn writeStringField(self: *Self, field_number: u32, str: []const u8) !void {
        try self.writeBytesField(field_number, str);
    }

    /// Write an embedded message field (wire type 2).
    /// The message bytes should already be encoded.
    pub fn writeMessageField(self: *Self, field_number: u32, message_bytes: []const u8) !void {
        try self.writeBytesField(field_number, message_bytes);
    }

    // ========================================================================
    // Packed repeated fields
    // ========================================================================

    /// Write a packed repeated varint field.
    pub fn writePackedVarintsField(self: *Self, field_number: u32, values: []const u64) !void {
        // First, encode all varints to a temporary buffer
        var temp = ProtoEncoder.init(self.allocator);
        defer temp.deinit();

        for (values) |v| {
            try temp.writeVarint(v);
        }

        // Write as length-delimited
        try self.writeBytesField(field_number, temp.getBytes());
    }

    /// Write a packed repeated fixed64 field.
    pub fn writePackedFixed64Field(self: *Self, field_number: u32, values: []const u64) !void {
        try self.writeTag(field_number, .length_delimited);
        try self.writeVarint(values.len * 8);
        for (values) |v| {
            try self.writeFixed64Raw(v);
        }
    }

    /// Write a packed repeated fixed32 field.
    pub fn writePackedFixed32Field(self: *Self, field_number: u32, values: []const u32) !void {
        try self.writeTag(field_number, .length_delimited);
        try self.writeVarint(values.len * 4);
        for (values) |v| {
            try self.writeFixed32Raw(v);
        }
    }

    /// Write a packed repeated float field.
    pub fn writePackedFloatsField(self: *Self, field_number: u32, values: []const f32) !void {
        try self.writeTag(field_number, .length_delimited);
        try self.writeVarint(values.len * 4);
        for (values) |v| {
            try self.writeFixed32Raw(@bitCast(v));
        }
    }

    /// Write a packed repeated double field.
    pub fn writePackedDoublesField(self: *Self, field_number: u32, values: []const f64) !void {
        try self.writeTag(field_number, .length_delimited);
        try self.writeVarint(values.len * 8);
        for (values) |v| {
            try self.writeFixed64Raw(@bitCast(v));
        }
    }
};

/// Calculate varint size in bytes for a given value.
pub fn varintSize(value: u64) usize {
    if (value == 0) return 1;
    var v = value;
    var size: usize = 0;
    while (v > 0) {
        size += 1;
        v >>= 7;
    }
    return size;
}

/// Calculate ZigZag-encoded varint size for a signed value.
pub fn signedVarintSize(value: i64) usize {
    const unsigned: u64 = @bitCast((value << 1) ^ (value >> 63));
    return varintSize(unsigned);
}

// ============================================================================
// Tests
// ============================================================================

test "varint single byte" {
    const allocator = std.testing.allocator;

    var encoder = ProtoEncoder.init(allocator);
    defer encoder.deinit();

    try encoder.writeVarint(1);
    try std.testing.expectEqual(@as(usize, 1), encoder.len());
    try std.testing.expectEqual(@as(u8, 0x01), encoder.getBytes()[0]);
}

test "varint 150" {
    const allocator = std.testing.allocator;

    var encoder = ProtoEncoder.init(allocator);
    defer encoder.deinit();

    // 150 = 0x96 0x01 (same as decoder test)
    try encoder.writeVarint(150);
    try std.testing.expectEqual(@as(usize, 2), encoder.len());
    try std.testing.expectEqual(@as(u8, 0x96), encoder.getBytes()[0]);
    try std.testing.expectEqual(@as(u8, 0x01), encoder.getBytes()[1]);
}

test "varint 300" {
    const allocator = std.testing.allocator;

    var encoder = ProtoEncoder.init(allocator);
    defer encoder.deinit();

    // 300 = 0xAC 0x02 (same as decoder test)
    try encoder.writeVarint(300);
    try std.testing.expectEqual(@as(usize, 2), encoder.len());
    try std.testing.expectEqual(@as(u8, 0xAC), encoder.getBytes()[0]);
    try std.testing.expectEqual(@as(u8, 0x02), encoder.getBytes()[1]);
}

test "signed varint positive" {
    const allocator = std.testing.allocator;

    var encoder = ProtoEncoder.init(allocator);
    defer encoder.deinit();

    // ZigZag: 1 -> 2 -> 0x02
    try encoder.writeSignedVarint(1);
    try std.testing.expectEqual(@as(u8, 0x02), encoder.getBytes()[0]);
}

test "signed varint negative" {
    const allocator = std.testing.allocator;

    var encoder = ProtoEncoder.init(allocator);
    defer encoder.deinit();

    // ZigZag: -1 -> 1 -> 0x01
    try encoder.writeSignedVarint(-1);
    try std.testing.expectEqual(@as(u8, 0x01), encoder.getBytes()[0]);
}

test "field tag" {
    const allocator = std.testing.allocator;

    var encoder = ProtoEncoder.init(allocator);
    defer encoder.deinit();

    // Field 1, wire type 0 (varint): (1 << 3) | 0 = 0x08
    try encoder.writeTag(1, .varint);
    try std.testing.expectEqual(@as(u8, 0x08), encoder.getBytes()[0]);

    encoder.reset();

    // Field 2, wire type 2 (length-delimited): (2 << 3) | 2 = 0x12
    try encoder.writeTag(2, .length_delimited);
    try std.testing.expectEqual(@as(u8, 0x12), encoder.getBytes()[0]);
}

test "string field" {
    const allocator = std.testing.allocator;

    var encoder = ProtoEncoder.init(allocator);
    defer encoder.deinit();

    // Field 2, string "testing"
    try encoder.writeStringField(2, "testing");

    const bytes = encoder.getBytes();
    // Tag: (2 << 3) | 2 = 0x12
    try std.testing.expectEqual(@as(u8, 0x12), bytes[0]);
    // Length: 7
    try std.testing.expectEqual(@as(u8, 7), bytes[1]);
    // String data
    try std.testing.expectEqualSlices(u8, "testing", bytes[2..9]);
}

test "fixed64 field" {
    const allocator = std.testing.allocator;

    var encoder = ProtoEncoder.init(allocator);
    defer encoder.deinit();

    try encoder.writeFixed64Field(1, 0x123456789ABCDEF0);

    const bytes = encoder.getBytes();
    // Tag: (1 << 3) | 1 = 0x09
    try std.testing.expectEqual(@as(u8, 0x09), bytes[0]);
    // Value in little-endian
    try std.testing.expectEqual(@as(u64, 0x123456789ABCDEF0), std.mem.readInt(u64, bytes[1..9], .little));
}

test "embedded message" {
    const allocator = std.testing.allocator;

    // Create inner message
    var inner = ProtoEncoder.init(allocator);
    defer inner.deinit();
    try inner.writeVarintField(1, 42);

    // Create outer message with embedded inner
    var outer = ProtoEncoder.init(allocator);
    defer outer.deinit();
    try outer.writeMessageField(2, inner.getBytes());

    const bytes = outer.getBytes();
    // Outer tag: (2 << 3) | 2 = 0x12
    try std.testing.expectEqual(@as(u8, 0x12), bytes[0]);
    // Length of inner: 2 bytes (tag + value)
    try std.testing.expectEqual(@as(u8, 2), bytes[1]);
    // Inner tag: (1 << 3) | 0 = 0x08
    try std.testing.expectEqual(@as(u8, 0x08), bytes[2]);
    // Inner value: 42
    try std.testing.expectEqual(@as(u8, 42), bytes[3]);
}

test "varint size calculation" {
    try std.testing.expectEqual(@as(usize, 1), varintSize(0));
    try std.testing.expectEqual(@as(usize, 1), varintSize(127));
    try std.testing.expectEqual(@as(usize, 2), varintSize(128));
    try std.testing.expectEqual(@as(usize, 2), varintSize(300));
    try std.testing.expectEqual(@as(usize, 5), varintSize(0xFFFFFFFF));
    try std.testing.expectEqual(@as(usize, 10), varintSize(0xFFFFFFFFFFFFFFFF));
}
