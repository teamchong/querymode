//! Lance file writer - encodes columnar data to Lance format.
//!
//! This module provides functionality to write Lance files in the browser
//! via WASM, supporting INSERT operations with OPFS storage.
//!
//! ## Lance File Layout
//! ```
//! +------------------+
//! | Column Data      |  <- Raw column bytes (page by page)
//! +------------------+
//! | Column Metadata  |  <- Protobuf-encoded column info
//! +------------------+
//! | Column Offsets   |  <- Array of u64 offsets to each column's metadata
//! +------------------+
//! | Global Buffers   |  <- Shared buffers (if any)
//! +------------------+
//! | Footer (40 bytes)|  <- Version, offsets, magic "LANC"
//! +------------------+
//! ```

const std = @import("std");

/// Data types supported for writing
pub const DataType = enum {
    int32,
    int64,
    float32,
    float64,
    string,
    bool,
    vector_f32,
};

/// Column schema definition
pub const ColumnSchema = struct {
    name: []const u8,
    data_type: DataType,
    nullable: bool = true,
    vector_dim: u32 = 0, // For vector types
};

/// A batch of column data to write
pub const ColumnBatch = struct {
    /// Column index
    column_index: u32,
    /// Raw data bytes (already encoded)
    data: []const u8,
    /// Number of rows in this batch
    row_count: u32,
    /// For strings: offsets buffer
    offsets: ?[]const u8 = null,
};

/// Plain encoder - encodes values to bytes
pub const PlainEncoder = struct {
    buffer: std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .buffer = std.ArrayListUnmanaged(u8){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.buffer.deinit(self.allocator);
    }

    pub fn reset(self: *Self) void {
        self.buffer.clearRetainingCapacity();
    }

    pub fn getBytes(self: Self) []const u8 {
        return self.buffer.items;
    }

    // ========================================================================
    // Int64 encoding
    // ========================================================================

    pub fn writeInt64(self: *Self, value: i64) !void {
        var bytes: [8]u8 = undefined;
        std.mem.writeInt(i64, &bytes, value, .little);
        try self.buffer.appendSlice(self.allocator, &bytes);
    }

    pub fn writeInt64Slice(self: *Self, values: []const i64) !void {
        for (values) |v| {
            try self.writeInt64(v);
        }
    }

    // ========================================================================
    // Int32 encoding
    // ========================================================================

    pub fn writeInt32(self: *Self, value: i32) !void {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(i32, &bytes, value, .little);
        try self.buffer.appendSlice(self.allocator, &bytes);
    }

    pub fn writeInt32Slice(self: *Self, values: []const i32) !void {
        for (values) |v| {
            try self.writeInt32(v);
        }
    }

    // ========================================================================
    // Float64 encoding
    // ========================================================================

    pub fn writeFloat64(self: *Self, value: f64) !void {
        var bytes: [8]u8 = undefined;
        const bits: u64 = @bitCast(value);
        std.mem.writeInt(u64, &bytes, bits, .little);
        try self.buffer.appendSlice(self.allocator, &bytes);
    }

    pub fn writeFloat64Slice(self: *Self, values: []const f64) !void {
        for (values) |v| {
            try self.writeFloat64(v);
        }
    }

    // ========================================================================
    // Float32 encoding
    // ========================================================================

    pub fn writeFloat32(self: *Self, value: f32) !void {
        var bytes: [4]u8 = undefined;
        const bits: u32 = @bitCast(value);
        std.mem.writeInt(u32, &bytes, bits, .little);
        try self.buffer.appendSlice(self.allocator, &bytes);
    }

    pub fn writeFloat32Slice(self: *Self, values: []const f32) !void {
        for (values) |v| {
            try self.writeFloat32(v);
        }
    }

    // ========================================================================
    // String encoding (produces offsets + data buffers)
    // ========================================================================

    pub fn writeStrings(self: *Self, values: []const []const u8, offsets_out: *std.ArrayListUnmanaged(u8), offsets_allocator: std.mem.Allocator) !void {
        var current_offset: u32 = 0;

        for (values) |str| {
            // Write string data
            try self.buffer.appendSlice(self.allocator, str);
            current_offset += @intCast(str.len);

            // Write offset (end position)
            var offset_bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &offset_bytes, current_offset, .little);
            try offsets_out.appendSlice(offsets_allocator, &offset_bytes);
        }
    }

    // ========================================================================
    // Boolean encoding (packed bits)
    // ========================================================================

    pub fn writeBools(self: *Self, values: []const bool) !void {
        const byte_count = (values.len + 7) / 8;

        var i: usize = 0;
        while (i < byte_count) : (i += 1) {
            var byte: u8 = 0;
            var bit: usize = 0;
            while (bit < 8 and i * 8 + bit < values.len) : (bit += 1) {
                if (values[i * 8 + bit]) {
                    byte |= @as(u8, 1) << @intCast(bit);
                }
            }
            try self.buffer.append(self.allocator, byte);
        }
    }

    // ========================================================================
    // Vector encoding (float32 array)
    // ========================================================================

    pub fn writeVectorF32(self: *Self, values: []const f32) !void {
        try self.writeFloat32Slice(values);
    }
};

/// Footer writer - creates the 40-byte Lance footer
pub const FooterWriter = struct {
    pub fn write(
        column_meta_start: u64,
        column_meta_offsets_start: u64,
        global_buff_offsets_start: u64,
        num_global_buffers: u32,
        num_columns: u32,
        major_version: u16,
        minor_version: u16,
    ) [40]u8 {
        var footer: [40]u8 = undefined;

        std.mem.writeInt(u64, footer[0..8], column_meta_start, .little);
        std.mem.writeInt(u64, footer[8..16], column_meta_offsets_start, .little);
        std.mem.writeInt(u64, footer[16..24], global_buff_offsets_start, .little);
        std.mem.writeInt(u32, footer[24..28], num_global_buffers, .little);
        std.mem.writeInt(u32, footer[28..32], num_columns, .little);
        std.mem.writeInt(u16, footer[32..34], major_version, .little);
        std.mem.writeInt(u16, footer[34..36], minor_version, .little);
        @memcpy(footer[36..40], "LANC");

        return footer;
    }
};

/// Protobuf encoder for column metadata
pub const ProtobufEncoder = struct {
    buffer: std.ArrayListUnmanaged(u8),
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .buffer = std.ArrayListUnmanaged(u8){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.buffer.deinit(self.allocator);
    }

    pub fn getBytes(self: Self) []const u8 {
        return self.buffer.items;
    }

    /// Write varint (variable-length integer)
    pub fn writeVarint(self: *Self, value: u64) !void {
        var v = value;
        while (v >= 0x80) {
            try self.buffer.append(self.allocator, @as(u8, @truncate(v)) | 0x80);
            v >>= 7;
        }
        try self.buffer.append(self.allocator, @truncate(v));
    }

    /// Write field tag (field number + wire type)
    pub fn writeTag(self: *Self, field_number: u32, wire_type: u3) !void {
        const tag = (@as(u64, field_number) << 3) | wire_type;
        try self.writeVarint(tag);
    }

    /// Write length-delimited bytes (wire type 2)
    pub fn writeBytes(self: *Self, field_number: u32, data: []const u8) !void {
        try self.writeTag(field_number, 2);
        try self.writeVarint(data.len);
        try self.buffer.appendSlice(self.allocator, data);
    }

    /// Write string (same as bytes)
    pub fn writeString(self: *Self, field_number: u32, str: []const u8) !void {
        try self.writeBytes(field_number, str);
    }

    /// Write varint field (wire type 0)
    pub fn writeVarintField(self: *Self, field_number: u32, value: u64) !void {
        try self.writeTag(field_number, 0);
        try self.writeVarint(value);
    }

    /// Write fixed64 field (wire type 1)
    pub fn writeFixed64(self: *Self, field_number: u32, value: u64) !void {
        try self.writeTag(field_number, 1);
        var bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &bytes, value, .little);
        try self.buffer.appendSlice(self.allocator, &bytes);
    }

    /// Write fixed32 field (wire type 5)
    pub fn writeFixed32(self: *Self, field_number: u32, value: u32) !void {
        try self.writeTag(field_number, 5);
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, value, .little);
        try self.buffer.appendSlice(self.allocator, &bytes);
    }
};

/// Column buffer info for tracking data positions
const ColumnBufferInfo = struct {
    offset: u64,
    size: u64,
    row_count: u32,
    /// For strings: secondary buffer (offsets)
    offsets_offset: u64,
    offsets_size: u64,
};

/// Lance file writer
pub const LanceWriter = struct {
    allocator: std.mem.Allocator,
    output: std.ArrayListUnmanaged(u8),
    schema: []const ColumnSchema,
    column_buffers: std.ArrayListUnmanaged(ColumnBufferInfo),
    column_metadata: std.ArrayListUnmanaged([]const u8),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, schema: []const ColumnSchema) Self {
        return Self{
            .allocator = allocator,
            .output = std.ArrayListUnmanaged(u8){},
            .schema = schema,
            .column_buffers = std.ArrayListUnmanaged(ColumnBufferInfo){},
            .column_metadata = std.ArrayListUnmanaged([]const u8){},
        };
    }

    pub fn deinit(self: *Self) void {
        self.output.deinit(self.allocator);
        self.column_buffers.deinit(self.allocator);
        for (self.column_metadata.items) |meta| {
            self.allocator.free(meta);
        }
        self.column_metadata.deinit(self.allocator);
    }

    /// Get the logical type string for a data type
    /// Note: Must match what the SQL executor expects to parse
    fn dataTypeToLogicalType(data_type: DataType) []const u8 {
        return switch (data_type) {
            .int32 => "int32",
            .int64 => "int64",
            .float32 => "float",
            .float64 => "double",  // executor checks for "double"
            .string => "string",
            .bool => "bool",
            .vector_f32 => "fixed_size_list:float32",
        };
    }

    /// Encode schema as protobuf (global buffer 0)
    fn encodeSchema(self: *Self) ![]const u8 {
        var schema_proto = ProtobufEncoder.init(self.allocator);
        defer schema_proto.deinit();

        // Encode each column as a Field message
        for (self.schema, 0..) |col, i| {
            var field_proto = ProtobufEncoder.init(self.allocator);
            defer field_proto.deinit();

            // Field 1: type (enum) - 2 = leaf (for data columns)
            try field_proto.writeVarintField(1, 2);

            // Field 2: name (string)
            try field_proto.writeString(2, col.name);

            // Field 3: id (int32) - column index
            try field_proto.writeVarintField(3, i);

            // Field 4: parent_id (int32) - -1 for root fields
            // Write as zigzag varint: -1 -> 1 (zigzag), but lance uses signed, so we need proper encoding
            // Actually, looking at the parser, it just reads as varint and bitcasts
            // For -1: as u32 it's 0xFFFFFFFF, which as varint is 5 bytes
            try field_proto.writeVarintField(4, 0xFFFFFFFF); // -1 as unsigned

            // Field 5: logical_type (string)
            try field_proto.writeString(5, dataTypeToLogicalType(col.data_type));

            // Field 6: nullable (bool)
            try field_proto.writeVarintField(6, if (col.nullable) 1 else 0);

            // Write Field as nested message in schema (field 1)
            try schema_proto.writeBytes(1, field_proto.getBytes());
        }

        return try self.allocator.dupe(u8, schema_proto.getBytes());
    }

    /// Write a batch of column data
    pub fn writeColumnBatch(self: *Self, batch: ColumnBatch) !void {
        const data_offset = self.output.items.len;
        const data_size = batch.data.len;

        // Write column data
        try self.output.appendSlice(self.allocator, batch.data);

        // Track string offsets buffer
        var offsets_offset: u64 = 0;
        var offsets_size: u64 = 0;
        if (batch.offsets) |offsets| {
            offsets_offset = self.output.items.len;
            offsets_size = offsets.len;
            try self.output.appendSlice(self.allocator, offsets);
        }

        try self.column_buffers.append(self.allocator, .{
            .offset = data_offset,
            .size = data_size,
            .row_count = batch.row_count,
            .offsets_offset = offsets_offset,
            .offsets_size = offsets_size,
        });
    }

    /// Encode a Page protobuf message
    fn encodePage(self: *Self, buf_info: ColumnBufferInfo) ![]const u8 {
        var page = ProtobufEncoder.init(self.allocator);
        defer page.deinit();

        // Field 1: buffer_offsets (packed repeated uint64)
        // For strings: [offsets_offset, data_offset], else: [data_offset]
        // Note: Lance reader expects buffer 0 = offsets, buffer 1 = data
        if (buf_info.offsets_size > 0) {
            // String column: two buffers (offsets first, then data)
            var offsets_proto = ProtobufEncoder.init(self.allocator);
            defer offsets_proto.deinit();
            try offsets_proto.writeVarint(buf_info.offsets_offset);
            try offsets_proto.writeVarint(buf_info.offset);
            try page.writeBytes(1, offsets_proto.getBytes());
        } else {
            // Non-string column: one buffer
            var offsets_proto = ProtobufEncoder.init(self.allocator);
            defer offsets_proto.deinit();
            try offsets_proto.writeVarint(buf_info.offset);
            try page.writeBytes(1, offsets_proto.getBytes());
        }

        // Field 2: buffer_sizes (packed repeated uint64)
        // For strings: [offsets_size, data_size], else: [data_size]
        if (buf_info.offsets_size > 0) {
            var sizes_proto = ProtobufEncoder.init(self.allocator);
            defer sizes_proto.deinit();
            try sizes_proto.writeVarint(buf_info.offsets_size);
            try sizes_proto.writeVarint(buf_info.size);
            try page.writeBytes(2, sizes_proto.getBytes());
        } else {
            var sizes_proto = ProtobufEncoder.init(self.allocator);
            defer sizes_proto.deinit();
            try sizes_proto.writeVarint(buf_info.size);
            try page.writeBytes(2, sizes_proto.getBytes());
        }

        // Field 3: length (row count)
        try page.writeVarintField(3, buf_info.row_count);

        // Field 5: priority (0)
        try page.writeVarintField(5, 0);

        return try self.allocator.dupe(u8, page.getBytes());
    }

    /// Finalize and return the complete Lance file bytes
    pub fn finalize(self: *Self) ![]const u8 {
        // Record column metadata start
        const column_meta_start = self.output.items.len;

        // Write column metadata for each column (proper Lance ColumnMetadata format)
        for (self.column_buffers.items) |buf_info| {
            var col_meta = ProtobufEncoder.init(self.allocator);
            defer col_meta.deinit();

            // Field 1: encoding (empty for now - plain encoding)
            // Skip for minimal format

            // Field 2: pages (one page per column)
            const page_bytes = try self.encodePage(buf_info);
            defer self.allocator.free(page_bytes);
            try col_meta.writeBytes(2, page_bytes);

            // Copy metadata bytes
            const meta = try self.allocator.dupe(u8, col_meta.getBytes());
            try self.column_metadata.append(self.allocator, meta);
            try self.output.appendSlice(self.allocator, meta);
        }

        // Record column metadata offsets start
        const column_meta_offsets_start = self.output.items.len;

        // Write column metadata offset table (position: u64, length: u64 per column = 16 bytes)
        var meta_offset: u64 = column_meta_start;
        for (self.column_metadata.items) |meta| {
            // Position
            var pos_bytes: [8]u8 = undefined;
            std.mem.writeInt(u64, &pos_bytes, meta_offset, .little);
            try self.output.appendSlice(self.allocator, &pos_bytes);
            // Length
            var len_bytes: [8]u8 = undefined;
            std.mem.writeInt(u64, &len_bytes, meta.len, .little);
            try self.output.appendSlice(self.allocator, &len_bytes);
            meta_offset += meta.len;
        }

        // Write global buffers (schema is global buffer 0)
        const schema_bytes = try self.encodeSchema();
        defer self.allocator.free(schema_bytes);

        const schema_offset = self.output.items.len;
        const schema_size = schema_bytes.len;
        try self.output.appendSlice(self.allocator, schema_bytes);

        // Record global buffers offset table start
        const global_buff_offsets_start = self.output.items.len;

        // Write global buffer offset table (position: u64, length: u64)
        // Global buffer 0: schema
        var schema_pos_bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &schema_pos_bytes, schema_offset, .little);
        try self.output.appendSlice(self.allocator, &schema_pos_bytes);

        var schema_len_bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &schema_len_bytes, schema_size, .little);
        try self.output.appendSlice(self.allocator, &schema_len_bytes);

        // Write footer
        const footer = FooterWriter.write(
            column_meta_start,
            column_meta_offsets_start,
            global_buff_offsets_start,
            1, // num_global_buffers (schema)
            @intCast(self.column_buffers.items.len),
            0, // major_version (Lance 2.0)
            3, // minor_version
        );
        try self.output.appendSlice(self.allocator, &footer);

        return self.output.items;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "plain encoder int64" {
    const allocator = std.testing.allocator;

    var encoder = PlainEncoder.init(allocator);
    defer encoder.deinit();

    try encoder.writeInt64(100);
    try encoder.writeInt64(-200);
    try encoder.writeInt64(300);

    const bytes = encoder.getBytes();
    try std.testing.expectEqual(@as(usize, 24), bytes.len);

    // Verify values
    try std.testing.expectEqual(@as(i64, 100), std.mem.readInt(i64, bytes[0..8], .little));
    try std.testing.expectEqual(@as(i64, -200), std.mem.readInt(i64, bytes[8..16], .little));
    try std.testing.expectEqual(@as(i64, 300), std.mem.readInt(i64, bytes[16..24], .little));
}

test "plain encoder float64" {
    const allocator = std.testing.allocator;

    var encoder = PlainEncoder.init(allocator);
    defer encoder.deinit();

    try encoder.writeFloat64(3.14159);
    try encoder.writeFloat64(-2.71828);

    const bytes = encoder.getBytes();
    try std.testing.expectEqual(@as(usize, 16), bytes.len);
}

test "plain encoder bools" {
    const allocator = std.testing.allocator;

    var encoder = PlainEncoder.init(allocator);
    defer encoder.deinit();

    // Pack 8 bools into 1 byte
    const values = [_]bool{ true, false, false, true, true, false, true, false };
    try encoder.writeBools(&values);

    const bytes = encoder.getBytes();
    try std.testing.expectEqual(@as(usize, 1), bytes.len);
    // Expected: 0b01011001 = 89
    try std.testing.expectEqual(@as(u8, 0b01011001), bytes[0]);
}

test "footer writer" {
    const footer = FooterWriter.write(
        1000, // column_meta_start
        2000, // column_meta_offsets_start
        3000, // global_buff_offsets_start
        5, // num_global_buffers
        10, // num_columns
        0, // major_version
        3, // minor_version
    );

    // Verify magic
    try std.testing.expectEqualSlices(u8, "LANC", footer[36..40]);

    // Verify values
    try std.testing.expectEqual(@as(u64, 1000), std.mem.readInt(u64, footer[0..8], .little));
    try std.testing.expectEqual(@as(u64, 2000), std.mem.readInt(u64, footer[8..16], .little));
    try std.testing.expectEqual(@as(u32, 10), std.mem.readInt(u32, footer[28..32], .little));
}

test "protobuf encoder varint" {
    const allocator = std.testing.allocator;

    var encoder = ProtobufEncoder.init(allocator);
    defer encoder.deinit();

    // Single byte varint (< 128)
    try encoder.writeVarint(100);
    try std.testing.expectEqual(@as(usize, 1), encoder.buffer.items.len);
    try std.testing.expectEqual(@as(u8, 100), encoder.buffer.items[0]);

    encoder.buffer.clearRetainingCapacity();

    // Two byte varint (300 = 0b100101100)
    try encoder.writeVarint(300);
    try std.testing.expectEqual(@as(usize, 2), encoder.buffer.items.len);
    try std.testing.expectEqual(@as(u8, 0xAC), encoder.buffer.items[0]); // 0b10101100
    try std.testing.expectEqual(@as(u8, 0x02), encoder.buffer.items[1]); // 0b00000010
}
