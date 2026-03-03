//! Lance File Writer - writes .lance data files with columnar encoding.
//!
//! This module provides functionality to write Lance data files that are
//! compatible with LanceDB readers. It supports multiple column types and
//! handles proper alignment and metadata encoding.
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
//! | Global Buffers   |  <- Shared buffers (schema)
//! +------------------+
//! | Global Offsets   |  <- Array of u64 offsets to global buffers
//! +------------------+
//! | Footer (40 bytes)|  <- Version, offsets, magic "LANC"
//! +------------------+
//! ```

const std = @import("std");
const proto_mod = @import("edgeq.proto");

const ProtoEncoder = proto_mod.ProtoEncoder;

/// Column data types supported for writing
pub const ColumnType = enum {
    int32,
    int64,
    float32,
    float64,
    string,
    bool,
    /// Fixed-size list of float32 (embedding vectors)
    fixed_size_list_f32,
};

/// Errors that can occur during lance file writing
pub const WriteError = error{
    OutOfMemory,
    ColumnNotFinalized,
    NoColumns,
    InvalidColumnType,
    RowCountMismatch,
};

/// Information about a written column
const ColumnInfo = struct {
    name: []const u8,
    col_type: ColumnType,
    data_offset: u64,
    data_size: u64,
    row_count: u64,
    nullable: bool,
    /// For fixed_size_list_f32: dimension of vector
    vector_dim: u32,
    /// For strings: offset where the offsets array starts within data
    offsets_offset: u64,
    offsets_size: u64,
};

/// Builds column data before adding to the writer.
///
/// Example:
/// ```zig
/// var builder = writer.addColumn("id", .int64);
/// try builder.appendInt64(&[_]i64{1, 2, 3});
/// try builder.finalize();
/// ```
pub const ColumnBuilder = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
    col_type: ColumnType,
    nullable: bool,
    vector_dim: u32,

    // Data buffers
    data: std.ArrayListUnmanaged(u8),
    offsets: std.ArrayListUnmanaged(u8), // For strings
    row_count: u64,
    finalized: bool,

    const Self = @This();

    fn init(allocator: std.mem.Allocator, name: []const u8, col_type: ColumnType) Self {
        return Self{
            .allocator = allocator,
            .name = name,
            .col_type = col_type,
            .nullable = true,
            .vector_dim = 0,
            .data = std.ArrayListUnmanaged(u8){},
            .offsets = std.ArrayListUnmanaged(u8){},
            .row_count = 0,
            .finalized = false,
        };
    }

    pub fn deinit(self: *Self) void {
        self.data.deinit(self.allocator);
        self.offsets.deinit(self.allocator);
    }

    /// Set whether this column is nullable.
    pub fn setNullable(self: *Self, nullable: bool) *Self {
        self.nullable = nullable;
        return self;
    }

    /// Set vector dimension (for fixed_size_list_f32).
    pub fn setVectorDim(self: *Self, dim: u32) *Self {
        self.vector_dim = dim;
        return self;
    }

    /// Append int64 values to the column.
    pub fn appendInt64(self: *Self, values: []const i64) !void {
        if (self.col_type != .int64) return WriteError.InvalidColumnType;

        for (values) |v| {
            var bytes: [8]u8 = undefined;
            std.mem.writeInt(i64, &bytes, v, .little);
            try self.data.appendSlice(self.allocator, &bytes);
        }
        self.row_count += values.len;
    }

    /// Append int32 values to the column.
    pub fn appendInt32(self: *Self, values: []const i32) !void {
        if (self.col_type != .int32) return WriteError.InvalidColumnType;

        for (values) |v| {
            var bytes: [4]u8 = undefined;
            std.mem.writeInt(i32, &bytes, v, .little);
            try self.data.appendSlice(self.allocator, &bytes);
        }
        self.row_count += values.len;
    }

    /// Append float64 values to the column.
    pub fn appendFloat64(self: *Self, values: []const f64) !void {
        if (self.col_type != .float64) return WriteError.InvalidColumnType;

        for (values) |v| {
            var bytes: [8]u8 = undefined;
            const bits: u64 = @bitCast(v);
            std.mem.writeInt(u64, &bytes, bits, .little);
            try self.data.appendSlice(self.allocator, &bytes);
        }
        self.row_count += values.len;
    }

    /// Append float32 values to the column.
    pub fn appendFloat32(self: *Self, values: []const f32) !void {
        if (self.col_type != .float32) return WriteError.InvalidColumnType;

        for (values) |v| {
            var bytes: [4]u8 = undefined;
            const bits: u32 = @bitCast(v);
            std.mem.writeInt(u32, &bytes, bits, .little);
            try self.data.appendSlice(self.allocator, &bytes);
        }
        self.row_count += values.len;
    }

    /// Append string values to the column.
    /// Writes both the concatenated string data and the offsets array.
    pub fn appendStrings(self: *Self, values: []const []const u8) !void {
        if (self.col_type != .string) return WriteError.InvalidColumnType;

        // Write starting offset (0 for first batch, or current position)
        if (self.offsets.items.len == 0) {
            var offset_bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &offset_bytes, 0, .little);
            try self.offsets.appendSlice(self.allocator, &offset_bytes);
        }

        for (values) |str| {
            // Write string data
            try self.data.appendSlice(self.allocator, str);

            // Write end offset
            var offset_bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &offset_bytes, @intCast(self.data.items.len), .little);
            try self.offsets.appendSlice(self.allocator, &offset_bytes);
        }
        self.row_count += values.len;
    }

    /// Append boolean values to the column (bit-packed).
    pub fn appendBools(self: *Self, values: []const bool) !void {
        if (self.col_type != .bool) return WriteError.InvalidColumnType;

        // Bit-pack: 8 bools per byte
        const start_row = self.row_count;
        self.row_count += values.len;

        // Calculate bytes needed
        const total_bytes = (self.row_count + 7) / 8;
        const current_bytes = self.data.items.len;

        // Extend buffer if needed
        if (total_bytes > current_bytes) {
            const new_bytes = total_bytes - current_bytes;
            try self.data.appendNTimes(self.allocator, 0, new_bytes);
        }

        // Set bits
        for (values, 0..) |v, i| {
            if (v) {
                const bit_idx = start_row + i;
                const byte_idx = bit_idx / 8;
                const bit_offset: u3 = @intCast(bit_idx % 8);
                self.data.items[byte_idx] |= @as(u8, 1) << bit_offset;
            }
        }
    }

    /// Append vector values (for fixed_size_list_f32).
    /// Each vector is dim float32 values.
    pub fn appendVectors(self: *Self, vectors: []const []const f32) !void {
        if (self.col_type != .fixed_size_list_f32) return WriteError.InvalidColumnType;
        if (self.vector_dim == 0) return WriteError.InvalidColumnType;

        for (vectors) |vec| {
            if (vec.len != self.vector_dim) return WriteError.InvalidColumnType;

            for (vec) |v| {
                var bytes: [4]u8 = undefined;
                const bits: u32 = @bitCast(v);
                std.mem.writeInt(u32, &bytes, bits, .little);
                try self.data.appendSlice(self.allocator, &bytes);
            }
        }
        self.row_count += vectors.len;
    }

    /// Append flat vector data (for fixed_size_list_f32).
    /// Data must be total_vectors * vector_dim floats.
    pub fn appendVectorData(self: *Self, flat_data: []const f32) !void {
        if (self.col_type != .fixed_size_list_f32) return WriteError.InvalidColumnType;
        if (self.vector_dim == 0) return WriteError.InvalidColumnType;
        if (flat_data.len % self.vector_dim != 0) return WriteError.InvalidColumnType;

        for (flat_data) |v| {
            var bytes: [4]u8 = undefined;
            const bits: u32 = @bitCast(v);
            std.mem.writeInt(u32, &bytes, bits, .little);
            try self.data.appendSlice(self.allocator, &bytes);
        }
        self.row_count += flat_data.len / self.vector_dim;
    }

    /// Mark the column as complete.
    pub fn finalize(self: *Self) !void {
        self.finalized = true;
    }
};

/// Lance file writer that manages multiple columns.
pub const LanceWriter = struct {
    allocator: std.mem.Allocator,
    output: std.ArrayListUnmanaged(u8),
    columns: std.ArrayListUnmanaged(ColumnInfo),
    builders: std.ArrayListUnmanaged(ColumnBuilder),

    const Self = @This();

    /// Create a new Lance file writer.
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .output = std.ArrayListUnmanaged(u8){},
            .columns = std.ArrayListUnmanaged(ColumnInfo){},
            .builders = std.ArrayListUnmanaged(ColumnBuilder){},
        };
    }

    /// Free all resources.
    pub fn deinit(self: *Self) void {
        self.output.deinit(self.allocator);
        self.columns.deinit(self.allocator);
        for (self.builders.items) |*builder| {
            builder.deinit();
        }
        self.builders.deinit(self.allocator);
    }

    /// Add a column and return a builder for populating it.
    pub fn addColumn(self: *Self, name: []const u8, col_type: ColumnType) !*ColumnBuilder {
        const builder = ColumnBuilder.init(self.allocator, name, col_type);
        try self.builders.append(self.allocator, builder);
        return &self.builders.items[self.builders.items.len - 1];
    }

    /// Finalize all columns and produce the complete .lance file bytes.
    pub fn finalize(self: *Self) ![]const u8 {
        if (self.builders.items.len == 0) return WriteError.NoColumns;

        // Check all columns are finalized
        for (self.builders.items) |builder| {
            if (!builder.finalized) return WriteError.ColumnNotFinalized;
        }

        // Write column data with proper alignment
        for (self.builders.items) |*builder| {
            const info = try self.writeColumnData(builder);
            try self.columns.append(self.allocator, info);
        }

        // Write column metadata
        const column_meta_start = self.output.items.len;
        var metadata_offsets = std.ArrayListUnmanaged(u64){};
        defer metadata_offsets.deinit(self.allocator);

        for (self.columns.items) |col| {
            try metadata_offsets.append(self.allocator, self.output.items.len);
            try self.writeColumnMetadata(col);
        }

        // Write column metadata offsets table
        const column_meta_offsets_start = self.output.items.len;
        for (metadata_offsets.items) |offset| {
            var bytes: [8]u8 = undefined;
            std.mem.writeInt(u64, &bytes, offset, .little);
            try self.output.appendSlice(self.allocator, &bytes);
        }

        // Write schema as global buffer
        const schema_offset = self.output.items.len;
        const schema_bytes = try self.encodeSchema();
        defer self.allocator.free(schema_bytes);
        try self.output.appendSlice(self.allocator, schema_bytes);
        const schema_size = self.output.items.len - schema_offset;

        // Write global buffer offsets table
        const global_buff_offsets_start = self.output.items.len;

        // Global buffer 0: schema (offset, size)
        var offset_bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &offset_bytes, schema_offset, .little);
        try self.output.appendSlice(self.allocator, &offset_bytes);

        var size_bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &size_bytes, schema_size, .little);
        try self.output.appendSlice(self.allocator, &size_bytes);

        // Write footer
        try self.writeFooter(
            column_meta_start,
            column_meta_offsets_start,
            global_buff_offsets_start,
            1, // num_global_buffers (schema)
            @intCast(self.columns.items.len),
        );

        return self.output.items;
    }

    fn writeColumnData(self: *Self, builder: *const ColumnBuilder) !ColumnInfo {
        // Align based on column type
        const alignment: usize = switch (builder.col_type) {
            .int64, .float64 => 8,
            .int32, .float32, .string, .fixed_size_list_f32 => 4,
            .bool => 1,
        };

        // Add padding for alignment
        const current = self.output.items.len;
        const padding = (alignment - (current % alignment)) % alignment;
        try self.output.appendNTimes(self.allocator, 0, padding);

        const data_offset = self.output.items.len;

        // For strings: write data first, then offsets
        if (builder.col_type == .string) {
            try self.output.appendSlice(self.allocator, builder.data.items);

            // Align offsets to 4 bytes
            const offsets_padding = (4 - (self.output.items.len % 4)) % 4;
            try self.output.appendNTimes(self.allocator, 0, offsets_padding);

            const offsets_offset = self.output.items.len - data_offset;
            try self.output.appendSlice(self.allocator, builder.offsets.items);

            return ColumnInfo{
                .name = builder.name,
                .col_type = builder.col_type,
                .data_offset = data_offset,
                .data_size = self.output.items.len - data_offset,
                .row_count = builder.row_count,
                .nullable = builder.nullable,
                .vector_dim = 0,
                .offsets_offset = offsets_offset,
                .offsets_size = builder.offsets.items.len,
            };
        }

        // For other types: just write data
        try self.output.appendSlice(self.allocator, builder.data.items);

        return ColumnInfo{
            .name = builder.name,
            .col_type = builder.col_type,
            .data_offset = data_offset,
            .data_size = builder.data.items.len,
            .row_count = builder.row_count,
            .nullable = builder.nullable,
            .vector_dim = builder.vector_dim,
            .offsets_offset = 0,
            .offsets_size = 0,
        };
    }

    fn writeColumnMetadata(self: *Self, col: ColumnInfo) !void {
        var encoder = ProtoEncoder.init(self.allocator);
        defer encoder.deinit();

        // Encode a Page message
        var page_encoder = ProtoEncoder.init(self.allocator);
        defer page_encoder.deinit();

        // Field 1: buffer_offsets (packed varints)
        if (col.col_type == .string) {
            // String: two buffers (offsets first in format, then data)
            var offsets_buf = ProtoEncoder.init(self.allocator);
            defer offsets_buf.deinit();
            try offsets_buf.writeVarint(col.data_offset + col.offsets_offset); // offsets position
            try offsets_buf.writeVarint(col.data_offset); // data position
            try page_encoder.writeBytesField(1, offsets_buf.getBytes());
        } else {
            // Single buffer
            var offsets_buf = ProtoEncoder.init(self.allocator);
            defer offsets_buf.deinit();
            try offsets_buf.writeVarint(col.data_offset);
            try page_encoder.writeBytesField(1, offsets_buf.getBytes());
        }

        // Field 2: buffer_sizes (packed varints)
        if (col.col_type == .string) {
            var sizes_buf = ProtoEncoder.init(self.allocator);
            defer sizes_buf.deinit();
            try sizes_buf.writeVarint(col.offsets_size);
            try sizes_buf.writeVarint(col.data_size - col.offsets_offset - col.offsets_size);
            try page_encoder.writeBytesField(2, sizes_buf.getBytes());
        } else {
            var sizes_buf = ProtoEncoder.init(self.allocator);
            defer sizes_buf.deinit();
            try sizes_buf.writeVarint(col.data_size);
            try page_encoder.writeBytesField(2, sizes_buf.getBytes());
        }

        // Field 3: length (row count)
        try page_encoder.writeVarintField(3, col.row_count);

        // Field 5: priority
        try page_encoder.writeVarintField(5, 0);

        // Write ColumnMetadata message
        // Field 2: pages (repeated Page)
        try encoder.writeMessageField(2, page_encoder.getBytes());

        try self.output.appendSlice(self.allocator, encoder.getBytes());
    }

    fn encodeSchema(self: *Self) ![]const u8 {
        var encoder = ProtoEncoder.init(self.allocator);
        defer encoder.deinit();

        for (self.columns.items, 0..) |col, i| {
            var field_encoder = ProtoEncoder.init(self.allocator);
            defer field_encoder.deinit();

            // Field 1: type (enum) - 2 = leaf (data columns)
            try field_encoder.writeVarintField(1, 2);

            // Field 2: name
            try field_encoder.writeStringField(2, col.name);

            // Field 3: id
            try field_encoder.writeVarintField(3, i);

            // Field 4: parent_id (-1 for root)
            try field_encoder.writeVarintField(4, 0xFFFFFFFF);

            // Field 5: logical_type
            const logical_type = columnTypeToLogicalType(col);
            try field_encoder.writeStringField(5, logical_type);

            // Field 6: nullable
            try field_encoder.writeVarintField(6, if (col.nullable) 1 else 0);

            // Write as nested message (field 1 in schema)
            try encoder.writeMessageField(1, field_encoder.getBytes());
        }

        return try self.allocator.dupe(u8, encoder.getBytes());
    }

    fn columnTypeToLogicalType(col: ColumnInfo) []const u8 {
        return switch (col.col_type) {
            .int32 => "int32",
            .int64 => "int64",
            .float32 => "float",
            .float64 => "double",
            .string => "string",
            .bool => "bool",
            .fixed_size_list_f32 => "fixed_size_list:float32",
        };
    }

    fn writeFooter(
        self: *Self,
        column_meta_start: u64,
        column_meta_offsets_start: u64,
        global_buff_offsets_start: u64,
        num_global_buffers: u32,
        num_columns: u32,
    ) !void {
        var footer: [40]u8 = undefined;

        std.mem.writeInt(u64, footer[0..8], column_meta_start, .little);
        std.mem.writeInt(u64, footer[8..16], column_meta_offsets_start, .little);
        std.mem.writeInt(u64, footer[16..24], global_buff_offsets_start, .little);
        std.mem.writeInt(u32, footer[24..28], num_global_buffers, .little);
        std.mem.writeInt(u32, footer[28..32], num_columns, .little);
        std.mem.writeInt(u16, footer[32..34], 0, .little); // major version (Lance 2.0)
        std.mem.writeInt(u16, footer[34..36], 3, .little); // minor version
        @memcpy(footer[36..40], "LANC");

        try self.output.appendSlice(self.allocator, &footer);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "lance writer: basic int64 column" {
    const allocator = std.testing.allocator;

    var writer = LanceWriter.init(allocator);
    defer writer.deinit();

    var col = try writer.addColumn("id", .int64);
    try col.appendInt64(&[_]i64{ 1, 2, 3, 4, 5 });
    try col.finalize();

    const data = try writer.finalize();

    // Verify magic at end
    try std.testing.expectEqualSlices(u8, "LANC", data[data.len - 4 ..]);

    // Verify we have data
    try std.testing.expect(data.len > 40);
}

test "lance writer: multiple columns" {
    const allocator = std.testing.allocator;

    var writer = LanceWriter.init(allocator);
    defer writer.deinit();

    var id_col = try writer.addColumn("id", .int64);
    try id_col.appendInt64(&[_]i64{ 1, 2, 3 });
    try id_col.finalize();

    var value_col = try writer.addColumn("value", .float64);
    try value_col.appendFloat64(&[_]f64{ 1.5, 2.5, 3.5 });
    try value_col.finalize();

    const data = try writer.finalize();

    // Verify magic
    try std.testing.expectEqualSlices(u8, "LANC", data[data.len - 4 ..]);

    // Verify footer has 2 columns
    const num_columns = std.mem.readInt(u32, data[data.len - 12 ..][0..4], .little);
    try std.testing.expectEqual(@as(u32, 2), num_columns);
}

test "lance writer: string column" {
    const allocator = std.testing.allocator;

    var writer = LanceWriter.init(allocator);
    defer writer.deinit();

    var col = try writer.addColumn("name", .string);
    try col.appendStrings(&[_][]const u8{ "hello", "world", "test" });
    try col.finalize();

    const data = try writer.finalize();

    // Verify magic
    try std.testing.expectEqualSlices(u8, "LANC", data[data.len - 4 ..]);
}

test "lance writer: bool column" {
    const allocator = std.testing.allocator;

    var writer = LanceWriter.init(allocator);
    defer writer.deinit();

    var col = try writer.addColumn("flag", .bool);
    try col.appendBools(&[_]bool{ true, false, true, true, false, false, true, false, true });
    try col.finalize();

    const data = try writer.finalize();

    // Verify magic
    try std.testing.expectEqualSlices(u8, "LANC", data[data.len - 4 ..]);
}

test "lance writer: vector column" {
    const allocator = std.testing.allocator;

    var writer = LanceWriter.init(allocator);
    defer writer.deinit();

    var col = try writer.addColumn("embedding", .fixed_size_list_f32);
    _ = col.setVectorDim(3);
    try col.appendVectorData(&[_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }); // 2 vectors of dim 3
    try col.finalize();

    const data = try writer.finalize();

    // Verify magic
    try std.testing.expectEqualSlices(u8, "LANC", data[data.len - 4 ..]);
}
