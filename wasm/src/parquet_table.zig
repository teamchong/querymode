//! Parquet Table wrapper for CLI
//!
//! Provides a Table-like interface for Parquet files to enable unified
//! query execution across both Lance and Parquet formats.

const std = @import("std");
const format = @import("lanceql.format");
const ParquetFile = format.ParquetFile;
const meta = format.parquet_metadata;
const Type = meta.Type;
const page_mod = @import("lanceql.encoding.parquet");
const PageReader = page_mod.PageReader;

pub const ParquetTableError = error{
    InvalidMagic,
    FileTooSmall,
    MetadataTooLarge,
    InvalidMetadata,
    UnsupportedVersion,
    OutOfMemory,
    UnexpectedEndOfData,
    MalformedVarint,
    InvalidType,
    InvalidFieldDelta,
    UnsupportedType,
    ColumnNotFound,
    NoRowGroups,
};

/// High-level Parquet table reader
pub const ParquetTable = struct {
    allocator: std.mem.Allocator,
    parquet_file: ParquetFile,
    data: []const u8,
    column_names: [][]const u8,

    const Self = @This();

    /// Open a Parquet table from bytes
    pub fn init(allocator: std.mem.Allocator, data: []const u8) ParquetTableError!Self {
        var parquet_file = ParquetFile.init(allocator, data) catch |err| {
            return switch (err) {
                error.FileTooSmall => ParquetTableError.FileTooSmall,
                error.InvalidMagic => ParquetTableError.InvalidMagic,
                error.MetadataTooLarge => ParquetTableError.MetadataTooLarge,
                error.InvalidMetadata => ParquetTableError.InvalidMetadata,
                error.OutOfMemory => ParquetTableError.OutOfMemory,
                error.UnexpectedEndOfData => ParquetTableError.UnexpectedEndOfData,
                error.MalformedVarint => ParquetTableError.MalformedVarint,
                error.InvalidType => ParquetTableError.InvalidType,
                error.InvalidFieldDelta => ParquetTableError.InvalidFieldDelta,
                else => ParquetTableError.InvalidMetadata,
            };
        };
        errdefer parquet_file.deinit();

        const column_names = parquet_file.getColumnNames() catch return ParquetTableError.OutOfMemory;

        return Self{
            .allocator = allocator,
            .parquet_file = parquet_file,
            .data = data,
            .column_names = column_names,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.column_names);
        self.parquet_file.deinit();
    }

    /// Get number of columns
    pub fn numColumns(self: Self) u32 {
        return @intCast(self.parquet_file.getNumColumns());
    }

    /// Get number of rows
    pub fn numRows(self: Self) usize {
        return @intCast(self.parquet_file.getNumRows());
    }

    /// Get column names
    pub fn getColumnNames(self: Self) [][]const u8 {
        return self.column_names;
    }

    /// Get column index by name
    pub fn columnIndex(self: Self, name: []const u8) ?usize {
        for (self.column_names, 0..) |col_name, i| {
            if (std.mem.eql(u8, col_name, name)) {
                return i;
            }
        }
        return null;
    }

    /// Get column type
    pub fn getColumnType(self: Self, col_idx: usize) ?Type {
        var leaf_idx: usize = 0;
        for (self.parquet_file.getSchema()) |elem| {
            if (elem.num_children == null) {
                if (leaf_idx == col_idx) {
                    return elem.type_;
                }
                leaf_idx += 1;
            }
        }
        return null;
    }

    /// Generic helper to read numeric column data across row groups
    fn readNumericColumn(self: *Self, comptime T: type, comptime field: []const u8, col_idx: usize) ParquetTableError![]T {
        const num_row_groups = self.parquet_file.getNumRowGroups();
        if (num_row_groups == 0) return ParquetTableError.NoRowGroups;

        var all_values = std.ArrayListUnmanaged(T){};
        errdefer all_values.deinit(self.allocator);

        for (0..num_row_groups) |rg_idx| {
            const chunk = self.parquet_file.getColumnChunk(rg_idx, col_idx) orelse continue;
            const col_meta = chunk.meta_data orelse continue;
            const col_data = self.parquet_file.getColumnData(rg_idx, col_idx) orelse continue;

            var reader = PageReader.init(col_data, col_meta.type_, null, col_meta.codec, self.allocator);
            defer reader.deinit();

            const decoded = reader.readAll() catch continue;
            defer {
                if (@field(decoded, field)) |v| self.allocator.free(v);
            }

            if (@field(decoded, field)) |values| {
                all_values.appendSlice(self.allocator, values) catch return ParquetTableError.OutOfMemory;
            }
        }

        return all_values.toOwnedSlice(self.allocator) catch return ParquetTableError.OutOfMemory;
    }

    pub fn readInt64Column(self: *Self, col_idx: usize) ParquetTableError![]i64 {
        return self.readNumericColumn(i64, "int64_values", col_idx);
    }

    pub fn readInt32Column(self: *Self, col_idx: usize) ParquetTableError![]i32 {
        return self.readNumericColumn(i32, "int32_values", col_idx);
    }

    pub fn readFloat64Column(self: *Self, col_idx: usize) ParquetTableError![]f64 {
        return self.readNumericColumn(f64, "double_values", col_idx);
    }

    pub fn readFloat32Column(self: *Self, col_idx: usize) ParquetTableError![]f32 {
        return self.readNumericColumn(f32, "float_values", col_idx);
    }

    /// Read string column data
    pub fn readStringColumn(self: *Self, col_idx: usize) ParquetTableError![][]const u8 {
        const num_row_groups = self.parquet_file.getNumRowGroups();
        if (num_row_groups == 0) return ParquetTableError.NoRowGroups;

        var all_values = std.ArrayListUnmanaged([]const u8){};
        errdefer all_values.deinit(self.allocator);

        for (0..num_row_groups) |rg_idx| {
            const chunk = self.parquet_file.getColumnChunk(rg_idx, col_idx) orelse continue;
            const col_meta = chunk.meta_data orelse continue;
            const col_data = self.parquet_file.getColumnData(rg_idx, col_idx) orelse continue;

            var reader = PageReader.init(
                col_data,
                col_meta.type_,
                null, // type_length only needed for fixed_len_byte_array
                col_meta.codec,
                self.allocator,
            );
            defer reader.deinit();

            const decoded = reader.readAll() catch continue;
            defer {
                if (decoded.binary_values) |v| self.allocator.free(v);
            }

            if (decoded.binary_values) |values| {
                // Copy strings
                for (values) |v| {
                    const copy = self.allocator.dupe(u8, v) catch return ParquetTableError.OutOfMemory;
                    all_values.append(self.allocator, copy) catch return ParquetTableError.OutOfMemory;
                }
            }
        }

        return all_values.toOwnedSlice(self.allocator) catch return ParquetTableError.OutOfMemory;
    }

    pub fn readBoolColumn(self: *Self, col_idx: usize) ParquetTableError![]bool {
        return self.readNumericColumn(bool, "bool_values", col_idx);
    }
};

// =============================================================================
// Tests
// =============================================================================

test "parquet_table: init and basic properties" {
    const allocator = std.testing.allocator;
    const data = @embedFile("../tests/fixtures/simple.parquet");

    var table = try ParquetTable.init(allocator, data);
    defer table.deinit();

    // Verify basic properties
    try std.testing.expectEqual(@as(u32, 3), table.numColumns());
    try std.testing.expectEqual(@as(usize, 5), table.numRows());

    // Verify column names
    const names = table.getColumnNames();
    try std.testing.expectEqual(@as(usize, 3), names.len);
    try std.testing.expectEqualStrings("id", names[0]);
    try std.testing.expectEqualStrings("name", names[1]);
    try std.testing.expectEqualStrings("value", names[2]);
}

test "parquet_table: column index lookup" {
    const allocator = std.testing.allocator;
    const data = @embedFile("../tests/fixtures/simple.parquet");

    var table = try ParquetTable.init(allocator, data);
    defer table.deinit();

    // Test columnIndex
    try std.testing.expectEqual(@as(?usize, 0), table.columnIndex("id"));
    try std.testing.expectEqual(@as(?usize, 1), table.columnIndex("name"));
    try std.testing.expectEqual(@as(?usize, 2), table.columnIndex("value"));
    try std.testing.expectEqual(@as(?usize, null), table.columnIndex("nonexistent"));
}

test "parquet_table: get column type" {
    const allocator = std.testing.allocator;
    const data = @embedFile("../tests/fixtures/simple.parquet");

    var table = try ParquetTable.init(allocator, data);
    defer table.deinit();

    // Verify column types
    try std.testing.expectEqual(Type.INT64, table.getColumnType(0).?);
    try std.testing.expectEqual(Type.BYTE_ARRAY, table.getColumnType(1).?);
    try std.testing.expectEqual(Type.DOUBLE, table.getColumnType(2).?);
    try std.testing.expectEqual(@as(?Type, null), table.getColumnType(99));
}

test "parquet_table: read int64 column" {
    const allocator = std.testing.allocator;
    const data = @embedFile("../tests/fixtures/simple.parquet");

    var table = try ParquetTable.init(allocator, data);
    defer table.deinit();

    const values = try table.readInt64Column(0);
    defer allocator.free(values);

    try std.testing.expectEqual(@as(usize, 5), values.len);
    try std.testing.expectEqual(@as(i64, 1), values[0]);
    try std.testing.expectEqual(@as(i64, 2), values[1]);
    try std.testing.expectEqual(@as(i64, 3), values[2]);
    try std.testing.expectEqual(@as(i64, 4), values[3]);
    try std.testing.expectEqual(@as(i64, 5), values[4]);
}

test "parquet_table: read string column" {
    const allocator = std.testing.allocator;
    const data = @embedFile("../tests/fixtures/simple.parquet");

    var table = try ParquetTable.init(allocator, data);
    defer table.deinit();

    const values = try table.readStringColumn(1);
    defer {
        for (values) |v| allocator.free(v);
        allocator.free(values);
    }

    try std.testing.expectEqual(@as(usize, 5), values.len);
    try std.testing.expectEqualStrings("alice", values[0]);
    try std.testing.expectEqualStrings("bob", values[1]);
    try std.testing.expectEqualStrings("charlie", values[2]);
    try std.testing.expectEqualStrings("diana", values[3]);
    try std.testing.expectEqualStrings("eve", values[4]);
}

test "parquet_table: read float64 column" {
    const allocator = std.testing.allocator;
    const data = @embedFile("../tests/fixtures/simple.parquet");

    var table = try ParquetTable.init(allocator, data);
    defer table.deinit();

    const values = try table.readFloat64Column(2);
    defer allocator.free(values);

    try std.testing.expectEqual(@as(usize, 5), values.len);
    // Check values are approximately correct (floating point)
    try std.testing.expectApproxEqAbs(@as(f64, 1.1), values[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 2.2), values[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 3.3), values[2], 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 4.4), values[3], 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 5.5), values[4], 0.01);
}
