//! Avro Table wrapper for SQL execution
//!
//! Provides a ParquetTable-like interface for Avro files by:
//! 1. Parsing Avro container format to read schema and data
//! 2. Exposing unified column read interface

const std = @import("std");
const AvroReader = @import("lanceql.encoding").AvroReader;
const AvroType = @import("lanceql.encoding").AvroType;
const format = @import("lanceql.format");
const Type = format.parquet_metadata.Type;
const table_utils = @import("lanceql.table_utils");

pub const AvroTableError = error{
    InvalidAvroFile,
    InvalidAvroMagic,
    InvalidMetadata,
    InvalidData,
    UnsupportedType,
    VarintTooLong,
    UnexpectedEof,
    NoDataFiles,
    OutOfMemory,
    PathTooLong,
    ReadFailed,
    ColumnNotFound,
    NoRowGroups,
};

/// High-level Avro table reader
pub const AvroTable = struct {
    allocator: std.mem.Allocator,
    reader: AvroReader,
    column_names: [][]const u8, // Cached column names

    const Self = @This();

    /// Open an Avro file from in-memory data
    pub fn init(allocator: std.mem.Allocator, data: []const u8) AvroTableError!Self {
        // Initialize reader
        var reader = AvroReader.init(allocator, data) catch {
            return AvroTableError.InvalidAvroFile;
        };
        errdefer reader.deinit();

        // Cache column names
        const num_cols = reader.columnCount();
        var column_names = allocator.alloc([]const u8, num_cols) catch {
            return AvroTableError.OutOfMemory;
        };
        errdefer allocator.free(column_names);

        for (0..num_cols) |i| {
            const name = reader.getFieldName(i) orelse "unknown";
            column_names[i] = allocator.dupe(u8, name) catch {
                // Clean up already allocated names
                for (0..i) |j| {
                    allocator.free(column_names[j]);
                }
                return AvroTableError.OutOfMemory;
            };
        }

        return Self{
            .allocator = allocator,
            .reader = reader,
            .column_names = column_names,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.column_names) |name| {
            self.allocator.free(name);
        }
        self.allocator.free(self.column_names);
        self.reader.deinit();
    }

    // =========================================================================
    // ParquetTable-compatible interface
    // =========================================================================

    /// Get number of columns
    pub fn numColumns(self: Self) u32 {
        return @intCast(self.reader.columnCount());
    }

    /// Get number of rows
    pub fn numRows(self: Self) usize {
        return self.reader.rowCount();
    }

    /// Get column names
    pub fn getColumnNames(self: Self) [][]const u8 {
        return self.column_names;
    }

    /// Get column index by name
    pub fn columnIndex(self: Self, name: []const u8) ?usize {
        return table_utils.findColumnIndex(self.column_names, name);
    }

    /// Get column type (mapped to Parquet types)
    pub fn getColumnType(self: Self, col_idx: usize) ?Type {
        const avro_type = self.reader.getFieldType(col_idx) orelse return null;
        return mapAvroToParquetType(avro_type);
    }

    /// Generic helper to convert array from one type to another
    fn convertArray(self: *Self, comptime From: type, comptime To: type, values: []const From) AvroTableError![]To {
        return table_utils.convertArray(self.allocator, From, To, values) catch {
            return AvroTableError.OutOfMemory;
        };
    }

    /// Read int64 column data
    pub fn readInt64Column(self: *Self, col_idx: usize) AvroTableError![]i64 {
        return self.reader.readLongColumn(col_idx) catch {
            return AvroTableError.ReadFailed;
        };
    }

    /// Read int32 column data (by reading int64 and converting)
    pub fn readInt32Column(self: *Self, col_idx: usize) AvroTableError![]i32 {
        const values64 = try self.readInt64Column(col_idx);
        defer self.allocator.free(values64);
        return self.convertArray(i64, i32, values64);
    }

    /// Read float64 column data
    pub fn readFloat64Column(self: *Self, col_idx: usize) AvroTableError![]f64 {
        return self.reader.readDoubleColumn(col_idx) catch {
            return AvroTableError.ReadFailed;
        };
    }

    /// Read float32 column data (by reading float64 and converting)
    pub fn readFloat32Column(self: *Self, col_idx: usize) AvroTableError![]f32 {
        const values64 = try self.readFloat64Column(col_idx);
        defer self.allocator.free(values64);
        return self.convertArray(f64, f32, values64);
    }

    /// Read string column data
    pub fn readStringColumn(self: *Self, col_idx: usize) AvroTableError![][]const u8 {
        return self.reader.readStringColumn(col_idx) catch {
            return AvroTableError.ReadFailed;
        };
    }

    /// Read bool column data
    pub fn readBoolColumn(self: *Self, col_idx: usize) AvroTableError![]bool {
        const values64 = try self.readInt64Column(col_idx);
        defer self.allocator.free(values64);
        return self.convertArray(i64, bool, values64);
    }

    /// Check if path is a valid Avro file
    pub fn isValid(path: []const u8) bool {
        const file = std.fs.cwd().openFile(path, .{}) catch return false;
        defer file.close();

        var header: [4]u8 = undefined;
        _ = file.read(&header) catch return false;

        return std.mem.eql(u8, &header, "Obj\x01");
    }
};

/// Map Avro types to Parquet types
fn mapAvroToParquetType(avro_type: AvroType) ?Type {
    return switch (avro_type) {
        .int_type => .int32,
        .long_type => .int64,
        .float_type => .float,
        .double_type => .double,
        .bytes, .string => .byte_array,
        .boolean => .boolean,
        else => null,
    };
}

test "avro_table: read simple fixture" {
    const allocator = std.testing.allocator;

    // Read file into memory
    const file = std.fs.cwd().openFile("tests/fixtures/simple.avro", .{}) catch |err| {
        std.debug.print("Failed to open Avro file: {}\n", .{err});
        return error.TestFailed;
    };
    defer file.close();
    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch |err| {
        std.debug.print("Failed to read Avro file: {}\n", .{err});
        return error.TestFailed;
    };
    defer allocator.free(data);

    var table = AvroTable.init(allocator, data) catch |err| {
        std.debug.print("Failed to parse Avro file: {}\n", .{err});
        return error.TestFailed;
    };
    defer table.deinit();

    // Check metadata
    try std.testing.expectEqual(@as(u32, 3), table.numColumns());
    try std.testing.expectEqual(@as(usize, 5), table.numRows());

    // Check column names
    const names = table.getColumnNames();
    try std.testing.expectEqual(@as(usize, 3), names.len);
    try std.testing.expectEqualStrings("id", names[0]);
    try std.testing.expectEqualStrings("name", names[1]);
    try std.testing.expectEqualStrings("value", names[2]);

    // Read id column
    const ids = try table.readInt64Column(0);
    defer allocator.free(ids);
    try std.testing.expectEqual(@as(usize, 5), ids.len);
    try std.testing.expectEqual(@as(i64, 1), ids[0]);
    try std.testing.expectEqual(@as(i64, 5), ids[4]);
}

test "avro_table: read string column" {
    const allocator = std.testing.allocator;

    const file = std.fs.cwd().openFile("tests/fixtures/simple.avro", .{}) catch return error.SkipZigTest;
    defer file.close();
    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch return error.SkipZigTest;
    defer allocator.free(data);

    var table = AvroTable.init(allocator, data) catch return error.SkipZigTest;
    defer table.deinit();

    const values = try table.readStringColumn(1);
    defer {
        for (values) |v| allocator.free(v);
        allocator.free(values);
    }

    try std.testing.expectEqual(@as(usize, 5), values.len);
    try std.testing.expectEqualStrings("alice", values[0]);
    try std.testing.expectEqualStrings("bob", values[1]);
}

test "avro_table: read float64 column" {
    const allocator = std.testing.allocator;

    const file = std.fs.cwd().openFile("tests/fixtures/simple.avro", .{}) catch return error.SkipZigTest;
    defer file.close();
    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch return error.SkipZigTest;
    defer allocator.free(data);

    var table = AvroTable.init(allocator, data) catch return error.SkipZigTest;
    defer table.deinit();

    const values = try table.readFloat64Column(2);
    defer allocator.free(values);

    try std.testing.expectEqual(@as(usize, 5), values.len);
    // Values start at 1.1, 2.2, ...
    try std.testing.expectApproxEqAbs(@as(f64, 1.1), values[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 2.2), values[1], 0.01);
}

test "avro_table: column index and type" {
    const allocator = std.testing.allocator;

    const file = std.fs.cwd().openFile("tests/fixtures/simple.avro", .{}) catch return error.SkipZigTest;
    defer file.close();
    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch return error.SkipZigTest;
    defer allocator.free(data);

    var table = AvroTable.init(allocator, data) catch return error.SkipZigTest;
    defer table.deinit();

    // Test columnIndex
    try std.testing.expectEqual(@as(?usize, 0), table.columnIndex("id"));
    try std.testing.expectEqual(@as(?usize, 1), table.columnIndex("name"));
    try std.testing.expectEqual(@as(?usize, 2), table.columnIndex("value"));
    try std.testing.expectEqual(@as(?usize, null), table.columnIndex("nonexistent"));

    // Test column types
    try std.testing.expectEqual(Type.int64, table.getColumnType(0).?);
    try std.testing.expectEqual(Type.byte_array, table.getColumnType(1).?);
    try std.testing.expectEqual(Type.double, table.getColumnType(2).?);
}
