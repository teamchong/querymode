//! Arrow IPC Table wrapper for SQL execution
//!
//! Provides a ParquetTable-like interface for Arrow IPC files by:
//! 1. Parsing Arrow IPC format to read schema and data
//! 2. Exposing unified column read interface

const std = @import("std");
const ArrowIpcReader = @import("lanceql.encoding").ArrowIpcReader;
const ArrowType = @import("lanceql.encoding").ArrowType;
const format = @import("lanceql.format");
const Type = format.parquet_metadata.Type;
const table_utils = @import("lanceql.table_utils");

pub const ArrowTableError = error{
    InvalidArrowFile,
    InvalidArrowMagic,
    InvalidSchemaMessage,
    InvalidRecordBatch,
    InvalidFooterLength,
    NoDataFiles,
    OutOfMemory,
    PathTooLong,
    ReadFailed,
    ColumnNotFound,
    NoRowGroups,
    UnsupportedType,
};

/// High-level Arrow IPC table reader
pub const ArrowTable = struct {
    allocator: std.mem.Allocator,
    reader: ArrowIpcReader,
    column_names: [][]const u8, // Cached column names

    const Self = @This();

    /// Open an Arrow IPC file from in-memory data
    pub fn init(allocator: std.mem.Allocator, data: []const u8) ArrowTableError!Self {
        // Initialize reader
        var reader = ArrowIpcReader.init(allocator, data) catch {
            return ArrowTableError.InvalidArrowFile;
        };
        errdefer reader.deinit();

        // Cache column names
        const num_cols = reader.columnCount();
        var column_names = allocator.alloc([]const u8, num_cols) catch {
            return ArrowTableError.OutOfMemory;
        };
        errdefer allocator.free(column_names);

        for (0..num_cols) |i| {
            const name = reader.getColumnName(i);
            column_names[i] = allocator.dupe(u8, name) catch {
                // Clean up already allocated names
                for (0..i) |j| {
                    allocator.free(column_names[j]);
                }
                return ArrowTableError.OutOfMemory;
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
        const arrow_type = self.reader.getColumnType(col_idx);
        return mapArrowToParquetType(arrow_type);
    }

    /// Generic helper to convert array from one type to another
    fn convertArray(self: *Self, comptime From: type, comptime To: type, values: []const From) ArrowTableError![]To {
        return table_utils.convertArray(self.allocator, From, To, values) catch {
            return ArrowTableError.OutOfMemory;
        };
    }

    /// Read int64 column data
    pub fn readInt64Column(self: *Self, col_idx: usize) ArrowTableError![]i64 {
        return self.reader.readInt64Column(col_idx) catch {
            return ArrowTableError.ReadFailed;
        };
    }

    /// Read int32 column data (by reading int64 and converting)
    pub fn readInt32Column(self: *Self, col_idx: usize) ArrowTableError![]i32 {
        const values64 = self.reader.readInt64Column(col_idx) catch {
            return ArrowTableError.ReadFailed;
        };
        defer self.allocator.free(values64);
        return self.convertArray(i64, i32, values64);
    }

    /// Read float64 column data
    pub fn readFloat64Column(self: *Self, col_idx: usize) ArrowTableError![]f64 {
        return self.reader.readFloat64Column(col_idx) catch {
            return ArrowTableError.ReadFailed;
        };
    }

    /// Read float32 column data (by reading float64 and converting)
    pub fn readFloat32Column(self: *Self, col_idx: usize) ArrowTableError![]f32 {
        const values64 = self.reader.readFloat64Column(col_idx) catch {
            return ArrowTableError.ReadFailed;
        };
        defer self.allocator.free(values64);
        return self.convertArray(f64, f32, values64);
    }

    /// Read string column data
    pub fn readStringColumn(self: *Self, col_idx: usize) ArrowTableError![][]const u8 {
        return self.reader.readStringColumn(col_idx) catch {
            return ArrowTableError.ReadFailed;
        };
    }

    /// Read bool column data
    pub fn readBoolColumn(self: *Self, col_idx: usize) ArrowTableError![]bool {
        const values64 = self.reader.readInt64Column(col_idx) catch {
            return ArrowTableError.ReadFailed;
        };
        defer self.allocator.free(values64);
        return self.convertArray(i64, bool, values64);
    }

    /// Check if path is a valid Arrow IPC file
    pub fn isValid(path: []const u8) bool {
        const file = std.fs.cwd().openFile(path, .{}) catch return false;
        defer file.close();

        var header: [6]u8 = undefined;
        _ = file.read(&header) catch return false;

        return std.mem.eql(u8, &header, "ARROW1");
    }
};

/// Map Arrow types to Parquet types
fn mapArrowToParquetType(arrow_type: ArrowType) ?Type {
    return switch (arrow_type) {
        .int8, .int16, .int32 => .int32,
        .int64 => .int64,
        .uint8, .uint16, .uint32 => .int32,
        .uint64 => .int64,
        .float32 => .float,
        .float64 => .double,
        .utf8, .large_utf8, .binary, .large_binary => .byte_array,
        .bool_type => .boolean,
        .date32, .date64 => .int32,
        .timestamp, .time32, .time64 => .int64,
        else => null,
    };
}

test "arrow_table: read simple fixture" {
    const allocator = std.testing.allocator;

    // Read file into memory
    const file = std.fs.cwd().openFile("tests/fixtures/simple.arrow", .{}) catch |err| {
        std.debug.print("Failed to open Arrow file: {}\n", .{err});
        return error.TestFailed;
    };
    defer file.close();
    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch |err| {
        std.debug.print("Failed to read Arrow file: {}\n", .{err});
        return error.TestFailed;
    };
    defer allocator.free(data);

    var table = ArrowTable.init(allocator, data) catch |err| {
        std.debug.print("Failed to parse Arrow file: {}\n", .{err});
        return error.TestFailed;
    };
    defer table.deinit();

    // Check metadata
    try std.testing.expectEqual(@as(u32, 3), table.numColumns());
    try std.testing.expectEqual(@as(usize, 5), table.numRows());

    // Check column names
    const names = table.getColumnNames();
    try std.testing.expectEqual(@as(usize, 3), names.len);

    // Read id column
    const ids = try table.readInt64Column(0);
    defer allocator.free(ids);
    try std.testing.expectEqual(@as(usize, 5), ids.len);
    try std.testing.expectEqual(@as(i64, 1), ids[0]);
    try std.testing.expectEqual(@as(i64, 5), ids[4]);
}

test "arrow_table: read string column" {
    const allocator = std.testing.allocator;

    const file = std.fs.cwd().openFile("tests/fixtures/simple.arrow", .{}) catch return error.SkipZigTest;
    defer file.close();
    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch return error.SkipZigTest;
    defer allocator.free(data);

    var table = ArrowTable.init(allocator, data) catch return error.SkipZigTest;
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

test "arrow_table: read float64 column" {
    const allocator = std.testing.allocator;

    const file = std.fs.cwd().openFile("tests/fixtures/simple.arrow", .{}) catch return error.SkipZigTest;
    defer file.close();
    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch return error.SkipZigTest;
    defer allocator.free(data);

    var table = ArrowTable.init(allocator, data) catch return error.SkipZigTest;
    defer table.deinit();

    const values = try table.readFloat64Column(2);
    defer allocator.free(values);

    try std.testing.expectEqual(@as(usize, 5), values.len);
    // Values start at 0.0, 1.1, 2.2, ...
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), values[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 1.1), values[1], 0.01);
}

test "arrow_table: column index and type" {
    const allocator = std.testing.allocator;

    const file = std.fs.cwd().openFile("tests/fixtures/simple.arrow", .{}) catch return error.SkipZigTest;
    defer file.close();
    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch return error.SkipZigTest;
    defer allocator.free(data);

    var table = ArrowTable.init(allocator, data) catch return error.SkipZigTest;
    defer table.deinit();

    // Test columnIndex
    try std.testing.expectEqual(@as(?usize, 0), table.columnIndex("col_0"));
    try std.testing.expectEqual(@as(?usize, 1), table.columnIndex("col_1"));
    try std.testing.expectEqual(@as(?usize, null), table.columnIndex("nonexistent"));

    // Test column types
    try std.testing.expectEqual(Type.int64, table.getColumnType(0).?);
    try std.testing.expectEqual(Type.byte_array, table.getColumnType(1).?);
    try std.testing.expectEqual(Type.double, table.getColumnType(2).?);
}
