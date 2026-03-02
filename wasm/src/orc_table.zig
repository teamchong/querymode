//! ORC Table wrapper for SQL execution
//!
//! Provides a ParquetTable-like interface for ORC files by:
//! 1. Parsing ORC format to read schema and data
//! 2. Exposing unified column read interface

const std = @import("std");
const OrcReader = @import("lanceql.encoding").OrcReader;
const OrcType = @import("lanceql.encoding").OrcType;
const format = @import("lanceql.format");
const Type = format.parquet_metadata.Type;
const table_utils = @import("lanceql.table_utils");

pub const OrcTableError = error{
    InvalidOrcFile,
    InvalidOrcMagic,
    InvalidPostScript,
    InvalidFooter,
    InvalidStripeFooter,
    InvalidStream,
    CompressionError,
    UnsupportedCompression,
    NoDataFiles,
    OutOfMemory,
    PathTooLong,
    ReadFailed,
    ColumnNotFound,
    NoRowGroups,
};

/// High-level ORC table reader
pub const OrcTable = struct {
    allocator: std.mem.Allocator,
    reader: OrcReader,
    column_names: [][]const u8, // Cached column names

    const Self = @This();

    /// Open an ORC file from in-memory data
    pub fn init(allocator: std.mem.Allocator, data: []const u8) OrcTableError!Self {
        // Initialize reader
        var reader = OrcReader.init(allocator, data) catch {
            return OrcTableError.InvalidOrcFile;
        };
        errdefer reader.deinit();

        // Cache column names
        const num_cols = reader.columnCount();
        var column_names = allocator.alloc([]const u8, num_cols) catch {
            return OrcTableError.OutOfMemory;
        };
        errdefer allocator.free(column_names);

        // Copy column names from reader using safe accessor
        for (0..num_cols) |i| {
            const name = reader.getColumnName(i) orelse "unknown";
            column_names[i] = allocator.dupe(u8, name) catch {
                // Clean up already allocated names
                for (0..i) |j| {
                    allocator.free(column_names[j]);
                }
                return OrcTableError.OutOfMemory;
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
        const orc_type = self.reader.getColumnType(col_idx) orelse return null;
        return mapOrcToParquetType(orc_type);
    }

    /// Generic helper to convert array from one type to another
    fn convertArray(self: *Self, comptime From: type, comptime To: type, values: []const From) OrcTableError![]To {
        return table_utils.convertArray(self.allocator, From, To, values) catch {
            return OrcTableError.OutOfMemory;
        };
    }

    /// Read int64 column data
    pub fn readInt64Column(self: *Self, col_idx: usize) OrcTableError![]i64 {
        return self.reader.readLongColumn(@intCast(col_idx + 1)) catch {
            return OrcTableError.ReadFailed;
        };
    }

    /// Read int32 column data (by reading int64 and converting)
    pub fn readInt32Column(self: *Self, col_idx: usize) OrcTableError![]i32 {
        const values64 = try self.readInt64Column(col_idx);
        defer self.allocator.free(values64);
        return self.convertArray(i64, i32, values64);
    }

    /// Read float64 column data
    pub fn readFloat64Column(self: *Self, col_idx: usize) OrcTableError![]f64 {
        return self.reader.readDoubleColumn(@intCast(col_idx + 1)) catch {
            return OrcTableError.ReadFailed;
        };
    }

    /// Read float32 column data (by reading float64 and converting)
    pub fn readFloat32Column(self: *Self, col_idx: usize) OrcTableError![]f32 {
        const values64 = try self.readFloat64Column(col_idx);
        defer self.allocator.free(values64);
        return self.convertArray(f64, f32, values64);
    }

    /// Read string column data
    pub fn readStringColumn(self: *Self, col_idx: usize) OrcTableError![][]const u8 {
        return self.reader.readStringColumn(@intCast(col_idx + 1)) catch {
            return OrcTableError.ReadFailed;
        };
    }

    /// Read bool column data
    pub fn readBoolColumn(self: *Self, col_idx: usize) OrcTableError![]bool {
        const values64 = try self.readInt64Column(col_idx);
        defer self.allocator.free(values64);
        return self.convertArray(i64, bool, values64);
    }

    /// Check if path is a valid ORC file
    pub fn isValid(path: []const u8) bool {
        const file = std.fs.cwd().openFile(path, .{}) catch return false;
        defer file.close();

        var header: [3]u8 = undefined;
        _ = file.read(&header) catch return false;

        return std.mem.eql(u8, &header, "ORC");
    }
};

/// Map ORC types to Parquet types
fn mapOrcToParquetType(orc_type: OrcType) ?Type {
    return switch (orc_type) {
        .boolean => .boolean,
        .byte, .short, .int => .int32,
        .long => .int64,
        .float => .float,
        .double => .double,
        .string, .varchar, .char, .binary => .byte_array,
        .date => .int32,
        .timestamp => .int64,
        else => null,
    };
}

test "orc_table: read simple fixture" {
    const allocator = std.testing.allocator;

    // Read file into memory
    const file = std.fs.cwd().openFile("tests/fixtures/simple.orc", .{}) catch |err| {
        std.debug.print("Failed to open ORC file: {}\n", .{err});
        return error.TestFailed;
    };
    defer file.close();
    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch |err| {
        std.debug.print("Failed to read ORC file: {}\n", .{err});
        return error.TestFailed;
    };
    defer allocator.free(data);

    var table = OrcTable.init(allocator, data) catch |err| {
        std.debug.print("Failed to parse ORC file: {}\n", .{err});
        return error.TestFailed;
    };
    defer table.deinit();

    // Check metadata
    try std.testing.expect(table.numColumns() >= 1);
    try std.testing.expect(table.numRows() >= 1);

    // Check column names
    const names = table.getColumnNames();
    try std.testing.expect(names.len >= 1);
}

test "orc_table: read string column" {
    const allocator = std.testing.allocator;

    const file = std.fs.cwd().openFile("tests/fixtures/simple.orc", .{}) catch return error.SkipZigTest;
    defer file.close();
    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch return error.SkipZigTest;
    defer allocator.free(data);

    var table = OrcTable.init(allocator, data) catch return error.SkipZigTest;
    defer table.deinit();

    // Find the string column (name column at index 1)
    const values = table.readStringColumn(1) catch return error.SkipZigTest;
    defer {
        for (values) |v| allocator.free(v);
        allocator.free(values);
    }

    try std.testing.expectEqual(@as(usize, 5), values.len);
    try std.testing.expectEqualStrings("alice", values[0]);
    try std.testing.expectEqualStrings("bob", values[1]);
}

test "orc_table: read float64 column" {
    const allocator = std.testing.allocator;

    const file = std.fs.cwd().openFile("tests/fixtures/simple.orc", .{}) catch return error.SkipZigTest;
    defer file.close();
    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch return error.SkipZigTest;
    defer allocator.free(data);

    var table = OrcTable.init(allocator, data) catch return error.SkipZigTest;
    defer table.deinit();

    // Read the value column (index 2)
    const values = table.readFloat64Column(2) catch return error.SkipZigTest;
    defer allocator.free(values);

    try std.testing.expectEqual(@as(usize, 5), values.len);
    // Values start at 1.1, 2.2, ...
    try std.testing.expectApproxEqAbs(@as(f64, 1.1), values[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 2.2), values[1], 0.01);
}

test "orc_table: column index and type" {
    const allocator = std.testing.allocator;

    const file = std.fs.cwd().openFile("tests/fixtures/simple.orc", .{}) catch return error.SkipZigTest;
    defer file.close();
    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch return error.SkipZigTest;
    defer allocator.free(data);

    var table = OrcTable.init(allocator, data) catch return error.SkipZigTest;
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
