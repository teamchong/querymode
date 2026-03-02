//! XLSX Table wrapper for SQL execution
//!
//! Provides a ParquetTable-like interface for Excel files by:
//! 1. Parsing XLSX format to read cell data
//! 2. Using first row as column headers
//! 3. Exposing unified column read interface

const std = @import("std");
const XlsxReader = @import("lanceql.encoding").XlsxReader;
const CellValue = @import("lanceql.encoding").CellValue;
const CellType = @import("lanceql.encoding").CellType;
const format = @import("lanceql.format");
const Type = format.parquet_metadata.Type;
const table_utils = @import("lanceql.table_utils");

pub const XlsxTableError = error{
    InvalidXlsxFile,
    InvalidZipMagic,
    InvalidZipFile,
    FileNotFound,
    UnsupportedCompression,
    DeflateNotSupported,
    NoDataFiles,
    OutOfMemory,
    PathTooLong,
    ReadFailed,
    ColumnNotFound,
    NoRowGroups,
    NoHeaderRow,
};

/// High-level XLSX table reader
pub const XlsxTable = struct {
    allocator: std.mem.Allocator,
    reader: XlsxReader,
    column_names: [][]const u8, // Extracted from first row
    has_header: bool,

    const Self = @This();

    /// Open an XLSX file from in-memory data
    pub fn init(allocator: std.mem.Allocator, data: []const u8) XlsxTableError!Self {
        // Initialize reader
        var reader = XlsxReader.init(allocator, data) catch {
            return XlsxTableError.InvalidXlsxFile;
        };
        errdefer reader.deinit();

        // Extract column names from first row
        const num_cols = reader.columnCount();
        var column_names = allocator.alloc([]const u8, num_cols) catch {
            return XlsxTableError.OutOfMemory;
        };
        errdefer allocator.free(column_names);

        var has_header = false;

        // Try to use first row as headers
        if (reader.rowCount() > 0) {
            var all_strings = true;
            for (0..num_cols) |col| {
                const cell = reader.getCell(0, col);
                if (cell) |c| {
                    switch (c) {
                        .string, .inline_string, .shared_string => {},
                        .number => {
                            // Numbers in header row - likely not a header
                            all_strings = false;
                        },
                        else => {},
                    }
                }
            }
            has_header = all_strings;
        }

        // Generate column names
        for (0..num_cols) |col| {
            if (has_header) {
                const cell = reader.getCell(0, col);
                if (cell) |c| {
                    switch (c) {
                        .string => |s| {
                            column_names[col] = allocator.dupe(u8, s) catch {
                                for (0..col) |j| allocator.free(column_names[j]);
                                return XlsxTableError.OutOfMemory;
                            };
                            continue;
                        },
                        .inline_string => |s| {
                            column_names[col] = allocator.dupe(u8, s) catch {
                                for (0..col) |j| allocator.free(column_names[j]);
                                return XlsxTableError.OutOfMemory;
                            };
                            continue;
                        },
                        else => {},
                    }
                }
            }

            // Default column name
            var name_buf: [16]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "col_{d}", .{col}) catch "col";
            column_names[col] = allocator.dupe(u8, name) catch {
                for (0..col) |j| allocator.free(column_names[j]);
                return XlsxTableError.OutOfMemory;
            };
        }

        return Self{
            .allocator = allocator,
            .reader = reader,
            .column_names = column_names,
            .has_header = has_header,
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

    /// Get number of rows (excluding header)
    pub fn numRows(self: Self) usize {
        const total = self.reader.rowCount();
        if (self.has_header and total > 0) {
            return total - 1;
        }
        return total;
    }

    /// Get column names
    pub fn getColumnNames(self: Self) [][]const u8 {
        return self.column_names;
    }

    /// Get column index by name
    pub fn columnIndex(self: Self, name: []const u8) ?usize {
        return table_utils.findColumnIndex(self.column_names, name);
    }

    /// Get column type (inferred from first data cell)
    pub fn getColumnType(self: Self, col_idx: usize) ?Type {
        // Look at first data row to infer type
        const data_row: usize = if (self.has_header) 1 else 0;
        const cell = self.reader.getCell(data_row, col_idx);
        if (cell) |c| {
            return switch (c) {
                .number => .double,
                .string, .inline_string, .shared_string => .byte_array,
                .boolean => .boolean,
                else => null,
            };
        }
        return null;
    }

    /// Get the starting row for data (after header if present)
    fn dataRowStart(self: Self) usize {
        return if (self.has_header) 1 else 0;
    }

    /// Read int64 column data
    pub fn readInt64Column(self: *Self, col_idx: usize) XlsxTableError![]i64 {
        const start_row = self.dataRowStart();
        const num_data_rows = self.numRows();

        var values = self.allocator.alloc(i64, num_data_rows) catch {
            return XlsxTableError.OutOfMemory;
        };
        errdefer self.allocator.free(values);

        for (0..num_data_rows) |i| {
            const cell = self.reader.getCell(start_row + i, col_idx);
            if (cell) |c| {
                switch (c) {
                    .number => |n| values[i] = @intFromFloat(n),
                    else => values[i] = 0,
                }
            } else {
                values[i] = 0;
            }
        }

        return values;
    }

    /// Generic helper to convert array from one type to another
    fn convertArray(self: *Self, comptime From: type, comptime To: type, values: []const From) XlsxTableError![]To {
        return table_utils.convertArray(self.allocator, From, To, values) catch {
            return XlsxTableError.OutOfMemory;
        };
    }

    /// Read int32 column data
    pub fn readInt32Column(self: *Self, col_idx: usize) XlsxTableError![]i32 {
        const values64 = try self.readInt64Column(col_idx);
        defer self.allocator.free(values64);
        return self.convertArray(i64, i32, values64);
    }

    /// Read float64 column data
    pub fn readFloat64Column(self: *Self, col_idx: usize) XlsxTableError![]f64 {
        const start_row = self.dataRowStart();
        const num_data_rows = self.numRows();

        var values = self.allocator.alloc(f64, num_data_rows) catch {
            return XlsxTableError.OutOfMemory;
        };
        errdefer self.allocator.free(values);

        for (0..num_data_rows) |i| {
            const cell = self.reader.getCell(start_row + i, col_idx);
            if (cell) |c| {
                switch (c) {
                    .number => |n| values[i] = n,
                    else => values[i] = 0,
                }
            } else {
                values[i] = 0;
            }
        }

        return values;
    }

    /// Read float32 column data
    pub fn readFloat32Column(self: *Self, col_idx: usize) XlsxTableError![]f32 {
        const values64 = try self.readFloat64Column(col_idx);
        defer self.allocator.free(values64);
        return self.convertArray(f64, f32, values64);
    }

    /// Read string column data
    pub fn readStringColumn(self: *Self, col_idx: usize) XlsxTableError![][]const u8 {
        const start_row = self.dataRowStart();
        const num_data_rows = self.numRows();

        var values = self.allocator.alloc([]const u8, num_data_rows) catch {
            return XlsxTableError.OutOfMemory;
        };
        errdefer self.allocator.free(values);

        for (0..num_data_rows) |i| {
            const cell = self.reader.getCell(start_row + i, col_idx);
            if (cell) |c| {
                const str = switch (c) {
                    .string => |s| s,
                    .inline_string => |s| s,
                    .number => |n| blk: {
                        // Convert number to string
                        var buf: [64]u8 = undefined;
                        const formatted = std.fmt.bufPrint(&buf, "{d}", .{n}) catch "";
                        break :blk self.allocator.dupe(u8, formatted) catch "";
                    },
                    else => "",
                };
                values[i] = self.allocator.dupe(u8, str) catch {
                    for (0..i) |j| self.allocator.free(values[j]);
                    return XlsxTableError.OutOfMemory;
                };
            } else {
                values[i] = self.allocator.dupe(u8, "") catch {
                    for (0..i) |j| self.allocator.free(values[j]);
                    return XlsxTableError.OutOfMemory;
                };
            }
        }

        return values;
    }

    /// Read bool column data
    pub fn readBoolColumn(self: *Self, col_idx: usize) XlsxTableError![]bool {
        const start_row = self.dataRowStart();
        const num_data_rows = self.numRows();

        var values = self.allocator.alloc(bool, num_data_rows) catch {
            return XlsxTableError.OutOfMemory;
        };
        errdefer self.allocator.free(values);

        for (0..num_data_rows) |i| {
            const cell = self.reader.getCell(start_row + i, col_idx);
            if (cell) |c| {
                switch (c) {
                    .boolean => |b| values[i] = b,
                    .number => |n| values[i] = n != 0,
                    else => values[i] = false,
                }
            } else {
                values[i] = false;
            }
        }

        return values;
    }

    /// Check if path is a valid XLSX file
    pub fn isValid(path: []const u8) bool {
        const file = std.fs.cwd().openFile(path, .{}) catch return false;
        defer file.close();

        var header: [4]u8 = undefined;
        _ = file.read(&header) catch return false;

        // Check for ZIP magic (PK\x03\x04)
        return std.mem.eql(u8, &header, "PK\x03\x04");
    }
};

test "xlsx_table: read simple fixture" {
    const allocator = std.testing.allocator;

    // Read file into memory
    const file = std.fs.cwd().openFile("tests/fixtures/simple.xlsx", .{}) catch |err| {
        std.debug.print("Failed to open XLSX file: {}\n", .{err});
        return error.TestFailed;
    };
    defer file.close();
    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch |err| {
        std.debug.print("Failed to read XLSX file: {}\n", .{err});
        return error.TestFailed;
    };
    defer allocator.free(data);

    var table = XlsxTable.init(allocator, data) catch |err| {
        std.debug.print("Failed to parse XLSX file: {}\n", .{err});
        return error.TestFailed;
    };
    defer table.deinit();

    // Check metadata
    try std.testing.expect(table.numColumns() >= 1);

    // Check column names
    const names = table.getColumnNames();
    try std.testing.expect(names.len >= 1);
}

test "xlsx_table: read string column" {
    const allocator = std.testing.allocator;

    const file = std.fs.cwd().openFile("tests/fixtures/simple.xlsx", .{}) catch return error.SkipZigTest;
    defer file.close();
    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch return error.SkipZigTest;
    defer allocator.free(data);

    var table = XlsxTable.init(allocator, data) catch return error.SkipZigTest;
    defer table.deinit();

    // Find the string column (name column at index 1)
    const values = table.readStringColumn(1) catch return error.SkipZigTest;
    defer {
        for (values) |v| allocator.free(v);
        allocator.free(values);
    }

    // Verify we got string values
    try std.testing.expect(values.len > 0);
    try std.testing.expectEqualStrings("alice", values[0]);
}

test "xlsx_table: read float64 column" {
    const allocator = std.testing.allocator;

    const file = std.fs.cwd().openFile("tests/fixtures/simple.xlsx", .{}) catch return error.SkipZigTest;
    defer file.close();
    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch return error.SkipZigTest;
    defer allocator.free(data);

    var table = XlsxTable.init(allocator, data) catch return error.SkipZigTest;
    defer table.deinit();

    // Read the value column (index 2)
    const values = table.readFloat64Column(2) catch return error.SkipZigTest;
    defer allocator.free(values);

    // Verify we got float values
    try std.testing.expect(values.len > 0);
    // Values start at 1.1, 2.2, ...
    try std.testing.expectApproxEqAbs(@as(f64, 1.1), values[0], 0.01);
}

test "xlsx_table: column index and type" {
    const allocator = std.testing.allocator;

    const file = std.fs.cwd().openFile("tests/fixtures/simple.xlsx", .{}) catch return error.SkipZigTest;
    defer file.close();
    const data = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch return error.SkipZigTest;
    defer allocator.free(data);

    var table = XlsxTable.init(allocator, data) catch return error.SkipZigTest;
    defer table.deinit();

    // Test columnIndex
    try std.testing.expectEqual(@as(?usize, 0), table.columnIndex("id"));
    try std.testing.expectEqual(@as(?usize, 1), table.columnIndex("name"));
    try std.testing.expectEqual(@as(?usize, 2), table.columnIndex("value"));
    try std.testing.expectEqual(@as(?usize, null), table.columnIndex("nonexistent"));

    // Test column types (XLSX infers from first data cell)
    try std.testing.expectEqual(Type.double, table.getColumnType(0).?);
    try std.testing.expectEqual(Type.byte_array, table.getColumnType(1).?);
    try std.testing.expectEqual(Type.double, table.getColumnType(2).?);
}
