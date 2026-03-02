//! Delta Lake Table wrapper for SQL execution
//!
//! Provides a ParquetTable-like interface for Delta Lake tables by:
//! 1. Parsing Delta transaction log to find data files
//! 2. Reading the underlying Parquet files
//! 3. Exposing unified column read interface

const std = @import("std");
const DeltaReader = @import("lanceql.encoding").DeltaReader;
const ParquetTable = @import("lanceql.parquet_table").ParquetTable;
const format = @import("lanceql.format");
const Type = format.parquet_metadata.Type;

pub const DeltaTableError = error{
    DeltaLogNotFound,
    NoDataFiles,
    ParquetReadFailed,
    OutOfMemory,
    PathTooLong,
    ReadFailed,
    InvalidMagic,
    FileTooSmall,
    MetadataTooLarge,
    InvalidMetadata,
    UnsupportedVersion,
    UnexpectedEndOfData,
    MalformedVarint,
    InvalidType,
    InvalidFieldDelta,
    UnsupportedType,
    ColumnNotFound,
    NoRowGroups,
};

/// High-level Delta Lake table reader
pub const DeltaTable = struct {
    allocator: std.mem.Allocator,
    delta_reader: DeltaReader,
    parquet_table: ParquetTable,
    parquet_data: []const u8, // Owned parquet file data

    const Self = @This();

    /// Open a Delta table from directory path
    pub fn init(allocator: std.mem.Allocator, path: []const u8) DeltaTableError!Self {
        // Parse Delta transaction log
        var delta_reader = DeltaReader.init(allocator, path) catch |err| {
            return switch (err) {
                error.DeltaLogNotFound => DeltaTableError.DeltaLogNotFound,
                error.PathTooLong => DeltaTableError.PathTooLong,
                error.ReadFailed => DeltaTableError.ReadFailed,
                error.OutOfMemory => DeltaTableError.OutOfMemory,
            };
        };
        errdefer delta_reader.deinit();

        // Get the first data file
        if (delta_reader.data_files.len == 0) {
            return DeltaTableError.NoDataFiles;
        }

        // Build full path to parquet file
        var path_buf: [4096]u8 = undefined;
        const parquet_path = std.fmt.bufPrint(&path_buf, "{s}/{s}", .{
            path,
            delta_reader.data_files[0],
        }) catch {
            return DeltaTableError.PathTooLong;
        };

        // Read parquet file
        const file = std.fs.cwd().openFile(parquet_path, .{}) catch {
            return DeltaTableError.ParquetReadFailed;
        };
        defer file.close();

        const parquet_data = file.readToEndAlloc(allocator, 500 * 1024 * 1024) catch {
            return DeltaTableError.OutOfMemory;
        };
        errdefer allocator.free(parquet_data);

        // Initialize ParquetTable
        const parquet_table = ParquetTable.init(allocator, parquet_data) catch |err| {
            return switch (err) {
                error.InvalidMagic => DeltaTableError.InvalidMagic,
                error.FileTooSmall => DeltaTableError.FileTooSmall,
                error.MetadataTooLarge => DeltaTableError.MetadataTooLarge,
                error.InvalidMetadata => DeltaTableError.InvalidMetadata,
                error.OutOfMemory => DeltaTableError.OutOfMemory,
                error.UnexpectedEndOfData => DeltaTableError.UnexpectedEndOfData,
                error.MalformedVarint => DeltaTableError.MalformedVarint,
                error.InvalidType => DeltaTableError.InvalidType,
                error.InvalidFieldDelta => DeltaTableError.InvalidFieldDelta,
                else => DeltaTableError.ParquetReadFailed,
            };
        };

        return Self{
            .allocator = allocator,
            .delta_reader = delta_reader,
            .parquet_table = parquet_table,
            .parquet_data = parquet_data,
        };
    }

    pub fn deinit(self: *Self) void {
        self.parquet_table.deinit();
        self.allocator.free(self.parquet_data);
        self.delta_reader.deinit();
    }

    // =========================================================================
    // ParquetTable-compatible interface (delegate to underlying ParquetTable)
    // =========================================================================

    /// Get number of columns
    pub fn numColumns(self: Self) u32 {
        return self.parquet_table.numColumns();
    }

    /// Get number of rows
    pub fn numRows(self: Self) usize {
        return self.parquet_table.numRows();
    }

    /// Get column names
    pub fn getColumnNames(self: Self) [][]const u8 {
        return self.parquet_table.getColumnNames();
    }

    /// Get column index by name
    pub fn columnIndex(self: Self, name: []const u8) ?usize {
        return self.parquet_table.columnIndex(name);
    }

    /// Get column type
    pub fn getColumnType(self: Self, col_idx: usize) ?Type {
        return self.parquet_table.getColumnType(col_idx);
    }

    /// Read int64 column data
    pub fn readInt64Column(self: *Self, col_idx: usize) DeltaTableError![]i64 {
        return self.parquet_table.readInt64Column(col_idx) catch |err| {
            return switch (err) {
                error.NoRowGroups => DeltaTableError.NoRowGroups,
                error.OutOfMemory => DeltaTableError.OutOfMemory,
                else => DeltaTableError.ParquetReadFailed,
            };
        };
    }

    /// Read int32 column data
    pub fn readInt32Column(self: *Self, col_idx: usize) DeltaTableError![]i32 {
        return self.parquet_table.readInt32Column(col_idx) catch |err| {
            return switch (err) {
                error.NoRowGroups => DeltaTableError.NoRowGroups,
                error.OutOfMemory => DeltaTableError.OutOfMemory,
                else => DeltaTableError.ParquetReadFailed,
            };
        };
    }

    /// Read float64 column data
    pub fn readFloat64Column(self: *Self, col_idx: usize) DeltaTableError![]f64 {
        return self.parquet_table.readFloat64Column(col_idx) catch |err| {
            return switch (err) {
                error.NoRowGroups => DeltaTableError.NoRowGroups,
                error.OutOfMemory => DeltaTableError.OutOfMemory,
                else => DeltaTableError.ParquetReadFailed,
            };
        };
    }

    /// Read float32 column data
    pub fn readFloat32Column(self: *Self, col_idx: usize) DeltaTableError![]f32 {
        return self.parquet_table.readFloat32Column(col_idx) catch |err| {
            return switch (err) {
                error.NoRowGroups => DeltaTableError.NoRowGroups,
                error.OutOfMemory => DeltaTableError.OutOfMemory,
                else => DeltaTableError.ParquetReadFailed,
            };
        };
    }

    /// Read string column data
    pub fn readStringColumn(self: *Self, col_idx: usize) DeltaTableError![][]const u8 {
        return self.parquet_table.readStringColumn(col_idx) catch |err| {
            return switch (err) {
                error.NoRowGroups => DeltaTableError.NoRowGroups,
                error.OutOfMemory => DeltaTableError.OutOfMemory,
                else => DeltaTableError.ParquetReadFailed,
            };
        };
    }

    /// Read bool column data
    pub fn readBoolColumn(self: *Self, col_idx: usize) DeltaTableError![]bool {
        return self.parquet_table.readBoolColumn(col_idx) catch |err| {
            return switch (err) {
                error.NoRowGroups => DeltaTableError.NoRowGroups,
                error.OutOfMemory => DeltaTableError.OutOfMemory,
                else => DeltaTableError.ParquetReadFailed,
            };
        };
    }

    /// Check if path is a valid Delta table
    pub fn isValid(path: []const u8) bool {
        return DeltaReader.isValid(path);
    }
};

test "delta_table: read simple fixture" {
    const allocator = std.testing.allocator;

    var table = DeltaTable.init(allocator, "tests/fixtures/simple.delta") catch |err| {
        std.debug.print("Failed to open Delta table: {}\n", .{err});
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
