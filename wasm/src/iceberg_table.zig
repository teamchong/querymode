//! Iceberg Table wrapper for SQL execution
//!
//! Provides a ParquetTable-like interface for Iceberg tables by:
//! 1. Parsing Iceberg metadata to find data files
//! 2. Reading the underlying Parquet files
//! 3. Exposing unified column read interface
//!
//! Note: Requires Iceberg table with snapshots containing data files.

const std = @import("std");
const IcebergReader = @import("lanceql.encoding").IcebergReader;
const ParquetTable = @import("lanceql.parquet_table").ParquetTable;
const format = @import("lanceql.format");
const Type = format.parquet_metadata.Type;

pub const IcebergTableError = error{
    IcebergMetadataNotFound,
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
    NoSnapshots,
};

/// High-level Iceberg table reader
pub const IcebergTable = struct {
    allocator: std.mem.Allocator,
    iceberg_reader: IcebergReader,
    parquet_table: ?ParquetTable,
    parquet_data: ?[]const u8, // Owned parquet file data

    const Self = @This();

    /// Open an Iceberg table from directory path
    pub fn init(allocator: std.mem.Allocator, path: []const u8) IcebergTableError!Self {
        // Parse Iceberg metadata
        var iceberg_reader = IcebergReader.init(allocator, path) catch |err| {
            return switch (err) {
                error.IcebergMetadataNotFound => IcebergTableError.IcebergMetadataNotFound,
                error.OutOfMemory => IcebergTableError.OutOfMemory,
            };
        };
        errdefer iceberg_reader.deinit();

        // Check if we have snapshots with data
        if (iceberg_reader.current_snapshot_id < 0) {
            // No snapshots yet - table is empty
            return Self{
                .allocator = allocator,
                .iceberg_reader = iceberg_reader,
                .parquet_table = null,
                .parquet_data = null,
            };
        }

        // Look for parquet files in data directory
        var path_buf: [4096]u8 = undefined;
        const data_path = std.fmt.bufPrint(&path_buf, "{s}/data", .{path}) catch {
            return IcebergTableError.PathTooLong;
        };

        var data_dir = std.fs.cwd().openDir(data_path, .{ .iterate = true }) catch {
            // No data directory - table is empty
            return Self{
                .allocator = allocator,
                .iceberg_reader = iceberg_reader,
                .parquet_table = null,
                .parquet_data = null,
            };
        };
        defer data_dir.close();

        // Find first parquet file
        var iter = data_dir.iterate();
        while (iter.next() catch null) |entry| {
            if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".parquet")) {
                // Build full path
                var full_path_buf: [4096]u8 = undefined;
                const full_path = std.fmt.bufPrint(&full_path_buf, "{s}/data/{s}", .{ path, entry.name }) catch {
                    return IcebergTableError.PathTooLong;
                };

                // Read parquet file
                const file = std.fs.cwd().openFile(full_path, .{}) catch {
                    return IcebergTableError.ParquetReadFailed;
                };
                defer file.close();

                const parquet_data = file.readToEndAlloc(allocator, 500 * 1024 * 1024) catch {
                    return IcebergTableError.OutOfMemory;
                };
                errdefer allocator.free(parquet_data);

                // Initialize ParquetTable
                const parquet_table = ParquetTable.init(allocator, parquet_data) catch |err| {
                    return switch (err) {
                        error.InvalidMagic => IcebergTableError.InvalidMagic,
                        error.FileTooSmall => IcebergTableError.FileTooSmall,
                        error.MetadataTooLarge => IcebergTableError.MetadataTooLarge,
                        error.InvalidMetadata => IcebergTableError.InvalidMetadata,
                        error.OutOfMemory => IcebergTableError.OutOfMemory,
                        error.UnexpectedEndOfData => IcebergTableError.UnexpectedEndOfData,
                        error.MalformedVarint => IcebergTableError.MalformedVarint,
                        error.InvalidType => IcebergTableError.InvalidType,
                        error.InvalidFieldDelta => IcebergTableError.InvalidFieldDelta,
                        else => IcebergTableError.ParquetReadFailed,
                    };
                };

                return Self{
                    .allocator = allocator,
                    .iceberg_reader = iceberg_reader,
                    .parquet_table = parquet_table,
                    .parquet_data = parquet_data,
                };
            }
        }

        // No parquet files found
        return Self{
            .allocator = allocator,
            .iceberg_reader = iceberg_reader,
            .parquet_table = null,
            .parquet_data = null,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.parquet_table) |*pq| {
            pq.deinit();
        }
        if (self.parquet_data) |data| {
            self.allocator.free(data);
        }
        self.iceberg_reader.deinit();
    }

    // =========================================================================
    // ParquetTable-compatible interface (delegate to underlying ParquetTable)
    // =========================================================================

    /// Get number of columns
    pub fn numColumns(self: Self) u32 {
        if (self.parquet_table) |pq| {
            return pq.numColumns();
        }
        return @intCast(self.iceberg_reader.num_columns);
    }

    /// Get number of rows
    pub fn numRows(self: Self) usize {
        if (self.parquet_table) |pq| {
            return pq.numRows();
        }
        return 0; // Empty table
    }

    /// Get column names
    pub fn getColumnNames(self: Self) [][]const u8 {
        if (self.parquet_table) |pq| {
            return pq.getColumnNames();
        }
        return &.{}; // Empty table
    }

    /// Get column index by name
    pub fn columnIndex(self: Self, name: []const u8) ?usize {
        if (self.parquet_table) |pq| {
            return pq.columnIndex(name);
        }
        return null;
    }

    /// Get column type
    pub fn getColumnType(self: Self, col_idx: usize) ?Type {
        if (self.parquet_table) |pq| {
            return pq.getColumnType(col_idx);
        }
        return null;
    }

    /// Read int64 column data
    pub fn readInt64Column(self: *Self, col_idx: usize) IcebergTableError![]i64 {
        if (self.parquet_table) |*pq| {
            return pq.readInt64Column(col_idx) catch |err| {
                return switch (err) {
                    error.NoRowGroups => IcebergTableError.NoRowGroups,
                    error.OutOfMemory => IcebergTableError.OutOfMemory,
                    else => IcebergTableError.ParquetReadFailed,
                };
            };
        }
        return IcebergTableError.NoDataFiles;
    }

    /// Read int32 column data
    pub fn readInt32Column(self: *Self, col_idx: usize) IcebergTableError![]i32 {
        if (self.parquet_table) |*pq| {
            return pq.readInt32Column(col_idx) catch |err| {
                return switch (err) {
                    error.NoRowGroups => IcebergTableError.NoRowGroups,
                    error.OutOfMemory => IcebergTableError.OutOfMemory,
                    else => IcebergTableError.ParquetReadFailed,
                };
            };
        }
        return IcebergTableError.NoDataFiles;
    }

    /// Read float64 column data
    pub fn readFloat64Column(self: *Self, col_idx: usize) IcebergTableError![]f64 {
        if (self.parquet_table) |*pq| {
            return pq.readFloat64Column(col_idx) catch |err| {
                return switch (err) {
                    error.NoRowGroups => IcebergTableError.NoRowGroups,
                    error.OutOfMemory => IcebergTableError.OutOfMemory,
                    else => IcebergTableError.ParquetReadFailed,
                };
            };
        }
        return IcebergTableError.NoDataFiles;
    }

    /// Read float32 column data
    pub fn readFloat32Column(self: *Self, col_idx: usize) IcebergTableError![]f32 {
        if (self.parquet_table) |*pq| {
            return pq.readFloat32Column(col_idx) catch |err| {
                return switch (err) {
                    error.NoRowGroups => IcebergTableError.NoRowGroups,
                    error.OutOfMemory => IcebergTableError.OutOfMemory,
                    else => IcebergTableError.ParquetReadFailed,
                };
            };
        }
        return IcebergTableError.NoDataFiles;
    }

    /// Read string column data
    pub fn readStringColumn(self: *Self, col_idx: usize) IcebergTableError![][]const u8 {
        if (self.parquet_table) |*pq| {
            return pq.readStringColumn(col_idx) catch |err| {
                return switch (err) {
                    error.NoRowGroups => IcebergTableError.NoRowGroups,
                    error.OutOfMemory => IcebergTableError.OutOfMemory,
                    else => IcebergTableError.ParquetReadFailed,
                };
            };
        }
        return IcebergTableError.NoDataFiles;
    }

    /// Read bool column data
    pub fn readBoolColumn(self: *Self, col_idx: usize) IcebergTableError![]bool {
        if (self.parquet_table) |*pq| {
            return pq.readBoolColumn(col_idx) catch |err| {
                return switch (err) {
                    error.NoRowGroups => IcebergTableError.NoRowGroups,
                    error.OutOfMemory => IcebergTableError.OutOfMemory,
                    else => IcebergTableError.ParquetReadFailed,
                };
            };
        }
        return IcebergTableError.NoDataFiles;
    }

    /// Check if path is a valid Iceberg table
    pub fn isValid(path: []const u8) bool {
        return IcebergReader.isValid(path);
    }
};

test "iceberg_table: init empty fixture" {
    const allocator = std.testing.allocator;

    // The test fixture has no data, so this tests the empty table case
    var table = IcebergTable.init(allocator, "tests/fixtures/simple.iceberg") catch |err| {
        std.debug.print("Failed to open Iceberg table: {}\n", .{err});
        return error.TestFailed;
    };
    defer table.deinit();

    // Check metadata
    try std.testing.expectEqual(@as(u32, 3), table.numColumns());
    // Empty table has 0 rows
    try std.testing.expectEqual(@as(usize, 0), table.numRows());
}
