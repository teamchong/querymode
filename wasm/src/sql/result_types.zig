//! Result Types - Data structures for SQL query results
//!
//! Contains Result, CachedColumn, JoinedData types used by the SQL executor.
//! Also includes LanceColumnType for unified type detection across Lance tables.

const std = @import("std");
const Table = @import("edgeq.table").Table;

/// Lance column types - unified type detection for Lance table logical types
pub const LanceColumnType = enum {
    timestamp_ns,
    timestamp_us,
    timestamp_ms,
    timestamp_s,
    date32,
    date64,
    int32,
    float32,
    bool_,
    int64,
    float64,
    string,
    unsupported,

    /// Detect column type from Lance logical_type string
    /// Precise type detection (order matters - check specific before general)
    pub fn fromLogicalType(logical_type: []const u8) LanceColumnType {
        // Timestamp types (check before generic "int" matches)
        if (std.mem.indexOf(u8, logical_type, "timestamp[ns") != null) return .timestamp_ns;
        if (std.mem.indexOf(u8, logical_type, "timestamp[us") != null) return .timestamp_us;
        if (std.mem.indexOf(u8, logical_type, "timestamp[ms") != null) return .timestamp_ms;
        if (std.mem.indexOf(u8, logical_type, "timestamp[s") != null) return .timestamp_s;
        if (std.mem.indexOf(u8, logical_type, "date32") != null) return .date32;
        if (std.mem.indexOf(u8, logical_type, "date64") != null) return .date64;
        // Explicit int32
        if (std.mem.eql(u8, logical_type, "int32")) return .int32;
        // float or float32
        if (std.mem.eql(u8, logical_type, "float") or std.mem.indexOf(u8, logical_type, "float32") != null) return .float32;
        // bool or boolean
        if (std.mem.eql(u8, logical_type, "bool") or std.mem.indexOf(u8, logical_type, "boolean") != null) return .bool_;
        // Default integers (int, int64, integer)
        if (std.mem.indexOf(u8, logical_type, "int") != null) return .int64;
        // double
        if (std.mem.indexOf(u8, logical_type, "double") != null) return .float64;
        // Strings (utf8 or string)
        if (std.mem.indexOf(u8, logical_type, "utf8") != null or std.mem.indexOf(u8, logical_type, "string") != null) return .string;
        return .unsupported;
    }

};

/// Query result in columnar format
pub const Result = struct {
    columns: []Column,
    row_count: usize,
    allocator: std.mem.Allocator,
    /// If false, column data is owned by executor's cache and should not be freed here
    owns_data: bool = true,

    pub const Column = struct {
        name: []const u8,
        data: ColumnData,
    };

    pub const ColumnData = union(enum) {
        int64: []i64,
        int32: []i32,
        float64: []f64,
        float32: []f32,
        bool_: []bool,
        string: [][]const u8,
        // Timestamp types (all stored as integers, semantic meaning differs)
        timestamp_s: []i64, // seconds since epoch
        timestamp_ms: []i64, // milliseconds since epoch
        timestamp_us: []i64, // microseconds since epoch
        timestamp_ns: []i64, // nanoseconds since epoch
        date32: []i32, // days since epoch
        date64: []i64, // milliseconds since epoch

        /// Get the length of the column data
        pub fn len(self: ColumnData) usize {
            return switch (self) {
                inline else => |data| data.len,
            };
        }

        /// Free the column data using the provided allocator
        pub fn free(self: ColumnData, allocator: std.mem.Allocator) void {
            switch (self) {
                .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| allocator.free(data),
                .int32, .date32 => |data| allocator.free(data),
                .float64 => |data| allocator.free(data),
                .float32 => |data| allocator.free(data),
                .bool_ => |data| allocator.free(data),
                .string => |data| {
                    for (data) |str| {
                        allocator.free(str);
                    }
                    allocator.free(data);
                },
            }
        }
    };

    pub fn deinit(self: *Result) void {
        if (self.owns_data) {
            for (self.columns) |col| {
                col.data.free(self.allocator);
            }
        }
        self.allocator.free(self.columns);
    }
};

/// Cached column data
pub const CachedColumn = union(enum) {
    int64: []i64,
    int32: []i32,
    float64: []f64,
    float32: []f32,
    bool_: []bool,
    string: [][]const u8,
    // Timestamp types
    timestamp_s: []i64,
    timestamp_ms: []i64,
    timestamp_us: []i64,
    timestamp_ns: []i64,
    date32: []i32,
    date64: []i64,

    /// Get the length of the column data
    pub fn len(self: CachedColumn) usize {
        return switch (self) {
            inline else => |data| data.len,
        };
    }

    /// Free the column data using the provided allocator
    pub fn free(self: CachedColumn, allocator: std.mem.Allocator) void {
        switch (self) {
            .int64, .timestamp_s, .timestamp_ms, .timestamp_us, .timestamp_ns, .date64 => |data| allocator.free(data),
            .int32, .date32 => |data| allocator.free(data),
            .float64 => |data| allocator.free(data),
            .float32 => |data| allocator.free(data),
            .bool_ => |data| allocator.free(data),
            .string => |data| {
                for (data) |str| {
                    allocator.free(str);
                }
                allocator.free(data);
            },
        }
    }
};

/// Materialized data from a JOIN operation
pub const JoinedData = struct {
    /// Column data by name (qualified with table alias if present)
    columns: std.StringHashMap(CachedColumn),
    /// Column names in order
    column_names: [][]const u8,
    /// Number of rows in the joined result
    row_count: usize,
    /// Allocator for cleanup
    allocator: std.mem.Allocator,
    /// Left table pointer (for schema access)
    left_table: *Table,

    pub fn deinit(self: *JoinedData) void {
        // Free column data
        var iter = self.columns.valueIterator();
        while (iter.next()) |col| {
            col.free(self.allocator);
        }
        self.columns.deinit();
        // Free column names
        for (self.column_names) |name| {
            self.allocator.free(name);
        }
        self.allocator.free(self.column_names);
    }
};

/// Active table source for query execution
pub const TableSource = union(enum) {
    /// Direct table
    direct: *Table,
    /// Joined table with materialized data
    joined: *JoinedData,

    pub fn getTable(self: TableSource) *Table {
        return switch (self) {
            .direct => |t| t,
            .joined => |jd| jd.left_table,
        };
    }
};

// ============================================================================
// Streaming Result Types (Late Materialization Support)
// ============================================================================

/// Specification for a column to be materialized
pub const ColumnSpec = struct {
    /// Physical column index in the table
    col_idx: u32,
    /// Column name for output
    name: []const u8,
    /// Column data type
    col_type: LanceColumnType,
};

/// A single batch of materialized rows
pub const StreamingBatch = struct {
    /// Materialized column data for this batch
    columns: []Result.Column,
    /// Number of rows in this batch
    row_count: usize,
    /// Allocator used for this batch
    allocator: std.mem.Allocator,

    pub fn deinit(self: *StreamingBatch) void {
        for (self.columns) |col| {
            col.data.free(self.allocator);
        }
        self.allocator.free(self.columns);
    }
};

/// Iterator-based streaming result for late materialization.
///
/// This enables memory-efficient query execution by:
/// 1. Keeping only the row_ids in memory (4 bytes per row)
/// 2. Materializing columns on-demand in batches
/// 3. Allowing batch-by-batch streaming to output (HTTP chunked transfer)
///
/// Memory profile for 1M rows:
/// - row_ids: 4MB
/// - batch (1000 rows): ~1MB
/// - Total peak: ~5-10MB instead of loading all columns
pub const StreamingResult = struct {
    /// Row indices that passed filtering, in output order
    row_ids: []const u32,
    /// Columns to materialize (from SELECT)
    select_columns: []ColumnSpec,
    /// Batch size for materialization
    batch_size: usize,
    /// Current position in row_ids
    cursor: usize,
    /// Allocator for batch allocations
    allocator: std.mem.Allocator,
    /// Whether this result owns the row_ids array
    owns_row_ids: bool,

    const Self = @This();

    /// Default batch size (1000 rows)
    pub const DEFAULT_BATCH_SIZE: usize = 1000;

    /// Initialize a streaming result.
    ///
    /// Parameters:
    ///   - row_ids: Row indices that passed filtering (takes ownership if owns_row_ids=true)
    ///   - select_columns: Column specifications for materialization
    ///   - batch_size: Number of rows to materialize per batch
    ///   - allocator: Allocator for batch allocations
    pub fn init(
        row_ids: []const u32,
        select_columns: []ColumnSpec,
        batch_size: usize,
        allocator: std.mem.Allocator,
        owns_row_ids: bool,
    ) Self {
        return Self{
            .row_ids = row_ids,
            .select_columns = select_columns,
            .batch_size = if (batch_size == 0) DEFAULT_BATCH_SIZE else batch_size,
            .cursor = 0,
            .allocator = allocator,
            .owns_row_ids = owns_row_ids,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.owns_row_ids) {
            self.allocator.free(self.row_ids);
        }
        for (self.select_columns) |col| {
            self.allocator.free(col.name);
        }
        self.allocator.free(self.select_columns);
    }

    /// Get total number of rows in the result
    pub fn totalRows(self: Self) usize {
        return self.row_ids.len;
    }

    /// Get number of remaining rows
    pub fn remainingRows(self: Self) usize {
        if (self.cursor >= self.row_ids.len) return 0;
        return self.row_ids.len - self.cursor;
    }

    /// Check if there are more batches
    pub fn hasMore(self: Self) bool {
        return self.cursor < self.row_ids.len;
    }

    /// Get the next batch of row indices to materialize.
    /// Returns null if no more batches.
    ///
    /// Caller should then use the returned indices with
    /// LazyTable.readInt64ColumnAtIndices() etc. to materialize each column.
    pub fn nextBatchIndices(self: *Self) ?[]const u32 {
        if (self.cursor >= self.row_ids.len) {
            return null;
        }

        const start = self.cursor;
        const end = @min(self.cursor + self.batch_size, self.row_ids.len);
        self.cursor = end;

        return self.row_ids[start..end];
    }

    /// Reset cursor to beginning for re-iteration
    pub fn reset(self: *Self) void {
        self.cursor = 0;
    }

    /// Get progress as a fraction (0.0 to 1.0)
    pub fn progress(self: Self) f64 {
        if (self.row_ids.len == 0) return 1.0;
        return @as(f64, @floatFromInt(self.cursor)) / @as(f64, @floatFromInt(self.row_ids.len));
    }

    /// Get approximate memory usage of the row_ids array
    pub fn rowIdsMemoryBytes(self: Self) usize {
        return self.row_ids.len * @sizeOf(u32);
    }
};

/// Statistics for streaming execution
pub const StreamingStats = struct {
    /// Total rows in result
    total_rows: usize,
    /// Number of batches materialized
    batches_materialized: usize,
    /// Total bytes read for materialization
    bytes_read: u64,
    /// Number of HTTP range requests made
    range_requests: usize,

    pub fn init() StreamingStats {
        return .{
            .total_rows = 0,
            .batches_materialized = 0,
            .bytes_read = 0,
            .range_requests = 0,
        };
    }
};
