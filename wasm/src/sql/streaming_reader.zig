//! Streaming Column Reader for SQL Executor
//!
//! Implements ve.ColumnReader interface for streaming query execution.
//! Provides batch reads from Lance tables for VECTOR_SIZE (2048) processing.

const std = @import("std");
const Table = @import("lanceql.table").Table;
const ve = @import("lanceql.vector_engine");

/// TableColumnReader - Implements ve.ColumnReader for streaming SQL execution
///
/// Provides batch reads from a Lance Table. Currently caches columns on first
/// access, then serves batches from cache. Future optimization: page-level reading.
pub const TableColumnReader = struct {
    table: *Table,
    allocator: std.mem.Allocator,
    row_count: usize,

    // Column data cache (loaded on demand)
    column_cache: std.AutoHashMap(usize, CachedColumn),

    // Column name to index mapping
    column_indices: std.StringHashMap(usize),

    const Self = @This();

    pub const CachedColumn = union(enum) {
        int64: []i64,
        int32: []i32,
        float64: []f64,
        float32: []f32,
        string: [][]const u8,
        bool_: []bool,
    };

    pub fn init(table: *Table, allocator: std.mem.Allocator) !Self {
        const row_count = table.rowCount(0) catch 0;

        // Build column name to index mapping
        var column_indices = std.StringHashMap(usize).init(allocator);
        if (table.getSchema()) |schema| {
            for (schema.fields, 0..) |field, idx| {
                try column_indices.put(field.name, idx);
            }
        }

        return Self{
            .table = table,
            .allocator = allocator,
            .row_count = row_count,
            .column_cache = std.AutoHashMap(usize, CachedColumn).init(allocator),
            .column_indices = column_indices,
        };
    }

    pub fn deinit(self: *Self) void {
        // Free cached column data
        var iter = self.column_cache.iterator();
        while (iter.next()) |entry| {
            switch (entry.value_ptr.*) {
                .int64 => |data| self.allocator.free(data),
                .int32 => |data| self.allocator.free(data),
                .float64 => |data| self.allocator.free(data),
                .float32 => |data| self.allocator.free(data),
                .string => |data| {
                    for (data) |s| self.allocator.free(s);
                    self.allocator.free(data);
                },
                .bool_ => |data| self.allocator.free(data),
            }
        }
        self.column_cache.deinit();
        self.column_indices.deinit();
    }

    /// Get column index by name
    pub fn getColumnIndex(self: *Self, name: []const u8) ?usize {
        return self.column_indices.get(name);
    }

    /// Ensure column is cached, loading if needed
    fn ensureCached(self: *Self, col_idx: usize) !void {
        if (self.column_cache.contains(col_idx)) return;

        // Determine column type and load
        const col_type = self.getColumnTypeInternal(col_idx);

        switch (col_type) {
            .int64 => {
                const data = self.table.readInt64Column(@intCast(col_idx)) catch return error.ReadError;
                try self.column_cache.put(col_idx, .{ .int64 = data });
            },
            .int32 => {
                const data = self.table.readInt32Column(@intCast(col_idx)) catch return error.ReadError;
                try self.column_cache.put(col_idx, .{ .int32 = data });
            },
            .float64 => {
                const data = self.table.readFloat64Column(@intCast(col_idx)) catch return error.ReadError;
                try self.column_cache.put(col_idx, .{ .float64 = data });
            },
            .float32 => {
                const data = self.table.readFloat32Column(@intCast(col_idx)) catch return error.ReadError;
                try self.column_cache.put(col_idx, .{ .float32 = data });
            },
            .string => {
                const data = self.table.readStringColumn(@intCast(col_idx)) catch return error.ReadError;
                try self.column_cache.put(col_idx, .{ .string = data });
            },
            .bool => {
                const data = self.table.readBoolColumn(@intCast(col_idx)) catch return error.ReadError;
                try self.column_cache.put(col_idx, .{ .bool_ = data });
            },
        }
    }

    fn getColumnTypeInternal(self: *Self, col_idx: usize) ve.ColumnType {
        const schema = self.table.getSchema() orelse return .int64;
        if (col_idx >= schema.fields.len) return .int64;

        const field = schema.fields[col_idx];
        const lt = field.logical_type;

        // Match logical_type strings to ve.ColumnType
        if (std.mem.eql(u8, lt, "int64")) return .int64;
        if (std.mem.eql(u8, lt, "int32")) return .int32;
        if (std.mem.eql(u8, lt, "float64") or std.mem.eql(u8, lt, "double")) return .float64;
        if (std.mem.eql(u8, lt, "float32") or std.mem.eql(u8, lt, "float")) return .float32;
        if (std.mem.eql(u8, lt, "string") or std.mem.eql(u8, lt, "utf8") or
            std.mem.eql(u8, lt, "large_string") or std.mem.eql(u8, lt, "large_utf8"))
        {
            return .string;
        }
        if (std.mem.eql(u8, lt, "bool") or std.mem.eql(u8, lt, "boolean")) return .bool;

        return .int64; // Default
    }

    // =========================================================================
    // ve.ColumnReader VTable Implementation
    // =========================================================================

    pub fn reader(self: *Self) ve.ColumnReader {
        return .{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    const vtable = ve.ColumnReader.VTable{
        .readBatchI64 = readBatchI64Fn,
        .readBatchF64 = readBatchF64Fn,
        .readBatchI32 = readBatchI32Fn,
        .readBatchF32 = readBatchF32Fn,
        .readBatchString = readBatchStringFn,
        .getRowCount = getRowCountFn,
        .getColumnType = getColumnTypeFn,
    };

    fn readBatchI64Fn(ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const i64 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.ensureCached(col_idx) catch return null;

        const cached = self.column_cache.get(col_idx) orelse return null;
        const data = switch (cached) {
            .int64 => |d| d,
            else => return null,
        };

        if (offset >= data.len) return null;
        const end = @min(offset + count, data.len);
        return data[offset..end];
    }

    fn readBatchF64Fn(ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const f64 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.ensureCached(col_idx) catch return null;

        const cached = self.column_cache.get(col_idx) orelse return null;
        const data = switch (cached) {
            .float64 => |d| d,
            else => return null,
        };

        if (offset >= data.len) return null;
        const end = @min(offset + count, data.len);
        return data[offset..end];
    }

    fn readBatchI32Fn(ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const i32 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.ensureCached(col_idx) catch return null;

        const cached = self.column_cache.get(col_idx) orelse return null;
        const data = switch (cached) {
            .int32 => |d| d,
            else => return null,
        };

        if (offset >= data.len) return null;
        const end = @min(offset + count, data.len);
        return data[offset..end];
    }

    fn readBatchF32Fn(ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const f32 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.ensureCached(col_idx) catch return null;

        const cached = self.column_cache.get(col_idx) orelse return null;
        const data = switch (cached) {
            .float32 => |d| d,
            else => return null,
        };

        if (offset >= data.len) return null;
        const end = @min(offset + count, data.len);
        return data[offset..end];
    }

    fn readBatchStringFn(ptr: *anyopaque, col_idx: usize, offset: usize, count: usize) ?[]const []const u8 {
        const self: *Self = @ptrCast(@alignCast(ptr));
        self.ensureCached(col_idx) catch return null;

        const cached = self.column_cache.get(col_idx) orelse return null;
        const data = switch (cached) {
            .string => |d| d,
            else => return null,
        };

        if (offset >= data.len) return null;
        const end = @min(offset + count, data.len);
        return data[offset..end];
    }

    fn getRowCountFn(ptr: *anyopaque) usize {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.row_count;
    }

    fn getColumnTypeFn(ptr: *anyopaque, col_idx: usize) ve.ColumnType {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.getColumnTypeInternal(col_idx);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "TableColumnReader: basic initialization" {
    // This test requires a Table instance - skipped for unit testing
    // Integration tests will cover this
}
