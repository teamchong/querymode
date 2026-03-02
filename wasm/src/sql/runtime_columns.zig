//! Runtime Columns - Dynamic struct building for JIT-compiled queries
//!
//! This module provides runtime construction of column struct memory layouts
//! that match the generated Zig code from FusedCodeGen. Since the generated
//! Columns and OutputBuffers structs have query-dependent shapes, we need
//! to build compatible memory layouts at runtime.
//!
//! Memory Layout:
//! The generated structs follow a predictable pattern:
//! ```
//! pub const Columns = struct {
//!     col1: [*]const T1,  // 8 bytes (pointer)
//!     col2: [*]const T2,  // 8 bytes (pointer)
//!     len: usize,         // 8 bytes
//! };
//! ```
//!
//! We build a byte buffer with pointers at each offset, then pass it to
//! the compiled function via `*anyopaque`.

const std = @import("std");
const fused_codegen = @import("codegen/fused_codegen.zig");
const plan_nodes = @import("planner/plan_nodes.zig");

const ColumnInfo = fused_codegen.ColumnInfo;
const ColumnType = plan_nodes.ColumnType;

/// Union of column data pointers by type
pub const ColumnDataPtr = union(enum) {
    i64: []i64,
    i32: []i32,
    i16: []i16,
    i8: []i8,
    u64: []u64,
    u32: []u32,
    u16: []u16,
    u8: []u8,
    f64: []f64,
    f32: []f32,
    bool_: []bool,
    string: [][]const u8,
    timestamp_ns: []i64,
    timestamp_us: []i64,
    timestamp_ms: []i64,
    date32: []i32,
    date64: []i64,
    vec_f32: []f32, // Flat array of f32 vector elements (row_count * dim)
    vec_f64: []f64, // Flat array of f64 vector elements (row_count * dim)
    empty: void, // For unallocated output buffers

    /// Get the raw pointer value for embedding in struct buffer
    pub fn rawPtr(self: ColumnDataPtr) usize {
        return switch (self) {
            .i64 => |s| @intFromPtr(s.ptr),
            .i32 => |s| @intFromPtr(s.ptr),
            .i16 => |s| @intFromPtr(s.ptr),
            .i8 => |s| @intFromPtr(s.ptr),
            .u64 => |s| @intFromPtr(s.ptr),
            .u32 => |s| @intFromPtr(s.ptr),
            .u16 => |s| @intFromPtr(s.ptr),
            .u8 => |s| @intFromPtr(s.ptr),
            .f64 => |s| @intFromPtr(s.ptr),
            .f32 => |s| @intFromPtr(s.ptr),
            .bool_ => |s| @intFromPtr(s.ptr),
            .string => |s| @intFromPtr(s.ptr),
            .timestamp_ns => |s| @intFromPtr(s.ptr),
            .timestamp_us => |s| @intFromPtr(s.ptr),
            .timestamp_ms => |s| @intFromPtr(s.ptr),
            .date32 => |s| @intFromPtr(s.ptr),
            .date64 => |s| @intFromPtr(s.ptr),
            .vec_f32 => |s| @intFromPtr(s.ptr),
            .vec_f64 => |s| @intFromPtr(s.ptr),
            .empty => 0,
        };
    }

    /// Get the length of the data
    pub fn len(self: ColumnDataPtr) usize {
        return switch (self) {
            .i64 => |s| s.len,
            .i32 => |s| s.len,
            .i16 => |s| s.len,
            .i8 => |s| s.len,
            .u64 => |s| s.len,
            .u32 => |s| s.len,
            .u16 => |s| s.len,
            .u8 => |s| s.len,
            .f64 => |s| s.len,
            .f32 => |s| s.len,
            .bool_ => |s| s.len,
            .string => |s| s.len,
            .timestamp_ns => |s| s.len,
            .timestamp_us => |s| s.len,
            .timestamp_ms => |s| s.len,
            .date32 => |s| s.len,
            .date64 => |s| s.len,
            .vec_f32 => |s| s.len,
            .vec_f64 => |s| s.len,
            .empty => 0,
        };
    }
};

/// Runtime-built column struct for passing to compiled functions
pub const RuntimeColumns = struct {
    /// Memory buffer matching generated struct layout
    buffer: []align(8) u8,

    /// Column data pointers (kept alive during function call)
    column_data: []ColumnDataPtr,

    /// Validity bitmaps for nullable columns (same length as column_data)
    /// null entry means column is not nullable
    validity_bitmaps: []?[]const u8,

    /// Column names (for SELECT * output mapping)
    column_names: []const []const u8,

    /// Row count
    row_count: usize,

    /// Whether this is an input (const) or output (mutable) struct
    is_input: bool,

    allocator: std.mem.Allocator,

    const Self = @This();

    /// Build a RuntimeColumns for input data (Columns struct)
    /// validity parameter is optional - if null, all columns are treated as non-nullable
    pub fn buildInput(
        allocator: std.mem.Allocator,
        layout: []const ColumnInfo,
        data: []const ColumnDataPtr,
        row_count: usize,
    ) !Self {
        return buildInputWithValidity(allocator, layout, data, null, row_count);
    }

    /// Build a RuntimeColumns for input data with validity bitmaps
    pub fn buildInputWithValidity(
        allocator: std.mem.Allocator,
        layout: []const ColumnInfo,
        data: []const ColumnDataPtr,
        validity: ?[]const ?[]const u8,
        row_count: usize,
    ) !Self {
        if (layout.len != data.len) return error.LayoutDataMismatch;
        if (validity) |v| {
            if (v.len != data.len) return error.ValidityDataMismatch;
        }

        const ptr_size = @sizeOf(usize);

        // Calculate buffer size accounting for nullable columns
        // Each column: ptr_size, plus additional ptr_size for validity if nullable
        var buffer_size: usize = 0;
        for (layout) |col| {
            buffer_size += ptr_size; // data pointer
            if (col.nullable) {
                buffer_size += ptr_size; // validity pointer
            }
        }
        buffer_size += ptr_size; // len field

        const buffer = try allocator.alignedAlloc(u8, .@"8", buffer_size);
        @memset(buffer, 0);

        // Copy data array
        const column_data = try allocator.alloc(ColumnDataPtr, data.len);
        @memcpy(column_data, data);

        // Create validity bitmaps array
        const validity_bitmaps = try allocator.alloc(?[]const u8, data.len);
        for (0..data.len) |i| {
            validity_bitmaps[i] = if (validity) |v| v[i] else null;
        }

        // Write pointers into buffer at their offsets
        var offset: usize = 0;
        for (layout, 0..) |col, i| {
            // Write data pointer
            const ptr_value = data[i].rawPtr();
            @as(*usize, @ptrCast(@alignCast(buffer.ptr + offset))).* = ptr_value;
            offset += ptr_size;

            // Write validity bitmap pointer for nullable columns
            if (col.nullable) {
                if (validity) |v| {
                    if (v[i]) |vb| {
                        @as(*usize, @ptrCast(@alignCast(buffer.ptr + offset))).* = @intFromPtr(vb.ptr);
                    }
                }
                offset += ptr_size;
            }
        }

        // Write len field at the end
        @as(*usize, @ptrCast(@alignCast(buffer.ptr + offset))).* = row_count;

        // Extract column names from layout
        const column_names = try allocator.alloc([]const u8, layout.len);
        for (layout, 0..) |col, i| {
            column_names[i] = col.name;
        }

        return Self{
            .buffer = buffer,
            .column_data = column_data,
            .validity_bitmaps = validity_bitmaps,
            .column_names = column_names,
            .row_count = row_count,
            .is_input = true,
            .allocator = allocator,
        };
    }

    /// Build a RuntimeColumns for output data (OutputBuffers struct)
    /// Allocates buffers for each column based on type
    pub fn buildOutput(
        allocator: std.mem.Allocator,
        layout: []const ColumnInfo,
        max_rows: usize,
    ) !Self {
        const ptr_size = @sizeOf(usize);
        // Buffer size = num columns * ptr_size (no len field in OutputBuffers)
        const buffer_size = layout.len * ptr_size;

        const buffer = try allocator.alignedAlloc(u8, .@"8", buffer_size);
        @memset(buffer, 0);

        var column_data = try allocator.alloc(ColumnDataPtr, layout.len);

        // Allocate output buffers for each column
        const column_names = try allocator.alloc([]const u8, layout.len);
        for (layout, 0..) |col, i| {
            column_data[i] = try allocateColumnBuffer(allocator, col.col_type, max_rows);
            column_names[i] = col.name;
            const ptr_value = column_data[i].rawPtr();
            const offset = col.offset;
            @as(*usize, @ptrCast(@alignCast(buffer.ptr + offset))).* = ptr_value;
        }

        // Output doesn't have validity bitmaps (yet)
        const validity_bitmaps = try allocator.alloc(?[]const u8, layout.len);
        for (0..layout.len) |i| {
            validity_bitmaps[i] = null;
        }

        return Self{
            .buffer = buffer,
            .column_data = column_data,
            .validity_bitmaps = validity_bitmaps,
            .column_names = column_names,
            .row_count = max_rows,
            .is_input = false,
            .allocator = allocator,
        };
    }

    /// Get pointer suitable for passing to compiled function
    pub fn asPtr(self: *Self) *anyopaque {
        return @ptrCast(self.buffer.ptr);
    }

    /// Get const pointer (for Columns struct)
    pub fn asConstPtr(self: *const Self) *const anyopaque {
        return @ptrCast(self.buffer.ptr);
    }

    pub fn deinit(self: *Self) void {
        // Free column data buffers if this is an output struct
        if (!self.is_input) {
            for (self.column_data) |col| {
                freeColumnBuffer(self.allocator, col);
            }
        }
        self.allocator.free(self.column_data);
        self.allocator.free(self.validity_bitmaps);
        self.allocator.free(self.column_names);
        self.allocator.free(self.buffer);
    }
};

/// Allocate a column buffer based on type
fn allocateColumnBuffer(allocator: std.mem.Allocator, col_type: ColumnType, row_count: usize) !ColumnDataPtr {
    return switch (col_type) {
        .i64 => .{ .i64 = try allocator.alloc(i64, row_count) },
        .i32 => .{ .i32 = try allocator.alloc(i32, row_count) },
        .i16 => .{ .i16 = try allocator.alloc(i16, row_count) },
        .i8 => .{ .i8 = try allocator.alloc(i8, row_count) },
        .u64 => .{ .u64 = try allocator.alloc(u64, row_count) },
        .u32 => .{ .u32 = try allocator.alloc(u32, row_count) },
        .u16 => .{ .u16 = try allocator.alloc(u16, row_count) },
        .u8 => .{ .u8 = try allocator.alloc(u8, row_count) },
        .f64 => .{ .f64 = try allocator.alloc(f64, row_count) },
        .f32 => .{ .f32 = try allocator.alloc(f32, row_count) },
        .bool => .{ .bool_ = try allocator.alloc(bool, row_count) },
        .string => .{ .string = try allocator.alloc([]const u8, row_count) },
        .timestamp_ns => .{ .timestamp_ns = try allocator.alloc(i64, row_count) },
        .timestamp_us => .{ .timestamp_us = try allocator.alloc(i64, row_count) },
        .timestamp_ms => .{ .timestamp_ms = try allocator.alloc(i64, row_count) },
        .timestamp_s => .{ .timestamp_ns = try allocator.alloc(i64, row_count) }, // Treat as ns
        .date32 => .{ .date32 = try allocator.alloc(i32, row_count) },
        .date64 => .{ .date64 = try allocator.alloc(i64, row_count) },
        .vec_f32 => .{ .vec_f32 = try allocator.alloc(f32, row_count) },
        .vec_f64 => .{ .vec_f64 = try allocator.alloc(f64, row_count) },
        .unknown => .{ .f64 = try allocator.alloc(f64, row_count) }, // Default to f64
        else => .{ .f64 = try allocator.alloc(f64, row_count) }, // Default for unsupported types
    };
}

/// Free a column buffer
fn freeColumnBuffer(allocator: std.mem.Allocator, col: ColumnDataPtr) void {
    switch (col) {
        .i64 => |s| allocator.free(s),
        .i32 => |s| allocator.free(s),
        .i16 => |s| allocator.free(s),
        .i8 => |s| allocator.free(s),
        .u64 => |s| allocator.free(s),
        .u32 => |s| allocator.free(s),
        .u16 => |s| allocator.free(s),
        .u8 => |s| allocator.free(s),
        .f64 => |s| allocator.free(s),
        .f32 => |s| allocator.free(s),
        .bool_ => |s| allocator.free(s),
        .string => |s| allocator.free(s),
        .timestamp_ns => |s| allocator.free(s),
        .timestamp_us => |s| allocator.free(s),
        .timestamp_ms => |s| allocator.free(s),
        .date32 => |s| allocator.free(s),
        .date64 => |s| allocator.free(s),
        .vec_f32 => |s| allocator.free(s),
        .vec_f64 => |s| allocator.free(s),
        .empty => {},
    }
}

// ============================================================================
// Tests
// ============================================================================

test "RuntimeColumns buildInput" {
    const allocator = std.testing.allocator;

    // Create test layout
    const layout = [_]ColumnInfo{
        .{ .name = "amount", .col_type = .f64, .offset = 0 },
        .{ .name = "count", .col_type = .i64, .offset = 8 },
    };

    // Create test data
    var f64_data = [_]f64{ 1.0, 2.0, 3.0 };
    var i64_data = [_]i64{ 10, 20, 30 };

    const data = [_]ColumnDataPtr{
        .{ .f64 = &f64_data },
        .{ .i64 = &i64_data },
    };

    var cols = try RuntimeColumns.buildInput(allocator, &layout, &data, 3);
    defer cols.deinit();

    // Verify buffer was created
    try std.testing.expect(cols.buffer.len > 0);
    try std.testing.expectEqual(@as(usize, 3), cols.row_count);
    try std.testing.expect(cols.is_input);
}

test "RuntimeColumns buildOutput" {
    const allocator = std.testing.allocator;

    // Create test layout
    const layout = [_]ColumnInfo{
        .{ .name = "result", .col_type = .f64, .offset = 0 },
        .{ .name = "flag", .col_type = .bool, .offset = 8 },
    };

    var out = try RuntimeColumns.buildOutput(allocator, &layout, 100);
    defer out.deinit();

    // Verify buffers were allocated
    try std.testing.expect(out.buffer.len > 0);
    try std.testing.expectEqual(@as(usize, 100), out.row_count);
    try std.testing.expect(!out.is_input);
    try std.testing.expectEqual(@as(usize, 2), out.column_data.len);
}

test "ColumnDataPtr rawPtr" {
    var data = [_]i64{ 1, 2, 3 };
    const col = ColumnDataPtr{ .i64 = &data };
    const ptr = col.rawPtr();
    try std.testing.expect(ptr != 0);
    try std.testing.expectEqual(@as(usize, 3), col.len());
}
