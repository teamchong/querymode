//! String Column Support for WASM
//!
//! Handles variable-length string data in Lance files.
//! Lance stores strings with:
//!   - buffer[0]: offsets (int32 or int64)
//!   - buffer[1]: data (UTF-8 bytes)

const std = @import("std");
const format = @import("format.zig");
const memory = @import("memory.zig");

const readU64LE = format.readU64LE;
const readU32LE = format.readU32LE;
const readVarint = format.readVarint;
const wasmAlloc = memory.wasmAlloc;

// ============================================================================
// Global State (accessed from parent module)
// ============================================================================

/// File data pointer (set by parent module)
pub var file_data: ?[]const u8 = null;
pub var num_columns: u32 = 0;
pub var column_meta_offsets_start: u64 = 0;

// ============================================================================
// Column Offset Entry
// ============================================================================

fn getColumnOffsetEntry(col_idx: u32) struct { pos: u64, len: u64 } {
    const data = file_data orelse return .{ .pos = 0, .len = 0 };
    if (col_idx >= num_columns) return .{ .pos = 0, .len = 0 };

    const entry_offset: usize = @intCast(column_meta_offsets_start + col_idx * 16);
    if (entry_offset + 16 > data.len) return .{ .pos = 0, .len = 0 };

    return .{
        .pos = readU64LE(data, entry_offset),
        .len = readU64LE(data, entry_offset + 8),
    };
}

// ============================================================================
// String Buffer Info Parsing
// ============================================================================

/// Get string buffer info from column metadata (for variable-length data)
fn getStringBufferInfo(col_meta: []const u8) struct {
    offsets_start: u64,
    offsets_size: u64,
    data_start: u64,
    data_size: u64,
    rows: u64,
} {
    var pos: usize = 0;
    var buffer_offsets: [2]u64 = .{ 0, 0 };
    var buffer_sizes: [2]u64 = .{ 0, 0 };
    var page_rows: u64 = 0;
    var buf_idx: usize = 0;

    while (pos < col_meta.len) {
        const tag = readVarint(col_meta, &pos);
        const field_num = tag >> 3;
        const wire_type: u3 = @truncate(tag);

        switch (field_num) {
            1 => { // encoding (length-delimited) - skip it
                if (wire_type == 2) {
                    const skip_len = readVarint(col_meta, &pos);
                    pos += @as(usize, @intCast(skip_len));
                }
            },
            2 => { // pages (length-delimited)
                if (wire_type != 2) break;
                const page_len = readVarint(col_meta, &pos);
                const page_end = pos + @as(usize, @intCast(page_len));

                // Parse page message
                while (pos < page_end and pos < col_meta.len) {
                    const page_tag = readVarint(col_meta, &pos);
                    const page_field = page_tag >> 3;
                    const page_wire: u3 = @truncate(page_tag);

                    switch (page_field) {
                        1 => { // buffer_offsets (packed repeated uint64)
                            if (page_wire == 2) {
                                const packed_len = readVarint(col_meta, &pos);
                                const packed_end = pos + @as(usize, @intCast(packed_len));
                                buf_idx = 0;
                                while (pos < packed_end and buf_idx < 2) {
                                    buffer_offsets[buf_idx] = readVarint(col_meta, &pos);
                                    buf_idx += 1;
                                }
                                pos = packed_end;
                            } else {
                                if (buf_idx < 2) {
                                    buffer_offsets[buf_idx] = readVarint(col_meta, &pos);
                                    buf_idx += 1;
                                }
                            }
                        },
                        2 => { // buffer_sizes (packed repeated uint64)
                            if (page_wire == 2) {
                                const packed_len = readVarint(col_meta, &pos);
                                const packed_end = pos + @as(usize, @intCast(packed_len));
                                buf_idx = 0;
                                while (pos < packed_end and buf_idx < 2) {
                                    buffer_sizes[buf_idx] = readVarint(col_meta, &pos);
                                    buf_idx += 1;
                                }
                                pos = packed_end;
                            } else {
                                if (buf_idx < 2) {
                                    buffer_sizes[buf_idx] = readVarint(col_meta, &pos);
                                    buf_idx += 1;
                                }
                            }
                        },
                        3 => { // length (rows)
                            page_rows = readVarint(col_meta, &pos);
                        },
                        else => {
                            if (page_wire == 0) {
                                _ = readVarint(col_meta, &pos);
                            } else if (page_wire == 2) {
                                const skip_len = readVarint(col_meta, &pos);
                                pos += @as(usize, @intCast(skip_len));
                            } else if (page_wire == 5) {
                                pos += 4;
                            } else if (page_wire == 1) {
                                pos += 8;
                            }
                        },
                    }
                }
                return .{
                    .offsets_start = buffer_offsets[0],
                    .offsets_size = buffer_sizes[0],
                    .data_start = buffer_offsets[1],
                    .data_size = buffer_sizes[1],
                    .rows = page_rows,
                };
            },
            else => {
                if (wire_type == 0) {
                    _ = readVarint(col_meta, &pos);
                } else if (wire_type == 2) {
                    const skip_len = readVarint(col_meta, &pos);
                    pos += @as(usize, @intCast(skip_len));
                } else if (wire_type == 5) {
                    pos += 4;
                } else if (wire_type == 1) {
                    pos += 8;
                }
            },
        }
    }

    return .{
        .offsets_start = 0,
        .offsets_size = 0,
        .data_start = 0,
        .data_size = 0,
        .rows = 0,
    };
}

// ============================================================================
// String Buffer Struct
// ============================================================================

/// Helper to get string column buffer info
pub const StringBuffer = struct {
    data: []const u8,
    offsets_start: usize,
    offsets_size: usize,
    data_start: usize,
    data_size: usize,
    rows: usize,
};

pub fn getStringBuffer(col_idx: u32) ?StringBuffer {
    const data = file_data orelse return null;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return null;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return null;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const info = getStringBufferInfo(col_meta);

    return .{
        .data = data,
        .offsets_start = @intCast(info.offsets_start),
        .offsets_size = @intCast(info.offsets_size),
        .data_start = @intCast(info.data_start),
        .data_size = @intCast(info.data_size),
        .rows = @intCast(info.rows),
    };
}

// ============================================================================
// WASM Exports
// ============================================================================

/// Debug: Get string column buffer info
/// Returns packed: high32=offsets_size, low32=data_size (both 0 if not a string column)
pub export fn debugStringColInfo(col_idx: u32) u64 {
    const buf = getStringBuffer(col_idx) orelse return 0;
    return (@as(u64, @intCast(buf.offsets_size)) << 32) | @as(u64, @intCast(buf.data_size));
}

/// Get number of strings in column
/// Returns 0 if not a string column (string columns have 2 buffers: offsets + data)
pub export fn getStringCount(col_idx: u32) u64 {
    const buf = getStringBuffer(col_idx) orelse return 0;
    if (buf.data_size == 0) return 0;
    return buf.rows;
}

/// Debug: Get detailed string read info for a specific row
/// Returns packed debug info for troubleshooting
pub export fn debugReadStringInfo(col_idx: u32, row_idx: u32) u64 {
    const buf = getStringBuffer(col_idx) orelse return 0xDEAD0001;
    if (buf.offsets_size == 0 or buf.data_size == 0) return 0xDEAD0004;
    if (row_idx >= buf.rows) return 0xDEAD0005;

    const offset_size = buf.offsets_size / buf.rows;
    if (offset_size != 4 and offset_size != 8) return 0xDEAD0006;

    var str_start: usize = 0;
    var str_end: usize = 0;

    if (offset_size == 4) {
        str_end = readU32LE(buf.data, buf.offsets_start + row_idx * 4);
        if (row_idx > 0) str_start = readU32LE(buf.data, buf.offsets_start + (row_idx - 1) * 4);
    } else {
        str_end = @intCast(readU64LE(buf.data, buf.offsets_start + row_idx * 8));
        if (row_idx > 0) str_start = @intCast(readU64LE(buf.data, buf.offsets_start + (row_idx - 1) * 8));
    }

    const str_len = if (str_end >= str_start) str_end - str_start else 0;
    return (@as(u64, @intCast(str_start)) << 32) | @as(u64, @intCast(str_len));
}

/// Debug: Get data_start position for string column
pub export fn debugStringDataStart(col_idx: u32) u64 {
    const buf = getStringBuffer(col_idx) orelse return 0;
    const ds: u32 = @intCast(@min(buf.data_start, 0xFFFFFFFF));
    const fl: u32 = @intCast(@min(buf.data.len, 0xFFFFFFFF));
    return (@as(u64, ds) << 32) | @as(u64, fl);
}

/// Read a single string at index into output buffer
/// Returns actual string length (may exceed out_max if truncated)
pub export fn readStringAt(col_idx: u32, row_idx: u32, out_ptr: [*]u8, out_max: usize) usize {
    const buf = getStringBuffer(col_idx) orelse return 0;
    if (buf.offsets_size == 0 or buf.data_size == 0) return 0;
    if (row_idx >= buf.rows) return 0;

    // Lance v2 uses N offsets for N strings (end positions, not N+1 start/end pairs)
    const offset_size = buf.offsets_size / buf.rows;
    if (offset_size != 4 and offset_size != 8) return 0;

    var str_start: usize = 0;
    var str_end: usize = 0;

    if (offset_size == 4) {
        str_end = readU32LE(buf.data, buf.offsets_start + row_idx * 4);
        if (row_idx > 0) str_start = readU32LE(buf.data, buf.offsets_start + (row_idx - 1) * 4);
    } else {
        str_end = @intCast(readU64LE(buf.data, buf.offsets_start + row_idx * 8));
        if (row_idx > 0) str_start = @intCast(readU64LE(buf.data, buf.offsets_start + (row_idx - 1) * 8));
    }

    if (str_end < str_start) return 0;
    const str_len = str_end - str_start;
    if (buf.data_start + str_end > buf.data.len) return 0;

    const copy_len = @min(str_len, out_max);
    @memcpy(out_ptr[0..copy_len], buf.data[buf.data_start + str_start ..][0..copy_len]);
    return str_len;
}

/// Read multiple strings at indices
/// Returns total bytes written to out_ptr
/// out_lengths receives the length of each string
pub export fn readStringsAtIndices(
    col_idx: u32,
    indices: [*]const u32,
    num_indices: usize,
    out_ptr: [*]u8,
    out_max: usize,
    out_lengths: [*]u32,
) usize {
    var total_written: usize = 0;

    for (0..num_indices) |i| {
        const remaining = if (total_written < out_max) out_max - total_written else 0;
        const len = readStringAt(col_idx, indices[i], out_ptr + total_written, remaining);
        out_lengths[i] = @intCast(len);
        total_written += @min(len, remaining);
    }

    return total_written;
}

/// Allocate string buffer
pub export fn allocStringBuffer(size: usize) ?[*]u8 {
    return wasmAlloc(size);
}

/// Allocate u32 buffer for lengths
pub export fn allocU32Buffer(count: usize) ?[*]u32 {
    const ptr = wasmAlloc(count * 4) orelse return null;
    return @ptrCast(@alignCast(ptr));
}

// ============================================================================
// Tests
// ============================================================================

test "string_column: StringBuffer struct" {
    const buf = StringBuffer{
        .data = &[_]u8{},
        .offsets_start = 100,
        .offsets_size = 400,
        .data_start = 500,
        .data_size = 1000,
        .rows = 100,
    };
    try std.testing.expectEqual(@as(usize, 100), buf.offsets_start);
    try std.testing.expectEqual(@as(usize, 400), buf.offsets_size);
    try std.testing.expectEqual(@as(usize, 500), buf.data_start);
    try std.testing.expectEqual(@as(usize, 1000), buf.data_size);
    try std.testing.expectEqual(@as(usize, 100), buf.rows);
}

test "string_column: getStringBuffer returns null without file" {
    file_data = null;
    num_columns = 0;
    const result = getStringBuffer(0);
    try std.testing.expect(result == null);
}
