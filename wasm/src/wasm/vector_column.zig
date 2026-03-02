//! Vector Column Support for WASM
//!
//! Handles fixed-size float arrays for embeddings in Lance files.
//! Vectors are stored as contiguous float32 arrays.

const std = @import("std");
const format = @import("format.zig");
const memory = @import("memory.zig");

const readU64LE = format.readU64LE;
const readF32LE = format.readF32LE;
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
// Column Buffer Struct
// ============================================================================

pub const ColumnBuffer = struct {
    data: []const u8,
    start: usize,
    size: usize,
    rows: u32,
};

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
// Page Buffer Info Parsing
// ============================================================================

/// Get page buffer info from column metadata protobuf
fn getPageBufferInfo(col_meta: []const u8) struct { offset: u64, size: u64, rows: u64 } {
    var pos: usize = 0;
    var page_offset: u64 = 0;
    var page_size: u64 = 0;
    var page_rows: u64 = 0;

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
                                // Read first offset only
                                if (pos < packed_end) {
                                    page_offset = readVarint(col_meta, &pos);
                                }
                                // Skip rest
                                pos = packed_end;
                            } else {
                                page_offset = readVarint(col_meta, &pos);
                            }
                        },
                        2 => { // buffer_sizes (packed repeated uint64)
                            if (page_wire == 2) {
                                const packed_len = readVarint(col_meta, &pos);
                                const packed_end = pos + @as(usize, @intCast(packed_len));
                                // Read first size only
                                if (pos < packed_end) {
                                    page_size = readVarint(col_meta, &pos);
                                }
                                // Skip rest
                                pos = packed_end;
                            } else {
                                page_size = readVarint(col_meta, &pos);
                            }
                        },
                        3 => { // length (rows)
                            page_rows = readVarint(col_meta, &pos);
                        },
                        else => {
                            // Skip field
                            if (page_wire == 0) {
                                _ = readVarint(col_meta, &pos);
                            } else if (page_wire == 2) {
                                const skip_len = readVarint(col_meta, &pos);
                                pos += @as(usize, @intCast(skip_len));
                            } else if (page_wire == 5) {
                                pos += 4; // 32-bit fixed
                            } else if (page_wire == 1) {
                                pos += 8; // 64-bit fixed
                            }
                        },
                    }
                }
                return .{ .offset = page_offset, .size = page_size, .rows = page_rows };
            },
            else => {
                // Skip other fields
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
    return .{ .offset = 0, .size = 0, .rows = 0 };
}

// ============================================================================
// Column Buffer Access
// ============================================================================

pub fn getColumnBuffer(col_idx: u32) ?ColumnBuffer {
    const data = file_data orelse return null;
    const entry = getColumnOffsetEntry(col_idx);
    if (entry.len == 0) return null;

    const col_meta_start: usize = @intCast(entry.pos);
    const col_meta_len: usize = @intCast(entry.len);
    if (col_meta_start + col_meta_len > data.len) return null;

    const col_meta = data[col_meta_start..][0..col_meta_len];
    const buf_info = getPageBufferInfo(col_meta);

    return .{
        .data = data,
        .start = @intCast(buf_info.offset),
        .size = @intCast(buf_info.size),
        .rows = @intCast(buf_info.rows),
    };
}

// ============================================================================
// WASM Exports
// ============================================================================

/// Get vector info from column: dimension and count
/// Vectors are stored as fixed-size arrays of float32
pub export fn getVectorInfo(col_idx: u32) u64 {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    if (buf.rows == 0) return 0;
    const dim = buf.size / (buf.rows * 4);
    return (@as(u64, buf.rows) << 32) | dim;
}

/// Read a single vector at index
/// Returns number of floats written
pub export fn readVectorAt(
    col_idx: u32,
    row_idx: u32,
    out_ptr: [*]f32,
    max_dim: usize,
) usize {
    const buf = getColumnBuffer(col_idx) orelse return 0;
    if (buf.rows == 0 or row_idx >= buf.rows) return 0;

    const dim = buf.size / (buf.rows * 4);
    if (dim == 0) return 0;

    const vec_start = buf.start + @as(usize, row_idx) * dim * 4;
    const actual_dim = @min(dim, max_dim);
    for (0..actual_dim) |i| out_ptr[i] = readF32LE(buf.data, vec_start + i * 4);
    return actual_dim;
}

/// Allocate float32 buffer for vectors
pub export fn allocFloat32Buffer(count: usize) ?[*]f32 {
    const ptr = wasmAlloc(count * 4) orelse return null;
    return @ptrCast(@alignCast(ptr));
}

/// Compute cosine similarity between two vectors
/// Returns similarity score (-1 to 1, higher is more similar)
pub export fn cosineSimilarity(
    vec_a: [*]const f32,
    vec_b: [*]const f32,
    dim: usize,
) f32 {
    var dot: f32 = 0;
    var norm_a: f32 = 0;
    var norm_b: f32 = 0;

    for (0..dim) |i| {
        const a = vec_a[i];
        const b = vec_b[i];
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }

    const denom = @sqrt(norm_a) * @sqrt(norm_b);
    if (denom == 0) return 0;
    return dot / denom;
}

// ============================================================================
// Tests
// ============================================================================

test "vector_column: ColumnBuffer struct" {
    const buf = ColumnBuffer{
        .data = &[_]u8{},
        .start = 1000,
        .size = 4000,
        .rows = 100,
    };
    try std.testing.expectEqual(@as(usize, 1000), buf.start);
    try std.testing.expectEqual(@as(usize, 4000), buf.size);
    try std.testing.expectEqual(@as(u32, 100), buf.rows);
}

test "vector_column: getColumnBuffer returns null without file" {
    file_data = null;
    num_columns = 0;
    const result = getColumnBuffer(0);
    try std.testing.expect(result == null);
}

test "vector_column: cosineSimilarity identical vectors" {
    const vec = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const result = cosineSimilarity(&vec, &vec, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result, 0.0001);
}

test "vector_column: cosineSimilarity orthogonal vectors" {
    const vec_a = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const vec_b = [_]f32{ 0.0, 1.0, 0.0, 0.0 };
    const result = cosineSimilarity(&vec_a, &vec_b, 4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result, 0.0001);
}

test "vector_column: cosineSimilarity opposite vectors" {
    const vec_a = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const vec_b = [_]f32{ -1.0, 0.0, 0.0, 0.0 };
    const result = cosineSimilarity(&vec_a, &vec_b, 4);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), result, 0.0001);
}

test "vector_column: cosineSimilarity zero vector" {
    const vec_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const vec_b = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const result = cosineSimilarity(&vec_a, &vec_b, 4);
    try std.testing.expectEqual(@as(f32, 0.0), result);
}
