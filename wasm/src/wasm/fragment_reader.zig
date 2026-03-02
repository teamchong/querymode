//! Fragment Reader for WASM
//!
//! Provides a high-level API for reading Lance fragment files.
//! Parses the footer, column metadata, and provides typed column access.

const std = @import("std");
const memory = @import("memory.zig");
const opfs = @import("opfs.zig");
const js = opfs.js;

// ============================================================================
// Constants
// ============================================================================

const MAX_COLUMNS = 64;

// ============================================================================
// Reader Struct
// ============================================================================

pub const ReaderColumnInfo = struct {
    name: [64]u8,
    name_len: usize,
    col_type: [16]u8,
    type_len: usize,
    nullable: bool,
    data_offset: u64,
    row_count: u64,
    data_size: u64,
    vector_dim: u32,
    cached_data: ?[]u8 = null, // Cache for on-demand loading
};

pub const FragmentReader = struct {
    data: ?[*]const u8 = null,
    len: usize = 0,
    num_columns: u32 = 0,
    column_meta_start: u64 = 0,
    column_meta_offsets_start: u64 = 0,
    columns: [MAX_COLUMNS]ReaderColumnInfo = undefined,
    opfs_handle: u32 = 0, // 0 means not an OPFS lazy handle
    is_lazy: bool = false,

    pub fn init(data: [*]const u8, len: usize) !FragmentReader {
        if (len < 40) return error.InvalidFile;

        var reader = FragmentReader{
            .data = data,
            .len = len,
        };

        // Parse footer (last 40 bytes)
        const footer_start = len - 40;

        // Check magic
        if (data[footer_start + 36] != 'L' or
            data[footer_start + 37] != 'A' or
            data[footer_start + 38] != 'N' or
            data[footer_start + 39] != 'C')
        {
            return error.InvalidMagic;
        }

        reader.column_meta_start = std.mem.readInt(u64, data[footer_start..][0..8], .little);
        reader.column_meta_offsets_start = std.mem.readInt(u64, data[footer_start + 8 ..][0..8], .little);
        reader.num_columns = std.mem.readInt(u32, data[footer_start + 28 ..][0..4], .little);

        if (reader.num_columns > MAX_COLUMNS) return error.TooManyColumns;

        // Parse column metadata
        for (0..reader.num_columns) |i| {
            const offset_pos: usize = @intCast(reader.column_meta_offsets_start + i * 8);
            const meta_offset = std.mem.readInt(u64, data[offset_pos..][0..8], .little);

            const next_offset = if (i + 1 < reader.num_columns)
                std.mem.readInt(u64, data[offset_pos + 8 ..][0..8], .little)
            else
                reader.column_meta_offsets_start;

            parseColumnMeta(data, meta_offset, next_offset, &reader.columns[i]);
        }

        return reader;
    }

    /// Initialize a lazy reader from OPFS - only reads footer and metadata, not column data
    pub fn initLazy(handle: u32) !FragmentReader {
        if (handle == 0) return error.InvalidHandle;

        const size = js.opfs_size(handle);
        if (size < 40) return error.InvalidFile;

        var reader = FragmentReader{
            .opfs_handle = handle,
            .is_lazy = true,
            .len = @intCast(size),
        };

        // Read just the footer (40 bytes)
        var footer_buf: [40]u8 = undefined;
        const footer_offset = size - 40;
        const bytes_read = js.opfs_read(handle, &footer_buf, 40, footer_offset);
        if (bytes_read != 40) return error.ReadError;

        // Check magic
        if (footer_buf[36] != 'L' or footer_buf[37] != 'A' or footer_buf[38] != 'N' or footer_buf[39] != 'C') {
            return error.InvalidMagic;
        }

        reader.column_meta_start = std.mem.readInt(u64, footer_buf[0..8], .little);
        reader.column_meta_offsets_start = std.mem.readInt(u64, footer_buf[8..16], .little);
        reader.num_columns = std.mem.readInt(u32, footer_buf[28..32], .little);

        if (reader.num_columns > MAX_COLUMNS) return error.TooManyColumns;

        // Read column metadata offsets
        const offsets_size = reader.num_columns * 8;
        const offsets_buf = memory.wasmAlloc(offsets_size) orelse return error.OutOfMemory;

        const offsets_read = js.opfs_read(handle, offsets_buf, offsets_size, reader.column_meta_offsets_start);
        if (offsets_read != offsets_size) return error.ReadError;

        // Read each column's metadata
        for (0..reader.num_columns) |i| {
            const meta_offset = std.mem.readInt(u64, offsets_buf[i * 8 ..][0..8], .little);
            const next_offset = if (i + 1 < reader.num_columns)
                std.mem.readInt(u64, offsets_buf[(i + 1) * 8 ..][0..8], .little)
            else
                reader.column_meta_offsets_start;

            const meta_size = next_offset - meta_offset;
            if (meta_size > 4096) continue; // Skip unreasonably large metadata

            const meta_buf = memory.wasmAlloc(@intCast(meta_size)) orelse continue;
            const meta_read = js.opfs_read(handle, meta_buf, @intCast(meta_size), meta_offset);
            if (meta_read != @as(usize, @intCast(meta_size))) continue;

            parseColumnMetaFromBuf(meta_buf, @intCast(meta_size), &reader.columns[i]);
        }

        return reader;
    }

    pub fn initDummy(row_count: u64) FragmentReader {
        var reader = FragmentReader{
            .len = 0,
            .num_columns = 1,
        };
        reader.columns[0].row_count = row_count;
        return reader;
    }

    pub fn getColumnCount(self: *const FragmentReader) u32 {
        return self.num_columns;
    }

    pub fn getRowCount(self: *const FragmentReader) u64 {
        if (self.num_columns == 0) return 0;
        return self.columns[0].row_count;
    }

    pub fn getColumnInfo(self: *const FragmentReader, col_idx: u32) ?*const ReaderColumnInfo {
        if (col_idx >= self.num_columns) return null;
        return &self.columns[col_idx];
    }

    pub fn fragmentGetColumnName(self: *const FragmentReader, col_idx: u32, out_ptr: [*]u8, max_len: usize) usize {
        if (col_idx >= self.num_columns) return 0;
        const info = &self.columns[col_idx];
        const copy_len = @min(info.name_len, max_len);
        @memcpy(out_ptr[0..copy_len], info.name[0..copy_len]);
        return copy_len;
    }

    pub fn fragmentGetColumnType(self: *const FragmentReader, col_idx: u32, out_ptr: [*]u8, max_len: usize) usize {
        if (col_idx >= self.num_columns) return 0;
        const info = &self.columns[col_idx];
        const copy_len = @min(info.type_len, max_len);
        @memcpy(out_ptr[0..copy_len], info.col_type[0..copy_len]);
        return copy_len;
    }

    /// Get raw pointer to column data
    /// For lazy readers, this returns null - use loadColumnData() first
    pub fn getColumnRawPtr(self: *const FragmentReader, col_idx: u32) ?[*]const u8 {
        if (col_idx >= self.num_columns) return null;

        // Fast path: data already in memory
        if (self.data) |data| {
            const info = &self.columns[col_idx];
            return data + @as(usize, @intCast(info.data_offset));
        }

        // Check if already cached
        if (self.columns[col_idx].cached_data) |cached| {
            return cached.ptr;
        }

        return null;
    }

    /// Load column data on-demand for lazy readers (mutable self required)
    pub fn loadColumnData(self: *FragmentReader, col_idx: u32) ?[*]const u8 {
        if (col_idx >= self.num_columns) return null;

        // Already loaded?
        if (self.data) |data| {
            const info = &self.columns[col_idx];
            return data + @as(usize, @intCast(info.data_offset));
        }

        if (self.columns[col_idx].cached_data) |cached| {
            return cached.ptr;
        }

        // Lazy load from OPFS
        if (self.is_lazy and self.opfs_handle != 0) {
            const info = &self.columns[col_idx];
            const size: usize = @intCast(info.data_size);
            if (size == 0) return null;

            const buf = memory.wasmAlloc(size) orelse return null;
            const bytes_read = js.opfs_read(self.opfs_handle, buf, size, info.data_offset);
            if (bytes_read != size) return null;

            // Cache the loaded data
            self.columns[col_idx].cached_data = buf[0..size];
            return buf;
        }

        return null;
    }

    pub fn fragmentReadInt64(self: *const FragmentReader, col_idx: u32, out_ptr: [*]i64, max_count: usize, start_row: u32) usize {
        if (col_idx >= self.num_columns) return 0;
        const data = self.data orelse return 0;
        const info = &self.columns[col_idx];

        if (start_row >= info.row_count) return 0;
        const count: usize = @intCast(@min(info.row_count - start_row, max_count));
        
        var i: usize = 0;
        while (i < count) : (i += 1) {
            const offset: usize = @intCast(info.data_offset + (start_row + i) * 8);
            out_ptr[i] = std.mem.readInt(i64, data[offset..][0..8], .little);
        }
        return count;
    }

    pub fn fragmentReadStringAt(self: *const FragmentReader, col_idx: u32, row_idx: u32, out_ptr: [*]u8, max_len: usize) usize {
        if (col_idx >= self.num_columns) return 0;
        const data = self.data orelse return 0;
        const info = &self.columns[col_idx];

        if (row_idx >= info.row_count) return 0;

        const offsets_size: usize = @intCast((info.row_count + 1) * 4);
        const offsets_start: usize = @intCast(info.data_offset + info.data_size - offsets_size);
        const data_start: usize = @intCast(info.data_offset);

        const start_offset = std.mem.readInt(u32, data[offsets_start + row_idx * 4 ..][0..4], .little);
        const end_offset = std.mem.readInt(u32, data[offsets_start + (row_idx + 1) * 4 ..][0..4], .little);

        const str_len = end_offset - start_offset;
        const copy_len = @min(str_len, max_len);

        @memcpy(out_ptr[0..copy_len], data[data_start + start_offset ..][0..copy_len]);
        return copy_len;
    }

    pub fn fragmentGetStringLength(self: *const FragmentReader, col_idx: u32, row_idx: u32) usize {
        if (col_idx >= self.num_columns) return 0;
        const data = self.data orelse return 0;
        const info = &self.columns[col_idx];

        if (row_idx >= info.row_count) return 0;

        const offsets_size: usize = @intCast((info.row_count + 1) * 4);
        const offsets_start: usize = @intCast(info.data_offset + info.data_size - offsets_size);

        const start_offset = std.mem.readInt(u32, data[offsets_start + row_idx * 4 ..][0..4], .little);
        const end_offset = std.mem.readInt(u32, data[offsets_start + (row_idx + 1) * 4 ..][0..4], .little);

        return end_offset - start_offset;
    }

    pub fn fragmentReadFloat64(self: *const FragmentReader, col_idx: u32, out_ptr: [*]f64, max_count: usize, start_row: u32) usize {
        if (col_idx >= self.num_columns) return 0;
        const data = self.data orelse return 0;
        const info = &self.columns[col_idx];

        if (start_row >= info.row_count) return 0;
        const count: usize = @intCast(@min(info.row_count - start_row, max_count));

        var i: usize = 0;
        while (i < count) : (i += 1) {
            const offset: usize = @intCast(info.data_offset + (start_row + i) * 8);
            const bits = std.mem.readInt(u64, data[offset..][0..8], .little);
            out_ptr[i] = @bitCast(bits);
        }
        return count;
    }

    pub fn fragmentReadInt32(self: *const FragmentReader, col_idx: u32, out_ptr: [*]i32, max_count: usize, start_row: u32) usize {
        if (col_idx >= self.num_columns) return 0;
        const data = self.data orelse return 0;
        const info = &self.columns[col_idx];

        if (start_row >= info.row_count) return 0;
        const count: usize = @intCast(@min(info.row_count - start_row, max_count));

        var i: usize = 0;
        while (i < count) : (i += 1) {
            const offset: usize = @intCast(info.data_offset + (start_row + i) * 4);
            out_ptr[i] = std.mem.readInt(i32, data[offset..][0..4], .little);
        }
        return count;
    }

    pub fn fragmentReadFloat32(self: *const FragmentReader, col_idx: u32, out_ptr: [*]f32, max_count: usize, start_row: u32) usize {
        if (col_idx >= self.num_columns) return 0;
        const data = self.data orelse return 0;
        const info = &self.columns[col_idx];

        if (start_row >= info.row_count) return 0;
        const count: usize = @intCast(@min(info.row_count - start_row, max_count));

        var i: usize = 0;
        while (i < count) : (i += 1) {
            const offset: usize = @intCast(info.data_offset + (start_row + i) * 4);
            const bits = std.mem.readInt(u32, data[offset..][0..4], .little);
            out_ptr[i] = @bitCast(bits);
        }
        return count;
    }

    pub fn fragmentGetColumnVectorDim(self: *const FragmentReader, col_idx: u32) u32 {
        if (col_idx >= self.num_columns) return 0;
        return self.columns[col_idx].vector_dim;
    }

    pub fn fragmentReadVectorAt(self: *const FragmentReader, col_idx: u32, row_idx: u32, out_ptr: [*]f32, max_floats: usize) usize {
        if (col_idx >= self.num_columns) return 0;
        const data = self.data orelse return 0;
        const info = &self.columns[col_idx];

        if (row_idx >= info.row_count) return 0;
        if (info.vector_dim == 0) return 0;

        const dim = info.vector_dim;
        const copy_count: usize = @min(dim, max_floats);
        const base_offset: usize = @intCast(info.data_offset + row_idx * dim * 4);

        var i: usize = 0;
        while (i < copy_count) : (i += 1) {
            const bits = std.mem.readInt(u32, data[base_offset + i * 4 ..][0..4], .little);
            out_ptr[i] = @bitCast(bits);
        }
        return copy_count;
    }
};

// ============================================================================
// Global Instance (for backward compatibility)
// ============================================================================

var global_reader: FragmentReader = undefined;
var is_initialized = false;

// ============================================================================
// Fragment Loading
// ============================================================================

/// Load a fragment for reading
pub export fn fragmentLoad(data: [*]const u8, len: usize) u32 {
    if (FragmentReader.init(data, len)) |reader| {
        global_reader = reader;
        is_initialized = true;
        return 1;
    } else |_| {
        return 0;
    }
}

fn parseColumnMeta(data: [*]const u8, start: u64, end: u64, info: *ReaderColumnInfo) void {
    info.* = .{
        .name = undefined,
        .name_len = 0,
        .col_type = undefined,
        .type_len = 0,
        .nullable = true,
        .data_offset = 0,
        .row_count = 0,
        .data_size = 0,
        .vector_dim = 0,
    };

    var pos: usize = @intCast(start);
    const end_pos: usize = @intCast(end);
    while (pos < end_pos) {
        const tag = data[pos];
        pos += 1;

        const field_num = tag >> 3;
        const wire_type = tag & 0x7;

        switch (field_num) {
            1 => { // name (string)
                if (wire_type == 2) {
                    const len = readVarintAt(data, &pos);
                    const copy_len = @min(len, 64);
                    @memcpy(info.name[0..copy_len], data[pos..][0..copy_len]);
                    info.name_len = copy_len;
                    pos += len;
                }
            },
            2 => { // type (string)
                if (wire_type == 2) {
                    const len = readVarintAt(data, &pos);
                    const copy_len = @min(len, 16);
                    @memcpy(info.col_type[0..copy_len], data[pos..][0..copy_len]);
                    info.type_len = copy_len;
                    pos += len;
                }
            },
            3 => { // nullable (varint)
                if (wire_type == 0) {
                    info.nullable = readVarintAt(data, &pos) != 0;
                }
            },
            4 => { // data_offset (fixed64)
                if (wire_type == 1) {
                    info.data_offset = std.mem.readInt(u64, data[pos..][0..8], .little);
                    pos += 8;
                }
            },
            5 => { // row_count (varint)
                if (wire_type == 0) {
                    info.row_count = readVarintAt(data, &pos);
                }
            },
            6 => { // data_size (varint)
                if (wire_type == 0) {
                    info.data_size = readVarintAt(data, &pos);
                }
            },
            7 => { // vector_dim (varint)
                if (wire_type == 0) {
                    info.vector_dim = @intCast(readVarintAt(data, &pos));
                }
            },
            else => {
                // Skip unknown field
                if (wire_type == 0) {
                    _ = readVarintAt(data, &pos);
                } else if (wire_type == 1) {
                    pos += 8;
                } else if (wire_type == 2) {
                    const len = readVarintAt(data, &pos);
                    pos += len;
                } else if (wire_type == 5) {
                    pos += 4;
                }
            },
        }
    }
}

fn readVarintAt(data: [*]const u8, pos: *usize) usize {
    var value: usize = 0;
    var shift: u5 = 0;

    while (true) {
        const byte = data[pos.*];
        pos.* += 1;
        value |= @as(usize, byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) break;
        shift += 7;
    }

    return value;
}

/// Parse column metadata from a buffer (used by lazy loading)
fn parseColumnMetaFromBuf(data: [*]const u8, len: usize, info: *ReaderColumnInfo) void {
    info.* = .{
        .name = undefined,
        .name_len = 0,
        .col_type = undefined,
        .type_len = 0,
        .nullable = true,
        .data_offset = 0,
        .row_count = 0,
        .data_size = 0,
        .vector_dim = 0,
        .cached_data = null,
    };

    var pos: usize = 0;
    while (pos < len) {
        const tag = data[pos];
        pos += 1;

        const field_num = tag >> 3;
        const wire_type = tag & 0x7;

        switch (field_num) {
            1 => {
                if (wire_type == 2) {
                    const str_len = readVarintAt(data, &pos);
                    const copy_len = @min(str_len, 64);
                    @memcpy(info.name[0..copy_len], data[pos..][0..copy_len]);
                    info.name_len = copy_len;
                    pos += str_len;
                }
            },
            2 => {
                if (wire_type == 2) {
                    const str_len = readVarintAt(data, &pos);
                    const copy_len = @min(str_len, 16);
                    @memcpy(info.col_type[0..copy_len], data[pos..][0..copy_len]);
                    info.type_len = copy_len;
                    pos += str_len;
                }
            },
            3 => {
                if (wire_type == 0) {
                    info.nullable = readVarintAt(data, &pos) != 0;
                }
            },
            4 => {
                if (wire_type == 1) {
                    info.data_offset = std.mem.readInt(u64, data[pos..][0..8], .little);
                    pos += 8;
                }
            },
            5 => {
                if (wire_type == 0) {
                    info.row_count = readVarintAt(data, &pos);
                }
            },
            6 => {
                if (wire_type == 0) {
                    info.data_size = readVarintAt(data, &pos);
                }
            },
            7 => {
                if (wire_type == 0) {
                    info.vector_dim = @intCast(readVarintAt(data, &pos));
                }
            },
            else => {
                if (wire_type == 0) {
                    _ = readVarintAt(data, &pos);
                } else if (wire_type == 1) {
                    pos += 8;
                } else if (wire_type == 2) {
                    const skip_len = readVarintAt(data, &pos);
                    pos += skip_len;
                } else if (wire_type == 5) {
                    pos += 4;
                }
            },
        }
    }
}

// ============================================================================
// Fragment Metadata Access
// ============================================================================

/// Get number of columns in loaded fragment
pub export fn fragmentGetColumnCount() u32 {
    if (!is_initialized) return 0;
    return global_reader.num_columns;
}

/// Get row count from loaded fragment
pub export fn fragmentGetRowCount() u64 {
    if (!is_initialized or global_reader.num_columns == 0) return 0;
    return global_reader.columns[0].row_count;
}

/// Get column name (returns length, writes to out_ptr)
pub export fn fragmentGetColumnName(col_idx: u32, out_ptr: [*]u8, max_len: usize) usize {
    if (!is_initialized or col_idx >= global_reader.num_columns) return 0;
    const info = &global_reader.columns[col_idx];
    const copy_len = @min(info.name_len, max_len);
    @memcpy(out_ptr[0..copy_len], info.name[0..copy_len]);
    return copy_len;
}

/// Get column type (returns length, writes to out_ptr)
pub export fn fragmentGetColumnType(col_idx: u32, out_ptr: [*]u8, max_len: usize) usize {
    if (!is_initialized or col_idx >= global_reader.num_columns) return 0;
    const info = &global_reader.columns[col_idx];
    const copy_len = @min(info.type_len, max_len);
    @memcpy(out_ptr[0..copy_len], info.col_type[0..copy_len]);
    return copy_len;
}

/// Get column vector dimension (0 if not a vector)
pub export fn fragmentGetColumnVectorDim(col_idx: u32) u32 {
    if (!is_initialized or col_idx >= global_reader.num_columns) return 0;
    return global_reader.columns[col_idx].vector_dim;
}

// ============================================================================
// Column Data Reading
// ============================================================================

/// Read int64 column data
pub export fn fragmentReadInt64(col_idx: u32, out_ptr: [*]i64, max_count: usize) usize {
    if (!is_initialized or col_idx >= global_reader.num_columns) return 0;
    const data = global_reader.data orelse return 0;
    const info = &global_reader.columns[col_idx];

    const count: usize = @intCast(@min(info.row_count, max_count));
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const offset: usize = @intCast(info.data_offset + i * 8);
        out_ptr[i] = std.mem.readInt(i64, data[offset..][0..8], .little);
    }
    return count;
}

/// Read int32 column data
pub export fn fragmentReadInt32(col_idx: u32, out_ptr: [*]i32, max_count: usize) usize {
    if (!is_initialized or col_idx >= global_reader.num_columns) return 0;
    const data = global_reader.data orelse return 0;
    const info = &global_reader.columns[col_idx];

    const count: usize = @intCast(@min(info.row_count, max_count));
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const offset: usize = @intCast(info.data_offset + i * 4);
        out_ptr[i] = std.mem.readInt(i32, data[offset..][0..4], .little);
    }
    return count;
}

/// Read float64 column data
pub export fn fragmentReadFloat64(col_idx: u32, out_ptr: [*]f64, max_count: usize) usize {
    if (!is_initialized or col_idx >= global_reader.num_columns) return 0;
    const data = global_reader.data orelse return 0;
    const info = &global_reader.columns[col_idx];

    const count: usize = @intCast(@min(info.row_count, max_count));
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const offset: usize = @intCast(info.data_offset + i * 8);
        const bits = std.mem.readInt(u64, data[offset..][0..8], .little);
        out_ptr[i] = @bitCast(bits);
    }
    return count;
}

/// Read float32 column data
pub export fn fragmentReadFloat32(col_idx: u32, out_ptr: [*]f32, max_count: usize) usize {
    if (!is_initialized or col_idx >= global_reader.num_columns) return 0;
    const data = global_reader.data orelse return 0;
    const info = &global_reader.columns[col_idx];

    const count: usize = @intCast(@min(info.row_count, max_count));
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const offset: usize = @intCast(info.data_offset + i * 4);
        const bits = std.mem.readInt(u32, data[offset..][0..4], .little);
        out_ptr[i] = @bitCast(bits);
    }
    return count;
}

/// Read bool column data (unpacked from bits)
pub export fn fragmentReadBool(col_idx: u32, out_ptr: [*]u8, max_count: usize) usize {
    if (!is_initialized or col_idx >= global_reader.num_columns) return 0;
    const data = global_reader.data orelse return 0;
    const info = &global_reader.columns[col_idx];

    const count: usize = @intCast(@min(info.row_count, max_count));
    const base_offset: usize = @intCast(info.data_offset);
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const byte_idx = i / 8;
        const bit_idx: u3 = @intCast(i % 8);
        const byte = data[base_offset + byte_idx];
        out_ptr[i] = if ((byte & (@as(u8, 1) << bit_idx)) != 0) 1 else 0;
    }
    return count;
}

// ============================================================================
// String Column Reading
// ============================================================================

/// Get string at index - returns length, writes to out_ptr
pub export fn fragmentReadStringAt(col_idx: u32, row_idx: u32, out_ptr: [*]u8, max_len: usize) usize {
    if (!is_initialized or col_idx >= global_reader.num_columns) return 0;
    const data = global_reader.data orelse return 0;
    const info = &global_reader.columns[col_idx];

    if (row_idx >= info.row_count) return 0;

    // String layout: [string_data][offsets]
    // offsets are at the end: (row_count + 1) * 4 bytes
    const offsets_size: usize = @intCast((info.row_count + 1) * 4);
    const offsets_start: usize = @intCast(info.data_offset + info.data_size - offsets_size);
    const data_start: usize = @intCast(info.data_offset);

    const start_offset = std.mem.readInt(u32, data[offsets_start + row_idx * 4 ..][0..4], .little);
    const end_offset = std.mem.readInt(u32, data[offsets_start + (row_idx + 1) * 4 ..][0..4], .little);

    const str_len = end_offset - start_offset;
    const copy_len = @min(str_len, max_len);

    @memcpy(out_ptr[0..copy_len], data[data_start + start_offset ..][0..copy_len]);
    return copy_len;
}

/// Get string length at index (useful for allocation)
pub export fn fragmentGetStringLength(col_idx: u32, row_idx: u32) usize {
    if (!is_initialized or col_idx >= global_reader.num_columns) return 0;
    const data = global_reader.data orelse return 0;
    const info = &global_reader.columns[col_idx];

    if (row_idx >= info.row_count) return 0;

    const offsets_size: usize = @intCast((info.row_count + 1) * 4);
    const offsets_start: usize = @intCast(info.data_offset + info.data_size - offsets_size);

    const start_offset = std.mem.readInt(u32, data[offsets_start + row_idx * 4 ..][0..4], .little);
    const end_offset = std.mem.readInt(u32, data[offsets_start + (row_idx + 1) * 4 ..][0..4], .little);

    return end_offset - start_offset;
}

// ============================================================================
// Vector Column Reading
// ============================================================================

/// Read vector at index - returns number of floats written
pub export fn fragmentReadVectorAt(col_idx: u32, row_idx: u32, out_ptr: [*]f32, max_floats: usize) usize {
    if (!is_initialized or col_idx >= global_reader.num_columns) return 0;
    const data = global_reader.data orelse return 0;
    const info = &global_reader.columns[col_idx];

    if (row_idx >= info.row_count) return 0;
    if (info.vector_dim == 0) return 0;

    const dim = info.vector_dim;
    const copy_count: usize = @min(dim, max_floats);
    const base_offset: usize = @intCast(info.data_offset + row_idx * dim * 4);

    var i: usize = 0;
    while (i < copy_count) : (i += 1) {
        const bits = std.mem.readInt(u32, data[base_offset + i * 4 ..][0..4], .little);
        out_ptr[i] = @bitCast(bits);
    }
    return copy_count;
}

// ============================================================================
// Tests
// ============================================================================

test "fragment_reader: readVarintAt" {
    // Test single-byte varint (value < 128)
    var data = [_]u8{42};
    var pos: usize = 0;
    const result = readVarintAt(&data, &pos);
    try std.testing.expectEqual(@as(usize, 42), result);
    try std.testing.expectEqual(@as(usize, 1), pos);
}

test "fragment_reader: readVarintAt multi-byte" {
    // Test multi-byte varint (value = 300 = 0x12C)
    // Encoded as: 0xAC 0x02 (128 + 44, 2)
    var data = [_]u8{ 0xAC, 0x02 };
    var pos: usize = 0;
    const result = readVarintAt(&data, &pos);
    try std.testing.expectEqual(@as(usize, 300), result);
    try std.testing.expectEqual(@as(usize, 2), pos);
}
