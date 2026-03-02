//! Lance File Writer for WASM
//!
//! Provides low-level and high-level APIs for writing Lance format files.
//! The low-level API (writer*) gives direct control over byte writing.
//! The high-level API (fragment*) manages columns, metadata, and footer automatically.

const std = @import("std");
const memory = @import("memory.zig");

const wasmAlloc = memory.wasmAlloc;

// Import js_log for debugging
extern fn js_log(ptr: [*]const u8, len: usize) void;

// ============================================================================
// Constants
// ============================================================================

const MAX_COLUMNS = 64;

// ============================================================================
// Column Types and Info
// ============================================================================

pub const ColumnType = enum(u8) {
    int64 = 0,
    int32 = 1,
    float64 = 2,
    float32 = 3,
    string = 4,
    bool = 5,
    vector = 6,
    uint8 = 7,
    list = 8,
};

const ColumnInfo = struct {
    name_buf: [64]u8,
    name_len: usize,
    col_type: ColumnType,
    data_offset: usize,
    data_size: usize,
    row_count: usize,
    vector_dim: u32, // Only for vector type
    nullable: bool,
};

// ============================================================================
// Writer State
// ============================================================================

// Static fallback buffer (256KB) for when dynamic allocation fails
// Must be 8-byte aligned for BigInt64Array compatibility in JavaScript
var static_writer_buffer: [256 * 1024]u8 align(8) = undefined;
var use_static_buffer: bool = false;

// Global temp ColumnInfo to avoid stack allocation in WASM
var global_temp_col: ColumnInfo = undefined;

var writer_buffer: ?[*]u8 = null;
var writer_buffer_len: usize = 0;
var writer_offset: usize = 0;

var fragment_columns: [MAX_COLUMNS]ColumnInfo = undefined;
var fragment_column_count: usize = 0;
var fragment_row_count: usize = 0;
var global_meta_offsets: [MAX_COLUMNS]usize = undefined;

// ============================================================================
// Low-Level Writer API
// ============================================================================

/// Initialize a new Lance file writer with capacity
pub export fn writerInit(capacity: usize) u32 {
    js_log("writerInit: enter", 17);

    // First, try to use static buffer if it fits (more reliable in WASM)
    if (capacity <= static_writer_buffer.len) {
        js_log("writerInit: using static buffer", 31);
        writer_buffer = &static_writer_buffer;
        writer_buffer_len = static_writer_buffer.len;
        writer_offset = 0;
        use_static_buffer = true;
        js_log("writerInit: static buffer ok", 28);
        return 1;
    }

    // Reuse existing buffer if large enough
    js_log("writerInit: check existing", 26);
    if (writer_buffer != null and writer_buffer_len >= capacity and !use_static_buffer) {
        js_log("writerInit: reuse buffer", 24);
        writer_offset = 0;
        return 1;
    }
    js_log("writerInit: no reuse", 20);

    // Free previous buffer (only if dynamically allocated)
    js_log("writerInit: check free", 22);
    if (writer_buffer != null and !use_static_buffer) {
        js_log("writerInit: freeing old buffer", 30);
        memory.wasmFree(writer_buffer.?, writer_buffer_len);
        writer_buffer = null;
        js_log("writerInit: freed", 17);
    }
    js_log("writerInit: about to alloc", 26);

    // Allocate 8-byte aligned memory (via memory wrapper)
    const slice_ptr = memory.wasmAlloc(capacity) orelse {
        js_log("writerInit: alloc failed, fallback", 35);
        // Fallback to static buffer even if too small
        writer_buffer = &static_writer_buffer;
        writer_buffer_len = static_writer_buffer.len;
        writer_offset = 0;
        use_static_buffer = true;
        return 1;
    };
    js_log("writerInit: alloc success", 25);
    writer_buffer = slice_ptr;
    writer_buffer_len = capacity;
    writer_offset = 0;
    use_static_buffer = false;
    js_log("writerInit: done", 16);
    return 1;
}

/// Get pointer to writer buffer for JS to write column data directly
pub export fn writerGetBuffer() ?[*]u8 {
    return writer_buffer;
}

/// Get current write offset
pub export fn writerGetOffset() usize {
    return writer_offset;
}

/// Write int64 values to buffer
pub export fn writerWriteInt64(values: [*]const i64, count: usize) u32 {
    js_log("writerWriteInt64: enter", 23);
    const buf = writer_buffer orelse {
        js_log("writerWriteInt64: no buffer", 27);
        return 0;
    };
    js_log("writerWriteInt64: got buffer", 28);
    const bytes_needed = count * 8;
    js_log("writerWriteInt64: checking space", 32);
    if (writer_offset + bytes_needed > writer_buffer_len) {
        js_log("writerWriteInt64: no space", 26);
        return 0;
    }
    js_log("writerWriteInt64: space ok", 26);

    js_log("writerWriteInt64: starting loop", 31);
    // Write using pointer cast for aligned writes
    const dst: [*]align(1) i64 = @ptrCast(&buf[writer_offset]);
    var i: usize = 0;
    while (i < count) : (i += 1) {
        dst[i] = values[i];
    }
    writer_offset += bytes_needed;
    js_log("writerWriteInt64: done", 22);
    return 1;
}

/// Write int32 values to buffer
pub export fn writerWriteInt32(values: [*]const i32, count: usize) u32 {
    const buf = writer_buffer orelse return 0;
    const bytes_needed = count * 4;
    if (writer_offset + bytes_needed > writer_buffer_len) return 0;

    var i: usize = 0;
    while (i < count) : (i += 1) {
        std.mem.writeInt(i32, buf[writer_offset..][0..4], values[i], .little);
        writer_offset += 4;
    }
    return 1;
}

/// Write float64 values to buffer
pub export fn writerWriteFloat64(values: [*]const f64, count: usize) u32 {
    const buf = writer_buffer orelse return 0;
    const bytes_needed = count * 8;
    if (writer_offset + bytes_needed > writer_buffer_len) return 0;

    var i: usize = 0;
    while (i < count) : (i += 1) {
        const bits: u64 = @bitCast(values[i]);
        std.mem.writeInt(u64, buf[writer_offset..][0..8], bits, .little);
        writer_offset += 8;
    }
    return 1;
}

/// Write float32 values to buffer
pub export fn writerWriteFloat32(values: [*]const f32, count: usize) u32 {
    const buf = writer_buffer orelse return 0;
    const bytes_needed = count * 4;
    if (writer_offset + bytes_needed > writer_buffer_len) return 0;

    var i: usize = 0;
    while (i < count) : (i += 1) {
        const bits: u32 = @bitCast(values[i]);
        std.mem.writeInt(u32, buf[writer_offset..][0..4], bits, .little);
        writer_offset += 4;
    }
    return 1;
}

/// Write raw bytes to buffer (for strings, vectors, etc)
pub export fn writerWriteBytes(data: [*]const u8, len: usize) u32 {
    const buf = writer_buffer orelse return 0;
    if (writer_offset + len > writer_buffer_len) return 0;

    // Manual copy to avoid @memcpy issues in WASM
    var i: usize = 0;
    while (i < len) : (i += 1) {
        buf[writer_offset + i] = data[i];
    }
    writer_offset += len;
    return 1;
}

/// Write u32 offset value (for string offsets)
pub export fn writerWriteOffset32(value: u32) u32 {
    const buf = writer_buffer orelse return 0;
    if (writer_offset + 4 > writer_buffer_len) return 0;

    std.mem.writeInt(u32, buf[writer_offset..][0..4], value, .little);
    writer_offset += 4;
    return 1;
}

/// Write u64 offset value
pub export fn writerWriteOffset64(value: u64) u32 {
    const buf = writer_buffer orelse return 0;
    if (writer_offset + 8 > writer_buffer_len) return 0;

    std.mem.writeInt(u64, buf[writer_offset..][0..8], value, .little);
    writer_offset += 8;
    return 1;
}

/// Write Lance footer (40 bytes)
pub export fn writerWriteFooter(
    column_meta_start: u64,
    column_meta_offsets_start: u64,
    global_buff_offsets_start: u64,
    num_global_buffers: u32,
    num_cols: u32,
    major_version: u16,
    minor_version: u16,
) u32 {
    const buf = writer_buffer orelse return 0;
    if (writer_offset + 40 > writer_buffer_len) return 0;

    std.mem.writeInt(u64, buf[writer_offset..][0..8], column_meta_start, .little);
    writer_offset += 8;
    std.mem.writeInt(u64, buf[writer_offset..][0..8], column_meta_offsets_start, .little);
    writer_offset += 8;
    std.mem.writeInt(u64, buf[writer_offset..][0..8], global_buff_offsets_start, .little);
    writer_offset += 8;
    std.mem.writeInt(u32, buf[writer_offset..][0..4], num_global_buffers, .little);
    writer_offset += 4;
    std.mem.writeInt(u32, buf[writer_offset..][0..4], num_cols, .little);
    writer_offset += 4;
    std.mem.writeInt(u16, buf[writer_offset..][0..2], major_version, .little);
    writer_offset += 2;
    std.mem.writeInt(u16, buf[writer_offset..][0..2], minor_version, .little);
    writer_offset += 2;
    @memcpy(buf[writer_offset..][0..4], "LANC");
    writer_offset += 4;

    return 1;
}

/// Write protobuf varint
pub export fn writerWriteVarint(value: u64) u32 {
    const buf = writer_buffer orelse return 0;
    var v = value;

    while (v >= 0x80) {
        if (writer_offset >= writer_buffer_len) return 0;
        buf[writer_offset] = @as(u8, @truncate(v)) | 0x80;
        writer_offset += 1;
        v >>= 7;
    }

    if (writer_offset >= writer_buffer_len) return 0;
    buf[writer_offset] = @truncate(v);
    writer_offset += 1;

    return 1;
}

/// Finalize and return the final file size
pub export fn writerFinalize() usize {
    return writer_offset;
}

/// Reset writer for next file
pub export fn writerReset() void {
    writer_offset = 0;
}

// ============================================================================
// High-Level Fragment Writer API
// ============================================================================
// Manages column schema, data offsets, and metadata writing automatically.
// JS just needs to: fragmentBegin -> fragmentAdd*Column (for each) -> fragmentEnd

/// Begin a new fragment (resets state)
pub export fn fragmentBegin(capacity: usize) u32 {
    if (writerInit(capacity) == 0) return 0;
    fragment_column_count = 0;
    fragment_row_count = 0;
    return 1;
}

/// Add a column with int64 data
pub export fn fragmentAddInt64Column(
    name_ptr: [*]const u8,
    name_len: usize,
    values: [*]const i64,
    count: usize,
    nullable: bool,
) u32 {
    js_log("fragmentAddInt64Column: enter", 29);
    if (fragment_column_count >= MAX_COLUMNS) {
        js_log("fragmentAddInt64Column: too many", 32);
        return 0;
    }
    js_log("fragmentAddInt64Column: col count ok", 36);

    // Align to 8 bytes for Int64
    js_log("fragmentAddInt64Column: pre-align", 33);
    const pre_offset = writer_offset;
    _ = pre_offset;
    const padding = (8 - (writer_offset % 8)) % 8;
    writer_offset += padding;
    js_log("fragmentAddInt64Column: aligned", 31);

    const data_offset = writer_offset;
    // Verify alignment
    if (data_offset % 8 != 0) {
        js_log("fragmentAddInt64Column: MISALIGN!", 36);
    }
    js_log("fragmentAddInt64Column: calling write", 37);
    if (writerWriteInt64(values, count) == 0) {
        js_log("fragmentAddInt64Column: write fail", 34);
        return 0;
    }
    js_log("fragmentAddInt64Column: write ok", 32);
    const data_size = writer_offset - data_offset;

    js_log("fragmentAddInt64Column: setting fields", 38);
    // Write directly to fragment_columns to avoid stack allocation
    fragment_columns[fragment_column_count].name_len = name_len;
    fragment_columns[fragment_column_count].col_type = .int64;
    fragment_columns[fragment_column_count].data_offset = data_offset;
    fragment_columns[fragment_column_count].data_size = data_size;
    fragment_columns[fragment_column_count].row_count = count;
    fragment_columns[fragment_column_count].vector_dim = 0;
    fragment_columns[fragment_column_count].nullable = nullable;
    js_log("fragmentAddInt64Column: fields set", 34);

    const len = if (name_len > 64) 64 else name_len;
    js_log("fragmentAddInt64Column: copying name", 36);

    // Manual copy to avoid @memcpy issues
    var ci: usize = 0;
    while (ci < len) : (ci += 1) {
        fragment_columns[fragment_column_count].name_buf[ci] = name_ptr[ci];
    }
    js_log("fragmentAddInt64Column: name copied", 35);

    fragment_column_count += 1;
    if (count > fragment_row_count) fragment_row_count = count;
    js_log("fragmentAddInt64Column: done", 28);

    return 1;
}

/// Add a column with int32 data
pub export fn fragmentAddInt32Column(
    name_ptr: [*]const u8,
    name_len: usize,
    values: [*]const i32,
    count: usize,
    nullable: bool,
) u32 {
    if (fragment_column_count >= MAX_COLUMNS) return 0;

    const data_offset = writer_offset;
    if (writerWriteInt32(values, count) == 0) return 0;
    const data_size = writer_offset - data_offset;

    var col = ColumnInfo{
        .name_buf = undefined,
        .name_len = name_len,
        .col_type = .int32,
        .data_offset = data_offset,
        .data_size = data_size,
        .row_count = count,
        .vector_dim = 0,
        .nullable = nullable,
    };
    const len = if (name_len > 64) 64 else name_len;
    @memcpy(col.name_buf[0..len], name_ptr[0..len]);
    fragment_columns[fragment_column_count] = col;
    fragment_column_count += 1;
    if (count > fragment_row_count) fragment_row_count = count;

    return 1;
}

/// Add a column with float64 data
pub export fn fragmentAddFloat64Column(
    name_ptr: [*]const u8,
    name_len: usize,
    values: [*]const f64,
    count: usize,
    nullable: bool,
) u32 {
    if (fragment_column_count >= MAX_COLUMNS) return 0;

    // Align to 8 bytes for Float64
    const padding = (8 - (writer_offset % 8)) % 8;
    writer_offset += padding;

    const data_offset = writer_offset;
    if (writerWriteFloat64(values, count) == 0) return 0;
    const data_size = writer_offset - data_offset;

    var col = ColumnInfo{
        .name_buf = undefined,
        .name_len = name_len,
        .col_type = .float64,
        .data_offset = data_offset,
        .data_size = data_size,
        .row_count = count,
        .vector_dim = 0,
        .nullable = nullable,
    };
    const len = if (name_len > 64) 64 else name_len;
    @memcpy(col.name_buf[0..len], name_ptr[0..len]);
    fragment_columns[fragment_column_count] = col;
    fragment_column_count += 1;
    if (count > fragment_row_count) fragment_row_count = count;

    return 1;
}

/// Add a column with float32 data
pub export fn fragmentAddFloat32Column(
    name_ptr: [*]const u8,
    name_len: usize,
    values: [*]const f32,
    count: usize,
    nullable: bool,
) u32 {
    if (fragment_column_count >= MAX_COLUMNS) return 0;

    const data_offset = writer_offset;
    if (writerWriteFloat32(values, count) == 0) return 0;
    const data_size = writer_offset - data_offset;

    var col = ColumnInfo{
        .name_buf = undefined,
        .name_len = name_len,
        .col_type = .float32,
        .data_offset = data_offset,
        .data_size = data_size,
        .row_count = count,
        .vector_dim = 0,
        .nullable = nullable,
    };
    const len = if (name_len > 64) 64 else name_len;
    @memcpy(col.name_buf[0..len], name_ptr[0..len]);
    fragment_columns[fragment_column_count] = col;
    fragment_column_count += 1;
    if (count > fragment_row_count) fragment_row_count = count;

    return 1;
}

/// Add a column with string data (data followed by offsets)
/// string_data: concatenated UTF-8 bytes
/// offsets: uint32 array of length count+1 (start positions + final end)
pub export fn fragmentAddStringColumn(
    name_ptr: [*]const u8,
    name_len: usize,
    string_data: [*]const u8,
    string_data_len: usize,
    offsets: [*]const u32,
    count: usize,
    nullable: bool,
) u32 {
    js_log("fragmentAddStringColumn: enter", 30);
    if (fragment_column_count >= MAX_COLUMNS) return 0;
    js_log("fragmentAddStringColumn: col count ok", 37);

    const data_offset = writer_offset;

    // Write string data
    js_log("fragmentAddStringColumn: writing str data", 41);
    if (writerWriteBytes(string_data, string_data_len) == 0) return 0;
    js_log("fragmentAddStringColumn: str data written", 41);

    // Pad to 4-byte alignment for offsets
    js_log("fragmentAddStringColumn: getting buf", 36);
    const buf = writer_buffer orelse return 0;
    js_log("fragmentAddStringColumn: got buf", 32);
    const padding = (4 - (writer_offset % 4)) % 4;
    var p: usize = 0;
    while (p < padding) : (p += 1) {
        if (writer_offset >= writer_buffer_len) return 0;
        buf[writer_offset] = 0;
        writer_offset += 1;
    }
    js_log("fragmentAddStringColumn: padded", 31);

    const offsets_start_in_file = writer_offset; // record this if needed, but we use data_size
    _ = offsets_start_in_file;

    // Write offsets (count + 1 values)
    const offsets_bytes = (count + 1) * 4;
    if (writer_offset + offsets_bytes > writer_buffer_len) return 0;
    js_log("fragmentAddStringColumn: writing offsets", 40);

    var i: usize = 0;
    while (i <= count) : (i += 1) {
        // Manual little-endian write to avoid std.mem.writeInt issues in WASM
        const val = offsets[i];
        buf[writer_offset] = @truncate(val);
        buf[writer_offset + 1] = @truncate(val >> 8);
        buf[writer_offset + 2] = @truncate(val >> 16);
        buf[writer_offset + 3] = @truncate(val >> 24);
        writer_offset += 4;
    }
    js_log("fragmentAddStringColumn: offsets written", 40);

    const data_size = writer_offset - data_offset;
    js_log("fragmentAddStringColumn: setting fields", 39);

    // Use global temp col to avoid stack allocation in WASM
    global_temp_col.name_len = name_len;
    global_temp_col.col_type = .string;
    global_temp_col.data_offset = data_offset;
    global_temp_col.data_size = data_size;
    global_temp_col.row_count = count;
    global_temp_col.vector_dim = 0;
    global_temp_col.nullable = nullable;
    js_log("fragmentAddStringColumn: fields set", 35);
    const len = if (name_len > 64) 64 else name_len;
    // Manual copy to avoid @memcpy issues in WASM
    js_log("fragmentAddStringColumn: copying name", 37);
    for (0..len) |j| {
        global_temp_col.name_buf[j] = name_ptr[j];
    }
    js_log("fragmentAddStringColumn: name copied", 36);
    fragment_columns[fragment_column_count] = global_temp_col;
    js_log("fragmentAddStringColumn: col stored", 35);
    fragment_column_count += 1;
    if (count > fragment_row_count) fragment_row_count = count;

    return 1;
}

/// Add a column with list data (JSON-encoded array in string format + offsets)
pub export fn fragmentAddListColumn(
    name_ptr: [*]const u8,
    name_len: usize,
    string_data: [*]const u8,
    string_data_len: usize,
    offsets: [*]const u32,
    count: usize,
    nullable: bool,
) u32 {
    if (fragment_column_count >= MAX_COLUMNS) return 0;

    const data_offset = writer_offset;

    // Write string data
    if (writerWriteBytes(string_data, string_data_len) == 0) return 0;

    // Pad to 4-byte alignment for offsets
    const buf = writer_buffer orelse return 0;
    const padding = (4 - (writer_offset % 4)) % 4;
    var p: usize = 0;
    while (p < padding) : (p += 1) {
        if (writer_offset >= writer_buffer_len) return 0;
        buf[writer_offset] = 0;
        writer_offset += 1;
    }

    // Write offsets (count + 1 values)
    const offsets_bytes = (count + 1) * 4;
    if (writer_offset + offsets_bytes > writer_buffer_len) return 0;

    var i: usize = 0;
    while (i <= count) : (i += 1) {
        std.mem.writeInt(u32, buf[writer_offset..][0..4], offsets[i], .little);
        writer_offset += 4;
    }

    const data_size = writer_offset - data_offset;

    var col = ColumnInfo{
        .name_buf = undefined,
        .name_len = name_len,
        .col_type = .list,
        .data_offset = data_offset,
        .data_size = data_size,
        .row_count = count,
        .vector_dim = 0,
        .nullable = nullable,
    };
    const len = if (name_len > 64) 64 else name_len;
    @memcpy(col.name_buf[0..len], name_ptr[0..len]);
    fragment_columns[fragment_column_count] = col;
    fragment_column_count += 1;
    if (count > fragment_row_count) fragment_row_count = count;

    return 1;
}

/// Add a column with boolean data (bit-packed)
pub export fn fragmentAddBoolColumn(
    name_ptr: [*]const u8,
    name_len: usize,
    packed_bits: [*]const u8,
    byte_count: usize,
    row_count: usize,
    nullable: bool,
) u32 {
    if (fragment_column_count >= MAX_COLUMNS) return 0;

    const data_offset = writer_offset;
    if (writerWriteBytes(packed_bits, byte_count) == 0) return 0;
    const data_size = writer_offset - data_offset;

    var col = ColumnInfo{
        .name_buf = undefined,
        .name_len = name_len,
        .col_type = .bool,
        .data_offset = data_offset,
        .data_size = data_size,
        .row_count = row_count,
        .vector_dim = 0,
        .nullable = nullable,
    };
    const len = if (name_len > 64) 64 else name_len;
    @memcpy(col.name_buf[0..len], name_ptr[0..len]);
    fragment_columns[fragment_column_count] = col;
    fragment_column_count += 1;
    if (row_count > fragment_row_count) fragment_row_count = row_count;

    return 1;
}

/// Add a column with vector data (float32 arrays, flattened)
pub export fn fragmentAddVectorColumn(
    name_ptr: [*]const u8,
    name_len: usize,
    values: [*]const f32,
    total_floats: usize,
    vector_dim: u32,
    nullable: bool,
) u32 {
    if (fragment_column_count >= MAX_COLUMNS) return 0;
    if (vector_dim == 0) return 0;

    // Align to 4 bytes for Vector (Float32)
    const padding = (4 - (writer_offset % 4)) % 4;
    writer_offset += padding;

    const data_offset = writer_offset;
    if (writerWriteFloat32(values, total_floats) == 0) return 0;
    const data_size = writer_offset - data_offset;

    const row_count = total_floats / vector_dim;

    var col = ColumnInfo{
        .name_buf = undefined,
        .name_len = name_len,
        .col_type = .vector,
        .data_offset = data_offset,
        .data_size = data_size,
        .row_count = row_count,
        .vector_dim = vector_dim,
        .nullable = nullable,
    };
    const len = if (name_len > 64) 64 else name_len;
    @memcpy(col.name_buf[0..len], name_ptr[0..len]);
    fragment_columns[fragment_column_count] = col;
    fragment_column_count += 1;
    if (row_count > fragment_row_count) fragment_row_count = row_count;

    return 1;
}

/// Helper to write column metadata in protobuf format
fn writeColumnMetadata(col: *const ColumnInfo) void {
    js_log("writeColumnMetadata: enter", 26);
    const buf = writer_buffer orelse return;

    // Field 1: name (string) - tag = (1 << 3) | 2 = 10
    js_log("writeColumnMetadata: field1 name", 32);
    buf[writer_offset] = 10;
    writer_offset += 1;
    writeVarintInternal(col.name_len);
    js_log("writeColumnMetadata: copy name", 30);
    // Manual copy to avoid @memcpy crashes in WASM
    var i: usize = 0;
    while (i < col.name_len) : (i += 1) {
        buf[writer_offset + i] = col.name_buf[i];
    }
    writer_offset += col.name_len;

    // Field 2: type (string) - tag = (2 << 3) | 2 = 18
    js_log("writeColumnMetadata: field2 type", 32);
    const type_str = switch (col.col_type) {
        .int64 => "int64",
        .int32 => "int32",
        .float64 => "float64",
        .float32 => "float32",
        .string => "string",
        .bool => "bool",
        .vector => "vector",
        .uint8 => "uint8",
        .list => "list",
    };
    buf[writer_offset] = 18;
    writer_offset += 1;
    writeVarintInternal(type_str.len);
    js_log("writeColumnMetadata: copy type", 30);
    // Manual copy
    var j: usize = 0;
    while (j < type_str.len) : (j += 1) {
        buf[writer_offset + j] = type_str[j];
    }
    writer_offset += type_str.len;

    // Field 3: nullable (varint) - tag = (3 << 3) | 0 = 24
    js_log("writeColumnMetadata: field3 null", 32);
    buf[writer_offset] = 24;
    writer_offset += 1;
    buf[writer_offset] = if (col.nullable) 1 else 0;
    writer_offset += 1;

    // Field 4: data_offset (fixed64) - tag = (4 << 3) | 1 = 33
    js_log("writeColumnMetadata: field4 offset", 34);
    buf[writer_offset] = 33;
    writer_offset += 1;
    std.mem.writeInt(u64, buf[writer_offset..][0..8], col.data_offset, .little);
    writer_offset += 8;

    // Field 5: row_count (varint) - tag = (5 << 3) | 0 = 40
    js_log("writeColumnMetadata: field5 rows", 32);
    buf[writer_offset] = 40;
    writer_offset += 1;
    writeVarintInternal(col.row_count);

    // Field 6: data_size (varint) - tag = (6 << 3) | 0 = 48
    js_log("writeColumnMetadata: field6 size", 32);
    buf[writer_offset] = 48;
    writer_offset += 1;
    writeVarintInternal(col.data_size);

    // Field 7: vector_dim (varint) - tag = (7 << 3) | 0 = 56, only if vector
    if (col.col_type == .vector and col.vector_dim > 0) {
        js_log("writeColumnMetadata: field7 dim", 31);
        buf[writer_offset] = 56;
        writer_offset += 1;
        writeVarintInternal(col.vector_dim);
    }
    js_log("writeColumnMetadata: done", 25);
}

fn writeVarintInternal(value: usize) void {
    const buf = writer_buffer orelse return;
    var v = value;

    while (v >= 0x80) {
        buf[writer_offset] = @as(u8, @truncate(v)) | 0x80;
        writer_offset += 1;
        v >>= 7;
    }
    buf[writer_offset] = @truncate(v);
    writer_offset += 1;
}

/// Finish the fragment - writes metadata, offsets table, and footer
/// Returns final file size, or 0 on error
pub export fn fragmentEnd() usize {
    js_log("fragmentEnd: enter", 18);
    if (fragment_column_count == 0) {
        js_log("fragmentEnd: no columns", 23);
        return 0;
    }
    js_log("fragmentEnd: has columns", 24);
    const buf = writer_buffer orelse {
        js_log("fragmentEnd: no buffer", 22);
        return 0;
    };
    js_log("fragmentEnd: has buffer", 23);

    // Record where column metadata starts
    const col_meta_start = writer_offset;
    js_log("fragmentEnd: col_meta_start set", 31);

    // Use global meta_offsets to avoid stack overflow in WASM
    js_log("fragmentEnd: starting col meta loop", 35);

    // Write column metadata
    for (0..fragment_column_count) |i| {
        js_log("fragmentEnd: col meta iter", 26);
        js_log("fragmentEnd: assigning offset", 29);
        global_meta_offsets[i] = writer_offset;
        js_log("fragmentEnd: offset assigned", 28);
        js_log("fragmentEnd: getting col ptr", 28);
        const col_ptr = &fragment_columns[i];
        js_log("fragmentEnd: got col ptr", 24);
        writeColumnMetadata(col_ptr);
        js_log("fragmentEnd: col meta written", 29);
    }
    js_log("fragmentEnd: all col meta done", 30);

    // Record where offsets table starts
    const col_meta_offsets_start = writer_offset;
    js_log("fragmentEnd: writing offsets table", 34);

    // Write metadata offsets table (uint64 per column)
    for (0..fragment_column_count) |i| {
        std.mem.writeInt(u64, buf[writer_offset..][0..8], global_meta_offsets[i], .little);
        writer_offset += 8;
    }
    js_log("fragmentEnd: offsets written", 28);

    // Global buffer offsets (none for now)
    const global_buff_offsets_start = writer_offset;
    js_log("fragmentEnd: writing footer", 27);

    // Write footer (40 bytes)
    std.mem.writeInt(u64, buf[writer_offset..][0..8], col_meta_start, .little);
    writer_offset += 8;
    std.mem.writeInt(u64, buf[writer_offset..][0..8], col_meta_offsets_start, .little);
    writer_offset += 8;
    std.mem.writeInt(u64, buf[writer_offset..][0..8], global_buff_offsets_start, .little);
    writer_offset += 8;
    std.mem.writeInt(u32, buf[writer_offset..][0..4], 0, .little); // num_global_buffers
    writer_offset += 4;
    std.mem.writeInt(u32, buf[writer_offset..][0..4], @intCast(fragment_column_count), .little);
    writer_offset += 4;
    std.mem.writeInt(u16, buf[writer_offset..][0..2], 0, .little); // major version (Lance 2.0)
    writer_offset += 2;
    std.mem.writeInt(u16, buf[writer_offset..][0..2], 3, .little); // minor version
    writer_offset += 2;
    js_log("fragmentEnd: writing magic", 26);
    @memcpy(buf[writer_offset..][0..4], "LANC");
    writer_offset += 4;
    js_log("fragmentEnd: done", 17);

    return writer_offset;
}

// ============================================================================
// Tests
// ============================================================================

test "lance_writer: writerInit" {
    const result = writerInit(1024);
    try std.testing.expectEqual(@as(u32, 1), result);
    try std.testing.expect(writer_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), writer_offset);
}

test "lance_writer: writerWriteInt64" {
    _ = writerInit(1024);
    const values = [_]i64{ 1, 2, 3 };
    const result = writerWriteInt64(&values, 3);
    try std.testing.expectEqual(@as(u32, 1), result);
    try std.testing.expectEqual(@as(usize, 24), writer_offset);
}

test "lance_writer: fragmentBegin" {
    const result = fragmentBegin(1024);
    try std.testing.expectEqual(@as(u32, 1), result);
    try std.testing.expectEqual(@as(usize, 0), fragment_column_count);
}
