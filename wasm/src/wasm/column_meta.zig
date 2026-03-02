//! Column Metadata Parsing
//!
//! Protobuf parsing utilities for Lance column metadata.

const format = @import("format.zig");

const readU64LE = format.readU64LE;
const readI64LE = format.readI64LE;
const readF64LE = format.readF64LE;

/// Column offset entry from metadata offsets table
pub const OffsetEntry = struct { pos: u64, len: u64 };

/// Page buffer info extracted from protobuf metadata
pub const PageBufferInfo = struct { offset: u64, size: u64, rows: u64 };

/// Get column offset entry from file data
pub fn getColumnOffsetEntry(
    file_data: []const u8,
    num_columns: u32,
    column_meta_offsets_start: u64,
    col_idx: u32,
) OffsetEntry {
    if (col_idx >= num_columns) return .{ .pos = 0, .len = 0 };

    const entry_offset: usize = @intCast(column_meta_offsets_start + col_idx * 16);
    if (entry_offset + 16 > file_data.len) return .{ .pos = 0, .len = 0 };

    return .{
        .pos = readU64LE(file_data, entry_offset),
        .len = readU64LE(file_data, entry_offset + 8),
    };
}

/// Parse varint from protobuf data
pub fn readVarint(data: []const u8, offset: *usize) u64 {
    var result: u64 = 0;
    var shift: u6 = 0;

    while (offset.* < data.len) {
        const byte = data[offset.*];
        offset.* += 1;
        result |= @as(u64, byte & 0x7F) << shift;
        if (byte & 0x80 == 0) break;
        shift +|= 7;
    }
    return result;
}

/// Get page buffer info from column metadata protobuf
pub fn getPageBufferInfo(col_meta: []const u8) PageBufferInfo {
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
                // Only read first page
                return .{ .offset = page_offset, .size = page_size, .rows = page_rows };
            },
            else => {
                // Skip field
                if (wire_type == 0) {
                    _ = readVarint(col_meta, &pos);
                } else if (wire_type == 2) {
                    const skip_len = readVarint(col_meta, &pos);
                    pos += @as(usize, @intCast(skip_len));
                } else if (wire_type == 5) {
                    pos += 4; // 32-bit fixed
                } else if (wire_type == 1) {
                    pos += 8; // 64-bit fixed
                }
            },
        }
    }

    return .{ .offset = page_offset, .size = page_size, .rows = page_rows };
}
